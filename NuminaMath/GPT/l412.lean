import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Exponential
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Inverse
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Binomial
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Probability.Process.Common
import RealTypes

namespace boys_in_class_l412_412240

-- Define the conditions given in the problem
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 4 * (boys + girls) / 7 ∧ girls = 3 * (boys + girls) / 7
def total_students (boys girls : ℕ) : Prop := boys + girls = 49

-- Define the statement to be proved
theorem boys_in_class (boys girls : ℕ) (h1 : ratio_boys_to_girls boys girls) (h2 : total_students boys girls) : boys = 28 :=
by
  sorry

end boys_in_class_l412_412240


namespace number_of_intersection_points_l412_412473

theorem number_of_intersection_points :
  let eq1 := (x - 2 * y + 1) * (4 * x + y - 5) = 0
  let eq2 := (x + 3 * y - 6) * (x^2 - 7 * y + 10) = 0
  ∃ h1 h2 h3 h4, (∀ x y, (eq1 ↔ ((x - 2 * y + 1 = 0) ∨ (4 * x + y - 5 = 0))) 
  ∧ (eq2 ↔ ((x + 3 * y - 6 = 0) ∨ (x^2 - 7 * y + 10 = 0)))) → 
  { (x, y) | (x - 2 * y + 1 = 0 ∨ 4 * x + y - 5 = 0) ∧ (x + 3 * y - 6 = 0 ∨ x^2 - 7 * y + 10 = 0) }.to_finset.card = 2
:= by
  sorry

end number_of_intersection_points_l412_412473


namespace sqrt_factorial_div_70_l412_412970

theorem sqrt_factorial_div_70 : sqrt (8! / 70) = 24 :=
sorry

end sqrt_factorial_div_70_l412_412970


namespace integer_values_count_l412_412988

theorem integer_values_count (x : ℤ) : 
  (∃ n : ℕ, n = 8 ∧ {x : ℤ | x^2 < 9 * x}.to_finset.card = n) :=
by
  sorry

end integer_values_count_l412_412988


namespace unique_twin_prime_power_sum_l412_412557

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n

def is_twin_prime (p : ℕ) : Prop := is_prime p ∧ is_prime (p + 2)

noncomputable def is_prime_power_sum (p : ℕ) (q : ℕ) (k : ℕ) : Prop :=
is_twin_prime p ∧ q^(k : ℕ) = p + (p + 2)

theorem unique_twin_prime_power_sum :
  ∃ p q k, is_prime_power_sum p q k ∧ p = 3 ∧ (p + 2) = 5 :=
by {
  sorry
}

end unique_twin_prime_power_sum_l412_412557


namespace football_shaped_area_l412_412663

variable (π : ℝ)

def side_length := 3
def square_area := side_length * side_length
def sector_area := (1/4 : ℝ) * π * (side_length * side_length)
def combined_area := 2 * sector_area - square_area

theorem football_shaped_area :
  combined_area = (9 / 2) * π - 9 :=
by sorry

end football_shaped_area_l412_412663


namespace fraction_to_decimal_l412_412479

theorem fraction_to_decimal :
  ∀ x : ℚ, x = 52 / 180 → x = 0.1444 := 
sorry

end fraction_to_decimal_l412_412479


namespace max_profit_price_l412_412877

-- Define the conditions
def hotel_rooms : ℕ := 50
def base_price : ℕ := 180
def price_increase : ℕ := 10
def expense_per_room : ℕ := 20

-- Define the price as a function of x
def room_price (x : ℕ) : ℕ := base_price + price_increase * x

-- Define the number of occupied rooms as a function of x
def occupied_rooms (x : ℕ) : ℕ := hotel_rooms - x

-- Define the profit function
def profit (x : ℕ) : ℕ := (room_price x - expense_per_room) * occupied_rooms x

-- The statement to be proven:
theorem max_profit_price : ∃ (x : ℕ), room_price x = 350 ∧ ∀ y : ℕ, profit y ≤ profit x :=
by
  sorry

end max_profit_price_l412_412877


namespace company_total_parts_l412_412375

noncomputable def total_parts_made (planning_days : ℕ) (initial_rate : ℕ) (extra_rate : ℕ) (extra_parts : ℕ) (x_days : ℕ) : ℕ :=
  let initial_production := planning_days * initial_rate
  let increased_rate := initial_rate + extra_rate
  let actual_production := x_days * increased_rate
  initial_production + actual_production

def planned_production (planning_days : ℕ) (initial_rate : ℕ) (x_days : ℕ) : ℕ :=
  planning_days * initial_rate + x_days * initial_rate

theorem company_total_parts
  (planning_days : ℕ)
  (initial_rate : ℕ)
  (extra_rate : ℕ)
  (extra_parts : ℕ)
  (x_days : ℕ)
  (h1 : planning_days = 3)
  (h2 : initial_rate = 40)
  (h3 : extra_rate = 7)
  (h4 : extra_parts = 150)
  (h5 : x_days = 21)
  (h6 : 7 * x_days = extra_parts) :
  total_parts_made planning_days initial_rate extra_rate extra_parts x_days = 1107 := by
  sorry

end company_total_parts_l412_412375


namespace average_chocolate_pieces_per_cookie_l412_412053

-- Definitions from the conditions
def number_of_cookies := 48
def number_of_chocolate_chips := 108
def number_of_m_and_ms := (1 / 3 : ℝ) * number_of_chocolate_chips
def total_number_of_chocolate_pieces := number_of_chocolate_chips + number_of_m_and_ms

-- Statement to prove
theorem average_chocolate_pieces_per_cookie : 
  total_number_of_chocolate_pieces / number_of_cookies = 3 := by
  sorry

end average_chocolate_pieces_per_cookie_l412_412053


namespace find_S15_l412_412543

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_S15 (h : is_arithmetic_sequence a) (h1 : a 7 + a 9 = 10) :  
  let S15 := (15 / 2) * (a 1 + a 15) in
  S15 = 75 :=
by
  sorry

end find_S15_l412_412543


namespace units_digit_product_l412_412971

theorem units_digit_product : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 := 
by
  sorry

end units_digit_product_l412_412971


namespace necessary_but_not_sufficient_condition_l412_412004

theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 - 4 = 0 → x + 2 = 0 :=
by
  sorry

end necessary_but_not_sufficient_condition_l412_412004


namespace number_of_incorrect_propositions_is_1_l412_412159

-- Define the propositions 
def proposition1 : Prop := ∀ (Q : Type) [quadrilateral Q], (∀ (side_length : ℝ), Q has 4 equal sides) → Q is square
def proposition2 : Prop := ∀ (Q : Type) [quadrilateral Q], Q has instability
def proposition3 : Prop := ∀ (T1 T2 : Type) [right_angled_triangle T1] [right_angled_triangle T2], 
  (∃ (θ1 θ2 : ℝ), T1 has acute angle θ1 ∧ T2 has acute angle θ2 ∧ θ1 = θ2) → T1 ≅ T2
def proposition4 : Prop := ∀ (Q : Type) [quadrilateral Q], (opposite_sides_parallel Q) → Q is parallelogram

-- Prove that the number of incorrect propositions is 1
theorem number_of_incorrect_propositions_is_1 :
  (¬proposition3 : Prop) :=
sorry

end number_of_incorrect_propositions_is_1_l412_412159


namespace factor_count_l412_412844

theorem factor_count : 
  let x := Polynomial ℤ
  in (x ^ 10 - x ^ 2).factorization.card = 6 :=
by
  sorry

end factor_count_l412_412844


namespace count_two_digit_prime_with_units_3_not_div_by_4_l412_412558

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def ends_with_three (n : ℕ) : Prop := n % 10 = 3

def not_divisible_by_four (n : ℕ) : Prop := ¬ (4 ∣ n)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem count_two_digit_prime_with_units_3_not_div_by_4 :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ ends_with_three n ∧ not_divisible_by_four n}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_prime_with_units_3_not_div_by_4_l412_412558


namespace solve_inequality_l412_412090

theorem solve_inequality (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∨ x ∈ Set.Icc 2 4) :=
sorry

end solve_inequality_l412_412090


namespace value_of_a_squared_plus_b_squared_plus_2ab_l412_412563

theorem value_of_a_squared_plus_b_squared_plus_2ab (a b : ℝ) (h : a + b = -1) :
  a^2 + b^2 + 2 * a * b = 1 :=
by sorry

end value_of_a_squared_plus_b_squared_plus_2ab_l412_412563


namespace num_divisors_121_num_divisors_1000_num_divisors_1000000000_l412_412177

theorem num_divisors_121 : ∀ n, n = 121 → (∃ p, p=set.prod ((2+1)) ∧ ∀ (d:ℕ), d = 3):= sorry                             
theorem num_divisors_1000 : ∀ n, n = 1000 → (∃ p, p=set.prod ((3+1) * (3+1)) ∧ ∀ (d:ℕ), d = 16):= sorry                             
theorem num_divisors_1000000000 : ∀ n, n = 1000000000 → (∃ p, p=set.prod ((9+1) * (9+1)) ∧ ∀ (d:ℕ), d = 100):= sorry                             

end num_divisors_121_num_divisors_1000_num_divisors_1000000000_l412_412177


namespace cos_triple_angle_l412_412202

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l412_412202


namespace fraction_equation_solution_l412_412502

theorem fraction_equation_solution (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := 
by
  sorry

end fraction_equation_solution_l412_412502


namespace order_of_children_l412_412401

/- 
  We have five people: Kolya, Yura, Olya, Ira, and Sasha.
  We need to ensure their positions according to the given conditions.
-/
inductive Person 
| Kolya | Yura | Olya | Ira | Sasha
open Person

structure Lineup :=
(order : List Person)
(is_valid : ∀ (p1 p2 : Person), p1 ≠ p2 → p1 ∈ order → p2 ∈ order → order.indexOf p1 ≠ order.indexOf p2)

def condition1 (l : Lineup) : Prop :=
  l.order.indexOf Kolya < l.order.indexOf Yura ∧ l.order.indexOf Yura < l.order.indexOf Ira

def condition2 (l : Lineup) : Prop :=
  abs (l.order.indexOf Kolya - l.order.indexOf Olya) > 1

def condition3 (l : Lineup) : Prop :=
  abs (l.order.indexOf Kolya - l.order.indexOf Sasha) > 1 ∧ 
  abs (l.order.indexOf Yura - l.order.indexOf Sasha) > 1 ∧ 
  abs (l.order.indexOf Olya - l.order.indexOf Sasha) > 1

def correct_order (l : Lineup) : Prop :=
  l.order = [Kolya, Yura, Olya, Ira, Sasha]

theorem order_of_children :
  ∃ (l : Lineup), condition1 l ∧ condition2 l ∧ condition3 l ∧ correct_order l := by 
  sorry

end order_of_children_l412_412401


namespace vector_subtraction_l412_412996

def a : ℝ × ℝ × ℝ := (5, -3, 2)
def b : ℝ × ℝ × ℝ := (-2, 4, -3)

theorem vector_subtraction :
  (a.1 - 4 * b.1, a.2 - 4 * b.2, a.3 - 4 * b.3) = (13, -19, 14) :=
sorry

end vector_subtraction_l412_412996


namespace sin_ratios_comparison_l412_412397

theorem sin_ratios_comparison :
  ∀ {α β γ δ : ℝ}, α = 1 ∧ β = 2 ∧ γ = 3 ∧ δ = 4 →
    sin α / sin β < sin γ / sin δ :=
by
  intros α β γ δ h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  sorry

end sin_ratios_comparison_l412_412397


namespace distance_from_focus_to_line_l412_412701

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412701


namespace denis_dartboard_score_l412_412076

theorem denis_dartboard_score :
  ∀ P1 P2 P3 P4 : ℕ,
  P1 = 30 → 
  P2 = 38 → 
  P3 = 41 → 
  P1 + P2 + P3 + P4 = 4 * ((P1 + P2 + P3 + P4) / 4) → 
  P4 = 34 :=
by
  intro P1 P2 P3 P4 hP1 hP2 hP3 hTotal
  have hSum := hP1.symm ▸ hP2.symm ▸ hP3.symm ▸ hTotal
  sorry

end denis_dartboard_score_l412_412076


namespace aarti_three_times_work_l412_412851

theorem aarti_three_times_work (d : ℕ) (h : d = 5) : 3 * d = 15 :=
by
  sorry

end aarti_three_times_work_l412_412851


namespace solve_equation_l412_412486

theorem solve_equation (x : ℝ) :
  x^4 + (3 - x)^4 = 98 →
  x = 1.5 + sqrt ((33 + sqrt 238.75) / 4) ∨
  x = 1.5 - sqrt ((33 + sqrt 238.75) / 4) ∨
  x = 1.5 + sqrt ((33 - sqrt 238.75) / 4) ∨
  x = 1.5 - sqrt ((33 - sqrt 238.75) / 4) :=
by 
  sorry

end solve_equation_l412_412486


namespace solve_for_a_l412_412589

noncomputable def binomial_coeff (n k : ℕ) : ℚ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem solve_for_a (a : ℚ) (x : ℚ) (h : (binomial_coeff 9 8 * (a / x)^(9-8) * (- (sqrt (x / 2)))^8 * x^(3 * 8 / 2 - 9) : ℚ) = 9 / 4) : a = 4 :=
sorry

end solve_for_a_l412_412589


namespace sum_of_common_ratios_l412_412626

noncomputable def geometric_sequence (m x : ℝ) : ℝ × ℝ × ℝ := (m, m * x, m * x^2)

theorem sum_of_common_ratios
  (m x y : ℝ)
  (h1 : x ≠ y)
  (h2 : m ≠ 0)
  (h3 : ∃ c3 c2 d3 d2 : ℝ, geometric_sequence m x = (m, c2, c3) ∧ geometric_sequence m y = (m, d2, d3) ∧ c3 - d3 = 3 * (c2 - d2)) :
  x + y = 3 := by
  sorry

end sum_of_common_ratios_l412_412626


namespace distance_from_focus_to_line_l412_412714

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412714


namespace constant_term_expansion_l412_412385

theorem constant_term_expansion :
  let a := 5 : ℤ
  let b := (1 : ℤ) / 3
  let n := 8
  let k := 4
  let binom := nat.choose n k
  let multiply_term := (binom * (a ^ k) * (b ^ k))
  multiply_term = (43750 : ℤ) / 81 := sorry

end constant_term_expansion_l412_412385


namespace non_similar_1500_pointed_stars_count_l412_412078

open Nat

/--
Determine how many non-similar regular 1500-pointed stars exist based on the rules that no three vertices are collinear, each segment intersects another, and all segments and angles are congruent with vertices turning counterclockwise at an angle less than 180 degrees at each vertex. 
-/
theorem non_similar_1500_pointed_stars_count : 
  let n := 1500 in
  let rel_prime_count := (filter (λ m, gcd m n = 1) (range (n + 1))).length in
  (rel_prime_count - 2) / 2 = 199 :=
by
  sorry

end non_similar_1500_pointed_stars_count_l412_412078


namespace golden_ratio_expression_l412_412441

variables (R : ℝ)
noncomputable def divide_segment (R : ℝ) := R^(R^(R^2 + 1/R) + 1/R) + 1/R

theorem golden_ratio_expression :
  (R = (1 / (1 + R))) →
  divide_segment R = 2 :=
by
  sorry

end golden_ratio_expression_l412_412441


namespace tv_purchase_time_l412_412411

-- Define the constants
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000

-- Define the total expenses
def total_expenses : ℕ := food_expenses + utilities_expenses + other_expenses

-- Define the disposable income
def disposable_income : ℕ := monthly_income - total_expenses

-- Define the amount needed to buy the TV
def amount_needed : ℕ := tv_cost - current_savings

-- Define the number of months needed to save the amount needed
def number_of_months : ℕ := amount_needed / disposable_income

-- The theorem specifying that we need 2 months to save enough money for the TV
theorem tv_purchase_time : number_of_months = 2 := by
  sorry

end tv_purchase_time_l412_412411


namespace distance_from_focus_to_line_l412_412742

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412742


namespace shorter_piece_length_l412_412403

theorem shorter_piece_length (total_len : ℝ) (h1 : total_len = 60)
                            (short_len long_len : ℝ) (h2 : long_len = (1 / 2) * short_len)
                            (h3 : short_len + long_len = total_len) :
  short_len = 40 := 
  sorry

end shorter_piece_length_l412_412403


namespace distance_from_focus_to_line_l412_412691

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412691


namespace positional_relationship_l412_412144

noncomputable def plane (α : Type) := α

structure Line (α : Type) :=
  (points : set α)
  (is_line : ∀ (p1 p2 : α), p1 ≠ p2 → p1 ∈ points → p2 ∈ points → ∃ (p : α), ∀ (p3 : α), p3 ∈ points ↔ (p = p3 ∨ ∀ (r : ℝ), p3 = (p + r • (p2 - p1))) )

def parallel {α : Type} (l1 l2 : Line α) := ∀ (p1 ∈ l1.points) (p2 ∈ l2.points), p1 = p2 ∨ (∃ (v : α), ∀ (r : ℝ), p2 = p1 + r • v)

def skew {α : Type} (l1 l2 : Line α) (plane_α : plane α) := ¬(parallel l1 l2) ∧ ∀ (p1 ∈ l1.points) (p2 ∈ l2.points), p1 ≠ p2

def in_plane {α : Type} (l : Line α) (plane_α : plane α) := ∀ (p ∈ l.points), p ∈ plane_α

theorem positional_relationship {α : Type} (a b : Line α) (plane_α : plane α)
  (h_parallel_a_plane: ∀ (p1 p2 : α), p1 ≠ p2 → p1 ∈ a.points → p2 ∈ plane_α → p1 ≠ p2)
  (h_b_in_plane: in_plane b plane_α) :
  parallel a b ∨ skew a b plane_α := 
sorry

end positional_relationship_l412_412144


namespace remainder_when_divided_by_55_l412_412274

/-- 
Let M be the number formed by writing the integers from 1 to 55 consecutively.
The remainder when M is divided by 55 is 50.
-/
theorem remainder_when_divided_by_55 (M : ℕ) (hM : M = 123456789101112....535455) :
  M % 55 = 50 :=
sorry

end remainder_when_divided_by_55_l412_412274


namespace smallest_number_of_small_bottles_l412_412428

def minimum_bottles_needed (large_bottle_capacity : ℕ) (small_bottle1 : ℕ) (small_bottle2 : ℕ) : ℕ :=
  if large_bottle_capacity = 720 ∧ small_bottle1 = 40 ∧ small_bottle2 = 45 then 16 else 0

theorem smallest_number_of_small_bottles :
  minimum_bottles_needed 720 40 45 = 16 := by
  sorry

end smallest_number_of_small_bottles_l412_412428


namespace number_of_eggs_l412_412328

-- Define the conditions as assumptions
variables (marbles : ℕ) (eggs : ℕ)
variables (eggs_A eggs_B eggs_C : ℕ)
variables (marbles_A marbles_B marbles_C : ℕ)

-- Conditions from the problem
axiom eggs_total : marbles = 4
axiom marbles_total : eggs = 15
axiom eggs_groups : eggs_A ≠ eggs_B ∧ eggs_B ≠ eggs_C ∧ eggs_A ≠ eggs_C
axiom marbles_diff1 : marbles_B - marbles_A = eggs_B
axiom marbles_diff2 : marbles_C - marbles_B = eggs_C

-- Prove that the number of eggs in each group is as specified in the answer
theorem number_of_eggs :
  eggs_A = 12 ∧ eggs_B = 1 ∧ eggs_C = 2 :=
by {
  sorry
}

end number_of_eggs_l412_412328


namespace m_parallel_beta_l412_412523

variables {Plane : Type} {Line : Type} 
variable (α β : Plane) (m : Line)

-- Conditions
def m_subset_alpha : Prop := m ⊆ α
def α_parallel_beta : Prop := parallel α β

theorem m_parallel_beta (h1 : m_subset_alpha α m) (h2 : α_parallel_beta α β) : parallel m β := 
sorry

end m_parallel_beta_l412_412523


namespace cosine_330_eq_sqrt3_div_2_l412_412463

theorem cosine_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
sorry

end cosine_330_eq_sqrt3_div_2_l412_412463


namespace sqrt_m_plus_4_defined_iff_l412_412843

theorem sqrt_m_plus_4_defined_iff (m : ℝ) : (m + 4 ≥ 0) ↔ (∃ x : ℝ, x = √(m + 4)) :=
by 
  sorry

end sqrt_m_plus_4_defined_iff_l412_412843


namespace afternoon_to_morning_ratio_l412_412439

theorem afternoon_to_morning_ratio (total_kg : ℕ) (afternoon_kg : ℕ) (morning_kg : ℕ) 
  (h1 : total_kg = 390) (h2 : afternoon_kg = 260) (h3 : morning_kg = total_kg - afternoon_kg) :
  afternoon_kg / morning_kg = 2 :=
sorry

end afternoon_to_morning_ratio_l412_412439


namespace sum_of_numbers_less_than_2_l412_412370

theorem sum_of_numbers_less_than_2:
  ∀ (a b c : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 2.2 :=
by
  -- We are stating that if a = 0.8, b = 1/2, and c = 0.9, and all are less than 2, then their sum is 2.2
  sorry

end sum_of_numbers_less_than_2_l412_412370


namespace inequality_proof_l412_412629

variable {R : Type*} [linear_ordered_field R]

theorem inequality_proof (n : ℕ) (a : fin n → R) (ha : ∀ i, 0 < a i) (h_sum : (∑ i, a i) < 1) :
  (∏ i, a i) * (1 - (∑ i, a i)) / ((∑ i, a i) * ∏ i, (1 - a i)) ≤ 1 / n^(n + 1) :=
sorry

end inequality_proof_l412_412629


namespace doré_change_l412_412648

def final_amount_back (pants_price shirt_price tie_price : ℝ) 
  (pants_discount shirt_discount loyalty_discount sales_tax : ℝ) 
  (num_pants num_shirts num_ties : ℕ) 
  (initial_payment : ℝ) : ℝ := 
let total_before_discounts := (num_pants * pants_price) + (num_shirts * shirt_price) + (num_ties * tie_price)
let first_pair_discount := pants_discount * pants_price
let second_shirt_discount := shirt_discount * shirt_price
let total_item_discounts := first_pair_discount + second_shirt_discount
let after_item_discounts := total_before_discounts - total_item_discounts
let loyalty_discount_amt := loyalty_discount * after_item_discounts
let total_after_loyalty_discount := after_item_discounts - loyalty_discount_amt
let tax := sales_tax * total_after_loyalty_discount
let final_total := total_after_loyalty_discount + tax
let rounded_final_total := (round (final_total * 100)) / 100
initial_payment - rounded_final_total

theorem doré_change :
  final_amount_back 60 45 20 0.15 0.10 0.05 0.075 3 2 1 500 = 217.62 :=
by sorry

end doré_change_l412_412648


namespace bob_has_more_money_l412_412027

variable (initial_investment : ℕ)
variable (alice_final : ℕ)
variable (bob_final : ℕ)

noncomputable def alice_investment (initial_investment : ℕ) : ℕ :=
  initial_investment * 2

noncomputable def bob_investment (initial_investment : ℕ) : ℕ :=
  initial_investment * 5 + initial_investment

theorem bob_has_more_money (initial_investment : ℕ) :
  bob_investment initial_investment - alice_investment initial_investment = 8000 :=
  sorry

example : bob_has_more_money 2000 := by
  unfold bob_has_more_money bob_investment alice_investment
  simp
  sorry

end bob_has_more_money_l412_412027


namespace distance_from_right_focus_to_line_l412_412771

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412771


namespace sum_of_v_values_is_zero_l412_412546

def v (x : ℝ) : ℝ := sorry

theorem sum_of_v_values_is_zero
  (h_odd : ∀ x : ℝ, v (-x) = -v x) :
  v (-3.14) + v (-1.57) + v (1.57) + v (3.14) = 0 :=
by
  sorry

end sum_of_v_values_is_zero_l412_412546


namespace solution_of_sqrt_equation_l412_412468

theorem solution_of_sqrt_equation (x : ℝ) (hx : x + 15 ≥ 0) :
  (∃ x : ℝ, sqrt (x + 15) - 9 / sqrt (x + 15) = 3 ∧ x = (18 * sqrt 5) / 4 - 6) :=
  sorry

end solution_of_sqrt_equation_l412_412468


namespace ratio_of_areas_l412_412388

theorem ratio_of_areas (r s_3 s_2 : ℝ) (h1 : s_3^2 = r^2) (h2 : s_2^2 = 2 * r^2) :
  (s_3^2 / s_2^2) = 1 / 2 := by
  sorry

end ratio_of_areas_l412_412388


namespace cos_triplet_angle_l412_412212

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l412_412212


namespace train_pass_bridge_time_l412_412443

/-- Length of the train in meters -/
def train_length : ℝ := 720

/-- Length of the bridge in meters -/
def bridge_length : ℝ := 280

/-- Speed of the train in km/h -/
def train_speed_kmh : ℝ := 78

/-- Convert speed from km/h to m/s -/
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

/-- Speed of the train in m/s -/
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

/-- Total distance to cover in meters -/
def total_distance : ℝ := train_length + bridge_length

/-- Time required to pass the bridge in seconds -/
def time_to_pass_bridge : ℝ := total_distance / train_speed_ms

theorem train_pass_bridge_time : time_to_pass_bridge ≈ 46.15 :=
by {
  simp [time_to_pass_bridge, total_distance, train_speed_ms, kmh_to_ms, train_length, bridge_length, train_speed_kmh],
  exact sorry
}

end train_pass_bridge_time_l412_412443


namespace distance_from_right_focus_to_line_l412_412765

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412765


namespace probability_three_diff_suits_l412_412669

theorem probability_three_diff_suits :
  let num_cards := 52
  let num_suits := 4
  let cards_per_suit := num_cards / num_suits
  -- Number of ways to choose 3 different cards from a deck
  let total_ways := finset.card (finset.comb_finset (finset.Ico 0 num_cards) 3)
  -- Number of ways to choose 1 card of each suit
  let ways_diff_suits := finset.card 
    (finset.product 
      (finset.product 
        (finset.Ico 0 cards_per_suit) 
        (finset.Ico cards_per_suit (2 * cards_per_suit))) 
      (finset.Ico (2 * cards_per_suit) num_cards))
  -- The probability is the ratio of these two numbers
  ways_diff_suits / total_ways = 169 / 425 := 
sorry

end probability_three_diff_suits_l412_412669


namespace find_quotient_l412_412618

-- Define LCM function for natural numbers
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Define M' and N'
def M' : ℕ := 
  lcm 10 (lcm 11 (lcm 12 (lcm 13 (lcm 14 (lcm 15 (lcm 16 (lcm 17 (lcm 18 (lcm 19 
  (lcm 20 (lcm 21 (lcm 22 (lcm 23 (lcm 24 (lcm 25 (lcm 26 (lcm 27 (lcm 28 (lcm 29 
  (lcm 30 (lcm 31 (lcm 32 (lcm 33 (lcm 34 (lcm 35 (lcm 36 (lcm 37 (lcm 38 (lcm 39 
  (lcm 40 (lcm 41 (lcm 42 (lcm 43 (lcm 44 (lcm 45 (lcm 46 (lcm 47 (lcm 48 (lcm 49 50)))))))))))))))))))))))))))))))))))))))))))))))) 

def N' : ℕ :=
  lcm M' (lcm 51 (lcm 52 (lcm 53 (lcm 54 (lcm 55 (lcm 56 (lcm 57 (lcm 58 (lcm 59 60))))))))

-- Finally, state the theorem
theorem find_quotient :
  N' / M' = 3137 := by
  sorry

end find_quotient_l412_412618


namespace imaginary_unit_expression_l412_412082

theorem imaginary_unit_expression :
  let i := complex.I in
  i + i^2 + i^3 + i^4 = 0 :=
by {
  sorry
}

end imaginary_unit_expression_l412_412082


namespace max_regular_hours_l412_412010

theorem max_regular_hours (regular_rate : ℝ) (overtime_rate : ℝ) (total_hours : ℝ) (total_compensation : ℝ) : 
  (regular_rate = 14) →
  (overtime_rate = 14 * 1.75) →
  (total_hours = 57.224489795918366) →
  (total_compensation = 982) →
  ∃ H : ℝ, H = 40 :=
by
  intros hr or th tc hr_eq or_eq th_eq tc_eq
  use 40
  -- Given the conditions are true, the calculation should hold.
  sorry

end max_regular_hours_l412_412010


namespace complex_problem_l412_412115

open Complex

def z : ℂ := 4 + 3 * I

theorem complex_problem :
  (conj z / abs (conj z)) = (4 / 5) - (3 / 5) * I := 
sorry

end complex_problem_l412_412115


namespace range_of_m_for_function_l412_412162

noncomputable def isFunctionDefinedForAllReal (f : ℝ → ℝ) := ∀ x : ℝ, true

theorem range_of_m_for_function :
  (∀ x : ℝ, x^2 - 2 * m * x + m + 2 > 0) ↔ (-1 < m ∧ m < 2) :=
sorry

end range_of_m_for_function_l412_412162


namespace min_value_of_modulus_l412_412627

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem min_value_of_modulus
  (p q r : ℤ)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_at_least_one_zero : p = 0 ∨ q = 0 ∨ r = 0)
  (h_omega : omega^3 = 1 ∧ omega ≠ 1) :
  ∃ (s : ℝ), s = Complex.abs (↑p + ↑q * omega^2 + ↑r * omega) ∧ s = Real.sqrt 3 := 
by
  have h_minimal : omega^2 + omega + 1 = 0, from sorry,
  have h_omega_squared : omega^2 = -omega - 1, from sorry,
  use Real.sqrt 3,
  split,
  {
    -- We would prove that the smallest value is exactly sqrt(3)
    sorry
  },
  {
    -- We would prove that this value is achievable.
    sorry
  }

#check min_value_of_modulus

end min_value_of_modulus_l412_412627


namespace g_at_6_l412_412347

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_at_3 : g 3 = 4

theorem g_at_6 : g 6 = 8 :=
by 
  sorry

end g_at_6_l412_412347


namespace maximum_value_l412_412284

noncomputable def max_expression (x y z : ℝ) : ℝ :=
  2 * x * y * real.sqrt 6 + 9 * y * z

theorem maximum_value : 
  ∃ x y z : ℝ, 
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
    x^2 + y^2 + z^2 = 1 ∧ 
    max_expression x y z = real.sqrt 87 :=
by {
  sorry
}

end maximum_value_l412_412284


namespace triangle_right_angle_solution_l412_412599

def is_right_angle (a b : ℝ × ℝ) : Prop := (a.1 * b.1 + a.2 * b.2 = 0)

def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem triangle_right_angle_solution (x : ℝ) (h1 : (2, -1) = (2, -1)) (h2 : (x, 3) = (x, 3)) : 
  is_right_angle (2, -1) (x, 3) ∨ 
  is_right_angle (2, -1) (vector_sub (x, 3) (2, -1)) ∨ 
  is_right_angle (x, 3) (vector_sub (x, 3) (2, -1)) → 
  x = 3 / 2 ∨ x = 4 :=
sorry

end triangle_right_angle_solution_l412_412599


namespace inequality_proof_l412_412538

theorem inequality_proof (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2) (h2 : a2 ≥ a3) (h3 : a3 > 0) 
  (h4 : b1 ≥ b2) (h5 : b2 ≥ b3) (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) : 
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := sorry

end inequality_proof_l412_412538


namespace distance_from_focus_to_line_l412_412736

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412736


namespace min_positive_period_l412_412809

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * sin (x + π / 4) ^ 2

theorem min_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ ε > 0, ε < T → ∀ x, f (x + ε) ≠ f x) := sorry

end min_positive_period_l412_412809


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412784

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412784


namespace arc_length_calc_l412_412043

noncomputable def arc_length_of_ln_over_2x : ℝ :=
  ∫ x in Real.sqrt 3 .. Real.sqrt 8, Real.sqrt (1 + (-(1 / x))^2)

theorem arc_length_calc :
  arc_length_of_ln_over_2x = 1 + (1 / 2) * Real.log (3 / 2) :=
by
  sorry

end arc_length_calc_l412_412043


namespace probability_of_sums_6_8_10_l412_412425

-- Define the first die with faces 1, 2, 3, 3, 4, 4
def Die1 : List ℕ := [1, 2, 3, 3, 4, 4]

-- Define the second die with faces 1, 2, 5, 6, 7, 9
def Die2 : List ℕ := [1, 2, 5, 6, 7, 9]

-- A function to compute the probability of an event from the given dice
noncomputable def probability_of_sum (target_sums : List ℕ) : ℚ :=
  let possible_outcomes := (list.product Die1 Die2).filter (fun (d : ℕ × ℕ) => (d.1 + d.2) ∈ target_sums)
  possible_outcomes.length / (Die1.length * Die2.length : ℚ)

-- The target sums we are interested in
def target_sums : List ℕ := [6, 8, 10]

-- The main statement asserting the probability of the sum being 6, 8, or 10 is 11/36
theorem probability_of_sums_6_8_10 : probability_of_sum target_sums = 11 / 36 := by
  sorry

end probability_of_sums_6_8_10_l412_412425


namespace find_speed_of_first_train_l412_412865

-- Define conditions
def length_of_first_train : ℝ := 250
def length_of_second_train : ℝ := 250.04
def speed_of_second_train : ℝ := 80
def crossing_time : ℝ := 9

-- Define conversion factor for km/hr to m/s
def km_per_hr_to_m_per_s : ℝ := 5 / 18

-- Define the total length of the trains
def total_length : ℝ := length_of_first_train + length_of_second_train

-- Define relative speed in km/hr and m/s
def relative_speed_m_per_s (v1_km_per_hr : ℝ) : ℝ :=
  (v1_km_per_hr + speed_of_second_train) * km_per_hr_to_m_per_s

-- Problem statement to prove
theorem find_speed_of_first_train (v1 : ℝ) :
  (total_length = relative_speed_m_per_s v1 * crossing_time) →
  (v1 = 120.016) :=
sorry

end find_speed_of_first_train_l412_412865


namespace circle_passes_through_fixed_point_area_circle_passes_through_range_area_of_quadrilateral_sine_value_angle_condition_l412_412121

-- Definitions for the conditions
def circle_center (α : ℝ) : ℝ × ℝ := (sqrt 2 * cos α - 1, sqrt 2 * sin α)
def circle_radius : ℝ := sqrt 2
def point_on_line (P : ℝ × ℝ) : Prop := P.1 + P.2 = 5
def circle_eq (x y α : ℝ) : Prop := (x - (sqrt 2 * cos α - 1))^2 + (y - sqrt 2 * sin α)^2 = 2

-- Statements to prove
theorem circle_passes_through_fixed_point :
  ∀ α ∈ [0, 2 * Real.pi], circle_eq (-1) 0 α :=
sorry

theorem area_circle_passes_through : 
  ∀ α ∈ [0, 2 * Real.pi], (π * (2 * sqrt 2)^2 = 8 * π) :=
sorry

theorem range_area_of_quadrilateral (P : ℝ × ℝ) (α : ℝ) :
  point_on_line P → ¬(2 * sqrt 3 ≤ 0.5 * 2 * |dist P (circle_center α)| * circle_radius 
    ∧ 0.5 * 2 * |dist P (circle_center α)| * circle_radius ≤ 2 * sqrt 15) :=
sorry

theorem sine_value_angle_condition (P : ℝ × ℝ) (α : ℝ) :
  point_on_line P → (circle_center α).1 * (circle_center α).2 = 0 →
  (sqrt 15 / 8 ≤ angle_sin_value ∧ angle_sin_value ≤ sqrt 3 / 2) :=
sorry

end circle_passes_through_fixed_point_area_circle_passes_through_range_area_of_quadrilateral_sine_value_angle_condition_l412_412121


namespace radius_of_box_inscribed_in_sphere_l412_412883

noncomputable def box_inscribed_in_sphere_radius : ℝ :=
let x := sorry in
let y := 18 - 4 * x in
let z := 3 * x in
let r := 1 / 2 * real.sqrt (x^2 + y^2 + z^2) in
have h1 : 2 * (x * y + y * z + x * z) = 162 := sorry,
have h2 : 4 * (x + y + z) = 72 := sorry,
have h3 : z = 3 * x := sorry,
begin
  sorry
end

theorem radius_of_box_inscribed_in_sphere : box_inscribed_in_sphere_radius = 9 :=
begin
  rw box_inscribed_in_sphere_radius,
  sorry
end

end radius_of_box_inscribed_in_sphere_l412_412883


namespace tangent_line_eq_at_x0_l412_412805

theorem tangent_line_eq_at_x0 : 
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.exp x) →
  (∀ x₀, x₀ = 0) →
  ∃ k : ℝ, k = 1 ∧ f x₀ = 1 →
  ∀ x y₀, y₀ = 1 → (y - y₀) = k * (x - x₀) → (y = x + 1) :=
by
  intro f hf hx₀
  use 1, (by { rw [hf, hx₀, Real.exp_zero] }) sorry

end tangent_line_eq_at_x0_l412_412805


namespace no_primes_divisible_by_45_l412_412181

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define 45 and its prime factors
def is_divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem to prove the number of prime numbers divisible by 45 is 0
theorem no_primes_divisible_by_45 : ∀ n : ℕ, is_prime n → is_divisible_by_45 n → false :=
by
  intro n
  assume h_prime h_div_45
  sorry

end no_primes_divisible_by_45_l412_412181


namespace distance_from_focus_to_line_l412_412730

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412730


namespace proper_sets_exist_l412_412024

def proper_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, (1 ≤ w ∧ w ≤ 500) → ∃ (used_weights : List ℕ), (used_weights ⊆ weights) ∧ (used_weights.sum = w ∧ ∀ (alternative_weights : List ℕ), (alternative_weights ⊆ weights ∧ alternative_weights.sum = w) → used_weights = alternative_weights)

theorem proper_sets_exist (weights : List ℕ) :
  (weights.sum = 500) → 
  ∃ (sets : List (List ℕ)), sets.length = 3 ∧ (∀ s ∈ sets, proper_set s) :=
by
  sorry

end proper_sets_exist_l412_412024


namespace Zircon_position_quarter_way_l412_412355

theorem Zircon_position_quarter_way (perigee apogee : ℝ) (H_perigee : perigee = 3) (H_apogee : apogee = 15) :
  ∃ P, P = 6.75 ∧
       P = perigee + (1/4) * (apogee - perigee) :=
by
  -- Introduce the main variables and assumptions
  let perigee := 3
  let apogee := 15
  let major_axis := perigee + apogee
  let quarter_way_position := perigee + (1 / 4) * (major_axis - perigee)
  use quarter_way_position
  split
  -- Show that P is correctly calculated
  { rw [←H_perigee, ←H_apogee], simp, norm_num, exact eq.refl 6.75 }
  -- Then show that quarter_way_position equals the computed P
  { rw [←H_perigee, ←H_apogee], simp, norm_num }

end Zircon_position_quarter_way_l412_412355


namespace find_values_l412_412515

variable (α : Real)

-- Given condition
def tan_neg_pi_sub_alpha : Real := -3
def tan_pi_minus_alpha : tan (π - α) = tan_neg_pi_sub_alpha

theorem find_values :
  (tan α = 3) ∧ 
  ( (sin (π - α) - cos (π + α) - sin (2 * π - α) + cos (-α)) / 
    (sin (π / 2 - α) + cos (3 * π / 2 - α)) = -4 ) :=
by
  -- Add the given conditions here
  have h1 : tan (π - α) = - tan α, from @tan_pi_minus α;
  -- Add the desired proof structure here
  sorry

end find_values_l412_412515


namespace total_prime_factors_l412_412495

theorem total_prime_factors :
  let expr := (3^5) * (5^7) * (7^4) * (11^2) * (13^3)
  (count_total_prime_factors expr) = 21 :=
sorry

-- Helper function to count the total number of prime factors
noncomputable def count_total_prime_factors (n : ℕ) : ℕ :=
  sorry

end total_prime_factors_l412_412495


namespace cos_triple_angle_l412_412201

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l412_412201


namespace initial_stock_decaf_percentage_l412_412427

-- Definitions as conditions of the problem
def initial_coffee_stock : ℕ := 400
def purchased_coffee_stock : ℕ := 100
def percentage_decaf_purchased : ℕ := 60
def total_percentage_decaf : ℕ := 32

/-- The proof problem statement -/
theorem initial_stock_decaf_percentage : 
  ∃ x : ℕ, x * initial_coffee_stock / 100 + percentage_decaf_purchased * purchased_coffee_stock / 100 = total_percentage_decaf * (initial_coffee_stock + purchased_coffee_stock) / 100 ∧ x = 25 :=
sorry

end initial_stock_decaf_percentage_l412_412427


namespace original_six_digit_number_is_285714_l412_412885

theorem original_six_digit_number_is_285714 
  (N : ℕ) 
  (h1 : ∃ x, N = 200000 + x ∧ 10 * x + 2 = 3 * (200000 + x)) :
  N = 285714 := 
sorry

end original_six_digit_number_is_285714_l412_412885


namespace total_journey_time_l412_412243

theorem total_journey_time
  (river_speed : ℝ)
  (boat_speed_still_water : ℝ)
  (distance_upstream : ℝ)
  (total_journey_time : ℝ) :
  river_speed = 2 → 
  boat_speed_still_water = 6 → 
  distance_upstream = 48 → 
  total_journey_time = (distance_upstream / (boat_speed_still_water - river_speed) + distance_upstream / (boat_speed_still_water + river_speed)) → 
  total_journey_time = 18 := 
by
  intros h1 h2 h3 h4
  sorry

end total_journey_time_l412_412243


namespace log_power_rule_l412_412221

-- Definitions and conditions
def x : ℝ := 3 ^ (5 : ℝ)
def log_base3 (y : ℝ) : ℝ := Real.log y / Real.log 3 

-- Proof statement
theorem log_power_rule (h : log_base3 x = 5) : log_base3 (x^3) = 15 := by
  sorry -- Proof is omitted

end log_power_rule_l412_412221


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412786

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412786


namespace possible_integer_lengths_count_l412_412565

variable (n : ℕ)

-- Conditions according to the triangle inequality theorem
def cond1 : Prop := n + 6 > 10
def cond2 : Prop := n + 10 > 6
def cond3 : Prop := 6 + 10 > n

-- Combining these conditions
def valid_third_side : Prop := 4 < n ∧ n < 16

-- The main statement we want to prove
theorem possible_integer_lengths_count (n : ℕ) : (∃ m : ℕ, m = (finset.filter (λ x, 4 < x ∧ x < 16) (finset.range 17)).card ∧ m = 11) :=
by
  sorry

end possible_integer_lengths_count_l412_412565


namespace regular_polygons_constructible_l412_412842

-- Define a right triangle where the smaller leg is half the length of the hypotenuse
structure RightTriangle30_60_90 :=
(smaller_leg hypotenuse : ℝ)
(ratio : smaller_leg = hypotenuse / 2)

-- Define the constructibility of polygons
def canConstructPolygon (n: ℕ) : Prop :=
n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 12

theorem regular_polygons_constructible (T : RightTriangle30_60_90) :
  ∀ n : ℕ, canConstructPolygon n :=
by
  intro n
  sorry

end regular_polygons_constructible_l412_412842


namespace sum_f_eq_645_l412_412974

noncomputable def f (x : ℝ) := 4^x / (4^x + 2)

theorem sum_f_eq_645 : (∑ k in Finset.range 1290 | 1 ≤ k + 1, f ((k + 1 : ℝ) / 1291) = 645 :=
by
  sorry

end sum_f_eq_645_l412_412974


namespace area_trapezoid_EFGH_l412_412837

def point := ℝ × ℝ

def E : point := (0, 0)
def F : point := (0, 3)
def G : point := (5, 3)
def H : point := (5, -4)

def length (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def EF := length E F
noncomputable def GH := length G H
noncomputable def height := real.abs (G.1 - E.1)

noncomputable def trapezoid_area (a b height : ℝ) : ℝ :=
  0.5 * (a + b) * height

theorem area_trapezoid_EFGH : trapezoid_area EF GH height = 25 := 
by
  sorry

end area_trapezoid_EFGH_l412_412837


namespace largest_integer_n_l412_412069

def floor (x : ℝ) := ⌊x⌋

theorem largest_integer_n (n : ℕ) (h : floor (real.sqrt n) = 5) : n ≤ 35 :=
by {
  sorry
}

end largest_integer_n_l412_412069


namespace find_OM_that_minimizes_find_cos_AMB_l412_412139

-- Conditions

def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)
def line_OP (t : ℝ) : ℝ × ℝ := (2 * t, t)

-- Questions

-- (1) Find \(\overrightarrow{OM}\) that minimizes \(\overrightarrow{MA} \cdot \overrightarrow{MB}\)
theorem find_OM_that_minimizes :
  ∃ M : ℝ × ℝ, (∃ t : ℝ, M = line_OP t) ∧
  (∀ N : ℝ × ℝ, ∃ t': ℝ, N = line_OP t' → 
  let MA := (OA.1 - M.1, OA.2 - M.2),
      MB := (OB.1 - M.1, OB.2 - M.2),
      NA := (OA.1 - N.1, OA.2 - N.2),
      NB := (OB.1 - N.1, OB.2 - N.2)
  in MA.1 * MB.1 + MA.2 * MB.2 ≤ NA.1 * NB.1 + NA.2 * NB.2) :=
sorry

-- (2) Find the cosine value of \(\angle AMB\) for the point \(M\) found in (1)
theorem find_cos_AMB :
  let M := (4, 2),
      MA := (OA.1 - M.1, OA.2 - M.2),
      MB := (OB.1 - M.1, OB.2 - M.2)
  in (MA.1 * MB.1 + MA.2 * MB.2) / (Math.sqrt (MA.1 ^ 2 + MA.2 ^ 2) * Math.sqrt (MB.1 ^ 2 + MB.2 ^ 2)) = -4 * (Math.sqrt 17) / 17 :=
sorry


end find_OM_that_minimizes_find_cos_AMB_l412_412139


namespace sum_of_ceil_sqrt_l412_412950

theorem sum_of_ceil_sqrt :
  (∑ n in finset.range (16 - 10 + 1), ⌈ real.sqrt (10 + n : ℝ) ⌉) +
  (∑ n in finset.range (25 - 17 + 1), ⌈ real.sqrt (17 + n : ℝ) ⌉) +
  (∑ n in finset.range (36 - 26 + 1), ⌈ real.sqrt (26 + n : ℝ) ⌉) +
  (∑ n in finset.range (40 - 37 + 1), ⌈ real.sqrt (37 + n : ℝ) ⌉) = 167 :=
by
  sorry

end sum_of_ceil_sqrt_l412_412950


namespace stratified_sampling_draws_l412_412670

theorem stratified_sampling_draws (A B C : ℕ) (total_samples : ℕ)
  (h_ratio : A.to_rat / B.to_rat = 5/3 ∧ A.to_rat / C.to_rat = 5/2)
  (h_total : total_samples = 100) :
  C = 20 := 
by sorry

end stratified_sampling_draws_l412_412670


namespace distance_from_focus_to_line_l412_412740

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412740


namespace distance_from_right_focus_to_line_l412_412720

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412720


namespace complement_union_containment_l412_412299

open Set

variables (U : Set ℝ) (A B : Set ℝ)
noncomputable theory

def U : Set ℝ := univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 2}

theorem complement_union_containment :
  (compl A ∪ B) ⊆ {x | x < 2} ∧ (compl A ∪ B) ⊆ {x | x < 3} :=
sorry

end complement_union_containment_l412_412299


namespace hcf_of_two_numbers_l412_412807

noncomputable def H : ℕ := 322 / 14

theorem hcf_of_two_numbers (H k : ℕ) (lcm_val : ℕ) :
  lcm_val = H * 13 * 14 ∧ 322 = H * k ∧ 322 / 14 = H → H = 23 :=
by
  sorry

end hcf_of_two_numbers_l412_412807


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412788

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412788


namespace selected_female_athletes_l412_412890

-- Definitions based on conditions
def total_male_athletes := 56
def total_female_athletes := 42
def selected_male_athletes := 8
def male_to_female_ratio := 4 / 3

-- Problem statement: Prove that the number of selected female athletes is 6
theorem selected_female_athletes :
  selected_male_athletes * (3 / 4) = 6 :=
by 
  -- Placeholder for the proof
  sorry

end selected_female_athletes_l412_412890


namespace trigonometric_proof_l412_412118

theorem trigonometric_proof
  (α β : ℝ) 
  (hα : 0 < α) (hβ : 0 < β) (hαβ : α + β < π)
  (hcos_α : cos α = 4/5)
  (hcos_αβ : cos (α + β) = 5/13) :
  sin (2 * α) = 24 / 25 ∧ cos β = 56 / 65 := by
  sorry

end trigonometric_proof_l412_412118


namespace distance_hyperbola_focus_to_line_l412_412773

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412773


namespace eleanor_cookies_sum_l412_412085

theorem eleanor_cookies_sum :
  let N := {n // n % 12 = 5 ∧ n % 8 = 2 ∧ n < 100} in
  ∑ n in N, n = 29 :=
by
  sorry

end eleanor_cookies_sum_l412_412085


namespace find_a_l412_412632

noncomputable def z (a : ℝ) : ℂ := (a / (1 - 2*I)) + I

theorem find_a (a : ℝ) (h : (z a).re = -(z a).im) : a = -5/3 :=
by
  sorry

end find_a_l412_412632


namespace common_roots_sum_is_12_l412_412931

theorem common_roots_sum_is_12
  (C D u v w t : ℚ)
  (h1 : u + v + w = 0)
  (h2 : uv + uw + vw = C)
  (h3 : uvw = 24)
  (h4 : u + v + t = -D)
  (h5 : uv + ut + vt = 0)
  (h6 : uvt = -96)
  (h_product : (∃ (a b c : ℚ), u * v = a * (c^(1/b)) ∧ a + b + c = 12)) :
  a + b + c = 12 := sorry

end common_roots_sum_is_12_l412_412931


namespace unique_seq_l412_412098

def seq (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j

theorem unique_seq (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n) : 
  seq a ↔ (∀ n, a n = n) := 
by
  intros
  sorry

end unique_seq_l412_412098


namespace cos_triple_angle_l412_412208

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l412_412208


namespace problem_statement_l412_412248

section
variables {t m : ℝ} (P : ℝ × ℝ) (x y : ℝ)

-- Definitions for the curves
def parametric_C1 (t m : ℝ) : ℝ × ℝ := (t, m + t)
def polar_C2 (ρ θ : ℝ) : ℝ := ρ^2 = 3 / (3 - 2 * (Real.cos θ)^2)

-- Translations from parametric or polar to Cartesian equations
def general_eq_C1 (x y m : ℝ) : Prop := x - y + m = 0
def cartesian_eq_C2 (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Proves that the minimum distance from point P to curve C1 implies certain values of m
def min_distance_C1_C2 (P : ℝ × ℝ) (d : ℝ) : Prop :=
  let α := Real.arccos ((√3) * Real.fst P + Real.sin (Real.snd P)) / 2 in
  let dist := Real.abs (2 * Real.cos (α + Real.pi / 6) + m) / √2 in
  d = 2 * √2 ∧ (m = -4 - √3 ∨ m = 6)

-- Main problem: proving all statements above
theorem problem_statement :
  (∀ t : ℝ, ∃ x y m, parametric_C1 t m = (x, y) → general_eq_C1 x y m) ∧
  (∀ ρ θ, polar_C2 ρ θ → (∃ x y, cartesian_eq_C2 x y ∧ 0 ≤ y ∧ y ≤ 1)) ∧
  (∀ P : ℝ × ℝ, (∃ d, min_distance_C1_C2 P d → d = 2 * √2 ∧ (m = -4 - √3 ∨ m = 6))) :=
sorry -- Proof to be filled in
end

end problem_statement_l412_412248


namespace square_side_length_of_rearranged_pentagons_l412_412679

theorem square_side_length_of_rearranged_pentagons :
  let area := 10 * 20,
      side := Real.sqrt area
  in side = 10 * Real.sqrt 2 :=
by
  let area := 10 * 20
  let side := Real.sqrt area
  have h : area = 200 := by calc area = 10 * 20 : by rfl; simp
  have h2 : side = Real.sqrt 200 := by rfl
  rw [h, Real.sqrt_inj_of_nonneg (by norm_num)],
  { norm_num }
  sorry

end square_side_length_of_rearranged_pentagons_l412_412679


namespace total_sections_l412_412857

theorem total_sections (boys girls gcd sections_boys sections_girls : ℕ) 
  (h_boys : boys = 408) 
  (h_girls : girls = 264) 
  (h_gcd: gcd = Nat.gcd boys girls)
  (h_sections_boys : sections_boys = boys / gcd)
  (h_sections_girls : sections_girls = girls / gcd)
  (h_total_sections : sections_boys + sections_girls = 28)
: sections_boys + sections_girls = 28 := by
  sorry

end total_sections_l412_412857


namespace log_sum_eq_two_l412_412581

variable {a : ℕ → ℝ}
variable (h_geom : ∀ n, a (n + 1) = (a 1) * (a n)^n)
variable (h_pos : ∀ n, a n > 0)
variable (h_condition : a 3 * a 8 = 9)

theorem log_sum_eq_two : log 3 (a 1) + log 3 (a 10) = 2 := by
  sorry

end log_sum_eq_two_l412_412581


namespace find_sum_of_constants_l412_412675

noncomputable def sum_of_constants (x : ℝ) : ℝ :=
  9 + 9 + 4 + 5

theorem find_sum_of_constants :
  (∃ x : ℝ, (1/x + 1/(x + 4) - 1/(x + 6) - 1/(x + 10) + 1/(x + 12) + 1/(x + 16) - 1/(x + 18) - 1/(x + 20) = 0) ∧ 
           (∃ a b c d : ℝ, x = -a ± sqrt (b ± c * sqrt d) ∧ a = 9 ∧ b = 9 ∧ c = 4 ∧ d = 5)) →
  sum_of_constants x = 27 := 
by
  sorry

end find_sum_of_constants_l412_412675


namespace number_of_integers_satisfying_inequality_l412_412984

theorem number_of_integers_satisfying_inequality :
  ∃ S : Finset ℤ, (∀ x ∈ S, x^2 < 9 * x) ∧ S.card = 8 :=
by
  sorry

end number_of_integers_satisfying_inequality_l412_412984


namespace rectangle_tileability_l412_412660

theorem rectangle_tileability (m n b : ℕ) : (∃ (k l : ℕ), m = 2 * b * k ∧ n = 2 * b * l) ↔ (∃ f : ℕ → ℕ → Prop, (∀ i j, f(i,j) → (1 ≤ i ∧ i ≤ m) ∧ (1 ≤ j ∧ j ≤ n))) :=
sorry

end rectangle_tileability_l412_412660


namespace number_of_herds_l412_412924

-- Definitions from the conditions
def total_sheep : ℕ := 60
def sheep_per_herd : ℕ := 20

-- The statement to prove
theorem number_of_herds : total_sheep / sheep_per_herd = 3 := by
  sorry

end number_of_herds_l412_412924


namespace tetrahedron_sum_squares_l412_412001

theorem tetrahedron_sum_squares (a b c : ℝ) (x y z : ℝ) 
  (h1 : a^2 = y^2 + z^2)
  (h2 : b^2 = x^2 + z^2)
  (h3 : c^2 = x^2 + y^2)
  (circumradius : sqrt (x^2 + y^2 + z^2) = 2):
  a^2 + b^2 + c^2 = 8 := 
by
  sorry

end tetrahedron_sum_squares_l412_412001


namespace distance_from_right_focus_to_line_l412_412717

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412717


namespace sin_sum_triangle_inequality_l412_412285

theorem sin_sum_triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_inequality_l412_412285


namespace day_100th_of_year_N_minus_1_l412_412604

/-- Define a function to calculate the day of the week given a specific day of the year and a reference weekday. -/
def day_of_week (day year : ℕ) (start_day : ℕ) : ℕ :=
  (start_day + day - 1) % 7

/-- Define if a year is a leap year -/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

noncomputable def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

/-- Main theorem to prove the day of the 100th day of year N-1 given the specific conditions. -/
theorem day_100th_of_year_N_minus_1 (N : ℕ)
  (h1 : day_of_week 300 N 2 = 2)
  (h2 : day_of_week 200 (N+1) 2 = 2) :
  day_of_week 100 (N-1) (day_of_week 1 (N-1) (day_of_week 365 (N-2) (day_of_week 1 (N-2) 2))) = 4 :=
sorry

end day_100th_of_year_N_minus_1_l412_412604


namespace distance_from_focus_to_line_l412_412744

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412744


namespace lines_are_skew_l412_412960

theorem lines_are_skew (b : ℝ) : 
  (∀ (t u : ℝ), 
    ⟨2 + 3 * t, 1 + 4 * t, b + 5 * t⟩ ≠ ⟨3 + 7 * u, 5 + 3 * u, 2 + u⟩) 
  ↔ b ≠ -(79 / 19) :=
sorry

end lines_are_skew_l412_412960


namespace sequence_sum_l412_412046

theorem sequence_sum : (Finset.range 101).sum (λ n, (-1)^n * (n + 1)) = 51 :=
by {
  sorry
}

end sequence_sum_l412_412046


namespace tangent_circle_ratio_l412_412327

/-- Definitions representing the geometric objects and given conditions -/
def Tangent (O D : Point) : Prop := sorry -- O being the circle and D the tangent at some point
def RatioACAB (A C B : Point) : Prop := 
  (length (line_segment AC)) / (length (line_segment AB)) = (2 / 5)

/-- Stating the given problem using Lean 4 -/
theorem tangent_circle_ratio (O A B C D F H E : Point)  
  (h1: Tangent O D)
  (h2: RatioACAB A C B) :
  (length (line_segment AF)) / (length (line_segment DF)) = (7 / 5) := 
sorry

end tangent_circle_ratio_l412_412327


namespace LineDoesNotIntersectParabola_sum_r_s_l412_412275

noncomputable def r : ℝ := -0.6
noncomputable def s : ℝ := 40.6
def Q : ℝ × ℝ := (10, -6)
def line_through_Q_with_slope (m : ℝ) (p : ℝ × ℝ) : ℝ := m * p.1 - 10 * m - 6
def parabola (x : ℝ) : ℝ := 2 * x^2

theorem LineDoesNotIntersectParabola (m : ℝ) :
  r < m ∧ m < s ↔ (m^2 - 4 * 2 * (10 * m + 6) < 0) :=
by sorry

theorem sum_r_s : r + s = 40 :=
by sorry

end LineDoesNotIntersectParabola_sum_r_s_l412_412275


namespace only_prop4_is_correct_l412_412901

variable {l : Type} {α : Type}

-- Definitions corresponding to propositions
def prop1 := ∀ {l : Type} {α : Type}, (∃^∞ x, x ∈ l ∧ x ∉ α) → (∀ y, y ∈ l → y ∉ α)
def prop2 := ∀ {l : Type} {α : Type}, (∀ y, y ∈ l → y ∈ α) → (∃ z, z ∈ α → z ∈ l)
def prop3 := ∀ {l m : Type} {α : Type}, (∀ y, y ∈ l → y ∈ m) ∧ (∀ z, z ∈ l → z ∈ α) → (∀ w, w ∈ m → w ∈ α)
def prop4 := ∀ {l : Type} {α : Type}, (∀ y, y ∈ l → y ∉ α) → (∀ z, z ∉ α → z ∉ l)

-- Proof statement asserting only Proposition ④ is correct
theorem only_prop4_is_correct : prop4 ∧ ¬ (prop1 ∨ prop2 ∨ prop3) :=
by
  sorry

end only_prop4_is_correct_l412_412901


namespace distance_hyperbola_focus_to_line_l412_412781

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412781


namespace intersection_complement_l412_412298

def U : Set ℤ := { x | -1 ≤ x ∧ x ≤ 5 }
def A : Set ℤ := { 1, 2, 5 }
def B : Set ℕ := { x | -1 < x ∧ x < 4 }

noncomputable def C_U_A : Set ℤ := U \ A

theorem intersection_complement (B_intersect_CU_A : B ∩ C_U_A = {0, 3}) : B ∩ C_U_A = {0, 3} :=
sorry

end intersection_complement_l412_412298


namespace surveys_from_retired_is_12_l412_412476

-- Define the given conditions
def ratio_retired : ℕ := 2
def ratio_current : ℕ := 8
def ratio_students : ℕ := 40
def total_surveys : ℕ := 300
def total_ratio : ℕ := ratio_retired + ratio_current + ratio_students

-- Calculate the expected number of surveys from retired faculty
def number_of_surveys_retired : ℕ := total_surveys * ratio_retired / total_ratio

-- Lean 4 statement for proof
theorem surveys_from_retired_is_12 :
  number_of_surveys_retired = 12 :=
by
  -- Proof to be filled in
  sorry

end surveys_from_retired_is_12_l412_412476


namespace max_value_of_expression_l412_412133

open Real

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  2 * x * y + y * z + 2 * z * x ≤ 4 / 7 := 
sorry

end max_value_of_expression_l412_412133


namespace phase_shift_of_cosine_l412_412491

def B : ℝ := 5
def C : ℝ := π / 2

theorem phase_shift_of_cosine : (C / B) = π / 10 := 
by
  sorry

end phase_shift_of_cosine_l412_412491


namespace geometric_sequence_common_ratio_l412_412678

theorem geometric_sequence_common_ratio (a1 a2 a3 : ℤ) (r : ℤ)
  (h1 : a1 = 9) (h2 : a2 = -18) (h3 : a3 = 36) (h4 : a2 / a1 = r) (h5 : a3 = a2 * r) :
  r = -2 := 
sorry

end geometric_sequence_common_ratio_l412_412678


namespace distance_to_line_is_sqrt5_l412_412798

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412798


namespace no_prime_divisible_by_45_l412_412188

theorem no_prime_divisible_by_45 : ∀ (p : ℕ), Prime p → ¬ (45 ∣ p) :=
by {
  intros p h_prime h_div,
  have h_factors := Nat.factors_unique,
  sorry
}

end no_prime_divisible_by_45_l412_412188


namespace sum_series_l412_412270

/-- Let a and b be relatively prime positive integers such that
a / b = 1 / 2^1 + 2 / 3^2 + 3 / 2^3 + 4 / 3^4 + 5 / 2^5 + ... (where the numerators always increase by 1, and the denominators alternate between powers of 2 and 3, with exponents also increasing by 1 for each subsequent term).
The task is to prove that a + b = 625. -/
theorem sum_series (a b : ℕ) (h_rel_prime : Nat.coprime a b)
    (h_series : (a : ℚ) / b = ∑' n, n / 3 ^ (n % 2)) : a + b = 625 :=
sorry

end sum_series_l412_412270


namespace probability_exactly_three_even_l412_412039

theorem probability_exactly_three_even (p : ℕ → ℚ) (n : ℕ) (k : ℕ) (h : p 20 = 1/2 ∧ n = 5 ∧ k = 3) :
  (∃ C : ℚ, (C = (Nat.choose n k : ℚ)) ∧ (p 20)^n = 1/32) → (C * 1/32 = 5/16) :=
by
  sorry

end probability_exactly_three_even_l412_412039


namespace evaluate_sum_l412_412946

theorem evaluate_sum :
  (∑ n in finset.range 31, (let k := n + 10 in
    if 10 ≤ k ∧ k ≤ 16 then 4 else
    if 17 ≤ k ∧ k ≤ 25 then 5 else
    if 26 ≤ k ∧ k ≤ 36 then 6 else
    if 37 ≤ k ∧ k ≤ 40 then 7 else 0)) = 167 :=
by sorry

end evaluate_sum_l412_412946


namespace a_plus_b_l412_412126

def smallest_fractional_factor_count (n : ℕ) : ℕ :=
  (n + 1) / 2

def f (x : ℕ) : ℕ :=
  let divisors := (Finset.range (x + 1)).filter (λ d, d > 0 ∧ x % d = 0)
  let smallest_divisors := (divisors.sort (· ≤ ·)).take (smallest_fractional_factor_count divisors.card)
  smallest_divisors.prod id

noncomputable def a : ℕ :=
  let eligible := (Finset.range 1000).filter (λ x, f x = x)
  if eligible.nonempty then eligible.min' eligible.nonempty else 0

noncomputable def b : ℕ :=
  let eligible_ns := (Finset.range 100).filter (λ n, ∃ y, (y > 1) ∧ (Finset.range (y + 1)).filter (λ d, d > 0 ∧ y % d = 0).card = n ∧ f y = y)
  if eligible_ns.nonempty then eligible_ns.min' eligible_ns.nonempty else 0

theorem a_plus_b : a + b = 31 := by
  -- Proof omitted
  sorry

end a_plus_b_l412_412126


namespace sufficient_but_not_necessary_condition_l412_412291

-- Definitions based on the given conditions:
def purely_imaginary (z : ℂ) : Prop := z.re = 0

-- The main theorem statement adhering to the given problem and solution:
theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → purely_imaginary ((↑(a - 1) * ↑(a + 2) : ℂ) + (↑(a + 3) : ℂ) * I)) ∧
  (purely_imaginary ((↑(a - 1) * ↑(a + 2) : ℂ) + (↑(a + 3) : ℂ) * I) → (a = 1 ∨ a = -2)) :=
sorry

end sufficient_but_not_necessary_condition_l412_412291


namespace raritet_meets_ferries_l412_412015

theorem raritet_meets_ferries :
  (∀ (n : ℕ), ∃ (ferry_departure : Nat), ferry_departure = n ∧ ferry_departure + 8 = 8) →
  (∀ (m : ℕ), ∃ (raritet_departure : Nat), raritet_departure = m ∧ raritet_departure + 8 = 8) →
  ∃ (total_meetings : Nat), total_meetings = 17 := 
by
  sorry

end raritet_meets_ferries_l412_412015


namespace babysitting_hours_to_cover_fees_l412_412639

-- Definitions based on the conditions
def initial_rate: ℝ := 10.0
def raise_rate: ℝ := 0.10
def hours_per_block: ℝ := 5.0
def application_fee: ℝ := 25.0
def number_of_colleges: ℝ := 6.0
def total_fees: ℝ := number_of_colleges * application_fee

-- Proposition to prove
theorem babysitting_hours_to_cover_fees: 
  real.bounded_above (λ n, n ∈ {k | ∑ i in range (k + 1), 
    if i = 0 then (hours_per_block * initial_rate)
    else (hours_per_block * (initial_rate * (1 + i * raise_rate)))
    ≥ total_fees}) :=
by
  sorry

end babysitting_hours_to_cover_fees_l412_412639


namespace CodgerNeedsTenPairs_l412_412467

def CodgerHasThreeFeet : Prop := true

def ShoesSoldInPairs : Prop := true

def ShoesSoldInEvenNumberedPairs : Prop := true

def CodgerOwnsOneThreePieceSet : Prop := true

-- Main theorem stating Codger needs 10 pairs of shoes to have 7 complete 3-piece sets
theorem CodgerNeedsTenPairs (h1 : CodgerHasThreeFeet) (h2 : ShoesSoldInPairs)
  (h3 : ShoesSoldInEvenNumberedPairs) (h4 : CodgerOwnsOneThreePieceSet) : 
  ∃ pairsToBuy : ℕ, pairsToBuy = 10 := 
by {
  -- We have to prove codger needs 10 pairs of shoes to have 7 complete 3-piece sets
  sorry
}

end CodgerNeedsTenPairs_l412_412467


namespace factorial_calculation_l412_412041

theorem factorial_calculation :
  7! - 6 * 6! - 2 * 6! = -720 := by
  sorry

end factorial_calculation_l412_412041


namespace concyclic_H_E_N_N1_N2_l412_412259

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def nine_point_center (A B C : Point) : Point := sorry
noncomputable def altitude (A B C : Point) : Point := sorry
noncomputable def salmon_circle_center (A O O₁ O₂ : Point) : Point := sorry
noncomputable def foot_of_perpendicular (O' B C : Point) : Point := sorry
noncomputable def is_concyclic (points : List Point) : Prop := sorry

theorem concyclic_H_E_N_N1_N2 (A B C D : Point):
  let H := altitude A B C
  let O := circumcenter A B C
  let O₁ := circumcenter A B D
  let O₂ := circumcenter A C D
  let N := nine_point_center A B C
  let N₁ := nine_point_center A B D
  let N₂ := nine_point_center A C D
  let O' := salmon_circle_center A O O₁ O₂
  let E := foot_of_perpendicular O' B C
  is_concyclic [H, E, N, N₁, N₂] :=
sorry

end concyclic_H_E_N_N1_N2_l412_412259


namespace geometric_seq_sum_l412_412151

theorem geometric_seq_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, a_n < a_{n + 1}) → 
  (a_1 = 1) → 
  (a_3 = 4) → 
  S_6 = 63 := 
  by sorry

end geometric_seq_sum_l412_412151


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412790

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412790


namespace Q_on_bisector_of__R_on_circumcircle_ABC_l412_412617

variables (A B C E D P Q K L R : Type*)
variables [MetricSpace P] [MetricSpace Q]
variables (h1 : E ∈ segment B B)
variables (h2 : D ∈ segment A A)
variables (h_BE_CD : distance B E = distance C D)
variables (h_P_def : P = intersection (lineSegment B E) (lineSegment C D))
variables (h_Q_def : ∃ Q2 : Type*, Q2 ≠ P ∧ Q2 ∈ circumcircle (triangle C D P) ∧ Q2 ∈ circumcircle (triangle B E P))
variables (h_K : K = midpoint B E)
variables (h_L : L = midpoint C D)
variables (h_R : ∃ R2 : Type*, R = intersection (perpendicular (line Q K) (singlePoint K)) (perpendicular (line Q L) (singlePoint L)))

-- Defining triangle and other geometric entities
noncomputable def triangle (A B C : Type*) := ∃ (P : Type*), ∀ (X : Type*), X ∈ circumscribedCircle P

-- Geometry problem proofs
theorem Q_on_bisector_of_∠BAC :
  ∃ (X : Type*), angleBisector (angle A B C) (X Q) := 
sorry

theorem R_on_circumcircle_ABC :
  ∃ (X : Type*), X ∈ circumscribedCircle (triangle A B C) (R) := 
sorry

end Q_on_bisector_of__R_on_circumcircle_ABC_l412_412617


namespace find_x_l412_412878

noncomputable def orig : (ℝ × ℝ) := (3, -2)
noncomputable def dest : ℝ → (ℝ × ℝ) := λ x, (x, 8)
noncomputable def length : ℝ := 13

theorem find_x (x : ℝ) (h : dist orig (dest x) = length) : x = 3 - real.sqrt 69 ∨ x = 3 + real.sqrt 69 :=
sorry

end find_x_l412_412878


namespace staircase_perimeter_l412_412256

theorem staircase_perimeter (area : ℝ) (side_length : ℝ) (num_sides : ℕ) (right_angles : Prop) :
  area = 85 ∧ side_length = 1 ∧ num_sides = 10 ∧ right_angles → 
  ∃ perimeter : ℝ, perimeter = 30.5 :=
by
  intro h
  sorry

end staircase_perimeter_l412_412256


namespace distance_from_focus_to_line_l412_412715

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412715


namespace chess_tournament_max_N_l412_412573

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l412_412573


namespace number_of_satisfying_integers_l412_412343

theorem number_of_satisfying_integers :
  ∃! (n : ℕ), n ∈ {1, 2} ∧ ∀ n ∈ {1, 2}, 25 - 5 * n > 12 :=
sorry

end number_of_satisfying_integers_l412_412343


namespace necessary_and_sufficient_condition_l412_412519

theorem necessary_and_sufficient_condition (x : ℝ) :
  (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := sorry

end necessary_and_sufficient_condition_l412_412519


namespace minimal_area_quadrilateral_l412_412815

noncomputable def solve_least_area_quadrilateral : ℂ → Prop :=
by sorry

theorem minimal_area_quadrilateral :
  (∀ (z : ℂ), (z - 4)^12 = 16 → (solve_least_area_quadrilateral z)) →
  (∃ (area : ℝ), area = √3 + 7) :=
by
  sorry

end minimal_area_quadrilateral_l412_412815


namespace geometric_progression_coincides_arithmetic_l412_412023

variables (a d q : ℝ)
variables (ap : ℕ → ℝ) (gp : ℕ → ℝ)

-- Define the N-th term of the arithmetic progression
def nth_term_ap (n : ℕ) : ℝ := a + n * d

-- Define the N-th term of the geometric progression
def nth_term_gp (n : ℕ) : ℝ := a * q^n

theorem geometric_progression_coincides_arithmetic :
  gp 3 = ap 10 →
  gp 4 = ap 74 :=
by
  intro h
  sorry

end geometric_progression_coincides_arithmetic_l412_412023


namespace quadratic_completion_l412_412077

theorem quadratic_completion (n a : ℝ) (h1 : a = 2 * n) (h2 : n^2 + 3 = 27) (h3 : a > 0) :
  a = 4 * sqrt 6 :=
by
  sorry

end quadratic_completion_l412_412077


namespace volume_comparison_l412_412417

noncomputable def side_length := ℝ
noncomputable def radius := ℝ
noncomputable def height := ℝ

variables (a r R : ℝ)

-- Surface areas are given to be equal
axiom surface_areas_equal : 6 * a ^ 2 = 4 * Real.pi * r ^ 2 ∧ 6 * Real.pi * R ^ 2 = 4 * Real.pi * r ^ 2

-- Volumes of the respective shapes
noncomputable def V_cube := a ^ 3
noncomputable def V_sphere := (4 / 3) * Real.pi * r ^ 3
noncomputable def V_cylinder := 2 * Real.pi * R ^ 3

theorem volume_comparison (h : surface_areas_equal a r R) :
  V_cube a < V_cylinder R ∧ V_cylinder R < V_sphere r :=
sorry

end volume_comparison_l412_412417


namespace num_int_values_satisfying_inequality_l412_412983

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l412_412983


namespace distance_to_line_is_sqrt5_l412_412800

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412800


namespace S6_sum_geometric_sequence_l412_412148

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0 a q := a
| (n+1) a q := a * (1 - q^(n+1)) / (1 - q)

theorem S6_sum_geometric_sequence :
  ∃ (a1 a3 : ℝ) (q : ℝ),
    (a1 * a1 - 5 * a1 + 4 = 0) ∧
    (a3 * a3 - 5 * a3 + 4 = 0) ∧
    (a3 = a1 * q^2) ∧
    (0 < q) ∧
    (a1 = 1) ∧
    (a3 = 4) →
    geometric_sequence_sum 6 1 2 = 63 := sorry

end S6_sum_geometric_sequence_l412_412148


namespace obtuse_vertex_angle_is_135_l412_412245

-- Define the obtuse scalene triangle with the given properties
variables {a b c : ℝ} (triangle : Triangle ℝ)
variables (φ : ℝ) (h_obtuse : φ > 90 ∧ φ < 180) (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_side_relation : a^2 + b^2 = 2 * c^2) (h_sine_obtuse : Real.sin φ = Real.sqrt 2 / 2)

-- The measure of the obtuse vertex angle is 135 degrees
theorem obtuse_vertex_angle_is_135 :
  φ = 135 := by
  sorry

end obtuse_vertex_angle_is_135_l412_412245


namespace simplify_expr1_simplify_expr2_l412_412919

-- Defining the necessary variables as real numbers for the proof
variables (x y : ℝ)

-- Prove the first expression simplification
theorem simplify_expr1 : 
  (x + 2 * y) * (x - 2 * y) - x * (x + 3 * y) = -4 * y^2 - 3 * x * y :=
  sorry

-- Prove the second expression simplification
theorem simplify_expr2 : 
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4 * x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
  sorry

end simplify_expr1_simplify_expr2_l412_412919


namespace fish_distribution_l412_412266

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end fish_distribution_l412_412266


namespace multiplication_integer_multiple_l412_412318

theorem multiplication_integer_multiple (a b n : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
(h_eq : 10000 * a + b = n * (a * b)) : n = 73 := 
sorry

end multiplication_integer_multiple_l412_412318


namespace sum_of_projections_area_UVW_l412_412258

noncomputable def triangle_XYZ := { x : ℝ × ℝ // x.1 < 10 ∧ x.2 < 8 }

def XY : ℝ := 6
def XZ : ℝ := 10
def YZ : ℝ := 8

def centroid (triangle : { x : ℝ × ℝ // x.1 < 10 ∧ x.2 < 8 }) : ℝ × ℝ := sorry
def projection (point : ℝ × ℝ) (line : set (ℝ × ℝ)) : ℝ × ℝ := sorry

def T := centroid triangle_XYZ
def U := projection T (set_of (λ p, p.2 = 0))
def V := projection T (set_of (λ p, p.1 = 0))
def W := projection T (set_of (λ p, p.1 + p.2 = 14)) -- 14 from Pythagorean theorem for right triangle

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def TU := distance T U
def TV := distance T V
def TW := distance T W

-- First goal: Show that the sum of distances equals 6.27
theorem sum_of_projections :
  TU + TV + TW = 6.27 :=
sorry

-- Second goal: Show area of triangle UVW
theorem area_UVW :
  ∃ A_{UVW} : ℝ, A_{UVW} = sorry :=
sorry

end sum_of_projections_area_UVW_l412_412258


namespace subset_count_l412_412469

theorem subset_count : 
  ∀ (X Y : Set ℕ), X = {1, 2, 3, 4, 5, 6} → {1, 2, 3} ⊆ Y → Y ⊆ X → finset.card ( { y | y ∈ {1, 2, 3, 4, 5, 6} ∧ {1, 2, 3} ⊆ y } ) = 8 := by
  sorry

end subset_count_l412_412469


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412789

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412789


namespace f_le_x_l412_412631

variable (f : ℝ → ℝ)

def differentiable_on_R (f : ℝ → ℝ) : Prop := differentiable ℝ f

def condition1 (f : ℝ → ℝ) : Prop :=
  differentiable_on_R f

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ deriv f x

theorem f_le_x (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  ∀ x : ℝ, f x ≤ x :=
sorry

end f_le_x_l412_412631


namespace range_of_a_l412_412175

theorem range_of_a (a : ℝ) : 
  let p := (1 < a)
  let q := (-2 < a ∧ a < 2)
  (p ∨ q) ∧ ¬ (p ∧ q) ↔ a ∈ Icc (-2 : ℝ) 1 ∪ Icc (2 : ℝ) (⊤ : ℝ) :=
by {
  sorry
}

end range_of_a_l412_412175


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412792

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412792


namespace no_prime_divisible_by_45_l412_412183

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l412_412183


namespace ax_by_leq_one_l412_412533

theorem ax_by_leq_one (a b x y : ℝ) 
  (h₁ : a^2 + b^2 = 1) 
  (h₂ : x^2 + y^2 = 1) : 
  ax + by ≤ 1 :=
by 
  sorry

end ax_by_leq_one_l412_412533


namespace no_primes_divisible_by_45_l412_412193

theorem no_primes_divisible_by_45 : ∀ p : ℕ, prime p → ¬ (45 ∣ p) := 
begin
  sorry
end

end no_primes_divisible_by_45_l412_412193


namespace distance_to_line_is_sqrt5_l412_412796

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412796


namespace range_of_f_l412_412517

noncomputable def f (x b : ℝ) : ℝ := 3^(x - b)

theorem range_of_f :
  (∃ b : ℝ, (∀ x, 2 ≤ x ∧ x ≤ 4 → f x b = 3^(x - b) ∧ f 2 b = 1)
  → set.range (λ x, f x b) = set.Icc 1 9) :=
begin
  sorry
end

end range_of_f_l412_412517


namespace amount_paid_l412_412309

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l412_412309


namespace fish_distribution_l412_412265

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end fish_distribution_l412_412265


namespace max_value_of_function_l412_412142

variable (x y : ℝ)

theorem max_value_of_function 
  (h : x^2 + y^2 = 25) : 
  (∃ max_z : ℝ, max_z = 6 * Real.sqrt 10 ∧ 
   ∀ z : ℝ, z = Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) → z ≤ max_z) :=
begin
  sorry
end

end max_value_of_function_l412_412142


namespace f2_g2_eq_2016_l412_412539

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions and properties
axiom symmetry_f (x : ℝ) : f(2 - x) = f(x)
axiom symmetry_g (x : ℝ) : g(2 - x) + g(x) = -4
axiom func_eqn (x : ℝ) : f(x) + g(x) = 9^x + x^3 + 1

-- Proof goal
theorem f2_g2_eq_2016 : f(2) * g(2) = 2016 := by
  sorry

end f2_g2_eq_2016_l412_412539


namespace sum_of_digits_of_minimal_N_l412_412619

def N (n : ℕ) : Prop := n > 0 ∧ 30 ∣ n
def P (n : ℕ) := (n + 1 - Nat.ceil (n / 3)) / (n + 1)

theorem sum_of_digits_of_minimal_N :
  ∃ (n : ℕ), N n ∧ P n < 2 / 3 ∧ ∑ d in (Nat.digits 10 n), d = 9 :=
by
  sorry

end sum_of_digits_of_minimal_N_l412_412619


namespace range_of_transformed_sine_function_l412_412097

theorem range_of_transformed_sine_function :
  (∀ y, ∃ x, (0 < x ∧ x < 2 * Real.pi / 3) ∧ y = 2 * Real.sin (x + Real.pi / 6) - 1) ↔ (0 < y ∧ y ≤ 1) :=
sorry

end range_of_transformed_sine_function_l412_412097


namespace range_of_a_l412_412280

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x > (2 / 3), (deriv (f a)) x > 0) → a > -(1 / 9) :=
by
  sorry

end range_of_a_l412_412280


namespace no_real_solution_for_equation_l412_412332

theorem no_real_solution_for_equation :
  ∀ x : ℝ, (x ≠ 1) → ¬ (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) :=
by
  intro x h,
  sorry

end no_real_solution_for_equation_l412_412332


namespace min_perimeter_l412_412022

theorem min_perimeter (a b : ℕ) (h1 : b = 3 * a) (h2 : 3 * a + 8 * a = 11) (h3 : 2 * a + 12 * a = 14)
  : 2 * (15 + 11) = 52 := 
sorry

end min_perimeter_l412_412022


namespace complement_intersection_l412_412552

noncomputable def real_universal_set : Set ℝ := Set.univ

noncomputable def set_A (x : ℝ) : Prop := x + 1 < 0
def A : Set ℝ := {x | set_A x}

noncomputable def set_B (x : ℝ) : Prop := x - 3 < 0
def B : Set ℝ := {x | set_B x}

noncomputable def complement_A : Set ℝ := {x | ¬set_A x}

noncomputable def intersection (S₁ S₂ : Set ℝ) : Set ℝ := {x | x ∈ S₁ ∧ x ∈ S₂}

theorem complement_intersection :
  intersection complement_A B = {x | -1 ≤ x ∧ x < 3} :=
sorry

end complement_intersection_l412_412552


namespace toothpick_grid_l412_412064

theorem toothpick_grid (l w : ℕ) (h_l : l = 45) (h_w : w = 25) :
  let effective_vertical_lines := l + 1 - (l + 1) / 5
  let effective_horizontal_lines := w + 1 - (w + 1) / 5
  let vertical_toothpicks := effective_vertical_lines * w
  let horizontal_toothpicks := effective_horizontal_lines * l
  let total_toothpicks := vertical_toothpicks + horizontal_toothpicks
  total_toothpicks = 1722 :=
by {
  sorry
}

end toothpick_grid_l412_412064


namespace range_of_x_l412_412074

def op (a b : ℝ) : ℝ :=
  if a > b then a * b + b else a * b - b

theorem range_of_x (x : ℝ) :
  op 3 (x + 2) > 0 → (-2 < x ∧ x < 1) ∨ (x > 1) :=
by
  sorry

end range_of_x_l412_412074


namespace distance_from_focus_to_line_l412_412729

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412729


namespace stockholm_uppsala_distance_l412_412683

variable (map_distance : ℝ) (scale_factor : ℝ)

def actual_distance (d : ℝ) (s : ℝ) : ℝ := d * s

theorem stockholm_uppsala_distance :
  actual_distance 65 20 = 1300 := by
  sorry

end stockholm_uppsala_distance_l412_412683


namespace find_unit_prices_max_type_A_books_l412_412870

-- Define the unit prices of type A and type B books
variables {A B : ℝ}

-- Define the number of type A and type B books to be purchased
variables {x y : ℝ}

-- The conditions given in the problem
def condition1 : Prop := A = B + 10
def condition2 : Prop := 3 * A + 2 * B = 130
def condition3 : Prop := x + y = 40
def condition4 : Prop := 30 * x + 20 * y ≤ 980

-- Problem statements to be proved
theorem find_unit_prices (h1 : condition1) (h2 : condition2) : A = 30 ∧ B = 20 :=
by
  sorry

theorem max_type_A_books (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : x ≤ 18 :=
by
  sorry

end find_unit_prices_max_type_A_books_l412_412870


namespace exists_theta_for_polynomial_sin_l412_412005

variable {a0 a1 a2 a3 an a θ : ℝ}

theorem exists_theta_for_polynomial_sin
  (f : ℝ → ℝ)
  (hf : f = λ x, a0 + a1 * x + a2 * x^2 + a3 * x^3 + an * x^n)
  (hθ : 0 < θ ∧ θ < π / 2) :
  ∃ θ ∈ (0:ℝ, π / 2),
    f (sin θ) = 
    a0 + a1 / 2 + a2 / 3 + a3 / 4 + an / (n + 1) :=
sorry

end exists_theta_for_polynomial_sin_l412_412005


namespace area_quadrilateral_is_correct_l412_412252

-- Circle equation as a given condition
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 2 = 0

-- All given point and lengths based on the problem statement
def E := (0, 1) -- Point E
def center := (2, 2) -- Center of the circle
def radius := Real.sqrt 10 -- Radius of the circle
def ac := 2 * Real.sqrt 10 -- Length of AC (diameter)
def bd := 2 * Real.sqrt 5 -- Length of BD (shortest chord through E perpendicular to AC)

-- The geometric conditions
axiom E_on_circle : circle_eq (E.1) (E.2)
axiom ac_diameter : True -- modeling the condition that AC is the diameter through E
axiom bd_perpendicular_ac : True -- modeling the perpendicular condition

-- The area of quadrilateral ABCD is computed
def area_ABCD : ℝ := (1/2) * ac * bd

theorem area_quadrilateral_is_correct :
  area_ABCD = 10 * Real.sqrt 2 :=
by sorry

end area_quadrilateral_is_correct_l412_412252


namespace peach_trees_count_l412_412254

-- Define the conditions
def apple_trees : ℕ := 30
def apples_per_tree : ℕ := 150
def peach_weight_per_tree : ℕ := 65
def total_fruit_mass : ℕ := 7425

-- Calculate total apples mass
def total_apples_mass := apple_trees * apples_per_tree

-- Define the total mass equation
def mass_equation (P : ℕ) : Prop := total_apples_mass + P * peach_weight_per_tree = total_fruit_mass

-- Prove the number of peach trees is 45
theorem peach_trees_count : ∃ P : ℕ, mass_equation P ∧ P = 45 :=
by
  existsi 45
  rw [mass_equation]
  split
  sorry

end peach_trees_count_l412_412254


namespace sqrt_ineq_l412_412325

open Real

theorem sqrt_ineq (α β : ℝ) (hα : 1 ≤ α) (hβ : 1 ≤ β) :
  Int.floor (sqrt α) + Int.floor (sqrt (α + β)) + Int.floor (sqrt β) ≥
    Int.floor (sqrt (2 * α)) + Int.floor (sqrt (2 * β)) := by sorry

end sqrt_ineq_l412_412325


namespace find_n_l412_412880

noncomputable def p (x : ℕ) (n : ℕ) : ℕ := sorry

theorem find_n (n : ℕ) (p : ℕ → ℕ → ℕ)
  (h₀ : ∀ k : ℕ, k ≤ n → p (3 * k) n = 2)
  (h1 : ∀ k : ℕ, k < n → p (3 * k + 1) n = 1)
  (h2 : ∀ k : ℕ, k < n → p (3 * k + 2) n = 0)
  (h3 : p (3 * n + 1) n = 730) :
  n = 4 :=
begin
  sorry
end

end find_n_l412_412880


namespace problem_statement_l412_412616

theorem problem_statement (a a1 a2 a3 a4 a5 : ℤ) :
  (2 * (-1) - 1)^5 = -243 → 
  (2 * x - 1)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  a - a1 + a2 - a3 + a4 - a5 = -243 :=
by
  assume h1 h2
  sorry

end problem_statement_l412_412616


namespace matrix_not_invertible_l412_412474

def is_not_invertible_matrix (y : ℝ) : Prop :=
  let a := 2 + y
  let b := 9
  let c := 4 - y
  let d := 10
  a * d - b * c = 0

theorem matrix_not_invertible (y : ℝ) : is_not_invertible_matrix y ↔ y = 16 / 19 :=
  sorry

end matrix_not_invertible_l412_412474


namespace complex_number_pow_l412_412232

theorem complex_number_pow {z : ℂ} (h : (1 + z) / (1 - z) = complex.I) : z^2023 = -complex.I := 
sorry

end complex_number_pow_l412_412232


namespace sum_v2_to_v7_correct_l412_412831

-- Define initial vectors
def v0 : ℝ × ℝ := ⟨2, 1⟩
def w0 : ℝ × ℝ := ⟨1, -1⟩

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector projection
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product u v) / (dot_product u u)
  in ⟨scalar * u.1, scalar * u.2⟩

-- Define the sequences vn and wn recursively
noncomputable def vn : ℕ → ℝ × ℝ
| 0       := v0
| 1       := proj v0 w0
| (n + 2) := proj v0 (proj w0 (vn n))

noncomputable def wn : ℕ → ℝ × ℝ
| 0       := w0
| (n + 1) := proj w0 (vn n)

-- Sum of vn from v2 to v7
noncomputable def sum_v2_to_v7 : ℝ × ℝ :=
  vn 2 + vn 3 + vn 4 + vn 5 + vn 6 + vn 7

theorem sum_v2_to_v7_correct :
  sum_v2_to_v7 = ⟨4 / 4950, 2 / 4950⟩ :=
sorry

end sum_v2_to_v7_correct_l412_412831


namespace problem_1_problem_2_l412_412917

-- Problem 1: Prove that (1) - 32 - (+11) + (-9) - (-16) = -36
theorem problem_1 : (1 - 32 - (+11) + (-9) - (-16) = -36) := sorry

-- Problem 2: Prove that -(1^4) - |0 - 1| * 2 - (-3)^2 / (-3/2) = 3
theorem problem_2 : (-(1^4) - abs(0 - 1) * 2 - (-3)^2 / (-3 / 2) = 3) := sorry

end problem_1_problem_2_l412_412917


namespace coeff_x2y2_eq_18_l412_412341

noncomputable def binomial_expansion_coefficient_x2y2 : ℕ :=
  (1 + 3 * x + 3 * x^2 + x^3) * (1 + 4 * y + 6 * y^2 + 4 * y^3 + y^4)

theorem coeff_x2y2_eq_18 : 
  binomial_expansion_coefficient_x2y2.coeff 2 2 = 18 := 
by 
  sorry

end coeff_x2y2_eq_18_l412_412341


namespace green_peaches_per_basket_l412_412363

theorem green_peaches_per_basket :
  ∀ (baskets : ℕ) (red_peaches_per_basket : ℕ) (total_peaches : ℕ) (green_peaches_per_basket : ℕ),
  baskets = 11 →
  red_peaches_per_basket = 10 →
  total_peaches = 308 →
  green_peaches_per_basket * baskets = total_peaches - (baskets * red_peaches_per_basket) →
  green_peaches_per_basket = 18 :=
by {
  intros baskets red_peaches_per_basket total_peaches green_peaches_per_basket,
  assume h1 h2 h3 h4,
  sorry
}

end green_peaches_per_basket_l412_412363


namespace greatest_prime_dividing_conditions_l412_412964

theorem greatest_prime_dividing_conditions (P : ℕ) (hp : P.prime) (h1 : P ∣ 1247) (h2 : P ∣ 1479) : P = 1 := 
sorry

end greatest_prime_dividing_conditions_l412_412964


namespace calculate_a3_b3_l412_412048

theorem calculate_a3_b3 (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by
  sorry

end calculate_a3_b3_l412_412048


namespace train_time_to_pass_bridge_approx_l412_412404

noncomputable def time_to_pass_bridge (length_train length_bridge : ℕ) (speed_kmh : ℝ) : ℝ :=
  let distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  distance / speed_ms

theorem train_time_to_pass_bridge_approx :
  time_to_pass_bridge 240 130 50 ≈ 26.64 :=
by
  sorry

end train_time_to_pass_bridge_approx_l412_412404


namespace blood_expiration_date_l412_412344

theorem blood_expiration_date :
  (10.factorial : ℕ) = 3628800 →
  let start_date := "January 1st"
  let start_time := 12 * 60 * 60 -- seconds since midnight
  let effective_period := 3628800 -- seconds
  let expiration_time := start_time + effective_period in
  expiration_time = 3628800 + 12 * 60 * 60 →
  expiration_date = "February 12th" :=
begin
  sorry
end

end blood_expiration_date_l412_412344


namespace max_possible_N_l412_412577

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l412_412577


namespace total_leaves_correct_l412_412035

-- Definitions based on conditions
def basil_pots := 3
def rosemary_pots := 9
def thyme_pots := 6

def basil_leaves_per_pot := 4
def rosemary_leaves_per_pot := 18
def thyme_leaves_per_pot := 30

-- Calculate the total number of leaves
def total_leaves : Nat :=
  (basil_pots * basil_leaves_per_pot) +
  (rosemary_pots * rosemary_leaves_per_pot) +
  (thyme_pots * thyme_leaves_per_pot)

-- The statement to prove
theorem total_leaves_correct : total_leaves = 354 := by
  sorry

end total_leaves_correct_l412_412035


namespace find_a_l412_412512

theorem find_a (a : ℝ) (h : 1 ∈ ({a + 2, (a + 1)^2, a^2 + 3a + 3} : set ℝ)) : a = 0 :=
sorry

end find_a_l412_412512


namespace problem_equivalent_proof_l412_412049

theorem problem_equivalent_proof :
  (real.sqrt 7 - 1) ^ 2 - (real.sqrt 14 - real.sqrt 2) * (real.sqrt 14 + real.sqrt 2) = -4 - 2 * real.sqrt 7 := 
sorry

end problem_equivalent_proof_l412_412049


namespace cos_triple_angle_l412_412206

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l412_412206


namespace threeQuantitiesChange_l412_412657

structure Triangle (A B P : Type) [MetricSpace P] :=
(A B : P)
(movingP : P → Prop)

def isMidpoint (P1 P2 M : Type) [MetricSpace P1] : Prop :=
  dist P1 M = dist P2 M

def verticalMovement {P : Type} [MetricSpace P] (P A B : P → Prop) : Prop :=
  (movingP P) ∧ (intersects A B P)

def unchangedLengthMN {P A B M N : Type} [MetricSpace P] : Prop :=
  ∀ P, verticalMovement P A B → isMidpoint P A M ∧ isMidpoint P B N → dist M N = dist (A + B) / 2

def changesPerimeter {P A B : Type} [MetricSpace P] : Prop :=
  ∀ P, verticalMovement P A B → (dist P A + dist A B + dist P B) ≠ const

def changesAreaTriangle {P A B : Type} [MetricSpace P] : Prop :=
  ∀ P, verticalMovement P A B → (1/2 * dist A B * height P A B) ≠ const

def changesAreaTrapezoid {P A B M N : Type} [MetricSpace P] : Prop :=
  ∀ P, verticalMovement P A B → isMidpoint P A M ∧ isMidpoint P B N → (1/2 * (dist A B + dist M N) * height M N P) ≠ const

theorem threeQuantitiesChange {P A B M N : Type} [MetricSpace P] :
  (unchangedLengthMN P A B M N) →
  (changesPerimeter P A B) →
  (changesAreaTriangle P A B) →
  (changesAreaTrapezoid P A B M N) →
  3 = (true_count [unchangedLengthMN, changesPerimeter, changesAreaTriangle, changesAreaTrapezoid])
:= sorry

end threeQuantitiesChange_l412_412657


namespace unique_solution_l412_412091

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (3 * 2^y - 1 = 2^x + 2^(-x)) ∧
  (3 * 2^z - 1 = 2^y + 2^(-y)) ∧
  (3 * 2^x - 1 = 2^z + 2^(-z))

theorem unique_solution : ∀ (x y z : ℝ), system_of_equations x y z → (x = 0 ∧ y = 0 ∧ z = 0) :=
  by
  sorry

end unique_solution_l412_412091


namespace quadrant_of_complex_number_l412_412157

def complex_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "I"
  else if z.re < 0 ∧ z.im > 0 then "II"
  else if z.re < 0 ∧ z.im < 0 then "III"
  else if z.re > 0 ∧ z.im < 0 then "IV"
  else "Origin or Axis"

theorem quadrant_of_complex_number : complex_quadrant (-1 + 2 * complex.i) = "II" :=
by
  -- Proof goes here
  sorry

end quadrant_of_complex_number_l412_412157


namespace a2020_lt_5_l412_412923

def sequence_a : ℕ → ℝ
| 0       := 1
| (n + 1) := (sequence_a n * sequence_b n + sequence_a n + 1) / sequence_b n
and sequence_b : ℕ → ℝ
| 0       := 2
| (n + 1) := (sequence_a n * sequence_b n + sequence_b n + 1) / sequence_a n

theorem a2020_lt_5 : sequence_a 2020 < 5 := 
by sorry

end a2020_lt_5_l412_412923


namespace gcd_power_diff_l412_412833

theorem gcd_power_diff (m n : ℕ) (h1 : m = 2^2021 - 1) (h2 : n = 2^2000 - 1) :
  Nat.gcd m n = 2097151 :=
by sorry

end gcd_power_diff_l412_412833


namespace parallelogram_area_ratio_l412_412861

variables {A B C D E : Type} [AddCommGroup A] [AffineSpace B A] [AddTorsor A B]

def is_parallelogram (AB AC AD : A) := (AB + AC = AD) ∧ (AC + AD = AB)
def is_midpoint (C D E : A) := (E = (C + D) / 2)
def area_ratio := 4

theorem parallelogram_area_ratio
  (A B C D E : B)
  (par : is_parallelogram (B - A) (D - A) (C - A))
  (midpoint : is_midpoint (C - A) (D - A) (E - A)) :
  (ratio_of_areas (triangle_area A D E) (parallelogram_area A B C D) = 1 / 4) :=
sorry

end parallelogram_area_ratio_l412_412861


namespace middle_tree_less_half_tallest_tree_l412_412360

theorem middle_tree_less_half_tallest_tree (T M S : ℝ)
  (hT : T = 108)
  (hS : S = 1/4 * M)
  (hS_12 : S = 12) :
  (1/2 * T) - M = 6 := 
sorry

end middle_tree_less_half_tallest_tree_l412_412360


namespace find_lcm_of_two_numbers_l412_412406

theorem find_lcm_of_two_numbers (A B : ℕ) (hcf : ℕ) (prod : ℕ) 
  (h1 : hcf = 22) (h2 : prod = 62216) (h3 : A * B = prod) (h4 : Nat.gcd A B = hcf) :
  Nat.lcm A B = 2828 := 
by
  sorry

end find_lcm_of_two_numbers_l412_412406


namespace sqrt_difference_eq_neg_six_sqrt_two_l412_412464

theorem sqrt_difference_eq_neg_six_sqrt_two :
  (Real.sqrt ((5 - 3 * Real.sqrt 2)^2)) - (Real.sqrt ((5 + 3 * Real.sqrt 2)^2)) = -6 * Real.sqrt 2 := 
sorry

end sqrt_difference_eq_neg_six_sqrt_two_l412_412464


namespace range_B_range_expr_l412_412244

variables (A B C a b c : ℝ)

-- Assuming the angle A, B, C are angles in important of only acute triangle ≤ π/2.
variable [dec_trivial (0 < B ∧ B < π/2)]

-- Assuming for all sides base a, b, and c for a stated sides in specified acute triangles.
variables (h1: 0 < a ∧ 0 < b ∧ 0 < c)
-- Assuming square difference to equal specific known terms.
variables (h2: a^2 - b^2 = b * c)

theorem range_B (h₁: 0 < A ∧ A = 2 * B) : π / 6 < B ∧ B < π / 4 := sorry

theorem range_expr : 2 * sqrt(30) ≤ 5 / tan B - 5 / tan A + 6 * sin A := sorry

end range_B_range_expr_l412_412244


namespace distance_from_focus_to_line_l412_412731

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412731


namespace complex_number_on_real_axis_l412_412292

theorem complex_number_on_real_axis (a : ℝ) :
  (∀ z : ℂ, z = (1 + complex.I) * (a + complex.I) → z.im = 0) → a = -1 :=
by
  intro hz
  have h := hz ((1 : ℂ) + complex.I) * (a : ℂ) + complex.I)
  simp [complex.add_im, complex.mul_im, complex.I_im, complex.one_im] at h
  sorry

end complex_number_on_real_axis_l412_412292


namespace freight_capacity_equation_l412_412869

theorem freight_capacity_equation
  (x : ℝ)
  (h1 : ∀ (capacity_large capacity_small : ℝ), capacity_large = capacity_small + 4)
  (h2 : ∀ (n_large n_small : ℕ), (n_large : ℝ) = 80 / (x + 4) ∧ (n_small : ℝ) = 60 / x → n_large = n_small) :
  (80 / (x + 4)) = (60 / x) :=
by
  sorry

end freight_capacity_equation_l412_412869


namespace complex_number_solution_l412_412079

noncomputable def abs (z : ℂ) := z.re^2 + z.im^2

theorem complex_number_solution :
  ∃ z : ℂ, (3 * z - 4 * complex.I * conj z = -5 + 4 * complex.I) ∧ (z = -1/7 + 8/7 * complex.I) := by
    sorry

end complex_number_solution_l412_412079


namespace frost_cakes_total_l412_412462

-- Conditions
def Cagney_time := 60 -- seconds per cake
def Lacey_time := 40  -- seconds per cake
def total_time := 10 * 60 -- 10 minutes in seconds

-- The theorem to prove
theorem frost_cakes_total (Cagney_time Lacey_time total_time : ℕ) (h1 : Cagney_time = 60) (h2 : Lacey_time = 40) (h3 : total_time = 600):
  (total_time / (Cagney_time * Lacey_time / (Cagney_time + Lacey_time))) = 25 :=
by
  -- Proof to be filled in
  sorry

end frost_cakes_total_l412_412462


namespace theater_seats_l412_412889

theorem theater_seats (t : ℕ) (h1 : 50 + (0.4 * t) + (0.5 * t) = t) : t = 500 := 
sorry

end theater_seats_l412_412889


namespace distance_from_hyperbola_focus_to_line_l412_412759

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412759


namespace sequence_sum_problem_l412_412040

theorem sequence_sum_problem :
  let seq := [72, 76, 80, 84, 88, 92, 96, 100, 104, 108]
  3 * (seq.sum) = 2700 :=
by
  sorry

end sequence_sum_problem_l412_412040


namespace find_hypotenuse_l412_412584

-- Define the conditions
def right_triangle_ratio_3_2 (α β : ℝ) : Prop :=
  α / β = 3 / 2 ∧ α + β = 90

def side_opposite_angle (α side : ℝ) : Prop :=
  sin α = side / hypotenuse

-- Prove the hypotenuse length
theorem find_hypotenuse (α β : ℝ) (hypotenuse : ℝ) (side : ℝ) 
  (h_ratio: right_triangle_ratio_3_2 α β) (h_side: side_opposite_angle α side) 
  (h_side_val: side = 6) : hypotenuse = 6 / sin 36 :=
sorry

end find_hypotenuse_l412_412584


namespace first_shaded_square_each_column_l412_412435

/-- A rectangular board with 10 columns, numbered starting from 
    1 to the nth square left-to-right and top-to-bottom. The student shades squares 
    that are perfect squares. Prove that the first shaded square ensuring there's at least 
    one shaded square in each of the 10 columns is 400. -/
theorem first_shaded_square_each_column : 
  (∃ n, (∀ k, 1 ≤ k ∧ k ≤ 10 → ∃ m, m^2 ≡ k [MOD 10] ∧ m^2 ≤ n) ∧ n = 400) :=
sorry

end first_shaded_square_each_column_l412_412435


namespace liquid_X_percent_in_mixed_solution_l412_412461

theorem liquid_X_percent_in_mixed_solution (wP wQ : ℝ) (xP xQ : ℝ) (mP mQ : ℝ) :
  xP = 0.005 * wP →
  xQ = 0.015 * wQ →
  wP = 200 →
  wQ = 800 →
  13 / 1000 * 100 = 1.3 :=
by
  intros h1 h2 h3 h4
  sorry

end liquid_X_percent_in_mixed_solution_l412_412461


namespace find_c_l412_412825

theorem find_c (a b c : ℤ) (N : ℤ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) (h4 : N ≠ 1) :
    (N^(1 / (a : ℚ)) * N^(1 / (b * a : ℚ)) * N^(1 / (c * b * a : ℚ))) = N^(5 / 8 : ℚ) → c = 8 :=
  sorry

end find_c_l412_412825


namespace pizza_slices_l412_412007

theorem pizza_slices (n : ℕ) (total_slices : ℕ) (cheese_only : ℕ) (mushroom_slices : ℕ)
  (h_total : total_slices = 10)
  (h_cheese_only : cheese_only = 5)
  (h_mushroom : mushroom_slices = 7)
  (h_every_slice_has_topping : ∀ i < total_slices, i ∉ (finset.range total_slices).filter (λ s, s ∉ cheese_only ∧ s ∉ mushroom_slices)) :
  ∃ n_slices, n_slices = 2 :=
by
  use 2
  sorry

end pizza_slices_l412_412007


namespace distance_from_focus_to_line_l412_412711

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412711


namespace evaluate_expression_when_c_is_4_l412_412955

variable (c : ℕ)

theorem evaluate_expression_when_c_is_4 : (c = 4) → ((c^2 - c! * (c - 1)^c)^2 = 3715584) :=
by
  -- This is where the proof would go, but we only need to set up the statement.
  sorry

end evaluate_expression_when_c_is_4_l412_412955


namespace cos_five_pi_over_four_l412_412088

theorem cos_five_pi_over_four : Real.cos (5 * Real.pi / 4) = -1 / Real.sqrt 2 := 
by
  sorry

end cos_five_pi_over_four_l412_412088


namespace visitors_on_previous_day_l412_412896

theorem visitors_on_previous_day (total_visitors : ℕ) (current_day_visitors : ℕ) (h1 : total_visitors = 406) (h2 : current_day_visitors = 132) :
  total_visitors - current_day_visitors = 274 := by
  rw [h1, h2]
  norm_num
  done

end visitors_on_previous_day_l412_412896


namespace base_area_of_sand_pile_l412_412421

theorem base_area_of_sand_pile :
  let length := 6
  let width := 1.5
  let height := 3
  let cone_height := 2
  let volume_rect := length * width * height
  let volume_cone := (1 / 3) * Math.pi * (r^2) * cone_height
  (volume_rect = volume_cone) → π * (r^2) = (81 / 2) :=
by
  sorry

end base_area_of_sand_pile_l412_412421


namespace geom_seq_general_term_sum_first_n_terms_l412_412124

noncomputable def geometric_seq (n : ℕ) : ℕ := 3^(n-1)

def seq_a3 := 9
def seq_diff := 24

theorem geom_seq_general_term :
  (a3 = 9) ∧ (a4 - a2 = 24) → ∀ n : ℕ, a_n = 3^(n-1) :=
by sorry

theorem sum_first_n_terms (n : ℕ) :
  let b := λ (n : ℕ), n * geometric_seq n
  S_n = ∑ i in range n, b i → S_n = (2*n - 1) * 3^n + 1 / 4 :=
by sorry

end geom_seq_general_term_sum_first_n_terms_l412_412124


namespace no_primes_divisible_by_45_l412_412191

theorem no_primes_divisible_by_45 : ∀ p : ℕ, prime p → ¬ (45 ∣ p) := 
begin
  sorry
end

end no_primes_divisible_by_45_l412_412191


namespace largest_of_five_consecutive_ints_15120_l412_412499

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end largest_of_five_consecutive_ints_15120_l412_412499


namespace unused_sector_angle_l412_412993

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)
noncomputable def central_angle (r base_circumference : ℝ) : ℝ := (base_circumference / (2 * Real.pi * r)) * 360
noncomputable def unused_angle (total_degrees used_angle : ℝ) : ℝ := total_degrees - used_angle

theorem unused_sector_angle (R : ℝ)
  (cone_radius := 15)
  (cone_volume := 675 * Real.pi)
  (total_circumference := 2 * Real.pi * R)
  (cone_height := 9)
  (slant_height := Real.sqrt (cone_radius^2 + cone_height^2))
  (base_circumference := 2 * Real.pi * cone_radius)
  (used_angle := central_angle slant_height base_circumference) :

  unused_angle 360 used_angle = 164.66 := by
  sorry

end unused_sector_angle_l412_412993


namespace minimum_possible_value_of_box_l412_412560

theorem minimum_possible_value_of_box :
  ∃ (a b : ℤ), a * b = 30 ∧ (min (({ (a, b) : ℤ × ℤ | a * b = 30 }.image (λ (pair : ℤ × ℤ), pair.fst^2 + pair.snd^2)).to_finset) = 61) ∧ ∃ (x : ℤ), ∃ (y : ℤ), ∃ (z : ℤ), x ≠ y ∧ y ≠ z ∧ z ≠ x :=
by
  sorry

end minimum_possible_value_of_box_l412_412560


namespace find_a_l412_412505

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l412_412505


namespace distance_hyperbola_focus_to_line_l412_412775

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412775


namespace sin_double_angle_sum_zero_l412_412979

theorem sin_double_angle_sum_zero
  (α β : ℝ)
  (h : sin α * sin β + cos α * cos β = 0) :
  sin (2 * α) + sin (2 * β) = 0 :=
sorry

end sin_double_angle_sum_zero_l412_412979


namespace max_value_fraction_diff_l412_412532

noncomputable def max_fraction_diff (a b : ℝ) : ℝ :=
  1 / a - 1 / b

theorem max_value_fraction_diff (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : 4 * a - b ≥ 2) :
  max_fraction_diff a b ≤ 1 / 2 :=
by
  sorry

end max_value_fraction_diff_l412_412532


namespace distance_from_focus_to_line_l412_412746

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412746


namespace billy_feed_days_l412_412908

theorem billy_feed_days :
  ∀ (horses : ℕ) (oats_per_feeding : ℕ) (feedings_per_day : ℕ) (total_oats : ℕ)
    (daily_oats_per_horse := oats_per_feeding * feedings_per_day)
    (daily_oats_all_horses := daily_oats_per_horse * horses),
    horses = 4 →
    oats_per_feeding = 4 →
    feedings_per_day = 2 →
    total_oats = 96 →
    total_oats / daily_oats_all_horses = 3 :=
by
  intros horses oats_per_feeding feedings_per_day total_oats daily_oats_per_horse daily_oats_all_horses
  assume h_horses : horses = 4
  assume h_oats_per_feeding : oats_per_feeding = 4
  assume h_feedings_per_day : feedings_per_day = 2
  assume h_total_oats : total_oats = 96
  rw [h_horses, h_oats_per_feeding, h_feedings_per_day, h_total_oats]
  simp [daily_oats_per_horse, daily_oats_all_horses]
  sorry

end billy_feed_days_l412_412908


namespace distance_from_focus_to_line_l412_412694

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412694


namespace feathers_to_cars_ratio_l412_412681

theorem feathers_to_cars_ratio (initial_feathers : ℕ) (final_feathers : ℕ) (cars_dodged : ℕ)
  (h₁ : initial_feathers = 5263) (h₂ : final_feathers = 5217) (h₃ : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 :=
by
  sorry

end feathers_to_cars_ratio_l412_412681


namespace initial_customers_l412_412893

variable (n : ℕ) -- initial number of customers
variable (left : ℕ) -- number of customers who left
variable (tables : ℕ) -- number of tables
variable (people_per_table : ℕ) -- people per table
variable (remaining : ℕ) -- remaining people after some left

-- conditions
def condition1 := left = 12
def condition2 := people_per_table = 8
def condition3 := tables = 4
def condition4 := remaining = tables * people_per_table

-- proof goal
theorem initial_customers (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : n = remaining + left := sorry

end initial_customers_l412_412893


namespace largest_integer_n_l412_412072

-- Define the floor function
def floor (x : ℝ) : ℤ := Int.ofNat ⌊x⌋

-- Define the problem statement
theorem largest_integer_n (n : ℕ) (h : floor (Real.sqrt n) = 5) : n ≤ 35 :=
by
  sorry

end largest_integer_n_l412_412072


namespace Emily_average_speed_l412_412938

noncomputable def Emily_run_distance : ℝ := 10

noncomputable def speed_first_uphill : ℝ := 4
noncomputable def distance_first_uphill : ℝ := 2

noncomputable def speed_first_downhill : ℝ := 6
noncomputable def distance_first_downhill : ℝ := 1

noncomputable def speed_flat_ground : ℝ := 5
noncomputable def distance_flat_ground : ℝ := 3

noncomputable def speed_second_uphill : ℝ := 4.5
noncomputable def distance_second_uphill : ℝ := 2

noncomputable def speed_second_downhill : ℝ := 6
noncomputable def distance_second_downhill : ℝ := 2

noncomputable def break_first : ℝ := 5 / 60
noncomputable def break_second : ℝ := 7 / 60
noncomputable def break_third : ℝ := 3 / 60

noncomputable def time_first_uphill : ℝ := distance_first_uphill / speed_first_uphill
noncomputable def time_first_downhill : ℝ := distance_first_downhill / speed_first_downhill
noncomputable def time_flat_ground : ℝ := distance_flat_ground / speed_flat_ground
noncomputable def time_second_uphill : ℝ := distance_second_uphill / speed_second_uphill
noncomputable def time_second_downhill : ℝ := distance_second_downhill / speed_second_downhill

noncomputable def total_running_time : ℝ := time_first_uphill + time_first_downhill + time_flat_ground + time_second_uphill + time_second_downhill
noncomputable def total_break_time : ℝ := break_first + break_second + break_third
noncomputable def total_time : ℝ := total_running_time + total_break_time

noncomputable def average_speed : ℝ := Emily_run_distance / total_time

theorem Emily_average_speed : abs (average_speed - 4.36) < 0.01 := by
  sorry

end Emily_average_speed_l412_412938


namespace solve_equation_l412_412676

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^2 - (x + 3) * (x - 3) = 4 * x - 1 ∧ x = 7 / 4 := 
by
  sorry

end solve_equation_l412_412676


namespace problem_statement_l412_412622

-- Defining basic entities
variables (a b : Line) (α β : Plane)

-- Given conditions
variables (h1 : a ∉ α) (h2 : a ∉ β)
variables (h3 : a ⊥ β) (h4 : b ∥ a)
variables (h5 : α ⊥ β)

-- Statement to be proved
theorem problem_statement : ¬ (b ∥ α) :=
sorry

end problem_statement_l412_412622


namespace function_even_iff_a_eq_one_l412_412233

theorem function_even_iff_a_eq_one (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = a * (3^x) + 1/(3^x)) → 
  (∀ x : ℝ, f x = f (-x)) ↔ a = 1 :=
by
  sorry

end function_even_iff_a_eq_one_l412_412233


namespace geometric_properties_of_triangle_with_altitudes_l412_412337

-- Definitions related to the geometric configurations
variables {A B C H : Type} [Triangle ABC AH BH CH] [Orthocenter H ABC]

-- Conditions derived from problem statements
def common_nine_point_circle :=
  ∃ (N : Circle), nine_point_circle N ABC ∧ nine_point_circle N HBC ∧
  nine_point_circle N AHC ∧ nine_point_circle N ABH

-- Shared property of Euler lines
def euler_lines_intersect :=
  ∃ (P : Point), euler_line_intersects_at P ABC ∧ euler_line_intersects_at P HBC ∧
  euler_line_intersects_at P AHC ∧ euler_line_intersects_at P ABH

-- Symmetric quadrilateral formation property
def circumcenter_symmetry :=
  let O := circumcenter ABC in
  let O1 := circumcenter HBC in
  let O2 := circumcenter AHC in
  let O3 := circumcenter ABH in
  symmetric_quadrilateral O O1 O2 O3 H ABC

-- Main theorem statement
theorem geometric_properties_of_triangle_with_altitudes :
  common_nine_point_circle ∧ euler_lines_intersect ∧ circumcenter_symmetry :=
by
  sorry

end geometric_properties_of_triangle_with_altitudes_l412_412337


namespace no_prime_divisible_by_45_l412_412184

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l412_412184


namespace valid_functions_l412_412094

theorem valid_functions (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) * g (x - y) = (g x + g y)^2 - 4 * x^2 * g y + 2 * y^2 * g x) :
  (∀ x, g x = 0) ∨ (∀ x, g x = x^2) :=
by sorry

end valid_functions_l412_412094


namespace find_number_chosen_l412_412849

theorem find_number_chosen (x : ℤ) (h : 4 * x - 138 = 102) : x = 60 := sorry

end find_number_chosen_l412_412849


namespace cos_triple_angle_l412_412218

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l412_412218


namespace angle_HAB_eq_3_angle_GAB_l412_412457

variables (A B C D E F G H : Type) [ordered_ring A] [ordered_ring B] [ordered_ring C] [ordered_ring D] 
[midpoint A D E] [midpoint B C F] (rectangle : rectangle A B C D)
(on_EF : lies_on_line G F E) (symmetric_point : symmetric H D (perpendicular_bisector A G))

theorem angle_HAB_eq_3_angle_GAB :
  ∠ HAB = 3 • ∠ GAB :=
sorry

end angle_HAB_eq_3_angle_GAB_l412_412457


namespace largest_lcm_15_l412_412387

def lcm (a b : ℕ) : ℕ := Nat.lcm a b -- Define lcm function

theorem largest_lcm_15 :
  max (max (max (max (max (lcm 15 3) (lcm 15 5)) (lcm 15 6)) (lcm 15 9)) (lcm 15 10)) (lcm 15 18) = 90 :=
by 
  sorry

end largest_lcm_15_l412_412387


namespace sum_of_midpoints_eq_15_l412_412816

theorem sum_of_midpoints_eq_15 (a b c d : ℝ) (h : a + b + c + d = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 :=
by sorry

end sum_of_midpoints_eq_15_l412_412816


namespace complex_equilateral_sum_l412_412057

theorem complex_equilateral_sum (a b c : ℂ) (h1 : ∃d : ℂ, d ≠ 0 ∧ |d| = 24 ∧ a = d + b + c) (h2 : |a + b + c| = 48) :
  |a * b + a * c + b * c| = 768 :=
sorry

end complex_equilateral_sum_l412_412057


namespace total_percent_decrease_l412_412866

theorem total_percent_decrease (V : ℝ) :
  let V1 := V * 0.70 in
  let V2 := V1 * 0.90 in
  let V3 := V2 * 0.80 in
  let V4 := V3 * 0.85 in
  ((V - V4) / V) * 100 = 57.16 :=
by
  sorry

end total_percent_decrease_l412_412866


namespace distance_from_hyperbola_focus_to_line_l412_412755

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412755


namespace min_value_of_x_plus_y_l412_412155

open Real

theorem min_value_of_x_plus_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0)
  (a : ℝ × ℝ := (1 - x, 4)) (b : ℝ × ℝ := (x, -y))
  (h₃ : ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)) :
  x + y = 9 :=
by
  sorry

end min_value_of_x_plus_y_l412_412155


namespace problem_d_l412_412399

variables (a b : ℝ)
variables (angle : ℝ) (magnitude_a magnitude_b : ℝ)

def angle_between_vectors : Prop := angle = 3 * Real.pi / 4
def magnitude_of_vectors : Prop := magnitude_a = 2 ∧ magnitude_b = Real.sqrt 2
def projection (a b : ℝ) : ℝ := (a * b) / (a^2)

theorem problem_d : angle_between_vectors a b angle ∧ magnitude_of_vectors magnitude_a magnitude_b →
  projection 2 b = - 1 / 2 :=
by
  sorry

end problem_d_l412_412399


namespace fungi_population_l412_412873

theorem fungi_population :
  ∀ (time_initial time_final : ℕ) (initial_population : ℕ) (doubling_interval : ℕ),
  time_initial = 0 → -- 1:00 p.m. is our reference time (0 minutes)
  time_final = 28 → -- Time elapsed (28 minutes from 1:00 p.m. to 1:28 p.m.)
  initial_population = 30 → -- Initially 30 fungi
  doubling_interval = 4 → -- Doubles every 4 minutes
  let number_of_intervals := time_final / doubling_interval,
      final_population := initial_population * 2^number_of_intervals in
  final_population = 3840 :=
by
  intros,
  let number_of_intervals := time_final / doubling_interval,
  let final_population := initial_population * 2^number_of_intervals,
  sorry

end fungi_population_l412_412873


namespace monotonicity_intervals_max_min_values_l412_412514

noncomputable def F (x : ℝ) : ℝ := ∫ t in 0..x, t^2 + 2 * t - 8

theorem monotonicity_intervals :
  (∀ x : ℝ, x ∈ Ioi 2 → deriv F x > 0) ∧ (∀ x : ℝ, x ∈ Icc 0 2 → deriv F x < 0) :=
sorry

theorem max_min_values :
  F 3 = -6 ∧ F 2 = -(28 / 3) :=
sorry

end monotonicity_intervals_max_min_values_l412_412514


namespace new_average_after_adding_13_l412_412610

theorem new_average_after_adding_13
  (numbers : Fin 15 → ℝ)
  (h_avg : (∑ i : Fin 15, numbers i) / 15 = 40) :
  ((∑ i : Fin 15, (numbers i + 13)) / 15) = 53 := 
by
  sorry

end new_average_after_adding_13_l412_412610


namespace smallest_number_increased_by_seven_divisible_by_37_47_53_l412_412390

theorem smallest_number_increased_by_seven_divisible_by_37_47_53 : 
  ∃ n : ℕ, (n + 7) % 37 = 0 ∧ (n + 7) % 47 = 0 ∧ (n + 7) % 53 = 0 ∧ n = 92160 :=
by
  sorry

end smallest_number_increased_by_seven_divisible_by_37_47_53_l412_412390


namespace more_customers_left_than_stayed_l412_412444

-- Define the initial number of customers.
def initial_customers : ℕ := 11

-- Define the number of customers who stayed behind.
def customers_stayed : ℕ := 3

-- Define the number of customers who left.
def customers_left : ℕ := initial_customers - customers_stayed

-- Prove that the number of customers who left is 5 more than those who stayed behind.
theorem more_customers_left_than_stayed : customers_left - customers_stayed = 5 := by
  -- Sorry to skip the proof 
  sorry

end more_customers_left_than_stayed_l412_412444


namespace bicycle_distance_l412_412008

theorem bicycle_distance (a t : ℝ) (h₁ : t > 0) :
  let rate := (a / 4) / t / 1000 * 60 in
  let time_in_hours := 5 / 60 in
  rate * time_in_hours = a / (800 * t) :=
by
  let rate := (a / 4) / t / 1000 * 60
  let time_in_hours := 5 / 60
  have : rate * time_in_hours = (a / 4 / t / 1000 * 60) * (5 / 60)
  calc
    rate * time_in_hours
        = ((a / 4) / t) / 1000 * 60 * (5 / 60) : by sorry
    ... = a / (800 * t) : by sorry

end bicycle_distance_l412_412008


namespace division_result_l412_412840

theorem division_result : (8900 / 6) / 4 = 370.8333 :=
by sorry

end division_result_l412_412840


namespace factor_expression_l412_412921

theorem factor_expression (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by
  sorry

end factor_expression_l412_412921


namespace determine_c_l412_412334

noncomputable def poly (a b c : ℤ) (z : ℂ) : ℂ :=
  a * z^4 + b * z^3 + c * z^2 + b * z + a

theorem determine_c (a b c : ℤ)
  (h_eq : poly a b c (3 + complex.I) = 0)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1) :
  abs c = 165 :=
sorry

end determine_c_l412_412334


namespace question1_question2_l412_412282

variable (x a : ℝ)
def p := x^2 - 4*a*x + 3*a^2 < 0
def q := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

-- Statement for part (1)
theorem question1 (h : a = 1) (hpq : p x 1 ∧ q x) : 2 < x ∧ x < 3 := sorry

-- Statement for part (2)
theorem question2 (h : ∀ x, q x → p x ∧ ¬(p x → q x)) : 1 < a ∧ a ≤ 2 := sorry

end question1_question2_l412_412282


namespace smallest_solution_eq_l412_412969

theorem smallest_solution_eq (x : ℝ) (h : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) :
  x = 4 - Real.sqrt 2 := 
  sorry

end smallest_solution_eq_l412_412969


namespace minimum_distance_parabola_line_l412_412966

-- Definitions
def parabola (x : ℝ) : ℝ := -x^2
def line (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ := abs (A * x0 + B * y0 + C) / real.sqrt (A ^ 2 + B ^ 2)

-- Lean statement of the problem
theorem minimum_distance_parabola_line :
  ∃ x y : ℝ, parabola x = y ∧ line x y → distance_point_to_line x y 4 3 (-8) = 4 / 3 :=
by
  sorry

end minimum_distance_parabola_line_l412_412966


namespace three_cards_different_suits_probability_l412_412667

-- Define the conditions and problem
noncomputable def prob_three_cards_diff_suits : ℚ :=
  have first_card_options := 52
  have second_card_options := 39
  have third_card_options := 26
  have total_ways_to_pick := (52 : ℕ) * (51 : ℕ) * (50 : ℕ)
  (39 / 51) * (26 / 50)

-- State our proof problem
theorem three_cards_different_suits_probability :
  prob_three_cards_diff_suits = 169 / 425 :=
sorry

end three_cards_different_suits_probability_l412_412667


namespace rectangular_prism_cut_corners_edges_l412_412934

def original_edges : Nat := 12
def corners : Nat := 8
def new_edges_per_corner : Nat := 3
def total_new_edges : Nat := corners * new_edges_per_corner

theorem rectangular_prism_cut_corners_edges :
  original_edges + total_new_edges = 36 := sorry

end rectangular_prism_cut_corners_edges_l412_412934


namespace distance_from_focus_to_line_l412_412743

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412743


namespace jill_total_tax_percent_l412_412316

-- Definitions based on conditions
def total_amount_excl_taxes : ℝ := 100  -- assuming $100 for easy calculation
def percent_clothing : ℝ := 0.40
def percent_food : ℝ := 0.30
def percent_other : ℝ := 0.30

def tax_rate_clothing : ℝ := 0.04
def tax_rate_food : ℝ := 0.0
def tax_rate_other : ℝ := 0.08

-- Definitions derived from conditions
def spent_clothing := total_amount_excl_taxes * percent_clothing
def spent_food := total_amount_excl_taxes * percent_food
def spent_other := total_amount_excl_taxes * percent_other

def tax_clothing := spent_clothing * tax_rate_clothing
def tax_food := spent_food * tax_rate_food
def tax_other := spent_other * tax_rate_other

def total_tax := tax_clothing + tax_other  -- No tax on food
def total_tax_pct := (total_tax / total_amount_excl_taxes) * 100

-- The main statement to prove
theorem jill_total_tax_percent : total_tax_pct = 4 := by
  -- The proof itself is not required, hence just inserting sorry
  sorry

end jill_total_tax_percent_l412_412316


namespace range_of_a_l412_412566

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → (a ≤ 1 ∨ a ≥ 3) :=
sorry

end range_of_a_l412_412566


namespace find_C_l412_412894

theorem find_C
  (A B C : ℕ)
  (h1 : A + B + C = 1000)
  (h2 : A + C = 700)
  (h3 : B + C = 600) :
  C = 300 := by
  sorry

end find_C_l412_412894


namespace proof_problem_l412_412034

variable (pots_basil pots_rosemary pots_thyme : ℕ)
variable (leaves_per_basil leaves_per_rosemary leaves_per_thyme : ℕ)
variable (total_leaves : ℕ)

-- Define the given conditions
def conditions : Prop :=
  pots_basil = 3 ∧
  leaves_per_basil = 4 ∧
  pots_rosemary = 9 ∧
  leaves_per_rosemary = 18 ∧
  pots_thyme = 6 ∧
  leaves_per_thyme = 30

-- Define the question and the correct answer
def correct_answer : Prop :=
  total_leaves = 354

-- Translate to proof problem
theorem proof_problem : conditions → (total_leaves = pots_basil * leaves_per_basil + pots_rosemary * leaves_per_rosemary + pots_thyme * leaves_per_thyme) → correct_answer :=
by
  intro h1 h2
  exact h2
  sorry -- proof placeholder

end proof_problem_l412_412034


namespace nested_sqrt_cos_eq_l412_412322

theorem nested_sqrt_cos_eq (n : ℕ) : 
  ∀ (n : ℕ), (\sqrt (2 + sqrt (2 + ⋯ + sqrt (2 + sqrt 2)))) = 2 * cos (Real.pi / (2 ^ (n + 1))) := 
begin
  sorry
end

end nested_sqrt_cos_eq_l412_412322


namespace find_t_l412_412279

theorem find_t (c d n s t : ℝ)
  (h1 : c * d = 4)
  (h2 : ∃ n, Polynomial.hasRoot (Polynomial.C 1 ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 4) c ∧ 
             Polynomial.hasRoot (Polynomial.C 1 ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 4) d)
  (h3 : Polynomial.hasRoot (Polynomial.C 1 ^ 2 - Polynomial.C s * Polynomial.X + Polynomial.C t) (c + 1 / d))
  (h4 : Polynomial.hasRoot (Polynomial.C 1 ^ 2 - Polynomial.C s * Polynomial.X + Polynomial.C t) (d + 1 / c)) :
  t = 25 / 4 := 
sorry

end find_t_l412_412279


namespace distance_from_hyperbola_focus_to_line_l412_412757

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412757


namespace each_gets_10_fish_l412_412264

-- Define the constants and conditions
constant Ittymangnark Kingnook Oomyapeck : Type
constant fish : Type
constant eyes_of_fish : fish → ℕ
constant oomyapeck_eats_eyes : ℕ := 22
constant oomyapeck_gives_dog : ℕ := 2
constant total_eyes_eaten_by_oomyapeck : ℕ := oomyapeck_eats_eyes - oomyapeck_gives_dog
constant number_of_fish_oomyapeck_eats : ℕ := total_eyes_eaten_by_oomyapeck / 2
constant total_fish_divided : ℕ := number_of_fish_oomyapeck_eats
constant fish_split_equally : ℕ := total_fish_divided

-- The theorem statement
theorem each_gets_10_fish (day : Type) (H : Ittymangnark ≠ Kingnook ∧ Kingnook ≠ Oomyapeck ∧ Ittymangnark ≠ Oomyapeck) : 
  (number_of_fish_oomyapeck_eats = 10) ∧ (fish_split_equally = 10) :=
by {
  sorry
}

end each_gets_10_fish_l412_412264


namespace cos_triple_angle_l412_412200

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l412_412200


namespace cos_triple_angle_l412_412220

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l412_412220


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412787

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412787


namespace ceil_sqrt_sum_l412_412941

theorem ceil_sqrt_sum : (∑ n in Finset.range 31, (nat_ceil (Real.sqrt (n + 10)))) = 167 := by
  sorry

end ceil_sqrt_sum_l412_412941


namespace ratio_of_areas_l412_412273

-- Definitions of points and pentagon
variables {A B C D E : Type} [AddCommGroup A] [AffineSpace A B]
variable (A B C D E : B)
variable (ABC G_A G_B G_C G_D G_E : B)

noncomputable def centroid (P Q R : B) : B := (1 / 3) • (P + Q + R)
noncomputable def centroid4 (P Q R S : B) : B := (1 / 4) • (P + Q + R + S)

-- Given centroids
axiom h_G_A : G_A = centroid4 B C D E
axiom h_G_B : G_B = centroid4 A C D E
axiom h_G_C : G_C = centroid4 A B D E
axiom h_G_D : G_D = centroid4 A B C E
axiom h_G_E : G_E = centroid4 A B C D

-- Proof of the ratio
theorem ratio_of_areas (h_convex : convex {A, B, C, D, E}) :
  area (convex_hull {G_A, G_B, G_C, G_D, G_E}) / area (convex_hull {A, B, C, D, E}) = 1 / 16 :=
sorry

end ratio_of_areas_l412_412273


namespace distance_hyperbola_focus_to_line_l412_412780

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412780


namespace volume_of_tetrahedron_l412_412496

-- Define the conditions
def height : ℝ := sorry   -- h
def a : ℝ := height * Real.sqrt 6   -- Side length of the base in terms of height

-- Define the volume of the regular triangular pyramid
noncomputable def volume_of_pyramid (h : ℝ) : ℝ :=
  (1 / 3) * ((a ^ 2 * Real.sqrt 3) / 4) * h

-- Theorem stating the volume of the regular triangular pyramid
theorem volume_of_tetrahedron (h : ℝ) : volume_of_pyramid h = (h ^ 3 * Real.sqrt 3) / 2 := 
  sorry

end volume_of_tetrahedron_l412_412496


namespace distance_from_right_focus_to_line_l412_412767

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412767


namespace smallest_n_for_integer_sqrt_l412_412222

theorem smallest_n_for_integer_sqrt (n : ℕ) (h1: n > 0) (h2: ∃ k : ℕ, sqrt (18 * n) = k) : n = 2 :=
sorry

end smallest_n_for_integer_sqrt_l412_412222


namespace number_of_integers_covered_l412_412653

-- Define the number line and the length condition
def unit_length_cm (p : ℝ) := p = 1
def length_AB_cm (length : ℝ) := length = 2009

-- Statement of the proof problem in Lean
theorem number_of_integers_covered (ab_length : ℝ) (unit_length : ℝ) 
    (h1 : unit_length_cm unit_length) (h2 : length_AB_cm ab_length) :
    ∃ n : ℕ, n = 2009 ∨ n = 2010 :=
by
  sorry

end number_of_integers_covered_l412_412653


namespace sin2alpha_plus_sin2beta_eq_zero_l412_412976

theorem sin2alpha_plus_sin2beta_eq_zero
  (α β : ℝ)
  (h : sin α * sin β + cos α * cos β = 0) :
  sin (2 * α) + sin (2 * β) = 0 :=
sorry

end sin2alpha_plus_sin2beta_eq_zero_l412_412976


namespace system_solutions_l412_412003

theorem system_solutions (x a : ℝ) (h1 : a = -3*x^2 + 5*x - 2) (h2 : (x + 2) * a = 4 * (x^2 - 1)) (hx : x ≠ -2) :
  (x = 0 ∧ a = -2) ∨ (x = 1 ∧ a = 0) ∨ (x = -8/3 ∧ a = -110/3) :=
  sorry

end system_solutions_l412_412003


namespace example_l412_412927

-- Define the operation a ⋆ b
def star (a b : ℕ) : ℕ :=
  a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

-- State the theorem
theorem example : star 3 2 = 125 := by
  sorry

end example_l412_412927


namespace find_a_l412_412510

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l412_412510


namespace number_of_red_balloons_l412_412824

-- Definitions for conditions
def balloons_total : ℕ := 85
def at_least_one_red (red blue : ℕ) : Prop := red ≥ 1 ∧ red + blue = balloons_total
def every_pair_has_blue (red blue : ℕ) : Prop := ∀ r1 r2, r1 < red → r2 < red → red = 1

-- Theorem to be proved
theorem number_of_red_balloons (red blue : ℕ) 
  (total : red + blue = balloons_total)
  (at_least_one : at_least_one_red red blue)
  (pair_condition : every_pair_has_blue red blue) : red = 1 :=
sorry

end number_of_red_balloons_l412_412824


namespace find_b_l412_412167

theorem find_b (b : ℝ) : 
    (∀ t : ℝ, (2 * t, 1 + b * t) = (1, 0) → t = 1 / 2 ∧ 1 + b * (1 / 2) = 0) →
    b = -2 :=
by
    intro h
    specialize h (1 / 2)
    have : (2 * (1 / 2), 1 + b * (1 / 2)) = (1, 0) := by simp
    specialize h this
    cases h with ht hb
    linarith

end find_b_l412_412167


namespace distance_from_right_focus_to_line_l412_412761

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412761


namespace find_length_of_AE_l412_412580

noncomputable theory

variables {A B C D E : Type} [ordered_ring A] [ordered_ring B] [ordered_ring C] [ordered_ring D] [ordered_ring E] 

def length_of_AE (AB CD AC : ℝ) (area_ratio : ℝ) (perimeter_ratio : ℝ) : ℝ :=
  if h1 : AB = 12 ∧ CD = 15 ∧ AC = 20 ∧ area_ratio = 3 / 5 ∧ perimeter_ratio = 4 / 5 
  then 20 * real.sqrt 15 / (5 + real.sqrt 15)
  else 0

theorem find_length_of_AE (AB CD AC : ℝ) (area_ratio : ℝ) (perimeter_ratio : ℝ) :
  AB = 12 → CD = 15 → AC = 20 → area_ratio = 3 / 5 → perimeter_ratio = 4 / 5 →
  length_of_AE AB CD AC area_ratio perimeter_ratio = 20 * real.sqrt 15 / (5 + real.sqrt 15) :=
by sorry

end find_length_of_AE_l412_412580


namespace distance_from_focus_to_line_l412_412688

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412688


namespace parabola_equation_point_P_coordinates_l412_412122

-- Definitions for the parabola, its focus, and the point M
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = -2 * p * x
def focus_x (p : ℝ) : ℝ := -p / 2
def M (p x₀ y₀ : ℝ) : Prop := x₀ = 1 / 5 - p / 4 ∧ y₀ = 4 / 5
def symmetric_point (p : ℝ) : Prop := ∃ M' : ℝ × ℝ, M' = (2 / 5, 8 / 5)

-- Proof statement for part (1)
theorem parabola_equation (p : ℝ) (h₀ : p > 0) (x₀ y₀ : ℝ)
  (hM : M p x₀ y₀) (h_sym : symmetric_point p) : 
  parabola p x₀ y₀ :=
begin
  sorry
end

-- Proof statement for part (2)
theorem point_P_coordinates (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ)
  (h₀ : p > 0) (hM : M p x₀ y₀) (h_sym : symmetric_point p)
  (hN : parabola p x₁ y₁ ∧ parabola p x₂ y₂) : 
  ∃ P : ℝ × ℝ, P = (0, -(6 + 2*Real.sqrt 21)/5) ∨ P = (0, -(6 - 2*Real.sqrt 21)/5) :=
begin
  sorry
end

end parabola_equation_point_P_coordinates_l412_412122


namespace zack_initial_marbles_l412_412402

theorem zack_initial_marbles :
  let a1 := 20
  let a2 := 30
  let a3 := 35
  let a4 := 25
  let a5 := 28
  let a6 := 40
  let r := 7
  let T := a1 + a2 + a3 + a4 + a5 + a6 + r
  T = 185 :=
by
  sorry

end zack_initial_marbles_l412_412402


namespace arithmetic_sequence_problem_l412_412568

noncomputable theory

open_locale classical

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition 1: The sequence {a_n} is arithmetic
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition 2 and Condition 3
axiom h1 : a 2 + S 3 = 4
axiom h2 : a 3 + S 5 = 12

-- Sum of the first n terms of an arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

-- Condition relating S and sum_of_first_n_terms
axiom S_eq_sum : ∀ n, S n = sum_of_first_n_terms a n

-- The statement to prove
theorem arithmetic_sequence_problem : a 4 + S 7 = 24 := 
  sorry

end arithmetic_sequence_problem_l412_412568


namespace original_water_amount_in_mixture_l412_412018

-- Define heat calculations and conditions
def latentHeatOfFusionIce : ℕ := 80       -- Latent heat of fusion for ice in cal/g
def initialTempWaterAdded : ℕ := 20      -- Initial temperature of added water in °C
def finalTempMixture : ℕ := 5            -- Final temperature of the mixture in °C
def specificHeatWater : ℕ := 1           -- Specific heat of water in cal/g°C

-- Define the known parameters of the problem
def totalMass : ℕ := 250               -- Total mass of the initial mixture in grams
def addedMassWater : ℕ := 1000         -- Mass of added water in grams
def initialTempMixtureIceWater : ℕ := 0  -- Initial temperature of the ice-water mixture in °C

-- Define the equation that needs to be solved
theorem original_water_amount_in_mixture (x : ℝ) :
  (250 - x) * 80 + (250 - x) * 5 + x * 5 = 15000 →
  x = 90.625 :=
by
  intro h
  sorry

end original_water_amount_in_mixture_l412_412018


namespace math_problem_l412_412195

theorem math_problem (x : ℝ) (h : x + real.sqrt (x^2 - 1) + (1 / (x - real.sqrt (x^2 - 1))) = 24) :
  x^2 + real.sqrt (x^4 - 1) + (1 / (x^2 + real.sqrt (x^4 - 1))) = 10525 / 144 :=
by
  sorry

end math_problem_l412_412195


namespace degree_of_polynomial_l412_412289

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, 1 < n → n < p → p % n ≠ 0

def polynomial_with_int_coeffs (f : ℕ → ℤ) (d : ℕ) : Prop :=
  ∃ coeffs : fin d.succ → ℤ, ∀ x, f x = ∑ i in finset.range d.succ, coeffs ⟨i, fin.is_lt _⟩ * x ^ i

def satisfies_conditions (p d : ℕ) (f : ℕ → ℤ) : Prop :=
  is_prime p ∧ polynomial_with_int_coeffs f d ∧ 
  f 0 = 0 ∧ f 1 = 1 ∧ 
  ∀ n : ℕ, n > 0 → (f n % p = 0 ∨ f n % p = 1)

theorem degree_of_polynomial (p d : ℕ) (f : ℕ → ℤ) : 
  satisfies_conditions p d f → d ≥ p - 1 :=
sorry

end degree_of_polynomial_l412_412289


namespace common_ratio_of_series_l412_412489

theorem common_ratio_of_series (a1 a2 : ℚ) (h1 : a1 = 5/6) (h2 : a2 = -4/9) :
  (a2 / a1) = -8/15 :=
by
  sorry

end common_ratio_of_series_l412_412489


namespace number_of_students_l412_412872

theorem number_of_students 
  (n : ℕ)
  (h1: 108 - 36 = 72)
  (h2: ∀ n > 0, 108 / n - 72 / n = 3) :
  n = 12 :=
sorry

end number_of_students_l412_412872


namespace determine_b_l412_412997

theorem determine_b (a b : ℝ) (h : a - 1 + 2 * a * complex.I = -4 + b * complex.I) : b = -6 :=
sorry

end determine_b_l412_412997


namespace domain_of_f_l412_412163

noncomputable def f (x : ℝ) : ℝ := log (1 - x) / log 10

theorem domain_of_f : 
  (∀ y : ℝ, f y < 0 → 0 < y ∧ y < 1) ↔ (∀ x, f x = y → 0 < x ∧ x < 1) := 
by
  sorry

end domain_of_f_l412_412163


namespace alpha_plus_beta_eq_7_over_6_pi_l412_412116

-- Define the problem constants and conditions
variables {α β : ℝ}
axiom α_range : 0 < α ∧ α < π
axiom β_range : 0 < β ∧ β < π
axiom sin_alpha_minus_beta : Real.sin (α - β) = 5 / 6
axiom tan_ratio : Real.tan α / Real.tan β = -1 / 4

-- State the theorem to prove
theorem alpha_plus_beta_eq_7_over_6_pi : α + β = 7 / 6 * π :=
by
  sorry

end alpha_plus_beta_eq_7_over_6_pi_l412_412116


namespace fraction_addition_l412_412916

theorem fraction_addition : (3 / 8) + (9 / 12) = 9 / 8 := sorry

end fraction_addition_l412_412916


namespace distance_from_focus_to_line_l412_412708

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412708


namespace part_I_part_II_l412_412137

noncomputable def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = sqrt (x - 1) + sqrt (2 - x)}
noncomputable def B (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≥ a}
noncomputable def U : Set ℝ := Set.univ
noncomputable def complement (s : Set ℝ) : Set ℝ := { x : ℝ | x ∉ s }

theorem part_I (a : ℝ) (ha: a = 2) : 
  (complement A).Inter (B a) = {y : ℝ | y ≥ 4} :=
by
  sorry

theorem part_II : 
  ((complement A).Union (B a)) = Set.univ → a ≤ 0 :=
by
  sorry

end part_I_part_II_l412_412137


namespace total_leaves_correct_l412_412036

-- Definitions based on conditions
def basil_pots := 3
def rosemary_pots := 9
def thyme_pots := 6

def basil_leaves_per_pot := 4
def rosemary_leaves_per_pot := 18
def thyme_leaves_per_pot := 30

-- Calculate the total number of leaves
def total_leaves : Nat :=
  (basil_pots * basil_leaves_per_pot) +
  (rosemary_pots * rosemary_leaves_per_pot) +
  (thyme_pots * thyme_leaves_per_pot)

-- The statement to prove
theorem total_leaves_correct : total_leaves = 354 := by
  sorry

end total_leaves_correct_l412_412036


namespace asymptote_of_hyperbola_l412_412249

constant a b p : ℝ
constant hyperbola_eq : ∀ x y : ℝ, a > 0 → b > 0 → (x^2 / a^2) - (y^2 / b^2) = 1
constant parabola_eq : ∀ x y : ℝ, p > 0 → x^2 = 2 * p * y
constant af_bf_condition : ∀ A B F O : ℝ, |A - F| + |B - F| = 4 * |O - F|

theorem asymptote_of_hyperbola:
  (∃ x y : ℝ, a > 0 ∧ b > 0 ∧ p > 0 ∧ 
  hyperbola_eq x y (by sorry) (by sorry) ∧ parabola_eq x y (by sorry) ∧ 
  af_bf_condition A B (sqrt 2 * b) O ) → 
  (∀ x : ℝ, y = sqrt 2 / 2 * x ∨ y = -sqrt 2 / 2 * x) :=
by
  sorry

end asymptote_of_hyperbola_l412_412249


namespace distance_from_hyperbola_focus_to_line_l412_412754

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412754


namespace tagged_fish_in_second_catch_l412_412242

theorem tagged_fish_in_second_catch :
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  (total_tagged / N) * total_caught = 5 :=
by
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  show (total_tagged / N) * total_caught = 5
  sorry

end tagged_fish_in_second_catch_l412_412242


namespace min_value_expression_l412_412092

theorem min_value_expression (x : ℝ) (hx : x > 0) : 2 * real.sqrt (2 * x) + 4 / x ≥ 6 :=
sorry

end min_value_expression_l412_412092


namespace altitudes_intersect_at_orthocenter_altitudes_properties_l412_412850

-- Define the conditions for the given triangle
variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Variables representing the orthocenter and circumradius
variable (H : A)
variable (R : ℝ)

-- Definitions of distances in the context of metric spaces
noncomputable def distance (x y : A) : ℝ := sorry

-- Definitions specific to the problem
noncomputable def AH := distance A H
noncomputable def BC := distance B C

-- Angle α is the angle at vertex A in △ABC
variable α : ℝ

-- Mathematically equivalent proof problem statements in Lean 4
theorem altitudes_intersect_at_orthocenter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (H : A) (R : ℝ) :
  (AH H = distance A H) := sorry

theorem altitudes_properties (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (H : A) (R : ℝ) (α : ℝ) 
  (AH := distance A H) (BC := distance B C) :
  (AH^2 + BC^2 = 4 * R^2) ∧ (AH = BC * |Real.cot α|) := sorry

end altitudes_intersect_at_orthocenter_altitudes_properties_l412_412850


namespace calculation_part_1_system_of_inequalities_l412_412918

theorem calculation_part_1 : 
  abs (-real.sqrt 3) + (3 - real.pi)^0 + (1 / 3 : ℝ)⁻² = real.sqrt 3 + 10 :=
by
  sorry

theorem system_of_inequalities (x : ℝ) : 
  (3 * x + 1 > 2 * (x - 1)) ∧ (x - 1 ≤ 3 * x + 3) ↔ x ≥ -2 :=
by
  sorry

end calculation_part_1_system_of_inequalities_l412_412918


namespace find_B_find_cos_A_plus_cos_C_l412_412526

-- Definitions and given conditions
variables {R : Type*} [Real R]
variables (A B C a b c : R) (m n : R × R)

-- Define vectors and perpendicular relationship
def m := (a, 2 * b)
def n := (Real.sqrt 3, -Real.sin A)

-- m perpendicular to n
def perp_m_n := (m.1 * n.1 + m.2 * n.2 = 0)

-- Define obtuse angle property of angle B
def obtuse_B := (Real.pi / 2 < B ∧ B < Real.pi)

-- Define cos A + cos C
def cos_A_plus_cos_C := Real.cos A + Real.cos C

-- Lean statement for the proof problems
theorem find_B (h1 : perp_m_n) (h2 : obtuse_B) : B = 2 * Real.pi / 3 :=
by sorry

theorem find_cos_A_plus_cos_C (h1 : perp_m_n) (h2 : obtuse_B) : 
  ∃ x : R, cos_A_plus_cos_C = x ∧ (3 / 2 < x ∧ x ≤ Real.sqrt 3) :=
by sorry

end find_B_find_cos_A_plus_cos_C_l412_412526


namespace intersection_complement_is_singleton_l412_412172

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 2, 5}

theorem intersection_complement_is_singleton : (U \ M) ∩ N = {1} := by
  sorry

end intersection_complement_is_singleton_l412_412172


namespace three_digit_integers_count_l412_412160

def digits : Finset ℕ := {2, 4, 7, 5}

def count_three_digit_integers (S : Finset ℕ) : ℕ :=
  S.card * (S.card - 1) * (S.card - 2)

theorem three_digit_integers_count : count_three_digit_integers digits = 24 :=
by
  simp [digits, count_three_digit_integers]
  sorry

end three_digit_integers_count_l412_412160


namespace player2_wins_l412_412371

-- Definitions for the initial conditions and game rules
def initial_piles := [10, 15, 20]
def split_rule (piles : List ℕ) (move : ℕ → ℕ × ℕ) : List ℕ :=
  let (pile1, pile2) := move (piles.head!)
  (pile1 :: pile2 :: piles.tail!)

-- Winning condition proof
theorem player2_wins :
  ∀ piles : List ℕ, piles = [10, 15, 20] →
  (∀ move_count : ℕ, move_count = 42 →
  (move_count > 0 ∧ ¬ ∃ split : ℕ → ℕ × ℕ, move_count % 2 = 1)) :=
by
  intro piles hpiles
  intro move_count hmove_count
  sorry

end player2_wins_l412_412371


namespace equilateral_triangle_ratio_l412_412654

theorem equilateral_triangle_ratio (A B C X Y Z : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (hABC : is_equilateral_triangle A B C)
  (hX : X ∈ line_segment A B)
  (hY : Y ∈ line_segment B C)
  (hZ : Z ∈ line_segment C A)
  (hRatio : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ AX / XB = x ∧ BY / YC = x ∧ CZ / ZA = x)
  (hArea : ∃ S : ℝ, area_triangle C X A _ + area_triangle B Z _ _ + area_triangle A Y _ B = ¼ * (area_triangle A B C)) :
  ∃ (x : ℝ), x = (3 - real.sqrt 5) / 2 := sorry

end equilateral_triangle_ratio_l412_412654


namespace sum_of_ceil_sqrt_l412_412947

theorem sum_of_ceil_sqrt :
  (∑ n in finset.range (16 - 10 + 1), ⌈ real.sqrt (10 + n : ℝ) ⌉) +
  (∑ n in finset.range (25 - 17 + 1), ⌈ real.sqrt (17 + n : ℝ) ⌉) +
  (∑ n in finset.range (36 - 26 + 1), ⌈ real.sqrt (26 + n : ℝ) ⌉) +
  (∑ n in finset.range (40 - 37 + 1), ⌈ real.sqrt (37 + n : ℝ) ⌉) = 167 :=
by
  sorry

end sum_of_ceil_sqrt_l412_412947


namespace matt_paper_piles_l412_412313

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end matt_paper_piles_l412_412313


namespace arrangement_of_books_l412_412442

-- Conditions: 
-- 4 copies of Introduction to Geometry
-- 5 copies of Introduction to Algebra with 
-- 2 specific copies of Introduction to Geometry must be adjacent

def num_ways_to_arrange_books : Nat :=
  let total_books := 9                        -- Total books = 4 Geometry + 5 Algebra
  let fixed_unit_slots := 8                   -- Consider the 2 specific Geometry books as a single unit
  let ways_to_arrange_slots := factorial fixed_unit_slots
  let ways_to_arrange_fixed_unit := 2         -- Internal arrangement of the unit of 2 Geometry books
  ways_to_arrange_slots * ways_to_arrange_fixed_unit / factorial 3 / factorial 5

-- The proof statement:
theorem arrangement_of_books : num_ways_to_arrange_books = 112 := sorry

end arrangement_of_books_l412_412442


namespace remainder_sum_abc_mod5_l412_412223

theorem remainder_sum_abc_mod5 (a b c : ℕ) (h1 : a < 5) (h2 : b < 5) (h3 : c < 5)
  (h4 : a * b * c ≡ 1 [MOD 5])
  (h5 : 4 * c ≡ 3 [MOD 5])
  (h6 : 3 * b ≡ 2 + b [MOD 5]) :
  (a + b + c) % 5 = 1 :=
  sorry

end remainder_sum_abc_mod5_l412_412223


namespace minimum_value_of_sum_2_l412_412134

noncomputable def minimum_value_of_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) : 
  Prop := 
  x + y = 2

theorem minimum_value_of_sum_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) :
  minimum_value_of_sum x y hx hy inequality := 
sorry

end minimum_value_of_sum_2_l412_412134


namespace find_complex_z_l412_412624

noncomputable def conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem find_complex_z (z : ℂ) (i : ℂ := complex.I) (condition : z * (conjugate z) = 2 * (conjugate z + i)) :
  z = 1 + i :=
sorry

end find_complex_z_l412_412624


namespace cos_triple_angle_l412_412210

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l412_412210


namespace largest_integer_n_l412_412068

def floor (x : ℝ) := ⌊x⌋

theorem largest_integer_n (n : ℕ) (h : floor (real.sqrt n) = 5) : n ≤ 35 :=
by {
  sorry
}

end largest_integer_n_l412_412068


namespace limit_derivative_eval_l412_412225

noncomputable def limit_eval (f : ℝ → ℝ) (x₀ : ℝ) : ℝ :=
  limit (λ k : ℝ, (f (x₀ - k) - f x₀) / (2 * k))

theorem limit_derivative_eval 
  (f : ℝ → ℝ) (x₀ : ℝ) (hf' : deriv f x₀ = 2) : 
  limit_eval f x₀ = -1 :=
by
  sorry

end limit_derivative_eval_l412_412225


namespace nat_numbers_eq_floor_condition_l412_412482

theorem nat_numbers_eq_floor_condition (a b : ℕ):
  (⌊(a ^ 2 : ℚ) / b⌋₊ + ⌊(b ^ 2 : ℚ) / a⌋₊ = ⌊((a ^ 2 + b ^ 2) : ℚ) / (a * b)⌋₊ + a * b) →
  (b = a ^ 2 + 1) ∨ (a = b ^ 2 + 1) :=
by
  sorry

end nat_numbers_eq_floor_condition_l412_412482


namespace cos_triple_angle_l412_412217

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l412_412217


namespace valid_number_is_18_l412_412846

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

def is_less_than_20 (n : ℕ) : Prop := n < 20

def is_valid_candidate (n : ℕ) : Prop := is_multiple_of_2 n ∧ is_less_than_20 n

theorem valid_number_is_18 : (15 = 18 ∨ 18 = 18 ∨ 29 = 18 ∨ 21 = 18) → is_valid_candidate 18 :=
by
  intro h
  have : is_multiple_of_2 18 := by simp [is_multiple_of_2]
  have : is_less_than_20 18 := by simp [is_less_than_20]
  exact ⟨this, this_1⟩

end valid_number_is_18_l412_412846


namespace total_amount_l412_412021

def shares (a b c : ℕ) : Prop :=
  b = 1800 ∧ 2 * b = 3 * a ∧ 3 * c = 4 * b

theorem total_amount (a b c : ℕ) (h : shares a b c) : a + b + c = 5400 :=
by
  have h₁ : 2 * b = 3 * a := h.2.1
  have h₂ : 3 * c = 4 * b := h.2.2
  have hb : b = 1800 := h.1
  sorry

end total_amount_l412_412021


namespace common_tangents_of_circles_l412_412342

noncomputable def circle1 := {center := (0, 0), radius := 1}
noncomputable def circle2 := {center := (3, 4), radius := 4}

theorem common_tangents_of_circles :
  let distance := Real.sqrt ((circle2.center.1 - circle1.center.1) ^ 2 
                               + (circle2.center.2 - circle1.center.2) ^ 2),
      radius_sum := circle1.radius + circle2.radius in
  distance = radius_sum →
  ∃ n, n = 3 ∧ (number_of_common_tangents circle1 circle2 = n) :=
by
  sorry

end common_tangents_of_circles_l412_412342


namespace carly_running_ratio_l412_412051

theorem carly_running_ratio
    (first_week_miles : ℕ)
    (second_week_miles : ℕ)
    (third_week_ratio : ℚ)
    (reduction_miles : ℕ)
    (fourth_week_miles : ℕ) :
    first_week_miles = 2 →
    second_week_miles = (2 * first_week_miles + 3) →
    fourth_week_miles = 4 →
    reduction_miles = 5 →
    (second_week_miles * third_week_ratio - reduction_miles = fourth_week_miles) →
    third_week_ratio = (9 / 7) :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4] at h5
  exact h5

end carly_running_ratio_l412_412051


namespace investment_interest_rate_calculation_l412_412420

theorem investment_interest_rate_calculation :
  let initial_investment : ℝ := 15000
  let first_year_rate : ℝ := 0.08
  let first_year_investment : ℝ := initial_investment * (1 + first_year_rate)
  let second_year_investment : ℝ := 17160
  ∃ (s : ℝ), (first_year_investment * (1 + s / 100) = second_year_investment) → s = 6 :=
by
  sorry

end investment_interest_rate_calculation_l412_412420


namespace distance_from_hyperbola_focus_to_line_l412_412752

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412752


namespace no_primes_divisible_by_45_l412_412178

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define 45 and its prime factors
def is_divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem to prove the number of prime numbers divisible by 45 is 0
theorem no_primes_divisible_by_45 : ∀ n : ℕ, is_prime n → is_divisible_by_45 n → false :=
by
  intro n
  assume h_prime h_div_45
  sorry

end no_primes_divisible_by_45_l412_412178


namespace value_of_expression_when_x_is_3_value_of_expression_when_x_is_neg_3_l412_412356

variable {a b c d x : ℝ}

-- Conditions
def additive_inverses (a b : ℝ) : Prop := a + b = 0
def multiplicative_inverses (c d : ℝ) : Prop := c * d = 1
def abs_val_3 (x : ℝ) : Prop := abs x = 3

-- Expression Definition
def algebraic_expression (x a b c d : ℝ) :=
  x^2 + (a + b - c*d) * x + sqrt (a + b) + real.cbrt (c*d)

theorem value_of_expression_when_x_is_3 : 
  additive_inverses a b → 
  multiplicative_inverses c d → 
  abs_val_3 3 → 
  algebraic_expression 3 a b c d = 7 := by
  sorry

theorem value_of_expression_when_x_is_neg_3 : 
  additive_inverses a b → 
  multiplicative_inverses c d → 
  abs_val_3 (-3) → 
  algebraic_expression (-3) a b c d = 13 := by
  sorry

end value_of_expression_when_x_is_3_value_of_expression_when_x_is_neg_3_l412_412356


namespace distance_to_line_is_sqrt5_l412_412795

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412795


namespace comet_stargazing_percentage_l412_412176

theorem comet_stargazing_percentage :
  ∀ (haley_hours : Float) (emma_minutes : Float) (james_hours : Float) (olivia_factor : Float)
    (comet_minutes : Float) (other_minutes : Float),
    haley_hours = 2 →
    emma_minutes = 45 →
    james_hours = 1.5 →
    olivia_factor = 3 →
    comet_minutes = 20 →
    other_minutes = 30 →
    let haley_minutes := haley_hours * 60 in
    let james_minutes := james_hours * 60 in
    let olivia_minutes := olivia_factor * james_minutes in
    let comet_actual_minutes := comet_minutes / 2 in
    let other_actual_minutes := other_minutes / 2 in
    let stargazing_minutes := comet_actual_minutes + other_actual_minutes in
    let total_minutes := haley_minutes + emma_minutes + james_minutes + olivia_minutes + stargazing_minutes in
    let percentage_comet := (comet_actual_minutes / total_minutes) * 100 in
    Float.round percentage_comet = 2 :=
by
  intros haley_hours emma_minutes james_hours olivia_factor comet_minutes other_minutes
  assume h1 h2 h3 h4 h5 h6
  let haley_minutes := haley_hours * 60
  let james_minutes := james_hours * 60
  let olivia_minutes := olivia_factor * james_minutes
  let comet_actual_minutes := comet_minutes / 2
  let other_actual_minutes := other_minutes / 2
  let stargazing_minutes := comet_actual_minutes + other_actual_minutes
  let total_minutes := haley_minutes + emma_minutes + james_minutes + olivia_minutes + stargazing_minutes
  let percentage_comet := (comet_actual_minutes / total_minutes) * 100
  have h_approx : Float.round percentage_comet = 2 := sorry
  exact h_approx

end comet_stargazing_percentage_l412_412176


namespace categorize_numbers_l412_412958

theorem categorize_numbers 
  (a b c d e f g h : ℝ)
  (ha : a = -(-3))
  (hb : b = -1)
  (hc : c = |{-1.333}|) -- Taking absolute value in decimal
  (hd : d = 0)
  (he : e = -22/7)
  (hf : f = (-2)^2)
  (hg : g = 3.14)
  (hh : h = -20 / 100) :
  ({a, c, f, g} = {3, 1.333, 4, 3.14}) ∧
  ({a, d, f} = {3, 0, 4}) ∧
  ({a, b, d, f} = {3, -1, 0, 4}) ∧
  ({e, hh} = {-22/7, -1/5}) :=
by sorry

end categorize_numbers_l412_412958


namespace distance_from_hyperbola_focus_to_line_l412_412760

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412760


namespace find_a_l412_412509

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l412_412509


namespace all_poli_pythagorean_l412_412905

def poli_pythagorean (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (A : Fin n → ℕ), (∀ i j, i ≠ j → A i ≠ A j) ∧
  (∀ i, ∃ k, (A i)^2 + (A ((i + 1) % n))^2 = k^2)

theorem all_poli_pythagorean (n : ℕ) : poli_pythagorean n :=
by
  intros
  sorry

end all_poli_pythagorean_l412_412905


namespace largest_of_five_consecutive_integers_with_product_15120_eq_9_l412_412501

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end largest_of_five_consecutive_integers_with_product_15120_eq_9_l412_412501


namespace distance_from_focus_to_line_l412_412747

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412747


namespace find_a_l412_412511

theorem find_a (a : ℝ) (h : 1 ∈ ({a + 2, (a + 1)^2, a^2 + 3a + 3} : set ℝ)) : a = 0 :=
sorry

end find_a_l412_412511


namespace cleaning_time_with_doubled_anne_speed_l412_412909

theorem cleaning_time_with_doubled_anne_speed :
  (A B : ℝ) -- Bruce's and Anne's cleaning rates as real numbers
  (h_anne_rate : A = 1 / 12) -- Anne's cleaning rate
  (h_combined_rate : B + A = 1 / 4) -- Bruce and Anne's combined rate
  (h_doubled_anne_rate : 2 * A) -- Anne's doubled rate
  : (1 / (B + 2 * A) = 3) :=  -- Proving the time to clean the house when Anne's speed is doubled
by
  -- This means setting up the proof here, typically replaced with proof code or tactic steps.
  sorry

end cleaning_time_with_doubled_anne_speed_l412_412909


namespace largest_integer_of_n_l412_412065

def floor (x : ℝ) := Int.floor x

theorem largest_integer_of_n (n : ℕ) (h : floor (Real.sqrt n) = 5) : n = 35 :=
by
  sorry

end largest_integer_of_n_l412_412065


namespace deg_d_eq_6_l412_412432

theorem deg_d_eq_6
  (f d q : Polynomial ℝ)
  (r : Polynomial ℝ)
  (hf : f.degree = 15)
  (hdq : (d * q + r) = f)
  (hq : q.degree = 9)
  (hr : r.degree = 4) :
  d.degree = 6 :=
by sorry

end deg_d_eq_6_l412_412432


namespace find_denominator_l412_412237

theorem find_denominator (y x : ℝ) (hy : y > 0) (h : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end find_denominator_l412_412237


namespace ceil_sqrt_sum_l412_412939

theorem ceil_sqrt_sum : (∑ n in Finset.range 31, (nat_ceil (Real.sqrt (n + 10)))) = 167 := by
  sorry

end ceil_sqrt_sum_l412_412939


namespace total_revenue_4706_l412_412019

noncomputable def totalTicketRevenue (seats : ℕ) (show2pm : ℕ × ℕ) (show5pm : ℕ × ℕ) (show8pm : ℕ × ℕ) : ℕ :=
  let revenue2pm := show2pm.1 * 4 + (seats - show2pm.1) * 6
  let revenue5pm := show5pm.1 * 5 + (seats - show5pm.1) * 8
  let revenue8pm := show8pm.1 * 7 + (show8pm.2 - show8pm.1) * 10
  revenue2pm + revenue5pm + revenue8pm

theorem total_revenue_4706 :
  totalTicketRevenue 250 (135, 250) (160, 250) (98, 225) = 4706 :=
by
  unfold totalTicketRevenue
  -- We provide the proof steps here in a real proof scenario.
  -- We are focusing on the statement formulation only.
  sorry

end total_revenue_4706_l412_412019


namespace percentage_of_students_with_glasses_l412_412367

theorem percentage_of_students_with_glasses : 
  ∀ (total_students num_without_glasses : ℕ),
  total_students = 325 → num_without_glasses = 195 →
  ((total_students - num_without_glasses) / total_students.toRat) * 100 = 40 := 
by
  intros total_students num_without_glasses h_total h_without
  sorry

end percentage_of_students_with_glasses_l412_412367


namespace distance_hyperbola_focus_to_line_l412_412782

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412782


namespace distance_hyperbola_focus_to_line_l412_412776

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412776


namespace area_of_polygon_ABHFGD_l412_412253

noncomputable def total_area_ABHFGD : ℝ :=
  let side_ABCD := 3
  let side_EFGD := 5
  let area_ABCD := side_ABCD * side_ABCD
  let area_EFGD := side_EFGD * side_EFGD
  let area_DBH := 0.5 * 3 * (3 / 2 : ℝ) -- Area of triangle DBH
  let area_DFH := 0.5 * 5 * (5 / 2 : ℝ) -- Area of triangle DFH
  area_ABCD + area_EFGD - (area_DBH + area_DFH)

theorem area_of_polygon_ABHFGD : total_area_ABHFGD = 25.5 := by
  sorry

end area_of_polygon_ABHFGD_l412_412253


namespace distance_from_right_focus_to_line_l412_412727

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412727


namespace sin_cos_identity_l412_412561

theorem sin_cos_identity (α : ℝ) (h : sin α = 2 * cos α) : sin α ^ 2 + 6 * cos α ^ 2 = 2 := by
  sorry

end sin_cos_identity_l412_412561


namespace even_function_iff_a_eq_2_l412_412998

variable (a b : ℝ)
def f (x : ℝ) : ℝ := real.cbrt (x^2) + (a - 2) / x + b

theorem even_function_iff_a_eq_2 (h : ∀ x : ℝ, f a b (-x) = f a b x) : a = 2 := by
  sorry

end even_function_iff_a_eq_2_l412_412998


namespace distance_MP_l412_412579

theorem distance_MP (R m : ℝ) (MN PQ : ℝ) (hMN_perpendicular_PQ : true) (hNQ : NQ = m) :
  ∃ MP : ℝ, MP = sqrt (4 * R^2 - m^2) :=
by
  sorry

end distance_MP_l412_412579


namespace perpendicular_lines_l412_412278

open Locale Classical

theorem perpendicular_lines
  (A B C O I_B I_C E F Y Z P : Type) 
  (hAtriangle : AcuteTriangle A B C)
  (hCircumcenter : Circumcenter O A B C)
  (hBexcenter : BExcenter I_B A B C)
  (hCexcenter : CExcenter I_C A B C)
  (hPoints_E_Y : ∃ Y, Y ∈ AC ∧ ∠ ABY = ∠ CBY ∧ BE ⟂ AC)
  (hPoints_F_Z : ∃ Z, Z ∈ AB ∧ ∠ ACZ = ∠ BCZ ∧ CF ⟂ AB)
  (hIntersection_P : Incidence I_B F I_C E P)
  : Perpendicular (Line P O) (Line Y Z) :=
by
  sorry

end perpendicular_lines_l412_412278


namespace student_correct_answers_l412_412025

theorem student_correct_answers :
  ∃ (C I : ℕ), (C - 2 * I = 70) ∧ (C + I = 100) ∧ (C = 90) :=
by {
  use [90, 10],
  split,
  { 
    sorry,  -- This is where the proof for the first part would go
  },
  split,
  { 
    sorry,  -- This is where the proof for the second part would go
  },
  refl   -- This part verifies that C = 90
}

end student_correct_answers_l412_412025


namespace cos_triple_angle_l412_412209

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l412_412209


namespace find_a_l412_412513

theorem find_a (a : ℝ) (h : 1 ∈ ({a + 2, (a + 1)^2, a^2 + 3a + 3} : set ℝ)) : a = 0 :=
sorry

end find_a_l412_412513


namespace sum_of_ages_l412_412330

variable (S M : ℝ)  -- Variables for Sarah's and Matt's ages

-- Conditions
def sarah_older := S = M + 8
def future_age_relationship := S + 10 = 3 * (M - 5)

-- Theorem: The sum of their current ages is 41
theorem sum_of_ages (h1 : sarah_older S M) (h2 : future_age_relationship S M) : S + M = 41 := by
  sorry

end sum_of_ages_l412_412330


namespace distance_from_focus_to_line_l412_412704

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412704


namespace problem_I_problem_II_problem_III_l412_412545

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x - Real.log x

theorem problem_I : 
  ∀ x, 0 < x → 
       (∀ x, 0 < x → x < 1/2 → monotone_decreasing (f x 1)) ∧ 
       (∀ x, x > 1/2 → monotone_increasing (f x 1)) :=
by 
  sorry

theorem problem_II :
  ∀ a, (∀ x, 1 ≤ x → x ≤ 2 → monotone_decreasing (f x a)) ↔ a ≤ -7/2 :=
by 
  sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x - Real.log x

theorem problem_III :
  ∃ a, a = Real.exp 2 ∧ ∀ x, 0 < x → x ≤ Real.exp 1 → (minimum (g x a)) = 3 :=
by 
  sorry

end problem_I_problem_II_problem_III_l412_412545


namespace no_solution_for_equation_l412_412932

theorem no_solution_for_equation (x y z : ℤ) : x^3 + y^3 ≠ 9 * z + 5 := 
by
  sorry

end no_solution_for_equation_l412_412932


namespace exists_point_D_l412_412257

-- Definitions for points in a triangle and the ratios
variables {α : Type*} [field α]
variables {A B C P Q : α}

-- Condition: P and Q satisfy given ratio constraints
def point_on_BC (x : α) : Prop := CP / CB = x ∧ x ≠ 0
def point_on_AC (x : α) : Prop := CQ / QA = x ∧ x ≠ 0

-- Statement to be proved
theorem exists_point_D (x : α) (h1 : point_on_BC x) (h2 : point_on_AC x) : 
  ∃ D : α, ∀ x, line_through D P Q :=
sorry

end exists_point_D_l412_412257


namespace functions_with_inverses_l412_412166

noncomputable def graphF := "not one-to-one"
noncomputable def graphG := "consists of two disconnected parts, each part is injective"
noncomputable def graphH := "a straight line with a negative slope"
noncomputable def graphI := "a circle"
noncomputable def graphJ := "a cubic function"

theorem functions_with_inverses : 
  ∀ (F G H I J : Type), F = graphF → G = graphG → H = graphH → I = graphI → J = graphJ → 
  has_inverse G ∧ has_inverse H ∧ has_inverse J ∧ ¬has_inverse F ∧ ¬has_inverse I :=
by
  sorry

end functions_with_inverses_l412_412166


namespace no_valid_N_exists_l412_412671

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

def digit_sum (digits : List ℕ) : ℕ := digits.sum

def digit_product (digits : List ℕ) : ℕ := digits.prod

def contains_at_least_three_fives (digits : List ℕ) : Prop :=
  (digits.filter (λ d, d = 5)).length ≥ 3

theorem no_valid_N_exists :
  ¬ ∃ N : List ℕ,
    N.length = 1989 ∧
    (∀ d ∈ N, is_valid_digit d) ∧
    contains_at_least_three_fives N ∧
    digit_sum N = digit_product N :=
by
  sorry

end no_valid_N_exists_l412_412671


namespace students_selected_from_grade_10_l412_412017

theorem students_selected_from_grade_10 (students_grade10 students_grade11 students_grade12 total_selected : ℕ)
  (h_grade10 : students_grade10 = 1200)
  (h_grade11 : students_grade11 = 1000)
  (h_grade12 : students_grade12 = 800)
  (h_total_selected : total_selected = 100) :
  students_grade10 * total_selected = 40 * (students_grade10 + students_grade11 + students_grade12) :=
by
  sorry

end students_selected_from_grade_10_l412_412017


namespace chess_tournament_max_N_l412_412574

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l412_412574


namespace avg_choc_pieces_per_cookie_l412_412055

theorem avg_choc_pieces_per_cookie {cookies chips mms pieces : ℕ} 
  (h1 : cookies = 48) 
  (h2 : chips = 108) 
  (h3 : mms = chips / 3) 
  (h4 : pieces = chips + mms) : 
  pieces / cookies = 3 := 
by sorry

end avg_choc_pieces_per_cookie_l412_412055


namespace find_eccentricity_of_ellipse_l412_412119

-- Definition of the circle and intersection points with the x-axis
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_intersection_x_axis : set (ℝ × ℝ) := { p | p.1 ≠ 0 ∧ circle p.1 p.2 ∧ p.2 = 0 }

-- Definition of the parabola and its focus
def parabola (x y : ℝ) : Prop := y^2 = 10 * x
def parabola_focus : ℝ × ℝ := (5 / 2, 0)

-- Definition of the ellipse with its foci and vertex
def foci : set (ℝ × ℝ) := { (2, 0), (-2, 0) }
def vertex : ℝ × ℝ := (5 / 2, 0)

-- Calculation of the eccentricity of the ellipse
def c : ℝ := 2
def a : ℝ := 5 / 2
def eccentricity : ℝ := c / a

theorem find_eccentricity_of_ellipse : eccentricity = 4 / 5 := by
  sorry

end find_eccentricity_of_ellipse_l412_412119


namespace sum_ceil_sqrt_10_to_40_l412_412954

noncomputable def sumCeilSqrt : ℕ :=
  ∑ n in Finset.range (40 - 10 + 1), (Int.ceil ((n + 10 : ℝ) ^ (1 / 2)))

theorem sum_ceil_sqrt_10_to_40 : sumCeilSqrt = 167 := by
  sorry

end sum_ceil_sqrt_10_to_40_l412_412954


namespace minimize_total_time_l412_412377

-- Define the conditions for x, y, and z
def valid_speeds (x y z : ℝ) : Prop :=
  30 ≤ x ∧ x ≤ 60 ∧ 40 ≤ y ∧ y ≤ 70 ∧ 50 ≤ z ∧ z ≤ 90 ∧
  (60 / x + 90 / y + 200 / z ≤ 10)

-- Define the goal
theorem minimize_total_time :
  ∃ (x y z : ℝ), valid_speeds x y z ∧ (60 / x + 90 / y + 200 / z) = 4.51 := 
begin
  -- Placeholder for actual proof
  sorry
end

end minimize_total_time_l412_412377


namespace smallest_n_with_square_ending_in_2016_l412_412605

theorem smallest_n_with_square_ending_in_2016 : 
  ∃ n : ℕ, (n^2 % 10000 = 2016) ∧ (n = 996) :=
by
  sorry

end smallest_n_with_square_ending_in_2016_l412_412605


namespace rooks_arrangement_count_l412_412002

-- Define the chessboard as an 8x8 grid of numbers from 0 to 7
def Chessboard := matrix (fin 8) (fin 8) ℕ

-- Define the condition of a valid arrangement of 8 rooks
def valid_arrangement (pos : fin 8 → fin 8 × fin 8) : Prop :=
  (∀ i j, i ≠ j → pos i ≠ pos j) ∧  -- No two rooks in the same row or column
  ((∀ i, pos i.snd) = finset.univ)  -- All columns must be accounted for

-- The chessboard we are given
def chessboard : Chessboard :=
  ![![0, 1, 2, 3, 4, 5, 6, 7],
    ![0, 1, 2, 3, 4, 5, 6, 7],
    ![0, 1, 2, 3, 4, 5, 6, 7],
    ![0, 1, 2, 3, 4, 5, 6, 7],
    ![7, 6, 5, 4, 3, 2, 1, 0],
    ![7, 6, 5, 4, 3, 2, 1, 0],
    ![7, 6, 5, 4, 3, 2, 1, 0],
    ![7, 6, 5, 4, 3, 2, 1, 0]]

-- The main theorem to prove
theorem rooks_arrangement_count :
  (finset.filter valid_arrangement (finset.univ : finset (fin 8 → fin 8 × fin 8))).card = 3456 := by sorry

end rooks_arrangement_count_l412_412002


namespace coins_arrangement_l412_412329

noncomputable def num_distinguishable_arrangements : ℕ :=
  ∑ c in Finset.range 11, Nat.choose 10 4

theorem coins_arrangement :
  num_distinguishable_arrangements = 2310 := by
  sorry

end coins_arrangement_l412_412329


namespace work_completion_l412_412854

theorem work_completion (Rp Rq Dp W : ℕ) 
  (h1 : Rq = W / 12) 
  (h2 : W = 4*Rp + 6*(Rp + Rq)) 
  (h3 : Rp = W / Dp) 
  : Dp = 20 :=
by
  sorry

end work_completion_l412_412854


namespace distance_from_focus_to_line_l412_412695

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412695


namespace accommodation_arrangement_count_l412_412876

-- Definitions based on problem conditions
def tripleRoom : Type := unit
def doubleRoom : Type := unit
def adult : Type := unit
def child : Type := unit

-- There is only one triple room
constant availableTripleRoom : tripleRoom

-- There are two double rooms
constant availableDoubleRoom1 : doubleRoom
constant availableDoubleRoom2 : doubleRoom

-- The people
constant adult1 : adult
constant adult2 : adult
constant adult3 : adult
constant child1 : child
constant child2 : child

-- Conditions to specify that children cannot stay in a room alone
axiom children_not_alone (r : Type) (a b : r) (b≠a) (c : child) : false

-- The actual problem statement
theorem accommodation_arrangement_count : ∃ (n : ℕ), n = 60 :=
by
  -- Placeholder for the actual proof
  sorry

end accommodation_arrangement_count_l412_412876


namespace solution_l412_412058

noncomputable def problem : Prop :=
  (1/2)⁻² - Real.log 2 / Real.log 2 - Real.log 5 / Real.log 2 = 3

theorem solution : problem := 
by
  sorry

end solution_l412_412058


namespace avg_choc_pieces_per_cookie_l412_412056

theorem avg_choc_pieces_per_cookie {cookies chips mms pieces : ℕ} 
  (h1 : cookies = 48) 
  (h2 : chips = 108) 
  (h3 : mms = chips / 3) 
  (h4 : pieces = chips + mms) : 
  pieces / cookies = 3 := 
by sorry

end avg_choc_pieces_per_cookie_l412_412056


namespace average_of_six_numbers_l412_412823

theorem average_of_six_numbers :
  (∀ a b : ℝ, (a + b) / 2 = 6.2) →
  (∀ c d : ℝ, (c + d) / 2 = 6.1) →
  (∀ e f : ℝ, (e + f) / 2 = 6.9) →
  ((a + b + c + d + e + f) / 6 = 6.4) :=
by
  intros h1 h2 h3
  -- Proof goes here, but will be skipped with sorry.
  sorry

end average_of_six_numbers_l412_412823


namespace geometric_sequence_term_l412_412593

theorem geometric_sequence_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_seq : ∀ n, a (n+1) = a n * q)
  (h_a2 : a 2 = 8) 
  (h_a5 : a 5 = 64) : 
  a 3 = 16 := 
by 
  sorry

end geometric_sequence_term_l412_412593


namespace sum_of_prime_factors_of_3030_l412_412393

def n : ℕ := 3030

theorem sum_of_prime_factors_of_3030 : 
  ∑ p in (n.factorization.support : finset ℕ), p = 111 := by
  sorry

end sum_of_prime_factors_of_3030_l412_412393


namespace ticket_price_l412_412303

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l412_412303


namespace sequence_general_formula_l412_412168

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- because sequences in the solution are 1-indexed.
  | 1 => 2
  | k+2 => sequence (k+1) + 3 * (k+1)

theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : 
  sequence n = 2 + 3 * n * (n - 1) / 2 :=
by
  sorry

#eval sequence 1  -- should output 2
#eval sequence 2  -- should output 5
#eval sequence 3  -- should output 11
#eval sequence 4  -- should output 20
#eval sequence 5  -- should output 32
#eval sequence 6  -- should output 47

end sequence_general_formula_l412_412168


namespace tom_watching_days_l412_412826

noncomputable def total_watch_time : ℕ :=
  30 * 22 + 28 * 25 + 27 * 29 + 20 * 31 + 25 * 27 + 20 * 35

noncomputable def daily_watch_time : ℕ := 2 * 60

theorem tom_watching_days : ⌈(total_watch_time / daily_watch_time : ℚ)⌉ = 35 := by
  sorry

end tom_watching_days_l412_412826


namespace part_one_part_two_l412_412161

noncomputable def f (x a : ℝ) : ℝ := a + 1 / (4 ^ x + 1)

theorem part_one (h : ∀ x : ℝ, f x (-f (-x) a)) : a = -1/2 :=
  sorry

theorem part_two (a : ℝ) (ha : a = -1/2) :
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 a > f x2 a :=
  sorry

end part_one_part_two_l412_412161


namespace angle_A_when_frac_sum_max_l412_412261

-- Defining the variables and the conditions
variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to the angles A, B, and C

-- Condition: The altitude on side BC is a/2
def altitude_condition (h : a > 0) : Prop :=
  let h_a := a in
  h_a / (2:ℝ) = some (fun x =>
    ∃ (h_bc : ℝ), -- Introduce the altitude on BC
      h_bc = a / 2 ∧
      x = h_bc 
  )

-- Define the maximum value condition
def max_frac_condition (h : b > 0 ∧ c > 0) : Prop :=
  let frac_sum := (c / b) + (b / c) in
  frac_sum = some (fun x =>
    ∃ (max_val : ℝ),
      max_val = 2 * real.sqrt 2 ∧
      x = max_val
  )

-- Lean statement to prove angle A is π/4 when the fraction reaches its maximum
theorem angle_A_when_frac_sum_max (h1 : altitude_condition a) (h2 : max_frac_condition b c) :
  A = (real.pi / 4:ℝ) :=
sorry

end angle_A_when_frac_sum_max_l412_412261


namespace math_problem_l412_412559

variable {x y z n : ℕ}

theorem math_problem (hx : ℝ) (hy : ℝ) (hz : ℝ) (hn : ℕ) 
  (h1 : x + y + z = 1) 
  (h2 : arctan x + arctan y + arctan z = π / 4) : 
  x^(2n + 1) + y^(2n + 1) + z^(2n + 1) = 1 := 
sorry

end math_problem_l412_412559


namespace commutative_star_not_associative_star_l412_412990

variable (x y z k : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 0 < k)

noncomputable def star (a b : ℝ) : ℝ := (a * b + k) / (a + b)

theorem commutative_star : star x y = star y x := by
  sorry

theorem not_associative_star : star (star x y) z ≠ star x (star y z) := by
  sorry

end commutative_star_not_associative_star_l412_412990


namespace max_days_for_C_l412_412811

-- Define the durations of the processes and the total project duration
def A := 2
def B := 5
def D := 4
def T := 9

-- Define the condition to prove the maximum days required for process C
theorem max_days_for_C (x : ℕ) (h : 2 + x + 4 = 9) : x = 3 := by
  sorry

end max_days_for_C_l412_412811


namespace moles_KOH_combined_l412_412095

-- Define the number of moles of KI produced
def moles_KI_produced : ℕ := 3

-- Define the molar ratio from the balanced chemical equation
def molar_ratio_KOH_NH4I_KI : ℕ := 1

-- The number of moles of KOH combined to produce the given moles of KI
theorem moles_KOH_combined (moles_KOH moles_NH4I : ℕ) (h : moles_NH4I = 3) 
  (h_produced : moles_KI_produced = 3) (ratio : molar_ratio_KOH_NH4I_KI = 1) :
  moles_KOH = 3 :=
by {
  -- Placeholder for proof, use sorry to skip proving
  sorry
}

end moles_KOH_combined_l412_412095


namespace seq_increasing_for_n_eq_2_seq_not_increasing_for_n_ge_3_find_n_for_increasing_sequence_l412_412975

-- Define the sequence a_k for given n and k 
def sequence_lcm (n : ℕ) (k : ℕ) : ℕ :=
  Nat.lcm_list (List.range n).map (λ i, k + i)

-- Condition: n ≥ 2
def n_ge_two (n : ℕ) : Prop :=
  n ≥ 2

-- Prove that for n = 2, the sequence a_k is always increasing
theorem seq_increasing_for_n_eq_2 : sequence_lcm 2 (k : ℕ) < sequence_lcm 2 (k + 1) :=
sorry

-- Prove that for n ≥ 3, the sequence is not ultimately increasing
theorem seq_not_increasing_for_n_ge_3 (n k : ℕ) (h : n ≥ 3) : sequence_lcm n k ≥ sequence_lcm n (k + 1) :=
sorry

-- Final conclusion theorem combining both cases
theorem find_n_for_increasing_sequence (n : ℕ) : n = 2 ↔ ∀ k, sequence_lcm n k < sequence_lcm n (k + 1) :=
begin
  split,
  { intro h,
    rw h,
    exact seq_increasing_for_n_eq_2, },
  { intro h,
    sorry,  -- Prove by contradiction that if all k, seq_increasing then n must be 2
  },
end

end seq_increasing_for_n_eq_2_seq_not_increasing_for_n_ge_3_find_n_for_increasing_sequence_l412_412975


namespace number_of_cloth_bags_l412_412336

-- Definitions based on the conditions
def dozen := 12

def total_peaches : ℕ := 5 * dozen
def peaches_in_knapsack : ℕ := 12
def peaches_per_bag : ℕ := 2 * peaches_in_knapsack

-- The proof statement
theorem number_of_cloth_bags :
  (total_peaches - peaches_in_knapsack) / peaches_per_bag = 2 := by
  sorry

end number_of_cloth_bags_l412_412336


namespace real_solutions_x4_3_minus_x4_eq_98_l412_412485

theorem real_solutions_x4_3_minus_x4_eq_98 (x : ℝ) :
  x^4 + (3 - x)^4 = 98 ↔ x = 1.5 + real.sqrt ((real.sqrt 1189 - 27) / 4) ∨ x = 1.5 - real.sqrt ((real.sqrt 1189 - 27) / 4) :=
by
  sorry

end real_solutions_x4_3_minus_x4_eq_98_l412_412485


namespace proof_none_of_these_l412_412283

-- Definitions based on given conditions
def x (t : ℝ) (ht : t > 0 ∧ t ≠ 1) : ℝ := t^(2/(t-1))
def y (t : ℝ) (ht : t > 0 ∧ t ≠ 1) : ℝ := t^((2 * t - 1)/(t-1))

-- Setting the problem condition
theorem proof_none_of_these (t : ℝ) (ht : t > 0 ∧ t ≠ 1) : 
  ¬(y t ht)^((x t ht)) = (x t ht)^((y t ht)) ∧
  ¬((y t ht)^(1/(x t ht)) = (x t ht)^((y t ht))) ∧
  ¬((y t ht)/(x t ht) = (x t ht)/(y t ht)) ∧
  ¬((x t ht)^(-(x t ht)) = (y t ht)^(-(y t ht))) :=
by
  sorry

end proof_none_of_these_l412_412283


namespace count_non_integer_angles_l412_412630

open Int

def interior_angle (n : ℕ) : ℕ := 180 * (n - 2) / n

def is_integer_angle (n : ℕ) : Prop := 180 * (n - 2) % n = 0

theorem count_non_integer_angles : ∃ (count : ℕ), count = 2 ∧ ∀ n, 3 ≤ n ∧ n < 12 → is_integer_angle n ↔ ¬ (count = count + 1) :=
sorry

end count_non_integer_angles_l412_412630


namespace correct_statements_count_l412_412806

/--
The problem involves three transformations on a trigonometric function and assesses certain properties of the final function.
Conditions:
  1. f(x) = 2 * sin x * cos x + sqrt 3 * cos 2x
  2. g(x) = f(x - π / 3)
  3. h(x) = g(x / 2)
Goal: To show that the number of true statements among the following is equal to 2
  1. The smallest positive period of the function h(x) is 2π.
  2. (π / 3, 0) is a center of symmetry of the graph of h(x).
  3. The equation of a symmetry axis of the graph of the function h(x) is x=5π/6.
  4. The function h(x) is monotonically increasing on the interval [-π/24, 5π/24].
-/
theorem correct_statements_count :
  let f := λ x : ℝ, 2 * sin x * cos x + sqrt 3 * cos (2 * x)
  let g := λ x : ℝ, f (x - π / 3)
  let h := λ x : ℝ, g (x / 2)
  (number_of_correct_statements h = 2) := 
sorry

end correct_statements_count_l412_412806


namespace find_k_l412_412853

theorem find_k (k : ℝ) : (1/2)^16 * (1/81)^k = 1/18^16 → k = 8 :=
by sorry

end find_k_l412_412853


namespace xiao_ming_winning_probability_type1_given_winning_probability_l412_412907

open ProbabilityTheory

-- Defining the events
def A1 : Event := sorry
def A2 : Event := sorry
def A3 : Event := sorry
def B : Event := sorry

-- Given probabilities
def P_A1 : ℝ := 0.5
def P_A2 : ℝ := 0.25
def P_A3 : ℝ := 0.25
def P_B_given_A1 : ℝ := 0.3
def P_B_given_A2 : ℝ := 0.4
def P_B_given_A3 : ℝ := 0.5

-- Proof problem statement
theorem xiao_ming_winning_probability :
  P(B) = P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 := sorry

theorem type1_given_winning_probability :
  P(A1 | B) = (P_A1 * P_B_given_A1) / (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3) := sorry

end xiao_ming_winning_probability_type1_given_winning_probability_l412_412907


namespace distance_from_focus_to_line_l412_412732

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412732


namespace integer_values_count_l412_412989

theorem integer_values_count (x : ℤ) : 
  (∃ n : ℕ, n = 8 ∧ {x : ℤ | x^2 < 9 * x}.to_finset.card = n) :=
by
  sorry

end integer_values_count_l412_412989


namespace correct_options_B_and_C_l412_412226

variable (m n : Type) [Line m] [Line n]
variable (α β γ : Type) [Plane α] [Plane β] [Plane γ]

variable (L_parallel : ∀(x y : Type) [Line x] [Line y], Prop)
variable (L_perpendicular : ∀(x y : Type) [Line x] [Line y], Prop)
variable (P_perpendicular : ∀(π ρ : Type) [Plane π] [Plane ρ], Prop)

variable (m_parallel_β : L_parallel m β)
variable (m_perpendicular_α : L_perpendicular m α)
variable (m_parallel_α : L_parallel m α)
variable (n_perpendicular_α : L_perpendicular n α)

theorem correct_options_B_and_C :
  (P_perpendicular α β) ∧ (L_perpendicular m n) := by
sorry

end correct_options_B_and_C_l412_412226


namespace largest_integer_n_l412_412073

-- Define the floor function
def floor (x : ℝ) : ℤ := Int.ofNat ⌊x⌋

-- Define the problem statement
theorem largest_integer_n (n : ℕ) (h : floor (Real.sqrt n) = 5) : n ≤ 35 :=
by
  sorry

end largest_integer_n_l412_412073


namespace molecular_weight_correct_l412_412044

def atomic_weights : Type := (Al : ℝ := 26.98, S : ℝ := 32.07, O : ℝ := 16.00, H : ℝ := 1.01)

def Al2_SO4_3_18H2O_molecular_weight (aw : atomic_weights) : ℝ :=
  let Al_weight := 2 * aw.Al
  let SO4_weight := 3 * (aw.S + 4 * aw.O)
  let H2O_weight := 18 * (2 * aw.H + aw.O)
  Al_weight + SO4_weight + H2O_weight

theorem molecular_weight_correct : 
  Al2_SO4_3_18H2O_molecular_weight atomic_weights = 666.53 :=
by 
  unfold Al2_SO4_3_18H2O_molecular_weight
  -- Here we would show the calculations for each of the weights
  -- and their sums
  sorry

end molecular_weight_correct_l412_412044


namespace distance_from_right_focus_to_line_l412_412718

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412718


namespace largest_of_five_consecutive_integers_with_product_15120_eq_9_l412_412500

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end largest_of_five_consecutive_integers_with_product_15120_eq_9_l412_412500


namespace fraction_less_than_mode_is_one_third_l412_412594

def list_data : List ℕ := [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21]

noncomputable def mode (l : List ℕ) : ℕ :=
  let grouped := l.groupBy id
  let sorted := grouped.qsort (λ a b => (a.snd).length > (b.snd).length)
  (sorted.head?).map (λ g => g.fst) |>.getD 0

theorem fraction_less_than_mode_is_one_third : 
  let m := mode list_data
  let less_than_mode := list_data.filter (λ x => x < m)
  (less_than_mode.length : ℚ) / list_data.length = 1 / 3 :=
by
  sorry

end fraction_less_than_mode_is_one_third_l412_412594


namespace minimum_t_condition_l412_412287

noncomputable theory

-- Definitions for the conditions
def N (n : ℕ) : set ℕ := {i | 1 ≤ i ∧ i ≤ n}
def is_distinguishable (N : set ℕ) (F : set (set ℕ)) := ∀ (i j : ℕ), i ≠ j → i ∈ N → j ∈ N → ∃ (A : set ℕ) (H : A ∈ F), (A ∩ {i, j}).card = 1 
def is_covering (N : set ℕ) (F : set (set ℕ)) := ∀ i ∈ N, ∃ A ∈ F, i ∈ A

-- Minimum value of t
def min_t (n : ℕ) : ℕ := Nat.ceil (Real.log2 n.to_real) + 1

-- Formal statement of the problem
theorem minimum_t_condition (n : ℕ) (h : n ≥ 2) (F : set (set ℕ)) :
  (is_distinguishable (N n) F) → (is_covering (N n) F) → ∃ t, t = min_t n := sorry

end minimum_t_condition_l412_412287


namespace sum_ceil_sqrt_10_to_40_l412_412951

noncomputable def sumCeilSqrt : ℕ :=
  ∑ n in Finset.range (40 - 10 + 1), (Int.ceil ((n + 10 : ℝ) ^ (1 / 2)))

theorem sum_ceil_sqrt_10_to_40 : sumCeilSqrt = 167 := by
  sorry

end sum_ceil_sqrt_10_to_40_l412_412951


namespace sin2alpha_plus_sin2beta_eq_zero_l412_412977

theorem sin2alpha_plus_sin2beta_eq_zero
  (α β : ℝ)
  (h : sin α * sin β + cos α * cos β = 0) :
  sin (2 * α) + sin (2 * β) = 0 :=
sorry

end sin2alpha_plus_sin2beta_eq_zero_l412_412977


namespace geometric_series_first_term_l412_412032

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 24)
  (h_sum : S = a / (1 - r)) : 
  a = 18 :=
by {
  -- valid proof body goes here
  sorry
}

end geometric_series_first_term_l412_412032


namespace range_of_f_l412_412096

-- Define the function according to the given problem
def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 5)

-- Statement of the problem: Prove that the range of f is all real numbers except 3
theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
by
  -- Placeholder for proof, as we are only required to write the statement
  sorry

end range_of_f_l412_412096


namespace validate_assignment_l412_412449

-- Define the statements as conditions
def S1 := "x = x + 1"
def S2 := "b ="
def S3 := "x = y = 10"
def S4 := "x + y = 10"

-- A function to check if a statement is a valid assignment
def is_valid_assignment (s : String) : Prop :=
  s = S1

-- The theorem statement proving that S1 is the only valid assignment
theorem validate_assignment : is_valid_assignment S1 ∧
                              ¬is_valid_assignment S2 ∧
                              ¬is_valid_assignment S3 ∧
                              ¬is_valid_assignment S4 :=
by
  sorry

end validate_assignment_l412_412449


namespace sum_perimeters_eq_180_l412_412450

theorem sum_perimeters_eq_180 (side : ℕ) (h : side = 30) : 
  let P1 := 3 * side in 
  let series := λ n, P1 * (1/2) ^ n in 
  ∑' n, series n = 180 := 
by 
  sorry

end sum_perimeters_eq_180_l412_412450


namespace convert_speed_l412_412886

theorem convert_speed (s_m_s : ℝ) (conversion_factor : ℝ) : s_m_s = 70.0056 → conversion_factor = 3.6 → s_m_s * conversion_factor = 252.02016 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end convert_speed_l412_412886


namespace triangle_perimeter_l412_412968

-- Define the points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 3⟩
def B : Point := ⟨2, 9⟩
def C : Point := ⟨7, 6⟩

-- Define the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the perimeter function for the triangle formed by A, B, and C
def perimeter (A B C : Point) : ℝ :=
  distance A B + distance B C + distance C A

-- Prove the perimeter of triangle ABC is 6 + 2 * Real.sqrt 34
theorem triangle_perimeter : perimeter A B C = 6 + 2 * Real.sqrt 34 :=
  sorry

end triangle_perimeter_l412_412968


namespace distance_from_focus_to_line_l412_412693

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412693


namespace lindsey_september_savings_l412_412640

theorem lindsey_september_savings :
  ∃ S : ℕ, S + 37 + 11 + 25 = 36 + 87 ∧ S = 50 :=
by
  use 50
  sorry

end lindsey_september_savings_l412_412640


namespace product_eq_5832_l412_412992

-- Define the integers A, B, C, D that satisfy the given conditions.
variables (A B C D : ℕ)

-- Define the conditions in the problem.
def conditions : Prop :=
  (A + B + C + D = 48) ∧
  (A + 3 = B - 3) ∧
  (A + 3 = C * 3) ∧
  (A + 3 = D / 3)

-- State the final theorem we want to prove.
theorem product_eq_5832 : conditions A B C D → A * B * C * D = 5832 :=
by 
  sorry

end product_eq_5832_l412_412992


namespace avg_age_of_women_l412_412855

theorem avg_age_of_women (T : ℕ) (W : ℕ) (T_avg : ℕ) (H1 : T_avg = T / 10)
  (H2 : (T_avg + 6) = ((T - 18 - 22 + W) / 10)) : (W / 2) = 50 :=
sorry

end avg_age_of_women_l412_412855


namespace sin4_minus_cos4_eq_l412_412531

-- Given the condition
def alpha : ℝ := sorry
axiom h_sin : Real.sin alpha = Math.sqrt 5 / 5

-- The statement to prove
theorem sin4_minus_cos4_eq : Real.sin α ^ 4 - Real.cos α ^ 4 = - (3 / 5) :=
by sorry

end sin4_minus_cos4_eq_l412_412531


namespace sum_series_l412_412087

theorem sum_series : (List.sum [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56, -59]) = -30 :=
by
  sorry

end sum_series_l412_412087


namespace optimal_investment_allocation_l412_412452

noncomputable def max_profit_rate_A : ℝ := 1.0
noncomputable def max_profit_rate_B : ℝ := 0.5
noncomputable def max_loss_rate_A : ℝ := 0.3
noncomputable def max_loss_rate_B : ℝ := 0.1
noncomputable def investment_limit : ℝ := 100000
noncomputable def max_allowable_loss : ℝ := 18000

theorem optimal_investment_allocation 
  (investment_A investment_B : ℝ) :
  investment_A + investment_B ≤ investment_limit ∧
  (investment_A * max_loss_rate_A) + (investment_B * max_loss_rate_B) ≤ max_allowable_loss ∧
  (investment_A * max_profit_rate_A) + (investment_B * max_profit_rate_B) = 
  (40000 * max_profit_rate_A) + (60000 * max_profit_rate_B) :=
begin
  sorry
end

end optimal_investment_allocation_l412_412452


namespace conclusion1_conclusion2_l412_412900

theorem conclusion1 (x y a b : ℝ) (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b :=
sorry

theorem conclusion2 (x a : ℝ) (h1 : (x-1)*(x^2 + a*x + 1) - x^2 = x^3 - (a-1)*x^2 - (1-a)*x - 1) : a = 1 :=
sorry

end conclusion1_conclusion2_l412_412900


namespace find_persons_in_first_group_l412_412424

theorem find_persons_in_first_group :
  let work_1 := (P : ℕ) * 12 * 5,
      work_2 := 30 * 15 * 6
  in work_1 = work_2 → P = 45 :=
by
  intros P work_1 work_2 h
  have : P * 12 * 5 = 30 * 15 * 6 := h
  sorry

end find_persons_in_first_group_l412_412424


namespace S6_sum_geometric_sequence_l412_412149

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0 a q := a
| (n+1) a q := a * (1 - q^(n+1)) / (1 - q)

theorem S6_sum_geometric_sequence :
  ∃ (a1 a3 : ℝ) (q : ℝ),
    (a1 * a1 - 5 * a1 + 4 = 0) ∧
    (a3 * a3 - 5 * a3 + 4 = 0) ∧
    (a3 = a1 * q^2) ∧
    (0 < q) ∧
    (a1 = 1) ∧
    (a3 = 4) →
    geometric_sequence_sum 6 1 2 = 63 := sorry

end S6_sum_geometric_sequence_l412_412149


namespace distance_from_focus_to_line_l412_412733

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412733


namespace distance_from_right_focus_to_line_l412_412763

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412763


namespace atomic_weight_Ba_l412_412093

noncomputable def molecular_weight_BaSO4 : ℝ := 233
noncomputable def atomic_weight_S : ℝ := 32.07
noncomputable def atomic_weight_O : ℝ := 16.00

theorem atomic_weight_Ba : 
  let weight_S_and_O := atomic_weight_S + 4 * atomic_weight_O in
  molecular_weight_BaSO4 - weight_S_and_O = 136.93 := 
by
  have weight_S := atomic_weight_S
  have weight_O := 4 * atomic_weight_O
  have weight_S_and_O := weight_S + weight_O
  have molecular_weight_BaSO4 := molecular_weight_BaSO4
  have weight_Ba := molecular_weight_BaSO4 - weight_S_and_O
  show weight_Ba = 136.93
  sorry

end atomic_weight_Ba_l412_412093


namespace no_prime_divisible_by_45_l412_412187

theorem no_prime_divisible_by_45 : ∀ (p : ℕ), Prime p → ¬ (45 ∣ p) :=
by {
  intros p h_prime h_div,
  have h_factors := Nat.factors_unique,
  sorry
}

end no_prime_divisible_by_45_l412_412187


namespace distance_from_focus_to_line_l412_412710

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412710


namespace largest_of_five_consecutive_ints_15120_l412_412498

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end largest_of_five_consecutive_ints_15120_l412_412498


namespace number_of_male_students_drawn_l412_412422

theorem number_of_male_students_drawn (total_students : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) (sample_size : ℕ)
    (H1 : total_students = 350)
    (H2 : total_male_students = 70)
    (H3 : total_female_students = 280)
    (H4 : sample_size = 50) :
    total_male_students * sample_size / total_students = 10 :=
by
  sorry

end number_of_male_students_drawn_l412_412422


namespace probability_condition_l412_412346

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - x - 2

def condition (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 5
def event (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2

theorem probability_condition : 
  (∫ x in (-5 : ℝ)..5, if event x then 1 else 0) / (∫ x in (-5 : ℝ)..5, 1) = 3 / 10 :=
by
  -- Introduce expressions and domains
  sorry

end probability_condition_l412_412346


namespace intersecting_chords_length_l412_412554

theorem intersecting_chords_length
  (h1 : ∃ c1 c2 : ℝ, c1 = 8 ∧ c2 = x + 4 * x ∧ x = 2)
  (h2 : ∀ (a b c d : ℝ), a * b = c * d → a = 4 ∧ b = 4 ∧ c = x ∧ d = 4 * x ∧ x = 2):
  (10 : ℝ) = (x + 4 * x) := by
  sorry

end intersecting_chords_length_l412_412554


namespace sqrt_cos_induction_l412_412158

theorem sqrt_cos_induction (n : ℕ) (hn : 0 < n) :
  let S (n : ℕ) := nat.rec_on n (2 : ℝ) (λ _ sqrt_prev, real.sqrt (2 + sqrt_prev))
  S n = 2 * real.cos (real.pi / 2^(n + 1)) :=
by sorry

end sqrt_cos_induction_l412_412158


namespace min_value_x_fraction_l412_412999

theorem min_value_x_fraction (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ ∀ y > 1, y + 1 / (y - 1) ≥ m :=
by
  sorry

end min_value_x_fraction_l412_412999


namespace convert_3652_from_base7_to_base10_l412_412267

def base7ToBase10(n : ℕ) := 
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d0 * (7^0) + d1 * (7^1) + d2 * (7^2) + d3 * (7^3)

theorem convert_3652_from_base7_to_base10 : base7ToBase10 3652 = 1360 :=
by
  sorry

end convert_3652_from_base7_to_base10_l412_412267


namespace gcd_lcm_product_l412_412493

theorem gcd_lcm_product (a b : ℕ) (g : gcd a b) (l : lcm a b) :
  a = 180 → b = 250 →
  g = 2 * 5 →
  l = 2^2 * 3^2 * 5^3 →
  g * l = 45000 := by
  intros
  sorry

end gcd_lcm_product_l412_412493


namespace largest_integer_of_n_l412_412066

def floor (x : ℝ) := Int.floor x

theorem largest_integer_of_n (n : ℕ) (h : floor (Real.sqrt n) = 5) : n = 35 :=
by
  sorry

end largest_integer_of_n_l412_412066


namespace distance_from_focus_to_line_l412_412702

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412702


namespace sequence_a_100_eq_4_l412_412596

theorem sequence_a_100_eq_4 (a : ℕ → ℝ) 
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + real.log (1 + 1 / n)) : 
  a 100 = 4 := 
sorry

end sequence_a_100_eq_4_l412_412596


namespace find_line_eq_show_point_on_circle_l412_412123

noncomputable section

variables {x y x0 y0 : ℝ} (P Q : ℝ × ℝ) (h1 : y0 ≠ 0)
  (h2 : P = (x0, y0))
  (h3 : P.1^2/4 + P.2^2/3 = 1)
  (h4 : Q = (x0/4, y0/3))

theorem find_line_eq (M : ℝ × ℝ) (hM : ∀ (M : ℝ × ℝ), 
  ((P.1 - M.1) , (P.2 - M.2)) • (Q.1 , Q.2) = 0) :
  ∀ (x0 y0 : ℝ), y0 ≠ 0 → ∀ (x y : ℝ), 
  (x0 * x / 4 + y0 * y / 3 = 1) :=
by sorry
  
theorem show_point_on_circle (F S : ℝ × ℝ)
  (hF : F = (1, 0)) (hs : ∀ (x0 y0 : ℝ), y0 ≠ 0 → 
  S = (4, 0) ∧ ((S.1 - P.1) ^ 2 + (S.2 - P.2) ^ 2 = 36)) :
  ∀ (x y : ℝ), 
  (x - 1) ^ 2 + y ^ 2 = 36 := 
by sorry

end find_line_eq_show_point_on_circle_l412_412123


namespace tan_theta_values_l412_412117

theorem tan_theta_values (θ : ℝ) (h₁ : 0 < θ ∧ θ < Real.pi / 2) (h₂ : 12 / Real.sin θ + 12 / Real.cos θ = 35) : 
  Real.tan θ = 4 / 3 ∨ Real.tan θ = 3 / 4 := 
by
  sorry

end tan_theta_values_l412_412117


namespace distance_from_focus_to_line_l412_412692

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412692


namespace distance_from_right_focus_to_line_l412_412768

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412768


namespace next_term_in_geometric_sequence_l412_412835

theorem next_term_in_geometric_sequence (y : ℝ) : 
  let a := 3
  let r := 4*y 
  let t4 := 192*y^3 
  r * t4 = 768*y^4 :=
by
  sorry

end next_term_in_geometric_sequence_l412_412835


namespace distance_to_line_is_sqrt5_l412_412799

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412799


namespace range_of_a_l412_412136

theorem range_of_a (a : ℝ) (p q : ℝ → Prop) :
  (p = λ x, |4 - x| ≤ 6) →
  (q = λ x, x^2 - 2*x + 1 - a^2 ≥ 0) →
  (∀ x, (¬p x → q x)) →
  (0 < a ∧ a ≤ 3) :=
by
  intros hp hq implication
  sorry

end range_of_a_l412_412136


namespace scaled_polynomial_roots_l412_412290

noncomputable def polynomial_with_scaled_roots : Polynomial ℂ :=
  Polynomial.X^3 - 3*Polynomial.X^2 + 5

theorem scaled_polynomial_roots :
  (∃ r1 r2 r3 : ℂ, polynomial_with_scaled_roots.eval r1 = 0 ∧ polynomial_with_scaled_roots.eval r2 = 0 ∧ polynomial_with_scaled_roots.eval r3 = 0 ∧
  (∃ q : Polynomial ℂ, q = Polynomial.X^3 - 9*Polynomial.X^2 + 135 ∧
  ∀ y, (q.eval y = 0 ↔ (polynomial_with_scaled_roots.eval (y / 3) = 0)))) := sorry

end scaled_polynomial_roots_l412_412290


namespace tickets_whack_a_mole_l412_412458

variable (W : ℕ) -- W represents the number of tickets won playing 'whack a mole'
variable (T1 : ℕ) -- T1 represents the number of tickets won playing 'skee ball'
variable (S : ℕ) -- S represents the number of tickets spent on a hat
variable (L : ℕ) -- L represents the number of tickets left

-- Defining the known values from the conditions
def tickets_skee_ball : ℕ := 25
def tickets_spent_hat : ℕ := 7
def tickets_left : ℕ := 50

-- The main statement that proves the number of tickets won playing 'whack a mole'
theorem tickets_whack_a_mole : W + tickets_skee_ball = tickets_left + tickets_spent_hat → W = 32 := 
by {
  intro h,
  have h1 : W + 25 = 50 + 7 := h,
  have h2 : W + 25 = 57 := h1,
  have h3 : W = 57 - 25,
  exact Nat.sub_self_dec_triv 57 25 sorry,
  exact eq.symm h3
}

end tickets_whack_a_mole_l412_412458


namespace find_alpha_beta_l412_412961

theorem find_alpha_beta :
  ∃ (α β : ℝ), 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1233) / (x^2 + 81*x - 3969)) ∧
  α + β = 143 :=
by
  let α := 41
  let β := 102
  use α, β
  constructor
  · intro x
    have h₁ : x^2 - 72*x + 1233 = (x - 41)*(x - 31) := by
      ring
    have h₂ : x^2 + 81*x - 3969 = (x - 39)*(x + 102) := by
      ring
    calc
      (x - α) / (x + β) = (x - 41) / (x + 102)                     : by rfl
      ... = ((x - 41)*(x - 31)) / ((x - 39)*(x + 102))             : by 
        rw [←div_mul_eq_div]
        ring
      ... = (x^2 - 72*x + 1233) / (x^2 + 81*x - 3969)              : by
        rw [h₁, h₂]
  · exact rfl

end find_alpha_beta_l412_412961


namespace parallel_vectors_l412_412174

theorem parallel_vectors (x : ℝ) (k : ℝ) (a b : ℝ × ℝ) 
  (h₁ : a = (3, 1)) (h₂ : b = (x, -3)) (h₃ : ∃ k, a = k • b) : x = -9 :=
by
  -- Proof goes here
  sorry

end parallel_vectors_l412_412174


namespace find_g5_l412_412348
-- Import the Mathlib library

-- Define the function g and the conditions provided
variable {g : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, g(x * y) = g(x) * g(y))
variable (h2 : g(0) ≠ 0)

-- State the theorem to be proved
theorem find_g5 : g(5) = 1 :=
by
  -- Placeholder for proof steps
  sorry

end find_g5_l412_412348


namespace rectangle_area_l412_412358

theorem rectangle_area (x : ℕ) (h : 14 * x = 126) : 
  let length := 4 * x,
      width := 3 * x in 
  length * width = 972 := 
sorry

end rectangle_area_l412_412358


namespace least_positive_integer_divisibility_l412_412834

theorem least_positive_integer_divisibility :
  ∃ n > 1, (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_divisibility_l412_412834


namespace probability_bernaldo_lt_silvia_l412_412459

def bernardo_numbers := {n : Finset ℕ | n ⊆ {1, 2, 3, 4, 5, 6, 7} ∧ n.card = 3 ∧ ∀ a b c, a ∈ n → b ∈ n → c ∈ n → a < b → b < c}
def silvia_numbers := {n : Finset ℕ | n ⊆ {2, 3, 4, 5, 6, 7, 8, 9} ∧ n.card = 3 ∧ ∀ a b c, a ∈ n → b ∈ n → c ∈ n → a > b → b > c}

def bernardo_number (n : Finset ℕ) : ℕ := n.val.foldl (+) 0 -- Just a placeholder for some function generating numbers
def silvia_number (n : Finset ℕ) : ℕ := n.val.foldl (+) 0  -- Just a placeholder for some function generating numbers

theorem probability_bernaldo_lt_silvia : 
  let b := bernardo_numbers,
      s := silvia_numbers in
  (∃ n₁ ∈ s, ∀ n₂ ∈ b, bernardo_number n₂ < silvia_number n₁) → 1 = 1 :=
by
  sorry

end probability_bernaldo_lt_silvia_l412_412459


namespace distance_to_line_is_sqrt5_l412_412797

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412797


namespace inequality_am_gm_l412_412323

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (1 / a + 1 / b + 1 / c) ≤ ((a^8 + b^8 + c^8) / (a^3 * b^3 * c^3)) :=
begin
  sorry
end

end inequality_am_gm_l412_412323


namespace evaluate_sum_l412_412944

theorem evaluate_sum :
  (∑ n in finset.range 31, (let k := n + 10 in
    if 10 ≤ k ∧ k ≤ 16 then 4 else
    if 17 ≤ k ∧ k ≤ 25 then 5 else
    if 26 ≤ k ∧ k ≤ 36 then 6 else
    if 37 ≤ k ∧ k ≤ 40 then 7 else 0)) = 167 :=
by sorry

end evaluate_sum_l412_412944


namespace cos_triple_angle_l412_412219

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l412_412219


namespace gcd_lcm_product_l412_412492

theorem gcd_lcm_product (a b : ℕ) (g : gcd a b) (l : lcm a b) :
  a = 180 → b = 250 →
  g = 2 * 5 →
  l = 2^2 * 3^2 * 5^3 →
  g * l = 45000 := by
  intros
  sorry

end gcd_lcm_product_l412_412492


namespace distance_planes_eq_distance_lines_l412_412145

variables {α β : Plane} {m n : Line}
variables (d1 d2 : ℝ)

-- Conditions
axiom parallel_planes : ∀ {α β : Plane}, parallel α β
axiom line_in_plane_m : ∀ {m : Line} {α : Plane}, contains α m
axiom line_in_plane_n : ∀ {n : Line} {β : Plane}, contains β n
axiom non_parallel_lines : ∀ {m n : Line}, ¬ parallel m n
axiom distance_planes_d1 : ∀ {α β : Plane}, distance α β = d1
axiom distance_lines_d2 : ∀ {m n : Line}, distance m n = d2

-- Proof goal
theorem distance_planes_eq_distance_lines :
  parallel_planes α β → 
  line_in_plane_m m α → line_in_plane_n n β → 
  non_parallel_lines m n → 
  distance_planes_d1 α β = distance_lines_d2 m n :=
by sorry

end distance_planes_eq_distance_lines_l412_412145


namespace father_current_age_l412_412426

namespace AgeProof

def daughter_age : ℕ := 10
def years_future : ℕ := 20

def father_age (D : ℕ) : ℕ := 4 * D

theorem father_current_age :
  ∀ D : ℕ, ∀ F : ℕ, (F = father_age D) →
  (F + years_future = 2 * (D + years_future)) →
  D = daughter_age →
  F = 40 :=
by
  intro D F h1 h2 h3
  sorry

end AgeProof

end father_current_age_l412_412426


namespace probability_product_multiple_of_4_l412_412300

def rolls (n : ℕ) : set ℕ := {i | i ∈ finset.range n}

def multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def probability_event (event : set ℕ) (total : ℕ) : ℚ :=
  (event.card : ℚ) / total

theorem probability_product_multiple_of_4 :
  let Liam_possibilities := rolls 12 in
  let Maya_possibilities := rolls 6 in
  let Liam_multiple_4 := {n ∈ Liam_possibilities | multiple_of_4 n} in
  let Maya_multiple_4 := {n ∈ Maya_possibilities | multiple_of_4 n} in
  let event := (Liam_multiple_4 ×ˢ Maya_possibilities) ∪ (Liam_possibilities ×ˢ Maya_multiple_4) ∪
    ({2} ×ˢ {2}) in  -- Pairing where both roll a 2 remains
  probability_event event (12 * 6) = 47 / 72 :=
by
  sorry

end probability_product_multiple_of_4_l412_412300


namespace original_flour_quantity_l412_412643

-- Definitions based on conditions
def flour_called (x : ℝ) : Prop := 
  -- total flour Mary uses is x + extra 2 cups, which equals to 9 cups.
  x + 2 = 9

-- The proof statement we need to show
theorem original_flour_quantity : ∃ x : ℝ, flour_called x ∧ x = 7 := 
  sorry

end original_flour_quantity_l412_412643


namespace pressure_force_half_ellipse_l412_412412

-- Define the conditions
def ellipse (a b x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the surface area element
def dS (a b x : ℝ) : ℝ := 
  (b / a) * Real.sqrt (a^2 - x^2)

-- Define the infinitesimal pressure force
def dF (ρ g a b x : ℝ) : ℝ :=
  ρ * g * x * dS a b x

-- Define the integral expression for total force
def total_force (ρ g a b : ℝ) : ℝ :=
  2 * ∫ x in 0..a, dF ρ g a b x

-- Prove that the total force is equal to the given expression
theorem pressure_force_half_ellipse (ρ g a b : ℝ) : 
  total_force ρ g a b = (2 * b * a^2 / 3) * ρ * g :=
by
  sorry

end pressure_force_half_ellipse_l412_412412


namespace probability_at_least_one_female_probability_all_males_given_at_least_one_male_l412_412892

-- Definitions and conditions
def total_volunteers := 7
def male_volunteers := 4
def female_volunteers := 3
def selected_volunteers := 3

-- The number of ways to choose 3 out of 7 volunteers
def total_combinations := Nat.choose total_volunteers selected_volunteers

-- The number of ways to choose 3 males out of 4 male volunteers
def male_combinations := Nat.choose male_volunteers selected_volunteers

-- Definition of the event of selecting all males, and related probabilities
def P_all_males := (male_combinations : ℝ) / (total_combinations : ℝ)

-- Statements to prove
theorem probability_at_least_one_female :
  1 - P_all_males = 31 / 35 :=
by sorry

-- Probability of at least one male volunteer being selected
def female_only_combinations : ℝ := Nat.choose female_volunteers selected_volunteers

def P_at_least_one_male :=
  1 - (female_only_combinations / total_combinations)

-- Conditional probability of selecting all males given at least one male
theorem probability_all_males_given_at_least_one_male :
  P_all_males / P_at_least_one_male = 2 / 17 :=
by sorry

end probability_at_least_one_female_probability_all_males_given_at_least_one_male_l412_412892


namespace number_of_piles_l412_412310

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end number_of_piles_l412_412310


namespace find_isosceles_triangle_points_l412_412527

theorem find_isosceles_triangle_points :
  ∃ X₁ X₂ X₃ X₄ X₅ : Point,
  (∀ X : Point,
    (|distance B X = distance C X ∨ distance B X = distance A X) ∧ 
    distance A X = distance B X) ↔
    X = X₁ ∨ X = X₂ ∨ X = X₃ ∨ X = X₄ ∨ X = X₅ :=
sorry

end find_isosceles_triangle_points_l412_412527


namespace ratio_of_apple_to_orange_cost_l412_412460

-- Define the costs of fruits based on the given conditions.
def cost_per_kg_oranges : ℝ := 12
def cost_per_kg_apples : ℝ := 2

-- The theorem to prove.
theorem ratio_of_apple_to_orange_cost : cost_per_kg_apples / cost_per_kg_oranges = 1 / 6 :=
by
  sorry

end ratio_of_apple_to_orange_cost_l412_412460


namespace ten_by_ten_checkerboard_squares_l412_412060

noncomputable def count_squares (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), (n - k) * (n - k)

theorem ten_by_ten_checkerboard_squares : count_squares 10 = 385 :=
  by sorry

end ten_by_ten_checkerboard_squares_l412_412060


namespace solve_for_y_l412_412230

theorem solve_for_y (x y : ℝ) (h1 : x ^ (2 * y) = 16) (h2 : x = 2) : y = 2 :=
by {
  sorry
}

end solve_for_y_l412_412230


namespace intersection_of_A_and_B_l412_412110

noncomputable def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
noncomputable def B : Set ℝ := { x | 0 ≤ x }

theorem intersection_of_A_and_B :
  { x | x ∈ A ∧ x ∈ B } = { x | 0 ≤ x ∧ x ≤ 3 } :=
  by sorry

end intersection_of_A_and_B_l412_412110


namespace remaining_number_l412_412353

theorem remaining_number (S : Finset ℕ) (hS : S = Finset.range 51) :
  ∃ n ∈ S, n % 2 = 0 := 
sorry

end remaining_number_l412_412353


namespace distance_from_focus_to_line_l412_412749

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412749


namespace distance_from_right_focus_to_line_l412_412721

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412721


namespace symmetric_circle_eq_l412_412231

theorem symmetric_circle_eq :
  \{\forall P : \mathbb{R}^2, (P.x + 2)^2 + (P.y - 1)^2 = 1 \leftrightarrow (P'.x - 2)^2 + (P'.y + 1)^2 = 1\} :=
by
  sorry

end symmetric_circle_eq_l412_412231


namespace distance_from_hyperbola_focus_to_line_l412_412758

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412758


namespace instantaneous_velocity_at_one_l412_412453

theorem instantaneous_velocity_at_one :
  let h (t : ℝ) := -4.9 * t^2 + 10 * t
  in deriv h 1 = 0.2 :=
by
  sorry

end instantaneous_velocity_at_one_l412_412453


namespace sum_of_sequence_eq_six_seventeenth_l412_412271

noncomputable def cn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.cos (n * Real.arctan (2 / 3))
noncomputable def dn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.sin (n * Real.arctan (2 / 3))

theorem sum_of_sequence_eq_six_seventeenth : 
  (∑' n : ℕ, (cn n * dn n / 8^n)) = 6/17 := sorry

end sum_of_sequence_eq_six_seventeenth_l412_412271


namespace floor_lg_sum_eq_92_l412_412620

def floor_lg_sum : ℕ :=
  ∑ n in Finset.range 101, Int.floor (Real.log n / Real.log 10)

theorem floor_lg_sum_eq_92 : floor_lg_sum = 92 :=
  sorry

end floor_lg_sum_eq_92_l412_412620


namespace distance_from_focus_to_line_l412_412700

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412700


namespace part1_part2_l412_412169

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else 
  let Sn := ∑ i in Finset.range n, a i
  in Sn^2 / (Sn - 1/2)

noncomputable def S (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  ∑ i in Finset.range n, a (i + 1)

def b (n : ℕ) : ℝ := S n / (2 * n + 1)

def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)

theorem part1 (n : ℕ) (h : n > 0) : 
  S n = 1 / (2 * n - 1) := 
sorry

theorem part2 (n : ℕ) : 
  T n < 1 / 2 := 
sorry

end part1_part2_l412_412169


namespace star_value_l412_412925

def star (A B : ℝ) : ℝ := (A + B) / 4

theorem star_value : star (star 3 15) 7 = 2.875 :=
  by
    sorry

end star_value_l412_412925


namespace average_chocolate_pieces_per_cookie_l412_412054

-- Definitions from the conditions
def number_of_cookies := 48
def number_of_chocolate_chips := 108
def number_of_m_and_ms := (1 / 3 : ℝ) * number_of_chocolate_chips
def total_number_of_chocolate_pieces := number_of_chocolate_chips + number_of_m_and_ms

-- Statement to prove
theorem average_chocolate_pieces_per_cookie : 
  total_number_of_chocolate_pieces / number_of_cookies = 3 := by
  sorry

end average_chocolate_pieces_per_cookie_l412_412054


namespace geometric_seq_sum_l412_412361

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q))
    (h2 : S 10 = 10) (h3 : S 30 = 70) (hq_pos : 0 < q) :
    S 40 = 150 := by
  sorry

end geometric_seq_sum_l412_412361


namespace polygon_sides_eq_ten_l412_412013

theorem polygon_sides_eq_ten (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) 
  (h2 : ∑ (fun i : ℕ, if i < n then 180 else 0), i from 0 to n-1 = (n - 2) * 180)
  (h3 : ∑ (fun i : ℕ, if i < n then 360 else 0), i from 0 to n-1 = 360) :
  n = 10 :=
begin
  sorry
end

end polygon_sides_eq_ten_l412_412013


namespace cos_triple_angle_l412_412203

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l412_412203


namespace sum_of_reciprocals_l412_412818

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 1 / x + 1 / y = 8 / 75 :=
by
  sorry

end sum_of_reciprocals_l412_412818


namespace smallest_x_for_multiple_l412_412391

theorem smallest_x_for_multiple (x : ℕ) (h₁: 450 = 2 * 3^2 * 5^2) (h₂: 800 = 2^6 * 5^2) : 
  ((450 * x) % 800 = 0) ↔ x ≥ 32 :=
by
  sorry

end smallest_x_for_multiple_l412_412391


namespace distance_from_focus_to_line_l412_412716

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412716


namespace distance_from_focus_to_line_l412_412687

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412687


namespace number_of_subsets_of_set_A_l412_412352

theorem number_of_subsets_of_set_A : 
  (finite (set.univ : set (fin 3)) ∧ fintype.card (set (fin 3)) = 8) :=
begin
  split,
  { apply set.finite_univ },
  { apply finset.card,
    sorry }
end

end number_of_subsets_of_set_A_l412_412352


namespace henry_correct_answers_l412_412583

theorem henry_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 :=
by
  sorry

end henry_correct_answers_l412_412583


namespace sum_of_fractions_l412_412913

theorem sum_of_fractions :
  (2 / 8) + (4 / 8) + (6 / 8) + (8 / 8) + (10 / 8) + 
  (12 / 8) + (14 / 8) + (16 / 8) + (18 / 8) + (20 / 8) = 13.75 :=
by sorry

end sum_of_fractions_l412_412913


namespace arithmetic_sequence_second_term_l412_412569

theorem arithmetic_sequence_second_term (S₃: ℕ) (a₁: ℕ) (h1: S₃ = 9) (h2: a₁ = 1) : 
∃ d a₂, 3 * a₁ + 3 * d = S₃ ∧ a₂ = a₁ + d ∧ a₂ = 3 :=
by
  sorry

end arithmetic_sequence_second_term_l412_412569


namespace simplify_expression_l412_412672

variable {x y : ℝ}

theorem simplify_expression : (x^5 * x^3 * y^2 * y^4) = (x^8 * y^6) := by
  sorry

end simplify_expression_l412_412672


namespace factorization_correct_l412_412957

theorem factorization_correct (x y : ℝ) :
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2) * (x^2 - 3*y + 2) :=
by
  sorry

end factorization_correct_l412_412957


namespace f_5_value_l412_412294

noncomputable def f : ℕ → ℝ
| 1 := 2
| (n+1) := (2 * f n + n) / 2

theorem f_5_value : f 5 = 7 := by
  sorry

end f_5_value_l412_412294


namespace chi_squared_confidence_l412_412029

theorem chi_squared_confidence {X Y : Type} (χ2_value : ℝ) :
  let χ2_distribution := sorry -- Define the chi squared distribution related to X and Y
  (χ2_value > some_threshold_XY) → (confidence_relation_XY χ2_value > some_confidence_level) :=
sorry

end chi_squared_confidence_l412_412029


namespace problem_lean_statement_l412_412047

lemma perfect_square_trinomial (a b : ℕ) : 
  a^2 + 2 * a * b + b^2 = (a + b)^2 := by
  sorry

theorem problem_lean_statement : 
  17^2 + 2 * 17 * 5 + 5^2 = 484 := by
  have h : 17^2 + 2 * 17 * 5 + 5^2 = (17 + 5)^2 := perfect_square_trinomial 17 5
  rw add_comm at h
  rw nat.pow_two at h
  rw nat.pow_two at h
  exact h

end problem_lean_statement_l412_412047


namespace range_f_side_length_c_l412_412112

noncomputable def m (x : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.sin x, 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * real.cos x, real.cos x ^ 2)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Theorem for the first part: Range of f(x)
theorem range_f : set.range f = set.Icc (-1) 3 := sorry

-- Definitions and theorems needed for the second part
noncomputable def triangle_a := 2
noncomputable def triangle_b (c : ℝ) : ℝ := 2 * c
noncomputable def triangle_cosA := 1/2
noncomputable def triangle_fA : ℝ := 2

-- Theorem for the side length c in triangle ABC
theorem side_length_c (c : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (hA : A = real.pi / 3) (hB : real.sin B = 2 * real.sin C) : 
  ∃ c : ℝ, c = 2 * real.sqrt 3 / 3 := sorry

end range_f_side_length_c_l412_412112


namespace find_angle_A_l412_412600

variable {a b c A B C : ℝ}

theorem find_angle_A (h1 : sin B - sin C = sin (A - C)) (h2 : A + B + C = π) :
  A = π / 3 :=
sorry

end find_angle_A_l412_412600


namespace integral_of_f_l412_412164

noncomputable def f : ℝ → ℝ
| x := if x ∈ Icc (-real.pi) 0 then sin x else if x ∈ Icc 0 1 then sqrt (1 - x^2) else 0

theorem integral_of_f :
  ∫ x in Icc (-real.pi) 1, f x = real.pi / 4 - 2 :=
by
  sorry

end integral_of_f_l412_412164


namespace min_cars_with_all_characteristics_l412_412819

-- Definitions from the problem conditions
def N : ℕ := 20
def A : ℕ := 14
def B : ℕ := 15
def C : ℕ := 17
def D : ℕ := 18

-- The statement to prove the minimum number of cars with all 4 characteristics
theorem min_cars_with_all_characteristics : ∃ x, x ≤ N ∧ x = N - ((N - A) + (N - B) + (N - C) + (N - D)) :=
by {
  use 4,
  sorry
}

end min_cars_with_all_characteristics_l412_412819


namespace uniqueness_of_fuction_function_solution_l412_412628

def T := { pqr : ℕ × ℕ × ℕ | true }

noncomputable def f : (ℕ × ℕ × ℕ) → ℝ
| (p, q, r) :=
  if p * q * r = 0 then 0 else 
    1 + (1 / 6) * (f(p + 1, q - 1, r) + f(p - 1, q + 1, r) +
                   f(p - 1, q, r + 1) + f(p + 1, q, r - 1) +
                   f(p, q + 1, r - 1) + f(p, q - 1, r + 1))

theorem uniqueness_of_fuction : 
  ∀ (p q r : ℕ), f(p, q, r) = if p * q * r = 0 then 0 else 
    1 + (1 / 6) * (f(p + 1, q - 1, r) + f(p - 1, q + 1, r) +
                   f(p - 1, q, r + 1) + f(p + 1, q, r - 1) +
                   f(p, q + 1, r - 1) + f(p, q - 1, r + 1))
                    :=
begin
  sorry
end

theorem function_solution : 
  ∀ (p q r : ℕ), f(p, q, r) = if p + q + r = 0 then 0 
                               else (3 * p * q * r) / (p + q + r):=
begin
  sorry
end

end uniqueness_of_fuction_function_solution_l412_412628


namespace parabola_vertex_x_coordinate_l412_412430

theorem parabola_vertex_x_coordinate 
  (a b c : ℝ)
  (h1 : 16 = a * 2^2 + b * 2 + c)
  (h2 : 16 = a * 8^2 + b * 8 + c)
  (h3 : 25 = a * 10^2 + b * 10 + c) :
  ∃ x_v : ℝ, x_v = 5 :=
begin
  use 5,
  sorry
end

end parabola_vertex_x_coordinate_l412_412430


namespace geometric_to_arithmetic_seq_l412_412108

theorem geometric_to_arithmetic_seq (a q : ℕ)
  (ha : a = 2) (hq : q = 3) :
  let seq := [a, a * q, a * q^2, a * q^3],
      new_seq := [a, a * q + 4, a * q^2, a * q^3 - 28] in
  seq = [2, 6, 18, 54] ∧
  (new_seq[1] - new_seq[0] = new_seq[2] - new_seq[1] ∧ new_seq[2] - new_seq[1] = new_seq[3] - new_seq[2]) :=
by
  sorry

end geometric_to_arithmetic_seq_l412_412108


namespace largest_integer_n_l412_412070

def floor (x : ℝ) := ⌊x⌋

theorem largest_integer_n (n : ℕ) (h : floor (real.sqrt n) = 5) : n ≤ 35 :=
by {
  sorry
}

end largest_integer_n_l412_412070


namespace smallest_positive_period_l412_412081

-- Define the function y
def y (x : ℝ) : ℝ := 2 * Real.cos (π / 5 + 3 * x)

-- State the theorem to prove the smallest positive period is 2π/3
theorem smallest_positive_period : ∃ T > 0, ∀ x : ℝ, y (x + T) = y x ∧ T = 2 * π / 3 :=
by
  sorry

end smallest_positive_period_l412_412081


namespace sum_of_tangent_points_l412_412061

noncomputable def f (x : ℝ) : ℝ := 
  max (max (-7 * x - 19) (3 * x - 1)) (5 * x + 3)

theorem sum_of_tangent_points :
  ∃ x4 x5 x6 : ℝ, 
  (∃ q : ℝ → ℝ, 
    (∀ x, q x = f x ∨ (q x - (-7 * x - 19)) = b * (x - x4)^2
    ∨ (q x - (3 * x - 1)) = b * (x - x5)^2 
    ∨ (q x - (5 * x + 3)) = b * (x - x6)^2)) ∧
  x4 + x5 + x6 = -3.2 :=
sorry

end sum_of_tangent_points_l412_412061


namespace sequence_sum_l412_412914

theorem sequence_sum :
  (∑ i in (finset.range 7), (if i % 2 = 0 then 3 + i * 10 else -(3 + (i - 1) * 10))) + 73 = 38 := by
  sorry

end sequence_sum_l412_412914


namespace cut_flow_equivalence_l412_412906

variable {N : Type*} [Network N] (S : Set N) (s : N) (f : N → N → ℝ)

-- assuming cuts and required properties for the cut in network
def is_cut (S : Set N) (N : Type*) : Prop := sorry
def f_function (S : Set N) (V : Set N) (f : N → N → ℝ) : ℝ := sorry

theorem cut_flow_equivalence (S : Set N) (s : N) (f : N → N → ℝ) :
  is_cut S N → f_function S S f = 0 → f_function S ⊤ f = f_function {s} ⊤ f := sorry

end cut_flow_equivalence_l412_412906


namespace cos_triplet_angle_l412_412211

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l412_412211


namespace problem_l412_412636

noncomputable def condition1 (ξ : ℕ → ℝ) : Prop :=
  ∃ f : ℕ → ℝ, (∀ n, f n = o ((λ n, (ξ n)^2) : ℕ → ℝ)) ∧ ∀ n, f n = (λ n, D (ξ n))

noncomputable def condition2 (ξ : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n > N, E (ξ n) > ε

theorem problem (ξ : ℕ → ℝ) (h_condition1 : condition1 ξ) (h_condition2 : condition2 ξ) :
  ∀ c : ℝ, ∀ ε > 0, ∃ N, ∀ n > N, P (ξ n ≥ c) > 1 - ε :=
sorry

end problem_l412_412636


namespace compute_A_plus_B_plus_C_l412_412591

theorem compute_A_plus_B_plus_C (M N : ℝ) (a b c : ℕ)
  (hM : M = 5)
  (hN : N = 12)
  (hA : ∃ (a b c : ℕ), A = a + b * Real.sqrt c ∧ c.prime^2 ∣ c = false ∧ a + b + c = 36) :
  a + b + c = 36 :=
by
  sorry

end compute_A_plus_B_plus_C_l412_412591


namespace base_ten_to_base_four_253_l412_412384

def base_four (n : ℕ) : ℕ :=
  let rec convert (n : ℕ) (pow : ℕ) : ℕ :=
    if n = 0 then 0
    else let d := n / pow in
         let r := n % pow in
         d * pow + convert r (pow / 4)
  convert n (4^3)

theorem base_ten_to_base_four_253 :
  base_four 253 = 3331 :=
by
  -- mathematical details skipped for brevity
  sorry

end base_ten_to_base_four_253_l412_412384


namespace rectangle_ratio_l412_412431

theorem rectangle_ratio 
    (b c : ℝ)
    (b_pos : b > 0)
    (c_pos : c > 0)
    (E_on_CD : ∃ λ : ℝ, E = (λ, c) ∧ 0 ≤ λ ∧ λ ≤ b)
    (isosceles_DBE : ∀ (x y : ℝ), (0, c) = (x, c) ∧ (b, 0) = (y, 0) → sqrt ((x - 0)^2 + (c - c)^2) = sqrt((x - b)^2 + (c - 0)^2))
    (right_angle_ABE : ∀ (x y : ℝ), (0, 0) = (x, 0) ∧ (b, 0) = (y, 0) → ⟨γ1, γ0⟩ = λ ∧ 0 < γ1 ∧ γ1 ≤ b → (c / λ) * (c / (λ - b)) = -1) : 
  (c / b) = Real.sqrt (Real.sqrt  5  - 2) := sorry

end rectangle_ratio_l412_412431


namespace task_completion_time_l412_412268

noncomputable def john_work_rate := (1: ℚ) / 20
noncomputable def jane_work_rate := (1: ℚ) / 12
noncomputable def combined_work_rate := john_work_rate + jane_work_rate
noncomputable def time_jane_disposed := 4

theorem task_completion_time :
  (∃ x : ℚ, (combined_work_rate * x + john_work_rate * time_jane_disposed = 1) ∧ (x + time_jane_disposed = 10)) :=
by
  use 6  
  sorry

end task_completion_time_l412_412268


namespace distance_from_focus_to_line_l412_412686

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412686


namespace iron_weight_l412_412659

theorem iron_weight 
  (A : ℝ) (hA : A = 0.83) 
  (I : ℝ) (hI : I = A + 10.33) : 
  I = 11.16 := 
by 
  sorry

end iron_weight_l412_412659


namespace reflection_matrix_correct_l412_412276

def normal_vector : Vect3 := ⟨2, -1, 1⟩

noncomputable def reflection_matrix : Matrix 3 3 ℚ := 
⟦[⟨-2/3, -4/3, 7/3⟩, ⟨4/3, -2/3, 4/3⟩, ⟨-5/3, 10/3, 5/3⟩]⟧

theorem reflection_matrix_correct (v : Vect3) : 
  let Rv := reflection_matrix.mul v 
  Rv = ⟨reflection_vector⟩ :=
sorry

end reflection_matrix_correct_l412_412276


namespace selling_price_in_august_correct_cost_price_correct_l412_412646

-- Define the conditions
def sales_in_july : ℝ := 22500
def sales_in_august : ℝ := 25000
def price_increase_aug_per_july : ℝ := 100
def sales_sep_perc_decrease : ℝ := 0.15
def store_profit_perc_sep : ℝ := 0.25

-- Define the selling price per bike in August and the cost price of each mountain bike
noncomputable def price_per_bike_in_august : ℝ := 1000
noncomputable def cost_price_per_bike : ℝ := 680

-- Prove that the selling price per bike in August is correct
theorem selling_price_in_august_correct :
    let p_july := price_per_bike_in_august - 100 in
    (sales_in_july / p_july = sales_in_august / price_per_bike_in_august) →
    price_per_bike_in_august = 1000 :=
by
    intros h
    sorry

-- Prove that the cost price of each mountain bike is correct
theorem cost_price_correct :
    let price_sep := price_per_bike_in_august * (1 - sales_sep_perc_decrease) in
    (price_sep = 850) →
    (850 - cost_price_per_bike) / cost_price_per_bike = store_profit_perc_sep →
    cost_price_per_bike = 680 :=
by
    intros h1 h2
    sorry

end selling_price_in_august_correct_cost_price_correct_l412_412646


namespace number_of_days_A_to_finish_remaining_work_l412_412009

theorem number_of_days_A_to_finish_remaining_work
  (A_days : ℕ) (B_days : ℕ) (B_work_days : ℕ) : 
  A_days = 9 → 
  B_days = 15 → 
  B_work_days = 10 → 
  ∃ d : ℕ, d = 3 :=
by 
  intros hA hB hBw
  sorry

end number_of_days_A_to_finish_remaining_work_l412_412009


namespace find_ab_pairs_l412_412483

theorem find_ab_pairs (a b : ℝ) : (a + b - 1)^2 = a^2 + b^2 - 1 ↔ (a = 1 ∧ b ∈ set.univ) ∨ (b = 1 ∧ a ∈ set.univ) :=
by
  sorry

end find_ab_pairs_l412_412483


namespace number_of_perfect_square_factors_l412_412928

theorem number_of_perfect_square_factors 
  (a b c d : ℕ)
  (h1 : a = 8)
  (h2 : b = 9)
  (h3 : c = 12)
  (h4 : d = 4) :
  (λ x1 x2 x3 x4 : ℕ, x1 * x2 * x3 * x4) (5) (5) (7) (3) = 525 :=
by
  -- conditions ensure that the number of perfect square factors is given by
  -- multiplying the number of factors for each prime power
  have fact1 : 5 = 5 := rfl,
  have fact2 : 5 = 5 := rfl,
  have fact3 : 7 = 7 := rfl,
  have fact4 : 3 = 3 := rfl,
  sorry

end number_of_perfect_square_factors_l412_412928


namespace distance_from_focus_to_line_l412_412690

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412690


namespace coefficient_of_x_squared_in_binomial_expansion_l412_412156

theorem coefficient_of_x_squared_in_binomial_expansion :
  let expr := (x - 1/x)^6 in
  (binomial_expansion_coefficient expr 2 = 15) :=
by
  sorry

end coefficient_of_x_squared_in_binomial_expansion_l412_412156


namespace amount_paid_l412_412308

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l412_412308


namespace num_int_values_satisfying_inequality_l412_412982

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l412_412982


namespace lucas_total_pages_l412_412642

-- Define the variables and conditions
def lucas_read_pages : Nat :=
  let pages_first_four_days := 4 * 20
  let pages_break_day := 0
  let pages_next_four_days := 4 * 30
  let pages_last_day := 15
  pages_first_four_days + pages_break_day + pages_next_four_days + pages_last_day

-- State the theorem
theorem lucas_total_pages :
  lucas_read_pages = 215 :=
sorry

end lucas_total_pages_l412_412642


namespace find_lambda_l412_412111

-- Declare the vectors a and b and the value lambda
variables (λ : ℝ)

-- Conditions: a = (2,1) and b = (-4,λ)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-4, λ)

-- The condition that a is parallel to b
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- The theorem statement asserting that the condition implies λ = -2
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = -2 :=
by
  sorry

end find_lambda_l412_412111


namespace set_intersection_l412_412170

theorem set_intersection (M N : Set ℝ) 
  (hM : M = {x | 2 * x - 3 < 1}) 
  (hN : N = {x | -1 < x ∧ x < 3}) : 
  (M ∩ N) = {x | -1 < x ∧ x < 2} := 
by 
  sorry

end set_intersection_l412_412170


namespace equilateral_triangle_coloring_l412_412451

theorem equilateral_triangle_coloring :
  ∀ (ABC : Triangle) (n : ℕ)
    (H1 : is_equilateral ABC)
    (H2 : num_subdivisions ABC n)
    (H3 : n = 9000000)
    (H4 : ∀ (v: Vertex), (color v = red ∨ color v = blue ∨ color v = green)),
  ∃ (a b c : Vertex), (color a = color b) ∧ (color b = color c) ∧ (is_parallel_triangle ABC a b c) :=
begin
  sorry
end

end equilateral_triangle_coloring_l412_412451


namespace product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l412_412236

theorem product_div_sum_eq_5 (x : ℤ) (h : (x^3 - x) / (3 * x) = 5) : x = 4 := by
  sorry

theorem quotient_integer_condition (x : ℤ) : ((∃ k : ℤ, x = 3 * k + 1) ∨ (∃ k : ℤ, x = 3 * k - 1)) ↔ ∃ q : ℤ, (x^3 - x) / (3 * x) = q := by
  sorry

theorem next_consecutive_set (x : ℤ) (h : x = 4) : x - 1 = 3 ∧ x = 4 ∧ x + 1 = 5 := by
  sorry

end product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l412_412236


namespace pinecone_count_is_180_l412_412674

noncomputable def pinecone_count : ℕ :=
  if h : (∃ n : ℕ,
    n < 350 ∧
    n % 2 = 0 ∧
    n % 3 = 0 ∧
    n % 4 = 0 ∧
    n % 5 = 0 ∧
    n % 6 = 0 ∧
    n % 7 ≠ 0 ∧
    n % 8 ≠ 0 ∧
    n % 9 = 0 ∧
    ∀ m : ℕ, (m % 2 = 0 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 ∧ m % 9 = 0 → m = n))
  then Classical.some h
  else 0

theorem pinecone_count_is_180 : pinecone_count = 180 :=
by
  sorry

end pinecone_count_is_180_l412_412674


namespace trailing_zeros_800_factorial_l412_412471

theorem trailing_zeros_800_factorial : 
  let num_trailing_zeros (n : ℕ) := (n / 5) + (n / 25) + (n / 125) + (n / 625) in
  num_trailing_zeros 800 = 199 :=
by
  sorry

end trailing_zeros_800_factorial_l412_412471


namespace locus_center_circle_l412_412521

open Real

theorem locus_center_circle (a x y : ℝ) (h₁ : (x, y) ∉ ((λ c, (0, a)).range)) 
  (h₂ : ∃ (C : ℝ × ℝ), ∀ (M N : ℝ × ℝ), M.1 = 0 ∧ N.1 = 0 ∧ (M.2 * N.2)^(0.5) = 2 * a -> || (C.2) || = sqrt(C.1^2 + (C.2 - a)^2)) :
  (∃ (locus : ℝ × ℝ), ∀ (C : ℝ × ℝ), locus = (C.1^2 = 2*a*C.2)) := 
sorry

end locus_center_circle_l412_412521


namespace find_m_l412_412227

theorem find_m (m : ℝ) : (1 ^ 2 + m * 1 - 6 = 0) → (m = 5) :=
by
  assume h : 1 ^ 2 + m * 1 - 6 = 0
  sorry

end find_m_l412_412227


namespace distance_from_focus_to_line_l412_412745

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412745


namespace vasya_sum_impossible_l412_412649

theorem vasya_sum_impossible (n : ℕ) (hn : n = 25) (pages : Fin n → Nat)
  (page_sum : Fin n → Nat)
  (numbered_from_one_to_n_hundred_n_ninety_two : ∀ p, p ∈ pages → p ≤ 192)
  (odd_evens_pairs : ∀ p, p ∈ pages → (∃ m, p = 2 * m + 1 ∨ p = 2 * (m + 1)))
  (sum_of_pages : ∑ i in finset.range n, page_sum i = 1998) : False :=
sorry

end vasya_sum_impossible_l412_412649


namespace optimal_purchasing_plan_l412_412011

def price_carnation := 5
def price_lily := 10
def total_flowers := 300
def max_carnations (x : ℕ) : Prop := x ≤ 2 * (total_flowers - x)

theorem optimal_purchasing_plan :
  ∃ (x y : ℕ), (x + y = total_flowers) ∧ (x = 200) ∧ (y = 100) ∧ (max_carnations x) ∧ 
  ∀ (x' y' : ℕ), (x' + y' = total_flowers) → max_carnations x' →
    (price_carnation * x + price_lily * y ≤ price_carnation * x' + price_lily * y') :=
by
  sorry

end optimal_purchasing_plan_l412_412011


namespace find_range_norm_b_l412_412555

open Real

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

-- Given conditions
variables {a b : ℝ}
variables (ha : ∥a∥ = 2) (h : inner (2 • a + b) b = 12)

-- Goal
theorem find_range_norm_b : ∥b∥ ≥ 2 ∧ ∥b∥ ≤ 6 :=
sorry

end find_range_norm_b_l412_412555


namespace missy_total_patients_l412_412650

theorem missy_total_patients 
  (P : ℕ)
  (h1 : ∀ x, (∃ y, y = ↑(1/3) * ↑x) → ∃ z, z = y * (120/100))
  (h2 : ∀ x, 5 * x = 5 * (x - ↑(1/3) * ↑x) + (120/100) * 5 * (↑(1/3) * ↑x))
  (h3 : 64 = 5 * (2/3) * (P : ℕ) + 6 * (1/3) * (P : ℕ)) :
  P = 12 :=
by
  sorry

end missy_total_patients_l412_412650


namespace emily_paid_for_each_skirt_l412_412086

theorem emily_paid_for_each_skirt
  (art_supplies_cost : ℤ)
  (num_skirts : ℤ)
  (shoe_original_price : ℤ)
  (shoe_discount_percent : ℝ)
  (total_spent : ℤ)
  (h1 : art_supplies_cost = 20)
  (h2 : num_skirts = 2)
  (h3 : shoe_original_price = 30)
  (h4 : shoe_discount_percent = 0.15)
  (h5 : total_spent = 50) :
  let shoe_discount := shoe_discount_percent * ↑shoe_original_price in
  let shoe_discount_price := shoe_original_price - shoe_discount in
  let total_spent_on_arts_and_shoes := art_supplies_cost + shoe_discount_price in
  let total_spent_on_skirts := total_spent - total_spent_on_arts_and_shoes in
  total_spent_on_skirts / num_skirts = (9 / 4) := by
  sorry

end emily_paid_for_each_skirt_l412_412086


namespace community_B_low_income_families_selected_l412_412871

theorem community_B_low_income_families_selected (A B C total_units : ℕ) 
    (hA : A = 360) 
    (hB : B = 270) 
    (hC : C = 180) 
    (h_total_units: total_units = 90)
    (h_total : A + B + C = 810) : 
  (total_units * B) / (A + B + C) = 30 :=
by
  rw [hA, hB, hC, h_total_units, h_total]
  simp
  norm_num
  sorry

end community_B_low_income_families_selected_l412_412871


namespace uniqueIdentityFunction_l412_412089

noncomputable def solveFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 
  (z + 1) * f (x + y) = f (x * f(z) + y) + f (y * f(z) + x)

theorem uniqueIdentityFunction : 
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  solveFunctionalEquation f →
  (∀ x, x > 0 → f x = x) :=
by 
  intros f f_pos f_solution
  sorry

end uniqueIdentityFunction_l412_412089


namespace distance_from_focus_to_line_l412_412735

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412735


namespace _l412_412655

variables (A B : Prop)
variables (P_A_hits : Prop) (P_B_hits : Prop)

-- Define the probabilities as given in the conditions.
def P_A_hits := 0.6
def P_B_hits := 0.5

-- Define the probability that the target is hit by at least one shooter.
def P_Hit := 1 - (1 - P_A_hits) * (1 - P_B_hits)

-- The theorem we want to prove.
example : P_Hit = 0.8 :=
by {
    -- sorry is used to skip the proof
    sorry,
}

end _l412_412655


namespace pentagon_condition_l412_412446

noncomputable def exists_pentagon_segments (a b c d e : ℝ) : Prop :=
  (a + b + c + d + e = 2) ∧ (a < 1) ∧ (b < 1) ∧ (c < 1) ∧ (d < 1) ∧ (e < 1)

theorem pentagon_condition (a b c d e : ℝ) (h : exists_pentagon_segments a b c d e) :
  a + b + c + d + e = 2 ∧ a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ e < 1 :=
by
  cases h,
  assumption

end pentagon_condition_l412_412446


namespace maximum_valid_subset_size_l412_412994

-- Definitions and conditions
def M : Set ℕ := {1, 2, ... 2008}
def valid_subset (A : Set ℕ) : Prop :=
  ∀ a b ∈ A, a ≠ b → ¬ (a + b) % Nat.gcd (a, b) = 0

-- Statement to be proven
theorem maximum_valid_subset_size :
  ∃ (A : Set ℕ), A ⊆ M ∧ valid_subset A ∧ A.card = 670 := sorry

end maximum_valid_subset_size_l412_412994


namespace partitionable_implies_not_mod3_eq1_l412_412548

noncomputable def partitionable (k : ℤ) : Prop :=
∃ (A B C : set ℤ), 
  (∀ x ∈ A, x ∈ {3^31, 3^31 + 1, ..., 3^31 + k}) ∧
  (∀ x ∈ B, x ∈ {3^31, 3^31 + 1, ..., 3^31 + k}) ∧
  (∀ x ∈ C, x ∈ {3^31, 3^31 + 1, ..., 3^31 + k}) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ∪ B ∪ C = {3^31, 3^31 + 1, ..., 3^31 + k}) ∧
  (∑ x in A, x = ∑ x in B, x ∧ ∑ x in B, x = ∑ x in C, x)

theorem partitionable_implies_not_mod3_eq1 (k : ℤ) 
  (h : partitionable k) 
  : k % 3 ≠ 1 := sorry

end partitionable_implies_not_mod3_eq1_l412_412548


namespace find_first_reappearance_l412_412808

theorem find_first_reappearance : 
  (let letters_cycle := 8
  let digits_cycle := 4
  Nat.lcm letters_cycle digits_cycle = 8) := 
begin
  sorry
end

end find_first_reappearance_l412_412808


namespace complex_conjugate_problem_l412_412541

noncomputable def conjugate_of_z (z : ℂ) (h : (z + complex.i) * (2 + complex.i) = 5) : ℂ :=
  complex.conj z

theorem complex_conjugate_problem (z : ℂ) (h : (z + complex.i) * (2 + complex.i) = 5) :
  conjugate_of_z z h = 2 + 2 * complex.i :=
sorry

end complex_conjugate_problem_l412_412541


namespace distance_from_focus_to_line_l412_412737

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412737


namespace range_of_a_l412_412980

theorem range_of_a
  (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 6)
  (y : ℝ) (hy : 0 < y)
  (h : (y / 4 - 2 * (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) :
  a ≤ 3 :=
sorry

end range_of_a_l412_412980


namespace cylinder_volume_l412_412829

-- Definitions based on conditions
def lateral_surface_to_rectangle (generatrix_a generatrix_b : ℝ) (volume : ℝ) :=
  -- Condition: Rectangle with sides 8π and 4π
  (generatrix_a = 8 * Real.pi ∧ volume = 32 * Real.pi^2) ∨
  (generatrix_a = 4 * Real.pi ∧ volume = 64 * Real.pi^2)

-- Statement
theorem cylinder_volume (generatrix_a generatrix_b : ℝ)
  (h : (generatrix_a = 8 * Real.pi ∨ generatrix_b = 4 * Real.pi) ∧ (generatrix_b = 4 * Real.pi ∨ generatrix_b = 8 * Real.pi)) :
  ∃ (volume : ℝ), lateral_surface_to_rectangle generatrix_a generatrix_b volume :=
sorry

end cylinder_volume_l412_412829


namespace dihedral_angle_square_fold_l412_412525

open EuclideanGeometry

/--
Given a square ABCD folded along its diagonal AC to form a triangle ADC, with β being the
angle between AD and the plane ABC. Prove that when β reaches its maximum value,
the size of the dihedral angle B-AC-D is 90°.
-/
theorem dihedral_angle_square_fold (A B C D: Point) (hAB: A ≠ B) (hBC: B ≠ C) (hCD: C ≠ D) (hDA: D ≠ A) 
  (hACfold: folded_along_diagonal A B C D (line_through A C)) (triangleADC: is_triangle A D C) 
  (planeABC: is_plane A B C) (planeADC: is_plane A D C) (β: angle (line_through A D) planeABC) :
  max_angle β → dihedral_angle B (line_through A C) D = 90 :=
by
  sorry

end dihedral_angle_square_fold_l412_412525


namespace find_number_l412_412396

theorem find_number (x : ℕ) :
  ((4 * x) / 8 = 6) ∧ ((4 * x) % 8 = 4) → x = 13 :=
by
  sorry

end find_number_l412_412396


namespace F_range_d2_minus_r2_exists_circle_M_l412_412522

-- Define the circle data
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ
  -- Conditions
  eq_cond : D^2 + E^2 = F^2
  F_pos : F > 0

def radius (C : Circle) : ℝ := 
  sqrt (C.D^2 + C.E^2 - 4 * C.F) / 2

def distance (C : Circle) (l : ℝ → ℝ) : ℝ :=
  abs (C.F - 2) / 2

-- Proof Problem 1: Prove range of F
theorem F_range (C : Circle) : C.F > 4 :=
by sorry

-- Proof Problem 2: Prove d^2 - r^2 = 1
theorem d2_minus_r2 (C : Circle) : 
    (distance C id)^2 - (radius C)^2 = 1 :=
by sorry

-- Proof Problem 3: There exists a circle M: x^2 + y^2 = 1 
theorem exists_circle_M (C : Circle) : ∃ (l : ℝ → ℝ), 
  l = id ∧ Circle where
    D := 0
    E := 0
    F := 1
      .eq_cond := by sorry 
      .F_pos := by sorry :=
by sorry

end F_range_d2_minus_r2_exists_circle_M_l412_412522


namespace inscribed_square_ratio_l412_412935

theorem inscribed_square_ratio (a : ℝ) (h : 0 < a) : 
  let side_large := a in
  let side_inscribed := a / 2 in
  let area_large := side_large^2 in
  let area_inscribed := side_inscribed^2 in
  area_inscribed / area_large = 1 / 4 :=
by
  sorry

end inscribed_square_ratio_l412_412935


namespace distance_from_right_focus_to_line_l412_412724

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412724


namespace probability_correct_arrangement_l412_412354

-- Definitions for conditions
def characters := {c : String | c = "医" ∨ c = "国"}

def valid_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"], ["国", "医", "医"]}

def correct_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"]}

-- Theorem statement
theorem probability_correct_arrangement :
  (correct_arrangements.card : ℚ) / (valid_arrangements.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_correct_arrangement_l412_412354


namespace factory_minimize_salary_l412_412874

theorem factory_minimize_salary :
  ∃ x : ℕ, ∃ W : ℕ,
    x + (120 - x) = 120 ∧
    800 * x + 1000 * (120 - x) = W ∧
    120 - x ≥ 3 * x ∧
    x = 30 ∧
    W = 114000 :=
  sorry

end factory_minimize_salary_l412_412874


namespace num_ways_to_place_balls_l412_412246

noncomputable def num_ways_place_balls : Nat :=
  let empty_spots := 4.factorial
  let choose_rows := Nat.choose 4 2
  let valid_configs := 2 + 12 + 4 + 4 + 12 + 2
  empty_spots * choose_rows * valid_configs

theorem num_ways_to_place_balls :
  num_ways_place_balls = 5184 :=
by
  unfold num_ways_place_balls
  rw [Nat.factorial, Nat.choose]
  -- Detailed steps to expand factorial and choose can go here
  -- But ultimately, we know:
  -- 4! = 24, Nat.choose 4 2 = 6, and valid_configs = 36
  -- Hence:
  exact dec_trivial

end num_ways_to_place_balls_l412_412246


namespace base_500_in_base_has_six_digits_l412_412413

theorem base_500_in_base_has_six_digits (b : ℕ) : b^5 ≤ 500 ∧ 500 < b^6 ↔ b = 3 := 
by
  sorry

end base_500_in_base_has_six_digits_l412_412413


namespace minimum_value_l412_412967

noncomputable def expr (x y : ℝ) := x^2 + x * y + y^2 - 3 * y

theorem minimum_value :
  ∃ x y : ℝ, expr x y = -3 ∧
  ∀ x' y' : ℝ, expr x' y' ≥ -3 :=
sorry

end minimum_value_l412_412967


namespace real_solutions_x4_3_minus_x4_eq_98_l412_412484

theorem real_solutions_x4_3_minus_x4_eq_98 (x : ℝ) :
  x^4 + (3 - x)^4 = 98 ↔ x = 1.5 + real.sqrt ((real.sqrt 1189 - 27) / 4) ∨ x = 1.5 - real.sqrt ((real.sqrt 1189 - 27) / 4) :=
by
  sorry

end real_solutions_x4_3_minus_x4_eq_98_l412_412484


namespace distance_from_right_focus_to_line_l412_412725

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412725


namespace spring_problem_l412_412478

theorem spring_problem (x y : ℝ) : 
  (∀ x, y = 0.5 * x + 12) →
  (0.5 * 3 + 12 = 13.5) ∧
  (y = 0.5 * x + 12) ∧
  (0.5 * 5.5 + 12 = 14.75) ∧
  (20 = 0.5 * 16 + 12) :=
by 
  sorry

end spring_problem_l412_412478


namespace juice_drinks_costs_2_l412_412381

-- Define the conditions and the proof problem
theorem juice_drinks_costs_2 (given_amount : ℕ) (amount_returned : ℕ) 
                            (pizza_cost : ℕ) (number_of_pizzas : ℕ) 
                            (number_of_juice_packs : ℕ) 
                            (total_spent_on_juice : ℕ) (cost_per_pack : ℕ) 
                            (h1 : given_amount = 50) (h2 : amount_returned = 22)
                            (h3 : pizza_cost = 12) (h4 : number_of_pizzas = 2)
                            (h5 : number_of_juice_packs = 2) 
                            (h6 : given_amount - amount_returned - number_of_pizzas * pizza_cost = total_spent_on_juice) 
                            (h7 : total_spent_on_juice / number_of_juice_packs = cost_per_pack) : 
                            cost_per_pack = 2 := by
  sorry

end juice_drinks_costs_2_l412_412381


namespace sum_int_values_l412_412392

theorem sum_int_values (sum : ℤ) : 
  (∀ n : ℤ, (20 % (2 * n - 1) = 0) → sum = 2) :=
by
  sorry

end sum_int_values_l412_412392


namespace goose_egg_count_l412_412652

theorem goose_egg_count
  (E : ℕ)
  (h1 : (1/3 : ℚ) * E.natAbs > 0)
  (h2 : (3/4 : ℚ) * ((1/3 : ℚ) * E.natAbs) > 0)
  (h3 : (2/5 : ℚ) * ((3/4 : ℚ) * ((1/3 : ℚ) * E.natAbs)) = 120) :
  E = 1200 := sorry

end goose_egg_count_l412_412652


namespace three_cards_different_suits_probability_l412_412666

-- Define the conditions and problem
noncomputable def prob_three_cards_diff_suits : ℚ :=
  have first_card_options := 52
  have second_card_options := 39
  have third_card_options := 26
  have total_ways_to_pick := (52 : ℕ) * (51 : ℕ) * (50 : ℕ)
  (39 / 51) * (26 / 50)

-- State our proof problem
theorem three_cards_different_suits_probability :
  prob_three_cards_diff_suits = 169 / 425 :=
sorry

end three_cards_different_suits_probability_l412_412666


namespace evaluate_expression_l412_412050

theorem evaluate_expression :
  (1 / 2)⁻¹ + (1 / 4)^0 - 9^(1 / 2) = 0 :=
by
  have h1 : (1 / 2)⁻¹ = 2 := by -- this is a simple reciprocal calculation
    sorry
  have h2 : (1 / 4)^0 = 1 := by -- this is any number raised to the 0 power, which is 1
    sorry
  have h3 : -9^(1 / 2) = -3 := by -- negative of the square root of 9
    sorry
  rw [h1, h2, h3]
  exact sub_self (2 + 1)

end evaluate_expression_l412_412050


namespace min_period_f_range_f_measure_angle_B_l412_412295

def f (x : ℝ) : ℝ := 6 * (Real.cos x)^2 - 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + 2

theorem min_period_f : Real.MinPeriod f = Real.pi := by
  sorry

theorem range_f : Set.Icc (5 - 2 * Real.sqrt 3) (5 + 2 * Real.sqrt 3) = f '' Set.I := by
  sorry

theorem measure_angle_B (B : ℝ) (h : f B = 2) (hB : 0 < B ∧ B < Real.pi / 2) : B = Real.pi / 3 := by
  sorry

end min_period_f_range_f_measure_angle_B_l412_412295


namespace project_completion_l412_412867

theorem project_completion (A B : ℕ) (ha: A = 20) (hb: B = 30) (quit_time: ℕ := 10) : 
    ∃ (days: ℕ), ((days - quit_time) / A) + (days / B) = 0.5 ∧ days = 14 := by
  sorry

end project_completion_l412_412867


namespace inequality_am_gm_l412_412324

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  6*a + 4*b + 5*c ≥ 5*sqrt(a*b) + 7*sqrt(a*c) + 3*sqrt(b*c) :=
sorry

end inequality_am_gm_l412_412324


namespace hexagon_angle_B_l412_412586

theorem hexagon_angle_B (N U M B E S : ℝ) (h1 : N = M) (h2 : M = B) (h3 : U + S = 180) : B = 135 := by
  have hsum : N + U + M + B + E + S = 720 := by
    sorry
  have hM : 3 * B + E + 180 = 720 := by
    rw [h1, h2, hsum]
    sorry
  have h4B : 4 * B = 540 := by
    sorry
  have h_res : B = 135 := by
    linarith
  exact h_res

end hexagon_angle_B_l412_412586


namespace distance_from_right_focus_to_line_l412_412722

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412722


namespace parabola_reflection_translation_result_l412_412879

theorem parabola_reflection_translation_result (a h k : ℝ) (ha : a ≠ 0) :
  let f := λ x : ℝ, a * (x - (h + 5))^2 + k in
  let g := λ x : ℝ, -a * (x - (h - 5))^2 - k in
  ∃ m b : ℝ,  (λ x, f x + g x) = λ x, m * x + b := 
sorry

end parabola_reflection_translation_result_l412_412879


namespace smallest_real_number_l412_412030

theorem smallest_real_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = sqrt 2 / 2) (h3 : c = sqrt 3 / 3) (h4 : d = sqrt 5 / 5) : d < a ∧ d < b ∧ d < c :=
by sorry

end smallest_real_number_l412_412030


namespace eq_of_fraction_eq_l412_412262

variable {R : Type*} [Field R]

theorem eq_of_fraction_eq (a b : R) (h : (1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b))) : a = b :=
sorry

end eq_of_fraction_eq_l412_412262


namespace largest_integer_of_n_l412_412067

def floor (x : ℝ) := Int.floor x

theorem largest_integer_of_n (n : ℕ) (h : floor (Real.sqrt n) = 5) : n = 35 :=
by
  sorry

end largest_integer_of_n_l412_412067


namespace sugar_concentration_after_adding_water_l412_412448

def initial_mass_of_sugar_water : ℝ := 90
def initial_sugar_concentration : ℝ := 0.10
def final_sugar_concentration : ℝ := 0.08
def mass_of_water_added : ℝ := 22.5

theorem sugar_concentration_after_adding_water 
  (m_sugar_water : ℝ := initial_mass_of_sugar_water)
  (c_initial : ℝ := initial_sugar_concentration)
  (c_final : ℝ := final_sugar_concentration)
  (m_water_added : ℝ := mass_of_water_added) :
  (m_sugar_water * c_initial = (m_sugar_water + m_water_added) * c_final) := 
sorry

end sugar_concentration_after_adding_water_l412_412448


namespace complete_residue_system_l412_412132

theorem complete_residue_system {m n : ℕ} {a : ℕ → ℕ} {b : ℕ → ℕ}
  (h₁ : ∀ i j, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ n → (a i) * (b j) % (m * n) ≠ (a i) * (b j)) :
  (∀ i₁ i₂, 1 ≤ i₁ → i₁ ≤ m → 1 ≤ i₂ → i₂ ≤ m → i₁ ≠ i₂ → (a i₁ % m ≠ a i₂ % m)) ∧ 
  (∀ j₁ j₂, 1 ≤ j₁ → j₁ ≤ n → 1 ≤ j₂ → j₂ ≤ n → j₁ ≠ j₂ → (b j₁ % n ≠ b j₂ % n)) := sorry

end complete_residue_system_l412_412132


namespace prove_statement_l412_412454

noncomputable def problem_statement : Prop :=
  let b_set := { b : ℤ | -6 ≤ b ∧ b ≤ 6 }
  let c_set := { c : ℤ | -6 ≤ c ∧ c ≤ 6 }
  let total_pairs := finset.card (b_set ×ˢ c_set)
  let valid_pairs := finset.card { p : ℤ × ℤ | let (b, c) := p in b ∈ b_set ∧ c ∈ c_set ∧ b^2 - 4 * c < 0 }
  valid_pairs = 90 ∧ total_pairs = 169 ∧ (valid_pairs:ℚ) / total_pairs = 90 / 169

theorem prove_statement : problem_statement := 
by {
  sorry
}

end prove_statement_l412_412454


namespace incircle_radius_of_DEF_l412_412378

-- Define the right triangle with given conditions
structure Triangle :=
  (DF DE EF : ℝ)
  (right_angle_at_F : (EF^2 + DE^2 = DF^2))
  (angle_E : ∠DEF = 45 * π / 180)

-- Inradius of a triangle given the sides and semiperimeter
noncomputable def inradius (t : Triangle) : ℝ := 
  let s := (t.DF + t.DE + t.EF) / 2 in
  let area := sqrt (s * (s - t.DF) * (s - t.DE) * (s - t.EF)) in
  area / s

-- Define the specific triangle DEF and verify the conditions
def DEF_incircle_radius : ℝ :=
  let DF := 8 in
  let DE := 8 in
  let EF := 8 * sqrt 2 in
  let s := (DF + DE + EF) / 2 in
  let area := 0.5 * DF * DE in
  area / s

-- The theorem we need to prove
theorem incircle_radius_of_DEF :
  DEF_incircle_radius = 8 - 4 * sqrt 2 :=
by
  -- placeholder - proof to be provided
  sorry

end incircle_radius_of_DEF_l412_412378


namespace yard_length_l412_412822

theorem yard_length (n_trees : ℕ) (d : ℕ) (n_trees = 18) (d = 15) : (n_trees - 1) * d = 255 :=
by
  -- proof goes here
  sorry

end yard_length_l412_412822


namespace amount_paid_l412_412307

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end amount_paid_l412_412307


namespace one_statement_is_true_l412_412317

theorem one_statement_is_true :
  ∃ (S1 S2 S3 S4 S5 : Prop),
    ((S1 ↔ (¬S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5)) ∧
     (S2 ↔ (¬S1 ∧ ¬S2 ∧ S3 ∧ S4 ∧ ¬S5)) ∧
     (S3 ↔ (¬S1 ∧ S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S4 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S5 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4 ∧ ¬S5))) ∧
    (S2) ∧ (¬S1) ∧ (¬S3) ∧ (¬S4) ∧ (¬S5) :=
by
  -- Proof goes here
  sorry

end one_statement_is_true_l412_412317


namespace average_cost_per_pencil_l412_412031

-- Define the problem conditions
def number_of_pencils : ℕ := 150
def cost_of_pencils : ℚ := 22.50
def shipping_cost : ℚ := 7.50

-- Define the total cost in cents
def total_cost_cents : ℚ := (cost_of_pencils + shipping_cost) * 100

-- The goal is to prove the average cost per pencil in cents
theorem average_cost_per_pencil : (total_cost_cents / number_of_pencils).round = 20 := by
sorry

end average_cost_per_pencil_l412_412031


namespace distance_from_focus_to_line_l412_412713

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412713


namespace optimal_tower_configuration_l412_412590

theorem optimal_tower_configuration (x y : ℕ) (h : x + 2 * y = 30) :
    x * y ≤ 112 := by
  sorry

end optimal_tower_configuration_l412_412590


namespace roots_quadratic_equation_l412_412828

/-- 
  Let x and y be roots of a quadratic equation, 
  with the conditions that x + y = 6 and |x - y| = 8.
  Prove that the quadratic equation with these roots is x^2 - 6x - 7 = 0.
-/
theorem roots_quadratic_equation
  (x y : ℝ)
  (h1 : x + y = 6)
  (h2 : |x - y| = 8)
  : (Polynomial.X^2 - 6 * Polynomial.X - 7 : Polynomial ℝ).is_root x 
    ∧ (Polynomial.X^2 - 6 * Polynomial.X - 7 : Polynomial ℝ).is_root y :=
sorry

end roots_quadratic_equation_l412_412828


namespace option_A_option_C_option_D_l412_412847

variable {x : ℝ} (f : ℝ → ℝ)

theorem option_A : (λ x => x + 1 / x)' = λ x => 1 - 1 / x^2 := sorry
theorem option_C : (λ x => x^2 / exp x)' = λ x => (2 * x - x^2) / exp x := sorry
theorem option_D : (λ x => sin (2 * x - 1))' = λ x => 2 * cos (2 * x - 1) := sorry

end option_A_option_C_option_D_l412_412847


namespace athlete_B_more_stable_l412_412247

variable (average_scores_A average_scores_B : ℝ)
variable (s_A_squared s_B_squared : ℝ)

theorem athlete_B_more_stable
  (h_avg : average_scores_A = average_scores_B)
  (h_var_A : s_A_squared = 1.43)
  (h_var_B : s_B_squared = 0.82) :
  s_A_squared > s_B_squared :=
by 
  rw [h_var_A, h_var_B]
  sorry

end athlete_B_more_stable_l412_412247


namespace determine_m_range_l412_412152

noncomputable def line_curve_no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ m * x → (m * x ≠ (abs x - 1) / (abs (x - 1))) ∧
           (m * x ≠ (abs x - 1) / (abs (1 - x))) 

theorem determine_m_range :
  ∀ (m : ℝ), line_curve_no_intersection m ↔ (-1 ≤ m ∧ m < -3 + 2 * real.sqrt 2) :=
by
  sorry

end determine_m_range_l412_412152


namespace perfect_square_in_interval_l412_412288

noncomputable def k : ℕ → ℕ
| 0 := 1
| (n + 1) := k n + 2 + 2 * n

def S (m : ℕ) : ℕ :=
(list.range m).sum k

theorem perfect_square_in_interval (n : ℕ) : 
  ∃ m : ℕ, S n ≤ m^2 ∧ m^2 < S (n + 1) := 
sorry

end perfect_square_in_interval_l412_412288


namespace distance_hyperbola_focus_to_line_l412_412779

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412779


namespace area_of_quadrilateral_MARE_l412_412255

def is_in_unit_circle (ω : set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 = 1

def is_diameter (a m : ℝ × ℝ) (ω : set (ℝ × ℝ)) : Prop :=
  is_in_unit_circle ω a ∧ is_in_unit_circle ω m ∧ dist a m = 2

def is_angle_bisector (a r e m : ℝ × ℝ) : Prop :=
  -- Definition of E lying on the angle bisector of ∠RAM
  sorry

def equal_area (a r m e : ℝ × ℝ) : Prop :=
  -- Definition that areas of triangles RAM and REM are equal
  sorry

theorem area_of_quadrilateral_MARE (ω : set (ℝ × ℝ)) (a r m e : ℝ × ℝ)
    (h1 : is_in_unit_circle ω a) (h2 : is_in_unit_circle ω r) (h3 : is_in_unit_circle ω m) (h4 : is_in_unit_circle ω e)
    (h5 : is_diameter a m ω) (h6 : is_angle_bisector a r e m) (h7 : equal_area a r m e) :
    -- We need to prove that the area of quadrilateral MARE is 8√2/9
    area_of_quadrilateral a r m e = 8 * real.sqrt 2 / 9 :=
sorry

end area_of_quadrilateral_MARE_l412_412255


namespace calories_in_serving_l412_412109

-- Define the weights and calorie densities of the ingredients
def lemonJuiceWeight : ℕ := 150
def sugarWeight : ℕ := 120
def waterWeight : ℕ := 350
def honeyWeight : ℕ := 80

def lemonJuiceCaloriesPer100g : ℕ := 25
def sugarCaloriesPer100g : ℕ := 386
def waterCaloriesPer100g : ℕ := 0
def honeyCaloriesPer100g : ℕ := 304

-- Define the total weight of the lemonade mixture
def totalWeight : ℕ := lemonJuiceWeight + sugarWeight + waterWeight + honeyWeight

-- Define the total calories of the lemonade mixture
def totalCalories : ℝ :=
  (lemonJuiceWeight * lemonJuiceCaloriesPer100g / 100) +
  (sugarWeight * sugarCaloriesPer100g / 100) +
  (honeyWeight * honeyCaloriesPer100g / 100) +
  (waterWeight * waterCaloriesPer100g / 100)

-- Define the weight of the serving
def servingWeight : ℕ := 150

-- Calculate the calories in the given serving
def servingCalories : ℝ := totalCalories * servingWeight / totalWeight

-- Assertion: Prove that the calories in 150 grams of lemonade mixture is approximately 160 calories
theorem calories_in_serving : servingCalories ≈ 160 := 
by
  sorry

end calories_in_serving_l412_412109


namespace sum_first_eighty_diff_even_odd_l412_412817

theorem sum_first_eighty_diff_even_odd : 
  (∑ i in finset.range 80, (2 * (i + 1))) - (∑ i in finset.range 80, (2 * (i + 1) - 1)) = 80 :=
by
  sorry

end sum_first_eighty_diff_even_odd_l412_412817


namespace exists_harmonic_sum_diff_less_than_0_001_l412_412326

theorem exists_harmonic_sum_diff_less_than_0_001 
  (A : Finset ℕ) (hA : A.card = 14) :
  ∃ (k : ℕ) (a b : Finset ℕ), 
    k ∈ {1, 2, 3, 4, 5, 6, 7} ∧
    a.card = k ∧ b.card = k ∧
    a ∩ b = ∅ ∧
    abs ((a.sum (λ x, 1 / (x : ℝ))) - (b.sum (λ x, 1 / (x : ℝ)))) < 0.001 := 
sorry

end exists_harmonic_sum_diff_less_than_0_001_l412_412326


namespace cos_triple_angle_l412_412197

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l412_412197


namespace tray_height_is_correct_l412_412887

noncomputable def square_paper_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : ℝ :=
let tray_height := 2 * real.sqrt 5 in
tray_height

theorem tray_height_is_correct :
  ∀ (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ),
  side_length = 120 ∧ cut_distance = real.sqrt 20 ∧ cut_angle = 45 →
  square_paper_height side_length cut_distance cut_angle = 2 * real.sqrt 5 :=
by
  intros side_length cut_distance cut_angle h,
  obtain ⟨hs, hd, ha⟩ := h,
  rw [← hs, ← hd, ← ha],
  sorry

end tray_height_is_correct_l412_412887


namespace range_of_b_l412_412529

noncomputable def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}
noncomputable def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem range_of_b (b : ℝ) : 
  (set_A ∩ set_B b = ∅) ↔ (b = 0 ∨ b ≥ 1/3 ∨ b ≤ -2) :=
sorry

end range_of_b_l412_412529


namespace general_formula_arithmetic_sequence_sum_of_first_n_terms_l412_412250

-- Define the arithmetic sequence {a_n} with given conditions and prove its general formula
def arithmetic_sequence {n : ℕ} (a : ℕ → ℕ) (a1 : ℕ) (d : ℕ) : Prop :=
  (a 2 = 5) ∧ (a 1 + a 3 + a 4 = 19) ∧ (∀ n, a n = a1 + (n - 1) * d)

theorem general_formula_arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) :
  arithmetic_sequence a a1 d → (∀ n, a n = 2 * n + 1) := 
by
  intros h
  sorry

-- Define sum of first {n} terms of sequence {c_n} as T_n and prove the given formula
def sequence_sum_conditions {n : ℕ} (S b c : ℕ → ℕ) (a : ℕ → ℕ) (λ : ℕ) : Prop :=
  (∀ n, S n + (a n - 1) / (2^n) = λ) ∧ (∀ n, c n = b (n + 1)) ∧ (∀ n, a n = 2 * n + 1)

theorem sum_of_first_n_terms (S b c : ℕ → ℕ) (a : ℕ → ℕ) (λ : ℕ) (T : ℕ → ℕ) :
  sequence_sum_conditions S b c a λ → (∀ n, T n = 1 - (n + 1) * (1 / 2)^n) :=
by
  intros h
  sorry

end general_formula_arithmetic_sequence_sum_of_first_n_terms_l412_412250


namespace solve_for_x_l412_412926

def star (a b : ℕ) := a * b + a + b

theorem solve_for_x : ∃ x : ℕ, star 3 x = 27 ∧ x = 6 :=
by {
  sorry
}

end solve_for_x_l412_412926


namespace local_tax_deduction_in_cents_l412_412897

def aliciaHourlyWageInDollars : ℝ := 25
def taxDeductionRate : ℝ := 0.02
def aliciaHourlyWageInCents := aliciaHourlyWageInDollars * 100

theorem local_tax_deduction_in_cents :
  taxDeductionRate * aliciaHourlyWageInCents = 50 :=
by 
  -- Proof goes here
  sorry

end local_tax_deduction_in_cents_l412_412897


namespace pastries_and_cost_correct_l412_412641

def num_pastries_lola := 13 + 10 + 8 + 6
def cost_lola := 13 * 0.50 + 10 * 1.00 + 8 * 3.00 + 6 * 2.00

def num_pastries_lulu := 16 + 12 + 14 + 9
def cost_lulu := 16 * 0.50 + 12 * 1.00 + 14 * 3.00 + 9 * 2.00

def num_pastries_lila := 22 + 15 + 10 + 12
def cost_lila := 22 * 0.50 + 15 * 1.00 + 10 * 3.00 + 12 * 2.00

def num_pastries_luka := 18 + 20 + 7 + 14 + 25
def cost_luka := 18 * 0.50 + 20 * 1.00 + 7 * 3.00 + 14 * 2.00 + 25 * 1.50

def total_pastries := num_pastries_lola + num_pastries_lulu + num_pastries_lila + num_pastries_luka
def total_cost := cost_lola + cost_lulu + cost_lila + cost_luka

theorem pastries_and_cost_correct :
  total_pastries = 231 ∧ total_cost = 328.00 :=
by
  sorry

end pastries_and_cost_correct_l412_412641


namespace volume_of_intersection_polyhedron_l412_412436

noncomputable def volume_of_intersection (a : ℝ) : ℝ := 
  (a^3 * Real.sqrt 2) / 54

theorem volume_of_intersection_polyhedron {a : ℝ} (h1 : a > 0) :
  volume_of_intersection(a) = (a^3 * Real.sqrt 2) / 54 :=
by
  unfold volume_of_intersection
  sorry

end volume_of_intersection_polyhedron_l412_412436


namespace smallest_n_correct_l412_412634

/-- The first term of the geometric sequence. -/
def a₁ : ℚ := 5 / 6

/-- The second term of the geometric sequence. -/
def a₂ : ℚ := 25

/-- The common ratio for the geometric sequence. -/
def r : ℚ := a₂ / a₁

/-- The nth term of the geometric sequence. -/
def a_n (n : ℕ) : ℚ := a₁ * r^(n - 1)

/-- The smallest n such that the nth term is divisible by 10^7. -/
def smallest_n : ℕ := 8

theorem smallest_n_correct :
  ∀ n : ℕ, (a₁ * r^(n - 1)) ∣ (10^7 : ℚ) ↔ n = smallest_n := 
sorry

end smallest_n_correct_l412_412634


namespace kim_pairs_of_shoes_l412_412612

theorem kim_pairs_of_shoes : ∃ n : ℕ, 2 * n + 1 = 14 ∧ (1 : ℚ) / (2 * n - 1) = (0.07692307692307693 : ℚ) :=
by
  sorry

end kim_pairs_of_shoes_l412_412612


namespace value_of_expression_l412_412841

theorem value_of_expression : 
  (3^1 - 2 + 6^2 - 1) ^ (-2) * 3 = 1 / 432 :=
by
  have h1 : 3^1 = 3 := by norm_num
  have h2 : 6^2 = 36 := by norm_num
  have h3 : 3 - 2 + 36 - 1 = 36 := by norm_num [h1, h2]
  have h4 : (36) ^ (-2) = 1 / 36^2 := by simp
  have h5 : 36^2 = 1296 := by norm_num
  have h6 : 1 / 1296 * 3 = 1 / 432 := by field_simp
  rw [h3, h4, h5, h6]
  sorry

end value_of_expression_l412_412841


namespace curve_symmetry_l412_412595

theorem curve_symmetry :
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧
  ∀ (ρ θ' : ℝ), ρ = 4 * Real.sin (θ' - Real.pi / 3) ↔ ρ = 4 * Real.sin ((θ - θ') - Real.pi / 3) :=
sorry

end curve_symmetry_l412_412595


namespace distance_from_right_focus_to_line_l412_412723

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412723


namespace cos_triple_angle_l412_412199

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l412_412199


namespace distance_from_right_focus_to_line_l412_412766

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412766


namespace correct_propositions_l412_412864

variables (α β γ : Plane) (l : Line)

def Proposition1 : Prop :=
  (α ⊥ β) ∧ (l ⊥ β) → (l ∥ α)

def Proposition2 : Prop :=
  (l ⊥ α) ∧ (l ∥ β) → (α ⊥ β)

def Proposition3 : Prop :=
  (∃ p1 p2 : Point, (p1 ∈ l) ∧ (p2 ∈ l) ∧ (dist p1 α = dist p2 α)) → (l ∥ α)

def Proposition4 : Prop :=
  (α ⊥ β) ∧ (α ∥ γ) → (γ ⊥ β)

theorem correct_propositions : ¬ Proposition1 α β l ∧ Proposition2 α β l ∧ ¬ Proposition3 α l ∧ Proposition4 α β γ :=
by
  sorry

end correct_propositions_l412_412864


namespace geometric_seq_sum_l412_412150

theorem geometric_seq_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, a_n < a_{n + 1}) → 
  (a_1 = 1) → 
  (a_3 = 4) → 
  S_6 = 63 := 
  by sorry

end geometric_seq_sum_l412_412150


namespace sum_ceil_sqrt_10_to_40_l412_412953

noncomputable def sumCeilSqrt : ℕ :=
  ∑ n in Finset.range (40 - 10 + 1), (Int.ceil ((n + 10 : ℝ) ^ (1 / 2)))

theorem sum_ceil_sqrt_10_to_40 : sumCeilSqrt = 167 := by
  sorry

end sum_ceil_sqrt_10_to_40_l412_412953


namespace ratio_c_to_a_l412_412504

theorem ratio_c_to_a (a c : ℝ) 
  (h₁ : a > 0) 
  (h₂ : c > 0) 
  (h3 : ∃ (points : fin 4 → ℝ × ℝ), 
    ∃ (segment_lengths : list ℝ), 
    (segment_lengths = [a, a, a, a, 2 * a, c]) ∧ 
    (is_45_45_90_triangle_with_hypotenuse (points 0) (points 1) (points 2) (points 3) 2 * a)) :
  c / a = real.sqrt 2 :=
by 
  sorry

/-- Definition for recognizing the special triangle --/
def is_45_45_90_triangle_with_hypotenuse (p1 p2 p3 p4 : ℝ × ℝ) (hypotenuse_length : ℝ) : Prop :=
(by sorry : Prop)

end ratio_c_to_a_l412_412504


namespace incorrect_statements_l412_412455

-- Definitions for the conditions
def stmtA : Prop := ∀ (P Q : Prop), (P → Q) → ¬(¬Q → ¬P)  -- Direct proof only assertion (should be false)
def stmtB : Prop := ∀ (P Q : Prop), (P → Q) ∨ (¬Q → ¬P)    -- Different logical steps assertion (should be true)
def stmtC : Prop := ∀ (P : Prop), P → (P → P)              -- Strict definitions assertion (should be true)
def stmtD : Prop := ∀ (P Q : Prop), (P → Q) ∧ (Q → P)       -- Depends on recognized principles (should be true)
def stmtE : Prop := ∀ (P Q : Prop), (P → Q) ↔ (¬Q → ¬P) ∧ (P ↔ Q)  -- Contrapositive and converse equivalency (should be false)

-- Theorem for the incorrect statements
theorem incorrect_statements : (stmtA = false) ∧ (stmtE = false) :=
by
suffices := sorry

end incorrect_statements_l412_412455


namespace cos_2_alpha_plus_beta_eq_l412_412140

variable (α β : ℝ)

def tan_roots_of_quadratic (x : ℝ) : Prop := x^2 + 5 * x - 6 = 0

theorem cos_2_alpha_plus_beta_eq :
  ∀ α β : ℝ, tan_roots_of_quadratic (Real.tan α) ∧ tan_roots_of_quadratic (Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 :=
by
  intros
  sorry

end cos_2_alpha_plus_beta_eq_l412_412140


namespace distance_from_right_focus_to_line_l412_412769

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412769


namespace total_widgets_sold_15_days_l412_412606

def widgets_sold (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * n

theorem total_widgets_sold_15_days :
  (Finset.range 15).sum widgets_sold = 359 :=
by
  sorry

end total_widgets_sold_15_days_l412_412606


namespace fixed_point_for_C2_l412_412154

theorem fixed_point_for_C2 
    (vertex_C1 : ℝ × ℝ) 
    (focus_C1 : ℝ × ℝ) 
    (eqn_C2 : ℝ → ℝ → ℝ → Prop) :
    vertex_C1 = (Real.sqrt 2 - 1, 1) →
    focus_C1 = (Real.sqrt 2 - 3 / 4, 1) →
    (∀ x y a b, 
        eqn_C2 y a b = y^2 - a * y + x + 2 * b = 0) →
    (∀ x y, (x, y) ∈ C1 ∧ eqn_C2 y a b → -- Intersection Condition
        ∀ k1 k2, k1 = 1 / (2 * y - 2) ∧ k2 = -1 / (2 * y - a) → -- Perpendicular Tangents Condition
        k1 * k2 = -1) →
    ∃ x y, eqn_C2 y 1 ((2 * 1 - 1 - 2 * Real.sqrt 2) / 4) → 
        (x, y) = (Real.sqrt 2 - 1 / 2, 1) :=
by
  sorry

end fixed_point_for_C2_l412_412154


namespace sets_equality_l412_412551

variables {α : Type*} (A B C : Set α)

theorem sets_equality (h1 : A ∪ B ⊆ C) (h2 : A ∪ C ⊆ B) (h3 : B ∪ C ⊆ A) : A = B ∧ B = C :=
by
  sorry

end sets_equality_l412_412551


namespace log_eq_condition_l412_412564

theorem log_eq_condition (x : ℝ) (h : log 2 (x + 3) - log 2 (x + 1) = 1) : x = 1 :=
by
  sorry

end log_eq_condition_l412_412564


namespace find_number_l412_412810

theorem find_number (a b N : ℕ) (h1 : b = 7) (h2 : b - a = 2) (h3 : a * b = 2 * (a + b) + N) : N = 11 :=
  sorry

end find_number_l412_412810


namespace _l412_412662

noncomputable def beppo_levi_theorem {α : Type*} [MeasureSpace α] (ξ : ℕ → α → ℝ) (xi : α → ℝ) :=
  (∀ n, E (λ a, abs (ξ n a)) < ∞) →
  (∀ n, E (ξ n) < ∞) →
  (∀ a, Monotone (λ n, ξ n a)) →
  (∀ a, ξ n a → xi a) →
  (E (λ a, abs (xi a)) < ∞ ∧ Monotone (λ n, E (ξ n)) ∧ Lim (λ n, E (ξ n)) = E (xi))

#check beppo_levi_theorem

end _l412_412662


namespace solution_of_diff_eq_l412_412099
open Nat Real

-- Define the system of differential equations
def dx_dt (x y t : ℝ) := 1 - 1 / y
def dy_dt (x y t : ℝ) := 1 / (x - t)

-- Define the functions x(t) and y(t)
def x_solution (t : ℝ) : ℝ := t + exp (-t)
def y_solution (t : ℝ) : ℝ := exp t

-- Initial conditions
def initial_conditions := (x_solution 0 = 1) ∧ (y_solution 0 = 1)

-- State the theorem
theorem solution_of_diff_eq (t : ℝ) : initial_conditions ∧ ((dx_dt (x_solution t) (y_solution t) t = 1 - 1 / (y_solution t)) ∧ (dy_dt (x_solution t) (y_solution t) t = 1 / ((x_solution t) - t))) :=
by
  -- skipped proof
  sorry

end solution_of_diff_eq_l412_412099


namespace distance_from_focus_to_line_l412_412748

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412748


namespace incenter_point_l412_412456

variables {A B C P D E F : Type}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variables (AB BC CA PD PE PF l s : ℝ)
variables [PointInsideTriangle A B C P] [PerpendicularDistances PD PE PF P]

/-- If P is a point inside triangle ABC such that the sum of ratios of sides to perpendicular distances from P to the sides is less than or equal l^2 / 2s, then P is the incenter of triangle ABC. -/
theorem incenter_point
  (h1 : Perimeter A B C = l)
  (h2 : Area A B C = s)
  (h3 : PerpendicularDistance P A B = PD)
  (h4 : PerpendicularDistance P B C = PE)
  (h5 : PerpendicularDistance P C A = PF)
  (h6 : \(\frac{AB}{PD} + \frac{BC}{PE} + \frac{CA}{PF} \leq \frac{l^2}{2s}\)) :
  IsIncenter P A B C :=
by
  sorry

end incenter_point_l412_412456


namespace proof_problem_l412_412902

namespace MathProof

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

theorem proof_problem :
  (∃ n : ℕ, n ≤ 10 ∧ is_even n ∧ ¬ is_composite n ∧ n = 2) ∧
  (∃ n : ℕ, n ≤ 10 ∧ is_odd n ∧ ¬ is_prime n ∧ n = 1) ∧
  (∃ n : ℕ, n ≤ 10 ∧ is_odd n ∧ is_composite n ∧ n = 9) ∧
  (∃ n : ℕ, is_prime n ∧ ∀ m : ℕ, is_prime m → m < n → false ∧ n = 2) ∧
  (∃ n : ℕ, is_composite n ∧ ∀ m : ℕ, is_composite m → m < n → false ∧ n = 4) :=
by
  split
  · exists 2; repeat {simp}; split; norm_num; exact dec_trivial
  split
  · exists 1; repeat {simp}; split; norm_num; exact dec_trivial
  split
  · exists 9; repeat {simp}; split; norm_num; exact dec_trivial
  split
  · exists 2; repeat {simp}
  · exists 4; repeat {simp}

end MathProof

end proof_problem_l412_412902


namespace thursday_on_100th_day_of_N_minus_1_l412_412602

-- We define the necessary conditions:
def day_of_week : ℕ → ℕ
| 0 := 0 -- Sunday
| 1 := 1 -- Monday
| 2 := 2 -- Tuesday
| 3 := 3 -- Wednesday
| 4 := 4 -- Thursday
| 5 := 5 -- Friday
| 6 := 6 -- Saturday
| (n+1) := (day_of_week n + 1) % 7

def is_leap_year (n : ℕ) : Prop :=
  (n % 4 = 0 ∧ n % 100 ≠ 0) ∨ (n % 400 = 0)

theorem thursday_on_100th_day_of_N_minus_1 (N : ℕ) 
(h1 : day_of_week 299 = 2) -- 300th day of N is a Tuesday
(h2 : day_of_week 199 = 2) -- 200th day of N+1 is a Tuesday
(h3 : is_leap_year N)  -- N is a leap year
: day_of_week 99 = 4 := -- 100th day of N-1 is a Thursday
sorry

end thursday_on_100th_day_of_N_minus_1_l412_412602


namespace selection_schemes_count_l412_412331

/-- There are 6 people, labeled P1, P2, P3, P4, P5, P6. Select 4 people from them such that:
     - One person visits Paris
     - One person visits London
     - One person visits Sydney
     - One person visits Moscow
   Person labeled P1 and P2 will not visit Paris.
   Prove that the total number of different selection schemes is 240. -/
theorem selection_schemes_count : 
  let total_people := 6,
      cities := 4,
      can_visit_paris := (total_people - 2),
      choose_paris := Nat.choose can_visit_paris 1,
      remaining_people := (total_people - 1),
      choose_others := Nat.choose remaining_people 3,
      permutations := Nat.factorial 3 in
  choose_paris * choose_others * permutations = 240 :=
by 
  sorry

end selection_schemes_count_l412_412331


namespace negation_of_forall_implies_exists_l412_412351

theorem negation_of_forall_implies_exists :
  (¬ ∀ x : ℝ, x^2 > 1) = (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_forall_implies_exists_l412_412351


namespace surface_area_after_removal_l412_412084

-- Define the dimensions of the larger cube and the corner cubes
def larger_cube_dim : ℝ := 5
def corner_cube_dim : ℝ := 2

-- Define the total number of corners in a cube
def num_corners : ℕ := 8

-- Define the removal of each corner cube and its dimensions
def edge_dim (n : ℕ) := if n = 0 then larger_cube_dim else corner_cube_dim  -- Leftover function to compute an edge's dimension (not used directly in statement)

-- The statement proving the surface area of the cube after corner removals is equal to the original surface area
theorem surface_area_after_removal:
  let original_area := 6 * (larger_cube_dim ^ 2) in
  original_area = 150 := 
sorry

end surface_area_after_removal_l412_412084


namespace trigonometric_identity_l412_412059

theorem trigonometric_identity :
  sin 12 * cos 36 * sin 48 * cos 72 * tan 18 = 
  (1 / 2) * (sin 12 ^ 2 + sin 12 * cos 6) * (sin 18 ^ 2 / cos 18) :=
by sorry

end trigonometric_identity_l412_412059


namespace bottles_from_shop_A_l412_412933

theorem bottles_from_shop_A (total_bottles : ℕ) (bottles_B : ℕ) (bottles_C : ℕ) :
  total_bottles = 550 → bottles_B = 180 → bottles_C = 220 → 
  (total_bottles - (bottles_B + bottles_C)) = 150 :=
by
  intros htotal hB hC
  rw [htotal, hB, hC]
  norm_num
  sorry

end bottles_from_shop_A_l412_412933


namespace distance_from_hyperbola_focus_to_line_l412_412750

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412750


namespace parabola_vertex_coords_l412_412821

theorem parabola_vertex_coords (a b c : ℝ) (ha : a = 1) (hb : b = -4) (hc : c = 3) :
    ∃ x y : ℝ, x = -b / (2 * a) ∧ y = (4 * a * c - b^2) / (4 * a) ∧ x = 2 ∧ y = -1 :=
by
  sorry

end parabola_vertex_coords_l412_412821


namespace distance_to_line_is_sqrt5_l412_412794

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412794


namespace max_partner_share_l412_412973

theorem max_partner_share (total_profit : ℕ) (ratios : List ℕ) (h_ratios : ratios = [1, 2, 3, 3, 6]) (h_total : total_profit = 36000) :
  let parts_sum := ratios.sum in
  let part_value := total_profit / parts_sum in
  let shares := ratios.map (λ r => r * part_value) in
  shares.maximum = 14400 := by
  sorry

end max_partner_share_l412_412973


namespace distance_from_focus_to_line_l412_412685

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412685


namespace distance_from_right_focus_to_line_l412_412719

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412719


namespace num_int_values_satisfying_inequality_l412_412981

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l412_412981


namespace unit_vector_perpendicular_to_a_l412_412553

-- Definitions of a vector and the properties of unit and perpendicular vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def is_unit_vector (v : Vector2D) : Prop :=
  v.x ^ 2 + v.y ^ 2 = 1

def is_perpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

-- Given vector a
def a : Vector2D := ⟨3, 4⟩

-- Coordinates of the unit vector that is perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ (b : Vector2D), is_unit_vector b ∧ is_perpendicular a b ∧
  (b = ⟨-4 / 5, 3 / 5⟩ ∨ b = ⟨4 / 5, -3 / 5⟩) :=
sorry

end unit_vector_perpendicular_to_a_l412_412553


namespace sum_ceil_sqrt_10_to_40_l412_412952

noncomputable def sumCeilSqrt : ℕ :=
  ∑ n in Finset.range (40 - 10 + 1), (Int.ceil ((n + 10 : ℝ) ^ (1 / 2)))

theorem sum_ceil_sqrt_10_to_40 : sumCeilSqrt = 167 := by
  sorry

end sum_ceil_sqrt_10_to_40_l412_412952


namespace joel_and_dad_age_l412_412609

theorem joel_and_dad_age :
  ∃ x : ℕ, x = 21 ∧ (37 + x = 2 * (8 + x)) :=
begin
  use 21,
  split,
  { refl, },
  { sorry, }
end

end joel_and_dad_age_l412_412609


namespace set_contains_difference_of_elements_l412_412229

variable {A : Set Int}

axiom cond1 (a : Int) (ha : a ∈ A) : 2 * a ∈ A
axiom cond2 (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a + b ∈ A

theorem set_contains_difference_of_elements 
  (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a - b ∈ A := by
  sorry

end set_contains_difference_of_elements_l412_412229


namespace distance_from_focus_to_line_l412_412709

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412709


namespace point_M_symmetric_to_A_relative_to_B_l412_412438

-- Definitions
variables (A B C D E M : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space M]
variables [has_smul ℝ A] [has_smul ℝ B] [has_smul ℝ C] [has_smul ℝ D] [has_smul ℝ E] [has_smul ℝ M]

-- Triangle construction items
variables (AB BC CA : ℝ)
variables (triangleABC : triangle A B C)
variables (equilateralADB : equilateral_triangle A D B)
variables (equilateralAEC : equilateral_triangle A E C)
variables (intersection_DE_AB : ∃ M, line(AB).intersection(line(DE)) = M)

-- To Prove
theorem point_M_symmetric_to_A_relative_to_B : 
  is_symmetric_relative_to intersection_DE_AB A B := 
sorry

end point_M_symmetric_to_A_relative_to_B_l412_412438


namespace number_with_largest_prime_factor_l412_412083

def largest_prime_factors (n : ℕ) : ℕ :=
  -- Function to get the largest prime factor of a number
  sorry

theorem number_with_largest_prime_factor :
  let numbers := [45, 65, 85, 117, 169] in
  largest_prime_factors 85 = 17 ∧
  ∀ n ∈ numbers, largest_prime_factors n ≤ 17 := sorry

end number_with_largest_prime_factor_l412_412083


namespace no_primes_divisible_by_45_l412_412179

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define 45 and its prime factors
def is_divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem to prove the number of prime numbers divisible by 45 is 0
theorem no_primes_divisible_by_45 : ∀ n : ℕ, is_prime n → is_divisible_by_45 n → false :=
by
  intro n
  assume h_prime h_div_45
  sorry

end no_primes_divisible_by_45_l412_412179


namespace metallic_weight_problem_l412_412429

variables {m1 m2 m3 m4 : ℝ}

theorem metallic_weight_problem
  (h_total : m1 + m2 + m3 + m4 = 35)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = (3/4) * m3)
  (h3 : m3 = (5/6) * m4) :
  m4 = 105 / 13 :=
sorry

end metallic_weight_problem_l412_412429


namespace MN_perpendicular_to_FN_l412_412656

open EuclideanGeometry

variables {A B C F P Q M N : Point}

-- Given conditions
def CircumscribedCircle (ABC : Triangle) : Circle := sorry
def is_on_circumcircle (F : Point) (circABC : Circle) : Prop := sorry
def projection (F : Point) (line : Line) : Point := sorry
def is_projection (P F : Point) (l : Line) : Prop := sorry

def midpoint (A B : Point) : Point := sorry
def is_midpoint (M : Point) (A B : Point) : Prop := sorry

-- Let F be a point on the circumscribed circle of ΔABC
axiom F_on_circumcircle : ∀ (ABC : Triangle) (circABC : Circle), (is_on_circumcircle F circABC)

-- Let P be the projection of F onto AB
axiom P_projection : is_projection P F (line_through A B)

-- Let Q be the projection of F onto AC
axiom Q_projection : is_projection Q F (line_through A C)

-- Let M be the midpoint of BC
axiom M_midpoint : is_midpoint M B C

-- Let N be the midpoint of PQ
axiom N_midpoint : is_midpoint N P Q

-- Prove that MN is perpendicular to FN
theorem MN_perpendicular_to_FN 
  (ABC : Triangle) (circABC : Circle)
  (hF_circumcircle: is_on_circumcircle F circABC)
  (hP_proj: is_projection P F (line_through A B))
  (hQ_proj: is_projection Q F (line_through A C))
  (hM_mid: is_midpoint M B C)
  (hN_mid: is_midpoint N P Q) :
  perpendicular (line_through M N) (line_through F N) :=
sorry

end MN_perpendicular_to_FN_l412_412656


namespace transformed_area_l412_412277

theorem transformed_area (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (area_T : ℝ) : 
  let det := A * D - B * C in
  det = 17 ∧ area_T = 3 → area_T' = det * area_T → area_T' = 51 :=
by
  intros det_eq det_val area_T_eq transformed_area_eq
  rw [det_eq] at det_val
  rw [area_T_eq] at transformed_area_eq
  rw [det_val, area_T_eq] at transformed_area_eq
  exact transformed_area_eq.symm sorry

end transformed_area_l412_412277


namespace max_value_sum_l412_412520

variable (n : ℕ) (x : Fin n → ℝ)

theorem max_value_sum 
  (h1 : ∀ i, 0 ≤ x i)
  (h2 : 2 ≤ n)
  (h3 : (Finset.univ : Finset (Fin n)).sum x = 1) :
  ∃ max_val, max_val = (1 / 4) :=
sorry

end max_value_sum_l412_412520


namespace maria_paid_9_l412_412305

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l412_412305


namespace marble_theorem_l412_412995

noncomputable def marble_problem (M : ℝ) : Prop :=
  let M_Pedro : ℝ := 0.7 * M
  let M_Ebony : ℝ := 0.85 * M_Pedro
  let M_Jimmy : ℝ := 0.7 * M_Ebony
  (M_Jimmy / M) * 100 = 41.65

theorem marble_theorem (M : ℝ) : marble_problem M := 
by
  sorry

end marble_theorem_l412_412995


namespace functional_equation_solution_exists_l412_412481

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution_exists (f : ℝ → ℝ) (h : ∀ x y, 0 < x → 0 < y → f x * f y = 2 * f (x + y * f x)) :
  ∃ c : ℝ, ∀ x, 0 < x → f x = x + c := 
sorry

end functional_equation_solution_exists_l412_412481


namespace ceil_sqrt_sum_l412_412942

theorem ceil_sqrt_sum : (∑ n in Finset.range 31, (nat_ceil (Real.sqrt (n + 10)))) = 167 := by
  sorry

end ceil_sqrt_sum_l412_412942


namespace fraction_computation_l412_412922

theorem fraction_computation :
  ((11^4 + 324) * (23^4 + 324) * (35^4 + 324) * (47^4 + 324) * (59^4 + 324)) / 
  ((5^4 + 324) * (17^4 + 324) * (29^4 + 324) * (41^4 + 324) * (53^4 + 324)) = 295.615 := 
sorry

end fraction_computation_l412_412922


namespace cargo_per_truck_is_2_5_l412_412105

-- Define our instance conditions
variables (x : ℝ) (n : ℕ)

-- Conditions extracted from the problem
def truck_capacity_change : Prop :=
  55 ≤ x ∧ x ≤ 64 ∧
  (x = (x / n - 0.5) * (n + 4))

-- Objective based on these conditions
theorem cargo_per_truck_is_2_5 :
  truck_capacity_change x n → (x = 60) → (n + 4 = 24) → (x / 24 = 2.5) :=
by 
  sorry

end cargo_per_truck_is_2_5_l412_412105


namespace min_PA_dot_PB_l412_412540

noncomputable section

-- Definitions of vectors and distances
variables {A B P : Type} [metric_space A] [inner_product_space ℝ A]
open_locale big_operators

def AB := (A B : A) → A
def AP := (A P : A) → A

-- Condition 1: The magnitude of the vector AB
axiom AB_magnitude : ∀ (A B : A), dist A B = 10

-- Condition 2: The inequality for any t ∈ ℝ
axiom AP_AB_inequality : ∀ (A B P : A) (t : ℝ), dist (AP A P - t • AB A B) 0 ≥ 3

-- Statement of the problem
theorem min_PA_dot_PB (A B P : A) :
  ∃ (d : ℝ), (dist (AP A P + AB A B) 0 = 6) ∧ 
  ((1/4) * ((dist (AP A P + AB A B) 0)^2 - (dist (AP A P - AB A B) 0)^2) = -16) :=
sorry

end min_PA_dot_PB_l412_412540


namespace inequalities_true_l412_412562

theorem inequalities_true (a b : ℝ) (h : a > b) : a^3 > b^3 ∧ 3 ^ a > 3 ^ b := 
by {
  have h1 : a^3 > b^3 := sorry,
  have h2 : 3 ^ a > 3 ^ b := sorry,
  exact ⟨h1, h2⟩,
}

end inequalities_true_l412_412562


namespace rhombus_diagonal_length_l412_412437

noncomputable def shortest_diagonal_length (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) : ℝ :=
  if h : d1 / d2 = ratio ∧ (d1 * d2 / 2) = area then
    d2
  else
    0

theorem rhombus_diagonal_length :
  let ratio := 5.0 / 3.0 in
  let area := 150.0 in
  shortest_diagonal_length (5 * 2 * Real.sqrt 5) (3 * 2 * Real.sqrt 5) area ratio = 6 * Real.sqrt 5 := by
  sorry

end rhombus_diagonal_length_l412_412437


namespace valid_A_values_count_l412_412102

theorem valid_A_values_count : 
  (A : ℕ) → (A ∣ 45) → (357 * 10000 + 1000 * 1 + 100 * A + 10 * 6) % 4 = 0 → 
  (357 * 10000 + 1000 * 1 + 100 * A + 10 * 6) % 5 = 0 → 
  A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 
  (Finset.card ((Finset.filter (λ x, x ∣ 45 ∧ 
                        (3571000 + 100 * x + 10 * 6)%4 = 0 ∧
                        (3571000 + 100 * x + 10 * 6)%5 = 0)
                        (Finset.range 10)) = 0) :=
by sorry

end valid_A_values_count_l412_412102


namespace distance_from_focus_to_line_l412_412699

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412699


namespace find_a_l412_412508

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l412_412508


namespace no_positive_integer_solutions_l412_412320

theorem no_positive_integer_solutions (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2 * n) * y * (y + 1) :=
by
  sorry

end no_positive_integer_solutions_l412_412320


namespace problem_1_problem_2_l412_412547

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * |x| - 2

theorem problem_1 : {x : ℝ | f x > 3} = {x : ℝ | x < -1 ∨ x > 5} :=
sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m ≤ 1 :=
sorry

end problem_1_problem_2_l412_412547


namespace gcd_12569_36975_l412_412490

-- Define the integers for which we need to find the gcd
def num1 : ℕ := 12569
def num2 : ℕ := 36975

-- The statement that the gcd of these two numbers is 1
theorem gcd_12569_36975 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcd_12569_36975_l412_412490


namespace arithmetic_square_root_l412_412147

noncomputable def cube_root (x : ℝ) : ℝ :=
  x^(1/3)

noncomputable def sqrt_int_part (x : ℝ) : ℤ :=
  ⌊Real.sqrt x⌋

theorem arithmetic_square_root 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (h1 : cube_root a = 2) 
  (h2 : b = sqrt_int_part 5) 
  (h3 : c = 4 ∨ c = -4) : 
  Real.sqrt (a + ↑b + c) = Real.sqrt 14 ∨ Real.sqrt (a + ↑b + c) = Real.sqrt 6 := 
sorry

end arithmetic_square_root_l412_412147


namespace problem_1_problem_2_problem_3_l412_412637

noncomputable def S_n (n : ℕ) : ℝ := (3^(n+1) - 3) / 2

def a_n (n : ℕ) : ℝ := 3^n

def b_n (n : ℕ) : ℝ := 1 / ((n + 1) * Math.log 3^(n))

def c_n (n : ℕ) : ℝ := (2 * n + 1) * 3^n

noncomputable def B_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, b_n k)

noncomputable def C_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, c_n k)

theorem problem_1 : ∀ n, a_n n = 3^n :=
by
  intros n
  unfold a_n
  sorry

theorem problem_2 : ∀ n, B_n n = n / (n + 1) :=
by
  intros n
  unfold B_n b_n
  sorry

theorem problem_3 : ∀ n, C_n n = n * 3^(n+1) :=
by
  intros n
  unfold C_n c_n
  sorry

end problem_1_problem_2_problem_3_l412_412637


namespace indefinite_integral_l412_412912

noncomputable def integrand (x : ℝ) : ℝ := (x^5 + 2 * x^4 - 2 * x^3 + 5 * x^2 - 7 * x + 9) / ((x + 3) * (x - 1) * x)

theorem indefinite_integral : 
  ∃ C : ℝ, (∫ x in Real.Ioo (-∞ : ℝ) (∞ : ℝ), integrand x) = (λ x, (x^3)/3 + x + 4 * Real.log (abs (x + 3)) + 2 * Real.log (abs (x - 1)) - 3 * Real.log (abs x) + C) :=
sorry

end indefinite_integral_l412_412912


namespace incorrect_expansion_brackets_l412_412899

theorem incorrect_expansion_brackets :
  let A := (5 * x - (x - 2 * y) = 5 * x - x + 2 * y)
  let B := (2 * a^2 + (3 * a - b) = 2 * a^2 + 3 * a - b)
  let C := ((x - 2 * y) - (x^2 - y^2) = x - 2 * y - x^2 + y^2)
  let D := (3 * x^2 - 3 * (x + 6) = 3 * x^2 - 3 * x - 6)
  in 3 * x^2 - 3 * (x + 6) ≠ 3 * x^2 - 3 * x - 6 := 
by 
  sorry

end incorrect_expansion_brackets_l412_412899


namespace no_real_m_for_parallel_lines_l412_412528

theorem no_real_m_for_parallel_lines : 
  ∀ (m : ℝ), ∃ (l1 l2 : ℝ × ℝ × ℝ), 
  (l1 = (2, (m + 1), 4)) ∧ (l2 = (m, 3, 4)) ∧ 
  ( ∀ (m : ℝ), -2 / (m + 1) = -m / 3 → false ) :=
by sorry

end no_real_m_for_parallel_lines_l412_412528


namespace f_f_f_f_f_3_eq_4_l412_412635

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_f_f_f_f_3_eq_4 : f (f (f (f (f 3)))) = 4 := 
  sorry

end f_f_f_f_f_3_eq_4_l412_412635


namespace workshop_total_workers_l412_412407

noncomputable def average_salary_of_all (W : ℕ) : ℝ := 8000
noncomputable def average_salary_of_technicians : ℝ := 12000
noncomputable def average_salary_of_non_technicians : ℝ := 6000

theorem workshop_total_workers
    (W : ℕ)
    (T : ℕ := 7)
    (N : ℕ := W - T)
    (h1 : (T + N) = W)
    (h2 : average_salary_of_all W = 8000)
    (h3 : average_salary_of_technicians = 12000)
    (h4 : average_salary_of_non_technicians = 6000)
    (h5 : (7 * 12000) + (N * 6000) = (7 + N) * 8000) :
  W = 21 :=
by
  sorry


end workshop_total_workers_l412_412407


namespace find_X_coordinates_l412_412858

noncomputable def X : ℝ × ℝ × ℝ :=
let x1 := -7/22 in
let x2 := -5/22 in
let x3 := 13/22 in
(x1, x2, x3)

theorem find_X_coordinates :
  let x1 := -7/22 in
  let x2 := -5/22 in
  let x3 := 13/22 in
  (x1 + 2 * x2 + 3 * x3 = 1) ∧
  ((-x1) + 3 * x2 + 4 * x3 = 2) ∧
  (2 * x1 - 3 * x2 + 5 * x3 = 3) :=
by
  sorry

end find_X_coordinates_l412_412858


namespace option_C_correct_l412_412898

theorem option_C_correct (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1 / 2 :=
sorry

end option_C_correct_l412_412898


namespace arrange_people_l412_412362

-- Define the conditions
def num_people := 8
def front_people := {6, 7, 8}
def last_person := 5
def remaining_people := {1, 2, 3, 4}

-- The main theorem based on the problem statement
theorem arrange_people : 
  -- Given 8 people numbered 1 to 8
  let positions : Finset (Fin 8) := Finset.univ \ {Fin.ofNat 5}
  ∧ let front_positions : Finset (Fin 8) := positions.choose 3
  → let arrangements := front_positions.card.factorial
  ∧ arrangements * front_positions.card = 210 :=
by {
  -- Proof is omitted
  sorry
}

end arrange_people_l412_412362


namespace AP_contains_100_consecutive_nines_l412_412130

theorem AP_contains_100_consecutive_nines (a d : ℕ) (h_inf : ∀ (n : ℕ), ∃ k : ℕ, a + k * d = n) :
  ∃ t : ℕ, (t.to_digits 10).tails.any (λ l, (l.take 100).all (λ x, x = 9)) :=
sorry

end AP_contains_100_consecutive_nines_l412_412130


namespace inequality_proof_l412_412516

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l412_412516


namespace range_of_a_l412_412234

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 3 → deriv (f a) x ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end range_of_a_l412_412234


namespace convert_base_5_to_decimal_l412_412143

theorem convert_base_5_to_decimal :
  let base_5_number := 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  in base_5_number = 254 :=
by
  let base_5_number : ℕ := 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  have h : base_5_number = 254 := by sorry
  exact h

end convert_base_5_to_decimal_l412_412143


namespace distance_to_line_is_sqrt5_l412_412804

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412804


namespace rectangle_triangle_ratio_l412_412882

noncomputable def ratio_longer_to_shorter_side (a b : ℝ) (h : a ≥ b) : ℝ := a / b

theorem rectangle_triangle_ratio
  (a b : ℝ)
  (h : a ≥ 2 * b)
  (h₁ : ∀ (x y : ℝ), ∃ (n : ℕ), ∃ (m k : ℕ), x+y= k + m  )
  (h₂ : ∀ (x : ℝ) (y : ℝ) (h: ratio_longer_to_shorter_side x y ≥ 2 ) )
  : ratio_longer_to_shorter_side a b ≥ 2 :=
  by
    sorry -- Proof goes here

end rectangle_triangle_ratio_l412_412882


namespace expected_value_of_win_l412_412875

noncomputable def expected_winnings : ℝ :=
  let probabilities := (1 / 8 : ℝ)
  let winnings := λ n : ℕ, if n = 8 then 3 * (8 - n) else (8 - n)
  (probabilities * (winnings 1 + winnings 2 + winnings 3 + winnings 4 + winnings 5 + winnings 6 + winnings 7 + winnings 8))

theorem expected_value_of_win : expected_winnings = 3.5 := by sorry

end expected_value_of_win_l412_412875


namespace product_of_values_l412_412080

theorem product_of_values (x : ℝ) : (∀ x, (2 * x^2 + 4 * x - 6 = 0 → ((x = 1 ∨ x = -3) ∧ ∏ roots, roots = -3))) := sorry

end product_of_values_l412_412080


namespace max_servings_hot_chocolate_l412_412881

def recipe_servings : ℕ := 5
def chocolate_required : ℕ := 2 -- squares of chocolate required for 5 servings
def sugar_required : ℚ := 1 / 4 -- cups of sugar required for 5 servings
def water_required : ℕ := 1 -- cups of water required (not limiting)
def milk_required : ℕ := 4 -- cups of milk required for 5 servings

def chocolate_available : ℕ := 5 -- squares of chocolate Jordan has
def sugar_available : ℚ := 2 -- cups of sugar Jordan has
def milk_available : ℕ := 7 -- cups of milk Jordan has
def water_available_lots : Prop := True -- Jordan has lots of water (not limited)

def servings_from_chocolate := (chocolate_available / chocolate_required) * recipe_servings
def servings_from_sugar := (sugar_available / sugar_required) * recipe_servings
def servings_from_milk := (milk_available / milk_required) * recipe_servings

def max_servings (a b c : ℚ) : ℚ := min (min a b) c

theorem max_servings_hot_chocolate :
  max_servings servings_from_chocolate servings_from_sugar servings_from_milk = 35 / 4 :=
by
  sorry

end max_servings_hot_chocolate_l412_412881


namespace tan_sin_eq_solution_unique_l412_412556

def S (x : ℝ) : ℝ := (Real.tan x) - (Real.sin x)

theorem tan_sin_eq_solution_unique :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arcsin 100 ∧ (Real.tan x = Real.tan (Real.sin x)) := sorry

end tan_sin_eq_solution_unique_l412_412556


namespace even_border_length_l412_412000

def board_dim : ℕ × ℕ := (2010, 2011)
def domino_dim : ℕ × ℕ := (2, 1)
def is_covered (B : ℕ × ℕ) (D : ℕ × ℕ) : Prop := true  -- Assume the board is completely covered with dominoes

theorem even_border_length :
  let (m, n) := board_dim in
  let (p, q) := domino_dim in
  is_covered (m, n) (p, q) →
  (∃ l : ℕ, (l % 2 = 0) ∧ border_length_between_horizontal_and_vertical_dominoes m n = l) :=
by
  sorry

end even_border_length_l412_412000


namespace distance_to_line_is_sqrt5_l412_412802

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412802


namespace true_proposition_l412_412135

-- Definitions based on the conditions
def p (x : ℝ) := x * (x - 1) ≠ 0 → x ≠ 0 ∧ x ≠ 1
def q (a b c : ℝ) := a > b → c > 0 → a * c > b * c

-- The theorem based on the question and the conditions
theorem true_proposition (x a b c : ℝ) (hp : p x) (hq_false : ¬ q a b c) : p x ∨ q a b c :=
by
  sorry

end true_proposition_l412_412135


namespace marbles_in_boxes_l412_412368

theorem marbles_in_boxes (marbles_per_box total_marbles : ℕ) (h1 : marbles_per_box = 6) (h2 : total_marbles = 18) :
  total_marbles / marbles_per_box = 3 :=
by
  rw [h1, h2]
  norm_num

end marbles_in_boxes_l412_412368


namespace intersection_complement_N_correct_l412_412171

-- Define the universal set U and subsets M and N
def U : set ℕ := {1, 2, 3, 4, 5, 6}
def M : set ℕ := {1, 4}
def N : set ℕ := {2, 3}

-- Define the complement of M relative to U
def complement_U_M : set ℕ := {x | x ∈ U ∧ x ∉ M}

-- Define the set intersection of complement_U_M and N
def intersection_complement_N : set ℕ := complement_U_M ∩ N

-- Prove that the intersection is equal to {2, 3}
theorem intersection_complement_N_correct : intersection_complement_N = {2, 3} := by
  sorry

end intersection_complement_N_correct_l412_412171


namespace distance_from_focus_to_line_l412_412734

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412734


namespace inequality_proof_l412_412006

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by sorry

end inequality_proof_l412_412006


namespace grandmother_current_age_l412_412400

theorem grandmother_current_age (yoojung_age_current yoojung_age_future grandmother_age_future : ℕ)
    (h1 : yoojung_age_current = 5)
    (h2 : yoojung_age_future = 10)
    (h3 : grandmother_age_future = 60) :
    grandmother_age_future - (yoojung_age_future - yoojung_age_current) = 55 :=
by 
  sorry

end grandmother_current_age_l412_412400


namespace positive_difference_of_prime_factors_l412_412838

theorem positive_difference_of_prime_factors :
  let n := 175616
  let factorization := 2^5 * 7^3
  (factorization = n) → ∃ a b: ℕ, nat.prime a ∧ nat.prime b ∧
    (factorization = (a^5) * (b^3)) ∧ (a = 2) ∧ (b = 7) ∧ (b - a = 5)
:= by
  sorry

end positive_difference_of_prime_factors_l412_412838


namespace distance_from_right_focus_to_line_l412_412762

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412762


namespace no_primes_divisible_by_45_l412_412180

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define 45 and its prime factors
def is_divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem to prove the number of prime numbers divisible by 45 is 0
theorem no_primes_divisible_by_45 : ∀ n : ℕ, is_prime n → is_divisible_by_45 n → false :=
by
  intro n
  assume h_prime h_div_45
  sorry

end no_primes_divisible_by_45_l412_412180


namespace intersection_of_lines_l412_412929

noncomputable def intersection_point : ℝ × ℝ :=
  let x := -1 / 2 in
  let y := 5 * x + 1 in
  (x, y)

theorem intersection_of_lines :
  let p := intersection_point in
  let line1 (p : ℝ × ℝ) := p.2 = 5 * p.1 + 1 in
  let line2 (p : ℝ × ℝ) := p.2 + 3 = -3 * p.1 in
  line1 p ∧ line2 p :=
by
  sorry

end intersection_of_lines_l412_412929


namespace ratio_of_population_l412_412052

theorem ratio_of_population (Z : ℕ) :
  let Y := 2 * Z
  let X := 3 * Y
  let W := X + Y
  X / (Z + W) = 2 / 3 :=
by
  sorry

end ratio_of_population_l412_412052


namespace removal_maximizes_pairs_sum_12_l412_412382

def original_list : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def valid_pairs (lst : List ℕ) : List (ℕ × ℕ) :=
  [(0, 12), (1, 11), (2, 10), (3, 9), (4, 8), (5, 7)]

def pairs_after_removal (n : ℕ) : List (ℕ × ℕ) :=
  valid_pairs (original_list.erase n).filter (λ (xy : ℕ × ℕ), xy.1 ≠ n ∧ xy.2 ≠ n)

def max_pairs_after_removal : ℕ :=
  6

theorem removal_maximizes_pairs_sum_12 :
  (∀ n : ℕ, n ≠ 6 → pairs_after_removal 6.length ≥ pairs_after_removal n.length) :=
by
  sorry

end removal_maximizes_pairs_sum_12_l412_412382


namespace coefficient_of_x_in_first_term_l412_412416

variable {a k n : ℝ} (x : ℝ)

theorem coefficient_of_x_in_first_term (h1 : (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
  (h2 : a - n + k = 7) :
  3 = 3 := 
sorry

end coefficient_of_x_in_first_term_l412_412416


namespace boards_per_package_calculation_l412_412419

-- Defining the conditions
def total_boards : ℕ := 154
def num_packages : ℕ := 52

-- Defining the division of total_boards by num_packages within rationals
def boards_per_package : ℚ := total_boards / num_packages

-- Prove that the boards per package is mathematically equal to the total boards divided by the number of packages
theorem boards_per_package_calculation :
  boards_per_package = 154 / 52 := by
  sorry

end boards_per_package_calculation_l412_412419


namespace distance_from_focus_to_line_l412_412728

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412728


namespace simplest_quadratic_radical_l412_412398

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (\sqrt{a^2 - b^2}) :=
begin
  -- Definitions
  let radical1 := \sqrt{12},
  let radical2 := \sqrt{\frac{x}{3}},
  let radical3 := \sqrt{a^4},
  let radical4 := \sqrt{a^2 - b^2},

  -- Prove that radical4 is the simplest
  -- Proof steps omitted

  sorry
end

end simplest_quadratic_radical_l412_412398


namespace find_least_k_l412_412357

section sequence_problem

-- Define the sequence {b_n}
def b : ℕ → ℝ
| 0     => 2
| (n+1) => sorry -- This definition is not needed for the problem statement itself 

-- Recurrence relation as a condition.
axiom recurrence_relation (n : ℕ) (hn : n ≥ 1) :
  3^(b (n+1) - b n) - 1 = 1 / (2 * n + 1/2)

-- Statement of the problem
theorem find_least_k (k : ℕ) (hk : k > 1) : ∃ k, (k > 1 ∧ ∃ m : ℕ, 3^m = 4*k + 2) := 
begin
  use 6,
  split,
  { exact nat.succ_pos 5, },
  { use 3, 
    norm_num,
  },
end

end sequence_problem

end find_least_k_l412_412357


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412793

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412793


namespace matt_paper_piles_l412_412312

theorem matt_paper_piles (n : ℕ) (h_n1 : 1000 < n) (h_n2 : n < 2000)
  (h2 : n % 2 = 1) (h3 : n % 3 = 1) (h4 : n % 4 = 1)
  (h5 : n % 5 = 1) (h6 : n % 6 = 1) (h7 : n % 7 = 1)
  (h8 : n % 8 = 1) : 
  ∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n = 1681 ∧ k = 41 :=
by
  use 41
  sorry

end matt_paper_piles_l412_412312


namespace sum_divisible_by_prime_l412_412661

theorem sum_divisible_by_prime (p : ℕ) [hp : Fact (Nat.Prime p)] (n : ℕ) (hn : n ≥ p) :
  p ∣ ∑ k in Finset.range ((n / p) + 1), (-1) ^ k * Nat.choose n (p * k) := 
sorry

end sum_divisible_by_prime_l412_412661


namespace distance_hyperbola_focus_to_line_l412_412777

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412777


namespace proof_vector_BC_l412_412238

noncomputable def verify_vector_BC : Prop :=
  ∃ (PA PQ BC : ℝ × ℝ), 
    PA = (4, 3) ∧ PQ = (1, 5) ∧
    BC = 3 * (2 * PQ - PA) ∧
    BC = (-6, 21)

theorem proof_vector_BC : verify_vector_BC :=
by {
  use [(4, 3), (1, 5), (-6,21)],
  split,
  { refl },
  split,
  { refl },
  split,
  { simp,  -- simplify the arithmetic
    norm_num },
  { refl }
}

end proof_vector_BC_l412_412238


namespace find_a_plus_b_l412_412319

-- Define the rectangle and its properties
structure Rectangle (A B C D : Point) :=
  (AB : Real)
  (BC : Real)
  (CD : Real)
  (DA : Real)
  (diagonal_AC : Line)

-- Define points and positions
structure Points (A B C D P : Point) :=
  (P_on_AC : P ∈ diagonal_AC)
  (AP_gt_CP : dist A P > dist C P)

-- Define circumcenters and angles
structure Circumcenters (O1 O2 : Point) (ABP CDP : Triangle) :=
  (O1_circumcenter : Circumcenter ABP O1)
  (O2_circumcenter : Circumcenter CDP O2)
  (O1P_O2P_90 : angle O1 P O2 = 90)

-- Main Properties
structure MainProperties :=
  (AB_length : Real := 8)
  (CD_length : Real := 15)
  (AP_expr : Real := 5 + sqrt 40.25)
  (sqrt_sum_eq_int : sqrt a + sqrt b ∈ ℕ)

-- Goal: Find a + b
theorem find_a_plus_b
  (A B C D P O1 O2 : Point)
  (Rect : Rectangle A B C D)
  (Pts : Points A B C D P)
  (Circum : Circumcenters O1 O2 (Triangle.mk A B P) (Triangle.mk C D P))
  (Props : MainProperties) :
  let a := (5 : Real)^2
  let b := 40.25
  a + b = 65.25 :=
sorry

end find_a_plus_b_l412_412319


namespace cos_triple_angle_l412_412204

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l412_412204


namespace whole_numbers_count_between_sqrt75_sqrt150_l412_412194

-- Define the square roots
def sqrt_75 : Real := Real.sqrt 75
def sqrt_150 : Real := Real.sqrt 150

-- Count the whole numbers between √75 and √150
def countWholeNumbersBetween (lower : Real) (upper : Real) : ℕ :=
  (upper.floor - lower.ceil + 1).to_nat

theorem whole_numbers_count_between_sqrt75_sqrt150 :
  countWholeNumbersBetween sqrt_75 sqrt_150 = 4 := by
  sorry

end whole_numbers_count_between_sqrt75_sqrt150_l412_412194


namespace apples_harvested_l412_412647

variable (A P : ℕ)
variable (h₁ : P = 3 * A) (h₂ : P - A = 120)

theorem apples_harvested : A = 60 := 
by
  -- proof will go here
  sorry

end apples_harvested_l412_412647


namespace derivative_of_odd_is_even_l412_412315

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Assume f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Assume g is the derivative of f
axiom g_derivative : ∀ x, g x = deriv f x

-- Goal: Prove that g is an even function, i.e., g(-x) = g(x)
theorem derivative_of_odd_is_even : ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_is_even_l412_412315


namespace range_of_h_range_of_k_l412_412165

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * log x / log 2
noncomputable def g (x : ℝ) : ℝ := log x / log 2
noncomputable def h (x : ℝ) : ℝ := (f x + 1) * g x

theorem range_of_h : ∀ x, 1 ≤ x ∧ x ≤ 4 → 0 ≤ h x ∧ h x ≤ 2 :=
by
  sorry

theorem range_of_k (k : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 4 → f(x * x) * f(sqrt x) > k * g x) ↔ k < -3 :=
by
  sorry

end range_of_h_range_of_k_l412_412165


namespace distance_from_focus_to_line_l412_412738

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l412_412738


namespace distance_from_focus_to_line_l412_412703

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412703


namespace distance_from_right_focus_to_line_l412_412726

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l412_412726


namespace trees_probability_l412_412016

theorem trees_probability (num_maple num_oak num_birch total_slots total_trees : ℕ) 
                         (maple_count oak_count birch_count : Prop)
                         (prob_correct : Prop) :
  num_maple = 4 →
  num_oak = 5 →
  num_birch = 6 →
  total_trees = 15 →
  total_slots = 10 →
  maple_count → oak_count → birch_count →
  prob_correct →
  (m + n = 57) :=
by
  intros
  sorry

end trees_probability_l412_412016


namespace num_girls_in_school_l412_412012

noncomputable def total_students : ℕ := 1600
noncomputable def sample_students : ℕ := 200
noncomputable def girls_less_than_boys_in_sample : ℕ := 10

-- Equations from conditions
def boys_in_sample (B G : ℕ) : Prop := G = B - girls_less_than_boys_in_sample
def sample_size (B G : ℕ) : Prop := B + G = sample_students

-- Proportion condition
def proportional_condition (G G_total : ℕ) : Prop := G * total_students = G_total * sample_students

-- Total number of girls in the school
def total_girls_in_school (G_total : ℕ) : Prop := G_total = 760

theorem num_girls_in_school :
  ∃ B G G_total : ℕ, boys_in_sample B G ∧ sample_size B G ∧ proportional_condition G G_total ∧ total_girls_in_school G_total :=
sorry

end num_girls_in_school_l412_412012


namespace evaluate_sum_l412_412945

theorem evaluate_sum :
  (∑ n in finset.range 31, (let k := n + 10 in
    if 10 ≤ k ∧ k ≤ 16 then 4 else
    if 17 ≤ k ∧ k ≤ 25 then 5 else
    if 26 ≤ k ∧ k ≤ 36 then 6 else
    if 37 ≤ k ∧ k ≤ 40 then 7 else 0)) = 167 :=
by sorry

end evaluate_sum_l412_412945


namespace fixed_points_a1_b_neg2_range_of_a_for_two_fixed_points_l412_412075

-- Definitions based on the conditions provided

-- Question 1: Proving fixed points for the given specific values of a and b
theorem fixed_points_a1_b_neg2 : 
  let f := λ x : ℝ, x^2 - x - 3 
  in (∃ x : ℝ, f x = x) → 
      (x = 3 ∨ x = -1) :=
by 
  intro f 
  sorry

-- Question 2: Proving the range of a such that the function always has two fixed points
theorem range_of_a_for_two_fixed_points : 
  ∀ (a : ℝ), 
    (∀ (b : ℝ), 
     let f := λ x : ℝ, a * x^2 + (b + 1) * x + b - 1 
     in (∃ x : ℝ, f x = x ∧ (x' : ℝ) x' ≠ x → f x' = x')) → 
      0 < a ∧ a < 1 :=
by 
  intro a
  sorry

end fixed_points_a1_b_neg2_range_of_a_for_two_fixed_points_l412_412075


namespace bryan_total_books_and_magazines_l412_412910

-- Define the conditions
def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def bookshelves : ℕ := 29

-- Define the total books and magazines
def total_books : ℕ := books_per_shelf * bookshelves
def total_magazines : ℕ := magazines_per_shelf * bookshelves
def total_books_and_magazines : ℕ := total_books + total_magazines

-- The proof problem statement
theorem bryan_total_books_and_magazines : total_books_and_magazines = 2436 := 
by
  sorry

end bryan_total_books_and_magazines_l412_412910


namespace find_a_l412_412507

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l412_412507


namespace most_likely_outcome_l412_412972

noncomputable def probability_a := (1 / 2)^5
noncomputable def probability_b := (1 / 2)^5
noncomputable def probability_c := (finset.card (finset.filter (λ (s : finset (fin 5)), finset.card s = 3) (finset.powerset (finset.univ : finset (fin 5)))) : ℚ) * (1 / 2)^5
noncomputable def probability_d := 2 * ((finset.card (finset.filter (λ (s : finset (fin 5)), finset.card s = 4) (finset.powerset (finset.univ : finset (fin 5)))) : ℚ) * (1 / 2)^5)

theorem most_likely_outcome :
  max (max probability_a (max probability_b probability_c)) probability_d = probability_c ∧ probability_d := by
  sorry

end most_likely_outcome_l412_412972


namespace interest_earned_is_correct_l412_412680

-- Define the principal amount, interest rate, and duration
def principal : ℝ := 2000
def rate : ℝ := 0.02
def duration : ℕ := 3

-- The compound interest formula to calculate the future value
def future_value (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- Calculate the interest earned
def interest (P : ℝ) (A : ℝ) : ℝ := A - P

-- Theorem statement: The interest Bart earns after 3 years is 122 dollars
theorem interest_earned_is_correct : interest principal (future_value principal rate duration) = 122 :=
by
  sorry

end interest_earned_is_correct_l412_412680


namespace tangent_segment_equality_l412_412414

theorem tangent_segment_equality
  {A B C D E F G O : Point}
  (hAB_intersects : is_chord AB O C D)
  (hAE_tangent : is_tangent A E O)
  (hBF_tangent : is_tangent B F O)
  (h_AC_eq_DB : AC = DB)
  (hEF_intersects_AB : intersects EF AB G) :
  AG = GB :=
sorry

end tangent_segment_equality_l412_412414


namespace exists_large_abs_component_l412_412409

theorem exists_large_abs_component :
  ∃ N : ℕ, ∀ A₀ : tuple ℝ n, ∃ x ∈ A_N,
    |x| ≥ (n / 2) := 
begin
  sorry,
end

end exists_large_abs_component_l412_412409


namespace Janice_time_left_l412_412607

-- Define the conditions as variables and parameters
def homework_time := 30
def cleaning_time := homework_time / 2
def dog_walking_time := homework_time + 5
def trash_time := homework_time / 6
def total_time_before_movie := 2 * 60

-- Calculation of total time required for all tasks
def total_time_required_for_tasks : Nat :=
  homework_time + cleaning_time + dog_walking_time + trash_time

-- Time left before the movie starts after completing all tasks
def time_left_before_movie : Nat :=
  total_time_before_movie - total_time_required_for_tasks

-- The final statement to prove
theorem Janice_time_left : time_left_before_movie = 35 :=
  by
    -- This will execute automatically to verify the theorem
    sorry

end Janice_time_left_l412_412607


namespace evaluate_sum_l412_412943

theorem evaluate_sum :
  (∑ n in finset.range 31, (let k := n + 10 in
    if 10 ≤ k ∧ k ≤ 16 then 4 else
    if 17 ≤ k ∧ k ≤ 25 then 5 else
    if 26 ≤ k ∧ k ≤ 36 then 6 else
    if 37 ≤ k ∧ k ≤ 40 then 7 else 0)) = 167 :=
by sorry

end evaluate_sum_l412_412943


namespace Richard_Orlando_ratio_l412_412608

def Jenny_cards : ℕ := 6
def Orlando_more_cards : ℕ := 2
def Total_cards : ℕ := 38

theorem Richard_Orlando_ratio :
  let Orlando_cards := Jenny_cards + Orlando_more_cards
  let Richard_cards := Total_cards - (Jenny_cards + Orlando_cards)
  let ratio := Richard_cards / Orlando_cards
  ratio = 3 :=
by
  sorry

end Richard_Orlando_ratio_l412_412608


namespace angle_DSO_105_l412_412598

namespace TriangleAngles

variables {D G O S : Type} [IsoscelesTriangle D G O DOG_angleEqual]  -- DOG is an isosceles triangle

def DOG_angleEqual (α β θ : ℝ) : Prop :=
α = β ∧ θ = 40 ∧ 2*α + θ = 180

theorem angle_DSO_105 (α β θ δ φ : ℝ) (h1 : DOG_angleEqual α β θ) (h2 : θ = 40) (h3 : δ = α/2) :
  δ + θ + φ = 180 → φ = 105 :=
by
  intros
  sorry

end TriangleAngles

end angle_DSO_105_l412_412598


namespace ceil_sqrt_sum_l412_412940

theorem ceil_sqrt_sum : (∑ n in Finset.range 31, (nat_ceil (Real.sqrt (n + 10)))) = 167 := by
  sorry

end ceil_sqrt_sum_l412_412940


namespace only_one_correct_l412_412447

variables (A B C : ℕ)

def pred1 : Prop := A > B
def pred2 : Prop := C > B ∧ C > A
def pred3 : Prop := C > B

theorem only_one_correct : (pred1 A B C ∨ pred2 A B C ∨ pred3 A B C) ∧ 
                           (pred1 A B C → ¬ pred2 A B C ∧ ¬ pred3 A B C) ∧
                           (pred2 A B C → ¬ pred1 A B C ∧ ¬ pred3 A B C) ∧
                           (pred3 A B C → ¬ pred1 A B C ∧ ¬ pred2 A B C) →
                           A > B ∧ B > C :=
begin
  sorry
end

end only_one_correct_l412_412447


namespace pm_perpendicular_bc_l412_412571

theorem pm_perpendicular_bc
  {A B C M E F P : Point}
  (hMmid : midpoint M B C)
  (hCircle : circle A M)
  (hE : on_circle E (circle A M))
  (hF : on_circle F (circle A M))
  (hE_AC : E ∈ line A C)
  (hF_AB : F ∈ line A B)
  (hTangent_E : tangent_circle P E (circle A M))
  (hTangent_F : tangent_circle P F (circle A M)) :
  perpendicular (line P M) (line B C) := 
sorry

end pm_perpendicular_bc_l412_412571


namespace real_solution_x_condition_l412_412930

theorem real_solution_x_condition (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
by
  sorry

end real_solution_x_condition_l412_412930


namespace pure_imaginary_a_eq_1_l412_412536

theorem pure_imaginary_a_eq_1 {a : ℝ} (h : (a-1) * (a+1+complex.i) = a^2 - 1 + (a-1) * complex.i) : (∃ b : ℝ, a = b ∧ a = 1) :=
sorry

end pure_imaginary_a_eq_1_l412_412536


namespace inequality_bound_l412_412224

theorem inequality_bound (a b c d e p q : ℝ) (hpq : 0 < p ∧ p ≤ q)
  (ha : p ≤ a ∧ a ≤ q) (hb : p ≤ b ∧ b ≤ q) (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
sorry

end inequality_bound_l412_412224


namespace part1_find_r_part2_sum_sequence_l412_412129

-- Condition: Sum of first n terms of arithmetic sequence
def S (n : ℕ) (r : ℤ) : ℤ := n^2 + r

-- Condition: Arithmetic sequence a_n with sum S(n) = n^2 + r
def a (n : ℕ) (r : ℤ) : ℤ := S n r - S (n-1) r

-- Condition: The b sequence definition
def b (n : ℕ) (r : ℤ) : ℤ := (a n r + 1) / 2

-- Target: Prove r = 0
theorem part1_find_r : ∀ (r : ℤ), (2 * a 2 r = a 1 r + a 3 r) → r = 0 := 
by
    intros
    -- Proof steps will go here
    sorry

-- Target: Sum of the first n terms of the given sequence
def T (n : ℕ) (r : ℤ) : ℚ := 
  (∑ i in Finset.range n, (1 : ℚ) / (b i r * b (i + 1) r))

-- Given a_n = 2n - 1 and r = 0
theorem part2_sum_sequence (n : ℕ) : ∑ i in Finset.range n, 1/ ((i : ℚ) * ((i + 1 : ℚ))) = n / (n + 1) :=
by
    intros
    -- Proof steps will go here
    sorry

end part1_find_r_part2_sum_sequence_l412_412129


namespace find_a5_of_geometric_sequence_l412_412542

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = a n * r

theorem find_a5_of_geometric_sequence (a : ℕ → ℝ) (h : geometric_sequence a)
  (h₀ : a 1 = 1) (h₁ : a 9 = 3) : a 5 = Real.sqrt 3 :=
sorry

end find_a5_of_geometric_sequence_l412_412542


namespace kite_diagonal_l412_412588

variable (A B C D : Type) [BD_numeric : ∀ (p q : Type), Prop → Prop]
variable (h1 : (BD_numeric AB CD (7 = 7)))
variable (h2 : (BD_numeric BC DA (19 = 11)))
variable (h3 : (BD_numeric AB DA 19))
variable (BD : ℤ)

theorem kite_diagonal :
  AB = 7 → BC = 19 → CD = 7 → DA = 11 → AB = CD ∧ BC = DA ∧ BD ≥ 13 ∧ BD ≤ 25 ∧ ∃ BD : ℤ, BD = 15 :=
by
  sorry

end kite_diagonal_l412_412588


namespace integer_values_count_l412_412987

theorem integer_values_count (x : ℤ) : 
  (∃ n : ℕ, n = 8 ∧ {x : ℤ | x^2 < 9 * x}.to_finset.card = n) :=
by
  sorry

end integer_values_count_l412_412987


namespace angle_MON_is_90_deg_l412_412597

-- Definition of the problem setup
variables {A B C M N O : Type}
variables [P : Point A] [Q : Point B] [R : Point C] [S : Point M] [T : Point N] [U : Point O]

-- Conditions
axiom angle_bisector_AM :
  is_angle_bisector A M C [P, Q, R, S]
axiom median_BN :
  is_median B N C [Q, R, T]
axiom intersect_at_O :
  intersects_at O [S, T, U]
axiom equal_areas :
  area_of_triangle A B M [P, Q, S] = area_of_triangle M N C [S, T, R]

-- Conclusion to prove
theorem angle_MON_is_90_deg :
  angle M O N [S, U, T] = 90 :=
by
  sorry

end angle_MON_is_90_deg_l412_412597


namespace find_a2_l412_412524

theorem find_a2 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * (a n - 1))
  (h2 : S 1 = a 1)
  (h3 : S 2 = a 1 + a 2) :
  a 2 = 4 :=
sorry

end find_a2_l412_412524


namespace vector_y_solution_l412_412638

open Matrix

theorem vector_y_solution :
  ∃ s : ℝ,
    let x := s • (col_vec [2, -1, 1] : Matrix (Fin 3) (Fin 1) ℝ),
        y := (col_vec [4, 2, -2] : Matrix (Fin 3) (Fin 1) ℝ) - x in
    y = col_vec [14 / 3, 4 / 3, -4 / 3] ∧
    ∀ a b, a ⬝ b = 0 := sorry

end vector_y_solution_l412_412638


namespace range_of_a_l412_412534

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
    (h_deriv : ∀ x ∈ set.Ioi 0, f' x = f'(x))
    (h_diff_eq : ∀ x ∈ set.Ioi 0, x * (f' x) - x * (f x) = exp x)
    (h_f1 : f 1 = 2 * exp 1)
    (h_f_constraint : f (1 - 1/(2*a)) ≤ exp (1/exp 1)) :
  (1/2 : ℝ) < a ∧ a ≤ exp 1 / (2 * (exp 1 - 1)) :=
sorry

end range_of_a_l412_412534


namespace part1_monotonic_increasing_interval_part2_decreasing_interval_part3_extreme_points_interval_l412_412281

-- Part 1
theorem part1_monotonic_increasing_interval (f : ℝ → ℝ) (a : ℝ) 
  (h_eq : f = λ x, x * (x^2 - 3 * x - 9)): 
  (∀ x, (3 * (x + 1) * (x - 3) > 0) → x ∈ set.Ioo (-∞) (-1) ∪ set.Ioo 3 ∞) := 
sorry

-- Part 2
theorem part2_decreasing_interval (f : ℝ → ℝ) (a : ℝ) 
  (h_eq : f = λ x, x * (x^2 - 3 * x + a)) : 
  (∀ x ∈ set.Ioo 1 2, (3 * x^2 - 6 * x + a ≤ 0)) → a ∈ set.Iio 0 := 
sorry

-- Part 3
theorem part3_extreme_points_interval (f : ℝ → ℝ) (a : ℝ) (x1 x2 : ℝ) 
  (h_eq : f = λ x, x * (x^2 - 3 * x + a))
  (h_extreme_points : x1 ∈ set.Ioo 0 2 ∧ x2 ∈ set.Ioo 0 2)
  (h_condition : |f(x1) - f(x2)| > |f(x1) + f(x2)|) : 
  0 < a ∧ a < 2.25 := 
sorry

end part1_monotonic_increasing_interval_part2_decreasing_interval_part3_extreme_points_interval_l412_412281


namespace maintain_order_time_l412_412372

theorem maintain_order_time :
  ∀ (x : ℕ), 
  (let ppl_per_min_norm := 9
   let ppl_per_min_cong := 3
   let total_people := 36 
   let teacher_time_saved := 6

   let time_without_order := total_people / ppl_per_min_cong
   let time_with_order := time_without_order - teacher_time_saved

   let ppl_passed_while_order := ppl_per_min_cong * x
   let ppl_passed_norm_order := ppl_per_min_norm * (time_with_order - x)

   ppl_passed_while_order + ppl_passed_norm_order = total_people) → 
  x = 3 :=
sorry

end maintain_order_time_l412_412372


namespace functional_equation_implies_constant_l412_412959

noncomputable def f : ℕ+ → ℕ+

theorem functional_equation_implies_constant :
  (∀ m n : ℕ+, f (m + n) * f (m - n) = f (m ^ 2)) →
  ∀ n : ℕ+, f n = 1 :=
by
  sorry

end functional_equation_implies_constant_l412_412959


namespace other_x_intercept_vertex_symmetric_l412_412103

theorem other_x_intercept_vertex_symmetric (a b c : ℝ)
  (h_vertex : ∀ x y : ℝ, (4, 10) = (x, y) → y = a * x^2 + b * x + c)
  (h_intercept : ∀ x : ℝ, (-1, 0) = (x, 0) → a * x^2 + b * x + c = 0) :
  a * 9^2 + b * 9 + c = 0 :=
sorry

end other_x_intercept_vertex_symmetric_l412_412103


namespace gcd_of_1237_and_1849_l412_412963

def gcd_1237_1849 : ℕ := 1

theorem gcd_of_1237_and_1849 : Nat.gcd 1237 1849 = gcd_1237_1849 := by
  sorry

end gcd_of_1237_and_1849_l412_412963


namespace candy_distribution_l412_412937

theorem candy_distribution :
  let candies := 8
  let bags := 3
  let atleast_one_per_bag (r b w : ℕ) := r ≥ 1 ∧ b ≥ 1 ∧ w ≥ 1 ∧ r + b + w = candies
  ∃ (r b w : ℕ), atleast_one_per_bag r b w ∧ (∑∑∑(r b w), r * b * w) = 846720 := sorry

end candy_distribution_l412_412937


namespace drums_needed_for_profit_l412_412611

def cost_to_enter_contest : ℝ := 10
def money_per_drum : ℝ := 0.025
def money_needed_for_profit (drums_hit : ℝ) : Prop :=
  drums_hit * money_per_drum > cost_to_enter_contest

theorem drums_needed_for_profit : ∃ D : ℝ, money_needed_for_profit D ∧ D = 400 :=
  by
  use 400
  sorry

end drums_needed_for_profit_l412_412611


namespace cos_triple_angle_l412_412198

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l412_412198


namespace day_100th_of_year_N_minus_1_l412_412603

/-- Define a function to calculate the day of the week given a specific day of the year and a reference weekday. -/
def day_of_week (day year : ℕ) (start_day : ℕ) : ℕ :=
  (start_day + day - 1) % 7

/-- Define if a year is a leap year -/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

noncomputable def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

/-- Main theorem to prove the day of the 100th day of year N-1 given the specific conditions. -/
theorem day_100th_of_year_N_minus_1 (N : ℕ)
  (h1 : day_of_week 300 N 2 = 2)
  (h2 : day_of_week 200 (N+1) 2 = 2) :
  day_of_week 100 (N-1) (day_of_week 1 (N-1) (day_of_week 365 (N-2) (day_of_week 1 (N-2) 2))) = 4 :=
sorry

end day_100th_of_year_N_minus_1_l412_412603


namespace af2_plus_bfg_plus_cg2_geq_0_l412_412321

theorem af2_plus_bfg_plus_cg2_geq_0 (a b c : ℝ) (f g : ℝ) :
  (a * f^2 + b * f * g + c * g^2 ≥ 0) ↔ (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) := 
sorry

end af2_plus_bfg_plus_cg2_geq_0_l412_412321


namespace distance_from_focus_to_line_l412_412705

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412705


namespace cos_triple_angle_l412_412196

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l412_412196


namespace ticket_price_l412_412301

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l412_412301


namespace analytical_expression_of_f_range_of_m_for_non_monotonicity_minimum_value_of_g_on_interval_l412_412127
open Real

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2*x + 15
def g (x m : ℝ) : ℝ := (1 - 2*m)*x - f(x)

theorem analytical_expression_of_f :
  ∀ x : ℝ, f(x + 1) - f(x) = -2*x + 1 ∧ f(2) = 15 :=
by 
  intro x
  split
  - sorry
  - sorry

theorem range_of_m_for_non_monotonicity :
  ∀ m : ℝ, ¬MonotonicOn (λ x : ℝ, g x m) (Set.Icc 0 2) ↔ -1/2 < m ∧ m < 3/2 :=
by 
  intro m
  sorry

theorem minimum_value_of_g_on_interval :
  ∀ (m : ℝ), 
  m ≤ -1/2 → g 0 m = -15 ∧ 
  (-1/2 < m ∧ m < 3/2 → g (m + 1/2) m = -m^2 - m - 61/4) ∧
  (m ≥ 3/2 → g 2 m = -4*m - 13) :=
by 
  intro m
  sorry

end analytical_expression_of_f_range_of_m_for_non_monotonicity_minimum_value_of_g_on_interval_l412_412127


namespace min_height_required_kingda_ka_l412_412644

-- Definitions of the given conditions
def brother_height : ℕ := 180
def mary_relative_height : ℚ := 2 / 3
def growth_needed : ℕ := 20

-- Definition and statement of the problem
def marys_height : ℚ := mary_relative_height * brother_height
def minimum_height_required : ℚ := marys_height + growth_needed

theorem min_height_required_kingda_ka :
  minimum_height_required = 140 := by
  sorry

end min_height_required_kingda_ka_l412_412644


namespace probability_of_four_green_marbles_l412_412665

-- Define the conditions: 10 green marbles, 5 purple marbles
def green_marbles : ℕ := 10
def purple_marbles : ℕ := 5
def total_marbles : ℕ := green_marbles + purple_marbles
def draws : ℕ := 7
def green_probability : ℚ := green_marbles / total_marbles
def purple_probability : ℚ := purple_marbles / total_marbles
def success_draws : ℕ := 4

-- Calculate the binomial coefficient
def binomial_coefficient : ℕ := nat.choose draws success_draws

-- Calculate the probability
def probability : ℚ := binomial_coefficient * (green_probability ^ success_draws) * (purple_probability ^ (draws - success_draws))

-- Prove that the resulting probability is approximately 0.256
theorem probability_of_four_green_marbles :
  probability ≈ 0.256 := 
begin
  -- We calculate the exact value 
  have exact_value : probability = 35 * ((2/3)^4) * ((1/3)^3), 
  {
    sorry
  },
  -- Numerically, this simplifies approximately to 0.256
  have numerical_value : 35 * ((2/3)^4) * ((1/3)^3) ≈ 0.256, 
  {
    sorry
  },
  exact eq.trans exact_value numerical_value
end

end probability_of_four_green_marbles_l412_412665


namespace percentage_of_cars_parked_illegally_l412_412241

theorem percentage_of_cars_parked_illegally
  (total_cars : ℕ)
  (towed_cars : ℕ)
  (illegally_parked_not_towed_percent : ℝ)
  (towed_percent: ℝ)
  (illegally_parked_towed_percent : ℝ)
  (percentage_illegally_parked : ℝ) :
  (total_cars = 100) →
  (towed_cars = total_cars * towed_percent) →
  (illegally_parked_towed_percent = 1.0 - illegally_parked_not_towed_percent) →
  (towed_cars = ilisub_graphs<|vq_4967|>id_graphs пог.)увуеьниMy_queue_nexts
end percentage_of_cars_parked_illegally_l412_412241


namespace periodic_derivatives_trig_l412_412518

theorem periodic_derivatives_trig (n : ℕ) (x : ℝ) :
  let f : ℕ → (ℝ → ℝ) := λ n, match n % 4 with
    | 0 => sin
    | 1 => cos
    | 2 => λ x, -sin x
    | 3 => λ x, -cos x
    | _ => sin -- Just to satisfy match exhaustive, never reached due to mod 4
  in f 2013 x = cos x :=
by
  sorry

end periodic_derivatives_trig_l412_412518


namespace region_area_l412_412832

theorem region_area : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 8 = 3*y - 6*x + 9) → 
  (π * (sqrt ((153 : ℝ) / 4))^2 = (153 * π) / 4) :=
by
  intro h
  sorry

end region_area_l412_412832


namespace cost_price_of_cloths_l412_412891

-- Definitions based on conditions
def SP_A := 8500 / 85
def Profit_A := 15
def CP_A := SP_A - Profit_A

def SP_B := 10200 / 120
def Profit_B := 12
def CP_B := SP_B - Profit_B

def SP_C := 4200 / 60
def Profit_C := 10
def CP_C := SP_C - Profit_C

-- Theorem to prove the cost prices
theorem cost_price_of_cloths :
    CP_A = 85 ∧
    CP_B = 73 ∧
    CP_C = 60 :=
by
    sorry

end cost_price_of_cloths_l412_412891


namespace bounded_function_solutions_l412_412962

theorem bounded_function_solutions (f : ℤ → ℤ) (bounded : ∃ C, ∀ n, |f n| ≤ C)
  (cond : ∀ n k, f(n+k) + f(k-n) = 2 * f(k) * f(n)) :
  (∀ n, f n = 0) ∨ (∀ n, f n = 1) ∨ (∀ n, (n % 2 = 0 → f n = 1) ∧ (n % 2 = 1 → f n = -1)) :=
by
  sorry

end bounded_function_solutions_l412_412962


namespace sum_of_ceil_sqrt_l412_412949

theorem sum_of_ceil_sqrt :
  (∑ n in finset.range (16 - 10 + 1), ⌈ real.sqrt (10 + n : ℝ) ⌉) +
  (∑ n in finset.range (25 - 17 + 1), ⌈ real.sqrt (17 + n : ℝ) ⌉) +
  (∑ n in finset.range (36 - 26 + 1), ⌈ real.sqrt (26 + n : ℝ) ⌉) +
  (∑ n in finset.range (40 - 37 + 1), ⌈ real.sqrt (37 + n : ℝ) ⌉) = 167 :=
by
  sorry

end sum_of_ceil_sqrt_l412_412949


namespace sum_of_series_l412_412862

theorem sum_of_series : 
  (6 + 16 + 26 + 36 + 46) + (14 + 24 + 34 + 44 + 54) = 300 :=
by
  sorry

end sum_of_series_l412_412862


namespace next_term_in_geometric_sequence_l412_412836

theorem next_term_in_geometric_sequence (y : ℝ) : 
  let a := 3
  let r := 4*y 
  let t4 := 192*y^3 
  r * t4 = 768*y^4 :=
by
  sorry

end next_term_in_geometric_sequence_l412_412836


namespace unit_digit_8_pow_1533_l412_412394

theorem unit_digit_8_pow_1533 : (8^1533 % 10) = 8 := by
  sorry

end unit_digit_8_pow_1533_l412_412394


namespace geometric_sequence_middle_term_find_a_in_geometric_sequence_l412_412813

theorem geometric_sequence_middle_term (a : ℝ) 
  (h : ∃ a : ℝ, 1 * a * 16) : a^2 = 16 :=
begin
  sorry,
end

theorem find_a_in_geometric_sequence (a : ℝ) 
  (h : a^2 = 16) : a = 4 ∨ a = -4 :=
begin
  sorry,
end

end geometric_sequence_middle_term_find_a_in_geometric_sequence_l412_412813


namespace average_of_three_quantities_l412_412338

theorem average_of_three_quantities (a b c d e : ℝ) 
    (h1 : (a + b + c + d + e) / 5 = 8)
    (h2 : (d + e) / 2 = 14) :
    (a + b + c) / 3 = 4 := 
sorry

end average_of_three_quantities_l412_412338


namespace integral_by_parts_general_l412_412859

variable {u v : ℝ → ℝ}
variable {a b : ℝ}
variable {n : ℕ}

theorem integral_by_parts_general
  (hu_cont : ∀ k : ℕ, k ≤ n → ContinuousOn (deriv^[k] u) (set.Icc a b))
  (hv_cont : ∀ k : ℕ, k ≤ n → ContinuousOn (deriv^[k] v) (set.Icc a b)) :
  ∫ x in a..b, u x * (deriv^[n] v) x = 
  (∑ i in finset.range n, (-1)^i * (deriv^[i] u) x * (deriv^[n-1-i] v) x) a b 
  - ∫ x in a..b, (deriv^[n] u) x * v x := 
sorry

end integral_by_parts_general_l412_412859


namespace necessary_condition_for_pure_imaginary_l412_412113

-- Given conditions: a, b ∈ ℝ
variables (a b : ℝ)

-- Definition of pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop :=
  ∃ y : ℝ, z = complex.I * y

-- Main theorem statement
theorem necessary_condition_for_pure_imaginary :
  (a = b) ↔ is_pure_imaginary (⟨a - b, a + b⟩) :=
by
  sorry

end necessary_condition_for_pure_imaginary_l412_412113


namespace distance_O_B_eq_1620_l412_412369

-- Defining the conditions
variables (A B O : Type) 
variables (dist : A → O → ℕ) (dist : O → B → ℕ) (dist : A → B → ℕ) 
variables (time12 time36 : ℕ)
variables (x : ℕ)

-- Conditions:
variables (A_eq_O : dist A O = 1620)
variables (equidistant_O : dist A B = 2 * x)
variables (meet_at_B : dist A B = 1620 + 3 * (1620 - x))

-- Proof Problem Statement
theorem distance_O_B_eq_1620 :
  dist O B = 1620 :=
by
  sorry

end distance_O_B_eq_1620_l412_412369


namespace f_5_5_l412_412535

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_even (x : ℝ) : f x = f (-x) := sorry

lemma f_recurrence (x : ℝ) : f (x + 2) = - (1 / f x) := sorry

lemma f_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) : f x = x := sorry

theorem f_5_5 : f 5.5 = 2.5 :=
by
  sorry

end f_5_5_l412_412535


namespace design_decimal_die_l412_412383

/-- Proof that it is possible to design a "decimal die" using a regular icosahedron
such that each number from 0 to 9 appears with equal probability. -/
theorem design_decimal_die : ∃ (icosahedron : Type) (label : fin 20 → fin 10),
  (∀ (n : fin 10), count (λ x, label x = n) (finset.fin_range 20).val = 2) :=
sorry

end design_decimal_die_l412_412383


namespace ratio_BM_MD_l412_412260

-- Definitions of the conditions
def BC : ℝ := 3
def CD : ℝ := 5
def angle_BCM : ℝ := π / 4  -- 45 degrees in radians
def angle_MCD : ℝ := π / 3  -- 60 degrees in radians

-- Theorem statement
theorem ratio_BM_MD : 
  (√6) / 5 := sorry

end ratio_BM_MD_l412_412260


namespace isosceles_triangle_x_sum_l412_412379

theorem isosceles_triangle_x_sum :
  ∀ (x : ℝ), (∃ (a b : ℝ), a + b + 60 = 180 ∧ (a = x ∨ b = x) ∧ (a = b ∨ a = 60 ∨ b = 60))
  → (60 + 60 + 60 = 180) :=
by
  intro x h
  sorry

end isosceles_triangle_x_sum_l412_412379


namespace distance_from_focus_to_line_l412_412697

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412697


namespace distinct_values_count_l412_412062

theorem distinct_values_count : 
  ∃ n, 
  (∀ (a b : ℕ), 
    (a ∈ {1, 3, 5, 7, 9, 11, 13, 15}.to_set) → 
    (b ∈ {1, 3, 5, 7, 9, 11, 13, 15}.to_set) → 
    (ab_value (a b) = ab_value (b a))) → 
  n = 36 :=
begin
  sorry
end

where ab_value (a b : ℕ) := a * b + a + b

end distinct_values_count_l412_412062


namespace ferris_wheel_capacity_l412_412423

theorem ferris_wheel_capacity :
  let num_seats := 4
  let people_per_seat := 4
  num_seats * people_per_seat = 16 := 
by
  let num_seats := 4
  let people_per_seat := 4
  sorry

end ferris_wheel_capacity_l412_412423


namespace distance_from_hyperbola_focus_to_line_l412_412753

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412753


namespace transformation_proof_l412_412376

def f (x : ℝ) : ℝ := 3 * Real.sin (1 / 2 * x + Real.pi / 5)
def g (x : ℝ) : ℝ := 3 * Real.sin (1 / 2 * x - Real.pi / 5)

theorem transformation_proof :
  ∀ x : ℝ, g (x - 4 * Real.pi / 5) = f x :=
by
  sorry

end transformation_proof_l412_412376


namespace Sn_minus_fraction_4_pow_a_n_l412_412549

-- Definitions
def seq_a (n : ℕ) : ℝ
def S (n : ℕ) : ℝ
axiom seq_a_init : seq_a 1 = 1
axiom seq_a_recur (n : ℕ) : seq_a n + seq_a (n + 1) = (1 / 4)^n
axiom S_def (n : ℕ) : S n = ∑ (i : ℕ) in finset.range n, (4 ^ i) * seq_a (i + 1)

-- Theorem statement
theorem Sn_minus_fraction_4_pow_a_n (n : ℕ) :
  S n - (4 ^ n / 5) * seq_a n = n / 5 :=
sorry

end Sn_minus_fraction_4_pow_a_n_l412_412549


namespace geometric_seq_fraction_l412_412567

theorem geometric_seq_fraction (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
  (h2 : (a 1 + 3 * a 3) / (a 2 + 3 * a 4) = 1 / 2) : 
  (a 4 * a 6 + a 6 * a 8) / (a 6 * a 8 + a 8 * a 10) = 1 / 16 :=
by
  sorry

end geometric_seq_fraction_l412_412567


namespace cos_triplet_angle_l412_412215

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l412_412215


namespace charlie_alice_balance_l412_412466

-- Define the expenditures of Charlie, Alice, and Bob
def expenditure_charlie : ℕ := 150
def expenditure_alice : ℕ := 180
def expenditure_bob : ℕ := 210

-- Calculate the total expenditure and individual share
def total_expenditure := expenditure_charlie + expenditure_alice + expenditure_bob
def individual_share := total_expenditure / 3

-- Charlie and Alice make payments to Bob to balance the shares
def c := individual_share - expenditure_charlie
def a := individual_share - expenditure_alice

-- Deduce the value of c - a
theorem charlie_alice_balance : c - a = 30 :=
by 
    have h_total : total_expenditure = 150 + 180 + 210 := rfl
    have h_share : individual_share = total_expenditure / 3 := rfl
    have h_c : c = individual_share - expenditure_charlie := rfl
    have h_a : a = individual_share - expenditure_alice := rfl
    calc 
      c - a 
        = (individual_share - expenditure_charlie) - (individual_share - expenditure_alice) : by rw [h_c, h_a]
    ... = 30 - 0 : by rw [individual_share, show individual_share = 180 by rfl, show 180 - 150 = 30 by rfl, show 180 - 180 = 0 by rfl]
    ... = 30 : by rfl

end charlie_alice_balance_l412_412466


namespace eccentricity_of_ellipse_l412_412138

/-- Given that F₁ and F₂ are the two foci of the ellipse C, and P is a point on C.
    If |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic sequence, then the eccentricity
    of C is 1/2. -/
theorem eccentricity_of_ellipse {C : Type} (F₁ F₂ P : C)
  (on_ellipse : P ∈ C) (is_foci : (F₁, F₂) ∈ C)
  (arith_seq : ∃ a b c, a + c = 2 * b ∧ |P - F₁| = a ∧ |F₁ - F₂| = b ∧ |P - F₂| = c) :
  eccentricity C = 1 / 2 :=
sorry

end eccentricity_of_ellipse_l412_412138


namespace sum_of_divisors_prime_products_eq_l412_412494

theorem sum_of_divisors_prime_products_eq (p1 q1 p2 q2 : ℕ) (hp1 : Nat.Prime p1) (hq1 : Nat.Prime q1) 
(hp2 : Nat.Prime p2) (hq2 : Nat.Prime q2) (h : σ (p1 * q1) = σ (p2 * q2)) :
  (1 + p1) * (1 + q1) = (1 + p2) * (1 + q2) :=
by
  sorry

end sum_of_divisors_prime_products_eq_l412_412494


namespace equal_trees_per_path_l412_412107

theorem equal_trees_per_path 
  (trees : Quadrant → Nat)
  (paths : List (Quadrant × Quadrant)) :
  (∀ path in paths, trees path.1 + trees path.2 = 2) ↔ 
  ((trees Quadrant.I = 1 ∧ trees Quadrant.II = 1 ∧ trees Quadrant.III = 1 ∧ trees Quadrant.IV = 1) ∨
   (trees Quadrant.I = 2 ∧ trees Quadrant.II = 0 ∧ trees Quadrant.III = 1 ∧ trees Quadrant.IV = 1)) :=
sorry

end equal_trees_per_path_l412_412107


namespace no_prime_divisible_by_45_l412_412186

theorem no_prime_divisible_by_45 : ∀ (p : ℕ), Prime p → ¬ (45 ∣ p) :=
by {
  intros p h_prime h_div,
  have h_factors := Nat.factors_unique,
  sorry
}

end no_prime_divisible_by_45_l412_412186


namespace problem_1_problem_2_problem_3_l412_412173

-- Define given circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4

-- (1) Define the proof of sinθ
theorem problem_1 (θ : ℝ) :
  ∃ l, ∃ A B : ℝ × ℝ, line_through_center (C1) l θ ∧ 
    intersects (C2) l A B ∧ 
    A_is_midpoint (center C1) B A ∧  (sin θ = ∀d. distance (center C2) l d * (√22 / 20)) := 
sorry

-- (2) Define the proof of the fixed point (4,1)
theorem problem_2 (m : ℝ) :
  ∃ P M N : ℝ × ℝ, 
    secant_lines_through (C2) P M N (m, 1) ∧ 
    fixed_point_m (4, 1) := 
sorry

-- (3) Define the proof of the range of lengths for ST
theorem problem_3 (x0 y0 : ℝ) : 
  point_on_circle (C2) x0 y0 ∧ 
  ∃ S T : ℝ × ℝ, 
    tangents_from (C1) (x0, y0) S T ∧
    length_ST S T ∈  (√2, 5√2 / 4) := 
sorry

end problem_1_problem_2_problem_3_l412_412173


namespace no_prime_divisible_by_45_l412_412182

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l412_412182


namespace incorrect_statement_b_l412_412991

theorem incorrect_statement_b
  (A : ∀ x, x > 1 → y(x) > 0)
  (B : (∃ x1 x2 : ℝ, x1 = 1 + sqrt 3 ∧ x2 = 1 - sqrt 3)
     ∧ (y(1) = -3)
     ∧ (∀ x, (x - 1)² - 3))
  : ∃ b, b = B ∧ b = False := 
begin
   sorry
end

end incorrect_statement_b_l412_412991


namespace sum_combinations_is_fibonacci_l412_412101

noncomputable def fib : Nat → Nat
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem sum_combinations_is_fibonacci (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range (Nat.floor ((n + 1) / 2) + 1),
    Nat.choose (n - i + 1) i) = fib (n + 1) :=
by
  sorry

end sum_combinations_is_fibonacci_l412_412101


namespace probability_three_diff_suits_l412_412668

theorem probability_three_diff_suits :
  let num_cards := 52
  let num_suits := 4
  let cards_per_suit := num_cards / num_suits
  -- Number of ways to choose 3 different cards from a deck
  let total_ways := finset.card (finset.comb_finset (finset.Ico 0 num_cards) 3)
  -- Number of ways to choose 1 card of each suit
  let ways_diff_suits := finset.card 
    (finset.product 
      (finset.product 
        (finset.Ico 0 cards_per_suit) 
        (finset.Ico cards_per_suit (2 * cards_per_suit))) 
      (finset.Ico (2 * cards_per_suit) num_cards))
  -- The probability is the ratio of these two numbers
  ways_diff_suits / total_ways = 169 / 425 := 
sorry

end probability_three_diff_suits_l412_412668


namespace film_radius_proof_l412_412884

-- Defining the given dimensions and conditions
def box_length := 8
def box_width := 4
def box_height := 15
def film_thickness := 0.2

-- Defining the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- Defining the radius of the circular film
def film_radius (volume : ℝ) : ℝ := real.sqrt (volume / (film_thickness * real.pi))

-- Proving the radius of the film given the conditions
theorem film_radius_proof : film_radius box_volume = real.sqrt(2400 / real.pi) :=
sorry

end film_radius_proof_l412_412884


namespace AC_perp_BD_l412_412286

-- Define four points in space
variables (A B C D : Type*)

-- Define the distances between points
variable (dist : A → A → ℝ)

-- Conditions
def AB_eq_BC := dist A B = dist B C
def CD_eq_DA := dist C D = dist D A

-- Statement to prove
theorem AC_perp_BD (h1 : AB_eq_BC dist) (h2 : CD_eq_DA dist) : 
  let M := midpoint dist A C in
  ⟪A, B⟫ * ⟪C, D⟫ = 0 :=
by sorry -- Proof will be filled in based on the steps provided


end AC_perp_BD_l412_412286


namespace range_of_a_l412_412114

noncomputable def condition_p (a : ℝ) : Prop := (4 * a - 3)^2 - 1 > 0

noncomputable def condition_q (a : ℝ) : Prop :=
  let re := (a + 1) / 2
  let im := (1 - a) / 2
  re > 0 ∧ im > 0

theorem range_of_a (a : ℝ) : (condition_p a ∧ ¬ condition_q a) ∨ (¬ condition_p a ∧ condition_q a) →
  (a ≤ -1 ∨ (a ≥ 1/2 ∧ a ≠ 1)) :=
begin
  sorry
end

end range_of_a_l412_412114


namespace pow_mod_equiv_l412_412839

theorem pow_mod_equiv (512 : ℕ) (n : ℕ) (hn : 512 ≡ 5 [MOD 13]) : 512 ^ 512 ≡ 1 [MOD 13] := by
  sorry

end pow_mod_equiv_l412_412839


namespace monotonic_decreasing_interval_l412_412349

theorem monotonic_decreasing_interval (f : ℝ → ℝ) (log_fn : ∀ u, log u = log2 u)
  (h_def : ∀ x, 0 < x → x < 4 → 4 * x - x ^ 2 > 0) :
  ∀ x, 2 < x → x < 4 → deriv (log (4 * x - x ^ 2)) < 0 := sorry

end monotonic_decreasing_interval_l412_412349


namespace number_of_integers_satisfying_inequality_l412_412985

theorem number_of_integers_satisfying_inequality :
  ∃ S : Finset ℤ, (∀ x ∈ S, x^2 < 9 * x) ∧ S.card = 8 :=
by
  sorry

end number_of_integers_satisfying_inequality_l412_412985


namespace multiples_of_four_l412_412365

theorem multiples_of_four (n : ℕ) (h : 25 = n - 3) : 
  ∃ m, 25 = (m + 1) - 1 ∧ m = 112 := 
by 
  exact ⟨112, rfl, rfl⟩
  sorry

end multiples_of_four_l412_412365


namespace math_problem_l412_412389

-- Statement of the theorem
theorem math_problem :
  (0.66)^3 - ((0.1)^3 / ((0.66)^2 + 0.066 + (0.1)^2)) = 0.3612 :=
by
  sorry -- Proof is not required

end math_problem_l412_412389


namespace calculate_area_of_region_l412_412911

theorem calculate_area_of_region :
  let region := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = 12}
  ∃ area, area = 17 * Real.pi
:= by
  sorry

end calculate_area_of_region_l412_412911


namespace chess_tournament_max_N_l412_412575

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l412_412575


namespace each_gets_10_fish_l412_412263

-- Define the constants and conditions
constant Ittymangnark Kingnook Oomyapeck : Type
constant fish : Type
constant eyes_of_fish : fish → ℕ
constant oomyapeck_eats_eyes : ℕ := 22
constant oomyapeck_gives_dog : ℕ := 2
constant total_eyes_eaten_by_oomyapeck : ℕ := oomyapeck_eats_eyes - oomyapeck_gives_dog
constant number_of_fish_oomyapeck_eats : ℕ := total_eyes_eaten_by_oomyapeck / 2
constant total_fish_divided : ℕ := number_of_fish_oomyapeck_eats
constant fish_split_equally : ℕ := total_fish_divided

-- The theorem statement
theorem each_gets_10_fish (day : Type) (H : Ittymangnark ≠ Kingnook ∧ Kingnook ≠ Oomyapeck ∧ Ittymangnark ≠ Oomyapeck) : 
  (number_of_fish_oomyapeck_eats = 10) ∧ (fish_split_equally = 10) :=
by {
  sorry
}

end each_gets_10_fish_l412_412263


namespace honors_distribution_correct_l412_412026

noncomputable def honors_distribution : ℕ := 114

theorem honors_distribution_correct :
  let honors := ["advanced individual", "business elite", "moral model", "new long march commando", "annual excellent employee"],
      people := ["person1", "person2", "person3"] in
  (∀ p ∈ people, ∃ h ∈ honors, h ≠ "moral model" ∨ h ≠ "new long march commando") →
  ∃ (distribution : list (string ⊕ string) × string), 
    list.length distribution = 5 ∧ 
    (∀ p ∈ people, ∃ h ∈ distribution, h.2 = p ∧ h.1 ∈ honors) ∧
    distribution.permutations.length = honors_distribution := sorry

end honors_distribution_correct_l412_412026


namespace volume_of_pyramid_l412_412434

-- Define the conditions
def pyramid_conditions : Prop :=
  ∃ (s h : ℝ),
  s^2 = 256 ∧
  ∃ (h_A h_C h_B : ℝ),
  ((∃ h_A, 128 = 1/2 * s * h_A) ∧
  (∃ h_C, 112 = 1/2 * s * h_C) ∧
  (∃ h_B, 96 = 1/2 * s * h_B)) ∧
  h^2 + (s/2)^2 = h_A^2 ∧
  h^2 = 256 - (s/2)^2 ∧
  h^2 + (s/4)^2 = h_B^2

-- Define the theorem
theorem volume_of_pyramid :
  pyramid_conditions → 
  ∃ V : ℝ, V = 682.67 * Real.sqrt 3 :=
sorry

end volume_of_pyramid_l412_412434


namespace distance_from_focus_to_line_l412_412689

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412689


namespace find_polynomial_R_l412_412410

-- Define the polynomials S(x), Q(x), and the remainder R(x)

noncomputable def S (x : ℝ) := 7 * x ^ 31 + 3 * x ^ 13 + 10 * x ^ 11 - 5 * x ^ 9 - 10 * x ^ 7 + 5 * x ^ 5 - 2
noncomputable def Q (x : ℝ) := x ^ 4 + x ^ 3 + x ^ 2 + x + 1
noncomputable def R (x : ℝ) := 13 * x ^ 3 + 5 * x ^ 2 + 12 * x + 3

-- Statement of the proof
theorem find_polynomial_R :
  ∃ (P : ℝ → ℝ), ∀ x : ℝ, S x = P x * Q x + R x := sorry

end find_polynomial_R_l412_412410


namespace distance_from_focus_to_line_l412_412707

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412707


namespace distance_from_focus_to_line_l412_412698

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412698


namespace continuous_f_imp_zero_l412_412615

noncomputable def sequence_of_functions (f : ℝ → ℝ) (n : ℕ) : (ℝ → ℝ) :=
  if n = 0 then f else λ x, ∫ t in 0..x, sequence_of_functions f (n-1) t

theorem continuous_f_imp_zero (f : ℝ → ℝ) 
  (h_continuous : continuous f)
  (h_def : ∀ n : ℕ, ∀ x : ℝ, sequence_of_functions f n x = 
    (if n = 0 then f x else ∫ t in 0..x, sequence_of_functions f (n-1) t))
  (h_zero_at_one : ∀ n : ℕ, sequence_of_functions f n 1 = 0)
  (x : ℝ) : 
  f x = 0 :=
by
  sorry

end continuous_f_imp_zero_l412_412615


namespace distance_hyperbola_focus_to_line_l412_412778

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412778


namespace range_a_l412_412623

noncomputable def f (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 3 then -x^2 + 4*x
  else if 3 < x ∧ x ≤ 4 then (x^2 + 2) / x
  else 0

def g (a x : ℝ) : ℝ := a*x + 1

theorem range_a (a : ℝ) :
  (∀ x1 ∈ set.Ioo (-4 : ℝ) (-2 : ℝ), ∃ x2 ∈ set.Icc (-2 : ℝ) (1 : ℝ), 
    g a x2 = f x1) ↔ (a ∈ set.Iic (-5/8) ∪ set.Ici (5/16)) :=
begin
  sorry
end

end range_a_l412_412623


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412791

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412791


namespace no_primes_divisible_by_45_l412_412190

theorem no_primes_divisible_by_45 : ∀ p : ℕ, prime p → ¬ (45 ∣ p) := 
begin
  sorry
end

end no_primes_divisible_by_45_l412_412190


namespace maria_paid_9_l412_412304

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l412_412304


namespace cos_triple_angle_l412_412216

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l412_412216


namespace solve_inequality_l412_412333

theorem solve_inequality : 
  ∀ x : ℝ, (| (2 * x - 1) / (x - 1) | > 2) ↔ (x ∈ set.Ioo (3 / 4) 1 ∪ set.Ioi 1) := by
  sorry

end solve_inequality_l412_412333


namespace max_AC_plus_sqrt3_BC_l412_412239

theorem max_AC_plus_sqrt3_BC
  (a b c : ℝ)
  (A B C : ℝ)
  (lengths : a² + b² = √3 * a * b + c²)
  (ab_eq_1 : ab = 1) :
  ∃ (max_val : ℝ), max_val = 2 * √7 ∧ max_val = AC + √3 * BC :=
by {
  sorry
}

end max_AC_plus_sqrt3_BC_l412_412239


namespace distance_hyperbola_focus_to_line_l412_412772

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412772


namespace v3_value_at_5_l412_412830

noncomputable def polynomial := λ x : ℝ, 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

def v3 (x : ℝ) := ((5 * x + 2) * x + 3.5) * x - 2.6

theorem v3_value_at_5 :
  v3 5 = 138.5 :=
sorry

end v3_value_at_5_l412_412830


namespace find_quotient_l412_412682

-- Defining the problem conditions as variables
variables (L S : Nat) (remainder quotient : Int)

-- Given conditions 
def condition1 : Prop := L = 2520
def condition2 : Prop := L - S = 2415
def condition3 : Prop := remainder = 15
def condition4 : Prop := L = S * quotient + remainder

-- The proof problem statement
theorem find_quotient (hL : condition1) (h_diff : condition2) (h_rem : condition3) (h_div : condition4) : quotient = 23 := by
  sorry

end find_quotient_l412_412682


namespace complex_number_real_part_l412_412633

theorem complex_number_real_part (x : ℝ) (z1 z2 : ℂ) 
    (h1 : z1 = 1 + complex.i) 
    (h2 : z2 = x - complex.i) 
    (hw : is_real (z1 * z2)) : 
    x = 1 := 
begin
    sorry
end

end complex_number_real_part_l412_412633


namespace number_of_bus_routes_is_seven_l412_412340

theorem number_of_bus_routes_is_seven :
  ∃ (R : ℕ), (∀ (stops : finset ℕ), 
    (∀ s₁ s₂ ∈ stops, s₁ ≠ s₂ → 
    ∃! r : ℕ, 
      r ∈ R ∧ 
      ∃ (route : finset ℕ), 
        route.card = 3 ∧ 
        r ∈ route ∧ 
        s₁ ∈ route ∧ 
        s₂ ∈ route) ∧ 
    (∀ r ∈ R, (∃ (route : finset ℕ), route.card = 3 ∧ r ∈ route)) ∧
        R.card > 1)  ∧
       R.card = 7 := 
begin 
  sorry 
end

end number_of_bus_routes_is_seven_l412_412340


namespace walking_rate_on_escalator_l412_412904

/-- If the escalator moves at 7 feet per second, is 180 feet long, and a person takes 20 seconds to cover this length, then the rate at which the person walks on the escalator is 2 feet per second. -/
theorem walking_rate_on_escalator 
  (escalator_rate : ℝ)
  (length : ℝ)
  (time : ℝ)
  (v : ℝ)
  (h_escalator_rate : escalator_rate = 7)
  (h_length : length = 180)
  (h_time : time = 20)
  (h_distance_formula : length = (v + escalator_rate) * time) :
  v = 2 :=
by
  sorry

end walking_rate_on_escalator_l412_412904


namespace ellipse_focus_point_coordinates_l412_412530

theorem ellipse_focus_point_coordinates :
  ∀ (x y : Real),
    (x^2 / 16 + y^2 / 9 = 1) →
    (∃ (P : ℝ × ℝ),
      P.1 = x ∧ P.2 = y ∧
      x > 0 ∧ y > 0 ∧
      ∠(F1, P, F2) = 60° ∧ 
      (x, y) = (8 * sqrt 7 / 7, 3 * sqrt 21 / 7)) :=
by
  intros x y h_ellipse h1 h2 h3
  -- The proof would go here
  sorry

end ellipse_focus_point_coordinates_l412_412530


namespace simplify_expr1_simplify_expr2_l412_412673

variable (a x y : ℝ)

theorem simplify_expr1 : a - 2a + 3a = 2a := by
  sorry

theorem simplify_expr2 : 3 * (2 * x - 7 * y) - (4 * x - 10 * y) = 2 * x - 11 * y := by
  sorry

end simplify_expr1_simplify_expr2_l412_412673


namespace thomas_total_drawings_l412_412373

theorem thomas_total_drawings :
  let colored_pencil_drawings := 14
  let blending_marker_drawings := 7
  let charcoal_drawings := 4
  colored_pencil_drawings + blending_marker_drawings + charcoal_drawings = 25 := 
by
  sorry

end thomas_total_drawings_l412_412373


namespace inequality_solution_l412_412503

theorem inequality_solution (x : ℝ) : (x^3 - 10 * x^2 > -25 * x) ↔ (0 < x ∧ x < 5) ∨ (5 < x) := 
sorry

end inequality_solution_l412_412503


namespace linear_regression_and_prediction_l412_412440

-- Definitions of given sums
def sum_x : ℕ := 80
def sum_y : ℕ := 20
def sum_xy : ℕ := 184
def sum_x_sq : ℕ := 720
def n : ℕ := 10

-- Definitions of calculated means
def x_bar := (sum_x : ℝ) / (n : ℝ)
def y_bar := (sum_y : ℝ) / (n : ℝ)

-- Definitions of parameters
def b_hat := (sum_xy - n * x_bar * y_bar) / (sum_x_sq - n * x_bar ^ 2)
def a_hat := y_bar - b_hat * x_bar

-- Definition of the regression equation
def regression_eq (x : ℝ) := b_hat * x + a_hat

-- Definition of the predicted value
def predict (x : ℝ) := regression_eq x

-- Main theorem statement
theorem linear_regression_and_prediction :
  (x_bar = 8) →
  (y_bar = 2) →
  (b_hat = 0.3) →
  (a_hat = -0.4) →
  (regression_eq 8 = 2) :=
by
  sorry -- Proof would go here

end linear_regression_and_prediction_l412_412440


namespace dealer_profit_percentage_l412_412014

-- Definitions of conditions
def cost_price (C : ℝ) : ℝ := C
def list_price (C : ℝ) : ℝ := 1.5 * C
def discount_rate : ℝ := 0.1
def discounted_price (C : ℝ) : ℝ := (1 - discount_rate) * list_price C
def price_for_45_articles (C : ℝ) : ℝ := 45 * discounted_price C
def cost_for_40_articles (C : ℝ) : ℝ := 40 * cost_price C

-- Statement of the problem
theorem dealer_profit_percentage (C : ℝ) (h₀ : C > 0) :
  (price_for_45_articles C - cost_for_40_articles C) / cost_for_40_articles C * 100 = 35 :=  
sorry

end dealer_profit_percentage_l412_412014


namespace magnitude_OB_l412_412146

-- Define the complex numbers corresponding to the vectors
def OA : ℂ := -2 + complex.i
def AB : ℂ := 3 + 2 * complex.i

-- Calculate OB
def OB : ℂ := OA + AB

-- Formulate the problem
theorem magnitude_OB : complex.abs OB = real.sqrt 10 := by
  -- Insert proof here
  sorry

end magnitude_OB_l412_412146


namespace mod_inverse_addition_l412_412480

theorem mod_inverse_addition :
  (3 * (7⁻¹ : ℚ) + 5 * (13⁻¹ : ℚ)) % 97 = 73 :=
by
  have h1: (7 : ℤ)⁻¹ % 97 = 83 := 
    sorry
  have h2: (13 : ℤ)⁻¹ % 97 = 82 := 
    sorry
  sorry

end mod_inverse_addition_l412_412480


namespace maria_paid_9_l412_412306

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l412_412306


namespace arrange_books_l412_412587

theorem arrange_books : ∃ n : ℕ, 
  (∃ (s1 s2 s3 s4 m1 m2 m3 m4 : Type),
   -- Ensure the chosen books are distinct
   ∀ (h_s1 : s1 ≠ s2) (h_s1 : s1 ≠ s3) (h_s1 : s1 ≠ s4) (h_s2 : s2 ≠ s3)
      (h_s2 : s2 ≠ s4) (h_s3 : s3 ≠ s4)
      (h_m1 : m1 ≠ m2) (h_m1 : m1 ≠ m3) (h_m1 : m1 ≠ m4) (h_m2 : m2 ≠ m3)
      (h_m2 : m2 ≠ m4) (h_m3 : m3 ≠ m4),
   -- Ensure science books are on both ends
   (∃ (book_ends : s1 × s2), ∃ book_mid : m1,
    -- Ensure remaining books are arranged in remaining positions
    (∃ (remaining_books : list (m2 ∪ m3 ∪ m4 ∪ s3 ∪ s4)), 
     -- The number of ways to arrange the books
     n = 12 * 4 * 120)) ) ∧ n = 5760 := sorry

end arrange_books_l412_412587


namespace distance_from_focus_to_line_l412_412741

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412741


namespace mean_salary_of_6_employees_l412_412350

theorem mean_salary_of_6_employees :
  let salaries := [1000, 2500, 3100, 3650, 1500, 2000] in
  let n := 6 in
  (salaries.sum / n : ℝ) = 2458.33 := by
  sorry

end mean_salary_of_6_employees_l412_412350


namespace sum_of_squares_l412_412374

theorem sum_of_squares (x y z : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (h_sum : x * 1 + y * 2 + z * 3 = 12) : x^2 + y^2 + z^2 = 56 :=
by
  sorry

end sum_of_squares_l412_412374


namespace michael_jessica_meet_time_l412_412314

theorem michael_jessica_meet_time :
  ∀ (initial_distance : ℝ) (michael_speed_ratio : ℝ) (combined_rate : ℝ) (initial_time : ℝ)
     (jessica_delay : ℝ) (michael_speed : ℝ) (jessica_speed : ℝ),
  (initial_distance = 24) →
  (michael_speed_ratio = 2) →
  (combined_rate = 1.2) →
  (initial_time = 8) →
  (jessica_delay = 4) →
  (michael_speed = michael_speed_ratio * jessica_speed) →
  (combined_rate = michael_speed + jessica_speed) →
  (jessica_speed = combined_rate / (1 + michael_speed_ratio)) →
  michael_speed = combined_rate * (michael_speed_ratio / (1 + michael_speed_ratio)) →
  let distance_after_8_minutes := initial_distance - combined_rate * initial_time in
  let michael_distance_during_delay := michael_speed * jessica_delay in
  let remaining_distance := distance_after_8_minutes - michael_distance_during_delay in
  let time_after_delay := remaining_distance / combined_rate in
  initial_time + jessica_delay + time_after_delay = 21.33 :=
begin
  intros,
  sorry
end

end michael_jessica_meet_time_l412_412314


namespace distance_from_right_focus_to_line_l412_412770

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412770


namespace monotonic_intervals_f_range_of_a_l412_412038

open Real

-- Function definition
def f (x : ℝ) : ℝ := 1 / (x * log x)

-- Conditions: x > 0 and x ≠ 1
def domain (x : ℝ) : Prop := x > 0 ∧ x ≠ 1

-- Proof of monotonic intervals
theorem monotonic_intervals_f : 
  (∀ x, domain x → 0 < x ∧ x < 1 / exp 1 → deriv f x > 0) ∧ 
  (∀ x, domain x → x > 1 / exp 1 ∧ x ≠ 1 → deriv f x < 0) := 
sorry

-- Proof for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → (1 / x) * log 2 > a * log x) → a > -exp 1 * log 2 :=
sorry

end monotonic_intervals_f_range_of_a_l412_412038


namespace general_formula_sum_first_n_terms_l412_412153

-- Definitions for the arithmetic sequence
variable (t n : ℕ)

def a1 := t^2 - t
def a2 := 4
def a3 := t^2 + t

-- Arithmetic sequence conditions
axiom seq_arithmetic (an : ℕ → ℕ) : (∀ n : ℕ, an (n + 1) - an n = an 1 - an 0) 

-- Given conditions
axiom seq_a1 : a1 t = t^2 - t
axiom seq_a2 : a2 = 4
axiom seq_a3 : a3 t = t^2 + t

-- Prove the general formula for an
theorem general_formula (an : ℕ → ℕ) (h1 : an 1 = a1 t)
  (h2 : an 2 = a2)
  (h3 : an 3 = a3 t)
  (h_arith : seq_arithmetic an) :
  an = (λ n, 2 * n) ∨ an = (λ n, 8 - 2 * n) := sorry

-- Given b_n satisfies log_2 b_n = a_n
def bn (an : ℕ → ℕ) : ℕ → ℕ := λ n, 2^(an n)

-- Sum of the sequence (a_n - 1) * b_n
def Sn (an bn : ℕ → ℕ) : ℕ → ℕ := 
  λ n, (finset.range n).sum (λ k, (an k - 1) * (bn k))

-- Prove the sum of the first n terms
theorem sum_first_n_terms (an : ℕ → ℕ) (bn : ℕ → ℕ)
  (h_an_inc : ∀ n m, n < m → an n < an m)
  (h_bn : ∀ n, log 2 (bn n) = an n) :
  Sn an bn n = (6 * n - 5) * 4^(n + 1) / 9 + 20 / 9 := sorry

end general_formula_sum_first_n_terms_l412_412153


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412783

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412783


namespace no_primes_divisible_by_45_l412_412192

theorem no_primes_divisible_by_45 : ∀ p : ℕ, prime p → ¬ (45 ∣ p) := 
begin
  sorry
end

end no_primes_divisible_by_45_l412_412192


namespace percent_equivalence_l412_412395

theorem percent_equivalence (y : ℝ) : 0.30 * (0.60 * y) = 0.18 * y :=
by sorry

end percent_equivalence_l412_412395


namespace distance_from_focus_to_line_l412_412739

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412739


namespace cos_triple_angle_l412_412205

theorem cos_triple_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l412_412205


namespace number_of_piles_l412_412311

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end number_of_piles_l412_412311


namespace complement_of_union_in_U_l412_412297

-- Define the universal set U
def U : Set ℕ := {x | x < 6 ∧ x > 0}

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of A ∪ B in U
def complement_U_union_A_B : Set ℕ := {x | x ∈ U ∧ x ∉ (A ∪ B)}

theorem complement_of_union_in_U : complement_U_union_A_B = {2, 4} :=
by {
  -- Placeholder for the proof
  sorry
}

end complement_of_union_in_U_l412_412297


namespace find_distance_between_anchor_points_l412_412418

noncomputable def distance_between_anchor_points 
    (pole_height : ℝ) (wire_length : ℝ) 
    (anchor_points_equally_spaced : Bool) 
    (equilateral_triangle : Bool) 
    (distance_to_anchor : ℝ) 
    (medians_equal : Bool) : ℝ 
    :=
    if (pole_height = 70) ∧ 
       (wire_length = 490) ∧ 
       (anchor_points_equally_spaced) ∧ 
       (equilateral_triangle) ∧ 
       (distance_to_anchor = 10 * real.sqrt 2352) ∧ 
       (medians_equal) 
    then 840 
    else 0

theorem find_distance_between_anchor_points :
    distance_between_anchor_points 70 490 true true (10 * real.sqrt 2352) true = 840 :=
sorry

end find_distance_between_anchor_points_l412_412418


namespace combined_mpg_l412_412664

theorem combined_mpg (m : ℝ) : 
  let ray_mpg := 35
  let tom_mpg := 25
  let alice_mpg := 20
  (3 * m) / (m / ray_mpg + m / tom_mpg + m / alice_mpg) = 25.30 :=
by
  let ray_mpg := 35
  let tom_mpg := 25
  let alice_mpg := 20
  let total_distance := 3 * m
  let total_gas_used := (m / ray_mpg) + (m / tom_mpg) + (m / alice_mpg)
  let combined_mpg := total_distance / total_gas_used
  have h : combined_mpg = 2100 / 83 := sorry
  rw h
  norm_num 

end combined_mpg_l412_412664


namespace part1_l412_412544

variable {x : ℝ} {a : ℝ}
def f (x : ℝ) (a : ℝ) := a * x + x * Real.log x

theorem part1 (h : ∀ x, deriv (f x) x = 1 + x⁻¹) (h1 : deriv (f x) (1 / Real.exp 1) = 1) : a = 1 :=
by
  sorry

end part1_l412_412544


namespace distance_from_hyperbola_focus_to_line_l412_412751

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412751


namespace fraction_power_rule_evaluate_fraction_power_l412_412477

theorem fraction_power_rule (a b n : ℕ) (hb : b ≠ 0) :
  (a / b : ℚ) ^ n = (a ^ n) / (b ^ n) := by sorry

theorem evaluate_fraction_power : (5 / 3 : ℚ) ^ 6 = 15625 / 729 :=
  by rw [fraction_power_rule 5 3 6]; norm_num

end fraction_power_rule_evaluate_fraction_power_l412_412477


namespace calories_per_slice_l412_412936

theorem calories_per_slice
  (total_calories : ℕ)
  (portion_eaten : ℕ)
  (percentage_eaten : ℝ)
  (slices_in_cheesecake : ℕ)
  (calories_in_slice : ℕ) :
  total_calories = 2800 →
  percentage_eaten = 0.25 →
  portion_eaten = 2 →
  portion_eaten = percentage_eaten * slices_in_cheesecake →
  calories_in_slice = total_calories / slices_in_cheesecake →
  calories_in_slice = 350 :=
by
  intros
  sorry

end calories_per_slice_l412_412936


namespace sarah_proof_l412_412497

-- Defining cards and conditions
inductive Card
| P : Card
| A : Card
| C5 : Card
| C4 : Card
| C7 : Card

-- Definition of vowel
def is_vowel : Card → Prop
| Card.P => false
| Card.A => true
| _ => false

-- Definition of prime numbers for the sides
def is_prime : Card → Prop
| Card.C5 => true
| Card.C4 => false
| Card.C7 => true
| _ => false

-- Tom's statement
def toms_statement (c : Card) : Prop :=
is_vowel c → is_prime c

-- Sarah shows Tom was wrong by turning over one card
theorem sarah_proof : ∃ c, toms_statement c = false ∧ c = Card.A :=
sorry

end sarah_proof_l412_412497


namespace distance_from_hyperbola_focus_to_line_l412_412756

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l412_412756


namespace union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l412_412272

section
  def A : Set ℝ := {x : ℝ | ∃ q : ℚ, x = q}
  def B : Set ℝ := {x : ℝ | ¬ ∃ q : ℚ, x = q}

  theorem union_rational_irrational_is_real : A ∪ B = Set.univ :=
  by
    sorry

  theorem intersection_rational_irrational_is_empty : A ∩ B = ∅ :=
  by
    sorry
end

end union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l412_412272


namespace dice_real_number_probability_l412_412845

theorem dice_real_number_probability :
  let dice_set := {1, 2, 3, 4, 5, 6}
  let all_pairs := (dice_set × dice_set).to_finset
  let successful_pairs := {p ∈ all_pairs | (p.1 = p.2)}
  (successful_pairs.card : ℚ) / (all_pairs.card : ℚ) = 1 / 6 :=
by
  sorry

end dice_real_number_probability_l412_412845


namespace least_possible_m_plus_n_l412_412625

theorem least_possible_m_plus_n :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
  Nat.gcd (m + n) 330 = 1 ∧ 
  n^n ∣ m^m ∧ ¬ m ∣ n ∧ 
  m + n = 154 :=
by
sory

end least_possible_m_plus_n_l412_412625


namespace distance_from_focus_to_line_l412_412706

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412706


namespace max_possible_N_l412_412576

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l412_412576


namespace distance_from_right_focus_to_line_l412_412764

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l412_412764


namespace problem_k_bound_l412_412550

theorem problem_k_bound (k : ℝ) (A : Set ℕ) (h1 : A = {x : ℕ | 1 < x ∧ x < Real.log k}) (h2 : 3 ≤ A.card) : k > Real.exp 4 :=
by
  sorry

end problem_k_bound_l412_412550


namespace percentage_of_y_l412_412228

theorem percentage_of_y (x y P : ℝ) (h1 : 0.10 * x = (P/100) * y) (h2 : x / y = 2) : P = 20 :=
sorry

end percentage_of_y_l412_412228


namespace calculate_expression_l412_412915

theorem calculate_expression :
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by sorry

end calculate_expression_l412_412915


namespace find_m_l412_412128

theorem find_m (m : ℝ) : (∀ x > 0, x + 3^m / x ≥ 6) → m = 2 := by
  sorry

end find_m_l412_412128


namespace dot_product_a_b_l412_412621

variables (a b : ℝ^3) (θ : ℝ)

-- Assume the given conditions
axiom norm_a : ‖a‖ = 8
axiom norm_b : ‖b‖ = 10
axiom angle_ab : θ = real.pi / 4  -- 45 degrees in radians

-- Goal statement
theorem dot_product_a_b : a ⬝ b = 40 * real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end dot_product_a_b_l412_412621


namespace sum_of_specific_angles_l412_412856

def parallelogram_smaller_angle (x : ℝ) := 4 * x
def parallelogram_larger_angle (x : ℝ) := 11 * x
def quadrilateral_angles (y : ℝ) := (5 * y, 6 * y, 7 * y, 12 * y)

theorem sum_of_specific_angles: 
  ∀ (x y : ℝ),
    4 * x + 11 * x = 180 →
    5 * y + 6 * y + 7 * y + 12 * y = 360 →
    4 * x + 7 * y = 132 := 
by
  intros x y hpx hqy
  have hx : x = 180 / 15 := by linarith
  have hy : y = 360 / 30 := by linarith
  rw [hx, hy]
  calc 4 * (180 / 15) + 7 * (360 / 30)
    = 4 * 12 + 7 * 12 : by norm_num
    = 48 + 84 : by norm_num
    = 132 : by norm_num
  sorry

end sum_of_specific_angles_l412_412856


namespace xy_eq_yx_l412_412613

theorem xy_eq_yx (n : ℕ) (hn : n > 0) : 
  let x := (1 + (1 : ℚ)/n) ^ n,
      y := (1 + (1 : ℚ)/n) ^ (n + 1)
  in x^y = y^x :=
by
  sorry

end xy_eq_yx_l412_412613


namespace intervals_total_length_le_half_l412_412120

open Set

theorem intervals_total_length_le_half :
  ∃ (M : Set ℝ), (∀ a b ∈ M, a ≠ b → abs (a - b) ≠ 0.1) ∧ 
  (∀ a b ∈ M, a < b → (∃ I : Set ℝ, I ⊆ Ioo (a / 10 - 1 / 10) (b / 10)) ∧ 
  ⋃₀ (M ∩ I) = I) ∧ 
  (measure_theory.measure_space.volume (⋃₀ M : Set ℝ) = 1) ∧
  (∑_{k ∈ (finset.range 10)} (measure_theory.measure_space.measure 
    (Ioo (k / 10) ((k + 1) / 10) ∩ M)) ≤ 1 / 2) :=
sorry

end intervals_total_length_le_half_l412_412120


namespace point_inside_circle_range_l412_412235

theorem point_inside_circle_range (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) :=
  by
  sorry

end point_inside_circle_range_l412_412235


namespace assertion1_false_assertion2_true_assertion3_false_assertion4_false_l412_412037

section

-- Assertion 1: ∀ x ∈ ℝ, x ≥ 1 is false
theorem assertion1_false : ¬(∀ x : ℝ, x ≥ 1) := 
sorry

-- Assertion 2: ∃ x ∈ ℕ, x ∈ ℝ is true
theorem assertion2_true : ∃ x : ℕ, (x : ℝ) = x := 
sorry

-- Assertion 3: ∀ x ∈ ℝ, x > 2 → x ≥ 3 is false
theorem assertion3_false : ¬(∀ x : ℝ, x > 2 → x ≥ 3) := 
sorry

-- Assertion 4: ∃ n ∈ ℤ, ∀ x ∈ ℝ, n ≤ x < n + 1 is false
theorem assertion4_false : ¬(∃ n : ℤ, ∀ x : ℝ, n ≤ x ∧ x < n + 1) := 
sorry

end

end assertion1_false_assertion2_true_assertion3_false_assertion4_false_l412_412037


namespace find_triplets_l412_412488

theorem find_triplets (u v w : ℝ):
  (u + v * w = 12) ∧ 
  (v + w * u = 12) ∧ 
  (w + u * v = 12) ↔ 
  (u = 3 ∧ v = 3 ∧ w = 3) ∨ 
  (u = -4 ∧ v = -4 ∧ w = -4) ∨ 
  (u = 1 ∧ v = 1 ∧ w = 11) ∨ 
  (u = 11 ∧ v = 1 ∧ w = 1) ∨ 
  (u = 1 ∧ v = 11 ∧ w = 1) := 
sorry

end find_triplets_l412_412488


namespace inverse_proportional_sqrt_l412_412820

theorem inverse_proportional_sqrt (z w : ℝ) (k : ℝ) 
  (h1 : ∀ z w, z * (sqrt w) = k)
  (h2 : w = 4)
  (h3 : z = 4)
  (h4 : z = 2) :
  w = 16 := 
by
  sorry

end inverse_proportional_sqrt_l412_412820


namespace smallest_olympic_integer_correct_smallest_olympic_odd_integer_correct_l412_412125

def λ (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    let α := n.trailing_zero
    let n' := n.shiftRight α
    if is_square n then
      ((α - 1) * (number_of_divisors n') - 1) / 2
    else
      ((α - 1) * (number_of_divisors n')) / 2
  else
    (number_of_divisors n) / 2

def is_olympic (n : ℕ) : Prop :=
  λ n = 2021

def smallest_olympic_integer (n : ℕ) : Prop :=
  is_olympic n ∧ ∀ m : ℕ, is_olympic m → n ≤ m

def smallest_olympic_odd_integer (n : ℕ) : Prop :=
  is_olympic n ∧ n % 2 = 1 ∧ ∀ m : ℕ, is_olympic m ∧ m % 2 = 1 → n ≤ m

theorem smallest_olympic_integer_correct :
  smallest_olympic_integer (2^48 * 3^42 * 5) :=
sorry

theorem smallest_olympic_odd_integer_correct :
  smallest_olympic_odd_integer (3^46 * 5^42 * 7) :=
sorry

end smallest_olympic_integer_correct_smallest_olympic_odd_integer_correct_l412_412125


namespace distance_to_line_is_sqrt5_l412_412803

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412803


namespace solve_first_system_solve_second_system_l412_412677

-- Define the first system of equations
def first_system (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)

-- Define the solution to the first system
def solution1 (x y : ℝ) : Prop := (x = 3) ∧ (y = -2)

-- Define the second system of equations
def second_system (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Define the solution to the second system
def solution2 (x y : ℝ) : Prop := (x = 4) ∧ (y = -2)

-- Define the problem statement in Lean
theorem solve_first_system : ∃ x y : ℝ, first_system x y ↔ solution1 x y :=
by
  sorry

theorem solve_second_system : ∃ x y : ℝ, second_system x y ↔ solution2 x y :=
by
  sorry

end solve_first_system_solve_second_system_l412_412677


namespace problem1_problem2_l412_412042

theorem problem1 : 2 * log 3 2 - log 3 (32 / 9) + log 3 8 - 25 ^ log 5 3 = -7 := 
sorry

theorem problem2 : (9/4) ^ (1/2) - (-7.8)^0 - (27/8) ^ (2/3) + (2/3) ^ (-2) = 1/2 :=
sorry

end problem1_problem2_l412_412042


namespace final_answer_after_subtracting_l412_412020

theorem final_answer_after_subtracting (n : ℕ) (h : n = 990) : (n / 9) - 100 = 10 :=
by
  sorry

end final_answer_after_subtracting_l412_412020


namespace fifth_term_is_correct_l412_412345

theorem fifth_term_is_correct:
  ∀ (x y: ℝ),
  (let seq := [x + y, x - y, x / y, x * y] in
  (seq.nth 0 = some (x + y)) ∧
  (seq.nth 1 = some (x - y)) ∧
  (seq.nth 2 = some (x / y)) ∧
  (seq.nth 3 = some (x * y)) ∧
  ((x - y) - (x + y) = -2 * y) ∧
  ((x / y) = x - 3 * y) ∧
  (x = 3 * y ^ 2 / (y - 1)) ∧
  (x * y / (x / y) = y ^ 2) ∧
  (x * y * y ^ 2 = x * y ^ 3)) → 
  (x * y ^ 3 = 3 * y ^ 5 / (y - 1)) :=
by
  intros x y
  sorry

end fifth_term_is_correct_l412_412345


namespace cos_triplet_angle_l412_412213

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l412_412213


namespace farmer_pays_per_acre_per_month_l412_412868

-- Define the conditions
def total_payment : ℕ := 600
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Define the problem to prove
theorem farmer_pays_per_acre_per_month :
  length_of_plot * width_of_plot / square_feet_per_acre > 0 ∧
  total_payment / (length_of_plot * width_of_plot / square_feet_per_acre) = 60 :=
by
  -- skipping the actual proof for now
  sorry

end farmer_pays_per_acre_per_month_l412_412868


namespace exists_real_b_l412_412614

noncomputable theory

variables {G : Type*} [add_comm_group G] [fintype G]
variables {n : ℕ} {k : ℕ} (g : fin k → G) (m : ℕ)

theorem exists_real_b 
  (h1 : G.card = n)
  (h2 : function.injective g)
  (h3 : ∀ g_i ∈ finset.univ.image g, g_i ≠ 0)
  : ∃ (b : ℝ), b ∈ Ioo 0 1 ∧ 
    ∃ H > 0, H < ∞ ∧
    (0 < fintype.card G → (∃ H > 0, H < ∞)) :=
sorry

end exists_real_b_l412_412614


namespace taxi_fare_ride_distance_l412_412572

theorem taxi_fare_ride_distance
  (initial_fare : ℝ := 3)
  (initial_distance : ℝ := 0.5)
  (additional_rate : ℝ := 0.25)
  (additional_distance_unit : ℝ := 0.1)
  (total_amount : ℝ := 15)
  (tip : ℝ := 3) :
  let remaining_amount := total_amount - tip in
  let equation := initial_fare + 
                   additional_rate * ((x - initial_distance) / additional_distance_unit) = remaining_amount in
  x = 4.1 :=
by
  sorry

end taxi_fare_ride_distance_l412_412572


namespace determine_N_l412_412472

open Matrix

noncomputable def cross_product (a b : Vector3) : Vector3 :=
⟨a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x⟩

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 4, -3],
  ![4, 0, -7],
  ![3, -7, 0]
]

theorem determine_N (v : Vector3) : 
  mul_vec N v = cross_product ⟨7, -3, 4⟩ v :=
sorry

end determine_N_l412_412472


namespace geometric_series_sum_l412_412956

noncomputable def infinite_geometric_series_sum : ℚ := 
  ∑' n : ℕ, ((-1) ^ n * (3 ^ n / 5 ^ (n + 1))) * 5

theorem geometric_series_sum :
  infinite_geometric_series_sum = 125 / 102 :=
by
  -- Condition definitions
  let a := (5 : ℚ) / 3
  let r := (-9 : ℚ) / 25
  
  -- Sum calculation via formula
  have S := a / (1 - r)
  
  -- Substitute by the values of a and r
  have : a = 5 / 3 := rfl
  have : r = -9 / 25 := rfl
  
  -- Confirm the series sum is equal to the correct answer
  rw [this, this] at S
  exact (S = 125 / 102 : ℚ)
  
  sorry

end geometric_series_sum_l412_412956


namespace inclination_angle_range_is_correct_l412_412812

-- Define the range of inclination angles of a straight line
def inclination_angle_range : Set Real :=
  {x | 0 ≤ x ∧ x < π}

-- Proof statement
theorem inclination_angle_range_is_correct : inclination_angle_range = {x | 0 ≤ x ∧ x < π} :=
sorry

end inclination_angle_range_is_correct_l412_412812


namespace largest_integer_n_l412_412071

-- Define the floor function
def floor (x : ℝ) : ℤ := Int.ofNat ⌊x⌋

-- Define the problem statement
theorem largest_integer_n (n : ℕ) (h : floor (Real.sqrt n) = 5) : n ≤ 35 :=
by
  sorry

end largest_integer_n_l412_412071


namespace cos_triple_angle_l412_412207

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l412_412207


namespace solve_problem1_solve_problem2_l412_412863

noncomputable def problem1 (α : ℝ) : Prop :=
  (Real.tan (Real.pi / 4 + α) = 1 / 2) →
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -2

noncomputable def problem2 : Prop :=
  Real.sin (Real.pi / 12) * Real.sin (5 * Real.pi / 12) = 1 / 4

-- theorems to be proved
theorem solve_problem1 (α : ℝ) : problem1 α := by
  sorry

theorem solve_problem2 : problem2 := by
  sorry

end solve_problem1_solve_problem2_l412_412863


namespace undefined_values_l412_412106

theorem undefined_values (b : ℝ) : (b^2 - 9 = 0) ↔ (b = -3 ∨ b = 3) := by
  sorry

end undefined_values_l412_412106


namespace distance_from_focus_to_line_l412_412696

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412696


namespace sum_of_ceil_sqrt_l412_412948

theorem sum_of_ceil_sqrt :
  (∑ n in finset.range (16 - 10 + 1), ⌈ real.sqrt (10 + n : ℝ) ⌉) +
  (∑ n in finset.range (25 - 17 + 1), ⌈ real.sqrt (17 + n : ℝ) ⌉) +
  (∑ n in finset.range (36 - 26 + 1), ⌈ real.sqrt (26 + n : ℝ) ⌉) +
  (∑ n in finset.range (40 - 37 + 1), ⌈ real.sqrt (37 + n : ℝ) ⌉) = 167 :=
by
  sorry

end sum_of_ceil_sqrt_l412_412948


namespace find_a_l412_412506

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l412_412506


namespace solve_equation_l412_412487

theorem solve_equation (x : ℝ) :
  x^4 + (3 - x)^4 = 98 →
  x = 1.5 + sqrt ((33 + sqrt 238.75) / 4) ∨
  x = 1.5 - sqrt ((33 + sqrt 238.75) / 4) ∨
  x = 1.5 + sqrt ((33 - sqrt 238.75) / 4) ∨
  x = 1.5 - sqrt ((33 - sqrt 238.75) / 4) :=
by 
  sorry

end solve_equation_l412_412487


namespace distance_hyperbola_focus_to_line_l412_412774

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l412_412774


namespace distance_to_line_is_sqrt5_l412_412801

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l412_412801


namespace minimum_abs_diff_ln_2x_l412_412131

theorem minimum_abs_diff_ln_2x :
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ (∃ x : ℝ, x > 0 ∧ f x = g x2) →
  f x1 = g x2 ∧ ∀ y1 y2 : ℝ, y1 > 0 ∧ y2 > 0 ∧ f y1 = g y2 → 
  |x1 - x2| = (1 + Real.log 2) / 2 :=
by
  let f (x : ℝ) := Real.log x
  let g (x : ℝ) := 2 * x
  sorry

end minimum_abs_diff_ln_2x_l412_412131


namespace distance_from_focus_to_line_l412_412712

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l412_412712


namespace last_digit_B_l412_412965

theorem last_digit_B 
  (B : ℕ) 
  (h : ∀ n : ℕ, n % 10 = (B - 287)^2 % 10 → n % 10 = 4) :
  (B = 5 ∨ B = 9) :=
sorry

end last_digit_B_l412_412965


namespace proof_problem_l412_412033

variable (pots_basil pots_rosemary pots_thyme : ℕ)
variable (leaves_per_basil leaves_per_rosemary leaves_per_thyme : ℕ)
variable (total_leaves : ℕ)

-- Define the given conditions
def conditions : Prop :=
  pots_basil = 3 ∧
  leaves_per_basil = 4 ∧
  pots_rosemary = 9 ∧
  leaves_per_rosemary = 18 ∧
  pots_thyme = 6 ∧
  leaves_per_thyme = 30

-- Define the question and the correct answer
def correct_answer : Prop :=
  total_leaves = 354

-- Translate to proof problem
theorem proof_problem : conditions → (total_leaves = pots_basil * leaves_per_basil + pots_rosemary * leaves_per_rosemary + pots_thyme * leaves_per_thyme) → correct_answer :=
by
  intro h1 h2
  exact h2
  sorry -- proof placeholder

end proof_problem_l412_412033


namespace cars_with_neither_l412_412651

theorem cars_with_neither (total_cars air_bag power_windows both : ℕ) 
                          (h1 : total_cars = 65) (h2 : air_bag = 45)
                          (h3 : power_windows = 30) (h4 : both = 12) : 
                          (total_cars - (air_bag + power_windows - both) = 2) :=
by
  sorry

end cars_with_neither_l412_412651


namespace david_found_money_on_street_l412_412470

def david_found (initial_amount evan_remaining_amount watch_cost received_amount : ℕ) :=
  watch_cost - evan_remaining_amount - initial_amount

theorem david_found_money_on_street :
  ∀ (initial_amount evan_remaining_amount watch_cost received_amount : ℕ),
    initial_amount = 1 →
    evan_remaining_amount = 7 →
    watch_cost = 20 →
    david_found initial_amount evan_remaining_amount watch_cost 12 = received_amount →
    received_amount = 12 :=
by
  intros initial_amount evan_remaining_amount watch_cost received_amount h1 h2 h3 h4
  rw [h1, h2, h3, david_found]
  sorry

end david_found_money_on_street_l412_412470


namespace set_representation_l412_412296

variable {U : Set ℝ} (E F : Set ℝ)

-- Definitions based on the conditions
def U_def : Set ℝ := Set.univ
def E_def : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def F_def : Set ℝ := {x | -1 < x ∧ x < 5}
def compl_E : Set ℝ := {x | -3 < x ∧ x < 2}

-- The proof statement
theorem set_representation :
  (compl_E E_def) ∩ F_def = {x | -1 < x ∧ x < 2} := by
  sorry

end set_representation_l412_412296


namespace cos_and_sin_double_angle_l412_412537

variables (θ : ℝ)

-- Conditions
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi

def sin_theta (θ : ℝ) : Prop :=
  Real.sin θ = -1 / 3

-- Problem statement
theorem cos_and_sin_double_angle (h1 : is_in_fourth_quadrant θ) (h2 : sin_theta θ) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 ∧ Real.sin (2 * θ) = -(4 * Real.sqrt 2 / 9) :=
sorry

end cos_and_sin_double_angle_l412_412537


namespace number_of_integers_satisfying_inequality_l412_412986

theorem number_of_integers_satisfying_inequality :
  ∃ S : Finset ℤ, (∀ x ∈ S, x^2 < 9 * x) ∧ S.card = 8 :=
by
  sorry

end number_of_integers_satisfying_inequality_l412_412986


namespace parabola_vertex_on_x_axis_l412_412570

theorem parabola_vertex_on_x_axis (c : ℝ) :
  let y := λ x : ℝ, x^2 + 2 * x + c in
  (let vertex_y := (4 * 1 * c - 2^2) / (4 * 1) in vertex_y = 0) → c = 1 :=
by
  intro y vertex_y_eq_y
  sorry

end parabola_vertex_on_x_axis_l412_412570


namespace globe_motion_is_rotation_l412_412814

-- Define a new inductive type for motion alternatives
inductive Motion 
| Translation : Motion
| Rotation : Motion
| Neither : Motion

-- Define the condition that a globe has a certain type of motion (Rotation)
axiom globe_motion : Motion := Motion.Rotation

-- Formulate the theorem statement that the globe's motion is Rotation
theorem globe_motion_is_rotation : globe_motion = Motion.Rotation := 
sorry

end globe_motion_is_rotation_l412_412814


namespace sum_of_first_100_terms_l412_412251

variable (a : ℕ → ℝ)
variable h₁ : ∀ n, a (n + 1) - a n = 1 / 2
variable h₂ : (Finset.sum (Finset.filter (fun n => Odd n) (Finset.range 100)) (fun n => a n)) = 60

theorem sum_of_first_100_terms :
  (Finset.sum (Finset.range 100) (fun n => a n)) = 145 :=
by 
  sorry

end sum_of_first_100_terms_l412_412251


namespace correct_symmetric_point_l412_412335

noncomputable def imaginary_unit : ℂ :=
  complex.I

noncomputable def complex_point : ℂ :=
  2 * imaginary_unit / (1 - imaginary_unit)

def symmetric_point (z : ℂ) : ℂ :=
  ⟨-z.re, z.im⟩

theorem correct_symmetric_point :
  symmetric_point complex_point = (1 : ℂ) :=
by
  sorry

end correct_symmetric_point_l412_412335


namespace distance_from_right_focus_to_line_is_sqrt5_l412_412785

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l412_412785


namespace equal_distance_from_O_to_B1_B2_l412_412408

variables {A B C H H_1 H_2 B_1 B_2 O : Point}
variables (triangle_ABC : IsRightTriangle A B C)
variables (altitude_BH : IsAltitude B H)
variables (circle_in_ABH_tangent :
  InscribedCircleTangents A B H H_1 B_1)
variables (circle_in_CBH_tangent :
  InscribedCircleTangents C B H H_2 B_2)

theorem equal_distance_from_O_to_B1_B2 :
  ∀ (O_center_circumcircle :
    CircumcenterOfTriangle H_1 B H_2),
  Distance O B_1 = Distance O B_2 :=
by
  sorry

end equal_distance_from_O_to_B1_B2_l412_412408


namespace prism_faces_l412_412433

def is_prism (e : ℕ) (f : ℕ) := 
  ∃ L : ℕ, e = 3 * L ∧ f = L + 2

theorem prism_faces (e : ℕ) (f : ℕ) (h : e = 27) : 
  is_prism e f → f = 11 := 
by
  intro h1
  obtain ⟨L, hL1, hL2⟩ := h1
  rw [h] at hL1
  have hL : L = 9 := by
    linarith
  rw [hL] at hL2
  exact hL2

end prism_faces_l412_412433


namespace parabola_max_value_l412_412104

theorem parabola_max_value 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x = - (x + 1)^2 + 3) : 
  ∃ x, y x = 3 ∧ ∀ x', y x' ≤ 3 :=
by
  sorry

end parabola_max_value_l412_412104


namespace cos_triplet_angle_l412_412214

theorem cos_triplet_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triplet_angle_l412_412214


namespace cos_A_side_c_l412_412415

-- helper theorem for cosine rule usage
theorem cos_A (a b c : ℝ) (cosA cosB cosC : ℝ) (h : 3 * a * cosA = c * cosB + b * cosC) : cosA = 1 / 3 :=
by
  sorry

-- main statement combining conditions 1 and 2 with side value results
theorem side_c (a b c : ℝ) (cosA cosB cosC : ℝ) (h1 : 3 * a * cosA = c * cosB + b * cosC) (h2 : cosB + cosC = 0) (h3 : a = 1) : c = 2 :=
by
  have h_cosA : cosA = 1 / 3 := cos_A a b c cosA cosB cosC h1
  sorry

end cos_A_side_c_l412_412415


namespace ticket_price_l412_412302

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l412_412302


namespace technicians_count_l412_412339

-- Define the conditions
def avg_sal_all (total_workers : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 850

def avg_sal_technicians (teches : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 1000

def avg_sal_rest (others : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 780

-- The main theorem to prove
theorem technicians_count (total_workers : ℕ)
  (teches others : ℕ)
  (total_salary : ℕ) :
  total_workers = 22 →
  total_salary = 850 * 22 →
  avg_sal_all total_workers 850 →
  avg_sal_technicians teches 1000 →
  avg_sal_rest others 780 →
  teches + others = total_workers →
  1000 * teches + 780 * others = total_salary →
  teches = 7 :=
by
  intros
  sorry

end technicians_count_l412_412339


namespace continuous_at_x1_discontinuous_at_x2_l412_412475

-- Define the function
def f (x : ℝ) : ℝ := 4 ^ (1 / (3 - x))

-- Define the points
def x1 := 1
def x2 := 3

-- Statement to prove continuity at x1
theorem continuous_at_x1 : ContinuousAt f x1 := sorry

-- Statement to prove discontinuity at x2
theorem discontinuous_at_x2 : ¬ ContinuousAt f x2 := sorry

end continuous_at_x1_discontinuous_at_x2_l412_412475


namespace initial_speed_l412_412888

-- Definitions of the conditions
def distance : ℝ := 24
def speed_1 : ℝ := 12
def time_difference : ℝ := 2 / 3  -- 40 minutes is 2/3 hours

-- The theorem statement
theorem initial_speed (v : ℝ) (h : v ≠ 0) :
  (distance / v) - (distance / speed_1) = time_difference → v = 9 := by
  sorry

end initial_speed_l412_412888


namespace mul_neg_x_squared_cubed_l412_412045

theorem mul_neg_x_squared_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 :=
sorry

end mul_neg_x_squared_cubed_l412_412045


namespace probability_at_least_one_expired_l412_412028

theorem probability_at_least_one_expired (total_bottles expired_bottles selected_bottles : ℕ) : 
  total_bottles = 10 → expired_bottles = 3 → selected_bottles = 3 → 
  (∃ probability, probability = 17 / 24) :=
by
  sorry

end probability_at_least_one_expired_l412_412028


namespace bus_stop_time_l412_412405

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℕ) 
(distance : ℕ) (time_without_stoppages time_with_stoppages : ℝ) :
  speed_without_stoppages = 80 ∧ speed_with_stoppages = 40 ∧ distance = 80 ∧
  time_without_stoppages = distance / speed_without_stoppages ∧
  time_with_stoppages = distance / speed_with_stoppages →
  (time_with_stoppages - time_without_stoppages) * 60 = 30 :=
by
  sorry

end bus_stop_time_l412_412405


namespace probability_of_painted_cubes_identical_l412_412380

noncomputable def probability_of_identical_cubes : ℚ :=
  1 / 64

theorem probability_of_painted_cubes_identical :
  ∀ (cube1 cube2 : Cube),
  (∀ (face : Face), face ∈ cube1.faces → face ∈ {black, white, red}) ∧
  (∀ (face : Face), face ∈ cube2.faces → face ∈ {black, white, red}) ∧
  (∃ (bfaces1 bfaces2 : set Face),
      (bfaces1 ⊆ cube1.faces ∧ bfaces2 ⊆ cube2.faces) ∧
      (bfaces1.card ≥ 4 ∧ bfaces2.card ≥ 4) ∧
      (are_rotationally_identical cube1 cube2))
  → probability_of_identical_cubes = 1 / 64 := by 
  sorry

end probability_of_painted_cubes_identical_l412_412380


namespace speed_difference_l412_412895

theorem speed_difference (distance : ℕ) (time_jordan time_alex : ℕ) (h_distance : distance = 12) (h_time_jordan : time_jordan = 10) (h_time_alex : time_alex = 15) :
  (distance / (time_jordan / 60) - distance / (time_alex / 60) = 24) := by
  -- Lean code to correctly parse and understand the natural numbers, division, and maintain the theorem structure.
  sorry

end speed_difference_l412_412895


namespace more_green_than_yellow_l412_412364

-- Define constants
def red_peaches : ℕ := 2
def yellow_peaches : ℕ := 6
def green_peaches : ℕ := 14

-- Prove the statement
theorem more_green_than_yellow : green_peaches - yellow_peaches = 8 :=
by
  sorry

end more_green_than_yellow_l412_412364


namespace convert_15_deg_to_rad_l412_412063

theorem convert_15_deg_to_rad (deg_to_rad : ℝ := Real.pi / 180) : 
  15 * deg_to_rad = Real.pi / 12 :=
by sorry

end convert_15_deg_to_rad_l412_412063


namespace tori_trash_outside_classrooms_l412_412827

theorem tori_trash_outside_classrooms
  (total_trash : ℕ)
  (classroom_trash : ℕ)
  (h1 : total_trash = 1576)
  (h2 : classroom_trash = 344) :
  total_trash - classroom_trash = 1232 := 
by
  rw [h1, h2]
  exact Nat.sub_eq_iff_eq_add.mpr rfl

end tori_trash_outside_classrooms_l412_412827


namespace thursday_on_100th_day_of_N_minus_1_l412_412601

-- We define the necessary conditions:
def day_of_week : ℕ → ℕ
| 0 := 0 -- Sunday
| 1 := 1 -- Monday
| 2 := 2 -- Tuesday
| 3 := 3 -- Wednesday
| 4 := 4 -- Thursday
| 5 := 5 -- Friday
| 6 := 6 -- Saturday
| (n+1) := (day_of_week n + 1) % 7

def is_leap_year (n : ℕ) : Prop :=
  (n % 4 = 0 ∧ n % 100 ≠ 0) ∨ (n % 400 = 0)

theorem thursday_on_100th_day_of_N_minus_1 (N : ℕ) 
(h1 : day_of_week 299 = 2) -- 300th day of N is a Tuesday
(h2 : day_of_week 199 = 2) -- 200th day of N+1 is a Tuesday
(h3 : is_leap_year N)  -- N is a leap year
: day_of_week 99 = 4 := -- 100th day of N-1 is a Thursday
sorry

end thursday_on_100th_day_of_N_minus_1_l412_412601


namespace customers_per_table_l412_412445

theorem customers_per_table (total_tables : ℝ) (left_tables : ℝ) (total_customers : ℕ)
  (h1 : total_tables = 44.0)
  (h2 : left_tables = 12.0)
  (h3 : total_customers = 256) :
  total_customers / (total_tables - left_tables) = 8 :=
by {
  sorry
}

end customers_per_table_l412_412445


namespace theta_sufficient_sin_half_l412_412141

theorem theta_sufficient_sin_half (θ : ℝ) : 
  (θ = π / 6) → (sin θ = 1 / 2) ∧ 
  ((sin θ = 1 / 2) → (θ = π / 6 → False)) :=
by
  sorry

end theta_sufficient_sin_half_l412_412141


namespace carolyn_sum_of_removed_numbers_l412_412465

theorem carolyn_sum_of_removed_numbers (n : ℕ) (h : n = 6) : 
  let game : list ℕ := [1, 2, 3, 4, 5, 6],
      rounds : list (list ℕ × list ℕ) :=
        [(game, [2]),
         ([1, 2, 3, 4, 5, 6].erase 2 ++ [2], [1]),
         ([3, 4, 5, 6], []),
         ([3, 4, 5, 6].erase 6 ++ [6], [3]),
         ([3, 4, 5], []),
         ([4, 5], []),
         ([4, 5], [4, 5])],
      carolyn_moves := [2, 6],
      carolyn_sum := 2 + 6
  in carolyn_sum = 8 := 
by 
  have h_rounds : rounds = 
    [([1, 2, 3, 4, 5, 6], [2]), 
     ([3, 4, 5, 6], [1]), 
     ([3, 4, 5, 6], []), 
     ([3, 4, 5, 6], [3]), 
     ([4, 5], []), 
     ([4, 5], []), 
     ([4, 5], [4, 5])] := rfl, sorry

end carolyn_sum_of_removed_numbers_l412_412465


namespace ratio_AD_BC_l412_412592

-- Definitions of the conditions
def TriangleEquilateral (A B C : Type) : Prop :=
  sorry -- Placeholder definition for an equilateral triangle

def TriangleIsoscelesRight (B C D : Type) : Prop :=
  sorry -- Placeholder definition for an isosceles right triangle

-- Main hypothesis
variable (A B C D : Type)
variable [TriangleEquilateral A B C] [TriangleIsoscelesRight B C D]

-- Main theorem statement
theorem ratio_AD_BC : ∀ (A B C D : Type) [TriangleEquilateral A B C] [TriangleIsoscelesRight B C D], 
  (AD A C D) / (BC B C) = √(3/2) :=
by
  sorry -- Proof to be filled in

end ratio_AD_BC_l412_412592


namespace bones_in_graveyard_l412_412582

theorem bones_in_graveyard :
  let total_bones := 
    25 * 25 + 
    10 * 31 + 
    5 * 12 + 
    5 * 24 + 
    2 * 14 + 
    3 * 34 in
  total_bones = 1245 :=
by
  -- Definitions (conditions)
  let total_skeletons := 45
  let adult_women_fraction := 5 / 9
  let adult_men_fraction := 2 / 9
  let children_fraction := 1 / 9
  let teenagers_fraction := 1 / 9

  let adult_women_skeletons := total_skeletons * adult_women_fraction
  let adult_men_skeletons := total_skeletons * adult_men_fraction
  let children_skeletons := total_skeletons * children_fraction
  let teenagers_skeletons := total_skeletons * teenagers_fraction
  
  let infants_skeletons := 2
  let dogs_skeletons := 3

  let adult_women_bones := 25
  let adult_men_bones := adult_women_bones + 6
  let children_bones := adult_women_bones - 13
  let teenagers_bones := 2 * children_bones
  let infants_bones := adult_women_bones - 11
  let dogs_bones := adult_men_bones + 3

  -- Total bones calculation (final proof statement)
  have : adult_women_skeletons * adult_women_bones + 
         adult_men_skeletons * adult_men_bones + 
         children_skeletons * children_bones + 
         teenagers_skeletons * teenagers_bones + 
         infants_skeletons * infants_bones + 
         dogs_skeletons * dogs_bones = total_bones,
  from sorry, -- calculation and verification goes here
  exact this

end bones_in_graveyard_l412_412582


namespace no_prime_divisible_by_45_l412_412185

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l412_412185


namespace max_possible_N_l412_412578

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l412_412578


namespace luminosity_ratio_of_sun_and_sirius_l412_412585

theorem luminosity_ratio_of_sun_and_sirius :
  let m1 := -26.7
  let m2 := -1.45
  let lhs := m2 - m1
  let rhs := (5 / 2) * log10 (E1 / E2)
  lhs = rhs → E1 / E2 = 10 ^ 10.1 :=
begin
  intro h,
  sorry
end

end luminosity_ratio_of_sun_and_sirius_l412_412585


namespace ajays_monthly_income_l412_412852

theorem ajays_monthly_income :
  ∀ (I : ℝ), 
  (0.50 * I) + (0.25 * I) + (0.15 * I) + 9000 = I → I = 90000 :=
by
  sorry

end ajays_monthly_income_l412_412852


namespace modified_sum_eq_3S_add_9_l412_412359

variable (a b S : ℝ)
variable (h₁ : a + b = S)

def modified_first (a : ℝ) : ℝ := 3 * a + 4
def modified_second (b : ℝ) : ℝ := 2 * b + 5

theorem modified_sum_eq_3S_add_9 : (modified_first a) + (modified_second b) = 3 * S + 9 :=
by
  calc
    (3 * a + 4) + (2 * b + 5) = 3 * a + 2 * b + 4 + 5 : by ring
    ...                       = 3 * (a + b) + 9       : by ring
    ...                       = 3 * S + 9             : by rw [h₁]

end modified_sum_eq_3S_add_9_l412_412359


namespace distance_from_focus_to_line_l412_412684

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l412_412684


namespace sin_double_angle_sum_zero_l412_412978

theorem sin_double_angle_sum_zero
  (α β : ℝ)
  (h : sin α * sin β + cos α * cos β = 0) :
  sin (2 * α) + sin (2 * β) = 0 :=
sorry

end sin_double_angle_sum_zero_l412_412978


namespace repeating_decimal_to_fraction_l412_412848

theorem repeating_decimal_to_fraction : (0.2727272727 : ℝ) = 3 / 11 := 
sorry

end repeating_decimal_to_fraction_l412_412848


namespace permutation_five_out_of_seven_l412_412366

theorem permutation_five_out_of_seven : ∃ (k : ℕ), k = nat.fact 7 / nat.fact (7 - 5) ∧ k = 2520 :=
by 
  have h := nat.factorial_eq_prod_range_one_succ 7,
  have h_r := nat.factorial_eq_prod_range_one_succ (7 - 5),
  -- The required number of permutations P(7, 5) = 7! / (7-5)!
  sorry

end permutation_five_out_of_seven_l412_412366


namespace number_of_music_files_l412_412903

-- The conditions given in the problem
variable {M : ℕ} -- M is a natural number representing the initial number of music files

-- Conditions: Initial state and changes
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23
def remaining_files : ℕ := 2

-- Statement of the theorem
theorem number_of_music_files (h : M + initial_video_files - files_deleted = remaining_files) : M = 4 :=
  by
  -- Proof goes here
  sorry

end number_of_music_files_l412_412903


namespace triangle_similarity_iff_angle_ABC_eq_60_l412_412269

variables {A B C D E M : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point M]
variables (ABC : Triangle A B C) (AEM : Triangle A B E) (MCA : Triangle M C A)
variables (M : Midpoint B C)
variables (h1 : RightAngle A)
variables (h2 : AB < AC)
variables (h3 : D = Inter AC (PerpendicularLine M BC))
variables (h4 : E = Inter (ParallelLine M AC) (PerpendicularLine B BD))

theorem triangle_similarity_iff_angle_ABC_eq_60 :
  (TriangleSimilar AEM MCA) ↔ (Angle B A C = 60 * Degree) :=
sorry

end triangle_similarity_iff_angle_ABC_eq_60_l412_412269


namespace no_prime_divisible_by_45_l412_412189

theorem no_prime_divisible_by_45 : ∀ (p : ℕ), Prime p → ¬ (45 ∣ p) :=
by {
  intros p h_prime h_div,
  have h_factors := Nat.factors_unique,
  sorry
}

end no_prime_divisible_by_45_l412_412189


namespace jason_picked_7_pears_l412_412645

def pears_picked_by_jason (total_pears mike_pears : ℕ) : ℕ :=
  total_pears - mike_pears

theorem jason_picked_7_pears :
  pears_picked_by_jason 15 8 = 7 :=
by
  -- Proof is required but we can insert sorry here to skip it for now
  sorry

end jason_picked_7_pears_l412_412645


namespace max_distance_between_two_points_l412_412386

noncomputable def dist_3d (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

def sphere1 := Sphere.mk (-5, -15, 10) 25
def sphere2 := Sphere.mk (20, 15, -25) 95

theorem max_distance_between_two_points :
  let d := dist_3d sphere1.center sphere2.center
  d = 5 * real.sqrt 110 → 
  ∃ A ∈ {p : ℝ × ℝ × ℝ | dist_3d p sphere1.center = sphere1.radius},
  ∃ B ∈ {p : ℝ × ℝ × ℝ | dist_3d p sphere2.center = sphere2.radius},
  dist_3d A B = 120 + 5 * real.sqrt 110 := by
  sorry

end max_distance_between_two_points_l412_412386


namespace length_of_AB_l412_412658

noncomputable def ratio3to5 (AP PB : ℝ) : Prop := AP / PB = 3 / 5
noncomputable def ratio4to5 (AQ QB : ℝ) : Prop := AQ / QB = 4 / 5
noncomputable def pointDistances (P Q : ℝ) : Prop := P - Q = 3

theorem length_of_AB (A B P Q : ℝ) (P_on_AB : P > A ∧ P < B) (Q_on_AB : Q > A ∧ Q < B)
  (middle_side : P < (A + B) / 2 ∧ Q < (A + B) / 2)
  (h1 : ratio3to5 (P - A) (B - P))
  (h2 : ratio4to5 (Q - A) (B - Q))
  (h3 : pointDistances P Q) : B - A = 43.2 := 
sorry

end length_of_AB_l412_412658


namespace Q_neither_not_sufficient_nor_necessary_for_P_l412_412293

variable {α : Type*} [LinearOrderedField α]

def P (a1 b1 c1 a2 b2 c2 x : α) : Prop :=
  (a1 * x^2 + b1 * x + c1 > 0) = (a2 * x^2 + b2 * x + c2 > 0)

def Q (a1 b1 c1 a2 b2 c2 : α) : Prop :=
  a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

theorem Q_neither_not_sufficient_nor_necessary_for_P (a1 b1 c1 a2 b2 c2 : α) :
  ¬(Q a1 b1 c1 a2 b2 c2 → P a1 b1 c1 a2 b2 c2) ∧ ¬(P a1 b1 c1 a2 b2 c2 → Q a1 b1 c1 a2 b2 c2) :=
by
  sorry

end Q_neither_not_sufficient_nor_necessary_for_P_l412_412293


namespace sum_of_all_intersections_is_39_l412_412100

-- Five distinct lines in a plane, counted by their unique points of intersection
def sum_of_possible_intersections (lines: Fin 5 → Line) : ℕ :=
  let possible_values : List ℕ := [0, 1, 2, 3, 4, 5, 6, 8, 10]
  List.sum possible_values

-- The realization of lines given problem conditions
noncomputable def five_lines : Fin 5 → Line := sorry

theorem sum_of_all_intersections_is_39 :
  sum_of_possible_intersections five_lines = 39 :=
by sorry

end sum_of_all_intersections_is_39_l412_412100


namespace price_increase_60_percent_l412_412920

variable (P A B : ℝ)

-- Conditions as hypotheses
def condition1 : Prop := ∃ (k : ℝ), P = (1 + k) * A
def condition2 : Prop := P = 2 * B
def condition3 : Prop := P = 0.8888888888888889 * (A + B)

-- The theorem to prove
theorem price_increase_60_percent (h1 : condition1 P A B) (h2 : condition2 P B) (h3 : condition3 P A B) : 
  (P - A) / A = 0.6 :=
sorry

end price_increase_60_percent_l412_412920


namespace jerry_continues_indefinitely_l412_412860

def cell := ℤ × ℤ -- assuming cells can be represented as pairs of integers

inductive color
| black
| white

structure game_state :=
(inf_field : cell → color)
(initial_black_cells : fin 4 → cell) -- exactly 4 initial black cells
(turn : ℕ)

-- condition for Tom winning
def tom_wins (g : game_state) : Prop :=
  ¬ ∃ (c1 c2 : cell), c1 ≠ c2 ∧ g.inf_field c1 = color.black ∧ g.inf_field c2 = color.black ∧ adjacent c1 c2

-- "adjacent" indicates adjacency on a hexagonal grid
def adjacent (c1 c2 : cell) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 = c2.2 - 1)) ∨
  (c1.1 = c2.1 + 1 ∧ (c1.2 = c2.2 - 1 ∨ c1.2 = c2.2)) ∨
  (c1.1 = c2.1 - 1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 = c2.2))

-- statement that proves Jerry can continue the game indefinitely
theorem jerry_continues_indefinitely (g : game_state) : ¬ tom_wins g :=
sorry

end jerry_continues_indefinitely_l412_412860
