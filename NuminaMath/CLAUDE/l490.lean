import Mathlib

namespace textbook_order_cost_l490_49079

/-- Calculates the total cost of a textbook order with discounts applied --/
def calculate_order_cost (quantities : List Nat) (prices : List Float) (discount_threshold : Nat) (discount_rate : Float) : Float :=
  let total_cost := List.sum (List.zipWith (λ q p => q.toFloat * p) quantities prices)
  let discounted_cost := List.sum (List.zipWith 
    (λ q p => 
      if q ≥ discount_threshold then
        q.toFloat * p * (1 - discount_rate)
      else
        q.toFloat * p
    ) quantities prices)
  discounted_cost

theorem textbook_order_cost : 
  let quantities := [35, 35, 20, 30, 25, 15]
  let prices := [7.50, 10.50, 12.00, 9.50, 11.25, 6.75]
  let discount_threshold := 30
  let discount_rate := 0.1
  calculate_order_cost quantities prices discount_threshold discount_rate = 1446.00 := by
  sorry

end textbook_order_cost_l490_49079


namespace seating_theorem_l490_49001

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (abc : ℕ) (de : ℕ) : ℕ :=
  factorial n - (factorial (n - 2) * factorial 3) - 
  (factorial (n - 1) * factorial 2) + 
  (factorial (n - 3) * factorial 3 * factorial 2)

theorem seating_theorem : 
  seating_arrangements 10 3 2 = 2853600 := by sorry

end seating_theorem_l490_49001


namespace ellipse_b_value_l490_49068

/-- Definition of an ellipse with foci and a point on it -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- The vectors PF1 and PF2 are perpendicular -/
def vectors_perpendicular (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

/-- The area of triangle PF1F2 is 9 -/
def triangle_area_is_9 (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  abs (PF1.1 * PF2.2 - PF1.2 * PF2.1) / 2 = 9

/-- Main theorem -/
theorem ellipse_b_value (e : Ellipse) 
  (h_perp : vectors_perpendicular e) 
  (h_area : triangle_area_is_9 e) : 
  e.b = 3 := by
  sorry

end ellipse_b_value_l490_49068


namespace no_combination_for_3_4_meters_l490_49090

theorem no_combination_for_3_4_meters :
  ¬ ∃ (a b : ℕ), 0.7 * (a : ℝ) + 0.8 * (b : ℝ) = 3.4 := by
  sorry

end no_combination_for_3_4_meters_l490_49090


namespace martha_butterflies_l490_49098

/-- The number of black butterflies in Martha's collection --/
def black_butterflies (total blue yellow : ℕ) : ℕ :=
  total - blue - yellow

/-- Theorem stating the number of black butterflies in Martha's collection --/
theorem martha_butterflies :
  ∀ (total blue yellow : ℕ),
    total = 11 →
    blue = 4 →
    blue = 2 * yellow →
    black_butterflies total blue yellow = 5 := by
  sorry

end martha_butterflies_l490_49098


namespace rectangle_side_lengths_l490_49023

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℕ
  b : ℕ

/-- Represents a square with side length s -/
structure Square where
  s : ℕ

theorem rectangle_side_lengths 
  (rect : Rectangle)
  (sq1 sq2 : Square)
  (h1 : sq1.s = sq2.s)
  (h2 : rect.a + rect.b = 19)
  (h3 : 2 * (rect.a + rect.b) + 2 * sq1.s = 48)
  (h4 : 2 * (rect.a + rect.b) + 4 * sq1.s = 58)
  (h5 : rect.a > rect.b)
  (h6 : rect.a ≤ 13) :
  rect.a = 12 ∧ rect.b = 7 := by
sorry

end rectangle_side_lengths_l490_49023


namespace monochromatic_4cycle_exists_l490_49095

/-- A color for an edge -/
inductive Color
| Red
| Blue

/-- A graph with 6 vertices -/
def Graph6 := Fin 6 → Fin 6 → Color

/-- A 4-cycle in a graph -/
def IsCycle4 (g : Graph6) (v1 v2 v3 v4 : Fin 6) (c : Color) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v4 ∧ v4 ≠ v1 ∧
  g v1 v2 = c ∧ g v2 v3 = c ∧ g v3 v4 = c ∧ g v4 v1 = c

/-- The main theorem: every 6-vertex complete graph with red/blue edges contains a monochromatic 4-cycle -/
theorem monochromatic_4cycle_exists (g : Graph6) 
  (complete : ∀ u v : Fin 6, u ≠ v → (g u v = Color.Red ∨ g u v = Color.Blue)) :
  ∃ (v1 v2 v3 v4 : Fin 6) (c : Color), IsCycle4 g v1 v2 v3 v4 c :=
sorry

end monochromatic_4cycle_exists_l490_49095


namespace sampling_probabilities_equal_l490_49080

/-- The total number of parts -/
def total_parts : ℕ := 160

/-- The number of first-class products -/
def first_class : ℕ := 48

/-- The number of second-class products -/
def second_class : ℕ := 64

/-- The number of third-class products -/
def third_class : ℕ := 32

/-- The number of substandard products -/
def substandard : ℕ := 16

/-- The sample size -/
def sample_size : ℕ := 20

/-- The probability of selection in simple random sampling -/
def p₁ : ℚ := sample_size / total_parts

/-- The probability of selection in stratified sampling -/
def p₂ : ℚ := sample_size / total_parts

/-- The probability of selection in systematic sampling -/
def p₃ : ℚ := sample_size / total_parts

theorem sampling_probabilities_equal :
  p₁ = p₂ ∧ p₂ = p₃ ∧ p₁ = 1/8 :=
sorry

end sampling_probabilities_equal_l490_49080


namespace value_of_expression_l490_49020

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end value_of_expression_l490_49020


namespace solution_concentration_change_l490_49052

/-- Given an initial solution concentration, a replacement solution concentration,
    and the fraction of the solution replaced, calculate the new concentration. -/
def new_concentration (initial_conc replacement_conc fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced)) + (replacement_conc * fraction_replaced)

/-- Theorem stating that replacing 0.7142857142857143 of a 60% solution with a 25% solution
    results in a new concentration of 0.21285714285714285 -/
theorem solution_concentration_change : 
  new_concentration 0.60 0.25 0.7142857142857143 = 0.21285714285714285 := by sorry

end solution_concentration_change_l490_49052


namespace product_first_two_terms_of_specific_sequence_l490_49089

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem product_first_two_terms_of_specific_sequence :
  ∃ (a₁ : ℝ),
    arithmetic_sequence a₁ 1 5 = 11 ∧
    arithmetic_sequence a₁ 1 1 * arithmetic_sequence a₁ 1 2 = 56 :=
by sorry

end product_first_two_terms_of_specific_sequence_l490_49089


namespace triangle_midline_lengths_l490_49042

/-- Given a triangle with side lengths a, b, and c, the lengths of its midlines are half the lengths of the opposite sides. -/
theorem triangle_midline_lengths (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (s_a s_b s_c : ℝ),
    s_a = (1/2) * b ∧
    s_b = (1/2) * c ∧
    s_c = (1/2) * a :=
by sorry

end triangle_midline_lengths_l490_49042


namespace tyson_total_score_l490_49013

/-- Calculates the total points scored in a basketball game given the number of three-pointers, two-pointers, and one-pointers made. -/
def total_points (three_pointers two_pointers one_pointers : ℕ) : ℕ :=
  3 * three_pointers + 2 * two_pointers + one_pointers

/-- Theorem stating that given Tyson's scoring record, he scored a total of 75 points. -/
theorem tyson_total_score :
  total_points 15 12 6 = 75 := by
  sorry

#eval total_points 15 12 6

end tyson_total_score_l490_49013


namespace closest_point_l490_49045

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2 + 5*t
  | 1 => -3 + 7*t
  | 2 => -3 - 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => 4
  | 2 => 5

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 7
  | 2 => -2

theorem closest_point :
  let t := 43 / 78
  (v t - a) • direction = 0 ∧
  ∀ s, s ≠ t → ‖v s - a‖ > ‖v t - a‖ :=
sorry

end closest_point_l490_49045


namespace sector_area_l490_49069

/-- The area of a sector with radius 6 and central angle 60° is 6π. -/
theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 60 * π / 180) :
  (θ / (2 * π)) * π * r^2 = 6 * π := by
  sorry

end sector_area_l490_49069


namespace tan_sum_15_30_l490_49028

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_sum_15_30 :
  tan (15 * π / 180) + tan (30 * π / 180) + tan (15 * π / 180) * tan (30 * π / 180) = 1 :=
by
  -- Assume the trigonometric identity for the tangent of the sum of two angles
  have tan_sum_identity : ∀ A B : ℝ, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B) := sorry
  -- Assume that tan 45° = 1
  have tan_45 : tan (45 * π / 180) = 1 := sorry
  sorry -- The proof goes here

end tan_sum_15_30_l490_49028


namespace sum_of_specific_coefficients_l490_49087

/-- The coefficient of x^m * y^n in the expansion of (1+x)^4 * (1+y)^6 -/
def P (m n : ℕ) : ℕ := Nat.choose 4 m * Nat.choose 6 n

/-- The sum of coefficients of x^2*y^1 and x^1*y^2 in the expansion of (1+x)^4 * (1+y)^6 is 96 -/
theorem sum_of_specific_coefficients : P 2 1 + P 1 2 = 96 := by
  sorry

end sum_of_specific_coefficients_l490_49087


namespace range_of_m_l490_49048

-- Define the function f(x)
def f (x b c : ℝ) : ℝ := -2 * x^2 + b * x + c

-- State the theorem
theorem range_of_m (b c : ℝ) :
  (∀ x, f x b c > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, -1 ≤ x ∧ x ≤ 0 → ∃ m, f x b c + m ≥ 4) →
  ∃ m₀, ∀ m, m ≥ m₀ ↔ (∀ x, -1 ≤ x ∧ x ≤ 0 → f x b c + m ≥ 4) :=
sorry

end range_of_m_l490_49048


namespace simplify_expression_l490_49072

theorem simplify_expression (x : ℚ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
  sorry

end simplify_expression_l490_49072


namespace passenger_difference_l490_49024

structure BusRoute where
  initial_passengers : ℕ
  first_passengers : ℕ
  final_passengers : ℕ
  terminal_passengers : ℕ

def BusRoute.valid (route : BusRoute) : Prop :=
  route.initial_passengers = 30 ∧
  route.terminal_passengers = 14 ∧
  route.first_passengers * 3 = route.final_passengers

theorem passenger_difference (route : BusRoute) (h : route.valid) :
  ∃ y : ℕ, route.first_passengers + y = route.initial_passengers + 6 :=
by sorry

end passenger_difference_l490_49024


namespace hyperbola_sum_l490_49084

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = -3) →
  (k = 1) →
  (c = Real.sqrt 41) →
  (a = 4) →
  (c^2 = a^2 + b^2) →
  (h + k + a + b = 7) := by
  sorry

end hyperbola_sum_l490_49084


namespace triangle_side_sum_range_l490_49077

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = √3 and (2c-a)/b * cos(B) = cos(A), prove that a+c is in the range (√3, 2√3]. -/
theorem triangle_side_sum_range (a b c A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b = Real.sqrt 3 →
  (2*c - a)/b * Real.cos B = Real.cos A →
  ∃ (x : ℝ), Real.sqrt 3 < x ∧ x ≤ 2 * Real.sqrt 3 ∧ a + c = x :=
by sorry

end triangle_side_sum_range_l490_49077


namespace equation_roots_l490_49014

theorem equation_roots : ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 ↔ x = 2 ∨ x = 4 := by
  sorry

end equation_roots_l490_49014


namespace min_value_f_min_value_expression_l490_49070

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f x ≥ m :=
sorry

-- Theorem for the minimum value of 1/(a²+1) + 4/(b²+1)
theorem min_value_expression (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ min_val : ℝ, min_val = 9/4 ∧
  1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ min_val :=
sorry

end min_value_f_min_value_expression_l490_49070


namespace hamburger_buns_cost_l490_49025

/-- The cost of hamburger buns given Lauren's grocery purchase --/
theorem hamburger_buns_cost : 
  ∀ (meat_price meat_weight lettuce_price tomato_price tomato_weight
     pickle_price pickle_discount paid change bun_price : ℝ),
  meat_price = 3.5 →
  meat_weight = 2 →
  lettuce_price = 1 →
  tomato_price = 2 →
  tomato_weight = 1.5 →
  pickle_price = 2.5 →
  pickle_discount = 1 →
  paid = 20 →
  change = 6 →
  bun_price = paid - change - (meat_price * meat_weight + lettuce_price + 
    tomato_price * tomato_weight + pickle_price - pickle_discount) →
  bun_price = 1.5 := by
sorry

end hamburger_buns_cost_l490_49025


namespace facial_tissue_price_decrease_l490_49073

/-- The percent decrease in price per box of facial tissue during a sale -/
theorem facial_tissue_price_decrease (original_price sale_price : ℚ) : 
  original_price = 5 / 4 →
  sale_price = 4 / 5 →
  abs ((original_price - sale_price) / original_price - 9 / 25) < 1 / 100 := by
  sorry

#eval (5/4 : ℚ) -- Original price per box
#eval (4/5 : ℚ) -- Sale price per box
#eval ((5/4 - 4/5) / (5/4) : ℚ) -- Actual percent decrease

end facial_tissue_price_decrease_l490_49073


namespace hexagon_perimeter_is_24_l490_49078

/-- A hexagon with specific properties -/
structure Hexagon :=
  (AB EF BE AF CD DF : ℝ)
  (ab_ef_eq : AB = EF)
  (be_af_eq : BE = AF)
  (ab_length : AB = 3)
  (be_length : BE = 4)
  (cd_length : CD = 5)
  (df_length : DF = 5)

/-- The perimeter of the hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BE + h.CD + h.DF + h.EF + h.AF

/-- Theorem stating that the perimeter of the hexagon is 24 units -/
theorem hexagon_perimeter_is_24 (h : Hexagon) : perimeter h = 24 := by
  sorry


end hexagon_perimeter_is_24_l490_49078


namespace line_equation_a_value_l490_49027

/-- Given a line passing through points (4,3) and (12,-3), if its equation is in the form (x/a) + (y/b) = 1, then a = 8 -/
theorem line_equation_a_value (a b : ℝ) : 
  (∀ x y : ℝ, (x / a + y / b = 1) ↔ (3 * x + 4 * y = 24)) →
  ((4 : ℝ) / a + (3 : ℝ) / b = 1) →
  ((12 : ℝ) / a + (-3 : ℝ) / b = 1) →
  a = 8 := by
sorry

end line_equation_a_value_l490_49027


namespace rachel_steps_up_correct_l490_49097

/-- The number of steps Rachel climbed going up the Eiffel Tower -/
def steps_up : ℕ := 567

/-- The number of steps Rachel climbed going down the Eiffel Tower -/
def steps_down : ℕ := 325

/-- The total number of steps Rachel climbed -/
def total_steps : ℕ := 892

/-- Theorem: The number of steps Rachel climbed going up is correct -/
theorem rachel_steps_up_correct : steps_up + steps_down = total_steps := by
  sorry

end rachel_steps_up_correct_l490_49097


namespace dish_washing_time_l490_49039

-- Define the given constants
def sweep_time_per_room : ℕ := 3
def laundry_time_per_load : ℕ := 9
def anna_rooms_swept : ℕ := 10
def billy_laundry_loads : ℕ := 2
def billy_dishes_to_wash : ℕ := 6

-- Define the theorem
theorem dish_washing_time :
  ∃ (dish_time : ℕ),
    dish_time = 2 ∧
    sweep_time_per_room * anna_rooms_swept =
    laundry_time_per_load * billy_laundry_loads + billy_dishes_to_wash * dish_time :=
by sorry

end dish_washing_time_l490_49039


namespace max_selection_ways_l490_49002

/-- The total number of socks -/
def total_socks : ℕ := 2017

/-- The function to calculate the number of ways to select socks -/
def selection_ways (partition : List ℕ) : ℕ :=
  partition.prod

/-- The theorem stating the maximum number of ways to select socks -/
theorem max_selection_ways :
  ∃ (partition : List ℕ),
    partition.sum = total_socks ∧
    ∀ (other_partition : List ℕ),
      other_partition.sum = total_socks →
      selection_ways other_partition ≤ selection_ways partition ∧
      selection_ways partition = 3^671 * 4 :=
sorry

end max_selection_ways_l490_49002


namespace passenger_arrangement_l490_49064

def arrange_passengers (n : ℕ) (r : ℕ) : ℕ :=
  -- Define the function to calculate the number of arrangements
  sorry

theorem passenger_arrangement :
  arrange_passengers 5 3 = 150 := by
  sorry

end passenger_arrangement_l490_49064


namespace problem_solution_l490_49076

theorem problem_solution (a b : ℝ) : 
  |a + 1| + (b - 2)^2 = 0 → (a + b)^9 + a^6 = 2 := by
  sorry

end problem_solution_l490_49076


namespace unique_zero_in_interval_l490_49012

theorem unique_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ (Set.Ioo 0 2) ∧ x^2 - a*x + 1 = 0 :=
by sorry

end unique_zero_in_interval_l490_49012


namespace doctors_visit_insurance_coverage_percentage_l490_49066

def doctors_visit_cost : ℝ := 300
def cats_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60
def total_paid_after_insurance : ℝ := 135

theorem doctors_visit_insurance_coverage_percentage :
  let total_cost := doctors_visit_cost + cats_visit_cost
  let total_insurance_coverage := total_cost - total_paid_after_insurance
  let doctors_visit_coverage := total_insurance_coverage - pet_insurance_coverage
  doctors_visit_coverage / doctors_visit_cost = 0.75 := by sorry

end doctors_visit_insurance_coverage_percentage_l490_49066


namespace largest_common_divisor_of_S_l490_49081

def S : Set ℕ := {n : ℕ | ∃ (d₁ d₂ d₃ : ℕ), d₁ > d₂ ∧ d₂ > d₃ ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₁ ≠ n ∧ d₂ ≠ n ∧ d₃ ≠ n ∧ d₁ + d₂ + d₃ > n}

theorem largest_common_divisor_of_S : ∀ n ∈ S, 6 ∣ n ∧ ∀ k : ℕ, (∀ m ∈ S, k ∣ m) → k ≤ 6 :=
sorry

end largest_common_divisor_of_S_l490_49081


namespace fibonacci_rabbit_problem_l490_49043

/-- Fibonacci sequence representing the number of adult rabbit pairs -/
def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

/-- The number of adult rabbit pairs after n months -/
def adult_rabbits (n : ℕ) : ℕ := fibonacci n

theorem fibonacci_rabbit_problem :
  adult_rabbits 12 = 233 := by sorry

end fibonacci_rabbit_problem_l490_49043


namespace maggi_cupcakes_l490_49016

/-- Proves that Maggi ate 0 cupcakes given the initial number of packages,
    cupcakes per package, and cupcakes left. -/
theorem maggi_cupcakes (initial_packages : ℕ) (cupcakes_per_package : ℕ) (cupcakes_left : ℕ)
    (h1 : initial_packages = 3)
    (h2 : cupcakes_per_package = 4)
    (h3 : cupcakes_left = 12) :
    initial_packages * cupcakes_per_package - cupcakes_left = 0 := by
  sorry

end maggi_cupcakes_l490_49016


namespace sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l490_49019

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let a : ℝ := 3
  let b : ℝ := -24
  let c : ℝ := 98
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = 8 :=
by sorry

end sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l490_49019


namespace sinusoidal_function_translation_l490_49018

/-- Given a function f(x) = sin(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - Smallest positive period is π
    - Graph is translated left by π/6 units
    - Resulting function is odd
    Then f(x) = sin(2x - π/3) -/
theorem sinusoidal_function_translation (f : ℝ → ℝ) (ω φ : ℝ) 
    (h_omega : ω > 0)
    (h_phi : |φ| < π/2)
    (h_period : ∀ x, f (x + π) = f x)
    (h_smallest_period : ∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π)
    (h_translation : ∀ x, f (x + π/6) = -f (-x + π/6)) :
  ∀ x, f x = Real.sin (2*x - π/3) := by
  sorry

end sinusoidal_function_translation_l490_49018


namespace power_of_three_equation_l490_49088

theorem power_of_three_equation (x : ℝ) : 
  4 * (3 : ℝ)^x = 243 → (x + 1) * (x - 1) = 16.696 := by
  sorry

end power_of_three_equation_l490_49088


namespace remainder_theorem_l490_49059

theorem remainder_theorem : (439 * 319 * 2012 + 2013) % 7 = 1 := by
  sorry

end remainder_theorem_l490_49059


namespace half_angle_quadrant_l490_49000

-- Define a function to check if an angle is in the first quadrant
def is_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define a function to check if an angle is in the first or third quadrant
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, (n * 360 < α ∧ α < n * 360 + 45) ∨ 
            (n * 360 + 180 < α ∧ α < n * 360 + 225)

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end half_angle_quadrant_l490_49000


namespace base_eight_subtraction_l490_49041

/-- Represents a number in base 8 -/
def BaseEight : Type := ℕ

/-- Converts a base 8 number to its decimal representation -/
def to_decimal (n : BaseEight) : ℕ := sorry

/-- Converts a decimal number to its base 8 representation -/
def to_base_eight (n : ℕ) : BaseEight := sorry

/-- Subtracts two base 8 numbers -/
def base_eight_sub (a b : BaseEight) : BaseEight := sorry

/-- Theorem stating that 46₈ - 27₈ = 17₈ in base 8 -/
theorem base_eight_subtraction :
  base_eight_sub (to_base_eight 38) (to_base_eight 23) = to_base_eight 15 := by sorry

end base_eight_subtraction_l490_49041


namespace apple_running_rate_l490_49093

/-- Given Mac's and Apple's running rates, prove that Apple's rate is 3 miles per hour -/
theorem apple_running_rate (mac_rate apple_rate : ℝ) : 
  mac_rate = 4 →  -- Mac's running rate is 4 miles per hour
  (24 / mac_rate) * 60 + 120 = (24 / apple_rate) * 60 →  -- Mac runs 24 miles 120 minutes faster than Apple
  apple_rate = 3 :=  -- Apple's running rate is 3 miles per hour
by
  sorry


end apple_running_rate_l490_49093


namespace triumphal_arch_proportion_l490_49017

/-- Represents the number of photographs of each type of attraction -/
structure Photos where
  cathedrals : ℕ
  arches : ℕ
  waterfalls : ℕ
  castles : ℕ

/-- Represents the total number of each type of attraction seen -/
structure Attractions where
  cathedrals : ℕ
  arches : ℕ
  waterfalls : ℕ
  castles : ℕ

/-- The main theorem stating the proportion of photographs featuring triumphal arches -/
theorem triumphal_arch_proportion
  (p : Photos) (a : Attractions)
  (half_photographed : p.cathedrals + p.arches + p.waterfalls + p.castles = (a.cathedrals + a.arches + a.waterfalls + a.castles) / 2)
  (cathedral_arch_ratio : a.cathedrals = 3 * a.arches)
  (castle_waterfall_equal : a.castles = a.waterfalls)
  (quarter_castles : 4 * p.castles = p.cathedrals + p.arches + p.waterfalls + p.castles)
  (half_castles_photographed : 2 * p.castles = a.castles)
  (all_arches_photographed : p.arches = a.arches) :
  4 * p.arches = p.cathedrals + p.arches + p.waterfalls + p.castles :=
by sorry

end triumphal_arch_proportion_l490_49017


namespace acute_angle_tangent_implies_a_equals_one_l490_49007

/-- The curve C: y = x^3 - 2ax^2 + 2ax -/
def C (a : ℤ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℤ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

theorem acute_angle_tangent_implies_a_equals_one (a : ℤ) :
  (∀ x : ℝ, C_derivative a x > 0) →
  a = 1 := by
  sorry

end acute_angle_tangent_implies_a_equals_one_l490_49007


namespace sector_area_l490_49092

/-- Given a circular sector with perimeter 6 cm and central angle 1 radian, its area is 3 cm² -/
theorem sector_area (r : ℝ) (h1 : r + r + r = 6) (h2 : 1 = 1) : r * r / 2 = 3 := by
  sorry

end sector_area_l490_49092


namespace jimmy_passing_points_l490_49099

/-- The minimum number of points required to pass to the next class -/
def min_points_to_pass : ℕ := 50

/-- The number of points earned per exam -/
def points_per_exam : ℕ := 20

/-- The number of exams taken -/
def num_exams : ℕ := 3

/-- The number of points lost for bad behavior -/
def points_lost_behavior : ℕ := 5

/-- The maximum number of additional points Jimmy can lose and still pass -/
def max_additional_points_to_lose : ℕ := 5

theorem jimmy_passing_points :
  max_additional_points_to_lose = 
    points_per_exam * num_exams - points_lost_behavior - min_points_to_pass := by
  sorry

end jimmy_passing_points_l490_49099


namespace rectangular_to_polar_equivalence_l490_49046

/-- Proves the equivalence of rectangular and polar coordinate equations --/
theorem rectangular_to_polar_equivalence 
  (x y ρ θ : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) : 
  y^2 = 12*x ↔ ρ * Real.sin θ^2 = 12 * Real.cos θ :=
by sorry

end rectangular_to_polar_equivalence_l490_49046


namespace ellipse_point_inside_circle_l490_49026

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (h_ab : a > b) 
  (h_b_pos : b > 0) 
  (h_e : c / a = 1 / 2) 
  (x₁ x₂ : ℝ) 
  (h_roots : x₁ * x₂ = -c / a ∧ x₁ + x₂ = -b / a) : 
  x₁^2 + x₂^2 < 2 := by
sorry

end ellipse_point_inside_circle_l490_49026


namespace octagon_handshake_distance_l490_49057

theorem octagon_handshake_distance (n : ℕ) (r : ℝ) (h1 : n = 8) (h2 : r = 50) :
  let points := n
  let radius := r
  let connections_per_point := n - 3
  let angle_between_points := 2 * Real.pi / n
  let distance_to_third := radius * Real.sqrt (2 - Real.sqrt 2)
  let total_distance := n * connections_per_point * distance_to_third
  total_distance = 1600 * Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end octagon_handshake_distance_l490_49057


namespace equilateral_condition_isosceles_condition_l490_49050

-- Define a triangle ABC with side lengths a, b, c
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

-- Define equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem 1
theorem equilateral_condition (t : Triangle) :
  abs (t.a - t.b) + abs (t.b - t.c) = 0 → is_equilateral t :=
by sorry

-- Theorem 2
theorem isosceles_condition (t : Triangle) :
  (t.a - t.b) * (t.b - t.c) = 0 → is_isosceles t :=
by sorry

end equilateral_condition_isosceles_condition_l490_49050


namespace rest_of_body_length_l490_49094

theorem rest_of_body_length (total_height : ℝ) (leg_ratio : ℝ) (head_ratio : ℝ) 
  (h1 : total_height = 60)
  (h2 : leg_ratio = 1/3)
  (h3 : head_ratio = 1/4) :
  total_height - (leg_ratio * total_height) - (head_ratio * total_height) = 25 := by
  sorry

end rest_of_body_length_l490_49094


namespace least_integer_absolute_value_l490_49021

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, |2*y + 9| ≤ 20 → y ≥ -14) ∧ |2*(-14) + 9| ≤ 20 := by
  sorry

end least_integer_absolute_value_l490_49021


namespace car_average_speed_l490_49038

/-- Proves that the average speed of a car traveling 90 km/h for the first hour
    and 60 km/h for the second hour is 75 km/h. -/
theorem car_average_speed :
  let speed1 : ℝ := 90  -- Speed in the first hour (km/h)
  let speed2 : ℝ := 60  -- Speed in the second hour (km/h)
  let time : ℝ := 2     -- Total time (hours)
  let total_distance : ℝ := speed1 + speed2  -- Total distance traveled (km)
  let average_speed : ℝ := total_distance / time  -- Average speed (km/h)
  average_speed = 75
  := by sorry

end car_average_speed_l490_49038


namespace pizza_slices_ordered_l490_49063

/-- The number of friends Ron ate pizza with -/
def num_friends : ℕ := 2

/-- The number of slices each person ate -/
def slices_per_person : ℕ := 4

/-- The total number of people eating pizza (Ron + his friends) -/
def total_people : ℕ := num_friends + 1

/-- Theorem: The total number of pizza slices ordered is at least 12 -/
theorem pizza_slices_ordered (num_friends : ℕ) (slices_per_person : ℕ) (total_people : ℕ) :
  num_friends = 2 →
  slices_per_person = 4 →
  total_people = num_friends + 1 →
  total_people * slices_per_person ≥ 12 := by
  sorry

end pizza_slices_ordered_l490_49063


namespace minimum_correct_answers_l490_49049

def test_score (correct : ℕ) : ℤ :=
  4 * correct - (25 - correct)

theorem minimum_correct_answers : 
  ∀ x : ℕ, x ≤ 25 → test_score x > 70 → x ≥ 19 :=
by
  sorry

end minimum_correct_answers_l490_49049


namespace diameters_sum_equals_legs_sum_l490_49040

/-- A right-angled triangle with its circumscribed and inscribed circles -/
structure RightTriangle where
  /-- First leg of the right triangle -/
  a : ℝ
  /-- Second leg of the right triangle -/
  b : ℝ
  /-- Hypotenuse of the right triangle -/
  c : ℝ
  /-- Radius of the circumscribed circle -/
  R : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Condition: a and b are positive -/
  ha : 0 < a
  /-- Condition: a and b are positive -/
  hb : 0 < b
  /-- Pythagorean theorem -/
  pythagorean : a^2 + b^2 = c^2
  /-- Relation between hypotenuse and circumscribed circle diameter -/
  circum_diam : c = 2 * R
  /-- Relation for inscribed circle radius in a right triangle -/
  inscribed_radius : r = (a + b - c) / 2

/-- The sum of the diameters of the circumscribed and inscribed circles 
    is equal to the sum of the legs in a right-angled triangle -/
theorem diameters_sum_equals_legs_sum (t : RightTriangle) : 2 * t.R + 2 * t.r = t.a + t.b := by
  sorry

end diameters_sum_equals_legs_sum_l490_49040


namespace competition_score_l490_49060

theorem competition_score (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) 
  (total_score : ℤ) (score_difference : ℤ) :
  total_questions = 10 →
  correct_points = 5 →
  incorrect_points = -2 →
  total_score = 58 →
  score_difference = 14 →
  ∃ (a_correct : ℕ) (b_correct : ℕ),
    a_correct + b_correct ≤ total_questions ∧
    a_correct * correct_points + (total_questions - a_correct) * incorrect_points +
    b_correct * correct_points + (total_questions - b_correct) * incorrect_points = total_score ∧
    a_correct * correct_points + (total_questions - a_correct) * incorrect_points -
    (b_correct * correct_points + (total_questions - b_correct) * incorrect_points) = score_difference ∧
    a_correct = 8 :=
by sorry

end competition_score_l490_49060


namespace triangle_angle_x_l490_49005

theorem triangle_angle_x (x : ℝ) : 
  x > 0 ∧ 3*x > 0 ∧ 40 > 0 ∧ 
  x + 3*x + 40 = 180 → 
  x = 35 := by
sorry

end triangle_angle_x_l490_49005


namespace sugar_packet_weight_l490_49061

-- Define the number of packets sold per week
def packets_per_week : ℕ := 20

-- Define the total weight of sugar sold per week in kilograms
def total_weight_kg : ℕ := 2

-- Define the conversion factor from kilograms to grams
def kg_to_g : ℕ := 1000

-- Theorem stating that each packet weighs 100 grams
theorem sugar_packet_weight :
  (total_weight_kg * kg_to_g) / packets_per_week = 100 := by
sorry

end sugar_packet_weight_l490_49061


namespace exists_sum_of_digits_div_by_11_l490_49053

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that in any 39 consecutive natural numbers, 
    there is at least one whose sum of digits is divisible by 11 -/
theorem exists_sum_of_digits_div_by_11 (n : ℕ) : 
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 38 ∧ (sum_of_digits k) % 11 = 0 := by
  sorry

end exists_sum_of_digits_div_by_11_l490_49053


namespace sasha_questions_per_hour_l490_49083

theorem sasha_questions_per_hour 
  (initial_questions : ℕ)
  (work_hours : ℕ)
  (remaining_questions : ℕ)
  (h1 : initial_questions = 60)
  (h2 : work_hours = 2)
  (h3 : remaining_questions = 30) :
  (initial_questions - remaining_questions) / work_hours = 15 := by
sorry

end sasha_questions_per_hour_l490_49083


namespace skateboard_distance_l490_49034

/-- Sequence representing the distance covered by the skateboard in each second -/
def skateboardSequence (n : ℕ) : ℕ := 8 + 10 * (n - 1)

/-- The total distance traveled by the skateboard in 20 seconds -/
def totalDistance : ℕ := (Finset.range 20).sum skateboardSequence

/-- Theorem stating that the total distance traveled is 2060 inches -/
theorem skateboard_distance : totalDistance = 2060 := by
  sorry

end skateboard_distance_l490_49034


namespace first_consignment_cost_price_l490_49004

/-- Represents a consignment of cloth -/
structure Consignment where
  length : ℕ
  profit_per_meter : ℚ

/-- Calculates the cost price per meter for a given consignment -/
def cost_price_per_meter (c : Consignment) (selling_price : ℚ) : ℚ :=
  (selling_price - c.profit_per_meter * c.length) / c.length

theorem first_consignment_cost_price 
  (c1 : Consignment)
  (c2 : Consignment)
  (c3 : Consignment)
  (selling_price : ℚ) :
  c1.length = 92 ∧ 
  c1.profit_per_meter = 24 ∧
  c2.length = 120 ∧
  c2.profit_per_meter = 30 ∧
  c3.length = 75 ∧
  c3.profit_per_meter = 20 ∧
  selling_price = 9890 →
  cost_price_per_meter c1 selling_price = 83.50 := by
    sorry

end first_consignment_cost_price_l490_49004


namespace twelve_chairs_subsets_l490_49029

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets with at least four adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs in a circle, there are 1704 subsets with at least four adjacent chairs -/
theorem twelve_chairs_subsets : subsets_with_adjacent_chairs n = 1704 := by sorry

end twelve_chairs_subsets_l490_49029


namespace paint_remaining_l490_49010

theorem paint_remaining (initial_paint : ℚ) : initial_paint = 2 →
  let day1_remaining := initial_paint / 2
  let day2_remaining := day1_remaining * 3 / 4
  let day3_remaining := day2_remaining * 2 / 3
  day3_remaining = initial_paint / 2 := by
  sorry

end paint_remaining_l490_49010


namespace intersection_empty_implies_t_bound_l490_49015

def M : Set (ℝ × ℝ) := {p | p.1^3 + 8*p.2^3 + 6*p.1*p.2 ≥ 1}

def D (t : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ t^2}

theorem intersection_empty_implies_t_bound 
  (t : ℝ) 
  (h : t ≠ 0) 
  (h_empty : D t ∩ M = ∅) : 
  -Real.sqrt 5 / 5 < t ∧ t < Real.sqrt 5 / 5 := by
  sorry

end intersection_empty_implies_t_bound_l490_49015


namespace parallel_condition_l490_49085

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation for a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition 
  (α β : Plane) (a : Line) 
  (h_subset : subset a α) : 
  (∀ α β a, plane_parallel α β → line_plane_parallel a β) ∧ 
  (∃ α β a, line_plane_parallel a β ∧ ¬plane_parallel α β) :=
sorry

end parallel_condition_l490_49085


namespace largest_digit_rounding_l490_49009

def number (d : ℕ) : ℕ := 5400000000 + d * 10000000 + 9607502

def rounds_to_5_5_billion (n : ℕ) : Prop :=
  5450000000 ≤ n ∧ n < 5550000000

theorem largest_digit_rounding :
  ∀ d : ℕ, d ≤ 9 →
    (rounds_to_5_5_billion (number d) ↔ 5 ≤ d) ∧
    (d = 9 ↔ ∀ k : ℕ, k ≤ 9 ∧ rounds_to_5_5_billion (number k) → k ≤ d) :=
by sorry

end largest_digit_rounding_l490_49009


namespace purum_elementary_students_l490_49058

theorem purum_elementary_students (total : ℕ) (difference : ℕ) : total = 41 → difference = 3 →
  ∃ purum non_purum : ℕ, purum = non_purum + difference ∧ purum + non_purum = total ∧ purum = 22 :=
by sorry

end purum_elementary_students_l490_49058


namespace bus_passengers_l490_49056

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (got_off : ℕ) : 
  initial = 28 → got_on = 7 → got_off = 9 → 
  initial + got_on - got_off = 26 := by
  sorry

end bus_passengers_l490_49056


namespace recurring_decimal_fraction_sum_l490_49031

theorem recurring_decimal_fraction_sum (a b : ℕ+) :
  (a.val : ℚ) / (b.val : ℚ) = 36 / 99 →
  Nat.gcd a.val b.val = 1 →
  a.val + b.val = 15 := by
  sorry

end recurring_decimal_fraction_sum_l490_49031


namespace constant_dot_product_l490_49074

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + 3*y^2/4 = 1

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a tangent line to circle O
def tangent_line (k m : ℝ) (x y : ℝ) : Prop := y = k*x + m ∧ 1 + k^2 = m^2

-- Define the intersection points of the tangent line and ellipse C
def intersection_points (k m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧ 
  tangent_line k m x₁ y₁ ∧ tangent_line k m x₂ y₂

-- Theorem statement
theorem constant_dot_product :
  ∀ (k m x₁ y₁ x₂ y₂ : ℝ),
  intersection_points k m x₁ y₁ x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end constant_dot_product_l490_49074


namespace erasers_in_box_l490_49008

/-- The number of erasers left in the box after a series of operations -/
def erasers_left (initial : ℕ) (taken : ℕ) (added : ℕ) : ℕ :=
  let remaining := initial - taken
  let half_taken := remaining / 2
  remaining - half_taken + added

/-- Theorem stating the number of erasers left in the box -/
theorem erasers_in_box : erasers_left 320 67 30 = 157 := by
  sorry

end erasers_in_box_l490_49008


namespace unique_solution_iff_sqrt_three_l490_49055

/-- The function f(x) = x^2 + a|x| + a^2 - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * |x| + a^2 - 3

/-- The theorem stating that the equation f(x) = 0 has a unique real solution iff a = √3 -/
theorem unique_solution_iff_sqrt_three (a : ℝ) :
  (∃! x : ℝ, f a x = 0) ↔ a = Real.sqrt 3 := by sorry

end unique_solution_iff_sqrt_three_l490_49055


namespace restaurant_bill_split_l490_49096

def bill : ℚ := 314.16
def payment_per_person : ℚ := 34.91
def total_payment : ℚ := 314.19

theorem restaurant_bill_split :
  ∃ (n : ℕ), n > 0 ∧ 
  (n : ℚ) * payment_per_person ≥ bill ∧
  (n : ℚ) * payment_per_person < bill + 1 ∧
  n * payment_per_person = total_payment ∧
  n = 8 := by sorry

end restaurant_bill_split_l490_49096


namespace units_digit_of_7_cubed_l490_49033

theorem units_digit_of_7_cubed (n : ℕ) : n = 7^3 → n % 10 = 3 := by
  sorry

end units_digit_of_7_cubed_l490_49033


namespace cricket_team_age_difference_l490_49054

theorem cricket_team_age_difference 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_avg_age : ℝ) 
  (h1 : team_size = 11) 
  (h2 : team_avg_age = 26) 
  (h3 : wicket_keeper_age_diff = 3) 
  (h4 : remaining_avg_age = 23) : 
  team_avg_age - ((team_size * team_avg_age - (team_avg_age + wicket_keeper_age_diff + team_avg_age)) / (team_size - 2)) = 0.33 := by
sorry

end cricket_team_age_difference_l490_49054


namespace range_of_a_l490_49044

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a ≥ 0) ∧
  (∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0) →
  a ≥ 0 :=
by sorry

end range_of_a_l490_49044


namespace triangle_altitude_and_median_l490_49035

/-- Triangle ABC with vertices A(3,-4), B(6,0), and C(-5,2) -/
structure Triangle where
  A : ℝ × ℝ := (3, -4)
  B : ℝ × ℝ := (6, 0)
  C : ℝ × ℝ := (-5, 2)

/-- Line equation in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of altitude BD and median BE -/
theorem triangle_altitude_and_median (t : Triangle) :
  ∃ (altitude_bd median_be : LineEquation),
    altitude_bd = ⟨4, -3, -24⟩ ∧
    median_be = ⟨1, -7, -6⟩ := by
  sorry

end triangle_altitude_and_median_l490_49035


namespace y_coordinate_is_1000_l490_49003

/-- A straight line in the xy-plane with given properties -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point on a line -/
structure Point where
  x : ℝ
  y : ℝ

/-- The y-coordinate of a point on a line can be calculated using the line's equation -/
def y_coordinate (l : Line) (p : Point) : ℝ :=
  l.slope * p.x + l.y_intercept

/-- Theorem: The y-coordinate of the specified point on the given line is 1000 -/
theorem y_coordinate_is_1000 (l : Line) (p : Point)
  (h1 : l.slope = 9.9)
  (h2 : l.y_intercept = 10)
  (h3 : p.x = 100) :
  y_coordinate l p = 1000 := by
  sorry

end y_coordinate_is_1000_l490_49003


namespace exactly_one_correct_statement_l490_49065

/-- Rules of the oblique projection drawing method -/
structure ObliqueProjectionRules where
  parallelism_preserved : Bool
  x_axis_length_preserved : Bool
  y_axis_length_halved : Bool

/-- Statements about intuitive diagrams -/
structure IntuitiveDiagramStatements where
  equal_angles_preserved : Bool
  equal_segments_preserved : Bool
  longest_segment_preserved : Bool
  midpoint_preserved : Bool

/-- Theorem: Exactly one statement is correct given the oblique projection rules -/
theorem exactly_one_correct_statement 
  (rules : ObliqueProjectionRules)
  (statements : IntuitiveDiagramStatements) :
  rules.parallelism_preserved ∧
  rules.x_axis_length_preserved ∧
  rules.y_axis_length_halved →
  (statements.equal_angles_preserved = false) ∧
  (statements.equal_segments_preserved = false) ∧
  (statements.longest_segment_preserved = false) ∧
  (statements.midpoint_preserved = true) :=
sorry

end exactly_one_correct_statement_l490_49065


namespace hyperbola_equation_prove_hyperbola_equation_l490_49051

/-- The standard equation of a hyperbola with given foci and passing through a specific point. -/
theorem hyperbola_equation (h : ℝ → ℝ → Prop) (f : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  -- Given hyperbola with equation x^2/16 - y^2/9 = 1
  (∀ x y, h x y ↔ x^2/16 - y^2/9 = 1) →
  -- The new hyperbola has the same foci as the given one
  (∃ c : ℝ, c^2 = 25 ∧ f = (c, 0) ∨ f = (-c, 0)) →
  -- The new hyperbola passes through the point P
  (p = (-Real.sqrt 5 / 2, -Real.sqrt 6)) →
  -- The standard equation of the new hyperbola is x^2/1 - y^2/24 = 1
  (∀ x y, (x^2/1 - y^2/24 = 1) ↔ 
    ((x - f.1)^2 + y^2)^(1/2) - ((x + f.1)^2 + y^2)^(1/2) = 2 * Real.sqrt (f.1^2 - 1))

/-- Proof of the hyperbola equation -/
theorem prove_hyperbola_equation : ∃ h f p, hyperbola_equation h f p := by
  sorry

end hyperbola_equation_prove_hyperbola_equation_l490_49051


namespace hyperbola_dimensions_l490_49067

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  foci_to_asymptote : ℝ
  asymptote_slope : ℝ
  foci_distance : foci_to_asymptote = 2
  asymptote_parallel : asymptote_slope = 1/2

/-- The theorem stating the specific dimensions of the hyperbola -/
theorem hyperbola_dimensions (h : Hyperbola) : h.a = 4 ∧ h.b = 2 := by
  sorry

end hyperbola_dimensions_l490_49067


namespace exists_special_function_l490_49062

theorem exists_special_function :
  ∃ f : ℕ+ → ℕ+,
    (∀ m n : ℕ+, m < n → f m < f n) ∧
    f 1 = 2 ∧
    ∀ n : ℕ+, f (f n) = f n + n :=
by sorry

end exists_special_function_l490_49062


namespace cylinder_volume_doubling_l490_49030

/-- Given a cylinder with original volume 10 cubic feet, prove that doubling its height
    while keeping the radius constant results in a new volume of 20 cubic feet. -/
theorem cylinder_volume_doubling (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 10 → π * r^2 * (2 * h) = 20 :=
by sorry

end cylinder_volume_doubling_l490_49030


namespace vector_subtraction_l490_49006

/-- Given two vectors AB and AC in R², prove that CB = AB - AC -/
theorem vector_subtraction (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, 3]) (h2 : AC = ![-1, 2]) :
  (fun i => AB i - AC i) = ![3, 1] := by
  sorry

end vector_subtraction_l490_49006


namespace sarah_pizza_consumption_l490_49082

theorem sarah_pizza_consumption (total_slices : ℕ) (eaten_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 20 →
  eaten_slices = 3 →
  shared_slice = 1/3 →
  (eaten_slices : ℚ) / total_slices + shared_slice / total_slices = 1/6 := by
  sorry

end sarah_pizza_consumption_l490_49082


namespace no_quadratic_polynomial_satisfies_conditions_l490_49075

theorem no_quadratic_polynomial_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ), 
    (∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)) ∧ 
    (∀ x, f (x^2) = x^4) ∧
    (∀ x, f (f x) = (x^2 + 1)^4) :=
by sorry

end no_quadratic_polynomial_satisfies_conditions_l490_49075


namespace february_first_is_sunday_l490_49091

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to check if a given day is Monday -/
def isMonday (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Monday => true
  | _ => false

/-- Theorem: In a leap year, if February has exactly four Mondays, then February 1st must be a Sunday -/
theorem february_first_is_sunday (february : List FebruaryDate) 
  (leap_year : february.length = 29)
  (four_mondays : (february.filter (fun d => isMonday d.dayOfWeek)).length = 4) :
  (february.head?.map (fun d => d.dayOfWeek) = some DayOfWeek.Sunday) :=
by
  sorry


end february_first_is_sunday_l490_49091


namespace find_k_l490_49032

theorem find_k (x y k : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -3) 
  (h3 : 2 * x^2 + k * x * y = 4) : 
  k = 2/3 := by
sorry

end find_k_l490_49032


namespace kevins_stamps_l490_49037

theorem kevins_stamps (carl_stamps : ℕ) (difference : ℕ) (h1 : carl_stamps = 89) (h2 : difference = 32) :
  carl_stamps - difference = 57 := by
  sorry

end kevins_stamps_l490_49037


namespace polynomial_symmetry_l490_49086

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 7, where a, b, and c are constants,
    if f(-2011) = -17, then f(2011) = 31. -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 7
  f (-2011) = -17 → f 2011 = 31 := by
  sorry

end polynomial_symmetry_l490_49086


namespace carpet_area_in_sq_yards_l490_49022

def living_room_length : ℝ := 18
def living_room_width : ℝ := 9
def storage_room_side : ℝ := 3
def sq_feet_per_sq_yard : ℝ := 9

theorem carpet_area_in_sq_yards :
  (living_room_length * living_room_width + storage_room_side * storage_room_side) / sq_feet_per_sq_yard = 19 := by
  sorry

end carpet_area_in_sq_yards_l490_49022


namespace towel_shrinkage_l490_49047

theorem towel_shrinkage (L B : ℝ) (h_positive : L > 0 ∧ B > 0) :
  let new_length := 0.8 * L
  let new_area := 0.72 * (L * B)
  ∃ new_breadth : ℝ, new_breadth = 0.9 * B ∧ new_length * new_breadth = new_area :=
by
  sorry

end towel_shrinkage_l490_49047


namespace base7_to_base10_conversion_l490_49071

-- Define the base 7 number as a list of digits
def base7_number : List Nat := [2, 5, 3, 4]

-- Define the conversion function from base 7 to base 10
def base7_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem base7_to_base10_conversion :
  base7_to_base10 base7_number = 956 := by
  sorry

end base7_to_base10_conversion_l490_49071


namespace acid_concentration_theorem_l490_49011

def acid_concentration_problem (acid1 acid2 acid3 : ℝ) (water : ℝ) : Prop :=
  let water1 := (acid1 / 0.05) - acid1
  let water2 := water - water1
  let conc2 := acid2 / (acid2 + water2)
  conc2 = 70 / 300 →
  let total_water := water1 + water2
  (acid3 / (acid3 + total_water)) * 100 = 10.5

theorem acid_concentration_theorem :
  acid_concentration_problem 10 20 30 255.714 :=
by sorry

end acid_concentration_theorem_l490_49011


namespace race_problem_l490_49036

/-- The race problem on a circular lake -/
theorem race_problem (lake_circumference : ℝ) (serezha_speed : ℝ) (dima_run_speed : ℝ) 
  (serezha_time : ℝ) (dima_run_time : ℝ) :
  serezha_speed = 20 →
  dima_run_speed = 6 →
  serezha_time = 0.5 →
  dima_run_time = 0.25 →
  ∃ (total_time : ℝ), total_time = 37.5 / 60 := by
  sorry

#check race_problem

end race_problem_l490_49036
