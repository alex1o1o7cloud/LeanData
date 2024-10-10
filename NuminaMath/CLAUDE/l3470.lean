import Mathlib

namespace delores_money_theorem_l3470_347045

/-- The amount of money Delores had at first, given her purchases and remaining money. -/
def delores_initial_money (computer_price printer_price headphones_price discount_rate remaining_money : ℚ) : ℚ :=
  let discounted_computer_price := computer_price * (1 - discount_rate)
  let total_spent := discounted_computer_price + printer_price + headphones_price
  total_spent + remaining_money

/-- Theorem stating that Delores had $470 at first. -/
theorem delores_money_theorem :
  delores_initial_money 400 40 60 (1/10) 10 = 470 := by
  sorry

end delores_money_theorem_l3470_347045


namespace range_of_a_l3470_347076

theorem range_of_a (a : ℝ) : 
  (∀ x, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ ¬(|x - 1| < 1)) →
  a ≤ 2 :=
by
  sorry

end range_of_a_l3470_347076


namespace smallest_valid_n_l3470_347027

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  2 * n = 10 * ones + tens + 3

theorem smallest_valid_n :
  is_valid 12 ∧ ∀ m : ℕ, is_valid m → 12 ≤ m :=
sorry

end smallest_valid_n_l3470_347027


namespace integer_root_characterization_l3470_347001

def polynomial (x b : ℤ) : ℤ := x^4 + 4*x^3 + 4*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_characterization (b : ℤ) :
  has_integer_root b ↔ b ∈ ({-38, -21, -2, 10, 13, 34} : Set ℤ) := by
  sorry

end integer_root_characterization_l3470_347001


namespace systematic_sampling_theorem_l3470_347043

theorem systematic_sampling_theorem (population : ℕ) (sample_size : ℕ) 
  (h1 : population = 1650) (h2 : sample_size = 35) :
  ∃ (exclude : ℕ) (segment_size : ℕ),
    exclude = population % sample_size ∧
    segment_size = (population - exclude) / sample_size ∧
    exclude = 5 ∧
    segment_size = 47 := by
  sorry

end systematic_sampling_theorem_l3470_347043


namespace systematic_sample_fourth_element_l3470_347000

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < s.sample_size ∧ n = s.first_element + k * s.interval

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 52)
  (h_sample : s.sample_size = 4)
  (h_5 : s.contains 5)
  (h_31 : s.contains 31)
  (h_44 : s.contains 44)
  : s.contains 18 := by
  sorry

#check systematic_sample_fourth_element

end systematic_sample_fourth_element_l3470_347000


namespace max_balls_for_five_weighings_impossibility_more_than_243_balls_l3470_347024

/-- The number of weighings required to identify the lighter ball -/
def num_weighings : ℕ := 5

/-- The maximum number of balls that can be tested with the given number of weighings -/
def max_balls : ℕ := 3^num_weighings

/-- Theorem stating that the maximum number of balls is 243 given 5 weighings -/
theorem max_balls_for_five_weighings :
  num_weighings = 5 → max_balls = 243 := by
  sorry

/-- Theorem stating that it's impossible to identify the lighter ball among more than 243 balls with 5 weighings -/
theorem impossibility_more_than_243_balls (n : ℕ) :
  num_weighings = 5 → n > 243 → ¬(∃ strategy : Unit, True) := by
  sorry

end max_balls_for_five_weighings_impossibility_more_than_243_balls_l3470_347024


namespace solve_linear_equation_l3470_347046

theorem solve_linear_equation (x : ℝ) (h : x + 1 = 4) : x = 3 := by
  sorry

end solve_linear_equation_l3470_347046


namespace necessary_condition_abs_l3470_347016

theorem necessary_condition_abs (x y : ℝ) (hx : x > 0) : x > |y| → x > y := by
  sorry

end necessary_condition_abs_l3470_347016


namespace jim_victory_percentage_l3470_347051

def total_votes : ℕ := 6000
def geoff_percent : ℚ := 1/200

theorem jim_victory_percentage (laura_votes geoff_votes jim_votes : ℕ) :
  geoff_votes = (geoff_percent * total_votes).num ∧
  laura_votes = 2 * geoff_votes ∧
  jim_votes = total_votes - (laura_votes + geoff_votes) ∧
  geoff_votes + 3000 > laura_votes ∧
  geoff_votes + 3000 > jim_votes →
  (jim_votes : ℚ) / total_votes ≥ 5052 / 10000 :=
by sorry

end jim_victory_percentage_l3470_347051


namespace total_pencils_l3470_347034

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) :
  jessica_pencils = 8 →
  sandy_pencils = 8 →
  jason_pencils = 8 →
  jessica_pencils + sandy_pencils + jason_pencils = 24 :=
by sorry

end total_pencils_l3470_347034


namespace total_sales_equals_250_l3470_347052

/-- Represents the commission rate as a percentage -/
def commission_rate : ℚ := 5 / 100

/-- Represents the commission earned in Rupees -/
def commission_earned : ℚ := 25 / 2

/-- Calculates the total sales given the commission rate and commission earned -/
def total_sales (rate : ℚ) (earned : ℚ) : ℚ := earned / rate

/-- Theorem stating that the total sales equal 250 Rupees -/
theorem total_sales_equals_250 : 
  total_sales commission_rate commission_earned = 250 := by
  sorry

end total_sales_equals_250_l3470_347052


namespace greatest_integer_abs_inequality_l3470_347014

theorem greatest_integer_abs_inequality :
  (∃ (x : ℤ), ∀ (y : ℤ), |3*y - 2| ≤ 21 → y ≤ x) ∧
  (∀ (x : ℤ), |3*x - 2| ≤ 21 → x ≤ 7) :=
by sorry

end greatest_integer_abs_inequality_l3470_347014


namespace simplify_expression_l3470_347011

theorem simplify_expression (x : ℝ) : (3*x + 20) + (200*x + 45) = 203*x + 65 := by
  sorry

end simplify_expression_l3470_347011


namespace share_difference_l3470_347055

theorem share_difference (total : ℕ) (a b c : ℕ) : 
  total = 120 →
  a = b + 20 →
  a < c →
  b = 20 →
  c - a = 20 := by
sorry

end share_difference_l3470_347055


namespace num_perfect_square_factors_is_525_l3470_347015

/-- The number of positive perfect square factors of 2^8 * 3^9 * 5^12 * 7^4 -/
def num_perfect_square_factors : ℕ := 525

/-- The exponents of prime factors in the given product -/
def prime_exponents : List ℕ := [8, 9, 12, 4]

/-- Counts the number of even numbers (including 0) up to and including a given number -/
def count_even_numbers_up_to (n : ℕ) : ℕ :=
  (n / 2) + 1

/-- Theorem: The number of positive perfect square factors of 2^8 * 3^9 * 5^12 * 7^4 is 525 -/
theorem num_perfect_square_factors_is_525 :
  num_perfect_square_factors = (prime_exponents.map count_even_numbers_up_to).prod :=
sorry

end num_perfect_square_factors_is_525_l3470_347015


namespace apple_cost_price_l3470_347072

/-- The cost price of an apple given its selling price and loss ratio. -/
def cost_price (selling_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  selling_price / (1 - loss_ratio)

/-- Theorem stating the cost price of an apple given specific conditions. -/
theorem apple_cost_price :
  let selling_price : ℚ := 17
  let loss_ratio : ℚ := 1/6
  cost_price selling_price loss_ratio = 20.4 := by
sorry

end apple_cost_price_l3470_347072


namespace final_water_percentage_l3470_347068

/-- Calculates the final percentage of water in a mixture after adding water -/
theorem final_water_percentage
  (initial_mixture : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h_initial_mixture : initial_mixture = 50)
  (h_initial_water_percentage : initial_water_percentage = 10)
  (h_added_water : added_water = 25) :
  let initial_water := initial_mixture * (initial_water_percentage / 100)
  let final_water := initial_water + added_water
  let final_mixture := initial_mixture + added_water
  (final_water / final_mixture) * 100 = 40 := by
sorry


end final_water_percentage_l3470_347068


namespace parabola_properties_l3470_347025

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the parabola at a given x -/
def Parabola.eval (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A parabola that passes through the given points -/
def specificParabola : Parabola :=
  { a := sorry
    b := sorry
    c := sorry }

theorem parabola_properties :
  let p := specificParabola
  -- The parabola passes through the given points
  (p.eval (-2) = 0) ∧
  (p.eval (-1) = 4) ∧
  (p.eval 0 = 6) ∧
  (p.eval 1 = 6) →
  -- 1. The parabola opens downwards
  (p.a < 0) ∧
  -- 2. The axis of symmetry is x = 1/2
  (- p.b / (2 * p.a) = 1/2) ∧
  -- 3. The maximum value of the function is 25/4
  (p.c - p.b^2 / (4 * p.a) = 25/4) := by
  sorry

end parabola_properties_l3470_347025


namespace inequality_relation_l3470_347057

theorem inequality_relation (n : ℕ) (hn : n > 1) :
  (1 : ℝ) / n > Real.log ((n + 1 : ℝ) / n) ∧
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := by
  sorry

end inequality_relation_l3470_347057


namespace leftover_floss_amount_l3470_347050

/-- Calculates the amount of leftover floss when distributing to students -/
def leftover_floss (num_students : ℕ) (floss_per_student : ℚ) (floss_per_packet : ℕ) : ℚ :=
  let total_needed : ℚ := num_students * floss_per_student
  let packets_needed : ℕ := (total_needed / floss_per_packet).ceil.toNat
  packets_needed * floss_per_packet - total_needed

/-- Theorem stating the leftover floss amount for the given problem -/
theorem leftover_floss_amount :
  leftover_floss 20 (3/2) 35 = 5 := by
sorry

end leftover_floss_amount_l3470_347050


namespace total_amount_proof_l3470_347060

def coffee_maker_price : ℝ := 70
def blender_price : ℝ := 100
def coffee_maker_discount : ℝ := 0.2
def blender_discount : ℝ := 0.15
def num_coffee_makers : ℕ := 2

def total_price : ℝ :=
  (num_coffee_makers : ℝ) * coffee_maker_price * (1 - coffee_maker_discount) +
  blender_price * (1 - blender_discount)

theorem total_amount_proof :
  total_price = 197 := by sorry

end total_amount_proof_l3470_347060


namespace dinner_bill_problem_l3470_347097

theorem dinner_bill_problem (P : ℝ) : 
  P > 0 →  -- Assuming the price is positive
  (0.9 * P + 0.15 * P) = (0.9 * P + 0.15 * 0.9 * P + 0.51) →
  P = 34 := by
  sorry

#check dinner_bill_problem

end dinner_bill_problem_l3470_347097


namespace parabola_vertex_l3470_347084

/-- The parabola defined by the equation y = 3(x-1)^2 + 2 has vertex at (1, 2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3*(x-1)^2 + 2 → (1, 2) = (x, y) := by
sorry

end parabola_vertex_l3470_347084


namespace similar_triangle_perimeter_l3470_347009

/-- Given an isosceles triangle with two equal sides of 15 cm and a base of 24 cm,
    prove that a similar triangle with a base of 60 cm has a perimeter of 135 cm. -/
theorem similar_triangle_perimeter 
  (original_equal_sides : ℝ)
  (original_base : ℝ)
  (similar_base : ℝ)
  (h_isosceles : original_equal_sides = 15)
  (h_original_base : original_base = 24)
  (h_similar_base : similar_base = 60) :
  let scale_factor := similar_base / original_base
  let similar_equal_sides := original_equal_sides * scale_factor
  similar_equal_sides * 2 + similar_base = 135 :=
by
  sorry

#check similar_triangle_perimeter

end similar_triangle_perimeter_l3470_347009


namespace arithmetic_expression_equality_l3470_347073

theorem arithmetic_expression_equality : 
  (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + 0.009 + (0.06 : ℝ)^2 = 0.006375 := by
  sorry

end arithmetic_expression_equality_l3470_347073


namespace jellybean_box_capacity_l3470_347049

theorem jellybean_box_capacity 
  (bert_capacity : ℕ)
  (bert_volume : ℝ)
  (lisa_volume : ℝ)
  (h1 : bert_capacity = 150)
  (h2 : lisa_volume = 24 * bert_volume)
  (h3 : ∀ (c : ℝ) (v : ℝ), c / v = bert_capacity / bert_volume → c = (v / bert_volume) * bert_capacity)
  : (lisa_volume / bert_volume) * bert_capacity = 3600 :=
by sorry

end jellybean_box_capacity_l3470_347049


namespace john_bought_three_sodas_l3470_347075

/-- Given a payment, cost per soda, and change received, calculate the number of sodas bought --/
def sodas_bought (payment : ℕ) (cost_per_soda : ℕ) (change : ℕ) : ℕ :=
  (payment - change) / cost_per_soda

/-- Theorem: John bought 3 sodas --/
theorem john_bought_three_sodas :
  sodas_bought 20 2 14 = 3 := by
  sorry

end john_bought_three_sodas_l3470_347075


namespace divisors_of_2_pow_56_minus_1_l3470_347012

theorem divisors_of_2_pow_56_minus_1 :
  ∃ (a b : ℕ), 95 < a ∧ a < 105 ∧ 95 < b ∧ b < 105 ∧
  a ≠ b ∧
  (2^56 - 1) % a = 0 ∧ (2^56 - 1) % b = 0 ∧
  (∀ c : ℕ, 95 < c ∧ c < 105 → (2^56 - 1) % c = 0 → c = a ∨ c = b) ∧
  a = 101 ∧ b = 127 :=
sorry

end divisors_of_2_pow_56_minus_1_l3470_347012


namespace reduction_equivalence_l3470_347032

def operation (seq : Vector ℤ 8) : Vector ℤ 8 :=
  Vector.ofFn (λ i => |seq.get i - seq.get ((i + 1) % 8)|)

def all_equal (seq : Vector ℤ 8) : Prop :=
  ∀ i j, seq.get i = seq.get j

def all_zero (seq : Vector ℤ 8) : Prop :=
  ∀ i, seq.get i = 0

def reduces_to_equal (init : Vector ℤ 8) : Prop :=
  ∃ n : ℕ, all_equal (n.iterate operation init)

def reduces_to_zero (init : Vector ℤ 8) : Prop :=
  ∃ n : ℕ, all_zero (n.iterate operation init)

theorem reduction_equivalence (init : Vector ℤ 8) :
  reduces_to_equal init ↔ reduces_to_zero init :=
sorry

end reduction_equivalence_l3470_347032


namespace circle_chord_and_area_l3470_347066

theorem circle_chord_and_area (r : ℝ) (d : ℝ) (h1 : r = 5) (h2 : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  let area := π * r^2
  chord_length = 6 ∧ area = 25 * π := by
  sorry

end circle_chord_and_area_l3470_347066


namespace fraction_problem_l3470_347070

theorem fraction_problem (f : ℚ) : 
  0.60 * 412.5 = f * 412.5 + 110 → f = 1/3 := by
  sorry

end fraction_problem_l3470_347070


namespace circular_triangle_angle_sum_l3470_347093

/-- Represents a circular triangle --/
structure CircularTriangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  r_a : ℝ  -- radius of arc forming side a
  r_b : ℝ  -- radius of arc forming side b
  r_c : ℝ  -- radius of arc forming side c
  s_a : Int  -- sign of side a (1 or -1)
  s_b : Int  -- sign of side b (1 or -1)
  s_c : Int  -- sign of side c (1 or -1)

/-- The theorem about the sum of angles in a circular triangle --/
theorem circular_triangle_angle_sum (t : CircularTriangle) :
  t.A + t.B + t.C - (t.s_a : ℝ) * (t.a / t.r_a) - (t.s_b : ℝ) * (t.b / t.r_b) - (t.s_c : ℝ) * (t.c / t.r_c) = π :=
by sorry

end circular_triangle_angle_sum_l3470_347093


namespace exists_quadrilateral_perpendicular_diagonals_not_all_natural_cubed_greater_than_squared_l3470_347094

-- Define a structure for a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a function to check if diagonals are perpendicular
def diagonalsPerpendicular (q : Quadrilateral) : Prop :=
  sorry

-- Statement 1
theorem exists_quadrilateral_perpendicular_diagonals :
  ∃ q : Quadrilateral, diagonalsPerpendicular q :=
sorry

-- Statement 2
theorem not_all_natural_cubed_greater_than_squared :
  ¬ ∀ x : ℕ, x^3 > x^2 :=
sorry

end exists_quadrilateral_perpendicular_diagonals_not_all_natural_cubed_greater_than_squared_l3470_347094


namespace half_angle_quadrant_l3470_347022

/-- An angle is in the third quadrant if it's between 180° and 270° (or equivalent in radians) -/
def is_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

/-- An angle is in the second or fourth quadrant if it's between 90° and 180° or between 270° and 360° (or equivalent in radians) -/
def is_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, (k * 2 * Real.pi + Real.pi / 2 < α ∧ α < k * 2 * Real.pi + Real.pi) ∨
           (k * 2 * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * 2 * Real.pi)

theorem half_angle_quadrant (α : Real) :
  is_third_quadrant α → is_second_or_fourth_quadrant (α / 2) := by
  sorry

end half_angle_quadrant_l3470_347022


namespace quadratic_function_properties_l3470_347031

/-- A quadratic function f(x) = x^2 + ax + b with specific properties -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

theorem quadratic_function_properties (a b : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), QuadraticFunction a b y ≥ QuadraticFunction a b x ∧ QuadraticFunction a b x = 2) →
  (∀ (x : ℝ), QuadraticFunction a b (2 - x) = QuadraticFunction a b x) →
  (∃ (m n : ℝ), m < n ∧
    (∀ (x : ℝ), m ≤ x ∧ x ≤ n → QuadraticFunction a b x ≤ 6) ∧
    (∃ (x : ℝ), m ≤ x ∧ x ≤ n ∧ QuadraticFunction a b x = 6)) →
  (∃ (m n : ℝ), n - m = 4 ∧
    ∀ (m' n' : ℝ), (∀ (x : ℝ), m' ≤ x ∧ x ≤ n' → QuadraticFunction a b x ≤ 6) →
    (∃ (x : ℝ), m' ≤ x ∧ x ≤ n' ∧ QuadraticFunction a b x = 6) →
    n' - m' ≤ 4) :=
by sorry

end quadratic_function_properties_l3470_347031


namespace f_inequality_l3470_347078

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem f_inequality (h1 : ∀ x, HasDerivAt f (f' x) x) (h2 : ∀ x, f' x < f x) : 
  f 3 < Real.exp 3 * f 0 := by
  sorry

end f_inequality_l3470_347078


namespace ninety_nine_squared_l3470_347056

theorem ninety_nine_squared : 99 * 99 = 9801 := by
  sorry

end ninety_nine_squared_l3470_347056


namespace second_hole_depth_l3470_347081

/-- Calculates the depth of a second hole given the conditions of two digging scenarios -/
theorem second_hole_depth
  (workers₁ : ℕ) (hours₁ : ℕ) (depth₁ : ℝ)
  (workers₂ : ℕ) (hours₂ : ℕ) :
  workers₁ = 45 →
  hours₁ = 8 →
  depth₁ = 30 →
  workers₂ = workers₁ + 45 →
  hours₂ = 6 →
  (workers₂ * hours₂ : ℝ) * (depth₁ / (workers₁ * hours₁ : ℝ)) = 45 :=
by sorry

end second_hole_depth_l3470_347081


namespace baker_cakes_remaining_l3470_347018

theorem baker_cakes_remaining (initial_cakes bought_cakes sold_cakes : ℕ) 
  (h1 : initial_cakes = 173)
  (h2 : bought_cakes = 103)
  (h3 : sold_cakes = 86) :
  initial_cakes + bought_cakes - sold_cakes = 190 := by
  sorry

end baker_cakes_remaining_l3470_347018


namespace one_fourth_of_8_point_8_l3470_347017

theorem one_fourth_of_8_point_8 : 
  (8.8 / 4 : ℚ) = 11 / 5 := by sorry

end one_fourth_of_8_point_8_l3470_347017


namespace exam_average_l3470_347061

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) (h1 : n1 = 15) (h2 : n2 = 10) 
  (h3 : avg1 = 80/100) (h4 : avg2 = 90/100) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 84/100 := by
  sorry

end exam_average_l3470_347061


namespace integral_exp_plus_2x_equals_e_l3470_347058

theorem integral_exp_plus_2x_equals_e : ∫ x in (0 : ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end integral_exp_plus_2x_equals_e_l3470_347058


namespace two_number_problem_l3470_347013

theorem two_number_problem :
  ∃ (x y : ℝ), 38 + 2 * x = 124 ∧ x + 3 * y = 47 ∧ x = 43 ∧ y = 4 / 3 := by
  sorry

end two_number_problem_l3470_347013


namespace smallest_three_digit_perfect_square_append_l3470_347010

theorem smallest_three_digit_perfect_square_append : ∃ (n : ℕ), 
  (n = 183) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(∃ k : ℕ, 1000 * m + (m + 1) = k^2)) ∧
  (∃ k : ℕ, 1000 * n + (n + 1) = k^2) := by
  sorry

end smallest_three_digit_perfect_square_append_l3470_347010


namespace intersection_nonempty_iff_m_leq_neg_one_l3470_347047

/-- Set A defined by the equation x^2 + mx - y + 2 = 0 -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + m * p.1 - p.2 + 2 = 0}

/-- Set B defined by the equation x - y + 1 = 0 with 0 ≤ x ≤ 2 -/
def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

/-- The main theorem stating that A ∩ B is nonempty if and only if m ≤ -1 -/
theorem intersection_nonempty_iff_m_leq_neg_one (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 := by
  sorry

end intersection_nonempty_iff_m_leq_neg_one_l3470_347047


namespace line_l_line_l_l3470_347002

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

/-- The equation of line l' that passes through (-1, 3) and is parallel to l -/
def line_l'_parallel (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

/-- The equation of line l' that is symmetric to l about the y-axis -/
def line_l'_symmetric (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

/-- Point (-1, 3) -/
def point : ℝ × ℝ := (-1, 3)

theorem line_l'_parallel_correct :
  (∀ x y, line_l'_parallel x y ↔ (∃ k, y - point.2 = k * (x - point.1) ∧
    ∀ x₁ y₁ x₂ y₂, line_l x₁ y₁ → line_l x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = k)) ∧
  line_l'_parallel point.1 point.2 :=
sorry

theorem line_l'_symmetric_correct :
  ∀ x y, line_l'_symmetric x y ↔ line_l (-x) y :=
sorry

end line_l_line_l_l3470_347002


namespace geometric_sequence_a7_l3470_347004

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 11 = 16 →
  a 7 = 4 := by
  sorry

end geometric_sequence_a7_l3470_347004


namespace donald_drinks_nine_l3470_347030

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℕ := 2 * paul_bottles + 3

/-- Theorem stating that Donald drinks 9 bottles of juice per day -/
theorem donald_drinks_nine : donald_bottles = 9 := by
  sorry

end donald_drinks_nine_l3470_347030


namespace quadratic_equation_transformation_l3470_347088

theorem quadratic_equation_transformation (a b c : ℝ) : 
  (∀ x, a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) →
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end quadratic_equation_transformation_l3470_347088


namespace tomato_seed_planting_l3470_347041

theorem tomato_seed_planting (mike_morning mike_afternoon ted_morning ted_afternoon total : ℕ) : 
  mike_morning = 50 →
  ted_morning = 2 * mike_morning →
  mike_afternoon = 60 →
  ted_afternoon < mike_afternoon →
  total = mike_morning + ted_morning + mike_afternoon + ted_afternoon →
  total = 250 →
  mike_afternoon - ted_afternoon = 20 := by
sorry

end tomato_seed_planting_l3470_347041


namespace beyonce_song_count_l3470_347086

/-- The number of singles released by Beyonce -/
def singles : Nat := 5

/-- The number of albums with 15 songs -/
def albums_15 : Nat := 2

/-- The number of songs in each of the albums_15 -/
def songs_per_album_15 : Nat := 15

/-- The number of albums with 20 songs -/
def albums_20 : Nat := 1

/-- The number of songs in each of the albums_20 -/
def songs_per_album_20 : Nat := 20

/-- The total number of songs released by Beyonce -/
def total_songs : Nat := singles + albums_15 * songs_per_album_15 + albums_20 * songs_per_album_20

theorem beyonce_song_count : total_songs = 55 := by
  sorry

end beyonce_song_count_l3470_347086


namespace expression_evaluation_l3470_347044

theorem expression_evaluation : 2 * (5 * 9) + 3 * (4 * 11) + (2^3 * 7) + 6 * (3 * 5) = 368 := by
  sorry

end expression_evaluation_l3470_347044


namespace system_solutions_l3470_347059

def system_solution (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∧
  (x^(Real.log x) * y^(Real.log y) = 243) ∧
  ((3 / Real.log x) * x * y^(Real.log y) = 1)

theorem system_solutions :
  {(x, y) : ℝ × ℝ | system_solution x y} =
  {(9, 3), (3, 9), (1/9, 1/3), (1/3, 1/9)} := by
sorry

end system_solutions_l3470_347059


namespace parallelepiped_net_theorem_l3470_347096

/-- Represents a parallelepiped with dimensions length, width, and height -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of unfolded parallelepiped -/
structure Net where
  squares : ℕ

/-- Calculates the surface area of a parallelepiped -/
def surfaceArea (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.length * p.height + p.width * p.height)

/-- Unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := surfaceArea p }

/-- Removes one square from a net -/
def removeSquare (n : Net) : Net :=
  { squares := n.squares - 1 }

theorem parallelepiped_net_theorem (p : Parallelepiped) 
  (h1 : p.length = 2) (h2 : p.width = 1) (h3 : p.height = 1) :
  ∃ (n : Net), 
    (unfold p).squares = 10 ∧ 
    (removeSquare (unfold p)).squares = 9 ∧
    ∃ (valid : Bool), valid = true :=
  sorry

end parallelepiped_net_theorem_l3470_347096


namespace bottle_capacity_proof_l3470_347040

theorem bottle_capacity_proof (x : ℚ) : 
  (16/3 : ℚ) / 8 * x + 16/3 = 8 → x = 4 := by
  sorry

end bottle_capacity_proof_l3470_347040


namespace cubic_equation_solution_l3470_347033

theorem cubic_equation_solution (p q : ℝ) :
  ∃ x : ℝ, x^3 + p*x + q = 0 ∧
  x = -(Real.rpow ((q/2) + Real.sqrt ((q^2/4) + (p^3/27))) (1/3)) -
      (Real.rpow ((q/2) - Real.sqrt ((q^2/4) + (p^3/27))) (1/3)) :=
by sorry

end cubic_equation_solution_l3470_347033


namespace equation_real_solution_l3470_347038

theorem equation_real_solution (x : ℝ) :
  (∀ y : ℝ, ∃ z : ℝ, x^2 + y^2 + z^2 + 2*x*y*z = 1) ↔ (x = 1 ∨ x = -1) :=
sorry

end equation_real_solution_l3470_347038


namespace existence_of_common_element_l3470_347021

theorem existence_of_common_element (ε : ℝ) (h_ε_pos : 0 < ε) (h_ε_bound : ε < 1/2) :
  ∃ m : ℕ+, ∀ x : ℝ, ∃ i : ℕ+, ∃ k : ℤ, i.val ≤ m.val ∧ |i.val • x - k| ≤ ε :=
sorry

end existence_of_common_element_l3470_347021


namespace number_less_than_abs_is_negative_l3470_347023

theorem number_less_than_abs_is_negative (x : ℝ) : x < |x| → x < 0 := by
  sorry

end number_less_than_abs_is_negative_l3470_347023


namespace blue_balls_unchanged_jungkook_blue_balls_l3470_347029

/-- Represents the number of balls of each color Jungkook has -/
structure BallCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Jungkook's initial ball count -/
def initial_count : BallCount :=
  { red := 5, blue := 4, yellow := 3 }

/-- Yoon-gi gives Jungkook a yellow ball -/
def give_yellow_ball (count : BallCount) : BallCount :=
  { count with yellow := count.yellow + 1 }

/-- The number of blue balls remains unchanged after receiving a yellow ball -/
theorem blue_balls_unchanged (count : BallCount) :
  (give_yellow_ball count).blue = count.blue :=
by sorry

/-- Jungkook has 4 blue balls after receiving a yellow ball from Yoon-gi -/
theorem jungkook_blue_balls :
  (give_yellow_ball initial_count).blue = 4 :=
by sorry

end blue_balls_unchanged_jungkook_blue_balls_l3470_347029


namespace grass_stains_count_l3470_347020

theorem grass_stains_count (grass_stain_time marinara_stain_time total_time : ℕ) 
  (marinara_stain_count : ℕ) (h1 : grass_stain_time = 4) 
  (h2 : marinara_stain_time = 7) (h3 : marinara_stain_count = 1) 
  (h4 : total_time = 19) : 
  ∃ (grass_stain_count : ℕ), 
    grass_stain_count * grass_stain_time + 
    marinara_stain_count * marinara_stain_time = total_time ∧ 
    grass_stain_count = 3 :=
by
  sorry

end grass_stains_count_l3470_347020


namespace condition_one_condition_two_l3470_347085

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for condition (1)
theorem condition_one :
  ∃! a : ℝ, A a ∩ B = A a ∪ B := by sorry

-- Theorem for condition (2)
theorem condition_two :
  ∃! a : ℝ, (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) := by sorry

end condition_one_condition_two_l3470_347085


namespace cafeteria_apples_l3470_347092

/-- The cafeteria problem -/
theorem cafeteria_apples (initial : ℕ) (used : ℕ) (bought : ℕ) :
  initial = 38 → used = 20 → bought = 28 → initial - used + bought = 46 := by
  sorry

end cafeteria_apples_l3470_347092


namespace arithmetic_calculation_l3470_347090

theorem arithmetic_calculation : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end arithmetic_calculation_l3470_347090


namespace inspection_ratio_l3470_347007

theorem inspection_ratio (j n : ℝ) (hj : j > 0) (hn : n > 0) : 
  0.005 * j + 0.007 * n = 0.0075 * (j + n) → n / j = 5 := by sorry

end inspection_ratio_l3470_347007


namespace light_bulb_investigation_l3470_347071

/-- Represents the method of investigation -/
inductive InvestigationMethod
  | SamplingSurvey
  | Census

/-- Represents the characteristics of the investigation -/
structure InvestigationCharacteristics where
  largeQuantity : Bool
  destructiveTesting : Bool

/-- Determines the appropriate investigation method based on the characteristics -/
def appropriateMethod (chars : InvestigationCharacteristics) : InvestigationMethod :=
  if chars.largeQuantity && chars.destructiveTesting then
    InvestigationMethod.SamplingSurvey
  else
    InvestigationMethod.Census

/-- Theorem stating that for light bulb service life investigation with given characteristics, 
    sampling survey is the appropriate method -/
theorem light_bulb_investigation 
  (chars : InvestigationCharacteristics) 
  (h1 : chars.largeQuantity = true) 
  (h2 : chars.destructiveTesting = true) : 
  appropriateMethod chars = InvestigationMethod.SamplingSurvey := by
  sorry

end light_bulb_investigation_l3470_347071


namespace blue_shells_count_l3470_347089

theorem blue_shells_count (total purple pink yellow orange : ℕ) 
  (h_total : total = 65)
  (h_purple : purple = 13)
  (h_pink : pink = 8)
  (h_yellow : yellow = 18)
  (h_orange : orange = 14) :
  total - (purple + pink + yellow + orange) = 12 := by
  sorry

end blue_shells_count_l3470_347089


namespace quadratic_root_l3470_347065

/-- Given a quadratic equation 2x^2 + 3x - k = 0 where k = 44, 
    prove that 4 is one of its roots. -/
theorem quadratic_root : ∃ x : ℝ, 2 * x^2 + 3 * x - 44 = 0 ∧ x = 4 := by
  sorry

end quadratic_root_l3470_347065


namespace dogs_liking_no_food_l3470_347069

def total_dogs : ℕ := 80
def watermelon_dogs : ℕ := 18
def salmon_dogs : ℕ := 58
def chicken_dogs : ℕ := 16
def watermelon_and_salmon : ℕ := 7
def chicken_and_salmon : ℕ := 6
def chicken_and_watermelon : ℕ := 4
def all_three : ℕ := 3

theorem dogs_liking_no_food : 
  total_dogs - (watermelon_dogs + salmon_dogs + chicken_dogs
              - watermelon_and_salmon - chicken_and_salmon - chicken_and_watermelon
              + all_three) = 2 := by
  sorry

end dogs_liking_no_food_l3470_347069


namespace kenny_trumpet_practice_l3470_347003

/-- Given Kenny's activities and their durations, prove that he practiced trumpet for 40 hours. -/
theorem kenny_trumpet_practice (x y z w : ℕ) : 
  let basketball : ℕ := 10
  let running : ℕ := 2 * basketball
  let trumpet : ℕ := 2 * running
  let other_activities : ℕ := x + y + z + w
  other_activities = basketball + running + trumpet - 5
  → trumpet = 40 := by
sorry

end kenny_trumpet_practice_l3470_347003


namespace circle_radius_in_isosceles_triangle_l3470_347053

theorem circle_radius_in_isosceles_triangle (a b c : Real) (r_p r_q : Real) : 
  a = 60 → b = 60 → c = 40 → r_p = 12 →
  -- Triangle ABC is isosceles with AB = AC = 60 and BC = 40
  -- Circle P has radius r_p = 12 and is tangent to AC and BC
  -- Circle Q is externally tangent to P and tangent to AB and BC
  -- No point of circle Q lies outside of triangle ABC
  r_q = 36 - 4 * Real.sqrt 14 := by
  sorry

end circle_radius_in_isosceles_triangle_l3470_347053


namespace problem_figure_perimeter_l3470_347035

/-- Represents the figure described in the problem -/
structure SquareFigure where
  stackHeight : Nat
  stackGap : Nat
  topSquares : Nat
  bottomSquares : Nat

/-- Calculates the perimeter of the square figure -/
def perimeterOfSquareFigure (fig : SquareFigure) : Nat :=
  let horizontalSegments := fig.topSquares * 2 + fig.bottomSquares * 2
  let verticalSegments := fig.stackHeight * 2 * 2 + fig.topSquares * 2
  horizontalSegments + verticalSegments

/-- The specific figure described in the problem -/
def problemFigure : SquareFigure :=
  { stackHeight := 3
  , stackGap := 1
  , topSquares := 3
  , bottomSquares := 2 }

theorem problem_figure_perimeter :
  perimeterOfSquareFigure problemFigure = 22 := by
  sorry

end problem_figure_perimeter_l3470_347035


namespace quadratic_equation_roots_range_l3470_347005

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 2*x - 1 = 0 ∧ (m - 1) * y^2 - 2*y - 1 = 0) ↔ 
  (m ≥ 0 ∧ m ≠ 1) :=
sorry

end quadratic_equation_roots_range_l3470_347005


namespace cos_600_degrees_l3470_347082

theorem cos_600_degrees : Real.cos (600 * π / 180) = - (1 / 2) := by
  sorry

end cos_600_degrees_l3470_347082


namespace ball_probability_theorem_l3470_347037

/-- Represents the two boxes containing balls -/
inductive Box
| A
| B

/-- Represents the color of the balls -/
inductive Color
| Red
| White

/-- Represents the number of balls in each box before transfer -/
def initial_count : Box → Color → ℕ
| Box.A, Color.Red => 4
| Box.A, Color.White => 2
| Box.B, Color.Red => 2
| Box.B, Color.White => 3

/-- Represents the probability space for this problem -/
structure BallProbability where
  /-- The probability of event A (red ball taken from box A) -/
  prob_A : ℝ
  /-- The probability of event B (white ball taken from box A) -/
  prob_B : ℝ
  /-- The probability of event C (red ball taken from box B after transfer) -/
  prob_C : ℝ
  /-- The conditional probability of C given A -/
  prob_C_given_A : ℝ

/-- The main theorem that encapsulates the problem -/
theorem ball_probability_theorem (p : BallProbability) : 
  p.prob_A + p.prob_B = 1 ∧ 
  p.prob_A * p.prob_B = 0 ∧
  p.prob_C_given_A = 1/2 ∧ 
  p.prob_C = 4/9 := by
  sorry


end ball_probability_theorem_l3470_347037


namespace money_ratio_to_jenna_l3470_347042

/-- Represents the financial transaction scenario with John, his uncle, and Jenna --/
def john_transaction (money_from_uncle money_to_jenna groceries_cost money_remaining : ℚ) : Prop :=
  money_from_uncle - money_to_jenna - groceries_cost = money_remaining

/-- Theorem stating the ratio of money given to Jenna to money received from uncle --/
theorem money_ratio_to_jenna :
  ∃ (money_to_jenna : ℚ),
    john_transaction 100 money_to_jenna 40 35 ∧
    money_to_jenna / 100 = 1 / 4 := by
  sorry

end money_ratio_to_jenna_l3470_347042


namespace minimum_values_l3470_347028

theorem minimum_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 + y^2 ≥ 1/2) ∧ (1/x + 1/y + 1/(x*y) ≥ 8) := by
  sorry

end minimum_values_l3470_347028


namespace stating_auntie_em_parking_probability_l3470_347008

/-- The number of parking spaces in the lot -/
def total_spaces : ℕ := 20

/-- The number of cars that arrive before Auntie Em -/
def cars_before : ℕ := 15

/-- The number of spaces Auntie Em's SUV requires -/
def suv_spaces : ℕ := 2

/-- The probability that Auntie Em can park her SUV -/
def prob_auntie_em_can_park : ℚ := 232 / 323

/-- 
Theorem stating that the probability of Auntie Em being able to park her SUV
is equal to 232/323, given the conditions of the parking lot problem.
-/
theorem auntie_em_parking_probability :
  let remaining_spaces := total_spaces - cars_before
  let total_arrangements := Nat.choose total_spaces cars_before
  let unfavorable_arrangements := Nat.choose (remaining_spaces + cars_before - 1) (remaining_spaces - 1)
  (1 : ℚ) - (unfavorable_arrangements : ℚ) / (total_arrangements : ℚ) = prob_auntie_em_can_park :=
by sorry

end stating_auntie_em_parking_probability_l3470_347008


namespace other_solution_quadratic_l3470_347080

theorem other_solution_quadratic (h : 48 * (3/4)^2 + 31 = 74 * (3/4) - 16) :
  48 * (11/12)^2 + 31 = 74 * (11/12) - 16 := by
  sorry

end other_solution_quadratic_l3470_347080


namespace midpoint_coordinate_sum_l3470_347048

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, -2) and (2, 10) is 9. -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 8
  let y1 : ℝ := -2
  let x2 : ℝ := 2
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 := by
  sorry

end midpoint_coordinate_sum_l3470_347048


namespace simplify_expression_l3470_347019

theorem simplify_expression (y : ℝ) : 5*y + 6*y + 7*y + 2 = 18*y + 2 := by
  sorry

end simplify_expression_l3470_347019


namespace two_numbers_problem_l3470_347087

theorem two_numbers_problem (x y : ℝ) 
  (sum_condition : x + y = 15)
  (relation_condition : 3 * x = 5 * y - 11)
  (smaller_number : x = 7) :
  y = 8 := by
  sorry

end two_numbers_problem_l3470_347087


namespace equation_solution_l3470_347064

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end equation_solution_l3470_347064


namespace system_solution_ratio_l3470_347091

theorem system_solution_ratio (x y c d : ℝ) : 
  (4 * x - 3 * y = c) →
  (6 * y - 8 * x = d) →
  (d ≠ 0) →
  (∃ x y, (4 * x - 3 * y = c) ∧ (6 * y - 8 * x = d)) →
  c / d = -1 / 2 := by
sorry

end system_solution_ratio_l3470_347091


namespace arc_length_formula_l3470_347083

theorem arc_length_formula (r : ℝ) (θ : ℝ) (h : r = 8) (h' : θ = 5 * π / 3) :
  r * θ = 40 * π / 3 := by
  sorry

end arc_length_formula_l3470_347083


namespace balloon_difference_l3470_347099

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687

theorem balloon_difference : james_balloons - (amy_balloons + felix_balloons) = 22 := by
  sorry

end balloon_difference_l3470_347099


namespace angle_C_is_30_degrees_l3470_347074

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle measure function
def angle_measure (T : Triangle) (vertex : ℕ) : ℝ := sorry

-- Define the side length function
def side_length (T : Triangle) (side : ℕ) : ℝ := sorry

theorem angle_C_is_30_degrees (T : Triangle) :
  angle_measure T 1 = π / 4 →  -- ∠A = 45°
  side_length T 1 = Real.sqrt 2 →  -- AB = √2
  side_length T 2 = 2 →  -- BC = 2
  angle_measure T 3 = π / 6  -- ∠C = 30°
  := by sorry

end angle_C_is_30_degrees_l3470_347074


namespace power_of_power_l3470_347098

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l3470_347098


namespace stamp_difference_l3470_347063

theorem stamp_difference (k a : ℕ) (h1 : k * 3 = a * 5) 
  (h2 : (k - 12) * 6 = (a + 12) * 8) : k - 12 - (a + 12) = 32 := by
  sorry

end stamp_difference_l3470_347063


namespace largest_integer_solution_negative_six_is_largest_largest_integer_is_negative_six_l3470_347036

theorem largest_integer_solution (x : ℤ) : (7 - 3 * x > 22) ↔ (x ≤ -6) :=
  sorry

theorem negative_six_is_largest : ∃ (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22)) :=
  sorry

theorem largest_integer_is_negative_six : (∃! (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22))) ∧ 
  (∀ (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22)) → x = -6) :=
  sorry

end largest_integer_solution_negative_six_is_largest_largest_integer_is_negative_six_l3470_347036


namespace largest_x_value_l3470_347079

theorem largest_x_value (x : ℝ) : 
  (x / 4 + 1 / (6 * x) = 2 / 3) → 
  x ≤ (4 + Real.sqrt 10) / 3 ∧ 
  ∃ y, y / 4 + 1 / (6 * y) = 2 / 3 ∧ y = (4 + Real.sqrt 10) / 3 :=
by sorry

end largest_x_value_l3470_347079


namespace weight_per_rep_l3470_347006

-- Define the given conditions
def reps_per_set : ℕ := 10
def num_sets : ℕ := 3
def total_weight : ℕ := 450

-- Define the theorem to prove
theorem weight_per_rep :
  total_weight / (reps_per_set * num_sets) = 15 := by
  sorry

end weight_per_rep_l3470_347006


namespace system_solution_equation_solution_l3470_347039

-- Problem 1: System of equations
theorem system_solution (x y : ℝ) : x + 2*y = 3 ∧ 2*x - y = 1 → x = 1 ∧ y = 1 := by
  sorry

-- Problem 2: Algebraic equation
theorem equation_solution (x : ℝ) : x ≠ 1 → (1 / (x - 1) + 2 = 5 / (1 - x)) → x = -2 := by
  sorry

end system_solution_equation_solution_l3470_347039


namespace election_defeat_margin_l3470_347062

theorem election_defeat_margin 
  (total_votes : ℕ) 
  (invalid_votes : ℕ) 
  (defeated_percentage : ℚ) :
  total_votes = 90830 →
  invalid_votes = 83 →
  defeated_percentage = 45/100 →
  ∃ (valid_votes winning_votes losing_votes : ℕ),
    valid_votes = total_votes - invalid_votes ∧
    losing_votes = ⌊(defeated_percentage : ℝ) * valid_votes⌋ ∧
    winning_votes = valid_votes - losing_votes ∧
    winning_votes - losing_votes = 9074 :=
by sorry

end election_defeat_margin_l3470_347062


namespace certain_event_red_ball_l3470_347067

/-- A bag containing colored balls -/
structure Bag where
  yellow : ℕ
  red : ℕ

/-- The probability of drawing at least one red ball when drawing two balls from the bag -/
def prob_at_least_one_red (b : Bag) : ℚ :=
  1 - (b.yellow / (b.yellow + b.red)) * ((b.yellow - 1) / (b.yellow + b.red - 1))

/-- Theorem stating that drawing at least one red ball is a certain event 
    when drawing two balls from a bag with one yellow and three red balls -/
theorem certain_event_red_ball : 
  let b : Bag := { yellow := 1, red := 3 }
  prob_at_least_one_red b = 1 := by
  sorry

end certain_event_red_ball_l3470_347067


namespace cos_sin_eq_sin_cos_third_l3470_347077

theorem cos_sin_eq_sin_cos_third (x : ℝ) :
  -π ≤ x ∧ x ≤ π ∧ Real.cos (Real.sin x) = Real.sin (Real.cos (x / 3)) → x = 0 :=
by sorry

end cos_sin_eq_sin_cos_third_l3470_347077


namespace no_solutions_for_equation_l3470_347095

theorem no_solutions_for_equation : ¬∃ (n : ℕ+), (1 + 1 / n.val : ℝ) ^ (n.val + 1) = (1 + 1 / 1998 : ℝ) ^ 1998 := by
  sorry

end no_solutions_for_equation_l3470_347095


namespace arithmetic_progression_poly_j_value_l3470_347026

/-- A polynomial of degree 4 with four distinct real roots in arithmetic progression -/
structure ArithmeticProgressionPoly where
  j : ℝ
  k : ℝ
  roots_distinct : True
  roots_real : True
  roots_arithmetic : True

/-- The value of j in the polynomial x^4 + jx^2 + kx + 81 with four distinct real roots in arithmetic progression is -10 -/
theorem arithmetic_progression_poly_j_value (p : ArithmeticProgressionPoly) : p.j = -10 := by
  sorry

end arithmetic_progression_poly_j_value_l3470_347026


namespace common_tangent_intersection_l3470_347054

/-- Ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Ellipse C₂ -/
def C₂ (x y : ℝ) : Prop := (x-2)^2 + 4*y^2 = 1

/-- Common tangent to C₁ and C₂ -/
def common_tangent (x y : ℝ) : Prop :=
  ∃ (k b : ℝ), y = k*x + b ∧
    (∀ x' y', C₁ x' y' → (y' - (k*x' + b))^2 ≥ (k*(x - x'))^2) ∧
    (∀ x' y', C₂ x' y' → (y' - (k*x' + b))^2 ≥ (k*(x - x'))^2)

theorem common_tangent_intersection :
  ∃ (x y : ℝ), common_tangent x y ∧ y = 0 ∧ x = 4 :=
sorry

end common_tangent_intersection_l3470_347054
