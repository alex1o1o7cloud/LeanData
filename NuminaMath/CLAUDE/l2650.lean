import Mathlib

namespace area_cubic_line_theorem_l2650_265002

noncomputable def area_cubic_line (a b c d p q α β : ℝ) : ℝ :=
  |a| / 12 * (β - α)^4

theorem area_cubic_line_theorem (a b c d p q α β : ℝ) 
  (ha : a ≠ 0) 
  (hαβ : α ≠ β) 
  (htouch : ∀ x, a * x^3 + b * x^2 + c * x + d = p * x + q → x = α → 
    (3 * a * x^2 + 2 * b * x + c = p))
  (hintersect : a * β^3 + b * β^2 + c * β + d = p * β + q) :
  area_cubic_line a b c d p q α β = 
    ∫ x in α..β, |a * x^3 + b * x^2 + c * x + d - (p * x + q)| :=
by sorry

end area_cubic_line_theorem_l2650_265002


namespace simplify_fraction_l2650_265028

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by sorry

end simplify_fraction_l2650_265028


namespace hyperbola_equation_l2650_265098

/-- Given a hyperbola with the specified properties, prove its equation is x²/8 - y²/8 = 1 -/
theorem hyperbola_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (c / a = Real.sqrt 2) →                   -- Eccentricity is √2
  (4 / c = 1) →                             -- Slope of line through F(-c,0) and P(0,4) is 1
  (a = b) →                                 -- Equilateral hyperbola
  (∀ x y : ℝ, x^2 / 8 - y^2 / 8 = 1) :=     -- Resulting equation
by sorry

end hyperbola_equation_l2650_265098


namespace exam_max_score_l2650_265047

/-- The maximum score awarded in an exam given the following conditions:
    1. Gibi scored 59 percent
    2. Jigi scored 55 percent
    3. Mike scored 99 percent
    4. Lizzy scored 67 percent
    5. The average mark scored by all 4 students is 490 -/
theorem exam_max_score :
  let gibi_percent : ℚ := 59 / 100
  let jigi_percent : ℚ := 55 / 100
  let mike_percent : ℚ := 99 / 100
  let lizzy_percent : ℚ := 67 / 100
  let num_students : ℕ := 4
  let average_score : ℚ := 490
  let total_score : ℚ := average_score * num_students
  let sum_percentages : ℚ := gibi_percent + jigi_percent + mike_percent + lizzy_percent
  max_score * sum_percentages = total_score →
  max_score = 700 := by
sorry


end exam_max_score_l2650_265047


namespace rain_probability_l2650_265020

/-- Given probabilities of rain events in counties, prove the probability of rain on both days -/
theorem rain_probability (p_monday p_tuesday p_no_rain : ℝ) 
  (h1 : p_monday = 0.6)
  (h2 : p_tuesday = 0.55)
  (h3 : p_no_rain = 0.25) :
  p_monday + p_tuesday - (1 - p_no_rain) = 0.4 :=
by sorry

end rain_probability_l2650_265020


namespace one_third_of_recipe_l2650_265018

theorem one_third_of_recipe (full_recipe : ℚ) (one_third_recipe : ℚ) : 
  full_recipe = 17 / 3 ∧ one_third_recipe = full_recipe / 3 → one_third_recipe = 17 / 9 := by
  sorry

#check one_third_of_recipe

end one_third_of_recipe_l2650_265018


namespace complex_imaginary_solution_l2650_265076

/-- Given that z = m^2 - (1-i)m is an imaginary number, prove that m = 1 -/
theorem complex_imaginary_solution (m : ℂ) : 
  let z := m^2 - (1 - Complex.I) * m
  (∃ (y : ℝ), z = Complex.I * y) ∧ z ≠ 0 → m = 1 := by
sorry

end complex_imaginary_solution_l2650_265076


namespace range_of_t_l2650_265071

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, (|x - t| < 1 → 1 < x ∧ x ≤ 4)) →
  (2 ≤ t ∧ t ≤ 3) :=
by sorry

end range_of_t_l2650_265071


namespace trig_expression_value_l2650_265051

theorem trig_expression_value (α : ℝ) (h : Real.tan α = -2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / Real.sin α ^ 2 = -5/4 := by
  sorry

end trig_expression_value_l2650_265051


namespace steps_down_empire_state_proof_l2650_265049

/-- The number of steps taken to get down the Empire State Building -/
def steps_down_empire_state : ℕ := sorry

/-- The number of steps taken from the Empire State Building to Madison Square Garden -/
def steps_to_madison_square : ℕ := 315

/-- The total number of steps taken to get to Madison Square Garden -/
def total_steps : ℕ := 991

/-- Theorem stating that the number of steps taken to get down the Empire State Building is 676 -/
theorem steps_down_empire_state_proof : 
  steps_down_empire_state = total_steps - steps_to_madison_square := by sorry

end steps_down_empire_state_proof_l2650_265049


namespace geometric_sequence_ratio_l2650_265022

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  prop1 : a 5 * a 7 = 2
  prop2 : a 2 + a 10 = 3

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 12 / seq.a 4 = 2 ∨ seq.a 12 / seq.a 4 = 1/2 := by
  sorry

end geometric_sequence_ratio_l2650_265022


namespace triangle_height_and_median_l2650_265089

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def triangle : Triangle := {
  A := (4, 0)
  B := (6, 6)
  C := (0, 2)
}

def is_height_equation (t : Triangle) (eq : ℝ → ℝ → ℝ) : Prop :=
  let (x₁, y₁) := t.A
  ∀ x y, eq x y = 0 ↔ 3 * x + 2 * y - 12 = 0

def is_median_equation (t : Triangle) (eq : ℝ → ℝ → ℝ) : Prop :=
  let (x₁, y₁) := t.B
  ∀ x y, eq x y = 0 ↔ x + 2 * y - 18 = 0

theorem triangle_height_and_median :
  ∃ (height_eq median_eq : ℝ → ℝ → ℝ),
    is_height_equation triangle height_eq ∧
    is_median_equation triangle median_eq :=
  sorry

end triangle_height_and_median_l2650_265089


namespace triangle_inequality_theorem_l2650_265043

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the inequality function
def inequality_function (t : Triangle) : ℝ :=
  t.a^2 * t.b * (t.a - t.b) + t.b^2 * t.c * (t.b - t.c) + t.c^2 * t.a * (t.c - t.a)

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  inequality_function t ≥ 0 ∧
  (inequality_function t = 0 ↔ t.a = t.b ∧ t.b = t.c) := by
  sorry

end triangle_inequality_theorem_l2650_265043


namespace runner_speed_l2650_265035

/-- Proves that a runner covering 11.4 km in 2 minutes has a speed of 95 m/s -/
theorem runner_speed : ∀ (distance : ℝ) (time : ℝ),
  distance = 11.4 ∧ time = 2 →
  (distance * 1000) / (time * 60) = 95 := by
  sorry

end runner_speed_l2650_265035


namespace weight_of_Na2Ca_CO3_2_l2650_265004

-- Define molar masses of elements
def Na_mass : ℝ := 22.99
def Ca_mass : ℝ := 40.08
def C_mass : ℝ := 12.01
def O_mass : ℝ := 16.00

-- Define the number of atoms in Na2Ca(CO3)2
def Na_count : ℕ := 2
def Ca_count : ℕ := 1
def C_count : ℕ := 2
def O_count : ℕ := 6

-- Define the number of moles of Na2Ca(CO3)2
def moles : ℝ := 3.75

-- Define the molar mass of Na2Ca(CO3)2
def Na2Ca_CO3_2_mass : ℝ :=
  Na_count * Na_mass + Ca_count * Ca_mass + C_count * C_mass + O_count * O_mass

-- Theorem: The weight of 3.75 moles of Na2Ca(CO3)2 is 772.8 grams
theorem weight_of_Na2Ca_CO3_2 : moles * Na2Ca_CO3_2_mass = 772.8 := by
  sorry

end weight_of_Na2Ca_CO3_2_l2650_265004


namespace cos_pi_minus_alpha_l2650_265096

theorem cos_pi_minus_alpha (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (π - α) = -1/6 := by
sorry

end cos_pi_minus_alpha_l2650_265096


namespace smallest_checkered_rectangle_l2650_265031

/-- A rectangle that can be divided into both 1 × 13 rectangles and three-cell corners -/
structure CheckeredRectangle where
  width : ℕ
  height : ℕ
  dividable_13 : width * height % 13 = 0
  dividable_3 : width ≥ 2 ∧ height ≥ 2

/-- The area of a CheckeredRectangle -/
def area (r : CheckeredRectangle) : ℕ := r.width * r.height

/-- The perimeter of a CheckeredRectangle -/
def perimeter (r : CheckeredRectangle) : ℕ := 2 * (r.width + r.height)

/-- The set of all valid CheckeredRectangles -/
def valid_rectangles : Set CheckeredRectangle :=
  {r : CheckeredRectangle | true}

theorem smallest_checkered_rectangle :
  ∃ (r : CheckeredRectangle),
    r ∈ valid_rectangles ∧
    area r = 78 ∧
    (∀ (s : CheckeredRectangle), s ∈ valid_rectangles → area s ≥ area r) ∧
    (∃ (p : List ℕ), p = [38, 58, 82] ∧ (perimeter r) ∈ p) :=
by
  sorry

end smallest_checkered_rectangle_l2650_265031


namespace A_3_1_l2650_265057

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_1 : A 3 1 = 10 := by
  sorry

end A_3_1_l2650_265057


namespace least_sum_with_conditions_l2650_265083

theorem least_sum_with_conditions (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 210 = 1)
  (h2 : ∃ k : ℕ, m^m.val = k * n^n.val)
  (h3 : ¬∃ k : ℕ, m = k * n) :
  (∀ p q : ℕ+, 
    Nat.gcd (p + q) 210 = 1 → 
    (∃ k : ℕ, p^p.val = k * q^q.val) → 
    (¬∃ k : ℕ, p = k * q) → 
    m + n ≤ p + q) →
  m + n = 407 := by
sorry

end least_sum_with_conditions_l2650_265083


namespace tangent_chord_distance_l2650_265066

theorem tangent_chord_distance (R a : ℝ) (h : R > 0) :
  let x := R
  let m := 2 * R
  16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0 :=
by sorry

end tangent_chord_distance_l2650_265066


namespace sum_of_repeating_decimals_l2650_265033

/-- Definition of the repeating decimal 0.4444... -/
def repeating_4 : ℚ := 4 / 9

/-- Definition of the repeating decimal 0.3535... -/
def repeating_35 : ℚ := 35 / 99

/-- The sum of the repeating decimals 0.4444... and 0.3535... is equal to 79/99 -/
theorem sum_of_repeating_decimals : repeating_4 + repeating_35 = 79 / 99 := by
  sorry

end sum_of_repeating_decimals_l2650_265033


namespace minimum_value_of_f_plus_f_deriv_l2650_265054

/-- The function f(x) = -x^3 + ax^2 - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f_deriv (a : ℝ) :
  (∃ (x : ℝ), f_deriv a x = 0 ∧ x = 2) →
  (∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 → n' ∈ Set.Icc (-1 : ℝ) 1 →
      f a m + f_deriv a n ≤ f a m' + f_deriv a n') →
  ∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_deriv a n = -13 :=
sorry

end minimum_value_of_f_plus_f_deriv_l2650_265054


namespace jessica_cut_orchids_l2650_265075

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 2

/-- The number of orchids in the vase after cutting -/
def final_orchids : ℕ := 21

/-- The number of orchids Jessica cut -/
def orchids_cut : ℕ := final_orchids - initial_orchids

theorem jessica_cut_orchids : orchids_cut = 19 := by
  sorry

end jessica_cut_orchids_l2650_265075


namespace arithmetic_sequence_a6_l2650_265053

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_a6 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 = 1 → a 7 = 16 → a 6 = 11 := by
  sorry

end arithmetic_sequence_a6_l2650_265053


namespace smallest_two_digit_number_divisible_by_170_l2650_265014

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem smallest_two_digit_number_divisible_by_170 :
  ∃ (N : ℕ), is_two_digit N ∧
  (sum_of_digits (10^N - N) % 170 = 0) ∧
  (∀ (M : ℕ), is_two_digit M → sum_of_digits (10^M - M) % 170 = 0 → N ≤ M) ∧
  N = 20 := by sorry

end smallest_two_digit_number_divisible_by_170_l2650_265014


namespace triangle_inequality_l2650_265012

/-- Given a non-isosceles triangle with sides a, b, c and area S,
    prove the inequality relating the sides and the area. -/
theorem triangle_inequality (a b c S : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- non-isosceles condition
  S > 0 →  -- area is positive
  S = Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * 
    (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) →  -- Heron's formula
  (a^3 / ((a-b)*(a-c))) + (b^3 / ((b-c)*(b-a))) + 
    (c^3 / ((c-a)*(c-b))) > 2 * 3^(3/4) * S^(1/2) := by
  sorry

end triangle_inequality_l2650_265012


namespace parabola_min_y_l2650_265086

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  y + x = (y - x)^2 + 3*(y - x) + 3

/-- Theorem stating the minimum value of y for points on the parabola -/
theorem parabola_min_y :
  (∀ x y : ℝ, parabola_eq x y → y ≥ -1/2) ∧
  (∃ x y : ℝ, parabola_eq x y ∧ y = -1/2) := by
sorry

end parabola_min_y_l2650_265086


namespace triangular_pyramid_can_be_oblique_l2650_265092

/-- A pyramid with a regular triangular base and isosceles triangular lateral faces -/
structure TriangularPyramid where
  /-- The base of the pyramid is a regular triangle -/
  base_is_regular : Bool
  /-- Each lateral face is an isosceles triangle -/
  lateral_faces_isosceles : Bool

/-- Definition of an oblique pyramid -/
def is_oblique_pyramid (p : TriangularPyramid) : Prop :=
  ∃ (lateral_edge base_edge : ℝ), lateral_edge ≠ base_edge

/-- Theorem stating that a TriangularPyramid can be an oblique pyramid -/
theorem triangular_pyramid_can_be_oblique (p : TriangularPyramid) 
  (h1 : p.base_is_regular = true) 
  (h2 : p.lateral_faces_isosceles = true) : 
  ∃ (q : TriangularPyramid), is_oblique_pyramid q :=
sorry

end triangular_pyramid_can_be_oblique_l2650_265092


namespace thirteen_pow_seven_mod_eleven_l2650_265064

theorem thirteen_pow_seven_mod_eleven : 13^7 % 11 = 7 := by
  sorry

end thirteen_pow_seven_mod_eleven_l2650_265064


namespace apple_pie_cost_per_serving_l2650_265016

/-- Calculates the cost per serving of an apple pie given the ingredients and their costs. -/
def cost_per_serving (granny_smith_weight : Float) (granny_smith_price : Float)
                     (gala_weight : Float) (gala_price : Float)
                     (honeycrisp_weight : Float) (honeycrisp_price : Float)
                     (pie_crust_price : Float) (lemon_price : Float) (butter_price : Float)
                     (servings : Nat) : Float :=
  let total_cost := granny_smith_weight * granny_smith_price +
                    gala_weight * gala_price +
                    honeycrisp_weight * honeycrisp_price +
                    pie_crust_price + lemon_price + butter_price
  total_cost / servings.toFloat

/-- The cost per serving of the apple pie is $1.16375. -/
theorem apple_pie_cost_per_serving :
  cost_per_serving 0.5 1.80 0.8 2.20 0.7 2.50 2.50 0.60 1.80 8 = 1.16375 := by
  sorry

end apple_pie_cost_per_serving_l2650_265016


namespace additional_toothpicks_3_to_5_l2650_265067

/-- The number of toothpicks needed for a staircase of n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else if n = 2 then 10
  else if n = 3 then 18
  else toothpicks (n - 1) + 2 * n + 2

theorem additional_toothpicks_3_to_5 :
  toothpicks 5 - toothpicks 3 = 22 :=
sorry

end additional_toothpicks_3_to_5_l2650_265067


namespace fourth_number_in_sequence_l2650_265095

theorem fourth_number_in_sequence (s : Fin 7 → ℝ) 
  (h1 : (s 0 + s 1 + s 2 + s 3) / 4 = 4)
  (h2 : (s 3 + s 4 + s 5 + s 6) / 4 = 4)
  (h3 : (s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6) / 7 = 3) :
  s 3 = 5.5 := by
  sorry

end fourth_number_in_sequence_l2650_265095


namespace simplify_fraction_l2650_265008

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (10 * x * y^2) / (5 * x * y) = 2 * y :=
sorry

end simplify_fraction_l2650_265008


namespace initial_children_count_l2650_265081

/-- The number of children who got off the bus -/
def children_off : ℕ := 22

/-- The number of children left on the bus after some got off -/
def children_left : ℕ := 21

/-- The initial number of children on the bus -/
def initial_children : ℕ := children_off + children_left

theorem initial_children_count : initial_children = 43 := by
  sorry

end initial_children_count_l2650_265081


namespace michael_pet_sitting_cost_l2650_265091

/-- Calculates the total cost of pet sitting for one night -/
def pet_sitting_cost (num_cats num_dogs num_parrots num_fish : ℕ) 
                     (cost_per_cat cost_per_dog cost_per_parrot cost_per_fish : ℕ) : ℕ :=
  num_cats * cost_per_cat + 
  num_dogs * cost_per_dog + 
  num_parrots * cost_per_parrot + 
  num_fish * cost_per_fish

/-- Theorem: The total cost of pet sitting for Michael's pets for one night is $106 -/
theorem michael_pet_sitting_cost : 
  pet_sitting_cost 2 3 1 4 13 18 10 4 = 106 := by
  sorry

end michael_pet_sitting_cost_l2650_265091


namespace trigonometric_identities_l2650_265065

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 ∧
  sin α * cos α + cos α ^ 2 = 3 / 5 := by
  sorry

end trigonometric_identities_l2650_265065


namespace gcd_cube_plus_eight_and_n_plus_three_l2650_265097

theorem gcd_cube_plus_eight_and_n_plus_three (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 + 2^3) (n + 3) = 9 := by sorry

end gcd_cube_plus_eight_and_n_plus_three_l2650_265097


namespace quadratic_function_specific_points_l2650_265010

/-- A quadratic function passing through three specific points has a specific value for 3a - 2b + c -/
theorem quadratic_function_specific_points (a b c : ℤ) : 
  (1^2 * a + 1 * b + c = 6) → 
  ((-1)^2 * a + (-1) * b + c = 4) → 
  (0^2 * a + 0 * b + c = 3) → 
  3*a - 2*b + c = 7 := by
  sorry

end quadratic_function_specific_points_l2650_265010


namespace decimal_ratio_is_half_l2650_265088

/-- The decimal representation of 0.8571 repeating -/
def decimal_8571 : ℚ := 8571 / 9999

/-- The decimal representation of 2.142857 repeating -/
def decimal_2142857 : ℚ := 2142857 / 999999

/-- The main theorem stating that the ratio of the two decimals is 1/2 -/
theorem decimal_ratio_is_half : decimal_8571 / (2 + decimal_2142857) = 1 / 2 := by
  sorry

end decimal_ratio_is_half_l2650_265088


namespace min_guesses_correct_l2650_265073

/-- The minimum number of guesses required to determine a binary string -/
def minGuesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  minGuesses n k = (if n = 2 * k then 2 else 1) :=
by sorry

end min_guesses_correct_l2650_265073


namespace sum_of_products_l2650_265058

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 124) :
  x*y + y*z + x*z = 48 := by
sorry

end sum_of_products_l2650_265058


namespace total_fruits_in_baskets_total_fruits_proof_l2650_265068

/-- Given a group of 4 fruit baskets, where the first three baskets contain 9 apples, 
    15 oranges, and 14 bananas each, and the fourth basket contains 2 less of each fruit 
    compared to the other baskets, prove that the total number of fruits is 70. -/
theorem total_fruits_in_baskets : ℕ :=
  let apples_per_basket : ℕ := 9
  let oranges_per_basket : ℕ := 15
  let bananas_per_basket : ℕ := 14
  let num_regular_baskets : ℕ := 3
  let fruits_per_regular_basket : ℕ := apples_per_basket + oranges_per_basket + bananas_per_basket
  let fruits_in_regular_baskets : ℕ := fruits_per_regular_basket * num_regular_baskets
  let reduction_in_last_basket : ℕ := 2
  let fruits_in_last_basket : ℕ := fruits_per_regular_basket - (3 * reduction_in_last_basket)
  let total_fruits : ℕ := fruits_in_regular_baskets + fruits_in_last_basket
  70

theorem total_fruits_proof : total_fruits_in_baskets = 70 := by
  sorry

end total_fruits_in_baskets_total_fruits_proof_l2650_265068


namespace max_triangle_area_l2650_265048

/-- Parabola with focus at (0,1) and equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (0, 1)

/-- Vector from F to a point -/
def vec (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - F.1, p.2 - F.2)

/-- Condition that A, B, C are on the parabola and FA + FB + FC = 0 -/
def PointsCondition (A B C : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ C ∈ Parabola ∧
  vec A + vec B + vec C = (0, 0)

/-- Area of a triangle given three points -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The theorem to be proved -/
theorem max_triangle_area :
  ∀ A B C : ℝ × ℝ,
  PointsCondition A B C →
  TriangleArea A B C ≤ (3 * Real.sqrt 6) / 2 :=
sorry

end max_triangle_area_l2650_265048


namespace mixed_number_properties_l2650_265019

theorem mixed_number_properties :
  let x : ℚ := -1 - 2/7
  (1 / x = -7/9) ∧
  (-x = 1 + 2/7) ∧
  (|x| = 1 + 2/7) :=
by sorry

end mixed_number_properties_l2650_265019


namespace line_ellipse_intersection_angle_range_l2650_265038

/-- The range of inclination angles for which a line intersects an ellipse at two distinct points -/
theorem line_ellipse_intersection_angle_range 
  (A : ℝ × ℝ) 
  (l : ℝ → ℝ × ℝ) 
  (α : ℝ) 
  (ellipse : ℝ × ℝ → Prop) : 
  A = (-2, 0) →
  (∀ t, l t = (-2 + t * Real.cos α, t * Real.sin α)) →
  (ellipse (x, y) ↔ x^2 / 2 + y^2 = 1) →
  (∃ B C, B ≠ C ∧ ellipse B ∧ ellipse C ∧ ∃ t₁ t₂, l t₁ = B ∧ l t₂ = C) ↔
  (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
  (Real.pi - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < Real.pi) :=
by sorry

end line_ellipse_intersection_angle_range_l2650_265038


namespace expression_simplification_l2650_265007

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l2650_265007


namespace max_expression_value_l2650_265023

def max_expression (a b c d : ℕ) : ℕ := c * a^(b + d)

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    max_expression a b c d = 1024 ∧
    ∀ (w x y z : ℕ),
      w ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      x ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      y ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      z ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
      max_expression w x y z ≤ 1024 :=
by
  sorry

end max_expression_value_l2650_265023


namespace arithmetic_sequence_sum_l2650_265070

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
    common difference d = 2, and a₅ = 10, prove that S₁₀ = 110 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * 2)) →  -- sum formula
  a 5 = 10 →  -- given condition
  S 10 = 110 := by
sorry

end arithmetic_sequence_sum_l2650_265070


namespace g_equals_power_of_two_times_power_of_three_num_divisors_g_2023_equals_4096576_l2650_265074

/-- Function g(n) returns the smallest positive integer k such that 1/k has exactly n digits after the decimal point in base 6 notation -/
def g (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem stating that g(n) = 2^n * 3^n for all positive integers n -/
theorem g_equals_power_of_two_times_power_of_three (n : ℕ+) :
  g n = 2^(n : ℕ) * 3^(n : ℕ) :=
sorry

/-- The number of positive integer divisors of g(2023) -/
def num_divisors_g_2023 : ℕ :=
  (2023 + 1)^2

/-- Theorem stating that the number of positive integer divisors of g(2023) is 4096576 -/
theorem num_divisors_g_2023_equals_4096576 :
  num_divisors_g_2023 = 4096576 :=
sorry

end g_equals_power_of_two_times_power_of_three_num_divisors_g_2023_equals_4096576_l2650_265074


namespace product_remainder_l2650_265024

theorem product_remainder (a b : ℕ) :
  a % 3 = 1 → b % 3 = 2 → (a * b) % 3 = 2 := by
  sorry

end product_remainder_l2650_265024


namespace paint_intensity_after_replacement_l2650_265087

/-- Calculates the new paint intensity after partial replacement -/
def new_paint_intensity (initial_intensity : ℝ) (replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

/-- Theorem: Given the specified conditions, the new paint intensity is 0.4 (40%) -/
theorem paint_intensity_after_replacement :
  let initial_intensity : ℝ := 0.5
  let replacement_intensity : ℝ := 0.25
  let replacement_fraction : ℝ := 0.4
  new_paint_intensity initial_intensity replacement_intensity replacement_fraction = 0.4 := by
sorry

#eval new_paint_intensity 0.5 0.25 0.4

end paint_intensity_after_replacement_l2650_265087


namespace soccer_lineup_theorem_l2650_265005

/-- The number of ways to choose a soccer lineup -/
def soccer_lineup_count : ℕ := 18 * (Nat.choose 17 4) * (Nat.choose 13 3) * (Nat.choose 10 3)

/-- Theorem stating the number of possible soccer lineups -/
theorem soccer_lineup_theorem : soccer_lineup_count = 147497760 := by
  sorry

end soccer_lineup_theorem_l2650_265005


namespace logic_statement_l2650_265060

theorem logic_statement :
  let p : Prop := 2 + 3 = 5
  let q : Prop := 5 < 4
  (p ∨ q) ∧ ¬(p ∧ q) ∧ p ∧ ¬q := by sorry

end logic_statement_l2650_265060


namespace min_value_of_expression_l2650_265059

/-- Given a line mx + ny + 2 = 0 intersecting a circle (x+3)^2 + (y+1)^2 = 1 at a chord of length 2,
    the minimum value of 1/m + 3/n is 6, where m > 0 and n > 0 -/
theorem min_value_of_expression (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), m*x + n*y + 2 = 0 ∧ (x+3)^2 + (y+1)^2 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), m*x₁ + n*y₁ + 2 = 0 ∧ m*x₂ + n*y₂ + 2 = 0 ∧
                         (x₁+3)^2 + (y₁+1)^2 = 1 ∧ (x₂+3)^2 + (y₂+1)^2 = 1 ∧
                         (x₁-x₂)^2 + (y₁-y₂)^2 = 4) →
  (1/m + 3/n ≥ 6) ∧ (∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 1/m₀ + 3/n₀ = 6) :=
by sorry

end min_value_of_expression_l2650_265059


namespace remainder_of_3n_mod_7_l2650_265052

theorem remainder_of_3n_mod_7 (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 := by
  sorry

end remainder_of_3n_mod_7_l2650_265052


namespace product_45_sum_5_l2650_265084

theorem product_45_sum_5 (v w x y z : ℤ) : 
  v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ v ≠ z ∧
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
  x ≠ y ∧ x ≠ z ∧
  y ≠ z ∧
  v * w * x * y * z = 45 →
  v + w + x + y + z = 5 := by
sorry

end product_45_sum_5_l2650_265084


namespace project_hours_difference_l2650_265094

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 144) 
  (kate_hours pat_hours mark_hours : ℕ) 
  (h_pat_kate : pat_hours = 2 * kate_hours)
  (h_pat_mark : pat_hours * 3 = mark_hours)
  (h_sum : kate_hours + pat_hours + mark_hours = total_hours) :
  mark_hours - kate_hours = 80 := by
sorry

end project_hours_difference_l2650_265094


namespace plane_perpendicular_from_line_l2650_265082

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_from_line
  (α β γ : Plane) (l : Line)
  (distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h1 : perpendicular_line_plane l α)
  (h2 : parallel l β) :
  perpendicular α β :=
sorry

end plane_perpendicular_from_line_l2650_265082


namespace guest_speaker_payment_l2650_265042

theorem guest_speaker_payment (n : ℕ) : 
  (n ≥ 200 ∧ n < 300 ∧ n % 100 ≥ 40 ∧ n % 10 = 4 ∧ n % 13 = 0) → n = 274 :=
by sorry

end guest_speaker_payment_l2650_265042


namespace equation_solutions_l2650_265044

theorem equation_solutions :
  {x : ℝ | x * (2 * x + 1) = 2 * x + 1} = {-1/2, 1} := by
  sorry

end equation_solutions_l2650_265044


namespace trapezoid_segment_equality_l2650_265015

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid -/
structure Trapezoid where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point2D) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p2.x - p1.x) * (p4.y - p3.y)

/-- Checks if a point is on a line segment -/
def isOnSegment (p q r : Point2D) : Prop :=
  q.x <= max p.x r.x ∧ q.x >= min p.x r.x ∧
  q.y <= max p.y r.y ∧ q.y >= min p.y r.y

/-- Represents the intersection of two line segments -/
def intersect (p1 p2 p3 p4 : Point2D) : Option Point2D :=
  sorry -- Implementation omitted for brevity

theorem trapezoid_segment_equality (ABCD : Trapezoid) (M N K L : Point2D) :
  areParallel ABCD.B ABCD.C M N →
  isOnSegment ABCD.A ABCD.B M →
  isOnSegment ABCD.C ABCD.D N →
  intersect M N ABCD.A ABCD.C = some K →
  intersect M N ABCD.B ABCD.D = some L →
  (K.x - M.x)^2 + (K.y - M.y)^2 = (L.x - N.x)^2 + (L.y - N.y)^2 := by
  sorry

end trapezoid_segment_equality_l2650_265015


namespace P_necessary_not_sufficient_for_Q_l2650_265046

theorem P_necessary_not_sufficient_for_Q :
  (∀ x : ℝ, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ (x + 2) * (x - 1) ≥ 0) := by
  sorry

end P_necessary_not_sufficient_for_Q_l2650_265046


namespace function_behavior_l2650_265006

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

-- Define the interval
def interval : Set ℝ := Set.Ioo 2 4

-- Theorem statement
theorem function_behavior (x y : ℝ) (hx : x ∈ interval) (hy : y ∈ interval) :
  (x < 3 ∧ y < 3 → f x > f y) ∧
  (x > 3 ∧ y > 3 → f x < f y) ∧
  (x < 3 ∧ y > 3 → f x > f y) :=
sorry

end function_behavior_l2650_265006


namespace sportswear_problem_l2650_265085

/-- Sportswear Problem -/
theorem sportswear_problem 
  (first_batch_cost : ℝ) 
  (second_batch_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : first_batch_cost = 12000)
  (h2 : second_batch_cost = 26400)
  (h3 : selling_price = 150) :
  ∃ (first_batch_quantity second_batch_quantity : ℕ),
    (second_batch_quantity = 2 * first_batch_quantity) ∧
    (second_batch_cost / second_batch_quantity = first_batch_cost / first_batch_quantity + 10) ∧
    (second_batch_quantity = 240) ∧
    (first_batch_quantity * (selling_price - first_batch_cost / first_batch_quantity) +
     second_batch_quantity * (selling_price - second_batch_cost / second_batch_quantity) = 15600) := by
  sorry

end sportswear_problem_l2650_265085


namespace non_basketball_theater_percentage_l2650_265056

/-- Represents the student body of Maple Town High School -/
structure School where
  total : ℝ
  basketball : ℝ
  theater : ℝ
  both : ℝ

/-- The conditions given in the problem -/
def school_conditions (s : School) : Prop :=
  s.basketball = 0.7 * s.total ∧
  s.theater = 0.4 * s.total ∧
  s.both = 0.2 * s.basketball ∧
  (s.basketball - s.both) = 0.6 * (s.total - s.theater)

/-- The theorem to be proved -/
theorem non_basketball_theater_percentage (s : School) 
  (h : school_conditions s) : 
  (s.theater - s.both) / (s.total - s.basketball) = 0.87 := by
  sorry

end non_basketball_theater_percentage_l2650_265056


namespace quadratic_function_property_l2650_265069

def f (m n x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_function_property (m n : ℝ) :
  (∀ x ∈ Set.Icc 1 5, |f m n x| ≤ 2) →
  (f m n 1 - 2*(f m n 3) + f m n 5 = 8) ∧ (m = -6 ∧ n = 7) := by
  sorry

end quadratic_function_property_l2650_265069


namespace intersection_A_B_l2650_265011

-- Define the set of positive integers
def PositiveInt : Set ℕ := {n : ℕ | n > 0}

-- Define set A
def A : Set ℕ := {x ∈ PositiveInt | x ≤ Real.exp 1}

-- Define set B
def B : Set ℕ := {0, 1, 2, 3}

-- Theorem to prove
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l2650_265011


namespace divide_fraction_by_integer_l2650_265027

theorem divide_fraction_by_integer : (3 : ℚ) / 7 / 4 = 3 / 28 := by
  sorry

end divide_fraction_by_integer_l2650_265027


namespace unique_base_representation_l2650_265000

theorem unique_base_representation : 
  ∃! (x y z b : ℕ), 
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x ≥ 1 ∧
    y < b ∧
    z < b ∧
    b > 1 ∧
    x = 5 ∧
    y = 9 ∧
    z = 11 ∧
    b = 19 := by
  sorry

end unique_base_representation_l2650_265000


namespace butterflies_fraction_l2650_265093

theorem butterflies_fraction (initial : ℕ) (remaining : ℕ) : 
  initial = 9 → remaining = 6 → (initial - remaining : ℚ) / initial = 1/3 := by
  sorry

end butterflies_fraction_l2650_265093


namespace class_size_is_20_l2650_265021

/-- Represents the number of students in a class with specific age distributions. -/
def num_students : ℕ := by sorry

/-- The average age of all students in the class. -/
def average_age : ℝ := 20

/-- The average age of a group of 9 students. -/
def average_age_group1 : ℝ := 11

/-- The average age of a group of 10 students. -/
def average_age_group2 : ℝ := 24

/-- The age of the 20th student. -/
def age_20th_student : ℝ := 61

/-- Theorem stating that the number of students in the class is 20. -/
theorem class_size_is_20 : num_students = 20 := by sorry

end class_size_is_20_l2650_265021


namespace factory_weekly_production_l2650_265072

/-- Calculates the weekly toy production of a factory -/
def weekly_production (days_worked : ℕ) (daily_production : ℕ) : ℕ :=
  days_worked * daily_production

/-- Proves that the factory produces 4340 toys per week -/
theorem factory_weekly_production :
  let days_worked : ℕ := 2
  let daily_production : ℕ := 2170
  weekly_production days_worked daily_production = 4340 := by
sorry

end factory_weekly_production_l2650_265072


namespace percentage_not_liking_basketball_is_52_percent_l2650_265080

/-- Represents the school population and basketball preferences --/
structure School where
  total_students : ℕ
  male_ratio : ℚ
  female_ratio : ℚ
  male_basketball_ratio : ℚ
  female_basketball_ratio : ℚ

/-- Calculates the percentage of students who don't like basketball --/
def percentage_not_liking_basketball (s : School) : ℚ :=
  let male_count := s.total_students * s.male_ratio / (s.male_ratio + s.female_ratio)
  let female_count := s.total_students * s.female_ratio / (s.male_ratio + s.female_ratio)
  let male_playing := male_count * s.male_basketball_ratio
  let female_playing := female_count * s.female_basketball_ratio
  let total_not_playing := s.total_students - (male_playing + female_playing)
  total_not_playing / s.total_students * 100

/-- The main theorem to prove --/
theorem percentage_not_liking_basketball_is_52_percent :
  let s : School := {
    total_students := 1000,
    male_ratio := 3/5,
    female_ratio := 2/5,
    male_basketball_ratio := 2/3,
    female_basketball_ratio := 1/5
  }
  percentage_not_liking_basketball s = 52 := by
  sorry

end percentage_not_liking_basketball_is_52_percent_l2650_265080


namespace fruit_seller_apples_l2650_265029

theorem fruit_seller_apples : ∀ (original : ℕ), 
  (original : ℝ) * (1 - 0.3) = 420 → original = 600 := by
  sorry

end fruit_seller_apples_l2650_265029


namespace blocks_needed_for_wall_l2650_265032

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : Set ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions -/
def calculateBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating that 540 blocks are needed for the given wall -/
theorem blocks_needed_for_wall :
  let wall := WallDimensions.mk 120 9
  let block := BlockDimensions.mk {2, 1.5, 1} 1
  calculateBlocksNeeded wall block = 540 := by
    sorry

end blocks_needed_for_wall_l2650_265032


namespace geometric_sequence_ratio_l2650_265017

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_n being the sum of the first n terms, 
    prove that S_4 / a_2 = -15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 4 / a 2 = -15/2 := by
sorry

end geometric_sequence_ratio_l2650_265017


namespace sandwiches_needed_l2650_265099

theorem sandwiches_needed (total_people children adults : ℕ) 
  (h1 : total_people = 219)
  (h2 : children = 125)
  (h3 : adults = 94)
  (h4 : total_people = children + adults)
  (h5 : children * 4 + adults * 3 = 782) : 
  children * 4 + adults * 3 = 782 := by
  sorry

end sandwiches_needed_l2650_265099


namespace butterfly_cocoon_time_l2650_265034

theorem butterfly_cocoon_time :
  ∀ (cocoon_time larva_time : ℕ),
    cocoon_time + larva_time = 120 →
    larva_time = 3 * cocoon_time →
    cocoon_time = 30 := by
  sorry

end butterfly_cocoon_time_l2650_265034


namespace legs_multiple_of_heads_l2650_265009

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Fin 3 → ℕ
| 0 => 2  -- Ducks
| 1 => 4  -- Cows
| 2 => 4  -- Buffaloes

/-- Represents the number of animals of each type -/
structure AnimalCounts where
  ducks : ℕ
  cows : ℕ
  buffaloes : ℕ
  buffalo_count_eq : buffaloes = 6

/-- Calculates the total number of legs -/
def total_legs (counts : AnimalCounts) : ℕ :=
  counts.ducks * legs_per_animal 0 +
  counts.cows * legs_per_animal 1 +
  counts.buffaloes * legs_per_animal 2

/-- Calculates the total number of heads -/
def total_heads (counts : AnimalCounts) : ℕ :=
  counts.ducks + counts.cows + counts.buffaloes

/-- The theorem to be proved -/
theorem legs_multiple_of_heads (counts : AnimalCounts) :
  ∃ m : ℕ, m ≥ 2 ∧ total_legs counts = m * total_heads counts + 12 ∧
  ∀ k : ℕ, k < m → ¬(total_legs counts = k * total_heads counts + 12) :=
sorry

end legs_multiple_of_heads_l2650_265009


namespace feb_2_is_tuesday_l2650_265050

-- Define the days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to get the day of the week given a number of days before Sunday
def daysBefore (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Saturday
  | 2 => DayOfWeek.Friday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Wednesday
  | 5 => DayOfWeek.Tuesday
  | _ => DayOfWeek.Monday

-- Theorem statement
theorem feb_2_is_tuesday (h : DayOfWeek.Sunday = daysBefore 0) :
  DayOfWeek.Tuesday = daysBefore 12 := by
  sorry


end feb_2_is_tuesday_l2650_265050


namespace min_value_expression_min_value_achievable_l2650_265062

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 :=
by sorry

end min_value_expression_min_value_achievable_l2650_265062


namespace triangle_area_squared_l2650_265055

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circle
def Circle := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 16}

-- Define the conditions
def isInscribed (t : Triangle) : Prop :=
  t.A ∈ Circle ∧ t.B ∈ Circle ∧ t.C ∈ Circle

def angleA (t : Triangle) : ℝ := sorry

def sideDifference (t : Triangle) : ℝ := sorry

def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area_squared (t : Triangle) 
  (h1 : isInscribed t)
  (h2 : angleA t = π / 3)  -- 60 degrees in radians
  (h3 : sideDifference t = 4)
  : (area t)^2 = 192 := by
  sorry

end triangle_area_squared_l2650_265055


namespace system_solution_l2650_265079

theorem system_solution : ∃ (X Y : ℝ), 
  (X + (X + 2*Y) / (X^2 + Y^2) = 2 ∧ 
   Y + (2*X - Y) / (X^2 + Y^2) = 0) ↔ 
  ((X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1)) := by
  sorry

end system_solution_l2650_265079


namespace equation_solutions_l2650_265013

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 5 = 10) ∧
  (∃ x : ℚ, 2 * x + 4 * (2 * x - 3) = 6 - 2 * (x + 1)) :=
by
  constructor
  · use 5
    norm_num
  · use 4/3
    norm_num
    
#check equation_solutions

end equation_solutions_l2650_265013


namespace badge_exchange_l2650_265026

theorem badge_exchange (x : ℕ) : 
  (x + 5 - (24 * (x + 5)) / 100 + (20 * x) / 100 = x - (20 * x) / 100 + (24 * (x + 5)) / 100 - 1) → 
  (x = 45 ∧ x + 5 = 50) :=
by sorry

end badge_exchange_l2650_265026


namespace unique_solution_l2650_265045

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1, v.2.2 * w.1 - v.1 * w.2.2, v.1 * w.2.1 - v.2.1 * w.1)

theorem unique_solution (a b c d e f : ℝ) :
  cross_product (3, a, c) (6, b, d) = (0, 0, 0) ∧
  cross_product (4, b, f) (8, e, d) = (0, 0, 0) →
  (a, b, c, d, e, f) = (1, 2, 1, 2, 4, 1) :=
by sorry

end unique_solution_l2650_265045


namespace farm_animals_l2650_265061

theorem farm_animals (total_animals : ℕ) (num_ducks : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 11)
  (h2 : num_ducks = 6)
  (h3 : total_legs = 32) :
  total_animals - num_ducks = 5 :=
by
  sorry

end farm_animals_l2650_265061


namespace smallest_n_with_square_and_fifth_power_l2650_265063

theorem smallest_n_with_square_and_fifth_power : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x : ℕ), 2 * n = x^2) ∧ 
    (∃ (y : ℕ), 3 * n = y^5)) →
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 2 * m = x^2) → 
    (∃ (y : ℕ), 3 * m = y^5) → 
    m ≥ 2592) ∧
  (∃ (x : ℕ), 2 * 2592 = x^2) ∧ 
  (∃ (y : ℕ), 3 * 2592 = y^5) :=
by sorry

end smallest_n_with_square_and_fifth_power_l2650_265063


namespace count_pairs_theorem_l2650_265090

/-- The number of integer pairs (m, n) satisfying the given inequality -/
def count_pairs : ℕ := 1000

/-- The lower bound for m -/
def m_lower_bound : ℕ := 1

/-- The upper bound for m -/
def m_upper_bound : ℕ := 3000

/-- Predicate to check if a pair (m, n) satisfies the inequality -/
def satisfies_inequality (m n : ℕ) : Prop :=
  (5 : ℝ)^n < (3 : ℝ)^m ∧ (3 : ℝ)^m < (3 : ℝ)^(m+1) ∧ (3 : ℝ)^(m+1) < (5 : ℝ)^(n+1)

theorem count_pairs_theorem :
  ∃ S : Finset (ℕ × ℕ),
    S.card = count_pairs ∧
    (∀ (m n : ℕ), (m, n) ∈ S ↔ 
      m_lower_bound ≤ m ∧ m ≤ m_upper_bound ∧ satisfies_inequality m n) :=
sorry

end count_pairs_theorem_l2650_265090


namespace min_value_implies_a_eq_6_l2650_265039

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1/3 then 3 - Real.sin (a * x) else a * x + Real.log x / Real.log 3

-- State the theorem
theorem min_value_implies_a_eq_6 (a : ℝ) (h1 : a > 0) :
  (∀ x, f a x ≥ 1) ∧ (∃ x, f a x = 1) → a = 6 := by
  sorry

end min_value_implies_a_eq_6_l2650_265039


namespace multiple_properties_l2650_265025

-- Define x and y as integers
variable (x y : ℤ)

-- Define the conditions
def x_multiple_of_4 : Prop := ∃ k : ℤ, x = 4 * k
def y_multiple_of_9 : Prop := ∃ m : ℤ, y = 9 * m

-- Theorem to prove
theorem multiple_properties
  (hx : x_multiple_of_4 x)
  (hy : y_multiple_of_9 y) :
  (∃ n : ℤ, y = 3 * n) ∧ (∃ p : ℤ, x - y = 4 * p) :=
by sorry

end multiple_properties_l2650_265025


namespace problem_solution_l2650_265001

/-- Proposition p: x² - 4ax + 3a² < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: |x - 3| < 1 -/
def q (x : ℝ) : Prop := |x - 3| < 1

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ (4/3 ≤ a ∧ a ≤ 2)) :=
sorry

end problem_solution_l2650_265001


namespace range_of_a_l2650_265078

theorem range_of_a (a : ℝ) : a > 0 →
  (((∀ x y : ℝ, x < y → a^x > a^y) ↔ ¬(∀ x : ℝ, x^2 - 3*a*x + 1 > 0)) ↔
   (2/3 ≤ a ∧ a < 1)) :=
by sorry

end range_of_a_l2650_265078


namespace set_union_problem_l2650_265003

theorem set_union_problem (A B : Set ℝ) (m : ℝ) :
  A = {0, m} →
  B = {0, 2} →
  A ∪ B = {0, 1, 2} →
  m = 1 := by
sorry

end set_union_problem_l2650_265003


namespace arithmetic_geometric_harmonic_mean_sum_of_squares_l2650_265077

theorem arithmetic_geometric_harmonic_mean_sum_of_squares
  (a b c : ℝ)
  (h_arithmetic : (a + b + c) / 3 = 8)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 5)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 3) :
  a^2 + b^2 + c^2 = 326 :=
by sorry

end arithmetic_geometric_harmonic_mean_sum_of_squares_l2650_265077


namespace algebraic_expression_value_l2650_265041

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 4*x - 1 = 0) :
  2*x^4 + 8*x^3 - 4*x^2 - 8*x + 1 = -1 := by
  sorry

end algebraic_expression_value_l2650_265041


namespace congruence_solution_and_sum_l2650_265040

theorem congruence_solution_and_sum (x : ℤ) : 
  (15 * x + 3) % 21 = 9 → 
  ∃ (a m : ℤ), x % m = a ∧ 
                a < m ∧ 
                m = 7 ∧ 
                a = 6 ∧ 
                a + m = 13 := by
  sorry

end congruence_solution_and_sum_l2650_265040


namespace exactly_100_valid_rules_l2650_265030

/-- A type representing a set of 100 cards drawn from an infinite deck of real numbers. -/
def CardSet := Fin 100 → ℝ

/-- A rule for determining the winner between two sets of cards. -/
def WinningRule := CardSet → CardSet → Bool

/-- The condition that the winner only depends on the relative order of the 200 cards. -/
def RelativeOrderCondition (rule : WinningRule) : Prop :=
  ∀ (A B : CardSet) (f : ℝ → ℝ), StrictMono f →
    rule A B = rule (f ∘ A) (f ∘ B)

/-- The condition that if a_i > b_i for all i, then A beats B. -/
def DominanceCondition (rule : WinningRule) : Prop :=
  ∀ (A B : CardSet), (∀ i, A i > B i) → rule A B

/-- The transitivity condition: if A beats B and B beats C, then A beats C. -/
def TransitivityCondition (rule : WinningRule) : Prop :=
  ∀ (A B C : CardSet), rule A B → rule B C → rule A C

/-- A valid rule satisfies all three conditions. -/
def ValidRule (rule : WinningRule) : Prop :=
  RelativeOrderCondition rule ∧ DominanceCondition rule ∧ TransitivityCondition rule

/-- The main theorem: there are exactly 100 valid rules. -/
theorem exactly_100_valid_rules :
  ∃! (rules : Finset WinningRule), rules.card = 100 ∧ ∀ rule ∈ rules, ValidRule rule :=
sorry

end exactly_100_valid_rules_l2650_265030


namespace assembly_line_arrangements_l2650_265037

def num_tasks : ℕ := 5

theorem assembly_line_arrangements :
  (Finset.range num_tasks).card.factorial = 120 := by
  sorry

end assembly_line_arrangements_l2650_265037


namespace not_necessarily_right_triangle_l2650_265036

theorem not_necessarily_right_triangle (a b c : ℝ) : 
  a^2 = 5 → b^2 = 12 → c^2 = 13 → 
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
  sorry

end not_necessarily_right_triangle_l2650_265036
