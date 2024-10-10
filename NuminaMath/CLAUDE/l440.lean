import Mathlib

namespace equation_solution_l440_44052

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
  (-15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1) ↔ (x = 5/4 ∨ x = -2) :=
by sorry

end equation_solution_l440_44052


namespace quadratic_equation_conversion_l440_44012

/-- Given a quadratic equation 2x^2 = 7x - 5, prove that when converted to the general form
    ax^2 + bx + c = 0, the coefficient of the linear term (b) is -7 and the constant term (c) is 5 -/
theorem quadratic_equation_conversion :
  ∃ (a b c : ℝ), (∀ x, 2 * x^2 = 7 * x - 5) →
  (∀ x, a * x^2 + b * x + c = 0) ∧ b = -7 ∧ c = 5 := by
sorry

end quadratic_equation_conversion_l440_44012


namespace point_inside_circle_range_l440_44058

/-- Given that the point (1, 1) is inside the circle (x-a)^2+(y+a)^2=4, 
    prove that the range of a is -1 < a < 1 -/
theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end point_inside_circle_range_l440_44058


namespace fraction_problem_l440_44034

theorem fraction_problem (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, 11 / 7 + x / (2 * q + p) = 2 ∧ x = 6 :=
by sorry

end fraction_problem_l440_44034


namespace y_equation_proof_l440_44022

theorem y_equation_proof (y : ℝ) (h : y + 1/y = 3) : y^6 - 8*y^3 + 4*y = 20*y - 5 := by
  sorry

end y_equation_proof_l440_44022


namespace triangle_pentagon_side_ratio_l440_44010

/-- The ratio of side lengths of an equilateral triangle and a regular pentagon with equal perimeters -/
theorem triangle_pentagon_side_ratio :
  let triangle_perimeter : ℝ := 60
  let pentagon_perimeter : ℝ := 60
  let triangle_side : ℝ := triangle_perimeter / 3
  let pentagon_side : ℝ := pentagon_perimeter / 5
  triangle_side / pentagon_side = 5 / 3 :=
by sorry

end triangle_pentagon_side_ratio_l440_44010


namespace spinsters_and_cats_l440_44062

theorem spinsters_and_cats (spinsters : ℕ) (cats : ℕ) : 
  spinsters = 18 → 
  spinsters * 9 = cats * 2 → 
  cats - spinsters = 63 := by
sorry

end spinsters_and_cats_l440_44062


namespace jorge_simon_age_difference_l440_44067

/-- Represents a person's age at a given year -/
structure AgeAtYear where
  age : ℕ
  year : ℕ

/-- Calculates the age difference between two people -/
def ageDifference (person1 : AgeAtYear) (person2 : AgeAtYear) : ℕ :=
  if person1.year = person2.year then
    if person1.age ≥ person2.age then person1.age - person2.age else person2.age - person1.age
  else
    sorry -- We don't handle different years in this simplified version

theorem jorge_simon_age_difference :
  let jorge2005 : AgeAtYear := { age := 16, year := 2005 }
  let simon2010 : AgeAtYear := { age := 45, year := 2010 }
  let yearDiff : ℕ := simon2010.year - jorge2005.year
  let jorgeAge2010 : ℕ := jorge2005.age + yearDiff
  ageDifference { age := simon2010.age, year := simon2010.year } { age := jorgeAge2010, year := simon2010.year } = 24 := by
  sorry


end jorge_simon_age_difference_l440_44067


namespace rotate_180_proof_l440_44081

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a line 180 degrees around the origin -/
def rotate180 (l : Line) : Line :=
  { a := l.a, b := l.b, c := -l.c }

theorem rotate_180_proof (l : Line) (h : l = { a := 1, b := -1, c := 4 }) :
  rotate180 l = { a := 1, b := -1, c := -4 } := by
  sorry

end rotate_180_proof_l440_44081


namespace equation_solution_difference_l440_44064

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ + 3)^2 / (3 * x₁ + 29) = 2 ∧
  (x₂ + 3)^2 / (3 * x₂ + 29) = 2 ∧
  x₁ ≠ x₂ ∧
  x₂ - x₁ = 14 := by
sorry

end equation_solution_difference_l440_44064


namespace imaginary_part_of_one_plus_i_to_fifth_l440_44085

def complex_i : ℂ := Complex.I

theorem imaginary_part_of_one_plus_i_to_fifth (h : complex_i ^ 2 = -1) :
  Complex.im ((1 : ℂ) + complex_i) ^ 5 = -4 := by sorry

end imaginary_part_of_one_plus_i_to_fifth_l440_44085


namespace tyrone_eric_marbles_l440_44088

/-- Proves that Tyrone gave 10 marbles to Eric -/
theorem tyrone_eric_marbles : ∀ x : ℕ,
  (100 : ℕ) - x = 3 * ((20 : ℕ) + x) → x = 10 := by
  sorry

end tyrone_eric_marbles_l440_44088


namespace secret_spread_exceeds_3000_l440_44049

def secret_spread (n : ℕ) : ℕ := 3^(n-1)

theorem secret_spread_exceeds_3000 :
  ∃ (n : ℕ), n = 9 ∧ secret_spread n > 3000 :=
by sorry

end secret_spread_exceeds_3000_l440_44049


namespace largest_three_digit_number_with_conditions_l440_44084

theorem largest_three_digit_number_with_conditions : ∃ n : ℕ, 
  (n ≤ 999 ∧ n ≥ 100) ∧ 
  (∃ k : ℕ, n = 7 * k + 2) ∧ 
  (∃ m : ℕ, n = 4 * m + 1) ∧ 
  (∀ x : ℕ, (x ≤ 999 ∧ x ≥ 100) → 
    (∃ k : ℕ, x = 7 * k + 2) → 
    (∃ m : ℕ, x = 4 * m + 1) → 
    x ≤ n) ∧
  n = 989 :=
by sorry

end largest_three_digit_number_with_conditions_l440_44084


namespace z_in_fourth_quadrant_l440_44068

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (2 + Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant :
  Complex.re z > 0 ∧ Complex.im z < 0 :=
by sorry

end z_in_fourth_quadrant_l440_44068


namespace peaches_picked_l440_44024

def initial_peaches : ℝ := 34.0
def total_peaches : ℕ := 120

theorem peaches_picked (picked : ℕ) : 
  picked = total_peaches - Int.floor initial_peaches := by sorry

end peaches_picked_l440_44024


namespace y_intercept_of_line_l440_44008

/-- Given a line l with parametric equations x = 4 - 4t and y = -2 + 3t, where t ∈ ℝ,
    the y-intercept of line l is 1. -/
theorem y_intercept_of_line (l : Set (ℝ × ℝ)) : 
  (∀ t : ℝ, (4 - 4*t, -2 + 3*t) ∈ l) → 
  (0, 1) ∈ l := by
  sorry

end y_intercept_of_line_l440_44008


namespace chaz_final_floor_l440_44045

def elevator_problem (start_floor : ℕ) (first_down : ℕ) (second_down : ℕ) : ℕ :=
  start_floor - first_down - second_down

theorem chaz_final_floor :
  elevator_problem 11 2 4 = 5 := by
  sorry

end chaz_final_floor_l440_44045


namespace h_not_prime_l440_44031

/-- The function h(n) as defined in the problem -/
def h (n : ℕ+) : ℤ := n.val^4 - 500 * n.val^2 + 625

/-- Theorem stating that h(n) is not prime for any positive integer n -/
theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by sorry

end h_not_prime_l440_44031


namespace gift_wrapping_combinations_l440_44046

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 5

/-- Whether red ribbon is available -/
def red_ribbon_available : Prop := true

/-- The number of invalid combinations due to supply issue -/
def invalid_combinations : ℕ := 5

theorem gift_wrapping_combinations : 
  wrapping_paper_varieties * ribbon_colors * gift_card_types - invalid_combinations = 195 := by
  sorry

end gift_wrapping_combinations_l440_44046


namespace sum_18_29_in_base3_l440_44018

/-- Converts a natural number from base 10 to base 3 -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def fromBase3 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_29_in_base3 :
  toBase3 (18 + 29) = [1, 2, 0, 2] :=
sorry

end sum_18_29_in_base3_l440_44018


namespace intersection_point_slope_l440_44000

/-- Given three lines in a plane, if two of them intersect at a point on the third line, 
    then the slope of one of the intersecting lines is 4. -/
theorem intersection_point_slope (k : ℝ) : 
  (∃ x y : ℝ, y = -2*x + 4 ∧ y = k*x ∧ y = x + 2) → k = 4 := by
  sorry

end intersection_point_slope_l440_44000


namespace function_properties_l440_44075

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_odd (fun x ↦ f (x + 1/2)))
  (h2 : ∀ x, f (2 - 3*x) = f (3*x)) :
  f (-1/2) = 0 ∧ 
  is_even (fun x ↦ f (x + 2)) ∧ 
  is_odd (fun x ↦ f (x - 1/2)) := by
sorry

end function_properties_l440_44075


namespace smallest_k_divides_k_210_divides_smallest_k_is_210_l440_44017

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides : 
  ∀ k : ℕ, k > 0 → (∀ z : ℂ, f z = 0 → z^k = 1) → k ≥ 210 :=
by sorry

theorem k_210_divides : 
  ∀ z : ℂ, f z = 0 → z^210 = 1 :=
by sorry

theorem smallest_k_is_210 : 
  (∃ k : ℕ, k > 0 ∧ (∀ z : ℂ, f z = 0 → z^k = 1)) ∧
  (∀ k : ℕ, k > 0 → (∀ z : ℂ, f z = 0 → z^k = 1) → k ≥ 210) ∧
  (∀ z : ℂ, f z = 0 → z^210 = 1) :=
by sorry

end smallest_k_divides_k_210_divides_smallest_k_is_210_l440_44017


namespace cubic_value_given_quadratic_l440_44042

theorem cubic_value_given_quadratic (x : ℝ) : 
  x^2 + x - 1 = 0 → x^3 + 2*x^2 + 2005 = 2006 := by
  sorry

end cubic_value_given_quadratic_l440_44042


namespace product_of_numbers_l440_44095

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end product_of_numbers_l440_44095


namespace final_piece_count_l440_44029

/-- Represents the number of pieces after each cut -/
structure PaperCuts where
  initial : Nat
  first_cut : Nat
  second_cut : Nat
  third_cut : Nat
  fourth_cut : Nat

/-- The cutting process as described in the problem -/
def cutting_process : PaperCuts :=
  { initial := 1
  , first_cut := 10
  , second_cut := 19
  , third_cut := 28
  , fourth_cut := 37 }

/-- Theorem stating that the final number of pieces is 37 -/
theorem final_piece_count :
  (cutting_process.fourth_cut = 37) := by sorry

end final_piece_count_l440_44029


namespace fruit_tree_problem_l440_44079

theorem fruit_tree_problem (initial_apples : ℕ) (pick_ratio : ℚ) : 
  initial_apples = 180 →
  pick_ratio = 3 / 5 →
  ∃ (initial_plums : ℕ),
    initial_plums * 3 = initial_apples ∧
    (initial_apples + initial_plums) * (1 - pick_ratio) = 96 := by
  sorry

end fruit_tree_problem_l440_44079


namespace line_points_product_l440_44003

theorem line_points_product (x y : ℝ) : 
  (∃ k : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ k ↔ p.2 = (1/4) * p.1) ∧ 
    (x, 8) ∈ k ∧ 
    (20, y) ∈ k ∧ 
    x * y = 160) → 
  y = 5 := by
sorry

end line_points_product_l440_44003


namespace segment_length_parallel_to_x_axis_l440_44013

/-- Given two points M and N, where M's coordinates depend on parameter a,
    and MN is parallel to the x-axis, prove that the length of MN is 6. -/
theorem segment_length_parallel_to_x_axis 
  (a : ℝ) 
  (M : ℝ × ℝ := (a + 3, a - 4))
  (N : ℝ × ℝ := (-1, -2))
  (h_parallel : M.2 = N.2) : 
  abs (M.1 - N.1) = 6 := by
sorry

end segment_length_parallel_to_x_axis_l440_44013


namespace problem_solution_l440_44051

theorem problem_solution (x y : ℝ) (hx : x = 1 - Real.sqrt 2) (hy : y = 1 + Real.sqrt 2) : 
  x^2 + 3*x*y + y^2 = 3 ∧ y/x - x/y = -4 * Real.sqrt 2 := by
  sorry

end problem_solution_l440_44051


namespace casper_candy_problem_l440_44006

theorem casper_candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_after_eating := day1_remaining - (day1_remaining / 4)
  let day2_remaining := day2_after_eating + 5 - 5
  day2_remaining = 10 → initial_candies = 58 := by
  sorry

end casper_candy_problem_l440_44006


namespace product_equality_l440_44087

theorem product_equality (square : ℕ) : 
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * square → square = 50 := by
sorry

end product_equality_l440_44087


namespace x_minus_reciprocal_equals_one_l440_44055

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the fractional part function
noncomputable def fracPart (x : ℝ) : ℝ :=
  x - intPart x

-- Main theorem
theorem x_minus_reciprocal_equals_one (x : ℝ) 
  (h1 : x > 0)
  (h2 : (intPart x : ℝ)^2 = x * fracPart x) : 
  x - 1/x = 1 := by
  sorry

end x_minus_reciprocal_equals_one_l440_44055


namespace mixed_nuts_cost_per_serving_l440_44098

/-- Calculates the cost per serving of mixed nuts in cents -/
def cost_per_serving (bag_cost : ℚ) (bag_content : ℚ) (coupon_value : ℚ) (serving_size : ℚ) : ℚ :=
  ((bag_cost - coupon_value) / bag_content) * serving_size * 100

/-- Theorem: The cost per serving of mixed nuts is 50 cents -/
theorem mixed_nuts_cost_per_serving :
  cost_per_serving 25 40 5 1 = 50 := by
  sorry

#eval cost_per_serving 25 40 5 1

end mixed_nuts_cost_per_serving_l440_44098


namespace parabola_m_value_l440_44094

/-- A parabola with equation x² = my and a point M(x₀, -3) on it. -/
structure Parabola where
  m : ℝ
  x₀ : ℝ
  eq : x₀^2 = m * (-3)

/-- The distance from a point to the focus of the parabola. -/
def distance_to_focus (p : Parabola) : ℝ := 5

/-- Theorem: If a point M(x₀, -3) on the parabola x² = my has a distance of 5 to the focus, then m = -8. -/
theorem parabola_m_value (p : Parabola) (h : distance_to_focus p = 5) : p.m = -8 := by
  sorry

end parabola_m_value_l440_44094


namespace blake_change_l440_44039

theorem blake_change (oranges apples mangoes initial : ℕ) 
  (h_oranges : oranges = 40)
  (h_apples : apples = 50)
  (h_mangoes : mangoes = 60)
  (h_initial : initial = 300) :
  initial - (oranges + apples + mangoes) = 150 := by
  sorry

end blake_change_l440_44039


namespace factorization_problem_1_factorization_problem_2_l440_44082

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3*x + 3*y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (a x : ℝ) :
  2*a*(x^2 + 1)^2 - 8*a*x^2 = 2*a*(x - 1)^2*(x + 1)^2 := by sorry

end factorization_problem_1_factorization_problem_2_l440_44082


namespace equation_solution_inequalities_solution_l440_44065

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℝ, x * (x - 4) = x - 6 ↔ x = 2 ∨ x = 3 := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∀ x : ℝ, (4 * x - 2 ≥ 3 * (x - 1) ∧ (x - 5) / 2 + 1 > x - 3) ↔ -1 ≤ x ∧ x < 3 := by sorry

end equation_solution_inequalities_solution_l440_44065


namespace f_neg_one_eq_neg_one_l440_44093

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem f_neg_one_eq_neg_one
  (h_odd : IsOdd f)
  (h_f_one : f 1 = 1) :
  f (-1) = -1 :=
sorry

end f_neg_one_eq_neg_one_l440_44093


namespace line_equation_from_circle_intersection_l440_44032

/-- Given a circle and a line intersecting it, prove the equation of the line. -/
theorem line_equation_from_circle_intersection (a : ℝ) (h_a : a < 3) :
  let circle := fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + a = 0
  let midpoint := (-2, 3)
  ∃ (A B : ℝ × ℝ),
    circle A.1 A.2 ∧
    circle B.1 B.2 ∧
    (A.1 + B.1) / 2 = midpoint.1 ∧
    (A.2 + B.2) / 2 = midpoint.2 →
    ∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b ↔ x - y + 5 = 0 :=
by sorry

end line_equation_from_circle_intersection_l440_44032


namespace larger_solution_of_quadratic_l440_44036

theorem larger_solution_of_quadratic (x : ℝ) : 
  (2 * x^2 - 14 * x - 84 = 0) → (∃ y : ℝ, 2 * y^2 - 14 * y - 84 = 0 ∧ y ≠ x) → 
  (x = 14 ∨ x = -3) → (x = 14 ∨ (x = -3 ∧ 14 > x)) :=
sorry

end larger_solution_of_quadratic_l440_44036


namespace quadratic_transformation_l440_44023

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := x^2 - 16*x + 15

/-- The transformed quadratic function -/
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

theorem quadratic_transformation :
  ∃ b c : ℝ, (∀ x : ℝ, f x = g x b c) ∧ b + c = -57 := by sorry

end quadratic_transformation_l440_44023


namespace aaron_reading_challenge_l440_44071

theorem aaron_reading_challenge (average_pages : ℕ) (total_days : ℕ) (day1 day2 day3 day4 day5 : ℕ) :
  average_pages = 15 →
  total_days = 6 →
  day1 = 18 →
  day2 = 12 →
  day3 = 23 →
  day4 = 10 →
  day5 = 17 →
  ∃ (day6 : ℕ), (day1 + day2 + day3 + day4 + day5 + day6) / total_days = average_pages ∧ day6 = 10 :=
by sorry

end aaron_reading_challenge_l440_44071


namespace divisibility_problem_l440_44073

theorem divisibility_problem (a b n : ℤ) : 
  n = 10 * a + b → (17 ∣ (a - 5 * b)) → (17 ∣ n) := by sorry

end divisibility_problem_l440_44073


namespace f_decreasing_intervals_l440_44069

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem f_decreasing_intervals (ω : ℝ) (h_ω : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π + π / 6) (k * π + π / 3)) := by
  sorry

end f_decreasing_intervals_l440_44069


namespace sum_difference_is_thirteen_l440_44047

def star_list : List Nat := List.range 30

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem sum_difference_is_thirteen :
  star_list.sum - emilio_list.sum = 13 := by
  sorry

end sum_difference_is_thirteen_l440_44047


namespace a_range_l440_44083

-- Define proposition P
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Theorem statement
theorem a_range (a : ℝ) : P a ∧ ¬(Q a) ↔ 1 < a ∧ a < 4 := by
  sorry

end a_range_l440_44083


namespace function_inequality_l440_44004

theorem function_inequality (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 8/5) :
  ∀ x : ℝ, a ≤ x ∧ x ≤ 2*a - 1 → |x + a| + |2*x - 3| ≤ |x + 3| := by
sorry

end function_inequality_l440_44004


namespace product_equality_l440_44048

theorem product_equality : ∃ x : ℝ, 469138 * x = 4690910862 ∧ x = 10000.1 := by
  sorry

end product_equality_l440_44048


namespace bottle_cap_weight_l440_44092

theorem bottle_cap_weight (caps_per_ounce : ℕ) (total_caps : ℕ) (total_weight : ℕ) :
  caps_per_ounce = 7 →
  total_caps = 2016 →
  total_weight = total_caps / caps_per_ounce →
  total_weight = 288 :=
by sorry

end bottle_cap_weight_l440_44092


namespace tailor_time_calculation_l440_44016

-- Define the time ratios
def shirt_time : ℚ := 1
def pants_time : ℚ := 2
def jacket_time : ℚ := 3

-- Define the reference quantities
def ref_shirts : ℕ := 2
def ref_pants : ℕ := 3
def ref_jackets : ℕ := 4
def ref_total_time : ℚ := 10

-- Define the quantities to calculate
def calc_shirts : ℕ := 14
def calc_pants : ℕ := 10
def calc_jackets : ℕ := 2

-- Theorem statement
theorem tailor_time_calculation :
  let base_time := ref_total_time / (ref_shirts * shirt_time + ref_pants * pants_time + ref_jackets * jacket_time)
  calc_shirts * (base_time * shirt_time) + calc_pants * (base_time * pants_time) + calc_jackets * (base_time * jacket_time) = 20 := by
  sorry

end tailor_time_calculation_l440_44016


namespace parents_average_age_l440_44005

theorem parents_average_age
  (num_grandparents num_parents num_grandchildren : ℕ)
  (avg_age_grandparents avg_age_grandchildren avg_age_family : ℚ)
  (h1 : num_grandparents = 2)
  (h2 : num_parents = 2)
  (h3 : num_grandchildren = 3)
  (h4 : avg_age_grandparents = 64)
  (h5 : avg_age_grandchildren = 6)
  (h6 : avg_age_family = 32)
  (h7 : (num_grandparents + num_parents + num_grandchildren : ℚ) * avg_age_family =
        num_grandparents * avg_age_grandparents +
        num_parents * (num_grandparents * avg_age_grandparents + num_parents * avg_age_family + num_grandchildren * avg_age_grandchildren - (num_grandparents + num_parents + num_grandchildren) * avg_age_family) / num_parents +
        num_grandchildren * avg_age_grandchildren) :
  (num_grandparents * avg_age_grandparents + num_parents * avg_age_family + num_grandchildren * avg_age_grandchildren - (num_grandparents + num_parents + num_grandchildren) * avg_age_family) / num_parents = 39 :=
sorry

end parents_average_age_l440_44005


namespace floor_plus_self_unique_solution_l440_44076

theorem floor_plus_self_unique_solution (r : ℝ) : 
  (⌊r⌋ : ℝ) + r = 16.5 ↔ r = 8.5 := by
  sorry

end floor_plus_self_unique_solution_l440_44076


namespace seed_to_sprout_probability_is_correct_l440_44091

/-- The germination rate of a batch of seeds -/
def germination_rate : ℝ := 0.9

/-- The survival rate of sprouts after germination -/
def survival_rate : ℝ := 0.8

/-- The probability that a randomly selected seed will grow into a sprout -/
def seed_to_sprout_probability : ℝ := germination_rate * survival_rate

/-- Theorem: The probability that a randomly selected seed will grow into a sprout is 0.72 -/
theorem seed_to_sprout_probability_is_correct : seed_to_sprout_probability = 0.72 := by
  sorry

end seed_to_sprout_probability_is_correct_l440_44091


namespace min_perimeter_triangle_l440_44037

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (O P A B P1 P2 : Plane)

-- Define the angle
variable (angle : Plane → Plane → Plane → Prop)

-- Define the property of being inside an angle
variable (inside_angle : Plane → Plane → Plane → Plane → Prop)

-- Define the property of a point being on a line
variable (on_line : Plane → Plane → Plane → Prop)

-- Define the reflection of a point over a line
variable (reflect : Plane → Plane → Plane → Plane)

-- Define the perimeter of a triangle
variable (perimeter : Plane → Plane → Plane → ℝ)

-- Define the theorem
theorem min_perimeter_triangle 
  (h_acute : angle O A B)
  (h_inside : inside_angle O A B P)
  (h_P1 : P1 = reflect O A P)
  (h_P2 : P2 = reflect O B P)
  (h_A_on_side : on_line O A A)
  (h_B_on_side : on_line O B B)
  (h_A_on_P1P2 : on_line P1 P2 A)
  (h_B_on_P1P2 : on_line P1 P2 B) :
  ∀ A' B', on_line O A A' → on_line O B B' → 
    perimeter P A B ≤ perimeter P A' B' :=
sorry

end min_perimeter_triangle_l440_44037


namespace molecular_weight_4_moles_BaBr2_l440_44041

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Br : ℝ := 79.90

-- Define molecular weight of BaBr2
def molecular_weight_BaBr2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Br

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem molecular_weight_4_moles_BaBr2 :
  moles * molecular_weight_BaBr2 = 1188.52 := by sorry

end molecular_weight_4_moles_BaBr2_l440_44041


namespace f_geq_two_range_of_x_l440_44096

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem 1: f(x) ≥ 2 for all real x
theorem f_geq_two (x : ℝ) : f x ≥ 2 := by
  sorry

-- Theorem 2: If f(x) ≥ (|2b+1| - |1-b|) / |b| for all non-zero real b,
-- then x ≤ -1.5 or x ≥ 1.5
theorem range_of_x (x : ℝ) 
  (h : ∀ b : ℝ, b ≠ 0 → f x ≥ (|2*b + 1| - |1 - b|) / |b|) : 
  x ≤ -1.5 ∨ x ≥ 1.5 := by
  sorry

end f_geq_two_range_of_x_l440_44096


namespace fourth_side_length_l440_44053

/-- A quadrilateral inscribed in a circle with radius 200√2, where three sides have length 200 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of three sides of the quadrilateral -/
  side_length : ℝ
  /-- The fourth side of the quadrilateral -/
  fourth_side : ℝ
  /-- Assertion that the radius is 200√2 -/
  radius_eq : radius = 200 * Real.sqrt 2
  /-- Assertion that three sides have length 200 -/
  three_sides_eq : side_length = 200

/-- Theorem stating that the fourth side of the quadrilateral has length 500 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.fourth_side = 500 := by
  sorry

#check fourth_side_length

end fourth_side_length_l440_44053


namespace pau_total_chicken_l440_44001

def kobe_order : ℕ := 5

def pau_order (kobe : ℕ) : ℕ := 2 * kobe

def total_pau_order (kobe : ℕ) : ℕ := 2 * pau_order kobe

theorem pau_total_chicken :
  total_pau_order kobe_order = 20 :=
by
  sorry

end pau_total_chicken_l440_44001


namespace triangle_angle_sum_l440_44007

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 75) (h2 : B = 40) : C = 65 := by
  sorry

end triangle_angle_sum_l440_44007


namespace expression_simplification_l440_44009

theorem expression_simplification (x y : ℚ) (hx : x = 3) (hy : y = -1/3) :
  3 * x * y^2 - (x * y - 2 * (2 * x * y - 3/2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y = -3 := by
  sorry

end expression_simplification_l440_44009


namespace positive_quadratic_intervals_l440_44021

theorem positive_quadratic_intervals (x : ℝ) : 
  (x - 2) * (x + 3) > 0 ↔ x < -3 ∨ x > 2 := by sorry

end positive_quadratic_intervals_l440_44021


namespace football_field_area_is_9600_l440_44025

/-- The total area of a football field in square yards -/
def football_field_area : ℝ := 9600

/-- The total amount of fertilizer used on the entire field in pounds -/
def total_fertilizer : ℝ := 1200

/-- The amount of fertilizer used on a part of the field in pounds -/
def partial_fertilizer : ℝ := 700

/-- The area covered by the partial fertilizer in square yards -/
def partial_area : ℝ := 5600

/-- Theorem stating that the football field area is 9600 square yards -/
theorem football_field_area_is_9600 : 
  football_field_area = (total_fertilizer * partial_area) / partial_fertilizer := by
  sorry

end football_field_area_is_9600_l440_44025


namespace mental_competition_result_l440_44038

/-- Represents the number of students who correctly answered each problem -/
structure ProblemCounts where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the scores for each problem -/
def problem_scores : Fin 3 → ℕ
  | 0 => 20  -- Problem a
  | 1 => 25  -- Problem b
  | 2 => 25  -- Problem c
  | _ => 0   -- This case should never occur due to Fin 3

theorem mental_competition_result 
  (counts : ProblemCounts)
  (h1 : counts.a + counts.b = 29)
  (h2 : counts.a + counts.c = 25)
  (h3 : counts.b + counts.c = 20)
  (h4 : counts.a + counts.b + counts.c ≥ 1 + 3 * 15 + 1)  -- At least one correct + 15 with two correct + one with all correct
  (h5 : counts.a + counts.b + counts.c - (3 + 2 * 15) ≥ 0)  -- Non-negative number of students with only one correct
  : 
  (counts.a + counts.b + counts.c - (3 + 2 * 15) = 4) ∧  -- 4 students answered only one question correctly
  (((counts.a * problem_scores 0) + (counts.b * problem_scores 1) + (counts.c * problem_scores 2) + 70) / (counts.a + counts.b + counts.c - (3 + 2 * 15) + 15 + 1) = 42) -- Average score is 42
  := by sorry


end mental_competition_result_l440_44038


namespace log_sum_equals_two_l440_44063

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 2 + Real.log 0.04 / Real.log 2 = 2 := by
  sorry

end log_sum_equals_two_l440_44063


namespace bus_departure_interval_l440_44090

/-- Represents the time interval between bus departures -/
def bus_interval (total_time minutes_per_hour : ℕ) (num_buses : ℕ) : ℚ :=
  total_time / ((num_buses - 1) * minutes_per_hour)

theorem bus_departure_interval (total_time minutes_per_hour : ℕ) (num_buses : ℕ) 
  (h1 : total_time = 60) 
  (h2 : minutes_per_hour = 60)
  (h3 : num_buses = 11) :
  bus_interval total_time minutes_per_hour num_buses = 6 := by
  sorry

#eval bus_interval 60 60 11

end bus_departure_interval_l440_44090


namespace quadratic_root_m_value_l440_44027

theorem quadratic_root_m_value :
  ∀ m : ℝ, ((-1 : ℝ)^2 + m * (-1) + 1 = 0) → m = 2 := by
  sorry

end quadratic_root_m_value_l440_44027


namespace zuca_win_probability_l440_44061

/-- The Game played on a regular hexagon --/
structure TheGame where
  /-- Number of vertices in the hexagon --/
  vertices : Nat
  /-- Number of players --/
  players : Nat
  /-- Probability of Bamal and Halvan moving to adjacent vertices --/
  prob_adjacent : ℚ
  /-- Probability of Zuca moving to adjacent or opposite vertices --/
  prob_zuca_move : ℚ

/-- The specific instance of The Game as described in the problem --/
def gameInstance : TheGame :=
  { vertices := 6
  , players := 3
  , prob_adjacent := 1/2
  , prob_zuca_move := 1/3 }

/-- The probability that Zuca hasn't lost when The Game ends --/
def probZucaWins (g : TheGame) : ℚ :=
  29/90

/-- Theorem stating that the probability of Zuca not losing is 29/90 --/
theorem zuca_win_probability (g : TheGame) :
  g = gameInstance → probZucaWins g = 29/90 := by
  sorry

end zuca_win_probability_l440_44061


namespace probability_of_pirate_letter_l440_44057

def probability_letters : Finset Char := {'P', 'R', 'O', 'B', 'A', 'I', 'L', 'T', 'Y'}
def pirate_letters : Finset Char := {'P', 'I', 'R', 'A', 'T', 'E'}

def total_tiles : ℕ := 11

theorem probability_of_pirate_letter :
  (probability_letters ∩ pirate_letters).card / total_tiles = 5 / 11 := by
  sorry

end probability_of_pirate_letter_l440_44057


namespace sylvester_theorem_l440_44050

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define when a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define when points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  ∃ l : Line, pointOnLine p1 l ∧ pointOnLine p2 l ∧ pointOnLine p3 l

-- Define when a set of points is not all collinear
def notAllCollinear (E : Set Point) : Prop :=
  ∃ p1 p2 p3 : Point, p1 ∈ E ∧ p2 ∈ E ∧ p3 ∈ E ∧ ¬collinear p1 p2 p3

-- Sylvester's theorem statement
theorem sylvester_theorem (E : Set Point) (h1 : E.Finite) (h2 : notAllCollinear E) :
  ∃ l : Line, ∃ p1 p2 : Point, p1 ∈ E ∧ p2 ∈ E ∧ p1 ≠ p2 ∧
    pointOnLine p1 l ∧ pointOnLine p2 l ∧
    ∀ p3 : Point, p3 ∈ E → pointOnLine p3 l → (p3 = p1 ∨ p3 = p2) :=
  sorry

end sylvester_theorem_l440_44050


namespace problem_solution_l440_44040

theorem problem_solution (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : 
  x^2 + y^2 = 697 ∧ x + y = Real.sqrt 769 := by
  sorry

end problem_solution_l440_44040


namespace common_chord_of_circles_l440_44099

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 2*y - 13 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 16*y - 25 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 4*x + 3*y - 2 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, (C₁ x y ∧ C₂ x y) → common_chord x y :=
by sorry

end common_chord_of_circles_l440_44099


namespace original_ratio_proof_l440_44020

theorem original_ratio_proof (x y : ℕ+) (h1 : y = 24) (h2 : (x + 6 : ℚ) / y = 1 / 2) : 
  (x : ℚ) / y = 1 / 4 := by
  sorry

end original_ratio_proof_l440_44020


namespace cistern_wet_surface_area_l440_44035

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The wet surface area of a cistern with given dimensions is 49 square meters -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 6 4 1.25 = 49 := by sorry

end cistern_wet_surface_area_l440_44035


namespace odd_digits_in_base4_350_l440_44044

-- Define a function to convert a number from base 10 to base 4
def toBase4 (n : ℕ) : List ℕ := sorry

-- Define a function to count odd digits in a list of digits
def countOddDigits (digits : List ℕ) : ℕ := sorry

-- Theorem statement
theorem odd_digits_in_base4_350 :
  countOddDigits (toBase4 350) = 4 := by sorry

end odd_digits_in_base4_350_l440_44044


namespace f_max_min_on_interval_l440_44070

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 0]
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

-- Theorem stating the maximum and minimum values of f(x) on the given interval
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧ 
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧ 
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -18 :=
sorry

end f_max_min_on_interval_l440_44070


namespace vectors_perpendicular_l440_44019

def vector_angle (u v : ℝ × ℝ) : ℝ := sorry

theorem vectors_perpendicular : 
  let u : ℝ × ℝ := (3, -4)
  let v : ℝ × ℝ := (4, 3)
  vector_angle u v = 90 := by sorry

end vectors_perpendicular_l440_44019


namespace christen_peeled_23_potatoes_l440_44066

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  totalPotatoes : ℕ
  homerRate : ℕ
  christenInitialRate : ℕ
  christenFinalRate : ℕ
  homerAloneTime : ℕ
  workTogetherTime : ℕ
  christenBreakTime : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- The theorem stating that Christen peeled 23 potatoes -/
theorem christen_peeled_23_potatoes :
  let scenario := PotatoPeeling.mk 60 4 6 4 5 3 2
  christenPeeledPotatoes scenario = 23 := by
  sorry

end christen_peeled_23_potatoes_l440_44066


namespace cyclist_speed_calculation_l440_44080

/-- Given two cyclists, Joann and Fran, this theorem proves the required speed for Fran
    to cover the same distance as Joann in a different amount of time. -/
theorem cyclist_speed_calculation (joann_speed joann_time fran_time : ℝ) 
    (hjs : joann_speed = 15) 
    (hjt : joann_time = 4)
    (hft : fran_time = 5) : 
  joann_speed * joann_time / fran_time = 12 := by
  sorry

#check cyclist_speed_calculation

end cyclist_speed_calculation_l440_44080


namespace company_workers_count_l440_44002

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  workers_per_lead : ℕ
  leads_per_supervisor : ℕ
  num_supervisors : ℕ

/-- Calculates the number of workers in a company given its hierarchical structure -/
def calculate_workers (ch : CompanyHierarchy) : ℕ :=
  ch.num_supervisors * ch.leads_per_supervisor * ch.workers_per_lead

/-- Theorem stating that a company with the given hierarchical structure and 13 supervisors has 390 workers -/
theorem company_workers_count :
  let ch : CompanyHierarchy := {
    workers_per_lead := 10,
    leads_per_supervisor := 3,
    num_supervisors := 13
  }
  calculate_workers ch = 390 := by sorry

end company_workers_count_l440_44002


namespace overlapping_triangles_area_l440_44054

/-- The area common to two overlapping right triangles -/
theorem overlapping_triangles_area :
  let triangle1_hypotenuse : ℝ := 10
  let triangle1_angle1 : ℝ := 30 * π / 180
  let triangle1_angle2 : ℝ := 60 * π / 180
  let triangle2_hypotenuse : ℝ := 15
  let triangle2_angle1 : ℝ := 45 * π / 180
  let triangle2_angle2 : ℝ := 45 * π / 180
  let overlap_length : ℝ := 5
  ∃ (common_area : ℝ), common_area = (25 * Real.sqrt 3) / 8 := by
  sorry

end overlapping_triangles_area_l440_44054


namespace james_payment_is_correct_l440_44026

/-- Calculates James's payment for stickers given the number of packs, stickers per pack,
    cost per sticker, discount rate, tax rate, and friend's contribution ratio. -/
def james_payment (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ)
                  (discount_rate : ℚ) (tax_rate : ℚ) (friend_contribution_ratio : ℚ) : ℚ :=
  let total_cost := packs * stickers_per_pack * cost_per_sticker
  let discounted_cost := total_cost * (1 - discount_rate)
  let taxed_cost := discounted_cost * (1 + tax_rate)
  taxed_cost * (1 - friend_contribution_ratio)

/-- Proves that James's payment is $36.38 given the specific conditions of the problem. -/
theorem james_payment_is_correct :
  james_payment 8 40 (25 / 100) (15 / 100) (7 / 100) (1 / 2) = 3638 / 100 := by
  sorry

end james_payment_is_correct_l440_44026


namespace binary_addition_subtraction_l440_44014

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

theorem binary_addition_subtraction :
  let a := [true, true, false, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, false, false, true] -- 1001₂
  let d := [true, false, true, false] -- 1010₂
  let result := [true, false, true, true, true] -- 10111₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d = binary_to_nat result := by
  sorry

end binary_addition_subtraction_l440_44014


namespace no_real_solutions_l440_44078

theorem no_real_solutions : ¬ ∃ x : ℝ, (5*x)/(x^2 + 2*x + 4) + (6*x)/(x^2 - 4*x + 4) = -1 := by
  sorry

end no_real_solutions_l440_44078


namespace jeremy_scrabble_score_l440_44074

/-- Calculates the score for a three-letter word in Scrabble with given letter values and a triple word score -/
def scrabble_score (first_letter_value : ℕ) (middle_letter_value : ℕ) (last_letter_value : ℕ) : ℕ :=
  3 * (first_letter_value + middle_letter_value + last_letter_value)

/-- Theorem: The score for Jeremy's word is 30 points -/
theorem jeremy_scrabble_score :
  scrabble_score 1 8 1 = 30 := by
  sorry

end jeremy_scrabble_score_l440_44074


namespace equation_solution_l440_44086

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (7*x)^5 = (14*x)^4 ∧ x = 16/7 := by sorry

end equation_solution_l440_44086


namespace exists_power_two_minus_one_divisible_by_n_l440_44033

theorem exists_power_two_minus_one_divisible_by_n (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ (n ∣ 2^k - 1) :=
sorry

end exists_power_two_minus_one_divisible_by_n_l440_44033


namespace triangle_properties_l440_44011

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  -- Area condition
  3 * Real.sin t.A = (1/2) * t.b * t.c * Real.sin t.A ∧
  -- Perimeter condition
  t.a + t.b + t.c = 4 * (Real.sqrt 2 + 1) ∧
  -- Sine condition
  Real.sin t.B + Real.sin t.C = Real.sqrt 2 * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.a = 4 ∧ 
  Real.cos t.A = 1/3 ∧ 
  Real.cos (2 * t.A - π/3) = (4 * Real.sqrt 6 - 7) / 18 := by
  sorry


end triangle_properties_l440_44011


namespace prob_one_heads_is_half_l440_44028

/-- A coin toss outcome -/
inductive CoinToss
| Heads
| Tails

/-- Result of two successive coin tosses -/
def TwoTosses := (CoinToss × CoinToss)

/-- All possible outcomes of two successive coin tosses -/
def allOutcomes : Finset TwoTosses := sorry

/-- Outcomes with exactly one heads -/
def oneHeadsOutcomes : Finset TwoTosses := sorry

/-- Probability of an event in a finite sample space -/
def probability (event : Finset TwoTosses) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_one_heads_is_half :
  probability oneHeadsOutcomes = 1 / 2 := by sorry

end prob_one_heads_is_half_l440_44028


namespace expression_simplification_l440_44059

theorem expression_simplification (y : ℝ) :
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) =
  8 * y^3 - 12 * y^2 + 20 * y - 24 := by
  sorry

end expression_simplification_l440_44059


namespace factors_of_72_l440_44043

theorem factors_of_72 : Nat.card (Nat.divisors 72) = 12 := by
  sorry

end factors_of_72_l440_44043


namespace journey_equation_correct_l440_44056

/-- Represents a journey with a stop in between -/
structure Journey where
  preBrakeSpeed : ℝ
  postBrakeSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  brakeTime : ℝ

/-- Checks if the given equation correctly represents the journey -/
def isCorrectEquation (j : Journey) (equation : ℝ → Prop) : Prop :=
  ∀ t, equation t ↔ 
    j.preBrakeSpeed * t + j.postBrakeSpeed * (j.totalTime - j.brakeTime - t) = j.totalDistance

theorem journey_equation_correct (j : Journey) 
    (h1 : j.preBrakeSpeed = 60)
    (h2 : j.postBrakeSpeed = 80)
    (h3 : j.totalDistance = 220)
    (h4 : j.totalTime = 4)
    (h5 : j.brakeTime = 2/3) :
    isCorrectEquation j (fun t ↦ 60 * t + 80 * (10/3 - t) = 220) := by
  sorry

#check journey_equation_correct

end journey_equation_correct_l440_44056


namespace cube_square_equation_solution_l440_44015

theorem cube_square_equation_solution :
  2^3 - 7 = 3^2 + (-8) := by sorry

end cube_square_equation_solution_l440_44015


namespace trig_expression_simplification_l440_44060

theorem trig_expression_simplification :
  (Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) =
  (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.sqrt 3) :=
by sorry

end trig_expression_simplification_l440_44060


namespace partition_twelve_possible_partition_twentytwo_impossible_l440_44097

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def valid_partition (s : Set ℕ) (n : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    partition.length = n ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ s ∧ pair.2 ∈ s) ∧
    (∀ x : ℕ, x ∈ s → ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (x = pair.1 ∨ x = pair.2)) ∧
    (∀ (pair1 pair2 : ℕ × ℕ), pair1 ∈ partition → pair2 ∈ partition → pair1 ≠ pair2 →
      is_prime (pair1.1 + pair1.2) ∧
      is_prime (pair2.1 + pair2.2) ∧
      pair1.1 + pair1.2 ≠ pair2.1 + pair2.2)

theorem partition_twelve_possible : 
  valid_partition (Finset.range 12).toSet 6 := sorry

theorem partition_twentytwo_impossible : 
  ¬ valid_partition (Finset.range 22).toSet 11 := sorry

end partition_twelve_possible_partition_twentytwo_impossible_l440_44097


namespace smallest_positive_integer_satisfying_congruences_l440_44089

theorem smallest_positive_integer_satisfying_congruences : ∃! b : ℕ+, 
  (b : ℤ) % 3 = 2 ∧ 
  (b : ℤ) % 4 = 3 ∧ 
  (b : ℤ) % 5 = 4 ∧ 
  (b : ℤ) % 7 = 6 ∧ 
  ∀ c : ℕ+, 
    ((c : ℤ) % 3 = 2 ∧ 
     (c : ℤ) % 4 = 3 ∧ 
     (c : ℤ) % 5 = 4 ∧ 
     (c : ℤ) % 7 = 6) → 
    b ≤ c := by
  sorry

end smallest_positive_integer_satisfying_congruences_l440_44089


namespace rearrangement_impossibility_l440_44077

theorem rearrangement_impossibility : ¬ ∃ (arrangement : Fin 3972 → ℕ),
  (∀ i : Fin 1986, ∃ (m n : Fin 3972), m < n ∧ 
    arrangement m = i.val + 1 ∧ 
    arrangement n = i.val + 1 ∧ 
    n.val - m.val - 1 = i.val) ∧
  (∀ k : Fin 3972, ∃ i : Fin 1986, arrangement k = i.val + 1) :=
sorry

end rearrangement_impossibility_l440_44077


namespace pauls_money_duration_l440_44030

/-- Given Paul's earnings and spending, prove how long the money will last. -/
theorem pauls_money_duration (lawn_money weed_money weekly_spending : ℕ) 
  (h1 : lawn_money = 44)
  (h2 : weed_money = 28)
  (h3 : weekly_spending = 9) :
  (lawn_money + weed_money) / weekly_spending = 8 := by
  sorry

end pauls_money_duration_l440_44030


namespace sum_of_three_consecutive_integers_l440_44072

theorem sum_of_three_consecutive_integers : ∃ (n : ℤ),
  (n - 1) + n + (n + 1) = 21 ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 17) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 11) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 25) ∧
  ¬(∃ (m : ℤ), (m - 1) + m + (m + 1) = 8) :=
by sorry

end sum_of_three_consecutive_integers_l440_44072
