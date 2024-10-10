import Mathlib

namespace paige_pencils_at_home_l950_95051

/-- The number of pencils Paige had in her backpack -/
def pencils_in_backpack : ℕ := 2

/-- The difference between the number of pencils at home and in the backpack -/
def pencil_difference : ℕ := 13

/-- The number of pencils Paige had at home -/
def pencils_at_home : ℕ := pencils_in_backpack + pencil_difference

theorem paige_pencils_at_home :
  pencils_at_home = 15 := by sorry

end paige_pencils_at_home_l950_95051


namespace least_exponent_sum_for_2000_l950_95011

def is_valid_representation (powers : List ℤ) : Prop :=
  (2000 : ℚ) = (powers.map (λ x => (2 : ℚ) ^ x)).sum ∧
  powers.Nodup ∧
  ∃ x ∈ powers, x < 0

theorem least_exponent_sum_for_2000 :
  ∃ (powers : List ℤ),
    is_valid_representation powers ∧
    ∀ (other_powers : List ℤ),
      is_valid_representation other_powers →
      (powers.sum ≤ other_powers.sum) :=
by sorry

end least_exponent_sum_for_2000_l950_95011


namespace part1_part2_l950_95076

-- Definition of "shifted equation"
def is_shifted_equation (f g : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x = 0 ∧ g y = 0 ∧ x = y + 1

-- Part 1
theorem part1 : is_shifted_equation (λ x => 2*x + 1) (λ x => 2*x + 3) := by sorry

-- Part 2
theorem part2 : ∃ m : ℝ, 
  is_shifted_equation 
    (λ x => 3*(x-1) - m - (m+3)/2) 
    (λ x => 2*(x-3) - 1 - (3-(x+1))) ∧ 
  m = 5 := by sorry

end part1_part2_l950_95076


namespace system_solution_iff_b_in_range_l950_95023

/-- The system of equations has at least one solution for any value of parameter a 
    if and only if b is in the specified range -/
theorem system_solution_iff_b_in_range (b : ℝ) : 
  (∀ a : ℝ, ∃ x y : ℝ, 
    x * Real.cos a + y * Real.sin a + 3 ≤ 0 ∧ 
    x^2 + y^2 + 8*x - 4*y - b^2 + 6*b + 11 = 0) ↔ 
  (b ≤ -2 * Real.sqrt 5 ∨ b ≥ 6 + 2 * Real.sqrt 5) :=
sorry

end system_solution_iff_b_in_range_l950_95023


namespace fraction_equality_l950_95068

theorem fraction_equality : (3 * 4 + 5) / 7 = 17 / 7 := by
  sorry

end fraction_equality_l950_95068


namespace geometric_sequence_fifth_term_l950_95045

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : a 3 * a 7 = 3 ∧ a 3 + a 7 = 4) :
  a 5 = Real.sqrt 3 := by
  sorry

end geometric_sequence_fifth_term_l950_95045


namespace problem_figure_total_triangles_l950_95041

/-- Represents a triangular figure composed of equilateral triangles --/
structure TriangularFigure where
  rows : ℕ
  bottom_row_count : ℕ

/-- Calculates the total number of triangles in the figure --/
def total_triangles (figure : TriangularFigure) : ℕ :=
  sorry

/-- The specific triangular figure described in the problem --/
def problem_figure : TriangularFigure :=
  { rows := 4
  , bottom_row_count := 4 }

/-- Theorem stating that the total number of triangles in the problem figure is 16 --/
theorem problem_figure_total_triangles :
  total_triangles problem_figure = 16 := by sorry

end problem_figure_total_triangles_l950_95041


namespace paco_cookie_difference_l950_95056

/-- Given Paco's cookie situation, prove that he ate 9 more cookies than he gave away -/
theorem paco_cookie_difference (initial_cookies : ℕ) (cookies_given : ℕ) (cookies_eaten : ℕ)
  (h1 : initial_cookies = 41)
  (h2 : cookies_given = 9)
  (h3 : cookies_eaten = 18) :
  cookies_eaten - cookies_given = 9 := by
  sorry

end paco_cookie_difference_l950_95056


namespace polynomial_evaluation_l950_95053

-- Define the polynomial p
def p (x : ℝ) : ℝ := sorry

-- State the theorem
theorem polynomial_evaluation (y : ℝ) :
  (p (y^2 + 1) = 6 * y^4 - y^2 + 5) →
  (p (y^2 - 1) = 6 * y^4 - 25 * y^2 + 31) :=
by sorry

end polynomial_evaluation_l950_95053


namespace problem_solution_l950_95057

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 30)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end problem_solution_l950_95057


namespace ratio_of_divisor_sums_l950_95063

def N : ℕ := 48 * 48 * 55 * 125 * 81

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 510 := by sorry

end ratio_of_divisor_sums_l950_95063


namespace sufficient_but_not_necessary_condition_l950_95075

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, x > 2 → (x + 1) * (x - 2) > 0) ∧
  (∃ x : ℝ, (x + 1) * (x - 2) > 0 ∧ ¬(x > 2)) :=
by sorry

end sufficient_but_not_necessary_condition_l950_95075


namespace product_equals_sum_exists_percentage_calculation_l950_95062

-- Problem 1
theorem product_equals_sum_exists : ∃ (a b c : ℤ), a * b * c = a + b + c := by
  sorry

-- Problem 2
theorem percentage_calculation : (12.5 / 100) * 44 = 5.5 := by
  sorry

end product_equals_sum_exists_percentage_calculation_l950_95062


namespace rational_equation_solution_l950_95008

theorem rational_equation_solution :
  ∃ x : ℚ, (x^2 - 7*x + 10) / (x^2 - 6*x + 5) = (x^2 - 4*x - 21) / (x^2 - 3*x - 18) ∧ x = 7/2 := by
  sorry

end rational_equation_solution_l950_95008


namespace quadratic_solution_proof_l950_95022

theorem quadratic_solution_proof (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end quadratic_solution_proof_l950_95022


namespace smallest_x_l950_95099

theorem smallest_x (x a b : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (h3 : x > 0) : x ≥ 200000 := by
  sorry

end smallest_x_l950_95099


namespace pi_half_not_in_M_l950_95024

-- Define the set M
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem pi_half_not_in_M : π / 2 ∉ M := by
  sorry

end pi_half_not_in_M_l950_95024


namespace denominator_numerator_difference_l950_95052

/-- The repeating decimal 0.868686... -/
def F : ℚ := 86 / 99

/-- F expressed as a decimal is 0.868686... (infinitely repeating) -/
axiom F_decimal : F = 0.868686

theorem denominator_numerator_difference :
  (F.den : ℤ) - (F.num : ℤ) = 13 := by sorry

end denominator_numerator_difference_l950_95052


namespace cubic_function_range_l950_95078

/-- If f(x) = x^3 - a and the graph of f(x) does not pass through the second quadrant, then a ∈ [0, +∞) -/
theorem cubic_function_range (a : ℝ) : 
  (∀ x : ℝ, (x ≤ 0 ∧ x^3 - a ≥ 0) → False) → 
  a ∈ Set.Ici (0 : ℝ) := by
  sorry

end cubic_function_range_l950_95078


namespace intersection_of_A_and_B_l950_95089

def A : Set ℝ := {x | |x - 3| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end intersection_of_A_and_B_l950_95089


namespace simultaneous_divisibility_l950_95086

theorem simultaneous_divisibility (x y : ℤ) :
  (17 ∣ (2 * x + 3 * y)) ↔ (17 ∣ (9 * x + 5 * y)) :=
by sorry

end simultaneous_divisibility_l950_95086


namespace total_remaining_value_l950_95005

/-- Represents the types of gift cards Jack has --/
inductive GiftCardType
  | BestBuy
  | Target
  | Walmart
  | Amazon

/-- Represents the initial number and value of each type of gift card --/
def initial_gift_cards : List (GiftCardType × Nat × Nat) :=
  [(GiftCardType.BestBuy, 5, 500),
   (GiftCardType.Target, 3, 250),
   (GiftCardType.Walmart, 7, 100),
   (GiftCardType.Amazon, 2, 1000)]

/-- Represents the number of gift cards Jack sent codes for --/
def sent_gift_cards : List (GiftCardType × Nat) :=
  [(GiftCardType.BestBuy, 1),
   (GiftCardType.Walmart, 2),
   (GiftCardType.Amazon, 1)]

/-- Calculates the total value of remaining gift cards --/
def remaining_value (initial : List (GiftCardType × Nat × Nat)) (sent : List (GiftCardType × Nat)) : Nat :=
  sorry

/-- Theorem stating that the total value of gift cards Jack can still return is $4250 --/
theorem total_remaining_value : 
  remaining_value initial_gift_cards sent_gift_cards = 4250 := by
  sorry

end total_remaining_value_l950_95005


namespace casino_chip_loss_difference_l950_95084

theorem casino_chip_loss_difference : 
  ∀ (x y : ℕ), 
    x + y = 16 →  -- Total number of chips lost
    20 * x + 100 * y = 880 →  -- Value of lost chips
    x - y = 2 :=  -- Difference in number of chips lost
by
  sorry

end casino_chip_loss_difference_l950_95084


namespace right_triangle_proof_l950_95066

theorem right_triangle_proof (n : ℝ) (hn : n > 0) :
  let a := 2*n^2 + 2*n + 1
  let b := 2*n^2 + 2*n
  let c := 2*n + 1
  a^2 = b^2 + c^2 := by sorry

end right_triangle_proof_l950_95066


namespace planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_not_all_lines_perpendicular_to_line_are_parallel_not_all_planes_perpendicular_to_plane_are_parallel_or_intersect_l950_95019

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect_plane : Plane → Plane → Prop)

-- Theorem statements
theorem planes_perpendicular_to_line_are_parallel 
  (p1 p2 : Plane) (l : Line) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_plane p1 p2 :=
sorry

theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_line l1 l2 :=
sorry

theorem not_all_lines_perpendicular_to_line_are_parallel : 
  ∃ (l1 l2 l3 : Line), 
    perpendicular_line_line l1 l3 ∧ 
    perpendicular_line_line l2 l3 ∧ 
    ¬(parallel_line l1 l2) :=
sorry

theorem not_all_planes_perpendicular_to_plane_are_parallel_or_intersect : 
  ∃ (p1 p2 p3 : Plane), 
    perpendicular_plane_plane p1 p3 ∧ 
    perpendicular_plane_plane p2 p3 ∧ 
    ¬(parallel_plane p1 p2 ∨ intersect_plane p1 p2) :=
sorry

end planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_not_all_lines_perpendicular_to_line_are_parallel_not_all_planes_perpendicular_to_plane_are_parallel_or_intersect_l950_95019


namespace inequalities_proof_l950_95038

theorem inequalities_proof (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequalities_proof_l950_95038


namespace range_of_m_l950_95067

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧ 
   (∃ y : ℝ, (y - m + 1) * (y - m - 1) ≤ 0 ∧ |y - 3| > 2)) →
  2 ≤ m ∧ m ≤ 4 := by
sorry

end range_of_m_l950_95067


namespace solution_set_inequality_l950_95083

theorem solution_set_inequality (x : ℝ) : 
  1 / x < 1 / 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 3 :=
sorry

end solution_set_inequality_l950_95083


namespace rectangular_field_width_l950_95035

theorem rectangular_field_width (length width : ℝ) : 
  length = 24 ∧ length = 2 * width - 3 → width = 13.5 := by
  sorry

end rectangular_field_width_l950_95035


namespace circles_externally_tangent_l950_95020

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The equation of the first circle: x^2 + y^2 = 1 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The equation of the second circle: x^2 + y^2 - 6x - 8y + 9 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

theorem circles_externally_tangent :
  externally_tangent (0, 0) (3, 4) 1 4 := by
  sorry

end circles_externally_tangent_l950_95020


namespace shortest_distance_to_x_axis_l950_95050

/-- Two points on a parabola -/
structure PointsOnParabola where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  on_parabola₁ : x₁^2 = 4*y₁
  on_parabola₂ : x₂^2 = 4*y₂
  distance : Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 6

/-- Theorem: The shortest distance from the midpoint of AB to the x-axis is 2 -/
theorem shortest_distance_to_x_axis (p : PointsOnParabola) :
  (p.y₁ + p.y₂) / 2 ≥ 2 := by sorry

end shortest_distance_to_x_axis_l950_95050


namespace point_not_in_reflected_rectangle_l950_95025

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  ⟨-p.x, p.y⟩

/-- The set of vertices of the original rectangle -/
def original_vertices : Set Point :=
  {⟨1, 3⟩, ⟨1, 1⟩, ⟨4, 1⟩, ⟨4, 3⟩}

/-- The set of vertices of the reflected rectangle -/
def reflected_vertices : Set Point :=
  original_vertices.image reflect_y

/-- The point in question -/
def point_to_check : Point :=
  ⟨-3, 4⟩

theorem point_not_in_reflected_rectangle :
  point_to_check ∉ reflected_vertices :=
sorry

end point_not_in_reflected_rectangle_l950_95025


namespace linear_function_problem_l950_95006

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

-- State the theorem
theorem linear_function_problem (f : ℝ → ℝ) 
  (h_linear : LinearFunction f)
  (h_diff : f 10 - f 5 = 20)
  (h_f0 : f 0 = 3) :
  f 15 - f 5 = 40 := by
  sorry

end linear_function_problem_l950_95006


namespace g_composition_of_three_l950_95037

def g (n : ℤ) : ℤ :=
  if n < 5 then 2 * n^2 + 3 else 4 * n + 1

theorem g_composition_of_three : g (g (g 3)) = 341 := by
  sorry

end g_composition_of_three_l950_95037


namespace pagoda_lights_l950_95012

theorem pagoda_lights (n : ℕ) (total : ℕ) (h1 : n = 7) (h2 : total = 381) :
  ∃ (a : ℕ), 
    a * (1 - (1/2)^n) / (1 - 1/2) = total ∧ 
    a * (1/2)^(n-1) = 3 :=
sorry

end pagoda_lights_l950_95012


namespace pendant_prices_and_optimal_plan_l950_95090

/-- The price of a "Bing Dwen Dwen" pendant in yuan -/
def bing_price : ℝ := 8

/-- The price of a "Shuey Rong Rong" pendant in yuan -/
def shuey_price : ℝ := 10

/-- The cost of 2 "Bing Dwen Dwen" and 1 "Shuey Rong Rong" pendants -/
def cost1 : ℝ := 26

/-- The cost of 4 "Bing Dwen Dwen" and 3 "Shuey Rong Rong" pendants -/
def cost2 : ℝ := 62

/-- The total number of pendants to purchase -/
def total_pendants : ℕ := 100

/-- The number of "Bing Dwen Dwen" pendants in the optimal plan -/
def optimal_bing : ℕ := 75

/-- The number of "Shuey Rong Rong" pendants in the optimal plan -/
def optimal_shuey : ℕ := 25

/-- The minimum cost for the optimal plan -/
def min_cost : ℝ := 850

theorem pendant_prices_and_optimal_plan :
  (2 * bing_price + shuey_price = cost1) ∧
  (4 * bing_price + 3 * shuey_price = cost2) ∧
  (optimal_bing + optimal_shuey = total_pendants) ∧
  (3 * optimal_shuey ≥ optimal_bing) ∧
  (optimal_bing * bing_price + optimal_shuey * shuey_price = min_cost) ∧
  (∀ x y : ℕ, x + y = total_pendants → 3 * y ≥ x → 
    x * bing_price + y * shuey_price ≥ min_cost) :=
by sorry

#check pendant_prices_and_optimal_plan

end pendant_prices_and_optimal_plan_l950_95090


namespace increasing_function_a_range_l950_95072

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) → -3 ≤ a ∧ a ≤ 3 := by
  sorry

end increasing_function_a_range_l950_95072


namespace perpendicular_line_through_point_l950_95000

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_through_point : 
  ∃ (l : Line), 
    perpendicular l { a := 3, b := -5, c := 6 } ∧ 
    point_on_line (-1) 2 l ∧
    l = { a := 5, b := 3, c := -1 } :=
sorry

end perpendicular_line_through_point_l950_95000


namespace power_of_product_l950_95074

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end power_of_product_l950_95074


namespace monotonicity_condition_even_function_condition_minimum_value_l950_95021

-- Define the function f(x) = x^2 + 2ax
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x

-- Define the domain [-5, 5]
def domain : Set ℝ := Set.Icc (-5) 5

-- Statement 1: Monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x < f a y) ∨
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x > f a y) ↔
  a ≤ -5 ∨ a ≥ 5 :=
sorry

-- Statement 2: Even function condition and extrema
theorem even_function_condition (a : ℝ) :
  (∀ x ∈ domain, f a x - 2*x = f a (-x) - 2*(-x)) →
  a = 1 ∧ 
  (∀ x ∈ domain, f a x ≤ 35) ∧
  (∀ x ∈ domain, f a x ≥ -1) ∧
  (∃ x ∈ domain, f a x = 35) ∧
  (∃ x ∈ domain, f a x = -1) :=
sorry

-- Statement 3: Minimum value
theorem minimum_value (a : ℝ) :
  (a ≥ 5 → ∀ x ∈ domain, f a x ≥ 25 - 10*a) ∧
  (a ≤ -5 → ∀ x ∈ domain, f a x ≥ 25 + 10*a) ∧
  (-5 < a ∧ a < 5 → ∀ x ∈ domain, f a x ≥ -a^2) :=
sorry

end monotonicity_condition_even_function_condition_minimum_value_l950_95021


namespace largest_band_formation_l950_95047

/-- Represents a rectangular band formation -/
structure BandFormation where
  m : ℕ  -- Total number of band members
  r : ℕ  -- Number of rows
  x : ℕ  -- Number of members in each row

/-- Checks if a band formation is valid according to the problem conditions -/
def isValidFormation (f : BandFormation) : Prop :=
  f.r * f.x + 5 = f.m ∧
  (f.r - 3) * (f.x + 2) = f.m ∧
  f.m < 100

/-- The theorem stating the largest possible number of band members -/
theorem largest_band_formation :
  ∃ (f : BandFormation), isValidFormation f ∧
    ∀ (g : BandFormation), isValidFormation g → g.m ≤ f.m :=
by sorry

end largest_band_formation_l950_95047


namespace expression_simplification_l950_95060

theorem expression_simplification :
  let x := Real.pi / 18  -- 10 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) /
  (Real.sin (17 * x) - Real.sqrt (1 - Real.sin (17 * x) ^ 2)) = -1 := by
  sorry

end expression_simplification_l950_95060


namespace consecutive_page_numbers_sum_l950_95028

theorem consecutive_page_numbers_sum (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  y = x + 1 ∧ z = y + 1 ∧
  x * y = 20412 →
  x + y + z = 429 := by
sorry

end consecutive_page_numbers_sum_l950_95028


namespace sandwich_non_condiment_percentage_l950_95079

theorem sandwich_non_condiment_percentage
  (total_weight : ℝ)
  (condiment_weight : ℝ)
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end sandwich_non_condiment_percentage_l950_95079


namespace trapezoid_segment_property_l950_95058

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 120 = longer_base
  midline_ratio_condition : midline_ratio = 3 / 4

/-- The main theorem -/
theorem trapezoid_segment_property (t : Trapezoid) : 
  ⌊(t.equal_area_segment^2) / 120⌋ = 217 := by
  sorry

end trapezoid_segment_property_l950_95058


namespace brother_age_proof_l950_95091

def brother_age_in_5_years (nick_age : ℕ) : ℕ :=
  let sister_age := nick_age + 6
  let brother_age := (nick_age + sister_age) / 2
  brother_age + 5

theorem brother_age_proof (nick_age : ℕ) (h : nick_age = 13) :
  brother_age_in_5_years nick_age = 21 := by
  sorry

end brother_age_proof_l950_95091


namespace house_value_correct_l950_95030

/-- Represents the inheritance distribution problem --/
structure InheritanceProblem where
  totalBrothers : Nat
  housesCount : Nat
  moneyPaidPerOlderBrother : Nat
  totalInheritance : Nat

/-- Calculates the value of one house given the inheritance problem --/
def houseValue (problem : InheritanceProblem) : Nat :=
  let olderBrothersCount := problem.housesCount
  let youngerBrothersCount := problem.totalBrothers - olderBrothersCount
  let totalMoneyPaid := olderBrothersCount * problem.moneyPaidPerOlderBrother
  let inheritancePerBrother := problem.totalInheritance / problem.totalBrothers
  (inheritancePerBrother * problem.totalBrothers - totalMoneyPaid) / problem.housesCount

/-- Theorem stating that the house value is correct for the given problem --/
theorem house_value_correct (problem : InheritanceProblem) :
  problem.totalBrothers = 5 →
  problem.housesCount = 3 →
  problem.moneyPaidPerOlderBrother = 2000 →
  problem.totalInheritance = 15000 →
  houseValue problem = 3000 := by
  sorry

#eval houseValue { totalBrothers := 5, housesCount := 3, moneyPaidPerOlderBrother := 2000, totalInheritance := 15000 }

end house_value_correct_l950_95030


namespace quadratic_function_uniqueness_l950_95046

/-- A quadratic function with vertex (h, k) and y-intercept (0, y0) -/
def QuadraticFunction (a b c h k y0 : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k ∧
  c = y0 ∧
  -b / (2 * a) = h ∧
  a * h^2 + b * h + c = k

theorem quadratic_function_uniqueness (a b c : ℝ) : 
  QuadraticFunction a b c 2 (-1) 11 → a = 3 ∧ b = -12 ∧ c = 11 := by
  sorry

end quadratic_function_uniqueness_l950_95046


namespace trig_identity_l950_95081

theorem trig_identity (α : ℝ) (h1 : α ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.sin (α + π/4) = -1/3) : 
  Real.sin (2*α) / Real.cos (π/4 - α) = 7/3 := by
  sorry

end trig_identity_l950_95081


namespace range_of_a_l950_95071

/-- Proposition p: x^2 + 2ax + 4 > 0 holds for all x ∈ ℝ -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: x^2 - (a+1)x + 1 ≤ 0 has an empty solution set -/
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - (a+1)*x + 1 > 0

/-- The disjunction p ∨ q is true -/
axiom h1 (a : ℝ) : p a ∨ q a

/-- The conjunction p ∧ q is false -/
axiom h2 (a : ℝ) : ¬(p a ∧ q a)

/-- The range of values for a is (-3, -2] ∪ [1, 2) -/
theorem range_of_a : 
  {a : ℝ | (a > -3 ∧ a ≤ -2) ∨ (a ≥ 1 ∧ a < 2)} = {a : ℝ | p a ∨ q a ∧ ¬(p a ∧ q a)} :=
sorry

end range_of_a_l950_95071


namespace division_problem_l950_95092

theorem division_problem (total : ℚ) (a b c : ℚ) 
  (h1 : total = 544)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : c = 384 := by
  sorry

end division_problem_l950_95092


namespace integral_roots_system_l950_95049

theorem integral_roots_system : ∃! (x y z : ℤ),
  (z : ℝ) ^ (x : ℝ) = (y : ℝ) ^ (2 * x : ℝ) ∧
  (2 : ℝ) ^ (z : ℝ) = 2 * (4 : ℝ) ^ (x : ℝ) ∧
  x + y + z = 16 ∧
  x = 4 ∧ y = 3 ∧ z = 9 := by
sorry

end integral_roots_system_l950_95049


namespace max_value_and_sum_l950_95087

theorem max_value_and_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 4050) : 
  let M := a*c + 3*b*c + 2*c*d + 8*d*e
  ∃ (a_M b_M c_M d_M e_M : ℝ),
    (∀ a' b' c' d' e' : ℝ, 
      0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 
      a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 4050 → 
      a'*c' + 3*b'*c' + 2*c'*d' + 8*d'*e' ≤ M) ∧
    M = 4050 * Real.sqrt 14 ∧
    M + a_M + b_M + c_M + d_M + e_M = 4050 * Real.sqrt 14 + 90 :=
by sorry

end max_value_and_sum_l950_95087


namespace walkers_speed_l950_95077

theorem walkers_speed (speed_man2 : ℝ) (distance_apart : ℝ) (time : ℝ) (speed_man1 : ℝ) :
  speed_man2 = 12 →
  distance_apart = 2 →
  time = 1 →
  speed_man2 * time - speed_man1 * time = distance_apart →
  speed_man1 = 10 := by
sorry

end walkers_speed_l950_95077


namespace quadratic_roots_equal_and_real_l950_95069

theorem quadratic_roots_equal_and_real (a c : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, a * x^2 - 2 * x * Real.sqrt 2 + c = 0) ∧
  ((-2 * Real.sqrt 2)^2 - 4 * a * c = 0) →
  ∃! x : ℝ, a * x^2 - 2 * x * Real.sqrt 2 + c = 0 :=
by sorry

end quadratic_roots_equal_and_real_l950_95069


namespace unique_n_solution_l950_95002

def is_not_divisible_by_cube_of_prime (x : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^3 ∣ x)

theorem unique_n_solution :
  ∃! n : ℕ, n ≥ 1 ∧
    ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧
      is_not_divisible_by_cube_of_prime (a^2 + b + 3) ∧
      n = (a * b + 3 * b + 8) / (a^2 + b + 3) ∧
      n = 3 :=
sorry

end unique_n_solution_l950_95002


namespace unique_solution_l950_95080

/-- The exponent function for our problem -/
def f (m : ℕ+) : ℤ := m^2 - 2*m - 3

/-- The condition that the exponent is negative -/
def condition1 (m : ℕ+) : Prop := f m < 0

/-- The condition that the exponent is odd -/
def condition2 (m : ℕ+) : Prop := ∃ k : ℤ, f m = 2*k + 1

/-- The theorem stating that 2 is the only positive integer satisfying all conditions -/
theorem unique_solution :
  ∃! m : ℕ+, condition1 m ∧ condition2 m ∧ m = 2 :=
sorry

end unique_solution_l950_95080


namespace tv_show_average_episodes_l950_95095

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

#check tv_show_average_episodes

end tv_show_average_episodes_l950_95095


namespace min_x_given_inequality_l950_95054

theorem min_x_given_inequality (x : ℝ) :
  (∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) →
  x ≥ -1 ∧ ∀ y : ℝ, (∀ a : ℝ, a > 0 → y^2 ≤ 1 + a) → y ≥ x :=
by sorry

end min_x_given_inequality_l950_95054


namespace toms_candy_problem_l950_95039

/-- Tom's candy problem -/
theorem toms_candy_problem (initial : Nat) (bought : Nat) (total : Nat) (friend_gave : Nat) : 
  initial = 2 → 
  bought = 10 → 
  total = 19 → 
  initial + bought + friend_gave = total → 
  friend_gave = 7 := by
  sorry

#check toms_candy_problem

end toms_candy_problem_l950_95039


namespace expression_value_at_two_l950_95061

theorem expression_value_at_two :
  let x : ℕ := 2
  x + x * (x ^ x) = 10 := by sorry

end expression_value_at_two_l950_95061


namespace gas_used_l950_95026

theorem gas_used (initial_gas final_gas : ℝ) (h1 : initial_gas = 0.5) (h2 : final_gas = 0.17) :
  initial_gas - final_gas = 0.33 := by
sorry

end gas_used_l950_95026


namespace age_difference_l950_95018

theorem age_difference (anand_age_10_years_ago bala_age_10_years_ago : ℕ) : 
  anand_age_10_years_ago = bala_age_10_years_ago / 3 →
  anand_age_10_years_ago + 10 = 15 →
  (bala_age_10_years_ago + 10) - (anand_age_10_years_ago + 10) = 10 := by
  sorry

end age_difference_l950_95018


namespace emily_necklaces_l950_95048

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 16) (h2 : beads_per_necklace = 8) :
  total_beads / beads_per_necklace = 2 := by
  sorry

end emily_necklaces_l950_95048


namespace proportionality_check_l950_95004

-- Define the type of proportionality
inductive Proportionality
  | Direct
  | Inverse
  | Neither

-- Define a function to check proportionality
def check_proportionality (eq : ℝ → ℝ → Prop) : Proportionality :=
  sorry

-- Theorem statement
theorem proportionality_check :
  (check_proportionality (fun x y => 2*x + y = 5) = Proportionality.Neither) ∧
  (check_proportionality (fun x y => 4*x*y = 15) = Proportionality.Inverse) ∧
  (check_proportionality (fun x y => x = 7*y) = Proportionality.Direct) ∧
  (check_proportionality (fun x y => 2*x + 3*y = 12) = Proportionality.Neither) ∧
  (check_proportionality (fun x y => x/y = 4) = Proportionality.Direct) :=
by sorry

end proportionality_check_l950_95004


namespace jerry_tips_problem_l950_95032

/-- The amount Jerry needs to earn on the fifth night to achieve an average of $50 per night -/
theorem jerry_tips_problem (
  days_per_week : ℕ)
  (target_average : ℝ)
  (past_earnings : List ℝ)
  (h1 : days_per_week = 5)
  (h2 : target_average = 50)
  (h3 : past_earnings = [20, 60, 15, 40]) :
  target_average * days_per_week - past_earnings.sum = 115 := by
  sorry

end jerry_tips_problem_l950_95032


namespace f_n_ratio_theorem_l950_95098

noncomputable section

def f (x : ℝ) : ℝ := (x^2 + 1) / (2*x)

def f_n : ℕ → ℝ → ℝ
| 0, x => x
| n+1, x => f (f_n n x)

def N (n : ℕ) : ℕ := 2^n

theorem f_n_ratio_theorem (x : ℝ) (n : ℕ) (hx : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) :
  (f_n n x) / (f_n (n+1) x) = 1 + 1 / f ((((x+1)/(x-1)) ^ (N n))) :=
sorry

end f_n_ratio_theorem_l950_95098


namespace floor_ceil_sum_l950_95033

theorem floor_ceil_sum : ⌊(3.998 : ℝ)⌋ + ⌈(7.002 : ℝ)⌉ = 11 := by sorry

end floor_ceil_sum_l950_95033


namespace rectangular_field_area_l950_95016

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio. -/
theorem rectangular_field_area (perimeter : ℝ) (width_ratio : ℝ) : 
  perimeter = 72 ∧ width_ratio = 1/3 → 
  (perimeter / (4 * (1 + width_ratio))) * (perimeter * width_ratio / (4 * (1 + width_ratio))) = 243 := by
  sorry

end rectangular_field_area_l950_95016


namespace trip_time_at_new_speed_l950_95017

-- Define the original speed, time, and new speed
def original_speed : ℝ := 80
def original_time : ℝ := 3
def new_speed : ℝ := 50

-- Define the constant distance
def distance : ℝ := original_speed * original_time

-- Theorem to prove
theorem trip_time_at_new_speed :
  distance / new_speed = 4.8 := by sorry

end trip_time_at_new_speed_l950_95017


namespace triangle_side_length_l950_95093

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a = Real.sqrt 3 →
  Real.sin B = 1 / 2 →
  C = π / 6 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b = 1 :=
by sorry

end triangle_side_length_l950_95093


namespace dividend_calculation_l950_95029

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 3) : 
  divisor * quotient + remainder = 165 := by
  sorry

end dividend_calculation_l950_95029


namespace lines_do_not_intersect_l950_95064

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

theorem lines_do_not_intersect (k : ℝ) : 
  (parallel 
    { point := (1, 3), direction := (2, -5) }
    { point := (-1, 4), direction := (3, k) }) ↔ 
  k = -15/2 := by
  sorry

end lines_do_not_intersect_l950_95064


namespace psychiatric_sessions_l950_95042

theorem psychiatric_sessions 
  (total_patients : ℕ) 
  (total_sessions : ℕ) 
  (first_patient_sessions : ℕ) 
  (second_patient_additional_sessions : ℕ) :
  total_patients = 4 →
  total_sessions = 25 →
  first_patient_sessions = 6 →
  second_patient_additional_sessions = 5 →
  total_sessions - (first_patient_sessions + (first_patient_sessions + second_patient_additional_sessions)) = 8 :=
by sorry

end psychiatric_sessions_l950_95042


namespace expression_evaluation_l950_95097

theorem expression_evaluation (a b c : ℝ) : 
  (a - (b - c)) - ((a - b) - c) = 2 * c := by
  sorry

end expression_evaluation_l950_95097


namespace discount_calculation_l950_95055

theorem discount_calculation (CP : ℝ) (MP SP discount : ℝ) : 
  MP = 1.1 * CP → 
  SP = 0.99 * CP → 
  discount = MP - SP → 
  discount = 0.11 * CP :=
by sorry

end discount_calculation_l950_95055


namespace inequality_range_l950_95070

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3) 1 :=
by sorry

end inequality_range_l950_95070


namespace larger_integer_value_l950_95065

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℝ) / (b : ℝ) = 7 / 3)
  (h_product : (a : ℕ) * b = 168) : 
  (a : ℝ) = 14 * Real.sqrt 2 :=
sorry

end larger_integer_value_l950_95065


namespace points_in_quadrants_I_and_II_l950_95085

def in_quadrant_I_or_II (x y : ℝ) : Prop := (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > 6 - x → in_quadrant_I_or_II x y := by
  sorry

end points_in_quadrants_I_and_II_l950_95085


namespace lidia_remaining_money_l950_95007

/-- Proves the remaining money after Lidia buys her needed apps -/
theorem lidia_remaining_money 
  (app_cost : ℕ) 
  (apps_needed : ℕ) 
  (available_money : ℕ) 
  (h1 : app_cost = 4)
  (h2 : apps_needed = 15)
  (h3 : available_money = 66) :
  available_money - (app_cost * apps_needed) = 6 :=
by sorry

end lidia_remaining_money_l950_95007


namespace arithmetic_progression_bijection_l950_95034

theorem arithmetic_progression_bijection (f : ℕ → ℕ) (hf : Function.Bijective f) :
  ∃ a b c : ℕ, (b - a = c - b) ∧ (f a < f b) ∧ (f b < f c) := by
  sorry

end arithmetic_progression_bijection_l950_95034


namespace alice_quarters_l950_95003

/-- Represents the number of quarters Alice had initially -/
def initial_quarters : ℕ := 20

/-- Represents the number of nickels Alice received after exchange -/
def total_nickels : ℕ := 100

/-- Represents the value of a regular nickel in dollars -/
def regular_nickel_value : ℚ := 1/20

/-- Represents the value of an iron nickel in dollars -/
def iron_nickel_value : ℚ := 3

/-- Represents the proportion of iron nickels -/
def iron_nickel_proportion : ℚ := 1/5

/-- Represents the proportion of regular nickels -/
def regular_nickel_proportion : ℚ := 4/5

/-- Represents the total value of all nickels in dollars -/
def total_value : ℚ := 64

theorem alice_quarters :
  (iron_nickel_proportion * total_nickels * iron_nickel_value + 
   regular_nickel_proportion * total_nickels * regular_nickel_value = total_value) ∧
  (initial_quarters * 5 = total_nickels) := by
  sorry

end alice_quarters_l950_95003


namespace james_caprisun_purchase_l950_95096

/-- The total cost of James' Capri-sun purchase -/
def total_cost (num_boxes : ℕ) (pouches_per_box : ℕ) (cost_per_pouch : ℚ) : ℚ :=
  (num_boxes * pouches_per_box : ℕ) * cost_per_pouch

/-- Theorem stating the total cost of James' purchase -/
theorem james_caprisun_purchase :
  total_cost 10 6 (20 / 100) = 12 := by
  sorry

end james_caprisun_purchase_l950_95096


namespace two_bagels_solution_l950_95040

/-- Represents the number of items bought in a week -/
structure WeeklyPurchase where
  bagels : ℕ
  muffins : ℕ
  donuts : ℕ

/-- Checks if the weekly purchase is valid (totals to 6 days) -/
def isValidPurchase (wp : WeeklyPurchase) : Prop :=
  wp.bagels + wp.muffins + wp.donuts = 6

/-- Calculates the total cost in cents -/
def totalCost (wp : WeeklyPurchase) : ℕ :=
  60 * wp.bagels + 45 * wp.muffins + 30 * wp.donuts

/-- Checks if the total cost is a whole number of dollars -/
def isWholeDollarAmount (wp : WeeklyPurchase) : Prop :=
  totalCost wp % 100 = 0

/-- Main theorem: There exists a valid purchase with 2 bagels that costs a whole dollar amount -/
theorem two_bagels_solution :
  ∃ (wp : WeeklyPurchase), wp.bagels = 2 ∧ isValidPurchase wp ∧ isWholeDollarAmount wp :=
sorry

end two_bagels_solution_l950_95040


namespace species_x_count_day_6_l950_95027

/-- Represents the number of days passed -/
def days : ℕ := 6

/-- The population growth factor for Species X per day -/
def species_x_growth : ℕ := 2

/-- The population growth factor for Species Y per day -/
def species_y_growth : ℕ := 4

/-- The total number of ants on Day 0 -/
def initial_total : ℕ := 40

/-- The total number of ants on Day 6 -/
def final_total : ℕ := 21050

/-- Theorem stating that the number of Species X ants on Day 6 is 2304 -/
theorem species_x_count_day_6 : ℕ := by
  sorry

end species_x_count_day_6_l950_95027


namespace fractional_factorial_max_test_points_l950_95009

/-- The number of experiments in the fractional factorial design. -/
def num_experiments : ℕ := 6

/-- The maximum number of test points that can be handled. -/
def max_test_points : ℕ := 20

/-- Theorem stating that given 6 experiments in a fractional factorial design,
    the maximum number of test points that can be handled is 20. -/
theorem fractional_factorial_max_test_points :
  ∀ n : ℕ, n ≤ 2^num_experiments - 1 → n ≤ max_test_points :=
by sorry

end fractional_factorial_max_test_points_l950_95009


namespace jack_sent_three_bestbuy_cards_l950_95043

def total_requested : ℕ := 6 * 500 + 9 * 200

def walmart_sent : ℕ := 2

def walmart_value : ℕ := 200

def bestbuy_value : ℕ := 500

def remaining_value : ℕ := 3900

def bestbuy_sent : ℕ := 3

theorem jack_sent_three_bestbuy_cards :
  total_requested - remaining_value = walmart_sent * walmart_value + bestbuy_sent * bestbuy_value :=
by sorry

end jack_sent_three_bestbuy_cards_l950_95043


namespace height_area_ratio_not_always_equal_l950_95014

-- Define the properties of an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  side : ℝ
  perimeter : ℝ
  area : ℝ
  base_positive : 0 < base
  height_positive : 0 < height
  side_positive : 0 < side
  perimeter_eq : perimeter = base + 2 * side
  area_eq : area = (1/2) * base * height

-- Theorem statement
theorem height_area_ratio_not_always_equal : 
  ∃ (t1 t2 : IsoscelesTriangle), t1.height ≠ t2.height ∧ 
    (t1.height / t2.height ≠ t1.area / t2.area) := by
  sorry


end height_area_ratio_not_always_equal_l950_95014


namespace ellipse_triangle_area_implies_segment_length_l950_95088

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (0, 3)

-- Define the triangle area function
noncomputable def triangleArea (P A B : ℝ × ℝ) : ℝ := sorry

-- Define the length function
noncomputable def segmentLength (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_area_implies_segment_length :
  ∀ P A : ℝ × ℝ,
  ellipse P.1 P.2 →
  (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → triangleArea Q A B ≥ 1) →
  (∃ R : ℝ × ℝ, ellipse R.1 R.2 ∧ triangleArea R A B = 5) →
  segmentLength A B = Real.sqrt 7 := by
  sorry

end ellipse_triangle_area_implies_segment_length_l950_95088


namespace tara_savings_loss_l950_95013

/-- The amount Tara had saved before losing all her savings -/
def amount_lost : ℕ := by sorry

theorem tara_savings_loss :
  let clarinet_cost : ℕ := 90
  let initial_savings : ℕ := 10
  let book_price : ℕ := 5
  let total_books_sold : ℕ := 25
  amount_lost = 45 := by sorry

end tara_savings_loss_l950_95013


namespace store_shelves_theorem_l950_95044

/-- Calculates the number of shelves needed to display coloring books -/
def shelves_needed (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (initial_stock - books_sold) / books_per_shelf

/-- Theorem: Given the specific conditions, the number of shelves used is 9 -/
theorem store_shelves_theorem :
  shelves_needed 87 33 6 = 9 := by
  sorry

end store_shelves_theorem_l950_95044


namespace fixed_distance_from_linear_combination_l950_95073

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given vectors a and b, if q satisfies ‖q - b‖ = 3 ‖q - a‖, 
    then q is at a fixed distance from (9/8)a + (-1/8)b. -/
theorem fixed_distance_from_linear_combination (a b q : E) 
  (h : ‖q - b‖ = 3 * ‖q - a‖) :
  ∃ (c : ℝ), ∀ (q : E), ‖q - b‖ = 3 * ‖q - a‖ → 
    ‖q - ((9/8 : ℝ) • a + (-1/8 : ℝ) • b)‖ = c :=
sorry

end fixed_distance_from_linear_combination_l950_95073


namespace repeating_decimal_bounds_l950_95010

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℕ
  fractionalPart : List ℕ
  repeatingPart : List ℕ

/-- Converts a RepeatingDecimal to a real number -/
noncomputable def RepeatingDecimal.toReal (d : RepeatingDecimal) : ℝ :=
  sorry

/-- Generates all possible repeating decimals from a given decimal string -/
def generateRepeatingDecimals (s : String) : List RepeatingDecimal :=
  sorry

/-- Finds the maximum repeating decimal from a list of repeating decimals -/
noncomputable def findMaxRepeatingDecimal (decimals : List RepeatingDecimal) : RepeatingDecimal :=
  sorry

/-- Finds the minimum repeating decimal from a list of repeating decimals -/
noncomputable def findMinRepeatingDecimal (decimals : List RepeatingDecimal) : RepeatingDecimal :=
  sorry

theorem repeating_decimal_bounds :
  let decimals := generateRepeatingDecimals "0.20120415"
  let maxDecimal := findMaxRepeatingDecimal decimals
  let minDecimal := findMinRepeatingDecimal decimals
  maxDecimal = { integerPart := 0, fractionalPart := [2, 0, 1, 2, 0, 4, 1], repeatingPart := [5] } ∧
  minDecimal = { integerPart := 0, fractionalPart := [2], repeatingPart := [0, 1, 2, 0, 4, 1, 5] } :=
by sorry

end repeating_decimal_bounds_l950_95010


namespace aunt_gemma_dog_food_l950_95059

/-- Calculates the amount of food each dog consumes per meal given the total amount of food,
    number of days it lasts, number of dogs, and number of meals per day. -/
def food_per_meal_per_dog (total_food : ℕ) (num_days : ℕ) (num_dogs : ℕ) (meals_per_day : ℕ) : ℕ :=
  (total_food * 1000) / (num_days * num_dogs * meals_per_day)

theorem aunt_gemma_dog_food :
  let num_sacks : ℕ := 2
  let weight_per_sack : ℕ := 50  -- in kg
  let num_days : ℕ := 50
  let num_dogs : ℕ := 4
  let meals_per_day : ℕ := 2
  food_per_meal_per_dog (num_sacks * weight_per_sack) num_days num_dogs meals_per_day = 250 := by
  sorry

end aunt_gemma_dog_food_l950_95059


namespace trig_identity_l950_95015

theorem trig_identity (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.tan θ + (Real.tan θ)⁻¹ = 4 := by
  sorry

end trig_identity_l950_95015


namespace modular_arithmetic_problem_l950_95094

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (7 * a) % 56 = 1 ∧ (13 * b) % 56 = 1 ∧ 
  (3 * a + 9 * b) % 56 = 29 := by
  sorry

end modular_arithmetic_problem_l950_95094


namespace shaded_region_perimeter_l950_95036

/-- The perimeter of the shaded region formed by four touching circles -/
theorem shaded_region_perimeter (c : ℝ) (h : c = 48) : 
  let r := c / (2 * Real.pi)
  let arc_length := c / 4
  4 * arc_length = 48 := by
  sorry

end shaded_region_perimeter_l950_95036


namespace inequality_solution_range_of_a_l950_95082

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (x a : ℝ) : ℝ := 2 * |x| + a

-- Part 1: Inequality solution
theorem inequality_solution :
  {x : ℝ | f x ≤ g x (-1)} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a
theorem range_of_a (h : ∃ x₀ : ℝ, f x₀ ≥ (1/2) * g x₀ a) :
  a ≤ 2 :=
sorry

end inequality_solution_range_of_a_l950_95082


namespace sum_first_ten_even_numbers_l950_95001

-- Define the first 10 even numbers
def firstTenEvenNumbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

-- Theorem: The sum of the first 10 even numbers is 110
theorem sum_first_ten_even_numbers :
  firstTenEvenNumbers.sum = 110 := by
  sorry

end sum_first_ten_even_numbers_l950_95001


namespace club_average_age_l950_95031

theorem club_average_age (women : ℕ) (men : ℕ) (children : ℕ)
  (women_avg : ℝ) (men_avg : ℝ) (children_avg : ℝ)
  (h_women : women = 12)
  (h_men : men = 18)
  (h_children : children = 10)
  (h_women_avg : women_avg = 32)
  (h_men_avg : men_avg = 38)
  (h_children_avg : children_avg = 10) :
  (women * women_avg + men * men_avg + children * children_avg) / (women + men + children) = 29.2 := by
  sorry

end club_average_age_l950_95031
