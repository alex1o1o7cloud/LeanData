import Mathlib

namespace claire_gift_card_value_l1446_144654

/-- The value of Claire's gift card -/
def gift_card_value : ℚ := 100

/-- Cost of a latte -/
def latte_cost : ℚ := 3.75

/-- Cost of a croissant -/
def croissant_cost : ℚ := 3.50

/-- Cost of a cookie -/
def cookie_cost : ℚ := 1.25

/-- Number of days Claire buys coffee and pastry -/
def days : ℕ := 7

/-- Number of cookies Claire buys -/
def num_cookies : ℕ := 5

/-- Amount left on the gift card after spending -/
def amount_left : ℚ := 43

/-- Theorem stating the value of Claire's gift card -/
theorem claire_gift_card_value :
  gift_card_value = 
    (latte_cost + croissant_cost) * days + 
    cookie_cost * num_cookies + 
    amount_left :=
by sorry

end claire_gift_card_value_l1446_144654


namespace difference_of_squares_simplification_l1446_144650

theorem difference_of_squares_simplification : (164^2 - 148^2) / 16 = 312 := by
  sorry

end difference_of_squares_simplification_l1446_144650


namespace number_difference_l1446_144677

theorem number_difference (x y : ℤ) (h1 : x + y = 62) (h2 : y = 25) : |x - y| = 12 := by
  sorry

end number_difference_l1446_144677


namespace rectangles_on_4x4_grid_l1446_144635

/-- The number of rectangles on a 4x4 grid with sides parallel to axes -/
def num_rectangles : ℕ := 36

/-- The number of ways to choose 2 items from 4 -/
def choose_two_from_four : ℕ := 6

theorem rectangles_on_4x4_grid :
  num_rectangles = choose_two_from_four * choose_two_from_four :=
by sorry

end rectangles_on_4x4_grid_l1446_144635


namespace number_of_divisors_30030_l1446_144612

def number_to_factorize : Nat := 30030

/-- The number of positive divisors of 30030 is 64 -/
theorem number_of_divisors_30030 : 
  (Nat.divisors number_to_factorize).card = 64 := by
  sorry

end number_of_divisors_30030_l1446_144612


namespace base8_addition_l1446_144697

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Addition in base-8 --/
def add_base8 (a b c : ℕ) : ℕ :=
  base10_to_base8 (base8_to_base10 a + base8_to_base10 b + base8_to_base10 c)

theorem base8_addition :
  add_base8 246 573 62 = 1123 := by sorry

end base8_addition_l1446_144697


namespace solution_set_quadratic_inequality_l1446_144655

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by
sorry

end solution_set_quadratic_inequality_l1446_144655


namespace well_digging_cost_l1446_144690

/-- The cost of digging a cylindrical well -/
theorem well_digging_cost (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : 
  depth = 14 → diameter = 3 → cost_per_cubic_meter = 16 →
  ∃ (total_cost : ℝ), abs (total_cost - 1584.24) < 0.01 ∧ 
  total_cost = cost_per_cubic_meter * Real.pi * (diameter / 2)^2 * depth := by
sorry

end well_digging_cost_l1446_144690


namespace sugar_recipe_problem_l1446_144605

/-- The number of recipes that can be accommodated given a certain amount of sugar and recipe requirement -/
def recipes_accommodated (total_sugar : ℚ) (sugar_per_recipe : ℚ) : ℚ :=
  total_sugar / sugar_per_recipe

/-- The problem statement -/
theorem sugar_recipe_problem :
  let total_sugar : ℚ := 56 / 3  -- 18⅔ cups
  let sugar_per_recipe : ℚ := 3 / 2  -- 1½ cups
  recipes_accommodated total_sugar sugar_per_recipe = 112 / 9 :=
by
  sorry

#eval (112 : ℚ) / 9  -- Should output 12⁴⁄₉

end sugar_recipe_problem_l1446_144605


namespace cars_in_parking_lot_l1446_144663

theorem cars_in_parking_lot (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 48) (h2 : wheels_per_car = 4) :
  total_wheels / wheels_per_car = 12 := by
  sorry

end cars_in_parking_lot_l1446_144663


namespace betty_books_l1446_144646

theorem betty_books : ∀ (b : ℕ), 
  (b + (b + b / 4) = 45) → b = 20 := by
  sorry

end betty_books_l1446_144646


namespace line_and_symmetric_point_l1446_144658

/-- Given a line with inclination angle 135° passing through (1,1), 
    prove its equation and find the symmetric point of (3,4) with respect to it. -/
theorem line_and_symmetric_point :
  let l : Set (ℝ × ℝ) := {(x, y) | x + y - 2 = 0}
  let P : ℝ × ℝ := (1, 1)
  let A : ℝ × ℝ := (3, 4)
  let inclination_angle : ℝ := 135 * (π / 180)
  -- Line l passes through P
  (P ∈ l) →
  -- The slope of l is tan(135°)
  (∀ (x y : ℝ), (x, y) ∈ l → y - P.2 = Real.tan inclination_angle * (x - P.1)) →
  -- The equation of l is x + y - 2 = 0
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + y - 2 = 0) ∧
  -- The symmetric point A' of A with respect to l has coordinates (-2, -1)
  (∃ (A' : ℝ × ℝ), 
    -- A' is on the opposite side of l from A
    (A'.1 + A'.2 - 2) * (A.1 + A.2 - 2) < 0 ∧
    -- The midpoint of AA' is on l
    ((A.1 + A'.1) / 2 + (A.2 + A'.2) / 2 - 2 = 0) ∧
    -- AA' is perpendicular to l
    ((A'.2 - A.2) / (A'.1 - A.1)) * Real.tan inclination_angle = -1 ∧
    -- A' has coordinates (-2, -1)
    A' = (-2, -1)) := by
  sorry

end line_and_symmetric_point_l1446_144658


namespace frame_width_l1446_144665

/-- Given a frame with three square photo openings, this theorem proves that
    the width of the frame is 5 cm under the specified conditions. -/
theorem frame_width (s : ℝ) (d : ℝ) : 
  s > 0 →  -- side length of square opening is positive
  d > 0 →  -- frame width is positive
  4 * s = 60 →  -- perimeter of one photo opening
  2 * ((3 * s + 4 * d) + (s + 2 * d)) = 180 →  -- total perimeter of the frame
  d = 5 := by
  sorry

end frame_width_l1446_144665


namespace triangle_side_length_l1446_144676

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- Angle B is 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given condition
  (b = 2 * Real.sqrt 2) :=  -- Side length b is 2√2
by sorry

end triangle_side_length_l1446_144676


namespace chicken_egg_production_l1446_144629

/-- Given that 6 chickens lay 30 eggs in 5 days, prove that 10 chickens will lay 80 eggs in 8 days. -/
theorem chicken_egg_production 
  (initial_chickens : ℕ) 
  (initial_eggs : ℕ) 
  (initial_days : ℕ)
  (new_chickens : ℕ) 
  (new_days : ℕ)
  (h1 : initial_chickens = 6)
  (h2 : initial_eggs = 30)
  (h3 : initial_days = 5)
  (h4 : new_chickens = 10)
  (h5 : new_days = 8) :
  (new_chickens * new_days * initial_eggs) / (initial_chickens * initial_days) = 80 :=
by sorry

end chicken_egg_production_l1446_144629


namespace circle_through_M_same_center_as_C_l1446_144689

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 11 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Define the equation of the circle we want to prove
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 13

-- State the theorem
theorem circle_through_M_same_center_as_C :
  ∀ (x y : ℝ),
  (∃ (h k r : ℝ), ∀ (u v : ℝ), circle_C u v ↔ (u - h)^2 + (v - k)^2 = r^2) →
  circle_equation point_M.1 point_M.2 ∧
  (∀ (u v : ℝ), circle_C u v ↔ circle_equation u v) :=
sorry

end circle_through_M_same_center_as_C_l1446_144689


namespace parallel_line_correct_perpendicular_bisector_correct_l1446_144694

-- Define the points
def P : ℝ × ℝ := (-1, 3)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the parallel line through P
def parallel_line (x y : ℝ) : Prop := x - 2*y + 7 = 0

-- Define the perpendicular bisector of AB
def perpendicular_bisector (x y : ℝ) : Prop := 4*x - 2*y - 5 = 0

-- Theorem 1: The parallel line passes through P and is parallel to the original line
theorem parallel_line_correct :
  parallel_line P.1 P.2 ∧
  ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), original_line (x + k) (y + k/2) :=
sorry

-- Theorem 2: The perpendicular bisector is correct
theorem perpendicular_bisector_correct :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  perpendicular_bisector midpoint.1 midpoint.2 ∧
  (B.2 - A.2) * (B.1 - A.1) * 2 = -1 :=
sorry

end parallel_line_correct_perpendicular_bisector_correct_l1446_144694


namespace tenth_term_is_512_l1446_144672

/-- A sequence where each term is twice the previous term, starting with 1 -/
def doubling_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * doubling_sequence n

/-- The 10th term of the doubling sequence is 512 -/
theorem tenth_term_is_512 : doubling_sequence 9 = 512 := by
  sorry

end tenth_term_is_512_l1446_144672


namespace cube_coverage_l1446_144696

/-- Represents a rectangular strip of size 1 × 2 -/
structure Rectangle where
  length : Nat
  width : Nat

/-- Represents a cube of size n × n × n -/
structure Cube where
  size : Nat

/-- Predicate to check if a rectangle abuts exactly five others -/
def abutsFiveOthers (r : Rectangle) : Prop :=
  sorry

/-- Predicate to check if a cube's surface can be covered with rectangles -/
def canBeCovered (c : Cube) (r : Rectangle) : Prop :=
  sorry

theorem cube_coverage (n : Nat) :
  (∃ c : Cube, c.size = n ∧ ∃ r : Rectangle, r.length = 2 ∧ r.width = 1 ∧
    canBeCovered c r ∧ abutsFiveOthers r) ↔ Even n :=
sorry

end cube_coverage_l1446_144696


namespace equation_solution_l1446_144628

theorem equation_solution : ∃ x : ℚ, 300 * 2 + (12 + 4) * x / 8 = 602 :=
  by
    use 1
    sorry

#check equation_solution

end equation_solution_l1446_144628


namespace area_between_circles_and_xaxis_l1446_144649

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_between_circles_and_xaxis :
  let c1 : Circle := { center := (3, 3), radius := 3 }
  let c2 : Circle := { center := (9, 3), radius := 3 }
  areaRegion c1 c2 = 18 - (9 * Real.pi / 2) := by
  sorry

end area_between_circles_and_xaxis_l1446_144649


namespace hyperbola_standard_equation_l1446_144670

/-- The standard equation of a hyperbola with given foci and asymptotes -/
theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) →
  (a^2 + b^2 = 10) →
  (b / a = 1 / 2) →
  (a^2 = 8 ∧ b^2 = 2) :=
by sorry

end hyperbola_standard_equation_l1446_144670


namespace number_problem_l1446_144641

theorem number_problem (x : ℝ) : 0.4 * x - 30 = 50 → x = 200 := by
  sorry

end number_problem_l1446_144641


namespace simplify_sqrt_sum_l1446_144627

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l1446_144627


namespace latus_rectum_of_parabola_l1446_144680

/-- Given a parabola with equation y^2 = 8x, prove that its latus rectum has equation x = -2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y^2 = 8*x → (∃ (a : ℝ), a = -2 ∧ ∀ (x₀ y₀ : ℝ), y₀^2 = 8*x₀ → x₀ = a → 
    (x₀, y₀) ∈ {p : ℝ × ℝ | p.1 = a ∧ p.2^2 = 8*p.1}) :=
by sorry

end latus_rectum_of_parabola_l1446_144680


namespace rectangular_field_area_l1446_144630

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 105 rupees at 25 paise per meter has an area of 10800 square meters -/
theorem rectangular_field_area (length width : ℝ) (fencing_cost : ℝ) : 
  length / width = 4 / 3 →
  fencing_cost = 105 →
  (2 * (length + width)) * 0.25 = fencing_cost * 100 →
  length * width = 10800 := by
  sorry

end rectangular_field_area_l1446_144630


namespace complex_quadratic_roots_l1446_144623

theorem complex_quadratic_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = Complex.I * 2 ∧ 
  z₂ = -2 - Complex.I * 2 ∧ 
  z₁^2 + 2*z₁ = -3 + Complex.I * 4 ∧
  z₂^2 + 2*z₂ = -3 + Complex.I * 4 := by
sorry

end complex_quadratic_roots_l1446_144623


namespace cube_sum_reciprocal_l1446_144660

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cube_sum_reciprocal_l1446_144660


namespace min_distance_sum_parabola_to_lines_l1446_144631

/-- The minimum sum of distances from a point on the parabola y^2 = 4x to two lines -/
theorem min_distance_sum_parabola_to_lines : 
  let l₁ := {(x, y) : ℝ × ℝ | 4 * x - 3 * y + 6 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x = -1}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4 * x}
  let dist_to_l₁ (a : ℝ) := |4 * a^2 - 6 * a + 6| / 5
  let dist_to_l₂ (a : ℝ) := |a^2 + 1|
  ∃ (min_dist : ℝ), min_dist = 2 ∧ 
    ∀ (a : ℝ), (dist_to_l₁ a + dist_to_l₂ a) ≥ min_dist :=
by sorry


end min_distance_sum_parabola_to_lines_l1446_144631


namespace min_lcm_x_z_l1446_144651

theorem min_lcm_x_z (x y z : ℕ) (h1 : Nat.lcm x y = 18) (h2 : Nat.lcm y z = 20) :
  ∃ (x' z' : ℕ), Nat.lcm x' z' = 90 ∧ ∀ (x'' z'' : ℕ), 
    Nat.lcm x'' y = 18 → Nat.lcm y z'' = 20 → Nat.lcm x'' z'' ≥ 90 := by
  sorry

end min_lcm_x_z_l1446_144651


namespace function_parity_l1446_144620

noncomputable def f (x : ℝ) : ℝ := 1/x - 3^x
noncomputable def g (x : ℝ) : ℝ := 2^x - 2^(-x)
def h (x : ℝ) : ℝ := x^2 + |x|
noncomputable def k (x : ℝ) : ℝ := Real.log ((x+1)/(x-1))

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem function_parity :
  (¬ is_odd f ∧ ¬ is_even f) ∧
  (is_odd g ∨ is_even g) ∧
  (is_odd h ∨ is_even h) ∧
  (is_odd k ∨ is_even k) :=
sorry

end function_parity_l1446_144620


namespace not_perfect_square_l1446_144604

theorem not_perfect_square : 
  (∃ x : ℝ, (6:ℝ)^210 = x^2) ∧
  (∀ x : ℝ, (7:ℝ)^301 ≠ x^2) ∧
  (∃ x : ℝ, (8:ℝ)^402 = x^2) ∧
  (∃ x : ℝ, (9:ℝ)^302 = x^2) ∧
  (∃ x : ℝ, (10:ℝ)^404 = x^2) :=
by sorry

end not_perfect_square_l1446_144604


namespace negation_of_existence_l1446_144673

theorem negation_of_existence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) := by sorry

end negation_of_existence_l1446_144673


namespace power_function_m_equals_four_l1446_144610

/-- A function f is a power function if it has the form f(x) = ax^b where a and b are constants and a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ b

/-- Given f(x) = (m^2 - 3m - 3)x^(√m) is a power function, prove that m = 4 -/
theorem power_function_m_equals_four (m : ℝ) 
  (h : is_power_function (fun x ↦ (m^2 - 3*m - 3) * x^(Real.sqrt m))) : 
  m = 4 := by
  sorry


end power_function_m_equals_four_l1446_144610


namespace quadratic_function_condition_l1446_144667

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x^2 + m

-- Theorem statement
theorem quadratic_function_condition (m : ℝ) : 
  (∀ x, ∃ a b c, f m x = a * x^2 + b * x + c ∧ a ≠ 0) ↔ m = 1 := by
  sorry

end quadratic_function_condition_l1446_144667


namespace triangle_perimeter_l1446_144640

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  b = 9 →
  a = 2 * c →
  B = π / 3 →
  a + b + c = 9 + 9 * Real.sqrt 3 :=
by sorry

end triangle_perimeter_l1446_144640


namespace f_min_at_neg_15_div_2_f_unique_min_at_neg_15_div_2_l1446_144602

/-- The quadratic function f(x) = x^2 + 15x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 15*x + 3

/-- Theorem stating that f(x) is minimized when x = -15/2 -/
theorem f_min_at_neg_15_div_2 :
  ∀ x : ℝ, f (-15/2) ≤ f x :=
by
  sorry

/-- Theorem stating that -15/2 is the unique minimizer of f(x) -/
theorem f_unique_min_at_neg_15_div_2 :
  ∀ x : ℝ, x ≠ -15/2 → f (-15/2) < f x :=
by
  sorry

end f_min_at_neg_15_div_2_f_unique_min_at_neg_15_div_2_l1446_144602


namespace curve_C_range_l1446_144637

/-- The curve C is defined by the equation x^2 + y^2 + 2ax - 4ay + 5a^2 - 4 = 0 -/
def C (a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem: If all points on curve C are in the second quadrant, then a > 2 -/
theorem curve_C_range (a : ℝ) :
  (∀ x y : ℝ, C a x y → second_quadrant x y) → a > 2 := by
  sorry

end curve_C_range_l1446_144637


namespace andrea_rhinestone_ratio_l1446_144606

/-- Proves that the ratio of rhinestones Andrea bought to the total rhinestones needed is 1:3 -/
theorem andrea_rhinestone_ratio :
  let total_needed : ℕ := 45
  let found_in_supplies : ℕ := total_needed / 5
  let still_needed : ℕ := 21
  let bought : ℕ := total_needed - found_in_supplies - still_needed
  (bought : ℚ) / total_needed = 1 / 3 := by
  sorry

end andrea_rhinestone_ratio_l1446_144606


namespace shopkeeper_oranges_l1446_144618

/-- The number of oranges bought by a shopkeeper -/
def oranges : ℕ := sorry

/-- The number of bananas bought by the shopkeeper -/
def bananas : ℕ := 400

/-- The percentage of oranges that are not rotten -/
def good_orange_percentage : ℚ := 85 / 100

/-- The percentage of bananas that are not rotten -/
def good_banana_percentage : ℚ := 92 / 100

/-- The percentage of all fruits that are in good condition -/
def total_good_percentage : ℚ := 878 / 1000

theorem shopkeeper_oranges :
  (↑oranges * good_orange_percentage + ↑bananas * good_banana_percentage) / (↑oranges + ↑bananas) = total_good_percentage ∧
  oranges = 600 := by sorry

end shopkeeper_oranges_l1446_144618


namespace rational_function_with_infinite_integer_values_is_polynomial_l1446_144659

/-- A rational function is a quotient of two real polynomials -/
def RationalFunction (f : ℝ → ℝ) : Prop :=
  ∃ p q : Polynomial ℝ, q ≠ 0 ∧ ∀ x, f x = (p.eval x) / (q.eval x)

/-- A function that takes integer values at infinitely many integer points -/
def IntegerValuesAtInfinitelyManyPoints (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m > n, ∃ k : ℤ, f k = m

/-- Main theorem: If f is a rational function and takes integer values at infinitely many
    integer points, then f is a polynomial -/
theorem rational_function_with_infinite_integer_values_is_polynomial
  (f : ℝ → ℝ) (hf : RationalFunction f) (hi : IntegerValuesAtInfinitelyManyPoints f) :
  ∃ p : Polynomial ℝ, ∀ x, f x = p.eval x :=
sorry

end rational_function_with_infinite_integer_values_is_polynomial_l1446_144659


namespace domino_partition_exists_l1446_144668

/-- Represents a domino piece with two numbers -/
structure Domino :=
  (a b : Nat)
  (h1 : a ≤ 6)
  (h2 : b ≤ 6)

/-- The set of all domino pieces in a standard double-six set -/
def dominoSet : Finset Domino :=
  sorry

/-- The sum of points on all domino pieces -/
def totalSum : Nat :=
  sorry

/-- A partition of the domino set into 4 groups -/
def Partition := Fin 4 → Finset Domino

theorem domino_partition_exists :
  ∃ (p : Partition),
    (∀ i j, i ≠ j → Disjoint (p i) (p j)) ∧
    (∀ i, (p i).sum (λ d => d.a + d.b) = 21) ∧
    (∀ d ∈ dominoSet, ∃ i, d ∈ p i) :=
  sorry

end domino_partition_exists_l1446_144668


namespace solution_set_f_non_empty_solution_set_l1446_144688

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 7| + 3*m

-- Theorem 1: Solution set of f(x) + x^2 - 4 > 0
theorem solution_set_f (x : ℝ) : f x + x^2 - 4 > 0 ↔ x > 2 ∨ x < -1 := by sorry

-- Theorem 2: Condition for non-empty solution set of f(x) < g(x)
theorem non_empty_solution_set (m : ℝ) :
  (∃ x : ℝ, f x < g m x) ↔ m > 3 := by sorry

end solution_set_f_non_empty_solution_set_l1446_144688


namespace simplify_expression_l1446_144633

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end simplify_expression_l1446_144633


namespace layla_fish_food_total_l1446_144687

/-- The total amount of food Layla needs to give her fish -/
def total_fish_food (goldfish_count : ℕ) (goldfish_food : ℚ) 
                    (swordtail_count : ℕ) (swordtail_food : ℚ) 
                    (guppy_count : ℕ) (guppy_food : ℚ) : ℚ :=
  goldfish_count * goldfish_food + swordtail_count * swordtail_food + guppy_count * guppy_food

/-- Theorem stating the total amount of food Layla needs to give her fish -/
theorem layla_fish_food_total : 
  total_fish_food 2 1 3 2 8 (1/2) = 12 := by
  sorry

end layla_fish_food_total_l1446_144687


namespace fayes_initial_money_l1446_144691

/-- Proves that Faye's initial amount of money was $20 --/
theorem fayes_initial_money :
  ∀ (X : ℝ),
  (X + 2*X - (10*1.5 + 5*3) = 30) →
  X = 20 :=
by
  sorry

end fayes_initial_money_l1446_144691


namespace probability_of_two_in_pascal_triangle_l1446_144600

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the occurrences of a specific number in Pascal's Triangle -/
def countOccurrences (triangle : List (List ℕ)) (target : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of elements in Pascal's Triangle -/
def totalElements (triangle : List (List ℕ)) : ℕ :=
  sorry

/-- The main theorem: probability of selecting 2 from first 20 rows of Pascal's Triangle -/
theorem probability_of_two_in_pascal_triangle :
  let triangle := PascalTriangle 20
  let occurrences := countOccurrences triangle 2
  let total := totalElements triangle
  (occurrences : ℚ) / total = 6 / 35 := by
  sorry

end probability_of_two_in_pascal_triangle_l1446_144600


namespace quadratic_minimum_at_positive_x_l1446_144614

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem quadratic_minimum_at_positive_x :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f y ≥ f x :=
sorry

end quadratic_minimum_at_positive_x_l1446_144614


namespace molecular_weight_BaF2_is_175_l1446_144674

/-- The molecular weight of BaF2 in grams per mole. -/
def molecular_weight_BaF2 : ℝ := 175

/-- The number of moles of BaF2 in the given condition. -/
def moles_BaF2 : ℝ := 6

/-- The total weight of the given moles of BaF2 in grams. -/
def total_weight_BaF2 : ℝ := 1050

/-- Theorem stating that the molecular weight of BaF2 is 175 grams/mole. -/
theorem molecular_weight_BaF2_is_175 :
  molecular_weight_BaF2 = total_weight_BaF2 / moles_BaF2 :=
by sorry

end molecular_weight_BaF2_is_175_l1446_144674


namespace binomial_expansion_term_sum_l1446_144657

theorem binomial_expansion_term_sum (n : ℕ) (b : ℝ) : 
  n ≥ 2 → 
  b ≠ 0 → 
  (Nat.choose n 3 : ℝ) * b^(n-3) + (Nat.choose n 4 : ℝ) * b^(n-4) = 0 → 
  n = 4 := by
sorry

end binomial_expansion_term_sum_l1446_144657


namespace janet_income_difference_l1446_144695

/-- Calculates the difference in monthly income between freelancing and current job for Janet --/
theorem janet_income_difference :
  let hours_per_week : ℕ := 40
  let weeks_per_month : ℕ := 4
  let current_hourly_rate : ℚ := 30
  let freelance_hourly_rate : ℚ := 40
  let extra_fica_per_week : ℚ := 25
  let healthcare_premium_per_month : ℚ := 400

  let current_monthly_income : ℚ := hours_per_week * weeks_per_month * current_hourly_rate
  let freelance_gross_monthly_income : ℚ := hours_per_week * weeks_per_month * freelance_hourly_rate
  let additional_monthly_costs : ℚ := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month
  let freelance_net_monthly_income : ℚ := freelance_gross_monthly_income - additional_monthly_costs

  freelance_net_monthly_income - current_monthly_income = 1100 := by
  sorry

end janet_income_difference_l1446_144695


namespace wooden_block_length_l1446_144653

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Define the initial length in meters
def initial_length_m : ℝ := 31

-- Define the additional length in centimeters
def additional_length_cm : ℝ := 30

-- Theorem to prove
theorem wooden_block_length :
  (initial_length_m * meters_to_cm + additional_length_cm) = 3130 := by
  sorry

end wooden_block_length_l1446_144653


namespace isosceles_triangle_base_length_l1446_144678

/-- An isosceles triangle with congruent sides of length 8 cm and perimeter of 26 cm has a base of length 10 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side perimeter : ℝ),
  congruent_side = 8 →
  perimeter = 26 →
  perimeter = 2 * congruent_side + base →
  base = 10 := by
sorry

end isosceles_triangle_base_length_l1446_144678


namespace flag_distribution_l1446_144666

theorem flag_distribution (F : ℕ) (blue_flags red_flags : ℕ) :
  F % 2 = 0 →
  F = blue_flags + red_flags →
  blue_flags ≥ (3 * F) / 10 →
  red_flags ≥ F / 4 →
  (F / 2 - (3 * F) / 10 - F / 4) / (F / 2) = 1 / 10 :=
by sorry

end flag_distribution_l1446_144666


namespace hex_conversion_sum_l1446_144647

/-- Converts a hexadecimal number to decimal --/
def hex_to_decimal (hex : String) : ℕ := sorry

/-- Converts a decimal number to radix 7 --/
def decimal_to_radix7 (n : ℕ) : String := sorry

/-- Converts a radix 7 number to decimal --/
def radix7_to_decimal (r7 : String) : ℕ := sorry

/-- Converts a decimal number to hexadecimal --/
def decimal_to_hex (n : ℕ) : String := sorry

/-- Adds two hexadecimal numbers and returns the result in hexadecimal --/
def add_hex (hex1 : String) (hex2 : String) : String := sorry

theorem hex_conversion_sum :
  let initial_hex := "E78"
  let decimal := hex_to_decimal initial_hex
  let radix7 := decimal_to_radix7 decimal
  let back_to_decimal := radix7_to_decimal radix7
  let final_hex := decimal_to_hex back_to_decimal
  add_hex initial_hex final_hex = "1CF0" := by sorry

end hex_conversion_sum_l1446_144647


namespace sqrt_of_sqrt_81_l1446_144684

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by sorry

end sqrt_of_sqrt_81_l1446_144684


namespace inequality_multiplication_l1446_144685

theorem inequality_multiplication (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end inequality_multiplication_l1446_144685


namespace altitude_segment_length_l1446_144669

/-- Represents an acute triangle with two altitudes dividing the sides. -/
structure AcuteTriangleWithAltitudes where
  -- The lengths of the segments created by the altitudes
  a : ℝ
  b : ℝ
  c : ℝ
  y : ℝ
  -- Conditions
  acute : a > 0 ∧ b > 0 ∧ c > 0 ∧ y > 0
  a_val : a = 7
  b_val : b = 4
  c_val : c = 3

/-- The theorem stating that y = 12/7 in the given triangle configuration. -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.y = 12/7 := by
  sorry

end altitude_segment_length_l1446_144669


namespace sufficient_not_necessary_condition_l1446_144648

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3) ∧
  (∃ a b : ℝ, a + b > 3 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end sufficient_not_necessary_condition_l1446_144648


namespace jordan_max_points_l1446_144671

structure BasketballGame where
  threePointAttempts : ℕ
  twoPointAttempts : ℕ
  freeThrowAttempts : ℕ
  threePointSuccess : ℚ
  twoPointSuccess : ℚ
  freeThrowSuccess : ℚ

def totalShots (game : BasketballGame) : ℕ :=
  game.threePointAttempts + game.twoPointAttempts + game.freeThrowAttempts

def totalPoints (game : BasketballGame) : ℚ :=
  3 * game.threePointSuccess * game.threePointAttempts +
  2 * game.twoPointSuccess * game.twoPointAttempts +
  game.freeThrowSuccess * game.freeThrowAttempts

theorem jordan_max_points :
  ∀ (game : BasketballGame),
  game.threePointSuccess = 1/4 →
  game.twoPointSuccess = 2/5 →
  game.freeThrowSuccess = 4/5 →
  totalShots game = 50 →
  totalPoints game ≤ 39 :=
by sorry

end jordan_max_points_l1446_144671


namespace prove_my_current_age_l1446_144683

/-- The age at which my dog was born -/
def age_when_dog_born : ℕ := 15

/-- The age my dog will be in two years -/
def dog_age_in_two_years : ℕ := 4

/-- My current age -/
def my_current_age : ℕ := age_when_dog_born + (dog_age_in_two_years - 2)

theorem prove_my_current_age : my_current_age = 17 := by
  sorry

end prove_my_current_age_l1446_144683


namespace number_of_books_a_l1446_144634

/-- Proves that the number of books (a) is 12, given the conditions -/
theorem number_of_books_a (total : ℕ) (diff : ℕ) : 
  (total = 20) → (diff = 4) → ∃ (a b : ℕ), (a + b = total) ∧ (a = b + diff) ∧ (a = 12) :=
by
  sorry

end number_of_books_a_l1446_144634


namespace biggest_collection_l1446_144638

def yoongi_collection : ℕ := 4
def jungkook_collection : ℕ := 6 * 3
def yuna_collection : ℕ := 5

theorem biggest_collection :
  max yoongi_collection (max jungkook_collection yuna_collection) = jungkook_collection :=
by sorry

end biggest_collection_l1446_144638


namespace total_boys_count_l1446_144661

theorem total_boys_count (average_all : ℝ) (average_passed : ℝ) (average_failed : ℝ) (passed_count : ℕ) :
  average_all = 37 →
  average_passed = 39 →
  average_failed = 15 →
  passed_count = 110 →
  ∃ (total_count : ℕ), 
    total_count = passed_count + (total_count - passed_count) ∧
    (average_all * total_count : ℝ) = average_passed * passed_count + average_failed * (total_count - passed_count) ∧
    total_count = 120 :=
by
  sorry

end total_boys_count_l1446_144661


namespace min_x_prime_factorization_l1446_144615

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    x ≥ 13^6 * 5^7 ∧
    (∀ (x' : ℕ+) (a' b' c' d' : ℕ), 5 * x'^7 = 13 * y^11 → x' = a'^c' * b'^d' → x' ≥ x) ∧
    a + b + c + d = 31 :=
by sorry

end min_x_prime_factorization_l1446_144615


namespace initial_number_of_persons_l1446_144639

theorem initial_number_of_persons (n : ℕ) 
  (h1 : (3.5 : ℝ) * n = 28)
  (h2 : (90 : ℝ) - 62 = 28) : 
  n = 8 := by
  sorry

end initial_number_of_persons_l1446_144639


namespace special_function_properties_l1446_144642

def I : Set ℝ := Set.Icc (-1) 1

structure SpecialFunction (f : ℝ → ℝ) : Prop where
  domain : ∀ x, x ∈ I → f x ≠ 0 → True
  additive : ∀ x y, x ∈ I → y ∈ I → f (x + y) = f x + f y
  positive : ∀ x, x > 0 → x ∈ I → f x > 0

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, x ∈ I → f (-x) = -f x) ∧
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) :=
by sorry

end special_function_properties_l1446_144642


namespace jenny_ate_65_squares_l1446_144682

/-- The number of chocolate squares Mike ate -/
def mike_squares : ℕ := 20

/-- The number of chocolate squares Jenny ate -/
def jenny_squares : ℕ := 3 * mike_squares + 5

/-- Theorem stating that Jenny ate 65 chocolate squares -/
theorem jenny_ate_65_squares : jenny_squares = 65 := by
  sorry

end jenny_ate_65_squares_l1446_144682


namespace mathematics_players_l1446_144619

/-- Theorem: Number of players taking mathematics in Riverdale Academy volleyball team -/
theorem mathematics_players (total : ℕ) (physics : ℕ) (both : ℕ) : 
  total = 15 → physics = 9 → both = 4 → (∃ (math : ℕ), math = 10) :=
by sorry

end mathematics_players_l1446_144619


namespace cos_seven_pi_sixths_l1446_144601

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l1446_144601


namespace students_passed_both_tests_l1446_144645

theorem students_passed_both_tests
  (total : ℕ)
  (passed_chinese : ℕ)
  (passed_english : ℕ)
  (failed_both : ℕ)
  (h1 : total = 50)
  (h2 : passed_chinese = 40)
  (h3 : passed_english = 31)
  (h4 : failed_both = 4) :
  total - failed_both = passed_chinese + passed_english - (passed_chinese + passed_english - (total - failed_both)) :=
by sorry

end students_passed_both_tests_l1446_144645


namespace no_three_intersections_l1446_144625

-- Define a circle in Euclidean space
structure EuclideanCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an intersection point
def IntersectionPoint (c1 c2 : EuclideanCircle) := 
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2}

-- Theorem statement
theorem no_three_intersections 
  (c1 c2 : EuclideanCircle) 
  (h_distinct : c1 ≠ c2) : 
  ¬∃ (p1 p2 p3 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ IntersectionPoint c1 c2 ∧
    p2 ∈ IntersectionPoint c1 c2 ∧
    p3 ∈ IntersectionPoint c1 c2 :=
sorry

end no_three_intersections_l1446_144625


namespace unique_divisible_by_18_l1446_144613

/-- Represents a four-digit number in the form x28x --/
def fourDigitNumber (x : ℕ) : ℕ := x * 1000 + 280 + x

/-- Checks if a natural number is a single digit (0-9) --/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem unique_divisible_by_18 :
  ∃! x : ℕ, isSingleDigit x ∧ (fourDigitNumber x % 18 = 0) ∧ x = 4 := by sorry

end unique_divisible_by_18_l1446_144613


namespace three_integers_problem_l1446_144621

theorem three_integers_problem :
  ∃ (x y z : ℤ),
    (x + y) / 2 + z = 42 ∧
    (y + z) / 2 + x = 13 ∧
    (x + z) / 2 + y = 37 := by
  sorry

end three_integers_problem_l1446_144621


namespace max_value_quadratic_sum_l1446_144617

theorem max_value_quadratic_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - x*y + y^2 = 9) : 
  x^2 + x*y + y^2 ≤ 27 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 - a*b + b^2 = 9 ∧ a^2 + a*b + b^2 = 27 := by
  sorry

end max_value_quadratic_sum_l1446_144617


namespace felix_tree_chopping_l1446_144664

theorem felix_tree_chopping (trees_per_sharpen : ℕ) (sharpen_cost : ℕ) (total_spent : ℕ) : 
  trees_per_sharpen = 13 → 
  sharpen_cost = 5 → 
  total_spent = 35 → 
  ∃ (trees_chopped : ℕ), trees_chopped ≥ 91 ∧ trees_chopped ≥ (total_spent / sharpen_cost) * trees_per_sharpen :=
by
  sorry

end felix_tree_chopping_l1446_144664


namespace book_price_calculation_l1446_144608

theorem book_price_calculation (P : ℝ) : 
  P * 0.85 * 1.40 = 476 → P = 400 := by
  sorry

end book_price_calculation_l1446_144608


namespace min_perimeter_special_triangle_l1446_144607

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem min_perimeter_special_triangle :
  ∃ (c : ℕ), 
    is_valid_triangle 24 51 c ∧ 
    (∀ (x : ℕ), is_valid_triangle 24 51 x → triangle_perimeter 24 51 c ≤ triangle_perimeter 24 51 x) ∧
    triangle_perimeter 24 51 c = 103 :=
by sorry

end min_perimeter_special_triangle_l1446_144607


namespace last_two_digits_of_seven_power_l1446_144699

theorem last_two_digits_of_seven_power (n : ℕ) : 7^(5^6) ≡ 7 [MOD 100] := by
  sorry

end last_two_digits_of_seven_power_l1446_144699


namespace addition_preserves_inequality_l1446_144616

theorem addition_preserves_inequality (a b c d : ℝ) : 
  a < b → c < d → a + c < b + d := by
  sorry

end addition_preserves_inequality_l1446_144616


namespace fraction_inequality_solution_set_l1446_144686

theorem fraction_inequality_solution_set :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by sorry

end fraction_inequality_solution_set_l1446_144686


namespace cube_root_inequality_l1446_144679

theorem cube_root_inequality (x : ℝ) : 
  x > 0 → (x^(1/3) < 3*x ↔ x > 1/(3*Real.sqrt 3)) := by sorry

end cube_root_inequality_l1446_144679


namespace fraction_problem_l1446_144693

theorem fraction_problem (f : ℝ) : f * 50.0 - 4 = 6 → f = 0.2 := by
  sorry

end fraction_problem_l1446_144693


namespace sine_of_sum_inverse_sine_and_tangent_l1446_144624

theorem sine_of_sum_inverse_sine_and_tangent :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sine_of_sum_inverse_sine_and_tangent_l1446_144624


namespace total_cinnamon_swirls_l1446_144656

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- Theorem: If there are 3 people eating an equal number of cinnamon swirls, 
    and one person ate 4 pieces, then the total number of pieces is 12. -/
theorem total_cinnamon_swirls : 
  num_people * janes_pieces = 12 := by sorry

end total_cinnamon_swirls_l1446_144656


namespace distance_from_origin_of_complex_fraction_l1446_144636

theorem distance_from_origin_of_complex_fraction : 
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end distance_from_origin_of_complex_fraction_l1446_144636


namespace total_paint_used_l1446_144611

/-- The amount of paint Joe uses at two airports over two weeks -/
def paint_used (paint1 paint2 : ℝ) (week1_ratio1 week2_ratio1 week1_ratio2 week2_ratio2 : ℝ) : ℝ :=
  let remaining1 := paint1 * (1 - week1_ratio1)
  let used1 := paint1 * week1_ratio1 + remaining1 * week2_ratio1
  let remaining2 := paint2 * (1 - week1_ratio2)
  let used2 := paint2 * week1_ratio2 + remaining2 * week2_ratio2
  used1 + used2

/-- Theorem stating the total amount of paint Joe uses at both airports -/
theorem total_paint_used :
  paint_used 360 600 (1/4) (1/6) (1/3) (1/5) = 415 := by
  sorry

end total_paint_used_l1446_144611


namespace prime_divisor_of_fermat_number_l1446_144603

theorem prime_divisor_of_fermat_number (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ 2^(2^n) + 1) : 2^(n+1) ∣ p - 1 := by
  sorry

end prime_divisor_of_fermat_number_l1446_144603


namespace gain_percent_when_cost_equals_sell_l1446_144652

/-- Proves that if the cost price of 50 articles equals the selling price of 25 articles, 
    then the gain percent is 100%. -/
theorem gain_percent_when_cost_equals_sell (C S : ℝ) 
  (h : 50 * C = 25 * S) : (S - C) / C * 100 = 100 := by
  sorry

end gain_percent_when_cost_equals_sell_l1446_144652


namespace smallest_three_digit_multiple_of_13_l1446_144626

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n → n ≥ 104 :=
by sorry

end smallest_three_digit_multiple_of_13_l1446_144626


namespace cube_split_73_l1446_144662

/-- The first "split number" of m^3 -/
def firstSplitNumber (m : ℕ) : ℕ := m^2 - m + 1

/-- Predicate to check if a number is one of the "split numbers" of m^3 -/
def isSplitNumber (m : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < m ∧ n = firstSplitNumber m + 2 * k

theorem cube_split_73 (m : ℕ) (h1 : m > 1) (h2 : isSplitNumber m 73) : m = 9 := by
  sorry

end cube_split_73_l1446_144662


namespace tan_15_identity_l1446_144698

theorem tan_15_identity : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_15_identity_l1446_144698


namespace min_distance_curve_line_l1446_144622

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.exp x + x

def line (x : ℝ) : ℝ := 3 * x - 1

theorem min_distance_curve_line :
  ∃ (d : ℝ), d = (3 * Real.sqrt 10) / 10 ∧
  ∀ (x₁ x₂ : ℝ), 
    let y₁ := curve x₁
    let y₂ := line x₂
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by sorry

end min_distance_curve_line_l1446_144622


namespace line_intersection_area_ratio_l1446_144675

/-- Given a line y = b - 2x where 0 < b < 6, intersecting the y-axis at P and the line x=6 at S,
    if the ratio of the area of triangle QRS to the area of triangle QOP is 4:9,
    then b = √(1296/11). -/
theorem line_intersection_area_ratio (b : ℝ) : 
  0 < b → b < 6 → 
  let line := fun x => b - 2 * x
  let P := (0, b)
  let S := (6, line 6)
  let Q := (b / 2, 0)
  let R := (6, 0)
  let area_QOP := (1 / 2) * (b / 2) * b
  let area_QRS := (1 / 2) * (6 - b / 2) * |b - 12|
  area_QRS / area_QOP = 4 / 9 →
  b = Real.sqrt (1296 / 11) := by
sorry

end line_intersection_area_ratio_l1446_144675


namespace fraction_simplification_l1446_144681

theorem fraction_simplification :
  (1 - 1/3) / (1 - 1/2) = 4/3 := by
  sorry

end fraction_simplification_l1446_144681


namespace total_arc_length_is_900_l1446_144644

/-- A triangle with its circumcircle -/
structure CircumscribedTriangle where
  /-- The radius of the circumcircle -/
  radius : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ

/-- The total length of arcs XX', YY', and ZZ' in a circumscribed triangle -/
def total_arc_length (t : CircumscribedTriangle) : ℝ := sorry

/-- Theorem: The total length of arcs XX', YY', and ZZ' is 900° -/
theorem total_arc_length_is_900 (t : CircumscribedTriangle) 
  (h1 : t.radius = 5) 
  (h2 : t.perimeter = 24) : 
  total_arc_length t = 900 := by sorry

end total_arc_length_is_900_l1446_144644


namespace meaningful_expression_range_l1446_144632

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / (x - 2)) ↔ (x ≥ -1 ∧ x ≠ 2) :=
by sorry

end meaningful_expression_range_l1446_144632


namespace expression_value_l1446_144643

theorem expression_value (x : ℝ) (h : x = -2) : (3 * x - 4)^2 = 100 := by
  sorry

end expression_value_l1446_144643


namespace min_value_product_min_value_product_achieved_l1446_144692

theorem min_value_product (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (h : 1/x + 1/y + 1/z + 1/u = 8) : 
  x^3 * y^2 * z * u^2 ≥ 1/432 :=
sorry

theorem min_value_product_achieved (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (h : 1/x + 1/y + 1/z + 1/u = 8) : 
  ∃ (x' y' z' u' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ u' > 0 ∧ 
    1/x' + 1/y' + 1/z' + 1/u' = 8 ∧ 
    x'^3 * y'^2 * z' * u'^2 = 1/432 :=
sorry

end min_value_product_min_value_product_achieved_l1446_144692


namespace decimal_132_to_binary_l1446_144609

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

-- Theorem statement
theorem decimal_132_to_binary :
  decimalToBinary 132 = [true, false, false, false, false, true, false, false] := by
  sorry

#eval decimalToBinary 132

end decimal_132_to_binary_l1446_144609
