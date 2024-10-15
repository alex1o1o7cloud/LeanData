import Mathlib

namespace NUMINAMATH_CALUDE_acute_triangle_sine_sum_l2010_201065

theorem acute_triangle_sine_sum (α β γ : Real) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_acute_triangle : α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_sum_l2010_201065


namespace NUMINAMATH_CALUDE_two_triangles_exist_l2010_201004

/-- Represents a triangle with side lengths and heights -/
structure Triangle where
  a : ℝ
  m_b : ℝ
  m_c : ℝ

/-- Given conditions for the triangle construction problem -/
def givenConditions : Triangle where
  a := 6
  m_b := 1
  m_c := 2

/-- Predicate to check if a triangle satisfies the given conditions -/
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = givenConditions.a ∧ t.m_b = givenConditions.m_b ∧ t.m_c = givenConditions.m_c

/-- Theorem stating that exactly two distinct triangles satisfy the given conditions -/
theorem two_triangles_exist :
  ∃ (t1 t2 : Triangle), satisfiesConditions t1 ∧ satisfiesConditions t2 ∧ t1 ≠ t2 ∧
  ∀ (t : Triangle), satisfiesConditions t → (t = t1 ∨ t = t2) :=
sorry

end NUMINAMATH_CALUDE_two_triangles_exist_l2010_201004


namespace NUMINAMATH_CALUDE_sequence_integer_count_l2010_201023

def sequence_term (n : ℕ) : ℚ :=
  12150 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  ∃ (k : ℕ), k = 5 ∧
  (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
  (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l2010_201023


namespace NUMINAMATH_CALUDE_james_cattle_profit_l2010_201098

/-- Represents the profit calculation for James' cattle business --/
theorem james_cattle_profit :
  let num_cattle : ℕ := 100
  let total_buying_cost : ℚ := 40000
  let buying_cost_per_cattle : ℚ := total_buying_cost / num_cattle
  let feeding_cost_per_cattle : ℚ := buying_cost_per_cattle * 1.2
  let total_feeding_cost_per_month : ℚ := feeding_cost_per_cattle * num_cattle
  let months_held : ℕ := 6
  let total_feeding_cost : ℚ := total_feeding_cost_per_month * months_held
  let total_cost : ℚ := total_buying_cost + total_feeding_cost
  let weight_per_cattle : ℕ := 1000
  let june_price_per_pound : ℚ := 2.2
  let total_selling_price : ℚ := num_cattle * weight_per_cattle * june_price_per_pound
  let profit : ℚ := total_selling_price - total_cost
  profit = -108000 := by sorry

end NUMINAMATH_CALUDE_james_cattle_profit_l2010_201098


namespace NUMINAMATH_CALUDE_green_fruits_vs_red_peaches_green_peaches_vs_yellow_apples_l2010_201076

/-- Represents the number of red peaches in the basket -/
def red_peaches : ℕ := 5

/-- Represents the number of green peaches in the basket -/
def green_peaches : ℕ := 11

/-- Represents the number of yellow apples in the basket -/
def yellow_apples : ℕ := 8

/-- Represents the number of green apples in the basket -/
def green_apples : ℕ := 15

/-- Theorem stating the difference between green fruits and red peaches -/
theorem green_fruits_vs_red_peaches : 
  green_peaches + green_apples - red_peaches = 21 := by sorry

/-- Theorem stating the difference between green peaches and yellow apples -/
theorem green_peaches_vs_yellow_apples : 
  green_peaches - yellow_apples = 3 := by sorry

end NUMINAMATH_CALUDE_green_fruits_vs_red_peaches_green_peaches_vs_yellow_apples_l2010_201076


namespace NUMINAMATH_CALUDE_intersection_distance_l2010_201057

/-- The distance between the intersection points of the line y = 1 - x and the circle x^2 + y^2 = 8 is equal to √30 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = 8) ∧ 
    (B.1^2 + B.2^2 = 8) ∧ 
    (A.2 = 1 - A.1) ∧ 
    (B.2 = 1 - B.1) ∧ 
    (A ≠ B) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 30) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2010_201057


namespace NUMINAMATH_CALUDE_product_65_55_l2010_201055

theorem product_65_55 : 65 * 55 = 3575 := by
  sorry

end NUMINAMATH_CALUDE_product_65_55_l2010_201055


namespace NUMINAMATH_CALUDE_min_value_of_f_l2010_201021

/-- The function f(x) = 4x^2 - 12x + 9 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x + 9

/-- The minimum value of f(x) is 0 -/
theorem min_value_of_f : ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2010_201021


namespace NUMINAMATH_CALUDE_board_numbers_product_l2010_201037

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {2, 6, 10, 10, 12, 14, 16, 18, 20, 24} → 
  a * b * c * d * e = -3003 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_product_l2010_201037


namespace NUMINAMATH_CALUDE_glorias_cash_was_150_l2010_201019

/-- Calculates Gloria's initial cash given the cabin cost, tree counts, tree prices, and leftover cash --/
def glorias_initial_cash (cabin_cost : ℕ) (cypress_count pine_count maple_count : ℕ) 
  (cypress_price pine_price maple_price : ℕ) (leftover_cash : ℕ) : ℕ :=
  cabin_cost + leftover_cash - (cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price)

/-- Theorem stating that Gloria's initial cash was $150 --/
theorem glorias_cash_was_150 : 
  glorias_initial_cash 129000 20 600 24 100 200 300 350 = 150 := by
  sorry


end NUMINAMATH_CALUDE_glorias_cash_was_150_l2010_201019


namespace NUMINAMATH_CALUDE_largest_circle_radius_is_two_l2010_201091

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (c, 0) and radius r -/
structure Circle where
  c : ℝ
  r : ℝ
  h_positive_r : 0 < r

/-- Returns true if the circle is entirely contained within the ellipse -/
def circleInEllipse (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, (x - c.c)^2 + y^2 = c.r^2 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- Returns true if the circle is tangent to the ellipse -/
def circleTangentToEllipse (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, (x - c.c)^2 + y^2 = c.r^2 ∧ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating that the largest circle centered at a focus of the ellipse
    and entirely contained within it has radius 2 -/
theorem largest_circle_radius_is_two (e : Ellipse) (c : Circle) 
    (h_a : e.a = 7) (h_b : e.b = 5) (h_c : c.c = 2 * Real.sqrt 6) : 
    circleInEllipse e c ∧ circleTangentToEllipse e c → c.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_radius_is_two_l2010_201091


namespace NUMINAMATH_CALUDE_sphere_surface_area_cuboid_l2010_201063

/-- The surface area of a sphere circumscribing a cuboid with dimensions 2, 1, and 1 is 6π. -/
theorem sphere_surface_area_cuboid : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    (2 : ℝ)^2 + 1^2 + 1^2 = (2*r)^2 ∧ 
    4 * Real.pi * r^2 = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_cuboid_l2010_201063


namespace NUMINAMATH_CALUDE_negation_of_negation_one_l2010_201047

theorem negation_of_negation_one : -(-1) = 1 := by sorry

end NUMINAMATH_CALUDE_negation_of_negation_one_l2010_201047


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2010_201001

/-- Given positive real numbers a, b, c satisfying the condition,
    the minimum value of the expression is 50 -/
theorem min_value_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  ∃ m : ℝ, m = 50 ∧ ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    x/y + y/z + z/x + y/x + z/y + x/z = 10 →
    (x/y + y/z + z/x)^2 + (y/x + z/y + x/z)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2010_201001


namespace NUMINAMATH_CALUDE_probability_is_zero_l2010_201078

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the picture taken of the track -/
structure Picture where
  coverageFraction : ℝ
  centerPosition : ℝ  -- Position on track (0 ≤ position < 1)

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (rachel : Runner) (robert : Runner) (pic : Picture) (timeElapsed : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability is zero for the given conditions -/
theorem probability_is_zero :
  ∀ (rachel : Runner) (robert : Runner) (pic : Picture) (t : ℝ),
    rachel.lapTime = 120 →
    robert.lapTime = 75 →
    rachel.direction = true →
    robert.direction = false →
    pic.coverageFraction = 1/3 →
    pic.centerPosition = 0 →
    15 * 60 ≤ t ∧ t < 16 * 60 →
    probabilityBothInPicture rachel robert pic t = 0 :=
  sorry

end NUMINAMATH_CALUDE_probability_is_zero_l2010_201078


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2010_201017

theorem sandy_shopping_money (initial_amount : ℝ) (spent_percentage : ℝ) (remaining_amount : ℝ) : 
  spent_percentage = 30 →
  remaining_amount = 140 →
  (1 - spent_percentage / 100) * initial_amount = remaining_amount →
  initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2010_201017


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_factorial_series_l2010_201088

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Function to get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The series in question -/
def series : List ℕ := [1, 2, 5, 13, 34]

/-- Theorem stating that the sum of the last two digits of the factorial series is 23 -/
theorem sum_of_last_two_digits_of_factorial_series : 
  (series.map (λ n => lastTwoDigits (factorial n))).sum = 23 := by sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_factorial_series_l2010_201088


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2010_201011

theorem alcohol_mixture_proof (x y z final_volume final_alcohol : ℝ) :
  x = 300 ∧ y = 600 ∧ z = 300 ∧
  (0.1 * x + 0.3 * y + 0.4 * z) / (x + y + z) = 0.22 ∧
  y = 2 * z →
  final_volume = x + y + z ∧
  final_alcohol = 0.22 * final_volume :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2010_201011


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_x_l2010_201042

theorem inequality_holds_for_all_x : ∀ x : ℝ, x + 2 < x + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_x_l2010_201042


namespace NUMINAMATH_CALUDE_disk_ratio_theorem_l2010_201027

/-- Represents a disk with a center point and a radius. -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two disks are tangent to each other. -/
def areTangent (d1 d2 : Disk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 = (d1.radius + d2.radius)^2

/-- Checks if two disks have disjoint interiors. -/
def haveDisjointInteriors (d1 d2 : Disk) : Prop :=
  (d1.center.1 - d2.center.1)^2 + (d1.center.2 - d2.center.2)^2 > (d1.radius + d2.radius)^2

theorem disk_ratio_theorem (d1 d2 d3 d4 : Disk) 
  (h_equal_size : d1.radius = d2.radius ∧ d2.radius = d3.radius)
  (h_smaller : d4.radius < d1.radius)
  (h_tangent : areTangent d1 d2 ∧ areTangent d2 d3 ∧ areTangent d3 d1 ∧ 
               areTangent d1 d4 ∧ areTangent d2 d4 ∧ areTangent d3 d4)
  (h_disjoint : haveDisjointInteriors d1 d2 ∧ haveDisjointInteriors d2 d3 ∧ 
                haveDisjointInteriors d3 d1 ∧ haveDisjointInteriors d1 d4 ∧ 
                haveDisjointInteriors d2 d4 ∧ haveDisjointInteriors d3 d4) :
  d4.radius / d1.radius = (2 * Real.sqrt 3 - 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_disk_ratio_theorem_l2010_201027


namespace NUMINAMATH_CALUDE_reinforcement_size_l2010_201033

/-- Calculates the size of a reinforcement given initial garrison size, provision duration, and new provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) (days_before_reinforcement : ℕ) (new_provisions : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provisions
  let remaining_provisions := initial_garrison * (initial_provisions - days_before_reinforcement)
  let reinforcement := (remaining_provisions / new_provisions) - initial_garrison
  reinforcement

/-- Theorem stating that given the problem conditions, the reinforcement size is 1600. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 54 18 20 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l2010_201033


namespace NUMINAMATH_CALUDE_cubic_real_root_l2010_201031

/-- The cubic equation with real coefficients c and d, having -3 - 4i as a root, has -4 as its real root -/
theorem cubic_real_root (c d : ℝ) (h : c * (-3 - 4*I)^3 + 4 * (-3 - 4*I)^2 + d * (-3 - 4*I) - 100 = 0) :
  ∃ x : ℝ, c * x^3 + 4 * x^2 + d * x - 100 = 0 ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_real_root_l2010_201031


namespace NUMINAMATH_CALUDE_fraction_inequality_l2010_201018

theorem fraction_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (a + 1)) + (b^2 / (b + 1)) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2010_201018


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2010_201032

theorem cube_root_equation_solution (y : ℝ) :
  (5 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 1/16 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2010_201032


namespace NUMINAMATH_CALUDE_max_constant_inequality_l2010_201071

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x^2 + y^2 = 1) :
  ∃ (c : ℝ), c = 1/2 ∧ x^6 + y^6 ≥ c*x*y ∧ ∀ (c' : ℝ), (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 + y'^2 = 1 → x'^6 + y'^6 ≥ c'*x'*y') → c' ≤ c :=
sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l2010_201071


namespace NUMINAMATH_CALUDE_y_value_l2010_201080

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2010_201080


namespace NUMINAMATH_CALUDE_rectangle_squares_sum_l2010_201038

theorem rectangle_squares_sum (a b : ℝ) : 
  a + b = 3 → a * b = 1 → a^2 + b^2 = 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_squares_sum_l2010_201038


namespace NUMINAMATH_CALUDE_expression_theorem_l2010_201035

-- Define the expression E as a function of x
def E (x : ℝ) : ℝ := 6 * x + 45

-- State the theorem
theorem expression_theorem (x : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₂ - x₁ = 12) →
  (E x) / (2 * x + 15) = 3 →
  E x = 6 * x + 45 := by
sorry

end NUMINAMATH_CALUDE_expression_theorem_l2010_201035


namespace NUMINAMATH_CALUDE_ellipse_equation_l2010_201008

/-- Represents an ellipse with axes aligned to the coordinate system -/
structure Ellipse where
  a : ℝ  -- Half-length of the major axis
  b : ℝ  -- Half-length of the minor axis
  c : ℝ  -- Distance from center to focus

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Theorem: Given the specified conditions, prove the standard equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) 
    (h1 : e.a + e.b = 9)  -- Sum of half-lengths of axes is 18/2 = 9
    (h2 : e.c = 3)        -- One focus is at (3, 0)
    (h3 : e.c^2 = e.a^2 - e.b^2)  -- Relationship between a, b, and c
    : standard_equation e = λ x y ↦ x^2 / 25 + y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2010_201008


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2010_201015

def complex_number : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem complex_number_in_second_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2010_201015


namespace NUMINAMATH_CALUDE_no_integer_n_makes_complex_fifth_power_real_l2010_201007

theorem no_integer_n_makes_complex_fifth_power_real : 
  ¬∃ (n : ℤ), (Complex.I : ℂ).im * ((n + 2 * Complex.I)^5).im = 0 := by sorry

end NUMINAMATH_CALUDE_no_integer_n_makes_complex_fifth_power_real_l2010_201007


namespace NUMINAMATH_CALUDE_group_meal_cost_l2010_201041

/-- The cost of a meal for a group at a restaurant -/
def mealCost (totalPeople : ℕ) (kids : ℕ) (adultMealPrice : ℕ) : ℕ :=
  (totalPeople - kids) * adultMealPrice

/-- Theorem: The meal cost for a group of 13 people with 9 kids is $28 -/
theorem group_meal_cost : mealCost 13 9 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_l2010_201041


namespace NUMINAMATH_CALUDE_ratio_problem_l2010_201036

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 2 / 7)
  (h3 : c / d = 4) :
  d / a = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2010_201036


namespace NUMINAMATH_CALUDE_right_triangle_geometric_sequence_sine_l2010_201016

theorem right_triangle_geometric_sequence_sine (a b c : Real) :
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- The sides form a geometric sequence
  (b / a = c / b ∨ a / b = b / c) →
  -- The sine of the smallest angle
  min (a / c) (b / c) = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_sequence_sine_l2010_201016


namespace NUMINAMATH_CALUDE_square_side_length_l2010_201046

theorem square_side_length (x : ℝ) (h : x > 0) : 4 * x = 2 * (x ^ 2) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2010_201046


namespace NUMINAMATH_CALUDE_point_difference_l2010_201020

/-- Represents a basketball player with their score and penalties. -/
structure Player where
  score : ℕ
  penalties : List ℕ

/-- Calculates the final score of a player after subtracting penalties. -/
def finalScore (p : Player) : ℤ :=
  p.score - p.penalties.sum

/-- Represents a basketball team with a list of players. -/
structure Team where
  players : List Player

/-- Calculates the total score of a team. -/
def teamScore (t : Team) : ℤ :=
  t.players.map finalScore |>.sum

/-- The given data for Team A. -/
def teamA : Team := {
  players := [
    { score := 12, penalties := [2] },
    { score := 18, penalties := [2, 2, 2] },
    { score := 5,  penalties := [] },
    { score := 7,  penalties := [3, 3] },
    { score := 6,  penalties := [1] }
  ]
}

/-- The given data for Team B. -/
def teamB : Team := {
  players := [
    { score := 10, penalties := [1, 1] },
    { score := 9,  penalties := [2] },
    { score := 12, penalties := [] },
    { score := 8,  penalties := [1, 1, 1] },
    { score := 5,  penalties := [3] },
    { score := 4,  penalties := [] }
  ]
}

/-- The main theorem stating the point difference between Team B and Team A. -/
theorem point_difference : teamScore teamB - teamScore teamA = 5 := by
  sorry


end NUMINAMATH_CALUDE_point_difference_l2010_201020


namespace NUMINAMATH_CALUDE_sophie_bought_five_cupcakes_l2010_201072

/-- The number of cupcakes Sophie bought -/
def num_cupcakes : ℕ := sorry

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of doughnuts Sophie bought -/
def num_doughnuts : ℕ := 6

/-- The price of each doughnut in dollars -/
def doughnut_price : ℚ := 1

/-- The number of apple pie slices Sophie bought -/
def num_pie_slices : ℕ := 4

/-- The price of each apple pie slice in dollars -/
def pie_slice_price : ℚ := 2

/-- The number of cookies Sophie bought -/
def num_cookies : ℕ := 15

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 3/5

/-- The total amount Sophie spent in dollars -/
def total_spent : ℚ := 33

/-- Theorem stating that Sophie bought 5 cupcakes -/
theorem sophie_bought_five_cupcakes :
  num_cupcakes = 5 ∧
  (num_cupcakes : ℚ) * cupcake_price +
  (num_doughnuts : ℚ) * doughnut_price +
  (num_pie_slices : ℚ) * pie_slice_price +
  (num_cookies : ℚ) * cookie_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sophie_bought_five_cupcakes_l2010_201072


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l2010_201089

-- Define the triangles and their side lengths
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

def triangle_PQR : Triangle :=
  { side1 := 10,
    side2 := 12,
    side3 := 0 }  -- We don't know the length of PR

def triangle_STU : Triangle :=
  { side1 := 5,
    side2 := 0,  -- We need to prove this is 6
    side3 := 0 }  -- We need to prove this is 6

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.side1 = k * t1.side1 ∧
    t2.side2 = k * t1.side2 ∧
    t2.side3 = k * t1.side3

-- Define the theorem
theorem triangle_similarity_theorem :
  similar triangle_PQR triangle_STU →
  triangle_STU.side2 = 6 ∧
  triangle_STU.side3 = 6 ∧
  triangle_STU.side1 + triangle_STU.side2 + triangle_STU.side3 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l2010_201089


namespace NUMINAMATH_CALUDE_prob_all_painted_10_beads_l2010_201086

/-- A circular necklace with beads -/
structure Necklace :=
  (num_beads : ℕ)

/-- The number of beads selected for painting -/
def num_selected : ℕ := 5

/-- Function to calculate the probability of all beads being painted -/
noncomputable def prob_all_painted (n : Necklace) : ℚ :=
  sorry

/-- Theorem stating the probability of all beads being painted for a 10-bead necklace -/
theorem prob_all_painted_10_beads :
  prob_all_painted { num_beads := 10 } = 17 / 42 :=
sorry

end NUMINAMATH_CALUDE_prob_all_painted_10_beads_l2010_201086


namespace NUMINAMATH_CALUDE_u_n_satisfies_property_u_n_is_smallest_u_n_equals_2n_minus_1_l2010_201034

/-- Given a positive integer n, u_n is the smallest positive integer such that
    for every positive integer d, the number of numbers divisible by d
    in any u_n consecutive positive odd numbers is no less than
    the number of numbers divisible by d in the set of odd numbers 1, 3, 5, ..., 2n-1 -/
def u_n (n : ℕ+) : ℕ :=
  2 * n.val - 1

/-- For any positive integer n, u_n satisfies the required property -/
theorem u_n_satisfies_property (n : ℕ+) :
  ∀ (d : ℕ+) (a : ℕ),
    (∀ k : Fin (2 * n.val - 1), ∃ m : ℕ, 2 * (a + k.val) - 1 = d * (2 * m + 1)) →
    (∃ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * (2 * m + 1)) :=
  sorry

/-- u_n is the smallest positive integer satisfying the required property -/
theorem u_n_is_smallest (n : ℕ+) :
  ∀ m : ℕ+, m.val < u_n n →
    ∃ (d : ℕ+) (a : ℕ),
      (∀ k : Fin m, ∃ l : ℕ, 2 * (a + k.val) - 1 = d * (2 * l + 1)) ∧
      ¬(∃ k : Fin n, ∃ l : ℕ, 2 * k.val + 1 = d * (2 * l + 1)) :=
  sorry

/-- The main theorem stating that u_n is equal to 2n - 1 -/
theorem u_n_equals_2n_minus_1 (n : ℕ+) :
  u_n n = 2 * n.val - 1 :=
  sorry

end NUMINAMATH_CALUDE_u_n_satisfies_property_u_n_is_smallest_u_n_equals_2n_minus_1_l2010_201034


namespace NUMINAMATH_CALUDE_bmw_cars_sold_l2010_201049

/-- The total number of cars sold -/
def total_cars : ℕ := 250

/-- The percentage of Audi cars sold -/
def audi_percent : ℚ := 10 / 100

/-- The percentage of Toyota cars sold -/
def toyota_percent : ℚ := 20 / 100

/-- The percentage of Acura cars sold -/
def acura_percent : ℚ := 15 / 100

/-- The percentage of Ford cars sold -/
def ford_percent : ℚ := 25 / 100

/-- The percentage of BMW cars sold -/
def bmw_percent : ℚ := 1 - (audi_percent + toyota_percent + acura_percent + ford_percent)

theorem bmw_cars_sold : 
  ⌊(bmw_percent * total_cars : ℚ)⌋ = 75 := by sorry

end NUMINAMATH_CALUDE_bmw_cars_sold_l2010_201049


namespace NUMINAMATH_CALUDE_work_completion_l2010_201005

theorem work_completion (days1 days2 men2 : ℕ) 
  (h1 : days1 = 80)
  (h2 : days2 = 56)
  (h3 : men2 = 20)
  (h4 : ∀ m d, m * d = men2 * days2) : 
  ∃ men1 : ℕ, men1 = 14 ∧ men1 * days1 = men2 * days2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l2010_201005


namespace NUMINAMATH_CALUDE_truck_initial_momentum_l2010_201028

/-- Initial momentum of a truck -/
theorem truck_initial_momentum
  (v : ℝ) -- Initial velocity
  (F : ℝ) -- Constant force applied to stop the truck
  (x : ℝ) -- Distance traveled before stopping
  (t : ℝ) -- Time taken to stop
  (h1 : v > 0) -- Assumption: initial velocity is positive
  (h2 : F > 0) -- Assumption: force is positive
  (h3 : x > 0) -- Assumption: distance is positive
  (h4 : t > 0) -- Assumption: time is positive
  (h5 : x = (v * t) / 2) -- Relation between distance, velocity, and time
  (h6 : F * t = v) -- Relation between force, time, and velocity change
  : ∃ (m : ℝ), m * v = (2 * F * x) / v :=
sorry

end NUMINAMATH_CALUDE_truck_initial_momentum_l2010_201028


namespace NUMINAMATH_CALUDE_xy_power_2018_l2010_201059

theorem xy_power_2018 (x y : ℝ) (h : |x - 1/2| + (y + 2)^2 = 0) : (x*y)^2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_power_2018_l2010_201059


namespace NUMINAMATH_CALUDE_car_speed_problem_l2010_201003

/-- Proves that given two cars P and R traveling 900 miles, where car P takes 2 hours less
    time than car R and has an average speed 10 miles per hour greater than car R,
    the average speed of car R is 62.25 miles per hour. -/
theorem car_speed_problem (speed_r : ℝ) : 
  (900 / speed_r - 2 = 900 / (speed_r + 10)) → speed_r = 62.25 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2010_201003


namespace NUMINAMATH_CALUDE_lowest_discount_l2010_201090

theorem lowest_discount (cost_price marked_price : ℝ) (min_profit_margin : ℝ) : 
  cost_price = 100 → 
  marked_price = 150 → 
  min_profit_margin = 0.05 → 
  ∃ (discount : ℝ), 
    discount = 0.7 ∧ 
    marked_price * discount = cost_price * (1 + min_profit_margin) ∧
    ∀ (d : ℝ), d > discount → marked_price * d > cost_price * (1 + min_profit_margin) :=
by sorry


end NUMINAMATH_CALUDE_lowest_discount_l2010_201090


namespace NUMINAMATH_CALUDE_expression_evaluation_l2010_201058

theorem expression_evaluation (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hsum : x + 1/y ≠ 0) : 
  (x^2 + 1/y^2) / (x + 1/y) = x - 1/y := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2010_201058


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l2010_201025

theorem greatest_integer_radius (r : ℕ) : (r : ℝ) ^ 2 * Real.pi < 90 * Real.pi → r ≤ 9 :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l2010_201025


namespace NUMINAMATH_CALUDE_valid_coloring_exists_l2010_201073

/-- A type representing a coloring of regions in a plane --/
def Coloring (n : ℕ) := Fin n → Bool

/-- A predicate that checks if a coloring is valid for n lines --/
def IsValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → c i ≠ c j

/-- The main theorem stating that a valid coloring exists for any number of lines --/
theorem valid_coloring_exists (n : ℕ) (h : n > 0) : 
  ∃ (c : Coloring n), IsValidColoring n c := by
  sorry

#check valid_coloring_exists

end NUMINAMATH_CALUDE_valid_coloring_exists_l2010_201073


namespace NUMINAMATH_CALUDE_rhombus_perimeter_given_side_l2010_201077

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- The perimeter of a rhombus is four times its side length -/
def perimeter (r : Rhombus) : ℝ := 4 * r.side_length

theorem rhombus_perimeter_given_side (r : Rhombus) (h : r.side_length = 2) : perimeter r = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_given_side_l2010_201077


namespace NUMINAMATH_CALUDE_problem_solution_l2010_201066

theorem problem_solution (x y : ℝ) : 
  x > 0 → x = 3 → x + y = 60 * (1 / x) → y = 17 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2010_201066


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l2010_201085

/-- The minimum distance from a point on the ellipse x = 4cos(θ), y = 3sin(θ) to the line x - y - 6 = 0 is √2/2 -/
theorem min_distance_ellipse_to_line :
  let ellipse := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = 4 * Real.cos θ ∧ y = 3 * Real.sin θ}
  let line := {(x, y) : ℝ × ℝ | x - y - 6 = 0}
  ∀ p ∈ ellipse, (
    let dist := fun q : ℝ × ℝ => |q.1 - q.2 - 6| / Real.sqrt 2
    ∃ q ∈ line, dist q = Real.sqrt 2 / 2 ∧ ∀ r ∈ line, dist p ≥ Real.sqrt 2 / 2
  ) := by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l2010_201085


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l2010_201092

/-- Given n ≥ 2 distinct integers, the polynomial f(x) = (x - a₁)(x - a₂) ... (x - aₙ) - 1 is irreducible over the integers. -/
theorem polynomial_irreducibility (n : ℕ) (a : Fin n → ℤ) (h1 : n ≥ 2) (h2 : Function.Injective a) :
  Irreducible (((Polynomial.X : Polynomial ℤ) - (Finset.univ.prod (fun i => Polynomial.X - Polynomial.C (a i)))) - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l2010_201092


namespace NUMINAMATH_CALUDE_perfect_square_values_l2010_201068

theorem perfect_square_values (a n : ℕ) : 
  (a ^ 2 + a + 1589 = n ^ 2) ↔ (a = 1588 ∨ a = 28 ∨ a = 316 ∨ a = 43) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_values_l2010_201068


namespace NUMINAMATH_CALUDE_triangle_area_is_integer_l2010_201040

-- Define a point in the plane
structure Point where
  x : Int
  y : Int

-- Define a function to check if a number is odd
def isOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Define a triangle with three points
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Define the area of a triangle
def triangleArea (t : Triangle) : Rat :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  Rat.ofInt (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

-- Theorem statement
theorem triangle_area_is_integer (t : Triangle) :
  t.p1 = Point.mk 1 1 →
  (isOdd t.p2.x ∧ isOdd t.p2.y) →
  (isOdd t.p3.x ∧ isOdd t.p3.y) →
  t.p2 ≠ t.p3 →
  ∃ n : Int, triangleArea t = n := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_integer_l2010_201040


namespace NUMINAMATH_CALUDE_cos_negative_750_degrees_l2010_201097

theorem cos_negative_750_degrees : Real.cos ((-750 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_750_degrees_l2010_201097


namespace NUMINAMATH_CALUDE_initial_chairs_per_row_l2010_201030

theorem initial_chairs_per_row (rows : ℕ) (extra_chairs : ℕ) (total_chairs : ℕ) :
  rows = 7 →
  extra_chairs = 11 →
  total_chairs = 95 →
  ∃ (chairs_per_row : ℕ), chairs_per_row * rows + extra_chairs = total_chairs ∧ chairs_per_row = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_chairs_per_row_l2010_201030


namespace NUMINAMATH_CALUDE_equation_solution_l2010_201054

theorem equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^(x - 2) = 9^(x + 2) ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2010_201054


namespace NUMINAMATH_CALUDE_stating_meal_distribution_count_l2010_201061

/-- Represents the number of people having dinner -/
def n : ℕ := 12

/-- Represents the number of meal types -/
def meal_types : ℕ := 4

/-- Represents the number of people who ordered each meal type -/
def people_per_meal : ℕ := 3

/-- Represents the number of people who should receive their ordered meal type -/
def correct_meals : ℕ := 2

/-- 
Theorem stating that the number of ways to distribute meals 
such that exactly two people receive their ordered meal type is 88047666
-/
theorem meal_distribution_count : 
  (Nat.choose n correct_meals) * (Nat.factorial (n - correct_meals)) = 88047666 := by
  sorry

end NUMINAMATH_CALUDE_stating_meal_distribution_count_l2010_201061


namespace NUMINAMATH_CALUDE_orthocenter_property_l2010_201029

/-- An acute-angled triangle with its orthocenter properties -/
structure AcuteTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Altitude lengths
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  -- Distances from vertices to orthocenter
  d_a : ℝ
  d_b : ℝ
  d_c : ℝ
  -- Conditions
  acute : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angles : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- The orthocenter property for acute-angled triangles -/
theorem orthocenter_property (t : AcuteTriangle) :
  t.m_a * t.d_a + t.m_b * t.d_b + t.m_c * t.d_c = (t.a^2 + t.b^2 + t.c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_property_l2010_201029


namespace NUMINAMATH_CALUDE_students_left_in_classroom_l2010_201014

theorem students_left_in_classroom 
  (total_students : ℕ) 
  (painting_fraction : ℚ) 
  (playing_fraction : ℚ) 
  (h1 : total_students = 50) 
  (h2 : painting_fraction = 3/5) 
  (h3 : playing_fraction = 1/5) : 
  total_students - (painting_fraction * total_students + playing_fraction * total_students) = 10 := by
sorry

end NUMINAMATH_CALUDE_students_left_in_classroom_l2010_201014


namespace NUMINAMATH_CALUDE_power_of_power_l2010_201093

theorem power_of_power (a : ℝ) : (a^2)^4 = a^8 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l2010_201093


namespace NUMINAMATH_CALUDE_magnitude_2a_minus_b_l2010_201053

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem magnitude_2a_minus_b : ‖(2 • a) - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_2a_minus_b_l2010_201053


namespace NUMINAMATH_CALUDE_max_speed_is_four_l2010_201069

/-- Represents the scenario of two pedestrians traveling between points A and B. -/
structure PedestrianScenario where
  route1_length : ℝ
  route2_length : ℝ
  first_section_length : ℝ
  time_difference : ℝ
  speed_difference : ℝ

/-- Calculates the maximum average speed of the first pedestrian on the second section. -/
def max_average_speed (scenario : PedestrianScenario) : ℝ :=
  4 -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the maximum average speed is 4 km/h given the scenario conditions. -/
theorem max_speed_is_four (scenario : PedestrianScenario) 
  (h1 : scenario.route1_length = 19)
  (h2 : scenario.route2_length = 12)
  (h3 : scenario.first_section_length = 11)
  (h4 : scenario.time_difference = 2)
  (h5 : scenario.speed_difference = 0.5) :
  max_average_speed scenario = 4 := by
  sorry

#check max_speed_is_four

end NUMINAMATH_CALUDE_max_speed_is_four_l2010_201069


namespace NUMINAMATH_CALUDE_min_n_for_S_n_gt_1020_l2010_201079

/-- The sum of the first n terms in the sequence -/
def S_n (n : ℕ) : ℤ := 2 * (2^n - 1) - n

/-- The proposition that 10 is the minimum value of n such that S_n > 1020 -/
theorem min_n_for_S_n_gt_1020 :
  (∀ k < 10, S_n k ≤ 1020) ∧ S_n 10 > 1020 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_S_n_gt_1020_l2010_201079


namespace NUMINAMATH_CALUDE_odd_function_property_l2010_201099

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_property :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x - 1) →    -- f(x) = x - 1 for x > 0
  (∀ x < 0, f x * f (-x) ≤ 0) -- f(x)f(-x) ≤ 0 for x < 0
:= by sorry

end NUMINAMATH_CALUDE_odd_function_property_l2010_201099


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l2010_201045

/-- The number of red roses at Greene Nursery -/
def red_roses : ℕ := 1491

/-- The number of yellow carnations at Greene Nursery -/
def yellow_carnations : ℕ := 3025

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 1768

/-- The total number of flowers at Greene Nursery -/
def total_flowers : ℕ := red_roses + yellow_carnations + white_roses

theorem greene_nursery_flower_count : total_flowers = 6284 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l2010_201045


namespace NUMINAMATH_CALUDE_product_equals_fraction_l2010_201024

/-- The repeating decimal 0.456̄ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal 0.456̄ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̄ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l2010_201024


namespace NUMINAMATH_CALUDE_sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero_l2010_201081

open Real

theorem sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero :
  ∀ x : ℝ, (sin x + cos x = 0) ↔ ((cos (2 * x)) / (sin x - cos x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero_l2010_201081


namespace NUMINAMATH_CALUDE_number_count_l2010_201026

theorem number_count (avg_all : ℝ) (avg_group1 : ℝ) (avg_group2 : ℝ) (avg_group3 : ℝ) 
  (h1 : avg_all = 2.80)
  (h2 : avg_group1 = 2.4)
  (h3 : avg_group2 = 2.3)
  (h4 : avg_group3 = 3.7) :
  ∃ (n : ℕ), n = 6 ∧ (2 * avg_group1 + 2 * avg_group2 + 2 * avg_group3) / n = avg_all := by
  sorry

end NUMINAMATH_CALUDE_number_count_l2010_201026


namespace NUMINAMATH_CALUDE_smallest_cube_multiple_l2010_201082

theorem smallest_cube_multiple : ∃ (x : ℕ), x > 0 ∧ (∃ (M : ℤ), 2520 * x = M^3) ∧ 
  (∀ (y : ℕ), y > 0 → (∃ (N : ℤ), 2520 * y = N^3) → x ≤ y) ∧ x = 3675 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_multiple_l2010_201082


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2010_201044

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 + 5*x + 8)*(x + 2) - (x + 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x + 2)*(x + 6) = 
  6*x^3 + 21*x^2 + 18*x := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2010_201044


namespace NUMINAMATH_CALUDE_cone_height_ratio_l2010_201056

/-- Represents a cone with height, slant height, and central angle of unfolded lateral surface -/
structure Cone where
  height : ℝ
  slant_height : ℝ
  central_angle : ℝ

/-- The theorem statement -/
theorem cone_height_ratio (A B : Cone) :
  A.slant_height = B.slant_height →
  A.central_angle + B.central_angle = 2 * Real.pi →
  A.central_angle * A.slant_height^2 / (B.central_angle * B.slant_height^2) = 2 →
  A.height / B.height = Real.sqrt 10 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l2010_201056


namespace NUMINAMATH_CALUDE_total_amount_to_pay_l2010_201050

def original_balance : ℝ := 150
def finance_charge_percentage : ℝ := 0.02

theorem total_amount_to_pay : 
  original_balance * (1 + finance_charge_percentage) = 153 := by sorry

end NUMINAMATH_CALUDE_total_amount_to_pay_l2010_201050


namespace NUMINAMATH_CALUDE_derek_water_addition_l2010_201083

/-- The amount of water Derek added to the bucket -/
def water_added (initial final : ℝ) : ℝ := final - initial

theorem derek_water_addition (initial final : ℝ) 
  (h1 : initial = 3)
  (h2 : final = 9.8) :
  water_added initial final = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_derek_water_addition_l2010_201083


namespace NUMINAMATH_CALUDE_min_days_person_A_l2010_201000

/-- Represents the number of days a person takes to complete the project alone -/
structure PersonSpeed where
  days : ℕ
  days_positive : days > 0

/-- Represents the work done by a person in a day -/
def work_rate (speed : PersonSpeed) : ℚ :=
  1 / speed.days

/-- The total project work is 1 -/
def total_work : ℚ := 1

/-- Theorem stating the minimum number of days person A must work -/
theorem min_days_person_A (
  speed_A speed_B speed_C : PersonSpeed)
  (h_A : speed_A.days = 24)
  (h_B : speed_B.days = 36)
  (h_C : speed_C.days = 60)
  (total_days : ℕ)
  (h_total_days : total_days ≤ 18)
  (h_integer_days : ∃ (days_A days_B days_C : ℕ),
    days_A + days_B + days_C = total_days ∧
    days_A * work_rate speed_A + days_B * work_rate speed_B + days_C * work_rate speed_C = total_work) :
  ∃ (min_days_A : ℕ), min_days_A = 6 ∧
    ∀ (days_A : ℕ), 
      (∃ (days_B days_C : ℕ),
        days_A + days_B + days_C = total_days ∧
        days_A * work_rate speed_A + days_B * work_rate speed_B + days_C * work_rate speed_C = total_work) →
      days_A ≥ min_days_A :=
by sorry

end NUMINAMATH_CALUDE_min_days_person_A_l2010_201000


namespace NUMINAMATH_CALUDE_equilateral_triangle_grid_polygon_area_l2010_201048

/-- Represents an equilateral triangular grid -/
structure EquilateralTriangularGrid where
  sideLength : ℕ
  totalPoints : ℕ

/-- Represents a polygon on the grid -/
structure Polygon (G : EquilateralTriangularGrid) where
  vertices : ℕ
  nonSelfIntersecting : Bool
  usesAllPoints : Bool

/-- The area of a polygon on an equilateral triangular grid -/
noncomputable def polygonArea (G : EquilateralTriangularGrid) (S : Polygon G) : ℝ :=
  sorry

theorem equilateral_triangle_grid_polygon_area 
  (G : EquilateralTriangularGrid) 
  (S : Polygon G) :
  G.sideLength = 20 ∧ 
  G.totalPoints = 210 ∧ 
  S.vertices = 210 ∧ 
  S.nonSelfIntersecting = true ∧ 
  S.usesAllPoints = true →
  polygonArea G S = 52 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_grid_polygon_area_l2010_201048


namespace NUMINAMATH_CALUDE_power_equation_solution_l2010_201009

theorem power_equation_solution (p : ℕ) : 64^5 = 8^p → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2010_201009


namespace NUMINAMATH_CALUDE_representatives_count_l2010_201002

/-- The number of ways to select representatives from boys and girls -/
def select_representatives (num_boys num_girls num_representatives : ℕ) : ℕ :=
  Nat.choose num_boys 2 * Nat.choose num_girls 1 +
  Nat.choose num_boys 1 * Nat.choose num_girls 2

/-- Theorem stating that selecting 3 representatives from 5 boys and 3 girls,
    with both genders represented, can be done in 45 ways -/
theorem representatives_count :
  select_representatives 5 3 3 = 45 := by
  sorry

#eval select_representatives 5 3 3

end NUMINAMATH_CALUDE_representatives_count_l2010_201002


namespace NUMINAMATH_CALUDE_strawberry_problem_l2010_201043

theorem strawberry_problem (initial : Float) (eaten : Float) (remaining : Float) :
  initial = 78.0 → eaten = 42.0 → remaining = initial - eaten → remaining = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_problem_l2010_201043


namespace NUMINAMATH_CALUDE_problem_solution_l2010_201060

/-- Represents a three-digit number in the form abc --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem problem_solution :
  ∀ (a b : Nat),
  let n1 := ThreeDigitNumber.mk 3 a 7 (by sorry)
  let n2 := ThreeDigitNumber.mk 6 b 1 (by sorry)
  (n1.toNat + 294 = n2.toNat) →
  (n2.toNat % 7 = 0) →
  a + b = 8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2010_201060


namespace NUMINAMATH_CALUDE_range_of_a_l2010_201039

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3/4| ≤ 1/4
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ 0 ≤ a ∧ a ≤ 1/2 ∧ (a ≠ 0 ∨ a ≠ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2010_201039


namespace NUMINAMATH_CALUDE_triangle_side_length_l2010_201051

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) - Real.cos (2 * x / 2)^2 + 1/2

theorem triangle_side_length 
  (A B C : ℝ) 
  (hA : 0 < A ∧ A < π) 
  (hB : 0 < B ∧ B < π) 
  (hC : 0 < C ∧ C < π) 
  (hABC : A + B + C = π) 
  (hf : f A = 1/2) 
  (ha : Real.sqrt 3 = (Real.sin B / Real.sin A)) 
  (hB : Real.sin B = 2 * Real.sin C) : 
  Real.sin C / Real.sin A = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2010_201051


namespace NUMINAMATH_CALUDE_rectangle_diagonal_length_l2010_201067

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The length of the diagonal of a rectangle -/
def diagonal_length (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_diagonal_length :
  ∀ r : Rectangle, r.area = 16 ∧ r.perimeter = 18 → diagonal_length r = 7 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_length_l2010_201067


namespace NUMINAMATH_CALUDE_doughnut_savings_l2010_201012

/-- The cost of one dozen doughnuts -/
def cost_one_dozen : ℕ := 8

/-- The cost of two dozens of doughnuts -/
def cost_two_dozens : ℕ := 14

/-- The number of sets when buying one dozen at a time -/
def sets_one_dozen : ℕ := 6

/-- The number of sets when buying two dozens at a time -/
def sets_two_dozens : ℕ := 3

/-- Theorem stating the savings when buying 3 sets of 2 dozens instead of 6 sets of 1 dozen -/
theorem doughnut_savings : 
  sets_one_dozen * cost_one_dozen - sets_two_dozens * cost_two_dozens = 6 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_savings_l2010_201012


namespace NUMINAMATH_CALUDE_percentage_men_science_majors_l2010_201094

/-- Represents the composition of a college class -/
structure ClassComposition where
  total : ℝ
  women : ℝ
  men : ℝ
  scienceMajors : ℝ
  womenScienceMajors : ℝ
  nonScienceMajors : ℝ

/-- Theorem stating the percentage of men who are science majors -/
theorem percentage_men_science_majors (c : ClassComposition) : 
  c.total > 0 ∧ 
  c.women = 0.6 * c.total ∧ 
  c.men = 0.4 * c.total ∧ 
  c.nonScienceMajors = 0.6 * c.total ∧
  c.womenScienceMajors = 0.2 * c.women →
  (c.scienceMajors - c.womenScienceMajors) / c.men = 0.7 := by
  sorry

#check percentage_men_science_majors

end NUMINAMATH_CALUDE_percentage_men_science_majors_l2010_201094


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2010_201006

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a + b + c = 18) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2010_201006


namespace NUMINAMATH_CALUDE_homomorphism_characterization_l2010_201010

theorem homomorphism_characterization (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y) →
  ∃ a : ℤ, ∀ x : ℤ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_homomorphism_characterization_l2010_201010


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2010_201084

theorem children_neither_happy_nor_sad 
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_happy_nor_sad_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 16)
  (h5 : girls = 44)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_happy_nor_sad_boys = 4)
  : total_children - happy_children - sad_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2010_201084


namespace NUMINAMATH_CALUDE_vector_parallel_sum_l2010_201096

/-- Given vectors a and b, if a is parallel to (a + b), then the second component of b is 3. -/
theorem vector_parallel_sum (a b : ℝ × ℝ) (h : ∃ (k : ℝ), a = k • (a + b)) :
  a.1 = 1 ∧ a.2 = 1 ∧ b.1 = 3 → b.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_sum_l2010_201096


namespace NUMINAMATH_CALUDE_height_prediction_approximate_l2010_201075

/-- Regression model for height prediction -/
def height_model (x : ℝ) : ℝ := 7.19 * x + 73.93

/-- The age at which we want to predict the height -/
def prediction_age : ℝ := 10

/-- The predicted height at the given age -/
def predicted_height : ℝ := height_model prediction_age

theorem height_prediction_approximate :
  ∃ ε > 0, abs (predicted_height - 145.83) < ε :=
sorry

end NUMINAMATH_CALUDE_height_prediction_approximate_l2010_201075


namespace NUMINAMATH_CALUDE_quadratic_intercept_l2010_201022

/-- A quadratic function with vertex (4, 9) and x-intercept (0, 0) has its other x-intercept at x = 8 -/
theorem quadratic_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9) →  -- vertex form
  (a * 0^2 + b * 0 + c = 0) →                       -- (0, 0) is an x-intercept
  (∃ x ≠ 0, a * x^2 + b * x + c = 0 ∧ x = 8) :=     -- other x-intercept is at x = 8
by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_l2010_201022


namespace NUMINAMATH_CALUDE_sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two_l2010_201074

theorem sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two : 
  Real.sqrt 18 - 3 * Real.sqrt (1/2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_minus_three_sqrt_half_plus_sqrt_two_l2010_201074


namespace NUMINAMATH_CALUDE_power_of_seven_l2010_201070

theorem power_of_seven (k : ℕ) (h : 7^k = 2) : 7^(2*k + 2) = 784 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_l2010_201070


namespace NUMINAMATH_CALUDE_smallest_positive_root_floor_l2010_201087

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

theorem smallest_positive_root_floor :
  ∃ s : ℝ, s > 0 ∧ g s = 0 ∧ (∀ t, t > 0 ∧ g t = 0 → s ≤ t) ∧ 4 ≤ s ∧ s < 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_root_floor_l2010_201087


namespace NUMINAMATH_CALUDE_chris_previous_savings_l2010_201095

/-- Represents the amount of money Chris received as birthday gifts in different currencies -/
structure BirthdayGifts where
  usd : ℝ
  eur : ℝ
  cad : ℝ
  gbp : ℝ

/-- Represents the conversion rates from different currencies to USD -/
structure ConversionRates where
  eur_to_usd : ℝ
  cad_to_usd : ℝ
  gbp_to_usd : ℝ

/-- Calculates Chris's savings before his birthday -/
def calculate_previous_savings (gifts : BirthdayGifts) (rates : ConversionRates) (total_after : ℝ) : ℝ :=
  total_after - (gifts.usd + 
                 gifts.eur * rates.eur_to_usd + 
                 gifts.cad * rates.cad_to_usd + 
                 gifts.gbp * rates.gbp_to_usd)

/-- Theorem stating that Chris's savings before his birthday were 128.80 USD -/
theorem chris_previous_savings 
  (gifts : BirthdayGifts) 
  (rates : ConversionRates) 
  (total_after : ℝ) : 
  gifts.usd = 25 ∧ 
  gifts.eur = 20 ∧ 
  gifts.cad = 75 ∧ 
  gifts.gbp = 30 ∧
  rates.eur_to_usd = 1 / 0.85 ∧
  rates.cad_to_usd = 1 / 1.25 ∧
  rates.gbp_to_usd = 1 / 0.72 ∧
  total_after = 279 →
  calculate_previous_savings gifts rates total_after = 128.80 := by
    sorry

end NUMINAMATH_CALUDE_chris_previous_savings_l2010_201095


namespace NUMINAMATH_CALUDE_debby_flour_problem_l2010_201064

theorem debby_flour_problem (x : ℝ) : x + 4 = 16 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_flour_problem_l2010_201064


namespace NUMINAMATH_CALUDE_pencil_rows_l2010_201052

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) :
  total_pencils / pencils_per_row = 7 :=
by sorry

end NUMINAMATH_CALUDE_pencil_rows_l2010_201052


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l2010_201062

/-- Given a quadratic function f(x) = ax^2 - c, 
    if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20 -/
theorem quadratic_function_bounds (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - c
  (-4 : ℝ) ≤ f 1 ∧ f 1 ≤ -1 ∧ -1 ≤ f 2 ∧ f 2 ≤ 5 → 
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l2010_201062


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_five_l2010_201013

theorem fifteenth_odd_multiple_of_five : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∀ k < n, ∃ m : ℕ, k = 2 * m + 1 ∧ ∃ l : ℕ, k = 5 * l) →
  (∃ m : ℕ, n = 2 * m + 1) ∧ 
  (∃ l : ℕ, n = 5 * l) ∧ 
  n = 145 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_five_l2010_201013
