import Mathlib

namespace dog_grouping_theorem_l1490_149067

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of dogs -/
def total_dogs : ℕ := 12

/-- The number of dogs in Fluffy's group -/
def fluffy_group_size : ℕ := 3

/-- The number of dogs in Nipper's group -/
def nipper_group_size : ℕ := 5

/-- The number of dogs in the third group -/
def third_group_size : ℕ := 4

theorem dog_grouping_theorem :
  choose (total_dogs - 2) (fluffy_group_size - 1) *
  choose (total_dogs - fluffy_group_size - 1) (nipper_group_size - 1) = 3150 := by
  sorry

end dog_grouping_theorem_l1490_149067


namespace original_average_l1490_149092

theorem original_average (n : ℕ) (a : ℝ) (b : ℝ) (c : ℝ) :
  n > 0 →
  n = 15 →
  b = 13 →
  c = 53 →
  (a + b = c) →
  a = 40 := by
sorry

end original_average_l1490_149092


namespace miles_on_wednesday_l1490_149012

/-- Represents the miles run by Mrs. Hilt on different days of the week -/
structure RunningWeek where
  monday : ℕ
  wednesday : ℕ
  friday : ℕ
  total : ℕ

/-- Theorem stating that Mrs. Hilt ran 2 miles on Wednesday -/
theorem miles_on_wednesday (week : RunningWeek) 
  (h1 : week.monday = 3)
  (h2 : week.friday = 7)
  (h3 : week.total = 12)
  : week.wednesday = 2 := by
  sorry

end miles_on_wednesday_l1490_149012


namespace incorrect_calculation_l1490_149066

theorem incorrect_calculation (a : ℝ) : a^3 + a^3 ≠ 2*a^6 := by
  sorry

end incorrect_calculation_l1490_149066


namespace coefficient_x3y0_l1490_149002

/-- The coefficient of x^m * y^n in the expansion of (1+x)^6 * (1+y)^4 -/
def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem coefficient_x3y0 : f 3 0 = 20 := by
  sorry

end coefficient_x3y0_l1490_149002


namespace circle_C_properties_l1490_149083

-- Define the circle C
def circle_C (x y k : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - k = 0

-- Define the center of the circle
def center_of_circle (h k : ℝ) : Prop := 
  ∀ x y k, circle_C x y k ↔ (x - h)^2 + (y - k)^2 = k + 5

-- Define the radius of the circle
def radius_of_circle (r k : ℝ) : Prop := 
  ∀ x y, circle_C x y k ↔ (x - 1)^2 + (y + 2)^2 = r^2

-- Theorem statements
theorem circle_C_properties :
  (∀ k, (∃ x y, circle_C x y k) → k > -5) ∧
  center_of_circle 1 (-2) ∧
  radius_of_circle 3 4 ∧
  (∀ k, (∃ x, circle_C x 0 k) ∧ (∀ y, y ≠ 0 → ¬circle_C x y k) → k = -1) :=
sorry

end circle_C_properties_l1490_149083


namespace shoe_pairs_in_box_l1490_149081

theorem shoe_pairs_in_box (total_shoes : ℕ) (prob_matching : ℚ) : 
  total_shoes = 200 →
  prob_matching = 1 / 199 →
  (total_shoes / 2 : ℕ) = 100 :=
by sorry

end shoe_pairs_in_box_l1490_149081


namespace sum_abc_values_l1490_149047

theorem sum_abc_values (a b c : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1)
  (hab : b = a^2 / (2 - a^2))
  (hbc : c = b^2 / (2 - b^2))
  (hca : a = c^2 / (2 - c^2)) :
  (a + b + c = 6) ∨ (a + b + c = -4) ∨ (a + b + c = -6) := by
  sorry

end sum_abc_values_l1490_149047


namespace binomial_expansion_equality_l1490_149024

theorem binomial_expansion_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  (45 : ℝ) * p^8 * q^2 = (120 : ℝ) * p^7 * q^3 → 
  p = 8/11 := by
sorry

end binomial_expansion_equality_l1490_149024


namespace derivative_f_at_pi_l1490_149094

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_f_at_pi : 
  deriv f π = -1 / (π^2) := by sorry

end derivative_f_at_pi_l1490_149094


namespace power_function_m_squared_minus_three_l1490_149033

/-- A function f(x) is a power function if it can be written as f(x) = ax^n, where a and n are constants and n ≠ 0. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), n ≠ 0 ∧ ∀ x, f x = a * x^n

/-- Given that y = (m^2 - 3)x^(2m) is a power function with respect to x, prove that m = ±2. -/
theorem power_function_m_squared_minus_three (m : ℝ) :
  is_power_function (λ x => (m^2 - 3) * x^(2*m)) → m = 2 ∨ m = -2 := by
  sorry

end power_function_m_squared_minus_three_l1490_149033


namespace parabola_and_line_theorem_l1490_149010

/-- A parabola with focus F and point A on it -/
structure Parabola where
  p : ℝ
  m : ℝ
  h : p > 0

/-- The distance from point A to the focus F is 5 -/
def distance_condition (par : Parabola) : Prop :=
  4 + par.p / 2 = 5

/-- Point A lies on the parabola -/
def point_on_parabola (par : Parabola) : Prop :=
  par.m^2 = 2 * par.p * 4

/-- m is positive -/
def m_positive (par : Parabola) : Prop :=
  par.m > 0

/-- A line that passes through point A -/
structure Line where
  k : ℝ
  b : ℝ

/-- The line intersects the parabola at exactly one point -/
def line_intersects_once (par : Parabola) (l : Line) : Prop :=
  (∀ x y, y = l.k * x + l.b → y^2 = 4 * x) →
  (∃! x, (l.k * x + l.b)^2 = 4 * x)

theorem parabola_and_line_theorem (par : Parabola) 
  (h1 : distance_condition par)
  (h2 : point_on_parabola par)
  (h3 : m_positive par) :
  (par.p = 2 ∧ par.m = 4) ∧
  (∃ l1 l2 : Line, 
    (l1.k = -2 ∧ l1.b = 4 ∧ line_intersects_once par l1) ∧
    (l2.k = 0 ∧ l2.b = 4 ∧ line_intersects_once par l2)) := by
  sorry

end parabola_and_line_theorem_l1490_149010


namespace distribute_four_items_three_bags_l1490_149022

/-- The number of ways to distribute n distinct items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 14 ways to distribute 4 distinct items into 3 identical bags, allowing empty bags. -/
theorem distribute_four_items_three_bags : distribute 4 3 = 14 := by sorry

end distribute_four_items_three_bags_l1490_149022


namespace die_throw_outcomes_l1490_149029

/-- Represents the number of sides on a fair cubic die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 4

/-- Represents the number of different outcomes required to stop -/
def differentOutcomes : ℕ := 3

/-- Calculates the total number of different outcomes for the die throws -/
def totalOutcomes : ℕ := numSides * (numSides - 1) * (numSides - 2) * differentOutcomes

theorem die_throw_outcomes :
  totalOutcomes = 270 :=
sorry

end die_throw_outcomes_l1490_149029


namespace line_circle_intersection_range_l1490_149099

theorem line_circle_intersection_range (m : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m ∧ 
    x₁^2 + y₁^2 + 4*x₁ + 2 = 0 ∧ 
    x₂^2 + y₂^2 + 4*x₂ + 2 = 0) → 
  0 < m ∧ m < 4 :=
by sorry

end line_circle_intersection_range_l1490_149099


namespace radical_simplification_l1490_149018

-- Define the statement
theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (11 * q) * Real.sqrt (8 * q^3) * Real.sqrt (9 * q^5) = 28 * q^4 * Real.sqrt q :=
by sorry

end radical_simplification_l1490_149018


namespace clock_equivalent_hours_l1490_149034

theorem clock_equivalent_hours : ∃ (n : ℕ), n > 6 ∧ n ≡ n^2 [ZMOD 24] ∧
  ∀ (m : ℕ), m > 6 ∧ m < n → ¬(m ≡ m^2 [ZMOD 24]) :=
by sorry

end clock_equivalent_hours_l1490_149034


namespace escalator_solution_l1490_149093

/-- Represents the escalator problem with given conditions -/
structure EscalatorProblem where
  total_steps : ℕ
  escalator_speed : ℚ
  walking_speed : ℚ
  first_condition : 26 + 30 * escalator_speed = total_steps
  second_condition : 34 + 18 * escalator_speed = total_steps

/-- The solution to the escalator problem -/
theorem escalator_solution (problem : EscalatorProblem) : problem.total_steps = 46 := by
  sorry

#check escalator_solution

end escalator_solution_l1490_149093


namespace min_value_reciprocal_sum_l1490_149017

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^((a + b)/2) = Real.sqrt 3 → 
  (1/a + 1/b) ≥ 4 :=
by sorry

end min_value_reciprocal_sum_l1490_149017


namespace quadratic_root_transformation_l1490_149013

theorem quadratic_root_transformation (p q r u v : ℝ) : 
  (p * u^2 + q * u + r = 0) → 
  (p * v^2 + q * v + r = 0) → 
  ((2*p*u + q)^2 - q^2 + 4*p*r = 0) ∧ 
  ((2*p*v + q)^2 - q^2 + 4*p*r = 0) :=
by sorry

end quadratic_root_transformation_l1490_149013


namespace min_value_trig_fraction_min_value_is_one_l1490_149004

theorem min_value_trig_fraction (x : ℝ) :
  (Real.sin x)^5 + (Real.cos x)^5 + 1 ≥ (Real.sin x)^3 + (Real.cos x)^3 + 1 := by
  sorry

theorem min_value_is_one :
  ∀ x : ℝ, ((Real.sin x)^5 + (Real.cos x)^5 + 1) / ((Real.sin x)^3 + (Real.cos x)^3 + 1) ≥ 1 := by
  sorry

end min_value_trig_fraction_min_value_is_one_l1490_149004


namespace smallest_range_of_four_integers_with_mean_2017_l1490_149089

/-- Given four different positive integers with a mean of 2017, 
    the smallest possible range between the largest and smallest of these integers is 4. -/
theorem smallest_range_of_four_integers_with_mean_2017 :
  ∀ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (a + b + c + d) / 4 = 2017 →
  (∀ (w x y z : ℕ), 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (w + x + y + z) / 4 = 2017 →
    max w (max x (max y z)) - min w (min x (min y z)) ≥ 4) ∧
  (∃ (p q r s : ℕ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 →
    (p + q + r + s) / 4 = 2017 →
    max p (max q (max r s)) - min p (min q (min r s)) = 4) :=
by sorry

end smallest_range_of_four_integers_with_mean_2017_l1490_149089


namespace biography_increase_l1490_149043

theorem biography_increase (B : ℝ) (N : ℝ) (h1 : N > 0) (h2 : B > 0)
  (h3 : 0.20 * B + N = 0.30 * (B + N)) :
  (N / (0.20 * B)) * 100 = 100 / 1.4 := by
  sorry

end biography_increase_l1490_149043


namespace white_pawn_on_white_square_l1490_149057

/-- Represents a chessboard with white and black pawns. -/
structure Chessboard where
  white_pawns : ℕ
  black_pawns : ℕ
  pawns_on_white_squares : ℕ
  pawns_on_black_squares : ℕ

/-- Theorem: Given a chessboard with more white pawns than black pawns,
    and more pawns on white squares than on black squares,
    there exists at least one white pawn on a white square. -/
theorem white_pawn_on_white_square (board : Chessboard)
  (h1 : board.white_pawns > board.black_pawns)
  (h2 : board.pawns_on_white_squares > board.pawns_on_black_squares) :
  ∃ (white_pawns_on_white_squares : ℕ), white_pawns_on_white_squares > 0 := by
  sorry

end white_pawn_on_white_square_l1490_149057


namespace f_max_min_on_interval_l1490_149044

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = min) ∧
    max = 18 ∧ min = -2 :=
sorry

end f_max_min_on_interval_l1490_149044


namespace inequality_implies_b_minus_a_equals_two_l1490_149038

theorem inequality_implies_b_minus_a_equals_two (a b : ℝ) :
  (∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) →
  b - a = 2 := by
sorry

end inequality_implies_b_minus_a_equals_two_l1490_149038


namespace P_in_fourth_quadrant_l1490_149086

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : Point :=
  { x := 2023, y := -2024 }

/-- Theorem stating that P is in the fourth quadrant -/
theorem P_in_fourth_quadrant : fourth_quadrant P := by
  sorry

end P_in_fourth_quadrant_l1490_149086


namespace power_of_sum_equals_225_l1490_149095

theorem power_of_sum_equals_225 : (3^2 + 6)^(4/2) = 225 := by sorry

end power_of_sum_equals_225_l1490_149095


namespace apples_per_pie_is_seven_l1490_149052

/-- Calculates the number of apples used per pie given the initial conditions -/
def apples_per_pie (
  total_apples : ℕ)
  (num_children : ℕ)
  (apples_per_child : ℕ)
  (num_pies : ℕ)
  (remaining_apples : ℕ) : ℕ :=
  let apples_for_teachers := num_children * apples_per_child
  let apples_for_pies := total_apples - apples_for_teachers - remaining_apples
  apples_for_pies / num_pies

/-- Proves that the number of apples used per pie is 7 under the given conditions -/
theorem apples_per_pie_is_seven :
  apples_per_pie 50 2 6 2 24 = 7 := by
  sorry

end apples_per_pie_is_seven_l1490_149052


namespace arithmetic_progression_cubes_l1490_149046

theorem arithmetic_progression_cubes (x y z : ℤ) : 
  x < y ∧ y < z ∧ y = (x + z) / 2 → ¬(y^3 = (x^3 + z^3) / 2) :=
by sorry

end arithmetic_progression_cubes_l1490_149046


namespace lateral_surface_area_of_given_pyramid_l1490_149045

/-- A right truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  height : ℝ
  volume : ℝ
  base_ratio : ℝ × ℝ

/-- The lateral surface area of a truncated pyramid -/
def lateral_surface_area (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The given truncated pyramid -/
def given_pyramid : TruncatedPyramid :=
  { height := 3,
    volume := 38,
    base_ratio := (4, 9) }

/-- Theorem: The lateral surface area of the given truncated pyramid is 10√19 -/
theorem lateral_surface_area_of_given_pyramid :
  lateral_surface_area given_pyramid = 10 * Real.sqrt 19 := by
  sorry

end lateral_surface_area_of_given_pyramid_l1490_149045


namespace adult_meal_cost_l1490_149014

/-- Calculates the cost of an adult meal given the total number of people,
    number of kids, and total cost for a group at a restaurant where kids eat free. -/
theorem adult_meal_cost (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) :
  total_people = 9 →
  num_kids = 2 →
  total_cost = 14 →
  (total_cost / (total_people - num_kids : ℚ)) = 2 := by
  sorry

end adult_meal_cost_l1490_149014


namespace movies_to_watch_l1490_149051

theorem movies_to_watch (total_movies : ℕ) (watched_movies : ℕ) 
  (h1 : total_movies = 35) (h2 : watched_movies = 18) :
  total_movies - watched_movies = 17 := by
  sorry

end movies_to_watch_l1490_149051


namespace students_with_both_calculation_l1490_149087

/-- The number of students who brought both apples and bananas -/
def students_with_both : ℕ := sorry

/-- The number of students who brought apples -/
def students_with_apples : ℕ := 12

/-- The number of students who brought bananas -/
def students_with_bananas : ℕ := 8

/-- The number of students who brought only one type of fruit -/
def students_with_one_fruit : ℕ := 10

theorem students_with_both_calculation : 
  students_with_both = students_with_apples + students_with_bananas - students_with_one_fruit :=
by sorry

end students_with_both_calculation_l1490_149087


namespace expression_equivalence_l1490_149007

theorem expression_equivalence (a : ℝ) : 
  (a^2 + a - 2) / (a^2 + 3*a + 2) * (5 * (a + 1)^2) = 5*a^2 - 5 :=
by sorry

end expression_equivalence_l1490_149007


namespace minimum_peanuts_l1490_149016

theorem minimum_peanuts (N A₁ A₂ A₃ A₄ A₅ : ℕ) : 
  N = 5 * A₁ + 1 ∧
  4 * A₁ = 5 * A₂ + 1 ∧
  4 * A₂ = 5 * A₃ + 1 ∧
  4 * A₃ = 5 * A₄ + 1 ∧
  4 * A₄ = 5 * A₅ + 1 →
  N ≥ 3121 ∧ (N = 3121 → 
    A₁ = 624 ∧ A₂ = 499 ∧ A₃ = 399 ∧ A₄ = 319 ∧ A₅ = 255) :=
by sorry

#check minimum_peanuts

end minimum_peanuts_l1490_149016


namespace sum_of_three_squares_divisibility_l1490_149020

theorem sum_of_three_squares_divisibility (N : ℕ) :
  (∃ a b c : ℤ, (N : ℤ) = a^2 + b^2 + c^2 ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ x y z : ℤ, (N : ℤ) = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end sum_of_three_squares_divisibility_l1490_149020


namespace green_bay_high_relay_race_length_l1490_149070

/-- The length of a relay race given the number of team members and distance per member -/
def relay_race_length (team_members : ℕ) (distance_per_member : ℕ) : ℕ :=
  team_members * distance_per_member

/-- Theorem: The relay race length for 5 team members running 30 meters each is 150 meters -/
theorem green_bay_high_relay_race_length :
  relay_race_length 5 30 = 150 := by
  sorry

end green_bay_high_relay_race_length_l1490_149070


namespace rhombus_perimeter_l1490_149000

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 := by
  sorry

#check rhombus_perimeter

end rhombus_perimeter_l1490_149000


namespace max_value_of_f_l1490_149048

-- Define the function
def f (x : ℝ) := abs (x^2 - 4) - 6*x

-- State the theorem
theorem max_value_of_f :
  ∃ (b : ℝ), b = 12 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → f x ≤ b) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 5 ∧ f x = b) :=
by
  sorry

end max_value_of_f_l1490_149048


namespace vector_relationships_l1490_149011

/-- Given vector a and unit vector b, prove their parallel and perpendicular relationships -/
theorem vector_relationships (a b : ℝ × ℝ) :
  a = (3, 4) →
  norm b = 1 →
  (b.1 * a.2 = b.2 * a.1 → b = (3/5, 4/5) ∨ b = (-3/5, -4/5)) ∧
  (b.1 * a.1 + b.2 * a.2 = 0 → b = (-4/5, 3/5) ∨ b = (4/5, -3/5)) :=
by sorry

end vector_relationships_l1490_149011


namespace parabola_focus_distance_l1490_149075

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define perpendicularity of PA to directrix
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  A.1 = -2 ∧ P.2 = A.2

-- Define the slope of AF
def slope_AF (A : ℝ × ℝ) : Prop :=
  (A.2 - 0) / (A.1 - 2) = -Real.sqrt 3

-- Theorem statement
theorem parabola_focus_distance 
  (P : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (h1 : point_on_parabola P) 
  (h2 : perpendicular_to_directrix P A) 
  (h3 : slope_AF A) : 
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 8 :=
sorry

end parabola_focus_distance_l1490_149075


namespace kids_savings_l1490_149096

/-- The total amount saved by three kids given their coin collections -/
def total_savings (teagan_pennies rex_nickels toni_dimes : ℕ) : ℚ :=
  (teagan_pennies : ℚ) * (1 / 100) +
  (rex_nickels : ℚ) * (5 / 100) +
  (toni_dimes : ℚ) * (10 / 100)

/-- Theorem stating that the total savings of the three kids is $40 -/
theorem kids_savings : total_savings 200 100 330 = 40 := by
  sorry

end kids_savings_l1490_149096


namespace equation_solution_l1490_149061

theorem equation_solution (a b : ℂ) (h1 : (2 : ℂ) * a ≠ 0) (h2 : (2 : ℂ) * a + (3 : ℂ) * b ≠ 0) 
  (h3 : ((2 : ℂ) * a + (3 : ℂ) * b) / ((2 : ℂ) * a) = ((3 : ℂ) * b) / ((2 : ℂ) * a + (3 : ℂ) * b)) :
  (a.im ≠ 0 ∧ b.im = 0) ∨ (a.im = 0 ∧ b.im ≠ 0) ∨ (a.im ≠ 0 ∧ b.im ≠ 0) :=
by sorry

end equation_solution_l1490_149061


namespace fraction_product_l1490_149023

theorem fraction_product : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by
  sorry

end fraction_product_l1490_149023


namespace fraction_equality_l1490_149068

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 9 / 53 := by
  sorry

end fraction_equality_l1490_149068


namespace meaningful_range_l1490_149049

def is_meaningful (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -1 ∧ x ≠ 0 :=
by sorry

end meaningful_range_l1490_149049


namespace unique_number_with_three_prime_divisors_l1490_149058

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 11))) →
  11 ∣ x →
  x = 59048 :=
by sorry

end unique_number_with_three_prime_divisors_l1490_149058


namespace max_cards_jasmine_can_buy_l1490_149069

/-- The maximum number of cards Jasmine can buy given her budget and the pricing conditions --/
theorem max_cards_jasmine_can_buy :
  let initial_price : ℚ := 95 / 100  -- $0.95 per card
  let discounted_price : ℚ := 85 / 100  -- $0.85 per card
  let budget : ℚ := 9  -- $9.00 budget
  let discount_threshold : ℕ := 6  -- Discount applies after 6 cards

  ∃ (n : ℕ), 
    (n ≤ discount_threshold ∧ n * initial_price ≤ budget) ∨
    (n > discount_threshold ∧ 
     discount_threshold * initial_price + (n - discount_threshold) * discounted_price ≤ budget) ∧
    ∀ (m : ℕ), m > n → 
      (m ≤ discount_threshold → m * initial_price > budget) ∧
      (m > discount_threshold → 
       discount_threshold * initial_price + (m - discount_threshold) * discounted_price > budget) ∧
    n = 9 :=
by
  sorry


end max_cards_jasmine_can_buy_l1490_149069


namespace chef_potato_problem_l1490_149041

/-- The number of potatoes a chef needs to cook -/
def total_potatoes (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  already_cooked + (remaining_cooking_time / cooking_time_per_potato)

/-- Proof that the chef needs to cook 12 potatoes in total -/
theorem chef_potato_problem : 
  total_potatoes 6 6 36 = 12 := by
  sorry

end chef_potato_problem_l1490_149041


namespace sum_of_coefficients_l1490_149090

theorem sum_of_coefficients : 
  let p (x : ℝ) := -3*(x^8 - x^5 + 2*x^3 - 6) + 5*(x^4 + 3*x^2) - 4*(x^6 - 5)
  p 1 = 48 := by sorry

end sum_of_coefficients_l1490_149090


namespace unique_four_digit_products_l1490_149036

def digit_product (n : ℕ) : ℕ :=
  if n < 1000 ∨ n > 9999 then 0
  else (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def is_unique_product (n : ℕ) : Prop :=
  ∃! x : ℕ, x ≥ 1000 ∧ x ≤ 9999 ∧ digit_product x = n

theorem unique_four_digit_products :
  {n : ℕ | is_unique_product n} = {1, 625, 2401, 4096, 6561} :=
by sorry

end unique_four_digit_products_l1490_149036


namespace right_triangle_area_from_sticks_l1490_149039

/-- Represents a stick of length 24 cm that can be broken into two pieces -/
structure Stick :=
  (length : ℝ := 24)
  (piece1 : ℝ)
  (piece2 : ℝ)
  (break_constraint : piece1 + piece2 = length)

/-- Represents a right triangle formed from three sticks -/
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (hypotenuse : ℝ)
  (pythagorean : leg1^2 + leg2^2 = hypotenuse^2)

/-- Theorem stating that if a right triangle can be formed from three 24 cm sticks
    (one of which is broken), then its area is 216 square centimeters -/
theorem right_triangle_area_from_sticks 
  (s1 s2 : Stick) (s3 : Stick) (t : RightTriangle)
  (h1 : s1.length = 24 ∧ s2.length = 24 ∧ s3.length = 24)
  (h2 : t.leg1 = s1.piece1 ∧ t.leg2 = s2.length ∧ t.hypotenuse = s1.piece2 + s3.length) :
  t.leg1 * t.leg2 / 2 = 216 := by
  sorry

end right_triangle_area_from_sticks_l1490_149039


namespace cost_price_of_ball_l1490_149040

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) :
  selling_price = 720 →
  num_balls_sold = 20 →
  num_balls_loss = 5 →
  ∃ (cost_price : ℕ),
    cost_price * num_balls_sold - selling_price = cost_price * num_balls_loss ∧
    cost_price = 48 := by
  sorry

end cost_price_of_ball_l1490_149040


namespace one_third_of_nine_x_minus_three_l1490_149063

theorem one_third_of_nine_x_minus_three (x : ℝ) : (1 / 3) * (9 * x - 3) = 3 * x - 1 := by
  sorry

end one_third_of_nine_x_minus_three_l1490_149063


namespace polygon_area_l1490_149082

-- Define the polygon
structure Polygon :=
  (sides : ℕ)
  (perimeter : ℝ)
  (num_squares : ℕ)
  (congruent_sides : Bool)
  (perpendicular_sides : Bool)

-- Define the properties of our specific polygon
def special_polygon : Polygon :=
  { sides := 28,
    perimeter := 56,
    num_squares := 25,
    congruent_sides := true,
    perpendicular_sides := true }

-- Theorem statement
theorem polygon_area (p : Polygon) (h1 : p = special_polygon) : 
  (p.perimeter / p.sides)^2 * p.num_squares = 100 := by
  sorry

end polygon_area_l1490_149082


namespace stratified_sampling_l1490_149098

theorem stratified_sampling (total_students : ℕ) (sample_size : ℕ) (first_grade : ℕ) (second_grade : ℕ) :
  total_students = 2000 →
  sample_size = 100 →
  first_grade = 30 →
  second_grade = 30 →
  sample_size = first_grade + second_grade + (sample_size - first_grade - second_grade) →
  (sample_size - first_grade - second_grade) = 40 :=
by sorry

end stratified_sampling_l1490_149098


namespace shaded_area_rectangle_l1490_149025

/-- The area of the shaded region in a rectangle with specific dimensions and unshaded triangles --/
theorem shaded_area_rectangle (rectangle_length : ℝ) (rectangle_width : ℝ)
  (triangle_base : ℝ) (triangle_height : ℝ) :
  rectangle_length = 12 →
  rectangle_width = 5 →
  triangle_base = 2 →
  triangle_height = 5 →
  rectangle_length * rectangle_width - 2 * (1/2 * triangle_base * triangle_height) = 50 := by
  sorry

end shaded_area_rectangle_l1490_149025


namespace full_price_tickets_l1490_149015

theorem full_price_tickets (total : ℕ) (reduced : ℕ) (h1 : total = 25200) (h2 : reduced = 5400) :
  total - reduced = 19800 := by
  sorry

end full_price_tickets_l1490_149015


namespace original_room_population_l1490_149001

theorem original_room_population (x : ℚ) : 
  (x / 4 : ℚ) - (x / 12 : ℚ) = 15 → x = 60 := by
  sorry

end original_room_population_l1490_149001


namespace cargo_volume_maximized_l1490_149097

/-- Represents the number of round trips as a function of the number of small boats towed -/
def roundTrips (x : ℝ) : ℝ := -2 * x + 24

/-- Represents the total cargo volume as a function of the number of small boats towed -/
def cargoVolume (x : ℝ) (M : ℝ) : ℝ := M * x * roundTrips x

theorem cargo_volume_maximized :
  ∀ M : ℝ, M > 0 →
  ∀ x : ℝ, x > 0 →
  cargoVolume 6 M ≥ cargoVolume x M ∧
  roundTrips 4 = 16 ∧
  roundTrips 7 = 10 :=
sorry

end cargo_volume_maximized_l1490_149097


namespace min_value_fraction_l1490_149050

theorem min_value_fraction (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  1 / x + 4 / (1 - x) ≥ 9 ∧
  (1 / x + 4 / (1 - x) = 9 ↔ x = 1 / 3) :=
by sorry

end min_value_fraction_l1490_149050


namespace cookie_distribution_l1490_149028

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 :=
by
  sorry

end cookie_distribution_l1490_149028


namespace ice_water_volume_change_l1490_149084

theorem ice_water_volume_change (v : ℝ) (h : v > 0) :
  let ice_volume := v * (1 + 1/11)
  (ice_volume - v) / ice_volume = 1/12 := by
sorry

end ice_water_volume_change_l1490_149084


namespace polynomial_factorization_l1490_149003

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) := by
  sorry

end polynomial_factorization_l1490_149003


namespace regular_polygon_interior_angle_l1490_149080

theorem regular_polygon_interior_angle (n : ℕ) (h : n > 2) :
  (n - 2) * 180 / n = 140 → n = 9 := by sorry

end regular_polygon_interior_angle_l1490_149080


namespace circle_center_and_radius_l1490_149062

theorem circle_center_and_radius :
  let equation := (fun (x y : ℝ) => x^2 + y^2 - 2*x - 5 = 0)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) ∧ 
    radius = Real.sqrt 6 ∧
    ∀ (x y : ℝ), equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l1490_149062


namespace marbles_problem_l1490_149085

theorem marbles_problem (initial_marbles given_marbles remaining_marbles : ℕ) : 
  given_marbles = 8 → remaining_marbles = 24 → initial_marbles = given_marbles + remaining_marbles →
  initial_marbles = 32 := by
sorry

end marbles_problem_l1490_149085


namespace midpoint_trajectory_l1490_149076

/-- The trajectory of the midpoint of a line segment connecting a point on a unit circle to a fixed point -/
theorem midpoint_trajectory (a b x y : ℝ) : 
  a^2 + b^2 = 1 →  -- P(a,b) is on the unit circle
  x = (a + 3) / 2 ∧ y = b / 2 →  -- M(x,y) is the midpoint of PQ
  (2*x - 3)^2 + 4*y^2 = 1 := by sorry

end midpoint_trajectory_l1490_149076


namespace bumper_car_line_problem_l1490_149073

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 10 + 15 = 17) → initial_people = 12 := by
  sorry

end bumper_car_line_problem_l1490_149073


namespace birthday_paradox_l1490_149065

theorem birthday_paradox (people : Finset ℕ) (birthdays : ℕ → Fin 366) :
  people.card = 367 → ∃ i j : ℕ, i ∈ people ∧ j ∈ people ∧ i ≠ j ∧ birthdays i = birthdays j :=
sorry

end birthday_paradox_l1490_149065


namespace nolan_saving_months_l1490_149035

def monthly_savings : ℕ := 3000
def total_saved : ℕ := 36000

theorem nolan_saving_months :
  total_saved / monthly_savings = 12 :=
by sorry

end nolan_saving_months_l1490_149035


namespace quadratic_function_properties_l1490_149054

/-- Quadratic function y = ax² - 4ax + 3a -/
def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 * a

theorem quadratic_function_properties :
  (∀ x, quadratic_function 1 x ≥ -1) ∧
  (∃ x, quadratic_function 1 x = -1) ∧
  (∀ x ∈ Set.Icc 1 4, quadratic_function (4/3) x ≤ 4) ∧
  (∃ x ∈ Set.Icc 1 4, quadratic_function (4/3) x = 4) :=
by sorry

end quadratic_function_properties_l1490_149054


namespace frood_game_theorem_l1490_149072

/-- Score for dropping n froods -/
def droppingScore (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Score for eating n froods -/
def eatingScore (n : ℕ) : ℕ := n^2

/-- The least number of froods for which dropping earns more points than eating -/
def leastFroods : ℕ := 21

theorem frood_game_theorem :
  (∀ k < leastFroods, droppingScore k ≤ eatingScore k) ∧
  (droppingScore leastFroods > eatingScore leastFroods) :=
sorry

end frood_game_theorem_l1490_149072


namespace equation_solutions_l1490_149078

theorem equation_solutions :
  (∀ x : ℝ, (x + 2)^2 = 3*(x + 2) ↔ x = -2 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13) := by
  sorry

end equation_solutions_l1490_149078


namespace gcf_of_18_and_10_l1490_149053

theorem gcf_of_18_and_10 (h : Nat.lcm 18 10 = 36) : Nat.gcd 18 10 = 5 := by
  sorry

end gcf_of_18_and_10_l1490_149053


namespace arccos_cos_eleven_l1490_149042

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 4 * Real.pi := by sorry

end arccos_cos_eleven_l1490_149042


namespace product_of_A_and_B_l1490_149059

theorem product_of_A_and_B (A B : ℝ) (h1 : 3/9 = 6/A) (h2 : 6/A = B/63) : A * B = 378 := by
  sorry

end product_of_A_and_B_l1490_149059


namespace expression_evaluation_l1490_149056

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 - 3*z^2 + 2*x*y + 2*y*z - 2*x*z = -44 := by
  sorry

end expression_evaluation_l1490_149056


namespace distinct_values_in_sequence_l1490_149009

def is_valid_f (f : ℕ → ℕ) : Prop :=
  f 1 = 1 ∧
  (∀ a b : ℕ, 0 < a → 0 < b → a ≤ b → f a ≤ f b) ∧
  (∀ a : ℕ, 0 < a → f (2 * a) = f a + 1)

theorem distinct_values_in_sequence (f : ℕ → ℕ) (hf : is_valid_f f) :
  Finset.card (Finset.image f (Finset.range 2015)) = 11 := by
  sorry

end distinct_values_in_sequence_l1490_149009


namespace remainder_13_pow_2031_mod_100_l1490_149027

theorem remainder_13_pow_2031_mod_100 : 13^2031 % 100 = 17 := by
  sorry

end remainder_13_pow_2031_mod_100_l1490_149027


namespace mean_inequalities_l1490_149006

theorem mean_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  (x + y + z) / 3 > (x * y * z) ^ (1/3) ∧ (x * y * z) ^ (1/3) > 3 * x * y * z / (x * y + y * z + z * x) :=
by sorry

end mean_inequalities_l1490_149006


namespace flowers_per_bouquet_is_nine_l1490_149021

/-- Calculates the number of flowers per bouquet given the initial number of seeds,
    the number of flowers killed, and the number of bouquets to be made. -/
def flowersPerBouquet (seedsPerColor : ℕ) (redKilled yellowKilled orangeKilled purpleKilled : ℕ)
    (numBouquets : ℕ) : ℕ :=
  let redSurvived := seedsPerColor - redKilled
  let yellowSurvived := seedsPerColor - yellowKilled
  let orangeSurvived := seedsPerColor - orangeKilled
  let purpleSurvived := seedsPerColor - purpleKilled
  let totalSurvived := redSurvived + yellowSurvived + orangeSurvived + purpleSurvived
  totalSurvived / numBouquets

/-- Theorem stating that the number of flowers per bouquet is 9 under the given conditions. -/
theorem flowers_per_bouquet_is_nine :
  flowersPerBouquet 125 45 61 30 40 36 = 9 := by
  sorry

#eval flowersPerBouquet 125 45 61 30 40 36

end flowers_per_bouquet_is_nine_l1490_149021


namespace fundraiser_total_l1490_149005

/-- Calculates the total money raised from a class fundraiser --/
def totalMoneyRaised (numStudentsBrownies : ℕ) (numBrowniesPerStudent : ℕ) 
                     (numStudentsCookies : ℕ) (numCookiesPerStudent : ℕ)
                     (numStudentsDonuts : ℕ) (numDonutsPerStudent : ℕ)
                     (priceBrownie : ℚ) (priceCookie : ℚ) (priceDonut : ℚ) : ℚ :=
  (numStudentsBrownies * numBrowniesPerStudent : ℚ) * priceBrownie +
  (numStudentsCookies * numCookiesPerStudent : ℚ) * priceCookie +
  (numStudentsDonuts * numDonutsPerStudent : ℚ) * priceDonut

theorem fundraiser_total : 
  totalMoneyRaised 50 20 30 36 25 18 (3/2) (9/4) 3 = 5280 := by
  sorry

end fundraiser_total_l1490_149005


namespace not_parabola_l1490_149071

theorem not_parabola (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  ¬∃ (a b c : Real), ∀ (x y : Real),
    x^2 * Real.sin α + y^2 * Real.cos α = 1 ↔ y = a*x^2 + b*x + c :=
sorry

end not_parabola_l1490_149071


namespace equation_solution_l1490_149077

theorem equation_solution : ∃ x : ℚ, (1/8 : ℚ) + 8/x = 15/x + (1/15 : ℚ) ∧ x = 120 := by
  sorry

end equation_solution_l1490_149077


namespace sqrt_plus_square_zero_implies_diff_four_l1490_149064

theorem sqrt_plus_square_zero_implies_diff_four (m n : ℝ) : 
  Real.sqrt (m - 3) + (n + 1)^2 = 0 → m - n = 4 := by
sorry

end sqrt_plus_square_zero_implies_diff_four_l1490_149064


namespace right_triangle_segment_relation_l1490_149030

/-- Given a right-angled triangle with legs of lengths a and b, and a segment of length d
    connecting the right angle vertex to the hypotenuse forming an angle δ with leg a,
    prove that 1/d = (cos δ)/a + (sin δ)/b. -/
theorem right_triangle_segment_relation (a b d : ℝ) (δ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hd : d > 0) (hδ : 0 < δ ∧ δ < π / 2) :
    1 / d = (Real.cos δ) / a + (Real.sin δ) / b := by
  sorry

end right_triangle_segment_relation_l1490_149030


namespace inverse_mod_89_l1490_149032

theorem inverse_mod_89 (h : (9⁻¹ : ZMod 89) = 79) : (81⁻¹ : ZMod 89) = 11 := by
  sorry

end inverse_mod_89_l1490_149032


namespace max_min_difference_z_l1490_149019

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 15) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w, (∃ u v, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 15) → w ≤ z_max) ∧
    (∀ w, (∃ u v, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 15) → w ≥ z_min) ∧
    z_max - z_min = 8 :=
sorry

end max_min_difference_z_l1490_149019


namespace k_range_theorem_l1490_149055

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - x * log x + 2

theorem k_range_theorem (a b : ℝ) (h1 : 1/2 ≤ a) (h2 : a < b) 
  (h3 : ∀ x ∈ Set.Icc a b, ∃ k : ℝ, f x = k * (x + 2)) :
  ∃ k : ℝ, 1 < k ∧ k ≤ (9 + 2 * log 2) / 10 :=
sorry

end k_range_theorem_l1490_149055


namespace isosceles_tetrahedron_ratio_bounds_l1490_149008

/-- An isosceles tetrahedron with edge lengths a, b, and c. -/
structure IsoscelesTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The circumradius of the tetrahedron. -/
noncomputable def R (t : IsoscelesTetrahedron) : ℝ :=
  sorry

/-- The circumradius of the base triangle. -/
noncomputable def r (t : IsoscelesTetrahedron) : ℝ :=
  sorry

/-- The theorem stating the bounds of the ratio r/R. -/
theorem isosceles_tetrahedron_ratio_bounds (t : IsoscelesTetrahedron) :
    2 * Real.sqrt 2 / 3 ≤ r t / R t ∧ r t / R t < 1 := by
  sorry

end isosceles_tetrahedron_ratio_bounds_l1490_149008


namespace difference_local_face_value_65793_l1490_149037

/-- The difference between the local value and face value of a digit in a numeral -/
def local_face_value_difference (numeral : ℕ) (digit : ℕ) (place : ℕ) : ℕ :=
  digit * (10 ^ place) - digit

/-- The hundreds place in a decimal number system -/
def hundreds_place : ℕ := 2

theorem difference_local_face_value_65793 :
  local_face_value_difference 65793 7 hundreds_place = 693 := by
  sorry

end difference_local_face_value_65793_l1490_149037


namespace max_value_of_f_l1490_149026

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 12*x + 16

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 3 → f y ≤ f x) ∧
  f x = 32 := by
  sorry

end max_value_of_f_l1490_149026


namespace cube_surface_area_increase_l1490_149031

/-- Theorem: When each edge of a cube is increased by p%, 
    the surface area of the cube is increased by 2p + (p^2/100)%. -/
theorem cube_surface_area_increase (p : ℝ) :
  let original_edge : ℝ → ℝ := λ s => s
  let increased_edge : ℝ → ℝ := λ s => s * (1 + p / 100)
  let original_surface_area : ℝ → ℝ := λ s => 6 * s^2
  let increased_surface_area : ℝ → ℝ := λ s => 6 * (increased_edge s)^2
  let percent_increase : ℝ → ℝ := λ s => 
    (increased_surface_area s - original_surface_area s) / original_surface_area s * 100
  ∀ s > 0, percent_increase s = 2 * p + p^2 / 100 :=
by sorry


end cube_surface_area_increase_l1490_149031


namespace wen_family_movie_cost_l1490_149088

def ticket_cost (regular_price : ℚ) (discount : ℚ) : ℚ :=
  regular_price * (1 - discount)

theorem wen_family_movie_cost :
  let senior_price : ℚ := 6
  let senior_discount : ℚ := 1/4
  let children_discount : ℚ := 1/2
  let regular_price : ℚ := senior_price / (1 - senior_discount)
  let num_people_per_generation : ℕ := 2
  
  num_people_per_generation * senior_price +
  num_people_per_generation * regular_price +
  num_people_per_generation * (ticket_cost regular_price children_discount) = 36
  := by sorry

end wen_family_movie_cost_l1490_149088


namespace group_distribution_methods_l1490_149079

theorem group_distribution_methods (total_boys : ℕ) (total_girls : ℕ)
  (group_size : ℕ) (boys_per_group : ℕ) (girls_per_group : ℕ) :
  total_boys = 6 →
  total_girls = 4 →
  group_size = 5 →
  boys_per_group = 3 →
  girls_per_group = 2 →
  (Nat.choose total_boys boys_per_group * Nat.choose total_girls girls_per_group) / 2 = 60 :=
by sorry

end group_distribution_methods_l1490_149079


namespace calculation_difference_l1490_149060

theorem calculation_difference : 
  (0.70 * 120 - ((6/9) * 150 / (0.80 * 250))) - (0.18 * 180 * (5/7) * 210) = -4776.5 := by
  sorry

end calculation_difference_l1490_149060


namespace probability_at_least_one_woman_l1490_149074

def total_employees : ℕ := 10
def men : ℕ := 6
def women : ℕ := 4
def unavailable_men : ℕ := 1
def unavailable_women : ℕ := 1
def selection_size : ℕ := 3

def available_men : ℕ := men - unavailable_men
def available_women : ℕ := women - unavailable_women
def total_available : ℕ := available_men + available_women

theorem probability_at_least_one_woman :
  (1 - (Nat.choose available_men selection_size : ℚ) / (Nat.choose total_available selection_size : ℚ)) = 23/28 := by
  sorry

end probability_at_least_one_woman_l1490_149074


namespace octal_subtraction_theorem_l1490_149091

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Conversion from octal to decimal --/
def from_octal (n : OctalNumber) : ℕ :=
  sorry

theorem octal_subtraction_theorem :
  octal_sub (to_octal 52) (to_octal 27) = to_octal 25 :=
sorry

end octal_subtraction_theorem_l1490_149091
