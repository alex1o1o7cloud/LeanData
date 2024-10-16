import Mathlib

namespace NUMINAMATH_CALUDE_joe_fruit_probability_l1990_199050

def num_meals : ℕ := 4
def num_fruit_types : ℕ := 3

def prob_same_fruit_all_meals : ℚ := (1 / num_fruit_types) ^ num_meals

theorem joe_fruit_probability :
  1 - (num_fruit_types * prob_same_fruit_all_meals) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l1990_199050


namespace NUMINAMATH_CALUDE_church_capacity_l1990_199008

/-- Calculates the number of usable chairs in a church with three sections -/
def total_usable_chairs : ℕ :=
  let section1_rows : ℕ := 15
  let section1_chairs_per_row : ℕ := 8
  let section1_unusable_per_row : ℕ := 3
  let section2_rows : ℕ := 20
  let section2_chairs_per_row : ℕ := 6
  let section2_unavailable_rows : ℕ := 2
  let section3_rows : ℕ := 25
  let section3_chairs_per_row : ℕ := 10
  let section3_unusable_every_second : ℕ := 5

  let section1_usable := section1_rows * (section1_chairs_per_row - section1_unusable_per_row)
  let section2_usable := (section2_rows - section2_unavailable_rows) * section2_chairs_per_row
  let section3_usable := (section3_rows / 2) * section3_chairs_per_row + 
                         (section3_rows - section3_rows / 2) * (section3_chairs_per_row - section3_unusable_every_second)

  section1_usable + section2_usable + section3_usable

theorem church_capacity : total_usable_chairs = 373 := by
  sorry

end NUMINAMATH_CALUDE_church_capacity_l1990_199008


namespace NUMINAMATH_CALUDE_equal_values_l1990_199000

-- Define the algebraic expression
def f (a : ℝ) : ℝ := a^4 - 2*a^2 + 3

-- State the theorem
theorem equal_values : f 2 = f (-2) := by sorry

end NUMINAMATH_CALUDE_equal_values_l1990_199000


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_tangent_points_coordinates_l1990_199060

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

-- Theorem to prove the y-coordinates
theorem y_coordinates (x : ℝ) :
  (x = 1 ∨ x = -1) → (f x = 0 ∨ f x = -4) :=
by sorry

-- Main theorem combining the above results
theorem tangent_points_coordinates :
  ∃ (x y : ℝ), (f' x = 4 ∧ f x = y) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_y_coordinates_tangent_points_coordinates_l1990_199060


namespace NUMINAMATH_CALUDE_two_balls_same_box_probability_l1990_199073

theorem two_balls_same_box_probability :
  let num_balls : ℕ := 3
  let num_boxes : ℕ := 5
  let total_outcomes : ℕ := num_boxes ^ num_balls
  let favorable_outcomes : ℕ := (num_balls.choose 2) * num_boxes * (num_boxes - 1)
  favorable_outcomes / total_outcomes = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_two_balls_same_box_probability_l1990_199073


namespace NUMINAMATH_CALUDE_equation_solutions_inequality_system_solution_l1990_199006

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - 2*x - 4 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := 4*(x - 1) < x + 2 ∧ (x + 7) / 3 > x

-- Theorem for the equation solutions
theorem equation_solutions : 
  ∃ (x1 x2 : ℝ), x1 = 1 + Real.sqrt 5 ∧ x2 = 1 - Real.sqrt 5 ∧ 
  equation x1 ∧ equation x2 ∧ 
  ∀ (x : ℝ), equation x → x = x1 ∨ x = x2 := by sorry

-- Theorem for the inequality system solution
theorem inequality_system_solution :
  ∀ (x : ℝ), inequality_system x ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_inequality_system_solution_l1990_199006


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1990_199086

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1990_199086


namespace NUMINAMATH_CALUDE_children_per_seat_l1990_199067

theorem children_per_seat (total_children : ℕ) (total_seats : ℕ) (h1 : total_children = 58) (h2 : total_seats = 29) :
  total_children / total_seats = 2 := by
  sorry

end NUMINAMATH_CALUDE_children_per_seat_l1990_199067


namespace NUMINAMATH_CALUDE_arccos_negative_half_l1990_199048

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l1990_199048


namespace NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l1990_199034

-- Define the original equation
def original_equation (x y : ℝ) : Prop := y^4 - 16*x^4 = 8*y^2 - 4

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 4*x^2 = 4

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := y^2 + 4*x^2 = 4

-- Theorem stating that the original equation is equivalent to the union of a hyperbola and an ellipse
theorem original_eq_hyperbola_and_ellipse :
  ∀ x y : ℝ, original_equation x y ↔ (hyperbola_equation x y ∨ ellipse_equation x y) :=
sorry

end NUMINAMATH_CALUDE_original_eq_hyperbola_and_ellipse_l1990_199034


namespace NUMINAMATH_CALUDE_parking_theorem_l1990_199045

/-- The number of ways to park 5 trains on 5 tracks with one restriction -/
def parking_arrangements (n : ℕ) (restricted_train : ℕ) (restricted_track : ℕ) : ℕ :=
  (n - 1) * Nat.factorial (n - 1)

theorem parking_theorem :
  parking_arrangements 5 1 1 = 96 :=
by sorry

end NUMINAMATH_CALUDE_parking_theorem_l1990_199045


namespace NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l1990_199069

theorem p_or_q_necessary_not_sufficient :
  (∀ p q : Prop, (¬p → (p ∨ q))) ∧
  (∃ p q : Prop, (p ∨ q) ∧ ¬(¬p → False)) :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l1990_199069


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l1990_199064

theorem x_minus_y_equals_eight (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.25 * y) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l1990_199064


namespace NUMINAMATH_CALUDE_product_positive_l1990_199079

theorem product_positive (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^4 - y^4 > x) (h2 : y^4 - x^4 > y) : x * y > 0 :=
by sorry

end NUMINAMATH_CALUDE_product_positive_l1990_199079


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l1990_199032

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ ((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l1990_199032


namespace NUMINAMATH_CALUDE_nina_money_problem_l1990_199049

theorem nina_money_problem (x : ℚ) :
  (5 * x = 8 * (x - 1.25)) → (5 * x = 50 / 3) := by
  sorry

end NUMINAMATH_CALUDE_nina_money_problem_l1990_199049


namespace NUMINAMATH_CALUDE_largest_reciprocal_l1990_199022

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = -1/4 → b = 2/7 → c = -2 → d = 3 → e = -3/2 → 
  (1/b > 1/a ∧ 1/b > 1/c ∧ 1/b > 1/d ∧ 1/b > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l1990_199022


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1990_199082

def A (a : ℝ) : Set ℝ := {3, 4, a^2 - 3*a - 1}
def B (a : ℝ) : Set ℝ := {2*a, -3}

theorem set_intersection_problem (a : ℝ) :
  (A a ∩ B a = {-3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1990_199082


namespace NUMINAMATH_CALUDE_perpendicular_chords_sum_l1990_199040

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a chord passing through the focus
structure ChordThroughFocus where
  a : PointOnParabola
  b : PointOnParabola
  passes_through_focus : True  -- We assume this property without proving it

-- Define perpendicular chords
def perpendicular (c1 c2 : ChordThroughFocus) : Prop := True  -- We assume this property without proving it

-- Define the length of a chord
noncomputable def chord_length (c : ChordThroughFocus) : ℝ := sorry

-- Theorem statement
theorem perpendicular_chords_sum (ab cd : ChordThroughFocus) 
  (h_perp : perpendicular ab cd) : 
  1 / chord_length ab + 1 / chord_length cd = 1/4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_chords_sum_l1990_199040


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_l1990_199063

theorem equidistant_points_on_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (4 * x₁ + 3 * y₁ = 12) ∧
    (4 * x₂ + 3 * y₂ = 12) ∧
    (|x₁| = |y₁|) ∧
    (|x₂| = |y₂|) ∧
    (x₁ > 0 ∧ y₁ > 0) ∧
    (x₂ > 0 ∧ y₂ < 0) ∧
    ¬∃ (x₃ y₃ : ℝ),
      (4 * x₃ + 3 * y₃ = 12) ∧
      (|x₃| = |y₃|) ∧
      ((x₃ < 0 ∧ y₃ > 0) ∨ (x₃ < 0 ∧ y₃ < 0)) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_l1990_199063


namespace NUMINAMATH_CALUDE_tank_capacity_correct_l1990_199027

/-- The capacity of a tank in litres. -/
def tank_capacity : ℝ := 1592

/-- The time in hours it takes for the leak to empty the full tank. -/
def leak_empty_time : ℝ := 7

/-- The rate at which the inlet pipe fills the tank in litres per minute. -/
def inlet_rate : ℝ := 6

/-- The time in hours it takes to empty the tank when both inlet and leak are open. -/
def combined_empty_time : ℝ := 12

/-- Theorem stating that the tank capacity is correct given the conditions. -/
theorem tank_capacity_correct : 
  tank_capacity = 
    (inlet_rate * 60 * combined_empty_time * leak_empty_time) / 
    (leak_empty_time - combined_empty_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_correct_l1990_199027


namespace NUMINAMATH_CALUDE_sector_arc_length_l1990_199068

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  r = 3 → θ_deg = 150 → l = (5 * π) / 2 → 
  l = r * ((θ_deg * π) / 180) :=
sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1990_199068


namespace NUMINAMATH_CALUDE_expression_evaluation_l1990_199077

theorem expression_evaluation :
  let x : ℚ := -1/3
  (-5 * x^2 + 4 + x) - 3 * (-2 * x^2 + x - 1) = 70/9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1990_199077


namespace NUMINAMATH_CALUDE_orange_count_correct_l1990_199038

/-- Given the initial number of oranges, the number thrown away, and the number added,
    calculate the final number of oranges in the bin. -/
def final_oranges (initial thrown_away added : ℕ) : ℕ :=
  initial - thrown_away + added

/-- Theorem stating that the final number of oranges is correct. -/
theorem orange_count_correct (initial thrown_away added : ℕ) 
  (h : thrown_away ≤ initial) : 
  final_oranges initial thrown_away added = initial - thrown_away + added := by
  sorry

end NUMINAMATH_CALUDE_orange_count_correct_l1990_199038


namespace NUMINAMATH_CALUDE_total_salary_is_583_l1990_199033

/-- The total amount paid to two employees per week, given their relative salaries -/
def total_salary (n_salary : ℝ) : ℝ :=
  n_salary + 1.2 * n_salary

/-- Proof that the total salary for two employees is $583 per week -/
theorem total_salary_is_583 :
  total_salary 265 = 583 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_is_583_l1990_199033


namespace NUMINAMATH_CALUDE_range_of_m_l1990_199095

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (x - a) / 3 < 0 ∧ 2 * (x - 5) < 3 * x - 8

-- Define the solution set
def solution_set (a : ℝ) : Set ℤ :=
  {x : ℤ | inequality_system x a}

-- State the theorem
theorem range_of_m (a : ℝ) (m : ℝ) :
  (∀ x : ℤ, x ∈ solution_set a ↔ (x = -1 ∨ x = 0)) →
  (10 * a = 2 * m + 5) →
  -2.5 < m ∧ m ≤ 2.5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1990_199095


namespace NUMINAMATH_CALUDE_impossible_ratio_l1990_199072

theorem impossible_ratio (n : ℕ) (boys girls : ℕ) : 
  30 < n → n < 40 → boys + girls = n → ¬(3 * girls = 7 * boys) := by
  sorry

end NUMINAMATH_CALUDE_impossible_ratio_l1990_199072


namespace NUMINAMATH_CALUDE_problem_solution_l1990_199078

theorem problem_solution (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1990_199078


namespace NUMINAMATH_CALUDE_upstream_distance_l1990_199061

theorem upstream_distance
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h1 : boat_speed = 20)
  (h2 : downstream_distance = 96)
  (h3 : downstream_time = 3)
  (h4 : upstream_time = 11)
  : ∃ (upstream_distance : ℝ), upstream_distance = 88 :=
by
  sorry

#check upstream_distance

end NUMINAMATH_CALUDE_upstream_distance_l1990_199061


namespace NUMINAMATH_CALUDE_number_of_white_balls_l1990_199030

/-- Given a bag with red and white balls, prove the number of white balls when probability of drawing red is known -/
theorem number_of_white_balls (n : ℕ) : 
  (8 : ℚ) / (8 + n) = (2 : ℚ) / 5 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_white_balls_l1990_199030


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l1990_199092

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = 1 ∧ Real.arcsin x + Real.arcsin (x - 1) = Real.arccos (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l1990_199092


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1990_199042

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to calculate the units digit of a^n
def powerUnitsDigit (a n : ℕ) : ℕ :=
  unitsDigit ((unitsDigit a)^n)

theorem units_digit_sum_of_powers : 
  unitsDigit ((35 : ℕ)^87 + (93 : ℕ)^53) = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1990_199042


namespace NUMINAMATH_CALUDE_same_color_probability_l1990_199016

/-- The probability of drawing two balls of the same color from a bag containing green and white balls. -/
theorem same_color_probability (green white : ℕ) (h : green = 10 ∧ white = 8) :
  let total := green + white
  let prob_green := (green * (green - 1)) / (total * (total - 1))
  let prob_white := (white * (white - 1)) / (total * (total - 1))
  (prob_green + prob_white : ℚ) = 73 / 153 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l1990_199016


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1990_199031

theorem degree_to_radian_conversion (θ_deg : ℝ) (θ_rad : ℝ) :
  θ_deg = 150 ∧ θ_rad = θ_deg * (π / 180) → θ_rad = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1990_199031


namespace NUMINAMATH_CALUDE_other_x_intercept_l1990_199001

/-- A quadratic function with vertex (5, 10) and one x-intercept at (-1, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a ≠ 0 → -b / (2 * a) = 5
  vertex_y : a ≠ 0 → a * 5^2 + b * 5 + c = 10
  x_intercept : a * (-1)^2 + b * (-1) + c = 0

/-- The x-coordinate of the other x-intercept is 11 -/
theorem other_x_intercept (f : QuadraticFunction) :
  ∃ x : ℝ, x ≠ -1 ∧ f.a * x^2 + f.b * x + f.c = 0 ∧ x = 11 :=
sorry

end NUMINAMATH_CALUDE_other_x_intercept_l1990_199001


namespace NUMINAMATH_CALUDE_f_properties_l1990_199056

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)

theorem f_properties :
  (∀ x : ℝ, f x > 0 ↔ x > 0) ∧
  (∀ s t : ℝ, s > 0 → t > 0 → f (s + t) > f s + f t) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1990_199056


namespace NUMINAMATH_CALUDE_exponential_shift_fixed_point_l1990_199012

/-- For a > 0 and a ≠ 1, the function f(x) = a^(x-2) passes through the point (2, 1) -/
theorem exponential_shift_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2)
  f 2 = 1 := by sorry

end NUMINAMATH_CALUDE_exponential_shift_fixed_point_l1990_199012


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l1990_199019

/-- The function f(x) = x^3 + 3x^2 + 6x - 10 --/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem smallest_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (a*x + b*y + c = 0 ↔ y - f x = f' x * (x - x)))  -- Tangent line equation
    ∧ (∀ x₀ : ℝ, f' x₀ ≥ f' (-1))  -- Slope at x = -1 is the smallest
    ∧ a = 3 ∧ b = -1 ∧ c = -11  -- Coefficients of the tangent line equation
:= by sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l1990_199019


namespace NUMINAMATH_CALUDE_min_a_for_p_true_l1990_199080

-- Define the set of x
def X : Set ℝ := { x | 1 ≤ x ∧ x ≤ 9 }

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x ∈ X, x^2 - a*x + 36 ≤ 0

-- Theorem statement
theorem min_a_for_p_true : 
  (∃ a : ℝ, p a) → (∀ a : ℝ, p a → a ≥ 12) ∧ p 12 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_p_true_l1990_199080


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1990_199051

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1990_199051


namespace NUMINAMATH_CALUDE_lemons_for_lemonade_l1990_199046

/-- Given that 30 lemons make 40 gallons of lemonade, 
    prove that 7.5 lemons are needed for 10 gallons. -/
theorem lemons_for_lemonade :
  let lemons_for_40 : ℚ := 30
  let gallons_40 : ℚ := 40
  let target_gallons : ℚ := 10
  (lemons_for_40 / gallons_40) * target_gallons = 7.5 := by sorry

end NUMINAMATH_CALUDE_lemons_for_lemonade_l1990_199046


namespace NUMINAMATH_CALUDE_cube_monotone_l1990_199025

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l1990_199025


namespace NUMINAMATH_CALUDE_bees_after_five_days_l1990_199047

/-- The number of bees in the hive after n days -/
def bees_after_days (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * bees_after_days (n - 1)

/-- The theorem stating that after 5 days, there will be 1024 bees in the hive -/
theorem bees_after_five_days : bees_after_days 5 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_bees_after_five_days_l1990_199047


namespace NUMINAMATH_CALUDE_beach_trip_driving_time_l1990_199055

theorem beach_trip_driving_time :
  ∀ (x : ℝ),
  (2.5 * (2 * x) + 2 * x = 14) →
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_beach_trip_driving_time_l1990_199055


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1990_199029

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (α : Plane) (a b : Line) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1990_199029


namespace NUMINAMATH_CALUDE_ruby_count_l1990_199015

theorem ruby_count (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 :=
by sorry

end NUMINAMATH_CALUDE_ruby_count_l1990_199015


namespace NUMINAMATH_CALUDE_megan_carrots_count_l1990_199035

/-- The total number of carrots Megan has after picking, throwing out some, and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that Megan's total carrots can be calculated using the given formula. -/
theorem megan_carrots_count (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ)
    (h1 : initial ≥ thrown_out) :
  total_carrots initial thrown_out picked_next_day = initial - thrown_out + picked_next_day :=
by
  sorry

#eval total_carrots 19 4 46  -- Should evaluate to 61

end NUMINAMATH_CALUDE_megan_carrots_count_l1990_199035


namespace NUMINAMATH_CALUDE_triangle_area_with_given_sides_l1990_199081

theorem triangle_area_with_given_sides :
  let a : ℝ := 65
  let b : ℝ := 60
  let c : ℝ := 25
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 750 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_sides_l1990_199081


namespace NUMINAMATH_CALUDE_highest_power_of_three_N_l1990_199057

/-- Concatenates a list of integers into a single integer -/
def concatenate_integers (list : List Int) : Int :=
  sorry

/-- Generates a list of 2-digit integers from 73 to 29 in descending order -/
def generate_list : List Int :=
  sorry

/-- The number N formed by concatenating 2-digit integers from 73 to 29 in descending order -/
def N : Int := concatenate_integers generate_list

/-- The highest power of 3 that divides a given integer -/
def highest_power_of_three (n : Int) : Int :=
  sorry

theorem highest_power_of_three_N :
  highest_power_of_three N = 0 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_N_l1990_199057


namespace NUMINAMATH_CALUDE_h_of_three_equals_five_l1990_199097

-- Define the function h
def h (x : ℝ) : ℝ := 2*(x-2) + 3

-- State the theorem
theorem h_of_three_equals_five : h 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_h_of_three_equals_five_l1990_199097


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1990_199059

theorem nested_fraction_evaluation :
  1 + 3 / (4 + 5 / (6 + 7/8)) = 85/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1990_199059


namespace NUMINAMATH_CALUDE_jonathan_first_name_length_l1990_199043

/-- The number of letters in Jonathan's first name -/
def jonathan_first_name : ℕ := by sorry

/-- The number of letters in Jonathan's surname -/
def jonathan_surname : ℕ := 10

/-- The number of letters in Jonathan's sister's first name -/
def sister_first_name : ℕ := 5

/-- The number of letters in Jonathan's sister's surname -/
def sister_surname : ℕ := 10

/-- The total number of letters in both their names -/
def total_letters : ℕ := 33

theorem jonathan_first_name_length :
  jonathan_first_name = 8 :=
by
  have h1 : jonathan_first_name + jonathan_surname + sister_first_name + sister_surname = total_letters := by sorry
  sorry

end NUMINAMATH_CALUDE_jonathan_first_name_length_l1990_199043


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1990_199071

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 7 → x^4 + 1/x^4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1990_199071


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l1990_199020

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(((m + 1) % 12 = 0) ∧ ((m + 1) % 18 = 0) ∧ ((m + 1) % 24 = 0) ∧ ((m + 1) % 32 = 0) ∧ ((m + 1) % 40 = 0))) ∧
  ((n + 1) % 12 = 0) ∧ ((n + 1) % 18 = 0) ∧ ((n + 1) % 24 = 0) ∧ ((n + 1) % 32 = 0) ∧ ((n + 1) % 40 = 0) →
  n = 2879 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l1990_199020


namespace NUMINAMATH_CALUDE_time_to_cut_one_piece_l1990_199018

-- Define the total number of pieces
def total_pieces : ℕ := 146

-- Define the total time taken in seconds
def total_time : ℕ := 580

-- Define the time taken to cut one piece
def time_per_piece : ℚ := total_time / total_pieces

-- Theorem to prove
theorem time_to_cut_one_piece : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) ∧ |time_per_piece - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_time_to_cut_one_piece_l1990_199018


namespace NUMINAMATH_CALUDE_factorial_equality_l1990_199075

theorem factorial_equality (N : ℕ) (h : N > 0) :
  (7 : ℕ).factorial * (11 : ℕ).factorial = 18 * N.factorial → N = 11 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l1990_199075


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1990_199098

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3) = 3 * q^3 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1990_199098


namespace NUMINAMATH_CALUDE_complete_square_plus_integer_l1990_199094

theorem complete_square_plus_integer :
  ∃ (k : ℤ) (b : ℝ), ∀ (x : ℝ), x^2 + 14*x + 60 = (x + b)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_complete_square_plus_integer_l1990_199094


namespace NUMINAMATH_CALUDE_three_numbers_ratio_l1990_199053

theorem three_numbers_ratio (a b c : ℝ) : 
  (a : ℝ) / 2 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 4 →
  a^2 + b^2 + c^2 = 725 →
  (a = 10 ∧ b = 15 ∧ c = 20) ∨ (a = -10 ∧ b = -15 ∧ c = -20) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_ratio_l1990_199053


namespace NUMINAMATH_CALUDE_reciprocal_inequality_not_always_true_l1990_199083

theorem reciprocal_inequality_not_always_true : 
  ¬ (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x > y → (1 / x) < (1 / y)) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_not_always_true_l1990_199083


namespace NUMINAMATH_CALUDE_range_of_a_l1990_199028

theorem range_of_a (p q : Prop) (h1 : ∀ x : ℝ, x > 0 → x + 1/x > a → a < 2)
  (h2 : (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) → (a ≤ -1 ∨ a ≥ 1))
  (h3 : q) (h4 : ¬p) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1990_199028


namespace NUMINAMATH_CALUDE_polygon_diagonals_minus_sides_l1990_199090

theorem polygon_diagonals_minus_sides (n : ℕ) (h : n = 105) : 
  (n * (n - 3)) / 2 - n = 5250 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_minus_sides_l1990_199090


namespace NUMINAMATH_CALUDE_plot_perimeter_l1990_199054

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ
  length_width_relation : length = width + 10
  cost_equation : fencing_cost = (2 * (length + width)) * fencing_rate

/-- The perimeter of the rectangular plot is 300 meters -/
theorem plot_perimeter (plot : RectangularPlot) 
  (h : plot.fencing_rate = 6.5 ∧ plot.fencing_cost = 1950) : 
  2 * (plot.length + plot.width) = 300 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_l1990_199054


namespace NUMINAMATH_CALUDE_winter_clothes_cost_theorem_l1990_199089

/-- Represents the cost calculation for winter clothes with a discount --/
def winter_clothes_cost (total_children : ℕ) (toddlers : ℕ) 
  (toddler_cost school_cost preteen_cost teen_cost : ℕ) 
  (discount_percent : ℕ) : ℕ :=
  let school_age := 2 * toddlers
  let preteens := school_age / 2
  let teens := 4 * toddlers + toddlers
  let total_cost := toddler_cost * toddlers + 
                    school_cost * school_age + 
                    preteen_cost * preteens + 
                    teen_cost * teens
  let discount := preteen_cost * preteens * discount_percent / 100
  total_cost - discount

/-- Theorem stating the total cost of winter clothes after discount --/
theorem winter_clothes_cost_theorem :
  winter_clothes_cost 60 6 35 45 55 65 30 = 2931 := by
  sorry

#eval winter_clothes_cost 60 6 35 45 55 65 30

end NUMINAMATH_CALUDE_winter_clothes_cost_theorem_l1990_199089


namespace NUMINAMATH_CALUDE_trig_identity_l1990_199009

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^2 + (Real.cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1990_199009


namespace NUMINAMATH_CALUDE_solution_value_l1990_199041

theorem solution_value (p q : ℝ) : 
  (3 * p^2 - 5 * p = 12) → 
  (3 * q^2 - 5 * q = 12) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1990_199041


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1990_199002

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1990_199002


namespace NUMINAMATH_CALUDE_power_of_power_l1990_199084

theorem power_of_power (k : ℕ+) : (k^5)^3 = k^5 * k^5 * k^5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1990_199084


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1990_199062

theorem cement_mixture_weight (sand_ratio : ℚ) (water_ratio : ℚ) (gravel_weight : ℚ) 
  (h1 : sand_ratio = 1/2)
  (h2 : water_ratio = 1/5)
  (h3 : gravel_weight = 15) :
  ∃ (total_weight : ℚ), 
    sand_ratio * total_weight + water_ratio * total_weight + gravel_weight = total_weight ∧
    total_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l1990_199062


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l1990_199066

theorem integer_roots_quadratic (p q : ℕ) : 
  (∃ x y : ℤ, x^2 - p*q*x + p + q = 0 ∧ y^2 - p*q*y + p + q = 0 ∧ x ≠ y) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 2) ∨ (p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l1990_199066


namespace NUMINAMATH_CALUDE_sum_of_trapezoid_angles_product_of_reciprocals_rational_square_plus_one_ge_one_rhombus_symmetry_l1990_199011

-- Definition of a trapezoid
structure Trapezoid where
  angles : Fin 4 → ℝ

-- Definition of reciprocals
def Reciprocals (x y : ℝ) : Prop := x * y = 1

-- Theorem statements
theorem sum_of_trapezoid_angles (t : Trapezoid) : 
  (Finset.sum Finset.univ t.angles) = 360 := by sorry

theorem product_of_reciprocals {x y : ℝ} (h : Reciprocals x y) : 
  x * y = 1 := by sorry

theorem rational_square_plus_one_ge_one (a : ℚ) : 
  a^2 + 1 ≥ 1 := by sorry

-- Definition of a rhombus
structure Rhombus where
  -- Add necessary fields

-- Definitions for symmetry properties
def CentrallySymmetric (R : Rhombus) : Prop := sorry
def Axisymmetric (R : Rhombus) : Prop := sorry

theorem rhombus_symmetry (R : Rhombus) : 
  CentrallySymmetric R ∧ Axisymmetric R := by sorry

end NUMINAMATH_CALUDE_sum_of_trapezoid_angles_product_of_reciprocals_rational_square_plus_one_ge_one_rhombus_symmetry_l1990_199011


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1990_199023

/-- An ellipse with equation x^2 / (2-k) + y^2 / (2k-1) = 1 and foci on the y-axis has k in the range (1, 2) -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (2-k) + y^2 / (2*k-1) = 1) →  -- equation represents an ellipse
  (∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, x^2 / (2-k) + y^2 / (2*k-1) = 1 → y^2 ≥ c^2) →  -- foci on y-axis
  1 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1990_199023


namespace NUMINAMATH_CALUDE_m_range_l1990_199037

/-- The statement "The equation x^2 + 2x + m = 0 has no real roots" -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

/-- The statement "The equation x^2/(m-1) + y^2 = 1 is an ellipse with foci on the x-axis" -/
def q (m : ℝ) : Prop := m > 2 ∧ ∀ x y : ℝ, x^2/(m-1) + y^2 = 1 → ∃ c : ℝ, c^2 = m - 1

theorem m_range (m : ℝ) : (¬(¬(p m)) ∧ ¬(p m ∧ q m)) → (1 < m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1990_199037


namespace NUMINAMATH_CALUDE_unique_representation_theorem_l1990_199070

-- Define a type for representing a person (boy or girl)
inductive Person : Type
| boy : Person
| girl : Person

-- Define a function to convert a natural number to a list of 5 binary digits
def toBinaryDigits (n : Nat) : List Bool :=
  List.reverse (List.take 5 (List.map (fun i => n / 2^i % 2 = 1) (List.range 5)))

-- Define a function to convert a list of binary digits to a list of persons
def binaryToPersons (bits : List Bool) : List Person :=
  List.map (fun b => if b then Person.boy else Person.girl) bits

-- Define a function to convert a list of persons back to a natural number
def personsToNumber (persons : List Person) : Nat :=
  List.foldl (fun acc p => 2 * acc + match p with
    | Person.boy => 1
    | Person.girl => 0) 0 persons

-- Theorem statement
theorem unique_representation_theorem (n : Nat) (h : n > 0 ∧ n ≤ 31) :
  ∃! (arrangement : List Person),
    arrangement.length = 5 ∧
    personsToNumber arrangement = n :=
  sorry

end NUMINAMATH_CALUDE_unique_representation_theorem_l1990_199070


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1990_199087

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a + b = 120 →      -- Sum of two angles is 4/3 of right angle (90° * 4/3 = 120°)
  b = a + 36 →       -- One angle is 36° larger than the other
  max a (max b c) = 78 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1990_199087


namespace NUMINAMATH_CALUDE_solution_set_exponential_inequality_l1990_199036

theorem solution_set_exponential_inequality :
  ∀ x : ℝ, (2 : ℝ) ^ (x^2 - 5*x + 5) > (1/2 : ℝ) ↔ x < 2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_exponential_inequality_l1990_199036


namespace NUMINAMATH_CALUDE_mia_has_110_dollars_l1990_199007

/-- The amount of money Darwin has -/
def darwins_money : ℕ := 45

/-- The amount of money Mia has -/
def mias_money : ℕ := 2 * darwins_money + 20

theorem mia_has_110_dollars : mias_money = 110 := by
  sorry

end NUMINAMATH_CALUDE_mia_has_110_dollars_l1990_199007


namespace NUMINAMATH_CALUDE_fraction_equality_l1990_199010

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 4) :
  (a - 2*a*b - b) / (2*a + 7*a*b - 2*b) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1990_199010


namespace NUMINAMATH_CALUDE_expression_equality_l1990_199091

theorem expression_equality : 
  (-3^2 ≠ -2^3) ∧ 
  ((-3)^2 ≠ (-2)^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1990_199091


namespace NUMINAMATH_CALUDE_correct_registration_sequence_l1990_199093

-- Define the registration steps
inductive RegistrationStep
  | collectTicket
  | register
  | takeTests
  | takePhoto

-- Define a type for sequences of registration steps
def RegistrationSequence := List RegistrationStep

-- Define the given sequence of steps
def givenSequence : RegistrationSequence := 
  [RegistrationStep.register, RegistrationStep.takePhoto, 
   RegistrationStep.collectTicket, RegistrationStep.takeTests]

-- Define a function to check if a sequence is correct
def isCorrectSequence (seq : RegistrationSequence) : Prop :=
  seq = givenSequence

-- Theorem stating that the given sequence is correct
theorem correct_registration_sequence :
  isCorrectSequence givenSequence := by
  sorry

end NUMINAMATH_CALUDE_correct_registration_sequence_l1990_199093


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l1990_199014

theorem ordered_pair_solution :
  ∀ x y : ℝ,
  (x + y = (6 - x) + (6 - y)) →
  (x - y = (x - 2) + (y - 2)) →
  (x = 2 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l1990_199014


namespace NUMINAMATH_CALUDE_complex_2_minus_3i_in_fourth_quadrant_l1990_199024

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_2_minus_3i_in_fourth_quadrant :
  is_in_fourth_quadrant (2 - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_2_minus_3i_in_fourth_quadrant_l1990_199024


namespace NUMINAMATH_CALUDE_all_cells_equal_l1990_199058

/-- Represents an infinite grid of natural numbers -/
def Grid := ℤ → ℤ → ℕ

/-- The condition that each cell's value is greater than or equal to the arithmetic mean of its four neighboring cells -/
def ValidGrid (g : Grid) : Prop :=
  ∀ i j : ℤ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

/-- The theorem stating that all cells in a valid grid must contain the same number -/
theorem all_cells_equal (g : Grid) (h : ValidGrid g) : 
  ∀ i j k l : ℤ, g i j = g k l :=
sorry

end NUMINAMATH_CALUDE_all_cells_equal_l1990_199058


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1990_199021

theorem digit_sum_problem :
  ∀ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    22 * (a + b + c) = 462 →
    ((a = 4 ∧ b = 8 ∧ c = 9) ∨
     (a = 5 ∧ b = 7 ∧ c = 9) ∨
     (a = 6 ∧ b = 7 ∧ c = 8) ∨
     (a = 8 ∧ b = 4 ∧ c = 9) ∨
     (a = 7 ∧ b = 5 ∧ c = 9) ∨
     (a = 7 ∧ b = 6 ∧ c = 8) ∨
     (a = 9 ∧ b = 4 ∧ c = 8) ∨
     (a = 9 ∧ b = 5 ∧ c = 7) ∨
     (a = 8 ∧ b = 6 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1990_199021


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1990_199026

/-- A geometric sequence with first term 2 and fifth term 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) / a n = a 2 / a 1) ∧ a 1 = 2 ∧ a 5 = 4

theorem geometric_sequence_ninth_term (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1990_199026


namespace NUMINAMATH_CALUDE_sixth_term_is_32_l1990_199085

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  (a 2 + a 3) / (a 1 + a 2) = 2 ∧ a 4 = 8

/-- Theorem stating that for a geometric sequence satisfying the given conditions, the 6th term is 32 -/
theorem sixth_term_is_32 (a : ℕ → ℝ) 
    (h_geo : is_geometric_sequence a) 
    (h_cond : sequence_conditions a) : 
  a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_32_l1990_199085


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l1990_199005

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (40 * π / 180)) / Real.cos (10 * π / 180) =
  (1/2 + Real.cos (20 * π / 180)^2) / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l1990_199005


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l1990_199099

theorem salary_savings_percentage (last_year_salary : ℝ) (last_year_savings_percentage : ℝ) : 
  last_year_savings_percentage > 0 →
  (0.15 * (1.1 * last_year_salary) = 1.65 * (last_year_savings_percentage / 100 * last_year_salary)) →
  last_year_savings_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l1990_199099


namespace NUMINAMATH_CALUDE_max_distance_C_D_l1990_199004

open Complex

/-- The set of solutions to z^4 - 16 = 0 -/
def C : Set ℂ := {z : ℂ | z^4 - 16 = 0}

/-- The set of solutions to z^4 - 16z^3 + 48z^2 - 64z + 64 = 0 -/
def D : Set ℂ := {z : ℂ | z^4 - 16*z^3 + 48*z^2 - 64*z + 64 = 0}

/-- The maximum distance between any point in C and any point in D is 2 -/
theorem max_distance_C_D : 
  ∃ (c : C) (d : D), ∀ (c' : C) (d' : D), abs (c - d) ≥ abs (c' - d') ∧ abs (c - d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_C_D_l1990_199004


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1990_199074

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (3 * X ^ 2 - 20 * X + 68 : Polynomial ℚ) = (X - 4) * q + 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1990_199074


namespace NUMINAMATH_CALUDE_envelope_count_l1990_199088

/-- The weight of one envelope in grams -/
def envelope_weight : ℝ := 8.5

/-- The total weight of all envelopes in kilograms -/
def total_weight : ℝ := 7.48

/-- The number of envelopes sent -/
def num_envelopes : ℕ := 880

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℝ := 1000

theorem envelope_count :
  (total_weight * kg_to_g) / envelope_weight = num_envelopes := by
  sorry

end NUMINAMATH_CALUDE_envelope_count_l1990_199088


namespace NUMINAMATH_CALUDE_remainder_19_power_1999_mod_25_l1990_199017

theorem remainder_19_power_1999_mod_25 : 19^1999 % 25 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_19_power_1999_mod_25_l1990_199017


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1990_199039

-- Define the propositions p and q as functions from real numbers to propositions
def p (x : ℝ) : Prop := |2*x - 3| < 1

def q (x : ℝ) : Prop := x * (x - 3) < 0

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1990_199039


namespace NUMINAMATH_CALUDE_original_number_proof_l1990_199076

theorem original_number_proof (x : ℚ) : (1 + 1 / x = 5 / 2) → x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1990_199076


namespace NUMINAMATH_CALUDE_sin_cos_difference_21_81_l1990_199065

theorem sin_cos_difference_21_81 :
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) -
  Real.cos (21 * π / 180) * Real.sin (81 * π / 180) =
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_21_81_l1990_199065


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1990_199013

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given a geometric sequence a with a₃ = 2 and a₅ = 6, a₉ = 54. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_3 : a 3 = 2) 
    (h_5 : a 5 = 6) : 
  a 9 = 54 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1990_199013


namespace NUMINAMATH_CALUDE_closest_fraction_to_one_l1990_199003

theorem closest_fraction_to_one : 
  let fractions : List ℚ := [7/8, 8/7, 9/10, 10/11, 11/10]
  ∀ f ∈ fractions, |10/11 - 1| ≤ |f - 1| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_to_one_l1990_199003


namespace NUMINAMATH_CALUDE_system_solution_l1990_199052

theorem system_solution (x y : ℝ) : 
  (x + y = 1 ∧ x - y = 3) → (x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1990_199052


namespace NUMINAMATH_CALUDE_tinsel_count_l1990_199044

/-- The number of pieces of tinsel in each box of Christmas decorations. -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box. -/
def trees_per_box : ℕ := 1

/-- The number of snow globes in each box. -/
def snow_globes_per_box : ℕ := 5

/-- The total number of boxes distributed. -/
def total_boxes : ℕ := 12

/-- The total number of decorations handed out. -/
def total_decorations : ℕ := 120

/-- Theorem stating that the number of pieces of tinsel in each box is 4. -/
theorem tinsel_count : 
  total_boxes * (tinsel_per_box + trees_per_box + snow_globes_per_box) = total_decorations :=
by sorry

end NUMINAMATH_CALUDE_tinsel_count_l1990_199044


namespace NUMINAMATH_CALUDE_lawn_maintenance_time_l1990_199096

theorem lawn_maintenance_time (mow_time fertilize_time total_time : ℕ) : 
  mow_time = 40 →
  fertilize_time = 2 * mow_time →
  total_time = mow_time + fertilize_time →
  total_time = 120 := by
sorry

end NUMINAMATH_CALUDE_lawn_maintenance_time_l1990_199096
