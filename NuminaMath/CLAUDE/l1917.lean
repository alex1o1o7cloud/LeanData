import Mathlib

namespace NUMINAMATH_CALUDE_ball_placement_count_l1917_191726

theorem ball_placement_count :
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 3
  let placement_options := n_boxes ^ n_balls
  placement_options = 81 :=
by sorry

end NUMINAMATH_CALUDE_ball_placement_count_l1917_191726


namespace NUMINAMATH_CALUDE_wall_building_time_l1917_191708

/-- Given that 18 persons can build a 140 m long wall in 42 days, 
    prove that 30 persons will take 18 days to complete a similar 100 m long wall. -/
theorem wall_building_time 
  (original_workers : ℕ) 
  (original_length : ℝ) 
  (original_days : ℕ) 
  (new_workers : ℕ) 
  (new_length : ℝ) 
  (h1 : original_workers = 18) 
  (h2 : original_length = 140) 
  (h3 : original_days = 42) 
  (h4 : new_workers = 30) 
  (h5 : new_length = 100) :
  (new_length / new_workers) / (original_length / original_workers) * original_days = 18 :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l1917_191708


namespace NUMINAMATH_CALUDE_x0_value_l1917_191760

open Real

noncomputable def f (x : ℝ) : ℝ := x * (2016 + log x)

theorem x0_value (x0 : ℝ) (h : deriv f x0 = 2017) : x0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l1917_191760


namespace NUMINAMATH_CALUDE_probability_exactly_two_ones_equals_fraction_l1917_191789

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_outcome : ℕ := 1
def target_count : ℕ := 2

def probability_exactly_two_ones : ℚ :=
  (num_dice.choose target_count : ℚ) * 
  (1 / num_sides) ^ target_count * 
  ((num_sides - 1) / num_sides) ^ (num_dice - target_count)

theorem probability_exactly_two_ones_equals_fraction :
  probability_exactly_two_ones = (66 * 5^10 : ℚ) / (36 * 6^10) := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_two_ones_equals_fraction_l1917_191789


namespace NUMINAMATH_CALUDE_cos_alpha_cos_2alpha_distinct_digits_l1917_191741

/-- Represents a repeating decimal of the form 0.aḃ -/
def repeating_decimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 90

theorem cos_alpha_cos_2alpha_distinct_digits :
  ∃! (a b c d : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    repeating_decimal a b = Real.cos α ∧
    repeating_decimal c d = -Real.cos (2 * α) ∧
    a = 1 ∧ b = 6 ∧ c = 9 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_cos_2alpha_distinct_digits_l1917_191741


namespace NUMINAMATH_CALUDE_planned_speed_calculation_l1917_191747

theorem planned_speed_calculation (distance : ℝ) (speed_multiplier : ℝ) (time_saved : ℝ) 
  (h1 : distance = 180)
  (h2 : speed_multiplier = 1.2)
  (h3 : time_saved = 0.5)
  : ∃ v : ℝ, v > 0 ∧ distance / v = distance / (speed_multiplier * v) + time_saved ∧ v = 60 := by
  sorry

end NUMINAMATH_CALUDE_planned_speed_calculation_l1917_191747


namespace NUMINAMATH_CALUDE_worker_a_time_proof_l1917_191780

/-- The time it takes for Worker A to complete the job alone -/
def worker_a_time : ℝ := 8.4

/-- The time it takes for Worker B to complete the job alone -/
def worker_b_time : ℝ := 6

/-- The time it takes for both workers to complete the job together -/
def combined_time : ℝ := 3.428571428571429

theorem worker_a_time_proof :
  (1 / worker_a_time) + (1 / worker_b_time) = (1 / combined_time) :=
sorry

end NUMINAMATH_CALUDE_worker_a_time_proof_l1917_191780


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1917_191797

/-- A line y = 3x + d is tangent to the parabola y^2 = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) : 
  (∃! x : ℝ, (3 * x + d)^2 = 12 * x) ↔ d = 1 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1917_191797


namespace NUMINAMATH_CALUDE_e_power_necessary_not_sufficient_for_ln_l1917_191745

theorem e_power_necessary_not_sufficient_for_ln (x : ℝ) :
  (∃ y, (Real.exp y > 1 ∧ Real.log y ≥ 0)) ∧
  (∀ z, Real.log z < 0 → Real.exp z > 1) :=
sorry

end NUMINAMATH_CALUDE_e_power_necessary_not_sufficient_for_ln_l1917_191745


namespace NUMINAMATH_CALUDE_cashback_less_profitable_l1917_191798

structure Bank where
  forecasted_cashback : ℝ
  actual_cashback : ℝ

structure Customer where
  is_savvy : Bool
  cashback_optimized : ℝ

def cashback_program (b : Bank) (customers : List Customer) : Prop :=
  b.actual_cashback > b.forecasted_cashback

def growing_savvy_customers (customers : List Customer) : Prop :=
  (customers.filter (λ c => c.is_savvy)).length > 
  (customers.filter (λ c => !c.is_savvy)).length

theorem cashback_less_profitable 
  (b : Bank) 
  (customers : List Customer) 
  (h1 : growing_savvy_customers customers) 
  (h2 : ∀ c ∈ customers, c.is_savvy → c.cashback_optimized > 0) :
  cashback_program b customers :=
by
  sorry

#check cashback_less_profitable

end NUMINAMATH_CALUDE_cashback_less_profitable_l1917_191798


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l1917_191770

theorem unique_solution_linear_system :
  ∃! (x y : ℝ), x + y = 5 ∧ x + 2*y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l1917_191770


namespace NUMINAMATH_CALUDE_square_difference_equation_l1917_191700

theorem square_difference_equation : 9^2 - 8^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equation_l1917_191700


namespace NUMINAMATH_CALUDE_polynomial_never_33_l1917_191749

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_never_33_l1917_191749


namespace NUMINAMATH_CALUDE_slower_train_speed_l1917_191750

theorem slower_train_speed
  (train_length : ℝ)
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : faster_train_speed = 45)
  (h3 : passing_time = 23.998080153587715) :
  ∃ (slower_train_speed : ℝ),
    slower_train_speed = 30 ∧
    slower_train_speed + faster_train_speed = (2 * train_length / 1000) / (passing_time / 3600) :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l1917_191750


namespace NUMINAMATH_CALUDE_quarter_circle_area_l1917_191718

/-- The area of a quarter circle with radius 2 is equal to π -/
theorem quarter_circle_area : 
  let r : Real := 2
  let circle_area : Real := π * r^2
  let quarter_circle_area : Real := circle_area / 4
  quarter_circle_area = π := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_area_l1917_191718


namespace NUMINAMATH_CALUDE_square_circle_radius_l1917_191703

theorem square_circle_radius (r : ℝ) (h : r > 0) :
  4 * r * Real.sqrt 2 = π * r^2 → r = 4 * Real.sqrt 2 / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_radius_l1917_191703


namespace NUMINAMATH_CALUDE_factorization_equality_l1917_191730

theorem factorization_equality (x₁ x₂ : ℝ) :
  x₁^3 - 2*x₁^2*x₂ - x₁ + 2*x₂ = (x₁ - 1) * (x₁ + 1) * (x₁ - 2*x₂) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1917_191730


namespace NUMINAMATH_CALUDE_sin_squared_alpha_minus_pi_fourth_l1917_191787

theorem sin_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2/3) :
  Real.sin (α - Real.pi/4)^2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_alpha_minus_pi_fourth_l1917_191787


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_l1917_191717

theorem min_sum_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 * (1 / Real.rpow 162 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_l1917_191717


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1917_191754

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < b ∧ b < a
  h_c : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The condition that a point on the short axis and the two foci form an equilateral triangle -/
def equilateral_triangle_condition (e : Ellipse) : Prop :=
  e.a = 2 * e.c

/-- The condition that the shortest distance from the foci to the endpoints of the major axis is √3 -/
def shortest_distance_condition (e : Ellipse) : Prop :=
  e.a - e.c = Real.sqrt 3

theorem ellipse_equation_from_conditions (e : Ellipse)
  (h_triangle : equilateral_triangle_condition e)
  (h_distance : shortest_distance_condition e) :
  ∀ x y : ℝ, ellipse_equation e x y ↔ x^2 / 12 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1917_191754


namespace NUMINAMATH_CALUDE_female_workers_l1917_191776

/-- Represents the number of workers in a company --/
structure Company where
  male : ℕ
  female : ℕ
  male_no_plan : ℕ
  female_no_plan : ℕ

/-- The conditions of the company --/
def company_conditions (c : Company) : Prop :=
  c.male = 112 ∧
  c.male_no_plan = (40 * c.male) / 100 ∧
  c.female_no_plan = (25 * c.female) / 100 ∧
  (30 * (c.male_no_plan + c.female_no_plan)) / 100 = c.male_no_plan ∧
  (60 * (c.male - c.male_no_plan + c.female - c.female_no_plan)) / 100 = (c.male - c.male_no_plan)

/-- The theorem to be proved --/
theorem female_workers (c : Company) : company_conditions c → c.female = 420 := by
  sorry

end NUMINAMATH_CALUDE_female_workers_l1917_191776


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l1917_191706

theorem sin_alpha_plus_pi_fourth (α : Real) :
  (Complex.mk (Real.sin α - 3/5) (Real.cos α - 4/5)).re = 0 →
  Real.sin (α + Real.pi/4) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l1917_191706


namespace NUMINAMATH_CALUDE_unique_solution_l1917_191769

/-- The polynomial P(x) = 3x^4 + ax^3 + bx^2 - 16x + 55 -/
def P (a b x : ℝ) : ℝ := 3 * x^4 + a * x^3 + b * x^2 - 16 * x + 55

/-- The first divisibility condition -/
def condition1 (a b : ℝ) : Prop :=
  P a b (-4/3) = 23

/-- The second divisibility condition -/
def condition2 (a b : ℝ) : Prop :=
  P a b 3 = 10

theorem unique_solution :
  ∃! (a b : ℝ), condition1 a b ∧ condition2 a b ∧ a = -29 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1917_191769


namespace NUMINAMATH_CALUDE_factors_of_1728_l1917_191731

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_1728 : number_of_factors 1728 = 28 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1728_l1917_191731


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l1917_191705

theorem consecutive_integers_median (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) :
  let median := S / n
  median = 729 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l1917_191705


namespace NUMINAMATH_CALUDE_smallest_covering_segment_l1917_191779

/-- Represents an equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_unit : side_length = 1

/-- Represents a sliding segment in the triangle -/
structure SlidingSegment where
  length : ℝ
  covers_triangle : Prop

/-- The smallest sliding segment that covers the triangle has length 2/3 -/
theorem smallest_covering_segment (triangle : EquilateralTriangle) :
  ∃ (d : ℝ), d = 2/3 ∧ 
  (∀ (s : SlidingSegment), s.covers_triangle → s.length ≥ d) ∧
  (∃ (s : SlidingSegment), s.covers_triangle ∧ s.length = d) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_segment_l1917_191779


namespace NUMINAMATH_CALUDE_square_root_of_one_fourth_l1917_191736

theorem square_root_of_one_fourth :
  {x : ℚ | x^2 = (1 : ℚ) / 4} = {(1 : ℚ) / 2, -(1 : ℚ) / 2} := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_one_fourth_l1917_191736


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l1917_191735

theorem chocolate_box_problem (B : ℕ) : 
  (B : ℚ) - ((1/4 : ℚ) * B - 5) - ((1/4 : ℚ) * B - 10) = 110 → B = 190 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l1917_191735


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l1917_191759

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 4) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l1917_191759


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_equal_l1917_191795

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a transversal
def transversal (t l1 l2 : Line) : Prop := sorry

-- Define alternate interior angles
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 t : Line) : Prop := sorry

-- Define corresponding angles
def correspondingAngles (a1 a2 : Angle) (l1 l2 t : Line) : Prop := sorry

-- Theorem statement
theorem parallel_lines_corresponding_angles_not_always_equal :
  ∃ (l1 l2 t : Line) (a1 a2 : Angle),
    parallel l1 l2 ∧
    transversal t l1 l2 ∧
    correspondingAngles a1 a2 l1 l2 t ∧
    a1 ≠ a2 :=
  sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_equal_l1917_191795


namespace NUMINAMATH_CALUDE_second_division_percentage_l1917_191788

theorem second_division_percentage 
  (total_students : ℕ) 
  (first_division_percentage : ℚ) 
  (just_passed : ℕ) 
  (h1 : total_students = 300)
  (h2 : first_division_percentage = 28/100)
  (h3 : just_passed = 54)
  : (↑(total_students - (total_students * first_division_percentage).floor - just_passed) / total_students : ℚ) = 54/100 := by
  sorry

end NUMINAMATH_CALUDE_second_division_percentage_l1917_191788


namespace NUMINAMATH_CALUDE_red_paint_percentage_l1917_191796

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : ℝ
  red : ℝ
  white : ℝ
  total : ℝ
  blue_percentage : ℝ
  red_percentage : ℝ
  white_percentage : ℝ

/-- Given a paint mixture with 70% blue paint, 140 ounces of blue paint, 
    and 20 ounces of white paint, prove that 20% of the mixture is red paint -/
theorem red_paint_percentage 
  (mixture : PaintMixture) 
  (h1 : mixture.blue_percentage = 0.7) 
  (h2 : mixture.blue = 140) 
  (h3 : mixture.white = 20) 
  (h4 : mixture.total = mixture.blue + mixture.red + mixture.white) 
  (h5 : mixture.blue_percentage + mixture.red_percentage + mixture.white_percentage = 1) :
  mixture.red_percentage = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_red_paint_percentage_l1917_191796


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1917_191732

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : 3 * (a 3)^2 - 11 * (a 3) + 9 = 0 ∧ 3 * (a 9)^2 - 11 * (a 9) + 9 = 0) :
  (a 6)^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1917_191732


namespace NUMINAMATH_CALUDE_ratio_average_l1917_191790

theorem ratio_average (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  c = 24 →
  (a + b + c) / 3 = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_average_l1917_191790


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l1917_191767

/-- The perimeter of a rectangular sheet folded along its diagonal -/
theorem folded_rectangle_perimeter (length width : ℝ) (h1 : length = 20) (h2 : width = 12) :
  2 * (length + width) = 64 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l1917_191767


namespace NUMINAMATH_CALUDE_tip_percentage_is_22_percent_l1917_191746

/-- Calculates the tip percentage given the total amount spent, food price, and sales tax rate. -/
def calculate_tip_percentage (total_spent : ℚ) (food_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let sales_tax := food_price * sales_tax_rate
  let tip := total_spent - (food_price + sales_tax)
  (tip / food_price) * 100

/-- Theorem stating that under the given conditions, the tip percentage is 22%. -/
theorem tip_percentage_is_22_percent :
  let total_spent : ℚ := 132
  let food_price : ℚ := 100
  let sales_tax_rate : ℚ := 10 / 100
  calculate_tip_percentage total_spent food_price sales_tax_rate = 22 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_22_percent_l1917_191746


namespace NUMINAMATH_CALUDE_johns_average_speed_l1917_191775

/-- Proves that John's average speed was 40 miles per hour given the conditions of the problem -/
theorem johns_average_speed 
  (john_time : ℝ) 
  (beth_route_difference : ℝ)
  (beth_time_difference : ℝ)
  (beth_speed : ℝ)
  (h1 : john_time = 30 / 60) -- John's time in hours
  (h2 : beth_route_difference = 5) -- Beth's route was 5 miles longer
  (h3 : beth_time_difference = 20 / 60) -- Beth's additional time in hours
  (h4 : beth_speed = 30) -- Beth's average speed in miles per hour
  : (beth_speed * (john_time + beth_time_difference) - beth_route_difference) / john_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_johns_average_speed_l1917_191775


namespace NUMINAMATH_CALUDE_f_of_4_6_l1917_191738

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem f_of_4_6 : f (4, 6) = (10, -2) := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_6_l1917_191738


namespace NUMINAMATH_CALUDE_point_on_line_l1917_191739

/-- A point on a line in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines a line y = mx + c -/
structure Line2D where
  m : ℝ
  c : ℝ

/-- Theorem: If a point P(a,b) lies on the line y = -3x - 4, then b + 3a + 4 = 0 -/
theorem point_on_line (P : Point2D) (L : Line2D) 
  (h1 : L.m = -3)
  (h2 : L.c = -4)
  (h3 : P.y = L.m * P.x + L.c) :
  P.y + 3 * P.x + 4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l1917_191739


namespace NUMINAMATH_CALUDE_gas_price_calculation_l1917_191728

/-- Proves that the actual cost of gas per gallon is $1.80 given the problem conditions -/
theorem gas_price_calculation (expected_price : ℝ) : 
  (12 * expected_price = 10 * (expected_price + 0.3)) → 
  (expected_price + 0.3 = 1.8) := by
  sorry

#check gas_price_calculation

end NUMINAMATH_CALUDE_gas_price_calculation_l1917_191728


namespace NUMINAMATH_CALUDE_positive_integer_solutions_l1917_191755

theorem positive_integer_solutions :
  ∀ (a b c x y z : ℕ+),
    (a + b + c = x * y * z ∧ x + y + z = a * b * c) ↔
    ((x = 3 ∧ y = 2 ∧ z = 1 ∧ a = 3 ∧ b = 2 ∧ c = 1) ∨
     (x = 3 ∧ y = 3 ∧ z = 1 ∧ a = 5 ∧ b = 2 ∧ c = 1) ∨
     (x = 5 ∧ y = 2 ∧ z = 1 ∧ a = 3 ∧ b = 3 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l1917_191755


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l1917_191709

theorem opposite_of_one_half : 
  (-(1/2) : ℚ) = (-1/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l1917_191709


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_l1917_191793

/-- Given a parabola x² = 16y, prove that a circle centered at its focus
    and tangent to its directrix has the equation x² + (y - 4)² = 64 -/
theorem circle_equation_from_parabola (x y : ℝ) :
  (x^2 = 16*y) →  -- Parabola equation
  ∃ (h k r : ℝ),
    (h = 0 ∧ k = 4) →  -- Focus (circle center) at (0, 4)
    ((x - h)^2 + (y - k)^2 = r^2) →  -- General circle equation
    (y = -4 → (x - h)^2 + (y - k)^2 = r^2)  -- Circle tangent to directrix y = -4
    →
    ((x - 0)^2 + (y - 4)^2 = 64)  -- Specific circle equation
    :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_l1917_191793


namespace NUMINAMATH_CALUDE_polynomial_intersection_l1917_191744

-- Define the polynomials f and h
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def h (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Define the theorem
theorem polynomial_intersection (a b p q : ℝ) : 
  -- f and h are distinct polynomials
  (∃ x, f a b x ≠ h p q x) →
  -- The x-coordinate of the vertex of f is a root of h
  h p q (-a/2) = 0 →
  -- The x-coordinate of the vertex of h is a root of f
  f a b (-p/2) = 0 →
  -- Both f and h have the same minimum value
  (∃ y, f a b (-a/2) = y ∧ h p q (-p/2) = y) →
  -- The graphs of f and h intersect at the point (50, -50)
  f a b 50 = -50 ∧ h p q 50 = -50 →
  -- Conclusion: a + p = 0
  a + p = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l1917_191744


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l1917_191714

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 7

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := morning_emails - email_difference

theorem jack_afternoon_emails : afternoon_emails = 3 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l1917_191714


namespace NUMINAMATH_CALUDE_circle_radius_range_l1917_191733

/-- Given two circles O₁ and O₂, with O₁ having radius 1 and O₂ having radius r,
    and the distance between their centers being 5,
    if there exists a point P on O₂ such that PO₁ = 2,
    then the radius r of O₂ is between 3 and 7 inclusive. -/
theorem circle_radius_range (r : ℝ) :
  let O₁ : ℝ × ℝ := (0, 0)  -- Assuming O₁ is at the origin for simplicity
  let O₂ : ℝ × ℝ := (5, 0)  -- Assuming O₂ is on the x-axis
  ∃ (P : ℝ × ℝ), 
    (P.1 - O₂.1)^2 + P.2^2 = r^2 ∧  -- P is on circle O₂
    (P.1 - O₁.1)^2 + P.2^2 = 4      -- PO₁ = 2
  → 3 ≤ r ∧ r ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_range_l1917_191733


namespace NUMINAMATH_CALUDE_chord_ratio_implies_slope_l1917_191701

theorem chord_ratio_implies_slope (k : ℝ) (h1 : k > 0) : 
  let l := {(x, y) : ℝ × ℝ | y = k * x}
  let C1 := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  let C2 := {(x, y) : ℝ × ℝ | (x - 3)^2 + y^2 = 1}
  let chord1 := {p : ℝ × ℝ | p ∈ l ∩ C1}
  let chord2 := {p : ℝ × ℝ | p ∈ l ∩ C2}
  (∃ (p q : ℝ × ℝ), p ∈ chord1 ∧ q ∈ chord1 ∧ p ≠ q) →
  (∃ (r s : ℝ × ℝ), r ∈ chord2 ∧ s ∈ chord2 ∧ r ≠ s) →
  (∃ (p q r s : ℝ × ℝ), p ∈ chord1 ∧ q ∈ chord1 ∧ r ∈ chord2 ∧ s ∈ chord2 ∧
    dist p q / dist r s = 3) →
  k = 1/3 :=
by sorry


end NUMINAMATH_CALUDE_chord_ratio_implies_slope_l1917_191701


namespace NUMINAMATH_CALUDE_complex_number_problem_l1917_191720

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
def condition1 : Prop := (z + 3 + 4 * Complex.I).im = 0
def condition2 : Prop := (z / (1 - 2 * Complex.I)).im = 0
def condition3 (m : ℝ) : Prop := 
  let w := (z - m * Complex.I)^2
  w.re < 0 ∧ w.im > 0

-- State the theorem
theorem complex_number_problem (h1 : condition1 z) (h2 : condition2 z) :
  z = 2 - 4 * Complex.I ∧ 
  ∃ m₀ : ℝ, ∀ m : ℝ, condition3 z m ↔ m < m₀ ∧ m₀ = -6 :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1917_191720


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l1917_191753

def candy_problem (morning_boxes afternoon_multiplier given_away : ℕ) : ℕ :=
  morning_boxes + afternoon_multiplier * morning_boxes - given_away

theorem oliver_candy_boxes :
  candy_problem 8 3 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l1917_191753


namespace NUMINAMATH_CALUDE_conference_duration_theorem_l1917_191784

/-- The duration of the conference in minutes -/
def conference_duration (first_session_hours : ℕ) (first_session_minutes : ℕ) 
  (second_session_hours : ℕ) (second_session_minutes : ℕ) : ℕ :=
  (first_session_hours * 60 + first_session_minutes) + 
  (second_session_hours * 60 + second_session_minutes)

/-- Theorem stating the total duration of the conference -/
theorem conference_duration_theorem : 
  conference_duration 8 15 3 40 = 715 := by sorry

end NUMINAMATH_CALUDE_conference_duration_theorem_l1917_191784


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1917_191729

/-- The ellipse defined by x²/4 + y²/m = 1 -/
def is_ellipse (x y m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

/-- The line y = mx + 2 -/
def is_line (x y m : ℝ) : Prop :=
  y = m * x + 2

/-- The line is tangent to the ellipse -/
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), is_ellipse x y m ∧ is_line x y m ∧
  ∀ (x' y' : ℝ), is_ellipse x' y' m → is_line x' y' m → (x = x' ∧ y = y')

theorem tangent_line_to_ellipse (m : ℝ) :
  is_tangent m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l1917_191729


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l1917_191773

/-- Two congruent right circular cones with a sphere inside -/
structure ConeFigure where
  /-- Base radius of each cone -/
  base_radius : ℝ
  /-- Height of each cone -/
  cone_height : ℝ
  /-- Distance from base to intersection point of axes -/
  intersection_distance : ℝ
  /-- Radius of the sphere -/
  sphere_radius : ℝ
  /-- Condition: base radius is 5 -/
  base_radius_eq : base_radius = 5
  /-- Condition: cone height is 10 -/
  cone_height_eq : cone_height = 10
  /-- Condition: intersection distance is 5 -/
  intersection_eq : intersection_distance = 5
  /-- Condition: sphere lies within both cones -/
  sphere_within_cones : sphere_radius ≤ intersection_distance

/-- The maximum possible value of r^2 is 80 -/
theorem max_sphere_radius_squared (cf : ConeFigure) : 
  ∃ (max_r : ℝ), ∀ (r : ℝ), cf.sphere_radius = r → r^2 ≤ max_r^2 ∧ max_r^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l1917_191773


namespace NUMINAMATH_CALUDE_second_car_speed_l1917_191768

/-- 
Given two cars starting from opposite ends of a 105-mile highway, 
with one car traveling at 15 mph and both meeting after 3 hours, 
prove that the speed of the second car is 20 mph.
-/
theorem second_car_speed 
  (highway_length : ℝ) 
  (first_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 105) 
  (h2 : first_car_speed = 15) 
  (h3 : meeting_time = 3) : 
  ∃ (second_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    second_car_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l1917_191768


namespace NUMINAMATH_CALUDE_farmer_pumpkin_seeds_per_row_l1917_191783

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  radishes : ℕ
  radish_per_row : ℕ
  rows_per_bed : ℕ
  plant_beds : ℕ

/-- Calculates the number of pumpkin seeds per row -/
def pumpkin_seeds_per_row (fp : FarmerPlanting) : ℕ :=
  fp.pumpkin_seeds / (fp.plant_beds * fp.rows_per_bed - fp.bean_seedlings / fp.bean_per_row - fp.radishes / fp.radish_per_row)

/-- Theorem stating that given the specific planting scenario, the farmer plants 7 pumpkin seeds per row -/
theorem farmer_pumpkin_seeds_per_row :
  let fp : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    pumpkin_seeds := 84,
    radishes := 48,
    radish_per_row := 6,
    rows_per_bed := 2,
    plant_beds := 14
  }
  pumpkin_seeds_per_row fp = 7 := by
  sorry

end NUMINAMATH_CALUDE_farmer_pumpkin_seeds_per_row_l1917_191783


namespace NUMINAMATH_CALUDE_problem_solution_l1917_191719

theorem problem_solution :
  (∀ x : ℝ, x^2 - 5*x + 4 < 0 ∧ (x - 2)*(x - 5) < 0 ↔ 2 < x ∧ x < 4) ∧
  (∀ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, (x - 2)*(x - 5) < 0 → x^2 - 5*a*x + 4*a^2 < 0) ∧ 
    (∃ x : ℝ, x^2 - 5*a*x + 4*a^2 < 0 ∧ (x - 2)*(x - 5) ≥ 0) ↔
    5/4 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1917_191719


namespace NUMINAMATH_CALUDE_art_gallery_display_ratio_l1917_191724

theorem art_gallery_display_ratio :
  let total_pieces : ℕ := 2700
  let sculptures_not_displayed : ℕ := 1200
  let paintings_not_displayed : ℕ := sculptures_not_displayed / 3
  let pieces_not_displayed : ℕ := sculptures_not_displayed + paintings_not_displayed
  let pieces_displayed : ℕ := total_pieces - pieces_not_displayed
  let sculptures_displayed : ℕ := pieces_displayed / 6
  pieces_displayed / total_pieces = 11 / 27 :=
by
  sorry

end NUMINAMATH_CALUDE_art_gallery_display_ratio_l1917_191724


namespace NUMINAMATH_CALUDE_teal_color_perception_l1917_191781

theorem teal_color_perception (total : ℕ) (kinda_blue : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  kinda_blue = 90 →
  both = 35 →
  neither = 25 →
  ∃ kinda_green : ℕ, kinda_green = 70 ∧ 
    kinda_green + kinda_blue - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_perception_l1917_191781


namespace NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_lower_bound_l1917_191762

/-- Represents a tiling of a rectangle -/
structure Tiling (m n : ℕ) :=
  (pieces : ℕ)
  (is_valid : Bool)

/-- The number of ways to tile a 5 × 2k rectangle with 2k pieces -/
def tiling_count (k : ℕ) : ℕ := sorry

theorem rectangle_tiling (n : ℕ) (t : Tiling 5 n) :
  t.pieces = n ∧ t.is_valid → Even n := by sorry

theorem tiling_count_lower_bound (k : ℕ) :
  k ≥ 3 → tiling_count k > 2 * 3^(k-1) := by sorry

end NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_lower_bound_l1917_191762


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1917_191766

theorem cubic_root_sum (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
  (h3 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1917_191766


namespace NUMINAMATH_CALUDE_maximize_product_l1917_191756

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 40) :
  x^3 * y^4 ≤ (160/7)^3 * (120/7)^4 ∧
  (x^3 * y^4 = (160/7)^3 * (120/7)^4 ↔ x = 160/7 ∧ y = 120/7) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l1917_191756


namespace NUMINAMATH_CALUDE_carol_cupcakes_l1917_191721

/-- Calculates the total number of cupcakes Carol has after selling some and making more. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (new_made : ℕ) : ℕ :=
  initial - sold + new_made

/-- Proves that Carol has 49 cupcakes in total given the initial conditions. -/
theorem carol_cupcakes : total_cupcakes 30 9 28 = 49 := by
  sorry

end NUMINAMATH_CALUDE_carol_cupcakes_l1917_191721


namespace NUMINAMATH_CALUDE_brick_length_proof_l1917_191752

/-- Given a courtyard and bricks with specific dimensions, prove the length of each brick -/
theorem brick_length_proof (courtyard_length : ℝ) (courtyard_breadth : ℝ) 
  (brick_breadth : ℝ) (total_bricks : ℕ) :
  courtyard_length = 20 →
  courtyard_breadth = 16 →
  brick_breadth = 0.1 →
  total_bricks = 16000 →
  (courtyard_length * courtyard_breadth * 10000) / (total_bricks * brick_breadth * 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_proof_l1917_191752


namespace NUMINAMATH_CALUDE_sum_of_specific_common_multiples_l1917_191710

theorem sum_of_specific_common_multiples (a b : ℕ) (h : Nat.lcm a b = 21) :
  (9 * 21) + (10 * 21) + (11 * 21) = 630 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_common_multiples_l1917_191710


namespace NUMINAMATH_CALUDE_bowling_money_theorem_l1917_191742

/-- The cost of renting bowling shoes for a day -/
def shoe_rental_cost : ℚ := 0.50

/-- The cost of bowling one game -/
def game_cost : ℚ := 1.75

/-- The maximum number of complete games the person can bowl -/
def max_games : ℕ := 7

/-- The total amount of money the person has -/
def total_money : ℚ := shoe_rental_cost + max_games * game_cost

theorem bowling_money_theorem :
  total_money = 12.75 := by sorry

end NUMINAMATH_CALUDE_bowling_money_theorem_l1917_191742


namespace NUMINAMATH_CALUDE_gcd_7163_209_l1917_191761

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcd_7163_209_l1917_191761


namespace NUMINAMATH_CALUDE_square_side_length_range_l1917_191737

theorem square_side_length_range (area : ℝ) (h : area = 15) :
  ∃ x : ℝ, x > 3 ∧ x < 4 ∧ x^2 = area := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l1917_191737


namespace NUMINAMATH_CALUDE_compare_fractions_l1917_191751

theorem compare_fractions : -8 / 21 > -3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l1917_191751


namespace NUMINAMATH_CALUDE_wheat_purchase_proof_l1917_191723

/-- The cost of wheat in cents per pound -/
def wheat_cost : ℚ := 72

/-- The cost of oats in cents per pound -/
def oats_cost : ℚ := 36

/-- The total amount of wheat and oats bought in pounds -/
def total_amount : ℚ := 30

/-- The total amount spent in cents -/
def total_spent : ℚ := 1620

/-- The amount of wheat bought in pounds -/
def wheat_amount : ℚ := 15

theorem wheat_purchase_proof :
  ∃ (oats_amount : ℚ),
    wheat_amount + oats_amount = total_amount ∧
    wheat_cost * wheat_amount + oats_cost * oats_amount = total_spent :=
by sorry

end NUMINAMATH_CALUDE_wheat_purchase_proof_l1917_191723


namespace NUMINAMATH_CALUDE_different_meal_combinations_l1917_191748

theorem different_meal_combinations (n : Nat) (h : n = 12) :
  (n * (n - 1) : Nat) = 132 := by
  sorry

end NUMINAMATH_CALUDE_different_meal_combinations_l1917_191748


namespace NUMINAMATH_CALUDE_motorcycle_theorem_l1917_191799

def motorcycle_problem (k r t : ℝ) (h1 : k > 0) (h2 : r > 0) (h3 : t > 0) : Prop :=
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ v2 > 0 ∧
    r * (v1 - v2) = k ∧
    t * (v1 + v2) = k ∧
    v1 / v2 = |r + t| / |r - t|

theorem motorcycle_theorem (k r t : ℝ) (h1 : k > 0) (h2 : r > 0) (h3 : t > 0) :
  motorcycle_problem k r t h1 h2 h3 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_theorem_l1917_191799


namespace NUMINAMATH_CALUDE_john_miles_conversion_l1917_191791

/-- Converts a base-7 number represented as a list of digits to its base-10 equivalent -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number of miles John cycled -/
def johnMilesBase7 : List Nat := [6, 1, 5, 3]

theorem john_miles_conversion :
  base7ToBase10 johnMilesBase7 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_john_miles_conversion_l1917_191791


namespace NUMINAMATH_CALUDE_problem_statement_l1917_191740

theorem problem_statement (x y : ℝ) (h : x + y = 1) :
  (x^2 + 3*y^2 ≥ 3/4) ∧
  (x*y > 0 → ∀ a : ℝ, a ≤ 5/2 → 1/x + 1/y ≥ |a - 2| + |a + 1|) ∧
  (∀ a : ℝ, (x*y > 0 → 1/x + 1/y ≥ |a - 2| + |a + 1|) ↔ a ≤ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1917_191740


namespace NUMINAMATH_CALUDE_rocket_altitude_time_rocket_minimum_time_l1917_191792

/-- Represents the distance covered by the rocket in a given second -/
def distance_at_second (n : ℕ) : ℝ := 2 + 2 * (n - 1)

/-- Represents the total distance covered by the rocket after n seconds -/
def total_distance (n : ℕ) : ℝ := (n : ℝ) * 2 + (n : ℝ) * ((n : ℝ) - 1)

/-- The theorem stating that the rocket reaches 240 km altitude in 15 seconds -/
theorem rocket_altitude_time : total_distance 15 = 240 := by
  sorry

/-- The theorem stating that 15 is the minimum time to reach 240 km -/
theorem rocket_minimum_time (n : ℕ) : 
  n < 15 → total_distance n < 240 := by
  sorry

end NUMINAMATH_CALUDE_rocket_altitude_time_rocket_minimum_time_l1917_191792


namespace NUMINAMATH_CALUDE_math_team_selection_l1917_191716

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem math_team_selection :
  let girls := 5
  let boys := 5
  let girls_on_team := 3
  let boys_on_team := 2
  (choose girls girls_on_team) * (choose boys boys_on_team) = 100 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l1917_191716


namespace NUMINAMATH_CALUDE_c_share_correct_l1917_191722

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share (total_profit : ℕ) (investments : List ℕ) (partner_index : ℕ) : ℕ :=
  sorry

theorem c_share_correct (total_profit : ℕ) (investments : List ℕ) :
  total_profit = 90000 →
  investments = [30000, 45000, 50000] →
  calculate_share total_profit investments 2 = 36000 :=
by sorry

end NUMINAMATH_CALUDE_c_share_correct_l1917_191722


namespace NUMINAMATH_CALUDE_selection_theorem_l1917_191757

/-- The number of ways to select one person from a department with n employees -/
def selectOne (n : ℕ) : ℕ := n

/-- The total number of ways to select one person from three departments -/
def totalWays (deptA deptB deptC : ℕ) : ℕ :=
  selectOne deptA + selectOne deptB + selectOne deptC

theorem selection_theorem :
  totalWays 2 4 3 = 9 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l1917_191757


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1917_191764

/-- Represents the number of households selected in a stratum -/
structure StratumSample where
  total : ℕ
  selected : ℕ

/-- Represents the stratified sample of households -/
structure StratifiedSample where
  high_income : StratumSample
  middle_income : StratumSample
  low_income : StratumSample

def total_households (s : StratifiedSample) : ℕ :=
  s.high_income.total + s.middle_income.total + s.low_income.total

def total_selected (s : StratifiedSample) : ℕ :=
  s.high_income.selected + s.middle_income.selected + s.low_income.selected

theorem stratified_sample_theorem (s : StratifiedSample) :
  s.high_income.total = 120 →
  s.middle_income.total = 200 →
  s.low_income.total = 160 →
  s.high_income.selected = 6 →
  total_households s = 480 →
  total_selected s = 24 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l1917_191764


namespace NUMINAMATH_CALUDE_three_integer_pairs_satisfy_equation_l1917_191707

theorem three_integer_pairs_satisfy_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^3 + y^2 = 2*y + 1) ∧ 
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_pairs_satisfy_equation_l1917_191707


namespace NUMINAMATH_CALUDE_angelinas_speed_l1917_191771

/-- Proves that Angelina's speed from grocery to gym is 24 meters per second -/
theorem angelinas_speed (v : ℝ) : 
  v > 0 →  -- Assume positive speed
  720 / v - 480 / (2 * v) = 40 →  -- Time difference condition
  2 * v = 24 := by
sorry

end NUMINAMATH_CALUDE_angelinas_speed_l1917_191771


namespace NUMINAMATH_CALUDE_sector_min_perimeter_l1917_191704

theorem sector_min_perimeter (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (1/2 * l * r = 4) → (l + 2*r ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_sector_min_perimeter_l1917_191704


namespace NUMINAMATH_CALUDE_perfect_cube_condition_l1917_191725

/-- A polynomial x^3 + px^2 + qx + n is a perfect cube if and only if q = p^2 / 3 and n = p^3 / 27 -/
theorem perfect_cube_condition (p q n : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, x^3 + p*x^2 + q*x + n = (x + a)^3) ↔ 
  (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_condition_l1917_191725


namespace NUMINAMATH_CALUDE_photo_arrangements_3_3_1_l1917_191777

/-- The number of possible arrangements for a group photo -/
def photo_arrangements (num_boys num_girls : ℕ) : ℕ :=
  let adjacent_choices := num_boys * num_girls
  let remaining_boys := num_boys - 1
  let remaining_girls := num_girls - 1
  let remaining_arrangements := (remaining_boys * (remaining_boys - 1)) * 
                                (remaining_girls * (remaining_girls - 1) * 
                                 (remaining_boys + remaining_girls) * 
                                 (remaining_boys + remaining_girls - 1))
  2 * adjacent_choices * remaining_arrangements

/-- Theorem stating the number of arrangements for 3 boys, 3 girls, and 1 teacher -/
theorem photo_arrangements_3_3_1 :
  photo_arrangements 3 3 = 432 := by
  sorry

#eval photo_arrangements 3 3

end NUMINAMATH_CALUDE_photo_arrangements_3_3_1_l1917_191777


namespace NUMINAMATH_CALUDE_cube_sum_prime_power_l1917_191765

theorem cube_sum_prime_power (a b p n : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < p ∧ 0 < n ∧ Nat.Prime p ∧ a^3 + b^3 = p^n →
  (∃ k : ℕ, 0 < k ∧
    ((a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k - 2) ∨
     (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k - 1) ∨
     (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k - 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sum_prime_power_l1917_191765


namespace NUMINAMATH_CALUDE_vector_parallel_solution_l1917_191743

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_solution :
  ∃ (x : ℝ), parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_vector_parallel_solution_l1917_191743


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l1917_191734

/-- Proves that the number of years until a man's age is twice his son's age is 2 -/
theorem mans_age_twice_sons_age (man_age son_age years : ℕ) : 
  man_age = son_age + 28 →
  son_age = 26 →
  (man_age + years) = 2 * (son_age + years) →
  years = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l1917_191734


namespace NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l1917_191712

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define what it means for two lines to be parallel
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Define a perpendicular segment between two lines
def PerpendicularSegment (l₁ l₂ : Line) : Type := sorry

-- Define the length of a perpendicular segment
def Length (seg : PerpendicularSegment l₁ l₂) : ℝ := sorry

-- Theorem statement
theorem perpendicular_segments_equal_length 
  (l₁ l₂ : Line) (h : Parallel l₁ l₂) :
  ∀ (seg₁ seg₂ : PerpendicularSegment l₁ l₂), 
  Length seg₁ = Length seg₂ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l1917_191712


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1917_191782

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 3 * x^2 + 4 = 5 * x - 7 ↔ x = a + b * I ∨ x = a - b * I) →
  a + b^2 = 137 / 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1917_191782


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1917_191772

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + (m+1)*x + 16 = (x + a)^2) → m = 7 ∨ m = -9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1917_191772


namespace NUMINAMATH_CALUDE_first_platform_length_is_150_l1917_191785

/-- The length of a train in meters -/
def train_length : ℝ := 150

/-- The length of the second platform in meters -/
def second_platform_length : ℝ := 250

/-- The time taken to cross the first platform in seconds -/
def time_first_platform : ℝ := 15

/-- The time taken to cross the second platform in seconds -/
def time_second_platform : ℝ := 20

/-- The length of the first platform in meters -/
def first_platform_length : ℝ := 150

theorem first_platform_length_is_150 :
  (train_length + first_platform_length) / time_first_platform =
  (train_length + second_platform_length) / time_second_platform :=
sorry

end NUMINAMATH_CALUDE_first_platform_length_is_150_l1917_191785


namespace NUMINAMATH_CALUDE_base_prime_1260_l1917_191702

/-- Base prime representation of a number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Prime factorization of 1260 -/
def PrimeFactorization1260 : List (ℕ × ℕ) :=
  [(2, 2), (3, 2), (5, 1), (7, 1)]

/-- Theorem: The base prime representation of 1260 is [2, 2, 1, 2] -/
theorem base_prime_1260 : 
  BasePrimeRepresentation 1260 = [2, 2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_1260_l1917_191702


namespace NUMINAMATH_CALUDE_max_m_value_l1917_191713

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, a > 0 → b > 0 → m / (3 * a + b) - 3 / a - 1 / b ≤ 0) : 
  m ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1917_191713


namespace NUMINAMATH_CALUDE_prob_sum_5_is_one_ninth_l1917_191711

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := sides * sides

/-- The number of favorable outcomes (sum of 5) -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def prob_sum_5 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_5_is_one_ninth :
  prob_sum_5 = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_sum_5_is_one_ninth_l1917_191711


namespace NUMINAMATH_CALUDE_granger_spam_cans_l1917_191715

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  spam_price : ℕ
  peanut_butter_price : ℕ
  bread_price : ℕ

/-- Represents the quantities of items bought --/
structure Quantities where
  spam_cans : ℕ
  peanut_butter_jars : ℕ
  bread_loaves : ℕ

/-- Calculates the total cost of the groceries --/
def total_cost (items : GroceryItems) (quantities : Quantities) : ℕ :=
  items.spam_price * quantities.spam_cans +
  items.peanut_butter_price * quantities.peanut_butter_jars +
  items.bread_price * quantities.bread_loaves

/-- Theorem stating that Granger bought 4 cans of Spam --/
theorem granger_spam_cans :
  ∀ (items : GroceryItems) (quantities : Quantities),
    items.spam_price = 3 →
    items.peanut_butter_price = 5 →
    items.bread_price = 2 →
    quantities.peanut_butter_jars = 3 →
    quantities.bread_loaves = 4 →
    total_cost items quantities = 59 →
    quantities.spam_cans = 4 :=
by sorry

end NUMINAMATH_CALUDE_granger_spam_cans_l1917_191715


namespace NUMINAMATH_CALUDE_lavender_candles_count_l1917_191794

theorem lavender_candles_count (almond coconut lavender : ℕ) : 
  almond = 10 →
  coconut = (3 * almond) / 2 →
  lavender = 2 * coconut →
  lavender = 30 := by
sorry

end NUMINAMATH_CALUDE_lavender_candles_count_l1917_191794


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_eq_one_g_monotone_increasing_sum_of_zeros_lt_two_l1917_191786

noncomputable section

variables (x : ℝ) (p : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

def g (x : ℝ) (p : ℝ) : ℝ := Real.log x - 2*(x-p)/(x+p) - Real.log p

theorem unique_solution_implies_a_eq_one :
  (∃! x, f 1 x = 0) → 1 = 1 := by sorry

theorem g_monotone_increasing (hp : p > 0) :
  Monotone (g · p) := by sorry

theorem sum_of_zeros_lt_two :
  ∃ x₁ x₂, x₁ < x₂ ∧ f 1 x₁ = 0 ∧ f 1 x₂ = 0 → x₁ + x₂ < 2 := by sorry

end

end NUMINAMATH_CALUDE_unique_solution_implies_a_eq_one_g_monotone_increasing_sum_of_zeros_lt_two_l1917_191786


namespace NUMINAMATH_CALUDE_spurs_team_size_l1917_191774

/-- The number of basketballs each player has -/
def basketballs_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := 242

/-- The number of players on the team -/
def number_of_players : ℕ := total_basketballs / basketballs_per_player

theorem spurs_team_size : number_of_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_spurs_team_size_l1917_191774


namespace NUMINAMATH_CALUDE_expired_bottle_probability_l1917_191758

theorem expired_bottle_probability (total_bottles : ℕ) (expired_bottles : ℕ) 
  (prob_both_unexpired : ℚ) :
  total_bottles = 30 →
  expired_bottles = 3 →
  prob_both_unexpired = 351 / 435 →
  (1 - prob_both_unexpired : ℚ) = 28 / 145 :=
by sorry

end NUMINAMATH_CALUDE_expired_bottle_probability_l1917_191758


namespace NUMINAMATH_CALUDE_dinners_sold_in_four_days_l1917_191763

/-- Calculates the total number of dinners sold over 4 days given specific sales patterns. -/
def total_dinners_sold (monday : ℕ) : ℕ :=
  let tuesday := monday + 40
  let wednesday := tuesday / 2
  let thursday := wednesday + 3
  monday + tuesday + wednesday + thursday

/-- Theorem stating that given the specific sales pattern, 203 dinners were sold over 4 days. -/
theorem dinners_sold_in_four_days : total_dinners_sold 40 = 203 := by
  sorry

end NUMINAMATH_CALUDE_dinners_sold_in_four_days_l1917_191763


namespace NUMINAMATH_CALUDE_tripod_height_is_2_sqrt_5_l1917_191727

/-- A tripod with two legs of length 6 and one leg of length 4 -/
structure Tripod :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (leg3 : ℝ)
  (h : leg1 = 6)
  (i : leg2 = 6)
  (j : leg3 = 4)

/-- The height of the tripod when fully extended -/
def tripod_height (t : Tripod) : ℝ := sorry

/-- Theorem stating that the height of the tripod is 2√5 -/
theorem tripod_height_is_2_sqrt_5 (t : Tripod) : tripod_height t = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_tripod_height_is_2_sqrt_5_l1917_191727


namespace NUMINAMATH_CALUDE_sum_diff_parity_l1917_191778

theorem sum_diff_parity (a b : ℤ) : Even (a + b) ↔ Even (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sum_diff_parity_l1917_191778
