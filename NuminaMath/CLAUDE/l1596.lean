import Mathlib

namespace NUMINAMATH_CALUDE_jelly_beans_remaining_l1596_159638

theorem jelly_beans_remaining (initial_beans : ℕ) (total_people : ℕ) (first_group : ℕ) (second_group : ℕ) 
  (beans_per_second_group : ℕ) :
  initial_beans = 8000 →
  total_people = 10 →
  first_group = 6 →
  second_group = 4 →
  beans_per_second_group = 400 →
  total_people = first_group + second_group →
  initial_beans - (first_group * (2 * beans_per_second_group) + second_group * beans_per_second_group) = 1600 :=
by sorry

end NUMINAMATH_CALUDE_jelly_beans_remaining_l1596_159638


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l1596_159653

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

def inverse_A : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]

theorem matrix_inverse_proof :
  (Matrix.det A ≠ 0 ∧ A * inverse_A = 1 ∧ inverse_A * A = 1) ∨
  (Matrix.det A = 0 ∧ inverse_A = 0) := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l1596_159653


namespace NUMINAMATH_CALUDE_square_difference_l1596_159628

theorem square_difference (m n : ℕ+) 
  (h : (2001 : ℕ) * m ^ 2 + m = (2002 : ℕ) * n ^ 2 + n) :
  ∃ k : ℕ, (m : ℤ) - (n : ℤ) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1596_159628


namespace NUMINAMATH_CALUDE_max_g_given_max_f_l1596_159641

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem max_g_given_max_f (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 1, |f a b c x| ≤ 1) →
  (∃ a' b' c', ∀ x ∈ Set.Icc 0 1, |g a' b' c' x| ≤ 8 ∧ 
    ∃ x' ∈ Set.Icc 0 1, |g a' b' c' x'| = 8) :=
sorry

end NUMINAMATH_CALUDE_max_g_given_max_f_l1596_159641


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1596_159627

-- Define the vertices of the trapezoid
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (8, 0)
def v3 : ℝ × ℝ := (6, 10)
def v4 : ℝ × ℝ := (2, 10)

-- Define the trapezoid
def isosceles_trapezoid (v1 v2 v3 v4 : ℝ × ℝ) : Prop :=
  -- Add conditions for isosceles trapezoid here
  True

-- Calculate the area of the trapezoid
def trapezoid_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  -- Add area calculation here
  0

-- Theorem statement
theorem isosceles_trapezoid_area :
  isosceles_trapezoid v1 v2 v3 v4 →
  trapezoid_area v1 v2 v3 v4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1596_159627


namespace NUMINAMATH_CALUDE_point_location_l1596_159645

theorem point_location (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) :
  (x * y = -1) ∧ (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_location_l1596_159645


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1596_159631

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a :
  {a : ℝ | Q a ⊆ P} = {0, 1/3, -1/2} := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1596_159631


namespace NUMINAMATH_CALUDE_burger_cost_l1596_159698

/-- Represents the cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ
  fry : ℕ

/-- Alice's purchase -/
def alice_purchase (c : Cost) : ℕ :=
  4 * c.burger + 2 * c.soda + 3 * c.fry

/-- Bill's purchase -/
def bill_purchase (c : Cost) : ℕ :=
  3 * c.burger + c.soda + 2 * c.fry

theorem burger_cost :
  ∃ (c : Cost), alice_purchase c = 480 ∧ bill_purchase c = 360 ∧ c.burger = 80 :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_l1596_159698


namespace NUMINAMATH_CALUDE_max_intersection_points_l1596_159658

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two polygons in a plane -/
structure PolygonConfiguration where
  Q₁ : ConvexPolygon
  Q₂ : ConvexPolygon
  no_shared_segments : Bool
  potentially_intersect : Bool

/-- Theorem: Maximum number of intersection points between two convex polygons -/
theorem max_intersection_points (config : PolygonConfiguration) 
  (h1 : config.Q₁.convex = true)
  (h2 : config.Q₂.convex = true)
  (h3 : config.Q₂.sides ≥ config.Q₁.sides + 3)
  (h4 : config.no_shared_segments = true)
  (h5 : config.potentially_intersect = true) :
  (max_intersections : ℕ) → max_intersections = config.Q₁.sides * config.Q₂.sides :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1596_159658


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1596_159602

/-- The new area of a rectangle after changing its dimensions -/
def new_area (original_area : ℝ) (length_increase : ℝ) (width_decrease : ℝ) : ℝ :=
  original_area * (1 + length_increase) * (1 - width_decrease)

theorem rectangle_area_change :
  new_area 432 0.2 0.1 = 466.56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1596_159602


namespace NUMINAMATH_CALUDE_square_of_difference_l1596_159639

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1596_159639


namespace NUMINAMATH_CALUDE_mark_gigs_total_duration_l1596_159667

/-- Represents the duration of Mark's gigs over two weeks -/
def MarkGigsDuration : ℕ :=
  let days_in_two_weeks : ℕ := 2 * 7
  let gigs_count : ℕ := days_in_two_weeks / 2
  let short_song_duration : ℕ := 5
  let long_song_duration : ℕ := 2 * short_song_duration
  let gig_duration : ℕ := 2 * short_song_duration + long_song_duration
  gigs_count * gig_duration

theorem mark_gigs_total_duration :
  MarkGigsDuration = 140 := by
  sorry

end NUMINAMATH_CALUDE_mark_gigs_total_duration_l1596_159667


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l1596_159646

/-- Represents the fare structure of a taxi service -/
structure TaxiFare where
  baseFare : ℝ
  mileageRate : ℝ
  minuteRate : ℝ

/-- Calculates the total fare for a taxi trip -/
def calculateFare (fare : TaxiFare) (miles : ℝ) (minutes : ℝ) : ℝ :=
  fare.baseFare + fare.mileageRate * miles + fare.minuteRate * minutes

/-- Theorem: Given the fare structure and initial trip data, 
    a 60-mile trip lasting 90 minutes will cost $200 -/
theorem taxi_fare_calculation 
  (fare : TaxiFare)
  (h1 : fare.baseFare = 20)
  (h2 : fare.minuteRate = 0.5)
  (h3 : calculateFare fare 40 60 = 140) :
  calculateFare fare 60 90 = 200 := by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_calculation_l1596_159646


namespace NUMINAMATH_CALUDE_log3_45_not_expressible_l1596_159647

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Given conditions
axiom log3_27 : log 3 27 = 3
axiom log3_81 : log 3 81 = 4

-- Define the property of being expressible without logarithmic tables
def expressible_without_tables (x : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ), f (log 3 27) (log 3 81) = log 3 x

-- Theorem statement
theorem log3_45_not_expressible :
  ¬ expressible_without_tables 45 :=
sorry

end NUMINAMATH_CALUDE_log3_45_not_expressible_l1596_159647


namespace NUMINAMATH_CALUDE_min_value_problem_l1596_159695

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_constraint : x + Real.sqrt 3 * y + z = 6) :
  ∃ (min_val : ℝ), min_val = 37/4 ∧ 
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
  x' + Real.sqrt 3 * y' + z' = 6 → 
  x'^3 + y'^2 + 3*z' ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1596_159695


namespace NUMINAMATH_CALUDE_edward_pen_expenses_l1596_159624

/-- Given Edward's initial money, book expenses, and remaining money, 
    calculate the amount spent on pens. -/
theorem edward_pen_expenses (initial_money : ℕ) (book_expenses : ℕ) (remaining_money : ℕ) 
    (h1 : initial_money = 41)
    (h2 : book_expenses = 6)
    (h3 : remaining_money = 19) :
  initial_money - book_expenses - remaining_money = 16 := by
  sorry

end NUMINAMATH_CALUDE_edward_pen_expenses_l1596_159624


namespace NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_l1596_159652

theorem xy_positive_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x * y > 0 → |x + y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x + y| = |x| + |y| ∧ x * y ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_l1596_159652


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1596_159619

def cloth_problem (total_meters : ℝ) (total_price : ℝ) (loss_per_meter : ℝ) (discount_rate : ℝ) : Prop :=
  let selling_price_per_meter : ℝ := total_price / total_meters
  let discounted_price_per_meter : ℝ := selling_price_per_meter * (1 - discount_rate)
  let cost_price_per_meter : ℝ := discounted_price_per_meter + loss_per_meter
  cost_price_per_meter = 130

theorem cloth_cost_price :
  cloth_problem 450 45000 40 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1596_159619


namespace NUMINAMATH_CALUDE_playground_children_count_l1596_159682

/-- Calculate the final number of children on the playground --/
theorem playground_children_count 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (additional_girls : ℕ) 
  (additional_boys : ℕ) 
  (children_leaving : ℕ) 
  (h1 : initial_girls = 28)
  (h2 : initial_boys = 35)
  (h3 : additional_girls = 5)
  (h4 : additional_boys = 7)
  (h5 : children_leaving = 15) : 
  (initial_girls + initial_boys + additional_girls + additional_boys) - children_leaving = 60 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l1596_159682


namespace NUMINAMATH_CALUDE_area_of_shaded_region_l1596_159665

/-- Given a square composed of 25 congruent smaller squares with a diagonal of 10 cm,
    the total area of all 25 squares is 50 square cm. -/
theorem area_of_shaded_region (diagonal : ℝ) (num_squares : ℕ) : 
  diagonal = 10 → num_squares = 25 → (diagonal^2 / 2) = 50 := by sorry

end NUMINAMATH_CALUDE_area_of_shaded_region_l1596_159665


namespace NUMINAMATH_CALUDE_square_root_7396_squared_l1596_159683

theorem square_root_7396_squared : (Real.sqrt 7396)^2 = 7396 := by sorry

end NUMINAMATH_CALUDE_square_root_7396_squared_l1596_159683


namespace NUMINAMATH_CALUDE_cube_split_contains_2015_l1596_159610

def split_sum (m : ℕ) : ℕ := (m + 2) * (m - 1) / 2

theorem cube_split_contains_2015 (m : ℕ) (h1 : m > 1) :
  (split_sum m ≥ 1007) ∧ (split_sum (m - 1) < 1007) → m = 45 :=
sorry

end NUMINAMATH_CALUDE_cube_split_contains_2015_l1596_159610


namespace NUMINAMATH_CALUDE_solve_for_y_l1596_159632

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1596_159632


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l1596_159601

theorem rectangle_length_fraction_of_circle_radius : 
  ∀ (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ),
  square_area = 900 →
  rectangle_area = 120 →
  rectangle_breadth = 10 →
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l1596_159601


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l1596_159629

/-- The speed of a canoe downstream given its upstream speed and the stream speed -/
theorem canoe_downstream_speed (upstream_speed stream_speed : ℝ) :
  upstream_speed = 3 →
  stream_speed = 4.5 →
  upstream_speed + 2 * stream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l1596_159629


namespace NUMINAMATH_CALUDE_triangle_inequality_implies_equilateral_l1596_159672

/-- A triangle with sides a, b, c, area S, and centroid distances x, y, z from the vertices. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The theorem stating that if a triangle satisfies the given inequality, it is equilateral. -/
theorem triangle_inequality_implies_equilateral (t : Triangle) :
  (t.x + t.y + t.z)^2 ≤ (t.a^2 + t.b^2 + t.c^2)/2 + 2*t.S*Real.sqrt 3 →
  t.a = t.b ∧ t.b = t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_implies_equilateral_l1596_159672


namespace NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l1596_159687

theorem cosine_sine_ratio_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l1596_159687


namespace NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l1596_159623

theorem least_integer_square_72_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 72 ∧ ∀ y : ℤ, y^2 = 2*y + 72 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l1596_159623


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1596_159616

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1596_159616


namespace NUMINAMATH_CALUDE_cupcakes_sold_katie_sold_20_l1596_159607

/-- Represents the cupcake sale problem -/
def cupcake_sale (initial : ℕ) (additional : ℕ) (final : ℕ) : ℕ :=
  initial + additional - final

/-- Theorem: The number of cupcakes sold is equal to the total made minus the final number -/
theorem cupcakes_sold (initial additional final : ℕ) :
  cupcake_sale initial additional final = (initial + additional) - final :=
by
  sorry

/-- Corollary: In Katie's specific case, she sold 20 cupcakes -/
theorem katie_sold_20 :
  cupcake_sale 26 20 26 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_sold_katie_sold_20_l1596_159607


namespace NUMINAMATH_CALUDE_broker_commission_slump_l1596_159674

theorem broker_commission_slump (X : ℝ) (h : X > 0) :
  let Y : ℝ := (4 / 5) * X
  let income_unchanged := 0.04 * X = 0.05 * Y
  let slump_percentage := (1 - Y / X) * 100
  income_unchanged → slump_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_broker_commission_slump_l1596_159674


namespace NUMINAMATH_CALUDE_ratio_200_percent_l1596_159600

theorem ratio_200_percent (x : ℝ) : (6 : ℝ) / x = 2 → x = 3 :=
  sorry

end NUMINAMATH_CALUDE_ratio_200_percent_l1596_159600


namespace NUMINAMATH_CALUDE_miriam_flowers_per_day_l1596_159633

/-- The number of flowers Miriam can take care of in 6 days -/
def total_flowers : ℕ := 360

/-- The number of days Miriam works -/
def work_days : ℕ := 6

/-- The number of flowers Miriam can take care of in one day -/
def flowers_per_day : ℕ := total_flowers / work_days

theorem miriam_flowers_per_day : flowers_per_day = 60 := by
  sorry

end NUMINAMATH_CALUDE_miriam_flowers_per_day_l1596_159633


namespace NUMINAMATH_CALUDE_triangle_has_at_least_two_acute_angles_l1596_159670

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the property that the sum of angles in a triangle is 180°
def validTriangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define an acute angle
def isAcute (angle : Real) : Prop :=
  0 < angle ∧ angle < 90

-- Theorem: A triangle has at least two acute angles
theorem triangle_has_at_least_two_acute_angles (t : Triangle) 
  (h : validTriangle t) : 
  (isAcute t.angle1 ∧ isAcute t.angle2) ∨ 
  (isAcute t.angle1 ∧ isAcute t.angle3) ∨ 
  (isAcute t.angle2 ∧ isAcute t.angle3) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_has_at_least_two_acute_angles_l1596_159670


namespace NUMINAMATH_CALUDE_lucas_L10_units_digit_l1596_159634

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem lucas_L10_units_digit :
  unitsDigit (lucas (lucas 10)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucas_L10_units_digit_l1596_159634


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l1596_159609

theorem quadratic_roots_nature (a : ℝ) (h : a < -1) :
  ∃ (x₁ x₂ : ℝ), 
    (a^3 + 1) * x₁^2 + (a^2 + 1) * x₁ - (a + 1) = 0 ∧
    (a^3 + 1) * x₂^2 + (a^2 + 1) * x₂ - (a + 1) = 0 ∧
    x₁ > 0 ∧ x₂ < 0 ∧ |x₂| < x₁ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l1596_159609


namespace NUMINAMATH_CALUDE_evaluate_expression_l1596_159699

theorem evaluate_expression : (0.5^4 / 0.05^3) = 500 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1596_159699


namespace NUMINAMATH_CALUDE_afternoon_eggs_count_l1596_159678

def initial_eggs : ℕ := 20
def morning_eggs : ℕ := 4
def remaining_eggs : ℕ := 13

theorem afternoon_eggs_count : initial_eggs - morning_eggs - remaining_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_eggs_count_l1596_159678


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1596_159637

theorem complex_fraction_simplification :
  (5 + 7*I) / (2 + 3*I) = 31/13 - (1/13)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1596_159637


namespace NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_ten_l1596_159679

/-- Three points in R² are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

/-- The theorem states that the points (1, -2), (3, k), and (6, 2k - 2) are collinear 
    if and only if k = -10. -/
theorem collinear_points_iff_k_eq_neg_ten :
  ∀ k : ℝ, collinear (1, -2) (3, k) (6, 2*k - 2) ↔ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_ten_l1596_159679


namespace NUMINAMATH_CALUDE_martha_turtles_l1596_159621

theorem martha_turtles (martha_turtles : ℕ) (marion_turtles : ℕ) : 
  marion_turtles = martha_turtles + 20 →
  martha_turtles + marion_turtles = 100 →
  martha_turtles = 40 := by
sorry

end NUMINAMATH_CALUDE_martha_turtles_l1596_159621


namespace NUMINAMATH_CALUDE_total_money_proof_l1596_159644

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ 
  (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
sorry

end NUMINAMATH_CALUDE_total_money_proof_l1596_159644


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1596_159604

theorem solution_set_equivalence (x : ℝ) :
  (x + 1) * (x - 1) < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1596_159604


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1596_159694

-- Problem 1
theorem problem_1 (z : ℂ) (h : z = (Complex.I - 1) / Real.sqrt 2) :
  z^20 + z^10 + 1 = -Complex.I := by sorry

-- Problem 2
theorem problem_2 (z : ℂ) (h : Complex.abs (z - (3 + 4*Complex.I)) = 1) :
  4 ≤ Complex.abs z ∧ Complex.abs z ≤ 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1596_159694


namespace NUMINAMATH_CALUDE_fourth_month_sales_l1596_159686

def sales_month1 : ℕ := 6535
def sales_month2 : ℕ := 6927
def sales_month3 : ℕ := 6855
def sales_month5 : ℕ := 6562
def sales_month6 : ℕ := 4891
def required_average : ℕ := 6500
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_month4 : ℕ),
    (sales_month1 + sales_month2 + sales_month3 + sales_month4 + sales_month5 + sales_month6) / num_months = required_average ∧
    sales_month4 = 7230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l1596_159686


namespace NUMINAMATH_CALUDE_jean_jail_time_l1596_159671

theorem jean_jail_time 
  (arson_counts : ℕ)
  (burglary_charges : ℕ)
  (arson_sentence : ℕ)
  (burglary_sentence : ℕ)
  (h1 : arson_counts = 3)
  (h2 : burglary_charges = 2)
  (h3 : arson_sentence = 36)
  (h4 : burglary_sentence = 18)
  : 
  arson_counts * arson_sentence + 
  burglary_charges * burglary_sentence + 
  (6 * burglary_charges) * (burglary_sentence / 3) = 216 := by
  sorry

#check jean_jail_time

end NUMINAMATH_CALUDE_jean_jail_time_l1596_159671


namespace NUMINAMATH_CALUDE_unique_valid_pair_l1596_159668

def has_one_solution (a b c : ℝ) : Prop :=
  (b^2 - 4*a*c = 0) ∧ (a ≠ 0)

def valid_pair (b c : ℕ+) : Prop :=
  has_one_solution 1 (2*b) (2*c) ∧ has_one_solution 1 (3*c) (3*b)

theorem unique_valid_pair : ∃! p : ℕ+ × ℕ+, valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_pair_l1596_159668


namespace NUMINAMATH_CALUDE_sin_cos_range_l1596_159669

theorem sin_cos_range (x : ℝ) : 29/27 ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧ Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_range_l1596_159669


namespace NUMINAMATH_CALUDE_product_inspection_theorem_l1596_159620

/-- Represents a collection of products -/
structure ProductCollection where
  total : ℕ
  selected : ℕ

/-- Defines the concept of a population in statistics -/
def population (pc : ProductCollection) : ℕ := pc.total

/-- Defines the concept of a sample in statistics -/
def sample (pc : ProductCollection) : ℕ := pc.selected

/-- Defines the concept of sample size in statistics -/
def sampleSize (pc : ProductCollection) : ℕ := pc.selected

theorem product_inspection_theorem (pc : ProductCollection) 
  (h1 : pc.total = 80) 
  (h2 : pc.selected = 10) 
  (h3 : pc.selected ≤ pc.total) : 
  population pc = 80 ∧ 
  sampleSize pc = 10 ∧ 
  sample pc ≤ population pc := by
  sorry


end NUMINAMATH_CALUDE_product_inspection_theorem_l1596_159620


namespace NUMINAMATH_CALUDE_jane_dolls_l1596_159661

theorem jane_dolls (total : ℕ) (difference : ℕ) : 
  total = 32 → difference = 6 → ∃ (jane : ℕ), jane + (jane + difference) = total ∧ jane = 13 := by
sorry

end NUMINAMATH_CALUDE_jane_dolls_l1596_159661


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l1596_159630

theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 402*x₁ + k = 0 ∧ 
                x₂^2 - 402*x₂ + k = 0 ∧ 
                x₁ + 3 = 80 * x₂) → 
  k = 1985 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l1596_159630


namespace NUMINAMATH_CALUDE_three_card_permutations_standard_deck_l1596_159666

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard deck of cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The number of ways to choose three different cards in a specific order -/
def three_card_permutations (d : Deck) : ℕ :=
  d.total_cards * (d.total_cards - 1) * (d.total_cards - 2)

/-- Theorem: The number of ways to choose three different cards in a specific order
    from a standard deck is 132600 -/
theorem three_card_permutations_standard_deck :
  three_card_permutations standard_deck = 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_permutations_standard_deck_l1596_159666


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1596_159615

/-- Set M defined as {x | x^2 - x < 0} -/
def M : Set ℝ := {x | x^2 - x < 0}

/-- Set N defined as {x | |x| < 2} -/
def N : Set ℝ := {x | |x| < 2}

/-- Theorem stating that the intersection of M and N equals M -/
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1596_159615


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1596_159691

/-- The equation of a line perpendicular to x + y = 0 and passing through (-1, 0) -/
theorem perpendicular_line_equation :
  let l1 : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}  -- Line x + y = 0
  let point : ℝ × ℝ := (-1, 0)                -- Given point
  let l2 : Set (ℝ × ℝ) := {p | p.1 - p.2 + 1 = 0}  -- Claimed perpendicular line
  (∀ p ∈ l2, (p.1 - point.1) * (p.1 + p.2) = -(p.2 - point.2) * (p.1 + p.2)) ∧  -- Perpendicularity condition
  (point ∈ l2)  -- Point (-1, 0) lies on the line
  :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1596_159691


namespace NUMINAMATH_CALUDE_sock_problem_solution_l1596_159659

/-- Represents the number of pairs of socks at each price point --/
structure SockInventory where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Calculates the total number of sock pairs --/
def total_pairs (s : SockInventory) : ℕ :=
  s.two_dollar + s.four_dollar + s.five_dollar

/-- Calculates the total cost of all socks --/
def total_cost (s : SockInventory) : ℕ :=
  2 * s.two_dollar + 4 * s.four_dollar + 5 * s.five_dollar

theorem sock_problem_solution :
  ∃ (s : SockInventory),
    total_pairs s = 15 ∧
    total_cost s = 41 ∧
    s.two_dollar ≥ 1 ∧
    s.four_dollar ≥ 1 ∧
    s.five_dollar ≥ 1 ∧
    s.two_dollar = 11 :=
by sorry

#check sock_problem_solution

end NUMINAMATH_CALUDE_sock_problem_solution_l1596_159659


namespace NUMINAMATH_CALUDE_defeated_candidate_vote_percentage_l1596_159689

theorem defeated_candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_votes : ℕ)
  (losing_margin : ℕ)
  (h_total : total_votes = 12600)
  (h_invalid : invalid_votes = 100)
  (h_margin : losing_margin = 5000) :
  (((total_votes - invalid_votes : ℚ) / 2 - losing_margin) / (total_votes - invalid_votes)) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_defeated_candidate_vote_percentage_l1596_159689


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l1596_159681

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 37

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

theorem fgh_supermarkets_count :
  us_supermarkets = 37 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l1596_159681


namespace NUMINAMATH_CALUDE_parkway_soccer_players_l1596_159697

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) 
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : girls_not_playing = 135)
  (h4 : (86 : ℚ) / 100 * (total_students - (total_students - boys - girls_not_playing)) = boys - (total_students - boys - girls_not_playing)) :
  total_students - (total_students - boys - girls_not_playing) = 250 := by
  sorry

end NUMINAMATH_CALUDE_parkway_soccer_players_l1596_159697


namespace NUMINAMATH_CALUDE_prize_probability_l1596_159676

theorem prize_probability (p : ℝ) (h : p = 0.9) :
  Nat.choose 5 3 * p^3 * (1 - p)^2 = Nat.choose 5 3 * 0.9^3 * 0.1^2 := by
  sorry

end NUMINAMATH_CALUDE_prize_probability_l1596_159676


namespace NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l1596_159618

/-- Represents the rules and structure of the arm wrestling tournament -/
structure Tournament :=
  (num_athletes : Nat)
  (point_diff_limit : Nat)
  (extra_point_rule : Bool)

/-- Calculates the minimum number of rounds required for a tournament -/
def min_rounds (t : Tournament) : Nat :=
  sorry

/-- The main theorem stating that a tournament with 510 athletes requires at least 9 rounds -/
theorem arm_wrestling_tournament_rounds :
  ∀ (t : Tournament),
    t.num_athletes = 510 ∧
    t.point_diff_limit = 1 ∧
    t.extra_point_rule = true →
    min_rounds t ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l1596_159618


namespace NUMINAMATH_CALUDE_pet_store_kittens_l1596_159622

theorem pet_store_kittens (initial : ℕ) (final : ℕ) (new : ℕ) : 
  initial = 6 → final = 9 → new = final - initial → new = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l1596_159622


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l1596_159693

/-- A rectangular box with given surface area and edge length sum has a specific sum of interior diagonal lengths. -/
theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + c * a) = 206)
  (h_edge_sum : 4 * (a + b + c) = 64) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l1596_159693


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1596_159680

/-- Given three points A, B, and C in 2D space, this function checks if they are collinear --/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.2 = AB.2 * BC.1

/-- Theorem stating that if A(k, 12), B(4, 5), and C(10, k) are collinear, then k = 11 or k = -2 --/
theorem collinear_points_k_value (k : ℝ) :
  are_collinear (k, 12) (4, 5) (10, k) → k = 11 ∨ k = -2 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l1596_159680


namespace NUMINAMATH_CALUDE_chef_pies_l1596_159625

theorem chef_pies (apple_pies pecan_pies pumpkin_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pecan_pies = 4) 
  (h3 : pumpkin_pies = 7) : 
  apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end NUMINAMATH_CALUDE_chef_pies_l1596_159625


namespace NUMINAMATH_CALUDE_stream_speed_l1596_159654

/-- Given a boat that travels at 14 km/hr in still water and covers 72 km downstream in 3.6 hours,
    prove that the speed of the stream is 6 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 14 →
  distance = 72 →
  time = 3.6 →
  stream_speed = (distance / time) - boat_speed →
  stream_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1596_159654


namespace NUMINAMATH_CALUDE_alloy_mixing_solution_exists_l1596_159613

/-- Represents an alloy of copper and tin -/
structure Alloy where
  mass : ℝ
  copper_percentage : ℝ

/-- Proves that a solution exists for the alloy mixing problem if and only if p is within the specified range -/
theorem alloy_mixing_solution_exists (alloy1 alloy2 : Alloy) (target_mass : ℝ) (p : ℝ) :
  alloy1.mass = 3 ∧ 
  alloy1.copper_percentage = 40 ∧
  alloy2.mass = 7 ∧
  alloy2.copper_percentage = 30 ∧
  target_mass = 8 →
  (∃ x : ℝ, 
    0 ≤ x ∧ 
    x ≤ alloy1.mass ∧ 
    0 ≤ target_mass - x ∧ 
    target_mass - x ≤ alloy2.mass ∧
    (alloy1.copper_percentage / 100 * x + alloy2.copper_percentage / 100 * (target_mass - x)) / target_mass = p / 100) ↔
  31.25 ≤ p ∧ p ≤ 33.75 := by
  sorry

#check alloy_mixing_solution_exists

end NUMINAMATH_CALUDE_alloy_mixing_solution_exists_l1596_159613


namespace NUMINAMATH_CALUDE_number_problem_l1596_159655

theorem number_problem (x : ℝ) : (6 * x) / 1.5 = 3.8 → x = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1596_159655


namespace NUMINAMATH_CALUDE_gilled_mushroom_count_l1596_159696

/-- Represents the number of mushrooms on a log -/
structure MushroomCount where
  total : ℕ
  gilled : ℕ
  spotted : ℕ

/-- Conditions for the mushroom problem -/
def mushroom_conditions (m : MushroomCount) : Prop :=
  m.total = 30 ∧
  m.gilled + m.spotted = m.total ∧
  m.spotted = 9 * m.gilled

/-- Theorem stating the number of gilled mushrooms -/
theorem gilled_mushroom_count (m : MushroomCount) :
  mushroom_conditions m → m.gilled = 3 := by
  sorry

end NUMINAMATH_CALUDE_gilled_mushroom_count_l1596_159696


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1596_159626

theorem quadratic_inequality_solution_set (a : ℝ) :
  {x : ℝ | x^2 - (2*a + 1)*x + a^2 + a < 0} = Set.Ioo a (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1596_159626


namespace NUMINAMATH_CALUDE_min_bullseyes_is_52_l1596_159605

/-- The number of shots in the archery tournament -/
def total_shots : ℕ := 120

/-- Chelsea's minimum score on each shot -/
def chelsea_min_score : ℕ := 5

/-- Score for a bullseye -/
def bullseye_score : ℕ := 12

/-- Chelsea's lead at halfway point -/
def chelsea_lead : ℕ := 60

/-- The number of shots taken so far -/
def shots_taken : ℕ := total_shots / 2

/-- Function to calculate the minimum number of bullseyes Chelsea needs to guarantee victory -/
def min_bullseyes_for_victory : ℕ :=
  let max_opponent_score := shots_taken * bullseye_score + chelsea_lead
  let chelsea_non_bullseye_score := (total_shots - shots_taken) * chelsea_min_score
  ((max_opponent_score - chelsea_non_bullseye_score) / (bullseye_score - chelsea_min_score)) + 1

/-- Theorem stating that the minimum number of bullseyes Chelsea needs is 52 -/
theorem min_bullseyes_is_52 : min_bullseyes_for_victory = 52 := by
  sorry

end NUMINAMATH_CALUDE_min_bullseyes_is_52_l1596_159605


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1596_159608

theorem quadratic_solution_difference_squared :
  ∀ α β : ℝ,
  (α^2 - 5*α + 6 = 0) →
  (β^2 - 5*β + 6 = 0) →
  (α ≠ β) →
  (α - β)^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1596_159608


namespace NUMINAMATH_CALUDE_max_red_surface_area_76_l1596_159684

/-- Represents a small cube with two red faces -/
inductive SmallCube
| Adjacent : SmallCube  -- Two adjacent faces are red
| Opposite : SmallCube  -- Two opposite faces are red

/-- Configuration of small cubes -/
structure CubeConfiguration where
  total : Nat
  adjacent : Nat
  opposite : Nat

/-- Represents the large cube assembled from small cubes -/
structure LargeCube where
  config : CubeConfiguration
  side_length : Nat

/-- Calculates the maximum red surface area of the large cube -/
def max_red_surface_area (lc : LargeCube) : Nat :=
  sorry

/-- Theorem stating the maximum red surface area for the given configuration -/
theorem max_red_surface_area_76 :
  ∀ (lc : LargeCube),
    lc.config.total = 64 ∧
    lc.config.adjacent = 20 ∧
    lc.config.opposite = 44 ∧
    lc.side_length = 4 →
    max_red_surface_area lc = 76 :=
  sorry

end NUMINAMATH_CALUDE_max_red_surface_area_76_l1596_159684


namespace NUMINAMATH_CALUDE_interest_percentage_calculation_l1596_159656

/-- Calculates the interest percentage for a purchase with a payment plan -/
theorem interest_percentage_calculation (purchase_price : ℝ) (down_payment : ℝ) (monthly_payment : ℝ) (num_months : ℕ) :
  purchase_price = 110 →
  down_payment = 10 →
  monthly_payment = 10 →
  num_months = 12 →
  let total_paid := down_payment + (monthly_payment * num_months)
  let interest_paid := total_paid - purchase_price
  let interest_percentage := (interest_paid / purchase_price) * 100
  ∃ ε > 0, |interest_percentage - 18.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_interest_percentage_calculation_l1596_159656


namespace NUMINAMATH_CALUDE_bank_profit_l1596_159649

/-- Bank's profit calculation -/
theorem bank_profit 
  (K : ℝ) (p p₁ : ℝ) (n : ℕ) 
  (h₁ : p₁ > p) 
  (h₂ : p > 0) 
  (h₃ : p₁ > 0) :
  K * ((1 + p₁ / 100) ^ n - (1 + p / 100) ^ n) = 
  K * ((1 + p₁ / 100) ^ n - (1 + p / 100) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_bank_profit_l1596_159649


namespace NUMINAMATH_CALUDE_case_one_case_two_l1596_159690

-- Define the set M
def M := {f : ℤ → ℝ | f 0 ≠ 0 ∧ ∀ n m : ℤ, f n * f m = f (n + m) + f (n - m)}

-- Statement for the first case
theorem case_one (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = 5/2) :
  ∀ n : ℤ, f n = 2^n + 2^(-n) :=
sorry

-- Statement for the second case
theorem case_two (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = Real.sqrt 3) :
  ∀ n : ℤ, f n = 2 * Real.cos (π * n / 6) :=
sorry

end NUMINAMATH_CALUDE_case_one_case_two_l1596_159690


namespace NUMINAMATH_CALUDE_total_toy_count_l1596_159663

/-- The number of toys each person has -/
structure ToyCount where
  jaxon : ℕ
  gabriel : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def problem_conditions (tc : ToyCount) : Prop :=
  tc.jaxon = 15 ∧
  tc.gabriel = 2 * tc.jaxon ∧
  tc.jerry = tc.gabriel + 8

/-- The theorem to prove -/
theorem total_toy_count (tc : ToyCount) 
  (h : problem_conditions tc) : tc.jaxon + tc.gabriel + tc.jerry = 83 := by
  sorry


end NUMINAMATH_CALUDE_total_toy_count_l1596_159663


namespace NUMINAMATH_CALUDE_whole_number_between_constraints_l1596_159648

theorem whole_number_between_constraints (N : ℤ) : 
  (6 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7.5) ↔ N ∈ ({25, 26, 27, 28, 29} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_whole_number_between_constraints_l1596_159648


namespace NUMINAMATH_CALUDE_circle_area_theorem_l1596_159662

/-- Given a circle with radius R and four smaller circles with radius R/2 constructed
    as described in the problem, this theorem states that the sum of the areas of the
    overlapping parts of the smaller circles equals the area of the original circle
    minus the areas of the non-overlapping parts of the smaller circles. -/
theorem circle_area_theorem (R : ℝ) (h : R > 0) :
  let big_circle_area := π * R^2
  let small_circle_area := π * (R/2)^2
  let segment_area := (π/4 - 1/2) * R^2
  let overlap_area := 2 * segment_area
  overlap_area = big_circle_area - 4 * (small_circle_area - segment_area) := by
  sorry


end NUMINAMATH_CALUDE_circle_area_theorem_l1596_159662


namespace NUMINAMATH_CALUDE_tiffany_pies_eaten_l1596_159636

theorem tiffany_pies_eaten (pies_per_day : ℕ) (days : ℕ) (cans_per_pie : ℕ) (remaining_cans : ℕ) : 
  pies_per_day = 3 → days = 11 → cans_per_pie = 2 → remaining_cans = 58 →
  (pies_per_day * days * cans_per_pie - remaining_cans) / cans_per_pie = 4 := by
sorry

end NUMINAMATH_CALUDE_tiffany_pies_eaten_l1596_159636


namespace NUMINAMATH_CALUDE_ratio_and_closest_whole_number_l1596_159650

theorem ratio_and_closest_whole_number : 
  let ratio := (10^2010 + 10^2013) / (10^2011 + 10^2014)
  ratio = 1/10 ∧ 
  ∀ n : ℤ, |ratio - (n : ℚ)| ≥ |ratio - 0| :=
by sorry

end NUMINAMATH_CALUDE_ratio_and_closest_whole_number_l1596_159650


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1596_159677

theorem consecutive_numbers_sum (a b c d : ℤ) : 
  b = a + 1 → 
  c = b + 1 → 
  d = c + 1 → 
  b * c = 2970 → 
  a + d = 113 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1596_159677


namespace NUMINAMATH_CALUDE_quadratic_root_form_l1596_159660

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l1596_159660


namespace NUMINAMATH_CALUDE_curve_crosses_at_point_one_eight_l1596_159611

-- Define the curve
def x (t : ℝ) : ℝ := 2 * t^2 + 1
def y (t : ℝ) : ℝ := 2 * t^3 - 6 * t^2 + 8

-- Theorem statement
theorem curve_crosses_at_point_one_eight :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 1 ∧ y a = 8 := by
  sorry

end NUMINAMATH_CALUDE_curve_crosses_at_point_one_eight_l1596_159611


namespace NUMINAMATH_CALUDE_expression_factorization_l1596_159606

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1596_159606


namespace NUMINAMATH_CALUDE_inequality_proof_l1596_159657

/-- The function f(x) = |x - a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- The theorem to be proved -/
theorem inequality_proof (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
  (h3 : Set.Icc 0 2 = {x : ℝ | f 1 x ≤ 1})
  (h4 : 1/m + 1/(2*n) = 1) :
  m + 4*n ≥ 2*Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1596_159657


namespace NUMINAMATH_CALUDE_relationship_abc_l1596_159617

theorem relationship_abc : 
  let a : ℝ := 1 + Real.sqrt 7
  let b : ℝ := Real.sqrt 3 + Real.sqrt 5
  let c : ℝ := 4
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1596_159617


namespace NUMINAMATH_CALUDE_expression_evaluation_l1596_159643

theorem expression_evaluation (a : ℝ) (h : a = 2023) : 
  ((a + 1) / (a - 1) + 1) / (2 * a / (a^2 - 1)) = 2024 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1596_159643


namespace NUMINAMATH_CALUDE_sum_of_digits_inequality_l1596_159612

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_inequality (N : ℕ) :
  sum_of_digits N ≤ 5 * sum_of_digits (5^5 * N) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_inequality_l1596_159612


namespace NUMINAMATH_CALUDE_set_forms_triangle_l1596_159673

/-- Triangle Inequality Theorem: A set of three positive real numbers a, b, c can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set (7, 15, 10) can form a triangle. -/
theorem set_forms_triangle : can_form_triangle 7 15 10 := by
  sorry


end NUMINAMATH_CALUDE_set_forms_triangle_l1596_159673


namespace NUMINAMATH_CALUDE_max_dot_product_l1596_159651

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1 ∧ x ≠ 2 ∧ x ≠ -2

-- Define points A, B, and D
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)
def D : ℝ × ℝ := (1, 0)

-- Define the slope product condition for point E
def slope_product_condition (E : ℝ × ℝ) : Prop :=
  let (x, y) := E
  (y / (x - A.1)) * (y / (x - B.1)) = -1/4

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product :
  ∀ P Q : ℝ × ℝ,
  C P.1 P.2 →
  C Q.1 Q.2 →
  ∃ k : ℝ, Q.2 - D.2 = k * (Q.1 - D.1) ∧ P.2 - D.2 = k * (P.1 - D.1) →
  dot_product P Q ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l1596_159651


namespace NUMINAMATH_CALUDE_jean_jane_money_total_jean_jane_money_total_proof_l1596_159664

/-- Given that Jean has three times as much money as Jane, and Jean has $57,
    prove that their combined total is $76. -/
theorem jean_jane_money_total : ℕ → ℕ → Prop :=
  fun jean_money jane_money =>
    (jean_money = 3 * jane_money) →
    (jean_money = 57) →
    (jean_money + jane_money = 76)

/-- The actual theorem instance -/
theorem jean_jane_money_total_proof : jean_jane_money_total 57 19 := by
  sorry

end NUMINAMATH_CALUDE_jean_jane_money_total_jean_jane_money_total_proof_l1596_159664


namespace NUMINAMATH_CALUDE_jenny_wedding_budget_l1596_159614

/-- Calculates the total catering budget for a wedding --/
def totalCateringBudget (totalGuests : ℕ) (steakMultiplier : ℕ) (steakCost chickenCost : ℚ) : ℚ :=
  let chickenGuests := totalGuests / (steakMultiplier + 1)
  let steakGuests := totalGuests - chickenGuests
  steakGuests * steakCost + chickenGuests * chickenCost

/-- Proves that the total catering budget for Jenny's wedding is $1860 --/
theorem jenny_wedding_budget :
  totalCateringBudget 80 3 25 18 = 1860 := by
  sorry

end NUMINAMATH_CALUDE_jenny_wedding_budget_l1596_159614


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1596_159640

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 4, a_3 * a_5 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) (h_a4 : a 4 = 4) : a 3 * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1596_159640


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1596_159635

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | x > 3}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1596_159635


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l1596_159675

/-- A furniture shop owner charges 10% more than the cost price. If a customer paid Rs. 8800 for a computer table, then the cost price of the computer table was Rs. 8000. -/
theorem computer_table_cost_price (selling_price : ℝ) (markup_percentage : ℝ) 
  (h1 : selling_price = 8800)
  (h2 : markup_percentage = 0.10) : 
  ∃ (cost_price : ℝ), cost_price = 8000 ∧ selling_price = cost_price * (1 + markup_percentage) := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l1596_159675


namespace NUMINAMATH_CALUDE_price_difference_is_1090_l1596_159688

/-- The difference in cents between the TV advertiser price and the in-store price for a microwave --/
def price_difference : ℚ :=
  let in_store_price : ℚ := 149.95
  let tv_payment : ℚ := 27.99
  let shipping_fee : ℚ := 14.95
  let warranty_fee : ℚ := 5.95
  let tv_price : ℚ := 5 * tv_payment + shipping_fee + warranty_fee
  (tv_price - in_store_price) * 100

/-- The price difference is 1090 cents --/
theorem price_difference_is_1090 : 
  price_difference = 1090 := by sorry

end NUMINAMATH_CALUDE_price_difference_is_1090_l1596_159688


namespace NUMINAMATH_CALUDE_fruit_salad_weight_l1596_159692

/-- The amount of melon in pounds used in the fruit salad -/
def melon_weight : ℚ := 0.25

/-- The amount of berries in pounds used in the fruit salad -/
def berries_weight : ℚ := 0.38

/-- The total amount of fruit in pounds used in the fruit salad -/
def total_fruit_weight : ℚ := melon_weight + berries_weight

theorem fruit_salad_weight : total_fruit_weight = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_weight_l1596_159692


namespace NUMINAMATH_CALUDE_gcd_product_l1596_159603

theorem gcd_product (a b a' b' : ℕ+) (d d' : ℕ+) 
  (h1 : d = Nat.gcd a b) (h2 : d' = Nat.gcd a' b') : 
  Nat.gcd (a * a') (Nat.gcd (a * b') (Nat.gcd (b * a') (b * b'))) = d * d' := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_l1596_159603


namespace NUMINAMATH_CALUDE_stratified_sampling_senior_managers_l1596_159685

theorem stratified_sampling_senior_managers 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (senior_managers : ℕ) 
  (h1 : total_population = 200) 
  (h2 : sample_size = 40) 
  (h3 : senior_managers = 10) :
  (sample_size : ℚ) / total_population * senior_managers = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_senior_managers_l1596_159685


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l1596_159642

/-- The amount of milk Rachel drinks given the initial amount and fractions poured and drunk -/
theorem rachel_milk_consumption (initial_milk : ℚ) 
  (h1 : initial_milk = 3 / 7)
  (poured_fraction : ℚ) 
  (h2 : poured_fraction = 1 / 2)
  (drunk_fraction : ℚ)
  (h3 : drunk_fraction = 3 / 4) : 
  drunk_fraction * (poured_fraction * initial_milk) = 9 / 56 := by
  sorry

#check rachel_milk_consumption

end NUMINAMATH_CALUDE_rachel_milk_consumption_l1596_159642
