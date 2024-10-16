import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solution_l1597_159747

theorem inequality_system_solution :
  let S := {x : ℝ | 3 < x ∧ x ≤ 4}
  S = {x : ℝ | 3*x + 4 ≥ 4*x ∧ 2*(x - 1) + x > 7} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1597_159747


namespace NUMINAMATH_CALUDE_solve_work_problem_l1597_159741

def work_problem (a_days b_days : ℕ) (b_share : ℚ) : Prop :=
  a_days > 0 ∧ b_days > 0 ∧ b_share > 0 →
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let total_rate : ℚ := a_rate + b_rate
  let a_proportion : ℚ := a_rate / total_rate
  let b_proportion : ℚ := b_rate / total_rate
  let total_amount : ℚ := b_share / b_proportion
  total_amount = 1000

theorem solve_work_problem :
  work_problem 30 20 600 := by
  sorry

end NUMINAMATH_CALUDE_solve_work_problem_l1597_159741


namespace NUMINAMATH_CALUDE_cody_chocolate_boxes_cody_bought_seven_boxes_l1597_159737

theorem cody_chocolate_boxes : ℕ → Prop :=
  fun x =>
    -- x is the number of boxes of chocolate candy
    -- 3 is the number of boxes of caramel candy
    -- 8 is the number of pieces in each box
    -- 80 is the total number of pieces
    x * 8 + 3 * 8 = 80 →
    x = 7

-- The proof
theorem cody_bought_seven_boxes : cody_chocolate_boxes 7 := by
  sorry

end NUMINAMATH_CALUDE_cody_chocolate_boxes_cody_bought_seven_boxes_l1597_159737


namespace NUMINAMATH_CALUDE_senior_class_size_l1597_159759

/-- The number of students in the senior class at East High School -/
def total_students : ℕ := 400

/-- The proportion of students who play sports -/
def sports_proportion : ℚ := 52 / 100

/-- The proportion of sports-playing students who play soccer -/
def soccer_proportion : ℚ := 125 / 1000

/-- The number of students who play soccer -/
def soccer_players : ℕ := 26

theorem senior_class_size :
  (total_students : ℚ) * sports_proportion * soccer_proportion = soccer_players := by
  sorry

end NUMINAMATH_CALUDE_senior_class_size_l1597_159759


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1597_159734

/-- The number of distinct rectangles in a square grid --/
def num_rectangles (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem: In a 5x5 square grid, the number of distinct rectangles
    with sides parallel to the grid lines is 100 --/
theorem rectangles_in_5x5_grid :
  num_rectangles 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1597_159734


namespace NUMINAMATH_CALUDE_pinterest_group_average_pins_l1597_159736

/-- The average number of pins contributed per day by each member in a Pinterest group. -/
def average_pins_per_day (
  group_size : ℕ
  ) (
  initial_pins : ℕ
  ) (
  final_pins : ℕ
  ) (
  days : ℕ
  ) (
  deleted_pins_per_week_per_person : ℕ
  ) : ℚ :=
  let total_deleted_pins := (group_size * deleted_pins_per_week_per_person * (days / 7) : ℚ)
  let total_new_pins := (final_pins - initial_pins : ℚ) + total_deleted_pins
  total_new_pins / (group_size * days : ℚ)

/-- Theorem stating that the average number of pins contributed per day is 10. -/
theorem pinterest_group_average_pins :
  average_pins_per_day 20 1000 6600 30 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_average_pins_l1597_159736


namespace NUMINAMATH_CALUDE_sum_31_22_base4_l1597_159714

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_31_22_base4 :
  toBase4 (31 + 22) = [3, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_31_22_base4_l1597_159714


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l1597_159765

/-- Given a polynomial f(x,y,z) = x³ + 2y³ + 4z³ - 6xyz, prove that for all real numbers a, b, and c,
    f(a,b,c) = 0 if and only if a + b∛2 + c∛4 = 0 -/
theorem polynomial_equivalence (a b c : ℝ) :
  a^3 + 2*b^3 + 4*c^3 - 6*a*b*c = 0 ↔ a + b*(2^(1/3)) + c*(4^(1/3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l1597_159765


namespace NUMINAMATH_CALUDE_money_lasts_four_weeks_l1597_159719

def total_earnings : ℕ := 27
def weekly_expenses : ℕ := 6

theorem money_lasts_four_weeks :
  (total_earnings / weekly_expenses : ℕ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_money_lasts_four_weeks_l1597_159719


namespace NUMINAMATH_CALUDE_inverse_function_value_l1597_159705

/-- Given that f is the inverse function of g(x) = ax, and f(4) = 2, prove that a = 2 -/
theorem inverse_function_value (a : ℝ) (f g : ℝ → ℝ) :
  (∀ x, g x = a * x) →  -- g is defined as g(x) = ax
  (∀ x, f (g x) = x) →  -- f is the inverse function of g
  (∀ x, g (f x) = x) →  -- f is the inverse function of g (reverse composition)
  f 4 = 2 →             -- f(4) = 2
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_value_l1597_159705


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1597_159782

/-- Represents the measurement error in a rectangle's dimensions and area -/
structure RectangleMeasurement where
  length_excess : ℝ  -- Percentage excess in length measurement
  width_deficit : ℝ  -- Percentage deficit in width measurement
  area_error : ℝ     -- Percentage error in calculated area

/-- Theorem stating the relationship between measurement errors in a rectangle -/
theorem rectangle_measurement_error 
  (r : RectangleMeasurement) 
  (h1 : r.length_excess = 8)
  (h2 : r.area_error = 2.6) :
  r.width_deficit = 5 :=
sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1597_159782


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1597_159735

theorem smallest_number_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 2 ∧ 
  x % 7 = 3 ∧ 
  ∀ y : ℕ, y > 0 → y % 3 = 1 → y % 4 = 2 → y % 7 = 3 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1597_159735


namespace NUMINAMATH_CALUDE_train_speed_l1597_159722

theorem train_speed (t_pole : ℝ) (t_cross : ℝ) (l_stationary : ℝ) :
  t_pole = 8 →
  t_cross = 18 →
  l_stationary = 400 →
  ∃ v l, v = l / t_pole ∧ v = (l + l_stationary) / t_cross ∧ v = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l1597_159722


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l1597_159769

theorem min_value_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), (∀ (x : ℝ), Real.sqrt (x^2 + (1 + x)^2) + Real.sqrt ((1 + x)^2 + (1 - x)^2) ≥ Real.sqrt 5) ∧
  (Real.sqrt (y^2 + (1 + y)^2) + Real.sqrt ((1 + y)^2 + (1 - y)^2) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l1597_159769


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l1597_159764

theorem average_of_three_numbers (x : ℝ) : 
  (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l1597_159764


namespace NUMINAMATH_CALUDE_find_n_l1597_159755

theorem find_n (n : ℕ) : lcm n 16 = 48 → gcd n 16 = 4 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1597_159755


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1597_159717

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h₁ : d ≠ 0
  h₂ : ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: If a₁, a₄, and a₅ of an arithmetic sequence form a geometric sequence,
    then the common ratio of this geometric sequence is 1/3 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h : (seq.a 4) ^ 2 = (seq.a 1) * (seq.a 5)) :
  (seq.a 4) / (seq.a 1) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1597_159717


namespace NUMINAMATH_CALUDE_parabola_property_l1597_159727

/-- Given a parabola y = ax² + bx + c with the following points:
    (-2, 0), (-1, 4), (0, 6), (1, 6)
    Prove that (a - b + c)(4a + 2b + c) > 0 -/
theorem parabola_property (a b c : ℝ) : 
  (4 * a - 2 * b + c = 0) →
  (a - b + c = 4) →
  (c = 6) →
  (a * 1^2 + b * 1 + c = 6) →
  (a - b + c) * (4 * a + 2 * b + c) > 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_property_l1597_159727


namespace NUMINAMATH_CALUDE_total_legs_in_pasture_l1597_159757

/-- The number of cows in the pasture -/
def num_cows : ℕ := 115

/-- The number of legs each cow has -/
def legs_per_cow : ℕ := 4

/-- Theorem: The total number of legs seen in a pasture with 115 cows, 
    where each cow has 4 legs, is equal to 460. -/
theorem total_legs_in_pasture : num_cows * legs_per_cow = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_pasture_l1597_159757


namespace NUMINAMATH_CALUDE_hen_count_is_28_l1597_159713

/-- Represents the count of animals on a farm -/
structure FarmCount where
  hens : ℕ
  cows : ℕ

/-- Checks if the farm count satisfies the given conditions -/
def isValidCount (farm : FarmCount) : Prop :=
  farm.hens + farm.cows = 48 ∧
  2 * farm.hens + 4 * farm.cows = 136

theorem hen_count_is_28 :
  ∃ (farm : FarmCount), isValidCount farm ∧ farm.hens = 28 :=
by
  sorry

#check hen_count_is_28

end NUMINAMATH_CALUDE_hen_count_is_28_l1597_159713


namespace NUMINAMATH_CALUDE_sum_of_ages_in_ten_years_l1597_159715

/-- Theorem: Sum of ages in 10 years -/
theorem sum_of_ages_in_ten_years (my_current_age brother_current_age : ℕ) : 
  my_current_age = 20 →
  my_current_age + 10 = 2 * (brother_current_age + 10) →
  (my_current_age + 10) + (brother_current_age + 10) = 45 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_ten_years_l1597_159715


namespace NUMINAMATH_CALUDE_cannot_determine_charles_loss_l1597_159731

def willie_initial : ℕ := 48
def charles_initial : ℕ := 14
def willie_future : ℕ := 13

theorem cannot_determine_charles_loss :
  ∀ (charles_loss : ℕ),
  ∃ (willie_loss : ℕ),
  willie_initial - willie_loss = willie_future ∧
  charles_initial ≥ charles_loss ∧
  ∃ (charles_loss' : ℕ),
  charles_loss' ≠ charles_loss ∧
  charles_initial ≥ charles_loss' ∧
  willie_initial - willie_loss = willie_future :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_charles_loss_l1597_159731


namespace NUMINAMATH_CALUDE_science_club_problem_l1597_159791

theorem science_club_problem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : math = 85)
  (h3 : physics = 60)
  (h4 : both = 20) :
  total - (math + physics - both) = 25 := by
sorry

end NUMINAMATH_CALUDE_science_club_problem_l1597_159791


namespace NUMINAMATH_CALUDE_solve_family_income_l1597_159725

def family_income_problem (initial_members : ℕ) (new_average : ℚ) (deceased_income : ℚ) : Prop :=
  initial_members = 4 →
  new_average = 650 →
  deceased_income = 1178 →
  ∃ (initial_average : ℚ),
    initial_average * initial_members = new_average * (initial_members - 1) + deceased_income ∧
    initial_average = 782

theorem solve_family_income : family_income_problem 4 650 1178 := by
  sorry

end NUMINAMATH_CALUDE_solve_family_income_l1597_159725


namespace NUMINAMATH_CALUDE_lucas_100th_mod8_l1597_159721

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

def lucas_mod8 (n : ℕ) : ℕ := lucas n % 8

theorem lucas_100th_mod8 : lucas_mod8 99 = 7 := by sorry

end NUMINAMATH_CALUDE_lucas_100th_mod8_l1597_159721


namespace NUMINAMATH_CALUDE_existence_of_subset_l1597_159797

theorem existence_of_subset (n : ℕ+) (t : ℝ) (a : Fin (2*n.val-1) → ℝ) (ht : t ≠ 0) :
  ∃ (s : Finset (Fin (2*n.val-1))), s.card = n.val ∧
    ∀ (i j : Fin (2*n.val-1)), i ∈ s → j ∈ s → i ≠ j → a i - a j ≠ t :=
sorry

end NUMINAMATH_CALUDE_existence_of_subset_l1597_159797


namespace NUMINAMATH_CALUDE_peach_count_l1597_159773

/-- Calculates the total number of peaches after picking -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem: The total number of peaches is the sum of initial and picked peaches -/
theorem peach_count (initial picked : ℕ) :
  total_peaches initial picked = initial + picked := by
  sorry

end NUMINAMATH_CALUDE_peach_count_l1597_159773


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1597_159738

theorem max_value_of_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
    (a + b + c)^2 / (a^2 + b^2 + c^2) = 3) ∧ 
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 1 → 
    (p + q + r)^2 / (p^2 + q^2 + r^2) ≤ 3) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1597_159738


namespace NUMINAMATH_CALUDE_no_integer_solution_l1597_159788

theorem no_integer_solution : ¬ ∃ (a b c : ℤ), a^2 + b^2 - 8*c = 6 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1597_159788


namespace NUMINAMATH_CALUDE_parabola_vertex_in_first_quadrant_l1597_159702

/-- Given a parabola y = -x^2 + (a+1)x + (a+2) where a > 1, 
    its vertex lies in the first quadrant -/
theorem parabola_vertex_in_first_quadrant (a : ℝ) (h : a > 1) :
  let f (x : ℝ) := -x^2 + (a+1)*x + (a+2)
  let vertex_x := (a+1)/2
  let vertex_y := f vertex_x
  vertex_x > 0 ∧ vertex_y > 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_in_first_quadrant_l1597_159702


namespace NUMINAMATH_CALUDE_prob_win_5_eq_prob_win_total_eq_l1597_159777

/-- Probability of Team A winning a single game -/
def p : ℝ := 0.6

/-- Probability of Team B winning a single game -/
def q : ℝ := 1 - p

/-- Number of games in the series -/
def n : ℕ := 7

/-- Number of games needed to win the series -/
def k : ℕ := 4

/-- Probability of Team A winning the championship after exactly 5 games -/
def prob_win_5 : ℝ := Nat.choose 4 3 * p^4 * q

/-- Probability of Team A winning the championship -/
def prob_win_total : ℝ := 
  p^4 + Nat.choose 4 3 * p^4 * q + Nat.choose 5 3 * p^4 * q^2 + Nat.choose 6 3 * p^4 * q^3

/-- Theorem stating the probability of Team A winning after exactly 5 games -/
theorem prob_win_5_eq : prob_win_5 = 0.20736 := by sorry

/-- Theorem stating the overall probability of Team A winning the championship -/
theorem prob_win_total_eq : prob_win_total = 0.710208 := by sorry

end NUMINAMATH_CALUDE_prob_win_5_eq_prob_win_total_eq_l1597_159777


namespace NUMINAMATH_CALUDE_x_power_4374_minus_reciprocal_l1597_159792

theorem x_power_4374_minus_reciprocal (x : ℂ) : 
  x - (1 / x) = -Complex.I * Real.sqrt 6 → x^4374 - (1 / x^4374) = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_4374_minus_reciprocal_l1597_159792


namespace NUMINAMATH_CALUDE_max_value_d_l1597_159729

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_d_l1597_159729


namespace NUMINAMATH_CALUDE_min_value_theorem_l1597_159739

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) :
  2 / m + 1 / n ≥ 4 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 2 ∧ 2 / m₀ + 1 / n₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1597_159739


namespace NUMINAMATH_CALUDE_quadrilateral_k_value_l1597_159763

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral formed by two lines and the positive semi-axes -/
structure Quadrilateral where
  l₁ : Line
  l₂ : Line

/-- Predicate to check if a quadrilateral has a circumscribed circle -/
def has_circumscribed_circle (q : Quadrilateral) : Prop :=
  sorry

/-- The quadrilateral formed by the given lines and axes -/
def quad (k : ℝ) : Quadrilateral :=
  { l₁ := { a := 1, b := 3, c := -7 },
    l₂ := { a := k, b := 1, c := -2 } }

theorem quadrilateral_k_value :
  ∀ k : ℝ, has_circumscribed_circle (quad k) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_k_value_l1597_159763


namespace NUMINAMATH_CALUDE_count_convex_polygons_l1597_159723

/-- A point in the 2D plane with integer coordinates -/
structure Point :=
  (x : ℕ)
  (y : ℕ)

/-- A convex polygon with vertices as a list of points -/
structure ConvexPolygon :=
  (vertices : List Point)
  (is_convex : Bool)

/-- Function to check if a polygon contains the required three consecutive vertices -/
def has_required_vertices (p : ConvexPolygon) : Bool :=
  sorry

/-- Function to count the number of valid convex polygons -/
def count_valid_polygons : ℕ :=
  sorry

/-- The main theorem stating that the count of valid convex polygons is 77 -/
theorem count_convex_polygons :
  count_valid_polygons = 77 :=
sorry

end NUMINAMATH_CALUDE_count_convex_polygons_l1597_159723


namespace NUMINAMATH_CALUDE_cone_height_from_cylinder_l1597_159711

/-- Given a cylinder and cones with specified dimensions, prove the height of the cones. -/
theorem cone_height_from_cylinder (cylinder_radius cylinder_height cone_radius : ℝ) 
  (num_cones : ℕ) (h_cylinder_radius : cylinder_radius = 12) 
  (h_cylinder_height : cylinder_height = 10) (h_cone_radius : cone_radius = 4) 
  (h_num_cones : num_cones = 135) : 
  ∃ (cone_height : ℝ), 
    cone_height = 2 ∧ 
    (π * cylinder_radius^2 * cylinder_height = 
     num_cones * (1/3 * π * cone_radius^2 * cone_height)) := by
  sorry


end NUMINAMATH_CALUDE_cone_height_from_cylinder_l1597_159711


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l1597_159752

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric --/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℝ := sorry

/-- Checks if a polygon is inside a triangle --/
def isInside (p : Polygon) (t : Triangle) : Prop := sorry

/-- Theorem: The largest possible area of a centrally symmetric polygon inside a triangle is 2/3 of the triangle's area --/
theorem largest_centrally_symmetric_polygon_area (t : Triangle) :
  ∃ (p : Polygon), isCentrallySymmetric p ∧ isInside p t ∧
    ∀ (q : Polygon), isCentrallySymmetric q → isInside q t →
      area p ≥ area q ∧ area p = (2/3) * area (Polygon.mk [t.A, t.B, t.C]) :=
sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l1597_159752


namespace NUMINAMATH_CALUDE_balloons_given_to_sandy_l1597_159710

def initial_red_balloons : ℕ := 31
def remaining_red_balloons : ℕ := 7

theorem balloons_given_to_sandy :
  initial_red_balloons - remaining_red_balloons = 24 :=
by sorry

end NUMINAMATH_CALUDE_balloons_given_to_sandy_l1597_159710


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l1597_159783

def total_people : ℕ := 10
def math_majors : ℕ := 5
def physics_majors : ℕ := 3
def chemistry_majors : ℕ := 2

theorem math_majors_consecutive_probability :
  let total_arrangements := Nat.choose total_people math_majors
  let consecutive_arrangements := total_people
  (consecutive_arrangements : ℚ) / total_arrangements = 5 / 126 := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l1597_159783


namespace NUMINAMATH_CALUDE_liquid_depth_inverted_cone_l1597_159786

/-- Represents a right circular cone. -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the liquid in the cone. -/
structure Liquid where
  depthPointDown : ℝ
  depthPointUp : ℝ

/-- Theorem stating the relationship between cone dimensions, liquid depth, and the expression m - n∛p. -/
theorem liquid_depth_inverted_cone (c : Cone) (l : Liquid) 
  (h_height : c.height = 12)
  (h_radius : c.baseRadius = 5)
  (h_depth_down : l.depthPointDown = 9)
  (h_p_cube_free : ∀ (q : ℕ), q > 1 → ¬(q ^ 3 ∣ 37)) :
  ∃ (m n : ℕ), m = 12 ∧ n = 3 ∧ l.depthPointUp = m - n * (37 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_liquid_depth_inverted_cone_l1597_159786


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l1597_159728

theorem correct_mark_calculation (n : ℕ) (initial_avg : ℚ) (wrong_mark : ℚ) (correct_avg : ℚ) :
  n = 10 →
  initial_avg = 100 →
  wrong_mark = 90 →
  correct_avg = 92 →
  ∃ x : ℚ, (n : ℚ) * initial_avg - wrong_mark + x = (n : ℚ) * correct_avg ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l1597_159728


namespace NUMINAMATH_CALUDE_uncool_parents_count_l1597_159716

theorem uncool_parents_count (total_students cool_dads cool_moms both_cool : ℕ) 
  (h1 : total_students = 35)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 11) :
  total_students - (cool_dads + cool_moms - both_cool) = 8 := by
sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l1597_159716


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l1597_159771

theorem average_of_five_numbers (x : ℝ) : 
  (3 + 5 + 6 + 8 + x) / 5 = 7 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l1597_159771


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l1597_159762

/-- Given that B is a digit in base 5 and b is a base greater than 6,
    such that BBB₅ = 44ᵦ, prove that the smallest possible sum of B + b is 8. -/
theorem smallest_sum_B_plus_b : ∃ (B b : ℕ),
  (0 < B) ∧ (B < 5) ∧  -- B is a digit in base 5
  (b > 6) ∧            -- b is a base greater than 6
  (31 * B = 4 * b + 4) ∧  -- BBB₅ = 44ᵦ
  (∀ (B' b' : ℕ), 
    (0 < B') ∧ (B' < 5) ∧ (b' > 6) ∧ (31 * B' = 4 * b' + 4) →
    B + b ≤ B' + b') :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l1597_159762


namespace NUMINAMATH_CALUDE_max_guaranteed_sum_l1597_159724

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents a sign that can be placed -/
inductive Sign
| Pos
| Neg

/-- Represents the state of the game -/
structure GameState where
  numbers : List ℕ
  signs : List Sign

/-- The strategy function type -/
def Strategy := GameState → Sign

/-- The result of playing the game -/
def play_game (stratA stratB : Strategy) : ℤ := sorry

/-- The theorem stating the maximum guaranteed sum for Player B -/
theorem max_guaranteed_sum :
  ∃ (stratB : Strategy), ∀ (stratA : Strategy),
    |play_game stratA stratB| ≥ 30 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_sum_l1597_159724


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l1597_159790

theorem complex_equation_solutions : 
  ∃ (S : Finset ℂ), (∀ z ∈ S, Complex.abs z < 24 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧ 
                    Finset.card S = 8 ∧
                    ∀ z, Complex.abs z < 24 → Complex.exp z = (z - 2) / (z + 2) → z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l1597_159790


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1597_159726

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 65 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1597_159726


namespace NUMINAMATH_CALUDE_pudong_exemplifies_ideal_pattern_l1597_159748

-- Define the characteristics of city cluster development
structure CityClusterDevelopment where
  aggregation : Bool
  radiation : Bool
  mutualInfluence : Bool

-- Define the development pattern of Pudong, Shanghai
def pudongDevelopment : CityClusterDevelopment :=
  { aggregation := true,
    radiation := true,
    mutualInfluence := true }

-- Define the ideal world city cluster development pattern
def idealCityClusterPattern : CityClusterDevelopment :=
  { aggregation := true,
    radiation := true,
    mutualInfluence := true }

-- Theorem statement
theorem pudong_exemplifies_ideal_pattern :
  pudongDevelopment = idealCityClusterPattern :=
by sorry

end NUMINAMATH_CALUDE_pudong_exemplifies_ideal_pattern_l1597_159748


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1597_159787

theorem fraction_irreducible (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1597_159787


namespace NUMINAMATH_CALUDE_f_properties_l1597_159795

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3

-- State the theorem
theorem f_properties (a : ℝ) (h_a_pos : a > 0) :
  (∀ x ≥ 3, f a x ≥ 0) → a ≥ 1 ∧
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  ∃ s : ℝ, 2 < s ∧ s < 4 ∧ ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁^2 + x₂^2 = s :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1597_159795


namespace NUMINAMATH_CALUDE_trig_identity_l1597_159753

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1597_159753


namespace NUMINAMATH_CALUDE_special_function_inequality_l1597_159775

/-- A non-negative differentiable function satisfying certain conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  non_negative : ∀ x ∈ domain, f x ≥ 0
  differentiable : DifferentiableOn ℝ f domain
  condition : ∀ x ∈ domain, x * (deriv f x) + f x ≤ 0

/-- Theorem statement -/
theorem special_function_inequality (φ : SpecialFunction) (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a < b) :
    b * φ.f b ≤ a * φ.f a := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l1597_159775


namespace NUMINAMATH_CALUDE_count_prime_pairs_sum_50_l1597_159774

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def primePairSum50 (p q : ℕ) : Prop := isPrime p ∧ isPrime q ∧ p + q = 50

theorem count_prime_pairs_sum_50 : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = count ∧ 
    (∀ (p q : ℕ), (p, q) ∈ pairs ↔ primePairSum50 p q ∧ p ≤ q) ∧
    count = 4 :=
sorry

end NUMINAMATH_CALUDE_count_prime_pairs_sum_50_l1597_159774


namespace NUMINAMATH_CALUDE_circle_tangent_radius_l1597_159730

/-- The radius of a circle tangent to another circle and a line -/
theorem circle_tangent_radius (P : ℝ × ℝ) (R : ℝ) : 
  let A : ℝ × ℝ := (3, 1)
  let M : ℝ × ℝ := (0, 2)
  let l (x : ℝ) := -3/4 * x - 11/4
  -- Circle ⊙P passes through point A(3,1)
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = R^2 →
  -- Circle ⊙P is tangent to circle ⊙M: x² + (y-2)² = 4
  (P.1 - M.1)^2 + (P.2 - M.2)^2 = (R + 2)^2 ∨ (P.1 - M.1)^2 + (P.2 - M.2)^2 = (R - 2)^2 →
  -- Circle ⊙P is tangent to line l: y = -3/4x - 11/4
  (3 * P.1 + 4 * P.2 + 11) / 5 = R →
  -- The radius of circle ⊙P is 3
  R = 3 := by
sorry


end NUMINAMATH_CALUDE_circle_tangent_radius_l1597_159730


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1597_159700

theorem absolute_value_equation (x : ℝ) : 
  |(-2 : ℝ)| * (|(-25 : ℝ)| - |x|) = 40 ↔ |x| = 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1597_159700


namespace NUMINAMATH_CALUDE_gervais_driving_days_l1597_159778

theorem gervais_driving_days :
  let gervais_avg_miles_per_day : ℝ := 315
  let henri_total_miles : ℝ := 1250
  let difference_in_miles : ℝ := 305
  let gervais_days : ℝ := (henri_total_miles - difference_in_miles) / gervais_avg_miles_per_day
  gervais_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_gervais_driving_days_l1597_159778


namespace NUMINAMATH_CALUDE_batsman_average_l1597_159799

/-- Calculates the average runs for a batsman given two sets of matches --/
def average_runs (matches1 : ℕ) (average1 : ℕ) (matches2 : ℕ) (average2 : ℕ) : ℚ :=
  let total_runs := matches1 * average1 + matches2 * average2
  let total_matches := matches1 + matches2
  (total_runs : ℚ) / total_matches

/-- Proves that the average runs for 45 matches is 42 given the specified conditions --/
theorem batsman_average :
  average_runs 30 50 15 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l1597_159799


namespace NUMINAMATH_CALUDE_triangle_theorem_l1597_159744

/-- Triangle ABC with sides a, b, c opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a + t.b + t.c) * (t.a - t.b + t.c) = t.a * t.c)
  (h2 : Real.sin t.A * Real.sin t.C = (Real.sqrt 3 - 1) / 4) :
  t.B = 2 * Real.pi / 3 ∧ (t.C = Real.pi / 12 ∨ t.C = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1597_159744


namespace NUMINAMATH_CALUDE_intersection_circle_passes_through_zero_one_l1597_159746

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure TripleIntersectingParabola where
  a : ℝ
  b : ℝ
  distinct_intersections : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0 ∧ b ≠ 0

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersection_circle (p : TripleIntersectingParabola) : Set (ℝ × ℝ) :=
  { point | ∃ (center : ℝ × ℝ) (radius : ℝ),
    (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 ∧
    (0 - center.1)^2 + (p.b - center.2)^2 = radius^2 ∧
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    x₁^2 + p.a*x₁ + p.b = 0 ∧ x₂^2 + p.a*x₂ + p.b = 0 ∧
    (x₁ - center.1)^2 + (0 - center.2)^2 = radius^2 ∧
    (x₂ - center.1)^2 + (0 - center.2)^2 = radius^2 }

/-- The main theorem stating that the intersection circle passes through (0,1) -/
theorem intersection_circle_passes_through_zero_one (p : TripleIntersectingParabola) :
  (0, 1) ∈ intersection_circle p := by
  sorry

end NUMINAMATH_CALUDE_intersection_circle_passes_through_zero_one_l1597_159746


namespace NUMINAMATH_CALUDE_sequence_comparison_l1597_159708

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, b (n + 1) = b n * q

/-- All terms of the sequence are positive -/
def all_positive (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0

theorem sequence_comparison
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (hpos : all_positive b)
  (h1 : a 1 = b 1)
  (h11 : a 11 = b 11) :
  a 6 > b 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_comparison_l1597_159708


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l1597_159707

theorem solution_set_implies_a_range (a : ℝ) :
  (∀ x, (a - 3) * x > 1 ↔ x < 1 / (a - 3)) →
  a < 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l1597_159707


namespace NUMINAMATH_CALUDE_largest_number_l1597_159703

theorem largest_number (a b c d e : ℝ) : 
  a = 13579 + 1 / 2468 →
  b = 13579 - 1 / 2468 →
  c = 13579 * (1 / 2468) →
  d = 13579 / (1 / 2468) →
  e = 13579.2468 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1597_159703


namespace NUMINAMATH_CALUDE_milk_for_six_cookies_l1597_159772

/-- The number of cookies that can be baked with 1 gallon of milk -/
def cookies_per_gallon : ℕ := 24

/-- The number of quarts in a gallon -/
def quarts_per_gallon : ℕ := 4

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 6

/-- Calculate the amount of milk in quarts needed to bake a given number of cookies -/
def milk_needed (cookies : ℕ) : ℚ :=
  (cookies : ℚ) * (quarts_per_gallon : ℚ) / (cookies_per_gallon : ℚ)

/-- Theorem: The amount of milk needed to bake 6 cookies is 1 quart -/
theorem milk_for_six_cookies :
  milk_needed target_cookies = 1 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_six_cookies_l1597_159772


namespace NUMINAMATH_CALUDE_distribution_four_to_three_l1597_159720

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distributionCount (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribution_four_to_three :
  distributionCount 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribution_four_to_three_l1597_159720


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1597_159761

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, 2^n < 1000) ↔ (∀ n : ℕ, 2^n ≥ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1597_159761


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_l1597_159780

/-- The definition of a repeating decimal 0.3333... -/
def repeating_third : ℚ := 1/3

/-- Proof that 1 - 0.3333... = 2/3 -/
theorem one_minus_repeating_third :
  1 - repeating_third = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_l1597_159780


namespace NUMINAMATH_CALUDE_correct_division_l1597_159784

theorem correct_division (x : ℤ) : x + 4 = 40 → x / 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l1597_159784


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1597_159743

theorem quadratic_equation_root (k l m : ℝ) :
  (2 * (k - l) * 2^2 + 3 * (l - m) * 2 + 4 * (m - k) = 0) →
  (∃ x : ℝ, 2 * (k - l) * x^2 + 3 * (l - m) * x + 4 * (m - k) = 0 ∧ x ≠ 2) →
  (∃ x : ℝ, 2 * (k - l) * x^2 + 3 * (l - m) * x + 4 * (m - k) = 0 ∧ x = (m - k) / (k - l)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1597_159743


namespace NUMINAMATH_CALUDE_remaining_fruit_cost_is_eight_l1597_159796

/-- Represents the cost of fruit remaining in Tanya's bag after half fell out --/
def remaining_fruit_cost (pear_count : ℕ) (pear_price : ℚ) 
                         (apple_count : ℕ) (apple_price : ℚ)
                         (pineapple_count : ℕ) (pineapple_price : ℚ) : ℚ :=
  ((pear_count : ℚ) * pear_price + 
   (apple_count : ℚ) * apple_price + 
   (pineapple_count : ℚ) * pineapple_price) / 2

/-- Theorem stating the cost of remaining fruit excluding plums --/
theorem remaining_fruit_cost_is_eight :
  remaining_fruit_cost 6 1.5 4 0.75 2 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fruit_cost_is_eight_l1597_159796


namespace NUMINAMATH_CALUDE_expression_evaluation_l1597_159789

theorem expression_evaluation : 2197 + 180 / 60 * 3 - 197 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1597_159789


namespace NUMINAMATH_CALUDE_parking_cost_theorem_l1597_159733

/-- The number of hours covered by the initial parking cost -/
def initial_hours : ℝ := 2

/-- The initial parking cost -/
def initial_cost : ℝ := 10

/-- The cost per additional hour -/
def additional_hour_cost : ℝ := 1.75

/-- The total number of hours parked -/
def total_hours : ℝ := 9

/-- The average cost per hour for the total parking time -/
def average_cost_per_hour : ℝ := 2.4722222222222223

theorem parking_cost_theorem :
  initial_hours * initial_cost + (total_hours - initial_hours) * additional_hour_cost =
  total_hours * average_cost_per_hour := by
  sorry

end NUMINAMATH_CALUDE_parking_cost_theorem_l1597_159733


namespace NUMINAMATH_CALUDE_original_class_size_l1597_159760

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ N : ℕ,
    N * original_avg + new_students * new_avg = (N + new_students) * (original_avg - avg_decrease) ∧
    N = 12 :=
by sorry

end NUMINAMATH_CALUDE_original_class_size_l1597_159760


namespace NUMINAMATH_CALUDE_fraction_value_l1597_159750

theorem fraction_value (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ (n : ℤ), x / y = ↑n) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1597_159750


namespace NUMINAMATH_CALUDE_arithmetic_equation_l1597_159709

theorem arithmetic_equation : 6 + 18 / 3 - 4^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l1597_159709


namespace NUMINAMATH_CALUDE_inverse_sum_bound_l1597_159781

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- State the theorem
theorem inverse_sum_bound 
  (k : ℝ) (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < 2)
  (h4 : f k α = 0) (h5 : f k β = 0) :
  1/α + 1/β < 4 :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_bound_l1597_159781


namespace NUMINAMATH_CALUDE_credit_card_balance_proof_l1597_159718

def calculate_final_balance (initial_balance : ℝ)
  (month1_interest : ℝ)
  (month2_charges : ℝ) (month2_interest : ℝ)
  (month3_charges : ℝ) (month3_payment : ℝ) (month3_interest : ℝ)
  (month4_charges : ℝ) (month4_payment : ℝ) (month4_interest : ℝ) : ℝ :=
  let balance1 := initial_balance * (1 + month1_interest)
  let balance2 := (balance1 + month2_charges) * (1 + month2_interest)
  let balance3 := ((balance2 + month3_charges) - month3_payment) * (1 + month3_interest)
  let balance4 := ((balance3 + month4_charges) - month4_payment) * (1 + month4_interest)
  balance4

theorem credit_card_balance_proof :
  calculate_final_balance 50 0.2 20 0.2 30 10 0.25 40 20 0.15 = 189.75 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_proof_l1597_159718


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1597_159742

/-- Represents the number of items of each product type in a sample -/
structure SampleCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total sample size -/
def totalSampleSize (s : SampleCounts) : ℕ :=
  s.typeA + s.typeB + s.typeC

/-- Represents the production ratio of the three product types -/
def productionRatio : Fin 3 → ℕ
| 0 => 1  -- Type A
| 1 => 3  -- Type B
| 2 => 5  -- Type C

theorem stratified_sample_size 
  (s : SampleCounts) 
  (h_ratio : s.typeA * productionRatio 1 = s.typeB * productionRatio 0 ∧ 
             s.typeB * productionRatio 2 = s.typeC * productionRatio 1) 
  (h_typeB : s.typeB = 12) : 
  totalSampleSize s = 36 := by
sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l1597_159742


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l1597_159794

theorem least_n_with_gcd_conditions : ∃ (n : ℕ), 
  (n > 500) ∧ 
  (Nat.gcd 70 (n + 150) = 35) ∧ 
  (Nat.gcd (n + 70) 150 = 50) ∧ 
  (∀ m : ℕ, m > 500 → Nat.gcd 70 (m + 150) = 35 → Nat.gcd (m + 70) 150 = 50 → m ≥ n) ∧
  n = 1015 := by
sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l1597_159794


namespace NUMINAMATH_CALUDE_power_division_rule_l1597_159779

theorem power_division_rule (a : ℝ) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1597_159779


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l1597_159776

theorem football_team_right_handed_players (total_players : ℕ) (throwers : ℕ) :
  total_players = 70 →
  throwers = 37 →
  (total_players - throwers) % 3 = 0 →
  59 = throwers + (total_players - throwers) * 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l1597_159776


namespace NUMINAMATH_CALUDE_rice_containers_l1597_159745

theorem rice_containers (total_weight : ℚ) (container_capacity : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 25 / 2 →
  container_capacity = 50 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce : ℚ) / container_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_containers_l1597_159745


namespace NUMINAMATH_CALUDE_sum_ratio_l1597_159732

def sean_sum : ℕ → ℕ
| 0 => 0
| (n + 1) => sean_sum n + if (n + 1) * 3 ≤ 600 then (n + 1) * 3 else 0

def julie_sum : ℕ → ℕ
| 0 => 0
| (n + 1) => julie_sum n + if n + 1 ≤ 300 then n + 1 else 0

theorem sum_ratio :
  (sean_sum 200 : ℚ) / (julie_sum 300 : ℚ) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_ratio_l1597_159732


namespace NUMINAMATH_CALUDE_remainder_problem_l1597_159767

theorem remainder_problem (x y : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + y^2) % 41 = 0) :
  (x + y^3 + 7) % 61 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1597_159767


namespace NUMINAMATH_CALUDE_instrument_probability_l1597_159798

/-- The probability of selecting a cello and a viola made from the same tree -/
theorem instrument_probability (total_cellos : ℕ) (total_violas : ℕ) (same_tree_pairs : ℕ) :
  total_cellos = 800 →
  total_violas = 600 →
  same_tree_pairs = 100 →
  (same_tree_pairs : ℚ) / (total_cellos * total_violas) = 1 / 4800 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l1597_159798


namespace NUMINAMATH_CALUDE_remainder_of_M_divided_by_500_l1597_159740

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (λ acc i => acc * factorial (i + 1)) 1

def M : ℕ := (product_of_factorials.digits 10).reverse.takeWhile (·= 0) |>.length

theorem remainder_of_M_divided_by_500 : M % 500 = 391 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_M_divided_by_500_l1597_159740


namespace NUMINAMATH_CALUDE_inequality_proof_l1597_159770

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1597_159770


namespace NUMINAMATH_CALUDE_solutions_nonempty_and_finite_l1597_159785

def solution_set (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {(x, y, z) | Real.sqrt ((x^2 : ℝ) + y + n) + Real.sqrt ((y^2 : ℝ) + x + n) = z}

theorem solutions_nonempty_and_finite (n : ℕ) :
  (solution_set n).Nonempty ∧ (solution_set n).Finite :=
sorry

end NUMINAMATH_CALUDE_solutions_nonempty_and_finite_l1597_159785


namespace NUMINAMATH_CALUDE_chess_competition_participants_l1597_159793

def is_valid_num_high_school_students (n : ℕ) : Prop :=
  let total_players := n + 2
  let total_games := total_players * (total_players - 1) / 2
  let remaining_points := total_games - 8
  remaining_points % n = 0

theorem chess_competition_participants : 
  ∀ n : ℕ, n > 2 → (is_valid_num_high_school_students n ↔ n = 7 ∨ n = 14) :=
by sorry

end NUMINAMATH_CALUDE_chess_competition_participants_l1597_159793


namespace NUMINAMATH_CALUDE_m_eq_2_necessary_not_sufficient_l1597_159701

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_eq_2_necessary_not_sufficient :
  (∀ m : ℝ, A m ∩ B = {4} → m = 2 ∨ m = -2) ∧
  (∃ m : ℝ, m = 2 ∧ A m ∩ B = {4}) ∧
  (∃ m : ℝ, m = -2 ∧ A m ∩ B = {4}) :=
sorry

end NUMINAMATH_CALUDE_m_eq_2_necessary_not_sufficient_l1597_159701


namespace NUMINAMATH_CALUDE_beach_attendance_l1597_159712

theorem beach_attendance (initial_group : ℕ) (joined : ℕ) (left : ℕ) : 
  initial_group = 3 → joined = 100 → left = 40 → 
  initial_group + joined - left = 63 := by
  sorry

end NUMINAMATH_CALUDE_beach_attendance_l1597_159712


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1597_159749

/-- Given real number m and vectors a and b in ℝ², prove that if a ⊥ b, then |a + b| = √34 -/
theorem vector_sum_magnitude (m : ℝ) (a b : ℝ × ℝ) : 
  a = (m + 2, 1) → 
  b = (1, -2*m) → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_vector_sum_magnitude_l1597_159749


namespace NUMINAMATH_CALUDE_village_population_l1597_159766

theorem village_population (P : ℝ) 
  (h1 : 0.9 * P * 0.8 = 4500) : P = 6250 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1597_159766


namespace NUMINAMATH_CALUDE_midpoint_chain_l1597_159758

theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 4) →        -- AG = 4
  (B - A = 128) :=     -- AB = 128
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1597_159758


namespace NUMINAMATH_CALUDE_min_milk_candies_l1597_159706

/-- Represents the number of chocolate candies -/
def chocolate : ℕ := sorry

/-- Represents the number of watermelon candies -/
def watermelon : ℕ := sorry

/-- Represents the number of milk candies -/
def milk : ℕ := sorry

/-- The number of watermelon candies is at most 3 times the number of chocolate candies -/
axiom watermelon_condition : watermelon ≤ 3 * chocolate

/-- The number of milk candies is at least 4 times the number of chocolate candies -/
axiom milk_condition : milk ≥ 4 * chocolate

/-- The total number of chocolate and watermelon candies is no less than 2020 -/
axiom total_condition : chocolate + watermelon ≥ 2020

/-- The minimum number of milk candies required is 2020 -/
theorem min_milk_candies : milk ≥ 2020 := by sorry

end NUMINAMATH_CALUDE_min_milk_candies_l1597_159706


namespace NUMINAMATH_CALUDE_exam_scores_theorem_l1597_159756

/-- A type representing a student's scores in three tasks -/
structure StudentScores :=
  (task1 : Nat)
  (task2 : Nat)
  (task3 : Nat)

/-- A predicate that checks if all scores are between 0 and 7 -/
def validScores (s : StudentScores) : Prop :=
  0 ≤ s.task1 ∧ s.task1 ≤ 7 ∧
  0 ≤ s.task2 ∧ s.task2 ≤ 7 ∧
  0 ≤ s.task3 ∧ s.task3 ≤ 7

/-- A predicate that checks if one student's scores are greater than or equal to another's -/
def scoresGreaterOrEqual (s1 s2 : StudentScores) : Prop :=
  s1.task1 ≥ s2.task1 ∧ s1.task2 ≥ s2.task2 ∧ s1.task3 ≥ s2.task3

/-- The main theorem to be proved -/
theorem exam_scores_theorem (students : Finset StudentScores) 
    (h : students.card = 49)
    (h_valid : ∀ s ∈ students, validScores s) :
  ∃ s1 s2 : StudentScores, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ scoresGreaterOrEqual s1 s2 :=
sorry

end NUMINAMATH_CALUDE_exam_scores_theorem_l1597_159756


namespace NUMINAMATH_CALUDE_odd_function_zeros_and_equation_root_l1597_159704

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_zeros_and_equation_root (f : ℝ → ℝ) (zeros : Finset ℝ) :
  isOdd f →
  zeros.card = 2017 →
  (∀ x ∈ zeros, f x = 0) →
  ∃ r ∈ Set.Ioo 0 1, 2^r + r - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_zeros_and_equation_root_l1597_159704


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1597_159754

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬p x) := by sorry

theorem negation_of_proposition : 
  (¬∃ x₀ : ℝ, (2 : ℝ)^x₀ ≠ 1) ↔ (∀ x₀ : ℝ, (2 : ℝ)^x₀ = 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1597_159754


namespace NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l1597_159768

def is_increasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n < a (n + 1)

def sequence_a (c : ℝ) (n : ℕ+) : ℝ :=
  |n.val - c|

theorem c_leq_one_sufficient_not_necessary (c : ℝ) :
  (c ≤ 1 → is_increasing (sequence_a c)) ∧
  ¬(is_increasing (sequence_a c) → c ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l1597_159768


namespace NUMINAMATH_CALUDE_count_divisible_by_11_is_18_l1597_159751

/-- The number obtained by writing the integers 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of b_k divisible by 11 for 1 ≤ k ≤ 100 -/
def count_divisible_by_11 : ℕ := sorry

theorem count_divisible_by_11_is_18 : count_divisible_by_11 = 18 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_is_18_l1597_159751
