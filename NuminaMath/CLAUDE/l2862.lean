import Mathlib

namespace NUMINAMATH_CALUDE_unpainted_face_area_l2862_286270

/-- A right circular cylinder with given dimensions -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The unpainted face created by slicing the cylinder -/
def UnpaintedFace (c : Cylinder) (arcAngle : ℝ) : ℝ := sorry

/-- Theorem stating the area of the unpainted face for the given cylinder and arc angle -/
theorem unpainted_face_area (c : Cylinder) (h1 : c.radius = 6) (h2 : c.height = 8) (h3 : arcAngle = 2 * π / 3) :
  UnpaintedFace c arcAngle = 16 * π + 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_face_area_l2862_286270


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2862_286241

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -4.5 < x ∧ x < 3.5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2862_286241


namespace NUMINAMATH_CALUDE_binomial_5_choose_3_l2862_286214

theorem binomial_5_choose_3 : Nat.choose 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_5_choose_3_l2862_286214


namespace NUMINAMATH_CALUDE_pipe_length_difference_l2862_286273

theorem pipe_length_difference (total_length shorter_length : ℕ) 
  (h1 : total_length = 68)
  (h2 : shorter_length = 28)
  (h3 : shorter_length < total_length - shorter_length) :
  total_length - shorter_length - shorter_length = 12 :=
by sorry

end NUMINAMATH_CALUDE_pipe_length_difference_l2862_286273


namespace NUMINAMATH_CALUDE_fifth_day_income_correct_l2862_286249

/-- Calculates the income for the fifth day given the income for four days and the average income for five days. -/
def fifth_day_income (day1 day2 day3 day4 average : ℝ) : ℝ :=
  5 * average - (day1 + day2 + day3 + day4)

/-- Theorem stating that the calculated fifth day income is correct. -/
theorem fifth_day_income_correct (day1 day2 day3 day4 day5 average : ℝ) 
  (h_average : average = (day1 + day2 + day3 + day4 + day5) / 5) :
  fifth_day_income day1 day2 day3 day4 average = day5 := by
  sorry

#eval fifth_day_income 250 400 750 400 460

end NUMINAMATH_CALUDE_fifth_day_income_correct_l2862_286249


namespace NUMINAMATH_CALUDE_wendy_packaging_theorem_l2862_286225

/-- Represents the number of chocolates Wendy can package in a given time -/
def chocolates_packaged (packaging_rate : ℕ) (packaging_time : ℕ) (work_time : ℕ) : ℕ :=
  (packaging_rate * 12 * (work_time * 60 / packaging_time))

/-- Proves that Wendy can package 1152 chocolates in 4 hours -/
theorem wendy_packaging_theorem :
  chocolates_packaged 2 5 240 = 1152 := by
  sorry

#eval chocolates_packaged 2 5 240

end NUMINAMATH_CALUDE_wendy_packaging_theorem_l2862_286225


namespace NUMINAMATH_CALUDE_locus_of_M_l2862_286298

-- Define the constant k
variable (k : ℝ)

-- Define the coordinates of points A and B
variable (xA yA xB yB : ℝ)

-- Define the coordinates of point M
variable (xM yM : ℝ)

-- Axioms based on the problem conditions
axiom perpendicular_axes : xA * xB + yA * yB = 0
axiom A_on_x_axis : yA = 0
axiom B_on_y_axis : xB = 0
axiom sum_of_distances : Real.sqrt (xA^2 + yA^2) + Real.sqrt (xB^2 + yB^2) = k
axiom M_on_circumcircle : (xM - xA)^2 + (yM - yA)^2 = (xM - xB)^2 + (yM - yB)^2

-- Theorem statement
theorem locus_of_M : 
  (xM - k/2)^2 + (yM - k/2)^2 = k^2/2 := by sorry

end NUMINAMATH_CALUDE_locus_of_M_l2862_286298


namespace NUMINAMATH_CALUDE_solve_equation_l2862_286289

theorem solve_equation : ∃ x : ℝ, 8 * x - (5 * 0.85 / 2.5) = 5.5 ∧ x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2862_286289


namespace NUMINAMATH_CALUDE_jug_problem_l2862_286281

theorem jug_problem (Cx Cy : ℝ) (h1 : Cx > 0) (h2 : Cy > 0) : 
  (1/6 : ℝ) * Cx = (2/3 : ℝ) * Cy → 
  (1/9 : ℝ) * Cx = (1/3 : ℝ) * Cy - (1/3 : ℝ) * Cy := by
sorry

end NUMINAMATH_CALUDE_jug_problem_l2862_286281


namespace NUMINAMATH_CALUDE_probability_ratio_l2862_286226

def total_slips : ℕ := 40
def distinct_numbers : ℕ := 8
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 4

def probability_same_number (n : ℕ) : ℚ :=
  (n * slips_per_number.choose drawn_slips) / total_slips.choose drawn_slips

def probability_two_pairs (n : ℕ) : ℚ :=
  (n.choose 2 * slips_per_number.choose 2 * slips_per_number.choose 2) / total_slips.choose drawn_slips

theorem probability_ratio :
  probability_two_pairs distinct_numbers / probability_same_number distinct_numbers = 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l2862_286226


namespace NUMINAMATH_CALUDE_button_collection_value_l2862_286235

theorem button_collection_value (total_buttons : ℕ) (sample_buttons : ℕ) (sample_value : ℚ) :
  total_buttons = 10 →
  sample_buttons = 2 →
  sample_value = 8 →
  (sample_value / sample_buttons) * total_buttons = 40 := by
sorry

end NUMINAMATH_CALUDE_button_collection_value_l2862_286235


namespace NUMINAMATH_CALUDE_second_fragment_speed_is_52_l2862_286284

/-- Represents the motion of a firecracker that explodes into two fragments -/
structure Firecracker where
  initial_speed : ℝ
  explosion_time : ℝ
  gravity : ℝ
  first_fragment_horizontal_speed : ℝ

/-- Calculates the speed of the second fragment after explosion -/
def second_fragment_speed (f : Firecracker) : ℝ :=
  sorry

/-- Theorem stating that the speed of the second fragment is 52 m/s -/
theorem second_fragment_speed_is_52 (f : Firecracker) 
  (h1 : f.initial_speed = 20)
  (h2 : f.explosion_time = 3)
  (h3 : f.gravity = 10)
  (h4 : f.first_fragment_horizontal_speed = 48) :
  second_fragment_speed f = 52 :=
  sorry

end NUMINAMATH_CALUDE_second_fragment_speed_is_52_l2862_286284


namespace NUMINAMATH_CALUDE_thirteen_points_guarantee_win_thirteen_smallest_guarantee_l2862_286253

/-- Represents the points awarded for each position in a race -/
def race_points : Fin 3 → ℕ
  | 0 => 5  -- First place
  | 1 => 3  -- Second place
  | 2 => 1  -- Third place
  | _ => 0  -- This case should never occur due to Fin 3

/-- The total number of races -/
def num_races : ℕ := 3

/-- A function to calculate the maximum points possible for the second-place student -/
def max_second_place_points : ℕ := sorry

/-- Theorem stating that 13 points guarantees more points than any other student -/
theorem thirteen_points_guarantee_win :
  ∀ (student_points : ℕ),
    student_points ≥ 13 →
    student_points > max_second_place_points :=
  sorry

/-- Theorem stating that 13 is the smallest number of points that guarantees a win -/
theorem thirteen_smallest_guarantee :
  ∀ (n : ℕ),
    n < 13 →
    ∃ (other_points : ℕ),
      other_points ≥ n ∧
      other_points ≤ max_second_place_points :=
  sorry

end NUMINAMATH_CALUDE_thirteen_points_guarantee_win_thirteen_smallest_guarantee_l2862_286253


namespace NUMINAMATH_CALUDE_bug_probability_after_12_meters_l2862_286294

/-- Probability of the bug being at vertex A after crawling n meters -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - P n) / 3

/-- Edge length of the tetrahedron in meters -/
def edgeLength : ℕ := 2

/-- Number of edges traversed after 12 meters -/
def edgesTraversed : ℕ := 12 / edgeLength

theorem bug_probability_after_12_meters :
  P edgesTraversed = 44287 / 177147 := by sorry

end NUMINAMATH_CALUDE_bug_probability_after_12_meters_l2862_286294


namespace NUMINAMATH_CALUDE_line_2x_plus_1_not_in_fourth_quadrant_l2862_286227

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Defines the fourth quadrant of the 2D plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Checks if a given line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.y_intercept ∧ fourth_quadrant x y

/-- The main theorem stating that the line y = 2x + 1 does not pass through the fourth quadrant -/
theorem line_2x_plus_1_not_in_fourth_quadrant :
  ¬ passes_through_fourth_quadrant (Line.mk 2 1) := by
  sorry


end NUMINAMATH_CALUDE_line_2x_plus_1_not_in_fourth_quadrant_l2862_286227


namespace NUMINAMATH_CALUDE_positive_root_k_values_negative_solution_k_range_l2862_286293

-- Define the equation
def equation (x k : ℝ) : Prop :=
  4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)

-- Part 1: Positive root case
theorem positive_root_k_values (k : ℝ) :
  (∃ x > 0, equation x k) → (k = 6 ∨ k = -8) := by sorry

-- Part 2: Negative solution case
theorem negative_solution_k_range (k : ℝ) :
  (∃ x < 0, equation x k) → (k < -1 ∧ k ≠ -8) := by sorry

end NUMINAMATH_CALUDE_positive_root_k_values_negative_solution_k_range_l2862_286293


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2862_286278

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis : 
  let P : Point2D := { x := -2, y := 5 }
  reflect_x P = { x := -2, y := -5 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2862_286278


namespace NUMINAMATH_CALUDE_cell_phone_company_customers_l2862_286236

theorem cell_phone_company_customers (us_customers other_customers : ℕ) 
  (h1 : us_customers = 723)
  (h2 : other_customers = 6699) :
  us_customers + other_customers = 7422 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_company_customers_l2862_286236


namespace NUMINAMATH_CALUDE_conditional_probability_first_class_l2862_286252

/-- A box containing products -/
structure Box where
  total : Nat
  firstClass : Nat
  secondClass : Nat
  h_sum : firstClass + secondClass = total

/-- The probability of selecting a first-class product on the second draw
    given that a first-class product was selected on the first draw -/
def conditionalProbability (b : Box) : ℚ :=
  (b.firstClass - 1 : ℚ) / (b.total - 1 : ℚ)

theorem conditional_probability_first_class
  (b : Box)
  (h_total : b.total = 4)
  (h_first : b.firstClass = 3)
  (h_second : b.secondClass = 1) :
  conditionalProbability b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_first_class_l2862_286252


namespace NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l2862_286258

theorem right_triangles_shared_hypotenuse (b : ℝ) (h : b ≥ Real.sqrt 3) :
  let BC : ℝ := 1
  let AC : ℝ := b
  let AD : ℝ := 2
  let AB : ℝ := Real.sqrt (AC^2 + BC^2)
  let BD : ℝ := Real.sqrt (AB^2 - AD^2)
  BD = Real.sqrt (b^2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l2862_286258


namespace NUMINAMATH_CALUDE_movie_ticket_ratio_l2862_286234

def horror_tickets : ℕ := 93
def romance_tickets : ℕ := 25
def ticket_difference : ℕ := 18

theorem movie_ticket_ratio :
  (horror_tickets : ℚ) / romance_tickets = 93 / 25 ∧
  horror_tickets = romance_tickets + ticket_difference :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_ratio_l2862_286234


namespace NUMINAMATH_CALUDE_line_param_values_l2862_286287

/-- The line equation y = (1/3)x + 3 parameterized as (x, y) = (-5, r) + t(m, -6) -/
def line_equation (x y : ℝ) : Prop := y = (1/3) * x + 3

/-- The parameterization of the line -/
def line_param (t r m : ℝ) (x y : ℝ) : Prop :=
  x = -5 + t * m ∧ y = r + t * (-6)

/-- Theorem stating that r = 4/3 and m = 0 for the given line and parameterization -/
theorem line_param_values :
  ∃ (r m : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ line_param t r m x y) ∧ r = 4/3 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_param_values_l2862_286287


namespace NUMINAMATH_CALUDE_max_sum_squares_l2862_286208

theorem max_sum_squares (a b c d : ℝ) 
  (h1 : a + b = 18)
  (h2 : a * b + c + d = 91)
  (h3 : a * d + b * c = 195)
  (h4 : c * d = 120) :
  a^2 + b^2 + c^2 + d^2 ≤ 82 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 18 ∧
    a' * b' + c' + d' = 91 ∧
    a' * d' + b' * c' = 195 ∧
    c' * d' = 120 ∧
    a'^2 + b'^2 + c'^2 + d'^2 = 82 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l2862_286208


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_225pieces_l2862_286259

/-- Represents a square board with side length and number of pieces it's cut into -/
structure Board where
  side_length : ℕ
  num_pieces : ℕ

/-- Calculates the maximum possible total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  sorry

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 pieces -/
theorem max_cut_length_30x30_225pieces :
  let b : Board := { side_length := 30, num_pieces := 225 }
  max_cut_length b = 1065 := by
  sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_225pieces_l2862_286259


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2862_286222

theorem divisibility_implies_equality (a b : ℕ+) (h : (a + b) ∣ (5 * a + 3 * b)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2862_286222


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2862_286230

/-- Given that M(5,5) is the midpoint of line segment CD and C has coordinates (10,10),
    prove that the sum of the coordinates of point D is 0. -/
theorem midpoint_coordinate_sum (C D M : ℝ × ℝ) : 
  M = (5, 5) → 
  C = (10, 10) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2862_286230


namespace NUMINAMATH_CALUDE_money_left_relation_l2862_286210

/-- The relationship between money left and masks bought -/
theorem money_left_relation (initial_amount : ℝ) (mask_price : ℝ) (x : ℝ) (y : ℝ) :
  initial_amount = 60 →
  mask_price = 2 →
  y = initial_amount - mask_price * x →
  y = 60 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_money_left_relation_l2862_286210


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_18_l2862_286268

theorem consecutive_even_numbers_sum_18 (n : ℤ) : 
  (n - 2) + n + (n + 2) = 18 → (n - 2 = 4 ∧ n = 6 ∧ n + 2 = 8) := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_18_l2862_286268


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2862_286217

theorem square_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = Real.sqrt 2020) :
  x^2 + 1/x^2 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2862_286217


namespace NUMINAMATH_CALUDE_lattice_points_bound_l2862_286274

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  area : ℝ
  semiperimeter : ℝ
  lattice_points : ℕ

/-- Theorem: For any convex figure, the number of lattice points inside
    is greater than the difference between its area and semiperimeter -/
theorem lattice_points_bound (figure : ConvexFigure) :
  figure.lattice_points > figure.area - figure.semiperimeter := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_bound_l2862_286274


namespace NUMINAMATH_CALUDE_min_PM_AB_implies_line_AB_l2862_286221

/-- Circle M with equation x^2 + y^2 - 2x - 2y - 2 = 0 -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- Line l with equation 2x + y + 2 = 0 -/
def line_l (x y : ℝ) : Prop :=
  2*x + y + 2 = 0

/-- Point P on line l -/
structure Point_P where
  x : ℝ
  y : ℝ
  on_line_l : line_l x y

/-- Tangent line from P to circle M -/
def is_tangent (P : Point_P) (A : ℝ × ℝ) : Prop :=
  circle_M A.1 A.2 ∧ 
  ∃ (t : ℝ), A.1 = P.x + t * (A.1 - P.x) ∧ A.2 = P.y + t * (A.2 - P.y)

/-- The equation of line AB: 2x + y + 1 = 0 -/
def line_AB (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

theorem min_PM_AB_implies_line_AB :
  ∀ (P : Point_P) (A B : ℝ × ℝ),
  is_tangent P A → is_tangent P B →
  (∀ (Q : Point_P) (C D : ℝ × ℝ),
    is_tangent Q C → is_tangent Q D →
    (P.x - 1)^2 + (P.y - 1)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤
    (Q.x - 1)^2 + (Q.y - 1)^2 * ((C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 := by
  sorry

end NUMINAMATH_CALUDE_min_PM_AB_implies_line_AB_l2862_286221


namespace NUMINAMATH_CALUDE_opposite_numbers_cube_root_l2862_286244

theorem opposite_numbers_cube_root (x y : ℝ) : 
  y = -x → 3 * x - 4 * y = 7 → (x * y) ^ (1/3 : ℝ) = -1 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_cube_root_l2862_286244


namespace NUMINAMATH_CALUDE_cinema_visitors_l2862_286248

theorem cinema_visitors (female_visitors : ℕ) (female_office_workers : ℕ) 
  (male_excess : ℕ) (male_non_workers : ℕ) 
  (h1 : female_visitors = 1518)
  (h2 : female_office_workers = 536)
  (h3 : male_excess = 525)
  (h4 : male_non_workers = 1257) :
  female_office_workers + (female_visitors + male_excess - male_non_workers) = 1322 := by
  sorry

end NUMINAMATH_CALUDE_cinema_visitors_l2862_286248


namespace NUMINAMATH_CALUDE_stating_men_meet_at_calculated_point_l2862_286206

/-- Two men walk towards each other from points A and B, which are 90 miles apart. -/
def total_distance : ℝ := 90

/-- The speed of the man starting from point A in miles per hour. -/
def speed_a : ℝ := 5

/-- The initial speed of the man starting from point B in miles per hour. -/
def initial_speed_b : ℝ := 2

/-- The hourly increase in speed for the man starting from point B. -/
def speed_increase_b : ℝ := 1

/-- The number of hours the man from A waits before starting. -/
def wait_time : ℕ := 1

/-- The total time in hours until the men meet. -/
def total_time : ℕ := 10

/-- The distance from point B where the men meet. -/
def meeting_point : ℝ := 52.5

/-- 
Theorem stating that the men meet at the specified distance from B after the given time,
given their walking patterns.
-/
theorem men_meet_at_calculated_point :
  let distance_a := speed_a * (total_time - wait_time)
  let distance_b := (total_time / 2 : ℝ) * (initial_speed_b + initial_speed_b + speed_increase_b * (total_time - 1))
  distance_a + distance_b = total_distance ∧ distance_b = meeting_point := by sorry

end NUMINAMATH_CALUDE_stating_men_meet_at_calculated_point_l2862_286206


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2862_286295

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the focal length of the hyperbola is 2√5 -/
theorem hyperbola_focal_length 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (h_vertex_focus : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 - y₁^2/b^2 = 1 ∧ 
    y₂^2 = 2*p*x₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) 
  (h_asymptote_directrix : ∃ (k : ℝ), 
    (-2)^2/a^2 - (-1)^2/b^2 = k^2 ∧ 
    -2 = -p/2) : 
  2 * (a^2 + b^2).sqrt = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2862_286295


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2862_286262

theorem rational_equation_solution (x : ℚ) :
  x ≠ 3 →
  (x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2 →
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2862_286262


namespace NUMINAMATH_CALUDE_diegos_stamp_collection_cost_l2862_286263

def brazil_stamps : ℕ := 6 + 9
def peru_stamps : ℕ := 8 + 5
def colombia_stamps : ℕ := 7 + 6

def brazil_cost : ℚ := 0.07
def peru_cost : ℚ := 0.05
def colombia_cost : ℚ := 0.07

def total_cost : ℚ := 
  brazil_stamps * brazil_cost + 
  peru_stamps * peru_cost + 
  colombia_stamps * colombia_cost

theorem diegos_stamp_collection_cost : total_cost = 2.61 := by
  sorry

end NUMINAMATH_CALUDE_diegos_stamp_collection_cost_l2862_286263


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l2862_286207

theorem cubic_equation_sum (r s t : ℝ) : 
  r^3 - 4*r^2 + 4*r = 6 →
  s^3 - 4*s^2 + 4*s = 6 →
  t^3 - 4*t^2 + 4*t = 6 →
  r*s/t + s*t/r + t*r/s = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l2862_286207


namespace NUMINAMATH_CALUDE_max_principals_is_five_l2862_286237

/-- Represents the duration of the entire period in years -/
def total_period : ℕ := 15

/-- Represents the length of each principal's term in years -/
def term_length : ℕ := 3

/-- Calculates the maximum number of principals that can serve in the given period -/
def max_principals : ℕ := total_period / term_length

theorem max_principals_is_five : max_principals = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_five_l2862_286237


namespace NUMINAMATH_CALUDE_log_850_between_consecutive_integers_l2862_286200

theorem log_850_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < b ∧ a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_850_between_consecutive_integers_l2862_286200


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2862_286275

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → f (x * f y) = f (x * y) + x) →
  (∀ x : ℝ, x > 0 → f x = x + 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2862_286275


namespace NUMINAMATH_CALUDE_least_number_with_remainder_seven_l2862_286233

theorem least_number_with_remainder_seven (n : ℕ) : n = 1547 ↔ 
  (∀ d ∈ ({11, 17, 21, 29, 35} : Set ℕ), n % d = 7) ∧
  (∀ m < n, ∃ d ∈ ({11, 17, 21, 29, 35} : Set ℕ), m % d ≠ 7) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_seven_l2862_286233


namespace NUMINAMATH_CALUDE_locus_of_symmetric_points_l2862_286257

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Check if a point is on the x-axis -/
def isOnXAxis (p : Point2D) : Prop := p.y = 0

/-- Check if a point is on the y-axis -/
def isOnYAxis (p : Point2D) : Prop := p.x = 0

/-- Check if three points form a right angle -/
def isRightAngle (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The line symmetric to a point with respect to the coordinate axes -/
def symmetricLine (m : Point2D) : Set Point2D :=
  {n : Point2D | n.x * m.y = n.y * m.x}

/-- The main theorem -/
theorem locus_of_symmetric_points (m : Point2D) 
  (h1 : m ≠ origin) 
  (h2 : ¬isOnXAxis m) 
  (h3 : ¬isOnYAxis m) :
  ∀ (p q : Point2D), 
    isOnXAxis p → isOnYAxis q → isRightAngle p m q →
    ∃ (n : Point2D), n ∈ symmetricLine m :=
by sorry

end NUMINAMATH_CALUDE_locus_of_symmetric_points_l2862_286257


namespace NUMINAMATH_CALUDE_min_repetitions_divisible_by_15_l2862_286204

def repeated_2002_plus_15 (n : ℕ) : ℕ :=
  2002 * (10^(4*n) - 1) / 9 * 10 + 15

theorem min_repetitions_divisible_by_15 :
  ∀ k : ℕ, k < 3 → ¬(repeated_2002_plus_15 k % 15 = 0) ∧
  repeated_2002_plus_15 3 % 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_repetitions_divisible_by_15_l2862_286204


namespace NUMINAMATH_CALUDE_orange_calorie_distribution_l2862_286212

theorem orange_calorie_distribution 
  (num_oranges : ℕ) 
  (pieces_per_orange : ℕ) 
  (num_people : ℕ) 
  (calories_per_orange : ℕ) 
  (h1 : num_oranges = 5)
  (h2 : pieces_per_orange = 8)
  (h3 : num_people = 4)
  (h4 : calories_per_orange = 80) :
  (num_oranges * calories_per_orange) / (num_people) = 100 := by
  sorry

end NUMINAMATH_CALUDE_orange_calorie_distribution_l2862_286212


namespace NUMINAMATH_CALUDE_oliver_fruit_consumption_l2862_286272

/-- The number of fruits Oliver consumed -/
def fruits_consumed (initial_cherries initial_strawberries initial_blueberries
                     remaining_cherries remaining_strawberries remaining_blueberries : ℝ) : ℝ :=
  (initial_cherries - remaining_cherries) +
  (initial_strawberries - remaining_strawberries) +
  (initial_blueberries - remaining_blueberries)

/-- Theorem stating that Oliver consumed 17.2 fruits in total -/
theorem oliver_fruit_consumption :
  fruits_consumed 16.5 10.7 20.2 6.3 8.4 15.5 = 17.2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_fruit_consumption_l2862_286272


namespace NUMINAMATH_CALUDE_merchant_profit_l2862_286254

theorem merchant_profit (C S : ℝ) (h : 22 * C = 16 * S) : 
  (S - C) / C * 100 = 37.5 := by sorry

end NUMINAMATH_CALUDE_merchant_profit_l2862_286254


namespace NUMINAMATH_CALUDE_unique_divisible_sum_l2862_286267

theorem unique_divisible_sum (p : ℕ) (h_prime : Nat.Prime p) :
  ∃! n : ℕ, (p * n) % (p + n) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_sum_l2862_286267


namespace NUMINAMATH_CALUDE_f_monotonicity_and_equality_l2862_286245

noncomputable def f (x : ℝ) : ℝ := (Real.exp 1) * x / Real.exp x

theorem f_monotonicity_and_equality (e : ℝ) (he : e = Real.exp 1) :
  (∀ x y, x < y → x < 1 → y < 1 → f x < f y) ∧
  (∀ x y, x < y → 1 < x → 1 < y → f y < f x) ∧
  (∀ x, x > 0 → f (1 - x) ≠ f (1 + x)) ∧
  (f (1 - 0) = f (1 + 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_equality_l2862_286245


namespace NUMINAMATH_CALUDE_isosceles_triangle_legs_l2862_286276

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2

-- Define the theorem
theorem isosceles_triangle_legs (t : IsoscelesTriangle) :
  t.side1 + t.side2 + t.base = 18 ∧ (t.side1 = 8 ∨ t.base = 8) →
  t.side1 = 8 ∨ t.side1 = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_legs_l2862_286276


namespace NUMINAMATH_CALUDE_expand_expression_l2862_286265

theorem expand_expression (x : ℝ) : (7*x + 5) * 3*x^2 = 21*x^3 + 15*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2862_286265


namespace NUMINAMATH_CALUDE_min_digits_of_m_l2862_286220

theorem min_digits_of_m (n : ℤ) : 
  let m := (n - 1001) * (n - 2001) * (n - 2002) * (n - 3001) * (n - 3002) * (n - 3003)
  m > 0 → m ≥ 10^10 :=
by sorry

end NUMINAMATH_CALUDE_min_digits_of_m_l2862_286220


namespace NUMINAMATH_CALUDE_track_completion_time_l2862_286266

/-- Time to complete a circular track -/
def complete_track_time (half_track_time : ℝ) : ℝ :=
  2 * half_track_time

/-- Theorem: The time to complete the circular track is 6 minutes -/
theorem track_completion_time :
  let half_track_time : ℝ := 3
  complete_track_time half_track_time = 6 := by
  sorry


end NUMINAMATH_CALUDE_track_completion_time_l2862_286266


namespace NUMINAMATH_CALUDE_expected_girls_left_ten_boys_seven_girls_l2862_286215

/-- The expected number of girls standing to the left of all boys in a random lineup -/
def expected_girls_left (num_boys : ℕ) (num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1 : ℚ)

/-- Theorem: In a random lineup of 10 boys and 7 girls, the expected number of girls 
    standing to the left of all boys is 7/11 -/
theorem expected_girls_left_ten_boys_seven_girls :
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_ten_boys_seven_girls_l2862_286215


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2862_286292

theorem rectangle_dimensions : ∃ (x y : ℝ), 
  y = x + 3 ∧ 
  2 * (2 * (x + y)) = x * y ∧ 
  x = 8 ∧ 
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2862_286292


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2862_286219

theorem quadratic_equation_solution (m : ℝ) : 
  (m - 1 ≠ 0) → (m^2 - 3*m + 2 = 0) → (m = 2) := by
  sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2862_286219


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l2862_286291

theorem min_value_sum_fractions (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  let u := (3*a^2 - a)/(1 + a^2) + (3*b^2 - b)/(1 + b^2) + (3*c^2 - c)/(1 + c^2)
  u ≥ 0 ∧ (u = 0 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l2862_286291


namespace NUMINAMATH_CALUDE_triangle_side_length_l2862_286288

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 7 → b = 8 → A = π/3 → (c = 3 ∨ c = 5) → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2862_286288


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l2862_286255

/-- Yan's problem setup -/
structure YanProblem where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to home
  z : ℝ  -- Distance from Yan to school
  h_positive : w > 0 ∧ x > 0 ∧ z > 0  -- Positive distances and speed
  h_between : x + z > 0  -- Yan is between home and school
  h_equal_time : z / w = x / w + (x + z) / (5 * w)  -- Equal time condition

/-- The main theorem: ratio of distances is 2/3 -/
theorem yan_distance_ratio (p : YanProblem) : p.x / p.z = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_yan_distance_ratio_l2862_286255


namespace NUMINAMATH_CALUDE_mikey_jelly_beans_mikey_jelly_beans_holds_l2862_286296

/-- Proves that Mikey has 19 jelly beans given the conditions of the problem -/
theorem mikey_jelly_beans : ℕ → ℕ → ℕ → Prop :=
  fun napoleon sedrich mikey =>
    napoleon = 17 →
    sedrich = napoleon + 4 →
    2 * (napoleon + sedrich) = 4 * mikey →
    mikey = 19

/-- The theorem holds for the given values -/
theorem mikey_jelly_beans_holds : 
  ∃ (napoleon sedrich mikey : ℕ), mikey_jelly_beans napoleon sedrich mikey :=
by
  sorry

end NUMINAMATH_CALUDE_mikey_jelly_beans_mikey_jelly_beans_holds_l2862_286296


namespace NUMINAMATH_CALUDE_sample_size_is_selected_size_l2862_286242

/-- Represents the total number of first-year high school students -/
def population_size : ℕ := 1320

/-- Represents the number of students selected for measurement -/
def selected_size : ℕ := 220

/-- Theorem stating that the sample size is equal to the number of selected students -/
theorem sample_size_is_selected_size : 
  selected_size = 220 := by sorry

end NUMINAMATH_CALUDE_sample_size_is_selected_size_l2862_286242


namespace NUMINAMATH_CALUDE_red_pepper_weight_l2862_286228

theorem red_pepper_weight (total_weight green_weight : ℚ) 
  (h1 : total_weight = 0.66)
  (h2 : green_weight = 0.33) :
  total_weight - green_weight = 0.33 := by
sorry

end NUMINAMATH_CALUDE_red_pepper_weight_l2862_286228


namespace NUMINAMATH_CALUDE_johns_total_income_this_year_l2862_286299

/-- Calculates the total income (salary + bonus) for the current year given the previous year's salary and bonus, and the current year's salary. -/
def totalIncomeCurrentYear (prevSalary prevBonus currSalary : ℕ) : ℕ :=
  let bonusRate := prevBonus / prevSalary
  let currBonus := currSalary * bonusRate
  currSalary + currBonus

/-- Theorem stating that John's total income this year is $220,000 -/
theorem johns_total_income_this_year :
  totalIncomeCurrentYear 100000 10000 200000 = 220000 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_income_this_year_l2862_286299


namespace NUMINAMATH_CALUDE_mckenna_start_time_l2862_286277

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Mckenna's work schedule -/
structure WorkSchedule where
  officeEndTime : Time
  meetingEndTime : Time
  workDuration : Nat
  totalWorkDuration : Nat

/-- Calculate the difference between two times in hours -/
def timeDifference (t1 t2 : Time) : Nat :=
  sorry

/-- Calculate the time after adding hours to a given time -/
def addHours (t : Time) (hours : Nat) : Time :=
  sorry

theorem mckenna_start_time (schedule : WorkSchedule)
  (h1 : schedule.officeEndTime = ⟨11, 0⟩)
  (h2 : schedule.meetingEndTime = ⟨13, 0⟩)
  (h3 : schedule.workDuration = 2)
  (h4 : schedule.totalWorkDuration = 7) :
  timeDifference ⟨8, 0⟩ (addHours schedule.meetingEndTime schedule.workDuration) = schedule.totalWorkDuration :=
sorry

end NUMINAMATH_CALUDE_mckenna_start_time_l2862_286277


namespace NUMINAMATH_CALUDE_mixture_weight_l2862_286218

/-- Given a mixture of zinc, copper, and silver in the ratio 9 : 11 : 7,
    where 27 kg of zinc is used, the total weight of the mixture is 81 kg. -/
theorem mixture_weight (zinc copper silver : ℕ) (zinc_weight : ℝ) :
  zinc = 9 →
  copper = 11 →
  silver = 7 →
  zinc_weight = 27 →
  (zinc_weight / zinc) * (zinc + copper + silver) = 81 :=
by sorry

end NUMINAMATH_CALUDE_mixture_weight_l2862_286218


namespace NUMINAMATH_CALUDE_slant_height_and_height_not_unique_l2862_286216

/-- Represents a right triangular pyramid with a square base -/
structure RightTriangularPyramid where
  base_side : ℝ
  height : ℝ
  slant_height : ℝ

/-- Predicate to check if two pyramids are different -/
def different_pyramids (p1 p2 : RightTriangularPyramid) : Prop :=
  p1.base_side ≠ p2.base_side ∧ p1.height = p2.height ∧ p1.slant_height = p2.slant_height

/-- Theorem stating that slant height and height do not uniquely specify the pyramid -/
theorem slant_height_and_height_not_unique :
  ∃ (p1 p2 : RightTriangularPyramid), different_pyramids p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_slant_height_and_height_not_unique_l2862_286216


namespace NUMINAMATH_CALUDE_probability_intersection_independent_events_l2862_286239

theorem probability_intersection_independent_events 
  (a b : Set ℝ) 
  (p : Set ℝ → ℝ) 
  (h1 : p a = 5/7) 
  (h2 : p b = 2/5) 
  (h3 : p (a ∩ b) = p a * p b) : 
  p (a ∩ b) = 2/7 := by
sorry

end NUMINAMATH_CALUDE_probability_intersection_independent_events_l2862_286239


namespace NUMINAMATH_CALUDE_fib_F15_units_digit_l2862_286261

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of F_{F₁₅} is 5 -/
theorem fib_F15_units_digit :
  unitsDigit (fib (fib 15)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fib_F15_units_digit_l2862_286261


namespace NUMINAMATH_CALUDE_volume_ratio_l2862_286205

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
def volume_S (B : Prism) (r : ℝ) (a b c d : ℝ) : ℝ :=
  a * r^3 + b * r^2 + c * r + d

theorem volume_ratio (B : Prism) (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  b * c / (a * d) = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_l2862_286205


namespace NUMINAMATH_CALUDE_b_work_time_l2862_286285

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 5
def work_rate_BC : ℚ := 1 / 3
def work_rate_AC : ℚ := 1 / 2

-- Theorem to prove
theorem b_work_time (work_rate_B : ℚ) : 
  work_rate_A + (work_rate_BC - work_rate_B) = work_rate_AC → 
  (1 : ℚ) / work_rate_B = 30 := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l2862_286285


namespace NUMINAMATH_CALUDE_book_exchange_ways_l2862_286224

theorem book_exchange_ways (n₁ n₂ k : ℕ) (h₁ : n₁ = 6) (h₂ : n₂ = 8) (h₃ : k = 3) : 
  (n₁.choose k) * (n₂.choose k) = 1120 := by
  sorry

end NUMINAMATH_CALUDE_book_exchange_ways_l2862_286224


namespace NUMINAMATH_CALUDE_factorial_difference_l2862_286250

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 8 = 3588480 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2862_286250


namespace NUMINAMATH_CALUDE_problem_1_l2862_286211

theorem problem_1 : |-2| + (1/3)⁻¹ - (Real.sqrt 3 - 2021)^0 - Real.sqrt 3 * Real.tan (π/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2862_286211


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l2862_286240

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 10 →
  capacity_ratio = 2 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 25 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l2862_286240


namespace NUMINAMATH_CALUDE_lineup_ways_eq_choose_four_from_fourteen_l2862_286246

/-- The number of ways to choose an 8-player lineup from 18 players,
    including two sets of twins that must be in the lineup. -/
def lineup_ways (total_players : ℕ) (lineup_size : ℕ) (twin_pairs : ℕ) : ℕ :=
  Nat.choose (total_players - 2 * twin_pairs) (lineup_size - 2 * twin_pairs)

/-- Theorem stating that the number of ways to choose the lineup
    is equal to choosing 4 from 14 players. -/
theorem lineup_ways_eq_choose_four_from_fourteen :
  lineup_ways 18 8 2 = Nat.choose 14 4 := by
  sorry

end NUMINAMATH_CALUDE_lineup_ways_eq_choose_four_from_fourteen_l2862_286246


namespace NUMINAMATH_CALUDE_gcd_333_481_l2862_286286

theorem gcd_333_481 : Nat.gcd 333 481 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_333_481_l2862_286286


namespace NUMINAMATH_CALUDE_sum_in_base5_l2862_286201

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_in_base5 : toBase5 (45 + 78) = [4, 4, 3] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base5_l2862_286201


namespace NUMINAMATH_CALUDE_unique_triangle_function_l2862_286251

def IsNonDegenerateTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def SatisfiesTriangleCondition (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → IsNonDegenerateTriangle x (f y) (f (y + f x - 1))

theorem unique_triangle_function :
  ∃! f : ℕ → ℕ, (∀ x : ℕ, x > 0 → f x > 0) ∧ SatisfiesTriangleCondition f ∧ (∀ x : ℕ, x > 0 → f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_function_l2862_286251


namespace NUMINAMATH_CALUDE_log_inequality_l2862_286232

theorem log_inequality : Real.log 2 / Real.log 3 < Real.log 3 / Real.log 2 ∧ 
                         Real.log 3 / Real.log 2 < Real.log 5 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2862_286232


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2862_286238

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the side of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_side_length :
  ∀ t : IsoscelesTrapezoid,
    t.base1 = 11 ∧ t.base2 = 17 ∧ t.area = 56 →
    side_length t = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2862_286238


namespace NUMINAMATH_CALUDE_mower_blades_cost_is_47_l2862_286290

/-- The amount Mike made mowing lawns -/
def total_earnings : ℕ := 101

/-- The number of games Mike could buy with the remaining money -/
def num_games : ℕ := 9

/-- The cost of each game -/
def game_cost : ℕ := 6

/-- The amount Mike spent on new mower blades -/
def mower_blades_cost : ℕ := total_earnings - (num_games * game_cost)

theorem mower_blades_cost_is_47 : mower_blades_cost = 47 := by
  sorry

end NUMINAMATH_CALUDE_mower_blades_cost_is_47_l2862_286290


namespace NUMINAMATH_CALUDE_f_eight_equals_zero_l2862_286243

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_eight_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x) :
  f 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_eight_equals_zero_l2862_286243


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l2862_286231

/-- Given a quadratic function y = ax² + bx + c satisfying specific conditions,
    prove that the distance between its roots is √17/2 -/
theorem quadratic_roots_distance (a b c : ℝ) : 
  (a*(-1)^2 + b*(-1) + c = -1) →
  (a*0^2 + b*0 + c = -2) →
  (a*1^2 + b*1 + c = 1) →
  let f := fun x => a*x^2 + b*x + c
  let roots := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = Real.sqrt 17 / 2 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_distance_l2862_286231


namespace NUMINAMATH_CALUDE_count_integer_points_on_line_l2862_286271

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Check if a point is strictly between two other points -/
def strictly_between (p q r : IntPoint) : Prop :=
  (p.x < q.x ∧ q.x < r.x) ∨ (r.x < q.x ∧ q.x < p.x)

/-- The line passing through two points -/
def line_through (p q : IntPoint) (r : IntPoint) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- The main theorem -/
theorem count_integer_points_on_line :
  let A : IntPoint := ⟨3, 3⟩
  let B : IntPoint := ⟨120, 150⟩
  ∃! (points : Finset IntPoint),
    (∀ p ∈ points, line_through A B p ∧ strictly_between A p B) ∧
    points.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_points_on_line_l2862_286271


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l2862_286260

theorem chocolate_bars_distribution (total_bars : ℕ) (num_small_boxes : ℕ) 
  (h1 : total_bars = 504) (h2 : num_small_boxes = 18) :
  total_bars / num_small_boxes = 28 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l2862_286260


namespace NUMINAMATH_CALUDE_overlap_percentage_l2862_286223

theorem overlap_percentage (square_side : ℝ) (rect_width rect_length : ℝ) 
  (overlap_rect_width overlap_rect_length : ℝ) :
  square_side = 12 →
  rect_width = 9 →
  rect_length = 12 →
  overlap_rect_width = 12 →
  overlap_rect_length = 18 →
  (((square_side + rect_width - overlap_rect_length) * rect_width) / 
   (overlap_rect_width * overlap_rect_length)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_percentage_l2862_286223


namespace NUMINAMATH_CALUDE_david_solo_completion_time_l2862_286264

/-- The number of days it takes David to complete the job alone -/
def david_solo_days : ℝ := 12

/-- The number of days David works alone before Moore joins -/
def david_solo_work : ℝ := 6

/-- The number of days it takes David and Moore to complete the job together -/
def david_moore_total : ℝ := 6

/-- The number of days it takes David and Moore to complete the remaining job after David works alone -/
def david_moore_remaining : ℝ := 3

theorem david_solo_completion_time :
  (david_solo_work / david_solo_days) + 
  (david_moore_remaining / david_moore_total) = 1 :=
sorry

end NUMINAMATH_CALUDE_david_solo_completion_time_l2862_286264


namespace NUMINAMATH_CALUDE_paving_cost_example_l2862_286209

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

theorem paving_cost_example :
  paving_cost 5.5 4 700 = 15400 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_example_l2862_286209


namespace NUMINAMATH_CALUDE_jiaqi_pe_grade_l2862_286256

/-- Calculates the final grade based on component scores and weights -/
def calculate_grade (extracurricular_score : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  0.2 * extracurricular_score + 0.3 * midterm_score + 0.5 * final_score

/-- Proves that Jiaqi's physical education grade for the semester is 95.3 points -/
theorem jiaqi_pe_grade :
  let max_score : ℝ := 100
  let extracurricular_weight : ℝ := 0.2
  let midterm_weight : ℝ := 0.3
  let final_weight : ℝ := 0.5
  let jiaqi_extracurricular : ℝ := 96
  let jiaqi_midterm : ℝ := 92
  let jiaqi_final : ℝ := 97
  calculate_grade jiaqi_extracurricular jiaqi_midterm jiaqi_final = 95.3 := by
  sorry

end NUMINAMATH_CALUDE_jiaqi_pe_grade_l2862_286256


namespace NUMINAMATH_CALUDE_stars_per_bottle_l2862_286269

/-- Given Shiela's paper stars and number of classmates, prove the number of stars per bottle. -/
theorem stars_per_bottle (total_stars : ℕ) (num_classmates : ℕ) 
  (h1 : total_stars = 45) 
  (h2 : num_classmates = 9) : 
  total_stars / num_classmates = 5 := by
  sorry

#check stars_per_bottle

end NUMINAMATH_CALUDE_stars_per_bottle_l2862_286269


namespace NUMINAMATH_CALUDE_lillian_cupcakes_l2862_286213

/-- Represents the number of dozen cupcakes Lillian can bake and ice --/
def cupcakes_dozen : ℕ := by sorry

theorem lillian_cupcakes :
  let initial_sugar : ℕ := 3
  let bags_bought : ℕ := 2
  let sugar_per_bag : ℕ := 6
  let sugar_for_batter : ℕ := 1
  let sugar_for_frosting : ℕ := 2
  
  let total_sugar : ℕ := initial_sugar + bags_bought * sugar_per_bag
  let sugar_per_dozen : ℕ := sugar_for_batter + sugar_for_frosting
  
  cupcakes_dozen = total_sugar / sugar_per_dozen ∧ cupcakes_dozen = 5 := by sorry

end NUMINAMATH_CALUDE_lillian_cupcakes_l2862_286213


namespace NUMINAMATH_CALUDE_math_contest_grade11_score_l2862_286297

theorem math_contest_grade11_score (n : ℕ) (grade11_score : ℝ) :
  let grade11_count : ℝ := 0.2 * n
  let grade12_count : ℝ := 0.8 * n
  let overall_average : ℝ := 78
  let grade12_average : ℝ := 75
  (grade11_count * grade11_score + grade12_count * grade12_average) / n = overall_average →
  grade11_score = 90 := by
sorry

end NUMINAMATH_CALUDE_math_contest_grade11_score_l2862_286297


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l2862_286229

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2033 = i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l2862_286229


namespace NUMINAMATH_CALUDE_total_distance_is_164_l2862_286203

-- Define the parameters
def flat_speed : ℝ := 20
def flat_time : ℝ := 4.5
def uphill_speed : ℝ := 12
def uphill_time : ℝ := 2.5
def downhill_speed : ℝ := 24
def downhill_time : ℝ := 1.5
def walking_distance : ℝ := 8

-- Define the function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem total_distance_is_164 :
  distance flat_speed flat_time +
  distance uphill_speed uphill_time +
  distance downhill_speed downhill_time +
  walking_distance = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_164_l2862_286203


namespace NUMINAMATH_CALUDE_unique_root_of_increasing_function_l2862_286283

theorem unique_root_of_increasing_function (f : ℝ → ℝ) (h : Monotone f) :
  ∃! x, f x = 0 ∨ (∀ x, f x ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_increasing_function_l2862_286283


namespace NUMINAMATH_CALUDE_total_cost_is_540_l2862_286202

def cherry_price : ℝ := 5
def olive_price : ℝ := 7
def bag_count : ℕ := 50
def discount_rate : ℝ := 0.1

def discounted_price (original_price : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def total_cost : ℝ :=
  bag_count * (discounted_price cherry_price + discounted_price olive_price)

theorem total_cost_is_540 : total_cost = 540 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_540_l2862_286202


namespace NUMINAMATH_CALUDE_green_ball_probability_l2862_286247

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containers : List Container := [
  ⟨10, 2⟩,  -- Container I
  ⟨3, 5⟩,   -- Container II
  ⟨2, 6⟩,   -- Container III
  ⟨5, 3⟩    -- Container IV
]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

theorem green_ball_probability : 
  (containers.map (fun c => containerProbability * greenProbability c)).sum = 23 / 48 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2862_286247


namespace NUMINAMATH_CALUDE_N_mod_52_l2862_286282

/-- The number formed by concatenating integers from 1 to 51 -/
def N : ℕ := sorry

/-- The remainder when N is divided by 52 -/
def remainder : ℕ := N % 52

theorem N_mod_52 : remainder = 13 := by sorry

end NUMINAMATH_CALUDE_N_mod_52_l2862_286282


namespace NUMINAMATH_CALUDE_euler_product_theorem_l2862_286280

theorem euler_product_theorem : ∀ (z₁ z₂ : ℂ),
  (z₁ = Complex.exp (Complex.I * Real.pi / 3)) →
  (z₂ = Complex.exp (Complex.I * Real.pi / 6)) →
  z₁ * z₂ = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_euler_product_theorem_l2862_286280


namespace NUMINAMATH_CALUDE_smallest_result_l2862_286279

def S : Set ℕ := {6, 8, 10, 12, 14, 16}

def process (a b c : ℕ) : ℕ := (a + b) * c - 10

def valid_choice (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∀ a b c : ℕ, valid_choice a b c →
    98 ≤ min (process a b c) (min (process a c b) (process b c a)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l2862_286279
