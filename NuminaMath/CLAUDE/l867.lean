import Mathlib

namespace NUMINAMATH_CALUDE_circle_and_line_problem_l867_86721

/-- Given a circle A with center at (-1, 2) tangent to line m: x + 2y + 7 = 0,
    and a moving line l passing through B(-2, 0) intersecting circle A at M and N,
    prove the equation of circle A and find the equations of line l when |MN| = 2√19. -/
theorem circle_and_line_problem :
  ∀ (A : ℝ × ℝ) (m : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) (M N : ℝ × ℝ),
  A = (-1, 2) →
  (∀ x y, m x y ↔ x + 2*y + 7 = 0) →
  (∃ r : ℝ, ∀ x y, (x + 1)^2 + (y - 2)^2 = r^2 ↔ m x y) →
  (∀ x, l x 0 ↔ x = -2) →
  (∃ x y, l x y ∧ (x + 1)^2 + (y - 2)^2 = 20) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 4*19 →
  ((∀ x y, (x + 1)^2 + (y - 2)^2 = 20 ↔ (x - A.1)^2 + (y - A.2)^2 = 20) ∧
   ((∀ x y, l x y ↔ 3*x - 4*y + 6 = 0) ∨ (∀ x y, l x y ↔ x = -2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l867_86721


namespace NUMINAMATH_CALUDE_point_b_coordinate_l867_86768

theorem point_b_coordinate (b : ℝ) : 
  (|(-2) - b| = 3) ↔ (b = -5 ∨ b = 1) := by sorry

end NUMINAMATH_CALUDE_point_b_coordinate_l867_86768


namespace NUMINAMATH_CALUDE_table_tennis_probabilities_l867_86789

def num_players : ℕ := 6
def num_players_A : ℕ := 3
def num_players_B : ℕ := 1
def num_players_C : ℕ := 2

def probability_at_least_one_C : ℚ := 3/5
def probability_same_association : ℚ := 4/15

theorem table_tennis_probabilities :
  (num_players = num_players_A + num_players_B + num_players_C) →
  (probability_at_least_one_C = 3/5) ∧
  (probability_same_association = 4/15) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_probabilities_l867_86789


namespace NUMINAMATH_CALUDE_all_twentynine_l867_86729

/-- A function that represents a circular arrangement of 2017 integers. -/
def CircularArrangement := Fin 2017 → ℤ

/-- Predicate to check if five consecutive elements in the arrangement are "arrangeable". -/
def IsArrangeable (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2017, arr i - arr (i + 1) + arr (i + 2) - arr (i + 3) + arr (i + 4) = 29

/-- Theorem stating that if all consecutive five-tuples in a circular arrangement of 2017 integers
    are arrangeable, then all integers in the arrangement must be 29. -/
theorem all_twentynine (arr : CircularArrangement) (h : IsArrangeable arr) :
    ∀ i : Fin 2017, arr i = 29 := by
  sorry

end NUMINAMATH_CALUDE_all_twentynine_l867_86729


namespace NUMINAMATH_CALUDE_equilateral_triangle_l867_86724

theorem equilateral_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ a + c > b)
  (side_relation : a^2 + 2*b^2 + c^2 - 2*b*(a + c) = 0) : 
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_l867_86724


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l867_86746

theorem simplify_fraction_product : (240 / 24) * (7 / 140) * (6 / 4) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l867_86746


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l867_86785

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem fixed_points_of_f_composition (x : ℝ) : 
  f (f x) = f x ↔ x ∈ ({-1, 0, 4, 5} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l867_86785


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_values_l867_86796

def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - a) = 0}

def B : Set ℝ := {x | (x - 2) * (x - 3) = 0}

theorem intersection_empty_iff_a_values (a : ℝ) :
  A a ∩ B = ∅ ↔ a = 1 ∨ a = 4 ∨ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_values_l867_86796


namespace NUMINAMATH_CALUDE_combination_problem_l867_86761

theorem combination_problem (n : ℕ) : 
  n * (n - 1) = 42 → n.choose 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_combination_problem_l867_86761


namespace NUMINAMATH_CALUDE_double_negation_and_abs_value_l867_86777

theorem double_negation_and_abs_value : 
  (-(-2) = 2) ∧ (-(abs (-2)) = -2) := by sorry

end NUMINAMATH_CALUDE_double_negation_and_abs_value_l867_86777


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l867_86704

theorem gcd_of_powers_of_two : 
  Nat.gcd (2^2050 - 1) (2^2040 - 1) = 2^10 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l867_86704


namespace NUMINAMATH_CALUDE_fuchsia_survey_l867_86739

theorem fuchsia_survey (total : ℕ) (kinda_pink : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 100)
  (h_kinda_pink : kinda_pink = 60)
  (h_both : both = 27)
  (h_neither : neither = 17) :
  ∃ (purply : ℕ), purply = 50 ∧ purply = total - (kinda_pink - both + neither) :=
by sorry

end NUMINAMATH_CALUDE_fuchsia_survey_l867_86739


namespace NUMINAMATH_CALUDE_smallest_k_value_l867_86773

theorem smallest_k_value (p q r s k : ℕ+) : 
  (p + 2*q + 3*r + 4*s = k) →
  (4*p = 3*q) →
  (4*p = 2*r) →
  (4*p = s) →
  (∀ p' q' r' s' k' : ℕ+, 
    (p' + 2*q' + 3*r' + 4*s' = k') →
    (4*p' = 3*q') →
    (4*p' = 2*r') →
    (4*p' = s') →
    k ≤ k') →
  k = 77 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_value_l867_86773


namespace NUMINAMATH_CALUDE_min_deliveries_to_breakeven_l867_86750

def van_cost : ℕ := 8000
def earning_per_delivery : ℕ := 15
def gas_cost_per_delivery : ℕ := 5

theorem min_deliveries_to_breakeven :
  ∃ (d : ℕ), d * (earning_per_delivery - gas_cost_per_delivery) ≥ van_cost ∧
  ∀ (k : ℕ), k * (earning_per_delivery - gas_cost_per_delivery) ≥ van_cost → k ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_deliveries_to_breakeven_l867_86750


namespace NUMINAMATH_CALUDE_stock_percentage_l867_86784

/-- The percentage of a stock given certain conditions -/
theorem stock_percentage (income : ℝ) (investment : ℝ) (percentage : ℝ) : 
  income = 1000 →
  investment = 10000 →
  income = (percentage * investment) / 100 →
  percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_stock_percentage_l867_86784


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l867_86787

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perp m α) 
  (h3 : perp n α) : 
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l867_86787


namespace NUMINAMATH_CALUDE_claire_gift_card_balance_l867_86788

/-- Calculates the remaining balance on Claire's gift card after a week of purchases. -/
def remaining_balance (gift_card_value : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (days : ℕ) (cookie_cost : ℚ) (num_cookies : ℕ) : ℚ :=
  gift_card_value - 
  ((latte_cost + croissant_cost) * days + cookie_cost * num_cookies)

/-- Proves that Claire will have $43.00 left on her gift card after a week of purchases. -/
theorem claire_gift_card_balance : 
  remaining_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_claire_gift_card_balance_l867_86788


namespace NUMINAMATH_CALUDE_chess_team_boys_count_l867_86793

theorem chess_team_boys_count :
  ∀ (total_members : ℕ) (total_attendees : ℕ) (boys : ℕ) (girls : ℕ),
  total_members = 30 →
  total_attendees = 20 →
  total_members = boys + girls →
  total_attendees = boys + (girls / 3) →
  boys = 15 := by
sorry

end NUMINAMATH_CALUDE_chess_team_boys_count_l867_86793


namespace NUMINAMATH_CALUDE_paper_folding_ratio_l867_86713

theorem paper_folding_ratio : 
  let square_side : ℝ := 8
  let folded_height : ℝ := square_side / 2
  let folded_width : ℝ := square_side
  let cut_height : ℝ := folded_height / 3
  let small_rect_height : ℝ := cut_height
  let small_rect_width : ℝ := folded_width
  let large_rect_height : ℝ := folded_height - cut_height
  let large_rect_width : ℝ := folded_width
  let small_rect_perimeter : ℝ := 2 * (small_rect_height + small_rect_width)
  let large_rect_perimeter : ℝ := 2 * (large_rect_height + large_rect_width)
  small_rect_perimeter / large_rect_perimeter = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_paper_folding_ratio_l867_86713


namespace NUMINAMATH_CALUDE_actual_length_is_320_l867_86770

/-- Blueprint scale factor -/
def scale_factor : ℝ := 20

/-- Measured length on the blueprint in cm -/
def measured_length : ℝ := 16

/-- Actual length of the part in cm -/
def actual_length : ℝ := measured_length * scale_factor

/-- Theorem stating that the actual length is 320cm -/
theorem actual_length_is_320 : actual_length = 320 := by
  sorry

end NUMINAMATH_CALUDE_actual_length_is_320_l867_86770


namespace NUMINAMATH_CALUDE_number_problem_l867_86706

theorem number_problem (x : ℚ) : x^2 + 105 = (x - 19)^2 → x = 128/19 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l867_86706


namespace NUMINAMATH_CALUDE_special_curve_hyperbola_range_l867_86730

/-- A curve defined by the equation x^2 / (m + 2) + y^2 / (m + 1) = 1 --/
def is_special_curve (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m + 1) = 1

/-- The condition for the curve to be a hyperbola with foci on the x-axis --/
def is_hyperbola_x_foci (m : ℝ) : Prop :=
  (m + 2 > 0) ∧ (m + 1 < 0)

/-- The main theorem stating the range of m for which the curve is a hyperbola with foci on the x-axis --/
theorem special_curve_hyperbola_range (m : ℝ) :
  is_special_curve m ∧ is_hyperbola_x_foci m ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_special_curve_hyperbola_range_l867_86730


namespace NUMINAMATH_CALUDE_sphere_radius_is_six_l867_86744

/-- The shadow length of the sphere -/
def sphere_shadow : ℝ := 12

/-- The height of the meter stick -/
def stick_height : ℝ := 1.5

/-- The shadow length of the meter stick -/
def stick_shadow : ℝ := 3

/-- The radius of the sphere -/
def sphere_radius : ℝ := 6

/-- Theorem stating that the radius of the sphere is 6 meters given the conditions -/
theorem sphere_radius_is_six :
  stick_height / stick_shadow = sphere_radius / sphere_shadow :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_is_six_l867_86744


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l867_86728

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l867_86728


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_129_6_l867_86707

theorem percentage_of_360_equals_129_6 : 
  (129.6 / 360) * 100 = 36 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_129_6_l867_86707


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l867_86742

-- Define the two similar triangles
def Triangle1 : Type := Unit
def Triangle2 : Type := Unit

-- Define the height ratio
def height_ratio : ℚ := 2 / 3

-- Define the sum of perimeters
def total_perimeter : ℝ := 50

-- Define the perimeters of the two triangles
def perimeter1 : ℝ := 20
def perimeter2 : ℝ := 30

-- Theorem statement
theorem similar_triangles_perimeter :
  (perimeter1 / perimeter2 = height_ratio) ∧
  (perimeter1 + perimeter2 = total_perimeter) :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l867_86742


namespace NUMINAMATH_CALUDE_count_seven_to_800_l867_86712

def count_seven (n : ℕ) : ℕ := 
  let units := n / 10
  let tens := n / 100
  let hundreds := if n ≥ 700 then 100 else 0
  units + tens * 10 + hundreds

theorem count_seven_to_800 : count_seven 800 = 260 := by sorry

end NUMINAMATH_CALUDE_count_seven_to_800_l867_86712


namespace NUMINAMATH_CALUDE_age_ratio_l867_86718

theorem age_ratio (sachin_age rahul_age : ℕ) : 
  sachin_age = 49 → 
  rahul_age = sachin_age + 14 → 
  (sachin_age : ℚ) / rahul_age = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_l867_86718


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l867_86753

/-- Given a large rectangle containing a smaller shaded rectangle, 
    where the total area is 180 square inches and the shaded area is 120 square inches,
    prove that the perimeter of the non-shaded region is 32 inches. -/
theorem non_shaded_perimeter (total_area shaded_area : ℝ) 
  (h1 : total_area = 180)
  (h2 : shaded_area = 120)
  (h3 : ∃ (a b : ℝ), a * b = total_area - shaded_area ∧ a + b = 16) :
  2 * 16 = 32 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l867_86753


namespace NUMINAMATH_CALUDE_comparison_proofs_l867_86726

theorem comparison_proofs :
  (-2.3 < 2.4) ∧ (-3/4 > -5/6) ∧ (0 > -Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_comparison_proofs_l867_86726


namespace NUMINAMATH_CALUDE_interest_rate_proof_l867_86764

/-- Given simple interest and compound interest for 2 years, prove the interest rate -/
theorem interest_rate_proof (P : ℝ) (R : ℝ) : 
  (2 * P * R / 100 = 600) →  -- Simple interest condition
  (P * ((1 + R / 100)^2 - 1) = 630) →  -- Compound interest condition
  R = 10 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l867_86764


namespace NUMINAMATH_CALUDE_shortest_ribbon_length_l867_86711

theorem shortest_ribbon_length (ribbon_length : ℕ) : 
  (ribbon_length % 2 = 0) ∧ 
  (ribbon_length % 5 = 0) ∧ 
  (ribbon_length % 7 = 0) ∧ 
  (∀ x : ℕ, x < ribbon_length → (x % 2 = 0 ∧ x % 5 = 0 ∧ x % 7 = 0) → False) → 
  ribbon_length = 70 := by
sorry

end NUMINAMATH_CALUDE_shortest_ribbon_length_l867_86711


namespace NUMINAMATH_CALUDE_teacher_total_score_l867_86748

/-- Calculates the total score of a teacher based on their written test and interview scores -/
def calculate_total_score (written_score : ℝ) (interview_score : ℝ) 
  (written_weight : ℝ) (interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem: The teacher's total score is 72 points -/
theorem teacher_total_score : 
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  calculate_total_score written_score interview_score written_weight interview_weight = 72 := by
sorry

end NUMINAMATH_CALUDE_teacher_total_score_l867_86748


namespace NUMINAMATH_CALUDE_area_ratio_lateral_angle_relation_area_ratio_bounds_l867_86762

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  -- Add necessary fields here
  mk ::

/-- The ratio of cross-section area to lateral surface area -/
def area_ratio (p : RegularQuadPyramid) : ℝ := sorry

/-- The angle between two adjacent lateral faces -/
def lateral_face_angle (p : RegularQuadPyramid) : ℝ := sorry

/-- Theorem about the relationship between area ratio and lateral face angle -/
theorem area_ratio_lateral_angle_relation (p : RegularQuadPyramid) :
  lateral_face_angle p = Real.arccos (8 * (area_ratio p)^2 - 1) :=
sorry

/-- Theorem about the permissible values of the area ratio -/
theorem area_ratio_bounds (p : RegularQuadPyramid) :
  0 < area_ratio p ∧ area_ratio p < Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_lateral_angle_relation_area_ratio_bounds_l867_86762


namespace NUMINAMATH_CALUDE_ellipse_equation_l867_86738

/-- The equation of an ellipse passing through points (1, √3/2) and (2, 0) -/
theorem ellipse_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  (1 / a^2 + (Real.sqrt 3 / 2)^2 / b^2 = 1) ∧ 
  (4 / a^2 = 1) ∧
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l867_86738


namespace NUMINAMATH_CALUDE_new_average_weight_l867_86758

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_avg_weight : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 29 → 
  initial_avg_weight = 28 → 
  new_student_weight = 10 → 
  let total_weight := initial_students * initial_avg_weight + new_student_weight
  let new_total_students := initial_students + 1
  (total_weight / new_total_students : ℝ) = 27.4 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l867_86758


namespace NUMINAMATH_CALUDE_initial_speed_proof_l867_86701

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 10

/-- The height of the building in meters -/
def h : ℝ := 180

/-- The time taken to fall the last 60 meters in seconds -/
def t : ℝ := 1

/-- The distance fallen in the last second in meters -/
def d : ℝ := 60

/-- The initial downward speed of the object in m/s -/
def v₀ : ℝ := 25

theorem initial_speed_proof : 
  ∃ (v : ℝ), v = v₀ ∧ 
  d = (v + v₀) / 2 * t ∧ 
  v^2 = v₀^2 + 2 * g * (h - d) :=
sorry

end NUMINAMATH_CALUDE_initial_speed_proof_l867_86701


namespace NUMINAMATH_CALUDE_number_property_l867_86780

theorem number_property : ∃! x : ℝ, x - 18 = 3 * (86 - x) :=
  sorry

end NUMINAMATH_CALUDE_number_property_l867_86780


namespace NUMINAMATH_CALUDE_square_circle_overlap_ratio_l867_86765

theorem square_circle_overlap_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let square_side := 2 * r
  let overlap_area := square_side^2
  overlap_area / circle_area = 4 / π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_overlap_ratio_l867_86765


namespace NUMINAMATH_CALUDE_unique_root_condition_l867_86702

/-- The equation √(ax² + ax + 2) = ax + 2 has a unique real root if and only if a = -8 or a ≥ 1 -/
theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (a * x^2 + a * x + 2) = a * x + 2) ↔ (a = -8 ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_condition_l867_86702


namespace NUMINAMATH_CALUDE_nancy_books_l867_86791

theorem nancy_books (alyssa_books : ℕ) (nancy_multiplier : ℕ) : 
  alyssa_books = 36 → nancy_multiplier = 7 → alyssa_books * nancy_multiplier = 252 := by
  sorry

end NUMINAMATH_CALUDE_nancy_books_l867_86791


namespace NUMINAMATH_CALUDE_solve_system_l867_86723

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 14) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -1/11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l867_86723


namespace NUMINAMATH_CALUDE_intersection_points_count_l867_86798

/-- The number of intersection points between y = |3x + 6| and y = -|4x - 3| -/
theorem intersection_points_count : ∃! p : ℝ × ℝ, 
  (|3 * p.1 + 6| = p.2) ∧ (-|4 * p.1 - 3| = p.2) := by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l867_86798


namespace NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_sum_of_squares_l867_86778

theorem quadratic_equation_from_sum_and_sum_of_squares 
  (x₁ x₂ : ℝ) 
  (h_sum : x₁ + x₂ = 3) 
  (h_sum_squares : x₁^2 + x₂^2 = 5) :
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_sum_of_squares_l867_86778


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l867_86795

/-- Given a geometric sequence with sum of first n terms Sn = k · 3^n + 1, k = -1 -/
theorem geometric_sequence_sum (n : ℕ) (k : ℝ) :
  (∀ n, ∃ Sn : ℝ, Sn = k * 3^n + 1) →
  (∃ a : ℕ → ℝ, ∀ i j, i < j → a i * a j = (a i)^2) →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l867_86795


namespace NUMINAMATH_CALUDE_star_difference_l867_86741

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by sorry

end NUMINAMATH_CALUDE_star_difference_l867_86741


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l867_86755

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l867_86755


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l867_86775

theorem sum_of_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 2*n) :
  a 3 + a 4 + a 5 + a 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l867_86775


namespace NUMINAMATH_CALUDE_basketball_team_math_enrollment_l867_86781

theorem basketball_team_math_enrollment (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 12 →
  both_subjects = 5 →
  (∃ (math_players : ℕ), math_players = total_players - physics_players + both_subjects ∧ math_players = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_math_enrollment_l867_86781


namespace NUMINAMATH_CALUDE_staircase_perimeter_l867_86779

/-- Given a rectangle with a staircase-shaped region removed, 
    if the remaining area is 104 square feet, 
    then the perimeter of the remaining region is 52.4 feet. -/
theorem staircase_perimeter (width height : ℝ) (area remaining_area : ℝ) : 
  width = 10 →
  area = width * height →
  remaining_area = area - 40 →
  remaining_area = 104 →
  width + height + 3 + 5 + 20 = 52.4 :=
by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l867_86779


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_35_seconds_l867_86732

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : Real) 
  (train_speed : Real) 
  (train_length : Real) 
  (initial_lead : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_lead + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 35 seconds -/
theorem train_passes_jogger_in_35_seconds : 
  train_passing_jogger 9 45 110 240 = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_35_seconds_l867_86732


namespace NUMINAMATH_CALUDE_jennifer_fruits_left_l867_86759

/-- Calculates the number of fruits Jennifer has left after giving some to her sister. -/
def fruits_left (initial_pears initial_oranges : ℕ) (apples_multiplier : ℕ) (given_away : ℕ) : ℕ :=
  let initial_apples := initial_pears * apples_multiplier
  let remaining_pears := initial_pears - given_away
  let remaining_oranges := initial_oranges - given_away
  let remaining_apples := initial_apples - given_away
  remaining_pears + remaining_oranges + remaining_apples

/-- Theorem stating that Jennifer has 44 fruits left after giving some to her sister. -/
theorem jennifer_fruits_left : 
  fruits_left 10 20 2 2 = 44 := by sorry

end NUMINAMATH_CALUDE_jennifer_fruits_left_l867_86759


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l867_86751

theorem shaded_square_area_ratio :
  ∀ (n : ℕ) (large_square_side : ℝ) (small_square_side : ℝ),
    n = 4 →
    large_square_side = n * small_square_side →
    small_square_side > 0 →
    (2 * small_square_side^2) / (large_square_side^2) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l867_86751


namespace NUMINAMATH_CALUDE_sum_proper_divisors_243_l867_86737

theorem sum_proper_divisors_243 : 
  (Finset.filter (fun x => x ≠ 243 ∧ 243 % x = 0) (Finset.range 244)).sum id = 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_243_l867_86737


namespace NUMINAMATH_CALUDE_remainder_problem_l867_86731

theorem remainder_problem (N : ℤ) : 
  N % 899 = 63 → N % 29 = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l867_86731


namespace NUMINAMATH_CALUDE_least_divisible_by_five_smallest_primes_l867_86794

def five_smallest_primes : List Nat := [2, 3, 5, 7, 11]

def product_of_primes : Nat := five_smallest_primes.prod

theorem least_divisible_by_five_smallest_primes :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ five_smallest_primes, n % p = 0) → n ≥ product_of_primes) ∧
  (∀ p ∈ five_smallest_primes, product_of_primes % p = 0) :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_five_smallest_primes_l867_86794


namespace NUMINAMATH_CALUDE_shoe_discount_percentage_l867_86769

def original_price : ℝ := 62.50 + 3.75
def amount_saved : ℝ := 3.75
def amount_spent : ℝ := 62.50

theorem shoe_discount_percentage : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |((amount_saved / original_price) * 100 - 6)| < ε :=
sorry

end NUMINAMATH_CALUDE_shoe_discount_percentage_l867_86769


namespace NUMINAMATH_CALUDE_power_product_equals_2025_l867_86743

theorem power_product_equals_2025 (a b : ℕ) (h1 : 5^a = 3125) (h2 : 3^b = 81) :
  5^(a - 3) * 3^(2*b - 4) = 2025 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_2025_l867_86743


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l867_86749

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x * y - x - y = 3) :
  (∃ (m : ℝ), m = 9 ∧ ∀ z w, z > 0 → w > 0 → z * w - z - w = 3 → x * y ≤ z * w) ∧
  (∃ (n : ℝ), n = 6 ∧ ∀ z w, z > 0 → w > 0 → z * w - z - w = 3 → x + y ≤ z + w) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l867_86749


namespace NUMINAMATH_CALUDE_diagonal_intersection_theorem_l867_86792

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem diagonal_intersection_theorem (ABCD : Quadrilateral) (E : Point) :
  isConvex ABCD →
  distance ABCD.A ABCD.B = 9 →
  distance ABCD.C ABCD.D = 12 →
  distance ABCD.A ABCD.C = 14 →
  E = lineIntersection ABCD.A ABCD.C ABCD.B ABCD.D →
  triangleArea ABCD.A E ABCD.D = triangleArea ABCD.B E ABCD.C →
  distance ABCD.A E = 6 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_theorem_l867_86792


namespace NUMINAMATH_CALUDE_special_polynomial_inequality_l867_86715

/-- A polynomial with real coefficients that has three positive real roots and a negative value at x = 0 -/
structure SpecialPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  has_three_positive_roots : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    ∀ x, a * x^3 + b * x^2 + c * x + d = a * (x - x₁) * (x - x₂) * (x - x₃)
  negative_at_zero : d < 0

/-- The inequality holds for special polynomials -/
theorem special_polynomial_inequality (φ : SpecialPolynomial) :
  2 * φ.b^3 + 9 * φ.a^2 * φ.d - 7 * φ.a * φ.b * φ.c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_inequality_l867_86715


namespace NUMINAMATH_CALUDE_max_popsicles_is_18_l867_86709

/-- Represents the number of popsicles in a package -/
inductive Package
| Single : Package
| FourPack : Package
| SevenPack : Package
| NinePack : Package

/-- Returns the cost of a package in dollars -/
def cost (p : Package) : ℕ :=
  match p with
  | Package.Single => 2
  | Package.FourPack => 5
  | Package.SevenPack => 8
  | Package.NinePack => 10

/-- Returns the number of popsicles in a package -/
def popsicles (p : Package) : ℕ :=
  match p with
  | Package.Single => 1
  | Package.FourPack => 4
  | Package.SevenPack => 7
  | Package.NinePack => 9

/-- Represents a combination of packages -/
def Combination := List Package

/-- Calculates the total cost of a combination -/
def totalCost (c : Combination) : ℕ :=
  c.map cost |>.sum

/-- Calculates the total number of popsicles in a combination -/
def totalPopsicles (c : Combination) : ℕ :=
  c.map popsicles |>.sum

/-- Checks if a combination is within budget -/
def withinBudget (c : Combination) : Prop :=
  totalCost c ≤ 20

/-- Theorem: The maximum number of popsicles Pablo can buy with $20 is 18 -/
theorem max_popsicles_is_18 :
  ∀ c : Combination, withinBudget c → totalPopsicles c ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_is_18_l867_86709


namespace NUMINAMATH_CALUDE_terminal_side_in_fourth_quadrant_l867_86772

def angle_in_fourth_quadrant (α : Real) : Prop :=
  -2 * Real.pi < α ∧ α < -3 * Real.pi / 2

theorem terminal_side_in_fourth_quadrant :
  angle_in_fourth_quadrant (-5) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_fourth_quadrant_l867_86772


namespace NUMINAMATH_CALUDE_unique_triple_solution_l867_86725

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l867_86725


namespace NUMINAMATH_CALUDE_square_circle_radius_l867_86756

theorem square_circle_radius (square_perimeter : ℝ) (circle_radius : ℝ) : 
  square_perimeter = 28 →
  circle_radius = square_perimeter / 4 →
  circle_radius = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_radius_l867_86756


namespace NUMINAMATH_CALUDE_remainder_problem_l867_86727

theorem remainder_problem (N : ℤ) (h : N % 242 = 100) : N % 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l867_86727


namespace NUMINAMATH_CALUDE_power_of_seven_mod_four_l867_86740

theorem power_of_seven_mod_four : 7^150 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_four_l867_86740


namespace NUMINAMATH_CALUDE_payment_is_two_l867_86720

def payment_per_window (stories : ℕ) (windows_per_floor : ℕ) (subtraction_rate : ℚ)
  (days_taken : ℕ) (final_payment : ℚ) : ℚ :=
  let total_windows := stories * windows_per_floor
  let subtraction := (days_taken / 3 : ℚ) * subtraction_rate
  let original_payment := final_payment + subtraction
  original_payment / total_windows

theorem payment_is_two :
  payment_per_window 3 3 1 6 16 = 2 := by sorry

end NUMINAMATH_CALUDE_payment_is_two_l867_86720


namespace NUMINAMATH_CALUDE_evaluate_fraction_l867_86700

theorem evaluate_fraction (a b : ℤ) (h1 : a = 7) (h2 : b = -3) :
  3 / (a - b) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l867_86700


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l867_86714

/-- Fibonacci-like sequence defined by F_0 = 0, F_1 = 1, F_i = 3F_{i-1} - F_{i-2} for i ≥ 2 -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required conditions for P = 23 -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 0 = 0 ∧ F 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 2 → F n = 3 * F (n - 1) - F (n - 2)) ∧ F 12 % 23 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l867_86714


namespace NUMINAMATH_CALUDE_sammy_bottle_caps_l867_86786

theorem sammy_bottle_caps :
  ∀ (billie janine sammy : ℕ),
    billie = 2 →
    janine = 3 * billie →
    sammy = janine + 2 →
    sammy = 8 := by
  sorry

end NUMINAMATH_CALUDE_sammy_bottle_caps_l867_86786


namespace NUMINAMATH_CALUDE_percentage_deposited_approx_28_percent_l867_86747

def deposit : ℝ := 4500
def monthly_income : ℝ := 16071.42857142857

theorem percentage_deposited_approx_28_percent :
  ∃ ε > 0, ε < 0.01 ∧ |deposit / monthly_income * 100 - 28| < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_deposited_approx_28_percent_l867_86747


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l867_86703

theorem a_minus_b_equals_two (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_two_l867_86703


namespace NUMINAMATH_CALUDE_two_equidistant_points_l867_86733

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Three distinct lines in a plane -/
structure ThreeLines :=
  (l₁ : Line)
  (l₂ : Line)
  (l₃ : Line)
  (distinct : l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃)

/-- l₂ intersects l₁ -/
def intersects (l₁ l₂ : Line) : Prop :=
  l₁.slope ≠ l₂.slope

/-- l₃ is parallel to l₁ -/
def parallel (l₁ l₃ : Line) : Prop :=
  l₁.slope = l₃.slope

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A point is equidistant from three lines -/
def equidistant (p : Point) (lines : ThreeLines) : Prop := sorry

/-- The main theorem -/
theorem two_equidistant_points (lines : ThreeLines) 
  (h₁ : intersects lines.l₁ lines.l₂)
  (h₂ : parallel lines.l₁ lines.l₃) :
  ∃! (p₁ p₂ : Point), p₁ ≠ p₂ ∧ 
    equidistant p₁ lines ∧ 
    equidistant p₂ lines ∧
    ∀ (p : Point), equidistant p lines → p = p₁ ∨ p = p₂ :=
sorry

end NUMINAMATH_CALUDE_two_equidistant_points_l867_86733


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l867_86776

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l867_86776


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l867_86790

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ,
    5 - 2*x ≥ 1 →
    x + 3 > 0 →
    x + 1 ≠ 0 →
    (2 + x) * (2 - x) ≠ 0 →
    (x^2 - 4*x + 4) / (x + 1) / ((3 / (x + 1)) - x + 1) = (2 - x) / (2 + x) ∧
    (2 - 0) / (2 + 0) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l867_86790


namespace NUMINAMATH_CALUDE_system_solution_exists_l867_86783

theorem system_solution_exists : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  (4 * x₁ - 3 * y₁ = -3 ∧ 8 * x₁ + 5 * y₁ = 11 + x₁^2) ∧
  (4 * x₂ - 3 * y₂ = -3 ∧ 8 * x₂ + 5 * y₂ = 11 + x₂^2) ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_exists_l867_86783


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l867_86754

theorem fraction_sum_equals_decimal : (1 : ℚ) / 10 + 2 / 20 - 3 / 60 = (15 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l867_86754


namespace NUMINAMATH_CALUDE_function_nonnegative_implies_a_range_l867_86760

theorem function_nonnegative_implies_a_range 
  (f : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ 0) 
  (h_def : ∀ x, f x = x^2 + a*x + 3 - a) : 
  a ∈ Set.Icc (-7 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_function_nonnegative_implies_a_range_l867_86760


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_6_l867_86797

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

theorem five_digit_multiple_of_6 (d : ℕ) :
  d < 10 →
  is_multiple_of_6 (47690 + d) →
  d = 4 ∨ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_6_l867_86797


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l867_86736

theorem smallest_common_multiple_of_9_and_6 :
  ∃ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 9 ∣ m → 6 ∣ m → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l867_86736


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l867_86735

theorem sum_of_digits_8_pow_2004 : ∃ (n : ℕ), 
  8^2004 % 100 = n ∧ (n / 10 + n % 10 = 7) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l867_86735


namespace NUMINAMATH_CALUDE_middle_card_is_six_l867_86719

/-- Represents a set of three cards with positive integers -/
structure CardSet where
  left : Nat
  middle : Nat
  right : Nat
  sum_is_17 : left + middle + right = 17
  increasing : left < middle ∧ middle < right

/-- Predicate to check if a number allows for multiple possibilities when seen on the left -/
def leftIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.left = n ∧ cs2.left = n ∧ cs1 ≠ cs2

/-- Predicate to check if a number allows for multiple possibilities when seen on the right -/
def rightIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.right = n ∧ cs2.right = n ∧ cs1 ≠ cs2

/-- Predicate to check if a number allows for multiple possibilities when seen in the middle -/
def middleIndeterminate (n : Nat) : Prop :=
  ∃ (cs1 cs2 : CardSet), cs1.middle = n ∧ cs2.middle = n ∧ cs1 ≠ cs2

/-- The main theorem stating that the middle card must be 6 -/
theorem middle_card_is_six :
  ∀ (cs : CardSet),
    leftIndeterminate cs.left →
    rightIndeterminate cs.right →
    middleIndeterminate cs.middle →
    cs.middle = 6 := by
  sorry

end NUMINAMATH_CALUDE_middle_card_is_six_l867_86719


namespace NUMINAMATH_CALUDE_total_cost_calculation_l867_86799

/-- The total cost of remaining balloons for Sam and Mary -/
def total_cost (s a m c : ℝ) : ℝ :=
  ((s - a) + m) * c

/-- Theorem stating the total cost of remaining balloons for Sam and Mary -/
theorem total_cost_calculation (s a m c : ℝ) 
  (hs : s = 6) (ha : a = 5) (hm : m = 7) (hc : c = 9) : 
  total_cost s a m c = 72 := by
  sorry

#eval total_cost 6 5 7 9

end NUMINAMATH_CALUDE_total_cost_calculation_l867_86799


namespace NUMINAMATH_CALUDE_triangle_perimeter_l867_86767

/-- Proves that a triangle with inradius 2.5 cm and area 45 cm² has a perimeter of 36 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 45 → A = r * (p / 2) → p = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l867_86767


namespace NUMINAMATH_CALUDE_illumination_theorem_l867_86722

/-- Represents a rectangular room with a point light source and a mirror --/
structure IlluminatedRoom where
  length : ℝ
  width : ℝ
  height : ℝ
  mirror_width : ℝ
  light_source : ℝ × ℝ × ℝ

/-- Calculates the fraction of walls not illuminated in the room --/
def fraction_not_illuminated (room : IlluminatedRoom) : ℚ :=
  17 / 32

/-- Theorem stating that the fraction of walls not illuminated is 17/32 --/
theorem illumination_theorem (room : IlluminatedRoom) :
  fraction_not_illuminated room = 17 / 32 := by
  sorry

end NUMINAMATH_CALUDE_illumination_theorem_l867_86722


namespace NUMINAMATH_CALUDE_map_to_actual_distance_l867_86708

/-- Given a map distance between two towns and a scale factor, calculate the actual distance -/
theorem map_to_actual_distance 
  (map_distance : ℝ) 
  (scale_factor : ℝ) 
  (h1 : map_distance = 45) 
  (h2 : scale_factor = 10) : 
  map_distance * scale_factor = 450 := by
  sorry

end NUMINAMATH_CALUDE_map_to_actual_distance_l867_86708


namespace NUMINAMATH_CALUDE_rhombus_sides_equal_is_universal_and_true_l867_86705

/-- A rhombus is a quadrilateral with four equal sides --/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The proposition "All sides of a rhombus are equal" is universal and true --/
theorem rhombus_sides_equal_is_universal_and_true :
  (∀ r : Rhombus, ∀ i j : Fin 4, r.sides i = r.sides j) ∧
  (∃ r : Rhombus, True) :=
sorry

end NUMINAMATH_CALUDE_rhombus_sides_equal_is_universal_and_true_l867_86705


namespace NUMINAMATH_CALUDE_smallest_number_proof_l867_86771

def smallest_number : ℕ := 910314816600

theorem smallest_number_proof :
  (∀ i ∈ Finset.range 28, smallest_number % (i + 1) = 0) ∧
  smallest_number % 29 ≠ 0 ∧
  smallest_number % 30 ≠ 0 ∧
  (∀ n < smallest_number, 
    (∀ i ∈ Finset.range 28, n % (i + 1) = 0) →
    (n % 29 = 0 ∨ n % 30 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l867_86771


namespace NUMINAMATH_CALUDE_solution_xy_l867_86717

theorem solution_xy (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x + y ≠ 0) 
  (h3 : (x + y) / x = y / (x + y)) 
  (h4 : x = 2 * y) : 
  x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_solution_xy_l867_86717


namespace NUMINAMATH_CALUDE_job_completion_time_l867_86763

/-- The time taken for three workers to complete a job together, given their individual efficiencies -/
theorem job_completion_time 
  (sakshi_time : ℝ) 
  (tanya_efficiency : ℝ) 
  (rahul_efficiency : ℝ) 
  (h1 : sakshi_time = 20) 
  (h2 : tanya_efficiency = 1.25) 
  (h3 : rahul_efficiency = 1.5) : 
  (1 / (1 / sakshi_time + tanya_efficiency * (1 / sakshi_time) + rahul_efficiency * tanya_efficiency * (1 / sakshi_time))) = 160 / 33 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l867_86763


namespace NUMINAMATH_CALUDE_tangent_line_proof_l867_86752

-- Define the given curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the given line
def line1 (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line we want to prove
def line2 (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The point (x₀, y₀) is on line2
    line2 x₀ y₀ ∧
    -- line2 is tangent to the curve at (x₀, y₀)
    (deriv f x₀ = -3) ∧
    -- line1 and line2 are perpendicular
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      line1 x₁ y₁ → line1 x₂ y₂ → x₁ ≠ x₂ →
      line2 x₁ y₁ → line2 x₂ y₂ → x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * (y₂ - y₁) / (x₂ - x₁) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l867_86752


namespace NUMINAMATH_CALUDE_equation_solution_l867_86716

theorem equation_solution (x : ℝ) : 
  Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 2 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l867_86716


namespace NUMINAMATH_CALUDE_david_rosy_age_difference_l867_86757

/-- David and Rosy's ages problem -/
theorem david_rosy_age_difference :
  ∀ (david_age rosy_age : ℕ),
    rosy_age = 12 →
    david_age + 6 = 2 * (rosy_age + 6) →
    david_age - rosy_age = 18 :=
by sorry

end NUMINAMATH_CALUDE_david_rosy_age_difference_l867_86757


namespace NUMINAMATH_CALUDE_min_value_expression_l867_86745

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 4) :
  (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) ≥ 192 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 4 ∧
    (a₀ + 3 * b₀) * (2 * b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 192 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l867_86745


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l867_86774

/-- The number of peaches Sally picked at the orchard -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Theorem: Sally picked 42 peaches at the orchard -/
theorem sally_picked_42_peaches (initial final : ℕ) 
  (h1 : initial = 13) 
  (h2 : final = 55) : 
  peaches_picked initial final = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l867_86774


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l867_86766

theorem cubic_roots_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) → 
  (0 < b ∧ b < 1) → 
  (0 < c ∧ c < 1) → 
  a ≠ b → b ≠ c → a ≠ c →
  40 * a^3 - 70 * a^2 + 32 * a - 3 = 0 →
  40 * b^3 - 70 * b^2 + 32 * b - 3 = 0 →
  40 * c^3 - 70 * c^2 + 32 * c - 3 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l867_86766


namespace NUMINAMATH_CALUDE_whistle_cost_l867_86782

theorem whistle_cost (total_cost yoyo_cost : ℕ) (h1 : total_cost = 38) (h2 : yoyo_cost = 24) :
  total_cost - yoyo_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_whistle_cost_l867_86782


namespace NUMINAMATH_CALUDE_expansion_and_a4_imply_a_and_sum_l867_86710

/-- The expansion of (2x - a)^7 in terms of (x+1) -/
def expansion (a : ℝ) (x : ℝ) : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  λ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ =>
    a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7

theorem expansion_and_a4_imply_a_and_sum :
  ∀ a : ℝ, ∀ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ,
    (∀ x : ℝ, (2*x - a)^7 = expansion a x a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇) →
    a₄ = -560 →
    a = -1 ∧ |a₁| + |a₂| + |a₃| + |a₅| + |a₆| + |a₇| = 2186 :=
by sorry

end NUMINAMATH_CALUDE_expansion_and_a4_imply_a_and_sum_l867_86710


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l867_86734

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_area := 6 * s^2
  let new_edge := 1.25 * s
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l867_86734
