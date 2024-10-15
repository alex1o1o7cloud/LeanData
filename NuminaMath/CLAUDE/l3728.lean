import Mathlib

namespace NUMINAMATH_CALUDE_post_office_mail_count_l3728_372878

/-- The number of pieces of mail handled by a post office in six months -/
def mail_in_six_months (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) (num_months : ℕ) : ℕ :=
  (letters_per_day + packages_per_day) * (days_per_month * num_months)

/-- Theorem stating that the post office handles 14,400 pieces of mail in six months -/
theorem post_office_mail_count :
  mail_in_six_months 60 20 30 6 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_post_office_mail_count_l3728_372878


namespace NUMINAMATH_CALUDE_division_scaling_l3728_372832

theorem division_scaling (a b c : ℝ) (h : a / b = c) : (a / 100) / (b / 100) = c := by
  sorry

end NUMINAMATH_CALUDE_division_scaling_l3728_372832


namespace NUMINAMATH_CALUDE_min_value_of_a_l3728_372801

theorem min_value_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) → 
  a ≥ 4 ∧ ∀ b : ℝ, b > 0 → (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + b / y) ≥ 9) → b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3728_372801


namespace NUMINAMATH_CALUDE_parallelogram_intersection_l3728_372899

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point) (b : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Checks if a point is inside a parallelogram -/
def isInside (p : Point) (para : Parallelogram) : Prop := sorry

/-- Checks if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Checks if a point lies on a line -/
def isOnLine (p : Point) (l : Line) : Prop := sorry

/-- Checks if three lines intersect at a single point -/
def intersectAtOnePoint (l1 l2 l3 : Line) : Prop := sorry

theorem parallelogram_intersection 
  (ABCD : Parallelogram) 
  (M : Point) 
  (P Q R S : Point)
  (PR QS BS PD MC : Line)
  (h1 : isInside M ABCD)
  (h2 : isParallel PR (Line.mk ABCD.B ABCD.C))
  (h3 : isParallel QS (Line.mk ABCD.A ABCD.B))
  (h4 : isOnLine P (Line.mk ABCD.A ABCD.B))
  (h5 : isOnLine Q (Line.mk ABCD.B ABCD.C))
  (h6 : isOnLine R (Line.mk ABCD.C ABCD.D))
  (h7 : isOnLine S (Line.mk ABCD.D ABCD.A))
  (h8 : PR = Line.mk P R)
  (h9 : QS = Line.mk Q S)
  (h10 : BS = Line.mk ABCD.B S)
  (h11 : PD = Line.mk P ABCD.D)
  (h12 : MC = Line.mk M ABCD.C)
  : intersectAtOnePoint BS PD MC := sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_l3728_372899


namespace NUMINAMATH_CALUDE_zoo_trip_result_l3728_372805

def zoo_trip (initial_students_class1 initial_students_class2 parent_chaperones teachers students_left chaperones_left : ℕ) : ℕ :=
  let total_initial_students := initial_students_class1 + initial_students_class2
  let total_initial_adults := parent_chaperones + teachers
  let remaining_students := total_initial_students - students_left
  let remaining_chaperones := parent_chaperones - chaperones_left
  remaining_students + remaining_chaperones + teachers

theorem zoo_trip_result :
  zoo_trip 10 10 5 2 10 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_result_l3728_372805


namespace NUMINAMATH_CALUDE_reflection_x_axis_example_l3728_372837

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- The theorem stating that the reflection of (3, 4, -5) across the x-axis is (3, -4, 5) -/
theorem reflection_x_axis_example : 
  reflect_x_axis { x := 3, y := 4, z := -5 } = { x := 3, y := -4, z := 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_x_axis_example_l3728_372837


namespace NUMINAMATH_CALUDE_dianna_problem_l3728_372868

def correct_expression (f : ℤ) : ℤ := 1 - (2 - (3 - (4 + (5 - f))))

def misinterpreted_expression (f : ℤ) : ℤ := 1 - 2 - 3 - 4 + 5 - f

theorem dianna_problem : ∃ f : ℤ, correct_expression f = misinterpreted_expression f ∧ f = 2 := by
  sorry

end NUMINAMATH_CALUDE_dianna_problem_l3728_372868


namespace NUMINAMATH_CALUDE_S_value_S_approx_l3728_372811

/-- Define the sum S as a function of n, where n is the number of terms -/
def S (n : ℕ) : ℚ :=
  let rec aux (k : ℕ) : ℚ :=
    if k = 0 then 5005
    else (5005 - k : ℚ) + (1/2) * aux (k-1)
  aux n

/-- The main theorem stating that S(5000) is equal to 5009 - (1/2^5000) -/
theorem S_value : S 5000 = 5009 - (1/2)^5000 := by
  sorry

/-- Corollary stating that S(5000) is approximately equal to 5009 -/
theorem S_approx : abs (S 5000 - 5009) < 1 := by
  sorry

end NUMINAMATH_CALUDE_S_value_S_approx_l3728_372811


namespace NUMINAMATH_CALUDE_president_vp_from_six_l3728_372863

/-- The number of ways to choose a President and Vice-President from a group of n people -/
def choose_president_and_vp (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 30 ways to choose a President and Vice-President from 6 people -/
theorem president_vp_from_six : choose_president_and_vp 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_president_vp_from_six_l3728_372863


namespace NUMINAMATH_CALUDE_equation_solution_l3728_372809

theorem equation_solution : ∃ x : ℝ, (x / 3 + (30 - x) / 2 = 5) ∧ (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3728_372809


namespace NUMINAMATH_CALUDE_jenny_sleep_duration_l3728_372861

/-- Calculates the total minutes of sleep given the number of hours and minutes per hour. -/
def total_minutes_of_sleep (hours : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  hours * minutes_per_hour

/-- Proves that 8 hours of sleep is equivalent to 480 minutes. -/
theorem jenny_sleep_duration :
  total_minutes_of_sleep 8 60 = 480 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sleep_duration_l3728_372861


namespace NUMINAMATH_CALUDE_final_surface_area_l3728_372821

/-- Represents the cube structure after modifications --/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCornerCubes : Nat
  removedCentralCube : Nat
  removedCenterUnits : Bool

/-- Calculates the surface area of the modified cube structure --/
def surfaceArea (cube : ModifiedCube) : Nat :=
  let totalSmallCubes := (cube.initialSize / cube.smallCubeSize) ^ 3
  let remainingCubes := totalSmallCubes - cube.removedCornerCubes - cube.removedCentralCube
  let initialSurfaceArea := remainingCubes * (6 * cube.smallCubeSize ^ 2)
  let additionalInternalSurface := if cube.removedCenterUnits then remainingCubes * 6 else 0
  initialSurfaceArea + additionalInternalSurface

/-- The main theorem stating the surface area of the final structure --/
theorem final_surface_area :
  let cube : ModifiedCube := {
    initialSize := 12,
    smallCubeSize := 3,
    removedCornerCubes := 8,
    removedCentralCube := 1,
    removedCenterUnits := true
  }
  surfaceArea cube = 3300 := by
  sorry

end NUMINAMATH_CALUDE_final_surface_area_l3728_372821


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l3728_372820

theorem largest_square_tile_size 
  (length width : ℕ) 
  (h_length : length = 378) 
  (h_width : width = 595) : 
  ∃ (tile_size : ℕ), 
    tile_size = Nat.gcd length width ∧ 
    tile_size = 7 ∧
    length % tile_size = 0 ∧ 
    width % tile_size = 0 ∧
    ∀ (larger_size : ℕ), larger_size > tile_size → 
      length % larger_size ≠ 0 ∨ width % larger_size ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l3728_372820


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3728_372881

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let a : ℝ := 3
  let b : ℝ := -12
  let c : ℝ := 9
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3728_372881


namespace NUMINAMATH_CALUDE_leanna_money_l3728_372847

/-- The amount of money Leanna has to spend -/
def total_money : ℕ := 37

/-- The price of a CD -/
def cd_price : ℕ := 14

/-- The price of a cassette -/
def cassette_price : ℕ := 9

/-- Leanna can spend all her money on two CDs and a cassette -/
axiom scenario1 : 2 * cd_price + cassette_price = total_money

/-- Leanna can buy one CD and two cassettes and have $5 left over -/
axiom scenario2 : cd_price + 2 * cassette_price + 5 = total_money

theorem leanna_money : total_money = 37 := by
  sorry

end NUMINAMATH_CALUDE_leanna_money_l3728_372847


namespace NUMINAMATH_CALUDE_problem_solution_l3728_372835

noncomputable section

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ, Real.cos θ)
def O : ℝ × ℝ := (0, 0)

def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem problem_solution (θ : ℝ) :
  (vec_length (vec A (C θ)) = vec_length (vec B (C θ)) → Real.tan θ = 1/2) ∧
  (dot_product (vec O A + 2 • vec O B) (vec O (C θ)) = 1 → Real.sin θ * Real.cos θ = -3/8) := by
  sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3728_372835


namespace NUMINAMATH_CALUDE_parabola_intersection_l3728_372827

/-- Prove that for a parabola y² = 2px (p > 0) with focus F on the x-axis, 
    if a line with slope angle π/4 passes through F and intersects the parabola at points A and B, 
    and the perpendicular bisector of AB passes through (0, 2), then p = 4/5. -/
theorem parabola_intersection (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∃ F : ℝ × ℝ, F.2 = 0 ∧ F.1 = p / 2) →
  (∀ x y : ℝ, y ^ 2 = 2 * p * x) →
  (∃ m b : ℝ, m = 1 ∧ A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b) →
  (A.2 ^ 2 = 2 * p * A.1 ∧ B.2 ^ 2 = 2 * p * B.1) →
  ((A.1 + B.1) / 2 = 3 * p / 2 ∧ (A.2 + B.2) / 2 = p) →
  (∃ m' b' : ℝ, m' = -1 ∧ 2 = m' * 0 + b') →
  p = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3728_372827


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3728_372853

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x > 0 → 3 * x + 1 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3728_372853


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3728_372848

theorem initial_money_calculation (initial_money : ℚ) : 
  (2/5 : ℚ) * initial_money = 200 → initial_money = 500 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3728_372848


namespace NUMINAMATH_CALUDE_line_vector_coefficient_l3728_372893

/-- Given vectors a and b in a real vector space, if k*a + (2/5)*b lies on the line
    passing through a and b, then k = 3/5 -/
theorem line_vector_coefficient (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V]
  (a b : V) (k : ℝ) :
  (∃ t : ℝ, k • a + (2/5) • b = a + t • (b - a)) →
  k = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_coefficient_l3728_372893


namespace NUMINAMATH_CALUDE_balls_per_color_l3728_372839

theorem balls_per_color 
  (total_balls : ℕ) 
  (num_colors : ℕ) 
  (h1 : total_balls = 350) 
  (h2 : num_colors = 10) 
  (h3 : total_balls % num_colors = 0) : 
  total_balls / num_colors = 35 := by
sorry

end NUMINAMATH_CALUDE_balls_per_color_l3728_372839


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3728_372883

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 5*x^2 + 20*x + 45 → y ≥ y_min ∧ y_min = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3728_372883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3728_372874

/-- An arithmetic sequence with first term 11, last term 101, and common difference 5 has 19 terms. -/
theorem arithmetic_sequence_terms : ∀ (a : ℕ → ℕ),
  (a 0 = 11) →  -- First term is 11
  (∀ n, a (n + 1) - a n = 5) →  -- Common difference is 5
  (∃ k, a k = 101) →  -- Last term is 101
  (∃ k, k = 19 ∧ a (k - 1) = 101) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3728_372874


namespace NUMINAMATH_CALUDE_unused_streetlights_l3728_372826

/-- Given the number of streetlights bought by the New York City Council, 
    the number of squares in New York, and the number of streetlights per square, 
    calculate the number of unused streetlights. -/
theorem unused_streetlights (total : ℕ) (squares : ℕ) (per_square : ℕ) 
    (h1 : total = 200) (h2 : squares = 15) (h3 : per_square = 12) : 
  total - squares * per_square = 20 := by
  sorry

end NUMINAMATH_CALUDE_unused_streetlights_l3728_372826


namespace NUMINAMATH_CALUDE_balloon_cost_difference_l3728_372892

/-- The cost of a helium balloon in dollars -/
def helium_cost : ℚ := 1.50

/-- The cost of a foil balloon in dollars -/
def foil_cost : ℚ := 2.50

/-- The number of helium balloons Allan bought -/
def allan_helium : ℕ := 2

/-- The number of foil balloons Allan bought -/
def allan_foil : ℕ := 3

/-- The number of helium balloons Jake bought -/
def jake_helium : ℕ := 4

/-- The number of foil balloons Jake bought -/
def jake_foil : ℕ := 2

/-- The total cost of Allan's balloons -/
def allan_total : ℚ := allan_helium * helium_cost + allan_foil * foil_cost

/-- The total cost of Jake's balloons -/
def jake_total : ℚ := jake_helium * helium_cost + jake_foil * foil_cost

/-- Theorem stating the difference in cost between Jake's and Allan's balloons -/
theorem balloon_cost_difference : jake_total - allan_total = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_balloon_cost_difference_l3728_372892


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l3728_372856

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the focus of the hyperbola
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y + x = 0

-- Theorem stating the distance from focus to asymptote
theorem distance_focus_to_asymptote :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (x y : ℝ), C x y → asymptote x y →
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l3728_372856


namespace NUMINAMATH_CALUDE_one_fifth_of_number_l3728_372802

theorem one_fifth_of_number (x : ℚ) : (3/10 : ℚ) * x = 12 → (1/5 : ℚ) * x = 8 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_number_l3728_372802


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3728_372857

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 ≥ m^2 - 3*m) → 
  m < 1 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3728_372857


namespace NUMINAMATH_CALUDE_intersection_M_N_l3728_372869

def M : Set ℕ := {1, 2, 3, 4}

def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

theorem intersection_M_N : M ∩ N = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3728_372869


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l3728_372810

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is (13/7, 41/14, 55/7) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (13/7, 41/14, 55/7) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l3728_372810


namespace NUMINAMATH_CALUDE_ball_count_theorem_l3728_372896

theorem ball_count_theorem (red_balls : ℕ) (white_balls : ℕ) (total_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 4 →
  total_balls = red_balls + white_balls →
  prob_red = 1/4 →
  (red_balls : ℚ) / total_balls = prob_red →
  white_balls = 12 := by
sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l3728_372896


namespace NUMINAMATH_CALUDE_investment_growth_l3728_372823

/-- Represents the initial investment amount in dollars -/
def initial_investment : ℝ := 295097.57

/-- Represents the future value in dollars -/
def future_value : ℝ := 600000

/-- Represents the annual interest rate as a decimal -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of compounding periods per year -/
def compounding_periods_per_year : ℕ := 2

/-- Represents the number of years for the investment -/
def investment_years : ℕ := 12

/-- Theorem stating that the initial investment grows to the future value 
    under the given conditions -/
theorem investment_growth (ε : ℝ) (h_ε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  |future_value - initial_investment * (1 + annual_interest_rate / compounding_periods_per_year) ^ (compounding_periods_per_year * investment_years)| < δ ∧
  δ < ε :=
sorry

end NUMINAMATH_CALUDE_investment_growth_l3728_372823


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3728_372870

def sachin_age : ℚ := 24.5
def age_difference : ℕ := 7

theorem age_ratio_proof :
  let rahul_age : ℚ := sachin_age + age_difference
  (sachin_age / rahul_age) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3728_372870


namespace NUMINAMATH_CALUDE_hyperbola_y_coordinate_comparison_l3728_372849

/-- Given two points on a hyperbola, prove that the y-coordinate of the point with smaller x-coordinate is greater -/
theorem hyperbola_y_coordinate_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h_positive : k > 0)
  (h_point_A : y₁ = k / 2)
  (h_point_B : y₂ = k / 3) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_y_coordinate_comparison_l3728_372849


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l3728_372862

/-- Given a parabola E and a line intersecting it, prove properties about the parabola equation and slopes of lines connecting intersection points to a fixed point. -/
theorem parabola_and_line_intersection (p m : ℝ) (h_p : p > 0) : 
  let E := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let L := {(x, y) : ℝ × ℝ | x = m*y + 3}
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let C := (-3, 0)
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (A ∈ E ∧ A ∈ L) ∧ 
    (B ∈ E ∧ B ∈ L) ∧ 
    (x₁ * x₂ + y₁ * y₂ = 6) →
    (p = 1/2) ∧
    (let k₁ := (y₁ - 0) / (x₁ - (-3))
     let k₂ := (y₂ - 0) / (x₂ - (-3))
     1/k₁^2 + 1/k₂^2 - 2*m^2 = 24) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l3728_372862


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3728_372812

/-- Theorem: For a parallelogram with area 216 cm² and height 18 cm, the base length is 12 cm. -/
theorem parallelogram_base_length
  (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 216)
  (h_height : height = 18)
  (h_parallelogram : area = base * height) :
  base = 12 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3728_372812


namespace NUMINAMATH_CALUDE_max_value_of_f_l3728_372851

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem max_value_of_f (a : ℝ) :
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f a x ≥ f a 2) →
  (∃ x, f a x = 18 ∧ ∀ y, f a y ≤ f a x) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3728_372851


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l3728_372859

/-- The radius of the inscribed circle of a triangle with sides 3, 4, and 5 is 1 -/
theorem inscribed_circle_radius_345_triangle : 
  ∀ (a b c : ℝ) (r : ℝ), 
    a = 3 ∧ b = 4 ∧ c = 5 →
    (a + b + c) / 2 = 6 →
    r = 6 / ((a + b + c) / 2) →
    r = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l3728_372859


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l3728_372819

theorem sum_of_squares_divisible_by_seven (x y : ℤ) : 
  (7 ∣ x^2 + y^2) → (7 ∣ x) ∧ (7 ∣ y) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l3728_372819


namespace NUMINAMATH_CALUDE_parallelogram_area_l3728_372888

theorem parallelogram_area (base : ℝ) (slant_height : ℝ) (angle : ℝ) :
  base = 10 →
  slant_height = 6 →
  angle = 30 * π / 180 →
  base * (slant_height * Real.sin angle) = 30 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3728_372888


namespace NUMINAMATH_CALUDE_function_representation_flexibility_l3728_372850

-- Define a function type
def Function (α : Type) (β : Type) := α → β

-- State the theorem
theorem function_representation_flexibility 
  {α β : Type} (f : Function α β) : 
  ¬ (∀ (formula : α → β), f = formula) :=
sorry

end NUMINAMATH_CALUDE_function_representation_flexibility_l3728_372850


namespace NUMINAMATH_CALUDE_right_triangle_median_hypotenuse_l3728_372894

/-- 
A right triangle with hypotenuse length 6 has a median to the hypotenuse of length 3.
-/
theorem right_triangle_median_hypotenuse : 
  ∀ (a b c : ℝ), 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  c = 6 →           -- Hypotenuse length is 6
  ∃ (m : ℝ),        -- There exists a median m
    m^2 = (a^2 + b^2) / 4 ∧  -- Median formula
    m = 3 :=        -- Median length is 3
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_hypotenuse_l3728_372894


namespace NUMINAMATH_CALUDE_two_lines_exist_l3728_372829

-- Define the lines given in the problem
def line_l1 (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0
def line_l2 (x y : ℝ) : Prop := x + y + 2 = 0
def line_perp (x y : ℝ) : Prop := 2 * x - y + 7 = 0

-- Define the intersection point of l1 and l2
def intersection_point : ℝ × ℝ := (-1, -1)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 1)

-- Define the equations of the lines we need to prove
def line_L1 (x y : ℝ) : Prop := x + 2 * y + 3 = 0
def line_L2 (x y : ℝ) : Prop := x - 3 * y + 6 = 0

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the theorem
theorem two_lines_exist :
  ∃ (L1 L2 : ℝ → ℝ → Prop),
    (∀ x y, line_l1 x y ∧ line_l2 x y → L1 x y) ∧
    (∃ m1 m2, perpendicular m1 m2 ∧
      (∀ x y, line_perp x y ↔ y = m1 * x + 7/2) ∧
      (∀ x y, L1 x y ↔ y = m2 * x + (intersection_point.2 - m2 * intersection_point.1))) ∧
    L1 = line_L1 ∧
    L2 given_point.1 given_point.2 ∧
    (∃ a b, a + b = -4 ∧ ∀ x y, L2 x y ↔ x / a + y / b = 1) ∧
    L2 = line_L2 :=
  sorry

end NUMINAMATH_CALUDE_two_lines_exist_l3728_372829


namespace NUMINAMATH_CALUDE_company_workforce_l3728_372898

theorem company_workforce (initial_total : ℕ) : 
  (initial_total * 60 = initial_total * 100 * 60 / 100) →
  ((initial_total + 24) * 55 = (initial_total * 60) * 100 / (initial_total + 24)) →
  (initial_total + 24 = 288) := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l3728_372898


namespace NUMINAMATH_CALUDE_BI_length_is_15_over_4_l3728_372895

/-- Two squares ABCD and EFGH with parallel sides -/
structure ParallelSquares :=
  (A B C D E F G H : ℝ × ℝ)

/-- Point where CG intersects BD -/
def I (squares : ParallelSquares) : ℝ × ℝ := sorry

/-- Length of BD -/
def BD_length (squares : ParallelSquares) : ℝ := 10

/-- Area of triangle BFC -/
def area_BFC (squares : ParallelSquares) : ℝ := 3

/-- Area of triangle CHD -/
def area_CHD (squares : ParallelSquares) : ℝ := 5

/-- Length of BI -/
def BI_length (squares : ParallelSquares) : ℝ := sorry

theorem BI_length_is_15_over_4 (squares : ParallelSquares) :
  BI_length squares = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_BI_length_is_15_over_4_l3728_372895


namespace NUMINAMATH_CALUDE_sound_engineer_selection_probability_l3728_372880

theorem sound_engineer_selection_probability :
  let total_candidates : ℕ := 5
  let selected_engineers : ℕ := 3
  let specific_engineers : ℕ := 2

  let total_combinations := Nat.choose total_candidates selected_engineers
  let favorable_outcomes := 
    Nat.choose specific_engineers 1 * Nat.choose (total_candidates - specific_engineers) (selected_engineers - 1) +
    Nat.choose specific_engineers 2 * Nat.choose (total_candidates - specific_engineers) (selected_engineers - 2)

  (favorable_outcomes : ℚ) / total_combinations = 9 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_sound_engineer_selection_probability_l3728_372880


namespace NUMINAMATH_CALUDE_fewer_twos_to_hundred_l3728_372845

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fewer_twos_to_hundred_l3728_372845


namespace NUMINAMATH_CALUDE_exactly_one_true_l3728_372816

-- Define the three propositions
def prop1 : Prop := ∀ x : ℝ, x^4 > x^2

def prop2 : Prop := (∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q))

def prop3 : Prop := (¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0))

-- Theorem statement
theorem exactly_one_true : (prop1 ∨ prop2 ∨ prop3) ∧ ¬(prop1 ∧ prop2) ∧ ¬(prop1 ∧ prop3) ∧ ¬(prop2 ∧ prop3) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l3728_372816


namespace NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a_l3728_372875

theorem a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a :
  (∀ a : ℝ, a > 2 → a^2 > 2*a) ∧
  (∃ a : ℝ, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a_l3728_372875


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_minimized_l3728_372803

/-- The eccentricity of an ellipse passing through (3, 2) when a² + b² is minimized -/
theorem ellipse_eccentricity_minimized (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : (3:ℝ)^2 / a^2 + (2:ℝ)^2 / b^2 = 1) :
  let e := Real.sqrt (1 - b^2 / a^2)
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' > b' → (3:ℝ)^2 / a'^2 + (2:ℝ)^2 / b'^2 = 1 →
    a^2 + b^2 ≤ a'^2 + b'^2) →
  e = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_minimized_l3728_372803


namespace NUMINAMATH_CALUDE_triangle_min_ab_value_l3728_372842

theorem triangle_min_ab_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (2 * c * Real.cos B = 2 * a + b) →
  (1 / 2 * c = 1 / 2 * a * b * Real.sin C) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' * b' ≥ a * b) →
  a * b = 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_min_ab_value_l3728_372842


namespace NUMINAMATH_CALUDE_customers_in_us_l3728_372885

theorem customers_in_us (total : ℕ) (other_countries : ℕ) (h1 : total = 7422) (h2 : other_countries = 6699) :
  total - other_countries = 723 := by
  sorry

end NUMINAMATH_CALUDE_customers_in_us_l3728_372885


namespace NUMINAMATH_CALUDE_marco_new_cards_l3728_372836

/-- Given a total number of cards, calculate the number of new cards obtained by trading
    one-fifth of the duplicate cards, where duplicates are one-fourth of the total. -/
def new_cards (total : ℕ) : ℕ :=
  let duplicates := total / 4
  duplicates / 5

theorem marco_new_cards :
  new_cards 500 = 25 := by sorry

end NUMINAMATH_CALUDE_marco_new_cards_l3728_372836


namespace NUMINAMATH_CALUDE_rectangle_area_inequality_l3728_372855

theorem rectangle_area_inequality : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 16 * 10 = 23 * 7 + ε := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_inequality_l3728_372855


namespace NUMINAMATH_CALUDE_sphere_cross_section_area_l3728_372813

theorem sphere_cross_section_area (R d : ℝ) (h1 : R = 3) (h2 : d = 2) :
  let r := (R^2 - d^2).sqrt
  π * r^2 = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_cross_section_area_l3728_372813


namespace NUMINAMATH_CALUDE_correct_article_usage_l3728_372877

/-- Represents the possible article choices --/
inductive Article
  | A
  | The
  | None

/-- Represents a sentence with two article slots --/
structure Sentence where
  firstArticle : Article
  secondArticle : Article

/-- Checks if the article usage is correct for the given sentence --/
def isCorrectArticleUsage (s : Sentence) : Prop :=
  s.firstArticle = Article.A ∧ s.secondArticle = Article.None

/-- Theorem stating that the correct article usage is "a" for the first blank and no article for the second --/
theorem correct_article_usage :
  ∃ (s : Sentence), isCorrectArticleUsage s :=
sorry


end NUMINAMATH_CALUDE_correct_article_usage_l3728_372877


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l3728_372858

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 3
  f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l3728_372858


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3728_372866

theorem rectangle_max_area (p : ℝ) (h1 : p = 40) : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ w = 2 * l ∧ l * w = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3728_372866


namespace NUMINAMATH_CALUDE_odd_function_graph_point_l3728_372890

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_graph_point (f : ℝ → ℝ) (a : ℝ) :
  is_odd_function f → f (-a) = -f a :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_graph_point_l3728_372890


namespace NUMINAMATH_CALUDE_part_one_part_two_l3728_372891

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, (2 < x ∧ x < 3) → ∃ a : ℝ, a > 0 ∧ a < x ∧ x < 3*a) →
  ∃ a : ℝ, 1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3728_372891


namespace NUMINAMATH_CALUDE_limit_sqrt_minus_one_over_x_l3728_372807

theorem limit_sqrt_minus_one_over_x (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = (1 - Real.sqrt (x + 1)) / x) :
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds (-1/2)) := by
sorry

end NUMINAMATH_CALUDE_limit_sqrt_minus_one_over_x_l3728_372807


namespace NUMINAMATH_CALUDE_football_angles_l3728_372817

-- Define the football structure
structure Football :=
  (edge_length : ℝ)
  (pentagon_sides : ℕ)
  (hexagon_sides : ℕ)
  (pentagons_per_hexagon : ℕ)

-- Define the angles between faces
def angle_between_hexagons (f : Football) : ℝ := sorry
def angle_between_hexagon_and_pentagon (f : Football) : ℝ := sorry

-- Theorem statement
theorem football_angles 
  (f : Football) 
  (h1 : f.edge_length = 1)
  (h2 : f.pentagon_sides = 5)
  (h3 : f.hexagon_sides = 6)
  (h4 : f.pentagons_per_hexagon = 5) :
  ∃ (α β : ℝ), 
    α = angle_between_hexagons f ∧
    β = angle_between_hexagon_and_pentagon f ∧
    ∃ (t1 t2 : ℝ → ℝ), 
      (t1 = Real.tan) ∧ 
      (t2 = Real.tan) ∧
      (t1 α = (Real.sqrt (3 * 3 - 2 * 2)) / 2) ∧
      (t2 β = (Real.sqrt (5 - 2 * Real.sqrt 5)) / (3 - Real.sqrt 5)) :=
sorry

end NUMINAMATH_CALUDE_football_angles_l3728_372817


namespace NUMINAMATH_CALUDE_james_shoe_purchase_l3728_372846

theorem james_shoe_purchase (price1 price2 : ℝ) : 
  price1 = 40 →
  price2 = 60 →
  let cheaper_price := min price1 price2
  let discounted_price2 := price2 - cheaper_price / 2
  let total_before_extra_discount := price1 + discounted_price2
  let extra_discount := total_before_extra_discount / 4
  let final_price := total_before_extra_discount - extra_discount
  final_price = 45 := by sorry

end NUMINAMATH_CALUDE_james_shoe_purchase_l3728_372846


namespace NUMINAMATH_CALUDE_ten_streets_intersections_l3728_372882

/-- Represents a city with straight streets -/
structure City where
  num_streets : ℕ
  no_parallel_streets : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (city : City) : ℕ :=
  if city.num_streets ≤ 1 then 0
  else (city.num_streets - 1) * (city.num_streets - 2) / 2

/-- Theorem: A city with 10 straight streets where no two are parallel has 45 intersections -/
theorem ten_streets_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel_streets = true →
  max_intersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_streets_intersections_l3728_372882


namespace NUMINAMATH_CALUDE_election_winner_margin_l3728_372838

theorem election_winner_margin (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (992 : ℚ) / total_votes = 62 / 100) : 
  992 - (total_votes - 992) = 384 := by
sorry

end NUMINAMATH_CALUDE_election_winner_margin_l3728_372838


namespace NUMINAMATH_CALUDE_delta_value_l3728_372865

-- Define the simultaneous equations
def simultaneous_equations (x y z : ℝ) : Prop :=
  x - y - z = -1 ∧ y - x - z = -2 ∧ z - x - y = -4

-- Define β
def β : ℕ := 5

-- Define γ
def γ : ℕ := 2

-- Define the polynomial equation
def polynomial_equation (a b δ : ℝ) : Prop :=
  ∃ t : ℝ, t^4 + a*t^2 + b*t + δ = 0 ∧
           1^4 + a*1^2 + b*1 + δ = 0 ∧
           γ^4 + a*γ^2 + b*γ + δ = 0 ∧
           (γ^2)^4 + a*(γ^2)^2 + b*(γ^2) + δ = 0

-- Theorem statement
theorem delta_value :
  ∃ x y z a b : ℝ,
    simultaneous_equations x y z →
    polynomial_equation a b (-56) :=
sorry

end NUMINAMATH_CALUDE_delta_value_l3728_372865


namespace NUMINAMATH_CALUDE_tree_height_problem_l3728_372887

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 20 →  -- One tree is 20 feet taller than the other
  h₁ / h₂ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 50  -- The shorter tree is 50 feet tall
:= by sorry

end NUMINAMATH_CALUDE_tree_height_problem_l3728_372887


namespace NUMINAMATH_CALUDE_dodgeball_team_size_l3728_372804

/-- Given a dodgeball team with the following conditions:
  * The team scored 39 points total
  * One player (Emily) scored 23 points
  * Everyone else scored 2 points each
  This theorem proves that the total number of players on the team is 9. -/
theorem dodgeball_team_size :
  ∀ (total_points : ℕ) (emily_points : ℕ) (points_per_other : ℕ),
    total_points = 39 →
    emily_points = 23 →
    points_per_other = 2 →
    ∃ (team_size : ℕ),
      team_size = (total_points - emily_points) / points_per_other + 1 ∧
      team_size = 9 :=
by sorry

end NUMINAMATH_CALUDE_dodgeball_team_size_l3728_372804


namespace NUMINAMATH_CALUDE_boy_late_to_school_l3728_372886

/-- Proves that a boy traveling to school was 1 hour late on the first day given specific conditions -/
theorem boy_late_to_school (distance : ℝ) (speed_day1 speed_day2 : ℝ) (early_time : ℝ) : 
  distance = 60 ∧ 
  speed_day1 = 10 ∧ 
  speed_day2 = 20 ∧ 
  early_time = 1 ∧
  distance / speed_day2 + early_time = distance / speed_day1 - 1 →
  distance / speed_day1 - (distance / speed_day2 + early_time) = 1 :=
by
  sorry

#check boy_late_to_school

end NUMINAMATH_CALUDE_boy_late_to_school_l3728_372886


namespace NUMINAMATH_CALUDE_longest_length_is_three_smallest_square_is_1444_l3728_372806

/-- A number is a perfect square with n identical non-zero last digits if it's
    a square and its last n digits in base 10 are the same and non-zero. -/
def is_perfect_square_with_n_identical_last_digits (x n : ℕ) : Prop :=
  ∃ k : ℕ, x = k^2 ∧
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧
  ∀ i : ℕ, i < n → (x / 10^i) % 10 = d

/-- The longest possible length for which a perfect square ends with
    n identical non-zero digits is 3. -/
theorem longest_length_is_three :
  (∀ n : ℕ, ∃ x : ℕ, is_perfect_square_with_n_identical_last_digits x n) →
  (∀ m : ℕ, m > 3 → ¬∃ x : ℕ, is_perfect_square_with_n_identical_last_digits x m) :=
sorry

/-- The smallest perfect square with 3 identical non-zero last digits is 1444. -/
theorem smallest_square_is_1444 :
  is_perfect_square_with_n_identical_last_digits 1444 3 ∧
  ∀ x : ℕ, x < 1444 → ¬is_perfect_square_with_n_identical_last_digits x 3 :=
sorry

end NUMINAMATH_CALUDE_longest_length_is_three_smallest_square_is_1444_l3728_372806


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3728_372814

/-- Given a function f: ℝ → ℝ with a tangent line at x=1 described by the equation 3x+y-4=0,
    prove that f(1) + f'(1) = -2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y : ℝ, y = f x → (3 * 1 + f 1 - 4 = 0 ∧ 3 * x + y - 4 = 0)) : 
    f 1 + (deriv f) 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3728_372814


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3728_372831

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3728_372831


namespace NUMINAMATH_CALUDE_anne_cleaning_time_l3728_372867

variable (B A C : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + C = 1/4
def condition2 : Prop := B + 2*A + 3*C = 1/3
def condition3 : Prop := B + C = 1/6

-- Theorem statement
theorem anne_cleaning_time 
  (h1 : condition1 B A C) 
  (h2 : condition2 B A C) 
  (h3 : condition3 B C) : 
  1/A = 12 := by
sorry

end NUMINAMATH_CALUDE_anne_cleaning_time_l3728_372867


namespace NUMINAMATH_CALUDE_parabola_intersection_locus_l3728_372808

/-- Given a parabola y² = 2px with vertex at the origin, 
    prove that the locus of intersection points forms another parabola -/
theorem parabola_intersection_locus (p : ℝ) (h : p > 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ x y : ℝ, y ^ 2 = 2 * p * x → 
      ∃ (x₁ y₁ : ℝ), 
        y₁ ^ 2 = 2 * p * x₁ ∧ 
        (y - y₁) = -(y₁ / p) * (x - x₁) ∧
        y = (p / y₁) * (x - p / 2) ∧
        f x = y) ∧
    (∀ x : ℝ, (f x) ^ 2 = (p / 2) * (x - p / 2)) := by
  sorry


end NUMINAMATH_CALUDE_parabola_intersection_locus_l3728_372808


namespace NUMINAMATH_CALUDE_plane_equation_l3728_372833

theorem plane_equation (s t x y z : ℝ) : 
  (∃ (s t : ℝ), x = 3 + 2*s - 3*t ∧ y = 1 + s ∧ z = 4 - 3*s + t) ↔ 
  (x - 7*y + 3*z - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_l3728_372833


namespace NUMINAMATH_CALUDE_arithmetic_fraction_difference_l3728_372828

theorem arithmetic_fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_fraction_difference_l3728_372828


namespace NUMINAMATH_CALUDE_age_of_17th_student_l3728_372843

theorem age_of_17th_student
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_students_group1 : Nat)
  (average_age_group1 : ℝ)
  (num_students_group2 : Nat)
  (average_age_group2 : ℝ)
  (h1 : total_students = 17)
  (h2 : average_age_all = 17)
  (h3 : num_students_group1 = 5)
  (h4 : average_age_group1 = 14)
  (h5 : num_students_group2 = 9)
  (h6 : average_age_group2 = 16) :
  ℝ := by
  sorry

#check age_of_17th_student

end NUMINAMATH_CALUDE_age_of_17th_student_l3728_372843


namespace NUMINAMATH_CALUDE_vector_properties_l3728_372889

/-- Given two vectors a and b in ℝ³, prove properties about their components --/
theorem vector_properties (a b : ℝ × ℝ × ℝ) :
  let x := a.2.2
  let y := b.2.1
  (a = (2, 4, x) ∧ ‖a‖ = 6) →
  (x = 4 ∨ x = -4) ∧
  (a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ (k : ℝ), a = k • b) →
  x + y = 6 := by sorry

end NUMINAMATH_CALUDE_vector_properties_l3728_372889


namespace NUMINAMATH_CALUDE_value_of_a_l3728_372854

-- Define the operation *
def star (x y : ℝ) : ℝ := x + y - x * y

-- Define a
def a : ℝ := star 1 (star 0 1)

-- Theorem statement
theorem value_of_a : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3728_372854


namespace NUMINAMATH_CALUDE_shortest_player_height_l3728_372872

theorem shortest_player_height (tallest_height : Float) (height_difference : Float) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height - height_difference = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_shortest_player_height_l3728_372872


namespace NUMINAMATH_CALUDE_elective_schemes_count_l3728_372884

def total_courses : ℕ := 10
def courses_to_choose : ℕ := 3
def conflicting_courses : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem elective_schemes_count :
  (choose conflicting_courses 1 * choose (total_courses - conflicting_courses) (courses_to_choose - 1)) +
  (choose (total_courses - conflicting_courses) courses_to_choose) = 98 :=
by sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l3728_372884


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3728_372852

-- Define the equations of the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3728_372852


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3728_372841

-- Define the surface area of the cube
def surface_area : ℝ := 150

-- Theorem stating that a cube with surface area 150 has volume 125
theorem cube_volume_from_surface_area :
  ∃ (s : ℝ), s > 0 ∧ 6 * s^2 = surface_area ∧ s^3 = 125 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3728_372841


namespace NUMINAMATH_CALUDE_kozlov_inequality_l3728_372818

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_kozlov_inequality_l3728_372818


namespace NUMINAMATH_CALUDE_love_betty_jane_l3728_372864

variable (A B : Prop)

theorem love_betty_jane : ((A → B) → A) → (A ∧ (B ∨ ¬B)) :=
  sorry

end NUMINAMATH_CALUDE_love_betty_jane_l3728_372864


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3728_372834

theorem triangle_angle_measure (D E F : ℝ) : 
  D = E →                         -- Two angles are congruent
  F = D + 40 →                    -- One angle is 40 degrees more than the congruent angles
  D + E + F = 180 →               -- Sum of angles in a triangle is 180 degrees
  F = 86.67 :=                    -- The measure of angle F is 86.67 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3728_372834


namespace NUMINAMATH_CALUDE_total_matches_is_seventeen_l3728_372825

/-- Calculates the number of matches in a round-robin tournament for n teams -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents a football competition with the given structure -/
structure FootballCompetition where
  totalTeams : ℕ
  groupSize : ℕ
  numGroups : ℕ
  semiFinalistPerGroup : ℕ
  semiFinalsLegs : ℕ
  finalMatches : ℕ

/-- Calculates the total number of matches in the competition -/
def totalMatches (comp : FootballCompetition) : ℕ :=
  (comp.numGroups * roundRobinMatches comp.groupSize) +
  (comp.numGroups * comp.semiFinalistPerGroup * comp.semiFinalsLegs / 2) +
  comp.finalMatches

/-- The specific football competition described in the problem -/
def specificCompetition : FootballCompetition :=
  { totalTeams := 8
  , groupSize := 4
  , numGroups := 2
  , semiFinalistPerGroup := 2
  , semiFinalsLegs := 2
  , finalMatches := 1 }

theorem total_matches_is_seventeen :
  totalMatches specificCompetition = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_is_seventeen_l3728_372825


namespace NUMINAMATH_CALUDE_inequality_proof_l3728_372815

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq : 1/a + 1/b + 1/c ≥ a + b + c) :
  a + b + c ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3728_372815


namespace NUMINAMATH_CALUDE_sean_needs_six_packs_l3728_372822

def bedroom_bulbs : ℕ := 2
def bathroom_bulbs : ℕ := 1
def kitchen_bulbs : ℕ := 1
def basement_bulbs : ℕ := 4
def bulbs_per_pack : ℕ := 2

def total_non_garage_bulbs : ℕ := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

def garage_bulbs : ℕ := total_non_garage_bulbs / 2

def total_bulbs : ℕ := total_non_garage_bulbs + garage_bulbs

theorem sean_needs_six_packs : (total_bulbs + bulbs_per_pack - 1) / bulbs_per_pack = 6 := by
  sorry

end NUMINAMATH_CALUDE_sean_needs_six_packs_l3728_372822


namespace NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l3728_372830

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  /-- The length of PB -/
  pb : ℝ
  /-- The length of BQ -/
  bq : ℝ
  /-- The length of PR (diagonal) -/
  pr : ℝ
  /-- The length of QS (diagonal) -/
  qs : ℝ
  /-- PB is positive -/
  pb_pos : pb > 0
  /-- BQ is positive -/
  bq_pos : bq > 0
  /-- PR is positive -/
  pr_pos : pr > 0
  /-- QS is positive -/
  qs_pos : qs > 0
  /-- PR ≠ QS (to ensure the rhombus is not a square) -/
  diag_neq : pr ≠ qs

/-- The perimeter of the rectangle containing the inscribed rhombus -/
def rectanglePerimeter (r : InscribedRhombus) : ℝ := sorry

/-- Theorem stating the perimeter of the rectangle for the given measurements -/
theorem inscribed_rhombus_rectangle_perimeter :
  let r : InscribedRhombus := {
    pb := 15,
    bq := 20,
    pr := 30,
    qs := 40,
    pb_pos := by norm_num,
    bq_pos := by norm_num,
    pr_pos := by norm_num,
    qs_pos := by norm_num,
    diag_neq := by norm_num
  }
  rectanglePerimeter r = 672 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_rectangle_perimeter_l3728_372830


namespace NUMINAMATH_CALUDE_article_cost_price_l3728_372824

theorem article_cost_price 
  (C M : ℝ) 
  (h1 : 0.95 * M = 1.4 * C) 
  (h2 : 0.95 * M = 70) : 
  C = 50 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3728_372824


namespace NUMINAMATH_CALUDE_helicopter_performance_l3728_372873

/-- Heights of helicopter A's performances in km -/
def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]

/-- Heights of helicopter B's performances in km -/
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

/-- The highest altitude reached by helicopter A -/
def highest_altitude_A : ℝ := 3.6

/-- The final altitude of helicopter A after 5 performances -/
def final_altitude_A : ℝ := 3.4

/-- The required height change for helicopter B's 5th performance -/
def height_change_B : ℝ := -0.2

theorem helicopter_performance :
  (heights_A.maximum? = some highest_altitude_A) ∧
  (heights_A.sum = final_altitude_A) ∧
  (heights_B.sum + height_change_B = final_altitude_A) := by
  sorry

end NUMINAMATH_CALUDE_helicopter_performance_l3728_372873


namespace NUMINAMATH_CALUDE_playlist_song_length_l3728_372840

theorem playlist_song_length 
  (total_songs : Nat) 
  (song1_length : Nat) 
  (song2_length : Nat) 
  (total_playtime : Nat) 
  (playlist_repeats : Nat) 
  (h1 : total_songs = 3) 
  (h2 : song1_length = 3) 
  (h3 : song2_length = 3) 
  (h4 : total_playtime = 40) 
  (h5 : playlist_repeats = 5) :
  ∃ (song3_length : Nat), 
    song1_length + song2_length + song3_length = total_playtime / playlist_repeats ∧ 
    song3_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_playlist_song_length_l3728_372840


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l3728_372897

theorem shortest_side_right_triangle (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_a : a = 7) (h_b : b = 24) : 
  min a b = 7 := by sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l3728_372897


namespace NUMINAMATH_CALUDE_min_sum_given_product_min_sum_equality_case_l3728_372800

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 8*b = a*b → a + b ≥ 18 := by
  sorry

-- The equality case
theorem min_sum_equality_case (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 8*b = a*b → (a + b = 18 ↔ a = 12 ∧ b = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_min_sum_equality_case_l3728_372800


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_proposition_l3728_372844

theorem negation_of_universal_quantifier (P : ℝ → Prop) :
  (¬ ∀ x ∈ Set.Ici 1, P x) ↔ (∃ x ∈ Set.Ici 1, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x ∈ Set.Ici 1, x^2 - 2*x + 1 ≥ 0) ↔ (∃ x ∈ Set.Ici 1, x^2 - 2*x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_proposition_l3728_372844


namespace NUMINAMATH_CALUDE_prob_no_adjacent_birch_value_l3728_372876

/-- The number of pine trees -/
def num_pine : ℕ := 6

/-- The number of cedar trees -/
def num_cedar : ℕ := 5

/-- The number of birch trees -/
def num_birch : ℕ := 7

/-- The total number of trees -/
def total_trees : ℕ := num_pine + num_cedar + num_birch

/-- The number of slots for birch trees -/
def num_slots : ℕ := num_pine + num_cedar + 1

/-- The probability of no two birch trees being adjacent when arranged randomly -/
def prob_no_adjacent_birch : ℚ := (num_slots.choose num_birch : ℚ) / (total_trees.choose num_birch)

theorem prob_no_adjacent_birch_value : prob_no_adjacent_birch = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_birch_value_l3728_372876


namespace NUMINAMATH_CALUDE_regular_iff_all_face_angles_equal_exists_non_regular_with_five_equal_angles_l3728_372871

-- Define a tetrahedron
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a function to calculate the angle between two faces of a tetrahedron
def angleBetweenFaces (t : Tetrahedron) (face1 : Fin 4) (face2 : Fin 4) : ℝ := sorry

-- Define what it means for a tetrahedron to be regular
def isRegular (t : Tetrahedron) : Prop := sorry

-- Define what it means for all face angles to be equal
def allFaceAnglesEqual (t : Tetrahedron) : Prop :=
  ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l → angleBetweenFaces t i j = angleBetweenFaces t k l

-- Define what it means for five out of six face angles to be equal
def fiveFaceAnglesEqual (t : Tetrahedron) : Prop :=
  ∃ (i j k l m n : Fin 4), i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    angleBetweenFaces t i j = angleBetweenFaces t k l ∧
    angleBetweenFaces t i j = angleBetweenFaces t m n ∧
    (∀ (a b : Fin 4), a ≠ b → 
      angleBetweenFaces t a b = angleBetweenFaces t i j ∨
      angleBetweenFaces t a b = angleBetweenFaces t k l ∨
      angleBetweenFaces t a b = angleBetweenFaces t m n)

-- Theorem 1: A tetrahedron is regular if and only if all face angles are equal
theorem regular_iff_all_face_angles_equal (t : Tetrahedron) :
  isRegular t ↔ allFaceAnglesEqual t := by sorry

-- Theorem 2: There exists a non-regular tetrahedron with five equal face angles
theorem exists_non_regular_with_five_equal_angles :
  ∃ (t : Tetrahedron), fiveFaceAnglesEqual t ∧ ¬isRegular t := by sorry

end NUMINAMATH_CALUDE_regular_iff_all_face_angles_equal_exists_non_regular_with_five_equal_angles_l3728_372871


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l3728_372860

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) :
  (x + y) / 2 = 20 →
  Real.sqrt (x * y) = Real.sqrt 110 →
  x^2 + y^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l3728_372860


namespace NUMINAMATH_CALUDE_valid_paths_count_l3728_372879

/-- Represents the number of paths on a complete 9x3 grid -/
def total_paths : ℕ := 220

/-- Represents the number of paths through each forbidden segment -/
def forbidden_segment_paths : ℕ := 70

/-- Represents the number of forbidden segments -/
def num_forbidden_segments : ℕ := 2

/-- Theorem stating the number of valid paths on the grid with forbidden segments -/
theorem valid_paths_count : 
  total_paths - (forbidden_segment_paths * num_forbidden_segments) = 80 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l3728_372879
