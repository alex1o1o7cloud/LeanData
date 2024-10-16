import Mathlib

namespace NUMINAMATH_CALUDE_mrs_wong_valentines_l1055_105502

def valentines_problem (initial : ℕ) (given_away : ℕ) : Prop :=
  initial - given_away = 22

theorem mrs_wong_valentines : valentines_problem 30 8 := by
  sorry

end NUMINAMATH_CALUDE_mrs_wong_valentines_l1055_105502


namespace NUMINAMATH_CALUDE_solution_count_l1055_105588

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = x + y + c

/-- The theorem stating the number of solutions based on the value of c -/
theorem solution_count (c : ℝ) :
  (c = 0 ∧ ∃! f : ℝ → ℝ, SatisfiesEquation f c ∧ f = id) ∨
  (c ≠ 0 ∧ ¬∃ f : ℝ → ℝ, SatisfiesEquation f c) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l1055_105588


namespace NUMINAMATH_CALUDE_residue_negative_1234_mod_32_l1055_105579

theorem residue_negative_1234_mod_32 : Int.mod (-1234) 32 = 14 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_1234_mod_32_l1055_105579


namespace NUMINAMATH_CALUDE_speed_increase_percentage_l1055_105521

def initial_speed : ℝ := 80
def training_weeks : ℕ := 16
def speed_gain_per_week : ℝ := 1

def final_speed : ℝ := initial_speed + (speed_gain_per_week * training_weeks)

theorem speed_increase_percentage :
  (final_speed - initial_speed) / initial_speed * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_percentage_l1055_105521


namespace NUMINAMATH_CALUDE_difference_of_squares_l1055_105556

theorem difference_of_squares : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1055_105556


namespace NUMINAMATH_CALUDE_total_buyers_is_140_l1055_105569

/-- The number of buyers who visited a store over three consecutive days -/
def total_buyers (day_before_yesterday yesterday today : ℕ) : ℕ :=
  day_before_yesterday + yesterday + today

/-- Theorem stating the total number of buyers over three days -/
theorem total_buyers_is_140 :
  ∃ (yesterday today : ℕ),
    yesterday = 50 / 2 ∧
    today = yesterday + 40 ∧
    total_buyers 50 yesterday today = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_buyers_is_140_l1055_105569


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1055_105537

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x = -3 ∧ 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 12) ∧
    (∀ y : ℝ, (3 * y / (y - 3) + (3 * y^2 - 27) / y = 12) → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1055_105537


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1055_105574

/-- An ellipse with center at the origin, one focus at (0,2), intersected by the line y = 3x + 7 
    such that the midpoint of the intersection chord has a y-coordinate of 1 -/
structure SpecialEllipse where
  /-- The equation of the ellipse in the form (x²/a²) + (y²/b²) = 1 -/
  equation : ℝ → ℝ → Prop
  /-- One focus of the ellipse is at (0,2) -/
  focus_at_0_2 : ∃ (x y : ℝ), equation x y ∧ x = 0 ∧ y = 2
  /-- The line y = 3x + 7 intersects the ellipse -/
  intersects_line : ∃ (x y : ℝ), equation x y ∧ y = 3*x + 7
  /-- The midpoint of the intersection chord has a y-coordinate of 1 -/
  midpoint_y_is_1 : 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      equation x₁ y₁ ∧ y₁ = 3*x₁ + 7 ∧
      equation x₂ y₂ ∧ y₂ = 3*x₂ + 7 ∧
      (y₁ + y₂) / 2 = 1

/-- The equation of the special ellipse is x²/8 + y²/12 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : 
  e.equation = fun x y => x^2/8 + y^2/12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1055_105574


namespace NUMINAMATH_CALUDE_intersection_trisection_l1055_105503

/-- A line y = mx + b intersecting a circle and a hyperbola -/
structure IntersectingLine where
  m : ℝ
  b : ℝ
  h_m : |m| < 1
  h_b : |b| < 1

/-- Points of intersection with the circle x^2 + y^2 = 1 -/
def circle_intersection (l : IntersectingLine) : Set (ℝ × ℝ) :=
  {(x, y) | y = l.m * x + l.b ∧ x^2 + y^2 = 1}

/-- Points of intersection with the hyperbola x^2 - y^2 = 1 -/
def hyperbola_intersection (l : IntersectingLine) : Set (ℝ × ℝ) :=
  {(x, y) | y = l.m * x + l.b ∧ x^2 - y^2 = 1}

/-- Trisection property of the intersection points -/
def trisects (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ∈ Set.Icc (0 : ℝ) 1 ∧
    P = (1 - t) • R + t • S ∧
    Q = (1 - t) • S + t • R ∧
    t = 1/3 ∨ t = 2/3

/-- Main theorem: Intersection points trisect implies specific values for m and b -/
theorem intersection_trisection (l : IntersectingLine)
  (hP : P ∈ circle_intersection l) (hQ : Q ∈ circle_intersection l)
  (hR : R ∈ hyperbola_intersection l) (hS : S ∈ hyperbola_intersection l)
  (h_trisect : trisects P Q R S) :
  (l.m = 0 ∧ l.b = 2/5 * Real.sqrt 5) ∨
  (l.m = 0 ∧ l.b = -2/5 * Real.sqrt 5) ∨
  (l.m = 2/5 * Real.sqrt 5 ∧ l.b = 0) ∨
  (l.m = -2/5 * Real.sqrt 5 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_trisection_l1055_105503


namespace NUMINAMATH_CALUDE_noahs_closet_capacity_l1055_105552

theorem noahs_closet_capacity (ali_capacity : ℕ) (noah_total_capacity : ℕ) : 
  ali_capacity = 200 → noah_total_capacity = 100 → 
  (noah_total_capacity / 2 : ℚ) / ali_capacity = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_noahs_closet_capacity_l1055_105552


namespace NUMINAMATH_CALUDE_textbook_page_ratio_l1055_105577

/-- Proves the ratio of math textbook pages to the sum of history and geography textbook pages -/
theorem textbook_page_ratio : ∀ (history geography math science : ℕ) (total : ℕ),
  history = 160 →
  geography = history + 70 →
  science = 2 * history →
  total = history + geography + math + science →
  total = 905 →
  (math : ℚ) / (history + geography : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_textbook_page_ratio_l1055_105577


namespace NUMINAMATH_CALUDE_veg_eaters_count_l1055_105586

/-- Represents the number of people in different dietary categories in a family. -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ

/-- Calculates the total number of people who eat veg in the family. -/
def total_veg_eaters (diet : FamilyDiet) : ℕ :=
  diet.only_veg + diet.both_veg_and_non_veg

/-- Theorem stating that the total number of people who eat veg in the family is 19. -/
theorem veg_eaters_count (diet : FamilyDiet)
  (h1 : diet.only_veg = 13)
  (h2 : diet.only_non_veg = 8)
  (h3 : diet.both_veg_and_non_veg = 6) :
  total_veg_eaters diet = 19 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l1055_105586


namespace NUMINAMATH_CALUDE_find_special_number_l1055_105544

theorem find_special_number : 
  ∃ n : ℕ, 
    (∃ k : ℕ, 3 * n = 2 * k + 1) ∧ 
    (∃ m : ℕ, 3 * n = 9 * m) ∧ 
    (∀ x : ℕ, x < n → ¬((∃ k : ℕ, 3 * x = 2 * k + 1) ∧ (∃ m : ℕ, 3 * x = 9 * m))) :=
by sorry

end NUMINAMATH_CALUDE_find_special_number_l1055_105544


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1055_105595

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℝ),
    is_arithmetic_sequence (-1) a₁ a₂ 8 →
    is_geometric_sequence (-1) b₁ b₂ b₃ (-4) →
    (a₁ * a₂) / b₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1055_105595


namespace NUMINAMATH_CALUDE_number_difference_l1055_105593

theorem number_difference (n : ℕ) (h : n = 15) : n * 13 - n = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1055_105593


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_ratio_l1055_105576

-- Define a right-angled triangle with sides forming an arithmetic sequence
structure RightTriangleArithmeticSides where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  arithmetic_sequence : ∃ d : ℝ, b = a + d ∧ c = b + d

-- Theorem statement
theorem right_triangle_arithmetic_sides_ratio 
  (t : RightTriangleArithmeticSides) : 
  ∃ k : ℝ, t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sides_ratio_l1055_105576


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1055_105547

theorem trigonometric_equality : 2 * Real.tan (π / 3) + Real.tan (π / 4) - 4 * Real.cos (π / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1055_105547


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1055_105509

theorem absolute_value_equation_solution (x : ℝ) : 
  |2*x - 1| + |x - 2| = |x + 1| ↔ 1/2 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1055_105509


namespace NUMINAMATH_CALUDE_expression_evaluation_l1055_105590

theorem expression_evaluation : 2 - 3 * (-4) + 5 - (-6) * 7 = 61 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1055_105590


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1055_105543

theorem sufficient_not_necessary (a b : ℝ) :
  (b > a ∧ a > 0 → (a + 2) / (b + 2) > a / b) ∧
  ∃ a b : ℝ, (a + 2) / (b + 2) > a / b ∧ ¬(b > a ∧ a > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1055_105543


namespace NUMINAMATH_CALUDE_smallest_number_with_condition_condition_satisfied_by_725_l1055_105517

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n - 1)).filter (fun d => d ≠ 1 ∧ n % d = 0)

def divisors_condition (n : ℕ) : Prop :=
  let divs := proper_divisors n
  let largest_sum := (Finset.max' divs (by sorry) + Finset.max' (divs.erase (Finset.max' divs (by sorry))) (by sorry))
  let smallest_sum := (Finset.min' divs (by sorry) + Finset.min' (divs.erase (Finset.min' divs (by sorry))) (by sorry))
  ¬(largest_sum % smallest_sum = 0)

theorem smallest_number_with_condition :
  ∀ n : ℕ, n < 725 → ¬(ends_with_five n ∧ divisors_condition n) :=
by sorry

theorem condition_satisfied_by_725 :
  ends_with_five 725 ∧ divisors_condition 725 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_condition_condition_satisfied_by_725_l1055_105517


namespace NUMINAMATH_CALUDE_z_value_l1055_105583

theorem z_value (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x*y - 9) : z = 0 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l1055_105583


namespace NUMINAMATH_CALUDE_vector_subtraction_l1055_105538

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1055_105538


namespace NUMINAMATH_CALUDE_inequality_proof_l1055_105506

theorem inequality_proof (a : ℝ) : 3 * (1 + a^2 + a^4) - (1 + a + a^2)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1055_105506


namespace NUMINAMATH_CALUDE_jenna_work_hours_l1055_105592

def concert_ticket_cost : ℝ := 181
def drink_ticket_cost : ℝ := 7
def num_drink_tickets : ℕ := 5
def hourly_wage : ℝ := 18
def salary_percentage : ℝ := 0.1
def weeks_per_month : ℕ := 4

theorem jenna_work_hours :
  ∀ (weekly_hours : ℝ),
  (concert_ticket_cost + num_drink_tickets * drink_ticket_cost = 
   salary_percentage * (weekly_hours * hourly_wage * weeks_per_month)) →
  weekly_hours = 30 := by
  sorry

end NUMINAMATH_CALUDE_jenna_work_hours_l1055_105592


namespace NUMINAMATH_CALUDE_faster_pipe_rate_l1055_105539

/-- Given two pipes with different filling rates, prove that the faster pipe is 4 times faster than the slower pipe. -/
theorem faster_pipe_rate (slow_rate fast_rate : ℝ) : 
  slow_rate > 0 →
  fast_rate > slow_rate →
  (1 : ℝ) / slow_rate = 180 →
  1 / (slow_rate + fast_rate) = 36 →
  fast_rate = 4 * slow_rate :=
by sorry

end NUMINAMATH_CALUDE_faster_pipe_rate_l1055_105539


namespace NUMINAMATH_CALUDE_log_inequality_l1055_105564

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + Real.sqrt x) < Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1055_105564


namespace NUMINAMATH_CALUDE_mary_crayons_left_l1055_105514

/-- Represents the number of crayons Mary has left after giving some away -/
def crayons_left (initial_green initial_blue initial_yellow given_green given_blue given_yellow : ℕ) : ℕ :=
  (initial_green - given_green) + (initial_blue - given_blue) + (initial_yellow - given_yellow)

/-- Theorem stating that Mary has 14 crayons left after giving some away -/
theorem mary_crayons_left : 
  crayons_left 5 8 7 3 1 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_crayons_left_l1055_105514


namespace NUMINAMATH_CALUDE_parallel_plane_through_point_l1055_105507

def plane_equation (x y z : ℝ) := 3*x - 2*y + 4*z - 32

theorem parallel_plane_through_point :
  let given_plane (x y z : ℝ) := 3*x - 2*y + 4*z - 6
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ ∃ (t : ℝ), given_plane x y z = t) ∧ 
  plane_equation 2 (-3) 5 = 0 ∧
  (∃ (A B C D : ℤ), ∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
  (∃ (A : ℤ), A > 0 ∧ ∀ (x y z : ℝ), plane_equation x y z = A*x + plane_equation 0 1 0*y + plane_equation 0 0 1*z + plane_equation 0 0 0) ∧
  (Nat.gcd (Int.natAbs 3) (Int.natAbs (-2)) = 1 ∧ 
   Nat.gcd (Int.natAbs 3) (Int.natAbs 4) = 1 ∧ 
   Nat.gcd (Int.natAbs 3) (Int.natAbs (-32)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_plane_through_point_l1055_105507


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l1055_105567

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l1055_105567


namespace NUMINAMATH_CALUDE_license_plate_increase_l1055_105518

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible license plates in the old scheme -/
def old_scheme_count : ℕ := num_letters * (num_digits ^ 5)

/-- The number of possible license plates in the new scheme -/
def new_scheme_count : ℕ := (num_letters ^ 2) * (num_digits ^ 4)

/-- The ratio of new scheme count to old scheme count -/
def license_plate_ratio : ℚ := new_scheme_count / old_scheme_count

theorem license_plate_increase : license_plate_ratio = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l1055_105518


namespace NUMINAMATH_CALUDE_cuboidal_box_volume_l1055_105589

/-- Represents a cuboidal box with given face areas -/
structure CuboidalBox where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- Calculates the volume of a cuboidal box given its face areas -/
def volume (box : CuboidalBox) : ℝ :=
  sorry

/-- Theorem stating that a cuboidal box with face areas 120, 72, and 60 has volume 720 -/
theorem cuboidal_box_volume :
  ∀ (box : CuboidalBox),
    box.area1 = 120 ∧ box.area2 = 72 ∧ box.area3 = 60 →
    volume box = 720 :=
by sorry

end NUMINAMATH_CALUDE_cuboidal_box_volume_l1055_105589


namespace NUMINAMATH_CALUDE_current_speed_l1055_105565

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 20)
  (h2 : speed_against_current = 14) :
  ∃ (current_speed : ℝ), current_speed = 3 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l1055_105565


namespace NUMINAMATH_CALUDE_octagon_trapezoid_area_l1055_105555

/-- The area of a trapezoid formed by four consecutive vertices of a regular octagon --/
theorem octagon_trapezoid_area (side_length : ℝ) (h : side_length = 6) :
  let diagonal_ratio : ℝ := Real.sqrt (4 + 2 * Real.sqrt 2)
  let height : ℝ := side_length * diagonal_ratio * (Real.sqrt (2 - Real.sqrt 2) / 2)
  let area : ℝ := side_length * height
  area = 18 * Real.sqrt (16 - 4 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_octagon_trapezoid_area_l1055_105555


namespace NUMINAMATH_CALUDE_minimum_value_and_range_of_a_l1055_105561

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1/2 * x^2 - x

theorem minimum_value_and_range_of_a :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = 2) →
  (∃ x_min : ℝ, x_min > 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ f a x_min) →
  (f a 2 = -2 * Real.log 2) ∧
  (∀ x : ℝ, x > Real.exp 1 → f a x - a * x > 0) →
  a ≤ (Real.exp 2 - 2 * Real.exp 1) / (2 * (Real.exp 1 - 1)) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_of_a_l1055_105561


namespace NUMINAMATH_CALUDE_value_of_x_l1055_105551

theorem value_of_x : (2015^2 - 2015) / 2015 = 2014 := by sorry

end NUMINAMATH_CALUDE_value_of_x_l1055_105551


namespace NUMINAMATH_CALUDE_P_homogeneous_P_sum_condition_P_initial_condition_P_unique_l1055_105519

/-- A homogeneous polynomial of degree n in x and y satisfying specific conditions -/
def P (n : ℕ+) : ℝ → ℝ → ℝ := fun x y ↦ (x + y) ^ (n.val - 1) * (x - 2*y)

/-- P is a homogeneous polynomial of degree n -/
theorem P_homogeneous (n : ℕ+) (t x y : ℝ) : 
  P n (t * x) (t * y) = t ^ n.val * P n x y := by sorry

/-- P satisfies the given sum condition -/
theorem P_sum_condition (n : ℕ+) (a b c : ℝ) :
  P n (a + b) c + P n (b + c) a + P n (c + a) b = 0 := by sorry

/-- P satisfies the given initial condition -/
theorem P_initial_condition (n : ℕ+) : P n 1 0 = 1 := by sorry

/-- P is the unique polynomial satisfying all conditions -/
theorem P_unique (n : ℕ+) (Q : ℝ → ℝ → ℝ) 
  (h_homogeneous : ∀ t x y, Q (t * x) (t * y) = t ^ n.val * Q x y)
  (h_sum : ∀ a b c, Q (a + b) c + Q (b + c) a + Q (c + a) b = 0)
  (h_initial : Q 1 0 = 1) :
  Q = P n := by sorry

end NUMINAMATH_CALUDE_P_homogeneous_P_sum_condition_P_initial_condition_P_unique_l1055_105519


namespace NUMINAMATH_CALUDE_six_students_adjacent_permutations_l1055_105526

/-- The number of permutations of n elements where two specific elements must be adjacent -/
def adjacent_permutations (n : ℕ) : ℕ :=
  2 * Nat.factorial (n - 1)

/-- Theorem: The number of permutations of 6 students where 2 specific students
    must stand next to each other is 240 -/
theorem six_students_adjacent_permutations :
  adjacent_permutations 6 = 240 := by
  sorry

#eval adjacent_permutations 6

end NUMINAMATH_CALUDE_six_students_adjacent_permutations_l1055_105526


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l1055_105532

/-- The increase in radius and height of a cylinder that results in quadrupling its volume -/
theorem cylinder_volume_increase (x : ℝ) : x > 0 →
  π * (10 + x)^2 * (5 + x) = 4 * (π * 10^2 * 5) →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l1055_105532


namespace NUMINAMATH_CALUDE_train_speed_l1055_105524

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 240) (h2 : time = 16) :
  length / time = 15 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1055_105524


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1055_105549

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 2 →                        -- given a₂ = 2
  a 5 = 1/4 →                      -- given a₅ = 1/4
  q = 1/2 :=                       -- prove q = 1/2
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1055_105549


namespace NUMINAMATH_CALUDE_absent_student_percentage_l1055_105596

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 180)
  (h2 : boys = 100)
  (h3 : girls = 80)
  (h4 : total_students = boys + girls)
  (absent_boys_ratio : ℚ)
  (absent_girls_ratio : ℚ)
  (h5 : absent_boys_ratio = 1 / 5)
  (h6 : absent_girls_ratio = 1 / 4) :
  (((boys * absent_boys_ratio + girls * absent_girls_ratio) / total_students) : ℚ) = 2222 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_absent_student_percentage_l1055_105596


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1055_105557

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1055_105557


namespace NUMINAMATH_CALUDE_combined_exterior_angles_pentagon_hexagon_l1055_105545

-- Define the sum of exterior angles for any convex polygon
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Define a pentagon
def pentagon : ℕ := 5

-- Define a hexagon
def hexagon : ℕ := 6

-- Theorem statement
theorem combined_exterior_angles_pentagon_hexagon :
  sum_exterior_angles pentagon + sum_exterior_angles hexagon = 720 := by
  sorry

end NUMINAMATH_CALUDE_combined_exterior_angles_pentagon_hexagon_l1055_105545


namespace NUMINAMATH_CALUDE_only_one_statement_correct_l1055_105504

theorem only_one_statement_correct : 
  ¬(∀ (a b : ℤ), a < b → a^2 < b^2) ∧ 
  ¬(∀ (a : ℤ), a^2 > 0) ∧ 
  ¬(∀ (a : ℤ), -a < 0) ∧ 
  (∀ (a b c : ℤ), a * c^2 < b * c^2 → a < b) :=
by sorry

end NUMINAMATH_CALUDE_only_one_statement_correct_l1055_105504


namespace NUMINAMATH_CALUDE_train_speed_l1055_105582

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 3000) (h2 : time = 120) :
  length / time * (3600 / 1000) = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1055_105582


namespace NUMINAMATH_CALUDE_speed_of_train_b_l1055_105558

/-- Theorem: Speed of Train B
Given two trains A and B traveling in opposite directions, meeting at some point,
with train A reaching its destination 9 hours after meeting and traveling at 70 km/h,
and train B reaching its destination 4 hours after meeting,
prove that the speed of train B is 157.5 km/h. -/
theorem speed_of_train_b (speed_a : ℝ) (time_a time_b : ℝ) (speed_b : ℝ) :
  speed_a = 70 →
  time_a = 9 →
  time_b = 4 →
  speed_a * time_a = speed_b * time_b →
  speed_b = 157.5 := by
sorry

end NUMINAMATH_CALUDE_speed_of_train_b_l1055_105558


namespace NUMINAMATH_CALUDE_solution_is_3_minus_i_l1055_105501

/-- Definition of the determinant operation -/
def det (a b c d : ℂ) : ℂ := a * d - b * c

/-- Theorem stating that the complex number z satisfying the given equation is 3 - i -/
theorem solution_is_3_minus_i :
  ∃ z : ℂ, det 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I :=
sorry

end NUMINAMATH_CALUDE_solution_is_3_minus_i_l1055_105501


namespace NUMINAMATH_CALUDE_ribbon_length_l1055_105523

theorem ribbon_length (A : ℝ) (π_estimate : ℝ) (extra : ℝ) : 
  A = 154 → π_estimate = 22 / 7 → extra = 2 →
  ∃ (r : ℝ), 
    A = π_estimate * r^2 ∧ 
    2 * π_estimate * r + extra = 46 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_l1055_105523


namespace NUMINAMATH_CALUDE_paris_study_time_l1055_105500

/-- Calculates the total study time for a student during a semester. -/
def totalStudyTime (semesterWeeks : ℕ) (weekdayStudyHours : ℕ) (saturdayStudyHours : ℕ) (sundayStudyHours : ℕ) : ℕ :=
  let weekdaysTotalHours := 5 * weekdayStudyHours
  let weekendTotalHours := saturdayStudyHours + sundayStudyHours
  let weeklyTotalHours := weekdaysTotalHours + weekendTotalHours
  semesterWeeks * weeklyTotalHours

theorem paris_study_time :
  totalStudyTime 15 3 4 5 = 360 := by
  sorry

end NUMINAMATH_CALUDE_paris_study_time_l1055_105500


namespace NUMINAMATH_CALUDE_total_fish_caught_l1055_105529

def fishing_problem (leo_fish agrey_fish total_fish : ℕ) : Prop :=
  (leo_fish = 40) ∧
  (agrey_fish = leo_fish + 20) ∧
  (total_fish = leo_fish + agrey_fish)

theorem total_fish_caught : ∃ (leo_fish agrey_fish total_fish : ℕ),
  fishing_problem leo_fish agrey_fish total_fish ∧ total_fish = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l1055_105529


namespace NUMINAMATH_CALUDE_article_price_proof_l1055_105508

-- Define the original price
def original_price : ℝ := 2500

-- Define the profit percentage
def profit_percentage : ℝ := 0.25

-- Define the profit amount
def profit_amount : ℝ := 625

-- Theorem statement
theorem article_price_proof :
  profit_amount = original_price * profit_percentage :=
by
  sorry

#check article_price_proof

end NUMINAMATH_CALUDE_article_price_proof_l1055_105508


namespace NUMINAMATH_CALUDE_number_in_scientific_notation_l1055_105575

/-- Scientific notation representation of a positive real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem number_in_scientific_notation :
  toScientificNotation 36600 = ScientificNotation.mk 3.66 4 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_number_in_scientific_notation_l1055_105575


namespace NUMINAMATH_CALUDE_min_value_theorem_l1055_105554

theorem min_value_theorem (a : ℝ) (h : a > 3) :
  a + 1 / (a - 3) ≥ 5 ∧ (a + 1 / (a - 3) = 5 ↔ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1055_105554


namespace NUMINAMATH_CALUDE_star_18_6_l1055_105546

/-- The star operation defined for integers -/
def star (a b : ℤ) : ℚ := a - a / b

/-- Theorem stating that 18 ★ 6 = 15 -/
theorem star_18_6 : star 18 6 = 15 := by sorry

end NUMINAMATH_CALUDE_star_18_6_l1055_105546


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1055_105513

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1055_105513


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1055_105535

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  (α = β ∨ α = γ ∨ β = γ) →  -- The triangle is isosceles
  (α = 110 ∨ β = 110 ∨ γ = 110) →  -- One of the angles is 110°
  (α = 35 ∨ β = 35 ∨ γ = 35) :=  -- One of the base angles is 35°
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1055_105535


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1055_105533

/-- Given three rectangles with equal areas and integer sides, where one side is 37, prove that a specific side length is 1406. -/
theorem rectangle_side_length (a b : ℕ) : 
  let S := 37 * (a + b)  -- Common area of the rectangles
  -- ABCD area
  S = 37 * (a + b) →
  -- DEFG area
  S = a * 1406 →
  -- CEIH area
  S = b * 38 →
  -- All sides are integers
  a > 0 → b > 0 →
  -- DG length
  1406 = 1406 := by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1055_105533


namespace NUMINAMATH_CALUDE_crimson_valley_skirts_l1055_105578

theorem crimson_valley_skirts 
  (azure_skirts : ℕ) 
  (seafoam_skirts : ℕ) 
  (purple_skirts : ℕ) 
  (crimson_skirts : ℕ) 
  (h1 : azure_skirts = 90)
  (h2 : seafoam_skirts = 2 * azure_skirts / 3)
  (h3 : purple_skirts = seafoam_skirts / 4)
  (h4 : crimson_skirts = purple_skirts / 3) :
  crimson_skirts = 5 := by
  sorry

end NUMINAMATH_CALUDE_crimson_valley_skirts_l1055_105578


namespace NUMINAMATH_CALUDE_publishing_break_even_point_l1055_105510

/-- Represents the break-even point calculation for a publishing company --/
theorem publishing_break_even_point 
  (fixed_cost : ℝ) 
  (variable_cost_per_book : ℝ) 
  (selling_price_per_book : ℝ) 
  (h1 : fixed_cost = 56430)
  (h2 : variable_cost_per_book = 8.25)
  (h3 : selling_price_per_book = 21.75) :
  ∃ (x : ℝ), 
    x = 4180 ∧ 
    fixed_cost + x * variable_cost_per_book = x * selling_price_per_book :=
by sorry

end NUMINAMATH_CALUDE_publishing_break_even_point_l1055_105510


namespace NUMINAMATH_CALUDE_table_tennis_sequences_l1055_105585

/-- Represents a sequence of matches in the table tennis competition -/
def MatchSequence := List ℕ

/-- The number of players in each team -/
def teamSize : ℕ := 5

/-- Calculates the number of possible sequences for a given player finishing the competition -/
def sequencesForPlayer (player : ℕ) : ℕ := sorry

/-- Calculates the total number of possible sequences for one team winning -/
def totalSequencesOneTeam : ℕ :=
  (List.range teamSize).map sequencesForPlayer |>.sum

/-- The total number of possible sequences in the competition -/
def totalSequences : ℕ := 2 * totalSequencesOneTeam

theorem table_tennis_sequences :
  totalSequences = 252 := by sorry

end NUMINAMATH_CALUDE_table_tennis_sequences_l1055_105585


namespace NUMINAMATH_CALUDE_smallest_stairs_count_l1055_105580

theorem smallest_stairs_count : ∃ (n : ℕ), n > 15 ∧ n % 6 = 4 ∧ n % 7 = 3 ∧ ∀ (m : ℕ), m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_stairs_count_l1055_105580


namespace NUMINAMATH_CALUDE_potatoes_remaining_l1055_105531

/-- Calculates the number of potatoes left after distribution -/
def potatoes_left (total : ℕ) (to_gina : ℕ) : ℕ :=
  let to_tom := 2 * to_gina
  let to_anne := to_tom / 3
  total - (to_gina + to_tom + to_anne)

/-- Theorem stating that 47 potatoes are left after distribution -/
theorem potatoes_remaining : potatoes_left 300 69 = 47 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_remaining_l1055_105531


namespace NUMINAMATH_CALUDE_cube_with_specific_digits_l1055_105584

theorem cube_with_specific_digits : ∃! n : ℕ, 
  (n^3 ≥ 30000 ∧ n^3 < 40000) ∧ 
  (n^3 % 10 = 4) ∧
  (n = 34) := by
  sorry

end NUMINAMATH_CALUDE_cube_with_specific_digits_l1055_105584


namespace NUMINAMATH_CALUDE_solve_for_k_l1055_105515

theorem solve_for_k : ∃ k : ℝ, ((-1) - k * 2 = 7) ∧ k = -4 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l1055_105515


namespace NUMINAMATH_CALUDE_inequality_solution_l1055_105594

theorem inequality_solution (a x : ℝ) : 
  (x - a) * (x - a^2) < 0 ↔ 
  ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) ∨ 
  (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a) ∨ 
  (a = 0 ∨ a = 1 ∧ False) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1055_105594


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1055_105548

/-- Represents a trapezoid ABCD with side lengths AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem stating that if the area ratio of triangles ABC to ADC is 8:2
    and AB + CD = 250, then AB = 200 -/
theorem trapezoid_segment_length (t : Trapezoid) 
    (h1 : (t.AB / t.CD) = 4)  -- Ratio of areas is equivalent to ratio of bases
    (h2 : t.AB + t.CD = 250) : 
  t.AB = 200 := by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1055_105548


namespace NUMINAMATH_CALUDE_part_one_part_two_l1055_105562

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 5|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x ≥ 2 * |x + 5|} = {x : ℝ | x ≤ -2} := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 8) → (a ≥ 3 ∨ a ≤ -13) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1055_105562


namespace NUMINAMATH_CALUDE_equal_roots_count_l1055_105520

/-- The number of real values of p for which the roots of x^2 - px + p^2 = 0 are equal is 1 -/
theorem equal_roots_count (p : ℝ) : ∃! p, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! x, x^2 - p*x + p^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_count_l1055_105520


namespace NUMINAMATH_CALUDE_tate_had_32_tickets_l1055_105525

/-- The number of tickets Tate and Peyton have together -/
def total_tickets : ℕ := 51

/-- The number of additional tickets Tate buys -/
def additional_tickets : ℕ := 2

/-- Tate's initial number of tickets -/
def tate_initial_tickets : ℕ → Prop := λ t => 
  ∃ (tate_total peyton_total : ℕ),
    tate_total = t + additional_tickets ∧
    peyton_total = tate_total / 2 ∧
    tate_total + peyton_total = total_tickets

theorem tate_had_32_tickets : tate_initial_tickets 32 := by
  sorry

end NUMINAMATH_CALUDE_tate_had_32_tickets_l1055_105525


namespace NUMINAMATH_CALUDE_gcd_of_g_103_104_l1055_105550

/-- The function g as defined in the problem -/
def g (x : ℤ) : ℤ := x^2 - x + 2025

/-- The theorem stating that the GCD of g(103) and g(104) is 2 -/
theorem gcd_of_g_103_104 : Int.gcd (g 103) (g 104) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_g_103_104_l1055_105550


namespace NUMINAMATH_CALUDE_negative_a_to_zero_power_l1055_105573

theorem negative_a_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_to_zero_power_l1055_105573


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l1055_105598

/-- A square inscribed in a circle, which is inscribed in a rectangle --/
structure SquareCircleRectangle where
  /-- Side length of the square --/
  s : ℝ
  /-- Radius of the circle --/
  r : ℝ
  /-- Width of the rectangle --/
  w : ℝ
  /-- Length of the rectangle --/
  l : ℝ
  /-- The square's diagonal is the circle's diameter --/
  h1 : s * Real.sqrt 2 = 2 * r
  /-- The circle's diameter is the rectangle's width --/
  h2 : 2 * r = w
  /-- The rectangle's length is twice its width --/
  h3 : l = 2 * w
  /-- The square's diagonal is 4 units --/
  h4 : s * Real.sqrt 2 = 4

/-- The area of the rectangle is 8 square units --/
theorem rectangle_area_is_eight (scr : SquareCircleRectangle) : scr.l * scr.w = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l1055_105598


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l1055_105560

/-- Proves that given the spending percentages and savings amount, the monthly income is 40000 --/
theorem monthly_income_calculation (income : ℝ) 
  (household_percent : income * (45 / 100) = income * 0.45)
  (clothes_percent : income * (25 / 100) = income * 0.25)
  (medicines_percent : income * (7.5 / 100) = income * 0.075)
  (savings : income * (1 - 0.45 - 0.25 - 0.075) = 9000) :
  income = 40000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l1055_105560


namespace NUMINAMATH_CALUDE_real_roots_condition_one_root_triple_other_l1055_105568

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x + y = a ∧ 1/x + 1/y = 1/b

-- Theorem for real roots condition
theorem real_roots_condition (a b : ℝ) :
  (∃ x y, system x y a b) ↔ (a > 0 ∧ b ≤ a/4) ∨ (a < 0 ∧ b ≥ a/4) :=
sorry

-- Theorem for one root being three times the other
theorem one_root_triple_other (a b : ℝ) :
  (∃ x y, system x y a b ∧ x = 3*y) ↔ b = 3*a/16 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_one_root_triple_other_l1055_105568


namespace NUMINAMATH_CALUDE_quadratic_properties_l1055_105536

/-- The quadratic function f(x) = ax² + 4x + 2 passing through (3, -4) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 2

/-- The value of a for which f(x) passes through (3, -4) -/
def a : ℝ := -2

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := 1

theorem quadratic_properties :
  (f a 3 = -4) ∧
  (∀ x : ℝ, f a x = f a (2 * axis_of_symmetry - x)) ∧
  (∀ x : ℝ, x ≥ axis_of_symmetry → ∀ y : ℝ, y > x → f a y < f a x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1055_105536


namespace NUMINAMATH_CALUDE_min_a_value_l1055_105530

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

noncomputable def g (a e : ℝ) (x : ℝ) : ℝ := -e * x^2 + a * x

theorem min_a_value (e : ℝ) (he : e = Real.exp 1) :
  (∀ x₁ : ℝ, ∃ x₂ ∈ Set.Icc (1/3 : ℝ) 2, f x₁ ≤ g 2 e x₂) ∧
  (∀ ε > 0, ∃ x₁ : ℝ, ∀ x₂ ∈ Set.Icc (1/3 : ℝ) 2, f x₁ > g (2 - ε) e x₂) :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l1055_105530


namespace NUMINAMATH_CALUDE_log_simplification_l1055_105581

theorem log_simplification (a b c d x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) : 
  Real.log (a^2 / b) + Real.log (b / c^2) + Real.log (c / d) - Real.log (a^2 * y / (d^3 * x)) = 
  Real.log (d^2 * x / y) := by
sorry

end NUMINAMATH_CALUDE_log_simplification_l1055_105581


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l1055_105566

theorem quadratic_rational_solutions (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l1055_105566


namespace NUMINAMATH_CALUDE_find_b_value_l1055_105541

theorem find_b_value (a b : ℝ) (h1 : 2 * a + 3 = 5) (h2 : b - a = 2) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1055_105541


namespace NUMINAMATH_CALUDE_methane_combustion_l1055_105542

/-- Represents the balanced chemical equation for methane combustion -/
structure MethaneReaction where
  ch4 : ℚ
  o2 : ℚ
  co2 : ℚ
  h2o : ℚ
  balanced : ch4 = 1 ∧ o2 = 2 ∧ co2 = 1 ∧ h2o = 2

/-- Theorem stating the number of moles of CH₄ required and CO₂ formed when 2 moles of O₂ react -/
theorem methane_combustion (reaction : MethaneReaction) (o2_moles : ℚ) 
  (h_o2 : o2_moles = 2) : 
  let ch4_required := o2_moles / reaction.o2 * reaction.ch4
  let co2_formed := ch4_required / reaction.ch4 * reaction.co2
  ch4_required = 1 ∧ co2_formed = 1 := by
  sorry


end NUMINAMATH_CALUDE_methane_combustion_l1055_105542


namespace NUMINAMATH_CALUDE_dot_product_NO_NM_l1055_105522

-- Define the function f(x) = x^2 + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the theorem
theorem dot_product_NO_NM :
  ∀ x : ℝ,
  0 < x → x < 2 →
  let M : ℝ × ℝ := (x, f x)
  let N : ℝ × ℝ := (0, 1)
  let O : ℝ × ℝ := (0, 0)
  (M.1 - O.1)^2 + (M.2 - O.2)^2 = 27 →
  let NO : ℝ × ℝ := (N.1 - O.1, N.2 - O.2)
  let NM : ℝ × ℝ := (M.1 - N.1, M.2 - N.2)
  NO.1 * NM.1 + NO.2 * NM.2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_NO_NM_l1055_105522


namespace NUMINAMATH_CALUDE_walk_group_legs_and_wheels_l1055_105540

/-- Calculates the total number of legs and wheels in a group of humans, dogs, and wheelchairs. -/
def total_legs_and_wheels (num_humans : ℕ) (num_dogs : ℕ) (num_wheelchairs : ℕ) : ℕ :=
  num_humans * 2 + num_dogs * 4 + num_wheelchairs * 4

/-- Proves that the total number of legs and wheels in the given group is 22. -/
theorem walk_group_legs_and_wheels :
  total_legs_and_wheels 3 3 1 = 22 := by
  sorry

end NUMINAMATH_CALUDE_walk_group_legs_and_wheels_l1055_105540


namespace NUMINAMATH_CALUDE_max_discriminant_quadratic_l1055_105563

theorem max_discriminant_quadratic (a b c u v w : ℤ) :
  u ≠ v ∧ u ≠ w ∧ v ≠ w →
  a * u^2 + b * u + c = 0 →
  a * v^2 + b * v + c = 0 →
  a * w^2 + b * w + c = 2 →
  ∃ (max : ℤ), max = 16 ∧ b^2 - 4*a*c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_discriminant_quadratic_l1055_105563


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l1055_105528

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y -/
def combined_mixture (x y : SeedMixture) (x_proportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * x_proportion + y.ryegrass * (1 - x_proportion)
  , bluegrass := x.bluegrass * x_proportion + y.bluegrass * (1 - x_proportion)
  , fescue := x.fescue * x_proportion + y.fescue * (1 - x_proportion) }

theorem ryegrass_percentage_in_x 
  (x : SeedMixture) 
  (y : SeedMixture) 
  (h1 : x.bluegrass = 60)
  (h2 : x.ryegrass + x.bluegrass + x.fescue = 100)
  (h3 : y.ryegrass = 25)
  (h4 : y.fescue = 75)
  (h5 : y.ryegrass + y.bluegrass + y.fescue = 100)
  (h6 : (combined_mixture x y (2/3)).ryegrass = 35) :
  x.ryegrass = 40 := by
    sorry

#check ryegrass_percentage_in_x

end NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l1055_105528


namespace NUMINAMATH_CALUDE_randys_brother_biscuits_l1055_105572

/-- The number of biscuits Randy's brother ate -/
def biscuits_eaten (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (remaining : ℕ) : ℕ :=
  initial + from_father + from_mother - remaining

/-- Theorem stating the number of biscuits Randy's brother ate -/
theorem randys_brother_biscuits :
  biscuits_eaten 32 13 15 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_randys_brother_biscuits_l1055_105572


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1055_105534

theorem quadratic_roots_to_coefficients 
  (a b p q : ℝ) 
  (h1 : Complex.I ^ 2 = -1) 
  (h2 : (2 + a * Complex.I) ^ 2 + p * (2 + a * Complex.I) + q = 0) 
  (h3 : (b + Complex.I) ^ 2 + p * (b + Complex.I) + q = 0) : 
  p = -4 ∧ q = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1055_105534


namespace NUMINAMATH_CALUDE_cos_equality_theorem_l1055_105505

theorem cos_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (812 * π / 180)) ↔ (n = 92 ∨ n = 268) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_theorem_l1055_105505


namespace NUMINAMATH_CALUDE_multiplication_increase_l1055_105597

theorem multiplication_increase (x : ℝ) : x * 20 = 20 + 280 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_increase_l1055_105597


namespace NUMINAMATH_CALUDE_hundredth_number_is_hundred_l1055_105511

def counting_sequence (n : ℕ) : ℕ := n

theorem hundredth_number_is_hundred :
  counting_sequence 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_number_is_hundred_l1055_105511


namespace NUMINAMATH_CALUDE_ambiguous_triangle_case_l1055_105559

/-- Given two sides and an angle of a triangle, proves the existence of conditions
    for obtaining two different values for the third side. -/
theorem ambiguous_triangle_case (a b : ℝ) (α : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b)
  (h4 : 0 < α) (h5 : α < π) :
  ∃ c1 c2 : ℝ, c1 ≠ c2 ∧ 
  (∃ β γ : ℝ, 
    0 < β ∧ 0 < γ ∧ 
    α + β + γ = π ∧
    a / Real.sin α = b / Real.sin β ∧
    a / Real.sin α = c1 / Real.sin γ) ∧
  (∃ β' γ' : ℝ, 
    0 < β' ∧ 0 < γ' ∧ 
    α + β' + γ' = π ∧
    a / Real.sin α = b / Real.sin β' ∧
    a / Real.sin α = c2 / Real.sin γ') :=
sorry

end NUMINAMATH_CALUDE_ambiguous_triangle_case_l1055_105559


namespace NUMINAMATH_CALUDE_female_democrats_count_l1055_105570

-- Define the total number of participants
def total_participants : ℕ := 840

-- Define the ratio of female Democrats to total females
def female_democrat_ratio : ℚ := 1/2

-- Define the ratio of male Democrats to total males
def male_democrat_ratio : ℚ := 1/4

-- Define the ratio of all Democrats to total participants
def total_democrat_ratio : ℚ := 1/3

-- Theorem statement
theorem female_democrats_count :
  ∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    (female_democrat_ratio * female_participants + male_democrat_ratio * male_participants : ℚ) = 
      total_democrat_ratio * total_participants ∧
    female_democrat_ratio * female_participants = 140 :=
sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1055_105570


namespace NUMINAMATH_CALUDE_maria_roses_l1055_105553

/-- The number of roses Maria bought -/
def roses : ℕ := sorry

/-- The price of each flower -/
def flower_price : ℕ := 6

/-- The number of daisies Maria bought -/
def daisies : ℕ := 3

/-- The total amount Maria spent -/
def total_spent : ℕ := 60

theorem maria_roses :
  roses * flower_price + daisies * flower_price = total_spent →
  roses = 7 := by sorry

end NUMINAMATH_CALUDE_maria_roses_l1055_105553


namespace NUMINAMATH_CALUDE_die_roll_count_l1055_105587

theorem die_roll_count (total_sides : ℕ) (red_sides : ℕ) (prob : ℚ) : 
  total_sides = 10 →
  red_sides = 3 →
  prob = 147/1000 →
  (red_sides / total_sides : ℚ) * (1 - red_sides / total_sides : ℚ)^2 = prob →
  3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_count_l1055_105587


namespace NUMINAMATH_CALUDE_percentage_of_a_l1055_105512

-- Define the four numbers
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := a = 0.12 * b
def condition2 : Prop := b = 0.40 * c
def condition3 : Prop := c = 0.75 * d
def condition4 : Prop := d = 1.50 * (a + b)

-- Define the theorem
theorem percentage_of_a (h1 : condition1 a b) (h2 : condition2 b c) 
                        (h3 : condition3 c d) (h4 : condition4 a b d) :
  (a / (b + c + d)) * 100 = (1 / 43.166) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_a_l1055_105512


namespace NUMINAMATH_CALUDE_triangle_max_area_l1055_105599

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (2 * a + b) * Real.cos C + c * Real.cos B = 0 →
  c = 6 →
  ∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 ∧
    ∀ (S' : ℝ), S' = 1/2 * a * b * Real.sin C → S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1055_105599


namespace NUMINAMATH_CALUDE_max_regions_is_nine_l1055_105527

/-- Represents a square in a 2D plane -/
structure Square where
  -- We don't need to define the internals of the square for this problem

/-- The number of regions created by two intersecting squares -/
def num_regions (s1 s2 : Square) : ℕ := sorry

/-- The maximum number of regions that can be created by two intersecting squares -/
def max_regions : ℕ := sorry

/-- Theorem: The maximum number of regions created by two intersecting squares is 9 -/
theorem max_regions_is_nine : max_regions = 9 := by sorry

end NUMINAMATH_CALUDE_max_regions_is_nine_l1055_105527


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1055_105571

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- base radius
  let h : ℝ := 4  -- height
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  π * r * l = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1055_105571


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l1055_105591

theorem quadratic_function_minimum (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → a * (x - 1)^2 - a ≥ -4) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 4 ∧ a * (x - 1)^2 - a = -4) →
  a = 4 ∨ a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l1055_105591


namespace NUMINAMATH_CALUDE_sandwich_count_l1055_105516

theorem sandwich_count (billy_sandwiches : ℕ) (katelyn_extra : ℕ) :
  billy_sandwiches = 49 →
  katelyn_extra = 47 →
  (billy_sandwiches + katelyn_extra + billy_sandwiches + (billy_sandwiches + katelyn_extra) / 4 = 169) :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l1055_105516
