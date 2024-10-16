import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_f_l1097_109760

-- Define the function f(x)
def f (x : ℝ) := x * (1 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x, 0 < x → x < 1 → f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1097_109760


namespace NUMINAMATH_CALUDE_escalator_ride_time_l1097_109702

/-- Represents the time it takes Clea to descend the escalator under different conditions -/
structure EscalatorTime where
  stationary : ℝ  -- Time to walk down stationary escalator
  moving : ℝ      -- Time to walk down moving escalator
  riding : ℝ      -- Time to ride down without walking

/-- The theorem states that given the times for walking down stationary and moving escalators,
    the time to ride without walking can be determined -/
theorem escalator_ride_time (et : EscalatorTime) 
  (h1 : et.stationary = 75) 
  (h2 : et.moving = 30) : 
  et.riding = 50 := by
  sorry

end NUMINAMATH_CALUDE_escalator_ride_time_l1097_109702


namespace NUMINAMATH_CALUDE_transmitter_find_probability_l1097_109757

/-- Represents a license plate format for government vehicles in Kerrania -/
structure LicensePlate :=
  (first_two : Fin 100)
  (second : Fin 10)
  (last_two : Fin 100)
  (letters : Fin 3 × Fin 3)

/-- Conditions for a valid government license plate -/
def is_valid_plate (plate : LicensePlate) : Prop :=
  plate.first_two = 79 ∧
  (plate.second = 3 ∨ plate.second = 5) ∧
  (plate.last_two / 10 = plate.last_two % 10)

/-- Number of vehicles police can inspect in 3 hours -/
def inspected_vehicles : ℕ := 18

/-- Total number of possible valid license plates -/
def total_valid_plates : ℕ := 180

/-- Probability of finding the transmitter within 3 hours -/
def find_probability : ℚ := 1 / 10

theorem transmitter_find_probability :
  (inspected_vehicles : ℚ) / total_valid_plates = find_probability :=
sorry

end NUMINAMATH_CALUDE_transmitter_find_probability_l1097_109757


namespace NUMINAMATH_CALUDE_range_of_f_l1097_109750

def f (x : Int) : Int := x + 1

def domain : Set Int := {-1, 0, 1, 2}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {0, 1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1097_109750


namespace NUMINAMATH_CALUDE_deepak_age_l1097_109705

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, prove Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1097_109705


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1097_109752

/-- Given three points that represent three of the four endpoints of the axes of an ellipse -/
def point1 : ℝ × ℝ := (-2, 4)
def point2 : ℝ × ℝ := (3, -2)
def point3 : ℝ × ℝ := (8, 4)

/-- The theorem stating that the distance between the foci of the ellipse is 2√11 -/
theorem ellipse_foci_distance :
  ∃ (a b : ℝ) (center : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (center.1 - a = point1.1 ∨ center.1 - a = point2.1 ∨ center.1 - a = point3.1) ∧
    (center.1 + a = point1.1 ∨ center.1 + a = point2.1 ∨ center.1 + a = point3.1) ∧
    (center.2 - b = point1.2 ∨ center.2 - b = point2.2 ∨ center.2 - b = point3.2) ∧
    (center.2 + b = point1.2 ∨ center.2 + b = point2.2 ∨ center.2 + b = point3.2) ∧
    2 * Real.sqrt (max a b ^ 2 - min a b ^ 2) = 2 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1097_109752


namespace NUMINAMATH_CALUDE_business_ownership_l1097_109738

theorem business_ownership (total_value : ℝ) (sold_fraction : ℝ) (sold_value : ℝ) 
  (h1 : total_value = 75000)
  (h2 : sold_fraction = 3/5)
  (h3 : sold_value = 15000) :
  (sold_value / sold_fraction) / total_value = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_business_ownership_l1097_109738


namespace NUMINAMATH_CALUDE_lines_parallel_implies_a_eq_one_lines_perpendicular_implies_a_eq_zero_l1097_109783

/-- Two lines in the xy-plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define the first line l₁: x + ay - 2a - 2 = 0 -/
def l₁ (a : ℝ) : Line2D := ⟨1, a, -2*a - 2⟩

/-- Define the second line l₂: ax + y - 1 - a = 0 -/
def l₂ (a : ℝ) : Line2D := ⟨a, 1, -1 - a⟩

/-- Two lines are parallel if their slopes are equal -/
def parallel (l₁ l₂ : Line2D) : Prop := l₁.a * l₂.b = l₂.a * l₁.b

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l₁ l₂ : Line2D) : Prop := l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem lines_parallel_implies_a_eq_one :
  ∀ a : ℝ, parallel (l₁ a) (l₂ a) → a = 1 := by sorry

theorem lines_perpendicular_implies_a_eq_zero :
  ∀ a : ℝ, perpendicular (l₁ a) (l₂ a) → a = 0 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_implies_a_eq_one_lines_perpendicular_implies_a_eq_zero_l1097_109783


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1097_109712

theorem complex_equation_solution (a : ℝ) : (Complex.mk 2 a) * (Complex.mk a (-2)) = 8 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1097_109712


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l1097_109790

theorem sum_of_cyclic_equations (x y z : ℝ) 
  (eq1 : x + y = 1) 
  (eq2 : y + z = 1) 
  (eq3 : z + x = 1) : 
  x + y + z = 3/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l1097_109790


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1097_109706

theorem simplify_fraction_product : 8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1097_109706


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l1097_109730

theorem smallest_four_digit_mod_9 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 → 1007 ≤ n) ∧
  1000 ≤ 1007 ∧ 1007 < 10000 ∧ 1007 % 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l1097_109730


namespace NUMINAMATH_CALUDE_remainder_of_eighteen_divided_by_seven_l1097_109743

theorem remainder_of_eighteen_divided_by_seven : ∃ k : ℤ, 18 = 7 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_eighteen_divided_by_seven_l1097_109743


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1097_109723

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℤ) :
  total_questions = 120 →
  correct_answers = 75 →
  total_marks = 180 →
  (∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧
    score_per_correct = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1097_109723


namespace NUMINAMATH_CALUDE_fifteenth_triangular_number_is_120_and_even_l1097_109762

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 and it is even -/
theorem fifteenth_triangular_number_is_120_and_even :
  triangular_number 15 = 120 ∧ Even (triangular_number 15) := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_number_is_120_and_even_l1097_109762


namespace NUMINAMATH_CALUDE_intersection_complement_equals_one_l1097_109704

universe u

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem intersection_complement_equals_one : M ∩ (U \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_one_l1097_109704


namespace NUMINAMATH_CALUDE_line_segment_proportion_l1097_109789

-- Define the line segments as real numbers (representing their lengths in cm)
def a : ℝ := 1
def b : ℝ := 4
def c : ℝ := 2

-- Define the proportion relationship
def are_proportional (a b c d : ℝ) : Prop := a * d = b * c

-- State the theorem
theorem line_segment_proportion :
  ∀ d : ℝ, are_proportional a b c d → d = 8 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l1097_109789


namespace NUMINAMATH_CALUDE_grade_improvement_l1097_109782

/-- Represents the distribution of grades --/
structure GradeDistribution where
  a : ℕ  -- number of 1's
  b : ℕ  -- number of 2's
  c : ℕ  -- number of 3's
  d : ℕ  -- number of 4's
  e : ℕ  -- number of 5's

/-- Calculates the average grade --/
def averageGrade (g : GradeDistribution) : ℚ :=
  (g.a + 2 * g.b + 3 * g.c + 4 * g.d + 5 * g.e) / (g.a + g.b + g.c + g.d + g.e)

/-- Represents the change in grade distribution after changing 1's to 3's --/
def changeGrades (g : GradeDistribution) : GradeDistribution :=
  { a := 0, b := g.b, c := g.c + g.a, d := g.d, e := g.e }

theorem grade_improvement (g : GradeDistribution) :
  averageGrade g < 3 → averageGrade (changeGrades g) ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_grade_improvement_l1097_109782


namespace NUMINAMATH_CALUDE_i_to_2016_equals_1_l1097_109744

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2016_equals_1 : i ^ 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2016_equals_1_l1097_109744


namespace NUMINAMATH_CALUDE_samantha_birthday_next_monday_l1097_109795

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Calculates the day of the week for June 18 in a given year, 
    given the day of the week for June 18 in the previous year -/
def nextJune18 (prevDay : DayOfWeek) (year : Nat) : DayOfWeek :=
  sorry

/-- Finds the next year when June 18 falls on a Monday, given a starting year and day -/
def nextMondayJune18 (startYear : Nat) (startDay : DayOfWeek) : Nat :=
  sorry

theorem samantha_birthday_next_monday (startYear : Nat) (startDay : DayOfWeek) :
  startYear = 2009 →
  startDay = DayOfWeek.Friday →
  ¬isLeapYear startYear →
  nextMondayJune18 startYear startDay = 2017 :=
sorry

end NUMINAMATH_CALUDE_samantha_birthday_next_monday_l1097_109795


namespace NUMINAMATH_CALUDE_easter_egg_baskets_l1097_109779

/-- The number of Easter egg baskets given the number of people, eggs per person, and eggs per basket -/
def number_of_baskets (total_people : ℕ) (eggs_per_person : ℕ) (eggs_per_basket : ℕ) : ℕ :=
  (total_people * eggs_per_person) / eggs_per_basket

/-- Theorem stating that the number of Easter egg baskets is 15 -/
theorem easter_egg_baskets :
  let total_people : ℕ := 2 + 10 + 1 + 7
  let eggs_per_person : ℕ := 9
  let eggs_per_basket : ℕ := 12
  number_of_baskets total_people eggs_per_person eggs_per_basket = 15 := by
  sorry

#eval number_of_baskets 20 9 12

end NUMINAMATH_CALUDE_easter_egg_baskets_l1097_109779


namespace NUMINAMATH_CALUDE_expression_simplification_l1097_109703

theorem expression_simplification (y : ℝ) : 7*y + 8 - 2*y + 15 = 5*y + 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1097_109703


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1097_109791

theorem quadratic_roots_expression (p q : ℝ) : 
  (3 * p^2 - 7 * p + 4 = 0) →
  (3 * q^2 - 7 * q + 4 = 0) →
  p ≠ q →
  (5 * p^3 - 5 * q^3) / (p - q) = 185 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1097_109791


namespace NUMINAMATH_CALUDE_work_completion_l1097_109799

/-- The number of days A takes to complete the work alone -/
def days_A : ℝ := 4

/-- The number of days B takes to complete the work alone -/
def days_B : ℝ := 12

/-- The number of days B takes to finish the remaining work after A leaves -/
def days_B_remaining : ℝ := 4.000000000000001

/-- The number of days A and B work together -/
def days_together : ℝ := 2

theorem work_completion :
  let rate_A := 1 / days_A
  let rate_B := 1 / days_B
  let rate_together := rate_A + rate_B
  rate_together * days_together + rate_B * days_B_remaining = 1 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_l1097_109799


namespace NUMINAMATH_CALUDE_number_problem_l1097_109728

theorem number_problem : ∃ x : ℝ, x > 0 ∧ 0.9 * x = (4/5 * 25) + 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1097_109728


namespace NUMINAMATH_CALUDE_boys_meeting_time_l1097_109786

/-- Two boys running on a circular track meet after a specific time -/
theorem boys_meeting_time (track_length : Real) (speed1 speed2 : Real) :
  track_length = 4800 ∧ 
  speed1 = 60 * (1000 / 3600) ∧ 
  speed2 = 100 * (1000 / 3600) →
  track_length / (speed1 + speed2) = 108 := by
  sorry

end NUMINAMATH_CALUDE_boys_meeting_time_l1097_109786


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l1097_109736

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, b n > 0
  h_q_gt_one : q > 1
  h_geometric : ∀ n, b (n + 1) = b n * q

/-- In a geometric sequence with positive terms and common ratio greater than 1,
    the sum of the 6th and 7th terms is less than the sum of the 4th and 9th terms -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.b 6 + seq.b 7 < seq.b 4 + seq.b 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l1097_109736


namespace NUMINAMATH_CALUDE_power_equation_solution_l1097_109793

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 9^3 * 81^2 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1097_109793


namespace NUMINAMATH_CALUDE_moon_arrangements_count_l1097_109711

/-- The number of distinct arrangements of letters in "MOON" -/
def moon_arrangements : ℕ := 12

/-- The total number of letters in "MOON" -/
def total_letters : ℕ := 4

/-- The number of times 'O' appears in "MOON" -/
def o_count : ℕ := 2

/-- Theorem stating that the number of distinct arrangements of letters in "MOON" is 12 -/
theorem moon_arrangements_count : 
  moon_arrangements = (total_letters.factorial) / (o_count.factorial) := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_count_l1097_109711


namespace NUMINAMATH_CALUDE_min_dot_product_l1097_109739

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

/-- The circle C equation -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- A point on the circle C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : circle_C x y

/-- The dot product of tangent vectors PA and PB -/
def dot_product (P : PointOnC) (A B : ℝ × ℝ) : ℝ :=
  let PA := (A.1 - P.x, A.2 - P.y)
  let PB := (B.1 - P.x, B.2 - P.y)
  PA.1 * PB.1 + PA.2 * PB.2

/-- The theorem statement -/
theorem min_dot_product :
  ∀ P : PointOnC, ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    dot_product P A B ≥ 18 * Real.sqrt 2 - 27 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1097_109739


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l1097_109778

theorem magic_8_ball_probability :
  let n : ℕ := 6  -- total number of questions
  let k : ℕ := 3  -- number of positive answers we're looking for
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 160/729 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l1097_109778


namespace NUMINAMATH_CALUDE_inequality_proof_l1097_109753

theorem inequality_proof (p q r : ℝ) (n : ℕ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hpqr : p * q * r = 1) :
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1097_109753


namespace NUMINAMATH_CALUDE_field_length_problem_l1097_109734

theorem field_length_problem (w l : ℝ) (h1 : l = 2 * w) (h2 : 36 = (1 / 8) * (l * w)) : l = 24 :=
by sorry

end NUMINAMATH_CALUDE_field_length_problem_l1097_109734


namespace NUMINAMATH_CALUDE_seashells_given_to_jessica_l1097_109735

theorem seashells_given_to_jessica (original_seashells : ℕ) (seashells_left : ℕ) 
  (h1 : original_seashells = 56)
  (h2 : seashells_left = 22)
  (h3 : seashells_left < original_seashells) :
  original_seashells - seashells_left = 34 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_jessica_l1097_109735


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1097_109720

def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_function_properties (b c d : ℝ) :
  f b c d 0 = 2 ∧ 
  (∀ x, (6:ℝ)*x - f b c d (-1) + 7 = 0 ↔ x = -1) →
  f b c d (-1) = 1 ∧
  ∀ x, f b c d x = x^3 - 3*x^2 - 3*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1097_109720


namespace NUMINAMATH_CALUDE_scale_tower_height_l1097_109729

/-- Given a cylindrical tower and its scaled-down model, calculates the height of the model. -/
theorem scale_tower_height (actual_height : ℝ) (actual_volume : ℝ) (model_volume : ℝ) 
  (h1 : actual_height = 60) 
  (h2 : actual_volume = 80000)
  (h3 : model_volume = 0.5) :
  actual_height / Real.sqrt (actual_volume / model_volume) = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_scale_tower_height_l1097_109729


namespace NUMINAMATH_CALUDE_equation_solution_l1097_109751

theorem equation_solution (a b : ℝ) (h1 : 3 = (a + 5).sqrt) (h2 : 3 = (7 * a - 2 * b + 1)^(1/3)) :
  ∃ x : ℝ, (a * (x - 2)^2 - 9 * b = 0) ∧ (x = 7/2 ∨ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1097_109751


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1097_109771

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the skew relation between two lines
variable (skew_lines : Line → Line → Prop)

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : contained_in b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1097_109771


namespace NUMINAMATH_CALUDE_cubic_sum_from_system_l1097_109709

theorem cubic_sum_from_system (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^3 + y^3 = 416000 / 729 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_from_system_l1097_109709


namespace NUMINAMATH_CALUDE_no_real_solutions_l1097_109747

theorem no_real_solutions :
  ¬ ∃ x : ℝ, Real.sqrt (3 * x - 2) + 8 / Real.sqrt (3 * x - 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1097_109747


namespace NUMINAMATH_CALUDE_other_frisbee_price_is_3_l1097_109761

/-- Represents the price and sales of frisbees in a sporting goods store --/
structure FrisbeeSales where
  total_sold : ℕ
  total_receipts : ℕ
  price_other : ℚ
  min_sold_at_4 : ℕ

/-- Checks if the given FrisbeeSales satisfies the problem conditions --/
def is_valid_sale (sale : FrisbeeSales) : Prop :=
  sale.total_sold = 60 ∧
  sale.total_receipts = 204 ∧
  sale.min_sold_at_4 = 24 ∧
  sale.price_other * (sale.total_sold - sale.min_sold_at_4) + 4 * sale.min_sold_at_4 = sale.total_receipts

/-- Theorem stating that the price of the other frisbees is $3 --/
theorem other_frisbee_price_is_3 :
  ∀ (sale : FrisbeeSales), is_valid_sale sale → sale.price_other = 3 := by
  sorry

end NUMINAMATH_CALUDE_other_frisbee_price_is_3_l1097_109761


namespace NUMINAMATH_CALUDE_jack_marbles_l1097_109766

theorem jack_marbles (x : ℕ) : 
  (x ≥ 33) →  -- Jack must start with at least 33 marbles to share
  (x - 33 = 29) →  -- After sharing 33, Jack ends with 29
  x = 62 := by
sorry

end NUMINAMATH_CALUDE_jack_marbles_l1097_109766


namespace NUMINAMATH_CALUDE_straight_part_length_l1097_109726

/-- Represents a river with straight and crooked parts -/
structure River where
  total_length : ℝ
  straight_length : ℝ
  crooked_length : ℝ

/-- The condition that the straight part is three times shorter than the crooked part -/
def straight_three_times_shorter (r : River) : Prop :=
  r.straight_length = r.crooked_length / 3

/-- The theorem stating the length of the straight part given the conditions -/
theorem straight_part_length (r : River) 
  (h1 : r.total_length = 80)
  (h2 : r.total_length = r.straight_length + r.crooked_length)
  (h3 : straight_three_times_shorter r) : 
  r.straight_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_straight_part_length_l1097_109726


namespace NUMINAMATH_CALUDE_simplify_expression_l1097_109792

theorem simplify_expression (y : ℝ) : (5 * y)^3 + (4 * y) * (y^2) = 129 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1097_109792


namespace NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l1097_109770

theorem cube_sum_minus_triple_product (x y z : ℝ) 
  (h1 : x + y + z = 8) 
  (h2 : x*y + y*z + z*x = 20) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 32 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l1097_109770


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1097_109741

-- Define propositions p and q
variable (p q : Prop)

-- Define the original implication and its contrapositive
def original_implication := p → q
def contrapositive := ¬q → ¬p

-- Define necessary and sufficient conditions
def necessary (p q : Prop) := q → p
def sufficient (p q : Prop) := p → q

theorem p_necessary_not_sufficient (h1 : ¬(original_implication p q)) (h2 : contrapositive p q) :
  necessary p q ∧ ¬(sufficient p q) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1097_109741


namespace NUMINAMATH_CALUDE_f_x_plus_2_l1097_109721

/-- Given a function f where f(x) = x(x-1)/2, prove that f(x+2) = (x+2)(x+1)/2 -/
theorem f_x_plus_2 (f : ℝ → ℝ) (h : ∀ x, f x = x * (x - 1) / 2) :
  ∀ x, f (x + 2) = (x + 2) * (x + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_2_l1097_109721


namespace NUMINAMATH_CALUDE_apple_selection_probability_l1097_109732

def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def selected_apples : ℕ := 3

theorem apple_selection_probability :
  (Nat.choose green_apples 2 * Nat.choose yellow_apples 1) / Nat.choose total_apples selected_apples = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_selection_probability_l1097_109732


namespace NUMINAMATH_CALUDE_stone_pile_impossibility_l1097_109784

theorem stone_pile_impossibility :
  ∀ (n : ℕ) (stones piles : ℕ → ℕ),
  (stones 0 = 1001 ∧ piles 0 = 1) →
  (∀ k, stones (k + 1) + piles (k + 1) = stones k + piles k) →
  (∀ k, stones (k + 1) = stones k - 1) →
  (∀ k, piles (k + 1) = piles k + 1) →
  ¬∃ k, stones k = 3 * piles k :=
by sorry

end NUMINAMATH_CALUDE_stone_pile_impossibility_l1097_109784


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1097_109780

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1097_109780


namespace NUMINAMATH_CALUDE_kids_difference_l1097_109773

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 6) 
  (h2 : tuesday = 5) : 
  monday - tuesday = 1 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l1097_109773


namespace NUMINAMATH_CALUDE_tan_G_in_right_triangle_l1097_109710

theorem tan_G_in_right_triangle (GH FG : ℝ) (h_right_triangle : GH^2 + FG^2 = 25^2)
  (h_GH : GH = 20) (h_FG : FG = 25) : Real.tan (Real.arcsin (GH / FG)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_G_in_right_triangle_l1097_109710


namespace NUMINAMATH_CALUDE_adjacent_negative_product_l1097_109725

def a (n : ℕ) : ℤ := 2 * n - 17

theorem adjacent_negative_product :
  ∀ n : ℕ, (a n * a (n + 1) < 0) ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_adjacent_negative_product_l1097_109725


namespace NUMINAMATH_CALUDE_f_properties_l1097_109748

-- Define the properties of function f
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

def is_negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x < 0

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- State the theorem
theorem f_properties (f : ℝ → ℝ) 
  (h1 : is_additive f) (h2 : is_negative_for_positive f) : 
  is_odd f ∧ is_decreasing f := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1097_109748


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l1097_109798

theorem cupboard_cost_price (selling_price selling_price_with_profit : ℝ) 
  (h1 : selling_price = 0.88 * 6250)
  (h2 : selling_price_with_profit = 1.12 * 6250)
  (h3 : selling_price_with_profit = selling_price + 1500) : 
  6250 = 6250 := by
sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l1097_109798


namespace NUMINAMATH_CALUDE_legs_minus_twice_heads_diff_l1097_109787

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Nat → Nat
| 0 => 2  -- Chicken
| 1 => 4  -- Cow
| _ => 0  -- Other animals (not used in this problem)

/-- Calculates the total number of legs in the group -/
def total_legs (num_chickens num_cows : Nat) : Nat :=
  legs_per_animal 0 * num_chickens + legs_per_animal 1 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads (num_chickens num_cows : Nat) : Nat :=
  num_chickens + num_cows

/-- The main theorem stating the difference between legs and twice the heads -/
theorem legs_minus_twice_heads_diff (num_chickens : Nat) : 
  total_legs num_chickens 7 - 2 * total_heads num_chickens 7 = 14 := by
  sorry

#check legs_minus_twice_heads_diff

end NUMINAMATH_CALUDE_legs_minus_twice_heads_diff_l1097_109787


namespace NUMINAMATH_CALUDE_clothing_distribution_l1097_109714

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 39)
  (h2 : first_load = 19)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 4 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l1097_109714


namespace NUMINAMATH_CALUDE_range_of_a_l1097_109756

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ∈ {x : ℝ | x ≥ 5} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1097_109756


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l1097_109777

theorem magnitude_of_complex_fourth_power :
  Complex.abs ((5 : ℂ) + (2 * Complex.I * Real.sqrt 3)) ^ 4 = 1369 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l1097_109777


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1097_109797

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1097_109797


namespace NUMINAMATH_CALUDE_circle_intersection_chord_l1097_109794

/-- Given two circles C₁ and C₂, where C₁ passes through the center of C₂,
    the equation of their chord of intersection is 5x + y - 19 = 0 -/
theorem circle_intersection_chord 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, C₁ x y ↔ (x + 1)^2 + y^2 = r^2) 
  (h₂ : ∀ x y, C₂ x y ↔ (x - 4)^2 + (y - 1)^2 = 4) 
  (h₃ : C₁ 4 1) :
  ∀ x y, (C₁ x y ∧ C₂ x y) ↔ 5*x + y - 19 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_chord_l1097_109794


namespace NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l1097_109731

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat_record (ounces_per_hamburger : ℕ) (last_year_winner_ounces : ℕ) : ℕ :=
  (last_year_winner_ounces / ounces_per_hamburger) + 1

/-- Theorem stating that Tonya needs to eat 22 hamburgers to beat last year's winner -/
theorem tonya_needs_22_hamburgers :
  hamburgers_to_beat_record 4 84 = 22 := by
  sorry

end NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l1097_109731


namespace NUMINAMATH_CALUDE_negation_of_existential_l1097_109746

theorem negation_of_existential (p : Prop) :
  (¬ ∃ (x : ℝ), x^2 > 1) ↔ (∀ (x : ℝ), x^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_l1097_109746


namespace NUMINAMATH_CALUDE_distinct_cube_edge_colorings_l1097_109740

/-- The group of rotations of the cube -/
structure CubeRotationGroup where
  D : Type
  mul : D → D → D

/-- The permutation group of the edges of the cube induced by the rotation group -/
structure EdgePermutationGroup where
  W : Type
  comp : W → W → W

/-- The cycle index polynomial for the permutation group (W, ∘) -/
def cycle_index_polynomial (W : EdgePermutationGroup) : ℕ :=
  sorry

/-- The number of distinct colorings for a given permutation type -/
def colorings_for_permutation (perm_type : String) : ℕ :=
  sorry

/-- Theorem: The number of distinct ways to color the edges of a cube with 3 red, 3 blue, and 6 yellow edges is 780 -/
theorem distinct_cube_edge_colorings :
  let num_edges : ℕ := 12
  let num_red : ℕ := 3
  let num_blue : ℕ := 3
  let num_yellow : ℕ := 6
  (num_red + num_blue + num_yellow = num_edges) →
  (∃ (W : EdgePermutationGroup),
    (cycle_index_polynomial W *
     (colorings_for_permutation "1^12" +
      8 * colorings_for_permutation "3^4" +
      6 * colorings_for_permutation "1^2 2^5")) / 24 = 780) :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_edge_colorings_l1097_109740


namespace NUMINAMATH_CALUDE_egg_count_theorem_l1097_109768

/-- Represents a carton of eggs -/
structure EggCarton where
  total_yolks : ℕ
  double_yolk_eggs : ℕ

/-- Calculate the number of eggs in a carton -/
def count_eggs (carton : EggCarton) : ℕ :=
  carton.double_yolk_eggs + (carton.total_yolks - 2 * carton.double_yolk_eggs)

/-- Theorem: A carton with 17 yolks and 5 double-yolk eggs contains 12 eggs -/
theorem egg_count_theorem (carton : EggCarton) 
  (h1 : carton.total_yolks = 17) 
  (h2 : carton.double_yolk_eggs = 5) : 
  count_eggs carton = 12 := by
  sorry

#eval count_eggs { total_yolks := 17, double_yolk_eggs := 5 }

end NUMINAMATH_CALUDE_egg_count_theorem_l1097_109768


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_one_half_open_l1097_109700

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define set N
def N : Set ℝ := {x : ℝ | x < 1}

-- Theorem statement
theorem M_intersect_N_equals_zero_one_half_open : M ∩ N = Set.Icc 0 1 ∩ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_one_half_open_l1097_109700


namespace NUMINAMATH_CALUDE_simplify_expression_l1097_109727

theorem simplify_expression (n : ℕ) : 
  (3^(n+4) - 3*(3^n) - 3^(n+2)) / (3*(3^(n+3))) = 23/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1097_109727


namespace NUMINAMATH_CALUDE_parametric_equation_of_lineL_l1097_109718

/-- A line in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (l : Line2D) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

/-- The line passing through (3, 5) and parallel to (4, 2) -/
def lineL : Line2D :=
  { point := (3, 5)
    direction := (4, 2) }

/-- Theorem: The parametric equation (x - 3)/4 = (y - 5)/2 represents lineL -/
theorem parametric_equation_of_lineL :
  ∀ x y : ℝ, pointOnLine lineL (x, y) ↔ (x - 3) / 4 = (y - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_equation_of_lineL_l1097_109718


namespace NUMINAMATH_CALUDE_triangle_area_l1097_109733

theorem triangle_area (A B C : ℝ) (a : ℝ) (h1 : a = 2) (h2 : C = π/4) (h3 : Real.tan (B/2) = 1/2) :
  (1/2) * a * (Real.sin C) * (8 * Real.sqrt 2 / 7) = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1097_109733


namespace NUMINAMATH_CALUDE_factorial_ratio_50_48_l1097_109754

theorem factorial_ratio_50_48 : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_50_48_l1097_109754


namespace NUMINAMATH_CALUDE_min_value_expression_l1097_109764

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  ∃ (min : ℝ), min = Real.sqrt 10 + Real.sqrt 5 ∧
  ∀ (x : ℝ), (a * c / b) + (c / (a * b)) - (c / 2) + (Real.sqrt 5 / (c - 2)) ≥ x → x ≤ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1097_109764


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1097_109763

/-- Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a = b) →  -- Perpendicular asymptotes condition
  (2 * (a^2 + b^2).sqrt = 8) →  -- Focal length condition
  ((a^2 + b^2).sqrt / a = Real.sqrt 2) := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1097_109763


namespace NUMINAMATH_CALUDE_inequality_proof_l1097_109724

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1097_109724


namespace NUMINAMATH_CALUDE_inequality_implication_l1097_109755

theorem inequality_implication (x y : ℝ) (h : x > y) : (1/2 : ℝ)^x < (1/2 : ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1097_109755


namespace NUMINAMATH_CALUDE_adams_money_from_mother_l1097_109785

/-- Given Adam's initial savings and final total, prove the amount his mother gave him. -/
theorem adams_money_from_mother (initial_savings final_total : ℕ) 
  (h1 : initial_savings = 79)
  (h2 : final_total = 92)
  : final_total - initial_savings = 13 := by
  sorry

end NUMINAMATH_CALUDE_adams_money_from_mother_l1097_109785


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1097_109769

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x, x^3 - 24*x^2 + 88*x - 75 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 24*s^2 + 88*s - 75) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 256 :=
by
  sorry

#check root_sum_reciprocal

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1097_109769


namespace NUMINAMATH_CALUDE_distinct_products_count_l1097_109749

def S : Finset ℕ := {1, 3, 7, 9, 13}

def products : Finset ℕ :=
  (S.powerset.filter (λ s => s.card ≥ 2)).image (λ s => s.prod id)

theorem distinct_products_count : products.card = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_products_count_l1097_109749


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l1097_109758

/-- Proves that the initial volume of a mixture is 150 liters, given the conditions of the problem -/
theorem mixture_volume_proof (initial_water_percentage : Real) 
                              (added_water : Real) 
                              (final_water_percentage : Real) : 
  initial_water_percentage = 0.1 →
  added_water = 30 →
  final_water_percentage = 0.25 →
  ∃ (initial_volume : Real),
    initial_volume * initial_water_percentage + added_water = 
    (initial_volume + added_water) * final_water_percentage ∧
    initial_volume = 150 := by
  sorry


end NUMINAMATH_CALUDE_mixture_volume_proof_l1097_109758


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1097_109716

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube. -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  ∃ (inner_cube_volume : ℝ),
    inner_cube_volume = 192 * Real.sqrt 3 ∧
    inner_cube_volume = (outer_cube_edge / Real.sqrt 3) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1097_109716


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_l1097_109759

theorem circle_area_with_diameter (d : ℝ) (A : ℝ) :
  d = 7.5 →
  A = π * (d / 2)^2 →
  A = 14.0625 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_l1097_109759


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1097_109742

theorem sqrt_equation_solution : 
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 13 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1097_109742


namespace NUMINAMATH_CALUDE_abs_value_sum_and_diff_l1097_109781

theorem abs_value_sum_and_diff (a b : ℝ) :
  (abs a = 5 ∧ abs b = 3) →
  ((a > 0 ∧ b < 0) → a + b = 2) ∧
  (abs (a + b) = a + b → (a - b = 2 ∨ a - b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_abs_value_sum_and_diff_l1097_109781


namespace NUMINAMATH_CALUDE_line_extraction_theorem_l1097_109719

-- Define a structure for a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def areParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- Function to check if a line intersects all given lines
def intersectsAllLines (l : Line) (l1 l2 l3 l4 : Line) : Prop :=
  ∃ p1 p2 p3 p4 : Point,
    pointOnLine p1 l ∧ pointOnLine p1 l1 ∧
    pointOnLine p2 l ∧ pointOnLine p2 l2 ∧
    pointOnLine p3 l ∧ pointOnLine p3 l3 ∧
    pointOnLine p4 l ∧ pointOnLine p4 l4

-- Function to check if segments are in given ratios
def segmentsInRatio (l : Line) (l1 l2 l3 : Line) (r1 r2 : ℝ) : Prop :=
  ∃ p1 p2 p3 : Point,
    pointOnLine p1 l ∧ pointOnLine p1 l1 ∧
    pointOnLine p2 l ∧ pointOnLine p2 l2 ∧
    pointOnLine p3 l ∧ pointOnLine p3 l3 ∧
    (p2.x - p1.x)^2 + (p2.y - p1.y)^2 = r1 * ((p3.x - p2.x)^2 + (p3.y - p2.y)^2) ∧
    (p3.x - p2.x)^2 + (p3.y - p2.y)^2 = r2 * ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Theorem statement
theorem line_extraction_theorem (l1 l2 l3 l4 : Line) (r1 r2 : ℝ) :
  (∃ l : Line, intersectsAllLines l l1 l2 l3 l4 ∧ segmentsInRatio l l1 l2 l3 r1 r2) ∨
  (∃ m : Line, segmentsInRatio m l1 l2 l3 r1 r2 ∧ (areParallel m l4 ∨ m = l4)) :=
sorry

end NUMINAMATH_CALUDE_line_extraction_theorem_l1097_109719


namespace NUMINAMATH_CALUDE_first_class_students_l1097_109774

theorem first_class_students (avg_first : ℝ) (num_second : ℕ) (avg_second : ℝ) (avg_total : ℝ) 
  (h1 : avg_first = 40)
  (h2 : num_second = 50)
  (h3 : avg_second = 60)
  (h4 : avg_total = 52.5) :
  ∃ (num_first : ℕ), 
    (num_first : ℝ) * avg_first + (num_second : ℝ) * avg_second = 
    (num_first + num_second : ℝ) * avg_total ∧ num_first = 30 :=
by sorry

end NUMINAMATH_CALUDE_first_class_students_l1097_109774


namespace NUMINAMATH_CALUDE_givenEquationIsParabola_l1097_109713

/-- Represents a conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines if an equation represents a parabola -/
def isParabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ 
    ∀ x y : ℝ, f x y ↔ (a * y^2 + b * y + c * x + d = 0 ∨ a * x^2 + b * x + c * y + d = 0)

/-- The given equation -/
def givenEquation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

/-- Theorem stating that the given equation represents a parabola -/
theorem givenEquationIsParabola : isParabola givenEquation := by
  sorry

/-- The conic type of the given equation is a parabola -/
def conicTypeOfGivenEquation : ConicType := ConicType.Parabola

end NUMINAMATH_CALUDE_givenEquationIsParabola_l1097_109713


namespace NUMINAMATH_CALUDE_composite_sequence_existence_l1097_109776

theorem composite_sequence_existence (m : ℕ) (hm : m > 0) :
  ∃ n : ℕ, ∀ i : ℤ, -m ≤ i ∧ i ≤ m → 
    (2 : ℕ)^n + i > 0 ∧ ¬(Nat.Prime ((2 : ℕ)^n + i).toNat) := by
  sorry

end NUMINAMATH_CALUDE_composite_sequence_existence_l1097_109776


namespace NUMINAMATH_CALUDE_same_solution_implies_c_equals_nine_l1097_109737

theorem same_solution_implies_c_equals_nine (x c : ℝ) :
  (3 * x + 5 = 4) ∧ (c * x + 6 = 3) → c = 9 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_equals_nine_l1097_109737


namespace NUMINAMATH_CALUDE_equal_share_problem_l1097_109707

theorem equal_share_problem (total_amount : ℚ) (num_people : ℕ) :
  total_amount = 3.75 →
  num_people = 3 →
  total_amount / num_people = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_problem_l1097_109707


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1097_109715

/-- Given an arithmetic sequence with certain properties, prove its maximum sum -/
theorem arithmetic_sequence_max_sum (k : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) :
  k ≥ 2 →
  S (k - 1) = 8 →
  S k = 0 →
  S (k + 1) = -10 →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∃ d : ℤ, ∀ n, a (n + 1) - a n = d) →
  (∃ n : ℕ, ∀ m : ℕ, S m ≤ S n) →
  (∃ n : ℕ, S n = 20) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1097_109715


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l1097_109767

theorem cupboard_cost_price (C : ℝ) : C = 7450 :=
  let selling_price := C * 0.86
  let profitable_price := C * 1.14
  have h1 : profitable_price = selling_price + 2086 := by sorry
  sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l1097_109767


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l1097_109722

-- Define the sets P and Q
def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | -2 < x ∧ x < 0}

-- Define the open interval (-2, 1)
def openInterval : Set ℝ := {x | -2 < x ∧ x < 1}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = openInterval := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l1097_109722


namespace NUMINAMATH_CALUDE_adams_final_balance_l1097_109772

/-- Calculates the final balance after a series of transactions --/
def final_balance (initial : ℚ) (spent : List ℚ) (received : List ℚ) : ℚ :=
  initial - spent.sum + received.sum

/-- Theorem: Adam's final balance is $10.75 --/
theorem adams_final_balance :
  let initial : ℚ := 5
  let spent : List ℚ := [2, 1.5, 0.75]
  let received : List ℚ := [3, 2, 5]
  final_balance initial spent received = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_adams_final_balance_l1097_109772


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1097_109701

theorem absolute_value_equation (x z : ℝ) 
  (h : |2*x - Real.log z| = 2*x + Real.log z) : x * (z - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1097_109701


namespace NUMINAMATH_CALUDE_polynomial_invariant_under_increment_l1097_109796

def P (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x

theorem polynomial_invariant_under_increment :
  ∀ x : ℝ, P x = P (x + 1) ↔ x = 1 ∨ x = 4/3 := by sorry

end NUMINAMATH_CALUDE_polynomial_invariant_under_increment_l1097_109796


namespace NUMINAMATH_CALUDE_percentage_equality_l1097_109717

theorem percentage_equality :
  ∃! k : ℚ, (k / 100) * 25 = (20 / 100) * 30 := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l1097_109717


namespace NUMINAMATH_CALUDE_r_value_for_s_seven_l1097_109775

/-- Given R = 2gS + 3 and R = 23 when S = 5, prove that R = 31 when S = 7 -/
theorem r_value_for_s_seven (g : ℝ) : 
  (∀ S : ℝ, 2 * g * S + 3 = 23 → S = 5) → 
  (∃ R : ℝ, ∀ S : ℝ, R = 2 * g * S + 3) → 
  (∃ R : ℝ, R = 2 * g * 7 + 3 ∧ R = 31) := by
sorry


end NUMINAMATH_CALUDE_r_value_for_s_seven_l1097_109775


namespace NUMINAMATH_CALUDE_subset_implies_lower_bound_l1097_109745

theorem subset_implies_lower_bound (a : ℝ) : 
  let M := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
  let N := {x : ℝ | x ≤ a}
  M ⊆ N → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_lower_bound_l1097_109745


namespace NUMINAMATH_CALUDE_only_finance_opposite_meanings_l1097_109708

-- Define a type for quantity pairs
inductive QuantityPair
  | Distance (d1 d2 : ℕ)
  | Finance (f1 f2 : ℤ)
  | HeightWeight (h w : ℚ)
  | Scores (s1 s2 : ℕ)

-- Define a function to check if a pair has opposite meanings
def hasOppositeMeanings (pair : QuantityPair) : Prop :=
  match pair with
  | QuantityPair.Finance f1 f2 => f1 * f2 < 0
  | _ => False

-- Theorem statement
theorem only_finance_opposite_meanings 
  (a : QuantityPair) 
  (b : QuantityPair) 
  (c : QuantityPair) 
  (d : QuantityPair) 
  (ha : a = QuantityPair.Distance 500 200)
  (hb : b = QuantityPair.Finance (-3000) 12000)
  (hc : c = QuantityPair.HeightWeight 1.5 (-2.4))
  (hd : d = QuantityPair.Scores 50 70) :
  hasOppositeMeanings b ∧ 
  ¬hasOppositeMeanings a ∧ 
  ¬hasOppositeMeanings c ∧ 
  ¬hasOppositeMeanings d := by
  sorry

end NUMINAMATH_CALUDE_only_finance_opposite_meanings_l1097_109708


namespace NUMINAMATH_CALUDE_number_of_girls_is_760_l1097_109788

/-- Represents the number of students in a school survey --/
structure SchoolSurvey where
  total_students : ℕ
  sample_size : ℕ
  girls_sampled_difference : ℕ

/-- Calculates the number of girls in the school based on survey data --/
def number_of_girls (survey : SchoolSurvey) : ℕ :=
  survey.total_students / 2 - survey.girls_sampled_difference * (survey.total_students / survey.sample_size / 2)

/-- Theorem stating that given the survey conditions, the number of girls in the school is 760 --/
theorem number_of_girls_is_760 (survey : SchoolSurvey) 
    (h1 : survey.total_students = 1600)
    (h2 : survey.sample_size = 200)
    (h3 : survey.girls_sampled_difference = 10) : 
  number_of_girls survey = 760 := by
  sorry

#eval number_of_girls { total_students := 1600, sample_size := 200, girls_sampled_difference := 10 }

end NUMINAMATH_CALUDE_number_of_girls_is_760_l1097_109788


namespace NUMINAMATH_CALUDE_second_meeting_day_correct_l1097_109765

/-- Represents the number of days between visits for each schoolchild -/
def VisitSchedule : Fin 4 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 6
  | 3 => 9

/-- The day when all schoolchildren meet for the second time -/
def SecondMeetingDay : ℕ := 360

theorem second_meeting_day_correct :
  SecondMeetingDay = 2 * Nat.lcm (VisitSchedule 0) (Nat.lcm (VisitSchedule 1) (Nat.lcm (VisitSchedule 2) (VisitSchedule 3))) :=
by sorry

#check second_meeting_day_correct

end NUMINAMATH_CALUDE_second_meeting_day_correct_l1097_109765
