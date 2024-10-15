import Mathlib

namespace NUMINAMATH_CALUDE_star_value_for_specific_conditions_l2981_298183

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value_for_specific_conditions (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 16) 
  (h4 : a^2 + b^2 = 136) : 
  star a b = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_star_value_for_specific_conditions_l2981_298183


namespace NUMINAMATH_CALUDE_max_l_value_l2981_298155

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 8 * x + 3

-- Define the condition for l(a)
def is_valid_l (a : ℝ) (l : ℝ) : Prop :=
  a < 0 ∧ l > 0 ∧ ∀ x ∈ Set.Icc 0 l, |f a x| ≤ 5

-- Define l(a) as the supremum of valid l values
noncomputable def l (a : ℝ) : ℝ :=
  ⨆ (l : ℝ) (h : is_valid_l a l), l

-- State the theorem
theorem max_l_value :
  ∃ (a : ℝ), a < 0 ∧ l a = (Real.sqrt 5 + 1) / 2 ∧
  ∀ (b : ℝ), b < 0 → l b ≤ l a :=
sorry

end NUMINAMATH_CALUDE_max_l_value_l2981_298155


namespace NUMINAMATH_CALUDE_solve_equation_l2981_298143

theorem solve_equation : 
  ∃ x : ℝ, (4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470) ∧ (x = 13.26) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2981_298143


namespace NUMINAMATH_CALUDE_shift_increasing_interval_l2981_298161

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shift_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 4)) (-6) (-1) := by
  sorry

end NUMINAMATH_CALUDE_shift_increasing_interval_l2981_298161


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2981_298158

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_p = q and S_q = p where p ≠ q, then S_{p+q} = -(p + q) -/
theorem arithmetic_sequence_sum_property (a : ArithmeticSequence) (p q : ℕ) 
    (h1 : a.S p = q)
    (h2 : a.S q = p)
    (h3 : p ≠ q) : 
  a.S (p + q) = -(p + q) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2981_298158


namespace NUMINAMATH_CALUDE_product_digit_sum_base7_l2981_298160

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  toBase7 (toBase10 a * toBase10 b)

theorem product_digit_sum_base7 : 
  sumDigitsBase7 (multiplyBase7 35 42) = 21 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base7_l2981_298160


namespace NUMINAMATH_CALUDE_john_old_cards_l2981_298176

/-- The number of baseball cards John puts on each page of the binder -/
def cards_per_page : ℕ := 3

/-- The number of new cards John has -/
def new_cards : ℕ := 8

/-- The total number of pages John used in the binder -/
def total_pages : ℕ := 8

/-- The number of old cards John had -/
def old_cards : ℕ := total_pages * cards_per_page - new_cards

theorem john_old_cards : old_cards = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_old_cards_l2981_298176


namespace NUMINAMATH_CALUDE_juice_price_ratio_l2981_298128

theorem juice_price_ratio :
  let volume_A : ℝ := 1.25  -- Brand A's volume relative to Brand B
  let price_A : ℝ := 0.85   -- Brand A's price relative to Brand B
  let unit_price_ratio := (price_A / volume_A) / 1  -- Ratio of unit prices (A / B)
  unit_price_ratio = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_juice_price_ratio_l2981_298128


namespace NUMINAMATH_CALUDE_solution_set_of_increasing_function_l2981_298174

theorem solution_set_of_increasing_function 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) : 
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_increasing_function_l2981_298174


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2981_298145

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 0)
  (h_sum2 : a 4 + a 5 + a 6 = 18) :
  ∀ n : ℕ, a n = 2 * n - 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2981_298145


namespace NUMINAMATH_CALUDE_davids_biology_marks_l2981_298114

theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 91)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 78)
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) :
  biology = 85 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l2981_298114


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l2981_298192

-- Define the quadratic function
def y (x m : ℝ) : ℝ := x^2 + 2*m*x - 3*m + 1

-- Define the conditions
def condition1 (p q : ℝ) : Prop := 4*p^2 + 9*q^2 = 2
def condition2 (x p q : ℝ) : Prop := (1/2)*x + 3*p*q = 1

-- State the theorem
theorem quadratic_function_minimum (x p q m : ℝ) :
  condition1 p q →
  condition2 x p q →
  (∀ x', y x' m ≥ 1) →
  (∃ x'', y x'' m = 1) →
  (m = -3 ∨ m = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l2981_298192


namespace NUMINAMATH_CALUDE_stations_visited_l2981_298187

theorem stations_visited (total_nails : ℕ) (nails_per_station : ℕ) (h1 : total_nails = 140) (h2 : nails_per_station = 7) :
  total_nails / nails_per_station = 20 := by
sorry

end NUMINAMATH_CALUDE_stations_visited_l2981_298187


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2981_298154

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2981_298154


namespace NUMINAMATH_CALUDE_rolling_circle_trajectory_is_line_segment_l2981_298142

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point on a circle -/
structure PointOnCircle where
  circle : Circle
  angle : ℝ  -- Angle from the positive x-axis

/-- Represents the trajectory of a point -/
inductive Trajectory
  | LineSegment : (ℝ × ℝ) → (ℝ × ℝ) → Trajectory

/-- The trajectory of a fixed point on a circle rolling inside another circle -/
def rollingCircleTrajectory (stationaryCircle : Circle) (movingCircle : Circle) (fixedPoint : PointOnCircle) : Trajectory :=
  sorry

/-- Main theorem: The trajectory of a fixed point on a circle rolling inside another circle with twice its radius is a line segment -/
theorem rolling_circle_trajectory_is_line_segment
  (stationaryCircle : Circle)
  (movingCircle : Circle)
  (fixedPoint : PointOnCircle)
  (h1 : movingCircle.radius = stationaryCircle.radius / 2)
  (h2 : fixedPoint.circle = movingCircle) :
  ∃ (p q : ℝ × ℝ), rollingCircleTrajectory stationaryCircle movingCircle fixedPoint = Trajectory.LineSegment p q ∧
                    p = stationaryCircle.center :=
  sorry

end NUMINAMATH_CALUDE_rolling_circle_trajectory_is_line_segment_l2981_298142


namespace NUMINAMATH_CALUDE_hat_cost_l2981_298182

/-- The cost of each hat when a person has enough hats for 2 weeks and the total cost is $700 -/
theorem hat_cost (num_weeks : ℕ) (days_per_week : ℕ) (total_cost : ℕ) : 
  num_weeks = 2 → days_per_week = 7 → total_cost = 700 → 
  total_cost / (num_weeks * days_per_week) = 50 := by
  sorry

end NUMINAMATH_CALUDE_hat_cost_l2981_298182


namespace NUMINAMATH_CALUDE_negative_64_to_7_6th_l2981_298135

theorem negative_64_to_7_6th : ∃ (z : ℂ), z^6 = (-64)^7 ∧ z = 128 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_7_6th_l2981_298135


namespace NUMINAMATH_CALUDE_whole_substitution_problems_l2981_298157

theorem whole_substitution_problems :
  -- Problem 1
  (∀ m n : ℝ, m - n = -1 → 2 * (m - n)^2 + 18 = 20) ∧
  -- Problem 2
  (∀ m n : ℝ, m^2 + 2*m*n = 10 ∧ n^2 + 3*m*n = 6 → 2*m^2 + n^2 + 7*m*n = 26) ∧
  -- Problem 3
  (∀ a b c m : ℝ, a*(-1)^5 + b*(-1)^3 + c*(-1) - 5 = m → 
    a*(1)^5 + b*(1)^3 + c*(1) - 5 = -m - 10) :=
by sorry

end NUMINAMATH_CALUDE_whole_substitution_problems_l2981_298157


namespace NUMINAMATH_CALUDE_student_survey_l2981_298195

theorem student_survey (french_and_english : ℕ) (french_not_english : ℕ) 
  (h1 : french_and_english = 20)
  (h2 : french_not_english = 60)
  (h3 : french_and_english + french_not_english = (2 : ℝ) / 5 * total_students) :
  total_students = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_student_survey_l2981_298195


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l2981_298112

/-- Given two perpendicular lines and the foot of the perpendicular, prove that a + b + c = -4 -/
theorem perpendicular_lines_sum (a b c : ℝ) : 
  (∀ x y, a * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + b = 0) →  -- lines are perpendicular
  (a + 4 * c - 2 = 0) →  -- foot of perpendicular satisfies first line equation
  (2 - 5 * c + b = 0) →  -- foot of perpendicular satisfies second line equation
  (a * 2 + 4 * 5 = 0) →  -- perpendicularity condition
  a + b + c = -4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l2981_298112


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2981_298177

noncomputable section

/-- Two circles with common center and a point configuration --/
structure CircleConfig where
  a : ℝ
  b : ℝ
  h_ab : a > b

variable (cfg : CircleConfig)

/-- The locus of points Si --/
def locus (cfg : CircleConfig) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ cfg.b^2 / cfg.a ∧
    p.1 = t * cfg.a^2 / cfg.b^2 ∧
    p.2^2 = cfg.b^2 - (t * cfg.a / cfg.b)^2}

/-- The ellipse with major axis 2a and minor axis 2b --/
def ellipse (cfg : CircleConfig) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / cfg.a^2 + p.2^2 / cfg.b^2 = 1 ∧ p.1 ≥ 0}

/-- The main theorem --/
theorem locus_is_ellipse (cfg : CircleConfig) :
  locus cfg = ellipse cfg := by sorry

end

end NUMINAMATH_CALUDE_locus_is_ellipse_l2981_298177


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2981_298147

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Theorem statement
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2981_298147


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l2981_298103

/-- Proves that John's candy store spending is $0.88 given his allowance and spending pattern -/
theorem johns_candy_store_spending (allowance : ℚ) : 
  allowance = 33/10 →
  let arcade_spending := 3/5 * allowance
  let remaining_after_arcade := allowance - arcade_spending
  let toy_store_spending := 1/3 * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 88/100 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l2981_298103


namespace NUMINAMATH_CALUDE_problem_solution_l2981_298118

def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x, m = 4 → (p x ∧ q x m) → (4 < x ∧ x < 5)) ∧
  ((∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x m)) → (5/3 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2981_298118


namespace NUMINAMATH_CALUDE_isosceles_max_perimeter_l2981_298191

-- Define a triangle
structure Triangle where
  base : ℝ
  angle : ℝ
  side1 : ℝ
  side2 : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.base + t.side1 + t.side2

-- Define an isosceles triangle
def isIsosceles (t : Triangle) : Prop := t.side1 = t.side2

-- Theorem statement
theorem isosceles_max_perimeter (b a : ℝ) :
  ∀ (t : Triangle), t.base = b → t.angle = a →
    ∃ (t_iso : Triangle), t_iso.base = b ∧ t_iso.angle = a ∧ isIsosceles t_iso ∧
      perimeter t_iso ≥ perimeter t :=
sorry

end NUMINAMATH_CALUDE_isosceles_max_perimeter_l2981_298191


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l2981_298185

/-- Given a point P(-4t, 3t) on the terminal side of angle θ, where t ≠ 0,
    the value of 2sinθ + cosθ is either 2/5 or -2/5. -/
theorem angle_terminal_side_value (t : ℝ) (θ : ℝ) (h : t ≠ 0) :
  let P : ℝ × ℝ := (-4 * t, 3 * t)
  (∃ (k : ℝ), k > 0 ∧ P = k • (Real.cos θ, Real.sin θ)) →
  2 * Real.sin θ + Real.cos θ = 2 / 5 ∨ 2 * Real.sin θ + Real.cos θ = -2 / 5 :=
by sorry


end NUMINAMATH_CALUDE_angle_terminal_side_value_l2981_298185


namespace NUMINAMATH_CALUDE_conference_left_handed_fraction_l2981_298165

theorem conference_left_handed_fraction 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h1 : red + blue = total) 
  (h2 : red = 2 * blue) 
  (h3 : red > 0) 
  (h4 : blue > 0) : 
  (red * (1/3) + blue * (2/3)) / total = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_conference_left_handed_fraction_l2981_298165


namespace NUMINAMATH_CALUDE_sqrt_five_power_calculation_l2981_298139

theorem sqrt_five_power_calculation :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 78125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_power_calculation_l2981_298139


namespace NUMINAMATH_CALUDE_sum_of_min_max_cubic_expression_l2981_298194

theorem sum_of_min_max_cubic_expression (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 30) :
  let f := fun (x y z w : ℝ) => 4 * (x^3 + y^3 + z^3 + w^3) - 3 * (x^2 + y^2 + z^2 + w^2)^2
  (⨅ (p : Fin 4 → ℝ) (h : p 0 + p 1 + p 2 + p 3 = 10 ∧ p 0^2 + p 1^2 + p 2^2 + p 3^2 = 30), f (p 0) (p 1) (p 2) (p 3)) +
  (⨆ (p : Fin 4 → ℝ) (h : p 0 + p 1 + p 2 + p 3 = 10 ∧ p 0^2 + p 1^2 + p 2^2 + p 3^2 = 30), f (p 0) (p 1) (p 2) (p 3)) = 404 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_min_max_cubic_expression_l2981_298194


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l2981_298101

-- Define the function
def f (x : ℝ) : ℝ := 3*x - x^2

-- State the theorem
theorem monotonic_increasing_interval :
  ∀ x y : ℝ, x < y ∧ x < (3/2) ∧ y < (3/2) → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l2981_298101


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l2981_298168

/-- Calculates the total cost of circus tickets -/
def total_ticket_cost (adult_price children_price senior_price : ℕ) 
  (adult_count children_count senior_count : ℕ) : ℕ :=
  adult_price * adult_count + children_price * children_count + senior_price * senior_count

/-- Proves that the total cost of circus tickets for the given quantities and prices is $318 -/
theorem circus_ticket_cost : 
  total_ticket_cost 55 28 42 4 2 1 = 318 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l2981_298168


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l2981_298184

/-- A line in 3D space -/
structure Line3D where
  -- We represent a line by a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Perpendicular relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicularity
  sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallelism
  sorry

theorem perpendicular_lines_theorem (a b c d : Line3D) 
  (h1 : perpendicular a b)
  (h2 : perpendicular b c)
  (h3 : perpendicular c d)
  (h4 : perpendicular d a) :
  parallel b d ∨ parallel a c :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l2981_298184


namespace NUMINAMATH_CALUDE_diamond_symmetry_lines_l2981_298180

/-- Definition of the diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ◊ y = y ◊ x forms four lines -/
theorem diamond_symmetry_lines :
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1} =
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2} :=
by sorry

end NUMINAMATH_CALUDE_diamond_symmetry_lines_l2981_298180


namespace NUMINAMATH_CALUDE_power_sum_properties_l2981_298141

theorem power_sum_properties (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
sorry

end NUMINAMATH_CALUDE_power_sum_properties_l2981_298141


namespace NUMINAMATH_CALUDE_greatest_multiple_of_three_cubed_less_than_1000_l2981_298126

theorem greatest_multiple_of_three_cubed_less_than_1000 :
  ∃ (x : ℕ), 
    x > 0 ∧ 
    ∃ (k : ℕ), x = 3 * k ∧ 
    x^3 < 1000 ∧
    ∀ (y : ℕ), y > 0 → (∃ (m : ℕ), y = 3 * m) → y^3 < 1000 → y ≤ x ∧
    x = 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_three_cubed_less_than_1000_l2981_298126


namespace NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_l2981_298119

theorem not_p_or_not_q_false_implies (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : 
  (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_l2981_298119


namespace NUMINAMATH_CALUDE_new_average_after_changes_l2981_298148

theorem new_average_after_changes (numbers : Finset ℕ) (original_sum : ℕ) : 
  numbers.card = 15 → 
  original_sum = numbers.sum id →
  original_sum / numbers.card = 40 →
  let new_sum := original_sum + 9 * 10 - 6 * 5
  new_sum / numbers.card = 44 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_changes_l2981_298148


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2981_298166

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2981_298166


namespace NUMINAMATH_CALUDE_square_width_proof_l2981_298151

theorem square_width_proof (rectangle_length : ℝ) (rectangle_width : ℝ) (area_difference : ℝ) :
  rectangle_length = 3 →
  rectangle_width = 6 →
  area_difference = 7 →
  ∃ (square_width : ℝ), square_width^2 = rectangle_length * rectangle_width - area_difference :=
by
  sorry

end NUMINAMATH_CALUDE_square_width_proof_l2981_298151


namespace NUMINAMATH_CALUDE_show_dog_cost_l2981_298115

/-- Proves that the cost of each show dog is $250 given the problem conditions -/
theorem show_dog_cost (num_dogs : ℕ) (num_puppies : ℕ) (puppy_price : ℕ) (total_profit : ℕ) : 
  num_dogs = 2 →
  num_puppies = 6 →
  puppy_price = 350 →
  total_profit = 1600 →
  (num_puppies * puppy_price - total_profit) / num_dogs = 250 := by
  sorry

end NUMINAMATH_CALUDE_show_dog_cost_l2981_298115


namespace NUMINAMATH_CALUDE_triangle_inequality_l2981_298150

/-- 
Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if a^2 - √3ab + b^2 = 1 and c = 1, then 1 < √3a - b < √3.
-/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a^2 - Real.sqrt 3 * a * b + b^2 = 1 ∧  -- Given condition
  c = 1 →  -- Given condition
  1 < Real.sqrt 3 * a - b ∧ Real.sqrt 3 * a - b < Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2981_298150


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2981_298111

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2981_298111


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_two_l2981_298131

def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}

theorem intersection_implies_a_equals_two (a : ℝ) :
  A a ∩ B a = {3} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_two_l2981_298131


namespace NUMINAMATH_CALUDE_max_sum_abc_l2981_298108

def An (a n : ℕ) : ℚ := a * (10^n - 1) / 9
def Bn (b n : ℕ) : ℚ := b * (10^(2*n) - 1) / 9
def Cn (c n : ℕ) : ℚ := c * (10^(2*n) - 1) / 9

theorem max_sum_abc (a b c n : ℕ) :
  (a ∈ Finset.range 10 ∧ a ≠ 0) →
  (b ∈ Finset.range 10 ∧ b ≠ 0) →
  (c ∈ Finset.range 10 ∧ c ≠ 0) →
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^2 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^2) →
  a + b + c ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abc_l2981_298108


namespace NUMINAMATH_CALUDE_intersection_A_B_l2981_298125

def A : Set ℝ := { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2981_298125


namespace NUMINAMATH_CALUDE_ratio_equality_l2981_298134

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc_sq : a^2 + b^2 + c^2 = 1)
  (sum_xyz_sq : x^2 + y^2 + z^2 = 4)
  (sum_prod : a*x + b*y + c*z = 2) :
  (a + b + c) / (x + y + z) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2981_298134


namespace NUMINAMATH_CALUDE_flight_savings_l2981_298106

theorem flight_savings (delta_price united_price : ℝ) 
  (delta_discount united_discount : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  delta_discount = 0.20 →
  united_discount = 0.30 →
  united_price * (1 - united_discount) - delta_price * (1 - delta_discount) = 90 := by
  sorry

end NUMINAMATH_CALUDE_flight_savings_l2981_298106


namespace NUMINAMATH_CALUDE_extremum_implies_b_value_l2981_298172

/-- A function f with a real parameter a -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_b_value (a b : ℝ) :
  (f' a b 1 = 0) →  -- Derivative is zero at x = 1
  (f a b 1 = 10) →  -- Function value is 10 at x = 1
  b = -11 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_b_value_l2981_298172


namespace NUMINAMATH_CALUDE_balls_per_package_l2981_298175

theorem balls_per_package (total_packages : Nat) (total_balls : Nat) 
  (h1 : total_packages = 21) 
  (h2 : total_balls = 399) : 
  (total_balls / total_packages : Nat) = 19 := by
  sorry

end NUMINAMATH_CALUDE_balls_per_package_l2981_298175


namespace NUMINAMATH_CALUDE_sum_of_roots_l2981_298122

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 14*p*x - 15*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 14*r*x - 15*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 3150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2981_298122


namespace NUMINAMATH_CALUDE_rice_division_l2981_298178

/-- 
Given an arithmetic sequence of three terms (a, b, c) where:
- The sum of the terms is 180
- The difference between the first and third term is 36
This theorem proves that the middle term (b) is equal to 60.
-/
theorem rice_division (a b c : ℕ) : 
  a + b + c = 180 →
  a - c = 36 →
  b = 60 := by
  sorry


end NUMINAMATH_CALUDE_rice_division_l2981_298178


namespace NUMINAMATH_CALUDE_workshop_problem_l2981_298163

theorem workshop_problem :
  ∃ (x y : ℕ),
    x ≥ 1 ∧ y ≥ 1 ∧
    6 + 11 * (x - 1) = 7 + 10 * (y - 1) ∧
    100 ≤ 6 + 11 * (x - 1) ∧
    6 + 11 * (x - 1) ≤ 200 ∧
    x = 12 ∧ y = 13 :=
by sorry

end NUMINAMATH_CALUDE_workshop_problem_l2981_298163


namespace NUMINAMATH_CALUDE_definite_integral_problem_l2981_298133

open Real MeasureTheory Interval Set

theorem definite_integral_problem :
  ∫ x in Icc 0 π, (2 * x^2 + 4 * x + 7) * cos (2 * x) = π := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_problem_l2981_298133


namespace NUMINAMATH_CALUDE_road_repair_hours_l2981_298190

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 57)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 19)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people2 * days2 * hours2) = people2 * days2 * (people1 * days1 * hours2)) :
  ∃ hours1 : ℕ, hours1 = 5 ∧ people1 * days1 * hours1 = people2 * days2 * hours2 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_hours_l2981_298190


namespace NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l2981_298169

theorem smallest_gcd_of_multiples (m n : ℕ+) (h : Nat.gcd m n = 15) :
  ∃ (k : ℕ), k ≥ 30 ∧ Nat.gcd (14 * m) (20 * n) = k ∧
  ∀ (j : ℕ), j < 30 → Nat.gcd (14 * m) (20 * n) ≠ j :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l2981_298169


namespace NUMINAMATH_CALUDE_judys_shopping_cost_l2981_298121

/-- Represents Judy's shopping trip cost calculation --/
theorem judys_shopping_cost :
  let carrot_cost : ℕ := 5 * 1
  let milk_cost : ℕ := 3 * 3
  let pineapple_cost : ℕ := 2 * (4 / 2)
  let flour_cost : ℕ := 2 * 5
  let ice_cream_cost : ℕ := 7
  let total_before_coupon := carrot_cost + milk_cost + pineapple_cost + flour_cost + ice_cream_cost
  let coupon_value : ℕ := 5
  let coupon_threshold : ℕ := 25
  total_before_coupon ≥ coupon_threshold →
  total_before_coupon - coupon_value = 30 :=
by sorry

end NUMINAMATH_CALUDE_judys_shopping_cost_l2981_298121


namespace NUMINAMATH_CALUDE_unique_solution_equals_three_l2981_298113

theorem unique_solution_equals_three :
  ∃! (x : ℝ), (x^2 - t*x + 36 = 0) ∧ (x^2 - 8*x + t = 0) ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equals_three_l2981_298113


namespace NUMINAMATH_CALUDE_max_value_theorem_l2981_298100

theorem max_value_theorem (m n : ℝ) 
  (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) 
  (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) : 
  (∀ x y : ℝ, 0 ≤ x - y ∧ x - y ≤ 1 ∧ 2 ≤ x + y ∧ x + y ≤ 4 → m - 2*n ≥ x - 2*y) →
  2019*m + 2020*n = 2019 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2981_298100


namespace NUMINAMATH_CALUDE_largest_s_value_l2981_298110

theorem largest_s_value (r s : ℕ) : 
  r ≥ s → 
  s ≥ 5 → 
  (r - 2) * s * 61 = (s - 2) * r * 60 → 
  s ≤ 121 ∧ ∃ r' : ℕ, r' ≥ 121 ∧ (r' - 2) * 121 * 61 = (121 - 2) * r' * 60 :=
by sorry

end NUMINAMATH_CALUDE_largest_s_value_l2981_298110


namespace NUMINAMATH_CALUDE_exists_selling_price_with_50_percent_profit_l2981_298144

/-- Represents the pricing model for a printer -/
structure PricingModel where
  baseSellPrice : ℝ
  baseProfit : ℝ
  taxRate1 : ℝ
  taxRate2 : ℝ
  taxThreshold1 : ℝ
  taxThreshold2 : ℝ
  discountRate : ℝ
  discountIncrement : ℝ

/-- Calculates the selling price that yields the target profit percentage -/
def findSellingPrice (model : PricingModel) (targetProfit : ℝ) : ℝ :=
  sorry

/-- Theorem: There exists a selling price that yields a 50% profit on the cost of the printer -/
theorem exists_selling_price_with_50_percent_profit (model : PricingModel) :
  ∃ (sellPrice : ℝ), findSellingPrice model 0.5 = sellPrice :=
by
  sorry

end NUMINAMATH_CALUDE_exists_selling_price_with_50_percent_profit_l2981_298144


namespace NUMINAMATH_CALUDE_triangle_angle_side_ratio_l2981_298197

/-- In a triangle ABC, if the ratio of angles A:B:C is 3:1:2, then the ratio of sides a:b:c is 2:1:√3 -/
theorem triangle_angle_side_ratio (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
  (h_angle_ratio : A = 3 * B ∧ C = 2 * B) : 
  ∃ (k : ℝ), a = 2 * k ∧ b = k ∧ c = Real.sqrt 3 * k := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_side_ratio_l2981_298197


namespace NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l2981_298189

/-- Represents a sewage treatment equipment model -/
structure EquipmentModel where
  price : ℕ  -- Price in million yuan
  capacity : ℕ  -- Capacity in tons/month

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelA : ℕ  -- Number of Model A units
  modelB : ℕ  -- Number of Model B units

def modelA : EquipmentModel := { price := 12, capacity := 240 }
def modelB : EquipmentModel := { price := 10, capacity := 200 }

def totalEquipment : ℕ := 10
def budgetConstraint : ℕ := 105
def minTreatmentCapacity : ℕ := 2040

def totalCost (plan : PurchasePlan) : ℕ :=
  plan.modelA * modelA.price + plan.modelB * modelB.price

def totalCapacity (plan : PurchasePlan) : ℕ :=
  plan.modelA * modelA.capacity + plan.modelB * modelB.capacity

def isValidPlan (plan : PurchasePlan) : Prop :=
  plan.modelA + plan.modelB = totalEquipment ∧
  totalCost plan ≤ budgetConstraint ∧
  totalCapacity plan ≥ minTreatmentCapacity

def optimalPlan : PurchasePlan := { modelA := 1, modelB := 9 }

theorem optimal_plan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
by sorry

end NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l2981_298189


namespace NUMINAMATH_CALUDE_roots_power_sum_divisible_l2981_298124

/-- Given two roots of a quadratic equation with a prime coefficient,
    their p-th powers sum to a multiple of p². -/
theorem roots_power_sum_divisible (p : ℕ) (x₁ x₂ : ℝ) 
  (h_prime : Nat.Prime p) 
  (h_p_gt_two : p > 2) 
  (h_roots : x₁^2 - p*x₁ + 1 = 0 ∧ x₂^2 - p*x₂ + 1 = 0) : 
  ∃ (k : ℤ), x₁^p + x₂^p = k * p^2 := by
  sorry

#check roots_power_sum_divisible

end NUMINAMATH_CALUDE_roots_power_sum_divisible_l2981_298124


namespace NUMINAMATH_CALUDE_multiply_24_to_get_2376_l2981_298152

theorem multiply_24_to_get_2376 (x : ℚ) : 24 * x = 2376 → x = 99 := by
  sorry

end NUMINAMATH_CALUDE_multiply_24_to_get_2376_l2981_298152


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2981_298129

theorem quadratic_inequality_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) ↔ m > 13 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2981_298129


namespace NUMINAMATH_CALUDE_garden_dimensions_and_walkway_area_l2981_298138

/-- A rectangular garden with a surrounding walkway. -/
structure Garden where
  breadth : ℝ
  length : ℝ
  walkwayWidth : ℝ

/-- Properties of the garden based on the problem conditions. -/
def GardenProperties (g : Garden) : Prop :=
  g.length = 3 * g.breadth ∧
  2 * (g.length + g.breadth) = 40 ∧
  g.walkwayWidth = 1 ∧
  (g.length + 2 * g.walkwayWidth) * (g.breadth + 2 * g.walkwayWidth) = 120

theorem garden_dimensions_and_walkway_area 
  (g : Garden) 
  (h : GardenProperties g) : 
  g.length = 15 ∧ g.breadth = 5 ∧ 
  ((g.length + 2 * g.walkwayWidth) * (g.breadth + 2 * g.walkwayWidth) - g.length * g.breadth) = 45 :=
by sorry

end NUMINAMATH_CALUDE_garden_dimensions_and_walkway_area_l2981_298138


namespace NUMINAMATH_CALUDE_initial_sand_calculation_l2981_298188

/-- The amount of sand lost during the trip in pounds -/
def sand_lost : ℝ := 2.4

/-- The amount of sand remaining at arrival in pounds -/
def sand_remaining : ℝ := 1.7

/-- The initial amount of sand on the truck in pounds -/
def initial_sand : ℝ := sand_lost + sand_remaining

theorem initial_sand_calculation : initial_sand = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_initial_sand_calculation_l2981_298188


namespace NUMINAMATH_CALUDE_part1_part2_l2981_298198

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * (x - 1)

-- Part 1: Prove that a = 1/2 given the conditions
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) → a = 1/2 := by sorry

-- Part 2: Characterize the solution set for f(x) < 0 when a > 0
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, f a x < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
     (a = 1 ∧ False) ∨
     (a > 1 ∧ 1/a < x ∧ x < 1))) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2981_298198


namespace NUMINAMATH_CALUDE_function_upper_bound_l2981_298164

theorem function_upper_bound
  (a r : ℝ)
  (ha : a > 1)
  (hr : r > 1)
  (f : ℝ → ℝ)
  (hf_pos : ∀ x > 0, f x > 0)
  (hf_cond1 : ∀ x > 0, (f x)^2 ≤ a * x^r * f (x/a))
  (hf_cond2 : ∀ x > 0, x < 1/2^2000 → f x < 2^2000) :
  ∀ x > 0, f x ≤ x^r * a^(1-r) := by
sorry

end NUMINAMATH_CALUDE_function_upper_bound_l2981_298164


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2981_298107

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (5 * x + 9) = 12) ∧ (x = 27) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2981_298107


namespace NUMINAMATH_CALUDE_smallest_number_with_17_proper_factors_l2981_298179

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def number_of_proper_factors (n : ℕ) : ℕ := (number_of_factors n) - 2

theorem smallest_number_with_17_proper_factors :
  ∃ (n : ℕ), n > 0 ∧ 
    number_of_factors n = 19 ∧ 
    number_of_proper_factors n = 17 ∧
    ∀ (m : ℕ), m > 0 → number_of_factors m = 19 → m ≥ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_17_proper_factors_l2981_298179


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2981_298137

/-- Represents a number with an integer part and a repeating decimal part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The given repeating decimal 5.341341341... -/
def givenNumber : RepeatingDecimal :=
  { integerPart := 5, repeatingPart := 341 }

/-- Theorem stating that 5.341341341... equals 5336/999 -/
theorem repeating_decimal_equals_fraction :
  toRational givenNumber = 5336 / 999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2981_298137


namespace NUMINAMATH_CALUDE_rectangle_division_l2981_298109

theorem rectangle_division (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  let x := b / 3
  let y := a / 3
  (x * a) / ((b - x) * a) = 1 / 2 ∧
  (2 * a + 2 * x) / (2 * a + 2 * (b - x)) = 3 / 5 →
  (y * b) / ((a - y) * b) = 1 / 2 →
  (2 * y + 2 * b) / (2 * (a - y) + 2 * b) = 20 / 19 :=
by sorry


end NUMINAMATH_CALUDE_rectangle_division_l2981_298109


namespace NUMINAMATH_CALUDE_hidden_piece_area_l2981_298167

/-- Represents the surface areas of the 7 visible pieces of the wooden block -/
def visible_areas : List ℝ := [148, 46, 72, 28, 88, 126, 58]

/-- The total number of pieces the wooden block is cut into -/
def total_pieces : ℕ := 8

/-- Theorem: Given a wooden block cut into 8 pieces, where the surface areas of 7 pieces are known,
    and the sum of these areas is 566, the surface area of the 8th piece is 22. -/
theorem hidden_piece_area (h1 : visible_areas.length = total_pieces - 1)
                          (h2 : visible_areas.sum = 566) : 
  ∃ (hidden_area : ℝ), hidden_area = 22 ∧ 
    visible_areas.sum + hidden_area = (visible_areas.sum + hidden_area) / 2 * 2 := by
  sorry

end NUMINAMATH_CALUDE_hidden_piece_area_l2981_298167


namespace NUMINAMATH_CALUDE_second_grade_sample_l2981_298127

/-- Given a total sample size and ratios for three grades, 
    calculate the number of students to be drawn from a specific grade. -/
def stratified_sample (total_sample : ℕ) (ratio1 ratio2 ratio3 : ℕ) (target_ratio : ℕ) : ℕ :=
  (target_ratio * total_sample) / (ratio1 + ratio2 + ratio3)

/-- Theorem: Given a total sample of 50 and ratios 3:3:4, 
    the number of students from the grade with ratio 3 is 15. -/
theorem second_grade_sample :
  stratified_sample 50 3 3 4 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_sample_l2981_298127


namespace NUMINAMATH_CALUDE_tenth_group_draw_l2981_298149

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  group_size : Nat
  first_draw : Nat

/-- Calculates the number drawn from a specific group in systematic sampling -/
def draw_from_group (s : SystematicSampling) (group : Nat) : Nat :=
  s.first_draw + s.group_size * (group - 1)

theorem tenth_group_draw (s : SystematicSampling) 
  (h1 : s.total_students = 1000)
  (h2 : s.sample_size = 100)
  (h3 : s.group_size = 10)
  (h4 : s.first_draw = 6) :
  draw_from_group s 10 = 96 := by
  sorry

end NUMINAMATH_CALUDE_tenth_group_draw_l2981_298149


namespace NUMINAMATH_CALUDE_fertilizer_on_half_field_l2981_298156

/-- Theorem: Amount of fertilizer on half a football field -/
theorem fertilizer_on_half_field (total_area : ℝ) (total_fertilizer : ℝ) 
  (h1 : total_area = 7200)
  (h2 : total_fertilizer = 1200) :
  (total_fertilizer / total_area) * (total_area / 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_fertilizer_on_half_field_l2981_298156


namespace NUMINAMATH_CALUDE_z_equation_l2981_298105

theorem z_equation (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y - 2*x*y ≠ 0) :
  (1/x + 1/y = 2 + 1/z) → z = (x*y)/(x + y - 2*x*y) := by
  sorry

end NUMINAMATH_CALUDE_z_equation_l2981_298105


namespace NUMINAMATH_CALUDE_inequalities_proof_l2981_298130

theorem inequalities_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2981_298130


namespace NUMINAMATH_CALUDE_bankers_discount_equation_l2981_298199

/-- The banker's discount (BD) for a certain sum of money. -/
def BD : ℚ := 80

/-- The true discount (TD) for the same sum of money. -/
def TD : ℚ := 70

/-- The present value (PV) of the sum due. -/
def PV : ℚ := 490

/-- Theorem stating that the given BD, TD, and PV satisfy the banker's discount equation. -/
theorem bankers_discount_equation : BD = TD + TD^2 / PV := by sorry

end NUMINAMATH_CALUDE_bankers_discount_equation_l2981_298199


namespace NUMINAMATH_CALUDE_max_distance_between_generatrices_l2981_298196

/-- The maximum distance between two generatrices of two cones with a common base -/
theorem max_distance_between_generatrices (r h H : ℝ) (h_pos : 0 < h) (H_pos : 0 < H) (h_le_H : h ≤ H) :
  ∃ (d : ℝ), d = (h + H) * r / Real.sqrt (r^2 + H^2) ∧
  ∀ (d' : ℝ), d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_generatrices_l2981_298196


namespace NUMINAMATH_CALUDE_number_sum_proof_l2981_298186

theorem number_sum_proof : ∃ x : ℤ, x + 15 = 96 ∧ x = 81 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_proof_l2981_298186


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2981_298162

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallelLine : Line → Line → Prop)

-- Define the intersection of two planes resulting in a line
variable (planeIntersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection
  (l m : Line) (α β : Plane)
  (h1 : planeIntersection α β = l)
  (h2 : parallelLinePlane m α)
  (h3 : parallelLinePlane m β) :
  parallelLine m l :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2981_298162


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2981_298170

/-- A rhombus with diagonals in the ratio 3:4 and sum 56 has perimeter 80 -/
theorem rhombus_perimeter (d₁ d₂ s : ℝ) : 
  d₁ > 0 → d₂ > 0 → s > 0 →
  d₁ / d₂ = 3 / 4 → 
  d₁ + d₂ = 56 → 
  s^2 = (d₁/2)^2 + (d₂/2)^2 → 
  4 * s = 80 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2981_298170


namespace NUMINAMATH_CALUDE_a0_value_l2981_298117

theorem a0_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₀ = 32 := by
sorry

end NUMINAMATH_CALUDE_a0_value_l2981_298117


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2981_298104

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_10 : a 10 = 30)
  (h_20 : a 20 = 50) :
  ∃ b c : ℝ, ∀ n : ℕ, a n = b * n + c ∧ b = 2 ∧ c = 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2981_298104


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l2981_298146

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l2981_298146


namespace NUMINAMATH_CALUDE_same_side_of_line_l2981_298140

/-- 
Given a line 2x - y + 1 = 0 and two points (1, 2) and (1, 0),
prove that these points are on the same side of the line.
-/
theorem same_side_of_line (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 1 ∧ y₁ = 2 ∧ x₂ = 1 ∧ y₂ = 0 →
  (2 * x₁ - y₁ + 1 > 0) ∧ (2 * x₂ - y₂ + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_same_side_of_line_l2981_298140


namespace NUMINAMATH_CALUDE_inheritance_distribution_l2981_298193

structure Relative where
  name : String
  amount : ℕ

structure Couple where
  husband : Relative
  wife : Relative

def total_inheritance : ℕ := 1000
def wives_total : ℕ := 396

theorem inheritance_distribution (john henry tom : Relative) (katherine jane mary : Relative) :
  john.name = "John Smith" →
  henry.name = "Henry Snooks" →
  tom.name = "Tom Crow" →
  katherine.name = "Katherine" →
  jane.name = "Jane" →
  mary.name = "Mary" →
  jane.amount = katherine.amount + 10 →
  mary.amount = jane.amount + 10 →
  katherine.amount + jane.amount + mary.amount = wives_total →
  john.amount = katherine.amount →
  henry.amount = (3 * jane.amount) / 2 →
  tom.amount = 2 * mary.amount →
  john.amount + henry.amount + tom.amount + katherine.amount + jane.amount + mary.amount = total_inheritance →
  ∃ (c1 c2 c3 : Couple),
    c1.husband = john ∧ c1.wife = katherine ∧
    c2.husband = henry ∧ c2.wife = jane ∧
    c3.husband = tom ∧ c3.wife = mary :=
by
  sorry

end NUMINAMATH_CALUDE_inheritance_distribution_l2981_298193


namespace NUMINAMATH_CALUDE_least_marbles_thirty_two_satisfies_george_marbles_l2981_298102

theorem least_marbles (n : ℕ) : 
  (n % 7 = 1 ∧ n % 4 = 2 ∧ n % 6 = 3) → n ≥ 32 :=
by sorry

theorem thirty_two_satisfies : 
  32 % 7 = 1 ∧ 32 % 4 = 2 ∧ 32 % 6 = 3 :=
by sorry

theorem george_marbles : 
  ∃ (n : ℕ), n % 7 = 1 ∧ n % 4 = 2 ∧ n % 6 = 3 ∧ 
  ∀ (m : ℕ), (m % 7 = 1 ∧ m % 4 = 2 ∧ m % 6 = 3) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_thirty_two_satisfies_george_marbles_l2981_298102


namespace NUMINAMATH_CALUDE_candy_remaining_l2981_298173

theorem candy_remaining (initial : ℕ) (talitha solomon maya : ℕ) 
  (h1 : initial = 572)
  (h2 : talitha = 183)
  (h3 : solomon = 238)
  (h4 : maya = 127) :
  initial - (talitha + solomon + maya) = 24 :=
by sorry

end NUMINAMATH_CALUDE_candy_remaining_l2981_298173


namespace NUMINAMATH_CALUDE_perfume_price_change_l2981_298153

def original_price : ℝ := 1200
def increase_rate : ℝ := 0.10
def decrease_rate : ℝ := 0.15

theorem perfume_price_change :
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - decrease_rate)
  original_price - final_price = 78 := by
sorry

end NUMINAMATH_CALUDE_perfume_price_change_l2981_298153


namespace NUMINAMATH_CALUDE_expected_successful_trials_value_l2981_298116

/-- A trial is successful if at least one of two dice shows a 4 or a 5 -/
def is_successful_trial (dice1 dice2 : Nat) : Bool :=
  dice1 = 4 ∨ dice1 = 5 ∨ dice2 = 4 ∨ dice2 = 5

/-- The probability of a successful trial -/
def prob_success : ℚ := 5 / 9

/-- The number of trials -/
def num_trials : ℕ := 10

/-- The expected number of successful trials -/
def expected_successful_trials : ℚ := num_trials * prob_success

theorem expected_successful_trials_value :
  expected_successful_trials = 50 / 9 := by sorry

end NUMINAMATH_CALUDE_expected_successful_trials_value_l2981_298116


namespace NUMINAMATH_CALUDE_simplify_x_expression_simplify_a_expression_l2981_298171

-- First equation
theorem simplify_x_expression (x : ℝ) : 3 * x^4 * x^2 + (2 * x^2)^3 = 11 * x^6 := by
  sorry

-- Second equation
theorem simplify_a_expression (a : ℝ) : 3 * a * (9 * a + 3) - 4 * a * (2 * a - 1) = 19 * a^2 + 13 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_x_expression_simplify_a_expression_l2981_298171


namespace NUMINAMATH_CALUDE_prime_cube_difference_equation_l2981_298136

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem prime_cube_difference_equation :
  ∀ p q r : ℕ,
  is_prime p ∧ is_prime q ∧ is_prime r →
  p^3 - q^3 = 5*r →
  p = 7 ∧ q = 2 ∧ r = 67 :=
sorry

end NUMINAMATH_CALUDE_prime_cube_difference_equation_l2981_298136


namespace NUMINAMATH_CALUDE_joels_board_games_l2981_298132

theorem joels_board_games (stuffed_animals action_figures puzzles total_toys joels_toys : ℕ)
  (h1 : stuffed_animals = 18)
  (h2 : action_figures = 42)
  (h3 : puzzles = 13)
  (h4 : total_toys = 108)
  (h5 : joels_toys = 22) :
  ∃ (board_games sisters_toys : ℕ),
    sisters_toys * 3 = joels_toys ∧
    stuffed_animals + action_figures + board_games + puzzles + sisters_toys * 3 = total_toys ∧
    board_games = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_joels_board_games_l2981_298132


namespace NUMINAMATH_CALUDE_cube_structure_surface_area_total_surface_area_is_1266_l2981_298159

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.sideLength ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℕ := 6 * c.sideLength ^ 2

/-- Represents the structure formed by the cubes -/
structure CubeStructure where
  cubes : List Cube
  stackedCubes : List Cube
  adjacentCube : Cube
  topCube : Cube

/-- Theorem stating the total surface area of the cube structure -/
theorem cube_structure_surface_area (cs : CubeStructure) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem total_surface_area_is_1266 (cs : CubeStructure) 
  (h1 : cs.cubes.length = 8)
  (h2 : (cs.cubes.map Cube.volume) = [1, 8, 27, 64, 125, 216, 512, 729])
  (h3 : cs.stackedCubes.length = 6)
  (h4 : cs.stackedCubes = (cs.cubes.take 6).reverse)
  (h5 : cs.adjacentCube = cs.cubes[6])
  (h6 : cs.adjacentCube.sideLength = 6)
  (h7 : cs.stackedCubes[4].sideLength = 5)
  (h8 : cs.topCube = cs.cubes[7])
  (h9 : cs.topCube.sideLength = 8) :
  cube_structure_surface_area cs = 1266 :=
sorry

end NUMINAMATH_CALUDE_cube_structure_surface_area_total_surface_area_is_1266_l2981_298159


namespace NUMINAMATH_CALUDE_ellipse_properties_l2981_298120

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 2

-- Define the eccentricity relation
def eccentricity_relation (a b : ℝ) : Prop :=
  (2 / a)^2 = 1 / 2

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ 
    ellipse_C x₁ y₁ (2 * Real.sqrt 2) 2 ∧
    ellipse_C x₂ y₂ (2 * Real.sqrt 2) 2 ∧
    line_l x₁ y₁ k ∧ line_l x₂ y₂ k

-- Define the focus inside circle condition
def focus_inside_circle (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂, 
    ellipse_C x₁ y₁ (2 * Real.sqrt 2) 2 →
    ellipse_C x₂ y₂ (2 * Real.sqrt 2) 2 →
    line_l x₁ y₁ k → line_l x₂ y₂ k →
    (x₁ - 2) * (x₂ - 2) + y₁ * y₂ < 0

theorem ellipse_properties :
  ∀ a b : ℝ,
    ellipse_C 0 0 a b →
    focal_distance 2 →
    eccentricity_relation a b →
    (a = 2 * Real.sqrt 2 ∧ b = 2) ∧
    (∀ k : ℝ, intersects_at_two_points k →
      (focus_inside_circle k ↔ k < 1/8)) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2981_298120


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l2981_298181

theorem opposite_of_negative_six : -((-6) : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l2981_298181


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2981_298123

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The point through which the line passes -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the line: 2x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_equation :
  (∀ x y, line_equation x y ↔ 
    (y - point.2 = f' point.1 * (x - point.1) ∧
     f point.1 = point.2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2981_298123
