import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_l680_68082

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 0

-- State the theorem
theorem domain_f_2x (h : ∀ x ∈ domain_f_x_plus_1, f (x + 1) = f (x + 1)) :
  {x : ℝ | f (2 * x) = f (2 * x)} = Set.Ico 0 (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_l680_68082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_four_seventeenths_l680_68014

/-- Represents the dimensions of a rectangular yard with two isosceles right triangular flower beds. -/
structure YardDimensions where
  trapezoid_short_side : ℚ
  trapezoid_long_side : ℚ

/-- Calculates the fraction of the yard occupied by the flower beds. -/
def flower_bed_fraction (d : YardDimensions) : ℚ :=
  let triangle_leg := (d.trapezoid_long_side - d.trapezoid_short_side) / 2
  let flower_bed_area := 2 * (triangle_leg * triangle_leg / 2)
  let yard_area := d.trapezoid_long_side * triangle_leg
  flower_bed_area / yard_area

/-- Theorem stating that the fraction of the yard occupied by the flower beds is 4/17. -/
theorem flower_bed_fraction_is_four_seventeenths (d : YardDimensions)
  (h1 : d.trapezoid_short_side = 18)
  (h2 : d.trapezoid_long_side = 34) :
  flower_bed_fraction d = 4 / 17 := by
  sorry

#eval flower_bed_fraction { trapezoid_short_side := 18, trapezoid_long_side := 34 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_four_seventeenths_l680_68014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_path_l680_68071

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflect a point over the y-axis -/
def reflectOverY (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Reflect a point over the x-axis -/
def reflectOverX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The laser beam path theorem -/
theorem laser_beam_path : 
  let start : Point := { x := 4, y := 6 }
  let finish : Point := { x := 8, y := 6 }
  let reflected_finish := reflectOverY (reflectOverX (reflectOverY finish))
  distance start reflected_finish = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_path_l680_68071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tuesdays_fridays_in_30_day_month_l680_68047

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq, Inhabited

/-- A function that determines if a given day can be the first day of a 30-day month
    with equal number of Tuesdays and Fridays -/
def canBeFirstDay (d : DayOfWeek) : Bool :=
  match d with
  | DayOfWeek.Sunday => true
  | DayOfWeek.Monday => false
  | DayOfWeek.Tuesday => false
  | DayOfWeek.Wednesday => true
  | DayOfWeek.Thursday => false
  | DayOfWeek.Friday => false
  | DayOfWeek.Saturday => true

/-- List of all days of the week -/
def allDays : List DayOfWeek :=
  [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
   DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]

/-- The main theorem stating that exactly 3 days of the week can be
    the first day of a 30-day month with equal Tuesdays and Fridays -/
theorem equal_tuesdays_fridays_in_30_day_month :
  (allDays.filter canBeFirstDay).length = 3 := by
  sorry

#eval allDays.filter canBeFirstDay

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tuesdays_fridays_in_30_day_month_l680_68047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_zero_minimum_at_one_l680_68009

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (a * x^2 - (3*a + 1) * x + 3*a + 2) * Real.exp x

-- Part I
theorem tangent_slope_zero (a : ℝ) :
  (deriv (f a)) 2 = 0 → a = 1/2 := by sorry

-- Part II
theorem minimum_at_one (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 1) → a > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_zero_minimum_at_one_l680_68009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l680_68020

-- Define the power function
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_value (α : ℝ) :
  f α 8 = 2 → f α (-1/8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l680_68020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_24_l680_68050

def a (n : ℕ) : ℤ := 2 * n - 49

def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

theorem min_sum_at_24 :
  ∀ k : ℕ, k > 0 → S 24 ≤ S k :=
by
  sorry

#eval S 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_24_l680_68050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_six_circles_l680_68059

/-- The area of the shaded region formed by the intersection of six circles --/
theorem shaded_area_six_circles (r : ℝ) (h : r = 5) :
  let sector_area := π * r^2 / 4
  let triangle_area := Real.sqrt 3 * r^2 / 4
  let checkered_area := sector_area - triangle_area / 3
  6 * 2 * checkered_area = 75 * π - 25 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_six_circles_l680_68059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_theorem_l680_68080

/-- Represents a domino tile with two numbers -/
structure Domino :=
  (first : Nat)
  (second : Nat)

/-- Represents a position in the rectangular arrangement -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the arrangement of dominoes -/
def Arrangement := Position → Option Domino

/-- Checks if a position is on the border of the rectangle -/
def is_border (p : Position) (rows cols : Nat) : Prop :=
  p.row = 0 ∨ p.row = rows - 1 ∨ p.col = 0 ∨ p.col = cols - 1

/-- Checks if a number is on a domino -/
def number_on_domino (d : Domino) (n : Nat) : Prop :=
  d.first = n ∨ d.second = n

/-- The main theorem -/
theorem domino_arrangement_theorem
  (arr : Arrangement)
  (rows cols : Nat)
  (h_complete : ∀ p, p.row < rows → p.col < cols → (arr p).isSome)
  (h_squares : ∀ n, ∃ p, p.row < rows ∧ p.col < cols ∧
    (arr p).isSome ∧ 
    (∀ d, arr p = some d → number_on_domino d n) ∧
    (arr (Position.mk (p.row + 1) p.col)).isSome ∧
    (∀ d, arr (Position.mk (p.row + 1) p.col) = some d → number_on_domino d n) ∧
    (arr (Position.mk p.row (p.col + 1))).isSome ∧
    (∀ d, arr (Position.mk p.row (p.col + 1)) = some d → number_on_domino d n) ∧
    (arr (Position.mk (p.row + 1) (p.col + 1))).isSome ∧
    (∀ d, arr (Position.mk (p.row + 1) (p.col + 1)) = some d → number_on_domino d n))
  (h_border : ∀ p, is_border p rows cols →
    (arr p).isSome → ∀ d, arr p = some d → ¬number_on_domino d 0) :
  ∀ p, (arr p).isSome → 
    (∀ d, arr p = some d → number_on_domino d 0) →
    ¬is_border p rows cols :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_theorem_l680_68080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l680_68031

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x - y = 2

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) : ℝ := 
  |x - y - 2| / Real.sqrt 2

/-- Maximum distance from circle to line -/
theorem max_distance_circle_to_line :
  ∃ (max_dist : ℝ), max_dist = 1 + Real.sqrt 2 ∧
  ∀ (x y : ℝ), circle_eq x y →
    distance_point_to_line x y ≤ max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l680_68031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_symmetry_l680_68023

-- Define the line
def line (a : ℝ) (x : ℝ) : ℝ := a * x + 1

-- Define the circle
def circle_eq (a b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + a*x + b*y - 4 = 0

-- Define the symmetry condition
def symmetric_wrt_y_eq_x (M N : ℝ × ℝ) : Prop :=
  N.1 = M.2 ∧ N.2 = M.1

-- Theorem statement
theorem line_circle_intersection_symmetry (a b : ℝ) (M N : ℝ × ℝ) :
  circle_eq a b M.1 M.2 →
  circle_eq a b N.1 N.2 →
  line a M.1 = M.2 →
  line a N.1 = N.2 →
  symmetric_wrt_y_eq_x M N →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_symmetry_l680_68023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_campaign_donation_fraction_l680_68068

theorem campaign_donation_fraction (max_donation : ℚ) 
  (num_max_donors : ℕ) (total_raised : ℚ) (fraction_of_max : ℚ) :
  max_donation = 1200 →
  num_max_donors = 500 →
  total_raised = 3750000 →
  (max_donation * num_max_donors + 
   3 * num_max_donors * max_donation * fraction_of_max = 
   (2 : ℚ) / 5 * total_raised) →
  fraction_of_max = 1 / 2 := by
  sorry

#check campaign_donation_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_campaign_donation_fraction_l680_68068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l680_68087

noncomputable section

/-- The volume of a solid obtained by rotating a region around the y-axis --/
def rotationVolume (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.pi * ∫ y in a..b, (f y)^2 - (g y)^2

/-- The first bounding function --/
def f (y : ℝ) : ℝ := y^2 + 2

/-- The second bounding function --/
def g (y : ℝ) : ℝ := y^(1/3)

/-- The lower bound of integration --/
def a : ℝ := 0

/-- The upper bound of integration --/
def b : ℝ := 1

theorem volume_of_rotation :
  rotationVolume f g a b = 24 * Real.pi / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l680_68087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l680_68066

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧
  (∀ ε > 0, ∀ x : ℝ, f (x + Real.pi/2) = f x) ∧
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l680_68066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l680_68092

-- Define the function f(x) = 2^x + 3x - 7
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + 3*x - 7

-- State the theorem
theorem root_in_interval :
  (f 0 < 0) → (f 2 > 0) → ∃ x, x ∈ Set.Ioo 0 2 ∧ f x = 0 :=
by
  sorry

#check root_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l680_68092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_properties_remaining_square_side_length_is_correct_l680_68012

/-- Represents the process of dividing a square into 9 parts and removing the middle square. -/
noncomputable def squareDivisionProcess (n : ℕ) : ℕ × ℝ :=
  (8^n, 1 - (8/9)^n)

/-- Theorem stating the properties of the square division process after n iterations. -/
theorem square_division_properties (n : ℕ) :
  let (remaining_squares, removed_area) := squareDivisionProcess n
  (remaining_squares = 8^n) ∧ 
  (removed_area = 1 - (8/9)^n) :=
by sorry

/-- The side length of the remaining squares after n iterations. -/
noncomputable def remaining_square_side_length (n : ℕ) : ℝ := (1/3)^n

/-- Theorem stating that the side length of remaining squares is 1/3^n. -/
theorem remaining_square_side_length_is_correct (n : ℕ) :
  remaining_square_side_length n = (1/3)^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_properties_remaining_square_side_length_is_correct_l680_68012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_correct_grid_black_probability_l680_68015

/-- Represents a square on the 4x4 grid -/
structure Square where
  x : Fin 4
  y : Fin 4

/-- The type of color a square can have -/
inductive Color
  | White
  | Black

/-- Represents the 4x4 grid -/
def Grid := Square → Color

/-- Probability of a square being black initially -/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates a square 180 degrees -/
def rotate (s : Square) : Square :=
  ⟨3 - s.x, 3 - s.y⟩

/-- Checks if a square is in the center of the grid -/
def is_center (s : Square) : Prop :=
  (s.x = 1 ∨ s.x = 2) ∧ (s.y = 1 ∨ s.y = 2)

/-- The probability that the grid is entirely black after the operation -/
noncomputable def prob_all_black : ℝ := (1 : ℝ) / 1048576

theorem prob_all_black_correct :
  prob_all_black = (initial_black_prob ^ 4) * ((initial_black_prob ^ 2) ^ 8) := by
  sorry

/-- Main theorem: The probability of the grid being entirely black after the operation is 1/1048576 -/
theorem grid_black_probability :
  prob_all_black = (1 : ℝ) / 1048576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_correct_grid_black_probability_l680_68015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_zero_l680_68090

-- Define the function f and its derivative
def f : ℝ → ℝ := λ x => x^4 - 2*x^2 - 5
def f' : ℝ → ℝ := λ x => 4*x^3 - 4*x

-- State the conditions
axiom f_at_zero : f 0 = -5
axiom f'_def : ∀ x, f' x = 4 * x^3 - 4 * x

-- State the theorem
theorem f_max_at_zero :
  (∀ x, f x ≤ f 0) ∧ (∃ x, f x = f 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_zero_l680_68090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l680_68052

/-- Given b > 0, defines the curve C: x² + 3y² = 3b² -/
def C (b : ℝ) (x y : ℝ) : Prop := x^2 + 3*y^2 = 3*b^2

/-- The line l: y = x - √2b -/
def L (b : ℝ) (x y : ℝ) : Prop := y = x - Real.sqrt 2 * b

theorem constant_sum_of_squares 
  (b : ℝ) 
  (hb : b > 0) 
  (A B P : ℝ × ℝ) 
  (hA : C b A.1 A.2 ∧ L b A.1 A.2) 
  (hB : C b B.1 B.2 ∧ L b B.1 B.2) 
  (hP : C b P.1 P.2) 
  (lambda mu : ℝ) 
  (h_decomp : P.1 = lambda * A.1 + mu * B.1 ∧ P.2 = lambda * A.2 + mu * B.2) :
  lambda^2 + mu^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l680_68052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_pqd_is_13_l680_68057

-- Define the piecewise function f
noncomputable def f (p q d : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then p * x + 4
  else if x = 0 then p * q
  else q * x + d

-- State the theorem
theorem sum_of_pqd_is_13 (p q d : ℕ) :
  f p q d 3 = 7 ∧ f p q d 0 = 6 ∧ f p q d (-3) = -12 → p + q + d = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_pqd_is_13_l680_68057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_connection_point_l680_68078

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Finds the Torricelli point of a triangle -/
noncomputable def torricelliPoint (a b c : Point) : Point :=
  sorry

/-- Theorem: Optimal connection point for three wells -/
theorem optimal_connection_point (a b c : Point) :
  let p := if (angle a b c ≥ 2 * Real.pi / 3) then a
            else if (angle b c a ≥ 2 * Real.pi / 3) then b
            else if (angle c a b ≥ 2 * Real.pi / 3) then c
            else torricelliPoint a b c
  ∀ q : Point,
    distance p a + distance p b + distance p c ≤
    distance q a + distance q b + distance q c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_connection_point_l680_68078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_sector_volume_formula_l680_68006

/-- The volume of a spherical sector -/
noncomputable def spherical_sector_volume (R h : ℝ) : ℝ := (2 * Real.pi * R^2 * h) / 3

/-- Theorem: The volume of a spherical sector with radius R and height h is (2πR²h)/3 -/
theorem spherical_sector_volume_formula (R h : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) :
  spherical_sector_volume R h = (2 * Real.pi * R^2 * h) / 3 := by
  -- The proof is omitted for now
  sorry

#check spherical_sector_volume_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_sector_volume_formula_l680_68006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l680_68018

noncomputable def complex_z : ℂ := (1 - Complex.I) / (1 + Complex.I)

theorem abs_z_equals_one : Complex.abs complex_z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l680_68018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_total_time_is_20_l680_68084

/-- The time it takes for x and y to complete the work together -/
def total_time : ℝ := 10

/-- The time y works on the project -/
def y_work_time : ℝ := 6

/-- The time x works alone before y joins -/
def x_solo_time : ℝ := 4

/-- The time it takes y to complete the work alone -/
def y_solo_time : ℝ := 12

/-- The time it takes x to complete the work alone -/
noncomputable def x_total_time : ℝ := 20

/-- The portion of work completed per day by x -/
noncomputable def x_daily_work : ℝ := 1 / x_total_time

/-- The portion of work completed per day by y -/
noncomputable def y_daily_work : ℝ := 1 / y_solo_time

/-- The total amount of work to be done -/
def total_work : ℝ := 1

theorem x_total_time_is_20 :
  x_solo_time * x_daily_work + 
  y_work_time * (x_daily_work + y_daily_work) = total_work →
  x_total_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_total_time_is_20_l680_68084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l680_68027

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalAreaPyramid (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge^2 - (baseEdge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid 
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units -/
theorem pyramid_area_theorem : 
  totalAreaPyramid 8 7 = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l680_68027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l680_68021

noncomputable def a_sequence (n : ℕ+) : ℝ := 3 * (5/6)^(n.val - 1) + 1

theorem sequence_formula (n : ℕ+) :
  let S : ℕ+ → ℝ := λ k ↦ k - 5 * a_sequence k + 23
  S n = n - 5 * a_sequence n + 23 →
  a_sequence n = 3 * (5/6)^(n.val - 1) + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l680_68021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l680_68025

noncomputable def arithmetic_sequence_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem arithmetic_sequence_terms : 
  ∃ (n : ℕ), arithmetic_sequence_sum 4 3 n = 650 ∧ n = 20 := by
  use 20
  constructor
  · simp [arithmetic_sequence_sum]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l680_68025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_for_given_tan_l680_68042

theorem sin_value_for_given_tan (α : Real) : 
  α ∈ Set.Ioo π (3 * π / 2) → 
  Real.tan α = 4 / 3 → 
  Real.sin α = -4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_for_given_tan_l680_68042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hounds_score_l680_68048

/-- 
Given:
- The total points scored in a basketball game is 82
- The Foxes won by a margin of 18 points
Prove that the Hounds scored 32 points
-/
theorem hounds_score (total_points foxes_score hounds_score margin : ℕ) : 
  total_points = 82 →
  margin = 18 →
  foxes_score = hounds_score + margin →
  total_points = foxes_score + hounds_score →
  hounds_score = 32 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check hounds_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hounds_score_l680_68048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_wall_l680_68067

/-- Calculates the number of bricks needed to build a wall -/
def bricks_needed (wall_length wall_height wall_thickness : ℚ) 
                  (brick_length brick_width brick_height : ℚ) : ℕ :=
  let wall_volume := wall_length * wall_height * wall_thickness
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume / brick_volume).ceil.toNat

/-- Theorem stating the number of bricks needed for the given wall and brick dimensions -/
theorem bricks_for_wall : 
  bricks_needed 800 100 5 25 11 6 = 243 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_wall_l680_68067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carter_cards_l680_68054

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carter_cards (marcus_cards carter_cards : ℕ) (difference : ℕ) 
  (h1 : marcus_cards = 210)
  (h2 : marcus_cards = difference + carter_cards)
  (h3 : difference = 58) : 
  carter_cards = 152 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carter_cards_l680_68054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equation_l680_68046

theorem complex_modulus_equation (a : ℝ) :
  let z : ℂ := 2 + a * Complex.I
  Complex.abs ((1 - Complex.I) * z) = 4 ↔ a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equation_l680_68046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_ten_in_three_integers_l680_68091

theorem multiple_of_ten_in_three_integers (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  ∃ a b : ℤ, (a ∈ ({x, y, z} : Set ℤ) ∧ b ∈ ({x, y, z} : Set ℤ) ∧ a ≠ b) ∧ 10 ∣ (a^5 * b^3 - a^3 * b^5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_ten_in_three_integers_l680_68091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_existence_l680_68003

def is_arithmetic_sequence (seq : List ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i + 1 < seq.length → seq[i + 1]! - seq[i]! = d

def is_geometric_sequence (seq : List ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : ℕ, i + 1 < seq.length → seq[i + 1]! / seq[i]! = r

theorem sequences_existence : ∃ (ap : List ℝ) (gp : List ℝ),
  ap.length = 8 ∧
  gp.length = 4 ∧
  is_arithmetic_sequence ap ∧
  is_geometric_sequence gp ∧
  ap[0]! = 1 ∧
  gp[0]! = 1 ∧
  ap[7]! = gp[3]! ∧
  (gp.sum = ap[7]! + 21) ∧
  ((ap = [1, 10, 19, 28, 37, 46, 55, 64] ∧ gp = [1, 4, 16, 64]) ∨
   (ap = [1, -17, -35, -53, -71, -89, -107, -125] ∧ gp = [1, -5, 25, -125])) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_existence_l680_68003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l680_68062

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := y = x - Real.sqrt 3

-- Define the right focus of the ellipse
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l680_68062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_properties_l680_68036

/-- A line passing through two points with the same x-coordinate has an undefined slope and a 90° angle of inclination. -/
theorem vertical_line_properties (m n : ℝ) (hn : n ≠ 0) :
  let C : ℝ × ℝ := (m, n)
  let D : ℝ × ℝ := (m, -n)
  (∀ k : ℝ, (D.2 - C.2) ≠ k * (D.1 - C.1)) ∧ 
  (Real.pi / 2 = Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_properties_l680_68036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expressions_equality_l680_68058

-- Define constants for common trigonometric values
noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def sin30 : ℝ := 1 / 2
noncomputable def tan30 : ℝ := Real.sqrt 3 / 3
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos60 : ℝ := 1 / 2
noncomputable def sin60 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3

theorem trig_expressions_equality :
  (2 * cos45 - (3/2) * tan30 * cos30 + sin60 ^ 2 = Real.sqrt 2) ∧
  ((sin30)⁻¹ * (sin60 - cos45) - Real.sqrt ((1 - tan60) ^ 2) = 1 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expressions_equality_l680_68058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_energy_minimized_l680_68049

/-- Energy function for salmon swimming upstream -/
noncomputable def energy (k : ℝ) (v : ℝ) : ℝ := 100 * k * v^3 / (v - 3)

/-- The speed at which energy consumption is minimized -/
def optimal_speed : ℝ := 4.5

theorem salmon_energy_minimized (k : ℝ) (h_k : k > 0) :
  ∀ v > 3, energy k v ≥ energy k optimal_speed := by
  sorry

#check salmon_energy_minimized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_energy_minimized_l680_68049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_functions_l680_68000

noncomputable section

-- Define the interval (0, +∞)
def OpenPositiveReals := {x : ℝ | x > 0}

-- Define the functions
def f_A (x : ℝ) : ℝ := -x + 3
def f_B (x : ℝ) : ℝ := -abs (x - 1)
def f_C (x : ℝ) : ℝ := (x + 1)^2
def f_D (x : ℝ) : ℝ := 1 / x

-- Define monotonically increasing function
def MonotonicallyIncreasing (f : ℝ → ℝ) (S : Set ℝ) :=
  ∀ (x y : ℝ), x ∈ S → y ∈ S → x < y → f x < f y

-- State the theorem
theorem monotonically_increasing_functions :
  MonotonicallyIncreasing f_C OpenPositiveReals ∧
  ¬MonotonicallyIncreasing f_A OpenPositiveReals ∧
  ¬MonotonicallyIncreasing f_B OpenPositiveReals ∧
  ¬MonotonicallyIncreasing f_D OpenPositiveReals :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_functions_l680_68000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l680_68010

-- Define the ellipse
noncomputable def ellipse (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem ellipse_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = 1/2) 
  (h4 : ∃ c, distance (-c) 0 2 1 = Real.sqrt 10) :
  (∀ x y, ellipse x y a b ↔ ellipse x y 2 (Real.sqrt 3)) ∧
  (∀ k m, 
    let l := λ x => k * x + m
    ∃ x1 y1 x2 y2, 
      ellipse x1 y1 2 (Real.sqrt 3) ∧ 
      ellipse x2 y2 2 (Real.sqrt 3) ∧
      y1 = l x1 ∧ 
      y2 = l x2 ∧ 
      x1 ≠ 2 ∧ 
      x2 ≠ 2 ∧ 
      x1 ≠ -2 ∧ 
      x2 ≠ -2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = (2 - x1)^2 + y1^2 + (2 - x2)^2 + y2^2
    → l (2/7) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l680_68010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_great_wall_scientific_notation_l680_68098

/-- Represents the length of the Great Wall in meters -/
def great_wall_length : ℝ := 6700000

/-- Expresses a real number in scientific notation -/
noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let base := Real.log x / Real.log 10
  let exponent := Int.floor base
  (x / (10 ^ exponent), exponent)

/-- Theorem stating that the Great Wall's length in scientific notation is 6.7 × 10^6 -/
theorem great_wall_scientific_notation :
  scientific_notation great_wall_length = (6.7, 6) := by
  sorry

#eval great_wall_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_great_wall_scientific_notation_l680_68098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_323_l680_68007

theorem divisibility_by_323 (n : ℕ) :
  323 ∣ (20^n + 16^n - 3^n - 1) ↔ ∃ k : ℕ, n = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_323_l680_68007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l680_68083

/-- An ellipse with eccentricity √6/3 and points A and B on it -/
structure EllipseWithPoints where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 2/3
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_ellipse : (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ (B.1^2 / a^2 + B.2^2 / b^2 = 1)
  h_midpoint : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (3, 1)

/-- The main theorem about the ellipse and its points -/
theorem ellipse_theorem (E : EllipseWithPoints) :
  -- The circle with diameter AB is tangent to √2x + y - 1 = 0
  (∃ (c : ℝ × ℝ), ((c.1 - 3)^2 + (c.2 - 1)^2) * 2 = (E.A.1 - E.B.1)^2 + (E.A.2 - E.B.2)^2 ∧
    |Real.sqrt 2 * c.1 + c.2 - 1| = Real.sqrt ((E.A.1 - E.B.1)^2 + (E.A.2 - E.B.2)^2) / 2) →
  -- Then the line AB has equation x + y - 4 = 0
  (∀ (x y : ℝ), (x - E.A.1) * (E.B.2 - E.A.2) = (y - E.A.2) * (E.B.1 - E.A.1) ↔ x + y = 4) ∧
  -- And the ellipse has equation x^2/24 + y^2/8 = 1
  E.a^2 = 24 ∧ E.b^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l680_68083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_intersection_range_l680_68063

structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + m * p.2 + m = 0}

theorem line_segment_intersection_range (PQ : LineSegment) 
  (h1 : PQ.start = (-1, 1)) (h2 : PQ.endpoint = (2, 2)) :
  ∀ m : ℝ, (Line m ∩ Set.Icc PQ.start PQ.endpoint).Nonempty → -3 < m ∧ m < -2/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_intersection_range_l680_68063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l680_68028

/-- The area of the region between two externally tangent circles with radii 3 and 4,
    and a third circle that circumscribes them. -/
noncomputable def shaded_area : ℝ := 24 * Real.pi

/-- The radius of the larger circle that circumscribes two externally tangent circles
    with radii 3 and 4. -/
def large_circle_radius : ℝ := 7

theorem shaded_area_proof :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 4
  let R : ℝ := large_circle_radius
  shaded_area = R^2 * Real.pi - r₁^2 * Real.pi - r₂^2 * Real.pi :=
by
  -- Unfold definitions
  unfold shaded_area large_circle_radius
  -- Perform algebraic manipulations
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l680_68028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_process_iff_equilateral_l680_68064

/-- A triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The semi-perimeter of a triangle -/
noncomputable def semi_perimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- The process of forming a new triangle -/
noncomputable def next_triangle (t : Triangle) : Triangle :=
  { a := semi_perimeter t - t.a,
    b := semi_perimeter t - t.b,
    c := semi_perimeter t - t.c,
    pos_a := sorry,
    pos_b := sorry,
    pos_c := sorry,
    triangle_inequality := sorry }

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

/-- The process can be repeated indefinitely -/
def indefinite_process (t : Triangle) : Prop :=
  ∀ n : ℕ, ∃ t' : Triangle, t' = (next_triangle^[n]) t

/-- The main theorem -/
theorem indefinite_process_iff_equilateral (t : Triangle) :
  indefinite_process t ↔ is_equilateral t := by
  sorry

#check indefinite_process_iff_equilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_process_iff_equilateral_l680_68064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_and_value_l680_68053

/-- The ellipse equation --/
noncomputable def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

/-- The line equation --/
def line (x y : ℝ) : Prop := x - 2*y - 12 = 0

/-- The distance function from a point to the line --/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - 2*y - 12| / Real.sqrt 5

/-- The theorem statement --/
theorem min_distance_point_and_value :
  ∃ (x y : ℝ),
    ellipse x y ∧
    (∀ (x' y' : ℝ), ellipse x' y' → distance_to_line x y ≤ distance_to_line x' y') ∧
    x = 2 ∧ y = -3 ∧
    distance_to_line x y = (4 * Real.sqrt 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_and_value_l680_68053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_zero_rational_one_point_four_one_four_rational_sqrt_nine_rational_l680_68086

theorem sqrt_two_irrational :
  ∃ (x : ℝ), Irrational x ∧ x = Real.sqrt 2 :=
by
  sorry

theorem zero_rational : ∃ (q : ℚ), (q : ℝ) = 0 :=
by
  sorry

theorem one_point_four_one_four_rational : 
  ∃ (q : ℚ), (q : ℝ) = 1.414 :=
by
  sorry

theorem sqrt_nine_rational : ∃ (q : ℚ), (q : ℝ) = Real.sqrt 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_zero_rational_one_point_four_one_four_rational_sqrt_nine_rational_l680_68086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_l680_68033

-- Define the piecewise function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 0 then x^3 + 1 else a * x^3 + b

-- State the theorem
theorem even_function_property (a b : ℝ) :
  (∀ x, f x a b = f (-x) a b) →
  2^a + b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_l680_68033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_10_sum_row_11_sum_row_sum_pattern_l680_68026

-- Define Pascal's Triangle using well-founded recursion
def pascal : ℕ → ℕ → ℕ
| 0, _ => 1
| n+1, 0 => 1
| n+1, k+1 => pascal n k + pascal n (k+1)

-- Define the sum of a row in Pascal's Triangle
def rowSum (n : ℕ) : ℕ :=
  (List.range (n+1)).map (pascal n) |> List.sum

-- Theorem for Row 10
theorem row_10_sum : rowSum 10 = 2^10 := by sorry

-- Theorem for Row 11
theorem row_11_sum : rowSum 11 = 2^11 := by sorry

-- Theorem to verify the pattern
theorem row_sum_pattern (n : ℕ) : rowSum n = 2^n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_10_sum_row_11_sum_row_sum_pattern_l680_68026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_theorem_l680_68013

/-- A coloring is "good" if at least two points are black and one of the two arcs formed by these two black endpoints contains exactly n points. -/
def is_good_coloring (n : ℕ) (E : Finset ℕ) (black_points : Finset ℕ) : Prop :=
  (black_points.card ≥ 2) ∧
  (∃ (a b : ℕ), a ∈ black_points ∧ b ∈ black_points ∧ a ≠ b ∧
    ((Finset.Icc a b ∩ E).card = n ∨ (Finset.Icc b a ∩ E).card = n))

/-- The minimum number of black points that guarantees a "good" coloring. -/
def k_min (n : ℕ) : ℕ :=
  if 3 ∣ (2 * n - 1) then n - 1 else n

theorem circle_coloring_theorem (n : ℕ) (hn : n ≥ 3) :
  let E : Finset ℕ := Finset.range (2 * n - 1)
  ∀ (k : ℕ), k ≥ k_min n →
    ∀ (black_points : Finset ℕ), black_points ⊆ E ∧ black_points.card = k →
      is_good_coloring n E black_points :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_theorem_l680_68013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_is_increasing_on_positive_reals_l680_68004

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x)

-- Theorem 1: f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  simp [f]
  sorry

-- Theorem 2: f is increasing on (0, +∞)
theorem f_is_increasing_on_positive_reals : 
  ∀ x y : ℝ, 0 < x → x < y → f x < f y := by
  intros x y hx hxy
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_is_increasing_on_positive_reals_l680_68004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l680_68041

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the eccentricity
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

-- Define the line DE
def LineDE (f₁ : ℝ × ℝ) (slope : ℝ) : Set (ℝ × ℝ) := {p | p.2 - f₁.2 = slope * (p.1 - f₁.1)}

-- Define the triangle ADE
def TriangleADE (a d e : ℝ × ℝ) : Set (ℝ × ℝ) := {p | p = a ∨ p = d ∨ p = e}

-- Define the perimeter of a triangle
noncomputable def Perimeter (t : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (a b : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : Eccentricity a b = 1/2) 
  (f₁ f₂ : ℝ × ℝ) 
  (h₄ : f₁ ∈ Ellipse a b ∧ f₂ ∈ Ellipse a b) 
  (d e : ℝ × ℝ) 
  (h₅ : d ∈ Ellipse a b ∧ e ∈ Ellipse a b) 
  (h₆ : d ∈ LineDE f₁ (Real.sqrt 3 / 3) ∧ e ∈ LineDE f₁ (Real.sqrt 3 / 3)) 
  (h₇ : Real.sqrt ((d.1 - e.1)^2 + (d.2 - e.2)^2) = 6) :
  Perimeter (TriangleADE (0, b) d e) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l680_68041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_grid_midpoint_property_l680_68070

/-- A point in a 2D grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A regular hexagonal grid --/
structure HexagonalGrid where
  nodes : Finset GridPoint

/-- Check if a point is a midpoint of two other points --/
def isMidpoint (p1 p2 m : GridPoint) : Prop :=
  2 * m.x = p1.x + p2.x ∧ 2 * m.y = p1.y + p2.y

/-- The main theorem --/
theorem hexagonal_grid_midpoint_property (grid : HexagonalGrid) :
  ∃ (nodes : Finset GridPoint), nodes ⊆ grid.nodes ∧ nodes.card = 9 ∧
  ∀ (subset : Finset GridPoint), subset ⊆ nodes →
    subset.card = 5 →
    ∃ (p1 p2 m : GridPoint), p1 ∈ subset ∧ p2 ∈ subset ∧ m ∈ grid.nodes ∧
    p1 ≠ p2 ∧ isMidpoint p1 p2 m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_grid_midpoint_property_l680_68070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sixth_power_not_equal_cube_root_log_square_not_equal_square_log_incorrect_statements_l680_68077

-- Statement A
theorem sqrt_sixth_power_not_equal_cube_root :
  ¬(∀ x : ℝ, (x^2)^(1/6 : ℝ) = x^(1/3 : ℝ)) :=
sorry

-- Statement D
theorem log_square_not_equal_square_log :
  ¬(∀ x : ℝ, Real.log (x^2) = (Real.log x)^2) :=
sorry

-- Combined theorem
theorem incorrect_statements :
  (¬(∀ x : ℝ, (x^2)^(1/6 : ℝ) = x^(1/3 : ℝ))) ∧
  (¬(∀ x : ℝ, Real.log (x^2) = (Real.log x)^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sixth_power_not_equal_cube_root_log_square_not_equal_square_log_incorrect_statements_l680_68077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l680_68061

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 6

-- Define the equation f(f(x)) = 5
def equation (x : ℝ) : Prop := f (f x) = 5

-- Theorem stating that the equation has exactly 3 solutions
theorem equation_has_three_solutions :
  ∃ (a b c : ℝ), (∀ x : ℝ, equation x ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry

#check equation_has_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l680_68061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l680_68099

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
def PointOnEllipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- The foci of the ellipse -/
noncomputable def Foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  ((c, 0), (-c, 0))

/-- Intersection points of PF₁ and PF₂ with the ellipse -/
def IntersectionPoints (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  PointOnEllipse a b P ∧ PointOnEllipse a b A ∧ PointOnEllipse a b B ∧
  (F₁.1 - P.1) * (A.2 - P.2) = (F₁.2 - P.2) * (A.1 - P.1) ∧
  (F₂.1 - P.1) * (B.2 - P.2) = (F₂.2 - P.2) * (B.1 - P.1)

/-- The main theorem -/
theorem ellipse_ratio_sum (a b : ℝ) (e : Ellipse a b) (P A B : ℝ × ℝ) :
  let (F₁, F₂) := Foci a b
  IntersectionPoints a b P F₁ F₂ A B →
  (dist P F₁ / dist F₁ A) + (dist P F₂ / dist F₂ B) = 2 * Real.sqrt (1 - b^2/a^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l680_68099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l680_68089

/-- The positive slope of the asymptote of a hyperbola -/
noncomputable def asymptote_slope (a b c : ℝ) : ℝ := b / a

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) - Real.sqrt ((x - 8)^2 + (y + 3)^2) = 4

/-- Theorem: The positive slope of the asymptote of the given hyperbola is √5/2 -/
theorem hyperbola_asymptote_slope :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  c^2 = a^2 + b^2 ∧
  c = 3 ∧
  a = 2 ∧
  asymptote_slope a b c = Real.sqrt 5 / 2 := by
  -- Proof goes here
  sorry

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l680_68089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l680_68095

/-- Represents a quadrilateral PQRS with specific properties -/
structure Quadrilateral :=
  (PQ : ℝ)
  (QS : ℝ)
  (PR : ℝ)
  (angle_QPR : ℝ)

/-- Calculates the area of the quadrilateral PQRS -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  7 * Real.sqrt 3 + 2 * Real.sqrt 33

/-- Theorem stating that the area of the quadrilateral PQRS is 7√3 + 2√33 square miles -/
theorem area_of_quadrilateral (q : Quadrilateral) 
  (h1 : q.PQ = 4)
  (h2 : q.QS = 5)
  (h3 : q.PR = 7)
  (h4 : q.angle_QPR = Real.pi / 3) : 
  area q = 7 * Real.sqrt 3 + 2 * Real.sqrt 33 := by
  sorry

#check area_of_quadrilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l680_68095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l680_68029

noncomputable def sequence_x : ℕ → ℝ
  | 0 => 0
  | n + 1 => 3 * sequence_x n + Real.sqrt (8 * (sequence_x n)^2 + 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  (Real.sqrt 2 / 8) * ((3 + 2 * Real.sqrt 2)^n - (3 - 2 * Real.sqrt 2)^n)

theorem sequence_general_term :
  ∀ n : ℕ, sequence_x n = general_term n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l680_68029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_BMN_l680_68065

-- Define the plane
variable (plane : Type) [NormedAddCommGroup plane] [InnerProductSpace ℝ plane] [FiniteDimensional ℝ plane]

-- Define points A, B, and P
variable (A B P : plane)

-- Define the condition |PA| + |PB| = 2
variable (h_ellipse : ‖P - A‖ + ‖P - B‖ = 2)

-- Define the trajectory of P as an ellipse
def trajectory (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l: y = k
def line (k : ℝ) (x y : ℝ) : Prop := y = k ∧ k > 0

-- Define the area of triangle BMN
noncomputable def area_BMN (k : ℝ) : ℝ := k / Real.sqrt (1 + 4 * k^2)

-- State the theorem
theorem max_area_BMN :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (k' : ℝ), k' > 0 → area_BMN k' ≤ area_BMN k) ∧
  area_BMN k = 1/2 ∧
  k = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_BMN_l680_68065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_third_l680_68024

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sqrt 3 * Real.cos x

-- State the theorem
theorem f_derivative_at_pi_third : 
  (deriv f) (π / 3) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_third_l680_68024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_one_l680_68085

open Real

theorem max_value_implies_a_equals_one :
  ∀ a : ℝ,
  let f := λ x : ℝ ↦ a * x * sin x - 3/2
  (∀ x ∈ Set.Icc 0 (π/2), f x ≤ (π - 3)/2) ∧
  (∃ x ∈ Set.Icc 0 (π/2), f x = (π - 3)/2) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_one_l680_68085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_exists_l680_68017

theorem original_number_exists : ∃ x : ℤ, (73 * x - 17) / 5 - (61 * x + 23) / 7 = 183 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_exists_l680_68017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irises_needed_for_ratio_irises_needed_for_ratio_is_20_l680_68032

/-- Calculate the number of irises needed to maintain a given ratio with roses -/
theorem irises_needed_for_ratio (initial_roses : ℕ) (roses_given_away : ℕ) (roses_added : ℕ) : ℕ :=
  let final_roses : ℕ := initial_roses - roses_given_away + roses_added
  let irises_to_roses_ratio : ℚ := 3 / 7
  let irises_needed : ℕ := (irises_to_roses_ratio * final_roses).ceil.toNat
  irises_needed

theorem irises_needed_for_ratio_is_20 
  (initial_roses : ℕ) 
  (roses_given_away : ℕ) 
  (roses_added : ℕ) 
  (h1 : initial_roses = 35)
  (h2 : roses_given_away = 15)
  (h3 : roses_added = 25) : 
  irises_needed_for_ratio initial_roses roses_given_away roses_added = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irises_needed_for_ratio_irises_needed_for_ratio_is_20_l680_68032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badge_pricing_theorem_l680_68022

noncomputable def f (x : ℝ) : ℝ := 2 * x + 288 / x

theorem badge_pricing_theorem :
  (∀ x ∈ ({2, 6, 32} : Set ℝ), f x = if x = 2 then 148 else if x = 6 then 60 else 73) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f x₀ ≤ f x ∧ f x₀ = 48) ∧
  (∀ k : ℝ, (k ≥ 13 ↔ ∀ x ≥ k, k * f x - 32 * k - 210 ≥ 0)) := by
  sorry

#check badge_pricing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_badge_pricing_theorem_l680_68022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l680_68030

/-- The fixed point A lies on the line ax - y + 4 - 2a = 0 for all a ∈ ℝ -/
def A : ℝ × ℝ := (2, 4)

/-- The line y = x - 2 in point-normal form -/
noncomputable def line (p : ℝ × ℝ) : ℝ := p.1 - p.2 - 2

/-- The distance from a point to a line in point-normal form -/
noncomputable def distance (p : ℝ × ℝ) : ℝ := |line p| / Real.sqrt 2

theorem distance_to_line : distance A = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l680_68030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l680_68038

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ,  cos θ]]

noncomputable def angle : ℝ := 150 * π / 180

theorem rotation_150_degrees :
  rotation_matrix angle = ![![-Real.sqrt 3 / 2, -1 / 2],
                            ![1 / 2, -Real.sqrt 3 / 2]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l680_68038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_groceries_expense_is_3500_l680_68088

/-- Calculates the amount spent on groceries given expense and savings information --/
def calculate_groceries_expense (rent milk education petrol miscellaneous savings : ℚ) 
  (savings_percentage : ℚ) : ℚ :=
  let total_salary := savings / savings_percentage
  let total_expenses := (1 - savings_percentage) * total_salary
  total_expenses - (rent + milk + education + petrol + miscellaneous)

/-- Theorem stating that given the specific expenses and savings, the groceries expense is 3500 --/
theorem groceries_expense_is_3500 : 
  calculate_groceries_expense 5000 1500 2500 2000 3940 2160 (1/10) = 3500 := by
  sorry

#eval calculate_groceries_expense 5000 1500 2500 2000 3940 2160 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_groceries_expense_is_3500_l680_68088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l680_68051

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x - x - 1

/-- Helper function to calculate the area of the triangle formed by the tangent line at a point -/
noncomputable def area_of_triangle_formed_by_tangent_line (f : ℝ → ℝ) (a : ℝ) : ℝ := 
  sorry -- This would be defined based on the tangent line and coordinate axes

theorem f_properties :
  (∃ (A : ℝ), A = 2 ∧ A = area_of_triangle_formed_by_tangent_line f 0) ∧
  (∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 0 ∧
    ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l680_68051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_product_sum_l680_68011

def P (C : Finset Nat) : Nat :=
  if C.card = 0 then 1 else C.prod id

theorem subset_product_sum (n : Nat) :
  (Finset.powerset (Finset.range n)).sum (fun C => 1 / (P C : ℚ)) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_product_sum_l680_68011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_comparison_l680_68016

-- Define lg as the base-2 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_comparison : 
  (0 < lg 2) ∧ (lg 2 < 1) → lg 2 > (lg 2)^2 ∧ (lg 2)^2 > lg (lg 2) := by
  sorry

#check log_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_comparison_l680_68016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumscribed_circle_l680_68039

-- Define the slopes of the lines
noncomputable def m1 : ℝ := -1/3
noncomputable def m2 : ℝ → ℝ := λ k => k

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop := m1 * m2 k = -1

-- Define the theorem
theorem quadrilateral_circumscribed_circle (k : ℝ) :
  perpendicular k → k = 3 := by
  intro h
  -- The proof goes here
  sorry

#check quadrilateral_circumscribed_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circumscribed_circle_l680_68039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_logarithm_l680_68069

theorem arithmetic_sequence_logarithm (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  let seq := λ k : ℕ => Real.log (a^(2 + 4*(k-1)) * b^(4 + 5*(k-1)))
  (∀ k : ℕ, seq (k+1) - seq k = seq 2 - seq 1) →
  (∃ n : ℕ, seq 10 = Real.log (a^n)) →
  n = 38 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_logarithm_l680_68069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l680_68072

open Real

-- Define the interval
noncomputable def a : ℝ := 0.00005
noncomputable def b : ℝ := 0.0005

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x)

-- Define the set of x-intercepts within the interval
def X : Set ℝ := {x | a < x ∧ x < b ∧ f x = 0}

-- State the theorem
theorem x_intercepts_count : ∃ S : Finset ℝ, S.card = 5730 ∧ ∀ x ∈ S, x ∈ X := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_count_l680_68072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l680_68034

/-- Definition of a magic square -/
def is_magic_square {n : ℕ} (square : Fin n → Fin n → ℕ) : Prop :=
  (∀ i j, square i j ∈ Finset.range (n^2 + 1) \ {0}) ∧
  (∀ i, (Finset.univ.sum (λ j => square i j)) = (Finset.univ.sum (λ j => square j i))) ∧
  (∀ i j i' j', i ≠ j → square i j ≠ square i' j')

/-- The sum of each row and column in an n x n magic square -/
theorem magic_square_sum {n : ℕ} (square : Fin n → Fin n → ℕ) 
  (h : is_magic_square square) : 
  ∀ i, (Finset.univ.sum (λ j => square i j)) = n * (n^2 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l680_68034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_on_circles_l680_68094

-- Define a circle as a center point and a radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line as two points
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

-- Helper functions (declarations only)
def is_common_chord : Circle → Circle → Circle → ℝ × ℝ → ℝ × ℝ → Prop := sorry
def line_through : Line → ℝ × ℝ → Prop := sorry
def point_on_circle : ℝ × ℝ → Circle → Prop := sorry
def point_between : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop := sorry
def distance : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Define the theorem
theorem constant_ratio_on_circles 
  (c1 c2 c3 : Circle) 
  (A B : ℝ × ℝ) 
  (h_common_chord : is_common_chord c1 c2 c3 A B) 
  (l : Line) 
  (h_line_through_A : line_through l A) 
  (h_line_not_AB : l ≠ Line.mk A B) 
  (X Y Z : ℝ × ℝ) 
  (h_X_on_c1 : point_on_circle X c1) 
  (h_Y_on_c2 : point_on_circle Y c2) 
  (h_Z_on_c3 : point_on_circle Z c3) 
  (h_Y_between_XZ : point_between Y X Z) :
  ∃ (k : ℝ), ∀ (X' Y' Z' : ℝ × ℝ), 
    point_on_circle X' c1 → 
    point_on_circle Y' c2 → 
    point_on_circle Z' c3 → 
    point_between Y' X' Z' → 
    line_through (Line.mk A X') A → 
    Line.mk A X' ≠ Line.mk A B → 
    distance X' Y' / distance Y' Z' = k :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_on_circles_l680_68094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_D_defined_for_one_and_two_l680_68073

-- Define the four functions as noncomputable
noncomputable def f_A (x : ℝ) : ℝ := 1 / (x - 2)
noncomputable def f_B (x : ℝ) : ℝ := 1 / (x - 1)
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (x - 2)
noncomputable def f_D (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- Theorem stating that only f_D is defined for both x = 1 and x = 2
theorem only_f_D_defined_for_one_and_two :
  (¬ ∃ y, f_A 1 = y) ∧
  (¬ ∃ y, f_A 2 = y) ∧
  (¬ ∃ y, f_B 1 = y) ∧
  (∃ y, f_B 2 = y) ∧
  (¬ ∃ y, f_C 1 = y) ∧
  (∃ y, f_C 2 = y) ∧
  (∃ y, f_D 1 = y) ∧
  (∃ y, f_D 2 = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_D_defined_for_one_and_two_l680_68073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_terms_l680_68060

/-- Sum of 2013-th powers of digits of a natural number -/
def S (x : ℕ) : ℕ := sorry

/-- Sequence defined by a₁ = 2013 and aₙ₊₁ = S(aₙ) -/
def a : ℕ → ℕ
  | 0 => 2013  -- Add this case to handle n = 0
  | n + 1 => S (a n)

/-- Theorem stating the existence of distinct i and j such that aᵢ = a⃗ -/
theorem exists_equal_terms : ∃ i j : ℕ, i ≠ j ∧ i > 0 ∧ j > 0 ∧ a i = a j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equal_terms_l680_68060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_function_implies_positive_exponent_l680_68001

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem increasing_power_function_implies_positive_exponent :
  (∀ x y : ℝ, 0 < x ∧ x < y → power_function a x < power_function a y) →
  a > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_function_implies_positive_exponent_l680_68001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l680_68002

theorem cosine_of_angle_through_point (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = -1 ∧ r * Real.sin α = 2) →
  Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l680_68002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_testes_suitable_for_meiosis_observation_l680_68093

/-- Represents different types of cells -/
inductive CellType
| SomaticCell
| PrimordialGermCell
| SpermCell
| EggCell

/-- Represents different types of cell division -/
inductive DivisionType
| Mitosis
| Meiosis
| Amitosis

/-- Represents different types of organs -/
inductive OrganType
| Reproductive
| NonReproductive

/-- Represents different types of observable processes -/
inductive ObservableProcess
| Meiosis
| Other

/-- Represents the suitability of a material for observing a process -/
def Suitability := Bool

/-- Function to determine the type of cell division a cell undergoes -/
def cellDivisionType (c : CellType) : DivisionType := 
  sorry

/-- Function to determine if an organ is reproductive -/
def isReproductive (o : OrganType) : Bool := 
  sorry

/-- Function to determine if an organ produces sperm -/
def producesSperm (o : OrganType) : Bool := 
  sorry

/-- Function to determine the suitability of an organ for observing a process -/
def suitabilityForObservation (o : OrganType) (p : ObservableProcess) : Suitability := 
  sorry

theorem testes_suitable_for_meiosis_observation :
  ∀ (c : CellType) (o : OrganType) (p : ObservableProcess),
    (∀ c', cellDivisionType c' ∈ ({DivisionType.Mitosis, DivisionType.Meiosis, DivisionType.Amitosis} : Set DivisionType)) →
    (p = ObservableProcess.Meiosis → cellDivisionType c = DivisionType.Meiosis) →
    (cellDivisionType c = DivisionType.Meiosis → isReproductive o = true) →
    (isReproductive o = true ∧ producesSperm o = true → 
      suitabilityForObservation o ObservableProcess.Meiosis = true) →
    (o = OrganType.Reproductive ∧ producesSperm o = true) →
    suitabilityForObservation o p = true :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_testes_suitable_for_meiosis_observation_l680_68093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_theorem_l680_68005

-- Define the integral condition
noncomputable def integral_condition (m : ℝ) : Prop :=
  ∫ x in (1 : ℝ)..m, (2*x - 1) = 6

-- Define the sum of coefficients of the binomial expansion
noncomputable def sum_of_coefficients (m : ℝ) : ℝ :=
  (1 - 2)^(3*m : ℝ)

-- State the theorem
theorem sum_of_coefficients_theorem (m : ℝ) :
  integral_condition m → sum_of_coefficients m = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_theorem_l680_68005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_paints_60_l680_68079

/-- Represents the amount of work done by each person -/
structure WorkShare where
  allen : ℕ
  ben : ℕ
  charlie : ℕ

/-- Calculates Charlie's share of work given the total area and work ratio -/
def charlies_work (total_area : ℕ) (ratio : WorkShare) : ℕ :=
  total_area * ratio.charlie / (ratio.allen + ratio.ben + ratio.charlie)

/-- Theorem: Charlie paints 60 square feet given the conditions -/
theorem charlie_paints_60 (total_area : ℕ) (ratio : WorkShare) 
  (h1 : total_area = 300)
  (h2 : ratio = { allen := 3, ben := 5, charlie := 2 }) :
  charlies_work total_area ratio = 60 := by
  sorry

#eval charlies_work 300 { allen := 3, ben := 5, charlie := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_paints_60_l680_68079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_l680_68019

/-- Represents a track with semicircular ends and straight sections -/
structure Track where
  inner_radius : ℝ
  width : ℝ
  straight_length : ℝ

/-- Calculates the total length of a lap on the track -/
noncomputable def lap_length (t : Track) (outer : Bool) : ℝ :=
  2 * t.straight_length + 2 * Real.pi * (t.inner_radius + if outer then t.width else 0)

/-- Theorem stating that the runner's speed is π/3 m/s given the track conditions -/
theorem runner_speed (t : Track) (time_diff : ℝ) :
  t.width = 8 ∧ time_diff = 48 →
  ∃ s : ℝ, s = Real.pi / 3 ∧ 
    lap_length t true / s = lap_length t false / s + time_diff := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_l680_68019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_and_roots_l680_68044

theorem inequalities_and_roots (a b : ℝ) (ha : a ≥ b) (hb : b > 0) (hab : a + b = 1) :
  (∀ m n : ℕ+, m < n → a^(m : ℝ) - a^(n : ℝ) ≥ b^(m : ℝ) - b^(n : ℝ) ∧ b^(m : ℝ) - b^(n : ℝ) > 0) ∧
  (∀ n : ℕ+, ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo (-1) 1 ∧ x₂ ∈ Set.Ioo (-1) 1 ∧ 
    x₁^2 - b^(n : ℝ) * x₁ - a^(n : ℝ) = 0 ∧ x₂^2 - b^(n : ℝ) * x₂ - a^(n : ℝ) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_and_roots_l680_68044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_l680_68008

/-- The number of regions formed by n intersecting circles in a plane -/
def f (n : ℕ) : ℕ := n^2 - n + 2

/-- A circle in the plane -/
structure Circle where
  id : ℕ

/-- Predicate for a point being on a circle -/
def circle_intersect (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Theorem: The number of regions formed by n intersecting circles -/
theorem circle_regions (n : ℕ) :
  (∀ i j : Circle, i.id ≠ j.id → (∃ p q : ℝ × ℝ, p ≠ q ∧ circle_intersect i p ∧ circle_intersect i q ∧ 
                                   circle_intersect j p ∧ circle_intersect j q)) →
  (∀ i j k : Circle, i.id ≠ j.id ∧ j.id ≠ k.id ∧ i.id ≠ k.id → 
    ¬∃ p : ℝ × ℝ, circle_intersect i p ∧ circle_intersect j p ∧ circle_intersect k p) →
  f 1 = 2 →
  f 2 = 4 →
  f 3 = 8 →
  f n = n^2 - n + 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_l680_68008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l680_68055

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin x ^ 2

theorem f_properties :
  ∃ (period : ℝ) (max_val : ℝ) (min_val : ℝ),
    period = Real.pi ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = max_val) ∧
    max_val = 2 ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = min_val) ∧
    min_val = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l680_68055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_level_drop_l680_68081

noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def heightChange (r : ℝ) (v : ℝ) : ℝ := v / (Real.pi * r^2)

theorem oil_level_drop (r_stationary h_stationary : ℝ) 
  (r1 h1 r2 h2 r3 h3 : ℝ) : 
  r_stationary = 100 ∧ h_stationary = 25 ∧ 
  r1 = 5 ∧ h1 = 12 ∧ 
  r2 = 6 ∧ h2 = 15 ∧ 
  r3 = 7 ∧ h3 = 18 → 
  heightChange r_stationary 
    (cylinderVolume r1 h1 + cylinderVolume r2 h2 + cylinderVolume r3 h3) = 0.1722 := by
  sorry

#check oil_level_drop

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_level_drop_l680_68081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_one_value_l680_68035

/-- Given a cubic polynomial f(x) = x^3 + ax^2 + bx + c with 1 < a < b < c,
    and a polynomial h(x) with leading coefficient 1 whose roots are the
    squares of the reciprocals of the roots of f(x), prove that
    h(1) = (1 + a^2 + b^2 + c^2) / c^2 -/
theorem h_one_value (a b c : ℝ) (h f : ℝ → ℝ) 
    (h_monic : ∃ p q r : ℝ, ∀ x, h x = x^3 + p * x^2 + q * x + r)
    (f_def : ∀ x, f x = x^3 + a * x^2 + b * x + c)
    (h_roots : ∀ r, f r = 0 → h (1 / r^2) = 0)
    (a_lt_b : a < b) (b_lt_c : b < c) (one_lt_a : 1 < a) :
  h 1 = (1 + a^2 + b^2 + c^2) / c^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_one_value_l680_68035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_y_order_l680_68040

/-- Given an inverse proportion function and three points on it, prove the order of their y-coordinates -/
theorem inverse_proportion_y_order (a : ℝ) (y₁ y₂ y₃ : ℝ) (h_a : a ≠ 0) :
  (∀ x, x ≠ 0 → -a^2 / x = -a^2 / x) →  -- function definition
  -a^2 / (-3) = y₁ →  -- point 1
  -a^2 / (-15) = y₂ →  -- point 2
  -a^2 / 2 = y₃ →  -- point 3
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_y_order_l680_68040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_edge_sum_rounded_l680_68043

/-- The sum of edge lengths of a right pyramid with square base -/
theorem pyramid_edge_sum_rounded (base_side : ℝ) (height : ℝ) : base_side = 8 → height = 10 → 
  ⌊4 * base_side + 4 * Real.sqrt ((base_side / 2 * Real.sqrt 2) ^ 2 + height ^ 2) + 0.5⌋ = 78 := by
  sorry

#check pyramid_edge_sum_rounded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_edge_sum_rounded_l680_68043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_in_right_triangle_l680_68074

theorem cos_C_in_right_triangle (A B C : Real) 
  (h1 : A + B + C = 180) 
  (h2 : A = 90)
  (h3 : Real.tan C = 1 / 2)
  (h4 : B = 60) : 
  Real.cos C = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_in_right_triangle_l680_68074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swift_speed_solution_l680_68097

/-- Represents the problem of finding Ms. Swift's ideal speed --/
def SwiftSpeedProblem (leave_time : ℚ) (arrival_time : ℚ) (slow_speed : ℚ) (fast_speed : ℚ) (late_time : ℚ) (early_time : ℚ) : Prop :=
  ∃ (distance : ℚ) (ideal_time : ℚ),
    -- The distance is the same for both speeds
    distance = slow_speed * (ideal_time + late_time) ∧
    distance = fast_speed * (ideal_time - early_time) ∧
    -- The ideal speed
    let ideal_speed := distance / ideal_time
    -- The ideal speed is approximately 41.22 mph
    abs (ideal_speed - 41.22) < 0.01

/-- Theorem stating the solution to Ms. Swift's speed problem --/
theorem swift_speed_solution :
  SwiftSpeedProblem (7.5) (8) (35) (50) (1/12) (1/12) := by
  sorry

#check swift_speed_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swift_speed_solution_l680_68097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_l680_68075

-- Define the passenger capacity function
noncomputable def p (t : ℝ) : ℝ :=
  if 2 ≤ t ∧ t < 10 then 1200 - 10 * (10 - t)^2
  else if 10 ≤ t ∧ t ≤ 20 then 1200
  else 0

-- Define the net profit per minute function
noncomputable def Q (t : ℝ) : ℝ := (6 * p t - 3360) / t - 360

-- Theorem statement
theorem max_net_profit :
  ∃ (t : ℝ), 2 ≤ t ∧ t ≤ 20 ∧
  Q t = 120 ∧
  ∀ (s : ℝ), 2 ≤ s ∧ s ≤ 20 → Q s ≤ Q t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_l680_68075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l680_68037

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def angles_in_geometric_progression (t : Triangle) : Prop :=
  ∃ q : Real, q > 0 ∧ t.A * q = t.B ∧ t.B * q = t.C

def side_condition (t : Triangle) : Prop :=
  t.b^2 - t.a^2 = t.a * t.c

theorem angle_B_measure (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : angles_in_geometric_progression t)
  (h3 : side_condition t) :
  t.B = 2 * Real.pi / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_l680_68037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l680_68045

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := x / (1 + x)

-- State the theorems
theorem derivative_f (x : ℝ) : 
  deriv f x = 6 * x + Real.cos x - x * Real.sin x := by sorry

theorem derivative_g (x : ℝ) (h : x ≠ -1) : 
  deriv g x = 1 / (1 + x)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_derivative_g_l680_68045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_on_interval_l680_68096

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - 1/2 * x^2 - 2*x + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 2

-- Statement to prove
theorem f_max_on_interval :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_on_interval_l680_68096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_f_prime_at_one_l680_68076

noncomputable section

variable (f : ℝ → ℝ)

-- The tangent line at (1, f(1)) has equation 2x + y - 3 = 0
axiom tangent_line : 2 * 1 + f 1 - 3 = 0

-- The derivative of f at x = 1 is the slope of the tangent line
axiom derivative_at_one : deriv f 1 = -2

theorem f_plus_f_prime_at_one : deriv f 1 + f 1 = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_f_prime_at_one_l680_68076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l680_68056

def x : ℝ × ℝ × ℝ := (3, -3, 4)
def p : ℝ × ℝ × ℝ := (1, 0, 2)
def q : ℝ × ℝ × ℝ := (0, 1, 1)
def r : ℝ × ℝ × ℝ := (2, -1, 4)

theorem vector_decomposition :
  x = p + (-2 : ℝ) • q + r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l680_68056
