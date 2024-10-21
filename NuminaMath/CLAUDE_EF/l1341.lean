import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_truck_percentage_l1341_134140

/-- Proves that the percentage of black trucks among all trucks is 20% -/
theorem black_truck_percentage (total_vehicles : ℕ) (total_trucks : ℕ) 
  (h_total : total_vehicles = 90)
  (h_trucks : total_trucks = 50)
  (h_red_trucks : total_trucks / 2 = 25)
  (h_white_trucks : (⌊(17 : ℚ) / 100 * total_vehicles⌋ : ℤ) = 15) : 
  (((total_trucks - (total_trucks / 2) - 15) : ℚ) / total_trucks) * 100 = 20 := by
  sorry

#check black_truck_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_truck_percentage_l1341_134140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_max_area_l1341_134135

/-- The length of the fence enclosing the garden -/
noncomputable def fence_length : ℝ := 20

/-- The area of a rectangular garden given its width -/
noncomputable def garden_area (width : ℝ) : ℝ := width * (fence_length / 2 - width)

/-- The maximum area of the garden -/
noncomputable def max_garden_area : ℝ := 24

/-- Theorem stating that the maximum area of the garden is 24 square meters -/
theorem garden_max_area :
  ∃ (width : ℝ), 0 < width ∧ width < fence_length / 2 ∧
  garden_area width = max_garden_area ∧
  ∀ (w : ℝ), 0 < w → w < fence_length / 2 → garden_area w ≤ max_garden_area := by
  sorry

#check garden_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_max_area_l1341_134135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_circle_completeness_smallest_t_for_complete_circle_l1341_134171

/-- The set of points on the curve r = cos θ for θ in [0, t] -/
def CosCirclePoints (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ p = (Real.cos θ, Real.sin θ * Real.cos θ)}

/-- The set of points on a unit circle -/
def UnitCircle : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p = (Real.cos θ, Real.sin θ)}

theorem cos_circle_completeness :
  ∀ t : ℝ, t > 0 → (CosCirclePoints t = UnitCircle ↔ t ≥ Real.pi) ∧
  (∀ s : ℝ, s > 0 → s < Real.pi → CosCirclePoints s ≠ UnitCircle) := by
  sorry

theorem smallest_t_for_complete_circle :
  ∃! t : ℝ, t > 0 ∧ CosCirclePoints t = UnitCircle ∧
  ∀ s : ℝ, s > 0 → CosCirclePoints s = UnitCircle → s ≥ t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_circle_completeness_smallest_t_for_complete_circle_l1341_134171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l1341_134113

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
noncomputable def g (c d x : ℝ) : ℝ := Real.exp x * (c*x + d)

-- Define the theorem
theorem function_properties_and_inequality 
  (a b c d k : ℝ) :
  -- Conditions
  (f a b 0 = 2) →
  (g c d 0 = 2) →
  ((deriv (f a b)) 0 = 4) →
  ((deriv (g c d)) 0 = 4) →
  -- Conclusions
  (a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∧
  (∀ x ≥ -2, f 4 2 x ≤ k * g 2 2 x ↔ 1 ≤ k ∧ k ≤ Real.exp 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l1341_134113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_added_numbers_l1341_134118

theorem mean_of_added_numbers (initial_count : ℕ) (initial_mean : ℝ) 
  (added_count : ℕ) (new_mean : ℝ) : 
  initial_count = 7 → 
  initial_mean = 63 → 
  added_count = 3 → 
  new_mean = 78 → 
  (((initial_count + added_count) * new_mean - initial_count * initial_mean) / added_count) = 113 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_added_numbers_l1341_134118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_sum_units_digit_l1341_134150

/-- Represents a number in base 8 -/
structure OctalNumber where
  value : ℕ

/-- Converts an octal number to its decimal representation -/
def octal_to_decimal (n : OctalNumber) : ℕ := sorry

/-- Adds two octal numbers -/
def octal_add (a b : OctalNumber) : OctalNumber := sorry

/-- Gets the units digit of an octal number -/
def units_digit (n : OctalNumber) : ℕ := sorry

/-- Creates an OctalNumber from a natural number -/
def mk_octal (n : ℕ) : OctalNumber := ⟨n⟩

/-- Theorem: The units digit of 135₈ + 157₈ + 163₈ in base 8 is 6 -/
theorem octal_sum_units_digit :
  let sum := octal_add (octal_add (mk_octal 135) (mk_octal 157)) (mk_octal 163)
  units_digit sum = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_sum_units_digit_l1341_134150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1341_134125

noncomputable def f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1341_134125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1341_134198

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x ^ (1/3)
def g (x : ℝ) : ℝ := x

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l1341_134198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_2002_l1341_134130

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def subtract_digit_sum (n : ℕ) : ℕ :=
  n - digit_sum n

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

theorem cannot_reach_2002 (start : ℕ) (h : is_six_digit start) :
  ∀ k : ℕ, (Nat.iterate subtract_digit_sum k start) ≠ 2002 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_reach_2002_l1341_134130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_one_sufficient_not_necessary_l1341_134119

-- Define the complex number z
def z (a : ℝ) : ℂ := (a - 2*Complex.I) * (1 + Complex.I)

-- Define the point M in the complex plane
def M (a : ℝ) : ℝ × ℝ := (a + 2, a - 2)

-- Define the condition for being in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Statement of the theorem
theorem a_equals_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → in_fourth_quadrant (M a)) ∧
  ¬(∀ a : ℝ, in_fourth_quadrant (M a) → a = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_one_sufficient_not_necessary_l1341_134119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1341_134161

/-- The number of balls in the bag -/
def total_balls : ℕ := 8

/-- The number of black balls in the bag -/
def black_balls : ℕ := 1

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The probability of drawing two balls of different colors -/
theorem different_color_probability : 
  (Nat.choose total_balls 2 : ℚ) * (19 : ℚ) / (28 : ℚ) = 
  (black_balls * white_balls + black_balls * red_balls + white_balls * red_balls : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1341_134161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_of_primes_l1341_134146

theorem arithmetic_progression_of_primes (p : ℕ) (a : ℕ → ℕ) (d : ℕ) : 
  Prime p →
  (∀ i, i < p → Prime (a i)) →
  (∀ i, i < p - 1 → a (i + 1) = a i + d) →
  (∀ i j, i < j → j < p → a i < a j) →
  a 0 > p →
  p ∣ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_of_primes_l1341_134146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MON_is_right_angle_l1341_134142

/-- Ellipse C with equation 9x² + 16y² = 1 -/
def C : Set (ℝ × ℝ) := {p | 9 * p.1^2 + 16 * p.2^2 = 1}

/-- Circle O with equation x² + y² = 1/25 -/
def O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1/25}

/-- Line l is tangent to circle O -/
def is_tangent_to_O (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ O ∧ p ∈ l ∧ ∀ q : ℝ × ℝ, q ∈ O → q ∈ l → q = p

/-- M and N are intersection points of line l and ellipse C -/
def intersects_C_at (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop :=
  M ∈ C ∧ N ∈ C ∧ M ∈ l ∧ N ∈ l ∧ M ≠ N

/-- Angle between two vectors -/
noncomputable def angle (O M N : ℝ × ℝ) : ℝ :=
  Real.arccos (((M.1 - O.1) * (N.1 - O.1) + (M.2 - O.2) * (N.2 - O.2)) /
    (((M.1 - O.1)^2 + (M.2 - O.2)^2).sqrt * ((N.1 - O.1)^2 + (N.2 - O.2)^2).sqrt))

/-- Theorem: If line l is tangent to O and intersects C at M and N, then angle MON is π/2 -/
theorem angle_MON_is_right_angle
  (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ)
  (h_tangent : is_tangent_to_O l)
  (h_intersect : intersects_C_at l M N) :
  angle (0, 0) M N = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MON_is_right_angle_l1341_134142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_proof_l1341_134154

-- Define a line by two points
def Line (x1 y1 x2 y2 : ℝ) : Set (ℝ × ℝ) :=
  {(m, b) | m = (y2 - y1) / (x2 - x1) ∧ y1 = m * x1 + b}

-- Theorem statement
theorem line_proof (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1 = 1 ∧ y1 = 3)
  (h2 : x2 = 3 ∧ y2 = 7)
  (h3 : x3 = 5 ∧ y3 = 11)
  (h4 : x2 ≠ x1) :
  ∃ (m b : ℝ), 
    (m, b) ∈ Line x1 y1 x2 y2 ∧ 
    m + b = 3 ∧ 
    y3 = m * x3 + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_proof_l1341_134154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1341_134186

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a - 1)^2 + (y - Real.sqrt (3 * a))^2 = 1

-- Define the dot product condition
def dot_product_condition (x y : ℝ) : Prop :=
  x * (x - 2) + y * y = 8

-- Define the existence of point P
def exists_point_P (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ dot_product_condition x y

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, exists_point_P a ↔ (a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1341_134186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_over_quadratic_plus_linear_is_even_l1341_134158

theorem factorial_over_quadratic_plus_linear_is_even (n : ℕ) : 
  ∃ (k : ℤ), (2 * k : ℚ) = (Nat.factorial (n - 1)) / ((n^2 + n) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_over_quadratic_plus_linear_is_even_l1341_134158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_circles_area_l1341_134156

/-- Two circles with the same center, where one is tangential to the other from the inside -/
structure TangentialCircles where
  center : ℝ × ℝ
  radius_inner : ℝ
  radius_outer : ℝ
  tangential : radius_outer = radius_inner + (1 / Real.sqrt Real.pi)

/-- The area of a circle given its radius -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- Theorem: If two circles share the same center, one is tangential to the other from the inside,
    and the area of the smaller circle is 9 square inches, then the area of the larger circle is 16 square inches -/
theorem tangential_circles_area (c : TangentialCircles) 
    (h : circle_area c.radius_inner = 9) : 
    circle_area c.radius_outer = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_circles_area_l1341_134156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_sum_l1341_134100

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2 * Real.sin (x + Real.pi / 4) + 2 * x^2 + x) / (2 * x^2 + Real.cos x)

theorem f_max_min_sum :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ (M + m = 2) := by
  sorry

#check f_max_min_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_sum_l1341_134100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1341_134149

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - π/3) = 1/5) 
  (h2 : α ∈ Set.Ioo 0 (π/2)) : 
  Real.cos α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1341_134149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_length_l1341_134187

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def CircleWith (r : ℝ) (A B C D : Point) : Prop := sorry
def SegmentIntersectsAt (A C B D P : Point) : Prop := sorry
def SegmentLength (A B : Point) : ℝ := sorry

theorem intersection_point_length (A B C D P : Point) (r : ℝ) : 
  CircleWith r A B C D →
  SegmentIntersectsAt A C B D P →
  SegmentLength A P = 9 →
  SegmentLength P C = 2 →
  SegmentLength B D = 10 →
  SegmentLength B P < SegmentLength D P →
  SegmentLength B P = 5 - Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_length_l1341_134187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1341_134114

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + Real.sqrt (3 - x)

-- Define the domain of f
def domain_f : Set ℝ := {x | x > -1 ∧ x ≤ 3}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1341_134114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_drivers_and_second_trip_time_l1341_134122

-- Define the time type
structure Time where
  minutes : ℕ

-- Define the trip duration
def one_way_trip_duration : Time := ⟨160⟩  -- 2 hours and 40 minutes in minutes

-- Define the rest duration
def rest_duration : Time := ⟨60⟩  -- 1 hour in minutes

-- Define the time when Driver A returns
def driver_a_return : Time := ⟨760⟩  -- 12:40 PM in minutes since midnight

-- Define the function to calculate the next departure time
def next_departure (prev_departure : Time) : Time :=
  ⟨prev_departure.minutes + 2 * one_way_trip_duration.minutes + rest_duration.minutes⟩

-- Define the function to check if a driver is available
def driver_available (departure_time : Time) (last_trip_time : Time) : Prop :=
  departure_time.minutes ≥ last_trip_time.minutes + 2 * one_way_trip_duration.minutes + rest_duration.minutes

-- Theorem statement
theorem min_drivers_and_second_trip_time :
  ∃ (num_drivers : ℕ) (second_trip_time : Time),
    num_drivers = 4 ∧
    second_trip_time = ⟨640⟩ ∧  -- 10:40 AM in minutes since midnight
    (∀ (t : Time), t.minutes ≥ 0 → t.minutes < 1440 →
      ∃ (driver : ℕ), driver < num_drivers ∧
        driver_available t (next_departure ⟨t.minutes - 205⟩)) ∧
    (∀ (n : ℕ), n < num_drivers →
      ¬ driver_available second_trip_time (next_departure ⟨second_trip_time.minutes - 205⟩)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_drivers_and_second_trip_time_l1341_134122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1341_134173

-- Define the circle
def circle_set : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define the midpoint of chord PQ
def chord_midpoint : ℝ × ℝ := (1, 2)

-- Define the equation of line PQ
def line_PQ (x y : ℝ) : Prop := x + 2*y - 5 = 0

-- Define a line segment
def in_line_segment (P Q : ℝ × ℝ) (point : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ point = (t • P.1 + (1 - t) • Q.1, t • P.2 + (1 - t) • Q.2)

-- Theorem statement
theorem chord_equation : 
  ∀ (P Q : ℝ × ℝ), 
  P ∈ circle_set → Q ∈ circle_set →
  ((P.1 + Q.1)/2, (P.2 + Q.2)/2) = chord_midpoint →
  ∀ (x y : ℝ), in_line_segment P Q (x, y) → line_PQ x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1341_134173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_distance_bounds_l1341_134178

/-- The hyperbola x²/9 - y²/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The circle (x+5)² + y² = 1 -/
def circle1 (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

/-- The circle (x-5)² + y² = 4 -/
def circle2 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 4

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem hyperbola_circle_distance_bounds :
  ∀ (px py mx my nx ny : ℝ),
  hyperbola px py →
  px > 0 →
  circle1 mx my →
  circle2 nx ny →
  3 ≤ distance px py mx my - distance px py nx ny ∧
  distance px py mx my - distance px py nx ny ≤ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_distance_bounds_l1341_134178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l1341_134181

-- Define the points
variable (P M N F : ℝ × ℝ)

-- Define the conditions
def on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0
def on_x_axis (M : ℝ × ℝ) : Prop := M.2 = 0
def fixed_point_F : ℝ × ℝ := (1, 0)

-- Vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def vec_dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Conditions as propositions
def condition1 (P M N : ℝ × ℝ) : Prop :=
  vec_add (vec_sub N P) (vec_scale (1/2) (vec_sub M N)) = (0, 0)

def condition2 (P M F : ℝ × ℝ) : Prop :=
  vec_dot (vec_sub M P) (vec_sub F P) = 0

-- Theorem statement
theorem trajectory_and_min_area 
  (P M N : ℝ × ℝ) 
  (hP : on_y_axis P) 
  (hM : on_x_axis M) 
  (h1 : condition1 P M N) 
  (h2 : condition2 P M fixed_point_F) :
  (∃ (E : ℝ → ℝ), E = λ x ↦ 2 * Real.sqrt x) ∧
  (∃ (min_area : ℝ), min_area = 2 ∧ 
    ∀ (A B : ℝ × ℝ), A ≠ B → 
      (∃ (m : ℝ), A.1 = m * A.2 + 1 ∧ B.1 = m * B.2 + 1) →
      (A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1) →
      2 ≤ (1/2) * |A.2 - B.2|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l1341_134181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_irrational_in_set_l1341_134111

def given_set : Set ℝ := {Real.sqrt 3, Real.sqrt 4, 3.1415, 5/6}

theorem one_irrational_in_set : ∃! x, x ∈ given_set ∧ Irrational x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_irrational_in_set_l1341_134111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1341_134147

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 2 => 3 * sequence_a (n + 1) + 1

theorem sequence_a_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a (n + 1) + 1/2 = 3 * (sequence_a n + 1/2)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 1/2 * (3^n - 1)) := by
  sorry

#check sequence_a_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1341_134147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_theorem_l1341_134179

/-- Given an initial mixture and added water, calculate the new alcohol percentage -/
noncomputable def new_alcohol_percentage (initial_volume : ℝ) (initial_alcohol_percentage : ℝ) (added_water : ℝ) : ℝ :=
  let initial_alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let new_total_volume := initial_volume + added_water
  (initial_alcohol_volume / new_total_volume) * 100

/-- Theorem: Adding 5 liters of water to 15 liters of 20% alcohol mixture results in 15% alcohol -/
theorem alcohol_dilution_theorem :
  new_alcohol_percentage 15 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_dilution_theorem_l1341_134179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1341_134137

noncomputable def f (x : ℝ) := Real.sqrt (x / (2 - x)) - Real.log (1 - x)

theorem domain_of_f : 
  { x : ℝ | x ∈ Set.Icc 0 1 ∧ x ≠ 1 ∧ x / (2 - x) ≥ 0 ∧ 1 - x > 0 } = Set.Ico 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1341_134137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_die_configuration_l1341_134199

def Die := Finset ℕ

def isMultipleOfThree (n : ℕ) : Bool :=
  n % 3 = 0

def isEven (n : ℕ) : Bool :=
  n % 2 = 0

def validDie (d : Die) : Prop :=
  d.card = 6 ∧
  (d.filter (fun n => isMultipleOfThree n)).card = 3 ∧
  (d.filter (fun n => isEven n)).card = 2

theorem unique_die_configuration :
  ∃! d : Die, validDie d ∧ d = ({1, 2, 3, 3, 5, 6} : Finset ℕ) :=
by sorry

#eval isMultipleOfThree 3
#eval isEven 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_die_configuration_l1341_134199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_cube_root_of_5a_minus_1_l1341_134131

-- Define the real number a
variable (a : ℝ)

-- Define the condition that the square root of a+3 is ±4
def condition : Prop := (a + 3 = 4^2) ∨ (a + 3 = (-4)^2)

-- State the theorem
theorem square_and_cube_root_of_5a_minus_1 (h : condition a) :
  (Real.sqrt (5 * a - 1) = 8) ∧ ((5 * a - 1)^(1/3 : ℝ) = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_cube_root_of_5a_minus_1_l1341_134131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_count_proof_l1341_134132

-- Define the number of coins of each type
def num_coins : ℕ := sorry

-- Define the value of each coin type in rupees
def one_rupee_value : ℚ := 1
def fifty_paise_value : ℚ := 1/2
def twenty_five_paise_value : ℚ := 1/4

-- Define the total value of all coins
def total_value : ℚ := 70

-- Theorem statement
theorem coin_count_proof :
  (num_coins : ℚ) * one_rupee_value +
  (num_coins : ℚ) * fifty_paise_value +
  (num_coins : ℚ) * twenty_five_paise_value = total_value →
  num_coins = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_count_proof_l1341_134132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l1341_134180

open Real

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3

-- Define the radii of the four snowballs
def r1 : ℝ := 4
def r2 : ℝ := 6
def r3 : ℝ := 3
def r4 : ℝ := 7

-- Theorem statement
theorem snowman_volume :
  sphere_volume r1 + sphere_volume r2 + sphere_volume r3 + sphere_volume r4 = (2600 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l1341_134180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1341_134103

/-- Calculates the final amount after compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_growth :
  let principal : ℝ := 12000
  let rate : ℝ := 0.045
  let time : ℕ := 3
  let final_amount := compound_interest principal rate time
  round_to_nearest final_amount = 13674 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1341_134103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1341_134190

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 1 / x

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_line_equation :
  let (x₀, y₀) := intersection_point
  let m := deriv f x₀
  ∀ x y, y - y₀ = m * (x - x₀) ↔ x - 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1341_134190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_of_x_l1341_134193

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

theorem min_abs_sum_of_x (a : ℝ) :
  (∃ x₁ x₂, f a x₁ * f a x₂ = -16) →
  (f a (-π/6) = f a (π/6)) →
  (∃ x₁ x₂, f a x₁ * f a x₂ = -16 ∧ |x₁ + x₂| = 2*π/3 ∧ 
    ∀ y₁ y₂, f a y₁ * f a y₂ = -16 → |y₁ + y₂| ≥ 2*π/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_of_x_l1341_134193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_problem_l1341_134123

/-- The constant ratio function -/
noncomputable def k (x y : ℝ) : ℝ := (3 * x - 4) / (y + 15)

/-- The theorem statement -/
theorem constant_ratio_problem :
  ∀ x₀ y₀ : ℝ,
  (k x₀ y₀ = k 4 5) →  -- The ratio is constant
  (y₀ = 5 → x₀ = 4) →  -- Given condition: y = 5 when x = 4
  (y₀ = 20 → x₀ = 6) ∧ (x₀ = 6 → y₀ = 20) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_problem_l1341_134123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_approx_l1341_134133

/-- Approximate value of pi --/
noncomputable def π : ℝ := Real.pi

/-- Conversion factor from radians to degrees --/
noncomputable def radiansToDegrees : ℝ := 180 / π

/-- The angle in radians --/
def angle : ℝ := 5

/-- Convert radians to degrees --/
noncomputable def angleDegrees : ℝ := angle * radiansToDegrees

/-- Reference angle in the fourth quadrant --/
noncomputable def referenceAngle : ℝ := angleDegrees - 270

/-- Theorem stating that sin(5) is approximately equal to -0.959 --/
theorem sin_five_approx :
  ∃ ε > 0, |Real.sin angle + 0.959| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_approx_l1341_134133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_32_l1341_134169

/-- A geometric sequence with a₂ = 2 and a₆ = 8 -/
def GeometricSequence : Type := {a : ℕ → ℝ // a 2 = 2 ∧ a 6 = 8 ∧ ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m}

/-- The 10th term of the geometric sequence is 32 -/
theorem tenth_term_is_32 : ∀ a : GeometricSequence, a.val 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_32_l1341_134169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nap_duration_l1341_134145

/-- Represents Prudence's sleep schedule and calculates her nap duration --/
def PrudenceSleepSchedule : Nat → Nat → Nat → Nat → Nat → Nat → Nat :=
  λ (weekdaySleepHours weekendNightSleepHours totalSleepIn4Weeks numberOfWeeks weekdaysPerWeek napDaysPerWeek) =>
    let weekdaySleep : Nat := weekdaysPerWeek * weekdaySleepHours
    let weekendNightSleep : Nat := (7 - weekdaysPerWeek) * weekendNightSleepHours
    let weeklySleepWithoutNaps : Nat := weekdaySleep + weekendNightSleep
    (totalSleepIn4Weeks - numberOfWeeks * weeklySleepWithoutNaps) / (numberOfWeeks * napDaysPerWeek)

theorem nap_duration :
  PrudenceSleepSchedule 6 9 200 4 5 2 = 4 := by
  -- Proof goes here
  sorry

#check nap_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nap_duration_l1341_134145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_amount_is_14493_l1341_134108

/-- Represents the estate division problem --/
structure EstateDivision where
  /-- Amount received by the first person --/
  a : ℝ
  /-- Amount received by the second person --/
  b : ℝ
  /-- Amount received by the third person --/
  c : ℝ
  /-- Condition: No amount is within 30% of another --/
  h1 : b ≥ 1.3 * a ∧ c ≥ 1.3 * b
  /-- Condition: Smallest possible range between highest and lowest amounts is $10000 --/
  h2 : c - a = 10000

/-- Theorem: One of the amounts received is $14,493 --/
theorem one_amount_is_14493 (ed : EstateDivision) : 
  (⌊ed.a⌋ : ℤ) = 14493 ∨ (⌊ed.b⌋ : ℤ) = 14493 ∨ (⌊ed.c⌋ : ℤ) = 14493 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_amount_is_14493_l1341_134108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5y2_in_expansion_l1341_134112

noncomputable def coefficient (x y : ℝ) (n m : ℕ) (expr : ℝ) : ℝ :=
  sorry

theorem coefficient_x5y2_in_expansion : ∀ x y : ℝ,
  (coefficient x y 5 2 ((x - y) * (x + 2*y)^6)) = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x5y2_in_expansion_l1341_134112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_and_range_theorem_l1341_134162

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 6*x < -5}
def B : Set ℝ := {x | 1 < Real.exp ((x-2) * Real.log 2) ∧ Real.exp ((x-2) * Real.log 2) ≤ 16}
def C (a : ℝ) : Set ℝ := {x | (2*a - x)*(x - a - 1) > 0}

-- State the theorem
theorem sets_and_range_theorem (a : ℝ) :
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 6}) ∧
  (Aᶜ = {x : ℝ | x ≤ 1 ∨ x ≥ 5}) ∧
  (A ∩ C a = C a → (1/2 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 5/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_and_range_theorem_l1341_134162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1341_134167

/-- A function f from positive reals to reals satisfying f(x+y) = f(x) + f(y) -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x + y) = f x + f y

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) 
  (h8 : f 8 = 4) : 
  f 2 = 1 := by
  sorry

#check functional_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1341_134167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_payment_l1341_134152

/-- Represents the hourly wage of candidate q -/
def q : ℝ := sorry

/-- Represents the hourly wage of candidate p -/
def p : ℝ := sorry

/-- Represents the number of hours candidate p would take to complete the job -/
def h : ℝ := sorry

/-- Candidate p's hourly wage is 50% more than candidate q's -/
axiom p_wage : p = 1.5 * q

/-- Candidate p's hourly wage is 8 dollars greater than candidate q's -/
axiom wage_difference : p = q + 8

/-- Candidate q requires 10 more hours than candidate p to complete the job -/
axiom hours_difference : h + 10 = h

/-- The total payment for the project is the same regardless of who is hired -/
axiom equal_payment : p * h = q * (h + 10)

/-- Theorem: The total payment for the project is $480 -/
theorem project_payment : p * h = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_payment_l1341_134152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_m_l1341_134183

/-- Given an ellipse with equation x^2/25 + y^2/m^2 = 1 (m > 0) and one focus at (0,4), 
    prove that m = √41 -/
theorem ellipse_focus_m (m : ℝ) : 
  m > 0 → 
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/25 + p.2^2/m^2 = 1}) → 
  (0, 4) ∈ {p : ℝ × ℝ | ∃ c, p.1^2/25 + p.2^2/m^2 = 1 ∧ (p.1 - c)^2 + p.2^2 = m^2} → 
  m = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_m_l1341_134183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_sequence_properties_l1341_134110

def f (n : ℕ) (x : ℝ) : ℝ := x^2 - 2*(n + 1)*x + n^2 + 5*n - 7

def a (n : ℕ) : ℝ := 3*n - 8

def b (n : ℕ) : ℝ := |3*n - 8|

theorem vertex_sequence_properties (n : ℕ) :
  (∀ k, a (k + 1) - a k = 3) ∧
  (∀ x, f n x ≥ f n (n + 1 : ℝ)) ∧
  (f n (n + 1 : ℝ) = a n) ∧
  b n = |a n| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_sequence_properties_l1341_134110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_formula_l1341_134105

/-- A convex quadrilateral with perpendicular diagonals -/
structure ConvexQuadrilateral where
  -- The lengths of the diagonals
  a : ℝ
  b : ℝ
  -- Assumption that the quadrilateral is convex and diagonals are perpendicular
  convex : True
  perp_diagonals : True

/-- The area of the quadrilateral formed by the midpoints of the sides -/
noncomputable def midpoint_quadrilateral_area (q : ConvexQuadrilateral) : ℝ :=
  1/4 * q.a * q.b

/-- Theorem stating that the area of the midpoint quadrilateral is 1/4 * a * b -/
theorem midpoint_quadrilateral_area_formula (q : ConvexQuadrilateral) :
  midpoint_quadrilateral_area q = 1/4 * q.a * q.b := by
  -- Unfold the definition of midpoint_quadrilateral_area
  unfold midpoint_quadrilateral_area
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_formula_l1341_134105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_81_l1341_134124

/-- The sum of the proper divisors of 81 is 40 -/
theorem sum_proper_divisors_81 : (Finset.filter (· < 81) (Nat.divisors 81)).sum id = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_81_l1341_134124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1341_134136

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (2 - x)

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ici (-1) ∩ Set.Iio 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1341_134136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1341_134116

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 else x^2 + 1

-- Define the solution set
noncomputable def solution_set : Set ℝ :=
  {x : ℝ | x ≤ -1 ∨ x = -1 + Real.sqrt 2}

-- State the theorem
theorem solution_set_correct :
  ∀ x : ℝ, f (1 - x^2) = f (2*x) ↔ x ∈ solution_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1341_134116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1341_134153

/-- Defines an isosceles triangle -/
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- An isosceles triangle with two sides of length 5 and one side of length 2 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  IsoscelesTriangle a b c →
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1341_134153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l1341_134157

theorem range_of_a_for_false_proposition (a : ℝ) :
  (¬ ∀ x : ℝ, (2 : ℝ)^(x^2 + a*x) ≤ 1/2) ↔ -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_false_proposition_l1341_134157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_exam_l1341_134159

/-- Given a mock exam with 400 students, where 60% of boys and 80% of girls cleared the cut-off,
    and the total percentage of students qualifying is 65%, prove that 100 girls appeared in the examination. -/
theorem girls_in_exam (total_students : ℕ) (boys_clear_rate : ℚ) (girls_clear_rate : ℚ) (total_clear_rate : ℚ) :
  total_students = 400 →
  boys_clear_rate = 60 / 100 →
  girls_clear_rate = 80 / 100 →
  total_clear_rate = 65 / 100 →
  ∃ (girls : ℕ), girls = 100 ∧ 
    ∃ (boys : ℕ), boys + girls = total_students ∧
      (boys_clear_rate * (boys : ℚ) + girls_clear_rate * (girls : ℚ)) / (total_students : ℚ) = total_clear_rate :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_exam_l1341_134159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_simultaneous_return_time_min_simultaneous_return_time_is_12_l1341_134191

def horse_lap_times : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_simultaneous_return (t : Nat) (lap_times : List Nat) : Bool :=
  (lap_times.filter (fun lap_time => t % lap_time = 0)).length ≥ 5

theorem min_simultaneous_return_time :
  ∃ (t : Nat), t > 0 ∧ is_simultaneous_return t horse_lap_times ∧
  ∀ (s : Nat), s > 0 ∧ s < t → ¬is_simultaneous_return s horse_lap_times :=
by sorry

theorem min_simultaneous_return_time_is_12 :
  ∃ (t : Nat), t = 12 ∧ is_simultaneous_return t horse_lap_times ∧
  ∀ (s : Nat), s > 0 ∧ s < t → ¬is_simultaneous_return s horse_lap_times :=
by sorry

#eval is_simultaneous_return 12 horse_lap_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_simultaneous_return_time_min_simultaneous_return_time_is_12_l1341_134191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1341_134126

noncomputable def f (x : ℝ) : ℝ := (3*x - 8)*(x - 2)*(x + 1)/(x - 1)

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  f x ≥ 0 ↔ x ∈ Set.Iic (-1) ∪ Set.Ioo 1 2 ∪ Set.Ioi (8/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1341_134126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_properties_l1341_134192

-- Define the curves f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the conditions
def passes_through_P (h : ℝ → ℝ) : Prop := h 1 = 2

-- Define the common tangent condition
def common_tangent (f g : ℝ → ℝ) : Prop :=
  (deriv f) 1 = (deriv g) 1

-- Define the distance function
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x - y - 2| / Real.sqrt 10

-- State the theorem
theorem curves_properties :
  ∃ (a b c : ℝ),
    passes_through_P (f a) ∧
    passes_through_P (g b c) ∧
    common_tangent (f a) (g b c) ∧
    a = 1 ∧ b = 2 ∧ c = -1 ∧
    (∀ x : ℝ, distance_to_line x (g b c x) ≥ 3*Real.sqrt 10/40) ∧
    (∃ x : ℝ, distance_to_line x (g b c x) = 3*Real.sqrt 10/40) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_properties_l1341_134192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raised_bed_height_is_one_foot_l1341_134163

/-- Represents the dimensions and soil requirements for raised beds -/
structure RaisedBed where
  num_beds : ℕ
  length : ℚ
  width : ℚ
  soil_per_bag : ℚ
  total_bags : ℕ

/-- Calculates the height of a raised bed given its specifications -/
def calculate_height (rb : RaisedBed) : ℚ :=
  let total_volume := (rb.total_bags : ℚ) * rb.soil_per_bag
  let single_bed_volume := total_volume / (rb.num_beds : ℚ)
  single_bed_volume / (rb.length * rb.width)

/-- Theorem stating that the height of the raised beds is 1 foot -/
theorem raised_bed_height_is_one_foot (rb : RaisedBed) 
    (h1 : rb.num_beds = 2)
    (h2 : rb.length = 8)
    (h3 : rb.width = 4)
    (h4 : rb.soil_per_bag = 4)
    (h5 : rb.total_bags = 16) : 
  calculate_height rb = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raised_bed_height_is_one_foot_l1341_134163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l1341_134106

theorem eigenvalues_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 5; 4, 3]
  ∀ (k : ℝ), (∃ (v : Fin 2 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v) ↔ (k = 3 + 2 * Real.sqrt 5 ∨ k = 3 - 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalues_of_specific_matrix_l1341_134106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_pass_probability_l1341_134128

/-- A sequence of 50 distinct real numbers -/
def mySequence : Finset ℝ := sorry

/-- The number of elements in the sequence -/
def n : ℕ := 50

/-- The initial position of the element we're tracking -/
def initial_pos : ℕ := 10

/-- The final position of the element we're tracking -/
def final_pos : ℕ := 15

/-- The probability of the element moving from initial_pos to final_pos after one bubble pass -/
def probability : ℚ := 1 / 240

theorem bubble_pass_probability (s : Finset ℝ) (h : s.card = n) :
  (∀ x y, x ∈ s → y ∈ s → x ≠ y) →  -- All elements are distinct
  probability = (initial_pos.factorial * (n - final_pos).factorial) / n.factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_pass_probability_l1341_134128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_redistribution_l1341_134160

/-- Represents the number of marbles Tyrone gives to Eric -/
def marbles_given : ℕ → Prop := fun _ => True

/-- The initial number of marbles Tyrone has -/
def tyrone_initial : ℕ := 120

/-- The initial number of marbles Eric has -/
def eric_initial : ℕ := 15

/-- Theorem stating the number of marbles Tyrone gives to Eric -/
theorem marbles_redistribution :
  ∃ x : ℕ, marbles_given x ∧
  (tyrone_initial - x = 3 * (eric_initial + x)) ∧
  x = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_redistribution_l1341_134160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pasture_rent_calculation_l1341_134129

/-- Represents a milkman's grazing data -/
structure MilkmanData where
  cows : ℕ
  months : ℕ

/-- Calculates the total cow-months for a list of MilkmanData -/
def totalCowMonths (data : List MilkmanData) : ℕ :=
  data.foldl (fun acc d => acc + d.cows * d.months) 0

/-- Calculates the total rent given the total cow-months and the rent per cow-month -/
def totalRent (totalCowMonths : ℕ) (rentPerCowMonth : ℚ) : ℚ :=
  (totalCowMonths : ℚ) * rentPerCowMonth

theorem pasture_rent_calculation (milkmenData : List MilkmanData) 
    (hA : milkmenData.length = 10)
    (hData : milkmenData = [
      ⟨36, 4⟩, ⟨18, 7⟩, ⟨45, 5⟩, ⟨32, 3⟩, ⟨20, 6⟩,
      ⟨25, 4⟩, ⟨15, 8⟩, ⟨30, 5⟩, ⟨40, 6⟩, ⟨50, 4⟩
    ])
    (hARent : (2240 : ℚ) / (36 * 4 : ℚ) = 15.56) :
    ∃ (rentApprox : ℚ), abs (rentApprox - 23668.56) < 0.01 ∧ 
    rentApprox = totalRent (totalCowMonths milkmenData) ((2240 : ℚ) / (36 * 4 : ℚ)) :=
  sorry

#eval totalCowMonths [
  ⟨36, 4⟩, ⟨18, 7⟩, ⟨45, 5⟩, ⟨32, 3⟩, ⟨20, 6⟩,
  ⟨25, 4⟩, ⟨15, 8⟩, ⟨30, 5⟩, ⟨40, 6⟩, ⟨50, 4⟩
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pasture_rent_calculation_l1341_134129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_routes_P_to_R_l1341_134174

/-- Represents a point in the graph --/
inductive Point : Type
| P : Point
| Q : Point
| R : Point
| S : Point
| T : Point

/-- Represents a direct path between two points --/
inductive DirectPath : Point → Point → Prop
| PQ : DirectPath Point.P Point.Q
| PS : DirectPath Point.P Point.S
| QR : DirectPath Point.Q Point.R
| QT : DirectPath Point.Q Point.T
| SR : DirectPath Point.S Point.R
| TR : DirectPath Point.T Point.R

/-- Represents a route from one point to another --/
def Route : Point → Point → Type :=
  λ start finish => List (Σ' (a b : Point), DirectPath a b)

/-- Count the number of routes between two points --/
def countRoutes (start finish : Point) : Nat :=
  sorry

/-- The main theorem: There are 3 different routes from P to R --/
theorem three_routes_P_to_R :
  countRoutes Point.P Point.R = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_routes_P_to_R_l1341_134174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shuttle_average_speed_l1341_134141

/-- Represents the speed of a space shuttle at a given altitude -/
structure ShuttleSpeed where
  altitude : ℝ
  speed_kms : ℝ

/-- Converts speed from km/s to km/h -/
noncomputable def speed_kmh (s : ShuttleSpeed) : ℝ :=
  s.speed_kms * 3600

/-- Calculates the average speed between two altitudes -/
noncomputable def average_speed (s1 s2 : ShuttleSpeed) : ℝ :=
  (speed_kmh s1 + speed_kmh s2) / 2

/-- Theorem: The average speed of the space shuttle between 300 km and 800 km altitudes is 23400 km/h -/
theorem shuttle_average_speed :
  let s1 : ShuttleSpeed := ⟨300, 7⟩
  let s2 : ShuttleSpeed := ⟨800, 6⟩
  average_speed s1 s2 = 23400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shuttle_average_speed_l1341_134141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_implication_not_necessary_condition_l1341_134184

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def subsequence (a : ℕ → ℕ → ℝ) (f : ℕ → ℕ) : ℕ → ℝ :=
  λ n ↦ a n (f n)

theorem geometric_sequence_implication :
  ∀ a : ℕ → ℝ,
  is_geometric_sequence a →
  (is_geometric_sequence (subsequence (λ _ ↦ a) (λ k ↦ 2*k - 1)) ∧
   is_geometric_sequence (subsequence (λ _ ↦ a) (λ k ↦ 2*k))) :=
by
  sorry

theorem not_necessary_condition :
  ∃ a : ℕ → ℝ,
  (is_geometric_sequence (subsequence (λ _ ↦ a) (λ k ↦ 2*k - 1)) ∧
   is_geometric_sequence (subsequence (λ _ ↦ a) (λ k ↦ 2*k))) ∧
  ¬(is_geometric_sequence a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_implication_not_necessary_condition_l1341_134184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_is_ten_l1341_134195

/-- The set S of integers from 1 to 100 -/
def S : Set ℕ := Finset.range 100

/-- A partition of S is a list of non-empty, pairwise disjoint subsets whose union is S -/
def IsPartitionOf (p : List (Finset ℕ)) (S : Set ℕ) : Prop :=
  (∀ s, s ∈ p → s.Nonempty) ∧
  (∀ s t, s ∈ p → t ∈ p → s ≠ t → Disjoint s t) ∧
  (↑(Finset.biUnion p.toFinset id) = S)

/-- The average of a non-empty finite set of natural numbers -/
noncomputable def average (s : Finset ℕ) : ℚ :=
  (s.sum (λ x => (x : ℚ))) / s.card

/-- The score of a partition is the average of the averages of its subsets -/
noncomputable def score (p : List (Finset ℕ)) : ℚ :=
  (p.map average).sum / p.length

/-- The minimum score theorem -/
theorem min_score_is_ten :
  ∀ p : List (Finset ℕ), IsPartitionOf p S → score p ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_is_ten_l1341_134195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_by_repeating_third_l1341_134143

theorem divide_by_repeating_third : 8 / (1/3 : ℚ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_by_repeating_third_l1341_134143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_OA_length_BC_l1341_134115

-- Define the circles in polar coordinates
noncomputable def circle1 (θ : Real) : Real := 4 * Real.cos θ
noncomputable def circle2 (θ : Real) : Real := 2 * Real.sin θ

-- Define the intersection point A
noncomputable def θ_A : Real := Real.arctan 2

-- Theorem for the slope of OA
theorem slope_OA : 
  Real.tan θ_A = 2 := by sorry

-- Define points B and C
noncomputable def ρ_B : Real := circle1 (θ_A - Real.pi/2)
noncomputable def ρ_C : Real := circle2 (θ_A + Real.pi/2)

-- Theorem for the length of BC
theorem length_BC : 
  ρ_B + ρ_C = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_OA_length_BC_l1341_134115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1341_134138

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 4

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) : ℝ := 
  |x + y - 4| / Real.sqrt 2

-- Theorem statement
theorem max_distance_curve_to_line :
  ∃ (max_dist : ℝ), 
    (∀ (x y : ℝ), curve_C x y → distance_point_to_line x y ≤ max_dist) ∧
    (∃ (x y : ℝ), curve_C x y ∧ distance_point_to_line x y = max_dist) ∧
    max_dist = 3 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1341_134138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_people_motorcycle_problem_l1341_134172

/-- Represents a person's position and mode of transport -/
structure PersonState where
  position : ℝ
  onMotorcycle : Bool

/-- Represents the state of the entire group -/
structure GroupState where
  person1 : PersonState
  person2 : PersonState
  person3 : PersonState
  time : ℝ

/-- Checks if a given group state is valid according to the problem constraints -/
def isValidState (s : GroupState) : Prop :=
  s.person1.position ≥ 0 ∧ s.person1.position ≤ 60 ∧
  s.person2.position ≥ 0 ∧ s.person2.position ≤ 60 ∧
  s.person3.position ≥ 0 ∧ s.person3.position ≤ 60 ∧
  s.time ≥ 0 ∧ s.time ≤ 3 ∧
  (Bool.toNat s.person1.onMotorcycle + Bool.toNat s.person2.onMotorcycle + Bool.toNat s.person3.onMotorcycle) ≤ 2

/-- Represents a valid move in the problem -/
def ValidMove (start finish : GroupState) : Prop :=
  isValidState start ∧ isValidState finish ∧
  finish.time > start.time ∧
  (finish.person1.position - start.person1.position) / (finish.time - start.time) ≤ (if start.person1.onMotorcycle then 50 else 5) ∧
  (finish.person2.position - start.person2.position) / (finish.time - start.time) ≤ (if start.person2.onMotorcycle then 50 else 5) ∧
  (finish.person3.position - start.person3.position) / (finish.time - start.time) ≤ (if start.person3.onMotorcycle then 50 else 5)

/-- The theorem to be proved -/
theorem three_people_motorcycle_problem :
  ∃ (n : ℕ) (states : Fin (n + 1) → GroupState),
    states 0 = {
      person1 := { position := 0, onMotorcycle := false },
      person2 := { position := 0, onMotorcycle := false },
      person3 := { position := 0, onMotorcycle := false },
      time := 0
    } ∧
    (∀ i : Fin n, ValidMove (states i) (states (i + 1))) ∧
    (states n).person1.position = 60 ∧
    (states n).person2.position = 60 ∧
    (states n).person3.position = 60 ∧
    (states n).time ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_people_motorcycle_problem_l1341_134172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_length_l1341_134148

open Real

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  side_ab : dist A B > 0
  side_bc : dist B C > 0
  side_ca : dist C A > 0

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define a semicircle
def Semicircle (X Y : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist P ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) = r ∧
               (P.1 - X.1) * (Y.1 - X.1) + (P.2 - X.2) * (Y.2 - X.2) ≥ 0}

-- Define the inscribed semicircle
def InscribedSemicircle (A B C X Y D : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), Semicircle X Y r ⊆ {P : ℝ × ℝ | Triangle A B C} ∧
  D ∈ Semicircle X Y r ∧ 
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ D = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2)) ∧
  ∃ s, 0 ≤ s ∧ s ≤ 1 ∧ X = (A.1 + s * (B.1 - A.1), A.2 + s * (B.2 - A.2)) ∧
  ∃ u, 0 ≤ u ∧ u ≤ 1 ∧ Y = (A.1 + u * (C.1 - A.1), A.2 + u * (C.2 - A.2))

-- Define the midpoint
def Midpoint (O X Y : ℝ × ℝ) : Prop :=
  O = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem inscribed_semicircle_length (A B C X Y D O : ℝ × ℝ) :
  Triangle A B C →
  RightAngle B A C →
  InscribedSemicircle A B C X Y D →
  Midpoint O X Y →
  dist A B = 3 →
  dist A C = 4 →
  dist A X = 9/4 →
  dist A O = 39/32 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_length_l1341_134148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_increase_theorem_l1341_134104

theorem circle_increase_theorem (r : ℝ) (h : r > 0) :
  let new_radius := 3 * r
  let area_increase_percent := (π * new_radius^2 - π * r^2) / (π * r^2) * 100
  let circumference_increase_percent := (2 * π * new_radius - 2 * π * r) / (2 * π * r) * 100
  area_increase_percent = 800 ∧ circumference_increase_percent = 200 := by
  -- Introduce the local definitions
  let new_radius := 3 * r
  let area_increase_percent := (π * new_radius^2 - π * r^2) / (π * r^2) * 100
  let circumference_increase_percent := (2 * π * new_radius - 2 * π * r) / (2 * π * r) * 100

  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_increase_theorem_l1341_134104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_remainders_l1341_134185

/-- The count of positive three-digit integers with specific remainders -/
theorem three_digit_integers_with_remainders :
  let S : Finset ℕ := Finset.filter (fun n => 
    100 ≤ n ∧ n < 1000 ∧ 
    n % 5 = 3 ∧ 
    n % 7 = 6 ∧ 
    n % 10 = 9) (Finset.range 1000)
  S.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_remainders_l1341_134185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1341_134168

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi / 2) * (Real.sqrt 3 * Real.sin x + Real.cos x)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ y, ∃ x, f x = y) ↔ (∀ y, -1/2 ≤ y ∧ y ≤ 3/2) ∧
  (∀ A B C a b c : ℝ,
    f A = 1 →
    a = Real.sqrt 3 →
    b + c = 3 →
    a = (b^2 + c^2 - 2*b*c*Real.cos A).sqrt →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1341_134168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_neg_reals_g_min_value_on_interval_l1341_134109

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 + 2 / (2^x - 1)

-- Part 1: g is decreasing on (-∞, 0)
theorem g_decreasing_on_neg_reals :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → g x₁ > g x₂ := by
  sorry

-- Part 2: Minimum value of g on (-∞, -1] is -3
theorem g_min_value_on_interval :
  ∀ x : ℝ, x ≤ -1 → g x ≥ -3 ∧ g (-1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_neg_reals_g_min_value_on_interval_l1341_134109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_cube_roots_l1341_134127

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 7 → x = Real.sqrt 7 ∨ x = -Real.sqrt 7) ∧ 
  (∀ y : ℝ, y^3 = -8/27 → y = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_cube_roots_l1341_134127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_function_l1341_134176

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

-- State the theorem
theorem inverse_equals_function (k : ℝ) :
  k ≠ -9/4 →
  (∀ x : ℝ, Function.invFun (g k) x = g k x) ↔ 
  (k < -9/4 ∨ k > -9/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_function_l1341_134176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_totient_congruence_l1341_134121

theorem prime_sum_totient_congruence (p n : ℕ) (hp : Nat.Prime p) (hn : 1 < n ∧ n ≤ p) :
  (Nat.totient ((n^p - 1) / (n - 1))) % p = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_totient_congruence_l1341_134121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_power_sum_l1341_134165

theorem sin_cos_power_sum (x : ℝ) :
  Real.sin x + Real.cos x = -1 → Real.sin x ^ 2005 + Real.cos x ^ 2005 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_power_sum_l1341_134165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l1341_134164

/-- The time taken for a, b, and c together to finish the task -/
noncomputable def time_abc : ℝ := 5

/-- The time taken for a alone to finish the task -/
noncomputable def time_a : ℝ := 9

/-- The time taken for a and c together to finish the task -/
noncomputable def time_ac : ℝ := 7

/-- The rate at which a, b, and c work together -/
noncomputable def rate_abc : ℝ := 1 / time_abc

/-- The rate at which a works alone -/
noncomputable def rate_a : ℝ := 1 / time_a

/-- The rate at which a and c work together -/
noncomputable def rate_ac : ℝ := 1 / time_ac

/-- The time taken for b alone to finish the task -/
noncomputable def time_b : ℝ := 315 / 26

theorem b_alone_time : time_b = 315 / 26 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l1341_134164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l1341_134166

/-- The radius of the inscribed sphere in a regular quadrilateral pyramid -/
noncomputable def inscribedSphereRadius (a b : ℝ) : ℝ :=
  (a * Real.sqrt (2 * (b^2 - a^2/2))) / (2 * (1 + Real.sqrt (4*b^2 - a^2)))

/-- Theorem: The radius of the inscribed sphere in a regular quadrilateral pyramid
    with base side a and slant edge b is equal to the given formula -/
theorem inscribed_sphere_radius_formula (a b : ℝ) (ha : a > 0) (hb : b > a/Real.sqrt 2) :
  inscribedSphereRadius a b = (a * Real.sqrt (2 * (b^2 - a^2/2))) / (2 * (1 + Real.sqrt (4*b^2 - a^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l1341_134166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l1341_134175

-- Define the function a as noncomputable
noncomputable def a (m : ℝ) (k : ℝ) : ℝ := (2 * k + m) ^ k

-- State the theorem
theorem solve_for_m :
  ∀ m : ℝ, a m (a m (a m 0)) = 343 → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l1341_134175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_order_l1341_134151

/-- Represents the possible positions of coins -/
inductive Position : Type
  | Top
  | SecondFromTop
  | ThirdFromTop
  | FourthFromTop
  | FifthFromTop
  | Bottom

/-- Represents a coin -/
structure Coin where
  label : Char
  position : Position

/-- Defines the relation "is above" between two coins -/
def isAbove (c1 c2 : Coin) : Prop :=
  match c1.position, c2.position with
  | Position.Top, _ => true
  | Position.SecondFromTop, Position.ThirdFromTop => true
  | Position.SecondFromTop, Position.FourthFromTop => true
  | Position.SecondFromTop, Position.FifthFromTop => true
  | Position.SecondFromTop, Position.Bottom => true
  | Position.ThirdFromTop, Position.FourthFromTop => true
  | Position.ThirdFromTop, Position.FifthFromTop => true
  | Position.ThirdFromTop, Position.Bottom => true
  | Position.FourthFromTop, Position.FifthFromTop => true
  | Position.FourthFromTop, Position.Bottom => true
  | Position.FifthFromTop, Position.Bottom => true
  | _, _ => false

/-- The set of all coins -/
def coins : List Coin :=
  [ Coin.mk 'F' Position.Top,
    Coin.mk 'B' Position.SecondFromTop,
    Coin.mk 'D' Position.ThirdFromTop,
    Coin.mk 'C' Position.FourthFromTop,
    Coin.mk 'A' Position.FifthFromTop,
    Coin.mk 'E' Position.Bottom ]

/-- Function to get a coin by its label -/
def getCoin (label : Char) : Coin :=
  match coins.find? (fun c => c.label == label) with
  | some coin => coin
  | none => Coin.mk label Position.Bottom  -- Default case, should not occur

/-- The theorem stating the correct order of coins -/
theorem coin_order :
  (∀ x ∈ coins, x.label ≠ 'F' → isAbove (getCoin 'F') x) ∧
  (isAbove (getCoin 'F') (getCoin 'B') ∧ isAbove (getCoin 'B') (getCoin 'A') ∧ isAbove (getCoin 'B') (getCoin 'C') ∧ isAbove (getCoin 'B') (getCoin 'E')) ∧
  (isAbove (getCoin 'F') (getCoin 'D') ∧ isAbove (getCoin 'D') (getCoin 'A') ∧ isAbove (getCoin 'D') (getCoin 'C') ∧ ¬isAbove (getCoin 'D') (getCoin 'E') ∧ ¬isAbove (getCoin 'D') (getCoin 'B')) ∧
  (isAbove (getCoin 'A') (getCoin 'E') ∧ ∀ x ∈ coins, x.label ≠ 'E' → isAbove x (getCoin 'A')) ∧
  (isAbove (getCoin 'C') (getCoin 'E')) →
  (getCoin 'F').position = Position.Top ∧
  (getCoin 'B').position = Position.SecondFromTop ∧
  (getCoin 'D').position = Position.ThirdFromTop ∧
  (getCoin 'C').position = Position.FourthFromTop ∧
  (getCoin 'A').position = Position.FifthFromTop ∧
  (getCoin 'E').position = Position.Bottom :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_order_l1341_134151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l1341_134107

noncomputable def f (x : ℝ) := (Real.log (5 - x^2)) / Real.sqrt (2*x - 3)

theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ (3/2 ≤ x ∧ x < Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l1341_134107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1341_134102

/-- Given an angle α whose terminal side passes through point P(-x, -6),
    prove that if cos α = 4/5, then x = -8. -/
theorem angle_terminal_side (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (-x, -6) ∧ P.1 = -x * Real.cos α ∧ P.2 = -x * Real.sin α) →
  Real.cos α = 4/5 →
  x = -8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1341_134102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_value_l1341_134134

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * Real.pi * x - Real.pi / 6)

theorem periodic_sine_value (ω : ℝ) (h1 : ω > 0) (h2 : (2 : ℝ) / (ω * Real.pi) = 1 / 5) :
  f ω (1 / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_value_l1341_134134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_4pi_l1341_134177

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + 9*y^2 + 18*y + 1 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 4 * Real.pi

/-- Theorem stating that the area of the ellipse is 4π -/
theorem ellipse_area_is_4pi :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x - 2)^2 / a^2 + (y + 1)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_is_4pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_4pi_l1341_134177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_implies_midpoint_l1341_134170

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point D on side AB
variable (D : ℝ × ℝ)

-- Define point E on segment CD
variable (E : ℝ × ℝ)

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define the condition that D lies on AB
def D_on_AB (A B D : ℝ × ℝ) : Prop := sorry

-- Define the condition that E lies on CD
def E_on_CD (C D E : ℝ × ℝ) : Prop := sorry

-- Define the condition about the areas
def area_condition (A B C D E : ℝ × ℝ) : Prop :=
  triangleArea A C E + triangleArea B D E = (1/2) * triangleArea A B C

-- Define what it means for a point to be the midpoint of a segment
def is_midpoint (M P Q : ℝ × ℝ) : Prop := sorry

-- The theorem to be proved
theorem area_condition_implies_midpoint
  (h1 : D_on_AB A B D)
  (h2 : E_on_CD C D E)
  (h3 : area_condition A B C D E) :
  is_midpoint D A B ∨ is_midpoint E C D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_implies_midpoint_l1341_134170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoey_finishes_on_sunday_l1341_134196

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  deriving Repr

/-- Calculates the sum of the first n natural numbers -/
def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Advances a day of the week by one day -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Advances a day of the week by a given number of days -/
def advanceDay (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDay (nextDay start) n

theorem zoey_finishes_on_sunday :
  advanceDay DayOfWeek.Sunday (sumFirstN 20) = DayOfWeek.Sunday := by
  sorry

#eval advanceDay DayOfWeek.Sunday (sumFirstN 20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoey_finishes_on_sunday_l1341_134196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_usage_first_third_l1341_134155

/-- Proves the amount of fuel that can be used in the first third of a trip given the total fuel and usage in other parts of the trip. -/
theorem fuel_usage_first_third 
  (total_fuel : ℝ) 
  (h1 : total_fuel = 60) 
  (h2 : total_fuel > 0) 
  (second_third_usage : ℝ) 
  (h3 : second_third_usage = total_fuel / 3) 
  (final_third_usage : ℝ) 
  (h4 : final_third_usage = second_third_usage / 2) : 
  total_fuel - (second_third_usage + final_third_usage) = 30 := by
  sorry

#check fuel_usage_first_third

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_usage_first_third_l1341_134155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l1341_134120

theorem complex_magnitude_problem :
  Complex.abs (2 + Complex.I^2 + 2*Complex.I^3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l1341_134120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_equals_10100_l1341_134101

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => a n + (2 * a n) / (n + 1)

theorem a_100_equals_10100 : a 100 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_equals_10100_l1341_134101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_length_equality_l1341_134117

/-- Proves that the length of the platform is equal to the length of the train -/
theorem train_platform_length_equality 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (train_length : ℝ) 
  (h1 : train_speed = 126)
  (h2 : crossing_time = 1)
  (h3 : train_length = 1050) : 
  ∃ (platform_length : ℝ), platform_length = train_length := by
  -- Convert speed from km/hr to m/s
  let speed_ms := train_speed * 1000 / 3600
  
  -- Calculate total distance covered
  let total_distance := speed_ms * (crossing_time * 60)
  
  -- Calculate platform length
  let platform_length := total_distance - train_length
  
  -- Prove that platform_length equals train_length
  exists platform_length
  sorry -- Skip the actual proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_length_equality_l1341_134117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_calculation_l1341_134194

-- Define the cuboid properties
def length : ℝ := 10
def breadth : ℝ := 8
def surface_area : ℝ := 480

-- Define the surface area formula
def surface_area_formula (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

-- Theorem statement
theorem cuboid_height_calculation :
  ∃ h : ℝ, surface_area_formula length breadth h = surface_area ∧ 
  |h - 8.89| < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_calculation_l1341_134194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_equilateral_triangle_revolution_l1341_134197

/-- The surface area of a solid of revolution generated by rotating an equilateral triangle -/
noncomputable def surface_area_triangle_revolution (a : ℝ) : ℝ :=
  (11/2) * Real.pi * a^2

/-- Theorem stating the surface area of the solid of revolution -/
theorem surface_area_equilateral_triangle_revolution (a : ℝ) (h : a > 0) :
  let triangle_side_length := a
  let axis_distance := (3/2) * a
  surface_area_triangle_revolution a = (11/2) * Real.pi * a^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_equilateral_triangle_revolution_l1341_134197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l1341_134144

theorem complex_magnitude_proof : Complex.abs (1 - (5/4)*Complex.I) = Real.sqrt 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l1341_134144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_implies_m_values_l1341_134189

/-- The distance from a point (x, y) to the line ax + by + c = 0 is given by
    |ax + by + c| / √(a² + b²) --/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Given two points A(3, 2) and B(-1, 4), and a line mx + y + 3 = 0,
    prove that if the distances from A and B to the line are equal,
    then m = 1/2 or m = -6 --/
theorem equal_distance_implies_m_values (m : ℝ) :
  distance_point_to_line 3 2 m 1 3 = distance_point_to_line (-1) 4 m 1 3 →
  m = 1/2 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_implies_m_values_l1341_134189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_E_to_C_l1341_134182

/-- A chessboard with points on its border -/
structure Chessboard where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Distance from E to C on a specific chessboard -/
theorem distance_E_to_C (cb : Chessboard) 
  (h1 : distance cb.A cb.B = 30)
  (h2 : distance cb.B cb.C = 80)
  (h3 : distance cb.C cb.D = 236)
  (h4 : distance cb.D cb.E = 86)
  (h5 : distance cb.E cb.A = 40) :
  distance cb.E cb.C = 63.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_E_to_C_l1341_134182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_three_solutions_l1341_134188

theorem equation_three_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x^2 * (x - 1) * (x - 2) = 0) ∧ (Finset.card S = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_three_solutions_l1341_134188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_nineteen_l1341_134139

-- Define the expression
noncomputable def expression : ℝ := 
  (2 * Real.sqrt 2) ^ (2/3 : ℝ) * (0.1⁻¹) - Real.log 2 - Real.log 5

-- State the theorem
theorem expression_equals_nineteen : expression = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_nineteen_l1341_134139
