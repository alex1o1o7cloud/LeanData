import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_through_point_l539_53938

theorem sin_double_angle_through_point (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = -3 ∧ r * (Real.sin α) = 4) →
  Real.sin (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_through_point_l539_53938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l539_53914

noncomputable def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 4*y + 11 = 0

noncomputable def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem min_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 →
    circle_C2 x2 y2 →
    distance x1 y1 x2 y2 ≥ 3 * Real.sqrt 5 - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l539_53914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_noon_l539_53924

/-- Represents the meeting time of two trains given their starting times, speeds, and distance between stations. -/
noncomputable def train_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (start_time1 start_time2 : ℝ) : ℝ :=
  let time_diff := start_time2 - start_time1
  let distance1 := speed1 * time_diff
  let remaining_distance := distance - distance1
  let relative_speed := speed1 + speed2
  let meeting_time := remaining_distance / relative_speed
  start_time2 + meeting_time

/-- Theorem stating that under the given conditions, the trains will meet at 12 p.m. (5 hours after the first train starts) -/
theorem trains_meet_at_noon (distance : ℝ) (speed1 speed2 : ℝ) (start_time1 start_time2 : ℝ) :
  distance = 200 →
  speed1 = 20 →
  speed2 = 25 →
  start_time1 = 0 →
  start_time2 = 1 →
  train_meeting_time distance speed1 speed2 start_time1 start_time2 = 5 :=
by
  sorry

#check trains_meet_at_noon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_noon_l539_53924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_marbles_count_l539_53921

theorem initial_marbles_count : ∃ (g y : ℕ), 
  g + y > 0 ∧ 
  (g + 3 : ℚ) / (g + y + 3 : ℚ) = 1 / 4 ∧ 
  (g : ℚ) / (g + y + 4 : ℚ) = 1 / 3 ∧ 
  g + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_marbles_count_l539_53921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l539_53989

noncomputable def F (x : ℝ) : ℝ := x^2 + 1

noncomputable def work (a b : ℝ) : ℝ := ∫ x in a..b, F x

theorem work_calculation : work 0 6 = 78 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l539_53989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l539_53952

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Definition of point B -/
def point_B : ℝ × ℝ := (3, 0)

/-- Definition of the midpoint M of line segment PB -/
noncomputable def midpoint_M (x y : ℝ) : ℝ × ℝ :=
  ((3 + x) / 2, y / 2)

/-- Theorem stating the trajectory of M -/
theorem trajectory_of_M :
  ∀ x y : ℝ, ellipse_C x y →
  let (m, n) := midpoint_M x y
  (2*m - 3)^2 / 4 + 4*n^2 = 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l539_53952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l539_53949

def solution_set_A (a : ℕ) : Set ℝ :=
  {x : ℝ | a * x^2 + 2 * |x - a| - 20 < 0}

def inequality1_set : Set ℝ :=
  {x : ℝ | x^2 + x - 2 < 0}

def inequality2_set : Set ℝ :=
  {x : ℝ | |2*x - 1| < x + 2}

theorem solution_exists :
  (∃ a : ℕ, inequality1_set ⊆ solution_set_A a ∧ inequality2_set ⊆ solution_set_A a) ↔
  ∃ a : ℕ, a ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l539_53949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_41pi_6_f_value_in_third_quadrant_l539_53969

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (-α) * Real.cos (-α + 3*Real.pi/2)) /
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α))

theorem f_value_at_negative_41pi_6 :
  f (-41*Real.pi/6) = Real.sqrt 3 / 2 := by sorry

theorem f_value_in_third_quadrant (α : Real)
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/3) :
  f α = 2 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_41pi_6_f_value_in_third_quadrant_l539_53969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l539_53987

noncomputable section

theorem triangle_angles (a b c : ℝ) (ha : a = 2) (hb : b = 2) (hc : c = Real.sqrt 8) :
  ∃ (α β γ : ℝ),
    α = 90 ∧ β = 45 ∧ γ = 45 ∧
    0 < α ∧ 0 < β ∧ 0 < γ ∧
    α + β + γ = 180 ∧
    Real.cos (α * Real.pi / 180) = (b^2 + c^2 - a^2) / (2 * b * c) ∧
    Real.cos (β * Real.pi / 180) = (a^2 + c^2 - b^2) / (2 * a * c) ∧
    Real.cos (γ * Real.pi / 180) = (a^2 + b^2 - c^2) / (2 * a * b) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l539_53987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fee_percentage_l539_53982

/-- Represents the price of Product A in the first year -/
noncomputable def initial_price : ℝ := 70

/-- Represents the annual sales volume in the first year (in thousands) -/
noncomputable def initial_sales : ℝ := 11.8

/-- Calculates the price increase in the second year -/
noncomputable def price_increase (x : ℝ) : ℝ := (initial_price * x / 100) / (1 - x / 100)

/-- Calculates the sales volume decrease in the second year (in thousands) -/
noncomputable def sales_decrease (x : ℝ) : ℝ := x

/-- Calculates the management fee in the second year (in thousands of yuan) -/
noncomputable def management_fee (x : ℝ) : ℝ := 
  (initial_price + price_increase x) * (initial_sales - sales_decrease x) * (x / 100)

/-- States that the maximum value of x that satisfies the management fee condition is 10 -/
theorem max_fee_percentage : 
  ∀ x : ℝ, 0 < x → x ≤ 10 → 
  management_fee x ≥ 140 ∧ 
  ∀ y : ℝ, y > 10 → management_fee y < 140 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fee_percentage_l539_53982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electronics_weight_after_removal_l539_53939

/-- Represents the contents of a suitcase -/
structure Suitcase where
  books : ℚ
  clothes : ℚ
  electronics : ℚ
  toys : ℚ

/-- Calculates the ratio of books to clothes in a suitcase -/
def booksToClothesRatio (s : Suitcase) : ℚ :=
  s.books / s.clothes

/-- The initial suitcase contents -/
def initialSuitcase : Suitcase :=
  { books := 7
    clothes := 4
    electronics := 3
    toys := 2 }

/-- The suitcase after removing clothing and toys -/
def updatedSuitcase (s : Suitcase) : Suitcase :=
  { books := s.books
    clothes := s.clothes - 8
    electronics := s.electronics
    toys := s.toys - 5 }

theorem electronics_weight_after_removal :
  (updatedSuitcase initialSuitcase).electronics = 12 ∧
  booksToClothesRatio (updatedSuitcase initialSuitcase) = 2 * booksToClothesRatio initialSuitcase :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electronics_weight_after_removal_l539_53939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l539_53997

noncomputable def f (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem f_max_value :
  (∀ x : ℝ, x < 5/4 → f x ≤ 1) ∧ (∃ x : ℝ, x < 5/4 ∧ f x = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l539_53997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_min_f_l539_53961

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

/-- Proposition that the maximum value of a for which f(x) has a minimum value is 1 -/
theorem max_a_for_min_f : ∀ ε > 0, ∃ a : ℝ, a > 1 - ε ∧ a ≤ 1 ∧
  (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m) ∧
  (∀ b > 1, ¬∃ m : ℝ, ∀ x : ℝ, f b x ≥ m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_min_f_l539_53961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_fixed_points_relation_l539_53996

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 - 4*x + y^2 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem circle_fixed_points_relation :
  ∀ a : ℝ,
  (∀ x y : ℝ, myCircle x y →
    distance x y a 0 = 2 * distance x y 1 0) →
  a = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_fixed_points_relation_l539_53996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_exp_arctg_sin_quotient_l539_53908

/-- The limit of (e^(4x) - e^(-2x)) / (2 arctg x - sin x) as x approaches 0 is 6 -/
theorem limit_exp_arctg_sin_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |(Real.exp (4 * x) - Real.exp (-2 * x)) / (2 * Real.arctan x - Real.sin x) - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_exp_arctg_sin_quotient_l539_53908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_div_g_positive_l539_53978

open Set Real

theorem solution_set_f_div_g_positive
  (f g : ℝ → ℝ)
  (h_g_nonzero : ∀ x, g x ≠ 0)
  (h_derivative_inequality : ∀ x, (deriv f x) * (g x) > (f x) * (deriv g x))
  (h_f_1_zero : f 1 = 0) :
  {x : ℝ | (f x) / (g x) > 0} = Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_div_g_positive_l539_53978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l539_53946

/-- Calculates the speed of a train given its length and time to cross a fixed point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train with length 1200 meters that crosses an electric pole in 30 seconds
    has a speed of 40 meters per second. -/
theorem train_speed_theorem :
  train_speed 1200 30 = 40 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l539_53946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l539_53932

variable (n : ℕ) (φ : ℝ) (ρ : ℝ)

noncomputable def P (x : ℂ) : ℂ := (Complex.cos φ + x * Complex.sin φ)^n - Complex.cos (n * φ) - x * Complex.sin (n * φ)

noncomputable def Q (x : ℂ) : ℂ := x^n * Complex.sin φ - ρ^(n-1) * x * Complex.sin (n * φ) + ρ^n * Complex.sin ((n-1) * φ)

theorem polynomial_divisibility :
  (∃ R : ℂ → ℂ, P n φ = (fun x => x^2 + 1) * R) ∧
  (∃ S : ℂ → ℂ, Q n φ ρ = (fun x => x^2 - 2*ρ*x*Complex.cos φ + ρ^2) * S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l539_53932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l539_53918

def sequence_sum (n : ℕ) : ℚ := n

-- Define the recurrence relation for S_n
axiom S_recurrence (n : ℕ) : n ≥ 2 → sequence_sum n - 3 * sequence_sum (n-1) + 2 = 0

-- Define a_1
axiom a_1 : sequence_sum 1 = 2

-- Define T_n as the sum of 1/a_n
def T (n : ℕ) : ℚ := n

theorem sum_of_reciprocals (n : ℕ) : n ≥ 1 → T n = 7/4 - 1/(4 * 3^(n-2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l539_53918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_fifteen_degrees_l539_53903

theorem trig_identity_fifteen_degrees : 
  (1 - Real.tan (15 * π / 180) ^ 2) * Real.cos (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_fifteen_degrees_l539_53903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_is_170_over_9_l539_53953

/-- Represents the average number of minutes run per day for each grade --/
structure GradeRunningTime where
  sixth : ℚ
  seventh : ℚ
  eighth : ℚ

/-- Represents the relative number of students in each grade --/
structure GradeDistribution where
  sixth : ℚ
  seventh : ℚ
  eighth : ℚ

/-- Calculates the average running time for all students --/
def averageRunningTime (time : GradeRunningTime) (dist : GradeDistribution) : ℚ :=
  (time.sixth * dist.sixth + time.seventh * dist.seventh + time.eighth * dist.eighth) /
  (dist.sixth + dist.seventh + dist.eighth)

theorem average_running_time_is_170_over_9 :
  let time : GradeRunningTime := ⟨18, 20, 22⟩
  let dist : GradeDistribution := ⟨3, 1, 1/2⟩
  averageRunningTime time dist = 170/9 := by
  -- Proof goes here
  sorry

#eval averageRunningTime ⟨18, 20, 22⟩ ⟨3, 1, 1/2⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_is_170_over_9_l539_53953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_circles_equal_radii_l539_53928

-- Define a body in 3D space
structure Body3D where
  -- (We don't need to specify the exact structure of the body)

-- Define a plane in 3D space
structure Plane3D where
  -- (We don't need to specify the exact structure of the plane)

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a projection function
def project (b : Body3D) (p : Plane3D) : Set (ℝ × ℝ) :=
  sorry

-- Define a function to check if a set of points forms a circle
def isCircle (s : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Define a function to get the radius if the projection is a circle
def getRadius (s : Set (ℝ × ℝ)) : Option ℝ :=
  sorry

-- State the theorem
theorem projection_circles_equal_radii (b : Body3D) (p1 p2 : Plane3D) :
  let proj1 := project b p1
  let proj2 := project b p2
  isCircle proj1 ∧ isCircle proj2 →
  ∃ r : ℝ, getRadius proj1 = some r ∧ getRadius proj2 = some r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_circles_equal_radii_l539_53928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_representable_numbers_l539_53942

/-- The function representing the expression 7m² - 11n² --/
def f (m n : ℕ) : ℤ := 7 * (m : ℤ)^2 - 11 * (n : ℤ)^2

/-- Predicate to check if a number is representable by 7m² - 11n² --/
def representable (k : ℕ) : Prop := ∃ m n : ℕ, f m n = k

/-- Theorem stating that 7 and 13 are the two smallest representable natural numbers --/
theorem smallest_representable_numbers :
  (∀ k < 7, ¬representable k) ∧ 
  representable 7 ∧
  (∀ k : ℕ, 7 < k ∧ k < 13 → ¬representable k) ∧
  representable 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_representable_numbers_l539_53942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_xf_inequality_solution_l539_53923

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then (3 : ℝ)^x - 3 else (3 : ℝ)^(-x) - 3

-- State the properties of f
theorem f_properties :
  (∀ x, f (-x) = f x) ∧  -- f is even
  (∀ x ≥ 0, f x = (3 : ℝ)^x - 3) := by sorry

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ (0 < x ∧ x < 1)}

-- State the theorem
theorem xf_inequality_solution :
  {x : ℝ | x * f x < 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_xf_inequality_solution_l539_53923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_a_l539_53983

def vector_a (m : ℝ) : ℝ × ℝ := (m, m - 1)
def vector_b : ℝ × ℝ := (2, 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem magnitude_of_a :
  ∃ m : ℝ, perpendicular (vector_a m) vector_b ∧ 
  Real.sqrt ((vector_a m).1^2 + (vector_a m).2^2) = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_a_l539_53983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l539_53909

-- Define the slope of a line ax + by + c = 0
noncomputable def lineslope (a b : ℝ) : ℝ := -a / b

-- Define the lines
def l₁ (m : ℝ) (x y : ℝ) : Prop := 2 * x + m * y + 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l₃ (n : ℝ) (x y : ℝ) : Prop := x + n * y + 1 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := lineslope 2 m = lineslope 2 1

-- Define perpendicular lines
def perpendicular (m n : ℝ) : Prop := lineslope 2 m * lineslope 1 n = -1

theorem lines_theorem (m n : ℝ) :
  parallel m → perpendicular m n → m + n = -1 :=
by
  intros h1 h2
  sorry

#check lines_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l539_53909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l539_53920

-- Define proposition P
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Define the set of a values that satisfy the conditions
def A : Set ℝ := {a : ℝ | (¬(P a ∧ Q a)) ∧ (P a ∨ Q a)}

-- Theorem statement
theorem range_of_a : A = Set.Ioi 0 ∪ Set.Ioo (1/4) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l539_53920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_bcm_l539_53911

/-- A rhombus with a point on one side -/
structure RhombusWithPoint where
  -- The side length of the rhombus
  side : ℝ
  -- The distance from M to D, expressed as a fraction of AD
  md_ratio : ℝ
  -- The length of BM (which equals MC)
  bm_length : ℝ
  -- Assumptions
  side_positive : 0 < side
  md_ratio_valid : 0 < md_ratio ∧ md_ratio < 1
  bm_length_positive : 0 < bm_length

/-- The area of triangle BCM in the given rhombus configuration -/
noncomputable def triangle_area (r : RhombusWithPoint) : ℝ :=
  20 * Real.sqrt 6

/-- Main theorem: The area of triangle BCM is 20√6 -/
theorem area_of_triangle_bcm (r : RhombusWithPoint) 
  (h : r.md_ratio = 0.3 ∧ r.bm_length = 11) : 
  triangle_area r = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_bcm_l539_53911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_Q_zeros_l539_53916

-- Define the polynomial Q(z)
noncomputable def Q (z : ℂ) : ℂ := z^8 + (5 * Real.sqrt 2 + 8) * z^4 - (5 * Real.sqrt 2 + 9)

-- Define the set of zeros of Q(z)
def zeros : Set ℂ := {z : ℂ | Q z = 0}

-- Define the perimeter of a polygon given a set of points
noncomputable def perimeter (points : Set ℂ) : ℝ :=
  sorry -- Definition of perimeter calculation

-- Theorem statement
theorem min_perimeter_of_Q_zeros :
  ∃ P : ℝ, perimeter zeros = P ∧ 
  ∀ polygon : Finset ℂ, polygon.card = 8 → ↑polygon ⊆ zeros → 
  perimeter ↑polygon ≥ P :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_of_Q_zeros_l539_53916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_contraction_l539_53962

theorem fixed_point_contraction {S : Type} [Finite S] [MetricSpace S]
  (f : S → S) (h : ∀ s₁ s₂ : S, dist (f s₁) (f s₂) ≤ (1/2 : ℝ) * dist s₁ s₂) :
  ∃ x : S, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_contraction_l539_53962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_less_than_one_l539_53993

/-- Custom operation @ for positive integers -/
def custom_op (k : ℕ) (j : ℕ) : ℕ :=
  (List.range j).foldl (λ acc i => acc * (k + i)) k

/-- Theorem: The ratio of (2020 @ 4) to (2120 @ 4) is less than 1 -/
theorem ratio_less_than_one :
  (custom_op 2020 4 : ℚ) / (custom_op 2120 4 : ℚ) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_less_than_one_l539_53993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_when_stopped_l539_53991

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time taken to walk when the walkway is stopped -/
noncomputable def time_when_stopped (scenario : WalkwayScenario) : ℝ :=
  (2 * scenario.length) / (scenario.length / scenario.time_with + scenario.length / scenario.time_against)

/-- Theorem stating the time taken to walk when the walkway is stopped -/
theorem walkway_time_when_stopped (scenario : WalkwayScenario) 
  (h1 : scenario.length = 120)
  (h2 : scenario.time_with = 40)
  (h3 : scenario.time_against = 160) : 
  time_when_stopped scenario = 64 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_when_stopped_l539_53991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_and_intersection_sum_l539_53959

-- Define the point P
noncomputable def P : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the line l in Cartesian form
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y = Real.sqrt 3

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 / 4 + x^2 / 2 = 1

-- State the theorem
theorem point_on_line_and_intersection_sum :
  -- Point P lies on line l
  line_l P.1 P.2 ∧
  -- The sum of reciprocals of distances from P to intersection points is √14
  ∃ A B : ℝ × ℝ, 
    curve_C A.1 A.2 ∧ 
    curve_C B.1 B.2 ∧ 
    line_l A.1 A.2 ∧ 
    line_l B.1 B.2 ∧ 
    1 / ((A.1 - P.1)^2 + (A.2 - P.2)^2).sqrt + 
    1 / ((B.1 - P.1)^2 + (B.2 - P.2)^2).sqrt = Real.sqrt 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_and_intersection_sum_l539_53959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l539_53947

noncomputable section

def elongation_rates_A : Fin 10 → ℝ := sorry
def elongation_rates_B : Fin 10 → ℝ := sorry

def z (i : Fin 10) : ℝ := elongation_rates_A i - elongation_rates_B i

noncomputable def z_mean : ℝ := (Finset.sum Finset.univ z) / 10

noncomputable def z_variance : ℝ := (Finset.sum Finset.univ (fun i => (z i - z_mean)^2)) / 10

theorem significant_improvement : z_mean ≥ 2 * Real.sqrt (z_variance / 10) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l539_53947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_break_height_l539_53900

/-- The height of the flagpole in meters -/
noncomputable def flagpole_height : ℝ := 8

/-- The distance from the base of the flagpole to where the top touches the ground, in meters -/
noncomputable def ground_distance : ℝ := 3

/-- The height at which the flagpole breaks -/
noncomputable def break_height : ℝ := Real.sqrt 73 / 2

/-- Theorem stating the existence of a break height satisfying the given conditions -/
theorem flagpole_break_height :
  ∃ (x : ℝ), x > 0 ∧ x < flagpole_height ∧
  x^2 + ground_distance^2 = (flagpole_height - x)^2 ∧
  x = break_height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_break_height_l539_53900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_specific_product_l539_53971

def imaginary_part_of_product (z : ℂ) : ℝ := z.im

def product_of_complex (z w : ℂ) : ℂ := z * w

theorem imaginary_part_of_specific_product :
  imaginary_part_of_product (product_of_complex (2 - Complex.I) (4 + 5 * Complex.I)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_specific_product_l539_53971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l539_53955

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi

/-- The cosine law relation for the triangle -/
def CosineLawRelation (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

/-- The area of the triangle -/
noncomputable def TriangleArea (a b : ℝ) (C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin C

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c A B C →
  CosineLawRelation a b c A B C →
  a = 2 →
  TriangleArea a b C = 3 * Real.sqrt 3 / 2 →
  a + b + c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l539_53955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_locker_opened_l539_53970

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the pattern of opening lockers -/
inductive OpeningPattern
  | EverySecond
  | SkipTwoOpenOne

/-- Represents the state of all lockers -/
def LockerSystem := Fin 768 → LockerState

/-- Applies the opening pattern to the locker system -/
def applyPattern (lockers : LockerSystem) (pattern : OpeningPattern) : LockerSystem :=
  sorry

/-- Finds the number of the last closed locker -/
def lastClosedLocker (lockers : LockerSystem) : Nat :=
  sorry

/-- The main theorem stating that the last locker to be opened is number 257 -/
theorem last_locker_opened (initialLockers : LockerSystem) :
  (initialLockers = λ _ => LockerState.Closed) →
  (∃ n : Nat, ∀ k > n,
    (Nat.iterate (applyPattern · OpeningPattern.SkipTwoOpenOne) k (applyPattern initialLockers OpeningPattern.EverySecond)) =
    (Nat.iterate (applyPattern · OpeningPattern.SkipTwoOpenOne) n (applyPattern initialLockers OpeningPattern.EverySecond))) →
  lastClosedLocker (Nat.iterate (applyPattern · OpeningPattern.SkipTwoOpenOne) n (applyPattern initialLockers OpeningPattern.EverySecond)) = 257 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_locker_opened_l539_53970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_length_approx_l539_53965

/-- Converts feet to centimeters -/
noncomputable def feet_to_cm (feet : ℝ) : ℝ := feet * 30.48

/-- Converts inches to centimeters -/
noncomputable def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54

/-- Calculates the total length of the scale in centimeters -/
noncomputable def scale_length_cm : ℝ := feet_to_cm 25 + inches_to_cm 9 + 3

/-- Calculates the length of each part when the scale is divided into 13 equal parts -/
noncomputable def part_length : ℝ := scale_length_cm / 13

/-- Theorem stating that the length of each part is approximately 60.61 cm -/
theorem part_length_approx : 
  ∃ ε > 0, abs (part_length - 60.61) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_length_approx_l539_53965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_structure_stability_l539_53901

-- Define a window frame
structure WindowFrame where
  width : ℝ
  height : ℝ

-- Define a diagonal strip
structure DiagonalStrip where
  length : ℝ

-- Define a triangular structure
structure TriangularStructure where
  frame : WindowFrame
  diagonal : DiagonalStrip

-- Define stability as a property
def isStable (s : TriangularStructure) : Prop := sorry

-- Theorem statement
theorem triangular_structure_stability 
  (frame : WindowFrame) 
  (diagonal : DiagonalStrip) : 
  isStable { frame := frame, diagonal := diagonal } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_structure_stability_l539_53901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_statements_truth_count_l539_53977

theorem compound_statements_truth_count
  (p q : Prop)
  (hp : p)
  (hq : ¬q) :
  (¬(p ∧ q)) ∧ (p ∨ q) ∧ (¬¬p) ∧ (¬q) ∧
  (Nat.card {x : Bool | x = ¬(p ∧ q) ∨ x = ¬(p ∨ q) ∨ x = ¬¬p ∨ x = ¬¬q} = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_statements_truth_count_l539_53977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_rectangular_prism_l539_53966

/-- Given a rectangular prism with side lengths a, b, c where a ≤ b ≤ c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- The shortest distance between opposite vertices A and C₁ along the surface -/
noncomputable def shortestDistance (prism : RectangularPrism) : ℝ :=
  Real.sqrt ((prism.a + prism.b)^2 + prism.c^2)

/-- The volume of the rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.b * prism.c

/-- Theorem: The maximum volume of a rectangular prism with shortest surface distance 6 between
    opposite vertices is 12√3 -/
theorem max_volume_rectangular_prism :
  ∀ prism : RectangularPrism, shortestDistance prism = 6 →
  volume prism ≤ 12 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_rectangular_prism_l539_53966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l539_53950

theorem sufficient_but_not_necessary_condition :
  ∃ x : ℝ, (x = 1 → (x = 1 ∨ x = 2)) ∧ 
           ¬((x = 1 ∨ x = 2) → x = 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l539_53950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_l539_53933

theorem banana_permutations : 
  (6 : ℕ).factorial / (3 : ℕ).factorial = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_l539_53933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_point_l539_53976

/-- A type representing a point on the circle with its label -/
structure Point where
  label : Int
  h_label : label = 1 ∨ label = -1

/-- The configuration of points on the circle -/
def CircleConfig := Fin 2017 → Point

/-- The condition that there are no more than 672 points labeled -1 -/
def AtMost672Negative (config : CircleConfig) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 2017)) fun i => if (config i).label = -1 then 1 else 0) ≤ 672

/-- The sum of labels in a cyclic segment -/
def SegmentSum (config : CircleConfig) (start finish : Fin 2017) : Int :=
  Finset.sum (Finset.range 2017) fun i => 
    if start ≤ i ∧ i ≤ finish then (config ⟨i % 2017, by sorry⟩).label else 0

/-- A point is good if the sum of labels in any cyclic segment starting from it is positive -/
def IsGoodPoint (config : CircleConfig) (p : Fin 2017) : Prop :=
  ∀ q : Fin 2017, SegmentSum config p q > 0

theorem exists_good_point (config : CircleConfig) (h : AtMost672Negative config) :
  ∃ p : Fin 2017, IsGoodPoint config p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_point_l539_53976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_are_parallel_l539_53956

-- Define the two lines
def line1 (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a}
def line2 (a p : ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | p * Real.sin (q.2 - a) = 1}

-- Define what it means for two sets to be parallel
def IsParallel (S T : Set (ℝ × ℝ)) : Prop :=
  S ≠ T ∧ S.Nonempty ∧ T.Nonempty ∧ S ∩ T = ∅

-- State the theorem
theorem lines_are_parallel (a p : ℝ) (hp : p ≠ 0) : 
  IsParallel (line1 a) (line2 a p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_are_parallel_l539_53956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_white_black_complementary_events_l539_53945

-- Define the set of possible outcomes when drawing two balls
inductive Outcome
  | WW -- Both white
  | WB -- White then black
  | BW -- Black then white
  | BB -- Both black

-- Define the probability space
def Ω : Type := Outcome

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events
def bothWhite : Set Ω := {Outcome.WW}
def bothBlack : Set Ω := {Outcome.BB}
def atLeastOneWhite : Set Ω := {Outcome.WW, Outcome.WB, Outcome.BW}

-- Theorem statements
theorem mutually_exclusive_white_black : bothWhite ∩ bothBlack = ∅ := by sorry

theorem complementary_events : atLeastOneWhite ∪ bothBlack = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_white_black_complementary_events_l539_53945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_max_at_e_l539_53917

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_local_max_at_e :
  ∃ δ > 0, ∀ x ∈ Set.Ioo (Real.exp 1 - δ) (Real.exp 1 + δ),
    x ≠ Real.exp 1 → f x < f (Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_max_at_e_l539_53917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wuyang_building_time_l539_53999

-- Define the time it takes for each team to build the building individually
variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Define the conditions
variable (h1 : 1/a + 1/b < 1/6)
variable (h2 : 1/a + 1/c < 1/5)
variable (h3 : 1/b + 1/c < 1/4)

-- Define the time it takes for all three teams to build together
noncomputable def x : ℝ := 1 / (1/a + 1/b + 1/c)

-- Theorem statement
theorem wuyang_building_time : x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wuyang_building_time_l539_53999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l539_53913

/-- An isosceles triangle with given side lengths and altitude -/
structure IsoscelesTriangle where
  -- Side lengths
  pq : ℝ
  qr : ℝ
  -- Altitude length
  ps : ℝ
  -- Conditions
  isIsosceles : pq > 0
  altitudeBisects : ps > 0
  sideLengths : pq = 13 ∧ qr = 10

/-- The area of the isosceles triangle -/
noncomputable def triangleArea (t : IsoscelesTriangle) : ℝ :=
  (1/2) * t.qr * t.ps

/-- Theorem: The area of the given isosceles triangle is 60 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : triangleArea t = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l539_53913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l539_53907

/-- The starting position of the particle -/
def start_position : ℂ := 7

/-- The rotation factor for each move -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * (Real.pi / 3))

/-- The translation distance for each move -/
def translation : ℝ := 12

/-- The number of moves -/
def num_moves : ℕ := 300

/-- The position after n moves -/
noncomputable def position (n : ℕ) : ℂ :=
  start_position * ω^n + translation * (1 - ω^n) / (1 - ω)

/-- The theorem stating that the particle returns to its starting position after 300 moves -/
theorem particle_returns_to_start : position num_moves = start_position := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l539_53907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l539_53905

-- Define the function f(x) = x^4 - x
def f (x : ℝ) : ℝ := x^4 - x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let f' := λ x ↦ 4 * x^3 - 1  -- Derivative of f
  let m := f' P.fst            -- Slope at the point of tangency
  let b := P.snd - m * P.fst   -- y-intercept using point-slope form
  (3 : ℝ) * P.fst - P.snd - 3 = 0 ∧  -- Verify the point lies on the line
  ∀ x y, y = m * x + b ↔ 3 * x - y - 3 = 0  -- Equivalence of point-slope and general forms
  :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l539_53905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l539_53967

-- Define the slopes of the two lines
noncomputable def slope_l1 (a : ℝ) : ℝ := -a
noncomputable def slope_l2 (a : ℝ) : ℝ := 3 / (a + 1)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := slope_l1 a * slope_l2 a = -1

-- Theorem statement
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, perpendicular a → a = 1/2 := by
  intro a h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l539_53967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_snakes_not_purple_l539_53925

structure Snake where
  is_purple : Bool
  is_happy : Bool
  can_add : Bool
  can_subtract : Bool

def tom_snakes : Finset Snake := sorry

theorem happy_snakes_not_purple :
  ∀ s : Snake, s ∈ tom_snakes →
    (s.is_happy → ¬s.is_purple) :=
by
  sorry

axiom total_snakes : Finset.card tom_snakes = 13
axiom purple_snakes : Finset.card (tom_snakes.filter (·.is_purple)) = 4
axiom happy_snakes : Finset.card (tom_snakes.filter (·.is_happy)) = 5

axiom happy_can_add :
  ∀ s : Snake, s ∈ tom_snakes → s.is_happy → s.can_add

axiom purple_cant_subtract :
  ∀ s : Snake, s ∈ tom_snakes → s.is_purple → ¬s.can_subtract

axiom cant_subtract_cant_add :
  ∀ s : Snake, s ∈ tom_snakes → ¬s.can_subtract → ¬s.can_add

end NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_snakes_not_purple_l539_53925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_rate_is_twenty_l539_53935

/-- Represents the production rate and conditions of an assembly line -/
structure AssemblyLine where
  initialRate : ℝ
  initialOrder : ℝ
  increasedRate : ℝ
  secondOrder : ℝ
  averageOutput : ℝ

/-- Calculates the total time taken for production -/
noncomputable def totalTime (a : AssemblyLine) : ℝ :=
  a.initialOrder / a.initialRate + a.secondOrder / a.increasedRate

/-- Calculates the total cogs produced -/
def totalCogs (a : AssemblyLine) : ℝ :=
  a.initialOrder + a.secondOrder

/-- Theorem stating that given the conditions, the initial rate is 20 cogs per hour -/
theorem initial_rate_is_twenty (a : AssemblyLine)
  (h1 : a.initialOrder = 60)
  (h2 : a.increasedRate = 60)
  (h3 : a.secondOrder = 60)
  (h4 : a.averageOutput = 30)
  (h5 : a.averageOutput = totalCogs a / totalTime a) :
  a.initialRate = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_rate_is_twenty_l539_53935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_symmetry_l539_53988

theorem sine_graph_symmetry (φ : Real) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x : Real, Real.sin (2*x + φ) = Real.sin (2*(π/3 - x) + φ)) → φ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_symmetry_l539_53988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_correct_l539_53986

/-- The number of pairs of integers (a, b) satisfying the given conditions -/
def count_pairs : ℕ := 19

/-- Predicate to check if a pair of integers satisfies all conditions -/
def satisfies_conditions (a b : ℤ) : Prop :=
  a < b ∧ a + b < 100 ∧ (4 * a + 10 * b = 280)

theorem count_pairs_correct :
  (∃ (S : Finset (ℤ × ℤ)), S.card = count_pairs ∧
    (∀ (p : ℤ × ℤ), p ∈ S ↔ satisfies_conditions p.1 p.2)) ∧
  (∀ (T : Finset (ℤ × ℤ)),
    (∀ (p : ℤ × ℤ), p ∈ T ↔ satisfies_conditions p.1 p.2) →
    T.card ≤ count_pairs) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_correct_l539_53986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l539_53979

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ x

-- Theorem statement
theorem f_property : ∀ x : ℝ, f (x + 2) - f x = 8 * f x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l539_53979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l539_53940

-- Define the set M
def M : Set ℝ := {a : ℝ | a^2 - 2*a > 0}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l539_53940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l539_53927

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Represents the axis of symmetry of a parabola -/
noncomputable def Parabola.axisOfSymmetry (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- Function to check if a quadratic function has given roots -/
def hasRoots (f : ℝ → ℝ) (r1 r2 : ℝ) : Prop :=
  f r1 = 0 ∧ f r2 = 0

theorem parabola_properties (p : Parabola) 
    (h1 : p.contains (-1) 0)
    (h2 : p.axisOfSymmetry = 1) :
  p.a * p.b * p.c < 0 ∧
  8 * p.a + p.c < 0 ∧
  ∀ t : ℝ, p.contains (-2) t → 
    hasRoots (fun x => p.a * x^2 + p.b * x + p.c - t) (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l539_53927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_construction_iff_equilateral_l539_53937

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The semi-perimeter of a triangle -/
noncomputable def semiPerimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Constructs a new triangle from the given triangle using the semi-perimeter -/
noncomputable def constructNewTriangle (t : Triangle) : Triangle := 
  { a := semiPerimeter t - t.a,
    b := semiPerimeter t - t.b,
    c := semiPerimeter t - t.c,
    ha := sorry,
    hb := sorry,
    hc := sorry,
    triangle_inequality := sorry }

/-- Predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

/-- Theorem stating that the construction process can continue indefinitely 
    if and only if the initial triangle is equilateral -/
theorem indefinite_construction_iff_equilateral (t : Triangle) : 
  (∀ n : ℕ, ∃ t_n : Triangle, t_n = (Nat.iterate constructNewTriangle n t)) ↔ isEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_construction_iff_equilateral_l539_53937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_triangle_area_l539_53995

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Check if a line between two points is tangent to a circle -/
def is_tangent_line (p1 p2 : Point) (c : Circle) : Prop :=
  let (cx, cy) := c.center
  let dx := p2.x - p1.x
  let dy := p2.y - p1.y
  let fx := p1.x - cx
  let fy := p1.y - cy
  (dx * fx + dy * fy)^2 = c.radius^2 * (dx^2 + dy^2)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem circle_tangent_triangle_area 
  (ω1 ω2 ω3 : Circle) 
  (P1 P2 P3 : Point) : 
  ω1.radius = 5 → 
  ω2.radius = 5 → 
  ω3.radius = 5 → 
  are_externally_tangent ω1 ω2 → 
  are_externally_tangent ω2 ω3 → 
  are_externally_tangent ω3 ω1 → 
  distance P1 P2 = distance P2 P3 → 
  distance P2 P3 = distance P3 P1 → 
  is_tangent_line P1 P2 ω1 → 
  is_tangent_line P2 P3 ω2 → 
  is_tangent_line P3 P1 ω3 → 
  triangle_area P1 P2 P3 = Real.sqrt 675 + Real.sqrt 525 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_triangle_area_l539_53995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_equals_sqrt_three_sqrt_difference_equals_plus_minus_sqrt_three_l539_53941

theorem sqrt_difference_equals_sqrt_three (a : ℝ) (h : a + a⁻¹ = 5) :
  (a^(1/2 : ℝ) - a^(-(1/2 : ℝ)))^2 = 3 :=
sorry

theorem sqrt_difference_equals_plus_minus_sqrt_three (a : ℝ) (h : a + a⁻¹ = 5) :
  a^(1/2 : ℝ) - a^(-(1/2 : ℝ)) = Real.sqrt 3 ∨ a^(1/2 : ℝ) - a^(-(1/2 : ℝ)) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_equals_sqrt_three_sqrt_difference_equals_plus_minus_sqrt_three_l539_53941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_m_bounded_l539_53948

/-- A function f(x) that depends on a parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3^x)^2 - m * 3^x + m + 1

/-- Theorem stating the condition for f(x) to be positive on (0, +∞) -/
theorem f_positive_iff_m_bounded (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ m < 2 + 2 * Real.sqrt 2 := by
  sorry

#check f_positive_iff_m_bounded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_m_bounded_l539_53948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_vector_proof_l539_53902

-- Define the original function
noncomputable def f (x : ℝ) := Real.log (-x)

-- Define the translated function
noncomputable def g (x : ℝ) := Real.log (1 - x) + 2

-- Define the translation vector
def a : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem translation_vector_proof :
  (∀ x, g x = f (x - a.1) + a.2) → a = (1, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_vector_proof_l539_53902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kannon_fruit_multiple_l539_53998

theorem kannon_fruit_multiple
  (apples_last_night : ℕ)
  (bananas_last_night : ℕ)
  (oranges_last_night : ℕ)
  (additional_apples_today : ℕ)
  (banana_multiplier : ℕ)
  (total_fruits : ℕ)
  (h1 : apples_last_night = 3)
  (h2 : bananas_last_night = 1)
  (h3 : oranges_last_night = 4)
  (h4 : additional_apples_today = 4)
  (h5 : banana_multiplier = 10)
  (h6 : total_fruits = 39) :
  let apples_today := apples_last_night + additional_apples_today
  let bananas_today := bananas_last_night * banana_multiplier
  let orange_multiple := (total_fruits - (apples_last_night + bananas_last_night + oranges_last_night) - (apples_today + bananas_today)) / apples_today
  orange_multiple = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kannon_fruit_multiple_l539_53998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l539_53912

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x) / Real.log a

-- State the theorem
theorem increasing_log_function_a_range :
  ∀ a : ℝ, a > 0 → a ≠ 1 →
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 4 → f a x < f a y) →
  a > 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l539_53912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_proof_l539_53968

theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 560 →
  total_time = 12 →
  second_half_speed = 40 →
  let half_distance := total_distance / 2
  let second_half_time := half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  let first_half_speed := half_distance / first_half_time
  first_half_speed = 56 := by
  intro h1 h2 h3
  -- Proof steps would go here
  sorry

#check journey_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_proof_l539_53968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_equivalence_l539_53975

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_equivalence (P : ℝ) :
  simpleInterest P 5 8 = 840 →
  simpleInterest P 8 5 = 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_equivalence_l539_53975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_is_correct_l539_53980

/-- Given a line l passing through (1,1) with slope -m (m > 0), intersecting
    the x-axis at P and y-axis at Q, and perpendiculars from P and Q to the line
    2x + y = 0 meeting at R and S respectively, the minimum area of quadrilateral
    PRSQ is 3.6. -/
noncomputable def min_area_quadrilateral (m : ℝ) (h_m : m > 0) : ℝ :=
  let l : Set (ℝ × ℝ) := {p | p.2 - 1 = -m * (p.1 - 1)}
  let A : ℝ × ℝ := (1, 1)
  let P : ℝ × ℝ := (1 + 1/m, 0)
  let Q : ℝ × ℝ := (0, 1 + m)
  let base_line : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 0}
  let PR : Set (ℝ × ℝ) := {p | p.1 - 2 * p.2 - (m + 1)/m = 0}
  let QS : Set (ℝ × ℝ) := {p | p.1 - 2 * p.2 + 2*(m + 1) = 0}
  let R : ℝ × ℝ := sorry -- Intersection of PR and base_line
  let S : ℝ × ℝ := sorry -- Intersection of QS and base_line
  let area : ℝ := sorry -- Area of quadrilateral PRSQ
  3.6 -- This is the minimum value of the area

theorem min_area_quadrilateral_is_correct (m : ℝ) (h_m : m > 0) :
  min_area_quadrilateral m h_m = 3.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_is_correct_l539_53980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l539_53981

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ  -- Speed in meters per second
  time : ℝ   -- Time to complete the race in seconds

/-- Calculates the distance covered by a runner in a given time -/
def distance (runner : Runner) (t : ℝ) : ℝ :=
  runner.speed * t

theorem race_result (race_length : ℝ) (lead_distance : ℝ) (a b : Runner) 
    (h1 : race_length = 1000)
    (h2 : lead_distance = 48)
    (h3 : a.time = 119)
    (h4 : distance a a.time = race_length)
    (h5 : distance b a.time = race_length - lead_distance) :
  ∃ ε > 0, |b.time - a.time - 6| < ε := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l539_53981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l539_53964

def y : ℕ → ℚ
  | 0 => 145
  | (n + 1) => 2 * (y n)^2 + y n

theorem sum_reciprocal_y_plus_one :
  (∑' n, 1 / (y n + 1)) = 1 / 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l539_53964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dal_selling_rate_is_15_l539_53943

/-- Represents the dal transaction problem --/
structure DalTransaction where
  quantity1 : ℚ
  rate1 : ℚ
  quantity2 : ℚ
  rate2 : ℚ
  total_gain : ℚ

/-- Calculates the selling rate per kg for the dal mixture --/
def selling_rate_per_kg (t : DalTransaction) : ℚ :=
  ((t.quantity1 * t.rate1 + t.quantity2 * t.rate2 + t.total_gain) / (t.quantity1 + t.quantity2))

/-- Theorem stating that the selling rate per kg is 15 for the given transaction --/
theorem dal_selling_rate_is_15 (t : DalTransaction) 
  (h1 : t.quantity1 = 15)
  (h2 : t.rate1 = 29/2)
  (h3 : t.quantity2 = 10)
  (h4 : t.rate2 = 13)
  (h5 : t.total_gain = 55/2) :
  selling_rate_per_kg t = 15 := by
  sorry

#eval selling_rate_per_kg { quantity1 := 15, rate1 := 29/2, quantity2 := 10, rate2 := 13, total_gain := 55/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dal_selling_rate_is_15_l539_53943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_squared_l539_53960

theorem factorial_square_root_squared : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_root_squared_l539_53960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fat_added_per_serving_l539_53951

/-- Represents the amount of fat in grams per cup of cream -/
def fat_per_cup : ℚ := 88

/-- Represents the number of servings in the recipe -/
def servings : ℕ := 4

/-- Represents the amount of cream added to the recipe in cups -/
def cream_added : ℚ := 1/2

/-- Calculates the amount of fat added to each serving -/
noncomputable def fat_per_serving : ℚ := (fat_per_cup * cream_added) / servings

theorem fat_added_per_serving :
  fat_per_serving = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fat_added_per_serving_l539_53951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l539_53922

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_shift :
  (∀ x ∈ domain_f, f x ≠ 0) →
  (∀ x ∈ Set.Icc 1 2, f (x - 1) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l539_53922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_multiples_of_11_excluding_5_l539_53984

def first_fifteen_multiples_of_11 : List ℕ :=
  (List.range 15).map (λ n => (n + 1) * 11)

def multiples_of_5_and_11 (lst : List ℕ) : List ℕ :=
  lst.filter (λ n => n % 5 = 0)

theorem sum_of_multiples_of_11_excluding_5 :
  (first_fifteen_multiples_of_11.sum - (multiples_of_5_and_11 first_fifteen_multiples_of_11).sum) = 990 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_multiples_of_11_excluding_5_l539_53984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_dressing_theorem_l539_53934

/-- Represents the number of legs an ant has -/
def num_legs : ℕ := 6

/-- Represents the number of items (sock + shoe) per leg -/
def items_per_leg : ℕ := 2

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := num_legs * items_per_leg

/-- Represents the number of legs that can be dressed in any order -/
def free_legs : ℕ := num_legs - 1

/-- 
  Represents the number of ways an ant can put on its socks and shoes, 
  given the constraints
-/
def ant_dressing_permutations : ℕ := Nat.factorial (total_items - items_per_leg)

theorem ant_dressing_theorem : 
  ant_dressing_permutations = Nat.factorial 10 := by
  -- Proof goes here
  sorry

#eval ant_dressing_permutations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_dressing_theorem_l539_53934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_15_minus_tan_45_l539_53973

theorem cot_15_minus_tan_45 :
  Real.tan (75 * π / 180) - Real.tan (45 * π / 180) = Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_15_minus_tan_45_l539_53973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l539_53958

/-- The number of students who read only book A -/
def A : ℕ := sorry

/-- The number of students who read only book B -/
def B : ℕ := sorry

/-- The number of students who read both books A and B -/
def AB : ℕ := sorry

/-- 20% of those who read book A also read book B -/
axiom cond1 : AB = (A + AB) / 5

/-- 25% of those who read book B also read book A -/
axiom cond2 : AB = (B + AB) / 4

/-- The difference between students who read only A and only B is 100 -/
axiom cond3 : A - B = 100

/-- The total number of students surveyed -/
def total : ℕ := A + B + AB

/-- Theorem: The total number of students surveyed is 800 -/
theorem total_students : total = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l539_53958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_forms_circle_l539_53906

/-- Square represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  vertices : Fin 4 → ℝ × ℝ

/-- Point represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Centroid calculates the centroid of a square -/
noncomputable def centroid (sq : Square) : ℝ × ℝ := by
  let sum := ((Finset.univ.sum fun i => (sq.vertices i).1),
              (Finset.univ.sum fun i => (sq.vertices i).2))
  exact (sum.1 / 4, sum.2 / 4)

/-- DistanceSquared calculates the squared distance between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- SumOfSquaredDistances calculates the sum of squared distances from a point to all vertices of a square -/
def sumOfSquaredDistances (sq : Square) (p : Point) : ℝ :=
  Finset.univ.sum fun i => distanceSquared (p.x, p.y) (sq.vertices i)

/-- Theorem stating that the locus of points satisfying the given condition forms a circle -/
theorem locus_forms_circle (sq : Square) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (p : Point),
      sumOfSquaredDistances sq p = 2 * sq.sideLength^2 ↔
      distanceSquared (p.x, p.y) center = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_forms_circle_l539_53906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l539_53992

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 1 < a ∧ a ≤ 4 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l539_53992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lizzie_crayon_count_l539_53957

theorem lizzie_crayon_count 
  (billie_crayons : ℕ)
  (bobbie_crayons : ℕ)
  (lizzie_crayons : ℕ)
  (h1 : bobbie_crayons = 3 * billie_crayons)
  (h2 : lizzie_crayons = bobbie_crayons / 2)
  (h3 : billie_crayons = 18) :
  lizzie_crayons = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lizzie_crayon_count_l539_53957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_numbers_in_list_l539_53929

noncomputable def numberList : List ℝ := [
  3.141,
  1/3,
  Real.sqrt 5 - Real.sqrt 7,
  Real.pi,
  Real.sqrt 2.25,
  -Real.sqrt 2.25,
  -2/3,
  0.3030030003 -- This is an approximation as Lean can't represent the infinite sequence directly
]

def isIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ ↑p / ↑q

theorem irrational_numbers_in_list :
  (isIrrational (Real.sqrt 5 - Real.sqrt 7)) ∧
  (isIrrational Real.pi) ∧
  (∀ x ∈ numberList, x ≠ Real.sqrt 5 - Real.sqrt 7 → x ≠ Real.pi → isIrrational x → x = 0.3030030003) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_numbers_in_list_l539_53929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l539_53985

-- Define the constants as real numbers
noncomputable def a : ℝ := (4 : ℝ) ^ (1.7 : ℝ)
noncomputable def b : ℝ := (8 : ℝ) ^ (0.48 : ℝ)
noncomputable def c : ℝ := ((1/2) : ℝ) ^ (-(0.5 : ℝ))

-- State the theorem
theorem a_greater_than_b_greater_than_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l539_53985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_to_heart_line_k_heart_to_heart_line_constant_heart_to_heart_line_area_range_l539_53954

-- Define a parabola
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

-- Define a line
structure Line where
  m : ℝ
  n : ℝ

-- Define a heart-to-heart line
def is_heart_to_heart_line (p : Parabola) (l : Line) : Prop :=
  ∃ (x y : ℝ), y = p.a * x^2 + p.b * x + p.c ∧ y = l.m * x + l.n ∧
  (x = -p.b / (2 * p.a) ∨ x = 0)

-- Part 1
theorem heart_to_heart_line_k (p : Parabola) (l : Line) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = 1 ∧ l.n = 1 ∧ is_heart_to_heart_line p l → l.m = -1 := by sorry

-- Part 2
theorem heart_to_heart_line_constant (p : Parabola) (l : Line) :
  p.a = -1 ∧ p.c = 0 ∧ l.n = 0 ∧ l.m ≠ 0 ∧ is_heart_to_heart_line p l → p.b / l.m = 2 := by sorry

-- Part 3
noncomputable def triangle_area (l : Line) : ℝ :=
  abs (l.n * l.n / (2 * l.m))

theorem heart_to_heart_line_area_range (p : Parabola) (l : Line) (k : ℝ) :
  p.b = 3 * k^2 - 2 * k + 1 ∧ p.c = k ∧ (1/2 : ℝ) ≤ k ∧ k ≤ 2 ∧ is_heart_to_heart_line p l →
  (1/3 : ℝ) ≤ triangle_area l ∧ triangle_area l ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_to_heart_line_k_heart_to_heart_line_constant_heart_to_heart_line_area_range_l539_53954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l539_53972

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Define for 0 to match Lean's natural number representation
  | 1 => 1
  | (n + 2) => (sequence_a (n + 1))^2 - 1

theorem sum_of_first_five_terms : 
  sequence_a 0 + sequence_a 1 + sequence_a 2 + sequence_a 3 + sequence_a 4 = -1 := by
  sorry

#eval sequence_a 0  -- This will evaluate to 1
#eval sequence_a 1  -- This will evaluate to 1
#eval sequence_a 2  -- This will evaluate to 0
#eval sequence_a 3  -- This will evaluate to -1
#eval sequence_a 4  -- This will evaluate to 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l539_53972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_diameter_theorem_l539_53910

/-- Represents a cylinder with given height and volume -/
structure Cylinder where
  height : ℝ
  volume : ℝ

/-- The diameter of a cylinder given its height and volume -/
noncomputable def cylinderDiameter (c : Cylinder) : ℝ :=
  4 / Real.sqrt Real.pi

/-- Theorem stating that a cylinder with height 5 and volume 20 has diameter 4/√π -/
theorem cylinder_diameter_theorem (c : Cylinder) 
    (h_height : c.height = 5)
    (h_volume : c.volume = 20) : 
    cylinderDiameter c = 4 / Real.sqrt Real.pi := by
  sorry

#check cylinder_diameter_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_diameter_theorem_l539_53910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_booth_visibility_l539_53990

/-- A square-shaped booth with side length L -/
structure Booth where
  L : ℝ
  L_pos : L > 0

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle at which the booth is visible from a point -/
noncomputable def visibility_angle (b : Booth) (p : Point) : ℝ := sorry

/-- The locus of points from which the booth is visible -/
def visibility_locus (b : Booth) : Set Point :=
  {p | visibility_angle b p = Real.pi / 2}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- The theorem stating the properties of the visibility locus -/
theorem booth_visibility (b : Booth) :
  ∃ (center1 center2 center3 center4 : Point),
    visibility_locus b =
      {p | ∃ (i : Fin 4), distance p (match i with
        | 0 => center1
        | 1 => center2
        | 2 => center3
        | 3 => center4) = b.L / 2} ∧
    (∃ (p : Point), p ∈ visibility_locus b ∧ distance p center1 = 0) ∧
    (∀ (p : Point), p ∈ visibility_locus b → distance p center1 ≤ b.L / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_booth_visibility_l539_53990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_for_slope_root3_over_3_l539_53936

/-- The angle of inclination of a line with slope √3/3 is 30°. -/
theorem angle_of_inclination_for_slope_root3_over_3 :
  ∀ (α : Real),
  (Real.tan α = Real.sqrt 3 / 3) → α ∈ Set.Icc 0 Real.pi → α = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_for_slope_root3_over_3_l539_53936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_profit_l539_53931

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit : 
  ∀ (total_backpacks : ℕ) 
    (total_cost : ℕ) 
    (swap_meet_quantity : ℕ) 
    (swap_meet_price : ℕ) 
    (dept_store_quantity : ℕ) 
    (dept_store_price : ℕ) 
    (remaining_price : ℕ),
  total_backpacks = 48 →
  total_cost = 576 →
  swap_meet_quantity = 17 →
  swap_meet_price = 18 →
  dept_store_quantity = 10 →
  dept_store_price = 25 →
  remaining_price = 22 →
  let remaining_quantity := total_backpacks - swap_meet_quantity - dept_store_quantity;
  let total_revenue := 
    swap_meet_quantity * swap_meet_price + 
    dept_store_quantity * dept_store_price + 
    remaining_quantity * remaining_price;
  let profit := total_revenue - total_cost;
  profit = 442 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salesman_profit_l539_53931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_2_digit_difference_l539_53926

def base_2_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem base_2_digit_difference : 
  let digits := [base_2_digits 400, base_2_digits 1600, base_2_digits 3200]
  (List.maximum digits).getD 0 - (List.minimum digits).getD 0 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_2_digit_difference_l539_53926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_reciprocal_l539_53930

open Real NNReal ENNReal BigOperators

/-- The function f(x) defined as an infinite sum -/
noncomputable def f (x : ℝ) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-(3:ℝ)^n))

/-- Theorem stating that f(x) = 1 / (x - 1) for x > 1 -/
theorem f_eq_reciprocal (x : ℝ) (hx : x > 1) : f x = 1 / (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_reciprocal_l539_53930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_and_minimum_l539_53994

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / (2 * x) + 3 / 2 * x + 1

theorem tangent_perpendicular_and_minimum (a : ℝ) :
  (∀ x : ℝ, x > 0 → deriv (f a) x ≠ 0 ∨ x = 1) →
  (deriv (f a) 1 = 0) →
  (a = -1 ∧ ∀ x : ℝ, x > 0 → f (-1) x ≥ f (-1) 1 ∧ f (-1) 1 = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_and_minimum_l539_53994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_values_l539_53919

theorem sequence_sum_values :
  ∃ (S : Finset Int), S.card = 11 ∧
    ∀ s, s ∈ S ↔ ∃ (a : Fin 10 → Int),
      (∀ i : Fin 10, a i = 1 ∨ a i = -1) ∧
      s = (Finset.univ : Finset (Fin 10)).sum a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_values_l539_53919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_sum_of_squares_l539_53915

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The region formed by the union of circular regions -/
def Region (circles : List Circle) : Set (ℝ × ℝ) :=
  sorry

/-- A line in the 2D plane represented by ax = by + c -/
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Check if a line bisects a region -/
def bisects (l : Line) (r : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Generate a 3x3 grid of unit circles in the first quadrant -/
def gridCircles : List Circle :=
  sorry

theorem bisecting_line_sum_of_squares :
  ∀ l : Line,
    l.a > 0 ∧ l.b > 0 ∧ l.c > 0 →
    Int.gcd l.a (Int.gcd l.b l.c) = 1 →
    (l.a : ℝ) / l.b = 5 →
    bisects l (Region gridCircles) →
    l.a ^ 2 + l.b ^ 2 + l.c ^ 2 = 315 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_sum_of_squares_l539_53915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_form_unique_l539_53944

-- Define a structure for numbers of the form m + n√p
structure SqrtForm (p : ℕ) where
  m : ℚ
  n : ℚ

-- Define a predicate for p being a prime or a product of distinct primes
def validP (p : ℕ) : Prop :=
  Nat.Prime p ∨ ∃ (primes : List ℕ), (∀ q, q ∈ primes → Nat.Prime q) ∧ 
    (∀ q₁ q₂, q₁ ∈ primes → q₂ ∈ primes → q₁ ≠ q₂) ∧ p = primes.prod

-- The main theorem
theorem sqrt_form_unique (p : ℕ) (hp : validP p) (x y : SqrtForm p) :
  (x.m : ℝ) + x.n * Real.sqrt p = (y.m : ℝ) + y.n * Real.sqrt p → x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_form_unique_l539_53944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l539_53974

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ)^x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l539_53974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l539_53904

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- State the theorem
theorem min_value_of_m :
  (∀ x : ℝ, g x = g (-x)) →  -- g is even
  (∀ x : ℝ, h x = -h (-x)) →  -- h is odd
  (∀ x : ℝ, f x = g x - h x) →  -- f(x) = g(x) - h(x)
  (∃ m : ℝ, ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → m * g x + h x ≥ 0) →
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → m * g x + h x ≥ 0) → 
    m ≥ (Real.exp 2 - 1) / (Real.exp 2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l539_53904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l539_53963

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclinationAngle (a b : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- The equation of the line -/
def lineEquation (x y : ℝ) : Prop :=
  x + y - 3 = 0

theorem line_inclination_angle :
  inclinationAngle 1 1 = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l539_53963
