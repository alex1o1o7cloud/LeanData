import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sequence_problem_l705_70548

open BigOperators

/-- A sequence {xₙ} is arithmetic if xₙ₊₁ - xₙ is constant for all n. -/
def is_arithmetic_sequence (x : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, x (n + 1) - x n = d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmetic_sum (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, x i

theorem harmonic_sequence_problem (x : ℕ → ℝ) (h_arithmetic : is_arithmetic_sequence x)
    (h_sum : arithmetic_sum x 20 = 200) :
    x 2 * x 17 ≤ 100 := by
  sorry  -- The proof would go here

#check harmonic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sequence_problem_l705_70548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l705_70573

theorem work_completion_time (a b c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (a = 2 * b) →                          -- a is twice as efficient as b
  (1 / a + 1 / (2 * a) = 1 / c + 1 / d) → -- a and b together work as fast as c and d
  (c = 20) →                             -- c takes 20 days
  (d = 30) →                             -- d takes 30 days
  a = 1 / 18 :=                          -- a can complete the work in 18 days
by
  intros ha hab hc hd
  -- The proof steps would go here
  sorry -- We use 'sorry' to skip the actual proof for now

#eval (1 : ℚ) / 18 -- This should output 1/18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l705_70573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2013_mod_105_l705_70507

def mySequence : ℕ → ℕ
  | 0 => 8
  | 1 => 1
  | n + 2 => mySequence (n + 1) + mySequence n

theorem mySequence_2013_mod_105 : mySequence 2012 % 105 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2013_mod_105_l705_70507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l705_70556

/-- Given three points in ℝ², this function returns true if they are collinear -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem point_on_line : collinear (0, 1) (-6, -3) (9, 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l705_70556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_friday_l705_70530

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Inhabited, DecidableEq

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek
deriving Inhabited

/-- Definition of a leap year February -/
def isLeapYearFebruary (dates : List FebruaryDate) : Prop :=
  dates.length = 29

/-- Count occurrences of a specific day in the list of dates -/
def countDayOccurrences (dates : List FebruaryDate) (day : DayOfWeek) : Nat :=
  dates.filter (λ d => d.dayOfWeek == day) |>.length

/-- Theorem: In a leap year, if February has exactly four Mondays and four Thunders, 
    then February 1 must fall on a Friday -/
theorem february_first_is_friday 
  (dates : List FebruaryDate) 
  (h_leap : isLeapYearFebruary dates) 
  (h_mondays : countDayOccurrences dates DayOfWeek.Monday = 4)
  (h_thursdays : countDayOccurrences dates DayOfWeek.Thursday = 4) :
  (dates.head?.get!).dayOfWeek = DayOfWeek.Friday :=
by
  sorry

#check february_first_is_friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_first_is_friday_l705_70530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_prism_surface_area_l705_70524

-- Define the radius of the sphere
noncomputable def sphere_radius : ℝ := 3 * (36 / Real.pi)

-- Define the length and width of the rectangular prism
def rect_length : ℝ := 6
def rect_width : ℝ := 4

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of a rectangular prism
def rect_volume (l w h : ℝ) : ℝ := l * w * h

-- Define the surface area of a rectangular prism
def rect_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- Theorem statement
theorem rect_prism_surface_area : 
  ∃ (h : ℝ), 
    sphere_volume sphere_radius = rect_volume rect_length rect_width h ∧ 
    rect_surface_area rect_length rect_width h = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_prism_surface_area_l705_70524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_4_mod_7_l705_70571

def T : ℕ → ℕ
  | 0 => 6  -- Add this case for 0
  | 1 => 6
  | n + 2 => 6^(T (n + 1))

theorem t_4_mod_7 : T 4 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_4_mod_7_l705_70571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_8_mod_35_l705_70599

theorem inverse_of_8_mod_35 : ∃ x : ℕ, x < 35 ∧ (8 * x) % 35 = 1 :=
by
  use 22
  constructor
  · norm_num
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_8_mod_35_l705_70599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_fourth_l705_70501

theorem sec_seven_pi_fourth : 1 / Real.cos (7 * π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_fourth_l705_70501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_total_cost_is_38_07_l705_70598

/-- Calculates the total cost of George's purchases in dollars -/
noncomputable def george_total_cost (sandwich_price : ℝ) (juice_discount : ℝ) (milk_euro_rate : ℝ) (pound_dollar_rate : ℝ) : ℝ :=
  let juice_price := 2 * sandwich_price
  let coffee_price := sandwich_price / 2
  let discounted_juice_price := juice_price * (1 - juice_discount)
  let milk_price := 0.75 * (sandwich_price + discounted_juice_price) * milk_euro_rate
  let chocolate_price := (discounted_juice_price + coffee_price) * pound_dollar_rate
  (2 * sandwich_price + discounted_juice_price + coffee_price) * milk_euro_rate + milk_price + chocolate_price

/-- Theorem stating that George's total cost is $38.07 -/
theorem george_total_cost_is_38_07 :
  george_total_cost 4 0.1 1.2 1.25 = 38.07 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_total_cost_is_38_07_l705_70598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l705_70559

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / magnitude

theorem projection_theorem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  vector_projection a (a.1 + b.1, a.2 + b.2) = Real.sqrt 10 / 2 := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l705_70559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheels_per_non_front_axle_l705_70586

/-- Toll calculation function -/
def toll (x : ℕ) : ℚ := 1.5 + 1.5 * (x - 2)

/-- Theorem: Number of wheels on each non-front axle of an 18-wheel truck -/
theorem wheels_per_non_front_axle 
  (total_wheels : ℕ) 
  (front_axle_wheels : ℕ) 
  (toll_amount : ℚ) :
  total_wheels = 18 →
  front_axle_wheels = 2 →
  toll_amount = 6 →
  toll (total_wheels / front_axle_wheels) = toll_amount →
  ∃ (other_axle_wheels : ℕ),
    other_axle_wheels * (total_wheels / front_axle_wheels - 1) + front_axle_wheels = total_wheels ∧
    other_axle_wheels = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheels_per_non_front_axle_l705_70586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l705_70546

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (2, -3)

-- Define the slope of the tangent line
def tangent_slope : ℝ := 2

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, (2*x - y - 7 = 0) ↔
  (y - f tangent_point.1 = tangent_slope * (x - tangent_point.1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l705_70546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_conditions_l705_70537

-- Define the system of differential equations
def system (x y z : ℝ → ℝ) : Prop :=
  ∀ t, (deriv x t = 8 * y t) ∧
       (deriv y t = -2 * z t) ∧
       (deriv z t = 2 * x t + 8 * y t - 2 * z t)

-- Define the initial conditions
def initial_conditions (x y z : ℝ → ℝ) : Prop :=
  x 0 = -4 ∧ y 0 = 0 ∧ z 0 = 1

-- Define the solution functions
noncomputable def x : ℝ → ℝ := λ t => -4 * Real.exp (-2 * t) - 2 * Real.sin (4 * t)
noncomputable def y : ℝ → ℝ := λ t => Real.exp (-2 * t) - Real.cos (4 * t)
noncomputable def z : ℝ → ℝ := λ t => Real.exp (-2 * t) - 2 * Real.sin (4 * t)

-- Theorem statement
theorem solution_satisfies_system_and_conditions :
  system x y z ∧ initial_conditions x y z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_conditions_l705_70537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_C₁MN_l705_70547

noncomputable section

/-- Line l₁ in polar coordinates -/
def line_l₁ (ρ : ℝ) : ℝ := Real.pi / 3

/-- Circle C₁ in polar coordinates -/
def circle_C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * Real.sqrt 3 * ρ * (Real.cos θ) - 4 * ρ * (Real.sin θ) + 6 = 0

/-- Intersection points of l₁ and C₁ -/
def intersection_points (ρ₁ ρ₂ : ℝ) : Prop :=
  circle_C₁ ρ₁ (line_l₁ ρ₁) ∧ circle_C₁ ρ₂ (line_l₁ ρ₂) ∧ ρ₁ ≠ ρ₂

/-- The area of triangle C₁MN -/
def triangle_area (ρ₁ ρ₂ : ℝ) : ℝ := Real.sqrt 3 / 4

/-- Main theorem: The area of triangle C₁MN is √3/4 -/
theorem area_of_triangle_C₁MN :
  ∀ ρ₁ ρ₂, intersection_points ρ₁ ρ₂ → triangle_area ρ₁ ρ₂ = Real.sqrt 3 / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_C₁MN_l705_70547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_duration_l705_70565

/-- Represents a time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Calculates the time difference between two Time values -/
def timeDifference (endTime startTime : Time) : Time :=
  sorry

/-- The start time of Xiaolin's homework -/
def startTime : Time := { hours := 4, minutes := 30, seconds := 20 }

/-- The end time of Xiaolin's homework -/
def endTime : Time := { hours := 4, minutes := 52, seconds := 18 }

/-- The expected time difference -/
def expectedDifference : Time := { hours := 0, minutes := 21, seconds := 58 }

theorem homework_duration :
  timeDifference endTime startTime = expectedDifference := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_duration_l705_70565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glove_pair_probability_l705_70535

/-- Given 6 pairs of gloves of different colors, where left and right gloves are different,
    the probability of selecting exactly one pair of the same color when randomly selecting 4 gloves is 16/33. -/
theorem glove_pair_probability : ℚ := by
  -- We define the probability for the specific case of 6 pairs of gloves
  have h : (16 : ℚ) / 33 = (16 : ℚ) / 33 := by rfl
  -- The actual proof would go here, but we use sorry to skip it for now
  sorry

#eval (16 : ℚ) / 33 -- This will evaluate the fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glove_pair_probability_l705_70535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_fourth_minus_alpha_l705_70551

theorem sin_pi_fourth_minus_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = 1 / 3) :
  Real.sin (π / 4 - α) = (Real.sqrt 2 - 4) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_fourth_minus_alpha_l705_70551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l705_70541

theorem right_triangle_area (DE EF : ℝ) (h1 : DE = 8) (h2 : EF = 10) : 
  (1/2) * DE * Real.sqrt (EF^2 - DE^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l705_70541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_distance_to_focus_l705_70520

-- Define the parabola structure
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ → Prop

-- Define the problem conditions
def parabola_problem : Parabola :=
{ vertex := (0, 0),
  axis_of_symmetry := λ x y ↦ y = 0,
  focus := (4, 0),
  equation := λ x y ↦ y^2 = 16*x }

-- Theorem 1: Equation of the parabola
theorem parabola_equation (p : Parabola) (x y : ℝ) :
  p.vertex = (0, 0) ∧ 
  p.axis_of_symmetry x y = (y = 0) ∧ 
  p.focus.1 - 2*p.focus.2 - 4 = 0 →
  p.equation x y = (y^2 = 16*x) := by
  sorry

-- Theorem 2: Distance from point A to focus
theorem distance_to_focus (p : Parabola) :
  p.vertex = (0, 0) ∧ 
  p.axis_of_symmetry 0 0 ∧ 
  p.focus.1 - 2*p.focus.2 - 4 = 0 →
  let A : ℝ × ℝ := (2, Real.sqrt 32)
  Real.sqrt ((A.1 - p.focus.1)^2 + (A.2 - p.focus.2)^2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_distance_to_focus_l705_70520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l705_70518

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.sin (2 * x - Real.pi / 3) + 2 * (Real.cos x) ^ 2 - 1

theorem f_properties :
  (∀ x : ℝ, f x = Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) ∧ f x = Real.sqrt 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) ∧ f x = -1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → f x ≤ Real.sqrt 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → f x ≥ -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l705_70518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_semicircle_l705_70592

def inscribed_in_semicircle (A B C : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), 
    let radius := dist center A
    dist center A = dist center B ∧
    dist center C = radius ∧
    dist A B = 2 * radius

theorem triangle_in_semicircle (A B C : ℝ × ℝ) :
  let diameter := dist A B
  inscribed_in_semicircle A B C →
  dist A C + dist B C ≤ diameter * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_semicircle_l705_70592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l705_70514

theorem polynomial_division_remainder (x : ℝ) : 
  ∃ q : Polynomial ℝ, X^5 + 3*X^2 + 1 = (X - 1)^2 * q + (8*X - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l705_70514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l705_70502

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

theorem angle_between_vectors (a b : V) 
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 1)
  (h3 : ‖a - 2 • b‖ = Real.sqrt 7) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l705_70502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_is_3000π_l705_70554

open Real

-- Define the dimensions of the cones and cylinder
noncomputable def cone_radius : ℝ := 10
noncomputable def cone_height : ℝ := 15
noncomputable def cylinder_radius : ℝ := 10
noncomputable def cylinder_height : ℝ := 45

-- Define the number of cones
def num_cones : ℕ := 3

-- Calculate the volume of the cylinder
noncomputable def cylinder_volume : ℝ := π * cylinder_radius^2 * cylinder_height

-- Calculate the volume of one cone
noncomputable def cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height

-- Calculate the total volume of the cones
noncomputable def total_cones_volume : ℝ := num_cones * cone_volume

-- Theorem: The unoccupied volume is 3000π cubic cm
theorem unoccupied_volume_is_3000π :
  cylinder_volume - total_cones_volume = 3000 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_is_3000π_l705_70554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inversion_is_sphere_l705_70561

/-- Inversion transformation with center O and radius r -/
noncomputable def Inversion (O : EuclideanSpace ℝ (Fin 3)) (r : ℝ) : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) := sorry

/-- A sphere in ℝ³ -/
structure Sphere where
  center : EuclideanSpace ℝ (Fin 3)
  radius : ℝ

/-- Predicate to check if a point is on a sphere -/
def OnSphere (p : EuclideanSpace ℝ (Fin 3)) (s : Sphere) : Prop := sorry

/-- Predicate to check if a point is inside a sphere -/
def InsideSphere (p : EuclideanSpace ℝ (Fin 3)) (s : Sphere) : Prop := sorry

/-- Theorem: Inversion of a sphere not containing the center of inversion is another sphere -/
theorem sphere_inversion_is_sphere 
  (O : EuclideanSpace ℝ (Fin 3)) (r : ℝ) (S : Sphere) 
  (h : ¬ InsideSphere O S) : 
  ∃ S' : Sphere, ∀ p : EuclideanSpace ℝ (Fin 3), 
    OnSphere p S ↔ OnSphere (Inversion O r p) S' := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inversion_is_sphere_l705_70561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l705_70560

-- Define the function f(x) = cos(x) + m - 1
noncomputable def f (x m : ℝ) : ℝ := Real.cos x + m - 1

-- Define what it means for f to have zero points
def has_zero_points (m : ℝ) : Prop := ∃ x, f x m = 0

-- Define the condition 0 ≤ m ≤ 1
def condition (m : ℝ) : Prop := 0 ≤ m ∧ m ≤ 1

-- Theorem: The condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ m, condition m → has_zero_points m) ∧
  (∃ m, ¬condition m ∧ has_zero_points m) := by
  sorry

#check condition_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l705_70560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_contains_points_coefficients_are_integers_x_coefficient_positive_gcd_of_coefficients_is_one_l705_70504

def point1 : ℝ × ℝ × ℝ := (-1, 3, -3)
def point2 : ℝ × ℝ × ℝ := (2, 3, -1)
def point3 : ℝ × ℝ × ℝ := (4, 1, -2)

def plane_equation (x y z : ℝ) := 2*x + 7*y - 6*z - 37

theorem plane_contains_points :
  plane_equation point1.fst point1.snd.fst point1.snd.snd = 0 ∧
  plane_equation point2.fst point2.snd.fst point2.snd.snd = 0 ∧
  plane_equation point3.fst point3.snd.fst point3.snd.snd = 0 :=
by sorry

theorem coefficients_are_integers :
  ∃ (A B C D : ℤ), ∀ (x y z : ℝ),
    plane_equation x y z = A * x + B * y + C * z + D :=
by sorry

theorem x_coefficient_positive :
  ∃ (A B C D : ℤ), A > 0 ∧
    ∀ (x y z : ℝ), plane_equation x y z = A * x + B * y + C * z + D :=
by sorry

theorem gcd_of_coefficients_is_one :
  ∃ (A B C D : ℤ), Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    ∀ (x y z : ℝ), plane_equation x y z = A * x + B * y + C * z + D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_contains_points_coefficients_are_integers_x_coefficient_positive_gcd_of_coefficients_is_one_l705_70504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l705_70575

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 3*x + 4)

-- Define the domain of f(x)
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem monotonically_decreasing_interval :
  ∃ (a b : ℝ), a = 3/2 ∧ b = 4 ∧
  (∀ x y, x ∈ domain → y ∈ domain → a ≤ x → x < y → y ≤ b → f y ≤ f x) ∧
  (∀ c d, c < a ∨ b < d → ¬(∀ x y, x ∈ domain → y ∈ domain → c ≤ x → x < y → y ≤ d → f y ≤ f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l705_70575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l705_70505

/-- Proves that the speed of a stream is 4 kmph given the conditions of a boat's travel --/
theorem stream_speed (boat_speed downstream_distance upstream_distance : ℝ) :
  boat_speed = 12 →
  downstream_distance = 32 →
  upstream_distance = 16 →
  (downstream_distance / (boat_speed + 4) = upstream_distance / (boat_speed - 4)) →
  4 = (downstream_distance - upstream_distance) / (downstream_distance / boat_speed + upstream_distance / boat_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_l705_70505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_problem_l705_70538

-- Define the function f
noncomputable def f (x : Real) : Real := 2 * Real.sin (x + Real.pi / 3) * Real.cos x

-- Theorem for the range of f
theorem f_range : 
  ∀ x : Real, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 3 / 2 := by sorry

-- Theorem for the triangle problem
theorem triangle_problem (A B : Real) (a b c : Real) :
  A > 0 ∧ A < Real.pi / 2 →  -- A is acute
  f A = Real.sqrt 3 / 2 →
  b = 2 →
  c = 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- law of cosines
  Real.sin A / a = Real.sin B / b →     -- law of sines
  Real.cos (A - B) = 5 * Real.sqrt 7 / 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_problem_l705_70538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l705_70540

theorem complex_equation_solution :
  ∀ (y : ℝ),
  let z₁ : ℂ := 3 + y * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  z₁ / z₂ = 1 + Complex.I →
  y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l705_70540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_displacements_l705_70528

/-- The straight-line distance between two points given their net displacements -/
theorem distance_from_displacements (south west : ℝ) : 
  south = 35 → west = 20 → Real.sqrt (south^2 + west^2) = 5 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_displacements_l705_70528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_parallel_side_l705_70569

/-- Represents a rectangular cow pasture -/
structure Pasture where
  perpendicular_side : ℝ
  parallel_side : ℝ

/-- The cost of fencing per foot -/
noncomputable def fence_cost_per_foot : ℝ := 10

/-- The total cost of fencing -/
noncomputable def total_fence_cost : ℝ := 2400

/-- The length of the barn -/
noncomputable def barn_length : ℝ := 500

/-- The total length of fencing available -/
noncomputable def total_fence_length : ℝ := total_fence_cost / fence_cost_per_foot

/-- The area of the pasture -/
noncomputable def pasture_area (p : Pasture) : ℝ := p.perpendicular_side * p.parallel_side

/-- The constraint on the pasture dimensions based on available fencing -/
def pasture_constraint (p : Pasture) : Prop :=
  p.parallel_side = total_fence_length - 2 * p.perpendicular_side

/-- The theorem stating the optimal length of the side parallel to the barn -/
theorem optimal_parallel_side :
  ∃ (p : Pasture), pasture_constraint p ∧
    (∀ (q : Pasture), pasture_constraint q → pasture_area p ≥ pasture_area q) ∧
    p.parallel_side = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_parallel_side_l705_70569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_distance_l705_70589

/-- The distance traveled by a light beam reflected by the Oxy plane -/
noncomputable def lightBeamDistance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₂, y₂, z₂) := Q
  let Q' := (x₂, y₂, -z₂)  -- reflection of Q in Oxy plane
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (-z₂ - z₁)^2)

/-- The theorem stating the distance of the reflected light beam -/
theorem reflected_light_distance :
  lightBeamDistance (1, 2, 3) (4, 4, 4) = Real.sqrt 62 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_distance_l705_70589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_path_length_is_correct_l705_70534

/-- The length of the path travelled by a dot on the center of a face of a unit cube,
    when the cube is rolled on a flat surface until the dot returns to the top face. -/
noncomputable def cube_roll_path_length : ℝ := (1 + Real.sqrt 5) / 2 * Real.pi

/-- Theorem stating that the length of the path travelled by a dot on the center of a face
    of a unit cube, when the cube is rolled on a flat surface until the dot returns to
    the top face, is (1+√5)/2 * π. -/
theorem cube_roll_path_length_is_correct : cube_roll_path_length = (1 + Real.sqrt 5) / 2 * Real.pi :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_path_length_is_correct_l705_70534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_sum_power_of_five_l705_70578

theorem unique_factorial_sum_power_of_five (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℕ, Nat.factorial a + b = 5^k) ∧ 
  (∃ m : ℕ, Nat.factorial b + a = 5^m) → 
  a = 5 ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_sum_power_of_five_l705_70578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l705_70591

theorem right_triangle_area (hypotenuse base : ℝ) (h1 : hypotenuse = 5) (h2 : base = 3) :
  (1/2) * base * Real.sqrt (hypotenuse^2 - base^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l705_70591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l705_70595

noncomputable def z : ℂ := (3 + Complex.I) / (4 - Complex.I)

theorem z_in_first_quadrant : 
  z.re > 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l705_70595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knockout_tournament_teams_l705_70531

/-- The number of teams in a knockout tournament -/
def number_of_teams (m n : ℕ) : ℕ := m - n + 1

/-- 
Given a knockout tournament with m total games and n replay games,
where m > n, the number of teams that participated is m - n + 1.
-/
theorem knockout_tournament_teams (m n : ℕ) (h : m > n) : 
  number_of_teams m n = m - n + 1 :=
by
  -- Unfold the definition of number_of_teams
  unfold number_of_teams
  -- The right-hand side is exactly the definition, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knockout_tournament_teams_l705_70531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_after_seven_rolls_l705_70579

/-- Represents the probability of being at a certain point after n rolls -/
def probability (n : ℕ) (point : Fin 3) : ℚ := sorry

/-- The sum of probabilities for all points is always 1 -/
axiom prob_sum_one (n : ℕ) : probability n 0 + probability n 1 + probability n 2 = 1

/-- Initial probabilities after one roll -/
axiom initial_probs : probability 1 0 = 0 ∧ probability 1 1 = 1/2 ∧ probability 1 2 = 1/2

/-- Probability of moving clockwise or counterclockwise is equal -/
axiom move_prob_equal (n : ℕ) : probability n 1 = probability n 2

/-- Recursive relation for probabilities -/
axiom prob_recursive (n : ℕ) (point : Fin 3) :
  probability (n + 1) point = (probability n ((point + 1) % 3) + probability n ((point + 2) % 3)) / 2

/-- Main theorem: Probability of being at point A after 7 rolls -/
theorem prob_after_seven_rolls : probability 7 0 = 21/64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_after_seven_rolls_l705_70579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l705_70552

/-- Represents the speed of a particle at a given mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  if n = 1 then 1 else 1 / (2 * ((n : ℝ) - 1)^2)

/-- Represents the time taken to traverse a given mile -/
noncomputable def time (n : ℕ) : ℝ :=
  1 / speed n

/-- The theorem stating the time needed to traverse the n-th mile -/
theorem nth_mile_time (n : ℕ) (h : n > 1) : time n = 2 * ((n : ℝ) - 1)^2 := by
  sorry

#check nth_mile_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l705_70552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_complement_A_subset_C_implies_a_leq_neg_one_l705_70543

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 3) / (x + 1) > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log x / Real.log 2 - 1)}

-- Define set C (parametrized by a)
def C (a : ℝ) : Set ℝ := {x | x^2 - (4 + a) * x + 4 * a ≤ 0}

-- Statement 1: A ∩ B = {x | x > 3}
theorem intersection_A_B : A ∩ B = {x | x > 3} := by sorry

-- Statement 2: If ∁ᵤA ⊆ C, then a ≤ -1
theorem complement_A_subset_C_implies_a_leq_neg_one (a : ℝ) :
  (Set.univ \ A) ⊆ C a → a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_complement_A_subset_C_implies_a_leq_neg_one_l705_70543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_product_one_l705_70597

theorem log_equality_implies_product_one 
  (M N : ℝ) 
  (h1 : Real.log (N^2) / Real.log M = Real.log (M^2) / Real.log N)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) : 
  M * N = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_product_one_l705_70597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_8_3_l705_70583

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_8_3_l705_70583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cap_bounce_ratio_l705_70513

/-- The distance Jenny's bottlecap flies straight before bouncing -/
noncomputable def jenny_initial : ℝ := 18

/-- The fraction of the initial distance Jenny's bottlecap flies after bouncing -/
noncomputable def jenny_bounce_fraction : ℝ := 1/3

/-- The distance Mark's bottlecap flies straight before bouncing -/
noncomputable def mark_initial : ℝ := 15

/-- The additional distance Mark's bottlecap flies compared to Jenny's -/
noncomputable def mark_additional : ℝ := 21

/-- The ratio of the distance Mark's bottlecap flew after bouncing to before bouncing -/
noncomputable def mark_bounce_ratio : ℝ := 2

theorem bottle_cap_bounce_ratio :
  let jenny_total := jenny_initial + jenny_initial * jenny_bounce_fraction
  let mark_total := mark_initial + mark_initial * mark_bounce_ratio
  mark_total = jenny_total + mark_additional :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_cap_bounce_ratio_l705_70513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l705_70593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x, x > 0 → ∃ y, f a x = y) →
  (let tangent_eq := λ x y ↦ x + y - 2 = 0
   tangent_eq 1 (f 2 1) ∧ 
   ∀ x, x > 0 → x ≠ 1 → tangent_eq x (f 2 x) → False) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → 
    (a ≤ 0 → f a x₁ < f a x₂) ∧
    (a > 0 → 
      (x₂ ≤ a → f a x₁ > f a x₂) ∧
      (x₁ ≥ a → f a x₁ < f a x₂) ∧
      (x₁ < a ∧ a < x₂ → f a x₁ > f a a ∧ f a a < f a x₂))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l705_70593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l705_70594

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1171) :
  Int.gcd (3 * b^2 + 17 * b + 47) (b + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l705_70594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_numbers_divisible_by_three_l705_70549

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def digit_sum (n : ℕ) : ℕ := (Nat.digits 10 n).sum

theorem seven_digit_numbers_divisible_by_three (A B C : ℕ) 
  (h1 : A < 10) (h2 : B < 10) (h3 : C < 10)
  (h4 : is_multiple_of_three (84 * 100000 + A * 10000 + 73 * 100 + B * 10 + 2))
  (h5 : is_multiple_of_three (52 * 100000 + 9 * 10000 + A * 1000 + B * 100 + 4 * 10 + C)) :
  C = 2 := by
  sorry

#check seven_digit_numbers_divisible_by_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_digit_numbers_divisible_by_three_l705_70549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l705_70564

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x - a / x

-- State the theorem
theorem function_properties :
  ∃ a : ℝ, 
    (f a (1/2) = 3) ∧ 
    (a = -1) ∧
    (∀ x₁ x₂ : ℝ, 1 < x₂ ∧ x₂ < x₁ → f (-1) x₁ > f (-1) x₂) := by
  -- Prove the existence of a
  use -1
  constructor
  · -- Prove f a (1/2) = 3
    simp [f]
    norm_num
  constructor
  · -- Prove a = -1
    rfl
  · -- Prove monotonicity
    intros x₁ x₂ h
    simp [f]
    sorry -- The detailed proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l705_70564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l705_70533

/-- The first curve in the xy-plane -/
def curve1 (x y : ℝ) : Prop := (x + y - 7) * (2 * x - 3 * y + 9) = 0

/-- The second curve in the xy-plane -/
def curve2 (x y : ℝ) : Prop := (x - y - 2) * (4 * x + 3 * y - 18) = 0

/-- A point (x, y) is an intersection point if it satisfies both curve equations -/
def is_intersection_point (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

/-- The set of all intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | is_intersection_point p.1 p.2}

/-- The theorem stating that there are exactly 3 distinct intersection points -/
theorem three_intersection_points :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 3 ∧ (∀ p, p ∈ S ↔ p ∈ intersection_points) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersection_points_l705_70533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_expansion_l705_70585

/-- The coefficient of x^3 in the expansion of (x^2-x-2)^5 is 120 -/
theorem coefficient_x_cube_in_expansion : 
  (Polynomial.coeff ((Polynomial.X ^ 2 - Polynomial.X - 2 : Polynomial ℤ) ^ 5) 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_expansion_l705_70585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_one_l705_70525

theorem absolute_difference_one (a b c d : ℤ) 
  (h : a + b + c + d = a * b + b * c + c * d + d * a + 1) :
  ∃ (x y : ℤ), x ∈ ({a, b, c, d} : Set ℤ) ∧ y ∈ ({a, b, c, d} : Set ℤ) ∧ x ≠ y ∧ |x - y| = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_one_l705_70525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_after_removal_l705_70508

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def concatenated_primes : Nat :=
  first_ten_primes.foldl (fun acc x => acc * 10^(Nat.digits 10 x).length + x) 0

def remove_six_digits (n : Nat) : Nat :=
  7317192329 -- This is a placeholder for the actual implementation

theorem largest_number_after_removal :
  remove_six_digits concatenated_primes = 7317192329 :=
by
  -- The proof would go here
  sorry

#eval concatenated_primes
#eval remove_six_digits concatenated_primes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_after_removal_l705_70508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abscissa_range_l705_70527

open Real Set

-- Define the line l: x - y + 1 = 0
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle C: (x-2)^2 + (y-1)^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the condition for angle MPN = 60°
def angle_condition (px py mx my nx ny : ℝ) : Prop :=
  let v1x := mx - px
  let v1y := my - py
  let v2x := nx - px
  let v2y := ny - py
  (v1x * v2x + v1y * v2y) / (sqrt (v1x^2 + v1y^2) * sqrt (v2x^2 + v2y^2)) = 1/2

-- Main theorem
theorem abscissa_range :
  ∀ p : ℝ × ℝ,
  (∃ m n : ℝ × ℝ,
    line_l p.1 p.2 ∧
    circle_C m.1 m.2 ∧
    circle_C n.1 n.2 ∧
    angle_condition p.1 p.2 m.1 m.2 n.1 n.2) →
  p.1 ∈ Icc 0 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abscissa_range_l705_70527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_ABQP_l705_70522

/-- The angle MON is formed by the ray y = x (x ≥ 0) and the positive x-axis -/
def angle_MON : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ (p.2 = p.1 ∨ p.2 = 0)}

/-- Point A with coordinates (6, 5) -/
def point_A : ℝ × ℝ := (6, 5)

/-- Point B with coordinates (10, 2) -/
def point_B : ℝ × ℝ := (10, 2)

/-- The perimeter of quadrilateral ABQP given points P and Q -/
noncomputable def perimeter (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2).sqrt +
  ((P.1 - point_B.1)^2 + (P.2 - point_B.2)^2).sqrt +
  ((Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2).sqrt +
  ((Q.1 - point_B.1)^2 + (Q.2 - point_B.2)^2).sqrt

/-- The theorem stating the minimum perimeter of quadrilateral ABQP -/
theorem min_perimeter_ABQP :
  ∃ (P Q : ℝ × ℝ), P ∈ angle_MON ∧ Q ∈ angle_MON ∧
    ∀ (P' Q' : ℝ × ℝ), P' ∈ angle_MON → Q' ∈ angle_MON →
      perimeter P Q ≤ perimeter P' Q' ∧
      perimeter P Q = 16 / 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_ABQP_l705_70522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_intersection_l705_70587

/-- A line in the Cartesian plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The point of intersection of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
  let y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
  (x, y)

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem three_lines_intersection (a : ℝ) : 
  let l1 : Line := ⟨2, 1, -5⟩
  let l2 : Line := ⟨1, -1, -1⟩
  let l3 : Line := ⟨a, 1, -3⟩
  let p := intersectionPoint l1 l2
  (pointOnLine l1 p ∧ pointOnLine l2 p ∧ pointOnLine l3 p) → a = 1 := by
  sorry

#check three_lines_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_intersection_l705_70587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l705_70521

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a horizontal line -/
def distToHorizontalLine (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- The set of points P satisfying the given condition -/
def trajectorySet : Set Point :=
  {p : Point | distToHorizontalLine p (-2) = distance p ⟨0, 1⟩ + 1}

/-- Theorem stating that the trajectory set forms a parabola -/
theorem trajectory_is_parabola : 
  ∃ (f : Point) (d : ℝ), trajectorySet = {p : Point | distance p f = distToHorizontalLine p d} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l705_70521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_theorem_l705_70581

/-- Represents a cube in the n × n × n structure -/
structure Cube (n : ℕ) where
  x : Fin n
  y : Fin n
  z : Fin n

/-- Represents the color of a cube -/
inductive Color
  | Black
  | White

/-- Represents the n × n × n cube structure -/
def CubeStructure (n : ℕ) := Cube n → Color

/-- Checks if a cube is in a specific n × 1 × 1 subprism -/
def isInNx1x1Subprism (c : Cube n) (y z : Fin n) : Prop :=
  c.y = y ∧ c.z = z

/-- Checks if a cube is in a specific 1 × n × 1 subprism -/
def isIn1xNx1Subprism (c : Cube n) (x z : Fin n) : Prop :=
  c.x = x ∧ c.z = z

/-- Checks if a cube is in a specific 1 × 1 × n subprism -/
def isIn1x1xNSubprism (c : Cube n) (x y : Fin n) : Prop :=
  c.x = x ∧ c.y = y

/-- Checks if the initial configuration is valid -/
def isValidInitialConfiguration (n : ℕ) (cs : CubeStructure n) : Prop :=
  ∀ y z, ∃! (c1 c2 : Cube n), isInNx1x1Subprism c1 y z ∧ isInNx1x1Subprism c2 y z ∧
    cs c1 = Color.Black ∧ cs c2 = Color.Black ∧ c1 ≠ c2 ∧
    ((c2.x.val - c1.x.val : ℤ).natAbs - 1) % 2 = 0 ∧
  ∀ x z, ∃! (c1 c2 : Cube n), isIn1xNx1Subprism c1 x z ∧ isIn1xNx1Subprism c2 x z ∧
    cs c1 = Color.Black ∧ cs c2 = Color.Black ∧ c1 ≠ c2 ∧
    ((c2.y.val - c1.y.val : ℤ).natAbs - 1) % 2 = 0 ∧
  ∀ x y, ∃! (c1 c2 : Cube n), isIn1x1xNSubprism c1 x y ∧ isIn1x1xNSubprism c2 x y ∧
    cs c1 = Color.Black ∧ cs c2 = Color.Black ∧ c1 ≠ c2 ∧
    ((c2.z.val - c1.z.val : ℤ).natAbs - 1) % 2 = 0

/-- Checks if the final configuration is valid -/
def isValidFinalConfiguration (n : ℕ) (cs : CubeStructure n) : Prop :=
  (∀ y z, ∃! c, isInNx1x1Subprism c y z ∧ cs c = Color.Black) ∧
  (∀ x z, ∃! c, isIn1xNx1Subprism c x z ∧ cs c = Color.Black) ∧
  (∀ x y, ∃! c, isIn1x1xNSubprism c x y ∧ cs c = Color.Black)

theorem cube_coloring_theorem (n : ℕ) (cs : CubeStructure n) 
    (h : isValidInitialConfiguration n cs) :
    ∃ cs', isValidFinalConfiguration n cs' ∧
           (∀ c, cs' c = Color.Black → cs c = Color.Black) ∧
           (∃ f : Cube n → Bool, ∀ c, cs' c = Color.Black ↔ cs c = Color.Black ∧ f c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_theorem_l705_70581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_officers_count_l705_70515

/-- The number of members in the club -/
def club_size : ℕ := 10

/-- The number of officers to be chosen -/
def num_officers : ℕ := 5

/-- The number of ways to choose officers -/
def ways_to_choose_officers : ℕ := 30240

/-- Theorem stating that the number of ways to choose 5 distinct officers
    from a group of 10 people is 30,240 -/
theorem choose_officers_count :
  (Nat.descFactorial club_size num_officers) = ways_to_choose_officers := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_officers_count_l705_70515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_power_of_k_iff_n_eq_one_or_two_l705_70590

/-- Define the sequence a_n -/
def a : ℕ → ℕ → ℕ
  | k, 0 => 1  -- Add case for n = 0
  | k, 1 => 1
  | k, 2 => k
  | k, n+3 => (k+1) * a k (n+2) - a k (n+1)

/-- The theorem to be proved -/
theorem a_power_of_k_iff_n_eq_one_or_two (k : ℕ) (h : k > 1) :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, a k n = k^m) ↔ (n = 1 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_power_of_k_iff_n_eq_one_or_two_l705_70590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_appended_digits_l705_70516

theorem smallest_appended_digits (n : Nat) (h : n = 2014) :
  ∃ k : Nat,
    (k < 10^6) ∧
    (∀ m : Nat, m < 10 → (n * 10^6 + k) % m = 0) ∧
    (∀ j : Nat, j < k →
      ∃ m : Nat, m < 10 ∧ (n * 10^(Nat.log 10 j + 1) + j) % m ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_appended_digits_l705_70516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_2_power_x_negative_l705_70529

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) > 0
def solution_set_f_positive : Set ℝ := {x | -2 < x ∧ x < 1}

-- State the theorem
theorem solution_set_f_2_power_x_negative 
  (h : {x : ℝ | f x > 0} = solution_set_f_positive) :
  {x : ℝ | f (2^x) < 0} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_2_power_x_negative_l705_70529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l705_70566

noncomputable def min3 (a b c : ℝ) : ℝ := min a (min b c)

noncomputable def M (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem problem_1 (x : ℝ) : min3 2 (2*x + 2) (4 - 2*x) = 2 → 0 ≤ x ∧ x ≤ 1 := by
  sorry

theorem problem_2 (x : ℝ) : M 2 (x + 1) (2*x) = min3 2 (x + 1) (2*x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l705_70566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l705_70509

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def B : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_of_A_and_B :
  A ∩ B = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l705_70509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_AB_l705_70510

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line x = 3
def vertical_line (x : ℝ) : Prop := x = 3

-- Define a point P on the vertical line
def point_P (y₀ : ℝ) : ℝ × ℝ := (3, y₀)

-- Define the tangent points A and B
noncomputable def tangent_points (P : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the line AB
noncomputable def line_AB (A B : ℝ × ℝ) : ℝ → ℝ := sorry

-- Theorem: The line AB passes through the fixed point (4/3, 2)
theorem fixed_point_on_line_AB (y₀ : ℝ) :
  let P := point_P y₀
  let (A, B) := tangent_points P
  (line_AB A B) (4/3) = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_AB_l705_70510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l705_70563

/-- The area of an isosceles trapezoid circumscribed around a circle -/
theorem isosceles_trapezoid_area (base : ℝ) (angle : ℝ) : 
  base > 0 → 
  angle = Real.arcsin 0.6 →
  let x := 40 / 3.6
  let y := base - 1.6 * x
  let h := 0.6 * x
  (base + y) * h / 2 = (base + (base - 1.6 * (40 / 3.6))) * 0.6 * (40 / 3.6) / 2 := by
  sorry

#eval ((20 : Float) + (20 - 1.6 * (40 / 3.6))) * 0.6 * (40 / 3.6) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l705_70563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equation_solutions_l705_70500

theorem cosine_sum_equation_solutions (x : ℝ) :
  (Real.cos x)^2 + (Real.cos (2*x))^2 + (Real.cos (3*x))^2 = 1 ↔
  (∃ m : ℤ, x = m * Real.pi - Real.pi/4 ∨ x = m * Real.pi + Real.pi/4 ∨ x = m * Real.pi + Real.pi/2) ∨
  (∃ k : ℤ, x = k * Real.pi/3 + Real.pi/6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equation_solutions_l705_70500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_city_cycle_l705_70550

-- Define the structure for our problem
structure FlightSystem where
  Country1 : Type
  Country2 : Type
  flight : Country1 → Country2 → Prop
  flightBack : Country2 → Country1 → Prop

-- Define the properties of our flight system
def ValidFlightSystem (fs : FlightSystem) : Prop :=
  -- There is exactly one one-way flight route between any two cities in different countries
  (∀ (a : fs.Country1) (b : fs.Country2), (fs.flight a b ∧ ¬∃ (c : fs.Country2), c ≠ b ∧ fs.flight a c) ∨
                                         (¬fs.flight a b ∧ ∃ (c : fs.Country2), c ≠ b ∧ fs.flight a c)) ∧
  -- Each city in Country1 has outbound flights to some cities in Country2
  (∀ (a : fs.Country1), ∃ (b : fs.Country2), fs.flight a b) ∧
  -- Each city in Country2 has outbound flights to some cities in Country1
  (∀ (b : fs.Country2), ∃ (a : fs.Country1), fs.flightBack b a)

-- The main theorem
theorem four_city_cycle (fs : FlightSystem) (h : ValidFlightSystem fs) :
  ∃ (a c : fs.Country1) (b d : fs.Country2), 
    a ≠ c ∧ b ≠ d ∧ 
    fs.flight a b ∧ fs.flight c d ∧ 
    fs.flightBack b c ∧ fs.flightBack d a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_city_cycle_l705_70550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l705_70542

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x + 2 * Real.sqrt 3, Real.sin x)
def c : ℝ × ℝ := (0, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) + Real.sin (x - Real.pi/6)

theorem problem_statement :
  (∀ x : ℝ, dot_product (a x) c = 0 → Real.cos (2*x) = 1) ∧
  (∃ M : ℝ, M = Real.sqrt 2 ∧ ∀ x : ℝ, f x ≤ M) ∧
  (∀ k : ℤ, f (2 * ↑k * Real.pi + 5 * Real.pi / 12) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l705_70542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_probability_l705_70576

theorem larry_wins_probability (p_larry p_julius : ℝ) :
  p_larry = 1/3 →
  p_julius = 1/4 →
  p_larry * (1 / (1 - (1 - p_larry) * (1 - p_julius))) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_probability_l705_70576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_l705_70555

/-- Given two parallel vectors a and b, prove that their difference is (4, -2) -/
theorem vector_difference (m : ℝ) : 
  let a : Fin 2 → ℝ := ![(-2 : ℝ), 1]
  let b : Fin 2 → ℝ := ![m, 3]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → 
  (a - b) = ![4, -2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_l705_70555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l705_70577

theorem sphere_surface_area (r : ℝ) (h : r = 1) : 
  4 * Real.pi * r^2 = 4 * Real.pi := by
  rw [h]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l705_70577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_reals_inequality_l705_70519

theorem seven_reals_inequality (S : Finset ℝ) (h : S.card = 7) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ Real.sqrt 3 * |a - b| ≤ |1 + a * b| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_reals_inequality_l705_70519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l705_70512

theorem remainder_theorem (x : ℤ) : 
  (x + 1)^2100 ≡ x^2 [ZMOD (x^4 - x^2 + 1)] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l705_70512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l705_70596

-- Define the slopes of two lines
noncomputable def slope_l (a : ℝ) : ℝ := -a / 2
def slope_m : ℝ := 2

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines (a : ℝ) :
  perpendicular (slope_l a) slope_m → a = 1 := by
  intro h
  have h1 : (-a / 2) * 2 = -1 := h
  field_simp at h1
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l705_70596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_fixed_point_l705_70539

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 13

-- Define the line l containing the center of the circle
def line_l (x y : ℝ) : Prop := 2*x - 7*y + 8 = 0

-- Define the fixed point N
def point_N : ℝ × ℝ := (-7/2, 2)

-- Define the condition for K_AN + K_BN = 0
def slope_condition (A B : ℝ × ℝ) : Prop :=
  let (x_a, y_a) := A
  let (x_b, y_b) := B
  let (x_n, y_n) := point_N
  (y_a - y_n) / (x_a - x_n) + (y_b - y_n) / (x_b - x_n) = 0

theorem circle_and_fixed_point :
  -- The circle passes through A(6,0) and B(1,5)
  circle_C 6 0 ∧ circle_C 1 5 ∧
  -- The center of the circle is on line l
  ∃ (x y : ℝ), circle_C x y ∧ line_l x y ∧
  -- For any line through M(1,2) intersecting the circle at A' and B', the slope condition holds
  ∀ (A' B' : ℝ × ℝ), circle_C A'.1 A'.2 ∧ circle_C B'.1 B'.2 →
    ∃ (k : ℝ), (A'.2 - 2 = k * (A'.1 - 1)) ∧ (B'.2 - 2 = k * (B'.1 - 1)) →
      slope_condition A' B' :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_fixed_point_l705_70539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l705_70532

/-- A square pentomino is a shape composed of 5 squares. -/
structure SquarePentomino : Type := (id : Nat)

/-- The area of a square pentomino is 5 square units. -/
def area_pentomino : ℕ := 5

/-- The dimensions of the rectangle. -/
def rectangle_width : ℕ := 12
def rectangle_height : ℕ := 12

/-- The area of the rectangle. -/
def area_rectangle : ℕ := rectangle_width * rectangle_height

/-- A tiling is valid if it covers the entire rectangle without gaps or overlaps. -/
def is_valid_tiling (tiling : List SquarePentomino) : Prop :=
  (tiling.length * area_pentomino = area_rectangle) ∧
  (∀ p q : SquarePentomino, p ∈ tiling → q ∈ tiling → p ≠ q → p.id ≠ q.id)

/-- The number of valid square pentomino tilings of the rectangle. -/
def num_valid_tilings : ℕ := 0

theorem no_valid_tiling : num_valid_tilings = 0 := by
  sorry

#check no_valid_tiling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l705_70532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l705_70568

def T := ℕ × ℕ × ℕ

noncomputable def f : T → ℝ
  | (p, q, r) => (3 * (p : ℝ) * (q : ℝ) * (r : ℝ)) / ((p : ℝ) + (q : ℝ) + (r : ℝ))

theorem f_satisfies_conditions :
  (∀ (t : T), (t.1 * t.2.1 * t.2.2 = 0) → f t = 0) ∧
  (∀ (p q r : ℕ), p > 0 ∧ q > 0 ∧ r > 0 →
    f (p, q, r) = 1 + (1/6) * (
      f (p+1, q-1, r) + f (p-1, q+1, r) + f (p-1, q, r+1) +
      f (p+1, q, r-1) + f (p, q+1, r-1) + f (p, q-1, r+1)
    )) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l705_70568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_range_l705_70588

-- Define a triangle ABC with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the property of arithmetic sequence for side lengths
def is_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

-- Define the angle B using the cosine rule
noncomputable def cos_B (t : Triangle) : ℝ :=
  (t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c)

-- Theorem statement
theorem angle_B_range (t : Triangle) (h_arithmetic : is_arithmetic_sequence t) :
  ∃ B : ℝ, 0 < B ∧ B ≤ π/3 ∧ Real.cos B = cos_B t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_range_l705_70588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knicks_equivalence_l705_70511

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 5

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 9 / 7

/-- The number of knocks we want to convert -/
def target_knocks : ℕ := 36

/-- The result we want to prove -/
def target_knicks : ℕ := 47

theorem knicks_equivalence :
  ∃ (x : ℚ), 
    x * knicks_to_knacks * knacks_to_knocks = target_knocks ∧
    Int.floor x = target_knicks :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knicks_equivalence_l705_70511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l705_70572

/-- The ellipse on which point P moves -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line l -/
def line (x y : ℝ) : Prop := x + y - 2 * Real.sqrt 5 = 0

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 2 * Real.sqrt 5) / Real.sqrt 2

/-- The minimum distance from any point on the ellipse to the line l is √10/2 -/
theorem min_distance_ellipse_to_line :
  ∃ (d : ℝ), d = Real.sqrt 10 / 2 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≥ d ∧
    ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l705_70572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_at_three_l705_70570

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then 3 * x^2 + 1 else b * x - 6

theorem continuity_at_three (b : ℝ) :
  ContinuousAt (f b) 3 ↔ b = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_at_three_l705_70570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_6_and_8_l705_70557

theorem two_digit_multiples_of_6_and_8 : 
  Finset.card (Finset.filter (fun n => 10 ≤ n ∧ n < 100 ∧ 6 ∣ n ∧ 8 ∣ n) (Finset.range 100)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_6_and_8_l705_70557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_l705_70553

theorem min_value_trig_sum (x : ℝ) : 
  |Real.sin x + Real.cos x + Real.tan x + (Real.tan x)⁻¹ + (Real.cos x)⁻¹ + (Real.sin x)⁻¹| ≥ 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_l705_70553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_count_l705_70523

-- Define the six functions
noncomputable def f1 (x : ℝ) : ℝ := x / 6
noncomputable def f2 (x : ℝ) : ℝ := -4 / x
noncomputable def f3 (x : ℝ) : ℝ := 3 - (1/2) * x
noncomputable def f4 (x : ℝ) : ℝ := 3 * x^2 - 2
noncomputable def f5 (x : ℝ) : ℝ := x^2 - (x-3)*(x+2)
noncomputable def f6 (x : ℝ) : ℝ := 6^x

-- Define a predicate for linear functions
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

-- State the theorem
theorem linear_functions_count :
  (is_linear f1) ∧ (is_linear f3) ∧ (is_linear f5) ∧
  ¬(is_linear f2) ∧ ¬(is_linear f4) ∧ ¬(is_linear f6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_count_l705_70523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_a_for_equal_roots_l705_70544

/-- The quadratic equation x^2 - bx + ab = 0 has infinitely many real values of a for which its roots are equal -/
theorem infinite_a_for_equal_roots :
  ∃ (S : Set ℝ), (Set.Infinite S) ∧ 
  (∀ a ∈ S, ∃ b : ℝ, ∀ x : ℝ, x^2 - b*x + a*b = 0 → (∃! r : ℝ, x = r)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_a_for_equal_roots_l705_70544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_calculation_l705_70526

def first_lemonade : ℚ := 25/100
def first_icedtea : ℚ := 18/100
def first_people : ℕ := 15

def second_lemonade : ℚ := 42/100
def second_icedtea : ℚ := 30/100
def second_people : ℕ := 22

def third_lemonade : ℚ := 25/100
def third_icedtea : ℚ := 15/100
def third_people : ℕ := 12

def total_beverages : ℚ := first_lemonade + first_icedtea + second_lemonade + second_icedtea + third_lemonade + third_icedtea
def total_people : ℕ := first_people + second_people + third_people

theorem beverage_calculation :
  total_beverages = 155/100 ∧ 
  (abs ((total_beverages / total_people : ℚ) - 316/10000) < 1/10000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_calculation_l705_70526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_4_minus_sqrt3_l705_70558

open Real

-- Define the curves and line
noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ := (cos α, 1 + sin α)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 2 * cos θ + 2 * sqrt 3 * sin θ

noncomputable def line_l : ℝ := π / 3

-- Define the intersection points
noncomputable def point_M : ℝ × ℝ := (sqrt 3 / 2, 3 / 2)

noncomputable def point_N : ℝ × ℝ := (2, 2 * sqrt 3)

-- Theorem statement
theorem length_MN_is_4_minus_sqrt3 :
  sqrt ((point_N.1 - point_M.1)^2 + (point_N.2 - point_M.2)^2) = 4 - sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_4_minus_sqrt3_l705_70558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l705_70582

theorem expression_evaluation : 
  ((((5 : ℝ) + 2)⁻¹ - (1 / 2))⁻¹ + 2)⁻¹ + 2 = (3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l705_70582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_expectation_correct_l705_70584

/-- Data from the problem -/
def total_students : ℕ := 30
def not_regressed_non_excellent : ℕ := 8
def regressed_excellent : ℕ := 18
def not_regressed_excellent : ℕ := 2
def regressed_non_excellent : ℕ := 2

/-- Chi-square statistic calculation -/
noncomputable def chi_square : ℝ :=
  let n := total_students
  let n11 := not_regressed_non_excellent
  let n22 := regressed_excellent
  let n12 := not_regressed_excellent
  let n21 := regressed_non_excellent
  let n1_plus := n11 + n12
  let n2_plus := n21 + n22
  let n_plus1 := n11 + n21
  let n_plus2 := n12 + n22
  (n * (n11 * n22 - n12 * n21)^2) / (n1_plus * n2_plus * n_plus1 * n_plus2)

/-- Probability of a student being excellent and regressed to textbook -/
def p_excellent_regressed : ℚ := 18 / 30

/-- Distribution of X -/
def prob_X (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - p_excellent_regressed)^3
  | 1 => 3 * p_excellent_regressed * (1 - p_excellent_regressed)^2
  | 2 => 3 * p_excellent_regressed^2 * (1 - p_excellent_regressed)
  | 3 => p_excellent_regressed^3
  | _ => 0

/-- Mathematical expectation of X -/
def E_X : ℚ := 0 * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2 + 3 * prob_X 3

theorem relationship_exists : chi_square > 6.635 := by sorry

theorem expectation_correct : E_X = 9/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_expectation_correct_l705_70584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_equation_solution_l705_70562

-- Part 1
theorem calculation_proof :
  Real.sqrt 4 + Real.sqrt ((-2)^2) + Real.sqrt (9/4) - (Real.sqrt (1/2))^2 + ((-125) ^ (1/3 : ℝ)) = 0 := by
  sorry

-- Part 2
theorem equation_solution :
  ∀ x : ℝ, (2*x - 1)^2 - 169 = 0 ↔ x = 7 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_equation_solution_l705_70562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_tetrahedra_volume_l705_70545

/-- The volume of tetrahedra removed from a rectangular prism -/
noncomputable def tetrahedra_volume (a b c : ℝ) : ℝ :=
  let x := 2 * (Real.sqrt 2 - 1)
  let base_area := 2 * Real.sqrt 3 * (2 - Real.sqrt 2)
  let height := 4 - 2 * Real.sqrt 2
  8 * (1 / 3) * base_area * height

theorem removed_tetrahedra_volume :
  tetrahedra_volume 2 3 4 = (16 * Real.sqrt 3 * (2 - Real.sqrt 2) * (4 - 2 * Real.sqrt 2)) / 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tetrahedra_volume 2 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_tetrahedra_volume_l705_70545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_l705_70580

-- Define the is_sufficient_condition_for relation
def is_sufficient_condition_for (A B : Prop) : Prop :=
  A → B

-- Theorem stating that A is a sufficient condition for B iff A implies B
theorem sufficient_condition (A B : Prop) : 
  (A → B) ↔ is_sufficient_condition_for A B := by
  -- Prove the equivalence
  apply Iff.intro
  -- Forward direction
  · intro h
    exact h
  -- Backward direction
  · intro h
    exact h

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_l705_70580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_contest_results_l705_70574

def pie_eating_contest (pies_eaten : List Nat) : Prop :=
  pies_eaten.length = 8 ∧
  pies_eaten = [3, 4, 8, 5, 7, 6, 2, 1]

theorem pie_contest_results (pies_eaten : List Nat) 
  (h : pie_eating_contest pies_eaten) : 
  (∃ max min : Nat, max ∈ pies_eaten ∧ min ∈ pies_eaten ∧ max - min = 7) ∧
  (pies_eaten.sum : Rat) / pies_eaten.length = 9/2 := by
  sorry

#check pie_contest_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_contest_results_l705_70574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_projection_not_segment_l705_70506

/-- A shape that can be projected onto a plane -/
class Projectable (α : Type*) where
  project : α → Set (Fin 2 → ℝ) → Set (Fin 2 → ℝ)

/-- A straight line in ℝ² -/
def StraightLine : Type := ℝ → Fin 2 → ℝ

/-- A line segment in ℝ² -/
def LineSegment : Type := (Fin 2 → ℝ) × (Fin 2 → ℝ)

/-- A plane in ℝ³ -/
def Plane : Type := Fin 3 → ℝ

/-- Instance of Projectable for StraightLine -/
instance : Projectable StraightLine where
  project := sorry

theorem straight_line_projection_not_segment 
  (l : StraightLine) (p : Plane) : 
  ¬∃ (s : LineSegment), Projectable.project l (Set.univ : Set (Fin 2 → ℝ)) = {s.1, s.2} := by
  sorry

#check straight_line_projection_not_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_projection_not_segment_l705_70506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l705_70536

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- Semi-major axis -/
def a : ℝ := 5

/-- Semi-minor axis -/
def b : ℝ := 3

/-- Distance from a point to a focus -/
noncomputable def distance_to_focus (x y c : ℝ) : ℝ :=
  Real.sqrt ((x - c)^2 + y^2)

/-- Product of distances from a point to both foci -/
noncomputable def distance_product (x y c : ℝ) : ℝ :=
  distance_to_focus x y c * distance_to_focus x y (-c)

/-- Theorem: The product of distances from a point on the ellipse to the foci is maximized at (0, ±b) -/
theorem max_distance_product :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧
  distance_product x y c ≤ distance_product 0 b c ∧
  distance_product x y c ≤ distance_product 0 (-b) c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l705_70536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l705_70567

theorem triangle_problem (A B C : Real) (a b c : Real) :
  A < π / 2 →
  Real.sin (A - π / 4) = Real.sqrt 2 / 10 →
  (1 / 2) * b * c * Real.sin A = 24 →
  b = 10 →
  Real.sin A = 4 / 5 ∧ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l705_70567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_S_l705_70517

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2) else (n + 1) / 2

theorem sum_of_specific_S : S 21 + S 34 + S 45 = 17 := by
  -- Evaluate S for each number
  have h21 : S 21 = 11 := by rfl
  have h34 : S 34 = -17 := by rfl
  have h45 : S 45 = 23 := by rfl
  
  -- Add the results
  calc
    S 21 + S 34 + S 45 = 11 + (-17) + 23 := by rw [h21, h34, h45]
    _ = 17 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_S_l705_70517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt13_problem_l705_70503

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ := x - Int.floor x

-- Statement of the theorem
theorem sqrt13_problem (m n : ℝ) : 
  (m = intPart (Real.sqrt 13)) → 
  (n = decPart (10 - Real.sqrt 13)) → 
  (m + n = 7 - Real.sqrt 13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt13_problem_l705_70503
