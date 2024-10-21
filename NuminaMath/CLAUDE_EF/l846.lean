import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_orientation_theorem_l846_84656

/-- Represents a point on the surface of a cube -/
structure CubePoint where
  x : Real
  y : Real
  z : Real
  -- Ensure the point is on the surface of a unit cube
  surface : (x = 0 ∨ x = 1) ∨ (y = 0 ∨ y = 1) ∨ (z = 0 ∨ z = 1)

/-- Represents an orientation of a cube -/
structure CubeOrientation where
  -- Simplified representation using three angles
  angleX : Real
  angleY : Real
  angleZ : Real

/-- Function to determine which points touch the surface for a given orientation -/
def touchingSurface (points : Finset CubePoint) (orientation : CubeOrientation) : Finset CubePoint :=
  sorry

theorem cube_orientation_theorem (points : Finset CubePoint) :
  points.card = 100 →
  ∃ (o1 o2 : CubeOrientation), o1 ≠ o2 ∧ touchingSurface points o1 ≠ touchingSurface points o2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_orientation_theorem_l846_84656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fib_representation_l846_84651

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Predicate to check if a list of indices satisfies the required conditions -/
def valid_indices (ks : List ℕ) : Prop :=
  ks.length > 0 ∧
  (∀ i, i + 1 < ks.length → ks.get! i > ks.get! (i+1) + 1) ∧
  ks.get! (ks.length - 1) > 1

/-- Sum of Fibonacci numbers at given indices -/
def fib_sum (ks : List ℕ) : ℕ :=
  ks.foldl (λ acc k => acc + fib k) 0

/-- Theorem: Unique representation of natural numbers as sum of Fibonacci numbers -/
theorem unique_fib_representation (n : ℕ) :
  ∃! ks : List ℕ, valid_indices ks ∧ fib_sum ks = n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fib_representation_l846_84651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelism_transitivity_perpendicular_parallel_implication_perpendicular_line_plane_implication_all_statements_correct_l846_84682

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (lies_on : Line → Plane → Prop)

-- Statement 1
theorem parallelism_transitivity (α β γ : Plane) :
  parallel α β → parallel β γ → parallel γ α := by
  sorry

-- Statement 2
theorem perpendicular_parallel_implication (α β γ : Plane) (l : Line) :
  perpendicular_plane l α → parallel β γ → perpendicular_plane l β := by
  sorry

-- Statement 3
theorem perpendicular_line_plane_implication (m n : Line) (β : Plane) :
  perpendicular_plane m β → perpendicular m n → ¬lies_on n β → parallel_line_plane n β := by
  sorry

-- All statements are correct
theorem all_statements_correct :
  (∀ α β γ : Plane, parallel α β → parallel β γ → parallel γ α) ∧
  (∀ α β γ : Plane, ∀ l : Line, perpendicular_plane l α → parallel β γ → perpendicular_plane l β) ∧
  (∀ m n : Line, ∀ β : Plane, perpendicular_plane m β → perpendicular m n → ¬lies_on n β → parallel_line_plane n β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelism_transitivity_perpendicular_parallel_implication_perpendicular_line_plane_implication_all_statements_correct_l846_84682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_depth_calculation_l846_84669

/-- Represents the cross-section of a tunnel -/
structure TunnelCrossSection where
  topWidth : ℝ
  bottomWidth : ℝ
  area : ℝ

/-- Calculates the depth of a tunnel given its cross-section -/
noncomputable def tunnelDepth (t : TunnelCrossSection) : ℝ :=
  (2 * t.area) / (t.topWidth + t.bottomWidth)

theorem tunnel_depth_calculation (t : TunnelCrossSection) 
  (h1 : t.topWidth = 15)
  (h2 : t.bottomWidth = 5)
  (h3 : t.area = 400) :
  tunnelDepth t = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_depth_calculation_l846_84669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_with_tan_l846_84607

theorem sin_double_angle_with_tan (x : ℝ) (h : Real.tan x = 1/3) : Real.sin (2*x) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_with_tan_l846_84607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l846_84687

/-- A normally distributed random variable -/
structure NormalRV (μ : ℝ) (σ : ℝ) where
  X : ℝ → ℝ  -- The random variable as a function

/-- Probability measure for the normal distribution -/
noncomputable def P (X : ℝ → ℝ) : Set ℝ → ℝ := sorry

/-- Theorem: For a normally distributed random variable X with 
    P(X > 5) = P(X < -1) = 0.2, we have P(2 < X < 5) = 0.3 -/
theorem normal_distribution_probability 
  (μ σ : ℝ) (X : NormalRV μ σ) 
  (h1 : P X.X {x | x > 5} = 0.2) 
  (h2 : P X.X {x | x < -1} = 0.2) : 
  P X.X {x | 2 < x ∧ x < 5} = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l846_84687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l846_84612

/-- The curve C in the xy-plane -/
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- The line l in the xy-plane -/
noncomputable def l (x y : ℝ) : Prop := y = -Real.sqrt 3 * x + 5

/-- The point D -/
noncomputable def D : ℝ × ℝ := (-Real.sqrt 3 / 2, 1 / 2)

/-- Distance function from a point to a line -/
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  |y + Real.sqrt 3 * x - 5| / Real.sqrt 4

theorem max_distance_point :
  C D.1 D.2 ∧
  ∀ (x y : ℝ), C x y → distanceToLine x y ≤ distanceToLine D.1 D.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l846_84612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_inscribed_in_circle_is_rectangle_l846_84670

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ
  radius_pos : radius > 0

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → Point

-- Define the property of being a parallelogram
def IsParallelogram (p : Parallelogram) : Prop :=
  let v := p.vertices
  (v 0).x - (v 1).x = (v 3).x - (v 2).x ∧
  (v 0).y - (v 1).y = (v 3).y - (v 2).y ∧
  (v 1).x - (v 2).x = (v 0).x - (v 3).x ∧
  (v 1).y - (v 2).y = (v 0).y - (v 3).y

-- Define the property of being inscribed in a circle
def InscribedInCircle (p : Parallelogram) (c : Circle) : Prop :=
  ∀ i, ((p.vertices i).x - c.center.x)^2 + ((p.vertices i).y - c.center.y)^2 = c.radius^2

-- Define a rectangle
def IsRectangle (p : Parallelogram) : Prop :=
  let v := p.vertices
  ((v 1).x - (v 0).x) * ((v 2).x - (v 1).x) + ((v 1).y - (v 0).y) * ((v 2).y - (v 1).y) = 0 ∧
  ((v 2).x - (v 1).x) * ((v 3).x - (v 2).x) + ((v 2).y - (v 1).y) * ((v 3).y - (v 2).y) = 0 ∧
  ((v 3).x - (v 2).x) * ((v 0).x - (v 3).x) + ((v 3).y - (v 2).y) * ((v 0).y - (v 3).y) = 0 ∧
  ((v 0).x - (v 3).x) * ((v 1).x - (v 0).x) + ((v 0).y - (v 3).y) * ((v 1).y - (v 0).y) = 0

-- Theorem statement
theorem parallelogram_inscribed_in_circle_is_rectangle 
  (p : Parallelogram) (c : Circle) :
  IsParallelogram p → InscribedInCircle p c → IsRectangle p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_inscribed_in_circle_is_rectangle_l846_84670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solutions_l846_84697

theorem inequality_solutions : 
  ∃ (n : ℕ), n = (Finset.filter (λ y : ℕ ↦ 5 < 2 * y + 4 ∧ y > 0) (Finset.range 11)).card ∧ n = 10 :=
by
  -- We'll use 'use' to provide the existential witness
  use 10
  -- Then we'll prove the conjunction
  apply And.intro
  · -- First part: showing that the cardinality is indeed 10
    sorry -- Proof details omitted
  · -- Second part: trivial equality
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solutions_l846_84697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_and_front_length_l846_84647

-- Define variables
variable (x y : ℝ)
variable (S P : ℝ)

-- Define constants
def colored_steel_price : ℝ := 450
def composite_steel_price : ℝ := 200
def roof_price : ℝ := 200
def max_cost : ℝ := 32000

-- Define the cost function
def cost_function (x y : ℝ) : ℝ :=
  2 * x * colored_steel_price + 2 * y * composite_steel_price + x * y * roof_price

-- Define the area function
def area_function (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem max_area_and_front_length :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    cost_function x y ≤ max_cost ∧
    area_function x y = 100 ∧
    x = 20 / 3 ∧
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → cost_function x' y' ≤ max_cost →
      area_function x' y' ≤ area_function x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_and_front_length_l846_84647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l846_84684

theorem angle_in_second_quadrant (α : ℝ) : 
  (0 < α) ∧ (α < π) ∧ (Real.sin α * Real.tan α < 0) → 
  (π / 2 < α) ∧ (α < π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l846_84684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l846_84679

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * Real.cos (2 * x) + 2

theorem f_properties :
  ∀ (k : ℤ),
    (∀ x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12),
      Monotone f) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3), f x ≤ 4) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3), 2 - Real.sqrt 3 ≤ f x) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3), f x = 4) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3), f x = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l846_84679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_for_inequality_l846_84637

theorem existence_of_k_for_inequality :
  ∃ k : ℝ, -2 * Real.sqrt 2 < k ∧ k ≤ -3/2 ∧
  ∀ x : ℝ, x ∈ Set.Icc 1 2 →
    (Real.log ((Real.sqrt (x^2 + k*x + 3) - 1) : ℝ) / Real.log 4 + 
     Real.log ((x^2 + k*x + 2) : ℝ) / Real.log 3 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_for_inequality_l846_84637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_squares_l846_84648

theorem quadratic_roots_sum_squares (h : ℝ) :
  (∃ x y : ℝ, x^2 + 4*h*x - 5 = 0 ∧ y^2 + 4*h*y - 5 = 0 ∧ x^2 + y^2 = 13) →
  |h| = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_squares_l846_84648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l846_84643

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l846_84643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_equals_one_l846_84630

/-- The line l with equation y = 2x + 1 -/
def line_l (x y : ℝ) : Prop := y = 2 * x + 1

/-- The circle C centered at the origin with radius a > 0 -/
def circle_C (x y a : ℝ) : Prop := x^2 + y^2 = a^2 ∧ a > 0

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |2 * x - y + 1| / Real.sqrt 5

/-- The maximum distance from any point on circle C to line l -/
noncomputable def max_distance (a : ℝ) : ℝ := a + Real.sqrt 5 / 5

theorem circle_radius_equals_one (a : ℝ) : 
  (∀ x y : ℝ, circle_C x y a → distance_to_line x y ≤ max_distance a) ∧ 
  (∃ x y : ℝ, circle_C x y a ∧ distance_to_line x y = max_distance a) →
  max_distance a = Real.sqrt 5 / 5 + 1 →
  a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_equals_one_l846_84630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_ratio_l846_84686

-- Define the right triangle with leg lengths 4 and 8
def leg1 : ℝ := 4
def leg2 : ℝ := 8

-- Define the area of the triangle
noncomputable def area : ℝ := (1/2) * leg1 * leg2

-- Define the hypotenuse using the Pythagorean theorem
noncomputable def hypotenuse : ℝ := Real.sqrt (leg1^2 + leg2^2)

-- Define the perimeter of the triangle
noncomputable def perimeter : ℝ := leg1 + leg2 + hypotenuse

-- Theorem stating the ratio of area to perimeter
theorem area_perimeter_ratio :
  area / perimeter = 3 - Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_ratio_l846_84686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_find_range_of_a_l846_84609

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - b * x

-- Statement 1
theorem find_b (a : ℝ) (h : a ≠ 1) :
  (∃ b : ℝ, (deriv (f a b)) 1 = 0) → 
  (∃ b : ℝ, (deriv (f a b)) 1 = 0 ∧ b = 1) :=
sorry

-- Statement 2
def range_of_a : Set ℝ := 
  Set.Ioo (- Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1

theorem find_range_of_a :
  {a : ℝ | a ≠ 1 ∧ ∃ x₀ : ℝ, x₀ ≥ 1 ∧ f a 1 x₀ < a / (a - 1)} = range_of_a :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_find_range_of_a_l846_84609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_finger_number_l846_84653

def g : ℕ → ℕ
| 0 => 1
| 1 => 0
| 2 => 3
| 3 => 2
| 4 => 5
| 5 => 4
| 6 => 7
| 7 => 6
| 8 => 9
| 9 => 8
| n => n  -- For completeness, though not used in the problem

def larry_sequence : ℕ → ℕ
| 0 => 4  -- Start with 4 on the pinky finger
| n+1 => g (larry_sequence n)

theorem tenth_finger_number : larry_sequence 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_finger_number_l846_84653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xenia_earnings_l846_84631

/-- Xenia's earnings problem -/
theorem xenia_earnings (hours_week1 hours_week2 : ℕ) 
  (extra_earnings bonus_threshold : ℚ) (bonus : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 26 →
  extra_earnings = 60.20 →
  bonus_threshold = 25 →
  bonus = 15 →
  hours_week2 > bonus_threshold →
  (((extra_earnings - bonus) / (hours_week2 - hours_week1)) * 
   (hours_week1 + hours_week2 : ℚ) + bonus) = 278.60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xenia_earnings_l846_84631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendees_l846_84695

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday

-- Define the people
inductive Person
| anna
| bill
| carl
| dave

-- Define a function that returns whether a person can attend on a given day
def canAttend (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.anna, Day.monday => true
  | Person.anna, Day.wednesday => true
  | Person.bill, Day.tuesday => true
  | Person.bill, Day.thursday => true
  | Person.bill, Day.friday => true
  | Person.carl, Day.monday => true
  | Person.carl, Day.tuesday => true
  | Person.carl, Day.thursday => true
  | Person.carl, Day.friday => true
  | Person.dave, Day.tuesday => true
  | Person.dave, Day.wednesday => true
  | _, _ => false

-- Define a function that counts the number of attendees for a given day
def countAttendees (d : Day) : Nat :=
  (List.filter (fun p => canAttend p d) [Person.anna, Person.bill, Person.carl, Person.dave]).length

-- Theorem statement
theorem max_attendees :
  (∀ d : Day, countAttendees d ≤ 2) ∧
  (countAttendees Day.monday = 2) ∧
  (countAttendees Day.wednesday = 2) ∧
  (countAttendees Day.thursday = 2) ∧
  (countAttendees Day.friday = 2) ∧
  (countAttendees Day.tuesday < 2) :=
by
  sorry

#eval countAttendees Day.monday
#eval countAttendees Day.tuesday
#eval countAttendees Day.wednesday
#eval countAttendees Day.thursday
#eval countAttendees Day.friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendees_l846_84695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_item_cost_proof_l846_84663

def calculate_item_cost (initial_amount : ℚ) (given_to_mom : ℚ) (num_items : ℕ) (final_amount : ℚ) : ℚ :=
  let remaining_after_mom := initial_amount - given_to_mom
  let invested := remaining_after_mom / 2
  let before_shopping := remaining_after_mom - invested
  let spent_on_items := before_shopping - final_amount
  spent_on_items / (num_items : ℚ)

theorem item_cost_proof (initial_amount given_to_mom final_amount : ℚ) (num_items : ℕ)
  (h1 : initial_amount = 25)
  (h2 : given_to_mom = 8)
  (h3 : num_items = 5)
  (h4 : final_amount = 6) :
  calculate_item_cost initial_amount given_to_mom num_items final_amount = 1/2 := by
  sorry

#eval calculate_item_cost 25 8 5 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_item_cost_proof_l846_84663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_imply_c_magnitude_l846_84610

theorem polynomial_roots_imply_c_magnitude (c : ℂ) : 
  (∃ (S : Finset ℂ), S.card = 5 ∧ 
    (∀ x, x ∈ S ↔ (x^2 - 3*x + 3) * (x^2 - c*x + 1) * (x^2 - 2*x + 5) = 0)) →
  Complex.abs c = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_imply_c_magnitude_l846_84610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_55_divisors_l846_84604

def number_of_divisors (n : ℕ) : ℕ := (Finset.filter (λ d ↦ n % d = 0) (Finset.range (n + 1))).card

def is_smallest_with_55_divisors (n : ℕ) : Prop :=
  number_of_divisors n = 55 ∧ ∀ m : ℕ, m < n → number_of_divisors m ≠ 55

theorem smallest_number_with_55_divisors :
  is_smallest_with_55_divisors (3^4 * 2^10) := by
  sorry

#eval number_of_divisors (3^4 * 2^10)
#eval 3^4 * 2^10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_55_divisors_l846_84604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l846_84675

-- Define the set of possible x values
def possible_x : Set ℝ := {-2, 0, 2, 3}

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt ((x + 1) / (2 - x)) = Real.sqrt (x + 1) / Real.sqrt (2 - x)

-- Define the domain conditions
def domain_conditions (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ 2 - x > 0

-- Theorem statement
theorem unique_solution :
  ∃! x, x ∈ possible_x ∧ equation x ∧ domain_conditions x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l846_84675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_index_is_four_l846_84635

/-- A positive, non-constant geometric progression -/
structure GeometricProgression where
  b : ℝ  -- initial term
  q : ℝ  -- common ratio
  b_pos : b > 0
  q_neq_1 : q ≠ 1
  q_pos : q > 0

/-- The nth term of a geometric progression -/
noncomputable def term (gp : GeometricProgression) (n : ℕ) : ℝ := gp.b * gp.q ^ (n - 1)

/-- The arithmetic mean of the 3rd, 4th, and 8th terms -/
noncomputable def mean_3_4_8 (gp : GeometricProgression) : ℝ :=
  (term gp 3 + term gp 4 + term gp 8) / 3

theorem smallest_index_is_four (gp : GeometricProgression) :
  ∃ k : ℕ, k ≥ 1 ∧ mean_3_4_8 gp = term gp k ∧
  ∀ j : ℕ, j ≥ 1 → mean_3_4_8 gp = term gp j → k ≤ j :=
by
  -- The proof goes here
  sorry

#check smallest_index_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_index_is_four_l846_84635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l846_84638

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (-x^2 + 4*x)

-- Theorem statement
theorem monotonic_increase_interval :
  ∃ (a b : ℝ), a = 0 ∧ b = 2 ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → f x₁ < f x₂) ∧
  (∀ ε > 0, ∃ x₁ x₂, x₁ < a ∧ b < x₂ ∧ f x₁ ≥ f x₂) := by
  sorry

#check monotonic_increase_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l846_84638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_representation_correct_answer_is_C_l846_84602

/-- The number represented by a counter with 'a' beads in the tens place and 4 in the ones place is 10a + 4 -/
theorem counter_representation (a : ℕ) : 10 * a + 4 = 10 * a + 4 := by
  -- This is a trivial equality, so we can prove it directly
  rfl

/-- The correct answer is option C: 10a + 4 -/
theorem correct_answer_is_C (a : ℕ) : 
  (10 * a + 4 = 4 * a) ∨ 
  (10 * a + 4 = a + 4) ∨ 
  (10 * a + 4 = 10 * a + 4) ∨ 
  (10 * a + 4 = a + 40) := by
  -- We choose the third option
  apply Or.inr
  apply Or.inr
  apply Or.inl
  -- This is a trivial equality
  rfl

#check counter_representation
#check correct_answer_is_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_representation_correct_answer_is_C_l846_84602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_theorem_l846_84652

-- Define the geometric sequence and its sum
noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

-- Define the conditions
def S₃_condition (a₁ q : ℝ) : Prop := geometric_sum a₁ q 3 = 3/2
def S₆_condition (a₁ q : ℝ) : Prop := geometric_sum a₁ q 6 = 21/16

-- Define the b_n sequence
noncomputable def b_n (lambda : ℝ) (a_n : ℕ → ℝ) (n : ℕ) : ℝ := lambda * a_n n - n^2

-- The main theorem
theorem geometric_sequence_theorem (a₁ q lambda : ℝ) :
  S₃_condition a₁ q →
  S₆_condition a₁ q →
  (∀ n : ℕ, geometric_sequence a₁ q n = 2 * (-1/2)^(n-1)) ∧
  (∀ n : ℕ, b_n lambda (geometric_sequence a₁ q) (n+1) < b_n lambda (geometric_sequence a₁ q) n → 
    -1 < lambda ∧ lambda < 10/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_theorem_l846_84652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l846_84659

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

/-- Check if a line passes through a given point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Find the x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.intercept / l.slope

/-- Check if a line intersects both axes -/
def intersects_axes (l : Line) : Prop :=
  l.slope ≠ 0 ∧ l.intercept ≠ 0

theorem line_satisfies_conditions :
  let l : Line := ⟨8/5, 4⟩
  passes_through l (-5) (-4) ∧
  intersects_axes l ∧
  triangle_area (abs (x_intercept l)) (abs l.intercept) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l846_84659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccot_inequality_arccot_equality_l846_84699

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + sequence_a n

theorem arccot_inequality (n : ℕ) :
  Real.arctan (1 / sequence_a n) ≤ Real.arctan (1 / sequence_a (n + 1)) + Real.arctan (1 / sequence_a (n + 2)) :=
by sorry

theorem arccot_equality (n : ℕ) :
  Real.arctan (1 / sequence_a n) = Real.arctan (1 / sequence_a (n + 1)) + Real.arctan (1 / sequence_a (n + 2)) ↔ Even n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccot_inequality_arccot_equality_l846_84699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l846_84605

noncomputable def g (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem g_range : Set.range g = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l846_84605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l846_84694

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3)

theorem f_decreasing_interval : 
  ∀ x y, Real.pi / 3 < x ∧ x < y ∧ y < 4 * Real.pi / 3 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l846_84694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_probability_l846_84672

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def total_outcomes : ℕ := card_set.card * card_set.card

def favorable_outcomes : ℕ := 
  (card_set.filter (λ x => ∃ y ∈ card_set, x > y)).sum (λ x => 
    (card_set.filter (λ y => y < x)).card)

theorem card_drawing_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_probability_l846_84672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l846_84626

noncomputable def f (x : ℝ) := Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

theorem f_properties :
  (f (π / 3) = Real.sqrt 3 / 2 - 1 / 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ Real.sqrt 2 - 1) ∧
  (f (3 * π / 8) = Real.sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l846_84626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_b_is_four_l846_84641

/-- The sum of possible values for b, where b is a coordinate of a side of a rectangle -/
def sum_of_possible_b : ℝ := 4

/-- The height of the rectangle -/
def rectangle_height : ℝ := 5

/-- The x-coordinate of one vertical side of the rectangle -/
def fixed_x : ℝ := 2

/-- Theorem stating that the sum of possible values for b is 4 -/
theorem sum_of_possible_b_is_four :
  ∃ (b₁ b₂ : ℝ),
    (b₁ ≠ b₂) ∧
    (b₁ ≠ fixed_x) ∧
    (b₂ ≠ fixed_x) ∧
    (|b₁ - fixed_x| = rectangle_height ∨ |b₂ - fixed_x| = rectangle_height) ∧
    sum_of_possible_b = b₁ + b₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_b_is_four_l846_84641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_segment_height_formula_l846_84639

/-- The height of a circular segment given the diameter and base -/
noncomputable def circularSegmentHeight (d a : ℝ) : ℝ := (d - Real.sqrt (d^2 - a^2)) / 2

/-- Theorem: The height of a circular segment with diameter d and base a 
    is given by (d - √(d² - a²)) / 2 -/
theorem circular_segment_height_formula (d a : ℝ) (h_pos : d > 0) (h_base : 0 < a ∧ a < d) :
  ∃ h : ℝ, h > 0 ∧ h = circularSegmentHeight d a ∧ 
  h^2 - d * h + (a^2 / 4) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_segment_height_formula_l846_84639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_digit_avg_count_l846_84655

/-- A function that checks if a three-digit number has its middle digit as the average of its first and last digits -/
def is_middle_avg (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 10 % 10) * 2 = (n / 100) + (n % 10)

/-- The count of three-digit numbers where the middle digit is the average of the first and last digits -/
def count_middle_avg : ℕ := (Finset.range 900).filter (λ n => is_middle_avg (n + 100)) |>.card

/-- Theorem stating that there are exactly 45 three-digit numbers where the middle digit is the average of the first and last digits -/
theorem middle_digit_avg_count : count_middle_avg = 45 := by
  sorry

#eval count_middle_avg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_digit_avg_count_l846_84655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l846_84608

theorem fraction_equality : 
  (Real.sqrt 2 * (Real.sqrt 3 - Real.sqrt 7)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  (30 - 10 * Real.sqrt 5 - 6 * Real.sqrt 21 + 2 * Real.sqrt 105) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l846_84608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l846_84618

/-- The ellipse represented by the equation x²/4 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 = 1}

/-- A line with slope 1 -/
def SlopeLine (t : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 + t}

/-- The length of a chord formed by the intersection of a line with slope 1 and the ellipse -/
noncomputable def ChordLength (t : ℝ) : ℝ :=
  (4 * Real.sqrt 2 * Real.sqrt (5 - t^2)) / 5

theorem max_chord_length :
  (∃ (t : ℝ), ChordLength t = 4 * Real.sqrt 10 / 5) ∧
  (∀ (t : ℝ), ChordLength t ≤ 4 * Real.sqrt 10 / 5) := by
  sorry

#check max_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l846_84618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_savings_percentage_l846_84698

-- Define the sale condition
noncomputable def sale_condition (original_price : ℝ) (sale_price : ℝ) : Prop :=
  sale_price = (6 / 8) * original_price

-- Define the savings percentage
noncomputable def savings_percentage (original_price : ℝ) (sale_price : ℝ) : ℝ :=
  (original_price - sale_price) / original_price * 100

-- Theorem statement
theorem sale_savings_percentage :
  ∀ (original_price : ℝ) (sale_price : ℝ),
  original_price > 0 →
  sale_condition original_price sale_price →
  savings_percentage original_price sale_price = 25 := by
  intros original_price sale_price h_positive h_condition
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_savings_percentage_l846_84698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l846_84674

/-- Ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_y_axis : Bool
  sum_of_distances : ℝ
  eccentricity : ℝ
  set : Set (ℝ × ℝ)

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  slope : ℝ
  y_intercept : ℝ

/-- Theorem about the ellipse equation and area of triangle -/
theorem ellipse_properties (C : Ellipse) (L : IntersectingLine) : 
  C.center = (0, 0) → 
  C.foci_on_y_axis = true → 
  C.sum_of_distances = 4 → 
  C.eccentricity = Real.sqrt 3 / 2 → 
  (∀ (x y : ℝ), (x, y) ∈ C.set ↔ y^2 / 4 + x^2 = 1) ∧ 
  (∃ (S : Set ℝ), S = {area | ∃ (A B : ℝ × ℝ), 
    A ∈ C.set ∧ B ∈ C.set ∧ 
    A.2 = L.slope * A.1 + L.y_intercept ∧ 
    B.2 = L.slope * B.1 + L.y_intercept ∧ 
    area = abs (A.1 * B.2 - A.2 * B.1) / 2} ∧ 
    Set.Icc (0 : ℝ) (Real.sqrt 3 / 2) = S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l846_84674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_equation_l846_84678

theorem product_of_roots_quadratic_equation :
  let a : ℝ := 24
  let b : ℝ := 36
  let c : ℝ := -216
  let equation := fun x : ℝ ↦ a * x^2 + b * x + c
  ∃ x₁ x₂ : ℝ, equation x₁ = 0 ∧ equation x₂ = 0 ∧ x₁ * x₂ = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_equation_l846_84678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c₁_c₂_not_collinear_l846_84691

def a : Fin 3 → ℝ := ![4, 2, 9]
def b : Fin 3 → ℝ := ![0, -1, 3]
def c₁ : Fin 3 → ℝ := λ i => 4 * b i - 3 * a i
def c₂ : Fin 3 → ℝ := λ i => 4 * a i - 3 * b i

theorem c₁_c₂_not_collinear : ¬ ∃ (k : ℝ), ∀ i, c₁ i = k * c₂ i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c₁_c₂_not_collinear_l846_84691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l846_84634

/-- Represents the characteristics and investment details of a stock --/
structure Stock where
  dividend_rate : ℚ  -- Dividend rate as a rational number
  market_price : ℚ   -- Market price per $100 of face value
  annual_income : ℚ  -- Annual income from the investment

/-- Calculates the amount invested in a stock given its characteristics --/
def amount_invested (s : Stock) : ℚ :=
  (s.annual_income / s.dividend_rate) * (s.market_price / 100)

/-- Theorem stating that for a stock with given characteristics, 
    the amount invested is $6800 --/
theorem investment_calculation (s : Stock) 
  (h1 : s.dividend_rate = 1/5)
  (h2 : s.market_price = 136)
  (h3 : s.annual_income = 1000) :
  amount_invested s = 6800 := by
  sorry

#eval amount_invested { dividend_rate := 1/5, market_price := 136, annual_income := 1000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l846_84634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l846_84665

/-- Represents the game state -/
structure GameState where
  n : ℕ
  chosen : Set ℝ

/-- Checks if a move is valid -/
def validMove (state : GameState) (x : ℝ) : Prop :=
  x ∈ Set.Icc 0 (state.n : ℝ) ∧
  ∀ y ∈ state.chosen, |x - y| > 2

/-- Represents a winning strategy for Bela -/
def belaWinningStrategy (n : ℕ) : Prop :=
  ∃ (strategy : GameState → ℝ),
    ∀ (state : GameState),
      validMove state (strategy state) →
      ¬∃ (x : ℝ), validMove {n := state.n, chosen := state.chosen ∪ {strategy state}} x

/-- The main theorem stating Bela always wins -/
theorem bela_always_wins (n : ℕ) (h : n > 10) : belaWinningStrategy n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l846_84665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_15_factorial_l846_84640

def largest_perfect_square_divisor (n : Nat) : Nat :=
  sorry

def prime_factorization (n : Nat) : List (Nat × Nat) :=
  sorry

def square_root_exponents (factors : List (Nat × Nat)) : List Nat :=
  sorry

theorem sum_of_exponents_15_factorial :
  (square_root_exponents (prime_factorization (largest_perfect_square_divisor (Nat.factorial 15)))).sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_15_factorial_l846_84640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sturdy_square_with_18_dominoes_l846_84636

/-- A domino is a 2 × 1 tile that covers two unit squares. -/
structure Domino where
  x : ℕ
  y : ℕ

/-- A square grid of size n × n. -/
structure Grid (n : ℕ) where
  dominoes : List Domino

/-- A line in the grid. -/
inductive Line
  | Vertical (x : ℕ)
  | Horizontal (y : ℕ)

/-- A line intersects a domino if it crosses the domino. -/
def line_intersects_domino (l : Line) (d : Domino) : Prop :=
  match l with
  | Line.Vertical x => d.x < x ∧ x < d.x + 2
  | Line.Horizontal y => d.y < y ∧ y < d.y + 1

/-- A square grid is sturdy if no straight line (other than the boundaries) 
    formed by domino edges connects opposite sides. -/
def is_sturdy (g : Grid 6) : Prop :=
  ∀ l : Line, ∃ d : Domino, d ∈ g.dominoes ∧ line_intersects_domino l d

/-- The main theorem stating that it's impossible to construct a 6 × 6 sturdy square 
    using 18 domino tiles. -/
theorem no_sturdy_square_with_18_dominoes : 
  ¬∃ (g : Grid 6), g.dominoes.length = 18 ∧ is_sturdy g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sturdy_square_with_18_dominoes_l846_84636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_grid_l846_84676

/-- Represents a 3x3 grid with letters A, B, and C --/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a grid is valid according to the problem conditions --/
def is_valid_grid (g : Grid) : Prop :=
  -- A is in the upper left corner
  g 0 0 = 0 ∧
  -- B is in the middle of the first column
  g 1 0 = 1 ∧
  -- Each row contains one of each letter
  (∀ i : Fin 3, Finset.card (Finset.image (g i) Finset.univ) = 3) ∧
  -- Each column contains one of each letter
  (∀ j : Fin 3, Finset.card (Finset.image (fun i => g i j) Finset.univ) = 3)

/-- There is exactly one valid grid arrangement --/
theorem unique_valid_grid : ∃! g : Grid, is_valid_grid g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_grid_l846_84676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_continued_fraction_is_golden_ratio_l846_84650

/-- The value of the infinite continued fraction 1 + 1/(1 + 1/(1 + ...)) -/
noncomputable def infinite_continued_fraction : ℝ :=
  Real.sqrt 2 + Real.sqrt (2 + Real.sqrt 2)

/-- The golden ratio -/
noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem infinite_continued_fraction_is_golden_ratio :
  infinite_continued_fraction = golden_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_continued_fraction_is_golden_ratio_l846_84650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_exam_l846_84642

open Finset

theorem pigeonhole_principle_exam (students : Finset ℕ) (exercises : Finset ℕ) 
  (student_solution : ℕ → ℕ) :
  students.card = 16 →
  exercises.card = 3 →
  (∀ s ∈ students, student_solution s ∈ exercises) →
  ∃ (e : ℕ), e ∈ exercises ∧ (students.filter (λ s => student_solution s = e)).card ≥ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_exam_l846_84642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_angle_theorem_l846_84625

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the perpendicular AK
def perpendicular (rect : Rectangle) (K : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the angle between two vectors
def angle (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- Define the ratio of angles
def angle_ratio (α β : ℝ) (r : ℚ) : Prop :=
  α = r * β

theorem perpendicular_angle_theorem (rect : Rectangle) (K : EuclideanSpace ℝ (Fin 2)) :
  perpendicular rect K →
  angle_ratio (angle rect.B rect.A K) (angle rect.D rect.A K) (3 : ℚ) →
  angle K rect.A rect.C = 45 :=
sorry

#check perpendicular_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_angle_theorem_l846_84625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l846_84633

-- Define the distance between the cyclists
def distance : ℝ := 12

-- Define the time it takes for the cyclists to meet when traveling towards each other
def meeting_time : ℝ := 2

-- Define the time it takes for the faster cyclist to overtake the slower one
def overtake_time : ℝ := 6

-- Define the speeds of the cyclists
def speed_slower : ℝ := 2
def speed_faster : ℝ := 4

-- State the theorem
theorem cyclist_speed_ratio : 
  speed_faster = 2 * speed_slower ∧
  (speed_faster + speed_slower) * meeting_time = distance ∧
  (speed_faster - speed_slower) * overtake_time = distance →
  speed_faster / speed_slower = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_ratio_l846_84633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_transformed_function_l846_84688

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (6 * x + Real.pi / 4)

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (2 * x)

def is_symmetry_center (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem symmetry_center_of_transformed_function :
  is_symmetry_center transformed_function (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_transformed_function_l846_84688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_reflection_range_l846_84677

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x + y - 4 = 0
def CenterLine := {p : ℝ × ℝ | p.1 + p.2 = 4}

-- Define the point A
def A : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def XAxis := {p : ℝ × ℝ | p.2 = 0}

theorem circle_equation_and_reflection_range :
  ∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C passes through (1, 2) and (2, 1)
    (1, 2) ∈ C ∧ (2, 1) ∈ C ∧
    -- Center of C is on the line x + y - 4 = 0
    center ∈ CenterLine ∧
    -- C is defined by the Circle function
    C = Circle center radius ∧
    -- The equation of C is (x-2)^2 + (y-2)^2 = 1
    center = (2, 2) ∧ radius = 1 ∧
    -- The range of a for the reflection point M(a, 0) is [-3/4, 1]
    ∃ (reflection_range : Set ℝ),
      reflection_range = {a : ℝ | -3/4 ≤ a ∧ a ≤ 1} ∧
      ∀ a : ℝ, (a ∈ reflection_range ↔
        ∃ (ray : ℝ → ℝ × ℝ),
          -- Ray starts from A
          ray 0 = A ∧
          -- Ray hits the x-axis at M(a, 0)
          ∃ t : ℝ, ray t ∈ XAxis ∧ (ray t).1 = a ∧
          -- Ray is reflected onto circle C
          ∃ u : ℝ, ray u ∈ C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_reflection_range_l846_84677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l846_84657

/-- Represents the radius of the circular top view of the geometric body -/
noncomputable def radius : ℝ := 3

/-- Represents the volume of a single geometric body -/
noncomputable def body_volume : ℝ := (1/3) * Real.pi * radius^2 * (radius * Real.sqrt 3)

/-- Represents the edge length of the resulting cube -/
noncomputable def cube_edge : ℝ := (3 * body_volume) ^ (1/3)

/-- Theorem stating that the surface area of the cube formed by melting three geometric bodies
    with equilateral triangle front and side views and circular top view with radius 3
    is equal to 5433π² -/
theorem cube_surface_area :
  6 * cube_edge^2 = 5433 * Real.pi^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l846_84657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_area_swept_l846_84628

/-- The length of the minute hand in units -/
noncomputable def minute_hand_length : ℝ := 10

/-- The time elapsed in minutes -/
noncomputable def time_elapsed : ℝ := 35

/-- The angle swept by the minute hand in radians -/
noncomputable def angle_swept : ℝ := (time_elapsed / 60) * 2 * Real.pi

/-- The area swept by the minute hand -/
noncomputable def area_swept : ℝ := (1 / 2) * minute_hand_length^2 * angle_swept

theorem minute_hand_area_swept :
  area_swept = (175 * Real.pi) / 3 := by
  -- Expand the definitions
  unfold area_swept angle_swept minute_hand_length time_elapsed
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_area_swept_l846_84628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_drive_cost_theorem_l846_84601

/-- The cost of two hard drives in dollars -/
def two_drive_cost : ℚ := 50

/-- The number of hard drives being purchased -/
def num_drives : ℕ := 7

/-- The discount rate for buying more than four hard drives -/
def discount_rate : ℚ := 1/10

/-- The minimum number of drives to qualify for the discount -/
def discount_threshold : ℕ := 4

/-- Calculates the total cost of purchasing hard drives with a potential discount -/
def total_cost (two_drive_cost : ℚ) (num_drives : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let single_drive_cost := two_drive_cost / 2
  let total_without_discount := single_drive_cost * num_drives
  if num_drives > discount_threshold then
    total_without_discount * (1 - discount_rate)
  else
    total_without_discount

theorem hard_drive_cost_theorem :
  total_cost two_drive_cost num_drives discount_rate discount_threshold = 315/2 := by
  sorry

#eval total_cost two_drive_cost num_drives discount_rate discount_threshold

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_drive_cost_theorem_l846_84601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l846_84668

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos : 0 < r

/-- The distance from the center to a focus of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Predicate to check if a circle is tangent to and contained within an ellipse -/
def is_tangent_and_contained (e : Ellipse) (c : Circle) : Prop :=
  c.h = focal_distance e ∧ c.k = 0 ∧
  ∃ (x y : ℝ), (x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
                ((x - c.h)^2 + y^2 = c.r^2) ∧
                ∀ (x' y' : ℝ), (x'^2 / e.a^2 + y'^2 / e.b^2 ≤ 1) →
                                ((x' - c.h)^2 + y'^2 ≥ c.r^2)

theorem ellipse_tangent_circle_radius 
  (e : Ellipse) 
  (h_e : e.a = 6 ∧ e.b = 5) 
  (c : Circle) 
  (h_c : is_tangent_and_contained e c) : 
  c.r = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l846_84668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_odd_l846_84661

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The power function with exponent a -/
noncomputable def PowerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ Real.rpow x a

theorem power_function_domain_and_odd (a : ℝ) :
  (∀ x, PowerFunction a x ∈ Set.univ) ∧ IsOdd (PowerFunction a) ↔ a = 1 ∨ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_odd_l846_84661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steel_bolt_force_18_inch_l846_84622

/-- The force required to loosen a bolt, excluding the additional force for steel bolts -/
noncomputable def base_force (length : ℝ) : ℝ := 3600 / length

/-- The total force required to loosen a steel bolt -/
noncomputable def total_force (length : ℝ) : ℝ := base_force length + 50

theorem steel_bolt_force_18_inch : total_force 18 = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steel_bolt_force_18_inch_l846_84622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equals_mean_l846_84613

open Real

noncomputable def numbers (y : ℝ) : Finset ℝ := {3, 7, 9, y, 20}

def is_median (x : ℝ) (s : Finset ℝ) : Prop :=
  2 * (s.filter (· ≤ x)).card ≥ s.card ∧
  2 * (s.filter (· ≥ x)).card ≥ s.card

theorem median_equals_mean :
  ∀ y : ℝ, (∃ x ∈ numbers y, is_median x (numbers y) ∧ x = (numbers y).toList.sum / (numbers y).card) ↔ y = -4 := by
  sorry

#check median_equals_mean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equals_mean_l846_84613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_are_coplanar_l846_84615

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points
variable (O A B C P : V)

-- Define the non-collinearity condition
def non_collinear (A B C : V) : Prop :=
  ∀ (α β γ : ℝ), α • (B - A) + β • (C - A) + γ • (C - B) = 0 → α = 0 ∧ β = 0 ∧ γ = 0

-- Define the coplanarity condition
def coplanar (P A B C : V) : Prop :=
  ∃ (α β γ : ℝ), P - A = α • (B - A) + β • (C - A)

-- State the theorem
theorem points_are_coplanar
  (h_non_collinear : non_collinear A B C)
  (h_relation : (2 : ℝ) • (P - O) = (-(A - O) + (B - O) + 2 • (C - O))) :
  coplanar P A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_are_coplanar_l846_84615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l846_84611

theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (eccentricity_eq : Real.sqrt 5 = Real.sqrt ((a^2 + b^2) / a^2)) 
  (distance_to_asymptote : 8 = (b * Real.sqrt (a^2 + b^2)) / Real.sqrt (a^2 + b^2)) : 
  2 * a = 8 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l846_84611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l846_84619

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (-3, m)
def vector_b : ℝ × ℝ := (2, -2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem perpendicular_vectors_magnitude (m : ℝ) :
  dot_product (vector_a m) vector_b = 0 →
  magnitude (vector_a m) = 3 * Real.sqrt 2 :=
by
  intro h
  -- The proof steps would go here
  sorry

#eval vector_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l846_84619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l846_84664

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  4 * t.a^2 * (Real.cos t.B)^2 + 4 * t.b^2 * (Real.sin t.A)^2 = 3 * t.b^2 - 3 * t.c^2

/-- Part 1: Prove that a + 6c cos B = 0 -/
theorem part1 (t : Triangle) (h : condition t) : t.a + 6 * t.c * Real.cos t.B = 0 := by
  sorry

/-- Part 2: Maximum area when b = 1 -/
theorem part2 (t : Triangle) (h : t.b = 1) : 
  ∃ (maxArea : ℝ), maxArea = 3/14 ∧ 
  ∀ (area : ℝ), area = 1/2 * t.a * t.c * Real.sin t.B → area ≤ maxArea := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l846_84664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l846_84693

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, -x^2 + 4*x + 3 > 0)) ↔ (∃ x : ℝ, -x^2 + 4*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l846_84693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_l846_84644

/-- Given a yard of length 375 metres with 37 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 375/36 metres. -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 375) (h2 : num_trees = 37) :
  yard_length / (num_trees - 1) = 375 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_l846_84644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l846_84689

/-- A piecewise function f(x) defined as follows:
    f(x) = a/x for x ≥ 1
    f(x) = -x + 3a for x < 1
    where a is a real number -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then a / x else -x + 3 * a

/-- The theorem states that if f is monotonic on ℝ,
    then a is in the interval [1/2, +∞) -/
theorem monotonic_f_implies_a_range (a : ℝ) :
  Monotone (f a) → a ∈ Set.Ici (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l846_84689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l846_84667

noncomputable def f (x : ℝ) : ℝ := -(Real.cos x)^2 + Real.sqrt 3 * Real.cos x + 5/4

theorem f_extrema :
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ x : ℝ, f x ≥ 1/4 - Real.sqrt 3) ∧
  (∀ k : ℤ, f (2 * k * Real.pi + Real.pi / 6) = 2) ∧
  (∀ k : ℤ, f (2 * k * Real.pi + 11 * Real.pi / 6) = 2) ∧
  (∀ k : ℤ, f (2 * k * Real.pi + Real.pi) = 1/4 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l846_84667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_form_l846_84671

def a : ℕ → ℤ
| 0 => 1
| 1 => 4
| n + 2 => 5 * a (n + 1) - a n

theorem sequence_form (n : ℕ) : ∃ c d : ℤ, a n = c^2 + 3 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_form_l846_84671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_intersection_l846_84629

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ

/-- Tangent line to a parabola at a point -/
noncomputable def tangent_line (c : Parabola) (p : Point) : Line :=
  { m := p.x / c.p
  , b := p.y - (p.x^2) / (2 * c.p) }

/-- Intersection point of two lines -/
noncomputable def line_intersection (l1 l2 : Line) : Point :=
  { x := (l2.b - l1.b) / (l1.m - l2.m)
  , y := l1.m * ((l2.b - l1.b) / (l1.m - l2.m)) + l1.b }

/-- Main theorem -/
theorem parabola_tangent_intersection
  (c : Parabola)
  (f : Point)
  (p q : Point)
  (h1 : p.x^2 = 2 * c.p * p.y)
  (h2 : q.x^2 = 2 * c.p * q.y)
  (h3 : f.x = 0 ∧ f.y = c.p / 2)
  (h4 : ∃ (k : ℝ), p.y = k * p.x + c.p / 2 ∧ q.y = k * q.x + c.p / 2)
  : (line_intersection (tangent_line c p) (tangent_line c q)).y = -c.p / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_intersection_l846_84629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_different_numerators_l846_84600

/-- The set of rational numbers with repeating decimal expansion 0.abcabcabc... -/
def S : Set ℚ :=
  {r : ℚ | 0 < r ∧ r < 1 ∧ ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧
    r = (100 * a + 10 * b + c : ℚ) / 999}

/-- The number of different numerators required to represent all elements of S in lowest terms -/
def num_different_numerators : ℕ := 660

/-- Theorem stating that the number of different numerators is correct -/
theorem count_different_numerators :
  (Finset.filter (fun n => Nat.Coprime n 999) (Finset.range 1000)).card = num_different_numerators :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_different_numerators_l846_84600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kiepert_hyperbola_center_coordinates_l846_84621

/-- Given a triangle with side lengths a, b, and c, this theorem states the barycentric and trilinear coordinates of the center of the Kiepert hyperbola. -/
theorem kiepert_hyperbola_center_coordinates (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let barycentric := ((b^2 - c^2)^2, (c^2 - a^2)^2, (a^2 - b^2)^2)
  let trilinear := ((b^2 - c^2)^2 / 2, (c^2 - a^2)^2 / 2, (a^2 - b^2)^2 / 2)
  (∃ k : ℝ, k ≠ 0 ∧ barycentric = k • (b^2 - c^2, c^2 - a^2, a^2 - b^2)) ∧
  (∃ l : ℝ, l ≠ 0 ∧ trilinear = l • ((b^2 - c^2) / 2, (c^2 - a^2) / 2, (a^2 - b^2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kiepert_hyperbola_center_coordinates_l846_84621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_to_bernard_time_l846_84690

/-- Calculates the total time for June to bike to Bernard's house, including breaks -/
noncomputable def time_to_bernards_house (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) (break_time : ℝ) : ℝ :=
  let biking_rate := distance_to_julia / time_to_julia
  let biking_time := distance_to_bernard / biking_rate
  let num_breaks := 1 -- Assuming one break for every 1.5 miles
  biking_time + num_breaks * break_time

/-- Theorem stating that June's travel time to Bernard's house is 21.8 minutes -/
theorem june_to_bernard_time :
  let distance_to_julia : ℝ := 1.5
  let time_to_julia : ℝ := 6
  let distance_to_bernard : ℝ := 4.2
  let break_time : ℝ := 5
  time_to_bernards_house distance_to_julia time_to_julia distance_to_bernard break_time = 21.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_to_bernard_time_l846_84690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_property_l846_84696

/-- A coloring function that assigns one of three colors to each number in {1,...,n} -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

/-- Property that must hold for the least n -/
def HasPropertyForN (n : ℕ) : Prop :=
  ∀ (c : Coloring n),
    ∃ (a b : Fin n),
      a ≠ b ∧ 
      c a = c b ∧ 
      IsPerfectSquare (Int.natAbs (a.val - b.val))

theorem least_n_with_property :
  (∀ m < 28, ¬HasPropertyForN m) ∧ HasPropertyForN 28 := by
  sorry

#check least_n_with_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_with_property_l846_84696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_population_west_percentage_l846_84654

/-- Represents the population (in millions) of an ethnic group in a specific region -/
structure RegionalPopulation where
  northeast : ℕ
  midwest : ℕ
  south : ℕ
  west : ℕ

/-- The U.S. population data for different ethnic groups in 1980 -/
def us_population_1980 : RegionalPopulation → RegionalPopulation
| ⟨50, 60, 70, 40⟩ => RegionalPopulation.mk 50 60 70 40  -- White
| ⟨7, 8, 20, 3⟩ => RegionalPopulation.mk 7 8 20 3  -- Black
| ⟨2, 3, 4, 10⟩ => RegionalPopulation.mk 2 3 4 10  -- Asian
| ⟨2, 2, 3, 5⟩ => RegionalPopulation.mk 2 2 3 5  -- Other
| _ => RegionalPopulation.mk 0 0 0 0  -- Default case

/-- The Asian population data in 1980 -/
def asian_population : RegionalPopulation :=
  us_population_1980 ⟨2, 3, 4, 10⟩

/-- Calculates the total population across all regions -/
def total_population (pop : RegionalPopulation) : ℕ :=
  pop.northeast + pop.midwest + pop.south + pop.west

/-- Calculates the percentage of a part relative to a whole, rounded to the nearest percent -/
def percentage (part : ℕ) (whole : ℕ) : ℕ :=
  ((part * 100 + whole / 2) / whole)

/-- Theorem: The percentage of the U.S. Asian population living in the West in 1980 was approximately 53% -/
theorem asian_population_west_percentage :
  percentage asian_population.west (total_population asian_population) = 53 := by
  sorry

#eval percentage asian_population.west (total_population asian_population)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_population_west_percentage_l846_84654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_is_circle_l846_84683

-- Define the circle L
noncomputable def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the center and radius of circle L
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def r : ℝ := 1

-- Define the point S
noncomputable def S : ℝ × ℝ := (r/4, 0)

-- Define a chord of L passing through S
noncomputable def Chord (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (S.1 + t * Real.cos a, S.2 + t * Real.sin a) ∧ 
               (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the midpoint of a chord
noncomputable def Midpoint (chord : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Theorem: The locus of midpoints is a circle
theorem midpoint_locus_is_circle :
  ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    ∀ a : ℝ, Midpoint (Chord a) ∈ Circle center radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_is_circle_l846_84683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_division_l846_84632

theorem factorial_division :
  (Nat.factorial 9 : ℕ) = 362880 → (Nat.factorial 9 : ℕ) / (Nat.factorial 4 : ℕ) = 15120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_division_l846_84632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l846_84649

theorem lottery_probability (n m k : ℕ) (hn : n = 45) (hm : m = 6) (hk : k = 1) :
  (m * Nat.choose (n - m) (m - k) : ℚ) / Nat.choose n m = 
  (m * Nat.choose (n - m) (m - k) : ℚ) / Nat.choose n m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l846_84649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_lawyers_percentage_l846_84623

/-- Represents a study group with women and lawyers -/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℚ
  woman_lawyer_probability : ℚ

/-- Calculates the percentage of women who are lawyers in the study group -/
noncomputable def percentage_women_lawyers (group : StudyGroup) : ℚ :=
  (group.woman_lawyer_probability / group.women_percentage) * 100

/-- Theorem stating that for a study group with 70% women and 0.28 probability of selecting a woman lawyer,
    the percentage of women who are lawyers is 40% -/
theorem women_lawyers_percentage (group : StudyGroup) 
    (h1 : group.women_percentage = 7/10)
    (h2 : group.woman_lawyer_probability = 28/100) : 
  percentage_women_lawyers group = 40 := by
  unfold percentage_women_lawyers
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_lawyers_percentage_l846_84623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pile_splitting_l846_84620

/-- Two natural numbers are similar if they differ by no more than a factor of two -/
def similar (a b : ℕ) : Prop := 
  a ≤ 2 * b ∧ b ≤ 2 * a

/-- A valid split is a list of natural numbers where any two elements are similar -/
def valid_split (l : List ℕ) : Prop :=
  ∀ a b, a ∈ l → b ∈ l → similar a b

/-- A sequence of splits is valid if each split in the sequence is valid -/
def valid_split_sequence (s : List (List ℕ)) : Prop :=
  ∀ split, split ∈ s → valid_split split

theorem pile_splitting (n : ℕ) : 
  ∃ (s : List (List ℕ)), 
    valid_split_sequence s ∧ 
    s.head? = some [n] ∧ 
    s.reverse.head? = some (List.replicate n 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pile_splitting_l846_84620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l846_84645

/-- The area of a triangle given its vertices' coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Theorem: The area of the right triangle ABC is 22 square units -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (3, 4)
  let B : ℝ × ℝ := (-1, 2)
  let C : ℝ × ℝ := (5, -6)
  triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = 22 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l846_84645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l846_84614

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the function h
def h (f : ℝ → ℝ) (m : ℝ) (x : ℝ) : ℝ := 2 * f x + 1 - m

theorem function_and_range_proof 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < Real.pi) 
  (h_max : f A ω φ (Real.pi / 12) = 3) 
  (h_min : f A ω φ (7 * Real.pi / 12) = -3) :
  (∃ (m : ℝ), 
    (1 + 3 * Real.sqrt 3 ≤ m ∧ m < 7) ∧ 
    (∃ (x₁ x₂ : ℝ), 
      x₁ ≠ x₂ ∧ 
      -Real.pi / 3 ≤ x₁ ∧ x₁ ≤ Real.pi / 6 ∧ 
      -Real.pi / 3 ≤ x₂ ∧ x₂ ≤ Real.pi / 6 ∧ 
      h (f 3 2 (Real.pi / 3)) m x₁ = 0 ∧ 
      h (f 3 2 (Real.pi / 3)) m x₂ = 0)) ∧
  (∀ (x : ℝ), f A ω φ x = f 3 2 (Real.pi / 3) x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l846_84614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_is_borel_l846_84681

open MeasureTheory

theorem continuous_is_borel {n : ℕ} (hn : n ≥ 1) (f : ℝ → ℝ) (hf : Continuous f) :
  Measurable f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_is_borel_l846_84681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_percentage_l846_84606

/-- Represents a solution with a given volume and alcohol percentage -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the volume of alcohol in a given solution -/
noncomputable def alcoholVolume (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage / 100

/-- Calculates the percentage of alcohol in a mixture of two solutions -/
noncomputable def mixturePercentage (s1 s2 : Solution) : ℝ :=
  (alcoholVolume s1 + alcoholVolume s2) / (s1.volume + s2.volume) * 100

theorem alcohol_mixture_percentage :
  let x : Solution := ⟨300, 10⟩
  let y : Solution := ⟨900, 30⟩
  mixturePercentage x y = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_percentage_l846_84606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_sin_2x0_l846_84666

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * (Real.cos (Real.pi / 2 + x))^2 - 2 * Real.sin (Real.pi + x) * Real.cos x - Real.sqrt 3

theorem extreme_values_and_sin_2x0 :
  (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ f x = 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ f x = 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 1 ≤ f x ∧ f x ≤ 2) ∧
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (3 * Real.pi / 4) Real.pi →
    f (x₀ - Real.pi / 6) = 10 / 13 →
    Real.sin (2 * x₀) = -(5 + 12 * Real.sqrt 3) / 26) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_sin_2x0_l846_84666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_theorem_l846_84662

/-- Calculates the average cost per pencil in cents, rounded to the nearest whole number -/
def average_cost_per_pencil (num_pencils : ℕ) (initial_cost shipping_cost : ℚ) (discount_rate : ℚ) : ℕ :=
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_cost := discounted_cost + shipping_cost
  let total_cents := (total_cost * 100).floor
  let avg_cost := total_cents / num_pencils
  avg_cost.toNat  -- Changed from avg_cost.round to avg_cost.toNat

/-- Theorem stating that the average cost per pencil is 11 cents under the given conditions -/
theorem pencil_cost_theorem :
  average_cost_per_pencil 300 (28.5) (8.25) (0.1) = 11 := by
  -- Unfold the definition and simplify
  unfold average_cost_per_pencil
  -- Perform the calculation steps
  simp [Rat.floor]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_theorem_l846_84662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_percentage_increase_l846_84617

/-- Represents the bike ride scenario -/
structure BikeRide where
  first_hour : ℝ
  second_hour : ℝ
  third_hour : ℝ
  total_distance : ℝ

/-- Calculates the percentage increase between two values -/
noncomputable def percentage_increase (old_value : ℝ) (new_value : ℝ) : ℝ :=
  (new_value - old_value) / old_value * 100

/-- Theorem stating the conditions and the result to be proven -/
theorem bike_ride_percentage_increase (ride : BikeRide) 
  (h1 : ride.second_hour = 24)
  (h2 : ride.first_hour < ride.second_hour)
  (h3 : ride.third_hour = 1.25 * ride.second_hour)
  (h4 : ride.total_distance = ride.first_hour + ride.second_hour + ride.third_hour)
  (h5 : ride.total_distance = 74) :
  percentage_increase ride.first_hour ride.second_hour = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_percentage_increase_l846_84617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_range_of_c_l846_84680

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Helper function for the derivative of f
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Part I: Tangent line equation
theorem tangent_line_at_zero (a b c : ℝ) :
  ∃ (m k : ℝ), ∀ x, m * x + k = f a b c x + (f' a b c 0) * (x - 0) - f a b c 0 ∧ m = b ∧ k = c :=
by
  sorry

-- Part II: Range of c
theorem range_of_c :
  ∃ (lower upper : ℝ), lower = 0 ∧ upper = 32/27 ∧
  ∀ c, (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f 4 4 c x = 0 ∧ f 4 4 c y = 0 ∧ f 4 4 c z = 0) ↔
       (lower < c ∧ c < upper) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_range_of_c_l846_84680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_binomial_unique_l846_84660

/-- Represents a binomial expression -/
structure Binomial where
  a : ℝ
  b : ℝ

/-- Checks if an expression is a perfect square of a binomial -/
def is_square_of_binomial (expr : Binomial × Binomial) : Prop :=
  expr.1 = expr.2

/-- The given expressions -/
def expressions : List (Binomial × Binomial) :=
  [({ a := 1,  b := 1},  { a := -1, b := 1}),   -- (x+y)(-x+y)
   ({ a := 2,  b := -1}, { a := 1,  b := 2}),   -- (2x-y)(x+2y)
   ({ a := 2,  b := -3}, { a := 2,  b := -3}),  -- (2m-3n)(2m-3n)
   ({ a := -2, b := 1},  { a := -2, b := -1})]  -- (-2x+y)(-2y-x)

/-- The theorem to prove -/
theorem square_of_binomial_unique :
  ∃! expr, expr ∈ expressions ∧ is_square_of_binomial expr :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_binomial_unique_l846_84660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_starts_third_year_l846_84673

/-- Represents the financial state of the fishing company -/
structure FishingCompany where
  initialCost : ℕ
  annualIncome : ℕ
  firstYearExpense : ℕ
  annualExpenseIncrease : ℕ

/-- Calculates the cumulative profit up to a given year -/
def cumulativeProfit (c : FishingCompany) (year : ℕ) : ℤ :=
  (c.annualIncome * year : ℤ) - 
  (c.initialCost : ℤ) - 
  ((c.firstYearExpense * year + c.annualExpenseIncrease * (year * (year - 1) / 2)) : ℤ)

/-- Theorem stating that the company starts to make a profit in the third year -/
theorem profit_starts_third_year (c : FishingCompany) 
  (h1 : c.initialCost = 980000)
  (h2 : c.annualIncome = 500000)
  (h3 : c.firstYearExpense = 120000)
  (h4 : c.annualExpenseIncrease = 40000) :
  cumulativeProfit c 3 > 0 ∧ ∀ y : ℕ, y < 3 → cumulativeProfit c y ≤ 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_starts_third_year_l846_84673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_ellipse_l846_84658

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polar_to_cartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ, y := p.ρ * Real.sin p.θ }

/-- The given polar equation -/
def polar_equation (p : PolarPoint) : Prop :=
  p.ρ^2 = 12 / (3 * (Real.cos p.θ)^2 + 4 * (Real.sin p.θ)^2)

/-- The scaling transformation -/
noncomputable def scaling_transform (p : CartesianPoint) : CartesianPoint :=
  { x := p.x / 2, y := p.y * Real.sqrt 3 / 3 }

/-- Predicate to check if a point is on an ellipse -/
def is_on_ellipse (p : CartesianPoint) : Prop :=
  p.x^2 / (3/4) + p.y^2 / (4/3) = 1

/-- The main theorem to prove -/
theorem polar_to_ellipse :
  ∀ p : PolarPoint, 
    polar_equation p → 
    is_on_ellipse (scaling_transform (polar_to_cartesian p)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_ellipse_l846_84658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_prime_power_l846_84627

theorem arithmetic_progression_prime_power (n : ℕ) (a : ℕ → ℕ) 
  (h_arithmetic : ∃ (k d : ℤ), ∀ i, (a i : ℤ) = k + d * i)
  (h_divides : ∀ i, 1 ≤ i → i < n → i ∣ a i)
  (h_not_divides : ¬(n ∣ a n)) :
  ∃ p : ℕ, Prime p ∧ ∃ k : ℕ, n = p ^ k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_prime_power_l846_84627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l846_84646

/-- Represents a segment of the journey --/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the average speed given a list of journey segments --/
noncomputable def averageSpeed (segments : List Segment) : ℝ :=
  let totalDistance := segments.foldl (fun acc s => acc + s.speed * s.duration) 0
  let totalTime := segments.foldl (fun acc s => acc + s.duration) 0
  totalDistance / totalTime

/-- The journey from City A to City B --/
def journey : List Segment := [
  ⟨40, 1⟩,   -- City A to City C
  ⟨30, 2⟩,   -- City C to City D
  ⟨60, 3⟩,   -- City D to some point
  ⟨50, 1⟩,   -- Through intermittent heavy rain
  ⟨70, 1.5⟩  -- Last point to City B
]

/-- Theorem stating that the average speed of the journey is approximately 51.18 mph --/
theorem journey_average_speed :
  abs (averageSpeed journey - 51.18) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l846_84646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_negative_s_l846_84624

def is_valid_p (p : ℕ) : Prop := 1 ≤ p ∧ p ≤ 10

def s (p : ℕ) : ℤ := p^2 - 13*p + 40

def count_negative_s : ℕ := (Finset.range 10).filter (λ p => s (p + 1) < 0) |>.card

theorem probability_negative_s :
  (count_negative_s : ℚ) / 10 = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_negative_s_l846_84624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l846_84616

/-- The constant term in the binomial expansion of (3x - 2/x)^8 -/
def constant_term : ℕ := 112

/-- The binomial expansion of (3x - 2/x)^8 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (3*x - 2/x)^8

/-- Theorem stating that the constant term in the binomial expansion of (3x - 2/x)^8 is 112 -/
theorem constant_term_of_binomial_expansion :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = binomial_expansion x) ∧ 
  (∃ c : ℝ, c = constant_term ∧ ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) :=
by
  sorry

#check constant_term_of_binomial_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l846_84616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l846_84692

open Real

noncomputable def ω : ℝ := sorry

noncomputable def f (x : ℝ) : ℝ := sin (ω * x) * cos (ω * x) + Real.sqrt 3 * sin (ω * x)^2 - Real.sqrt 3 / 2

noncomputable def g (x : ℝ) : ℝ := f (x + π / 6) - 1 / 2

theorem max_triangle_area 
  (h_ω : ω > 0)
  (h_period : ∀ x, f (x + π) = f x)
  (h_g : g (A / 2) = 0)
  (h_a : a = 1)
  (h_acute : 0 < A ∧ A < π / 2)
  (h_triangle : 0 < B ∧ 0 < C ∧ A + B + C = π) :
  ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ 
    (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
      1 / 2 * b' * c' * sin A ≤ 1 / 2 * b * c * sin A) ∧
    1 / 2 * b * c * sin A = (2 + Real.sqrt 3) / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l846_84692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_and_area_l846_84603

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + c

theorem function_roots_and_area (c : ℝ) :
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ f c r₁ = 0 ∧ f c r₂ = 0) →
  (c = -5/27 ∨ c = 1) ∧
  (c = 1 ∧ f c (-1/3) > 0 →
    ∫ x in Set.Icc (-1) 1, |f c x| = 4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_and_area_l846_84603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_divisible_by_24_l846_84685

theorem divisor_sum_divisible_by_24 (n : ℕ) (h : 24 ∣ (n + 1)) :
  24 ∣ (Finset.sum (Nat.divisors n) id) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_divisible_by_24_l846_84685
