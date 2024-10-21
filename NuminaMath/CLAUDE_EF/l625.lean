import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l625_62579

/-- The time (in seconds) it takes for two trains to meet -/
noncomputable def time_to_meet (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := length1 + length2 + initial_distance
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  total_distance / relative_speed

/-- Theorem stating that the time for two trains to meet is approximately 9.77 seconds -/
theorem trains_meet_time :
  let length1 : ℝ := 120
  let length2 : ℝ := 210
  let initial_distance : ℝ := 80
  let speed1 : ℝ := 69
  let speed2 : ℝ := 82
  abs (time_to_meet length1 length2 initial_distance speed1 speed2 - 9.77) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l625_62579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l625_62568

noncomputable section

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x / (1 - x) + a / ((1 - x) * (1 - a)) + (b - x) / ((1 - x) * (1 - a) * (1 - b))

theorem f_domain_and_range (ha : a ≠ 1) (hb : b ≠ 1) :
  (∀ x : ℝ, f a b x ≠ 0 ↔ x ∉ ({1, a, b} : Set ℝ)) ∧
  (Set.range (f a b) = {(a + b - a * b) / ((1 - a) * (1 - b))}) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l625_62568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l625_62545

-- Define the circle equation
def circle_equation (ρ θ : Real) : Prop := ρ = 2 * Real.cos θ

-- Define the line equation
def line_equation (θ : Real) : Prop := θ = Real.pi / 3

-- Define the constraint
def positive_ρ (ρ : Real) : Prop := ρ > 0

-- Theorem statement
theorem intersection_point :
  ∃ (ρ θ : Real),
    circle_equation ρ θ ∧
    line_equation θ ∧
    positive_ρ ρ ∧
    ρ = 1 ∧
    θ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l625_62545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_comparison_l625_62598

noncomputable section

-- Define the inverse proportion function
def f (x : ℝ) : ℝ := 3 / x

-- Define the coordinates of points A and B
def x₁ : ℝ := -1
def x₂ : ℝ := -3

-- State the theorem
theorem inverse_proportion_comparison :
  ∀ y₁ y₂ : ℝ,
  (f x₁ = y₁) → (f x₂ = y₂) →
  y₁ < y₂ := by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_comparison_l625_62598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_at_x1_l625_62537

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - Real.log x / Real.log 3

-- State the theorem
theorem f_positive_at_x1 
  (x₀ : ℝ) 
  (x₁ : ℝ) 
  (h₀ : f x₀ = 0) 
  (h₁ : 0 < x₁) 
  (h₂ : x₁ < x₀) : 
  f x₁ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_at_x1_l625_62537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_minus_x_l625_62560

theorem integral_exp_minus_x : ∫ x in (Set.Icc 0 1), Real.exp (-x) = 1 - Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_minus_x_l625_62560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_scores_probability_l625_62524

/-- Represents the probability distribution of a student's score -/
structure ScoreDistribution :=
  (p0 p1 p2 p3 p4 p5 : ℚ)
  (sum_to_one : p0 + p1 + p2 + p3 + p4 + p5 = 1)

/-- The probability distributions for each student -/
def student_distributions : List ScoreDistribution :=
  [
    ⟨0, 0, 1/4, 1/2, 1/4, 0, by norm_num⟩,
    ⟨0, 1/2, 1/2, 0, 0, 0, by norm_num⟩,
    ⟨0, 0, 0, 0, 0, 1, by norm_num⟩,
    ⟨0, 0, 0, 0, 1/2, 1/2, by norm_num⟩,
    ⟨0, 1/4, 1/2, 1/4, 0, 0, by norm_num⟩,
    ⟨1/4, 1/2, 1/4, 0, 0, 0, by norm_num⟩,
    ⟨0, 0, 0, 1/4, 1/2, 1/4, by norm_num⟩
  ]

/-- The probability that the sum of scores is exactly 20 -/
def prob_sum_20 : ℚ := 105/512

theorem sum_of_scores_probability :
  prob_sum_20 = 105/512 := by
  -- The proof goes here
  sorry

#eval prob_sum_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_scores_probability_l625_62524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_transport_time_l625_62544

/-- Calculates the total driving time to transport cattle to safety -/
noncomputable def total_driving_time (total_cattle : ℕ) (truck_capacity : ℕ) (distance : ℝ) (speed : ℝ) : ℝ :=
  let num_trips := (total_cattle + truck_capacity - 1) / truck_capacity
  let round_trip_time := 2 * distance / speed
  (num_trips : ℝ) * round_trip_time

/-- The total driving time to transport 400 cattle with given conditions is 40 hours -/
theorem cattle_transport_time :
  total_driving_time 400 20 60 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_transport_time_l625_62544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l625_62549

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

def tangent_line (a : ℝ) (x y : ℝ) : Prop := y = (Real.exp a) * x

theorem tangent_through_origin :
  ∃ a : ℝ, 
    (tangent_line a 0 0) ∧ 
    (∃ x : ℝ, x ≠ 0 ∧ tangent_line a x (curve x)) ∧
    (∀ x y : ℝ, tangent_line a x y ↔ Real.exp x - y = 0) := by
  sorry

#check tangent_through_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l625_62549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_simplification_l625_62526

theorem sine_cosine_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_simplification_l625_62526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l625_62592

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 4)

theorem f_properties :
  -- Smallest positive period
  (∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  (let T := 2 * Real.pi / 3; 
    T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  
  -- Monotonically increasing intervals
  (∀ k : ℤ, ∀ x y : ℝ, 
    -2 * k * Real.pi / 3 - Real.pi / 4 ≤ x ∧ x < y ∧ y ≤ 2 * k * Real.pi / 3 + Real.pi / 12 → f x < f y) ∧
  
  -- Maximum and minimum values in [-π/6, π/6]
  (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 2) ∧
  (∃ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 6 ∧ f x = 2) ∧
  (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 6 → f x ≥ -Real.sqrt 2) ∧
  (∃ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 6 ∧ f x = -Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l625_62592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_bound_l625_62570

-- Define the sequence x_n
noncomputable def x : ℕ → ℝ
  | 0 => 0  -- Define a value for 0 to cover all natural numbers
  | 1 => 0  -- Define a value for 1 to cover all natural numbers
  | 2 => Real.sqrt (2 + 1/2)
  | 3 => Real.sqrt (2 + (3 + 1/3) ^ (1/3))
  | 4 => Real.sqrt (2 + (3 + (4 + 1/4) ^ (1/4)) ^ (1/3))
  | n + 1 => Real.sqrt (2 + (3 + (4 + (n + 1/n) ^ (1/n)) ^ (1/4)) ^ (1/3))

-- State the theorem
theorem x_difference_bound (n : ℕ) (h : n ≥ 2) : x (n + 1) - x n < 1 / n.factorial := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_difference_bound_l625_62570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l625_62589

/-- Represents a quadrilateral ABCD with specific side lengths and a right angle -/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (angle_CDA : ℝ)

/-- The area of a quadrilateral ABCD with given properties -/
noncomputable def quadrilateral_area (q : Quadrilateral) : ℝ :=
  84.5 + 32.5 * Real.sqrt 2

/-- Theorem stating that for a quadrilateral with given properties, its area is 84.5 + 32.5√2 -/
theorem quadrilateral_area_theorem (q : Quadrilateral) 
  (h1 : q.AB = 10)
  (h2 : q.BC = 5)
  (h3 : q.CD = 13)
  (h4 : q.DA = 13)
  (h5 : q.angle_CDA = 90) :
  quadrilateral_area q = 84.5 + 32.5 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l625_62589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l625_62557

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2)^(x-1) else (1/2)^(-x-1)

-- State the theorem
theorem solution_set_of_inequality (h1 : ∀ x, f (-x) = f x) :
  {x : ℝ | f x - x^2 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l625_62557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_a_range_l625_62597

open Real

/-- The function f(x) = (x+1)e^x - a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * exp x - a

/-- f(x) has exactly two zeros -/
def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
  ∀ x, f a x = 0 → x = x₁ ∨ x = x₂

theorem f_two_zeros_a_range (a : ℝ) :
  has_exactly_two_zeros a → -1 / exp 2 < a ∧ a < 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_a_range_l625_62597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_five_l625_62525

theorem tan_sum_equals_five (α : ℝ) : 
  Real.tan (-α - 4/3 * Real.pi) = -5 → Real.tan (Real.pi/3 + α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_five_l625_62525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_example_l625_62523

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℤ) : ℕ :=
  Int.toNat ((aₙ - a₁) / d + 1)

/-- Theorem: The arithmetic sequence starting with 3, ending with 58, 
    and having a common difference of 5 contains 12 numbers. -/
theorem arithmetic_sequence_length_example : 
  arithmetic_sequence_length 3 58 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_example_l625_62523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernardo_wins_with_smallest_number_smallest_winning_number_properties_smallest_winning_number_is_nine_l625_62567

theorem bernardo_wins_with_smallest_number : 
  ∃ M : ℕ, 
    (M ≤ 1199) ∧ 
    (27 * M + 900 < 1200) ∧ 
    (27 * M + 975 ≥ 1200) ∧ 
    (∀ N : ℕ, N < M → (27 * N + 900 < 1200 → 27 * N + 975 < 1200)) :=
by
  -- The proof goes here
  sorry

-- Use 'noncomputable def' instead of '#eval' for the choose function
noncomputable def smallest_winning_number := (bernardo_wins_with_smallest_number).choose

-- Add a theorem to state the properties of the smallest winning number
theorem smallest_winning_number_properties :
  let M := smallest_winning_number
  (M ≤ 1199) ∧ 
  (27 * M + 900 < 1200) ∧ 
  (27 * M + 975 ≥ 1200) ∧ 
  (∀ N : ℕ, N < M → (27 * N + 900 < 1200 → 27 * N + 975 < 1200)) :=
by
  -- The proof goes here
  sorry

-- Add a theorem to state that the smallest winning number is 9
theorem smallest_winning_number_is_nine :
  smallest_winning_number = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernardo_wins_with_smallest_number_smallest_winning_number_properties_smallest_winning_number_is_nine_l625_62567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l625_62574

noncomputable def star (a b : ℝ) : ℝ := (10 : ℝ)^a * (10 : ℝ)^b

theorem star_properties :
  (star 12 3 = (10 : ℝ)^15) ∧
  (star 4 8 = (10 : ℝ)^12) ∧
  (∀ a b c : ℝ, star (a + b) c = star a (b + c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l625_62574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_theorems_l625_62530

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (inPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelPlanePlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLinePlane : Line → Plane → Prop)
variable (perpendicularPlanePlane : Plane → Plane → Prop)
variable (planeMeet : Plane → Plane → Line → Prop)

-- Theorem statement
theorem plane_theorems 
  (α β : Plane) 
  (h_distinct : α ≠ β) :
  (∀ (l1 l2 l3 l4 : Line), 
    inPlane l1 α ∧ inPlane l2 α ∧ intersect l1 l2 ∧ 
    inPlane l3 β ∧ inPlane l4 β ∧ 
    parallel l1 l3 ∧ parallel l2 l4 → 
    parallelPlanePlane α β) ∧ 
  (∀ (l m : Line), 
    ¬inPlane l α ∧ inPlane m α ∧ parallel l m → 
    parallelLinePlane l α) ∧
  ¬(∀ (l m : Line), 
    planeMeet α β l ∧ inPlane m α ∧ perpendicular m l → 
    perpendicularPlanePlane α β) ∧
  ¬(∀ (l m n : Line), 
    inPlane m α ∧ inPlane n α ∧ intersect m n ∧ 
    perpendicular l m ∧ perpendicular l n ↔ 
    perpendicularLinePlane l α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_theorems_l625_62530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l625_62504

-- Define the necessary structures and functions if they're not already in Mathlib
structure Point where
  x : ℝ
  y : ℝ

def dist (A B : Point) : ℝ := sorry
def angle (A B C : Point) : ℝ := sorry
def area (t : Point × Point × Point) : ℝ := sorry

theorem right_triangle_properties (A B C : Point) 
  (h_right : angle B A C = Real.pi / 2)
  (h_sin : Real.sin (angle B A C) = 3/5) 
  (h_hypotenuse : dist A C = 15) :
  dist A B = 9 ∧ dist B C = 12 ∧ area (A, B, C) = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l625_62504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l625_62539

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x ≤ 0 then x^2 + 6*x
  else 0  -- This else case is added to make the function total

theorem domain_of_f : 
  Set.range f = Set.Icc (-2 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l625_62539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l625_62559

theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1 / 2) * L * W) = 2 := by
  intros L W hL hW
  field_simp
  ring
  
#check rectangle_triangle_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l625_62559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_relation_on_line_l625_62575

/-- Given two points M(2,a) and N(3,b) on the line y = 2x + 1, prove that a < b -/
theorem point_relation_on_line (a b : ℝ) : 
  (2, a) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} →
  (3, b) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} →
  a < b :=
by
  intros h1 h2
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_relation_on_line_l625_62575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l625_62594

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x / 4 + 5
def g (x : ℝ) : ℝ := 4 - x

-- State the theorem
theorem find_a : ∀ a : ℝ, f (g a) = 7 → a = -4 := by
  intro a h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l625_62594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l625_62510

/-- The parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- The distance from a point to the y-axis -/
def distToYAxis (p : ℝ × ℝ) : ℝ := |p.1|

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_distance
  (P : ℝ × ℝ)
  (h₁ : P ∈ Parabola)
  (h₂ : distToYAxis P = 3) :
  distance P Focus = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l625_62510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l625_62590

/-- Given an angle α in the second quadrant and a point P(x,4) on its terminal side
    where cos(α) = (1/5)x, prove that sin(α) = 4/5 -/
theorem sin_alpha_value (α : Real) (x : Real) 
    (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
    (h2 : x < 0)  -- x-coordinate is negative in second quadrant
    (h3 : Real.cos α = (1/5) * x) : 
  Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l625_62590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_one_l625_62556

-- Define the equilateral triangle
def triangle_side_length : ℝ := 2

-- Define the points of the triangle
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency condition
def is_tangent_to_line (c : Circle) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ p, l p ∧ (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

-- Define the x-axis and y-axis
def x_axis (p : ℝ × ℝ) : Prop := p.2 = 0
def y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

-- Define the side AB of the triangle
def side_AB (p : ℝ × ℝ) : Prop := 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ p.2 = 0

-- State the theorem
theorem circle_radius_is_one (c : Circle) :
  is_tangent_to_line c side_AB ∧
  is_tangent_to_line c x_axis ∧
  is_tangent_to_line c y_axis →
  c.radius = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_one_l625_62556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfying_condition_l625_62595

def satisfies_condition (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ w : Fin 2 → ℝ, N.mulVec w = (-7 : ℝ) • w

theorem matrix_satisfying_condition (N : Matrix (Fin 2) (Fin 2) ℝ) :
  satisfies_condition N → N = ![![-7, 0], ![0, -7]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfying_condition_l625_62595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_cutting_l625_62529

theorem ribbon_cutting (num_spools : ℕ) (spool_length : ℝ) (piece_length : ℝ) : 
  num_spools = 5 → 
  spool_length = 60 → 
  piece_length = 1.5 → 
  ⌊num_spools * (spool_length / piece_length - 1)⌋ = 195 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_cutting_l625_62529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_even_l625_62509

/-- A point with integer coordinates in a plane -/
structure IntPoint where
  x : Int
  y : Int

/-- A pentagon with vertices having integer coordinates -/
structure IntPentagon where
  vertices : Fin 5 → IntPoint

/-- The side length between two points -/
def sideLength (a b : IntPoint) : Int :=
  (a.x - b.x)^2 + (a.y - b.y)^2

/-- The perimeter of a pentagon -/
def perimeter (p : IntPentagon) : Int :=
  (Finset.sum (Finset.range 5) fun i => sideLength (p.vertices i) (p.vertices (i + 1)))

theorem pentagon_perimeter_even (p : IntPentagon) 
  (h : ∀ i : Fin 5, Odd (sideLength (p.vertices i) (p.vertices ((i + 1) % 5)))) :
  Even (perimeter p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_even_l625_62509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_determines_length_l625_62542

/-- Represents a 3D region composed of a cylinder and two cones -/
structure Region3D where
  cylinderRadius : ℝ
  coneBaseRadius : ℝ
  coneHeight : ℝ
  cylinderLength : ℝ

/-- Calculates the volume of the region -/
noncomputable def volume (r : Region3D) : ℝ :=
  Real.pi * r.cylinderRadius^2 * r.cylinderLength + 
  (2/3) * Real.pi * r.coneBaseRadius^2 * r.coneHeight

/-- Theorem stating the relationship between the region's dimensions and volume -/
theorem volume_determines_length (r : Region3D) 
  (h1 : r.cylinderRadius = 4)
  (h2 : r.coneBaseRadius = 4)
  (h3 : r.coneHeight = 4)
  (h4 : volume r = 288 * Real.pi) :
  r.cylinderLength = 46/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_determines_length_l625_62542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l625_62516

def S : Finset ℕ := {1, 2, 3, 4, 5}

theorem number_of_subsets : 
  (Finset.filter (fun X => {1, 2} ⊆ X ∧ X ⊆ S) (Finset.powerset S)).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l625_62516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_given_number_l625_62591

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Represents the place value in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands

/-- Definition for the given number -/
def given_number : ScientificNotation :=
  { coefficient := 1.50, exponent := 5 }

/-- Definition for significant figures -/
def significant_figures (n : ScientificNotation) : List Nat :=
  sorry

/-- Definition for place of accuracy -/
def place_of_accuracy (n : ScientificNotation) : PlaceValue :=
  sorry

/-- Theorem stating the properties of the given number -/
theorem properties_of_given_number :
  (place_of_accuracy given_number = PlaceValue.Thousands) ∧
  (significant_figures given_number = [1, 5, 0]) ∧
  (significant_figures given_number).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_given_number_l625_62591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l625_62550

noncomputable def f (x : ℝ) := Real.cos x * Real.sin x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1) ∧
  (f (-Real.pi/12) = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l625_62550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l625_62564

/-- Given two circles of radius 5 that are externally tangent to each other 
    and internally tangent to a larger circle of radius 13, 
    the distance between their points of tangency with the larger circle is 65/4 -/
theorem tangent_circles_distance (r₁ r₂ R : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 5) (h₃ : R = 13) : 
  let d := R - r₁
  let AB := (2 * R * r₁) / d
  AB = 65 / 4 ∧ Int.gcd 65 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l625_62564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_solution_l625_62507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^2 - 1
  else if -1 < x ∧ x < 0 then -(x^2 - 1)
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem odd_function_solution :
  ∀ (x₀ : ℝ),
  (∀ x, -1 < x ∧ x < 1 → f (-x) = -f x) →  -- f is odd on (-1,1)
  (∀ x, 0 < x ∧ x < 1 → f x = x^2 - 1) →   -- f(x) = x^2 - 1 for x in (0,1)
  f x₀ = 1/2 →                             -- f(x₀) = 1/2
  x₀ = -Real.sqrt 2/2 :=                   -- x₀ = -√2/2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_solution_l625_62507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_eq_neg_two_l625_62511

/-- Given a function f such that f(2x+1) = 3x-5 for all x, prove that f(3) = -2 -/
theorem f_of_three_eq_neg_two (f : ℝ → ℝ) (h : ∀ x, f (2*x+1) = 3*x-5) : f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_eq_neg_two_l625_62511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l625_62508

-- Define the basic geometric objects
structure Line
structure Plane
structure Parallelepiped
structure RightPrism
structure RightAngledTriangle
structure Cone
structure SolidOfRevolution

-- Define the geometric relations
def isParallelTo : Line → Line → Prop := sorry
def isParallelToPlane : Line → Plane → Prop := sorry
def isSubsetOf : Line → Plane → Prop := sorry
def areSkew : Line → Line → Prop := sorry
def hasRectangularFaces : Parallelepiped → Nat → Prop := sorry
def rotateAboutLeg : RightAngledTriangle → Line → SolidOfRevolution := sorry

-- Define the propositions
def proposition1 (p : Parallelepiped) : Prop :=
  hasRectangularFaces p 2 → p ∈ Set.univ

def proposition2 (t : RightAngledTriangle) (l : Line) : Prop :=
  rotateAboutLeg t l ∈ Set.univ

def proposition3 (a b : Line) (α : Plane) : Prop :=
  isSubsetOf b α → isParallelTo a b → isParallelToPlane a α

def proposition4 : Prop :=
  ∃ (a b : Line) (α β : Plane),
    areSkew a b ∧
    isSubsetOf a α ∧
    isSubsetOf b β ∧
    isParallelToPlane a β ∧
    isParallelToPlane b α

theorem geometric_propositions :
  (¬ ∀ p, proposition1 p) ∧
  (∀ t l, proposition2 t l) ∧
  (¬ ∀ a b α, proposition3 a b α) ∧
  proposition4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l625_62508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotone_interval_l625_62562

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := -Real.cos (2 * x)

theorem max_monotone_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, Monotone (fun x ↦ g x)) ↔ a ≤ Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotone_interval_l625_62562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_all_reals_l625_62520

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^3 - 0.5*x^2) / (x^2 + 2*x + 2)

-- State the theorem
theorem range_of_f_is_all_reals :
  ∀ y : ℝ, ∃ x : ℝ, f x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_all_reals_l625_62520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_for_equation_l625_62553

theorem valid_pairs_for_equation (a b : ℕ+) (q r : ℕ) : 
  (a.val^2 + b.val^2 = q * (a.val + b.val) + r) →
  (0 ≤ r ∧ r < a.val + b.val) →
  (q^2 + r = 1977) →
  ((a.val = 50 ∧ b.val = 37) ∨ 
   (a.val = 50 ∧ b.val = 7) ∨ 
   (a.val = 37 ∧ b.val = 50) ∨ 
   (a.val = 7 ∧ b.val = 50)) := by
  sorry

#check valid_pairs_for_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_for_equation_l625_62553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_BC_proof_l625_62528

/-- The distance between points B and C in kilometers. -/
noncomputable def distance_BC : ℝ := 290

/-- The average speed from A to C in km/h. -/
noncomputable def speed_AC : ℝ := 75

/-- The average speed from C to B in km/h. -/
noncomputable def speed_CB : ℝ := 145

/-- The total time from A to B in hours. -/
noncomputable def time_AB : ℝ := 4.8

/-- The average return speed from B to C in km/h. -/
noncomputable def return_speed_BC : ℝ := 100

/-- The time from B to C on return in hours. -/
noncomputable def return_time_BC : ℝ := 2

/-- The average return speed from C to A in km/h. -/
noncomputable def return_speed_CA : ℝ := 70

/-- The distance between points A and C in kilometers. -/
noncomputable def distance_AC : ℝ := time_AB * speed_AC * speed_CB / (speed_AC + speed_CB)

theorem distance_BC_proof : 
  distance_BC = return_speed_BC * return_time_BC ∧
  distance_AC / speed_AC + distance_BC / speed_CB = time_AB ∧
  (distance_AC + distance_BC) / return_speed_BC = return_time_BC + distance_AC / return_speed_CA :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_BC_proof_l625_62528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_comparison_l625_62577

-- Define the triangles and hexagons
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define the properties of the hexagons
def is_regular_hexagon (h : Hexagon) : Prop := sorry

def inscribed_in_ABD (h : Hexagon) (t : Triangle) : Prop := sorry

def inscribed_in_CBD (h : Hexagon) (t : Triangle) : Prop := sorry

def sides_on_AB_AD (h : Hexagon) (t : Triangle) : Prop := sorry

def vertex_on_BD (h : Hexagon) (t : Triangle) : Prop := sorry

def vertices_on_CB_CD (h : Hexagon) (t : Triangle) : Prop := sorry

def side_on_BD (h : Hexagon) (t : Triangle) : Prop := sorry

-- Define the area of a hexagon
noncomputable def area (h : Hexagon) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_comparison 
  (t : Triangle) 
  (h1 h2 : Hexagon) 
  (reg1 : is_regular_hexagon h1)
  (reg2 : is_regular_hexagon h2)
  (in_ABD : inscribed_in_ABD h1 t)
  (in_CBD : inscribed_in_CBD h2 t)
  (sides1 : sides_on_AB_AD h1 t)
  (vertex1 : vertex_on_BD h1 t)
  (vertices2 : vertices_on_CB_CD h2 t)
  (side2 : side_on_BD h2 t) :
  area h1 > area h2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_comparison_l625_62577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_f_2_eq_neg_3_l625_62548

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The specific function we're working with -/
noncomputable def f : ℝ → ℝ :=
  λ x => if x < 0 then x^2 - 1 else -(((-x)^2 - 1))

theorem f_is_odd_and_f_2_eq_neg_3 :
  OddFunction f ∧ (∀ x < 0, f x = x^2 - 1) → f 2 = -3 :=
by
  sorry

#check f_is_odd_and_f_2_eq_neg_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_f_2_eq_neg_3_l625_62548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l625_62534

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 2
noncomputable def g (x : ℝ) : ℝ := (8 - 2*x) / 4

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = g p.1}

-- State the theorem
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 4 ∧
    y₁ + y₂ + y₃ = 4 := by
  sorry

#check intersection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l625_62534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_ten_factors_l625_62532

def has_exactly_n_factors (n : ℕ) (k : ℕ) : Prop :=
  (Finset.filter (λ i ↦ k % i = 0) (Finset.range (k + 1))).card = n

theorem least_integer_with_ten_factors :
  ∀ k : ℕ, k > 0 → has_exactly_n_factors 10 k → k ≥ 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_ten_factors_l625_62532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_margaret_speed_problem_l625_62599

theorem margaret_speed_problem (d t : ℝ) 
  (h1 : d = 50 * (t - 1/12))
  (h2 : d = 30 * (t + 1/12)) :
  d / t = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_margaret_speed_problem_l625_62599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_20_10_l625_62563

noncomputable section

/-- The amount of metal wasted when cutting a circular piece from a rectangular sheet
    and then inscribing a rectangle in the circular piece. -/
def metal_waste (length width : ℝ) : ℝ :=
  let radius := min length width / 2
  let circle_area := Real.pi * radius^2
  let inscribed_rect_width := (2 * radius) / Real.sqrt 5
  let inscribed_rect_length := 2 * inscribed_rect_width
  let inscribed_rect_area := inscribed_rect_width * inscribed_rect_length
  length * width - inscribed_rect_area

/-- Theorem stating that the metal waste for a 20x10 rectangular sheet is 160 square units. -/
theorem metal_waste_20_10 :
  metal_waste 20 10 = 160 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_20_10_l625_62563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_between_lines_l625_62505

/-- The distance between two parallel lines given by equations ax + by + c1 = 0 and ax + by + c2 = 0 -/
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

/-- The function representing the squared distance between l1 and l2 -/
def squared_distance (t : ℝ) : ℝ :=
  (2*t^2 - 2*t + 3)^2

/-- Theorem stating that t = 1/2 minimizes the squared distance -/
theorem minimize_distance_between_lines :
  ∃ (t : ℝ), t = 1/2 ∧
  ∀ (s : ℝ), squared_distance t ≤ squared_distance s := by
  sorry

#check minimize_distance_between_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_between_lines_l625_62505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_seven_fourths_squared_l625_62533

theorem ceiling_seven_fourths_squared : ⌈((7:ℚ)/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_seven_fourths_squared_l625_62533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defendant_not_guilty_l625_62543

-- Define the types of witnesses and their possible states
inductive Witness : Type
| A
| B
| C

inductive State : Type
| Human
| Zombie

-- Define the meaning of "бал"
def bal_meaning (w : Witness) (s : State) : Prop :=
  match s with
  | State.Human => (w = Witness.B) → ("бал" = "yes")
  | State.Zombie => (w = Witness.B) → ("бал" = "no")

-- Define the condition that if C is a zombie, A and B are brothers
def C_zombie_implies_AB_brothers (state : Witness → State) : Prop :=
  state Witness.C = State.Zombie →
    ((state Witness.A = State.Zombie ∧ state Witness.B = State.Zombie) ∨
     (state Witness.A = State.Human ∧ state Witness.B = State.Human))

-- Define A's response
def A_response (state : Witness → State) : Prop :=
  match state Witness.A with
  | State.Human => "бал" = "yes"
  | State.Zombie => "бал" = "no"

-- Define whether the defendant is guilty
def defendant_guilty : Prop := False

-- Define C's statement
def C_statement (state : Witness → State) : Prop :=
  match state Witness.C with
  | State.Human => ¬defendant_guilty
  | State.Zombie => defendant_guilty

-- The main theorem
theorem defendant_not_guilty (state : Witness → State)
  (h1 : C_zombie_implies_AB_brothers state)
  (h2 : bal_meaning Witness.B (state Witness.B))
  (h3 : A_response state)
  (h4 : C_statement state) :
  ¬defendant_guilty :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_defendant_not_guilty_l625_62543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_percent_increase_l625_62514

noncomputable def question_value : Fin 15 → ℝ
  | 0 => 150
  | 1 => 250
  | 2 => 400
  | 3 => 550
  | 4 => 1200
  | 5 => 2500
  | 6 => 5000
  | 7 => 10000
  | 8 => 20000
  | 9 => 40000
  | 10 => 80000
  | 11 => 160000
  | 12 => 320000
  | 13 => 650000
  | 14 => 1300000

noncomputable def percent_increase (a b : ℝ) : ℝ :=
  (b - a) / a * 100

def pairs_to_check : List (Fin 15 × Fin 15) :=
  [(0, 1), (2, 3), (3, 4), (12, 13), (13, 14)]

theorem smallest_percent_increase :
  ∀ (pair : Fin 15 × Fin 15), pair ∈ pairs_to_check →
    percent_increase (question_value pair.fst) (question_value pair.snd) ≥
    percent_increase (question_value 2) (question_value 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_percent_increase_l625_62514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_movement_theorem_l625_62500

/-- Represents a chess grid with numbers indicating piece movement --/
structure ChessGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (top_numbers : List ℕ)
  (left_numbers : List ℕ)

/-- Checks if the given numbers are consistent with a valid chess piece movement --/
def is_valid_movement (grid : ChessGrid) : Prop :=
  grid.top_numbers.sum = grid.left_numbers.sum

/-- The specific grid given in the problem --/
def problem_grid (x : ℕ) : ChessGrid :=
  { rows := 4
  , cols := 4
  , top_numbers := [1, 3, 2, 1]
  , left_numbers := [1, 2, 3, x] }

/-- The theorem to be proved --/
theorem chess_movement_theorem : 
  ∃ x : ℕ, is_valid_movement (problem_grid x) ∧ x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_movement_theorem_l625_62500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_sequence_count_l625_62517

theorem number_sequence_count (s : List ℝ) 
  (h1 : s.sum / s.length = 22)
  (h2 : (s.take 6).sum / 6 = 19)
  (h3 : (s.drop (s.length - 6)).sum / 6 = 27)
  (h4 : s.length > 5)
  (h5 : s[5] = 34) :
  s.length = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_sequence_count_l625_62517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_to_sum_2017_l625_62587

/-- Sequence defined by a₁ = 1, a₂ = 2, and aₖ₊₂ = aₖ₊₁ + aₖ for k ∈ ℕ -/
def a : ℕ → ℕ
  | 0 => 1  -- We define a₀ = 1 to handle the zero case
  | 1 => 1
  | 2 => 2
  | n + 3 => a (n + 2) + a (n + 1)

/-- Number of ways to write n as a sum of distinct elements of the sequence a -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 24 ways to write 2017 as a sum of distinct elements of a -/
theorem ways_to_sum_2017 : f 2017 = 24 := by
  sorry

#eval a 5  -- This will evaluate a₅ to check if the function is working correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_to_sum_2017_l625_62587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l625_62581

/-- Given a square, this function returns the area of an inscribed square whose corners
    are placed at points one-quarter and three-quarters along each side of the given square. -/
noncomputable def inscribed_square_area (side_length : ℝ) : ℝ :=
  (side_length / 2) ^ 2

/-- The ratio of the area of the inscribed square to the area of the given square. -/
noncomputable def area_ratio (side_length : ℝ) : ℝ :=
  inscribed_square_area side_length / (side_length ^ 2)

theorem inscribed_square_area_ratio :
  ∀ (side_length : ℝ), side_length > 0 → area_ratio side_length = 1 / 4 := by
  intro side_length h_positive
  unfold area_ratio inscribed_square_area
  simp [h_positive]
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l625_62581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_of_roots_l625_62576

theorem cube_sum_of_roots (u v w : ℝ) : 
  (u - (17 : ℝ)^(1/3)) * (u - (67 : ℝ)^(1/3)) * (u - (97 : ℝ)^(1/3)) = 1/2 →
  (v - (17 : ℝ)^(1/3)) * (v - (67 : ℝ)^(1/3)) * (v - (97 : ℝ)^(1/3)) = 1/2 →
  (w - (17 : ℝ)^(1/3)) * (w - (67 : ℝ)^(1/3)) * (w - (97 : ℝ)^(1/3)) = 1/2 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 184.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_of_roots_l625_62576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_equivalence_l625_62531

theorem sin_transformation_equivalence (x : ℝ) :
  Real.sin (3 * (x - π / 18)) = Real.sin (3 * x - π / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_equivalence_l625_62531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_nine_halves_l625_62536

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

-- State the theorem
theorem sum_of_f_equals_nine_halves :
  f (1/9) + f (1/7) + f (1/5) + f (1/3) + f 1 + f 3 + f 5 + f 7 + f 9 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_nine_halves_l625_62536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_l625_62538

-- Define the fractional part function
noncomputable def fract (x : ℝ) : ℝ := x - ⌊x⌋

theorem existence_of_x (k : ℕ) (h_k : k > 1) : 
  ∃ x : ℝ, ∀ n : ℕ, n < 1398 → 
    (fract (x^n) < fract (x^(n-1))) ↔ k ∣ n :=
sorry

#check existence_of_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_l625_62538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l625_62522

/-- Conic section C in polar coordinates -/
def C (ρ θ : ℝ) : Prop := ρ^2 = 12 / (3 + Real.sin θ^2)

/-- Point A -/
noncomputable def A : ℝ × ℝ := (0, -Real.sqrt 3)

/-- Left focus F₁ -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Right focus F₂ -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Line passing through F₁ and parallel to AF₂ -/
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

/-- Theorem stating the product of distances from F₁ to intersection points -/
theorem intersection_distance_product :
  ∃ M N : ℝ × ℝ,
    C (Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2)) (Real.arctan ((M.2 - F₁.2) / (M.1 - F₁.1))) ∧
    C (Real.sqrt ((N.1 - F₁.1)^2 + (N.2 - F₁.2)^2)) (Real.arctan ((N.2 - F₁.2) / (N.1 - F₁.1))) ∧
    line M.1 M.2 ∧
    line N.1 N.2 ∧
    (M.1 - F₁.1)^2 + (M.2 - F₁.2)^2 * ((N.1 - F₁.1)^2 + (N.2 - F₁.2)^2) = (12/5)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l625_62522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l625_62513

theorem average_after_removal (nums : Finset ℕ) (sum : ℕ) : 
  nums.card = 50 →
  sum = nums.sum id →
  sum / nums.card = 50 →
  45 ∈ nums →
  55 ∈ nums →
  let remaining := nums.erase 45 |>.erase 55
  (sum - 45 - 55) / remaining.card = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l625_62513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_vehicle_wheels_l625_62535

/-- Represents the number of wheels on a vehicle -/
def Wheels := Nat

/-- Represents the count of a specific type of vehicle -/
def VehicleCount := Nat

/-- The total number of wheels for all vehicles -/
def TotalWheels : Nat := 66

/-- The number of vehicles of the unknown type -/
def UnknownVehicleCount : Nat := 16

/-- The number of wheels on a two-wheeler -/
def TwoWheelerWheels : Nat := 2

theorem unknown_vehicle_wheels :
  ∃! (w : Nat),
    w > 0 ∧
    ∃ (twoWheelerCount : Nat),
      TotalWheels = twoWheelerCount * TwoWheelerWheels + UnknownVehicleCount * w ∧
      twoWheelerCount ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unknown_vehicle_wheels_l625_62535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_function_l625_62571

theorem min_value_exponential_function :
  ∀ x : ℝ, (4 : ℝ)^x - 2^(x+1) + 2 ≥ 0 ∧
  ∃ x₀ : ℝ, x₀ = 1 ∧ (4 : ℝ)^x₀ - 2^(x₀+1) + 2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_function_l625_62571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_is_zero_l625_62527

-- Define the set of integers we're looking for
def A : Set ℤ := sorry

-- Define the injective function from reals to reals
def f : ℝ → ℝ := sorry

-- Main theorem
theorem unique_set_is_zero 
  (h1 : ∀ n : ℕ+, {x : ℝ | ∃ y, (f^[n] y - y) = x} = {x : ℝ | ∃ a ∈ A, x = a + n})
  (h2 : Function.Injective f) :
  A = {0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_is_zero_l625_62527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l625_62518

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if sin²A - sin²B = 2 sin B sin C and c = 3b, then A = π/3 --/
theorem triangle_angle_value (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Valid angle ranges
  A + B + C = π →  -- Angle sum in a triangle
  Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c →  -- Law of sines
  (Real.sin A)^2 - (Real.sin B)^2 = 2 * Real.sin B * Real.sin C →  -- Given condition
  c = 3 * b →  -- Given condition
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l625_62518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_l625_62584

-- Define the curves
noncomputable def f (x : ℝ) := Real.cos x
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.sin x

-- Define the domain
def domain : Set ℝ := Set.Ioo 0 (Real.pi / 2)

-- Define the intersection point A
noncomputable def A : ℝ × ℝ :=
  let x := Real.pi / 6
  (x, f x)

-- Define the slopes of tangent lines at A
noncomputable def m_f : ℝ := -Real.sin (A.1)
noncomputable def m_g : ℝ := Real.sqrt 3 * Real.cos (A.1)

-- Define the x-intercepts B and C
noncomputable def B : ℝ := A.1 - A.2 / m_f
noncomputable def C : ℝ := A.1 - A.2 / m_g

-- State the theorem
theorem length_BC : |B - C| = 4 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_l625_62584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l625_62541

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t - 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation (t : ℝ) :
  (x t)^2 + 4/5 * (x t) * (y t) + 13/25 * (y t)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l625_62541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_nine_l625_62582

/-- A number with 1998 digits -/
def N : ℕ := sorry

/-- The sum of digits of N -/
def x : ℕ := sorry

/-- The sum of digits of x -/
def y : ℕ := sorry

/-- The sum of digits of y -/
def z : ℕ := sorry

/-- Helper function to calculate sum of digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- N has 1998 digits -/
axiom N_digits : N ≥ 10^1997 ∧ N < 10^1998

/-- N is divisible by 9 -/
axiom N_div_9 : N % 9 = 0

/-- x is the sum of N's digits -/
axiom x_def : x = sum_of_digits N

/-- y is the sum of x's digits -/
axiom y_def : y = sum_of_digits x

/-- z is the sum of y's digits -/
axiom z_def : z = sum_of_digits y

theorem z_equals_nine : z = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_nine_l625_62582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_isosceles_trapezoid_is_frustum_l625_62569

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The isosceles trapezoid has an axis of symmetry -/
  has_axis_of_symmetry : Prop

/-- A frustum -/
structure Frustum where

/-- Rotation of a 2D shape around an axis to form a 3D shape -/
def rotate (shape : Type) (axis : Prop) : Type := sorry

/-- The result of rotating an isosceles trapezoid around its axis of symmetry is a frustum -/
theorem rotate_isosceles_trapezoid_is_frustum (T : IsoscelesTrapezoid) :
  rotate IsoscelesTrapezoid T.has_axis_of_symmetry = Frustum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_isosceles_trapezoid_is_frustum_l625_62569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_terms_count_l625_62502

/-- The number of nonzero terms in the expansion of (x-3)(3x^2-2x+6)+2(x^3+x^2-4x) -/
theorem nonzero_terms_count : ∃ (p : Polynomial ℚ), 
  p = (X - 3) * (3 * X^2 - 2 * X + 6) + 2 * (X^3 + X^2 - 4 * X) ∧ 
  p.support.card = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonzero_terms_count_l625_62502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_quadrilateral_l625_62596

open Complex Real

-- Define the equation
def equation (z : ℂ) : Prop := (z - 4) ^ 12 = 64

-- Define the set of solutions
def solutions : Finset ℂ := sorry

-- Define the property that solutions form a regular dodecagon
def forms_regular_dodecagon (S : Finset ℂ) : Prop := sorry

-- Define a function to get four consecutive vertices
def four_consecutive_vertices (S : Finset ℂ) : Finset ℂ := sorry

-- Define the area of a quadrilateral given by four complex points
noncomputable def quadrilateral_area (a b c d : ℂ) : ℝ := sorry

-- Theorem statement
theorem min_area_of_quadrilateral :
  forms_regular_dodecagon solutions →
  ∃ (vertices : Finset ℂ), 
    vertices ⊆ solutions ∧ 
    vertices.card = 4 ∧
    (∀ (a b c d : ℂ), {a, b, c, d} = vertices → quadrilateral_area a b c d = sqrt 6 / 2) ∧
    (∀ (other_vertices : Finset ℂ), 
      other_vertices ⊆ solutions → 
      other_vertices.card = 4 → 
      ∀ (a b c d : ℂ), {a, b, c, d} = other_vertices → 
        quadrilateral_area a b c d ≥ sqrt 6 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_quadrilateral_l625_62596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_with_no_common_points_are_parallel_or_skew_l625_62519

-- Define a custom type for lines in 3D space
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

-- Define what it means for two lines to have no common points
def noCommonPoints (l1 l2 : Line3D) : Prop :=
  ∀ (t s : ℝ), (λ i => l1.point i + t * l1.direction i) ≠ (λ i => l2.point i + s * l2.direction i)

-- Define parallel lines
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ (k : ℝ), ∀ i, l1.direction i = k * l2.direction i

-- Define skew lines
def skew (l1 l2 : Line3D) : Prop :=
  ¬ ∃ (plane : (Fin 3 → ℝ) → Prop), 
    (∀ (t : ℝ), plane (λ i => l1.point i + t * l1.direction i)) ∧
    (∀ (s : ℝ), plane (λ i => l2.point i + s * l2.direction i))

-- The theorem to prove
theorem lines_with_no_common_points_are_parallel_or_skew (l1 l2 : Line3D) :
  noCommonPoints l1 l2 → parallel l1 l2 ∨ skew l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_with_no_common_points_are_parallel_or_skew_l625_62519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_coffee_price_l625_62506

/-- Represents the price of coffee in yuan -/
structure CoffeePrice where
  price : ℝ

/-- The original price of coffee -/
noncomputable def original_price : CoffeePrice :=
  ⟨36⟩

/-- The cost of the first cup of coffee -/
noncomputable def first_cup_cost : CoffeePrice :=
  original_price

/-- The cost of the second cup of coffee -/
noncomputable def second_cup_cost : CoffeePrice :=
  ⟨original_price.price / 2⟩

/-- The cost of the third cup of coffee -/
def third_cup_cost : CoffeePrice :=
  ⟨3⟩

/-- The average cost of three cups of coffee -/
def average_cost : CoffeePrice :=
  ⟨19⟩

/-- Theorem stating that the original price of coffee is 36 yuan -/
theorem original_coffee_price :
  (first_cup_cost.price + second_cup_cost.price + third_cup_cost.price) / 3 = average_cost.price →
  original_price.price = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_coffee_price_l625_62506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_over_four_l625_62512

theorem cos_alpha_minus_pi_over_four (α : Real) 
  (h1 : Real.tan α = -3/4) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sqrt 2 * Real.cos (α - π/4) = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_over_four_l625_62512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_squared_correlation_relationship_l625_62585

-- Define the K^2 statistic for a 2x2 contingency table
noncomputable def K_squared (n a b c d : ℝ) : ℝ :=
  (n * (a * d - b * c)^2) / ((a + c) * (b + d) * (a + b) * (c + d))

-- Define a measure of correlation (this is a placeholder and should be defined properly)
noncomputable def correlation_measure (a b c d : ℝ) : ℝ := sorry

-- State the theorem
theorem K_squared_correlation_relationship 
  (n a b c d : ℝ) (h1 : n > 0) (h2 : a ≥ 0) (h3 : b ≥ 0) (h4 : c ≥ 0) (h5 : d ≥ 0) :
  ∀ (k1 k2 : ℝ), 
    K_squared n a b c d < k1 → 
    K_squared n a b c d < k2 → 
    k1 < k2 → 
    correlation_measure a b c d < correlation_measure a b c d := by
  sorry

#check K_squared_correlation_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_squared_correlation_relationship_l625_62585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_l625_62593

/-- Represents a line in a plane -/
structure Line where
  -- We'll use a simplified representation of a line
  slope : ℝ
  intercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- A line is perpendicular to another line if their slopes are negative reciprocals -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

/-- The theorem states that if two lines are parallel and a third line is perpendicular to one of them, 
    then it is also perpendicular to the other parallel line -/
theorem perpendicular_to_parallel (l1 l2 l3 : Line) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 := by
  intro h_parallel h_perp
  simp [parallel, perpendicular] at *
  rw [h_parallel] at h_perp
  exact h_perp

#check perpendicular_to_parallel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_l625_62593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l625_62547

/-- The function f(x) = (1/2)x^2 + x - 2ln(x) -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + x - 2 * Real.log x

/-- The minimum value of f(x) is 3/2 -/
theorem f_min_value : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 3/2 := by
  -- We claim that x = 1 is the minimizer
  use 1
  
  constructor
  · -- Prove 1 > 0
    linarith
  
  · intro y hy
    constructor
    · sorry -- Proof that f(y) ≥ f(1) for all y > 0
    · -- Prove f(1) = 3/2
      calc
        f 1 = (1/2) * 1^2 + 1 - 2 * Real.log 1 := rfl
        _ = (1/2) + 1 - 2 * 0 := by simp [Real.log_one]
        _ = 3/2 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l625_62547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_point_within_distance_minimal_distance_is_optimal_l625_62555

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  not_parallel : a ≠ 0 ∨ b ≠ 0

/-- Distance from a point (x, y) to a line ax + by + c = 0 --/
noncomputable def distance_to_line (x y : ℤ) (l : Line) : ℝ :=
  |l.a * (x : ℝ) + l.b * (y : ℝ) + l.c| / Real.sqrt (l.a^2 + l.b^2)

/-- The minimal distance that satisfies the condition --/
noncomputable def minimal_distance : ℝ := 1 / (2 * Real.sqrt 2)

/-- Theorem statement --/
theorem exists_integer_point_within_distance (l : Line) :
  ∃ (x y : ℤ), distance_to_line x y l ≤ minimal_distance := by
  sorry

/-- Main theorem: The minimal distance is 1 / (2√2) --/
theorem minimal_distance_is_optimal :
  ∀ d : ℝ, d < minimal_distance →
    ∃ l : Line, ∀ x y : ℤ, distance_to_line x y l > d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_point_within_distance_minimal_distance_is_optimal_l625_62555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l625_62558

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2 * f x) = f (x - f y) + f x + y) →
  (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l625_62558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equivalence_complement_intersection_l625_62573

-- Define the sets A and B
def A : Set ℝ := {x | (1/2 : ℝ) ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x < 8}
def B : Set ℝ := {x | 5/(x+2) ≥ 1}

-- Theorem for the first part of the question
theorem set_equivalence : 
  A = {x : ℝ | -1 ≤ x ∧ x < 3} ∧ 
  B = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the question
theorem complement_intersection : 
  (Aᶜ ∩ B) = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3} := by sorry

-- Note: Aᶜ denotes the complement of A in ℝ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equivalence_complement_intersection_l625_62573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_cosh_curve_l625_62566

/-- The length of the arc of the curve y = cosh(x) between points (x₁, y₁) and (x₂, y₂) -/
noncomputable def arcLength (x₁ x₂ : ℝ) : ℝ :=
  Real.sinh x₂ - Real.sinh x₁

/-- Theorem stating that arcLength is the correct formula for the length of the arc of y = cosh(x) between (x₁, y₁) and (x₂, y₂) -/
theorem arc_length_cosh_curve (x₁ x₂ : ℝ) :
  arcLength x₁ x₂ = ∫ x in x₁..x₂, Real.sqrt (1 + (Real.sinh x)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_cosh_curve_l625_62566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l625_62551

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + a*x - 4*y + 1 = 0

-- Define the line passing through P(0,1) with slope 1
def line_P (x y : ℝ) : Prop :=
  y = x + 1

-- Define point P
def point_P : ℝ × ℝ := (0, 1)

-- Define the theorem
theorem circle_problem (a : ℝ) :
  -- Part 1: Value of a
  (∃ A B : ℝ × ℝ, circle_C a A.1 A.2 ∧ circle_C a B.1 B.2 ∧
    line_P A.1 A.2 ∧ line_P B.1 B.2 ∧ point_P = ((A.1 + B.1)/2, (A.2 + B.2)/2)) →
  a = 2 ∧
  -- Part 2: Maximum area of triangle ABE
  (∀ E : ℝ × ℝ, circle_C a E.1 E.2 → E ≠ A → E ≠ B →
    ∃ S : ℝ, S ≤ 2 + 2*Real.sqrt 2 ∧ 
    S = abs ((A.1 - E.1)*(B.2 - E.2) - (B.1 - E.1)*(A.2 - E.2)) / 2) ∧
  -- Part 3: Minimum distance and coordinates of M
  (∀ M : ℝ × ℝ, ¬circle_C a M.1 M.2 →
    ∃ N : ℝ × ℝ, circle_C a N.1 N.2 ∧
    (M.1 - N.1)*(N.1 + 1) + (M.2 - N.2)*(N.2 - 2) = 0 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 →
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 1/2 ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 1/2 ↔ M = (1/2, 1/2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l625_62551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_sign_name_is_obelus_l625_62546

/-- The name of the division sign with specific historical and etymological properties -/
def division_sign_name : String := "Obelus"

/-- The historical use of the division sign in late medieval period -/
def late_medieval_use (symbol : String) : Prop :=
  ∃ (description : String), description = "was used to mark words or passages as spurious, corrupt, or doubtful"

/-- The etymological origin of the division sign name -/
def greek_origin (name : String) : Prop :=
  ∃ (origin : String), origin = "comes from an ancient Greek word meaning a sharpened stick or pointed pillar"

/-- Theorem stating the correct name of the division sign based on its properties -/
theorem division_sign_name_is_obelus :
  late_medieval_use division_sign_name ∧ 
  greek_origin division_sign_name ∧
  division_sign_name = "Obelus" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_sign_name_is_obelus_l625_62546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_obtainable_l625_62572

/-- A card is a pair of integers (a, b) where a < b and b - a is divisible by 7 -/
def Card : Type := { p : ℤ × ℤ // p.1 < p.2 ∧ (p.2 - p.1) % 7 = 0 }

/-- The original card (5, 19) -/
def original_card : Card :=
  ⟨(5, 19), by
    constructor
    · exact lt_trans (by norm_num : (5 : ℤ) < 10) (by norm_num : (10 : ℤ) < 19)
    · norm_num⟩

/-- A card (a, b) can be obtained from the original card if a = 5 and b = 5 + 7k for some integer k -/
theorem card_obtainable (c : Card) : 
  (∃ k : ℤ, c.1.1 = 5 ∧ c.1.2 = 5 + 7 * k) ↔ 
  (c.1.1 = original_card.1.1 ∧ (c.1.2 - c.1.1) % 7 = 0 ∧ c.1.1 < c.1.2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_obtainable_l625_62572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kings_tour_properties_l625_62565

/-- Represents a king's tour on an 8x8 chessboard -/
structure KingsTour where
  path : List (Fin 8 × Fin 8)
  start_end_same : path.head? = path.get? (path.length - 1)
  all_squares_visited : ∀ (i j : Fin 8), (i, j) ∈ path
  no_repeats : path.Nodup
  closed : path.length = 65  -- 64 squares + return to start

/-- Counts the number of horizontal and vertical moves in a king's tour -/
def count_hv_moves (tour : KingsTour) : ℕ :=
  sorry

/-- Calculates the length of a king's tour path -/
noncomputable def path_length (tour : KingsTour) : ℝ :=
  sorry

/-- Proves the properties of a king's tour -/
theorem kings_tour_properties (tour : KingsTour) :
  count_hv_moves tour ≥ 28 ∧
  path_length tour ≥ 64 ∧
  path_length tour ≤ 28 + 36 * Real.sqrt 2 := by
  sorry

#check kings_tour_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kings_tour_properties_l625_62565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_as_consecutive_product_l625_62586

theorem largest_factorial_as_consecutive_product : 
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ, n * Nat.factorial (n - 1) = (n - 5 + a) * Nat.factorial (n - 5 + a - 1) / (a * Nat.factorial (a - 1))) ∧
  (∃ a : ℕ, 7 * Nat.factorial 6 = (7 - 5 + a) * Nat.factorial (7 - 5 + a - 1) / (a * Nat.factorial (a - 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_factorial_as_consecutive_product_l625_62586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_value_is_25000_l625_62540

/-- Represents the sales problem with given conditions --/
structure SalesProblem where
  hourly_rate : ℚ
  commission_rate : ℚ
  hours_worked : ℚ
  budget_percentage : ℚ
  insurance_amount : ℚ

/-- Calculates the total value of items sold given the sales problem parameters --/
def calculate_sales_value (p : SalesProblem) : ℚ :=
  let total_earnings := p.insurance_amount / (1 - p.budget_percentage)
  let base_salary := p.hourly_rate * p.hours_worked
  (total_earnings - base_salary) / p.commission_rate

/-- Theorem stating that the total value of items sold is $25,000 --/
theorem sales_value_is_25000 (p : SalesProblem) 
  (h1 : p.hourly_rate = 15/2)
  (h2 : p.commission_rate = 4/25)
  (h3 : p.hours_worked = 160)
  (h4 : p.budget_percentage = 19/20)
  (h5 : p.insurance_amount = 260) :
  calculate_sales_value p = 25000 := by
  sorry

def main : IO Unit := do
  let result := calculate_sales_value { 
    hourly_rate := 15/2,
    commission_rate := 4/25,
    hours_worked := 160,
    budget_percentage := 19/20,
    insurance_amount := 260
  }
  IO.println s!"The total value of items sold: ${result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_value_is_25000_l625_62540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_l625_62521

/-- Represents a pair of integers (a, b) where a < b --/
structure IntPair where
  a : Int
  b : Int
  h : a < b

/-- The set of integers from 1 to 3000 --/
def IntSet : Set Int := {i | 1 ≤ i ∧ i ≤ 3000}

/-- A function that takes a natural number k and returns k pairs of integers --/
def choosePairs (k : Nat) : Fin k → IntPair :=
  sorry

/-- All chosen pairs are from the IntSet --/
axiom pairs_in_set : ∀ (k : Nat) (i : Fin k), (choosePairs k i).a ∈ IntSet ∧ (choosePairs k i).b ∈ IntSet

/-- No two pairs share a common element --/
axiom pairs_distinct : ∀ (k : Nat) (i j : Fin k), i ≠ j →
  (choosePairs k i).a ≠ (choosePairs k j).a ∧
  (choosePairs k i).a ≠ (choosePairs k j).b ∧
  (choosePairs k i).b ≠ (choosePairs k j).a ∧
  (choosePairs k i).b ≠ (choosePairs k j).b

/-- All sums a_i + b_i are distinct --/
axiom sums_distinct : ∀ (k : Nat) (i j : Fin k), i ≠ j →
  (choosePairs k i).a + (choosePairs k i).b ≠ (choosePairs k j).a + (choosePairs k j).b

/-- All sums a_i + b_i are ≤ 3000 --/
axiom sums_bounded : ∀ (k : Nat) (i : Fin k), (choosePairs k i).a + (choosePairs k i).b ≤ 3000

theorem max_pairs : (∃ k : Nat, ∀ m : Nat, m > k → ¬∃ (f : Fin m → IntPair),
  (∀ i, (f i).a ∈ IntSet ∧ (f i).b ∈ IntSet) ∧
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i j, i ≠ j → (f i).a + (f i).b ≠ (f j).a + (f j).b) ∧
  (∀ i, (f i).a + (f i).b ≤ 3000)) ∧
  (∃ (f : Fin 1199 → IntPair),
  (∀ i, (f i).a ∈ IntSet ∧ (f i).b ∈ IntSet) ∧
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i j, i ≠ j → (f i).a + (f i).b ≠ (f j).a + (f j).b) ∧
  (∀ i, (f i).a + (f i).b ≤ 3000)) := by
  sorry

#check max_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_l625_62521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_integer_point_on_circle_l625_62578

theorem at_most_one_integer_point_on_circle (r : ℝ) :
  ∃! p : ℤ × ℤ, ((p.1 : ℝ) - Real.sqrt 2) ^ 2 + ((p.2 : ℝ) - Real.sqrt 3) ^ 2 = r ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_integer_point_on_circle_l625_62578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_point_coordinates_l625_62561

/-- A parabola with equation y^2 = -2px where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- A point on the parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = -2 * par.p * x

/-- The focus of the parabola -/
noncomputable def focus (par : Parabola) : ℝ × ℝ := (-par.p / 2, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_equation_and_point_coordinates
  (par : Parabola)
  (M : PointOnParabola par)
  (h_x : M.x = -9)
  (h_dist : distance (M.x, M.y) (focus par) = 10) :
  (par.p = 2 ∧ M.y = 6 ∨ M.y = -6) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_point_coordinates_l625_62561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_l625_62583

/-- Calculates the annual income from a stock investment -/
noncomputable def annual_income (investment : ℝ) (stock_price : ℝ) (dividend_rate : ℝ) (par_value : ℝ) : ℝ :=
  let num_shares := investment / stock_price
  let dividend_per_share := par_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem stating that the annual income from the given investment is $3000 -/
theorem investment_income : 
  annual_income 6800 136 0.6 100 = 3000 := by
  -- Unfold the definition of annual_income
  unfold annual_income
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_l625_62583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_max_area_l625_62503

open Real

-- Define the vectors m and n
noncomputable def m (A : ℝ) : ℝ × ℝ := (sin A, 1/2)
noncomputable def n (A : ℝ) : ℝ × ℝ := (3, sin A + sqrt 3 * cos A)

-- Define collinearity of two 2D vectors
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem 1: If m and n are collinear, then A = π/3
theorem angle_value (A : ℝ) (h : collinear (m A) (n A)) : A = π/3 := by
  sorry

-- Define the area of a triangle given two sides and the included angle
noncomputable def triangle_area (b c A : ℝ) : ℝ :=
  1/2 * b * c * sin A

-- Theorem 2: Given BC = 2 and A = π/3, the maximum area of triangle ABC is √3
--            and this occurs when the triangle is equilateral
theorem max_area (b c : ℝ) (h1 : b * c = 4) (h2 : b > 0) (h3 : c > 0) :
  let S := triangle_area b c (π/3)
  (∀ b' c', b' * c' = 4 → triangle_area b' c' (π/3) ≤ S) ∧
  S = sqrt 3 ∧
  (S = sqrt 3 → b = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_max_area_l625_62503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l625_62580

/-- Given a reflection that maps (2, 6) to (4, -4), prove that it maps (1, 4) to (3.2, -2.6) -/
theorem reflection_problem (reflection : ℝ × ℝ → ℝ × ℝ) 
  (h1 : reflection (2, 6) = (4, -4)) :
  reflection (1, 4) = (3.2, -2.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l625_62580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_has_twenty_mmm_l625_62554

/-- Calculates the number of M&M's Ben has given Bryan's Skittles count and the candy difference -/
def ben_mmm_count (bryan_skittles : ℕ) (candy_difference : ℕ) : ℕ :=
  bryan_skittles - candy_difference

/-- The main theorem that proves Ben has 20 M&M's -/
theorem ben_has_twenty_mmm :
  ben_mmm_count 50 30 = 20 := by
  -- Unfold the definition of ben_mmm_count
  unfold ben_mmm_count
  -- Simplify the arithmetic
  simp

-- Evaluate the function to check the result
#eval ben_mmm_count 50 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_has_twenty_mmm_l625_62554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l625_62552

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 3)

theorem range_of_f :
  Set.range f = {y : ℝ | 0 < y ∧ y ≤ 1/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l625_62552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l625_62501

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ a d : ℝ, 
    Real.sqrt (49 + k) = a ∧ 
    Real.sqrt (225 + k) = a + d ∧ 
    Real.sqrt (484 + k) = a + 2*d) ↔ 
  k = 324 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l625_62501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l625_62588

noncomputable def f (x : ℝ) : ℝ := (1 - Real.log x) / (1 + Real.log x)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 1 ∧ f x = y) ↔ -1 < y ∧ y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l625_62588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l625_62515

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) : 
  pen_cost > 10 ∧ 
  masha_spent = 357 ∧ 
  olya_spent = 441 ∧ 
  masha_spent % pen_cost = 0 ∧ 
  olya_spent % pen_cost = 0 →
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
  intro h
  sorry

#check total_pens_bought

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l625_62515
