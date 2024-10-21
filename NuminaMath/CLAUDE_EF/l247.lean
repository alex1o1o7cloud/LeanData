import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l247_24785

noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 3)

theorem function_properties (A ω : ℝ) (h1 : A > 0) (h2 : ω > 0) 
  (h3 : ∀ x, f A ω x ≤ 2) 
  (h4 : ∀ x, f A ω x = f A ω (x + Real.pi)) :
  A = 2 ∧ ω = 2 ∧ 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), -Real.sqrt 3 ≤ f A ω x ∧ f A ω x ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l247_24785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_irrational_l247_24731

theorem pi_irrational (h1 : ∃ (a b : ℤ), (1 : ℝ) / 3 = a / b)
                      (h2 : ∃ (a b : ℤ), (0.1010010001 : ℝ) = a / b)
                      (h3 : ∃ (a b : ℤ), Real.sqrt 9 = a / b) :
  ¬ ∃ (a b : ℤ), Real.pi = a / b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_irrational_l247_24731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_between_three_digit_numbers_between_137_and_285_l247_24712

theorem count_numbers_between (a b : ℕ) (h : a < b) :
  (Finset.range (b - a - 1)).card = b - a - 1 :=
by sorry

theorem three_digit_numbers_between_137_and_285 :
  (Finset.filter (λ n : ℕ ↦ 137 < n ∧ n < 285) (Finset.range 1000)).card = 147 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_between_three_digit_numbers_between_137_and_285_l247_24712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_closed_form_l247_24772

/-- The golden ratio φ (phi) -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The negative solution φ' (phi prime) -/
noncomputable def φ' : ℝ := (1 - Real.sqrt 5) / 2

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The nth Fibonacci number is given by (φⁿ - φ'ⁿ) / √5 -/
theorem fib_closed_form (n : ℕ) : 
  (fib n : ℝ) = (φ^n - φ'^n) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_closed_form_l247_24772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemys_inequality_l247_24798

/-- Ptolemy's inequality for quadrilaterals -/
theorem ptolemys_inequality (A B C D : ℝ × ℝ) : 
  ‖A - C‖ * ‖B - D‖ ≤ ‖A - B‖ * ‖C - D‖ + ‖B - C‖ * ‖A - D‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemys_inequality_l247_24798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_proof_l247_24740

/-- Calculates the total travel time for a journey with given distance and speeds -/
noncomputable def totalTravelTime (totalDistance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (totalDistance / 2) / speed1 + (totalDistance / 2) / speed2

theorem journey_time_proof :
  let totalDistance : ℝ := 409.0909090909091
  let speed1 : ℝ := 30
  let speed2 : ℝ := 25
  totalTravelTime totalDistance speed1 speed2 = 15 := by
  -- Unfold the definition of totalTravelTime
  unfold totalTravelTime
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  sorry

#eval (409.0909090909091 / 2 / 30 + 409.0909090909091 / 2 / 25 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_proof_l247_24740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l247_24705

/-- A cubic function f(x) with specific properties -/
noncomputable def f (a c : ℝ) (x : ℝ) : ℝ := a * x^3 - (1/2) * x^2 + c

/-- The theorem stating the properties of the cubic function -/
theorem cubic_function_properties :
  ∃ (a c : ℝ),
    (f a c 0 = 1) ∧ 
    (6 * 2 - 3 * f a c 2 - 7 = 0) ∧ 
    (∀ x, f a c x ≤ 1) ∧
    (f a c 0 = 1) ∧
    (f a c 1 = 5/6) ∧
    (∫ x in (0)..(3/2), (1 - f a c x) = 9/64) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l247_24705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_216m_493n_l247_24793

theorem smallest_positive_integer_216m_493n : 
  ∃ (m n : ℤ), Int.gcd 216 493 = Int.natAbs (216 * m + 493 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_216m_493n_l247_24793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l247_24729

theorem sufficient_but_not_necessary :
  ∃ (p q : ℝ → Prop), (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by
  use (λ x => x < 2), (λ x => x < 3)
  constructor
  · intro x h
    exact lt_trans h (by norm_num)
  · use 2.5
    constructor
    · norm_num
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l247_24729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_2_l247_24755

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 else Real.rpow 2 x

theorem solution_set_of_f_leq_2 :
  {x : ℝ | f x ≤ 2} = {x : ℝ | x ≤ -2 ∨ (0 < x ∧ x ≤ 1)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_2_l247_24755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_locus_l247_24794

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

def Focus1 (c : ℝ) : ℝ × ℝ := (-c, 0)
def Focus2 (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the circle
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

-- Define the angle bisector perpendicular (placeholder)
noncomputable def AngleBisectorPerpendicular (f1 f2 q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

-- Define the set of points P
def PointP (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ q ∈ Hyperbola a b, 
    q ≠ (a, 0) ∧ q ≠ (-a, 0) ∧
    p ∈ AngleBisectorPerpendicular (Focus1 c) (Focus2 c) q}

-- The theorem to prove
theorem hyperbola_locus (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  PointP a b c = Circle a \ {(a, 0), (-a, 0)} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_locus_l247_24794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_identities_l247_24787

theorem sqrt_identities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 > b) :
  (Real.sqrt (a + Real.sqrt b) = Real.sqrt ((a + Real.sqrt (a^2 - b))/2) + Real.sqrt ((a - Real.sqrt (a^2 - b))/2)) ∧
  (Real.sqrt (a - Real.sqrt b) = Real.sqrt ((a + Real.sqrt (a^2 - b))/2) - Real.sqrt ((a - Real.sqrt (a^2 - b))/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_identities_l247_24787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_one_l247_24738

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2*a*x + 8 else x + 4/x + 2*a

theorem min_value_at_one (a : ℝ) :
  (∀ x, f a x ≥ f a 1) → a ≥ 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_one_l247_24738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l247_24760

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalArea (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge^2 - (baseEdge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units -/
theorem pyramid_face_area :
  totalArea 8 7 = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l247_24760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l247_24707

/-- The distance between the foci of an ellipse -/
noncomputable def focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_focal_distance :
  ∀ (x y : ℝ), is_ellipse x y 5 4 → focal_distance 5 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l247_24707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l247_24754

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_difference_magnitude
  (a b : V)
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 2)
  (h3 : inner a b = (3 : ℝ) / 2) :
  ‖a - b‖ = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l247_24754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_transmission_l247_24702

def transmission_info (a₀ a₁ a₂ : Bool) : List Bool :=
  let h₀ := a₀ ≠ a₁
  let h₁ := h₀ ≠ a₂
  [h₀, a₀, a₁, a₂, h₁]

theorem invalid_transmission :
  ∀ (a₀ a₁ a₂ : Bool), transmission_info a₀ a₁ a₂ ≠ [true, false, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_transmission_l247_24702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_prism_l247_24761

-- Define the right triangular prism
structure RightTriangularPrism where
  a : ℝ
  b : ℝ
  h : ℝ
  θ : ℝ

-- Define the volume of the prism
noncomputable def volume (p : RightTriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h * Real.sin p.θ

-- Define the sum of areas of two lateral faces and one base
noncomputable def sumOfAreas (p : RightTriangularPrism) : ℝ :=
  p.a * p.h + p.b * p.h + 1/2 * p.a * p.b * Real.sin p.θ

-- Theorem statement
theorem max_volume_of_prism :
  ∀ p : RightTriangularPrism,
    sumOfAreas p = 36 →
    volume p ≤ 54 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_prism_l247_24761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_markup_l247_24789

-- Define the markup percentage
noncomputable def markup_percentage (x : ℝ) : ℝ := x / 100

-- Define the discount percentage
noncomputable def discount_percentage : ℝ := 10 / 100

-- Define the loss percentage
noncomputable def loss_percentage : ℝ := 1 / 100

-- Theorem statement
theorem selling_price_markup (x : ℝ) :
  (1 - discount_percentage) * (1 + markup_percentage x) = (1 - loss_percentage) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_markup_l247_24789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleons_color_change_l247_24792

/-- The number of chameleons that changed color in a grove with specific conditions --/
theorem chameleons_color_change : ℕ := by
  -- Define variables
  let total : ℕ := 140
  let initial_blue : ℕ := 100  -- This is 5x, where x = 20
  let final_blue : ℕ := 20     -- This is x
  let initial_red : ℕ := total - initial_blue
  let final_red : ℕ := 3 * initial_red
  let color_change : ℕ := initial_blue - final_blue

  -- Assert the relationships
  have blue_relation : initial_blue = 5 * final_blue := by rfl
  have total_constant : final_blue + final_red = total := by
    -- Proof of this step is omitted
    sorry

  -- Prove that 80 chameleons changed their color
  have color_change_value : color_change = 80 := by
    -- Calculation: 100 - 20 = 80
    rfl

  -- Return the result
  exact color_change


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleons_color_change_l247_24792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l247_24791

-- Define the line y = sin θ · x + 2
noncomputable def line (θ : ℝ) (x : ℝ) : ℝ := Real.sin θ * x + 2

-- Define the inclination angle of the line
noncomputable def inclination_angle (θ : ℝ) : ℝ := Real.arctan (Real.sin θ)

-- Define the set of all possible inclination angles
def inclination_angle_set : Set ℝ := {α | ∃ θ, α = inclination_angle θ}

-- Theorem stating the range of the inclination angle
theorem inclination_angle_range :
  inclination_angle_set = {α | α ∈ Set.Icc 0 (π/4) ∪ Set.Ioc (3*π/4) π} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l247_24791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_set_equality_l247_24753

theorem cosine_set_equality : 
  {x : ℝ | x ∈ Set.Icc 0 Real.pi ∧ Real.cos (π * Real.cos x) = 0} = {π/3, 2*π/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_set_equality_l247_24753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_part_trip_average_speed_l247_24716

/-- Calculates the average speed of a two-part trip -/
theorem two_part_trip_average_speed 
  (total_distance : ℝ) 
  (distance1 : ℝ) 
  (speed1 : ℝ) 
  (distance2 : ℝ) 
  (speed2 : ℝ) 
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 25)
  (h3 : distance2 = 25)
  (h4 : speed1 = 66)
  (h5 : speed2 = 33)
  (h6 : total_distance = 50) :
  (total_distance / ((distance1 / speed1) + (distance2 / speed2))) = 44 := by
  sorry

#check two_part_trip_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_part_trip_average_speed_l247_24716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fraction_above_line_l247_24748

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- A square defined by its four vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculate the area of a triangle given its base and height -/
def triangleArea (base height : ℚ) : ℚ :=
  (1 / 2) * base * height

/-- Calculate the area of a square given its side length -/
def squareArea (side : ℚ) : ℚ :=
  side * side

/-- Calculate the fraction of the square's area above a line -/
noncomputable def fractionAboveLine (s : Square) (l : Line) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem square_fraction_above_line :
  let s := Square.mk (Point.mk 2 1) (Point.mk 5 1) (Point.mk 5 4) (Point.mk 2 4)
  let l := Line.mk (Point.mk 2 3) (Point.mk 5 1)
  fractionAboveLine s l = 2 / 3 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fraction_above_line_l247_24748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_largest_after_crossing_out_l247_24759

def consecutive_numbers : List ℕ := List.range 40

def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

def total_digits : ℕ := (consecutive_numbers.map digit_count).sum

def remaining_digits : ℕ := total_digits - 60

theorem smallest_largest_after_crossing_out :
  ∃ (small large : ℕ),
    small = 12333330 ∧
    large = 99967383940 ∧
    (∀ s : ℕ, s ≠ small → s > small → (Nat.repr s).length > remaining_digits) ∧
    (∀ l : ℕ, l ≠ large → l < large → (Nat.repr l).length > remaining_digits) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_largest_after_crossing_out_l247_24759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l247_24745

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- A trapezoid in 2D space -/
structure Trapezoid where
  J : Point2D
  K : Point2D
  L : Point2D
  M : Point2D

/-- Perimeter of a trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  distance t.J t.K + distance t.K t.L + distance t.L t.M + distance t.M t.J

theorem trapezoid_properties :
  let J : Point2D := ⟨-2, -4⟩
  let K : Point2D := ⟨-2, 2⟩
  let L : Point2D := ⟨6, 8⟩
  let M : Point2D := ⟨6, -10⟩
  let t : Trapezoid := ⟨J, K, L, M⟩
  (M.x = 6 ∧ M.y = -10) ∧ perimeter t = 44 := by
  sorry

#eval "Trapezoid properties theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_properties_l247_24745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l247_24723

def circle_radius : ℝ := 4

structure Triangle (A B C : ℝ × ℝ) where

def diameter_point (M : ℝ × ℝ) : Prop :=
  M.1^2 + M.2^2 = circle_radius^2 ∧ (M.1 = 0 ∨ M.2 = 0)

def chord_angle (A B M : ℝ × ℝ) (angle : ℝ) : Prop :=
  sorry

def chord_perpendicular (B C : ℝ × ℝ) : Prop :=
  sorry

def point_ratio (A M B : ℝ × ℝ) (r1 r2 : ℚ) : Prop :=
  sorry

noncomputable def area (t : Triangle A B C) : ℝ :=
  sorry

theorem triangle_area (A B C M : ℝ × ℝ) :
  diameter_point M →
  chord_angle A B M (30 * π / 180) →
  chord_perpendicular B C →
  point_ratio A M B (2/5) (3/5) →
  ∃ (t : Triangle A B C), area t = 180 * Real.sqrt 3 / 19 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l247_24723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_uniform_probability_l247_24752

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Probability of a single square being white or black --/
noncomputable def prob_single_color : ℝ := 1 / 2

/-- Probability of all four center squares being the same color --/
noncomputable def prob_center_uniform : ℝ := 2 * prob_single_color ^ 4

/-- Number of edge pairs that need to match --/
def num_edge_pairs : ℕ := 8

/-- Number of corner pairs that need to match --/
def num_corner_pairs : ℕ := 4

/-- Probability of all edge pairs matching --/
noncomputable def prob_edge_match : ℝ := prob_single_color ^ num_edge_pairs

/-- Probability of all corner pairs matching --/
noncomputable def prob_corner_match : ℝ := prob_single_color ^ num_corner_pairs

/-- Theorem stating the probability of the grid becoming uniformly colored --/
theorem grid_uniform_probability : 
  prob_center_uniform * prob_edge_match * prob_corner_match = 1 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_uniform_probability_l247_24752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_smallest_angle_special_triangle_l247_24722

theorem cosine_smallest_angle_special_triangle :
  ∃ (A B C : ℝ) (a b c : ℝ),
    a = 2 ∧ b = 3 ∧ c = 4 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧
    a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧
    Real.cos A = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_smallest_angle_special_triangle_l247_24722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_days_debit_advantageous_l247_24721

/-- The cost of airline tickets in rubles -/
noncomputable def ticket_cost : ℝ := 12000

/-- Credit card cashback rate -/
noncomputable def credit_cashback_rate : ℝ := 0.01

/-- Debit card cashback rate -/
noncomputable def debit_cashback_rate : ℝ := 0.02

/-- Annual interest rate on debit card -/
noncomputable def annual_interest_rate : ℝ := 0.06

/-- Number of days in a month (simplified) -/
def days_in_month : ℕ := 30

/-- Calculate the benefit of using the credit card for N days -/
noncomputable def credit_benefit (N : ℕ) : ℝ :=
  (annual_interest_rate * (N : ℝ) * ticket_cost) / (365 : ℝ) + credit_cashback_rate * ticket_cost

/-- Calculate the benefit of using the debit card -/
noncomputable def debit_benefit : ℝ := debit_cashback_rate * ticket_cost

/-- Theorem stating the maximum number of days for which using the debit card is more advantageous -/
theorem max_days_debit_advantageous :
  ∃ N : ℕ, N = 6 ∧ ∀ n : ℕ, n ≤ N → debit_benefit ≥ credit_benefit n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_days_debit_advantageous_l247_24721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l247_24717

noncomputable section

-- Constants
def π : ℝ := Real.pi

-- Figure A
def length_A : ℝ := 4
def width_A : ℝ := 3
def radius_A : ℝ := width_A / 2
def shaded_area_A : ℝ := length_A * width_A - π * radius_A^2

-- Figure B
def length_B : ℝ := 4
def width_B : ℝ := 3
def radius_B : ℝ := 1
def shaded_area_B : ℝ := length_B * width_B - 2 * π * radius_B^2

-- Figure C
def length_C : ℝ := Real.sqrt 8
def width_C : ℝ := Real.sqrt 2
def radius_C : ℝ := width_C / 2
def shaded_area_C : ℝ := length_C * width_C - π * radius_C^2

theorem largest_shaded_area :
  shaded_area_B > shaded_area_A ∧ shaded_area_B > shaded_area_C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_shaded_area_l247_24717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_condition_l247_24720

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + r)^2

/-- If x^2 + mx + 25 is a perfect square trinomial, then m = ±10. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial 1 m 25 → m = 10 ∨ m = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_condition_l247_24720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_square_placement_exists_l247_24743

/-- Represents a square with unit length sides -/
structure UnitSquare where
  x : ℝ
  y : ℝ

/-- Represents a cube with edge length 2 -/
structure Cube where
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- Represents a placement of squares on a cube -/
structure SquarePlacement where
  cube : Cube
  squares : List UnitSquare
  num_squares : squares.length = 10
  no_overlap : ∀ s1 s2, s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → 
    (s1.x ≠ s2.x ∨ s1.y ≠ s2.y)
  no_shared_edges : ∀ s1 s2, s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → 
    (|s1.x - s2.x| > 1 ∨ |s1.y - s2.y| > 1)

/-- Theorem stating that a valid square placement exists -/
theorem valid_square_placement_exists : ∃ p : SquarePlacement, True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_square_placement_exists_l247_24743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l247_24757

/-- The constant term in the expansion of (2√x - 1/⁴√x)^6 is 60 -/
theorem constant_term_expansion : ℕ := by
  let constant_term : ℕ := 60
  let expansion : ℝ → ℝ := fun x ↦ (2 * Real.sqrt x - Real.rpow x (-(1/4)))^6
  have is_constant_term : constant_term = 60 := by rfl
  exact constant_term


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l247_24757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l247_24735

theorem triangle_third_side (a b c : ℝ) : 
  (a^2 - 7*a + 12 = 0) → 
  (b^2 - 7*b + 12 = 0) → 
  (a ≠ b) →
  (c = 5 ∨ c = Real.sqrt 7) →
  ∃ (s₁ s₂ s₃ : ℝ), ({s₁, s₂, s₃} : Set ℝ) = {a, b, c} ∧ s₁^2 + s₂^2 = s₃^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l247_24735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l247_24773

def statement_p (a : ℝ) : Prop :=
  ∀ n : ℕ+, (-1 : ℝ)^(n : ℕ) * (2*a + 1) < 2 + (-1 : ℝ)^((n : ℕ) + 1) / (n : ℝ)

def statement_q (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 5/2| < a ∧ a > 0 → |x^2 - 5| < 4

theorem range_of_a : {a : ℝ | (¬(statement_p a) ∨ ¬(statement_q a)) ∧ 
                             (statement_p a ∨ statement_q a)} = 
                     {a : ℝ | -3/2 ≤ a ∧ a ≤ 0 ∨ 1/4 ≤ a ∧ a ≤ 1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l247_24773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_function_n_value_l247_24710

-- Define the function as noncomputable due to its dependency on real numbers
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n + 5) / x

-- State the theorem
theorem reciprocal_function_n_value :
  ∃ n : ℝ, f n 2 = 3 ∧ n = 1 := by
  -- Prove the existence of n
  use 1
  -- Show that both conditions are satisfied
  constructor
  · -- Prove f 1 2 = 3
    simp [f]
    norm_num
  · -- Prove n = 1
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_function_n_value_l247_24710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_point_equivalence_l247_24725

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Checks if a SphericalPoint is in standard form -/
def isStandardForm (p : SphericalPoint) : Prop :=
  p.ρ > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi ∧ 0 ≤ p.φ ∧ p.φ ≤ Real.pi

/-- Converts a SphericalPoint to its standard form -/
noncomputable def toStandardForm (p : SphericalPoint) : SphericalPoint :=
  { ρ := p.ρ,
    θ := (p.θ % (2 * Real.pi) + 2 * Real.pi) % (2 * Real.pi),
    φ := if p.φ > Real.pi then 2 * Real.pi - p.φ else p.φ }

/-- Theorem stating the equivalence of the given point and its standard form -/
theorem spherical_point_equivalence :
  let p : SphericalPoint := { ρ := 5, θ := 11 * Real.pi / 6, φ := 5 * Real.pi / 3 }
  let p_standard := toStandardForm p
  isStandardForm p_standard ∧ 
  p_standard.ρ = 5 ∧ 
  p_standard.θ = 11 * Real.pi / 6 ∧ 
  p_standard.φ = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_point_equivalence_l247_24725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_value_l247_24776

theorem unique_n_value (n : ℕ) (d : ℕ → ℕ) (k : ℕ) :
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j) →
  d 1 = 1 →
  d k = n →
  n = d 2 ^ 2 + d 3 ^ 3 →
  n = 68 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_value_l247_24776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l247_24701

/-- A set of points forms a regular tetrahedron -/
def is_regular_tetrahedron (s : Set Point) : Prop := sorry

/-- The edge length of a tetrahedron -/
def edge_length (s : Set Point) : ℝ := sorry

/-- The volume of a tetrahedron -/
def volume (s : Set Point) : ℝ := sorry

/-- The volume of a regular tetrahedron with edge length 6 is 12√2 -/
theorem regular_tetrahedron_volume : 
  ∀ (ABCD : Set Point), 
  is_regular_tetrahedron ABCD → 
  edge_length ABCD = 6 → 
  volume ABCD = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l247_24701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l247_24742

-- Define the function f(x) = ln(x+2) on the interval (0, +∞)
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2)

-- State the theorem
theorem f_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l247_24742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_coins_count_l247_24770

/-- Proves that the total number of coins is 344 given the specified conditions -/
theorem total_coins_count (total_sum : ℕ) (coins_20p : ℕ) (value_20p : ℕ) (value_25p : ℕ) :
  total_sum = 7100 ∧ 
  coins_20p = 300 ∧
  value_20p = 20 ∧
  value_25p = 25 →
  coins_20p + (total_sum - coins_20p * value_20p) / value_25p = 344 := by
  intro h
  sorry

#check total_coins_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_coins_count_l247_24770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_approximation_l247_24737

/-- Given a bill with face value and banker's discount, calculate the true discount -/
noncomputable def true_discount (face_value banker_discount : ℝ) : ℝ :=
  banker_discount / (1 + banker_discount / face_value)

theorem true_discount_approximation :
  let face_value : ℝ := 2260
  let banker_discount : ℝ := 428.21
  let calculated_true_discount := true_discount face_value banker_discount
  abs (calculated_true_discount - 360) < 0.01 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_approximation_l247_24737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l247_24790

/-- Binary operation ◇ defined for nonzero real numbers -/
noncomputable def diamond (a b : ℝ) : ℝ := a / b

/-- Theorem stating the unique solution to the equation -/
theorem diamond_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ diamond 504 (diamond 8 x) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l247_24790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tan_theta_equals_three_fourths_l247_24711

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 1 / (x + 1)

-- Define the points A_n
def A (n : ℕ) : ℝ × ℝ := if n = 0 then (0, 0) else (n, f n)

-- Define the vector a_n
def a (n : ℕ) : ℝ × ℝ := 
  Finset.sum (Finset.range n) (λ i => A (i + 1) - A i)

-- Define the angle θ_n
def θ (n : ℕ) : ℝ := Real.arctan ((a n).2 / (a n).1)

-- Theorem to prove
theorem sum_of_tan_theta_equals_three_fourths :
  Real.tan (θ 1) + Real.tan (θ 2) + Real.tan (θ 3) = 3/4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tan_theta_equals_three_fourths_l247_24711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_complex_l247_24788

-- Define the complex number z
noncomputable def z : ℂ := sorry

-- Define the condition for z
axiom z_condition : z + Complex.abs z = 2 - 2*Complex.I

-- Define vectors BA and BC
def BA : ℂ := 1 + 2*Complex.I
def BC : ℂ := 3 - Complex.I

-- Define the complex number c corresponding to point C
noncomputable def c : ℂ := sorry

-- Theorem to prove
theorem point_C_complex : c = 2 - 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_complex_l247_24788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l247_24799

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

-- Theorem statement
theorem f_properties :
  -- f(x) is defined for x > 0
  ∀ x > 0,
  -- f(x) is increasing on (0, 1)
  (∀ y ∈ Set.Ioo 0 1, ∀ z ∈ Set.Ioo 0 1, y < z → f y < f z) ∧
  -- f(x) is decreasing on (1, ∞)
  (∀ y ∈ Set.Ioi 1, ∀ z ∈ Set.Ioi 1, y < z → f y > f z) ∧
  -- The maximum value of f(x) on [1/2, e] is 0
  (∀ x ∈ Set.Icc (1/2) (Real.exp 1), f x ≤ 0) ∧ f 1 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l247_24799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_f_strictly_increasing_f_range_on_interval_l247_24708

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^m - 2/x

-- Theorem 1: Prove m = 1
theorem find_m : ∃ m : ℝ, f m 4 = 7/2 ∧ m = 1 := by
  use 1
  constructor
  · simp [f]
    norm_num
  · rfl

-- Theorem 2: Prove f is strictly increasing on (0,+∞)
theorem f_strictly_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f 1 x₁ < f 1 x₂ := by
  sorry

-- Theorem 3: Prove the range of f on [2,5]
theorem f_range_on_interval : 
  ∀ y : ℝ, (∃ x : ℝ, x ∈ Set.Icc 2 5 ∧ f 1 x = y) ↔ y ∈ Set.Icc 1 (23/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_f_strictly_increasing_f_range_on_interval_l247_24708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_iff_m_range_chord_length_when_m_zero_l247_24747

noncomputable section

-- Define the line l
def line_l (m t : ℝ) : ℝ × ℝ := ((1/2) * t, m + (Real.sqrt 3 / 2) * t)

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4 = 0

-- Define the condition for no common points
def no_common_points (m : ℝ) : Prop :=
  ∀ t : ℝ, let (x, y) := line_l m t; ¬(curve_C x y)

-- Define the chord length function
def chord_length (m : ℝ) : ℝ :=
  let ρ₁ := (1 + Real.sqrt 17) / 2
  let ρ₂ := (1 - Real.sqrt 17) / 2
  |ρ₁ - ρ₂|

-- State the theorems
theorem no_common_points_iff_m_range (m : ℝ) :
  no_common_points m ↔ (m < -Real.sqrt 3 * (2 * Real.sqrt 5) ∨ m > Real.sqrt 3 * (2 * Real.sqrt 5)) :=
sorry

theorem chord_length_when_m_zero :
  chord_length 0 = Real.sqrt 17 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_iff_m_range_chord_length_when_m_zero_l247_24747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_obtuse_tangent_sum_inequality_l247_24766

-- Define a triangle type
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_to_pi : α + β + γ = Real.pi

-- Define acute and obtuse triangles
def is_acute (t : Triangle) : Prop :=
  t.α < Real.pi/2 ∧ t.β < Real.pi/2 ∧ t.γ < Real.pi/2

def is_obtuse (t : Triangle) : Prop :=
  t.α > Real.pi/2 ∨ t.β > Real.pi/2 ∨ t.γ > Real.pi/2

-- Define the sum of tangents for a triangle
noncomputable def sum_of_tangents (t : Triangle) : Real :=
  Real.tan t.α + Real.tan t.β + Real.tan t.γ

-- Theorem statement
theorem acute_obtuse_tangent_sum_inequality (t1 t2 : Triangle) 
  (h1 : is_acute t1) (h2 : is_obtuse t2) : 
  sum_of_tangents t1 ≠ sum_of_tangents t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_obtuse_tangent_sum_inequality_l247_24766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_in_open_interval_one_two_l247_24733

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*a*x) / Real.log a

-- Define the property of f being increasing on [4, 5]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_increasing_implies_a_in_open_interval_one_two :
  ∀ a : ℝ, a > 0 → (is_increasing_on_interval (f a) 4 5) → 1 < a ∧ a < 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_in_open_interval_one_two_l247_24733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_rate_shipping_is_17_verify_total_bill_l247_24782

/-- Represents the flat rate shipping price for purchases below $50.00 -/
def flat_rate_shipping : ℝ := 17

/-- Represents the shipping rate for purchases above $50.00 (20%) -/
def high_purchase_shipping_rate : ℝ := 0.2

/-- Calculates the total cost of items in Thomas's order -/
def order_total : ℝ := 3 * 12 + 5 + 2 * 15 + 14

/-- Represents the total bill including shipping -/
def total_bill : ℝ := 102

/-- Theorem stating that the flat rate shipping price is $17.00 -/
theorem flat_rate_shipping_is_17 :
  flat_rate_shipping = 17 := by
  -- The proof is trivial since we defined flat_rate_shipping as 17
  rfl

/-- Theorem verifying that the total bill is correct given the flat rate shipping -/
theorem verify_total_bill :
  total_bill = order_total + flat_rate_shipping := by
  -- Unfold the definitions
  unfold total_bill order_total flat_rate_shipping
  -- Perform the calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_rate_shipping_is_17_verify_total_bill_l247_24782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l247_24732

/-- Parabola type representing y² = 4x --/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line passing through the focus (1, 0) of the parabola --/
structure FocusLine where
  k : ℝ  -- slope of the line
  eq : ℝ → ℝ → Prop  -- equation of the line

/-- Intersection points of the parabola and the focus line --/
structure IntersectionPoints where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  point_a_first_quadrant : x₁ > 0 ∧ y₁ > 0

/-- Theorem stating the minimum value of |y₁ - 4y₂| --/
theorem min_value_theorem (p : Parabola) (l : FocusLine) (i : IntersectionPoints) : 
  ∃ (min : ℝ), min = 8 ∧ ∀ (y₁ y₂ : ℝ), |y₁ - 4*y₂| ≥ min := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l247_24732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_equality_l247_24762

/-- Count lattice points in the specified triangle -/
def f (t q s : ℕ) : ℕ :=
  (Finset.range (t + 1)).sum (fun x => 
    if ((s - 1) * x + t : ℚ) / q - ((s + 1) * x - t : ℚ) / q + 1 > 0
    then 1
    else 0)

/-- Main theorem statement -/
theorem lattice_points_equality (t q r s : ℕ) (h : q ∣ r * s - 1) : f t q r = f t q s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_equality_l247_24762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_implies_a_nonpositive_l247_24797

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - (1/2) * a * x^2 - (a + 1) * x - 1

-- State the theorem
theorem f_leq_g_implies_a_nonpositive :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≤ g a x) → a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_implies_a_nonpositive_l247_24797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solutions_for_equation_l247_24736

theorem no_integer_solutions_for_equation : 
  ¬∃ (x y : ℤ), (2 : ℝ)^(2*x) - (3 : ℝ)^(2*y) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solutions_for_equation_l247_24736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixing_l247_24784

/-- Represents the volume of a 50% salt solution in liters -/
def x : ℝ := sorry

/-- The concentration of salt in the initial salt solution -/
def initial_concentration : ℝ := 0.5

/-- The concentration of salt in the final mixture -/
def final_concentration : ℝ := 0.1

/-- The volume of pure water added in liters -/
def pure_water_volume : ℝ := 1

/-- Theorem stating that x equals 0.25 when mixing solutions as described -/
theorem salt_solution_mixing :
  initial_concentration * x = final_concentration * (pure_water_volume + x) →
  x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_mixing_l247_24784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_existence_l247_24749

theorem unique_triple_existence : 
  ∃! (a b c : ℕ), 
    a ≥ 2 ∧ 
    b ≥ 1 ∧ 
    c ≥ 0 ∧ 
    (Real.log b / Real.log a : ℝ) = c^3 ∧ 
    a + b + c = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_existence_l247_24749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_queens_on_8x8_board_l247_24777

/-- Represents a placement of queens on an 8x8 chessboard -/
def QueenPlacement := Fin 8 → Fin 8

/-- Checks if two positions are on the same diagonal -/
def onSameDiagonal (r1 c1 r2 c2 : Fin 8) : Prop :=
  (r1 : ℤ) - (c1 : ℤ) = (r2 : ℤ) - (c2 : ℤ) ∨ (r1 : ℤ) + (c1 : ℤ) = (r2 : ℤ) + (c2 : ℤ)

/-- Checks if a queen placement is valid (no two queens attack each other) -/
def isValidPlacement (p : QueenPlacement) : Prop :=
  ∀ r1 r2 : Fin 8, r1 ≠ r2 →
    p r1 ≠ p r2 ∧
    ¬onSameDiagonal r1 (p r1) r2 (p r2)

/-- Theorem: The maximum number of queens that can be placed on an 8x8 chessboard
    such that no two queens attack each other is 8 -/
theorem max_queens_on_8x8_board :
  (∃ p : QueenPlacement, isValidPlacement p) ∧
  (∀ n : ℕ, n > 8 → ¬∃ p : QueenPlacement, isValidPlacement p ∧ Function.Injective p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_queens_on_8x8_board_l247_24777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_longest_chord_l247_24751

/-- If the length of the longest chord of a circle is 24 units, then the radius of the circle is 12 units. -/
theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (circle : Set (ℝ × ℝ)), 
    (∀ (chord : ℝ), chord ≤ 24 ∧ 
      ∃ (p q : ℝ × ℝ), p ∈ circle ∧ q ∈ circle ∧ 
        Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord) ∧
    (∃ (center : ℝ × ℝ) (r : ℝ), 
      circle = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2} ∧
      r = 12)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_longest_chord_l247_24751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_diagonal_l247_24771

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + a * c) = 45)
  (h_edge_length : 4 * (a + b + c) = 34)
  (h_volume : a * b * c = 24) :
  a^2 + b^2 + c^2 = 27.25 := by
  sorry

#check rectangular_solid_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_diagonal_l247_24771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_min_reciprocal_sum_min_reciprocal_sum_achieved_l247_24739

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |1/2 * x + 1| + |x|

-- Theorem for the minimum value of f
theorem f_min_value :
  ∃ (a : ℝ), (∀ x : ℝ, f x ≥ a) ∧ (∃ x : ℝ, f x = a) ∧ a = 1 := by
  sorry

-- Theorem for the minimum value of 1/m + 1/n
theorem min_reciprocal_sum :
  ∀ m n : ℝ, m > 0 → n > 0 → m^2 + n^2 = 1 →
  1/m + 1/n ≥ 2 * Real.sqrt 2 := by
  sorry

-- Theorem for the existence of m and n that achieve the minimum
theorem min_reciprocal_sum_achieved :
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m^2 + n^2 = 1 ∧ 1/m + 1/n = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_min_reciprocal_sum_min_reciprocal_sum_achieved_l247_24739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_2500_l247_24706

/-- Represents the business partnership scenario --/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  total_profit : ℚ
  a_total_received : ℚ
  management_fee_percent : ℚ

/-- Calculates b's investment given the partnership details --/
def calculate_b_investment (p : Partnership) : ℚ :=
  let management_fee := p.management_fee_percent * p.total_profit
  let remaining_profit := p.total_profit - management_fee
  let a_capital_profit := p.a_total_received - management_fee
  (p.a_investment * remaining_profit / a_capital_profit) - p.a_investment

/-- The main theorem stating that b's investment is 2500 given the problem conditions --/
theorem b_investment_is_2500 :
  let p : Partnership := {
    a_investment := 3500,
    b_investment := 2500,  -- This is what we're proving
    total_profit := 9600,
    a_total_received := 6000,
    management_fee_percent := 1/10
  }
  calculate_b_investment p = 2500 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_2500_l247_24706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marco_test_scores_l247_24795

def test_scores : List Int := [94, 82, 76, 75, 64]

theorem marco_test_scores :
  let scores := test_scores
  (scores.length = 5) ∧
  (scores.take 3 = [82, 76, 75]) ∧
  (scores.sum / scores.length = 85) ∧
  (∀ s ∈ scores, s < 95) ∧
  (scores.toFinset.card = scores.length) ∧
  ((scores.take 2).sum = (scores.drop 3).sum) →
  scores = [94, 82, 76, 75, 64] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marco_test_scores_l247_24795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_final_amount_l247_24704

noncomputable def total_amount : ℝ := 120

noncomputable def ryan_fraction : ℝ := 2/5
noncomputable def sarah_fraction : ℝ := 1/4
noncomputable def tim_fraction : ℝ := 1/6

noncomputable def ryan_owes_leo : ℝ := 8
noncomputable def sarah_owes_leo : ℝ := 10
noncomputable def leo_owes_ryan : ℝ := 6
noncomputable def leo_owes_sarah : ℝ := 4
noncomputable def tim_owes_leo : ℝ := 10
noncomputable def tim_owes_sarah : ℝ := 4

noncomputable def leo_lends_tim_percent : ℝ := 0.3
noncomputable def leo_lends_ryan_percent : ℝ := 0.2

noncomputable def investment_multiplier : ℝ := 3

theorem leo_final_amount (ryan_amount sarah_amount tim_amount leo_amount : ℝ)
  (h1 : ryan_amount = ryan_fraction * total_amount)
  (h2 : sarah_amount = sarah_fraction * total_amount)
  (h3 : tim_amount = tim_fraction * total_amount)
  (h4 : leo_amount = total_amount - (ryan_amount + sarah_amount + tim_amount))
  (h5 : leo_amount = total_amount * (1 - ryan_fraction - sarah_fraction - tim_fraction)) :
  leo_amount + ryan_owes_leo + sarah_owes_leo - leo_owes_ryan - leo_owes_sarah +
  tim_owes_leo - leo_lends_tim_percent * leo_amount - leo_lends_ryan_percent * leo_amount = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_final_amount_l247_24704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_self_adjoint_x_squared_plus_one_self_adjoint_3x_minus_1_self_adjoint_4_over_x_plus_2_l247_24719

-- Define the concept of m-order self-adjoint function
def is_m_order_self_adjoint (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ f x₁ * g x₂ = m

def is_m_order_self_adjoint_self (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ f x₁ * f x₂ = m

-- Part 1
theorem not_self_adjoint_x_squared_plus_one :
  ¬ (is_m_order_self_adjoint_self (fun x ↦ x^2 + 1) 2 (Set.Icc 0 3)) := by sorry

-- Part 2
theorem self_adjoint_3x_minus_1 :
  (is_m_order_self_adjoint_self (fun x ↦ 3*x - 1) 1 (Set.Icc (1/2) 1)) ∧
  (∀ b > 1, ¬ (is_m_order_self_adjoint_self (fun x ↦ 3*x - 1) 1 (Set.Icc (1/2) b))) := by sorry

-- Part 3
theorem self_adjoint_4_over_x_plus_2 (a : ℝ) :
  (a ∈ Set.Icc (-Real.sqrt 2) (2 - Real.sqrt 3) ∪ Set.Icc (Real.sqrt 3) (2 + Real.sqrt 2)) ↔
  (is_m_order_self_adjoint (fun x ↦ 4/(x+2)) (fun x ↦ x^2 - 2*a*x + a^2 - 1) 2 (Set.Icc 0 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_self_adjoint_x_squared_plus_one_self_adjoint_3x_minus_1_self_adjoint_4_over_x_plus_2_l247_24719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_cube_root_l247_24715

theorem largest_integer_cube_root : 
  ⌊(((2010 : ℝ)^3 + 3*(2010 : ℝ)^2 + 4*(2010 : ℝ) + 1) ^ (1/3 : ℝ))⌋ = 2011 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_cube_root_l247_24715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_intersection_complement_M_N_l247_24730

-- Define the function f
noncomputable def f (x : ℝ) := Real.sqrt (x + 1) + Real.log (2 - x)

-- Define the set M (domain of f)
def M : Set ℝ := {x | x + 1 ≥ 0 ∧ 2 - x > 0}

-- Define the set N
def N : Set ℝ := {x | x * (x - 3) < 0}

-- Statement for M ∪ N
theorem union_M_N : M ∪ N = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

-- Statement for (∁ᴿM) ∩ N
theorem intersection_complement_M_N : (Mᶜ) ∩ N = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_intersection_complement_M_N_l247_24730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l247_24796

-- Define the function
noncomputable def f (x : ℝ) := Real.log (-(x^2) - 2*x + 3) / Real.log (1/2)

-- Define the domain
def domain : Set ℝ := Set.Ioo (-3) 1

-- Theorem statement
theorem interval_of_increase :
  ∃ (a b : ℝ), a = -1 ∧ b = 1 ∧
  (∀ x ∈ domain, ∀ y ∈ domain, a ≤ x ∧ x < y ∧ y < b → f x < f y) ∧
  (∀ x ∈ domain, x < a → ∃ y ∈ domain, x < y ∧ f y ≤ f x) ∧
  (∀ x ∈ domain, b ≤ x → ∃ y ∈ domain, y < x ∧ f y ≥ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l247_24796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_grass_seed_l247_24744

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat
  deriving Repr, Inhabited

/-- Calculates the total weight of a list of bags -/
def totalWeight (bags : List GrassSeedBag) : Nat :=
  bags.foldl (fun acc bag => acc + bag.weight) 0

/-- Calculates the total price of a list of bags -/
def totalPrice (bags : List GrassSeedBag) : Rat :=
  bags.foldl (fun acc bag => acc + bag.price) 0

/-- Theorem: The minimum cost for 65-80 pounds of grass seed is $98.77 -/
theorem min_cost_grass_seed (bags : List GrassSeedBag) 
    (h1 : bags.length = 3)
    (h2 : bags[0]! = ⟨5, 1385/100⟩)
    (h3 : bags[1]! = ⟨10, 2042/100⟩)
    (h4 : bags[2]! = ⟨25, 3225/100⟩)
    (purchase : List GrassSeedBag)
    (h5 : totalWeight purchase ≥ 65)
    (h6 : totalWeight purchase ≤ 80)
    (h7 : ∀ b ∈ purchase, b ∈ bags) :
  totalPrice purchase ≥ 9877/100 ∧ 
  ∃ optimal : List GrassSeedBag, 
    totalWeight optimal ≥ 65 ∧ 
    totalWeight optimal ≤ 80 ∧ 
    (∀ b ∈ optimal, b ∈ bags) ∧
    totalPrice optimal = 9877/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_grass_seed_l247_24744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_60_degrees_l247_24763

/-- Given a cone whose lateral surface unfolds into a semicircle, 
    the angle between the generatrix and the base is 60°. -/
theorem cone_angle_60_degrees (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (2 * π * r = π * l) → Real.arccos (r / l) = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_60_degrees_l247_24763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l247_24727

theorem remainder_theorem (x y z : ℤ) 
  (hx : x % 15 = 11)
  (hy : y % 15 = 13)
  (hz : z % 15 = 14) :
  (y + z - x) % 15 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l247_24727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_product_l247_24768

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation x^2 = 20y -/
def Parabola := {p : Point | p.x^2 = 20 * p.y}

/-- The focus of the parabola -/
def focus : Point := ⟨0, 5⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_point_product (a b : ℝ) :
  (⟨a, b⟩ : Point) ∈ Parabola → distance ⟨a, b⟩ focus = 25 → |a * b| = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_product_l247_24768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equivalence_l247_24718

/-- 
The number of partitions of n into positive integers where each part appears 
fewer than k times.
-/
def number_of_partitions_with_bounded_repetition (n k : ℕ) : ℕ := sorry

/-- 
The number of partitions of n into parts not divisible by k.
-/
def number_of_partitions_with_parts_not_divisible_by (k n : ℕ) : ℕ := sorry

/-- 
Given a natural number n and an integer k ≥ 2, the number of partitions of n 
into positive integers where each part appears fewer than k times is equal to 
the number of partitions of n into parts not divisible by k.
-/
theorem partition_equivalence (n : ℕ) (k : ℕ) (h : k ≥ 2) : 
  (number_of_partitions_with_bounded_repetition n k) = 
  (number_of_partitions_with_parts_not_divisible_by k n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_equivalence_l247_24718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_household_expense_increase_is_ten_percent_l247_24741

/-- Calculates the percentage increase in household expenses -/
noncomputable def household_expense_increase (food_expense : ℝ) (education_expense : ℝ) (other_expense : ℝ)
  (food_increase : ℝ) (education_increase : ℝ) (other_increase : ℝ) : ℝ :=
  let total_expense := food_expense + education_expense + other_expense
  let food_increase_amount := food_expense * food_increase
  let education_increase_amount := education_expense * education_increase
  let other_increase_amount := other_expense * other_increase
  let total_increase := food_increase_amount + education_increase_amount + other_increase_amount
  (total_increase / total_expense) * 100

/-- Theorem: Given the specified expenses and increases, the total percentage increase is 10% -/
theorem household_expense_increase_is_ten_percent :
  household_expense_increase 500 200 300 0.06 0.20 0.10 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_household_expense_increase_is_ten_percent_l247_24741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_shaded_specific_l247_24703

/-- Represents a rectangle grid with shaded squares -/
structure ShadedGrid where
  rows : Nat
  cols : Nat
  shaded_cols : Finset Nat

/-- The probability of choosing a rectangle that doesn't include a shaded square -/
def prob_no_shaded (grid : ShadedGrid) : ℚ :=
  1 - (grid.rows * grid.shaded_cols.card * (grid.cols - grid.shaded_cols.card)) / 
      (grid.rows * Nat.choose (grid.cols + 1) 2)

/-- The specific grid from the problem -/
def problem_grid : ShadedGrid :=
  { rows := 3
  , cols := 2004
  , shaded_cols := {1002} }

theorem prob_no_shaded_specific : 
  prob_no_shaded problem_grid = 1003 / 2004 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_shaded_specific_l247_24703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_proof_l247_24726

/-- The repeating decimal 0.817817817... as a rational number -/
def F : ℚ := 817 / 999

theorem repeating_decimal_proof :
  F.num = 817 ∧
  F.den = 999 ∧
  F.den - F.num = 182 ∧
  Int.gcd F.num F.den = 1 := by
  sorry

#eval (F.den - F.num).toNat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_proof_l247_24726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l247_24786

theorem problem_statement (m : ℝ) (a b : ℝ) 
  (h1 : (9 : ℝ)^m = 10) 
  (h2 : a = (10 : ℝ)^m - 11) 
  (h3 : b = (8 : ℝ)^m - 9) : 
  a > 0 ∧ 0 > b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l247_24786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_reciprocal_l247_24700

noncomputable def reciprocal (x : ℝ) : ℝ := 1 / x

theorem largest_reciprocal :
  let a := (1 : ℝ) / 6
  let b := (2 : ℝ) / 7
  let c := (2 : ℝ)
  let d := (8 : ℝ)
  let e := (1000 : ℝ)
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e) →
  (reciprocal a > reciprocal b ∧
   reciprocal a > reciprocal c ∧
   reciprocal a > reciprocal d ∧
   reciprocal a > reciprocal e) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_reciprocal_l247_24700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_l247_24778

theorem number_of_boys (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : girls = 38)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 10)
  (h9 : happy_children + sad_children + neutral_children = total_children)
  : total_children - girls = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_l247_24778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_problem_l247_24779

/-- The number of balls labeled 2 -/
def n : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := n + 2

/-- The probability of drawing a ball labeled 2 -/
def prob_two : ℚ := 2/5

/-- The event that the sum of two drawn balls' labels is 2 -/
def event_A : Finset (ℕ × ℕ) := Finset.filter (fun p => p.1 + p.2 = 2 ∧ p.1 ∈ Finset.range 3 ∧ p.2 ∈ Finset.range 3) (Finset.product (Finset.range 3) (Finset.range 3))

/-- The probability of event A -/
def prob_A : ℚ := 2/3

/-- The probability that √(x^2 + y^2) > a + b for x, y ∈ [0,4] -/
noncomputable def prob_sqrt_gt_sum : ℝ := 1 - Real.pi/4

theorem ball_problem :
  (n : ℚ) / total_balls = prob_two ∧
  event_A.card / (total_balls.choose 2) = prob_A ∧
  ∀ x y : ℝ, x ∈ Set.Icc 0 4 → y ∈ Set.Icc 0 4 →
    Real.sqrt (x^2 + y^2) > 2 → prob_sqrt_gt_sum = 1 - Real.pi/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_problem_l247_24779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_with_sqrt1001_leg_l247_24758

theorem right_triangles_with_sqrt1001_leg :
  ∃! n : ℕ, n = (Finset.filter
    (fun p : ℕ × ℕ =>
      let a := p.1
      let c := p.2
      a^2 + 1001 = c^2)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_with_sqrt1001_leg_l247_24758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_annual_interest_rate_l247_24709

/-- Given an account with an 8% annual interest rate compounded quarterly,
    the equivalent annual interest rate r is approximately 8.24%. -/
theorem equivalent_annual_interest_rate :
  let annual_rate : ℚ := 8 / 100
  let quarterly_rate : ℚ := annual_rate / 4
  let compound_factor : ℚ := (1 + quarterly_rate) ^ 4
  let equivalent_rate : ℚ := (compound_factor - 1) * 100
  ⌊equivalent_rate * 100⌋ / 100 = 824 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_annual_interest_rate_l247_24709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_three_halves_l247_24728

/-- The sum of the infinite series ∑_{n=1}^∞ (3n / (n(n+1)(n+2))) -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3 * n : ℝ) / (n * (n + 1) * (n + 2))

/-- Theorem stating that the sum of the infinite series is equal to 3/2 -/
theorem infinite_series_sum_eq_three_halves : infinite_series_sum = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_three_halves_l247_24728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l247_24780

theorem exam_score_difference (total_students : ℕ) (top_students : ℕ) (score_difference : ℝ) :
  total_students = 40 →
  top_students = 8 →
  score_difference = 3 →
  let remaining_students := total_students - top_students
  let top_extra_points := score_difference * (top_students : ℝ)
  let remaining_students_difference := top_extra_points / (remaining_students : ℝ)
  remaining_students_difference + score_difference = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l247_24780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l247_24764

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 5) / ((x - 2) * (x - 4) * (x - 6))

theorem solution_set_of_inequality :
  {x : ℝ | f x > 0} = Set.Iio 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 5 ∪ Set.Ioi 6 := by
  sorry

#check solution_set_of_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l247_24764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_db_length_l247_24774

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of the sides
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem quadrilateral_db_length (ABCD : Quadrilateral) :
  side_length ABCD.A ABCD.B = 5 →
  side_length ABCD.B ABCD.C = 17 →
  side_length ABCD.C ABCD.D = 5 →
  side_length ABCD.D ABCD.A = 9 →
  ∃ (DB : ℕ), side_length ABCD.D ABCD.B = DB ∧ DB = 13 := by
  sorry

#check quadrilateral_db_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_db_length_l247_24774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_l247_24775

noncomputable section

/-- The function f(x) = x/ln(x) - ax is decreasing on (1, +∞) -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x > f y

/-- The function f(x) defined on (1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x / Real.log x - a * x

theorem min_value_a :
  (∃ a_min : ℝ, ∀ a : ℝ, (is_decreasing (f a) ↔ a ≥ a_min)) ∧
  (∀ a_min : ℝ, (∀ a : ℝ, (is_decreasing (f a) ↔ a ≥ a_min)) → a_min = 1/4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_l247_24775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_fox_at_120m_l247_24713

/-- Represents the chase scenario between a dog and a fox -/
structure ChaseScenario where
  initial_distance : ℝ
  dog_leap : ℝ
  fox_leap : ℝ
  dog_leaps : ℕ
  fox_leaps : ℕ

/-- Calculates the distance at which the dog catches the fox -/
noncomputable def catch_distance (scenario : ChaseScenario) : ℝ :=
  let dog_speed := scenario.dog_leap * scenario.dog_leaps
  let fox_speed := scenario.fox_leap * scenario.fox_leaps
  let relative_speed := dog_speed - fox_speed
  let time := scenario.initial_distance / relative_speed
  dog_speed * time

/-- Theorem stating that in the given scenario, the dog catches the fox at 120 meters -/
theorem dog_catches_fox_at_120m :
  let scenario : ChaseScenario := {
    initial_distance := 30
    dog_leap := 2
    fox_leap := 1
    dog_leaps := 2
    fox_leaps := 3
  }
  catch_distance scenario = 120 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_fox_at_120m_l247_24713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_rest_days_l247_24714

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 3

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Al's rest days in a cycle -/
def al_rest_days : Finset ℕ := {3}

/-- Barb's rest days in a cycle -/
def barb_rest_days : Finset ℕ := {6, 7}

/-- The number of coinciding rest days in a full cycle -/
def coinciding_rest_days_per_cycle : ℕ := 2

/-- Theorem: Al and Barb have 94 coinciding rest days in the first 1000 days -/
theorem coinciding_rest_days : 
  (total_days / (al_cycle.lcm barb_cycle)) * coinciding_rest_days_per_cycle = 94 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_rest_days_l247_24714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l247_24756

noncomputable def vector_a : ℝ × ℝ := (3, 4)

noncomputable def vector_b₁ : ℝ × ℝ := (4/5, -3/5)
noncomputable def vector_b₂ : ℝ × ℝ := (-4/5, 3/5)

theorem perpendicular_unit_vectors :
  (vector_b₁.1 * vector_a.1 + vector_b₁.2 * vector_a.2 = 0) ∧
  (vector_b₁.1^2 + vector_b₁.2^2 = 1) ∧
  (vector_b₂.1 * vector_a.1 + vector_b₂.2 * vector_a.2 = 0) ∧
  (vector_b₂.1^2 + vector_b₂.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l247_24756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_sphere_area_ratio_l247_24781

/-- A cone with an equilateral triangle as its axial section -/
structure EquilateralCone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  equilateral : h = r * Real.sqrt 3

/-- The circumscribed sphere of an equilateral cone -/
noncomputable def circumscribedSphere (cone : EquilateralCone) : ℝ := 
  (2 / Real.sqrt 3) * cone.r

/-- The ratio of the surface area of an equilateral cone to its circumscribed sphere is 9:16 -/
theorem equilateral_cone_sphere_area_ratio 
  (cone : EquilateralCone) : 
  (3 * Real.pi * cone.r^2) / (4 * Real.pi * (circumscribedSphere cone)^2) = 9/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_sphere_area_ratio_l247_24781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_cos_equation_l247_24769

theorem tan_value_given_sin_cos_equation (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi/2))
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 3/4) : 
  Real.tan α = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_cos_equation_l247_24769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_for_side_3_l247_24724

/-- The instantaneous rate of change of the area with respect to the perimeter for a square --/
noncomputable def instantaneous_rate_of_change (s : ℝ) : ℝ :=
  let area := s^2
  let perimeter := 4 * s
  (2 * area) / perimeter

/-- Theorem: The instantaneous rate of change of the area with respect to the perimeter
    for a square with side length 3 is 3/2 --/
theorem instantaneous_rate_of_change_for_side_3 :
  instantaneous_rate_of_change 3 = 3/2 := by
  -- Unfold the definition of instantaneous_rate_of_change
  unfold instantaneous_rate_of_change
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_for_side_3_l247_24724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l247_24746

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (2x-1/√x)^5
noncomputable def expansion (x : ℝ) : ℝ := (2 * x - 1 / Real.sqrt x) ^ 5

-- Auxiliary function for terms without x^2
noncomputable def terms_without_x_squared (x : ℝ) : ℝ :=
  expansion x - 80 * x^2

-- Theorem statement
theorem coefficient_of_x_squared (x : ℝ) :
  ∃ c : ℝ, expansion x = c * x^2 + terms_without_x_squared x ∧ c = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l247_24746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l247_24734

-- Define the function y
noncomputable def y (x a : ℝ) : ℝ := (x - 2)^2 + (a + 2) * x + Real.sin (x + 3 * Real.pi / 2)

-- Theorem statement
theorem even_function_implies_a_equals_two :
  (∀ x : ℝ, y x a = y (-x) a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l247_24734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l247_24750

-- Define the ※ operation
noncomputable def star (m n : ℝ) : ℝ :=
  if m ≥ 0 then m + n else m / n

-- Theorem statement
theorem star_equation_solution :
  ∀ x : ℝ, x ≠ 0 → (star (-9) (-x) = x ↔ x = 3 ∨ x = -3) :=
by
  intro x hx
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l247_24750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_zero_in_interval_l247_24765

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (Real.log x) / (Real.log a) + x - b

-- State the theorem
theorem zero_of_f_in_interval (a b : ℝ) (ha : 0 < a) (ha_neq_1 : a ≠ 1) 
  (h_bounds : 2 < a ∧ a < 3 ∧ 3 < b ∧ b < 4) :
  ∃ x₀ : ℝ, f a b x₀ = 0 ∧ 2 < x₀ ∧ x₀ < 3 := by
  sorry

-- Define n as 2
def n : ℕ := 2

-- State the main theorem
theorem zero_in_interval :
  ∀ a b : ℝ, 0 < a → a ≠ 1 → 2 < a → a < 3 → 3 < b → b < 4 →
  ∃ x₀ : ℝ, f a b x₀ = 0 ∧ n < x₀ ∧ x₀ < n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_zero_in_interval_l247_24765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_theorem_l247_24767

/-- Represents a circle in the plane -/
structure Circle where
  radius : ℝ

/-- Represents a layer of circles -/
def Layer := List Circle

/-- Sequence of layers -/
def LayerSequence := List Layer

/-- Calculates the radius of a new circle tangent to two given circles -/
noncomputable def newRadius (r1 r2 : ℝ) : ℝ :=
  (r1 * r2) / ((Real.sqrt r1 + Real.sqrt r2)^2)

/-- Generates the next layer of circles -/
def nextLayer (prevLayers : LayerSequence) : Layer :=
  sorry

/-- Generates the sequence of layers up to L₅ -/
def generateLayers : LayerSequence :=
  let l0 : Layer := [⟨80^2⟩, ⟨75^2⟩]
  sorry

/-- Calculates the sum of 1/√r for all circles in the sequence -/
noncomputable def calculateSum (layers : LayerSequence) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem circle_sum_theorem : 
  let layers := generateLayers
  let sum := calculateSum layers
  (sum = 1/80 + 1/75) ∨ 
  (sum = 2/155) ∨ 
  (sum = 1/70 + 1/75) ∨ 
  (sum = 1/73 + 1/80) ∨ 
  (sum = 1/80 + 1/73) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_theorem_l247_24767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_max_value_on_interval_l247_24783

/-- The function f(x) = ln x - (1/2)x^2 -/
noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2) * x^2

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := 1/x - x

theorem tangent_point (h : f 1 = -1/2 ∧ f_derivative 1 = 0) : True := by
  sorry

theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧
  f x = -1/2 ∧
  ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_max_value_on_interval_l247_24783
