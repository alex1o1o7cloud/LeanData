import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travelers_meet_time_l78_7830

/-- Represents the distance traveled by the second traveler in t hours -/
noncomputable def distance_B (t : ℝ) : ℝ :=
  (t / 2) * (7.5 + 0.5 * t)

/-- Theorem stating that the travelers meet after 8 hours -/
theorem travelers_meet_time :
  ∃ (t : ℝ), t = 8 ∧ 5 * t + distance_B t = 100 := by
  use 8
  constructor
  · rfl
  · sorry  -- Proof details omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travelers_meet_time_l78_7830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_nonzero_digits_100_factorial_l78_7865

theorem last_three_nonzero_digits_100_factorial :
  ∃ k : ℕ, (Finset.prod (Finset.range 100) (λ i => i + 1)) = k * 10^24 + 376 ∧ k < 10^3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_nonzero_digits_100_factorial_l78_7865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_in_plane_l78_7839

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relation for a point being on a line
variable (isOn : Point → Line → Prop)

-- Define the relation for a line being in a plane
variable (isInPlane : Line → Plane → Prop)

-- Define the specific point, line, and plane
variable (M : Point) (m : Line) (α : Plane)

-- State the theorem
theorem point_on_line_in_plane (h1 : isOn M m) (h2 : isInPlane m α) :
  M ∈ {x : Point | isOn x m} ∧ {x : Point | isOn x m} ⊆ {x : Point | ∃ l : Line, isInPlane l α ∧ isOn x l} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_in_plane_l78_7839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l78_7821

open Real

-- Define the functions f and F
noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := (f x - a) / x

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), F a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (exp 1), F a x = 3/2) →
  a = -sqrt (exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l78_7821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_seat_probability_in_specific_theatre_l78_7869

/-- Represents a theatre with rows and seats per row. -/
structure Theatre where
  rows : ℕ
  seatsPerRow : ℕ

/-- Calculates the total number of seats in the theatre. -/
def Theatre.totalSeats (t : Theatre) : ℕ := t.rows * t.seatsPerRow

/-- Calculates the number of adjacent seat pairs in the theatre. -/
def Theatre.adjacentPairs (t : Theatre) : ℕ := t.rows * (t.seatsPerRow - 1)

/-- Calculates the probability of two people occupying adjacent seats when choosing independently. -/
def adjacentSeatProbability (t : Theatre) : ℚ :=
  (t.adjacentPairs : ℚ) / (Nat.choose (t.totalSeats) 2 : ℚ)

theorem adjacent_seat_probability_in_specific_theatre :
  let t : Theatre := { rows := 10, seatsPerRow := 10 }
  adjacentSeatProbability t = 1 / 55 := by
  sorry

#eval adjacentSeatProbability { rows := 10, seatsPerRow := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_seat_probability_in_specific_theatre_l78_7869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l78_7859

/-- Represents the dimensions of a rectangular garden -/
structure RectGarden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
noncomputable def rectArea (g : RectGarden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden -/
noncomputable def rectPerimeter (g : RectGarden) : ℝ := 2 * (g.length + g.width)

/-- Calculates the side length of a square garden with the same perimeter as a given rectangular garden -/
noncomputable def squareSideLength (g : RectGarden) : ℝ := (rectPerimeter g) / 4

/-- Calculates the area of a square garden with the same perimeter as a given rectangular garden -/
noncomputable def squareArea (g : RectGarden) : ℝ := (squareSideLength g) ^ 2

/-- Theorem stating that transforming a 30ft by 12ft rectangular garden into a square garden
    with the same perimeter results in an increase of 81 square feet in area -/
theorem garden_area_increase :
  let g : RectGarden := ⟨30, 12⟩
  squareArea g - rectArea g = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l78_7859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_super_ball_distance_l78_7837

/-- The total distance traveled by a bouncing ball -/
noncomputable def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (bounces : ℕ) : ℝ :=
  let rec bounce (height : ℝ) (n : ℕ) (acc : ℝ) : ℝ :=
    if n = 0 then
      acc + height  -- Final fall
    else
      let newHeight := height * bounceRatio
      bounce newHeight (n - 1) (acc + height + newHeight)
  bounce initialHeight bounces initialHeight

/-- Theorem: A ball dropped from 20 meters, bouncing three times with 2/3 ratio, travels 2060/27 meters -/
theorem super_ball_distance :
  totalDistance 20 (2/3) 3 = 2060/27 := by
  sorry  -- Proof omitted

-- For verification (commented out as it might not be computable)
-- #eval totalDistance 20 (2/3) 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_super_ball_distance_l78_7837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_thrower_l78_7800

/-- Represents the possible room numbers of the boys -/
inductive Room
  | R302 | R401 | R501 | R502 | R402

/-- Represents the statements made by each boy -/
structure Statement where
  notMe : Bool
  fromAbove : Bool
  sawIt : Bool

/-- Represents a boy with his room and statement -/
structure Boy where
  room : Room
  statement : Statement

/-- The main theorem to prove -/
theorem firecracker_thrower (boys : List Boy)
  (h1 : boys.length = 5)
  (h2 : ∃! b, b ∈ boys ∧ b.statement ≠ Statement.mk True (¬b.statement.fromAbove) (¬b.statement.sawIt))
  (h3 : ∀ b, b ∈ boys → b.room = Room.R302 → b.statement = Statement.mk True False False)
  (h4 : ∃ b1 b2, b1 ∈ boys ∧ b2 ∈ boys ∧ b1.statement.fromAbove ∧ b2.statement.fromAbove ∧ b1 ≠ b2)
  : ∃ b, b ∈ boys ∧ b.room = Room.R302 ∧ b.statement ≠ Statement.mk True (¬b.statement.fromAbove) (¬b.statement.sawIt) :=
by
  sorry

#check firecracker_thrower

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firecracker_thrower_l78_7800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_c_l78_7885

theorem right_triangle_sin_c (A B C : ℝ) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2) (h5 : 0 < C) (h6 : C < π/2) : 
  A + B + C = π → Real.sin B = 1 → Real.sin A = 8/17 → Real.sin C = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_c_l78_7885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_height_l78_7870

/-- The height of a balloon given two angles of depression and horizontal distance --/
theorem balloon_height (β γ : Real) (d : Real) (hβ : β = 35.5 * Real.pi / 180) (hγ : γ = 23.25 * Real.pi / 180) (hd : d = 2500) :
  ∃ h : Real, abs (h - 1334) < 1 ∧ h^2 * ((Real.tan (Real.pi/2 - γ))^2 - (Real.tan (Real.pi/2 - β))^2) = d^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_height_l78_7870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_problem_l78_7811

/-- Calculates the speed of the slower train given the conditions of the problem -/
noncomputable def slower_train_speed (faster_train_speed : ℝ) (train1_length train2_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := (train1_length + train2_length) / 1000 -- Convert to km
  let crossing_time_hours := crossing_time / 3600 -- Convert to hours
  let relative_speed := total_distance / crossing_time_hours
  faster_train_speed - relative_speed

/-- The theorem stating the speed of the slower train given the problem conditions -/
theorem slower_train_speed_problem :
  let faster_train_speed : ℝ := 46
  let train1_length : ℝ := 200
  let train2_length : ℝ := 150
  let crossing_time : ℝ := 210
  abs (slower_train_speed faster_train_speed train1_length train2_length crossing_time - 40.05) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_problem_l78_7811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l78_7883

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  /-- The standard form equation of the hyperbola: x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The eccentricity of the hyperbola -/
  eccentricity : ℝ

/-- Represents a line in the xy-plane -/
structure Line where
  /-- The equation of the line: ax + by + c = 0 -/
  equation : ℝ → ℝ → Prop

/-- Given conditions for the hyperbola -/
def hyperbola_conditions (C : Hyperbola) (l : Line) : Prop :=
  ∃ (x y : ℝ),
    l.equation x y ∧
    y = 0 ∧
    ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x' y' : ℝ), l.equation x' y' ↔ l.equation (k * x') (k * y')

/-- The theorem to be proved -/
theorem hyperbola_theorem (C : Hyperbola) (l : Line) :
  hyperbola_conditions C l →
  (l.equation = λ x y ↦ 4 * x - 3 * y + 20 = 0) →
  (C.equation = λ x y ↦ x^2 / 9 - y^2 / 16 = 1) ∧
  (C.eccentricity = 5 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l78_7883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_972_l78_7824

/-- A geometric sequence with 9 terms, first term 4, and last term 78732 -/
def GeometricSequence : Type := 
  { a : Fin 9 → ℝ // a 0 = 4 ∧ a 8 = 78732 ∧ ∀ i j, i < j → (a j) / (a i) = (a 1) / (a 0) }

theorem sixth_term_is_972 (a : GeometricSequence) : a.val 5 = 972 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_972_l78_7824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_A_l78_7828

open Real

theorem max_tan_A :
  ∃ (max_tan_A : ℝ), max_tan_A = 4/3 ∧ 
  ∀ (A B : ℝ), 0 < A ∧ A < π/2 → 0 < B ∧ B < π/2 →
  Real.sin A / Real.sin B = Real.sin (A + B) →
  Real.tan A ≤ max_tan_A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_A_l78_7828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_theorem_l78_7875

theorem function_composition_theorem (A B : ℝ) (h : A ≠ B) :
  let f : ℝ → ℝ := λ x ↦ 3 * A * x + 2 * B
  let g : ℝ → ℝ := λ x ↦ 2 * B * x + 3 * A
  (∀ x : ℝ, f (g x) - g (f x) = 3 * (B - A)) →
  A + B = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_theorem_l78_7875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_angle_between_vectors_l78_7890

def vector_a : Fin 3 → ℝ := ![3, 4, -3]
def vector_b : Fin 3 → ℝ := ![5, -3, 1]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (Finset.range 3).sum (λ i => v i * w i)

noncomputable def angle_between (v w : Fin 3 → ℝ) : ℝ :=
  Real.arccos ((dot_product v w) / (Real.sqrt (dot_product v v) * Real.sqrt (dot_product w w)))

theorem vectors_perpendicular (a b : Fin 3 → ℝ) :
  dot_product a b = 0 → angle_between a b = Real.pi / 2 := by
  sorry

theorem angle_between_vectors :
  angle_between vector_a vector_b = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_angle_between_vectors_l78_7890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l78_7814

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the Cartesian equation of C₃
def C₃_cartesian (x y : ℝ) : Prop := 2 * x - y = 0

-- State the theorem
theorem curves_intersection :
  -- Part 1: Cartesian equation of C₁
  (∀ x y, y ≥ 0 → (∃ t, C₁ t = (x, y)) ↔ y^2 = 6*x - 2) ∧
  -- Part 2: Intersection points of C₃ with C₁
  (∃ θ₁ θ₂, C₃ θ₁ = (1/2, 1) ∧ C₃ θ₂ = (1, 2) ∧
    C₃_cartesian (1/2) 1 ∧ C₃_cartesian 1 2 ∧
    (∃ t₁ t₂, C₁ t₁ = (1/2, 1) ∧ C₁ t₂ = (1, 2))) ∧
  -- Part 3: Intersection points of C₃ with C₂
  (∃ θ₃ θ₄, C₃ θ₃ = (-1/2, -1) ∧ C₃ θ₄ = (-1, -2) ∧
    C₃_cartesian (-1/2) (-1) ∧ C₃_cartesian (-1) (-2) ∧
    (∃ s₁ s₂, C₂ s₁ = (-1/2, -1) ∧ C₂ s₂ = (-1, -2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l78_7814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l78_7858

open Complex

noncomputable def complex_sum : ℂ :=
  exp (8 * Real.pi * I / 40) + exp (13 * Real.pi * I / 40) +
  exp (18 * Real.pi * I / 40) + exp (23 * Real.pi * I / 40) +
  exp (28 * Real.pi * I / 40) + exp (33 * Real.pi * I / 40)

theorem complex_sum_argument :
  arg complex_sum = 41 * Real.pi / 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l78_7858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_C_identities_l78_7846

/-- S function -/
noncomputable def S (a : ℝ) (x : ℝ) : ℝ := (a^x - a^(-x)) / 2

/-- C function -/
noncomputable def C (a : ℝ) (x : ℝ) : ℝ := (a^x + a^(-x)) / 2

/-- Main theorem -/
theorem S_C_identities (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, S a (x + y) = S a x * C a y + C a x * S a y) ∧
  (∀ x y : ℝ, S a (x - y) = S a x * C a y - C a x * S a y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_C_identities_l78_7846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_inequality_proof_l78_7882

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

-- Statement for the tangent line at x = 0
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), ∀ x, m * x + b = (deriv f) 0 * x + f 0 ∧ m = 1 ∧ b = 0 := by sorry

-- Statement for the inequality
theorem inequality_proof (a : ℝ) (h : a ≥ 1) :
  ∀ x > -1, f x ≤ a^2 * Real.exp x - a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_inequality_proof_l78_7882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l78_7863

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 3 * a * x + 2 * Real.cos x

-- Define the slopes of the tangent lines
noncomputable def slope_f (x : ℝ) : ℝ := -Real.exp x - 1
noncomputable def slope_g (a : ℝ) (x : ℝ) : ℝ := 3 * a - 2 * Real.sin x

-- State the theorem
theorem tangent_perpendicular_range (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, (slope_f x₁) * (slope_g a x₂) = -1) ↔
  -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l78_7863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angular_acceleration_formula_l78_7802

/-- The instantaneous angular acceleration of a reel -/
noncomputable def instantaneous_angular_acceleration (v r h : ℝ) : ℝ :=
  h * v^2 / (2 * Real.pi * r^3)

/-- 
Theorem: The instantaneous angular acceleration of a reel with radius r, 
given a tape moving at constant speed v and tape thickness h, 
is equal to hv²/(2πr³).
-/
theorem angular_acceleration_formula 
  (v r h : ℝ) 
  (hv : v > 0) 
  (hr : r > 0) 
  (hh : h > 0) :
  instantaneous_angular_acceleration v r h = h * v^2 / (2 * Real.pi * r^3) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angular_acceleration_formula_l78_7802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_order_l78_7899

open Real

-- Define the functions
noncomputable def f (x : ℝ) := (2 : ℝ)^x + x - 2
noncomputable def g (x : ℝ) := (3 : ℝ)^x + x - 2
noncomputable def h (x : ℝ) := log x + x - 2

-- State the theorem
theorem zero_order (a b c : ℝ) 
  (ha : f a = 0) 
  (hb : g b = 0) 
  (hc : h c = 0) : 
  b < a ∧ a < c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_order_l78_7899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_near_n_l78_7889

theorem sum_of_squares_near_n (n : ℤ) (hn : n > 10000) :
  ∃ m : ℤ, ∃ a b : ℤ,
    m = a^2 + b^2 ∧
    0 < m - n ∧
    (m - n : ℚ) < 3 * (n : ℚ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_near_n_l78_7889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_positions_l78_7857

/-- Represents a position where an additional square can be attached --/
inductive AttachmentPosition
| Top
| Middle
| Bottom
| Left
| Right
| TopLeft
| TopRight
| BottomLeft
| BottomRight
| Center

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Congruence relation for squares --/
def Square.congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side

notation:50 s1 " ≅ " s2 => Square.congruent s1 s2

/-- Represents the L-shaped polygon --/
structure LPolygon where
  squares : Fin 5 → Square
  congruent : ∀ i j : Fin 5, squares i ≅ squares j

/-- Represents the resulting polygon after attaching an additional square --/
structure ResultingPolygon where
  base : LPolygon
  additionalSquare : Square
  attachmentPosition : AttachmentPosition
  congruentToBase : ∀ i : Fin 5, base.squares i ≅ additionalSquare

/-- Predicate to check if a resulting polygon can be folded into a cube with one face missing --/
def canFoldIntoCube (p : ResultingPolygon) : Prop := sorry

/-- The main theorem to be proved --/
theorem four_valid_positions (allPositions : Fin 10 → AttachmentPosition) :
  ∃ validPositions : Finset AttachmentPosition,
    validPositions.card = 4 ∧
    (∀ pos, pos ∈ validPositions ↔ 
      ∃ (p : ResultingPolygon), p.attachmentPosition = pos ∧ canFoldIntoCube p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_valid_positions_l78_7857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_inequality_l78_7851

theorem positive_integer_inequality (m : ℕ+) : -3 / (m : ℚ) > -3 / 7 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_inequality_l78_7851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_discount_l78_7849

/-- Applies a discount percentage to a price -/
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount / 100)

/-- The bookstore discount problem -/
theorem bookstore_discount (initial_price : ℝ) (autumn_discount : ℝ) (loyalty_discount : ℝ) (random_discount : ℝ) (final_price : ℝ) :
  initial_price = 230 →
  autumn_discount = 25 →
  loyalty_discount = 20 →
  random_discount = 50 →
  final_price = 69 →
  apply_discount (apply_discount (apply_discount initial_price autumn_discount) loyalty_discount) random_discount = final_price :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_discount_l78_7849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_norms_squared_l78_7886

-- We don't need to redefine midpoint as it's likely already defined in Mathlib
-- def midpoint (a b : ℝ × ℝ) : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem vector_sum_norms_squared (a b : ℝ × ℝ) :
  let m : ℝ × ℝ := (5, 3)
  (a.1 + b.1) / 2 = m.1 →
  (a.2 + b.2) / 2 = m.2 →
  a.1 * b.1 + a.2 * b.2 = -4 →
  (a.1^2 + a.2^2) + (b.1^2 + b.2^2) = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_norms_squared_l78_7886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_is_one_fourth_l78_7810

/-- Represents the duration of the meeting window in minutes -/
noncomputable def meetingWindow : ℝ := 60

/-- Represents the duration of the friend's stay in minutes -/
noncomputable def friendStayDuration : ℝ := 15

/-- Calculates the probability of Bob meeting his friend -/
noncomputable def meetingProbability : ℝ :=
  (friendStayDuration * meetingWindow - friendStayDuration^2 / 2) / meetingWindow^2

/-- Theorem stating that the probability of Bob meeting his friend is 1/4 -/
theorem meeting_probability_is_one_fourth :
  meetingProbability = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_is_one_fourth_l78_7810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l78_7829

theorem undefined_values_count : ∃! (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 + 4*x - 5) * (x - 4) = 0) ∧ 
  (∀ x ∉ S, (x^2 + 4*x - 5) * (x - 4) ≠ 0) ∧ 
  Finset.card S = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l78_7829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l78_7826

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * Real.pi / 2 - x) ^ 2 + Real.sin (x + Real.pi)

-- State the theorem
theorem f_range : 
  (∀ x : ℝ, -1 ≤ f x ∧ f x ≤ 5/4) ∧ 
  (∃ x : ℝ, f x = -1) ∧ 
  (∃ x : ℝ, f x = 5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l78_7826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l78_7844

noncomputable section

/-- The hyperbola equation -/
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- The eccentricity of a hyperbola -/
def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity_value (a b : ℝ) :
  (∃ A B : ℝ × ℝ, 
    hyperbola a b A.1 A.2 ∧ 
    hyperbola a b B.1 B.2 ∧
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧
    (A.1 > 0 ∧ A.2 > 0) ∧ 
    (B.1 > 0 ∧ B.2 < 0)) →
  (∃ F : ℝ × ℝ, 
    F.1 < 0 ∧ 
    F.2 = 0 ∧ 
    (∀ F' : ℝ × ℝ, F'.1 < 0 ∧ F'.2 = 0 → 
      ∃ A B : ℝ × ℝ, 
        hyperbola a b A.1 A.2 ∧ 
        hyperbola a b B.1 B.2 ∧
        ellipse A.1 A.2 ∧ 
        ellipse B.1 B.2 ∧
        (A.1 > 0 ∧ A.2 > 0) ∧ 
        (B.1 > 0 ∧ B.2 < 0) ∧
        Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) + 
        Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) + 
        Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥
        Real.sqrt ((A.1 - F'.1)^2 + (A.2 - F'.2)^2) + 
        Real.sqrt ((B.1 - F'.1)^2 + (B.2 - F'.2)^2) + 
        Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))) →
  hyperbola_eccentricity a b = Real.sqrt 13 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_value_l78_7844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_of_log_abs_through_origin_l78_7832

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

def tangent1 (x y : ℝ) : Prop := x - Real.exp 1 * y = 0
def tangent2 (x y : ℝ) : Prop := x + Real.exp 1 * y = 0

theorem tangent_lines_of_log_abs_through_origin :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    f 0 = 0 ∧
    y₁ = f x₁ ∧ y₂ = f x₂ ∧
    tangent1 0 0 ∧ tangent2 0 0 ∧
    tangent1 x₁ y₁ ∧ tangent2 x₂ y₂ ∧
    (tangent1 x₁ y₁ ↔ y₁ = (deriv f) x₁ * x₁) ∧
    (tangent2 x₂ y₂ ↔ y₂ = (deriv f) x₂ * x₂) :=
by
  sorry

#check tangent_lines_of_log_abs_through_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_of_log_abs_through_origin_l78_7832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sin_inequality_l78_7820

theorem tan_sin_inequality : Real.tan 3 < Real.sin 2 ∧ Real.sin 2 < Real.tan 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sin_inequality_l78_7820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_P_to_l_l78_7877

/-- The point P in the 2D plane -/
def P : ℝ × ℝ := (-2, 0)

/-- The line l in the 2D plane, parameterized by λ -/
def l (lambda : ℝ) : ℝ → ℝ → Prop :=
  fun x y => (1 + 3*lambda)*x + (1 + 2*lambda)*y = 2 + 5*lambda

/-- The maximum distance from point P to line l -/
noncomputable def max_distance : ℝ := Real.sqrt 10

/-- Theorem stating that the maximum distance from P to l is √10 -/
theorem max_distance_from_P_to_l :
  ∀ lambda : ℝ, ∀ x y : ℝ, l lambda x y →
  (∀ x' y' : ℝ, l lambda x' y' →
    (x' - P.1)^2 + (y' - P.2)^2 ≤ max_distance^2) ∧
  (∃ x' y' : ℝ, l lambda x' y' ∧
    (x' - P.1)^2 + (y' - P.2)^2 = max_distance^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_P_to_l_l78_7877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_parallel_lines_l78_7864

-- Define the points
variable (A B C D E O T : ℝ × ℝ)

-- Define the trapezoid ABCE
def is_trapezoid (A B C E : ℝ × ℝ) : Prop := sorry

-- Define that D is on AE
def point_on_segment (D A E : ℝ × ℝ) : Prop := sorry

-- Define equality of areas
def area_equal (ABCD CDE : Set (ℝ × ℝ)) : Prop := sorry

-- Define parallelogram
def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- Define intersection of diagonals
def diagonals_intersect_at (A B C D O : ℝ × ℝ) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

theorem trapezoid_parallel_lines 
  (h1 : is_trapezoid A B C E)
  (h2 : point_on_segment D A E)
  (h3 : area_equal {A, B, C, D} {C, D, E})
  (h4 : is_parallelogram A B C D)
  (h5 : diagonals_intersect_at A B C D O)
  (h6 : point_on_segment T D E)
  (h7 : parallel {O, T} {B, E}) :
  parallel {O, D} {C, T} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_parallel_lines_l78_7864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l78_7873

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2

noncomputable def g (x : ℝ) : ℝ := f (1/2) x * Real.exp x

theorem extremum_and_monotonicity :
  (∃ (a : ℝ), ∀ (x : ℝ), deriv (f a) x = 0 ↔ x = -4/3) →
  (∃ (a : ℝ), a = 1/2) ∧
  (∀ (x : ℝ), 
    (x < -4 → (deriv g x < 0)) ∧
    (-4 < x ∧ x < -1 → (deriv g x > 0)) ∧
    (-1 < x ∧ x < 0 → (deriv g x < 0)) ∧
    (0 < x → (deriv g x > 0))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l78_7873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_cost_l78_7853

/-- Represents the dimensions and costs of a rectangular box without a lid -/
structure Box where
  volume : ℝ
  height : ℝ
  bottom_cost : ℝ
  side_cost : ℝ

/-- Calculates the total cost of the box given its length -/
noncomputable def total_cost (b : Box) (length : ℝ) : ℝ :=
  b.bottom_cost * length * (b.volume / (b.height * length)) +
  2 * b.side_cost * b.height * (length + b.volume / (b.height * length))

/-- Theorem stating the minimum total cost of the box -/
theorem min_total_cost (b : Box) 
  (h_volume : b.volume = 48)
  (h_height : b.height = 3)
  (h_bottom_cost : b.bottom_cost = 15)
  (h_side_cost : b.side_cost = 12) :
  ∃ (length : ℝ), ∀ (x : ℝ), x > 0 → total_cost b length ≤ total_cost b x ∧ total_cost b length = 816 := by
  sorry

#check min_total_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_cost_l78_7853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l78_7817

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A = 3 * Real.pi / 4 ∧ t.b = Real.sqrt 2 * t.c

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : triangle_area t = 2) : 
  Real.cos t.B * Real.cos t.C = 3 * Real.sqrt 2 / 5 ∧ 
  t.a = 2 * Real.sqrt 5 ∧ t.b = 2 * Real.sqrt 2 ∧ t.c = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l78_7817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l78_7843

noncomputable section

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 5 * Real.cos θ, Real.sqrt 5 * Real.sin θ)

noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (1 - Real.sqrt 2 / 2 * t, -Real.sqrt 2 / 2 * t)

-- Define the domain for θ
def θ_domain (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ Real.pi / 2

-- Define the first quadrant
def first_quadrant (p : ℝ × ℝ) : Prop := 0 ≤ p.1 ∧ 0 ≤ p.2

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, first_quadrant p ∧
  (∃ θ, θ_domain θ ∧ C₁ θ = p) ∧
  (∃ t, C₂ t = p) ∧
  p = (2, 1) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l78_7843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l78_7801

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x ≥ 0, (x - 1) * f x ≥ m * x^2 - 1) ↔ m ≤ 1/2 := by
  sorry

-- Part 2
theorem part_two : 
  ∀ x > 0, f x > 4 * Real.log x + 8 - 8 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l78_7801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l78_7840

theorem function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h1 : ∀ x, (deriv (deriv f)) x > f x) 
  (h2 : x₁ < x₂) : 
  Real.exp x₁ * f x₂ > Real.exp x₂ * f x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l78_7840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l78_7893

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The focal length of an ellipse -/
noncomputable def Ellipse.focalLength (e : Ellipse) : ℝ :=
  Real.sqrt (e.a ^ 2 - e.b ^ 2)

/-- Theorem: For an ellipse with equation x^2/25 + y^2/m^2 = 1 (m > 0) and focal length 8,
    the value of m is either 3 or √41 -/
theorem ellipse_m_values (m : ℝ) (h_m_pos : 0 < m) :
  let e := Ellipse.mk 5 m ⟨by norm_num, h_m_pos⟩
  e.focalLength = 4 → m = 3 ∨ m = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l78_7893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_monotonic_increase_f_greater_g_f_plus_g_inequality_l78_7813

-- Define the functions
noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := -1/2 * x^2 + x
noncomputable def G (x : ℝ) := 2 * f x + g x

-- Theorem 1
theorem G_monotonic_increase :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → G x₁ < G x₂ := by sorry

-- Theorem 2
theorem f_greater_g :
  ∀ x, x > 0 → f (x + 1) > g x := by sorry

-- Theorem 3
theorem f_plus_g_inequality :
  ∀ k, k < 1 → ∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x + g x - 1/2 > k * (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_monotonic_increase_f_greater_g_f_plus_g_inequality_l78_7813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_conditions_l78_7871

theorem system_solution_conditions (n p : ℕ) (hn : n > 0) (hp : p > 1) :
  (∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p ^ z) ↔
  (∃ z : ℕ, z > 0 ∧ n % (p - 1) = (p ^ z) % (p - 1) ∧ n > p ^ z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_conditions_l78_7871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l78_7888

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x + Real.exp x

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (let (x₀, y₀) := point
     y₀ = f x₀ ∧
     m = (deriv f) x₀ ∧
     y₀ = m * x₀ + b) ∧
    m * point.1 - point.2 + b = 0 ∧
    m = 2 ∧ b = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l78_7888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_AC_approx_l78_7879

/-- A quadrilateral with specific side lengths -/
structure Quadrilateral where
  AB : ℝ
  DC : ℝ
  AD : ℝ
  h_AB : AB = 17
  h_DC : DC = 25
  h_AD : AD = 8

/-- The length of the diagonal AC in the quadrilateral -/
noncomputable def diagonal_AC (q : Quadrilateral) : ℝ :=
  Real.sqrt (q.AD^2 + q.DC^2 + q.AB^2 - q.AD * q.DC)

/-- Theorem stating that the diagonal AC is approximately 33.6 -/
theorem diagonal_AC_approx (q : Quadrilateral) :
  |diagonal_AC q - 33.6| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_AC_approx_l78_7879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l78_7827

open Real

-- Define the function as noncomputable due to its dependence on π
noncomputable def f (x : ℝ) := Real.sin (x + π)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc (π / 2) π,
  ∀ y ∈ Set.Icc (π / 2) π,
  x < y → f x < f y :=
by
  -- Proof is omitted using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l78_7827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_positive_l78_7848

theorem sin_sum_positive (x : Real) (h : 0 < x ∧ x < Real.pi) :
  Real.sin x + (1/2) * Real.sin (2*x) + (1/3) * Real.sin (3*x) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_positive_l78_7848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_relation_l78_7809

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - x + b

-- Define the solution set of the first inequality
def solution_set_f (a b : ℝ) : Set ℝ := {x | f a b x ≥ 0}

-- Define the quadratic function with swapped coefficients
def g (a b : ℝ) (x : ℝ) : ℝ := b * x^2 - x + a

-- Define the solution set of the second inequality
def solution_set_g (a b : ℝ) : Set ℝ := {x | g a b x ≤ 0}

-- Theorem statement
theorem solution_sets_relation (a b : ℝ) :
  solution_set_f a b = Set.Icc (-2) 1 → solution_set_g a b = Set.Icc (-1/2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_relation_l78_7809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_layoffs_eq_596_l78_7841

def initial_employees : ℕ := 1000

def layoff_percentages : List ℚ := [1/10, 12/100, 15/100, 1/5, 1/4]

def round_down (x : ℚ) : ℕ := Int.toNat (Int.floor x)

def layoff_round (employees : ℕ) (percentage : ℚ) : ℕ :=
  round_down (percentage * employees)

def remaining_employees (employees : ℕ) (laid_off : ℕ) : ℕ :=
  employees - laid_off

def total_layoffs (initial : ℕ) (percentages : List ℚ) : ℕ :=
  percentages.foldl
    (fun acc percentage =>
      let remaining := remaining_employees initial acc
      acc + layoff_round remaining percentage)
    0

theorem total_layoffs_eq_596 :
  total_layoffs initial_employees layoff_percentages = 596 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_layoffs_eq_596_l78_7841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l78_7816

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- Theorem: Conversion of given polar coordinates to rectangular coordinates -/
theorem polar_to_rectangular_conversion :
  let point1 : ℝ × ℝ := polar_to_rectangular 5 (3 * π / 4)
  let point2 : ℝ × ℝ := polar_to_rectangular 6 (5 * π / 3)
  (point1 = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2)) ∧
  (point2 = (3, -3 * Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l78_7816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_shot_probability_fifth_shot_probability_equals_247_432_l78_7838

/-- The probability that the fifth shot is taken by player A in a basketball shooting game -/
theorem fifth_shot_probability (
  prob_A prob_B : ℝ
) (h_prob_A : prob_A = 1/2)
  (h_prob_B : prob_B = 1/3) : ℝ := by
  let prob_A_miss := 1 - prob_A
  let prob_B_miss := 1 - prob_B
  
  -- Scenario 1: A makes the first four shots
  let scenario1 := prob_A^4

  -- Scenario 2: Among the first four shots, B shoots once and misses
  let scenario2 := 3 * prob_A^3 * prob_B_miss

  -- Scenario 3: Among the first four shots, B shoots twice
  let scenario3 := prob_A^2 * (prob_B^2 + 2 * prob_A_miss * prob_B * prob_B_miss)

  -- Scenario 4: Among the first four shots, B shoots three times
  let scenario4 := prob_A * prob_B^2 * prob_B_miss

  exact scenario1 + scenario2 + scenario3 + scenario4

/-- The probability that the fifth shot is taken by player A is equal to 247/432 -/
theorem fifth_shot_probability_equals_247_432 : 
  fifth_shot_probability (1/2) (1/3) rfl rfl = 247/432 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_shot_probability_fifth_shot_probability_equals_247_432_l78_7838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_theorem_l78_7819

theorem obtuse_angle_theorem (α : ℝ) (h1 : 90 < α ∧ α < 180) 
  (h2 : Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1) : 
  α = 140 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_theorem_l78_7819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_dimensions_l78_7892

-- Define the properties of the cone
def cone_volume : ℝ := 30
def lateral_surface_area_ratio : ℝ := 5

-- Define the relationships between dimensions
def radius (r : ℝ) : Prop := r = (45 / (Real.pi * Real.sqrt 6)) ^ (1/3)
def cone_height (h r : ℝ) : Prop := h = 2 * Real.sqrt 6 * r
def slant_height (l r : ℝ) : Prop := l = 5 * r

-- State the theorem
theorem cone_dimensions (r h l : ℝ) : 
  radius r ∧ cone_height h r ∧ slant_height l r → 
  (1/3 * Real.pi * r^2 * h = cone_volume) ∧
  (Real.pi * r * l = lateral_surface_area_ratio * Real.pi * r^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_dimensions_l78_7892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l78_7895

/-- A triangle with side lengths 7, 24, and 25 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 7
  hb : b = 24
  hc : c = 25

/-- The triangle is right-angled -/
def is_right_triangle (t : RightTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The inradius of the triangle -/
noncomputable def inradius (t : RightTriangle) : ℝ :=
  (t.a * t.b) / (2 * (t.a + t.b + t.c))

/-- The circumradius of the triangle -/
noncomputable def circumradius (t : RightTriangle) : ℝ :=
  t.c / 2

theorem right_triangle_properties (t : RightTriangle) :
  is_right_triangle t ∧ inradius t = 3 ∧ circumradius t = 12.5 := by
  sorry

#check right_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l78_7895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l78_7867

/-- The volume of a parallelepiped constructed on three vectors is the absolute value of their scalar triple product. -/
def volume (a b c : Fin 3 → ℝ) : ℝ := |Matrix.det ![a, b, c]|

/-- The three vectors that define the parallelepiped. -/
def a : Fin 3 → ℝ := ![1, 2, 3]
def b : Fin 3 → ℝ := ![0, 1, 1]
def c : Fin 3 → ℝ := ![2, 1, -1]

/-- The theorem stating that the volume of the parallelepiped constructed on vectors a, b, and c is 4. -/
theorem parallelepiped_volume : volume a b c = 4 := by
  -- The proof goes here
  sorry

#eval volume a b c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l78_7867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sphere_to_triangle_l78_7825

-- Define the sphere and triangle properties
noncomputable def sphere_volume : ℝ := 4 * Real.sqrt 3 * Real.pi
noncomputable def triangle_side_length : ℝ := 2 * Real.sqrt 2

-- Helper function (not to be proved, just for type-checking)
noncomputable def max_distance_from_sphere_surface_to_inscribed_triangle (V : ℝ) (s : ℝ) : ℝ :=
  sorry

-- Define the theorem
theorem max_distance_sphere_to_triangle (V : ℝ) (s : ℝ) 
  (hV : V = sphere_volume) (hs : s = triangle_side_length) :
  ∃ (d : ℝ), d = (4 * Real.sqrt 3) / 3 ∧ 
  d = max_distance_from_sphere_surface_to_inscribed_triangle V s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sphere_to_triangle_l78_7825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l78_7872

/-- Represents a geometric sequence with a common ratio of 2 -/
noncomputable def GeometricSequence (a : ℝ) : ℕ → ℝ := fun n => a * 2^(n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSum (a : ℝ) (n : ℕ) : ℝ := a * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_sum_eight (a : ℝ) :
  GeometricSum a 4 = 1 → GeometricSum a 8 = 17 := by
  sorry

#check geometric_sequence_sum_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l78_7872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unrolled_coins_value_l78_7845

/-- Calculate the value of unrolled coins --/
theorem unrolled_coins_value
  (sarah_quarters : ℕ)
  (sarah_dimes : ℕ)
  (mark_quarters : ℕ)
  (mark_dimes : ℕ)
  (quarters_per_roll : ℕ)
  (dimes_per_roll : ℕ)
  (h1 : sarah_quarters = 157)
  (h2 : sarah_dimes = 342)
  (h3 : mark_quarters = 211)
  (h4 : mark_dimes = 438)
  (h5 : quarters_per_roll = 25)
  (h6 : dimes_per_roll = 40) :
  (((sarah_quarters + mark_quarters) % quarters_per_roll : ℚ) * 25 +
   ((sarah_dimes + mark_dimes) % dimes_per_roll : ℚ) * 10) / 100 = 13/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unrolled_coins_value_l78_7845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l78_7878

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x - 3)

-- Define the interval [3, +∞)
def interval : Set ℝ := { x | x ≥ 3 }

-- Theorem statement
theorem f_monotone_increasing : 
  StrictMono (f ∘ (fun x ↦ (x : interval) : interval → ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l78_7878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_duration_l78_7880

/-- Calculates the time taken (in minutes) for a chase between a policeman and a criminal
    given their initial distance, speeds, and final distance. -/
noncomputable def chase_time (initial_distance : ℝ) (criminal_speed policeman_speed : ℝ) (final_distance : ℝ) : ℝ :=
  ((initial_distance - final_distance) / (policeman_speed - criminal_speed)) * 60

/-- Theorem stating that under the given conditions, the chase time is 5000 minutes. -/
theorem chase_duration :
  let initial_distance : ℝ := 180
  let criminal_speed : ℝ := 8
  let policeman_speed : ℝ := 9
  let final_distance : ℝ := 96.66666666666667
  chase_time initial_distance criminal_speed policeman_speed final_distance = 5000 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_duration_l78_7880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_sum_l78_7876

noncomputable def larger_circle_area : ℝ := 64 * Real.pi

def smaller_circle_diameter (r : ℝ) : Prop := r^2 * Real.pi = larger_circle_area ∧ r = r / 2

theorem shaded_area_sum :
  ∀ r : ℝ, smaller_circle_diameter r →
  (larger_circle_area / 2) + ((r/2)^2 * Real.pi / 2) = 40 * Real.pi :=
by
  intro r h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_sum_l78_7876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_beam_length_in_cube_l78_7854

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents a light beam path in the cube -/
structure LightBeamPath where
  cube : Cube
  startPoint : Point3D
  reflectionPoint : Point3D

/-- Calculate the total length of the light beam path -/
noncomputable def totalLightBeamLength (path : LightBeamPath) : ℝ :=
  sorry

/-- Main theorem: The total length of the light beam path in the given scenario -/
theorem light_beam_length_in_cube :
  ∀ (c : Cube) (start : Point3D) (reflect : Point3D),
    c.edgeLength = 14 →
    start.x = 0 ∧ start.y = 0 ∧ start.z = 0 →
    reflect.x = 14 ∧ reflect.y = 8 ∧ reflect.z = 3 →
    totalLightBeamLength { cube := c, startPoint := start, reflectionPoint := reflect } = 14 * Real.sqrt 285 := by
  sorry

#eval Float.sqrt 285

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_beam_length_in_cube_l78_7854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l78_7831

theorem complex_fraction_equality : 
  (3 - 4 * Complex.I) * (1 + Complex.I)^3 / (4 + 3 * Complex.I) = 2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l78_7831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l78_7807

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 5 = 10) :
  Real.log (a 3) + Real.log (a 4) = Real.log 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l78_7807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_triangle_l78_7808

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that when a = 5, b = 3, and c = 7, the largest internal angle is 2π/3 -/
theorem largest_angle_in_triangle (A B C : ℝ) (a b c : ℝ) :
  a = 5 → b = 3 → c = 7 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  a = b * Real.sin A / Real.sin B →
  b = c * Real.sin B / Real.sin C →
  c = a * Real.sin C / Real.sin A →
  max A (max B C) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_triangle_l78_7808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nathan_path_distance_l78_7833

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The path Nathan takes -/
def nathanPath : List Point :=
  [⟨0, 0⟩, ⟨0, -50⟩, ⟨30, -50⟩, ⟨30, -30⟩, ⟨20, -30⟩]

/-- The final position after northeast movement -/
noncomputable def finalPosition : Point :=
  ⟨20 + 30 * Real.sqrt 2 / 2, -30 + 30 * Real.sqrt 2 / 2⟩

theorem nathan_path_distance :
  distance (nathanPath.head!) finalPosition = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nathan_path_distance_l78_7833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_overlap_minimum_omega_l78_7834

/-- The minimum value of ω that satisfies the conditions of the tangent function overlap -/
theorem tan_overlap_minimum_omega : ∃ ω_min : ℝ, ω_min > 0 ∧
  (∀ ω : ℝ, ω > 0 →
    (∀ x : ℝ, Real.tan (ω * (x - π/6) + π/4) = Real.tan (ω * x + π/6)) →
    ω ≥ ω_min) ∧
  (∃ x : ℝ, Real.tan (ω_min * (x - π/6) + π/4) = Real.tan (ω_min * x + π/6)) ∧
  ω_min = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_overlap_minimum_omega_l78_7834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_angle_range_l78_7855

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := -4 * Real.exp x / ((Real.exp x + 1)^2)

-- Define the slant angle of the tangent line
noncomputable def slant_angle (x : ℝ) : ℝ := Real.arctan (f' x)

-- Theorem statement
theorem slant_angle_range :
  ∀ x : ℝ, 3 * Real.pi / 4 ≤ slant_angle x ∧ slant_angle x < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_angle_range_l78_7855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_side_minimum_l78_7818

theorem triangle_angle_and_side_minimum (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  Real.sqrt 3 * Real.cos A + a * Real.sin B = Real.sqrt 3 * c →
  a + 2 * c = 6 →
  B = π / 3 ∧ 
  ∀ (b' : ℝ), b' ^ 2 ≥ 27 / 7 ∧ 
  (b' ^ 2 = 27 / 7 → b = b') := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_side_minimum_l78_7818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ripe_bananas_weight_ripe_bananas_weight_example_l78_7894

theorem ripe_bananas_weight (bunches_of_eight : ℕ) (bunches_of_seven : ℕ) 
  (bananas_per_bunch_eight : ℕ) (bananas_per_bunch_seven : ℕ) 
  (banana_weight : ℕ) (ripe_fraction : ℚ) : ℕ :=
  let total_bananas := bunches_of_eight * bananas_per_bunch_eight + 
                       bunches_of_seven * bananas_per_bunch_seven
  let ripe_bananas := (ripe_fraction * total_bananas).floor.toNat
  ripe_bananas * banana_weight

theorem ripe_bananas_weight_example : 
  ripe_bananas_weight 6 5 8 7 100 (3/4) = 6200 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ripe_bananas_weight_ripe_bananas_weight_example_l78_7894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l78_7803

def f (n : ℕ) : ℚ := (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_lower_bound (n : ℕ) (h : n ≥ 2) :
  f (2^n) ≥ (n + 2 : ℚ) / 2 := by
  sorry

-- Additional hypotheses based on given conditions
axiom f_2 : f 2 = 3 / 2
axiom f_4 : f 4 > 2
axiom f_8 : f 8 > 5 / 2
axiom f_16 : f 16 > 3
axiom f_32 : f 32 > 7 / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l78_7803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_difference_is_25_l78_7836

noncomputable section

-- Define the points that the lines pass through
def l_point1 : ℝ × ℝ := (0, 5)
def l_point2 : ℝ × ℝ := (3, 0)
def m_point1 : ℝ × ℝ := (0, 2)
def m_point2 : ℝ × ℝ := (7, 0)

-- Define the slope of a line given two points
def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the y-intercept of a line given a point and slope
def y_intercept (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  p.2 - m * p.1

-- Define the x-coordinate where a line reaches y = 10
def x_at_y_10 (m b : ℝ) : ℝ :=
  (10 - b) / m

-- Theorem statement
theorem x_coordinate_difference_is_25 :
  let l_slope := line_slope l_point1 l_point2
  let m_slope := line_slope m_point1 m_point2
  let l_intercept := y_intercept l_point1 l_slope
  let m_intercept := y_intercept m_point1 m_slope
  let l_x := x_at_y_10 l_slope l_intercept
  let m_x := x_at_y_10 m_slope m_intercept
  |l_x - m_x| = 25 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_difference_is_25_l78_7836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_methods_l78_7898

/-- The number of ways to assign n students to k subjects, where each student chooses one subject. -/
def number_of_ways_to_assign (n k : ℕ) : ℕ := k^n

theorem enrollment_methods (n k : ℕ) : 
  n > 0 → k > 0 → number_of_ways_to_assign n k = k^n :=
by
  intros hn hk
  unfold number_of_ways_to_assign
  rfl

#eval number_of_ways_to_assign 4 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_methods_l78_7898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_and_fruit_seeds_in_sample_l78_7812

/-- Represents the number of varieties for each seed type -/
structure SeedVarieties where
  beans : ℕ
  vegetables : ℕ
  rice : ℕ
  fruits : ℕ

/-- Calculates the expected number of seeds in a stratified sample -/
def expectedSeedsInSample (varieties : SeedVarieties) (sampleSize : ℕ) (seedTypes : List (SeedVarieties → ℕ)) : ℚ :=
  let totalVarieties := varieties.beans + varieties.vegetables + varieties.rice + varieties.fruits
  let probability : ℚ := sampleSize / totalVarieties
  (seedTypes.map (λ f => f varieties)).sum * probability

theorem vegetable_and_fruit_seeds_in_sample
  (varieties : SeedVarieties)
  (h1 : varieties.beans = 40)
  (h2 : varieties.vegetables = 10)
  (h3 : varieties.rice = 30)
  (h4 : varieties.fruits = 20)
  (sampleSize : ℕ)
  (h5 : sampleSize = 20) :
  expectedSeedsInSample varieties sampleSize [SeedVarieties.vegetables, SeedVarieties.fruits] = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_and_fruit_seeds_in_sample_l78_7812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_for_five_pow_nine_sum_l78_7862

/-- The largest positive integer k for which 5^9 can be expressed as the sum of k consecutive positive integers is 1250. -/
theorem largest_k_for_five_pow_nine_sum : 
  (∃ (k : ℕ), k > 0 ∧ 
    (∃ (n : ℕ), (Finset.range k).sum (λ i => n + i + 1) = 5^9) ∧ 
    (∀ (m : ℕ), m > k → 
      ¬∃ (p : ℕ), (Finset.range m).sum (λ i => p + i + 1) = 5^9)) → 
  (∃ (n : ℕ), (Finset.range 1250).sum (λ i => n + i + 1) = 5^9) ∧
  (∀ (m : ℕ), m > 1250 → 
    ¬∃ (p : ℕ), (Finset.range m).sum (λ i => p + i + 1) = 5^9) :=
by sorry

#check largest_k_for_five_pow_nine_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_for_five_pow_nine_sum_l78_7862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_l78_7806

/-- Given a function f(x) = 3x^(a-2) - 2 that passes through the point (2, 4), prove that a = 3 -/
theorem function_passes_through_point (a : ℝ) : 
  (fun x : ℝ ↦ 3 * x^(a - 2) - 2) 2 = 4 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_l78_7806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_m_value_l78_7887

def z (m : ℝ) : ℂ := (2*m^2 - 3*m - 2) + (m^2 - 3*m + 2)*Complex.I

theorem pure_imaginary_m_value :
  ∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 → m = -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_m_value_l78_7887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l78_7881

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * Real.cos (2 * ω * x)

theorem omega_value (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x : ℝ, f ω (x + 2 * π / 3) = f ω x) 
                    (h3 : ∀ T : ℝ, T > 0 → T < 2 * π / 3 → ∃ x : ℝ, f ω (x + T) ≠ f ω x) : 
  ω = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l78_7881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_probabilities_l78_7852

/-- Represents a class of students -/
structure ClassInfo where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  has_male_president : Bool
  h_total : total_students = male_students + female_students
  h_president : has_male_president → male_students > 0

/-- The probability of an event in a finite sample space -/
def probability (event : ℕ) (total : ℕ) : ℚ :=
  ↑event / ↑total

/-- Theorem about probabilities in a class -/
theorem class_probabilities (c : ClassInfo) 
  (h_total : c.total_students = 40)
  (h_male : c.male_students = 25)
  (h_female : c.female_students = 15)
  (h_president : c.has_male_president = true) :
  probability 1 c.total_students = 1 / 40 ∧
  probability c.female_students c.total_students = 3 / 8 ∧
  probability 0 c.female_students = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_probabilities_l78_7852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l78_7897

/-- The quadrilateral region defined by the given inequalities -/
def QuadrilateralRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ 2 * p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The length of a side of the quadrilateral -/
noncomputable def SideLength (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

/-- The theorem stating that the longest side of the quadrilateral has length 4√2 -/
theorem longest_side_length : 
  ∃ (a b : ℝ × ℝ), a ∈ QuadrilateralRegion ∧ b ∈ QuadrilateralRegion ∧
    SideLength a b = 4 * Real.sqrt 2 ∧
    ∀ (c d : ℝ × ℝ), c ∈ QuadrilateralRegion → d ∈ QuadrilateralRegion →
      SideLength c d ≤ 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l78_7897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l78_7861

def valid_numbers : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ ∃ m k : ℕ, n = 2^m * 5^k}

def is_valid_selection (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ s.toSet ⊆ valid_numbers ∧ 20 ∈ s ∧
  (∃ k : ℕ, (s.toList.map Real.log).sum = k)

noncomputable def number_of_valid_selections : ℕ := sorry

theorem lottery_probability : 
  (number_of_valid_selections : ℚ) / (Nat.choose 14 5 : ℚ) = 10 / 3003 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l78_7861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_percentage_theorem_l78_7842

def total_employees : ℕ := 3600
def initial_clerical_ratio : ℚ := 1/3
def clerical_reduction_ratio : ℚ := 1/6

def remaining_clerical_percentage : ℚ :=
  let initial_clerical := total_employees * initial_clerical_ratio
  let clerical_reduction := initial_clerical * clerical_reduction_ratio
  let remaining_clerical := initial_clerical - clerical_reduction
  let remaining_total := total_employees - clerical_reduction
  (remaining_clerical / remaining_total) * 100

theorem clerical_percentage_theorem :
  ∃ (ε : ℚ), abs (remaining_clerical_percentage - 29.41) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clerical_percentage_theorem_l78_7842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_calculation_l78_7805

-- Define the rectangle as a structure
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ

-- State the theorem
theorem rectangle_width_calculation (r : Rectangle) 
  (h1 : r.length = 2) 
  (h2 : r.area = 8) : 
  r.width = 4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_calculation_l78_7805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonpositive_l78_7866

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem probability_f_nonpositive : 
  let a := -5
  let b := 5
  let roots := {x : ℝ | f x = 0}
  ∃ r₁ r₂ : ℝ, r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ < r₂ ∧ 
    (∀ x ∈ Set.Icc a b, f x ≤ 0 ↔ r₁ ≤ x ∧ x ≤ r₂) ∧
    (r₂ - r₁) / (b - a) = 3 / 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonpositive_l78_7866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_dried_grapes_is_twenty_percent_l78_7868

/-- Calculates the percentage of water in dried grapes -/
noncomputable def water_percentage_in_dried_grapes (fresh_water_percentage : ℝ) 
  (fresh_weight : ℝ) (dried_weight : ℝ) : ℝ :=
  let non_water_content := (1 - fresh_water_percentage) * fresh_weight
  let water_in_dried := dried_weight - non_water_content
  (water_in_dried / dried_weight) * 100

/-- Theorem: The percentage of water in dried grapes is 20% -/
theorem water_percentage_in_dried_grapes_is_twenty_percent :
  water_percentage_in_dried_grapes 0.9 30 3.75 = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval water_percentage_in_dried_grapes 0.9 30 3.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_dried_grapes_is_twenty_percent_l78_7868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l78_7822

open Real

/-- Given a triangle ABC and a point O inside it satisfying certain vector conditions,
    prove that the ratio of weighted areas of subtriangles to the area of ABC is 11/6. -/
theorem triangle_area_ratio (A B C O : ℝ × ℝ) : 
  let v (P Q : ℝ × ℝ) := (Q.1 - P.1, Q.2 - P.2)
  let S (P Q R : ℝ × ℝ) := abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2
  v O A + 2 • v O B + 3 • v O C = 3 • v A B + 2 • v B C + v C A →
  (S A O B + 2 * S B O C + 3 * S C O A) / S A B C = 11 / 6 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l78_7822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l78_7896

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the distance between two points on a line that intersects a sphere -/
noncomputable def intersectionDistance (start : Point3D) (endpoint : Point3D) (sphere : Sphere) : ℝ :=
  sorry

/-- The theorem stating the distance between intersection points -/
theorem intersection_distance_theorem :
  let start : Point3D := ⟨3, 0, -1⟩
  let endpoint : Point3D := ⟨1, -4, -5⟩
  let sphere : Sphere := ⟨⟨1, 1, 1⟩, 2⟩
  intersectionDistance start endpoint sphere = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l78_7896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_coprime_proper_divisor_l78_7835

theorem existence_of_coprime_proper_divisor (n : ℕ) (h : n > 3) :
  ∃ d : ℕ, d ∣ (2^(Nat.totient n) - 1) ∧ d ≠ 1 ∧ d ≠ (2^(Nat.totient n) - 1) ∧ Nat.Coprime d n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_coprime_proper_divisor_l78_7835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_colony_limit_l78_7860

/-- Represents the number of days it takes for a colony to reach the habitat's limit -/
def days_to_limit : ℕ := 24

/-- The size of a colony after a given number of days -/
def colony_size (days : ℕ) : ℕ := 2^days

/-- The habitat's limit size -/
def habitat_limit : ℕ := colony_size days_to_limit

/-- Two colonies starting simultaneously take 24 days to reach the habitat's limit -/
axiom two_colonies_limit : colony_size 23 = habitat_limit / 2

theorem single_colony_limit : days_to_limit = 24 := by
  -- The proof goes here
  sorry

#eval days_to_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_colony_limit_l78_7860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l78_7815

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_from_focus_to_line :
  distance_to_line right_focus 1 2 (-8) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l78_7815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_increase_after_price_drop_and_sales_boost_l78_7856

theorem revenue_increase_after_price_drop_and_sales_boost :
  ∀ (original_price quantity : ℝ),
  original_price > 0 → quantity > 0 →
  let new_price := original_price * (1 - 0.2)
  let new_quantity := quantity * (1 + 0.7)
  let original_revenue := original_price * quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_increase_after_price_drop_and_sales_boost_l78_7856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l78_7804

/-- IsTriangle a b c means a, b, c form a valid triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    prove that B = π/6 and if b = 1, then -1 < √3*a - c ≤ 2 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) 
    (h_triangle : IsTriangle a b c)
    (h_angles : A + B + C = Real.pi)
    (h_sides : (2*a - Real.sqrt 3*c)^2 = 4*b^2 - c^2) :
  B = Real.pi/6 ∧ (b = 1 → -1 < Real.sqrt 3*a - c ∧ Real.sqrt 3*a - c ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l78_7804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_average_age_l78_7850

theorem family_average_age 
  (initial_average_age : ℝ) 
  (years_passed : ℕ) 
  (child_age : ℕ) 
  (family_members : ℕ) :
  initial_average_age = 23 →
  years_passed = 5 →
  child_age = 1 →
  family_members = 3 →
  (2 * initial_average_age + 2 * (years_passed : ℝ) + (child_age : ℝ)) / (family_members : ℝ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_average_age_l78_7850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_from_line_l78_7874

/-- The line that intersects the coordinate axes -/
def line (x y : ℝ) : Prop := 3 * x - 2 * y + 12 = 0

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intercept : ℝ := -4

/-- The y-coordinate of the intersection point with the y-axis -/
def y_intercept : ℝ := 6

/-- The center of the circle -/
def center : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
noncomputable def radius : ℝ := Real.sqrt 13

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 13

theorem circle_from_line :
  ∀ x y : ℝ,
  line x y →
  (x = x_intercept ∧ y = 0) ∨ (x = 0 ∧ y = y_intercept) →
  circle_equation x y := by
  sorry

#check circle_from_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_from_line_l78_7874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_intersection_ratios_l78_7891

/-- Given a triangle ABC with median AA₁, and point C₁ on AB such that AC₁ : C₁B = 1 : 2,
    if M is the intersection point of AA₁ and CC₁, then AM : MA₁ = 1 : 1 and CM : MC₁ = 3 : 1 -/
theorem triangle_median_intersection_ratios 
  (A B C A₁ C₁ M : EuclideanSpace ℝ (Fin 2)) :
  -- AA₁ is a median
  (midpoint ℝ B C = A₁) →
  -- C₁ lies on AB
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C₁ = (1 - t) • A + t • B) →
  -- AC₁ : C₁B = 1 : 2
  (dist A C₁ / dist C₁ B = 1 / 2) →
  -- M is the intersection point of AA₁ and CC₁
  (∃ s t : ℝ, M = (1 - s) • A + s • A₁ ∧ M = (1 - t) • C + t • C₁) →
  -- AM : MA₁ = 1 : 1
  (dist A M / dist M A₁ = 1) ∧
  -- CM : MC₁ = 3 : 1
  (dist C M / dist M C₁ = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_intersection_ratios_l78_7891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l78_7847

-- Define the signum function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x = 0 then 0
  else -1

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sgn x - 2 * x

-- Theorem statement
theorem zeros_of_f : {x : ℝ | f x = 0} = {-1/2, 0, 1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_l78_7847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l78_7823

theorem complex_magnitude_problem :
  Complex.abs ((2 - Complex.I)^2 / Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l78_7823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_continuous_l78_7884

theorem power_function_continuous (n : ℕ) :
  Continuous (fun x : ℝ ↦ (x : ℝ)^n) := by
  apply continuous_pow

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_continuous_l78_7884
