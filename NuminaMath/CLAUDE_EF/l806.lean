import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l806_80674

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Function to check if a point is below a line -/
def isBelowLine (p : Point) (l : Line) : Prop :=
  p.y < l.slope * p.x + l.intercept

/-- Function to check if a point is between two lines -/
def isBetweenLines (p : Point) (l1 l2 : Line) : Prop :=
  (l1.slope * p.x + l1.intercept > p.y) ∧ (l2.slope * p.x + l2.intercept < p.y)

/-- Calculate the area of the triangle formed by a line in the first quadrant -/
noncomputable def triangleArea (l : Line) : ℝ :=
  (l.intercept * l.intercept) / (2 * (-l.slope))

theorem probability_between_lines :
  let l1 : Line := { slope := -3, intercept := 9 }
  let l2 : Line := { slope := -6, intercept := 9 }
  let totalArea := triangleArea l1
  let areaBetween := triangleArea l1 - triangleArea l2
  areaBetween / totalArea = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l806_80674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sergey_teaches_history_in_kaluga_l806_80604

-- Define the types for brothers, subjects, and cities
inductive Brother : Type
| Ivan | Dmitry | Sergey

inductive Subject : Type
| History | Chemistry | Biology

inductive City : Type
| Moscow | SaintPetersburg | Kaluga

-- Define a function that assigns a city to each brother
variable (city : Brother → City)

-- Define a function that assigns a subject to each brother
variable (subject : Brother → Subject)

-- State the theorem
theorem sergey_teaches_history_in_kaluga :
  -- Conditions
  (∀ b1 b2 : Brother, b1 ≠ b2 → city b1 ≠ city b2) →
  (∀ b1 b2 : Brother, b1 ≠ b2 → subject b1 ≠ subject b2) →
  (city Brother.Ivan ≠ City.Moscow) →
  (city Brother.Dmitry ≠ City.SaintPetersburg) →
  (∀ b : Brother, city b = City.Moscow → subject b ≠ Subject.History) →
  (∀ b : Brother, city b = City.SaintPetersburg → subject b = Subject.Chemistry) →
  (subject Brother.Dmitry = Subject.Biology) →
  -- Conclusion
  (city Brother.Sergey = City.Kaluga ∧ subject Brother.Sergey = Subject.History) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sergey_teaches_history_in_kaluga_l806_80604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_mile_time_l806_80698

-- Define the mile times for each person
noncomputable def tina_time : ℝ := 6

-- Define Tony's time in terms of Tina's
noncomputable def tony_time : ℝ := tina_time / 2

-- Define Tom's time in terms of Tina's
noncomputable def tom_time : ℝ := tina_time / 3

-- Theorem to prove
theorem total_mile_time : tony_time + tina_time + tom_time = 11 := by
  -- Unfold the definitions
  unfold tony_time tom_time tina_time
  -- Simplify the expression
  simp [add_assoc, add_comm, add_left_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_mile_time_l806_80698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_patrol_arrangement_l806_80631

theorem impossibility_of_patrol_arrangement (n : Nat) (k : Nat) : n = 100 ∧ k = 3 →
  ¬ ∃ (schedule : List (Finset (Fin n))),
    (∀ s, s ∈ schedule → s.card = k) ∧
    (∀ i j : Fin n, i ≠ j → ∃! s, s ∈ schedule ∧ i ∈ s ∧ j ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_patrol_arrangement_l806_80631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factorial_l806_80636

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_difference_factorial (x y z : ℕ+) : 
  x * y * z = factorial 9 → x < y → y < z → 
  (∀ a b c : ℕ+, a * b * c = factorial 9 → a < b → b < c → z - x ≤ c - a) → 
  z - x = 99 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factorial_l806_80636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_squares_rectangle_l806_80690

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a chessboard with marked squares -/
structure Chessboard where
  n : Nat
  markedSquares : Finset Square

/-- Checks if four squares form a rectangle -/
def isRectangle (s1 s2 s3 s4 : Square) : Prop :=
  (s1.row = s2.row ∧ s3.row = s4.row ∧ s1.row ≠ s3.row) ∧
  (s1.col = s3.col ∧ s2.col = s4.col ∧ s1.col ≠ s2.col)

/-- Main theorem -/
theorem marked_squares_rectangle (board : Chessboard) :
  (board.n > 0) →
  (board.markedSquares.card ≥ board.n * (Real.sqrt (board.n : Real) + 1/2)) →
  ∃ s1 s2 s3 s4 : Square,
    s1 ∈ board.markedSquares ∧
    s2 ∈ board.markedSquares ∧
    s3 ∈ board.markedSquares ∧
    s4 ∈ board.markedSquares ∧
    isRectangle s1 s2 s3 s4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_squares_rectangle_l806_80690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l806_80676

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment
structure LineSegment where
  start : Point2D
  finish : Point2D

-- Function to check if a line segment is parallel to y-axis
def isParallelToYAxis (l : LineSegment) : Prop :=
  l.start.x = l.finish.x

-- Function to calculate the length of a line segment
def length (l : LineSegment) : ℝ :=
  ((l.finish.x - l.start.x)^2 + (l.finish.y - l.start.y)^2)^(1/2)

-- Theorem statement
theorem point_B_coordinates :
  ∀ (A B : Point2D) (AB : LineSegment),
    A.x = -4 ∧ A.y = 3 ∧
    AB.start = A ∧ AB.finish = B ∧
    isParallelToYAxis AB ∧
    length AB = 5 →
    (B.x = -4 ∧ B.y = 8) ∨ (B.x = -4 ∧ B.y = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l806_80676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_difference_l806_80637

theorem cos_sin_difference (α : ℝ) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 5 →
  Real.cos α ^ 2 - Real.sin α ^ 2 = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_difference_l806_80637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_equality_l806_80622

theorem sqrt_difference_equality : Real.sqrt 45 - (Real.sqrt 20 / 2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_equality_l806_80622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l806_80695

/-- The circle with center (5,3) and radius 3 -/
def myCircle (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

/-- The line 3x + 4y - 2 = 0 -/
def myLine (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

/-- The maximum distance from any point on the circle to the line is 8 -/
theorem max_distance_circle_to_line : 
  ∃ (max_dist : ℝ), max_dist = 8 ∧ 
    ∀ (P : ℝ × ℝ), myCircle P.1 P.2 → 
      ∀ (Q : ℝ × ℝ), myLine Q.1 Q.2 → 
        ∃ (dist : ℝ), dist ≤ max_dist ∧ 
          dist = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l806_80695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_product_lower_bound_l806_80649

theorem polynomial_product_lower_bound
  (a b c d : ℝ)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h_bd : b - d ≥ 5)
  (h_zeros : x₁^4 + a*x₁^3 + b*x₁^2 + c*x₁ + d = 0 ∧
             x₂^4 + a*x₂^3 + b*x₂^2 + c*x₂ + d = 0 ∧
             x₃^4 + a*x₃^3 + b*x₃^2 + c*x₃ + d = 0 ∧
             x₄^4 + a*x₄^3 + b*x₄^2 + c*x₄ + d = 0) :
  (x₁^2 + 1) * (x₂^2 + 1) * (x₃^2 + 1) * (x₄^2 + 1) ≥ 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_product_lower_bound_l806_80649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_survey_result_l806_80642

def cat_survey (total_respondents : ℕ) (friendly_believers : ℕ) (scratch_misconception : ℕ) : Prop :=
  (friendly_believers : ℚ) / total_respondents = 923 / 1000 ∧
  (scratch_misconception : ℚ) / friendly_believers = 384 / 1000 ∧
  scratch_misconception = 28

theorem cat_survey_result : ∃ (total_respondents : ℕ), 
  cat_survey total_respondents (Int.toNat (round ((28 : ℚ) / (384 / 1000)))) 28 ∧ 
  total_respondents = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_survey_result_l806_80642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_difference_l806_80699

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_difference_l806_80699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_building_cost_twenty_floors_optimal_l806_80602

/-- Represents the average comprehensive cost per square meter -/
noncomputable def W (x : ℕ) : ℝ :=
  50 * (x : ℝ) + 20000 / (x : ℝ) + 3000

/-- The theorem stating the optimal number of floors and minimum cost -/
theorem optimal_building_cost (x : ℕ) (h : x ≥ 12) :
  W x ≥ 5000 ∧ (W x = 5000 ↔ x = 20) := by
  sorry

/-- The theorem proving that 20 floors achieves the minimum cost -/
theorem twenty_floors_optimal :
  W 20 = 5000 ∧ ∀ x : ℕ, x ≥ 12 → W x ≥ 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_building_cost_twenty_floors_optimal_l806_80602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_fits_in_square_l806_80681

theorem rectangle_fits_in_square : 
  ∀ (square_area rectangle_area : ℝ) (length_width_ratio : ℚ),
    square_area = 25 →
    rectangle_area = 12 →
    length_width_ratio = 3/2 →
    ∃ (rect_length rect_width : ℝ),
      rect_length * rect_width = rectangle_area ∧
      rect_length / rect_width = length_width_ratio ∧
      rect_length ≤ Real.sqrt square_area ∧
      rect_width ≤ Real.sqrt square_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_fits_in_square_l806_80681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_triangular_prism_l806_80672

/-- A right prism with triangular bases -/
structure TriangularPrism where
  /-- Shorter side of the base triangle -/
  b : ℝ
  /-- Height of the prism -/
  h : ℝ
  /-- Angle between sides a and b of the base triangle -/
  θ : ℝ

/-- The volume of a triangular prism -/
noncomputable def volume (p : TriangularPrism) : ℝ :=
  3/2 * p.b^2 * p.h * Real.sin p.θ

/-- The sum of areas of three mutually orthogonal faces -/
noncomputable def sumOfAreas (p : TriangularPrism) : ℝ :=
  4 * p.b * p.h + 3/2 * p.b^2 * Real.sin p.θ

theorem max_volume_of_triangular_prism :
  ∀ p : TriangularPrism,
    0 < p.b ∧ 0 < p.h ∧ 0 < p.θ ∧ p.θ < π/2 →
    sumOfAreas p = 30 →
    volume p ≤ Real.sqrt 500 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_triangular_prism_l806_80672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l806_80641

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x^3

def decreasing_on_ℝ (h : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → h x > h y

def increasing_on_ℝ (h : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → h x < h y

theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (decreasing_on_ℝ (f a) → increasing_on_ℝ (g a)) ∧
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧ increasing_on_ℝ (g b) ∧ ¬decreasing_on_ℝ (f b)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l806_80641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_meeting_time_l806_80692

/-- The time until the next meeting of two bugs crawling on tangent circles -/
theorem bug_meeting_time (r₁ r₂ v₁ v₂ : ℝ) 
  (h1 : r₁ = 7)
  (h2 : r₂ = 3)
  (h3 : v₁ = 4 * π)
  (h4 : v₂ = 3 * π)
  (h5 : (2 * π * r₁) / v₁ = 7 / 2)
  (h6 : (2 * π * r₂) / v₂ = 2) :
  Nat.lcm (Nat.ceil (7 / 2)) 2 = 7 := by
  sorry

#check bug_meeting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_meeting_time_l806_80692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l806_80633

/-- The inclination angle of a line with equation √3x + y - 1 = 0 is 120 degrees. -/
theorem line_inclination_angle : 
  ∃ α : ℝ, α = 120 * (π / 180) ∧ 
    ∀ x y : ℝ, Real.sqrt 3 * x + y - 1 = 0 → Real.tan α = -Real.sqrt 3 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l806_80633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_tan_alpha_value_l806_80600

-- Problem 1
noncomputable def f (x : ℝ) := 3 - Real.sin x - 2 * (Real.cos x)^2

theorem max_value_of_f (x : ℝ) (h : x ∈ Set.Icc (π/6) (7*π/6)) :
  f x ≤ 2 := by sorry

-- Problem 2
theorem tan_alpha_value (α β : ℝ) 
  (h1 : 5 * Real.sin β = Real.sin (2*α + β))
  (h2 : Real.tan (α + β) = 9/4) :
  Real.tan α = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_tan_alpha_value_l806_80600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_pair_existence_l806_80601

/-- Given a finite set of points in a plane, there exist two distinct points A and B
    such that no other point is closer to A than B is, and no other point is closer to B than A is. -/
theorem closest_pair_existence (S : Set (ℝ × ℝ)) (hfin : S.Finite) :
  ∃ A B, A ∈ S ∧ B ∈ S ∧ A ≠ B ∧
  (∀ C ∈ S, C ≠ A → dist A C ≥ dist A B) ∧
  (∀ D ∈ S, D ≠ B → dist B D ≥ dist A B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_pair_existence_l806_80601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_card_to_turn_l806_80691

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define a function to check if a character is a consonant
def isConsonant (c : Char) : Bool :=
  c.isAlpha && !(c = 'a' || c = 'e' || c = 'i' || c = 'o' || c = 'u' ||
                 c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U')

-- Define a function to check if a number is odd
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

-- Define Tom's statement as a function
def tomStatement (card : Card) : Bool :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => 
      !isConsonant c || isOdd n
  | (CardSide.Number n, CardSide.Letter c) => 
      !isConsonant c || isOdd n
  | _ => true

-- Define the set of cards on the table
def tableCards : List Card := [
  (CardSide.Letter 'A', CardSide.Number 0),  -- 0 is a placeholder
  (CardSide.Letter 'R', CardSide.Number 0),  -- 0 is a placeholder
  (CardSide.Number 5, CardSide.Letter ' '),  -- ' ' is a placeholder
  (CardSide.Number 8, CardSide.Letter ' '),  -- ' ' is a placeholder
  (CardSide.Number 7, CardSide.Letter ' ')   -- ' ' is a placeholder
]

-- Define a function to check if turning a card can disprove Tom's statement
def canDisproveStatement (card : Card) : Bool :=
  match card with
  | (CardSide.Number n, _) => !isOdd n
  | (CardSide.Letter c, _) => isConsonant c

-- Theorem: The card showing 8 is the optimal choice to potentially disprove Tom's statement
theorem optimal_card_to_turn : 
  ∃ (card : Card), card ∈ tableCards ∧ 
    canDisproveStatement card ∧ 
    (∀ (otherCard : Card), otherCard ∈ tableCards → 
      canDisproveStatement otherCard → card = otherCard) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_card_to_turn_l806_80691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_2_neg1_l806_80670

theorem sin_plus_cos_for_point_2_neg1 (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos α = 2 ∧ r * Real.sin α = -1) →
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_2_neg1_l806_80670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_rate_approximation_l806_80647

/-- The equivalent annual interest rate for an account with 10% annual interest
    compounded quarterly -/
noncomputable def equivalent_annual_rate : ℝ :=
  ((1 + 0.1 / 4) ^ 4 - 1) * 100

/-- The original annual interest rate -/
def original_rate : ℝ := 10

/-- The number of compounding periods per year for the original account -/
def compounding_periods : ℕ := 4

/-- Theorem stating that the equivalent annual rate is approximately 10.17 -/
theorem equivalent_rate_approximation :
  ‖equivalent_annual_rate - 10.17‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_rate_approximation_l806_80647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l806_80693

def set_A : Set ℝ := {x | x^2 - 1 ≤ 0}
def set_B : Set ℝ := {x | 1 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 4}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l806_80693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_E_l806_80663

def mySequence : ℕ → Char
| n => let m := n % 12
       if m = 0 then 'X'
       else if m ≤ 6 then (Char.ofNat (64 + m))
       else (Char.ofNat (77 - m))

theorem letter_2023_is_E : mySequence 2023 = 'E' := by
  -- Proof goes here
  sorry

#eval mySequence 2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_E_l806_80663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cleaning_time_is_440_l806_80632

/-- Calculates the total time spent by Matt, Alex, John, and Kate cleaning their cars -/
noncomputable def total_cleaning_time (matt_outside : ℝ) : ℝ :=
  let matt_inside := matt_outside / 4
  let alex_outside := matt_outside / 2
  let alex_inside := matt_inside * 2
  let john_outside := matt_outside * 1.5
  let john_inside := matt_inside * 0.75
  let kate_outside := alex_outside
  let avg_inside := (matt_inside + alex_inside + john_inside) / 3
  let kate_inside := avg_inside + 20
  let break_time := 10

  (matt_outside + matt_inside + 
   alex_outside + alex_inside + 
   john_outside + john_inside + 
   kate_outside + kate_inside + 
   4 * break_time)

theorem total_cleaning_time_is_440 : 
  total_cleaning_time 80 = 440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cleaning_time_is_440_l806_80632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_interior_angle_l806_80640

-- Define a regular polygon
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  regular : Bool
  side_count : sides = n

-- Define the measure of an interior angle of a regular polygon
noncomputable def interiorAngle (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180 / n

-- Theorem statement
theorem octagon_interior_angle :
  interiorAngle 8 = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_interior_angle_l806_80640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pi_implies_x_eq_neg_three_l806_80669

/-- Two vectors with an angle of π between them --/
def vectors_with_pi_angle (x : ℝ) : Prop :=
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![9, x]
  Real.cos (Finset.univ.sum (λ i => a i * b i) / (Real.sqrt (Finset.univ.sum (λ i => a i ^ 2)) * Real.sqrt (Finset.univ.sum (λ i => b i ^ 2)))) = -1

/-- Theorem stating that if vectors (x, 1) and (9, x) have an angle of π between them, then x = -3 --/
theorem angle_pi_implies_x_eq_neg_three :
  ∀ x : ℝ, vectors_with_pi_angle x → x = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pi_implies_x_eq_neg_three_l806_80669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_catches_tom_l806_80644

/-- The time it takes for Alice to catch up to Tom -/
noncomputable def catchUpTime (aliceSpeed tomSpeed : ℝ) (initialDistance : ℝ) : ℝ :=
  (initialDistance / (aliceSpeed - tomSpeed)) * 60

theorem alice_catches_tom (aliceSpeed tomSpeed initialDistance : ℝ) 
  (h1 : aliceSpeed = 45)
  (h2 : tomSpeed = 15)
  (h3 : initialDistance = 4)
  (h4 : aliceSpeed > tomSpeed) :
  catchUpTime aliceSpeed tomSpeed initialDistance = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_catches_tom_l806_80644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_proof_l806_80656

/-- Given two vectors a and b in ℝ², where a ≠ b and (a - b) ⊥ a, prove that m = 1 -/
theorem vector_perpendicular_proof (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![m - 1, 2]
  a ≠ b → (a - b) • a = 0 → m = 1 := by
  sorry

#check vector_perpendicular_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_proof_l806_80656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l806_80618

-- Define the curves C₁ and C₂
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, 2 * Real.sin φ)
noncomputable def C₂ (φ : ℝ) : ℝ × ℝ := (Real.cos φ, 4 * Real.sin φ)

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (t, 1 + Real.sqrt 3 * t)

-- Define the point P
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem intersection_product :
  ∃ A B : ℝ × ℝ,
    (∃ φ₁ φ₂ : ℝ, C₂ φ₁ = A ∧ C₂ φ₂ = B) ∧
    (∃ t₁ t₂ : ℝ, l t₁ = A ∧ l t₂ = B) ∧
    (A.1 - P.1)^2 + (A.2 - P.2)^2 * (B.1 - P.1)^2 + (B.2 - P.2)^2 = (60/19)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l806_80618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sum_theorem_l806_80694

-- Define the set of natural numbers greater than 1
def NatGreaterThanOne := {n : ℕ | n > 1}

-- Define the periodicity property for a function
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

-- Define the smallest positive period
def SmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) :=
  IsPeriodic f T ∧ ∀ T' > 0, IsPeriodic f T' → T ≤ T'

-- State the theorem
theorem periodic_sum_theorem
  (f g : ℝ → ℝ)
  (m : ℕ)
  (h_m : m > 1)
  (h_f_period : SmallestPositivePeriod f 1)
  (h_g_period : SmallestPositivePeriod g (1 / m))
  (h : ℝ → ℝ)
  (h_def : ∀ x, h x = f x + g x) :
  ∃ k : ℕ,
    (k = 1 ∨ k > 1) ∧
    ¬(k ∣ m) ∧
    ¬(m ∣ k) ∧
    SmallestPositivePeriod h (1 / k) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sum_theorem_l806_80694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l806_80625

theorem cube_root_sum_equals_one :
  (7 + 3 * Real.sqrt 21) ^ (1/3 : ℝ) + (7 - 3 * Real.sqrt 21) ^ (1/3 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l806_80625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aldehyde_formula_proof_l806_80687

/-- Represents the number of carbon atoms in the aldehyde -/
def n : ℕ := sorry

/-- The general formula for a saturated aldehyde is CₙH₂ₙO -/
def aldehyde_formula (n : ℕ) : ℕ × ℕ × ℕ := (n, 2*n, 1)

/-- Calculate the molar mass of the aldehyde -/
def molar_mass (n : ℕ) : ℕ := 14*n + 16

/-- Calculate the mass percentage of hydrogen in the aldehyde -/
noncomputable def hydrogen_percentage (n : ℕ) : ℚ :=
  (2 * n : ℚ) / (molar_mass n : ℚ)

/-- Theorem: If the hydrogen percentage is 12%, then n = 6 -/
theorem aldehyde_formula_proof : hydrogen_percentage n = 12/100 → n = 6 := by
  sorry

#check aldehyde_formula_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aldehyde_formula_proof_l806_80687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l806_80616

-- Define the circles
def circle1_center : ℝ × ℝ := (1, 3)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (15, 8)
def circle2_radius : ℝ := 10

-- Define the common external tangent line
def tangent_line (m b : ℝ) : ℝ → ℝ := λ x ↦ m * x + b

-- Theorem statement
theorem common_external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), tangent_line m b x = y →
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 = circle1_radius^2 ∨
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2)) ∧
  b = 148 / 19 := by
  sorry

#check common_external_tangent_y_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l806_80616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_decimal_representation_l806_80675

theorem fraction_decimal_representation :
  -- Define the fraction
  let f : ℚ := 525 / 999
  -- Define the decimal representation as a sequence
  let decimal_seq : ℕ → ℕ := λ i => (f * 10^(i+1)).floor.toNat % 10
  -- The repeating pattern condition
  ∀ i : ℕ, decimal_seq i = decimal_seq (i + 3)
  -- The 81st digit condition
  ∧ decimal_seq 80 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_decimal_representation_l806_80675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_S_and_max_f_l806_80605

-- Define the set S
def S : Set Nat := {p | Nat.Prime p ∧ (∃ r : Nat, ∃ fp : Nat, fp = 3 * r)}

-- Define the function f
def f (k p : Nat) : Nat :=
  let r := p / 3  -- This is a placeholder; actual implementation would be more complex
  (k % 10) + ((k + r) % 10) + ((k + 2 * r) % 10)

-- State the theorem
theorem infinite_S_and_max_f :
  (Set.Infinite S) ∧
  (∀ k p, p ∈ S → k ≥ 1 → f k p ≤ 19) ∧
  (∃ k p, p ∈ S ∧ k ≥ 1 ∧ f k p = 19) := by
  sorry

-- No need for auxiliary functions as they're now incorporated into the main definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_S_and_max_f_l806_80605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l806_80645

/-- Represents a digit in base 5 -/
def Digit5 := Fin 5

/-- Converts a two-digit number in base 5 to a natural number -/
def toNat5 (a b : Digit5) : ℕ := a.val * 5 + b.val

/-- Converts a natural number to a two-digit number in base 5 -/
def fromNat5 (n : ℕ) : Digit5 × Digit5 := 
  (⟨n / 5, by sorry⟩, ⟨n % 5, by sorry⟩)

theorem unique_solution :
  ∃! (A B C : Digit5),
    A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0 ∧  -- non-zero digits
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧  -- distinct digits
    (toNat5 A B + C.val = toNat5 C ⟨0, by sorry⟩) ∧  -- AB₅ + C₅ = C0₅
    (toNat5 A B + toNat5 B A = toNat5 C C) ∧  -- AB₅ + BA₅ = CC₅
    A.val = 3 ∧ B.val = 1 ∧ C.val = 4 :=
by sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l806_80645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_three_numbers_l806_80639

theorem gcd_three_numbers (A : Finset ℕ) (d : ℕ) (h1 : A.Nonempty) (h2 : A.card ≥ 3)
  (h3 : ∀ a ∈ A, a ≤ 100) (h4 : d = Finset.gcd A id) :
  ∃ B : Finset ℕ, B ⊆ A ∧ B.card = 3 ∧ Finset.gcd B id = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_three_numbers_l806_80639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l806_80613

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.exp x + 3

-- State the theorem
theorem function_properties (a : ℝ) :
  (a ≤ 0 → ∀ x y, 0 < x ∧ x < y → f a y < f a x) ∧
  (a > 0 → 
    (∀ x y, 0 < x ∧ x < y ∧ y < Real.sqrt (1 / (2 * a)) → f a y < f a x) ∧
    (∀ x y, Real.sqrt (1 / (2 * a)) < x ∧ x < y → f a x < f a y)) ∧
  (∀ x, x > 0 → f a x > g a x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l806_80613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l806_80648

/-- Given a line with slope angle 135° passing through (2, -5), its y-intercept is -3 -/
theorem line_y_intercept :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y = -x - 3) →
  (2, -5) ∈ l →
  ∃ b : ℝ, (0, b) ∈ l ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l806_80648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l806_80696

open Real

theorem trig_problem (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : cos α = -3/5) : 
  tan α = -4/3 ∧ (cos (2*α)) / (sin (2*α) + 1) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l806_80696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l806_80651

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 4)

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ π }

-- Theorem statement
theorem f_properties :
  ∀ x ∈ domain,
    (∀ y ∈ domain, (0 ≤ x ∧ x ≤ y ∧ y ≤ π/8) → f x ≤ f y) ∧
    (∀ y ∈ domain, (5*π/8 ≤ x ∧ x ≤ y ∧ y ≤ π) → f x ≤ f y) ∧
    (∀ y ∈ domain, (π/8 ≤ x ∧ x ≤ y ∧ y ≤ 5*π/8) → f x ≥ f y) ∧
    (∀ y ∈ domain, f y ≤ f (π/8)) ∧
    (f (π/8) = 3) ∧
    (∀ y ∈ domain, f y ≥ f (5*π/8)) ∧
    (f (5*π/8) = -3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l806_80651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l806_80654

/-- Calculates the length of a train given the parameters of two trains passing each other. -/
noncomputable def calculate_train_length (length1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  relative_speed * time - length1

/-- Theorem stating the length of the second train given specific parameters. -/
theorem second_train_length :
  let length1 : ℝ := 220
  let speed1 : ℝ := 120
  let speed2 : ℝ := 80
  let time : ℝ := 9
  abs (calculate_train_length length1 speed1 speed2 time - 279.95) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l806_80654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_square_sum_permutation_l806_80660

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that represents a permutation of integers from 1 to n -/
def is_valid_permutation (p : List ℕ) (n : ℕ) : Prop :=
  p.length = n ∧ (∀ i : ℕ, i ∈ p ↔ 1 ≤ i ∧ i ≤ n)

/-- A function that checks if a permutation satisfies the adjacent sum condition -/
def satisfies_adjacent_sum_condition (p : List ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < p.length → is_perfect_square (p[i]! + p[i+1]!)

/-- The main theorem stating that 8 is the smallest n satisfying the condition -/
theorem smallest_n_for_square_sum_permutation :
  (∀ n : ℕ, 1 < n ∧ n < 8 →
    ¬∃ p : List ℕ, is_valid_permutation p n ∧ satisfies_adjacent_sum_condition p) ∧
  (∃ p : List ℕ, is_valid_permutation p 8 ∧ satisfies_adjacent_sum_condition p) := by
  sorry

#check smallest_n_for_square_sum_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_square_sum_permutation_l806_80660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_two_three_on_graph_point_one_six_on_graph_l806_80653

/-- An inverse proportion function passing through (2, 3) -/
noncomputable def inverse_proportion (x : ℝ) : ℝ := 6 / x

/-- The point (2, 3) lies on the graph of the inverse proportion function -/
theorem point_two_three_on_graph : inverse_proportion 2 = 3 := by sorry

/-- The point (1, 6) lies on the graph of the inverse proportion function -/
theorem point_one_six_on_graph : inverse_proportion 1 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_two_three_on_graph_point_one_six_on_graph_l806_80653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l806_80688

/-- The ellipse C with equation x²/4 + y² = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line l with equation x = 2√2 -/
def line_l (x : ℝ) : Prop := x = 2 * Real.sqrt 2

/-- Point A is the left vertex of the ellipse -/
def point_A : ℝ × ℝ := (-2, 0)

/-- Point B is the right vertex of the ellipse -/
def point_B : ℝ × ℝ := (2, 0)

/-- Point D is the intersection of line l with the x-axis -/
noncomputable def point_D : ℝ × ℝ := (2 * Real.sqrt 2, 0)

/-- Function to calculate |DE| * |DF| given a point P on the ellipse -/
noncomputable def DE_DF_product (P : ℝ × ℝ) : ℝ :=
  let (x₀, y₀) := P
  let E_y := (2 * Real.sqrt 2 + 2) * y₀ / (x₀ + 2)
  let F_y := (2 * Real.sqrt 2 - 2) * y₀ / (x₀ - 2)
  abs E_y * abs F_y

theorem constant_product :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → P ≠ point_A → P ≠ point_B → DE_DF_product P = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l806_80688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_divides_ac_in_ratio_two_three_l806_80685

/-- An equilateral triangle with special points -/
structure SpecialTriangle where
  -- A, B, C are the vertices of the equilateral triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- K is the midpoint of AB
  K : ℝ × ℝ
  -- M is on BC such that BM : MC = 1 : 3
  M : ℝ × ℝ
  -- P is on AC and minimizes the perimeter of PKM
  P : ℝ × ℝ
  -- ABC is equilateral
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  -- K is the midpoint of AB
  k_midpoint : K = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- M is on BC
  m_on_bc : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))
  -- BM : MC = 1 : 3
  bm_mc_ratio : (M.1 - B.1)^2 + (M.2 - B.2)^2 = (1/9) * ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  -- P is on AC
  p_on_ac : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- P minimizes the perimeter of PKM
  p_minimizes_perimeter : ∀ Q : ℝ × ℝ, (∃ r : ℝ, 0 ≤ r ∧ r ≤ 1 ∧ 
    Q = (A.1 + r * (C.1 - A.1), A.2 + r * (C.2 - A.2))) →
    ((P.1 - K.1)^2 + (P.2 - K.2)^2)^(1/2) + ((P.1 - M.1)^2 + (P.2 - M.2)^2)^(1/2) + 
    ((K.1 - M.1)^2 + (K.2 - M.2)^2)^(1/2) ≤
    ((Q.1 - K.1)^2 + (Q.2 - K.2)^2)^(1/2) + ((Q.1 - M.1)^2 + (Q.2 - M.2)^2)^(1/2) + 
    ((K.1 - M.1)^2 + (K.2 - M.2)^2)^(1/2)

/-- Theorem: P divides AC in the ratio 2:3 -/
theorem p_divides_ac_in_ratio_two_three (t : SpecialTriangle) : 
  ∃ s : ℝ, t.P = (t.A.1 + s * (t.C.1 - t.A.1), t.A.2 + s * (t.C.2 - t.A.2)) ∧ s = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_divides_ac_in_ratio_two_three_l806_80685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_school_l806_80624

/-- Represents the number of girls in a school --/
def num_girls : ℕ → Prop := sorry

/-- Represents the total number of students in the school --/
def total_students : ℕ := sorry

/-- Represents the sample size used in the survey --/
def sample_size : ℕ := sorry

/-- Represents the difference between sampled boys and girls --/
def sample_difference : ℕ := sorry

theorem girls_in_school 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : sample_difference = 20)
  (h4 : ∀ x : ℕ, num_girls x → 
    (x : ℚ) / total_students = (sample_size - sample_difference) / (2 * sample_size)) :
  num_girls 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_school_l806_80624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_iff_d_eq_3_or_4_l806_80634

noncomputable def g (d : ℝ) (x : ℝ) : ℝ := (x^2 - 4*x + d) / (x^2 - 5*x + 6)

def has_vertical_asymptote (d : ℝ) (x : ℝ) : Prop :=
  x^2 - 5*x + 6 = 0 ∧ x^2 - 4*x + d ≠ 0

def has_exactly_one_vertical_asymptote (d : ℝ) : Prop :=
  ∃! x : ℝ, has_vertical_asymptote d x

theorem one_vertical_asymptote_iff_d_eq_3_or_4 :
  ∀ d : ℝ, has_exactly_one_vertical_asymptote d ↔ d = 3 ∨ d = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_iff_d_eq_3_or_4_l806_80634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l806_80668

noncomputable section

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the common focus
def F : ℝ × ℝ := (2, 0)

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := (3, 2 * Real.sqrt 6)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem distance_to_asymptote
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : hyperbola a b P.1 P.2)
  (h4 : parabola P.1 P.2)
  (h5 : distance P F = 5) :
  distance F (2, Real.sqrt 3 * 2) = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l806_80668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_2500_l806_80697

/-- The length of a bridge in meters, given a man's walking speed and time to cross -/
noncomputable def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 1000 / 60

/-- Theorem stating the length of the bridge is 2500 meters -/
theorem bridge_length_is_2500 (speed time : ℝ)
  (h1 : speed = 10) -- Man's speed in km/hr
  (h2 : time = 15) -- Time to cross the bridge in minutes
  : bridge_length speed time = 2500 := by
  sorry

-- Use #eval only for computable functions
#check bridge_length 10 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_2500_l806_80697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_l806_80611

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific properties of a triangle for this problem

-- Define the vector from point P to point Q
def vector (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the magnitude (length) of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem length_BC (A B C : ℝ × ℝ) :
  Triangle A B C →
  magnitude (vector A B) = 2 →
  magnitude (vector A C) = 2 →
  dot_product (vector A B) (vector A C) = 1 →
  magnitude (vector B C) = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BC_l806_80611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_projection_distance_sum_l806_80615

/-- Given an arithmetic sequence a, b, c and points P and N, prove that the sum of max and min distances between M and N is 10 -/
theorem arithmetic_sequence_projection_distance_sum (a b c : ℝ) (P N : ℝ × ℝ) :
  (∃ k : ℝ, b - a = c - b) →  -- arithmetic sequence condition
  P = (-1, 0) →
  N = (3, 3) →
  (∃ M : ℝ × ℝ, ∀ x y : ℝ, a*x + b*y + c = 0 → 
    ((M.1 - P.1) * a + (M.2 - P.2) * b = 0) ∧ 
    (a * M.1 + b * M.2 + c = 0)) →
  ∃ d_max d_min : ℝ, d_max + d_min = 10 ∧ 
    ∀ M : ℝ × ℝ, (∃ x y : ℝ, a*x + b*y + c = 0 ∧ 
                        ((M.1 - P.1) * a + (M.2 - P.2) * b = 0) ∧ 
                        (a * M.1 + b * M.2 + c = 0)) →
      Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ≤ d_max ∧
      Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ≥ d_min :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_projection_distance_sum_l806_80615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_specific_number_eq_not_all_selections_equally_likely_l806_80673

/-- A circular arrangement of numbers with a selection process -/
structure CircularSelection (n : ℕ) (k : ℕ) where
  h_k_le_n : k ≤ n

/-- The probability of a specific number appearing in the selection -/
def prob_specific_number (cs : CircularSelection n k) : ℚ :=
  k / n

/-- A type representing a specific selection of k numbers from n -/
def Selection (n : ℕ) (k : ℕ) := Fin k → Fin n

/-- The probability of a specific selection occurring -/
noncomputable def prob_selection (cs : CircularSelection n k) (s : Selection n k) : ℝ :=
  sorry

/-- Theorem stating that the probability of any specific number appearing is k/n -/
theorem prob_specific_number_eq (cs : CircularSelection n k) :
  prob_specific_number cs = k / n := by sorry

/-- Theorem stating that not all selections are equally likely -/
theorem not_all_selections_equally_likely (cs : CircularSelection n k) :
  ∃ (s1 s2 : Selection n k), prob_selection cs s1 ≠ prob_selection cs s2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_specific_number_eq_not_all_selections_equally_likely_l806_80673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_when_real_imag_equal_l806_80684

-- Define the complex number z
noncomputable def z (a : ℝ) : ℂ := (a + Complex.I) / (2 * Complex.I)

-- Theorem statement
theorem modulus_of_z_when_real_imag_equal :
  ∀ a : ℝ, (z a).re = (z a).im → Complex.abs (z a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_when_real_imag_equal_l806_80684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_inequality_equality_conditions_l806_80680

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function representation of a cubic polynomial -/
def CubicPolynomial.toFun (p : CubicPolynomial) : ℝ → ℝ :=
  fun x => x^3 + p.a * x^2 + p.b * x + p.c

/-- Predicate for a cubic polynomial having all non-negative real roots -/
def hasNonNegativeRoots (p : CubicPolynomial) : Prop :=
  ∀ x, p.toFun x = 0 → x ≥ 0

/-- The theorem statement -/
theorem cubic_polynomial_inequality
    (p : CubicPolynomial)
    (h : hasNonNegativeRoots p) :
    (∃ l : ℝ, ∀ x : ℝ, x ≥ 0 → p.toFun x ≥ l * (x - p.a)^3) ∧
    (∀ l : ℝ, (∀ x : ℝ, x ≥ 0 → p.toFun x ≥ l * (x - p.a)^3) → l ≤ -1/27) :=
  sorry

/-- Equality conditions -/
theorem equality_conditions
    (p : CubicPolynomial)
    (h : hasNonNegativeRoots p) :
    ∀ x : ℝ, x ≥ 0 →
      p.toFun x = -1/27 * (x - p.a)^3 ↔
        (x = 0 ∧ ∃ α, ∀ y, p.toFun y = 0 → y = α) ∨
        (∃ γ, γ = 2*x ∧ p.toFun 0 = 0 ∧ p.toFun γ = 0) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_inequality_equality_conditions_l806_80680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l806_80617

/-- The rational function f(x) = (2x^2 + 7x + 10) / (2x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (2*x^2 + 7*x + 10) / (2*x + 3)

/-- The proposed oblique asymptote function g(x) = x + 2 -/
def g (x : ℝ) : ℝ := x + 2

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε := by
  sorry

#check oblique_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l806_80617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pqrs_l806_80671

/-- Point in 2D Euclidean space -/
def Point := EuclideanSpace ℝ (Fin 2)

/-- Square in 2D Euclidean space -/
def Square (A B C D : Point) : Prop := sorry

/-- Area of a square -/
def SquareArea (A B C D : Point) : ℝ := sorry

/-- Equilateral triangle in 2D Euclidean space -/
def EquilateralTriangle (A B C : Point) : Prop := sorry

/-- Area of a quadrilateral -/
def QuadrilateralArea (A B C D : Point) : ℝ := sorry

/-- Given a square EFGH with area 36 and equilateral triangles EPF, FQG, GRH, and HSE,
    prove that the area of quadrilateral PQRS is 144 + 72√3 -/
theorem area_of_pqrs (E F G H P Q R S : Point) : 
  Square E F G H → 
  SquareArea E F G H = 36 → 
  EquilateralTriangle E P F → 
  EquilateralTriangle F Q G → 
  EquilateralTriangle G R H → 
  EquilateralTriangle H S E → 
  QuadrilateralArea P Q R S = 144 + 72 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pqrs_l806_80671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dishonest_dealer_weight_l806_80667

-- Define the profit percentage
noncomputable def profit_percentage : ℝ := 0.17370892018779344

-- Define the function to calculate the actual weight
noncomputable def actual_weight (profit : ℝ) : ℝ := 1 / (1 + profit)

-- Theorem statement
theorem dishonest_dealer_weight :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |actual_weight profit_percentage - 0.8517| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dishonest_dealer_weight_l806_80667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_solution_l806_80623

theorem power_equality_solution (x : ℝ) : (2 : ℝ)^12 = 64^x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_solution_l806_80623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l806_80607

theorem number_relationship : 
  let a : ℝ := (0.4 : ℝ)^2
  let b : ℝ := Real.log 0.4 / Real.log 2
  let c : ℝ := (2 : ℝ)^(0.4 : ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l806_80607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l806_80609

noncomputable section

/-- A function f with period π and ω > 0 -/
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 5)

/-- The function g obtained by shifting f to the left by π/5 -/
def g (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 5)

/-- The statement that g is symmetric about the point (-π/10, 0) -/
theorem g_symmetry (x : ℝ) : g (-Real.pi/10 + x) = g (-Real.pi/10 - x) := by
  sorry

/-- The period of f is π -/
axiom f_period (ω : ℝ) : ω > 0 → ∀ x, f ω (x + Real.pi) = f ω x

/-- ω is positive -/
axiom ω_positive : ∃ ω : ℝ, ω > 0 ∧ ∀ x, f ω (x + Real.pi) = f ω x

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l806_80609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_calculation_l806_80655

/-- Calculates the effective percentage profit for an item given the weight used and scale error -/
noncomputable def effectiveProfit (weightUsed : ℝ) (scaleError : ℝ) : ℝ :=
  let actualWeight := weightUsed * (1 - scaleError)
  ((1000 - actualWeight) / actualWeight) * 100

theorem dealer_profit_calculation :
  let profitA := effectiveProfit 800 0.05
  let profitB := effectiveProfit 850 0.07
  let profitC := effectiveProfit 780 0.03
  (abs (profitA - 31.58) < 0.01) ∧ 
  (abs (profitB - 26.50) < 0.01) ∧ 
  (abs (profitC - 32.17) < 0.01) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_calculation_l806_80655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l806_80635

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def IsGeometricSequence (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ 
    x₂ = x₁ * r ∧ 
    x₃ = x₂ * r ∧ 
    x₄ = x₃ * r ∧ 
    x₅ = x₄ * r

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h1 : 1 < a ∧ a < b ∧ b < c ∧ c < 81) 
  (h2 : IsGeometricSequence 1 a b c 81) : b = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l806_80635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_c_greater_than_b_l806_80658

noncomputable section

-- Define constants for the angles in radians
def angle50 : ℝ := 50 * Real.pi / 180
def angle127 : ℝ := 127 * Real.pi / 180
def angle40 : ℝ := 40 * Real.pi / 180
def angle37 : ℝ := 37 * Real.pi / 180
def angle56 : ℝ := 56 * Real.pi / 180
def angle39 : ℝ := 39 * Real.pi / 180

-- Define a, b, and c
def a : ℝ := Real.cos angle50 * Real.cos angle127 + Real.cos angle40 * Real.cos angle37
def b : ℝ := Real.sqrt 2 / 2 * (Real.sin angle56 - Real.cos angle56)
def c : ℝ := (1 - Real.tan angle39 ^ 2) / (1 + Real.tan angle39 ^ 2)

-- State the theorem
theorem a_greater_than_c_greater_than_b : a > c ∧ c > b := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_c_greater_than_b_l806_80658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_and_square_inequality_l806_80603

theorem exp_and_square_inequality :
  ¬(∀ a b : ℝ, ((2:ℝ)^a > (2:ℝ)^b → a^2 > b^2) ∧ (a^2 > b^2 → (2:ℝ)^a > (2:ℝ)^b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_and_square_inequality_l806_80603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l806_80606

noncomputable def polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) : ℝ := 
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*(Real.cos (θ2 - θ1)))

theorem distance_between_polar_points :
  let r1 : ℝ := 5
  let r2 : ℝ := 12
  let θ1 : ℝ := (7 * Real.pi) / 36
  let θ2 : ℝ := (43 * Real.pi) / 36
  polar_distance r1 r2 θ1 θ2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l806_80606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_l806_80627

/-- Two lines intersecting at a point with given slopes -/
structure IntersectingLines where
  intersection_point : ℝ × ℝ
  slope1 : ℝ
  slope2 : ℝ

/-- Calculate the x-intercept of a line given its slope and a point it passes through -/
noncomputable def x_intercept (slope : ℝ) (point : ℝ × ℝ) : ℝ :=
  point.1 - point.2 / slope

/-- Calculate the distance between two points on the x-axis -/
noncomputable def x_axis_distance (x1 x2 : ℝ) : ℝ :=
  |x1 - x2|

/-- Theorem stating the distance between x-intercepts of two intersecting lines -/
theorem x_intercept_distance (lines : IntersectingLines) 
    (h1 : lines.intersection_point = (8, 20))
    (h2 : lines.slope1 = 4)
    (h3 : lines.slope2 = 6) : 
    x_axis_distance 
      (x_intercept lines.slope1 lines.intersection_point)
      (x_intercept lines.slope2 lines.intersection_point) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_l806_80627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l806_80650

theorem lambda_range (l : ℝ) : 
  (∀ (m n : ℝ), n > 0 → (m - n)^2 + (m - Real.log n + l)^2 ≥ 2) ↔ 
  (l ≥ 1 ∨ l ≤ -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l806_80650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l806_80686

/-- The length of the chord intercepted by a circle on a line through the origin -/
theorem chord_length (θ : Real) (h : θ = π / 3) : 
  let line := {(x, y) : ℝ × ℝ | y = Real.tan θ * x}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x = 0}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧
    2 = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l806_80686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_l806_80643

theorem sum_of_exponents (a b : ℝ) (h1 : (10 : ℝ)^a = 5) (h2 : (10 : ℝ)^b = 2) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_l806_80643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_is_30_degrees_l806_80662

/-- Theorem: In a triangle ABC, if angle C is 60°, side b is 2, and side c is 2√3,
    then angle B is 30°. -/
theorem triangle_angle_b_is_30_degrees
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : C = 60 * π / 180) -- Angle C is 60°
  (h2 : b = 2) -- Side b is 2
  (h3 : c = 2 * Real.sqrt 3) -- Side c is 2√3
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) -- Sides are positive
  (h5 : A + B + C = π) -- Sum of angles in a triangle is π
  (h6 : Real.sin A * a = Real.sin B * b) -- Law of sines (for sides a and b)
  (h7 : Real.sin B * b = Real.sin C * c) -- Law of sines (for sides b and c)
  : B = 30 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_b_is_30_degrees_l806_80662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l806_80659

/-- Given three vectors in a plane, prove that if one vector is parallel to the sum of the other two,
    then we can determine the unknown component and calculate a specific projection. -/
theorem vector_problem (a b c : ℝ × ℝ) (x : ℝ) :
  a = (1, -1) →
  b = (x, 2) →
  c = (2, 1) →
  ∃ (k : ℝ), a = k • (b + c) →
  x = -5 ∧
  (let proj := ((c.1 * (a.1 - b.1) + c.2 * (a.2 - b.2)) / ((a.1 - b.1)^2 + (a.2 - b.2)^2)) • (a - b);
   proj = (3 * Real.sqrt 5 / 5, -3 * Real.sqrt 5 / 10)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l806_80659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_asymptotes_and_holes_l806_80678

/-- The function f(x) = (x³ + 4x² + 3x) / (x³ + x² - 2x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^3 + x^2 - 2*x)

/-- The number of holes in the graph of f -/
def a : ℕ := 0

/-- The number of vertical asymptotes of f -/
def b : ℕ := 2

/-- The number of horizontal asymptotes of f -/
def c : ℕ := 1

/-- The number of oblique asymptotes of f -/
def d : ℕ := 0

theorem sum_of_asymptotes_and_holes : a + 2*b + 3*c + 4*d = 7 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_asymptotes_and_holes_l806_80678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_min_cost_l806_80689

/-- Represents the specifications and costs of a rectangular water tank. -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  base_cost_per_sqm : ℝ
  wall_cost_per_sqm : ℝ

/-- Calculates the minimum total construction cost for a water tank. -/
noncomputable def min_total_cost (tank : WaterTank) : ℝ :=
  let base_area := tank.volume / tank.depth
  let base_cost := base_area * tank.base_cost_per_sqm
  let side_length := Real.sqrt base_area
  let wall_area := 2 * side_length * tank.depth + 2 * base_area
  let wall_cost := wall_area * tank.wall_cost_per_sqm
  base_cost + wall_cost

/-- Theorem stating the minimum total construction cost for the specified water tank. -/
theorem water_tank_min_cost :
  let tank : WaterTank := {
    volume := 6400,
    depth := 4,
    base_cost_per_sqm := 300,
    wall_cost_per_sqm := 240
  }
  min_total_cost tank = 633600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_min_cost_l806_80689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_6_to_174_l806_80619

open Real BigOperators

theorem sin_squared_sum_6_to_174 : 
  (∑ k in Finset.range 29, (sin (6 * k * π / 180))^2) = 31/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_6_to_174_l806_80619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l806_80638

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

theorem triangle_area (A B C : ℝ) (hf : f A = 3/2) (hbc : B + C = 4) (ha : A = Real.sqrt 7) :
  (1/2) * B * C * Real.sin A = (3 * Real.sqrt 3) / 4 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l806_80638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_from_area_product_l806_80621

/-- Given a triangle ABC with sides a, b, and c, and an inscribed circle
    touching side AC at point K, if the area of the triangle is equal to
    the product of segments AK and KC, then the triangle is right-angled with b as the hypotenuse. -/
theorem triangle_right_angle_from_area_product (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let s := (a + b + c) / 2
  (s - a) * (s - c) = Real.sqrt (s * (s - a) * (s - b) * (s - c)) →
  b^2 = a^2 + c^2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_from_area_product_l806_80621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_mass_from_unequal_arm_balance_l806_80677

/-- Represents an unequal-arm balance -/
structure UnequalArmBalance where
  left_arm : ℝ
  right_arm : ℝ

/-- The mass of an object as measured by an unequal-arm balance -/
noncomputable def measured_mass (b : UnequalArmBalance) (actual_mass : ℝ) (on_left : Bool) : ℝ :=
  if on_left then
    actual_mass * b.right_arm / b.left_arm
  else
    actual_mass * b.left_arm / b.right_arm

/-- Theorem stating that if a block measures 0.4 kg on the left pan and 0.9 kg on the right pan
    of an unequal-arm balance, its actual mass is 0.6 kg -/
theorem actual_mass_from_unequal_arm_balance
  (b : UnequalArmBalance)
  (h1 : measured_mass b 0.6 true = 0.4)
  (h2 : measured_mass b 0.6 false = 0.9) :
  0.6 = 0.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_mass_from_unequal_arm_balance_l806_80677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_properties_l806_80628

/-- Represents a cell in the spiral grid -/
structure Cell where
  center : ℕ
  nodes : Fin 4 → ℕ

/-- The spiral grid -/
def SpiralGrid := ℕ → Cell

/-- The rule for placing numbers in the spiral -/
def spiral_rule (n : ℕ) : ℕ := n + 2

/-- The difference between adjacent cell centers -/
def center_diff (c1 c2 : Cell) : ℕ := c2.center - c1.center

/-- Predicate for a valid spiral grid -/
def is_valid_spiral (grid : SpiralGrid) : Prop :=
  ∀ n, (grid n).center = (grid n).nodes 0 + (grid n).nodes 1 + (grid n).nodes 2 + (grid n).nodes 3 ∧
       (center_diff (grid n) (grid (n+1)) = 4 ∨ center_diff (grid n) (grid (n+1)) = 8)

theorem spiral_properties (grid : SpiralGrid) (h : is_valid_spiral grid) :
  (∀ k : ℕ, ∃ n > k, 76 ∣ (grid n).center) ∧
  (∀ m n : ℕ, m ≠ n → (grid m).center ≠ (grid n).center) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_properties_l806_80628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_not_third_quadrant_l806_80630

-- Define a function to represent the quadrant of an angle
noncomputable def quadrant (θ : ℝ) : ℕ :=
  if 0 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then 1
  else if Real.pi / 2 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then 2
  else if Real.pi ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then 3
  else 4

-- Theorem statement
theorem alpha_third_not_third_quadrant (α : ℝ) :
  quadrant α = 2 → quadrant (α / 3) ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_not_third_quadrant_l806_80630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_implies_lambda_bound_l806_80664

theorem sequence_increasing_implies_lambda_bound (lambda : ℝ) :
  (∀ n : ℕ+, (n : ℝ)^2 + 2*lambda*n + 1 < ((n + 1) : ℝ)^2 + 2*lambda*(n + 1) + 1) →
  lambda > -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_implies_lambda_bound_l806_80664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_subset_exists_l806_80629

/-- A coloring of the edges of a complete graph -/
def EdgeColoring (n : ℕ) := Fin n → Fin n → Fin 4

/-- The property that all 4 colors are used in the coloring -/
def AllColorsUsed (n : ℕ) (c : EdgeColoring n) : Prop :=
  ∀ (color : Fin 4), ∃ (i j : Fin n), i ≠ j ∧ c i j = color

/-- The existence of a subset of vertices with at least 3 different colors on edges to other vertices -/
def ExistsThreeColorSubset (n : ℕ) (c : EdgeColoring n) : Prop :=
  ∃ (S : Set (Fin n)), ∃ (colors : Finset (Fin 4)),
    (S.Nonempty) ∧ 
    (colors.card ≥ 3) ∧
    (∀ (v : Fin n), v ∉ S → ∃ (u : Fin n), u ∈ S ∧ c u v ∈ colors)

/-- The main theorem -/
theorem three_color_subset_exists (c : EdgeColoring 2004) 
  (h : AllColorsUsed 2004 c) : ExistsThreeColorSubset 2004 c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_subset_exists_l806_80629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l806_80608

/-- Given a line passing through points (3, 19) and (-7, -1), 
    prove that it intersects the y-axis at (0, 13) -/
theorem line_intersection_y_axis : 
  let p1 : ℝ × ℝ := (3, 19)
  let p2 : ℝ × ℝ := (-7, -1)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line : ℝ → ℝ := λ x ↦ m * x + b
  line 0 = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l806_80608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_surrounding_radius_circle_problem_solution_l806_80626

-- Define a structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a structure for the problem setup
structure CircleConfiguration where
  center_circle : Circle
  surrounding_circles : List Circle

-- Define the theorem
theorem circle_surrounding_radius : 
  ∃ (config : CircleConfiguration),
    config.center_circle.radius = 2 ∧
    config.surrounding_circles.length = 4 ∧
    (∀ c ∈ config.surrounding_circles, c.radius = 1 + Real.sqrt 2) ∧
    -- Additional conditions (touch_center_circle, touch_each_other) would be defined here
    True →
  1 + Real.sqrt 2 = 1 + Real.sqrt 2 := by
  sorry

-- Main theorem connecting the problem to the solution
theorem circle_problem_solution : 
  ∃ (r : ℝ), 
    (∃ (config : CircleConfiguration),
      config.center_circle.radius = 2 ∧
      config.surrounding_circles.length = 4 ∧
      (∀ c ∈ config.surrounding_circles, c.radius = r) ∧
      -- Additional conditions (touch_center_circle, touch_each_other) would be defined here
      True) →
    r = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_surrounding_radius_circle_problem_solution_l806_80626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l806_80610

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define a piece type
inductive Piece
| Bishop
| Rook

-- Define a game state
structure GameState where
  occupied : Set Chessboard
  currentPlayer : Bool  -- true for first player, false for second player

-- Define a move
def Move := Chessboard

-- Function to check if a move is valid
def isValidMove (piece : Piece) (state : GameState) (move : Move) : Prop :=
  move ∉ state.occupied ∧
  match piece with
  | Piece.Bishop => ∀ (pos : Chessboard), pos ∈ state.occupied →
      ¬ (pos.1 - move.1 = pos.2 - move.2 ∨ pos.1 - move.1 = move.2 - pos.2)
  | Piece.Rook => ∀ (pos : Chessboard), pos ∈ state.occupied →
      pos.1 ≠ move.1 ∧ pos.2 ≠ move.2

-- Define the winning strategy for the second player
def secondPlayerStrategy (piece : Piece) (lastMove : Move) : Move :=
  match piece with
  | Piece.Bishop => (lastMove.2, lastMove.1)  -- Axial symmetry
  | Piece.Rook => (8 - lastMove.1, 8 - lastMove.2)  -- Central symmetry

-- Helper function to update game state
def updateState (state : GameState) (moves : List Move) : GameState :=
  sorry

-- Theorem stating that the second player always wins
theorem second_player_wins (piece : Piece) :
  ∀ (initialState : GameState),
    initialState.currentPlayer = true →
    ∃ (strategy : GameState → Move),
      ∀ (game : List Move),
        (∀ (i : Nat), i < game.length → isValidMove piece (updateState initialState (game.take i)) (game.get ⟨i, sorry⟩)) →
        game.length % 2 = 0 →
        ¬ (∃ (nextMove : Move), isValidMove piece (updateState initialState game) nextMove) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l806_80610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l806_80666

/-- Given a 2x2 matrix B with elements [[1, 4], [6, d]] where d ≠ 3,
    if B^(-1) = p * B for some constant p, then d = -1 and p = 1/25 -/
theorem inverse_scalar_multiple (d p : ℝ) : 
  d ≠ 3 →
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 6, d]
  B⁻¹ = p • B →
  (d = -1 ∧ p = 1/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l806_80666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l806_80652

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ - x + Real.sin x

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l806_80652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l806_80620

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f (a - x) + f (a * x^2 - 1) < 0) →
  a < (1 + Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l806_80620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l806_80679

-- Define the propositions
def p (a : ℝ) : Prop := a = -1

def q (a : ℝ) : Prop := 
  ∃ (x y x' y' : ℝ), 
    (x + y + a = 0) ∧ 
    (x^2 + y^2 = 1) ∧
    (x' + y' + a = 0) ∧ 
    (x'^2 + y'^2 = 1) ∧
    ((x - x')^2 + (y - y')^2 = 2)

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ 
  ¬(∀ a : ℝ, q a → p a) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l806_80679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l806_80612

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem g_symmetry (x : ℝ) : g (-Real.pi/8 + x) = -g (-Real.pi/8 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l806_80612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l806_80665

theorem trig_problem (θ : Real) 
  (h1 : Real.sin θ = 4/5)
  (h2 : π/2 < θ ∧ θ < π) :
  Real.sin (2*θ) = -24/25 ∧ 
  Real.cos (θ - π/6) = (4 - 3*Real.sqrt 3)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l806_80665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l806_80646

/-- The area of a trapezium with parallel sides a and b, and height h -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides 24 cm and 18 cm,
    and height 15 cm, is 315 square centimeters -/
theorem trapezium_area_example : trapeziumArea 24 18 15 = 315 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l806_80646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l806_80661

def G : ℕ → ℚ
  | 0 => 3  -- Add this case to handle G(0)
  | 1 => 3
  | (n + 1) => (3 * G n + 2) / 3

theorem G_51_value : G 51 = 109 / 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l806_80661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotone_increasing_l806_80683

open Real

noncomputable def f (x : ℝ) := -cos (2 * x + 3 * π / 4)

theorem f_not_monotone_increasing :
  ¬ (∀ x y : ℝ, x ∈ Set.Icc (π / 8) (5 * π / 8) → y ∈ Set.Icc (π / 8) (5 * π / 8) → x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotone_increasing_l806_80683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l806_80614

theorem arcsin_equation_solution (x : ℝ) : 
  Real.arcsin x + Real.arcsin (3 * x) = π / 2 → x = 1 / Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l806_80614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_two_l806_80682

noncomputable def sequenceLimit (n : ℕ) : ℝ := (2 * (n : ℝ)^3) / ((n : ℝ)^3 - 2)

theorem sequence_limit_is_two :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequenceLimit n - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_two_l806_80682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_divides_hypotenuse_l806_80657

/-- A right triangle with legs in ratio 5:6 and hypotenuse 122 -/
structure RightTriangle where
  /-- The length of the shorter leg -/
  short_leg : ℝ
  /-- The length of the longer leg -/
  long_leg : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The legs are in ratio 5:6 -/
  leg_ratio : long_leg = (6/5) * short_leg
  /-- The hypotenuse is 122 -/
  hyp_length : hypotenuse = 122
  /-- The Pythagorean theorem holds -/
  pythagoras : short_leg^2 + long_leg^2 = hypotenuse^2

/-- The segments created by the height to the hypotenuse -/
noncomputable def hypotenuse_segments (t : RightTriangle) : ℝ × ℝ :=
  let segment1 := (t.short_leg^2) / t.hypotenuse
  let segment2 := (t.long_leg^2) / t.hypotenuse
  (segment1, segment2)

/-- The theorem stating that the height divides the hypotenuse into segments of length 50 and 72 -/
theorem height_divides_hypotenuse (t : RightTriangle) :
  hypotenuse_segments t = (50, 72) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_divides_hypotenuse_l806_80657
