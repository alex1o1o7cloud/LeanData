import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_sharing_ratio_l38_3843

def initial_investment_A : ℕ := 3500
def initial_investment_B : ℕ := 15750
def months_before_B_joins : ℕ := 8
def total_months : ℕ := 12

def effective_capital_A : ℕ := initial_investment_A * total_months
def effective_capital_B : ℕ := initial_investment_B * (total_months - months_before_B_joins)

def gcd_value : ℕ := Nat.gcd effective_capital_A effective_capital_B

theorem profit_sharing_ratio :
  (effective_capital_A / gcd_value) = 2 ∧ (effective_capital_B / gcd_value) = 3 := by
  sorry

#eval effective_capital_A
#eval effective_capital_B
#eval gcd_value
#eval effective_capital_A / gcd_value
#eval effective_capital_B / gcd_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_sharing_ratio_l38_3843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_score_l38_3848

theorem math_competition_score (total_questions : ℕ) (correct_points : ℕ) (incorrect_deduction : ℕ) (final_score : ℤ) (correct_answers : ℕ) : 
  total_questions = 12 →
  correct_points = 10 →
  incorrect_deduction = 5 →
  final_score = 75 →
  (correct_answers : ℤ) * (correct_points : ℤ) + (total_questions - correct_answers) * (-incorrect_deduction : ℤ) = final_score →
  correct_answers = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_score_l38_3848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l38_3866

theorem two_true_propositions : 
  let prop1 := (¬∀ x : ℝ, x^2 - x > 0) ↔ (∃ x : ℝ, x^2 - x < 0)
  let prop2 := ∀ x : ℕ, x > 0 → Odd (2 * x^4 + 1)
  let prop3 := ∀ x : ℝ, |2*x - 1| > 1 → (0 < 1/x ∧ 1/x < 1) ∨ 1/x < 0
  (¬prop1 ∧ prop2 ∧ prop3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l38_3866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_45_degrees_l38_3811

/-- Given a line passing through points A(2, 4) and B(1, m) with a slope angle of 45°, prove that m = 3. -/
theorem line_slope_45_degrees (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (2, 4) ∈ line ∧ 
    (1, m) ∈ line ∧ 
    ∃ (angle : ℝ), angle = 45 ∧ Real.tan (angle * π / 180) = (4 - m) / (2 - 1)) → 
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_45_degrees_l38_3811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z6_magnitude_l38_3816

def z : ℕ → ℂ
  | 0 => 1
  | n + 1 => (z n)^2 + 1 + Complex.I

theorem z6_magnitude :
  Complex.abs (z 6) = Real.sqrt 3835182225545 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z6_magnitude_l38_3816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l38_3819

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Define the property of having four distinct real solutions
def has_four_distinct_real_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (∀ i : ℝ, i = x₁ ∨ i = x₂ ∨ i = x₃ ∨ i = x₄ → a * (f i)^2 - f i + 1 = 0)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_four_distinct_real_solutions a ↔ 0 < a ∧ a < 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l38_3819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_formula_l38_3809

/-- A geometric sequence with specific first four terms -/
noncomputable def GeometricSequence (x y : ℝ) : ℕ → ℝ
  | 0 => x + y
  | 1 => x - y
  | 2 => x / y
  | 3 => x * y
  | n + 4 => GeometricSequence x y 3 * ((x - y) / (x + y))

/-- The fifth term of the geometric sequence -/
noncomputable def fifthTerm (x y : ℝ) : ℝ := GeometricSequence x y 4

theorem fifth_term_formula (x y : ℝ) (h : y ≠ 0) (h' : x + y ≠ 0) :
  fifthTerm x y = (x * y * (x - y)) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_formula_l38_3809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_19_innings_l38_3884

/-- Represents the average score of a cricketer after a certain number of innings -/
structure CricketerScore where
  innings : ℕ
  average : ℝ

/-- Calculates the new average score after an additional inning -/
noncomputable def newAverage (prev : CricketerScore) (runsScored : ℕ) : ℝ :=
  (prev.innings * prev.average + runsScored) / (prev.innings + 1)

/-- Theorem stating the average score after 19 innings -/
theorem average_after_19_innings
  (prev : CricketerScore)
  (h1 : prev.innings = 18)
  (h2 : newAverage prev 98 = prev.average + 4) :
  newAverage prev 98 = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_19_innings_l38_3884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ratio_right_triangles_l38_3829

/-- The radius of the incircle of a right triangle with area a -/
noncomputable def incircle_radius_right_triangle (a : ℝ) : ℝ := sorry

/-- The radius of the circumcircle of a right triangle with area b -/
noncomputable def circumcircle_radius_right_triangle (b : ℝ) : ℝ := sorry

theorem min_area_ratio_right_triangles (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h_incircle_circumcircle : ∃ (r : ℝ), r > 0 ∧ 
    r = incircle_radius_right_triangle a ∧ 
    r = circumcircle_radius_right_triangle b) :
  (a / b : ℝ) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ratio_right_triangles_l38_3829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l38_3852

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define that P is the midpoint of chord AB
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define that A and B are on the circle
def on_circle (point : ℝ × ℝ) : Prop := my_circle point.1 point.2

-- Define the equation of line AB
def line_equation (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  is_midpoint P A B →
  on_circle A →
  on_circle B →
  ∀ x y : ℝ,
  line_equation x y ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) :=
by
  sorry

#check chord_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l38_3852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_key_points_l38_3836

open Real

/-- Definition of n-key point for sin x -/
def is_n_key_point (a : ℝ) (n : ℕ) : Prop :=
  ∃ (l : ℝ → ℝ), (l a = sin a) ∧ 
  (∀ x, l x = (cos a) * (x - a) + sin a) ∧
  (∃ (s : Finset ℝ), s.card = n ∧ ∀ x ∈ s, sin x = l x)

/-- The set of all 1-key points -/
def one_key_points : Set ℝ := {a | ∃ k : ℤ, a = k * π}

/-- The set of all points kπ + 2 where k is an integer -/
def two_key_candidates : Set ℝ := {a | ∃ k : ℤ, a = k * π + 2}

/-- Helper function to convert Real to Nat -/
noncomputable def realToNat (x : ℝ) : ℕ := 
  Nat.floor (Int.floor (|2 * x / π|) + 1)

theorem sin_key_points :
  (∀ a, a ∈ one_key_points ↔ is_n_key_point a 1) ∧
  (∀ a, a ∈ two_key_candidates → is_n_key_point a 2) ∧
  (∀ a₀ n₀, is_n_key_point a₀ n₀ → ∀ k : ℤ, is_n_key_point (k * π + a₀) n₀ ∧ is_n_key_point (k * π - a₀) n₀) ∧
  (∀ a₀, tan a₀ = a₀ → is_n_key_point a₀ (realToNat a₀)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_key_points_l38_3836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l38_3885

/-- The distance between two parallel lines given by their coefficients -/
noncomputable def distance_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / Real.sqrt (a^2 + b^2)

/-- Theorem stating that the distance between the given parallel lines is 1/2 -/
theorem distance_between_specific_lines :
  distance_parallel_lines 6 (-8) (-2) (-7) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l38_3885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_conditions_l38_3815

theorem integer_pairs_satisfying_conditions : 
  let pairs := {(x, y) : ℕ × ℕ | x ≤ y ∧ (Real.sqrt 1992 : ℝ) = Real.sqrt x + Real.sqrt y}
  (Finset.card (Finset.filter (λ p => p ∈ pairs) (Finset.range 1993 ×ˢ Finset.range 1993)) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_conditions_l38_3815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_positions_l38_3841

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := {P | dist O P = r}

-- Define the center and points
def O : ℝ × ℝ := (0, 0)  -- Assume the center is at the origin for simplicity
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the conditions
def radius : ℝ := 6
def PO_distance : ℝ := 4
def QO_distance : ℝ := 6

-- State the theorem
theorem point_positions :
  (dist O P = PO_distance) →
  (dist O Q = QO_distance) →
  (P ∈ {X : ℝ × ℝ | dist O X < radius}) ∧ 
  (Q ∈ Circle O radius) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_positions_l38_3841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_triangle_area_l38_3851

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := Real.sqrt (1 - E.b^2 / E.a^2)

/-- The area of a triangle given three side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem ellipse_eccentricity_and_triangle_area 
  (E : Ellipse) 
  (h_eq : E.a^2 = 49 ∧ E.b^2 = 24) 
  (P : PointOnEllipse E) 
  (h_perp : ∃ (F₁ F₂ : ℝ × ℝ), 
    (P.x - F₁.1)*(P.x - F₂.1) + (P.y - F₁.2)*(P.y - F₂.2) = 0 ∧ 
    (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4*(E.a^2 - E.b^2)) :
  eccentricity E = 5/7 ∧ 
  ∃ (a b c : ℝ), triangle_area a b c = 24 ∧ 
    a^2 + b^2 = 4*(E.a^2 - E.b^2) ∧ 
    (a + b)^2 = 4*E.a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_triangle_area_l38_3851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l38_3828

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define point P on C
noncomputable def P : ℝ × ℝ := (2 * Real.sqrt 6 / 3, 1)

-- Define orthocenter H
noncomputable def H : ℝ × ℝ := (2 * Real.sqrt 6 / 3, -5/3)

-- Define foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define left vertex A
def A : ℝ × ℝ := (-2, 0)

-- Define the line l
def l (x y : ℝ) : Prop := y = 2 * (x - 1)

-- Statement of the theorem
theorem ellipse_line_theorem :
  C P.1 P.2 ∧
  (∃ (k₁ k₂ : ℝ), k₁ + k₂ = -1/2 ∧
    (∀ (D E : ℝ × ℝ), C D.1 D.2 ∧ C E.1 E.2 ∧ l D.1 D.2 ∧ l E.1 E.2 →
      k₁ = (D.2 - A.2) / (D.1 - A.1) ∧
      k₂ = (E.2 - A.2) / (E.1 - A.1))) →
  ∀ (x y : ℝ), l x y ↔ y = 2 * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l38_3828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_C_identities_l38_3842

/-- S function definition -/
noncomputable def S (a : ℝ) (x : ℝ) : ℝ := a^x - a^(-x)

/-- C function definition -/
noncomputable def C (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

/-- Main theorem -/
theorem S_C_identities (a : ℝ) (x y : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (2 * S a (x + y) = S a x * C a y + C a x * S a y) ∧
  (2 * S a (x - y) = S a x * C a y - C a x * S a y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_C_identities_l38_3842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_appleby_class_size_l38_3886

theorem mrs_appleby_class_size :
  ∀ (g : ℕ), 
  let b := g + 3
  let total_jelly_beans := 600
  let remaining_jelly_beans := 16
  let distributed_jelly_beans := total_jelly_beans - remaining_jelly_beans
  g * g + (g + 3) * (g + 8) = distributed_jelly_beans →
  g + b = 31 :=
by
  intro g
  intro h
  sorry

#check mrs_appleby_class_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_appleby_class_size_l38_3886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_proof_l38_3877

noncomputable section

/-- The length of Emily's step -/
def E : ℝ := 1

/-- The length of the ship in Emily's steps -/
def L : ℝ := 80 / 3

/-- The effective step of the ship (distance the ship travels while Emily takes one step against the current) -/
def S : ℝ := 1

/-- The effect of the river current on the ship's movement in Emily's steps -/
def C : ℝ := S / 8

/-- Emily's steps count when walking from back to front -/
def steps_against_current : ℕ := 320

/-- Emily's steps count when walking from front to back -/
def steps_with_current : ℕ := 80

theorem ship_length_proof :
  (steps_against_current : ℝ) * E = L + steps_against_current * (S - C) ∧
  (steps_with_current : ℝ) * E = L - steps_with_current * (S + C) →
  L = 80 / 3 * E := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_proof_l38_3877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_sixth_card_is_joker_l38_3831

/-- Represents the cards in the sequence -/
inductive Card
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Joker
  deriving Repr

/-- The sequence of cards -/
def cardSequence : List Card := 
  [Card.Ace, Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, 
   Card.Eight, Card.Nine, Card.Ten, Card.Jack, Card.Queen, Card.King, Card.Joker]

/-- The length of the card sequence -/
def sequenceLength : Nat := cardSequence.length

/-- Function to get the nth card in the repeating sequence -/
def nthCard (n : Nat) : Card :=
  cardSequence[n % sequenceLength]'(by
    have h : sequenceLength > 0 := by simp [sequenceLength, cardSequence]
    apply Nat.mod_lt n h
  )

theorem fifty_sixth_card_is_joker : 
  nthCard 56 = Card.Joker := by sorry

#eval nthCard 56

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_sixth_card_is_joker_l38_3831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_beta_five_l38_3847

/-- Given that α is inversely proportional to 1/β and α = 8 when β = 2,
    prove that α = 20 when β = 5. -/
theorem alpha_value_for_beta_five
  (α : ℝ → ℝ)
  (proportional : ∃ k : ℝ, ∀ β : ℝ, β ≠ 0 → α β = k * β)
  (initial_condition : α 2 = 8) :
  α 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_beta_five_l38_3847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_flip_theorem_l38_3808

/-- Represents a cell in the grid -/
inductive Cell
| Positive
| Negative

/-- Represents a move that can flip 7 cells in a specific pattern -/
structure Move where
  pattern : List (Nat × Nat)
  h_size : pattern.length = 7

/-- Represents the grid -/
def Grid (m n : Nat) := Array (Array Cell)

/-- Checks if a number is a multiple of 4 -/
def isMultipleOf4 (n : Nat) : Prop := ∃ k, n = 4 * k

/-- Applies a move to the grid -/
def applyMove {m n : Nat} (grid : Grid m n) (move : Move) : Grid m n := sorry

/-- Checks if all cells in the grid have been flipped -/
def allCellsFlipped {m n : Nat} (original final : Grid m n) : Prop := sorry

theorem grid_flip_theorem (m n : Nat) :
  (∃ (moves : List Move), ∀ (original : Grid m n),
    ∃ (final : Grid m n),
      (final = moves.foldl applyMove original) ∧
      (allCellsFlipped original final)) ↔
  (isMultipleOf4 m ∧ isMultipleOf4 n) := by sorry

#check grid_flip_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_flip_theorem_l38_3808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_assignment_unique_l38_3838

-- Define the colors
inductive Color
| Red
| Blue

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

-- Define a clothing item
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the clothing assignment for all children
def ClothingAssignment := Child → Clothing

-- Define the conditions
def satisfiesConditions (assignment : ClothingAssignment) : Prop :=
  -- Alyna and Bohdan have red t-shirts
  (assignment Child.Alyna).tshirt = Color.Red ∧
  (assignment Child.Bohdan).tshirt = Color.Red ∧
  -- Alyna and Bohdan have shorts of different colors
  (assignment Child.Alyna).shorts ≠ (assignment Child.Bohdan).shorts ∧
  -- Vika and Grysha have t-shirts of different colors
  (assignment Child.Vika).tshirt ≠ (assignment Child.Grysha).tshirt ∧
  -- Vika and Grysha both have blue shorts
  (assignment Child.Vika).shorts = Color.Blue ∧
  (assignment Child.Grysha).shorts = Color.Blue ∧
  -- The girls (Alyna and Vika) have t-shirts of different colors
  (assignment Child.Alyna).tshirt ≠ (assignment Child.Vika).tshirt ∧
  -- The girls (Alyna and Vika) have shorts of different colors
  (assignment Child.Alyna).shorts ≠ (assignment Child.Vika).shorts

-- Define the correct assignment
def correctAssignment : ClothingAssignment :=
  fun c => match c with
    | Child.Alyna => { tshirt := Color.Red, shorts := Color.Red }
    | Child.Bohdan => { tshirt := Color.Red, shorts := Color.Blue }
    | Child.Vika => { tshirt := Color.Blue, shorts := Color.Blue }
    | Child.Grysha => { tshirt := Color.Red, shorts := Color.Blue }

-- Theorem to prove
theorem clothing_assignment_unique :
  ∀ (assignment : ClothingAssignment),
    satisfiesConditions assignment → assignment = correctAssignment :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_assignment_unique_l38_3838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_canonical_equations_l38_3824

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations 
  (plane1 : ∀ (x y z : ℝ), 2*x - 3*y + z + 6 = 0)
  (plane2 : ∀ (x y z : ℝ), x - 3*y - 2*z + 3 = 0) :
  ∃ (t : ℝ), ∃ (x y z : ℝ), x = 9*t - 3 ∧ y = 5*t ∧ z = -3*t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_canonical_equations_l38_3824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canteen_position_l38_3876

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

theorem canteen_position (road : Set Point) (girls_camp boys_camp canteen : Point) :
  girls_camp.x = 0 ∧ 
  girls_camp.y = 450 ∧
  boys_camp.x = 700 ∧
  boys_camp.y = 450 ∧
  canteen.y = 0 ∧
  canteen ∈ road ∧
  distance canteen girls_camp = distance canteen boys_camp →
  distance canteen girls_camp = 538 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canteen_position_l38_3876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_consecutive_numbers_l38_3844

def set_size : ℕ := 10
def selection_size : ℕ := 3

def is_consecutive (a b : ℕ) : Prop := b = a + 1

def has_at_least_two_consecutive (selection : Finset ℕ) : Prop :=
  ∃ a b c, a ∈ selection ∧ b ∈ selection ∧ c ∈ selection ∧ (is_consecutive a b ∨ is_consecutive b c)

def total_combinations : ℕ := Nat.choose set_size selection_size

def favorable_outcomes : ℕ := 64

theorem probability_of_consecutive_numbers :
  (favorable_outcomes : ℚ) / total_combinations = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_consecutive_numbers_l38_3844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l38_3802

noncomputable def area_triangle (a b c : ℝ) : ℝ := (1 / 2) * a * b * Real.sin c

theorem triangle_properties (a b c A B C : ℝ) : 
  c = 2 → C = π / 3 →
  (a = 2 * Real.sqrt 3 / 3 → A = π / 6) ∧
  (Real.sin B = 2 * Real.sin A → 
    area_triangle a b C = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l38_3802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l38_3887

noncomputable def sequence_a (θ : ℝ) : ℕ → ℝ
  | 0 => 1
  | 1 => 1 - 2 * Real.sin θ ^ 2 * Real.cos θ ^ 2
  | (n + 2) => sequence_a θ (n + 1) - sequence_a θ n * Real.sin θ ^ 2 * Real.cos θ ^ 2

theorem sequence_a_bounds (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) (n : ℕ) :
  1 / (2 ^ n) ≤ sequence_a θ n ∧
  sequence_a θ n ≤ 1 - Real.sin (2 * θ) ^ n * (1 - 1 / (2 ^ n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l38_3887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_speed_theorem_l38_3858

noncomputable def late_speed : ℝ := 30

noncomputable def early_speed : ℝ := 50

noncomputable def late_time : ℝ := 5 / 60

noncomputable def early_time : ℝ := -5 / 60

noncomputable def exact_speed : ℝ := 37.5

theorem commute_speed_theorem :
  ∃ (d t : ℝ),
    d > 0 ∧ t > 0 ∧
    d = late_speed * (t + late_time) ∧
    d = early_speed * (t + early_time) ∧
    d = exact_speed * t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_speed_theorem_l38_3858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l38_3889

/-- Given five points on a line, the sum of squared distances from an arbitrary point
    to these five points is minimized when the arbitrary point is at the mean position. -/
theorem min_sum_squared_distances (P₁ P₂ P₃ P₄ P₅ : ℝ) :
  let mean := (P₁ + P₂ + P₃ + P₄ + P₅) / 5
  ∀ P : ℝ, (P - P₁)^2 + (P - P₂)^2 + (P - P₃)^2 + (P - P₄)^2 + (P - P₅)^2
         ≥ (mean - P₁)^2 + (mean - P₂)^2 + (mean - P₃)^2 + (mean - P₄)^2 + (mean - P₅)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l38_3889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l38_3857

def matrix_power (A : Matrix (Fin 3) (Fin 3) ℚ) (m : ℕ) : Matrix (Fin 3) (Fin 3) ℚ :=
  A ^ m

theorem matrix_equation_solution (b : ℚ) (m : ℕ) :
  matrix_power
    (![![1, 3, b],
       ![0, 1, 5],
       ![0, 0, 1]])
    m
  =
    ![![1, 33, 6006],
       ![0, 1, 55],
       ![0, 0, 1]]
  →
  b + m = 432 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l38_3857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_at_85km_l38_3867

/-- Represents the hiking and cycling scenario --/
structure HikingCycling where
  a_speed1 : ℝ
  a_time1 : ℝ
  a_speed2 : ℝ
  a_time2 : ℝ
  a_break : ℝ
  a_speed3 : ℝ
  a_time3 : ℝ
  b_delay : ℝ
  b_speed : ℝ

/-- The distance where B catches up with A --/
noncomputable def catchup_distance (scenario : HikingCycling) : ℝ :=
  scenario.b_speed * ((scenario.a_speed1 * scenario.a_time1 + 
                       scenario.a_speed2 * scenario.a_time2 + 
                       scenario.a_speed3 * scenario.a_time3) / 
                      (scenario.b_speed - scenario.a_speed3))

/-- Theorem stating that B catches up with A at 85 km --/
theorem catchup_at_85km (scenario : HikingCycling) 
  (h1 : scenario.a_speed1 = 10)
  (h2 : scenario.a_time1 = 2)
  (h3 : scenario.a_speed2 = 5)
  (h4 : scenario.a_time2 = 3)
  (h5 : scenario.a_break = 1)
  (h6 : scenario.a_speed3 = 8)
  (h7 : scenario.a_time3 = 2)
  (h8 : scenario.b_delay = 7)
  (h9 : scenario.b_speed = 20) :
  catchup_distance scenario = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_at_85km_l38_3867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l38_3825

noncomputable def f (x : ℝ) := 2 * Real.sin (0.5 * x - Real.pi/6)

theorem sine_function_properties :
  let amplitude := 2
  let period := 4 * Real.pi
  let initial_phase := -Real.pi/6
  let max_value := 2
  let min_value := -2
  (∀ x : ℝ, f x ≤ amplitude ∧ f x ≥ -amplitude) ∧
  (∀ x : ℝ, f (x + period) = f x) ∧
  (f (4*Real.pi/3) = 2) ∧
  (∀ k : ℤ, f (4*k*Real.pi + 4*Real.pi/3) = max_value) ∧
  (∀ k : ℤ, f (4*k*Real.pi - 2*Real.pi/3) = min_value) ∧
  (∀ k : ℤ, ∀ x : ℝ, 4*k*Real.pi - 2*Real.pi/3 ≤ x ∧ x ≤ 4*k*Real.pi + 4*Real.pi/3 →
    ∀ y : ℝ, 4*k*Real.pi - 2*Real.pi/3 ≤ y ∧ y ≤ x → f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l38_3825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_obtuse_greater_than_sin_sum_l38_3870

theorem sin_obtuse_greater_than_sin_sum (A B : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  π/2 < B ∧ B < π →  -- B is obtuse
  Real.sin B > Real.sin (A + B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_obtuse_greater_than_sin_sum_l38_3870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_size_sin_B_plus_sin_C_l38_3875

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * (Real.sin t.A) ^ 2 + 3 * Real.cos (t.B + t.C) = 0 ∧
  t.area = 5 * Real.sqrt 3 ∧
  t.a = Real.sqrt 21

-- Theorem 1: Size of angle A
theorem angle_A_size (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Value of sin B + sin C
theorem sin_B_plus_sin_C (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.B + Real.sin t.C = 9 * Real.sqrt 7 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_size_sin_B_plus_sin_C_l38_3875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_equals_one_l38_3871

theorem xy_equals_one (x y : ℝ) : (5 : ℝ) ^ ((x + y) ^ 2) / (5 : ℝ) ^ ((x - y) ^ 2) = 625 → x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_equals_one_l38_3871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l38_3837

-- Define the function as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x + 3)

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Iic 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l38_3837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_time_proof_l38_3823

/-- The time taken by worker c to finish the job alone -/
noncomputable def time_c : ℝ := 60

/-- The rate at which a job is completed -/
noncomputable def job_rate (time : ℝ) : ℝ := 1 / time

/-- The combined rate of multiple workers -/
noncomputable def combined_rate (rates : List ℝ) : ℝ := rates.sum

theorem c_time_proof (a b c d : ℝ) 
  (h1 : combined_rate [job_rate a, job_rate b] = job_rate 15)
  (h2 : combined_rate [job_rate a, job_rate b, job_rate c] = job_rate 11)
  (h3 : combined_rate [job_rate c, job_rate d] = job_rate 20)
  (h4 : job_rate d = job_rate 30) :
  job_rate c = job_rate time_c := by
  sorry

#check c_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_time_proof_l38_3823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_speed_B_is_12_l38_3801

/-- The speed of student B in km/h -/
noncomputable def speed_B : ℝ := 12

/-- The speed of student A in km/h -/
noncomputable def speed_A : ℝ := 1.2 * speed_B

/-- The distance from school to the activity location in km -/
noncomputable def distance : ℝ := 12

/-- The time difference in hours between A and B's arrival -/
noncomputable def time_diff : ℝ := 1/6

theorem student_B_speed :
  distance / speed_B - time_diff = distance / speed_A :=
by sorry

theorem speed_B_is_12 : speed_B = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_speed_speed_B_is_12_l38_3801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l38_3865

theorem sin_cos_relation (α : ℝ) : 
  Real.sin (π + α) = 1/3 → Real.cos ((3/2) * π - α) = -1/3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l38_3865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_l38_3854

theorem triangle_special_angle (a b c : ℝ) (h : a^2 = b^2 + c^2 + b*c) :
  (b^2 + c^2 - a^2) / (2*b*c) = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_l38_3854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_cubic_roots_l38_3897

theorem triangle_area_from_cubic_roots (a b c : ℝ) : 
  (a^3 - 5*a^2 + 6*a - 3/5 = 0) →
  (b^3 - 5*b^2 + 6*b - 3/5 = 0) →
  (c^3 - 5*c^2 + 6*c - 3/5 = 0) →
  let K := Real.sqrt ((a + b + c) / 2 * 
    ((a + b + c) / 2 - a) * 
    ((a + b + c) / 2 - b) * 
    ((a + b + c) / 2 - c))
  K = 2 * Real.sqrt 21 / 5 :=
by
  intros ha hb hc
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_cubic_roots_l38_3897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l38_3896

/-- Translates a point in a 2D plane by a given vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem midpoint_after_translation 
  (B G : ℝ × ℝ) 
  (h_B : B = (1, 3)) 
  (h_G : G = (5, 3)) 
  (v : ℝ × ℝ) 
  (h_v : v = (3, -4)) :
  translate ((B.1 + G.1) / 2, (B.2 + G.2) / 2) v = (6, -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l38_3896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_division_l38_3846

/-- Represents a trapezoid ABCD with bases AD and BC -/
structure Trapezoid where
  a : ℝ  -- length of base AD
  b : ℝ  -- length of base BC
  h : ℝ  -- height of the trapezoid

/-- Represents a point M on the extension of BC -/
def ExtensionPoint (t : Trapezoid) := ℝ

/-- Calculates the ratio of the area cut off by AM to the total area of the trapezoid -/
noncomputable def area_ratio (t : Trapezoid) (m : ExtensionPoint t) : ℝ :=
  sorry

/-- The theorem statement -/
theorem trapezoid_area_division (t : Trapezoid) (m : ExtensionPoint t) :
  (∃ x : ℝ, x = m ∧ 
    (area_ratio t m = 1/4 ∨ area_ratio t m = 3/4)) →
  (m = (t.a * (3 * t.a - t.b)) / (t.a + t.b) ∨ 
   m = (t.a * (t.a - 3 * t.b)) / (3 * (t.a + t.b))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_division_l38_3846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_v_onto_Q_l38_3807

/-- The plane defined by 2x - 3y + 5z = 0 -/
def Q : Set (Fin 3 → ℝ) := {v | 2 * v 0 - 3 * v 1 + 5 * v 2 = 0}

/-- The vector to be projected -/
def v : Fin 3 → ℝ := ![2, 3, 4]

/-- The normal vector of the plane -/
def n : Fin 3 → ℝ := ![2, -3, 5]

/-- The projection of v onto the plane Q -/
noncomputable def projection (v : Fin 3 → ℝ) (Q : Set (Fin 3 → ℝ)) : Fin 3 → ℝ := sorry

theorem projection_of_v_onto_Q : 
  projection v Q = ![23/19, 159/38, 77/38] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_v_onto_Q_l38_3807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_minus_two_equals_negative_one_l38_3874

theorem division_minus_two_equals_negative_one (x : ℚ) :
  x = (-3)⁻¹ - 2 → (x = -1 ↔ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_minus_two_equals_negative_one_l38_3874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_in_row_l38_3855

theorem tiles_in_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 225 ∧ tile_size = 0.5 → (Real.sqrt room_area * 12 / tile_size : ℝ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_in_row_l38_3855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_cannot_form_triangle_l38_3894

/-- Vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the magnitude of a 3D vector -/
noncomputable def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

/-- Check if three vectors can form a non-degenerate triangle -/
def canFormNonDegenerateTriangle (a b c : Vector3D) : Prop :=
  (magnitude a + magnitude b > magnitude c) ∧
  (magnitude a + magnitude c > magnitude b) ∧
  (magnitude b + magnitude c > magnitude a)

/-- Given vectors -/
def a : Vector3D := { x := -1, y := 2, z := -1 }
def b : Vector3D := { x := -3, y := 6, z := -3 }
def c : Vector3D := { x := -2, y := 4, z := -2 }

/-- Theorem: The given vectors cannot form a non-degenerate triangle -/
theorem vectors_cannot_form_triangle : ¬(canFormNonDegenerateTriangle a b c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_cannot_form_triangle_l38_3894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_camp_inequality_l38_3898

theorem olympic_camp_inequality (a b k : ℕ) (h1 : b % 2 = 1) (h2 : b ≥ 3) 
  (h3 : ∀ (e1 e2 : ℕ), e1 ≠ e2 → e1 < b → e2 < b → 
    (∃ (same_opinion : Finset ℕ), same_opinion.card ≤ k)) :
  (k : ℚ) / a ≥ (b - 1) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_camp_inequality_l38_3898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_over_sixth_root_of_seven_l38_3832

theorem fourth_root_over_sixth_root_of_seven :
  (7 : ℝ) ^ (1/4 : ℝ) / (7 : ℝ) ^ (1/6 : ℝ) = (7 : ℝ) ^ (1/12 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_over_sixth_root_of_seven_l38_3832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_trisection_point_l38_3863

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to calculate a trisection point of a line segment -/
noncomputable def trisectionPoint (p1 p2 : Point2D) (m n : ℝ) : Point2D :=
  { x := (m * p2.x + n * p1.x) / (m + n),
    y := (m * p2.y + n * p1.y) / (m + n) }

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem line_through_trisection_point :
  ∃ (l : Line2D),
    l.a = 1 ∧ l.b = -4 ∧ l.c = 13 ∧
    pointOnLine { x := 3, y := 4 } l ∧
    (pointOnLine (trisectionPoint { x := -4, y := 5 } { x := 5, y := -1 } 1 2) l ∨
     pointOnLine (trisectionPoint { x := -4, y := 5 } { x := 5, y := -1 } 2 1) l) := by
  sorry

#check line_through_trisection_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_trisection_point_l38_3863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l38_3813

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∃ k : ℕ, k > 2 ∧ k < n ∧ (Nat.choose n k : ℝ) * ((Nat.choose n (k - 2) : ℝ)⁻¹) = 14/3) →
  (∃ r : ℕ, (Nat.choose n r : ℝ) * (1/3)^r * x^((n - 5*r)/2 : ℝ) = 5 ∧ (n - 5*r)/2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l38_3813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l38_3834

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x ≤ 0 then x^2 + 6*x
  else 0  -- This else case is added to make the function total

-- Define the range of f
def range_f : Set ℝ := {y | ∃ x, f x = y}

-- Theorem statement
theorem range_of_f : range_f = Set.Icc (-8 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l38_3834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_99_value_l38_3878

/-- Sequence {aₙ} defined by the recurrence relation aₙ₊₁ = aₙ + n/2 and initial term a₁ = 2 -/
def a : ℕ → ℚ
  | 0 => 2  -- Define a₀ to be 2 as well, to handle the Nat.zero case
  | n + 1 => a n + n / 2

/-- The 99th term of the sequence {aₙ} is equal to 2427.5 -/
theorem a_99_value : a 99 = 2427.5 := by
  sorry

#eval a 99  -- This will compute the actual value for verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_99_value_l38_3878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_in_right_triangle_l38_3814

/-- The side length of a square inscribed in a right triangle -/
noncomputable def inscribed_square_side (a b : ℝ) : ℝ :=
  (a * b * (a + b)) / (a^2 + b^2)

/-- Theorem: The side length of a square inscribed in a right triangle
    with legs 7 and 24 is 4375/768 -/
theorem inscribed_square_in_right_triangle :
  inscribed_square_side 7 24 = 4375 / 768 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_in_right_triangle_l38_3814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_zero_l38_3803

/-- The curve defined by x = y^3 -/
def curve1 (x y : ℝ) : Prop := x = y^3

/-- The line defined by x + y = 1 -/
def curve2 (x y : ℝ) : Prop := x + y = 1

/-- A point (x, y) is an intersection if it satisfies both curve equations -/
def is_intersection (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

/-- The distance between two points in ℝ² -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem intersection_distance_zero : 
  ∀ x1 y1 x2 y2 : ℝ, is_intersection x1 y1 → is_intersection x2 y2 → 
    distance x1 y1 x2 y2 = 0 := 
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_zero_l38_3803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l38_3879

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x = 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x : ℝ, x ≠ 1 ∧ (x - a) * (x - (a + 2)) ≤ 0) → 
  a ∈ Set.Icc (-1) 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l38_3879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l38_3826

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then x^2 - 1
  else if -2 ≤ x ∧ x < 0 then -x^2
  else 0  -- This case should never occur in our problem

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, a * x₁ + a = f x₂) ↔
  a ∈ Set.Icc (-4/3 : ℝ) 1 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l38_3826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_proof_l38_3882

open Real

theorem trig_identity_proof (α β : ℝ) 
  (h1 : sin α + sin β = -21/65)
  (h2 : cos α + cos β = -27/65)
  (h3 : 5*π/2 < α ∧ α < 3*π)
  (h4 : -π/2 < β ∧ β < 0) :
  sin ((α + β) / 2) = -7 / Real.sqrt 130 ∧ 
  cos ((α + β) / 2) = -9 / Real.sqrt 130 := by
  sorry

#check trig_identity_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_proof_l38_3882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l38_3860

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the square
def Square := {p : Point | 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 2}

-- Define a coloring function
def Coloring := Point → Color

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem same_color_unit_distance (c : Coloring) : 
  ∃ (p1 p2 : Point), p1 ∈ Square ∧ p2 ∈ Square ∧ 
  c p1 = c p2 ∧ distance p1 p2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l38_3860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_intersection_l38_3822

noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  ((b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁), (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁))

def point_on_line (x y a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

def perpendicular_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

theorem perpendicular_line_through_intersection :
  let p := intersection_point 3 1 (-1) 1 2 (-7)
  point_on_line p.1 p.2 2 (-1) 6 ∧
  point_on_line p.1 p.2 3 1 (-1) ∧
  point_on_line p.1 p.2 1 2 (-7) ∧
  perpendicular_lines 2 (-1) 1 2 :=
by
  sorry

#check perpendicular_line_through_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_intersection_l38_3822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l38_3893

def team_scores : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

def lost_by_one (score : Nat) : Bool := score ≤ 12

def opponent_score (team_score : Nat) : Nat :=
  if lost_by_one team_score then team_score + 1 else team_score / 2

theorem opponent_total_score :
  (team_scores.filter lost_by_one).length = 6 →
  (team_scores.map opponent_score).sum = 105 := by
  intro h
  sorry

#eval (team_scores.filter lost_by_one).length
#eval (team_scores.map opponent_score).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l38_3893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_placements_eq_5_factorial_l38_3869

/-- The number of ways to place 5 distinct pawns on a 5x5 chess board,
    such that each column and each row contains exactly one pawn. -/
def chessboardPlacements : ℕ := Nat.factorial 5

/-- Theorem stating that the number of valid pawn placements on a 5x5 chess board
    is equal to 5 factorial. -/
theorem chessboard_placements_eq_5_factorial :
  chessboardPlacements = Nat.factorial 5 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_placements_eq_5_factorial_l38_3869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_journey_equation_correct_l38_3888

-- Define the total distance
noncomputable def total_distance : ℝ := 3.3

-- Define the speeds for each section
noncomputable def uphill_speed : ℝ := 3
noncomputable def flat_speed : ℝ := 4
noncomputable def downhill_speed : ℝ := 5

-- Define the times for each journey
noncomputable def time_A_to_B : ℝ := 51 / 60
noncomputable def time_B_to_A : ℝ := 53 / 60

-- Define the variables for section distances
variable (x y : ℝ)

-- Theorem statement
theorem return_journey_equation_correct 
  (h1 : x / uphill_speed + y / flat_speed + (total_distance - x - y) / downhill_speed = time_A_to_B)
  (h2 : x ≥ 0)
  (h3 : y ≥ 0)
  (h4 : x + y ≤ total_distance) :
  x / downhill_speed + y / flat_speed + (total_distance - x - y) / uphill_speed = time_B_to_A := by
  sorry

#check return_journey_equation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_journey_equation_correct_l38_3888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_containing_polygon_l38_3872

/-- A convex polygon in the plane -/
structure ConvexPolygon where
  -- We assume the existence of a type for convex polygons
  -- The actual implementation is not provided here

/-- The area of a shape -/
noncomputable def area {α : Type} (shape : α) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  -- We assume the existence of a type for rectangles
  -- The actual implementation is not provided here

/-- Predicate to check if one shape contains another -/
def contains {α β : Type} (outer : α) (inner : β) : Prop := sorry

theorem rectangle_containing_polygon :
  ∀ (P : ConvexPolygon), area P = 1 →
  ∃ (R : Rectangle), contains R P ∧ area R ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_containing_polygon_l38_3872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_point_l38_3827

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 5 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  abs (Real.sqrt 3 * x + y - 5) / 2

-- State the theorem
theorem shortest_distance_point :
  let D : ℝ × ℝ := (Real.sqrt 3 / 2, 3/2)
  curve_C D.1 D.2 ∧
  ∀ (x y : ℝ), curve_C x y → distance_to_line D.1 D.2 ≤ distance_to_line x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_point_l38_3827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_through_points_l38_3835

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-5, 6)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a circle given its radius
noncomputable def circleArea (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem circle_area_through_points :
  circleArea (distance A B) = 100 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_through_points_l38_3835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_count_ways_l38_3892

-- Define the function outside the theorem
def number_of_ways_always_leading (m n : ℕ) : ℕ :=
  sorry

theorem election_count_ways (m n : ℕ) (h : m > n) :
  ∃ (count : ℕ), count = (m - n) * (Nat.choose (m + n) m) / (m + n) ∧
  count = number_of_ways_always_leading m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_count_ways_l38_3892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l38_3833

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n+2 => 1 + ((-1)^(n+2) / sequence_a (n+1))

theorem a_5_value : sequence_a 5 = 2/3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l38_3833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cone_height_is_12_l38_3830

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the height of the smaller cone removed from a frustum -/
noncomputable def smaller_cone_height (f : Frustum) : ℝ :=
  let lower_radius := (f.lower_base_area / Real.pi).sqrt
  let upper_radius := (f.upper_base_area / Real.pi).sqrt
  let total_height := f.altitude * (lower_radius / (lower_radius - upper_radius))
  total_height - f.altitude

/-- Theorem stating the height of the smaller cone for the given frustum -/
theorem smaller_cone_height_is_12 (f : Frustum) 
  (h_altitude : f.altitude = 30)
  (h_lower_area : f.lower_base_area = 196 * Real.pi)
  (h_upper_area : f.upper_base_area = 16 * Real.pi) :
  smaller_cone_height f = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cone_height_is_12_l38_3830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_is_negative_six_l38_3805

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^2 - z = 5 - 5*i

-- Theorem statement
theorem product_of_real_parts_is_negative_six :
  ∃ (z₁ z₂ : ℂ), equation z₁ ∧ equation z₂ ∧ z₁ ≠ z₂ ∧ (Complex.re z₁) * (Complex.re z₂) = -6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_is_negative_six_l38_3805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_log_value_l38_3864

noncomputable def given_sum : ℝ := 1.6666666666666667

noncomputable def log_9_27 : ℝ := (3 : ℝ) / 2

noncomputable def x : ℝ := given_sum - log_9_27

theorem other_log_value : x = 0.1666666666666667 := by
  -- Unfold the definitions
  unfold x
  unfold given_sum
  unfold log_9_27
  
  -- Perform the calculation
  norm_num
  
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_log_value_l38_3864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l38_3859

theorem sin_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = 4/5) (h4 : Real.cos (α + β) = -3/5) : Real.sin β = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l38_3859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l38_3856

/-- The distance between the center of the circle defined by x^2 + y^2 = 4x + 6y + 3 
    and the point (10, -2) is √89. -/
theorem circle_center_distance : 
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 = 4*x + 6*y + 3
  let center : ℝ × ℝ := (2, 3)
  let point : ℝ × ℝ := (10, -2)
  ∃ c : ℝ × ℝ, (∀ x y, circle_eq x y ↔ (x - c.1)^2 + (y - c.2)^2 = 16) ∧
  Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) = Real.sqrt 89 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l38_3856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cyclic_polynomial_for_distinct_integers_l38_3820

theorem no_cyclic_polynomial_for_distinct_integers :
  ∀ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ¬∃ (P : ℤ → ℤ), (∀ x : ℤ, ∃ (p : Polynomial ℤ), P x = p.eval x) ∧
                    P a = b ∧ P b = c ∧ P c = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cyclic_polynomial_for_distinct_integers_l38_3820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_triangles_l38_3810

/-- The value of a square -/
def square : ℝ := sorry

/-- The value of a triangle -/
def triangle : ℝ := sorry

/-- All squares have the same value -/
axiom square_constant : ∀ s₁ s₂ : ℝ, s₁ = square → s₂ = square → s₁ = s₂

/-- All triangles have the same value -/
axiom triangle_constant : ∀ t₁ t₂ : ℝ, t₁ = triangle → t₂ = triangle → t₁ = t₂

/-- First equation: 3 * square + 2 * triangle = 27 -/
axiom eq_one : 3 * square + 2 * triangle = 27

/-- Second equation: 2 * square + 3 * triangle = 23 -/
axiom eq_two : 2 * square + 3 * triangle = 23

/-- The sum of four triangles equals 12 -/
theorem sum_of_four_triangles : 4 * triangle = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_triangles_l38_3810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_to_gym_l38_3873

-- Define the distances and time difference
def distance_home_to_grocery : ℝ := 250
def distance_grocery_to_gym : ℝ := 360
def time_difference : ℝ := 70

-- Define Angelina's speed from home to grocery
noncomputable def speed_home_to_grocery : ℝ → ℝ := λ v => v

-- Define Angelina's speed from grocery to gym
noncomputable def speed_grocery_to_gym : ℝ → ℝ := λ v => 2 * v

-- Define the time taken for each journey
noncomputable def time_home_to_grocery : ℝ → ℝ := λ v => distance_home_to_grocery / (speed_home_to_grocery v)
noncomputable def time_grocery_to_gym : ℝ → ℝ := λ v => distance_grocery_to_gym / (speed_grocery_to_gym v)

-- State the theorem
theorem angelinas_speed_to_gym :
  ∃ v : ℝ, v > 0 ∧ 
  time_home_to_grocery v - time_grocery_to_gym v = time_difference ∧
  speed_grocery_to_gym v = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_to_gym_l38_3873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l38_3812

def A : Set ℤ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℤ := {x : ℤ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l38_3812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l38_3839

/-- The radius of the inscribed circle in a triangle with sides a, b, and c --/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

/-- Theorem: The radius of the inscribed circle in triangle DEF is 121/29 --/
theorem inscribed_circle_radius_specific_triangle :
  inscribed_circle_radius 26 15 17 = 121 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l38_3839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_point_in_sphere_l38_3890

/-- The side length of the cube -/
noncomputable def cube_side : ℝ := 4

/-- The radius of the sphere -/
noncomputable def sphere_radius : ℝ := Real.sqrt 2.25

/-- The volume of the cube -/
noncomputable def cube_volume : ℝ := cube_side ^ 3

/-- The volume of the sphere -/
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3

/-- The probability of a randomly selected point in the cube being inside the sphere -/
noncomputable def probability : ℝ := sphere_volume / cube_volume

theorem probability_of_point_in_sphere :
  probability = (4.5 * Real.pi) / 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_point_in_sphere_l38_3890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_sqrt_5_l38_3853

/-- The minimum length of a tangent line from a point on y = -1 to the circle (x+3)² + (y-2)² = 4 -/
noncomputable def min_tangent_length : ℝ := Real.sqrt 5

/-- The circle C: (x+3)² + (y-2)² = 4 -/
def circle_equation (x y : ℝ) : Prop := (x + 3)^2 + (y - 2)^2 = 4

/-- The line y = -1 -/
def line_equation (y : ℝ) : Prop := y = -1

/-- Theorem stating the minimum length of the tangent line -/
theorem min_tangent_length_is_sqrt_5 :
  ∀ (x₀ : ℝ), 
  ∃ (l : ℝ), l ≥ min_tangent_length ∧ 
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation y ∧
  (y - (-1))^2 + (x - x₀)^2 = l^2 :=
by
  sorry

#check min_tangent_length_is_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_sqrt_5_l38_3853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l38_3849

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) :
  simpleInterest P 4 5 = P / 5 := by
  unfold simpleInterest
  field_simp
  ring

#check interest_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l38_3849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l38_3845

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (6 * x^2 - x - 2)

-- Define the set of x-values where vertical asymptotes occur
def vertical_asymptotes : Set ℝ := {x : ℝ | 6 * x^2 - x - 2 = 0}

-- Theorem statement
theorem vertical_asymptotes_of_f :
  vertical_asymptotes = {-1/2, 2/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_of_f_l38_3845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_radius_side_ratio_l38_3862

/-- Definition: A regular hexagon with radius r and side length s -/
def is_regular_hexagon (r s : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (vertices : Fin 6 → ℝ × ℝ),
    (∀ i, dist center (vertices i) = r) ∧
    (∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = s) ∧
    (∀ i j, i ≠ j → dist (vertices i) (vertices j) = s)

/-- The ratio of the radius of the circumscribed circle to the side length of a regular hexagon is 1:1 -/
theorem regular_hexagon_radius_side_ratio :
  ∀ (r s : ℝ), r > 0 → s > 0 →
  is_regular_hexagon r s → r / s = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_radius_side_ratio_l38_3862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circles_value_l38_3818

/-- The distance between the closest points of two circles with centers at (5, 5) and (22, 13), 
    both tangent to the x-axis -/
noncomputable def distance_between_circles : ℝ :=
  let c1 : ℝ × ℝ := (5, 5)
  let c2 : ℝ × ℝ := (22, 13)
  let r1 : ℝ := c1.2  -- radius of first circle
  let r2 : ℝ := c2.2  -- radius of second circle
  let center_distance : ℝ := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  center_distance - (r1 + r2)

theorem distance_between_circles_value : distance_between_circles = Real.sqrt 353 - 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circles_value_l38_3818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_cardinalities_of_S_l38_3881

-- Define the set of digits
def Digits : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define the set T
def T : Finset Nat := Finset.filter 
  (λ x => x ≥ 10 ∧ (x / 10) ∈ Digits ∧ (x % 10) ∈ Digits ∧ (x / 10) < (x % 10))
  (Finset.range 100)

-- Define the property of S containing all six digits
def containsAllDigits (S : Finset Nat) : Prop :=
  ∀ d, d ∈ Digits → ∃ x ∈ S, d = x / 10 ∨ d = x % 10

-- Define the property of no three numbers in S using all six digits
def noThreeUseAllDigits (S : Finset Nat) : Prop :=
  ∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    ({a / 10, a % 10, b / 10, b % 10, c / 10, c % 10} : Finset Nat) ≠ Digits

-- Theorem statement
theorem possible_cardinalities_of_S :
  ∀ S : Finset Nat,
    S ⊆ T →
    containsAllDigits S →
    noThreeUseAllDigits S →
    (4 : Nat) ≤ S.card ∧ S.card ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_cardinalities_of_S_l38_3881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_9801_l38_3850

/-- The sum of numbers from 1 to n with alternating signs based on perfect squares -/
def alternatingSum (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i => 
    let k := (i + 1 : ℕ)
    let sqrtK := Nat.sqrt k
    if sqrtK * sqrtK == k ∧ sqrtK % 2 == 1 then -k
    else if sqrtK * sqrtK == k ∧ sqrtK % 2 == 0 then k
    else if (sqrtK + 1) * (sqrtK + 1) > k ∧ sqrtK % 2 == 1 then k
    else -k)

theorem alternating_sum_9801 : alternatingSum 9801 = -9801 := by
  sorry

#eval alternatingSum 9801

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_9801_l38_3850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_minimum_l38_3880

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + a + 1)

theorem f_monotonicity_and_minimum (a : ℝ) :
  (∀ x, a ≥ 0 → (deriv (f a)) x < 0) ∧
  (a < 0 → ∃ x₁ x₂, ∀ x, 
    (x < x₁ → (deriv (f a)) x > 0) ∧ 
    (x₁ < x ∧ x < x₂ → (deriv (f a)) x < 0) ∧ 
    (x > x₂ → (deriv (f a)) x > 0)) ∧
  (-1 < a ∧ a < 0 → ∀ x ∈ Set.Icc 1 2, f a x ≥ f a 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_minimum_l38_3880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equality_l38_3840

def U : Set ℕ := {x : ℕ | 0 < x ∧ x < 9}

def M : Set ℕ := {3, 4, 5}

def P : Set ℕ := {1, 3, 6}

theorem complement_intersection_equality : 
  ({2, 7, 8} : Set ℕ) = (U \ M) ∩ (U \ P) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equality_l38_3840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_l38_3800

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := 1 + (n - 1)

noncomputable def sum_arithmetic_sequence (n : ℕ) : ℝ := n * (1 + arithmetic_sequence n) / 2

noncomputable def ratio (n : ℕ) : ℝ := (sum_arithmetic_sequence n + 8) / arithmetic_sequence n

theorem min_ratio :
  ∃ (m : ℝ), m = 9/2 ∧ ∀ (n : ℕ), n ≥ 1 → ratio n ≥ m := by
  sorry

#check min_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_l38_3800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_tangent_circle_l38_3806

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - x^2 / m^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Define the asymptotes of the hyperbola
def asymptotes (m : ℝ) (x y : ℝ) : Prop := y = x/m ∨ y = -x/m

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), asymptotes m x y ∧ circle_eq x y

theorem hyperbola_asymptotes_tangent_circle (m : ℝ) 
  (h1 : m > 0) 
  (h2 : is_tangent m) : 
  m = Real.sqrt 3 / 3 := by
  sorry

#check hyperbola_asymptotes_tangent_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_tangent_circle_l38_3806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_over_four_l38_3895

theorem cos_alpha_minus_pi_over_four (α : ℝ) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) (h3 : Real.tan α = 2) : 
  Real.cos (α - Real.pi / 4) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_over_four_l38_3895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_m_is_neg_one_b_subset_a_iff_m_in_open_interval_l38_3804

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_is_neg_one :
  (A ∩ B (-1) = {x | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B (-1) = {x | x ≥ -1}) := by sorry

-- Theorem for part (2)
theorem b_subset_a_iff_m_in_open_interval :
  ∀ m : ℝ, B m ⊆ A ↔ m ∈ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_m_is_neg_one_b_subset_a_iff_m_in_open_interval_l38_3804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l38_3821

def my_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n ≥ 2, a n = a (n + 1) + n) ∧ a 1 = 1

theorem a_4_value (a : ℕ → ℤ) (h : my_sequence a) : a 4 = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_value_l38_3821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_exists_l38_3891

/-- A 3D point -/
structure Point3 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A 3D line represented by a point and a direction vector -/
structure Line3 where
  point : Point3
  direction : Point3

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (a b : Line3) : Prop :=
  ¬ (a.direction.x * b.direction.y = a.direction.y * b.direction.x ∧
     a.direction.x * b.direction.z = a.direction.z * b.direction.x) ∧
  ¬ ∃ (t s : ℝ), a.point.x + t * a.direction.x = b.point.x + s * b.direction.x ∧
                 a.point.y + t * a.direction.y = b.point.y + s * b.direction.y ∧
                 a.point.z + t * a.direction.z = b.point.z + s * b.direction.z

/-- A point is not on a line if it does not belong to the set of points on that line. -/
def point_not_on_line (P : Point3) (l : Line3) : Prop :=
  ¬ ∃ (t : ℝ), P.x = l.point.x + t * l.direction.x ∧
               P.y = l.point.y + t * l.direction.y ∧
               P.z = l.point.z + t * l.direction.z

/-- A line is perpendicular to another line if their direction vectors are orthogonal. -/
def is_perpendicular (l1 l2 : Line3) : Prop :=
  l1.direction.x * l2.direction.x + l1.direction.y * l2.direction.y + l1.direction.z * l2.direction.z = 0

/-- A line is perpendicular to two other lines if it is perpendicular to both of them. -/
def perpendicular_to_both (l a b : Line3) : Prop :=
  is_perpendicular l a ∧ is_perpendicular l b

/-- The main theorem stating that there exists a unique line through a point
    perpendicular to two given skew lines. -/
theorem unique_perpendicular_line_exists (a b : Line3) (P : Point3)
  (h1 : are_skew a b) (h2 : point_not_on_line P a) (h3 : point_not_on_line P b) :
  ∃! (l : Line3), (l.point = P) ∧ perpendicular_to_both l a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_line_exists_l38_3891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_calculation_l38_3817

theorem rectangle_length_calculation (square_side : ℝ) (rectangle_width : ℝ) 
  (h1 : square_side = 20)
  (h2 : rectangle_width = 14) : 
  (4 * square_side - 2 * rectangle_width) / 2 = 26 := by
  -- Define square_perimeter
  let square_perimeter := 4 * square_side
  -- Define rectangle_length
  let rectangle_length := (square_perimeter - 2 * rectangle_width) / 2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_calculation_l38_3817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_630_l38_3861

theorem number_of_divisors_630 : 
  (Finset.filter (λ x : ℕ => x ∣ 630) (Finset.range (630 + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_630_l38_3861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l38_3883

-- Define the quadratic equation and its roots
def quadratic_equation (t q : ℝ) (x : ℝ) : Prop := x^2 - t*x + q = 0

noncomputable def roots_of_quadratic (t q : ℝ) : Set ℝ :=
  {x | quadratic_equation t q x}

-- Define the condition for the sum of powers
def sum_of_powers_equal (α β : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → n ≤ 2010 → α^n + β^n = α + β

-- Define the function we want to maximize
noncomputable def f (α β : ℝ) : ℝ := 1 / α^2011 + 1 / β^2011

-- State the theorem
theorem max_value_of_f (t q : ℝ) (α β : ℝ) 
  (h1 : α ∈ roots_of_quadratic t q) 
  (h2 : β ∈ roots_of_quadratic t q)
  (h3 : sum_of_powers_equal α β) :
  ∃ (M : ℝ), M = 2 ∧ f α β ≤ M ∧ ∃ (α' β' : ℝ), f α' β' = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l38_3883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_triangular_pyramid_l38_3899

/-- Represents a triangular pyramid with vertices S, A, B, and C -/
structure TriangularPyramid where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

/-- The maximum volume of a triangular pyramid with given constraints -/
noncomputable def max_volume (p : TriangularPyramid) : ℝ :=
  8 * Real.sqrt 6

/-- Theorem stating the maximum volume of the triangular pyramid under given constraints -/
theorem max_volume_of_triangular_pyramid (p : TriangularPyramid) 
  (h1 : p.SA = 4)
  (h2 : p.SB ≥ 7)
  (h3 : p.SC ≥ 9)
  (h4 : p.AB = 5)
  (h5 : p.BC ≤ 6)
  (h6 : p.AC ≤ 8) :
  max_volume p = 8 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_triangular_pyramid_l38_3899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l38_3868

/-- The length of the rope in meters -/
noncomputable def rope_length : ℝ := 3

/-- The minimum length of each piece in meters -/
noncomputable def min_piece_length : ℝ := 1

/-- The probability of cutting the rope such that both pieces are at least min_piece_length long -/
noncomputable def probability_both_pieces_long : ℝ := 
  (rope_length - 2 * min_piece_length) / rope_length

theorem probability_is_one_third : 
  probability_both_pieces_long = 1/3 := by
  -- Expand the definition of probability_both_pieces_long
  unfold probability_both_pieces_long
  -- Substitute the values of rope_length and min_piece_length
  simp [rope_length, min_piece_length]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l38_3868
