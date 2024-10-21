import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_statement_C_l986_98627

-- Define complex numbers z₁ and z₂
def z₁ : ℂ := 1 - Complex.I
def z₂ : ℂ := -2 + 3 * Complex.I

-- Statement B
theorem statement_B : ∀ a b : ℝ, z₁ * (a + Complex.I) = z₂ + b * Complex.I → a * b = -3 := by
  sorry

-- Statement C
theorem statement_C : ∀ p q : ℝ, z₁ ^ 2 + p * z₁ + q = 0 → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_statement_C_l986_98627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_diagonal_difference_l986_98681

def calendar_matrix : Matrix (Fin 5) (Fin 5) ℕ := fun i j => i.val * 5 + j.val + 1

def reverse_row (row : Fin 5 → ℕ) : Fin 5 → ℕ := fun j => row (4 - j)

def modified_matrix : Matrix (Fin 5) (Fin 5) ℕ := fun i j =>
  if i = 1 ∨ i = 2 ∨ i = 4 then reverse_row (calendar_matrix i) j
  else calendar_matrix i j

def main_diagonal_sum : ℕ := (Finset.range 5).sum (fun i => modified_matrix i i)

def anti_diagonal_sum : ℕ := (Finset.range 5).sum (fun i => modified_matrix i (4 - i))

theorem calendar_diagonal_difference :
  (Int.natAbs (main_diagonal_sum - anti_diagonal_sum) : ℕ) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_diagonal_difference_l986_98681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l986_98695

def a : ℕ → ℤ
  | 0 => 1
  | n + 1 => 2 * a n + n^2

theorem a_general_term (n : ℕ) : 
  a n = 7 * 2^(n-1) - n^2 - 2*n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l986_98695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l986_98688

/-- Hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1

/-- Parabola with equation y = (1/8)x^2 -/
def Parabola := {p : ℝ × ℝ | p.2 = (1/8) * p.1^2}

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focus of the parabola y = (1/8)x^2 -/
def parabola_focus : ℝ × ℝ := (0, 2)

theorem hyperbola_eccentricity (h : Hyperbola) 
  (common_focus : parabola_focus ∈ {p : ℝ × ℝ | p.2^2 / h.a^2 - p.1^2 / h.b^2 = 1})
  (chord_length : ∃ x : ℝ, 2 * h.b * Real.sqrt (4 / h.a^2 - 1) = 2 * Real.sqrt 3 / 3) :
  eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l986_98688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_height_l986_98628

/-- The height of a square pyramid with given base edge and face angle -/
noncomputable def pyramid_height (base_edge : ℝ) (face_angle : ℝ) : ℝ :=
  (base_edge * Real.sqrt 6) / 6

theorem square_pyramid_height :
  let base_edge : ℝ := 26
  let face_angle : ℝ := 120 * π / 180  -- Convert degrees to radians
  pyramid_height base_edge face_angle = 13 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_height_l986_98628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_decreasing_on_interval_l986_98600

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The interval (-1, 1) -/
def interval : Set ℝ := Set.Ioo (-1) 1

/-- A function is decreasing on an interval if for any two points in the interval, 
    the function value at the larger point is less than or equal to the function value at the smaller point -/
def isDecreasing (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → g y ≤ g x

theorem not_decreasing_on_interval : ¬(isDecreasing f interval) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_decreasing_on_interval_l986_98600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l986_98619

-- Define the function f as noncomputable
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 3^(x - b)

-- State the theorem
theorem range_of_f (b : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f b x ∈ Set.Icc 1 9) ∧
  (∃ x₁ x₂, 2 ≤ x₁ ∧ x₁ ≤ 4 ∧ 2 ≤ x₂ ∧ x₂ ≤ 4 ∧ f b x₁ = 1 ∧ f b x₂ = 9) :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l986_98619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_126_l986_98613

/-- A rectangle with an inscribed rhombus -/
structure RectangleWithRhombus where
  -- The length of JE
  je : ℝ
  -- The length of EF (side of the rhombus)
  ef : ℝ
  -- The length of KM (diagonal of the rectangle)
  km : ℝ
  -- Assumption that je, ef, and km are positive
  je_pos : 0 < je
  ef_pos : 0 < ef
  km_pos : 0 < km

/-- The perimeter of the rectangle JKLM -/
noncomputable def perimeter (r : RectangleWithRhombus) : ℝ :=
  2 * (r.km + Real.sqrt (r.km^2 - (r.je^2 + r.ef^2)))

/-- Theorem stating that the perimeter of the rectangle is 126 -/
theorem perimeter_is_126 (r : RectangleWithRhombus) 
    (h1 : r.je = 18) (h2 : r.ef = 22) (h3 : r.km = 35) : 
    perimeter r = 126 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_126_l986_98613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_german_team_goals_l986_98675

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def twoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  ∀ x : ℕ, twoCorrect x ↔ x ∈ ({11, 12, 14, 16, 17} : Set ℕ) :=
by sorry

#check german_team_goals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_german_team_goals_l986_98675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_proof_l986_98641

theorem factor_proof (initial_number final_result : ℕ) : 
  initial_number = 17 → final_result = 117 → 
  ∃ (factor : ℚ), (2 * initial_number + 5) * factor = final_result ∧ factor = 3 := by
  intro h1 h2
  use 3
  constructor
  · rw [h1, h2]
    norm_num
  · rfl

#check factor_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_proof_l986_98641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_circle_property_l986_98691

-- Define the trajectory C
noncomputable def trajectory_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent line l
noncomputable def tangent_line (k b x y : ℝ) : Prop := y = k*x + b

-- Define the point P (tangent point)
noncomputable def point_P (k : ℝ) : ℝ × ℝ := (1/k^2, 2/k)

-- Define the point Q (intersection with x=-1)
noncomputable def point_Q (k : ℝ) : ℝ × ℝ := (-1, -k + 1/k)

-- Define the fixed point M
def point_M : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem parabola_tangent_circle_property :
  ∀ (k : ℝ), k ≠ 0 →
  let P := point_P k
  let Q := point_Q k
  let M := point_M
  ∃ (x y : ℝ),
    trajectory_C x y ∧
    tangent_line k (1/k) x y ∧
    (M.1 - P.1) * (M.1 - Q.1) + (M.2 - P.2) * (M.2 - Q.2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_circle_property_l986_98691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l986_98671

-- Define positive integers x and y
variable (x y : ℕ)

-- Define the condition that x divided by y has remainder 9
def has_remainder_9 (x y : ℕ) : Prop :=
  ∃ q : ℕ, x = y * q + 9

-- Define that y is close to 20
def close_to_20 (y : ℕ) : Prop :=
  ∃ ε : ℚ, ε > 0 ∧ ε < 1 ∧ (y : ℚ) > 19 ∧ (y : ℚ) < 21

-- Theorem statement
theorem division_with_remainder
  (h1 : has_remainder_9 x y)
  (h2 : close_to_20 y) :
  ∃ q : ℕ, x = y * q + 9 :=
by
  -- The proof is exactly what h1 states, so we can use it directly
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l986_98671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dugu_win_probability_l986_98690

-- Define the probability of winning the first game
noncomputable def initial_prob : ℝ := 0.4

-- Define the change in probability after winning or losing a game
noncomputable def prob_increase : ℝ := 0.1
noncomputable def prob_decrease : ℝ := 0.1

-- Define a function to calculate the probability of winning a game based on the previous game's outcome
noncomputable def next_game_prob (prev_prob : ℝ) (won_prev : Bool) : ℝ :=
  if won_prev then min (prev_prob + prob_increase) 1
  else max (prev_prob - prob_decrease) 0

-- Define a function to calculate the probability of winning exactly n games out of m games
noncomputable def prob_win_n_out_of_m (n m : ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem dugu_win_probability :
  prob_win_n_out_of_m 3 3 + prob_win_n_out_of_m 3 4 = 0.236 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dugu_win_probability_l986_98690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l986_98630

theorem min_value_expression : 
  ∃ (m : ℝ), (∀ (x y : ℝ), 
    (3 * Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (8 - 4 * Real.sqrt 3) * Real.sin x + 2) * 
    (3 + 2 * Real.sqrt (11 - Real.sqrt 3) * Real.cos y - Real.cos (2 * y)) ≥ m) ∧ 
  (Int.floor m = -33) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l986_98630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_implications_l986_98615

-- Define the points as variables in the complex plane
variable (X Y Z A B C D E F M : ℂ)

-- Define the similarity ratio
variable (z : ℂ)

-- State the given conditions
axiom similar_triangles : 
  (D - B) / (A - B) = (E - A) / (C - A) ∧ 
  (F - E) / (D - E) = (X - Y) / (Z - Y) ∧
  (D - B) / (A - B) = z

axiom M_midpoint : M = (B + C) / 2

-- State the theorem to be proved
theorem triangle_similarity_implications :
  (abs (Complex.arg ((F - A) / (M - A))) = abs (Complex.arg ((X - Y) / (Z - Y)) - Complex.arg ((X - Z) / (Y - Z)))) ∧
  (Complex.abs ((F - A) / (M - A)) = 2 * Complex.abs (X - Y) * Complex.abs (X - Z) / (Complex.abs (Y - Z))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_implications_l986_98615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_test_count_l986_98614

theorem joes_test_count (initial_avg new_avg lowest_score : ℝ) : 
  initial_avg = 60 →
  new_avg = 65 →
  lowest_score = 45 →
  ∃ n : ℕ, n > 1 ∧
    (n : ℝ) * initial_avg - lowest_score = (n - 1 : ℝ) * new_avg ∧
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_test_count_l986_98614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoebox_width_is_six_l986_98663

/-- A shoebox with a square block inside -/
structure Shoebox where
  height : ℝ
  blockSide : ℝ
  uncoveredArea : ℝ

/-- The width of the shoebox -/
noncomputable def shoeboxWidth (box : Shoebox) : ℝ :=
  (box.blockSide * box.blockSide + box.uncoveredArea) / box.height

/-- Theorem stating the width of the shoebox is 6 inches -/
theorem shoebox_width_is_six (box : Shoebox) 
    (h1 : box.height = 4)
    (h2 : box.blockSide = 4)
    (h3 : box.uncoveredArea = 8) : 
  shoeboxWidth box = 6 := by
  sorry

#eval "Shoebox theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoebox_width_is_six_l986_98663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l986_98686

/-- The internal angle of a regular n-gon --/
noncomputable def internal_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

theorem regular_polygon_sides :
  ∀ n : ℕ, n > 2 →
  internal_angle (2 * n) = internal_angle n + 15 →
  n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l986_98686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l986_98620

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^2))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ∈ Set.univ} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l986_98620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l986_98666

theorem proper_subsets_count (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  (M ∩ N = {2}) →
  (Finset.powerset ((M ∪ N).toFinite.toFinset)).card - 1 = 7 := by
  intros M N h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l986_98666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l986_98653

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the second derivative of f
def f'' : ℝ → ℝ := sorry

-- State the theorem
theorem inequality_solution_set 
  (h1 : f 1 = Real.exp 1)
  (h2 : ∀ x : ℝ, f'' x > f x) :
  ∀ x : ℝ, f x < Real.exp x ↔ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l986_98653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dilation_triangle_area_l986_98637

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a dilation transformation -/
def dilate (center : Point) (factor : ℝ) (p : Point) : Point :=
  { x := center.x + factor * (p.x - center.x)
  , y := center.y + factor * (p.y - center.y) }

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem square_dilation_triangle_area :
  ∀ (original : Square) (dilated : Square),
    original.a = { x := 0, y := 0 } →
    original.b = { x := 1, y := 0 } →
    original.c = { x := 1, y := 1 } →
    original.d = { x := 0, y := 1 } →
    (∃ k : ℝ, dilated.a = dilate original.a k original.a ∧
              dilated.b = dilate original.a k original.b ∧
              dilated.c = dilate original.a k original.c ∧
              dilated.d = dilate original.a k original.d) →
    distance original.b dilated.c = 29 →
    triangleArea original.b original.d dilated.c = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dilation_triangle_area_l986_98637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_identity_l986_98626

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C : V)

theorem vector_identity : A - B - 2 • (A - C) + (B - C) = C - A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_identity_l986_98626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_cant_meet_on_street_l986_98631

-- Define the structure of the city
structure City where
  intersections : Type
  streets : intersections → intersections → Prop

-- Define the movement rules
inductive Move
  | straight
  | turnLeft
  | turnRight

-- Define the state of a car
structure CarState (city : City) where
  position : city.intersections
  direction : city.intersections

-- Define the rules for updating car state
def updateCarState {city : City} (state : CarState city) (move : Move) : CarState city :=
  sorry

-- Define the concept of two cars meeting on a street
def meetOnStreet {city : City} (car1 car2 : CarState city) : Prop :=
  sorry

-- Theorem statement
theorem cars_cant_meet_on_street (city : City) (initialState : CarState city) :
  ∀ (moves1 moves2 : List Move),
    let finalState1 := moves1.foldl updateCarState initialState
    let finalState2 := moves2.foldl updateCarState initialState
    ¬(meetOnStreet finalState1 finalState2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_cant_meet_on_street_l986_98631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l986_98636

def digit_set : Finset Nat := {2, 3, 5, 6}

def digit_count : Nat → Nat
  | 2 => 1
  | 3 => 1
  | 5 => 3
  | 6 => 2
  | _ => 0

def is_valid_number (n : Nat) : Bool :=
  n ≥ 100 ∧ n ≤ 999 ∧
  List.all [0, 1, 2] (fun d =>
    let digit := (n / (10 ^ d)) % 10
    digit ∈ digit_set ∧
    (List.filter (· = digit) [n % 10, (n / 10) % 10, (n / 100) % 10]).length ≤ digit_count digit
  )

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 1000)).card = 43 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l986_98636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_merchants_l986_98645

-- Define the type for people in the company
inductive Person : Type
  | A | B | C | D | E | F | G
deriving Repr, DecidableEq, Fintype

-- Define the occupation type
inductive Occupation : Type
  | Engineer
  | Merchant
deriving Repr, DecidableEq

-- Define a function to determine if a person is an engineer or merchant
def occupation : Person → Occupation
  | Person.F => Occupation.Engineer
  | Person.G => Occupation.Engineer
  | _ => Occupation.Merchant  -- We'll assume others are merchants by default

-- Define a function to determine if a statement is true or false based on the person's occupation
def isTruthful (p : Person) (statement : Prop) : Prop :=
  match occupation p with
  | Occupation.Engineer => statement
  | Occupation.Merchant => ¬statement

-- Define the chain of statements
def chainOfStatements : Prop :=
  isTruthful Person.A (
    isTruthful Person.B (
      isTruthful Person.C (
        isTruthful Person.D (
          isTruthful Person.E (
            isTruthful Person.F (occupation Person.G ≠ Occupation.Engineer)
          )
        )
      )
    )
  )

-- Define C's statement about D
def cStatementAboutD : Prop :=
  isTruthful Person.C (occupation Person.D = Occupation.Merchant)

-- Theorem to prove
theorem number_of_merchants :
  occupation Person.A = Occupation.Merchant ∧
  chainOfStatements ∧
  cStatementAboutD →
  (∃ (n : Nat), n = 3 ∧ 
    n = (Finset.filter (λ p : Person => occupation p = Occupation.Merchant) Finset.univ).card) :=
by
  sorry

#eval Finset.filter (λ p : Person => occupation p = Occupation.Merchant) Finset.univ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_merchants_l986_98645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_when_absolute_log_equal_l986_98638

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_absolute_log_equal
  (a b : ℝ)
  (h1 : a ≠ b)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : f a = f b) :
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_one_when_absolute_log_equal_l986_98638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_defined_iff_m_gt_six_l986_98661

/-- The function c(x) parameterized by m -/
noncomputable def c (m : ℝ) (x : ℝ) : ℝ := (3 * x^2 - 4 * x + m) / (7 * x^2 + m - 6)

/-- Theorem stating that c(x) is defined for all real x iff m > 6 -/
theorem c_defined_iff_m_gt_six (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, c m x = y) ↔ m > 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_defined_iff_m_gt_six_l986_98661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_superabundant_numbers_l986_98609

/-- h(n) is the product of all divisors of n -/
def h (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

/-- A number n is superabundant if h(h(n)) = n^2 + 2n -/
def is_superabundant (n : ℕ) : Prop :=
  h (h n) = n^2 + 2*n

theorem no_superabundant_numbers :
  ¬∃ n : ℕ, n > 0 ∧ is_superabundant n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_superabundant_numbers_l986_98609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l986_98617

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := ((x + 5) / 3) ^ (1/3 : ℝ)

-- State the theorem
theorem g_equality (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -65/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l986_98617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equals_set_l986_98674

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 3}

theorem complement_intersection_equals_set : 
  (U \ M) ∩ (U \ N) = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_equals_set_l986_98674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_is_hundred_percent_l986_98612

/-- Represents the cost price of an article -/
noncomputable def cost_price : ℝ := sorry

/-- Represents the selling price of an article -/
noncomputable def selling_price : ℝ := sorry

/-- The condition that the cost of 20 articles equals the selling price of 10 articles -/
def price_condition : Prop := 20 * cost_price = 10 * selling_price

/-- Calculate the gain percent -/
noncomputable def gain_percent : ℝ := (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that under the given condition, the gain percent is 100% -/
theorem gain_is_hundred_percent (h : price_condition) : gain_percent = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_is_hundred_percent_l986_98612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digital_music_library_space_l986_98677

/-- Calculates the average disk space per hour of music in a digital library -/
def avgDiskSpacePerHour (daysOfMusic : ℕ) (totalDiskSpace : ℕ) : ℕ :=
  let hoursOfMusic := daysOfMusic * 24
  let exactAvg : ℚ := totalDiskSpace / hoursOfMusic
  (exactAvg + 1/2).floor.toNat

theorem digital_music_library_space (daysOfMusic : ℕ) (totalDiskSpace : ℕ) 
  (h1 : daysOfMusic = 15) (h2 : totalDiskSpace = 20000) :
  avgDiskSpacePerHour daysOfMusic totalDiskSpace = 56 := by
  sorry

#eval avgDiskSpacePerHour 15 20000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digital_music_library_space_l986_98677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_sum_l986_98657

theorem unit_digit_sum (Q : ℕ) : 
  (8^Q + 7^(10*Q) + 6^(100*Q) + 5^(1000*Q)) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_sum_l986_98657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_simplest_l986_98616

noncomputable def expr1 : ℝ := Real.sqrt (3/2)
noncomputable def expr2 : ℝ := Real.sqrt 2
noncomputable def expr3 : ℝ := Real.sqrt 18
noncomputable def expr4 : ℝ := Real.sqrt 0.2

-- Define a predicate for being a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ), x = Real.sqrt a

-- Define a predicate for being the simplest form of a quadratic radical
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  is_quadratic_radical x ∧
  ∀ (y : ℝ), is_quadratic_radical y → (∃ (n : ℕ), y = n * x ∨ y = (1/n : ℚ) * x)

-- State the theorem
theorem sqrt_2_simplest :
  is_simplest_quadratic_radical expr2 ∧
  ¬is_simplest_quadratic_radical expr1 ∧
  ¬is_simplest_quadratic_radical expr3 ∧
  ¬is_simplest_quadratic_radical expr4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_simplest_l986_98616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l986_98622

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (π / 2 + α) = -3 / 5) 
  (h2 : α > -π ∧ α < 0) : 
  Real.tan α = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l986_98622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_six_balls_four_boxes_l986_98618

-- Define the number_of_distributions function
def number_of_distributions (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

-- Theorem for the general case
theorem ball_distribution (n : ℕ) (k : ℕ) : 
  number_of_distributions n k = Nat.choose (n - 1) (k - 1) := by
  rfl

-- Define the specific problem
def balls : ℕ := 6
def boxes : ℕ := 4

-- Theorem for the specific problem
theorem six_balls_four_boxes : 
  number_of_distributions balls boxes = 10 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_six_balls_four_boxes_l986_98618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l986_98623

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway : ℚ
  city : ℚ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def maxDistance (efficiency : SUVFuelEfficiency) (fuel : ℚ) : ℚ :=
  max (efficiency.highway * fuel) (efficiency.city * fuel)

/-- Theorem: The maximum distance an SUV can be driven on 25 gallons of gasoline is 305 miles -/
theorem suv_max_distance :
  let efficiency : SUVFuelEfficiency := ⟨12.2, 7.6⟩
  let availableFuel : ℚ := 25
  maxDistance efficiency availableFuel = 305 := by
  sorry

#eval maxDistance ⟨12.2, 7.6⟩ 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l986_98623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_wasted_is_48_l986_98604

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a circle -/
structure Circle where
  radius : ℝ

/-- Calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Calculate the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Calculate the area of the largest rectangle that can be inscribed in a circle -/
def largestInscribedRectangleArea (c : Circle) : ℝ := 2 * c.radius^2

/-- Calculate the total metal wasted in the cutting process -/
noncomputable def metalWasted (plate : Rectangle) (cut : Circle) : ℝ :=
  rectangleArea plate - circleArea cut + circleArea cut - largestInscribedRectangleArea cut

/-- Theorem stating that the metal wasted is 48 cm² -/
theorem metal_wasted_is_48 (plate : Rectangle) (cut : Circle) :
  plate.length = 10 ∧ plate.width = 8 ∧ cut.radius = 4 →
  metalWasted plate cut = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_wasted_is_48_l986_98604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_of_translated_sqrt_l986_98685

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) - 3

-- State the theorem
theorem min_point_of_translated_sqrt (x : ℝ) :
  x ≥ 4 → f x ≥ f 4 ∧ f 4 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_point_of_translated_sqrt_l986_98685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_eq_fibonacci_l986_98689

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sum of binomial coefficients on a diagonal of Pascal's triangle -/
def diagonalSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => binomial (n - k) k)

/-- The sum of binomial coefficients on a diagonal of Pascal's triangle
    is equal to the (n+1)th Fibonacci number -/
theorem diagonal_sum_eq_fibonacci (n : ℕ) :
  diagonalSum n = fib (n + 1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sum_eq_fibonacci_l986_98689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_variance_transformation_l986_98629

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

def transform (xs : List ℝ) (k : ℝ) : List ℝ :=
  xs.map (fun x => x + k)

theorem mean_variance_transformation (xs : List ℝ) (k : ℝ) :
  mean (transform xs k) = mean xs + k ∧
  variance (transform xs k) = variance xs := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_variance_transformation_l986_98629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l986_98607

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1/2 else 2^x

-- Theorem statement
theorem f_composition_half : f (f (1/2)) = 2 := by
  -- Evaluate f(1/2)
  have h1 : f (1/2) = 1 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(1/2))
  have h2 : f (f (1/2)) = f 1 := by
    rw [h1]
  
  -- Simplify f(1)
  have h3 : f 1 = 2 := by
    simp [f]
  
  -- Combine the steps
  rw [h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l986_98607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_pair_is_five_seven_l986_98659

/-- Definition of the sequence of positive integer pairs -/
def pairSequence : ℕ → (ℕ × ℕ)
| 0 => (1, 1)  -- Adding the case for 0
| n + 1 =>
  let sum := (n + 2).sqrt + 1
  let prev := pairSequence n
  let first := sum - prev.2
  if first > sum - 1 then (1, sum - 1) else (first, sum - first)

/-- Theorem: The 60th pair in the sequence is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  pairSequence 59 = (5, 7) := by
  sorry

#eval pairSequence 59  -- This will evaluate the 60th pair (0-indexed)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_pair_is_five_seven_l986_98659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_cross_product_equation_l986_98693

open BigOperators

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def CrossProduct : V → V → V := sorry

notation a " × " b => CrossProduct a b

theorem vector_cross_product_equation 
  (a b c : V) (h : a + b + (2 : ℝ) • c = 0) :
  ∃! k : ℝ, k • (b × a) + 2 • (b × c) + (c × a) = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_cross_product_equation_l986_98693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_arithmetic_sequence_l986_98601

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem minimize_sum_arithmetic_sequence 
  (a₁ : ℝ) 
  (h₁ : a₁ = -9) 
  (h₂ : ∃ d : ℝ, arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 5 = -6) :
  ∃ n : ℕ, n = 5 ∧ 
    ∀ m : ℕ, arithmetic_sum a₁ (Classical.choose h₂) n ≤ arithmetic_sum a₁ (Classical.choose h₂) m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_arithmetic_sequence_l986_98601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l986_98610

noncomputable def hyperbola_center : ℝ × ℝ := (1, -3)
noncomputable def hyperbola_focus : ℝ × ℝ := (1 + 3 * Real.sqrt 5, -3)
noncomputable def hyperbola_vertex : ℝ × ℝ := (4, -3)

noncomputable def h : ℝ := hyperbola_center.1
noncomputable def k : ℝ := hyperbola_center.2

noncomputable def a : ℝ := |hyperbola_vertex.1 - hyperbola_center.1|
noncomputable def c : ℝ := |hyperbola_focus.1 - hyperbola_center.1|
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l986_98610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniformity_comparison_l986_98679

/-- Represents a class of students with their test score variance -/
structure StudentClass where
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one class has more uniform scores than another -/
def more_uniform (c1 c2 : StudentClass) : Prop :=
  c1.variance < c2.variance

theorem uniformity_comparison (A B : StudentClass) 
  (h : A.variance = 13.2 ∧ B.variance = 26.26) : 
  more_uniform A B := by
  unfold more_uniform
  rw [h.1, h.2]
  norm_num
  
#check uniformity_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniformity_comparison_l986_98679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_count_l986_98699

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Square
  | RectangleNotSquare
  | RhombusNotSquare
  | KiteNotRhombus
  | IsoscelesTrapezoidNotParallelogram
deriving Repr, BEq, Inhabited

-- Define a predicate for having a point equidistant from all vertices
def has_equidistant_point (q : Quadrilateral) : Bool :=
  match q with
  | Quadrilateral.Square => true
  | Quadrilateral.RectangleNotSquare => true
  | Quadrilateral.RhombusNotSquare => false
  | Quadrilateral.KiteNotRhombus => false
  | Quadrilateral.IsoscelesTrapezoidNotParallelogram => true

-- Define a list of all quadrilateral types
def all_quadrilaterals : List Quadrilateral :=
  [Quadrilateral.Square, Quadrilateral.RectangleNotSquare, Quadrilateral.RhombusNotSquare,
   Quadrilateral.KiteNotRhombus, Quadrilateral.IsoscelesTrapezoidNotParallelogram]

-- Theorem statement
theorem equidistant_point_count :
  (all_quadrilaterals.filter has_equidistant_point).length = 3 := by
  -- The proof goes here
  sorry

#eval all_quadrilaterals.filter has_equidistant_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_count_l986_98699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_score_l986_98632

/-- Calculates the average score for the last 3 matches in a cricket series -/
theorem cricket_average_score (total_matches : ℕ) (all_matches_avg : ℝ) 
  (first_five_avg : ℝ) (next_four_avg : ℝ) : ℝ :=
  by
  -- Assume the given conditions
  have h1 : total_matches = 12 := by sorry
  have h2 : all_matches_avg = 62 := by sorry
  have h3 : first_five_avg = 52 := by sorry
  have h4 : next_four_avg = 58 := by sorry

  -- Calculate total runs for all matches
  let total_runs := all_matches_avg * (total_matches : ℝ)

  -- Calculate total runs for first 5 matches
  let first_five_runs := first_five_avg * 5

  -- Calculate total runs for next 4 matches
  let next_four_runs := next_four_avg * 4

  -- Calculate total runs for last 3 matches
  let last_three_runs := total_runs - (first_five_runs + next_four_runs)

  -- Calculate average score for last 3 matches
  let last_three_avg := last_three_runs / 3

  -- Prove that the average score for the last 3 matches is 84
  have h5 : last_three_avg = 84 := by sorry

  exact last_three_avg

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_score_l986_98632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_not_q_l986_98683

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Define proposition p
def p : Prop := ∀ x > 0, f x ≥ 2

-- Define proposition q
def q : Prop := ∃ x₀ < 0, f x₀ ≤ -2

-- Theorem to prove
theorem p_or_not_q : p ∨ (¬q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_or_not_q_l986_98683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l986_98660

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := Real.log x

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Assume P is on curve1 and Q is on curve2
axiom P_on_curve1 : curve1 P.1 = P.2
axiom Q_on_curve2 : curve2 Q.1 = Q.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance_PQ : 
  ∃ (min_dist : ℝ), ∀ (P Q : ℝ × ℝ), 
    curve1 P.1 = P.2 → curve2 Q.1 = Q.2 → 
    distance P Q ≥ min_dist ∧ 
    ∃ (P' Q' : ℝ × ℝ), curve1 P'.1 = P'.2 ∧ curve2 Q'.1 = Q'.2 ∧ 
    distance P' Q' = min_dist ∧ min_dist = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l986_98660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruent_one_mod_five_l986_98694

/-- The sequence of numbers from 3 to 193 with a step of 10 -/
def mySequence : List ℕ := List.range 20 |>.map (λ n => 3 + 10 * n)

/-- The product of all numbers in the sequence -/
def myProduct : ℕ := mySequence.prod

/-- Theorem: The product of the sequence is congruent to 1 modulo 5 -/
theorem product_congruent_one_mod_five : myProduct ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruent_one_mod_five_l986_98694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_difference_l986_98644

theorem repeating_decimal_sum_difference : 
  (234 / 999 : ℚ) + (345 / 999 : ℚ) - (123 / 999 : ℚ) = 152 / 333 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_difference_l986_98644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_arithmetic_properties_l986_98650

theorem integer_arithmetic_properties :
  ((1 : ℤ) * (-21) + 12 = -9) ∧
  ((-52) + (-19) = -71) ∧
  (-1 - 1 = -2) ∧
  (-8 - 4 = -12) ∧
  (-2 - (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_arithmetic_properties_l986_98650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l986_98669

/-- Definition of the first line passing through A(0, 0) -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0

/-- Definition of the second line passing through B(1, 3) -/
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

/-- Definition of point A -/
def pointA : ℝ × ℝ := (0, 0)

/-- Definition of point B -/
def pointB : ℝ × ℝ := (1, 3)

/-- Definition of distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_range :
  ∀ m : ℝ, ∀ P : ℝ × ℝ,
  (line1 m P.1 P.2 ∧ line2 m P.1 P.2) →
  Real.sqrt 10 ≤ distance P pointA + distance P pointB ∧
  distance P pointA + distance P pointB ≤ 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l986_98669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_exceeds_wayne_in_2004_l986_98643

def steve_money (year : ℕ) : ℚ :=
  100 * 2^(year - 2000)

def wayne_money (year : ℕ) : ℚ :=
  10000 / 2^(year - 2000)

def first_year_steve_exceeds_wayne : ℕ :=
  2004

theorem steve_exceeds_wayne_in_2004 :
  (∀ y : ℕ, 2000 ≤ y ∧ y < 2004 → steve_money y ≤ wayne_money y) ∧
  steve_money first_year_steve_exceeds_wayne > wayne_money first_year_steve_exceeds_wayne :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_exceeds_wayne_in_2004_l986_98643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l986_98665

-- Define the regular hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the right pyramid
def right_pyramid (base : Set (ℝ × ℝ)) (height : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

-- Define the volume of a pyramid
noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1/3) * base_area * height

-- Theorem statement
theorem pyramid_volume_theorem :
  ∀ (base : Set (ℝ × ℝ)) (height : ℝ),
    base = regular_hexagon 10 →
    height = 10 →
    ∃ (pyramid : Set (ℝ × ℝ × ℝ)),
      pyramid = right_pyramid base height ∧
      pyramid_volume (150 * Real.sqrt 3) height = 500 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l986_98665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l986_98648

/-- Arithmetic sequence sum -/
def S (n : ℕ) : ℚ := sorry

/-- Arithmetic sequence -/
def a : ℕ → ℚ := sorry

theorem arithmetic_sequence_sum :
  (S 17 = 170) →
  (a 2000 = 2001) →
  (S 2008 = 2019044) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l986_98648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_solution_l986_98684

/-- Quadratic function type -/
def QuadraticFunction (a h k : ℝ) : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k

/-- Definition of the problem setup -/
structure QuadraticProblem where
  a : ℝ
  h : ℝ
  k : ℝ
  m : ℝ
  n : ℝ
  f : ℝ → ℝ
  a_neg : a < 0
  m_neg : m < 0
  n_pos : 0 < n
  point_A : f (-3) = m
  point_B : f (-1) = n
  point_C : f 1 = 0
  f_def : f = QuadraticFunction a h k

/-- Main theorem encapsulating all parts of the problem -/
theorem quadratic_problem_solution (p : QuadraticProblem) :
  (p.m = p.n → ((((-3) - (-1))^2 : ℝ)^(1/2) = 2 ∧ p.h = -2)) ∧
  (∃ x', x' = 2 * p.h - 1 ∧ p.f x' = 0 ∧ -1 < p.h ∧ p.h < 0) ∧
  (p.a = -1 → ∃ area : ℝ, area = 8 ∧ area = abs ((1 - (-3)) * (0 - p.m) / 2)) ∧
  (∃ a : ℝ, a = -1/4 ∧
    ∀ x₁ x₂ x₃ : ℝ,
      (∃ y₁, QuadraticFunction a p.h p.k x₁ = y₁) →
      (∃ y₂, QuadraticFunction a p.h p.k x₂ = y₂) →
      (∃ y₃, y₃ = -2 * a * p.h * x₃ + 2 * a * p.h) →
      (x₁ + x₂ - x₃ = -1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_solution_l986_98684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_separation_time_for_given_speeds_l986_98655

/-- The time it takes for two people traveling perpendicular to each other
    to be 100 miles apart, given their speeds. -/
noncomputable def separation_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : ℝ :=
  distance / Real.sqrt (speed1^2 + speed2^2)

/-- Theorem stating that the separation time for two people traveling
    at 10 mph and 8 mph to be 100 miles apart is 100 / √164. -/
theorem separation_time_for_given_speeds :
  separation_time 10 8 100 = 100 / Real.sqrt 164 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_separation_time_for_given_speeds_l986_98655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l986_98654

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * Real.sin x + 4 * Real.sqrt 3 * Real.cos x

theorem f_max_value : ∀ x : ℝ, f x ≤ 17 / 2 := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l986_98654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_from_lcm_and_ratio_l986_98602

theorem gcd_from_lcm_and_ratio (A B : ℕ) : 
  Nat.lcm A B = 120 → 
  A = 3 * (B / 4) → 
  Nat.gcd A B = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_from_lcm_and_ratio_l986_98602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_cost_is_correct_l986_98656

/-- The cost of a movie ticket given the following conditions:
  * Fred bought 2 tickets
  * Fred borrowed a movie for $6.79
  * Fred paid with a $20 bill
  * Fred received $1.37 in change
-/
noncomputable def movie_ticket_cost : ℚ :=
  let total_paid : ℚ := 20 - (137/100)
  let borrowed_movie_cost : ℚ := 679/100
  let num_tickets : ℕ := 2
  (total_paid - borrowed_movie_cost) / num_tickets

theorem movie_ticket_cost_is_correct :
  movie_ticket_cost = 592/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_cost_is_correct_l986_98656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_cone_B_is_5_l986_98670

/-- The height of cone B in a cylindrical tube with two cones A and B -/
noncomputable def height_of_cone_B : ℝ := 5

/-- The length of the cylindrical tube -/
noncomputable def tube_length : ℝ := 20

/-- The diameter of the cylindrical tube -/
noncomputable def tube_diameter : ℝ := 6

/-- The radius of the cylindrical tube -/
noncomputable def tube_radius : ℝ := tube_diameter / 2

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of cone A -/
noncomputable def volume_A : ℝ := cone_volume tube_radius (tube_length - height_of_cone_B)

/-- The volume of cone B -/
noncomputable def volume_B : ℝ := cone_volume tube_radius height_of_cone_B

/-- Theorem stating that the height of cone B is 5 -/
theorem height_of_cone_B_is_5 :
  height_of_cone_B = 5 ∧
  tube_length = 20 ∧
  tube_diameter = 6 ∧
  volume_A / volume_B = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_cone_B_is_5_l986_98670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_bottom_vertex_probability_l986_98633

structure Dodecahedron :=
  (vertices : Finset Nat)
  (top_vertices : Finset Nat)
  (bottom_vertices : Finset Nat)
  (adjacent : Nat → Finset Nat)

def random_walk (d : Dodecahedron) (start : Nat) : Nat → Nat
  | 0 => start
  | n + 1 => sorry -- Implement random selection from adjacent vertices

theorem dodecahedron_bottom_vertex_probability 
  (d : Dodecahedron)
  (start : Nat)
  (h_start : start ∈ d.top_vertices)
  (h_adj : ∀ v, v ∈ d.vertices → (d.adjacent v).card = 3)
  (h_bottom : d.bottom_vertices.card = 5) :
  (1 : ℝ) / 3 = 1 / 3 := by
  sorry

#check dodecahedron_bottom_vertex_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_bottom_vertex_probability_l986_98633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_properties_max_product_inequality_range_min_value_function_l986_98678

-- Part 1
theorem inequality_properties (a b : ℝ) (h1 : b < a) (h2 : a < 0) :
  a + b < a * b :=
sorry

-- Part 2
theorem max_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 :=
sorry

-- Part 3
theorem inequality_range (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 4) :
  1 / x + 4 / y ≥ 9 / 4 :=
sorry

-- Part 4
noncomputable def f (x : ℝ) : ℝ := (x + 1 / x)^2 - 1

theorem min_value_function (x : ℝ) (h : x ≠ 0) :
  f x ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_properties_max_product_inequality_range_min_value_function_l986_98678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AEC_l986_98651

noncomputable section

-- Define the square
def square_side : ℝ := 1

-- Define the points
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (1, 1)
def C' : ℝ × ℝ := (1, 0.25)

-- Define E as the intersection of BC and AB
def E : ℝ × ℝ := (4/7, 4/7)

-- Define the condition C'D = 1/4
def C'D_length : ℝ := 1/4

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem perimeter_of_triangle_AEC' :
  let perimeter := distance A E + distance E C' + distance C' A
  ∃ ε > 0, abs (perimeter - 2.1) < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AEC_l986_98651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_domain_range_minimum_l986_98692

noncomputable def f (x : ℝ) := Real.cos x

theorem cos_domain_range_minimum (a b : ℝ) :
  (∀ x, x ∈ Set.Icc a b → f x ∈ Set.Icc (-1/2) 1) →
  (∀ y, y ∈ Set.Icc (-1/2) 1 → ∃ x ∈ Set.Icc a b, f x = y) →
  b - a ≥ 2*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_domain_range_minimum_l986_98692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l986_98625

theorem alpha_plus_beta_value (α β : Real) :
  Real.cos (α - β) = Real.sqrt 5 / 5 →
  Real.cos (2 * α) = Real.sqrt 10 / 10 →
  0 < α → α < π/2 →
  0 < β → β < π/2 →
  α < β →
  α + β = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l986_98625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_and_terms_l986_98639

def polynomial (x y : ℝ) : ℝ := x * y^3 - x^2 + 7

-- Define polynomial_degree function
def polynomial_degree (p : (ℝ → ℝ → ℝ)) : ℕ := sorry

-- Define number_of_terms function
def number_of_terms (p : (ℝ → ℝ → ℝ)) : ℕ := sorry

theorem polynomial_degree_and_terms :
  (polynomial_degree polynomial) = 4 ∧ (number_of_terms polynomial) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_and_terms_l986_98639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_using_1_0_7_l986_98667

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def uses_digits_1_0_7 (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ ({a, b, c} : Finset ℕ) = {1, 0, 7}

theorem smallest_three_digit_using_1_0_7 :
  ∀ n : ℕ, (is_three_digit n ∧ uses_digits_1_0_7 n) → n ≥ 107 :=
by
  intro n ⟨h_three_digit, h_uses_digits⟩
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_using_1_0_7_l986_98667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainders_sum_quotient_l986_98652

theorem square_remainders_sum_quotient : 
  let remainders : Finset ℕ := (Finset.range 15).image (fun n => (n + 1)^2 % 16)
  let m : ℕ := remainders.sum id
  m / 16 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainders_sum_quotient_l986_98652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_equal_area_division_trapezoid_larger_base_width_l986_98662

theorem trapezoid_equal_area_division (a b h : ℝ) (h1 : a > b) (h2 : h > 0) :
  let x := b
  let y := a
  let area := (1/2) * (a + b) * h
  let part_area := (1/3) * area
  (1/2) * (a + x) * (h/3) = part_area ∧
  (1/2) * (x + y) * (h/3) = part_area ∧
  (1/2) * (y + b) * (h/3) = part_area :=
by
  -- Introduce the local definitions
  intro x y area part_area
  
  -- Prove each part of the conjunction
  constructor
  · -- Prove (1/2) * (a + x) * (h/3) = part_area
    sorry
  constructor
  · -- Prove (1/2) * (x + y) * (h/3) = part_area
    sorry
  · -- Prove (1/2) * (y + b) * (h/3) = part_area
    sorry

-- The width of the part adjacent to the larger base is a
theorem trapezoid_larger_base_width (a b h : ℝ) (h1 : a > b) (h2 : h > 0) :
  let x := b
  let y := a
  y = a :=
by
  -- Introduce the local definitions
  intro x y
  -- The result follows directly from the definition of y
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_equal_area_division_trapezoid_larger_base_width_l986_98662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l986_98696

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if vectors m and n are parallel, c = √3, and the area of the triangle is √15/4,
    then cos C = 1/4 and a = b = √2 -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (Real.cos B, Real.cos C)
  let n : ℝ × ℝ := (4 * a - b, c)
  (∃ (k : ℝ), m = k • n) →  -- This replaces Vector.IsParallel
  c = Real.sqrt 3 →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 / 4 →
  Real.cos C = 1 / 4 ∧ a = Real.sqrt 2 ∧ b = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l986_98696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_max_l986_98672

-- Define the function f(x) = x^2 * e^x
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

-- State the theorem
theorem f_monotonicity_and_max :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧  -- f is increasing on (-∞, -2)
  (∀ x y, -2 < x ∧ x < y ∧ y < 0 → f x > f y) ∧  -- f is decreasing on (-2, 0)
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧  -- f is increasing on (0, +∞)
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ 4 * Real.exp 2) := by  -- max value on [-2, 2] is 4e^2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_max_l986_98672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_2_f_nonnegative_characterization_l986_98649

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

-- Theorem 1: f is increasing when a = 2
theorem f_increasing_when_a_2 : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f 2 x₁ < f 2 x₂ := by
  sorry

-- Theorem 2: Characterization of a for which f(x) ≥ 0 has solution set [1,+∞)
theorem f_nonnegative_characterization :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) ∧ 
           (∀ x : ℝ, 0 < x → x < 1 → f a x < 0) ↔ 
           a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_2_f_nonnegative_characterization_l986_98649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_signals_properties_l986_98697

/-- A probabilistic experiment with 6 independent trials, each with probability 1/3 of success -/
structure TrafficSignals where
  num_signals : ℕ := 6
  prob_red : ℚ := 1/3
  independent : Bool := true

/-- The probability of encountering the first red light after passing two green lights -/
def prob_first_red_after_two (ts : TrafficSignals) : ℚ :=
  (1 - ts.prob_red) * (1 - ts.prob_red) * ts.prob_red

/-- The expected number of red lights encountered -/
def expected_red_lights (ts : TrafficSignals) : ℚ :=
  ts.num_signals * ts.prob_red

/-- The variance of the number of red lights encountered -/
def variance_red_lights (ts : TrafficSignals) : ℚ :=
  ts.num_signals * ts.prob_red * (1 - ts.prob_red)

theorem traffic_signals_properties (ts : TrafficSignals) :
  prob_first_red_after_two ts = 4/27 ∧
  expected_red_lights ts = 2 ∧
  variance_red_lights ts = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_signals_properties_l986_98697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_area_one_l986_98682

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  yIntercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

-- Function to calculate the area of a triangle formed by a line and the coordinate axes
noncomputable def triangleArea (l : Line) : ℝ :=
  let xIntercept := -l.yIntercept / l.slope
  let yIntercept := l.yIntercept
  abs (xIntercept * yIntercept) / 2

-- Theorem statement
theorem line_through_point_with_area_one :
  ∃ (l1 l2 : Line),
    (pointOnLine ⟨-2, 2⟩ l1) ∧
    (triangleArea l1 = 1) ∧
    (pointOnLine ⟨-2, 2⟩ l2) ∧
    (triangleArea l2 = 1) ∧
    ((l1.slope = -1/2 ∧ l1.yIntercept = 1) ∨
     (l2.slope = -2 ∧ l2.yIntercept = -2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_area_one_l986_98682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_sqrt_seven_l986_98646

/-- Given a segment of length 7√3, it is possible to construct a segment of length √7
    using compass and straightedge constructions. -/
theorem construct_sqrt_seven (segment : ℝ) (h : segment = 7 * Real.sqrt 3) :
  ∃ (constructed : ℝ), constructed = Real.sqrt 7 ∧ 
  ∃ (f : ℝ → Prop), f segment ∧ f constructed :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_sqrt_seven_l986_98646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_distance_l986_98664

/-- The distance from Andrey's home to the airport in kilometers -/
noncomputable def distance : ℝ := 180

/-- The initial speed Andrey drives at in km/h -/
noncomputable def initial_speed : ℝ := 60

/-- The increased speed Andrey drives at in km/h -/
noncomputable def increased_speed : ℝ := 90

/-- The time in hours Andrey drives at the initial speed -/
noncomputable def initial_drive_time : ℝ := 1

/-- The time in hours Andrey would be late if he continued at the initial speed -/
noncomputable def late_time : ℝ := 1/3

/-- The time in hours Andrey arrives early with the increased speed -/
noncomputable def early_time : ℝ := 1/3

theorem airport_distance :
  distance = initial_speed * initial_drive_time +
    increased_speed * (distance / increased_speed - initial_drive_time - early_time) ∧
  distance = initial_speed * initial_drive_time +
    initial_speed * (distance / initial_speed - initial_drive_time + late_time) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airport_distance_l986_98664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_greater_than_half_l986_98676

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) / (x + 2)

theorem increasing_f_implies_a_greater_than_half (a : ℝ) :
  (∀ x y : ℝ, -2 < x ∧ x < y → f a x < f a y) →
  a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_greater_than_half_l986_98676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_midpoint_l986_98640

/-- Given a line with equation x/4 + y/3 = 1, prove its slope and midpoint of intercepts. -/
theorem line_slope_and_midpoint :
  ∃ (slope : ℝ) (midpoint : ℝ × ℝ),
    (∀ x y : ℝ, x/4 + y/3 = 1 ↔ y = slope * x + 3) ∧
    slope = -3/4 ∧
    midpoint = (2, 3/2) ∧
    (∃ x_int y_int : ℝ, x_int/4 + 0/3 = 1 ∧ 0/4 + y_int/3 = 1 ∧
      midpoint = ((x_int + 0)/2, (0 + y_int)/2)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_midpoint_l986_98640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l986_98680

theorem complex_number_quadrant (a : ℝ) :
  ∀ b : ℝ, (((a - 4 : ℂ) + 5*Complex.I) * (-b^2 + 2*b - 6)).re > 0 ∧
           (((a - 4 : ℂ) + 5*Complex.I) * (-b^2 + 2*b - 6)).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l986_98680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l986_98687

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 / 10 + 3

/-- The point of tangency -/
def x₀ : ℝ := 2

/-- The slope of the tangent line -/
def m : ℝ := (deriv f) x₀

/-- The y-coordinate of the point of tangency -/
def y₀ : ℝ := f x₀

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀

theorem tangent_line_equation :
  tangent_line = fun x => (2/5) * x + 13/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l986_98687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_is_2sqrt14_l986_98603

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the spheres
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def SphereProblem (t : Triangle) (s1 s2 s3 : Sphere) : Prop :=
  -- Spheres s1 and s2 touch the plane of triangle ABC at A and B
  (s1.center.2.1 = 0 ∧ s2.center.2.1 = 0) ∧
  -- Sum of radii of s1 and s2 is 9
  (s1.radius + s2.radius = 9) ∧
  -- Distance between centers of s1 and s2 is √305
  ((s1.center.1 - s2.center.1)^2 + (s1.center.2.1 - s2.center.2.1)^2 + (s1.center.2.2 - s2.center.2.2)^2 = 305) ∧
  -- s3 has radius 7 and center at C
  (s3.radius = 7 ∧ s3.center = (t.C.1, t.C.2, 0)) ∧
  -- s3 touches s1 and s2 externally
  ((s1.center.1 - s3.center.1)^2 + (s1.center.2.1 - s3.center.2.1)^2 + (s1.center.2.2 - s3.center.2.2)^2 = (s1.radius + s3.radius)^2) ∧
  ((s2.center.1 - s3.center.1)^2 + (s2.center.2.1 - s3.center.2.1)^2 + (s2.center.2.2 - s3.center.2.2)^2 = (s2.radius + s3.radius)^2)

-- Define the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem circumradius_is_2sqrt14 (t : Triangle) (s1 s2 s3 : Sphere) :
  SphereProblem t s1 s2 s3 → circumradius t = 2 * Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_is_2sqrt14_l986_98603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l986_98606

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem line_equation_theorem (l : Line) : 
  (pointOnLine ⟨3, 4⟩ l) ∧ 
  (distancePointToLine ⟨-2, 2⟩ l = distancePointToLine ⟨4, -2⟩ l) →
  ((l.a = 2 ∧ l.b = 3 ∧ l.c = -18) ∨ (l.a = 2 ∧ l.b = -1 ∧ l.c = -2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l986_98606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_statements_count_l986_98608

/-- Represents the number of knights on the island -/
def r : ℕ := sorry

/-- Represents the number of liars on the island -/
def l : ℕ := sorry

/-- The total number of islanders -/
def total_islanders : ℕ := r + l

/-- The number of times "You are a liar!" was said -/
def liar_statements : ℕ := 230

/-- The condition that there are at least two knights and two liars -/
axiom at_least_two : r ≥ 2 ∧ l ≥ 2

/-- The condition that knights always tell the truth and liars always lie -/
axiom truth_lie_behavior : 2 * r * l = liar_statements

/-- The total number of statements made -/
def total_statements : ℕ := total_islanders * (total_islanders - 1)

/-- The number of times "You are a knight!" was said -/
def knight_statements : ℕ := total_statements - liar_statements

/-- The main theorem to prove -/
theorem knight_statements_count : knight_statements = 526 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knight_statements_count_l986_98608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_77_n_equals_7_count_l986_98634

theorem gcd_77_n_equals_7_count : 
  ∃! k : ℕ, k = (Finset.filter (λ n : ℕ ↦ 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 77 n = 7) (Finset.range 201)).card ∧ k = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_77_n_equals_7_count_l986_98634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l986_98624

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 2) ^ (2 * time) - 1)

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem stating the principal amount given the interest difference -/
theorem principal_from_interest_difference 
  (rate : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : rate = 0.10) 
  (h2 : time = 1) 
  (h3 : interest_difference = 3.50) :
  ∃ (principal : ℝ), 
    compound_interest principal rate time - simple_interest principal rate time = interest_difference ∧ 
    principal = 1400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l986_98624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_well_defined_l986_98621

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- Define a function to represent the nested multiplication
noncomputable def nestedStar : ℕ → ℝ
  | 0 => -1
  | n + 1 => star (nestedStar n) (-(n + 2 : ℝ))

-- Theorem statement
theorem nested_star_well_defined :
  ∀ n : ℕ, n ≤ 99 → ∃ x : ℝ, nestedStar n = x ∧ x ≠ 0 :=
by
  sorry

#check nested_star_well_defined

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_well_defined_l986_98621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_cost_calculation_l986_98673

/-- The cost of a dozen pears at Clark's Food Store -/
def pear_cost : ℕ → Prop := λ x => x > 0

/-- The cost of a dozen apples at Clark's Food Store -/
def apple_cost : ℕ := 40

/-- The number of dozens of each fruit Hank bought -/
def dozens_bought : ℕ := 14

/-- The total cost of Hank's purchase -/
def total_cost : ℕ := 1260

theorem pear_cost_calculation : pear_cost 50 := by
  unfold pear_cost
  apply Nat.lt_trans (by norm_num : 0 < 1)
  norm_num

#check pear_cost_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_cost_calculation_l986_98673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l986_98647

open Matrix

theorem matrix_determinant (z : ℂ) : 
  let M : Matrix (Fin 3) (Fin 3) ℂ := !![z + 2, z, z; z, z + 2, z + 1; z, z + 1, z + 2]
  Complex.abs (det M) = Complex.abs (3 * z^2 + 9 * z + 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_l986_98647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_proof_l986_98605

/-- Represents the time (in hours) it takes for the pump to fill the tank without the leak -/
noncomputable def pump_fill_time : ℝ := 2

/-- Represents the time (in hours) it takes for the pump and leak together to fill the tank -/
noncomputable def combined_fill_time : ℝ := 15/7

/-- Represents the time (in hours) it takes for the leak to drain the full tank -/
noncomputable def leak_drain_time : ℝ := 30

/-- Proves that the pump fill time is 2 hours and satisfies the equation relating to combined fill time and leak drain time -/
theorem pump_fill_time_proof :
  pump_fill_time = 2 ∧
  (1 / pump_fill_time - 1 / leak_drain_time) = 1 / combined_fill_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_proof_l986_98605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_in_one_day_l986_98635

/-- The birth rate in people per two seconds -/
def birth_rate : ℕ := 6

/-- The death rate in people per two seconds -/
def death_rate : ℕ := 3

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The net population increase in one day -/
def net_population_increase (b d s : ℕ) : ℕ :=
  (b - d) * s / 2

theorem population_increase_in_one_day :
  net_population_increase birth_rate death_rate seconds_per_day = 259200 := by
  sorry

#eval net_population_increase birth_rate death_rate seconds_per_day

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_in_one_day_l986_98635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l986_98698

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sin (2 * x) - 1

theorem f_properties :
  (∀ x ∈ Set.Icc (π/8) (5*π/8), 
    ∀ y ∈ Set.Icc (π/8) (5*π/8), 
    x < y → f x > f y) ∧
  (∀ x : ℝ, f (π/8 - x) = f (π/8 + x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l986_98698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_real_part_of_cube_l986_98668

def options : List ℂ := [-3, -2 + Complex.I, 1 + 2*Complex.I, 2 + Complex.I, -Complex.I]

def cube_real_part (z : ℂ) : ℝ := (z^3).re

theorem greatest_real_part_of_cube :
  ∃ (w : ℂ), w ∈ options ∧ 
  ∀ (z : ℂ), z ∈ options → cube_real_part z ≤ cube_real_part w ∧
  w = 2 + Complex.I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_real_part_of_cube_l986_98668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonneg_int_combinations_span_lattice_l986_98611

/-- Represents the property that vectors do not fall into a half-space containing the origin -/
def not_in_half_space (n : ℕ) (vectors : List (Fin n → ℤ)) : Prop :=
  ∃ (v : Fin n → ℝ), ∀ a ∈ vectors, 0 < (Finset.sum Finset.univ (λ i => v i * (a i)))

/-- Calculates the GCD of all n×n minor determinants of a matrix -/
noncomputable def gcd_of_minors (m n : ℕ) (matrix : Fin m → Fin n → ℤ) : ℕ :=
  sorry  -- Implementation details omitted for brevity

theorem nonneg_int_combinations_span_lattice
  (n m : ℕ)
  (vectors : Fin m → Fin n → ℤ)
  (h_m_ge_n : m ≥ n)
  (h_not_half_space : not_in_half_space n (List.ofFn vectors))
  (h_gcd_one : gcd_of_minors m n vectors = 1) :
  (Submodule.span ℤ {v : Fin n → ℤ | ∃ (coeffs : Fin m → ℕ),
    v = λ i => (Finset.sum Finset.univ (λ j => coeffs j * vectors j i))}) = ⊤ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonneg_int_combinations_span_lattice_l986_98611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_theorem_l986_98642

/-- Calculates the percentage yield of a stock -/
noncomputable def percentage_yield (face_value : ℝ) (dividend_rate : ℝ) (market_price : ℝ) : ℝ :=
  (face_value * dividend_rate / market_price) * 100

/-- Theorem: The percentage yield of a 10% stock with face value $100 quoted at 125 is 8% -/
theorem stock_yield_theorem :
  let face_value : ℝ := 100
  let dividend_rate : ℝ := 0.10
  let market_price : ℝ := 125
  percentage_yield face_value dividend_rate market_price = 8 := by
  -- Unfold the definition of percentage_yield
  unfold percentage_yield
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_theorem_l986_98642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l986_98658

open Real

-- Define the function f
noncomputable def f (x : ℝ) := (arccos x)^4 + (arcsin x)^4

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  ∃ y ∈ Set.Icc ((π^4)/16) ((π^4)/8),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc ((π^4)/16) ((π^4)/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l986_98658
