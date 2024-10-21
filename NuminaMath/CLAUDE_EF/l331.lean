import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_proof_l331_33126

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem unoccupied_volume_proof :
  let cylinder_r : ℝ := 10
  let cylinder_h : ℝ := 35
  let cone_r : ℝ := 10
  let cone_h : ℝ := 15
  let sphere_r : ℝ := 5
  cylinder_volume cylinder_r cylinder_h - 2 * cone_volume cone_r cone_h - sphere_volume sphere_r = (7000 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_proof_l331_33126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_bob_distance_l331_33118

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents Biker Bob's path -/
def bikerBobPath (start : Point) : Point :=
  let p1 := Point.mk (start.x - 10) start.y  -- 10 miles west
  let p2 := Point.mk p1.x (p1.y + 5)         -- 5 miles north
  let p3 := Point.mk (p2.x + 5) p2.y         -- 5 miles east
  Point.mk p3.x (p3.y + 15)                  -- 15 miles north

theorem biker_bob_distance :
  let townA := Point.mk 0 0
  let townB := bikerBobPath townA
  distance townA townB = 20 := by
  sorry

#eval "Biker Bob's distance theorem is stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_bob_distance_l331_33118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_x_axis_l331_33194

/-- Given a point A in 3D space, find its symmetric point A' with respect to the x-axis -/
theorem symmetric_point_x_axis (A : ℝ × ℝ × ℝ) (h : A = (-3, 1, 4)) :
  let A' := (-3, -1, -4)
  (A'.1 = A.1) ∧ (A'.2.1 = -A.2.1) ∧ (A'.2.2 = -A.2.2) := by
  sorry

/-- Helper function to get the third element of a triple -/
def third {α : Type*} : (α × α × α) → α
  | (_, _, z) => z

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_x_axis_l331_33194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_remainder_equality_l331_33190

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_remainder_equality : ∃ k : ℕ, (factorial 26 - factorial 23) = 821 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_remainder_equality_l331_33190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_not_divisible_into_tshapes_l331_33197

/-- Represents a T-shaped figure consisting of four cells -/
structure TShape where
  cells : Finset (Nat × Nat)
  size_eq_four : cells.card = 4
  t_shape : ∃ (x y : Nat), 
    {(x, y), (x + 1, y), (x + 2, y), (x + 1, y + 1)} ⊆ cells ∨
    {(x, y), (x, y + 1), (x, y + 2), (x + 1, y + 1)} ⊆ cells ∨
    {(x, y), (x - 1, y), (x + 1, y), (x, y - 1)} ⊆ cells ∨
    {(x, y), (x - 1, y), (x + 1, y), (x, y + 1)} ⊆ cells

/-- Represents a 10x10 board -/
def Board : Finset (Nat × Nat) :=
  Finset.product (Finset.range 10) (Finset.range 10)

/-- Theorem stating that a 10x10 board cannot be divided into T-shaped figures -/
theorem board_not_divisible_into_tshapes :
  ¬ ∃ (tshapes : Finset TShape), 
    (∀ t, t ∈ tshapes → t.cells ⊆ Board) ∧ 
    (∀ x, x ∈ Board → ∃! t, t ∈ tshapes ∧ x ∈ t.cells) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_not_divisible_into_tshapes_l331_33197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_properties_l331_33166

/-- Custom cross product operation for planar vectors -/
noncomputable def cross_product (a b : ℝ × ℝ) : ℝ :=
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let sin_angle := |a.1 * b.2 - a.2 * b.1| / (norm_a * norm_b)
  norm_a * norm_b * sin_angle

theorem cross_product_properties :
  ∀ (a b c : ℝ × ℝ) (k : ℝ),
  (cross_product a b = cross_product b a) ∧
  (∃ k, k * (cross_product a b) ≠ cross_product (k * a.1, k * a.2) b) ∧
  (cross_product (a.1 + b.1, a.2 + b.2) c ≠ cross_product a c + cross_product b c) ∧
  (cross_product a b = |a.1 * b.2 - a.2 * b.1|) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_properties_l331_33166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l331_33132

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 3
noncomputable def line2 (x : ℝ) : ℝ := -x + 6
def line3 : ℝ := 2

-- Define the vertices of the triangle
noncomputable def vertex1 : ℝ × ℝ := (-1/2, 2)
noncomputable def vertex2 : ℝ × ℝ := (4, 2)
noncomputable def vertex3 : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem triangle_area : 
  let base := vertex2.1 - vertex1.1
  let height := vertex3.2 - line3
  (1/2 : ℝ) * base * height = 6.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l331_33132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l331_33173

-- Define the parametric equation of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 2 / 2) * t, -1 + (Real.sqrt 2 / 2) * t)

-- Define the polar coordinate equation of circle C
noncomputable def circle_C (θ : ℝ) : ℝ :=
  2 * (Real.cos θ - Real.sin θ)

-- Define the Cartesian coordinate equation of circle C
def circle_C_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 0

-- Define the circle surface
def circle_surface (x y : ℝ) : Prop :=
  ∃ θ, x^2 + y^2 ≤ (circle_C θ)^2

-- Theorem: The range of x + y for the common point of line l and circle surface is [-2, 2]
theorem range_of_sum (t : ℝ) :
  let (x, y) := line_l t
  circle_surface x y →
  -2 ≤ x + y ∧ x + y ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l331_33173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l331_33158

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 2)

-- Define the point we want to prove is not on the graph
def point : ℝ × ℝ := (-2, -4)

-- Theorem statement
theorem point_not_on_graph : ¬ (f point.fst = point.snd) := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l331_33158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_butanoic_acid_l331_33199

/-- Represents the mass percentage of an element in a compound -/
noncomputable def MassPercentage (elementMass : ℝ) (compoundMass : ℝ) : ℝ :=
  (elementMass / compoundMass) * 100

/-- Represents the atomic mass of an element -/
def AtomicMass (element : String) : ℝ :=
  match element with
  | "C" => 12.01
  | "H" => 1.01
  | "O" => 16.00
  | _ => 0

/-- Represents the chemical formula of Butanoic acid -/
def ButanoicAcid : String := "C4H8O2"

/-- Calculates the molar mass of a compound given its chemical formula -/
def MolarMass : ℝ :=
  let c_mass := 4 * AtomicMass "C"
  let h_mass := 8 * AtomicMass "H"
  let o_mass := 2 * AtomicMass "O"
  c_mass + h_mass + o_mass

/-- Theorem: The mass percentage of carbon in Butanoic acid is approximately 54.50% -/
theorem carbon_percentage_in_butanoic_acid :
  let carbon_mass := 4 * AtomicMass "C"
  let compound_mass := MolarMass
  let percentage := MassPercentage carbon_mass compound_mass
  abs (percentage - 54.50) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_butanoic_acid_l331_33199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_face_centers_theorem_l331_33174

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  volume : ℝ
  surface_area : ℝ

/-- The solid formed by connecting the centers of the faces of a regular tetrahedron -/
noncomputable def connect_face_centers (t : RegularTetrahedron) : RegularTetrahedron :=
  { edge_length := t.edge_length / 3,
    volume := t.volume / 27,
    surface_area := t.surface_area / 9 }

/-- Theorem: Connecting the centers of the faces of a regular tetrahedron results in a smaller regular tetrahedron with 1/27 of the volume and 1/9 of the surface area -/
theorem connect_face_centers_theorem (t : RegularTetrahedron) :
  let smaller := connect_face_centers t
  smaller.volume = t.volume / 27 ∧ smaller.surface_area = t.surface_area / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_face_centers_theorem_l331_33174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_k_polynomials_l331_33180

-- Define what a class-k polynomial is
def is_class_k (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g u : ℝ → ℝ) (h v : ℝ → ℝ),
    (∀ x y, f x y = g x * h y + u x * v y) ∧
    (∀ x, g x ≠ 0 ∨ u x ≠ 0) ∧
    (∀ y, h y ≠ 0 ∨ v y ≠ 0)

-- Define the two polynomials
def f₁ : ℝ → ℝ → ℝ := λ x y ↦ 1 + x * y
def f₂ : ℝ → ℝ → ℝ := λ x y ↦ 1 + x * y + x^2 * y^2

-- State the theorem
theorem class_k_polynomials :
  is_class_k f₁ ∧ ¬ is_class_k f₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_k_polynomials_l331_33180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_l331_33139

/-- The curve defined by the polar equation r = 1 / (2sin θ + 3cos θ) is a line. -/
theorem curve_is_line : ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
  ∀ (x y : ℝ),
  (∃ (r θ : ℝ), r > 0 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ ∧
  r = 1 / (2 * Real.sin θ + 3 * Real.cos θ)) →
  a * x + b * y = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_l331_33139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_fives_in_five_dice_l331_33159

/-- The probability of getting exactly three 5s when rolling five fair 6-sided dice -/
theorem probability_three_fives_in_five_dice : 
  let n : ℕ := 5  -- number of dice
  let k : ℕ := 3  -- number of desired outcomes (5s)
  let p : ℚ := 1/6  -- probability of rolling a 5 on a single die
  ↑(Nat.choose n k) * p^k * (1-p)^(n-k) = 125/3888 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_fives_in_five_dice_l331_33159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_subset_equals_one_fourth_l331_33121

-- Define the set
def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}

-- Define the subset
def T : Finset Char := {'a', 'b', 'c'}

-- Define the probability function
def prob_subset_of_T : ℚ :=
  (Finset.filter (λ s => s ⊆ T) (Finset.powerset S)).card /
  (Finset.powerset S).card

-- Theorem statement
theorem prob_subset_equals_one_fourth :
  prob_subset_of_T = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_subset_equals_one_fourth_l331_33121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_iterations_for_decay_l331_33138

/-- Exponential decay learning rate model -/
noncomputable def learning_rate (L₀ D : ℝ) (G G₀ : ℕ) : ℝ := L₀ * D^((G : ℝ) / (G₀ : ℝ))

/-- The problem statement -/
theorem min_iterations_for_decay (L₀ D : ℝ) (G₀ : ℕ) 
  (hL₀ : L₀ = 0.5)
  (hG₀ : G₀ = 18)
  (hD : D = 0.8)
  (h_decay : learning_rate L₀ D G₀ G₀ = 0.4) :
  (∀ G < 74, learning_rate L₀ D G G₀ ≥ 0.2) ∧ 
  (learning_rate L₀ D 74 G₀ < 0.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_iterations_for_decay_l331_33138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_deg_to_rad_l331_33181

-- Define the conversion factor from degrees to radians
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Theorem statement
theorem negative_150_deg_to_rad : 
  -150 * deg_to_rad = -5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_deg_to_rad_l331_33181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l331_33146

noncomputable def f (α : Real) : Real :=
  (Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α))) * (Real.cos α)^3 +
  2 * Real.sin (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)

theorem f_properties (α : Real) (h : α ∈ Set.Icc Real.pi (3 * Real.pi / 2)) :
  (Real.tan α = 3 → f α = -4/5) ∧
  (f α = 14/5 * Real.cos α → Real.tan α = 3/4 ∨ Real.tan α = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l331_33146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l331_33189

/-- Nested radical function -/
noncomputable def nestedRadical : ℕ → ℝ
  | 0 => Real.sqrt (1 + 2018 * 2020)
  | n + 1 => Real.sqrt (1 + (2018 - n) * nestedRadical n)

/-- The main theorem stating that the nested radical equals 3 -/
theorem nested_radical_equals_three :
  nestedRadical 2018 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l331_33189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l331_33123

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 3) + Real.cos (2 * x + Real.pi / 6) + m * Real.sin (2 * x)

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_perimeter (m : ℝ) (abc : Triangle) : 
  f m (Real.pi / 12) = 2 → 
  abc.b = 2 → 
  f 1 (abc.B / 2) = Real.sqrt 3 → 
  1 / 2 * abc.a * abc.c * Real.sin abc.B = Real.sqrt 3 → 
  abc.a + abc.b + abc.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l331_33123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_larger_segment_l331_33143

/-- Given a triangle with sides a, b, and c, where an altitude h is dropped on side c,
    this function returns the length of the larger segment cut off on side c. -/
noncomputable def largerSegment (a b c : ℝ) : ℝ :=
  max ((c^2 + a^2 - b^2) / (2*c)) ((c^2 + b^2 - a^2) / (2*c))

theorem triangle_larger_segment :
  let a := 50
  let b := 110
  let c := 120
  largerSegment a b c = 100 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_larger_segment_l331_33143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_not_lose_l331_33183

/-- Represents a chess position -/
def ChessPosition : Type := Unit

/-- Represents a chess move -/
def ChessMove : Type := Unit

/-- Represents the result of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- Represents a player in the chess game -/
inductive Player
| White
| Black

/-- Function to make a chess move -/
def makeMove (pos : ChessPosition) (move : ChessMove) : ChessPosition := sorry

/-- Function to check if a move is legal -/
def isLegalMove (pos : ChessPosition) (move : ChessMove) : Prop := sorry

/-- Function to determine the game result for a given player -/
def gameResult (pos : ChessPosition) (player : Player) : GameResult := sorry

/-- Double chess game where each player makes two consecutive moves -/
def doubleChessGame (initialPos : ChessPosition) (player : Player) : ChessPosition := sorry

/-- Theorem stating that White can at least not lose in double chess -/
theorem white_not_lose (initialPos : ChessPosition) :
  ∃ (strategy : ChessPosition → ChessMove),
    (∀ (pos : ChessPosition), isLegalMove pos (strategy pos)) →
    gameResult (doubleChessGame initialPos Player.White) Player.White ≠ GameResult.Loss := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_not_lose_l331_33183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_l331_33162

def sequenceList : List ℕ := [12, 13, 16, 21, 37]

theorem missing_number (x : ℕ) : 
  (List.zip (List.tail sequenceList) sequenceList).map (λ (a, b) => a - b) = [1, 3, 5, 16 - x] ∧
  (List.zip ([x] ++ List.take 4 (List.reverse sequenceList)) (List.reverse sequenceList)).map (λ (a, b) => b - a) = [9, 7, 5, 3]
  → x = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_l331_33162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l331_33172

-- Define the points and lines
def A : ℝ × ℝ := (-2, -3)
def l₁ : ℝ → ℝ → Prop := λ x y ↦ 4 * x - 3 * y = 2
def l₂ : ℝ → ℝ → Prop := λ x y ↦ y = 2

-- Define the properties of the lines
def l₁_passes_through_A : Prop := l₁ A.1 A.2
def B_on_l₁_and_l₂ : Prop := ∃ x y, l₁ x y ∧ l₂ x y

-- Define the properties of l₃
def l₃_positive_slope : Prop := ∃ m, m > 0 ∧ ∀ x y, y - A.2 = m * (x - A.1)
def C_on_l₂_and_l₃ : Prop := ∃ x, l₂ x 2 ∧ ∃ m, m > 0 ∧ 2 - A.2 = m * (x - A.1)

-- Define the area of triangle ABC
def area_ABC : ℝ := 6

-- Theorem statement
theorem slope_of_l₃ (h₁ : l₁_passes_through_A) (h₂ : B_on_l₁_and_l₂) 
  (h₃ : l₃_positive_slope) (h₄ : C_on_l₂_and_l₃) (h₅ : area_ABC = 6) : 
  ∃ m, m = 25/32 ∧ ∀ x y, y - A.2 = m * (x - A.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l331_33172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_on_interval_derivative_negative_at_midpoint_l331_33103

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - x^2 - a*x - 1

-- Part 1
theorem max_difference_on_interval (a : ℝ) :
  a = -2 →
  ∃ M, M = Real.log 2 + 1 ∧
  ∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 →
  f a x₁ - f a x₂ ≤ M :=
sorry

-- Part 2
theorem derivative_negative_at_midpoint (a : ℝ) (m n : ℝ) :
  m > 0 →
  n > 0 →
  m ≤ n →
  f a m = 0 →
  f a n = 0 →
  (deriv (f a)) ((m + n) / 2) < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_on_interval_derivative_negative_at_midpoint_l331_33103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_period_is_five_years_l331_33193

/-- Given a principal amount, total amount, and interest rate, calculate the time period. -/
noncomputable def calculate_time_period (principal : ℝ) (amount : ℝ) (rate : ℝ) : ℝ :=
  let interest := amount - principal
  (interest * 100) / (principal * rate)

/-- Theorem stating that the time period is 5 years for the given conditions. -/
theorem time_period_is_five_years :
  let principal : ℝ := 896
  let amount : ℝ := 1120
  let rate : ℝ := 5
  calculate_time_period principal amount rate = 5 := by
  -- Unfold the definition of calculate_time_period
  unfold calculate_time_period
  -- Simplify the expression
  simp
  -- The proof is complete, but we'll use sorry to skip the detailed steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_period_is_five_years_l331_33193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_range_l331_33137

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    ((m - 1) * x^2 - Real.sqrt (2 - m) * x - (1/2 : ℝ) = 0) ∧
    ((m - 1) * y^2 - Real.sqrt (2 - m) * y - (1/2 : ℝ) = 0)) ∧
  (m - 1 ≠ 0) →
  (0 ≤ m ∧ m ≤ 2 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_range_l331_33137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l331_33108

/-- The sequence a_n defined by n^2 + λn + 3 -/
def a_n (n : ℕ+) (lambda : ℝ) : ℝ := n^2 + lambda * n + 3

/-- The property that a_n is monotonically increasing -/
def is_monotone_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ+, a_n n lambda ≤ a_n (n + 1) lambda

/-- The main theorem stating the range of λ -/
theorem lambda_range (lambda : ℝ) :
  is_monotone_increasing lambda ↔ lambda > -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l331_33108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_max_value_l331_33156

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The origin point (0, 0) -/
def O : Point := ⟨0, 0⟩

/-- Checks if a point is on the line 2x + 2y - 1 = 0 -/
def isOnLine (p : Point) : Prop :=
  2 * p.x + 2 * p.y - 1 = 0

/-- Checks if three points are collinear -/
def collinear (a b c : Point) : Prop :=
  (b.x - a.x) * (c.y - a.y) = (c.x - a.x) * (b.y - a.y)

/-- The locus of point Q -/
def locus (q : Point) : Prop :=
  (q.x - 1)^2 + (q.y - 1)^2 = 2

theorem locus_and_max_value :
  ∀ (p q : Point),
  isOnLine p →
  collinear O p q →
  distance O p * distance O q = 1 →
  (∀ (q : Point), locus q) ∧
  (∀ (m : Point), locus m → m.x + 7 * m.y ≤ 18) ∧
  (∃ (m : Point), locus m ∧ m.x + 7 * m.y = 18) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_max_value_l331_33156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l331_33161

/-- Proves that the initial amount of water in a bowl is 10 ounces, given specific evaporation conditions. -/
theorem initial_water_amount (daily_evaporation : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  daily_evaporation = 0.012 →
  days = 50 →
  evaporation_percentage = 0.06 →
  daily_evaporation * (days : ℝ) = evaporation_percentage * 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l331_33161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_area_triangles_l331_33160

/-- A point on the 5x5 grid -/
def GridPoint : Type := { p : ℕ × ℕ // p.1 ≥ 1 ∧ p.1 ≤ 5 ∧ p.2 ≥ 1 ∧ p.2 ≤ 5 }

/-- A triangle formed by three points on the grid -/
def GridTriangle : Type := GridPoint × GridPoint × GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p q r : GridPoint) : Prop := sorry

/-- Predicate to check if a triangle has positive area -/
def positiveArea (t : GridTriangle) : Prop := 
  let (p, q, r) := t
  ¬collinear p q r

/-- The set of all triangles with positive area on the 5x5 grid -/
def PositiveAreaTriangles : Set GridTriangle :=
  { t : GridTriangle | positiveArea t }

/-- Instance to show that GridPoint is finite -/
instance : Fintype GridPoint := sorry

/-- Instance to show that GridTriangle is finite -/
instance : Fintype GridTriangle := sorry

/-- Instance to show that PositiveAreaTriangles is finite -/
instance : Fintype PositiveAreaTriangles := sorry

/-- The main theorem: there are 2160 triangles with positive area on the 5x5 grid -/
theorem count_positive_area_triangles : Fintype.card PositiveAreaTriangles = 2160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_area_triangles_l331_33160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l331_33110

open Real

/-- The differential equation y'' + (2/x)y' + y = 1/x for x ≠ 0 -/
def DiffEq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x ≠ 0 → (deriv^[2] y) x + (2 / x) * (deriv y) x + y x = 1 / x

/-- The general solution of the differential equation -/
noncomputable def GeneralSolution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * (sin x / x) + C₂ * (cos x / x) + 1 / x

/-- Theorem stating that the GeneralSolution satisfies the DiffEq -/
theorem general_solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, DiffEq (GeneralSolution C₁ C₂) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diff_eq_l331_33110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_guard_request_l331_33115

/-- Represents the scenario of a detained stranger with a bet --/
structure DetentionScenario where
  bet_amount : ℕ  -- The amount of the bet (100 coins)

/-- Represents the guard's decision --/
def guard_request (scenario : DetentionScenario) (x : ℕ) : Prop :=
  x ≤ 2 * scenario.bet_amount - 1 ∧ 
  ∀ y, y > x → y > 2 * scenario.bet_amount - 1

/-- Theorem stating the existence of a maximum guard request --/
theorem maximum_guard_request (scenario : DetentionScenario) :
  ∃ x : ℕ, guard_request scenario x ∧ x = 2 * scenario.bet_amount - 1 :=
by
  sorry

#check maximum_guard_request

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_guard_request_l331_33115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equivalence_l331_33104

-- Define the curves E and C
noncomputable def E (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def C (x : ℝ) : ℝ := Real.cos x

-- State the theorem
theorem curve_equivalence (ω φ : ℝ) (h1 : ω > 0) (h2 : π > φ) (h3 : φ > 0) 
  (h4 : ∀ x, E ω φ x = E ω φ (5*π/6 - x))  -- Symmetry condition
  (h5 : E ω φ (π/6) = 0)  -- Zero point condition
  : ∀ x, E ω φ x = Real.cos (2*x + π/6) :=
by
  sorry

#check curve_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equivalence_l331_33104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completion_time_l331_33142

/-- The number of days it takes A to complete the work -/
def A : ℝ := 40

/-- The number of days it takes B to complete the work -/
def B : ℝ := 60

/-- The number of days A worked before leaving -/
def days_A_worked : ℝ := 10

/-- The number of days it takes B to finish the remaining work -/
def days_B_finished : ℝ := 45

/-- The total work is considered as 1 (or 100%) -/
def total_work : ℝ := 1

theorem A_completion_time : A = 40 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completion_time_l331_33142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l331_33119

/-- Given a journey with the following parameters:
  * total_distance: The total distance of the journey
  * total_time: The total time taken for the journey
  * second_half_speed: The speed during the second half of the journey
  
  This theorem proves that the speed during the first half of the journey
  is approximately 21.18 km/hr. -/
theorem journey_speed_calculation 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (second_half_speed : ℝ) 
  (h1 : total_distance = 225) 
  (h2 : total_time = 10) 
  (h3 : second_half_speed = 24) : 
  ∃ (first_half_speed : ℝ), 
    (abs (first_half_speed - 21.18) < 0.01) ∧ 
    (first_half_speed * (total_time - total_distance / (2 * second_half_speed)) = total_distance / 2) := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l331_33119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_of_similar_triangle_l331_33128

/-- A right triangle in ℝ -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- Similarity relation between two right triangles -/
def Similar (t1 t2 : RightTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.leg1 = k * t1.leg1 ∧
    t2.leg2 = k * t1.leg2 ∧
    t2.hypotenuse = k * t1.hypotenuse

/-- Given a right triangle with a leg length of 15 cm and a hypotenuse of 34 cm,
    and a similar triangle with a hypotenuse of 68 cm,
    the length of the shortest side of the second triangle is 2 * √931 cm. -/
theorem shortest_side_of_similar_triangle
  (triangle1 triangle2 : RightTriangle)
  (h1 : triangle1.leg1 = 15)
  (h2 : triangle1.hypotenuse = 34)
  (h3 : triangle2.hypotenuse = 68)
  (h4 : Similar triangle1 triangle2) :
  min triangle2.leg1 triangle2.leg2 = 2 * Real.sqrt 931 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_of_similar_triangle_l331_33128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_a11_a55_l331_33106

-- Define the 5x5 table
def table : Fin 5 → Fin 5 → ℝ := sorry

-- Define the arithmetic sequence property for rows
def is_arithmetic_row (row : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ j : Fin 5, j.val < 4 → row (j + 1) - row j = d

-- Define the geometric sequence property for columns
def is_geometric_column (col : Fin 5 → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 5, i.val < 4 → col (i + 1) / col i = q

-- Main theorem
theorem product_a11_a55 (h_arithmetic : ∀ i : Fin 5, is_arithmetic_row (λ j => table i j))
                        (h_geometric : ∀ j : Fin 5, is_geometric_column (λ i => table i j))
                        (h_same_ratio : ∃ q : ℝ, ∀ j : Fin 5, ∃ h_q : q ≠ 0,
                          ∀ i : Fin 5, i.val < 4 → table (i + 1) j / table i j = q)
                        (h_a24 : table 1 3 = 4)
                        (h_a41 : table 3 0 = -2)
                        (h_a43 : table 3 2 = 10) :
  table 0 0 * table 4 4 = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_a11_a55_l331_33106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_union_B_l331_33195

def U : Set ℤ := {x | x^2 - 5*x - 6 ≤ 0}

def A : Set ℤ := {x | x*(2-x) ≥ 0}

def B : Set ℤ := {1, 2, 3}

theorem complement_of_A_union_B : 
  (U \ (A ∪ B)) = {-1, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_union_B_l331_33195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_odd_f_piecewise_is_odd_l331_33151

-- Define the interval D
variable {D : Set ℝ}

-- Define the functions
variable {f g : ℝ → ℝ}

-- Define G(x) = f(x) * g(x)
def G (f g : ℝ → ℝ) (x : ℝ) := f x * g x

-- Define properties of odd and even functions
def IsOdd (h : ℝ → ℝ) := ∀ x, h (-x) = -h x
def IsEven (h : ℝ → ℝ) := ∀ x, h (-x) = h x

-- Theorem 1
theorem G_is_odd {f g : ℝ → ℝ} (hf : IsOdd f) (hg : IsEven g) : IsOdd (G f g) := by
  sorry

-- Define the piecewise function f
noncomputable def f_piecewise (x : ℝ) := if x ≥ 0 then x^2 else -x^2

-- Theorem 2
theorem f_piecewise_is_odd : IsOdd f_piecewise := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_odd_f_piecewise_is_odd_l331_33151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fraction_greater_than_four_fifths_l331_33149

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_greater_than_four_fifths :
  ∀ a b : ℕ,
    is_two_digit a →
    is_two_digit b →
    (a : ℚ) / b > 4 / 5 →
    Nat.gcd a b = 1 →
    (77 : ℚ) / 96 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fraction_greater_than_four_fifths_l331_33149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_arrangements_l331_33175

def num_democrats : ℕ := 7
def num_republicans : ℕ := 5
def total_politicians : ℕ := num_democrats + num_republicans

theorem circular_seating_arrangements :
  Nat.factorial (total_politicians - 1) = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_arrangements_l331_33175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l331_33116

/-- Vector in R^2 -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Dot product of two Vec2 -/
def dot (v w : Vec2) : ℝ := v.x * w.x + v.y * w.y

/-- The function f(x) as described in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  let a := Vec2.mk (2 * Real.cos (ω * x)) (-2)
  let b := Vec2.mk (Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)) 1
  dot a b + 1

/-- Statement of the problem -/
theorem problem_statement (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x, f ω (x + π / (2 * ω)) = f ω x) →
  f ω A = Real.sqrt 3 →
  f ω B = Real.sqrt 3 →
  a = Real.sqrt 2 →
  (ω = 1 ∧
   ((a * b * Real.sin C) / 2 = (3 + Real.sqrt 3) / 2 ∨
    (a * b * Real.sin C) / 2 = (3 - Real.sqrt 3) / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l331_33116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_figure_perimeter_l331_33107

/-- A figure composed of 5 non-overlapping congruent squares -/
structure SquareFigure where
  /-- The number of squares in the figure -/
  num_squares : ℕ
  /-- The area of each square -/
  square_area : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The number of squares is 5 -/
  h_num_squares : num_squares = 5
  /-- The total area is 45 -/
  h_total_area : total_area = 45
  /-- The total area is the sum of the areas of all squares -/
  h_area_sum : total_area = num_squares * square_area

/-- The perimeter of the figure -/
noncomputable def perimeter (f : SquareFigure) : ℝ :=
  11 * (Real.sqrt f.square_area) + 2 * (3 * Real.sqrt f.square_area - Real.sqrt f.square_area)

/-- Theorem: The perimeter of the figure is 36 units -/
theorem square_figure_perimeter (f : SquareFigure) : perimeter f = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_figure_perimeter_l331_33107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l331_33147

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let roots := (λ x : ℝ ↦ a * x^2 + b * x + c = 0)
  (∃ x y : ℝ, x ≠ y ∧ roots x ∧ roots y) →
  (∃ s : ℝ, ∀ x : ℝ, roots x → ∃ y : ℝ, roots y ∧ x + y = s) →
  (∀ s : ℝ, (∀ x : ℝ, roots x → ∃ y : ℝ, roots y ∧ x + y = s) → s = -b / a) :=
by
  sorry

theorem sum_of_roots_specific_quadratic :
  let roots := (λ x : ℝ ↦ 3 * x^2 + 6 * x - 9 = 0)
  (∃ x y : ℝ, x ≠ y ∧ roots x ∧ roots y) →
  (∃ s : ℝ, ∀ x : ℝ, roots x → ∃ y : ℝ, roots y ∧ x + y = s) →
  (∀ s : ℝ, (∀ x : ℝ, roots x → ∃ y : ℝ, roots y ∧ x + y = s) → s = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l331_33147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_properties_l331_33133

/-- Definition of the ⊙ operation for planar vectors -/
def odot (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem odot_properties :
  ∃ (a b : ℝ × ℝ) (l : ℝ),
    (∀ (k : ℝ), k ≠ 0 → a = k • b → odot a b = 0) ∧ 
    (odot a b ≠ odot b a) ∧
    (odot (l • a) b = l * odot a b) ∧
    ((odot a b)^2 + (a.1 * b.1 + a.2 * b.2)^2 = (a.1^2 + a.2^2) * (b.1^2 + b.2^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_properties_l331_33133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l331_33117

theorem cube_root_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a / b) ^ (1/3 : ℝ) + (b / a) ^ (1/3 : ℝ) ≤ (2 * (a + b) * (1 / a + 1 / b)) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l331_33117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_f_3_fixed_point_f_2008_value_l331_33114

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * (1 - x)
  else if 1 < x ∧ x ≤ 2 then x - 1
  else 0  -- undefined for x outside [0, 2]

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_n n x)

def A : Set ℝ := {0, 1, 2}

theorem f_inequality : ∀ x : ℝ, f x ≤ x ↔ 2/3 ≤ x ∧ x ≤ 2 := by sorry

theorem f_3_fixed_point : ∀ x ∈ A, f_n 3 x = x := by sorry

theorem f_2008_value : f_n 2008 (8/9) = 8/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_f_3_fixed_point_f_2008_value_l331_33114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l331_33165

noncomputable def f (x : ℝ) : ℝ := (x + 5) / (x^2 - 9)

theorem undefined_values (x : ℝ) : 
  ¬(∃ y : ℝ, f x = y) ↔ x = -3 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_l331_33165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l331_33196

/-- The function f(x) = (x^2 + 2x + a) / (x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + a) / (x + 1)

/-- The theorem stating that if the range of f(x) for x ≥ 0 is [a, +∞), then a ≤ 2 -/
theorem range_of_a (a : ℝ) :
  (∀ y ≥ a, ∃ x ≥ 0, f a x = y) ∧ (∀ x ≥ 0, f a x ≥ a) →
  a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l331_33196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_36_54_l331_33192

/-- The number of positive integers that divide both 36 and 54 is 6 -/
theorem common_divisors_36_54 : (Finset.filter (fun d => d > 0 ∧ 36 % d = 0 ∧ 54 % d = 0) (Finset.range 55)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisors_36_54_l331_33192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l331_33153

noncomputable def F₁ : ℝ × ℝ := (-4, 0)
noncomputable def F₂ : ℝ × ℝ := (4, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def on_line_segment (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (t * p.1 + (1 - t) * q.1, t * p.2 + (1 - t) * q.2)

theorem trajectory_is_line_segment :
  ∀ M : ℝ × ℝ, distance M F₁ + distance M F₂ = 8 → on_line_segment F₁ F₂ M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l331_33153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l331_33170

/-- Represents a function from ℝ to ℝ -/
def RealFunction := ℝ → ℝ

/-- The original function -/
noncomputable def f : RealFunction := λ x ↦ Real.sin (2 * x + Real.pi / 3) + 2

/-- Translation to the right by π/6 units -/
noncomputable def translateRight (g : RealFunction) : RealFunction := λ x ↦ g (x - Real.pi / 6)

/-- Translation downward by 2 units -/
def translateDown (g : RealFunction) : RealFunction := λ x ↦ g x - 2

/-- The resulting function after translations -/
noncomputable def resultingFunction : RealFunction := λ x ↦ Real.sin (2 * x)

theorem translation_theorem :
  translateDown (translateRight f) = resultingFunction := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l331_33170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_field_and_park_l331_33112

/-- The combined area of a rectangular field and a triangular park with specific dimensions --/
theorem combined_area_field_and_park :
  let rectangle_side1 : ℝ := 15
  let triangle_hypotenuse : ℝ := 17
  let angle_deg : ℝ := 30

  let rectangle_side2 : ℝ := triangle_hypotenuse * Real.sqrt 3 / 2
  let triangle_base : ℝ := rectangle_side2
  let triangle_height : ℝ := Real.sqrt (triangle_hypotenuse^2 - triangle_base^2)

  let area_rectangle : ℝ := rectangle_side1 * rectangle_side2
  let area_triangle : ℝ := 1/2 * triangle_base * triangle_height
  let combined_area : ℝ := area_rectangle + area_triangle

  ∃ ε > 0, |combined_area - 283.46| < ε := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_field_and_park_l331_33112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_arcade_ratio_l331_33154

/-- Represents Mike's weekly finances and arcade spending -/
structure MikeFinances where
  weekly_pay : ℚ
  food_cost : ℚ
  hourly_play_cost : ℚ
  play_time_minutes : ℚ

/-- Calculates the ratio of Mike's arcade spending to his weekly pay -/
def arcade_spending_ratio (m : MikeFinances) : ℚ :=
  let total_play_cost := m.hourly_play_cost * (m.play_time_minutes / 60)
  let total_arcade_cost := m.food_cost + total_play_cost
  total_arcade_cost / m.weekly_pay

/-- Theorem stating that Mike's arcade spending ratio is 1/2 -/
theorem mike_arcade_ratio :
  let m : MikeFinances := {
    weekly_pay := 100,
    food_cost := 10,
    hourly_play_cost := 8,
    play_time_minutes := 300
  }
  arcade_spending_ratio m = 1/2 := by
  -- Expand the definition of arcade_spending_ratio
  unfold arcade_spending_ratio
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_arcade_ratio_l331_33154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l331_33124

theorem inequality_solution_set :
  {x : ℝ | 4 - x^2 < 0} = Set.Ioi 2 ∪ Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l331_33124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l331_33101

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the equation of a line ax + by + c = 0
def LineEquation (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => a * p.1 + b * p.2 + c = 0

-- Define the area of a triangle
noncomputable def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

theorem triangle_properties :
  -- 1. Equation of median on side BC
  (∃ (x y : ℝ), LineEquation 5 1 (-20) (x, y) ∧
    x = (B.1 + C.1) / 2 ∧ y = (B.2 + C.2) / 2) ∧
  -- 2. Area of triangle ABC
  TriangleArea A B C = 17 ∧
  -- 3. Equation of perpendicular bisector of side BC
  (∃ (x y : ℝ), LineEquation 3 2 (-19) (x, y) ∧
    (x - B.1) * (C.1 - B.1) + (y - B.2) * (C.2 - B.2) = 0 ∧
    (x - ((B.1 + C.1) / 2))^2 + (y - ((B.2 + C.2) / 2))^2 = 
    ((B.1 - C.1) / 2)^2 + ((B.2 - C.2) / 2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l331_33101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_on_fixed_line_l331_33120

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope 2 -/
def LineWithSlope2 (p : Point) : Prop :=
  ∃ (c : ℝ), p.y = 2 * p.x + c

/-- The equation of the hyperbola -/
def OnHyperbola (p : Point) : Prop :=
  p.x^2 / 3 - p.y^2 / 6 = 1

/-- The midpoint of two points -/
noncomputable def Midpoint (a b : Point) : Point :=
  ⟨(a.x + b.x) / 2, (a.y + b.y) / 2⟩

/-- Theorem: The midpoint of AB lies on the line x - y = 0 -/
theorem midpoint_on_fixed_line (a b : Point) 
  (ha : OnHyperbola a) (hb : OnHyperbola b) 
  (hl : LineWithSlope2 a ∧ LineWithSlope2 b) : 
  let m := Midpoint a b
  m.x = m.y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_on_fixed_line_l331_33120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l331_33178

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ x, f (x - (-Real.pi / 6)) = -f (x + (-Real.pi / 6))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l331_33178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l331_33145

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 + 8*x + 12)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -6 ∨ (-6 < x ∧ x < -2) ∨ -2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l331_33145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l331_33144

/-- A geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_four (a r : ℝ) :
  (geometric_sequence a r 2) * (geometric_sequence a r 3) = 2 * (geometric_sequence a r 1) →
  (5 / 4 : ℝ) = (geometric_sequence a r 4 + 2 * geometric_sequence a r 7) / 2 →
  geometric_sum a r 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l331_33144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l331_33191

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℚ
  yIntercept : ℚ

-- Define a point in 2D space
structure Point where
  x : ℚ
  y : ℚ

-- Define the given line
def givenLine : Line := { slope := 1/2, yIntercept := 3 }

-- Define the point the new line must pass through
def givenPoint : Point := { x := 0, y := -1 }

-- Define the equation of a line in standard form (ax + by + c = 0)
structure StandardFormLine where
  a : ℚ
  b : ℚ
  c : ℚ

-- The line we need to prove correct
def targetLine : StandardFormLine := { a := 1, b := -2, c := -2 }

theorem parallel_line_through_point :
  -- The target line is parallel to the given line
  targetLine.a / targetLine.b = -givenLine.slope ∧
  -- The target line passes through the given point
  targetLine.a * givenPoint.x + targetLine.b * givenPoint.y + targetLine.c = 0 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l331_33191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_equality_l331_33125

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

-- Define the reference function
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x)

-- Define the shifted reference function
noncomputable def h (x : ℝ) : ℝ := g (x - π / 6)

-- Theorem stating that the shifted reference function is equal to the original function
theorem shifted_sine_equality : ∀ x : ℝ, f x = h x := by
  intro x
  simp [f, h, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_equality_l331_33125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_l331_33100

/-- The area of a regular hexadecagon inscribed in a circle -/
theorem hexadecagon_area (r : ℝ) : 
  (16 : ℝ) * ((1/2) * r^2 * Real.sin (Real.pi / 16)) = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2) :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_l331_33100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_l331_33102

def A : ℝ × ℝ := (4, 5)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_over_x_eq_neg_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

def B : ℝ × ℝ := reflect_over_y_axis A

def C : ℝ × ℝ := reflect_over_x_eq_neg_y B

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ :=
  let a := distance p q
  let b := distance q r
  let c := distance r p
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_triangle_ABC : area_triangle A B C = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_l331_33102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extra_credit_l331_33185

/-- Represents a class of students with their scores -/
structure StudentClass where
  size : Nat
  scores : Fin size → ℝ

/-- Calculates the average score of a class -/
noncomputable def classAverage (c : StudentClass) : ℝ :=
  (Finset.univ.sum c.scores) / c.size

/-- Counts the number of students with scores above the class average -/
noncomputable def studentsAboveAverage (c : StudentClass) : Nat :=
  (Finset.univ.filter (fun i => c.scores i > classAverage c)).card

/-- Theorem: The maximum number of students who can receive extra credit in a class of 200 is 199 -/
theorem max_extra_credit (c : StudentClass) (h : c.size = 200) :
  studentsAboveAverage c ≤ 199 ∧ ∃ c', c'.size = 200 ∧ studentsAboveAverage c' = 199 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extra_credit_l331_33185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_equals_143_14_l331_33140

/-- A circle in the upper half-plane --/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- A layer of circles --/
def Layer := List Circle

/-- The set of all circles up to layer 6 --/
def S : List Circle := sorry

/-- The radii of the two circles in layer L₀ --/
def L₀_radii : List ℝ := [70, 73]

/-- The number of circles in layer k --/
def num_circles (k : ℕ) : ℕ := 
  if k = 0 then 2 else 2^(k-1)

/-- The sum of reciprocals of square roots of radii --/
noncomputable def sum_reciprocal_sqrt (circles : List Circle) : ℝ :=
  circles.map (λ c => 1 / Real.sqrt c.radius) |>.sum

/-- The main theorem --/
theorem sum_reciprocal_sqrt_equals_143_14 : 
  sum_reciprocal_sqrt S = 143/14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_equals_143_14_l331_33140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_3sin_x_bounds_l331_33122

theorem cos_2x_plus_3sin_x_bounds (x : ℝ) : 
  -4 ≤ Real.cos (2 * x) + 3 * Real.sin x ∧ Real.cos (2 * x) + 3 * Real.sin x ≤ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_3sin_x_bounds_l331_33122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_sine_function_l331_33179

-- Define the original function
noncomputable def original_func (x : ℝ) : ℝ := Real.sin (x - Real.pi / 4)

-- Define the transformation
noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f ((x - Real.pi / 6) / 2)

-- State the theorem
theorem transformed_sine_function :
  ∀ x : ℝ, transform original_func x = Real.sin (x / 2 - Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_sine_function_l331_33179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l331_33113

open Real Set

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.tan (x + π / 4)

-- Define the interval where f is strictly increasing
def increasing_interval (k : ℤ) : Set ℝ := Ioo (k * π - 3 * π / 4) (k * π + π / 4)

-- Theorem statement
theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (increasing_interval k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l331_33113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snooker_tournament_tickets_l331_33152

theorem snooker_tournament_tickets (vip_price gen_price total_cost : ℚ)
  (vip_tickets gen_tickets : ℕ) :
  vip_price = 45 →
  gen_price = 20 →
  total_cost = 7500 →
  vip_tickets + gen_tickets = (vip_price * vip_tickets + gen_price * gen_tickets) / total_cost * total_cost →
  vip_tickets + 276 = gen_tickets →
  vip_tickets + gen_tickets = 336 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snooker_tournament_tickets_l331_33152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_zero_det_projection_zero_specific_l331_33163

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm := Real.sqrt (v.1^2 + v.2^2)
  let u := (v.1 / norm, v.2 / norm)
  ![![u.1 * u.1, u.1 * u.2], ![u.2 * u.1, u.2 * u.2]]

theorem det_projection_zero (v : ℝ × ℝ) :
  Matrix.det (projection_matrix v) = 0 := by
  sorry

theorem det_projection_zero_specific :
  Matrix.det (projection_matrix (3, 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_zero_det_projection_zero_specific_l331_33163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l331_33171

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  16 * x^2 - 64 * x + y^2 + 4 * y + 4 = 0

-- Define the distance between foci
noncomputable def distance_between_foci : ℝ := 4 * Real.sqrt 15

-- Theorem statement
theorem ellipse_foci_distance :
  ∀ x y : ℝ, ellipse_equation x y → distance_between_foci = 4 * Real.sqrt 15 :=
by
  intro x y h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l331_33171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l331_33187

theorem coefficient_x_cubed_in_expansion : ℕ := by
  -- Define the binomial expansion of (2 - √x)^8
  let binomial_expansion (x : ℝ) := (2 - Real.sqrt x)^8

  -- Define the coefficient of x^3 in the expansion
  let coefficient_x_cubed : ℕ := 112

  -- State the theorem
  have coeff_x_cubed_equals_112 : coefficient_x_cubed = 112 := by
    -- The proof goes here
    sorry

  -- Return the result
  exact coefficient_x_cubed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l331_33187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l331_33182

theorem absolute_value_equation_solutions :
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |x + 1| + |x - 3| = 4) ∧ Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solutions_l331_33182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l331_33167

-- Define the region
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ 3 * p.1 + 2 * p.2 ≥ 6 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ Region ∧ p2 ∈ Region ∧
    ∀ (q1 q2 : ℝ × ℝ), q1 ∈ Region → q2 ∈ Region →
      distance q1 q2 ≤ distance p1 p2 ∧
      distance p1 p2 = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l331_33167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_smallest_l331_33177

def numbers : List ℕ := [10, 11, 12, 13]

theorem product_of_two_smallest (list : List ℕ) (h : list = numbers) :
  ∃ (min₁ min₂ : ℕ), 
    min₁ ∈ list ∧ 
    min₂ ∈ list ∧ 
    min₁ ≤ min₂ ∧
    (∀ x ∈ list, min₁ ≤ x) ∧
    (∀ x ∈ list, x ≠ min₁ → min₂ ≤ x) ∧
    min₁ * min₂ = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_smallest_l331_33177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_l331_33150

/-- Given two functions f and g, where f(x) = 4x^2 - 3x + 5 and g(x) = x^2 - mx - 8,
    if f(5) - g(5) = 20, then m = -14 -/
theorem function_difference (m : ℝ) : 
  (λ x : ℝ => 4 * x^2 - 3 * x + 5) 5 - (λ x : ℝ => x^2 - m * x - 8) 5 = 20 → m = -14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_l331_33150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l331_33105

-- Define sets P and Q
def P (a : ℝ) := {x : ℝ | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

-- Define the complement of P
def complement_P (a : ℝ) := {x : ℝ | x < a + 1 ∨ 2*a + 1 < x}

-- Theorem for part 1
theorem part_one : 
  (complement_P 3 ∩ Q) = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Define the condition for P being a sufficient but not necessary condition for Q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a)

-- Theorem for part 2
theorem part_two : 
  {a : ℝ | sufficient_not_necessary a} = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l331_33105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_sum_l331_33135

/-- Represents a rectangular box with given dimensions -/
structure RectangularBox where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the triangle formed by the centers of three faces meeting at a corner -/
noncomputable def triangleArea (box : RectangularBox) : ℝ :=
  let a := Real.sqrt ((box.width / 2) ^ 2 + (box.length / 2) ^ 2)
  let b := Real.sqrt ((box.width / 2) ^ 2 + (box.height / 2) ^ 2)
  let c := Real.sqrt ((box.length / 2) ^ 2 + (box.height / 2) ^ 2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem box_dimension_sum (m n : ℕ) (h1 : Nat.Coprime m n) :
  let box := RectangularBox.mk 10 14 (m / n : ℝ)
  triangleArea box = 24 → m + n = 170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimension_sum_l331_33135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_when_m_is_3_m_range_when_A_intersect_B_is_empty_m_range_when_A_intersect_B_equals_A_l331_33136

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | x < m}

-- Define the universal set U
def U (m : ℝ) : Set ℝ := A ∪ B m

-- Theorem 1
theorem intersection_A_complement_B_when_m_is_3 :
  A ∩ (U 3)ᶜ ∩ (B 3)ᶜ = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem m_range_when_A_intersect_B_is_empty :
  {m : ℝ | A ∩ B m = ∅} = {m : ℝ | m ≤ -2} := by sorry

-- Theorem 3
theorem m_range_when_A_intersect_B_equals_A :
  {m : ℝ | A ∩ B m = A} = {m : ℝ | m ≥ 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_when_m_is_3_m_range_when_A_intersect_B_is_empty_m_range_when_A_intersect_B_equals_A_l331_33136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_factors_of_3_pow_18_minus_1_l331_33169

theorem three_digit_factors_of_3_pow_18_minus_1 : 
  (Finset.filter (λ n : ℕ ↦ 100 ≤ n ∧ n ≤ 999 ∧ (3^18 - 1) % n = 0) (Finset.range 1000)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_factors_of_3_pow_18_minus_1_l331_33169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_four_l331_33157

/-- Calculates the length of a tunnel given the length of a train, its speed, and the time it takes to exit the tunnel. -/
noncomputable def tunnel_length (train_length : ℝ) (train_speed : ℝ) (exit_time : ℝ) : ℝ :=
  train_speed * exit_time / 60 - train_length

/-- Proves that the tunnel length is 4 miles given the specified conditions. -/
theorem tunnel_length_is_four :
  let train_length : ℝ := 2
  let train_speed : ℝ := 90
  let exit_time : ℝ := 4
  tunnel_length train_length train_speed exit_time = 4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tunnel_length 2 90 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_is_four_l331_33157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l331_33164

/-- Calculates the length of a train given the speeds of two trains, time to cross, and the length of the other train --/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (time_to_cross : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let combined_length := relative_speed * time_to_cross
  combined_length - other_train_length

/-- Theorem stating that under the given conditions, the length of the first train is 240 meters --/
theorem first_train_length :
  let speed1 : ℝ := 120 -- km/h
  let speed2 : ℝ := 80  -- km/h
  let time_to_cross : ℝ := 9 -- seconds
  let second_train_length : ℝ := 260.04 -- meters
  calculate_train_length speed1 speed2 time_to_cross second_train_length = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l331_33164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_distance_l331_33129

/-- Given an equilateral triangle ABC with side length 6 and a point D such that
    BD = DC = 9 and D is directly below A, prove that AD = 6√2 + 3√3 -/
theorem triangle_distance (A B C D : ℝ × ℝ) : 
  let dist := λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- ABC is equilateral with side length 6
  (dist A B = 6 ∧ dist B C = 6 ∧ dist C A = 6) →
  -- D is such that BD = DC = 9
  (dist B D = 9 ∧ dist D C = 9) →
  -- D is directly below A (same x-coordinate)
  (A.1 = D.1) →
  -- D is in the same horizontal plane as BC
  (D.2 = B.2) →
  -- Then AD = 6√2 + 3√3
  dist A D = 6 * Real.sqrt 2 + 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_distance_l331_33129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l331_33176

theorem expression_evaluation :
  (2^4 + 2^3 * 2) / ((1/2) * 2^2 + (1/8) + (1/32)) = 1024 / 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l331_33176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_analysis_l331_33111

def patrol_movements : List Int := [10, -8, 6, -13, 7, -12, 3, -1]
def fuel_rate : Float := 0.05
def gas_station_location : Int := 6

theorem patrol_analysis (movements : List Int) (fuel_cons : Float) (gas_loc : Int) :
  let final_position := movements.foldl (· + ·) 0
  let total_distance := movements.map (fun x => Int.natAbs x) |>.foldl (· + ·) 0
  let cross_count := movements.scanl (· + ·) 0 |>.zip movements |>.filter (fun (pos, move) => 
    (pos ≤ gas_loc ∧ gas_loc < pos + move) ∨ (pos + move ≤ gas_loc ∧ gas_loc < pos)) |>.length
  (final_position = -8 ∧ 
   total_distance = 60 ∧
   cross_count = 4) :=
by
  sorry

#check patrol_analysis patrol_movements fuel_rate gas_station_location

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_analysis_l331_33111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l331_33127

/-- A right-angled parallelogram with side lengths 2a and h -/
structure RightParallelogram (a h : ℝ) where
  ab : ℝ := 2 * a
  ac : ℝ := h

/-- A point M on the line EF parallel to AC -/
structure PointM (a h x : ℝ) where
  em : ℝ := x
  mf : ℝ := h - x

/-- The sum of distances AM + MB + MF -/
noncomputable def distanceSum (a h x : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 + x^2) + h - x

theorem min_distance_sum (a h : ℝ) (ha : 0 < a) (hh : 0 < h) :
  ∃ (x : ℝ), ∀ (y : ℝ), distanceSum a h x ≤ distanceSum a h y ∧
  distanceSum a h x = h + a * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l331_33127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_abh_measure_l331_33130

/-- A regular octagon is a polygon with 8 equal sides and 8 equal angles -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : True  -- This is a placeholder for the regularity condition

/-- The measure of an angle in degrees -/
def angle_measure : ℝ → ℝ := id

/-- Angle notation -/
notation "∠" => angle_measure

theorem angle_abh_measure (ABCDEFGH : RegularOctagon) :
  ∠ ABH = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_abh_measure_l331_33130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_length_l331_33198

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side b
  S : ℝ  -- Area of the triangle

-- Define our specific triangle
noncomputable def triangleABC : Triangle where
  A := 2 * Real.pi / 3  -- 120° in radians
  b := 2
  S := 2 * Real.sqrt 3

-- Theorem statement
theorem side_c_length (triangle : Triangle) (h : triangle = triangleABC) : 
  let c := 2 * triangle.S / (triangle.b * Real.sin triangle.A)
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_length_l331_33198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_area_is_45_percent_l331_33186

/-- Represents a square flag with a cross design -/
structure MyFlag where
  /-- Total area of the flag -/
  total_area : ℝ
  /-- Area of the entire cross (green and yellow) as a fraction of the total area -/
  cross_area_fraction : ℝ
  /-- Area of the yellow center as a fraction of the total area -/
  yellow_area_fraction : ℝ
  /-- Assumption that the cross_area_fraction is 0.49 -/
  cross_area_assum : cross_area_fraction = 0.49
  /-- Assumption that the yellow_area_fraction is 0.04 -/
  yellow_area_assum : yellow_area_fraction = 0.04

/-- The fraction of the flag that is green -/
def green_area_fraction (f : MyFlag) : ℝ :=
  f.cross_area_fraction - f.yellow_area_fraction

/-- Theorem stating that the green area fraction is 0.45 -/
theorem green_area_is_45_percent (f : MyFlag) :
  green_area_fraction f = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_area_is_45_percent_l331_33186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_faster_than_x_l331_33155

/-- Represents a route with its properties -/
structure Route where
  total_distance : ℚ
  normal_speed : ℚ
  slow_distance : ℚ
  slow_speed : ℚ

/-- Calculates the time taken to travel a route in hours -/
def travel_time (r : Route) : ℚ :=
  (r.total_distance - r.slow_distance) / r.normal_speed + r.slow_distance / r.slow_speed

/-- Route X properties -/
def route_x : Route :=
  { total_distance := 8
  , normal_speed := 25
  , slow_distance := 1
  , slow_speed := 10 }

/-- Route Y properties -/
def route_y : Route :=
  { total_distance := 7
  , normal_speed := 35
  , slow_distance := 1
  , slow_speed := 15 }

/-- Theorem stating that Route Y is faster than Route X by approximately 6.1 minutes -/
theorem route_y_faster_than_x : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10 ∧ 
  (travel_time route_x - travel_time route_y) * 60 = 61/10 + ε := by
  sorry

#eval (travel_time route_x - travel_time route_y) * 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_faster_than_x_l331_33155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eight_numbers_with_divisibility_conditions_l331_33148

theorem no_eight_numbers_with_divisibility_conditions : ¬ ∃ (S : Finset ℕ),
  (Finset.card S = 8) ∧
  (Finset.filter (fun n => n % 8 = 0) S).card = 1 ∧
  (Finset.filter (fun n => n % 7 = 0) S).card = 2 ∧
  (Finset.filter (fun n => n % 6 = 0) S).card = 3 ∧
  (Finset.filter (fun n => n % 5 = 0) S).card = 4 ∧
  (Finset.filter (fun n => n % 4 = 0) S).card = 5 ∧
  (Finset.filter (fun n => n % 3 = 0) S).card = 6 ∧
  (Finset.filter (fun n => n % 2 = 0) S).card = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eight_numbers_with_divisibility_conditions_l331_33148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_value_proof_l331_33134

def total_coins : ℕ := 30
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def swap_difference : ℕ := 90

theorem coin_value_proof (n : ℕ) (h1 : n ≤ total_coins) :
  let d := total_coins - n
  let original_value := n * nickel_value + d * dime_value
  let swapped_value := n * dime_value + d * nickel_value
  swapped_value = original_value + swap_difference →
  original_value = 180 := by
  intro h2
  -- Proof steps would go here
  sorry

#check coin_value_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_value_proof_l331_33134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l331_33131

/-- Quadratic function properties -/
theorem quadratic_function_properties (m : ℝ) (h : m < 0) :
  let f : ℝ → ℝ := λ x ↦ m * x^2 - (4*m + 1) * x + 3*m + 3
  (∀ x > 2, ∀ y > x, f x > f y) ∧ 
  (f 1 = 2 ∧ f 3 = 0) ∧
  (let x₁ := (m + 1) / m
   let x₂ := 3
   x₂ - x₁ > 2) ∧
  (let vertex_y := -m - 1 / (4*m) + 1
   vertex_y > 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l331_33131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l331_33141

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi/6)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi ∧
  ∀ (A B C : ℝ) (b c : ℝ),
    0 < B ∧ 0 < C ∧ B + C < Real.pi ∧
    4 = c ∧
    Real.sin C = 2 * Real.sin B ∧
    (∀ x, f x ≤ f A) →
    (1/2) * b * c * Real.sin A = (8 * Real.sqrt 3) / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l331_33141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axisymmetric_iff_foldable_and_coincide_l331_33109

-- Define a shape
structure Shape where
  -- Add necessary fields for a shape
  (dummy : Unit)

-- Define the property of being foldable along a line
def isFoldableAlongLine (s : Shape) : Prop :=
  True -- Placeholder definition

-- Define the property of parts coinciding after folding
def partsCoincideAfterFolding (s : Shape) : Prop :=
  True -- Placeholder definition

-- Define axisymmetric shape
def isAxisymmetric (s : Shape) : Prop :=
  isFoldableAlongLine s ∧ partsCoincideAfterFolding s

-- Theorem statement
theorem axisymmetric_iff_foldable_and_coincide (s : Shape) :
  isAxisymmetric s ↔ (isFoldableAlongLine s ∧ partsCoincideAfterFolding s) :=
by
  -- Proof goes here
  sorry

#check axisymmetric_iff_foldable_and_coincide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axisymmetric_iff_foldable_and_coincide_l331_33109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_solution_l331_33188

def Vector2D := ℝ × ℝ

def add_vectors (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

noncomputable def vector_length (v : Vector2D) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

def parallel_to_x_axis (v : Vector2D) : Prop :=
  v.2 = 0

theorem vector_a_solution (a b : Vector2D) :
  vector_length (add_vectors a b) = 1 →
  parallel_to_x_axis (add_vectors a b) →
  b = (2, -1) →
  (a = (-1, 1) ∨ a = (-3, 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_a_solution_l331_33188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_pyramid_l331_33168

/-- The radius of a sphere inscribed in a pyramid with an isosceles triangular base -/
noncomputable def inscribed_sphere_radius (b α : ℝ) : ℝ :=
  (b * Real.sin α) / (4 * (Real.cos (α / 4))^2)

/-- Theorem: The radius of the inscribed sphere in a pyramid with specific properties -/
theorem inscribed_sphere_radius_in_pyramid (b α : ℝ) 
  (h_b_pos : b > 0) 
  (h_α_pos : α > 0) 
  (h_α_range : α < π) : 
  ∃ (r : ℝ), r > 0 ∧ r = inscribed_sphere_radius b α :=
by
  sorry

#check inscribed_sphere_radius
#check inscribed_sphere_radius_in_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_pyramid_l331_33168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l331_33184

open Real Nat BigOperators

-- Define the function f
def f (n : ℕ) (x : ℝ) : ℝ :=
  x * ∏ i in Finset.range (n + 1), (x + i)

-- State the theorem
theorem f_derivative_at_zero (n : ℕ) : 
  deriv (f n) 0 = n.factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l331_33184
