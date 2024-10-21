import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1266_126648

noncomputable def f (x : ℝ) := Real.cos x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max ∧ f x_max = Real.sqrt 2 / 2 + 1 / 2) ∧
  (∀ (A : ℝ), f (A / 2 + π / 8) = 1 →
    let S := 1 / 2 * 3 * 3 * Real.sin A
    S = 9 * Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1266_126648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1266_126679

/-- Represents the state of the game -/
structure GameState where
  sum : Nat
  player_turn : Bool

/-- Checks if a move is valid (between 1 and 10) -/
def valid_move (n : Nat) : Prop :=
  1 ≤ n ∧ n ≤ 10

/-- Defines a single move in the game -/
def make_move (state : GameState) (move : Nat) : GameState :=
  { sum := state.sum + move,
    player_turn := ¬state.player_turn }

/-- Checks if the game is won (sum is exactly 100) -/
def is_won (state : GameState) : Prop :=
  state.sum = 100

/-- Defines a winning strategy for the first player -/
def first_player_strategy (state : GameState) : Nat :=
  match state.sum with
  | 0 => 1
  | 11 => 1
  | 22 => 1
  | 33 => 1
  | 44 => 1
  | 55 => 1
  | 66 => 1
  | 77 => 1
  | 88 => 1
  | _ => 1  -- Placeholder, actual strategy would be more complex

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Nat),
    ∀ (game : List GameState),
      game.head?.isSome →
      (∀ i, i < game.length - 1 →
        (game.get ⟨i, by sorry⟩).player_turn →
        game.get ⟨i + 1, by sorry⟩ = make_move (game.get ⟨i, by sorry⟩) (strategy (game.get ⟨i, by sorry⟩))) →
      (∀ i, i < game.length - 1 →
        ¬(game.get ⟨i, by sorry⟩).player_turn →
        ∃ (move : Nat), valid_move move ∧
          game.get ⟨i + 1, by sorry⟩ = make_move (game.get ⟨i, by sorry⟩) move) →
      ∃ (final_state : GameState), final_state ∈ game ∧ is_won final_state ∧ ¬final_state.player_turn :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1266_126679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_decomposition_l1266_126681

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ),
    (Real.tan (22.5 * π / 180) = Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) - Real.sqrt (c : ℝ) - (d : ℝ)) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) →
    a + b + c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_decomposition_l1266_126681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tube_volume_difference_l1266_126610

noncomputable def amy_height : ℝ := 12
noncomputable def amy_circumference : ℝ := 10
noncomputable def carlos_height : ℝ := 8
noncomputable def carlos_circumference : ℝ := 14

noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (circumference ^ 2 * height) / (4 * Real.pi)

theorem tube_volume_difference :
  |cylinder_volume carlos_height carlos_circumference - 
   cylinder_volume amy_height amy_circumference| = 92 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tube_volume_difference_l1266_126610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_equals_expected_value_l1266_126604

noncomputable section

-- Define the radii of the three circles
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 5

-- Define the sides of the triangle formed by the centers of the circles
def side1 : ℝ := a + b
def side2 : ℝ := b + c
def side3 : ℝ := c + a

-- Define the semi-perimeter of the triangle
noncomputable def s : ℝ := (side1 + side2 + side3) / 2

-- Define the area of the triangle using Heron's formula
noncomputable def area : ℝ := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))

-- Define the radius of the inscribed circle
noncomputable def r : ℝ := area / s

-- Define the distances from the center of the inscribed circle to the sides of the triangle
noncomputable def d1 : ℝ := (2 * area) / side1
noncomputable def d2 : ℝ := (2 * area) / side2
noncomputable def d3 : ℝ := (2 * area) / side3

-- Theorem statement
theorem sum_of_distances_equals_expected_value :
  d1 + d2 + d3 = 5 / Real.sqrt 14 + 5 / Real.sqrt 21 + 5 / Real.sqrt 30 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_equals_expected_value_l1266_126604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1266_126644

-- Define the points as pairs of real numbers
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (14, 2)
def X : ℝ × ℝ := (5, 2)
def Y : ℝ × ℝ := (8, 6)
def Z : ℝ × ℝ := (11, 1)

-- Define a function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- State the theorem
theorem area_ratio :
  (triangleArea X Y Z) / (triangleArea A B C) = 27 / 103 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1266_126644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pattern_count_l1266_126692

/-- A pattern is a configuration of 2×1 rectangles in a 4×4 grid --/
def Pattern := Fin 4 → Fin 4 → Bool

/-- A pattern is valid if it uses exactly eight 2×1 rectangles --/
def isValid (p : Pattern) : Prop := sorry

/-- A pattern is vertically symmetrical --/
def isVerticallySymmetrical (p : Pattern) : Prop := sorry

/-- A pattern has two adjacent blank squares in the first row --/
def hasAdjacentBlanksInFirstRow (p : Pattern) : Prop := sorry

/-- Two patterns are equivalent under rotation --/
def areEquivalentUnderRotation (p1 p2 : Pattern) : Prop := sorry

/-- The set of unique patterns satisfying all conditions --/
def uniquePatterns : Set Pattern :=
  {p | isValid p ∧ isVerticallySymmetrical p ∧ hasAdjacentBlanksInFirstRow p}

/-- Fintype instance for Pattern --/
instance : Fintype Pattern := by sorry

/-- Decidable equality for Pattern --/
instance : DecidableEq Pattern := by sorry

/-- Fintype instance for uniquePatterns --/
instance : Fintype uniquePatterns := by sorry

/-- The set of representative patterns --/
def representativePatterns : Set Pattern :=
  {p ∈ uniquePatterns | ∀ q ∈ uniquePatterns, p = q ∨ ¬areEquivalentUnderRotation p q}

/-- Fintype instance for representativePatterns --/
instance : Fintype representativePatterns := by sorry

theorem unique_pattern_count :
  Fintype.card representativePatterns = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pattern_count_l1266_126692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l1266_126651

noncomputable def angle_between_vectors : ℝ := Real.pi / 3

theorem magnitude_of_vector_combination (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))))
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 3) :
  Real.sqrt ((2 * a.1 - 3 * b.1)^2 + (2 * a.2 - 3 * b.2)^2) = Real.sqrt 61 := by
  sorry

#check magnitude_of_vector_combination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l1266_126651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1266_126695

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The triangle OAB -/
structure Triangle where
  A : PolarPoint
  B : PolarPoint

/-- Properties of the triangle OAB -/
class IsIsoscelesRightTriangle (t : Triangle) where
  isosceles : t.A.r = Real.sqrt 2 * t.B.r
  right_angle : t.B.θ - t.A.θ = Real.pi / 4
  counterclockwise : t.B.θ > t.A.θ

/-- The circumscribed circle of triangle OAB -/
noncomputable def circumscribed_circle (t : Triangle) : ℝ → ℝ :=
  λ θ => 2 * Real.sin (θ + Real.pi / 3)

theorem triangle_properties (t : Triangle) [IsIsoscelesRightTriangle t]
  (h_A : t.A = PolarPoint.mk 2 (Real.pi / 6)) :
  (t.B = PolarPoint.mk (Real.sqrt 2) (5 * Real.pi / 12)) ∧
  (∀ θ, circumscribed_circle t θ = 2 * Real.sin (θ + Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1266_126695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_is_two_l1266_126694

def is_valid_digit_pair (a b : Nat) : Prop :=
  let num := a * 10 + b
  num % 17 = 0 ∨ num % 23 = 0

def is_valid_number (n : List Nat) : Prop :=
  n.length = 1992 ∧
  n.getLast? = some 1 ∧
  ∀ i, i < 1991 → is_valid_digit_pair (n.get! i) (n.get! (i + 1))

theorem first_digit_is_two (n : List Nat) :
  is_valid_number n → n.head? = some 2 :=
by sorry

#check first_digit_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_is_two_l1266_126694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_condition_non_negative_condition_inequality_proof_l1266_126689

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - a * x / (x + 1)

-- Theorem 1: When x=1 is an extreme point of f(x), a = 2
theorem extreme_point_condition (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  a = 2 := by sorry

-- Theorem 2: For f(x) to be non-negative on [0, +∞), 0 < a ≤ 1
theorem non_negative_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) ↔ (0 < a ∧ a ≤ 1) := by sorry

-- Theorem 3: (2015/2016)^2016 < 1/e
theorem inequality_proof :
  (2015 / 2016) ^ 2016 < 1 / Real.exp 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_condition_non_negative_condition_inequality_proof_l1266_126689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daniels_purchase_worth_l1266_126686

/-- The total worth of Daniel's purchases given the sales tax, tax rate, and cost of tax-free items -/
theorem daniels_purchase_worth (sales_tax : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) 
  (h1 : sales_tax = 0.30)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 19.7) :
  sales_tax / tax_rate + tax_free_cost = 24.7 := by
  sorry

#check daniels_purchase_worth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daniels_purchase_worth_l1266_126686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_selling_price_calculation_l1266_126699

def total_selling_price (cost_prices : List ℝ) (profit_margins : List ℝ) 
  (discount_second : ℝ) (discount_fourth : ℝ) : ℝ :=
  let selling_prices := List.zipWith (λ c p => c * (1 + p)) cost_prices profit_margins
  let discounted_second := selling_prices[1]! * (1 - discount_second)
  let discounted_fourth := selling_prices[3]! * (1 - discount_fourth)
  selling_prices[0]! + discounted_second + selling_prices[2]! + discounted_fourth

theorem total_selling_price_calculation :
  let cost_prices : List ℝ := [280, 350, 500, 600]
  let profit_margins : List ℝ := [0.30, 0.45, 0.25, 0.50]
  let discount_second : ℝ := 0.10
  let discount_fourth : ℝ := 0.05
  total_selling_price cost_prices profit_margins discount_second discount_fourth = 2300.75 := by
  sorry

#eval total_selling_price [280, 350, 500, 600] [0.30, 0.45, 0.25, 0.50] 0.10 0.05

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_selling_price_calculation_l1266_126699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_20_to_30_l1266_126621

def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 2)).all (λ m => n % (m + 2) ≠ 0)

def sumOfPrimesBetween (a b : ℕ) : ℕ :=
  (List.range (b - a - 1)).map (λ i => a + i + 1)
    |>.filter isPrime
    |>.sum

theorem sum_of_primes_20_to_30 : sumOfPrimesBetween 20 30 = 52 := by
  sorry

#eval sumOfPrimesBetween 20 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_20_to_30_l1266_126621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_cube_tiling_exists_l1266_126668

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Represents the 7-cube structure -/
structure SevenCubeShape where
  center : Point3D
  valid : ∀ p : Point3D, 
    (p = center) ∨ 
    ((p.x = center.x + 1 ∨ p.x = center.x - 1) ∧ p.y = center.y ∧ p.z = center.z) ∨
    (p.x = center.x ∧ (p.y = center.y + 1 ∨ p.y = center.y - 1) ∧ p.z = center.z) ∨
    (p.x = center.x ∧ p.y = center.y ∧ (p.z = center.z + 1 ∨ p.z = center.z - 1))

/-- A tiling of 3D space using the 7-cube shapes -/
def Tiling := ℤ → ℤ → ℤ → SevenCubeShape

/-- Predicate to check if a tiling is valid (uniform and gap-free) -/
def is_valid_tiling (t : Tiling) : Prop :=
  ∀ p : Point3D, ∃! s : SevenCubeShape, 
    (∃ i j k : ℤ, t i j k = s) ∧
    ((p = s.center) ∨ 
     ((p.x = s.center.x + 1 ∨ p.x = s.center.x - 1) ∧ p.y = s.center.y ∧ p.z = s.center.z) ∨
     (p.x = s.center.x ∧ (p.y = s.center.y + 1 ∨ p.y = s.center.y - 1) ∧ p.z = s.center.z) ∨
     (p.x = s.center.x ∧ p.y = s.center.y ∧ (p.z = s.center.z + 1 ∨ p.z = s.center.z - 1)))

/-- Theorem stating that a valid tiling exists -/
theorem seven_cube_tiling_exists : ∃ t : Tiling, is_valid_tiling t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_cube_tiling_exists_l1266_126668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l1266_126607

/-- The equation of a line passing through (1,2) -/
def line_equation (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y - 2 = k * (x - 1)

/-- The area of the triangle formed by a line and the positive axes -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  (1/2) * |(-2/k + 1) * (-k + 2)|

/-- The set of valid slopes for the line l -/
noncomputable def valid_slopes : Set ℝ :=
  {k | k = -1 ∨ k = -4 ∨ k = (13 + 3 * Real.sqrt 17) / 2 ∨ k = (13 - 3 * Real.sqrt 17) / 2}

theorem line_equation_theorem :
  ∀ k : ℝ, 
    k ≠ 0 → 
    (∃ x y : ℝ, line_equation k x y ∧ 2*x - 3*y + 4 = 0 ∧ x + 2*y - 5 = 0) →
    triangle_area k = 9/2 →
    k ∈ valid_slopes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l1266_126607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_squared_l1266_126601

-- Define the complex number z
variable (z : ℂ)

-- Define the condition that the real part of z is positive
def positive_real_part (z : ℂ) : Prop := 0 < z.re

-- Define the area of the parallelogram
noncomputable def parallelogram_area (z : ℂ) : ℝ := abs (z.im * (1/z).re - z.re * (1/z).im)

-- Define the magnitude of z + 1/z squared
noncomputable def magnitude_squared (z : ℂ) : ℝ := Complex.normSq (z + 1/z)

-- State the theorem
theorem min_magnitude_squared 
  (h_positive : positive_real_part z) 
  (h_area : parallelogram_area z = 45/47) : 
  ∃ (d : ℝ), d^2 = 78/47 ∧ ∀ w, positive_real_part w → parallelogram_area w = 45/47 → magnitude_squared w ≥ d^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_squared_l1266_126601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_determination_l1266_126677

-- Define the basic geometric objects
structure Point where

structure Line where

structure Plane where

-- Define the relationships between geometric objects
def collinear (p q r : Point) : Prop := sorry

def intersect (l₁ l₂ : Line) : Prop := sorry

def parallel (l₁ l₂ : Line) : Prop := sorry

def point_on_line (p : Point) (l : Line) : Prop := sorry

def point_on_plane (p : Point) (π : Plane) : Prop := sorry

def line_on_plane (l : Line) (π : Plane) : Prop := sorry

-- Define the plane determination methods
noncomputable def plane_from_two_intersecting_lines (l₁ l₂ : Line) : Plane := sorry

noncomputable def plane_from_two_parallel_lines (l₁ l₂ : Line) : Plane := sorry

noncomputable def plane_from_line_and_point (l : Line) (p : Point) : Plane := sorry

-- Theorem statement
theorem plane_determination :
  (∀ l₁ l₂ : Line, intersect l₁ l₂ → ∃! π : Plane, line_on_plane l₁ π ∧ line_on_plane l₂ π) ∧
  (∀ l₁ l₂ : Line, parallel l₁ l₂ → ∃! π : Plane, line_on_plane l₁ π ∧ line_on_plane l₂ π) ∧
  (∀ l : Line, ∀ p : Point, ¬point_on_line p l → ∃! π : Plane, line_on_plane l π ∧ point_on_plane p π) ∧
  ¬(∀ p₁ p₂ p₃ p₄ : Point, ¬collinear p₁ p₂ p₃ ∧ ¬collinear p₁ p₂ p₄ ∧ ¬collinear p₁ p₃ p₄ ∧ ¬collinear p₂ p₃ p₄ →
    ∃! π : Plane, point_on_plane p₁ π ∧ point_on_plane p₂ π ∧ point_on_plane p₃ π ∧ point_on_plane p₄ π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_determination_l1266_126677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_decryption_probability_l1266_126627

theorem password_decryption_probability :
  let p_a : ℚ := 1/5  -- Probability of success for A
  let p_b : ℚ := 1/3  -- Probability of success for B
  let p_c : ℚ := 1/4  -- Probability of success for C
  let p_not_decrypted : ℚ := (1 - p_a) * (1 - p_b) * (1 - p_c)  -- Probability of not being decrypted
  p_not_decrypted = 2/5 ∧ (1 - p_not_decrypted) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_decryption_probability_l1266_126627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l1266_126683

-- Define the sets S and T
def S : Set ℝ := {x | (x - 2) * (x - 3) ≥ 0}
def T : Set ℝ := {x | x > 0}

-- State the theorem
theorem set_intersection_theorem :
  S ∩ T = Set.Ioc 0 2 ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l1266_126683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_when_eccentricity_sqrt2_div_2_eccentricity_range_when_dot_product_less_than_7_l1266_126616

-- Define the ellipse structure
structure Ellipse where
  m : ℝ
  eccentricity : ℝ

-- Define the focus and directrices
def right_focus (e : Ellipse) : ℝ × ℝ := (e.m, 0)
def left_directrix (e : Ellipse) : ℝ → ℝ := λ x ↦ -e.m - 1
def right_directrix (e : Ellipse) : ℝ → ℝ := λ x ↦ e.m + 1

-- Define points A and B
def point_A (e : Ellipse) : ℝ × ℝ := (-e.m - 1, -e.m - 1)
def point_B (e : Ellipse) : ℝ × ℝ := (e.m + 1, e.m + 1)

-- Vector AF and FB
def vector_AF (e : Ellipse) : ℝ × ℝ := (2 * e.m + 1, e.m + 1)
def vector_FB (e : Ellipse) : ℝ × ℝ := (1, e.m + 1)

-- Dot product of AF and FB
def dot_product_AF_FB (e : Ellipse) : ℝ := 
  (vector_AF e).1 * (vector_FB e).1 + (vector_AF e).2 * (vector_FB e).2

-- Theorem 1: Ellipse equation when eccentricity is √2/2
theorem ellipse_equation_when_eccentricity_sqrt2_div_2 (e : Ellipse) :
  e.eccentricity = Real.sqrt 2 / 2 →
  ∃ (x y : ℝ), x^2 / 2 + y^2 = 1 := by
  sorry

-- Theorem 2: Eccentricity range when AF · FB < 7
theorem eccentricity_range_when_dot_product_less_than_7 (e : Ellipse) :
  dot_product_AF_FB e < 7 →
  0 < e.eccentricity ∧ e.eccentricity < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_when_eccentricity_sqrt2_div_2_eccentricity_range_when_dot_product_less_than_7_l1266_126616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_problem_l1266_126678

/-- Vegetable cost and pricing problem -/
theorem vegetable_problem :
  -- Part 1: Defining the equations
  let m : ℝ := 10
  let n : ℝ := 14
  -- Part 2: Defining the constraints
  let total_kg : ℕ := 100
  let max_type_a : ℕ := 60
  let max_investment : ℝ := 1168
  -- Part 3: Defining profit and donation
  let profit (x : ℝ) := 2 * x + 400
  let max_profit : ℝ := profit 60
  -- Theorem statements
  (15 * m + 20 * n = 430 ∧ 10 * m + 8 * n = 212) ∧
  (∀ x : ℕ, 0 < x ∧ x ≤ max_type_a ∧ (10 * x + 14 * (total_kg - x) ≤ max_investment) →
    x = 58 ∨ x = 59 ∨ x = 60) ∧
  (∃ a : ℝ, a = 1.8 ∧
    ∀ a' : ℝ, ((16 - 10 - 2 * a') * 60 + (18 - 14 - a') * 40 ≥ (10 * 60 + 14 * 40) * 0.2) →
    a' ≤ a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_problem_l1266_126678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_values_l1266_126671

theorem polynomial_constant_values (n : ℕ) (hn : n ≥ 4) :
  ∀ p : Polynomial ℤ,
  (∀ k : ℕ, k ≤ n + 1 → 0 ≤ p.eval (k : ℤ) ∧ p.eval (k : ℤ) ≤ n) →
  ∀ i j : ℕ, i ≤ n + 1 → j ≤ n + 1 → p.eval (i : ℤ) = p.eval (j : ℤ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_values_l1266_126671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1266_126673

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 ≠ 0 ∧ a = k * 7767) :
  Int.gcd (6 * a^2 + 49 * a + 108) (2 * a + 9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1266_126673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_winning_strategy_l1266_126613

/-- Represents a player in the game -/
inductive Player
| Bela
| Jenn

/-- Represents the game state -/
structure GameState where
  n : ℕ
  selected : Set ℝ
  currentPlayer : Player

/-- Represents a valid move in the game -/
def validMove (state : GameState) (move : ℝ) : Prop :=
  move ∈ Set.Icc 0 (state.n : ℝ) ∧
  ∀ x ∈ state.selected, |move - x| > 2

/-- Represents the game result -/
inductive GameResult
| BelaCertainWin
| JennCertainWin

/-- Placeholder for the optimal strategy function -/
noncomputable def optimalStrategy (state : GameState) : GameResult :=
  GameResult.BelaCertainWin  -- Placeholder implementation

/-- The main theorem stating Bela's winning strategy exists -/
theorem bela_winning_strategy (n : ℕ) (h : n > 10) :
  GameResult.BelaCertainWin = 
    (let initialState : GameState := ⟨n, ∅, Player.Bela⟩
     optimalStrategy initialState) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_winning_strategy_l1266_126613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_specific_equation_l1266_126672

/-- Represents an ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_specific_equation (e : Ellipse) 
  (h_point : e.equation 1 (3/2))
  (h_ecc : e.eccentricity = 1/2) :
  ∀ x y : ℝ, e.equation x y ↔ x^2/4 + y^2/3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_specific_equation_l1266_126672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l1266_126649

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem inverse_function_sum (p q r s : ℝ) :
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (∀ x, g p q r s (g p q r s x) = x) →
  p + s = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l1266_126649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1266_126623

-- Define the function
noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x)

-- State the theorem
theorem f_monotone_increasing_interval :
  ∃ (a : ℝ), ∀ (x y : ℝ), x > a ∧ y > x → f x < f y :=
by
  -- We claim that a = 2 satisfies the condition
  use 2
  -- Let x and y be real numbers satisfying the conditions
  intro x y hxy
  -- Extract the conditions from hxy
  have hx : x > 2 := hxy.1
  have hyx : y > x := hxy.2
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l1266_126623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_days_is_30_l1266_126602

/-- Represents the total amount of work to be done -/
noncomputable def W : ℝ := 1

/-- The number of days B takes to complete the work -/
noncomputable def B_days : ℝ := 30

/-- The number of days C takes to complete the work -/
noncomputable def C_days : ℝ := 29.999999999999996

/-- The number of days A worked -/
noncomputable def A_work_days : ℝ := 10

/-- The number of days B worked -/
noncomputable def B_work_days : ℝ := 10

/-- The number of days C worked -/
noncomputable def C_work_days : ℝ := 10

/-- The rate at which B completes work per day -/
noncomputable def B_rate : ℝ := W / B_days

/-- The rate at which C completes work per day -/
noncomputable def C_rate : ℝ := W / C_days

theorem A_work_days_is_30 :
  ∃ (A_days : ℝ), 
    A_days > 0 ∧
    (let A_rate := W / A_days
     A_work_days * A_rate + B_work_days * B_rate + C_work_days * C_rate = W) ∧
    A_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_days_is_30_l1266_126602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberto_outfits_l1266_126628

/-- The number of pairs of trousers Roberto has -/
def trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def jackets : ℕ := 4

/-- The number of pairs of shoes Roberto has -/
def shoes : ℕ := 2

/-- An outfit consists of a pair of trousers, a shirt, a jacket, and a pair of shoes -/
def outfit := Fin trousers × Fin shirts × Fin jackets × Fin shoes

/-- The total number of different outfits Roberto can create -/
def total_outfits : ℕ := trousers * shirts * jackets * shoes

theorem roberto_outfits : total_outfits = 320 := by
  unfold total_outfits trousers shirts jackets shoes
  rfl

#eval total_outfits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberto_outfits_l1266_126628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_to_line_specific_centroid_distance_l1266_126620

/-- Given a triangle ABC with perpendicular distances from its vertices to a line,
    the perpendicular distance from its centroid to the same line is the average of these distances. -/
theorem centroid_distance_to_line (a b c : ℝ) :
  (a + b + c) / 3 = ([a, b, c].sum) / 3 := by
  simp
  ring

/-- For a triangle with perpendicular distances 15, 8, and 30 from its vertices to a line,
    the perpendicular distance from its centroid to the same line is 53/3. -/
theorem specific_centroid_distance :
  (15 + 8 + 30) / 3 = 53 / 3 := by
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_to_line_specific_centroid_distance_l1266_126620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_pot_of_coffee_l1266_126658

/-- The cost per pot of coffee given the conditions in the problem -/
theorem cost_per_pot_of_coffee 
  (coffee_per_donut : ℝ)
  (coffee_per_pot : ℝ)
  (donuts_consumed : ℝ)
  (total_spent : ℝ) :
  coffee_per_donut = 2 →
  coffee_per_pot = 12 →
  donuts_consumed = 36 →
  total_spent = 18 →
  (total_spent / ((donuts_consumed * coffee_per_donut) / coffee_per_pot)) = 3 := by
  sorry

#check cost_per_pot_of_coffee

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_pot_of_coffee_l1266_126658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_block_addition_l1266_126633

/-- Represents the new water depth in a tank after adding a cubic block -/
noncomputable def new_water_depth (a : ℝ) : ℝ :=
  if a ≥ 28 then 30
  else if a = 8 then 10
  else if 8 < a ∧ a < 28 then a + 2
  else (5/4) * a

theorem water_depth_after_block_addition (a : ℝ) (h : a ≤ 30) :
  let tank_depth : ℝ := 30
  let tank_length : ℝ := 25
  let tank_width : ℝ := 20
  let block_edge : ℝ := 10
  let initial_water_volume := a * tank_length * tank_width
  let block_volume := block_edge ^ 3
  let new_water_volume := initial_water_volume + block_volume
  new_water_depth a = 
    if new_water_volume ≥ tank_depth * tank_length * tank_width 
    then tank_depth
    else if new_water_volume = block_volume + 8 * tank_length * tank_width
    then 10
    else if new_water_volume > block_volume + 8 * tank_length * tank_width ∧
            new_water_volume < tank_depth * tank_length * tank_width
    then (new_water_volume / (tank_length * tank_width))
    else (new_water_volume - block_volume * (1 - block_edge / tank_width)) / 
         (tank_length * tank_width) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_block_addition_l1266_126633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l1266_126603

/-- A sequence defined by its first four terms and a general formula -/
def mySequence : ℕ → ℚ
  | 1 => 3/2
  | 2 => 1
  | 3 => 5/8
  | 4 => 3/8
  | n => (n^2 - 11*n + 34) / 16

/-- Theorem stating that the general term formula is correct for the given sequence -/
theorem sequence_formula_correct : ∀ n : ℕ, n > 0 → mySequence n = (n^2 - 11*n + 34) / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l1266_126603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_ratio_l1266_126687

/-- Given a triangle ABC with points E on AC and G on AB, if BE intersects AG at Q,
    BQ:QE = 2:1, and GQ:QA = 3:4, then BG/GA = 3/4 -/
theorem triangle_segment_ratio (A B C E G Q : EuclideanSpace ℝ (Fin 2)) : 
  (∃ t : ℝ, E = t • A + (1 - t) • C) →  -- E is on AC
  (∃ s : ℝ, G = s • A + (1 - s) • B) →  -- G is on AB
  (∃ r : ℝ, Q = r • B + (1 - r) • E) →  -- Q is on BE
  (∃ u : ℝ, Q = u • A + (1 - u) • G) →  -- Q is on AG
  (dist B Q) / (dist Q E) = 2 / 1 →     -- BQ:QE = 2:1
  (dist G Q) / (dist Q A) = 3 / 4 →     -- GQ:QA = 3:4
  (dist B G) / (dist G A) = 3 / 4       -- BG:GA = 3:4
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_ratio_l1266_126687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_eighty_ninth_term_l1266_126682

def sequenceMultiples (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => sorry

theorem two_hundred_eighty_ninth_term :
  sequenceMultiples 289 = 2737 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_eighty_ninth_term_l1266_126682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_rectangle_count_l1266_126660

/-- The number of rectangles with integer sides for a given perimeter -/
def count_rectangles (perimeter : ℕ) : ℕ :=
  (perimeter / 4)

/-- Theorem: The number of rectangles with integer sides is equal for perimeters 1996 and 1998 -/
theorem equal_rectangle_count :
  count_rectangles 1996 = count_rectangles 1998 := by
  rfl

#eval count_rectangles 1996  -- Expected output: 499
#eval count_rectangles 1998  -- Expected output: 499

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_rectangle_count_l1266_126660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1266_126656

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3/4 * x + 5/4 else 2^x

-- Define the set of t that satisfies the equation
def solution_set : Set ℝ :=
  {t | f (f t) = 2^(f t)}

-- Theorem statement
theorem solution_set_characterization :
  solution_set = {t | t = -3 ∨ t ≥ -1/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1266_126656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OM_l1266_126666

-- Define the curve and line equations
def curve (x y : ℝ) : Prop := Real.sqrt 2 * x^2 + y^2 = 1
def line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the intersection points P and Q
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  curve P.1 P.2 ∧ line P.1 P.2 ∧ curve Q.1 Q.2 ∧ line Q.1 Q.2

-- Define the midpoint M of PQ
def midpoint_PQ (P Q M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

-- Theorem statement
theorem slope_of_OM (P Q M : ℝ × ℝ) :
  intersection_points P Q →
  midpoint_PQ P Q M →
  (M.2 / M.1 = Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OM_l1266_126666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_capital_ratio_l1266_126670

theorem partnership_capital_ratio :
  ∀ (total_capital profit : ℚ),
  total_capital > 0 →
  profit > 0 →
  let a_capital := (1 / 3) * total_capital;
  let b_capital := (1 / 4) * total_capital;
  let c_capital := (1 / 5) * total_capital;
  let d_capital := total_capital - (a_capital + b_capital + c_capital);
  let a_profit := 810;
  let total_profit := 2430;
  a_profit / total_profit = a_capital / total_capital →
  d_capital / total_capital = 13 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_capital_ratio_l1266_126670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l1266_126691

open Real

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := exp x - x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧ ∀ x ∈ Set.Icc (-1) 1, f x ≤ f c ∧ f c = exp 1 - 1 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l1266_126691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_value_l1266_126667

/-- Represents an investment with its value and return rate -/
structure Investment where
  value : ℝ
  returnRate : ℝ

/-- Calculates the total return of two investments -/
noncomputable def totalReturn (inv1 inv2 : Investment) : ℝ :=
  inv1.value * inv1.returnRate + inv2.value * inv2.returnRate

/-- Calculates the combined return rate of two investments -/
noncomputable def combinedReturnRate (inv1 inv2 : Investment) : ℝ :=
  totalReturn inv1 inv2 / (inv1.value + inv2.value)

theorem second_investment_value
  (inv1 inv2 : Investment)
  (h1 : inv1.value = 500)
  (h2 : inv1.returnRate = 0.07)
  (h3 : inv2.returnRate = 0.11)
  (h4 : combinedReturnRate inv1 inv2 = 0.10) :
  inv2.value = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_value_l1266_126667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_15_30_0_2_l1266_126622

noncomputable def arithmetic_series_sum (a₁ aₙ d : ℝ) : ℝ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_15_30_0_2 :
  arithmetic_series_sum 15 30 0.2 = 1710 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_15_30_0_2_l1266_126622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_distances_l1266_126614

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ → Prop :=
  λ ρ => ρ * (Real.sin θ)^2 = 4 * Real.cos θ

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + 2/Real.sqrt 5 * t, 1 + 1/Real.sqrt 5 * t)

/-- Point P -/
def point_P : ℝ × ℝ := (1, 1)

/-- Theorem stating the Cartesian equation of curve C and the sum of distances |PA| + |PB| -/
theorem curve_C_and_distances :
  (∀ x y : ℝ, (∃ θ : ℝ, curve_C θ (Real.sqrt (x^2 + y^2)) ∧ x = Real.sqrt (x^2 + y^2) * Real.cos θ ∧ y = Real.sqrt (x^2 + y^2) * Real.sin θ) ↔ y^2 = 4*x) ∧
  (∃ t₁ t₂ : ℝ, 
    let A := line_l t₁
    let B := line_l t₂
    (A.1)^2 = 4*(A.2) ∧ (B.1)^2 = 4*(B.2) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 4 * Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_distances_l1266_126614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approximation_l1266_126661

-- Define the number of revolutions
def revolutions : ℕ := 600

-- Define the total distance covered in centimeters
def total_distance : ℝ := 844.8

-- Define pi as a constant
noncomputable def π : ℝ := Real.pi

-- Define the radius of the wheel
noncomputable def radius : ℝ := total_distance / (2 * π * (revolutions : ℝ))

-- Theorem statement
theorem wheel_radius_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |radius - 0.224| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approximation_l1266_126661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_two_to_one_l1266_126638

noncomputable section

def total_distance : ℝ := 40
def half_distance : ℝ := 20
def second_half_time : ℝ := 24
def time_difference : ℝ := 12

def first_half_time : ℝ := second_half_time - time_difference

def speed_before : ℝ := half_distance / first_half_time
def speed_after : ℝ := half_distance / second_half_time

theorem speed_ratio_is_two_to_one :
  speed_before / speed_after = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_two_to_one_l1266_126638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_is_21_l1266_126647

/-- The speed of the cyclist in miles per hour -/
noncomputable def cyclist_speed : ℝ := 21

/-- The speed of the hiker in miles per hour -/
noncomputable def hiker_speed : ℝ := 7

/-- The time in hours that the cyclist travels after passing the hiker -/
noncomputable def cyclist_travel_time : ℝ := 5 / 60

/-- The time in hours that the cyclist waits for the hiker -/
noncomputable def cyclist_wait_time : ℝ := 15 / 60

theorem cyclist_speed_is_21 :
  cyclist_speed = 21 ∧
  hiker_speed * (cyclist_travel_time + cyclist_wait_time) = cyclist_speed * cyclist_travel_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_is_21_l1266_126647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l1266_126643

theorem solution_set_inequality (f : ℝ → ℝ) (hf1 : f 1 = 3) 
  (hf2 : ∀ x : ℝ, f x + (deriv (deriv f)) x < 2) :
  {x : ℝ | Real.exp x * f x > 2 * Real.exp x + Real.exp 1} = {x : ℝ | x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l1266_126643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_renovation_theorem_l1266_126612

/-- Represents the cost of renovating schools in millions of yuan -/
structure RenovationCost where
  typeA : ℝ
  typeB : ℝ

/-- Represents a renovation plan -/
structure RenovationPlan where
  typeACount : ℕ
  typeBCount : ℕ

def totalCost (cost : RenovationCost) (plan : RenovationPlan) : ℝ :=
  cost.typeA * plan.typeACount + cost.typeB * plan.typeBCount

def localContribution (plan : RenovationPlan) : ℝ :=
  0.1 * plan.typeACount + 0.15 * plan.typeBCount

theorem renovation_theorem (cost : RenovationCost) :
  (cost.typeA + 2 * cost.typeB = 2.3 ∧
   2 * cost.typeA + cost.typeB = 2.05) →
  (∀ plan : RenovationPlan,
    plan.typeACount + plan.typeBCount = 6 →
    totalCost cost plan - localContribution plan ≤ 3.8 →
    localContribution plan ≥ 0.7) →
  (cost.typeA = 60 ∧ cost.typeB = 85 ∧
   ∃ optimalPlan : RenovationPlan,
     optimalPlan.typeACount = 4 ∧
     optimalPlan.typeBCount = 2 ∧
     totalCost cost optimalPlan = 410 ∧
     ∀ plan : RenovationPlan,
       plan.typeACount + plan.typeBCount = 6 →
       totalCost cost plan ≥ totalCost cost optimalPlan) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_renovation_theorem_l1266_126612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_line_and_triangle_area_l1266_126605

noncomputable section

-- Define the points
def A : ℝ × ℝ := (5, 1)
def B : ℝ × ℝ := (5, -1)  -- Symmetric to A about x-axis
def C : ℝ × ℝ := (-5, -1)  -- Symmetric to A about origin

-- Define the midpoints
def midpoint_BA : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def midpoint_BC : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 5 = 0

-- Define the area of a triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem midpoint_line_and_triangle_area :
  (∀ x y, (x, y) = midpoint_BA ∨ (x, y) = midpoint_BC → line_equation x y) ∧
  triangle_area A B C = 10 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_line_and_triangle_area_l1266_126605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l1266_126675

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x > f a y) → 1 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l1266_126675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1266_126652

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n+1 => r * geometric_sequence a₁ r n

theorem geometric_sequence_sum (a₁ r : ℝ) (h1 : a₁ = 1) (h2 : r = -2) :
  let a := geometric_sequence a₁ r
  a 0 + |a 1| + a 2 + |a 3| = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1266_126652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drama_club_attendance_l1266_126618

theorem drama_club_attendance (total_tickets : ℕ) (adult_price student_price : ℚ) 
  (total_amount : ℚ) (student_count : ℕ) 
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : student_price = 6)
  (h4 : total_amount = 16200)
  (h5 : student_count = 300)
  : (total_tickets - student_count : ℕ) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drama_club_attendance_l1266_126618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1266_126617

-- Define the expressions
noncomputable def expr_A (x : ℝ) := Real.sqrt (8 * x)
noncomputable def expr_B (a b : ℝ) := Real.sqrt (3 * a^2 * b)
noncomputable def expr_C (x y : ℝ) := Real.sqrt (4 * x^2 + 25 * y^2)
noncomputable def expr_D (x : ℝ) := Real.sqrt (x / 2)

-- Define what it means for an expression to be simplifiable
def is_simplifiable {α : Type*} (e : α → ℝ) : Prop :=
  ∃ (f : α → ℝ), (∀ x, e x = f x) ∧ (∃ y, f y ≠ e y)

-- State the theorem
theorem simplest_quadratic_radical :
  ¬(is_simplifiable (λ (p : ℝ × ℝ) => expr_C p.1 p.2)) ∧ 
  (is_simplifiable expr_A ∨ is_simplifiable (λ (p : ℝ × ℝ) => expr_B p.1 p.2) ∨ is_simplifiable expr_D) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1266_126617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_max_l1266_126693

theorem triangle_sine_sum_max (A B C : Real) : 
  A + B + C = Real.pi → 
  A > 0 → B > 0 → C > 0 →
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_max_l1266_126693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_in_pascal_l1266_126630

/-- Pascal's triangle entry at row n and column k -/
def pascal : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n+1, 0 => 1
  | n+1, k+1 => pascal n k + pascal n (k+1)

/-- Predicate for a number being a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem smallest_four_digit_in_pascal :
  ∃ (n k : ℕ), pascal n k = 1000 ∧
    isFourDigit (pascal n k) ∧
    ∀ (m l : ℕ), isFourDigit (pascal m l) → pascal n k ≤ pascal m l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_in_pascal_l1266_126630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_l1266_126637

theorem ladder_slide (ladder_length initial_distance slip : ℝ) 
  (h1 : ladder_length = 40)
  (h2 : initial_distance = 9)
  (h3 : slip = 6) : 
  ∃ (slide : ℝ), 
    let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
    let new_height := initial_height - slip
    slide = Real.sqrt (45 + 12 * Real.sqrt 1519) - 9 ∧
    Real.sqrt (new_height^2 + (initial_distance + slide)^2) = ladder_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_l1266_126637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_equalization_theorem_l1266_126655

/-- Represents a circular arrangement of boxes with coins -/
structure CoinCircle where
  boxes : Fin 7 → ℕ
  deriving Repr

/-- Defines a valid move in the coin equalization game -/
def is_valid_move (c : CoinCircle) (i j : Fin 7) : Prop :=
  (i = j + 1) ∨ (j = i + 1) ∨ (i = 0 ∧ j = 6) ∨ (i = 6 ∧ j = 0)

/-- Applies a move to the coin circle -/
def apply_move (c : CoinCircle) (i j : Fin 7) : CoinCircle :=
  { boxes := fun k => if k = j then c.boxes k + 1 else if k = i then c.boxes k - 1 else c.boxes k }

/-- Checks if all boxes have the same number of coins -/
def is_equalized (c : CoinCircle) : Prop :=
  ∀ i j : Fin 7, c.boxes i = c.boxes j

/-- The minimum number of moves required to equalize the coins -/
noncomputable def min_moves_to_equalize (c : CoinCircle) : ℕ :=
  sorry -- Definition omitted for brevity

theorem coin_equalization_theorem :
  let initial_circle : CoinCircle := ⟨fun i => [8, 14, 18, 6, 10, 20, 15][i.val]⟩
  min_moves_to_equalize initial_circle = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_equalization_theorem_l1266_126655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1266_126631

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x - 2 * Real.sqrt 3 * Real.sin x ^ 2 + Real.sqrt 3

theorem f_properties :
  ∃ (α : ℝ), 
    π / 2 < α ∧ α < π ∧
    f (α / 2) = 10 / 13 ∧
    (∀ x, f x = 2 * Real.sin (2 * x + π / 3)) ∧
    (∀ y ∈ Set.Icc 0 (π / 2), -Real.sqrt 3 ≤ f y ∧ f y ≤ 2) ∧
    Real.sin α = (5 + 12 * Real.sqrt 3) / 26 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1266_126631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_range_l1266_126690

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x + Real.cos x

theorem perpendicular_tangents_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (a + Real.cos x₁ - Real.sin x₁) * (a + Real.cos x₂ - Real.sin x₂) = -1) ↔ 
  a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

#check perpendicular_tangents_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_range_l1266_126690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_eq_2014_l1266_126654

/-- A sequence {aₙ} defined by a₁ = 1 and aₙ₊₁ = aₙ + 1 for all n ≥ 1 -/
def a : ℕ → ℕ
  | 0 => 1  -- Add this case for n = 0
  | n + 1 => a n + 1

/-- The 2014th term of the sequence {aₙ} is equal to 2014 -/
theorem a_2014_eq_2014 : a 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_eq_2014_l1266_126654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_locus_l1266_126696

-- Define the plane as a real inner product space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the lines a, b, c as affine subspaces
variable (a b c : AffineSubspace ℝ V)

-- Axiom: a, b, c are distinct lines
axiom lines_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define a point in the space
variable (P : V)

-- Define the property of being a circumcenter
def is_circumcenter (P A B C : V) : Prop :=
  ‖P - A‖ = ‖P - B‖ ∧ ‖P - B‖ = ‖P - C‖

-- Theorem statement
theorem circumcenter_locus :
  ∃ (A B C : V), A ∈ a ∧ B ∈ b ∧ C ∈ c ∧ is_circumcenter P A B C :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_locus_l1266_126696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_M_l1266_126685

/-- Represents a point in the rectangular array -/
structure Point where
  row : Fin 6
  col : Nat

/-- The number of columns in the array -/
def M : Nat := 35

/-- The numbering of a point in the left-to-right scheme -/
def x (p : Point) : Nat :=
  p.row.val * M + p.col

/-- The numbering of a point in the top-to-bottom scheme -/
def y (p : Point) : Nat :=
  (p.col - 1) * 6 + p.row.val + 1

/-- The six selected points -/
def P : Fin 6 → Point := sorry

/-- The conditions given in the problem -/
axiom row_assignment : ∀ i : Fin 6, (P i).row = i

axiom point_relations : 
  x (P 0) = y (P 1) ∧
  x (P 1) = y (P 0) ∧
  x (P 2) = y (P 3) ∧
  x (P 3) = y (P 2) ∧
  x (P 4) = y (P 5) ∧
  x (P 5) = y (P 4)

axiom valid_columns : ∀ i : Fin 6, 0 < (P i).col ∧ (P i).col ≤ M

/-- The theorem to be proved -/
theorem smallest_M : M = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_M_l1266_126685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_min_circumcircle_area_l1266_126663

-- Define the fixed point F
def F : ℝ × ℝ := (0, 1)

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -1}

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define a function to check if a circle passes through F and is tangent to l
def is_valid_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - F.1)^2 + (center.2 - F.2)^2 = radius^2 ∧
  |center.2 + 1| = radius

-- Theorem for the trajectory equation
theorem trajectory_equation :
  ∀ p : ℝ × ℝ, p ∈ C ↔ ∃ r : ℝ, is_valid_circle p r :=
sorry

-- Define a function to calculate the area of a circle given its diameter
noncomputable def circle_area (diameter : ℝ) : ℝ := Real.pi * (diameter / 2)^2

-- Theorem for the minimum area of the circumcircle
theorem min_circumcircle_area :
  ∃ min_area : ℝ, min_area = 4 * Real.pi ∧
  ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
    let diameter := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    circle_area diameter ≥ min_area :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_min_circumcircle_area_l1266_126663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_sector_area_l1266_126641

/-- The radius of the circular spinner in centimeters -/
noncomputable def spinner_radius : ℝ := 5

/-- The probability of winning on one spin -/
noncomputable def win_probability : ℝ := 2/5

/-- The area of a circle with radius r -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- Theorem: The area of the WIN sector on a circular spinner with radius 5 cm 
    and win probability 2/5 is 10π square centimeters -/
theorem win_sector_area : 
  win_probability * circle_area spinner_radius = 10 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_sector_area_l1266_126641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_negative_6_l1266_126676

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We define a₀ = 3 to handle the zero case
  | 1 => 3
  | 2 => 6
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_2009_equals_negative_6 : sequence_a 2009 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_equals_negative_6_l1266_126676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_properties_chain_example_l1266_126662

-- Define the machine operation
def machine (x : ℚ) : ℚ :=
  if x ≤ 1/2 then 2*x else 2*(1-x)

-- Define the chain operation
def chain (x : ℚ) : ℕ → ℚ
  | 0 => x
  | n+1 => machine (chain x n)

theorem machine_properties :
  -- 1. Fixed point of the machine
  ∃ x : ℚ, 0 < x ∧ x < 1 ∧ machine x = x ∧ x = 2/3 ∧
  -- 2. Possible first numbers if fourth number is 1
  (∀ y : ℚ, 0 < y ∧ y < 1 ∧ chain y 3 = 1 → y ∈ ({1/8, 3/8, 5/8, 7/8} : Set ℚ)) ∧
  -- 3. First and eighth numbers are equal for m = 43, 127, 129
  (∀ m : ℕ, m ∈ ({43, 127, 129} : Set ℕ) → chain (2/m) 0 = chain (2/m) 7) :=
by sorry

-- Additional theorem to demonstrate the chain for the first part of the question
theorem chain_example :
  let start := 3/11
  chain start 0 = 3/11 ∧
  chain start 1 = 6/11 ∧
  chain start 2 = 10/11 ∧
  chain start 3 = 2/11 ∧
  chain start 4 = 4/11 ∧
  chain start 5 = 8/11 ∧
  chain start 6 = 6/11 ∧
  chain start 7 = 10/11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_properties_chain_example_l1266_126662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1266_126609

open Real

theorem triangle_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  tan B + tan C = (Real.sqrt 3 * cos A) / (cos B * cos C) →
  A = π/3 ∧ 
  (a = Real.sqrt 6 → 3 * Real.sqrt 2 < b + c ∧ b + c ≤ 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1266_126609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l1266_126642

theorem sin_2_cos_3_tan_4_negative : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l1266_126642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_points_order_l1266_126619

/-- The quadratic function f(x) = x² - 6x + c --/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Point A on the graph of f --/
def A (c : ℝ) : ℝ × ℝ := (1, f c 1)

/-- Point B on the graph of f --/
noncomputable def B (c : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2, f c (2 * Real.sqrt 2))

/-- Point C on the graph of f --/
def C (c : ℝ) : ℝ × ℝ := (4, f c 4)

theorem quadratic_points_order (c : ℝ) :
  (C c).2 < (B c).2 ∧ (B c).2 < (A c).2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_points_order_l1266_126619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_rate_definition_l1266_126657

/-- Frequency of an object -/
def frequency {α : Type*} [DecidableEq α] (object : α) (data : List α) : ℕ :=
  (data.filter (· == object)).length

/-- Total number of observations -/
def totalObservations {α : Type*} (data : List α) : ℕ :=
  data.length

/-- Frequency rate of an object -/
noncomputable def frequencyRate {α : Type*} [DecidableEq α] (object : α) (data : List α) : ℚ :=
  (frequency object data : ℚ) / (totalObservations data : ℚ)

/-- Theorem: The frequency rate is the ratio of an object's frequency to the total number of observations -/
theorem frequency_rate_definition {α : Type*} [DecidableEq α] (object : α) (data : List α) :
  frequencyRate object data = (frequency object data : ℚ) / (totalObservations data : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_rate_definition_l1266_126657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_equation_l1266_126664

/-- Data for island areas -/
noncomputable def island_areas : List ℝ := [6, 15, 25, 34]

/-- Data for number of plant species -/
noncomputable def plant_species : List ℝ := [5, 10, 15, 19]

/-- Sum of squared island areas for first 4 samples -/
noncomputable def sum_x_squared : ℝ := 2042

/-- Sum of products of island areas and plant species for first 4 samples -/
noncomputable def sum_xy : ℝ := 1201

/-- Number of samples -/
def n : ℕ := 4

/-- Mean of island areas -/
noncomputable def mean_x : ℝ := (List.sum island_areas) / n

/-- Mean of plant species -/
noncomputable def mean_y : ℝ := (List.sum plant_species) / n

/-- Slope of the linear regression line -/
noncomputable def b : ℝ := (sum_xy - n * mean_x * mean_y) / (sum_x_squared - n * mean_x^2)

/-- Intercept of the linear regression line -/
noncomputable def a : ℝ := mean_y - b * mean_x

theorem linear_regression_equation :
  b = 0.5 ∧ a = 2.25 := by
  sorry

#check linear_regression_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_equation_l1266_126664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_order_l1266_126646

/-- Inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_order (k x₁ x₂ x₃ : ℝ) :
  k < 0 →
  inverse_proportion k x₁ = 1 →
  inverse_proportion k x₂ = -5 →
  inverse_proportion k x₃ = 3 →
  x₁ < x₃ ∧ x₃ < x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_order_l1266_126646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_gates_is_four_at_least_four_gates_needed_l1266_126634

/-- Represents the ticket checking scenario during Spring Festival travel rush -/
structure TicketCheckingScenario where
  a : ℝ  -- Initial number of passengers
  x : ℝ  -- Rate of passenger increase per minute
  y : ℝ  -- Ticket checking speed per gate per minute
  h_a_pos : a > 0  -- Condition: a > 0

/-- Conditions for the ticket checking scenario -/
def scenario_conditions (s : TicketCheckingScenario) : Prop :=
  s.a + 30 * s.x = 30 * s.y ∧  -- Condition for 1 gate
  s.a + 10 * s.x = 2 * 10 * s.y  -- Condition for 2 gates

/-- The minimum number of gates needed to check all passengers within 5 minutes -/
noncomputable def min_gates_needed (s : TicketCheckingScenario) : ℕ :=
  Int.toNat ⌈(s.a + 5 * s.x) / (5 * s.y)⌉

/-- Theorem stating that the minimum number of gates needed is 4 -/
theorem min_gates_is_four (s : TicketCheckingScenario) 
  (h_cond : scenario_conditions s) : min_gates_needed s = 4 := by
  sorry

/-- Main theorem proving that at least 4 gates are needed -/
theorem at_least_four_gates_needed (s : TicketCheckingScenario) 
  (h_cond : scenario_conditions s) : min_gates_needed s ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_gates_is_four_at_least_four_gates_needed_l1266_126634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1266_126611

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Line where
  point1 : Point
  point2 : Point

-- Define the given points and circles
variable (O₁ O₂ X Y A B K L M N : Point)
variable (circle1 circle2 : Circle)

-- Define membership for points in circles and lines
def Point.in_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

def Point.on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.point1.y) * (l.point2.x - l.point1.x) = (p.x - l.point1.x) * (l.point2.y - l.point1.y)

-- Define tangency
def Line.tangent_to (l : Line) (c : Circle) : Prop :=
  ∃ p : Point, p.in_circle c ∧ p.on_line l ∧
  ∀ q : Point, q ≠ p → q.on_line l → ¬q.in_circle c

-- Define the conditions
axiom intersect : X.in_circle circle1 ∧ X.in_circle circle2 ∧ Y.in_circle circle1 ∧ Y.in_circle circle2
axiom tangent_AB : (Line.mk A B).tangent_to circle1 ∧ (Line.mk A B).tangent_to circle2
axiom A_on_circle1 : A.in_circle circle1
axiom B_on_circle2 : B.in_circle circle2
axiom tangent_X_O1 : (Line.mk X K).tangent_to circle1
axiom tangent_X_O2 : (Line.mk X L).tangent_to circle2
axiom K_on_O1O2 : K.on_line (Line.mk O₁ O₂)
axiom L_on_O1O2 : L.on_line (Line.mk O₁ O₂)
axiom M_on_circle2 : M.in_circle circle2 ∧ M ≠ B ∧ M.on_line (Line.mk B L)
axiom N_on_circle1 : N.in_circle circle1 ∧ N ≠ A ∧ N.on_line (Line.mk A K)

-- Define the theorem to be proved
theorem lines_concurrent :
  ∃ P : Point, P.on_line (Line.mk A M) ∧ P.on_line (Line.mk B N) ∧ P.on_line (Line.mk O₁ O₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1266_126611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_is_odd_function_l1266_126606

-- Define the function f(x) = sin(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Theorem statement
theorem sin_is_odd_function : 
  ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  unfold f
  simp [Real.sin_neg]
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_is_odd_function_l1266_126606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l1266_126669

/-- Given vectors a and b, if (a - l*b) is orthogonal to b, then l = 3/5 -/
theorem orthogonal_vectors (a b : ℝ × ℝ) (l : ℝ) :
  a = (1, 3) →
  b = (3, 4) →
  (a - l • b) • b = 0 →
  l = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l1266_126669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contracting_function_bound_l1266_126659

/-- A function f: [0,1] → ℝ satisfying the given conditions -/
def ContractingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1) ∧
  f 0 = f 1 ∧
  ∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂ - x₁|

/-- The main theorem to be proved -/
theorem contracting_function_bound
  (f : ℝ → ℝ) (h : ContractingFunction f) :
  ∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → |f x₂ - f x₁| < (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contracting_function_bound_l1266_126659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l1266_126684

theorem no_solution_for_equation (a b m : ℕ) (ha : a > 0) (hb : b > 0) (hm : m > 0) : 
  (a * b) ^ 2015 ≠ (a ^ 2 + b ^ 2) ^ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l1266_126684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_l1266_126653

theorem chess_tournament (m : ℕ) : m > 0 → (
  let total_players := 4 * m
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 2 * (total_matches / 5)
  let men_wins := 3 * (total_matches / 5)
  (women_wins + men_wins = total_matches) ∧
  (women_wins = 2 * m * (4 * m - 1)) ∧
  (men_wins = 6 * m * (4 * m - 1))
) → m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_l1266_126653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_calculation_l1266_126697

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Number of Carbon atoms in the compound -/
def C_atoms : ℕ := 10

/-- Number of Hydrogen atoms in the compound -/
def H_atoms : ℕ := 8

/-- Number of Oxygen atoms in the compound -/
def O_atoms : ℕ := 4

/-- Number of Nitrogen atoms in the compound -/
def N_atoms : ℕ := 3

/-- Number of Sulfur atoms in the compound -/
def S_atoms : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  C_weight * C_atoms +
  H_weight * H_atoms +
  O_weight * O_atoms +
  N_weight * N_atoms +
  S_weight * S_atoms

theorem molecular_weight_calculation :
  ∃ ε > 0, |molecular_weight - 298.334| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_molecular_weight_calculation_l1266_126697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1266_126698

open Complex

theorem trigonometric_equation_solution (z : ℂ) : 
  sin z + sin (2*z) + sin (3*z) = cos z + cos (2*z) + cos (3*z) ↔ 
  (∃ (k : ℤ), z = (2*π/3)*(3*k + 1) ∨ z = (2*π/3)*(3*k - 1)) ∨
  (∃ (k : ℤ), z = (π/8)*(4*k + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1266_126698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1266_126688

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the circumradius
noncomputable def circumradius (t : Triangle) : ℝ := Real.sqrt 3

-- State the given condition
axiom condition (t : Triangle) : t.a * Real.sin t.C + Real.sqrt 3 * t.c * Real.cos t.A = 0

-- Helper function for area (not part of the main statement)
noncomputable def area (t : Triangle) : ℝ := 
  (1 / 2) * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_properties (t : Triangle) :
  (t.a = 3) ∧
  (t.b + t.c = Real.sqrt 11 → area t = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1266_126688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_population_equation_l1266_126650

/-- The original population of the city --/
def original_population : ℝ := sorry

/-- The population increase --/
def population_increase : ℝ := 2000

/-- The percentage of population remaining after decrease --/
def remaining_percentage : ℝ := 0.85

/-- The final population difference compared to original plus increase --/
def final_population_difference : ℝ := 50

/-- Theorem stating the relationship between the original population and the changes --/
theorem city_population_equation :
  remaining_percentage * (original_population + population_increase) = 
  original_population + population_increase + final_population_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_population_equation_l1266_126650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_distribution_probability_l1266_126608

/-- A uniformly distributed random variable on [α, β] -/
structure UniformRV (α β : ℝ) where
  hαβ : α < β

/-- The probability density function for a uniform distribution -/
noncomputable def pdf (α β : ℝ) (X : UniformRV α β) : ℝ → ℝ :=
  λ x => if α ≤ x ∧ x ≤ β then 1 / (β - α) else 0

/-- The probability that X falls within an interval (γ, δ) -/
noncomputable def prob (α β γ δ : ℝ) (X : UniformRV α β) : ℝ :=
  ∫ x in γ..δ, pdf α β X x

/-- Theorem stating the probability for a uniform distribution -/
theorem uniform_distribution_probability (α β γ δ : ℝ) (X : UniformRV α β) 
    (hγδ : α ≤ γ ∧ γ < δ ∧ δ ≤ β) : 
  prob α β γ δ X = (δ - γ) / (β - α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_distribution_probability_l1266_126608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_trajectory_l1266_126665

/-- Proves that the trajectory of the center of a circle passing through (0,3) and 
    tangent to y+3=0 satisfies the equation x^2 = 12y -/
theorem circle_center_trajectory 
  (center : ℝ × ℝ) 
  (passes_through : (0, 3) ∈ Metric.sphere center (dist center (0, 3)))
  (tangent_to : ∀ p ∈ Metric.sphere center (dist center (0, 3)), p.2 ≥ -3) :
  center.1^2 = 12 * center.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_trajectory_l1266_126665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_closed_id_is_operation_preserving_l1266_126639

-- Define a closed set
def is_closed (M : Set ℝ) : Prop :=
  ∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M ∧ (x * y) ∈ M

-- Define an additive function
def is_additive {M : Type*} [Add M] (f : M → M) : Prop :=
  ∀ x y, f (x + y) = f x + f y

-- Define a multiplicative function
def is_multiplicative {M : Type*} [Mul M] (f : M → M) : Prop :=
  ∀ x y, f (x * y) = f x * f y

-- Define an operation-preserving function
def is_operation_preserving {M : Type*} [Add M] [Mul M] (f : M → M) : Prop :=
  is_additive f ∧ is_multiplicative f

-- Define the set S
def S : Set ℝ := {x | ∃ m n : ℚ, x = Real.sqrt 3 * m + n}

-- Theorem 1: S is closed
theorem S_is_closed : is_closed S := by
  sorry

-- Theorem 2: The identity function on Q is operation-preserving
theorem id_is_operation_preserving : is_operation_preserving (id : ℚ → ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_closed_id_is_operation_preserving_l1266_126639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cut_volume_ratio_l1266_126680

/-- A square-based pyramid -/
structure SquarePyramid where
  height : ℝ
  base_side : ℝ

/-- Represents a cut made parallel to the base of a pyramid -/
noncomputable def parallel_cut (p : SquarePyramid) (h : ℝ) : ℝ × ℝ :=
  (h, p.height - h)

/-- Volume of a pyramid -/
noncomputable def pyramid_volume (p : SquarePyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

/-- Volume of a frustum (truncated pyramid) -/
noncomputable def frustum_volume (p : SquarePyramid) (cut : ℝ × ℝ) : ℝ :=
  pyramid_volume p - pyramid_volume ⟨cut.1, p.base_side * (cut.1 / p.height)⟩

/-- Theorem: The ratio of volumes in a pyramid cut 2/3 of the way up -/
theorem pyramid_cut_volume_ratio (p : SquarePyramid) :
  let cut := parallel_cut p ((2/3) * p.height)
  let small_pyramid_volume := pyramid_volume ⟨cut.1, p.base_side * (cut.1 / p.height)⟩
  let frustum_volume := frustum_volume p cut
  small_pyramid_volume / frustum_volume = 8 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cut_volume_ratio_l1266_126680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_zero_and_k_representation_l1266_126600

/-- A number whose decimal representation contains only digits 0 and k -/
def OnlyZeroAndK (x : ℝ) (k : ℕ) : Prop :=
  ∃ (digits : ℕ → ℕ), x = (∑' i, (digits i : ℝ) * 10^i) ∧ ∀ i, digits i = 0 ∨ digits i = k

/-- Theorem: Any positive real number can be represented as the sum of nine numbers,
    each with decimal representations containing only digits 0 and k -/
theorem sum_of_nine_zero_and_k_representation (a : ℝ) (k : ℕ) 
    (h_a : a > 0) (h_k : k ≠ 0) (h_k_digit : k < 10) :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ), 
    a = b₁ + b₂ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ + b₉ ∧
    (∀ i, i ∈ [1,2,3,4,5,6,7,8,9] → OnlyZeroAndK (List.get! [b₁,b₂,b₃,b₄,b₅,b₆,b₇,b₈,b₉] (i-1)) k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_zero_and_k_representation_l1266_126600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_a_value_l1266_126615

theorem sin_a_value (a : ℝ) (h1 : π / 2 < a ∧ a < π) (h2 : Real.sin (a - π / 6) = 3 / 5) :
  Real.sin a = (3 * Real.sqrt 3 - 4) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_a_value_l1266_126615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1266_126629

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, -1)
def C : ℝ × ℝ := (4, -3)

noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁))

def is_right_triangle (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∨
  (x₁ - x₂) * (x₃ - x₂) + (y₁ - y₂) * (y₃ - y₂) = 0 ∨
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = 0

theorem triangle_properties :
  triangle_area A B C = 19/2 ∧ ¬(is_right_triangle A B C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1266_126629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_range_l1266_126632

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f a x < 2) →
  a ∈ Set.union (Set.Ioo (Real.sqrt 2 / 2) 1) (Set.Ioo 1 (Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_range_l1266_126632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_28_meters_l1266_126624

/-- The distance by which runner A beats runner B -/
noncomputable def distance_difference (distance : ℝ) (time_A : ℝ) (time_B : ℝ) : ℝ :=
  distance - (distance / time_B) * time_A

/-- Theorem: A beats B by 28 meters -/
theorem a_beats_b_by_28_meters (distance : ℝ) (time_A : ℝ) (time_B : ℝ)
  (h1 : distance = 224)
  (h2 : time_A = 28)
  (h3 : time_B = 32) :
  distance_difference distance time_A time_B = 28 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_difference 224 28 32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_28_meters_l1266_126624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_function_l1266_126626

noncomputable section

-- Define the original function
def f (x : ℝ) : ℝ := Real.sin x

-- Define the transformed function
def g (x : ℝ) : ℝ := Real.sin (x / 2 + Real.pi / 3)

-- Theorem statement
theorem transform_sin_function :
  ∀ x : ℝ, g x = f (2 * (x + Real.pi / 3)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_function_l1266_126626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1266_126635

/-- The function f(x) = x + 4/(x+2) for x > -2 -/
noncomputable def f (x : ℝ) : ℝ := x + 4 / (x + 2)

theorem f_minimum_value :
  (∀ x > -2, f x ≥ 2) ∧ (∃ x > -2, f x = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1266_126635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l1266_126645

/-- A parabola with equation y = x^2 + k -/
structure Parabola where
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A chord of a parabola -/
structure Chord (p : Parabola) where
  A : Point
  B : Point

/-- The distance between two points -/
noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- The t value for a chord and a point -/
noncomputable def t_value (p : Parabola) (chord : Chord p) (C : Point) : ℝ :=
  1 / (distance chord.A C)^2 + 1 / (distance chord.B C)^2

/-- The theorem statement -/
theorem parabola_chord_theorem (p : Parabola) (C : Point) 
  (h1 : C.x = 0) 
  (h2 : ∀ (chord : Chord p), chord.A.y = chord.A.x^2 + p.k)
  (h3 : ∀ (chord : Chord p), chord.B.y = chord.B.x^2 + p.k)
  (h4 : ∃ (t : ℝ), ∀ (chord : Chord p), t_value p chord C = t) :
  ∃ (t : ℝ), t = 4 ∧ ∀ (chord : Chord p), t_value p chord C = t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l1266_126645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_roots_for_quadratic_l1266_126636

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax^2 + bx + c = 0 are given by (-b ± √(b^2 - 4ac)) / (2a) -/
def quadraticRoots (a b c : ℤ) : Set ℝ :=
  {x : ℝ | (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0}

theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ p q : ℕ, 
    isPrime p ∧ isPrime q ∧ 
    quadraticRoots 1 (-41) k = {↑p, ↑q} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_roots_for_quadratic_l1266_126636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_specific_form_and_range_l1266_126640

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.cos (ω * x + φ)

theorem function_properties_imply_specific_form_and_range 
  (A ω φ : ℝ) (h_A : A > 0) (h_ω : ω > 0) (h_φ : |φ| ≤ π/2)
  (h_max : ∀ x, f A ω φ x ≤ 2)
  (h_period : ∀ x, f A ω φ (x + π/ω) = f A ω φ x)
  (h_symmetry : ∀ x, f A ω φ (π/6 - x) = f A ω φ (π/6 + x)) :
  (∀ x, f A ω φ x = 2 * Real.cos (2*x + π/3)) ∧
  (∀ x ∈ Set.Icc (-3*π/4) (π/2), 
    -1 ≤ (2 * Real.cos (2*x) * (1/2) + 1 * (-2 * Real.cos x) + 1/2) ∧
    (2 * Real.cos (2*x) * (1/2) + 1 * (-2 * Real.cos x) + 1/2) ≤ (2*Real.sqrt 2 + 1)/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_specific_form_and_range_l1266_126640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_expression_l1266_126674

theorem integer_expression (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ (m : ℤ), (n - 3*k + 2) * (Nat.factorial n) = (k + 2) * (Nat.factorial k) * (Nat.factorial (n - k)) * m ↔ k = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_expression_l1266_126674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_FV_sum_l1266_126625

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ  -- Vertex
  F : ℝ × ℝ  -- Focus

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Sum of possible FV values for a parabola with specific point B -/
theorem parabola_FV_sum (p : Parabola) :
  ∃ (B : ℝ × ℝ), 
    distance B p.F = 25 ∧ 
    distance B p.V = 28 → 
    (∃ (d1 d2 : ℝ), d1 + d2 = 50/3 ∧ 
      (distance p.F p.V = d1 ∨ distance p.F p.V = d2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_FV_sum_l1266_126625
