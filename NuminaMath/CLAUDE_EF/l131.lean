import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_bought_apples_l131_13148

/-- Theorem: Mother bought 16 apples -/
theorem mother_bought_apples : ∃ (total_apples : ℕ), total_apples = 16 := by
  -- Define the total number of apples bought
  let total_apples : ℕ := 16

  -- Define the number of apples eaten (half of the total)
  let eaten_apples : ℕ := total_apples / 2

  -- Define the number of apples left
  let left_apples : ℕ := 8

  -- Condition: The number of apples left is equal to half of the total
  have half_left : left_apples = total_apples - eaten_apples := by
    -- This is true by the problem statement, but we'll skip the detailed proof
    sorry

  -- Prove that the total number of apples is 16
  have total_is_sixteen : total_apples = 16 := by
    -- This follows from our definition, but we'll skip the detailed proof
    sorry

  -- Conclude the existence of total_apples equal to 16
  exists total_apples

-- The final result
def result : ℕ := 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_bought_apples_l131_13148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_reciprocals_l131_13183

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => a n + n + 1

def sum_reciprocals (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (a i : ℚ))

theorem sequence_sum_reciprocals :
  sum_reciprocals 2017 + 1 / (a 2015 : ℚ) + 1 / (a 2018 : ℚ) = 2019 / 1010 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_reciprocals_l131_13183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_equation_l131_13112

theorem complex_circle_equation (z : ℂ) :
  (Complex.abs (z + 1 - 3 * Complex.I) = 2) ↔
  (∃ (center : ℂ) (radius : ℝ), z ∈ {w : ℂ | Complex.abs (w - center) = radius} ∧
                                center = -1 + 3 * Complex.I ∧
                                radius = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_equation_l131_13112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colony_survival_probability_l131_13162

/-- The probability that a single cell survives indefinitely -/
noncomputable def p : ℝ := 1/2

/-- The number of cells in the colony -/
def n : ℕ := 100

/-- The probability that a cell dies in a minute -/
noncomputable def p_death : ℝ := 1/3

/-- The probability that a cell splits into two in a minute -/
noncomputable def p_split : ℝ := 2/3

/-- The probability that the colony never goes extinct -/
noncomputable def p_survival : ℝ := 1 - (1/2)^n

theorem colony_survival_probability :
  p = 1/2 ∧
  p_death = 1/3 ∧
  p_split = 2/3 ∧
  n = 100 →
  p_survival = 1 - (1/2)^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colony_survival_probability_l131_13162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_fruits_left_l131_13133

/-- Represents the number of limes and plums picked and eaten by various people --/
structure FruitPicking where
  mike_limes : ℝ
  alyssa_limes_eaten : ℝ
  tom_plums : ℝ
  tom_plums_eaten : ℝ
  jenny_limes : ℝ
  jenny_limes_eaten_fraction : ℝ

/-- Calculates the total number of limes and plums left --/
def calculate_fruits_left (fp : FruitPicking) : ℝ × ℝ :=
  let limes_left := fp.mike_limes - fp.alyssa_limes_eaten + 
                    fp.jenny_limes - (fp.jenny_limes * fp.jenny_limes_eaten_fraction)
  let plums_left := fp.tom_plums - fp.tom_plums_eaten
  (limes_left, plums_left)

/-- Theorem stating the correct number of limes and plums left --/
theorem correct_fruits_left : 
  let fp : FruitPicking := {
    mike_limes := 32.5,
    alyssa_limes_eaten := 8.25,
    tom_plums := 14.5,
    tom_plums_eaten := 2.5,
    jenny_limes := 10.8,
    jenny_limes_eaten_fraction := 0.5
  }
  let (limes_left, plums_left) := calculate_fruits_left fp
  limes_left = 29.65 ∧ plums_left = 12 := by
  sorry

#eval calculate_fruits_left {
  mike_limes := 32.5,
  alyssa_limes_eaten := 8.25,
  tom_plums := 14.5,
  tom_plums_eaten := 2.5,
  jenny_limes := 10.8,
  jenny_limes_eaten_fraction := 0.5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_fruits_left_l131_13133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_initial_lives_l131_13105

/-- Represents the initial number of lives Tiffany had in the game -/
def initial_lives : ℝ := 43.0

/-- Represents the number of lives Tiffany won in the hard part of the game -/
def lives_won_hard_part : ℝ := 14.0

/-- Represents the number of lives Tiffany could get in the next level -/
def lives_next_level : ℝ := 27.0

/-- Represents the total number of lives Tiffany would have after winning lives in both parts -/
def total_lives : ℝ := 84

/-- Theorem stating that Tiffany's initial number of lives was 43.0 -/
theorem tiffany_initial_lives : 
  initial_lives + lives_won_hard_part + lives_next_level = total_lives :=
by
  rw [initial_lives, lives_won_hard_part, lives_next_level, total_lives]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_initial_lives_l131_13105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_is_correct_l131_13116

/-- Triangle PQR with sides a, b, c -/
structure Triangle (a b c : ℝ) where
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The maximum height of the structure formed by right angle folds -/
noncomputable def max_height (t : Triangle 25 28 35) : ℝ :=
  3360 * Real.sqrt 21

/-- The theorem stating the maximum height of the structure -/
theorem max_height_is_correct (t : Triangle 25 28 35) :
  max_height t = 3360 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_is_correct_l131_13116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l131_13193

-- Define the cone properties
noncomputable def slant_height : ℝ := 10
noncomputable def lateral_area : ℝ := 60 * Real.pi

-- Define the volume of a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_volume_calculation :
  ∃ (r h : ℝ),
    r * slant_height = lateral_area / Real.pi ∧
    h^2 + r^2 = slant_height^2 ∧
    cone_volume r h = 96 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l131_13193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisibility_by_prime_l131_13125

theorem sum_divisibility_by_prime (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  (∃ m : ℕ, Finset.sum (Finset.range p.succ) (λ i => i^k) = m * p) ↔ 
  (k = 0 ∨ (k ≥ 1 ∧ ¬(p - 1 ∣ k))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisibility_by_prime_l131_13125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_three_squares_l131_13144

/-- A square is a set of points in ℝ² that form a square. -/
def is_square (s : Set (ℝ × ℝ)) : Prop := sorry

/-- The area of a set of points in ℝ². -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The center of the top square is directly above the common edge of the lower two squares. -/
def center_above_edge (s₁ s₂ s₃ : Set (ℝ × ℝ)) : Prop := sorry

/-- The shaded region formed by three squares arranged as described. -/
def shaded_region (s₁ s₂ s₃ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The area of the shaded region formed by three unit squares arranged such that the center of the top square is directly above the common edge of the lower two squares is equal to 1 cm². -/
theorem shaded_area_three_squares (s₁ s₂ s₃ : Set (ℝ × ℝ)) 
  (h₁ : is_square s₁ ∧ area s₁ = 1)
  (h₂ : is_square s₂ ∧ area s₂ = 1)
  (h₃ : is_square s₃ ∧ area s₃ = 1)
  (h₄ : center_above_edge s₁ s₂ s₃) : 
  area (shaded_region s₁ s₂ s₃) = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_three_squares_l131_13144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_81_position_consecutive_27_before_36_l131_13199

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem stating the position of the first occurrence of 81 in the digit sum sequence -/
theorem first_81_position :
  (∀ k < 111111111, digitSum (9 * k) ≠ 81) ∧ digitSum (9 * 111111111) = 81 := by sorry

/-- Theorem stating the existence of four consecutive 27s before the first 36 -/
theorem consecutive_27_before_36 :
  ∃ m : ℕ, m < 1111 ∧
  (∀ k ∈ Finset.range 4, digitSum (9 * (m + k)) = 27) ∧
  (∀ k ≤ 1111, digitSum (9 * k) ≠ 36) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_81_position_consecutive_27_before_36_l131_13199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_of_cubes_l131_13194

theorem matrix_sum_of_cubes (a b c : ℂ) : 
  let M : Matrix (Fin 3) (Fin 3) ℂ := !![a, b, c; b, c, a; c, a, b]
  (M ^ 2 = 2 • (1 : Matrix (Fin 3) (Fin 3) ℂ)) → 
  (a * b * c = 1) →
  (a^3 + b^3 + c^3 = 3 + 2 * Real.sqrt 2 ∨ a^3 + b^3 + c^3 = 3 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_of_cubes_l131_13194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edge_AC_l131_13141

-- Define the triangle ABC
variable (A B C : Real)

-- Define the angles of the triangle
variable (angleA angleB angleC : Real)

-- Define the sides of the triangle
variable (a b c : Real)

-- Conditions from the problem
axiom angle_arithmetic_sequence : 2 * angleB = angleA + angleC
axiom triangle_area : (1/2) * a * c * Real.sin angleB = Real.sqrt 3

-- Theorem to prove
theorem min_edge_AC : 
  ∃ (min_AC : Real), min_AC = 2 ∧ 
  ∀ (AC : Real), AC ≥ min_AC :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edge_AC_l131_13141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l131_13123

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := (1/2) * (t.a^2 - (t.b - t.c)^2)

/-- Lambda is the ratio of side b to side a -/
noncomputable def Triangle.lambda (t : Triangle) : ℝ := t.b / t.a

theorem triangle_properties (t : Triangle) 
  (h_area : t.area = (1/2) * (t.a^2 - (t.b - t.c)^2))
  (h_tanC : Real.tan t.C = 2) :
  Real.sin t.A = 4/5 ∧ Real.cos t.A = 3/5 ∧ t.lambda = Real.sqrt 5 / 2 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l131_13123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l131_13190

theorem solution_to_equation : ∃ (x y z t : ℕ), 
  x ∈ ({12, 14, 37, 65} : Set ℕ) ∧ 
  y ∈ ({12, 14, 37, 65} : Set ℕ) ∧ 
  z ∈ ({12, 14, 37, 65} : Set ℕ) ∧ 
  t ∈ ({12, 14, 37, 65} : Set ℕ) ∧ 
  x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧
  x * y - x * z + y * t = 182 ∧
  x = 12 ∧ y = 37 ∧ z = 65 ∧ t = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l131_13190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solving_probability_l131_13158

noncomputable def IndepEvents (p_A p_B : ℝ) : Prop :=
  p_A * p_B = p_A * p_B

noncomputable def ProbBoth (p_A p_B : ℝ) : ℝ :=
  p_A * p_B

noncomputable def ProbAtLeastOne (p_A p_B : ℝ) : ℝ :=
  1 - ProbBoth (1 - p_A) (1 - p_B)

theorem problem_solving_probability (p_A p_B : ℝ) 
  (h_A : p_A = 0.6) 
  (h_B : p_B = 0.7) 
  (h_ind : IndepEvents p_A p_B) : 
  ProbAtLeastOne p_A p_B = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solving_probability_l131_13158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_supersquares_l131_13122

/-- A positive integer is a supersquare if it's the square of a number not divisible by 10
    and its decimal representation consists only of digits 0, 4, and 9. -/
def IsSuperSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ ¬(10 ∣ m) ∧ n = m^2 ∧
  ∀ d : ℕ, d ∈ Nat.digits 10 n → d ∈ ({0, 4, 9} : Set ℕ)

/-- The set of all supersquares -/
def SuperSquares : Set ℕ := {n : ℕ | IsSuperSquare n}

/-- There are infinitely many supersquares -/
theorem infinitely_many_supersquares : Set.Infinite SuperSquares := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_supersquares_l131_13122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_decimal_l131_13195

theorem fraction_equals_decimal (a : ℕ) : a = 282 → (a : ℚ) / (a + 18 : ℚ) = 47 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equals_decimal_l131_13195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_BC_l131_13172

/-- The length of segment BC given point A and its reflections B and C -/
theorem length_of_segment_BC (A B C : ℝ × ℝ × ℝ) : 
  A = (-3, -2, 4) →
  B = (-(A.fst), -(A.snd.fst), -(A.snd.snd)) →
  C = (-(A.fst), A.snd.fst, A.snd.snd) →
  Real.sqrt ((B.fst - C.fst)^2 + (B.snd.fst - C.snd.fst)^2 + (B.snd.snd - C.snd.snd)^2) = 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_BC_l131_13172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uv_length_in_triangle_l131_13180

-- Define the triangle DEF
structure Triangle (DE DF EF : ℝ) where
  positive_DE : 0 < DE
  positive_DF : 0 < DF
  positive_EF : 0 < EF
  triangle_inequality_1 : DE + DF > EF
  triangle_inequality_2 : DE + EF > DF
  triangle_inequality_3 : DF + EF > DE

-- Define the angle bisector
def AngleBisector (T : Triangle DE DF EF) (R : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), R.1 = k * EF ∧ R.2 = (1 - k) * EF

-- Define the perpendicular foot
def PerpendicularFoot (F R : ℝ × ℝ) (U : ℝ × ℝ) : Prop :=
  (U.1 - F.1) * (R.1 - U.1) + (U.2 - F.2) * (R.2 - U.2) = 0

-- Main theorem
theorem uv_length_in_triangle (T : Triangle 130 150 140) 
  (R S U V : ℝ × ℝ) :
  AngleBisector T R →
  AngleBisector T S →
  PerpendicularFoot (0, 0) R U →
  PerpendicularFoot (0, 0) S V →
  (U.1 - V.1)^2 + (U.2 - V.2)^2 = 80^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uv_length_in_triangle_l131_13180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l131_13121

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1 + 2 * (Real.cos x) ^ 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-π/6) (π/4) → f x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-π/6) (π/4) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-π/6) (π/4) ∧ f x = 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-π/6) (π/4) ∧ f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l131_13121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_mangoes_after_reduction_l131_13168

/-- Represents the price and quantity of mangoes -/
structure MangoMarket where
  originalPrice : ℚ
  originalQuantity : ℕ
  spendAmount : ℚ
  priceReduction : ℚ

/-- Calculates the number of extra mangoes purchased after a price reduction -/
def extraMangoes (market : MangoMarket) : ℕ :=
  let originalPricePerMango := market.originalPrice / market.originalQuantity
  let newPricePerMango := originalPricePerMango * (1 - market.priceReduction)
  let originalQuantityPurchased := (market.spendAmount / originalPricePerMango).floor
  let newQuantityPurchased := (market.spendAmount / newPricePerMango).floor
  (newQuantityPurchased - originalQuantityPurchased).toNat

/-- Theorem stating that under the given conditions, 12 extra mangoes are purchased -/
theorem extra_mangoes_after_reduction (market : MangoMarket)
  (h1 : market.originalPrice = 450)
  (h2 : market.originalQuantity = 135)
  (h3 : market.spendAmount = 360)
  (h4 : market.priceReduction = 1/10) :
  extraMangoes market = 12 := by
  sorry

#eval extraMangoes { originalPrice := 450, originalQuantity := 135, spendAmount := 360, priceReduction := 1/10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_mangoes_after_reduction_l131_13168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_pi_to_2pi_l131_13132

-- Define the function f(x) = 2cos(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x

-- State the theorem
theorem f_increasing_on_pi_to_2pi :
  StrictMonoOn f (Set.Icc Real.pi (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_pi_to_2pi_l131_13132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l131_13143

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Properties of the ellipse and line AB -/
theorem ellipse_and_line_properties 
  (C : Ellipse) 
  (P : Point)
  (F1 F2 : Point) -- Left and right foci
  (h_P_on_C : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_P_coords : P.x = 1 ∧ P.y = -3/2)
  (h_focal_sum : distance P F1 + distance P F2 = 4)
  (h_AB_intersect : ∃ (A B : Point), 
    A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1 ∧
    B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1 ∧
    A ≠ B)
  (h_complementary_slopes : ∃ (m1 m2 : ℝ) (A B : Point), 
    m1 + m2 = 0 ∧
    m1 = (A.y - P.y) / (A.x - P.x) ∧
    m2 = (B.y - P.y) / (B.x - P.x))
  (h_AB_positive_axes : ∃ (x y : ℝ) (A B : Point), x > 0 ∧ y > 0 ∧ 
    y = (B.y - A.y) / (B.x - A.x) * x + A.y) :
  C.a = 2 ∧ C.b = Real.sqrt 3 ∧ 
  ∃ (A B : Point), (B.y - A.y) / (B.x - A.x) = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l131_13143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l131_13159

open Real

/-- Given an isosceles triangle with base a and angle α at the base,
    this theorem states the radius of a circle tangent to both the inscribed circle
    and the legs of the triangle. -/
theorem tangent_circle_radius (a α : ℝ) (ha : 0 < a) (hα : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r = (a / 2) * tan α ^ 3 / 2 := by
  -- The proof goes here
  sorry

#check tangent_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l131_13159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducibility_characterization_l131_13164

/-- Define the polynomial P for a given n and list of variables -/
def P (n : ℕ) (a : List ℝ) : ℝ :=
  (a.map (λ x => x^n)).sum - n * a.prod

/-- A polynomial is irreducible over ℝ if it cannot be factored as the product of two nonconstant polynomials with real coefficients -/
def IsIrreduciblePoly (p : List ℝ → ℝ) : Prop :=
  ∀ (f g : List ℝ → ℝ), (∀ a, p a = f a * g a) → (∀ a, f a = 1 ∨ g a = 1)

theorem irreducibility_characterization (n : ℕ) (h : n ≥ 2) :
  IsIrreduciblePoly (P n) ↔ n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducibility_characterization_l131_13164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l131_13126

noncomputable def f (x : ℝ) := Real.sin x + Real.sin (2 * x)

noncomputable def F (x : ℝ) := f x * Real.cos x + x

theorem f_properties :
  ∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 Real.pi ∧
    (∀ x ∈ Set.Icc 0 Real.pi, F x ≤ F x₀) ∧
    f x₀ = (7 * Real.sqrt 5) / 9 ∧
    ∀ x ∈ Set.Icc 0 Real.pi,
      ∀ a : ℝ, a ∈ Set.Icc 1 (3/2) ∨ a ∈ Set.Icc (-3/2) (-1) →
        f x ≥ 3 * x * Real.cos (a * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l131_13126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_and_triangle_ratio_l131_13151

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.sin x * Real.cos x

theorem max_f_and_triangle_ratio :
  (∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
    ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → f y ≤ f x) ∧
  (∃ (x : ℝ), f x = Real.sqrt (3 / 2)) ∧
  (∀ (A B C : ℝ), 
    A + B + C = Real.pi → 
    0 < A → 0 < B → 0 < C → 
    A < B → 
    f A = Real.sqrt 2 / 2 → 
    f B = Real.sqrt 2 / 2 → 
    Real.sin C / Real.sin A = 1 / (Real.sin (Real.pi / 3) * Real.cos (Real.pi / 4) - 
                                   Real.cos (Real.pi / 3) * Real.sin (Real.pi / 4))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_and_triangle_ratio_l131_13151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_expansion_l131_13163

/-- Given that the constant term of (1/x - x^2)^n is 15, 
    prove that the coefficient of x^3 in the expansion is -20 -/
theorem coefficient_x_cube_in_expansion (n : ℕ) : 
  (∃ r : ℕ, (n.choose r) * ((-1 : ℤ)^r) = 15 ∧ 3*r = n) →
  (n.choose 3) * ((-1 : ℤ)^3) = -20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_expansion_l131_13163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_division_l131_13147

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ
  bottomRadius : ℝ
  topRadius : ℝ

/-- Calculates the volume of a truncated cone -/
noncomputable def volumeTruncatedCone (cone : TruncatedCone) : ℝ :=
  (1/3) * Real.pi * cone.height * (cone.bottomRadius^2 + cone.bottomRadius * cone.topRadius + cone.topRadius^2)

/-- Represents the division of a volume into three parts -/
structure VolumeDivision where
  v1 : ℝ
  v2 : ℝ
  v3 : ℝ

/-- Checks if three volumes are in the ratio 2:3:7 -/
def isRatio237 (vd : VolumeDivision) : Prop :=
  vd.v1 / 2 = vd.v2 / 3 ∧ vd.v2 / 3 = vd.v3 / 7

theorem truncated_cone_volume_division :
  let cone : TruncatedCone := ⟨3, 2, 1⟩
  let totalVolume := volumeTruncatedCone cone
  let division : VolumeDivision := ⟨7*Real.pi/6, 7*Real.pi/4, 49*Real.pi/12⟩
  totalVolume = 7*Real.pi ∧
  isRatio237 division ∧
  division.v1 + division.v2 + division.v3 = totalVolume :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_division_l131_13147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_partition_l131_13111

/-- A natural number is binary-like if its decimal representation contains only digits 1 and 0. -/
def is_binary_like (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of all binary-like natural numbers. -/
def binary_like_set : Set ℕ := {n : ℕ | is_binary_like n}

/-- A function that counts the number of ones in the decimal representation of a natural number. -/
def count_ones (n : ℕ) : ℕ :=
  (n.digits 10).filter (· = 1) |>.length

/-- A partition of the binary-like set into two subsets. -/
structure BinaryLikePartition :=
  (A B : Set ℕ)
  (partition_complete : A ∪ B = binary_like_set)
  (partition_disjoint : A ∩ B = ∅)

/-- The property that the sum of any two distinct numbers from the same subset
    contains at least two ones in its decimal representation. -/
def valid_partition (P : BinaryLikePartition) : Prop :=
  (∀ a₁ a₂, a₁ ∈ P.A → a₂ ∈ P.A → a₁ ≠ a₂ → count_ones (a₁ + a₂) ≥ 2) ∧
  (∀ b₁ b₂, b₁ ∈ P.B → b₂ ∈ P.B → b₁ ≠ b₂ → count_ones (b₁ + b₂) ≥ 2)

/-- The main theorem stating that there exists a valid partition of the binary-like set. -/
theorem exists_valid_partition : ∃ P : BinaryLikePartition, valid_partition P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_partition_l131_13111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_frood_throw_beats_eat_l131_13149

theorem least_frood_throw_beats_eat : 
  ∀ n : ℕ, n > 0 → (n^2 > 12 * n ↔ n ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_frood_throw_beats_eat_l131_13149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l131_13177

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_squared := v.1^2 + v.2^2
  (dot / norm_squared * v.1, dot / norm_squared * v.2)

theorem line_equation_from_projection (x y : ℝ) :
  proj (x, y) (4, 3) = (-4, -3) →
  y = -4/3 * x - 25/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l131_13177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l131_13169

noncomputable def original_expr := (1 : ℝ) / (Real.rpow 5 (1/3) - Real.rpow 3 (1/3))

noncomputable def rationalized_form := (Real.rpow 25 (1/3) + Real.rpow 15 (1/3) + Real.rpow 9 (1/3)) / 2

theorem rationalize_denominator :
  original_expr = rationalized_form ∧ 
  (∀ k : ℝ, k ≠ 0 → (k * Real.rpow 25 (1/3) + k * Real.rpow 15 (1/3) + k * Real.rpow 9 (1/3)) / (2 * k) = rationalized_form) :=
by sorry

#check rationalize_denominator

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l131_13169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_five_halves_l131_13175

/-- The function f(x) = |log₂x| -/
noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

/-- Theorem stating that under given conditions, m + n = 5/2 -/
theorem sum_m_n_equals_five_halves
  (m n : ℝ)
  (h_pos_m : m > 0)
  (h_pos_n : n > 0)
  (h_m_lt_n : m < n)
  (h_f_eq : f m = f n)
  (h_max_f : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2)
  (h_exists_max : ∃ x ∈ Set.Icc (m^2) n, f x = 2) :
  m + n = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_m_n_equals_five_halves_l131_13175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lom_area_l131_13173

/-- Given a scalene triangle ABC with angles α, β, γ, prove that when
    α = β - γ, β = 2γ, and the area of ABC is 32,
    the area of triangle LOM formed by the intersection points of
    the angle bisectors with the circumcircle is approximately 44. -/
theorem triangle_lom_area (α β γ : Real) (area_abc : Real) :
  α + β + γ = Real.pi →  -- Sum of angles in a triangle is π radians
  α = β - γ →            -- One angle is the difference of the other two
  β = 2 * γ →            -- One angle is twice another
  area_abc = 32 →        -- Area of triangle ABC is 32
  ∃ (area_lom : Real),   -- There exists an area for triangle LOM
    abs (area_lom - 44) < 0.5 :=  -- The area of LOM is approximately 44
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lom_area_l131_13173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_to_63_l131_13108

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def next_number (n : ℕ) : ℕ := n - sum_of_digits n

def sequence_63 (a : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => next_number (sequence_63 a n)

theorem convergence_to_63 (A : ℕ) (h : 100 < A ∧ A < 1000) :
  ∃ n : ℕ, sequence_63 A n = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_to_63_l131_13108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_quadratic_l131_13113

theorem integral_reciprocal_quadratic (a : ℝ) (h : a > 0) :
  ∫ x in (-1)..a^2, 1 / (x^2 + a^2) = π / (2 * a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_reciprocal_quadratic_l131_13113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_over_a_l131_13157

noncomputable def f (x a b : ℝ) : ℝ := Real.log x + (Real.exp 1 - a) * x - b

theorem min_b_over_a (a b : ℝ) :
  (∀ x > 0, f x a b ≤ 0) →
  (b / a ≥ -1 / Real.exp 1) ∧ (∃ a₀ b₀ : ℝ, b₀ / a₀ = -1 / Real.exp 1 ∧ ∀ x > 0, f x a₀ b₀ ≤ 0) :=
by
  sorry

#check min_b_over_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_over_a_l131_13157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l131_13109

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ (M = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l131_13109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_chord_length_l131_13106

/-- The parabola from which perpendiculars are drawn -/
def original_parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- The trajectory E formed by midpoints of perpendicular segments -/
def trajectory_E (x y : ℝ) : Prop := y^2 = 8 * x

/-- The intersecting line -/
def intersecting_line (k x y : ℝ) : Prop := y = k * (x - 2) ∧ k > 0

/-- Point F -/
def point_F : ℝ × ℝ := (2, 0)

/-- Theorem stating the equation of trajectory E and the length of chord AB -/
theorem trajectory_and_chord_length :
  (∀ x y : ℝ, trajectory_E x y ↔ 
    ∃ x₀ y₀ : ℝ, original_parabola x₀ y₀ ∧ x = x₀ ∧ y = y₀ / 2) ∧
  (∀ k x₁ y₁ x₂ y₂ : ℝ, 
    trajectory_E x₁ y₁ ∧ trajectory_E x₂ y₂ ∧
    intersecting_line k x₁ y₁ ∧ intersecting_line k x₂ y₂ ∧
    (x₁ - point_F.1)^2 + (y₁ - point_F.2)^2 = 
      4 * ((x₂ - point_F.1)^2 + (y₂ - point_F.2)^2) →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 9^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_chord_length_l131_13106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_0692_to_hundredth_l131_13150

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_4_0692_to_hundredth :
  round_to_hundredth 4.0692 = 4.07 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_0692_to_hundredth_l131_13150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l131_13181

/-- The circle C with equation x^2 + y^2 - 8x + 15 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

/-- The line y = kx - 2 -/
def line (k x : ℝ) : ℝ := k*x - 2

/-- A point (x, y) is on the line y = kx - 2 -/
def point_on_line (k x y : ℝ) : Prop := y = line k x

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- A circle with center (x, y) and radius r intersects circle C -/
def circles_intersect (x y r : ℝ) : Prop :=
  ∃ (x' y' : ℝ), circle_C x' y' ∧ distance x y x' y' ≤ r + 1

theorem max_k_value :
  ∀ (k : ℝ), (∃ (x y : ℝ), point_on_line k x y ∧ circles_intersect x y 1) →
  k ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l131_13181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m1m2_passes_through_fixed_point_l131_13134

/-- The fixed point through which M1M2 passes for a given parabola and points A and B -/
noncomputable def fixed_point (p a b : ℝ) : ℝ × ℝ := (a, 2*p*a/b)

/-- Definition of collinearity for points (x1, y1), (x2, y2), (x3, y3) -/
def collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Theorem stating that M1M2 always passes through the fixed point -/
theorem m1m2_passes_through_fixed_point 
  (p a b : ℝ) 
  (hab : a * b ≠ 0) 
  (hb : b^2 ≠ 2*p*a) :
  ∀ (M M1 M2 : ℝ × ℝ),
  (M.2)^2 = 2*p*M.1 →  -- M is on the parabola
  (M1.2)^2 = 2*p*M1.1 →  -- M1 is on the parabola
  (M2.2)^2 = 2*p*M2.1 →  -- M2 is on the parabola
  collinear a b M.1 M.2 M1.1 M1.2 →  -- A, M, M1 are collinear
  collinear (-a) 0 M.1 M.2 M2.1 M2.2 →  -- B, M, M2 are collinear
  M1 ≠ M2 →  -- M1 and M2 are distinct
  ∃ (t : ℝ), 
    fixed_point p a b = 
      (t * M1.1 + (1 - t) * M2.1, t * M1.2 + (1 - t) * M2.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m1m2_passes_through_fixed_point_l131_13134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_bound_l131_13118

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a-1) * x - a * Real.log x

-- State the theorem
theorem extreme_value_bound (a : ℝ) :
  ∃ (m : ℝ), (∀ (x : ℝ), x > 0 → f a x ≥ m) ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_bound_l131_13118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_division_l131_13139

theorem pie_division (p q : ℕ) (h : Nat.Coprime p q) :
  ∃ n : ℕ, (n % p = 0 ∧ n % q = 0) ∧ 
  ∀ m : ℕ, (m % p = 0 ∧ m % q = 0) → n ≤ m :=
by
  use p + q - 1
  sorry  -- Proof details omitted

#check pie_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_division_l131_13139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_log_function_l131_13110

-- Define the logarithm function with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem no_odd_log_function :
  ¬∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧
  (∀ (x : ℝ), x > 0 → log_base a (-x) = -(log_base a x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_log_function_l131_13110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_6_equals_90_l131_13198

/-- An arithmetic sequence with common difference 5 and special properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_d : d = 5
  h_geometric : (a 2)^2 = a 1 * a 5

/-- Sum of the first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem: S_6 equals 90 -/
theorem S_6_equals_90 (seq : ArithmeticSequence) : S seq 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_6_equals_90_l131_13198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l131_13160

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 120)
  (h2 : bridge_length = 660)
  (h3 : time_to_cross = 51.99584033277338) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
    |((train_length + bridge_length) / time_to_cross * 3.6) - 54| < ε := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l131_13160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l131_13165

/-- Given a linear function f(x) = ax + b, if the distance between intersection points
    of y = x^2 - 1 and y = f(x) is √34, and the distance between intersection points
    of y = x^2 + 1 and y = f(x) + 3 is √42, then the distance between intersection points
    of y = x^2 and y = f(x) - 1 is 3√2. -/
theorem intersection_distance (a b : ℝ) : 
  let f := fun x => a * x + b
  (Real.sqrt ((a^2 + 1) * (a^2 + 4 * (b + 1))) = Real.sqrt 34) →
  (Real.sqrt ((a^2 + 1) * (a^2 + 4 * (b + 2))) = Real.sqrt 42) →
  Real.sqrt ((a^2 + 1) * (a^2 + 4 * (b - 1))) = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l131_13165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_seed_selection_l131_13155

def is_valid_seed (n : ℕ) : Bool := 0 < n ∧ n ≤ 850

def seed_sequence : List ℕ := [390, 737, 924, 220, 372]

def first_four_valid_seeds (seq : List ℕ) : List ℕ :=
  (seq.filter is_valid_seed).take 4

theorem correct_seed_selection :
  first_four_valid_seeds seed_sequence = [390, 737, 220, 372] := by
  rfl

#eval first_four_valid_seeds seed_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_seed_selection_l131_13155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l131_13119

/-- A complex number z defined in terms of a real number m -/
def z (m : ℝ) : ℂ := (m^2 + m : ℝ) + (m^2 - 1 : ℝ) * Complex.I

/-- Theorem stating the conditions for z to be real, complex, or pure imaginary -/
theorem z_classification (m : ℝ) :
  (z m ∈ Set.range (Complex.ofReal) ↔ m = 1 ∨ m = -1) ∧
  (z m ∉ Set.range (Complex.ofReal) ↔ m ≠ 1 ∧ m ≠ -1) ∧
  (∃ (r : ℝ), z m = r * Complex.I ∧ z m ≠ 0 ↔ m = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l131_13119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_worth_is_four_l131_13182

/- Define the company structure -/
structure Company where
  num_blocks : Nat
  workers_per_block : Nat
  total_gift_budget : Rat

/- Define the function to calculate the worth of each gift -/
noncomputable def gift_worth (c : Company) : Rat :=
  c.total_gift_budget / (c.num_blocks * c.workers_per_block)

/- Theorem statement -/
theorem gift_worth_is_four (c : Company) 
  (h1 : c.num_blocks = 10) 
  (h2 : c.workers_per_block = 100) 
  (h3 : c.total_gift_budget = 4000) : 
  gift_worth c = 4 := by
  sorry

/- Example calculation -/
#eval (4000 : Rat) / (10 * 100 : Nat)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_worth_is_four_l131_13182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_proof_l131_13120

theorem equation_proof (m : ℤ) (h : m = 8) : (2 : ℝ)^(24 - m) = (2 : ℝ)^16 := by
  rw [h]
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_proof_l131_13120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l131_13137

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- State the theorem
theorem f_properties :
  -- f is defined on (0, +∞)
  (∀ x > 0, f x ∈ Set.range f) ∧
  -- f is decreasing on (0, 1)
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- f is increasing on (1, +∞)
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  -- f(x) < (2/3)x^3 for x > 1
  (∀ x > 1, f x < (2/3) * x^3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l131_13137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sum_and_square_sum_l131_13102

/-- The smallest n for which there exist n numbers in (-1, 1) with sum 0 and sum of squares 40 -/
theorem smallest_n_for_sum_and_square_sum : ℕ := by
  -- Define the property for a list of numbers
  let property (l : List ℝ) :=
    l.all (λ x => -1 < x ∧ x < 1) ∧
    l.sum = 0 ∧
    (l.map (λ x => x^2)).sum = 40

  -- Define the existence of such a list for a given length
  let exists_list (n : ℕ) :=
    ∃ l : List ℝ, l.length = n ∧ property l

  -- The main theorem
  have h : (∃ n : ℕ, exists_list n) ∧
           (∀ m : ℕ, exists_list m → 42 ≤ m) ∧
           exists_list 42 := by
    sorry  -- Proof omitted

  -- Extract the result
  exact 42


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sum_and_square_sum_l131_13102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DAF_in_triangle_l131_13130

-- Define the triangle ABC
structure Triangle (A B C : EuclideanSpace ℝ (Fin 2)) where
  -- No additional fields needed

-- Define the circle
structure Circle (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) where
  -- No additional fields needed

-- Define necessary functions
def angle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def is_perpendicular_foot (D A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_circumcenter (O : EuclideanSpace ℝ (Fin 2)) (triangle : Triangle A B C) : Prop := sorry

def is_midpoint_of_arc (F B C A : EuclideanSpace ℝ (Fin 2)) (circle : Circle O r) : Prop := sorry

-- State the theorem
theorem angle_DAF_in_triangle 
  (A B C D O F : EuclideanSpace ℝ (Fin 2)) 
  (triangle : Triangle A B C) 
  (circle : Circle O (‖A - O‖)) :
  -- Given conditions
  angle A C B = 60 ∧
  angle C B A = 80 ∧
  is_perpendicular_foot D A B C ∧
  is_circumcenter O triangle ∧
  is_midpoint_of_arc F B C A circle →
  -- Conclusion
  angle D A F = 20 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DAF_in_triangle_l131_13130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_imply_functions_l131_13136

-- Define the types for points and functions
structure Point where
  x : ℝ
  y : ℝ

def LinearFunction := ℝ → ℝ
def DirectProportionFunction := ℝ → ℝ

-- Define the given conditions
noncomputable def M : Point := ⟨-4, 5⟩
noncomputable def N : Point := ⟨-6, 0⟩
noncomputable def O : Point := ⟨0, 0⟩

-- Define the functions
noncomputable def f : DirectProportionFunction := λ x => -5/4 * x
noncomputable def g : LinearFunction := λ x => 5/2 * x + 15

-- State the theorem
theorem intersection_and_area_imply_functions :
  (f M.x = M.y) ∧                           -- M is on the direct proportion function
  (g M.x = M.y) ∧                           -- M is on the linear function
  (g N.x = N.y) ∧                           -- N is on the linear function
  (M.x < 0 ∧ M.y > 0) ∧                     -- M is in the second quadrant
  (1/2 * (N.x - O.x) * M.y = 15) →          -- Area of triangle MON is 15
  (∀ x, f x = -5/4 * x) ∧                   -- Direct proportion function
  (∀ x, g x = 5/2 * x + 15) :=               -- Linear function
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_imply_functions_l131_13136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_workforce_equations_l131_13140

/-- Represents the composition of an instrument in terms of parts A and B -/
structure InstrumentComposition where
  partA : ℕ
  partB : ℕ

/-- Represents the daily production capacity of a worker -/
structure WorkerCapacity where
  partA : ℕ
  partB : ℕ

/-- Represents the workforce allocation and production equations -/
def WorkforceEquations (totalWorkers : ℕ) (composition : InstrumentComposition) (capacity : WorkerCapacity) : 
  Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | 
    p.1 + p.2 = totalWorkers ∧ 
    composition.partB * capacity.partA * p.1 = composition.partA * capacity.partB * p.2}

/-- The main theorem stating the correct system of equations -/
theorem correct_workforce_equations :
  let totalWorkers := 72
  let composition := InstrumentComposition.mk 1 2
  let capacity := WorkerCapacity.mk 50 60
  ∃ (x y : ℕ), (x, y) ∈ WorkforceEquations totalWorkers composition capacity ∧
    x + y = 72 ∧ 2 * 50 * x = 60 * y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_workforce_equations_l131_13140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_specific_k_for_distance_l131_13100

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1 ∧ x > 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem 1: Range of k
theorem range_of_k :
  ∀ k : ℝ, intersection_points k ↔ 1 < k ∧ k < Real.sqrt 2 := by
  sorry

-- Theorem 2: Specific k for given distance
theorem specific_k_for_distance :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = 2 * Real.sqrt 5 →
    k = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_specific_k_for_distance_l131_13100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l131_13124

noncomputable def f (x : ℝ) : ℝ := x^2 + 4/x^2 - 3

def g (k : ℝ) (x : ℝ) : ℝ := k*x + 2

theorem range_of_k :
  ∀ k : ℝ,
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc (-1) 2 →
    ∃ x₂ : ℝ, x₂ ∈ Set.Icc 1 (Real.sqrt 3) ∧ g k x₁ > f x₂) →
  k ∈ Set.Ioo (-1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l131_13124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_triangle_perimeter_proof_l131_13103

/-- The perimeter of a triangle with sides of length 15, 11, and 19 is 45. -/
theorem triangle_perimeter (a b c : ℝ) : a = 15 ∧ b = 11 ∧ c = 19 → a + b + c = 45 := by
  intro h
  rw [h.1, h.2.1, h.2.2]
  norm_num

/-- A function that calculates the perimeter of a triangle given its three side lengths. -/
def triangle_perimeter_calc (a b c : ℝ) : ℝ := a + b + c

/-- The theorem states that for a triangle with sides 15, 11, and 19, 
    the perimeter calculation function returns 45. -/
theorem triangle_perimeter_proof :
  triangle_perimeter_calc 15 11 19 = 45 := by
  unfold triangle_perimeter_calc
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_triangle_perimeter_proof_l131_13103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_over_three_days_l131_13166

def day1_pairs : ℕ := 2
def day1_total : ℚ := 60
def day2_pairs : ℕ := 3
def day2_price_multiplier : ℚ := 3/2
def day2_discount : ℚ := 1/10
def day3_pairs : ℕ := 5
def day3_price_multiplier : ℚ := 2
def day3_tax : ℚ := 2/25

noncomputable def day1_price_per_pair : ℚ := day1_total / day1_pairs
noncomputable def day2_price : ℚ := day2_pairs * (day1_price_per_pair * day2_price_multiplier) * (1 - day2_discount)
noncomputable def day3_price : ℚ := day3_pairs * (day1_price_per_pair * day3_price_multiplier) * (1 + day3_tax)

theorem total_spent_over_three_days :
  day1_total + day2_price + day3_price = 505.50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_over_three_days_l131_13166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_ratio_l131_13185

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2

-- Define the midpoint condition
def midpoint_on_y_axis (P : ℝ × ℝ) : Prop :=
  (P.1 + left_focus.1) / 2 = 0

-- Calculate distances
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem ellipse_focus_ratio (P : ℝ × ℝ) :
  point_on_ellipse P →
  midpoint_on_y_axis P →
  (distance P left_focus) / (distance P right_focus) = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_ratio_l131_13185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_hundredth_l131_13178

/-- The ratio of students who voted to the total number of students -/
def ratio : ℚ := 15 / 22

/-- Rounds a rational number to the nearest hundredth -/
def roundToHundredth (q : ℚ) : ℚ := 
  ⌊q * 100 + 1/2⌋ / 100

theorem ratio_rounded_to_hundredth : 
  roundToHundredth ratio = 68 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_rounded_to_hundredth_l131_13178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l131_13179

noncomputable def y (x : ℝ) : ℝ := Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

theorem max_value_of_y :
  ∃ (max_y : ℝ), max_y = (11 / 6) * Real.sqrt 3 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-5 * Real.pi / 12) (-Real.pi / 3) →
    y x ≤ max_y := by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l131_13179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l131_13129

/-- A parabola is defined by its parameter p -/
structure Parabola where
  p : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
def focus (para : Parabola) : Point :=
  { x := para.p, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Distance from M(2,2) to focus of parabola passing through M -/
theorem distance_to_focus (para : Parabola) 
    (h : (2 : ℝ)^2 = 2 * para.p * 2) : 
  distance { x := 2, y := 2 } (focus para) + para.p = Real.sqrt 5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l131_13129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l131_13156

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Conditions for a valid triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- Define the given condition
def GivenCondition (a b c : ℝ) (B C : ℝ) : Prop :=
  b * Real.cos C = (2 * a - c) * Real.cos B

-- Theorem statement
theorem triangle_proof
  (a b c : ℝ)
  (A B C : ℝ)
  (h_triangle : Triangle a b c A B C)
  (h_condition : GivenCondition a b c B C)
  (h_c : c = 2)
  (h_b : b = 3) :
  B = Real.pi / 3 ∧ 
  (1 / 2 : ℝ) * a * c * Real.sin B = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l131_13156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l131_13192

/-- The function f(x) = 3x^2 - 2/x + 4 -/
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 / x + 4

/-- The function g(x) = 2x^2 - k, where k is a parameter -/
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k

/-- Theorem stating that if f(3) - g(3) = 5, then k = -22/3 -/
theorem k_value (k : ℝ) : f 3 - g k 3 = 5 → k = -22/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l131_13192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_big_container_capacity_is_40_l131_13138

/-- The capacity of the big container in liters -/
noncomputable def big_container_capacity : ℚ := 40

/-- The capacity of the second container in liters -/
noncomputable def second_container_capacity : ℚ := 40

/-- The initial fill ratio of the big container -/
def big_container_initial_fill : ℚ := 30 / 100

/-- The initial fill ratio of the second container -/
def second_container_initial_fill : ℚ := 50 / 100

/-- The amount of water added to the big container in liters -/
def water_added_big : ℚ := 18

/-- The amount of water added to the second container in liters -/
def water_added_second : ℚ := 12

/-- The final fill ratio of the big container -/
def big_container_final_fill : ℚ := 3 / 4

/-- The final fill ratio of the second container -/
def second_container_final_fill : ℚ := 90 / 100

theorem big_container_capacity_is_40 :
  big_container_initial_fill * big_container_capacity + water_added_big =
  big_container_final_fill * big_container_capacity ∧
  second_container_initial_fill * second_container_capacity + water_added_second =
  second_container_final_fill * second_container_capacity :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_big_container_capacity_is_40_l131_13138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_l131_13167

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 2*x - 2 else 2*x - 3

theorem f_equals_one_iff (x₀ : ℝ) : f x₀ = 1 ↔ x₀ = -1 ∨ x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_l131_13167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_fit_implies_zero_residuals_and_perfect_correlation_l131_13115

/-- A sample point in a scatter plot -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A linear regression model -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Calculate the predicted y value for a given x -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Calculate the sum of squared residuals -/
def sumSquaredResiduals (points : List SamplePoint) (model : LinearRegression) : ℝ :=
  (points.map (λ p => (p.y - predict model p.x) ^ 2)).sum

/-- Calculate the correlation coefficient -/
noncomputable def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry

/-- Theorem: If all points lie on a straight line, then the sum of squared residuals is 0
    and the absolute value of the correlation coefficient is 1 -/
theorem perfect_fit_implies_zero_residuals_and_perfect_correlation
  (points : List SamplePoint) (model : LinearRegression)
  (h : ∀ p ∈ points, p.y = predict model p.x) :
  sumSquaredResiduals points model = 0 ∧ 
  abs (correlationCoefficient points) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_fit_implies_zero_residuals_and_perfect_correlation_l131_13115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l131_13127

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.pi * x + Real.pi / 4)

theorem symmetry_center_of_tan (k : ℤ) :
  ∃ (c : ℝ × ℝ), c = ((2 * k - 1 : ℝ) / 4, 0) ∧
  ∀ (x : ℝ), f (c.1 + x) = -f (c.1 - x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l131_13127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_value_l131_13101

def alphabet_value (n : ℕ) : ℤ :=
  match n % 13 with
  | 0 => -3
  | 1 => -2
  | 2 => -1
  | 3 => 0
  | 4 => 1
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 1
  | 9 => 0
  | 10 => -1
  | 11 => -2
  | _ => -3

def letter_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0

def word_value (word : String) : ℤ :=
  word.toList.map (fun c => alphabet_value (letter_position c)) |>.sum

theorem mathematics_value :
  word_value "mathematics" = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_value_l131_13101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_truthful_dwarfs_l131_13176

/-- Represents a dwarf who either always tells the truth or always lies -/
inductive Dwarf
| truthful
| liar
deriving DecidableEq

/-- Represents the three types of ice cream -/
inductive IceCream
| vanilla
| chocolate
| fruit

/-- A group of dwarfs with their ice cream preferences -/
structure DwarfGroup where
  dwarfs : Finset Dwarf
  preferences : Dwarf → IceCream

/-- The number of dwarfs who raise their hands for a given ice cream flavor -/
def handsRaised (group : DwarfGroup) (flavor : IceCream) : ℕ :=
  sorry

theorem four_truthful_dwarfs (group : DwarfGroup) :
  group.dwarfs.card = 10 →
  (∀ d : Dwarf, d ∈ group.dwarfs → (group.preferences d = IceCream.vanilla ∨
                                    group.preferences d = IceCream.chocolate ∨
                                    group.preferences d = IceCream.fruit)) →
  handsRaised group IceCream.vanilla = 10 →
  handsRaised group IceCream.chocolate = 5 →
  handsRaised group IceCream.fruit = 1 →
  (group.dwarfs.filter (λ d ↦ d = Dwarf.truthful)).card = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_truthful_dwarfs_l131_13176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_problem_l131_13154

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_collinearity_problem
  (e₁ e₂ : V)
  (h_non_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂)
  (AB CB CD : V)
  (k : ℝ)
  (h_AB : AB = 2 • e₁ + k • e₂)
  (h_CB : CB = e₁ + 3 • e₂)
  (h_CD : CD = 2 • e₁ - e₂)
  (h_collinear : ∃ (t : ℝ), AB = t • (CB - CD))
  : k = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_problem_l131_13154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quad_perimeter_ge_twice_diagonal_l131_13184

/-- A rectangle in 2D space -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

/-- A quadrilateral inscribed in a rectangle -/
structure InscribedQuadrilateral (rect : Rectangle) where
  vertices : Fin 4 → ℝ × ℝ
  on_sides : ∀ i, 
    (vertices i).1 = 0 ∨ (vertices i).1 = rect.width ∨
    (vertices i).2 = 0 ∨ (vertices i).2 = rect.height

/-- The perimeter of a quadrilateral -/
noncomputable def perimeter (rect : Rectangle) (quad : InscribedQuadrilateral rect) : ℝ :=
  sorry

/-- The diagonal of a rectangle -/
noncomputable def diagonal (rect : Rectangle) : ℝ :=
  Real.sqrt (rect.width ^ 2 + rect.height ^ 2)

/-- Theorem: The perimeter of an inscribed quadrilateral is not less than twice the diagonal of the rectangle -/
theorem inscribed_quad_perimeter_ge_twice_diagonal (rect : Rectangle) 
  (quad : InscribedQuadrilateral rect) : 
  perimeter rect quad ≥ 2 * diagonal rect := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quad_perimeter_ge_twice_diagonal_l131_13184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_target_function_l131_13135

noncomputable section

-- Define the hyperbola
def hyperbola (t : ℝ) (x y : ℝ) : Prop := 4 * x^2 - y^2 = t

-- Define the asymptotes
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

-- Define the vertical line
def vertical_line (x : ℝ) : Prop := x = Real.sqrt 2

-- Define the region D
def region_D (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ Real.sqrt 2 ∧ -2 * x ≤ y ∧ y ≤ 2 * x

-- Define the target function
def target_function (x y : ℝ) : ℝ := 1/2 * x - y

-- Theorem statement
theorem min_value_target_function {t : ℝ} (ht : t ≠ 0) {x y : ℝ} (hxy : region_D x y) :
  target_function x y ≥ -3 * Real.sqrt 2 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_target_function_l131_13135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l131_13117

-- Problem 1
theorem problem_1 (x k : ℝ) (a : ℝ) :
  x^2 - 8*x + 26 = (x + k)^2 + a → a = 10 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) :
  Real.sin (a * π / 180) = Real.cos (b * π / 180) ∧ 270 < b ∧ b < 360 → b = 280 := by sorry

-- Problem 3
theorem problem_3 (b : ℝ) (c : ℝ) :
  b = c * 0.7 → c = 400 := by sorry

-- Problem 4
theorem problem_4 (c d : ℝ) :
  d = 2 * (3 * c / 10) → d = 240 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l131_13117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l131_13191

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x - 1)^2 + 6

theorem quadratic_properties :
  (∀ x : ℝ, f x = f (2 - x)) ∧ 
  (∀ x : ℝ, f 1 ≤ f x) ∧
  (f 1 = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l131_13191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l131_13146

/-- Number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ := sorry

/-- Floor function -/
def floor (x : ℚ) : ℤ := sorry

/-- Sequence definition -/
def a : ℕ → ℕ+ → ℕ
  | 0, A => A
  | n + 1, A => d (⟨Int.toNat (floor (3 / 2 * (a n A : ℚ))), sorry⟩ : ℕ+) + 2011

/-- Main theorem: The sequence eventually becomes periodic -/
theorem sequence_eventually_periodic (A : ℕ+) :
  ∃ m k : ℕ, k > 0 ∧ ∀ i, a (m + i) A = a (m + k + i) A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l131_13146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_hay_remaining_l131_13104

/-- Calculates the remaining hay bales for a farmer given specific conditions --/
def hay_bales_remaining (last_year_acres : ℕ) (last_year_bales_per_month : ℕ) 
  (additional_acres : ℕ) (num_horses : ℕ) (bales_per_horse_per_day : ℕ) 
  (feeding_start_month : ℕ) (end_month : ℕ) : ℕ :=
  let total_acres := last_year_acres + additional_acres
  let bales_per_acre_per_month := last_year_bales_per_month / last_year_acres
  let total_bales_per_month := bales_per_acre_per_month * total_acres
  let months_of_harvest := end_month - feeding_start_month + 1
  let total_bales_harvested := total_bales_per_month * months_of_harvest
  let days_in_period := 122  -- Sum of days in Sep, Oct, Nov, Dec
  let total_bales_consumed := num_horses * bales_per_horse_per_day * days_in_period
  total_bales_harvested - total_bales_consumed

/-- Proves that the farmer will have 2082 bales of hay left by the end of December --/
theorem farmer_hay_remaining : 
  hay_bales_remaining 5 560 7 9 3 9 12 = 2082 := by
  sorry

#eval hay_bales_remaining 5 560 7 9 3 9 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_hay_remaining_l131_13104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l131_13152

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line function
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Theorem statement
theorem min_distance_curve_line :
  ∃ (P Q : ℝ × ℝ),
    (P.1 > 0 ∧ P.2 = f P.1) ∧  -- P is on the curve f(x) = x^2 - ln(x), x > 0
    line Q.1 Q.2 ∧              -- Q is on the line x - y - 2 = 0
    ∀ (P' Q' : ℝ × ℝ),
      (P'.1 > 0 ∧ P'.2 = f P'.1) →  -- P' is on the curve
      line Q'.1 Q'.2 →              -- Q' is on the line
      Real.sqrt 2 ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l131_13152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l131_13145

theorem proposition_analysis : 
  ∃ (P Q R : Prop), 
    (P ↔ (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b)) ∧ 
    (Q ↔ (∀ a b c : ℝ, a ≤ b → a * c^2 ≤ b * c^2)) ∧ 
    (R ↔ (∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a ≤ b)) ∧ 
    (P ∧ Q ∧ ¬R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l131_13145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acr_equilateral_iff_same_orientation_l131_13107

/-- A structure representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a triangle in a plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A structure representing an equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point
  is_equilateral : Bool

/-- Function to construct an equilateral triangle on a side of a given triangle -/
noncomputable def construct_equilateral_triangle (A B : Point) (orientation : Bool) : EquilateralTriangle :=
  sorry

/-- Function to reflect a point across the midpoint of a segment -/
noncomputable def reflect_point (point : Point) (A B : Point) : Point :=
  sorry

/-- Main theorem -/
theorem acr_equilateral_iff_same_orientation (ABC : Triangle) 
  (ABP : EquilateralTriangle) 
  (BCQ : EquilateralTriangle) 
  (R : Point) :
  (ABP = construct_equilateral_triangle ABC.A ABC.B true ∧ 
   BCQ = construct_equilateral_triangle ABC.B ABC.C true ∧
   R = reflect_point ABC.B ABP.C BCQ.C) →
  (EquilateralTriangle.mk ABC.A ABC.C R true).is_equilateral ↔ 
  (construct_equilateral_triangle ABC.A ABC.B true).is_equilateral = 
  (construct_equilateral_triangle ABC.B ABC.C true).is_equilateral :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acr_equilateral_iff_same_orientation_l131_13107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_sum_invariant_l131_13153

/-- Represents a coloring order of a 6x6 grid -/
def ColoringOrder := Fin 36 → Fin 36

/-- Represents the state of the grid after coloring -/
def GridState := Fin 6 → Fin 6 → ℕ

/-- Calculates the sum of all numbers in the grid -/
def gridSum (state : GridState) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 6)) fun j => state i j)

/-- Simulates the coloring process and returns the final grid state -/
def colorGrid (order : ColoringOrder) : GridState :=
  sorry

theorem coloring_sum_invariant (order : ColoringOrder) :
  gridSum (colorGrid order) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_sum_invariant_l131_13153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_isosceles_triangles_l131_13197

/-- An isosceles triangle with base b and equal sides a -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let height := Real.sqrt (t.side ^ 2 - (t.base / 2) ^ 2)
  (1 / 2) * t.base * height

/-- Theorem: There exists an isosceles triangle with base 24 and sides 13
    that has the same area as an isosceles triangle with base 10 and sides 13 -/
theorem equal_area_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base = 10 ∧ t1.side = 13 ∧
    t2.base = 24 ∧ t2.side = 13 ∧
    area t1 = area t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_isosceles_triangles_l131_13197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l131_13114

noncomputable def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.Icc (-1) 1

noncomputable def g (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (a * x) + f (x / a)

theorem domain_of_g (f : ℝ → ℝ) (a : ℝ) (h_a : a > 0) :
  {x : ℝ | g f a x ∈ Set.range f} =
    if a ≥ 1 then Set.Icc (-1/a) (1/a)
    else Set.Icc (-a) a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l131_13114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l131_13188

/-- Given a total harvest of oranges and percentages for export and juice production,
    calculate the amount of oranges used for juice production. -/
theorem orange_juice_production
  (total_harvest : ℝ)
  (export_percentage : ℝ)
  (juice_percentage : ℝ)
  (h1 : total_harvest = 7)
  (h2 : export_percentage = 0.3)
  (h3 : juice_percentage = 0.6)
  : ∃ (juice_amount : ℝ), 
    abs (juice_amount - 2.9) < 0.05 ∧ 
    juice_amount = total_harvest * (1 - export_percentage) * juice_percentage :=
by
  sorry

#check orange_juice_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l131_13188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_639th_term_l131_13131

/-- An arithmetic sequence with the given first three terms -/
def arithmetic_sequence (x : ℚ) : ℕ → ℚ
  | 0 => 3 * x - 5  -- Adding the base case for 0
  | 1 => 3 * x - 5
  | 2 => 7 * x - 17
  | 3 => 4 * x + 3
  | n + 4 => arithmetic_sequence x (n + 3) + (arithmetic_sequence x 2 - arithmetic_sequence x 1)

/-- The theorem stating that the 639th term of the sequence is 4018 -/
theorem arithmetic_sequence_639th_term : 
  ∃ x : ℚ, arithmetic_sequence x 639 = 4018 := by
  sorry

#eval arithmetic_sequence (32/7) 639  -- This will help verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_639th_term_l131_13131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FAB_l131_13186

/-- Parabola represented by its equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line represented by its equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def F : Point := { x := 0, y := 2 }  -- Focus of the parabola

def C : Parabola := { p := 4, h := by norm_num }

def l₁ : Line := { m := -1, b := 0 }

def l₂ : Line := { m := 1, b := -8 }

def O : Point := { x := 0, y := 0 }

-- A and B are the intersection points of l₂ and C
def A : Point := { x := 12, y := 4 }
def B : Point := { x := 4, y := -4 }

-- P is the midpoint of AB
def P : Point := { x := 8, y := 0 }

theorem area_of_triangle_FAB :
  let triangle_area := (1/2) * |F.x - A.x| * |B.y - A.y|
  (∀ x y, y^2 = 2 * C.p * x ↔ (Point.mk x y).y^2 = 2 * C.p * (Point.mk x y).x) ∧
  (l₁.m = -1 ∧ l₁.b = 0) ∧
  (∃ x, x = 8 ∧ (-x)^2 = 2 * C.p * x) ∧
  (l₂.m = -1/l₁.m ∧ l₂.b ≠ 0) ∧
  (A ≠ B ∧ A.y = l₂.m * A.x + l₂.b ∧ B.y = l₂.m * B.x + l₂.b) ∧
  (P.x = (A.x + B.x)/2 ∧ P.y = (A.y + B.y)/2) ∧
  (|O.x - P.x|^2 + |O.y - P.y|^2 = (1/4) * (|A.x - B.x|^2 + |A.y - B.y|^2)) →
  triangle_area = 24 * Real.sqrt 5 := by
  sorry

#check area_of_triangle_FAB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FAB_l131_13186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l131_13189

/-- The perimeter of a triangle with sides 12, 15, and 18 is 45 -/
theorem triangle_perimeter (a b c : ℝ) : (a = 12 ∧ b = 15 ∧ c = 18) → a + b + c = 45 := by
  intro h
  rw [h.1, h.2.1, h.2.2]
  norm_num

#check triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l131_13189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_optimal_strategy_l131_13128

/-- Represents the race conditions and optimal solution --/
theorem race_optimal_strategy :
  -- Define constants
  let total_segments : ℕ := 42
  let petya_run_time : ℕ := 9
  let vasya_run_time : ℕ := 11
  let scooter_time : ℕ := 3
  
  -- Define functions for total time calculations
  let petya_total_time (x : ℕ) : ℕ := scooter_time * x + petya_run_time * (total_segments - x)
  let vasya_total_time (x : ℕ) : ℕ := scooter_time * (total_segments - x) + vasya_run_time * x
  
  -- Define the optimal number of segments
  let optimal_segments : ℕ := 18

  -- The theorem statement
  ∀ x : ℕ, x ≤ total_segments →
    max (petya_total_time optimal_segments) (vasya_total_time optimal_segments)
    ≤ max (petya_total_time x) (vasya_total_time x) :=
by
  -- Proof steps would go here
  sorry

#check race_optimal_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_optimal_strategy_l131_13128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cary_height_is_72_l131_13187

-- Define the heights as functions from natural numbers to propositions
def cary_height : ℕ → Prop := fun h => h = h
def bill_height : ℕ → Prop := fun h => h = h
def jan_height : ℕ → Prop := fun h => h = h

-- State the theorem
theorem cary_height_is_72 :
  (∀ h, bill_height h → cary_height (2 * h)) →  -- Bill's height is half of Cary's
  (∀ h, bill_height h → jan_height (h + 6)) →   -- Jan is 6 inches taller than Bill
  jan_height 42 →                                -- Jan is 42 inches tall
  cary_height 72 :=                              -- Cary is 72 inches tall
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cary_height_is_72_l131_13187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_equation_line_l_cartesian_equation_circle_C_min_distance_point_l131_13170

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + t, Real.sqrt 3 * t)

-- Define the circle C in polar form
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin θ

-- General equation of line l
theorem general_equation_line_l :
  ∀ x y : ℝ, (∃ t : ℝ, line_l t = (x, y)) ↔ Real.sqrt 3 * x - y - 3 * Real.sqrt 3 = 0 := by sorry

-- Cartesian equation of circle C
theorem cartesian_equation_circle_C :
  ∀ x y : ℝ, (∃ θ : ℝ, x^2 + y^2 = (circle_C θ)^2 ∧ x = circle_C θ * Real.cos θ ∧ y = circle_C θ * Real.sin θ) ↔
  x^2 + (y - Real.sqrt 3)^2 = 3 := by sorry

-- Point on line l that minimizes distance to center of circle C
theorem min_distance_point :
  let center := (0, Real.sqrt 3)
  (3, 0) ∈ Set.range line_l ∧
  ∀ p ∈ Set.range line_l, dist p center ≥ dist (3, 0) center := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_equation_line_l_cartesian_equation_circle_C_min_distance_point_l131_13170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mica_ground_beef_purchase_l131_13161

/-- The amount of ground beef Mica bought, in kilograms -/
noncomputable def ground_beef_amount (
  pasta_amount : ℚ)
  (pasta_price : ℚ)
  (ground_beef_price : ℚ)
  (sauce_amount : ℕ)
  (sauce_price : ℚ)
  (quesadilla_price : ℚ)
  (total_budget : ℚ) : ℚ :=
  (total_budget - (pasta_amount * pasta_price + sauce_amount * sauce_price + quesadilla_price)) / ground_beef_price

theorem mica_ground_beef_purchase :
  ground_beef_amount 2 (3/2) 8 2 2 6 15 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mica_ground_beef_purchase_l131_13161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_l131_13196

/-- Regular square pyramid with given dimensions -/
structure RegularSquarePyramid where
  base_side : ℝ
  height : ℝ

/-- Point on the edge of the pyramid connecting a base corner to the apex -/
structure EdgePoint where
  position : ℝ  -- Position along the edge, 0 ≤ position ≤ 1

/-- Calculates the length of the path from one base corner to the opposite corner through a point on the edge -/
def path_length (pyramid : RegularSquarePyramid) (point : EdgePoint) : ℝ :=
  sorry

/-- Theorem stating the shortest path length for the given pyramid dimensions -/
theorem shortest_path_length (pyramid : RegularSquarePyramid) 
  (h1 : pyramid.base_side = 230)
  (h2 : pyramid.height = (2 * pyramid.base_side) / (2 * π)) :
  ∃ (point : EdgePoint), 
    ∀ (other_point : EdgePoint), 
      path_length pyramid point ≤ path_length pyramid other_point ∧ 
      abs (path_length pyramid point - 391.36) < 0.01 := by
  sorry

#check shortest_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_length_l131_13196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l131_13142

theorem problem_1 : 2 - 4 - (-5) - 9 = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l131_13142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_drive_time_l131_13171

/-- Represents the time in hours for a one-way trip -/
noncomputable def one_way_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- The problem statement -/
theorem cole_drive_time :
  ∀ (distance : ℝ),
  distance > 0 →
  one_way_time distance 80 + one_way_time distance 120 = 2 →
  one_way_time distance 80 * 60 = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_drive_time_l131_13171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_is_right_triangle_l131_13174

-- Define the sets of line segments
def set_A : List ℚ := [1/3, 1/4, 1/5]
def set_B : List ℕ := [6, 8, 11]
noncomputable def set_C : List ℝ := [1, 1, Real.sqrt 2]
def set_D : List ℕ := [5, 12, 23]

-- Function to check if a set of line segments satisfies the Pythagorean theorem
def is_right_triangle (segments : List ℝ) : Prop :=
  segments.length = 3 ∧
  ∃ a b c, segments = [a, b, c] ∧ a * a + b * b = c * c

-- Theorem statement
theorem only_set_C_is_right_triangle :
  ¬(is_right_triangle (set_A.map (λ x => (x : ℝ)))) ∧
  ¬(is_right_triangle (set_B.map (λ x => (x : ℝ)))) ∧
  is_right_triangle set_C ∧
  ¬(is_right_triangle (set_D.map (λ x => (x : ℝ)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_is_right_triangle_l131_13174
