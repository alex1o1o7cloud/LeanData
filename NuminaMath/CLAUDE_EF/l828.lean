import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_selection_theorem_l828_82810

/-- Represents the number of ways to select n items from 4 sequences of n items each,
    such that each index appears exactly once and no two consecutive indices
    (including n and 1) are from the same sequence. -/
def selection_ways (n : ℕ) : ℤ :=
  if n ≥ 2 then 3^n + 3 * (-1)^n else 4

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

theorem card_selection_theorem :
  selection_ways cards_per_suit = 3^cards_per_suit - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_selection_theorem_l828_82810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l828_82895

/-- The constant term in the expansion of (x - 1/x^2)^6 is 15 -/
theorem constant_term_expansion : ∃ (c : ℤ), c = 15 ∧ 
  ∀ (x : ℝ), x ≠ 0 → (x - 1/x^2)^6 = c + x * ((x - 1/x^2)^6 - c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l828_82895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l828_82835

/-- The function f(x) = 3^(2x) -/
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ (2 * x)

/-- Theorem: For the function f(x) = 3^(2x), f(x+1) - f(x) = 8f(x) for all real x -/
theorem f_difference (x : ℝ) : f (x + 1) - f x = 8 * f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l828_82835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unions_required_l828_82833

/-- Represents a union of countries -/
structure CountryUnion :=
  (countries : Finset Nat)
  (size_constraint : countries.card ≤ 50)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (total_countries : Nat)
  (unions : Finset CountryUnion)
  (country_in_union : Nat → CountryUnion → Prop)
  (all_countries_covered : ∀ c, c < total_countries → ∃ u ∈ unions, country_in_union c u)
  (all_pairs_in_union : ∀ c1 c2, c1 < total_countries → c2 < total_countries → c1 ≠ c2 →
    ∃ u ∈ unions, country_in_union c1 u ∧ country_in_union c2 u)

/-- The main theorem stating the minimum number of unions required -/
theorem min_unions_required (setup : ProblemSetup) (h : setup.total_countries = 100) :
  setup.unions.card ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unions_required_l828_82833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_implies_m_eq_neg_three_l828_82800

/-- A function f: ℝ → ℝ is linear if it can be written as f(x) = ax + b for some constants a and b. -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

/-- Given function defined by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 3) * (x ^ (m^2 - 8)) + m + 1

/-- Theorem stating that if f is a linear function, then m = -3 -/
theorem linear_function_implies_m_eq_neg_three :
  (∃ m : ℝ, IsLinearFunction (f m)) → (∃ m : ℝ, m = -3 ∧ IsLinearFunction (f m)) :=
by
  sorry

#check linear_function_implies_m_eq_neg_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_implies_m_eq_neg_three_l828_82800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_exponent_base_l828_82838

theorem fourth_exponent_base (b : ℕ) (x : ℕ) :
  (18 ^ 7) * 9 ^ (3 * 7 - 1) = (2 ^ 7) * (x ^ b) →
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_exponent_base_l828_82838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l828_82840

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Line structure -/
structure Line where
  m : ℝ
  c : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Intersection points of a line and a hyperbola -/
def intersection (h : Hyperbola) (l : Line) : Set Point := sorry

/-- Perpendicular bisector of two points -/
def perp_bisector (p1 p2 : Point) : Line := sorry

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.c

/-- Main theorem -/
theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (l : Line) 
  (A B : Point) 
  (h_intersect : A ∈ intersection h l ∧ B ∈ intersection h l)
  (h_distinct : A ≠ B)
  (h_line : l.m = 1 ∧ l.c = -2)
  (h_perp : point_on_line ⟨4, 0⟩ (perp_bisector A B)) :
  eccentricity h = 2 * Real.sqrt 3 / 3 := by sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l828_82840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l828_82807

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * x^2 + a * x + 4

-- State the theorem
theorem monotonic_decreasing_interval (a : ℝ) :
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 4 → f a x > f a y) ∧
  (∀ x, x < -1 → f a x < f a (-1)) ∧
  (∀ x, x > 4 → f a x < f a 4) →
  a = -4 := by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l828_82807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_x_positive_l828_82899

theorem sin_pi_plus_x_positive (x : ℝ) :
  (∃ k : ℤ, x ∈ Set.Ioo ((2 * k + 1) * Real.pi) ((2 * k + 2) * Real.pi)) ↔ Real.sin (Real.pi + x) > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_x_positive_l828_82899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l828_82846

-- Define the functions
noncomputable def f (m : ℝ) (x : ℝ) := Real.log x / Real.log (m - 1)
noncomputable def g (m : ℝ) (x : ℝ) := -(5 - 2*m)^x

-- Define the propositions
def p (m : ℝ) := ∀ x₁ x₂, x₁ < x₂ → f m x₁ > f m x₂
def q (m : ℝ) := ∀ x₁ x₂, x₁ < x₂ → g m x₁ > g m x₂

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m, p m ∧ q m) ∧ (∃ m, ¬p m ∧ q m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l828_82846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_bound_l828_82859

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if 3a² = 2b² + c², then S/(b² + 2c²) ≤ √14/24 -/
theorem triangle_area_ratio_bound (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  S = (1/2) * b * c * Real.sin A →
  3 * a^2 = 2 * b^2 + c^2 →
  S / (b^2 + 2 * c^2) ≤ Real.sqrt 14 / 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_bound_l828_82859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l828_82836

/-- Given an ellipse with equation 2x^2 + 3y^2 = 1, its major axis length is √2 -/
theorem ellipse_major_axis_length :
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), 2 * x^2 + 3 * y^2 = 1 → x^2 / a^2 + y^2 / b^2 = 1) ∧
    2 * a = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l828_82836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dp_dq_harmonic_mean_inequality_l828_82852

-- Define the triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the lengths of the legs
noncomputable def leg_length (t : RightTriangle) (leg : ℕ) : ℝ :=
  match leg with
  | 0 => Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  | _ => Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)

-- Define the foot of altitude D
noncomputable def altitude_foot (t : RightTriangle) : ℝ × ℝ := sorry

-- Define the feet of perpendiculars P and Q
noncomputable def perpendicular_foot (t : RightTriangle) (leg : ℕ) : ℝ × ℝ := sorry

-- Define the length of a segment between two points
noncomputable def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem dp_dq_harmonic_mean_inequality (t : RightTriangle) :
  let D := altitude_foot t
  let P := perpendicular_foot t 0
  let Q := perpendicular_foot t 1
  let a := leg_length t 0
  let b := leg_length t 1
  segment_length D P + segment_length D Q ≤ 2 * a * b / (a + b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dp_dq_harmonic_mean_inequality_l828_82852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l828_82850

theorem sin_alpha_value (α : Real) (m : Real) :
  α ∈ Set.Ioo (π / 2) π →  -- α is in the second quadrant
  m < 0 →  -- m is negative (implied by second quadrant)
  (∃ P : ℝ × ℝ, P = (m, Real.sqrt 5)) →  -- point on terminal side
  Real.cos α = (Real.sqrt 2 / 4) * m →  -- given cosine relation
  Real.sin α = Real.sqrt 10 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l828_82850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l828_82880

open Real

-- Define the function g in terms of f
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x * f (x + π/2)

-- Part 1
theorem part1 (x : ℝ) : g (λ y => cos y + sin y) x = cos (2 * x) := by
  sorry

-- Part 2
theorem part2 : ∃ x₁ x₂ : ℝ, 
  (∀ x : ℝ, g (λ y => abs (sin y) + cos y) x₁ ≤ g (λ y => abs (sin y) + cos y) x ∧ 
              g (λ y => abs (sin y) + cos y) x ≤ g (λ y => abs (sin y) + cos y) x₂) ∧
  ∀ y₁ y₂ : ℝ, (∀ x : ℝ, g (λ y => abs (sin y) + cos y) y₁ ≤ g (λ y => abs (sin y) + cos y) x ∧ 
                          g (λ y => abs (sin y) + cos y) x ≤ g (λ y => abs (sin y) + cos y) y₂) →
              abs (x₁ - x₂) ≤ abs (y₁ - y₂) ∧ abs (x₁ - x₂) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l828_82880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_semicircle_square_value_l828_82866

/-- The perimeter of a region bounded by four semicircular arcs constructed on the sides of a square with side length 1/π -/
noncomputable def perimeter_semicircle_square (π : ℝ) : ℝ :=
  2 + 4 / π

/-- Theorem stating that the perimeter of the described region is 2 + 4/π -/
theorem perimeter_semicircle_square_value (π : ℝ) (h : π > 0) :
  perimeter_semicircle_square π = 2 + 4 / π := by
  -- Unfold the definition of perimeter_semicircle_square
  unfold perimeter_semicircle_square
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_semicircle_square_value_l828_82866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l828_82886

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 3)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem max_product_of_distances :
  ∀ (m x y : ℝ),
  x + m * y + 2 = 0 →
  m * x - y - 2 * m + 3 = 0 →
  (distance (x, y) A) * (distance (x, y) B) ≤ 25 / 2 :=
by
  sorry

#check max_product_of_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l828_82886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l828_82854

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0 ∧ b > 0 ∧ c > 0

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The distance from the focus to the asymptote of a hyperbola -/
noncomputable def Hyperbola.focusToAsymptote (h : Hyperbola) : ℝ := 
  (h.b * h.c) / Real.sqrt (h.b^2 + h.a^2)

/-- The equation of a hyperbola in standard form -/
def Hyperbola.equation (h : Hyperbola) (x y : ℝ) : Prop := x^2 / h.a^2 - y^2 / h.b^2 = 1

theorem hyperbola_equation (h : Hyperbola) 
  (h_eccentricity : h.eccentricity = Real.sqrt 6 / 2)
  (h_focus_asymptote : h.focusToAsymptote = 1) :
  h.equation = fun x y => x^2 / 2 - y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l828_82854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_same_slope_through_point_l828_82879

/-- Given a line with equation 2x + 4y - 3 = 0, prove that the line
    x + 2y - 8 = 0 has the same slope and passes through (2, 3) -/
theorem line_with_same_slope_through_point :
  let l₁ : ℝ → ℝ → Prop := λ x y => 2 * x + 4 * y - 3 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => x + 2 * y - 8 = 0
  ∀ x y, l₂ x y ↔ (∃ k, ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)) ∧ l₂ 2 3 := by
  sorry

#check line_with_same_slope_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_same_slope_through_point_l828_82879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_congruence_l828_82844

theorem smallest_four_digit_congruence : ∃ (x : ℤ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (5 * x ≡ 15 [ZMOD 10]) ∧
  (3 * x + 10 ≡ 13 [ZMOD 8]) ∧
  (13 * x + 2 ≡ 2 [ZMOD 16]) ∧
  (∀ y : ℤ, y ≥ 1000 ∧ y < 10000 ∧
    (5 * y ≡ 15 [ZMOD 10]) ∧
    (3 * y + 10 ≡ 13 [ZMOD 8]) ∧
    (13 * y + 2 ≡ 2 [ZMOD 16]) →
    x ≤ y) ∧
  x = 1008 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_congruence_l828_82844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_opposite_changes_l828_82822

/-- The function f: (0,1) → (0,1) -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < (1/2 : ℝ) then x + (1/2 : ℝ) else x^2

/-- Sequence a_n -/
noncomputable def a : ℕ → ℝ → ℝ
  | 0, a₀ => a₀
  | n+1, a₀ => f (a n a₀)

/-- Sequence b_n -/
noncomputable def b : ℕ → ℝ → ℝ
  | 0, b₀ => b₀
  | n+1, b₀ => f (b n b₀)

/-- Theorem: There exists a positive integer n such that (a_n - a_{n-1})(b_n - b_{n-1}) < 0 -/
theorem exists_opposite_changes (a₀ b₀ : ℝ) 
  (h_a : 0 < a₀) (h_b : a₀ < b₀) (h_b1 : b₀ < 1) :
  ∃ n : ℕ+, (a n.val a₀ - a (n.val-1) a₀) * (b n.val b₀ - b (n.val-1) b₀) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_opposite_changes_l828_82822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_digit_proof_l828_82832

def missing_digit : Fin 10 := 6

theorem missing_digit_proof :
  ∃ (x : Fin 10), 
    (42 : ℤ) = (-80538738812075970 - x.val) ^ 3 + 80435758145817515 ^ 3 + 12602123297335631 ^ 3 
    ∧ x = missing_digit :=
by
  use missing_digit
  sorry

#eval missing_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_digit_proof_l828_82832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_melt_spillover_l828_82893

/-- The volume of water that spills out when ice melts in a full glass of saltwater --/
theorem ice_melt_spillover
  (m : ℝ)
  (ρ_n ρ_c ρ_ns : ℝ)
  (h_m : m = 100)
  (h_ρ_n : ρ_n = 0.92)
  (h_ρ_c : ρ_c = 0.952)
  (h_ρ_ns : ρ_ns = 1)
  (h_ρ_n_pos : ρ_n > 0)
  (h_ρ_c_pos : ρ_c > 0)
  (h_ρ_ns_pos : ρ_ns > 0) :
  ∃ (ΔV : ℝ), ΔV = m * (1 / ρ_n - 1 / ρ_c) ∧ abs (ΔV - 5.26) < 0.01 := by
  sorry

#check ice_melt_spillover

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_melt_spillover_l828_82893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_number_l828_82812

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 2000 ∧
  (Finset.card (Nat.divisors n)) = 14 ∧
  ∃ p, Nat.Prime p ∧ n % p = 0 ∧ p % 10 = 1

theorem unique_valid_number :
  ∃! n, is_valid_number n ∧ n = 1984 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_number_l828_82812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l828_82882

-- Renamed 'sequence' to 'is_special_sequence' to avoid naming conflict
def is_special_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1)^2 - a (n + 1) = a n)

theorem sequence_properties (a : ℕ → ℝ) (h : is_special_sequence a) :
  ((0 < a 1 ∧ a 1 < 2) → (∀ n, a (n + 1) > a n)) ∧
  (a 1 > 2 → (∀ n ≥ 2, 2 < a n ∧ a n < a 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l828_82882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_stolen_bag_in_two_weighings_l828_82878

/-- Represents a bag with its original weight -/
structure Bag where
  weight : Nat
  deriving Repr

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal
  | LeftHeavier
  | RightHeavier
  deriving Repr

/-- Represents a weighing on the balance scale -/
def Weighing := List Bag → List Bag → WeighResult

/-- The set of bags with their original weights -/
def bags : List Bag := [1, 2, 3, 4, 5, 6, 7, 8, 9].map (λ w => ⟨w⟩)

/-- Theorem stating that it's possible to determine the stolen bag in two weighings -/
theorem determine_stolen_bag_in_two_weighings :
  ∃ (weighing1 weighing2 : Weighing),
    ∀ (stolen_bag : Bag),
      stolen_bag ∈ bags →
      ∃ (result1 : WeighResult) (result2 : WeighResult),
        ∃ (group1 group2 : List Bag),
          (weighing1 group1 group2 = result1) ∧
          (weighing2 group1 group2 = result2) ∧
          (stolen_bag = 
            match result1, result2 with
            | _, _ => stolen_bag  -- Placeholder; actual logic would go here
          ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_stolen_bag_in_two_weighings_l828_82878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_distance_to_directrix_l828_82889

/-- Given a parabola y² = 2px (p > 0) with focus F and point A(0, 2),
    if the midpoint B of FA lies on the parabola,
    then the distance from B to the directrix is 3√2/4 -/
theorem parabola_midpoint_distance_to_directrix (p : ℝ) (h_p : p > 0) :
  let F : ℝ × ℝ := (p/2, 0)
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := ((F.1 + A.1)/2, (F.2 + A.2)/2)
  (B.2)^2 = 2*p*B.1 →
  |B.1 - (-p/2)| = 3*Real.sqrt 2/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_distance_to_directrix_l828_82889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_calculation_l828_82898

/-- The cost of a taxi ride given the distance and fixed parameters -/
noncomputable def taxi_cost (fixed_charge : ℝ) (base_distance : ℝ) (base_cost : ℝ) (distance : ℝ) : ℝ :=
  fixed_charge + (distance * (base_cost - fixed_charge) / base_distance)

theorem taxi_fare_calculation 
  (fixed_charge : ℝ) 
  (base_distance : ℝ) 
  (base_cost : ℝ) 
  (target_distance : ℝ)
  (h1 : fixed_charge = 15)
  (h2 : base_distance = 40)
  (h3 : base_cost = 95)
  (h4 : target_distance = 60) :
  taxi_cost fixed_charge base_distance base_cost target_distance = 135 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_calculation_l828_82898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increase_formula_triangle_area_increase_at_3_l828_82891

/-- Represents the increase in area of a triangle when its base and height are increased. -/
noncomputable def triangle_area_increase (x : ℝ) : ℝ :=
  let initial_base := 2 * x + 1
  let initial_height := x - 2
  let new_base := initial_base + 5
  let new_height := initial_height + 5
  let initial_area := (1 / 2) * initial_base * initial_height
  let new_area := (1 / 2) * new_base * new_height
  new_area - initial_area

/-- Theorem stating the increase in triangle area for any x. -/
theorem triangle_area_increase_formula (x : ℝ) :
  triangle_area_increase x = (15 / 2) * x + 10 := by sorry

/-- Theorem stating the increase in triangle area when x = 3. -/
theorem triangle_area_increase_at_3 :
  triangle_area_increase 3 = 32.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increase_formula_triangle_area_increase_at_3_l828_82891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xy_plane_l828_82809

theorem equidistant_point_in_xy_plane :
  let p : ℝ × ℝ × ℝ := (31/10, -11/5, 0)
  let a : ℝ × ℝ × ℝ := (0, 2, 0)
  let b : ℝ × ℝ × ℝ := (1, -1, 3)
  let c : ℝ × ℝ × ℝ := (4, 0, -2)
  let dist (x y : ℝ × ℝ × ℝ) := 
    (x.1 - y.1)^2 + (x.2.1 - y.2.1)^2 + (x.2.2 - y.2.2)^2
  dist p a = dist p b ∧ dist p a = dist p c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xy_plane_l828_82809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l828_82815

open Real

-- Define the function f(x, y, a) = x + y^2 * e^y - a
noncomputable def f (x y a : ℝ) : ℝ := x + y^2 * (exp y) - a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Set.Icc 0 1 → ∃! y, y ∈ Set.Icc (-1) 1 ∧ f x y a = 0) ↔ 
  a ∈ Set.Ioo (1 + exp (-1)) (exp 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l828_82815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l828_82856

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem stating that the area of a trapezium with parallel sides of lengths 12 and 16, 
    and a distance of 14 between them, is equal to 196. -/
theorem trapezium_area_example : trapeziumArea 12 16 14 = 196 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l828_82856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_coefficient_l828_82811

/-- A cubic function passing through three specific points has a specific coefficient. -/
theorem cubic_function_coefficient (p q r s : ℝ) (g : ℝ → ℝ) : 
  (∀ x, g x = p * x^3 + q * x^2 + r * x + s) →
  g (-2) = 0 →
  g 0 = -3 →
  g 2 = 0 →
  q = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_coefficient_l828_82811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l828_82876

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The sum of 78.2514 and 34.7859 rounded to the nearest hundredth is 113.04 -/
theorem sum_and_round : round_to_hundredth (78.2514 + 34.7859) = 113.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l828_82876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorize_quadratic_min_value_quadratic_max_value_quadratic_triangle_shape_isosceles_triangle_l828_82875

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 - 4*m - 5 = (m + 1)*(m - 5) := by
  sorry

-- Problem 2
theorem min_value_quadratic : 
  ∀ x : ℝ, x^2 - 6*x + 12 ≥ 3 := by
  sorry

-- Problem 3
theorem max_value_quadratic :
  let f := λ x : ℝ ↦ -x^2 + 2*x - 3
  ∃ x_max : ℝ, x_max = 1 ∧ 
    (∀ x : ℝ, f x ≤ f x_max) ∧ f x_max = -2 := by
  sorry

-- Problem 4
theorem triangle_shape (a b c : ℝ) :
  a^2 + b^2 + c^2 - 6*a - 10*b - 6*c + 43 = 0 →
  a = 3 ∧ b = 5 ∧ c = 3 := by
  sorry

-- Additional theorem to state it's an isosceles triangle
theorem isosceles_triangle (a b c : ℝ) :
  a = 3 ∧ b = 5 ∧ c = 3 →
  a = c ∧ a ≠ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorize_quadratic_min_value_quadratic_max_value_quadratic_triangle_shape_isosceles_triangle_l828_82875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l828_82834

theorem vector_magnitude (a b : EuclideanSpace ℝ (Fin 3)) : 
  ‖a‖ = 1 → ‖b‖ = 1 → ‖2 • a - b‖ = Real.sqrt 5 → ‖a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l828_82834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_envelope_probability_l828_82884

noncomputable def red_envelope_amounts : Finset ℚ := {1.49, 1.81, 2.19, 3.41, 0.62, 0.48}

noncomputable def favorable_pairs (s : Finset ℚ) : Finset (ℚ × ℚ) :=
  s.product s |>.filter (fun p => p.1 + p.2 ≥ 4 ∧ p.1 ≠ p.2)

theorem red_envelope_probability :
  (favorable_pairs red_envelope_amounts).card / Nat.choose red_envelope_amounts.card 2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_envelope_probability_l828_82884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_midline_l828_82874

/-- A sine function with positive constants a, b, c, and d -/
noncomputable def sine_function (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.sin (b * x + c) + d

theorem sine_function_midline
  (a b c d : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_oscillation : ∀ x : ℝ, -2 ≤ sine_function a b c d x ∧ sine_function a b c d x ≤ 4) :
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_midline_l828_82874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_import_tax_l828_82828

/-- Calculate the import tax for an item -/
noncomputable def import_tax (total_value : ℝ) (tax_rate : ℝ) (threshold : ℝ) : ℝ :=
  max 0 ((total_value - threshold) * tax_rate)

/-- The import tax calculation is correct -/
theorem correct_import_tax : 
  let total_value : ℝ := 2250
  let tax_rate : ℝ := 0.07
  let threshold : ℝ := 1000
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_import_tax_l828_82828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_ratio_l828_82826

theorem cube_volume_ratio (e : ℝ) (h : e > 0) : 
  (2*e)^3 / e^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_ratio_l828_82826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fe2o3_formation_heat_l828_82873

-- Define the heat of formation for compounds
def heat_of_formation (compound : String) : ℝ := sorry

-- Define the heat released in a reaction
def heat_released (reaction : String) : ℝ := sorry

-- Given conditions
axiom al_oxidation : heat_released "2Al + 1.5O₂ = Al₂O₃" = 1675.5

-- Define the reaction equation
axiom reaction_equation : heat_released "2Al + Fe₂O₃ = Al₂O₃ + 2Fe" = 
  heat_of_formation "Al₂O₃" - heat_of_formation "Fe₂O₃"

-- Theorem to prove
theorem fe2o3_formation_heat : heat_of_formation "Fe₂O₃" = 821.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fe2o3_formation_heat_l828_82873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_statement_l828_82804

-- Define the basic geometric objects
structure Point where

structure Line where

structure Plane where

-- Define the relationships between geometric objects
axiom Point.inPlane : Point → Plane → Prop
axiom Point.onLine : Point → Line → Prop
axiom Line.inPlane : Line → Plane → Prop
axiom Plane.intersect : Plane → Plane → Set Point
axiom Line.intersect : Line → Line → Set Point

-- Define the statements
def statement1 : Prop :=
  ∀ π₁ π₂ : Plane, ∀ p₁ p₂ p₃ : Point,
    Point.inPlane p₁ π₁ ∧ Point.inPlane p₁ π₂ ∧
    Point.inPlane p₂ π₁ ∧ Point.inPlane p₂ π₂ ∧
    Point.inPlane p₃ π₁ ∧ Point.inPlane p₃ π₂ →
    π₁ = π₂

def statement2 : Prop :=
  ∀ l₁ l₂ : Line, ∃ π : Plane, Line.inPlane l₁ π ∧ Line.inPlane l₂ π

def statement3 : Prop :=
  ∀ π₁ π₂ : Plane, ∀ l : Line, ∀ p : Point,
    Point.inPlane p π₁ ∧ Point.inPlane p π₂ ∧ Plane.intersect π₁ π₂ = {q | Point.onLine q l} →
    Point.onLine p l

def statement4 : Prop :=
  ∀ l₁ l₂ l₃ : Line, ∀ p : Point,
    (∃ q : Point, q ∈ Line.intersect l₁ l₂ ∧ q ∈ Line.intersect l₂ l₃ ∧ q ∈ Line.intersect l₃ l₁) →
    ∃ π : Plane, Line.inPlane l₁ π ∧ Line.inPlane l₂ π ∧ Line.inPlane l₃ π

-- The main theorem
theorem exactly_one_correct_statement :
  (statement1 = false) ∧ (statement2 = false) ∧ (statement3 = true) ∧ (statement4 = false) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_statement_l828_82804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_power_series_expansion_power_series_convergence_l828_82890

noncomputable def f (x : ℝ) := Real.log (x + Real.sqrt (1 + x^2))

noncomputable def series_term (n : ℕ) (x : ℝ) : ℝ :=
  (-1)^n * (2*n).factorial / (4^n * (n.factorial)^2) * x^(2*n+1) / (2*n+1 : ℝ)

noncomputable def power_series (x : ℝ) : ℝ := ∑' n, series_term n x

theorem f_power_series_expansion (x : ℝ) (h : |x| ≤ 1) :
  f x = power_series x := by
  sorry

theorem power_series_convergence (x : ℝ) :
  Summable (λ n => series_term n x) ↔ |x| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_power_series_expansion_power_series_convergence_l828_82890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l828_82885

/-- Calculates the time (in seconds) it takes for a train to cross an electric pole. -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem: A 75-meter long train running at 54 km/hr takes 5 seconds to cross an electric pole. -/
theorem train_crossing_pole_time :
  train_crossing_time 75 54 = 5 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l828_82885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_sum_l828_82803

theorem greatest_power_sum (a b : ℕ) : 
  b > 1 → 
  a^b < 500 → 
  (∀ c d : ℕ, c > 0 → d > 0 → c^d < 500 → c^d ≤ a^b) → 
  Even (a + b) → 
  a + b = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_sum_l828_82803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l828_82877

/-- The function f(x) defined for all real numbers x -/
noncomputable def f (x : ℝ) : ℝ := 1 / (|x + 3| + |x + 1| + |x - 2| + |x - 5|)

/-- Theorem stating that the maximum value of f(x) is 1/11 -/
theorem f_max_value :
  ∃ (M : ℝ), M = 1/11 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l828_82877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_sum_eq_floor_sqrt_quad_l828_82817

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_sqrt_sum_eq_floor_sqrt_quad (n : ℤ) :
  floor (Real.sqrt (n : ℝ) + Real.sqrt ((n + 1) : ℝ)) = floor (Real.sqrt ((4 * n + 2) : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_sum_eq_floor_sqrt_quad_l828_82817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_iff_a_zero_l828_82820

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - 1 / x + a

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = -f x

theorem odd_function_iff_a_zero (a : ℝ) :
  (∀ x, x ≠ 0 → f a (-x) = -(f a x)) ↔ a = 0 := by
  sorry

#check odd_function_iff_a_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_iff_a_zero_l828_82820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_square_nor_cube_if_digital_root_2356_l828_82871

/-- The digital root of an integer -/
def digitalRoot (n : ℤ) : ℕ :=
  if n = 0 then 0 else (((n.natAbs - 1) % 9 : ℕ) + 1)

/-- Theorem: If the digital root of an integer is 2, 3, 5, or 6, 
    then it's neither a perfect square nor a perfect cube of a positive integer -/
theorem not_square_nor_cube_if_digital_root_2356 (n : ℤ) :
  (digitalRoot n = 2 ∨ digitalRoot n = 3 ∨ digitalRoot n = 5 ∨ digitalRoot n = 6) →
  (∀ m : ℤ, m > 0 → n ≠ m ^ 2 ∧ n ≠ m ^ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_square_nor_cube_if_digital_root_2356_l828_82871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_test_l828_82839

/-- The number of students who received a question from each part of the course -/
def observed_frequencies : List ℕ := [26, 32, 17, 25]

/-- The total number of students -/
def total_students : ℕ := 100

/-- The expected frequency for each part under the null hypothesis -/
noncomputable def expected_frequency : ℝ := (total_students : ℝ) / 4

/-- The chi-square statistic for the observed frequencies -/
noncomputable def chi_square_statistic : ℝ :=
  (List.map (λ x => ((x : ℝ) - expected_frequency)^2 / expected_frequency) observed_frequencies).sum

/-- The critical value for a chi-square distribution with 3 degrees of freedom at α = 0.05 -/
def critical_value : ℝ := 7.815

/-- Theorem stating that the chi-square statistic is less than the critical value -/
theorem chi_square_test : chi_square_statistic < critical_value := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_test_l828_82839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l828_82855

/-- Given a quadratic function f(x) = x^2 + 4x + c, prove that f(1) > f(0) > f(-2) -/
theorem quadratic_inequality (c : ℝ) : 
  let f := λ x : ℝ => x^2 + 4*x + c
  f 1 > f 0 ∧ f 0 > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l828_82855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l828_82816

/-- Circle in rectangular coordinates -/
def circle_eq (x y : ℝ) : Prop :=
  (x - Real.sqrt 3)^2 + (y - 1)^2 = 4

/-- Line in polar coordinates -/
def line_eq (θ : ℝ) : Prop :=
  θ = Real.pi/3

/-- Theorem stating the area of the triangle -/
theorem triangle_area (x y ρ θ : ℝ) (h1 : circle_eq x y) (h2 : line_eq θ) :
  ∃ (area : ℝ), area = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l828_82816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l828_82842

-- Define the function representing the fraction in the inequality
noncomputable def f (x : ℝ) : ℝ := (x^2 - 16*x + 15) / (x^2 - 4*x + 5)

-- Define the inequality condition
def satisfies_inequality (x : ℝ) : Prop := -2 < f x ∧ f x < 2

-- Define the two intervals (exact definitions would depend on the specific solution)
def interval1 : Set ℝ := {x | x > 0} -- placeholder definition
def interval2 : Set ℝ := {x | x < 10} -- placeholder definition

-- The theorem to be proved
theorem inequality_solution (x : ℝ) :
  satisfies_inequality x ↔ x ∈ (interval1 ∩ interval2) := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_continuous : Continuous f := by
  sorry

lemma denominator_positive (x : ℝ) : 0 < x^2 - 4*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l828_82842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l828_82827

-- Define the basic geometric objects
variable (α : Type) -- Plane α
variable (a b : Type) -- Lines a and b

-- Define the geometric relationships
def intersect (l1 l2 : Type) : Prop := sorry
def parallel_to_plane (l : Type) (p : Type) : Prop := sorry
def intersects_plane (l : Type) (p : Type) : Prop := sorry

-- State the theorem
theorem line_plane_relationship
  (α : Type) (a b : Type)
  (h1 : intersect a b)
  (h2 : parallel_to_plane a α) :
  intersects_plane b α ∨ parallel_to_plane b α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l828_82827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_interesting_quadruples_approx_l828_82883

def is_interesting (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d > 2 * (b + c)

def count_interesting_quadruples : ℕ :=
  (List.range 15).foldr
    (λ a acc₁ => (List.range 15).foldr
      (λ b acc₂ => (List.range 15).foldr
        (λ c acc₃ => (List.range 15).foldr
          (λ d acc₄ => 
            if (1 ≤ a+1) ∧ (a+1 < b+1) ∧ (b+1 < c+1) ∧ (c+1 < d+1) ∧ (d+1 ≤ 15) ∧ ((a+1) + (d+1) > 2 * ((b+1) + (c+1)))
            then acc₄ + 1 
            else acc₄)
          acc₃)
        acc₂)
      acc₁)
    0

-- Approximate equality for naturals
def approx_equal (n m : ℕ) (ε : ℕ) : Prop :=
  (n ≤ m + ε) ∧ (m ≤ n + ε)

notation:50 a " ≈[" ε "] " b => approx_equal a b ε

theorem count_interesting_quadruples_approx :
  count_interesting_quadruples ≈[10] 500 := by
  sorry

#eval count_interesting_quadruples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_interesting_quadruples_approx_l828_82883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l828_82870

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f ((x - y)^2) = (f x)^2 - 2*x*(f y) + y^2) →
  (∀ x : ℝ, f x = x ∨ f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l828_82870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_form_min_distance_to_focus_l828_82862

/-- Parametric equations of the ellipse -/
noncomputable def ellipse_x (θ : ℝ) : ℝ := 2 * Real.cos θ
noncomputable def ellipse_y (θ : ℝ) : ℝ := 3 * Real.sin θ

/-- Standard form of the ellipse -/
theorem ellipse_standard_form (x y : ℝ) :
  (∃ θ, x = ellipse_x θ ∧ y = ellipse_y θ) ↔ x^2 / 4 + y^2 / 9 = 1 := by
  sorry

/-- Minimum distance from a point on the ellipse to its focus -/
theorem min_distance_to_focus :
  (∃ x y : ℝ, (∃ θ, x = ellipse_x θ ∧ y = ellipse_y θ) ∧
    ∀ x' y' : ℝ, (∃ θ', x' = ellipse_x θ' ∧ y' = ellipse_y θ') →
      (∃ fx fy : ℝ, (x - fx)^2 + (y - fy)^2 ≤ (x' - fx)^2 + (y' - fy)^2)) →
  (∃ d : ℝ, d = 3 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_form_min_distance_to_focus_l828_82862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_flip_probability_l828_82819

-- Define a fair coin
noncomputable def fair_coin (outcome : Bool) : ℚ := 1 / 2

-- Define the sequence of coin flips
def coin_flips : List Bool := [false, false, false, false, false, false]

-- Theorem statement
theorem seventh_flip_probability :
  fair_coin true = 1 / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_flip_probability_l828_82819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distances_l828_82892

/-- Represents a city on the map -/
structure City where
  name : String

/-- Represents the map with cities and distances -/
structure Map where
  scale : ℚ  -- km per cm
  distanceAB : ℚ  -- distance between A and B in cm
  ratioABC : ℚ  -- ratio of AB to BC

/-- Calculates the real distance between two cities given the map distance -/
def realDistance (m : Map) (mapDist : ℚ) : ℚ :=
  mapDist * m.scale

/-- Calculates the distance between City A and City C -/
def distanceAC (m : Map) : ℚ :=
  (m.distanceAB * m.scale) / (1 - m.ratioABC)

theorem city_distances (m : Map) 
    (h1 : m.scale = 20) 
    (h2 : m.distanceAB = 120) 
    (h3 : m.ratioABC = 3/4) : 
    realDistance m m.distanceAB = 2400 ∧ distanceAC m = 9600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distances_l828_82892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l828_82847

theorem cube_surface_area_increase (x : ℝ) (h : x > 0) : 
  (6 * (1.5 * x)^2 - 6 * x^2) / (6 * x^2) = 1.25 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l828_82847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fog_time_is_twenty_l828_82824

/-- Represents the cycling scenario with given speeds and total distance and time -/
structure CyclingScenario where
  clear_speed : ℝ  -- Speed in clear weather (miles per hour)
  fog_speed : ℝ    -- Speed in foggy conditions (miles per hour)
  total_distance : ℝ  -- Total distance covered (miles)
  total_time : ℝ      -- Total time spent (minutes)

/-- Calculates the time spent cycling in fog -/
noncomputable def time_in_fog (scenario : CyclingScenario) : ℝ :=
  let clear_speed_per_minute := scenario.clear_speed / 60
  let fog_speed_per_minute := scenario.fog_speed / 60
  ((clear_speed_per_minute * scenario.total_time - scenario.total_distance) /
   (clear_speed_per_minute - fog_speed_per_minute))

/-- Theorem stating that for the given scenario, the time spent in fog is 20 minutes -/
theorem fog_time_is_twenty (scenario : CyclingScenario)
  (h1 : scenario.clear_speed = 40)
  (h2 : scenario.fog_speed = 15)
  (h3 : scenario.total_distance = 25)
  (h4 : scenario.total_time = 50) :
  time_in_fog scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fog_time_is_twenty_l828_82824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l828_82805

theorem min_value_theorem (a b : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1 ∧ 
    (fun x ↦ x^3 + a*x^2 + b*x) x₁ = (fun x ↦ x^3 + a*x^2 + b*x) x₂ ∧
    (fun x ↦ x^3 + a*x^2 + b*x) x₂ = (fun x ↦ x^3 + a*x^2 + b*x) x₃) →
  abs a + 2 * abs b ≥ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l828_82805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_occupies_four_percent_l828_82851

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  -- Total area of the flag
  total_area : ℝ
  -- Percentage of area occupied by the entire cross (arms + center)
  cross_percentage : ℝ
  -- Percentage of area occupied by the arms of the cross (excluding center)
  arms_percentage : ℝ
  -- Assumption that the flag is square and the cross is symmetric
  h_square_symmetric : True
  -- The cross occupies 40% of the flag's area
  h_cross_area : cross_percentage = 0.4
  -- The arms occupy 36% of the flag's area
  h_arms_area : arms_percentage = 0.36

/-- The percentage of the flag's area occupied by the center of the cross -/
def center_percentage (flag : SquareFlag) : ℝ :=
  flag.cross_percentage - flag.arms_percentage

/-- Theorem stating that the center of the cross occupies 4% of the flag's area -/
theorem center_occupies_four_percent (flag : SquareFlag) :
  center_percentage flag = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_occupies_four_percent_l828_82851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_formula_a_3_equals_3_a_5_equals_13_l828_82831

-- Define the sequence recursively
def a : ℕ → ℤ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then a (n + 1) + 3^((n + 2) / 2) else a (n + 1) + (-1)^((n + 2) / 2)

-- Define the general term formula
def a_formula (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    (3^(n / 2) / 2) + ((-1)^(n / 2) / 2) - 1
  else
    (3^((n + 1) / 2) / 2) + ((-1)^((n - 1) / 2) / 2) - 1

-- Theorem statement
theorem a_equals_formula (n : ℕ) : n ≥ 1 → a n = a_formula n := by
  sorry

-- Additional theorems to verify specific values
theorem a_3_equals_3 : a 3 = 3 := by
  sorry

theorem a_5_equals_13 : a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_formula_a_3_equals_3_a_5_equals_13_l828_82831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_progress_check_days_l828_82864

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  initialWorkers : ℚ
  progressLength : ℚ
  extraWorkersNeeded : ℚ

/-- Calculates the number of days after which the progress was checked -/
def daysUntilCheck (project : RoadProject) : ℚ :=
  (project.progressLength * project.totalDays) / project.totalLength

/-- Theorem stating that the engineer checked the progress after 12 days -/
theorem progress_check_days (project : RoadProject) 
  (h1 : project.totalLength = 10)
  (h2 : project.totalDays = 60)
  (h3 : project.initialWorkers = 30)
  (h4 : project.progressLength = 2)
  (h5 : project.extraWorkersNeeded = 30) :
  daysUntilCheck project = 12 := by
  sorry

#check progress_check_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_progress_check_days_l828_82864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_decreasing_on_neg_interval_l828_82802

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x^(1/3) 
  else (-x)^(1/3)

theorem f_even : ∀ x, f (-x) = f x := by sorry

theorem f_decreasing_on_neg_interval : 
  ∀ x y, -2 < x ∧ x < y ∧ y < 0 → f x > f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_decreasing_on_neg_interval_l828_82802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_divided_by_point_three_repeating_l828_82869

/-- Given that 0.3̄ is equal to 1/3, prove that 8 ÷ 0.3̄ = 24 -/
theorem eight_divided_by_point_three_repeating : 8 / (1/3 : ℝ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_divided_by_point_three_repeating_l828_82869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_bil_june_earnings_l828_82868

/-- The percentage of Mrs. Bil's earnings compared to the family's total income in February -/
noncomputable def february_percentage : ℝ := 70

/-- The percentage increase in Mrs. Bil's earnings from May to June -/
noncomputable def june_increase : ℝ := 10

/-- The family's total income, assumed to be constant -/
noncomputable def total_income : ℝ := 100

/-- Mrs. Bil's earnings in February -/
noncomputable def february_earnings : ℝ := february_percentage * total_income / 100

/-- Mrs. Bil's earnings in May (assumed to be the same as February) -/
noncomputable def may_earnings : ℝ := february_earnings

/-- Mrs. Bil's earnings in June -/
noncomputable def june_earnings : ℝ := may_earnings * (1 + june_increase / 100)

/-- The percentage of Mrs. Bil's earnings in June compared to the family's total income -/
noncomputable def june_percentage : ℝ := june_earnings / total_income * 100

theorem mrs_bil_june_earnings :
  ∃ ε > 0, abs (june_percentage - 77) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_bil_june_earnings_l828_82868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l828_82857

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 1/a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a > 0) :
  -- Part 1: Solution set when a = 2
  (∀ x : ℝ, f 2 x > 3 ↔ x < -11/4 ∨ x > 1/4) ∧
  -- Part 2: Lower bound for f(m) + f(-1/m)
  (∀ m : ℝ, f a m + f a (-1/m) ≥ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l828_82857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_code_to_number_sequence_l828_82829

-- Define the mapping of symbols to numbers
def symbolToNumber (s : Char) : Nat :=
  match s with
  | '⟦' => 2
  | '⊥' => 3
  | '¬' => 4
  | '⊓' => 8
  | '□' => 5
  | _ => 0

-- Define the code as a string
def code : String := "⟦⊥¬⊓□"

-- Theorem to prove
theorem code_to_number_sequence :
  (code.data.map symbolToNumber) = [2, 3, 4, 8, 5] := by
  sorry

-- Define Δ as zero
def Δ : Nat := 0

#check code_to_number_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_code_to_number_sequence_l828_82829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l828_82845

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * (x - Real.pi / 2) + Real.pi / 3)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l828_82845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_distance_of_farthest_point_l828_82848

noncomputable def points : List (ℝ × ℝ) := [(2, 3), (4, 1), (5, -3), (7, 0), (-3, -5)]

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem farthest_point (p : ℝ × ℝ) (h : p ∈ points) :
  distance_from_origin p ≤ distance_from_origin (7, 0) := by
  sorry

theorem distance_of_farthest_point :
  distance_from_origin (7, 0) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_distance_of_farthest_point_l828_82848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l828_82825

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - Real.sin (2 * x)

theorem min_translation_for_symmetry (m : ℝ) :
  (m > 0) →
  (∀ x, f (x + m) = -f (-x - m)) →
  (∀ k, k > 0 → k < m → ¬(∀ x, f (x + k) = -f (-x - k))) →
  m = Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l828_82825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_proof_l828_82863

def digits : List Nat := [5, 9, 2]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ ∃ (a b c : Nat), List.Perm [a, b, c] digits ∧ n = 100 * a + 10 * b + c

def max_difference : Nat :=
  952 - 259

theorem greatest_difference_proof :
  ∀ (m n : Nat), is_valid_number m → is_valid_number n →
    m - n ≤ max_difference ∧
    (m - n = max_difference → m = 952 ∧ n = 259) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_proof_l828_82863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_is_one_third_l828_82843

/-- Parabola C: y² = 2px (p > 0) with focus F at distance 2 from directrix -/
structure Parabola where
  p : ℝ
  focus_distance : ℝ
  h_p_pos : p > 0
  h_focus_distance : focus_distance = 2

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Point Q satisfying PQ = 9QF -/
structure PointQ (C : Parabola) (P : PointOnParabola C) where
  x : ℝ
  y : ℝ
  h_relation : (P.x - x)^2 + (P.y - y)^2 = 81 * ((x - C.p)^2 + y^2)

/-- Slope of line OQ -/
noncomputable def slope_OQ (C : Parabola) (P : PointOnParabola C) (Q : PointQ C P) : ℝ :=
  Q.y / Q.x

/-- Theorem: Maximum slope of OQ is 1/3 -/
theorem max_slope_OQ_is_one_third (C : Parabola) :
  ∃ (P : PointOnParabola C) (Q : PointQ C P),
    ∀ (P' : PointOnParabola C) (Q' : PointQ C P'),
      slope_OQ C P Q ≤ 1/3 ∧ slope_OQ C P' Q' ≤ slope_OQ C P Q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_is_one_third_l828_82843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_co_complementary_equal_angles_l828_82806

-- Define the concept of co-complementary angles
def coComplementary (α β : Real) : Prop := α + β = Real.pi / 2

-- State the theorem
theorem inverse_co_complementary_equal_angles (α β : Real) :
  (∃ γ δ : Real, coComplementary α γ ∧ coComplementary β δ ∧ γ = δ) → α = β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_co_complementary_equal_angles_l828_82806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l828_82841

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line
def line (x y : ℝ) : ℝ := x - y - 4

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |line x y| / Real.sqrt 2

-- Theorem statement
theorem min_distance_to_line : 
  ∀ x > 0, ∃ y, y = curve x → 
    ∀ z > 0, ∀ w, w = curve z → 
      distance_to_line x y ≤ distance_to_line z w ∧
      ∃ p, distance_to_line x y = p * Real.sqrt 2 ∧ p = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l828_82841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l828_82894

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x / ((x - a) * (x + 1))

theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l828_82894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_attraction_city_A_prob_two_attractions_same_city_l828_82860

/-- Represents a city with attractions -/
structure City where
  name : String
  attractions : Finset String

/-- Represents the problem setup -/
structure TourismProblem where
  cityA : City
  cityB : City

/-- The specific instance of the problem -/
def problem : TourismProblem :=
  { cityA := { name := "A", attractions := {"A", "B"} }
    cityB := { name := "B", attractions := {"C", "D", "E"} } }

/-- The total number of attractions -/
def totalAttractions (p : TourismProblem) : Nat :=
  p.cityA.attractions.card + p.cityB.attractions.card

/-- Theorem for the probability of selecting exactly 1 attraction in City A -/
theorem prob_one_attraction_city_A (p : TourismProblem) :
  (p.cityA.attractions.card : Rat) / (totalAttractions p) = 2/5 := by
  sorry

/-- Theorem for the probability of selecting exactly 2 attractions in the same city -/
theorem prob_two_attractions_same_city (p : TourismProblem) :
  let total_combinations := (totalAttractions p).choose 2
  let same_city_combinations := p.cityA.attractions.card.choose 2 + p.cityB.attractions.card.choose 2
  (same_city_combinations : Rat) / total_combinations = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_attraction_city_A_prob_two_attractions_same_city_l828_82860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l828_82837

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 7768) :
  Int.gcd (4 * b^2 + 55 * b + 120) (3 * b + 12) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l828_82837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_congruent_non_similar_triangles_l828_82821

-- Define a triangle as a triple of natural numbers
def Triangle := Nat × Nat × Nat

-- Define a set of triangles with sides less than 8
def ValidTriangles : Set Triangle :=
  {t : Triangle | t.1 < 8 ∧ t.2.1 < 8 ∧ t.2.2 < 8}

-- Define a predicate for triangle inequality
def SatisfiesTriangleInequality (t : Triangle) : Prop :=
  t.1 + t.2.1 > t.2.2 ∧ t.1 + t.2.2 > t.2.1 ∧ t.2.1 + t.2.2 > t.1

-- Define a predicate for non-congruence
def NonCongruent (t1 t2 : Triangle) : Prop :=
  t1 ≠ t2

-- Define a predicate for non-similarity
def NonSimilar (t1 t2 : Triangle) : Prop :=
  ¬∃ (k : ℚ), k * t1.1 = t2.1 ∧ k * t1.2.1 = t2.2.1 ∧ k * t1.2.2 = t2.2.2

-- Define the set of valid, non-congruent, non-similar triangles
def ValidNonCongruentNonSimilarTriangles : Set Triangle :=
  {t ∈ ValidTriangles | SatisfiesTriangleInequality t ∧
    ∀ t' ∈ ValidTriangles, t ≠ t' → NonCongruent t t' ∧ NonSimilar t t'}

-- Theorem statement
theorem max_non_congruent_non_similar_triangles :
  ∃ (n : Nat), n = 15 ∧ ∃ (f : Fin n → Triangle), Function.Injective f ∧ (∀ i, f i ∈ ValidNonCongruentNonSimilarTriangles) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_congruent_non_similar_triangles_l828_82821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_iff_m_eq_neg_three_l828_82897

/-- A function is quadratic if it can be written in the form ax^2 + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given function parameterized by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 - m) * x^(m^2 - 7) - x + 1

theorem quadratic_iff_m_eq_neg_three :
  ∀ m : ℝ, IsQuadratic (f m) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_iff_m_eq_neg_three_l828_82897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l828_82861

/-- The constant term in the expansion of (√x + 1/(3x))^6, where the fourth term has the maximum binomial coefficient -/
theorem constant_term_expansion (x : ℝ) : 
  let expansion := (Real.sqrt x + 1 / (3 * x)) ^ 6
  let fourth_term_max := ∀ k, k ≠ 3 → Nat.choose 6 3 ≥ Nat.choose 6 k
  let constant_term := (Finset.range 7).sum (λ k ↦ 
    Nat.choose 6 k * (1/3)^k * x^((6-k)/2 - k/2))
  fourth_term_max → constant_term = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l828_82861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l828_82830

-- Define the function f(x) = x - e^x
noncomputable def f (x : ℝ) : ℝ := x - Real.exp x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 1 2 ∧ 
  (∀ x ∈ Set.Icc 1 2, f x ≤ f c) ∧
  f c = 1 - Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_in_interval_l828_82830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_m_range_l828_82896

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + m) / x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x + 2

theorem g_is_odd (m : ℝ) : ∀ x, x ≠ 0 → g m (-x) = -(g m x) := by sorry

theorem m_range (m : ℝ) :
  (m > -1/8 ∧ m < 1) ↔
  (∀ x ∈ Set.Icc (1/4 : ℝ) 1, f m x > 2*m - 2) ∧
  (∀ ε > 0, ∃ m' : ℝ, (|m' - (-1/8)| < ε ∨ |m' - 1| < ε) ∧
    ∃ x ∈ Set.Icc (1/4 : ℝ) 1, f m' x ≤ 2*m' - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_m_range_l828_82896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l828_82814

/-- A triangular pyramid with specific angle and edge length properties -/
structure TriangularPyramid where
  a : ℝ  -- Length of lateral edges
  apex_angle_1 : ℝ  -- First plane angle at apex
  apex_angle_2 : ℝ  -- Second plane angle at apex
  apex_angle_3 : ℝ  -- Third plane angle at apex
  h_positive : 0 < a
  h_angle_1 : apex_angle_1 = π / 4
  h_angle_2 : apex_angle_2 = π / 4
  h_angle_3 : apex_angle_3 = π / 3

/-- The volume of the triangular pyramid -/
noncomputable def pyramidVolume (p : TriangularPyramid) : ℝ := p.a^3 / 12

/-- Theorem: The volume of the specified triangular pyramid is a³/12 -/
theorem triangular_pyramid_volume (p : TriangularPyramid) :
  pyramidVolume p = p.a^3 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l828_82814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_value_l828_82813

/-- The correlation coefficient R² for the relationship between height and weight of female students -/
def correlation_coefficient : ℝ := 0.64

/-- The proportion of weight variation explained by height -/
def height_explanation : ℝ := 0.64

/-- The proportion of weight variation contributed by random error -/
def random_error_contribution : ℝ := 0.36

/-- The effect of height on weight is much greater than the effect of random error -/
axiom height_effect_greater : height_explanation > random_error_contribution

/-- The sum of height explanation and random error contribution equals 1 -/
axiom total_variation : height_explanation + random_error_contribution = 1

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) : Prop := abs (x - y) < 0.01

notation:50 x " ≈ " y => approx_equal x y

theorem correlation_coefficient_value : 
  correlation_coefficient ≈ height_explanation := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_value_l828_82813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_family_size_after_leaving_l828_82818

/-- Represents a family in Indira Nagar -/
structure Family where
  name : Char
  size : Nat
deriving Repr, DecidableEq

/-- The set of families in Indira Nagar -/
def indira_nagar_families : List Family :=
  [⟨'a', 7⟩, ⟨'b', 8⟩, ⟨'c', 10⟩, ⟨'d', 13⟩, ⟨'e', 6⟩, ⟨'f', 10⟩]

/-- The number of members that left each family -/
def members_left : Nat := 1

/-- Calculates the new size of a family after members leave -/
def new_family_size (f : Family) : Nat :=
  f.size - members_left

/-- Sum of new family sizes -/
def sum_new_sizes : Nat :=
  (indira_nagar_families.map new_family_size).sum

/-- Number of families -/
def num_families : Nat :=
  indira_nagar_families.length

/-- Theorem: The average number of members in each family after 1 member leaves is 8 -/
theorem average_family_size_after_leaving :
  sum_new_sizes / num_families = 8 := by
  sorry

#eval sum_new_sizes
#eval num_families
#eval sum_new_sizes / num_families

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_family_size_after_leaving_l828_82818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l828_82801

structure Plane where
  points : Type
  lines : Type
  circles : Type
  on_line : points → lines → Prop
  on_circle : points → circles → Prop
  midpoint : points → points → points → Prop
  perpendicular_foot : points → points → lines → Prop  -- Changed order of arguments
  tangent : lines → circles → Prop
  parallel : lines → lines → Prop
  distinct : points → points → Prop

theorem parallel_lines (π : Plane) 
  (A B M T X Y : π.points) (ω : π.circles) (AB BT AT XY : π.lines) :
  π.distinct A B →
  π.midpoint M A B →
  π.on_circle A ω →
  π.on_circle M ω →
  π.on_circle T ω →
  π.tangent BT ω →
  π.on_line X AB →
  π.on_line B AB →
  π.on_line A AB →
  π.distinct X B →
  π.perpendicular_foot Y A BT →
  (∃ (TB TX : π.points), π.on_line TB BT ∧ π.on_line TX BT ∧ TB = TX) →
  π.on_line T AT →
  π.on_line A AT →
  π.on_line X XY →
  π.on_line Y XY →
  π.parallel AT XY :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l828_82801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_in_first_quadrant_l828_82881

theorem conjugate_in_first_quadrant (z : ℂ) : 
  (Complex.I + 1) * z = Complex.abs (Complex.I + Real.sqrt 3) → 
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_in_first_quadrant_l828_82881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_x_2_l828_82849

noncomputable def g (x : ℝ) : ℝ := |⌊x + 2⌋| + |⌈x - 2⌉| - 3

theorem g_symmetric_about_x_2 : ∀ x : ℝ, g x = g (2 - x) := by
  intro x
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_x_2_l828_82849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l828_82823

theorem tan_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = Real.sqrt 2 / 3)
  (h2 : 0 < α) (h3 : α < Real.pi) :
  Real.tan (α - Real.pi/4) = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l828_82823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l828_82872

theorem trigonometric_identities (α β : Real) 
  (h1 : Real.sin α + Real.cos α = 3 * Real.sqrt 5 / 5)
  (h2 : α ∈ Set.Ioo 0 (π/4))
  (h3 : Real.sin (β - π/4) = 3/5)
  (h4 : β ∈ Set.Ioo (π/4) (π/2)) :
  Real.sin (2*α) = 4/5 ∧ 
  Real.tan (2*α) = 4/3 ∧ 
  Real.cos (α + 2*β) = -11 * Real.sqrt 5 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l828_82872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l828_82853

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def IsValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the specific conditions of our triangle
def OurTriangle (t : Triangle) : Prop :=
  IsValidTriangle t ∧
  t.B = Real.pi / 3 ∧  -- 60 degrees
  t.c = 4

-- Define the midpoint condition
def MidpointCondition (t : Triangle) (AM BM : Real) : Prop :=
  AM / BM = Real.sqrt 3

-- Theorem for the first part
theorem part1 (t : Triangle) (AM BM : Real) :
  OurTriangle t → MidpointCondition t AM BM → t.a = 4 := by sorry

-- Theorem for the second part
theorem part2 (t : Triangle) :
  OurTriangle t → t.b = 6 → 
  (1/2 * t.a * t.c * Real.sin t.B) = 2 * Real.sqrt 3 + 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l828_82853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_for_guarantee_l828_82888

-- Define the predicates before using them in the theorem
def is_two_boys {n : ℕ} (photo : Fin n) : Prop := sorry
def is_two_girls {n : ℕ} (photo : Fin n) : Prop := sorry
def same_children {n : ℕ} (photo1 photo2 : Fin n) : Prop := sorry

theorem min_photos_for_guarantee (n_girls : ℕ) (n_boys : ℕ) : 
  n_girls = 4 → n_boys = 8 → 
  ∃ (min_photos : ℕ), 
    min_photos = n_girls * n_boys + 1 ∧ 
    (∀ (photos : ℕ), photos ≥ min_photos → 
      (∃ (photo : Fin photos), is_two_boys photo ∨ is_two_girls photo) ∨
      (∃ (photo1 photo2 : Fin photos), photo1 ≠ photo2 ∧ same_children photo1 photo2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_for_guarantee_l828_82888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_sin_functions_l828_82867

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin x

-- Define the volume of revolution
noncomputable def volume_of_revolution (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  Real.pi * ∫ x in a..b, (f x)^2 - (g x)^2

-- State the theorem
theorem volume_of_sin_functions :
  volume_of_revolution 0 Real.pi f g = 4 * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_sin_functions_l828_82867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l828_82887

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * Real.sin (ω * x + Real.pi / 3) * Real.cos (ω * x) - Real.sqrt 3 / 2

theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_sym : ∀ x : ℝ, f ω x = f ω (Real.pi / 6 - x)) : 
  ω = 1 ∧ 
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) ∧ 
  f ω (Real.pi / 3) = 0 ∧
  (∀ x y : ℝ, -5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ Real.pi / 12 → f ω x < f ω y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l828_82887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_additional_conditions_l828_82858

-- Define the polynomials
def p₁ (x : ℝ) := x^3 - 12*x^2 + 47*x - 60
def p₂ (x : ℝ) := x^3 - 19*x^2 + 118*x - 240
def p₃ (x : ℝ) := x^4 - 18*x^3 + 119*x^2 - 342*x + 360

-- Define the roots
def roots₁ : List ℝ := [3, 4, 5]
def roots₂ : List ℝ := [5, 6, 8]
def roots₃ : List ℝ := [3, 4, 5, 6]

-- Theorem statement
theorem polynomial_roots :
  (∀ x ∈ roots₁, p₁ x = 0) ∧
  (∀ x ∈ roots₂, p₂ x = 0) ∧
  (∀ x ∈ roots₃, p₃ x = 0) :=
by
  sorry

-- Additional conditions
theorem additional_conditions :
  (∃ x₂ x₃, x₂ ∈ roots₁ ∧ x₃ ∈ roots₁ ∧ x₂ + x₃ = 9) ∧
  (∃ x₂ x₃, x₂ ∈ roots₂ ∧ x₃ ∈ roots₂ ∧ x₂ * x₃ = 48) ∧
  (∃ x₁ x₂ x₃ x₄, x₁ ∈ roots₃ ∧ x₂ ∈ roots₃ ∧ x₃ ∈ roots₃ ∧ x₄ ∈ roots₃ ∧ x₁ + x₂ = 7 ∧ x₃ * x₄ = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_additional_conditions_l828_82858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_divisor_property_l828_82865

def is_valid_n (n : ℕ) : Prop :=
  ∃ (p : ℕ), p > 1 ∧ Nat.Prime p ∧ n = 101 * p^2

theorem largest_n_with_divisor_property :
  ∀ n : ℕ, is_valid_n n → n ≤ 101^3 :=
by
  sorry

#check largest_n_with_divisor_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_divisor_property_l828_82865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vet_fees_dogs_value_verify_solution_l828_82808

/-- Vet fees for dogs during a pet adoption event --/
def vet_fees_dogs : ℕ := 15

/-- Vet fees for cats during a pet adoption event --/
def vet_fees_cats : ℕ := 13

/-- Number of families adopting dogs --/
def num_dogs_adopted : ℕ := 8

/-- Number of families adopting cats --/
def num_cats_adopted : ℕ := 3

/-- Fraction of fees donated back to the shelter --/
def donation_fraction : ℚ := 1/3

/-- Amount donated back to the shelter --/
def donation_amount : ℕ := 53

/-- Theorem stating that the vet fees for dogs are $15 --/
theorem vet_fees_dogs_value :
  vet_fees_dogs = 15 :=
by
  -- The proof goes here
  sorry

/-- Theorem verifying the solution --/
theorem verify_solution :
  (num_dogs_adopted * vet_fees_dogs + num_cats_adopted * vet_fees_cats : ℚ) * donation_fraction = donation_amount :=
by
  -- The verification proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vet_fees_dogs_value_verify_solution_l828_82808
