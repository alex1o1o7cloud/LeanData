import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_reads_three_books_l168_16849

/-- Calculates the number of books Billy can read during his weekend free time. -/
def books_read (free_time_per_day : ℕ) (weekend_days : ℕ) (video_game_percentage : ℚ)
               (reading_speed : ℕ) (pages_per_book : ℕ) : ℕ :=
  let total_free_time := free_time_per_day * weekend_days
  let reading_time := (total_free_time : ℚ) * (1 - video_game_percentage)
  let pages_read := (reading_time * reading_speed).floor
  (pages_read / pages_per_book).toNat

/-- Theorem stating that Billy reads 3 books during his weekend free time. -/
theorem billy_reads_three_books :
  books_read 8 2 (3/4) 60 80 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_reads_three_books_l168_16849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l168_16889

-- Define the triangle XYZ
structure RightTriangle where
  xy : ℝ
  xz : ℝ
  yz : ℝ
  is_right : xy^2 + xz^2 = yz^2

-- Define the inscribed circle
structure InscribedCircle (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ

-- Define the construction of PQ and RS
structure Construction (t : RightTriangle) where
  pq_height : ℝ
  rs_width : ℝ
  tangent_to_c1 : pq_height = rs_width

-- Define the inscribed circles C₂ and C₃
noncomputable def C2 (t : RightTriangle) (c : Construction t) : InscribedCircle t :=
  { center := (c.rs_width, t.xz - c.pq_height),
    radius := c.rs_width * c.pq_height / t.xz }

noncomputable def C3 (t : RightTriangle) (c : Construction t) : InscribedCircle t :=
  { center := (t.xy - c.rs_width, c.pq_height),
    radius := c.rs_width * (t.yz - c.pq_height) / t.xy }

-- The main theorem
theorem distance_between_centers
  (t : RightTriangle)
  (h1 : t.xy = 60)
  (h2 : t.xz = 80)
  (h3 : t.yz = 100)
  (c : Construction t) :
  let c2 := C2 t c
  let c3 := C3 t c
  (c2.center.1 - c3.center.1)^2 + (c2.center.2 - c3.center.2)^2 = 16050 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l168_16889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_for_30_degree_asymptote_l168_16863

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.b / h.a

theorem hyperbola_eccentricity_for_30_degree_asymptote (h : Hyperbola) 
  (h_slope : asymptote_slope h = Real.tan (π / 6)) : 
  eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_for_30_degree_asymptote_l168_16863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_range_of_a_l168_16807

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * log x
def g (a x : ℝ) : ℝ := a - x^2

-- Define the domain
def domain : Set ℝ := Set.Icc (-exp 1) (-1 / exp 1)

-- Theorem statement
theorem symmetry_implies_range_of_a :
  (∀ x ∈ domain, ∃ (a : ℝ), ∀ y ∈ domain, 
    f (-y) = g a y → a ∈ Set.Icc 1 (exp 2 - 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_range_of_a_l168_16807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_c_coordinates_l168_16885

/-- Triangle ABC with vertices A(4, 2), B(0, 5), and C on the line 3x - y = 0 -/
structure Triangle where
  C : ℝ × ℝ
  h1 : C.2 = 3 * C.1

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_c_coordinates (t : Triangle) :
  triangleArea (4, 2) (0, 5) t.C = 10 →
  t.C = (8/3, 8) := by
  sorry

#check triangle_c_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_c_coordinates_l168_16885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_symmetry_l168_16891

/-- A function with period 4π and phase shift π/6 -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

/-- The translated function -/
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - Real.pi / 3)

theorem translated_symmetry (ω : ℝ) (h₁ : ω > 0) (h₂ : ∀ x, f ω (x + 4 * Real.pi) = f ω x) :
  ∀ x, g ω (-x) = -g ω x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_symmetry_l168_16891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_players_same_flips_l168_16845

/-- The probability of getting heads on a fair coin flip -/
noncomputable def prob_heads : ℝ := 1 / 2

/-- The probability that a single player gets their first head on the n-th flip -/
noncomputable def prob_first_head (n : ℕ) : ℝ := prob_heads^n

/-- The probability that all three players get their first head on the n-th flip -/
noncomputable def prob_all_three (n : ℕ) : ℝ := (prob_first_head n)^3

/-- The sum of probabilities for all possible numbers of flips -/
noncomputable def total_probability : ℝ := ∑' n, prob_all_three n

/-- The theorem stating that the probability of all three players stopping after the same number of flips is 1/7 -/
theorem three_players_same_flips : total_probability = 1 / 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_players_same_flips_l168_16845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_directly_proportional_equation_c_is_directly_proportional_l168_16808

/-- Definition of direct proportion -/
noncomputable def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function representing the equation (2/3)x = y -/
noncomputable def f (x : ℝ) : ℝ := (2/3) * x

/-- Theorem stating that f represents a direct proportion -/
theorem f_is_directly_proportional : is_directly_proportional f := by
  use (2/3)
  intro x
  rfl

/-- Proof that the equation (2/3)x = y represents a direct proportion -/
theorem equation_c_is_directly_proportional :
  ∃ k : ℝ, ∀ x y : ℝ, y = (2/3) * x → y = k * x := by
  use (2/3)
  intro x y h
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_directly_proportional_equation_c_is_directly_proportional_l168_16808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l168_16886

theorem cosine_inequality (y : Real) :
  y ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos ((Real.pi / 2) + y) ≥ Real.cos (Real.pi / 2) - Real.cos y) ↔
  (y ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Icc (5 * Real.pi / 4) (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l168_16886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l168_16851

noncomputable section

structure RightTriangle (X Y Z : ℝ × ℝ) : Prop where
  is_right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

def circle_intersects (X Y Z W : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), 
    (center.1 - Y.1)^2 + (center.2 - Y.2)^2 = (center.1 - Z.1)^2 + (center.2 - Z.2)^2 ∧
    (W.1 - center.1)^2 + (W.2 - center.2)^2 = ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) / 4

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem triangle_segment_length 
  (X Y Z W : ℝ × ℝ) 
  (h1 : RightTriangle X Y Z) 
  (h2 : circle_intersects X Y Z W) 
  (h3 : distance X W = 3) 
  (h4 : distance Y W = 9) : 
  distance Z W = 27 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l168_16851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_unionized_men_l168_16888

theorem percentage_of_unionized_men (total_employees : ℕ) 
  (h1 : total_employees > 0)
  (percent_men : ℚ) 
  (h2 : percent_men = 52 / 100)
  (percent_unionized : ℚ) 
  (h3 : percent_unionized = 60 / 100)
  (percent_women_non_union : ℚ) 
  (h4 : percent_women_non_union = 75 / 100) :
  let num_men := (percent_men * total_employees : ℚ).floor
  let num_unionized := (percent_unionized * total_employees : ℚ).floor
  let num_non_union := total_employees - num_unionized
  let num_men_non_union := ((1 - percent_women_non_union) * num_non_union : ℚ).floor
  let num_men_unionized := num_men - num_men_non_union
  (num_men_unionized : ℚ) / (num_unionized : ℚ) = 70 / 100 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_unionized_men_l168_16888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_class_average_l168_16818

/-- Calculates the overall average marks per student given the number of students and mean marks for each section -/
def overall_average (students : List ℕ) (marks : List ℚ) : ℚ :=
  (List.sum (List.zipWith (λ s m => (s : ℚ) * m) students marks)) / (List.sum students : ℚ)

/-- Theorem stating that the overall average for the given problem is approximately 52.26 -/
theorem chemistry_class_average : 
  let students : List ℕ := [40, 35, 45, 42]
  let marks : List ℚ := [50, 60, 55, 45]
  abs (overall_average students marks - 52.26) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_class_average_l168_16818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ultra_deficient_numbers_l168_16826

/-- Sum of squares of all divisors of n -/
def g (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n) (Finset.range (n + 1))).sum (fun d => d * d)

/-- A positive integer n is ultra-deficient if g(g(n)) = n^2 + 2 -/
def is_ultra_deficient (n : ℕ) : Prop :=
  n > 0 ∧ g (g n) = n * n + 2

theorem no_ultra_deficient_numbers : ¬ ∃ n : ℕ, is_ultra_deficient n := by
  sorry

#eval g 1
#eval g 2
#eval g (g 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ultra_deficient_numbers_l168_16826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l168_16884

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The given triangle satisfies the conditions of the problem -/
def special_triangle (t : Triangle) : Prop :=
  2 * t.b = 2 * t.a * Real.cos t.C + t.c ∧
  t.a = 10 ∧
  (1/2) * t.b * t.c * Real.sin t.A = 8 * Real.sqrt 3

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : Real := t.a + t.b + t.c

/-- The theorem to be proved -/
theorem special_triangle_properties (t : Triangle) (h : special_triangle t) :
  t.A = π/3 ∧ perimeter t = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l168_16884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_sin_2x_value_l168_16872

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (cos (2 * x)) / (sin (x + π / 4))

-- Theorem for the domain of f
theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ x ∉ {y | ∃ k : ℤ, y = k * π - π / 4} :=
sorry

-- Theorem for the value of sin(2x) when f(x) = 4/3
theorem sin_2x_value (x : ℝ) (h : f x = 4 / 3) :
  sin (2 * x) = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_sin_2x_value_l168_16872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finiteness_equiv_finite_steps_l168_16850

/-- A step in an algorithm. -/
structure Step where
  -- You can add fields here if needed
  mk :: -- Empty structure for now

/-- An algorithm is a sequence of steps. -/
def Algorithm : Type := List Step

/-- The property of finiteness for an algorithm. -/
def is_finite (a : Algorithm) : Prop := 
  ∃ n : Nat, a.length = n

/-- The property that an algorithm has a finite number of steps. -/
def has_finite_steps (a : Algorithm) : Prop := 
  ∃ n : Nat, a.length = n

/-- Theorem stating that the finiteness of an algorithm is equivalent to having a finite number of steps. -/
theorem finiteness_equiv_finite_steps (a : Algorithm) : 
  is_finite a ↔ has_finite_steps a := by
  -- The proof is trivial since the definitions are identical
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finiteness_equiv_finite_steps_l168_16850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l168_16868

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.sqrt 2 + (Real.sqrt 3) / 2 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + (Real.sqrt 3) / 2 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - (Real.sqrt 3) / 2 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - (Real.sqrt 3) / 2 + Real.sqrt 6

-- State the theorem
theorem sum_of_reciprocals_squared : 
  (1/a + 1/b + 1/c + 1/d)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l168_16868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_2pi_l168_16855

theorem cos_alpha_minus_2pi (α : ℝ) : 
  Real.sin (Real.pi + α) = 3/5 → 
  (0 < -α ∧ -α < Real.pi/2) → 
  Real.cos (α - 2*Real.pi) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_2pi_l168_16855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_max_area_point_l168_16861

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + 2*t, 1/2 - t)

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (2*Real.cos θ, Real.sin θ)

-- Define the intersection points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 1)

-- Theorem for the distance between A and B
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 5 := by sorry

-- Define the point P on curve C
def P (θ : ℝ) : ℝ × ℝ := curve_C θ

-- Helper function to calculate the area of a triangle
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem for the coordinates of P that maximize the area of triangle ABP
theorem max_area_point : 
  ∃ θ : ℝ, P θ = (-Real.sqrt 2, -Real.sqrt 2 / 2) ∧ 
  ∀ φ : ℝ, area_triangle A B (P φ) ≤ area_triangle A B (P θ) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_max_area_point_l168_16861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_unfolded_surface_l168_16802

/-- Given a cone whose side surface unfolds into a semicircle with radius 1 cm,
    prove that its volume is (√3 * π) / 24 cm³. -/
theorem cone_volume_from_unfolded_surface (r : ℝ) (h : r = 1) : 
  (1/3) * π * (r/2)^2 * Real.sqrt (r^2 - (r/2)^2) = (Real.sqrt 3 * π) / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_unfolded_surface_l168_16802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l168_16833

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem m_range (m : ℝ) :
  (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) ↔ m ≤ -1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l168_16833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_a_range_l168_16869

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * log (a * x - a) + a

/-- The theorem statement -/
theorem f_positive_implies_a_range (a : ℝ) (h1 : a > 0) :
  (∀ x > 1, f a x > 0) → 0 < a ∧ a < exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_a_range_l168_16869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_six_l168_16835

open BigOperators

def S (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / (k.succ * (k.succ + 1))

theorem sum_equals_six (n : ℕ) : S n * S (n + 1) = 3/4 → n = 6 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_six_l168_16835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_digit_assignment_l168_16827

/-- Represents the different animal masks --/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Assigns a digit to each mask --/
def digit_assignment : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- Checks if a number is a perfect square --/
def is_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- The main theorem stating the correctness of the digit assignment --/
theorem correct_digit_assignment :
  (∀ m₁ m₂ : Mask, m₁ ≠ m₂ → digit_assignment m₁ ≠ digit_assignment m₂) ∧
  (is_perfect_square ((digit_assignment Mask.mouse) * 10 + digit_assignment Mask.elephant)) ∧
  (digit_assignment Mask.mouse * digit_assignment Mask.mouse % 10 = digit_assignment Mask.elephant) ∧
  (digit_assignment Mask.pig * digit_assignment Mask.pig % 10 = digit_assignment Mask.mouse) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_digit_assignment_l168_16827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_division_iff_trapezoid_l168_16882

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Define the properties of a convex quadrilateral
  is_convex : Bool

/-- A trapezoid -/
structure Trapezoid where
  -- Define the properties of a trapezoid
  base : ConvexQuadrilateral
  has_parallel_sides : Bool

/-- Two similar quadrilaterals -/
structure SimilarQuadrilaterals where
  -- Define the properties of similar quadrilaterals
  are_similar : Bool

/-- A line that divides a quadrilateral -/
structure DividingLine where
  -- Define the properties of a dividing line
  divides_quadrilateral : Bool

/-- Main theorem: A convex quadrilateral can be divided into two similar quadrilaterals
    by a straight line if and only if it is a trapezoid -/
theorem convex_quadrilateral_division_iff_trapezoid
  (q : ConvexQuadrilateral) :
  (∃ (l : DividingLine) (s : SimilarQuadrilaterals),
    l.divides_quadrilateral ∧ s.are_similar) ↔
  (∃ (t : Trapezoid), t.base = q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_division_iff_trapezoid_l168_16882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l168_16813

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 5 * x - 2
noncomputable def g (x : ℝ) : ℝ := 3 * x + 7
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x - 33) / 15

-- Theorem statement
theorem h_inverse_correct : 
  (∀ x, h (h_inv x) = x) ∧ (∀ x, h_inv (h x) = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l168_16813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_theorem_l168_16824

/-- For any even positive integer n, the least positive integer k₀ such that
    k₀ = f(x)(x+1)^m + g(x)(x^n + 1) for some polynomials f(x) and g(x) with
    integer coefficients, is equal to 2^m, where n = 2^r * m and m is odd. -/
theorem least_k_theorem (n : ℕ) (hn : Even n) (hn_pos : 0 < n) :
  ∃ (r m : ℕ) (hm : Odd m),
    n = 2^r * m ∧
    (∃ (k₀ : ℕ) (hk₀_pos : 0 < k₀),
      (∀ (k : ℕ) (hk_pos : 0 < k),
        (∃ (f g : Polynomial ℤ) (x : ℤ), k = f.eval x * (x + 1)^m + g.eval x * (x^n + 1)) →
        k₀ ≤ k) ∧
      k₀ = 2^m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_theorem_l168_16824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l168_16817

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem f_increasing_on_interval (ω φ : ℝ) 
  (h1 : φ > 0) 
  (h2 : -Real.pi < φ ∧ φ < 0) 
  (h3 : ∀ x, f ω φ (x + Real.pi) = f ω φ x) 
  (h4 : f ω φ (Real.pi/3) = 1) :
  StrictMonoOn (f ω φ) (Set.Icc (-Real.pi/6) (Real.pi/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l168_16817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l168_16819

noncomputable def x (t : ℝ) : ℝ := 2 * Real.exp (3 * t) - 4 * Real.exp (-t) - 2 * Real.exp t + 5 - 3 * t

noncomputable def y (t : ℝ) : ℝ := -4 + 6 * t + 4 * Real.exp (-t) + 2 * Real.exp (3 * t)

theorem cauchy_problem_solution :
  (∀ t, deriv x t = x t + 2 * y t - 9 * t) ∧
  (∀ t, deriv y t = 2 * x t + y t + 4 * Real.exp t) ∧
  x 0 = 1 ∧
  y 0 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l168_16819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l168_16840

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1999 * Real.sqrt (x + 1999 * Real.sqrt (x + 1999 * Real.sqrt (x + 1999 * Real.sqrt (2000 * x)))))

theorem equation_solutions :
  ∀ x : ℝ, f x = x ↔ x = 0 ∨ x = 2000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l168_16840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_intersection_l168_16805

-- Define the line l
def line (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the circle C
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the shortest chord length
noncomputable def shortest_chord_length : ℝ := 4 * Real.sqrt 5

-- Theorem statement
theorem shortest_chord_intersection :
  ∀ (m : ℝ), ∃ (x y : ℝ),
    line m x y ∧ circle_eq x y ∧
    (∀ (x' y' : ℝ), line m x' y' ∧ circle_eq x' y' →
      Real.sqrt ((x - x')^2 + (y - y')^2) ≥ shortest_chord_length) :=
by
  sorry

#check shortest_chord_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_intersection_l168_16805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_negative_domain_l168_16897

-- Define the function f for x > 0
noncomputable def f_positive (x : ℝ) : ℝ := x - Real.log (abs x)

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = x + Real.log (abs x) := by
  sorry

#check odd_function_negative_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_negative_domain_l168_16897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l168_16847

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then -2 * x - 3 else 2^(-x)

theorem f_composition_value : f (f (-3)) = 1/8 := by
  -- Evaluate f(-3)
  have h1 : f (-3) = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(3)
  have h2 : f 3 = 1/8 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc f (f (-3))
    = f 3 := by rw [h1]
    _ = 1/8 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l168_16847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l168_16859

/-- A line passing through (2, 1) that intersects the x and y axes -/
structure IntersectingLine where
  slope : ℝ

/-- The x-intercept of the line -/
noncomputable def x_intercept (l : IntersectingLine) : ℝ := 2 - 1 / l.slope

/-- The y-intercept of the line -/
noncomputable def y_intercept (l : IntersectingLine) : ℝ := 1 - 2 * l.slope

/-- The area of the triangle formed by the line and the axes -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ :=
  (1/2) * |x_intercept l| * |y_intercept l|

/-- The set of lines that satisfy the conditions -/
def satisfying_lines : Set IntersectingLine :=
  {l | triangle_area l = 4}

theorem exactly_three_lines :
  ∃ (l₁ l₂ l₃ : IntersectingLine),
    l₁ ∈ satisfying_lines ∧
    l₂ ∈ satisfying_lines ∧
    l₃ ∈ satisfying_lines ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ ∧
    ∀ (l : IntersectingLine), l ∈ satisfying_lines → l = l₁ ∨ l = l₂ ∨ l = l₃ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l168_16859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_pi_l168_16829

/-- A square pyramid with inscribed and circumscribed spheres, and additional tangent spheres -/
structure PyramidWithSpheres where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- The inscribed sphere has radius 1/3 of the circumscribed sphere -/
  inscribed_radius : r = R / 3

/-- The probability that a random point in the circumscribed sphere is inside one of the six smaller spheres -/
noncomputable def probability_in_smaller_spheres (p : PyramidWithSpheres) : ℝ :=
  6 * (4 * Real.pi * p.r^3 / 3) / (4 * Real.pi * p.R^3 / 3)

/-- Theorem stating the probability is 6/(36π) -/
theorem probability_is_one_sixth_pi (p : PyramidWithSpheres) :
  probability_in_smaller_spheres p = 6 / (36 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_pi_l168_16829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonals_l168_16874

-- Define Vertex as a type synonym for ℕ
def Vertex := ℕ

-- Define the function for diagonals from a single vertex
def diagonals_from_vertex (n : ℕ) (v : Vertex) : ℕ :=
  n - 3

-- Define the function for total diagonals
def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (∀ v : Vertex, diagonals_from_vertex n v = n - 3) ∧
  (total_diagonals n = n * (n - 3) / 2) :=
by
  constructor
  · intro v
    rfl  -- reflexivity proves the equality
  · rfl  -- reflexivity proves the equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonals_l168_16874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_change_l168_16801

/-- Represents the price changes of gasoline over 5 months -/
noncomputable def price_changes (P : ℝ) (y : ℝ) : ℝ :=
  P * (1 + 0.15) * (1 - 0.1) * (1 + 0.3) * (1 - y / 100) * (1 + 0.1)

/-- Theorem stating that when y is approximately 32, the final price equals the initial price -/
theorem gasoline_price_change (P : ℝ) (h : P > 0) :
  ∃ y : ℝ, abs (y - 32) < 1 ∧ price_changes P y = P := by
  sorry

#check gasoline_price_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_change_l168_16801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_properties_l168_16894

/-- Represents the problem-solving machine -/
structure Machine where
  /-- Number of problems -/
  n : ℕ
  /-- Time taken to solve the first problem -/
  b : ℝ
  /-- Common ratio of the geometric progression (time reduction factor) -/
  q : ℝ

/-- The sum of a geometric series -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem stating the properties of the machine -/
theorem machine_properties (m : Machine) : 
  (m.q > 0 ∧ m.q < 1) →
  (geometricSum (m.b * m.q) m.q (m.n - 1) = 63.5) →
  (geometricSum m.b m.q (m.n - 1) = 127) →
  (geometricSum (m.b * m.q^2) m.q (m.n - 4) = 30) →
  (m.n = 8 ∧ m.b + geometricSum (m.b * m.q) m.q (m.n - 1) = 127.5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_properties_l168_16894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l168_16822

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 12)

-- State the theorem
theorem max_value_of_g :
  ∃ (max_val : ℝ), 
    (∀ x ∈ Set.Icc (-Real.pi/8) (3*Real.pi/8), g x ≤ max_val) ∧
    (∃ x ∈ Set.Icc (-Real.pi/8) (3*Real.pi/8), g x = max_val) ∧
    (max_val = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l168_16822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_is_two_l168_16804

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftLighter : WeighResult
  | RightLighter : WeighResult

/-- Represents a set of parts to be weighed -/
structure PartSet where
  total : Nat
  defective : Nat
  h_defective_count : defective = 1
  h_defective_lighter : defective < total

/-- Represents a weighing strategy -/
def WeighStrategy := PartSet → Nat

/-- The minimum number of weighings required to find the defective part -/
def minWeighings : WeighStrategy := fun ps => 2

/-- Theorem stating that the minimum number of weighings is 2 -/
theorem min_weighings_is_two (ps : PartSet) (h : ps.total = 9) : 
  minWeighings ps = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_is_two_l168_16804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_survey_results_l168_16887

/-- Represents the proportion of people who buy rice on the nth purchase. -/
def a : ℕ → ℝ := sorry

/-- Represents the proportion of people who buy noodles on the nth purchase. -/
def b : ℕ → ℝ := sorry

/-- The initial condition for rice buyers. -/
axiom a_init : a 1 = 1/2

/-- The initial condition for noodle buyers. -/
axiom b_init : b 1 = 1/2

/-- The recursion relation for rice buyers. -/
axiom a_rec : ∀ n : ℕ, n ≥ 2 → a n = 4/5 * a (n-1) + 3/10 * b (n-1)

/-- The recursion relation for noodle buyers. -/
axiom b_rec : ∀ n : ℕ, n ≥ 2 → b n = 7/10 * b (n-1) + 1/5 * a (n-1)

theorem cafeteria_survey_results :
  (∀ n : ℕ, n ≥ 1 → a n + b n = 1) ∧
  (∀ n : ℕ, n ≥ 2 → a n = 1/2 * a (n-1) + 3/10) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 3/5 - 1/10 * (1/2)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2/5 + 1/10 * (1/2)^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_survey_results_l168_16887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solution_range_l168_16820

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function values at specific points
def func_values (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧
  quadratic a b c (-2) = m - 4.5 ∧
  quadratic a b c (-1) = m - 2 ∧
  quadratic a b c 0 = m - 0.5 ∧
  quadratic a b c 1 = m ∧
  quadratic a b c 2 = m - 0.5 ∧
  quadratic a b c 3 = m - 2 ∧
  quadratic a b c 4 = m - 4.5

-- Define the range of m
def m_range (m : ℝ) : Prop := 1 < m ∧ m < 1.5

-- Theorem statement
theorem positive_solution_range (a b c m : ℝ) (x₁ : ℝ) :
  a ≠ 0 →
  func_values a b c m →
  m_range m →
  quadratic a b c x₁ = 0 →
  x₁ > 0 →
  2 < x₁ ∧ x₁ < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_solution_range_l168_16820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l168_16890

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (b : ℝ) : Set ℝ := {b, 2}

theorem problem_solution (a b : ℝ) :
  (∃ (x : ℝ), U a = {2, 3, x}) ∧
  (A b)ᶜ = {5} →
  ((a = 2 ∧ b = 3) ∨ (a = -4 ∧ b = 3)) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l168_16890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l168_16842

/-- The function r(x) = 1 / (1-x)^3 -/
noncomputable def r (x : ℝ) : ℝ := 1 / (1 - x)^3

/-- The range of r(x) is (0, ∞) -/
theorem range_of_r : Set.range r = Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l168_16842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_primes_equation_l168_16809

theorem quadruple_primes_equation (p q r : ℕ) (n : ℕ+) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p^2 = q^2 + r^(n : ℕ) →
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_primes_equation_l168_16809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_ratios_l168_16848

/-- Represents a regular hexagon with side length a -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents the solid formed by rotating the hexagon around its diagonal -/
noncomputable def DiagonalRotation (h : RegularHexagon) : ℝ × ℝ :=
  let volume := h.side_length^3 * Real.pi
  let surface_area := 3 * Real.sqrt 3 * h.side_length^2 * Real.pi
  (volume, surface_area)

/-- Represents the solid formed by rotating the hexagon around its midline -/
noncomputable def MidlineRotation (h : RegularHexagon) : ℝ × ℝ :=
  let volume := 7 * h.side_length^3 * Real.pi * Real.sqrt 3 / 6
  let surface_area := 11 * Real.pi * h.side_length^2 / 2
  (volume, surface_area)

/-- Theorem stating the volume and surface area ratios of the two rotations -/
theorem rotation_ratios (h : RegularHexagon) :
  let (v1, s1) := DiagonalRotation h
  let (v2, s2) := MidlineRotation h
  v1 / v2 = 2 * Real.sqrt 3 / 7 ∧ s1 / s2 = 6 * Real.sqrt 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_ratios_l168_16848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l168_16857

-- Define the initial radii and the thickness of the layer lost
def initialRadii : List ℝ := [4, 6, 8]
def lostLayerThickness : ℝ := 0.5

-- Define a function to calculate the effective radius
noncomputable def effectiveRadius (r : ℝ) : ℝ := r - lostLayerThickness

-- Define a function to calculate the volume of a sphere
noncomputable def sphereVolume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Theorem statement
theorem snowman_volume :
  let effectiveRadii := initialRadii.map effectiveRadius
  let volumes := effectiveRadii.map sphereVolume
  volumes.sum = (841.5/3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l168_16857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_divisible_by_25_l168_16875

theorem four_digit_numbers_divisible_by_25 : 
  Finset.card (Finset.filter (λ n : ℕ => 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 25) (Finset.range 10000)) = 90 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_divisible_by_25_l168_16875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_in_interval_l168_16873

open Set Real Function

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonic on an interval if it's either increasing or decreasing -/
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x)

theorem two_zeros_in_interval
  (f : ℝ → ℝ) (a : ℝ)
  (h_cont : ContinuousOn f (Icc (-a) a))
  (h_even : IsEven f)
  (h_pos : 0 < a)
  (h_zero_cross : f 0 * f a < 0)
  (h_monotonic : IsMonotonicOn f 0 a) :
  ∃! (x y : ℝ), x ∈ Ioo (-a) 0 ∧ y ∈ Ioo 0 a ∧ f x = 0 ∧ f y = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_in_interval_l168_16873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_length_is_400_meters_l168_16836

/-- The side length of the large cube in centimeters -/
noncomputable def large_cube_side : ℝ := 100

/-- The side length of each small cube in centimeters -/
noncomputable def small_cube_side : ℝ := 5

/-- The number of small cubes that fit along one edge of the large cube -/
noncomputable def cubes_per_edge : ℝ := large_cube_side / small_cube_side

/-- The total number of small cubes that fit in the large cube -/
noncomputable def total_small_cubes : ℝ := cubes_per_edge ^ 3

/-- The length of the row formed by all small cubes in centimeters -/
noncomputable def row_length_cm : ℝ := total_small_cubes * small_cube_side

/-- The length of the row formed by all small cubes in meters -/
noncomputable def row_length_m : ℝ := row_length_cm / 100

theorem row_length_is_400_meters : row_length_m = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_length_is_400_meters_l168_16836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_mistakes_l168_16844

/-- Represents the number of mistakes Sanya makes in a word -/
def mistakes (word : String) : ℕ := sorry

/-- The words given in the problem -/
def tetrahedron : String := "TETRAHEDRON"
def dodecahedron : String := "DODECAHEDRON"
def icosahedron : String := "ICOSAHEDRON"
def octahedron : String := "OCTAHEDRON"

/-- Axioms based on the given conditions -/
axiom tetrahedron_mistakes : mistakes tetrahedron = 5
axiom dodecahedron_mistakes : mistakes dodecahedron = 6
axiom icosahedron_mistakes : mistakes icosahedron = 7

/-- The theorem to be proved -/
theorem octahedron_mistakes : mistakes octahedron = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_mistakes_l168_16844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_ride_time_l168_16828

/-- Represents the time it takes Sara to ride the moving walkway without walking -/
noncomputable def ride_time (stationary_time walking_time : ℝ) : ℝ :=
  (stationary_time * walking_time) / (stationary_time - walking_time)

/-- Theorem stating that Sara's ride time on the moving walkway is 45 seconds -/
theorem sara_ride_time :
  ride_time 90 30 = 45 := by
  -- Unfold the definition of ride_time
  unfold ride_time
  -- Simplify the expression
  simp [mul_div_assoc]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_ride_time_l168_16828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_m_l168_16832

noncomputable def hyperbola_equation (x y m : ℝ) : Prop := x^2 - y^2 / m = 1

noncomputable def focus_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus_m (m : ℝ) :
  (∀ x y, hyperbola_equation x y m) →
  focus_distance 1 (Real.sqrt m) = 3 →
  m = 8 := by
  intro h1 h2
  -- The proof steps would go here
  sorry

#check hyperbola_focus_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_m_l168_16832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangements_l168_16879

/-- Represents an octahedron with labeled vertices -/
structure Octahedron :=
  (vertices : Fin 6 → ℕ)
  (is_permutation : Function.Bijective vertices)

/-- The sum of integers from 1 to 6 -/
def sum_1_to_6 : ℕ := 21

/-- The number of faces in an octahedron -/
def num_faces : ℕ := 8

/-- The number of faces each vertex belongs to -/
def faces_per_vertex : ℕ := 4

/-- The face sum for a given face -/
def face_sum (o : Octahedron) (face : Fin 8) : ℕ :=
  sorry -- Definition of face sum

/-- Checks if all face sums are equal -/
def all_face_sums_equal (o : Octahedron) : Prop :=
  ∃ (s : ℕ), ∀ (face : Fin 8), (face_sum o face) = s

/-- Main theorem: There are no valid arrangements -/
theorem no_valid_arrangements :
  ¬ ∃ (o : Octahedron), all_face_sums_equal o :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangements_l168_16879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eight_fib_between_l168_16825

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Sum of eight consecutive Fibonacci numbers starting from the (k+1)th term -/
def sum_eight_fib (k : ℕ) : ℕ :=
  (List.range 8).foldl (fun acc i => acc + fib (k + i + 1)) 0

/-- Theorem: The sum of eight consecutive Fibonacci numbers is between the next two Fibonacci numbers -/
theorem sum_eight_fib_between (k : ℕ) :
  fib (k + 9) < sum_eight_fib k ∧ sum_eight_fib k < fib (k + 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eight_fib_between_l168_16825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_classification_l168_16898

-- Define the type of statement
inductive StatementType
  | Universal
  | Existential

-- Define a function to classify statements (this is a placeholder)
def StatementClassification (p : Prop) : StatementType :=
  sorry -- In a real implementation, this would determine if p is universal or existential

-- Define the statements using basic types and propositions
def statement1 : Prop := ∀ (p q r : ℝ), p = q → r = 0

def statement2 : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

def statement3 : Prop := ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ z ≠ x

def statement4 : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = b

-- Theorem to prove
theorem statement_classification :
  (StatementType.Universal = StatementClassification statement1) ∧
  (StatementType.Universal = StatementClassification statement2) ∧
  (StatementType.Existential = StatementClassification statement3) ∧
  (StatementType.Existential = StatementClassification statement4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_classification_l168_16898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_perfect_squares_l168_16852

def sequence_condition (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 2, a (n + 1) = 3 * a n - 3 * a (n - 1) + a (n - 2)) ∧
  (2 * a 1 = a 0 + a 2 - 2) ∧
  (∀ m : ℕ, ∃ k : ℕ, ∀ i : ℕ, i < m → ∃ j : ℕ, a (k + i) = j * j)

theorem sequence_is_perfect_squares (a : ℕ → ℕ) (h : sequence_condition a) :
  ∃ l : ℤ, ∀ n : ℕ, a n = (n + l) * (n + l) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_perfect_squares_l168_16852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_theorem_l168_16883

/-- Represents the dimensions and surface area of a cuboid -/
structure Cuboid where
  length : ℝ
  breadth : ℝ
  surfaceArea : ℝ

/-- Calculates the height of a cuboid given its length, breadth, and surface area -/
noncomputable def calculateHeight (c : Cuboid) : ℝ :=
  (c.surfaceArea - 2 * c.length * c.breadth) / (2 * (c.length + c.breadth))

/-- Theorem stating that for a cuboid with given dimensions, its height is approximately 22.22 cm -/
theorem cuboid_height_theorem (c : Cuboid) 
    (h1 : c.length = 8)
    (h2 : c.breadth = 10)
    (h3 : c.surfaceArea = 960) :
    ∃ ε > 0, |calculateHeight c - 22.22| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_theorem_l168_16883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elongation_rate_improvement_l168_16853

/-- Elongation rates for process A -/
def x : Fin 10 → ℝ := ![545, 533, 551, 522, 575, 544, 541, 568, 596, 548]

/-- Elongation rates for process B -/
def y : Fin 10 → ℝ := ![536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

/-- Difference in elongation rates -/
def z (i : Fin 10) : ℝ := x i - y i

/-- Sample mean of z -/
noncomputable def z_bar : ℝ := (Finset.univ.sum z) / 10

/-- Sample variance of z -/
noncomputable def s_squared : ℝ := (Finset.univ.sum (fun i => (z i - z_bar) ^ 2)) / 10

/-- Criterion for significant improvement -/
noncomputable def significant_improvement : Prop := z_bar ≥ 2 * Real.sqrt (s_squared / 10)

theorem elongation_rate_improvement : significant_improvement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elongation_rate_improvement_l168_16853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_problem_l168_16877

theorem binomial_expansion_problem (n : ℕ) 
  (hn : n > 0)
  (h : (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) = 56) :
  (n = 10) ∧ 
  (∃ (r : ℕ), r = 8 ∧ (Nat.choose n r) * (1/2)^r = 45/256) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_problem_l168_16877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_inequality_l168_16831

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x - 1)) / (x - 2)

-- State the theorem
theorem max_k_for_inequality :
  ∃ (k : ℕ), k = 3 ∧
  (∀ x > 2, f x > (k : ℝ) / (x - 1)) ∧
  (∀ k' > k, ∃ x > 2, f x ≤ (k' : ℝ) / (x - 1)) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_inequality_l168_16831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l168_16806

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
noncomputable def given_condition (t : Triangle) : Prop :=
  (3 * (t.a - t.b)) / t.c = (3 * Real.sin t.C - 2 * Real.sin t.B) / (Real.sin t.A + Real.sin t.B)

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1 / 2) * t.b * t.c * Real.sin t.A

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : given_condition t) 
  (h2 : triangle_area t = 2 * Real.sqrt 2) : 
  Real.cos t.A = 1 / 3 ∧ 
  ∃ (AD : ℝ), AD ≤ 2 ∧ 
    ∀ (AD' : ℝ), (∃ (D : ℝ), D > 0 ∧ D < t.c ∧ 
      AD' = (t.b * D / t.c + t.c - D) * Real.sin (t.A / 2)) → 
    AD' ≤ AD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l168_16806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l168_16895

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the angle of the line
noncomputable def line_angle : ℝ := Real.pi/6

-- Define the slope of the line
noncomputable def line_slope : ℝ := Real.tan line_angle

-- Define the line passing through the left focus
def line (x y : ℝ) : Prop :=
  y = line_slope * (x - left_focus.1) + left_focus.2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | hyperbola p.1 p.2 ∧ line p.1 p.2}

-- State the theorem
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l168_16895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_limit_condition_l168_16864

/-- Given an infinite geometric sequence with first term a₁ and common ratio q -/
def GeometricSequence (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁
  else a₁ * (1 - q^n) / (1 - q)

/-- The limit of the sum of an infinite geometric sequence -/
noncomputable def GeometricSumLimit (a₁ q : ℝ) : ℝ := a₁ / (1 - q)

theorem geometric_sum_limit_condition (a₁ q : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |GeometricSum a₁ q n - 1| < ε) →
  (a₁ + q = 1) ∧
  ¬(a₁ + q = 1 → ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |GeometricSum a₁ q n - 1| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_limit_condition_l168_16864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tiling_with_odd_domino_l168_16871

def domino (b : ℕ) (d : ℕ × ℕ) : Set (ℕ × ℕ) :=
  {(x, y) | (x = d.1 ∧ d.1 ≤ y ∧ y < d.1 + b) ∨ (y = d.2 ∧ d.2 ≤ x ∧ x < d.2 + b)}

theorem no_tiling_with_odd_domino 
  (b n m : ℕ) 
  (h_b_odd : Odd b) 
  (h_n_odd : Odd n) 
  (h_m_not_div4 : ¬ 4 ∣ m) : 
  (¬ ∃ (tiling : Set (ℕ × ℕ)), 
    (∀ (x y : ℕ), x < 2*b ∧ y < n → (∃ (d : ℕ × ℕ), d ∈ tiling ∧ (x, y) ∈ domino b d)) ∧
    (∀ (d1 d2 : ℕ × ℕ), d1 ∈ tiling → d2 ∈ tiling → d1 ≠ d2 → (domino b d1) ∩ (domino b d2) = ∅)) ∧
  (¬ ∃ (tiling : Set (ℕ × ℕ)), 
    (∀ (x y : ℕ), x < m ∧ y < n → (∃ (d : ℕ × ℕ), d ∈ tiling ∧ (x, y) ∈ domino b d)) ∧
    (∀ (d1 d2 : ℕ × ℕ), d1 ∈ tiling → d2 ∈ tiling → d1 ≠ d2 → (domino b d1) ∩ (domino b d2) = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tiling_with_odd_domino_l168_16871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_range_student_count_exact_l168_16803

/-- The number of 8th-grade students -/
def x : ℕ := sorry

/-- The retail price of one flag -/
def retail_price : ℚ := sorry

/-- The wholesale price of one flag -/
def wholesale_price : ℚ := sorry

/-- The total cost of flags at retail price -/
def retail_cost : ℚ := 240

/-- The total cost of flags at wholesale price -/
def wholesale_cost : ℚ := 240

/-- The number of additional flags to reach wholesale pricing -/
def additional_flags : ℕ := 60

/-- The threshold for wholesale pricing -/
def wholesale_threshold : ℕ := 300

/-- Conditions of the problem -/
axiom retail_condition : x * retail_price = retail_cost
axiom wholesale_condition : (x + additional_flags) * wholesale_price = wholesale_cost
axiom price_transition : x < wholesale_threshold ∧ x + additional_flags > wholesale_threshold
axiom price_equality : 360 * wholesale_price = 300 * retail_price

/-- Theorem: The number of 8th-grade students is between 240 and 300, inclusive -/
theorem student_count_range : 240 < x ∧ x ≤ 300 := by
  sorry

/-- Theorem: The number of 8th-grade students is 300 -/
theorem student_count_exact : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_range_student_count_exact_l168_16803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_equals_two_l168_16800

theorem trig_sum_equals_two :
  Real.sin (70 * π / 180) ^ 2 + Real.tan (225 * π / 180) + Real.sin (20 * π / 180) ^ 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_equals_two_l168_16800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_solution_l168_16876

theorem linear_equation_solution (m : ℤ) (x : ℚ) : 
  (x^(m.natAbs) - m * x + 1 = 0 ∧ 
   ∃ a b, a ≠ 0 ∧ x^(m.natAbs) - m * x + 1 = a * x + b) → 
  x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_solution_l168_16876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_buckets_count_l168_16814

/-- The volume of a large bucket in liters -/
def large_bucket_volume : ℚ := 4

/-- The volume of a small bucket in liters -/
def small_bucket_volume : ℚ := (large_bucket_volume - 3) / 2

/-- The number of large buckets used -/
def num_large_buckets : ℕ := 5

/-- The total volume of the tank in liters -/
def tank_volume : ℚ := 63

/-- The number of small buckets needed to fill the tank -/
noncomputable def num_small_buckets : ℕ := 
  (((tank_volume - ↑num_large_buckets * large_bucket_volume) / small_bucket_volume).floor).toNat

theorem small_buckets_count : num_small_buckets = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_buckets_count_l168_16814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_price_is_15_l168_16816

/-- Represents the price of a clock in Stella's antique shop. -/
def clock_price : ℕ → ℕ := sorry

/-- The number of dolls in Stella's antique shop. -/
def num_dolls : ℕ := 3

/-- The number of clocks in Stella's antique shop. -/
def num_clocks : ℕ := 2

/-- The number of glasses in Stella's antique shop. -/
def num_glasses : ℕ := 5

/-- The price of each doll in dollars. -/
def doll_price : ℕ := 5

/-- The price of each glass in dollars. -/
def glass_price : ℕ := 4

/-- The total cost Stella spent to buy all items in dollars. -/
def total_cost : ℕ := 40

/-- The profit Stella makes when selling all items in dollars. -/
def profit : ℕ := 25

theorem clock_price_is_15 : clock_price num_clocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_price_is_15_l168_16816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l168_16878

/-- Given vectors a = (-2,3) and b = (0,4), prove that the projection of a onto b is (0,3) -/
theorem projection_vector (a b : ℝ × ℝ) : 
  a = (-2, 3) → b = (0, 4) → 
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l168_16878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l168_16881

theorem triangle_side_length (R Q S : ℝ × ℝ) (cosR : ℝ) :
  cosR = 3/5 →
  Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 5 →
  (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0 →
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l168_16881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l168_16837

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 4 else Real.exp (x * Real.log 2)

theorem range_of_a (a : ℝ) (h : f a ≥ 2) : 
  a ∈ Set.Icc (-2) 0 ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l168_16837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_movement_l168_16866

/-- The number of degrees in a full rotation of a clock's second hand -/
noncomputable def full_rotation : ℝ := 360

/-- The number of minutes for a full rotation of a clock's second hand -/
noncomputable def minutes_per_rotation : ℝ := 60

/-- The number of degrees the second hand moves per minute -/
noncomputable def degrees_per_minute : ℝ := full_rotation / minutes_per_rotation

theorem second_hand_movement :
  degrees_per_minute = 6 := by
  -- Unfold the definitions
  unfold degrees_per_minute full_rotation minutes_per_rotation
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_movement_l168_16866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_cubic_function_l168_16858

theorem minimum_value_cubic_function (y : ℝ) (h : y > 0) :
  ∃ (min : ℝ), min = 14 ∧ (6 * y^3 + 8 * y^(-(3 : ℤ)) ≥ min) ∧ ∃ (y₀ : ℝ), y₀ > 0 ∧ 6 * y₀^3 + 8 * y₀^(-(3 : ℤ)) = min := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_cubic_function_l168_16858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l168_16896

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 25 ∧ 
  arr.toFinset = Finset.range 25 ∧
  ∀ i, (List.take 5 (List.drop i (arr ++ arr))).sum % 5 ∈ ({1, 4} : Finset Nat)

theorem no_valid_arrangement : ¬ ∃ arr : List Nat, is_valid_arrangement arr := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l168_16896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l168_16815

noncomputable def f (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l168_16815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_y_coord_l168_16821

/-- Given an equilateral triangle with two vertices at (0,7) and (10,7),
    and the third vertex in the first quadrant, 
    the y-coordinate of the third vertex is 7 + 5√3. -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (A B C : ℝ × ℝ),
  A = (0, 7) →
  B = (10, 7) →
  C.1 ≥ 0 ∧ C.2 ≥ 0 →  -- First quadrant condition
  ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - A‖ →  -- Equilateral condition
  C.2 = 7 + 5 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_y_coord_l168_16821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l168_16830

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = 2 * a) : 
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l168_16830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_shift_l168_16812

noncomputable def sample_mean (X : Finset ℝ) : ℝ := (X.sum id) / X.card

noncomputable def sample_variance (X : Finset ℝ) : ℝ :=
  (X.sum (λ x => (x - sample_mean X) ^ 2)) / X.card

theorem sample_shift (X : Finset ℝ) (c : ℝ) :
  sample_mean X = 10 →
  sample_variance X = 2 →
  sample_mean (X.image (λ x => x + c)) = sample_mean X + c ∧
  sample_variance (X.image (λ x => x + c)) = sample_variance X :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_shift_l168_16812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l168_16860

-- Define the sum of first n terms of an arithmetic sequence
def S (n : ℕ) : ℝ := sorry

-- Define the common difference of the arithmetic sequence
def d : ℝ := sorry

-- State the theorem
theorem arithmetic_sequence_ratio :
  S 8 = -3 * S 4 ∧ S 4 ≠ 0 →
  S 4 / S 12 = -1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l168_16860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_moves_on_circle_l168_16834

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the motion of points A and B
noncomputable def moveAlongCircle (p : Point) (c : Circle) (ω : ℝ) (t : ℝ) : Point :=
  { x := c.center.x + c.radius * Real.cos (ω * t)
  , y := c.center.y + c.radius * Real.sin (ω * t) }

-- Define an equilateral triangle
def isEquilateralTriangle (a b c : Point) : Prop :=
  let d₁ := ((a.x - b.x)^2 + (a.y - b.y)^2).sqrt
  let d₂ := ((b.x - c.x)^2 + (b.y - c.y)^2).sqrt
  let d₃ := ((c.x - a.x)^2 + (c.y - a.y)^2).sqrt
  d₁ = d₂ ∧ d₂ = d₃

-- Theorem statement
theorem point_c_moves_on_circle 
  (o₁ o₂ : Circle) 
  (ω : ℝ) 
  (h_ω : ω ≠ 0) 
  : ∃ (o₃ : Circle),
    ∀ (t : ℝ),
      let a := moveAlongCircle (Point.mk o₁.center.x o₁.center.y) o₁ ω t
      let b := moveAlongCircle (Point.mk o₂.center.x o₂.center.y) o₂ ω t
      let c := Point.mk (2*b.x - a.x) (2*b.y - a.y)  -- C is symmetric to A with respect to B
      isEquilateralTriangle a b c →
      ∃ (θ : ℝ), c = moveAlongCircle (Point.mk o₃.center.x o₃.center.y) o₃ ω t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_moves_on_circle_l168_16834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_fourth_term_is_one_hundredth_l168_16843

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define the base case for 0
  | 1 => 1  -- Keep the original base case for 1
  | (n + 2) => sequence_a (n + 1) / (3 * sequence_a (n + 1) + 1)

theorem thirty_fourth_term_is_one_hundredth :
  sequence_a 34 = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_fourth_term_is_one_hundredth_l168_16843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_mile_travel_revolutions_l168_16823

/-- The number of revolutions for a wheel with given diameter to travel a specific distance -/
noncomputable def wheel_revolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  (distance * 5280) / (Real.pi * diameter)

/-- Theorem stating the number of revolutions for a 10-foot diameter wheel to travel 2 miles -/
theorem two_mile_travel_revolutions :
  wheel_revolutions 10 2 = 1056 / Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_mile_travel_revolutions_l168_16823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_lines_l168_16839

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - x + (1/2) * x^2 - (1/3) * a * x^3

def g (a : ℝ) (x : ℝ) : ℝ := Real.log x + x - a * x^2

def x₂ (a : ℝ) : ℝ := (1 + Real.sqrt (1 + 8*a)) / (4*a)

theorem f_tangent_lines (a : ℝ) :
  (∀ x, x > 0 → MonotoneOn (g a) (Set.Ioo 0 (x₂ a))) ∧
  (∀ x, x > x₂ a → ¬MonotoneOn (g a) (Set.Ioi (x₂ a))) ∧
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0) ↔
  0 < a ∧ a < 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_tangent_lines_l168_16839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coby_travel_time_l168_16810

/-- Represents the travel details of Coby's road trip --/
structure RoadTrip where
  distance_washington_idaho : ℚ
  distance_idaho_nevada : ℚ
  speed_washington_idaho : ℚ
  speed_idaho_nevada : ℚ

/-- Calculates the total travel time for the road trip --/
def total_travel_time (trip : RoadTrip) : ℚ :=
  trip.distance_washington_idaho / trip.speed_washington_idaho +
  trip.distance_idaho_nevada / trip.speed_idaho_nevada

/-- Theorem stating that Coby's total travel time is 19 hours --/
theorem coby_travel_time :
  let trip : RoadTrip := {
    distance_washington_idaho := 640,
    distance_idaho_nevada := 550,
    speed_washington_idaho := 80,
    speed_idaho_nevada := 50
  }
  total_travel_time trip = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coby_travel_time_l168_16810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l168_16867

-- Define the functions f and g
noncomputable def f (x : ℝ) := 1 / Real.sqrt (1 - x)
noncomputable def g (x : ℝ) := Real.log (1 + x)

-- Define the domains M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | x > -1}

-- Theorem statement
theorem domain_intersection :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l168_16867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_3000_l168_16899

noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem smallest_n_exceeding_3000 :
  let a₁ := (3 : ℝ)
  let q := (4 : ℝ)
  ∀ n : ℕ, n < 6 → geometric_sum a₁ q n ≤ 3000 ∧
  geometric_sum a₁ q 6 > 3000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_3000_l168_16899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l168_16838

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The sine function with phase shift φ -/
noncomputable def SineWithPhase (φ : ℝ) : ℝ → ℝ :=
  fun x ↦ Real.sin (x + φ)

/-- φ = π/2 is a sufficient condition for SineWithPhase φ to be even -/
theorem sufficient_condition (φ : ℝ) (h : φ = Real.pi / 2) :
  IsEven (SineWithPhase φ) := by
  sorry

/-- φ = π/2 is not a necessary condition for SineWithPhase φ to be even -/
theorem not_necessary_condition :
  ∃ φ, φ ≠ Real.pi / 2 ∧ IsEven (SineWithPhase φ) := by
  sorry

/-- Main theorem: φ = π/2 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ φ, φ = Real.pi / 2 → IsEven (SineWithPhase φ)) ∧
  (∃ φ, φ ≠ Real.pi / 2 ∧ IsEven (SineWithPhase φ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l168_16838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_range_l168_16856

/-- Predicate to determine if an equation represents an ellipse -/
def IsEllipse (equation : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, a > 0 ∧ c > 0 ∧ a ≠ c ∧
  ∀ x y : ℝ, equation x y = a * x^2 + b * x * y + c * y^2 + d * x + e * y + f

/-- Given that the equation (m-1)x^2 + (3-m)y^2 = (m-1)(3-m) represents an ellipse,
    prove that the range of m is (1,2) ∪ (2,3). -/
theorem ellipse_m_range (m : ℝ) :
  (∀ x y : ℝ, IsEllipse (fun x y => (m - 1) * x^2 + (3 - m) * y^2 - (m - 1) * (3 - m))) →
  m ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_range_l168_16856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_a_plus_b_equals_45_l168_16893

theorem abs_a_plus_b_equals_45 (a b : ℝ) :
  (∀ x : ℝ, (7 * x - a)^2 = 49 * x^2 - b * x + 9) → |a + b| = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_a_plus_b_equals_45_l168_16893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l168_16865

theorem correct_proposition : 
  (∀ (p q : Prop), p ∧ ¬q → ¬(p ∧ q)) ∧ 
  (∀ x y : ℝ, (xy = 0 → x = 0) ↔ ¬(xy ≠ 0 ∧ x ≠ 0)) ∧ 
  (∃ α : ℝ, Real.sin α = 1/2 ∧ α ≠ π/6) ∧ 
  (∀ x : ℝ, (2 : ℝ)^x > 0) ↔ ¬(∃ x : ℝ, (2 : ℝ)^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l168_16865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_135_plus_i_sin_135_pow_60_l168_16854

/-- DeMoivre's Theorem for complex numbers in degrees -/
axiom deMoivre (θ : ℝ) (n : ℕ) : (Complex.exp (θ * Real.pi / 180 * Complex.I)) ^ n = Complex.exp (n * θ * Real.pi / 180 * Complex.I)

/-- Periodicity of complex exponential function -/
axiom complex_exp_period (z : ℂ) : Complex.exp (z + 2 * Real.pi * Complex.I) = Complex.exp z

theorem cos_135_plus_i_sin_135_pow_60 :
  (Complex.exp (135 * Real.pi / 180 * Complex.I)) ^ 60 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_135_plus_i_sin_135_pow_60_l168_16854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l168_16880

/-- An ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def Ellipse.equation (C : Ellipse) (x y : ℝ) : Prop :=
  x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (C : Ellipse) : ℝ :=
  Real.sqrt (1 - C.b^2 / C.a^2)

/-- A line passing through a point with a given slope -/
structure Line where
  x₀ : ℝ
  y₀ : ℝ
  m : ℝ

/-- The equation of a line -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y - l.y₀ = l.m * (x - l.x₀)

/-- The length of a line segment intersected by an ellipse -/
noncomputable def intersectionLength (C : Ellipse) (l : Line) : ℝ := sorry

theorem ellipse_properties (C : Ellipse) (h_point : C.equation 0 (Real.sqrt 3))
    (h_ecc : C.eccentricity = 1/2) :
    (C.a = 2 ∧ C.b = Real.sqrt 3) ∧
    (let l : Line := { x₀ := 1, y₀ := 0, m := 1 }
     intersectionLength C l = 24 * Real.sqrt 2 / 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l168_16880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_negative_terms_l168_16862

noncomputable def sequenceA (n : ℕ) : ℚ :=
  if n = 0 then 1 else 2 - 3 ^ n

theorem nine_negative_terms :
  let a := sequenceA
  (a 0 = 1) ∧
  (∀ n : ℕ, a (n + 1) - 3 * a n + 4 = 0) →
  (Finset.filter (fun i => a i < 0) (Finset.range 10)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_negative_terms_l168_16862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_min_value_achieved_l168_16841

open Real

theorem min_value_trigonometric_expression :
  ∀ x : ℝ, (sin x)^8 + (cos x)^8 + 3 ≥ 14/31 * ((sin x)^6 + (cos x)^6 + 3) := by
  sorry

theorem min_value_achieved :
  ∃ x : ℝ, (sin x)^8 + (cos x)^8 + 3 = 14/31 * ((sin x)^6 + (cos x)^6 + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_min_value_achieved_l168_16841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l168_16846

/-- Represents the time it takes Silvia to ride down an escalator -/
noncomputable def ride_time (walk_time_off : ℝ) (walk_time_on : ℝ) : ℝ :=
  (walk_time_off * walk_time_on) / (walk_time_off - walk_time_on)

theorem escalator_ride_time :
  let walk_time_off := (80 : ℝ)
  let walk_time_on := (28 : ℝ)
  let standing_time := ride_time walk_time_off walk_time_on
  ∃ ε > 0, |standing_time - 43| < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l168_16846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_and_lines_l168_16811

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the different relation for planes and lines
variable (different : Plane → Plane → Prop)
variable (different_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_planes_and_lines
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : different α β ∧ different α γ ∧ different β γ)
  (h_diff_lines : different_line m n)
  (h1 : perp_plane_line α n)
  (h2 : perp_plane_line β n)
  (h3 : perp_plane_line α m) :
  perp_plane_line β m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_and_lines_l168_16811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_same_color_selection_l168_16892

/-- Represents a collection of shoes -/
structure ShoeCollection where
  pairs : ℕ
  colors : ℕ
  shoesPerColor : ℕ

/-- Represents a selection of shoes -/
def Selection := ℕ

/-- The probability of selecting a specific combination of shoes -/
noncomputable def probability (collection : ShoeCollection) (selection : ℕ) : ℚ :=
  sorry

theorem impossible_same_color_selection 
  (collection : ShoeCollection) 
  (selection : ℕ) : 
  collection.pairs = 10 → 
  collection.colors = 10 → 
  collection.shoesPerColor = 2 → 
  selection = 3 → 
  probability collection selection = 0 := by
  sorry

#check impossible_same_color_selection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_same_color_selection_l168_16892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l168_16870

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem hyperbola_distance_theorem (x y : ℝ) :
  hyperbola x y →
  distance x y 5 0 = 15 →
  (distance x y (-5) 0 = 7 ∨ distance x y (-5) 0 = 23) :=
by
  sorry

#check hyperbola_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l168_16870
