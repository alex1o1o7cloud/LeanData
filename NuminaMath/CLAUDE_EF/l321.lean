import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_triangle_free_20_l321_32133

/-- A graph with no triangles --/
structure TriangleFreeGraph (n : ℕ) where
  edges : Finset (Fin n × Fin n)
  no_triangles : ∀ a b c : Fin n, 
    (⟨a, b⟩ ∈ edges ∧ ⟨b, c⟩ ∈ edges) → ⟨a, c⟩ ∉ edges

/-- The number of edges in a triangle-free graph --/
def edge_count {n : ℕ} (G : TriangleFreeGraph n) : ℕ := G.edges.card

/-- The theorem to be proved --/
theorem max_edges_triangle_free_20 :
  (∃ (G : TriangleFreeGraph 20), edge_count G = 100) ∧
  (∀ (G : TriangleFreeGraph 20), edge_count G ≤ 100) := by
  sorry

#check max_edges_triangle_free_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_triangle_free_20_l321_32133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_count_l321_32129

def sequenceA (n : ℕ) : ℚ :=
  8820 / 3^n

theorem sequence_integer_count :
  ∃ k : ℕ, k = 3 ∧ (∀ n < k, ∃ m : ℕ, sequenceA n = m) ∧
  (∀ n ≥ k, ¬∃ m : ℕ, sequenceA n = m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_count_l321_32129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_time_proof_l321_32100

noncomputable def jumps_per_second : ℝ := 1
noncomputable def situps_per_minute : ℝ := 25
noncomputable def pushups_per_minute : ℝ := 20

noncomputable def total_jumps : ℝ := 200
noncomputable def total_situps : ℝ := 150
noncomputable def total_pushups : ℝ := 100

noncomputable def time_for_jumps : ℝ := total_jumps / (jumps_per_second * 60)
noncomputable def time_for_situps : ℝ := total_situps / situps_per_minute
noncomputable def time_for_pushups : ℝ := total_pushups / pushups_per_minute

noncomputable def total_time : ℝ := time_for_jumps + time_for_situps + time_for_pushups

theorem exercise_time_proof : 
  ∃ ε > 0, |total_time - 14.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exercise_time_proof_l321_32100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_product_PQR_l321_32143

/-- A point on a 2D grid -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the triangle PQR -/
def P : Point := ⟨1, 5⟩
def Q : Point := ⟨5, 5⟩
def R : Point := ⟨1, 1⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Area of a right triangle given two sides -/
noncomputable def rightTriangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- Perimeter of a triangle given its three sides -/
def trianglePerimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- Theorem stating the product of area and perimeter of triangle PQR -/
theorem area_perimeter_product_PQR :
  let pq := distance P Q
  let pr := distance P R
  let qr := distance Q R
  let area := rightTriangleArea pq pr
  let perimeter := trianglePerimeter pq pr qr
  area * perimeter = 64 + 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_product_PQR_l321_32143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_e_inequality_implies_a_range_l321_32132

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

-- Part 1
theorem tangent_line_implies_a_equals_e (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, x > 0 ∧ f a x = x / Real.exp a + Real.exp 2 ∧ 
    (deriv (f a)) x = 1 / Real.exp a) → a = Real.exp 1 :=
by
  sorry

-- Part 2
theorem inequality_implies_a_range (a : ℝ) (h : a > 0) :
  (∀ m n : ℝ, m > 0 → n > 0 → m ≠ n → 
    Real.sqrt (m * n) + (m + n) / 2 > (m - n) / (f a m - f a n)) →
  a ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_e_inequality_implies_a_range_l321_32132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_through_focus_parabola_min_product_perpendicular_parabola_min_product_value_l321_32189

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/2, 1)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the line passing through two points
noncomputable def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define perpendicular lines
def perpendicular (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.1 - p1.1) + (p2.2 - p1.2) * (p3.2 - p1.2) = 0

theorem parabola_segment_length_through_focus
  (A B : ℝ × ℝ)
  (h1 : point_on_parabola A)
  (h2 : point_on_parabola B)
  (h3 : line_slope A B = 2)
  (h4 : ∃ t : ℝ, focus = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)) :
  distance A B = 5/2 := by
  sorry

theorem parabola_min_product_perpendicular
  (A B : ℝ × ℝ)
  (h1 : point_on_parabola A)
  (h2 : point_on_parabola B)
  (h3 : perpendicular (0, 0) A B) :
  ∀ C D : ℝ × ℝ, point_on_parabola C → point_on_parabola D → perpendicular (0, 0) C D →
    distance (0, 0) A * distance (0, 0) B ≤ distance (0, 0) C * distance (0, 0) D := by
  sorry

theorem parabola_min_product_value
  (A B : ℝ × ℝ)
  (h1 : point_on_parabola A)
  (h2 : point_on_parabola B)
  (h3 : perpendicular (0, 0) A B) :
  distance (0, 0) A * distance (0, 0) B = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_through_focus_parabola_min_product_perpendicular_parabola_min_product_value_l321_32189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_500th_term_l321_32185

def our_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2010 ∧ 
  a 2 = 2011 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem sequence_500th_term (a : ℕ → ℕ) (h : our_sequence a) : a 500 = 2177 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_500th_term_l321_32185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l321_32145

/-- Given positive integers a, b, and c where a < b < c, and a system of equations
    with exactly one solution, prove that the minimum value of c is 997. -/
theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
    (h3 : ∃! (x y : ℝ), 3 * x + y = 3000 ∧ 
      y = |x - a.val| + |x - b.val| + |x - c.val| + 10) :
    c ≥ 997 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l321_32145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l321_32114

noncomputable def f (x : ℝ) := 2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  let a := Real.pi / 4
  let b := Real.pi / 2
  (∀ x ∈ Set.Icc a b, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc a b, f x = 3) ∧
  (∃ x ∈ Set.Icc a b, f x = 2) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc a b, |f x - m| < 2) ↔ m ∈ Set.Ioo 1 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l321_32114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l321_32173

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_equation_solution (x : ℝ) :
  x > 0 ∧ x * floor x + 2022 = floor (x^2) ↔
  ∃ k : ℤ, k ≥ 2023 ∧ x = k + 2022 / (k : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l321_32173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l321_32106

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (a - 1) + f (2 * a^2) ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 1/2) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l321_32106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_triangle_l321_32171

theorem dot_product_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 - c^2 = Real.sqrt 3 * a * b →
  a * c * Real.sin B = 2 * Real.sqrt 3 * Real.sin C →
  a * b * Real.cos C = 3 := by
  intros ha hb hc hA hB hC hABC hab hac
  sorry

#check dot_product_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_triangle_l321_32171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choco_pie_divisors_l321_32138

theorem choco_pie_divisors : 
  let n : ℕ := 900
  (Finset.filter (λ d ↦ d ∣ n) (Finset.range (n + 1))).card = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choco_pie_divisors_l321_32138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_left_focus_l321_32112

/-- The coordinates of the left focus of an ellipse -/
noncomputable def left_focus (a b : ℝ) : ℝ × ℝ :=
  let c := (a^2 - b^2).sqrt
  (-c, 0)

/-- Theorem: The coordinates of the left focus of the ellipse x = 4cos(θ), y = 3sin(θ) are (-√7, 0) -/
theorem ellipse_left_focus :
  left_focus 4 3 = (-(7 : ℝ).sqrt, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_left_focus_l321_32112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l321_32125

noncomputable def a : ℝ := (3/5) ^ (2/5)
noncomputable def b : ℝ := (2/5) ^ (3/5)
noncomputable def c : ℝ := (2/5) ^ (2/5)

theorem relationship_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l321_32125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l321_32123

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 2 * x

-- Define the hyperbola
noncomputable def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the condition that a and b are positive
def positive_parameters (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Define the distance from focus to asymptote
noncomputable def distance_focus_to_asymptote (a b : ℝ) : ℝ := Real.sqrt 10 / 5

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : positive_parameters a b) 
  (h2 : distance_focus_to_asymptote a b = Real.sqrt 10 / 5) :
  eccentricity a b = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l321_32123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l321_32130

-- Define the constants
noncomputable def a : ℝ := Real.log 6 / Real.log 2
noncomputable def b : ℝ := Real.log 12 / Real.log 3
noncomputable def c : ℝ := (2 : ℝ) ^ (0.6 : ℝ)

-- State the theorem
theorem ordering_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l321_32130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l321_32150

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Theorem statement
theorem function_analysis (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 9) :
  -- 1. f(x) = 3^x
  (∀ x, f a x = f 3 x) ∧
  -- 2. Range of f(x) is (0, +∞)
  (∀ y, y > 0 → ∃ x, f a x = y) ∧
  -- 3. If (f(x))^2 - 2f(x) + k = 0 has exactly two distinct real roots, then k ∈ (0, 1)
  (∀ k : ℝ, (∃! s : Set ℝ, s.Finite ∧ s.Nonempty ∧ (∃ x y, x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ ∀ z, z ∈ s → z = x ∨ z = y) ∧ 
    ∀ x, x ∈ s ↔ (f a x)^2 - 2*(f a x) + k = 0) → 
    0 < k ∧ k < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l321_32150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l321_32188

theorem expression_evaluation : 
  (2 * Real.sqrt 2) ^ (2/3) * (0.1)⁻¹ - Real.log 2 - Real.log 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l321_32188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l321_32198

open Real

noncomputable def curve (x : ℝ) : ℝ := (2 * exp (x + 1)) / (exp (2 * x) + 1)

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (log x) / x + x - a

theorem a_range_theorem (a : ℝ) :
  (∃ x₀ y₀ : ℝ, curve x₀ = y₀ ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Iic (1 / exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l321_32198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_upper_bound_l321_32159

theorem cos_upper_bound (a : ℝ) : (¬ ∃ x : ℝ, Real.cos x ≥ a) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_upper_bound_l321_32159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l321_32199

/-- Intersection of an ellipse and a line -/
theorem ellipse_line_intersection
  (a b m c x : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) :
  (x^2 / a^2 + (m * x + c)^2 / b^2 = 1) ↔ 
  (x = (-a^2 * b * m * c + Real.sqrt (a^4 * b^2 * m^2 * c^2 - a^2 * b^2 * (a^2 * c^2 - a^2 * b^2) - a^2 * m^2 * (a^2 * c^2 - a^2 * b^2))) / (b^2 + a^2 * m^2) ∨
   x = (-a^2 * b * m * c - Real.sqrt (a^4 * b^2 * m^2 * c^2 - a^2 * b^2 * (a^2 * c^2 - a^2 * b^2) - a^2 * m^2 * (a^2 * c^2 - a^2 * b^2))) / (b^2 + a^2 * m^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l321_32199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_x₁_l321_32105

/-- The sequence defined by the recurrence relation -/
noncomputable def x : ℕ → ℝ → ℝ
  | 0, x₁ => x₁
  | n + 1, x₁ => x n x₁ * (x n x₁ + 1 / (n + 1))

/-- The property that the sequence is strictly increasing and bounded between 0 and 1 -/
def sequence_property (x₁ : ℝ) : Prop :=
  ∀ n : ℕ, 0 < x n x₁ ∧ x n x₁ < x (n + 1) x₁ ∧ x (n + 1) x₁ < 1

/-- Theorem stating the existence and uniqueness of x₁ satisfying the sequence property -/
theorem exists_unique_x₁ : ∃! x₁ : ℝ, sequence_property x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_x₁_l321_32105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_slope_l321_32184

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

/-- Theorem stating that for a hyperbola with eccentricity √3, the slope of its asymptotes is √2 -/
theorem hyperbola_asymptotes_slope (h : Hyperbola) 
  (h_eccentricity : eccentricity h = Real.sqrt 3) :
  asymptote_slope h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_slope_l321_32184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l321_32193

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.sqrt 3 * (Real.sin (x / 2))^2 + Real.sqrt 3

def has_intersections_distance (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ x₂ - x₁ = d ∧ f x₁ = 0 ∧ f x₂ = 0

theorem min_value_of_f (h : has_intersections_distance f (Real.pi / 2)) :
  ∃ x₀, x₀ ∈ Set.Icc 0 (Real.pi / 2) ∧
    (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x₀ ≤ f x) ∧
    f x₀ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l321_32193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_max_at_neg_one_implies_a_nonneg_three_zeros_implies_a_set_l321_32166

/-- The function f(x) defined as -x^2 + 2|x-a| --/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * abs (x - a)

/-- Theorem 1: If f is an even function, then a = 0 --/
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

/-- Theorem 2: If f reaches its maximum at x = -1, then a ≥ 0 --/
theorem max_at_neg_one_implies_a_nonneg (a : ℝ) :
  (∀ x : ℝ, f a (-1) ≥ f a x) → a ≥ 0 := by sorry

/-- Theorem 3: If f has exactly three zero points, then a ∈ {-1/2, 0, 1/2} --/
theorem three_zeros_implies_a_set (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧
    (∀ w : ℝ, f a w = 0 → w = x ∨ w = y ∨ w = z)) →
  a = -1/2 ∨ a = 0 ∨ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_max_at_neg_one_implies_a_nonneg_three_zeros_implies_a_set_l321_32166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shortest_side_l321_32102

/-- Given a triangle with medians of lengths 3, 4, and 5, prove that the shortest side has length 10/3 -/
theorem triangle_shortest_side (a b c : ℝ) (ma mb mc : ℝ) : 
  ma = 3 → mb = 4 → mc = 5 → 
  ma^2 = (2*b^2 + 2*c^2 - a^2)/4 → 
  mb^2 = (2*a^2 + 2*c^2 - b^2)/4 → 
  mc^2 = (2*a^2 + 2*b^2 - c^2)/4 → 
  min a (min b c) = 10/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shortest_side_l321_32102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_ratio_5_7_l321_32191

def adjacent_terms_ratio (n : ℕ) : Prop :=
  ∃ r : ℕ, (n.choose r : ℚ) / (n.choose (r + 1)) = 5 / 7

theorem min_n_for_ratio_5_7 :
  ∀ k : ℕ, 0 < k → k < 11 → ¬(adjacent_terms_ratio k) ∧ adjacent_terms_ratio 11 :=
by
  intro k hk_pos hk_lt_11
  sorry

#check min_n_for_ratio_5_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_ratio_5_7_l321_32191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_or_more_same_dice_l321_32165

/-- The probability of at least three out of five fair six-sided dice showing the same value -/
theorem prob_three_or_more_same_dice : (113 : ℚ) / 648 = 
  let n_dice : ℕ := 5
  let n_faces : ℕ := 6
  let prob_three_same : ℚ := (n_dice.choose 3) * (1 / n_faces^2) * ((n_faces - 1) / n_faces) * ((n_faces - 2) / n_faces)
  let prob_four_same : ℚ := (n_dice.choose 4) * (1 / n_faces^3) * ((n_faces - 1) / n_faces)
  let prob_five_same : ℚ := 1 / n_faces^4
  prob_three_same + prob_four_same + prob_five_same :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_or_more_same_dice_l321_32165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_values_l321_32108

/-- The function f(x) = (x^2 + 1)^2 / (x(x^2 - 1)) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

/-- The theorem stating that f takes the maximum number of values when |a| > 4 -/
theorem f_max_values (a : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ f x₄ = a) ↔
  abs a > 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_values_l321_32108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l321_32101

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

/-- Theorem: If S_3 = -3 and S_7 = 7 for an arithmetic sequence, then S_5 = 0 -/
theorem arithmetic_sequence_sum
  (seq : ArithmeticSequence)
  (h1 : S seq 3 = -3)
  (h2 : S seq 7 = 7) :
  S seq 5 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l321_32101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_on_line_l321_32137

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A parabola in standard form --/
inductive StandardParabola
  | VerticalAxis (p : ℝ) : StandardParabola  -- y² = 4px
  | HorizontalAxis (p : ℝ) : StandardParabola  -- x² = 4py

/-- The focus of a parabola --/
structure Focus where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line --/
def isOnLine (f : Focus) (l : Line) : Prop :=
  l.a * f.x + l.b * f.y + l.c = 0

/-- Check if a focus is on a coordinate axis --/
def isOnAxis (f : Focus) : Prop :=
  f.x = 0 ∨ f.y = 0

/-- Get the standard parabola form given a focus --/
noncomputable def getStandardParabola (f : Focus) : StandardParabola :=
  if f.x ≠ 0 then StandardParabola.VerticalAxis (f.x / 4)
  else StandardParabola.HorizontalAxis (-f.y / 4)

theorem parabola_focus_on_line :
  ∀ (f : Focus) (l : Line),
  l.a = 3 ∧ l.b = -4 ∧ l.c = -12 →
  isOnLine f l →
  isOnAxis f →
  (getStandardParabola f = StandardParabola.VerticalAxis 4 ∨
   getStandardParabola f = StandardParabola.HorizontalAxis (-3)) := by
  sorry

#check parabola_focus_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_on_line_l321_32137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_26_l321_32153

/-- The daily profit function for a mushroom processing factory -/
noncomputable def daily_profit (x t : ℝ) : ℝ := (100 * Real.exp 30 * (x - 20 - t)) / Real.exp x

/-- The conditions for the mushroom processing problem -/
def mushroom_problem (x t : ℝ) : Prop :=
  2 ≤ t ∧ t ≤ 5 ∧ 25 ≤ x ∧ x ≤ 40 ∧
  ∃ k : ℝ, k > 0 ∧ k / Real.exp 30 = 100

/-- The theorem stating the maximum profit when t = 5 -/
theorem max_profit_at_26 :
  ∀ x : ℝ, mushroom_problem x 5 →
  daily_profit 26 5 = 100 * Real.exp 4 ∧
  daily_profit x 5 ≤ daily_profit 26 5 := by
  sorry

#check max_profit_at_26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_26_l321_32153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l321_32183

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 1 then
    (1/2) * (1/x + x - 1 + x/(x-1))
  else
    0 -- Replace Real.arbitrary with a specific value, e.g., 0

theorem function_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 →
    f x + f (1 - 1/x) = 1/x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l321_32183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_value_l321_32148

def a : ℕ → ℚ
  | 0 => 1/2  -- Add this case for 0
  | 1 => 1/2
  | n + 1 => a n / (2 * a n + 1)

theorem a_1000_value : a 1000 = 1/2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_value_l321_32148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l321_32119

noncomputable section

open Real

-- Define the curve C
def curve_C (α : ℝ) : ℝ × ℝ :=
  (1 + (2 * sin α ^ 2) / (cos α ^ 2 - sin α ^ 2),
   (sqrt 2 * sin α * cos α) / (cos α ^ 2 - sin α ^ 2))

-- Define the line l
def line_l (ρ : ℝ) : ℝ × ℝ :=
  (ρ * cos (π / 6), ρ * sin (π / 6))

-- State the theorem
theorem curve_and_line_intersection :
  (∀ (x y : ℝ), (∃ α : ℝ, curve_C α = (x, y)) ↔ x^2 - 2*y^2 = 1) ∧
  (∃ ρ₁ ρ₂ : ℝ, 
    curve_C (arccos (1 / sqrt 3)) = line_l ρ₁ ∧
    curve_C (arccos (-1 / sqrt 3)) = line_l ρ₂ ∧
    ρ₁ = 2 ∧ ρ₂ = 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l321_32119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l321_32160

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0

-- Define the center of the circle
def circle_center (m : ℝ) : ℝ × ℝ :=
  (m + 3, 4*m^2 - 1)

-- Define the radius of the circle
noncomputable def circle_radius (m : ℝ) : ℝ :=
  Real.sqrt (-7*m^2 + 6*m + 1)

-- Theorem statement
theorem circle_properties :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_equation x y m) →
  (m < 1 ∧
   (0 < circle_radius m ∧ circle_radius m ≤ Real.sqrt 2) ∧
   (∃ f : ℝ → ℝ, f = (λ x ↦ 4*(x - 3)^2 - 1) ∧
    ∀ x : ℝ, x < 4 → (x, f x) = circle_center m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l321_32160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l321_32186

theorem stratified_sampling_theorem (total_students freshmen sophomores juniors sample_size : ℕ) 
  (h_total : total_students = 900)
  (h_freshmen : freshmen = 300)
  (h_sophomores : sophomores = 200)
  (h_juniors : juniors = 400)
  (h_sample : sample_size = 45)
  (h_sum : freshmen + sophomores + juniors = total_students) :
  (freshmen * sample_size) / total_students = 15 ∧ 
  (sophomores * sample_size) / total_students = 10 ∧ 
  (juniors * sample_size) / total_students = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l321_32186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l321_32127

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
noncomputable def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2
noncomputable def T (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * (1 - q^n) / (1 - q)

theorem sequence_problem (a₁ b₁ a₂ b₂ : ℝ) (d q : ℝ) :
  a₁ = -1 →
  b₁ = 1 →
  a₂ + b₂ = 2 →
  a₁ + b₁ = 5 →
  T b₁ q 3 = 21 →
  (∃ n : ℕ, geometric_sequence b₁ q n = 2^(n-1)) ∧
  (S a₁ d 3 = -6 ∨ S a₁ d 3 = 21) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l321_32127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_triangle_l321_32103

noncomputable def smallest_containing_triangle_side_length : ℝ := 
  (4 / Real.sqrt 3) * Real.sin (80 * Real.pi / 180) ^ 2

/-- The unit circle in ℝ² -/
def unit_circle : Set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 1}

/-- An equilateral triangle with side length a -/
def equilateral_triangle (a : ℝ) : Set (ℝ × ℝ) := sorry

/-- Three points are contained in a set -/
def points_contained (S : Set (ℝ × ℝ)) (p1 p2 p3 : ℝ × ℝ) : Prop :=
  p1 ∈ S ∧ p2 ∈ S ∧ p3 ∈ S

theorem smallest_containing_triangle :
  ∀ ε > 0, ∃ A B C : ℝ × ℝ,
    A ∈ unit_circle ∧ B ∈ unit_circle ∧ C ∈ unit_circle ∧
    (∀ a < smallest_containing_triangle_side_length + ε,
      ¬∃ P Q R : ℝ × ℝ, points_contained (equilateral_triangle a) A B C) ∧
    (∃ P Q R : ℝ × ℝ, points_contained (equilateral_triangle (smallest_containing_triangle_side_length + ε)) A B C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_triangle_l321_32103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l321_32120

-- Define the triangle sides
def a : ℝ := 15
def b : ℝ := 36
def c : ℝ := 39

-- Define the circle
def circle_diameter : ℝ := c

-- Define the areas of the non-triangular regions
variable (D E F : ℝ)

-- State the theorem
theorem circle_triangle_areas :
  -- The triangle is right-angled (Pythagorean theorem)
  a^2 + b^2 = c^2 →
  -- F is the largest area (half of the circle)
  F = π * (circle_diameter / 2)^2 / 2 →
  -- D and E are the other non-triangular regions
  D > 0 ∧ E > 0 →
  -- The sum of D, E, and the triangle area equals F
  D + E + (a * b / 2) = F := by
  sorry

#check circle_triangle_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l321_32120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_plate_price_l321_32156

/-- The price per pound of weight plates -/
noncomputable def price_per_pound (vest_cost plate_weight discounted_vest_cost savings : ℚ) : ℚ :=
  (discounted_vest_cost - savings - vest_cost) / plate_weight

/-- Theorem stating the price per pound of weight plates -/
theorem weight_plate_price :
  let vest_cost : ℚ := 250
  let plate_weight : ℚ := 200
  let original_vest_cost : ℚ := 700
  let discount : ℚ := 100
  let discounted_vest_cost : ℚ := original_vest_cost - discount
  let savings : ℚ := 110
  price_per_pound vest_cost plate_weight discounted_vest_cost savings = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_plate_price_l321_32156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_impossibility_l321_32180

theorem pythagorean_triple_impossibility :
  ¬∃ (m n : ℕ), 
    m > n ∧ 
    n > 0 ∧ 
    Odd m ∧ 
    Odd n ∧ 
    Nat.Coprime m n ∧
    (m^2 - n^2) / 2 = 6 ∧
    m * n = 8 ∧
    (m^2 + n^2) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_impossibility_l321_32180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_amount_in_container_l321_32151

/-- Proves that the initial amount of alcohol in a container is 4 quarts, given specific conditions -/
theorem alcohol_amount_in_container 
  (initial_water : ℝ) 
  (added_water : ℝ) 
  (ratio_alcohol : ℝ) 
  (ratio_water : ℝ) : 
  initial_water = 4 →
  added_water = 2.666666666666667 →
  ratio_alcohol = 3 →
  ratio_water = 5 →
  (ratio_alcohol / ratio_water) * (initial_water + added_water) = 4 :=
by
  intro h1 h2 h3 h4
  have final_water : ℝ := initial_water + added_water
  have alcohol : ℝ := (ratio_alcohol / ratio_water) * final_water
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_amount_in_container_l321_32151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_perpendicular_lines_l321_32152

-- Define the two given lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x - 5*y + 14 = 0

-- Define the point of intersection P
def P : ℝ × ℝ := (-2, 2)

-- Define the line we're comparing to
def compareLine (x y : ℝ) : Prop := 2*x - y + 7 = 0

-- Theorem statement
theorem intersection_parallel_perpendicular_lines :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →  -- The lines intersect
  (2 * P.1 - P.2 + 6 = 0) ∧  -- Parallel line passes through P
  (∀ (x y : ℝ), (2*x - y + 6 = 0) ↔ (∃ (k : ℝ), y - P.2 = k * (x - P.1) ∧ k = 2)) ∧  -- Parallel condition
  (P.1 + 2*P.2 - 2 = 0) ∧  -- Perpendicular line passes through P
  (∀ (x y : ℝ), (x + 2*y - 2 = 0) ↔ (∃ (k : ℝ), y - P.2 = k * (x - P.1) ∧ k = -1/2)) -- Perpendicular condition
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_perpendicular_lines_l321_32152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_l321_32162

theorem fraction_value : (Finset.prod (Finset.range 10) (λ i => i + 1)) / 
  ((Finset.sum (Finset.range 10) (λ i => i + 1)) * (1 * 2 * 3)) = 11000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_l321_32162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_time_l321_32190

/-- The time (in days) it takes for a given number of men to complete a task,
    given the work rate constant and the number of men. -/
noncomputable def time_for_task (k : ℝ) (men : ℝ) : ℝ := k / men

/-- The total time for digging and paving, given the number of men,
    the time for digging, and that paving takes half the time of digging. -/
noncomputable def total_time (men : ℝ) (dig_time : ℝ) : ℝ :=
  time_for_task (men * dig_time) men + time_for_task (men * dig_time / 2) men

theorem construction_time :
  let dig_men : ℝ := 20
  let dig_days : ℝ := 3
  let new_men : ℝ := 30
  total_time new_men dig_days = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_time_l321_32190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l321_32163

-- Define the points P, Q, R in 3D space
variable (P Q R : ℝ × ℝ × ℝ)

-- Define real numbers p, q, r
variable (p q r : ℝ)

-- Define the conditions
def midpoint_QR (P Q R : ℝ × ℝ × ℝ) (p q r : ℝ) : ℝ × ℝ × ℝ := (p, 0, 0)
def midpoint_PR (P Q R : ℝ × ℝ × ℝ) (p q r : ℝ) : ℝ × ℝ × ℝ := (0, q, 0)
def midpoint_PQ (P Q R : ℝ × ℝ × ℝ) (p q r : ℝ) : ℝ × ℝ × ℝ := (0, 0, r)

-- Define the distance function
noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := A
  let (x₂, y₂, z₂) := B
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

-- State the theorem
theorem triangle_ratio (P Q R : ℝ × ℝ × ℝ) (p q r : ℝ) 
  (h1 : midpoint_QR P Q R p q r = (p, 0, 0))
  (h2 : midpoint_PR P Q R p q r = (0, q, 0))
  (h3 : midpoint_PQ P Q R p q r = (0, 0, r)) :
  (distance P Q)^2 + (distance P R)^2 + (distance Q R)^2 = 8 * (p^2 + q^2 + r^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l321_32163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reroll_two_dice_l321_32154

/-- Represents a fair six-sided die --/
def Die := Fin 6

/-- The set of possible outcomes when rolling three dice --/
def ThreeDiceRoll := Die × Die × Die

/-- The sum of the numbers on three dice --/
def diceSum (roll : ThreeDiceRoll) : Nat :=
  roll.1.val + 1 + roll.2.1.val + 1 + roll.2.2.val + 1

/-- The optimal strategy for rerolling dice to get a sum of 7 --/
def optimalRerollStrategy (roll : ThreeDiceRoll) : Fin 4 :=
  sorry

/-- The probability of an event occurring when rolling three fair dice --/
noncomputable def probabilityOfEvent (event : ThreeDiceRoll → Prop) : ℚ :=
  sorry

/-- The main theorem: probability of rerolling exactly two dice in the optimal strategy --/
theorem probability_reroll_two_dice :
  probabilityOfEvent (fun roll => optimalRerollStrategy roll = 2) = 7 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reroll_two_dice_l321_32154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_distance_l321_32147

/- Define the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

/- Define point A -/
def A : ℝ × ℝ := (-2, 2)

/- Define point F (left focus) -/
def F : ℝ × ℝ := (-3, 0)

/- Define the distance function -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/- Define the function to be minimized -/
noncomputable def f (x y : ℝ) : ℝ := distance (x, y) A + (5/3) * distance (x, y) F

/- State the theorem -/
theorem ellipse_minimum_distance :
  ∀ x y : ℝ, is_on_ellipse x y →
    f x y ≥ 19/3 ∧
    (f (-5 * Real.sqrt 3 / 2) 2 = 19/3 ∧
     is_on_ellipse (-5 * Real.sqrt 3 / 2) 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_distance_l321_32147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integers_satisfying_inequality_l321_32122

theorem exists_integers_satisfying_inequality (x y : ℝ) :
  ∃ m n : ℤ, (x - ↑m)^2 + (y - ↑n) * (x - ↑m) + (y - ↑n)^2 ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integers_satisfying_inequality_l321_32122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ellipse_sine_ratio_l321_32116

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse defined by the equation x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on an ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Angle between two points with respect to the origin -/
noncomputable def angle (p1 p2 : Point) : ℝ := sorry

/-- Theorem: For a triangle ABC where A(0,4) and C(0,-4) are two vertices, 
    and B lies on the ellipse x²/9 + y²/25 = 1, 
    the ratio sin(A+C) / (sin A + sin C) = 4/5 -/
theorem triangle_ellipse_sine_ratio 
  (A B C : Point)
  (e : Ellipse)
  (h1 : A = ⟨0, 4⟩)
  (h2 : C = ⟨0, -4⟩)
  (h3 : e = ⟨3, 5⟩)
  (h4 : onEllipse B e) :
  Real.sin (angle B A + angle B C) / (Real.sin (angle B A) + Real.sin (angle B C)) = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ellipse_sine_ratio_l321_32116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BAO_calculation_l321_32170

theorem angle_BAO_calculation (AB AO : ℝ) (angle_BAO : ℝ) 
  (h1 : AB = 15)
  (h2 : AO = 8)
  (h3 : angle_BAO > 31 * Real.pi / 180)
  : angle_BAO = 150 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BAO_calculation_l321_32170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_properties_l321_32194

/-- Given a real constant a, functions f and g, and their intersection points -/
theorem intersection_points_properties (a : ℝ) (f g h : ℝ → ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  (f = λ x ↦ Real.log x) →
  (g = λ x ↦ a * x - 1) →
  (∀ x, h x = f x - g x) →
  (f x₁ = g x₁ ∧ f x₂ = g x₂) →
  (x₁ ≠ x₂) →
  (x₁ < x₂) →
  a ∈ Set.Ioo 0 1 ∧ 
  y₁ ∈ Set.Ioo (-1) 0 ∧
  Real.exp y₁ + Real.exp y₂ > 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_properties_l321_32194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l321_32181

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x + Real.sqrt 3 * Real.sin (2 * x)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_area (abc : Triangle) :
  f abc.A = 2 ∧
  abc.a = Real.sqrt 7 ∧
  Real.sin abc.B = 2 * Real.sin abc.C →
  (1 / 2) * abc.b * abc.c * Real.sin abc.A = 7 * Real.sqrt 3 / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l321_32181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_triangle_l321_32142

-- Define the coordinates of the three points
noncomputable def harry : ℝ × ℝ := (10, -3)
noncomputable def sandy : ℝ × ℝ := (2, 5)
noncomputable def luna : ℝ × ℝ := (-2, 9)

-- Define the centroid formula
noncomputable def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

-- Theorem statement
theorem centroid_of_triangle :
  centroid harry sandy luna = (10/3, 11/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_triangle_l321_32142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probabilities_l321_32175

/-- Player A's shooting percentage -/
noncomputable def player_a_percentage : ℝ := 0.6

/-- Player B's shooting percentage -/
noncomputable def player_b_percentage : ℝ := 0.8

/-- Probability of each player taking the first shot -/
noncomputable def first_shot_probability : ℝ := 0.5

/-- Probability that player B takes the second shot -/
noncomputable def prob_b_second_shot : ℝ := 0.6

/-- Probability that player A takes the i-th shot -/
noncomputable def prob_a_ith_shot (i : ℕ) : ℝ := 1/3 + (1/6) * (2/5)^(i-1)

/-- Expected number of times player A shoots in first n shots -/
noncomputable def expected_a_shots (n : ℕ) : ℝ := (5/18) * (1 - (2/5)^n) + n/3

theorem basketball_game_probabilities :
  (prob_b_second_shot = 0.6) ∧
  (∀ i : ℕ, prob_a_ith_shot i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expected_a_shots n = (5/18) * (1 - (2/5)^n) + n/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probabilities_l321_32175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l321_32167

open Polynomial

theorem polynomial_divisibility :
  ∃! (a : ℤ), ∃ (p : Polynomial ℤ), X^13 + X + C 94 = (X^2 - X + C a) * p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l321_32167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mixture_ratio_l321_32164

/-- Represents a salt solution with a given concentration and available amount -/
structure SaltSolution where
  concentration : ℚ
  available : ℚ

/-- Represents a mixture of two salt solutions -/
structure Mixture where
  solution_a : SaltSolution
  solution_b : SaltSolution
  ratio_a : ℚ
  ratio_b : ℚ

/-- Calculates the salt concentration of a mixture -/
def mixtureSaltConcentration (m : Mixture) : ℚ :=
  (m.solution_a.concentration * m.ratio_a + m.solution_b.concentration * m.ratio_b) / (m.ratio_a + m.ratio_b)

/-- Calculates the total volume of a mixture -/
def mixtureVolume (m : Mixture) : ℚ := m.ratio_a + m.ratio_b

theorem correct_mixture_ratio (solution_a solution_b : SaltSolution) :
  solution_a.concentration = 2/5 →
  solution_b.concentration = 4/5 →
  solution_a.available = 30 →
  solution_b.available = 60 →
  ∃ m : Mixture, m.solution_a = solution_a ∧ m.solution_b = solution_b ∧
    m.ratio_a = 3 ∧ m.ratio_b = 1 ∧
    mixtureSaltConcentration m = 1/2 ∧ mixtureVolume m = 4 := by
  sorry

#check correct_mixture_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mixture_ratio_l321_32164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_a_value_l321_32157

/-- Represents a parabola with equation x^2 = ay -/
structure Parabola where
  a : ℝ

/-- The focus of a parabola -/
noncomputable def Parabola.focus (p : Parabola) : ℝ × ℝ := (0, p.a / 4)

/-- Theorem: For a parabola with equation x^2 = ay and focus at (0, 5), the value of a is 20 -/
theorem parabola_a_value (p : Parabola) (h : Parabola.focus p = (0, 5)) : p.a = 20 := by
  sorry

#check parabola_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_a_value_l321_32157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_x_minus_sin_z_l321_32117

theorem min_value_cos_x_minus_sin_z :
  ∀ (x y z : ℝ),
    (2 * Real.sin x = Real.tan y) →
    (2 * Real.cos y = 1 / Real.tan z) →
    (Real.sin z = Real.tan x) →
    (∀ (x' y' z' : ℝ),
      (2 * Real.sin x' = Real.tan y') →
      (2 * Real.cos y' = 1 / Real.tan z') →
      (Real.sin z' = Real.tan x') →
      (Real.cos x - Real.sin z ≤ Real.cos x' - Real.sin z')) →
    Real.cos x - Real.sin z = -(5 * Real.sqrt 3) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_x_minus_sin_z_l321_32117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_representation_l321_32113

noncomputable def tan_22_5 : ℝ := Real.tan (22.5 * Real.pi / 180)

theorem tan_22_5_representation :
  ∃ (a b c d : ℕ), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (tan_22_5 = Real.sqrt (a : ℝ) - Real.sqrt (b : ℝ) + Real.sqrt (c : ℝ) - (d : ℝ)) ∧
    (a + b + c + d = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_representation_l321_32113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_between_circles_l321_32172

theorem area_increase_between_circles (π : ℝ) (h_π_pos : π > 0) : 
  let r1 : ℝ := 6
  let r2 : ℝ := 4
  let new_r1 : ℝ := r1 * 1.5
  let new_r2 : ℝ := r2 * 0.75
  let area_original := π * (r1^2 - r2^2)
  let area_new := π * (new_r1^2 - new_r2^2)
  let percent_increase := (area_new - area_original) / area_original * 100
  percent_increase = 260 := by
  sorry

#check area_increase_between_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_between_circles_l321_32172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_to_sum_2022_eq_338_l321_32111

/-- The number of ways to write 2022 as a sum of 2s and 3s -/
def ways_to_sum_2022 : ℕ :=
  Finset.card (Finset.filter (fun n => 6 * n + 6 * (337 - n) = 2022) (Finset.range 338))

/-- Theorem stating that there are 338 ways to write 2022 as a sum of 2s and 3s -/
theorem ways_to_sum_2022_eq_338 : ways_to_sum_2022 = 338 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_to_sum_2022_eq_338_l321_32111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarterback_pass_ratio_l321_32149

/-- Proves the ratio of passes thrown to the right side of the field
    to the passes thrown to the left side of the field is 2:1 --/
theorem quarterback_pass_ratio :
  ∃ (total_passes left_passes center_passes right_passes : ℕ),
    total_passes = 50 ∧
    left_passes = 12 ∧
    center_passes = left_passes + 2 ∧
    right_passes = total_passes - left_passes - center_passes ∧
    (right_passes : ℚ) / left_passes = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarterback_pass_ratio_l321_32149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l321_32124

theorem rectangle_area_increase (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  (1.2 * L * 1.2 * W - L * W) / (L * W) = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l321_32124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l321_32195

theorem max_imaginary_part_of_roots (z : ℂ) : 
  z^12 - z^9 + z^6 - z^3 + 1 = 0 → 
  ∃ (θ : ℝ), θ = 84 * π / 180 ∧ 
  ∀ (w : ℂ), w^12 - w^9 + w^6 - w^3 + 1 = 0 → 
  Complex.abs w.im ≤ Real.sin θ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l321_32195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l321_32107

theorem triangle_cosine_proof : 
  ∀ (a b c : ℝ),
  a = 3 ∧ b = 2 * Real.sqrt 2 ∧ c = 2 →
  (a^2 + b^2 - c^2) / (2 * a * b) = (13 * Real.sqrt 2) / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l321_32107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_sum_less_than_double_m_l321_32139

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 5 - a / Real.exp x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * f a x

-- Theorem for part I
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) → a ≥ 2 * Real.exp 1 := by
  sorry

-- Theorem for part II
theorem sum_less_than_double_m (a : ℝ) (m x₁ x₂ : ℝ) :
  m ≥ 1 → x₁ ≠ x₂ → g a x₁ + g a x₂ = 2 * g a m → x₁ + x₂ < 2 * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_sum_less_than_double_m_l321_32139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_tac_toe_probability_l321_32110

/-- A tic-tac-toe board -/
structure TicTacToeBoard :=
  (size : Nat)
  (crosses : Nat)
  (noughts : Nat)

/-- The number of winning positions on a tic-tac-toe board -/
def winningPositions (board : TicTacToeBoard) : Nat :=
  2 * board.size + 2

/-- The total number of ways to arrange noughts on the board -/
def totalArrangements (board : TicTacToeBoard) : Nat :=
  Nat.choose (board.size * board.size) board.noughts

/-- The probability of noughts being in a winning position -/
def winningProbability (board : TicTacToeBoard) : ℚ :=
  (winningPositions board : ℚ) / (totalArrangements board : ℚ)

/-- Theorem: The probability of three noughts being in a winning position
    on a 3x3 tic-tac-toe board filled with 6 crosses and 3 noughts is 2/21 -/
theorem tic_tac_toe_probability :
  let board : TicTacToeBoard := ⟨3, 6, 3⟩
  winningProbability board = 2 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_tac_toe_probability_l321_32110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_interior_angle_equality_l321_32161

-- Define the triangle ABC
variable (A B C : Plane)

-- Define interior points P and Q
variable (P Q : Plane)

-- Define the angle measure function
noncomputable def angle (p q r : Plane) : ℝ := sorry

-- State the theorem
theorem triangle_interior_angle_equality 
  (h1 : angle P B A = angle Q B C)
  (h2 : angle P C A = angle Q C B)
  : angle P A B = angle Q A C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_interior_angle_equality_l321_32161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l321_32192

/-- The focus of the parabola y = 4x^2 -/
noncomputable def focus : ℝ × ℝ := (0, 1/16)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2

/-- The distance from a point on the parabola to the focus -/
noncomputable def distance_to_focus (x y : ℝ) : ℝ := 
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2)

/-- The distance from a point on the parabola to the directrix -/
noncomputable def distance_to_directrix (x y : ℝ) : ℝ := 
  Real.sqrt ((y - (-focus.2))^2)

/-- Theorem: The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem parabola_focus : 
  ∀ x y : ℝ, parabola x y → distance_to_focus x y = distance_to_directrix x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l321_32192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_courts_count_l321_32121

-- Define the land area and court size
noncomputable def land_area : ℝ := 10000
noncomputable def court_size : ℝ := 1000

-- Define the construction cost function
noncomputable def f (x : ℝ) : ℝ := 800 * (1 + 1/5 * Real.log x)

-- Define the environmental protection fee
noncomputable def env_fee : ℝ := 1280000

-- Define the comprehensive cost function
noncomputable def g (x : ℝ) : ℝ := f x + env_fee / (x * court_size)

-- Define the domain
def is_valid_x (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem optimal_courts_count :
  ∃ (x : ℕ), is_valid_x x ∧
  (∀ (y : ℕ), is_valid_x y → g (x : ℝ) ≤ g (y : ℝ)) ∧
  x = 8 := by
  sorry

#check optimal_courts_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_courts_count_l321_32121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l321_32109

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

noncomputable def g (x m : ℝ) : ℝ := (1/2)^x - m

theorem problem_statement (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂ m) →
  m ≥ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l321_32109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_discarding_l321_32146

theorem average_after_discarding (numbers : Finset ℝ) (sum : ℝ) (avg : ℝ) : 
  numbers.card = 50 →
  sum = Finset.sum numbers id →
  avg = sum / 50 →
  avg = 62 →
  45 ∈ numbers →
  55 ∈ numbers →
  let remaining := numbers.erase 45
  let remaining' := remaining.erase 55
  (Finset.sum remaining' id / 48 : ℝ) = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_discarding_l321_32146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l321_32131

/-- Represents a bicycle wheel -/
structure Wheel where
  radius : ℝ
  revolutions : ℝ

/-- Calculates the distance traveled by a wheel -/
noncomputable def distanceTraveled (w : Wheel) : ℝ :=
  2 * Real.pi * w.radius * w.revolutions

theorem bicycle_wheel_revolutions
  (front_wheel : Wheel)
  (back_wheel : Wheel)
  (h1 : front_wheel.radius = 3)
  (h2 : front_wheel.revolutions = 150)
  (h3 : back_wheel.radius = 0.5)
  (h4 : distanceTraveled front_wheel = distanceTraveled back_wheel) :
  back_wheel.revolutions = 900 := by
  sorry

#check bicycle_wheel_revolutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l321_32131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_are_similar_l321_32178

/-- Triangle type -/
structure Triangle where
  -- Define triangle properties here
  mk ::

/-- Congruent angles predicate -/
def congruent_angles (t1 t2 : Triangle) : Prop :=
  sorry

/-- Proportional sides predicate -/
def proportional_sides (t1 t2 : Triangle) : Prop :=
  sorry

/-- Two triangles are similar if they have congruent angles and proportional sides -/
def are_similar (t1 t2 : Triangle) : Prop :=
  (congruent_angles t1 t2) ∧ (proportional_sides t1 t2)

theorem similar_triangles_are_similar (t1 t2 : Triangle) :
  are_similar t1 t2 → are_similar t1 t2 := by
  intro h
  exact h

#check similar_triangles_are_similar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_are_similar_l321_32178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_spotted_fraction_is_half_l321_32174

/-- Fraction of blue mushrooms with white spots -/
def blue_spotted_fraction (red_count green_count blue_count brown_count : ℕ)
  (red_spotted_fraction brown_spotted_fraction : ℚ)
  (total_spotted : ℕ) : ℚ :=
  (total_spotted - (red_spotted_fraction * red_count).floor - (brown_spotted_fraction * brown_count).floor) / blue_count

theorem blue_spotted_fraction_is_half :
  blue_spotted_fraction 12 14 6 6 (2/3) 1 17 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_spotted_fraction_is_half_l321_32174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_bridge_passing_time_l321_32141

/-- The time (in seconds) it takes for a ship to pass a bridge -/
noncomputable def time_to_pass_bridge (ship_length : ℝ) (ship_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := ship_length + bridge_length
  let ship_speed_ms := ship_speed_kmh * (1000 / 3600)
  total_distance / ship_speed_ms

/-- Theorem stating that the time for a ship of length 450 m, 
    traveling at 24 km/hr, to pass a bridge of length 900 m 
    is approximately 202.4 seconds -/
theorem ship_bridge_passing_time :
  let t := time_to_pass_bridge 450 24 900
  ∃ ε > 0, |t - 202.4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_bridge_passing_time_l321_32141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_haley_cider_theorem_l321_32135

/-- Represents the number of pints of cider Haley can make given the following conditions:
  * 20 golden delicious apples and 40 pink lady apples make one pint of cider
  * 6 farmhands can pick 240 apples per hour
  * Farmhands will work 5 hours
  * The ratio of golden delicious to pink lady apples is 1:2
-/
def haley_cider_pints : ℕ :=
  let golden_delicious_per_pint : ℕ := 20
  let pink_lady_per_pint : ℕ := 40
  let farmhands : ℕ := 6
  let apples_per_hour : ℕ := 240
  let work_hours : ℕ := 5
  let golden_delicious_ratio : ℕ := 1
  let pink_lady_ratio : ℕ := 2

  -- Total apples picked
  let total_apples : ℕ := farmhands * apples_per_hour * work_hours

  -- Number of golden delicious apples
  let golden_delicious_apples : ℕ := total_apples * golden_delicious_ratio / (golden_delicious_ratio + pink_lady_ratio)

  -- Number of pink lady apples
  let pink_lady_apples : ℕ := total_apples * pink_lady_ratio / (golden_delicious_ratio + pink_lady_ratio)

  -- Pints from golden delicious apples
  let golden_delicious_pints : ℕ := golden_delicious_apples / golden_delicious_per_pint

  -- Pints from pink lady apples
  let pink_lady_pints : ℕ := pink_lady_apples / pink_lady_per_pint

  -- Total pints
  golden_delicious_pints

theorem haley_cider_theorem : haley_cider_pints = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_haley_cider_theorem_l321_32135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l321_32179

-- Define the constants
noncomputable def a : ℝ := Real.exp (-0.02)
def b : ℝ := 0.01
noncomputable def c : ℝ := Real.log 1.01

-- State the theorem
theorem a_gt_b_gt_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l321_32179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_10_equals_50_l321_32196

def a (n : ℕ) : ℤ := 11 - 2 * n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i => Int.natAbs (a (i + 1)))

theorem S_10_equals_50 : S 10 = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_10_equals_50_l321_32196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l321_32155

/-- The time taken for the slower train to pass the driver of the faster train -/
noncomputable def train_passing_time (train_length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  train_length / (speed1 + speed2)

/-- Theorem stating the time taken for the slower train to pass the driver of the faster train -/
theorem train_passing_time_approx :
  let train_length : ℝ := 500
  let speed1 : ℝ := 45 * 1000 / 3600  -- 45 km/h converted to m/s
  let speed2 : ℝ := 30 * 1000 / 3600  -- 30 km/h converted to m/s
  ∃ ε > 0, |train_passing_time train_length speed1 speed2 - 24.01| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l321_32155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_eq_one_l321_32104

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x + 2 else -x^2

theorem f_of_f_one_eq_one : f (f 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_eq_one_l321_32104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimized_angle_line_equation_l321_32128

open Set Real

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | (p.1 + 2)^2 + p.2^2 = 4}

-- Define the other circle
def other_circle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 9}

-- Define the point P
def P : ℝ × ℝ := (-1, 1)

-- Define the property of external tangency
def externally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of a line passing through two points
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the angle between three points
noncomputable def angle (A C B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem minimized_angle_line_equation :
  ∀ A B : ℝ × ℝ,
  A ∈ circle_C →
  B ∈ circle_C →
  externally_tangent circle_C other_circle →
  P ∈ line_through A B →
  (∀ A' B' : ℝ × ℝ, A' ∈ circle_C → B' ∈ circle_C → P ∈ line_through A' B' → 
    angle A' (-2, 0) B' ≥ angle A (-2, 0) B) →
  line_through A B = {p : ℝ × ℝ | p.1 + p.2 = 0} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimized_angle_line_equation_l321_32128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l321_32169

noncomputable section

open Real

/-- Curve C in polar coordinates -/
def curve_C (θ : ℝ) : ℝ := 2 * cos θ + 2 * sqrt 3 * sin θ

/-- Line l₁ in polar coordinates -/
def line_l₁ (α : ℝ) : ℝ → Prop := fun θ ↦ θ = α

/-- Line l₂ in polar coordinates -/
def line_l₂ (α : ℝ) : ℝ → Prop := fun θ ↦ θ = α + π/3

/-- Intersection point A of C and l₁ -/
def point_A (α : ℝ) : ℝ := 4 * sin (α + π/6)

/-- Intersection point B of C and l₂ -/
def point_B (α : ℝ) : ℝ := 4 * cos α

/-- Sum of distances |OA| + |OB| -/
def sum_distances (α : ℝ) : ℝ := point_A α + point_B α

theorem max_sum_distances :
  ∀ α, 0 < α → α < π/2 →
  sum_distances α ≤ 4 * sqrt 3 ∧
  ∃ α₀, 0 < α₀ ∧ α₀ < π/2 ∧ sum_distances α₀ = 4 * sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l321_32169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l321_32115

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

-- State the theorem
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = 6 * Real.pi ∧ (∀ x, f (x + T) = f x)) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ x, f x ≤ M) ∧ (∃ y, f y = M)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l321_32115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_sequence_l321_32197

theorem smallest_number_in_sequence (seq : List ℕ) : 
  seq.length = 27 ∧ 
  (∀ i j, i < j → i < seq.length → j < seq.length → seq[i]! + 1 = seq[j]!) ∧
  seq.sum = 1998 →
  seq.head? = some 61 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_sequence_l321_32197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_equals_one_l321_32134

theorem intersection_implies_a_equals_one : 
  ∀ (a : ℤ), 
  let M : Set ℤ := {a, 0}
  let N : Set ℤ := {x : ℤ | 2 * x^2 - 3 * x < 0}
  (M ∩ N).Nonempty → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_equals_one_l321_32134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l321_32140

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) - Real.sqrt 3 * (Real.cos x)^2

-- Define the function g (stretched version of f)
noncomputable def g (x : ℝ) : ℝ := f (x/2)

-- Theorem statement
theorem function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧ 
  (∀ (x : ℝ), f x ≥ -(2 + Real.sqrt 3) / 2) ∧
  (∃ (x : ℝ), f x = -(2 + Real.sqrt 3) / 2) ∧
  (Set.Icc ((1 - Real.sqrt 3) / 2) ((2 - Real.sqrt 3) / 2) = 
    {y | ∃ (x : ℝ), x ∈ Set.Icc (π/2) π ∧ g x = y}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l321_32140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_skew_pairs_l321_32158

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where

/-- A line in three-dimensional space -/
structure Line3D where

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The main diagonal of a cube -/
def main_diagonal (c : Cube) : Line3D := sorry

/-- The edges of a cube -/
def cube_edges (c : Cube) : List Line3D := sorry

/-- Count the number of skew line pairs between a line and a list of lines -/
def count_skew_pairs (l : Line3D) (ls : List Line3D) : Nat := sorry

theorem cube_diagonal_skew_pairs (c : Cube) :
  count_skew_pairs (main_diagonal c) (cube_edges c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_skew_pairs_l321_32158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_DEF_l321_32118

theorem min_perimeter_triangle_DEF (d e f : ℕ) : 
  d > 0 ∧ e > 0 ∧ f > 0 →
  d + e > f ∧ d + f > e ∧ e + f > d →
  Real.cos (Real.arccos ((d^2 + e^2 - f^2) / (2 * d * e : ℝ))) = 3/5 →
  Real.cos (Real.arccos ((d^2 + f^2 - e^2) / (2 * d * f : ℝ))) = 24/25 →
  Real.cos (Real.arccos ((e^2 + f^2 - d^2) / (2 * e * f : ℝ))) = -1/3 →
  d + e + f ≥ 37 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_DEF_l321_32118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_negative_example_l321_32168

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ / 100 : ℝ)

theorem round_negative_example :
  round_to_hundredth (-23.4985) = -23.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_negative_example_l321_32168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l321_32176

/-- The differential equation y^(IV) + 4y''' + 8y'' + 8y' + 4y = 0 -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv^[4] y) x + 4 * (deriv^[3] y) x + 8 * (deriv^[2] y) x + 8 * (deriv y) x + 4 * y x = 0

/-- The proposed solution function -/
noncomputable def solution (C₁ C₂ C₃ C₄ : ℝ) (x : ℝ) : ℝ :=
  Real.exp (-x) * ((C₁ + C₃ * x) * Real.cos x + (C₂ + C₄ * x) * Real.sin x)

/-- Theorem stating that the proposed solution satisfies the differential equation -/
theorem solution_satisfies_equation :
  ∀ C₁ C₂ C₃ C₄, differential_equation (solution C₁ C₂ C₃ C₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l321_32176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_value_l321_32144

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 6 * x - 9 = 0

def root_form (x m n p : ℝ) : Prop := x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

theorem quadratic_root_value : 
  ∃ (m n p : ℕ+), 
    (∀ x : ℝ, quadratic_equation x → root_form x m.val n.val p.val) ∧ 
    Nat.gcd m.val (Nat.gcd n.val p.val) = 1 ∧
    n = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_value_l321_32144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cost_correct_water_cost_10_tons_water_cost_27_tons_l321_32136

/-- Water cost function based on tiered pricing system -/
noncomputable def water_cost (m : ℝ) : ℝ :=
  if m ≤ 20 then 1.6 * m
  else if m ≤ 30 then 2.4 * m - 16
  else 4.8 * m - 88

theorem water_cost_correct (m : ℝ) :
  (m ≥ 0 ∧ m ≤ 20 → water_cost m = 1.6 * m) ∧
  (m > 20 ∧ m ≤ 30 → water_cost m = 2.4 * m - 16) ∧
  (m > 30 → water_cost m = 4.8 * m - 88) := by
  sorry

-- Specific cases
theorem water_cost_10_tons : water_cost 10 = 16 := by
  sorry

theorem water_cost_27_tons : water_cost 27 = 48.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cost_correct_water_cost_10_tons_water_cost_27_tons_l321_32136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_property_triangle_median_property_l321_32126

open Real

def triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def cosine_law (a b c A B C : ℝ) : Prop :=
  cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧
  cos B = (c^2 + a^2 - b^2) / (2*c*a) ∧
  cos C = (a^2 + b^2 - c^2) / (2*a*b)

def sine_law (a b c A B C : ℝ) : Prop :=
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_angle_property 
  (a b c A B C : ℝ) 
  (h_triangle : triangle a b c) 
  (h_cosine : cosine_law a b c A B C) 
  (h_sine : sine_law a b c A B C) 
  (h_condition : (cos A)^2 - (cos B)^2 - (cos C)^2 = -1 + sin B * sin C) :
  A = π/3 := by sorry

theorem triangle_median_property 
  (a b c A B C M : ℝ) 
  (h_triangle : triangle a b c) 
  (h_cosine : cosine_law a b c A B C) 
  (h_sine : sine_law a b c A B C)
  (h_sum : b + c = 6)
  (h_median : M^2 = (2*b^2 + 2*c^2 - a^2) / 4) :
  M ≥ 3*Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_property_triangle_median_property_l321_32126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_proper_divisors_implies_perfect_square_l321_32187

/-- A positive integer with an odd number of proper divisors is a perfect square. -/
theorem odd_proper_divisors_implies_perfect_square (n : ℕ+) :
  (Odd (Finset.filter (fun d ↦ d ∣ n.val ∧ d ≠ 1 ∧ d ≠ n.val) (Finset.range (n.val + 1))).card) →
  ∃ m : ℕ, n.val = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_proper_divisors_implies_perfect_square_l321_32187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l321_32177

theorem absolute_value_expression (x : ℤ) (h : x = -2520) : 
  abs (abs (abs x + x) - 2 * abs x) + x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l321_32177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_ratio_l321_32182

theorem park_area_ratio (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3 * r)^2) = 1 / 9 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_ratio_l321_32182
