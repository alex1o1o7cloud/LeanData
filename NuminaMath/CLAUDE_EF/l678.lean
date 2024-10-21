import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l678_67831

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def a (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)
noncomputable def b (α : ℝ) : ℝ × ℝ := (Real.cos α, 0)
noncomputable def c (α : ℝ) : ℝ × ℝ := (-Real.sin α, 2)

noncomputable def P (α : ℝ) : ℝ × ℝ := (Real.cos α - 1/2 * Real.sin α, 1/2)

noncomputable def f (α : ℝ) : ℝ := (P α).1 * (c α).1 + (P α).2 * (c α).2

theorem problem_statement (α : ℝ) :
  (f α = 1/2 * (1 - Real.cos (2 * α))) ∧
  (Set.range f = Set.Icc (1/2) 1) ∧
  (∃ (k : ℝ), k • (P α - O) = c α - O →
    ‖(b α - O) + (P α - O)‖ = Real.sqrt (Real.cos α ^ 2 + (Real.cos α - 1/2 * Real.sin α) ^ 2 + 1/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l678_67831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equation_l678_67888

/-- Given a curve C with polar equation ρ²(1+3sin²θ) = 4, 
    its standard equation in Cartesian coordinates is x²/4 + y² = 1 -/
theorem polar_to_cartesian_equation :
  ∀ (ρ θ x y : ℝ),
  (ρ ≥ 0) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (ρ^2 * (1 + 3 * Real.sin θ^2) = 4) →
  (x^2 / 4 + y^2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equation_l678_67888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l678_67816

def x : ℕ → ℚ
  | 0 => 3  -- Add this case for n = 0
  | 1 => 3
  | n + 2 => (x (n + 1) ^ 2 + 2) / (2 * x (n + 1) - 1)

theorem x_general_term : ∀ n : ℕ, n ≥ 1 → x n = (2 * 2^(2^n) + 1) / (2^(2^n) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l678_67816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l678_67856

-- Define the set S
noncomputable def S : Set ℂ := {z : ℂ | z.re^2 + z.im^2 ≤ 4}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2 * Complex.I) * z

-- Theorem statement
theorem transform_stays_in_S : ∀ z ∈ S, transform z ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l678_67856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l678_67808

theorem sum_remainder_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 11) 
  (hb : b % 15 = 12) 
  (hc : c % 15 = 13) : 
  (a + b + c) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l678_67808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_l678_67806

theorem sqrt_expressions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (Real.sqrt x * Real.sqrt y = Real.sqrt (x*y))) →
  (∀ x : ℝ, x ≥ 0 → Real.sqrt (x^2) = x) →
  (∀ x : ℝ, x > 0 → (Real.sqrt x)^2 = x) →
  (1/2 * Real.sqrt 24 - Real.sqrt 3 * Real.sqrt 2 = 0) ∧
  ((2 * Real.sqrt 3 + 3 * Real.sqrt 2)^2 = 30 + 12 * Real.sqrt 6) :=
by
  sorry

#check sqrt_expressions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_l678_67806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_variety_cost_is_20_l678_67813

/-- Represents the cost and ratio of tea varieties in a blend -/
structure TeaBlend where
  cost1 : ℚ  -- Cost of first variety per kg
  ratio1 : ℚ  -- Ratio of first variety in blend
  ratio2 : ℚ  -- Ratio of second variety in blend
  sellPrice : ℚ  -- Selling price of blended tea per kg
  gainPercent : ℚ  -- Gain percentage

/-- Calculates the cost of the second variety of tea per kg -/
noncomputable def secondVarietyCost (blend : TeaBlend) : ℚ :=
  let totalRatio := blend.ratio1 + blend.ratio2
  let totalSellPrice := blend.sellPrice * totalRatio
  let costPrice := totalSellPrice / (1 + blend.gainPercent / 100)
  let cost1Total := blend.cost1 * blend.ratio1
  (costPrice - cost1Total) / blend.ratio2

/-- Theorem stating the cost of the second variety of tea -/
theorem second_variety_cost_is_20 (blend : TeaBlend) 
    (h1 : blend.cost1 = 18)
    (h2 : blend.ratio1 = 5)
    (h3 : blend.ratio2 = 3)
    (h4 : blend.sellPrice = 21)
    (h5 : blend.gainPercent = 12) :
    secondVarietyCost blend = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_variety_cost_is_20_l678_67813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l678_67833

/-- Parabola type representing y² = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_distance_theorem (C : Parabola) (A B : Point) :
  C.equation A.x A.y →
  B.x = 3 ∧ B.y = 0 →
  C.focus = (1, 0) →
  distance A ⟨C.focus.1, C.focus.2⟩ = distance B ⟨C.focus.1, C.focus.2⟩ →
  distance A B = Real.sqrt 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l678_67833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequences_l678_67842

def arithmetic_sequence (a : ℝ → ℝ) (a₁ d : ℝ) : Prop :=
  a 1 = a₁ ∧ ∀ n : ℝ, a (n + 1) = a n + d

def geometric_sequence (b : ℝ → ℝ) (b₁ r : ℝ) : Prop :=
  b 1 = b₁ ∧ ∀ n : ℝ, b (n + 1) = b n * r

theorem sum_of_sequences (a b : ℝ → ℝ) :
  arithmetic_sequence a 2 1 →
  geometric_sequence b 1 2 →
  a (b 1) + a (b 2) + a (b 3) + a (b 4) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequences_l678_67842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_circle_properties_l678_67841

-- Define the vector type
def MyVector : Type := ℝ × ℝ

-- Define the dot product for vectors
def dot_product (v w : MyVector) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity for vectors
def perpendicular (v w : MyVector) : Prop := dot_product v w = 0

-- Define the trajectory E
def trajectory_E (m : ℝ) (x y : ℝ) : Prop := m * x^2 + y^2 = 1

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4/5

-- Main theorem
theorem trajectory_and_circle_properties (m : ℝ) (x y : ℝ) :
  let a : MyVector := (m * x, y + 1)
  let b : MyVector := (x, y - 1)
  perpendicular a b ↔ trajectory_E m x y ∧
  (m = 1/4 → 
    ∃ (A B : ℝ × ℝ),
      my_circle A.1 A.2 ∧
      my_circle B.1 B.2 ∧
      trajectory_E (1/4) A.1 A.2 ∧
      trajectory_E (1/4) B.1 B.2 ∧
      dot_product A B = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_circle_properties_l678_67841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l678_67898

/-- A hyperbola passing through a specific point has asymptotes y = ±√2x -/
theorem hyperbola_asymptotes (b : ℝ) (h1 : b > 0) 
  (h2 : 3^2 - 4^2 / b^2 = 1) : 
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = Real.sqrt 2 * x) ∧ 
    (∀ x, g x = -(Real.sqrt 2 * x)) ∧
    (∀ x y, x^2 - y^2 / b^2 = 1 → (y = f x ∨ y = g x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l678_67898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base11_subtraction_addition_l678_67821

/-- Represents a number in base 11 --/
structure Base11 where
  digits : List (Fin 11)

/-- Convert a natural number to its base 11 representation --/
def toBase11 (n : ℕ) : Base11 := sorry

/-- Convert a base 11 number to a natural number --/
def fromBase11 (b : Base11) : ℕ := sorry

/-- Addition in base 11 --/
def addBase11 (a b : Base11) : Base11 := sorry

/-- Subtraction in base 11 --/
def subBase11 (a b : Base11) : Base11 := sorry

/-- Equality check for Base11 numbers --/
def eqBase11 (a b : Base11) : Prop := fromBase11 a = fromBase11 b

theorem base11_subtraction_addition :
  let a := toBase11 674
  let b := toBase11 279
  let c := toBase11 143
  let result := toBase11 559
  eqBase11 (subBase11 (addBase11 (subBase11 a b) c) result) (toBase11 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base11_subtraction_addition_l678_67821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_range_of_k_l678_67825

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k / x

-- Part 1
theorem min_value_of_f (k : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f k x ≥ f k x₀) ∧
  (∀ x > 0, deriv (f k) x = 0 → x = Real.exp 1) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f k x ≥ 2) :=
by
  sorry

-- Part 2
theorem range_of_k :
  (∀ k : ℝ, (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f k x₁ - f k x₂ < x₁ - x₂) →
   k ≥ 1/4) ∧
  (∀ k : ℝ, k ≥ 1/4 → ∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f k x₁ - f k x₂ < x₁ - x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_range_of_k_l678_67825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_equals_12_l678_67837

noncomputable def hyperbola_sum (center : ℝ × ℝ) (focus : ℝ × ℝ) (vertex : ℝ × ℝ) : ℝ :=
  let (h, k) := center
  let a := |vertex.2 - center.2|
  let c := |focus.2 - center.2|
  let b := Real.sqrt (c^2 - a^2)
  h + k + a + b

theorem hyperbola_sum_equals_12 :
  hyperbola_sum (0, 2) (0, 2 + 5 * Real.sqrt 2) (0, 7) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_equals_12_l678_67837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_motorboat_path_l678_67852

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the Apollonius circle -/
structure ApolloniusCircle where
  center : Point
  radius : ℝ

/-- Represents the motorboat's path -/
structure MotorboatPath where
  start : Point
  turn : Point
  destination : Point

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Checks if a line segment intersects with the Apollonius circle -/
def intersectsCircle (p q : Point) (circle : ApolloniusCircle) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    distance (Point.mk (p.x + t * (q.x - p.x)) (p.y + t * (q.y - p.y))) circle.center ≤ circle.radius

/-- Calculates the length of the motorboat's path -/
noncomputable def pathLength (path : MotorboatPath) : ℝ :=
  distance path.start path.turn + distance path.turn path.destination

theorem optimal_motorboat_path (m w : Point) (circle : ApolloniusCircle) :
  ∃ (path : MotorboatPath),
    path.start = m ∧
    path.destination = w ∧
    ¬intersectsCircle path.start path.turn circle ∧
    ¬intersectsCircle path.turn path.destination circle ∧
    (distance path.start path.turn)^2 + (distance path.turn path.destination)^2 = (distance m w)^2 ∧
    pathLength path = (2 * Real.sqrt 2 + 1) / 3 * distance m w ∧
    ∀ (other_path : MotorboatPath),
      other_path.start = m ∧
      other_path.destination = w ∧
      ¬intersectsCircle other_path.start other_path.turn circle ∧
      ¬intersectsCircle other_path.turn other_path.destination circle ∧
      (distance other_path.start other_path.turn)^2 + (distance other_path.turn other_path.destination)^2 = (distance m w)^2 →
      pathLength other_path ≥ pathLength path :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_motorboat_path_l678_67852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_cos_eq_zero_l678_67809

/-- A point is a symmetry center of cosine if and only if its x-coordinate is of the form π/2 + kπ for some integer k. -/
def is_symmetry_center_of_cos (m : ℝ) : Prop :=
  ∃ k : ℤ, m = Real.pi / 2 + k * Real.pi

/-- The cosine function -/
noncomputable def f (x : ℝ) : ℝ := Real.cos x

/-- If m is the x-coordinate of a symmetry center of cos x, then cos m = 0 -/
theorem symmetry_center_cos_eq_zero (m : ℝ) (h : is_symmetry_center_of_cos m) : f m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_cos_eq_zero_l678_67809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_graph_with_both_chromatic_numbers_le_two_l678_67860

/-- The chromatic number of a graph. -/
noncomputable def chromaticNumber {V : Type} (G : SimpleGraph V) : ℕ := sorry

/-- The complement of a graph. -/
def complement {V : Type} (G : SimpleGraph V) : SimpleGraph V := sorry

theorem no_graph_with_both_chromatic_numbers_le_two 
  {V : Type} (G : SimpleGraph V) [Fintype V] (h_size : Fintype.card V ≥ 3) :
  ¬(chromaticNumber G ≤ 2 ∧ chromaticNumber (complement G) ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_graph_with_both_chromatic_numbers_le_two_l678_67860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_point_p_distance_isosceles_trapezoid_point_p_existence_l678_67848

/-- IsoscelesTrapezoid represents an isosceles trapezoid with parallel sides a and b, and height m -/
structure IsoscelesTrapezoid where
  a : ℝ
  b : ℝ
  m : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < m

/-- Point P on the axis of symmetry of the trapezoid -/
noncomputable def P (t : IsoscelesTrapezoid) : ℝ := t.m / 2 - Real.sqrt ((t.m^2 - t.a * t.b) / 4)

theorem isosceles_trapezoid_point_p_distance (t : IsoscelesTrapezoid) :
  P t = t.m / 2 - Real.sqrt ((t.m^2 - t.a * t.b) / 4) :=
by sorry

theorem isosceles_trapezoid_point_p_existence (t : IsoscelesTrapezoid) :
  t.m^2 ≥ t.a * t.b ↔ P t ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_point_p_distance_isosceles_trapezoid_point_p_existence_l678_67848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l678_67891

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x - 1)

noncomputable def g (x : ℝ) : ℝ := sorry

theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, f a x = f a (-x) → x = 0) ∧
    (∀ x, f a x = -f a (-x)) ∧
    (∀ x, g (x + 2) = -g x) ∧
    (∀ x, 0 < x ∧ x ≤ 2 → g x = f a x) ∧
    (a = 1) ∧
    (g (-5) = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l678_67891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l678_67862

/-- Given a rectangle ABCD with known side lengths, calculate the length of the diagonal AC -/
theorem rectangle_diagonal_length (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 26) (h3 : AD = 9) :
  ∃ AC : ℝ, abs (AC - 34.2) < 0.1 := by
  sorry

#check rectangle_diagonal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l678_67862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_pigeonhole_l678_67889

theorem orange_crates_pigeonhole (total_crates min_oranges max_oranges : ℕ) :
  total_crates = 150 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ (n : ℕ), n ≥ 5 ∧ ∃ (orange_count : ℕ), 
    orange_count ≥ min_oranges ∧ 
    orange_count ≤ max_oranges ∧
    (Finset.filter (λ crate ↦ crate = orange_count) (Finset.range total_crates)).card ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_pigeonhole_l678_67889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l678_67857

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l678_67857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monogram_count_l678_67872

theorem monogram_count : ∃ (n : ℕ), n = Nat.choose 26 3 ∧ n = 2600 := by
  use 2600
  constructor
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monogram_count_l678_67872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_purchase_difference_l678_67896

/-- The difference between the total cost of cookies (including tax) and Diane's money -/
theorem cookie_purchase_difference (
  num_packages : ℕ)
  (cost_per_package : ℚ)
  (tax_rate : ℚ)
  (diane_money : ℚ)
  (h1 : num_packages = 3)
  (h2 : cost_per_package = 75 / 100)
  (h3 : tax_rate = 10 / 100)
  (h4 : diane_money = 55 / 100)
  : ℚ := by
  let total_cost_before_tax := num_packages * cost_per_package
  let tax_amount := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax_amount
  have h : total_cost_with_tax - diane_money = 193 / 100 := by sorry
  exact total_cost_with_tax - diane_money


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_purchase_difference_l678_67896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_double_heart_five_l678_67801

-- Define the heart function
def heart (x : ℕ) : ℕ :=
  if x < 10 then x else (x % 10) + heart (x / 10)

-- Define the set of two-digit numbers
def two_digit_numbers : Finset ℕ := Finset.range 90 |>.filter (λ x => x ≥ 10)

-- State the theorem
theorem count_double_heart_five :
  (two_digit_numbers.filter (λ x => heart (heart x) = 5)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_double_heart_five_l678_67801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_chord_theorem_l678_67899

-- Define the ellipse
def Ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def Foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  Ellipse a b F₁.1 F₁.2 ∧ Ellipse a b F₂.1 F₂.2

-- Define the line
def Line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the circle
def Circle (O F₁ F₂ : ℝ × ℝ) : Prop :=
  (F₁.1 - O.1)^2 + (F₁.2 - O.2)^2 = (F₂.1 - O.1)^2 + (F₂.2 - O.2)^2

theorem ellipse_and_chord_theorem (a b : ℝ) (F₁ F₂ : ℝ × ℝ) (O P A B : ℝ × ℝ) (k m lambda : ℝ) :
  a > b ∧ b > 0 →
  Foci F₁ F₂ a b →
  O = (0, 0) →
  P = (-1, Real.sqrt 2 / 2) →
  Ellipse a b P.1 P.2 →
  (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0 →
  Circle O F₁ F₂ →
  Line k m A.1 A.2 →
  Line k m B.1 B.2 →
  Ellipse a b A.1 A.2 →
  Ellipse a b B.1 B.2 →
  A ≠ B →
  A.1 * B.1 + A.2 * B.2 = lambda →
  3/4 ≤ lambda ∧ lambda ≤ 4/5 →
  (a^2 = 2 ∧ b^2 = 1) ∧
  (4 * Real.sqrt 2 / 5 ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
   Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt 6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_chord_theorem_l678_67899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leash_yard_fraction_l678_67836

/-- Represents a square yard with a leash tied to its center -/
structure LeashYard where
  side_length : ℝ
  leash_length : ℝ

/-- The fraction of the yard area that can be reached by the leash -/
noncomputable def reachable_fraction (yard : LeashYard) : ℝ :=
  (Real.pi * yard.leash_length ^ 2) / (yard.side_length ^ 2)

theorem leash_yard_fraction (yard : LeashYard) 
    (h₁ : yard.leash_length > 0)
    (h₂ : yard.side_length = 2 * yard.leash_length) : 
  reachable_fraction yard = Real.pi / 4 := by
  sorry

#check leash_yard_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leash_yard_fraction_l678_67836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_same_side_l678_67893

-- Define the sequences x_n and y_n
noncomputable def x : ℕ → ℝ := sorry
noncomputable def y : ℕ → ℝ := sorry

-- Define the conditions
axiom x_pos : ∀ n, x n > 0
axiom y_pos : ∀ n, y n > 0

axiom x_recurrence : ∀ n ≥ 1, x (n + 1) = Real.sqrt ((x n ^ 2 + x (n + 2) ^ 2) / 2)
axiom y_recurrence : ∀ n ≥ 1, y (n + 1) = ((Real.sqrt (y n) + Real.sqrt (y (n + 2))) / 2) ^ 2

axiom A1_A2018_distinct : (x 1, y 1) ≠ (x 2018, y 2018)

-- The theorem to prove
theorem points_on_same_side :
  ∀ i, 2 ≤ i ∧ i ≤ 2017 → (x i / y i) > (x 1 / y 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_same_side_l678_67893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_email_difference_l678_67874

theorem jack_email_difference 
  (morning_emails : Int)
  (afternoon_emails : Int)
  (h1 : morning_emails = 6)
  (h2 : afternoon_emails = 2) :
  morning_emails - afternoon_emails = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_email_difference_l678_67874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_fifth_power_of_three_prime_product_l678_67881

/-- Given a natural number N that is a product of three distinct primes,
    the number of distinct factors of N^5 is 216. -/
theorem factors_of_fifth_power_of_three_prime_product (N : ℕ) 
  (h1 : ∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ N = p * q * r) :
  (Finset.filter (· ∣ N^5) (Finset.range (N^5 + 1))).card = 216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_fifth_power_of_three_prime_product_l678_67881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_m_solvable_l678_67834

/-- Function f defined on (-π/2, π/2) -/
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < Real.pi/2 then Real.tan x / (Real.tan x + 1)
  else if x = 0 then 0
  else if -Real.pi/2 < x ∧ x < 0 then Real.tan x / (1 - Real.tan x)
  else 0  -- This case should never occur due to the domain restriction

/-- f is an odd function -/
axiom f_odd (x : ℝ) (h : -Real.pi/2 < x ∧ x < Real.pi/2) : f (-x) = -f x

/-- Theorem: f(x) = m has a solution in (-π/2, π/2) iff m ∈ (-1, 1) -/
theorem f_eq_m_solvable (m : ℝ) :
  (∃ x, -Real.pi/2 < x ∧ x < Real.pi/2 ∧ f x = m) ↔ -1 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_m_solvable_l678_67834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l678_67839

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem ellipse_foci_distance 
  (x y xf1 yf1 xf2 yf2 : ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_foci : ∀ (x' y' : ℝ), is_on_ellipse x' y' → 
    distance x' y' xf1 yf1 + distance x' y' xf2 yf2 = 6) 
  (h_dist : distance x y xf1 yf1 = 2) : 
  distance x y xf2 yf2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l678_67839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prescribed_given_allergic_l678_67800

/-- Probability that a patient is prescribed Undetenin -/
noncomputable def P_prescribed : ℝ := 0.10

/-- Probability that a patient is allergic to Undetenin -/
noncomputable def P_allergic : ℝ := 0.04

/-- Probability that a patient is allergic given that they are prescribed Undetenin -/
noncomputable def P_allergic_given_prescribed : ℝ := 0.02

/-- The probability that a patient allergic to Undetenin is prescribed the drug -/
noncomputable def P_prescribed_given_allergic : ℝ := 
  P_allergic_given_prescribed * P_prescribed / P_allergic

theorem probability_prescribed_given_allergic :
  P_prescribed_given_allergic = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prescribed_given_allergic_l678_67800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_proof_l678_67838

/-- The radius of a circle that is internally tangent to four identical circles of radius 1 arranged in a square. -/
noncomputable def large_circle_radius : ℝ := Real.sqrt 2 + 1

/-- Theorem stating that the radius of a circle internally tangent to four unit circles in a square arrangement is √2 + 1. -/
theorem large_circle_radius_proof :
  let small_circle_radius : ℝ := 1
  let small_circle_distance : ℝ := 2 * small_circle_radius
  let square_diagonal : ℝ := small_circle_distance * Real.sqrt 2
  large_circle_radius = (square_diagonal / 2) + small_circle_radius :=
by
  -- Proof goes here
  sorry

#check large_circle_radius_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_proof_l678_67838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convert_7_10_2_zimeters_to_zimimimeters_convert_10_6_zimeters_to_zimimimeters_transform_10_2017_to_10_11_l678_67819

-- Define the calculator operations
noncomputable def multiply_by_10_8 (x : ℝ) : ℝ := x * (10 ^ 8)
noncomputable def divide_by_10_5 (x : ℝ) : ℝ := x / (10 ^ 5)

-- Define the conversion factor from Zimeters to Zimimimeters
noncomputable def zimeter_to_zimimimeter (x : ℝ) : ℝ := x * (10 ^ 7)

-- Theorem for part a
theorem convert_7_10_2_zimeters_to_zimimimeters :
  ∃ (n m : ℕ), (fun x => (multiply_by_10_8^[n] ∘ divide_by_10_5^[m]) x) (7 * 10^2) = 7 * 10^9 := by
  sorry

-- Theorem for part b
theorem convert_10_6_zimeters_to_zimimimeters :
  ∃ (n m : ℕ), (fun x => (multiply_by_10_8^[n] ∘ divide_by_10_5^[m]) x) 1000 = 10^13 := by
  sorry

-- Theorem for part c
theorem transform_10_2017_to_10_11 :
  ∃ (n m : ℕ), (fun x => (multiply_by_10_8^[n] ∘ divide_by_10_5^[m]) x) (10^2017) = 10^11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convert_7_10_2_zimeters_to_zimimimeters_convert_10_6_zimeters_to_zimimimeters_transform_10_2017_to_10_11_l678_67819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_power_sequence_l678_67830

def sequenceQ (q₀ : ℕ) : ℕ → ℕ
  | 0 => q₀
  | n + 1 => (sequenceQ q₀ n - 1)^3 + 3

def is_prime_power (n : ℕ) : Prop := ∃ p k, Nat.Prime p ∧ n = p^k

theorem max_prime_power_sequence :
  ∀ q₀ : ℕ+, ∃ n : ℕ,
    (∀ i : ℕ, i ≤ n → is_prime_power (sequenceQ q₀ i)) ∧
    (∀ m : ℕ, m > n → ¬(∀ i : ℕ, i ≤ m → is_prime_power (sequenceQ q₀ i))) ∧
    n = 2 :=
by
  sorry

#check max_prime_power_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_power_sequence_l678_67830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_l678_67843

theorem tan_double_angle_specific (α : ℝ) 
  (h1 : Real.sin α = 2 * Real.sqrt 5 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan (2 * α) = -4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_specific_l678_67843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_first_half_score_l678_67897

def basketball_game (a r b d : ℝ) : Prop :=
  -- Team Fusion's scores form a geometric sequence
  let fusion := [a, a*r, a*r^2, a*r^3]
  -- Team Blitz's scores form an arithmetic sequence
  let blitz := [b, b+d, b+2*d, b+3*d]
  -- Game tied at end of first quarter
  a = b
  -- Game tied at end of third quarter
  ∧ (fusion[0]! + fusion[1]! + fusion[2]! = blitz[0]! + blitz[1]! + blitz[2]!)
  -- Team Fusion wins by 3 points
  ∧ (fusion.sum = blitz.sum + 3)
  -- Neither team scores more than 100 points
  ∧ fusion.sum ≤ 100 ∧ blitz.sum ≤ 100
  -- r > 1 to ensure increasing sequence for Team Fusion
  ∧ r > 1

theorem basketball_game_first_half_score 
  (a r b d : ℝ) (h : basketball_game a r b d) : 
  (a + a*r + b + (b+d)) = 28 := by
  sorry

#check basketball_game_first_half_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_first_half_score_l678_67897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meetings_count_l678_67867

/-- Represents the number of meetings between Michael and the garbage truck -/
def number_of_meetings : ℕ := 5

/-- Michael's walking speed in feet per second -/
noncomputable def michael_speed : ℝ := 5

/-- Distance between trash pails in feet -/
noncomputable def pail_distance : ℝ := 200

/-- Garbage truck's speed in feet per second -/
noncomputable def truck_speed : ℝ := 10

/-- Time the truck stops at each pail in seconds -/
noncomputable def truck_stop_time : ℝ := 30

/-- Initial distance between Michael and the truck in feet -/
noncomputable def initial_distance : ℝ := 200

/-- Time for one complete cycle of the truck's movement (moving + stopping) -/
noncomputable def truck_cycle_time : ℝ := pail_distance / truck_speed + truck_stop_time

/-- Distance Michael travels during one truck cycle -/
noncomputable def michael_distance_per_cycle : ℝ := michael_speed * truck_cycle_time

/-- Distance the truck travels during one cycle (equal to pail_distance) -/
noncomputable def truck_distance_per_cycle : ℝ := pail_distance

/-- Change in distance between Michael and the truck during one cycle -/
noncomputable def distance_change_per_cycle : ℝ := truck_distance_per_cycle - michael_distance_per_cycle

theorem meetings_count :
  number_of_meetings = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meetings_count_l678_67867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l678_67845

/-- Calculates the cost of fencing a rectangular field -/
noncomputable def fencing_cost (ratio_width : ℝ) (ratio_length : ℝ) (area : ℝ) (cost_per_metre : ℝ) : ℝ :=
  let width := Real.sqrt ((ratio_width * area) / (ratio_length * ratio_width))
  let length := (ratio_length / ratio_width) * width
  let perimeter := 2 * (width + length)
  perimeter * cost_per_metre

/-- Theorem stating the cost of fencing the given rectangular field -/
theorem fencing_cost_theorem :
  fencing_cost 3 4 9408 0.25 = 98 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l678_67845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_area_l678_67884

/-- A regular dodecagon inscribed in a circle with radius r has an area of 3r^2 -/
theorem dodecagon_area (r : ℝ) (r_pos : r > 0) : 
  (12 : ℝ) * ((1 / 2) * r^2 * Real.sin (2 * Real.pi / 24)) = 3 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_area_l678_67884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l678_67807

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-1 + 4*I) 3 (1 + I) = 5 - 5*I := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l678_67807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_op_value_l678_67815

-- Define the custom operation ⊗
noncomputable def vector_op (a b : ℝ × ℝ) (θ : ℝ) : ℝ :=
  (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) / Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))) * Real.cos θ

-- Theorem statement
theorem vector_op_value (a b : ℝ × ℝ) (θ : ℝ) :
  a ≠ (0, 0) →
  b ≠ (0, 0) →
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) ≥ Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) →
  0 < θ →
  θ < Real.pi / 4 →
  (∃ (n m : ℕ), vector_op a b θ = n / 2 ∧ vector_op b a θ = m / 2) →
  vector_op a b θ = 3 / 2 := by
  sorry

#check vector_op_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_op_value_l678_67815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_for_henrietta_house_l678_67805

/-- Calculates the number of gallons of paint needed to paint a house --/
noncomputable def paint_needed (living_room_area : ℝ) (num_bedrooms : ℕ) (bedroom_area : ℝ) (paint_coverage : ℝ) : ℝ :=
  let total_area := living_room_area + (num_bedrooms : ℝ) * bedroom_area
  total_area / paint_coverage

/-- Theorem: Given the specified conditions, 3 gallons of paint are needed --/
theorem paint_needed_for_henrietta_house : 
  paint_needed 600 3 400 600 = 3 := by
  unfold paint_needed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_for_henrietta_house_l678_67805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_and_n_l678_67863

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- State the theorem
theorem intersection_implies_m_and_n (m n : ℝ) : 
  A ∩ B m = Set.Ioo (-1) n → m = -1 ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_and_n_l678_67863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_of_9_equals_2_l678_67820

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem inverse_f_of_9_equals_2 :
  ∃ (f_inv : ℝ → ℝ), Function.RightInverse f_inv f ∧ f_inv 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_of_9_equals_2_l678_67820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ages_and_events_when_john_is_50_l678_67854

/-- Represents the ages and events of John, Alice, and Mike -/
structure FamilyAges where
  john_age : ℕ
  alice_age : ℕ
  mike_age : ℕ
  years_to_alice_marriage : ℕ
  years_to_john_college : ℕ
  years_to_mike_college : ℕ

/-- Theorem stating the ages and time passed when John is 50 -/
theorem ages_and_events_when_john_is_50 (f : FamilyAges)
  (h1 : f.john_age = 10)
  (h2 : f.alice_age = 2 * f.john_age)
  (h3 : f.mike_age = f.alice_age - 4)
  (h4 : f.years_to_alice_marriage = 5)
  (h5 : f.years_to_john_college = 2)
  (h6 : f.years_to_mike_college = 3) :
  let john_at_50 := 50
  let alice_at_john_50 := john_at_50 + (f.alice_age - f.john_age)
  let mike_at_john_50 := john_at_50 + (f.mike_age - f.john_age)
  let years_since_alice_marriage := john_at_50 - (f.alice_age + f.years_to_alice_marriage)
  let years_since_john_college := john_at_50 - (f.john_age + f.years_to_john_college)
  let years_since_mike_college := john_at_50 - (f.mike_age + f.years_to_mike_college)
  alice_at_john_50 = 60 ∧
  mike_at_john_50 = 56 ∧
  years_since_alice_marriage = 25 ∧
  years_since_john_college = 38 ∧
  years_since_mike_college = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ages_and_events_when_john_is_50_l678_67854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l678_67835

noncomputable def vector_a (α : Real) : Real × Real := (Real.sin α, Real.sqrt 2 / 2)

theorem cos_2α_value (α : Real) :
  (vector_a α).1^2 + (vector_a α).2^2 = (Real.sqrt 3 / 2)^2 →
  Real.cos (2 * α) = 1 / 2 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l678_67835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_value_l678_67850

-- Define the condition
axiom cos_sum_eq_sum_cos (α β : Real) : Real.cos (α + β) = Real.cos α + Real.cos β

-- Define the theorem to be proved
theorem max_cos_value :
  ∃ (max_cos : Real), (∀ α : Real, Real.cos α ≤ max_cos) ∧ (max_cos = Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_value_l678_67850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swallow_distance_equals_total_distance_l678_67894

/-- The distance flown by a swallow between two trains -/
noncomputable def swallow_distance (tgv_speed intercite_speed swallow_speed distance : ℝ) : ℝ :=
  (swallow_speed * distance) / (tgv_speed + intercite_speed)

theorem swallow_distance_equals_total_distance 
  (tgv_speed intercite_speed swallow_speed distance : ℝ) 
  (h1 : tgv_speed > 0)
  (h2 : intercite_speed > 0)
  (h3 : swallow_speed > tgv_speed)
  (h4 : distance > 0) :
  swallow_distance tgv_speed intercite_speed swallow_speed distance = distance :=
by
  sorry

#check swallow_distance_equals_total_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swallow_distance_equals_total_distance_l678_67894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_simplification_l678_67804

noncomputable def abs_2x_minus_1 (x : ℝ) : ℝ :=
  if x ≥ (1/2 : ℝ) then 2*x - 1 else 1 - 2*x

noncomputable def abs_x_minus_1_plus_abs_x_minus_3 (x : ℝ) : ℝ :=
  if x < (1 : ℝ) then 4 - 2*x
  else if x < (3 : ℝ) then 2
  else 2*x - 4

noncomputable def abs_abs_x_minus_1_minus_2_plus_abs_x_plus_1 (x : ℝ) : ℝ :=
  if x ≥ (3 : ℝ) then 2*x - 2
  else if x ≥ (1 : ℝ) then 4
  else if x ≥ (-1 : ℝ) then 2*x + 2
  else -2*x - 2

theorem absolute_value_simplification :
  ∀ x : ℝ,
    (|2*x - 1| = abs_2x_minus_1 x) ∧
    (|x - 1| + |x - 3| = abs_x_minus_1_plus_abs_x_minus_3 x) ∧
    (|(|x - 1| - 2)| + |x + 1| = abs_abs_x_minus_1_minus_2_plus_abs_x_plus_1 x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_simplification_l678_67804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l678_67846

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l678_67846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_properties_l678_67840

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a / x - x + a * Real.log x

-- Theorem statement
theorem extreme_points_properties (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂)) →
  (a > 4 ∧
   (∃ m : ℝ, m = -Real.exp 2 ∧
    ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 
      (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
      f a x₁ + f a x₂ - 3 * a ≥ m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_properties_l678_67840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l678_67824

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the distance between a point and a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Calculates the distance between a point and the x-axis -/
def distancePointToXAxis (p : Point) : ℝ :=
  abs p.y

/-- Represents the parabola y = (1/4)x^2 -/
def isOnParabola (p : Point) : Prop :=
  p.y = (1/4) * p.x^2

/-- The main theorem -/
theorem min_distance_sum (l : Line) :
  l.a = 1 ∧ l.b = 2 ∧ l.c = 4 →
  ∃ (minDist : ℝ), minDist = (6 * Real.sqrt 5) / 5 - 1 ∧
    ∀ (p : Point), isOnParabola p →
      distancePointToLine p l + distancePointToXAxis p ≥ minDist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l678_67824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_semicircle_with_perimeter_126_l678_67870

/-- The radius of a semi-circle with given perimeter -/
noncomputable def radius_of_semicircle (perimeter : ℝ) : ℝ :=
  perimeter / (Real.pi + 2)

/-- Theorem: The radius of a semi-circle with perimeter 126 cm is 126 / (π + 2) cm -/
theorem radius_of_semicircle_with_perimeter_126 :
  radius_of_semicircle 126 = 126 / (Real.pi + 2) := by
  -- Unfold the definition of radius_of_semicircle
  unfold radius_of_semicircle
  -- The result follows directly from the definition
  rfl

/-- Approximation of the radius for a semi-circle with perimeter 126 cm -/
def approx_radius : ℚ :=
  126 / (314159 / 100000 + 2)

#eval approx_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_semicircle_with_perimeter_126_l678_67870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_inequality_l678_67814

/-- A non-negative differentiable function satisfying a certain differential inequality -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  nonneg : ∀ x ∈ domain, f x ≥ 0
  diff : DifferentiableOn ℝ f domain
  ineq : ∀ x ∈ domain, x * (deriv f x) ≤ -f x

/-- The main theorem about the behavior of special functions -/
theorem special_function_inequality (φ : SpecialFunction) (a b : ℝ) 
    (ha : a ∈ φ.domain) (hb : b ∈ φ.domain) (hab : a < b) :
    a * φ.f a ≥ b * φ.f b :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_inequality_l678_67814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l678_67853

/-- Calculates the length of a platform given subway parameters -/
theorem platform_length_calculation (subway_speed_cm : ℝ) (subway_length : ℝ) (time : ℝ) :
  subway_speed_cm = 288 →
  subway_length = 20 →
  time = 25 →
  (subway_speed_cm / 100 * time - subway_length) = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l678_67853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_l678_67851

theorem largest_and_smallest (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  (∀ z ∈ ({x, x*y, x*y^2} : Set ℝ), x*y ≥ z) ∧ (∀ z ∈ ({x, x*y, x*y^2} : Set ℝ), x ≤ z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_l678_67851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_intersection_theorem_l678_67875

/-- The set of possible numbers of intersections for four distinct lines in a plane -/
def intersection_counts : Set ℕ := {0, 1, 3, 4, 5, 6}

/-- A line in a plane -/
structure Line where
  -- We'll leave this empty for now, as we don't need to specify its internals
  mk :: -- This allows us to create a Line without arguments

/-- Four distinct lines in a plane -/
structure FourLines where
  line1 : Line
  line2 : Line
  line3 : Line
  line4 : Line
  distinct : line1 ≠ line2 ∧ line1 ≠ line3 ∧ line1 ≠ line4 ∧ 
             line2 ≠ line3 ∧ line2 ≠ line4 ∧ line3 ≠ line4

/-- The number of intersections between four lines -/
def num_intersections (lines : FourLines) : ℕ :=
  sorry

/-- Theorem stating that the number of intersections of four distinct lines is in the set intersection_counts -/
theorem four_lines_intersection_theorem (lines : FourLines) :
  num_intersections lines ∈ intersection_counts :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_intersection_theorem_l678_67875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_periodic_l678_67879

-- Define the concept of a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define the concept of a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (g = Real.sin ∨ g = Real.cos) ∧ ∃ a b : ℝ, ∀ x, f x = a * g (b * x) + b

-- State the theorem
theorem cos_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (λ x => Real.cos x) →
  IsPeriodic (λ x => Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_is_periodic_l678_67879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_calculation_l678_67864

/-- Represents the number of students in a school year -/
abbrev StudentsInYear := Nat

/-- Represents the number of sampled students from a school year -/
abbrev SampledStudents := Nat

/-- Calculates the total number of students in the school given the sampling information -/
def totalStudents (totalSampled : SampledStudents) (secondYearSampled : SampledStudents) 
  (thirdYearSampled : SampledStudents) (firstYearTotal : StudentsInYear) : StudentsInYear :=
  (totalSampled * firstYearTotal) / (totalSampled - secondYearSampled - thirdYearSampled)

/-- Theorem stating that the total number of students is 3200 given the sampling information -/
theorem total_students_calculation :
  let totalSampled : SampledStudents := 60
  let secondYearSampled : SampledStudents := 20
  let thirdYearSampled : SampledStudents := 25
  let firstYearTotal : StudentsInYear := 800
  totalStudents totalSampled secondYearSampled thirdYearSampled firstYearTotal = 3200 := by
  sorry

#eval totalStudents 60 20 25 800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_calculation_l678_67864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_implies_a_bound_l678_67876

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 2) * x else (1/2)^x - 1

theorem monotonically_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 13/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_implies_a_bound_l678_67876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_to_march_ratio_l678_67858

/-- The number of window screens sold in January. -/
def january_sales : ℕ := sorry

/-- The number of window screens sold in February. -/
def february_sales : ℕ := sorry

/-- The number of window screens sold in March. -/
def march_sales : ℕ := sorry

/-- February sales are twice January sales. -/
axiom february_twice_january : february_sales = 2 * january_sales

/-- March sales are 8800. -/
axiom march_is_8800 : march_sales = 8800

/-- Total sales from January to March are 12100. -/
axiom total_sales : january_sales + february_sales + march_sales = 12100

/-- The ratio of February sales to March sales is 1:4. -/
theorem february_to_march_ratio : 
  ∃ (k : ℕ), k ≠ 0 ∧ february_sales * 4 = march_sales * 1 * k := by
  sorry

#check february_to_march_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_to_march_ratio_l678_67858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l678_67866

theorem parallelepiped_volume (k : ℝ) : k > 0 →
  |Matrix.det ![![3, 3, 4], ![1, k, 3], ![2, 2, k]]| = 18 →
  k = 2/3 ∨ k = 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l678_67866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l678_67828

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * Real.sin x - Real.sqrt 2) + Real.sqrt (1 - 2 * Real.cos x)

def domain (x : ℝ) : Prop :=
  ∃ k : ℤ, Real.pi / 3 + 2 * k * Real.pi ≤ x ∧ x < 3 * Real.pi / 4 + 2 * k * Real.pi

theorem function_domain :
  ∀ x : ℝ, (2 * Real.sin x - Real.sqrt 2 > 0 ∧ 1 - 2 * Real.cos x ≥ 0) ↔ domain x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l678_67828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_product_l678_67849

/-- The number of ones in the first factor -/
def n : ℕ := 2007

/-- The second factor -/
def m : ℕ := 2007

/-- The first factor: a number consisting of n ones -/
def first_factor : ℕ := (10^n - 1) / 9

/-- The product of the two factors -/
def product : ℕ := first_factor * m

/-- The sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ :=
  if k = 0 then 0 else k % 10 + sum_of_digits (k / 10)

theorem sum_of_digits_product : sum_of_digits product = 19035 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_product_l678_67849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_total_time_correct_l678_67859

/-- Represents the parameters of the ball's motion --/
structure BallMotion where
  g : ℝ  -- gravitational acceleration
  H : ℝ  -- initial height and height after first collision
  H' : ℝ  -- height after second collision
  v₀ : ℝ  -- initial velocity

/-- Calculates the total time until the ball comes to rest --/
noncomputable def totalTime (b : BallMotion) : ℝ :=
  (Real.sqrt (2 * b.g * b.H + b.v₀^2) - b.v₀) / b.g +
  2 * Real.sqrt (2 * b.H / b.g) / (1 - Real.sqrt (2 * b.g * b.H / (2 * b.g * b.H + b.v₀^2)))

/-- Theorem stating the relationship between initial velocity and heights --/
theorem initial_velocity (b : BallMotion) :
  b.v₀ = Real.sqrt (2 * b.g * b.H * (b.H / b.H' - 1)) :=
by sorry

/-- Theorem for the total time until the ball comes to rest --/
theorem total_time_correct (b : BallMotion) :
  totalTime b = (Real.sqrt (2 * b.g * b.H + b.v₀^2) - b.v₀) / b.g +
                2 * Real.sqrt (2 * b.H / b.g) / (1 - Real.sqrt (2 * b.g * b.H / (2 * b.g * b.H + b.v₀^2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_total_time_correct_l678_67859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_bar_breaks_l678_67847

/-- Represents a chocolate bar with a given number of pieces -/
structure ChocolateBar where
  pieces : ℕ

/-- Represents the process of breaking a chocolate bar -/
def breakBar (bar : ChocolateBar) : ChocolateBar :=
  { pieces := bar.pieces + 1 }

/-- The number of breaks required to separate a chocolate bar into individual pieces -/
def breaks_required (initial : ChocolateBar) (final : ChocolateBar) : ℕ :=
  final.pieces - initial.pieces

/-- Theorem: The number of breaks required to separate a 40-piece chocolate bar
    into individual pieces is 39 -/
theorem chocolate_bar_breaks :
  let initial_bar := ChocolateBar.mk 1
  let final_bar := ChocolateBar.mk 40
  breaks_required initial_bar final_bar = 39 := by
  sorry

#check chocolate_bar_breaks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_bar_breaks_l678_67847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_min_area_normal_chord_l678_67822

/-- A parabola is defined by the equation y = ax^2, where a is a positive constant. -/
def Parabola (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = a * p.1^2 ∧ a > 0}

/-- A normal chord is a line segment that is perpendicular to the tangent line of the parabola at a point. -/
def NormalChord (a : ℝ) (p : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 = -1 / (2 * a * p) * q.1 + (a * p^2 + p / (2 * a * p))}

/-- The area of the segment cut off by a normal chord. -/
noncomputable def SegmentArea (a : ℝ) (p : ℝ) : ℝ := sorry

/-- There exists a unique position for a normal chord that minimizes the area of the segment it cuts off from the parabola. -/
theorem exists_unique_min_area_normal_chord (a : ℝ) (h : a > 0) :
  ∃! p : ℝ, IsLocalMin (SegmentArea a) p ∧ ∃ y : ℝ, (p, y) ∈ Parabola a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_min_area_normal_chord_l678_67822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l678_67802

noncomputable def F (a x : ℝ) : ℝ := min (2 * |x - 1|) (x^2 - 2*a*x + 4*a - 2)

noncomputable def m (a : ℝ) : ℝ :=
  if a ≤ 2 + Real.sqrt 2 then 0 else -a^2 + 4*a - 2

theorem F_properties (a : ℝ) (ha : a ≥ 3) :
  (∀ x, F a x ≥ m a) ∧
  (∃ x, F a x = m a) ∧
  (∀ x, F a x = x^2 - 2*a*x + 4*a - 2 ↔ 2 ≤ x ∧ x ≤ 2*a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l678_67802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_selling_price_theorem_l678_67886

/-- Calculates the selling price for a desired profit percentage given the initial selling price and loss percentage. -/
noncomputable def calculate_new_selling_price (initial_selling_price : ℝ) (loss_percentage : ℝ) (desired_profit_percentage : ℝ) : ℝ :=
  let cost_price := initial_selling_price / (1 - loss_percentage / 100)
  cost_price * (1 + desired_profit_percentage / 100)

/-- Theorem stating that for a selling price of 11 Rs/kg resulting in a 10% loss,
    the selling price for a 10% profit is approximately 13.44 Rs/kg. -/
theorem mango_selling_price_theorem :
  let initial_selling_price := (11 : ℝ)
  let loss_percentage := (10 : ℝ)
  let desired_profit_percentage := (10 : ℝ)
  let new_selling_price := calculate_new_selling_price initial_selling_price loss_percentage desired_profit_percentage
  abs (new_selling_price - 13.44) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_selling_price_theorem_l678_67886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_S_l678_67829

noncomputable def f (x : ℝ) : ℝ := (3*x + 4) / (x + 3)

def S : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ f x = y}

theorem bounds_of_S :
  ∃ (m M : ℝ),
    (∀ y ∈ S, m ≤ y ∧ y < M) ∧
    m ∈ S ∧
    M ∉ S ∧
    m = 4/3 ∧
    M = 3 := by
  sorry

#check bounds_of_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_S_l678_67829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_count_is_33_l678_67883

/-- A point on a discrete 2D grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- A rectangle defined by its top-left and bottom-right corners --/
structure Rectangle where
  topLeft : GridPoint
  bottomRight : GridPoint

/-- The dimensions of our grid --/
def gridWidth : Nat := 4
def gridHeight : Nat := 3

/-- Function to check if a point is within the grid --/
def isValidPoint (p : GridPoint) : Bool :=
  p.x < gridWidth && p.y < gridHeight

/-- Function to check if a rectangle is valid (all points within grid and correct orientation) --/
def isValidRectangle (r : Rectangle) : Bool :=
  isValidPoint r.topLeft && isValidPoint r.bottomRight &&
  r.topLeft.x <= r.bottomRight.x && r.topLeft.y >= r.bottomRight.y

/-- Function to generate all possible rectangles --/
def generateAllRectangles : List Rectangle :=
  let points := List.range gridWidth >>= fun x =>
                List.range gridHeight >>= fun y =>
                [GridPoint.mk x y]
  points >>= fun tl =>
  points >>= fun br =>
  [Rectangle.mk tl br]

/-- Function to count all valid rectangles in the grid --/
def countRectangles : Nat :=
  (generateAllRectangles.filter isValidRectangle).length

/-- The main theorem stating that the number of rectangles is 33 --/
theorem rectangle_count_is_33 : countRectangles = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_count_is_33_l678_67883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_return_to_start_l678_67873

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the possible movements -/
inductive Move
  | up
  | down
  | right
  | left

/-- Applies a move to a point -/
def applyMove (p : Point) (m : Move) : Point :=
  match m with
  | Move.up => ⟨p.x, p.y + 2 * p.x⟩
  | Move.down => ⟨p.x, p.y - 2 * p.x⟩
  | Move.right => ⟨p.x + 2 * p.y, p.y⟩
  | Move.left => ⟨p.x - 2 * p.y, p.y⟩

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to a point -/
def applyMoveSequence (p : Point) (ms : MoveSequence) : Point :=
  ms.foldl applyMove p

/-- The starting point -/
noncomputable def startPoint : Point := ⟨1, Real.sqrt 2⟩

/-- Theorem: It is impossible to return to the starting point -/
theorem no_return_to_start (ms : MoveSequence) : 
  applyMoveSequence startPoint ms ≠ startPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_return_to_start_l678_67873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_common_ratio_l678_67826

/-- The common ratio of the geometric series 4/7 + 16/49 + 64/343 + ... is 4/7 -/
theorem geometric_series_common_ratio : 
  let a : ℕ → ℚ := fun n => (4 : ℚ) / 7 * (4 : ℚ) / 7 ^ n
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) / a n = r ∧ r = 4 / 7 :=
by
  -- Define a and r
  let a : ℕ → ℚ := fun n => (4 : ℚ) / 7 * (4 : ℚ) / 7 ^ n
  let r : ℚ := 4 / 7

  -- Prove the existence of r
  use r

  -- Prove that for all n, a(n+1) / a(n) = r and r = 4/7
  intro n
  apply And.intro

  -- Prove a(n+1) / a(n) = r
  calc
    a (n + 1) / a n = ((4 : ℚ) / 7 * (4 : ℚ) / 7 ^ (n + 1)) / ((4 : ℚ) / 7 * (4 : ℚ) / 7 ^ n) := by rfl
    _ = (4 : ℚ) / 7 := by sorry

  -- Prove r = 4/7
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_common_ratio_l678_67826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l678_67855

def f (x : ℝ) := x^3 - 3*x^2 + 2

theorem min_value_f : 
  ∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧ 
  ∀ y ∈ Set.Icc (-1) 1, f x ≤ f y ∧
  f x = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l678_67855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_figures_overlap_l678_67861

/-- A plane figure -/
structure PlaneFigure where
  -- We'll leave this abstract for now
  mk ::

/-- Congruence relation between two plane figures -/
def isCongruent (a b : PlaneFigure) : Prop :=
  sorry

/-- Relation indicating two figures are in the same plane -/
def inSamePlane (a b : PlaneFigure) : Prop :=
  sorry

/-- Relation indicating two figures are not parallel -/
def notParallel (a b : PlaneFigure) : Prop :=
  sorry

/-- Rotation transformation -/
def rotate (f : PlaneFigure) (center : Point) (angle : Real) : PlaneFigure :=
  sorry

/-- Translation transformation -/
def translate (f : PlaneFigure) (v : Real × Real) : PlaneFigure :=
  sorry

/-- Main theorem -/
theorem congruent_figures_overlap 
  (a b : PlaneFigure) 
  (h1 : isCongruent a b) 
  (h2 : inSamePlane a b) 
  (h3 : notParallel a b) : 
  ∃ (center : Point) (angle : Real) (v : Real × Real),
    translate (rotate b center angle) v = a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_figures_overlap_l678_67861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_outputs_l678_67817

-- Program 1
def program1_loop : Nat × Nat :=
  let rec loop (i a : Nat) (fuel : Nat) : Nat × Nat :=
    if fuel = 0 then (i, a)
    else if i > 6 then (i, a)
    else loop (i + 1) (a + 1) (fuel - 1)
  loop 1 2 6

-- Program 2
def program2_loop : Nat × Nat :=
  let rec loop (i x : Nat) (fuel : Nat) : Nat × Nat :=
    if fuel = 0 then (i, x)
    else if x = 200 then (i, x)
    else loop (i + 1) (x + 10) (fuel - 1)
  loop 1 100 10

theorem program_outputs :
  (program1_loop = (7, 8)) ∧
  (program2_loop = (11, 200)) := by
  sorry

#eval program1_loop
#eval program2_loop

end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_outputs_l678_67817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_zero_l678_67868

theorem complex_expression_equals_zero :
  (2 : ℂ) * Complex.I^5 + (1 - Complex.I)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_zero_l678_67868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_intersection_l678_67811

/-- A type representing points on a circle -/
def CirclePoint : Type := Fin 2023

/-- The total number of points on the circle -/
def totalPoints : Nat := 2023

/-- A function to check if two points are adjacent on the circle -/
def adjacent (a b : CirclePoint) : Prop :=
  (a.val + 1) % totalPoints = b.val ∨ (b.val + 1) % totalPoints = a.val

/-- A type representing a chord on the circle -/
structure Chord where
  p1 : CirclePoint
  p2 : CirclePoint

/-- A function to check if two chords intersect -/
noncomputable def intersect (c1 c2 : Chord) : Prop := sorry

/-- A type representing a selection of 10 points from the circle -/
structure Selection where
  points : Finset CirclePoint
  a : CirclePoint
  b : CirclePoint
  c : CirclePoint
  d : CirclePoint
  distinct : points.card = 10
  contains : a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ d ∈ points
  ab_adjacent : adjacent a b

/-- Probability type (placeholder) -/
noncomputable def Prob (p : Prop) : ℚ := sorry

theorem probability_of_intersection (s : Selection) :
  Prob (intersect (Chord.mk s.a s.b) (Chord.mk s.c s.d)) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_intersection_l678_67811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chain_payment_possible_l678_67865

/-- Represents a piece of the chain --/
inductive ChainPiece
  | Single : ChainPiece  -- A single link
  | Double : ChainPiece  -- Two connected links
  | Triple : ChainPiece  -- Three connected links
deriving BEq, Repr

/-- Represents the state of the chain after cutting --/
structure CutChain where
  pieces : List ChainPiece
deriving Repr

/-- Represents a payment made with chain pieces --/
structure Payment where
  given : List ChainPiece
  returned : List ChainPiece
deriving Repr

/-- The proposition that a valid payment sequence exists --/
def valid_payment_sequence (chain : CutChain) : Prop :=
  ∃ (payments : List Payment),
    payments.length = 6 ∧
    (∀ p ∈ payments, p.given.length - p.returned.length = 1) ∧
    (chain.pieces.length = 3) ∧
    (chain.pieces.count ChainPiece.Single = 1) ∧
    (chain.pieces.count ChainPiece.Double = 1) ∧
    (chain.pieces.count ChainPiece.Triple = 1)

theorem chain_payment_possible :
  ∃ (chain : CutChain), valid_payment_sequence chain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chain_payment_possible_l678_67865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l678_67892

def a : Fin 3 → ℝ := ![-2, -3, 1]
def b : Fin 3 → ℝ := ![2, 0, 4]
def c : Fin 3 → ℝ := ![-4, -6, 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

def is_parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

def is_perpendicular (v w : Fin 3 → ℝ) : Prop :=
  dot_product v w = 0

theorem vector_relationships : is_parallel a c ∧ is_perpendicular a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l678_67892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_covering_unit_rectangles_l678_67827

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane --/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- Checks if a point is inside a rectangle --/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  r.x ≤ p.x ∧ p.x ≤ r.x + r.width ∧ r.y ≤ p.y ∧ p.y ≤ r.y + r.height

/-- The theorem to be proved --/
theorem points_covering_unit_rectangles :
  ∃ (points : Finset Point),
    (Finset.card points = 1965) ∧
    (∀ (p : Point), p ∈ points → p.x ≥ 0 ∧ p.x ≤ 15 ∧ p.y ≥ 0 ∧ p.y ≤ 15) ∧
    (∀ (r : Rectangle),
      r.x ≥ 0 ∧ r.x + r.width ≤ 15 ∧ r.y ≥ 0 ∧ r.y + r.height ≤ 15 →
      r.width * r.height = 1 →
      ∃ (p : Point), p ∈ points ∧ pointInRectangle p r) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_covering_unit_rectangles_l678_67827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_proportional_to_time_test_score_calculation_l678_67878

theorem score_proportional_to_time (total_points : ℝ) (time1 time2 score1 : ℝ) 
    (h1 : time1 > 0) (h2 : time2 > 0) (h3 : score1 > 0) :
  let score2 := (score1 * time2) / time1
  score2 = (score1 * time2) / time1 := by 
  simp

theorem test_score_calculation 
    (total_points : ℝ) (time1 time2 score1 : ℝ) 
    (h1 : total_points = 150) 
    (h2 : time1 = 5) 
    (h3 : time2 = 7) 
    (h4 : score1 = 90) :
  let score2 := (score1 * time2) / time1
  score2 = 126 := by
  simp
  rw [h2, h3, h4]
  norm_num

#eval (90 * 7) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_proportional_to_time_test_score_calculation_l678_67878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sqrt_two_equals_two_plus_sqrt_two_l678_67869

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (2 - x)

-- State the theorem
theorem f_sqrt_two_equals_two_plus_sqrt_two :
  f (Real.sqrt 2) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sqrt_two_equals_two_plus_sqrt_two_l678_67869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_temperature_difference_l678_67882

/-- The temperature difference between Seattle and Denver at noon -/
def P : ℝ := sorry

/-- The temperature in Denver at noon -/
def D : ℝ := sorry

/-- The temperature in Seattle at noon -/
def S : ℝ := D + P

/-- The temperature in Seattle at 3:00 PM -/
def S_3pm : ℝ := S + 2

/-- The temperature in Denver at 3:00 PM -/
def D_3pm : ℝ := D + 5

/-- The temperature in Seattle at 6:00 PM -/
def S_6pm : ℝ := S_3pm - 3

/-- The temperature in Denver at 6:00 PM -/
def D_6pm : ℝ := D_3pm + 2

theorem unique_temperature_difference : S_6pm = D_6pm → P = 8 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_temperature_difference_l678_67882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_joined_triangles_l678_67810

/-- Represents a triangle with three side lengths -/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- The set of triangles given in the problem -/
def triangles : List Triangle := [
  ⟨5, 8, 10⟩,
  ⟨5, 10, 12⟩,
  ⟨5, 8, 12⟩
]

/-- The theorem stating the maximum perimeter -/
theorem max_perimeter_of_joined_triangles :
  ∃ (joinedPerimeter : ℕ),
    joinedPerimeter = 49 ∧
    ∀ (p : ℕ),
      (∃ (join : List (Nat × Nat)),
        p = (List.sum (triangles.map perimeter) -
             2 * (List.sum (join.map (fun x => x.1)))) ∧
        join.length = 2 ∧
        ∀ (side : Nat × Nat), side ∈ join →
          ∃ (t1 t2 : Triangle),
            t1 ∈ triangles ∧ t2 ∈ triangles ∧ t1 ≠ t2 ∧
            (side.1 = t1.side1 ∨ side.1 = t1.side2 ∨ side.1 = t1.side3) ∧
            (side.1 = t2.side1 ∨ side.1 = t2.side2 ∨ side.1 = t2.side3)) →
      p ≤ joinedPerimeter :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_joined_triangles_l678_67810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l678_67890

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

def is_tangent_to_line (c : Circle) (l : Line) : Prop :=
  sorry -- Definition of circle being tangent to a line

def point_between (p q r : ℝ × ℝ) : Prop :=
  sorry -- Definition of a point being between two other points

noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p
  let (x2, y2) := q
  let (x3, y3) := r
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

theorem circle_triangle_area 
  (cP cQ cR : Circle) (l : Line) (P' Q' R' : ℝ × ℝ) :
  cP.radius = 2 →
  cQ.radius = 3 →
  cR.radius = 4 →
  is_tangent_to_line cP l →
  is_tangent_to_line cQ l →
  is_tangent_to_line cR l →
  point_between P' Q' R' →
  are_externally_tangent cP cQ →
  are_externally_tangent cQ cR →
  triangle_area cP.center cQ.center cR.center = 6 :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_l678_67890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_iff_k_eq_one_period_pi_not_iff_a_eq_three_perpendicular_min_value_greater_than_two_l678_67832

-- Define the function for the first statement
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (Real.cos (k * x))^2 - (Real.sin (k * x))^2

-- Define the period condition for the first statement
def has_period_pi (k : ℝ) : Prop :=
  ∀ x, f k (x + Real.pi) = f k x

-- First statement
theorem not_iff_k_eq_one_period_pi :
  ¬ (has_period_pi 1 ↔ ∀ k, has_period_pi k → k = 1) := by
  sorry

-- Define the lines for the second statement
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

-- Define the perpendicularity condition for the second statement
def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2, line1 a x1 y1 → line2 a x2 y2 →
    a * 3 + 2 * (a - 1) = 0

-- Second statement
theorem not_iff_a_eq_three_perpendicular :
  ¬ (perpendicular 3 ↔ ∀ a, perpendicular a → a = 3) := by
  sorry

-- Define the function for the third statement
noncomputable def g (x : ℝ) : ℝ := (x^2 + 4) / Real.sqrt (x^2 + 3)

-- Third statement
theorem min_value_greater_than_two :
  ∀ x, g x > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_iff_k_eq_one_period_pi_not_iff_a_eq_three_perpendicular_min_value_greater_than_two_l678_67832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_prediction_l678_67877

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ
  a : ℝ

/-- Calculates the y-value for a given x using a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.a

theorem regression_prediction
  (sample_center : Point)
  (model : LinearRegression)
  (h1 : sample_center.x = 5)
  (h2 : sample_center.y = 50)
  (h3 : model.b = 7)
  (h4 : predict model sample_center.x = sample_center.y) :
  predict model 10 = 85 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_prediction_l678_67877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_std_dev_l678_67880

noncomputable def sample : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / xs.length

noncomputable def stdDev (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem sample_std_dev : stdDev sample = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_std_dev_l678_67880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l678_67887

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  first_term_def : ∀ n : ℕ, a n = a 1 + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_sum_9 (seq : ArithmeticSequence) 
  (h : seq.a 2 + seq.a 3 + seq.a 10 = 9) : 
  S seq 9 = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l678_67887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_salon_profit_l678_67818

/-- Rebecca's hair salon financial calculation --/
theorem rebecca_salon_profit
  (haircut_price : ℕ)
  (perm_price : ℕ)
  (dye_price : ℕ)
  (extension_price : ℕ)
  (haircut_supply : ℕ)
  (dye_supply : ℕ)
  (extension_supply : ℕ)
  (num_haircuts : ℕ)
  (num_perms : ℕ)
  (num_dyes : ℕ)
  (num_extensions : ℕ)
  (tips : ℕ)
  (daily_expenses : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_price = 60 →
  extension_price = 80 →
  haircut_supply = 5 →
  dye_supply = 10 →
  extension_supply = 25 →
  num_haircuts = 5 →
  num_perms = 3 →
  num_dyes = 2 →
  num_extensions = 1 →
  tips = 75 →
  daily_expenses = 45 →
  (haircut_price * num_haircuts +
   perm_price * num_perms +
   dye_price * num_dyes +
   extension_price * num_extensions +
   tips) -
  (haircut_supply * num_haircuts +
   dye_supply * num_dyes +
   extension_supply * num_extensions) -
  daily_expenses = 430 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_salon_profit_l678_67818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_2B_minus_A_M_independent_of_x_l678_67844

-- Define variables
variable (x y : ℝ)

-- Define expressions A and B
def A (x y : ℝ) : ℝ := 4 * x^2 - 4 * x * y + y^2
def B (x y : ℝ) : ℝ := x^2 + x * y - 5 * y^2

-- Define expression M
def M (x y : ℝ) : ℝ := (2 * x^2 + 3 * x * y + 2 * y) - 2 * (x^2 + x + y * x + 1)

-- Theorem 1: Simplification of 2B - A
theorem simplify_2B_minus_A (x y : ℝ) :
  2 * B x y - A x y = -2 * x^2 + 6 * x * y - 11 * y^2 := by
  sorry

-- Theorem 2: Value of y when M is independent of x
theorem M_independent_of_x (y : ℝ) :
  (∀ x, deriv (fun x => M x y) x = 0) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_2B_minus_A_M_independent_of_x_l678_67844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenga_round_determination_l678_67895

theorem jenga_round_determination 
  (total_blocks : Nat)
  (players : Nat)
  (blocks_per_round : Nat)
  (rounds : Nat)
  (blocks_before_jess : Nat)
  (h1 : total_blocks = 54)
  (h2 : players = 5)
  (h3 : blocks_per_round = players)
  (h4 : rounds = 5)
  (h5 : blocks_before_jess = 28) :
  (total_blocks - blocks_before_jess) / blocks_per_round + 1 = 6 := by
  sorry

#check jenga_round_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenga_round_determination_l678_67895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l678_67803

/-- The lateral surface area of a cone with diameter 2 and height 2 is √5π. -/
theorem cone_lateral_surface_area : 
  ∀ (cone : ℝ → ℝ → ℝ),
  (∀ d h, cone d h = π * (d / 2) * (Real.sqrt ((d / 2)^2 + h^2))) →
  cone 2 2 = Real.sqrt 5 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l678_67803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_l678_67823

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ 
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem max_perimeter (t : Triangle) (h : triangle_conditions t) : 
  t.a + t.b + t.c ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_l678_67823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l678_67812

-- Define the hyperbola
def hyperbola (n : ℝ) (x y : ℝ) : Prop := x^2 / n - y^2 = 1

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the area of a triangle given three points
noncomputable def area_triangle (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p₁.1 * (p₂.2 - p₃.2) + p₂.1 * (p₃.2 - p₁.2) + p₃.1 * (p₁.2 - p₂.2))

-- Theorem statement
theorem hyperbola_triangle_area
  (n : ℝ) (P F₁ F₂ : ℝ × ℝ) 
  (h₁ : n > 1)
  (h₂ : hyperbola n P.1 P.2)
  (h₃ : distance P F₁ + distance P F₂ = 2 * Real.sqrt (n + 2)) :
  area_triangle P F₁ F₂ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l678_67812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opening_night_price_is_10_l678_67871

/-- Represents the movie theater's pricing and sales data for a day --/
structure TheaterData where
  matineePrice : ℚ
  eveningPrice : ℚ
  popcornPrice : ℚ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  openingNightCustomers : ℕ
  totalEarnings : ℚ

/-- Calculates the opening night ticket price based on the theater data --/
def openingNightTicketPrice (data : TheaterData) : ℚ :=
  let totalCustomers := data.matineeCustomers + data.eveningCustomers + data.openingNightCustomers
  let popcornSales := (totalCustomers / 2 : ℚ) * data.popcornPrice
  let knownEarnings := data.matineePrice * data.matineeCustomers + 
                       data.eveningPrice * data.eveningCustomers + 
                       popcornSales
  (data.totalEarnings - knownEarnings) / data.openingNightCustomers

/-- Theorem stating that the opening night ticket price is $10 --/
theorem opening_night_price_is_10 (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.popcornPrice = 10)
  (h4 : data.matineeCustomers = 32)
  (h5 : data.eveningCustomers = 40)
  (h6 : data.openingNightCustomers = 58)
  (h7 : data.totalEarnings = 1670) :
  openingNightTicketPrice data = 10 := by
  sorry

#eval openingNightTicketPrice {
  matineePrice := 5,
  eveningPrice := 7,
  popcornPrice := 10,
  matineeCustomers := 32,
  eveningCustomers := 40,
  openingNightCustomers := 58,
  totalEarnings := 1670
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opening_night_price_is_10_l678_67871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l678_67885

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * log x - 1

noncomputable def g (x : ℝ) : ℝ := x / exp (x - 1)

theorem min_a_value (a : ℝ) :
  a < 0 →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 3 4 → x₂ ∈ Set.Icc 3 4 → x₁ ≠ x₂ →
    |f a x₂ - f a x₁| < |(1 / g x₂) - (1 / g x₁)|) →
  a ≥ 3 - (2/3) * exp 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l678_67885
