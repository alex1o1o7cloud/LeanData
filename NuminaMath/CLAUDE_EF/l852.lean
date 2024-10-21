import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_theorem_l852_85257

/-- The probability distribution of a discrete random variable X -/
noncomputable def P (n : ℕ) (a : ℝ) : ℝ := a / (n * (n + 1))

/-- The sum of probabilities for all possible values is 1 -/
axiom sum_prob (a : ℝ) : P 1 a + P 2 a + P 3 a + P 4 a = 1

/-- The probability of X being between 9/4 and 13/4 -/
noncomputable def prob_between (a : ℝ) : ℝ := P 3 a

theorem prob_theorem : ∃ a : ℝ, prob_between a = 5/48 := by
  -- We need to find the value of 'a' that satisfies the condition
  let a := 5/4
  
  -- Show that this 'a' satisfies the sum_prob axiom
  have h1 : P 1 a + P 2 a + P 3 a + P 4 a = 1 := by
    -- This step would require detailed calculation
    sorry
  
  -- Show that prob_between with this 'a' equals 5/48
  have h2 : prob_between a = 5/48 := by
    -- Expand the definition and simplify
    unfold prob_between P
    -- This step would require detailed calculation
    sorry
  
  -- Provide the existence proof
  exact ⟨a, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_theorem_l852_85257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_reach_one_l852_85273

def operation (n : ℕ) : ℕ :=
  (n + (5 - n % 5) % 5) / 5

def sequenceOp (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => operation (sequenceOp n k)

theorem impossible_to_reach_one (n : ℕ) :
  (∀ k, sequenceOp n k ≠ 1) ↔ ∃ m, n = 3 * m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_reach_one_l852_85273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_magnitudes_l852_85237

theorem comparison_magnitudes :
  (¬ (2 < -(5 : ℝ))) ∧
  (¬ (-1 > -0.01)) ∧
  (¬ (|(-3 : ℝ)| < |(3 : ℝ)|)) ∧
  (-(-5) > -7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_magnitudes_l852_85237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_reject_percentage_value_l852_85293

/-- The percentage of products rejected by John -/
noncomputable def john_reject_rate : ℝ := 0.5

/-- The percentage of products rejected by Jane -/
noncomputable def jane_reject_rate : ℝ := 0.7

/-- The ratio of products inspected by Jane compared to John -/
noncomputable def jane_inspect_ratio : ℝ := 1.25

/-- The total percentage of products rejected -/
noncomputable def total_reject_percentage : ℝ := 
  (john_reject_rate / 100 + jane_reject_rate / 100 * jane_inspect_ratio) / (1 + jane_inspect_ratio) * 100

theorem total_reject_percentage_value : 
  total_reject_percentage = (0.005 + 0.00875) / 2.25 * 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_reject_percentage_value_l852_85293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_card_is_three_l852_85271

/-- Represents a set of four distinct positive integers in increasing order that sum to 30 -/
structure CardSet where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  sum_30 : a + b + c + d = 30
  increasing : a < b ∧ b < c ∧ c < d

/-- Function to get the nth element of a CardSet -/
def CardSet.nth (s : CardSet) (n : Fin 4) : ℕ+ :=
  match n with
  | 0 => s.a
  | 1 => s.b
  | 2 => s.c
  | 3 => s.d

/-- Predicate to check if a player can uniquely determine the other three numbers -/
def can_determine (s : CardSet) (position : Fin 4) : Prop :=
  ∀ t : CardSet, t.nth position = s.nth position → t = s

/-- The main theorem stating that if no player can determine the other numbers, 
    then the second number must be 3 -/
theorem second_card_is_three (s : CardSet) 
  (h1 : ¬ can_determine s 0)
  (h2 : ¬ can_determine s 1)
  (h3 : ¬ can_determine s 2)
  (h4 : ¬ can_determine s 3) : 
  s.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_card_is_three_l852_85271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_monotonic_iff_m_in_range_l852_85249

/-- The function f(x) = x³ + (m/2 + 2)x² - 2x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 + (m/2 + 2)*x^2 - 2*x

/-- The derivative of f(x) -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + (m + 4)*x - 2

theorem not_always_monotonic_iff_m_in_range (m : ℝ) :
  (∀ t, t ∈ Set.Icc 1 2 → ∃ x₁ x₂, x₁ ∈ Set.Ioo t 3 ∧ x₂ ∈ Set.Ioo t 3 ∧ f_deriv m x₁ * f_deriv m x₂ < 0) ↔
  m ∈ Set.Ioo (-37/3) (-9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_monotonic_iff_m_in_range_l852_85249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l852_85226

-- Define the line and circle
def line (k : ℝ) (x y : ℝ) : Prop := x - y - k = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the points A, B, and O
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry
def O : ℝ × ℝ := (0, 0)

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem range_of_k (k : ℝ) :
  k > 0 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line k x₁ y₁ ∧ circle_eq x₁ y₁ ∧
    line k x₂ y₂ ∧ circle_eq x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂)) →
  vec_length (vec_add A B) ≥ Real.sqrt 3 * vec_length (vec_add A ((-B.1, -B.2))) →
  k ∈ Set.Icc (Real.sqrt 6) (2 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l852_85226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_2a_minus_b_l852_85295

noncomputable def a (θ : ℝ) : Fin 2 → ℝ := ![Real.cos θ, Real.sin θ]
noncomputable def b : Fin 2 → ℝ := ![Real.sqrt 3, -1]

theorem max_magnitude_2a_minus_b :
  ∀ θ : ℝ, ‖(2 • a θ) - b‖ ≤ 4 ∧
  ∀ ε > 0, ∃ θ : ℝ, ‖(2 • a θ) - b‖ > 4 - ε :=
by
  sorry

#check max_magnitude_2a_minus_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_2a_minus_b_l852_85295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_2018_l852_85214

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2
  | n + 1 => Real.sqrt (a n ^ 2 + 2)

noncomputable def b (n : ℕ) : ℝ := 4 / (a n ^ 2 * a (n + 1) ^ 2)

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (fun i => b (i + 1))

theorem sum_of_b_2018 : S 2018 = 2018 / 2019 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_2018_l852_85214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l852_85218

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (b : ℝ) : ℝ → ℝ
| x => if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

theorem odd_function_value (b : ℝ) :
  is_odd_function (f b) → f b (-1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l852_85218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_polynomials_divide_l852_85247

theorem all_polynomials_divide : 
  (∃ p : Polynomial ℂ, (X^60 - 1) = (X^2 + X + 1) * p) ∧
  (∃ q : Polynomial ℂ, (X^60 - 1) = (X^4 - 1) * q) ∧
  (∃ r : Polynomial ℂ, (X^60 - 1) = (X^5 - 1) * r) ∧
  (∃ s : Polynomial ℂ, (X^60 - 1) = (X^15 - 1) * s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_polynomials_divide_l852_85247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_points_l852_85216

/-- The distance between two points in polar coordinates -/
noncomputable def distance_polar (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) : ℝ :=
  Real.sqrt (r₁^2 + r₂^2 - 2 * r₁ * r₂ * Real.cos (φ₁ - φ₂))

/-- Theorem: Distance between points A(4, φ₁) and B(10, φ₂) in polar coordinates -/
theorem distance_specific_points (φ₁ φ₂ : ℝ) (h : φ₁ - φ₂ = π/4) :
  distance_polar 4 10 φ₁ φ₂ = Real.sqrt (116 - 40 * Real.sqrt 2) := by
  sorry

#check distance_specific_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_points_l852_85216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l852_85232

/-- The length of a train in meters -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem: A train with a speed of 90 km/hr that crosses a pole in 12 seconds has a length of 300 meters -/
theorem train_length_calculation :
  train_length 90 12 = 300 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l852_85232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_equals_expected_result_l852_85224

/-- The fourth term in the expansion of (2a/√x - 2√x/(3a²))^7 -/
noncomputable def fourth_term (a x : ℝ) : ℝ :=
  (Nat.choose 7 3) * (2 * a / Real.sqrt x)^4 * (2 * Real.sqrt x / (3 * a^2))^3

/-- The expected result of the fourth term -/
noncomputable def expected_result (a x : ℝ) : ℝ :=
  128 / (27 * a^2 * Real.sqrt x)

/-- Theorem stating that the fourth term equals the expected result -/
theorem fourth_term_equals_expected_result (a x : ℝ) (h1 : a ≠ 0) (h2 : x > 0) :
  fourth_term a x = expected_result a x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_equals_expected_result_l852_85224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_inequality_l852_85274

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem f_inequality (a : ℝ) (h : a > 5) :
  ∀ x ∈ Set.Icc 1 3, f (a * x + x^2) > f (2 * x^2 + 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_inequality_l852_85274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_radical_axis_l852_85261

-- Define the basic geometric entities
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the points and circles
variable (c₁ c₂ : Circle) -- The two given circles
variable (A B : Point) -- Points where the common external tangent touches the circles
variable (X Y : Point) -- Points on the circles where the third circle touches

-- Define the condition for the common external tangent
def is_common_external_tangent (c₁ c₂ : Circle) (A B : Point) : Prop :=
  sorry -- Placeholder for the actual condition

-- Define the condition for the existence of a third circle
def exists_third_circle (c₁ c₂ : Circle) (X Y : Point) : Prop :=
  sorry -- Placeholder for the actual condition

-- Define the radical axis
noncomputable def radical_axis (c₁ c₂ : Circle) : Set Point :=
  sorry -- Placeholder for the actual definition

-- Define a line through two points
def line_through (P Q : Point) : Set Point :=
  sorry -- Placeholder for the actual definition

-- The theorem to be proved
theorem locus_is_radical_axis 
  (h₁ : is_common_external_tangent c₁ c₂ A B)
  (h₂ : exists_third_circle c₁ c₂ X Y) :
  ∀ P : Point, P ∈ (line_through A X) ∩ (line_through B Y) → P ∈ radical_axis c₁ c₂ :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_radical_axis_l852_85261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_half_l852_85212

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add a case for 0
  | 1 => 1/2
  | (n+2) => 1 / (1 - sequence_a (n+1))

theorem a_2017_equals_half : sequence_a 2017 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_half_l852_85212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_pi_third_eq_half_l852_85275

noncomputable def f (x : Real) : Real := Real.cos (x / 2)

noncomputable def g (x : Real) : Real := f (x - Real.pi)

theorem g_pi_third_eq_half : g (Real.pi / 3) = 1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_pi_third_eq_half_l852_85275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l852_85291

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin x + Real.sqrt 3 * (2 * Real.cos x ^ 2 - 1)

theorem f_properties :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ 2) ∧
  (∀ (x : ℝ), f (x + π/2) = f (2*x)) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), 
    x ∈ Set.Icc (-5*π/24 + k*π/2) (π/24 + k*π/2) → 
    ∀ (y : ℝ), y ∈ Set.Icc (-5*π/24 + k*π/2) (π/24 + k*π/2) → 
      x ≤ y → f (2*x) ≤ f (2*y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l852_85291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_cutting_l852_85246

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  center : ℝ × ℝ

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def line_through_centers (rect : Rectangle) (circ : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • rect.center + t • circ.center}

-- Helper functions (defined as noncomputable)
noncomputable def area_left_of_line (p : ℝ × ℝ) (shape : Rectangle ⊕ Circle) : ℝ := 
  sorry

noncomputable def area_right_of_line (p : ℝ × ℝ) (shape : Rectangle ⊕ Circle) : ℝ := 
  sorry

-- Theorem statement
theorem cake_cutting (rect : Rectangle) (circ : Circle) :
  (∀ p ∈ line_through_centers rect circ, 
    (area_left_of_line p (Sum.inl rect) = area_right_of_line p (Sum.inl rect)) ∧
    (area_left_of_line p (Sum.inr circ) = area_right_of_line p (Sum.inr circ))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_cutting_l852_85246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_operator_continuous_not_exists_operator_monotone_l852_85282

-- Define the space of continuous functions on [0,1]
def ContinuousOn' (f : ℝ → ℝ) : Prop := Continuous f ∧ ∀ x, x ∈ Set.Icc 0 1

-- Define the space of monotonically increasing functions on [0,1]
def MonotoneOn' (f : ℝ → ℝ) : Prop := ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x ≤ y → f x ≤ f y

-- Statement for continuous functions
theorem exists_operator_continuous :
  ∃ (A : (ℝ → ℝ) → ℝ),
    (∀ f g : ℝ → ℝ, ContinuousOn' f → ContinuousOn' g →
      (∀ x ∈ Set.Icc 0 1, f x ≤ g x) →
      (∃ x ∈ Set.Icc 0 1, f x ≠ g x) →
      A f < A g) :=
sorry

-- Statement for monotonically increasing functions
theorem not_exists_operator_monotone :
  ¬∃ (A : (ℝ → ℝ) → ℝ),
    (∀ f g : ℝ → ℝ, MonotoneOn' f → MonotoneOn' g →
      (∀ x ∈ Set.Icc 0 1, f x ≤ g x) →
      (∃ x ∈ Set.Icc 0 1, f x ≠ g x) →
      A f < A g) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_operator_continuous_not_exists_operator_monotone_l852_85282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_coloring_theorem_l852_85297

/-- A coloring of a prism's edges that satisfies the required conditions -/
def ValidPrismColoring (n : ℕ) :=
  ∃ (coloring : Fin n × Fin 3 → Fin 3),
    (∀ face : Set (Fin n × Fin 3), ∃ (c1 c2 c3 : Fin 3), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      (∃ (e1 e2 e3 : Fin n × Fin 3), 
        coloring e1 = c1 ∧ coloring e2 = c2 ∧ coloring e3 = c3 ∧
        e1 ∈ face ∧ e2 ∈ face ∧ e3 ∈ face)) ∧
    (∀ vertex : Fin 3, ∃ (c1 c2 c3 : Fin 3), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      (∃ (e1 e2 e3 : Fin n × Fin 3),
        coloring e1 = c1 ∧ coloring e2 = c2 ∧ coloring e3 = c3 ∧
        e1.2 = vertex ∧ e2.2 = vertex ∧ e3.2 = vertex))

theorem prism_coloring_theorem :
  ValidPrismColoring 1995 ∧ ¬ValidPrismColoring 1996 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_coloring_theorem_l852_85297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pyramid_volume_l852_85290

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def pyramid_volume (r : ℝ) (t : ℝ) : ℝ := (2 * r^3 * Real.sqrt 3 * t^2) / (1 - t^2)

theorem min_pyramid_volume (V : ℝ) (h : V > 0) :
  ∃ (r : ℝ), sphere_volume r = V ∧
  ∃ (min_vol : ℝ), min_vol = (6 * V * Real.sqrt 3) / Real.pi ∧
  ∀ (t : ℝ), 0 < t ∧ t < 1 → pyramid_volume r t ≥ min_vol := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pyramid_volume_l852_85290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l852_85234

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- Point P on the asymptote of hyperbola C in the first quadrant -/
def P (h : Hyperbola a b) : ℝ × ℝ := (a, b)

/-- Left focus of hyperbola C -/
noncomputable def F₁ (h : Hyperbola a b) : ℝ × ℝ := (-Real.sqrt (a^2 + b^2), 0)

/-- Right focus of hyperbola C -/
noncomputable def F₂ (h : Hyperbola a b) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Eccentricity of hyperbola C -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

/-- Theorem stating the eccentricity of hyperbola C -/
theorem hyperbola_eccentricity (h : Hyperbola a b) :
  (distance (P h) (F₁ h))^2 * (distance (P h) (F₂ h))^2 = 9 * (distance (P h) (F₂ h))^4 →
  eccentricity h = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l852_85234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_equation_range_of_3x_minus_4y_existence_of_fixed_point_l852_85255

/-- Given a segment AB of length 2 with endpoints A and B sliding on the x-axis and y-axis respectively,
    M is the midpoint of AB, and C is the trajectory of M. -/
noncomputable def CurveC : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Q is a fixed point (0, 2/3) -/
noncomputable def Q : ℝ × ℝ := (0, 2/3)

/-- T is a point (0, 3/2) -/
noncomputable def T : ℝ × ℝ := (0, 3/2)

/-- λ is a constant 3/2 -/
noncomputable def lambda : ℝ := 3/2

theorem curve_c_equation :
  CurveC = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} :=
by sorry

theorem range_of_3x_minus_4y :
  ∀ p ∈ CurveC, 3 * p.1 - 4 * p.2 ∈ Set.Icc (-5 : ℝ) 5 :=
by sorry

theorem existence_of_fixed_point :
  ∀ s ∈ CurveC, dist s T = lambda * dist s Q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_equation_range_of_3x_minus_4y_existence_of_fixed_point_l852_85255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_neg_one_range_of_a_l852_85250

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * f a x
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2*a-1) * x + a - 1

-- Theorem 1: Maximum value of f when a = -1
theorem max_value_f_neg_one :
  ∃ (x_max : ℝ), x_max > 0 ∧ ∀ (x : ℝ), x > 0 → f (-1) x ≤ f (-1) x_max ∧ f (-1) x_max = -1 := by
  sorry

-- Theorem 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ≥ 1 → g a x ≤ h a x) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_neg_one_range_of_a_l852_85250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l852_85244

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem min_distance_sum (A B : ℝ × ℝ) :
  point_on_parabola A → point_on_parabola B → collinear A focus B →
  distance A focus + 4 * distance B focus ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l852_85244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_function_properties_l852_85245

-- Define a positive function on a domain
def PositiveFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Monotone f ∧ f '' Set.Icc a b = Set.Icc a b

-- Define the square root function
noncomputable def sqrt_func (x : ℝ) : ℝ := Real.sqrt x

-- Define the quadratic function with parameter m
def quad_func (m : ℝ) (x : ℝ) : ℝ := x^2 + m

theorem positive_function_properties :
  -- Part 1: Equivalent domain interval for sqrt_func
  (PositiveFunction sqrt_func (Set.Ici 0) ∧
   ∃ a b, a = 0 ∧ b = 1 ∧ sqrt_func '' Set.Icc a b = Set.Icc a b) ∧
  -- Part 2: Existence of m for quad_func
  (∃ m, PositiveFunction (quad_func m) (Set.Iic 0)) ∧
  -- Part 3: Range of m
  (∃ m_lower m_upper, m_lower = -1 ∧ m_upper = -3/4 ∧
   ∀ m, PositiveFunction (quad_func m) (Set.Iic 0) → m > m_lower ∧ m < m_upper) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_function_properties_l852_85245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_existence_l852_85210

-- Define the functions
def F : ℝ → ℝ := λ x => x^2

noncomputable def G : ℝ → ℝ := λ x => if x ≤ 0 then x else 2*x

def H : ℝ → ℝ := λ x => x^3

-- Define what it means for a function to have an inverse
def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_existence :
  ¬(has_inverse F) ∧ (has_inverse G) ∧ (has_inverse H) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_existence_l852_85210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l852_85296

/-- The circle defined by the equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 2*p.2 + 1 = 0}

/-- The line defined by the equation x - y = 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 = 2}

/-- The distance between a point and the line -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 2| / Real.sqrt 2

/-- The maximum distance from a point on the circle to the line is 1 + √2 -/
theorem max_distance_circle_to_line :
  ∃ (p : ℝ × ℝ), p ∈ Circle ∧ distanceToLine p = 1 + Real.sqrt 2 ∧
  ∀ (q : ℝ × ℝ), q ∈ Circle → distanceToLine q ≤ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l852_85296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_stable_points_theorem_l852_85217

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define fixed points and stable points
def isFixedPoint (f : RealFunction) (x : ℝ) : Prop := f x = x
def isStablePoint (f : RealFunction) (x : ℝ) : Prop := f (f x) = x

-- Define sets A and B
def A (f : RealFunction) : Set ℝ := {x | isFixedPoint f x}
def B (f : RealFunction) : Set ℝ := {x | isStablePoint f x}

-- Define specific functions
def g : RealFunction := λ x ↦ 2 * x - 1
def f (a : ℝ) : RealFunction := λ x ↦ a * x^2 - 1

-- Main theorem
theorem fixed_stable_points_theorem :
  (∀ f : RealFunction, A f ⊆ B f) ∧
  (∃! x : ℝ, isStablePoint g x ∧ x = 1) ∧
  (∀ a : ℝ, (A (f a) = B (f a) ∧ Set.Nonempty (A (f a))) → a ∈ Set.Icc (-1/4) (3/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_stable_points_theorem_l852_85217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_side_length_a_l852_85253

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.cos (t.A / 2) = 2 * Real.sqrt 5 / 5 ∧
  t.b * t.c * Real.cos t.A = 3 ∧
  t.b + t.c = 6

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) :
  (1/2) * t.b * t.c * Real.sqrt (1 - (Real.cos t.A)^2) = 2 :=
sorry

-- Theorem for the side length a
theorem side_length_a (t : Triangle) (h : triangle_conditions t) :
  t.a = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_side_length_a_l852_85253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_perimeter_smaller_l852_85227

/-- Represents a rectangle with sides a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- Calculates the length of the diagonal of a rectangle -/
noncomputable def diagonalLength (r : Rectangle) : ℝ := Real.sqrt (r.a^2 + r.b^2)

/-- Represents the pentagon formed by folding a rectangle along its diagonal -/
structure FoldedPentagon (r : Rectangle) where
  d : ℝ
  h : d = diagonalLength r

/-- Calculates the perimeter of the folded pentagon -/
noncomputable def pentagonPerimeter (r : Rectangle) (p : FoldedPentagon r) : ℝ :=
  r.a + r.b + p.d

/-- Theorem stating that the perimeter of the folded pentagon is less than the original rectangle perimeter -/
theorem folded_perimeter_smaller (r : Rectangle) (p : FoldedPentagon r) :
  pentagonPerimeter r p < rectanglePerimeter r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_perimeter_smaller_l852_85227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l852_85231

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.sqrt (x - 1)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l852_85231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_points_l852_85235

noncomputable section

-- Define the linear function
def linear_function (x : ℝ) : ℝ := -2 * x - 2

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (0, -2)

-- Define the area of triangle BOC
def area_BOC (C : ℝ × ℝ) : ℝ := abs (C.1 * C.2) / 2

-- Theorem statement
theorem linear_function_and_points :
  (linear_function point_A.1 = point_A.2) ∧
  (linear_function point_B.1 = point_B.2) ∧
  (∃ C₁ C₂ : ℝ × ℝ,
    C₁ ≠ C₂ ∧
    linear_function C₁.1 = C₁.2 ∧
    linear_function C₂.1 = C₂.2 ∧
    area_BOC C₁ = 4 ∧
    area_BOC C₂ = 4 ∧
    ((C₁ = (4, -10) ∧ C₂ = (-4, 6)) ∨ (C₁ = (-4, 6) ∧ C₂ = (4, -10)))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_points_l852_85235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_engineer_ratio_l852_85286

/-- Represents a group of teachers and engineers -/
structure TeacherEngineerGroup where
  numTeachers : ℕ
  numEngineers : ℕ
  avgAge : ℚ
  avgTeacherAge : ℚ
  avgEngineerAge : ℚ

/-- The ratio of teachers to engineers is 2:1 given the specified conditions -/
theorem teacher_engineer_ratio (g : TeacherEngineerGroup) 
  (h1 : g.avgAge = 45)
  (h2 : g.avgTeacherAge = 40)
  (h3 : g.avgEngineerAge = 55)
  (h4 : g.numTeachers > 0)
  (h5 : g.numEngineers > 0) :
  g.numTeachers = 2 * g.numEngineers :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_engineer_ratio_l852_85286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_l852_85277

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 + 6*x + 15
def g (x : ℝ) : ℝ := x^2 - 4*x + 8

-- Define the vertex of a quadratic function
noncomputable def vertex (h : ℝ → ℝ) : ℝ × ℝ :=
  let x := - (deriv h 0) / (2 * deriv (deriv h) 0)
  (x, h x)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem vertex_distance : distance (vertex f) (vertex g) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_l852_85277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l852_85208

def is_monotone_increasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≤ a (n + 1)

theorem lambda_range
  (a : ℕ+ → ℝ)
  (lambda : ℝ)
  (h1 : is_monotone_increasing a)
  (h2 : ∀ n : ℕ+, a n = 3^(n : ℕ) - lambda * 2^(n : ℕ)) :
  lambda < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l852_85208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_odd_l852_85260

noncomputable def f (x : ℝ) : ℝ := x / (2^x - 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem h_is_odd : ∀ x, h (-x) = -h x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_odd_l852_85260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_special_lines_l852_85252

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Determines if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Calculates the x-intercept of a line -/
noncomputable def Line.xIntercept (l : Line) : ℝ :=
  -l.yIntercept / l.slope

/-- Determines if a line has equal absolute values of intercepts on both axes -/
def Line.hasEqualAbsIntercepts (l : Line) : Prop :=
  abs l.xIntercept = abs l.yIntercept

/-- The set of lines passing through (1, 2) with equal absolute intercepts -/
def specialLines : Set Line :=
  {l : Line | l.passesThrough 1 2 ∧ l.hasEqualAbsIntercepts}

/-- There are exactly three special lines -/
theorem three_special_lines : ∃ (s : Finset Line), s.card = 3 ∧ ↑s = specialLines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_special_lines_l852_85252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_four_points_l852_85285

/-- A lattice point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Theorem: Given integers n and k satisfying the conditions,
    there exists a circle passing through at least four points
    from any selection of k lattice points. -/
theorem circle_through_four_points
  (n k : ℕ)
  (h_n : n ≥ 2)
  (h_k : k ≥ (5 * n - 2) / 2)
  (points : Finset LatticePoint)
  (h_points_count : points.card = k)
  (h_points_bounds : ∀ p ∈ points, 1 ≤ p.x ∧ p.x ≤ n ∧ 1 ≤ p.y ∧ p.y ≤ n) :
  ∃ (circle : Set (ℝ × ℝ)) (p₁ p₂ p₃ p₄ : LatticePoint),
    p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points ∧ p₄ ∈ points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    (↑p₁.x, ↑p₁.y) ∈ circle ∧ (↑p₂.x, ↑p₂.y) ∈ circle ∧
    (↑p₃.x, ↑p₃.y) ∈ circle ∧ (↑p₄.x, ↑p₄.y) ∈ circle :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_four_points_l852_85285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_7_to_123_l852_85281

theorem last_three_digits_of_7_to_123 :
  7^123 % 1000 = 717 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_7_to_123_l852_85281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_reaction_products_l852_85209

noncomputable def reaction_stoichiometry (reactant_a : ℝ) (reactant_b : ℝ) : ℝ := min reactant_a reactant_b

theorem chemical_reaction_products 
  (initial_NaOH : ℝ) 
  (initial_NH4Cl : ℝ) 
  (h1 : initial_NaOH = 4) 
  (h2 : initial_NH4Cl = 3) :
  let limiting_reactant := reaction_stoichiometry initial_NaOH initial_NH4Cl
  limiting_reactant = initial_NH4Cl ∧
  limiting_reactant = 3 ∧
  (∀ product : ℝ, product ∈ [3, 3, 3] → product = limiting_reactant) :=
by
  sorry

#check chemical_reaction_products

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_reaction_products_l852_85209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l852_85200

/-- The quadratic function f(x) = -3(x-2)^2 - 3 has a maximum value of -3. -/
theorem quadratic_max_value : 
  ∃ (max : ℝ), max = -3 ∧ ∀ (x : ℝ), -3 * (x - 2)^2 - 3 ≤ max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l852_85200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_area_line_l852_85299

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

noncomputable def Ellipse.leftFocus (e : Ellipse) : Point :=
  ⟨-e.a * e.eccentricity, 0⟩

noncomputable def Triangle.perimeter (A B F : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) +
  Real.sqrt ((A.x - F.x)^2 + (A.y - F.y)^2) +
  Real.sqrt ((B.x - F.x)^2 + (B.y - F.y)^2)

def Line.perpToXAxis (l : Line) : Prop :=
  l.m = 0  -- Changed from Real.infty to 0 for vertical lines

theorem ellipse_equation_and_max_area_line 
    (e : Ellipse) 
    (h_ecc : e.eccentricity = 1/2) 
    (h_max_perim : ∀ (A B : Point) (l : Line), 
      e.equation A ∧ e.equation B ∧ 
      A.y = l.m * A.x + l.c ∧ B.y = l.m * B.x + l.c ∧
      l.perpToXAxis →
      Triangle.perimeter A B (e.leftFocus) ≤ 8) :
  (e.equation = fun p => p.x^2 / 4 + p.y^2 / 3 = 1) ∧
  (∃ (l₁ l₂ : Line), 
    (∀ (l : Line) (A B : Point),
      e.equation A ∧ e.equation B ∧
      A.y = l.m * A.x + l.c ∧ B.y = l.m * B.x + l.c ∧
      A.x = -4 ∧ B.x = -4 →
      (l = l₁ ∨ l = l₂)) ∧
    (l₁.m = 2*Real.sqrt 21/3 ∧ l₁.c = -4) ∧
    (l₂.m = -2*Real.sqrt 21/3 ∧ l₂.c = -4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_max_area_line_l852_85299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_sequence_length_l852_85294

/-- A sequence of integers where the sum of every 11 consecutive numbers is 100 or 101 -/
def ValidSequence (s : List Int) : Prop :=
  ∀ i, i + 10 < s.length →
    (List.sum (List.take 11 (List.drop i s)) = 100 ∨
     List.sum (List.take 11 (List.drop i s)) = 101)

/-- The theorem stating the maximum length of a valid sequence -/
theorem max_valid_sequence_length :
  ∀ s : List Int, ValidSequence s → s.length ≤ 22 :=
by
  sorry

#check max_valid_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_sequence_length_l852_85294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_probability_sum_30_l852_85229

/-- Represents a specialized 20-faced die -/
structure SpecialDie where
  numbers : Finset ℕ
  has_x : Bool

/-- The first die with numbers 1-19 and an X -/
def die1 : SpecialDie :=
  { numbers := Finset.range 20 \ {20},
    has_x := true }

/-- The second die with numbers 1-9, 11-20, and an X -/
def die2 : SpecialDie :=
  { numbers := (Finset.range 20 \ {10}),
    has_x := true }

/-- The probability of rolling a sum of 30 with the two special dice -/
def probability_sum_30 (d1 d2 : SpecialDie) : ℚ := 9 / 400

/-- Proof of the probability -/
theorem prove_probability_sum_30 :
    probability_sum_30 die1 die2 = 9 / 400 := by
  -- Unfold the definition of probability_sum_30
  unfold probability_sum_30
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_probability_sum_30_l852_85229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l852_85288

/-- Given a line y = 2x + 5 parameterized as (x, y) = (s, -3) + t(3, m), prove that s = -4 and m = 6 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 5 ↔ ∃ t : ℝ, (x, y) = (s + 3*t, -3 + m*t)) → 
  s = -4 ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l852_85288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_trace_matrix_l852_85220

open Matrix

variable {n : ℕ}
variable (A : Matrix (Fin n) (Fin n) ℝ)

def traceMatrix : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (λ i j =>
    if i ≥ j then
      Matrix.trace ((A ^ (i.val + 1)))
    else if i.val + 1 = j.val then
      (i.val + 1 : ℝ)
    else
      0)

theorem det_trace_matrix :
  det A = (1 / n.factorial : ℝ) * det (traceMatrix A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_trace_matrix_l852_85220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l852_85206

noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 4)

theorem function_properties (A ω : ℝ) (hA : A > 0) (hω : ω > 0)
  (h_amplitude : ∀ x, |f A ω x| ≤ 2 ∧ ∃ y, |f A ω y| = 2)
  (h_period : ∀ x, f A ω (x + Real.pi) = f A ω x) :
  (∀ x, f A ω x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧
  (∀ x, 2 * Real.sin (x + 3 * Real.pi / 4) = f A ω (x / 2 + Real.pi / 4)) ∧
  (∀ x, 2 * Real.sin (x + 3 * Real.pi / 4) = 2 * Real.sin ((Real.pi / 2 - x) + 3 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l852_85206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_implies_x_coordinate_l852_85268

/-- Given two points A and B in R², and a vector m, 
    if AB is parallel to m, then the x-coordinate of B is 5. -/
theorem parallel_vector_implies_x_coordinate 
  (A B : ℝ × ℝ) (m : ℝ × ℝ) : 
  A = (1, 2) → 
  B.2 = 4 → 
  m = (2, 1) → 
  ∃ (k : ℝ), (B.1 - A.1, B.2 - A.2) = k • m → 
  B.1 = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_implies_x_coordinate_l852_85268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_projectile_speed_l852_85213

/-- Given two projectiles launched simultaneously from 1455 km apart, 
    with one traveling at 470 km/h, and both meeting after 90 minutes, 
    prove that the speed of the second projectile is 500 km/h. -/
theorem second_projectile_speed 
  (initial_distance : ℝ) 
  (first_speed : ℝ) 
  (meeting_time : ℝ) 
  (second_speed : ℝ) : 
  initial_distance = 1455 ∧ 
  first_speed = 470 ∧ 
  meeting_time = 1.5 →  -- 90 minutes converted to hours
  second_speed = 500 := by 
{
  intro h
  sorry  -- Placeholder for the actual proof
}

#check second_projectile_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_projectile_speed_l852_85213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_units_digit_l852_85284

/-- The set of positive five-digit integers -/
def FiveDigitIntegers : Finset ℕ := Finset.filter (fun n => 10000 ≤ n ∧ n ≤ 99999) (Finset.range 100000)

/-- The set of even digits -/
def EvenDigits : Finset ℕ := {0, 2, 4, 6, 8}

/-- The function that returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The probability of an event in a finite sample space -/
def probability (S E : Finset ℕ) : ℚ :=
  (E ∩ S).card / S.card

theorem prob_even_units_digit :
  probability FiveDigitIntegers (FiveDigitIntegers.filter (fun n => unitsDigit n ∈ EvenDigits)) = 1/2 := by
  sorry

#eval probability FiveDigitIntegers (FiveDigitIntegers.filter (fun n => unitsDigit n ∈ EvenDigits))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_units_digit_l852_85284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l852_85204

theorem functional_equation_solution (f g : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x + y * g x) = g x + x * f y) :
  (∀ x : ℝ, f x = x ∧ g x = x) ∨ (∀ x : ℝ, f x = 0 ∧ g x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l852_85204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_with_properties_l852_85251

noncomputable section

open Real

-- Define the given functions
def f₁ (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)
def f₂ (x : ℝ) : ℝ := 2 * sin (2 * x - π / 6)
def f₃ (x : ℝ) : ℝ := 2 * sin (x / 2 + π / 3)
def f₄ (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

-- Define the property of having a period of π
def has_period_pi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = f x

-- Define the property of being symmetric about x = π/3
def symmetric_about_pi_third (f : ℝ → ℝ) : Prop :=
  ∀ x, f (π/3 + x) = f (π/3 - x)

-- Theorem statement
theorem unique_function_with_properties :
  (has_period_pi f₂ ∧ symmetric_about_pi_third f₂) ∧
  (¬(has_period_pi f₁ ∧ symmetric_about_pi_third f₁)) ∧
  (¬(has_period_pi f₃ ∧ symmetric_about_pi_third f₃)) ∧
  (¬(has_period_pi f₄ ∧ symmetric_about_pi_third f₄)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_with_properties_l852_85251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unpartnered_students_l852_85230

/-- Represents a class with male and female students -/
structure GradeClass where
  males : ℕ
  females : ℕ

/-- Calculates the number of students unable to partner with the opposite gender -/
def unpartnered_students (classes : List GradeClass) : ℕ :=
  let total_males := classes.map (λ c => c.males) |>.sum
  let total_females := classes.map (λ c => c.females) |>.sum
  Int.natAbs (total_males - total_females)

/-- Theorem stating that no students are left unpartnered in the given scenario -/
theorem no_unpartnered_students : 
  let classes : List GradeClass := [
    ⟨17, 13⟩, -- First 6th grade class
    ⟨14, 18⟩, -- Second 6th grade class
    ⟨15, 17⟩, -- Third 6th grade class
    ⟨22, 20⟩  -- 7th grade class
  ]
  unpartnered_students classes = 0 := by
  sorry

#eval unpartnered_students [
  ⟨17, 13⟩, -- First 6th grade class
  ⟨14, 18⟩, -- Second 6th grade class
  ⟨15, 17⟩, -- Third 6th grade class
  ⟨22, 20⟩  -- 7th grade class
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unpartnered_students_l852_85230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l852_85279

def sequenceA (n : ℕ) : ℚ :=
  (1024000 : ℚ) / (4 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = ↑z

theorem last_integer_in_sequence :
  ∃ (k : ℕ), (is_integer (sequenceA k) ∧ sequenceA k = 250) ∧
  ∀ (m : ℕ), m > k → ¬ is_integer (sequenceA m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l852_85279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_satisfying_equation_l852_85243

open Real MeasureTheory

theorem sum_of_angles_satisfying_equation : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x < 2*π ∧ 
      sin x^4 - cos x^4 = 1 / cos x - 1 / sin x) ∧
    (S.sum id = 3*π/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_satisfying_equation_l852_85243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_theorem_l852_85225

theorem divisor_product_theorem (n : ℕ+) (N : ℕ) (h : N > 0) : 
  ∃ (divisors : Finset ℕ+), 
    (∀ d ∈ divisors, d ∣ n) ∧ 
    (∀ d : ℕ+, d ∣ n → d ∈ divisors) ∧
    (Finset.card divisors = N) ∧
    (Finset.prod divisors (λ d => d) = n ^ (N / 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_theorem_l852_85225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_car_speed_l852_85266

/-- Represents the speed of a car in miles per hour -/
def Speed := ℝ

/-- Represents the distance between two cars in miles -/
def Distance := ℝ

/-- Represents the time taken for one car to overtake another in hours -/
def Time := ℝ

/-- 
Given two cars traveling in the same direction, where:
- The black car travels at a constant speed of 50 miles per hour
- The first car is initially 20 miles ahead of the black car
- The black car overtakes the first car in exactly 1 hour
Then, the speed of the first car must be 30 miles per hour.
-/
theorem first_car_speed 
  (black_car_speed : Speed) 
  (initial_distance : Distance) 
  (overtake_time : Time) 
  (h1 : black_car_speed = (50 : ℝ))
  (h2 : initial_distance = (20 : ℝ))
  (h3 : overtake_time = (1 : ℝ)) :
  ∃ (first_car_speed : Speed), first_car_speed = (30 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_car_speed_l852_85266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_l852_85289

/-- Represents the percentage of cheating by the shopkeeper -/
noncomputable def cheat_percentage : ℝ := 10

/-- Calculates the actual amount received when buying given the nominal amount -/
noncomputable def amount_received (nominal_amount : ℝ) : ℝ :=
  nominal_amount * (1 + cheat_percentage / 100)

/-- Calculates the actual amount given when selling given the nominal amount -/
noncomputable def amount_given (nominal_amount : ℝ) : ℝ :=
  nominal_amount * (1 - cheat_percentage / 100)

/-- Calculates the gain percentage based on the nominal amount -/
noncomputable def gain_percentage (nominal_amount : ℝ) : ℝ :=
  ((amount_received nominal_amount - amount_given nominal_amount) / nominal_amount) * 100

/-- Theorem stating that the shopkeeper's total gain is 20% -/
theorem shopkeeper_gain :
  ∀ (nominal_amount : ℝ), nominal_amount > 0 → gain_percentage nominal_amount = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_l852_85289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l852_85223

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition (b+c):(c+a):(a+b) = 4:5:6
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.b + t.c = 4*k ∧ t.c + t.a = 5*k ∧ t.a + t.b = 6*k

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : ratio_condition t) :
  (¬ ∃! (t' : Triangle), ratio_condition t') ∧
  (∃ A : ℝ, Real.cos A < 0) ∧
  (∃ (A B C : ℝ), Real.sin A / Real.sin B = 7/5 ∧ Real.sin B / Real.sin C = 5/3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l852_85223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_in_terms_of_k_l852_85248

theorem sin_2alpha_in_terms_of_k (k : ℝ) (α : ℝ) 
  (h : Real.cos (π / 4 - α) = k) : 
  Real.sin (2 * α) = 2 * k^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_in_terms_of_k_l852_85248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_2_sqrt_5_l852_85264

/-- The projection of vector a onto vector b -/
noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)

/-- Given vectors a and b, prove that the projection of a onto b is 2√5 -/
theorem projection_equals_2_sqrt_5 :
  let a : ℝ × ℝ := (-3, 4)
  let b : ℝ × ℝ := (-2, 1)
  vector_projection a b = 2 * Real.sqrt 5 := by
  -- Unfold the definition of vector_projection
  unfold vector_projection
  -- Simplify the numerator and denominator
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_2_sqrt_5_l852_85264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_identity_l852_85205

/-- The function g(x) defined as bx / (2x + 3) -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (b * x) / (2 * x + 3)

/-- Theorem stating that g(g(x)) = x for all x ≠ -3/2 if and only if b = 5 or b = -3 -/
theorem g_composition_identity (b : ℝ) :
  (∀ x : ℝ, x ≠ -3/2 → g b (g b x) = x) ↔ (b = 5 ∨ b = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_identity_l852_85205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_correct_l852_85242

/-- The coefficient of x^2 in the expansion of (√x - 2)^5 -/
def coefficient_x_squared : ℤ := -10

/-- The binomial expansion of (√x - 2)^5 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (Real.sqrt x - 2)^5

/-- Theorem stating that the coefficient of x^2 in the expansion of (√x - 2)^5 is correct -/
theorem coefficient_x_squared_is_correct :
  ∃ (f : ℝ → ℝ), ∀ x, binomial_expansion x = coefficient_x_squared * x^2 + f x ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x| / x^2 < ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_correct_l852_85242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l852_85263

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) : ℝ) / 2 * d

-- State the theorem
theorem min_sum_arithmetic_sequence :
  ∀ d : ℝ,
  arithmetic_sequence (-11) d 4 + arithmetic_sequence (-11) d 6 = -6 →
  ∃ (n : ℕ),
    ∀ (m : ℕ),
      sum_arithmetic_sequence (-11) d n ≤ sum_arithmetic_sequence (-11) d m ∧
      n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l852_85263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_has_six_legs_l852_85267

/-- Represents the number of one-headed 34-legged creatures -/
def num_34_legged : ℕ := sorry

/-- Represents the number of three-headed Dragons -/
def num_dragons : ℕ := sorry

/-- Represents the number of legs of a three-headed Dragon -/
def dragon_legs : ℕ := sorry

/-- The total number of legs in the herd -/
def total_legs : ℕ := 286

/-- The total number of heads in the herd -/
def total_heads : ℕ := 31

/-- Theorem stating that given the conditions, a three-headed Dragon has 6 legs -/
theorem dragon_has_six_legs :
  (num_34_legged + 3 * num_dragons = total_heads) →
  (34 * num_34_legged + dragon_legs * num_dragons = total_legs) →
  dragon_legs = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_has_six_legs_l852_85267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l852_85219

theorem total_students (below_eight_percent : Real) 
                       (eight_years_old : Nat) 
                       (above_eight_ratio : Real) : Nat :=
  let total_students := 80
  have h1 : below_eight_percent = 0.25 := by sorry
  have h2 : eight_years_old = 36 := by sorry
  have h3 : above_eight_ratio = 2/3 := by sorry
  have h4 : total_students = 80 := by sorry
  total_students


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l852_85219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_game_not_fair_l852_85258

/-- Represents a card game with n players and n cards -/
structure CardGame where
  n : ℕ
  n_pos : 0 < n

/-- The probability of the k-th player winning -/
noncomputable def win_probability (game : CardGame) (k : ℕ) : ℝ :=
  (1 - 1 / (game.n : ℝ)) ^ ((k - 1) * game.n) / (game.n : ℝ)

/-- The game is fair if all players have equal probability of winning -/
def is_fair (game : CardGame) : Prop :=
  ∀ i j, i ≤ game.n → j ≤ game.n → win_probability game i = win_probability game j

/-- Theorem: The card game is not fair for any number of players greater than 1 -/
theorem card_game_not_fair (game : CardGame) (h : 1 < game.n) : ¬ is_fair game := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_game_not_fair_l852_85258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_triangle_formula_l852_85241

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithCircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of hypotenuse PR -/
  pr : ℝ
  /-- The triangle is right-angled -/
  right_angle : pq < pr
  /-- The circle is inscribed in the triangle -/
  inscribed : True

/-- The area of the portion of the inscribed circle outside the right triangle -/
noncomputable def areaOutsideTriangle (t : RightTriangleWithCircle) : ℝ :=
  28.125 * Real.pi - 56.25

/-- Theorem: The area of the portion of the circle outside the triangle is 28.125π - 56.25 -/
theorem area_outside_triangle_formula (t : RightTriangleWithCircle) 
  (h₁ : t.pq = 9) (h₂ : t.pr = 15) : 
  areaOutsideTriangle t = 28.125 * Real.pi - 56.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_triangle_formula_l852_85241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_term_to_form_complete_square_l852_85238

/-- Given a polynomial 4x^2 + 1, there exists a term that, when added, 
    forms a complete square. -/
theorem exists_term_to_form_complete_square :
  ∃ (t : ℝ → ℝ), (∀ x, ∃ (a : ℝ), (4 * x^2 + 1 + t x = a^2)) ∧ 
  (t = (λ x => 4 * x^4) ∨ 
   t = (λ x => 4 * x) ∨ 
   t = (λ x => -4 * x) ∨ 
   t = (λ _ => (-1 : ℝ)) ∨ 
   t = (λ x => -4 * x^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_term_to_form_complete_square_l852_85238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l852_85259

-- Define the triangles and points
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- State the theorem
theorem triangle_ratio_sum (A B C D E : ℝ × ℝ) (m n : ℕ) :
  Triangle A B C →
  Triangle A B D →
  distance A C = 6 →
  distance B C = 8 →
  distance A D = 15 →
  -- C and D on opposite sides of AB (we can't directly represent this geometrically)
  -- Line through D parallel to AC meets CB extended at E (implied by the ratio)
  (distance D E) / (distance D B) = m / n →
  Nat.Coprime m n →
  m + n = 14 := by
  sorry

#check triangle_ratio_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l852_85259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circle_theorem_l852_85228

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  angle_B : ℝ
  angle_D : ℝ
  convex : angle_B + angle_D > 180

-- Define a circle tangent to sides a, b, and c
structure TangentCircle (q : ConvexQuadrilateral) where
  tangent_a : Bool
  tangent_b : Bool
  tangent_c : Bool
  is_tangent : tangent_a ∧ tangent_b ∧ tangent_c
  intersects_d : Bool

-- Define the relationship between the quadrilateral and the circle
def CircleQuadrilateralRelation (q : ConvexQuadrilateral) (circle : TangentCircle q) :=
  (¬circle.intersects_d → q.a + q.c > q.b + q.d) ∧
  (circle.intersects_d → q.a + q.c < q.b + q.d)

-- The main theorem
theorem quadrilateral_circle_theorem (q : ConvexQuadrilateral) (circle : TangentCircle q) :
  CircleQuadrilateralRelation q circle := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma not_intersects_implies_inequality (q : ConvexQuadrilateral) (circle : TangentCircle q) 
  (h : ¬circle.intersects_d) : q.a + q.c > q.b + q.d := by
  sorry

lemma intersects_implies_inequality (q : ConvexQuadrilateral) (circle : TangentCircle q)
  (h : circle.intersects_d) : q.a + q.c < q.b + q.d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_circle_theorem_l852_85228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_very_good_elements_l852_85222

open BigOperators Nat

theorem good_very_good_elements (n : ℕ) (hn : n > 1) :
  let M := Finset.range (n^2 - 1)
  let good (a : ℕ) := ∃ b ∈ M, n^2 ∣ (a*b - b)
  let very_good (a : ℕ) := n^2 ∣ (a^2 - a)
  let g := (M.filter good).card
  let v := (M.filter very_good).card
  v^2 + v ≤ g ∧ g ≤ n^2 - n :=
by
  -- Define the set and conditions
  intro M good very_good g v
  
  -- Proof steps would go here
  sorry  -- Placeholder for the actual proof

#check good_very_good_elements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_very_good_elements_l852_85222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_house_time_paint_house_time_rounded_l852_85221

/-- The time it takes for two painters to complete a job together, given their individual rates -/
noncomputable def time_to_paint (rate_a rate_b : ℝ) : ℝ :=
  1 / (rate_a + rate_b)

/-- The theorem stating the time it takes for A and B to paint the house together -/
theorem paint_house_time : 
  let days_a : ℝ := 12
  let rate_a : ℝ := 1 / days_a
  let rate_b : ℝ := 1.75 * rate_a
  let time_together : ℝ := time_to_paint rate_a rate_b
  ∃ (ε : ℝ), ε > 0 ∧ |time_together - 4.36| < ε :=
by sorry

/-- The theorem stating that the result rounds to 4.5 days -/
theorem paint_house_time_rounded :
  let days_a : ℝ := 12
  let rate_a : ℝ := 1 / days_a
  let rate_b : ℝ := 1.75 * rate_a
  let time_together : ℝ := time_to_paint rate_a rate_b
  ∃ (ε : ℝ), ε > 0 ∧ |time_together - 4.5| < 0.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_house_time_paint_house_time_rounded_l852_85221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l852_85201

-- Define the set A
def A : Set ℤ := {-2, 0, 2}

-- Define the function f
def f (x : ℤ) : ℤ := Int.natAbs x

-- Define set B as the image of A under f
def B : Set ℤ := f '' A

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l852_85201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2010_in_third_quadrant_l852_85262

-- Define the concept of quadrant
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Function to determine the quadrant of an angle in degrees
noncomputable def angle_quadrant (angle : ℝ) : Quadrant :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle && normalized_angle < 90 then Quadrant.first
  else if 90 ≤ normalized_angle && normalized_angle < 180 then Quadrant.second
  else if 180 ≤ normalized_angle && normalized_angle < 270 then Quadrant.third
  else Quadrant.fourth

-- Theorem statement
theorem angle_2010_in_third_quadrant :
  angle_quadrant 2010 = Quadrant.third := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2010_in_third_quadrant_l852_85262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l852_85269

theorem complex_transformation (z : ℂ) : 
  z = -1 + 7*Complex.I → 
  2 * (Complex.exp (Complex.I * Real.pi / 3)) * z = -22 - (Real.sqrt 3 - 7) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_transformation_l852_85269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l852_85233

/-- The function q(x) -/
noncomputable def q (x : ℝ) : ℝ := (20/21) * x^2 + (40/21) * x - (60/21)

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  (q 1 = 0) ∧ 
  (q (-3) = 0) ∧ 
  (∃ (a b c : ℝ), ∀ x, q x = a * x^2 + b * x + c) ∧ 
  (q 4 = 20) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l852_85233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_root_sum_power_l852_85278

theorem last_two_digits_of_root_sum_power : 
  ∃ n : ℕ, (Real.sqrt 29 + Real.sqrt 21)^1984 = 100 * n + 71 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_root_sum_power_l852_85278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_period_is_three_years_l852_85254

/-- The amount of money lent (in Rs.) -/
noncomputable def loan_amount : ℝ := 3500

/-- The interest rate at which A lends to B (in percent) -/
noncomputable def interest_rate_A_to_B : ℝ := 10

/-- The interest rate at which B lends to C (in percent) -/
noncomputable def interest_rate_B_to_C : ℝ := 15

/-- B's total gain (in Rs.) -/
noncomputable def total_gain : ℝ := 525

/-- Calculate the annual interest (in Rs.) given a principal and interest rate -/
noncomputable def annual_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (rate / 100)

/-- Calculate the number of years for B to achieve the total gain -/
noncomputable def calculate_years : ℝ :=
  total_gain / (annual_interest loan_amount interest_rate_B_to_C - 
                annual_interest loan_amount interest_rate_A_to_B)

theorem gain_period_is_three_years :
  calculate_years = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_period_is_three_years_l852_85254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l852_85265

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) - Real.sqrt 3

theorem function_properties :
  -- The distance between adjacent axes of symmetry is π/2
  (∀ k : ℤ, f ((k + 1) * Real.pi / 4) = f (k * Real.pi / 4)) ∧
  -- g(x) is odd after translation
  (∀ x : ℝ, f (x - Real.pi / 6) + Real.sqrt 3 = -f (-x - Real.pi / 6) - Real.sqrt 3) ∧
  -- Axes of symmetry
  (∀ k : ℤ, ∀ x : ℝ, f (Real.pi / 12 + k * Real.pi / 2 + x) = f (Real.pi / 12 + k * Real.pi / 2 - x)) ∧
  -- Increasing intervals
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi - 5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + Real.pi / 12 → f x < f y) ∧
  -- Decreasing intervals
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi + Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 7 * Real.pi / 12 → f x > f y) ∧
  -- Range of m
  (∀ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → f x ^ 2 - (2 + m) * f x + 2 + m ≤ 0) ↔ m ≤ (-1 - 3 * Real.sqrt 3) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l852_85265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l852_85240

/-- The circle with center (2,2) and radius √5 -/
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (3, 1)

/-- The length of a chord passing through the given point -/
def chord_length (m : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ 
    ((x - point.1)^2 + (y - point.2)^2) = (m / 2)^2

theorem shortest_chord_length :
  ∃ (m : ℝ), chord_length m ∧ 
    (∀ (m' : ℝ), chord_length m' → m ≤ m') ∧
    m = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l852_85240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l852_85207

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

def ellipse_eq (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

def matrix_A (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 0],
    ![0, b]]

def transforms (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → ellipse_eq (A 0 0 * x + A 0 1 * y) (A 1 0 * x + A 1 1 * y)

theorem matrix_transformation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_transform : transforms (matrix_A a b)) :
  a = 2 ∧ b = Real.sqrt 3 ∧ 
  (matrix_A a b)⁻¹ = ![![1/2, 0],
                      ![0, Real.sqrt 3 / 3]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l852_85207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_when_k_nonpositive_l852_85276

/-- The function f(x) defined as (x-1)e^x - (k/2)x^2 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k / 2 * x^2

/-- Theorem stating the number of zeros for f(x) when k ≤ 0 -/
theorem zeros_of_f_when_k_nonpositive (k : ℝ) (h : k ≤ 0) :
  (k < 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) ∨
  (k = 0 ∧ ∃! x : ℝ, f k x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_when_k_nonpositive_l852_85276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_assignment_exists_l852_85211

/-- A valid vertex assignment for a regular 2^n-gon. -/
def VertexAssignment (n : ℕ) := Fin (2^n) → Fin (2^n)

/-- Checks if two n-digit binary numbers differ in exactly one digit. -/
def differByOneDigit (n : ℕ) (a b : Fin (2^n)) : Prop :=
  ∃! i : Fin n, (a.val / 2^i.val) % 2 ≠ (b.val / 2^i.val) % 2

/-- Checks if a VertexAssignment is valid according to the problem conditions. -/
def isValidAssignment (n : ℕ) (assignment : VertexAssignment n) : Prop :=
  (∀ v, ∀ i : Fin n, (assignment v).val / 2^i.val % 2 < 2) ∧ 
  (∀ v₁ v₂, v₁ ≠ v₂ → assignment v₁ ≠ assignment v₂) ∧
  (∀ v : Fin (2^n), differByOneDigit n (assignment v) (assignment ((v + 1) % (2^n))))

/-- The main theorem stating that a valid assignment exists for all n ≥ 2. -/
theorem valid_assignment_exists (n : ℕ) (h : n ≥ 2) : 
  ∃ (assignment : VertexAssignment n), isValidAssignment n assignment :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_assignment_exists_l852_85211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_camp_total_l852_85236

theorem boys_camp_total (total : ℕ) : 
  -- School A conditions
  (0.2 * (total : ℝ) = (77 : ℝ) / 0.7) →
  -- School B conditions
  (0.3 * (total : ℝ) = (72 : ℝ) / 0.4) →
  -- School C conditions
  (0.5 * (total : ℝ) = (98 : ℝ) / 0.4) →
  -- Conclusion
  total = 535 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_camp_total_l852_85236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_is_27_l852_85270

/-- A configuration of 9 points in 3D space. -/
structure PointConfiguration :=
  (points : Finset (Fin 9 → ℝ))
  (no_four_coplanar : ∀ (p₁ p₂ p₃ p₄ : Fin 9 → ℝ), p₁ ∈ points → p₂ ∈ points → p₃ ∈ points → p₄ ∈ points → 
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₁ ≠ p₄ → p₂ ≠ p₃ → p₂ ≠ p₄ → p₃ ≠ p₄ → 
    ¬(∃ (a b c d : ℝ), a • p₁ + b • p₂ + c • p₃ + d • p₄ = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)))

/-- A graph on the 9 points. -/
structure Graph (pc : PointConfiguration) :=
  (edges : Finset (Fin 9 × Fin 9))
  (no_tetrahedron : ∀ (v₁ v₂ v₃ v₄ : Fin 9), 
    (v₁, v₂) ∈ edges → (v₁, v₃) ∈ edges → (v₁, v₄) ∈ edges → 
    (v₂, v₃) ∈ edges → (v₂, v₄) ∈ edges → (v₃, v₄) ∈ edges → False)

/-- A triangle in the graph. -/
def Triangle (pc : PointConfiguration) (g : Graph pc) : Type :=
  { t : Fin 9 × Fin 9 × Fin 9 // 
    (t.1, t.2.1) ∈ g.edges ∧ (t.1, t.2.2) ∈ g.edges ∧ (t.2.1, t.2.2) ∈ g.edges }

/-- The number of triangles in a graph. -/
noncomputable def num_triangles (pc : PointConfiguration) (g : Graph pc) : ℕ :=
  (Finset.filter (λ t : Fin 9 × Fin 9 × Fin 9 => 
    (t.1, t.2.1) ∈ g.edges ∧ (t.1, t.2.2) ∈ g.edges ∧ (t.2.1, t.2.2) ∈ g.edges) 
    (Finset.univ.product (Finset.univ.product Finset.univ))).card

/-- The maximum number of triangles possible in any graph on the given point configuration. -/
noncomputable def max_triangles (pc : PointConfiguration) : ℕ :=
  ⨆ (g : Graph pc), num_triangles pc g

/-- The main theorem: The maximum number of triangles is 27. -/
theorem max_triangles_is_27 (pc : PointConfiguration) : max_triangles pc = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_is_27_l852_85270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l852_85215

/-- Calculates the time taken for a train to cross a signal pole given its length,
    the length of a platform it crosses, and the time taken to cross the platform. -/
noncomputable def time_to_cross_signal_pole (train_length platform_length : ℝ) (time_to_cross_platform : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_to_cross_platform
  train_length / train_speed

/-- Theorem stating that a train of length 425 meters crossing a platform of length 159.375 meters
    in 55 seconds will take 40 seconds to cross a signal pole. -/
theorem train_crossing_time :
  time_to_cross_signal_pole 425 159.375 55 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l852_85215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_inequality_holds_l852_85292

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x - 2 / x^3

theorem min_value_and_inequality (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a x ≤ f a y) ∧ f a (Real.sqrt (2/a)) = 0 ↔ a = 2 :=
sorry

theorem inequality_holds (x : ℝ) (h : x ∈ Set.Icc 1 2) :
  f 2 x ≤ f_derivative 2 x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_inequality_holds_l852_85292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sixty_degrees_l852_85287

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  b^2 + c^2 - a^2 = b*c → ∃ A : ℝ, 0 ≤ A ∧ A ≤ Real.pi ∧ Real.cos A = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sixty_degrees_l852_85287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_good_subset_l852_85280

theorem existence_of_good_subset (n : ℕ) (h : n ≥ 2) :
  ∀ (f : Finset (Fin (2 * n)) → Fin (2 * n)),
  ∃ (T : Finset (Fin (2 * n))),
    T.card = n ∧
    ∀ (k : Fin (2 * n)),
      k ∈ T →
      f (T.erase k) ≠ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_good_subset_l852_85280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l852_85298

/-- The polar equation of the circle -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ - 2 * Real.sqrt 3 * Real.sin θ

/-- The center of the circle in polar coordinates -/
noncomputable def circle_center : ℝ × ℝ := (2, -Real.pi/3)

/-- Theorem stating that the given point is the center of the circle -/
theorem circle_center_correct :
  let (r, θ) := circle_center
  ∀ ρ φ, circle_equation ρ φ →
    (r * Real.cos θ - ρ * Real.cos φ)^2 + (r * Real.sin θ - ρ * Real.sin φ)^2 = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l852_85298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l852_85256

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (1 + m * x) + x^2 / 2 - m * x

theorem f_properties :
  ∀ m : ℝ, 0 < m → m ≤ 1 →
  (∀ x : ℝ, -1 < x → x ≤ 0 → f 1 x ≤ x^3 / 3) ∧
  (∃! x : ℝ, f 1 x = 0) ∧
  (m < 1 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧
    ∀ x : ℝ, f m x = 0 → x = x₁ ∨ x = x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l852_85256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l852_85283

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x / 2) * Real.cos (x / 2) - Real.sqrt 2 * (Real.sin (x / 2))^2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 → f x ≥ -1 - Real.sqrt 2 / 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 ∧ f x = -1 - Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l852_85283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_line_equation_l852_85202

/-- Given a line l: mx - m²y - 1 = 0 passing through point P(2,1),
    the equation of a line whose angle of inclination is the complement
    of the angle of inclination of line l is x + y - 3 = 0 -/
theorem complementary_line_equation (m : ℝ) :
  (m * 2 - m^2 * 1 - 1 = 0) →
  (∃ (x y : ℝ), m * x - m^2 * y - 1 = 0 ∧ x = 2 ∧ y = 1) →
  (∃ (x y : ℝ), x + y - 3 = 0 ∧
    (∀ (x' y' : ℝ), m * x' - m^2 * y' - 1 = 0 →
      (Real.arctan 1 + Real.arctan m = π / 2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_line_equation_l852_85202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_of_bisecting_line_l852_85272

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a line bisects a circle -/
def bisects (l : Line) (c : Circle) : Prop :=
  sorry

/-- Determines if a line passes through the fourth quadrant -/
def passes_fourth_quadrant (l : Line) : Prop :=
  sorry

/-- The circle x^2 + y^2 - 2x - 4y = 0 -/
noncomputable def given_circle : Circle :=
  { center := (1, 2), radius := Real.sqrt 5 }

theorem slope_range_of_bisecting_line (l : Line) :
  bisects l given_circle →
  ¬passes_fourth_quadrant l →
  0 ≤ l.slope ∧ l.slope ≤ 2 := by
  sorry

#check slope_range_of_bisecting_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_of_bisecting_line_l852_85272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_negative_eleven_sixths_l852_85239

-- Define the function g as noncomputable
noncomputable def g (a b : ℝ) : ℝ :=
  if a + b ≤ 4 then
    (a * b + a - 3) / (3 * a)
  else
    (a * b + b + 3) / (-3 * b)

-- State the theorem
theorem g_sum_equals_negative_eleven_sixths :
  g 1 2 + g 3 2 = -11/6 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_negative_eleven_sixths_l852_85239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l852_85203

theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l852_85203
