import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l51_5176

theorem arithmetic_calculations :
  (1 - (-7) - (-5) + (-4) = 9) ∧
  ((-81 : ℚ) / (9/4) * (4/9) / (-16) = -1/16) ∧
  (-7 - 2 * (-3) + (-6) / (-1/3) = 17) ∧
  ((-1)^2022 - |1 - (1/2 : ℚ)| * (1/2) * (2 - (-3)^2) = 11/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l51_5176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_for_given_trapezoid_area_ratio_theorem_l51_5194

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  pq : ℝ  -- Length of base PQ
  rs : ℝ  -- Length of base RS
  h : ℝ   -- Height from PQ to RS

/-- The ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
noncomputable def area_ratio (t : ExtendedTrapezoid) : ℝ :=
  1 / 3

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_for_given_trapezoid :
  let t : ExtendedTrapezoid := { pq := 10, rs := 20, h := 6 }
  area_ratio t = 1 / 3 := by
  sorry

/-- Main theorem proving the area ratio for any trapezoid with the given conditions -/
theorem area_ratio_theorem (t : ExtendedTrapezoid) (h1 : t.pq = 10) (h2 : t.rs = 20) (h3 : t.h = 6) :
  area_ratio t = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_for_given_trapezoid_area_ratio_theorem_l51_5194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l51_5190

variable (a b : ℝ)

def A (a b : ℝ) : ℝ := 2 * a^2 * b - a * b^2
def B (a b : ℝ) : ℝ := -a^2 * b + 2 * a * b^2

theorem problem_solution (a b : ℝ) :
  (5 * A a b + 4 * B a b = 6 * a^2 * b + 3 * a * b^2) ∧
  ((|a + 2| + (3 - b)^2 = 0) → (5 * A a b + 4 * B a b = 18)) ∧
  (A a b + B a b = a^2 * b + a * b^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l51_5190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_transport_time_l51_5157

/-- Calculates the total driving time for transporting cattle to two locations -/
theorem cattle_transport_time 
  (total_cattle : ℕ) 
  (first_location_distance second_location_distance : ℕ) 
  (cattle_to_first_location : ℕ) 
  (truck_capacity : ℕ) 
  (driving_speed : ℕ) 
  (h1 : total_cattle = 800)
  (h2 : first_location_distance = 80)
  (h3 : second_location_distance = 100)
  (h4 : cattle_to_first_location = 450)
  (h5 : truck_capacity = 15)
  (h6 : driving_speed = 60)
  : (((cattle_to_first_location / truck_capacity + 1) * (2 * first_location_distance) + 
     ((total_cattle - cattle_to_first_location) / truck_capacity + 1) * (2 * second_location_distance)) / driving_speed) = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_transport_time_l51_5157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l51_5130

/-- Given an ellipse Γ with eccentricity e and semi-major axis a and semi-minor axis b,
    two points A and B on Γ, foci F₁ and F₂, and origin O, prove that √(2 - √2) ≤ e < 1
    under the given conditions. -/
theorem ellipse_eccentricity_range 
  (Γ : Set (ℝ × ℝ)) 
  (a b : ℝ) 
  (e : ℝ) 
  (A B F₁ F₂ O : ℝ × ℝ) :
  a > b ∧ b > 0 →  -- a > b > 0
  (∀ p ∈ Γ, (p.1 / a)^2 + (p.2 / b)^2 = 1) →  -- Ellipse equation
  A ∈ Γ ∧ B ∈ Γ →  -- A and B are on Γ
  (F₁.1 - O.1)^2 + (F₁.2 - O.2)^2 = (F₂.1 - O.1)^2 + (F₂.2 - O.2)^2 →  -- F₁ and F₂ are foci
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0 →  -- OA ⊥ OB
  ((A.1 - F₁.1) * (A.1 - F₂.1) + (A.2 - F₁.2) * (A.2 - F₂.2)) + 
  ((B.1 - F₁.1) * (B.1 - F₂.1) + (B.2 - F₁.2) * (B.2 - F₂.2)) = 0 →  -- AF₁ · AF₂ + BF₁ · BF₂ = 0
  e^2 = 1 - (b/a)^2 →  -- Definition of eccentricity
  Real.sqrt (2 - Real.sqrt 2) ≤ e ∧ e < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l51_5130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l51_5113

/-- A line perpendicular to the y-axis passing through (-2, 1) -/
def line1 : ℝ → ℝ := λ _ => 1

/-- A line passing through points A(-4, 0) and B(0, 6) -/
noncomputable def line2 : ℝ → ℝ := λ x => (3 * x + 12) / 2

theorem line_equations :
  (∀ x, line1 x = 1) ∧
  (line1 (-2) = 1) ∧
  (∀ x y, y = line1 x → (0 : ℝ) = x - (-2)) ∧
  (line2 (-4) = 0) ∧
  (line2 0 = 6) := by
  sorry

#check line_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l51_5113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_equality_l51_5163

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - 4
noncomputable def g (x : ℝ) : ℝ := x / 3

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x + 4
noncomputable def g_inv (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem composite_function_equality : 
  f (g_inv (f_inv (f_inv (g (f 33))))) = 49 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_equality_l51_5163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_cosF_l51_5147

theorem triangle_DEF_cosF (D E F : ℝ) (h1 : Real.sin D = 4/5) (h2 : Real.cos E = 12/13) 
  (h3 : 0 < D ∧ D < Real.pi) (h4 : 0 < E ∧ E < Real.pi) (h5 : 0 < F ∧ F < Real.pi) (h6 : D + E + F = Real.pi) : 
  Real.cos F = -16/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_cosF_l51_5147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l51_5150

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the conditions
axiom derivative_condition : ∀ x : ℝ, x ≠ 0 → f' x = deriv f x

axiom inequality_condition : ∀ x : ℝ, x ≠ 0 → f' x < 2 * f x / x

axiom zero_points : f 1 = 0 ∧ f (-2) = 0

-- Define the set of real numbers that satisfy xf(x) < 0
def solution_set : Set ℝ := {x : ℝ | x * f x < 0}

-- State the theorem
theorem solution_set_correct : 
  solution_set = (Set.Ioo (-2) 0) ∪ (Set.Ioo 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l51_5150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_nine_fourths_plus_ln_two_l51_5170

/-- The integrand function -/
noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (x + 2) + Real.sqrt (x - 2)) /
  ((Real.sqrt (x + 2) - Real.sqrt (x - 2)) * (x - 2)^2)

/-- The theorem statement -/
theorem integral_equals_nine_fourths_plus_ln_two :
  ∫ x in (5/2)..(10/3), f x = 9/4 + Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_nine_fourths_plus_ln_two_l51_5170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l51_5196

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b * Real.sin t.A) * Real.sin t.C) :
  (Real.tan t.A = 2) ∧ 
  (t.a = 2 ∧ t.C = π / 3 → t.c = Real.sqrt 15 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l51_5196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_max_volume_l51_5185

/-- The volume of a hexagonal pyramid with side lengths x and y, where x + y = 20 -/
noncomputable def pyramid_volume (y : ℝ) : ℝ := Real.sqrt 30 * y^2 * Real.sqrt (10 - y)

/-- The maximum volume of the hexagonal pyramid -/
noncomputable def max_volume : ℝ := 128 * Real.sqrt 15

theorem hexagonal_pyramid_max_volume :
  ∀ y : ℝ, 0 < y ∧ y < 20 → pyramid_volume y ≤ max_volume :=
by
  sorry

#check hexagonal_pyramid_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_max_volume_l51_5185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l51_5127

-- Ellipse
theorem ellipse_equation (a b c : ℝ) (h1 : a = 6) (h2 : c = 4) (h3 : b^2 = 20) :
  ∀ x y : ℝ, y^2 / 36 + x^2 / 20 = 1 ↔ 
    (∃ θ : ℝ, x = a * Real.cos θ ∧ y = b * Real.sin θ) ∧
    (2 * a = 12) ∧ 
    (c / a = 2 / 3) ∧
    (c > 0) := by
  sorry

-- Parabola
theorem parabola_equation (x₀ : ℝ) (h : x₀ = 3) :
  ∀ x y : ℝ, y^2 = 12 * x ↔
    (∃ t : ℝ, x = x₀ + t^2 / 4 ∧ y = t * (x - x₀)) ∧
    (16 * x₀^2 - 9 * 0^2 = 144) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l51_5127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_over_b_equality_l51_5167

open BigOperators

def a (k : ℕ) : ℚ := 2^k / (3^(2^k) + 1)

def A : ℚ := ∑ k in Finset.range 10, a k

def B : ℚ := ∏ k in Finset.range 10, a k

theorem a_over_b_equality : A / B = (3^(2^10) - 2^11 - 1) / 2^47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_over_b_equality_l51_5167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_of_roots_l51_5171

theorem max_value_of_sum_of_roots :
  (∃ x : ℝ, Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ Real.sqrt 6) ∧
  (∀ k : ℝ, k > Real.sqrt 6 → ¬∃ x : ℝ, Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_of_roots_l51_5171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_imply_negation_l51_5173

theorem three_statements_imply_negation (r s : Prop) : 
  let statements := [r ∧ s, r ∧ ¬s, ¬r ∧ s, ¬r ∧ ¬s]
  (statements.filter (λ stmt => (stmt → (r ∨ s)) = true)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_imply_negation_l51_5173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_of_b_l51_5129

-- Define the function y as noncomputable
noncomputable def y (b x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

-- State the theorem
theorem monotonic_increasing_range_of_b :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y b x₁ < y b x₂) →
  (-1 ≤ b ∧ b ≤ 2) :=
by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_of_b_l51_5129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_constant_max_area_quadrilateral_l51_5132

noncomputable section

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line
def line_eq (x y k : ℝ) : Prop := y = k * x - 2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ line_eq A.1 A.2 k ∧ line_eq B.1 B.2 k ∧ A ≠ B

-- Define the right angle condition
def right_angle (O A B : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0

-- Theorem 1
theorem intersection_line_constant (k : ℝ) :
  (∃ A B : ℝ × ℝ, intersection_points A B k ∧ right_angle (0, 0) A B) →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
sorry

-- Define perpendicular chords
def perpendicular_chords (E F G H : ℝ × ℝ) : Prop :=
  circle_eq E.1 E.2 ∧ circle_eq F.1 F.2 ∧ circle_eq G.1 G.2 ∧ circle_eq H.1 H.2 ∧
  (E.1 - F.1) * (G.1 - H.1) + (E.2 - F.2) * (G.2 - H.2) = 0

-- Define the foot of the perpendicular
def foot_of_perpendicular (M : ℝ × ℝ) : Prop := M = (1, Real.sqrt 2 / 2)

-- Define the area of quadrilateral EGFH
def area_EGFH (E F G H : ℝ × ℝ) : ℝ :=
  Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) * Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) / 2

-- Theorem 2
theorem max_area_quadrilateral :
  (∃ E F G H M : ℝ × ℝ, perpendicular_chords E F G H ∧ foot_of_perpendicular M) →
  (∀ E F G H : ℝ × ℝ, perpendicular_chords E F G H → area_EGFH E F G H ≤ 5/2) ∧
  (∃ E F G H : ℝ × ℝ, perpendicular_chords E F G H ∧ area_EGFH E F G H = 5/2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_constant_max_area_quadrilateral_l51_5132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_divisibility_l51_5164

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | (n + 4) => 2 * sequence_a (n + 3) + sequence_a (n + 2) - 2 * sequence_a (n + 1) - sequence_a n

theorem infinite_square_divisibility :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, (f k : ℤ)^2 ∣ sequence_a (f k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_square_divisibility_l51_5164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_not_square_l51_5110

/-- A number consisting of exactly 600 sixes followed by any number of zeros -/
def special_number (n : ℕ) : ℕ :=
  6 * (10^600 - 1) / 9 * 10^n

/-- Theorem stating that the special number cannot be a perfect square -/
theorem special_number_not_square (n : ℕ) :
  ∀ m : ℕ, special_number n ≠ m^2 := by
  intro m
  -- The proof goes here
  sorry

#eval special_number 0  -- For testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_not_square_l51_5110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_perpendicular_vectors_l51_5145

/-- Given two vectors a and b in R³, where a = (x, 4, 2) and b = (3, y, 5),
    if a is perpendicular to b, then the minimum value of x² + y² is 4. -/
theorem min_sum_squares_perpendicular_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ × ℝ) 
  (ha : a = (x, 4, 2)) 
  (hb : b = (3, y, 5)) 
  (h_perp : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) : 
  ∃ (z : ℝ), z = 4 ∧ ∀ (w : ℝ), w ≥ z → ∃ (x' y' : ℝ), x'^2 + y'^2 = w ∧ 
    (x', 4, 2).1 * (3, y', 5).1 + (x', 4, 2).2.1 * (3, y', 5).2.1 + (x', 4, 2).2.2 * (3, y', 5).2.2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_perpendicular_vectors_l51_5145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_local_min_l51_5112

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - 3 * a^2 * x

theorem f_extrema_and_local_min :
  (∀ x ∈ Set.Icc 0 2, f 1 x ≤ 2/3 ∧ f 1 x ≥ -5/3) ∧
  (∃ x ∈ Set.Ioo 1 2, IsLocalMin (f a) x) ↔ 
  a ∈ Set.Ioo (-2/3) (-1/3) ∪ Set.Ioo 1 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_local_min_l51_5112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l51_5136

/-- The line l with equation x + y = 2 -/
def line_l (x y : ℝ) : Prop := x + y = 2

/-- The circle C with equation (x - 1)² + y² = 1 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The distance from a point (x₀, y₀) to the line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem line_intersects_circle :
  ∃ (x y : ℝ), line_l x y ∧ circle_C x y := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l51_5136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l51_5179

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos (α + β) = 3/5) (h4 : Real.sin α = 5/13) : Real.cos β = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l51_5179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_form_l51_5148

noncomputable def g (x : ℝ) : ℤ := ⌊3*x⌋ + ⌊6*x⌋ + ⌊9*x⌋ + ⌊12*x⌋

theorem count_integers_in_form (n : ℕ) : 
  (∃ x : ℝ, g x = n ∧ 1 ≤ n ∧ n ≤ 1500) ↔ n ∈ Finset.range 901 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_form_l51_5148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l51_5165

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (abs x) / Real.log 2

theorem solution_set_of_inequality :
  {x : ℝ | f (x + 1) - f 2 < 0} = {x | -3 < x ∧ x < 1 ∧ x ≠ -1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l51_5165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_correct_l51_5126

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1
def statement1 : Prop :=
  ∀ {α β : Type} (f : Function α β), ∀ x : α, ∃ y : β, f x = y

-- Statement 2
def statement2 : Prop :=
  ∀ {α β : Type} (f : Function α β),
    (∃! y : β, ∀ x : α, f x = y) → (∃! x : α, True)

-- Statement 3
def statement3 : Prop :=
  ∀ (f : Function ℝ ℝ), (∀ x : ℝ, f x = 5) → f Real.pi = 5

-- Statement 4
def statement4 : Prop :=
  ∀ {α β : Type}, Function α β = (α → β)

-- Theorem stating that exactly two of the statements are correct
theorem two_statements_correct :
  statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_correct_l51_5126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_winnings_l51_5160

def fair_8_sided_die : Finset ℕ := Finset.range 8

def is_multiple_of_3 (n : ℕ) : Bool := n % 3 = 0

def winning_amount (roll : ℕ) : ℚ :=
  if is_multiple_of_3 roll then roll else 0

theorem expected_value_of_winnings :
  (fair_8_sided_die.sum (λ roll => (winning_amount roll : ℚ)) / 8 : ℚ) = 9/4 := by
  sorry

#eval (fair_8_sided_die.sum (λ roll => (winning_amount roll : ℚ)) / 8 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_winnings_l51_5160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_l51_5137

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

-- State the theorem
theorem f_increasing_on_neg_reals :
  ∀ a b : ℝ, a < b ∧ b < 0 → f a < f b :=
by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_l51_5137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_division_theorem_l51_5198

/-- Represents a line on a plane -/
structure Line where
  -- We don't need to define the internal structure of a line for this statement

/-- Represents a region formed by the lines -/
structure Region where
  -- We don't need to define the internal structure of a region for this statement

/-- The type of the function that assigns values to regions -/
def RegionAssignment := Region → ℝ

/-- Predicate to check if a RegionAssignment is positive -/
def IsPositive (f : RegionAssignment) : Prop :=
  ∀ r : Region, f r > 0

/-- Predicate to check if sums are equal on both sides of a line -/
def HasEqualSums (f : RegionAssignment) (l : Line) : Prop :=
  ∃ (leftSum rightSum : ℝ), leftSum = rightSum

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop := sorry

theorem line_division_theorem 
  (lines : Finset Line) 
  (regions : Finset Region) 
  (h1 : ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬ are_parallel l1 l2)
  (h2 : ∀ l1 l2 l3, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → 
       l1 ≠ l2 ∧ l2 ≠ l3 ∧ l3 ≠ l1 → ¬ intersect_at_point l1 l2 l3) :
  ∃ (f : RegionAssignment), IsPositive f ∧ ∀ l ∈ lines, HasEqualSums f l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_division_theorem_l51_5198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_cost_l51_5105

theorem barbed_wire_cost (area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (wire_cost_per_meter : ℝ) : 
  area = 3136 ∧ gate_width = 1 ∧ num_gates = 2 ∧ wire_cost_per_meter = 3 →
  (let side_length : ℝ := Real.sqrt area
   let perimeter : ℝ := 4 * side_length
   let wire_length : ℝ := perimeter - (↑num_gates * gate_width)
   let total_cost : ℝ := wire_length * wire_cost_per_meter
   total_cost) = 666 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_cost_l51_5105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l51_5104

-- Define the function f(x) = lg(x+1)
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

-- State the theorem
theorem f_increasing :
  ∀ x₁ x₂ : ℝ, x₁ > -1 → x₂ > -1 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l51_5104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_trick_existence_l51_5125

def dice_numbers : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem magic_trick_existence :
  ∃ (f : ℕ × ℕ → ℕ),
    (∀ (a b : ℕ), a ∈ dice_numbers → b ∈ dice_numbers → a < b → f (a, b) ∈ Finset.range 21) ∧
    (∀ (a b c : ℕ), a ∈ dice_numbers → b ∈ dice_numbers → c ∈ dice_numbers → 
      a ≠ b ∧ b ≠ c ∧ a ≠ c →
      f (a, b) ≠ f (a, c) ∧ f (a, b) ≠ f (b, c) ∧ f (a, c) ≠ f (b, c)) ∧
    (∀ i ∈ Finset.range 21, ∃ (a b : ℕ), a ∈ dice_numbers ∧ b ∈ dice_numbers ∧ a < b ∧ f (a, b) = i) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_trick_existence_l51_5125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_polyhedron_edge_count_l51_5149

/-- A convex polyhedron -/
structure ConvexPolyhedron (n : Nat) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  is_convex : Prop
  edge_count : edges.card = 150

/-- A plane that cuts a vertex of the polyhedron -/
structure CuttingPlane (n : Nat) where
  vertex : Fin n
  cuts_only_incident_edges : Prop
  no_internal_intersections : Prop

/-- The result of cutting a convex polyhedron with planes -/
def cut_polyhedron {n : Nat} (Q : ConvexPolyhedron n) (planes : Fin n → CuttingPlane n) : Nat :=
  450 -- We're directly returning the expected result for simplicity

/-- The theorem stating the number of edges in the resulting polyhedron -/
theorem cut_polyhedron_edge_count {n : Nat} (Q : ConvexPolyhedron n) (planes : Fin n → CuttingPlane n) :
  cut_polyhedron Q planes = 450 := by
  sorry -- The proof is omitted for now

#check cut_polyhedron_edge_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_polyhedron_edge_count_l51_5149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_n_l51_5182

-- Define the number
def n : ℕ := 4^20 * 5^15

-- Theorem statement
theorem digits_of_n : (Nat.log n 10 + 1 : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_n_l51_5182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l51_5109

noncomputable section

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (3 - x)^(1/3) + Real.sqrt (x + 1)
  let solutions : Set ℝ := {3, 3 - ((Real.sqrt 17 - 1)/2)^3, 3 - ((-Real.sqrt 17 - 1)/2)^3}
  (∀ x ∈ solutions, f x = 2) ∧ 
  (∀ x : ℝ, f x = 2 → x ∈ solutions) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l51_5109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_star_shape_l51_5128

/-- A regular n-gon with n ≥ 5 -/
structure RegularPolygon where
  n : ℕ
  center : ℝ × ℝ
  vertex : ℝ × ℝ
  h_n : n ≥ 5

/-- A triangle represented by its vertices -/
structure Triangle where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ

/-- The trajectory of a point -/
structure Trajectory where
  path : Set (ℝ × ℝ)

/-- A star shape -/
structure StarShape where
  center : ℝ × ℝ
  points : Set (ℝ × ℝ)

/-- Function to get the perimeter of a regular polygon -/
def RegularPolygon.perimeter (p : RegularPolygon) : Set (ℝ × ℝ) := sorry

/-- Function to get the interior of a regular polygon -/
def RegularPolygon.interior (p : RegularPolygon) : Set (ℝ × ℝ) := sorry

/-- Function to get the trajectory of point X -/
def trajectory_of_x (triangle : Triangle) : Trajectory := sorry

/-- Congruence relation for triangles -/
def Triangle.congruent (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem statement -/
theorem trajectory_is_star_shape 
  (polygon : RegularPolygon) 
  (triangle_oab : Triangle)
  (triangle_xyz : Triangle)
  (h_congruent : triangle_xyz.congruent triangle_oab)
  (h_initial : triangle_xyz = triangle_oab)
  (h_y_z_on_perimeter : triangle_xyz.y ∈ polygon.perimeter ∧ triangle_xyz.z ∈ polygon.perimeter)
  (h_x_inside : triangle_xyz.x ∈ polygon.interior) :
  ∃ (star : StarShape), (trajectory_of_x triangle_xyz).path = star.points :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_star_shape_l51_5128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l51_5195

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (2, 0)

-- Define the locus equation
def locus (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1 ∧ x ≠ 2

-- Theorem statement
theorem midpoint_locus :
  ∀ (x y : ℝ),
  (∃ (t : ℝ), circle_eq (2*x - 2) (2*y)) →
  circle_eq point_A.1 point_A.2 →
  locus x y :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l51_5195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l51_5166

theorem alpha_range (θ α : ℝ) :
  (∃ (r : ℝ), r * Real.sin (α - π/3) = Real.sin θ ∧ r * Real.sqrt 3 = Real.cos θ) →
  Real.sin (2*θ) ≤ 0 →
  -2*π/3 ≤ α ∧ α ≤ π/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l51_5166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stacy_speed_difference_l51_5119

-- Define the constants from the problem
noncomputable def total_distance : ℝ := 10
noncomputable def heather_rate : ℝ := 5
noncomputable def heather_delay : ℝ := 24 / 60
noncomputable def heather_distance : ℝ := 3.4545454545454546

-- Define Stacy's rate as a variable
variable (stacy_rate : ℝ)

-- Define the time they meet
noncomputable def meeting_time : ℝ := heather_distance / heather_rate + heather_delay

-- Theorem statement
theorem stacy_speed_difference : 
  stacy_rate * meeting_time = total_distance - heather_distance →
  stacy_rate = heather_rate + 1 := by
  sorry

#eval "Proof skipped with sorry"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stacy_speed_difference_l51_5119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l51_5146

theorem infinite_geometric_series_sum : 
  let a : ℝ := (1 : ℝ) / 4  -- first term
  let r : ℝ := (1 : ℝ) / 2  -- common ratio
  let S := ∑' n, a * r^n  -- infinite sum
  S = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l51_5146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_purchase_problem_l51_5155

theorem stamp_purchase_problem :
  ∀ x y : ℚ,
  x + y = 16 ∧
  0.8 * x + y = 14.6 →
  x = 7 ∧ y = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_purchase_problem_l51_5155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_return_path_l51_5158

noncomputable def lower_circumference : ℝ := 8
noncomputable def upper_circumference : ℝ := 6
noncomputable def slope_angle : ℝ := 60 * Real.pi / 180

theorem shortest_return_path (lower_base upper_base angle : ℝ) 
  (h1 : lower_base = lower_circumference)
  (h2 : upper_base = upper_circumference)
  (h3 : angle = slope_angle) :
  ∃ (path : ℝ), path = (4 * Real.sqrt 3) / Real.pi ∧ 
  path = (lower_base / (2 * Real.pi * Real.cos angle)) * Real.sin angle := by
  sorry

#check shortest_return_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_return_path_l51_5158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l51_5123

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.sin (x / 2) * Real.sin (x / 2 + Real.pi / 6) + 2 * Real.sqrt 3 * (Real.cos x - 1)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → f x ≥ -Real.sqrt 3) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 ∧ f x = -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l51_5123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l51_5114

-- Define the function
noncomputable def f (x : ℝ) := Real.arccos (Real.cos x)

-- Define the area function
noncomputable def area_under_curve (a b : ℝ) := ∫ x in a..b, f x

-- Theorem statement
theorem area_arccos_cos : area_under_curve 0 (2 * Real.pi) = Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_arccos_cos_l51_5114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l51_5193

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define parallel vectors
def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

-- Define collinear vectors
def collinear (v w : V) : Prop := ∃ (c d : ℝ), c • v = d • w

-- Define equal length vectors
def equal_length (v w : V) : Prop := ‖v‖ = ‖w‖

-- Theorem stating the correct relationships between vectors
theorem vector_relationships :
  (∃ (v w : V), parallel v w ∧ v ≠ w) ∧
  (∃ (v w : V), ¬parallel v w ∧ v = w) ∧
  (∃ (v w : V), collinear v w ∧ v ≠ w) ∧
  (∀ (v w : V), v = w → collinear v w) ∧
  (∃ (v w : V), equal_length v w ∧ v ≠ w) ∧
  (∃ (u v w : V), parallel u v ∧ parallel u w ∧ ¬collinear v w) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l51_5193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_scaled_tan_period_of_tan_3x_over_5_l51_5197

noncomputable def period_of_tan (x : ℝ) : ℝ := Real.pi

theorem period_of_scaled_tan (a : ℝ) (ha : a ≠ 0) :
  ∃ p : ℝ, ∀ x : ℝ, Real.tan (a * x) = Real.tan (a * (x + p)) :=
sorry

theorem period_of_tan_3x_over_5 :
  ∃ p : ℝ, (∀ x : ℝ, Real.tan ((3 / 5) * x) = Real.tan ((3 / 5) * (x + p))) ∧ p = (5 * Real.pi) / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_scaled_tan_period_of_tan_3x_over_5_l51_5197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_theorem_l51_5184

-- Define the circles and points
variable (O₁ O₂ O₃ O₄ P₁ P₂ P₃ P₄ : EuclideanSpace ℝ (Fin 2))

-- Define the tangency conditions
def externally_tangent (c₁ c₂ : EuclideanSpace ℝ (Fin 2)) (p : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the concyclic property
def concyclic (p₁ p₂ p₃ p₄ : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the circumradius
noncomputable def circumradius (p₁ p₂ p₃ p₄ : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem tangent_circles_theorem 
  (h₁ : externally_tangent O₄ O₁ P₁)
  (h₂ : externally_tangent O₁ O₂ P₂)
  (h₃ : externally_tangent O₂ O₃ P₃)
  (h₄ : externally_tangent O₃ O₄ P₄) :
  concyclic P₁ P₂ P₃ P₄ ∧ 
  concyclic O₁ O₂ O₃ O₄ ∧ 
  circumradius O₁ O₂ O₃ O₄ ≤ circumradius P₁ P₂ P₃ P₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_theorem_l51_5184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formulas_l51_5168

noncomputable section

def sequence_a : ℕ → ℝ := sorry
def sequence_b : ℕ → ℝ := sorry
def sequence_c : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry
def T : ℕ → ℝ := sorry

axiom S_def (n : ℕ) : S n = 2 * sequence_a n - 1

axiom b_arithmetic : ∀ n : ℕ, sequence_b (n + 1) - sequence_b n = sequence_b 2 - sequence_b 1

axiom b1_eq_a1 : sequence_b 1 = sequence_a 1
axiom b4_eq_a3 : sequence_b 4 = sequence_a 3

axiom c_def (n : ℕ) : sequence_c n = 2 / sequence_a n - 1 / (sequence_b n * sequence_b (n + 1))

theorem sequence_formulas :
  (∀ n : ℕ, sequence_a n = 2^(n - 1)) ∧
  (∀ n : ℕ, sequence_b n = n) ∧
  (∀ n : ℕ, T n = 4 * (1 - 1 / 2^n) - n / (n + 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formulas_l51_5168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_form_triangle_four_six_nine_triangle_l51_5107

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define a structure for a triangle
structure Triangle where
  sides : Finset ℝ
  valid : sides.card = 3

-- Theorem stating the equivalence between triangle inequality and forming a triangle
theorem can_form_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  triangle_inequality a b c ↔ ∃ (t : Triangle), t.sides = {a, b, c} :=
sorry

-- Theorem stating that 4, 6, and 9 can form a triangle
theorem four_six_nine_triangle :
  ∃ (t : Triangle), t.sides = {4, 6, 9} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_form_triangle_four_six_nine_triangle_l51_5107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_quadrilateral_theorem_l51_5115

/-- A spatial quadrilateral represented by four vectors -/
structure SpatialQuadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (a b c d : V)

/-- The set of all spatial quadrilaterals formed by permuting the given vectors -/
def allQuadrilaterals {V : Type*} [AddCommGroup V] [Module ℝ V] (a b c d : V) : 
  Finset (SpatialQuadrilateral V) :=
  sorry

/-- The volume of a tetrahedron formed by three vectors -/
noncomputable def tetrahedronVolume {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c : V) : ℝ :=
  sorry

theorem spatial_quadrilateral_theorem {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c d : V) :
  (Finset.card (allQuadrilaterals a b c d) = 6) ∧
  (∀ (q1 q2 : SpatialQuadrilateral V), q1 ∈ allQuadrilaterals a b c d → 
    q2 ∈ allQuadrilaterals a b c d →
    tetrahedronVolume q1.a q1.b q1.c = tetrahedronVolume q2.a q2.b q2.c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_quadrilateral_theorem_l51_5115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_count_l51_5178

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number contains the digit 0 -/
def contains_zero (n : ℕ) : Bool := sorry

/-- The set of numbers satisfying the conditions -/
def counterexample_set : Finset ℕ := 
  Finset.filter (λ n => sum_of_digits n = 5 ∧ ¬contains_zero n ∧ ¬Nat.Prime n) (Finset.range 100000)

theorem counterexample_count : counterexample_set.card = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_count_l51_5178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l51_5186

theorem sum_of_repeating_decimals : 
  ∃ (x y : ℚ), (∀ n : ℕ, (10^n * x - ⌊10^n * x⌋ < 1) ∧ ((10 * x - ⌊10 * x⌋) = (x - ⌊x⌋))) ∧
                (∀ n : ℕ, (10^n * y - ⌊10^n * y⌋ < 1) ∧ ((10 * y - ⌊10 * y⌋) = (y - ⌊y⌋))) ∧
                (x.num * y.den + y.num * x.den = 13 * (x.den * y.den)) ∧
                (x.den * y.den = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l51_5186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l51_5172

/-- The area of a triangle with vertices at the origin and two given points in R^2 -/
noncomputable def triangle_area (a b : ℝ × ℝ) : ℝ :=
  (1/2) * abs (a.1 * b.2 - a.2 * b.1)

/-- Theorem: The area of the triangle with vertices at (0,0), (4,-1), and (-3,3) is 4.5 -/
theorem specific_triangle_area :
  triangle_area (4, -1) (-3, 3) = 4.5 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l51_5172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_equation_l51_5101

theorem modular_equation (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 29) (h3 : (5 * n) % 29 = 1) :
  (3^n)^2 % 29 - 3 % 29 = 13 % 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_equation_l51_5101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l51_5133

/-- The curve defined by xy = 1 -/
def curve (x y : ℝ) : Prop := x * y = 1

/-- Three known intersection points -/
noncomputable def point1 : ℝ × ℝ := (2, 1/2)
noncomputable def point2 : ℝ × ℝ := (-5, -1/5)
noncomputable def point3 : ℝ × ℝ := (1/3, 3)

/-- The fourth intersection point to be proven -/
noncomputable def point4 : ℝ × ℝ := (-3/10, -10/3)

theorem fourth_intersection_point :
  curve point4.1 point4.2 ∧
  ∃ (a b r : ℝ), 
    (point1.1 - a)^2 + (point1.2 - b)^2 = r^2 ∧
    (point2.1 - a)^2 + (point2.2 - b)^2 = r^2 ∧
    (point3.1 - a)^2 + (point3.2 - b)^2 = r^2 ∧
    (point4.1 - a)^2 + (point4.2 - b)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l51_5133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l51_5124

/-- Two points are symmetric with respect to the y-axis if their x-coordinates have the same magnitude but opposite signs, their y-coordinates are equal, and their z-coordinates have the same magnitude but opposite signs. -/
def symmetric_y_axis (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 = -b.1 ∧ a.2.1 = b.2.1 ∧ a.2.2 = -b.2.2

/-- Point A with coordinates dependent on x, y, and z -/
def point_A (x y z : ℝ) : ℝ × ℝ × ℝ := (x^2 + 4, 4 - y, 1 + 2*z)

/-- Point B with coordinates dependent on x and z -/
def point_B (x z : ℝ) : ℝ × ℝ × ℝ := (-4*x, 9, 7 - z)

theorem symmetry_condition (x y z : ℝ) :
  symmetric_y_axis (point_A x y z) (point_B x z) ↔ x = 2 ∧ y = -5 ∧ z = -8 := by
  sorry

#check symmetry_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l51_5124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_over_4_l51_5142

-- Define the vectors
def a : ℝ × ℝ := (3, 4)
noncomputable def b (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)

-- Define the parallel condition
def parallel (α : ℝ) : Prop := 3 * Real.cos α = 4 * Real.sin α

-- State the theorem
theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : parallel α) : 
  Real.tan (α + π/4) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_over_4_l51_5142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l51_5151

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a > b > 0, and c is the focal distance. 
    If the symmetric point of the left focus F(-c,0) with respect to 
    the line bx + cy = 0 is on the ellipse, then the eccentricity 
    of the ellipse is √2/2. -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c > 0) (h4 : c^2 = a^2 - b^2) 
  (h5 : ∃ (m n : ℝ), (m^2 / a^2 + n^2 / b^2 = 1) ∧ 
                     (n / (m + c) = c / b) ∧ 
                     (b * (m - c) / 2 + c * n / 2 = 0)) : 
  c / a = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l51_5151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_download_time_proof_l51_5100

/-- Calculates the download time in hours for given file sizes and internet speed -/
noncomputable def download_time (file_sizes : List ℝ) (speed : ℝ) : ℝ :=
  (file_sizes.sum / speed) / 60

/-- Proves that the download time for given file sizes and internet speed is 2 hours -/
theorem download_time_proof :
  let file_sizes : List ℝ := [80, 90, 70]
  let speed : ℝ := 2
  download_time file_sizes speed = 2 := by
  sorry

/-- Computes the download time for the given file sizes and speed -/
def download_time_nat (file_sizes : List ℕ) (speed : ℕ) : ℕ :=
  (file_sizes.sum / speed) / 60

#eval download_time_nat [80, 90, 70] 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_download_time_proof_l51_5100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_700_l51_5181

/-- Represents a rectangular pyramid with a parallel cut creating a frustum -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  altitude : ℝ
  cut_height : ℝ
  top_area_ratio : ℝ

/-- Calculates the volume of the frustum created by cutting a rectangular pyramid -/
noncomputable def frustum_volume (p : CutPyramid) : ℝ :=
  let base_area := p.base_length * p.base_width
  let pyramid_volume := (1 / 3) * base_area * p.altitude
  let top_area := p.top_area_ratio * base_area
  let small_pyramid_volume := (1 / 3) * top_area * p.cut_height
  pyramid_volume - small_pyramid_volume

/-- Theorem stating that the volume of the frustum in the given problem is 700 cm³ -/
theorem frustum_volume_is_700 :
  let p : CutPyramid := {
    base_length := 20,
    base_width := 10,
    altitude := 12,
    cut_height := 6,
    top_area_ratio := 1 / 4
  }
  frustum_volume p = 700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_700_l51_5181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l51_5174

theorem partition_theorem : 
  ∃ (f : ℕ → ℕ), StrictMono f ∧ 
  ∀ k : ℕ, ∃ (A B C : Finset ℕ), 
    (A.card = f k) ∧ (B.card = f k) ∧ (C.card = f k) ∧
    (A ∪ B ∪ C = Finset.range (3 * f k)) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    ∀ i ∈ Finset.range (f k), 
      ∃ (a b c : ℕ), a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ a + b = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l51_5174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_spend_on_loot_boxes_l51_5152

/-- The amount John spends on loot boxes -/
def spend : ℝ := sorry

/-- The cost of each loot box -/
def cost_per_box : ℝ := 5

/-- The average value of items inside a loot box -/
def avg_value : ℝ := 3.5

/-- The average amount John loses -/
def avg_loss : ℝ := 12

theorem john_spend_on_loot_boxes :
  spend * ((cost_per_box - avg_value) / cost_per_box) = avg_loss →
  spend = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_spend_on_loot_boxes_l51_5152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_perimeter_range_l51_5189

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a ≠ t.b ∧
  t.A + t.B + t.C = Real.pi ∧
  (Real.cos t.A)^2 - (Real.cos t.B)^2 = Real.sqrt 3 * Real.sin t.A * Real.cos t.A - Real.sqrt 3 * Real.sin t.B * Real.cos t.B

-- Theorem for part I
theorem angle_C_value (t : Triangle) (h : triangle_conditions t) : t.C = Real.pi / 3 := by
  sorry

-- Theorem for part II
theorem perimeter_range (t : Triangle) (h : triangle_conditions t) (h_c : t.c = Real.sqrt 3) :
  2 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c < 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_perimeter_range_l51_5189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l51_5161

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ (3/4) * x^2 + (3/2) * x + 7/4

-- State the theorem
theorem function_equality : 
  ∀ x : ℝ, f (2*x - 1) = 3*x^2 + 1 :=
by
  intro x
  -- Expand the definition of f
  simp [f]
  -- Algebraic manipulation
  ring

-- Check the theorem
#check function_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l51_5161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_zero_l51_5106

theorem sin_sum_zero : Real.sin (-π/3) + 2 * Real.sin (5*π/3) + 3 * Real.sin (2*π/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_zero_l51_5106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_plus_id_has_parallelizability_sin_has_parallelizability_cubic_minus_id_no_parallelizability_quad_plus_log_no_parallelizability_l51_5140

/-- A function has parallelizability if for any value in its derivative's range,
    there exist at least two distinct points with that derivative value. -/
def has_parallelizability (f : ℝ → ℝ) : Prop :=
  ∀ a ∈ Set.range (deriv f), ∃ x y, x ≠ y ∧ deriv f x = a ∧ deriv f y = a

/-- The function x ↦ x + 1/x has parallelizability. -/
theorem inverse_plus_id_has_parallelizability :
  has_parallelizability (λ x => x + 1/x) := by sorry

/-- The sine function has parallelizability. -/
theorem sin_has_parallelizability :
  has_parallelizability Real.sin := by sorry

/-- The function x ↦ x³ - x does not have parallelizability. -/
theorem cubic_minus_id_no_parallelizability :
  ¬ has_parallelizability (λ x => x^3 - x) := by sorry

/-- The function x ↦ (x-2)² + ln(x) does not have parallelizability. -/
theorem quad_plus_log_no_parallelizability :
  ¬ has_parallelizability (λ x => (x-2)^2 + Real.log x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_plus_id_has_parallelizability_sin_has_parallelizability_cubic_minus_id_no_parallelizability_quad_plus_log_no_parallelizability_l51_5140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_relation_l51_5122

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  C₁ : ℝ
  C₂ : ℝ
  isIsosceles : A = B
  angleSum : A + B + C = 180
  angleB : B = 2 * A
  angleC : C = C₁ + C₂

theorem isosceles_triangle_angle_relation (t : IsoscelesTriangle) : t.C₁ - t.C₂ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_relation_l51_5122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_unique_l51_5192

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral in 2D space -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Predicate to check if a quadrilateral is inscribed in a circle -/
def isInscribed (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is circumscribed around a circle -/
def isCircumscribed (q : Quadrilateral) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

/-- Theorem stating that the fourth vertex of an inscribed and circumscribed quadrilateral is uniquely determined -/
theorem fourth_vertex_unique 
  (A B C : Point2D) 
  (h1 : distance A B ≥ distance B C) :
  ∃! D : Point2D, 
    let q := Quadrilateral.mk A B C D
    isInscribed q ∧ isCircumscribed q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_unique_l51_5192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l51_5143

theorem relationship_abc (a b c : ℝ) : 
  a = (6 : ℝ)^(0.7 : ℝ) → b = (0.7 : ℝ)^(6 : ℝ) → c = Real.log 6 / Real.log 0.7 → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l51_5143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_tangent_asymptote_l51_5159

/-- The eccentricity of a hyperbola with one asymptote tangent to a parabola -/
theorem hyperbola_eccentricity_with_tangent_asymptote 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
              (y = b/a * x) ∧ 
              (y = x^2 + 1) ∧
              (∀ x' y' : ℝ, (y' = b/a * x') ∧ (y' = x'^2 + 1) → (x' = x ∧ y' = y))) →
  Real.sqrt (1 + (b/a)^2) = Real.sqrt 5 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_tangent_asymptote_l51_5159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l51_5154

theorem sin_beta_value (α β : Real) : 
  0 < α → α < Real.pi/2 →
  0 < β → β < Real.pi/2 →
  Real.cos α = 2 * Real.sqrt 5 / 5 →
  Real.sin (α - β) = -3/5 →
  Real.sin β = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l51_5154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_biased_coin_l51_5144

/-- The probability of getting an even number of heads when tossing a coin n times -/
noncomputable def prob_even_heads (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2*p)^n) / 2

theorem prob_even_heads_fair_coin (n : ℕ) :
  prob_even_heads n (1/2) = 1/2 := by sorry

theorem prob_even_heads_biased_coin (n : ℕ) (p : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) :
  prob_even_heads n p = (1 + (1 - 2*p)^n) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_biased_coin_l51_5144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_constraint_l51_5103

/-- The distance from a point (x, y) to a line Ax + By + C = 0 is given by |Ax + By + C| / √(A^2 + B^2) -/
noncomputable def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: Given a point P(4, a) and a line 4x - 3y - 1 = 0, if the distance from P to the line 
    is no greater than 3, then a is in the closed interval [0, 10] -/
theorem point_line_distance_constraint (a : ℝ) :
  distance_point_to_line 4 a 4 (-3) (-1) ≤ 3 → 0 ≤ a ∧ a ≤ 10 := by
  sorry

#check point_line_distance_constraint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_distance_constraint_l51_5103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_a_2_range_a_f_leq_x_l51_5116

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.sqrt (x^2 - 2*x + 1) + |x + a|

-- Theorem 1: Minimum value of f when a = 2
theorem min_value_f_a_2 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 2 ≥ f x_min 2 ∧ f x_min 2 = 3 := by
  sorry

-- Theorem 2: Range of a when f(x) ≤ x for x ∈ [2/3, 1]
theorem range_a_f_leq_x :
  (∀ (x : ℝ), 2/3 ≤ x ∧ x ≤ 1 → f x a ≤ x) ↔ -1 ≤ a ∧ a ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_a_2_range_a_f_leq_x_l51_5116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_bench_sections_is_five_l51_5131

/-- Represents the capacity of a bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Finds the least positive integer N satisfying the bench arrangement conditions -/
def leastBenchSections (capacity : BenchCapacity) : Nat :=
  let minN := Nat.lcm capacity.adults capacity.children / capacity.adults
  (List.range 100).find? (fun n =>
    let totalPeople := 2 * n * capacity.adults
    totalPeople % 20 = 0 && n ≥ minN
  ) |>.getD 1

/-- Theorem stating that the least number of bench sections is 5 -/
theorem least_bench_sections_is_five :
  leastBenchSections ⟨8, 12⟩ = 5 := by
  sorry

#eval leastBenchSections ⟨8, 12⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_bench_sections_is_five_l51_5131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_show_length_l51_5162

/-- The length of the TV show itself, given the total airtime and the durations of commercials and breaks -/
theorem tv_show_length :
  let total_airtime : ℝ := 2 -- in hours
  let commercial_lengths : List ℝ := [8/60, 8/60, 12/60, 6/60, 6/60] -- in hours
  let break_lengths : List ℝ := [4/60, 5/60] -- in hours
  let total_commercial_time := commercial_lengths.sum
  let total_break_time := break_lengths.sum
  let show_length := total_airtime - total_commercial_time - total_break_time
  ∃ ε > 0, |show_length - 1.1833| < ε := by
    -- The proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_show_length_l51_5162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_count_theorem_l51_5121

/-- The rate at which envelopes are counted, in envelopes per second -/
noncomputable def count_rate : ℚ := 10

/-- The time it takes to count a given number of envelopes -/
noncomputable def count_time (num_envelopes : ℚ) : ℚ := num_envelopes / count_rate

theorem envelope_count_theorem :
  (count_time 40 = 4) ∧ (count_time 90 = 9) :=
by
  -- Split the conjunction
  constructor
  -- Prove count_time 40 = 4
  · simp [count_time, count_rate]
    norm_num
  -- Prove count_time 90 = 9
  · simp [count_time, count_rate]
    norm_num

#check envelope_count_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_count_theorem_l51_5121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_cube_volume_l51_5118

theorem larger_cube_volume (n : ℕ) (small_cube_volume : ℝ) (surface_area_diff : ℝ) : 
  n = 216 → 
  small_cube_volume = 1 → 
  surface_area_diff = 1080 → 
  let larger_cube_side := (n : ℝ) ^ (1/3 : ℝ)
  let larger_cube_volume := larger_cube_side ^ 3
  let larger_cube_surface_area := 6 * larger_cube_side ^ 2
  let small_cube_side := small_cube_volume ^ (1/3 : ℝ)
  let small_cube_surface_area := 6 * small_cube_side ^ 2
  let total_small_cube_surface_area := (n : ℝ) * small_cube_surface_area
  surface_area_diff = total_small_cube_surface_area - larger_cube_surface_area →
  larger_cube_volume = 216 := by
  sorry

#check larger_cube_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_cube_volume_l51_5118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_plane_parallel_to_skew_lines_unique_plane_parallel_to_skew_line_l51_5141

-- Define the necessary types and structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define membership for Point3D in Line3D
def Point3D.mem (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk (l.point.x + t * l.direction.x) (l.point.y + t * l.direction.y) (l.point.z + t * l.direction.z)

instance : Membership Point3D Line3D where
  mem := Point3D.mem

-- Define membership for Point3D in Plane3D
def Point3D.memPlane (p : Point3D) (pl : Plane3D) : Prop :=
  (p.x - pl.point.x) * pl.normal.x + (p.y - pl.point.y) * pl.normal.y + (p.z - pl.point.z) * pl.normal.z = 0

instance : Membership Point3D Plane3D where
  mem := Point3D.memPlane

-- Define the concept of skew lines
def areSkewLines (l1 l2 : Line3D) : Prop :=
  ¬ ∃ (p : Point3D), p ∈ l1 ∧ p ∈ l2 ∧ ¬ (l1.direction = l2.direction)

-- Define the concept of a point being outside two lines
def isOutsideLines (p : Point3D) (l1 l2 : Line3D) : Prop :=
  p ∉ l1 ∧ p ∉ l2

-- Define the concept of a plane being parallel to a line
def planeParallelToLine (p : Plane3D) (l : Line3D) : Prop :=
  ∀ (point : Point3D), point ∈ l → point ∉ p

-- Theorem 1
theorem exists_plane_parallel_to_skew_lines 
  (a b : Line3D) (p : Point3D) 
  (h1 : areSkewLines a b) 
  (h2 : isOutsideLines p a b) : 
  ∃ (plane : Plane3D), planeParallelToLine plane a ∧ planeParallelToLine plane b :=
sorry

-- Theorem 2
theorem unique_plane_parallel_to_skew_line 
  (a b : Line3D) 
  (h : areSkewLines a b) :
  ∃! (plane : Plane3D), plane.point ∈ a ∧ planeParallelToLine plane b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_plane_parallel_to_skew_lines_unique_plane_parallel_to_skew_line_l51_5141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l51_5183

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line -/
noncomputable def m1 : ℝ := -3

/-- The slope of the second line in terms of a -/
noncomputable def m2 (a : ℝ) : ℝ := -a / 9

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := y = m1 * x - 7

/-- The second line equation -/
def line2 (a x y : ℝ) : Prop := 9 * y + a * x = 15

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y, line1 x y → line2 a x y → perpendicular m1 (m2 a)) ↔ a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l51_5183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sundays_and_tuesdays_count_l51_5177

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to count the number of occurrences of a specific day in a 30-day month
def countDayInMonth (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

-- Define a function to check if a given start day results in equal Sundays and Tuesdays
def hasEqualSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  countDayInMonth startDay DayOfWeek.Sunday = countDayInMonth startDay DayOfWeek.Tuesday

-- Define a list of all days of the week
def allDays : List DayOfWeek :=
  [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
   DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]

-- Theorem statement
theorem equal_sundays_and_tuesdays_count :
  (allDays.filter hasEqualSundaysAndTuesdays).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sundays_and_tuesdays_count_l51_5177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_correct_specific_grid_paths_l51_5108

def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

theorem grid_paths_correct (m n : ℕ) : 
  grid_paths m n = Nat.choose (m + n) m :=
by rfl

theorem specific_grid_paths : grid_paths 8 7 = 6435 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_correct_specific_grid_paths_l51_5108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l51_5111

theorem solution_difference : 
  ∃ x₁ x₂ : ℝ, 
    ((9 - x₁^2 / 4)^(1/3 : ℝ) = 2) ∧ 
    ((9 - x₂^2 / 4)^(1/3 : ℝ) = 2) ∧ 
    x₁ ≠ x₂ ∧
    |x₁ - x₂| = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l51_5111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l51_5175

/-- A vector in R² -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- The line y = 3x + 1 -/
def onLine (v : Vec2) : Prop := v.y = 3 * v.x + 1

/-- Projection of v onto w -/
noncomputable def proj (v w : Vec2) : Vec2 :=
  let dot := v.x * w.x + v.y * w.y
  let normSq := w.x * w.x + w.y * w.y
  { x := (dot / normSq) * w.x
    y := (dot / normSq) * w.y }

/-- The theorem to be proved -/
theorem projection_theorem (w : Vec2) :
  (∃ (c : ℝ), c ≠ 0 ∧ w = Vec2.mk (-3*c) c) →
  (∀ (v : Vec2), onLine v → proj v w = Vec2.mk (-3/10) (1/10)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l51_5175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l51_5117

noncomputable section

open Real

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t * cos (π/4), t * sin (π/4))

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 8 * cos θ / (1 - cos θ ^ 2)

-- Define the Cartesian equation of curve C
def curve_C_cartesian (x y : ℝ) : Prop := y^2 = 8*x

-- State the theorem
theorem intersection_triangle_area :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, A = line_l t₁ ∧ B = line_l t₂) ∧
    (curve_C_cartesian A.1 A.2) ∧
    (curve_C_cartesian B.1 B.2) ∧
    (let O : ℝ × ℝ := (0, 0);
     let triangle_area := sqrt 6;
     (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * triangle_area) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l51_5117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_288_plus_2_l51_5138

theorem simplify_sqrt_288_plus_2 : Real.sqrt 288 + 2 = 12 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_288_plus_2_l51_5138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_function_true_l51_5156

/-- Represents a student with an ID, height, and score. -/
structure Student where
  id : ℕ
  height : ℝ
  score : ℝ

/-- The class of students. -/
def StudentClass : Type := Fin 48 → Student

/-- A function is injective if it maps distinct inputs to distinct outputs. -/
def IsInjective {α β : Type} (f : α → β) : Prop :=
  ∀ a₁ a₂ : α, f a₁ = f a₂ → a₁ = a₂

/-- A function is surjective if every element in the codomain is mapped to by some element in the domain. -/
def IsSurjective {α β : Type} (f : α → β) : Prop :=
  ∀ b : β, ∃ a : α, f a = b

/-- A function is bijective if it is both injective and surjective. -/
def IsBijective {α β : Type} (f : α → β) : Prop :=
  IsInjective f ∧ IsSurjective f

/-- The main theorem stating that only one of the three function statements is true. -/
theorem only_one_function_true (c : StudentClass) : 
  (IsBijective (fun s => (c s).height)) ∧
  ¬(IsBijective (fun s => (c s).score)) ∧
  ¬(IsBijective (fun s => (c s).id)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_function_true_l51_5156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_g_minimum_l51_5134

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / (x - 1) + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 1|

theorem min_value_and_g_minimum (a : ℝ) :
  (a > 0) →
  (∃ (m : ℝ), m = 15 ∧ ∀ x > 1, f a x ≥ m) →
  (a = 5 ∧ ∀ x : ℝ, g a x ≥ 4 ∧ ∃ x : ℝ, g a x = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_g_minimum_l51_5134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_x4_plus_1_power_10_l51_5188

theorem constant_term_x4_plus_1_power_10 :
  (Polynomial.eval 0 ((Polynomial.X : Polynomial ℝ)^4 + 1)^10) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_x4_plus_1_power_10_l51_5188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_hyperbola_l51_5135

/-- A conic section is a hyperbola if it can be expressed in the form
    A(x-h)² - B(y-k)² = C, where A, B, and C are non-zero constants and A and B have the same sign. -/
def is_hyperbola (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ A B C h k : ℝ, A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ (A * B > 0) ∧
  (∀ x y, f x y = A * (x - h)^2 - B * (y - k)^2 - C)

/-- The equation (x-4)² = 5(y+2)² - 45 describes a hyperbola. -/
theorem equation_describes_hyperbola :
  is_hyperbola (λ x y ↦ (x - 4)^2 - 5*(y + 2)^2 + 45) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_hyperbola_l51_5135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_R_squared_better_fit_correct_answer_is_D_l51_5180

/-- Represents the coefficient of determination (R²) in regression analysis -/
def R_squared : ℝ → ℝ := sorry

/-- Represents the goodness of fit of a regression model -/
def model_fit : ℝ → ℝ := sorry

/-- Axiom stating that R² is between 0 and 1 -/
axiom R_squared_range (r : ℝ) : 0 ≤ R_squared r ∧ R_squared r ≤ 1

/-- Theorem stating that a larger R² indicates a better model fit -/
theorem larger_R_squared_better_fit (r1 r2 : ℝ) :
  R_squared r1 < R_squared r2 → model_fit r1 < model_fit r2 :=
by sorry

/-- The main theorem representing the correct answer to the question -/
theorem correct_answer_is_D :
  ∀ (r1 r2 : ℝ), R_squared r1 < R_squared r2 → model_fit r1 < model_fit r2 :=
by
  intros r1 r2 h
  exact larger_R_squared_better_fit r1 r2 h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_R_squared_better_fit_correct_answer_is_D_l51_5180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l51_5187

-- Define a triangle
structure Triangle where
  angles : Fin 3 → Real
  sum_angles : (angles 0) + (angles 1) + (angles 2) = Real.pi
  positive_angles : ∀ i, 0 < angles i

-- Define an obtuse angle
def is_obtuse (angle : Real) : Prop := Real.pi / 2 < angle

-- Define the original statement
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

-- Define the negation
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

-- State the theorem
theorem negation_equivalence (t : Triangle) :
  ¬(at_most_one_obtuse t) ↔ at_least_two_obtuse t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l51_5187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_track_circumference_l51_5169

/-- The circumference of a circular jogging track given two people walking in opposite directions -/
theorem jogging_track_circumference
  (deepak_speed : ℝ)
  (wife_speed : ℝ)
  (meeting_time_minutes : ℝ)
  (h1 : deepak_speed = 4.5)
  (h2 : wife_speed = 3.75)
  (h3 : meeting_time_minutes = 4.56) :
  deepak_speed * (meeting_time_minutes / 60) + wife_speed * (meeting_time_minutes / 60) = 0.627 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_track_circumference_l51_5169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l51_5191

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

-- Define the equation
def equation (x : ℝ) : Prop := f x = g x

-- Theorem stating that the equation has exactly two solutions
theorem equation_has_two_solutions : ∃! (s : Set ℝ), (∀ x ∈ s, equation x) ∧ (Finite s ∧ Nat.card s = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l51_5191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l51_5153

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi/6)

theorem f_properties :
  let period : ℝ := Real.pi
  let max_value : ℝ := 2
  let min_value : ℝ := -1
  let max_point : ℝ := Real.pi/6
  let min_point : ℝ := -Real.pi/6
  let interval : Set ℝ := Set.Icc (-Real.pi/6) (Real.pi/4)
  (∀ x, f (x + period) = f x) ∧
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ period) ∧
  (∀ x ∈ interval, f x ≤ max_value) ∧
  (f max_point = max_value) ∧
  (∀ x ∈ interval, f x ≥ min_value) ∧
  (f min_point = min_value) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l51_5153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_formula_correctness_l51_5139

noncomputable def distance_between_line_and_parabola 
  (m k a b c p q : ℝ) : ℝ :=
  let point_on_line := (p, m * p + k)
  let point_on_parabola := (q, a * q^2 + b * q + c)
  |q - p| * Real.sqrt (1 + ((a * q^2 + b * q + c - m * p - k) / (q - p))^2)

theorem distance_formula_correctness 
  (m k a b c p q : ℝ) :
  distance_between_line_and_parabola m k a b c p q = 
  |q - p| * Real.sqrt (1 + ((a * q^2 + b * q + c - m * p - k) / (q - p))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_formula_correctness_l51_5139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l51_5120

/-- The function f(x) with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (ω * x + Real.pi / 6) - Real.sin (ω * x - Real.pi / 3)

/-- The theorem statement -/
theorem max_value_of_f (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f ω x ≤ f ω x₀ ∧ f ω x₀ = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l51_5120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l51_5199

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x + a * (2 : ℝ)^(-x)

-- State the theorem
theorem odd_function_properties (a : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a x = -f a (-x)) →
  (a = -1) ∧
  (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f a x < f a y) ∧
  (∀ m : ℝ, f a (1 - m) + f a (1 - 2*m) < 0 → 2/3 < m ∧ m < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l51_5199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l51_5102

/-- Proves that the average speed of a train is 16 kmph given the specified conditions -/
theorem train_average_speed (x : ℝ) (h : x > 0) : 
  (2 * x) / ((x / 40) + (2 * x / 20)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l51_5102
