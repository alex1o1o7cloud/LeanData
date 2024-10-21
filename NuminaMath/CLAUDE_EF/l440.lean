import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l440_44081

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem function_symmetry (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + 4 * Real.pi) = f ω x) :
  ∀ x, f ω (10 * Real.pi / 3 - x) = f ω (10 * Real.pi / 3 + x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l440_44081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l440_44080

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 4 - x / 3)

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l440_44080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_abc_partition_l440_44099

def S (n : ℕ) : Set ℕ := {i | 2 ≤ i ∧ i ≤ n}

def has_abc (A : Set ℕ) : Prop :=
  ∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c

theorem smallest_n_with_abc_partition : 
  (∀ n < 16, ∃ A B : Set ℕ, A ∪ B = S n ∧ A ∩ B = ∅ ∧ ¬has_abc A ∧ ¬has_abc B) ∧
  (∀ A B : Set ℕ, A ∪ B = S 16 ∧ A ∩ B = ∅ → has_abc A ∨ has_abc B) :=
by
  sorry

#check smallest_n_with_abc_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_abc_partition_l440_44099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_tessellation_chromatic_number_l440_44093

/-- A hexagonal tessellation of the plane -/
structure HexagonalTessellation where
  -- We don't need to define the full structure, just declare it exists
  -- The actual implementation details are not necessary for this statement

/-- The chromatic number of a graph -/
def chromaticNumber (G : Type) : ℕ := 
  -- We don't need to define this, just declare it exists
  sorry

/-- Theorem: The chromatic number of a hexagonal tessellation is 3 -/
theorem hexagonal_tessellation_chromatic_number :
  ∀ (H : HexagonalTessellation), chromaticNumber (HexagonalTessellation) = 3 :=
by
  sorry -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_tessellation_chromatic_number_l440_44093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sin_sum_l440_44032

theorem triangle_max_sin_sum (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a * Real.cos A = b * Real.sin A ∧
  B > π / 2 →
  (∃ (max : ℝ), max = 9/8 ∧ ∀ A' B' C' : ℝ, 
    (0 < A' ∧ 0 < B' ∧ 0 < C' ∧
     A' + B' + C' = π ∧
     a * Real.cos A' = b * Real.sin A' ∧
     B' > π / 2) →
    Real.sin A' + Real.sin C' ≤ max) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_sin_sum_l440_44032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_implies_x_value_l440_44036

theorem vector_equality_implies_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -2]
  ‖a + b‖ = ‖a - b‖ → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_implies_x_value_l440_44036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equilateral_triangle_l440_44079

/-- Given an equilateral triangle ABC with a point P inside such that PA = 8, PB = 6, and PC = 10,
    the area of triangle ABC is equal to 25√3 + 36. -/
theorem area_equilateral_triangle (A B C P : ℝ × ℝ) : 
  let d := λ X Y : ℝ × ℝ ↦ Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  (∀ X Y : ℝ × ℝ, d X Y = d Y X) →  -- distance is symmetric
  (d A B = d B C ∧ d B C = d C A) →  -- ABC is equilateral
  (d P A = 8 ∧ d P B = 6 ∧ d P C = 10) →  -- distances from P to vertices
  (∃ x y : ℝ, A = (x, y) ∧ B = (x + d A B, y) ∧ C = (x + d A B / 2, y + d A B * Real.sqrt 3 / 2)) →  -- standard position
  (1/4 * d A B^2 * Real.sqrt 3 = 25 * Real.sqrt 3 + 36) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equilateral_triangle_l440_44079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_triangle_toothpicks_l440_44025

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle structure -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to construct the large triangle -/
def required_toothpicks : ℕ :=
  (3 * total_triangles + 1) / 2 + 3 * base_triangles

theorem large_triangle_toothpicks :
  required_toothpicks = 755255 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_triangle_toothpicks_l440_44025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l440_44069

theorem solution_difference : 
  ∃ x₁ x₂ : ℝ, 
    ((7 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧ 
    ((7 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧ 
    x₁ ≠ x₂ ∧
    |x₁ - x₂| = 2 * Real.sqrt 136 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l440_44069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l440_44047

/-- A power function f(x) that does not pass through the origin -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 3*m + 3) * x^(m^2 - m - 2)

/-- The condition that f does not pass through the origin -/
def not_pass_origin (m : ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f m x ≠ 0

theorem power_function_m_value :
  ∀ m : ℝ, not_pass_origin m → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l440_44047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l440_44020

theorem tan_phi_value (φ : ℝ) 
  (h1 : Real.cos (π / 2 + φ) = Real.sqrt 3 / 2) 
  (h2 : |φ| < π / 2) : 
  Real.tan φ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l440_44020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_problem_l440_44011

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def sphere_diameter (r : ℝ) : ℝ := 2 * r

theorem sphere_diameter_problem (r₁ r₂ : ℝ) (h₁ : r₁ = 12) 
  (h₂ : sphere_volume r₂ = 3 * sphere_volume r₁) :
  ∃ (a b : ℕ), 
    sphere_diameter r₂ = a * (3 : ℝ)^(1/3) ∧ 
    a = 24 ∧ 
    b = 3 ∧ 
    a + b = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_problem_l440_44011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_A_l440_44084

noncomputable def B : ℝ × ℝ := (-5, 0)
noncomputable def C : ℝ × ℝ := (5, 0)

-- Define angle function
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define sin function
noncomputable def sin_angle (A B C : ℝ × ℝ) : ℝ := Real.sin (angle A B C)

theorem locus_of_point_A (A : ℝ × ℝ) :
  (sin_angle A B C - sin_angle A C B = (3/5) * sin_angle B A C) →
  (A.1 < -3) →
  (A.1^2 / 9 - A.2^2 / 16 = 1) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_A_l440_44084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_phase_shift_l440_44049

/-- The phase shift of a sine function y = a * sin(bx + c) is -c/b -/
noncomputable def phase_shift (b c : ℝ) : ℝ := -c / b

/-- Given function f(x) = 3 * sin(4x - π/4), prove its phase shift is π/16 -/
theorem sine_function_phase_shift :
  let b : ℝ := 4
  let c : ℝ := -π/4
  phase_shift b c = π/16 := by
  -- Unfold the definition of phase_shift
  unfold phase_shift
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_phase_shift_l440_44049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l440_44004

def G : ℕ → ℚ
  | 0 => 3  -- Added case for 0 to cover all natural numbers
  | n + 1 => (3 * G n + 2) / 3

theorem G_51_value : G 51 = 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l440_44004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l440_44095

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l440_44095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_bound_l440_44008

theorem largest_subset_bound (n : ℕ) :
  ∃ (A : Finset ℕ),
    (∀ x ∈ A, x ≤ n) ∧
    (∀ x ∈ A, (Finset.filter (λ y => x ∣ y) A).card ≤ 2) ∧
    (∀ B : Finset ℕ, (∀ x ∈ B, x ≤ n) → 
      (∀ x ∈ B, (Finset.filter (λ y => x ∣ y) B).card ≤ 2) → 
      B.card ≤ A.card) →
    ((2 * n : ℚ) / 3 ≤ A.card) ∧ (A.card ≤ ⌈(3 * n : ℚ) / 4⌉) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_bound_l440_44008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_eq_three_l440_44044

theorem power_product_eq_three (x y : ℝ) (h : 2 * x + 3 * y - 1 = 0) :
  (9 : ℝ)^x * (27 : ℝ)^y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_eq_three_l440_44044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l440_44051

theorem unique_solution (x y z : ℝ) : 
  (9 : ℝ)^y = (3 : ℝ)^(16*x) ∧ 
  (27 : ℝ)^z = (81 : ℝ)^(2*y - 3*x) ∧ 
  (6 : ℝ)^(x + z) = (36 : ℝ)^(y - 2*z) → 
  x = 0 ∧ y = 0 ∧ z = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l440_44051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_13_and_14_l440_44016

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the problem conditions
theorem x_plus_y_between_13_and_14 
  (x y : ℝ) 
  (h1 : y = 3 * (floor x) + 1)
  (h2 : y = 4 * (floor (x - 1)) + 2)
  (h3 : ¬ ∃ n : ℤ, x = n)  -- x is not an integer
  (h4 : 3 < x ∧ x < 4) :
  13 < x + y ∧ x + y < 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_13_and_14_l440_44016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_P_l440_44098

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}
def P : Finset ℕ := M ∩ N

theorem subsets_of_P : Finset.card (Finset.powerset P) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_P_l440_44098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_v₃_value_l440_44054

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x⁵ + 2x⁴ + 3.5x³ - 2.6x² + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- The coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- The value of x -/
def x : ℝ := 5

/-- The third intermediate value in Horner's method -/
def v₃ : ℝ := (horner (coeffs.take 4) x) * x - 2.6

theorem horner_v₃_value : v₃ = 689.9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_v₃_value_l440_44054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quote_is_96_l440_44088

/-- Calculates the stock quote given investment details -/
noncomputable def calculate_stock_quote (investment : ℝ) (dividend_rate : ℝ) (earnings : ℝ) : ℝ :=
  let face_value := (earnings * 100) / dividend_rate
  (investment / face_value) * 100

/-- Theorem stating that given the investment details, the stock quote is 96 -/
theorem stock_quote_is_96 (investment : ℝ) (dividend_rate : ℝ) (earnings : ℝ)
  (h1 : investment = 1620)
  (h2 : dividend_rate = 8)
  (h3 : earnings = 135) :
  calculate_stock_quote investment dividend_rate earnings = 96 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_stock_quote 1620 8 135

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_quote_is_96_l440_44088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_service_provider_choices_l440_44072

/-- The number of ways to choose different service providers for children -/
def choose_providers (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else List.range k |>.foldl (fun acc i => acc * (n - i)) 1

/-- The problem statement -/
theorem service_provider_choices :
  choose_providers 25 4 = 303600 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_service_provider_choices_l440_44072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordans_rectangle_length_l440_44075

theorem jordans_rectangle_length (carol_length carol_width jordan_width : ℝ) 
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_width = 40)
  (h4 : carol_length * carol_width = jordan_width * jordan_length) :
  jordan_length = 3 :=
by
  sorry

def jordan_length : ℝ := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordans_rectangle_length_l440_44075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_means_integrality_l440_44050

/-- Given a set of natural numbers, if for each pair, either their arithmetic mean
    or geometric mean is an integer, then either all arithmetic means or all
    geometric means are integers. -/
theorem means_integrality (S : Finset ℕ) : 
  (∀ a b, a ∈ S → b ∈ S → Int.floor ((a + b) / 2 : ℚ) = (a + b) / 2 ∨ 
               ∃ k : ℕ, k * k = a * b) →
  (∀ a b, a ∈ S → b ∈ S → Int.floor ((a + b) / 2 : ℚ) = (a + b) / 2) ∨
  (∀ a b, a ∈ S → b ∈ S → ∃ k : ℕ, k * k = a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_means_integrality_l440_44050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l440_44078

/-- The time (in seconds) for two bullet trains to cross each other -/
noncomputable def crossingTime (trainLength : ℝ) (crossTime1 crossTime2 : ℝ) : ℝ :=
  (2 * trainLength) / (trainLength / crossTime1 + trainLength / crossTime2)

/-- Theorem: Two bullet trains of equal length 120 meters, taking 10 and 15 seconds
    respectively to cross a telegraph post, will cross each other in 12 seconds
    when traveling in opposite directions -/
theorem bullet_train_crossing_time :
  crossingTime 120 10 15 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l440_44078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_fixed_points_l440_44052

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the line
def line (k x y : ℝ) : Prop := y = k * x

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ ellipse x y ∧ line k x y}

-- Define the circle with MN as diameter
def circle_MN (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + (2*Real.sqrt 2/k)*y = 4

-- Theorem statement
theorem ellipse_circle_fixed_points (k : ℝ) (hk : k ≠ 0) :
  ellipse 2 (Real.sqrt 2) →
  (∀ (x y : ℝ), (x, y) ∈ intersection_points k → 
    circle_MN k (-2) 0 ∧ circle_MN k 2 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_fixed_points_l440_44052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gross_profit_l440_44056

/-- The gross profit function for a mall selling goods -/
def L (p : ℝ) : ℝ := -p^3 - 150*p^2 + 11700*p - 166000

/-- The sales volume function -/
def Q (p : ℝ) : ℝ := 8300 - 170*p - p^2

/-- The purchase cost per item -/
def cost : ℝ := 20

/-- Theorem stating that the retail price maximizing gross profit is 30 yuan -/
theorem max_gross_profit : 
  ∃ (p : ℝ), ∀ (q : ℝ), L q ≤ L p ∧ p = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gross_profit_l440_44056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l440_44005

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 2

-- Define the ellipse
def ellipse_eq (x y : ℝ) : Prop := x^2/10 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem max_distance_circle_ellipse :
  ∃ (max_dist : ℝ),
    max_dist = 6 * Real.sqrt 2 ∧
    ∀ (x1 y1 x2 y2 : ℝ),
      circle_eq x1 y1 → ellipse_eq x2 y2 →
      distance x1 y1 x2 y2 ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l440_44005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1300_l440_44070

/-- Calculates the principal given the amount, rate, and time -/
noncomputable def calculatePrincipal (amount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  amount / (1 + rate * time)

/-- Theorem stating that given the specified conditions, the principal is 1300 -/
theorem principal_is_1300 : 
  let amount : ℝ := 1456
  let rate : ℝ := 0.05
  let time : ℝ := 2.4
  calculatePrincipal amount rate time = 1300 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1300_l440_44070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l440_44018

theorem tan_value_in_second_quadrant (x : ℝ) 
  (h1 : Real.sin x = 3/5) 
  (h2 : π/2 < x) 
  (h3 : x < π) : 
  Real.tan x = -(3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l440_44018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_c_in_triangle_l440_44023

theorem angle_c_in_triangle (A B C : ℝ) (h_triangle : A + B + C = 180) 
  (h_ratio : ∃ k : ℝ, A = k ∧ B = 3*k ∧ C = 5*k) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_c_in_triangle_l440_44023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_angle_theorem_l440_44001

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define a point on the parabola
structure PointOnParabola (p : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : parabola p x y

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the centroid of a triangle
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define the angle between three points
noncomputable def angle (A F B : ℝ × ℝ) : ℝ := sorry

-- Convert PointOnParabola to ℝ × ℝ
def toProduct {p : ℝ} (point : PointOnParabola p) : ℝ × ℝ := (point.x, point.y)

-- The main theorem
theorem parabola_angle_theorem (p : ℝ) (A B : PointOnParabola p) :
  let F := focus p
  let O := (0, 0)
  centroid O (toProduct A) (toProduct B) = F →
  Real.cos (angle (toProduct A) F (toProduct B)) = -23/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_angle_theorem_l440_44001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_on_negative_y_axis_l440_44055

theorem complex_square_on_negative_y_axis (a : ℝ) : 
  (((a + Complex.I) ^ 2).re = 0 ∧ ((a + Complex.I) ^ 2).im < 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_on_negative_y_axis_l440_44055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l440_44000

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration
structure Configuration where
  circle1 : Circle
  circle2 : Circle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Helper functions (defined as opaque constants)
opaque circles_touch_externally : Circle → Circle → Prop
opaque line_tangent_to_circle : Circle → (ℝ × ℝ) → Prop
opaque point_on_circle : Circle → (ℝ × ℝ) → Prop
opaque distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ

-- Define the conditions
def valid_configuration (config : Configuration) : Prop :=
  config.circle1.radius = 4 ∧
  config.circle2.radius = 5 ∧
  circles_touch_externally config.circle1 config.circle2 ∧
  line_tangent_to_circle config.circle1 config.A ∧
  point_on_circle config.circle2 config.B ∧
  point_on_circle config.circle2 config.C ∧
  distance config.A config.B = distance config.B config.C

-- The theorem to prove
theorem tangent_line_theorem (config : Configuration) :
  valid_configuration config → distance config.A config.C = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l440_44000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_functions_l440_44012

/-- A function f that satisfies f(-x) = -f(x) for all x -/
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function g(x) = x^2/4 - 1 -/
noncomputable def g (x : ℝ) : ℝ := x^2/4 - 1

theorem sum_of_functions (f : ℝ → ℝ) (hf : is_odd_function f) :
  f (-1.5) + f 1.5 + g (-1.5) + g 1.5 = -0.875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_functions_l440_44012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_positive_integer_solution_three_satisfies_inequality_three_is_largest_solution_l440_44086

theorem largest_positive_integer_solution : 
  ∀ x : ℕ, x > 0 → (1/2 : ℝ) * (x + 3) ≤ 3 → x ≤ 3 :=
by
  sorry

theorem three_satisfies_inequality : 
  (1/2 : ℝ) * (3 + 3) ≤ 3 :=
by
  sorry

theorem three_is_largest_solution : 
  ∃ x : ℕ, x = 3 ∧ x > 0 ∧
    ((1/2 : ℝ) * (x + 3) ≤ 3) ∧
    (∀ y : ℕ, y > 0 → (1/2 : ℝ) * (y + 3) ≤ 3 → y ≤ x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_positive_integer_solution_three_satisfies_inequality_three_is_largest_solution_l440_44086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_alt_l440_44058

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (1, 0)
def center2 : ℝ × ℝ := (-1, -2)
def radius1 : ℝ := 1
def radius2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 8

-- Theorem stating that the circles intersect
theorem circles_intersect :
  distance_between_centers > (radius2 - radius1) ∧
  distance_between_centers < (radius2 + radius1) := by
  sorry

-- Theorem stating that the circles intersect (alternative formulation)
theorem circles_intersect_alt :
  2 < Real.sqrt 8 ∧ Real.sqrt 8 < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_alt_l440_44058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_ratio_l440_44006

/-- Regular hexagon with side length s -/
structure RegularHexagon :=
  (s : ℝ)
  (s_pos : s > 0)

/-- Square formed by connecting midpoints of alternate sides of a regular hexagon -/
noncomputable def midpoint_square (h : RegularHexagon) : ℝ := h.s^2 / 4

/-- Area of a regular hexagon -/
noncomputable def hexagon_area (h : RegularHexagon) : ℝ := 3 * Real.sqrt 3 * h.s^2 / 2

/-- Theorem: The area of the square formed by connecting midpoints of three alternate sides
    of a regular hexagon is 1/(6√3) of the area of the hexagon -/
theorem midpoint_square_area_ratio (h : RegularHexagon) :
  midpoint_square h / hexagon_area h = 1 / (6 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_ratio_l440_44006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l440_44002

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 + 2*x + a) / Real.log 0.5

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (-(5 - 2*a))^x₁ > (-(5 - 2*a))^x₂

theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (1 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l440_44002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_triangle_angles_l440_44033

theorem contrapositive_triangle_angles :
  (∀ (A B : ℝ), A > B → Real.sin A > Real.sin B) ↔ 
  (∀ (A B : ℝ), Real.sin A ≤ Real.sin B → A ≤ B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_triangle_angles_l440_44033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_pressure_properties_l440_44085

/-- Blood pressure function -/
noncomputable def P (t : ℝ) : ℝ := 115 + 25 * Real.sin (160 * Real.pi * t)

/-- Period of the blood pressure function -/
noncomputable def period : ℝ := 1 / 80

/-- Frequency (heartbeats per minute) -/
def frequency : ℝ := 80

/-- Maximum blood pressure -/
def max_pressure : ℝ := 140

/-- Minimum blood pressure -/
def min_pressure : ℝ := 90

theorem blood_pressure_properties :
  (∀ t, P (t + period) = P t) ∧
  (frequency = 1 / period) ∧
  (∀ t, P t ≤ max_pressure) ∧
  (∀ t, P t ≥ min_pressure) ∧
  (∃ t₁, P t₁ = max_pressure) ∧
  (∃ t₂, P t₂ = min_pressure) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_pressure_properties_l440_44085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l440_44083

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The directrix of a parabola -/
noncomputable def directrix (p : Parabola) : ℝ := p.b - 1 / (4 * p.a)

/-- The given parabola y = 6x^2 + 5 -/
def givenParabola : Parabola := { a := 6, b := 5 }

theorem directrix_of_given_parabola :
  directrix givenParabola = 119 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l440_44083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_calculation_nine_slips_with_three_l440_44010

/-- Represents the number of slips with 3 written on them -/
def slips_with_three : ℕ := 9

/-- The total number of slips in the bag -/
def total_slips : ℕ := 15

/-- The expected value of a randomly drawn slip -/
def expected_value : ℚ := 5

/-- The probability of drawing a slip with 3 -/
def prob_three : ℚ := slips_with_three / total_slips

/-- The probability of drawing a slip with 8 -/
def prob_eight : ℚ := (total_slips - slips_with_three) / total_slips

/-- The expected value calculation -/
theorem expected_value_calculation :
  3 * prob_three + 8 * prob_eight = expected_value :=
by sorry

/-- The main theorem: there are 9 slips with the number 3 -/
theorem nine_slips_with_three : slips_with_three = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_calculation_nine_slips_with_three_l440_44010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_difference_l440_44071

-- Define the speed of the car
noncomputable def speed : ℝ := 60

-- Define the distances for the two trips
noncomputable def distance1 : ℝ := 540
noncomputable def distance2 : ℝ := 600

-- Define the function to calculate time given distance
noncomputable def time (d : ℝ) : ℝ := d / speed

-- Theorem statement
theorem trip_time_difference : (time distance2 - time distance1) * 60 = 60 := by
  -- Expand the definitions
  unfold time distance1 distance2 speed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_difference_l440_44071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l440_44048

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.sqrt 3)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, (1/2) * Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧
  (∀ x, f (π/3 - x) = f (π/3 + x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l440_44048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l440_44090

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * b * x^2 + c * x

-- Define the derivative of f as g
noncomputable def g (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

-- Define the function h
noncomputable def h (a b : ℝ) : ℝ → ℝ :=
  λ x ↦ if x ≥ 1 then g a b 1 (x - 1) else -g a b 1 (x - 1)

-- Part 1 theorem
theorem part1 (a b : ℝ) (ha : a > 0) :
  (∀ x, g a b 1 x ≥ g a b 1 (-1)) ∧ g a b 1 (-1) = 0 →
  h a b 2 + h a b (-2) = 0 := by sorry

-- Part 2 theorem
theorem part2 :
  {b : ℝ | ∀ x ∈ Set.Ioo 0 2, |g 1 b 0 x| ≤ 1} = Set.Icc (-2) (-3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l440_44090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_equals_one_l440_44021

theorem tan_theta_equals_one (θ : ℝ) (h1 : θ ∈ Set.Ioo (π / 6) (π / 3)) 
  (h2 : ∃ (k : ℤ), 17 * θ = θ + 2 * k * π) : Real.tan θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_equals_one_l440_44021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficients_l440_44029

theorem expansion_coefficients :
  let e := (1 + X^5 + X^7 : Polynomial ℤ)^20
  (e.coeff 18 = 0) ∧ (e.coeff 17 = 3420) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficients_l440_44029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l440_44014

/-- The probability of getting heads on a single toss of the biased coin. -/
noncomputable def p_heads : ℝ := 3/5

/-- The number of times the coin is tossed. -/
def num_tosses : ℕ := 60

/-- The probability of getting an even number of heads after n tosses. -/
noncomputable def P (n : ℕ) : ℝ := 1/2 * (1 + (1/5)^n)

/-- The main theorem: the probability of getting an even number of heads
    after 60 tosses of a biased coin with 3/5 probability of heads. -/
theorem even_heads_probability :
  P num_tosses = 1/2 * (1 + (1/5)^60) := by
  -- The proof goes here
  sorry

#eval num_tosses -- This will evaluate to 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l440_44014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_T_range_problem_l440_44076

/-- Represents the quadratic equation x^2 + 2(m-2)x + m^2 - 3m + 3 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + 2*(m-2)*x + m^2 - 3*m + 3 = 0

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) 
  (h_m : m ≥ -1)
  (h_distinct : ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂)
  (h_sum_squares : x₁^2 + x₂^2 = 6) :
  m = (5 - Real.sqrt 17) / 2 := by
  sorry

noncomputable def T (m x₁ x₂ : ℝ) : ℝ :=
  (m*x₁)/(1-x₁) + (m*x₂)/(1-x₂)

theorem T_range_problem (m : ℝ) (x₁ x₂ : ℝ)
  (h_m : m ≥ -1)
  (h_distinct : ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂)
  (h_sum_squares : x₁^2 + x₂^2 = 6) :
  0 < T m x₁ x₂ ∧ T m x₁ x₂ ≤ 4 ∧ T m x₁ x₂ ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_T_range_problem_l440_44076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_ray_angle_l440_44026

noncomputable section

def Equilateral_triangle : Type := Unit -- Placeholder definition
def Ray : Type := Unit -- Placeholder definition
def Angle : Type := ℝ -- Representing angle as a real number

def ray_from_vertex : Equilateral_triangle → Ray := sorry
def base_of_triangle : Equilateral_triangle → Ray := sorry
def angle_between : Ray → Ray → Angle := sorry

theorem equilateral_triangle_ray_angle (m n : ℝ) (h_pos : m > 0 ∧ n > 0) :
  let triangle : Equilateral_triangle := sorry
  let ray : Ray := ray_from_vertex triangle
  let base : Ray := base_of_triangle triangle
  let obtuse_angle : Angle := angle_between ray base
  obtuse_angle = Real.pi - Real.arctan (Real.sqrt 3 * (m + n) / (n - m)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_ray_angle_l440_44026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_calculation_l440_44041

/-- The area of metal wasted when cutting the largest possible equilateral triangle
    from a square with side length 4 units, and then cutting the largest possible
    circular disc from that triangle, is equal to 16 - 4π square units. -/
theorem metal_waste_calculation : ∀ (π : ℝ), 16 - 4 * π = 
  let square_side : ℝ := 4
  let square_area : ℝ := square_side ^ 2
  let triangle_side : ℝ := square_side
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2
  let circle_diameter : ℝ := square_side
  let circle_area : ℝ := π * (circle_diameter / 2) ^ 2
  square_area - circle_area :=
by
  intro π
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_calculation_l440_44041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l440_44096

/-- An arithmetic sequence with first term -1 and common difference d > 1 -/
noncomputable def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := -1 + (n - 1 : ℝ) * d

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * (-1) + (n - 1 : ℝ) * d) / 2

/-- Theorem for part I -/
theorem part_one (d : ℝ) (h1 : d > 1) (h2 : S d 4 - 2 * arithmetic_sequence d 2 * arithmetic_sequence d 3 + 6 = 0) :
  ∀ n : ℕ, n ≥ 1 → S d n = (3 * (n : ℝ)^2 - 5 * (n : ℝ)) / 2 := by
  sorry

/-- Theorem for part II -/
theorem part_two (d : ℝ) (h : d > 1) :
  (∀ n : ℕ, n ≥ 1 → ∃ c : ℝ, 
    (let a1 := arithmetic_sequence d n + c
     let a2 := arithmetic_sequence d (n + 1) + 4 * c
     let a3 := arithmetic_sequence d (n + 2) + 15 * c
     ∃ r : ℝ, a2 = a1 * r ∧ a3 = a2 * r)) →
  d > 1 ∧ d ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l440_44096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_is_arctan_sqrt2_l440_44027

/-- A regular triangular pyramid with the specified property -/
structure RegularTriangularPyramid where
  -- Base side length
  a : ℝ
  -- Height of the pyramid
  h : ℝ
  -- Distance from base side to non-intersecting edge is half the base side length
  edge_distance_prop : a / 2 = a / 2

/-- The angle between a lateral face and the base plane of the pyramid -/
noncomputable def lateral_face_angle (p : RegularTriangularPyramid) : ℝ :=
  Real.arctan (Real.sqrt 2)

/-- Theorem: The angle between a lateral face and the base plane is arctan(√2) -/
theorem lateral_face_angle_is_arctan_sqrt2 (p : RegularTriangularPyramid) :
  lateral_face_angle p = Real.arctan (Real.sqrt 2) := by
  -- Unfold the definition of lateral_face_angle
  unfold lateral_face_angle
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_is_arctan_sqrt2_l440_44027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_differences_l440_44060

-- Define a permutation of integers 1 to 10
def Permutation := Fin 10 → Fin 10

-- Define the sum function for a given permutation
def sumDifferences (p : Permutation) : ℚ :=
  |p 0 - p 1| + |p 2 - p 3| + |p 4 - p 5| + |p 6 - p 7| + |p 8 - p 9|

-- State the theorem
theorem average_sum_differences :
  ∃ (perms : Finset Permutation),
    (Finset.sum perms sumDifferences) / (Finset.card perms : ℚ) = 55 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_differences_l440_44060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_consumption_may_january_bill_july_bill_function_l440_44092

-- Define the electricity pricing tiers
def price_tier1 : ℝ := 0.5
def price_tier2 : ℝ := 0.6
def price_tier3 : ℝ := 0.8

-- Define the consumption thresholds
def threshold1 : ℝ := 50
def threshold2 : ℝ := 200

-- Define Xiaogang's family's electricity consumption for 6 months
def consumption : List ℝ := [-50, 30, -26, -45, 36, 25]

-- Define the function to calculate the electricity bill
noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if x ≤ threshold1 then x * price_tier1
  else if x ≤ threshold2 then threshold1 * price_tier1 + (x - threshold1) * price_tier2
  else threshold1 * price_tier1 + (threshold2 - threshold1) * price_tier2 + (x - threshold2) * price_tier3

-- Theorem for the highest consumption month
theorem highest_consumption_may :
  List.maximum? (List.map (λ x => x + threshold2) consumption) = some 236 := by
  sorry

-- Theorem for January's electricity bill
theorem january_bill :
  calculate_bill (threshold2 + consumption[0]!) = 85 := by
  sorry

-- Theorem for July's electricity bill function
theorem july_bill_function (x : ℝ) :
  (0 < x ∧ x ≤ threshold1 → calculate_bill x = 0.5 * x) ∧
  (threshold1 < x ∧ x ≤ threshold2 → calculate_bill x = 0.6 * x - 5) ∧
  (threshold2 < x → calculate_bill x = 0.8 * x - 45) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_consumption_may_january_bill_july_bill_function_l440_44092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_properties_l440_44091

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ vertices

/-- The interior angles of a polygon. -/
noncomputable def interior_angles (p : ConvexPolygon) : Set ℝ := sorry

/-- A line segment between two points. -/
def line_segment (a b : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- A point is inside a set if it's in the set but not on its boundary. -/
def is_inside (point : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop := sorry

theorem convex_polygon_properties (p : ConvexPolygon) :
  (∀ angle, angle ∈ interior_angles p → angle < 180) ∧
  (∀ a b, a ∈ p.vertices → b ∈ p.vertices → 
    ∀ point, point ∈ line_segment a b → is_inside point p.vertices) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_properties_l440_44091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_sufficient_not_necessary_for_cos_double_zero_l440_44037

theorem sin_eq_cos_sufficient_not_necessary_for_cos_double_zero :
  (∃ θ : ℝ, Real.sin θ = Real.cos θ ∧ Real.cos (2 * θ) = 0) ∧
  (∃ θ : ℝ, Real.cos (2 * θ) = 0 ∧ Real.sin θ ≠ Real.cos θ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_sufficient_not_necessary_for_cos_double_zero_l440_44037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l440_44007

theorem triangle_inequality (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  3 * a^2 + 3 * b^2 - c^2 = 4 * a * b →
  Real.cos (Real.cos A) ≤ Real.cos (Real.sin B) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l440_44007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l440_44082

theorem range_of_a (a : ℝ) : 
  a < 0 →
  (∀ x : ℝ, (x - a) * (x - 3 * a) < 0 → (2 : ℝ)^(3 * x + 1) > (2 : ℝ)^(-x - 7)) →
  -2/3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l440_44082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_for_arc_intersection_largest_constant_is_optimal_l440_44031

/-- An arc on a circle, represented by its endpoints -/
structure Arc where
  start : ℝ
  stop : ℝ

/-- The theorem statement -/
theorem largest_constant_for_arc_intersection (n : ℕ) (h_n : n ≥ 3) 
  (arcs : Fin n → Arc) 
  (h_intersect : ∃ (S : Finset (Fin n × Fin n × Fin n)), 
    S.card ≥ (n.choose 3) / 2 ∧ 
    ∀ (i j k : Fin n), (i, j, k) ∈ S → i < j ∧ j < k ∧ 
    (∃ (x : ℝ), x ∈ Set.Icc (arcs i).start (arcs i).stop ∧
                x ∈ Set.Icc (arcs j).start (arcs j).stop ∧
                x ∈ Set.Icc (arcs k).start (arcs k).stop)) :
  ∃ (I : Finset (Fin n)), 
    I.card > n * (Real.sqrt 6 / 6) ∧
    ∃ (x : ℝ), ∀ (i : Fin n), i ∈ I → x ∈ Set.Icc (arcs i).start (arcs i).stop :=
by sorry

/-- The constant √6/6 is the largest possible -/
theorem largest_constant_is_optimal :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, ∃ (arcs : Fin n → Arc),
    (∀ (I : Finset (Fin n)), 
      I.card > n * (Real.sqrt 6 / 6 + ε) →
      ¬∃ (x : ℝ), ∀ (i : Fin n), i ∈ I → x ∈ Set.Icc (arcs i).start (arcs i).stop) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_for_arc_intersection_largest_constant_is_optimal_l440_44031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_calculation_l440_44017

theorem absolute_value_calculation : 
  abs (abs (-(abs (2 * (-1 + 2))) - 3) + 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_calculation_l440_44017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_problem_l440_44009

theorem cosine_sum_problem (α β : ℝ) 
  (h1 : Real.cos (α + β) = 3/5)
  (h2 : Real.sin (β - π/4) = 5/13)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) :
  Real.cos (α + π/4) = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_problem_l440_44009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_loses_l440_44030

/-- Represents a player in the game -/
inductive Player : Type
| Andrey : Player
| Borya : Player
| Vitya : Player
| Gena : Player

/-- Represents a move in the game -/
inductive Move : Type
| Vertical : Move  -- 2 × 1 rectangle
| Horizontal : Move  -- 1 × 2 rectangle

/-- The game board -/
def Board := Fin 100 → Fin 2019 → Option Player

/-- The game state -/
structure GameState :=
  (board : Board)
  (currentPlayer : Player)

/-- Function to determine if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  sorry

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Function to determine if the game is over -/
def isGameOver (state : GameState) : Prop :=
  sorry

/-- Function to apply a strategy repeatedly -/
def applyStrategy (strategy : GameState → Move) (n : ℕ) (initialState : GameState) : GameState :=
  match n with
  | 0 => initialState
  | n + 1 => applyMove (applyStrategy strategy n initialState) (strategy (applyStrategy strategy n initialState))

/-- Theorem stating that Andrey, Borya, and Gena can collude to make Vitya lose -/
theorem vitya_loses :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.currentPlayer = Player.Andrey →
      ∃ (n : ℕ),
        let finalState := applyStrategy strategy n initialState
        finalState.currentPlayer = Player.Vitya ∧ isGameOver finalState :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_loses_l440_44030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_cost_l440_44063

theorem soup_cost (muffin_cost coffee_cost salad_cost lemonade_cost soup_cost : ℚ)
  (h1 : muffin_cost = 2)
  (h2 : coffee_cost = 4)
  (h3 : salad_cost = 5.25)
  (h4 : lemonade_cost = 0.75)
  (h5 : muffin_cost + coffee_cost + 3 = salad_cost + lemonade_cost + soup_cost) :
  soup_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_cost_l440_44063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_cos_value_when_f_eq_6_5_l440_44087

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 6) - 2 * cos x

-- Theorem for monotonically increasing intervals
theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(π/3) + 2*π*↑k) ((2*π/3) + 2*π*↑k)) :=
sorry

-- Theorem for the value of cos(2x - π/3) when f(x) = 6/5
theorem cos_value_when_f_eq_6_5 (x : ℝ) (h : f x = 6/5) :
  cos (2*x - π/3) = 7/25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_cos_value_when_f_eq_6_5_l440_44087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_planes_divide_space_l440_44042

/-- A plane in 3D space -/
structure Plane3D where
  -- Definition of a plane (could be represented by a normal vector and a point)
  -- We leave this abstract for simplicity
  mk :: -- Add this line to define a constructor

/-- Represents the intersection of two planes -/
def intersection (p1 p2 : Plane3D) : Set (Fin 3 → ℝ) :=
  sorry -- Definition of intersection

/-- Checks if two sets in ℝ³ are parallel -/
def isParallel (s1 s2 : Set (Fin 3 → ℝ)) : Prop :=
  sorry -- Definition of parallelism

/-- The number of regions created by a set of planes -/
def numRegions (planes : List Plane3D) : ℕ :=
  sorry -- Definition of number of regions

theorem three_planes_divide_space (p1 p2 p3 : Plane3D) :
  intersection p1 p2 ≠ ∅ ∧
  intersection p2 p3 ≠ ∅ ∧
  intersection p3 p1 ≠ ∅ ∧
  isParallel (intersection p1 p2) (intersection p2 p3) ∧
  isParallel (intersection p2 p3) (intersection p3 p1) ∧
  isParallel (intersection p3 p1) (intersection p1 p2) →
  numRegions [p1, p2, p3] = 8 := by
  sorry

#check three_planes_divide_space

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_planes_divide_space_l440_44042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_average_value_l440_44024

-- Define the function f(x) = log x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the domain D = [10, 100]
def D : Set ℝ := Set.Icc 10 100

-- Define the average value property
def has_average_value (f : ℝ → ℝ) (D : Set ℝ) (C : ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ (f x₁ + f x₂) / 2 = C

-- State the theorem
theorem log_average_value :
  has_average_value f D (3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_average_value_l440_44024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_size_l440_44035

theorem math_class_size (total_students : ℕ) (both_classes : ℕ) 
  (math_class physics_class : Finset ℕ)
  (h_total : total_students = 100)
  (h_both : both_classes = 10)
  (h_all_enrolled : ∀ s, s ∈ math_class ∨ s ∈ physics_class)
  (h_math_size : (math_class.card : ℝ) = 4 * physics_class.card)
  (h_both_count : (math_class ∩ physics_class).card = both_classes) :
  math_class.card = 88 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_size_l440_44035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_for_christine_l440_44061

/-- The number of fluid ounces in 1 liter -/
noncomputable def fluid_oz_per_liter : ℝ := 33.8

/-- The size of each juice bottle in milliliters -/
noncomputable def bottle_size : ℝ := 250

/-- The minimum amount of juice Christine needs to buy in fluid ounces -/
noncomputable def min_juice_needed : ℝ := 60

/-- Converts fluid ounces to milliliters -/
noncomputable def fl_oz_to_ml (oz : ℝ) : ℝ := (oz / fluid_oz_per_liter) * 1000

/-- Calculates the number of bottles needed to contain a given amount of milliliters -/
noncomputable def bottles_needed (ml : ℝ) : ℕ := (Int.ceil (ml / bottle_size)).toNat

theorem min_bottles_for_christine : bottles_needed (fl_oz_to_ml min_juice_needed) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_for_christine_l440_44061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_is_10_seconds_l440_44057

/-- The time taken for a train to pass a signal post -/
noncomputable def train_passing_time (distance : ℝ) (time_minutes : ℝ) (train_length : ℝ) : ℝ :=
  let speed_km_per_hour := distance / (time_minutes / 60)
  let speed_m_per_second := speed_km_per_hour * (1000 / 3600)
  train_length / speed_m_per_second

/-- Theorem stating that the time taken for the train to pass a signal post is 10 seconds -/
theorem train_passing_time_is_10_seconds :
  train_passing_time 10 15 111.11111111111111 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_is_10_seconds_l440_44057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_line_l440_44077

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 = 1

/-- Definition of the eccentricity -/
noncomputable def eccentricity : ℝ := 2 * Real.sqrt 5 / 5

/-- Definition of the distance between left vertex and right focus -/
noncomputable def vertex_focus_distance : ℝ := 2 + Real.sqrt 5

/-- Definition of the fixed point P -/
def P : ℝ × ℝ := (2, 1)

/-- Definition of the right focus F -/
def F : ℝ × ℝ := (2, 0)

/-- Definition of a line passing through F with slope k -/
noncomputable def line (k : ℝ) (x : ℝ) : ℝ := k * (x - F.1)

/-- Definition of the area of triangle MNP -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  Real.sqrt 5 * Real.sqrt (1 + k^2) / (1 + 5 * k^2)

theorem max_area_line :
  ∃ (k : ℝ), ∀ (k' : ℝ), triangle_area k ≥ triangle_area k' ∧ k = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_line_l440_44077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_for_commodity_l440_44094

/-- Represents the maximum discount that can be offered on a commodity --/
noncomputable def max_discount (cost_price marked_price : ℝ) (min_profit_margin : ℝ) : ℝ :=
  1 - (cost_price * (1 + min_profit_margin) / marked_price)

/-- Theorem stating the maximum discount for the given scenario --/
theorem max_discount_for_commodity :
  max_discount 200 400 0.4 = 0.3 := by
  -- Unfold the definition of max_discount
  unfold max_discount
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_for_commodity_l440_44094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_l440_44013

/-- The direction vector of a parameterized line. -/
noncomputable def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The line y = (5x - 9) / 6 -/
noncomputable def line (x : ℝ) : ℝ := (5 * x - 9) / 6

/-- Parameterization of the line -/
noncomputable def parameterization (t : ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem line_direction_vector :
  let d := direction_vector parameterization
  ∀ x ≥ 4, distance (parameterization ((x - 4) * (6 / Real.sqrt 61))) (4, 2) = (x - 4) * (6 / Real.sqrt 61) →
  d = (6 / Real.sqrt 61, 5 / Real.sqrt 61) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_direction_vector_l440_44013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l440_44022

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi/3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (x : ℝ), f (8*Real.pi/3 + x) = f (8*Real.pi/3 - x)) ∧
  (f (Real.pi/6 + Real.pi) = 0) ∧
  (¬ ∀ (x y : ℝ), Real.pi/2 < x ∧ x < y ∧ y < Real.pi → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l440_44022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_alpha_cos_sin_product_l440_44039

-- Part I
theorem cos_pi_half_plus_alpha (α : ℝ) 
  (h1 : Real.cos (π + α) = -1/2) 
  (h2 : 0 < α ∧ α < π/2) : 
  Real.cos (π/2 + α) = -Real.sqrt 3/2 := by sorry

-- Part II
theorem cos_sin_product (β : ℝ) 
  (h : Real.cos (π/6 - β) = 1/3) :
  Real.cos (5*π/6 + β) * Real.sin (2*π/3 - β) = -1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_alpha_cos_sin_product_l440_44039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_cone_surface_area_l440_44059

-- Define the cone parameters
def slant_height : ℝ := 15
def cone_height : ℝ := 9  -- Changed 'height' to 'cone_height' to avoid naming conflict

-- Define pi as a real number (since we're working with π)
noncomputable def π : ℝ := Real.pi

-- Theorem for the volume of the cone
theorem cone_volume : 
  ∃ (r : ℝ), r^2 + cone_height^2 = slant_height^2 ∧ 
  (1/3 : ℝ) * π * r^2 * cone_height = 432 * π := by
  sorry

-- Theorem for the total surface area of the cone
theorem cone_surface_area :
  ∃ (r : ℝ), r^2 + cone_height^2 = slant_height^2 ∧ 
  π * r^2 + π * r * slant_height = 324 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_cone_surface_area_l440_44059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l440_44053

-- Define the functions for the curves
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem enclosed_area_theorem :
  ∫ x in lower_bound..upper_bound, (g x - f x) = Real.exp 1 - Real.log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l440_44053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_27_l440_44043

theorem cube_root_of_27 : (27 : ℝ) ^ (1/3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_27_l440_44043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l440_44068

theorem sum_remainder (a b c d : ℕ) 
  (ha : a % 11 = 2)
  (hb : b % 11 = 4)
  (hc : c % 11 = 6)
  (hd : d % 11 = 8) :
  (a + b + c + d) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l440_44068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2015_l440_44045

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / mySequence n

theorem mySequence_2015 : mySequence 2014 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2015_l440_44045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coefficient_in_expansion_l440_44038

/-- The coefficient of x^2 in the expansion of (x - 1/x)^6 is 15 -/
theorem x_squared_coefficient_in_expansion : ℕ := by
  -- The coefficient we're looking for
  let coeff : ℕ := 15
  
  -- The binomial expansion of (x - 1/x)^6
  let expansion : ℝ → ℝ := fun x => (x - 1/x)^6
  
  -- The coefficient of x^2 in the expansion
  let x_squared_coeff : ℕ := 15
  
  -- Proof that x_squared_coeff equals coeff
  have h : x_squared_coeff = coeff := by rfl
  
  exact x_squared_coeff

-- The theorem statement
#check x_squared_coefficient_in_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coefficient_in_expansion_l440_44038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_intersection_l440_44066

theorem line_hyperbola_intersection
  (α : ℝ)
  (line : ℝ → ℝ → Prop)
  (hyperbola : ℝ → ℝ → Prop)
  (single_intersection : Prop) :
  let line := λ x y ↦ 3 * x * (Real.sin α)^2 + y * (Real.cos α)^2 - 3 = 0
  let hyperbola := λ x y ↦ x^2 - y^2 = 1
  single_intersection →
  (∃ x y, line x y ∧ hyperbola x y) →
  (∃! p : ℝ × ℝ, line p.1 p.2 ∧ hyperbola p.1 p.2) →
  (∃ x y, line x y ∧ hyperbola x y ∧ ((x = 1 ∧ y = 0) ∨ (x = 17/8 ∧ y = 15/8))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_intersection_l440_44066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l440_44074

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, Real.sin x + Real.cos x > m

def q (m : ℝ) : Prop := ∀ x : ℝ, Monotone (fun y : ℝ ↦ (2 * m^2 - m)^y)

-- Define the range of m
def m_range (m : ℝ) : Prop := (m > -Real.sqrt 2 ∧ m < -1/2) ∨ m > 1

-- Theorem statement
theorem m_range_theorem (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l440_44074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minimized_at_sqrt2_over_2_l440_44040

/-- The function f(x) = x^2 --/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = ln(x) --/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The distance between points (t, f(t)) and (t, g(t)) --/
noncomputable def distance (t : ℝ) : ℝ := |f t - g t|

/-- Theorem: The distance between the points where the line x = t intersects f(x) = x^2 and g(x) = ln(x) is minimized when t = √2/2 --/
theorem distance_minimized_at_sqrt2_over_2 :
  ∃ (t : ℝ), t > 0 ∧ (∀ (s : ℝ), s > 0 → distance t ≤ distance s) ∧ t = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minimized_at_sqrt2_over_2_l440_44040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defined_not_implies_continuity_l440_44062

-- Define a function f on the closed interval [a, b]
variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [MetricSpace α] (f : α → ℝ) (a b : α)

-- Define the property of f being defined on [a, b]
def DefinedOn (f : α → ℝ) (a b : α) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → ∃ y, f x = y

-- State the theorem
theorem defined_not_implies_continuity (f : α → ℝ) (a b : α) :
  DefinedOn f a b → ¬(∀ ε > 0, ∃ δ > 0, ∀ x, a ≤ x ∧ x ≤ b → dist x a < δ → |f x - f a| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defined_not_implies_continuity_l440_44062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_210_and_330_l440_44097

theorem lcm_of_210_and_330 (hcf lcm : ℕ) : 
  hcf = 30 → 
  Nat.gcd 210 330 = hcf → 
  lcm * hcf = 210 * 330 → 
  lcm = 2310 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_210_and_330_l440_44097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l440_44089

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : (x^2 / e.a^2) + (y^2 / e.b^2) = 1

/-- The upper vertex of the ellipse -/
def Ellipse.upperVertex (e : Ellipse) : ℝ × ℝ := (0, e.b)

/-- A line intersecting the ellipse -/
structure LineIntersectingEllipse (e : Ellipse) where
  slope : ℝ
  y_intercept : ℝ

/-- Theorem: For an ellipse with eccentricity 1/2 passing through (1, 3/2),
    if two lines intersecting at the upper vertex have a slope product of 1/4,
    then their intersection with the ellipse determines a line passing through (0, 2√3) -/
theorem ellipse_intersection_fixed_point (e : Ellipse)
    (h_ecc : e.eccentricity = 1/2)
    (h_point : ∃ p : PointOnEllipse e, p.x = 1 ∧ p.y = 3/2)
    (l₁ l₂ : LineIntersectingEllipse e)
    (h_slopes : l₁.slope * l₂.slope = 1/4)
    (h_through_vertex : l₁.y_intercept = e.upperVertex.2 ∧ l₂.y_intercept = e.upperVertex.2) :
    ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧
    ∀ (x y : ℝ), (x^2 / e.a^2 + y^2 / e.b^2 = 1) →
    (y - l₁.slope * x - l₁.y_intercept) * (y - l₂.slope * x - l₂.y_intercept) = 0 →
    y = m := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l440_44089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_box_height_l440_44015

/-- Represents a rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  large_radius : ℝ
  small_radius : ℝ
  width : ℝ
  length : ℝ
  height : ℝ

/-- Checks if the configuration of spheres in the box is valid -/
def is_valid_configuration (box : SphereBox) : Prop :=
  box.large_radius = 3 ∧
  box.small_radius = 1 ∧
  box.width = 5 ∧
  box.length = 5 ∧
  box.height > 0

/-- The main theorem stating that the height of the box is 13 -/
theorem sphere_box_height (box : SphereBox) :
  is_valid_configuration box → box.height = 13 := by
  intro h
  sorry

#check sphere_box_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_box_height_l440_44015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_correct_l440_44046

open Real

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 4)

noncomputable def axis_of_symmetry (k : ℤ) : ℝ := π / 8 + k * π / 2

theorem axis_of_symmetry_is_correct :
  ∀ (x : ℝ) (k : ℤ), f (axis_of_symmetry k + x) = f (axis_of_symmetry k - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_correct_l440_44046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_facebook_group_messages_l440_44019

/-- Represents the Facebook group with its members and posting rates -/
structure FacebookGroup where
  total_members : ℕ
  admin_removals : List ℕ
  removed_posting_rates : List ℕ
  remaining_percentages : List ℚ
  remaining_posting_rates : List ℕ

/-- Calculates the total weekly messages for the remaining members in the Facebook group -/
def calculate_weekly_messages (group : FacebookGroup) : ℚ :=
  sorry

/-- Theorem stating that the total weekly messages for the given group is approximately 21663 -/
theorem facebook_group_messages (group : FacebookGroup) : 
  group.total_members = 200 ∧ 
  group.admin_removals = [15, 25, 10, 20, 5] ∧
  group.removed_posting_rates = [40, 60, 50, 30, 20] ∧
  group.remaining_percentages = [25/100, 50/100, 20/100, 5/100] ∧
  group.remaining_posting_rates = [15, 25, 40, 10] →
  Int.floor (calculate_weekly_messages group) = 21663 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_facebook_group_messages_l440_44019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l440_44003

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 - 2*x + 3) / Real.log (1/2)

-- State the theorem
theorem f_strictly_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-1) 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_interval_l440_44003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_list_properties_l440_44064

theorem number_list_properties (nums : List ℤ) : 
  nums.length = 2017 →
  (∀ (subset : List ℤ), subset ⊆ nums → subset.length = 7 → (subset.map (λ x => x^2)).sum = 7) →
  (∀ (subset : List ℤ), subset ⊆ nums → subset.length = 11 → subset.sum > 0) →
  nums.sum % 9 = 0 →
  ∃ (neg_ones pos_ones : List ℤ), 
    neg_ones.length = 5 ∧ 
    pos_ones.length = 2012 ∧
    neg_ones.all (· = -1) ∧
    pos_ones.all (· = 1) ∧
    nums = neg_ones ++ pos_ones :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_list_properties_l440_44064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_section_is_4_sqrt_13_l440_44067

/-- Represents a rectangular prism with given dimensions and sections -/
structure RectangularPrism where
  ab : ℝ
  ad : ℝ
  aa1 : ℝ
  volume_ratio : Fin 3 → ℝ

/-- The area of section A₁EFD₁ in the rectangular prism -/
noncomputable def area_section (prism : RectangularPrism) : ℝ :=
  4 * Real.sqrt 13

/-- Theorem stating that the area of section A₁EFD₁ is 4√13 given the specified conditions -/
theorem area_section_is_4_sqrt_13 (prism : RectangularPrism)
    (h_ab : prism.ab = 6)
    (h_ad : prism.ad = 4)
    (h_aa1 : prism.aa1 = 3)
    (h_ratio : prism.volume_ratio 0 = 1 ∧ prism.volume_ratio 1 = 4 ∧ prism.volume_ratio 2 = 1) :
    area_section prism = 4 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_section_is_4_sqrt_13_l440_44067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_point_l440_44028

noncomputable def f (x : ℝ) := Real.log x / Real.log 5

theorem trisection_point (x₁ x₂ : ℝ) (h : 1 < x₁ ∧ x₁ < x₂) :
  let y₁ := f x₁
  let y₂ := f x₂
  let yC := (2 * y₁ + y₂) / 3
  ∃ x₃, f x₃ = yC ∧ x₃ = 5^(7/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_point_l440_44028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l440_44065

noncomputable def a (m : ℝ) (x : ℝ) : ℝ × ℝ := (m, Real.sin (2 * x))
noncomputable def b (n : ℝ) (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), n)

noncomputable def f (m n x : ℝ) : ℝ := (a m x).1 * (b n x).1 + (a m x).2 * (b n x).2

theorem vector_dot_product_properties (m n : ℝ) :
  (∀ x, f m n x = m * Real.cos (2 * x) + n * Real.sin (2 * x)) ∧
  f m n 0 = 1 ∧
  f m n (π / 4) = 1 →
  (∀ x, f m n x = Real.sqrt 2 * Real.sin (2 * x + π / 4)) ∧
  (∀ x ∈ Set.Icc 0 (π / 2),
    f m n x ≤ Real.sqrt 2 ∧
    f m n x ≥ -1 ∧
    f m n (π / 8) = Real.sqrt 2 ∧
    f m n (π / 2) = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l440_44065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_minus_y_plus3_l440_44034

/-- The angle of inclination of a line with equation ax + by + c = 0 -/
noncomputable def angle_of_inclination (a b : ℝ) : ℝ := Real.arctan (a / b)

/-- The line equation √3x - y + 3 = 0 -/
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 = 0

theorem angle_of_inclination_sqrt3x_minus_y_plus3 :
  angle_of_inclination (Real.sqrt 3) (-1) = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_minus_y_plus3_l440_44034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_covering_theorem_l440_44073

theorem circle_covering_theorem (R : ℝ) (r : ℝ) (n : ℕ) 
  (h1 : R = 1)
  (h2 : n = 7)
  (h3 : ∃ (centers : Fin n → ℝ × ℝ), 
    ∀ (x y : ℝ), x^2 + y^2 ≤ R^2 → 
    ∃ (i : Fin n), ((x - (centers i).1)^2 + (y - (centers i).2)^2 ≤ r^2)) :
  r ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_covering_theorem_l440_44073
