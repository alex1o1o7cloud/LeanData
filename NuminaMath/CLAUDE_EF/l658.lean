import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_corner_and_middle_l658_65827

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if a grid is valid according to the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ ({1, 2, 3} : Set ℕ)) ∧
  (∀ i, (Finset.univ.image (g i) : Set ℕ) = {1, 2, 3}) ∧
  (∀ j, (Finset.univ.image (λ i => g i j) : Set ℕ) = {1, 2, 3}) ∧
  g 0 0 = 2 ∧
  g 1 1 = 3 ∧
  g 2 0 + g 2 1 + g 2 2 = 7

theorem sum_of_corner_and_middle (g : Grid) (h : is_valid_grid g) :
  g 2 2 + g 1 2 = 4 := by
  sorry

#check sum_of_corner_and_middle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_corner_and_middle_l658_65827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_sum_of_squares_and_divisibility_l658_65814

theorem unique_prime_sum_of_squares_and_divisibility :
  ∃! (p : ℕ), Prime p ∧ (∃ (m n : ℕ), p = m^2 + n^2 ∧ p ∣ (m^3 + n^3 - 4)) ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_sum_of_squares_and_divisibility_l658_65814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_rational_l658_65862

-- Define the set S
def S : Set ℝ := sorry

-- Define the property that S is finite
axiom S_finite : Set.Finite S

-- Define the property that S is a subset of [0,1]
axiom S_subset : S ⊆ Set.Icc 0 1

-- Define x₀ and x₁
axiom x₀_in_S : (0 : ℝ) ∈ S
axiom x₁_in_S : (1 : ℝ) ∈ S

-- Define the property about distances
axiom distance_property : 
  ∀ (x y : ℝ), x ∈ S → y ∈ S → x ≠ y → 
    (|x - y| = 1 ∨ ∃ (a b : ℝ), a ∈ S ∧ b ∈ S ∧ (a, b) ≠ (x, y) ∧ |a - b| = |x - y|)

-- Theorem to prove
theorem all_rational : ∀ x ∈ S, ∃ (p q : ℤ), x = (p : ℝ) / q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_rational_l658_65862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l658_65803

-- Define the function f(x, m) = |mx³ - ln x|
noncomputable def f (x m : ℝ) : ℝ := |m * x^3 - Real.log x|

-- State the theorem
theorem min_m_value (m : ℝ) :
  (∀ x ∈ Set.Ioc (0 : ℝ) 1, f x m ≥ 1) ↔ m ≥ (1/3) * Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l658_65803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_equals_one_l658_65885

open Complex

theorem modulus_of_z_equals_one : 
  let z : ℂ := 2 / (1 - Complex.I * Real.sqrt 3)
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_equals_one_l658_65885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_age_in_ten_years_l658_65834

-- Define the current ages of Angela, Beth, and Claire
variable (angela_age : ℕ)
variable (beth_age : ℕ)
variable (claire_age : ℕ)

-- Condition 1: Angela is currently 3 times as old as Beth
axiom angela_three_times_beth : angela_age = 3 * beth_age

-- Condition 2: Ten years ago, the sum of their ages was twice the age of their cousin, Claire
axiom sum_ten_years_ago : (angela_age - 10) + (beth_age - 10) = 2 * (claire_age - 10)

-- Condition 3: Five years from now, the difference between Angela's age and Beth's age will be equal to Claire's age at present
axiom diff_five_years_future : (angela_age + 5) - (beth_age + 5) = claire_age

-- Theorem: Angela will be 25 years old in ten years
theorem angela_age_in_ten_years : angela_age + 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angela_age_in_ten_years_l658_65834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l658_65867

-- Define the structure for triangles
structure Triangle where
  -- We'll leave this empty for now as we don't need specific fields for this problem
  mk :: -- Empty constructor

-- Define the propositions
def proposition_A : Prop := ∀ x : ℝ, x ≠ 2 → x^2 - 3*x + 2 ≠ 0
def proposition_B : Prop := ∀ b : ℝ, b^2 = 9 → b = 3
def proposition_C : Prop := ∀ a b c : ℝ, a*c > b*c → a > b

-- Define the relationships between triangles
def Similar : Triangle → Triangle → Prop := sorry
def CorrespondingAnglesEqual : Triangle → Triangle → Prop := sorry

def proposition_D : Prop := 
  ∀ t1 t2 : Triangle, ¬(Similar t1 t2) → ¬(CorrespondingAnglesEqual t1 t2)

-- Theorem statement
theorem correct_proposition : 
  (∃ x : ℝ, x ≠ 2 ∧ x^2 - 3*x + 2 = 0) ∧ 
  (∃ b : ℝ, b^2 = 9 ∧ b ≠ 3) ∧ 
  (∃ a b c : ℝ, a*c > b*c ∧ a ≤ b) ∧ 
  proposition_D :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l658_65867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_gp_with_second_term_3_l658_65840

/-- A geometric progression is decreasing and infinite if its common ratio q is between 0 and 1 -/
def IsDecreasingInfiniteGP (q : ℝ) : Prop := 0 < q ∧ q < 1

/-- The sum of an infinite geometric progression with first term a and common ratio q -/
noncomputable def SumInfiniteGP (a q : ℝ) : ℝ := a / (1 - q)

/-- The second term of a geometric progression with first term a and common ratio q -/
def SecondTerm (a q : ℝ) : ℝ := a * q

theorem min_sum_gp_with_second_term_3 :
  ∀ a q : ℝ, IsDecreasingInfiniteGP q →
  SecondTerm a q = 3 →
  ∀ A : ℝ, A > 0 →
  A = SumInfiniteGP a q →
  A ≥ 12 := by
  sorry

#check min_sum_gp_with_second_term_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_gp_with_second_term_3_l658_65840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_l658_65890

noncomputable def arithmetic_sequence (a₁ : ℝ) : ℕ+ → ℝ := 
  λ n => a₁ + (2 * Real.pi / 3) * (↑n - 1)

def S (a₁ : ℝ) : Set ℝ := 
  {x | ∃ n : ℕ+, x = Real.cos (arithmetic_sequence a₁ n)}

theorem cosine_product (a₁ : ℝ) (a b : ℝ) (h : S a₁ = {a, b}) :
  a * b = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_l658_65890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_equals_nine_l658_65826

def r : Finset ℤ := {2, 3, 4, 5}
def b : Finset ℤ := {4, 5, 6, 7, 8}

theorem probability_sum_equals_nine :
  (Finset.card (Finset.filter (fun p => p.1 + p.2 = 9) (r.product b)) : ℚ) /
  ((Finset.card r : ℚ) * (Finset.card b : ℚ)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_equals_nine_l658_65826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_system_theorem_l658_65837

/-- A type representing coin denominations -/
inductive Denomination
  | one : Denomination
  | power (k : ℕ) : Denomination
deriving DecidableEq

/-- The value of a coin denomination for a given α -/
noncomputable def value (α : ℝ) : Denomination → ℝ
  | Denomination.one => 1
  | Denomination.power k => α ^ k

/-- A type representing a multiset of coins -/
def CoinSet := Multiset Denomination

/-- The total value of a set of coins for a given α -/
noncomputable def total_value (α : ℝ) (coins : CoinSet) : ℝ :=
  Multiset.sum (coins.map (value α))

/-- A predicate stating that a coin set is valid (no denomination used more than 6 times) -/
def valid_coin_set (coins : CoinSet) : Prop :=
  ∀ d, coins.count d ≤ 6

theorem coin_system_theorem :
  ∃ (α : ℝ), α > 2 ∧
  (∀ k : ℕ, k ≥ 1 → Irrational (α ^ k)) ∧
  (∀ n : ℕ, ∃ coins : CoinSet,
    valid_coin_set coins ∧
    total_value α coins = n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_system_theorem_l658_65837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l658_65865

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 0)^2 + (y - 3)^2 = 4

-- Define the line m
def line_m (x y : ℝ) : Prop := x + 3*y + 6 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a line passing through point A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection point N of line l and line m
noncomputable def point_N (k : ℝ) : ℝ × ℝ :=
  let x := (-3*k - 6) / (1 + 3*k)
  let y := (-5*k) / (1 + 3*k)
  (x, y)

-- Define the midpoint M of PQ (intersection of line l and circle C)
noncomputable def point_M (k : ℝ) : ℝ × ℝ :=
  let x := (-k^2 + 3*k) / (1 + k^2)
  let y := (3*k^2 + k) / (1 + k^2)
  (x, y)

-- Define the dot product of vectors AM and AN
noncomputable def dot_product (k : ℝ) : ℝ :=
  let (xm, ym) := point_M k
  let (xn, yn) := point_N k
  (xm - 1) * (xn - 1) + ym * yn

-- Theorem statement
theorem constant_dot_product :
  ∀ k : ℝ, k ≠ -1/3 → dot_product k = -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l658_65865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_range_of_a_l658_65817

/-- A function that always has two distinct fixed points for any real b -/
def has_two_distinct_fixed_points (f : ℝ → ℝ) : Prop :=
  ∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f x = x ∧ f y = y

/-- The function f(x) = ax^2 + bx - b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x - b

theorem fixed_points_range_of_a :
  (∀ a : ℝ, (∀ b : ℝ, has_two_distinct_fixed_points (f a b))) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_range_of_a_l658_65817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_one_l658_65882

-- Define the function f(x) = 2x log x + x - 1
noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x + x - 1

-- Theorem statement
theorem unique_solution_is_one :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_one_l658_65882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_thermometer_conversion_l658_65894

/-- A faulty thermometer with known calibration points and a current reading -/
structure FaultyThermometer where
  freezing_point : ℚ  -- Reading at 0°C
  boiling_point : ℚ   -- Reading at 100°C
  current_reading : ℚ

/-- Convert a faulty thermometer reading to the true temperature in Celsius -/
def true_temperature (t : FaultyThermometer) : ℚ :=
  (t.current_reading - t.freezing_point) * (100 / (t.boiling_point - t.freezing_point))

/-- Theorem stating that for the given faulty thermometer, 
    the true temperature is 200/13°C when it shows +17° -/
theorem faulty_thermometer_conversion :
  let t : FaultyThermometer := {
    freezing_point := 1,
    boiling_point := 105,
    current_reading := 17
  }
  true_temperature t = 200 / 13 := by
  -- Proof goes here
  sorry

#eval true_temperature {
  freezing_point := 1,
  boiling_point := 105,
  current_reading := 17
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_thermometer_conversion_l658_65894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l658_65813

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) (h1 : -π < φ) (h2 : φ < 0) 
  (h3 : ∀ x, f x φ = f (π/4 - x) φ) :
  ∃ (c : ℝ),
    (φ = -3*π/4) ∧ 
    (∀ k : ℤ, StrictMonoOn (fun y => f y (-3*π/4)) (Set.Icc (k*π + π/8) (k*π + 5*π/8))) ∧
    (∀ x y, 5*x - 2*y + c ≠ 0 ∨ y ≠ f x (-3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l658_65813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_rounded_l658_65860

def pages_to_print : ℕ := 350
def pages_per_minute : ℕ := 25

def time_to_print : ℚ := pages_to_print / pages_per_minute

theorem print_time_rounded : Int.floor (time_to_print + 1/2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_rounded_l658_65860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_26_equals_16_81_unique_n_for_16_81_l658_65897

def t : ℕ → ℚ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 => if (n + 2) % 2 = 0 then 1 + t ((n + 2) / 2) else (t (n + 1)) ^ 2

theorem t_26_equals_16_81 : t 26 = 16 / 81 := by sorry

theorem unique_n_for_16_81 (n : ℕ) (h : t n = 16 / 81) : n = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_26_equals_16_81_unique_n_for_16_81_l658_65897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_sum_cd_l658_65884

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := (x - 3) / (x^2 + c*x + d)

theorem asymptotes_sum_cd (c d : ℝ) :
  (∀ x ≠ 2, g c d x ≠ 0⁻¹) ∧
  (∀ x ≠ -3, g c d x ≠ 0⁻¹) ∧
  (∀ x, x ≠ 2 → x ≠ -3 → g c d x ≠ 0⁻¹) →
  c + d = -5 := by
  sorry

#check asymptotes_sum_cd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_sum_cd_l658_65884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l658_65893

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The right vertex of the hyperbola -/
def right_vertex (h : Hyperbola) : Point :=
  ⟨h.a, 0⟩

/-- The left focus of the hyperbola -/
noncomputable def left_focus (h : Hyperbola) : Point :=
  ⟨-Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Predicate to check if a triangle is acute -/
def is_acute_triangle (p q r : Point) : Prop :=
  sorry  -- Definition of acute triangle

/-- Main theorem -/
theorem hyperbola_eccentricity_range (h : Hyperbola)
    (p q : Point)
    (h_pq_on_hyperbola : (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1 ∧
                         (q.x^2 / h.a^2) - (q.y^2 / h.b^2) = 1)
    (h_pq_y_align : p.x = (left_focus h).x ∧ q.x = (left_focus h).x)
    (h_acute : is_acute_triangle p q (right_vertex h)) :
    1 < eccentricity h ∧ eccentricity h < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l658_65893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_60_degrees_max_area_angle_segment_area_l658_65841

-- Define the sector
structure Sector where
  α : Real  -- Central angle in radians
  R : Real  -- Radius
  l : Real  -- Arc length

-- Define the segment
structure Segment where
  s : Sector
  area : Real

-- Constants
noncomputable def π : Real := Real.pi

-- Theorem 1
theorem arc_length_60_degrees (s : Sector) (h1 : s.α = π/3) (h2 : s.R = 10) : 
  s.l = 10*π/3 := by sorry

-- Theorem 2
theorem max_area_angle (s : Sector) (h : s.l + 2*s.R = 20) :
  s.α = 2 → s.R = 5 ∧ s.l = 10 := by sorry

-- Theorem 3
theorem segment_area (seg : Segment) (h1 : seg.s.α = π/3) (h2 : seg.s.R = 2) :
  seg.area = 2*π/3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_60_degrees_max_area_angle_segment_area_l658_65841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_equals_negative_two_l658_65863

theorem cube_root_negative_eight_equals_negative_two :
  (∃ x : ℝ, x^3 = -8 ∧ x = -2) ∧
  (¬ (∀ x : ℝ, x^2 = 16 → x = 4 ∨ x = -4)) ∧
  (¬ (Real.sqrt ((-2)^2) = -2)) ∧
  (¬ (Real.sqrt (1/16) = 1/8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_eight_equals_negative_two_l658_65863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_preserved_parallelogram_preserved_square_not_preserved_rhombus_not_preserved_l658_65849

-- Define the oblique projection method
def oblique_projection (S : Type) : S → S := sorry

-- Define shapes
structure Triangle : Type := (vertices : Fin 3 → ℝ × ℝ)
structure Parallelogram : Type := (vertices : Fin 4 → ℝ × ℝ)
structure Square : Type := (vertices : Fin 4 → ℝ × ℝ)
structure Rhombus : Type := (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of oblique projection
axiom parallel_to_x_preserved {S : Type} (l : S) : 
  oblique_projection S l = l

axiom parallel_to_y_halved {S : Type} (l : S) : 
  ∃ (k : ℝ), k = 1/2 ∧ oblique_projection S l = sorry

axiom perpendicular_to_45deg {S : Type} (a b : S) : 
  sorry → sorry = 45

-- Theorem statements
theorem triangle_preserved :
  ∀ (t : Triangle), Triangle.mk (oblique_projection _ t.vertices) = sorry := sorry

theorem parallelogram_preserved :
  ∀ (p : Parallelogram), Parallelogram.mk (oblique_projection _ p.vertices) = sorry := sorry

theorem square_not_preserved :
  ∃ (s : Square), Square.mk (oblique_projection _ s.vertices) ≠ s := sorry

theorem rhombus_not_preserved :
  ∃ (r : Rhombus), Rhombus.mk (oblique_projection _ r.vertices) ≠ r := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_preserved_parallelogram_preserved_square_not_preserved_rhombus_not_preserved_l658_65849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l658_65848

theorem function_identity (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (g x - y) = g x + g (g y - g x) - x) : 
  ∀ x : ℝ, g x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l658_65848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sequence_ratio_l658_65899

/-- A geometric sequence with positive first term and a specific relation between terms -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  positive_start : 0 < a 1
  relation : ∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1)

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (seq : SpecialGeometricSequence) : ℝ :=
  seq.a 2 / seq.a 1

/-- Theorem: The common ratio of the special geometric sequence is 2 -/
theorem special_geometric_sequence_ratio (seq : SpecialGeometricSequence) :
  common_ratio seq = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sequence_ratio_l658_65899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_not_dividing_power_minus_p_l658_65886

theorem exists_prime_not_dividing_power_minus_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬(q ∣ (n^p : ℕ) - p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_not_dividing_power_minus_p_l658_65886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_time_approx_2_28_l658_65875

/-- The time in years for a principal to grow to a given amount at a given interest rate -/
noncomputable def time_to_grow (P A r : ℝ) : ℝ :=
  Real.log (A / P) / Real.log (1 + r)

/-- Theorem stating that the time for the given principal to grow to the given amount at the given interest rate is approximately 2.28 years -/
theorem growth_time_approx_2_28 :
  let P : ℝ := 958.9041095890411
  let A : ℝ := 1120
  let r : ℝ := 0.07
  abs (time_to_grow P A r - 2.28) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_growth_time_approx_2_28_l658_65875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_100_terms_l658_65831

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then n - 1 else n

def S (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (i + 1)) |>.sum

theorem sum_of_first_100_terms : S 100 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_100_terms_l658_65831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l658_65825

theorem problem_solution (a b : ℝ) (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l658_65825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l658_65878

noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem f_properties :
  (∀ x > 3, f x ≥ 9) ∧
  (∃ x > 3, f x = 9) ∧
  (∀ t : ℝ, (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ t ≤ -2 ∨ t > -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l658_65878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_divisible_by_2001_l658_65824

open Finset

def product_even (n : ℕ) : ℕ := (range n).prod (λ i => 2 * (i + 1))

def product_odd (n : ℕ) : ℕ := (range n).prod (λ i => 2 * i + 1)

theorem product_difference_divisible_by_2001 :
  ∃ k : ℤ, (product_even 1000 : ℤ) - (product_odd 1000 : ℤ) = 2001 * k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_divisible_by_2001_l658_65824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l658_65873

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^3 - 27) / (x + 27)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Iio (-27) ∪ Set.Ioi (-27) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l658_65873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_conditions_imply_c_equals_5_l658_65861

noncomputable def fraction (a b x : ℝ) : ℝ := (2 * x + a) / (x - b)

theorem fraction_conditions_imply_c_equals_5 (a b c : ℝ) :
  (∀ x, x ≠ 2 → fraction a b x ≠ 0⁻¹) →  -- fraction is undefined when x = 2
  (fraction a b (1/2) = 0) →              -- fraction equals 0 when x = 0.5
  (fraction a b c = 3) →                -- fraction equals 3 when x = c
  c = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_conditions_imply_c_equals_5_l658_65861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_3_value_l658_65805

-- Define the sequence a_n
def a : ℕ → ℤ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 - 2^(n + 1)

-- State the theorem
theorem a_3_value : a 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_3_value_l658_65805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_specific_frustum_l658_65872

/-- Represents a truncated pyramid (frustum) with square base and top --/
structure Frustum where
  baseSide : ℝ
  topSide : ℝ
  lateralEdge : ℝ

/-- The maximum distance between any two vertices of the frustum --/
noncomputable def maxDistance (f : Frustum) : ℝ :=
  Real.sqrt (max (f.baseSide ^ 2 * 2) ((f.baseSide - f.topSide) ^ 2 + f.lateralEdge ^ 2))

/-- Theorem stating the maximum distance for a specific frustum --/
theorem max_distance_specific_frustum :
  let f : Frustum := ⟨4, 2, 4⟩
  maxDistance f = Real.sqrt 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_specific_frustum_l658_65872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_line_l658_65891

/-- Given a line 3x - 4y + 12 = 0 intersecting the x-axis at point A and y-axis at point B,
    prove that the standard equation of the circle with diameter AB is (x + 2)² + (y - 3/2)² = 25/4 -/
theorem circle_equation_from_line (x y : ℝ) : 
  let line := {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 + 12 = 0}
  let A : ℝ × ℝ := (-4, 0)  -- x-intercept
  let B : ℝ × ℝ := (0, 3)   -- y-intercept
  (x + 2)^2 + (y - 3/2)^2 = 25/4 
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_line_l658_65891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_a_range_when_f3_less_than_11_2_l658_65856

/-- The function f(x) for a given a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1/a| + |x - a + 1|

/-- Theorem stating that f(x) ≥ 1 for all x and a > 0 -/
theorem f_lower_bound (a : ℝ) (h : a > 0) : ∀ x, f a x ≥ 1 := by
  sorry

/-- Theorem stating the range of a when f(3) < 11/2 -/
theorem a_range_when_f3_less_than_11_2 (a : ℝ) (h1 : a > 0) (h2 : f a 3 < 11/2) :
  2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_a_range_when_f3_less_than_11_2_l658_65856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_Q_to_P_l658_65812

noncomputable def circle_P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

noncomputable def Q (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

theorem max_distance_Q_to_P :
  ∀ α : ℝ, ∃ d : ℝ,
    (∀ p ∈ circle_P, dist (Q α) p ≤ d) ∧
    (∃ p ∈ circle_P, dist (Q α) p = d) ∧
    d = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_Q_to_P_l658_65812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_explicit_l658_65881

/-- A linear function f satisfying f(2) = 1 and f(3) = -5 -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (k b : ℝ), ∀ x, f x = k * x + b

/-- f(2) = 1 -/
axiom f_2 : f 2 = 1

/-- f(3) = -5 -/
axiom f_3 : f 3 = -5

/-- The explicit form of f(x) is -6x + 13 -/
theorem f_explicit : ∀ x, f x = -6 * x + 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_explicit_l658_65881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l658_65896

-- Define set M
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set N
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ico 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l658_65896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_men_count_l658_65888

theorem work_completion_men_count : ℕ := by
  -- Define the number of days for the first group
  let days_first_group : ℕ := 55
  -- Define the number of days for the second group
  let days_second_group : ℕ := 121
  -- Define the number of men in the second group
  let men_second_group : ℕ := 10
  -- Define the total work in man-days
  let total_work : ℕ := men_second_group * days_second_group
  -- Define the number of men in the first group
  let men_first_group : ℕ := total_work / days_first_group
  -- Theorem statement
  have : men_first_group = 22 := by sorry
  exact men_first_group


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_men_count_l658_65888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_charge_is_7_75_l658_65864

/-- Represents the charges and attendance for a 4-day show -/
structure ShowData where
  day1_charge : ℚ
  day2_charge : ℚ
  day3_charge : ℚ
  day4_charge : ℚ
  day1_attendance : ℚ
  day2_attendance : ℚ
  day3_attendance : ℚ
  day4_attendance : ℚ

/-- Calculates the average charge per person for the whole show -/
def averageCharge (data : ShowData) : ℚ :=
  let total_revenue := data.day1_charge * data.day1_attendance +
                       data.day2_charge * data.day2_attendance +
                       data.day3_charge * data.day3_attendance +
                       data.day4_charge * data.day4_attendance
  let total_attendance := data.day1_attendance + data.day2_attendance +
                          data.day3_attendance + data.day4_attendance
  total_revenue / total_attendance

/-- Theorem stating that the average charge per person for the given show data is 7.75 -/
theorem average_charge_is_7_75 (data : ShowData) (x : ℚ)
  (h1 : data.day1_charge = 25)
  (h2 : data.day2_charge = 15)
  (h3 : data.day3_charge = 15/2)
  (h4 : data.day4_charge = 5/2)
  (h5 : data.day1_attendance = 3 * x)
  (h6 : data.day2_attendance = 7 * x)
  (h7 : data.day3_attendance = 11 * x)
  (h8 : data.day4_attendance = 19 * x)
  (h9 : x > 0) :
  averageCharge data = 31/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_charge_is_7_75_l658_65864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_and_degeneracy_l658_65801

/-- An equilateral triangle in the plane -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : dist A B = dist B C ∧ dist B C = dist C A

/-- The circumcircle of a triangle -/
def circumcircle (t : EquilateralTriangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p t.A = dist p t.B ∧ dist p t.B = dist p t.C}

/-- Distance between two points -/
noncomputable def mydist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem triangle_formation_and_degeneracy 
  (t : EquilateralTriangle) (M : ℝ × ℝ) :
  (M ∉ circumcircle t → 
    mydist M t.A + mydist M t.B > mydist M t.C ∧
    mydist M t.B + mydist M t.C > mydist M t.A ∧
    mydist M t.A + mydist M t.C > mydist M t.B) ∧
  (M ∈ circumcircle t → 
    mydist M t.A + mydist M t.B = mydist M t.C ∨
    mydist M t.B + mydist M t.C = mydist M t.A ∨
    mydist M t.A + mydist M t.C = mydist M t.B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_and_degeneracy_l658_65801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_when_m_1_range_m_when_range_f_is_real_range_m_when_f_increasing_l658_65845

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x - m)

-- Theorem 1: Domain of f when m = 1
theorem domain_f_when_m_1 :
  {x : ℝ | (x > 1) ∨ (x < -1)} = {x : ℝ | ∃ y, f 1 x = y} := by
  sorry

-- Theorem 2: Range of m when range of f is ℝ
theorem range_m_when_range_f_is_real (m : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f m x = y) → (m ≥ 0 ∨ m ≤ -4) := by
  sorry

-- Theorem 3: Range of m when f is increasing on (-∞, 1)
theorem range_m_when_f_increasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f m x₁ < f m x₂) →
  (m ≥ 2 - Real.sqrt 2 ∧ m < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_when_m_1_range_m_when_range_f_is_real_range_m_when_f_increasing_l658_65845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_x_squared_greater_than_x_l658_65829

theorem solution_set_x_squared_greater_than_x :
  {x : ℝ | x^2 > x} = Set.Ioi 1 ∪ Set.Iio 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_x_squared_greater_than_x_l658_65829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_product_l658_65818

-- Define a function that is symmetric to log x about y = x
noncomputable def f (x : ℝ) : ℝ := 
  Real.exp (Real.log x)

-- Theorem statement
theorem symmetric_function_product (h : ∀ x > 0, f (Real.log x) = x) :
  f (Real.log 2) * f (Real.log 5) = 10 := by
  -- Replace log_two and log_five with their actual values
  have h1 : f (Real.log 2) = 2 := h 2 (by norm_num)
  have h2 : f (Real.log 5) = 5 := h 5 (by norm_num)
  -- Multiply the results
  calc
    f (Real.log 2) * f (Real.log 5) = 2 * 5 := by rw [h1, h2]
    _ = 10 := by norm_num
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_product_l658_65818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_schools_l658_65835

theorem stratified_sampling_schools (total_schools : ℕ) (elementary_schools : ℕ) (middle_schools : ℕ) 
  (sample_size : ℕ) (h1 : total_schools = 250) (h2 : elementary_schools = 150) (h3 : middle_schools = 75) 
  (h4 : sample_size = 30) :
  let prob := (sample_size : ℚ) / total_schools
  ⌊prob * elementary_schools⌋ = 18 ∧ ⌊prob * middle_schools⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_schools_l658_65835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l658_65877

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x / 3) * Real.cos (x / 3) - 2 * (Real.sin (x / 3))^2

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_problem (t : Triangle) (h1 : f t.C = 1) (h2 : t.b^2 = t.a * t.c) :
  (∀ x, -3 ≤ f x ∧ f x ≤ 1) ∧ Real.sin t.A = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l658_65877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nines_in_product_l658_65833

def product : Nat := 123456789 * 999999999

def digits (n : Nat) : List Nat :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

theorem no_nines_in_product : ∀ d : Nat, d ∈ digits product → d ≠ 9 :=
by
  intro d hd
  -- The proof goes here
  sorry

#eval digits product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nines_in_product_l658_65833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_polar_curve_l658_65830

/-- The length of the segment cut by ρ = 1 from ρ sin θ - ρ cos θ = 1 --/
theorem segment_length_polar_curve : 
  ∃ (ρ θ : ℝ), 
    ρ = 1 ∧ 
    ρ * Real.sin θ - ρ * Real.cos θ = 1 ∧ 
    2 * Real.sqrt (1 - (1 / Real.sqrt 2)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_polar_curve_l658_65830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_plus_two_beta_l658_65887

theorem cos_two_alpha_plus_two_beta (α β : Real) 
  (h1 : Real.sin (α - β) = 1/3)
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_plus_two_beta_l658_65887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l658_65857

/-- Given two vectors a and b in a real inner product space, 
    prove that |a + b| = √2 given |a| = |b| = 1 and |a - b| = √2 -/
theorem vector_sum_magnitude {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : ‖a - b‖ = Real.sqrt 2) : 
  ‖a + b‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l658_65857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l658_65802

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x + Real.pi / 4)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem smallest_positive_period_of_f :
  ∃ p : ℝ, p > 0 ∧ is_periodic f p ∧ ∀ q, q > 0 → is_periodic f q → p ≤ q :=
by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l658_65802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l658_65807

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.cos (ω * x) ^ 2 + 1 / 2

theorem problem_solution (ω : ℝ) (h_ω : ω > 0)
  (h_sym : f ω (π / 3) = f ω (-π / 3))
  (h_zero : f ω (π / 12) = 0) :
  ∃ (A B C a b : ℝ),
    ω = 1 ∧
    (∀ x, f ω x = Real.sin (2 * x - π / 6)) ∧
    (∀ x y, x ∈ Set.Icc (-π / 12) (π / 3) → y ∈ Set.Icc (-π / 12) (π / 3) → x < y → f ω x < f ω y) ∧
    (∀ x y, x ∈ Set.Icc (π / 3) (5 * π / 12) → y ∈ Set.Icc (π / 3) (5 * π / 12) → x < y → f ω x > f ω y) ∧
    Real.sqrt 3 = 2 ∧
    f ω C = 1 ∧
    (1 : ℝ) * Real.sin B = 2 * Real.sin A ∧
    C = π / 3 ∧
    a = 1 ∧
    b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l658_65807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l658_65889

-- Define the function f(x) = 3^x + x^3 - 2
noncomputable def f (x : ℝ) : ℝ := Real.rpow 3 x + x^3 - 2

-- State the theorem
theorem unique_zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 0 1 ∧ f x = 0 := by
  sorry

#check unique_zero_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_in_interval_l658_65889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_right_triangle_l658_65858

-- Define the triangle area function
noncomputable def triangle_DEF (base height : ℝ) : ℝ := (1/2) * base * height

-- Theorem statement
theorem area_of_right_triangle :
  ∀ (base height : ℝ),
  base = 12 →
  height = 15 →
  triangle_DEF base height = 90 :=
by
  -- Introduce variables and hypotheses
  intro base height h_base h_height
  -- Unfold the definition of triangle_DEF
  unfold triangle_DEF
  -- Rewrite using the hypotheses
  rw [h_base, h_height]
  -- Simplify the arithmetic
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_right_triangle_l658_65858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l658_65868

/-- The ratio of coefficients in the vector equation of point F on a line intersecting a parabola -/
theorem intersection_ratio (l : Set (ℝ × ℝ)) (A B F : ℝ × ℝ) (lambda mu : ℝ) :
  (∀ x y, (x, y) ∈ l ↔ Real.sqrt 3 * x - y - Real.sqrt 3 = 0) →  -- Line equation
  (∀ x y, y^2 = 4*x → (x, y) ∈ l → (x, y) = A ∨ (x, y) = B) →   -- Parabola intersection
  F.2 = 0 ∧ F ∈ l →                                              -- F on x-axis and line
  F = lambda • A + mu • B →                                      -- Vector equation
  lambda ≤ mu →                                                  -- Coefficient condition
  lambda / mu = 1 / 3 :=                                         -- Conclusion
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l658_65868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l658_65898

/-- Given a hyperbola with equation x^2 - 4x - 12y^2 + 24y = -36,
    the distance between its foci is (2 * √273) / 3 -/
theorem hyperbola_foci_distance :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x^2 - 4*x - 12*y^2 + 24*y = -36 →
      (x - 2)^2 / a^2 - (y - 1)^2 / b^2 = 1) ∧
    c^2 = a^2 + b^2 ∧
    2 * c = 2 * Real.sqrt 273 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l658_65898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l658_65846

theorem sum_of_integers (x y : ℤ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x - y = 8) (h4 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l658_65846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_is_24_over_5_l658_65851

/-- A right triangle with an inscribed semicircle -/
structure RightTriangleWithSemicircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- The triangle PQR is a right triangle with the right angle at R -/
  is_right_triangle : pq^2 + qr^2 = (pq^2 + qr^2)
  /-- A semicircle is inscribed in the triangle, touching PQ and QR at their midpoints and the hypotenuse PR -/
  has_inscribed_semicircle : True

/-- The radius of the inscribed semicircle in a right triangle -/
noncomputable def semicircle_radius (t : RightTriangleWithSemicircle) : ℝ :=
  (t.pq * t.qr) / (t.pq + t.qr + Real.sqrt (t.pq^2 + t.qr^2))

/-- Theorem: The radius of the inscribed semicircle in the given right triangle is 24/5 -/
theorem semicircle_radius_is_24_over_5 (t : RightTriangleWithSemicircle) 
  (h1 : t.pq = 15) (h2 : t.qr = 8) : semicircle_radius t = 24 / 5 := by
  sorry

#check semicircle_radius_is_24_over_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_is_24_over_5_l658_65851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_d_values_l658_65804

def is_valid_pattern (m : ℕ) (d : ℕ) : Prop :=
  ∃ (digits : List ℕ),
    digits.length = 90 ∧
    (∀ i, i ∈ digits → i = 2 ∨ i = d) ∧
    (∀ i, i < 12 → 
      (digits.take ((i + 1) * (i + 2) / 2)).count 2 = (i + 1) * (i + 2) / 2 - (i + 1) ∧
      (digits.take ((i + 1) * (i + 2) / 2)).count d = i + 1)

theorem valid_d_values (m d : ℕ) :
  is_valid_pattern m d →
  m % 9 = 0 →
  d = 5 ∨ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_d_values_l658_65804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l658_65844

/-- Given a triangle ABC with the following properties:
    - A is at the origin (0, 0)
    - B' is on the x-axis at distance b from A
    - C' is on the x-axis at distance c from A
    - B is b' units above B'
    - C is c' units above C'
    This theorem states that the volume of the solid of revolution
    formed by rotating the triangle ABC around the x-axis is
    (π/3) * (b' + c') * (bc' - b'c) -/
theorem volume_of_revolution (b c b' c' : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (b, b')
  let C : ℝ × ℝ := (c, c')
  (π/3) * (b' + c') * (b*c' - b'*c) = (π/3) * (b' + c') * (b*c' - b'*c) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l658_65844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_six_is_max_l658_65879

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the right triangle condition
def is_right_triangle (A B P : ℝ × ℝ) : Prop :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0

-- State the theorem
theorem max_m_value :
  ∀ m : ℝ, m > 0 →
  (∃ P : ℝ × ℝ, 
    circle_C P.1 P.2 ∧
    is_right_triangle (point_A m) (point_B m) P) →
  m ≤ 6 :=
by
  sorry

-- Prove that 6 is indeed the maximum value
theorem six_is_max :
  ∃ m : ℝ, m > 0 ∧ 
  (∃ P : ℝ × ℝ, 
    circle_C P.1 P.2 ∧
    is_right_triangle (point_A m) (point_B m) P) ∧
  m = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_six_is_max_l658_65879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l658_65847

-- Define the circle in polar coordinates
def polarCircle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the line in polar coordinates
def polarLine (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Theorem stating that the line is tangent to the circle
theorem line_tangent_to_circle :
  ∃ (ρ₀ θ₀ : ℝ), polarCircle ρ₀ θ₀ ∧ polarLine ρ₀ θ₀ ∧
  ∀ (ρ θ : ℝ), polarCircle ρ θ ∧ polarLine ρ θ → (ρ, θ) = (ρ₀, θ₀) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l658_65847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l658_65869

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (4^(-x) - 1) * (2^x - 3)^5
  ∃ (f : ℝ → ℝ), expansion = f x - 27 ∧ (∀ x, f x ≠ 0 → f x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l658_65869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l658_65821

/-- The complex number -3 - i√7 -/
noncomputable def z : ℂ := -3 - Complex.I * Real.sqrt 7

/-- The polynomial x² + 6x + 16 -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 16

theorem monic_quadratic_with_complex_root :
  (∀ x : ℂ, (p x.re : ℂ) = p x.re + x.im * 0) ∧  -- p has real coefficients
  (p z.re : ℂ) + z.im * Complex.I = 0 ∧          -- z is a root of p
  (∀ q : ℝ → ℝ, (∀ x : ℂ, (q x.re : ℂ) = q x.re + x.im * 0) →  -- for any real polynomial q
    (q z.re : ℂ) + z.im * Complex.I = 0 →        -- that has z as a root
    (∃ a b : ℝ, ∀ x : ℝ, q x = x^2 + a*x + b) →  -- and is monic quadratic
    (∀ x : ℝ, q x = p x)) :=                     -- q must equal p
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l658_65821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_eight_l658_65854

/-- A function that represents the set of possible digits for x and y -/
def possibleDigits : Set ℕ := {3, 58}

/-- A function that constructs the 6-digit number given x and y -/
def constructNumber (x y : ℕ) : ℕ := 460000 + x * 1000 + y * 100 + 12

/-- A predicate that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

/-- The theorem to be proved -/
theorem probability_divisible_by_eight :
  ∀ x y, x ∈ possibleDigits → y ∈ possibleDigits →
    isSingleDigit x → isSingleDigit y →
    (constructNumber x y) % 8 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_eight_l658_65854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_cos_identity_l658_65876

theorem sin_squared_cos_identity (θ : ℝ) : 
  (Real.sin θ)^2 * Real.cos θ = -(1/4) * Real.cos (3*θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_cos_identity_l658_65876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_x_intercept_on_line_y_intercept_on_line_l658_65852

/-- A line is described by the equation y + 3 = 5(x - 6). -/
def line_equation (x y : ℝ) : Prop := y + 3 = 5 * (x - 6)

/-- The x-intercept of the line. -/
noncomputable def x_intercept : ℝ := 33 / 5

/-- The y-intercept of the line. -/
def y_intercept : ℝ := -33

/-- The sum of x-intercept and y-intercept is -132/5. -/
theorem intercept_sum : x_intercept + y_intercept = -132 / 5 := by
  sorry

/-- The x-intercept satisfies the line equation when y = 0. -/
theorem x_intercept_on_line : line_equation x_intercept 0 := by
  sorry

/-- The y-intercept satisfies the line equation when x = 0. -/
theorem y_intercept_on_line : line_equation 0 y_intercept := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_x_intercept_on_line_y_intercept_on_line_l658_65852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_digit_square_problem_l658_65838

/-- A function that checks if a number is of the form 2525******89 -/
def isOfRequiredForm (n : Nat) : Prop :=
  ∃ (a b c d e f : Nat), n = 2525000000 + a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + 89

/-- The main theorem -/
theorem twelve_digit_square_problem (x : Nat) :
  (x^2 ≥ 100000000000) ∧ (x^2 < 1000000000000) ∧ isOfRequiredForm (x^2) →
  x = 502517 ∨ x = 502533 ∨ x = 502567 ∨ x = 502583 := by
  sorry

#check twelve_digit_square_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_digit_square_problem_l658_65838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_divisors_correct_l658_65843

/-- The number of integers between 1 and 1440 (inclusive) that are
    divisors of 1440 and multiples of 5 -/
def count_special_divisors : ℕ := 18

/-- A function that checks if a number is a divisor of 1440 and a multiple of 5 -/
def is_special_divisor (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 1440 ∧ 1440 % n = 0 ∧ n % 5 = 0

theorem count_special_divisors_correct :
  (Finset.filter (fun n => is_special_divisor n) (Finset.range 1441)).card = count_special_divisors := by
  sorry

#eval count_special_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_divisors_correct_l658_65843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vegetarian_dishes_l658_65832

def number_of_combinations (n : ℕ) : ℕ := Nat.choose 5 2 * Nat.choose n 2

theorem min_vegetarian_dishes : ∃ n : ℕ, n ≥ 7 ∧ 
  (∀ m : ℕ, m < 7 → number_of_combinations m ≤ 200) ∧
  (∀ k : ℕ, k ≥ 7 → number_of_combinations k > 200) := by
  sorry

#eval number_of_combinations 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vegetarian_dishes_l658_65832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l658_65855

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - x
def g (x : ℝ) : ℝ := |x| + |x-1| + 1

-- Define the region
def region (p q : ℝ) : Prop :=
  q ≤ (p+1)^2 / 4 ∧ 
  ((p ≤ 0 ∧ q < 2) ∨ (p > 0 ∧ q < 2 - p))

-- State the theorem
theorem area_of_region :
  ∃ A : Set (ℝ × ℝ), 
    (∀ (p q : ℝ), (p, q) ∈ A ↔ region p q) ∧
    (MeasureTheory.volume A = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l658_65855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l658_65815

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 18

/-- The center of the circle -/
def center : ℝ × ℝ := (3, -4)

/-- The given point -/
def given_point : ℝ × ℝ := (-3, 4)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_properties :
  (distance center given_point = 10) ∧
  (∀ x y : ℝ, circle_equation x y → distance center (x, y) = Real.sqrt 43) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l658_65815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l658_65850

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * 1000 / 3600 →
  platform_length = 250 →
  crossing_time = 22 →
  speed * crossing_time - platform_length = 190 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l658_65850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_one_cell_first_player_wins_two_cells_l658_65880

/-- Represents a cell on the infinite grid --/
structure Cell where
  x : ℤ
  y : ℤ
deriving Repr, DecidableEq

/-- Represents the color of a cell --/
inductive Color
  | Red
  | Blue
  | Uncolored
deriving Repr, DecidableEq

/-- Represents the game state --/
structure GameState where
  coloredCells : Cell → Color

/-- Represents a player's move --/
structure Move where
  cell : Cell
  color : Color

/-- Checks if four cells form a square --/
def formsSquare (c1 c2 c3 c4 : Cell) : Prop :=
  ∃ (d : ℤ), d ≠ 0 ∧
    ((c2.x - c1.x)^2 + (c2.y - c1.y)^2 = d^2) ∧
    ((c3.x - c2.x)^2 + (c3.y - c2.y)^2 = d^2) ∧
    ((c4.x - c3.x)^2 + (c4.y - c3.y)^2 = d^2) ∧
    ((c1.x - c4.x)^2 + (c1.y - c4.y)^2 = d^2)

/-- Represents a winning condition for the first player --/
def firstPlayerWins (state : GameState) : Prop :=
  ∃ (c1 c2 c3 c4 : Cell),
    formsSquare c1 c2 c3 c4 ∧
    state.coloredCells c1 = Color.Red ∧
    state.coloredCells c2 = Color.Red ∧
    state.coloredCells c3 = Color.Red ∧
    state.coloredCells c4 = Color.Red

/-- Function to apply a single move to a game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  { coloredCells := λ c => if c = move.cell then move.color else state.coloredCells c }

/-- Function to apply a series of moves to a game state --/
def applyMoves : ℕ → (GameState → Move) → (GameState → Move) → GameState
  | 0, _, _ => { coloredCells := λ _ => Color.Uncolored }
  | n+1, s1, s2 => 
    let state := applyMoves n s1 s2
    let move1 := s1 state
    let state' := applyMove state move1
    let move2 := s2 state'
    applyMove state' move2

/-- Theorem stating that the first player has a winning strategy when the second player colors one cell per turn --/
theorem first_player_wins_one_cell :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
    ∃ (n : ℕ), firstPlayerWins (applyMoves n strategy opponent_strategy) := by
  sorry

/-- Theorem stating that the first player has a winning strategy when the second player colors two cells per turn --/
theorem first_player_wins_two_cells :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move × Move),
    ∃ (n : ℕ), firstPlayerWins (applyMoves n strategy (λ s => (opponent_strategy s).1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_one_cell_first_player_wins_two_cells_l658_65880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_6_l658_65822

/-- A monic quintic polynomial with specific values at x = 1, 2, 3, 4, 5 -/
noncomputable def p : ℝ → ℝ := fun x ↦
  x^5 + p.a*x^4 + p.b*x^3 + p.c*x^2 + p.d*x + p.e
where
  a : ℝ := sorry
  b : ℝ := sorry
  c : ℝ := sorry
  d : ℝ := sorry
  e : ℝ := sorry

/-- p is monic quintic -/
axiom p_monic : p = fun x ↦ x^5 + p.a*x^4 + p.b*x^3 + p.c*x^2 + p.d*x + p.e

/-- Conditions for p at specific points -/
axiom p_1 : p 1 = 3
axiom p_2 : p 2 = 7
axiom p_3 : p 3 = 13
axiom p_4 : p 4 = 21
axiom p_5 : p 5 = 31

/-- Theorem: p(6) = 158 -/
theorem p_6 : p 6 = 158 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_6_l658_65822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l658_65871

/-- Represents the number of tulips in the arrangement -/
def tulips : ℕ := sorry

/-- Represents the number of lilies in the arrangement -/
def lilies : ℕ := sorry

/-- The total cost of the arrangement is $120 -/
axiom total_cost : 4 * tulips + 6 * lilies = 120

/-- The total number of flowers is 25 -/
axiom total_flowers : tulips + lilies = 25

/-- The arrangement consists of 15 tulips and 10 lilies -/
theorem unique_arrangement : tulips = 15 ∧ lilies = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l658_65871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l658_65842

theorem complex_modulus_example : 
  Complex.abs (1 + 2*Complex.I) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l658_65842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l658_65874

theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x + y) + x * f y - 2 * x * y - x + 2) : 
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l658_65874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l658_65892

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively. --/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem triangle_properties (t : Triangle) :
  (t.a / Real.sin t.B = t.b / Real.sin t.A → t.a = t.b) ∧
  (t.a = t.b * Real.sin t.C + t.c * Real.cos t.B → t.C = π / 4) ∧
  (Real.tan t.A + Real.tan t.B + Real.tan t.C < 0 → t.A > π / 2 ∨ t.B > π / 2 ∨ t.C > π / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l658_65892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l658_65853

/-- A function f(x) = sin(ω*x + π/3) with the given properties has ω = 14/3 -/
theorem omega_value (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x + π / 3)) →
  ω > 0 →
  f (π / 6) = f (π / 3) →
  (∃ m ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x ≥ f m) →
  (∀ x ∈ Set.Ioo (π / 6) (π / 3), ∃ y ∈ Set.Ioo (π / 6) (π / 3), f y > f x) →
  ω = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l658_65853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l658_65811

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ Real.sqrt 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = Real.sqrt 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l658_65811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_for_multiples_of_4_l658_65820

def is_multiple_of_4 (n : Nat) : Prop := ∃ k, n = 4 * k

theorem unique_digit_for_multiples_of_4 
  (A B C : Nat) 
  (h1 : A < 10) 
  (h2 : B < 10) 
  (h3 : C < 10) 
  (h4 : is_multiple_of_4 (6800000 + 80000 * A + 7000 + 100 * B + 32)) 
  (h5 : is_multiple_of_4 (4100000 + 10000 * B + 1000 * A + 500 + 10 * C + 9)) : 
  C = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_for_multiples_of_4_l658_65820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l658_65809

/-- Given log 2 = 0.3010 and log 3 = 0.4771, prove that the solution x to 2^(x+4) = 176 is approximately 3.46 -/
theorem log_equation_solution (h1 : Real.log 2 = 0.3010) (h2 : Real.log 3 = 0.4771) :
  ∃ x : ℝ, (2 : ℝ)^(x + 4) = 176 ∧ |x - 3.46| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l658_65809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_expenditure_l658_65800

/-- The annual average expenditure function for a family sedan -/
noncomputable def y (x : ℝ) : ℝ := (0.05 * x^2 + 2 * x + 1.25) / x

/-- The theorem stating the minimum value of the expenditure function -/
theorem min_expenditure :
  ∀ x > 0, y x ≥ 2.5 ∧ y 5 = 2.5 := by
  sorry

#check min_expenditure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_expenditure_l658_65800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_grid_l658_65866

/-- Represents a 3x3 grid with letters 'A', 'B', and 'C' -/
def Grid := Fin 3 → Fin 3 → Char

/-- Checks if a grid is valid according to the problem constraints -/
def is_valid_grid (g : Grid) : Prop :=
  (g 0 0 = 'A') ∧ 
  (g 1 1 = 'B') ∧
  (∀ i : Fin 3, Finset.toSet {g i 0, g i 1, g i 2} = Finset.toSet {'A', 'B', 'C'}) ∧
  (∀ j : Fin 3, Finset.toSet {g 0 j, g 1 j, g 2 j} = Finset.toSet {'A', 'B', 'C'})

/-- The main theorem stating that there is exactly one valid grid arrangement -/
theorem unique_valid_grid : ∃! g : Grid, is_valid_grid g := by
  sorry

#check unique_valid_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_grid_l658_65866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_a_b_same_day_l658_65823

/-- The number of people to be arranged --/
def n : ℕ := 8

/-- The number of days --/
def d : ℕ := 4

/-- The number of people arranged each day --/
def k : ℕ := 2

/-- The probability that A and B are arranged on the same day --/
def prob_same_day : ℚ := 1 / 7

/-- Theorem stating the probability of A and B being arranged on the same day --/
theorem prob_a_b_same_day :
  (Nat.choose n k * Nat.choose (n - k) k * Nat.choose (n - 2*k) k * Nat.choose (n - 3*k) k) ≠ 0 →
  (d * Nat.choose (n - 2) k * Nat.choose (n - k - 2) k * Nat.choose (n - 2*k - 2) k : ℚ) /
  (Nat.choose n k * Nat.choose (n - k) k * Nat.choose (n - 2*k) k * Nat.choose (n - 3*k) k : ℚ) = prob_same_day :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_a_b_same_day_l658_65823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l658_65819

noncomputable def f (x : ℝ) : ℝ := x^2 / 2 - 3 * Real.log x

def perpendicular_line (x y : ℝ) : Prop := x + 2 * y + 10 = 0

theorem tangent_point_abscissa :
  ∃ (x : ℝ), x > 0 ∧
    (∃ (y : ℝ), f x = y ∧
      (∃ (m : ℝ), (deriv f x = m) ∧
        (m * (-1/2) = -1) ∧
        (∀ (x' y' : ℝ), perpendicular_line x' y' → (y' - y = m * (x' - x)))))
  ∧ x = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l658_65819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zendaya_time_fraction_l658_65810

/-- Represents the time taken by Casey to complete the marathon -/
noncomputable def casey_time : ℝ := 6

/-- Represents the average time taken by Casey and Zendaya to complete the marathon -/
noncomputable def average_time : ℝ := 7

/-- Represents the fraction of Casey's time that Zendaya takes longer -/
noncomputable def fraction : ℝ := 1/3

/-- Theorem stating that the fraction of Casey's time that Zendaya takes longer is 1/3 -/
theorem zendaya_time_fraction :
  let zendaya_time := casey_time * (1 + fraction)
  (casey_time + zendaya_time) / 2 = average_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zendaya_time_fraction_l658_65810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l658_65816

/-- Volume of a regular tetrahedron given its edge length -/
noncomputable def volume_regular_tetrahedron (edge_length : ℝ) : ℝ :=
  (1 / 6) * edge_length^3 * (1 / Real.sqrt 2)

/-- The volume of a regular tetrahedron with edge length 1 is 1/(6√2) -/
theorem regular_tetrahedron_volume (edge_length : ℝ) (h : edge_length = 1) :
  volume_regular_tetrahedron edge_length = 1 / (6 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l658_65816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_phone_number_probability_l658_65828

def first_three_options : ℕ := 2  -- 293 or 296
def last_four_digits : ℕ := 4  -- 0, 2, 5, 8

def total_combinations : ℕ := first_three_options * (Nat.factorial last_four_digits)

theorem correct_phone_number_probability : 
  (1 : ℚ) / total_combinations = (1 : ℚ) / 48 := by
  -- Proof steps would go here
  sorry

#eval total_combinations  -- This will evaluate to 48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_phone_number_probability_l658_65828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_divisor_product_5832_l658_65839

open Nat Finset

def divisor_product (n : ℕ) : ℕ := (filter (· ∣ n) (range (n + 1))).prod id

theorem unique_number_with_divisor_product_5832 :
  ∃! n : ℕ, n > 0 ∧ divisor_product n = 5832 ∧ ∃ x y : ℕ, n = 2^x * 3^y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_divisor_product_5832_l658_65839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l658_65806

/-- Circle C₁ with equation x² + y² + 2x + 3y + 1 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 3*y + 1 = 0

/-- Circle C₂ with equation x² + y² + 4x + 3y + 2 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3*y + 2 = 0

/-- The center of circle C₁ -/
noncomputable def center_C₁ : ℝ × ℝ := (-1, -3/2)

/-- The center of circle C₂ -/
noncomputable def center_C₂ : ℝ × ℝ := (-2, -3/2)

/-- The radius of circle C₁ -/
noncomputable def radius_C₁ : ℝ := 3/2

/-- The radius of circle C₂ -/
noncomputable def radius_C₂ : ℝ := Real.sqrt 17 / 2

/-- The distance between the centers of C₁ and C₂ -/
noncomputable def distance_centers : ℝ := 1

/-- Theorem stating that circles C₁ and C₂ are intersecting -/
theorem circles_intersect : 
  radius_C₂ + radius_C₁ > distance_centers ∧ 
  distance_centers > abs (radius_C₂ - radius_C₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l658_65806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_equals_26_l658_65883

-- Define the series A
noncomputable def A : ℝ := ∑' n, if (n % 2 = 1 ∧ n % 5 ≠ 0) then ((-1)^((n - 1) / 2) / n^2) else 0

-- Define the series B
noncomputable def B : ℝ := ∑' n, if (n % 10 = 5) then ((-1)^((n - 5) / 10) / n^2) else 0

-- Theorem statement
theorem A_div_B_equals_26 : A / B = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_equals_26_l658_65883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_product_l658_65836

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Square EFGH on a 6x6 grid -/
structure SquareEFGH where
  E : GridPoint
  F : GridPoint
  G : GridPoint
  H : GridPoint

/-- Definition of the specific square EFGH from the problem -/
def specificSquareEFGH : SquareEFGH :=
  { E := ⟨1, 5⟩
    F := ⟨5, 6⟩
    G := ⟨6, 2⟩
    H := ⟨2, 1⟩ }

/-- Calculate the squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℕ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Calculate the area of the square -/
def squareArea (s : SquareEFGH) : ℝ :=
  (squaredDistance s.E s.F : ℝ)

/-- Calculate the perimeter of the square -/
noncomputable def squarePerimeter (s : SquareEFGH) : ℝ :=
  4 * Real.sqrt (squaredDistance s.E s.F : ℝ)

/-- The main theorem to prove -/
theorem area_perimeter_product (s : SquareEFGH) :
  s = specificSquareEFGH →
  squareArea s * squarePerimeter s = 68 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_perimeter_product_l658_65836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l658_65870

/-- The slope of the line -/
def m : ℚ := 4/3

/-- The y-intercept of the line -/
def b : ℚ := -20/3

/-- The line equation -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- Parameterization A -/
def param_A (t : ℚ) : ℚ × ℚ := (5 - 3*t, -4*t)

/-- Parameterization B -/
def param_B (t : ℚ) : ℚ × ℚ := (20 + 9*t, 4 + 12*t)

/-- Parameterization C -/
def param_C (t : ℚ) : ℚ × ℚ := (3 + 3*t/4, -7/3 + t)

/-- Parameterization D -/
def param_D (t : ℚ) : ℚ × ℚ := (15/4 + t, -1 + 4*t/3)

/-- Parameterization E -/
def param_E (t : ℚ) : ℚ × ℚ := (12*t, -20/3 - 16*t)

/-- A parameterization is valid if it satisfies the line equation for all t -/
def is_valid_param (p : ℚ → ℚ × ℚ) : Prop :=
  ∀ t, let (x, y) := p t; line_eq x y

theorem parameterization_validity :
  is_valid_param param_A ∧
  is_valid_param param_B ∧
  ¬is_valid_param param_C ∧
  ¬is_valid_param param_D ∧
  ¬is_valid_param param_E := by
  sorry

#check parameterization_validity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l658_65870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l658_65859

open Real

theorem equation_roots_properties (θ : ℝ) (m : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π/2))
  (h2 : 2*(sin θ)^2 - (Real.sqrt 3 + 1)*(sin θ) + m = 0)
  (h3 : 2*(cos θ)^2 - (Real.sqrt 3 + 1)*(cos θ) + m = 0) :
  (m = Real.sqrt 3 / 2) ∧ 
  ((sin θ / (1 - (cos θ / sin θ))) + (cos θ / (1 - (sin θ / cos θ))) = (Real.sqrt 3 + 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l658_65859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangent_lines_l658_65895

noncomputable section

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop :=
  (y^2 / a^2) + (x^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the parabola
def is_on_parabola (x y p : ℝ) : Prop :=
  x^2 = 2*p*y ∧ p > 0

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the common focus
def common_focus (e_a e_b p : ℝ) : Point :=
  ⟨0, p/2⟩

-- State the theorem
theorem perpendicular_tangent_lines 
  (a b p : ℝ)
  (P Q : Point)
  (h_P : is_on_parabola P.x P.y p)
  (h_Q : is_on_ellipse Q.x Q.y a b)
  (h_tangent : ∃ (m c : ℝ), P.y = m * P.x + c ∧ Q.y = m * Q.x + c) :
  let F := common_focus a b p
  (P.y - F.y) * (Q.y - F.y) = -(P.x - F.x) * (Q.x - F.x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangent_lines_l658_65895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l658_65808

-- Define the function f(x) = (x-3)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

-- State the theorem
theorem f_increasing_interval :
  ∀ x : ℝ, x > 2 → (∀ y : ℝ, y > x → f y > f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l658_65808
