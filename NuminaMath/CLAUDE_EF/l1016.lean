import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_division_l1016_101615

noncomputable def reciprocal_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (↑(i + 1) : ℚ))

theorem integer_part_of_division (n : ℕ) :
  Int.floor (10 / (reciprocal_sum n)) = 3 ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_division_l1016_101615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1016_101607

def S (n : ℕ) : Finset ℕ := Finset.range n

def f (n : ℕ) : ℕ :=
  Finset.card (Finset.filter (fun q : ℕ × ℕ × ℕ × ℕ => 
    q.1 ∈ S n ∧ q.2.1 ∈ S n ∧ q.2.2.1 ∈ S n ∧ q.2.2.2 ∈ S n ∧
    q.1 ≠ q.2.1 ∧ q.1 ≠ q.2.2.1 ∧ q.1 ≠ q.2.2.2 ∧
    q.2.1 ≠ q.2.2.1 ∧ q.2.1 ≠ q.2.2.2 ∧ q.2.2.1 ≠ q.2.2.2 ∧
    q.1 - q.2.1 = q.2.2.1 - q.2.2.2) (Finset.product (S n) (Finset.product (S n) (Finset.product (S n) (S n)))))

theorem problem_solution (n : ℕ) (h : n ≥ 4) :
  f 6 = 56 ∧ f 40 = 38000 ∧ 2022 ∣ f 1348 ∧ 2022 ∣ f 1350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1016_101607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_compose_g_eq_5_has_5_solutions_l1016_101611

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 7

-- Define the composite function g(g(x))
noncomputable def g_compose_g (x : ℝ) : ℝ := g (g x)

-- State the theorem
theorem g_compose_g_eq_5_has_5_solutions :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, g_compose_g x = 5 ∧
  ∀ x ∉ s, g_compose_g x ≠ 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_compose_g_eq_5_has_5_solutions_l1016_101611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_conic_parameter_correct_l1016_101629

/-- The value of m for which the ellipse x^2 + 9y^2 = 9 and the hyperbola x^2 - m(y+3)^2 = 1 are tangent -/
noncomputable def tangent_conic_parameter : ℝ := 8/9

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + 9*y^2 = 9

/-- The equation of the hyperbola -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  x^2 - m*(y+3)^2 = 1

/-- Two conics are tangent if they intersect at exactly one point -/
def are_tangent (ellipse hyperbola : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, ellipse p.fst p.snd ∧ hyperbola p.fst p.snd

/-- Theorem stating that the calculated tangent_conic_parameter is correct -/
theorem tangent_conic_parameter_correct :
  are_tangent ellipse_equation (hyperbola_equation tangent_conic_parameter) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_conic_parameter_correct_l1016_101629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_solution_set_is_real_l1016_101628

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |a*x - a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 1} = Set.Iic (1/2) ∪ Set.Ici (3/2) :=
sorry

-- Theorem 2: Condition for solution set to be ℝ
theorem solution_set_is_real :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_solution_set_is_real_l1016_101628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtracted_number_l1016_101692

theorem subtracted_number (a b : ℕ) (N : ℝ) : 
  ((a * 10 + b : ℝ) ^ (4.5 : ℝ) - N = (b * 10 + a : ℝ) ^ (4.5 : ℝ) - 7) → N = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtracted_number_l1016_101692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_l1016_101633

theorem sine_equality (ω φ : ℝ) (h_ω : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * x + φ) = Real.sin (ω * (x + 1) + φ) - Real.sin (ω * (x + 2) + φ)) →
  ∀ x : ℝ, Real.sin (ω * x + φ + 9 * ω) = Real.sin (ω * x + φ - 9 * ω) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_l1016_101633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_pass_through_P_l1016_101639

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  coords : ℝ × ℝ

-- Helper functions (defined as opaque for now)
opaque reflection_law : Point → Point → Point → Prop
opaque collinear : Point → Point → Point → Prop

-- Define the problem setup
def problem_setup (c1 c2 : Circle) (P Q : Point) (A B : ℕ → Point) : Prop :=
  -- Circles intersect at P and Q
  (P.coords ∈ {x : ℝ × ℝ | (x.1 - c1.center.1)^2 + (x.2 - c1.center.2)^2 = c1.radius^2}) ∧
  (P.coords ∈ {x : ℝ × ℝ | (x.1 - c2.center.1)^2 + (x.2 - c2.center.2)^2 = c2.radius^2}) ∧
  (Q.coords ∈ {x : ℝ × ℝ | (x.1 - c1.center.1)^2 + (x.2 - c1.center.2)^2 = c1.radius^2}) ∧
  (Q.coords ∈ {x : ℝ × ℝ | (x.1 - c2.center.1)^2 + (x.2 - c2.center.2)^2 = c2.radius^2}) ∧
  -- Rays from Q reflect off circles according to law of reflection
  (∀ i : ℕ, reflection_law Q (A i) (A (i+1))) ∧
  (∀ i : ℕ, reflection_law Q (B i) (B (i+1))) ∧
  -- A₁, B₁, and P are collinear
  collinear P (A 1) (B 1)

-- Define the theorem
theorem all_lines_pass_through_P 
  (c1 c2 : Circle) (P Q : Point) (A B : ℕ → Point)
  (h : problem_setup c1 c2 P Q A B) :
  ∀ i : ℕ, collinear P (A i) (B i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_pass_through_P_l1016_101639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_softball_game_duration_l1016_101601

/-- Proves that the duration of each game is 2 hours given the problem conditions -/
theorem softball_game_duration :
  let num_daughters : ℕ := 2
  let games_per_daughter : ℕ := 8
  let practice_hours_per_game : ℕ := 4
  let total_hours_at_field : ℕ := 96
  let total_games : ℕ := num_daughters * games_per_daughter
  let total_practice_hours : ℕ := total_games * practice_hours_per_game
  let total_game_hours : ℕ := total_hours_at_field - total_practice_hours
  let game_duration : ℚ := (total_game_hours : ℚ) / (total_games : ℚ)
  game_duration = 2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_softball_game_duration_l1016_101601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_e_l1016_101677

theorem find_e (a b c d e : ℝ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  ({a + b, a + c, b + c} : Set ℝ).Subset {32, 36, 37} →
  ({d + e, c + e} : Set ℝ) = {48, 51} →
  e = 55 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_e_l1016_101677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_of_squares_l1016_101676

theorem arithmetic_sequence_of_squares (y : ℝ) (h : y > 0) :
  (∃ a d : ℝ, ∀ n : ℕ, (2 + 2*n)^2 = a + n*d) → y^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_of_squares_l1016_101676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_tangent_circles_l1016_101661

theorem internal_tangent_circles (C : ℕ) : 
  C = 150 → 
  (∃ count : ℕ, count = (Finset.filter (λ r : ℕ ↦ 0 < r ∧ r < C ∧ C % r = 0) (Finset.range C)).card) →
  count = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_internal_tangent_circles_l1016_101661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_l1016_101683

open Set Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem triangle_condition (a : ℝ) : 
  (a > 0) → 
  (∀ m n p : ℝ, m ∈ Icc (1/3 : ℝ) 1 → n ∈ Icc (1/3 : ℝ) 1 → p ∈ Icc (1/3 : ℝ) 1 → 
    f a m + f a n > f a p ∧ f a n + f a p > f a m ∧ f a p + f a m > f a n) → 
  a ∈ Ioo (1/15 : ℝ) (1/9 : ℝ) ∪ Ico 1 (5/3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_l1016_101683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_percentage_l1016_101681

noncomputable section

/-- Represents the markup percentage applied to the cost price -/
def markup : ℝ := 0.75

/-- Represents the profit percentage after discount -/
def profit : ℝ := 0.225

/-- Calculates the marked price given the cost price and markup -/
noncomputable def markedPrice (costPrice : ℝ) : ℝ := costPrice * (1 + markup)

/-- Calculates the selling price given the cost price and profit -/
noncomputable def sellingPrice (costPrice : ℝ) : ℝ := costPrice * (1 + profit)

/-- Calculates the discount percentage -/
noncomputable def discountPercentage (costPrice : ℝ) : ℝ :=
  (markedPrice costPrice - sellingPrice costPrice) / markedPrice costPrice

theorem merchant_discount_percentage :
  ∀ (costPrice : ℝ), costPrice > 0 → discountPercentage costPrice = 0.3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_discount_percentage_l1016_101681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l1016_101603

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def is_focus (F : ℝ × ℝ) (a b : ℝ) : Prop :=
  F.1^2 / a^2 + F.2^2 / b^2 = 1 - b^2 / a^2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_product_of_distances (M F₁ F₂ : ℝ × ℝ) :
  ellipse M.1 M.2 →
  is_focus F₁ 3 2 →
  is_focus F₂ 3 2 →
  distance M F₁ * distance M F₂ ≤ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l1016_101603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1016_101673

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (-3 * x + 1) / (x - 2)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | x ≠ 2}

-- Define the range of f
def range : Set ℝ := {y : ℝ | ∃ x ∈ domain, f x = y}

-- Theorem: The range of f is all real numbers except -3
theorem range_of_f : range = {y : ℝ | y ≠ -3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1016_101673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_satisfying_inequality_l1016_101605

theorem least_integer_satisfying_inequality :
  ∃ m : ℤ, 
    (∀ x : ℝ, (14 * x - 7 * (3 * x - 8) < 4 * (25 + x)) → (m : ℝ) > x) ∧ 
    (∀ n : ℤ, n < m → ∃ x : ℝ, (14 * x - 7 * (3 * x - 8) < 4 * (25 + x)) ∧ (n : ℝ) ≤ x) ∧
    m = -3 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_satisfying_inequality_l1016_101605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_numbers_l1016_101694

def available_digits : List Nat := [3, 5, 7, 8, 9]

def is_valid_two_digit_number (n : Nat) : Bool :=
  10 ≤ n && n ≤ 99 &&
  available_digits.any (λ d1 => 
    available_digits.any (λ d2 => 
      d1 ≠ d2 && n = 10 * d1 + d2))

theorem count_two_digit_numbers :
  (available_digits.map (λ d1 => 
    available_digits.filter (λ d2 => d1 ≠ d2)
    |>.map (λ d2 => 10 * d1 + d2))
  |>.join
  |>.filter is_valid_two_digit_number
  |>.length) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_two_digit_numbers_l1016_101694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_on_circle_distance_l1016_101693

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Function to check if a point lies on a circle
def lieOnCircle (p : IntPoint) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : IntPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

-- Theorem statement
theorem three_points_on_circle_distance (c : Circle) (p1 p2 p3 : IntPoint)
  (h_radius : c.radius > 0)
  (h_distinct : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3)
  (h_on_circle : lieOnCircle p1 c ∧ lieOnCircle p2 c ∧ lieOnCircle p3 c) :
  ∃ (pi pj : IntPoint), pi ≠ pj ∧ 
    distance pi pj ≥ c.radius^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_on_circle_distance_l1016_101693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_135_degrees_l1016_101686

/-- Given a line with an angle of inclination of 135° passing through point (1,1), 
    its equation is y = -x + 2 -/
theorem line_equation_135_degrees (l : Set (ℝ × ℝ)) (A : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = 135 ∧ ∀ (x y : ℝ), (x, y) ∈ l → y - A.2 = Real.tan θ * (x - A.1)) ∧ 
  A ∈ l ∧ A.1 = 1 ∧ A.2 = 1 → 
  ∀ x y, (x, y) ∈ l ↔ y = -x + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_135_degrees_l1016_101686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_size_l1016_101675

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 99}

def satisfies_ratio (a b : ℕ) : Prop :=
  a ≠ b ∧ (1/2 : ℚ) ≤ (b : ℚ) / (a : ℚ) ∧ (b : ℚ) / (a : ℚ) ≤ 2

theorem minimum_selection_size :
  ∀ (k : ℕ), k ≥ 7 →
    ∀ (T : Finset ℕ), ↑T ⊆ S → T.card = k →
      ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ satisfies_ratio a b
  ∧
  ∃ (U : Finset ℕ), ↑U ⊆ S ∧ U.card = 6 ∧
    ∀ (a b : ℕ), a ∈ U → b ∈ U → ¬(satisfies_ratio a b)
  := by sorry

#check minimum_selection_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selection_size_l1016_101675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_problem_l1016_101672

theorem arithmetic_geometric_mean_problem (x y : ℤ) (a b : ℕ) :
  x ≠ y →
  x > 0 →
  y > 0 →
  1 ≤ b →
  b ≤ 9 →
  0 ≤ a →
  a ≤ 9 →
  (x + y : ℚ) / 2 = 10 * b + a →
  Real.sqrt (x * y : ℝ) = 10 * a + b →
  ∃ (k : ℕ), |x - y| = 9 * k →
  |x - y| = 63 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_problem_l1016_101672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1016_101688

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem max_sum_arithmetic_sequence (a₁ : ℝ) (h₁ : a₁ > 0) :
  ∃ (d : ℝ), a₁ + 9 * (arithmeticSequence a₁ d 6) = 0 →
  ∃ (n : ℕ), ∀ (k : ℕ), sumArithmeticSequence a₁ d n ≥ sumArithmeticSequence a₁ d k ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1016_101688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_two_range_of_a_for_inequality_l1016_101638

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

-- Theorem 1
theorem min_value_when_a_is_two :
  ∃ (m : ℝ), ∀ (x : ℝ), x > 0 → h 2 x ≥ m ∧ ∃ (x₀ : ℝ), x₀ > 0 ∧ h 2 x₀ = m ∧ m = 3 := by
  sorry

-- Theorem 2
theorem range_of_a_for_inequality :
  ∀ (a : ℝ), a > 0 →
  (∀ (x : ℝ), x ≥ 1 → h a x ≥ 1) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_two_range_of_a_for_inequality_l1016_101638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1016_101649

theorem product_remainder (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a % 7 = 2 → b % 7 = 3 → c % 7 = 4 → (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1016_101649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubles_daily_half_covered_fully_covered_days_to_cover_lake_is_48_l1016_101659

/-- Represents the size of the lily pad patch on a given day -/
def patch_size : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * patch_size n

/-- Represents the size of the lake -/
def lake_size : ℚ := 2^48

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def days_to_cover_lake : ℕ := 48

/-- Represents the number of days it takes for the lily pad patch to cover half of the lake -/
def days_to_cover_half : ℕ := 47

/-- The size of the lily pad patch doubles each day -/
theorem doubles_daily : ∀ (n : ℕ), 2 * (patch_size n) = patch_size (n + 1) := by
  intro n
  rfl

/-- The patch covers half the lake after 'days_to_cover_half' days -/
theorem half_covered : patch_size days_to_cover_half = lake_size / 2 := by
  sorry

/-- The patch covers the entire lake after 'days_to_cover_lake' days -/
theorem fully_covered : patch_size days_to_cover_lake = lake_size := by
  sorry

/-- Theorem: It takes 48 days for the lily pad patch to cover the entire lake -/
theorem days_to_cover_lake_is_48 : days_to_cover_lake = 48 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubles_daily_half_covered_fully_covered_days_to_cover_lake_is_48_l1016_101659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1016_101685

/-- Given that the terminal side of angle α rotates counterclockwise by π/6 and 
    intersects the unit circle at point (3√10/10, √10/10), and tan(α + β) = 2/5,
    prove the following statements. -/
theorem trigonometric_identities 
  (α β : ℝ) 
  (h1 : (3 * Real.sqrt 10 / 10 : ℝ) = Real.cos (α + π/6))
  (h2 : (Real.sqrt 10 / 10 : ℝ) = Real.sin (α + π/6))
  (h3 : Real.tan (α + β) = 2/5) :
  (Real.sin (2*α + π/6) = (3 * Real.sqrt 3 - 4) / 10) ∧ 
  (Real.tan (2*β - π/3) = 17/144) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1016_101685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manager_salary_is_185000_l1016_101632

/-- Represents the average salary of employees in a company --/
structure CompanySalaries where
  marketer_ratio : ℝ
  engineer_ratio : ℝ
  marketer_salary : ℝ
  engineer_salary : ℝ
  overall_average : ℝ

/-- Calculates the average salary of managers given the company's salary structure --/
noncomputable def manager_average_salary (c : CompanySalaries) : ℝ :=
  (c.overall_average - c.marketer_ratio * c.marketer_salary - c.engineer_ratio * c.engineer_salary) /
  (1 - c.marketer_ratio - c.engineer_ratio)

/-- Theorem stating that the average salary of managers is $185,000 given the specific conditions --/
theorem manager_salary_is_185000 (c : CompanySalaries) 
  (h1 : c.marketer_ratio = 0.7)
  (h2 : c.engineer_ratio = 0.1)
  (h3 : c.marketer_salary = 50000)
  (h4 : c.engineer_salary = 80000)
  (h5 : c.overall_average = 80000) :
  manager_average_salary c = 185000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manager_salary_is_185000_l1016_101632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_dot_product_and_cosine_l1016_101614

-- Define the vectors
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the point M on the line OP
def M (t : ℝ) : ℝ × ℝ := (2*t, t)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem minimize_dot_product_and_cosine :
  ∃ (t : ℝ),
    (∀ (s : ℝ), dot_product (vec_sub (M t) OA) (vec_sub (M t) OB) ≤ dot_product (vec_sub (M s) OA) (vec_sub (M s) OB)) ∧
    M t = (4, 2) ∧
    (dot_product (vec_sub (M t) OA) (vec_sub (M t) OB) / (magnitude (vec_sub (M t) OA) * magnitude (vec_sub (M t) OB)) = -4 * Real.sqrt 17 / 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_dot_product_and_cosine_l1016_101614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_victory_distance_check_result_l1016_101660

/-- Calculates the distance by which runner A beats runner B in a race. -/
noncomputable def distance_a_beats_b (race_length : ℝ) (a_time : ℝ) (time_difference : ℝ) : ℝ :=
  (race_length / a_time) * time_difference

/-- Theorem: In a 1000-meter race, if runner A completes the race in 192 seconds
    and beats runner B by 8 seconds, then A beats B by approximately 41.67 meters. -/
theorem race_victory_distance :
  let race_length : ℝ := 1000
  let a_time : ℝ := 192
  let time_difference : ℝ := 8
  abs (distance_a_beats_b race_length a_time time_difference - 41.67) < 0.01 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use a theorem instead
theorem check_result :
  abs (distance_a_beats_b 1000 192 8 - 41.6666666666667) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_victory_distance_check_result_l1016_101660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_moving_circle_l1016_101608

/-- A moving circle C with center (a, b) and radius r -/
structure MovingCircle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The moving circle C satisfies the given conditions -/
def satisfiesConditions (c : MovingCircle) : Prop :=
  -- Passes through (1, 0)
  (c.a - 1)^2 + c.b^2 = c.r^2 ∧
  -- Tangent to x = -1
  c.a + 1 = c.r ∧
  -- Always has a common point with y = x + 2√2 + 1
  |c.a - c.b + 2 * Real.sqrt 2 + 1| / Real.sqrt 2 ≤ c.r

/-- The area of a circle -/
noncomputable def circleArea (c : MovingCircle) : ℝ := Real.pi * c.r^2

/-- The theorem to be proved -/
theorem min_area_of_moving_circle :
  ∃ (c : MovingCircle), satisfiesConditions c ∧
    (∀ (c' : MovingCircle), satisfiesConditions c' → circleArea c ≤ circleArea c') ∧
    circleArea c = 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_moving_circle_l1016_101608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_to_sin_shift_l1016_101610

theorem cos_to_sin_shift (x : ℝ) :
  Real.sin (2*x + π/6) = Real.cos (2*(x - π/12) - π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_to_sin_shift_l1016_101610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1016_101623

-- Define a right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  angleC : C = 90
  angleA : A = Real.pi / 2 - B  -- Since it's a right triangle, A + B + C = 180°
  sideBC : Real

-- Theorem statement
theorem right_triangle_hypotenuse (triangle : RightTriangle) (α : Real) (a : Real)
    (h1 : triangle.A = α) 
    (h2 : triangle.sideBC = a) : 
  a / Real.sin α = sorry := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1016_101623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_maximizing_angle_l1016_101652

noncomputable def g (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 13)

theorem smallest_maximizing_angle :
  ∃ (x : ℝ),
    x > 0 ∧
    x = 8190 * Real.pi / 180 ∧
    (∀ (y : ℝ), g y ≤ g x) ∧
    (∀ (z : ℝ), 0 < z ∧ z < x → ∃ (w : ℝ), g z < g w) := by
  sorry

#check smallest_maximizing_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_maximizing_angle_l1016_101652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1016_101684

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12)

theorem function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ), f (k * Real.pi / 2 + Real.pi / 6 - x) = f (k * Real.pi / 2 + Real.pi / 6 + x)) ∧
  (Set.Icc (-Real.sqrt 3) 2 = {y | ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ g x = y}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1016_101684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l1016_101658

/-- The function f(x) = (x^2 + 2x + 3) / (x - 5) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 3) / (x - 5)

/-- Theorem: The function f has a vertical asymptote at x = 5 -/
theorem vertical_asymptote_at_five :
  ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 5| ∧ |x - 5| < δ → |f x| > L :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l1016_101658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinematic_academy_size_l1016_101665

/-- The smallest number of top-10 lists a film can appear on and still be considered for "movie of the year" -/
noncomputable def min_lists : ℚ := 192.5

/-- The fraction of top-10 lists a film must appear in to be considered for "movie of the year" -/
def required_fraction : ℚ := 1/4

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 770

theorem cinematic_academy_size :
  academy_members = Int.floor (min_lists / required_fraction) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinematic_academy_size_l1016_101665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_l1016_101620

noncomputable section

-- Define the triangle PQR
def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (10, 0)

-- Define the function to calculate the area of triangle PVW
noncomputable def area_PVW (s : ℝ) : ℝ :=
  (2/5) * (10 - s)^2

-- State the theorem
theorem horizontal_line_intersection (s : ℝ) :
  area_PVW s = 20 → s = 10 - 5 * Real.sqrt 2 :=
by
  intro h
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_intersection_l1016_101620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_to_base_area_ratio_is_sqrt_3_l1016_101609

/-- A regular triangular pyramid with a 90° plane angle at its vertex -/
structure RegularTriangularPyramid where
  /-- The side length of the base triangle -/
  base_side : ℝ
  /-- The base side is positive -/
  base_side_pos : base_side > 0
  /-- The plane angle at the vertex is 90° -/
  vertex_angle : Real.cos (π / 4) = 1 / Real.sqrt 2

/-- The ratio of the lateral surface area to the base area of a regular triangular pyramid -/
noncomputable def lateral_to_base_area_ratio (p : RegularTriangularPyramid) : ℝ :=
  3 / Real.sqrt 3

/-- 
Theorem: For a regular triangular pyramid with a 90° plane angle at its vertex,
the ratio of its lateral surface area to the area of its base is √3.
-/
theorem lateral_to_base_area_ratio_is_sqrt_3 (p : RegularTriangularPyramid) :
  lateral_to_base_area_ratio p = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_to_base_area_ratio_is_sqrt_3_l1016_101609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l1016_101690

/-- A right prism with a rhombic base -/
structure RhombicPrism where
  height : ℝ
  base_side : ℝ
  base_angle : ℝ

/-- The intersecting plane -/
structure IntersectingPlane where
  angle_with_base : ℝ

/-- The cross-section formed by the intersecting plane -/
noncomputable def cross_section_area (prism : RhombicPrism) (plane : IntersectingPlane) : ℝ :=
  4 / Real.sqrt 3

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_theorem (prism : RhombicPrism) (plane : IntersectingPlane) :
    prism.height = 1 →
    prism.base_side = 2 →
    prism.base_angle = π/6 →
    plane.angle_with_base = π/3 →
    cross_section_area prism plane = 4 / Real.sqrt 3 := by
  sorry

#check cross_section_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l1016_101690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l1016_101656

theorem power_equation (a b : ℝ) (h1 : (10 : ℝ)^a = 2) (h2 : (10 : ℝ)^b = 6) : 
  (10 : ℝ)^(2*a - 3*b) = 1/54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l1016_101656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_visible_same_color_l1016_101671

/-- Represents the color of a face -/
inductive Color
| Red
| Blue

/-- Represents a cube with 6 faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Defines a cube with one face fixed as blue -/
def cube_with_fixed_blue : Cube :=
  { faces := λ i => if i = 0 then Color.Blue else Color.Blue }  -- Face 0 is fixed blue

/-- Probability of a face being blue (excluding the fixed face) -/
noncomputable def prob_blue : ℝ := 1 / 2

/-- Checks if all visible vertical faces are the same color when placed on a given face -/
def all_visible_same_color (c : Cube) (bottom : Fin 6) : Prop :=
  ∃ color, ∀ i : Fin 6, i ≠ bottom ∧ i ≠ (5 - bottom) → c.faces i = color

/-- The main theorem to prove -/
theorem prob_all_visible_same_color :
  (1 / 32) * (1 + 1 + 4) = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_visible_same_color_l1016_101671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_tan_inverse_l1016_101663

/-- Given an angle x such that tan x = a/(2b) and tan 2x = 2b/(a + 2b),
    the least positive value of k where x = tan⁻¹ k is 0. -/
theorem least_positive_tan_inverse (a b x : ℝ) (h1 : Real.tan x = a / (2 * b)) 
  (h2 : Real.tan (2 * x) = (2 * b) / (a + 2 * b)) : 
  ∃ k : ℝ, k = 0 ∧ x = Real.arctan k ∧ ∀ k' : ℝ, k' > 0 → Real.arctan k' ≥ x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_tan_inverse_l1016_101663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1016_101657

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sqrt 3 * Real.cos C - Real.sin C = (Real.sqrt 3 * b) / a →
  b + c = 6 →
  D = ((b * Real.cos A, b * Real.sin A) + (c, 0)) / 2 →
  Real.sqrt ((D.1 - Real.cos A)^2 + (D.2 - Real.sin A)^2) = 2 * Real.sqrt 2 →
  A = 2 * π / 3 ∧ 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1016_101657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l1016_101645

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles in radians
  (a b c : ℝ)  -- Side lengths

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ 
  t.b = 2 * Real.sqrt 3 ∧ 
  t.A = Real.pi / 6  -- 30° in radians

-- State the theorem
theorem angle_B_value (t : Triangle) :
  problem_conditions t → t.B = Real.pi / 3 ∨ t.B = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l1016_101645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_of_cos_two_theta_l1016_101640

theorem sin_plus_cos_of_cos_two_theta (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < π / 2) -- θ is acute
  (h2 : Real.cos (2 * θ) = b) : 
  Real.sin θ + Real.cos θ = Real.sqrt (1 + Real.sqrt (1 - b^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_of_cos_two_theta_l1016_101640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_special_triangle_l1016_101635

/-- Given a triangle ABC where b = 5/8 * a and A = 2B, prove that cos A = 7/25 -/
theorem cosine_in_special_triangle (a b c : ℝ) (A B C : Real) :
  b = (5/8) * a →
  A = 2 * B →
  Real.cos A = 7/25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_in_special_triangle_l1016_101635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_l1016_101618

-- Define the function f(x) = a^x + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + 1

-- State the theorem
theorem function_passes_through_point
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify a^0 to 1
  simp [Real.rpow_zero]
  -- Evaluate 1 + 1
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_l1016_101618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equations_l1016_101664

-- Define the ellipse structure
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

-- Define the standard form of an ellipse equation
def StandardForm (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the conditions given in the problem
class EllipseConditions (e : Ellipse) where
  major_axis_triple : e.a = 3 * e.b
  passes_through_A : StandardForm e 3 0
  equilateral_triangle : ∃ (c : ℝ), c^2 = e.a^2 - e.b^2 ∧ e.b = c * Real.sqrt 3
  focus_to_vertex : ∃ (c : ℝ), c^2 = e.a^2 - e.b^2 ∧ e.a - c = Real.sqrt 3

-- Theorem statement
theorem ellipse_equations (e : Ellipse) [EllipseConditions e] :
  ((e.a = 3 ∧ e.b = 1) ∨ (e.a = 9 ∧ e.b = 3)) ∧
  ((e.a = 2 * Real.sqrt 3 ∧ e.b = 3) ∨ (e.a = 3 ∧ e.b = 2 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equations_l1016_101664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_ABD_collinear_l1016_101641

/-- Two non-zero vectors are not collinear if they are linearly independent -/
def not_collinear (e₁ e₂ : ℝ × ℝ) : Prop :=
  e₁ ≠ (0, 0) ∧ e₂ ≠ (0, 0) ∧ ¬∃ (k : ℝ), e₁ = k • e₂

/-- Define a point as a pair of real numbers -/
def Point := ℝ × ℝ

/-- Three points are collinear if the vector from the first to the second
    is a scalar multiple of the vector from the first to the third -/
def collinear (A B D : Point) : Prop :=
  ∃ (k : ℝ), (B.1 - A.1, B.2 - A.2) = k • (D.1 - A.1, D.2 - A.2)

theorem points_ABD_collinear
  (e₁ e₂ : ℝ × ℝ)
  (h_not_collinear : not_collinear e₁ e₂)
  (A B C D : Point)
  (h_AB : (B.1 - A.1, B.2 - A.2) = e₁ + 2 • e₂)
  (h_BC : (C.1 - B.1, C.2 - B.2) = 2 • e₁ + 7 • e₂)
  (h_CD : (D.1 - C.1, D.2 - C.2) = 3 • (e₁ + e₂)) :
  collinear A B D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_ABD_collinear_l1016_101641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_40_minutes_l1016_101637

/-- The distance between two runners after running in opposite directions for a given time -/
noncomputable def distance_between_runners (joe_speed : ℝ) (time : ℝ) : ℝ :=
  let pete_speed := joe_speed / 2
  (joe_speed * time) + (pete_speed * time)

/-- Theorem stating the distance between Joe and Pete after running for 40 minutes -/
theorem distance_after_40_minutes :
  distance_between_runners 0.266666666667 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_40_minutes_l1016_101637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_symmetry_l1016_101698

/-- The polar curve ρ = 4sin(θ - π/3) is symmetric about the line θ = 5π/6 -/
theorem polar_curve_symmetry :
  ∀ (ρ θ : ℝ), 
  ρ = 4 * Real.sin (θ - π/3) → 
  ∃ (ρ' θ' : ℝ), 
    ρ' = 4 * Real.sin (θ' - π/3) ∧ 
    ρ = ρ' ∧ 
    θ + θ' = 5*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_symmetry_l1016_101698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1016_101624

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point A
def A : ℝ × ℝ := (1, 4)

-- Define the distance from a point to the directrix
noncomputable def distance_to_directrix (P : ℝ × ℝ) : ℝ :=
  abs (P.1 + 1/4)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (P : ℝ × ℝ), parabola P.1 P.2 →
    distance_to_directrix P + distance P A ≥ min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1016_101624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_proposition_l1016_101654

def proposition_p : Prop := ∀ x y : ℝ, Real.sin x > Real.sin y → x > y

def proposition_q : Prop := ∀ x y : ℝ, x^2 + y^2 ≥ 2*x*y

theorem false_proposition :
  (¬proposition_p) ∧ proposition_q → ¬(proposition_p ∧ proposition_q) :=
by
  intro h
  intro h_p_and_q
  cases h with
  | intro h_not_p h_q =>
    have h_p := h_p_and_q.left
    exact h_not_p h_p

-- The proof is complete, but we can add more explanation if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_proposition_l1016_101654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_sum_equals_2034144_l1016_101662

open BigOperators

def expression (n : ℕ) : ℚ :=
  (1 / (2 * n - 1) - 1 / (2 * n + 1)) /
  (1 / (2 * n - 1) * 1 / (2 * n) * 1 / (2 * n + 1))

theorem expression_sum_equals_2034144 :
  ∑ n in Finset.range 1008, expression (n + 1) = 2034144 := by
  sorry

#eval ∑ n in Finset.range 1008, expression (n + 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_sum_equals_2034144_l1016_101662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l1016_101644

/-- The radius of each circle -/
noncomputable def radius : ℝ := 5

/-- The number of circles -/
def num_circles : ℕ := 6

/-- The number of circles covering each shaded region -/
def circles_per_region : ℕ := 3

/-- The area of the region covered by exactly three out of six circles of radius 5 units intersecting at the origin -/
noncomputable def shaded_area : ℝ := 37.5 * Real.pi - (25 * Real.sqrt 3) / 2

theorem shaded_area_proof :
  let total_area := num_circles * (
    (circles_per_region / num_circles : ℝ) * (π * radius^2 / 4) - 
    (radius^2 * Real.sqrt 3 / 4) / 3
  )
  total_area = shaded_area := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l1016_101644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1016_101653

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a) / x^2

-- Part 1: Prove that if f is even, then a = -1
theorem part1 (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by sorry

-- Define the simplified function g
noncomputable def g (x : ℝ) : ℝ := 1 - 1 / x^2

-- Part 2: Prove that g is decreasing on (-∞, 0)
theorem part2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 0 → g x₁ > g x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1016_101653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_construction_optimization_l1016_101678

/-- Cost function for bridge construction -/
noncomputable def cost_function (m : ℝ) (x : ℝ) : ℝ :=
  256 * (m / x - 1) + (2 + Real.sqrt x) * x * (m / x)

/-- Number of new piers needed -/
noncomputable def num_new_piers (m : ℝ) (x : ℝ) : ℝ :=
  m / x - 1

theorem bridge_construction_optimization :
  let m : ℝ := 640
  let optimal_x : ℝ := 64
  (∀ x : ℝ, 0 < x → x < m →
    cost_function m x ≥ cost_function m optimal_x) ∧
  num_new_piers m optimal_x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_construction_optimization_l1016_101678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_14_l1016_101606

-- Define the polynomial (x^2 + 1)(2x + 1)^3
def f (x : ℝ) : ℝ := (x^2 + 1) * (2*x + 1)^3

-- Theorem stating that the coefficient of x^3 in the expansion of f(x) is 14
theorem coefficient_of_x_cubed_is_14 :
  ∃ c₀ c₁ c₂ c₃ c₄ c₅ : ℝ,
    ∀ x, f x = c₀ + c₁*x + c₂*x^2 + c₃*x^3 + c₄*x^4 + c₅*x^5 ∧ c₃ = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_is_14_l1016_101606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_miscellaneous_expenses_l1016_101622

def monthly_salary (savings : ℚ) (savings_percentage : ℚ) : ℚ :=
  savings / savings_percentage

def total_expenses (rent milk groceries education petrol : ℚ) : ℚ :=
  rent + milk + groceries + education + petrol

def miscellaneous_expenses (salary expenses savings : ℚ) : ℚ :=
  salary - (expenses + savings)

theorem kishore_miscellaneous_expenses 
  (rent milk groceries education petrol : ℚ)
  (savings : ℚ)
  (savings_percentage : ℚ)
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : petrol = 2000)
  (h6 : savings = 2160)
  (h7 : savings_percentage = 1/10) :
  miscellaneous_expenses 
    (monthly_salary savings savings_percentage)
    (total_expenses rent milk groceries education petrol)
    savings
  = 3940 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_miscellaneous_expenses_l1016_101622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quality_related_to_renovation_distribution_and_mean_X_l1016_101651

-- Define the sample data
def before_first_class : ℕ := 120
def before_second_class : ℕ := 80
def after_first_class : ℕ := 150
def after_second_class : ℕ := 50
def total_sample : ℕ := 400

-- Define the chi-square test statistic function
def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the significance threshold
def significance_threshold : ℚ := 6635 / 1000

-- Define the theorem for the chi-square test
theorem quality_related_to_renovation :
  chi_square total_sample before_first_class after_second_class after_first_class before_second_class
  > significance_threshold := by sorry

-- Define the distribution of X
def distribution_X (x : ℕ) : ℚ :=
  match x with
  | 1 => 3 / 10
  | 2 => 6 / 10
  | 3 => 1 / 10
  | _ => 0

-- Define the expected value of X
def E_X : ℚ := 9 / 5

-- Define the theorem for the distribution and expected value
theorem distribution_and_mean_X :
  (∀ x, x ∈ ({1, 2, 3} : Set ℕ) → distribution_X x = (Nat.choose 3 x * Nat.choose 2 (3 - x)) / Nat.choose 5 3) ∧
  E_X = 1 * distribution_X 1 + 2 * distribution_X 2 + 3 * distribution_X 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quality_related_to_renovation_distribution_and_mean_X_l1016_101651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_numbers_sum_bound_l1016_101695

def CircleNumbers := List Int

def validCircleNumbers (nums : List Int) : Prop :=
  nums.length = 2002 ∧ ∀ n, n ∈ nums → (n = 1 ∨ n = -1)

def adjacentProducts (nums : List Int) : List Int :=
  List.zipWith (· * ·) nums (nums.drop 1 ++ nums.take 1)

theorem circle_numbers_sum_bound (nums : List Int) 
  (h_valid : validCircleNumbers nums) 
  (h_sum_neg : (adjacentProducts nums).sum < 0) : 
  abs (nums.sum) ≤ 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_numbers_sum_bound_l1016_101695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_won_three_tickets_l1016_101691

/-- The number of tickets Edward won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := sorry

/-- The number of tickets Edward won playing 'skee ball' -/
def skee_ball_tickets : ℕ := 5

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 4

/-- The number of candies Edward could buy -/
def candies_bought : ℕ := 2

/-- Theorem stating that Edward won 3 tickets playing 'whack a mole' -/
theorem edward_won_three_tickets : whack_a_mole_tickets = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_won_three_tickets_l1016_101691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_bought_six_kilos_potatoes_l1016_101636

/-- Represents Peter's shopping trip to the market -/
structure MarketTrip where
  initialMoney : ℚ
  potatoPrice : ℚ
  tomatoAmount : ℚ
  tomatoPrice : ℚ
  cucumberAmount : ℚ
  cucumberPrice : ℚ
  bananaAmount : ℚ
  bananaPrice : ℚ
  remainingMoney : ℚ

/-- Calculates the number of kilos of potatoes bought -/
def potatoesKilos (trip : MarketTrip) : ℚ :=
  let otherExpenses := trip.tomatoAmount * trip.tomatoPrice +
                       trip.cucumberAmount * trip.cucumberPrice +
                       trip.bananaAmount * trip.bananaPrice
  let totalSpent := trip.initialMoney - trip.remainingMoney
  let potatoExpense := totalSpent - otherExpenses
  potatoExpense / trip.potatoPrice

/-- Theorem stating that Peter bought 6 kilos of potatoes -/
theorem peter_bought_six_kilos_potatoes (trip : MarketTrip)
  (h1 : trip.initialMoney = 500)
  (h2 : trip.potatoPrice = 2)
  (h3 : trip.tomatoAmount = 9)
  (h4 : trip.tomatoPrice = 3)
  (h5 : trip.cucumberAmount = 5)
  (h6 : trip.cucumberPrice = 4)
  (h7 : trip.bananaAmount = 3)
  (h8 : trip.bananaPrice = 5)
  (h9 : trip.remainingMoney = 426) :
  potatoesKilos trip = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_bought_six_kilos_potatoes_l1016_101636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_inverse_proportion_to_area_pressure_direct_proportion_to_force_only_one_conclusion_correct_l1016_101642

-- Define pressure as a function of force and area
noncomputable def pressure (F S : ℝ) : ℝ := F / S

-- Theorem: When force is constant, pressure is inversely proportional to area
theorem pressure_inverse_proportion_to_area (k F₁ F₂ S₁ S₂ : ℝ) 
  (h_pos : k > 0) (h_F : F₁ = F₂) (h_S : S₁ ≠ 0 ∧ S₂ ≠ 0) :
  pressure F₁ S₁ / pressure F₂ S₂ = S₂ / S₁ := by
  sorry

-- Theorem: When area is constant, pressure is directly proportional to force
theorem pressure_direct_proportion_to_force (k F₁ F₂ S : ℝ)
  (h_pos : k > 0) (h_S : S ≠ 0) :
  pressure F₁ S / pressure F₂ S = F₁ / F₂ := by
  sorry

-- Theorem: Only one conclusion is correct
theorem only_one_conclusion_correct : True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_inverse_proportion_to_area_pressure_direct_proportion_to_force_only_one_conclusion_correct_l1016_101642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_greater_than_two_l1016_101612

open Real

theorem log_sum_greater_than_two (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ > 0)
  (h2 : x₂ > 0)
  (h3 : x₁ ≠ x₂)
  (h4 : 2 * log x₁ - a * x₁ = 0)
  (h5 : 2 * log x₂ - a * x₂ = 0) :
  log x₁ + log x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_greater_than_two_l1016_101612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_area_l1016_101630

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Theorem: The area of triangle PQR with given coordinates is 16 square units -/
theorem triangle_pqr_area :
  triangleArea (-3) 2 1 7 3 (-1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_area_l1016_101630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_values_l1016_101648

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem: Given an ellipse with equation x^2 / (10-a) + y^2 / (a-2) = 1 and eccentricity √2 / 2,
    the possible values of a are 14/3 or 22/3 -/
theorem ellipse_a_values (a : ℝ) :
  (∃ e : Ellipse, (e.a = Real.sqrt (10 - a) ∧ e.b = Real.sqrt (a - 2) ∧ eccentricity e = Real.sqrt 2 / 2) ∨
                  (e.a = Real.sqrt (a - 2) ∧ e.b = Real.sqrt (10 - a) ∧ eccentricity e = Real.sqrt 2 / 2)) →
  a = 14/3 ∨ a = 22/3 := by
  sorry

#check ellipse_a_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_values_l1016_101648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_13_same_digits_l1016_101602

/-- Returns true if all digits of a natural number are the same -/
def allDigitsSame (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, k < (Nat.digits 10 n).length → (Nat.digits 10 n).get ⟨k, by sorry⟩ = d

/-- The least positive integer that is a multiple of 13 and has all digits the same -/
def leastMultiple13SameDigits : ℕ := 111111

theorem least_multiple_13_same_digits :
  (leastMultiple13SameDigits % 13 = 0) ∧
  allDigitsSame leastMultiple13SameDigits ∧
  ∀ n : ℕ, n < leastMultiple13SameDigits →
    ¬(n % 13 = 0 ∧ allDigitsSame n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_13_same_digits_l1016_101602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1016_101604

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), line_eq x y ∧ circle_eq x y → 
      ∃ (x' y' : ℝ), line_eq x' y' ∧ circle_eq x' y' ∧ 
        (x - x')^2 + (y - y')^2 = chord_length^2) ∧
    chord_length = Real.sqrt 35 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1016_101604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l1016_101617

def t : ℕ → ℚ
  | 0 => 2  -- Add a case for 0
  | 1 => 2
  | n+2 => if n % 2 = 0 then 2 + t ((n+2)/2) else 2 / t (n+1)

theorem sequence_value (n : ℕ) : t n = 29/131 → n = 193 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l1016_101617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MON_is_90_degrees_l1016_101643

-- Define the circle equation
def circle_equation (x y D : ℝ) : Prop :=
  x^2 + y^2 + D*x - 4*y = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 2

-- Define the intersection points
def intersection_points (M N : ℝ × ℝ) (D : ℝ) : Prop :=
  circle_equation M.1 M.2 D ∧ 
  circle_equation N.1 N.2 D ∧ 
  line_equation M.2 ∧ 
  line_equation N.2

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the angle between three points
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_MON_is_90_degrees 
  (M N : ℝ × ℝ) 
  (h_intersection : intersection_points M N 0) :
  let O := origin
  angle O M N = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_MON_is_90_degrees_l1016_101643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1016_101679

/-- Triangle with vertices A(1, 2), B(2, 5), and C(3, 4) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Cosine of the angle between two vectors -/
noncomputable def cos_angle (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))

def triangle : Triangle :=
  { A := (1, 2)
    B := (2, 5)
    C := (3, 4) }

theorem triangle_properties (t : Triangle) (h : t = triangle) :
  let AB := distance t.A t.B
  let AC := distance t.A t.C
  let BC := distance t.B t.C
  let cos_A := cos_angle (t.B.1 - t.A.1, t.B.2 - t.A.2) (t.C.1 - t.A.1, t.C.2 - t.A.2)
  AB = Real.sqrt 10 ∧ 
  AC = 2 * Real.sqrt 2 ∧
  BC = Real.sqrt 2 ∧
  AC^2 + BC^2 = AB^2 ∧
  cos_A = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1016_101679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_condition_no_nonpositive_value_l1016_101621

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * (2*a + 1) * x^2 - 2*(a + 1) * x

-- Theorem 1: f has a local maximum at x = 1 iff a < -3/2
theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a x ≤ f a 1) ↔ a < -3/2 := by
  sorry

-- Theorem 2: There is no a such that f(x) ≤ 0 for some x ∈ [1, 2]
theorem no_nonpositive_value (a : ℝ) :
  ¬∃ x ∈ Set.Icc 1 2, f a x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_condition_no_nonpositive_value_l1016_101621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_shape_l1016_101668

-- Define the square PQRS
def PQRS : Set (ℝ × ℝ) := sorry

-- Define the four smaller squares inside PQRS
def smallSquare1 : Set (ℝ × ℝ) := sorry
def smallSquare2 : Set (ℝ × ℝ) := sorry
def smallSquare3 : Set (ℝ × ℝ) := sorry
def smallSquare4 : Set (ℝ × ℝ) := sorry

-- Define the T-shaped region
def TShape : Set (ℝ × ℝ) := sorry

-- Define the side lengths of the smaller squares
def sideLength1 : ℝ := 2
def sideLength2 : ℝ := 2
def sideLength3 : ℝ := 2
def sideLength4 : ℝ := 2

-- Define the lengths of the T-shape
def verticalLength : ℝ := 6
def horizontalLength : ℝ := 6

-- Theorem: The area of the T-shaped region is 20
theorem area_of_T_shape : MeasureTheory.volume TShape = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_shape_l1016_101668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_PAB_l1016_101667

-- Define the points and circle
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 16 = 0

-- Define the line l passing through A
def line_l (x y : ℝ) : Prop := (3*x - 4*y - 3 = 0) ∨ (x = 1)

-- Define the chord length condition
def chord_length_condition (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    l x₁ y₁ ∧ l x₂ y₂ ∧ 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 20

-- Helper function for area calculation
noncomputable def area_triangle (A B P : ℝ × ℝ) : ℝ := 
  abs ((A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2)) / 2)

-- Theorem statement
theorem max_area_PAB :
  chord_length_condition line_l →
  (∀ P : ℝ × ℝ, circle_equation P.1 P.2 → 
    2 * (area_triangle A B P) ≤ 14) ∧
  (∃ P : ℝ × ℝ, circle_equation P.1 P.2 ∧ 
    2 * (area_triangle A B P) = 14) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_PAB_l1016_101667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_value_l1016_101625

/-- The value of the infinite nested square root √(3 - √(3 - √(3 - ...))) -/
noncomputable def nested_sqrt : ℝ := 
  (-1 + Real.sqrt 13) / 2

/-- Theorem stating that the nested square root equals (-1 + √13) / 2 -/
theorem nested_sqrt_value : 
  nested_sqrt = Real.sqrt (3 - nested_sqrt) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_value_l1016_101625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l1016_101689

/-- Given two circles C1 and C2, and a moving circle C that is externally tangent to C1 and internally tangent to C2,
    this theorem proves that the trajectory of C's center is an ellipse and that a line intersecting this trajectory
    passes through a fixed point under certain conditions. -/
theorem trajectory_and_fixed_point 
  (C1 : ℝ → ℝ → Prop) 
  (C2 : ℝ → ℝ → Prop)
  (C : ℝ → ℝ → Prop)
  (T : ℝ → ℝ → Prop)
  (l : ℝ → ℝ)
  (k m : ℝ)
  (h1 : ∀ x y, C1 x y ↔ (x + 1)^2 + y^2 = 1)
  (h2 : ∀ x y, C2 x y ↔ (x - 1)^2 + y^2 = 9)
  (h3 : ∀ x y, C x y → (∃ r : ℝ, r > 0 ∧ 
               (∀ x' y', C1 x' y' → ((x - x')^2 + (y - y')^2 = (r + 1)^2)) ∧
               (∀ x' y', C2 x' y' → ((x - x')^2 + (y - y')^2 = (3 - r)^2))))
  (h4 : ∀ x y, T x y ↔ (∃ t : ℝ, C x y ∧ t ∈ Set.Icc 0 1))
  (h5 : ∀ x, l x = k * x + m)
  (h6 : ∃ M N : ℝ × ℝ, T M.1 M.2 ∧ T N.1 N.2 ∧ l M.1 = M.2 ∧ l N.1 = N.2 ∧ M.2 ≠ 0 ∧ N.2 ≠ 0)
  (h7 : ∃ A : ℝ × ℝ, T A.1 A.2 ∧ A.2 = 0 ∧ A.1 > 0)
  (h8 : ∀ M N : ℝ × ℝ, T M.1 M.2 → T N.1 N.2 → l M.1 = M.2 → l N.1 = N.2 → M.2 ≠ 0 → N.2 ≠ 0 →
        ∃ A : ℝ × ℝ, T A.1 A.2 ∧ A.2 = 0 ∧ A.1 > 0 ∧
        (M.1 - A.1) * (N.1 - A.1) + M.2 * N.2 = 0) :
  (∀ x y, T x y ↔ x^2/4 + y^2/3 = 1) ∧ 
  (∀ x, l x = k * (x - 2/7)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l1016_101689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_sum_l1016_101680

noncomputable section

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := 12 * x^2
def parabola2 (x : ℝ) : ℝ := x^2 / 5 + 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def V : ℝ × ℝ := (0, 1)

-- Define the triangles
structure RightIsoscelesTriangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

-- Define the specific triangles
def triangle_PQO : RightIsoscelesTriangle := sorry
def triangle_ABV : RightIsoscelesTriangle := sorry

-- Define the properties of the triangles
def is_inscribed_in_parabola1 (t : RightIsoscelesTriangle) : Prop := sorry
def is_inscribed_in_parabola2 (t : RightIsoscelesTriangle) : Prop := sorry
def has_right_angle_at_second_point (t : RightIsoscelesTriangle) : Prop := sorry
def first_point_in_first_quadrant (t : RightIsoscelesTriangle) : Prop := sorry
def second_point_in_second_quadrant (t : RightIsoscelesTriangle) : Prop := sorry

-- Define the relationship between A and Q
def A_y_coordinate (q : ℝ) : ℝ := sorry

-- Main theorem
theorem parabola_triangle_sum :
  ∃ (u v w : ℤ),
    is_inscribed_in_parabola1 triangle_PQO ∧
    is_inscribed_in_parabola2 triangle_ABV ∧
    has_right_angle_at_second_point triangle_PQO ∧
    has_right_angle_at_second_point triangle_ABV ∧
    first_point_in_first_quadrant triangle_PQO ∧
    first_point_in_first_quadrant triangle_ABV ∧
    second_point_in_second_quadrant triangle_PQO ∧
    second_point_in_second_quadrant triangle_ABV ∧
    (∃ (q : ℝ), A_y_coordinate q = u * q^2 + v * q + w) ∧
    u + v + w = 781 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_sum_l1016_101680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_phone_numbers_l1016_101619

/-- Represents a button on the phone keypad -/
structure Button where
  row : Nat
  col : Nat
  value : Nat

/-- Represents the phone keypad layout -/
def keypad : List Button := [
  ⟨1, 1, 1⟩, ⟨1, 2, 2⟩, ⟨1, 3, 3⟩,
  ⟨2, 1, 4⟩, ⟨2, 2, 5⟩, ⟨2, 3, 6⟩,
  ⟨3, 1, 7⟩, ⟨3, 2, 8⟩, ⟨3, 3, 9⟩,
  ⟨4, 1, 0⟩
]

/-- Checks if four buttons form a square on the keypad -/
def formSquare (b1 b2 b3 b4 : Button) : Prop := sorry

/-- Checks if a list of digits is in ascending order -/
def isAscending (digits : List Nat) : Prop := sorry

/-- Checks if all digits in a list are unique -/
def allUnique (digits : List Nat) : Prop := sorry

/-- Checks if a number is divisible by both 3 and 5 -/
def divisibleBy3And5 (n : Nat) : Prop := sorry

/-- Converts a list of digits to a natural number -/
def digitsToNat (digits : List Nat) : Nat := sorry

/-- The main theorem -/
theorem count_valid_phone_numbers :
  ∃ (validNumbers : List (List Nat)),
    (∀ n, n ∈ validNumbers →
      n.length = 9 ∧
      allUnique n ∧
      isAscending (n.take 4) ∧
      (∃ b1 b2 b3 b4, b1 ∈ keypad ∧ b2 ∈ keypad ∧ b3 ∈ keypad ∧ b4 ∈ keypad ∧ 
        formSquare b1 b2 b3 b4 ∧ n.take 4 = [b1.value, b2.value, b3.value, b4.value]) ∧
      (∃ b1 b2 b3 b4, b1 ∈ keypad ∧ b2 ∈ keypad ∧ b3 ∈ keypad ∧ b4 ∈ keypad ∧ 
        formSquare b1 b2 b3 b4 ∧ n.drop 5 = [b1.value, b2.value, b3.value, b4.value]) ∧
      divisibleBy3And5 (digitsToNat n)) ∧
    validNumbers.length = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_phone_numbers_l1016_101619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_pillar_activation_l1016_101699

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of n points in a plane -/
def PointConfiguration (n : ℕ) := Fin n → Point

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- A directed cycle in the point configuration -/
structure DirectedCycle (n : ℕ) where
  cycle : List (Fin n)
  cyclic : cycle.length > 0 ∧ cycle.head? = cycle.getLast?

/-- The main theorem -/
theorem energy_pillar_activation (n : ℕ) (h : n ≥ 3) 
  (config : PointConfiguration n) 
  (not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬areCollinear (config i) (config j) (config k)) :
  (∃ cycle : DirectedCycle n, True) ∧ 
  (∃ cycles : Finset (DirectedCycle n), cycles.card ≤ 2 * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_pillar_activation_l1016_101699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientists_with_odd_handshakes_is_even_l1016_101670

/-- 
Given a group of scientists shaking hands, the number of scientists 
who have shaken an odd number of hands is always even.
-/
theorem scientists_with_odd_handshakes_is_even (n : ℕ) (handshakes : Fin n → ℕ) : 
  Even (Finset.card (Finset.filter (fun i => Odd (handshakes i)) (Finset.univ : Finset (Fin n)))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientists_with_odd_handshakes_is_even_l1016_101670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1016_101631

noncomputable def f (x : ℝ) : ℝ := -x + 1/x

theorem max_value_of_f : 
  let a : ℝ := -2
  let b : ℝ := -1/3
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c ∧ f c = 3/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1016_101631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_painting_time_l1016_101627

/-- Represents the dimensions and features of a room to be painted -/
structure Room where
  length : ℚ
  width : ℚ
  height : ℚ
  long_wall_window_area : ℚ
  short_wall_windows_area : ℚ
  door_area : ℚ

/-- Calculates the time needed to paint a room given the room's dimensions and painting rate -/
def time_to_paint (room : Room) (painting_rate : ℚ) (coats : ℕ) : ℚ :=
  let total_wall_area := 2 * (room.length * room.height + room.width * room.height)
  let paintable_area := total_wall_area - (room.long_wall_window_area + room.short_wall_windows_area + room.door_area)
  (paintable_area * coats) / painting_rate

/-- Theorem stating that it takes Martha 37.5 hours to paint the kitchen -/
theorem martha_painting_time :
  let kitchen : Room := {
    length := 16,
    width := 12,
    height := 10,
    long_wall_window_area := 3 * 5,
    short_wall_windows_area := 2 * 2 * 6,
    door_area := 3 * 7
  }
  time_to_paint kitchen 40 3 = 75/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_painting_time_l1016_101627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l1016_101600

/-- Two lines are perpendicular if and only if the product of their slopes is -1 --/
axiom perpendicular_lines_slope_product {m1 m2 : ℝ} : 
  m1 * m2 = -1 ↔ (∃ (a b c d e f : ℝ), (∀ x y : ℝ, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0 ∧ 
    m1 = -a / b ∧ m2 = -d / e) → (a * x + b * y + c = 0 → d * x + e * y + f ≠ 0)))

/-- The theorem to be proved --/
theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ (x y : ℝ), a * x + 2 * y + 1 = 0 → x + y - 2 ≠ 0) → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l1016_101600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1016_101674

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (2 + i) / (1 + i)

theorem imaginary_part_of_z :
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1016_101674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_PQR_is_five_l1016_101616

-- Define the square pyramid
structure SquarePyramid where
  baseEdge : ℝ
  height : ℝ

-- Define a point on an edge of the pyramid
structure EdgePoint where
  edge : ℝ → ℝ  -- Function representing the edge
  t : ℝ         -- Parameter (0 ≤ t ≤ 1) representing position on edge

-- Define the specific pyramid and points
noncomputable def pyramid : SquarePyramid := ⟨4, 8⟩
noncomputable def pointP : EdgePoint := ⟨(λ t => t * pyramid.height), 1/4⟩
noncomputable def pointQ : EdgePoint := ⟨(λ t => t * pyramid.height), 1/4⟩
noncomputable def pointR : EdgePoint := ⟨(λ t => t * pyramid.height), 3/4⟩

-- Define a function to calculate the area of a triangle (placeholder)
noncomputable def areaOfTriangle (p q r : EdgePoint) : ℝ := sorry

-- Theorem statement
theorem area_of_PQR_is_five :
  let pqr_area := areaOfTriangle pointP pointQ pointR
  pqr_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_PQR_is_five_l1016_101616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_initial_seashells_l1016_101666

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of additional seashells Mike found later -/
def additional_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := 10.0

/-- Theorem stating that Mike initially found 6.0 seashells -/
theorem mike_initial_seashells :
  initial_seashells = 6.0 :=
by
  rfl

/-- Lemma showing the relationship between initial, additional, and total seashells -/
lemma seashell_equation :
  initial_seashells + additional_seashells = total_seashells :=
by
  simp [initial_seashells, additional_seashells, total_seashells]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_initial_seashells_l1016_101666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1016_101669

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := x^2 - 4

-- Theorem statement
theorem f_properties :
  (f 2 = -4/3 ∧ f' 2 = 0) ∧
  (∀ x : ℝ, -2 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x : ℝ, (x < -2 ∨ 2 < x) → f' x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1016_101669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_term_l1016_101647

theorem coefficient_of_x_term (x : ℝ) (h : x > 0) : 
  let expansion := (x - x^(-(1/2:ℝ)))^7
  ∃ (a b c : ℝ), expansion = a * x^2 + 35 * x + b * x^(-(1/2:ℝ)) + c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_term_l1016_101647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_matching_iff_second_player_wins_l1016_101682

/-- The second player has a winning strategy in the game. -/
def SecondPlayerWins {V : Type} (G : SimpleGraph V) :=
  ∀ (v₁ : V), ∃ (v₂ : V), G.Adj v₁ v₂ ∧
    ∀ (v₃ : V), G.Adj v₂ v₃ →
      ∃ (v₄ : V), G.Adj v₃ v₄ ∧
        ∀ (v₅ : V), G.Adj v₄ v₅ →
          ∃ (v₆ : V), G.Adj v₅ v₆ ∧ 
            -- Continue this pattern indefinitely
          True

/-- A perfect matching in a graph. -/
def PerfectMatching {V : Type} (G : SimpleGraph V) (N : ℕ) [Fintype V] :=
  ∃ (M : Finset (V × V)), 
    (∀ (e₁ e₂ : V × V), e₁ ∈ M → e₂ ∈ M → e₁ ≠ e₂ → e₁.1 ≠ e₂.1 ∧ e₁.1 ≠ e₂.2 ∧ e₁.2 ≠ e₂.1 ∧ e₁.2 ≠ e₂.2) ∧
    (∀ (e : V × V), e ∈ M → G.Adj e.1 e.2) ∧
    (M.card = N) ∧
    (∀ v : V, ∃ e ∈ M, v = e.1 ∨ v = e.2)

theorem perfect_matching_iff_second_player_wins 
  {V : Type} [Fintype V] (G : SimpleGraph V) (N : ℕ) (h : Fintype.card V = 2 * N) :
  PerfectMatching G N ↔ SecondPlayerWins G := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_matching_iff_second_player_wins_l1016_101682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_at_x_1_l1016_101626

noncomputable def f1 (x : ℝ) : ℝ := 2 * x - 1
noncomputable def f2 (x : ℝ) : ℝ := (x^3 - 1) / (x - 1)
noncomputable def f3 (x : ℝ) : ℝ := (x^3 - 1) / (x - 1)

theorem equations_at_x_1 :
  f2 1 = f3 1 ∧ f1 1 ≠ f2 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_at_x_1_l1016_101626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_count_l1016_101687

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℤ := floor (2 * x) + floor (3 * x) + floor (4 * x) + floor (5 * x)

theorem f_range_count :
  ∃! n : ℕ, n = (Set.range (fun x : ℝ => f x) ∩ Set.Icc 0 (f 100)).toFinite.toFinset.card ∧ n = 101 :=
by
  sorry

#check f_range_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_count_l1016_101687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_can_catch_bus_l1016_101646

theorem boy_can_catch_bus (bus_speed : ℝ) (x : ℝ) 
  (h1 : bus_speed > 0)
  (h2 : 0 < x)
  (h3 : x ≤ 1) : 
  (x ≤ 2 * (bus_speed / 4) / (bus_speed + bus_speed / 4)) ∨ 
  (1 - x ≤ 2 * (bus_speed / 4) / (bus_speed - bus_speed / 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_can_catch_bus_l1016_101646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_squared_l1016_101634

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on the hyperbola xy = 1 -/
def onHyperbola (p : Point) : Prop :=
  p.x * p.y = 1

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : IsoscelesRightTriangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- Theorem: The square of the area of an isosceles right triangle with vertices on xy = 1 and centroid at origin -/
theorem isosceles_right_triangle_area_squared
  (t : ℝ)
  (h_t_nonzero : t ≠ 0)
  (triangle : IsoscelesRightTriangle)
  (h_on_hyperbola : onHyperbola triangle.A ∧ onHyperbola triangle.B ∧ onHyperbola triangle.C)
  (h_centroid : centroid triangle = ⟨0, 0⟩)
  (h_isosceles_right : triangle.B = ⟨t, 1/t⟩ ∧ triangle.C = ⟨-t, -1/t⟩) :
  ∃ (area : ℝ), area^2 = 4 * (t^2 + 1/t^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_squared_l1016_101634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_games_l1016_101655

theorem soccer_team_games (win_percentage : ℚ) (games_won : ℕ) (total_games : ℕ) : 
  win_percentage = 60 / 100 →
  games_won = 78 →
  (win_percentage * total_games = games_won) →
  total_games = 130 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_games_l1016_101655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_pillar_in_crate_l1016_101696

/-- Represents the dimensions of a rectangular crate -/
structure Crate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical pillar -/
structure Pillar where
  radius : ℝ
  height : ℝ

/-- Checks if a pillar fits inside a crate -/
def fits_in_crate (p : Pillar) (c : Crate) : Prop :=
  p.radius * 2 ≤ c.length ∧ p.radius * 2 ≤ c.width ∧ p.height ≤ c.height

/-- Calculates the volume of a cylindrical pillar -/
noncomputable def pillar_volume (p : Pillar) : ℝ := Real.pi * p.radius^2 * p.height

/-- Theorem stating the largest pillar that fits in the crate -/
theorem largest_pillar_in_crate :
  let c : Crate := ⟨7, 8, 12⟩
  let p : Pillar := ⟨7, 7⟩
  fits_in_crate p c ∧
  (∀ q : Pillar, fits_in_crate q c → pillar_volume q ≤ pillar_volume p) ∧
  pillar_volume p = 343 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_pillar_in_crate_l1016_101696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_bread_weight_difference_l1016_101650

/-- The weight difference between a cake and a piece of bread -/
def weight_difference (cake_weight bread_weight : ℝ) : ℝ :=
  cake_weight - bread_weight

/-- The total weight of a given number of cakes -/
def total_cake_weight (num_cakes : ℕ) (cake_weight : ℝ) : ℝ :=
  (num_cakes : ℝ) * cake_weight

/-- The total weight of cakes and bread -/
def total_weight (num_cakes num_bread : ℕ) (cake_weight bread_weight : ℝ) : ℝ :=
  total_cake_weight num_cakes cake_weight + (num_bread : ℝ) * bread_weight

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

theorem cake_bread_weight_difference :
  ∃ (cake_weight bread_weight : ℝ),
    total_cake_weight 7 cake_weight = 1950 ∧
    total_weight 5 12 cake_weight bread_weight = 2750 ∧
    approx_equal (weight_difference cake_weight bread_weight) 165.47 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_bread_weight_difference_l1016_101650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1016_101697

theorem equation_solution : 
  ∃ x : ℚ, (4 : ℝ) ^ (6 * (x : ℝ) - 9) = (1 / 2 : ℝ) ^ (3 * (x : ℝ) + 7) ↔ x = 11 / 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1016_101697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_items_count_l1016_101613

theorem bakery_items_count : 49 + 19 + 22 = 90 := by
  -- Define the number of each item
  let bread_rolls := 49
  let croissants := 19
  let bagels := 22
  
  -- Perform the calculation
  calc
    49 + 19 + 22 = bread_rolls + croissants + bagels := by rfl
    _ = 90 := by native_decide

-- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_items_count_l1016_101613
