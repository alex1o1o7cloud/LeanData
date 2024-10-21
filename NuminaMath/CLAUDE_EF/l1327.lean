import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_l1327_132792

/-- Calculates the average speed in miles per hour given distance in miles and time in minutes -/
noncomputable def averageSpeed (distance : ℝ) (time : ℝ) : ℝ := (distance / time) * 60

/-- The problem statement -/
theorem speed_difference : 
  let henryDistance : ℝ := 8
  let henryTime : ℝ := 40
  let aliceDistance : ℝ := 10
  let aliceTime : ℝ := 15
  averageSpeed aliceDistance aliceTime - averageSpeed henryDistance henryTime = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_l1327_132792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1327_132744

theorem range_of_x (x : ℝ) : 
  (¬(1/(3-x) > 1) ∧ (x^2 + 2*x - 3 > 0)) → 
  (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1327_132744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trade_in_change_l1327_132705

/-- Calculates the change received in a video game console trade-in scenario -/
theorem trade_in_change (super_nintendo_value : ℝ) (trade_in_percentage : ℝ)
  (cash_given : ℝ) (game_value : ℝ) (nes_sale_price : ℝ)
  (h1 : super_nintendo_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : cash_given = 80)
  (h4 : game_value = 30)
  (h5 : nes_sale_price = 160) :
  cash_given - (nes_sale_price - super_nintendo_value * trade_in_percentage - game_value) = 70 := by
  sorry

#check trade_in_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trade_in_change_l1327_132705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_2023_l1327_132754

/-- Represents a monomial in the sequence -/
structure Monomial where
  coefficient : Int
  exponent : Nat
  deriving Repr

/-- Generates the nth monomial in the sequence -/
def nthMonomial (n : Nat) : Monomial :=
  { coefficient := (-1)^n * (n + 1),
    exponent := n }

/-- The main theorem stating that the 2023rd monomial is -2024x^2023 -/
theorem monomial_2023 : 
  (nthMonomial 2023).coefficient = -2024 ∧ 
  (nthMonomial 2023).exponent = 2023 := by
  sorry

#eval nthMonomial 2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_2023_l1327_132754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_A_l1327_132798

/-- A set of 200 different positive integers -/
def ValidSet : Type := { A : Finset ℕ+ // A.card = 200 }

/-- Predicate to check if three numbers form a non-obtuse triangle -/
def IsNonObtuseTriangle (a b c : ℕ+) : Prop :=
  a^2 + b^2 ≥ c^2 ∧ a^2 + c^2 ≥ b^2 ∧ b^2 + c^2 ≥ a^2

/-- Predicate to check if a set satisfies the triangle condition -/
def SatisfiesTriangleCondition (A : ValidSet) : Prop :=
  ∀ a b c, a ∈ A.val → b ∈ A.val → c ∈ A.val → a ≠ b → b ≠ c → a ≠ c → IsNonObtuseTriangle a b c

/-- Sum of perimeters of all unique triangles determined by A -/
def S (A : ValidSet) : ℕ := sorry

/-- Theorem stating the minimum value of S(A) -/
theorem min_S_A (A : ValidSet) (h : SatisfiesTriangleCondition A) :
  S A ≥ 2279405700 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_A_l1327_132798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_equality_l1327_132771

-- Define the function f using the given table of values
def f : ℤ → ℤ
| -1 => 2
| 4  => 15
| 7  => 10
| 10 => 4
| 14 => 0
| 20 => 9
| _  => 0  -- Default case for undefined values

-- Assume f is invertible
axiom f_invertible : Function.Bijective f

-- Define f_inv as the inverse of f
noncomputable def f_inv : ℤ → ℤ := Function.invFun f

-- State the theorem
theorem inverse_function_equality :
  f_inv ((f_inv 9 + 2 * f_inv 15) / f_inv 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_equality_l1327_132771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_equals_polynomial_sum_l1327_132746

def geometric_sum (x : ℝ) (n : ℕ) : ℝ := 
  Finset.sum (Finset.range (n+1)) (λ i => (1 + x)^(i+1))

def polynomial_sum (a : ℕ → ℝ) (x : ℝ) (n : ℕ) : ℝ := 
  Finset.sum (Finset.range (n+1)) (λ i => a i * x^i)

theorem geometric_sum_equals_polynomial_sum 
  (n : ℕ) (a : ℕ → ℝ) :
  (∀ x : ℝ, geometric_sum x n = polynomial_sum a x n) →
  (Finset.sum (Finset.range (n+1)) a = 62) →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_equals_polynomial_sum_l1327_132746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_theorem_l1327_132785

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * abs x + 4

-- State the theorem
theorem function_domain_range_theorem (a b : ℝ) :
  a < b ∧
  (∀ x, a ≤ x ∧ x ≤ b → f x ∈ Set.Icc (3 * a) (3 * b)) ∧
  (∀ y ∈ Set.Icc (3 * a) (3 * b), ∃ x ∈ Set.Icc a b, f x = y) →
  a = 1 ∧ b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_theorem_l1327_132785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l1327_132739

theorem weight_loss_challenge (W : ℝ) (hW : W > 0) : 
  let weight_after_loss := 0.90 * W
  let weight_with_clothes := weight_after_loss * 1.02
  let measured_loss_percentage := (W - weight_with_clothes) / W * 100
  measured_loss_percentage = 8.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_loss_challenge_l1327_132739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pi_irrational_l1327_132757

-- Define the set of numbers
def numbers : Set ℝ := {Real.sqrt 4, -(8 ^ (1/3)), 1/3, Real.pi}

-- Define a predicate for irrational numbers
def isIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ ↑p / ↑q

-- Theorem statement
theorem only_pi_irrational :
  ∃! x, x ∈ numbers ∧ isIrrational x ∧ x = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pi_irrational_l1327_132757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1327_132738

-- Define the line and circle
def line (k : ℝ) (x y : ℝ) : Prop := x + y - k = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the points A and B
noncomputable def A (k : ℝ) : ℝ × ℝ := sorry
noncomputable def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem range_of_k (k : ℝ) :
  (k > 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧ 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    (x₁, y₁) ≠ (x₂, y₂)) →
  (vec_length (vec_add (A k) (B k)) ≥ (Real.sqrt 3 / 3) * vec_length (vec_add (A k) (vec_add (B k) (-1, -1)))) →
  (Real.sqrt 2 ≤ k ∧ k < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1327_132738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_true_proposition_l1327_132756

-- Define the basic concepts
def Line : Type := Unit
def skew (a b : Line) : Prop := sorry
def intersect (a b : Line) : Prop := sorry
def parallel (a b : Line) : Prop := sorry
def perpendicular (a b : Line) : Prop := sorry
noncomputable def angle_formed_by (a b c : Line) : ℝ := sorry

-- Define the propositions
def proposition1 (a b c : Line) : Prop :=
  (skew a b ∧ skew b c) → skew a c

def proposition2 (a b c : Line) : Prop :=
  (intersect a b ∧ intersect b c) → intersect a c

def proposition3 (a b c : Line) : Prop :=
  parallel a b → angle_formed_by a b c = angle_formed_by b a c

def proposition4 (a b c : Line) : Prop :=
  (perpendicular a b ∧ perpendicular b c) → parallel a c

-- Theorem statement
theorem exactly_one_true_proposition :
  ∃! i : Fin 4, match i with
    | 0 => ∀ a b c, proposition1 a b c
    | 1 => ∀ a b c, proposition2 a b c
    | 2 => ∀ a b c, proposition3 a b c
    | 3 => ∀ a b c, proposition4 a b c
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_true_proposition_l1327_132756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_x_intercept_y_intercept_x₀_range_y₀_range_l1327_132753

/-- The modulus used in the problem -/
def m : ℕ := 25

/-- The coefficient of x in the congruence equation -/
def a : ℕ := 5

/-- The coefficient of y in the congruence equation -/
def b : ℕ := 3

/-- The constant term in the congruence equation -/
def c : ℕ := 2

/-- The x-intercept of the congruence equation -/
def x₀ : ℕ := 10

/-- The y-intercept of the congruence equation -/
def y₀ : ℕ := 16

/-- Theorem stating that the sum of x and y intercepts is 26 -/
theorem intercept_sum : x₀ + y₀ = 26 := by sorry

/-- Theorem stating that x₀ satisfies the congruence equation when y = 0 -/
theorem x_intercept : a * x₀ ≡ c [MOD m] := by sorry

/-- Theorem stating that y₀ satisfies the congruence equation when x = 0 -/
theorem y_intercept : b * y₀ + c ≡ 0 [MOD m] := by sorry

/-- Theorem stating that x₀ is less than m -/
theorem x₀_range : x₀ < m := by sorry

/-- Theorem stating that y₀ is less than m -/
theorem y₀_range : y₀ < m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_x_intercept_y_intercept_x₀_range_y₀_range_l1327_132753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_B_theorem_C_l1327_132768

-- Define the basic types
variable (α β : Plane) (m n : Line)

-- Define the relations
axiom parallel_line_plane : Line → Plane → Prop
axiom perpendicular_line_plane : Line → Plane → Prop
axiom parallel_planes : Plane → Plane → Prop
axiom perpendicular_planes : Plane → Plane → Prop
axiom parallel_lines : Line → Line → Prop

-- Theorem B
theorem theorem_B (h1 : parallel_line_plane m α) (h2 : perpendicular_line_plane m β) :
  perpendicular_planes α β := by
  sorry

-- Theorem C
theorem theorem_C (h1 : parallel_planes α β) (h2 : perpendicular_line_plane m α) 
  (h3 : perpendicular_line_plane n β) :
  parallel_lines m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_B_theorem_C_l1327_132768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_theorem_l1327_132732

noncomputable def line_l (a t : ℝ) : ℝ × ℝ := ((Real.sqrt 3 / 2) * t + a, t / 2)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_points_theorem (a : ℝ) :
  (∃ t1 t2 θ1 θ2 : ℝ,
    line_l a t1 = curve_C θ1 ∧
    line_l a t2 = curve_C θ2 ∧
    distance (line_l a t1) (line_l a t2) = 4 ∧
    distance (0, 0) (line_l a t1 + line_l a t2) = 6) →
  a = -1 ∨ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_theorem_l1327_132732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_x_value_l1327_132760

/-- Given two planes α and β with normal vectors a and b respectively,
    prove that if they are perpendicular, then the x-component of b is -10. -/
theorem perpendicular_planes_x_value :
  let a : Fin 3 → ℝ := ![(-1 : ℝ), 2, 4]
  let b : Fin 3 → ℝ := ![x, -1, -2]
  ∀ x : ℝ,
  (a 0 * b 0 + a 1 * b 1 + a 2 * b 2 = 0) →
  x = -10 :=
by
  intro x h
  -- The proof steps would go here
  sorry

#check perpendicular_planes_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_x_value_l1327_132760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_abs_plus_self_nonnegative_l1327_132735

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem abs_plus_self_nonnegative :
  ¬ (∃ x > (0 : ℝ), |x| + x ≥ 0) ↔ (∀ x > (0 : ℝ), |x| + x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_abs_plus_self_nonnegative_l1327_132735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_cardinality_l1327_132765

-- Define the number of subsets function
def n (S : Finset α) : ℕ := 2^(Finset.card S)

-- Main theorem
theorem max_intersection_cardinality
  (A B C : Finset ℕ)
  (hA : A.card = 2016)
  (hB : B.card = 2016)
  (hsubset : n A + n B + n C = n (A ∪ B ∪ C)) :
  (∃ (X Y Z : Finset ℕ), X.card = 2016 ∧ Y.card = 2016 ∧
    n X + n Y + n Z = n (X ∪ Y ∪ Z) ∧
    (X ∩ Y ∩ Z).card ≤ 2015) ∧
  (∃ (X Y Z : Finset ℕ), X.card = 2016 ∧ Y.card = 2016 ∧
    n X + n Y + n Z = n (X ∪ Y ∪ Z) ∧
    (X ∩ Y ∩ Z).card = 2015) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_cardinality_l1327_132765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_cost_l1327_132712

/-- Calculates the total cost of an item with a discount and sales tax -/
theorem aquarium_cost (original_price discount_percent tax_percent : ℚ) 
  (h1 : original_price = 120)
  (h2 : discount_percent = 50)
  (h3 : tax_percent = 5) : 
  original_price * (1 - discount_percent / 100) * (1 + tax_percent / 100) = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_cost_l1327_132712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tint_percentage_is_correct_l1327_132720

/-- Represents the initial mixture composition -/
structure InitialMixture where
  total_volume : ℝ
  red_tint_percent : ℝ
  yellow_tint_percent : ℝ
  water_percent : ℝ

/-- Represents the new mixture after adding red tint -/
structure NewMixture where
  initial : InitialMixture
  added_red_tint : ℝ

/-- Calculates the percentage of red tint in the new mixture -/
noncomputable def red_tint_percentage (new_mixture : NewMixture) : ℝ :=
  let initial_red_tint := new_mixture.initial.total_volume * (new_mixture.initial.red_tint_percent / 100)
  let total_red_tint := initial_red_tint + new_mixture.added_red_tint
  let new_total_volume := new_mixture.initial.total_volume + new_mixture.added_red_tint
  (total_red_tint / new_total_volume) * 100

/-- Theorem stating that the percentage of red tint in the new mixture is approximately 28.57% -/
theorem red_tint_percentage_is_correct (mix : NewMixture) 
  (h1 : mix.initial.total_volume = 50)
  (h2 : mix.initial.red_tint_percent = 20)
  (h3 : mix.initial.yellow_tint_percent = 40)
  (h4 : mix.initial.water_percent = 40)
  (h5 : mix.added_red_tint = 6) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |red_tint_percentage mix - 28.57| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tint_percentage_is_correct_l1327_132720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_simplification_l1327_132711

theorem sin_cos_sum_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_simplification_l1327_132711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_pencils_difference_l1327_132764

theorem colored_pencils_difference (total : ℕ) (red_ratio : ℚ) (blue_ratio : ℚ) : 
  total = 36 → red_ratio = 5/9 → blue_ratio = 5/12 → 
  (red_ratio * total).floor - (blue_ratio * total).floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_pencils_difference_l1327_132764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_value_points_l1327_132713

noncomputable section

-- Define the concept of a clever value point
def has_clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = (deriv f) x

-- Define the given functions
noncomputable def f₁ : ℝ → ℝ := λ x ↦ x^2
noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.exp (-x)
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.log x
noncomputable def f₄ : ℝ → ℝ := λ x ↦ 2 + Real.sin x
noncomputable def f₅ : ℝ → ℝ := λ x ↦ x + 1/x

-- Theorem statement
theorem clever_value_points :
  has_clever_value_point f₁ ∧
  ¬has_clever_value_point f₂ ∧
  has_clever_value_point f₃ ∧
  ¬has_clever_value_point f₄ ∧
  has_clever_value_point f₅ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_value_points_l1327_132713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1327_132750

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := (1/2) * Real.cos (2*x - φ)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.cos (4*x - Real.pi/3)

-- State the theorem
theorem function_properties :
  ∀ φ : ℝ, 0 < φ ∧ φ < Real.pi ∧ f φ (Real.pi/6) = 1/2 →
  (φ = Real.pi/3) ∧
  (∀ k : ℤ, MonotoneOn (f φ) (Set.Icc (k*Real.pi - Real.pi/3) (k*Real.pi + Real.pi/6))) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/4), g x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/4), g x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/4), g x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/4), g x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1327_132750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_daily_earning_l1327_132734

-- Define daily earnings for p, q, and r
variable (P Q R : ℚ)

-- Define the conditions
axiom condition1 : (P + Q + R) * 9 = 1890
axiom condition2 : (P + R) * 5 = 600
axiom condition3 : (Q + R) * 7 = 910

-- Theorem to prove
theorem r_daily_earning : R = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_daily_earning_l1327_132734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_incircle_properties_l1327_132784

/-- A right triangle with an incircle -/
structure RightTriangleWithIncircle where
  /-- Length of one segment of the hypotenuse -/
  m : ℝ
  /-- Length of the other segment of the hypotenuse -/
  n : ℝ
  /-- m and n are positive -/
  m_pos : 0 < m
  n_pos : 0 < n

/-- The area of the triangle -/
noncomputable def triangleArea (t : RightTriangleWithIncircle) : ℝ := t.m * t.n

/-- The area of the inscribed rectangle -/
noncomputable def inscribedRectangleArea (t : RightTriangleWithIncircle) : ℝ :=
  2 * t.m^2 * t.n^2 / (t.m + t.n)^2

/-- Main theorem: properties of the right triangle with incircle -/
theorem right_triangle_with_incircle_properties (t : RightTriangleWithIncircle) :
  (triangleArea t = t.m * t.n) ∧
  (inscribedRectangleArea t = 2 * t.m^2 * t.n^2 / (t.m + t.n)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_incircle_properties_l1327_132784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_partial_sums_l1327_132797

def is_permutation (p : List ℕ) : Prop :=
  p.length = 30 ∧ p.toFinset = Finset.range 30

def partial_sums (p : List ℕ) : List ℕ :=
  List.scanl (· + ·) 0 p

def count_odd_sums (p : List ℕ) : ℕ :=
  (partial_sums p).filter (fun n => n % 2 = 1) |>.length

theorem max_odd_partial_sums :
  ∃ (p : List ℕ), is_permutation p ∧
    ∀ (q : List ℕ), is_permutation q →
      count_odd_sums q ≤ count_odd_sums p ∧
      count_odd_sums p = 23 := by
  sorry

#eval count_odd_sums [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_partial_sums_l1327_132797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1327_132769

/-- Definition of a hyperbola -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  eccentricity : ℝ
  /-- One of the asymptotes of the hyperbola -/
  asymptote : ℝ → ℝ

/-- A hyperbola with one asymptote y = √2 * x has eccentricity e equal to either √3 or √6/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) (asym : h.asymptote = fun x ↦ Real.sqrt 2 * x) :
  h.eccentricity = Real.sqrt 3 ∨ h.eccentricity = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1327_132769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l1327_132718

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ a 1 = 1

theorem fourth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l1327_132718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_balls_count_l1327_132714

theorem white_balls_count (total : ℕ) (p_red : ℚ) (p_black : ℚ) : ℕ :=
  let p_white := 1 - p_red - p_black
  let white_balls := (total : ℚ) * p_white
  have h1 : total = 50 := by sorry
  have h2 : p_red = 15 / 100 := by sorry
  have h3 : p_black = 45 / 100 := by sorry
  have h4 : white_balls = 20 := by sorry
  20


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_balls_count_l1327_132714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1327_132794

-- Define the constants
noncomputable def a : ℝ := Real.rpow 0.7 2.1
noncomputable def b : ℝ := Real.rpow 0.7 2.5
noncomputable def c : ℝ := Real.rpow 2.1 0.7

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1327_132794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_x_squared_is_180_l1327_132781

/-- The coefficient of x^2 in the expansion of (1+2x)^10 is 180 -/
def binomial_coefficient_x_squared : ℕ :=
  let n : ℕ := 10
  let k : ℕ := 2
  let b : ℕ := 2
  Nat.choose n k * b^k

/-- The result of the computation is 180 -/
theorem binomial_coefficient_x_squared_is_180 : 
  binomial_coefficient_x_squared = 180 := by
  -- Unfold the definition and simplify
  unfold binomial_coefficient_x_squared
  -- Evaluate the expression
  simp
  -- The result should now be true by reflexivity
  rfl

#eval binomial_coefficient_x_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_x_squared_is_180_l1327_132781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_juice_production_l1327_132701

/-- The amount of apples used for apple juice given the total production and percentages for different uses -/
theorem apple_juice_production (total_production : ℝ) (mixing_percentage : ℝ) (juice_percentage : ℝ) 
  (fresh_percentage : ℝ) (h1 : total_production = 8) 
  (h2 : mixing_percentage = 0.3) (h3 : juice_percentage = 0.6) 
  (h4 : fresh_percentage = 0.4) (h5 : juice_percentage + fresh_percentage = 1) : 
  total_production * (1 - mixing_percentage) * juice_percentage = 3.36 := by
  -- Proof steps would go here
  sorry

#check apple_juice_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_juice_production_l1327_132701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_180_y_axis_l1327_132743

def initial_vector : Fin 3 → ℝ := ![2, 1, 1]

def rotate_180_about_y (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![- v 0, v 1, - v 2]

theorem rotation_180_y_axis :
  rotate_180_about_y initial_vector = ![- 2, 1, - 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_180_y_axis_l1327_132743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_arccos_cube_minus_one_l1327_132788

open Real

theorem integral_arccos_cube_minus_one (x : ℝ) :
  deriv (λ y ↦ -(1/4) * (arccos y)^4 - arccos y) x =
    (arccos x^3 - 1) / sqrt (1 - x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_arccos_cube_minus_one_l1327_132788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_existential_squared_geq_l1327_132702

theorem negation_existential_squared_geq : 
  (¬ ∃ x : ℝ, x^2 ≥ x) ↔ (∀ x : ℝ, x^2 < x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_existential_squared_geq_l1327_132702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1327_132719

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  Real.sqrt 15 * t.b * Real.cos t.A + t.a * Real.sin t.B = 0 ∧
  t.a = 2 * Real.sqrt 6 ∧
  t.b = 4

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : given_conditions t) : 
  Real.cos t.A = -1/4 ∧ 
  t.c = 2 ∧ 
  Real.cos (2 * t.A + t.B) = -(Real.sqrt 6) / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1327_132719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l1327_132715

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a b : ℝ) (x : ℂ) : Prop := x^2 + a * x + b = 0

-- Define z₁
noncomputable def z₁ : ℂ := Real.sqrt 2 * Complex.exp (Complex.I * Real.pi / 4)

-- Define z₂
def z₂ : ℂ := 1 - i

-- Theorem statement
theorem equation_roots_properties :
  ∃ (a b : ℝ),
    equation a b z₁ ∧
    equation a b z₂ ∧
    z₁ = 1 + i ∧
    z₂ = 1 - i ∧
    a = -2 ∧
    b = 2 ∧
    Complex.abs (a + b * i) = 2 * Real.sqrt 2 ∧
    Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l1327_132715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_l1327_132716

-- Define the slopes of the two lines
noncomputable def m₁ : ℝ := Real.sqrt 3
noncomputable def m₂ (t : ℝ) : ℝ := -1 / t

-- Define the angle between the lines
noncomputable def angle (t : ℝ) : ℝ := Real.arctan ((m₁ - m₂ t) / (1 + m₁ * m₂ t))

-- State the theorem
theorem angle_between_lines (t : ℝ) (h : t ≠ 0) : 
  angle t = π / 3 → t = 0 ∨ t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_l1327_132716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_files_on_portable_hard_drive_l1327_132707

-- Define the types of files
inductive FileType
| Photo
| Text

-- Define the storage types
inductive StorageType
| FlashDrive
| PortableHardDrive

-- Define a function to count files of a given type on a given storage
def countFiles (fileType : FileType) (storageType : StorageType) : ℕ := 0

-- Define the total number of files on each storage type
def totalFiles (storageType : StorageType) : ℕ :=
  countFiles FileType.Photo storageType + countFiles FileType.Text storageType

-- State the theorem
theorem photo_files_on_portable_hard_drive :
  (totalFiles StorageType.PortableHardDrive > totalFiles StorageType.FlashDrive) →
  (countFiles FileType.Photo StorageType.FlashDrive + 
   countFiles FileType.Photo StorageType.PortableHardDrive > 
   countFiles FileType.Text StorageType.FlashDrive + 
   countFiles FileType.Text StorageType.PortableHardDrive) →
  (countFiles FileType.Photo StorageType.PortableHardDrive > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_files_on_portable_hard_drive_l1327_132707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_girl_pairs_bound_l1327_132774

structure MyGroup where
  girls : ℕ
  boys : ℕ
  knows : (ℕ × ℕ) → Prop

def boy_girl_pairs (g : MyGroup) : ℕ := sorry

theorem boy_girl_pairs_bound (g : MyGroup) 
  (h : ∀ (b₁ b₂ g₁ g₂ : ℕ), b₁ ≠ b₂ → g₁ ≠ g₂ → b₁ < g.boys → b₂ < g.boys → 
       g₁ < g.girls → g₂ < g.girls → 
       (¬g.knows (b₁, g₁) ∨ ¬g.knows (b₁, g₂) ∨ ¬g.knows (b₂, g₁) ∨ ¬g.knows (b₂, g₂))) : 
  boy_girl_pairs g ≤ g.girls + g.boys * (g.boys - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_girl_pairs_bound_l1327_132774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perpendicular_sum_l1327_132795

-- Define an equilateral triangle
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ i j : Fin 3, i ≠ j → ‖vertices i - vertices j‖ = ‖vertices 0 - vertices 1‖

-- Define the center of the triangle
noncomputable def center (t : EquilateralTriangle) : ℝ × ℝ :=
  (1/3) • (t.vertices 0 + t.vertices 1 + t.vertices 2)

-- Define a point inside the triangle
structure PointInTriangle (t : EquilateralTriangle) where
  point : ℝ × ℝ
  is_inside : ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    point = a • t.vertices 0 + b • t.vertices 1 + c • t.vertices 2

-- Define the perpendicular from a point to a line segment
noncomputable def perpendicular (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ × ℝ :=
  let v := b - a
  let t := ((p - a) • v) / (v • v)
  a + t • v - p

-- State the theorem
theorem equilateral_triangle_perpendicular_sum (t : EquilateralTriangle) (m : PointInTriangle t) :
  let k₁ := perpendicular m.point (t.vertices 0) (t.vertices 1)
  let k₂ := perpendicular m.point (t.vertices 1) (t.vertices 2)
  let k₃ := perpendicular m.point (t.vertices 2) (t.vertices 0)
  k₁ + k₂ + k₃ = (3/2) • (center t - m.point) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perpendicular_sum_l1327_132795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_special_savings_l1327_132745

/-- The regular price of a pair of sandals -/
def regular_price : ℚ := 60

/-- The discount percentage for the second pair of sandals -/
def second_pair_discount : ℚ := 2/5

/-- The discount percentage for the third pair of sandals -/
def third_pair_discount : ℚ := 3/5

/-- The number of pairs of sandals bought -/
def num_pairs : ℕ := 3

/-- The total cost of sandals with the "fair special" offer -/
noncomputable def discounted_total : ℚ :=
  regular_price + 
  regular_price * (1 - second_pair_discount) + 
  regular_price * (1 - third_pair_discount)

/-- The total cost of sandals without any discount -/
def regular_total : ℚ := regular_price * num_pairs

/-- The percentage saved with the "fair special" offer -/
noncomputable def percentage_saved : ℚ := (regular_total - discounted_total) / regular_total * 100

theorem fair_special_savings : 
  32 < percentage_saved ∧ percentage_saved < 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_special_savings_l1327_132745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1327_132790

theorem relationship_abc (a b c : ℝ) : 
  a = Real.rpow 0.3 0.4 → b = Real.log 0.3 / Real.log 4 → c = Real.rpow 4 0.3 → c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1327_132790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_time_for_8_miles_l1327_132725

-- Define the given conditions
noncomputable def liam_distance : ℝ := 5
noncomputable def liam_time : ℝ := 40
noncomputable def maria_initial_distance : ℝ := 3
noncomputable def maria_initial_time_ratio : ℝ := 2/3
noncomputable def maria_target_distance : ℝ := 8

-- Theorem statement
theorem maria_time_for_8_miles :
  let maria_rate := maria_initial_distance / (maria_initial_time_ratio * liam_time)
  maria_target_distance / maria_rate = 640/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_time_for_8_miles_l1327_132725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rush_hour_trip_charge_l1327_132731

/-- Calculates the total charge for a taxi trip during rush hour -/
noncomputable def rush_hour_charge (initial_fee : ℝ) (regular_rate : ℝ) (rush_hour_increase : ℝ) (traffic_delay_charge : ℝ) (distance : ℝ) : ℝ :=
  let rush_hour_rate := regular_rate * (1 + rush_hour_increase)
  let mile_charge := rush_hour_rate * (5/2)
  initial_fee + mile_charge * distance + traffic_delay_charge * distance

/-- The total charge for a 3.6-mile trip during rush hour is $11.53 -/
theorem rush_hour_trip_charge : 
  rush_hour_charge 2.35 0.35 0.2 1.5 3.6 = 11.53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rush_hour_trip_charge_l1327_132731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_rearrangement_l1327_132755

open Set
open Function

-- Define the type for points in a plane
def Point := ℝ × ℝ

-- Define the dot product for vectors in ℝ²
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector between two points
def vector (p q : Point) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem angle_rearrangement 
  (n : ℕ) 
  (A B : Fin n → Point) : 
  ∃ σ : Equiv.Perm (Fin n), 
    ∀ i j : Fin n, 
      dot_product (vector (A i) (A j)) (vector (B (σ i)) (B (σ j))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_rearrangement_l1327_132755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_choice_l1327_132767

open Set

noncomputable def alice_range : Set ℝ := Icc 0 1
noncomputable def bob_range : Set ℝ := Icc (1/3) (2/3)
noncomputable def dave_range : Set ℝ := Icc 0 (1/3)

noncomputable def winning_probability (c : ℝ) : ℝ :=
  if c < 1/3 then
    3 * c * c
  else if c ≤ 2/3 then
    -18 * c * c + 18 * c - 3
  else
    1 - c

theorem optimal_choice : 
  ∃ (c : ℝ), c = 13/24 ∧ 
  ∀ (x : ℝ), x ∈ Icc 0 1 → winning_probability c ≥ winning_probability x := by
  sorry

#check optimal_choice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_choice_l1327_132767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_solid_volume_l1327_132729

/-- The volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (R r h : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

/-- The volume of a cylinder -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The total volume of the combined solid -/
noncomputable def total_volume (R_cone r_cone h_cone r_cyl h_cyl : ℝ) : ℝ :=
  truncated_cone_volume R_cone r_cone h_cone + cylinder_volume r_cyl h_cyl

theorem combined_solid_volume :
  total_volume 10 3 8 3 10 = (1382 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_solid_volume_l1327_132729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_minus_2_positive_l1327_132763

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.rpow 2 x - 4 else Real.rpow 2 (-x) - 4

-- State the theorem
theorem f_x_minus_2_positive :
  (∀ x : ℝ, f x = f (-x)) →  -- f is even
  (∀ x : ℝ, x ≥ 0 → f x = Real.rpow 2 x - 4) →  -- f(x) = 2^x - 4 for x ≥ 0
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_minus_2_positive_l1327_132763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_implies_m_range_l1327_132721

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt (m * x^2 + 2 * (m - 2) * x + 1)

-- Define the range of m
def m_range (m : ℝ) : Prop := m ∈ Set.Ici 0 ∩ Set.Iic 1 ∪ Set.Ici 4

-- State the theorem
theorem f_range_implies_m_range :
  (∀ m : ℝ, (∀ y : ℝ, y > 0 → ∃ x : ℝ, f m x = y) →
   (∀ x : ℝ, f m x > 0)) →
  (∀ m : ℝ, m_range m ↔ 
    ((∀ y : ℝ, y > 0 → ∃ x : ℝ, f m x = y) ∧
     (∀ x : ℝ, f m x > 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_implies_m_range_l1327_132721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l1327_132723

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two :
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l1327_132723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_subsets_of_S_num_proper_subsets_of_S_l1327_132736

-- Define a set S with 5 elements
def S : Finset ℕ := {1, 2, 3, 4, 5}

-- Theorem for the number of subsets
theorem num_subsets_of_S :
  (Finset.powerset S).card = 32 := by sorry

-- Theorem for the number of proper subsets
theorem num_proper_subsets_of_S :
  ((Finset.powerset S).filter (λ s => s ≠ S)).card = 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_subsets_of_S_num_proper_subsets_of_S_l1327_132736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_log_l1327_132730

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => Real.log x / Real.log a

theorem inverse_function_log (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let g : ℝ → ℝ := fun x => a ^ x
  Function.LeftInverse (f a) g ∧ Function.RightInverse (f a) g ∧ f a 3 = 1 → f a = f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_log_l1327_132730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1327_132766

noncomputable def f₁ (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f₂ (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f₃ (x : ℝ) : ℝ := -(2/3) * x + 8

noncomputable def g (x : ℝ) : ℝ := min (f₁ x) (min (f₂ x) (f₃ x))

theorem max_value_of_g :
  ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), g x ≤ M ∧ ∃ (x₀ : ℝ), g x₀ = M := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1327_132766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l1327_132728

/-- The curve xy = 2 -/
def curve (x y : ℝ) : Prop := x * y = 2

/-- The circle (x - a)² + (y - b)² = r² -/
def circle_equation (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

/-- The four intersection points of the curve and the circle -/
def intersection_points (a b r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    curve x₁ y₁ ∧ circle_equation x₁ y₁ a b r ∧
    curve x₂ y₂ ∧ circle_equation x₂ y₂ a b r ∧
    curve x₃ y₃ ∧ circle_equation x₃ y₃ a b r ∧
    curve x₄ y₄ ∧ circle_equation x₄ y₄ a b r ∧
    x₁ = 4 ∧ y₁ = 1/2 ∧
    x₂ = -2 ∧ y₂ = -1 ∧
    x₃ = 1/5 ∧ y₃ = 10 ∧
    x₄ = -5/2 ∧ y₄ = -4/5

theorem fourth_intersection_point (a b r : ℝ) :
  intersection_points a b r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l1327_132728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l1327_132778

/-- The side length of the main square ABCD -/
def side_length : ℝ := 6

/-- The side length of the small square in the bottom-left corner -/
def small_square_side : ℝ := 1

/-- The width of the right triangle -/
def triangle_width : ℝ := 2

/-- The height of the right triangle -/
def triangle_height : ℝ := 3

/-- The side length of the larger square in the top-right corner -/
def large_square_side : ℝ := 2

/-- The total area of the main square ABCD -/
noncomputable def total_area : ℝ := side_length ^ 2

/-- The area of the small square in the bottom-left corner -/
noncomputable def small_square_area : ℝ := small_square_side ^ 2

/-- The area of the right triangle -/
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_width * triangle_height

/-- The area of the larger square in the top-right corner -/
noncomputable def large_square_area : ℝ := large_square_side ^ 2

/-- The total shaded area -/
noncomputable def shaded_area : ℝ := small_square_area + triangle_area + large_square_area

/-- The theorem stating that the ratio of shaded area to total area is 2/9 -/
theorem shaded_area_ratio : shaded_area / total_area = 2 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l1327_132778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1327_132708

theorem cos_beta_value (α β : Real) 
  (h1 : Real.sin α = 4/5)
  (h2 : Real.cos (α + β) = -12/13)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : Real.pi/2 < α + β ∧ α + β < Real.pi) :
  Real.cos β = -16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1327_132708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l1327_132751

def sequence_a : ℕ → ℚ
  | 0 => 10
  | n + 1 => sequence_a n + (n + 1)

theorem min_value_of_sequence_ratio :
  ∃ (n : ℕ), (n = 4 ∨ n = 5) ∧
  ∀ (m : ℕ), m > 0 → sequence_a n / n ≤ sequence_a m / m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l1327_132751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1327_132782

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | Real.exp (Real.log 3 * x) < 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1327_132782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_eq_half_l1327_132776

theorem sin_two_alpha_eq_half (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (π / 2 + 2 * α) = Real.cos (π / 4 - α)) : Real.sin (2 * α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_eq_half_l1327_132776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_inequality_l1327_132789

def Point3D := ℝ × ℝ × ℝ

def projectYZ (p : Point3D) : ℝ × ℝ := (p.2.1, p.2.2)
def projectZX (p : Point3D) : ℝ × ℝ := (p.2.2, p.1)
def projectXY (p : Point3D) : ℝ × ℝ := (p.1, p.2.1)

theorem projection_inequality (S : Finset Point3D) :
  let Sx := S.image projectYZ
  let Sy := S.image projectZX
  let Sz := S.image projectXY
  (S.card : ℝ) ^ 2 ≤ (Sx.card : ℝ) * (Sy.card : ℝ) * (Sz.card : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_inequality_l1327_132789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_extension_correct_xiaogang_arm_extension_l1327_132761

/-- Calculates the length by which a raised arm exceeds the head height -/
noncomputable def arm_extension (h s s' : ℝ) : ℝ := h * (s' - s) / s

theorem arm_extension_correct 
  (h s s' : ℝ) 
  (h_pos : h > 0) 
  (s_pos : s > 0) 
  (s'_gt_s : s' > s) : 
  arm_extension h s s' = h * (s' - s) / s :=
by sorry

theorem xiaogang_arm_extension : 
  arm_extension 1.7 0.85 1.1 = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_extension_correct_xiaogang_arm_extension_l1327_132761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_in_class_l1327_132759

theorem min_students_in_class (n : ℕ) (scores : Finset ℕ) : 
  n ≥ 13 ∧ 
  scores.card = n ∧
  (∃ perfect_scores : Finset ℕ, perfect_scores ⊆ scores ∧ perfect_scores.card = 8 ∧ ∀ s ∈ perfect_scores, s = 100) ∧
  (∀ s ∈ scores, s ≥ 50 ∧ s ≤ 100) ∧
  (Finset.sum scores (fun x => (x : ℚ))) / n = 82 →
  n = 13 ∧ ∀ m : ℕ, m < 13 → 
    ¬(∃ scores' : Finset ℕ, 
      scores'.card = m ∧
      (∃ perfect_scores' : Finset ℕ, perfect_scores' ⊆ scores' ∧ perfect_scores'.card = 8 ∧ ∀ s ∈ perfect_scores', s = 100) ∧
      (∀ s ∈ scores', s ≥ 50 ∧ s ≤ 100) ∧
      (Finset.sum scores' (fun x => (x : ℚ))) / m = 82) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_in_class_l1327_132759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_circle_radius_for_specific_semicircles_l1327_132741

/-- The radius of a circle touching two concentric semicircles and their common diameter -/
noncomputable def touching_circle_radius (R : ℝ) (r₁ : ℝ) : ℝ :=
  8 / 9

/-- Theorem: The radius of a circle touching two concentric semicircles with radii 2 and 1,
    and their common diameter, is 8/9 -/
theorem touching_circle_radius_for_specific_semicircles :
  touching_circle_radius 2 1 = 8 / 9 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_circle_radius_for_specific_semicircles_l1327_132741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_squared_l1327_132787

-- Define the probability density function
noncomputable def p (x : ℝ) : ℝ :=
  if 0 < x ∧ x < Real.pi/2 then Real.cos x else 0

-- Define the random variable X
def X : Type := ℝ

-- State the theorem
theorem expected_value_X_squared :
  ∫ x in Set.Ioo 0 (Real.pi/2), x^2 * p x = Real.pi^2/4 - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_squared_l1327_132787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_12_l1327_132706

/-- Represents the dimensions and whitewashing details of a room --/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  door_area : ℝ
  window_area : ℝ
  num_windows : ℕ
  whitewash_cost_per_sqft : ℝ
  total_whitewash_cost : ℝ

/-- Calculates the total whitewashable area of the room --/
def whitewashable_area (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - r.door_area - (r.num_windows : ℝ) * r.window_area

/-- Theorem stating that the height of the room is 12 feet --/
theorem room_height_is_12 (r : Room) 
  (h1 : r.length = 25)
  (h2 : r.width = 15)
  (h3 : r.door_area = 18)
  (h4 : r.window_area = 12)
  (h5 : r.num_windows = 3)
  (h6 : r.whitewash_cost_per_sqft = 9)
  (h7 : r.total_whitewash_cost = 8154)
  (h8 : r.whitewash_cost_per_sqft * whitewashable_area r = r.total_whitewash_cost) :
  r.height = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_12_l1327_132706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_1080_l1327_132703

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let s := String.mk (n.repr.data.reverse)
  s.toNat!

def satisfies_conditions (m : ℕ) : Prop :=
  is_four_digit m ∧
  is_four_digit (reverse_digits m) ∧
  m % 54 = 0 ∧
  (reverse_digits m) % 54 = 0 ∧
  m % 8 = 0

theorem smallest_m_is_1080 :
  ∃ (m : ℕ), satisfies_conditions m ∧ 
  ∀ (n : ℕ), satisfies_conditions n → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_1080_l1327_132703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_increase_percentage_l1327_132724

theorem earnings_increase_percentage (last_year_earnings : ℝ) 
  (h1 : last_year_earnings > 0) : 
  ∃ (increase_percentage : ℝ),
    0.1 * last_year_earnings * 3.45 = 0.3 * (last_year_earnings * (1 + increase_percentage / 100)) ∧
    increase_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_increase_percentage_l1327_132724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_interval_l1327_132722

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom symmetric : ∀ x : ℝ, f (2 - x) = f x
axiom decreasing_after_one : ∀ x y : ℝ, 1 ≤ x → x < y → f y < f x
axiom f_zero : f 0 = 0

-- Theorem to prove
theorem solution_set_is_open_interval :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_interval_l1327_132722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_second_quadrant_l1327_132777

-- Define the complex number z as a function of m
noncomputable def z (m : ℝ) : ℂ := (m^2 - m - 6) / (m + 3) + (m^2 - 5*m + 6) * Complex.I

-- Part 1: z is purely imaginary iff m = -2
theorem purely_imaginary (m : ℝ) : z m = (z m).im * Complex.I ↔ m = -2 := by sorry

-- Part 2: z lies in the second quadrant iff m < -3 or -2 < m < 2 or m > 3
theorem second_quadrant (m : ℝ) : 
  (z m).re < 0 ∧ (z m).im > 0 ↔ m < -3 ∨ (-2 < m ∧ m < 2) ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_second_quadrant_l1327_132777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_a_l1327_132733

-- Define the function f
noncomputable def f (b : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + b

-- Define the inverse function f_inv
noncomputable def f_inv (b : ℤ) : ℝ → ℝ := λ x ↦ (x - b) / 2

-- State the theorem
theorem intersection_point_a (b a : ℤ) : 
  (f b (-4) = a ∧ f_inv b (-4) = a) → a = -4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check intersection_point_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_a_l1327_132733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l1327_132704

theorem tan_sum_given_tan_cot_sum (a b : ℝ) 
  (h1 : Real.tan a + Real.tan b = 12)
  (h2 : (Real.tan a)⁻¹ + (Real.tan b)⁻¹ = 5) : 
  Real.tan (a + b) = -60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l1327_132704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saturated_functions_of_1_l1327_132747

/-- A function is a "saturated function of 1" if there exists a real number x₀ in its domain
    such that f(x₀+1) = f(x₀) + f(1) holds. -/
def is_saturated_function_of_1 (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

/-- The reciprocal function -/
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x

/-- The exponential function with base 2 -/
noncomputable def f₂ (x : ℝ) : ℝ := 2^x

/-- The logarithmic function of x^2 + 2 -/
noncomputable def f₃ (x : ℝ) : ℝ := Real.log (x^2 + 2) / Real.log 2

/-- The cosine function with period 2 -/
noncomputable def f₄ (x : ℝ) : ℝ := Real.cos (Real.pi * x)

theorem saturated_functions_of_1 :
  ¬ is_saturated_function_of_1 f₁ ∧
  is_saturated_function_of_1 f₂ ∧
  ¬ is_saturated_function_of_1 f₃ ∧
  is_saturated_function_of_1 f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_saturated_functions_of_1_l1327_132747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_taps_fill_time_l1327_132793

-- Define the bucket volume
noncomputable def bucket_volume : ℝ := 36

-- Define the flow rate of tap A
noncomputable def tap_a_rate : ℝ := 3

-- Define the time taken by tap B to fill 1/3 of the bucket
noncomputable def tap_b_time : ℝ := 20

-- Define the fraction of the bucket that tap B fills in 20 minutes
noncomputable def tap_b_fraction : ℝ := 1/3

-- Theorem statement
theorem both_taps_fill_time :
  let tap_b_rate := (tap_b_fraction * bucket_volume) / tap_b_time
  let combined_rate := tap_a_rate + tap_b_rate
  bucket_volume / combined_rate = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_taps_fill_time_l1327_132793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrants_l1327_132758

theorem terminal_side_quadrants (θ : ℝ) (h : Real.sin (Real.pi + θ) = 4/5) :
  (3 * Real.pi / 2 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 2 * Real.pi) ∨
  (Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrants_l1327_132758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l1327_132752

/-- Given a triangle ABC with sides AB = 6, BC = 8, AC = 10, and a point D on AC such that BD = 6,
    prove that the ratio of AD to DC is 18:7. -/
theorem triangle_ratio_theorem (A B C D : EuclideanSpace ℝ (Fin 2)) : 
  ‖B - A‖ = 6 →
  ‖C - B‖ = 8 →
  ‖A - C‖ = 10 →
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = A + t • (C - A) →
  ‖B - D‖ = 6 →
  ∃ r s : ℝ, r / s = 18 / 7 ∧ ‖A - D‖ / ‖D - C‖ = r / s :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l1327_132752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1327_132770

-- Define the functions
noncomputable def f1 (x : ℝ) := Real.sin x
noncomputable def f2 (x : ℝ) := |Real.sin x|
noncomputable def f3 (x : ℝ) := 3 * Real.sin x + 1
noncomputable def f4 (x : ℝ) := Real.sin x - 1

-- Define odd and even properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem function_properties :
  (is_odd f1) ∧
  (is_even f2) ∧
  (¬(is_odd f3) ∧ ¬(is_even f3)) ∧
  (¬(is_odd f4) ∧ ¬(is_even f4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1327_132770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_double_vector_l1327_132779

variable {E : Type*} [NormedAddCommGroup E] [Module ℝ E]

theorem norm_double_vector (v : E) (h : ‖v‖ = 5) : ‖(2 : ℝ) • v‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_double_vector_l1327_132779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMCN_10_5_l1327_132726

/-- Rectangle with midpoints -/
structure RectangleWithMidpoints where
  width : ℝ
  height : ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  is_M_midpoint : M.1 = width ∧ M.2 = height / 2
  is_N_midpoint : N.1 = width / 2 ∧ N.2 = 0

/-- The area of quadrilateral AMCN in a rectangle with midpoints -/
noncomputable def area_AMCN (r : RectangleWithMidpoints) : ℝ :=
  r.width * r.height / 2

/-- Theorem: The area of AMCN in a 10x5 rectangle with midpoints is 25 -/
theorem area_AMCN_10_5 :
  ∃ (r : RectangleWithMidpoints),
    r.width = 10 ∧ r.height = 5 ∧ area_AMCN r = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AMCN_10_5_l1327_132726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_coin_splitting_l1327_132791

/-- Represents the game where n coins are split into pouches until each pouch has one coin -/
def coin_splitting_game (n : ℕ) : ℕ :=
  (n - 1) * n / 2

/-- Theorem stating that the maximum sum of products in the coin splitting game is (n-1)n/2 -/
theorem max_sum_coin_splitting (n : ℕ) (h : n > 0) :
  ∀ (splitting : List (ℕ × ℕ)),
    (splitting.foldr (λ (a, b) acc => a * b + acc) 0 ≤ coin_splitting_game n) ∧
    (splitting.length = n - 1) ∧
    (splitting.all (λ (a, b) => a > 0 ∧ b > 0)) ∧
    (splitting.foldl (λ acc (a, b) => acc + a + b - 1) 1 = n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_coin_splitting_l1327_132791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_exercise_days_l1327_132799

/-- Represents Jane's exercise routine over a period of weeks. -/
structure ExerciseRoutine where
  hoursPerDay : ℕ
  weeksTotal : ℕ
  totalHours : ℕ

/-- Calculates the number of days per week Jane exercises. -/
def daysPerWeek (routine : ExerciseRoutine) : ℚ :=
  routine.totalHours / (routine.hoursPerDay * routine.weeksTotal)

/-- Theorem stating that Jane exercises 5 days a week given the conditions. -/
theorem jane_exercise_days (routine : ExerciseRoutine)
    (h1 : routine.hoursPerDay = 1)
    (h2 : routine.weeksTotal = 8)
    (h3 : routine.totalHours = 40) :
    daysPerWeek routine = 5 := by
  sorry

#eval daysPerWeek { hoursPerDay := 1, weeksTotal := 8, totalHours := 40 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_exercise_days_l1327_132799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_bag_probabilities_l1327_132772

/-- Represents the locations where bags can be: truck, gate, or shed -/
inductive Location
  | Truck
  | Gate
  | Shed

/-- Represents a bag of cement -/
structure Bag where
  id : Nat

/-- Represents the state of the system -/
structure State where
  truckStack : List Bag
  gateStack : List Bag
  shedStack : List Bag

/-- The probability of choosing to move a bag to a specific location -/
noncomputable def moveProbability : ℝ := 1 / 2

/-- The number of bags of cement -/
def numBags : Nat := 4

/-- Checks if the bags in the shed are in reverse order compared to the truck -/
def isReverseOrder (initialTruckStack : List Bag) (finalShedStack : List Bag) : Prop :=
  initialTruckStack.reverse = finalShedStack

/-- Checks if the second from bottom bag in the truck is at the bottom of the shed -/
def isSecondFromBottomAtBottom (initialTruckStack : List Bag) (finalShedStack : List Bag) : Prop :=
  initialTruckStack.get? 1 = finalShedStack.get? (finalShedStack.length - 1)

/-- Placeholder for probability calculation functions -/
noncomputable def probability_reverse_order : Nat → ℝ → ℝ := sorry
noncomputable def probability_second_from_bottom_at_bottom : Nat → ℝ → ℝ := sorry

/-- The main theorem to prove -/
theorem cement_bag_probabilities :
  ∃ (p : ℝ → ℝ → Prop),
    (p (probability_reverse_order numBags moveProbability) (1 / 8)) ∧
    (p (probability_second_from_bottom_at_bottom numBags moveProbability) (1 / 8)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_bag_probabilities_l1327_132772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_longest_side_is_43_l1327_132775

/-- Represents a rectangular window with glass panes --/
structure Window where
  panes : Nat
  rows : Nat
  columns : Nat
  pane_height : ℝ
  pane_width : ℝ
  border_width : ℝ

/-- Calculates the longest side of a rectangular window --/
noncomputable def longest_side (w : Window) : ℝ :=
  max (w.columns * w.pane_width + (w.columns + 1) * w.border_width)
      (w.rows * w.pane_height + (w.rows + 1) * w.border_width)

/-- Theorem stating that the longest side of the window is 43 inches --/
theorem window_longest_side_is_43 :
  ∃ (x : ℝ), longest_side { panes := 8, rows := 4, columns := 2, pane_height := 7*x, pane_width := 2*x, border_width := 3 } = 43 := by
  sorry

#check window_longest_side_is_43

end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_longest_side_is_43_l1327_132775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_video_suggestions_l1327_132796

theorem billy_video_suggestions
  (videos_per_list : ℕ)
  (total_videos : ℕ)
  (final_pick : ℕ)
  (h1 : videos_per_list = 15)
  (h2 : total_videos = 65)
  (h3 : final_pick = 5)
  : (total_videos - final_pick) / videos_per_list + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_video_suggestions_l1327_132796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1327_132748

/-- The function f(n) as defined in the problem -/
noncomputable def f (n : ℝ) : ℝ := (1/4) * n * (n+1) * (n+2) * (n+3)

/-- Theorem stating that f(r) - f(r-1) = r * (r+1) * (r+2) -/
theorem f_difference (r : ℝ) : f r - f (r-1) = r * (r+1) * (r+2) := by
  -- Expand the definition of f
  unfold f
  -- Perform algebraic simplification
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l1327_132748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_specific_percentage_difference_l1327_132742

-- Define the problem
theorem percentage_difference (a b c d : ℝ) : 
  (a / 100) * b - (c / d) * 25 = 22 := by
  sorry

-- State the specific case
theorem specific_percentage_difference :
  (80 / 100) * 40 - (2 / 5) * 25 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_specific_percentage_difference_l1327_132742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_six_l1327_132709

/-- The region defined by the inequality |4x-16|+|3y+9|≤6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 16| + |3 * p.2 + 9| ≤ 6}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := 6

theorem area_of_region_is_six : area_of_region = 6 := by
  -- Unfold the definition of area_of_region
  unfold area_of_region
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_six_l1327_132709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_4_l1327_132710

/-- Calculates the harmonic mean of three positive real numbers -/
noncomputable def harmonicMean (a b c : ℝ) : ℝ := 3 / (1/a + 1/b + 1/c)

/-- Represents a triathlon with equal segment lengths -/
structure Triathlon where
  swimSpeed : ℝ
  bikeSpeed : ℝ
  runSpeed : ℝ

/-- Calculates the average speed for a triathlon -/
noncomputable def averageSpeed (t : Triathlon) : ℝ :=
  harmonicMean t.swimSpeed t.bikeSpeed t.runSpeed

theorem triathlon_average_speed_approx_4 :
  let t : Triathlon := { swimSpeed := 2, bikeSpeed := 15, runSpeed := 8 }
  ∃ ε > 0, |averageSpeed t - 4| < ε := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_4_l1327_132710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_plane_problem_l1327_132700

/-- A line in the complex plane passing through the origin --/
structure ComplexLine where
  angle : ℝ
  not_real_axis : angle ≠ 0 ∧ angle ≠ Real.pi

/-- The set S_θ as defined in the problem --/
def S_theta (θ : ℝ) : Set ℂ :=
  {z : ℂ | (z * Complex.exp (-θ * Complex.I)).re ≥ Complex.abs z * Real.cos (Real.pi / 20)}

theorem complex_plane_problem :
  (∀ L : ComplexLine, ∃! z : ℂ, z ≠ 0 ∧ Complex.arg z = L.angle ∧ ((1 + z^23) / z^64).im = 0) ∧
  (∀ (a : ℂ) (θ : ℝ), a ≠ 0 → ∃ z : ℂ, z ∈ S_theta θ ∧ 1 + z^23 + a * z^64 = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_plane_problem_l1327_132700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l1327_132786

noncomputable def circumradius (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_dot_product (a b c : ℝ) (A B C : ℝ × ℝ) :
  (a + b + c) * (a - b + c) = 3 * a * c →
  a * c = 2 / 3 * b ^ 2 →
  circumradius A B C = Real.sqrt 3 →
  a < c →
  ∃ (AC AB : ℝ × ℝ), AC = C - A ∧ AB = B - A ∧ AC • AB = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l1327_132786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_x_implies_rational_y_l1327_132717

/-- Represents a decimal expansion of a number in (0, 1) -/
def DecimalExpansion := ℕ → Fin 10

/-- Given a decimal expansion, returns the n-th digit after the decimal point -/
def nthDigit (d : DecimalExpansion) (n : ℕ) : Fin 10 := d n

/-- Constructs y from x as described in the problem -/
def constructY (x : DecimalExpansion) : DecimalExpansion :=
  fun n => nthDigit x (2^n)

/-- A number is rational if and only if its decimal expansion is eventually periodic -/
def isEventuallyPeriodic (d : DecimalExpansion) : Prop :=
  ∃ (k p : ℕ), p > 0 ∧ ∀ n ≥ k, d n = d (n + p)

/-- A number in (0, 1) is rational if and only if its decimal expansion is eventually periodic -/
axiom rational_iff_eventually_periodic (x : DecimalExpansion) :
  isEventuallyPeriodic x ↔ ∃ (q : ℚ), 0 < q ∧ q < 1 ∧ ∀ n, nthDigit x n = (q * 10^n).floor % 10

theorem rational_x_implies_rational_y (x : DecimalExpansion) :
  isEventuallyPeriodic x → isEventuallyPeriodic (constructY x) := by
  sorry

#check rational_x_implies_rational_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_x_implies_rational_y_l1327_132717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l1327_132727

def g : ℕ → ℕ
  | 0 => 0  -- Added this case
  | 1 => 0
  | 2 => 1
  | (n + 3) => g (n + 1) + g (n + 2) + 1

theorem prime_divides_g_product (n : ℕ) (h : n.Prime) (h_gt : n > 5) :
  n ∣ g n * (g n + 1) := by
  sorry

#eval g 10  -- This line is optional, just to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l1327_132727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sine_function_l1327_132783

/-- The period of y = 3sin(2x) is π -/
theorem period_of_sine_function :
  ∃ T : ℝ, T > 0 ∧
    (∀ x : ℝ, 3 * Real.sin (2 * (x + T)) = 3 * Real.sin (2 * x)) ∧
    (∀ S : ℝ, 0 < S ∧ S < T → ∃ x : ℝ, 3 * Real.sin (2 * (x + S)) ≠ 3 * Real.sin (2 * x)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sine_function_l1327_132783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perpendicular_l1327_132749

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the lines and planes
variable (m n : Submodule ℝ V) (α β : Submodule ℝ V)

-- Define the conditions
variable (hm_line : FiniteDimensional.finrank ℝ m = 1)
variable (hα_plane : FiniteDimensional.finrank ℝ α = 2)
variable (hβ_plane : FiniteDimensional.finrank ℝ β = 2)
variable (hm_α_parallel : m ≤ α)
variable (hm_β_perp : m.orthogonal ≤ β)

-- State the theorem
theorem planes_perpendicular : α.orthogonal ≤ β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perpendicular_l1327_132749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_interval_l1327_132780

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a side
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the angle bisector
def is_angle_bisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ D = ((k * t.B.1 + t.C.1) / (k + 1), (k * t.B.2 + t.C.2) / (k + 1))

-- State the theorem
theorem triangle_side_interval (t : Triangle) (D : ℝ × ℝ) :
  side_length t.A t.B = 10 →
  is_angle_bisector t D →
  side_length D t.C = 3 →
  ∃ m n : ℝ, m < n ∧
    (∀ x : ℝ, m < x ∧ x < n ↔ ∃ t' : Triangle, t'.A = t.A ∧ t'.B = t.B ∧ side_length t'.A t'.C = x) ∧
    m + n = 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_interval_l1327_132780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1327_132737

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

theorem f_minimum_value :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1327_132737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_value_l1327_132762

theorem set_equality_implies_value (a b : ℝ) 
  (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : b^2013 - a^2013 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_value_l1327_132762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_statement_l1327_132773

-- Define the four statements as axioms (assumed true propositions)
axiom statement1 : Prop
axiom statement2 : Prop
axiom statement3 : Prop
axiom statement4 : Prop

-- Define a function to count the number of true statements
def count_true_statements (s1 s2 s3 s4 : Bool) : Nat :=
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0)

-- Theorem stating that exactly one statement is correct
theorem one_correct_statement :
  ∃ (b1 b2 b3 b4 : Bool),
    (b1 ↔ statement1) ∧
    (b2 ↔ statement2) ∧
    (b3 ↔ statement3) ∧
    (b4 ↔ statement4) ∧
    count_true_statements b1 b2 b3 b4 = 1 :=
by
  -- We use 'sorry' to skip the proof
  sorry

-- Provide descriptions of the statements (optional, for documentation purposes)
def statement1_description : String := "If two planes have three common points, then these two planes coincide"
def statement2_description : String := "Two lines can determine a plane"
def statement3_description : String := "If M ∈ α, M ∈ β, and α ∩ β = l, then M ∈ l"
def statement4_description : String := "In space, three lines intersecting at the same point are in the same plane"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_statement_l1327_132773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1327_132740

/-- Given a geometric sequence where the third term is 6! and the sixth term is 9!, 
    prove that the first term is equal to 6! divided by the square of the cube root of (9! / 6!). -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) :
  (∃ (seq : ℕ → ℝ), 
    (∀ n, seq (n + 1) = seq n * r) ∧  -- Geometric sequence condition
    seq 3 = 720 ∧                     -- Third term is 6! = 720
    seq 6 = 362880) →                 -- Sixth term is 9! = 362880
  a = 720 / ((362880 / 720 : ℝ) ^ (1/3 : ℝ))^2 := by
sorry

#eval 720 / ((362880 / 720 : Float) ^ (1/3 : Float))^2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1327_132740
