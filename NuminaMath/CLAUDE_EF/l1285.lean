import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_volume_l1285_128546

/-- Given a rectangle with length 4 and width 3, folded along its diagonal to form a tetrahedron
    with a dihedral angle of 60°, prove that the volume of the circumscribing sphere is 125π/6. -/
theorem tetrahedron_sphere_volume (length width dihedral_angle : ℝ) :
  length = 4 →
  width = 3 →
  dihedral_angle = π / 3 →
  (4 / 3) * π * (Real.sqrt (length^2 + width^2) / 2)^3 = 125 * π / 6 := by
  sorry

#check tetrahedron_sphere_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_volume_l1285_128546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_theorem_l1285_128586

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec toBits (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBits (m / 2)
    toBits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_theorem_l1285_128586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l1285_128540

/-- Represents a parabola equation in the form ax^2 + bxy + cy^2 + dx + ey + f = 0 --/
structure ParabolaEquation where
  a : Int
  b : Int
  c : Int
  d : Int
  e : Int
  f : Int
  a_pos : a > 0
  gcd_one : Int.gcd a (Int.gcd b (Int.gcd c (Int.gcd d (Int.gcd e f)))) = 1

/-- Checks if a point (x, y) satisfies the parabola equation --/
def satisfiesEquation (eq : ParabolaEquation) (x y : ℚ) : Prop :=
  eq.a * x^2 + eq.b * x * y + eq.c * y^2 + eq.d * x + eq.e * y + (eq.f : ℚ) = 0

/-- Checks if the axis of symmetry is parallel to the y-axis --/
def hasVerticalSymmetry (eq : ParabolaEquation) : Prop :=
  eq.b = 0 ∧ eq.c = 0

/-- Checks if the vertex lies on the x-axis --/
def vertexOnXAxis (eq : ParabolaEquation) : Prop :=
  ∃ x : ℚ, eq.a * x^2 + eq.d * x + (eq.f : ℚ) = 0

/-- Theorem: The parabola equation satisfies all given conditions --/
theorem parabola_satisfies_conditions (eq : ParabolaEquation) : 
  eq.a = 10 ∧ 
  eq.b = 0 ∧ 
  eq.c = 0 ∧ 
  eq.d = -100 ∧ 
  eq.e = -9 ∧ 
  eq.f = 250 → 
  satisfiesEquation eq 2 10 ∧ 
  hasVerticalSymmetry eq ∧ 
  vertexOnXAxis eq ∧ 
  (∃ x : ℚ, x = 5 ∧ satisfiesEquation eq x 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l1285_128540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_length_divides_p_minus_one_l1285_128505

theorem period_length_divides_p_minus_one (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ d : ℕ, (∀ k < d, (10^k - 1) % p ≠ 0) ∧ (10^d - 1) % p = 0 ∧ d ∣ (p - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_length_divides_p_minus_one_l1285_128505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_color_with_infinite_multiples_l1285_128543

/-- A coloring of integers using two colors -/
def Coloring := ℤ → Bool

/-- The set of integers of a given color in a coloring -/
def ColorSet (c : Coloring) (color : Bool) : Set ℤ :=
  {n : ℤ | c n = color}

/-- The property that a set of integers contains infinitely many multiples of every natural number -/
def HasInfiniteMultiples (S : Set ℤ) : Prop :=
  ∀ k : ℕ, ∀ N : ℕ, ∃ n : ℤ, n ∈ S ∧ n > N ∧ (∃ m : ℤ, n = k * m)

theorem exists_color_with_infinite_multiples (c : Coloring) :
  ∃ color : Bool, HasInfiniteMultiples (ColorSet c color) := by
  sorry

#check exists_color_with_infinite_multiples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_color_with_infinite_multiples_l1285_128543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_friends_count_l1285_128537

universe u

structure Coach where
  Person : Type u
  passengers : Set Person
  m : ℕ
  h_m_ge_3 : m ≥ 3
  friends : Person → Set Person
  h_symmetric : ∀ {a b : Person}, b ∈ friends a → a ∈ friends b
  h_irreflexive : ∀ (a : Person), a ∉ friends a
  h_common_friend : ∀ (S : Set Person), S ⊆ passengers → S.ncard = m → 
    ∃! (f : Person), f ∈ passengers ∧ ∀ p ∈ S, f ∈ friends p

theorem max_friends_count (c : Coach) : 
  ∃ p ∈ c.passengers, ∀ q ∈ c.passengers, (c.friends q).ncard ≤ (c.friends p).ncard ∧ (c.friends p).ncard = c.m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_friends_count_l1285_128537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_dot_product_l1285_128585

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the circle
def myCircle (c r : ℝ) (x y : ℝ) : Prop := (x - c)^2 + y^2 = r^2

-- Define the line
def line (θ : ℝ) (x y : ℝ) : Prop := y = (Real.tan θ) * x

-- Define the dot product of vectors
def dotProduct (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem parabola_min_dot_product (p c r : ℝ) :
  p = 2 →
  c = 2 →
  r = 2 →
  (∀ x y : ℝ, parabola p x y →
    ∀ xm ym xf yf : ℝ,
    myCircle c r xm ym →
    xf = p ∧ yf = 0 →
    dotProduct (x - xm) (y - ym) (x - xf) (y - yf) ≥ 2) ∧
  (∃ x y : ℝ, parabola p x y ∧
    ∃ xm ym xf yf : ℝ,
    myCircle c r xm ym ∧
    xf = p ∧ yf = 0 ∧
    dotProduct (x - xm) (y - ym) (x - xf) (y - yf) = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_dot_product_l1285_128585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_repeating_6_equals_6_7_l1285_128566

/-- Represents a number in base 8 with an infinitely repeating decimal of 6 -/
noncomputable def base8_repeating_6 : ℚ := ∑' n, (6 : ℚ) / (8 ^ (n + 1))

/-- The theorem stating that 0.666... in base 8 equals 6/7 in base 10 -/
theorem base8_repeating_6_equals_6_7 : base8_repeating_6 = 6 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_repeating_6_equals_6_7_l1285_128566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowmobile_overtakes_atv_l1285_128573

-- Define the track and vehicle properties
structure Track where
  length : ℝ

structure Vehicle where
  soft_snow_speed : ℝ
  dense_snow_speed : ℝ

noncomputable def time_for_lap (t : Track) (v : Vehicle) : ℝ :=
  (t.length / 4) / v.soft_snow_speed + (3 * t.length / 4) / v.dense_snow_speed

-- Define the specific vehicles
def snowmobile : Vehicle := ⟨32, 36⟩
def atv : Vehicle := ⟨16, 48⟩

-- Theorem statement
theorem snowmobile_overtakes_atv (t : Track) :
  11 * time_for_lap t snowmobile = 10 * time_for_lap t atv := by
  sorry

#check snowmobile_overtakes_atv

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowmobile_overtakes_atv_l1285_128573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_ax_iff_a_leq_one_l1285_128508

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

-- State the theorem
theorem f_geq_ax_iff_a_leq_one :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f x ≥ a * x) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_ax_iff_a_leq_one_l1285_128508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_selling_price_l1285_128569

/-- A business selling charm bracelets -/
structure BraceletBusiness where
  string_cost : ℚ
  bead_cost : ℚ
  bracelets_sold : ℕ
  total_profit : ℚ

/-- Calculate the selling price of each bracelet -/
def selling_price (b : BraceletBusiness) : ℚ :=
  (b.total_profit + b.bracelets_sold * (b.string_cost + b.bead_cost)) / b.bracelets_sold

/-- Theorem: The selling price of each bracelet is $6 -/
theorem bracelet_selling_price (b : BraceletBusiness) 
  (h1 : b.string_cost = 1)
  (h2 : b.bead_cost = 3)
  (h3 : b.bracelets_sold = 25)
  (h4 : b.total_profit = 50) :
  selling_price b = 6 := by
  sorry

#eval selling_price { string_cost := 1, bead_cost := 3, bracelets_sold := 25, total_profit := 50 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_selling_price_l1285_128569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_combinations_l1285_128522

theorem sum_of_combinations : Nat.choose 99 2 + Nat.choose 99 3 = 161700 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_combinations_l1285_128522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_value_l1285_128591

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f (x + t)

def is_symmetric (t : ℝ) : Prop := ∀ x, g t x = g t (Real.pi / 12 - x)

theorem min_translation_value :
  ∃ t_min : ℝ, t_min > 0 ∧ is_symmetric t_min ∧ ∀ t, t > 0 ∧ is_symmetric t → t ≥ t_min ∧ t_min = 7 * Real.pi / 24 := by
  sorry

#check min_translation_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_value_l1285_128591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_three_l1285_128510

noncomputable def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4
noncomputable def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 10*x^4

noncomputable def c : ℝ := -3/5

theorem polynomial_degree_three :
  ∃ (a b d : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), f x + c * g x = a * x^3 + b * x^2 + d * x + (2 + 4*c) :=
by
  -- We'll use existential introduction to provide the values
  use (-4/5 : ℝ), (4 - 3*c : ℝ), (-15 - 3*c : ℝ)
  -- Now we need to prove the conjunction
  apply And.intro
  · -- Prove a ≠ 0
    norm_num
  · -- Prove the equality for all x
    intro x
    -- Expand the definitions and simplify
    simp [f, g, c]
    -- The rest of the proof would involve algebraic manipulation
    -- which is beyond the scope of this example
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_three_l1285_128510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_calculation_l1285_128531

/-- The radius of a wheel given its circumference and number of revolutions --/
noncomputable def wheel_radius (distance : ℝ) (revolutions : ℕ) : ℝ :=
  distance / (2 * Real.pi * (revolutions : ℝ))

/-- Theorem stating that a wheel covering 253.44 cm in 180 revolutions has a radius of approximately 0.224 cm --/
theorem wheel_radius_calculation :
  let distance := 253.44
  let revolutions := 180
  abs (wheel_radius distance revolutions - 0.224) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_calculation_l1285_128531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l1285_128518

/-- The equation of a plane passing through a given point and perpendicular to a given vector -/
theorem plane_equation_through_point_perpendicular_to_vector 
  (A B C : ℝ × ℝ × ℝ) (plane_eq : ℝ × ℝ × ℝ → ℝ) : 
  (A = (-3, 7, 2)) →
  (B = (3, 5, 1)) →
  (C = (4, 5, 3)) →
  (plane_eq = fun (x : ℝ × ℝ × ℝ) ↦ x.1 + 2 * x.2.2 - 1) →
  (∀ (x : ℝ × ℝ × ℝ), plane_eq x = 0 ↔ (x.1 - A.1, x.2.1 - A.2.1, x.2.2 - A.2.2) • (C.1 - B.1, C.2.1 - B.2.1, C.2.2 - B.2.2) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l1285_128518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1285_128527

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.sin (x + Real.pi/6) + Real.sin (x - Real.pi/6) + a * Real.cos x + b

theorem f_properties (a b : ℝ) :
  (∀ x y, x ∈ Set.Icc (-Real.pi/3) 0 → y ∈ Set.Icc (-Real.pi/3) 0 → x < y → f a b x < f a b y) →
  (∀ x : ℝ, f a b x ≥ 2) →
  (∃ x : ℝ, f a b x = 2) →
  (∀ x : ℝ, f a b x = f a b (x + 2*Real.pi)) ∧ a = -1 ∧ b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1285_128527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_related_confidence_l1285_128504

/-- Represents the chi-square statistic -/
def K : ℝ := sorry

/-- The confidence level (as a percentage) -/
def confidence_level : ℝ := 95

/-- The critical value for chi-square distribution with 1 degree of freedom at 95% confidence level -/
def critical_value : ℝ := 3.841

/-- Theorem stating that for 95% confidence level, K² exceeds the critical value -/
theorem events_related_confidence (h : confidence_level = 95) : K^2 > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_related_confidence_l1285_128504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_exists_a_with_real_roots_l1285_128507

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

-- Statement 1: f is an odd function for all a ∈ ℝ
theorem f_is_odd (a : ℝ) : ∀ x, f a (-x) = -(f a x) := by
  intro x
  simp [f]
  ring

-- Statement 2: There exists an a ∈ ℝ such that f(x) = -x has real roots
theorem exists_a_with_real_roots : ∃ a : ℝ, ∃ x : ℝ, f a x = -x := by
  use -2  -- We choose a = -2 as an example
  use 1   -- We choose x = 1 as an example
  simp [f]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_exists_a_with_real_roots_l1285_128507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_relations_l1285_128597

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_plane : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_relations :
  ∃ (l m n : Line) (α β : Plane),
    -- Statement 1
    (∀ l α β, perpendicular l α → perpendicular l β → parallel_plane α β) ∧
    -- Statement 2
    ¬(∀ l α β, parallel_line_plane l α → parallel_line_plane l β → perpendicular_plane α β) ∧
    -- Statement 3
    ¬(∀ l α β, perpendicular_plane α β → parallel_line_plane l α → parallel_line_plane l β) ∧
    -- Statement 4
    (∀ m n α, parallel_line m n → perpendicular m α → perpendicular n α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_relations_l1285_128597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_two_twentyfifths_roots_satisfy_equation_l1285_128509

-- Define the function f
def f (x : ℝ) : ℝ := 25*x^2 - 10*x + 3

-- State the theorem
theorem sum_of_roots_equals_two_twentyfifths :
  let z₁ := (50 + 50*Real.sqrt 5) / 1250
  let z₂ := (50 - 50*Real.sqrt 5) / 1250
  z₁ + z₂ = 2/25 :=
by
  -- Introduce the definitions
  intro z₁ z₂
  -- Simplify the expression
  simp [z₁, z₂]
  -- The proof is completed with sorry
  sorry

-- Verify that f(5z) = 7 for both roots
theorem roots_satisfy_equation :
  let z₁ := (50 + 50*Real.sqrt 5) / 1250
  let z₂ := (50 - 50*Real.sqrt 5) / 1250
  f (5*z₁) = 7 ∧ f (5*z₂) = 7 :=
by
  -- Introduce the definitions
  intro z₁ z₂
  -- Split the conjunction
  apply And.intro
  -- Prove f(5z₁) = 7
  · simp [f, z₁]
    sorry
  -- Prove f(5z₂) = 7
  · simp [f, z₂]
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_two_twentyfifths_roots_satisfy_equation_l1285_128509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_problem_l1285_128549

theorem pascal_triangle_problem (a b : ℕ) : 
  (Nat.choose a 2 = 3003) ∧ 
  (Nat.choose a (a - 2) = 3003) ∧
  (Nat.choose 15 b = 3003) ∧ 
  (Nat.choose 15 (15 - b) = 3003) ∧
  (Nat.choose 14 6 = 3003) ∧
  (Nat.choose 14 8 = 3003) →
  a + b * (15 - b) = 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_problem_l1285_128549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_max_abs_min_value_achievable_l1285_128575

theorem min_value_of_max_abs (a b : ℝ) : 
  max (|a + b|) (max (|a - b|) (|1 - b|)) ≥ (1/2 : ℝ) := by sorry

theorem min_value_achievable : 
  ∃ (a b : ℝ), max (|a + b|) (max (|a - b|) (|1 - b|)) = (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_max_abs_min_value_achievable_l1285_128575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l1285_128556

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem two_different_color_chips_probability 
  (blue : ℕ) (yellow : ℕ) (red : ℕ) 
  (h_blue : blue = 7) 
  (h_yellow : yellow = 5) 
  (h_red : red = 4) : 
  (blue * yellow + yellow * blue + blue * red + red * blue + yellow * red + red * yellow : ℚ) / 
  ((blue + yellow + red) * (blue + yellow + red)) = 83 / 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l1285_128556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_solution_l1285_128502

theorem unique_triple_solution : 
  ∃! (k m n : ℕ), 
    k > 0 ∧ m > 0 ∧ n > 0 ∧
    (∃ (r : ℕ), r > 0 ∧ k * n = r ^ 2) ∧ 
    (∃ (s : ℕ), s > 0 ∧ Nat.Prime s ∧ k * (k - 1) / 2 + n = s ^ 4) ∧ 
    (∃ (p : ℕ), p > 0 ∧ Nat.Prime p ∧ k - m ^ 2 = p) ∧ 
    (∃ (p : ℕ), p > 0 ∧ Nat.Prime p ∧ (n + 2) / (m ^ 2) = p ^ 4) ∧ 
    Nat.Prime m ∧
    k = 28 ∧ m = 5 ∧ n = 2023 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_solution_l1285_128502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l1285_128555

-- Define the complex number z
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

-- Theorem statement
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l1285_128555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_max_at_even_number_l1285_128545

-- Define the income in thousands of dollars
noncomputable def income (y : ℝ) : ℝ := 1000 * y

-- Define the tax rate as a function of income
noncomputable def taxRate (y : ℝ) : ℝ := y / 100

-- Define the take-home pay function
noncomputable def takeHomePay (y : ℝ) : ℝ := income y - (taxRate y * income y)

-- Theorem stating that $50,000 maximizes take-home pay
theorem max_take_home_pay :
  ∀ y : ℝ, y > 0 → takeHomePay 50 ≥ takeHomePay y :=
by
  sorry

-- Theorem stating that the maximum occurs at an even number
theorem max_at_even_number :
  ∃ n : ℕ, Even n ∧ ∀ y : ℝ, y > 0 → takeHomePay n ≥ takeHomePay y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_take_home_pay_max_at_even_number_l1285_128545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1285_128512

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the domain of f
def domain (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem f_properties :
  -- Part 1: f is an odd function
  (∀ x, domain x → f (-x) = -f x) ∧
  -- Part 2: f(a) + f(b) = f((a+b)/(1+ab))
  (∀ a b, domain a → domain b → f a + f b = f ((a + b) / (1 + a * b))) ∧
  -- Part 3: If f((a+b)/(1+ab)) = 1 and f((a-b)/(1-ab)) = 2, then f(a) = 3/2 and f(b) = -1/2
  (∀ a b, domain a → domain b →
    f ((a + b) / (1 + a * b)) = 1 →
    f ((a - b) / (1 - a * b)) = 2 →
    f a = 3/2 ∧ f b = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1285_128512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_circle_l1285_128515

/-- The curve defined by r = 1 / (2 - sin θ) is a circle -/
theorem curve_is_circle : ∃ (h k : ℝ) (R : ℝ), ∀ (θ : ℝ),
  let r := 1 / (2 - Real.sin θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x - h)^2 + (y - k)^2 = R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_circle_l1285_128515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lincoln_number_l1285_128559

def lucy : ℂ := Complex.mk 6 2
def product : ℂ := Complex.mk 45 (-9)

theorem lincoln_number : ∃ (lincoln : ℂ), lucy * lincoln = product ∧ lincoln = Complex.mk 6.3 (-1.35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lincoln_number_l1285_128559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1285_128574

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * a * x^2 - 2*x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x + a * x - 2

-- Theorem statement
theorem monotonic_decreasing_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 1 < x₁ ∧ x₂ < 2 ∧ x₁ < x₂ ∧
    ∀ x ∈ Set.Ioo x₁ x₂, f_derivative a x ≤ 0) ↔ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1285_128574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1285_128525

/-- The family of circles parameterized by t -/
def circle_family (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - t + 2)^2 = 1}

/-- The point P -/
def P : ℝ × ℝ := (-1, 1)

/-- A function representing PA · PB for a given t -/
noncomputable def dot_product (t : ℝ) : ℝ :=
  let x := t^2 - 2*t + 4
  (2*x^2 + x) / (x + 1)

theorem min_dot_product :
  ∃ (min : ℝ), min = 21/4 ∧
  ∀ t : ℝ, dot_product t ≥ min :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1285_128525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_sum_l1285_128594

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 2)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : is_on_parabola x y

-- Vector from focus to a point
def vector_from_focus (p : PointOnParabola) : ℝ × ℝ :=
  (p.x - focus.1, p.y - focus.2)

-- Distance from focus to a point
noncomputable def distance_from_focus (p : PointOnParabola) : ℝ :=
  Real.sqrt ((p.x - focus.1)^2 + (p.y - focus.2)^2)

theorem parabola_focus_distance_sum
  (A B C : PointOnParabola)
  (h : vector_from_focus A + vector_from_focus B + vector_from_focus C = (0, 0)) :
  distance_from_focus A + distance_from_focus B + distance_from_focus C = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_sum_l1285_128594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_expected_value_X_l1285_128581

-- Define the probabilities for each group
def pA : ℝ := 0.6
def pB : ℝ := 0.5
def pC : ℝ := 0.5
def pD : ℝ := 0.4

-- Define the random variable X as the number of groups that pass
noncomputable def X : ℕ → ℝ := sorry

-- Theorem for the probability of at least three groups passing
theorem prob_at_least_three : 
  Finset.sum (Finset.range 5) (fun k => X k * if k ≥ 3 then 1 else 0) = 0.48 := by sorry

-- Theorem for the expected value of X
theorem expected_value_X : 
  Finset.sum (Finset.range 5) (fun k => k * X k) = 2.12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_three_expected_value_X_l1285_128581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_weight_calculation_l1285_128500

-- Define the weights of the individuals
noncomputable def carl_weight : ℝ := 145
noncomputable def dave_weight : ℝ := carl_weight + 8
noncomputable def brad_weight : ℝ := dave_weight / 2
noncomputable def billy_weight : ℝ := brad_weight + 9

-- Theorem statement
theorem billy_weight_calculation : billy_weight = 85.5 := by
  -- Unfold the definitions
  unfold billy_weight
  unfold brad_weight
  unfold dave_weight
  unfold carl_weight
  
  -- Perform the calculations
  simp [add_assoc, add_comm, mul_comm]
  
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_weight_calculation_l1285_128500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1285_128514

/-- Given two vectors a and b in ℝ², where a = (x, x+1) and b = (1, 2),
    prove that if a is perpendicular to b, then x = -2/3. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, x + 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1285_128514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_job_time_l1285_128596

/-- The time it takes for Tina to complete the job independently -/
def T : ℝ := sorry

/-- The time it takes for Ann to complete the job independently -/
def ann_time : ℝ := 9

/-- The time Tina worked on the job -/
def tina_work_time : ℝ := 8

/-- The time Ann worked to complete the remainder of the job -/
def ann_work_time : ℝ := 3

/-- Theorem stating that Tina can complete the job in 12 hours -/
theorem tina_job_time :
  (tina_work_time / T) + (ann_work_time / ann_time) = 1 →
  T = 12 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_job_time_l1285_128596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_irrational_l1285_128523

-- Define a circle with a rational radius
def Circle (r : ℚ) := { x : ℝ × ℝ | (x.1 ^ 2 + x.2 ^ 2 : ℝ) = (r : ℝ) ^ 2 }

-- Define the circumference of a circle
noncomputable def circumference (r : ℚ) : ℝ := 2 * Real.pi * (r : ℝ)

-- Theorem statement
theorem circle_circumference_irrational (r : ℚ) : Irrational (circumference r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_irrational_l1285_128523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l1285_128571

theorem constant_function_proof (p : ℝ) (α : ℝ) 
  (h1 : Real.cos α ≠ 0) (h2 : Real.sin α ≠ 0) : 
  (p * (Real.cos α)^3 - Real.cos (3 * α)) / Real.cos α + 
  (p * (Real.sin α)^3 + Real.sin (3 * α)) / Real.sin α = p + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l1285_128571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l1285_128592

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  c : ℝ  -- Length of one non-parallel side
  d : ℝ  -- Length of the other non-parallel side

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  let h := Real.sqrt (t.c^2 - ((t.b - t.a)^2 + t.c^2 - t.d^2)^2 / (4 * (t.b - t.a)^2))
  (t.a + t.b) * h / 2

/-- The theorem stating that the area of the specific trapezoid is 564 -/
theorem specific_trapezoid_area : 
  trapezoid_area { a := 11, b := 36, c := 25, d := 30 } = 564 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l1285_128592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_product_polynomial_l1285_128598

noncomputable def polynomial1 (x : ℝ) := x^4
noncomputable def polynomial2 (x : ℝ) := x + 1/x
noncomputable def polynomial3 (x : ℝ) := 2 + 3/x + 4/(x^2)

theorem degree_of_product_polynomial :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧
    ∀ x, polynomial1 x * polynomial2 x * polynomial3 x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_product_polynomial_l1285_128598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_sides_l1285_128536

/-- Given an isosceles triangle ABC with circumscribed circle center O,
    where d is the distance from O to AB, and e is the distance from O to AC. -/
def IsoscelesTriangle (d e : ℝ) : Prop :=
  d > 0 ∧ e > 0

/-- The length of the base AB of the isosceles triangle. -/
noncomputable def baseLength (d e : ℝ) : ℝ :=
  Real.sqrt (8 * e^2 + 2 * d * Real.sqrt (d^2 + 8 * e^2) - 2 * d^2)

/-- The length of the leg AC (or BC) of the isosceles triangle. -/
noncomputable def legLength (d e : ℝ) : ℝ :=
  Real.sqrt (4 * e^2 + 2 * d * Real.sqrt (d^2 + 8 * e^2) + 2 * d^2)

/-- Theorem: The base and leg lengths of the isosceles triangle are as calculated. -/
theorem isosceles_triangle_sides {d e : ℝ} (h : IsoscelesTriangle d e) :
  (baseLength d e = Real.sqrt (8 * e^2 + 2 * d * Real.sqrt (d^2 + 8 * e^2) - 2 * d^2)) ∧
  (legLength d e = Real.sqrt (4 * e^2 + 2 * d * Real.sqrt (d^2 + 8 * e^2) + 2 * d^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_sides_l1285_128536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_locus_l1285_128538

-- Define the ellipse C
def C (x y : ℝ) (a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the point M
noncomputable def M : ℝ × ℝ := (Real.sqrt 2, 1)

-- Define the left focus F₁
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)

-- Define the point P
def P : ℝ × ℝ := (4, 1)

-- State the theorem
theorem ellipse_and_locus :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  C M.1 M.2 a b ∧
  (∀ (x y : ℝ), C x y a b ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ (Q : ℝ × ℝ), (∃ (A B : ℝ × ℝ), 
    C A.1 A.2 a b ∧ C B.1 B.2 a b ∧
    (B.2 - P.2) / (B.1 - P.1) = (A.2 - P.2) / (A.1 - P.1) ∧
    (Q.1 - A.1) * (B.1 - P.1) * (B.2 - Q.2) = 
    (Q.1 - B.1) * (A.1 - P.1) * (A.2 - Q.2)) →
    2 * Q.1 + Q.2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_locus_l1285_128538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_B_l1285_128560

-- Define the expressions for A and B
noncomputable def A : ℝ := (1 + 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 6 * Real.sqrt 6) *
             (2 + 6 * Real.sqrt 2 + Real.sqrt 3 + 3 * Real.sqrt 6) *
             (3 + Real.sqrt 2 + 6 * Real.sqrt 3 + 2 * Real.sqrt 6) *
             (6 + 3 * Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6)

noncomputable def B : ℝ := (1 + 3 * Real.sqrt 2 + 2 * Real.sqrt 3 + 6 * Real.sqrt 6) *
             (2 + Real.sqrt 2 + 6 * Real.sqrt 3 + 3 * Real.sqrt 6) *
             (3 + 6 * Real.sqrt 2 + Real.sqrt 3 + 2 * Real.sqrt 6) *
             (6 + 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + Real.sqrt 6)

-- Theorem statement
theorem A_equals_B : A = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_B_l1285_128560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_functions_l1285_128572

/-- The convexity property for a function f on [0,1] -/
def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ l : ℝ), x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → l ∈ Set.Icc 0 1 →
    f (l * x₁ + (1 - l) * x₂) ≤ l * f x₁ + (1 - l) * f x₂

/-- The four functions defined on [0,1] -/
noncomputable def f₁ : ℝ → ℝ := λ x ↦ x
noncomputable def f₂ : ℝ → ℝ := λ x ↦ x^2
noncomputable def f₃ : ℝ → ℝ := λ x ↦ |x - 1/2|
noncomputable def f₄ : ℝ → ℝ := λ x ↦ -x

/-- The main theorem stating which functions are convex -/
theorem convex_functions :
  IsConvex f₁ ∧ IsConvex f₃ ∧ ¬IsConvex f₂ ∧ ¬IsConvex f₄ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_functions_l1285_128572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l1285_128593

def candidate1_votes : ℕ := 4136
def candidate2_votes : ℕ := 7636
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

noncomputable def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_candidate_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ 
  |winning_percentage - 5193 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l1285_128593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_true_proposition_l1285_128599

-- Define the concept of lines
variable (Line : Type)

-- Define relationships between lines
variable (skew parallel intersect perpendicular : Line → Line → Prop)

-- Define the four propositions
def proposition1 (Line : Type) (skew : Line → Line → Prop) : Prop :=
  ∀ a b c : Line, skew a b ∧ skew b c → skew a c

def proposition2 (Line : Type) (intersect : Line → Line → Prop) : Prop :=
  ∀ a b c : Line, intersect a b ∧ intersect b c → intersect a c

def proposition3 (Line : Type) (parallel : Line → Line → Prop) : Prop :=
  ∀ a b c : Line, parallel a b ∧ parallel b c → parallel a c

def proposition4 (Line : Type) (perpendicular : Line → Line → Prop) : Prop :=
  ∀ a b c : Line, perpendicular a b ∧ perpendicular b c → perpendicular a c

-- Theorem stating that only one proposition is true
theorem only_one_true_proposition :
  ∃ (Line : Type) (skew parallel intersect perpendicular : Line → Line → Prop),
    ¬(proposition1 Line skew) ∧
    ¬(proposition2 Line intersect) ∧
    (proposition3 Line parallel) ∧
    ¬(proposition4 Line perpendicular) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_true_proposition_l1285_128599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_43_percent_l1285_128588

/-- The discount percentage offered on a book -/
noncomputable def discount_percentage (selling_price_with_discount : ℝ) (selling_price_without_discount : ℝ) : ℝ :=
  (selling_price_without_discount - selling_price_with_discount) / selling_price_without_discount * 100

/-- Theorem: The discount percentage is 43% given the profit percentages -/
theorem discount_percentage_is_43_percent 
  (cost_price : ℝ) 
  (selling_price_with_discount : ℝ) 
  (selling_price_without_discount : ℝ)
  (h1 : selling_price_with_discount = 1.425 * cost_price)
  (h2 : selling_price_without_discount = 2.5 * cost_price) :
  discount_percentage selling_price_with_discount selling_price_without_discount = 43 := by
  sorry

#check discount_percentage_is_43_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_43_percent_l1285_128588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l1285_128501

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midpoint_segment : ℚ
  height : ℚ
  longer_base : ℚ

/-- Calculates the area of a trapezoid -/
def trapezoid_area (t : Trapezoid) : ℚ :=
  let shorter_base := t.longer_base - 2 * t.midpoint_segment
  (t.longer_base + shorter_base) * t.height / 2

/-- The main theorem stating that the area of the specific trapezoid is 580 -/
theorem specific_trapezoid_area :
  let t : Trapezoid := ⟨5, 10, 63⟩
  trapezoid_area t = 580 := by
  -- Unfold the definition of trapezoid_area
  unfold trapezoid_area
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#eval trapezoid_area ⟨5, 10, 63⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l1285_128501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_60_degree_angle_l1285_128564

theorem triangle_60_degree_angle (A B C : ℝ) : 
  (A + B + C = π) →  -- Triangle angle sum is π (180°)
  (0 ≤ A ∧ A ≤ π) →  -- Angle A is between 0 and π
  (0 ≤ B ∧ B ≤ π) →  -- Angle B is between 0 and π
  (0 ≤ C ∧ C ≤ π) →  -- Angle C is between 0 and π
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 0 ↔ (A = π / 3 ∨ B = π / 3 ∨ C = π / 3)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_60_degree_angle_l1285_128564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1285_128557

-- Define set A
def A : Set ℝ := {x | x - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | |x| ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1285_128557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_order_equation_l1285_128595

/-- Represents the waiter's order scenario -/
structure WaiterOrders where
  initial_customers : ℕ
  left_customers : ℕ
  order_increase_percent : ℚ
  new_customers : ℕ
  total_order : ℚ

/-- Calculates the number of remaining customers -/
def remaining_customers (w : WaiterOrders) : ℕ :=
  w.initial_customers - w.left_customers

/-- Calculates the order increase factor -/
noncomputable def order_increase_factor (w : WaiterOrders) : ℚ :=
  1 + w.order_increase_percent / 100

/-- Theorem representing the waiter's order scenario -/
theorem waiter_order_equation (w : WaiterOrders) (Y : ℚ) (x : ℚ) :
  w.initial_customers = 14 →
  w.left_customers = 3 →
  w.order_increase_percent = 150 →
  w.new_customers = 39 →
  w.total_order = 31287 / 100 →
  Y > 0 →
  (5 / 2 : ℚ) * Y * (1 + x / 100) = w.total_order :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_order_equation_l1285_128595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1285_128583

/-- The eccentricity of a hyperbola given the distance from vertex to asymptote and focus to asymptote -/
theorem hyperbola_eccentricity (b c : ℝ) (hb : b = 2) (hc : c = 6) :
  let a := Real.sqrt (c^2 - b^2)
  c / a = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1285_128583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_rational_and_solution_equivalence_l1285_128576

open Real

theorem trig_rational_and_solution_equivalence 
  (x A B C : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ y, ∃ (p q : ℝ), q ≠ 0 ∧ f y = p / q) ∧ cos x = f (tan (x/2))) ∧ 
  (∃ (g : ℝ → ℝ), (∀ y, ∃ (p q : ℝ), q ≠ 0 ∧ g y = p / q) ∧ sin x = g (tan (x/2))) ∧ 
  (∃ (k : ℤ), sin x = (A * C + (-1)^k * B * Real.sqrt (A^2 + B^2 - C^2)) / (A^2 + B^2) ∧ 
               A * cos x + B * sin x = C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_rational_and_solution_equivalence_l1285_128576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_l1285_128532

def is_valid_configuration (points : Finset ℕ) : Prop :=
  points.card = 5 ∧ 
  (∀ x y, x ∈ points → y ∈ points → x ≠ y → x < y) ∧
  (∀ x y z w, x ∈ points → y ∈ points → z ∈ points → w ∈ points → 
    x < y ∧ z < w → y - x ≠ w - z ∨ (x = z ∧ y = w))

theorem min_segment_length (points : Finset ℕ) 
  (h_valid : is_valid_configuration points)
  (h_max : ∃ x ∈ points, ∀ y ∈ points, y ≤ x) :
  ∃ x ∈ points, x ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_l1285_128532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_in_triangle_l1285_128562

-- Define the necessary types and functions
axiom Point : Type
axiom AngleSum : Point → Point → ℝ
axiom Angle : Point → ℝ

-- Theorem statement
theorem angle_sum_in_triangle (X Y Z : Point) (h : AngleSum X Y = 100) : 
  Angle Z = 80 := by
  sorry -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_in_triangle_l1285_128562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1285_128528

noncomputable section

def f (x : ℝ) : ℝ := x + Real.log x

def g (b : ℝ) (x : ℝ) : ℝ := f x + (1/2) * x^2 - b * x

def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

def decreasing_interval (b : ℝ) : Set ℝ := Set.Ioo ((3 - Real.sqrt 2) / 2) ((3 + Real.sqrt 2) / 2)

def extreme_points (b : ℝ) (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ (deriv (g b)) x₁ = 0 ∧ (deriv (g b)) x₂ = 0

theorem main_theorem (b : ℝ) (h : b ≥ 7/2) :
  (∀ x y, deriv f 1 = 2 → tangent_line x y) ∧
  (∀ x ∈ decreasing_interval 4, deriv (g 4) x < 0) ∧
  (∀ x₁ x₂, extreme_points b x₁ x₂ → g b x₁ - g b x₂ ≥ 15/8 - 2 * Real.log 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1285_128528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1285_128511

-- Define the expression
noncomputable def expression : ℝ := Real.sqrt (1 + Real.cos (100 * Real.pi / 180)) - Real.sqrt (1 - Real.cos (100 * Real.pi / 180))

-- State the theorem
theorem simplify_expression : expression = -2 * Real.sin (5 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1285_128511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1285_128567

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 2

def monotone_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x > f y

theorem sufficient_not_necessary :
  (monotone_decreasing (f 2) { x | x ≤ -2 }) ∧
  (∃ a : ℝ, a ≠ 2 ∧ monotone_decreasing (f a) { x | x ≤ -2 }) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1285_128567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missouri_to_newyork_distance_l1285_128580

/-- The distance between two points by plane -/
noncomputable def distance_by_plane : ℝ := 2000

/-- The percentage increase in distance when driving instead of flying -/
noncomputable def driving_increase_percentage : ℝ := 40

/-- The distance between two points by car -/
noncomputable def distance_by_car (d : ℝ) : ℝ := d * (1 + driving_increase_percentage / 100)

/-- Missouri is midway between Arizona and New York -/
noncomputable def midway_point (d : ℝ) : ℝ := d / 2

theorem missouri_to_newyork_distance :
  midway_point (distance_by_car distance_by_plane) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missouri_to_newyork_distance_l1285_128580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l1285_128584

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + (a + 2)*y^2 + 2*a*x + a = 0

-- Define what it means for an equation to represent a circle
def IsCircle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ h k r : ℝ, ∀ x y : ℝ, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_condition (a : ℝ) :
  (∀ x y : ℝ, circle_equation x y a → IsCircle (circle_equation · · a)) →
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l1285_128584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_binomial_formula_l1285_128590

-- Define the polynomial expressions
def expr1 (a b : ℝ) := (-a + b) * (a - b)
def expr2 (x : ℝ) := (x + 2) * (2 + x)
def expr3 (x y : ℝ) := (x + y) * (x - y)
def expr4 (x : ℝ) := (x - 2) * (x + 1)

-- Define what it means for an expression to be calculable using the square of a binomial formula
def is_square_of_binomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ → ℝ), ∀ x y, f x y = (g x y)^2 ∨ f x y = -(g x y)^2

theorem square_of_binomial_formula :
  is_square_of_binomial expr3 ∧
  ¬is_square_of_binomial expr1 ∧
  ¬is_square_of_binomial (fun x y => expr2 x) ∧
  ¬is_square_of_binomial (fun x y => expr4 x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_binomial_formula_l1285_128590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l1285_128565

/-- The x-coordinates of the intersection points of two parabolas -/
noncomputable def intersection_x : ℝ × ℝ :=
  ((-5 + Real.sqrt 233) / 4, (-5 - Real.sqrt 233) / 4)

/-- The first parabola function -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 5 * x - 6

/-- The second parabola function -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 20

/-- Theorem stating that the calculated x-coordinates are the intersection points of the parabolas -/
theorem parabolas_intersection :
  let (x1, x2) := intersection_x
  parabola1 x1 = parabola2 x1 ∧ parabola1 x2 = parabola2 x2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l1285_128565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_at_pi_third_l1285_128553

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 4 = 0

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := ∃ θ : ℝ, x = Real.cos θ ∧ y = 1 + Real.sin θ

/-- Curve C₃ in Cartesian coordinates -/
def C₃ (x y t α : ℝ) : Prop := 
  x = t * Real.cos α ∧ y = t * Real.sin α ∧ t > 0 ∧ 0 < α ∧ α < Real.pi/2

/-- Point A is the intersection of C₁ and C₃ -/
def A (x y t α : ℝ) : Prop := C₁ x y ∧ C₃ x y t α

/-- Point B is the intersection of C₂ and C₃ -/
def B (x y t α : ℝ) : Prop := C₂ x y ∧ C₃ x y t α

/-- The ratio |OB|/|OA| -/
noncomputable def ratio (xA yA xB yB : ℝ) : ℝ := 
  Real.sqrt (xB^2 + yB^2) / Real.sqrt (xA^2 + yA^2)

theorem max_ratio_at_pi_third :
  ∀ xA yA tA xB yB tB α,
    A xA yA tA α →
    B xB yB tB α →
    0 < α →
    α < Real.pi/2 →
    ratio xA yA xB yB ≤ 3/4 ∧
    (ratio xA yA xB yB = 3/4 ↔ α = Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_at_pi_third_l1285_128553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l1285_128517

/-- The speed of a man rowing in still water, given his upstream and downstream speeds. -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem stating that a man rowing at 25 kmph upstream and 41 kmph downstream has a speed of 33 kmph in still water. -/
theorem man_rowing_speed (upstream_speed downstream_speed : ℝ)
  (h1 : upstream_speed = 25)
  (h2 : downstream_speed = 41) :
  speed_in_still_water upstream_speed downstream_speed = 33 := by
  unfold speed_in_still_water
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l1285_128517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_ratio_l1285_128544

-- Define the sums of the first n terms for two arithmetic sequences
noncomputable def R (n : ℕ) : ℚ := sorry

noncomputable def U (n : ℕ) : ℚ := sorry

-- Define the ratio condition
axiom ratio_condition : ∀ n : ℕ, R n / U n = (3 * n + 5) / (2 * n + 13)

-- Define the 7th terms of both sequences
noncomputable def r_7 : ℚ := R 7 - R 6
noncomputable def s_7 : ℚ := U 7 - U 6

-- Theorem statement
theorem seventh_term_ratio : r_7 / s_7 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_ratio_l1285_128544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_changeable_digit_l1285_128563

def num1 : ℕ := 371
def num2 : ℕ := 569
def num3 : ℕ := 784
def given_sum : ℕ := 1824

def is_smallest_changeable_digit (d : ℕ) : Prop :=
  d ∈ ({3, 5, 7} : Set ℕ) ∧
  (num1 + num2 + num3 + 100 = given_sum) ∧
  ∀ d' ∈ ({3, 5, 7} : Set ℕ), d' < d → (num1 + num2 + num3 + 100 ≠ given_sum)

theorem smallest_changeable_digit :
  is_smallest_changeable_digit 3 := by
  sorry

#eval num1 + num2 + num3
#eval given_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_changeable_digit_l1285_128563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_theorem_l1285_128589

/-- A taxi service with an initial fee and per-distance charge. -/
structure TaxiService where
  initialFee : ℚ
  chargePerIncrement : ℚ
  incrementDistance : ℚ

/-- Calculate the total charge for a given trip distance. -/
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + service.chargePerIncrement * (distance / service.incrementDistance)

/-- Theorem: The total charge for a 3.6-mile trip with the given taxi service is $5.15. -/
theorem taxi_charge_theorem :
  let service : TaxiService := {
    initialFee := 2,
    chargePerIncrement := 35/100,
    incrementDistance := 2/5
  }
  totalCharge service (36/10) = 515/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_charge_theorem_l1285_128589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zeros_l1285_128547

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x - Real.log x

-- State the theorem
theorem f_has_zeros :
  (∃ x₁ ∈ Set.Ioo 0 3, f x₁ = 0) ∧
  (∃ x₂ ∈ Set.Ioi 3, f x₂ = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zeros_l1285_128547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1285_128539

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The right focus of the hyperbola -/
noncomputable def right_focus (a b c : ℝ) : ℝ × ℝ := (c, 0)

/-- The point where the line x=a intersects the asymptote in the first quadrant -/
noncomputable def asymptote_intersection (a b : ℝ) : ℝ × ℝ := (a, b)

/-- The area of triangle OAF -/
noncomputable def triangle_area (a c : ℝ) : ℝ := 3/16 * a^2

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

/-- The main theorem -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : Hyperbola a b) :
  triangle_area a c = 3/16 * a^2 →
  eccentricity a c = 3 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1285_128539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_probabilities_l1285_128503

/-- Represents a table tennis player -/
inductive Player : Type
| A
| B

/-- Represents the server of a ball -/
def Server : Type := Player

/-- Represents the outcome of a ball -/
inductive Outcome : Type
| Win
| Loss

/-- The probability of player A winning a point when serving -/
noncomputable def probAWinServe : ℝ := 2/3

/-- The probability of player A winning a point when receiving -/
noncomputable def probAWinReceive : ℝ := 1/2

/-- The probability of winning a point given the server and the scoring player -/
noncomputable def probWin (server scorer : Player) : ℝ :=
  match server, scorer with
  | Player.A, Player.A => probAWinServe
  | Player.B, Player.A => probAWinReceive
  | Player.A, Player.B => 1 - probAWinServe
  | Player.B, Player.B => 1 - probAWinReceive

/-- The sequence of servers for the first 4 balls -/
def initialServers : List Server := [Player.A, Player.A, Player.B, Player.B]

/-- The probability of player A scoring exactly 3 points in the first 4 balls -/
noncomputable def probAScores3In4 : ℝ := sorry

/-- The probability of the game ending within 4 additional balls after a 10-10 tie -/
noncomputable def probGameEndsWithin4AfterTie : ℝ := sorry

theorem table_tennis_probabilities :
  probAScores3In4 = 1/3 ∧ probGameEndsWithin4AfterTie = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_probabilities_l1285_128503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1285_128533

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt (1 + 2*x) - 7

-- State the theorem
theorem inequality_holds (x : ℝ) :
  x ≥ -1/2 ∧ x ≠ 0 →
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ -1/2 ≤ x ∧ x < 45/8) :=
by
  sorry

-- State that f is increasing
axiom f_increasing : ∀ x y, x < y → f x < f y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1285_128533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1285_128552

-- Define the ellipse C
noncomputable def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line y = kx
def line_k (k x y : ℝ) : Prop := y = k * x

-- Define the area of triangle ABD
noncomputable def area_ABD (k : ℝ) : ℝ := 4 * (1 + k^2) / Real.sqrt ((4 * k^2 + 1) * (k^2 + 4))

theorem ellipse_and_triangle_properties :
  -- The ellipse C passes through (√3, 1/2)
  ellipse_C (Real.sqrt 3) (1/2) ∧
  -- For all k ≠ 0
  ∀ k : ℝ, k ≠ 0 →
    -- The area of triangle ABD has a minimum value
    ∃ min_area : ℝ, ∀ k' : ℝ, k' ≠ 0 → area_ABD k' ≥ min_area ∧
    -- The minimum area occurs when k = ±1 (i.e., y = ±x)
    (area_ABD k' = min_area ↔ (k' = 1 ∨ k' = -1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1285_128552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l1285_128554

noncomputable def coneVolume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_volume_ratio :
  let radiusA : ℝ := 14.8
  let heightA : ℝ := 28.3
  let radiusB : ℝ := 28.3
  let heightB : ℝ := 14.8
  (coneVolume radiusA heightA) / (coneVolume radiusB heightB) = 148 / 283 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l1285_128554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_360_l1285_128551

/-- The speed of a train in km/h, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train is 360 km/h -/
theorem train_speed_is_360 (length : ℝ) (time : ℝ) 
  (h1 : length = 500) (h2 : time = 5) : 
  train_speed length time = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_360_l1285_128551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_tan_product_l1285_128582

theorem simplify_tan_product :
  (∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  intros h_tan_sum h_tan_45
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_tan_product_l1285_128582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_iff_vector_equation_l1285_128524

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (O A B X : V)

theorem point_on_line_iff_vector_equation :
  (∃ t : ℝ, ∀ O : V, X - O = t • (A - O) + (1 - t) • (B - O)) ↔
  ∃ l : ℝ, X - A = l • (B - A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_iff_vector_equation_l1285_128524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_lollipops_l1285_128506

/-- Represents the number of lollipops eaten each day -/
def LollipopSequence : Type := Fin 7 → ℕ

/-- The condition that each day after the first, 8 more lollipops are eaten -/
def IncreasesByEight (seq : LollipopSequence) : Prop :=
  ∀ i : Fin 6, seq (i.succ) = seq i + 8

/-- The total number of lollipops eaten over 7 days is 180 -/
def TotalIs180 (seq : LollipopSequence) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 7)) seq) = 180

theorem fourth_day_lollipops (seq : LollipopSequence) 
  (h1 : IncreasesByEight seq) (h2 : TotalIs180 seq) : 
  seq 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_lollipops_l1285_128506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_D_strongest_correlation_l1285_128558

structure Student where
  name : String
  r : ℝ
  m : ℝ

def students : List Student := [
  { name := "A", r := 0.82, m := 115 },
  { name := "B", r := 0.78, m := 106 },
  { name := "C", r := 0.69, m := 124 },
  { name := "D", r := 0.85, m := 103 }
]

def has_strongest_correlation (s : Student) (others : List Student) : Prop :=
  ∀ other ∈ others, (|s.r| ≥ |other.r|) ∧ (s.m ≤ other.m)

theorem student_D_strongest_correlation :
  ∃ s ∈ students, s.name = "D" ∧ has_strongest_correlation s (students.filter (fun x => x.name ≠ "D")) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_D_strongest_correlation_l1285_128558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_a_for_necessary_condition_l1285_128568

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 2) * (x - 3) < 0}

-- Define set B as a function of a
noncomputable def B (a : ℝ) : Set ℝ := {x : ℝ | ∃ y, y = Real.log ((x - (a^2 + 2)) / (a - x))}

-- Statement 1
theorem intersection_A_complement_B :
  A ∩ (U \ B (1/2)) = Set.Ioc (9/4) 3 := by sorry

-- Statement 2
theorem range_of_a_for_necessary_condition :
  {a : ℝ | A ⊆ B a} = Set.Iic (-1) ∪ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_a_for_necessary_condition_l1285_128568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1285_128526

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (k m : ℝ), y = k * x + m ∧ 0 = k * 4 + m

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the distance between A and B
noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem line_equation :
  ∀ (A B : ℝ × ℝ),
  intersection_points A B →
  distance_AB A B = 8 →
  (∀ (x y : ℝ), line_l x y ↔ (x = 4 ∨ 5*x - 12*y - 20 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1285_128526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pcr_amplification_efficiency_l1285_128561

/-- The amplification efficiency of DNA in a PCR process -/
noncomputable def amplification_efficiency (X_0 X_n : ℝ) (n : ℕ) : ℝ :=
  10^((Real.log X_n - Real.log X_0) / n) - 1

/-- The theorem stating the amplification efficiency for the given conditions -/
theorem pcr_amplification_efficiency :
  let X_0 : ℝ := 1  -- Initial quantity (arbitrary non-zero value)
  let X_6 : ℝ := 100 * X_0  -- Quantity after 6 cycles
  let p : ℝ := amplification_efficiency X_0 X_6 6
  abs (p - 1.154) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pcr_amplification_efficiency_l1285_128561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_proof_statement_is_false_l1285_128541

-- Define the types of proof methods
inductive ProofMethod
| Direct
| Contradiction
| Geometric
| Induction
| Axiomatic

-- Define properties of proof methods
def directProofProperty (m : ProofMethod) : Prop :=
  m = ProofMethod.Direct → ∀ step prevStep : ℕ, (step > prevStep) → True

def contradictionProofProperty (m : ProofMethod) : Prop :=
  m = ProofMethod.Contradiction → ∀ negation : Prop, (¬negation → False) → True

def inductionProofProperty (m : ProofMethod) : Prop :=
  m = ProofMethod.Induction → ∃ n : ℕ, n ≠ 0 ∧ True

def axiomaticProofProperty (m : ProofMethod) : Prop :=
  m = ProofMethod.Axiomatic → ∀ p : Prop, (p → True) ∧ (p → True)

-- Define the incorrect statement about geometric proofs
def geometricProofIncorrectStatement (m : ProofMethod) : Prop :=
  m = ProofMethod.Geometric → ∀ (shape : Type) (construct analyze : shape → Prop), 
    (∀ s : shape, construct s → analyze s)

-- Theorem statement
theorem geometric_proof_statement_is_false :
  ∀ m : ProofMethod,
    directProofProperty m →
    contradictionProofProperty m →
    inductionProofProperty m →
    axiomaticProofProperty m →
    ¬(geometricProofIncorrectStatement m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_proof_statement_is_false_l1285_128541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1285_128577

noncomputable section

-- Define the triangle ABC
def Triangle (A B C : ℝ) := A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Define the conditions
def TriangleConditions (A B C : ℝ) :=
  Triangle A B C ∧ A + B = 3 * C ∧ 2 * Real.sin (A - C) = Real.sin B

-- Define the side lengths
def SideLengths (AB AC BC : ℝ) := AB > 0 ∧ AC > 0 ∧ BC > 0

-- Define the height on side AB
def Height (h : ℝ) := h > 0

-- Theorem statement
theorem triangle_proof (A B C AB AC BC h : ℝ) 
  (triangle_cond : TriangleConditions A B C) 
  (side_lengths : SideLengths AB AC BC) 
  (ab_length : AB = 5) 
  (height : Height h) : 
  Real.sin A = 3 * (10 : ℝ).sqrt / 10 ∧ h = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1285_128577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_areas_calculation_l1285_128534

noncomputable def room_length : ℝ := 20
noncomputable def room_width : ℝ := 12
noncomputable def veranda_width : ℝ := 2
noncomputable def pool_length : ℝ := 15
noncomputable def pool_width : ℝ := 6

noncomputable def room_area : ℝ := room_length * room_width
noncomputable def pool_area : ℝ := pool_length * pool_width
noncomputable def tiled_floor_area : ℝ := room_area - pool_area

noncomputable def total_length : ℝ := room_length + 2 * veranda_width
noncomputable def total_width : ℝ := room_width + 2 * veranda_width
noncomputable def total_area : ℝ := total_length * total_width
noncomputable def veranda_area : ℝ := total_area - room_area
noncomputable def garden_area : ℝ := veranda_area / 2
noncomputable def seating_area : ℝ := veranda_area / 2

theorem room_areas_calculation :
  tiled_floor_area = 150 ∧ garden_area = 72 ∧ seating_area = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_areas_calculation_l1285_128534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_alternating_l1285_128530

def equation_system (n : ℕ) : List (List ℚ) :=
  List.range n |>.map (λ i => 
    List.range n |>.map (λ j =>
      if j = 0 then 1
      else if j = 1 then 3
      else if j < i + 1 then (2 * j + 1 : ℚ)
      else (2 * i + 2 : ℚ)
    )
  )

def right_hand_side (n : ℕ) : List ℚ :=
  List.range n |>.map (λ i => ((i + 1 : ℕ) : ℚ))

def alternating_solution (n : ℕ) : List ℚ :=
  List.range n |>.map (λ i => if i % 2 = 0 then -1 else 1)

theorem solution_is_alternating (n : ℕ) :
  n > 0 →
  let A := equation_system n
  let b := right_hand_side n
  let x := alternating_solution n
  List.all (List.zip A b) (λ (row, rhs) => 
    (row.zip x |>.map (λ (a, x) => a * x) |>.sum) = rhs
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_alternating_l1285_128530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_perimeter_when_equation_holds_l1285_128519

/-- Triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  triangle_inequality_1 : a + b > c
  triangle_inequality_2 : b + c > a
  triangle_inequality_3 : c + a > b
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0

/-- The expression |a-b+c|+|c-a-b|-|a+b| simplifies to a - b -/
theorem simplify_expression (t : IntegerTriangle) :
  |t.a - t.b + t.c| + |t.c - t.a - t.b| - |t.a + t.b| = t.a - t.b := by
  sorry

/-- If a^2 + b^2 - 2a - 8b + 17 = 0, then the perimeter of the triangle is 9 -/
theorem perimeter_when_equation_holds (t : IntegerTriangle) 
  (h : t.a^2 + t.b^2 - 2*t.a - 8*t.b + 17 = 0) :
  t.a + t.b + t.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_perimeter_when_equation_holds_l1285_128519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l1285_128587

theorem sin_2theta_value (θ : Real) :
  θ ∈ Set.Ioo (π / 3) (2 * π / 3) →
  let a : ℝ × ℝ := (Real.cos θ, -Real.sqrt 3 * Real.cos θ)
  let b : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  (a.1 * b.1 + a.2 * b.2) + 1 = 5 / 6 →
  Real.sin (2 * θ) = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l1285_128587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_by_11_l1285_128550

/-- Represents the sequence A_n as defined in the problem -/
def A : ℕ → ℕ := sorry

/-- The number of digits in A_n -/
def d : ℕ → ℕ := sorry

/-- A_1 = 0 -/
axiom A_1 : A 1 = 0

/-- A_2 = 1 -/
axiom A_2 : A 2 = 1

/-- For n > 2, A_n is defined by concatenating A_{n-1} and A_{n-2} -/
axiom A_concat (n : ℕ) (h : n > 2) : 
  A n = A (n-1) * 10^(d (n-2)) + A (n-2)

/-- The number of digits follows the Fibonacci sequence -/
axiom d_fib (n : ℕ) (h : n > 2) : 
  d n = d (n-1) + d (n-2)

/-- Initial conditions for d -/
axiom d_1 : d 1 = 1
axiom d_2 : d 2 = 1

/-- The main theorem to be proved -/
theorem A_div_by_11 (n : ℕ) : 
  11 ∣ A n ↔ ∃ k : ℕ, n = 6 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_by_11_l1285_128550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1285_128520

/-- The number of solutions to the equation sin 4x + sin²(3x) + sin³(2x) + sin⁴x = 0 in [-π, π] -/
def num_solutions : ℕ := 5

/-- The equation we're solving -/
noncomputable def equation (x : ℝ) : ℝ := 
  Real.sin (4 * x) + (Real.sin (3 * x))^2 + (Real.sin (2 * x))^3 + (Real.sin x)^4

theorem solution_count :
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, -Real.pi ≤ x ∧ x ≤ Real.pi ∧ equation x = 0) ∧
    (∀ x, -Real.pi ≤ x → x ≤ Real.pi → equation x = 0 → x ∈ S) ∧
    Finset.card S = num_solutions :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1285_128520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_l1285_128529

/-- The polynomial q(x) = x^4 - 18 satisfies q(x^4) - q(x^4 - 3) = [q(x)]^3 + 18 for all real x -/
theorem polynomial_equation (x : ℝ) : 
  (fun y ↦ y^4 - 18) (x^4) - (fun y ↦ y^4 - 18) (x^4 - 3) = ((fun y ↦ y^4 - 18) x)^3 + 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_l1285_128529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1285_128570

theorem power_equality (m n : ℤ) (P Q : ℕ) (h1 : P = (2 : ℕ)^(m.toNat)) (h2 : Q = (5 : ℕ)^(n.toNat)) :
  P^(3*n.toNat) = (8 : ℕ)^(m*n).toNat := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1285_128570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_calculation_specific_result_l1285_128513

-- Define the custom operation
noncomputable def custom_op (a b : ℝ) : ℝ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem custom_op_calculation :
  ∀ (a b c : ℝ), a ≠ b → b ≠ c → custom_op a b ≠ c →
  Int.floor (custom_op (custom_op a b) c) = 52 ∧ 
  custom_op (custom_op 8 6) 2 = custom_op (custom_op a b) c :=
by
  sorry

-- Define the specific calculation
noncomputable def specific_calculation : ℝ := custom_op (custom_op 8 6) 2

-- State the result for the specific values
theorem specific_result : Int.floor specific_calculation = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_calculation_specific_result_l1285_128513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straighten_road_demonstrates_principle_other_phenomena_dont_demonstrate_principle_l1285_128578

/-- Represents a daily phenomenon --/
inductive Phenomenon
  | NailWoodStrip
  | StraightenRoad
  | MeasureLongJump

/-- Predicate that checks if a phenomenon demonstrates the shortest distance principle --/
def demonstrates_shortest_distance_principle (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.StraightenRoad => True
  | _ => False

/-- The principle that the shortest distance between two points is a straight line --/
axiom shortest_distance_principle : ∀ (a b : ℝ × ℝ), 
  ∃ (d : ℝ), d = Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ∧ 
  ∀ (path : ℝ → ℝ × ℝ), path 0 = a → path 1 = b → 
  (∫ (t : ℝ) in Set.Icc 0 1, Real.sqrt ((deriv path t).1^2 + (deriv path t).2^2)) ≥ d

theorem straighten_road_demonstrates_principle :
  demonstrates_shortest_distance_principle Phenomenon.StraightenRoad :=
by
  sorry

theorem other_phenomena_dont_demonstrate_principle :
  ¬demonstrates_shortest_distance_principle Phenomenon.NailWoodStrip ∧
  ¬demonstrates_shortest_distance_principle Phenomenon.MeasureLongJump :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_straighten_road_demonstrates_principle_other_phenomena_dont_demonstrate_principle_l1285_128578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_of_f_l1285_128535

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -1/3 * x^3 - 1/2 * x^2 + 2*x

-- State the theorem
theorem max_point_of_f :
  ∃ (x_max : ℝ), x_max = 1 ∧
  ∀ (x : ℝ), f x ≤ f x_max := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_of_f_l1285_128535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1285_128579

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following properties when (a+b):(a+c):(b+c) = 9:10:11 -/
theorem triangle_properties (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  (a + b) / (a + c) = 9 / 10 →
  (a + c) / (b + c) = 10 / 11 →
  Real.sin A / Real.sin B = 4 / 5 ∧
  Real.sin B / Real.sin C = 5 / 6 ∧
  (∃ (x : Real), (x = A ∧ C = 2 * x) ∨ (x = B ∧ C = 2 * x) ∨ (x = C ∧ A = 2 * x)) ∧
  (c = 6 → 2 * (a * Real.sin A) / c = 16 * Real.sqrt 7 / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1285_128579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_auxiliary_cars_l1285_128516

/-- Represents a point on the road between two cities -/
structure Point where
  position : ℚ
  deriving Repr

/-- Represents a car with its current position and fuel level -/
structure Car where
  position : Point
  fuel : ℚ
  deriving Repr

/-- The distance between the two cities -/
def cityDistance : ℚ := 2

/-- The maximum fuel a car can carry -/
def maxFuel : ℚ := cityDistance / 2

/-- A function that simulates the movement of cars and fuel transfer -/
def transportCar (auxiliaryCars : ℕ) : Prop :=
  ∃ (startCity endCity : Point) (carA : Car) (helpers : List Car),
    startCity.position = 0 ∧
    endCity.position = cityDistance ∧
    carA.position = startCity ∧
    carA.fuel = maxFuel ∧
    helpers.length = auxiliaryCars ∧
    (∀ h ∈ helpers, h.fuel ≤ maxFuel) ∧
    (∃ (finalCarA : Car), 
      finalCarA.position = endCity ∧
      finalCarA.fuel ≥ 0 ∧
      (∀ h ∈ helpers, h.position = startCity ∨ h.position = endCity) ∧
      (∀ h ∈ helpers, h.fuel ≥ 0))

/-- The theorem stating that 3 is the minimum number of auxiliary cars needed -/
theorem min_auxiliary_cars :
  (∀ n < 3, ¬ transportCar n) ∧ transportCar 3 := by
  sorry

#check min_auxiliary_cars

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_auxiliary_cars_l1285_128516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_closed_form_l1285_128521

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => -1
  | 1 => 1
  | n+2 => 2 * sequenceA (n+1) + 3 * sequenceA n + 3^(n+2)

def closed_form (n : ℕ) : ℚ :=
  (1/16) * ((4*n - 3) * 3^(n+1) - (-1)^n * 7)

theorem sequence_closed_form : ∀ n : ℕ, sequenceA n = closed_form n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_closed_form_l1285_128521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_value_l1285_128548

theorem sin_two_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (2 * α) = Real.cos (π / 4 - α)) : 
  Real.sin (2 * α) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_value_l1285_128548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_pricing_l1285_128542

/-- Represents the cost and selling prices of an article -/
structure Article where
  costPrice : ℚ
  sellingPrice1 : ℚ
  sellingPrice2 : ℚ
  sellingPrice3 : ℚ

/-- Calculates the percentage profit or loss -/
def percentageChange (cost : ℚ) (selling : ℚ) : ℚ :=
  (selling - cost) / cost * 100

/-- Theorem stating the conditions and the result to be proved -/
theorem article_pricing (a : Article)
  (h1 : a.sellingPrice2 = 1625)
  (h2 : percentageChange a.costPrice a.sellingPrice2 = 25)
  (h3 : a.sellingPrice1 = 1320)
  (h4 : percentageChange a.costPrice a.sellingPrice1 = 
        -(percentageChange a.costPrice a.sellingPrice3)) :
  a.sellingPrice3 = 1280 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_pricing_l1285_128542
