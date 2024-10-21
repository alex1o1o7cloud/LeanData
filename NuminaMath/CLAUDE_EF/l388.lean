import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normalized_det_unbounded_l388_38809

/-- Definition of the matrix A_n -/
noncomputable def A (n : ℕ) : Matrix (Fin (n-1)) (Fin (n-1)) ℝ :=
  Matrix.of (fun i j => if i = j then (i.val + 2 : ℝ) else 1)

/-- The sequence of normalized determinants -/
noncomputable def normalized_det (n : ℕ) : ℝ := (A n).det / n.factorial

/-- Theorem stating that the sequence of normalized determinants is unbounded -/
theorem normalized_det_unbounded :
  ¬ (∃ M : ℝ, ∀ n : ℕ, n > 1 → |normalized_det n| ≤ M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normalized_det_unbounded_l388_38809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_roots_l388_38869

theorem arithmetic_roots :
  (Real.sqrt 49 = 7) ∧
  ((-27/64 : Real)^(1/3) = -3/4) ∧
  (Real.sqrt (Real.sqrt 81) = 3 ∨ Real.sqrt (Real.sqrt 81) = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_roots_l388_38869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l388_38820

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x : ℝ, Real.cos x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l388_38820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_M_properties_l388_38801

-- Define the properties of set M
class HasSetMProperties (M : Set ℚ) where
  zero_in_M : (0 : ℚ) ∈ M
  one_in_M : (1 : ℚ) ∈ M
  closed_under_subtraction : ∀ x y, x ∈ M → y ∈ M → (x - y) ∈ M
  closed_under_reciprocal : ∀ x, x ∈ M → x ≠ 0 → (1 / x) ∈ M

-- Theorem statement
theorem set_M_properties {M : Set ℚ} [HasSetMProperties M] :
  ((1 / 3) ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x, x ∈ M → x^2 ∈ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_M_properties_l388_38801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cosine_l388_38896

noncomputable section

open Real

theorem smallest_angle_cosine (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.sin A / Real.sin B = 3 / 5 →
  Real.sin B / Real.sin C = 5 / 7 →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  A ≤ B ∧ A ≤ C →
  Real.cos A = 13 / 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cosine_l388_38896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_bus_children_l388_38841

/-- Represents the number of children on a bus at various stages --/
structure BusRide where
  initial : ℕ
  got_off : ℕ
  got_on : ℕ
  final : ℕ

/-- Theorem stating the initial number of children on the bus --/
theorem initial_bus_children (ride : BusRide) : 
  ride.got_off = 68 →
  ride.final = 12 →
  ride.got_off - ride.got_on = 24 →
  ride.initial - ride.got_off + ride.got_on = ride.final →
  ride.initial = 58 := by
  intro h1 h2 h3 h4
  sorry

#check initial_bus_children

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_bus_children_l388_38841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_l388_38853

theorem sin_cos_equation (k : ℤ) : 
  let x : ℝ := (π / 2) * (2 * k + 1)
  (1/2) * Real.sin (4 * x) * Real.sin x + Real.sin (2 * x) * Real.sin x = 2 * (Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_l388_38853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_card_is_one_three_l388_38845

-- Define the card type
inductive Card
  | one_two
  | one_three
  | two_three

-- Define the players
inductive Player
  | A
  | B
  | C

-- Function to assign cards to players
def card_assignment : Player → Card := sorry

-- A's statement
def A_statement (assignment : Player → Card) : Prop :=
  (assignment Player.A = Card.one_two ∧ assignment Player.B = Card.two_three) ∨
  (assignment Player.A = Card.one_three ∧ assignment Player.B = Card.one_two) ∨
  (assignment Player.A = Card.one_three ∧ assignment Player.B = Card.two_three)

-- B's statement
def B_statement (assignment : Player → Card) : Prop :=
  (assignment Player.B = Card.one_two ∧ assignment Player.C = Card.two_three) ∨
  (assignment Player.B = Card.one_three ∧ assignment Player.C = Card.one_two) ∨
  (assignment Player.B = Card.two_three ∧ assignment Player.C = Card.one_three)

-- C's statement
def C_statement (assignment : Player → Card) : Prop :=
  assignment Player.C ≠ Card.two_three

-- Theorem stating that A's card must be (1,3)
theorem A_card_is_one_three :
  ∀ assignment : Player → Card,
    (∀ p1 p2 : Player, p1 ≠ p2 → assignment p1 ≠ assignment p2) →
    A_statement assignment →
    B_statement assignment →
    C_statement assignment →
    assignment Player.A = Card.one_three :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_card_is_one_three_l388_38845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l388_38825

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) / x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l388_38825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l388_38858

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := sin (x + π / 6)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Theorem statement
theorem axis_of_symmetry_g :
  ∃ (k : ℤ), g ((k : ℝ) * π / 2 + π / 3) = 0 ∧
  (∀ (x : ℝ), g (((k : ℝ) * π + 2 * π / 3) - x) = g (((k : ℝ) * π + 2 * π / 3) + x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l388_38858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l388_38874

def A : Set ℝ := {x | x^2 - 2*x < 3}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-1) 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l388_38874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l388_38867

/-- The value of the infinite nested radical √(3 - √(3 - √(3 - √(3 - ...)))) -/
noncomputable def nestedRadical : ℝ := (Real.sqrt 13 - 1) / 2

/-- The infinite nested radical satisfies the equation x = √(3 - x) -/
axiom nested_radical_eq : nestedRadical = Real.sqrt (3 - nestedRadical)

/-- The value of the infinite nested radical is (√13 - 1) / 2 -/
theorem nested_radical_value : 
  nestedRadical = (Real.sqrt 13 - 1) / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l388_38867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_savings_difference_l388_38898

/-- Calculates the total cost of two pairs of shoes under Promotion A -/
noncomputable def costPromotionA (shoePrice : ℝ) : ℝ :=
  shoePrice + (shoePrice / 2)

/-- Calculates the total cost of two pairs of shoes under Promotion B -/
noncomputable def costPromotionB (shoePrice : ℝ) : ℝ :=
  shoePrice + (shoePrice - 15)

/-- The price of each pair of shoes -/
def shoePriceEach : ℝ := 40

theorem promotion_savings_difference :
  costPromotionB shoePriceEach - costPromotionA shoePriceEach = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_savings_difference_l388_38898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l388_38888

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^2 - 36)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x < -6 ∨ (-6 < x ∧ x < 6) ∨ 6 < x} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l388_38888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l388_38811

/-- The general term of the sequence -/
def a (n : ℕ) (lambda : ℝ) : ℝ := n^2 - (6 + 2*lambda)*n + 2014

/-- The minimum of the sequence occurs at n = 6 or n = 7 -/
def min_at_6_or_7 (lambda : ℝ) : Prop :=
  ∀ (n : ℕ), a n lambda ≥ min (a 6 lambda) (a 7 lambda)

theorem lambda_range (lambda : ℝ) :
  min_at_6_or_7 lambda → 5/2 ≤ lambda ∧ lambda ≤ 9/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l388_38811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_range_l388_38846

def z (a : ℝ) : ℂ := a + 2*Complex.I

theorem complex_second_quadrant_range (a : ℝ) :
  z a ^ 2 = 3 - Complex.I →
  (∃ (x y : ℝ), (z a) / (3 - Complex.I) = x + y*Complex.I ∧ x < 0 ∧ y > 0) ↔
  -1/2 < a ∧ a < -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_range_l388_38846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_range_k_l388_38822

noncomputable def f (k : ℝ) (x : ℝ) := k * Real.cos (k * x)

def monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

def range_k : Set ℝ :=
  Set.Icc (-6) (-4) ∪ Set.Ioo 0 3 ∪ Set.Icc 8 9 ∪ {-12}

theorem f_monotone_decreasing_implies_range_k :
  (∀ k, monotone_decreasing (f k) (Real.pi / 4) (Real.pi / 3)) →
  {k : ℝ | monotone_decreasing (f k) (Real.pi / 4) (Real.pi / 3)} = range_k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_range_k_l388_38822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_25_12_in_terms_of_a_and_b_fraction_value_l388_38821

-- Part 1
theorem log_25_12_in_terms_of_a_and_b (a b : ℝ) (h1 : (5 : ℝ)^a = 3) (h2 : (5 : ℝ)^b = 4) :
  Real.log 12 / Real.log 25 = (a + b) / 2 := by sorry

-- Part 2
theorem fraction_value (x : ℝ) (h : x^(1/2 : ℝ) + x^(-(1/2 : ℝ)) = 5) :
  x / (x^2 + 1) = 1 / 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_25_12_in_terms_of_a_and_b_fraction_value_l388_38821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l388_38868

/-- Represents a digit in base k -/
def Digit (k : ℕ) := Fin k

/-- Represents a number in base k as a list of digits -/
def BaseKNumber (k : ℕ) := List (Digit k)

/-- Sums the digits of a number in base k -/
def sumDigits {k : ℕ} (n : BaseKNumber k) : ℕ :=
  n.foldl (λ sum d => sum + d.val) 0

/-- Performs one step of the described process -/
def processStep {k : ℕ} (n : BaseKNumber k) : BaseKNumber k :=
  sorry  -- Implementation details omitted for brevity

/-- Generates the sequence of numbers as described in the problem -/
def generateSequence {k : ℕ} (start : BaseKNumber k) : ℕ → BaseKNumber k :=
  λ n => match n with
  | 0 => start
  | n+1 => processStep (generateSequence start n)

/-- Converts a natural number to its base k representation -/
def toBaseK (k : ℕ) (n : ℕ) : BaseKNumber k :=
  sorry  -- Implementation details omitted for brevity

/-- The main theorem to be proved -/
theorem sequence_convergence {k : ℕ} (h : k > 5) (start : BaseKNumber k) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
    generateSequence start n = toBaseK k (2 * (k - 1)^3) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l388_38868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l388_38807

-- Define the constants as noncomputable
noncomputable def p : ℝ := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def q : ℝ := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def r : ℝ := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7
noncomputable def s : ℝ := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7

-- State the theorem
theorem sum_of_reciprocals_squared : 
  (1/p + 1/q + 1/r + 1/s)^2 = 112/3481 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_l388_38807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l388_38881

noncomputable section

-- Define the quadratic inequality
def quad_ineq (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + m + 2 ≤ 0

-- Define the solution set M
def M (m : ℝ) : Set ℝ := {x | quad_ineq m x}

-- Define the function f
def f (m : ℝ) : ℝ := (m^2 + 2*m + 5) / (m + 1)

theorem problem_solution :
  (∀ m : ℝ, M m = ∅ ↔ -1 < m ∧ m < 2) ∧
  (∀ m : ℝ, -1 < m ∧ m < 2 → ∀ y : ℝ, f m ≤ y → 4 ≤ y) ∧
  (∀ m : ℝ, (M m ≠ ∅ ∧ M m ⊆ Set.Icc 1 4) ↔ 2 ≤ m ∧ m ≤ 18/7) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l388_38881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l388_38899

theorem range_of_g : 
  ∀ x : ℝ, 0 ≤ (Real.sin x)^4 - Real.tan x * Real.sin x * Real.cos x + (Real.cos x)^4 ∧
           (Real.sin x)^4 - Real.tan x * Real.sin x * Real.cos x + (Real.cos x)^4 ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l388_38899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_sum_l388_38827

noncomputable def f (x : ℝ) : ℝ :=
  if x < 15 then x + 2 else 2 * x + 1

theorem inverse_f_sum : (Function.invFun f) 10 + (Function.invFun f) 37 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_sum_l388_38827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_Q_l388_38814

/-- A point rotating on a unit circle. -/
structure RotatingPoint where
  x : ℝ → ℝ
  y : ℝ → ℝ
  on_unit_circle : ∀ t, x t ^ 2 + y t ^ 2 = 1

/-- The angular velocity of a rotating point. -/
noncomputable def angular_velocity (p : RotatingPoint) : ℝ :=
  sorry

/-- Whether a point rotates counterclockwise. -/
def rotates_counterclockwise (p : RotatingPoint) : Prop :=
  sorry

/-- Whether a point rotates clockwise. -/
def rotates_clockwise (p : RotatingPoint) : Prop :=
  sorry

/-- Point P rotating counterclockwise on the unit circle with angular velocity ω. -/
noncomputable def P (ω : ℝ) : RotatingPoint where
  x := λ t => Real.cos (ω * t)
  y := λ t => Real.sin (ω * t)
  on_unit_circle := by
    intro t
    simp [Real.cos_sq_add_sin_sq]

/-- Point Q defined in terms of P's coordinates. -/
noncomputable def Q (p : RotatingPoint) : RotatingPoint where
  x := λ t => -2 * p.x t * p.y t
  y := λ t => p.y t ^ 2 - p.x t ^ 2
  on_unit_circle := sorry

theorem rotation_of_Q (ω : ℝ) :
  rotates_counterclockwise (P ω) →
  angular_velocity (P ω) = ω →
  rotates_clockwise (Q (P ω)) ∧
  angular_velocity (Q (P ω)) = 2 * ω := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_Q_l388_38814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_winning_strategy_l388_38804

theorem alice_winning_strategy (r : ℝ) (h : r ≤ 3) :
  ∃ (strategy : Fin 6 → ℝ),
    (∀ i : Fin 6, 0 ≤ strategy i ∧ strategy i ≤ 1) ∧
    ∀ (placement : Fin 6 → Fin 3),
      |strategy (placement 0) - strategy (placement 1)| +
      |strategy (placement 2) - strategy (placement 3)| +
      |strategy (placement 4) - strategy (placement 5)| ≥ r :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_winning_strategy_l388_38804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_partition_exists_l388_38882

/-- A sequence of 2022 binary digits -/
def Sequence := Fin 2022 → Fin 2

/-- The set of all valid sequences -/
def AllSequences : Set Sequence :=
  {s | (Finset.card (Finset.filter (fun i => s i = 0) Finset.univ) = 1011) ∧
       (Finset.card (Finset.filter (fun i => s i = 1) Finset.univ) = 1011)}

/-- Two sequences are compatible if they match in exactly 4 positions -/
def compatible (s1 s2 : Sequence) : Prop :=
  (Finset.card (Finset.filter (fun i => s1 i = s2 i) Finset.univ)) = 4

/-- A partition of AllSequences into 20 groups -/
def Partition := Fin 20 → Set Sequence

/-- The main theorem stating that a valid partition exists -/
theorem sequence_partition_exists :
  ∃ (p : Partition), 
    (∀ s ∈ AllSequences, ∃ i, s ∈ p i) ∧ 
    (∀ i j s1 s2, s1 ∈ p i → s2 ∈ p j → i = j → ¬compatible s1 s2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_partition_exists_l388_38882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l388_38866

/-- Represents a triangular prism -/
structure TriangularPrism where
  volume : ℝ

/-- Represents a point on an edge of the prism -/
structure EdgePoint where
  ratio : ℝ

/-- Represents a pyramid within the prism -/
structure Pyramid where
  prism : TriangularPrism
  m : EdgePoint
  n : EdgePoint
  k : EdgePoint

/-- Helper function to calculate the volume of a pyramid (not implemented) -/
noncomputable def volume_of_pyramid (p : Pyramid) : ℝ := sorry

/-- Theorem statement for the maximum volume of the pyramid -/
theorem max_pyramid_volume 
  (prism : TriangularPrism)
  (m : EdgePoint)
  (n : EdgePoint)
  (k : EdgePoint)
  (h_prism_volume : prism.volume = 40)
  (h_m_ratio : m.ratio = 3/7)
  (h_n_ratio : n.ratio = 2/5)
  (h_k_ratio : k.ratio = 4/9) :
  ∃ (max_volume : ℝ), 
    (∀ (p : Pyramid), p.prism = prism ∧ p.m = m ∧ p.n = n ∧ p.k = k → 
      volume_of_pyramid p ≤ max_volume) ∧
    max_volume = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l388_38866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l388_38860

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

theorem interest_rate_proof (principal : ℝ) (principal_pos : 0 < principal) :
  ∃ rate : ℝ, simple_interest principal rate 10 = principal / 5 ∧ rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l388_38860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetry_axis_position_parabola_symmetry_axis_range_l388_38837

-- Define the parabola and its properties
def parabola (a b c t m n : ℝ) : Prop :=
  a > 0 ∧
  (∀ x y, y = a * x^2 + b * x + c ↔ (x, y) = (3, m) ∨ (x, y) = (t + 1, n)) ∧
  (∀ x, x = t ↔ ∀ y, y = a * x^2 + b * x + c → y = a * (2*t - x)^2 + b * (2*t - x) + c)

-- Theorem for part 1
theorem parabola_symmetry_axis_position
  (a b c t m n : ℝ) (h : parabola a b c t m n) (h_eq : m = n) :
  t = 4 := by sorry

-- Theorem for part 2
theorem parabola_symmetry_axis_range
  (a b c t m n : ℝ) (h : parabola a b c t m n) (h_lt : n < m ∧ m < c) :
  t > 4 ∨ (3/2 < t ∧ t < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetry_axis_position_parabola_symmetry_axis_range_l388_38837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l388_38847

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- Theorem statement
theorem range_of_a (a : ℝ) : f a < 1 ↔ -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l388_38847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l388_38890

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.sqrt (a * x^2 + a * x + 1)

-- State the theorem
theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l388_38890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l388_38886

def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_properties (a b c : ℝ) :
  (∀ x, (deriv (f · a b c)) x = 3*x^2 + 2*a*x + b) →
  (deriv (f · a b c) 1 = 3) →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), deriv (f · a b c) x = 0) →
  (3 - f 1 a b c + 1 = 0) →
  (a = 2 ∧ b = -4 ∧ c = 5) ∧
  (∀ x ∈ Set.Icc (-3) 1, f x a b c ≤ 13) ∧
  (∃ x ∈ Set.Icc (-3) 1, f x a b c = 13) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l388_38886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_segment_existence_l388_38857

-- Define the squares Q and R
def Q : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 500 ∧ 0 ≤ p.2 ∧ p.2 ≤ 500}
def R : Set (ℝ × ℝ) := {p | 125 ≤ p.1 ∧ p.1 ≤ 375 ∧ 125 ≤ p.2 ∧ p.2 ≤ 375}

-- Define the perimeter of Q
def perimeterQ : Set (ℝ × ℝ) := {p | p ∈ Q ∧ (p.1 = 0 ∨ p.1 = 500 ∨ p.2 = 0 ∨ p.2 = 500)}

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a line segment intersects R
def intersectsR (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    let x := p1.1 + t * (p2.1 - p1.1);
    let y := p1.2 + t * (p2.2 - p1.2);
    (x, y) ∈ R

-- State the theorem
theorem square_segment_existence :
  ∃ A B : ℝ × ℝ, A ∈ perimeterQ ∧ B ∈ perimeterQ ∧
    ¬(intersectsR A B) ∧ distance A B > 521 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_segment_existence_l388_38857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sum_constant_l388_38876

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define point M
def point_M : ℝ × ℝ := (3, -3)

-- Define point P where circle C intersects positive x-axis
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop := y + 3 = k * (x - 3)

-- Define the slope of a line passing through two points
noncomputable def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

theorem slope_sum_constant (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
  line_through_M k x₁ y₁ ∧ line_through_M k x₂ y₂ ∧
  x₁ ≠ x₂ →
  line_slope x₁ y₁ point_P.1 point_P.2 + line_slope x₂ y₂ point_P.1 point_P.2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_sum_constant_l388_38876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_puree_water_percentage_l388_38879

/-- Represents the production of tomato puree from tomato juice -/
structure TomatoProduction where
  juice_volume : ℝ
  juice_water_percentage : ℝ
  puree_volume : ℝ

/-- Calculates the water percentage in tomato puree -/
noncomputable def water_percentage_in_puree (prod : TomatoProduction) : ℝ :=
  let solid_volume := prod.juice_volume * (1 - prod.juice_water_percentage)
  let water_volume := prod.puree_volume - solid_volume
  (water_volume / prod.puree_volume) * 100

/-- Theorem stating that given the specific conditions, the water percentage in tomato puree is 20% -/
theorem tomato_puree_water_percentage :
  let prod := TomatoProduction.mk 40 0.9 5
  water_percentage_in_puree prod = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_puree_water_percentage_l388_38879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l388_38850

/-- The equation of a line passing through two points -/
noncomputable def line_equation (x₁ y₁ x₂ y₂ : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- The intersection point of two lines -/
noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  ((b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁), (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁))

/-- The midpoint of a line segment -/
noncomputable def midpoint_of_segment (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ :=
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

theorem line_equation_proof :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x + y - 2 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x - y - 4 = 0
  let P : ℝ × ℝ := intersection_point 1 1 (-2) 1 (-1) (-4)
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (5, 1)
  let Q : ℝ × ℝ := midpoint_of_segment A.1 A.2 B.1 B.2
  let l : ℝ → ℝ → Prop := line_equation P.1 P.2 Q.1 Q.2
  ∀ x y, l x y ↔ 3 * x + y - 8 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l388_38850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_purchase_strategy_l388_38842

/-- Represents the unit price of strawberries in yuan per kg -/
structure UnitPrice where
  first : ℚ
  second : ℚ

/-- Represents the purchase information for a person -/
structure Purchase where
  first_weight : ℚ
  second_weight : ℚ
  first_cost : ℚ
  second_cost : ℚ

/-- Calculates the average price for a purchase -/
noncomputable def average_price (p : Purchase) : ℚ :=
  (p.first_cost + p.second_cost) / (p.first_weight + p.second_weight)

/-- Theorem representing the strawberry purchase strategy problem -/
theorem strawberry_purchase_strategy 
  (price : UnitPrice) 
  (xiao_ming : Purchase) 
  (xiao_liang : Purchase) :
  price.first * xiao_ming.first_weight = price.second * xiao_ming.second_weight ∧
  xiao_ming.second_weight = xiao_ming.first_weight + 1/2 ∧
  xiao_liang.first_weight = xiao_liang.second_weight ∧
  xiao_liang.first_weight = 2 ∧
  price.first * xiao_liang.first_weight - price.second * xiao_liang.second_weight = 10 →
  average_price xiao_ming < average_price xiao_liang ∧
  price.first = 15 ∧
  price.second = 10 := by
  sorry

#check strawberry_purchase_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_purchase_strategy_l388_38842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l388_38832

-- Define the ☆ operation
noncomputable def star (a b : ℝ) : ℝ := (a + b) / a

-- State the theorem
theorem equation_solution :
  ∃ x : ℝ, star (star 4 3) x = 13 ∧ x = 21 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l388_38832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_gain_is_80_l388_38885

/-- Represents the gain per year in a lending transaction -/
noncomputable def gain_per_year (borrowed_amount : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) : ℝ :=
  borrowed_amount * (lend_rate - borrow_rate) / 100

/-- Proves that the gain per year is 80 for the given conditions -/
theorem lending_gain_is_80 :
  gain_per_year 4000 4 6 = 80 := by
  -- Unfold the definition of gain_per_year
  unfold gain_per_year
  -- Simplify the arithmetic
  simp [mul_sub_left_distrib, div_eq_mul_inv]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lending_gain_is_80_l388_38885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l388_38848

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x + Real.sqrt 3 * Real.sin (2 * x)

-- Define the angle A
noncomputable def A : ℝ := Real.pi / 3

-- Define the sides of the triangle
noncomputable def a : ℝ := Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 21 / 3
noncomputable def b : ℝ := 2 * c

-- Theorem statement
theorem triangle_area_proof :
  f A = 2 →
  (∃ (k : ℤ), -Real.pi/3 + k*Real.pi ≤ A ∧ A ≤ Real.pi/6 + k*Real.pi) →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (1/2) * b * c * Real.sin A = 7 * Real.sqrt 3 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l388_38848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l388_38862

/-- Represents an investment split into two parts with different interest rates -/
structure Investment where
  total : ℚ
  rate1 : ℚ
  rate2 : ℚ

/-- Calculates the average interest rate for an investment -/
def averageRate (inv : Investment) (amount1 : ℚ) : ℚ :=
  let amount2 := inv.total - amount1
  let interest1 := amount1 * inv.rate1
  let interest2 := amount2 * inv.rate2
  (interest1 + interest2) / inv.total

theorem investment_average_rate :
  ∀ (inv : Investment) (amount1 : ℚ),
    inv.total = 6000 ∧
    inv.rate1 = 3 / 100 ∧
    inv.rate2 = 5 / 100 ∧
    amount1 * inv.rate1 = (inv.total - amount1) * inv.rate2 →
    averageRate inv amount1 = 375 / 10000 := by
  sorry

#eval (375 : ℚ) / 10000  -- To verify the result is indeed 0.0375

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l388_38862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l388_38852

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x and y
noncomputable def x : ℂ := (-1 + i * Real.sqrt 5) / 2
noncomputable def y : ℂ := (-1 - i * Real.sqrt 5) / 2

-- State the theorem
theorem sum_of_powers : x^12 + y^12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l388_38852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_M_l388_38813

theorem existence_of_M (k l : ℕ+) : ∃ M : ℕ+, ∀ n : ℕ+, n > M → ¬ (∃ m : ℤ, ((k : ℝ) + 1/2)^(n:ℝ) + ((l : ℝ) + 1/2)^(n:ℝ) = m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_M_l388_38813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l388_38889

noncomputable def y (x : ℝ) : ℝ := 
  (Real.cos (Real.sqrt 2)) ^ (1/3) - (1/52) * (Real.cos (26*x))^2 / Real.sin (52*x)

theorem y_derivative (x : ℝ) (h : Real.sin (52*x) ≠ 0) : 
  deriv y x = 1 / (2 * (Real.sin (26*x))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l388_38889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_example_l388_38840

/-- The center of a hyperbola is the midpoint of its foci -/
noncomputable def hyperbola_center (f1 f2 : ℝ × ℝ) : ℝ × ℝ :=
  ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)

/-- Theorem: The center of a hyperbola with foci at (3, -2) and (-1, 6) is (1, 2) -/
theorem hyperbola_center_example : 
  hyperbola_center (3, -2) (-1, 6) = (1, 2) := by
  -- Unfold the definition of hyperbola_center
  unfold hyperbola_center
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_example_l388_38840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_safe_distance_is_optimal_l388_38844

/-- The smallest safe distance for a flea to jump over lava intervals -/
def smallest_safe_distance (A B : ℕ+) (n : ℕ) : ℕ :=
  (n - 1) * A + B

/-- Conditions for lava interval placement and flea jumps -/
structure LavaJumpProblem where
  A : ℕ+
  B : ℕ+
  n : ℕ
  h_A_lt_B : A < B
  h_B_lt_2A : B < 2 * A
  h_n_bounds : A / (n + 1 : ℝ) ≤ B - A ∧ B - A < A / n

/-- Represents a configuration of lava intervals -/
structure LavaConfiguration where
  -- Add appropriate fields here
  dummy : Unit

/-- Predicate to check if a flea can jump safely given a configuration -/
def flea_can_jump_safely (A B F : ℕ) (lava_config : LavaConfiguration) : Prop :=
  sorry -- Define the actual conditions here

/-- Main theorem: The smallest safe distance is (n-1)A + B -/
theorem smallest_safe_distance_is_optimal (p : LavaJumpProblem) :
  ∀ F : ℕ,
    (∀ lava_config : LavaConfiguration,
      flea_can_jump_safely p.A p.B F lava_config ↔
      F ≥ smallest_safe_distance p.A p.B p.n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_safe_distance_is_optimal_l388_38844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_general_term_inverse_minus_two_is_perfect_square_l388_38854

def a : ℕ → ℚ
  | 0 => 1/3
  | 1 => 1/3
  | n+2 => let a_n_minus_2 := a n
           let a_n_minus_1 := a (n+1)
           ((1 - 2*a_n_minus_2) * a_n_minus_1^2) / (2*a_n_minus_1^2 - 4*a_n_minus_2*a_n_minus_1^2 + a_n_minus_2)

noncomputable def general_term (n : ℕ) : ℝ :=
  ((13/3 - 5/2*Real.sqrt 3) * (7 + 4*Real.sqrt 3)^n + 
   (13/3 + 5/2*Real.sqrt 3) * (7 - 4*Real.sqrt 3)^n + 
   7/3)⁻¹

theorem a_equals_general_term (n : ℕ) : (a n : ℝ) = general_term n := by sorry

theorem inverse_minus_two_is_perfect_square (n : ℕ) : 
  ∃ m : ℝ, (a n)⁻¹ - 2 = m^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_general_term_inverse_minus_two_is_perfect_square_l388_38854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l388_38883

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3^(x - m)

-- Define the inverse function f⁻¹
noncomputable def f_inv (m : ℝ) (x : ℝ) : ℝ := 2 + (Real.log x) / (Real.log 3)

-- Define the function F
noncomputable def F (m : ℝ) (x : ℝ) : ℝ := (f_inv m x)^2 - f_inv m (x^2)

-- State the theorem
theorem range_of_F :
  ∀ m : ℝ, f m 2 = 1 →
  Set.range (F m) = Set.Icc 2 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l388_38883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_36_l388_38892

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : (Finset.filter (λ x ↦ 36 % x = 0) (Finset.range 37)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_36_l388_38892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_comparison_l388_38805

theorem sphere_volume_comparison (r₁ : ℝ) (r₂ : ℝ) (c d : ℕ+) : 
  r₁ = 7 → 
  (4 / 3) * Real.pi * r₂^3 = 3 * ((4 / 3) * Real.pi * r₁^3) →
  2 * r₂ = c * (d : ℝ)^(1/3) →
  ¬ ∃ (n : ℕ), n^3 = d →
  c + d = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_comparison_l388_38805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l388_38861

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_derivative_positive : ∀ x > 0, f x + x * (deriv f) x > 0

-- Define the variables
def a : ℝ := (4 : ℝ) ^ (1/5 : ℝ)
noncomputable def b (f : ℝ → ℝ) : ℝ := (Real.log 3 / Real.log 4) * f (Real.log 3 / Real.log 4)
noncomputable def c (f : ℝ → ℝ) : ℝ := (Real.log (1/16) / Real.log 4) * f (Real.log (1/16) / Real.log 4)

-- State the theorem
theorem abc_inequality (f : ℝ → ℝ) (h_f_odd : ∀ x, f (-x) = -f x) 
  (h_f_deriv : ∀ x > 0, f x + x * (deriv f) x > 0) : 
  c f > a ∧ a > b f := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l388_38861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_passes_through_point_l388_38865

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the conditions of the problem
def Problem (a b : ℝ) (F M : ℝ × ℝ) (k₁ k₂ : ℝ) : Prop :=
  a > b ∧ b > 0 ∧
  F = (2, 0) ∧
  M.2 > 0 ∧
  M ∈ Ellipse a b ∧
  (M.1 - F.1)^2 + M.2^2 = M.1^2 + M.2^2 ∧
  k₁ + k₂ = 8

-- Theorem statement
theorem ellipse_line_passes_through_point
  (a b : ℝ) (F M A B : ℝ × ℝ) (k₁ k₂ : ℝ)
  (h : Problem a b F M k₁ k₂)
  (hA : A ∈ Ellipse a b)
  (hB : B ∈ Ellipse a b)
  (hMA : (A.2 - M.2) / (A.1 - M.1) = k₁)
  (hMB : (B.2 - M.2) / (B.1 - M.1) = k₂) :
  ∃ (t : ℝ), (1 - t) • A + t • B = (-1/2, -2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_passes_through_point_l388_38865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l388_38816

theorem max_true_statements (a b : ℝ) : 
  (¬ ((1 / a > 1 / b) ∧ (a^2 < b^2) ∧ (a > b) ∧ (a > 0) ∧ (b > 0))) ∧
  (∃ (p q r s : Prop), 
    (p = (1 / a > 1 / b) ∨ p = (a^2 < b^2) ∨ p = (a > b) ∨ p = (a > 0) ∨ p = (b > 0)) ∧
    (q = (1 / a > 1 / b) ∨ q = (a^2 < b^2) ∨ q = (a > b) ∨ q = (a > 0) ∨ q = (b > 0)) ∧
    (r = (1 / a > 1 / b) ∨ r = (a^2 < b^2) ∨ r = (a > b) ∨ r = (a > 0) ∨ r = (b > 0)) ∧
    (s = (1 / a > 1 / b) ∨ s = (a^2 < b^2) ∨ s = (a > b) ∨ s = (a > 0) ∨ s = (b > 0)) ∧
    (p ∧ q ∧ r ∧ s) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (q ≠ r) ∧ (q ≠ s) ∧ (r ≠ s)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l388_38816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_plus_pi_over_4_l388_38824

noncomputable def angle_α (m : ℝ) : ℝ := Real.arctan (-2)

theorem tan_α_plus_pi_over_4 (m : ℝ) (h : m ≠ 0) :
  Real.tan (angle_α m + π/4) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_plus_pi_over_4_l388_38824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_bed_circumference_l388_38897

-- Define constants
def plants : ℕ := 19
def area_per_plant : ℝ := 4

-- Define the total area of the bed
noncomputable def total_area : ℝ := plants * area_per_plant

-- Define pi (we'll use Lean's built-in pi)
noncomputable def π : ℝ := Real.pi

-- Define the radius of the bed
noncomputable def radius : ℝ := (total_area / π).sqrt

-- Define the circumference of the bed
noncomputable def circumference : ℝ := 2 * π * radius

-- Theorem to prove
theorem circular_bed_circumference : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (circumference - 30.91) < ε := by
  -- We'll use 0.01 as our epsilon
  use 0.01
  apply And.intro
  · -- Prove ε > 0
    norm_num
  · -- Prove abs (circumference - 30.91) < ε
    sorry -- Detailed proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_bed_circumference_l388_38897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_linear_combinations_l388_38849

noncomputable def intPart (x : ℝ) : ℤ := ⌊x⌋

noncomputable def a (n : ℕ) : ℝ := n ^ (1/3 : ℝ)
noncomputable def b (n : ℕ) : ℝ := 1 / (a n - intPart (a n))
noncomputable def c (n : ℕ) : ℝ := 1 / (b n - intPart (b n))

theorem infinite_linear_combinations :
  ∃ S : Set ℕ, (Set.Infinite S) ∧
  (∀ n ∈ S, ¬∃ m : ℕ, n = m^3) ∧
  (∀ n ∈ S, ∃ r s t : ℤ, (r ≠ 0 ∨ s ≠ 0 ∨ t ≠ 0) ∧
    r • (a n) + s • (b n) + t • (c n) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_linear_combinations_l388_38849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_angle_difference_cos_α_value_cos_β_value_l388_38817

-- Problem 1
theorem cos_2α_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.tan α = 2) :
  Real.cos (2 * α) = -3/5 := by sorry

-- Problem 2
theorem angle_difference (α β : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : β ∈ Set.Ioo 0 π)
  (h3 : Real.tan α = 2) (h4 : Real.cos β = -7 * Real.sqrt 2 / 10) :
  2 * α - β = -π/4 := by sorry

-- Problem 3
theorem cos_α_value (α : Real) (h1 : α ∈ Set.Ioo (π/2) π)
  (h2 : Real.sin (α/2) + Real.cos (α/2) = Real.sqrt 6 / 2) :
  Real.cos α = -Real.sqrt 3 / 2 := by sorry

-- Problem 4
theorem cos_β_value (α β : Real) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : Real.sin (α/2) + Real.cos (α/2) = Real.sqrt 6 / 2) (h4 : Real.sin (α - β) = -3/5) :
  Real.cos β = -(4 * Real.sqrt 3 + 3) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_angle_difference_cos_α_value_cos_β_value_l388_38817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l388_38878

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and scalar
variable (a b : V) (lambda : ℝ)

-- Define the theorem
theorem parallel_vectors_lambda (h1 : ¬(∃ (k : ℝ), a = k • b)) 
  (h2 : ∃ (μ : ℝ), lambda • a + b = μ • (a + 2 • b)) : lambda = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l388_38878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l388_38834

def is_valid_pair (a b : ℂ) : Prop :=
  a^4 * b^6 = 1 ∧ a^8 * b^3 = 1

theorem count_valid_pairs :
  ∃! (s : Finset (ℂ × ℂ)), (∀ (p : ℂ × ℂ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧ s.card = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l388_38834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l388_38835

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x

theorem f_max_min :
  ∀ x : ℝ, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 →
  f x ≤ 2 ∧ -1 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l388_38835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_profit_calculation_l388_38875

/-- The profit distribution and total profit of a beadshop over six days --/
structure BeadshopProfit where
  monday_percent : ℚ
  tuesday_percent : ℚ
  wednesday_percent : ℚ
  thursday_percent : ℚ
  friday_percent : ℚ
  saturday_loss_percent : ℚ
  total_profit : ℚ

/-- Calculate the Friday profit given the beadshop's profit distribution --/
def friday_profit (bp : BeadshopProfit) : ℚ :=
  bp.friday_percent * bp.total_profit / (1 - bp.saturday_loss_percent)

/-- Theorem stating that the Friday profit is approximately 552.63 --/
theorem friday_profit_calculation (bp : BeadshopProfit) 
  (h1 : bp.monday_percent = 20/100)
  (h2 : bp.tuesday_percent = 18/100)
  (h3 : bp.wednesday_percent = 10/100)
  (h4 : bp.thursday_percent = 25/100)
  (h5 : bp.friday_percent = 15/100)
  (h6 : bp.saturday_loss_percent = 5/100)
  (h7 : bp.total_profit = 3500) :
  ∃ ε : ℚ, ε > 0 ∧ |friday_profit bp - 552.63| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_profit_calculation_l388_38875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_central_angle_l388_38887

/-- A cone with an equilateral triangle as its left view -/
structure EquilateralCone where
  /-- The radius of the base circle -/
  r : ℝ
  /-- The slant height of the cone -/
  slant_height : ℝ
  /-- Condition that the left view is an equilateral triangle -/
  equilateral : slant_height = 2 * r

/-- The central angle (in degrees) of the sector in the lateral surface development diagram -/
noncomputable def central_angle (cone : EquilateralCone) : ℝ :=
  360 * cone.r / cone.slant_height

theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  central_angle cone = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_central_angle_l388_38887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l388_38829

-- Define the slope range
def slope_range : Set ℝ := Set.Icc (-Real.sqrt 3) (Real.sqrt 3 / 3)

-- Define the inclination angle range
def angle_range : Set ℝ := Set.Icc 0 Real.pi

-- Define the relationship between slope and inclination angle
noncomputable def slope_to_angle (k : ℝ) : ℝ := Real.arctan k

-- Theorem statement
theorem inclination_angle_range (k : ℝ) (h_slope : k ∈ slope_range) :
  ∃ α ∈ angle_range, slope_to_angle k = α ∧ 
  (α ∈ Set.Icc 0 (Real.pi / 6) ∨ α ∈ Set.Icc ((2 * Real.pi) / 3) Real.pi) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l388_38829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deviation_representation_l388_38871

/-- Represents the mass deviation of a ping-pong ball from the standard mass -/
def MassDeviation := ℝ

/-- Represents a ping-pong ball -/
structure PingPongBall where
  massDeviation : MassDeviation

/-- Function to represent the mass deviation -/
def representDeviation (ball : PingPongBall) : ℝ :=
  ball.massDeviation

/-- Theorem: If a positive deviation is represented by its positive value,
    then a negative deviation should be represented by its negative value -/
theorem deviation_representation 
  (standard : ℝ) 
  (positive_ball negative_ball : PingPongBall) 
  (h1 : positive_ball.massDeviation = (0.02 : ℝ))
  (h2 : negative_ball.massDeviation = (-0.02 : ℝ))
  (h3 : representDeviation positive_ball = (0.02 : ℝ)) :
  representDeviation negative_ball = (-0.02 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deviation_representation_l388_38871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l388_38851

-- Define the function f(x) = x^2 + ln x - 4
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x - 4

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l388_38851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l388_38803

theorem arithmetic_calculations :
  ((-5) * 3 - 8 / (-2) = -11) ∧
  ((-1)^3 + (5 - (-3)^2) / 6 = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l388_38803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_from_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l388_38894

-- Define the basic geometric objects
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define a plane in 3D space
def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Submodule ℝ V

-- Define a line in 3D space
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Submodule ℝ V

-- Define perpendicularity between a line and a plane
def perpendicular_line_plane (l : Line V) (p : Plane V) : Prop := sorry

-- Define perpendicularity between two planes
def perpendicular_planes (p q : Plane V) : Prop := sorry

-- Define a line passing through a plane
def line_in_plane (l : Line V) (p : Plane V) : Prop := sorry

-- Define the line of intersection between two planes
noncomputable def intersection_line (p q : Plane V) : Line V := sorry

-- Statement 2
theorem perpendicular_planes_from_perpendicular_line 
  (p q : Plane V) (l : Line V) :
  line_in_plane l p ∧ perpendicular_line_plane l q → perpendicular_planes p q :=
by sorry

-- Statement 4
theorem non_perpendicular_line_in_perpendicular_planes 
  (p q : Plane V) (l : Line V) :
  perpendicular_planes p q ∧ 
  line_in_plane l p ∧ 
  ¬perpendicular_line_plane l (intersection_line p q) → 
  ¬perpendicular_line_plane l q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_from_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l388_38894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_QAMB_Q_coordinates_l388_38838

-- Define the circle M
def circle_M : ℝ × ℝ → Prop := λ p => p.1^2 + (p.2 - 2)^2 = 1

-- Define a point on the x-axis
def point_on_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0

-- Define tangent property
def is_tangent (Q A : ℝ × ℝ) (M : (ℝ × ℝ) → Prop) : Prop :=
  M A ∧ ∀ P, M P → dist Q P ≥ dist Q A

-- Function to calculate area of a quadrilateral
noncomputable def area_quadrilateral (Q A M B : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - M.1) * (A.2 - B.2) - (A.1 - B.1) * (Q.2 - M.2)) / 2

-- Theorem for minimum area
theorem min_area_QAMB (Q A B M : ℝ × ℝ) :
  circle_M M →
  point_on_x_axis Q →
  is_tangent Q A circle_M →
  is_tangent Q B circle_M →
  ∃ (area : ℝ), area ≥ Real.sqrt 3 ∧
  ∀ Q' A' B', circle_M M →
               point_on_x_axis Q' →
               is_tangent Q' A' circle_M →
               is_tangent Q' B' circle_M →
               area_quadrilateral Q' A' M B' ≥ area :=
sorry

-- Theorem for Q coordinates when |AB| = (4√2)/3
theorem Q_coordinates (Q A B M : ℝ × ℝ) (h : dist A B = (4 * Real.sqrt 2) / 3) :
  circle_M M →
  point_on_x_axis Q →
  is_tangent Q A circle_M →
  is_tangent Q B circle_M →
  Q.1 = Real.sqrt 5 ∨ Q.1 = -Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_QAMB_Q_coordinates_l388_38838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_from_octagon_centers_l388_38855

/-- The side length of each regular octagon -/
def octagon_side_length : ℝ := 2

/-- The number of octagons surrounding the central octagon -/
def surrounding_octagons : ℕ := 6

/-- The radius of the circumscribed circle of a regular octagon with side length s -/
noncomputable def octagon_circumradius (s : ℝ) : ℝ := s * (1 / Real.sin (45 * Real.pi / 180))

/-- The side length of the equilateral triangle formed by connecting 
    the centers of three adjacent surrounding octagons -/
noncomputable def triangle_side_length : ℝ := 2 * octagon_circumradius octagon_side_length

/-- The area of an equilateral triangle with side length a -/
noncomputable def equilateral_triangle_area (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2

theorem area_of_triangle_from_octagon_centers : 
  equilateral_triangle_area triangle_side_length = 24 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_from_octagon_centers_l388_38855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_second_less_than_first_l388_38836

/-- Represents a ball with a number -/
structure Ball where
  number : Fin 5
  deriving Fintype, DecidableEq

/-- The set of all balls -/
def allBalls : Finset Ball := Finset.univ

/-- The probability of drawing a specific ball -/
def probDraw (b : Ball) : ℚ := 1 / 5

/-- The probability of drawing two balls in sequence -/
def probDrawTwo (b1 b2 : Ball) : ℚ := probDraw b1 * probDraw b2

/-- The set of all possible outcomes when drawing two balls -/
def allOutcomes : Finset (Ball × Ball) := Finset.product allBalls allBalls

/-- The set of favorable outcomes (second ball's number less than first) -/
def favorableOutcomes : Finset (Ball × Ball) :=
  Finset.filter (fun (b1, b2) => b2.number < b1.number) allOutcomes

/-- The probability of drawing a second ball with a number less than the first -/
noncomputable def probSecondLessThanFirst : ℚ :=
  (Finset.sum favorableOutcomes (fun (b1, b2) => probDrawTwo b1 b2)) /
  (Finset.sum allOutcomes (fun (b1, b2) => probDrawTwo b1 b2))

theorem prob_second_less_than_first :
  probSecondLessThanFirst = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_second_less_than_first_l388_38836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_increasing_l388_38873

open Set
open Function

theorem function_g_increasing (a : ℝ) :
  let f : ℝ → ℝ := λ x => (1/3) * x^3 - a * x^2 + a * x + 2
  let f'' : ℝ → ℝ := λ x => (deriv^[2] f) x
  let g : ℝ → ℝ := λ x => (f'' x) / x
  (∃ x₀ ∈ Iio 1, ∀ x ∈ Iio 1, f'' x₀ ≤ f'' x) →
  StrictMonoOn g (Ioi 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_increasing_l388_38873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_result_l388_38895

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.sqrt (9 - x^2)

theorem integral_equals_result : 
  ∫ x in (0)..(3/2), f x = (3 * Real.pi / 4) - (9 * Real.sqrt 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_result_l388_38895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_l388_38800

def taxi_route : List Int := [5, -4, -8, 10, 3, -6]
def fuel_consumption : Float := 0.2
def gasoline_price : Float := 6.2

theorem taxi_journey :
  (taxi_route.sum = 0) ∧
  ((taxi_route.map Int.natAbs).sum.toFloat * fuel_consumption = 7.2) ∧
  ((taxi_route.map Int.natAbs).sum.toFloat * fuel_consumption * gasoline_price = 44.64) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_l388_38800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brianchon_theorem_l388_38880

-- Define the basic structures
structure Point : Type := (x y : ℝ)

structure Line : Type := (a b c : ℝ)

def Triangle := Point × Point × Point

-- Define the conditions
def satisfies_conditions (t : Triangle) (a₁ a₂ b₁ b₂ c₁ c₂ : Point) : Prop := 
  let (A, B, C) := t
  (∃ k : ℝ, a₁ = ⟨k * B.x + (1 - k) * C.x, k * B.y + (1 - k) * C.y⟩) ∧ 
  (∃ k : ℝ, a₂ = ⟨k * B.x + (1 - k) * C.x, k * B.y + (1 - k) * C.y⟩) ∧
  (∃ k : ℝ, b₁ = ⟨k * C.x + (1 - k) * A.x, k * C.y + (1 - k) * A.y⟩) ∧ 
  (∃ k : ℝ, b₂ = ⟨k * C.x + (1 - k) * A.x, k * C.y + (1 - k) * A.y⟩) ∧
  (∃ k : ℝ, c₁ = ⟨k * A.x + (1 - k) * B.x, k * A.y + (1 - k) * B.y⟩) ∧ 
  (∃ k : ℝ, c₂ = ⟨k * A.x + (1 - k) * B.x, k * A.y + (1 - k) * B.y⟩) ∧
  (B.x - a₁.x)^2 + (B.y - a₁.y)^2 = (a₂.x - C.x)^2 + (a₂.y - C.y)^2 ∧
  (C.x - b₁.x)^2 + (C.y - b₁.y)^2 = (b₂.x - A.x)^2 + (b₂.y - A.y)^2 ∧
  (A.x - c₁.x)^2 + (A.y - c₁.y)^2 = (c₂.x - B.x)^2 + (c₂.y - B.y)^2

-- Define intersection of lines
noncomputable def intersect (l₁ l₂ : Line) : Point := sorry

-- Define the points X, Y, Z
noncomputable def X (t : Triangle) (b₁ c₂ : Point) : Point := 
  let (A, B, C) := t
  intersect (Line.mk (B.y - b₁.y) (b₁.x - B.x) (B.x * b₁.y - B.y * b₁.x))
            (Line.mk (C.y - c₂.y) (c₂.x - C.x) (C.x * c₂.y - C.y * c₂.x))

noncomputable def Y (t : Triangle) (c₁ a₂ : Point) : Point := 
  let (A, B, C) := t
  intersect (Line.mk (C.y - c₁.y) (c₁.x - C.x) (C.x * c₁.y - C.y * c₁.x))
            (Line.mk (A.y - a₂.y) (a₂.x - A.x) (A.x * a₂.y - A.y * a₂.x))

noncomputable def Z (t : Triangle) (a₁ b₂ : Point) : Point := 
  let (A, B, C) := t
  intersect (Line.mk (A.y - a₁.y) (a₁.x - A.x) (A.x * a₁.y - A.y * a₁.x))
            (Line.mk (B.y - b₂.y) (b₂.x - B.x) (B.x * b₂.y - B.y * b₂.x))

-- Define concurrency
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- The main theorem
theorem brianchon_theorem (t : Triangle) (a₁ a₂ b₁ b₂ c₁ c₂ : Point) 
  (h : satisfies_conditions t a₁ a₂ b₁ b₂ c₁ c₂) : 
  let (A, B, C) := t
  let x := X t b₁ c₂
  let y := Y t c₁ a₂
  let z := Z t a₁ b₂
  are_concurrent 
    (Line.mk (y.y - A.y) (A.x - y.x) (A.y * y.x - A.x * y.y))
    (Line.mk (z.y - B.y) (B.x - z.x) (B.y * z.x - B.x * z.y))
    (Line.mk (x.y - C.y) (C.x - x.x) (C.y * x.x - C.x * x.y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brianchon_theorem_l388_38880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l388_38831

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x + |5 * x - 10|

-- Define the piecewise function
noncomputable def g (x : ℝ) : ℝ :=
  if x < 2 then -2 * x + 10
  else 8 * x - 10

-- Theorem stating the equivalence of f and g
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by
  intro x
  simp [f, g]
  split
  · -- Case: x < 2
    rw [abs_of_neg]
    · ring
    · linarith
  · -- Case: x ≥ 2
    rw [abs_of_nonneg]
    · ring
    · linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l388_38831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_iff_product_infinitely_many_not_sum_infinitely_many_sum_l388_38818

-- Define the set S
def S : Set ℝ := {x | ∃ q : ℚ, q > 0 ∧ x = q + 1/q}

-- Statement 1
theorem sum_iff_product (N : ℕ) :
  (∃ x y : ℝ, x ∈ S ∧ y ∈ S ∧ x + y = N) ↔ (∃ x y : ℝ, x ∈ S ∧ y ∈ S ∧ x * y = N) := by sorry

-- Statement 2
theorem infinitely_many_not_sum :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, ¬∃ x y : ℝ, x ∈ S ∧ y ∈ S ∧ x + y = f n := by sorry

-- Statement 3
theorem infinitely_many_sum :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, ∃ x y : ℝ, x ∈ S ∧ y ∈ S ∧ x + y = f n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_iff_product_infinitely_many_not_sum_infinitely_many_sum_l388_38818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sumEveryFourthOfSpecificSequence_l388_38810

def arithmeticSequence (n : ℕ) (a₁ d : ℚ) : ℕ → ℚ :=
  fun k => a₁ + (k - 1 : ℚ) * d

def sumOfArithmeticSequence (n : ℕ) (a₁ d : ℚ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

def sumEveryFourth (n : ℕ) (a₁ d : ℚ) : ℚ :=
  sumOfArithmeticSequence ((n + 3) / 4) a₁ (4 * d)

theorem sumEveryFourthOfSpecificSequence :
  ∃ (a₁ : ℚ),
    sumOfArithmeticSequence 2023 a₁ 2 = 6070 ∧
    sumEveryFourth 2023 a₁ 2 = 1521 := by
  sorry

#eval sumEveryFourth 2023 (6070 / 2023 - 2022) 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sumEveryFourthOfSpecificSequence_l388_38810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_derivative_reciprocal_l388_38893

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem limit_derivative_reciprocal (a : ℝ) (h : a ≠ 0) :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ →
    |(f x - f a) / (x - a) + 1 / a^2| < ε := by
  sorry

#check limit_derivative_reciprocal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_derivative_reciprocal_l388_38893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_sum_l388_38839

noncomputable def f (x : ℝ) : ℝ := max (5*x - 7) (max (2*x + 1) (-3*x + 11))

def is_tangent_at_three_points (p : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, p x = a*x^2 + b*x + c) ∧  -- p is a quadratic polynomial
    (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧  -- x₁, x₂, x₃ are distinct
    (p x₁ = f x₁ ∧ p x₂ = f x₂ ∧ p x₃ = f x₃) ∧  -- p touches f at these points
    (∀ x, p x ≤ f x)  -- p is below or equal to f everywhere

theorem tangent_points_sum (p : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : is_tangent_at_three_points p x₁ x₂ x₃) : 
  x₁ + x₂ + x₃ = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_sum_l388_38839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100th_term_l388_38802

def a : ℕ → ℕ
| 0 => 2  -- a₁ = 2
| n + 1 => a n + 2 * (n + 1)  -- aₙ₊₁ = aₙ + 2n

theorem a_100th_term : a 99 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100th_term_l388_38802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wythoff_winning_positions_l388_38812

/-- Wythoff game state -/
structure WythoffState :=
  (pile1 : ℕ) (pile2 : ℕ)

/-- Wythoff move -/
inductive WythoffMove
  | RemoveFromOne (pile : ℕ) (amount : ℕ)
  | RemoveFromBoth (amount : ℕ)

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Wythoff sequences -/
noncomputable def a (n : ℕ) : ℕ := Int.toNat ⌊(n : ℝ) * φ⌋
noncomputable def b (n : ℕ) : ℕ := Int.toNat ⌊(n : ℝ) * (φ + 1)⌋

/-- Valid Wythoff move -/
def validMove (state : WythoffState) (move : WythoffMove) : Prop :=
  match move with
  | WythoffMove.RemoveFromOne pile amount =>
      (pile = 1 ∧ amount > 0 ∧ amount ≤ state.pile1) ∨
      (pile = 2 ∧ amount > 0 ∧ amount ≤ state.pile2)
  | WythoffMove.RemoveFromBoth amount =>
      amount > 0 ∧ amount ≤ min state.pile1 state.pile2

/-- Winning position in Wythoff game -/
def isWinningPosition (state : WythoffState) : Prop :=
  ∃ n : ℕ, state.pile1 = a n ∧ state.pile2 = b n

/-- Theorem: Characterization of winning positions in Wythoff game -/
theorem wythoff_winning_positions :
  ∀ (state : WythoffState),
    isWinningPosition state ↔
    (∀ (move : WythoffMove),
      validMove state move →
      ¬isWinningPosition (match move with
        | WythoffMove.RemoveFromOne 1 amount => ⟨state.pile1 - amount, state.pile2⟩
        | WythoffMove.RemoveFromOne 2 amount => ⟨state.pile1, state.pile2 - amount⟩
        | WythoffMove.RemoveFromBoth amount => ⟨state.pile1 - amount, state.pile2 - amount⟩
        | _ => state  -- This case should never occur due to validMove
      )) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wythoff_winning_positions_l388_38812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l388_38819

def vector_problem (a b : ℝ → ℝ → ℝ → ℝ) : Prop :=
  let dot_product := (a 0 0 0) * (b 0 0 0) + (a 1 0 0) * (b 1 0 0) + (a 2 0 0) * (b 2 0 0)
  let norm_b := Real.sqrt ((b 0 0 0)^2 + (b 1 0 0)^2 + (b 2 0 0)^2)
  let projection := dot_product / norm_b
  dot_product = -8 ∧ projection = -3 * Real.sqrt 2

theorem vector_magnitude (a b : ℝ → ℝ → ℝ → ℝ) (h : vector_problem a b) : 
  Real.sqrt ((b 0 0 0)^2 + (b 1 0 0)^2 + (b 2 0 0)^2) = (4 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l388_38819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_configuration_l388_38843

/-- Represents a letter on the calendar --/
inductive CalendarLetter
| A
| B
| D
| E
| R

/-- Returns the date behind a given letter on the calendar --/
def date_behind (letter : CalendarLetter) (x y : ℕ) : ℕ :=
  match letter with
  | CalendarLetter.A => x + 2
  | CalendarLetter.B => x + 5
  | CalendarLetter.D => y
  | CalendarLetter.E => x
  | CalendarLetter.R => 2 * x + 12

theorem calendar_configuration (x y : ℕ) :
  date_behind CalendarLetter.D x y + date_behind CalendarLetter.E x y =
  date_behind CalendarLetter.A x y + 2 * date_behind CalendarLetter.B x y ∧
  date_behind CalendarLetter.R x y = 2 * date_behind CalendarLetter.E x y + 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_configuration_l388_38843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l388_38830

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The ratio of volumes of two cylinders formed from a 6x9 rectangle -/
theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_length : ℝ := 9
  let cylinder1_height : ℝ := rectangle_width
  let cylinder1_radius : ℝ := rectangle_length / (2 * Real.pi)
  let cylinder2_height : ℝ := rectangle_length
  let cylinder2_radius : ℝ := rectangle_width / (2 * Real.pi)
  let volume1 := cylinderVolume cylinder1_radius cylinder1_height
  let volume2 := cylinderVolume cylinder2_radius cylinder2_height
  max volume1 volume2 / min volume1 volume2 = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l388_38830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_fixed_point_l388_38856

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

-- Define the tangent line condition
def tangent_condition (k a : ℝ) : Prop :=
  (k = 4/3) ∨ (∃ b, a + 2*b = 2*a ∧ (a = 2 + Real.sqrt 5 ∨ a = 2 - Real.sqrt 5))

-- Define the point P and tangent condition
def point_P_condition (a b : ℝ) : Prop :=
  ¬(circle_M a b) ∧ (∃ x y, circle_M x y ∧ (x - a)^2 + (y - b)^2 = a^2 + b^2)

-- Define the circle with PM as diameter
def circle_PM (a b x y : ℝ) : Prop :=
  x^2 + y^2 - (2 + a)*x - (3 - 2*a)*y + 2 = 0

-- Theorem statement
theorem circle_tangent_and_fixed_point :
  ∀ (a b : ℝ), point_P_condition a b →
  (∃ (k : ℝ), tangent_condition k a) ∧
  (circle_PM a b (4/5) (2/5)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_fixed_point_l388_38856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l388_38884

/-- A circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ

/-- A line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given circle C -/
noncomputable def circle_C : PolarCircle :=
  { equation := fun θ => 4 * Real.cos θ }

/-- The given line l -/
noncomputable def line_l (m : ℝ) : ParametricLine :=
  { x := fun t => Real.sqrt 2 / 2 * t + m,
    y := fun t => Real.sqrt 2 / 2 * t }

/-- The theorem stating the value of m when the line is tangent to the circle -/
theorem tangent_line_m_value :
  ∀ m : ℝ, (∃ t : ℝ, (line_l m).x t = circle_C.equation t ∧
                     (line_l m).y t = circle_C.equation t) →
  m = 2 + 2 * Real.sqrt 2 ∨ m = 2 - 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_m_value_l388_38884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_b_more_stable_l388_38823

/-- Represents a class with its variance -/
structure ClassData where
  name : String
  variance : ℝ

/-- Definition of stability based on variance -/
def more_stable (c1 c2 : ClassData) : Prop :=
  c1.variance < c2.variance

theorem class_b_more_stable :
  let class_a : ClassData := { name := "A", variance := 6.3 }
  let class_b : ClassData := { name := "B", variance := 5.5 }
  more_stable class_b class_a :=
by
  -- Unfold the definition of more_stable
  unfold more_stable
  -- Simplify the inequality
  simp
  -- Prove the inequality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_b_more_stable_l388_38823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l388_38891

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) :
  let f := quadratic_function a b c
  (∀ p q : ℝ, p ≠ q → f p = f q → f (p + q) = c) ∧
  (∀ p q : ℝ, p ≠ q → f (p + q) = c → (p + q = 0 ∨ f p = f q)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l388_38891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l388_38877

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : t.B * 2 = t.A + t.C) :
  -- Property 1: B = π/3
  t.B = Real.pi / 3 ∧
  -- Property 2: If cos(2A) + cos(2B) + cos(2C) > 1, then the largest angle is greater than π/2
  (Real.cos (2 * t.A) + Real.cos (2 * t.B) + Real.cos (2 * t.C) > 1 →
    max t.A (max t.B t.C) > Real.pi / 2) ∧
  -- Property 3: If b² = bc·cos A + ac·cos B + ab·cos C, then 3A = C
  (t.b^2 = t.b * t.c * Real.cos t.A + t.a * t.c * Real.cos t.B + t.a * t.b * Real.cos t.C →
    3 * t.A = t.C) :=
by
  sorry -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l388_38877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l388_38859

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def triangle_condition (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (Real.sin t.B + Real.sin t.C - Real.sin t.A) = t.b * Real.sin t.C

/-- The area of the triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

theorem triangle_theorem (t : Triangle) 
  (h : triangle_condition t) : 
  t.A = 2 * Real.pi / 3 ∧ 
  ∃ (S : ℝ), t.a = Real.sqrt 3 → 
    ∀ (B C : ℝ), 
      S = triangle_area { t with B := B, C := C } → 
      S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l388_38859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_at_three_oclock_l388_38870

/-- Represents the position on a clock face in degrees -/
def ClockPosition := ℝ

/-- Represents a circular object with a radius -/
structure Circle where
  radius : ℝ
  
/-- Calculates the circumference of a circle -/
noncomputable def circumference (c : Circle) : ℝ := 2 * Real.pi * c.radius

/-- Represents the clock face -/
def clockFace : Circle := { radius := 20 }

/-- Represents the rolling disk -/
def rollingDisk : Circle := { radius := 5 }

/-- Calculates the angle traveled by the disk for a given arc length on the clock face -/
noncomputable def diskAngle (arcLength : ℝ) : ℝ :=
  (arcLength / clockFace.radius) * (clockFace.radius / rollingDisk.radius)

/-- Theorem: The disk completes one full rotation at 3 o'clock position -/
theorem disk_rotation_at_three_oclock :
  diskAngle (clockFace.radius * Real.pi / 2) = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_at_three_oclock_l388_38870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kiwi_count_l388_38872

def total_fruits : ℕ := 500
def apple_fraction : ℚ := 1/4
def orange_percentage : ℚ := 20/100
def strawberry_fraction : ℚ := 1/5

theorem kiwi_count : 
  total_fruits - (total_fruits * apple_fraction).floor.toNat - 
  (total_fruits * orange_percentage).floor.toNat - 
  (total_fruits * strawberry_fraction).floor.toNat = 175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kiwi_count_l388_38872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_mapping_l388_38826

def M : Set ℤ := {-2, 0, 2}
def P : Set ℤ := {-4, 0, 4}

def f (x : ℤ) : ℤ := x ^ 2

theorem f_is_mapping :
  (∀ x, x ∈ M → f x ∈ P) ∧
  (∀ x, x ∈ M → ∃! y, y ∈ P ∧ f x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_mapping_l388_38826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_form_l388_38828

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eqn : (y : ℝ) → (x : ℝ) → Prop := λ y x => (y^2 / a^2) - (x^2 / b^2) = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

theorem hyperbola_standard_form (h : Hyperbola) 
  (point_condition : h.eqn (Real.sqrt 3) (Real.sqrt 2))
  (eccentricity_condition : eccentricity h = Real.sqrt 2) :
  ∀ (y x : ℝ), h.eqn y x ↔ y^2 - x^2 = 1 := by
  sorry

#check hyperbola_standard_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_form_l388_38828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_properties_l388_38808

noncomputable def geometric_series (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

noncomputable def geometric_series_limit (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem geometric_series_properties :
  let a : ℝ := 3
  let r : ℝ := 1/4
  (∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_series a r n - L| < ε) ∧
  (geometric_series_limit a r = 4) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a * r^n| < ε) ∧
  (geometric_series_limit a r ≠ 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_properties_l388_38808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_place_l388_38815

/-- The distance to the place in km -/
noncomputable def distance : ℝ := 3

/-- The man's rowing speed in still water in km/h -/
noncomputable def still_water_speed : ℝ := 7.5

/-- The river's speed in km/h -/
noncomputable def river_speed : ℝ := 1.5

/-- The total time for the round trip in hours -/
noncomputable def total_time : ℝ := 50 / 60

theorem distance_to_place :
  distance = 3 ∧
  (distance / (still_water_speed - river_speed) +
   distance / (still_water_speed + river_speed)) = total_time :=
by
  constructor
  · -- Prove distance = 3
    rfl
  · -- Prove the equation
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_place_l388_38815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_log_half_23_l388_38833

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f has period 2
axiom f_periodic : ∀ x, f (x + 2) = f x

-- f(x) = 2^x when x ∈ (0,1)
axiom f_power_of_two : ∀ x, 0 < x → x < 1 → f x = (2 : ℝ)^x

-- Theorem to prove
theorem f_at_log_half_23 : f (Real.log 23 / Real.log (1/2)) = -23/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_log_half_23_l388_38833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_terminates_in_20_moves_l388_38806

/-- Represents a player's choice in the game -/
inductive GameChoice
  | Add
  | Subtract

/-- Represents the game state -/
structure GameState where
  a : ℕ  -- Current number held by player A
  step : ℕ  -- Current step of the game

/-- Applies the game rules to produce the next state -/
def applyGameRules (state : GameState) (b : ℕ) (choice : GameChoice) : GameState :=
  match choice with
  | GameChoice.Add => { a := state.a + b, step := state.step + 1 }
  | GameChoice.Subtract => { a := max state.a b - min state.a b, step := state.step + 1 }

/-- Checks if the game has terminated -/
def isTerminated (state : GameState) : Prop :=
  ∃ k : ℕ, state.a = 10^k

/-- The main theorem stating that the game can be terminated in at most 20 moves -/
theorem game_terminates_in_20_moves :
  ∀ (initial : ℕ),
    (initial ≥ 10^999 ∧ initial < 10^1000) →
    ∃ (moves : List ℕ),
      moves.length ≤ 20 ∧
      ∀ (choices : List GameChoice),
        choices.length = moves.length →
        isTerminated (List.foldl
          (fun (state : GameState) (pair : ℕ × GameChoice) =>
            applyGameRules state pair.1 pair.2)
          { a := initial, step := 0 }
          (List.zip moves choices)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_terminates_in_20_moves_l388_38806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l388_38864

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - Real.log x

-- Theorem statement
theorem f_properties :
  -- Part 1: Minimum value when a = 3/8
  (∃ (x : ℝ), x > 0 ∧ f (3/8) x = -1/2 - Real.log 2 ∧ ∀ (y : ℝ), y > 0 → f (3/8) y ≥ f (3/8) x) ∧
  -- Part 2: Exactly one zero when -1 ≤ a ≤ 0
  (∀ (a : ℝ), -1 ≤ a ∧ a ≤ 0 → ∃! (x : ℝ), x > 0 ∧ f a x = 0) ∧
  -- Part 3: Two zeros iff 0 < a < 1
  (∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ (0 < a ∧ a < 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l388_38864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l388_38863

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_from_focus_to_line :
  distance_to_line (right_focus.1) (right_focus.2) 1 2 (-8) = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l388_38863
