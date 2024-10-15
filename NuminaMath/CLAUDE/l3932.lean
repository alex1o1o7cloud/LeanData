import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_points_l3932_393273

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin point (0, 0, 0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- Point P with coordinates (1, 3, 5) -/
def P : Point3D := ⟨1, 3, 5⟩

/-- Point P' with coordinates (-1, -3, -5) -/
def P' : Point3D := ⟨-1, -3, -5⟩

/-- Check if two points are symmetric with respect to the origin -/
def isSymmetricToOrigin (a b : Point3D) : Prop :=
  a.x + b.x = 0 ∧ a.y + b.y = 0 ∧ a.z + b.z = 0

/-- Theorem stating that P and P' are symmetric with respect to the origin -/
theorem symmetric_points : isSymmetricToOrigin P P' := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l3932_393273


namespace NUMINAMATH_CALUDE_percentage_problem_l3932_393299

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3932_393299


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3932_393218

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ),
    Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
    q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
    q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
    q₄ ≠ q₅ ∧
    m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → m ≥ n) ∧
  n = 2310 := by
sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l3932_393218


namespace NUMINAMATH_CALUDE_function_minimum_condition_l3932_393200

/-- Given a function f(x) = e^x + ae^(-x) where a is a constant, 
    if f(x) ≥ f(0) for all x in [-1, 1], then a = 1 -/
theorem function_minimum_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, (Real.exp x + a * Real.exp (-x)) ≥ (1 + a)) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l3932_393200


namespace NUMINAMATH_CALUDE_max_revenue_at_18_75_l3932_393290

/-- The revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The theorem stating that 18.75 maximizes the revenue function --/
theorem max_revenue_at_18_75 :
  ∀ p : ℝ, p ≤ 30 → R p ≤ R 18.75 := by
  sorry

#check max_revenue_at_18_75

end NUMINAMATH_CALUDE_max_revenue_at_18_75_l3932_393290


namespace NUMINAMATH_CALUDE_sum_of_squares_l3932_393257

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 10)
  (eq2 : y^2 + 5*z = -10)
  (eq3 : z^2 + 7*x = -20) :
  x^2 + y^2 + z^2 = 20.75 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3932_393257


namespace NUMINAMATH_CALUDE_equation_solution_l3932_393236

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x + 1) - 2 * (x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3932_393236


namespace NUMINAMATH_CALUDE_boris_books_l3932_393280

theorem boris_books (boris_initial : ℕ) (cameron_initial : ℕ) : 
  cameron_initial = 30 →
  (3 * boris_initial / 4 : ℚ) + (2 * cameron_initial / 3 : ℚ) = 38 →
  boris_initial = 24 :=
by sorry

end NUMINAMATH_CALUDE_boris_books_l3932_393280


namespace NUMINAMATH_CALUDE_tetrahedron_edges_form_two_triangles_l3932_393204

-- Define a tetrahedron as a structure with 6 edges
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

-- Define a predicate for valid triangles
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem tetrahedron_edges_form_two_triangles (t : Tetrahedron) :
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin 6),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧
    i₅ ≠ i₆ ∧
    is_valid_triangle (t.edges i₁) (t.edges i₂) (t.edges i₃) ∧
    is_valid_triangle (t.edges i₄) (t.edges i₅) (t.edges i₆) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_edges_form_two_triangles_l3932_393204


namespace NUMINAMATH_CALUDE_accidental_multiplication_l3932_393207

theorem accidental_multiplication (x : ℕ) : x * 9 = 153 → x * 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_accidental_multiplication_l3932_393207


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l3932_393201

theorem triangle_sine_sum_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l3932_393201


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l3932_393211

theorem fraction_of_powers_equals_500 : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l3932_393211


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3932_393212

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement_equals_set : S ∩ (U \ T) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3932_393212


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3932_393277

theorem sum_of_two_numbers (x y : ℤ) : x = 18 ∧ y = 2 * x - 3 → x + y = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3932_393277


namespace NUMINAMATH_CALUDE_pinterest_group_initial_pins_l3932_393293

/-- Calculates the initial number of pins in a Pinterest group --/
def initial_pins (
  daily_contribution : ℕ)  -- Average daily contribution per person
  (weekly_deletion : ℕ)    -- Weekly deletion rate per person
  (group_size : ℕ)         -- Number of people in the group
  (days : ℕ)               -- Number of days
  (final_pins : ℕ)         -- Total pins after the given period
  : ℕ :=
  final_pins - (daily_contribution * group_size * days) + (weekly_deletion * group_size * (days / 7))

theorem pinterest_group_initial_pins :
  initial_pins 10 5 20 30 6600 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_initial_pins_l3932_393293


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3932_393209

-- Define the set P
def P : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define the set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = 1/2 * x^2 - 1}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3932_393209


namespace NUMINAMATH_CALUDE_passenger_disembark_ways_l3932_393254

theorem passenger_disembark_ways (n : ℕ) (s : ℕ) (h1 : n = 10) (h2 : s = 5) :
  s^n = 5^10 := by
  sorry

end NUMINAMATH_CALUDE_passenger_disembark_ways_l3932_393254


namespace NUMINAMATH_CALUDE_cube_sum_ge_squared_product_sum_l3932_393258

theorem cube_sum_ge_squared_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ a^2*b + b^2*c + c^2*a := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_squared_product_sum_l3932_393258


namespace NUMINAMATH_CALUDE_one_hundred_fiftieth_term_l3932_393270

/-- An arithmetic sequence with first term 2 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 2 + (n - 1) * 5

theorem one_hundred_fiftieth_term :
  arithmeticSequence 150 = 747 := by
  sorry

end NUMINAMATH_CALUDE_one_hundred_fiftieth_term_l3932_393270


namespace NUMINAMATH_CALUDE_gcd_91_49_l3932_393217

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_49_l3932_393217


namespace NUMINAMATH_CALUDE_inequality_proof_l3932_393205

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3932_393205


namespace NUMINAMATH_CALUDE_cyclic_matrix_determinant_zero_l3932_393250

theorem cyclic_matrix_determinant_zero (p q r : ℝ) (a b c d : ℝ) : 
  (a^4 + p*a^2 + q*a + r = 0) → 
  (b^4 + p*b^2 + q*b + r = 0) → 
  (c^4 + p*c^2 + q*c + r = 0) → 
  (d^4 + p*d^2 + q*d + r = 0) → 
  Matrix.det 
    ![![a, b, c, d],
      ![b, c, d, a],
      ![c, d, a, b],
      ![d, a, b, c]] = 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_matrix_determinant_zero_l3932_393250


namespace NUMINAMATH_CALUDE_train_schedule_l3932_393246

theorem train_schedule (x y z : ℕ) : 
  x < 24 → y < 24 → z < 24 →
  (60 * y + z) - (60 * x + y) = 60 * z + x →
  x = 0 ∨ x = 12 := by
sorry

end NUMINAMATH_CALUDE_train_schedule_l3932_393246


namespace NUMINAMATH_CALUDE_triple_characterization_l3932_393292

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ (a^n + 203) % (a^m + 1) = 0

def solution_set (k m : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 4*k+1), (2, 3, 6*k+2), (2, 4, 8*k+8), (2, 6, 12*k+9),
   (3, 2, 4*k+3), (4, 2, 4*k+4), (5, 2, 4*k+1), (8, 2, 4*k+3),
   (10, 2, 4*k+2), (203, m, (2*k+1)*m+1)}

theorem triple_characterization :
  ∀ a m n : ℕ, is_valid_triple a m n ↔ ∃ k : ℕ, (a, m, n) ∈ solution_set k m :=
sorry

end NUMINAMATH_CALUDE_triple_characterization_l3932_393292


namespace NUMINAMATH_CALUDE_discontinuity_coincidence_l3932_393267

-- Define the functions f, g, and h
variable (f g h : ℝ → ℝ)

-- Define the conditions
variable (hf_diff : Differentiable ℝ f)
variable (hg_mono : Monotone g)
variable (hh_mono : Monotone h)
variable (hf_deriv : ∀ x, deriv f x = f x + g x + h x)

-- State the theorem
theorem discontinuity_coincidence :
  ∀ x : ℝ, ¬(ContinuousAt g x) ↔ ¬(ContinuousAt h x) := by
  sorry

end NUMINAMATH_CALUDE_discontinuity_coincidence_l3932_393267


namespace NUMINAMATH_CALUDE_min_h_21_l3932_393225

-- Define a tenuous function
def Tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → h x + h y > y^2

-- Define the sum of h from 1 to 30
def SumH (h : ℕ → ℤ) : ℤ :=
  (List.range 30).map (λ i => h (i + 1)) |>.sum

-- Theorem statement
theorem min_h_21 (h : ℕ → ℤ) (hTenuous : Tenuous h) (hMinSum : ∀ g : ℕ → ℤ, Tenuous g → SumH g ≥ SumH h) :
  h 21 ≥ 312 := by
  sorry

end NUMINAMATH_CALUDE_min_h_21_l3932_393225


namespace NUMINAMATH_CALUDE_trig_inequality_l3932_393271

theorem trig_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (h_eq : Real.sin x = x * Real.cos y) : 
  x/2 < y ∧ y < x :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l3932_393271


namespace NUMINAMATH_CALUDE_evaluate_expression_l3932_393252

theorem evaluate_expression : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3932_393252


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3932_393285

theorem sum_of_four_numbers : 8765 + 7658 + 6587 + 5876 = 28868 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3932_393285


namespace NUMINAMATH_CALUDE_eggs_left_for_breakfast_l3932_393276

def total_eggs : ℕ := 3 * 12

def eggs_for_crepes : ℕ := total_eggs / 3

def eggs_after_crepes : ℕ := total_eggs - eggs_for_crepes

def eggs_for_cupcakes : ℕ := (eggs_after_crepes * 3) / 5

def eggs_left : ℕ := eggs_after_crepes - eggs_for_cupcakes

theorem eggs_left_for_breakfast : eggs_left = 10 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_for_breakfast_l3932_393276


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l3932_393281

/-- A sequence of natural numbers -/
def NatSequence := ℕ → ℕ

/-- The sum of the first k terms of a sequence -/
def PartialSum (a : NatSequence) (k : ℕ) : ℕ :=
  (Finset.range k).sum (fun i => a i)

/-- Predicate for a sequence containing each natural number exactly once -/
def ContainsEachNatOnce (a : NatSequence) : Prop :=
  ∀ n : ℕ, ∃! k : ℕ, a k = n

/-- Predicate for the divisibility condition -/
def DivisibilityCondition (a : NatSequence) : Prop :=
  ∀ k : ℕ, k ∣ PartialSum a k

theorem existence_of_special_sequence :
  ∃ a : NatSequence, ContainsEachNatOnce a ∧ DivisibilityCondition a := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l3932_393281


namespace NUMINAMATH_CALUDE_square_side_ratio_l3932_393223

theorem square_side_ratio (area_ratio : ℚ) : 
  area_ratio = 50 / 98 → 
  ∃ (a b c : ℕ), (a : ℚ) * Real.sqrt (b : ℚ) / (c : ℚ) = Real.sqrt (area_ratio) ∧ 
                  a = 5 ∧ b = 2 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l3932_393223


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3932_393272

theorem cube_volume_problem (s : ℝ) : 
  (s + 2)^2 * (s - 3) = s^3 + 19 → s^3 = (4 + Real.sqrt 47)^3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3932_393272


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_l3932_393222

theorem cube_sum_minus_product (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a * b + a * c + b * c = 40) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1575 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_l3932_393222


namespace NUMINAMATH_CALUDE_more_than_three_solutions_l3932_393264

/-- Represents a trapezoid with bases b₁ and b₂, and height h -/
structure Trapezoid where
  b₁ : ℕ
  b₂ : ℕ
  h : ℕ

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℕ :=
  (t.b₁ + t.b₂) * t.h / 2

/-- Predicate for valid trapezoid solutions -/
def isValidSolution (m n : ℕ) : Prop :=
  m + n = 6 ∧
  10 ∣ (10 * m) ∧
  10 ∣ (10 * n) ∧
  area { b₁ := 10 * m, b₂ := 10 * n, h := 60 } = 1800

theorem more_than_three_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card > 3 ∧ ∀ (p : ℕ × ℕ), p ∈ S → isValidSolution p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_more_than_three_solutions_l3932_393264


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3932_393230

/-- The number of ways to put n indistinguishable balls into k distinguishable boxes -/
def ball_distribution (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to put 6 indistinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : ball_distribution 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3932_393230


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3932_393219

theorem complex_equation_solution (i : ℂ) (x : ℂ) (h1 : i * i = -1) (h2 : i * x = 1 + i) : x = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3932_393219


namespace NUMINAMATH_CALUDE_roots_of_equation_l3932_393260

theorem roots_of_equation (x : ℝ) : (x - 5)^2 = 2*(x - 5) ↔ x = 5 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3932_393260


namespace NUMINAMATH_CALUDE_remainder_squared_pred_l3932_393244

theorem remainder_squared_pred (n : ℤ) (h : n % 5 = 3) : (n - 1)^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_pred_l3932_393244


namespace NUMINAMATH_CALUDE_hash_six_two_l3932_393274

-- Define the # operation
def hash (x y : ℝ) : ℝ := 4*x - 4*y

-- Theorem statement
theorem hash_six_two : hash 6 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_hash_six_two_l3932_393274


namespace NUMINAMATH_CALUDE_constant_point_of_quadratic_l3932_393221

/-- The quadratic function f(x) that depends on a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

/-- The theorem stating that (2, 13) is the unique constant point for f(x) -/
theorem constant_point_of_quadratic :
  ∃! p : ℝ × ℝ, ∀ m : ℝ, f m p.1 = p.2 ∧ p = (2, 13) :=
sorry

end NUMINAMATH_CALUDE_constant_point_of_quadratic_l3932_393221


namespace NUMINAMATH_CALUDE_dot_product_when_x_negative_one_parallel_vectors_when_x_eight_l3932_393255

def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

theorem dot_product_when_x_negative_one :
  (a.1 * (b (-1)).1 + a.2 * (b (-1)).2) = 31 := by sorry

theorem parallel_vectors_when_x_eight :
  (a.1 / (b 8).1 = a.2 / (b 8).2) := by sorry

end NUMINAMATH_CALUDE_dot_product_when_x_negative_one_parallel_vectors_when_x_eight_l3932_393255


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3932_393228

theorem fraction_equation_solution (a b : ℝ) (h1 : a ≠ b) (h2 : b = 1) 
  (h3 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4) : 
  a / b = (17 + Real.sqrt 269) / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3932_393228


namespace NUMINAMATH_CALUDE_no_common_real_solution_l3932_393253

theorem no_common_real_solution :
  ¬∃ (x y : ℝ), x^2 + y^2 + 8 = 0 ∧ x^2 - 5*y + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_real_solution_l3932_393253


namespace NUMINAMATH_CALUDE_mat_coverage_fraction_l3932_393287

/-- The fraction of a square tabletop covered by a circular mat -/
theorem mat_coverage_fraction (mat_diameter : ℝ) (table_side : ℝ) 
  (h1 : mat_diameter = 18) (h2 : table_side = 24) : 
  (π * (mat_diameter / 2)^2) / (table_side^2) = π / 7 := by
  sorry

end NUMINAMATH_CALUDE_mat_coverage_fraction_l3932_393287


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3932_393226

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  r * s = a^2 →      -- Geometric mean theorem for r
  r * s = b^2 →      -- Geometric mean theorem for s
  c = r + s →        -- c is divided into r and s
  a / b = 2 / 5 →    -- Given ratio of a to b
  r / s = 4 / 25 :=  -- Conclusion: ratio of r to s
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3932_393226


namespace NUMINAMATH_CALUDE_boat_meeting_times_l3932_393279

/-- Represents the meeting time of two boats given their speeds and the river current. -/
def meeting_time (speed_A speed_C current distance : ℝ) : Set ℝ :=
  let effective_speed_A := speed_A + current
  let effective_speed_C_against := speed_C - current
  let effective_speed_C_with := speed_C + current
  let time_opposite := distance / (effective_speed_A + effective_speed_C_against)
  let time_same_direction := distance / (effective_speed_A - effective_speed_C_with)
  {time_opposite, time_same_direction}

/-- The theorem stating the meeting times of the boats under given conditions. -/
theorem boat_meeting_times :
  meeting_time 7 3 2 20 = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_boat_meeting_times_l3932_393279


namespace NUMINAMATH_CALUDE_odd_sum_probability_in_4x4_grid_l3932_393231

theorem odd_sum_probability_in_4x4_grid : 
  let n : ℕ := 16
  let grid_size : ℕ := 4
  let total_arrangements : ℕ := n.factorial
  let valid_arrangements : ℕ := (Nat.choose grid_size 2)^2 * (n/2).factorial * (n/2).factorial
  (valid_arrangements : ℚ) / total_arrangements = 1 / 360 := by
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_in_4x4_grid_l3932_393231


namespace NUMINAMATH_CALUDE_symmetric_points_range_l3932_393243

open Real

theorem symmetric_points_range (g h : ℝ → ℝ) (a : ℝ) :
  (∀ x, 1/ℯ ≤ x → x ≤ ℯ → g x = a - x^2) →
  (∀ x, h x = 2 * log x) →
  (∃ x, 1/ℯ ≤ x ∧ x ≤ ℯ ∧ g x = -h x) →
  1 ≤ a ∧ a ≤ ℯ^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l3932_393243


namespace NUMINAMATH_CALUDE_postman_pete_miles_l3932_393256

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer where
  max_reading : Nat
  resets : Nat
  final_reading : Nat

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : Nat :=
  p.resets * (p.max_reading + 1) + p.final_reading

/-- Converts steps to miles given a steps-per-mile rate --/
def steps_to_miles (steps : Nat) (steps_per_mile : Nat) : Nat :=
  steps / steps_per_mile

/-- Theorem stating the total miles walked by Postman Pete --/
theorem postman_pete_miles :
  let p : Pedometer := { max_reading := 99999, resets := 50, final_reading := 25000 }
  let total_miles := steps_to_miles (total_steps p) 1500
  total_miles = 3350 := by
  sorry

end NUMINAMATH_CALUDE_postman_pete_miles_l3932_393256


namespace NUMINAMATH_CALUDE_xor_inequality_iff_even_l3932_393214

-- Define bitwise XOR operation
def bitwise_xor (a b : ℕ) : ℕ := sorry

-- Define the property that needs to be proven
def xor_inequality_property (a : ℕ) : Prop :=
  ∀ x y : ℕ, x > y → y ≥ 0 → bitwise_xor x (a * x) ≠ bitwise_xor y (a * y)

-- Theorem statement
theorem xor_inequality_iff_even (a : ℕ) :
  a > 0 → (xor_inequality_property a ↔ Even a) :=
sorry

end NUMINAMATH_CALUDE_xor_inequality_iff_even_l3932_393214


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l3932_393215

/-- The probability of selecting two chips of different colors from a bag with replacement -/
theorem two_different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ)
  (h_blue : blue = 6)
  (h_red : red = 5)
  (h_yellow : yellow = 4) :
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow = 148 / 225 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_chips_probability_l3932_393215


namespace NUMINAMATH_CALUDE_xiaohui_wins_l3932_393237

/-- Represents a student with their scores -/
structure Student where
  name : String
  mandarin : ℕ
  sports : ℕ
  tourism : ℕ

/-- Calculates the weighted score for a student given the weights -/
def weightedScore (s : Student) (w1 w2 w3 : ℕ) : ℚ :=
  (s.mandarin * w1 + s.sports * w2 + s.tourism * w3 : ℚ) / (w1 + w2 + w3 : ℚ)

/-- The theorem stating that Xiaohui wins -/
theorem xiaohui_wins : 
  let xiaocong : Student := ⟨"Xiaocong", 80, 90, 72⟩
  let xiaohui : Student := ⟨"Xiaohui", 90, 80, 70⟩
  weightedScore xiaohui 4 3 3 > weightedScore xiaocong 4 3 3 := by
  sorry


end NUMINAMATH_CALUDE_xiaohui_wins_l3932_393237


namespace NUMINAMATH_CALUDE_abc_sum_product_range_l3932_393203

theorem abc_sum_product_range (a b c : ℝ) (h : a + b + c = 3) :
  ∃ S : Set ℝ, S = Set.Iic 3 ∧ ∀ x : ℝ, x ∈ S ↔ ∃ a' b' c' : ℝ, a' + b' + c' = 3 ∧ a' * b' + a' * c' + b' * c' = x :=
sorry

end NUMINAMATH_CALUDE_abc_sum_product_range_l3932_393203


namespace NUMINAMATH_CALUDE_select_shoes_result_l3932_393234

/-- The number of ways to select 4 shoes from 5 pairs, with exactly one pair included -/
def select_shoes (n : ℕ) : ℕ :=
  let total_pairs := 5
  let pairs_to_choose := 1
  let single_shoes := n - 2 * pairs_to_choose
  let remaining_pairs := total_pairs - pairs_to_choose
  (total_pairs.choose pairs_to_choose) *
  (remaining_pairs.choose single_shoes) *
  (2^pairs_to_choose * 2^single_shoes)

theorem select_shoes_result : select_shoes 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_shoes_result_l3932_393234


namespace NUMINAMATH_CALUDE_sum_of_squares_and_cubes_l3932_393232

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  (∃ x y : ℤ, a^2 - 2*b = x^2 + y^2) ∧
  (∃ u v : ℤ, 3*a*b - a^3 = u^3 + v^3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_cubes_l3932_393232


namespace NUMINAMATH_CALUDE_power_equation_solution_l3932_393297

theorem power_equation_solution :
  ∃ (x : ℕ), (12 : ℝ)^x * 6^4 / 432 = 5184 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3932_393297


namespace NUMINAMATH_CALUDE_g_value_l3932_393294

-- Define the polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^3 - 2*x - 2
axiom sum_eq : ∀ x, f x + g x = -2 + x

-- State the theorem
theorem g_value : g = fun x ↦ -x^3 + 3*x := by sorry

end NUMINAMATH_CALUDE_g_value_l3932_393294


namespace NUMINAMATH_CALUDE_max_value_fraction_l3932_393220

theorem max_value_fraction (x : ℝ) :
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ 15 ∧
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3932_393220


namespace NUMINAMATH_CALUDE_temperature_84_latest_time_l3932_393288

/-- Temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The time when the temperature is 84 degrees -/
def temperature_84 (t : ℝ) : Prop := temperature t = 84

/-- The latest time when the temperature is 84 degrees -/
def latest_time_84 : ℝ := 11

theorem temperature_84_latest_time :
  temperature_84 latest_time_84 ∧
  ∀ t, t > latest_time_84 → ¬(temperature_84 t) :=
sorry

end NUMINAMATH_CALUDE_temperature_84_latest_time_l3932_393288


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l3932_393233

/-- Proves that the total amount of ethanol in a fuel tank is 30 gallons given specific conditions -/
theorem ethanol_in_fuel_tank (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) 
  (h1 : tank_capacity = 214)
  (h2 : fuel_a_volume = 106)
  (h3 : fuel_a_ethanol_percent = 0.12)
  (h4 : fuel_b_ethanol_percent = 0.16) :
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let total_ethanol := fuel_a_volume * fuel_a_ethanol_percent + fuel_b_volume * fuel_b_ethanol_percent
  total_ethanol = 30 := by
sorry


end NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l3932_393233


namespace NUMINAMATH_CALUDE_tangent_line_intercept_product_minimum_l3932_393235

/-- The minimum product of x and y intercepts of a tangent line to the unit circle -/
theorem tangent_line_intercept_product_minimum : ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), x^2 + y^2 = 1 → (x / a + y / b = 1 → False) ∨ (x / a + y / b ≠ 1)) →
  a * b ≥ 2 ∧ (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (∀ (x y : ℝ), x^2 + y^2 = 1 → (x / a₀ + y / b₀ = 1 → False) ∨ (x / a₀ + y / b₀ ≠ 1)) ∧
    a₀ * b₀ = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_product_minimum_l3932_393235


namespace NUMINAMATH_CALUDE_grocery_theorem_l3932_393261

def grocery_problem (initial_budget bread_cost candy_cost : ℚ) : ℚ :=
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let turkey_cost := remaining_after_bread_candy / 3
  remaining_after_bread_candy - turkey_cost

theorem grocery_theorem :
  grocery_problem 32 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_grocery_theorem_l3932_393261


namespace NUMINAMATH_CALUDE_bs_sequence_bounded_iff_f_null_l3932_393291

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = |a (n + 1) - a (n + 2)|

def is_bounded_sequence (a : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |a n| ≤ M

def f (a : ℕ → ℝ) (n k : ℕ) : ℝ :=
  a n * a k * (a n - a k)

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (is_bounded_sequence a ↔ ∀ n k : ℕ, f a n k = 0) :=
sorry

end NUMINAMATH_CALUDE_bs_sequence_bounded_iff_f_null_l3932_393291


namespace NUMINAMATH_CALUDE_sin_65pi_over_6_l3932_393240

theorem sin_65pi_over_6 : Real.sin (65 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_65pi_over_6_l3932_393240


namespace NUMINAMATH_CALUDE_f_two_roots_implies_a_gt_three_l3932_393265

/-- The function f(x) = x³ - ax² + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- f has two distinct positive roots -/
def has_two_distinct_positive_roots (a : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

theorem f_two_roots_implies_a_gt_three (a : ℝ) :
  has_two_distinct_positive_roots a → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_f_two_roots_implies_a_gt_three_l3932_393265


namespace NUMINAMATH_CALUDE_logo_enlargement_l3932_393213

/-- Calculates the height of a proportionally enlarged logo -/
def enlargedLogoHeight (originalWidth originalHeight newWidth : ℚ) : ℚ :=
  (newWidth / originalWidth) * originalHeight

/-- Theorem: The enlarged logo height is 8 inches -/
theorem logo_enlargement (originalWidth originalHeight newWidth : ℚ) 
  (h1 : originalWidth = 3)
  (h2 : originalHeight = 2)
  (h3 : newWidth = 12) :
  enlargedLogoHeight originalWidth originalHeight newWidth = 8 := by
  sorry

end NUMINAMATH_CALUDE_logo_enlargement_l3932_393213


namespace NUMINAMATH_CALUDE_triangle_areas_equal_l3932_393206

theorem triangle_areas_equal :
  let a : ℝ := 24
  let b : ℝ := 24
  let c : ℝ := 34
  let right_triangle_area := (1/2) * a * b
  let s := (a + b + c) / 2
  let general_triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  right_triangle_area = general_triangle_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_areas_equal_l3932_393206


namespace NUMINAMATH_CALUDE_equivalent_operations_l3932_393239

theorem equivalent_operations (x : ℚ) : (x * (3/4)) / (3/5) = x * (5/4) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l3932_393239


namespace NUMINAMATH_CALUDE_max_edges_l3932_393242

/-- A square partitioned into convex polygons -/
structure PartitionedSquare where
  n : ℕ  -- number of polygons
  v : ℕ  -- number of vertices
  e : ℕ  -- number of edges

/-- Euler's theorem for partitioned square -/
axiom euler_theorem (ps : PartitionedSquare) : ps.v - ps.e + ps.n = 1

/-- The degree of each vertex is at least 2, except for at most 4 corner vertices -/
axiom vertex_degree (ps : PartitionedSquare) : 2 * ps.e ≥ 3 * ps.v - 4

/-- Theorem: Maximum number of edges in a partitioned square -/
theorem max_edges (ps : PartitionedSquare) : ps.e ≤ 3 * ps.n + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_edges_l3932_393242


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l3932_393275

theorem polygon_interior_exterior_angles_equality (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l3932_393275


namespace NUMINAMATH_CALUDE_min_races_for_top_three_l3932_393241

/-- Represents a race track with a maximum capacity and a set of horses -/
structure RaceTrack where
  maxCapacity : Nat
  totalHorses : Nat

/-- Represents the minimum number of races needed to find the top n fastest horses -/
def minRacesForTopN (track : RaceTrack) (n : Nat) : Nat :=
  sorry

/-- Theorem stating that for a race track with 5 horse capacity and 25 total horses,
    the minimum number of races to find the top 3 fastest horses is 7 -/
theorem min_races_for_top_three (track : RaceTrack) :
  track.maxCapacity = 5 → track.totalHorses = 25 → minRacesForTopN track 3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_races_for_top_three_l3932_393241


namespace NUMINAMATH_CALUDE_george_socks_theorem_l3932_393259

/-- The number of socks George initially had -/
def initial_socks : ℕ := 28

/-- The number of socks George threw away -/
def thrown_away : ℕ := 4

/-- The number of new socks George bought -/
def new_socks : ℕ := 36

/-- The total number of socks George would have after the transactions -/
def final_socks : ℕ := 60

/-- Theorem stating that the initial number of socks is correct -/
theorem george_socks_theorem : 
  initial_socks - thrown_away + new_socks = final_socks :=
by sorry

end NUMINAMATH_CALUDE_george_socks_theorem_l3932_393259


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3932_393263

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 + 2 * Complex.I) * (1 - Complex.I) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3932_393263


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3932_393284

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3932_393284


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l3932_393251

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l3932_393251


namespace NUMINAMATH_CALUDE_pokemon_card_ratio_l3932_393238

theorem pokemon_card_ratio : ∀ (nicole cindy rex : ℕ),
  nicole = 400 →
  rex * 4 = 150 * 4 →
  2 * rex = nicole + cindy →
  cindy * 2 = nicole :=
by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_ratio_l3932_393238


namespace NUMINAMATH_CALUDE_sixteen_is_sixtyfour_percent_of_twentyfive_l3932_393216

theorem sixteen_is_sixtyfour_percent_of_twentyfive :
  ∀ x : ℚ, (16 : ℚ) = 64 / 100 * x → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_is_sixtyfour_percent_of_twentyfive_l3932_393216


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l3932_393262

theorem no_alpha_sequence_exists : ¬∃ (α : ℝ) (a : ℕ → ℝ), 
  (0 < α ∧ α < 1) ∧ 
  (∀ n : ℕ, 0 < a n) ∧
  (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l3932_393262


namespace NUMINAMATH_CALUDE_factor_proof_l3932_393268

theorem factor_proof : 
  (∃ n : ℤ, 65 = 5 * n) ∧ (∃ m : ℤ, 144 = 9 * m) := by sorry

end NUMINAMATH_CALUDE_factor_proof_l3932_393268


namespace NUMINAMATH_CALUDE_complex_square_real_implies_zero_l3932_393208

theorem complex_square_real_implies_zero (x : ℝ) :
  (Complex.I + x)^2 ∈ Set.range Complex.ofReal → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_real_implies_zero_l3932_393208


namespace NUMINAMATH_CALUDE_area_triangle_dbc_l3932_393286

/-- Given a triangle ABC with vertices A(0,8), B(0,0), C(10,0), and midpoints D of AB and E of BC,
    the area of triangle DBC is 20. -/
theorem area_triangle_dbc (A B C D E : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (1 / 2 : ℝ) * 10 * 4 = 20 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_dbc_l3932_393286


namespace NUMINAMATH_CALUDE_function_upper_bound_l3932_393282

theorem function_upper_bound (x : ℝ) (h : x ≥ 1) : (1 + Real.log x) / x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l3932_393282


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3932_393245

theorem quadratic_equation_properties (a b c : ℝ) (ha : a ≠ 0) :
  -- Statement 1
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  -- Statement 2
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + c = 0 ∧ a*y^2 + c = 0 →
    ∃ u v : ℝ, u ≠ v ∧ a*u^2 + b*u + c = 0 ∧ a*v^2 + b*v + c = 0) ∧
  -- Statement 4
  ∃ m n : ℝ, m ≠ n ∧ a*m^2 + b*m + c = a*n^2 + b*n + c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3932_393245


namespace NUMINAMATH_CALUDE_same_color_sock_probability_l3932_393298

def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

theorem same_color_sock_probability :
  let total_combinations := total_socks.choose 2
  let blue_combinations := blue_socks.choose 2
  let green_combinations := green_socks.choose 2
  let red_combinations := red_socks.choose 2
  let same_color_combinations := blue_combinations + green_combinations + red_combinations
  (same_color_combinations : ℚ) / total_combinations = 19 / 45 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_probability_l3932_393298


namespace NUMINAMATH_CALUDE_shooter_probability_l3932_393202

theorem shooter_probability (p_a p_b : ℝ) 
  (h_p_a : p_a = 3/4)
  (h_p_b : p_b = 2/3)
  (h_p_a_range : 0 ≤ p_a ∧ p_a ≤ 1)
  (h_p_b_range : 0 ≤ p_b ∧ p_b ≤ 1) :
  p_a * (1 - p_b) * (1 - p_b) + (1 - p_a) * p_b * (1 - p_b) + (1 - p_a) * (1 - p_b) * p_b = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l3932_393202


namespace NUMINAMATH_CALUDE_hexagon_tileable_with_squares_l3932_393247

-- Define a hexagon type
structure Hexagon :=
  (A B C D E F : ℝ × ℝ)

-- Define the property of being convex
def is_convex (h : Hexagon) : Prop := sorry

-- Define the property of being inscribed
def is_inscribed (h : Hexagon) : Prop := sorry

-- Define perpendicularity of segments
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Define equality of segments
def segments_equal (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

-- Define the property of being tileable with squares
def tileable_with_squares (h : Hexagon) : Prop := sorry

theorem hexagon_tileable_with_squares (h : Hexagon) 
  (convex : is_convex h)
  (inscribed : is_inscribed h)
  (perp_AD_CE : perpendicular h.A h.D h.C h.E)
  (eq_AD_CE : segments_equal h.A h.D h.C h.E)
  (perp_BE_AC : perpendicular h.B h.E h.A h.C)
  (eq_BE_AC : segments_equal h.B h.E h.A h.C)
  (perp_CF_EA : perpendicular h.C h.F h.E h.A)
  (eq_CF_EA : segments_equal h.C h.F h.E h.A) :
  tileable_with_squares h := by
  sorry

end NUMINAMATH_CALUDE_hexagon_tileable_with_squares_l3932_393247


namespace NUMINAMATH_CALUDE_max_plus_min_of_f_l3932_393289

noncomputable def f (x : ℝ) : ℝ := 1 + x / (x^2 + 1)

theorem max_plus_min_of_f : 
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
                (∀ x, N ≤ f x) ∧ (∃ x, f x = N) ∧ 
                (M + N = 2) :=
sorry

end NUMINAMATH_CALUDE_max_plus_min_of_f_l3932_393289


namespace NUMINAMATH_CALUDE_parabola_curve_intersection_l3932_393296

/-- The parabola defined by y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The curve defined by y = k/x where k > 0 -/
def curve (k x y : ℝ) : Prop := k > 0 ∧ y = k / x

/-- The focus of the parabola y = (1/4)x^2 -/
def focus : ℝ × ℝ := (0, 1)

/-- A point is on both the parabola and the curve -/
def intersection_point (k x y : ℝ) : Prop :=
  parabola x y ∧ curve k x y

/-- The line from a point to the focus is perpendicular to the y-axis -/
def perpendicular_to_y_axis (x y : ℝ) : Prop :=
  x = (focus.1 - x)

theorem parabola_curve_intersection (k : ℝ) :
  (∃ x y : ℝ, intersection_point k x y ∧ perpendicular_to_y_axis x y) →
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_curve_intersection_l3932_393296


namespace NUMINAMATH_CALUDE_not_proportional_D_l3932_393229

-- Define the equations
def equation_A (x y : ℝ) : Prop := x + y = 5
def equation_B (x y : ℝ) : Prop := 4 * x * y = 12
def equation_C (x y : ℝ) : Prop := x = 3 * y
def equation_D (x y : ℝ) : Prop := 4 * x + 2 * y = 8
def equation_E (x y : ℝ) : Prop := x / y = 4

-- Define direct and inverse proportionality
def directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Theorem statement
theorem not_proportional_D :
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_A x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_B x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_C x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_E x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  ¬(∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_D x y ↔ y = f x) ∧
                 (directly_proportional f ∨ inversely_proportional f)) :=
by sorry

end NUMINAMATH_CALUDE_not_proportional_D_l3932_393229


namespace NUMINAMATH_CALUDE_container_initial_percentage_l3932_393210

theorem container_initial_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 60 →
  added_water = 27 →
  final_fraction = 3/4 →
  (capacity * final_fraction - added_water) / capacity * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_container_initial_percentage_l3932_393210


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l3932_393278

theorem complex_magnitude_squared (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l3932_393278


namespace NUMINAMATH_CALUDE_bounded_region_area_l3932_393283

-- Define the equation
def equation (x y : ℝ) : Prop :=
  y^2 + 3*x*y + 60*|x| = 600

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation p.1 p.2}

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem bounded_region_area : area bounded_region = 800 := by sorry

end NUMINAMATH_CALUDE_bounded_region_area_l3932_393283


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3932_393249

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

-- Theorem statement
theorem unique_solution_exists :
  ∃! y : ℝ, otimes 2 y = 20 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3932_393249


namespace NUMINAMATH_CALUDE_sum_of_roots_l3932_393269

theorem sum_of_roots (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3932_393269


namespace NUMINAMATH_CALUDE_marla_horse_purchase_time_l3932_393266

/-- Represents the exchange rates and Marla's scavenging abilities in the post-apocalyptic wasteland -/
structure WastelandEconomy where
  lizard_to_caps : ℕ
  lizards_to_water : ℕ
  water_to_lizards : ℕ
  horse_to_water : ℕ
  daily_scavenge : ℕ
  nightly_cost : ℕ

/-- Calculates the number of days it takes Marla to collect enough bottle caps to buy a horse -/
def days_to_buy_horse (e : WastelandEconomy) : ℕ :=
  let caps_per_lizard := e.lizard_to_caps
  let water_per_horse := e.horse_to_water
  let lizards_per_horse := (water_per_horse * e.water_to_lizards) / e.lizards_to_water
  let caps_per_horse := lizards_per_horse * caps_per_lizard
  let daily_savings := e.daily_scavenge - e.nightly_cost
  caps_per_horse / daily_savings

/-- Theorem stating that it takes Marla 24 days to collect enough bottle caps to buy a horse -/
theorem marla_horse_purchase_time :
  days_to_buy_horse {
    lizard_to_caps := 8,
    lizards_to_water := 3,
    water_to_lizards := 5,
    horse_to_water := 80,
    daily_scavenge := 20,
    nightly_cost := 4
  } = 24 := by
  sorry

end NUMINAMATH_CALUDE_marla_horse_purchase_time_l3932_393266


namespace NUMINAMATH_CALUDE_power_of_power_l3932_393248

theorem power_of_power (x : ℝ) : (x^5)^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3932_393248


namespace NUMINAMATH_CALUDE_people_in_room_l3932_393227

/-- Proves that given the conditions in the problem, the number of people in the room is 67 -/
theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs →  -- Three-fifths of people are seated in four-fifths of chairs
  chairs - (4 : ℚ) / 5 * chairs = 10 →           -- 10 chairs are empty
  people = 67 := by
sorry


end NUMINAMATH_CALUDE_people_in_room_l3932_393227


namespace NUMINAMATH_CALUDE_apples_on_tree_l3932_393295

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 9

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := initial_apples - picked_apples

theorem apples_on_tree : remaining_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_on_tree_l3932_393295


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_m_value_l3932_393224

-- Define the point P
def P (m : ℤ) : ℝ × ℝ := (2 - m, m - 4)

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_third_quadrant_m_value :
  ∀ m : ℤ, in_third_quadrant (P m) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_m_value_l3932_393224
