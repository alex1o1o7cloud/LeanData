import Mathlib

namespace triangle_side_length_l371_37190

theorem triangle_side_length (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) →
  (|a + b - c| + |a - b - c| = 10) → 
  b = 5 := by
sorry

end triangle_side_length_l371_37190


namespace fraction_equality_l371_37100

theorem fraction_equality (a b : ℝ) (h : a ≠ b) :
  (a^2 - b^2) / (a - b)^2 = (a + b) / (a - b) := by
  sorry

end fraction_equality_l371_37100


namespace complex_product_real_l371_37191

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := x - I
  (z₁ * z₂).im = 0 → x = 1 := by
sorry

end complex_product_real_l371_37191


namespace range_of_a_open_interval_l371_37117

theorem range_of_a_open_interval :
  (∃ a : ℝ, ∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ ∃ a : ℝ, -2 < a ∧ a < 2 :=
by sorry

end range_of_a_open_interval_l371_37117


namespace unique_plane_through_skew_lines_l371_37193

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a simplified representation

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane in 3D space
  -- This is a simplified representation

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines to be skew
  sorry

/-- Predicate to check if a plane passes through a line -/
def passes_through (p : Plane3D) (l : Line3D) : Prop :=
  -- Define the condition for a plane to pass through a line
  sorry

/-- Predicate to check if a plane is parallel to a line -/
def is_parallel_to (p : Plane3D) (l : Line3D) : Prop :=
  -- Define the condition for a plane to be parallel to a line
  sorry

/-- Theorem stating the existence and uniqueness of a plane passing through one skew line and parallel to another -/
theorem unique_plane_through_skew_lines (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! p : Plane3D, passes_through p l1 ∧ is_parallel_to p l2 :=
sorry

end unique_plane_through_skew_lines_l371_37193


namespace no_positive_integer_solutions_l371_37152

theorem no_positive_integer_solutions :
  ¬∃ (x y z : ℕ+), x^2 + y^2 = 7 * z^2 :=
by sorry

end no_positive_integer_solutions_l371_37152


namespace right_triangle_with_bisected_hypotenuse_l371_37186

/-- A triangle with vertices A, B, and C -/
structure Triangle (V : Type*) where
  A : V
  B : V
  C : V

/-- A circle with center O and radius r -/
structure Circle (V : Type*) where
  O : V
  r : ℝ

/-- The property of being a right-angled triangle -/
def IsRightAngled {V : Type*} (t : Triangle V) : Prop :=
  sorry

/-- The property that a circle is constructed on a line segment as its diameter -/
def CircleOnDiameter {V : Type*} (c : Circle V) (A B : V) : Prop :=
  sorry

/-- The property that a point is the midpoint of a line segment -/
def IsMidpoint {V : Type*} (M A B : V) : Prop :=
  sorry

/-- The measure of an angle in degrees -/
def AngleMeasure {V : Type*} (A B C : V) : ℝ :=
  sorry

theorem right_triangle_with_bisected_hypotenuse 
  {V : Type*} (t : Triangle V) (c : Circle V) (M : V) :
  IsRightAngled t →
  CircleOnDiameter c t.A t.C →
  IsMidpoint M t.A t.B →
  AngleMeasure t.B t.A t.C = 45 ∧ 
  AngleMeasure t.A t.B t.C = 45 ∧ 
  AngleMeasure t.A t.C t.B = 90 :=
sorry

end right_triangle_with_bisected_hypotenuse_l371_37186


namespace fraction_equals_ratio_l371_37183

def numerator_terms : List Nat := [12, 28, 44, 60, 76]
def denominator_terms : List Nat := [8, 24, 40, 56, 72]

def fraction_term (n : Nat) : Rat :=
  (n^4 + 400 : Rat) / 1

theorem fraction_equals_ratio : 
  (List.prod (numerator_terms.map fraction_term)) / 
  (List.prod (denominator_terms.map fraction_term)) = 
  6712 / 148 := by sorry

end fraction_equals_ratio_l371_37183


namespace half_abs_diff_squares_l371_37128

theorem half_abs_diff_squares : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_l371_37128


namespace irrational_ratio_transformation_l371_37194

theorem irrational_ratio_transformation : ∃ x y : ℝ, 
  (Irrational x) ∧ (Irrational y) ∧ (x ≠ y) ∧ ((7 + x) / (11 + y) = 3 / 4) := by
  sorry

end irrational_ratio_transformation_l371_37194


namespace intersection_A_B_l371_37149

def A : Set ℝ := {x | x^2 - 4 > 0}
def B : Set ℝ := {x | x + 2 < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | x < -2} := by sorry

end intersection_A_B_l371_37149


namespace det_cyclic_matrix_cubic_roots_l371_37113

/-- Given a cubic equation x³ - 2x² + px + q = 0 with roots a, b, and c,
    the determinant of the matrix [[a,b,c],[b,c,a],[c,a,b]] is -p - 8 -/
theorem det_cyclic_matrix_cubic_roots (p q : ℝ) (a b c : ℝ) 
    (h₁ : a^3 - 2*a^2 + p*a + q = 0)
    (h₂ : b^3 - 2*b^2 + p*b + q = 0)
    (h₃ : c^3 - 2*c^2 + p*c + q = 0) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a,b,c; b,c,a; c,a,b]
  Matrix.det M = -p - 8 := by
sorry

end det_cyclic_matrix_cubic_roots_l371_37113


namespace parabola_vertex_in_fourth_quadrant_l371_37134

/-- Given a parabola y = 2x^2 + ax - 5 where a < 0, its vertex is in the fourth quadrant -/
theorem parabola_vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + a * x - 5
  let vertex_x : ℝ := -a / 4
  let vertex_y : ℝ := f vertex_x
  vertex_x > 0 ∧ vertex_y < 0 := by
  sorry

end parabola_vertex_in_fourth_quadrant_l371_37134


namespace purchase_equation_l371_37107

/-- 
Given a group of people jointly purchasing an item, where:
- Contributing 8 units per person results in an excess of 3 units
- Contributing 7 units per person results in a shortage of 4 units
Prove that the number of people satisfies the equation 8x - 3 = 7x + 4
-/
theorem purchase_equation (x : ℕ) 
  (h1 : 8 * x - 3 = (8 * x - 3)) 
  (h2 : 7 * x + 4 = (7 * x + 4)) : 
  8 * x - 3 = 7 * x + 4 := by
  sorry

end purchase_equation_l371_37107


namespace arithmetic_geometric_sequence_properties_l371_37177

def arithmetic_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℚ)
  (h_seq : arithmetic_geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end arithmetic_geometric_sequence_properties_l371_37177


namespace line_m_equation_l371_37139

-- Define the xy-plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a line in the xy-plane
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define the reflection of a point about a line
def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define the given conditions
def problem_setup :=
  ∃ (ℓ m : Line) (P P' P'' : Point),
    ℓ ≠ m ∧
    ℓ.a * 0 + ℓ.b * 0 + ℓ.c = 0 ∧
    m.a * 0 + m.b * 0 + m.c = 0 ∧
    ℓ = Line.mk 5 (-1) 0 ∧
    P = Point.mk (-1) 4 ∧
    P'' = Point.mk 4 1 ∧
    P' = reflect P ℓ ∧
    P'' = reflect P' m

-- State the theorem
theorem line_m_equation (h : problem_setup) :
  ∃ (m : Line), m = Line.mk 2 (-3) 0 :=
sorry

end line_m_equation_l371_37139


namespace secure_app_theorem_l371_37123

/-- Represents an online store application -/
structure OnlineStoreApp where
  paymentGateway : Bool
  dataEncryption : Bool
  transitEncryption : Bool
  codeObfuscation : Bool
  rootedDeviceRestriction : Bool
  antivirusAgent : Bool

/-- Defines the security level of an application -/
def securityLevel (app : OnlineStoreApp) : ℕ :=
  (if app.paymentGateway then 1 else 0) +
  (if app.dataEncryption then 1 else 0) +
  (if app.transitEncryption then 1 else 0) +
  (if app.codeObfuscation then 1 else 0) +
  (if app.rootedDeviceRestriction then 1 else 0) +
  (if app.antivirusAgent then 1 else 0)

/-- Defines a secure application -/
def isSecure (app : OnlineStoreApp) : Prop :=
  securityLevel app = 6

/-- Theorem: An online store app with all security measures implemented is secure -/
theorem secure_app_theorem (app : OnlineStoreApp) 
  (h1 : app.paymentGateway = true)
  (h2 : app.dataEncryption = true)
  (h3 : app.transitEncryption = true)
  (h4 : app.codeObfuscation = true)
  (h5 : app.rootedDeviceRestriction = true)
  (h6 : app.antivirusAgent = true) : 
  isSecure app :=
by
  sorry


end secure_app_theorem_l371_37123


namespace a_b_equality_l371_37162

theorem a_b_equality (a b : ℝ) 
  (h1 : a * b = 1) 
  (h2 : (a + b + 2) / 4 = 1 / (a + 1) + 1 / (b + 1)) : 
  a = 1 ∧ b = 1 := by
sorry

end a_b_equality_l371_37162


namespace log_base_2_derivative_l371_37131

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) :=
sorry

end log_base_2_derivative_l371_37131


namespace unique_digit_system_solution_l371_37137

theorem unique_digit_system_solution (a b c t x y : ℕ) 
  (unique_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ t ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ t ∧ a ≠ x ∧ a ≠ y ∧
                    b ≠ c ∧ b ≠ t ∧ b ≠ x ∧ b ≠ y ∧
                    c ≠ t ∧ c ≠ x ∧ c ≠ y ∧
                    t ≠ x ∧ t ≠ y ∧
                    x ≠ y)
  (eq1 : a + b = x)
  (eq2 : x + c = t)
  (eq3 : t + a = y)
  (eq4 : b + c + y = 20) :
  t = 10 := by
sorry

end unique_digit_system_solution_l371_37137


namespace triangle_side_length_l371_37133

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (h3 : B = π / 3) :
  let b := Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)
  b = 2 * Real.sqrt 3 := by sorry

end triangle_side_length_l371_37133


namespace quadrilateral_angle_theorem_l371_37156

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a predicate for convex quadrilaterals
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a predicate for correspondingly equal sides
def equalSides (q1 q2 : Quadrilateral) : Prop :=
  sideLength q1.A q1.B = sideLength q2.A q2.B ∧
  sideLength q1.B q1.C = sideLength q2.B q2.C ∧
  sideLength q1.C q1.D = sideLength q2.C q2.D ∧
  sideLength q1.D q1.A = sideLength q2.D q2.A

theorem quadrilateral_angle_theorem (q1 q2 : Quadrilateral) 
  (h_convex1 : isConvex q1) (h_convex2 : isConvex q2) 
  (h_equal_sides : equalSides q1 q2) 
  (h_angle_A : angle q1.D q1.A q1.B > angle q2.D q2.A q2.B) :
  angle q1.A q1.B q1.C < angle q2.A q2.B q2.C ∧
  angle q1.B q1.C q1.D > angle q2.B q2.C q2.D ∧
  angle q1.C q1.D q1.A < angle q2.C q2.D q2.A :=
sorry

end quadrilateral_angle_theorem_l371_37156


namespace quadratic_polynomial_inequality_l371_37102

/-- A quadratic polynomial with non-negative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: For any quadratic polynomial with non-negative coefficients and any real numbers x and y,
    the square of the polynomial evaluated at xy is less than or equal to 
    the product of the polynomial evaluated at x^2 and y^2 -/
theorem quadratic_polynomial_inequality (p : QuadraticPolynomial) (x y : ℝ) :
  (p.eval (x * y))^2 ≤ (p.eval (x^2)) * (p.eval (y^2)) := by
  sorry

end quadratic_polynomial_inequality_l371_37102


namespace mortezas_wish_impossible_l371_37114

theorem mortezas_wish_impossible :
  ¬ ∃ (x₁ x₂ x₃ x₄ x₅ x₆ S P : ℝ),
    (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ x₄) ∧ (x₁ ≠ x₅) ∧ (x₁ ≠ x₆) ∧
    (x₂ ≠ x₃) ∧ (x₂ ≠ x₄) ∧ (x₂ ≠ x₅) ∧ (x₂ ≠ x₆) ∧
    (x₃ ≠ x₄) ∧ (x₃ ≠ x₅) ∧ (x₃ ≠ x₆) ∧
    (x₄ ≠ x₅) ∧ (x₄ ≠ x₆) ∧
    (x₅ ≠ x₆) ∧
    ((x₁ + x₂ + x₃ = S) ∨ (x₁ * x₂ * x₃ = P)) ∧
    ((x₂ + x₃ + x₄ = S) ∨ (x₂ * x₃ * x₄ = P)) ∧
    ((x₃ + x₄ + x₅ = S) ∨ (x₃ * x₄ * x₅ = P)) ∧
    ((x₄ + x₅ + x₆ = S) ∨ (x₄ * x₅ * x₆ = P)) ∧
    ((x₅ + x₆ + x₁ = S) ∨ (x₅ * x₆ * x₁ = P)) ∧
    ((x₆ + x₁ + x₂ = S) ∨ (x₆ * x₁ * x₂ = P)) :=
by sorry

end mortezas_wish_impossible_l371_37114


namespace min_value_x_plus_y_l371_37171

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > y) (h2 : y > 0) 
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) : x + y ≥ 25 / 3 := by
  sorry

end min_value_x_plus_y_l371_37171


namespace unique_base_twelve_l371_37176

/-- Given a base b ≥ 10, this function checks if the equation 166 × 56 = 8590 is valid in base b -/
def is_valid_equation (b : ℕ) : Prop :=
  b ≥ 10 ∧ 
  (1 * b^2 + 6 * b + 6) * (5 * b + 6) = 8 * b^3 + 5 * b^2 + 9 * b + 0

/-- Theorem stating that 12 is the only base ≥ 10 satisfying the equation -/
theorem unique_base_twelve : 
  (∃ (b : ℕ), is_valid_equation b) ∧ 
  (∀ (b : ℕ), is_valid_equation b → b = 12) := by
  sorry

#check unique_base_twelve

end unique_base_twelve_l371_37176


namespace rectangles_must_be_squares_l371_37165

theorem rectangles_must_be_squares (n : ℕ) (is_prime : ℕ → Prop) 
  (total_squares : ℕ) (h_prime : is_prime total_squares) : 
  ∀ (a b : ℕ) (h_rect : ∀ i : Fin n, ∃ (k : ℕ), a * b = (total_squares / n) * k^2), a = b :=
by
  sorry

end rectangles_must_be_squares_l371_37165


namespace f_is_bitwise_or_l371_37118

/-- Bitwise OR operation for positive integers -/
def bitwiseOr (a b : ℕ+) : ℕ+ := sorry

/-- The function f we want to prove is equal to bitwise OR -/
noncomputable def f : ℕ+ → ℕ+ → ℕ+ := sorry

/-- Condition (i): f(a,b) ≤ a + b for all a, b ∈ ℤ⁺ -/
axiom condition_i (a b : ℕ+) : f a b ≤ a + b

/-- Condition (ii): f(a,f(b,c)) = f(f(a,b),c) for all a, b, c ∈ ℤ⁺ -/
axiom condition_ii (a b c : ℕ+) : f a (f b c) = f (f a b) c

/-- Condition (iii): Both (f(a,b) choose a) and (f(a,b) choose b) are odd numbers for all a, b ∈ ℤ⁺ -/
axiom condition_iii (a b : ℕ+) : Odd (Nat.choose (f a b) a) ∧ Odd (Nat.choose (f a b) b)

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- The main theorem: f is equal to bitwise OR -/
theorem f_is_bitwise_or : ∀ (a b : ℕ+), f a b = bitwiseOr a b := by sorry

end f_is_bitwise_or_l371_37118


namespace journey_length_l371_37150

theorem journey_length : 
  ∀ (x : ℚ), 
  (x / 4 : ℚ) + 25 + (x / 6 : ℚ) = x → 
  x = 300 / 7 := by
sorry

end journey_length_l371_37150


namespace puppies_remaining_l371_37140

def initial_puppies : ℕ := 12
def puppies_given_away : ℕ := 7

theorem puppies_remaining (initial : ℕ) (given_away : ℕ) :
  initial = initial_puppies →
  given_away = puppies_given_away →
  initial - given_away = 5 :=
by
  sorry

end puppies_remaining_l371_37140


namespace expression_equals_28_l371_37187

theorem expression_equals_28 : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 := by
  sorry

end expression_equals_28_l371_37187


namespace jakes_friend_candy_and_euros_l371_37173

/-- Proves the number of candies Jake's friend can purchase and the amount in Euros he will receive --/
theorem jakes_friend_candy_and_euros :
  let feeding_allowance : ℝ := 4
  let fraction_given : ℝ := 1/4
  let candy_price : ℝ := 0.2
  let discount : ℝ := 0.15
  let exchange_rate : ℝ := 0.85
  
  let money_given := feeding_allowance * fraction_given
  let discounted_price := candy_price * (1 - discount)
  let candies_purchasable := ⌊money_given / discounted_price⌋
  let euros_received := money_given * exchange_rate
  
  (candies_purchasable = 5) ∧ (euros_received = 0.85) :=
by
  sorry

#check jakes_friend_candy_and_euros

end jakes_friend_candy_and_euros_l371_37173


namespace special_polyhedron_hexagon_count_l371_37148

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  -- V: vertices, E: edges, F: faces, P: pentagonal faces, H: hexagonal faces
  V : ℕ
  E : ℕ
  F : ℕ
  P : ℕ
  H : ℕ
  vertex_degree : V * 3 = E * 2
  face_types : F = P + H
  euler : V - E + F = 2
  edge_count : E * 2 = P * 5 + H * 6
  both_face_types : P > 0 ∧ H > 0

/-- Theorem: In a SpecialPolyhedron, the number of hexagonal faces is at least 2 -/
theorem special_polyhedron_hexagon_count (poly : SpecialPolyhedron) : poly.H ≥ 2 := by
  sorry

end special_polyhedron_hexagon_count_l371_37148


namespace fraction_conversion_equivalence_l371_37154

theorem fraction_conversion_equivalence (x : ℚ) : 
  (x + 1) / (2 / 5) - (2 / 10 * x - 1) / (7 / 10) = 1 ↔ 
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 := by sorry

end fraction_conversion_equivalence_l371_37154


namespace ln_inequality_implies_p_range_l371_37121

theorem ln_inequality_implies_p_range (p : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x ≤ p * x - 1) → p ≥ 1 := by
  sorry

end ln_inequality_implies_p_range_l371_37121


namespace exist_a_b_l371_37178

theorem exist_a_b : ∃ (a b : ℝ),
  (a < 0) ∧
  (b = -a) ∧
  (b > 9/4) ∧
  (∀ x : ℝ, x < -1 → a * x > b) ∧
  (∀ y : ℝ, y^2 + 3*y + b > 0) := by
  sorry

end exist_a_b_l371_37178


namespace correct_scientific_notation_l371_37199

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- The number to be represented -/
def number : ℕ := 2400000

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation := {
  a := 2.4
  n := 6
  h1 := by sorry
  h2 := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem correct_scientific_notation : 
  (scientificForm.a * (10 : ℝ) ^ scientificForm.n) = number := by sorry

end correct_scientific_notation_l371_37199


namespace basketball_free_throws_l371_37122

theorem basketball_free_throws (two_point_shots three_point_shots free_throws : ℕ) : 
  (2 * two_point_shots = 3 * three_point_shots) →
  (free_throws = two_point_shots + 1) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 61) →
  free_throws = 13 := by
  sorry

end basketball_free_throws_l371_37122


namespace factorial_gcd_l371_37147

theorem factorial_gcd : Nat.gcd (Nat.factorial 6) (Nat.factorial 9) = Nat.factorial 6 := by
  sorry

end factorial_gcd_l371_37147


namespace decagon_triangles_l371_37104

theorem decagon_triangles : ∀ (n : ℕ), n = 10 → (n.choose 3) = 120 := by
  sorry

end decagon_triangles_l371_37104


namespace largest_prime_factor_f9_div_f3_l371_37153

def f (n : ℕ) : ℕ := (3^n + 1) / 2

theorem largest_prime_factor_f9_div_f3 :
  let ratio : ℕ := f 9 / f 3
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ ratio ∧ p = 37 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ ratio → q ≤ p :=
by sorry

end largest_prime_factor_f9_div_f3_l371_37153


namespace janice_work_hours_janice_work_hours_unique_l371_37180

/-- Calculates the total pay for a given number of hours worked -/
def totalPay (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    10 * hours
  else
    400 + 15 * (hours - 40)

/-- Theorem stating that 60 hours of work results in $700 pay -/
theorem janice_work_hours :
  totalPay 60 = 700 :=
by sorry

/-- Theorem stating that 60 is the unique number of hours that results in $700 pay -/
theorem janice_work_hours_unique :
  ∀ h : ℕ, totalPay h = 700 → h = 60 :=
by sorry

end janice_work_hours_janice_work_hours_unique_l371_37180


namespace largest_n_dividing_30_factorial_l371_37188

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem largest_n_dividing_30_factorial : 
  (∀ n : ℕ, n > 7 → ¬(divides (18^n) (factorial 30))) ∧ 
  (divides (18^7) (factorial 30)) := by
sorry

end largest_n_dividing_30_factorial_l371_37188


namespace no_eight_digit_six_times_l371_37141

theorem no_eight_digit_six_times : ¬ ∃ (N : ℕ), 
  (10000000 ≤ N) ∧ (N < 100000000) ∧
  (∃ (p q : ℕ), N = 10000 * p + q ∧ q < 10000 ∧ 10000 * q + p = 6 * N) :=
sorry

end no_eight_digit_six_times_l371_37141


namespace diameter_endpoint_theorem_l371_37105

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle --/
structure Diameter where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Theorem: Given a circle with center at (0,0) and one endpoint of a diameter at (3,4),
    the other endpoint of the diameter is at (-3, -4) --/
theorem diameter_endpoint_theorem (c : Circle) (d : Diameter) :
  c.center = (0, 0) ∧ d.endpoint1 = (3, 4) →
  d.endpoint2 = (-3, -4) := by
  sorry

end diameter_endpoint_theorem_l371_37105


namespace sixth_side_formula_l371_37112

/-- A hexagon described around a circle with six sides -/
structure CircumscribedHexagon where
  sides : Fin 6 → ℝ
  is_positive : ∀ i, sides i > 0

/-- The property that the sum of alternating sides in a circumscribed hexagon is constant -/
def alternating_sum_constant (h : CircumscribedHexagon) : Prop :=
  h.sides 0 + h.sides 2 + h.sides 4 = h.sides 1 + h.sides 3 + h.sides 5

theorem sixth_side_formula (h : CircumscribedHexagon) 
  (sum_constant : alternating_sum_constant h) :
  h.sides 5 = h.sides 0 - h.sides 1 + h.sides 2 - h.sides 3 + h.sides 4 := by
  sorry

end sixth_side_formula_l371_37112


namespace smallest_z_value_l371_37106

/-- Given consecutive positive integers w, x, y, z where w = n and z = w + 4,
    the smallest z satisfying w^3 + x^3 + y^3 = z^3 is 9. -/
theorem smallest_z_value (n : ℕ) (w x y z : ℕ) : 
  w = n → 
  x = n + 1 → 
  y = n + 2 → 
  z = n + 4 → 
  w^3 + x^3 + y^3 = z^3 → 
  z ≥ 9 :=
by sorry

end smallest_z_value_l371_37106


namespace intersection_of_M_and_N_l371_37129

def M : Set Int := {-1, 1}
def N : Set Int := {-1, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end intersection_of_M_and_N_l371_37129


namespace no_simultaneous_negative_polynomials_l371_37142

theorem no_simultaneous_negative_polynomials :
  ∀ (m n : ℝ), ¬(3 * m^2 + 4 * m * n - 2 * n^2 < 0 ∧ -m^2 - 4 * m * n + 3 * n^2 < 0) := by
  sorry

end no_simultaneous_negative_polynomials_l371_37142


namespace quadratic_complete_square_l371_37132

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + b)^2 = c
    where b and c are integers, prove that b + c = 11 -/
theorem quadratic_complete_square (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + b)^2 = c) → b + c = 11 := by
  sorry

end quadratic_complete_square_l371_37132


namespace cone_lateral_surface_area_l371_37175

/-- The lateral surface area of a cone with base radius 3 and lateral surface that unfolds into a semicircle is 18π. -/
theorem cone_lateral_surface_area (r : ℝ) (l : ℝ) : 
  r = 3 →
  π * l = 2 * π * r →
  π * r * l = 18 * π :=
by sorry

end cone_lateral_surface_area_l371_37175


namespace smallest_number_with_conditions_l371_37143

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 2) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 2) ∧ 
  (n % 11 = 0) ∧ 
  (∀ m : ℕ, m > 1 → 
    (m % 3 = 2) → 
    (m % 4 = 2) → 
    (m % 5 = 2) → 
    (m % 6 = 2) → 
    (m % 11 = 0) → 
    m ≥ n) ∧
  n = 242 :=
by sorry

end smallest_number_with_conditions_l371_37143


namespace cos_difference_of_complex_exponentials_l371_37138

theorem cos_difference_of_complex_exponentials 
  (θ φ : ℝ) 
  (h1 : Complex.exp (Complex.I * θ) = 4/5 + 3/5 * Complex.I)
  (h2 : Complex.exp (Complex.I * φ) = 5/13 + 12/13 * Complex.I) : 
  Real.cos (θ - φ) = -16/65 := by
  sorry

end cos_difference_of_complex_exponentials_l371_37138


namespace parallel_vectors_sum_l371_37192

/-- Given two parallel vectors a and b, prove that x + y = -9 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![-1, 2, 1]
  let b : Fin 3 → ℝ := ![3, x, y]
  (∃ (k : ℝ), ∀ i, b i = k * (a i)) →
  x + y = -9 := by
  sorry

end parallel_vectors_sum_l371_37192


namespace yan_distance_ratio_l371_37109

theorem yan_distance_ratio :
  ∀ (a b v : ℝ),
  a > 0 → b > 0 → v > 0 →
  (b / v = a / v + (a + b) / (7 * v)) →
  (a / b = 3 / 4) :=
by
  sorry

end yan_distance_ratio_l371_37109


namespace height_of_cube_with_corner_cut_l371_37167

/-- The height of a cube with one corner cut off -/
theorem height_of_cube_with_corner_cut (s : ℝ) (h : s = 2) :
  let diagonal := s * Real.sqrt 3
  let cut_face_side := diagonal / Real.sqrt 2
  let cut_face_area := Real.sqrt 3 / 4 * cut_face_side^2
  let pyramid_volume := 1 / 6 * s^3
  let remaining_height := s - (3 * pyramid_volume) / cut_face_area
  remaining_height = 2 - Real.sqrt 3 := by
sorry

end height_of_cube_with_corner_cut_l371_37167


namespace local_minimum_of_f_l371_37198

/-- The function f(x) = x³ - 4x² + 4x -/
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x

/-- The local minimum value of f(x) is 0 -/
theorem local_minimum_of_f :
  ∃ (a : ℝ), ∀ (x : ℝ), ∃ (ε : ℝ), ε > 0 ∧ 
    (∀ (y : ℝ), |y - a| < ε → f y ≥ f a) ∧
    f a = 0 :=
sorry

end local_minimum_of_f_l371_37198


namespace summer_birth_year_divisibility_l371_37195

theorem summer_birth_year_divisibility : ∃ (x y : ℕ), 
  x < y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (1961 - x) % x = 0 ∧ 
  (1961 - y) % y = 0 := by
sorry

end summer_birth_year_divisibility_l371_37195


namespace amy_small_gardens_l371_37101

def small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem amy_small_gardens :
  small_gardens 101 47 6 = 9 :=
by sorry

end amy_small_gardens_l371_37101


namespace square_difference_l371_37126

theorem square_difference (m n : ℝ) (h1 : m + n = 3) (h2 : m - n = 4) : m^2 - n^2 = 12 := by
  sorry

end square_difference_l371_37126


namespace p_necessary_not_sufficient_for_q_l371_37166

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (0 < x ∧ x < 7) → (|x - 2| < 5)) ∧
  (∃ x : ℝ, |x - 2| < 5 ∧ ¬(0 < x ∧ x < 7)) :=
by sorry

end p_necessary_not_sufficient_for_q_l371_37166


namespace gcd_of_256_162_450_l371_37111

theorem gcd_of_256_162_450 : Nat.gcd 256 (Nat.gcd 162 450) = 2 := by sorry

end gcd_of_256_162_450_l371_37111


namespace partner_investment_period_l371_37168

/-- Given two partners P and Q with investment and profit ratios, and Q's investment period,
    calculate P's investment period. -/
theorem partner_investment_period
  (investment_ratio_p investment_ratio_q : ℕ)
  (profit_ratio_p profit_ratio_q : ℕ)
  (q_months : ℕ)
  (h_investment : investment_ratio_p * 5 = investment_ratio_q * 7)
  (h_profit : profit_ratio_p * 9 = profit_ratio_q * 7)
  (h_q_months : q_months = 9) :
  ∃ (p_months : ℕ),
    p_months * profit_ratio_q * investment_ratio_q =
    q_months * profit_ratio_p * investment_ratio_p ∧
    p_months = 5 :=
by sorry

end partner_investment_period_l371_37168


namespace circle_line_distance_range_l371_37163

/-- Given a circle x^2 + y^2 = 9 and a line y = x + b, if there are exactly two points
    on the circle that have a distance of 1 to the line, then b is in the range
    (-4√2, -2√2) ∪ (2√2, 4√2) -/
theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), 
    p.1^2 + p.2^2 = 9 ∧ 
    q.1^2 + q.2^2 = 9 ∧ 
    p ≠ q ∧
    (abs (p.2 - p.1 - b) / Real.sqrt 2 = 1) ∧
    (abs (q.2 - q.1 - b) / Real.sqrt 2 = 1)) →
  (b > 2 * Real.sqrt 2 ∧ b < 4 * Real.sqrt 2) ∨ 
  (b < -2 * Real.sqrt 2 ∧ b > -4 * Real.sqrt 2) :=
by sorry

end circle_line_distance_range_l371_37163


namespace election_result_l371_37151

/-- The percentage of votes received by candidate A out of the total valid votes -/
def candidate_A_percentage : ℝ := 65

/-- The percentage of invalid votes out of the total votes -/
def invalid_vote_percentage : ℝ := 15

/-- The total number of votes cast in the election -/
def total_votes : ℕ := 560000

/-- The number of valid votes polled in favor of candidate A -/
def votes_for_candidate_A : ℕ := 309400

theorem election_result :
  (candidate_A_percentage / 100) * ((100 - invalid_vote_percentage) / 100) * total_votes = votes_for_candidate_A := by
  sorry

end election_result_l371_37151


namespace mean_home_runs_l371_37179

def number_of_players : ℕ := 3 + 5 + 3 + 1

def total_home_runs : ℕ := 5 * 3 + 8 * 5 + 9 * 3 + 11 * 1

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (number_of_players : ℚ) = 7.75 := by
  sorry

end mean_home_runs_l371_37179


namespace interference_facts_l371_37172

/-- A fact about light -/
inductive LightFact
  | SignalTransmission
  | SurfaceFlatness
  | PrismSpectrum
  | OilFilmColors

/-- Predicate to determine if a light fact involves interference -/
def involves_interference (fact : LightFact) : Prop :=
  match fact with
  | LightFact.SurfaceFlatness => true
  | LightFact.OilFilmColors => true
  | _ => false

/-- Theorem stating that only facts 2 and 4 involve interference -/
theorem interference_facts :
  (∀ f : LightFact, involves_interference f ↔ (f = LightFact.SurfaceFlatness ∨ f = LightFact.OilFilmColors)) :=
by sorry

end interference_facts_l371_37172


namespace nahco3_equals_nano3_l371_37196

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction :=
  (naHCO3 : Moles)
  (hNO3 : Moles)
  (naNO3 : Moles)
  (h2O : Moles)
  (cO2 : Moles)

/-- The chemical equation is balanced -/
axiom balanced_equation (r : Reaction) : r.naHCO3 = r.hNO3 ∧ r.naHCO3 = r.naNO3

/-- The number of moles of HNO3 combined equals the number of moles of NaNO3 formed -/
axiom hno3_equals_nano3 (r : Reaction) : r.hNO3 = r.naNO3

/-- The stoichiometric ratio of NaHCO3 to NaNO3 is 1:1 -/
axiom stoichiometric_ratio (r : Reaction) : r.naHCO3 = r.naNO3

/-- Theorem: The number of moles of NaHCO3 combined equals the number of moles of NaNO3 formed -/
theorem nahco3_equals_nano3 (r : Reaction) : r.naHCO3 = r.naNO3 := by
  sorry

end nahco3_equals_nano3_l371_37196


namespace wednesday_distance_l371_37155

/-- Terese's running schedule for the week -/
structure RunningSchedule where
  monday : Float
  tuesday : Float
  wednesday : Float
  thursday : Float

/-- Theorem: Given Terese's running schedule and average distance, prove she runs 3.6 miles on Wednesday -/
theorem wednesday_distance (schedule : RunningSchedule) 
  (h1 : schedule.monday = 4.2)
  (h2 : schedule.tuesday = 3.8)
  (h3 : schedule.thursday = 4.4)
  (h4 : (schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday) / 4 = 4) :
  schedule.wednesday = 3.6 := by
  sorry


end wednesday_distance_l371_37155


namespace arithmetic_sequence_1023rd_term_l371_37164

def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℚ) : Prop :=
  ∃ (d : ℚ), a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d

def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_1023rd_term (p r : ℚ) :
  arithmetic_sequence (2*p) 15 (4*p+r) (4*p-r) →
  nth_term (2*p) ((4*p-r) - (4*p+r)) 1023 = 61215 / 14 :=
by sorry

end arithmetic_sequence_1023rd_term_l371_37164


namespace sum_first_11_odd_numbers_l371_37115

theorem sum_first_11_odd_numbers : 
  (Finset.range 11).sum (fun i => 2 * i + 1) = 121 := by
  sorry

end sum_first_11_odd_numbers_l371_37115


namespace deck_size_l371_37169

/-- The number of cards in a deck of playing cards. -/
def num_cards : ℕ := 52

/-- The number of hearts on each card. -/
def hearts_per_card : ℕ := 4

/-- The cost of each cow in dollars. -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars. -/
def total_cost : ℕ := 83200

/-- The number of cows in Devonshire. -/
def num_cows : ℕ := total_cost / cost_per_cow

/-- The number of hearts in the deck. -/
def num_hearts : ℕ := num_cows / 2

theorem deck_size :
  num_cards = num_hearts / hearts_per_card ∧
  num_cows = 2 * num_hearts ∧
  num_cows * cost_per_cow = total_cost :=
by sorry

end deck_size_l371_37169


namespace football_tournament_scheduling_l371_37189

theorem football_tournament_scheduling (n : ℕ) (h_even : Even n) :
  ∃ schedule : Fin (n - 1) → Fin n → Fin n,
    (∀ round : Fin (n - 1), ∀ team : Fin n, 
      schedule round team ≠ team ∧ 
      (∀ other_team : Fin n, schedule round team = other_team → schedule round other_team = team)) ∧
    (∀ team1 team2 : Fin n, team1 ≠ team2 → 
      ∃! round : Fin (n - 1), schedule round team1 = team2 ∨ schedule round team2 = team1) := by
  sorry

end football_tournament_scheduling_l371_37189


namespace towel_area_decrease_l371_37103

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let new_length := 0.7 * L
  let new_breadth := 0.75 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.475
  := by sorry

end towel_area_decrease_l371_37103


namespace p_necessary_not_sufficient_for_q_l371_37144

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x| ≤ 2
def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ 
  (∃ x : ℝ, p x ∧ ¬(q x)) :=
by sorry

end p_necessary_not_sufficient_for_q_l371_37144


namespace heavy_wash_water_usage_l371_37110

/-- Represents the amount of water used for different types of washes -/
structure WashingMachine where
  heavy_wash : ℚ
  regular_wash : ℚ
  light_wash : ℚ

/-- Calculates the total water usage for a given washing machine and set of loads -/
def total_water_usage (wm : WashingMachine) (heavy_loads bleach_loads : ℕ) : ℚ :=
  wm.heavy_wash * heavy_loads +
  wm.regular_wash * 3 +
  wm.light_wash * (1 + bleach_loads)

/-- Theorem stating that the heavy wash uses 20 gallons of water -/
theorem heavy_wash_water_usage :
  ∃ (wm : WashingMachine),
    wm.regular_wash = 10 ∧
    wm.light_wash = 2 ∧
    total_water_usage wm 2 2 = 76 ∧
    wm.heavy_wash = 20 := by
  sorry

end heavy_wash_water_usage_l371_37110


namespace james_run_time_l371_37158

/-- The time it takes James to run 100 meters given John's performance and their speed differences -/
theorem james_run_time (john_total_time john_initial_distance john_initial_time total_distance
  james_initial_distance james_initial_time speed_difference : ℝ)
  (h1 : john_total_time = 13)
  (h2 : john_initial_distance = 4)
  (h3 : john_initial_time = 1)
  (h4 : total_distance = 100)
  (h5 : james_initial_distance = 10)
  (h6 : james_initial_time = 2)
  (h7 : speed_difference = 2)
  : ∃ james_total_time : ℝ, james_total_time = 11 :=
by sorry

end james_run_time_l371_37158


namespace binary_representation_theorem_l371_37135

def is_multiple_of_17 (n : ℕ) : Prop := ∃ k : ℕ, n = 17 * k

def binary_ones_count (n : ℕ) : ℕ := (n.digits 2).count 1

def binary_zeros_count (n : ℕ) : ℕ := (n.digits 2).length - binary_ones_count n

theorem binary_representation_theorem (n : ℕ) 
  (h1 : is_multiple_of_17 n) 
  (h2 : binary_ones_count n = 3) : 
  (binary_zeros_count n ≥ 6) ∧ 
  (binary_zeros_count n = 7 → Even n) := by
sorry

end binary_representation_theorem_l371_37135


namespace basketball_lineup_theorem_l371_37161

/-- The number of ways to choose 7 starters from 18 players, including a set of 3 triplets,
    with exactly two of the triplets in the starting lineup. -/
def basketball_lineup_count : ℕ := sorry

/-- The total number of players on the team. -/
def total_players : ℕ := 18

/-- The number of triplets in the team. -/
def triplets : ℕ := 3

/-- The number of starters to be chosen. -/
def starters : ℕ := 7

/-- The number of triplets that must be in the starting lineup. -/
def triplets_in_lineup : ℕ := 2

theorem basketball_lineup_theorem : 
  basketball_lineup_count = (Nat.choose triplets triplets_in_lineup) * 
    (Nat.choose (total_players - triplets) (starters - triplets_in_lineup)) := by sorry

end basketball_lineup_theorem_l371_37161


namespace subtraction_of_large_numbers_l371_37127

theorem subtraction_of_large_numbers : 
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end subtraction_of_large_numbers_l371_37127


namespace fraction_sum_equals_percentage_l371_37184

theorem fraction_sum_equals_percentage : (4/20 : ℚ) + (8/200 : ℚ) + (12/2000 : ℚ) = (246/1000 : ℚ) := by
  sorry

end fraction_sum_equals_percentage_l371_37184


namespace compound_molecular_weight_l371_37174

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of a compound is the sum of the atomic weights of its constituent atoms -/
def molecular_weight (n_weight o_weight : ℝ) (n_count o_count : ℕ) : ℝ :=
  n_weight * n_count + o_weight * o_count

/-- The molecular weight of the compound is 44.02 amu -/
theorem compound_molecular_weight :
  molecular_weight nitrogen_weight oxygen_weight nitrogen_count oxygen_count = 44.02 := by
  sorry

end compound_molecular_weight_l371_37174


namespace birch_trees_not_adjacent_probability_l371_37157

def total_trees : ℕ := 14
def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 5

theorem birch_trees_not_adjacent_probability : 
  let total_arrangements := Nat.choose total_trees birch_trees
  let non_birch_trees := maple_trees + oak_trees
  let valid_arrangements := Nat.choose (non_birch_trees + 1) birch_trees
  (valid_arrangements : ℚ) / total_arrangements = 18 / 143 := by
  sorry

end birch_trees_not_adjacent_probability_l371_37157


namespace product_mod_twelve_l371_37170

theorem product_mod_twelve : (95 * 97) % 12 = 11 := by
  sorry

end product_mod_twelve_l371_37170


namespace negation_of_universal_proposition_l371_37130

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) := by
  sorry

end negation_of_universal_proposition_l371_37130


namespace law_school_applicants_l371_37197

theorem law_school_applicants (total : ℕ) (pol_sci : ℕ) (high_gpa : ℕ) (pol_sci_high_gpa : ℕ) 
  (h1 : total = 40)
  (h2 : pol_sci = 15)
  (h3 : high_gpa = 20)
  (h4 : pol_sci_high_gpa = 5) :
  total - pol_sci - high_gpa + pol_sci_high_gpa = 10 :=
by sorry

end law_school_applicants_l371_37197


namespace even_operations_l371_37145

theorem even_operations (n : ℤ) (h : Even n) :
  Even (5 * n) ∧ Even (n ^ 2) ∧ Even (n ^ 3) := by
  sorry

end even_operations_l371_37145


namespace max_ab_line_circle_intersection_l371_37124

/-- Given a line ax + by - 8 = 0 (where a > 0 and b > 0) intersecting the circle x² + y² - 2x - 4y = 0
    with a chord length of 2√5, the maximum value of ab is 8. -/
theorem max_ab_line_circle_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x + b * y - 8 = 0 → x^2 + y^2 - 2*x - 4*y = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + b * y₁ - 8 = 0 ∧ 
    a * x₂ + b * y₂ - 8 = 0 ∧
    x₁^2 + y₁^2 - 2*x₁ - 4*y₁ = 0 ∧
    x₂^2 + y₂^2 - 2*x₂ - 4*y₂ = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 20) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' * b' ≤ a * b) →
  a * b = 8 :=
by sorry

end max_ab_line_circle_intersection_l371_37124


namespace student_event_combinations_l371_37146

theorem student_event_combinations : 
  let num_students : ℕ := 4
  let num_events : ℕ := 3
  num_events ^ num_students = 81 := by sorry

end student_event_combinations_l371_37146


namespace partial_fraction_decomposition_l371_37125

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 2) :
  5 * x / ((x - 4) * (x - 2)^2) = 5 / (x - 4) + (-5) / (x - 2) + (-5) / (x - 2)^2 :=
by sorry

end partial_fraction_decomposition_l371_37125


namespace books_bought_at_fair_l371_37159

theorem books_bought_at_fair (initial_books final_books : ℕ) 
  (h1 : initial_books = 9)
  (h2 : final_books = 12) :
  final_books - initial_books = 3 := by
sorry

end books_bought_at_fair_l371_37159


namespace square_plus_difference_of_squares_l371_37136

theorem square_plus_difference_of_squares (x y : ℝ) : 
  x^2 + (y - x) * (y + x) = y^2 := by
  sorry

end square_plus_difference_of_squares_l371_37136


namespace zero_exists_in_interval_l371_37181

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x - 3

-- State the theorem
theorem zero_exists_in_interval :
  ∃ c ∈ Set.Ioo (1/2 : ℝ) 1, f c = 0 :=
sorry

end zero_exists_in_interval_l371_37181


namespace treys_chores_l371_37108

theorem treys_chores (task_duration : ℕ) (total_time : ℕ) (shower_tasks : ℕ) (dinner_tasks : ℕ) :
  task_duration = 10 →
  total_time = 120 →
  shower_tasks = 1 →
  dinner_tasks = 4 →
  (total_time / task_duration) - shower_tasks - dinner_tasks = 7 :=
by sorry

end treys_chores_l371_37108


namespace correlation_coefficient_and_fit_l371_37185

/-- Represents the correlation coefficient in regression analysis -/
def correlation_coefficient : ℝ := sorry

/-- Represents the goodness of fit of a regression model -/
def goodness_of_fit : ℝ := sorry

/-- States that as the absolute value of the correlation coefficient 
    approaches 1, the goodness of fit improves -/
theorem correlation_coefficient_and_fit :
  ∀ ε > 0, ∃ δ > 0, ∀ R : ℝ,
    |R| > 1 - δ → goodness_of_fit > 1 - ε :=
sorry

end correlation_coefficient_and_fit_l371_37185


namespace quadratic_distinct_roots_l371_37160

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < (1/4 : ℝ) :=
by sorry

end quadratic_distinct_roots_l371_37160


namespace page_lines_increase_l371_37119

theorem page_lines_increase (original : ℕ) (increased : ℕ) (percentage : ℚ) : 
  percentage = 100/3 →
  increased = 240 →
  increased = original + (percentage / 100 * original).floor →
  increased - original = 60 :=
by
  sorry

end page_lines_increase_l371_37119


namespace xyz_value_l371_37182

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b - c) * (x + 2))
  (eq2 : b = (a - c) * (y + 2))
  (eq3 : c = (a - b) * (z + 2))
  (eq4 : x * y + x * z + y * z = 12)
  (eq5 : x + y + z = 6) :
  x * y * z = 7 := by
sorry

end xyz_value_l371_37182


namespace power_equation_solution_l371_37120

theorem power_equation_solution (n : ℕ) : 2^n = 8^20 → n = 60 := by
  sorry

end power_equation_solution_l371_37120


namespace rhombus_area_l371_37116

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 13) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 130 := by
  sorry

end rhombus_area_l371_37116
