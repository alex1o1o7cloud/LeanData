import Mathlib

namespace NUMINAMATH_CALUDE_ticket_circle_circumference_l1543_154330

/-- The circumference of a circle formed by overlapping tickets -/
theorem ticket_circle_circumference
  (ticket_length : ℝ)
  (overlap : ℝ)
  (num_tickets : ℕ)
  (h1 : ticket_length = 10.4)
  (h2 : overlap = 3.5)
  (h3 : num_tickets = 16) :
  (ticket_length - overlap) * num_tickets = 110.4 :=
by sorry

end NUMINAMATH_CALUDE_ticket_circle_circumference_l1543_154330


namespace NUMINAMATH_CALUDE_parabola_coeff_sum_l1543_154366

/-- A parabola with equation y = px^2 + qx + r, vertex (-3, 7), and passing through (-6, 4) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := -3
  vertex_y : ℝ := 7
  point_x : ℝ := -6
  point_y : ℝ := 4
  eq_at_vertex : 7 = p * (-3)^2 + q * (-3) + r
  eq_at_point : 4 = p * (-6)^2 + q * (-6) + r

/-- The sum of coefficients p, q, and r for the parabola is 7/3 -/
theorem parabola_coeff_sum (par : Parabola) : par.p + par.q + par.r = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coeff_sum_l1543_154366


namespace NUMINAMATH_CALUDE_factorial_base_312_b3_is_zero_l1543_154388

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Checks if a list of coefficients is a valid factorial base representation -/
def isValidFactorialBase (coeffs : List ℕ) : Prop :=
  ∀ (i : ℕ), i < coeffs.length → coeffs[i]! ≤ i + 1

/-- Computes the value represented by a list of coefficients in factorial base -/
def valueFromFactorialBase (coeffs : List ℕ) : ℕ :=
  coeffs.enum.foldl (fun acc (i, b) => acc + b * factorial (i + 1)) 0

/-- Theorem: The factorial base representation of 312 has b₃ = 0 -/
theorem factorial_base_312_b3_is_zero :
  ∃ (coeffs : List ℕ),
    isValidFactorialBase coeffs ∧
    valueFromFactorialBase coeffs = 312 ∧
    coeffs.length > 3 ∧
    coeffs[2]! = 0 :=
by sorry

end NUMINAMATH_CALUDE_factorial_base_312_b3_is_zero_l1543_154388


namespace NUMINAMATH_CALUDE_vertical_asymptotes_sum_l1543_154383

theorem vertical_asymptotes_sum (p q : ℚ) : 
  (∀ x, 4 * x^2 + 7 * x + 3 = 0 ↔ x = p ∨ x = q) →
  p + q = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_vertical_asymptotes_sum_l1543_154383


namespace NUMINAMATH_CALUDE_f_properties_l1543_154338

def f (x : ℝ) : ℝ := x^3 + 3*x^2

theorem f_properties :
  (f (-1) = 2) →
  (deriv f (-1) = -3) →
  (∃ (y : ℝ), y ∈ Set.Icc (-16) 4 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-4) 0 ∧ f x = y) ∧
  (∀ (t : ℝ), (∀ (x y : ℝ), t ≤ x ∧ x < y ∧ y ≤ t + 1 → f x > f y) ↔ t ∈ Set.Icc (-2) (-1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1543_154338


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1543_154351

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The terms a_1 + 1, a_3 + 2, and a_5 + 3 form a geometric sequence with ratio q -/
def GeometricSubsequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 3 + 2) = (a 1 + 1) * q ∧ (a 5 + 3) = (a 3 + 2) * q

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ArithmeticSequence a) (h2 : GeometricSubsequence a q) : q = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1543_154351


namespace NUMINAMATH_CALUDE_angle_A_measure_max_area_l1543_154300

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Condition that sides are positive
  ha : a > 0
  hb : b > 0
  hc : c > 0
  -- Condition that angles are between 0 and π
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Condition that angles sum to π
  hsum : A + B + C = π
  -- Law of cosines
  hlawA : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  hlawB : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  hlawC : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Part 1
theorem angle_A_measure (t : Triangle) (h : t.b^2 + t.c^2 - t.a^2 + t.b*t.c = 0) :
  t.A = 2*π/3 := by sorry

-- Part 2
theorem max_area (t : Triangle) (h1 : t.b^2 + t.c^2 - t.a^2 + t.b*t.c = 0) (h2 : t.a = Real.sqrt 3) :
  (t.b * t.c * Real.sin t.A / 2) ≤ Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_A_measure_max_area_l1543_154300


namespace NUMINAMATH_CALUDE_original_denominator_problem_l1543_154320

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →  -- Ensure the original fraction is well-defined
  (3 + 4 : ℚ) / (d + 4) = 1 / 3 → 
  d = 17 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l1543_154320


namespace NUMINAMATH_CALUDE_tan_30_degrees_l1543_154390

theorem tan_30_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l1543_154390


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1543_154329

theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 2) * x₁^2 + 2 * x₁ - 1 = 0 ∧ (k - 2) * x₂^2 + 2 * x₂ - 1 = 0) ↔
  (k > 1 ∧ k ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1543_154329


namespace NUMINAMATH_CALUDE_candy_distribution_l1543_154359

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) 
  (h1 : total_candy = 24) (h2 : num_friends = 5) :
  let pieces_to_remove := total_candy % num_friends
  let remaining_candy := total_candy - pieces_to_remove
  pieces_to_remove = 
    Nat.min pieces_to_remove (total_candy - (remaining_candy / num_friends) * num_friends) ∧
  (remaining_candy / num_friends) * num_friends = remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1543_154359


namespace NUMINAMATH_CALUDE_opposite_face_of_y_l1543_154319

-- Define a cube net
structure CubeNet where
  faces : Finset Char
  y_face : Char
  foldable : Bool

-- Define a property for opposite faces in a cube
def opposite_faces (net : CubeNet) (face1 face2 : Char) : Prop :=
  face1 ∈ net.faces ∧ face2 ∈ net.faces ∧ face1 ≠ face2

-- Theorem statement
theorem opposite_face_of_y (net : CubeNet) :
  net.faces = {'W', 'X', 'Y', 'Z', 'V', net.y_face} →
  net.foldable = true →
  net.y_face ≠ 'V' →
  opposite_faces net net.y_face 'V' :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_y_l1543_154319


namespace NUMINAMATH_CALUDE_min_value_expression_l1543_154331

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1543_154331


namespace NUMINAMATH_CALUDE_pattern_A_cannot_fold_into_cube_l1543_154367

/-- Represents a pattern of squares -/
inductive Pattern
  | A  -- Five squares in a cross shape
  | B  -- Four squares in a "T" shape
  | C  -- Six squares in a "T" shape with an additional square
  | D  -- Three squares in a straight line

/-- Number of squares in a pattern -/
def squareCount (p : Pattern) : Nat :=
  match p with
  | .A => 5
  | .B => 4
  | .C => 6
  | .D => 3

/-- Number of squares required to form a cube -/
def cubeSquareCount : Nat := 6

/-- Checks if a pattern can be folded into a cube -/
def canFoldIntoCube (p : Pattern) : Prop :=
  squareCount p = cubeSquareCount ∧ 
  (p ≠ Pattern.A) -- Pattern A cannot be closed even with 5 squares

/-- Theorem: Pattern A cannot be folded into a cube -/
theorem pattern_A_cannot_fold_into_cube : 
  ¬ (canFoldIntoCube Pattern.A) := by
  sorry


end NUMINAMATH_CALUDE_pattern_A_cannot_fold_into_cube_l1543_154367


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1543_154352

/-- Two vectors in ℝ² are parallel if the ratio of their components is constant -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

/-- The sum of two vectors in ℝ² -/
def vec_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_vectors_sum :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → vec_sum a b = (-2, -1) := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1543_154352


namespace NUMINAMATH_CALUDE_butternut_figurines_eq_four_l1543_154344

/-- The number of figurines that can be created from a block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from a block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- The number of blocks of basswood Adam has -/
def basswood_blocks : ℕ := 15

/-- The number of blocks of butternut wood Adam has -/
def butternut_blocks : ℕ := 20

/-- The number of blocks of Aspen wood Adam has -/
def aspen_blocks : ℕ := 20

/-- The number of figurines that can be created from a block of butternut wood -/
def butternut_figurines : ℕ := (total_figurines - basswood_blocks * basswood_figurines - aspen_blocks * aspen_figurines) / butternut_blocks

theorem butternut_figurines_eq_four : butternut_figurines = 4 := by
  sorry

end NUMINAMATH_CALUDE_butternut_figurines_eq_four_l1543_154344


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l1543_154384

/-- Proposition p: There exists a real number x such that x - 2 > log_10(x) -/
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x / Real.log 10

/-- Proposition q: For all real numbers x, x^2 > 0 -/
def q : Prop := ∀ x : ℝ, x^2 > 0

/-- Theorem: The conjunction of p and (not q) is true -/
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l1543_154384


namespace NUMINAMATH_CALUDE_clarinet_players_count_l1543_154368

/-- Represents the number of people in an orchestra section -/
structure OrchestraSection where
  count : ℕ

/-- Represents the composition of an orchestra -/
structure Orchestra where
  total : ℕ
  percussion : OrchestraSection
  brass : OrchestraSection
  strings : OrchestraSection
  flutes : OrchestraSection
  maestro : OrchestraSection
  clarinets : OrchestraSection

/-- Given an orchestra with the specified composition, prove that the number of clarinet players is 3 -/
theorem clarinet_players_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.percussion.count = 1)
  (h3 : o.brass.count = 7)
  (h4 : o.strings.count = 5)
  (h5 : o.flutes.count = 4)
  (h6 : o.maestro.count = 1)
  (h7 : o.total = o.percussion.count + o.brass.count + o.strings.count + o.flutes.count + o.maestro.count + o.clarinets.count) :
  o.clarinets.count = 3 := by
  sorry

end NUMINAMATH_CALUDE_clarinet_players_count_l1543_154368


namespace NUMINAMATH_CALUDE_parabola_properties_l1543_154328

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 6*x - 1

-- Theorem statement
theorem parabola_properties :
  -- Vertex coordinates
  (∃ (x y : ℝ), x = -3 ∧ y = -10 ∧ ∀ (t : ℝ), f t ≥ f x) ∧
  -- Axis of symmetry
  (∀ (x : ℝ), f (x - 3) = f (-x - 3)) ∧
  -- Y-axis intersection point
  f 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1543_154328


namespace NUMINAMATH_CALUDE_set_operations_l1543_154362

-- Define the sets A and B
def A : Set ℝ := {y | -1 < y ∧ y < 4}
def B : Set ℝ := {y | 0 < y ∧ y < 5}

-- Theorem statements
theorem set_operations :
  (Set.univ \ B = {y | y ≤ 0 ∨ y ≥ 5}) ∧
  (A ∪ B = {y | -1 < y ∧ y < 5}) ∧
  (A ∩ B = {y | 0 < y ∧ y < 4}) ∧
  (A ∩ (Set.univ \ B) = {y | -1 < y ∧ y ≤ 0}) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = {y | y ≤ -1 ∨ y ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1543_154362


namespace NUMINAMATH_CALUDE_partnership_investment_l1543_154371

/-- Represents a partnership with three partners -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  duration : ℝ
  a_share : ℝ
  b_share : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem partnership_investment (p : Partnership)
  (ha : p.a_investment = 11000)
  (hc : p.c_investment = 23000)
  (hd : p.duration = 8)
  (hsa : p.a_share = 2431)
  (hsb : p.b_share = 3315) :
  p.b_investment = 15000 := by
  sorry


end NUMINAMATH_CALUDE_partnership_investment_l1543_154371


namespace NUMINAMATH_CALUDE_oil_cylinder_capacity_l1543_154315

theorem oil_cylinder_capacity (capacity : ℚ) 
  (h1 : (3 : ℚ) / 4 * capacity = 27.5)
  (h2 : (9 : ℚ) / 10 * capacity = 35) : 
  capacity = 110 / 3 := by sorry

end NUMINAMATH_CALUDE_oil_cylinder_capacity_l1543_154315


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l1543_154327

theorem quadratic_function_inequality (a b c : ℝ) (h1 : b > 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (a + b + c) / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l1543_154327


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1543_154348

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 20*y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 5)

-- Define the asymptotes of the hyperbola
def hyperbola_asymptotes (x y : ℝ) : Prop := (3*x + 4*y = 0) ∨ (3*x - 4*y = 0)

-- Define the standard form of a hyperbola
def hyperbola_standard_form (a b : ℝ) (x y : ℝ) : Prop := y^2/a^2 - x^2/b^2 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∃ (x y : ℝ), parabola x y ∧
  (∃ (fx fy : ℝ), (fx, fy) = parabola_focus) ∧
  hyperbola_asymptotes x y →
  hyperbola_standard_form 3 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1543_154348


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l1543_154395

/-- Profit function for the bookstore --/
def profit (p : ℝ) : ℝ := (p - 2) * (110 - 4 * p)

/-- The optimal price is 15 --/
def optimal_price : ℕ := 15

theorem profit_maximized_at_optimal_price :
  ∀ p : ℕ, p ≤ 22 → profit p ≤ profit optimal_price :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l1543_154395


namespace NUMINAMATH_CALUDE_f_is_odd_l1543_154361

def f (x : ℝ) : ℝ := x^3 - x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l1543_154361


namespace NUMINAMATH_CALUDE_rope_contact_length_l1543_154379

/-- The length of rope in contact with a cylindrical tower, given specific conditions --/
theorem rope_contact_length 
  (rope_length : ℝ) 
  (tower_radius : ℝ) 
  (unicorn_height : ℝ) 
  (free_end_distance : ℝ) 
  (h1 : rope_length = 25) 
  (h2 : tower_radius = 10) 
  (h3 : unicorn_height = 3) 
  (h4 : free_end_distance = 5) : 
  ∃ (contact_length : ℝ), contact_length = rope_length - Real.sqrt 134 := by
  sorry

#check rope_contact_length

end NUMINAMATH_CALUDE_rope_contact_length_l1543_154379


namespace NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l1543_154389

theorem abc_and_fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 2) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_abc_and_fourth_power_sum_l1543_154389


namespace NUMINAMATH_CALUDE_correct_allocation_plans_l1543_154323

/-- Represents the number of factories --/
def num_factories : Nat := 4

/-- Represents the number of classes --/
def num_classes : Nat := 3

/-- Represents the requirement that at least one factory must have a class --/
def must_have_class : Nat := 1

/-- The number of different allocation plans --/
def allocation_plans : Nat := 57

/-- Theorem stating that the number of allocation plans is correct --/
theorem correct_allocation_plans :
  (num_factories = 4) →
  (num_classes = 3) →
  (must_have_class = 1) →
  (allocation_plans = 57) := by
  sorry

end NUMINAMATH_CALUDE_correct_allocation_plans_l1543_154323


namespace NUMINAMATH_CALUDE_andrew_steps_to_meet_ben_l1543_154372

/-- The distance between Andrew's and Ben's houses in feet -/
def distance : ℝ := 21120

/-- The ratio of Ben's speed to Andrew's speed -/
def speed_ratio : ℝ := 3

/-- The length of Andrew's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Andrew takes before meeting Ben -/
def steps : ℕ := 1760

theorem andrew_steps_to_meet_ben :
  (distance / (1 + speed_ratio)) / step_length = steps := by
  sorry

end NUMINAMATH_CALUDE_andrew_steps_to_meet_ben_l1543_154372


namespace NUMINAMATH_CALUDE_smallest_sum_sequence_l1543_154332

theorem smallest_sum_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = d ∧ B - A = d) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = r * B ∧ D = r * C) →  -- B, C, D form a geometric sequence
  C = (7 : ℚ) / 3 * B →  -- C/B = 7/3
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = d ∧ B' - A' = d) →
    (∃ r : ℚ, C' = r * B' ∧ D' = r * C') →
    C' = (7 : ℚ) / 3 * B' →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_sequence_l1543_154332


namespace NUMINAMATH_CALUDE_length_of_angle_bisector_l1543_154364

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 8 ∧ qr = 15 ∧ pr = 17

-- Define the angle bisector PS
def AngleBisector (P Q R S : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let qs := Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2)
  let rs := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  qs / rs = pq / qr

-- Theorem statement
theorem length_of_angle_bisector 
  (P Q R S : ℝ × ℝ) 
  (h1 : Triangle P Q R) 
  (h2 : AngleBisector P Q R S) : 
  Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = Real.sqrt 87.04 :=
by
  sorry

end NUMINAMATH_CALUDE_length_of_angle_bisector_l1543_154364


namespace NUMINAMATH_CALUDE_divisibility_by_nineteen_l1543_154347

theorem divisibility_by_nineteen (n : ℕ+) :
  ∃ k : ℤ, (5 ^ (2 * n.val - 1) : ℤ) + (3 ^ (n.val - 2) : ℤ) * (2 ^ (n.val - 1) : ℤ) = 19 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_nineteen_l1543_154347


namespace NUMINAMATH_CALUDE_fraction_difference_zero_l1543_154360

theorem fraction_difference_zero (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_zero_l1543_154360


namespace NUMINAMATH_CALUDE_point_inside_circle_a_range_l1543_154313

theorem point_inside_circle_a_range (a : ℝ) : 
  (((1 - a)^2 + (1 + a)^2) < 4) → (-1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_a_range_l1543_154313


namespace NUMINAMATH_CALUDE_rotated_line_equation_l1543_154345

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : a * x + b * y + c = 0

/-- Rotates a line counterclockwise by π/2 around its y-axis intersection --/
def rotate_line_pi_over_2 (l : Line) : Line :=
  sorry

theorem rotated_line_equation :
  let original_line : Line := ⟨2, -1, -2, by sorry⟩
  let rotated_line := rotate_line_pi_over_2 original_line
  rotated_line.a = 1 ∧ rotated_line.b = 2 ∧ rotated_line.c = 4 :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l1543_154345


namespace NUMINAMATH_CALUDE_fruit_store_total_weight_l1543_154334

/-- Given a store with apples and pears, where the weight of pears is three times
    that of apples, calculate the total weight of apples and pears. -/
theorem fruit_store_total_weight (apple_weight : ℕ) (pear_weight : ℕ) : 
  apple_weight = 3200 →
  pear_weight = 3 * apple_weight →
  apple_weight + pear_weight = 12800 := by
sorry

end NUMINAMATH_CALUDE_fruit_store_total_weight_l1543_154334


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l1543_154399

theorem arcsin_sin_eq_x_div_3 (x : ℝ) :
  -3 * π / 2 ≤ x ∧ x ≤ 3 * π / 2 →
  (Real.arcsin (Real.sin x) = x / 3 ↔ 
    x = -3 * π / 2 ∨ x = 0 ∨ x = 3 * π / 4 ∨ x = 3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_l1543_154399


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_399_l1543_154346

theorem greatest_prime_factor_of_399 : ∃ p : ℕ, p.Prime ∧ p ∣ 399 ∧ ∀ q : ℕ, q.Prime → q ∣ 399 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_399_l1543_154346


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1543_154356

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 / 5)
  (h4 : Real.cos (α + β) = 5 / 13) :
  (Real.cos β = 63 / 65) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1543_154356


namespace NUMINAMATH_CALUDE_lunch_combo_options_count_l1543_154336

/-- The number of lunch combo options for Terry at the salad bar -/
def lunch_combo_options : ℕ :=
  let lettuce_types : ℕ := 2
  let tomato_types : ℕ := 3
  let olive_types : ℕ := 4
  let soup_types : ℕ := 2
  lettuce_types * tomato_types * olive_types * soup_types

theorem lunch_combo_options_count : lunch_combo_options = 48 := by
  sorry

end NUMINAMATH_CALUDE_lunch_combo_options_count_l1543_154336


namespace NUMINAMATH_CALUDE_distance_between_points_l1543_154339

/-- The distance between two points (-3, 5) and (4, -9) is √245 -/
theorem distance_between_points : Real.sqrt 245 = Real.sqrt ((4 - (-3))^2 + (-9 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1543_154339


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1543_154309

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y / (2 * y + 3 * x) = 1) : 
  x / 2 + y / 3 ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    x₀ * y₀ / (2 * y₀ + 3 * x₀) = 1 ∧ x₀ / 2 + y₀ / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1543_154309


namespace NUMINAMATH_CALUDE_remainder_16_pow_2048_mod_11_l1543_154311

theorem remainder_16_pow_2048_mod_11 : 16^2048 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_16_pow_2048_mod_11_l1543_154311


namespace NUMINAMATH_CALUDE_sqrt_three_addition_l1543_154365

theorem sqrt_three_addition : 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_addition_l1543_154365


namespace NUMINAMATH_CALUDE_base5_242_equals_base10_72_l1543_154354

-- Define a function to convert a base 5 number to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

-- Theorem stating that 242 in base 5 is equal to 72 in base 10
theorem base5_242_equals_base10_72 :
  base5ToBase10 [2, 4, 2] = 72 := by
  sorry

end NUMINAMATH_CALUDE_base5_242_equals_base10_72_l1543_154354


namespace NUMINAMATH_CALUDE_sector_area_l1543_154357

/-- The area of a circular sector with central angle 5π/7 and perimeter 5π+14 is 35π/2 -/
theorem sector_area (r : ℝ) (h1 : r > 0) : 
  (5 / 7 * π * r + 2 * r = 5 * π + 14) →
  (1 / 2 * (5 / 7 * π) * r^2 = 35 * π / 2) := by
sorry


end NUMINAMATH_CALUDE_sector_area_l1543_154357


namespace NUMINAMATH_CALUDE_james_candy_payment_l1543_154393

/-- Proves that James paid $20 for candy given the conditions of the problem -/
theorem james_candy_payment (
  num_packs : ℕ)
  (price_per_pack : ℕ)
  (change_received : ℕ)
  (h1 : num_packs = 3)
  (h2 : price_per_pack = 3)
  (h3 : change_received = 11)
  : num_packs * price_per_pack + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_candy_payment_l1543_154393


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l1543_154350

/-- The height of a tree after n years, given its initial height and growth factor --/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (years : ℕ) : ℝ :=
  initial_height * growth_factor ^ years

/-- Theorem stating that a tree with initial height h quadrupling every year for 4 years
    reaches 256 feet if and only if h = 1 foot --/
theorem tree_growth_theorem (h : ℝ) : 
  tree_height h 4 4 = 256 ↔ h = 1 := by
  sorry

#check tree_growth_theorem

end NUMINAMATH_CALUDE_tree_growth_theorem_l1543_154350


namespace NUMINAMATH_CALUDE_problem_statement_l1543_154312

theorem problem_statement (a b : ℝ) (h1 : a = 2 + Real.sqrt 3) (h2 : b = 2 - Real.sqrt 3) :
  a^2 + 2*a*b - b*(3*a - b) = 13 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1543_154312


namespace NUMINAMATH_CALUDE_squares_sequence_correct_l1543_154301

/-- Represents the number of nonoverlapping unit squares in figure n -/
def squares (n : ℕ) : ℕ :=
  3 * n^2 + n + 1

theorem squares_sequence_correct : 
  squares 0 = 1 ∧ 
  squares 1 = 5 ∧ 
  squares 2 = 15 ∧ 
  squares 3 = 29 ∧ 
  squares 4 = 49 ∧ 
  squares 100 = 30101 :=
by sorry

end NUMINAMATH_CALUDE_squares_sequence_correct_l1543_154301


namespace NUMINAMATH_CALUDE_equation_solution_l1543_154363

theorem equation_solution :
  ∃! x : ℝ, 
    x > 10 ∧
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     15 / (Real.sqrt (x - 10) + 10) = 0) ∧
    x = 35 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1543_154363


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1543_154377

theorem smallest_number_proof (a b c d : ℝ) : 
  b = 4 * a →
  c = 2 * a →
  d = a + b + c →
  (a + b + c + d) / 4 = 77 →
  a = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1543_154377


namespace NUMINAMATH_CALUDE_alcohol_concentration_problem_l1543_154342

theorem alcohol_concentration_problem (vessel1_capacity vessel2_capacity total_liquid final_vessel_capacity : ℝ)
  (vessel2_concentration final_concentration : ℝ) :
  vessel1_capacity = 2 →
  vessel2_capacity = 6 →
  vessel2_concentration = 55 / 100 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  final_concentration = 37 / 100 →
  ∃ initial_concentration : ℝ,
    initial_concentration = 20 / 100 ∧
    initial_concentration * vessel1_capacity + vessel2_concentration * vessel2_capacity =
    final_concentration * final_vessel_capacity :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_problem_l1543_154342


namespace NUMINAMATH_CALUDE_d_share_is_750_l1543_154326

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℝ)

/-- Represents the distribution of money among 5 people -/
structure Distribution :=
  (a b c d e : Share)

/-- The total amount of money to be distributed -/
def total_amount (dist : Distribution) : ℝ :=
  dist.a.amount + dist.b.amount + dist.c.amount + dist.d.amount + dist.e.amount

/-- The condition that the distribution follows the proportion 5 : 2 : 4 : 3 : 1 -/
def proportional_distribution (dist : Distribution) : Prop :=
  5 * dist.b.amount = 2 * dist.a.amount ∧
  5 * dist.c.amount = 4 * dist.a.amount ∧
  5 * dist.d.amount = 3 * dist.a.amount ∧
  5 * dist.e.amount = 1 * dist.a.amount

/-- The condition that the combined share of A and C is 3/5 of the total amount -/
def combined_share_condition (dist : Distribution) : Prop :=
  dist.a.amount + dist.c.amount = 3/5 * total_amount dist

/-- The condition that E gets $250 less than B -/
def e_less_than_b_condition (dist : Distribution) : Prop :=
  dist.b.amount - dist.e.amount = 250

theorem d_share_is_750 (dist : Distribution) 
  (h1 : proportional_distribution dist)
  (h2 : combined_share_condition dist)
  (h3 : e_less_than_b_condition dist) :
  dist.d.amount = 750 := by
  sorry

end NUMINAMATH_CALUDE_d_share_is_750_l1543_154326


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1543_154316

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1543_154316


namespace NUMINAMATH_CALUDE_expression_evaluation_l1543_154387

theorem expression_evaluation : 4^3 - 4 * 4^2 + 6 * 4 - 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1543_154387


namespace NUMINAMATH_CALUDE_complement_of_M_l1543_154391

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M : Mᶜ = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1543_154391


namespace NUMINAMATH_CALUDE_rayman_workout_hours_l1543_154343

/-- Represents the workout hours of Rayman, Junior, and Wolverine in a week --/
structure WorkoutHours where
  rayman : ℝ
  junior : ℝ
  wolverine : ℝ

/-- Defines the relationship between Rayman's, Junior's, and Wolverine's workout hours --/
def valid_workout_hours (h : WorkoutHours) : Prop :=
  h.rayman = h.junior / 2 ∧
  h.wolverine = 2 * (h.rayman + h.junior) ∧
  h.wolverine = 60

/-- Theorem stating that Rayman works out for 10 hours in a week --/
theorem rayman_workout_hours (h : WorkoutHours) (hvalid : valid_workout_hours h) : 
  h.rayman = 10 := by
  sorry

#check rayman_workout_hours

end NUMINAMATH_CALUDE_rayman_workout_hours_l1543_154343


namespace NUMINAMATH_CALUDE_point_C_values_l1543_154324

-- Define the points on the number line
def point_A : ℝ := 2
def point_B : ℝ := -4

-- Define the property of equal distances between adjacent points
def equal_distances (c : ℝ) : Prop :=
  abs (point_A - point_B) = abs (point_B - c) ∧ 
  abs (point_A - point_B) = abs (point_A - c)

-- Theorem statement
theorem point_C_values : 
  ∀ c : ℝ, equal_distances c → (c = -10 ∨ c = 8) :=
by sorry

end NUMINAMATH_CALUDE_point_C_values_l1543_154324


namespace NUMINAMATH_CALUDE_stock_yield_inconsistency_l1543_154397

theorem stock_yield_inconsistency (price : ℝ) (price_pos : price > 0) :
  ¬(∃ (dividend : ℝ), 
    dividend = 0.2 * price ∧ 
    dividend / price = 0.1) :=
by
  sorry

#check stock_yield_inconsistency

end NUMINAMATH_CALUDE_stock_yield_inconsistency_l1543_154397


namespace NUMINAMATH_CALUDE_function_inequality_range_l1543_154353

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - a| + |x + 3*a - 2|
def g (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1

-- State the theorem
theorem function_inequality_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, f a x₁ > g a x₂) ↔ 
  (a ∈ Set.Ioo (-2 - Real.sqrt 5) (-2 + Real.sqrt 5) ∪ Set.Ioo 1 3) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l1543_154353


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1543_154396

theorem imaginary_part_of_complex_expression : 
  let i : ℂ := Complex.I
  (((i^2016) / (2*i - 1)) * i).im = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1543_154396


namespace NUMINAMATH_CALUDE_bobs_weight_l1543_154369

theorem bobs_weight (j b : ℝ) 
  (sum_condition : j + b = 180)
  (diff_condition : b - j = b / 2) : 
  b = 120 := by sorry

end NUMINAMATH_CALUDE_bobs_weight_l1543_154369


namespace NUMINAMATH_CALUDE_calculate_principal_l1543_154382

/-- Given a simple interest, interest rate, and time period, calculate the principal amount. -/
theorem calculate_principal (simple_interest rate time : ℝ) :
  simple_interest = 4020.75 →
  rate = 0.0875 →
  time = 5.5 →
  simple_interest = (8355.00 * rate * time) := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_l1543_154382


namespace NUMINAMATH_CALUDE_store_max_profit_l1543_154375

/-- A store selling clothing with the following conditions:
    - Cost price is 60 yuan per item
    - Selling price must not be lower than the cost price
    - Profit must not exceed 40%
    - Sales volume (y) and selling price (x) follow a linear function y = kx + b
    - When x = 80, y = 40
    - When x = 70, y = 50
    - 60 ≤ x ≤ 84
-/
theorem store_max_profit (x : ℝ) (y : ℝ) (k : ℝ) (b : ℝ) (W : ℝ → ℝ) :
  (∀ x, 60 ≤ x ∧ x ≤ 84) →
  (∀ x, y = k * x + b) →
  (80 * k + b = 40) →
  (70 * k + b = 50) →
  (∀ x, W x = (x - 60) * (k * x + b)) →
  (∃ x₀, ∀ x, 60 ≤ x ∧ x ≤ 84 → W x ≤ W x₀) →
  (∃ x₀, W x₀ = 864 ∧ x₀ = 84) := by
  sorry


end NUMINAMATH_CALUDE_store_max_profit_l1543_154375


namespace NUMINAMATH_CALUDE_total_fudge_eaten_l1543_154340

-- Define the conversion rate from pounds to ounces
def pounds_to_ounces : ℝ → ℝ := (· * 16)

-- Define the amount of fudge eaten by each person in pounds
def tomas_fudge : ℝ := 1.5
def katya_fudge : ℝ := 0.5
def boris_fudge : ℝ := 2

-- Theorem statement
theorem total_fudge_eaten :
  pounds_to_ounces tomas_fudge +
  pounds_to_ounces katya_fudge +
  pounds_to_ounces boris_fudge = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_fudge_eaten_l1543_154340


namespace NUMINAMATH_CALUDE_equation_solution_l1543_154376

theorem equation_solution : ∃ x : ℝ, 24 - 6 = 3 + x ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1543_154376


namespace NUMINAMATH_CALUDE_matching_shoe_probability_l1543_154321

/-- Represents a shoe pair --/
structure ShoePair :=
  (left : Nat)
  (right : Nat)

/-- The probability of selecting a matching pair of shoes --/
def probability_matching_pair (n : Nat) : Rat :=
  if n > 0 then 1 / n else 0

theorem matching_shoe_probability (cabinet : Finset ShoePair) :
  cabinet.card = 3 →
  probability_matching_pair cabinet.card = 1 / 3 := by
  sorry

#eval probability_matching_pair 3

end NUMINAMATH_CALUDE_matching_shoe_probability_l1543_154321


namespace NUMINAMATH_CALUDE_correct_calculation_l1543_154374

theorem correct_calculation (a : ℝ) : 3 * a^2 + 2 * a^2 = 5 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1543_154374


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l1543_154305

theorem set_equality_implies_sum (a b : ℝ) : 
  ({4, a} : Set ℝ) = ({2, a * b} : Set ℝ) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l1543_154305


namespace NUMINAMATH_CALUDE_square_perimeter_l1543_154378

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ strip_perimeter : ℝ, 
    strip_perimeter = 2 * (s + s / 4) ∧ 
    strip_perimeter = 40) →
  4 * s = 64 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1543_154378


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1543_154337

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + a*a + b = 0) ∧ (b^2 + a*b + b = 0) → a = 1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1543_154337


namespace NUMINAMATH_CALUDE_circle_integer_points_l1543_154341

theorem circle_integer_points 
  (center : ℝ × ℝ) 
  (h_center : center = (Real.sqrt 2, Real.sqrt 3)) :
  ∀ (A B : ℤ × ℤ), A ≠ B →
  ¬(∃ (r : ℝ), r > 0 ∧ 
    ((A.1 - center.1)^2 + (A.2 - center.2)^2 = r^2) ∧
    ((B.1 - center.1)^2 + (B.2 - center.2)^2 = r^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_integer_points_l1543_154341


namespace NUMINAMATH_CALUDE_unique_intersection_l1543_154349

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-1, 1-a, 9}

theorem unique_intersection (a : ℝ) : A a ∩ B a = {9} → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l1543_154349


namespace NUMINAMATH_CALUDE_list_price_calculation_l1543_154325

theorem list_price_calculation (list_price : ℝ) : 
  (list_price ≥ 0) →
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) →
  list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_list_price_calculation_l1543_154325


namespace NUMINAMATH_CALUDE_arith_progression_poly_j_value_l1543_154306

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithProgressionPoly where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ (i j : Fin 4), i ≠ j → zeros i ≠ zeros j
  arith_prog : ∃ (a d : ℝ), ∀ (i : Fin 4), zeros i = a + i.val * d
  is_zero : ∀ (i : Fin 4), zeros i ^ 4 + j * (zeros i ^ 2) + k * zeros i + 225 = 0

theorem arith_progression_poly_j_value (p : ArithProgressionPoly) : p.j = -50 := by
  sorry

end NUMINAMATH_CALUDE_arith_progression_poly_j_value_l1543_154306


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1543_154355

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a * b = -15) : 
  a^2 + b^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1543_154355


namespace NUMINAMATH_CALUDE_orange_balls_count_l1543_154386

theorem orange_balls_count (black white : ℕ) (p : ℚ) (orange : ℕ) : 
  black = 7 → 
  white = 6 → 
  p = 38095238095238093 / 100000000000000000 →
  (black : ℚ) / (orange + black + white : ℚ) = p →
  orange = 5 :=
by sorry

end NUMINAMATH_CALUDE_orange_balls_count_l1543_154386


namespace NUMINAMATH_CALUDE_triangle_properties_l1543_154317

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 2 ∧
  cos B = 3/5 →
  (b = 4 → sin A = 2/5) ∧
  (1/2 * a * c * sin B = 4 → b = Real.sqrt 17 ∧ c = 5) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1543_154317


namespace NUMINAMATH_CALUDE_ball_return_theorem_l1543_154314

/-- The number of ways a ball returns to the initial person after n passes among m people. -/
def ball_return_ways (m n : ℕ) : ℚ :=
  ((m - 1)^n : ℚ) / m + ((-1)^n : ℚ) * ((m - 1) : ℚ) / m

/-- Theorem: The number of ways a ball returns to the initial person after n passes among m people,
    where m ≥ 2, is given by ((m-1)^n / m) + ((-1)^n * (m-1) / m) -/
theorem ball_return_theorem (m n : ℕ) (h : m ≥ 2) :
  ∃ (a_n : ℕ → ℚ),
    (∀ k, a_n k = ball_return_ways m k) ∧
    (∀ k, a_n k ≥ 0) ∧
    (a_n 0 = 0) ∧
    (a_n 1 = 1) :=
  sorry


end NUMINAMATH_CALUDE_ball_return_theorem_l1543_154314


namespace NUMINAMATH_CALUDE_sum_in_base_nine_l1543_154370

/-- Represents a number in base 9 --/
def BaseNine : Type := List Nat

/-- Converts a base 9 number to its decimal representation --/
def to_decimal (n : BaseNine) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Adds two base 9 numbers --/
def add_base_nine (a b : BaseNine) : BaseNine :=
  sorry

/-- Theorem: The sum of 254₉, 367₉, and 142₉ is 774₉ in base 9 --/
theorem sum_in_base_nine :
  let a : BaseNine := [4, 5, 2]
  let b : BaseNine := [7, 6, 3]
  let c : BaseNine := [2, 4, 1]
  let result : BaseNine := [4, 7, 7]
  add_base_nine (add_base_nine a b) c = result :=
sorry

end NUMINAMATH_CALUDE_sum_in_base_nine_l1543_154370


namespace NUMINAMATH_CALUDE_minimum_time_to_fill_buckets_l1543_154335

def bucket_times : List Nat := [2, 4, 5, 7, 9]

def total_time (times : List Nat) : Nat :=
  (times.enum.map (fun (i, t) => t * (times.length - i))).sum

theorem minimum_time_to_fill_buckets :
  total_time bucket_times = 55 := by
  sorry

end NUMINAMATH_CALUDE_minimum_time_to_fill_buckets_l1543_154335


namespace NUMINAMATH_CALUDE_total_guppies_per_day_l1543_154373

/-- The number of guppies eaten by a moray eel per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish Jason has -/
def num_betta : ℕ := 5

/-- The number of guppies eaten by each betta fish per day -/
def betta_guppies : ℕ := 7

/-- The number of angelfish Jason has -/
def num_angelfish : ℕ := 3

/-- The number of guppies eaten by each angelfish per day -/
def angelfish_guppies : ℕ := 4

/-- The number of lionfish Jason has -/
def num_lionfish : ℕ := 2

/-- The number of guppies eaten by each lionfish per day -/
def lionfish_guppies : ℕ := 10

/-- Theorem stating the total number of guppies Jason needs to buy per day -/
theorem total_guppies_per_day :
  moray_eel_guppies +
  num_betta * betta_guppies +
  num_angelfish * angelfish_guppies +
  num_lionfish * lionfish_guppies = 87 := by
  sorry

end NUMINAMATH_CALUDE_total_guppies_per_day_l1543_154373


namespace NUMINAMATH_CALUDE_min_value_expression_l1543_154303

theorem min_value_expression (x y z : ℝ) 
  (hx : -1/2 < x ∧ x < 1/2) 
  (hy : -1/2 < y ∧ y < 1/2) 
  (hz : -1/2 < z ∧ z < 1/2) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + 
  (1 / ((1 + x) * (1 + y) * (1 + z))) + 
  1/2 ≥ 5/2 ∧ 
  (1 / ((1 - 0) * (1 - 0) * (1 - 0))) + 
  (1 / ((1 + 0) * (1 + 0) * (1 + 0))) + 
  1/2 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1543_154303


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l1543_154398

theorem systematic_sampling_interval 
  (total : ℕ) 
  (samples : ℕ) 
  (h1 : total = 231) 
  (h2 : samples = 22) :
  let adjusted_total := total - (total % samples)
  adjusted_total / samples = 10 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l1543_154398


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1543_154310

/-- Given two vectors a and b in ℝ², where a is parallel to b,
    prove that the magnitude of b is 2√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -2 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1543_154310


namespace NUMINAMATH_CALUDE_emmas_room_length_l1543_154380

/-- The length of Emma's room, given the width, tiled area, and fraction of room tiled. -/
theorem emmas_room_length (width : ℝ) (tiled_area : ℝ) (tiled_fraction : ℝ) :
  width = 12 →
  tiled_area = 40 →
  tiled_fraction = 1/6 →
  ∃ length : ℝ, length = 20 ∧ tiled_area = tiled_fraction * (width * length) := by
  sorry

end NUMINAMATH_CALUDE_emmas_room_length_l1543_154380


namespace NUMINAMATH_CALUDE_jacksons_vacuuming_time_l1543_154358

/-- Represents the problem of calculating Jackson's vacuuming time --/
theorem jacksons_vacuuming_time (vacuum_time : ℝ) : 
  vacuum_time = 2 :=
by
  have hourly_rate : ℝ := 5
  have dish_washing_time : ℝ := 0.5
  have bathroom_cleaning_time : ℝ := 3 * dish_washing_time
  have total_earnings : ℝ := 30
  have total_chore_time : ℝ := 2 * vacuum_time + dish_washing_time + bathroom_cleaning_time
  
  have h1 : hourly_rate * total_chore_time = total_earnings :=
    sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_jacksons_vacuuming_time_l1543_154358


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_113_l1543_154304

theorem first_nonzero_digit_of_1_over_113 : ∃ (n : ℕ) (r : ℚ), 
  (1 : ℚ) / 113 = n / 10 + r ∧ 
  0 < r ∧ 
  r < 1 / 10 ∧ 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_113_l1543_154304


namespace NUMINAMATH_CALUDE_equation_solutions_l1543_154318

theorem equation_solutions : 
  ∀ m n : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ 
  (m = 6 ∧ n = 3) ∨ (m = 9 ∧ n = 3) ∨ (m = 9 ∧ n = 5) ∨ (m = 54 ∧ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1543_154318


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1543_154392

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-2)*x^2 + a*x - 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a-2)*x + a

/-- Theorem: If f'(x) is even, then the tangent line at (1, f(1)) is 5x - y - 3 = 0 -/
theorem tangent_line_at_one (a : ℝ) :
  (∀ x, f' a x = f' a (-x)) →
  ∃ m b, ∀ x y, y = m*x + b ↔ y - f a 1 = (f' a 1) * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1543_154392


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l1543_154381

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 6 * y = 9

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -1/2 * x - 2

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, -3)

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through the given point
  perp_line point.1 point.2 ∧
  -- The two lines are perpendicular
  (∀ x₁ y₁ x₂ y₂ : ℝ, given_line x₁ y₁ → perp_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₁ - y₂) / (x₁ - x₂)) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l1543_154381


namespace NUMINAMATH_CALUDE_min_side_length_l1543_154308

/-- Given two triangles PQR and SQR sharing side QR, with PQ = 7 cm, PR = 15 cm, SR = 10 cm, and QS = 25 cm, the least possible integral length of QR is 16 cm. -/
theorem min_side_length (PQ PR SR QS : ℕ) (h1 : PQ = 7) (h2 : PR = 15) (h3 : SR = 10) (h4 : QS = 25) :
  (∃ QR : ℕ, QR > PR - PQ ∧ QR > QS - SR ∧ ∀ x : ℕ, (x > PR - PQ ∧ x > QS - SR) → x ≥ QR) →
  (∃ QR : ℕ, QR = 16 ∧ QR > PR - PQ ∧ QR > QS - SR ∧ ∀ x : ℕ, (x > PR - PQ ∧ x > QS - SR) → x ≥ QR) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l1543_154308


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1543_154302

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(2 : ℝ) / (x + 3)⌉ 
  else if x < -3 then ⌊(2 : ℝ) / (x + 3)⌋ 
  else 0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g : ¬ ∃ (x : ℝ), g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1543_154302


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l1543_154385

theorem weekend_rain_probability
  (p_saturday : ℝ)
  (p_sunday : ℝ)
  (p_sunday_given_saturday : ℝ)
  (h1 : p_saturday = 0.3)
  (h2 : p_sunday = 0.6)
  (h3 : p_sunday_given_saturday = 0.8) :
  1 - ((1 - p_saturday) * (1 - p_sunday) + p_saturday * (1 - p_sunday_given_saturday)) = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l1543_154385


namespace NUMINAMATH_CALUDE_vector_properties_l1543_154394

def a : ℝ × ℝ := (-4, 3)
def b : ℝ × ℝ := (7, 1)

theorem vector_properties :
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  angle = 3 * Real.pi / 4 ∧ proj = (-1/2) • b := by sorry

end NUMINAMATH_CALUDE_vector_properties_l1543_154394


namespace NUMINAMATH_CALUDE_system_solution_l1543_154333

theorem system_solution (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  x^2 + y * Real.sqrt (x * y) = 105 ∧
  y^2 + x * Real.sqrt (y * x) = 70 →
  x = 9 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1543_154333


namespace NUMINAMATH_CALUDE_division_problem_l1543_154307

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 23 →
  divisor = 5 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  quotient = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1543_154307


namespace NUMINAMATH_CALUDE_banana_arrangements_l1543_154322

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1543_154322
