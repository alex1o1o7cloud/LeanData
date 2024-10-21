import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_sum_l323_32351

/-- Represents a quadrilateral PQRS with vertices P(a,b), Q(b,c), R(-a,-b), and S(-b,-c) -/
structure Quadrilateral (a b c : ℤ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (a_gt_b : a > b)
  (b_gt_c : b > c)

/-- The area of the quadrilateral -/
noncomputable def area (q : Quadrilateral a b c) : ℝ :=
  Real.sqrt (((b - a : ℝ)^2 + (c - b : ℝ)^2) * ((b + a : ℝ)^2 + (c + b : ℝ)^2))

theorem quadrilateral_sum (a b c : ℤ) (q : Quadrilateral a b c) :
  area q = 20 → a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_sum_l323_32351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_theater_pricing_l323_32314

theorem movie_theater_pricing (child_price : ℚ) (child_count : ℕ) 
  (child_adult_diff : ℕ) (total_receipts : ℚ) :
  child_price = 4.5 →
  child_count = 48 →
  child_adult_diff = 20 →
  total_receipts = 405 →
  ∃ adult_price : ℚ, 
    adult_price = 6.75 ∧
    (child_price * child_count + 
     adult_price * (child_count - child_adult_diff) = total_receipts) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_theater_pricing_l323_32314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l323_32318

/-- Represents a single operation on a deck of cards -/
def Operation (n : ℕ) := { k : ℕ // k ≤ n }

/-- Represents a sequence of operations on a deck of cards -/
def OperationSequence (n : ℕ) := List (Operation n)

/-- Represents a permutation of n elements -/
def Permutation (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- The identity permutation -/
def idPerm (n : ℕ) : Permutation n := ⟨id, Function.bijective_id⟩

/-- The reverse permutation -/
noncomputable def revPerm (n : ℕ) : Permutation n :=
  ⟨λ i => ⟨n - 1 - i, by sorry⟩, by sorry⟩

/-- Applies a sequence of operations to a permutation -/
def applyOperations (n : ℕ) (p : Permutation n) (ops : OperationSequence n) : Permutation n :=
  sorry

theorem card_arrangement_theorem (n : ℕ) (h : n = 32) :
  (∀ (p : Permutation n), ∃ (ops : OperationSequence n),
    ops.length ≤ 5 ∧ applyOperations n (idPerm n) ops = p) ∧
  ¬∃ (ops : OperationSequence n),
    ops.length ≤ 4 ∧ applyOperations n (idPerm n) ops = revPerm n := by
  sorry

#check card_arrangement_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l323_32318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l323_32383

-- Define the function f(x) = 2cos(x) - 1
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x - 1

-- State the theorem about the maximum and minimum values of f
theorem f_max_min : (∀ x, f x ≤ 1) ∧ (∀ x, f x ≥ -3) ∧ (∃ x, f x = 1) ∧ (∃ x, f x = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l323_32383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_age_vasily_l323_32391

/-- The number of items to choose from -/
def n : ℕ := 64

/-- Fyodor's age -/
def F : ℕ := sorry

/-- Vasily's age -/
def V : ℕ := sorry

/-- The relationship between Vasily's and Fyodor's ages -/
axiom age_difference : V = F + 2

/-- Fyodor is at least 5 years old -/
axiom fyodor_min_age : F ≥ 5

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Fyodor always wins, meaning he has more choices than Vasily -/
axiom fyodor_wins : binomial n F > binomial n V

/-- The theorem stating the minimum age of Vasily -/
theorem min_age_vasily : V = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_age_vasily_l323_32391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l323_32312

/-- The ellipse representing curve C₁ -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line representing curve C₂ -/
def line (x y : ℝ) : Prop := x + y = 4

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The main theorem stating the minimum distance and the point where it occurs -/
theorem min_distance_ellipse_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧
    line x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), ellipse x₃ y₃ → line x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 ∧
    x₁ = 3/2 ∧ y₁ = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l323_32312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_paths_X_to_Z_l323_32369

/-- A point in the path -/
inductive Point
| X
| Y
| Z

/-- The number of paths between two points -/
def num_paths (a b : Point) : ℕ := sorry

theorem total_paths_X_to_Z :
  num_paths Point.X Point.Y = 3 →
  num_paths Point.Y Point.Z = 4 →
  num_paths Point.X Point.Z = 13 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_paths_X_to_Z_l323_32369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_time_is_five_l323_32352

/-- Represents the time it takes Mark to find parking each day -/
def parking_time : ℕ := 5

/-- Represents the number of workdays in a week -/
def workdays : ℕ := 5

/-- Represents the time it takes to walk into the courthouse -/
def walk_time : ℕ := 3

/-- Represents the time it takes to get through the metal detector on busy days -/
def busy_detector_time : ℕ := 30

/-- Represents the time it takes to get through the metal detector on less crowded days -/
def less_crowded_detector_time : ℕ := 10

/-- Represents the number of busy days in a week -/
def busy_days : ℕ := 2

/-- Represents the number of less crowded days in a week -/
def less_crowded_days : ℕ := 3

/-- Represents the total time spent on all activities in a week -/
def total_time : ℕ := 130

theorem parking_time_is_five : 
  parking_time = 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_time_is_five_l323_32352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l323_32324

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

-- Define the domain of f
def domain (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

-- Theorem statement
theorem interval_of_increase :
  ∀ x y : ℝ, domain x → domain y → x < y → x < -1 → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l323_32324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l323_32323

/-- Represents the transformation of the cosine function -/
noncomputable def transform_cosine (x : ℝ) : ℝ := 
  -Real.sin (x / 2 - Real.pi / 6)

/-- The original cosine function -/
noncomputable def original_cosine (x : ℝ) : ℝ := 
  Real.cos x

/-- The shifted cosine function -/
noncomputable def shifted_cosine (x : ℝ) : ℝ := 
  Real.cos (x + Real.pi / 3)

/-- The stretched cosine function -/
noncomputable def stretched_cosine (x : ℝ) : ℝ := 
  Real.cos (x / 2 + Real.pi / 3)

theorem cosine_transformation (x : ℝ) : 
  transform_cosine x = stretched_cosine x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l323_32323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_formation_l323_32358

def place_left (a b : ℕ) : ℕ := 100 * b + a

theorem three_digit_number_formation (a b : ℕ) : 
  10 ≤ a → a < 100 → 0 < b → b < 10 → 
  place_left a b = 100 * b + a := by
  intro h1 h2 h3 h4
  rfl

#eval place_left 42 7  -- Should output 742

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_formation_l323_32358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_exponential_at_one_l323_32350

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the inverse function f_inv
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem inverse_of_exponential_at_one :
  f_inv 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_exponential_at_one_l323_32350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_condition_l323_32301

/-- The ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- The line l with slope k passing through (-√2, 0) -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + Real.sqrt 2)

/-- Point on the ellipse C -/
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Vector from origin to a point -/
def vector_to_point (p : ℝ × ℝ) : ℝ × ℝ := p

/-- Vector addition -/
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

/-- Theorem: There exist points A, B, and P on the ellipse C such that OP = OA + OB
    if and only if the line AB has slope ±√2/2 -/
theorem ellipse_vector_sum_condition :
  ∃ (A B P : ℝ × ℝ), 
    point_on_ellipse A ∧ 
    point_on_ellipse B ∧ 
    point_on_ellipse P ∧ 
    vector_to_point P = vector_add (vector_to_point A) (vector_to_point B) ↔ 
    (∃ (k : ℝ), k = Real.sqrt 2 / 2 ∨ k = -(Real.sqrt 2 / 2)) ∧ 
    (∀ (x y : ℝ), line_l k x y → (point_on_ellipse (x, y) → x ≠ -(Real.sqrt 2))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_condition_l323_32301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l323_32371

-- Define f as a function from ℝ to Prop
def f : ℝ → Prop := λ x ↦ 0 ≤ x ∧ x ≤ 2

-- Define g as a function that transforms f
def g (f : ℝ → Prop) : ℝ → Prop := λ x ↦ f (2 * x)

-- Theorem stating the domain of g given the domain of f
theorem domain_of_g (f : ℝ → Prop) (h : ∀ x, f x ↔ (0 ≤ x ∧ x ≤ 2)) :
  ∀ x, g f x ↔ (0 ≤ x ∧ x ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l323_32371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_f_nonpositive_iff_a_leq_neg_inv_e_l323_32385

-- Define the function f(x) = ax + ln x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Theorem for part I
theorem max_value_when_a_is_one :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f 1 x ≤ f 1 (Real.exp 1) := by
  sorry

-- Theorem for part II
theorem f_nonpositive_iff_a_leq_neg_inv_e :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≤ 0) ↔ a ≤ -(1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_f_nonpositive_iff_a_leq_neg_inv_e_l323_32385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_x1995_x1996_l323_32344

def x : ℕ → ℕ
  | 0 => 19  -- Adding this case to cover Nat.zero
  | 1 => 19
  | 2 => 95
  | n + 3 => Nat.lcm (x (n + 2)) (x (n + 1)) + x (n + 1)

theorem gcd_x1995_x1996 : Nat.gcd (x 1995) (x 1996) = 19 := by
  sorry

#eval x 3  -- Adding this to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_x1995_x1996_l323_32344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_proof_l323_32393

/-- The original radius of a cylinder, given specific conditions -/
noncomputable def original_radius : ℝ := 4 + 4 * Real.sqrt 5

/-- The original height of the cylinder -/
def original_height : ℝ := 4

theorem cylinder_radius_proof :
  let r := original_radius
  let h := original_height
  π * (r + 8)^2 * h = π * r^2 * (3 * h) → r = 4 + 4 * Real.sqrt 5 :=
by
  intro h
  -- The proof steps would go here
  sorry

#eval original_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_proof_l323_32393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_third_vertex_l323_32377

/-- Type representing a point in a plane -/
structure Plane where
  x : ℝ
  y : ℝ

/-- Helper function to calculate the distance between two points in a plane -/
noncomputable def dist (P Q : Plane) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Predicate to check if a triangle is equilateral -/
def Equilateral (A B C : Plane) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

/-- Function to get the side length of a triangle -/
noncomputable def SideLength (A B C : Plane) : ℝ :=
  dist A B

/-- Given an equilateral triangle ABC with side length a < 5 and a point P such that AP = 2 and BP = 3,
    the maximum length of PC is (1/2) * √(29 - a² + 2√(3(26a² - a⁴ - 25))) -/
theorem max_distance_to_third_vertex (a : ℝ) (A B C P : Plane) :
  Equilateral A B C →
  SideLength A B C = a →
  a < 5 →
  dist A P = 2 →
  dist B P = 3 →
  ∃ (max_dist : ℝ), max_dist = (1/2) * Real.sqrt (29 - a^2 + 2 * Real.sqrt (3 * (26 * a^2 - a^4 - 25))) ∧
    ∀ (Q : Plane), dist A Q = 2 → dist B Q = 3 → dist C Q ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_third_vertex_l323_32377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_fourth_quadrant_l323_32320

theorem angle_fourth_quadrant (α : ℝ) (h1 : α ∈ Set.Icc (3 * π / 2) (2 * π)) 
  (h2 : Real.cos α = 1 / 3) : 
  Real.tan α = -2 * Real.sqrt 2 ∧ 
  (Real.sin α ^ 2 - Real.sqrt 2 * Real.sin α * Real.cos α) / (1 + Real.cos α ^ 2) = 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_fourth_quadrant_l323_32320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_renne_savings_period_l323_32354

/-- Calculates the number of months required to save for a vehicle purchase. -/
def months_to_save (monthly_earnings : ℚ) (saving_rate : ℚ) (vehicle_cost : ℚ) : ℚ :=
  vehicle_cost / (monthly_earnings * saving_rate)

/-- Proves that Renne needs 8 months to save for her dream vehicle. -/
theorem renne_savings_period :
  months_to_save 4000 (1/2) 16000 = 8 := by
  -- Unfold the definition of months_to_save
  unfold months_to_save
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_renne_savings_period_l323_32354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l323_32396

/-- Given a circle C with center (4, 0) and radius r, and a line y = kx tangent to C,
    if the surface area of the solid obtained by rotating C around the x-axis is 16π,
    then k = ± √3/3 -/
theorem tangent_line_to_circle (k r : ℝ) : 
  (∃ x y : ℝ, (x - 4)^2 + y^2 = r^2 ∧ y = k * x) →  -- Line is tangent to circle
  (4 * π * r^2 = 16 * π) →                         -- Surface area condition
  k = Real.sqrt 3 / 3 ∨ k = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l323_32396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_equation_l323_32355

/-- Represents a sequence of repeated digits -/
structure RepeatedDigit where
  digit : Nat
  count : Nat

/-- Converts a RepeatedDigit to its numerical value -/
def RepeatedDigit.toNat (rd : RepeatedDigit) : Nat :=
  rd.digit * (10 ^ rd.count - 1) / 9

/-- Adds two lists of RepeatedDigits without carry -/
def addWithoutCarry (l1 l2 : List RepeatedDigit) : List RepeatedDigit :=
  sorry

/-- The main theorem to prove -/
theorem repeated_digit_equation :
  ∃! (x y z : Nat),
    let lhs := addWithoutCarry
      [RepeatedDigit.mk 2 x, RepeatedDigit.mk 3 y, RepeatedDigit.mk 5 z]
      [RepeatedDigit.mk 3 z, RepeatedDigit.mk 5 x, RepeatedDigit.mk 2 y]
    let rhs := [RepeatedDigit.mk 5 3, RepeatedDigit.mk 7 2, RepeatedDigit.mk 8 3, RepeatedDigit.mk 5 1, RepeatedDigit.mk 7 3]
    lhs = rhs ∧ x + y + z = 12 :=
by
  sorry

#eval RepeatedDigit.toNat ⟨2, 3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_equation_l323_32355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l323_32373

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

/-- Helper function to calculate the area of a triangle given two sides and the included angle -/
noncomputable def area (t : AcuteTriangle) : ℝ := 
  1/2 * t.a * t.b * Real.sin t.C

/-- The theorem stating the properties and results for the given triangle -/
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b)
  (h2 : t.b = 1)
  (h3 : t.a = Real.sqrt 3) :
  t.A = Real.pi/3 ∧ area t = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l323_32373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_a_b_l323_32382

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) odd -/
def a : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) even -/
def b : ℕ := sorry

/-- The absolute difference between a and b is 104 -/
theorem abs_diff_a_b : |Int.ofNat a - Int.ofNat b| = 104 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_a_b_l323_32382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_increase_l323_32390

theorem batsman_average_increase (total_innings : ℕ) (last_inning_score : ℕ) (final_average : ℚ) : 
  total_innings = 11 → 
  last_inning_score = 100 → 
  final_average = 50 → 
  (final_average * total_innings - last_inning_score) / (total_innings - 1) + 5 = final_average := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_increase_l323_32390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divisible_by_sqrt3_d_l323_32399

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ := sorry

/-- Predicate for n being of the form 13p³ where p is a prime different from 13 -/
def is_valid_form (n : ℕ+) : Prop :=
  ∃ p : ℕ+, Nat.Prime p.val ∧ p ≠ 13 ∧ n = 13 * p^3

/-- The main theorem -/
theorem infinitely_many_n_divisible_by_sqrt3_d :
  ∃ f : ℕ → ℕ+, StrictMono f ∧
    (∀ k, is_valid_form (f k) ∧ (|Int.floor (Real.sqrt 3 * (d (f k) : ℝ))| : ℤ) ∣ (f k : ℤ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divisible_by_sqrt3_d_l323_32399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_to_cos_sin_identity_l323_32363

theorem tan_to_cos_sin_identity (α : Real) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_to_cos_sin_identity_l323_32363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_selection_theorem_l323_32317

theorem number_selection_theorem :
  ∀ (S : Finset ℕ),
    S.card = 55 ∧ (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 100) →
    (∃ a b, a ∈ S ∧ b ∈ S ∧ b - a = 9) ∧
    (∃ a b, a ∈ S ∧ b ∈ S ∧ b - a = 10) ∧
    (∃ a b, a ∈ S ∧ b ∈ S ∧ b - a = 12) ∧
    (∃ a b, a ∈ S ∧ b ∈ S ∧ b - a = 13) ∧
    (∃ T : Finset ℕ, T.card = 55 ∧ (∀ n, n ∈ T → 1 ≤ n ∧ n ≤ 100) ∧
      ∀ a b, a ∈ T → b ∈ T → b - a ≠ 11) :=
by
  sorry

#check number_selection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_selection_theorem_l323_32317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inscribed_in_cube_volume_ratio_l323_32389

/-- The ratio of the volume of a sphere inscribed in a cube to the volume of the cube,
    where the diameter of the sphere is half the side length of the cube. -/
noncomputable def sphere_cube_volume_ratio : ℝ := Real.pi / 48

/-- Theorem stating that the ratio of the volume of a sphere inscribed in a cube
    to the volume of the cube is π/48, where the diameter of the sphere is half
    the side length of the cube. -/
theorem sphere_inscribed_in_cube_volume_ratio :
  let s : ℝ := 1  -- Side length of the cube (arbitrary non-zero value)
  let r : ℝ := s / 4  -- Radius of the sphere
  let sphere_volume : ℝ := (4 / 3) * Real.pi * r^3
  let cube_volume : ℝ := s^3
  sphere_volume / cube_volume = sphere_cube_volume_ratio :=
by
  -- Placeholder for the actual proof
  sorry

#check sphere_inscribed_in_cube_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inscribed_in_cube_volume_ratio_l323_32389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_equals_cos_B_l323_32386

-- Define a right triangle ABC
structure RightTriangle where
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  angle_sum : B + C = 90  -- Sum of complementary angles in a right triangle

-- Theorem statement
theorem sin_C_equals_cos_B (abc : RightTriangle) (h : Real.cos abc.B = 3/5) : 
  Real.sin abc.C = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_equals_cos_B_l323_32386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l323_32397

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 3

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the set of x that satisfies the condition in part (I)
def solution_set_I : Set ℝ := Set.union (Set.Icc (-4) (-1)) (Set.Ioc 2 3)

-- Define the range of m for part (II)
def solution_set_II : Set ℝ := Set.Ioc 0 1

-- Theorem for part (I)
theorem part_I (x : ℝ) :
  (p x ∨ q x 2) ∧ ¬(p x ∧ q x 2) ↔ x ∈ solution_set_I :=
sorry

-- Theorem for part (II)
theorem part_II (m : ℝ) :
  (m > 0 ∧ ∀ x, q x m → p x ∧ ∃ y, p y ∧ ¬q y m) ↔ m ∈ solution_set_II :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l323_32397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_3_pow_89_l323_32339

/-- Sequence c_n defined recursively -/
def c : ℕ → ℕ
  | 0 => 3  -- Define for 0 to cover all natural numbers
  | 1 => 9
  | n + 2 => c (n + 1) * c n

/-- Theorem stating that the 10th term of sequence c_n equals 3^89 -/
theorem c_10_equals_3_pow_89 : c 10 = 3^89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_10_equals_3_pow_89_l323_32339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_20_tan_70_eq_1_l323_32379

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

theorem tan_20_tan_70_eq_1 :
  Real.tan (deg_to_rad 20) * Real.tan (deg_to_rad 70) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_20_tan_70_eq_1_l323_32379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_11_ending_7_l323_32336

/-- The count of positive multiples of 11 less than 2000 that end with the digit 7 -/
theorem count_multiples_of_11_ending_7 : 
  (Finset.filter (fun n => 0 < n ∧ n < 2000 ∧ n % 11 = 0 ∧ n % 10 = 7) (Finset.range 2000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_11_ending_7_l323_32336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l323_32334

theorem trig_expression_simplification :
  ∀ α : ℝ,
  (Real.sin (2 * Real.pi - α) * Real.cos (3 * Real.pi + α) * Real.cos ((3 * Real.pi) / 2 - α)) /
  (Real.sin (-Real.pi + α) * Real.sin (3 * Real.pi - α) * Real.cos (-α - Real.pi)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l323_32334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chord_theorem_l323_32313

/-- Arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y = 0

/-- Line equation -/
def line_eq (x y : ℝ) : Prop :=
  6*x - y + 1 = 0

/-- Theorem statement -/
theorem line_chord_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 4 = 12 →
  a 5 = 10 →
  (∃ x y : ℝ, circle_eq x y ∧ line_eq x y) →
  (∀ x y : ℝ, line_eq x y → (y - (a 3)*x) = (1 - 6*x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chord_theorem_l323_32313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l323_32392

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- State the theorem
theorem f_properties :
  (f (-4) = -2) ∧
  (f 3 = 6) ∧
  (f (f (-2)) = 8) ∧
  (∃! a : ℝ, a ≥ 2 ∧ f a = 10 ∧ a = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l323_32392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_down_is_39_l323_32322

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 90

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 900

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- Check if a number is the sum of two squares -/
def isSumOfTwoSquares (n : Nat) : Prop := sorry

/-- The cross-number puzzle -/
structure CrossNumber where
  across1 : ThreeDigitNumber
  across3 : ThreeDigitNumber
  across5 : TwoDigitNumber
  down1 : TwoDigitNumber
  down2 : TwoDigitNumber
  down4 : TwoDigitNumber

/-- The conditions of the cross-number puzzle -/
def validCrossNumber (c : CrossNumber) : Prop :=
  isPrime c.across1.val ∧
  isSumOfTwoSquares c.across1.val ∧
  c.across3.val = 2 * c.down2.val ∧
  ∃ p q : Nat, isPrime p ∧ isPrime q ∧ q = p + 4 ∧ c.down1.val = p * q ∧
  c.down4.val = (60 * c.across5.val) / 100

/-- The main theorem: proving that the answer to 2 DOWN is 39 -/
theorem two_down_is_39 : 
  ∀ c : CrossNumber, validCrossNumber c → c.down2.val = 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_down_is_39_l323_32322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_unit_circle_l323_32343

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Predicate to check if three points are within a circle of radius 1 -/
def within_unit_circle (p q r : Point) : Prop :=
  ∃ (center : Point), (distance center p ≤ 1) ∧ (distance center q ≤ 1) ∧ (distance center r ≤ 1)

/-- The main theorem -/
theorem points_in_unit_circle (n : ℕ) (points : Fin n → Point) 
  (h : ∀ (i j k : Fin n), within_unit_circle (points i) (points j) (points k)) :
  ∃ (center : Point), ∀ (i : Fin n), distance center (points i) ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_unit_circle_l323_32343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_value_no_intersection_l323_32378

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x^2 + x/4 + y/5 = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the slope of a line
noncomputable def line_slope (f : ℝ → ℝ) : ℝ := sorry

-- Theorem: The slope of the line is -5/4
theorem line_slope_value :
  ∃ f : ℝ → ℝ, (∀ x y, line_equation x y ↔ y = f x) ∧ line_slope f = -5/4 := by sorry

-- Theorem: The line does not intersect the circle
theorem no_intersection :
  ¬∃ x y, line_equation x y ∧ circle_equation x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_value_no_intersection_l323_32378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_at_point_six_l323_32364

noncomputable def x (p : ℝ) (q : ℝ) : ℝ :=
  if p ≤ 1/2 then 1 else q/p

theorem x_value_at_point_six (q : ℝ) (h : q = 0.4) :
  x 0.6 q = 2/3 := by
  rw [x]
  simp [h]
  norm_num

#eval (2 : ℚ) / 3  -- This line is added to show the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_at_point_six_l323_32364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circle_area_ratio_l323_32302

/-- Given an isosceles triangle with base angle α, the ratio of its area to the area of its circumscribed circle is (2 * sin(2α) * sin²(α)) / π. -/
theorem isosceles_triangle_circle_area_ratio (α : ℝ) :
  let triangle_area := (1/2) * (2 * Real.sin α)^2 * Real.sin (π - 2*α)
  let circle_area := π * ((Real.sin α) / (2 * Real.sin α))^2
  triangle_area / circle_area = (2 * Real.sin (2*α) * Real.sin α^2) / π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circle_area_ratio_l323_32302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_problem_l323_32310

theorem oranges_problem (total : ℕ) (ripe partially_ripe unripe spoiled : ℕ) 
  (ripe_eaten partially_ripe_eaten unripe_eaten spoiled_eaten : ℕ) :
  total = 480 →
  ripe = (3 * total) / 7 →
  partially_ripe = (2 * total) / 5 →
  unripe = total - ripe - partially_ripe - spoiled →
  spoiled = 10 →
  ripe_eaten = (7 * ripe) / 13 →
  partially_ripe_eaten = (5 * partially_ripe) / 9 →
  unripe_eaten = (3 * unripe) / 11 →
  spoiled_eaten = spoiled / 2 →
  total - ripe_eaten - partially_ripe_eaten - unripe_eaten - spoiled_eaten = 240 := by
  sorry

#check oranges_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_problem_l323_32310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l323_32309

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties (ω φ α : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 ≤ φ ∧ φ < π/2) 
  (h_sym : ∀ x, f ω φ (2*π/3 - x) = f ω φ (2*π/3 + x))
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_α : π/6 < α ∧ α < 2*π/3)
  (h_f_α : f ω φ (α/2) = Real.sqrt 3 / 4) :
  ω = 2 ∧ φ = -π/6 ∧ Real.cos (α + π/3) = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l323_32309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l323_32305

noncomputable section

def a (n : ℕ+) : ℝ := (-1/4)^(n : ℕ)

def S (n : ℕ+) : ℝ := -1/5 * (1 - (-1/4)^(n : ℕ))

def b (n : ℕ+) : ℝ := -1 - Real.log (abs (a n)) / Real.log 2

def T (n : ℕ+) : ℝ := (n : ℝ)^2

def c (n : ℕ+) : ℝ := b (n + 1) / (T n * T (n + 1))

def A (n : ℕ+) : ℝ := 1 - 1 / ((n + 1 : ℝ)^2)

theorem sequence_properties :
  (∀ n : ℕ+, a n = 5 * S n + 1) ∧
  (∀ n : ℕ+, a n = (-1/4)^(n : ℕ)) ∧
  (∀ n : ℕ+, A n = 1 - 1 / ((n + 1 : ℝ)^2)) ∧
  (∀ m k : ℕ+, ∃ n : ℕ+, |S m - S k| ≤ 32 * a n ↔ n = 2 ∨ n = 4) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l323_32305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_ratio_condition_l323_32304

theorem min_m_for_ratio_condition (n r : ℕ) (hn : n > 0) (hr : r > 0) :
  ∀ m : ℕ, m ≥ (n + 1) * r ↔
    ∀ (A : Fin r → Set ℕ), 
      (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) →
      (⋃ i : Fin r, A i) = Finset.range m →
      ∃ i : Fin r, ∃ a b : ℕ, a ∈ A i ∧ b ∈ A i ∧ a > b ∧ 1 < (a : ℚ) / b ∧ (a : ℚ) / b ≤ 1 + 1 / n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_ratio_condition_l323_32304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l323_32370

theorem negation_of_conditional (α β : ℝ) : 
  (¬(α = β → Real.sin α = Real.sin β)) ↔ (Real.sin α ≠ Real.sin β → α ≠ β) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l323_32370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_inverse_l323_32387

theorem min_value_cubic_inverse (y : ℝ) (hy : y > 0) : 
  9 * y^3 + 4 * y^(-6 : ℝ) ≥ 13 ∧ 
  (9 * y^3 + 4 * y^(-6 : ℝ) = 13 ↔ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_inverse_l323_32387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_with_2022_blocks_l323_32337

def has_2022_block (n : ℕ) : Prop :=
  ∃ s₁ s₂ : String, toString n = s₁ ++ "2022" ++ s₂

theorem exists_k_with_2022_blocks (n : ℕ+) :
  ∃ k : ℕ+, ∀ i : ℕ+, i ≤ n → has_2022_block (k.val ^ i.val) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_with_2022_blocks_l323_32337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_play_win_probability_l323_32315

def probability_best_play_wins (n m : ℕ) : ℚ :=
  let total_moms := 2 * n
  let jury_size := 2 * m
  1 / (Nat.choose total_moms n * Nat.choose total_moms jury_size) *
    (Finset.sum (Finset.range (jury_size + 1)) (λ q =>
      Nat.choose n q * Nat.choose n (jury_size - q) *
      (Finset.sum (Finset.range (min q (m - 1) + 1)) (λ t =>
        Nat.choose q t * Nat.choose (total_moms - q) (n - t)))))

theorem best_play_win_probability (n m : ℕ) (h : 2 * m ≤ n) :
  probability_best_play_wins n m =
    (1 : ℚ) / (Nat.choose (2*n) n * Nat.choose (2*n) (2*m)) *
    (Finset.sum (Finset.range (2*m + 1)) (λ q =>
      Nat.choose n q * Nat.choose n (2*m - q) *
      (Finset.sum (Finset.range (min q (m - 1) + 1)) (λ t =>
        Nat.choose q t * Nat.choose (2*n - q) (n - t))))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_play_win_probability_l323_32315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_340m_l323_32395

-- Define the parabola Γ
noncomputable def Γ (x : ℝ) : ℝ := (1/36) * x^2

-- Define the points
def A : ℝ × ℝ := (-36, 2)
def D : ℝ × ℝ := (0, -2)

-- Define the condition for point C
def C_condition (x : ℝ) : Prop := x > 6 * Real.sqrt 2

-- Define the intersection points M and N
noncomputable def M (x_C : ℝ) : ℝ × ℝ := 
  let x_M := (-2 * x_C - 72) / (x_C + 2)
  (x_M, Γ x_M)

noncomputable def N (x_C : ℝ) : ℝ × ℝ := 
  let x_N := 72 / x_C
  (x_N, Γ x_N)

-- Define point B
def B : ℝ × ℝ := (-2, 2)

-- State the theorem
theorem distance_AB_is_340m (x_C : ℝ) (h_C : C_condition x_C) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_340m_l323_32395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_gravitational_force_point_l323_32327

/-- The distance from the Sun to the point of equal gravitational attraction -/
noncomputable def equal_gravitational_point (d M m : ℝ) : ℝ :=
  d / (1 + Real.sqrt (m / M))

/-- Theorem stating the point of equal gravitational attraction -/
theorem equal_gravitational_force_point
  (d M m : ℝ)
  (h_d : d > 0)
  (h_M : M > 0)
  (h_m : m > 0) :
  let x := equal_gravitational_point d M m
  let G : ℝ := Classical.choose (exists_pos_real)  -- Gravitational constant
  G * (M / x^2) = G * (m / (d - x)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_gravitational_force_point_l323_32327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_neg_i_l323_32341

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := 2 / (1 + i)^2

-- Theorem statement
theorem z_equals_neg_i : z = -i := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equals_neg_i_l323_32341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_27_3_l323_32319

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_27_3_l323_32319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_tetrahedra_share_point_l323_32353

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents the area of a face of the tetrahedron -/
noncomputable def faceArea (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- Represents the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Represents the radius of the inscribed sphere in a tetrahedron -/
noncomputable def inradius (t : Tetrahedron) : ℝ := sorry

/-- Represents a smaller tetrahedron cut off by a plane parallel to a face -/
structure SmallerTetrahedron (t : Tetrahedron) where
  vertex : Point
  opposite_face : Fin 4

/-- Predicate to check if a point is inside a smaller tetrahedron -/
def isInside (p : Point) (st : SmallerTetrahedron t) : Prop := sorry

/-- Theorem: If in a tetrahedron ABCD with an inscribed sphere, 
    the faces BCD and ACD have the smallest areas, then the smaller tetrahedra 
    cut off by planes parallel to the faces can share a common internal point -/
theorem smaller_tetrahedra_share_point (t : Tetrahedron) 
  (h1 : faceArea t 0 ≤ faceArea t 1 ∧ faceArea t 0 ≤ faceArea t 2 ∧ faceArea t 0 ≤ faceArea t 3)
  (h2 : faceArea t 1 ≤ faceArea t 2 ∧ faceArea t 1 ≤ faceArea t 3) :
  ∃ (p : Point), ∀ (st : SmallerTetrahedron t), isInside p st := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_tetrahedra_share_point_l323_32353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l323_32333

theorem triangle_classification (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.cos A = b * Real.cos B →
  a / Real.sin A = b / Real.sin B →
  (A = B) ∨ (A + B = Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l323_32333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_range_l323_32326

/-- A function representing x + 4/x -/
noncomputable def f (x : ℝ) : ℝ := x + 4/x

/-- The theorem statement -/
theorem quadratic_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x ≤ 3 ∧ 0 < y ∧ y ≤ 3 ∧
   x^2 - (m+1)*x + 4 = 0 ∧ y^2 - (m+1)*y + 4 = 0) →
  3 < m ∧ m ≤ 10/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_range_l323_32326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_selling_price_l323_32325

/-- Calculates the selling price of a car given its purchase price, repair costs, and profit percentage. -/
def calculate_selling_price (purchase_price repair_costs : ℕ) (profit_percent : ℚ) : ℕ :=
  let total_cost := purchase_price + repair_costs
  let profit_decimal := profit_percent / 100
  let profit_amount := (total_cost : ℚ) * profit_decimal
  (((total_cost : ℚ) + profit_amount).floor : ℤ).toNat

/-- Theorem stating that for the given conditions, the selling price of the car is 61900. -/
theorem car_selling_price :
  calculate_selling_price 42000 13000 (12545454545454545 / 1000000000000000) = 61900 :=
by sorry

#eval calculate_selling_price 42000 13000 (12545454545454545 / 1000000000000000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_selling_price_l323_32325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_sum_l323_32365

-- Define the ellipse
structure Ellipse where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  sum_distances : ℝ

-- Define the parameters of the ellipse
noncomputable def ellipse_parameters (e : Ellipse) : ℝ × ℝ × ℝ × ℝ :=
  let c := (e.f2.1 - e.f1.1) / 2
  let a := e.sum_distances / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (e.f1.1 + e.f2.1) / 2
  let k := (e.f1.2 + e.f2.2) / 2
  (h, k, a, b)

-- Theorem statement
theorem ellipse_parameter_sum :
  let e := Ellipse.mk (0, 0) (6, 0) 10
  let (h, k, a, b) := ellipse_parameters e
  h + k + a + b = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_sum_l323_32365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_arithmetic_mean_l323_32376

theorem smallest_arithmetic_mean (a₁ a₂ a₃ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0)
  (h_eq : 2*a₁ + 3*a₂ + a₃ = 1) :
  let f := fun x y => (1/x + 1/y) / 2
  ∃ (m : ℝ), m = f (a₁ + a₂) (a₂ + a₃) ∧ 
    (∀ x y, x > 0 → y > 0 → 2*x + y = 1 → f x y ≥ m) ∧
    m = (3 + 2*Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_arithmetic_mean_l323_32376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l323_32346

/-- A trip with specific speed and distance conditions -/
structure Trip where
  first_segment_distance : ℝ
  first_segment_speed : ℝ
  last_segment_distance : ℝ
  last_segment_speed : ℝ
  average_speed : ℝ

/-- The total distance of the trip -/
noncomputable def total_distance (t : Trip) : ℝ :=
  t.first_segment_distance + t.last_segment_distance

/-- The total time of the trip -/
noncomputable def total_time (t : Trip) : ℝ :=
  t.first_segment_distance / t.first_segment_speed +
  t.last_segment_distance / t.last_segment_speed

/-- Theorem stating the total distance of the trip given the conditions -/
theorem trip_distance (t : Trip) 
  (h1 : t.first_segment_distance = 30)
  (h2 : t.first_segment_speed = 60)
  (h3 : t.last_segment_distance = 70)
  (h4 : t.last_segment_speed = 35)
  (h5 : t.average_speed = 40)
  (h6 : t.average_speed = total_distance t / total_time t) :
  total_distance t = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_distance_l323_32346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_f_l323_32388

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * Real.sqrt x + x

-- State the theorem
theorem evaluate_f : 3 * f 3 + f 9 = 135 + 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_f_l323_32388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_implies_a_eq_neg_four_l323_32303

noncomputable section

/-- A function f is strictly decreasing on an interval [a, b] if for any x, y in [a, b] with x < y, f(x) > f(y) -/
def StrictlyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

/-- The function f(x) = (1/3)x³ - (3/2)x² + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * x^2 + a * x + 4

/-- If f(x) = (1/3)x³ - (3/2)x² + ax + 4 is strictly decreasing on [-1, 4], then a = -4 -/
theorem f_strictly_decreasing_implies_a_eq_neg_four :
  (∃ a : ℝ, StrictlyDecreasing (f a) (-1) 4) →
  (∃ a : ℝ, StrictlyDecreasing (f a) (-1) 4 ∧ a = -4) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_implies_a_eq_neg_four_l323_32303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_ticket_win_probability_l323_32332

/-- Represents the lottery game parameters -/
structure LotteryGame where
  ticketCost : ℚ
  prizeMoney : ℚ
  winProbability : ℚ

/-- Represents the bus ticket scenario -/
structure BusTicketScenario where
  initialMoney : ℚ
  busTicketCost : ℚ
  game : LotteryGame

/-- Calculates the probability of winning enough money to buy the bus ticket -/
noncomputable def winProbability (scenario : BusTicketScenario) : ℚ :=
  let p := scenario.game.winProbability
  let q := 1 - p
  ((3 * p^2 - 2 * p^3) / (1 - 2*p + 4*p^2 - 2*p^3))

/-- Theorem: The probability of winning enough money to buy the bus ticket is approximately 0.033 -/
theorem bus_ticket_win_probability (scenario : BusTicketScenario) 
  (h1 : scenario.initialMoney = 20)
  (h2 : scenario.busTicketCost = 45)
  (h3 : scenario.game.ticketCost = 10)
  (h4 : scenario.game.prizeMoney = 30)
  (h5 : scenario.game.winProbability = 1/10) :
  ∃ ε > 0, |winProbability scenario - 33/1000| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_ticket_win_probability_l323_32332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perfect_squares_in_products_l323_32367

theorem max_perfect_squares_in_products (a b : ℕ) (h : a ≠ b) : 
  let products := [(a, a + 2), (a, b), (a, b + 2), (a + 2, b), (a + 2, b + 2), (b, b + 2)]
  (∃ (p q : ℕ × ℕ), p ∈ products ∧ q ∈ products ∧ p ≠ q ∧ 
    ∃ (x y : ℕ), p.1 * p.2 = x^2 ∧ q.1 * q.2 = y^2) ∧
  (∀ (p q r : ℕ × ℕ), p ∈ products ∧ q ∈ products ∧ r ∈ products ∧ 
    p ≠ q ∧ q ≠ r ∧ p ≠ r →
    ¬∃ (x y z : ℕ), p.1 * p.2 = x^2 ∧ q.1 * q.2 = y^2 ∧ r.1 * r.2 = z^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perfect_squares_in_products_l323_32367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l323_32306

-- Define the function f(x) = x - 1/x
noncomputable def f (x : ℝ) : ℝ := x - 1/x

-- Theorem statement
theorem f_properties :
  -- 1. f(2) = 3/2
  f 2 = 3/2 ∧
  -- 2. f(x) is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) ∧
  -- 3. The minimum value of f(x) on [2,5] is 3/2
  (∀ x : ℝ, 2 ≤ x → x ≤ 5 → 3/2 ≤ f x) ∧
  f 2 = 3/2 ∧
  -- 4. The maximum value of f(x) on [2,5] is 24/5
  (∀ x : ℝ, 2 ≤ x → x ≤ 5 → f x ≤ 24/5) ∧
  f 5 = 24/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l323_32306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cutting_tool_problem_l323_32307

/-- A machine-shop cutting tool problem -/
theorem cutting_tool_problem (O A B C : ℝ × ℝ) : 
  (∀ P : ℝ × ℝ, dist O P = 8 → P.1^2 + P.2^2 = 64) →  -- Circle equation
  dist A B = 8 →                                      -- Length of AB
  dist B C = 4 →                                      -- Length of BC
  (B.1 - A.1) * (C.2 - B.2) = (B.2 - A.2) * (C.1 - B.1) →  -- Right angle ABC
  dist O B = 8 →                                      -- B is on the circle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 4 :=                -- Distance from B to O squared
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cutting_tool_problem_l323_32307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_max_value_l323_32356

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

theorem f_monotonicity_and_max_value :
  (∀ x1 x2 : ℝ, -3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ -1 → f x1 < f x2) ∧
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ -1 → f x ≤ -2) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ -1 ∧ f x = -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_max_value_l323_32356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_solution_set_l323_32362

theorem sine_inequality_solution_set :
  let S := {x : ℝ | -Real.sqrt 2 / 2 < Real.sin x ∧ Real.sin x ≤ 1 / 2}
  S = {x : ℝ | ∃ k : ℤ, (-Real.pi/4 + 2*Real.pi*k < x ∧ x ≤ Real.pi/6 + 2*Real.pi*k) ∨ 
                        (5*Real.pi/6 + 2*Real.pi*k ≤ x ∧ x < 5*Real.pi/4 + 2*Real.pi*k)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_solution_set_l323_32362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_eight_l323_32331

/-- The area of a polygon formed by a 3x2 rectangle and a 2x2 right-angled triangle -/
noncomputable def polygon_area : ℝ :=
  let rectangle_width : ℝ := 3
  let rectangle_height : ℝ := 2
  let triangle_base : ℝ := 2
  let triangle_height : ℝ := 2
  let rectangle_area := rectangle_width * rectangle_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  rectangle_area + triangle_area

/-- Theorem stating that the polygon area is 8 square units -/
theorem polygon_area_is_eight : polygon_area = 8 := by
  -- Unfold the definition of polygon_area
  unfold polygon_area
  -- Simplify the arithmetic expressions
  simp [mul_add, add_mul]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_is_eight_l323_32331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l323_32374

/-- The volume of a regular triangular pyramid -/
noncomputable def regularTriangularPyramidVolume (R : ℝ) (α : ℝ) : ℝ :=
  (1/4) * R^3 * Real.sqrt 3 * (Real.sin (2*α))^3 * Real.tan α

/-- Theorem: The volume of a regular triangular pyramid circumscribed by a sphere -/
theorem regular_triangular_pyramid_volume
  (R : ℝ) (α : ℝ)
  (h_R_pos : R > 0)
  (h_α_pos : α > 0)
  (h_α_lt_pi_2 : α < π/2) :
  regularTriangularPyramidVolume R α =
    (1/4) * R^3 * Real.sqrt 3 * (Real.sin (2*α))^3 * Real.tan α :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l323_32374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_heights_interior_point_inequality_l323_32375

/-- Helper function to represent a line given two points -/
def line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Helper function to calculate the distance from a point to a line -/
def distance_to_line (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Helper function to check if a point is inside a triangle -/
def is_interior_point (P A B C : ℝ × ℝ) : Prop := sorry

/-- Triangle heights and interior point distances inequality theorem -/
theorem triangle_heights_interior_point_inequality 
  (m_a m_b m_c p_a p_b p_c : ℝ) 
  (h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ p_a > 0 ∧ p_b > 0 ∧ p_c > 0) 
  (h_triangle : ∃ (A B C : ℝ × ℝ), 
    m_a = distance_to_line A (line B C) ∧ 
    m_b = distance_to_line B (line A C) ∧ 
    m_c = distance_to_line C (line A B)) 
  (h_interior_point : ∃ (P : ℝ × ℝ), 
    p_a = distance_to_line P (line B C) ∧ 
    p_b = distance_to_line P (line A C) ∧ 
    p_c = distance_to_line P (line A B) ∧ 
    is_interior_point P A B C) : 
  m_a * m_b * m_c ≥ 27 * p_a * p_b * p_c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_heights_interior_point_inequality_l323_32375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l323_32359

/-- There exists a real number x such that 5^x * 12^0.25 * 60^0.75 = 300, 
    and this x is close to 1.886 -/
theorem power_equation_solution : 
  ∃ x : ℝ, (5 : ℝ)^x * (12 : ℝ)^(1/4) * (60 : ℝ)^(3/4) = 300 ∧ 
  (abs (x - 1.886) < 0.001) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l323_32359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_geometric_numbers_l323_32340

/-- A function that checks if a 3-digit number is geometric -/
def is_geometric (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a r : ℕ), r > 0 ∧
    (let d1 := n / 100
     let d2 := (n / 10) % 10
     let d3 := n % 10
     d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
     d1 = a ∧ d2 = a * r ∧ d3 = a * r * r)

/-- The smallest geometric 3-digit number -/
def smallest_geometric : ℕ := 124

/-- The largest geometric 3-digit number -/
def largest_geometric : ℕ := 972

theorem sum_of_extreme_geometric_numbers :
  is_geometric smallest_geometric ∧
  is_geometric largest_geometric ∧
  (∀ n : ℕ, is_geometric n → smallest_geometric ≤ n) ∧
  (∀ n : ℕ, is_geometric n → n ≤ largest_geometric) ∧
  smallest_geometric + largest_geometric = 1096 := by
  sorry

#eval smallest_geometric + largest_geometric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_geometric_numbers_l323_32340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_is_semicircle_l323_32329

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Define the range of θ
def θ_range : Set ℝ := {θ | -Real.pi/2 ≤ θ ∧ θ ≤ Real.pi/2}

-- Theorem statement
theorem curve_C_is_semicircle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ ∈ θ_range, 
      let (x, y) := C θ
      (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (∀ θ ∈ θ_range,
      let (x, _) := C θ
      0 ≤ x ∧ x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_is_semicircle_l323_32329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_max_product_l323_32330

/-- Given two positive real numbers m and n, and two vectors a and b in R^3 
    such that a is perpendicular to b, prove that the maximum value of mn is 3/2 -/
theorem perpendicular_vectors_max_product (m n : ℝ) (a b : ℝ × ℝ × ℝ) 
  (hm : m > 0) (hn : n > 0) 
  (ha : a = (m, 4, -3)) (hb : b = (1, n, 2)) 
  (hperp : m * 1 + 4 * n + (-3) * 2 = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * 1 + 4 * y + (-3) * 2 = 0 → x * y ≤ m * n) ∧ 
  m * n = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_max_product_l323_32330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_vowel_l323_32338

def set1 : Finset Char := {'a', 'b', 'e'}
def set2 : Finset Char := {'k', 'i', 'm', 'n', 'o', 'p'}
def set3 : Finset Char := {'r', 's', 't', 'u', 'v', 'w'}

def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u']

noncomputable def probAtLeastOneVowel : ℚ :=
  1 - (1 - (set1.filter (fun c => isVowel c)).card / set1.card) *
      (1 - (set2.filter (fun c => isVowel c)).card / set2.card) *
      (1 - (set3.filter (fun c => isVowel c)).card / set3.card)

theorem prob_at_least_one_vowel :
  probAtLeastOneVowel = 22 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_vowel_l323_32338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_victories_exist_l323_32357

/-- Represents a volleyball tournament -/
structure Tournament where
  teams : Finset Nat
  victories : Nat → Nat
  played_all : ∀ i j, i ∈ teams → j ∈ teams → i ≠ j → (victories i + victories j = teams.card - 1)
  cycle_exists : ∃ a b c, a ∈ teams ∧ b ∈ teams ∧ c ∈ teams ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (victories a > victories b) ∧ (victories b > victories c) ∧ (victories c > victories a)

/-- There exist at least two teams with the same number of victories -/
theorem same_victories_exist (t : Tournament) :
  ∃ i j, i ∈ t.teams ∧ j ∈ t.teams ∧ i ≠ j ∧ t.victories i = t.victories j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_victories_exist_l323_32357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l323_32349

theorem shaded_areas_equality (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 2) :
  (∃ r : Real, r > 0 ∧ 
    ((r^2 * Real.tan θ) / 2 - (θ * r^2) / 2 = (θ * r^2) / 2)) ↔ 
  Real.tan θ = 2 * θ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l323_32349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l323_32335

/-- Calculates the approximate distance in miles given map scale and measured distance -/
noncomputable def calculate_distance (map_inches : ℝ) (map_miles : ℝ) (measured_cm : ℝ) : ℝ :=
  let inches_per_cm := 1 / 2.54
  let measured_inches := measured_cm * inches_per_cm
  let miles_per_inch := map_miles / map_inches
  measured_inches * miles_per_inch

/-- Theorem stating that the calculated distance is approximately 976.32 miles -/
theorem distance_calculation :
  let map_inches : ℝ := 2.5
  let map_miles : ℝ := 40
  let measured_cm : ℝ := 155
  |calculate_distance map_inches map_miles measured_cm - 976.32| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l323_32335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l323_32342

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x^2 + (2 + Real.log a) * x + Real.log b

-- State the theorem
theorem function_properties (a b : ℝ) 
  (h1 : f a b (-1) = -2)
  (h2 : ∀ x : ℝ, f a b x ≥ 2 * x) :
  a = 100 ∧ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l323_32342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_when_lateral_is_half_total_l323_32328

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the lateral surface area of a cylinder -/
noncomputable def lateralSurfaceArea (c : Cylinder) : ℝ := 2 * Real.pi * c.radius * c.height

/-- Calculates the total surface area of a cylinder -/
noncomputable def totalSurfaceArea (c : Cylinder) : ℝ := 
  2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius^2

/-- Theorem: For a cylinder with base radius 3, if the lateral surface area is 1/2 of 
    the total surface area, then the height of the cylinder is 3 -/
theorem cylinder_height_when_lateral_is_half_total (c : Cylinder) 
    (h_radius : c.radius = 3) 
    (h_lateral_half_total : lateralSurfaceArea c = (1/2) * totalSurfaceArea c) : 
    c.height = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_when_lateral_is_half_total_l323_32328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l323_32300

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 9 = 1

noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem hyperbola_foci_distance :
  ∀ (x y : ℝ), hyperbola_equation x y →
  2 * focal_distance 5 3 = 2 * Real.sqrt 34 :=
by
  sorry

#check hyperbola_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l323_32300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toadon_population_percentage_l323_32361

/-- Prove that Toadon's population is 60% of Gordonia's population -/
theorem toadon_population_percentage (total_population : ℕ) (gordonia_ratio : ℚ) (lake_bright_population : ℕ) :
  total_population = 80000 →
  gordonia_ratio = 1/2 →
  lake_bright_population = 16000 →
  (total_population - (gordonia_ratio * ↑total_population).floor - lake_bright_population) / 
  ((gordonia_ratio * ↑total_population).floor) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toadon_population_percentage_l323_32361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_n_nonzero_det_l323_32347

/-- Definition of the matrix A --/
def A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  Matrix.of (λ i j => (i.val^j.val + j.val^i.val) % 3)

/-- The theorem to be proved --/
theorem greatest_n_nonzero_det :
  ∀ k : ℕ, k > 5 → Matrix.det (A k) = 0 ∧
  Matrix.det (A 5) ≠ 0 ∧
  ∀ m : ℕ, m > 5 → Matrix.det (A m) = 0 := by
  sorry

#check greatest_n_nonzero_det

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_n_nonzero_det_l323_32347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_n_and_range_l323_32380

noncomputable def m : ℝ × ℝ := (1, 1)
noncomputable def q : ℝ × ℝ := (1, 0)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

noncomputable def n : ℝ × ℝ := (0, -1)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

noncomputable def p : ℝ × ℝ := (Real.cos A, 2 * (Real.cos (C/2))^2)

theorem vector_n_and_range :
  angle_between m n = 3 * Real.pi / 4 ∧
  dot_product m n = -1 ∧
  vector_length (q.1 + n.1, q.2 + n.2) = vector_length (q.1 - n.1, q.2 - n.2) ∧
  A + C = 2 * B ∧
  n = (0, -1) ∧
  Real.sqrt 2 / 2 ≤ vector_length (n.1 + p.1, n.2 + p.2) ∧
  vector_length (n.1 + p.1, n.2 + p.2) < Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_n_and_range_l323_32380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_base_solutions_l323_32345

def is_valid_base_pair (b₁ b₂ : ℕ) : Prop :=
  b₁ % 5 = 0 ∧ b₂ % 5 = 0 ∧ (b₁ + b₂) * 40 / 2 = 1800 ∧ b₁ ≤ b₂

-- We need to make this function decidable
def is_valid_base_pair_dec (b₁ b₂ : ℕ) : Bool :=
  b₁ % 5 = 0 && b₂ % 5 = 0 && (b₁ + b₂) * 40 / 2 = 1800 && b₁ ≤ b₂

theorem trapezoid_base_solutions : 
  ∃! n : ℕ, n = (Finset.filter (λ (p : ℕ × ℕ) => is_valid_base_pair_dec p.1 p.2) 
    (Finset.product (Finset.range 1801) (Finset.range 1801))).card ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_base_solutions_l323_32345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_ratio_l323_32381

/-- Helper function to represent the area of the quadrilateral formed by joining midpoints -/
noncomputable def area_of_quadrilateral_formed_by_midpoints (Q : ℝ) : ℝ := 
  (1/4) * Q

/-- Given a quadrilateral with area Q, joining its midpoints forms a quadrilateral with area N = Q/4 -/
theorem midpoint_quadrilateral_area_ratio (Q N : ℝ) 
  (h : Q > 0) -- Ensure Q is positive (implicitly assuming N is also positive)
  (h_midpoint : N = area_of_quadrilateral_formed_by_midpoints Q) : N = (1/4) * Q := by
  -- Unfold the definition of area_of_quadrilateral_formed_by_midpoints
  unfold area_of_quadrilateral_formed_by_midpoints at h_midpoint
  -- The proof follows directly from the definition
  exact h_midpoint


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_ratio_l323_32381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l323_32311

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the right vertex of the hyperbola
def right_vertex : ℝ × ℝ := (2, 0)

-- Define the asymptotes of the hyperbola
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Define a circle
def circle_eq (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem statement
theorem circle_equation :
  ∃ (r : ℝ), 
    (∀ x y : ℝ, asymptote x y → 
      (circle_eq right_vertex r x y → 
        ∃ t : ℝ, x = t ∧ asymptote t y)) →
    r^2 = 4/5 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l323_32311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l323_32360

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

-- Define the Cartesian equation of curve C
def C_cartesian (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define perpendicularity of two vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem curve_C_properties :
  -- Part 1: Prove that the Cartesian equation of C is correct
  (∀ x y : ℝ, (∃ ρ θ : ℝ, C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ C_cartesian x y) ∧
  -- Part 2: Prove the sum of reciprocal squared distances for perpendicular points
  (∀ ρ₁ θ₁ ρ₂ θ₂ : ℝ, 
    C ρ₁ θ₁ → C ρ₂ θ₂ → 
    perpendicular (ρ₁ * Real.cos θ₁) (ρ₁ * Real.sin θ₁) (ρ₂ * Real.cos θ₂) (ρ₂ * Real.sin θ₂) →
    1 / ρ₁^2 + 1 / ρ₂^2 = 10 / 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l323_32360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_equality_l323_32366

open Real

-- Problem 1
theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (2 * π - α) * Real.sin (π + α) * Real.cos (-π - α)) / (Real.sin (3 * π - α) * Real.cos (π - α)) = Real.sin α :=
by sorry

-- Problem 2
theorem trigonometric_equality (x : ℝ) :
  Real.cos x / (1 - Real.sin x) = (1 + Real.sin x) / Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_trigonometric_equality_l323_32366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l323_32394

theorem sine_difference_value (α : ℝ) 
  (h1 : Real.cos α = 3/5) 
  (h2 : α ∈ Set.Ioo 0 (Real.pi/2)) : 
  Real.sin (α - Real.pi/6) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l323_32394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_inequality_l323_32372

noncomputable def nested_radical (k : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => Real.sqrt (k + nested_radical k n)

theorem nested_radical_inequality (n : ℕ) :
  nested_radical 2 n + nested_radical 6 n < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_inequality_l323_32372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_and_mode_of_data_set_l323_32368

def data_set : List ℕ := [1, 1, 2, 3, 3, 3, 3, 4, 5, 5]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

def mode (lst : List ℕ) : ℕ :=
  lst.foldr (λ x acc => if (lst.count x) > (lst.count acc) then x else acc) 0

theorem mean_and_mode_of_data_set :
  mean data_set = 3 ∧ mode data_set = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_and_mode_of_data_set_l323_32368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_equation_l323_32321

open Real

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.exp (tan (x / 2))

-- State the theorem
theorem y_satisfies_equation (x : ℝ) : 
  (deriv y x) * sin x = y x * log (y x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_equation_l323_32321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l323_32384

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (7 * x^2 + 4) / (4 * x^2 + 3 * x - 1)

-- State the theorem
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → |f x - 7/4| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l323_32384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l323_32316

/-- The volume of a cylinder with given radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The ratio of volumes of two cylinders formed from a 7 × 12 rectangle -/
theorem rectangle_cylinder_volume_ratio :
  let rect_width : ℝ := 7
  let rect_height : ℝ := 12
  let cylinder1_radius : ℝ := rect_width / (2 * Real.pi)
  let cylinder1_height : ℝ := rect_height
  let cylinder2_radius : ℝ := rect_height / (2 * Real.pi)
  let cylinder2_height : ℝ := rect_width
  let volume1 := cylinderVolume cylinder1_radius cylinder1_height
  let volume2 := cylinderVolume cylinder2_radius cylinder2_height
  (max volume1 volume2) / (min volume1 volume2) = 84 / 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l323_32316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l323_32398

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (Real.pi / 2) → f (Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l323_32398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l323_32308

/-- A right pyramid with a square base and an equilateral triangle face -/
structure RightPyramid where
  /-- The side length of the equilateral triangle face -/
  side_length : ℝ
  /-- Assertion that the side length is positive -/
  side_length_pos : side_length > 0

/-- The volume of the right pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (1/3) * p.side_length^2 * (p.side_length * Real.sqrt 3 / 2)

/-- Theorem stating the volume of a specific right pyramid -/
theorem volume_of_specific_pyramid :
  ∃ (p : RightPyramid), p.side_length = 6 ∧ volume p = 36 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l323_32308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l323_32348

noncomputable def f (l : ℝ) (x : ℝ) : ℝ := l * Real.log (x + 1) - Real.sin x

theorem function_properties :
  ∀ l : ℝ,
  (∀ x : ℝ, x > 0 → f l (x + 2 * Real.pi) = f l x → l = 0) ∧
  (∃! x : ℝ, x ≥ Real.pi / 2 ∧ f 1 x = 0) ∧
  ((∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f l x ≥ 2 * (1 - Real.exp x)) → l ≥ -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l323_32348
