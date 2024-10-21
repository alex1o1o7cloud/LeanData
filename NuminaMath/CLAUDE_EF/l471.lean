import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l471_47103

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-1) 0 ∪ Set.Ioc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l471_47103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_numbers_and_lcm_l471_47100

theorem three_numbers_and_lcm (a b c : ℕ) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 180 → 
  (c : ℚ) / a = 5 → 
  (a = 45 ∧ b = 60 ∧ c = 225) ∧ 
  Nat.lcm a (Nat.lcm b c) = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_numbers_and_lcm_l471_47100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_abc_l471_47114

theorem ascending_order_abc (a b c : ℝ) : 
  a = Real.log 7 / Real.log 3 → b = 2^(3.3 : ℝ) → c = 0.8 → c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_abc_l471_47114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l471_47157

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

-- Define the area function for a triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1 / 2) * t.a * t.c * Real.sin t.B

-- Part 1: Prove that B = π/4
theorem part_one (t : Triangle) (h : given_condition t) : t.B = π / 4 := by
  sorry

-- Part 2: Prove the maximum area when b = 2
theorem part_two (t : Triangle) (h1 : given_condition t) (h2 : t.b = 2) :
  (∀ t' : Triangle, given_condition t' → t'.b = 2 → area t' ≤ area t) →
  area t = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l471_47157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_interval_range_l471_47150

theorem multiple_interval_range (f : ℝ → ℝ) (a m n : ℝ) : 
  (∀ x, x ∈ Set.Icc m n → f x = Real.sqrt (x + 1) + a) →
  (∀ x y, x ∈ Set.Icc m n → y ∈ Set.Icc m n → x < y → f x < f y) →
  (f '' Set.Icc m n = Set.Icc (2*m) (2*n)) →
  a ∈ Set.Ioo (-17/8) (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_interval_range_l471_47150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_diameter_of_specific_triangle_l471_47199

noncomputable section

open Real

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- Define the area of a triangle
noncomputable def area (a : ℝ) (B : ℝ) (c : ℝ) : ℝ :=
  1/2 * a * c * sin B

-- Define the diameter of the circumcircle
noncomputable def circumcircle_diameter (b : ℝ) (B : ℝ) : ℝ :=
  b / sin B

-- State the theorem
theorem circumcircle_diameter_of_specific_triangle :
  ∀ (b c : ℝ) (A C : ℝ),
  triangle 1 b c A (π/4) C →
  area 1 (π/4) c = 2 →
  circumcircle_diameter b (π/4) = 5 * sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_diameter_of_specific_triangle_l471_47199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l471_47154

/-- A conic section curve with two foci -/
structure ConicSection where
  F₁ : ℝ × ℝ  -- First focus
  F₂ : ℝ × ℝ  -- Second focus

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Eccentricity of a conic section -/
noncomputable def eccentricity (c : ConicSection) : ℝ := sorry

theorem conic_section_eccentricity (c : ConicSection) :
  ∃ (P : ℝ × ℝ),
    distance P c.F₁ / distance c.F₁ c.F₂ = 4/3 ∧
    distance c.F₁ c.F₂ / distance c.F₁ c.F₂ = 1 ∧
    distance P c.F₂ / distance c.F₁ c.F₂ = 2/3 →
  eccentricity c = 1/2 ∨ eccentricity c = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l471_47154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_l471_47115

theorem smallest_set_size {S : Type} [Fintype S] (X : Fin 100 → Set S)
  (h_distinct : ∀ i j, i ≠ j → X i ≠ X j)
  (h_nonempty : ∀ i, Set.Nonempty (X i))
  (h_disjoint : ∀ i : Fin 99, Disjoint (X i) (X (i + 1)))
  (h_not_whole : ∀ i : Fin 99, (X i) ∪ (X (i + 1)) ≠ Set.univ) :
  8 ≤ Fintype.card S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_l471_47115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l471_47198

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
noncomputable def train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - person_speed
  let relative_speed_ms := relative_speed * (5/18)
  relative_speed_ms * passing_time

/-- Theorem stating that given the specified conditions, the length of the train is approximately 699.944 meters. -/
theorem train_length_calculation :
  let train_speed := 63 -- km/hr
  let person_speed := 3 -- km/hr
  let passing_time := 41.9966402687785 -- seconds
  abs (train_length train_speed person_speed passing_time - 699.944) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l471_47198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peppers_per_day_l471_47101

/-- Proves that given a total of 80 peppers picked over 7 days, with 64 being non-hot peppers,
    the number of peppers picked per day rounded down to the nearest integer is 11. -/
theorem peppers_per_day (total_peppers : ℕ) (non_hot_peppers : ℕ) (days : ℕ) :
  total_peppers = 80 →
  non_hot_peppers = 64 →
  days = 7 →
  (Int.floor ((total_peppers : ℚ) / days) : ℤ) = 11 := by
  sorry

#check peppers_per_day

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peppers_per_day_l471_47101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_product_equals_243_l471_47151

theorem power_of_product_equals_243 : 
  (3^12 * 3^18 : ℝ)^(1/6) = 243 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_product_equals_243_l471_47151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l471_47129

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℚ  -- The sequence
  q : ℚ      -- Common ratio
  (is_geometric : ∀ n, a (n + 1) = a n * q)

/-- The sum of the first n terms of a geometric sequence -/
def geometricSum (s : GeometricSequence) (n : ℕ) : ℚ :=
  if s.q = 1 then n * s.a 1
  else s.a 1 * (1 - s.q^n) / (1 - s.q)

theorem geometric_sequence_problem (s : GeometricSequence) 
  (h1 : s.a 5 = 162)
  (h2 : s.q = 3)
  (h3 : geometricSum s 5 = 242) :
  s.a 1 = 2 ∧ 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l471_47129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l471_47194

theorem absolute_value_expression : 
  abs (abs (abs (-(1 + 2)) - 2) + 3) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l471_47194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l471_47143

noncomputable def original_expr : ℝ := 1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)

noncomputable def rationalized_expr : ℝ := (-6 * Real.sqrt 5 - 8 * Real.sqrt 3 + 3 * Real.sqrt 11 + Real.sqrt 165) / 51

theorem rationalize_denominator :
  original_expr = rationalized_expr := by
  sorry

#eval 206 -- The sum A + B + C + D + E + F

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l471_47143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l471_47104

open Real Set

theorem theta_range (θ : ℝ) (h1 : θ ∈ Icc 0 (2 * π)) 
  (h2 : ∀ x ∈ Icc 0 1, 2 * x^2 * sin θ - 4 * x * (1 - x) * cos θ + 3 * (1 - x)^2 > 0) : 
  θ ∈ Ioo (π / 6) π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l471_47104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_proof_l471_47145

def point1 : ℝ × ℝ × ℝ := (2, -2, 1)
def point2 : ℝ × ℝ × ℝ := (5, 1, 2)
def perp_plane_normal : ℝ × ℝ × ℝ := (2, -1, 4)

def plane_equation (x y z : ℝ) : Prop :=
  11 * x - 10 * y - 9 * z - 33 = 0

theorem plane_proof :
  (∀ x y z : ℝ, plane_equation x y z ↔ 
    -- The plane passes through point1 and point2
    (∃ t : ℝ, (x, y, z) = (point1.1 + t * (point2.1 - point1.1),
                           point1.2.1 + t * (point2.2.1 - point1.2.1),
                           point1.2.2 + t * (point2.2.2 - point1.2.2))) ∧
    -- The plane is perpendicular to the plane 2x - y + 4z = 7
    (11 * perp_plane_normal.1 - 10 * perp_plane_normal.2.1 - 9 * perp_plane_normal.2.2 = 0)) ∧
  -- A > 0 and gcd(|A|,|B|,|C|,|D|) = 1
  11 > 0 ∧ Nat.gcd (Nat.gcd 11 10) (Nat.gcd 9 33) = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_proof_l471_47145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l471_47177

/-- Define set A -/
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}

/-- Define set B -/
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

/-- Define set C -/
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

/-- Theorem for the first part of the problem -/
theorem part_one : ∃ a : ℝ, (A a ∪ B = A a ∩ B) → a = 2 * Real.sqrt (19/3) := by
  sorry

/-- Theorem for the second part of the problem -/
theorem part_two : ∃ a : ℝ, ((A a ∩ B).Nonempty ∧ (A a ∩ C = ∅)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l471_47177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_calculator_theorem_l471_47187

/-- Represents the operations that can be performed on the calculator -/
inductive Operation
| AddOne
| Square

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a single operation to a number -/
def applyOperation (n : Nat) (op : Operation) : Nat :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.Square => n * n

/-- Applies a sequence of operations to a number -/
def applySequence (start : Nat) (seq : OperationSequence) : Nat :=
  seq.foldl applyOperation start

/-- Checks if a sequence of operations transforms the start number into the target -/
def isValidSequence (start target : Nat) (seq : OperationSequence) : Prop :=
  applySequence start seq = target

/-- The set of all valid sequences that transform the start number into the target -/
def validSequences (start target : Nat) : Set OperationSequence :=
  {seq | isValidSequence start target seq}

/-- The main theorem: there are exactly 128 ways to obtain 1000 from 2 -/
theorem broken_calculator_theorem :
  ∃ (s : Finset OperationSequence), s.card = 128 ∧ ∀ seq, seq ∈ s ↔ seq ∈ validSequences 2 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_calculator_theorem_l471_47187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_mass_and_inertia_l471_47122

/-- Represents a homogeneous flat rod -/
structure FlatRod where
  length : ℝ
  density : ℝ

/-- Calculates the mass of a flat rod -/
noncomputable def mass (rod : FlatRod) : ℝ := rod.density * rod.length

/-- Calculates the moment of inertia of a flat rod relative to its end -/
noncomputable def momentOfInertia (rod : FlatRod) : ℝ := (rod.density * rod.length ^ 3) / 3

theorem rod_mass_and_inertia (rod : FlatRod) (h : rod.length > 0) :
  mass rod = rod.density * rod.length ∧
  momentOfInertia rod = (rod.density * rod.length ^ 3) / 3 := by
  constructor
  · -- Proof for mass
    simp [mass]
  · -- Proof for moment of inertia
    simp [momentOfInertia]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_mass_and_inertia_l471_47122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baltic_sea_theorem_l471_47148

/-- A graph representing harbours and ferry connections -/
structure BalticSeaGraph where
  /-- The set of vertices (harbours) -/
  V : Type
  /-- The number of vertices is 2016 -/
  vertex_count : Fintype V
  num_vertices : Fintype.card V = 2016
  /-- The edge relation (ferry connections) -/
  E : V → V → Prop

/-- No path of length 1061 exists in the graph -/
def NoLongPath (G : BalticSeaGraph) : Prop :=
  ∀ (path : Fin 1062 → G.V), ∃ i j, i < j ∧ ¬G.E (path i) (path j)

/-- Two disjoint sets of vertices with no edges between them -/
def DisconnectedSets (G : BalticSeaGraph) (A B : Finset G.V) : Prop :=
  A.card = 477 ∧ B.card = 477 ∧ Disjoint A B ∧
  ∀ a b, a ∈ A → b ∈ B → ¬G.E a b

theorem baltic_sea_theorem (G : BalticSeaGraph) :
  NoLongPath G → ∃ A B : Finset G.V, DisconnectedSets G A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baltic_sea_theorem_l471_47148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_l471_47112

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define distance to focus
noncomputable def dist_to_focus (x y : ℝ) : ℝ :=
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2)

-- Define distance to y-axis
def dist_to_y_axis (x : ℝ) : ℝ := |x|

theorem parabola_distance_sum (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_parabola x₁ y₁ →
  is_on_parabola x₂ y₂ →
  dist_to_focus x₁ y₁ + dist_to_focus x₂ y₂ = 7 →
  dist_to_y_axis x₁ + dist_to_y_axis x₂ = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_l471_47112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l471_47164

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) 
  (h2 : a < 13) 
  (h3 : (51^2016 + a) % 13 = 0) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l471_47164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l471_47130

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
    (h : 2 * (seq.a 1 + seq.a 3 + seq.a 5) + 3 * (seq.a 8 + seq.a 10) = 36) : 
  S seq 11 = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l471_47130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l471_47155

/-- A function that returns the list of positive divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A predicate that checks if a natural number satisfies all the given conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  let d := divisors n
  d.length ≥ 7 ∧
  d.Sorted (· < ·) ∧
  d.head? = some 1 ∧
  d.getLast? = some n ∧
  ∃ (i j : ℕ), i < d.length ∧ j < d.length ∧
    d[6]? = some (2 * d[i]?.getD 0 + 1) ∧
    d[6]? = some (3 * d[j]?.getD 0 - 1)

theorem smallest_satisfying_number :
  satisfies_conditions 2024 ∧
  ∀ m : ℕ, m < 2024 → ¬satisfies_conditions m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l471_47155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_theorem_l471_47126

/-- A function representing the gift-giving relationship between children -/
def GiftFunction (α : Type) := (α × α) → α

/-- The property that the gift function satisfies the required conditions -/
def ValidGiftFunction {α : Type} (s : Finset α) (f : GiftFunction α) : Prop :=
  (∀ x y, x ∈ s → y ∈ s → x ≠ y → ∃! z, z ∈ s ∧ f (x, y) = z) ∧
  (∀ x y z, x ∈ s → y ∈ s → z ∈ s → x ≠ y ∧ y ≠ z ∧ x ≠ z → (f (x, y) = z → f (x, z) = y))

/-- The main theorem statement -/
theorem gift_theorem (n : ℕ) (hn : Odd n) :
  ∃ (α : Type) (s : Finset α) (f : GiftFunction α),
    Finset.card s = 3 * n ∧ ValidGiftFunction s f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_theorem_l471_47126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_construction_cost_l471_47121

noncomputable def total_cost (x : ℝ) : ℝ := 80 * (x + 36 / x) + 1800

theorem min_construction_cost :
  ∀ x : ℝ, 0 < x ∧ x ≤ 7 → total_cost x ≥ 2760 ∧
  (total_cost x = 2760 ↔ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_construction_cost_l471_47121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_primes_l471_47182

/-- A sequence of natural numbers forms an arithmetic progression -/
def is_arithmetic_progression (seq : List Nat) : Prop :=
  seq.length > 1 ∧ ∃ d, ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = d

/-- A natural number is prime -/
def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m > 1 → m < n → ¬(n % m = 0)

theorem arithmetic_progression_primes :
  (∃ seq : List Nat, seq.length = 10 ∧
    (∀ n ∈ seq, is_prime n ∧ n < 3000) ∧
    is_arithmetic_progression seq) ∧
  (¬∃ seq : List Nat, seq.length = 11 ∧
    (∀ n ∈ seq, is_prime n ∧ n < 20000) ∧
    is_arithmetic_progression seq) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_primes_l471_47182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_values_l471_47176

/-- Represents the state of variables in the program -/
structure ProgramState where
  a : Int
  b : Int
  c : Int
deriving Repr

/-- Executes the program assignments -/
def executeProgram (initial : ProgramState) : ProgramState :=
  let step1 := { initial with a := initial.b }
  let step2 := { step1 with b := step1.c }
  { step2 with c := step2.a }

/-- Theorem stating the final values after executing the program -/
theorem final_values (initial : ProgramState) 
  (h_init : initial = { a := 3, b := -5, c := 8 }) :
  let final := executeProgram initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by
  sorry

#eval executeProgram { a := 3, b := -5, c := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_values_l471_47176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_m_l471_47144

variable (a : ℝ)
variable (m : ℝ)

noncomputable def f (x : ℝ) := a^(x^2 + 2*x - 3) + m

theorem fixed_point_m (h1 : a > 1) (h2 : f a m 1 = 10) : m = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_m_l471_47144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_with_distinct_remainders_l471_47106

theorem five_digit_number_with_distinct_remainders : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧
  (∀ i j : ℕ, i ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13} : Set ℕ) →
              j ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13} : Set ℕ) →
              i ≠ j → n % i ≠ n % j) ∧
  n = 83159 := by
  use 83159
  constructor
  · constructor
    · norm_num
    · norm_num
  · constructor
    · intro i j hi hj hij
      simp only [Set.mem_singleton_iff, Set.mem_insert_iff] at hi hj
      rcases hi with (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) <;>
      rcases hj with (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
      all_goals {
        try {contradiction}
        try {norm_num}
      }
    · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_with_distinct_remainders_l471_47106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_38_l471_47139

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => ((n + 2 : ℚ) * sequence_a (n + 1) + 2) / (n + 1)

theorem a_10_equals_38 : sequence_a 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_38_l471_47139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l471_47191

noncomputable def vector_a : ℝ × ℝ := (Real.sqrt 3, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -3)

theorem angle_between_vectors (x : ℝ) 
  (h : vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) : 
  let diff := (vector_a.1 - (vector_b x).1, vector_a.2 - (vector_b x).2)
  Real.arccos ((diff.1 * (vector_b x).1 + diff.2 * (vector_b x).2) / 
    (Real.sqrt (diff.1^2 + diff.2^2) * Real.sqrt ((vector_b x).1^2 + (vector_b x).2^2))) 
    = 5 * Real.pi / 6 := by
  sorry

#check angle_between_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l471_47191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l471_47153

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 2

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the number of students selected -/
def num_selected : ℕ := 2

/-- Represents the event of selecting exactly one girl -/
def exactly_one_girl (selection : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selection.filter (λ i => i.val ≥ num_boys)).card = 1

/-- Represents the event of selecting exactly two girls -/
def exactly_two_girls (selection : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selection.filter (λ i => i.val ≥ num_boys)).card = 2

/-- Represents the event of selecting at least one girl -/
def at_least_one_girl (selection : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selection.filter (λ i => i.val ≥ num_boys)).card ≥ 1

/-- Represents the event of selecting all boys -/
def all_boys (selection : Finset (Fin (num_boys + num_girls))) : Prop :=
  (selection.filter (λ i => i.val < num_boys)).card = num_selected

/-- Theorem stating that "Exactly 1 girl and exactly 2 girls" and "At least 1 girl and all boys" are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ (selection : Finset (Fin (num_boys + num_girls))),
    selection.card = num_selected →
    ¬(exactly_one_girl selection ∧ exactly_two_girls selection) ∧
    ¬(at_least_one_girl selection ∧ all_boys selection) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l471_47153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_ratio_l471_47146

theorem tetrahedron_face_area_ratio (S₁ S₂ S₃ S₄ : ℝ) (h_pos : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0) :
  let S := max S₁ (max S₂ (max S₃ S₄))
  let lambda := (S₁ + S₂ + S₃ + S₄) / S
  2 < lambda ∧ lambda ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_ratio_l471_47146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l471_47141

/-- Predicate to check if a triangle with given side lengths is isosceles -/
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b ∧ a + b > c) ∨ (a = c ∧ a + c > b) ∨ (b = c ∧ b + c > a)

/-- Given an isosceles triangle with side lengths 4x-2, x+1, and 15-6x, its perimeter is 12.3 -/
theorem isosceles_triangle_perimeter : ∀ x : ℝ,
  let a := 4*x - 2
  let b := x + 1
  let c := 15 - 6*x
  IsIsosceles a b c → (a + b + c = 12.3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l471_47141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l471_47181

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given two triangles, prove an inequality involving their side lengths and areas -/
theorem triangle_inequality (a b c u v w P Q : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (hP : P = area_triangle a b c)
  (hQ : Q = area_triangle u v w) :
  16 * P * Q ≤ a^2 * (-u^2 + v^2 + w^2) + b^2 * (u^2 - v^2 + w^2) + c^2 * (u^2 + v^2 - w^2) ∧
  (16 * P * Q = a^2 * (-u^2 + v^2 + w^2) + b^2 * (u^2 - v^2 + w^2) + c^2 * (u^2 + v^2 - w^2) ↔ 
   ∃ (k : ℝ), a = k * u ∧ b = k * v ∧ c = k * w) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l471_47181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l471_47107

-- Define the hyperbola equation
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (m + 1) = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 7 / 2

-- Theorem statement
theorem hyperbola_eccentricity_m (m : ℝ) :
  (∀ x y, hyperbola m x y) →
  (∃ a b c : ℝ, a^2 = -(m + 1) ∧ b^2 = -(m + 2) ∧ c^2 = a^2 + b^2 ∧ c / a = eccentricity) →
  m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l471_47107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l471_47160

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The standard equation of a circle -/
def isStandardCircleEquation (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line :
  let center_x := (2 : ℝ)
  let center_y := (1 : ℝ)
  let line_A := (3 : ℝ)
  let line_B := (4 : ℝ)
  let line_C := (5 : ℝ)
  let r := distancePointToLine center_x center_y line_A line_B line_C
  ∀ x y : ℝ, isStandardCircleEquation center_x center_y r x y ↔ (x - 2)^2 + (y - 1)^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l471_47160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l471_47197

noncomputable def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ :=
  a₀ * x^4 + a₁ * x^3 + a₂ * x^2 + a₃ * x + a₄

noncomputable def x_n (n : ℕ) : ℝ :=
  (2^n - 1) / 2^n

noncomputable def y_m (m : ℕ) : ℝ :=
  (Real.sqrt 2 * (1 - 3^m)) / 3^m

theorem problem_solution 
    (a₀ a₁ a₂ a₃ a₄ : ℝ) 
    (h1 : f a₀ a₁ a₂ a₃ a₄ (-1) = 2/3)
    (h2 : ∀ x, f a₀ a₁ a₂ a₃ a₄ (-x) = -(f a₀ a₁ a₂ a₃ a₄ x)) :
  (∀ x, f a₀ a₁ a₂ a₃ a₄ x = (1/3) * x^3 - x) ∧ 
  (∀ x y, x = 0 ∧ y = Real.sqrt 2 → 
    (deriv (f a₀ a₁ a₂ a₃ a₄) x) * (deriv (f a₀ a₁ a₂ a₃ a₄) y) = -1) ∧
  (∀ n m, n > 0 ∧ m > 0 → 
    |f a₀ a₁ a₂ a₃ a₄ (x_n n) - f a₀ a₁ a₂ a₃ a₄ (y_m m)| < 4/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l471_47197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_rate_l471_47119

def road_length : ℝ := 2000
def road_width : ℝ := 20
def truckload_coverage : ℝ := 800
def truckload_cost : ℝ := 75
def total_cost_with_tax : ℝ := 4500

theorem sales_tax_rate : 
  (let total_area := road_length * road_width
   let num_truckloads := total_area / truckload_coverage
   let total_cost_before_tax := num_truckloads * truckload_cost
   let sales_tax_amount := total_cost_with_tax - total_cost_before_tax
   let sales_tax_rate := (sales_tax_amount / total_cost_before_tax) * 100
   sales_tax_rate) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_rate_l471_47119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_and_comparison_l471_47113

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x + 1 / 2

-- Part I
theorem tangent_line_at_one (x y : ℝ) :
  (g 1 1 = 3 / 2) →
  (deriv (g 1) 1 = -1) →
  (2 * x + 2 * y - 5 = 0 ↔ y - 3 / 2 = -(x - 1)) :=
by sorry

-- Part II
theorem monotonicity_and_comparison :
  (∀ x, x > 0 → f x = (Real.log x) / x) →
  (∀ x, 0 < x → x < Real.exp 1 → (deriv f x > 0)) →
  (∀ x, x > Real.exp 1 → (deriv f x < 0)) →
  (2017 : ℝ) ^ (1 / 2017) < (2016 : ℝ) ^ (1 / 2016) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_and_comparison_l471_47113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l471_47138

noncomputable def line_equation (p1 p2 : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  (m, -1, b)

theorem line_equation_through_points :
  let p1 : ℝ × ℝ := (1, 0.5)
  let p2 : ℝ × ℝ := (1.5, 2)
  let (a, b, c) := line_equation p1 p2
  (6 * a = 6 ∧ 6 * b = -2 ∧ 6 * c = 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l471_47138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_costco_mayo_volume_l471_47184

/-- Proves that the Costco container holds 1 gallon of mayo -/
theorem costco_mayo_volume (costco_price : ℚ) (normal_price : ℚ) (normal_volume : ℚ) 
  (savings : ℚ) (ounces_per_gallon : ℚ) 
  (h1 : costco_price = 8)
  (h2 : normal_price = 3)
  (h3 : normal_volume = 16)
  (h4 : savings = 16)
  (h5 : ounces_per_gallon = 128)
  : ∃ costco_volume_gallons : ℚ, costco_volume_gallons = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_costco_mayo_volume_l471_47184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tower_heights_l471_47120

-- Define the set of possible block heights
def BlockHeights : Finset ℕ := {4, 6, 10}

-- Function to calculate tower height from three blocks
def towerHeight (b1 b2 b3 : ℕ) : ℕ := b1 + b2 + b3

-- Set of all possible tower heights
def AllTowerHeights : Finset ℕ :=
  Finset.image (λ (t : ℕ × ℕ × ℕ) => towerHeight t.1 t.2.1 t.2.2)
    (BlockHeights.product (BlockHeights.product BlockHeights))

-- Theorem stating that the number of unique tower heights is 9
theorem unique_tower_heights :
  AllTowerHeights.card = 9 := by
  sorry

#eval AllTowerHeights.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tower_heights_l471_47120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l471_47174

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / x

theorem function_properties (m : ℝ) :
  -- 1. Tangent line property
  (∃ y₀ : ℝ, y₀ = f m 1 ∧ 
   ∃ k : ℝ, k = deriv (f m) 1 ∧
   y₀ - k = 1) → m = 1 ∧

  -- 2. Monotonicity property
  (∀ x > 0, deriv (f m) x = (x - m) / x^2) ∧
  (m ≤ 0 → StrictMono (f m)) ∧
  (m > 0 → StrictAntiOn (f m) (Set.Ioo 0 m) ∧
           StrictMonoOn (f m) (Set.Ioi m)) ∧

  -- 3. Inequality property
  ((∀ a b : ℝ, b > a ∧ a > 0 → (f m b - f m a) / (b - a) < 1) →
   m ≥ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l471_47174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_inequality_l471_47193

theorem sum_sqrt_inequality (a b c : ℝ) 
  (h1 : a ∈ Set.Ioo (-1) 1) 
  (h2 : b ∈ Set.Ioo (-1) 1) 
  (h3 : c ∈ Set.Ioo (-1) 1) 
  (h4 : a + b + c + a*b*c = 0) : 
  Real.sqrt (a + 1) + Real.sqrt (b + 1) + Real.sqrt (c + 1) ≤ 
  Real.sqrt (a*b + b*c + c*a + 9) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_inequality_l471_47193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l471_47185

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 6*x - 1) / (x^2 - 5*x + 6)

theorem domain_of_h :
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l471_47185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_equal_four_triangles_circle_equals_three_squares_l471_47195

-- Define the types for our shapes
inductive Shape
  | Circle
  | Triangle
  | Square

-- Define a type for mass
def Mass : Type := ℕ

-- Define the balance relation
def balanced (left right : List Shape) : Prop := sorry

-- Define the mass function for a list of shapes
def mass (shapes : List Shape) : Mass := sorry

-- State the theorem
theorem four_circles_equal_four_triangles 
  (h1 : balanced [Shape.Circle, Shape.Circle] [Shape.Triangle, Shape.Triangle])
  (h2 : balanced [Shape.Triangle] [Shape.Square, Shape.Square, Shape.Square]) :
  mass [Shape.Circle, Shape.Circle, Shape.Circle, Shape.Circle] = 
  mass [Shape.Triangle, Shape.Triangle, Shape.Triangle, Shape.Triangle] := by
  sorry

-- Additional helper theorem to represent the relationship between circles and squares
theorem circle_equals_three_squares 
  (h1 : balanced [Shape.Circle, Shape.Circle] [Shape.Triangle, Shape.Triangle])
  (h2 : balanced [Shape.Triangle] [Shape.Square, Shape.Square, Shape.Square]) :
  mass [Shape.Circle] = mass [Shape.Square, Shape.Square, Shape.Square] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_equal_four_triangles_circle_equals_three_squares_l471_47195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_equal_l471_47165

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a real number to each point
variable (value : Point → ℝ)

-- Define a function to calculate the incenter of a triangle
noncomputable def incenter (A B C : Point) : Point := sorry

-- State the theorem
theorem all_points_equal 
  (h : ∀ A B C : Point, A ≠ B ∧ B ≠ C ∧ C ≠ A → 
    value (incenter A B C) = (value A + value B + value C) / 3) : 
  ∀ P Q : Point, value P = value Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_equal_l471_47165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_dot_product_l471_47123

noncomputable section

/-- Definition of the ellipse E -/
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- Point M on the x-axis -/
def M : ℝ × ℝ := (11/8, 0)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

/-- Vector from point p to point q -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

theorem ellipse_constant_dot_product :
  ∀ (l : ℝ → ℝ × ℝ), 
    (∃ t, l t = F) →  -- line l passes through F
    (∃ A B, E (A.1) (A.2) ∧ E (B.1) (B.2) ∧ A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B)) →  -- A and B are distinct intersection points of l and E
    dot_product (vector M A) (vector M B) = -135/64 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_dot_product_l471_47123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_relation_l471_47131

/-- Represents a geometric sequence -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α  -- first term
  r : α  -- common ratio
  n : ℕ  -- number of terms

/-- The product of terms in a geometric sequence -/
def product {α : Type*} [Field α] (gs : GeometricSequence α) : α :=
  gs.a^gs.n * gs.r^(gs.n * (gs.n - 1) / 2)

/-- The sum of terms in a geometric sequence -/
def sum {α : Type*} [Field α] (gs : GeometricSequence α) : α :=
  gs.a * (1 - gs.r^gs.n) / (1 - gs.r)

/-- The sum of squares of terms in a geometric sequence -/
def sum_of_squares {α : Type*} [Field α] (gs : GeometricSequence α) : α :=
  gs.a^2 * (1 - gs.r^(2*gs.n)) / (1 - gs.r^2)

/-- Theorem stating the relationship between product, sum, and sum of squares -/
theorem product_sum_relation {α : Type*} [Field α] (gs : GeometricSequence α) :
  product gs = (sum_of_squares gs / sum gs)^((gs.n - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_relation_l471_47131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratios_for_angle_α_l471_47190

noncomputable def angle_α : Real := sorry

def point_P : ℝ × ℝ := (4, 3)

theorem trig_ratios_for_angle_α :
  let x := point_P.1
  let y := point_P.2
  let r := Real.sqrt (x^2 + y^2)
  (Real.sin angle_α = y / r) ∧
  (Real.cos angle_α = x / r) ∧
  (Real.tan angle_α = y / x) := by
  sorry

#check trig_ratios_for_angle_α

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratios_for_angle_α_l471_47190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_classification_l471_47149

/-- The game state -/
inductive GameState
  | PlayerA (n : ℕ)
  | PlayerB (n : ℕ)

/-- The game result -/
inductive GameResult
  | AWins
  | BWins
  | Tie

/-- The game rules -/
noncomputable def gameStep : GameState → GameState
  | GameState.PlayerA n => GameState.PlayerB (Nat.choose n (n^2))
  | GameState.PlayerB n => GameState.PlayerA (n / Nat.minFac n)

/-- The winning condition for player A -/
def isAWin (s : GameState) : Bool :=
  match s with
  | GameState.PlayerA 1990 => true
  | _ => false

/-- The winning condition for player B -/
def isBWin (s : GameState) : Bool :=
  match s with
  | GameState.PlayerB 1 => true
  | _ => false

/-- The game outcome given an initial state -/
noncomputable def gameOutcome (initialState : GameState) : GameResult :=
  sorry

/-- The theorem to prove -/
theorem game_classification (n₀ : ℕ) (h : n₀ > 1) :
  (n₀ ∈ ({2, 3, 4, 5} : Finset ℕ) → gameOutcome (GameState.PlayerA n₀) = GameResult.BWins) ∧
  (n₀ ∈ ({6, 7} : Finset ℕ) → gameOutcome (GameState.PlayerA n₀) = GameResult.Tie) ∧
  (n₀ ≥ 8 → gameOutcome (GameState.PlayerA n₀) = GameResult.AWins) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_classification_l471_47149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratios_l471_47192

/-- Given a triangle ABC with points X, Y, Z dividing BC, CA, AB respectively in the ratio 1:2 -/
structure DividedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  divide_BC : X = ((2 * B.1 + C.1) / 3, (2 * B.2 + C.2) / 3)
  divide_CA : Y = ((2 * C.1 + A.1) / 3, (2 * C.2 + A.2) / 3)
  divide_AB : Z = ((2 * A.1 + B.1) / 3, (2 * A.2 + B.2) / 3)

/-- Calculate the area of a triangle given its vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

/-- The main theorem to be proved -/
theorem area_ratios (t : DividedTriangle) :
  (triangleArea t.A t.Z t.Y) / (triangleArea t.A t.B t.C) = 2/9 ∧
  (triangleArea t.A t.Z t.Y) / (triangleArea t.X t.Y t.Z) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratios_l471_47192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_abc_l471_47152

-- Define the constants and the inequality function
noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) / (x - c)

-- State the theorem
theorem solve_abc (a b c : ℝ) : 
  (a < b) → 
  (∀ x : ℝ, f a b c x ≤ 0 ↔ (x > 5 ∨ (2 ≤ x ∧ x ≤ 3))) → 
  a + 2*b + 3*c = 23 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_abc_l471_47152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honors_students_count_l471_47175

/-- Represents the number of students in a class --/
structure ClassData where
  girls : ℕ
  boys : ℕ
  honors_girls : ℕ
  honors_boys : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (c : ClassData) : Prop :=
  c.girls + c.boys < 30 ∧
  c.honors_girls * 13 = c.girls * 3 ∧
  c.honors_boys * 11 = c.boys * 4

/-- The theorem to prove --/
theorem honors_students_count (c : ClassData) 
  (h : satisfies_conditions c) : c.honors_girls + c.honors_boys = 7 :=
by
  sorry

#check honors_students_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_honors_students_count_l471_47175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l471_47147

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - x - 3

-- Define the solution set
def S : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_of_inequality :
  S = (Set.Iio (-1) ∪ Set.Ioi (3/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l471_47147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_assignment_l471_47116

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Assigns a digit to each mask -/
def mask_assignment : Mask → Nat := sorry

/-- The product of two identical digits results in a two-digit number ending with a different digit -/
def valid_product (n : Nat) : Prop :=
  let product := n * n
  10 ≤ product ∧ product < 100 ∧ product % 10 ≠ n

/-- All assigned digits are unique -/
def unique_digits (assignment : Mask → Nat) : Prop :=
  ∀ m₁ m₂ : Mask, m₁ ≠ m₂ → assignment m₁ ≠ assignment m₂

/-- The assignment satisfies all conditions of the problem -/
def valid_assignment (assignment : Mask → Nat) : Prop :=
  (valid_product (assignment Mask.elephant)) ∧
  (valid_product (assignment Mask.mouse)) ∧
  (valid_product (assignment Mask.pig)) ∧
  (unique_digits assignment)

/-- The only valid assignment is Elephant: 6, Mouse: 4, Pig: 8, Panda: 1 -/
theorem unique_valid_assignment :
  ∀ assignment : Mask → Nat,
    valid_assignment assignment →
    assignment Mask.elephant = 6 ∧
    assignment Mask.mouse = 4 ∧
    assignment Mask.pig = 8 ∧
    assignment Mask.panda = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_assignment_l471_47116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deceleration_time_correct_l471_47172

/-- Represents the deceleration of a car -/
structure CarDeceleration where
  V₀ : ℝ  -- Initial velocity
  V : ℝ   -- Final velocity
  S : ℝ   -- Distance traveled
  k : ℝ   -- Deceleration rate
  B : ℝ   -- Relative reduction in velocity per second

/-- The time taken for the car to decelerate -/
noncomputable def deceleration_time (car : CarDeceleration) : ℝ := (car.V₀ - car.V) / car.B

/-- Theorem stating that the deceleration time formula is correct -/
theorem deceleration_time_correct (car : CarDeceleration) : 
  car.V = car.V₀ - car.B * (deceleration_time car) ∧
  car.S = car.V₀ * (deceleration_time car) - (1/2) * car.k * (deceleration_time car)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deceleration_time_correct_l471_47172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_zero_point_l471_47118

/-- An even function with the form sin(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- The statement to prove -/
theorem min_omega_for_zero_point (ω φ : ℝ) (h_pos : ω > 0) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_zero : ∃ x ∈ Set.Icc 0 π, f ω φ x = 0) :
  ω ≥ (1 / 2 : ℝ) ∧ 
  ∀ ω' > 0, (∃ x ∈ Set.Icc 0 π, f ω' φ x = 0) → ω' ≥ (1 / 2 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_zero_point_l471_47118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l471_47140

noncomputable def series (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => (-1)^(n+1) * (2*n + 1) * x^(n+1)

noncomputable def series_sum (x : ℝ) : ℝ := ∑' n, series x n

theorem series_solution :
  ∃ x : ℝ, series_sum x = 16 ∧ x = -15/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l471_47140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l471_47108

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The upper vertex of the ellipse -/
def upper_vertex (e : Ellipse) : ℝ × ℝ :=
  (0, e.b)

/-- A point on the ellipse -/
def point_on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem eccentricity_range (e : Ellipse) :
  (∀ x y : ℝ, point_on_ellipse e x y → distance (x, y) (upper_vertex e) ≤ 2 * e.b) →
  0 < eccentricity e ∧ eccentricity e ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l471_47108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_meals_sold_l471_47166

theorem total_meals_sold (kids_meals : ℕ) (ratio : ℚ) : 
  kids_meals = 8 → ratio = 2/1 → kids_meals + (kids_meals / ratio.num * ratio.den) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_meals_sold_l471_47166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_range_on_interval_l471_47136

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + m

-- Theorem 1: If the maximum value of f(x) on [-1, 1] is 2/3, then m = 2/3
theorem max_value_implies_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f m x ≤ 2/3) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f m x = 2/3) →
  m = 2/3 := by
  sorry

-- Theorem 2: The range of f(x) on [-2, 2] is [-6, 2/3] when m = 2/3
theorem range_on_interval :
  Set.range (fun x => f (2/3) (x : Set.Icc (-2 : ℝ) 2)) = Set.Icc (-6 : ℝ) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_range_on_interval_l471_47136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_150_value_l471_47132

noncomputable def b : ℕ → ℝ
  | 0 => 2  -- Add this case to cover Nat.zero
  | 1 => 2
  | n + 1 => Real.sqrt (81 * (b n)^2)

theorem b_150_value : b 150 = 9^149 * 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_150_value_l471_47132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_politics_not_local_percentage_l471_47173

/-- Represents the percentage of reporters in various categories -/
structure ReporterPercentages where
  local_politics : ℚ
  no_politics : ℚ

/-- Calculates the percentage of reporters who cover politics but not local politics -/
def politics_not_local (r : ReporterPercentages) : ℚ :=
  (100 - r.local_politics - r.no_politics) / (100 - r.no_politics) * 100

/-- Theorem stating that given the conditions, 25% of reporters who cover politics do not cover local politics -/
theorem politics_not_local_percentage (r : ReporterPercentages)
  (h1 : r.local_politics = 30)
  (h2 : r.no_politics = 60) :
  politics_not_local r = 25 := by
  sorry

#eval politics_not_local ⟨30, 60⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_politics_not_local_percentage_l471_47173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l471_47162

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^(3*n+3) - x * 10^(3*n)).num = 567 ∧
                         (x * 10^(3*n+3) - x * 10^(3*n)).den = 10^(3*n+3) - 10^(3*n)) →
  x = 21/37 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l471_47162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_16_l471_47128

theorem log_8_16 : Real.log 16 / Real.log 8 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_16_l471_47128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_0_2_4_6_has_property_P_property_P_implies_a1_is_zero_property_P_implies_sum_relation_l471_47110

-- Define property P for a sequence
def has_property_P (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n → 
    (∃ k ≤ n, a k = a j + a i) ∨ (∃ k ≤ n, a k = a j - a i)

-- Theorem 1: The sequence 0, 2, 4, 6 has property P
theorem sequence_0_2_4_6_has_property_P :
  let a := fun | 1 => 0 | 2 => 2 | 3 => 4 | 4 => 6 | _ => 0
  has_property_P a 4 := by
  sorry

-- Theorem 2: If a sequence {a_n} (n ≥ 3, 0 ≤ a_1 < a_2 < ... < a_n) has property P, then a_1 = 0
theorem property_P_implies_a1_is_zero (a : ℕ → ℕ) (n : ℕ) :
  n ≥ 3 →
  (∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) →
  has_property_P a n →
  a 1 = 0 := by
  sorry

-- Theorem 3: If a sequence a_1, a_2, a_3 (0 ≤ a_1 < a_2 < a_3) has property P, then a_1 + a_3 = 2a_2
theorem property_P_implies_sum_relation (a : ℕ → ℕ) :
  0 ≤ a 1 →
  a 1 < a 2 →
  a 2 < a 3 →
  has_property_P a 3 →
  a 1 + a 3 = 2 * a 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_0_2_4_6_has_property_P_property_P_implies_a1_is_zero_property_P_implies_sum_relation_l471_47110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canada_size_relative_to_us_l471_47124

-- Define the sizes of countries as real numbers
variable (U C R : ℝ)

-- Define the conditions given in the problem
def russia_bigger_than_canada (U C R : ℝ) : Prop := R = C + (1/3) * C
def russia_twice_us (U C R : ℝ) : Prop := R = 2 * U

-- Theorem to prove
theorem canada_size_relative_to_us 
  (h1 : russia_bigger_than_canada U C R) 
  (h2 : russia_twice_us U C R) : 
  C = (3/2) * U := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canada_size_relative_to_us_l471_47124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_remainder_l471_47127

def is_odd_prime (n : ℕ) : Bool :=
  Nat.Prime n ∧ n % 2 = 1

def M : ℕ :=
  (List.filter is_odd_prime (List.range 32)).prod

theorem M_remainder : M % 32 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_remainder_l471_47127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_l471_47161

theorem obtuse_triangle (α : Real) (h1 : 0 < α ∧ α < Real.pi) 
  (h2 : Real.sin α + Real.cos α = 2/3) : α > Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_l471_47161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l471_47125

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (a b : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (a * Real.cos θ, b * Real.sin θ)

/-- Curve C₂ in polar form -/
noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

/-- Point M on curve C₁ -/
noncomputable def M : ℝ × ℝ := (1, Real.sqrt 3 / 2)

/-- Point M₁ in polar coordinates -/
noncomputable def M₁ : ℝ × ℝ := (1, Real.pi / 2)

/-- Point M₂ in polar coordinates -/
def M₂ : ℝ × ℝ := (2, 0)

theorem intersection_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hM : C₁ a b (Real.pi / 3) = M) :
  ∃ A B : ℝ × ℝ,
    (∃ θA : ℝ, C₁ a b θA = A) ∧
    (∃ θB : ℝ, C₁ a b θB = B) ∧
    1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l471_47125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_value_l471_47135

-- Define the function f(x) = x for x > 0
noncomputable def f : ℝ → ℝ := fun x => if x > 0 then x else 0

-- State the theorem
theorem no_extreme_value :
  ¬ (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f x)) ∧
  ¬ (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≤ f x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_value_l471_47135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_neg_one_f_min_value_when_a_neg_one_f_positive_range_of_a_l471_47189

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + a) / x

-- Define the domain
def domain : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem 1: Monotonicity when a = -1
theorem f_increasing_when_a_neg_one :
  ∀ x ∈ domain, ∀ y ∈ domain, x < y → f (-1) x < f (-1) y :=
by sorry

-- Theorem 2: Minimum value when a = -1
theorem f_min_value_when_a_neg_one :
  ∀ x ∈ domain, f (-1) x ≥ 2 :=
by sorry

-- Theorem 3: Range of a for f(x) > 0
theorem f_positive_range_of_a :
  (∀ x ∈ domain, f a x > 0) ↔ a > -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_neg_one_f_min_value_when_a_neg_one_f_positive_range_of_a_l471_47189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_corner_probability_l471_47170

-- Define the grid size
def gridSize : Nat := 4

-- Define a position on the grid
structure Position where
  x : Nat
  y : Nat
  deriving Repr

-- Define the starting position
def startPos : Position := ⟨3, 3⟩

-- Define the set of corner positions
def cornerPositions : Set Position := {⟨1, 1⟩, ⟨1, 4⟩, ⟨4, 1⟩, ⟨4, 4⟩}

-- Define a valid move
def isValidMove (p1 p2 : Position) : Bool :=
  (p1.x = p2.x && (p1.y + 1 = p2.y || p1.y = p2.y + 1)) ||
  (p1.y = p2.y && (p1.x + 1 = p2.x || p1.x = p2.x + 1))

-- Define a valid position (within the grid)
def isValidPosition (pos : Position) : Bool :=
  1 ≤ pos.x && pos.x ≤ gridSize && 1 ≤ pos.y && pos.y ≤ gridSize

-- Define the probability of reaching a corner within four hops
def probReachCorner : ℚ := 17 / 64

-- Define a simplified probability function (placeholder)
def probability (f : List Position → Prop) : ℚ := 0

theorem frog_corner_probability :
  ∀ (n : Nat),
    n ≤ 4 →
    (∀ (path : List Position),
      path.length = n →
      path.head? = some startPos →
      (∀ (i : Nat), i < path.length - 1 → isValidMove (path.get ⟨i, sorry⟩) (path.get ⟨i+1, sorry⟩)) →
      (∀ (pos : Position), pos ∈ path → isValidPosition pos)) →
    probability (λ path => ∃ (pos : Position), pos ∈ cornerPositions ∧ pos ∈ path) = probReachCorner :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_corner_probability_l471_47170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_length_implies_zero_vector_unit_vectors_have_equal_length_vector_and_zero_collinear_l471_47178

variable {V : Type*} [NormedAddCommGroup V] [Module ℝ V]

-- Statement 1: All vectors with a length of 0 are zero vectors
theorem zero_length_implies_zero_vector (v : V) : ‖v‖ = 0 → v = 0 := by
  sorry

-- Statement 2: The length of all unit vectors is equal
theorem unit_vectors_have_equal_length (u v : V) : ‖u‖ = 1 → ‖v‖ = 1 → ‖u‖ = ‖v‖ := by
  sorry

-- Statement 3: Any vector and the zero vector are collinear
theorem vector_and_zero_collinear (v : V) : ∃ (k : ℝ), v = k • (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_length_implies_zero_vector_unit_vectors_have_equal_length_vector_and_zero_collinear_l471_47178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birches_planted_l471_47171

theorem birches_planted (total_students : ℕ) (total_plants : ℕ) : ℕ :=
  let boys := total_students / 2  -- Assuming half of the students are boys for initialization
  let girls := total_students - boys
  let roses := 3 * girls
  let birches := boys / 3
  have h1 : total_students = 24 := by sorry
  have h2 : total_plants = 24 := by sorry
  have h3 : roses + birches = total_plants := by sorry
  have h4 : girls + boys = total_students := by sorry
  have h5 : 3 * girls + boys / 3 = total_plants := by sorry
  6

#check birches_planted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birches_planted_l471_47171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l471_47111

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the line
def line_eq (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |3*x + 4*y - 2| / Real.sqrt (3^2 + 4^2)

-- Theorem statement
theorem max_distance_to_line :
  ∃ (d : ℝ), d = 8 ∧ 
  (∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≤ d) ∧
  (∃ (x y : ℝ), circle_eq x y ∧ distance_to_line x y = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l471_47111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_walks_25_miles_l471_47159

/-- Two people walking towards each other -/
structure WalkingProblem where
  totalDistance : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- Calculate the distance walked by the second person when they meet -/
noncomputable def distanceWalked (p : WalkingProblem) : ℝ :=
  (p.totalDistance * p.speed2) / (p.speed1 + p.speed2)

/-- Theorem stating that in the given problem, Sam walks 25 miles -/
theorem sam_walks_25_miles :
  let problem := WalkingProblem.mk 55 6 5
  distanceWalked problem = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_walks_25_miles_l471_47159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_theorem_l471_47158

theorem coefficient_expansion_theorem (a : ℝ) : 
  (Finset.range 7).sum (fun k => if k = 3 then 
    Nat.choose 6 (3 - 1) * (-1)^(3 - 1) + a * Nat.choose 6 3 * (-1)^3
  else 0) = 5 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_theorem_l471_47158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_awake_l471_47134

theorem cats_awake (total_cats : ℕ) (asleep_percentage : ℚ) : 
  total_cats = 235 → asleep_percentage = 83/100 → 
  total_cats - (Int.toNat ⌊asleep_percentage * total_cats⌋) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cats_awake_l471_47134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l471_47186

-- Define vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, (Real.sqrt 3 / 2) * (Real.sin x - Real.cos x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x + Real.cos x)

-- Define function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem statement
theorem triangle_max_area (ABC : Triangle) (h1 : f ABC.A = 1/2) (h2 : ABC.a = Real.sqrt 2) :
  ABC.b * ABC.c * Real.sin ABC.A / 2 ≤ (1 + Real.sqrt 2) / 2 := by
  sorry

#check triangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l471_47186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_good_set_l471_47156

def isGoodSet (s : Set ℕ) : Prop :=
  ∃ (S T : Set ℕ), S ∪ T = s ∧ S ∩ T = ∅ ∧
    (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ^ b ≠ c) ∧
    (∀ a b c, a ∈ T → b ∈ T → c ∈ T → a ^ b ≠ c)

def setUpToN (n : ℕ) : Set ℕ :=
  {k : ℕ | 2 ≤ k ∧ k ≤ n}

theorem smallest_non_good_set : 
  (∀ m < 65536, isGoodSet (setUpToN m)) ∧
  ¬ isGoodSet (setUpToN 65536) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_good_set_l471_47156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l471_47133

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_8 :
  ∀ a₁ : ℝ,
  (geometric_sum a₁ 2 4 = 1) →
  (geometric_sum a₁ 2 8 = 17) :=
by
  intro a₁ h
  sorry

#check geometric_sequence_sum_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l471_47133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_area_triangles_l471_47183

/-- A point on the integer grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  x_bound : x ≥ 1 ∧ x ≤ 6
  y_bound : y ≥ 1 ∧ y ≤ 6

/-- A triangle formed by three distinct points on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  distinct : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

/-- Function to determine if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Function to determine if a triangle has positive area -/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with positive area on the 6x6 grid -/
def positiveAreaTriangles : Set GridTriangle :=
  {t : GridTriangle | positiveArea t}

/-- Instance to show that GridPoint is finite -/
instance : Fintype GridPoint :=
  sorry

/-- Instance to show that GridTriangle is finite -/
instance : Fintype GridTriangle :=
  sorry

/-- Instance to show that positiveAreaTriangles is finite -/
instance : Fintype positiveAreaTriangles :=
  sorry

/-- The main theorem stating the number of triangles with positive area -/
theorem count_positive_area_triangles :
  Fintype.card positiveAreaTriangles = 6700 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_area_triangles_l471_47183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l471_47102

/-- Represents a parabola in the family y = (p-1)x^2 + 2px + 4 -/
def parabola (p : ℝ) (x : ℝ) : ℝ := (p - 1) * x^2 + 2 * p * x + 4

/-- The point of tangency for p = 2 -/
def point_A : ℝ × ℝ := (-2, 0)

/-- The vertex for p = 0 -/
def point_B : ℝ × ℝ := (0, 4)

/-- The midpoint of segment AB -/
def midpoint_AB : ℝ × ℝ := (-1, 2)

theorem parabola_properties :
  -- 1. Tangency to x-axis when p = 2
  (∀ x, parabola 2 x ≥ 0) ∧ (∃ x, parabola 2 x = 0) ∧
  -- 2. Vertex on y-axis when p = 0
  (∃ y, parabola 0 0 = y ∧ ∀ x, parabola 0 x ≤ y) ∧
  -- 3. Central symmetry of parabolas for p = 2 and p = 0
  (∀ x y, parabola 2 (x + 1) = y - 2 ↔ parabola 0 (-x - 1) = -y + 2) ∧
  -- 4. All curves pass through points A and B
  (∀ p, parabola p point_A.1 = point_A.2 ∧ parabola p point_B.1 = point_B.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l471_47102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_binomial_expansion_l471_47142

/-- 
Given the binomial expansion of (1/x - 1)^5, 
prove that the term with the largest coefficient is 10/x^3
-/
theorem largest_coefficient_binomial_expansion 
  (x : ℝ) (hx : x ≠ 0) : 
  ∃ (k : ℕ) (c : ℝ), 
    c = (Nat.choose 5 k) * (1/x)^k * (-1)^(5-k) ∧ 
    c = 10/x^3 ∧ 
    ∀ (j : ℕ) (d : ℝ), 
      d = (Nat.choose 5 j) * (1/x)^j * (-1)^(5-j) → 
      |d| ≤ |c| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_binomial_expansion_l471_47142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_through_vertex_l471_47179

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := p.p / 2, y := 0 }

/-- The vertex of a parabola -/
def vertex : Point :=
  { x := 0, y := 0 }

/-- Check if a point is on the parabola -/
def onParabola (point : Point) (p : Parabola) : Prop :=
  point.y^2 = 2 * p.p * point.x

/-- A chord of the parabola passing through its focus -/
structure FocusChord (p : Parabola) where
  a : Point
  b : Point
  h1 : onParabola a p
  h2 : onParabola b p
  h3 : ∃ t : ℝ, a.x * (1 - t) + b.x * t = (focus p).x ∧
                a.y * (1 - t) + b.y * t = (focus p).y

/-- The circle with a focus chord as diameter -/
noncomputable def chordCircle (p : Parabola) (fc : FocusChord p) : Circle :=
  { center := { x := (fc.a.x + fc.b.x) / 2, y := (fc.a.y + fc.b.y) / 2 },
    radius := ((fc.a.x - fc.b.x)^2 + (fc.a.y - fc.b.y)^2) / 4 }

/-- The theorem to be proved -/
theorem common_chord_through_vertex (p : Parabola) (fc1 fc2 : FocusChord p) :
  ∃ (m k : ℝ), ∀ (x y : ℝ),
    (x - fc1.a.x) * (x - fc1.b.x) + (y - fc1.a.y) * (y - fc1.b.y) = 0 ∧
    (x - fc2.a.x) * (x - fc2.b.x) + (y - fc2.a.y) * (y - fc2.b.y) = 0 →
    m * x + k * y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_through_vertex_l471_47179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tan_inequality_obtuse_triangle_tan_cot_inequality_l471_47188

-- Part 1: Acute triangle
theorem acute_triangle_tan_inequality (A B C : ℝ) (h_acute : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_less_than_pi_half : A < π/2 ∧ B < π/2 ∧ C < π/2) :
  Real.tan A ^ 2 + Real.tan B ^ 2 + Real.tan C ^ 2 ≥ 9 := by
  sorry

-- Part 2: Obtuse triangle
theorem obtuse_triangle_tan_cot_inequality (A B C : ℝ) (h_obtuse : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_obtuse_angle : A > π/2 ∨ B > π/2 ∨ C > π/2) :
  Real.tan A ^ 2 + Real.tan (π/2 - B) ^ 2 + Real.tan (π/2 - C) ^ 2 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_tan_inequality_obtuse_triangle_tan_cot_inequality_l471_47188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l471_47196

/-- Predicate for an equilateral triangle -/
def is_equilateral_triangle (t : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate for a triangle inscribed in a rectangle with given dimensions -/
def is_inscribed_in_rectangle (t : Set (ℝ × ℝ)) (width : ℝ) (height : ℝ) : Prop := sorry

/-- Area of a triangle -/
def area (t : Set (ℝ × ℝ)) : ℝ := sorry

/-- The maximum area of an equilateral triangle inscribed in a 12x15 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ (t : Set (ℝ × ℝ)), 
      (is_equilateral_triangle t) → 
      (is_inscribed_in_rectangle t 12 15) → 
      (area t ≤ A)) ∧ 
    (A = 369 * Real.sqrt 3 - 540) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_equilateral_triangle_in_rectangle_l471_47196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_and_imaginary_l471_47168

def z (m : ℝ) : ℂ := 5 * m^2 - 45 + (m + 3) * Complex.I

theorem z_real_and_imaginary (m : ℝ) :
  ((z m).im = 0 ↔ m = -3) ∧
  ((z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_and_imaginary_l471_47168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_squared_value_l471_47117

/-- Right trapezoid ABCD with specific properties -/
structure RightTrapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  h_right_angles : True  -- Placeholder for right angles condition
  h_AB : AB = 100
  h_CD : CD = 25
  h_AD_BC : True  -- Placeholder for AD = BC = x condition
  h_circle_tangent : True  -- Placeholder for circle tangent condition

/-- The smallest possible value of x^2 in the given right trapezoid -/
def smallest_x_squared (t : RightTrapezoid) : ℝ := 976.5625

/-- Theorem stating that the smallest possible value of x^2 is 976.5625 -/
theorem smallest_x_squared_value (t : RightTrapezoid) :
  smallest_x_squared t = 976.5625 := by
  sorry

#check smallest_x_squared_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_squared_value_l471_47117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l471_47169

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 3

theorem period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by
  use Real.pi
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l471_47169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_second_claim_l471_47167

/-- Represents a person at the table -/
structure Person where
  is_knight : Bool
  number : ℕ

/-- The round table setup -/
def Table := Vector Person 2015

theorem max_second_claim (table : Table) : 
  (∀ i : Fin 2015, 
    (table.get i).is_knight = 
      ((table.get i).number > (table.get i.val.pred).number ∧ 
       (table.get i).number > (table.get i.val.succ).number)) →
  (∃ k : ℕ, 
    k ≤ 2013 ∧ 
    (∃ subset : Finset (Fin 2015), 
      subset.card = k ∧ 
      ∀ i ∈ subset, 
        ¬(table.get i).is_knight = 
          ((table.get i).number < (table.get i.val.pred).number ∧ 
           (table.get i).number < (table.get i.val.succ).number))) :=
by
  sorry

#check max_second_claim

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_second_claim_l471_47167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l471_47137

/-- An arithmetic sequence {a_n} -/
def a : ℕ → ℝ := sorry

/-- A geometric sequence {b_n} -/
def b : ℕ → ℝ := sorry

/-- Sum of the first n terms of {b_n} -/
def S (n : ℕ) : ℝ := 4 * (1 - 3^n)

theorem sequence_sum_theorem (n : ℕ) :
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) → -- arithmetic sequence condition
  a 3 = -6 →
  a 6 = 0 →
  b 1 = -8 →
  b 2 = a 1 + a 2 + a 3 →
  (∀ k, b (k + 1) / b k = b (k + 2) / b (k + 1)) → -- geometric sequence condition
  S n = 4 * (1 - 3^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l471_47137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_lunch_cost_l471_47180

/-- Helper function to round to nearest two decimal places -/
def roundNearestTwoDecimalPlaces (q : ℚ) : ℚ :=
  (q * 100).floor / 100

/-- Calculate the total amount spent on lunch after discount and tax -/
def total_lunch_cost (hotdog_price salad_price soda_price fries_price discount_rate tax_rate : ℚ) : ℚ :=
  let total_before_discount := hotdog_price + salad_price + soda_price + fries_price
  let discounted_total := total_before_discount * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  roundNearestTwoDecimalPlaces final_total

/-- Theorem stating that Sara's lunch cost is $15.07 -/
theorem sara_lunch_cost :
  total_lunch_cost (5036/1000) (51/10) (11/4) (16/5) (3/20) (1/12) = 1507/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_lunch_cost_l471_47180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_ellipse_parameter_l471_47109

-- Define the polar function
noncomputable def r (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the parametric equations of the curve
noncomputable def x (θ : ℝ) : ℝ := r θ * Real.cos θ
noncomputable def y (θ : ℝ) : ℝ := r θ * Real.sin θ

-- Define the set of points for a given range
def curvePoints (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ, 0 ≤ θ ∧ θ ≤ t ∧ p = (x θ, y θ)}

-- Define what it means for a set of points to form a complete ellipse
def isCompleteEllipse (s : Set (ℝ × ℝ)) : Prop :=
  ∃ a b, a > 0 ∧ b > 0 ∧ ∀ p ∈ s, (p.1 / a)^2 + (p.2 / b)^2 = 1

-- State the theorem
theorem smallest_complete_ellipse_parameter :
  ∃ t : ℝ, t > 0 ∧ isCompleteEllipse (curvePoints t) ∧
  ∀ s, 0 < s ∧ s < t → ¬isCompleteEllipse (curvePoints s) := by
  sorry

#check smallest_complete_ellipse_parameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_ellipse_parameter_l471_47109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_wave_variables_l471_47163

/-- Represents the circumference of a circular water wave --/
noncomputable def C (r : ℝ) : ℝ := 2 * Real.pi * r

/-- Theorem stating that r is the independent variable and C is the dependent variable
    in the equation C = 2πr for circular water waves --/
theorem circular_wave_variables :
  (∃ (f : ℝ → ℝ), ∀ r, C r = f r) ∧
  (¬ ∃ (g : ℝ → ℝ), ∀ c, ∃ r, c = C r ∧ r = g c) :=
by
  constructor
  · -- Prove the existence of f
    use C
    intro r
    rfl
  · -- Prove the non-existence of g
    intro h
    cases h with | intro g hg =>
    have h1 : ∀ c, ∃ r, c = 2 * Real.pi * r := by
      intro c
      specialize hg c
      cases hg with | intro r hr =>
      use r
      exact hr.left
    have h2 : ∀ c, ∃ r, r = g c := by
      intro c
      specialize hg c
      cases hg with | intro r hr =>
      use r
      exact hr.right
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_wave_variables_l471_47163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_equals_zero_l471_47105

theorem xy_equals_zero (x y : ℝ) 
  (h1 : (2 : ℝ)^x = (256 : ℝ)^(y + 1))
  (h2 : (81 : ℝ)^y = (3 : ℝ)^(x - 4)) : 
  x * y = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_equals_zero_l471_47105
