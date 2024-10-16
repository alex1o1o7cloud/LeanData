import Mathlib

namespace NUMINAMATH_CALUDE_root_equation_difference_l1757_175747

theorem root_equation_difference (a b : ℤ) :
  (∃ x : ℝ, x^2 = 7 - 4 * Real.sqrt 3 ∧ x^2 + a * x + b = 0) →
  b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_difference_l1757_175747


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1757_175759

theorem sin_product_equals_one_sixteenth : 
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * 
  Real.sin (54 * π / 180) * Real.sin (72 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1757_175759


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1757_175707

/-- Quadratic equation type -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  m : ℝ

/-- Checks if x is a root of the quadratic equation -/
def is_root (eq : QuadraticEquation) (x : ℝ) : Prop :=
  eq.a * x^2 + eq.b * x + eq.m = 0

/-- Defines the relationship between roots of a quadratic equation -/
def roots_relationship (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 + 5*x₁*x₂ - x₁^2*x₂^2 = 0

theorem quadratic_equation_properties (eq : QuadraticEquation) 
  (h_eq : eq.a = 2 ∧ eq.b = 4) :
  (is_root eq 1 → eq.m = -6 ∧ is_root eq (-3)) ∧
  (∃ x₁ x₂ : ℝ, is_root eq x₁ ∧ is_root eq x₂ ∧ roots_relationship x₁ x₂ → eq.m = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1757_175707


namespace NUMINAMATH_CALUDE_function_positive_implies_a_bound_l1757_175767

/-- Given a function f(x) = x^2 - ax + 2 that is positive for all x > 2,
    prove that a ≤ 3. -/
theorem function_positive_implies_a_bound (a : ℝ) :
  (∀ x > 2, x^2 - a*x + 2 > 0) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_implies_a_bound_l1757_175767


namespace NUMINAMATH_CALUDE_hiring_probability_l1757_175773

theorem hiring_probability (n m k : ℕ) (hn : n = 5) (hm : m = 3) (hk : k = 2) :
  let total_combinations := Nat.choose n m
  let favorable_combinations := total_combinations - Nat.choose (n - k) m
  (favorable_combinations : ℚ) / total_combinations = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_hiring_probability_l1757_175773


namespace NUMINAMATH_CALUDE_unique_solution_circle_equation_l1757_175754

theorem unique_solution_circle_equation :
  ∃! (x y : ℝ), (x - 5)^2 + (y - 6)^2 + (x - y)^2 = 1/3 ∧
  x = 16/3 ∧ y = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_circle_equation_l1757_175754


namespace NUMINAMATH_CALUDE_work_completion_time_l1757_175738

theorem work_completion_time (b_time : ℕ) (joint_time : ℕ) (b_remaining_time : ℕ) :
  b_time = 40 → joint_time = 9 → b_remaining_time = 23 →
  ∃ (a_time : ℕ),
    (joint_time : ℚ) * ((1 : ℚ) / a_time + (1 : ℚ) / b_time) + 
    (b_remaining_time : ℚ) * ((1 : ℚ) / b_time) = 1 ∧
    a_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1757_175738


namespace NUMINAMATH_CALUDE_problem_solution_l1757_175739

theorem problem_solution (x y : ℚ) : 
  x = 103 → x^3 * y - 4 * x^2 * y + 4 * x * y = 1106600 → y = 1085/1030 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1757_175739


namespace NUMINAMATH_CALUDE_rhombus_properties_l1757_175763

/-- Given a rhombus with diagonals of 18 inches and 24 inches, this theorem proves:
    1. The perimeter of the rhombus is 60 inches.
    2. The area of a triangle formed by one side of the rhombus and half of each diagonal is 67.5 square inches. -/
theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 24) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  (4 * s = 60) ∧ ((s * (d1 / 2)) / 2 = 67.5) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_properties_l1757_175763


namespace NUMINAMATH_CALUDE_watermelon_problem_l1757_175701

theorem watermelon_problem (selling_price : ℕ) (total_profit : ℕ) (watermelons_left : ℕ) :
  selling_price = 3 →
  total_profit = 105 →
  watermelons_left = 18 →
  selling_price * ((total_profit / selling_price) + watermelons_left) = 53 * selling_price :=
by sorry

end NUMINAMATH_CALUDE_watermelon_problem_l1757_175701


namespace NUMINAMATH_CALUDE_ordering_of_a_b_c_l1757_175737

theorem ordering_of_a_b_c : 
  let a : ℝ := Real.exp 0.25
  let b : ℝ := 1
  let c : ℝ := -4 * Real.log 0.75
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_a_b_c_l1757_175737


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1757_175732

/-- Given a geometric sequence {a_n} where a_2 and a_6 are roots of x^2 - 34x + 64 = 0, a_4 = 8 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- a is a geometric sequence
  (a 2 * a 2 - 34 * a 2 + 64 = 0) →  -- a_2 is a root of x^2 - 34x + 64 = 0
  (a 6 * a 6 - 34 * a 6 + 64 = 0) →  -- a_6 is a root of x^2 - 34x + 64 = 0
  (a 4 = 8) := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l1757_175732


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l1757_175731

theorem complex_pure_imaginary (a : ℝ) : 
  let z : ℂ := a + 4*I
  (∃ (b : ℝ), (2 - I) * z = b*I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l1757_175731


namespace NUMINAMATH_CALUDE_sixth_term_of_sequence_l1757_175761

/-- Given a sequence {a_n} where a_1 = 1 and a_{n+1} = a_n + 2 for n ≥ 1, prove that a_6 = 11 -/
theorem sixth_term_of_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) : 
  a 6 = 11 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_sequence_l1757_175761


namespace NUMINAMATH_CALUDE_tangent_line_at_one_range_of_m_l1757_175722

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (x - 1) * Real.log x - m * (x + 1)

-- Part 1: Tangent line equation
theorem tangent_line_at_one (m : ℝ) (h : m = 1) :
  ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
  (∀ x y : ℝ, y = f x m → (a * x + b * y + c = 0 ↔ x = 1)) :=
sorry

-- Part 2: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_range_of_m_l1757_175722


namespace NUMINAMATH_CALUDE_three_cones_apex_angle_l1757_175770

/-- Represents a cone with vertex at point A -/
structure Cone where
  apexAngle : ℝ

/-- Represents the configuration of three cones -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  touchingPlane : Bool
  sameSide : Bool

/-- The theorem statement -/
theorem three_cones_apex_angle 
  (config : ConeConfiguration)
  (h1 : config.cone1 = config.cone2)
  (h2 : config.cone3.apexAngle = π / 2)
  (h3 : config.touchingPlane)
  (h4 : config.sameSide) :
  config.cone1.apexAngle = 2 * Real.arctan (4 / 5) := by
  sorry


end NUMINAMATH_CALUDE_three_cones_apex_angle_l1757_175770


namespace NUMINAMATH_CALUDE_min_value_sum_l1757_175751

theorem min_value_sum (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h : u * Real.sqrt (v * w) + v * Real.sqrt (w * u) + w * Real.sqrt (u * v) ≥ 1) :
  u + v + w ≥ Real.sqrt 3 ∧ 
  ∃ (u₀ v₀ w₀ : ℝ), u₀ > 0 ∧ v₀ > 0 ∧ w₀ > 0 ∧
    u₀ * Real.sqrt (v₀ * w₀) + v₀ * Real.sqrt (w₀ * u₀) + w₀ * Real.sqrt (u₀ * v₀) ≥ 1 ∧
    u₀ + v₀ + w₀ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1757_175751


namespace NUMINAMATH_CALUDE_equality_iff_inequality_holds_l1757_175718

theorem equality_iff_inequality_holds (x y : ℝ) : x = y ↔ x * y ≥ ((x + y) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_equality_iff_inequality_holds_l1757_175718


namespace NUMINAMATH_CALUDE_Egypt_India_traditional_l1757_175784

-- Define the types of countries and population growth patterns
inductive CountryType
| Developed
| Developing

inductive GrowthPattern
| Traditional
| Modern

-- Define a function to determine the growth pattern based on country type
def typicalGrowthPattern (ct : CountryType) : GrowthPattern :=
  match ct with
  | CountryType.Developed => GrowthPattern.Modern
  | CountryType.Developing => GrowthPattern.Traditional

-- Define specific countries
def Egypt : CountryType := CountryType.Developing
def India : CountryType := CountryType.Developing

-- China is an exception
def China : CountryType := CountryType.Developing
axiom China_exception : typicalGrowthPattern China = GrowthPattern.Modern

-- Theorem to prove
theorem Egypt_India_traditional :
  typicalGrowthPattern Egypt = GrowthPattern.Traditional ∧
  typicalGrowthPattern India = GrowthPattern.Traditional :=
sorry

end NUMINAMATH_CALUDE_Egypt_India_traditional_l1757_175784


namespace NUMINAMATH_CALUDE_notebook_count_l1757_175785

theorem notebook_count : ∃ (N : ℕ), 
  (∃ (S : ℕ), N = 4 * S + 3) ∧ 
  (∃ (S : ℕ), N + 6 = 5 * S) ∧ 
  N = 39 := by
  sorry

end NUMINAMATH_CALUDE_notebook_count_l1757_175785


namespace NUMINAMATH_CALUDE_divisibility_problem_l1757_175726

def solution_set : Set Int :=
  {-21, -9, -5, -3, -1, 1, 2, 4, 5, 6, 7, 9, 11, 15, 27}

theorem divisibility_problem :
  ∀ x : Int, x ≠ 3 ∧ (x - 3 ∣ x^3 - 3) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1757_175726


namespace NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l1757_175715

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def decimal_2468_7 : Nat := base_to_decimal [8, 6, 4, 2] 7
def decimal_121_5 : Nat := base_to_decimal [1, 2, 1] 5
def decimal_3451_6 : Nat := base_to_decimal [1, 5, 4, 3] 6
def decimal_7891_7 : Nat := base_to_decimal [1, 9, 8, 7] 7

theorem base_conversion_and_arithmetic :
  (decimal_2468_7 / decimal_121_5 : Nat) - decimal_3451_6 + decimal_7891_7 = 2059 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l1757_175715


namespace NUMINAMATH_CALUDE_movie_theater_sections_l1757_175764

theorem movie_theater_sections (total_seats : ℕ) (seats_per_section : ℕ) 
  (h1 : total_seats = 270) 
  (h2 : seats_per_section = 30) 
  (h3 : seats_per_section > 0) :
  total_seats / seats_per_section = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_sections_l1757_175764


namespace NUMINAMATH_CALUDE_no_rotation_matrix_exists_zero_matrix_is_answer_l1757_175774

theorem no_rotation_matrix_exists : ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]] := by
  sorry

theorem zero_matrix_is_answer : 
  (∀ (A : Matrix (Fin 2) (Fin 2) ℝ), (0 : Matrix (Fin 2) (Fin 2) ℝ) * A ≠ ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) ∧
  (¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) :=
by
  sorry

end NUMINAMATH_CALUDE_no_rotation_matrix_exists_zero_matrix_is_answer_l1757_175774


namespace NUMINAMATH_CALUDE_stating_sheets_taken_exists_l1757_175777

/-- Represents the total number of pages in Hiram's algebra notes -/
def total_pages : ℕ := 60

/-- Represents the total number of sheets in Hiram's algebra notes -/
def total_sheets : ℕ := 30

/-- Represents the average of the remaining page numbers -/
def target_average : ℕ := 21

/-- 
Theorem stating that there exists a number of consecutive sheets taken 
such that the average of the remaining page numbers is the target average
-/
theorem sheets_taken_exists : 
  ∃ c : ℕ, c > 0 ∧ c < total_sheets ∧
  ∃ b : ℕ, b ≥ 0 ∧ b + c ≤ total_sheets ∧
  (b * (2 * b + 1) + 
   ((2 * (b + c) + 1 + total_pages) * (total_pages - 2 * c - 2 * b)) / 2) / 
   (total_pages - 2 * c) = target_average :=
sorry

end NUMINAMATH_CALUDE_stating_sheets_taken_exists_l1757_175777


namespace NUMINAMATH_CALUDE_polyhedron_20_faces_l1757_175703

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : Nat
  is_triangular : Bool

/-- The number of edges in a polyhedron -/
def num_edges (p : Polyhedron) : Nat :=
  3 * p.faces / 2

/-- The number of vertices in a polyhedron -/
def num_vertices (p : Polyhedron) : Nat :=
  p.faces + 2 - num_edges p

/-- Theorem: A polyhedron with 20 triangular faces has 30 edges and 12 vertices -/
theorem polyhedron_20_faces (p : Polyhedron) 
  (h1 : p.faces = 20) 
  (h2 : p.is_triangular = true) : 
  num_edges p = 30 ∧ num_vertices p = 12 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_20_faces_l1757_175703


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1757_175771

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 1 < x ∧ x < 3) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1757_175771


namespace NUMINAMATH_CALUDE_tangent_line_to_cubic_l1757_175724

/-- The tangent line to a cubic curve at a specific point -/
theorem tangent_line_to_cubic (a k b : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ 
    y = x^3 + a*x + 1 ∧ 
    y = k*x + b ∧ 
    (3 * x^2 + a) = k) →
  b = -15 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_cubic_l1757_175724


namespace NUMINAMATH_CALUDE_work_completion_days_l1757_175780

/-- Given a number of people P and the number of days D it takes them to complete a work,
    prove that D = 4 when double the number of people can do half the work in 2 days. -/
theorem work_completion_days (P : ℕ) (D : ℕ) (h : P * D = 2 * P * 2 * 2) : D = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l1757_175780


namespace NUMINAMATH_CALUDE_boat_speed_current_l1757_175797

/-- Proves that given a boat with a constant speed of 16 mph relative to water,
    making an upstream trip in 20 minutes and a downstream trip in 15 minutes,
    the speed of the current is 16/7 mph. -/
theorem boat_speed_current (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) :
  boat_speed = 16 ∧ upstream_time = 20 / 60 ∧ downstream_time = 15 / 60 →
  ∃ current_speed : ℝ,
    (boat_speed - current_speed) * upstream_time = (boat_speed + current_speed) * downstream_time ∧
    current_speed = 16 / 7 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_current_l1757_175797


namespace NUMINAMATH_CALUDE_parabola_parameter_l1757_175762

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, if a line through F makes an angle of π/3
    with the x-axis and intersects the parabola at points A and B with |AB| = 8, then p = 3. -/
theorem parabola_parameter (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x) →
  (∃ m b, ∀ x y, y = m*x + b ∧ m = Real.sqrt 3) →
  (∀ x y, y^2 = 2*p*x → (∃ t, x = t ∧ y = Real.sqrt 3 * (t - p/2))) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_l1757_175762


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1757_175705

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/5) : 
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1757_175705


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1757_175714

def alice_number : Nat := 30

-- Function to check if all prime factors of a are prime factors of b
def all_prime_factors_of (a b : Nat) : Prop := 
  ∀ p : Nat, Nat.Prime p → (p ∣ a → p ∣ b)

theorem smallest_bob_number : 
  ∃ bob_number : Nat, 
    (all_prime_factors_of alice_number bob_number) ∧ 
    (all_prime_factors_of bob_number alice_number) ∧ 
    (∀ n : Nat, n < bob_number → 
      ¬(all_prime_factors_of alice_number n ∧ all_prime_factors_of n alice_number)) ∧
    bob_number = alice_number := by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1757_175714


namespace NUMINAMATH_CALUDE_position_after_five_steps_l1757_175704

/-- A student's walk on a number line --/
structure StudentWalk where
  total_steps : ℕ
  total_distance : ℝ
  step_length : ℝ
  marking_distance : ℝ

/-- The position after a certain number of steps --/
def position_after_steps (walk : StudentWalk) (steps : ℕ) : ℝ :=
  walk.step_length * steps

/-- The theorem to prove --/
theorem position_after_five_steps (walk : StudentWalk) 
  (h1 : walk.total_steps = 8)
  (h2 : walk.total_distance = 48)
  (h3 : walk.marking_distance = 3)
  (h4 : walk.step_length = walk.total_distance / walk.total_steps) :
  position_after_steps walk 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_position_after_five_steps_l1757_175704


namespace NUMINAMATH_CALUDE_evaluate_fraction_at_negative_three_l1757_175758

theorem evaluate_fraction_at_negative_three :
  let x : ℚ := -3
  (5 + 2 * x * (x + 5) - 5^2) / (2 * x - 5 + 2 * x^3) = 32 / 65 := by
sorry

end NUMINAMATH_CALUDE_evaluate_fraction_at_negative_three_l1757_175758


namespace NUMINAMATH_CALUDE_bank_deposit_theorem_l1757_175713

def initial_deposit : ℝ := 20000
def term : ℝ := 2
def annual_interest_rate : ℝ := 0.0325

theorem bank_deposit_theorem :
  initial_deposit * (1 + annual_interest_rate * term) = 21300 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_theorem_l1757_175713


namespace NUMINAMATH_CALUDE_system_solution_l1757_175742

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x + y = 5*k ∧ x - 2*y = -k ∧ 2*x - y = 8) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1757_175742


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1757_175706

theorem first_discount_percentage (original_price final_price : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840) : ∃ x : ℝ,
  final_price = original_price * (100 - x) / 100 * 90 / 100 * 95 / 100 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1757_175706


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1757_175721

theorem polynomial_factorization (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1757_175721


namespace NUMINAMATH_CALUDE_combined_polyhedron_faces_l1757_175755

/-- A regular tetrahedron -/
structure Tetrahedron :=
  (edge_length : ℝ)

/-- A regular octahedron -/
structure Octahedron :=
  (edge_length : ℝ)

/-- A polyhedron formed by combining a tetrahedron and an octahedron -/
structure CombinedPolyhedron :=
  (tetra : Tetrahedron)
  (octa : Octahedron)
  (combined : tetra.edge_length = octa.edge_length)

/-- The number of faces in the combined polyhedron -/
def num_faces (p : CombinedPolyhedron) : ℕ := 7

theorem combined_polyhedron_faces (p : CombinedPolyhedron) : 
  num_faces p = 7 := by sorry

end NUMINAMATH_CALUDE_combined_polyhedron_faces_l1757_175755


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1757_175756

theorem polynomial_divisibility (a : ℤ) : ∃ k : ℤ, (3*a + 5)^2 - 4 = k * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1757_175756


namespace NUMINAMATH_CALUDE_parabola_range_l1757_175708

/-- A parabola with the equation y = ax² - 3x + 1 -/
structure Parabola where
  a : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The axis of symmetry for a parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := sorry

/-- Predicate to check if a point is on the parabola -/
def isOnParabola (point : Point) (p : Parabola) : Prop :=
  point.y = p.a * point.x^2 - 3 * point.x + 1

/-- Predicate to check if the parabola intersects a line segment at two distinct points -/
def intersectsTwice (p : Parabola) (p1 p2 : Point) : Prop := sorry

theorem parabola_range (p : Parabola) (A B M N : Point)
    (hA : isOnParabola A p)
    (hB : isOnParabola B p)
    (hM : M.x = -1 ∧ M.y = -2)
    (hN : N.x = 3 ∧ N.y = 2)
    (hAB : ∀ x₀, |A.x - x₀| > |B.x - x₀| → A.y > B.y)
    (hIntersect : intersectsTwice p M N) :
    10/9 ≤ p.a ∧ p.a < 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_range_l1757_175708


namespace NUMINAMATH_CALUDE_no_prime_solution_for_equation_l1757_175741

theorem no_prime_solution_for_equation : 
  ¬∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_for_equation_l1757_175741


namespace NUMINAMATH_CALUDE_fish_count_l1757_175749

theorem fish_count (bass trout bluegill : ℕ) : 
  bass = 32 →
  trout = bass / 4 →
  bluegill = 2 * bass →
  bass + trout + bluegill = 104 := by
sorry

end NUMINAMATH_CALUDE_fish_count_l1757_175749


namespace NUMINAMATH_CALUDE_equal_money_after_11_weeks_l1757_175735

/-- Carol's initial amount in dollars -/
def carol_initial : ℕ := 40

/-- Carol's weekly savings in dollars -/
def carol_savings : ℕ := 12

/-- Mike's initial amount in dollars -/
def mike_initial : ℕ := 150

/-- Mike's weekly savings in dollars -/
def mike_savings : ℕ := 2

/-- The number of weeks it takes for Carol and Mike to have the same amount of money -/
def weeks_to_equal_money : ℕ := 11

theorem equal_money_after_11_weeks :
  carol_initial + carol_savings * weeks_to_equal_money =
  mike_initial + mike_savings * weeks_to_equal_money :=
by sorry

end NUMINAMATH_CALUDE_equal_money_after_11_weeks_l1757_175735


namespace NUMINAMATH_CALUDE_springdale_rainfall_l1757_175723

theorem springdale_rainfall (first_week : ℝ) (second_week : ℝ) : 
  second_week = 1.5 * first_week →
  second_week = 24 →
  first_week + second_week = 40 := by
sorry

end NUMINAMATH_CALUDE_springdale_rainfall_l1757_175723


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1757_175710

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n}, if a_1 + a_19 = 10, then a_10 = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + a 19 = 10) : 
  a 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1757_175710


namespace NUMINAMATH_CALUDE_anna_candy_purchase_l1757_175779

def candy_problem (initial_money : ℚ) (gum_price : ℚ) (gum_quantity : ℕ) 
  (chocolate_price : ℚ) (cane_price : ℚ) (cane_quantity : ℕ) (money_left : ℚ) : Prop :=
  ∃ (chocolate_quantity : ℕ),
    initial_money - 
    (gum_price * gum_quantity + 
     chocolate_price * chocolate_quantity + 
     cane_price * cane_quantity) = money_left ∧
    chocolate_quantity = 5

theorem anna_candy_purchase : 
  candy_problem 10 1 3 1 0.5 2 1 := by sorry

end NUMINAMATH_CALUDE_anna_candy_purchase_l1757_175779


namespace NUMINAMATH_CALUDE_pyramid_section_volume_l1757_175791

/-- Given a pyramid with base area 3 and volume 3, and two parallel cross-sections with areas 1 and 2,
    the volume of the part of the pyramid between these cross-sections is (2√6 - √3) / 3. -/
theorem pyramid_section_volume 
  (base_area : ℝ) 
  (pyramid_volume : ℝ) 
  (section_area_1 : ℝ) 
  (section_area_2 : ℝ) 
  (h_base_area : base_area = 3) 
  (h_pyramid_volume : pyramid_volume = 3) 
  (h_section_area_1 : section_area_1 = 1) 
  (h_section_area_2 : section_area_2 = 2) : 
  ∃ (section_volume : ℝ), section_volume = (2 * Real.sqrt 6 - Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_section_volume_l1757_175791


namespace NUMINAMATH_CALUDE_fraction_transformation_l1757_175727

theorem fraction_transformation (a b : ℝ) (h : a * b > 0) :
  (3 * a + 2 * (3 * b)) / (2 * (3 * a) * (3 * b)) = (1 / 3) * ((a + 2 * b) / (2 * a * b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1757_175727


namespace NUMINAMATH_CALUDE_kylies_daisies_l1757_175766

/-- Proves that Kylie's initial number of daisies is 5 given the problem conditions -/
theorem kylies_daisies (initial : ℕ) (sister_gift : ℕ) (remaining : ℕ) : 
  sister_gift = 9 → 
  remaining = 7 → 
  (initial + sister_gift) / 2 = remaining → 
  initial = 5 := by
sorry

end NUMINAMATH_CALUDE_kylies_daisies_l1757_175766


namespace NUMINAMATH_CALUDE_final_number_is_100_l1757_175728

def board_numbers : List ℚ := List.map (λ i => 1 / i) (List.range 100)

def combine (a b : ℚ) : ℚ := a * b + a + b

theorem final_number_is_100 (numbers : List ℚ) (h : numbers = board_numbers) :
  (numbers.foldl combine 0 : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_100_l1757_175728


namespace NUMINAMATH_CALUDE_prism_with_12_edges_has_quadrilateral_base_l1757_175792

/-- A prism with n sides in its base has 3n edges. -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- The number of sides in the base of a prism with 12 edges. -/
def base_sides : ℕ := 4

/-- Theorem: A prism with 12 edges has a quadrilateral base. -/
theorem prism_with_12_edges_has_quadrilateral_base :
  prism_edges base_sides = 12 :=
sorry

end NUMINAMATH_CALUDE_prism_with_12_edges_has_quadrilateral_base_l1757_175792


namespace NUMINAMATH_CALUDE_gcd_45885_30515_l1757_175743

theorem gcd_45885_30515 : Nat.gcd 45885 30515 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45885_30515_l1757_175743


namespace NUMINAMATH_CALUDE_projection_vector_is_correct_l1757_175783

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line l -/
def line_l : ParametricLine :=
  { x := λ t => 2 + 3*t,
    y := λ t => 3 + 2*t }

/-- The line m -/
def line_m : ParametricLine :=
  { x := λ s => 4 + 2*s,
    y := λ s => 5 + 3*s }

/-- Direction vector of line l -/
def dir_l : Vector2D :=
  { x := 3,
    y := 2 }

/-- Direction vector of line m -/
def dir_m : Vector2D :=
  { x := 2,
    y := 3 }

/-- The vector perpendicular to the direction of line m -/
def perp_m : Vector2D :=
  { x := 3,
    y := -2 }

/-- The theorem to prove -/
theorem projection_vector_is_correct :
  ∃ (k : ℝ),
    let v : Vector2D := { x := k * perp_m.x, y := k * perp_m.y }
    v.x + v.y = 3 ∧
    v.x = 9 ∧
    v.y = -6 := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_is_correct_l1757_175783


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l1757_175769

theorem no_solution_to_equation :
  ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^2 - 15 * x) / (x^2 - 5 * x) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l1757_175769


namespace NUMINAMATH_CALUDE_worker_idle_days_l1757_175746

theorem worker_idle_days 
  (total_days : ℕ) 
  (pay_per_work_day : ℕ) 
  (deduction_per_idle_day : ℕ) 
  (total_payment : ℕ) 
  (h1 : total_days = 60)
  (h2 : pay_per_work_day = 20)
  (h3 : deduction_per_idle_day = 3)
  (h4 : total_payment = 280) :
  ∃ (idle_days : ℕ) (work_days : ℕ),
    idle_days + work_days = total_days ∧
    pay_per_work_day * work_days - deduction_per_idle_day * idle_days = total_payment ∧
    idle_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_worker_idle_days_l1757_175746


namespace NUMINAMATH_CALUDE_roses_apple_sharing_l1757_175768

/-- Given that Rose has 9 apples and each friend receives 3 apples,
    prove that the number of friends Rose shares her apples with is 3. -/
theorem roses_apple_sharing :
  let total_apples : ℕ := 9
  let apples_per_friend : ℕ := 3
  total_apples / apples_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_roses_apple_sharing_l1757_175768


namespace NUMINAMATH_CALUDE_adam_earnings_l1757_175752

def lawn_pay : ℝ := 9
def car_pay : ℝ := 15
def dog_pay : ℝ := 5

def total_lawns : ℕ := 12
def total_cars : ℕ := 6
def total_dogs : ℕ := 4

def forgotten_lawns : ℕ := 8
def forgotten_cars : ℕ := 2
def forgotten_dogs : ℕ := 1

def bonus_rate : ℝ := 0.1

def completed_task_types : ℕ := 3

theorem adam_earnings : 
  let completed_lawns := total_lawns - forgotten_lawns
  let completed_cars := total_cars - forgotten_cars
  let completed_dogs := total_dogs - forgotten_dogs
  let base_earnings := completed_lawns * lawn_pay + completed_cars * car_pay + completed_dogs * dog_pay
  let bonus := base_earnings * bonus_rate * completed_task_types
  let total_earnings := base_earnings + bonus
  total_earnings = 122.1 := by sorry

end NUMINAMATH_CALUDE_adam_earnings_l1757_175752


namespace NUMINAMATH_CALUDE_line_increase_l1757_175794

/-- Given a line where an increase of 4 units in x results in an increase of 6 units in y,
    prove that an increase of 12 units in x results in an increase of 18 units in y. -/
theorem line_increase (f : ℝ → ℝ) (x : ℝ) :
  (f (x + 4) - f x = 6) → (f (x + 12) - f x = 18) := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l1757_175794


namespace NUMINAMATH_CALUDE_simplify_expression_l1757_175790

theorem simplify_expression (x y : ℝ) : 3*y + 5*y + 6*y + 2*x + 4*x = 14*y + 6*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1757_175790


namespace NUMINAMATH_CALUDE_existence_of_multiple_representations_l1757_175793

def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

theorem existence_of_multiple_representations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ a b c d : ℕ,
      indecomposable n a ∧
      indecomposable n b ∧
      indecomposable n c ∧
      indecomposable n d ∧
      r = a * b ∧
      r = c * d ∧
      (a ≠ c ∨ b ≠ d) ∧
      (a ≠ d ∨ b ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_multiple_representations_l1757_175793


namespace NUMINAMATH_CALUDE_curve_crosses_itself_l1757_175744

/-- The x-coordinate of a point on the curve -/
def x (t k : ℝ) : ℝ := t^2 + k

/-- The y-coordinate of a point on the curve -/
def y (t k : ℝ) : ℝ := t^3 - k*t + 5

/-- Theorem stating that the curve crosses itself at (18,5) when k = 9 -/
theorem curve_crosses_itself : 
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    x t₁ 9 = x t₂ 9 ∧ 
    y t₁ 9 = y t₂ 9 ∧
    x t₁ 9 = 18 ∧ 
    y t₁ 9 = 5 :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_itself_l1757_175744


namespace NUMINAMATH_CALUDE_production_time_calculation_l1757_175719

/-- The number of days it takes for a given number of machines to produce a certain amount of product P -/
def production_time (num_machines : ℕ) (units : ℝ) : ℝ := sorry

/-- The number of units produced by a given number of machines in a certain number of days -/
def units_produced (num_machines : ℕ) (days : ℝ) : ℝ := sorry

theorem production_time_calculation :
  let d := production_time 5 x
  let x : ℝ := units_produced 5 d
  units_produced 20 2 = 2 * x →
  d = 4 := by sorry

end NUMINAMATH_CALUDE_production_time_calculation_l1757_175719


namespace NUMINAMATH_CALUDE_fraction_problem_l1757_175772

theorem fraction_problem (f : ℚ) : f * 20 + 7 = 17 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1757_175772


namespace NUMINAMATH_CALUDE_line_direction_vector_k_l1757_175720

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  let dir := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ (t : ℝ), dir = (3 * t, k * t)

/-- The main theorem stating that k = -3 for the given line -/
theorem line_direction_vector_k (k : ℝ) : 
  Line (2, -1) (-4, 5) k → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_k_l1757_175720


namespace NUMINAMATH_CALUDE_cricket_game_initial_overs_l1757_175789

/-- Proves that the number of overs played initially is 10 in a cricket game scenario -/
theorem cricket_game_initial_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 282) (h2 : initial_rate = 3.2) 
  (h3 : required_rate = 6.25) (h4 : remaining_overs = 40) : 
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  initial_rate * initial_overs + required_rate * remaining_overs = target :=
by sorry

end NUMINAMATH_CALUDE_cricket_game_initial_overs_l1757_175789


namespace NUMINAMATH_CALUDE_lisa_window_width_l1757_175730

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions and layout of a window -/
structure Window where
  pane : Pane
  rows : ℕ
  columns : ℕ
  border_width : ℝ

/-- Calculates the total width of the window -/
def window_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- The main theorem stating the width of Lisa's window -/
theorem lisa_window_width :
  ∃ (w : Window),
    w.rows = 3 ∧
    w.columns = 4 ∧
    w.border_width = 3 ∧
    w.pane.width / w.pane.height = 3 / 4 ∧
    window_width w = 51 :=
sorry

end NUMINAMATH_CALUDE_lisa_window_width_l1757_175730


namespace NUMINAMATH_CALUDE_game_theorists_board_size_l1757_175734

/-- Represents the voting process for the game theorists' leadership board. -/
def BoardVotingProcess (initial_members : ℕ) : Prop :=
  ∃ (final_members : ℕ),
    -- The final number of members is less than or equal to the initial number
    final_members ≤ initial_members ∧
    -- The final number of members is of the form 2^n - 1
    ∃ (n : ℕ), final_members = 2^n - 1 ∧
    -- There is no larger number of the form 2^m - 1 that's less than or equal to the initial number
    ∀ (m : ℕ), 2^m - 1 ≤ initial_members → m ≤ n

/-- The theorem stating the result of the voting process for 2020 initial members. -/
theorem game_theorists_board_size :
  BoardVotingProcess 2020 → ∃ (final_members : ℕ), final_members = 1023 :=
by
  sorry


end NUMINAMATH_CALUDE_game_theorists_board_size_l1757_175734


namespace NUMINAMATH_CALUDE_quadrilateral_JMIT_cyclic_l1757_175711

structure Triangle (α : Type*) [Field α] where
  a : α
  b : α
  c : α

def incenter {α : Type*} [Field α] (t : Triangle α) : α :=
  -(t.a * t.b + t.b * t.c + t.c * t.a)

def excenter {α : Type*} [Field α] (t : Triangle α) : α :=
  t.a * t.b - t.b * t.c + t.c * t.a

def midpoint_BC {α : Type*} [Field α] (t : Triangle α) : α :=
  (t.b^2 + t.c^2) / 2

def symmetric_point {α : Type*} [Field α] (t : Triangle α) : α :=
  2 * t.a^2 - t.b * t.c

def is_cyclic {α : Type*} [Field α] (a b c d : α) : Prop :=
  ∃ (k : α), k ≠ 0 ∧ (b - a) * (d - c) = k * (c - a) * (d - b)

theorem quadrilateral_JMIT_cyclic {α : Type*} [Field α] (t : Triangle α) 
  (h1 : t.a^2 ≠ 0) (h2 : t.b^2 ≠ 0) (h3 : t.c^2 ≠ 0)
  (h4 : t.a^2 * t.a^2 = 1) (h5 : t.b^2 * t.b^2 = 1) (h6 : t.c^2 * t.c^2 = 1) :
  is_cyclic (excenter t) (midpoint_BC t) (incenter t) (symmetric_point t) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_JMIT_cyclic_l1757_175711


namespace NUMINAMATH_CALUDE_isbn_check_digit_l1757_175712

/-- Calculates the sum S for an ISBN --/
def calculate_S (A B C D E F G H I : ℕ) : ℕ :=
  10 * A + 9 * B + 8 * C + 7 * D + 6 * E + 5 * F + 4 * G + 3 * H + 2 * I

/-- Determines the check digit J based on the remainder r --/
def determine_J (r : ℕ) : ℕ :=
  if r = 0 then 0
  else if r = 1 then 10  -- Representing 'x' as 10
  else 11 - r

/-- Theorem: For the ISBN 962y707015, y = 7 --/
theorem isbn_check_digit (y : ℕ) (hy : y < 10) :
  let S := calculate_S 9 6 2 y 7 0 7 0 1
  let r := S % 11
  determine_J r = 5 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_isbn_check_digit_l1757_175712


namespace NUMINAMATH_CALUDE_binomial_product_l1757_175798

variable (x : ℝ)

theorem binomial_product :
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l1757_175798


namespace NUMINAMATH_CALUDE_find_divisor_l1757_175778

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 56 → quotient = 4 → divisor * quotient = dividend → divisor = 14 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1757_175778


namespace NUMINAMATH_CALUDE_min_triangles_is_eighteen_l1757_175787

/-- Represents a non-convex hexagon formed by removing one corner square from an 8x8 chessboard -/
structure ChessboardHexagon where
  area : ℝ
  side_length : ℝ

/-- Calculates the minimum number of congruent triangles needed to partition the ChessboardHexagon -/
def min_congruent_triangles (h : ChessboardHexagon) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of congruent triangles is 18 -/
theorem min_triangles_is_eighteen (h : ChessboardHexagon) 
  (h_area : h.area = 63)
  (h_side : h.side_length = 8) : 
  min_congruent_triangles h = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_is_eighteen_l1757_175787


namespace NUMINAMATH_CALUDE_team_formation_count_l1757_175740

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of female teachers -/
def num_female : ℕ := 4

/-- The number of male teachers -/
def num_male : ℕ := 5

/-- The total number of teachers to be selected -/
def team_size : ℕ := 3

theorem team_formation_count : 
  choose num_female 1 * choose num_male 2 + choose num_female 2 * choose num_male 1 = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_count_l1757_175740


namespace NUMINAMATH_CALUDE_expected_value_decahedral_die_l1757_175702

/-- A fair decahedral die with faces numbered 1 to 10 -/
def DecahedralDie : Finset ℕ := Finset.range 10

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 10

/-- The expected value of rolling the decahedral die -/
def expected_value : ℚ := (DecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair decahedral die with faces numbered 1 to 10 is 5.5 -/
theorem expected_value_decahedral_die : expected_value = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_decahedral_die_l1757_175702


namespace NUMINAMATH_CALUDE_polynomial_derivative_bound_l1757_175717

theorem polynomial_derivative_bound (p : ℝ → ℝ) :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → |p x| ≤ 1) →
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → |(deriv p) x| ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_bound_l1757_175717


namespace NUMINAMATH_CALUDE_four_students_three_events_sign_up_l1757_175709

/-- The number of ways for students to sign up for events -/
def num_ways_to_sign_up (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_events ^ num_students

/-- Theorem: Four students choosing from three events results in 81 possible arrangements -/
theorem four_students_three_events_sign_up :
  num_ways_to_sign_up 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_events_sign_up_l1757_175709


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1757_175736

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 132 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1757_175736


namespace NUMINAMATH_CALUDE_sequence_remainder_l1757_175782

theorem sequence_remainder (n : ℕ) : (7 * n + 4) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_remainder_l1757_175782


namespace NUMINAMATH_CALUDE_angle_C_value_l1757_175760

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleC : ℝ

-- State the theorem
theorem angle_C_value (t : Triangle) 
  (h : t.a^2 + t.b^2 - t.c^2 + t.a * t.b = 0) : 
  t.angleC = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l1757_175760


namespace NUMINAMATH_CALUDE_complex_equal_parts_l1757_175795

theorem complex_equal_parts (a : ℝ) : 
  (Complex.re ((1 + a * Complex.I) * (2 + Complex.I)) = 
   Complex.im ((1 + a * Complex.I) * (2 + Complex.I))) → 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l1757_175795


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l1757_175781

theorem cupcakes_per_package 
  (initial_cupcakes : ℕ) 
  (eaten_cupcakes : ℕ) 
  (total_packages : ℕ) 
  (h1 : initial_cupcakes = 39) 
  (h2 : eaten_cupcakes = 21) 
  (h3 : total_packages = 6) 
  (h4 : eaten_cupcakes < initial_cupcakes) : 
  (initial_cupcakes - eaten_cupcakes) / total_packages = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l1757_175781


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l1757_175750

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 2 * x + y → x + y ≥ 2 * Real.sqrt 2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l1757_175750


namespace NUMINAMATH_CALUDE_region_perimeter_l1757_175725

theorem region_perimeter (total_area : ℝ) (num_squares : ℕ) (row1_squares row2_squares : ℕ) :
  total_area = 400 →
  num_squares = 8 →
  row1_squares = 3 →
  row2_squares = 5 →
  row1_squares + row2_squares = num_squares →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := (2 * (row1_squares + row2_squares) + 2) * side_length
  perimeter = 90 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_region_perimeter_l1757_175725


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l1757_175745

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = x^2 + ax + b -/
def g (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The condition of symmetry about x = 1 -/
def symmetry_condition (a b : ℝ) : Prop :=
  ∀ x, g a b x = f (2 - x)

/-- Theorem: If f and g are symmetrical about x = 1, then a = -4 and b = 4 -/
theorem symmetry_implies_values :
  ∀ a b, symmetry_condition a b → a = -4 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l1757_175745


namespace NUMINAMATH_CALUDE_mixed_fraction_power_product_l1757_175799

theorem mixed_fraction_power_product :
  (1 + 2/3)^4 * (-3/5)^5 = -3/5 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_power_product_l1757_175799


namespace NUMINAMATH_CALUDE_max_passable_levels_l1757_175775

/-- Represents the maximum number of points obtainable from a single dice throw -/
def max_dice_points : ℕ := 6

/-- Represents the pass condition for a level in the "pass-through game" -/
def pass_condition (n : ℕ) : ℕ := 2^n

/-- Represents the maximum sum of points obtainable from n dice throws -/
def max_sum_points (n : ℕ) : ℕ := n * max_dice_points

/-- Theorem stating the maximum number of levels that can be passed in the "pass-through game" -/
theorem max_passable_levels : 
  ∃ (max_level : ℕ), 
    (∀ n : ℕ, n ≤ max_level → max_sum_points n > pass_condition n) ∧ 
    (∀ n : ℕ, n > max_level → max_sum_points n ≤ pass_condition n) :=
sorry

end NUMINAMATH_CALUDE_max_passable_levels_l1757_175775


namespace NUMINAMATH_CALUDE_f_pi_plus_3_l1757_175788

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem f_pi_plus_3 (a b : ℝ) :
  f a b (-3) = 5 → f a b (Real.pi + 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_plus_3_l1757_175788


namespace NUMINAMATH_CALUDE_shelf_capacity_l1757_175796

/-- The number of CDs each rack can hold -/
def cds_per_rack : ℕ := 8

/-- The total number of CDs the shelf can hold -/
def total_cds : ℕ := 32

/-- The number of racks the shelf can hold -/
def num_racks : ℕ := total_cds / cds_per_rack

theorem shelf_capacity : num_racks = 4 := by
  sorry

end NUMINAMATH_CALUDE_shelf_capacity_l1757_175796


namespace NUMINAMATH_CALUDE_blue_zone_points_l1757_175765

-- Define the target structure
structure Target where
  r : ℝ  -- radius of bullseye
  rings : Fin 4 → ℝ  -- radii of the rings

-- Define the point system
def points (t : Target) (zone : Fin 5) : ℝ :=
  sorry

-- Properties of the target
axiom ring_width (t : Target) (i : Fin 4) : t.rings i = (i.val + 1 : ℝ) * t.r

-- Inverse proportionality of points to hit probability
axiom inverse_proportionality (t : Target) (zone1 zone2 : Fin 5) :
  points t zone1 * (2 * (zone1.val + 1 : ℝ) * t.r - (2 * zone1.val + 1 : ℝ) * t.r) =
  points t zone2 * (2 * (zone2.val + 1 : ℝ) * t.r - (2 * zone2.val + 1 : ℝ) * t.r)

-- Bullseye points
axiom bullseye_points (t : Target) : points t 0 = 315

-- Theorem to prove
theorem blue_zone_points (t : Target) : points t 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_blue_zone_points_l1757_175765


namespace NUMINAMATH_CALUDE_f_range_f_period_one_l1757_175776

-- Define the nearest integer function
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if x - ⌊x⌋ ≤ 1/2 then ⌊x⌋ else ⌈x⌉

-- Define the function f(x) = x - {x}
noncomputable def f (x : ℝ) : ℝ := x - nearest_integer x

-- Theorem stating the range of f(x)
theorem f_range : Set.range f = Set.Ioc (-1/2) (1/2) := by sorry

-- Theorem stating that f(x) has a period of 1
theorem f_period_one (x : ℝ) : f (x + 1) = f x := by sorry

end NUMINAMATH_CALUDE_f_range_f_period_one_l1757_175776


namespace NUMINAMATH_CALUDE_problem_statement_l1757_175733

theorem problem_statement (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x^2 - 4*1*x + 3*1^2 < 0 ∧ (x - 3) / (x + 2) < 0 ↔ 1 < x ∧ x < 3) ∧
  ({a : ℝ | a > 0 ∧
    (∀ x : ℝ, (x - 3) / (x + 2) ≥ 0 → x^2 - 4*a*x + 3*a^2 ≥ 0) ∧
    (∃ x : ℝ, (x - 3) / (x + 2) < 0 ∧ x^2 - 4*a*x + 3*a^2 < 0)} =
   {a : ℝ | 0 < a ∧ a ≤ 1}) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1757_175733


namespace NUMINAMATH_CALUDE_set_expressions_correct_l1757_175716

def solution_set : Set ℝ := {x | x^2 - 4 = 0}
def prime_set : Set ℕ := {p | Nat.Prime p ∧ 0 < 2 * p ∧ 2 * p < 18}
def even_set : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def fourth_quadrant : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 < 0}

theorem set_expressions_correct :
  solution_set = {-2, 2} ∧
  prime_set = {2, 3, 5, 7} ∧
  even_set = {x | ∃ n : ℤ, x = 2 * n} ∧
  fourth_quadrant = {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0} :=
by sorry

end NUMINAMATH_CALUDE_set_expressions_correct_l1757_175716


namespace NUMINAMATH_CALUDE_min_bing_toys_l1757_175748

/-- Represents the cost and pricing of Olympic mascot toys --/
structure OlympicToys where
  bing_cost : ℕ  -- Cost of Bing Dwen Dwen
  shuey_cost : ℕ  -- Cost of Shuey Rongrong
  bing_price : ℕ  -- Selling price of Bing Dwen Dwen
  shuey_price : ℕ  -- Selling price of Shuey Rongrong

/-- Theorem about the minimum number of Bing Dwen Dwen toys to purchase --/
theorem min_bing_toys (t : OlympicToys) 
  (h1 : 4 * t.bing_cost + 5 * t.shuey_cost = 1000)
  (h2 : 5 * t.bing_cost + 10 * t.shuey_cost = 1550)
  (h3 : t.bing_price = 180)
  (h4 : t.shuey_price = 100)
  (h5 : ∀ x : ℕ, x + (180 - x) = 180)
  (h6 : ∀ x : ℕ, x * (t.bing_price - t.bing_cost) + (180 - x) * (t.shuey_price - t.shuey_cost) ≥ 4600) :
  ∃ (min_bing : ℕ), min_bing = 100 ∧ 
    ∀ (x : ℕ), x ≥ min_bing → 
      x * (t.bing_price - t.bing_cost) + (180 - x) * (t.shuey_price - t.shuey_cost) ≥ 4600 :=
sorry

end NUMINAMATH_CALUDE_min_bing_toys_l1757_175748


namespace NUMINAMATH_CALUDE_sqrt_solution_l1757_175757

theorem sqrt_solution (x : ℝ) (h : x > 0) : 
  let y : ℝ → ℝ := λ x => Real.sqrt x
  2 * y x * (deriv y x) = 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_solution_l1757_175757


namespace NUMINAMATH_CALUDE_smallest_fraction_l1757_175786

theorem smallest_fraction (x : ℝ) (h : x = 9) : 
  min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min (x/8) (x^2/64)))) = 8/(x+2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l1757_175786


namespace NUMINAMATH_CALUDE_fraction_problem_l1757_175700

theorem fraction_problem (N : ℝ) (x : ℝ) : 
  (0.40 * N = 420) → 
  (x * (1/3) * (2/5) * N = 35) → 
  x = 1/4 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1757_175700


namespace NUMINAMATH_CALUDE_equation_solutions_l1757_175729

def equation (x : ℝ) : Prop :=
  x ≠ 2/3 ∧ x ≠ -3 ∧ (8*x + 3) / (3*x^2 + 8*x - 6) = 3*x / (3*x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1757_175729


namespace NUMINAMATH_CALUDE_smallest_base_sum_l1757_175753

theorem smallest_base_sum : ∃ (c d : ℕ), 
  c ≠ d ∧ 
  c > 9 ∧ 
  d > 9 ∧ 
  8 * c + 9 = 9 * d + 8 ∧ 
  c + d = 19 ∧ 
  (∀ (c' d' : ℕ), c' ≠ d' → c' > 9 → d' > 9 → 8 * c' + 9 = 9 * d' + 8 → c' + d' ≥ 19) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_sum_l1757_175753
