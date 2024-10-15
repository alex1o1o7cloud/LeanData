import Mathlib

namespace NUMINAMATH_CALUDE_min_cans_is_281_l3704_370403

/-- The number of liters of Maaza --/
def maaza : ℕ := 50

/-- The number of liters of Pepsi --/
def pepsi : ℕ := 144

/-- The number of liters of Sprite --/
def sprite : ℕ := 368

/-- The function to calculate the minimum number of cans required --/
def min_cans (m p s : ℕ) : ℕ :=
  (m / Nat.gcd m (Nat.gcd p s)) + (p / Nat.gcd m (Nat.gcd p s)) + (s / Nat.gcd m (Nat.gcd p s))

/-- Theorem stating that the minimum number of cans required is 281 --/
theorem min_cans_is_281 : min_cans maaza pepsi sprite = 281 := by
  sorry

end NUMINAMATH_CALUDE_min_cans_is_281_l3704_370403


namespace NUMINAMATH_CALUDE_cost_price_is_118_l3704_370451

/-- Calculates the cost price per meter of cloth -/
def cost_price_per_meter (total_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_length) / total_length

/-- Theorem: The cost price of one meter of cloth is 118 Rs -/
theorem cost_price_is_118 (total_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
    (h1 : total_length = 80)
    (h2 : selling_price = 10000)
    (h3 : profit_per_meter = 7) :
  cost_price_per_meter total_length selling_price profit_per_meter = 118 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_118_l3704_370451


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l3704_370458

theorem closest_integer_to_cube_root_200 : 
  ∀ n : ℤ, |n^3 - 200| ≥ |6^3 - 200| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l3704_370458


namespace NUMINAMATH_CALUDE_expression_evaluation_l3704_370461

theorem expression_evaluation : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3704_370461


namespace NUMINAMATH_CALUDE_no_nonnegative_solutions_quadratic_l3704_370452

theorem no_nonnegative_solutions_quadratic :
  ∀ x : ℝ, x ≥ 0 → x^2 + 6*x + 9 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_nonnegative_solutions_quadratic_l3704_370452


namespace NUMINAMATH_CALUDE_passing_percentage_problem_l3704_370409

/-- The passing percentage problem -/
theorem passing_percentage_problem (mike_score : ℕ) (shortfall : ℕ) (max_marks : ℕ) 
  (h1 : mike_score = 212)
  (h2 : shortfall = 25)
  (h3 : max_marks = 790) :
  let passing_marks : ℕ := mike_score + shortfall
  let passing_percentage : ℚ := (passing_marks : ℚ) / max_marks * 100
  ∃ ε > 0, abs (passing_percentage - 30) < ε := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_problem_l3704_370409


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3704_370496

theorem sqrt_equation_solutions :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 - x) + Real.sqrt x
  ∀ x : ℝ, f x = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3704_370496


namespace NUMINAMATH_CALUDE_rectangular_park_diagonal_l3704_370477

theorem rectangular_park_diagonal (x y : ℝ) (h_positive : x > 0 ∧ y > 0) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_park_diagonal_l3704_370477


namespace NUMINAMATH_CALUDE_largest_n_is_max_factorization_exists_l3704_370478

/-- The largest value of n for which 4x^2 + nx + 96 can be factored as two linear factors with integer coefficients -/
def largest_n : ℕ := 385

/-- A structure representing the factorization of 4x^2 + nx + 96 -/
structure Factorization where
  a : ℤ
  b : ℤ
  h1 : (4 * X + a) * (X + b) = 4 * X^2 + largest_n * X + 96

/-- Theorem stating that largest_n is indeed the largest value for which the factorization exists -/
theorem largest_n_is_max :
  ∀ n : ℕ, n > largest_n →
    ¬∃ (f : Factorization), (4 * X + f.a) * (X + f.b) = 4 * X^2 + n * X + 96 :=
by sorry

/-- Theorem stating that a factorization exists for largest_n -/
theorem factorization_exists : ∃ (f : Factorization), True :=
by sorry

end NUMINAMATH_CALUDE_largest_n_is_max_factorization_exists_l3704_370478


namespace NUMINAMATH_CALUDE_tesseract_sum_l3704_370471

/-- A tesseract is a 4-dimensional hypercube -/
structure Tesseract where

/-- The number of edges in a tesseract -/
def Tesseract.edges (t : Tesseract) : ℕ := 32

/-- The number of vertices in a tesseract -/
def Tesseract.vertices (t : Tesseract) : ℕ := 16

/-- The number of faces in a tesseract -/
def Tesseract.faces (t : Tesseract) : ℕ := 24

/-- The sum of edges, vertices, and faces in a tesseract is 72 -/
theorem tesseract_sum (t : Tesseract) : 
  t.edges + t.vertices + t.faces = 72 := by sorry

end NUMINAMATH_CALUDE_tesseract_sum_l3704_370471


namespace NUMINAMATH_CALUDE_greatest_number_with_conditions_l3704_370485

theorem greatest_number_with_conditions : ∃ n : ℕ, 
  n < 150 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  n % 3 = 0 ∧
  ∀ m : ℕ, m < 150 → (∃ k : ℕ, m = k^2) → m % 3 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_conditions_l3704_370485


namespace NUMINAMATH_CALUDE_edges_ge_twice_faces_l3704_370467

/-- A bipartite planar graph. -/
structure BipartitePlanarGraph where
  V : Type* -- Vertices
  E : Type* -- Edges
  F : Type* -- Faces
  edge_count : ℕ
  face_count : ℕ
  is_bipartite : Prop
  is_planar : Prop
  edge_count_ge_two : edge_count ≥ 2

/-- Theorem: In a bipartite planar graph with at least 2 edges, 
    the number of edges is at least twice the number of faces. -/
theorem edges_ge_twice_faces (G : BipartitePlanarGraph) : 
  G.edge_count ≥ 2 * G.face_count := by
  sorry

end NUMINAMATH_CALUDE_edges_ge_twice_faces_l3704_370467


namespace NUMINAMATH_CALUDE_economy_relationship_l3704_370499

/-- Given an economy with product X, price P, and total cost C, prove the relationship
    between these variables and calculate specific values. -/
theorem economy_relationship (k k' : ℝ) : 
  (∀ (X P : ℝ), X * P = k) →  -- X is inversely proportional to P
  (200 : ℝ) * 10 = k →        -- When P = 10, X = 200
  (∀ (C X : ℝ), C = k' * X) → -- C is directly proportional to X
  4000 = k' * 200 →           -- When X = 200, C = 4000
  (∃ (X C : ℝ), X * 50 = k ∧ C = k' * X ∧ X = 40 ∧ C = 800) := by
sorry

end NUMINAMATH_CALUDE_economy_relationship_l3704_370499


namespace NUMINAMATH_CALUDE_coefficient_proof_l3704_370439

theorem coefficient_proof (n : ℤ) :
  (∃! (count : ℕ), count = 25 ∧
    count = (Finset.filter (fun i => 1 < 4 * i + 7 ∧ 4 * i + 7 < 100) (Finset.range 200)).card) →
  ∃ (a : ℤ), ∀ (x : ℤ), (a * x + 7 = 4 * x + 7) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_proof_l3704_370439


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l3704_370414

-- Define the equation
def equation (a x : ℝ) : Prop :=
  x + |x| = 2 * Real.sqrt (3 + 2*a*x - 4*a)

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a | (0 < a ∧ a < 3/4) ∨ (a > 3)}

-- Theorem statement
theorem equation_two_distinct_roots (a : ℝ) :
  (∃ x y, x ≠ y ∧ equation a x ∧ equation a y) ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l3704_370414


namespace NUMINAMATH_CALUDE_hyperbola_intersection_line_l3704_370470

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Main theorem
theorem hyperbola_intersection_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ →
    hyperbola x₂ y₂ →
    is_midpoint x₁ y₁ x₂ y₂ 1 3 →
    (∀ x y, line x y ↔ (y - 3 = x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_line_l3704_370470


namespace NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l3704_370475

theorem quadratic_equation_from_root_properties (a b c : ℝ) (h_sum : b / a = -4) (h_product : c / a = 3) :
  ∃ (k : ℝ), k ≠ 0 ∧ k * (a * X^2 + b * X + c) = X^2 - 4*X + 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l3704_370475


namespace NUMINAMATH_CALUDE_john_lost_socks_l3704_370413

/-- The number of individual socks lost given initial pairs and maximum remaining pairs -/
def socks_lost (initial_pairs : ℕ) (max_remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * max_remaining_pairs

theorem john_lost_socks (initial_pairs : ℕ) (max_remaining_pairs : ℕ) 
  (h1 : initial_pairs = 10) (h2 : max_remaining_pairs = 7) : 
  socks_lost initial_pairs max_remaining_pairs = 6 := by
  sorry

#eval socks_lost 10 7

end NUMINAMATH_CALUDE_john_lost_socks_l3704_370413


namespace NUMINAMATH_CALUDE_pants_cost_is_correct_l3704_370421

/-- The cost of one pair of pants in dollars -/
def pants_cost : ℝ := 80

/-- The cost of one T-shirt in dollars -/
def tshirt_cost : ℝ := 20

/-- The cost of one pair of shoes in dollars -/
def shoes_cost : ℝ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.1

/-- The total cost after discount for Eugene's purchase -/
def total_cost_after_discount : ℝ := 558

theorem pants_cost_is_correct : 
  (4 * tshirt_cost + 3 * pants_cost + 2 * shoes_cost) * (1 - discount_rate) = total_cost_after_discount :=
by sorry

end NUMINAMATH_CALUDE_pants_cost_is_correct_l3704_370421


namespace NUMINAMATH_CALUDE_octagonal_cube_removed_volume_l3704_370407

/-- The volume of tetrahedra removed from a cube of side length 2 to make octagonal faces -/
theorem octagonal_cube_removed_volume :
  let cube_side : ℝ := 2
  let octagon_side : ℝ := 2 * (Real.sqrt 2 - 1)
  let tetrahedron_height : ℝ := 2 / Real.sqrt 2
  let tetrahedron_base_area : ℝ := 2 * (3 - 2 * Real.sqrt 2)
  let single_tetrahedron_volume : ℝ := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let total_removed_volume : ℝ := 8 * single_tetrahedron_volume
  total_removed_volume = (80 - 56 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_octagonal_cube_removed_volume_l3704_370407


namespace NUMINAMATH_CALUDE_unique_solution_l3704_370423

-- Define the circles
variable (A B C D E F : ℕ)

-- Define the conditions
def valid_arrangement (A B C D E F : ℕ) : Prop :=
  -- All numbers are between 1 and 6
  (A ∈ Finset.range 6) ∧ (B ∈ Finset.range 6) ∧ (C ∈ Finset.range 6) ∧
  (D ∈ Finset.range 6) ∧ (E ∈ Finset.range 6) ∧ (F ∈ Finset.range 6) ∧
  -- All numbers are distinct
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  -- Sums on each line are equal
  A + C + D = A + B ∧
  A + C + D = B + D + F ∧
  A + C + D = E + F ∧
  A + C + D = E + B + C

-- Theorem statement
theorem unique_solution :
  ∀ A B C D E F : ℕ, valid_arrangement A B C D E F → A = 6 ∧ B = 3 :=
sorry


end NUMINAMATH_CALUDE_unique_solution_l3704_370423


namespace NUMINAMATH_CALUDE_point_on_bisector_l3704_370460

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = x -/
def line_y_eq_x (p : Point) : Prop := p.y = p.x

theorem point_on_bisector (a b : ℝ) : 
  let A : Point := ⟨a, b⟩
  let B : Point := ⟨b, a⟩
  A = B → line_y_eq_x A := by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_l3704_370460


namespace NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l3704_370472

/-- The repeating cycle of the decimal representation of 1/13 -/
def cycle : List Nat := [0, 7, 6, 9, 2, 3]

/-- The length of the repeating cycle -/
def cycle_length : Nat := 6

/-- The position we're interested in -/
def target_position : Nat := 150

/-- Theorem: The 150th digit after the decimal point in the decimal 
    representation of 1/13 is 3 -/
theorem digit_150_of_one_thirteenth (h : cycle = [0, 7, 6, 9, 2, 3]) :
  cycle[target_position % cycle_length] = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l3704_370472


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3704_370498

theorem inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < -6 ↔ x ∈ Set.Ioo (-9/2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3704_370498


namespace NUMINAMATH_CALUDE_dandelion_seeds_l3704_370487

theorem dandelion_seeds (S : ℕ) : 
  (2/3 : ℚ) * (5/6 : ℚ) * (1/2 : ℚ) * S = 75 → S = 540 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seeds_l3704_370487


namespace NUMINAMATH_CALUDE_mikes_score_l3704_370462

theorem mikes_score (max_score : ℕ) (pass_percentage : ℚ) (shortfall : ℕ) (actual_score : ℕ) : 
  max_score = 750 → 
  pass_percentage = 30 / 100 → 
  shortfall = 13 → 
  actual_score = max_score * pass_percentage - shortfall →
  actual_score = 212 :=
by sorry

end NUMINAMATH_CALUDE_mikes_score_l3704_370462


namespace NUMINAMATH_CALUDE_sum_of_products_zero_l3704_370425

theorem sum_of_products_zero 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 43) :
  x*y + y*z + x*z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_zero_l3704_370425


namespace NUMINAMATH_CALUDE_power_function_through_point_l3704_370474

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 / 2 → f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3704_370474


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3704_370424

/-- The minimum value of a quadratic function y = (x - a)(x - b) -/
theorem quadratic_minimum_value (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) :
  ∃ x₀, ∀ x, (x - a) * (x - b) ≥ (x₀ - a) * (x₀ - b) ∧ 
  (x₀ - a) * (x₀ - b) = -(|a - b| / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3704_370424


namespace NUMINAMATH_CALUDE_can_determine_native_types_l3704_370495

/-- Represents the type of native: Knight or Liar -/
inductive NativeType
| Knight
| Liar

/-- Represents a native on the island -/
structure Native where
  type : NativeType
  leftNeighborAge : ℕ
  rightNeighborAge : ℕ

/-- The circle of natives -/
def NativeCircle := Vector Native 50

/-- Represents the statements made by a native -/
structure Statement where
  declaredLeftAge : ℕ
  declaredRightAge : ℕ

/-- Function to get the statements of all natives -/
def getAllStatements (circle : NativeCircle) : Vector Statement 50 := sorry

/-- Predicate to check if a native's statement is consistent with their type -/
def isConsistentStatement (native : Native) (statement : Statement) : Prop :=
  match native.type with
  | NativeType.Knight => 
      statement.declaredLeftAge = native.leftNeighborAge ∧ 
      statement.declaredRightAge = native.rightNeighborAge
  | NativeType.Liar => 
      (statement.declaredLeftAge = native.leftNeighborAge + 1 ∧ 
       statement.declaredRightAge = native.rightNeighborAge - 1) ∨
      (statement.declaredLeftAge = native.leftNeighborAge - 1 ∧ 
       statement.declaredRightAge = native.rightNeighborAge + 1)

/-- Main theorem: It's always possible to determine the identity of each native -/
theorem can_determine_native_types (circle : NativeCircle) :
  ∃ (determinedTypes : Vector NativeType 50),
    ∀ (i : Fin 50), 
      (circle.get i).type = determinedTypes.get i ∧
      isConsistentStatement (circle.get i) ((getAllStatements circle).get i) :=
sorry

end NUMINAMATH_CALUDE_can_determine_native_types_l3704_370495


namespace NUMINAMATH_CALUDE_line_parameterization_l3704_370440

def is_valid_parameterization (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), p = (a, 3 * a - 4, b) ∧ v = (1/3, 1, 1)

theorem line_parameterization 
  (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  is_valid_parameterization p v ↔
    (∃ (t : ℝ), 
      let (x, y, z) := p + t • v
      y = 3 * x - 4 ∧ z = t) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3704_370440


namespace NUMINAMATH_CALUDE_negation_equivalence_l3704_370441

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (a % 2 = 0 ∧ c % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3704_370441


namespace NUMINAMATH_CALUDE_max_stamps_for_50_dollars_max_stamps_is_maximum_l3704_370443

/-- The maximum number of stamps that can be purchased with a given budget and stamp price. -/
def maxStamps (budget : ℕ) (stampPrice : ℕ) : ℕ :=
  (budget / stampPrice : ℕ)

/-- Theorem stating the maximum number of stamps that can be purchased with $50 when each stamp costs 45 cents. -/
theorem max_stamps_for_50_dollars : maxStamps 5000 45 = 111 := by
  sorry

/-- Proof that the calculated maximum is indeed the largest possible number of stamps. -/
theorem max_stamps_is_maximum (budget : ℕ) (stampPrice : ℕ) :
  ∀ n : ℕ, n * stampPrice ≤ budget → n ≤ maxStamps budget stampPrice := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_for_50_dollars_max_stamps_is_maximum_l3704_370443


namespace NUMINAMATH_CALUDE_polynomial_has_real_root_l3704_370444

theorem polynomial_has_real_root (b : ℝ) : 
  ∃ x : ℝ, x^4 + b*x^3 + 2*x^2 + b*x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_real_root_l3704_370444


namespace NUMINAMATH_CALUDE_fifteenth_term_of_ap_l3704_370454

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- Theorem: The 15th term of an arithmetic progression with first term 2 and common difference 3 is 44 -/
theorem fifteenth_term_of_ap : arithmeticProgressionTerm 2 3 15 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_ap_l3704_370454


namespace NUMINAMATH_CALUDE_correct_number_of_hens_l3704_370404

/-- Given a total number of animals and feet, calculate the number of hens -/
def number_of_hens (total_animals : ℕ) (total_feet : ℕ) : ℕ :=
  2 * total_animals - total_feet / 2

theorem correct_number_of_hens :
  let total_animals := 46
  let total_feet := 140
  number_of_hens total_animals total_feet = 22 := by
  sorry

#eval number_of_hens 46 140

end NUMINAMATH_CALUDE_correct_number_of_hens_l3704_370404


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3704_370448

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3704_370448


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_main_theorem_l3704_370457

/-- A four-digit palindrome between 1000 and 10000 -/
def FourDigitPalindrome : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ ∃ a b : ℕ, n = 1000 * a + 100 * b + 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 }

/-- The theorem stating that all four-digit palindromes are divisible by 11 -/
theorem all_four_digit_palindromes_divisible_by_11 (n : FourDigitPalindrome) : 11 ∣ n.val := by
  sorry

/-- The probability that a randomly chosen four-digit palindrome is divisible by 11 -/
theorem probability_palindrome_divisible_by_11 : ℚ :=
  1

/-- The main theorem proving that the probability is 1 -/
theorem main_theorem : probability_palindrome_divisible_by_11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_main_theorem_l3704_370457


namespace NUMINAMATH_CALUDE_range_of_g_l3704_370430

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {-π/3, π/3} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l3704_370430


namespace NUMINAMATH_CALUDE_cube_greater_than_l3704_370486

theorem cube_greater_than (x y : ℝ) (h : x > y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_greater_than_l3704_370486


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3704_370431

/-- The eccentricity of a hyperbola x^2 - y^2/4 = 1 is √5 -/
theorem hyperbola_eccentricity :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  x^2 - y^2 / 4 = 1 → e = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3704_370431


namespace NUMINAMATH_CALUDE_even_count_in_pascal_triangle_l3704_370415

/-- Pascal's Triangle coefficient -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Count even numbers in a single row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ :=
  (List.range (row + 1)).filter (fun k => isEven (binomial row k)) |>.length

/-- Count even numbers in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ :=
  (List.range n).map countEvenInRow |>.sum

/-- Theorem: There are 64 even integers in the first 15 rows of Pascal's Triangle -/
theorem even_count_in_pascal_triangle : countEvenInTriangle 15 = 64 := by
  sorry

end NUMINAMATH_CALUDE_even_count_in_pascal_triangle_l3704_370415


namespace NUMINAMATH_CALUDE_sausage_problem_l3704_370438

theorem sausage_problem (initial_sausages : ℕ) (remaining_sausages : ℕ) 
  (h1 : initial_sausages = 600)
  (h2 : remaining_sausages = 45) :
  ∃ (x : ℚ), 
    0 < x ∧ x < 1 ∧
    remaining_sausages = (1/4 : ℚ) * (1/2 : ℚ) * (1 - x) * initial_sausages ∧
    x = (2/5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sausage_problem_l3704_370438


namespace NUMINAMATH_CALUDE_same_distance_different_time_l3704_370481

/-- Calculates the required average speed for a rider to cover the same distance as another rider in a different time. -/
theorem same_distance_different_time 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 5) : 
  (joann_speed * joann_time) / fran_time = 12 := by
  sorry

#check same_distance_different_time

end NUMINAMATH_CALUDE_same_distance_different_time_l3704_370481


namespace NUMINAMATH_CALUDE_tree_planting_multiple_l3704_370401

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The total number of trees planted by all grades -/
def total_trees : ℕ := 240

/-- The multiple of 5th graders' trees compared to 6th graders' trees -/
def m : ℕ := 3

/-- Theorem stating that m is the correct multiple -/
theorem tree_planting_multiple :
  m * trees_5th - 30 = total_trees - trees_4th - trees_5th := by
  sorry

#check tree_planting_multiple

end NUMINAMATH_CALUDE_tree_planting_multiple_l3704_370401


namespace NUMINAMATH_CALUDE_max_table_sum_l3704_370436

def numbers : List ℕ := [2, 3, 5, 7, 11, 13]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset = numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem max_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
    table_sum top left ≤ 420 :=
  sorry

end NUMINAMATH_CALUDE_max_table_sum_l3704_370436


namespace NUMINAMATH_CALUDE_atomic_number_calculation_l3704_370490

/-- Represents an atomic element -/
structure Element where
  massNumber : ℕ
  neutronCount : ℕ
  atomicNumber : ℕ

/-- The relation between mass number, neutron count, and atomic number in an element -/
def isValidElement (e : Element) : Prop :=
  e.massNumber = e.neutronCount + e.atomicNumber

theorem atomic_number_calculation (e : Element)
  (h1 : e.massNumber = 288)
  (h2 : e.neutronCount = 169)
  (h3 : isValidElement e) :
  e.atomicNumber = 119 := by
  sorry

#check atomic_number_calculation

end NUMINAMATH_CALUDE_atomic_number_calculation_l3704_370490


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3704_370428

theorem min_value_and_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + Real.sqrt 2 * b + Real.sqrt 3 * c = 2 * Real.sqrt 3) :
  (∃ m : ℝ, 
    (∀ a' b' c' : ℝ, a' + Real.sqrt 2 * b' + Real.sqrt 3 * c' = 2 * Real.sqrt 3 → 
      a'^2 + b'^2 + c'^2 ≥ m) ∧ 
    (a^2 + b^2 + c^2 = m) ∧
    m = 2) ∧
  (∃ p q : ℝ, ∀ x : ℝ, (|x - 3| ≥ 2 ↔ x^2 + p*x + q ≥ 0) ∧ p = -6) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3704_370428


namespace NUMINAMATH_CALUDE_temperature_decrease_fraction_l3704_370427

theorem temperature_decrease_fraction (current_temp : ℝ) (decrease : ℝ) 
  (h1 : current_temp = 84)
  (h2 : decrease = 21) :
  (current_temp - decrease) / current_temp = 3/4 := by
sorry

end NUMINAMATH_CALUDE_temperature_decrease_fraction_l3704_370427


namespace NUMINAMATH_CALUDE_ratio_abc_l3704_370484

theorem ratio_abc (a b c : ℝ) (ha : a ≠ 0) 
  (h : 14 * (a^2 + b^2 + c^2) = (a + 2*b + 3*c)^2) : 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k ∧ b = 2*k ∧ c = 3*k := by
  sorry

end NUMINAMATH_CALUDE_ratio_abc_l3704_370484


namespace NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l3704_370411

/-- A quadratic function f(x) = x^2 + 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- The discriminant of the quadratic function f -/
def discriminant (a : ℝ) : ℝ := 4 - 4*a

theorem no_real_roots_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, f a x ≠ 0) → a > 1 := by
  sorry

#check no_real_roots_implies_a_greater_than_one

end NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l3704_370411


namespace NUMINAMATH_CALUDE_power_equality_no_quadratic_term_l3704_370483

-- Define the variables
variable (x y a b : ℝ)

-- Theorem 1
theorem power_equality (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b := by sorry

-- Theorem 2
theorem no_quadratic_term (h : ∀ x, (x - 1) * (x^2 + a*x + 1) = x^3 + c*x + d) : a = 1 := by sorry

end NUMINAMATH_CALUDE_power_equality_no_quadratic_term_l3704_370483


namespace NUMINAMATH_CALUDE_not_always_reducible_box_dimension_l3704_370416

structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  positive_dimensions : 0 < length ∧ 0 < width ∧ 0 < height

def fits_in (p q : RectangularParallelepiped) : Prop :=
  p.length ≤ q.length ∧ p.width ≤ q.width ∧ p.height ≤ q.height

def is_defective (original defective : RectangularParallelepiped) : Prop :=
  (defective.length < original.length ∧ defective.width = original.width ∧ defective.height = original.height) ∨
  (defective.length = original.length ∧ defective.width < original.width ∧ defective.height = original.height) ∨
  (defective.length = original.length ∧ defective.width = original.width ∧ defective.height < original.height)

theorem not_always_reducible_box_dimension 
  (box : RectangularParallelepiped) 
  (parallelepipeds : List RectangularParallelepiped) 
  (original_parallelepipeds : List RectangularParallelepiped) 
  (h1 : ∀ p ∈ parallelepipeds, fits_in p box)
  (h2 : parallelepipeds.length = original_parallelepipeds.length)
  (h3 : ∀ (i : Fin parallelepipeds.length), is_defective (original_parallelepipeds[i]) (parallelepipeds[i])) :
  ¬ (∀ (reduced_box : RectangularParallelepiped), 
    (reduced_box.length < box.length ∨ reduced_box.width < box.width ∨ reduced_box.height < box.height) → 
    (∀ p ∈ parallelepipeds, fits_in p reduced_box)) :=
by sorry

end NUMINAMATH_CALUDE_not_always_reducible_box_dimension_l3704_370416


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3704_370408

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = 9) ∧ (x₁^2 - 5*x₁ - 26 = 4*x₁ + 21) ∧ (x₂^2 - 5*x₂ - 26 = 4*x₂ + 21)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3704_370408


namespace NUMINAMATH_CALUDE_function_properties_l3704_370446

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.sin x + a

theorem function_properties (t : ℝ) :
  (∃ a : ℝ, f a π = 1 ∧ f a t = 2) →
  (∃ a : ℝ, a = 1 ∧ f a (-t) = 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3704_370446


namespace NUMINAMATH_CALUDE_representatives_selection_theorem_l3704_370493

/-- The number of ways to select 3 representatives from 3 different companies -/
def selectRepresentatives (totalCompanies : ℕ) (companiesWithOneRep : ℕ) (repsFromSpecialCompany : ℕ) : ℕ :=
  Nat.choose repsFromSpecialCompany 1 * Nat.choose companiesWithOneRep 2 +
  Nat.choose companiesWithOneRep 3

/-- Theorem stating that the number of ways to select 3 representatives from 3 different companies
    out of 5 companies (where one company has 2 representatives and the others have 1 each) is 16 -/
theorem representatives_selection_theorem :
  selectRepresentatives 5 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_theorem_l3704_370493


namespace NUMINAMATH_CALUDE_magnitude_z_squared_l3704_370420

-- Define the complex number z
def z : ℂ := 1 + Complex.I^5

-- Theorem statement
theorem magnitude_z_squared : Complex.abs (z^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_squared_l3704_370420


namespace NUMINAMATH_CALUDE_max_value_of_f_l3704_370450

-- Define the function
def f (x : ℝ) : ℝ := 5 * x - 4 * x^2 + 6

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), (∀ (x : ℝ), f x ≤ max) ∧ (max = 121 / 16) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3704_370450


namespace NUMINAMATH_CALUDE_min_edges_theorem_l3704_370455

/-- A simple graph with 19998 vertices -/
structure Graph :=
  (vertices : Finset Nat)
  (edges : Finset (Nat × Nat))
  (simple : ∀ e ∈ edges, e.1 ≠ e.2)
  (vertex_count : vertices.card = 19998)

/-- A subgraph of G with 9999 vertices -/
def Subgraph (G : Graph) :=
  {G' : Graph | G'.vertices ⊆ G.vertices ∧ G'.edges ⊆ G.edges ∧ G'.vertices.card = 9999}

/-- The condition that any subgraph with 9999 vertices has at least 9999 edges -/
def SubgraphEdgeCondition (G : Graph) :=
  ∀ G' ∈ Subgraph G, G'.edges.card ≥ 9999

/-- The theorem stating that G has at least 49995 edges -/
theorem min_edges_theorem (G : Graph) (h : SubgraphEdgeCondition G) :
  G.edges.card ≥ 49995 := by
  sorry

end NUMINAMATH_CALUDE_min_edges_theorem_l3704_370455


namespace NUMINAMATH_CALUDE_sum_of_complex_sequence_l3704_370429

theorem sum_of_complex_sequence : 
  let n : ℕ := 150
  let a₀ : ℤ := -74
  let b₀ : ℤ := 30
  let d : ℤ := 1
  let sum : ℂ := (↑n / 2 : ℚ) * ↑(2 * a₀ + (n - 1) * d) + 
                 (↑n / 2 : ℚ) * ↑(2 * b₀ + (n - 1) * d) * Complex.I
  sum = 75 + 15675 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_sum_of_complex_sequence_l3704_370429


namespace NUMINAMATH_CALUDE_erica_ride_duration_l3704_370468

/-- The duration in minutes that Dave can ride the merry-go-round -/
def dave_duration : ℝ := 10

/-- The factor by which Chuck can ride longer than Dave -/
def chuck_factor : ℝ := 5

/-- The percentage longer that Erica can ride compared to Chuck -/
def erica_percentage : ℝ := 0.30

/-- The duration in minutes that Chuck can ride the merry-go-round -/
def chuck_duration : ℝ := dave_duration * chuck_factor

/-- The duration in minutes that Erica can ride the merry-go-round -/
def erica_duration : ℝ := chuck_duration * (1 + erica_percentage)

/-- Theorem stating that Erica can ride for 65 minutes -/
theorem erica_ride_duration : erica_duration = 65 := by sorry

end NUMINAMATH_CALUDE_erica_ride_duration_l3704_370468


namespace NUMINAMATH_CALUDE_yunas_average_score_l3704_370418

/-- Given Yuna's average score for May and June and her July score, 
    calculate her average score over the three months. -/
theorem yunas_average_score 
  (may_june_avg : ℝ) 
  (july_score : ℝ) 
  (h1 : may_june_avg = 84) 
  (h2 : july_score = 96) : 
  (2 * may_june_avg + july_score) / 3 = 88 := by
  sorry

#eval (2 * 84 + 96) / 3  -- This should evaluate to 88

end NUMINAMATH_CALUDE_yunas_average_score_l3704_370418


namespace NUMINAMATH_CALUDE_city_female_population_l3704_370464

/-- Calculates the female population of a city given specific demographic information. -/
theorem city_female_population
  (total_population : ℕ)
  (migrant_percentage : ℚ)
  (rural_migrant_percentage : ℚ)
  (local_female_percentage : ℚ)
  (rural_migrant_female_percentage : ℚ)
  (urban_migrant_female_percentage : ℚ)
  (h_total : total_population = 728400)
  (h_migrant : migrant_percentage = 35 / 100)
  (h_rural : rural_migrant_percentage = 20 / 100)
  (h_local_female : local_female_percentage = 48 / 100)
  (h_rural_female : rural_migrant_female_percentage = 30 / 100)
  (h_urban_female : urban_migrant_female_percentage = 40 / 100) :
  ∃ (female_population : ℕ), female_population = 324128 :=
by
  sorry


end NUMINAMATH_CALUDE_city_female_population_l3704_370464


namespace NUMINAMATH_CALUDE_sugar_left_l3704_370406

/-- Given a recipe requiring 2 cups of sugar, if you can make 0.165 of the recipe,
    then you have 0.33 cups of sugar left. -/
theorem sugar_left (full_recipe : ℝ) (fraction_possible : ℝ) (sugar_left : ℝ) :
  full_recipe = 2 →
  fraction_possible = 0.165 →
  sugar_left = full_recipe * fraction_possible →
  sugar_left = 0.33 := by
sorry

end NUMINAMATH_CALUDE_sugar_left_l3704_370406


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3704_370400

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3704_370400


namespace NUMINAMATH_CALUDE_min_point_sum_l3704_370412

-- Define the function f(x) = 3x - x³
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 - 3 * x^2

-- Theorem statement
theorem min_point_sum :
  ∃ (a b : ℝ), (∀ x, f x ≥ f a) ∧ (f a = b) ∧ (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_min_point_sum_l3704_370412


namespace NUMINAMATH_CALUDE_bookstore_ratio_l3704_370417

theorem bookstore_ratio : 
  ∀ (sarah_paperback sarah_hardback brother_total : ℕ),
    sarah_paperback = 6 →
    sarah_hardback = 4 →
    brother_total = 10 →
    ∃ (brother_paperback brother_hardback : ℕ),
      brother_paperback = sarah_paperback / 3 →
      brother_hardback + brother_paperback = brother_total →
      (brother_hardback : ℚ) / sarah_hardback = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_ratio_l3704_370417


namespace NUMINAMATH_CALUDE_cookies_needed_l3704_370465

/-- Given 6.0 people who each should receive 24.0 cookies, prove that the total number of cookies needed is 144.0. -/
theorem cookies_needed (people : Float) (cookies_per_person : Float) (h1 : people = 6.0) (h2 : cookies_per_person = 24.0) :
  people * cookies_per_person = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_cookies_needed_l3704_370465


namespace NUMINAMATH_CALUDE_optimal_range_golden_section_l3704_370480

theorem optimal_range_golden_section (m : ℝ) : 
  (1000 ≤ m) →  -- The optimal range starts at 1000
  (1000 + (m - 1000) * 0.618 = 1618) →  -- The good point is determined by the golden ratio
  (m = 2000) :=  -- We want to prove that m = 2000
by
  sorry

end NUMINAMATH_CALUDE_optimal_range_golden_section_l3704_370480


namespace NUMINAMATH_CALUDE_ap_has_twelve_terms_l3704_370426

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ
  odd_sum : ℝ
  even_sum : ℝ
  last_term : ℝ
  third_term : ℝ

/-- The conditions of the arithmetic progression -/
def APConditions (ap : ArithmeticProgression) : Prop :=
  Even ap.n ∧
  ap.odd_sum = 36 ∧
  ap.even_sum = 42 ∧
  ap.last_term = ap.a + 12 ∧
  ap.third_term = 6 ∧
  ap.third_term = ap.a + 2 * ap.d ∧
  ap.odd_sum = (ap.n / 2 : ℝ) * (ap.a + (ap.a + (ap.n - 2) * ap.d)) ∧
  ap.even_sum = (ap.n / 2 : ℝ) * ((ap.a + ap.d) + (ap.a + (ap.n - 1) * ap.d))

/-- The theorem to be proved -/
theorem ap_has_twelve_terms (ap : ArithmeticProgression) :
  APConditions ap → ap.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_twelve_terms_l3704_370426


namespace NUMINAMATH_CALUDE_first_obtuse_triangle_l3704_370479

/-- Represents a triangle with three angles -/
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

/-- Constructs the pedal triangle of a given triangle -/
def pedal_triangle (t : Triangle) : Triangle :=
  { angle1 := 180 - 2 * t.angle1,
    angle2 := 180 - 2 * t.angle2,
    angle3 := 180 - 2 * t.angle3 }

/-- Checks if a triangle is obtuse -/
def is_obtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

/-- Generates the nth pedal triangle in the sequence -/
def nth_pedal_triangle (n : Nat) : Triangle :=
  match n with
  | 0 => { angle1 := 59.5, angle2 := 60, angle3 := 60.5 }
  | n + 1 => pedal_triangle (nth_pedal_triangle n)

theorem first_obtuse_triangle :
  ∀ n : Nat, n < 6 → ¬(is_obtuse (nth_pedal_triangle n)) ∧
  is_obtuse (nth_pedal_triangle 6) :=
by sorry

end NUMINAMATH_CALUDE_first_obtuse_triangle_l3704_370479


namespace NUMINAMATH_CALUDE_min_sum_of_radii_l3704_370442

/-
  Define a regular tetrahedron with edge length 1
-/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_unit : edge_length = 1)

/-
  Define a sphere inside the tetrahedron
-/
structure Sphere :=
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

/-
  Define the property of a sphere being tangent to three faces of the tetrahedron
-/
def is_tangent_to_three_faces (s : Sphere) (t : RegularTetrahedron) (vertex : ℝ × ℝ × ℝ) : Prop :=
  sorry  -- This would involve complex geometric conditions

/-
  State the theorem
-/
theorem min_sum_of_radii (t : RegularTetrahedron) 
  (s1 s2 : Sphere) 
  (h1 : is_tangent_to_three_faces s1 t (0, 0, 0))  -- Assume A is at (0,0,0)
  (h2 : is_tangent_to_three_faces s2 t (1, 0, 0))  -- Assume B is at (1,0,0)
  : 
  s1.radius + s2.radius ≥ (Real.sqrt 6 - 1) / 5 := by
  sorry


end NUMINAMATH_CALUDE_min_sum_of_radii_l3704_370442


namespace NUMINAMATH_CALUDE_seventh_term_is_64_l3704_370492

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  first_third_product : a 1 * a 3 = 4
  ninth_term : a 9 = 256

/-- The 7th term of the geometric sequence is 64 -/
theorem seventh_term_is_64 (seq : GeometricSequence) : seq.a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_64_l3704_370492


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3704_370402

theorem largest_integer_with_remainder (n : ℕ) : 
  (∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n) ∧ 
  n < 100 ∧ 
  n % 7 = 4 → 
  n = 95 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3704_370402


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l3704_370497

/-- Given a paint mixture with ratio blue:green:white as 5:3:7,
    prove that using 21 quarts of white paint requires 9 quarts of green paint. -/
theorem paint_mixture_ratio (blue green white : ℚ) 
  (ratio : blue / green = 5 / 3 ∧ green / white = 3 / 7) 
  (white_amount : white = 21) : green = 9 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l3704_370497


namespace NUMINAMATH_CALUDE_coating_time_for_given_problem_l3704_370410

/-- Represents the properties of the sphere coating problem -/
structure SphereCoating where
  copper_sphere_diameter : ℝ
  silver_layer_thickness : ℝ
  hydrogen_production : ℝ
  hydrogen_silver_ratio : ℝ
  silver_density : ℝ

/-- Calculates the time required for coating the sphere -/
noncomputable def coating_time (sc : SphereCoating) : ℝ :=
  sorry

/-- Theorem stating the coating time for the given problem -/
theorem coating_time_for_given_problem :
  let sc : SphereCoating := {
    copper_sphere_diameter := 3,
    silver_layer_thickness := 0.05,
    hydrogen_production := 11.11,
    hydrogen_silver_ratio := 1 / 108,
    silver_density := 10.5
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |coating_time sc - 987| < ε :=
sorry

end NUMINAMATH_CALUDE_coating_time_for_given_problem_l3704_370410


namespace NUMINAMATH_CALUDE_solve_for_m_l3704_370466

theorem solve_for_m : ∃ m : ℚ, 
  (∃ x y : ℚ, 3 * x - 4 * (m - 1) * y + 30 = 0 ∧ x = 2 ∧ y = -3) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3704_370466


namespace NUMINAMATH_CALUDE_coefficient_x3_in_expansion_l3704_370494

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the sum of coefficients
def sumCoefficients (n : ℕ) : ℕ := (3 : ℕ) ^ n

-- Define the function to calculate the coefficient of x³
def coefficientX3 (n : ℕ) : ℕ := 8 * binomial n 3

-- Theorem statement
theorem coefficient_x3_in_expansion :
  ∃ n : ℕ, sumCoefficients n = 243 ∧ coefficientX3 n = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_in_expansion_l3704_370494


namespace NUMINAMATH_CALUDE_smallest_power_of_1512_l3704_370491

theorem smallest_power_of_1512 :
  ∃ (n : ℕ), 1512 * 49 = n^3 ∧
  ∀ (x : ℕ), x > 0 ∧ x < 49 → ¬∃ (m : ℕ), ∃ (k : ℕ), 1512 * x = m^k := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_of_1512_l3704_370491


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3704_370473

theorem inequality_solution_set (a : ℝ) : 
  (∀ x, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3704_370473


namespace NUMINAMATH_CALUDE_triangle_properties_l3704_370432

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : (Real.cos t.C) / (Real.sin t.C) = (Real.cos t.A + Real.cos t.B) / (Real.sin t.A + Real.sin t.B)) :
  t.C = π / 3 ∧ 
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3704_370432


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l3704_370456

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 18 ≥ 0}
def B : Set ℝ := {x | (x+5)/(x-14) ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+1}

-- Theorem for part (1)
theorem complement_B_intersect_A : 
  (Set.univ \ B) ∩ A = Set.Iic (-5) ∪ Set.Ici 14 := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : 
  (B ∩ C a = C a) ↔ a ≥ -5/2 := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l3704_370456


namespace NUMINAMATH_CALUDE_smallest_number_in_special_set_l3704_370469

theorem smallest_number_in_special_set (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 26 →
  b = 27 →
  c = b + 5 →
  a < b ∧ b < c →
  a = 19 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_in_special_set_l3704_370469


namespace NUMINAMATH_CALUDE_point_not_in_region_l3704_370422

theorem point_not_in_region (m : ℝ) : 
  (1 : ℝ) - (m^2 - 2*m + 4)*(1 : ℝ) + 6 ≤ 0 ↔ m ≤ -1 ∨ m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3704_370422


namespace NUMINAMATH_CALUDE_expression_equals_two_fifths_l3704_370488

theorem expression_equals_two_fifths :
  (((3^1 : ℚ) - 6 + 4^2 - 3)⁻¹ * 4) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_fifths_l3704_370488


namespace NUMINAMATH_CALUDE_quadruple_solutions_l3704_370453

def is_solution (a b c k : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ k > 0 ∧
  a^2 + b^2 + 16*c^2 = 9*k^2 + 1

theorem quadruple_solutions :
  ∀ a b c k : ℕ,
    is_solution a b c k ↔
      ((a, b, c, k) = (3, 3, 2, 3) ∨
       (a, b, c, k) = (3, 17, 3, 7) ∨
       (a, b, c, k) = (17, 3, 3, 7) ∨
       (a, b, c, k) = (3, 37, 3, 13) ∨
       (a, b, c, k) = (37, 3, 3, 13)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solutions_l3704_370453


namespace NUMINAMATH_CALUDE_angleBMeasureApprox_l3704_370489

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : ℝ  -- Measure of angle A in degrees
  B : ℝ  -- Measure of angle B in degrees
  C : ℝ  -- Measure of angle C in degrees
  isIsosceles : B = C
  angleRelation : C = 3 * A + 10
  angleSum : A + B + C = 180

/-- The measure of angle B in the isosceles triangle -/
def angleBMeasure (triangle : IsoscelesTriangle) : ℝ := triangle.B

/-- Theorem stating the measure of angle B -/
theorem angleBMeasureApprox (triangle : IsoscelesTriangle) : 
  ∃ ε > 0, |angleBMeasure triangle - 550/7| < ε :=
sorry

end NUMINAMATH_CALUDE_angleBMeasureApprox_l3704_370489


namespace NUMINAMATH_CALUDE_jill_draws_spade_prob_l3704_370449

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Represents the probability of drawing a spade from a standard deck -/
def ProbSpade : ℚ := NumSpades / StandardDeck

/-- Represents the probability of not drawing a spade from a standard deck -/
def ProbNotSpade : ℚ := 1 - ProbSpade

/-- Represents the probability that Jack draws a spade -/
def ProbJackSpade : ℚ := ProbSpade

/-- Represents the probability that Jill draws a spade -/
def ProbJillSpade : ℚ := ProbNotSpade * ProbSpade

/-- Represents the probability that John draws a spade -/
def ProbJohnSpade : ℚ := ProbNotSpade * ProbNotSpade * ProbSpade

/-- Represents the probability that a spade is drawn in one cycle -/
def ProbSpadeInCycle : ℚ := ProbJackSpade + ProbJillSpade + ProbJohnSpade

theorem jill_draws_spade_prob : 
  ProbJillSpade / ProbSpadeInCycle = 12 / 37 := by sorry

end NUMINAMATH_CALUDE_jill_draws_spade_prob_l3704_370449


namespace NUMINAMATH_CALUDE_problem_solution_l3704_370445

theorem problem_solution : 
  (∀ π : ℝ, (π - 2)^0 + (-1)^3 = 0) ∧ 
  (∀ m n : ℝ, (3*m + n) * (m - 2*n) = 3*m^2 - 5*m*n - 2*n^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3704_370445


namespace NUMINAMATH_CALUDE_coins_in_second_stack_l3704_370459

theorem coins_in_second_stack (total_coins : ℕ) (first_stack : ℕ) (h1 : total_coins = 12) (h2 : first_stack = 4) :
  total_coins - first_stack = 8 := by
  sorry

end NUMINAMATH_CALUDE_coins_in_second_stack_l3704_370459


namespace NUMINAMATH_CALUDE_store_customers_l3704_370433

/-- Proves that the number of customers is 1000 given the specified conditions --/
theorem store_customers (return_rate : ℝ) (book_price : ℝ) (final_sales : ℝ) :
  return_rate = 0.37 →
  book_price = 15 →
  final_sales = 9450 →
  (1 - return_rate) * book_price * (final_sales / ((1 - return_rate) * book_price)) = 1000 := by
sorry

#eval (1 - 0.37) * 15 * (9450 / ((1 - 0.37) * 15)) -- Should output 1000.0

end NUMINAMATH_CALUDE_store_customers_l3704_370433


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3704_370447

/-- Calculates the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 36 →
  crossing_time = 24.198064154867613 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 131.98064154867613 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3704_370447


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l3704_370482

/-- Represents a cube with painted diagonal stripes on each face -/
structure StripedCube where
  /-- The number of faces on the cube -/
  num_faces : Nat
  /-- The number of possible stripe orientations per face -/
  orientations_per_face : Nat
  /-- The total number of possible stripe combinations -/
  total_combinations : Nat
  /-- The number of favorable outcomes (continuous stripes) -/
  favorable_outcomes : Nat

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe (cube : StripedCube) : Rat :=
  cube.favorable_outcomes / cube.total_combinations

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  ∃ (cube : StripedCube),
    cube.num_faces = 6 ∧
    cube.orientations_per_face = 2 ∧
    cube.total_combinations = 2^6 ∧
    cube.favorable_outcomes = 6 ∧
    probability_continuous_stripe cube = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l3704_370482


namespace NUMINAMATH_CALUDE_tax_base_amount_theorem_l3704_370437

/-- Calculates the base amount given the tax rate and tax amount -/
def calculate_base_amount (tax_rate : ℚ) (tax_amount : ℚ) : ℚ :=
  tax_amount / (tax_rate / 100)

/-- Theorem: Given a tax rate of 65% and a tax amount of $65, the base amount is $100 -/
theorem tax_base_amount_theorem :
  let tax_rate : ℚ := 65
  let tax_amount : ℚ := 65
  calculate_base_amount tax_rate tax_amount = 100 := by
  sorry

#eval calculate_base_amount 65 65

end NUMINAMATH_CALUDE_tax_base_amount_theorem_l3704_370437


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3704_370476

/-- Calculate the selling price of a cycle given its cost price and gain percent. -/
theorem cycle_selling_price (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) :
  cost_price = 450 →
  gain_percent = 15.56 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 520.02 := by
  sorry


end NUMINAMATH_CALUDE_cycle_selling_price_l3704_370476


namespace NUMINAMATH_CALUDE_cube_coloring_count_l3704_370463

/-- The number of ways to color a cube with two colors -/
def cube_colorings : ℕ :=
  let faces := 6  -- number of faces on a cube
  let colors := 2  -- number of colors (red and blue)
  -- The actual calculation is not provided, as per the instructions
  20  -- The result we want to prove

/-- Theorem stating that the number of valid cube colorings is 20 -/
theorem cube_coloring_count : cube_colorings = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l3704_370463


namespace NUMINAMATH_CALUDE_books_count_l3704_370419

/-- The total number of books owned by six friends -/
def total_books (sandy benny tim rachel alex jordan : ℕ) : ℕ :=
  sandy + benny + tim + rachel + alex + jordan

/-- Theorem stating the total number of books owned by the six friends -/
theorem books_count :
  ∃ (sandy benny tim rachel alex jordan : ℕ),
    sandy = 10 ∧
    benny = 24 ∧
    tim = 33 ∧
    rachel = 2 * benny ∧
    alex = tim / 2 - 3 ∧
    jordan = sandy + benny ∧
    total_books sandy benny tim rachel alex jordan = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_books_count_l3704_370419


namespace NUMINAMATH_CALUDE_expression_evaluation_l3704_370434

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 2 - 3
  (2*a + Real.sqrt 3) * (2*a - Real.sqrt 3) - 3*a*(a - 2) + 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3704_370434


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l3704_370405

theorem solve_equation_for_x : ∃ x : ℚ, (3 * x / 7) - 2 = 12 ∧ x = 98 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l3704_370405


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l3704_370435

/-- The common factor of the polynomial 2m^2n + 6mn - 4m^3n is 2mn -/
theorem common_factor_of_polynomial (m n : ℤ) : 
  ∃ (k : ℤ), 2 * m^2 * n + 6 * m * n - 4 * m^3 * n = 2 * m * n * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l3704_370435
