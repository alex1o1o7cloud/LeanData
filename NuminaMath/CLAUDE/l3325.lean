import Mathlib

namespace NUMINAMATH_CALUDE_point_on_y_axis_l3325_332519

/-- Given that point P(2-a, a-3) lies on the y-axis, prove that a = 2 -/
theorem point_on_y_axis (a : ℝ) : (2 - a = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3325_332519


namespace NUMINAMATH_CALUDE_exists_polyhedron_with_hidden_vertices_l3325_332572

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_valid : True  -- Additional conditions for a valid polyhedron

/-- Checks if a point is outside a polyhedron -/
def is_outside (P : Polyhedron) (Q : Fin 3 → ℝ) : Prop :=
  Q ∉ P.vertices ∧ ∀ f ∈ P.faces, Q ∉ f

/-- Checks if a line segment intersects the interior of a polyhedron -/
def intersects_interior (P : Polyhedron) (A B : Fin 3 → ℝ) : Prop :=
  ∃ C : Fin 3 → ℝ, C ≠ A ∧ C ≠ B ∧ C ∈ P.vertices ∧ 
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = λ i => (1 - t) * A i + t * B i

/-- The main theorem -/
theorem exists_polyhedron_with_hidden_vertices : 
  ∃ (P : Polyhedron) (Q : Fin 3 → ℝ), 
    is_outside P Q ∧ 
    ∀ V ∈ P.vertices, intersects_interior P Q V :=
  sorry

end NUMINAMATH_CALUDE_exists_polyhedron_with_hidden_vertices_l3325_332572


namespace NUMINAMATH_CALUDE_cubic_factorization_l3325_332536

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3325_332536


namespace NUMINAMATH_CALUDE_incorrect_inference_l3325_332502

-- Define the types for our geometric objects
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)

-- Define the relationships between geometric objects
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_on_plane : Line → Plane → Prop)

-- State the theorem
theorem incorrect_inference
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ (l : Line) (α : Plane) (A : Point),
    (¬ line_on_plane l α ∧ on_line A l) → ¬ on_plane A α) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_inference_l3325_332502


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3325_332503

theorem complex_number_in_first_quadrant (z : ℂ) : 
  z / (z - Complex.I) = Complex.I → 
  (z.re > 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3325_332503


namespace NUMINAMATH_CALUDE_pi_irrational_l3325_332551

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem pi_irrational :
  is_rational (-4/7) →
  is_rational 3.333333 →
  is_rational 1.010010001 →
  ¬ is_rational Real.pi :=
by sorry

end NUMINAMATH_CALUDE_pi_irrational_l3325_332551


namespace NUMINAMATH_CALUDE_unique_solution_l3325_332562

/-- Represents the denomination of a coin -/
inductive Coin : Type
  | One : Coin
  | Two : Coin
  | Five : Coin
  | Ten : Coin
  | Twenty : Coin

/-- The value of a coin in forints -/
def coin_value : Coin → Nat
  | Coin.One => 1
  | Coin.Two => 2
  | Coin.Five => 5
  | Coin.Ten => 10
  | Coin.Twenty => 20

/-- Represents the count of each coin type -/
structure CoinCount where
  one : Nat
  two : Nat
  five : Nat
  ten : Nat
  twenty : Nat

/-- The given coin count from the problem -/
def problem_coin_count : CoinCount :=
  { one := 3, two := 9, five := 5, ten := 6, twenty := 3 }

/-- Check if a number can be represented by the given coin count -/
def can_represent (n : Nat) (cc : CoinCount) : Prop :=
  ∃ (a b c d e : Nat),
    a ≤ cc.twenty ∧ b ≤ cc.ten ∧ c ≤ cc.five ∧ d ≤ cc.two ∧ e ≤ cc.one ∧
    n = a * 20 + b * 10 + c * 5 + d * 2 + e * 1

/-- The set of drawn numbers -/
def drawn_numbers : Finset Nat :=
  {34, 33, 29, 19, 18, 17, 16}

/-- The theorem to be proved -/
theorem unique_solution :
  (∀ n ∈ drawn_numbers, n ≤ 35) ∧
  (drawn_numbers.card = 7) ∧
  (∀ n ∈ drawn_numbers, can_represent n problem_coin_count) ∧
  (∀ s : Finset Nat, s ≠ drawn_numbers →
    s.card = 7 →
    (∀ n ∈ s, n ≤ 35) →
    (∀ n ∈ s, can_represent n problem_coin_count) →
    False) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3325_332562


namespace NUMINAMATH_CALUDE_ship_cargo_theorem_l3325_332571

def initial_cargo : ℕ := 5973
def loaded_cargo : ℕ := 8723

theorem ship_cargo_theorem : 
  initial_cargo + loaded_cargo = 14696 := by sorry

end NUMINAMATH_CALUDE_ship_cargo_theorem_l3325_332571


namespace NUMINAMATH_CALUDE_exam_items_count_l3325_332552

theorem exam_items_count :
  ∀ (X : ℕ) (E M : ℕ),
    M = 24 →
    M = E / 2 + 6 →
    X = E + 4 →
    X = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_items_count_l3325_332552


namespace NUMINAMATH_CALUDE_f_inequality_l3325_332548

/-- A function that is continuous and differentiable on ℝ -/
def ContinuousDifferentiableFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ Differentiable ℝ f

theorem f_inequality (f : ℝ → ℝ) 
  (h_f : ContinuousDifferentiableFunction f)
  (h_ineq : ∀ x, 2 * f x - deriv f x > 0) :
  f 1 > f 2 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3325_332548


namespace NUMINAMATH_CALUDE_evaluate_expression_l3325_332561

theorem evaluate_expression (c d : ℝ) (h1 : c = 3) (h2 : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3325_332561


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l3325_332545

theorem fraction_addition_simplification :
  (1 : ℚ) / 462 + 23 / 42 = 127 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l3325_332545


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l3325_332555

/-- An arithmetic sequence with a_4 = -14 and common difference d = 3 -/
def ArithmeticSequence (n : ℕ) : ℤ := 3*n - 26

/-- The sum of the first n terms of the arithmetic sequence -/
def SequenceSum (n : ℕ) : ℤ := n * (ArithmeticSequence 1 + ArithmeticSequence n) / 2

theorem arithmetic_sequence_min_sum :
  (∀ m : ℕ, SequenceSum m ≥ SequenceSum 8) ∧
  SequenceSum 8 = -100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l3325_332555


namespace NUMINAMATH_CALUDE_ages_four_years_ago_l3325_332538

/-- Represents the ages of four people: Amar, Akbar, Anthony, and Alex -/
structure Ages :=
  (amar : ℕ)
  (akbar : ℕ)
  (anthony : ℕ)
  (alex : ℕ)

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.amar + ages.akbar + ages.anthony + ages.alex = 88 ∧
  (ages.amar - 4) + (ages.akbar - 4) + (ages.anthony - 4) = 66 ∧
  ages.amar = 2 * ages.alex ∧
  ages.akbar = ages.amar - 3

/-- The theorem to be proved -/
theorem ages_four_years_ago (ages : Ages) 
  (h : satisfies_conditions ages) : 
  (ages.amar - 4) + (ages.akbar - 4) + (ages.anthony - 4) + (ages.alex - 4) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ages_four_years_ago_l3325_332538


namespace NUMINAMATH_CALUDE_train_length_l3325_332515

/-- The length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmh = 72 → time_sec = 15 → length = (speed_kmh * 1000 / 3600) * time_sec → length = 300 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3325_332515


namespace NUMINAMATH_CALUDE_surface_area_of_six_cubes_l3325_332542

/-- Represents the configuration of 6 cubes fastened together -/
structure CubeConfiguration where
  numCubes : Nat
  edgeLength : ℝ
  numConnections : Nat

/-- Calculates the total surface area of the cube configuration -/
def totalSurfaceArea (config : CubeConfiguration) : ℝ :=
  (config.numCubes * 6 - 2 * config.numConnections) * config.edgeLength ^ 2

/-- Theorem stating that the total surface area of the given configuration is 26 square units -/
theorem surface_area_of_six_cubes :
  ∀ (config : CubeConfiguration),
    config.numCubes = 6 ∧
    config.edgeLength = 1 ∧
    config.numConnections = 10 →
    totalSurfaceArea config = 26 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_six_cubes_l3325_332542


namespace NUMINAMATH_CALUDE_triangle_solution_l3325_332533

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem setup
def triangle_problem (t : Triangle) (median_CM : Line) (altitude_BH : Line) : Prop :=
  t.A = (5, 1) ∧
  median_CM = ⟨2, -1, -5⟩ ∧
  altitude_BH = ⟨1, -2, -5⟩

-- Theorem statement
theorem triangle_solution (t : Triangle) (median_CM : Line) (altitude_BH : Line) 
  (h : triangle_problem t median_CM altitude_BH) :
  (∃ (line_AC : Line), line_AC = ⟨2, 1, -11⟩) ∧ 
  t.B = (-1, -3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_solution_l3325_332533


namespace NUMINAMATH_CALUDE_pyramid_top_value_l3325_332568

/-- Represents a three-level pyramid of numbers -/
structure NumberPyramid where
  bottomLeft : ℕ
  bottomRight : ℕ
  middleLeft : ℕ
  middleRight : ℕ
  top : ℕ

/-- Checks if a NumberPyramid is valid according to the sum rule -/
def isValidPyramid (p : NumberPyramid) : Prop :=
  p.middleLeft = p.bottomLeft ∧
  p.middleRight = p.bottomRight ∧
  p.top = p.middleLeft + p.middleRight

theorem pyramid_top_value (p : NumberPyramid) 
  (h1 : p.bottomLeft = 35)
  (h2 : p.bottomRight = 47)
  (h3 : isValidPyramid p) : 
  p.top = 82 := by
  sorry

#check pyramid_top_value

end NUMINAMATH_CALUDE_pyramid_top_value_l3325_332568


namespace NUMINAMATH_CALUDE_number_of_intersection_points_l3325_332550

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- Define the curve equation
def curve (x y : ℝ) : Prop := y^2 / 9 - (x * abs x) / 4 = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  y = line x ∧ curve x y

-- Theorem statement
theorem number_of_intersection_points :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    is_intersection_point p₁.1 p₁.2 ∧
    is_intersection_point p₂.1 p₂.2 ∧
    is_intersection_point p₃.1 p₃.2 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    ∀ (q : ℝ × ℝ), is_intersection_point q.1 q.2 → q = p₁ ∨ q = p₂ ∨ q = p₃ :=
by sorry

end NUMINAMATH_CALUDE_number_of_intersection_points_l3325_332550


namespace NUMINAMATH_CALUDE_unique_solution_trig_system_l3325_332566

theorem unique_solution_trig_system (x : ℝ) :
  (Real.arccos (3 * x) - Real.arcsin x = π / 6 ∧
   Real.arccos (3 * x) + Real.arcsin x = 5 * π / 6) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trig_system_l3325_332566


namespace NUMINAMATH_CALUDE_rectangle_area_l3325_332514

theorem rectangle_area (length width : ℝ) (h1 : length = Real.sqrt 6) (h2 : width = Real.sqrt 3) :
  length * width = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3325_332514


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l3325_332518

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1) + a

-- Define the given line
def given_line (x : ℝ) : ℝ := 2 * x - 6

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, tangent_line a x = given_line x) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_one_l3325_332518


namespace NUMINAMATH_CALUDE_algebra_test_female_count_l3325_332521

theorem algebra_test_female_count :
  ∀ (total_average : ℝ) (male_count : ℕ) (male_average female_average : ℝ),
    total_average = 90 →
    male_count = 8 →
    male_average = 85 →
    female_average = 92 →
    ∃ (female_count : ℕ),
      (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
      female_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_female_count_l3325_332521


namespace NUMINAMATH_CALUDE_egg_leftover_proof_l3325_332517

/-- The number of eggs left over when selling a given number of eggs in cartons of 10 -/
def leftover_eggs (total_eggs : ℕ) : ℕ :=
  total_eggs % 10

theorem egg_leftover_proof (john_eggs maria_eggs nikhil_eggs : ℕ) 
  (h1 : john_eggs = 45)
  (h2 : maria_eggs = 38)
  (h3 : nikhil_eggs = 29) :
  leftover_eggs (john_eggs + maria_eggs + nikhil_eggs) = 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_leftover_proof_l3325_332517


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3325_332509

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

def B : Set ℤ := {x | ∃ k : ℕ, k < 3 ∧ x = 2 * k + 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3325_332509


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3325_332506

theorem complex_fraction_equality : (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3325_332506


namespace NUMINAMATH_CALUDE_submerged_sphere_pressure_l3325_332574

/-- The total water pressure on a submerged sphere -/
theorem submerged_sphere_pressure
  (diameter : ℝ) (depth : ℝ) (ρ : ℝ) (g : ℝ) :
  diameter = 4 →
  depth = 3 →
  ρ > 0 →
  g > 0 →
  (∫ x in (-2 : ℝ)..2, 4 * π * ρ * g * (depth + x)) = 64 * π * ρ * g :=
by sorry

end NUMINAMATH_CALUDE_submerged_sphere_pressure_l3325_332574


namespace NUMINAMATH_CALUDE_allan_bought_three_balloons_l3325_332570

/-- The number of balloons Allan bought at the park -/
def balloons_bought_by_allan : ℕ := 3

/-- Allan's initial number of balloons -/
def allan_initial_balloons : ℕ := 2

/-- Jake's initial number of balloons -/
def jake_initial_balloons : ℕ := 6

theorem allan_bought_three_balloons :
  balloons_bought_by_allan = 3 ∧
  allan_initial_balloons = 2 ∧
  jake_initial_balloons = 6 ∧
  jake_initial_balloons = (allan_initial_balloons + balloons_bought_by_allan + 1) :=
by sorry

end NUMINAMATH_CALUDE_allan_bought_three_balloons_l3325_332570


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l3325_332532

def a (n : ℕ) : ℚ := 9/2 - n

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, a (n + 1) - a n = -1) ∧
  (Finset.sum (Finset.range 20) a = -120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l3325_332532


namespace NUMINAMATH_CALUDE_fourth_month_sale_proof_l3325_332565

/-- Calculates the sale in the fourth month given sales for other months and the average --/
def fourthMonthSale (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

theorem fourth_month_sale_proof (sale1 sale2 sale3 sale5 sale6 average : ℕ) 
  (h1 : sale1 = 3435)
  (h2 : sale2 = 3927)
  (h3 : sale3 = 3855)
  (h5 : sale5 = 3562)
  (h6 : sale6 = 1991)
  (h_avg : average = 3500) :
  fourthMonthSale sale1 sale2 sale3 sale5 sale6 average = 4230 := by
  sorry

#eval fourthMonthSale 3435 3927 3855 3562 1991 3500

end NUMINAMATH_CALUDE_fourth_month_sale_proof_l3325_332565


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3325_332582

theorem algebraic_expression_equality (a b : ℝ) (h : a^2 + 2*b^2 - 1 = 0) :
  (a - b)^2 + b*(2*a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3325_332582


namespace NUMINAMATH_CALUDE_inequality_proof_l3325_332597

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3325_332597


namespace NUMINAMATH_CALUDE_minimize_expression_l3325_332526

theorem minimize_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 2) (h3 : a + b = 3) :
  ∃ (min_a : ℝ), min_a = 2/3 ∧ 
    ∀ (x : ℝ), x > 0 → x + b = 3 → (4/x + 1/(b-2) ≥ 4/min_a + 1/(b-2)) :=
by sorry

end NUMINAMATH_CALUDE_minimize_expression_l3325_332526


namespace NUMINAMATH_CALUDE_parabola_translation_l3325_332558

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem parabola_translation (x : ℝ) :
  let original := Parabola.mk 3 0 0
  let translated := translate original 1 2
  translated.a * x^2 + translated.b * x + translated.c = 3 * (x + 1)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3325_332558


namespace NUMINAMATH_CALUDE_total_repair_time_l3325_332583

/-- Represents the time in minutes required for each repair task for different shoe types -/
structure ShoeRepairTime where
  buckle : ℕ
  strap : ℕ
  sole : ℕ

/-- Represents the number of shoes repaired in a session -/
structure SessionRepair where
  flat : ℕ
  sandal : ℕ
  highHeel : ℕ

/-- Calculates the total repair time for a given shoe type and quantity -/
def repairTime (time : ShoeRepairTime) (quantity : ℕ) : ℕ :=
  (time.buckle + time.strap + time.sole) * quantity

/-- Calculates the total repair time for a session -/
def sessionTime (flat : ShoeRepairTime) (sandal : ShoeRepairTime) (highHeel : ShoeRepairTime) (session : SessionRepair) : ℕ :=
  repairTime flat session.flat + repairTime sandal session.sandal + repairTime highHeel session.highHeel

theorem total_repair_time :
  let flat := ShoeRepairTime.mk 3 8 9
  let sandal := ShoeRepairTime.mk 4 5 0
  let highHeel := ShoeRepairTime.mk 6 12 10
  let session1 := SessionRepair.mk 6 4 3
  let session2 := SessionRepair.mk 4 7 5
  let breakTime := 15
  sessionTime flat sandal highHeel session1 + sessionTime flat sandal highHeel session2 + breakTime = 538 := by
  sorry

end NUMINAMATH_CALUDE_total_repair_time_l3325_332583


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l3325_332504

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_valid_integer : 
  (∀ m : ℕ, is_valid m → m ≤ 125) ∧ is_valid 125 :=
by sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l3325_332504


namespace NUMINAMATH_CALUDE_franks_fruits_l3325_332501

/-- The total number of fruits left after Frank's dog eats some -/
def fruits_left (apples_on_tree apples_on_ground oranges_on_tree oranges_on_ground apples_eaten oranges_eaten : ℕ) : ℕ :=
  (apples_on_tree + apples_on_ground - apples_eaten) + (oranges_on_tree + oranges_on_ground - oranges_eaten)

/-- Theorem stating the total number of fruits left in Frank's scenario -/
theorem franks_fruits :
  fruits_left 5 8 7 10 3 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_franks_fruits_l3325_332501


namespace NUMINAMATH_CALUDE_unique_valid_number_l3325_332578

def is_valid_number (n : ℕ) : Prop :=
  765400 ≤ n ∧ n ≤ 765499 ∧ n % 24 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 765455 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3325_332578


namespace NUMINAMATH_CALUDE_figure_102_squares_l3325_332547

/-- A function representing the number of non-overlapping unit squares in the nth figure -/
def g (n : ℕ) : ℕ := 2 * n^2 - 2 * n + 1

/-- Theorem stating that the 102nd figure contains 20605 non-overlapping unit squares -/
theorem figure_102_squares : g 102 = 20605 := by
  sorry

/-- Lemma verifying the given initial conditions -/
lemma initial_conditions :
  g 1 = 1 ∧ g 2 = 5 ∧ g 3 = 13 ∧ g 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_figure_102_squares_l3325_332547


namespace NUMINAMATH_CALUDE_c_share_is_36_l3325_332586

/-- Represents the rental information for a person --/
structure RentalInfo where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given person --/
def calculateShare (totalRent : ℚ) (totalOxMonths : ℕ) (info : RentalInfo) : ℚ :=
  totalRent * (info.oxen * info.months : ℚ) / totalOxMonths

theorem c_share_is_36 
  (totalRent : ℚ)
  (a b c : RentalInfo)
  (h_total_rent : totalRent = 140)
  (h_a : a = ⟨10, 7⟩)
  (h_b : b = ⟨12, 5⟩)
  (h_c : c = ⟨15, 3⟩) :
  calculateShare totalRent (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) c = 36 := by
  sorry

#check c_share_is_36

end NUMINAMATH_CALUDE_c_share_is_36_l3325_332586


namespace NUMINAMATH_CALUDE_new_average_after_multipliers_l3325_332505

theorem new_average_after_multipliers (original_list : List ℝ) 
  (h1 : original_list.length = 7)
  (h2 : original_list.sum / original_list.length = 20)
  (multipliers : List ℝ := [2, 3, 4, 5, 6, 7, 8]) :
  (List.zipWith (· * ·) original_list multipliers).sum / original_list.length = 100 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_multipliers_l3325_332505


namespace NUMINAMATH_CALUDE_amy_baskets_l3325_332522

/-- The number of baskets Amy will fill with candies -/
def num_baskets : ℕ :=
  let chocolate_bars := 5
  let mms := 7 * chocolate_bars
  let marshmallows := 6 * mms
  let total_candies := chocolate_bars + mms + marshmallows
  let candies_per_basket := 10
  total_candies / candies_per_basket

theorem amy_baskets : num_baskets = 25 := by
  sorry

end NUMINAMATH_CALUDE_amy_baskets_l3325_332522


namespace NUMINAMATH_CALUDE_strip_covering_theorem_l3325_332576

/-- A strip of width w -/
def Strip (w : ℝ) := Set (ℝ × ℝ)

/-- A set of points can be covered by a strip -/
def Coverable (S : Set (ℝ × ℝ)) (w : ℝ) :=
  ∃ (strip : Strip w), S ⊆ strip

/-- Main theorem -/
theorem strip_covering_theorem (S : Set (ℝ × ℝ)) (n : ℕ) 
  (h1 : Fintype S)
  (h2 : Fintype.card S = n)
  (h3 : n ≥ 3)
  (h4 : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → 
    Coverable {A, B, C} 1) :
  Coverable S 2 := by
  sorry

end NUMINAMATH_CALUDE_strip_covering_theorem_l3325_332576


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l3325_332587

theorem bakery_flour_usage (wheat_flour : Real) (total_flour : Real) (white_flour : Real) :
  wheat_flour = 0.2 →
  total_flour = 0.3 →
  white_flour = total_flour - wheat_flour →
  white_flour = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l3325_332587


namespace NUMINAMATH_CALUDE_water_added_to_container_l3325_332599

theorem water_added_to_container (capacity initial_percentage final_fraction : ℝ) 
  (h1 : capacity = 80)
  (h2 : initial_percentage = 0.30)
  (h3 : final_fraction = 3/4) : 
  capacity * (final_fraction - initial_percentage) = 36 := by
sorry

end NUMINAMATH_CALUDE_water_added_to_container_l3325_332599


namespace NUMINAMATH_CALUDE_divisibility_by_17_l3325_332539

theorem divisibility_by_17 (x y : ℤ) : 
  (∃ k : ℤ, 2*x + 3*y = 17*k) → (∃ m : ℤ, 9*x + 5*y = 17*m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_17_l3325_332539


namespace NUMINAMATH_CALUDE_cube_cutting_theorem_l3325_332540

/-- Represents a cube with an integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes -/
def CubeSet := List Cube

/-- Calculates the total volume of a set of cubes -/
def totalVolume (cubes : CubeSet) : ℕ :=
  cubes.map (fun c => c.edge ^ 3) |>.sum

/-- Represents a cutting and reassembly solution -/
structure Solution where
  pieces : ℕ
  isValid : Bool

/-- Theorem stating the existence of a valid solution -/
theorem cube_cutting_theorem (original : CubeSet) (target : CubeSet) : 
  (original = [Cube.mk 14, Cube.mk 10] ∧ 
   target = [Cube.mk 13, Cube.mk 11, Cube.mk 6] ∧
   totalVolume original = totalVolume target) →
  ∃ (sol : Solution), sol.pieces = 11 ∧ sol.isValid = true := by
  sorry

#check cube_cutting_theorem

end NUMINAMATH_CALUDE_cube_cutting_theorem_l3325_332540


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l3325_332513

-- Define the cost of a pencil in cents
def pencil_cost : ℕ := 12

-- Define the total amount paid by seventh graders in cents
def seventh_grade_total : ℕ := 192

-- Define the total amount paid by sixth graders in cents
def sixth_grade_total : ℕ := 252

-- Define the number of sixth graders
def total_sixth_graders : ℕ := 35

-- Theorem statement
theorem pencil_buyers_difference : 
  (sixth_grade_total / pencil_cost) - (seventh_grade_total / pencil_cost) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l3325_332513


namespace NUMINAMATH_CALUDE_distance_covered_l3325_332528

/-- Proves that the total distance covered is 8 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) (h1 : walking_speed = 4)
  (h2 : running_speed = 8) (h3 : total_time = 0.75) : ∃ (distance : ℝ),
  distance = 8 ∧ total_time = (distance / 2) / walking_speed + (distance / 2) / running_speed :=
by sorry

end NUMINAMATH_CALUDE_distance_covered_l3325_332528


namespace NUMINAMATH_CALUDE_juan_and_maria_distance_l3325_332563

/-- The combined distance covered by two runners given their speeds and times -/
def combined_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem stating that the combined distance of Juan and Maria is 658.5 miles -/
theorem juan_and_maria_distance :
  combined_distance 9.5 30 8.3 45 = 658.5 := by
  sorry

end NUMINAMATH_CALUDE_juan_and_maria_distance_l3325_332563


namespace NUMINAMATH_CALUDE_average_age_problem_l3325_332579

theorem average_age_problem (devin_age eden_age mom_age : ℕ) : 
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l3325_332579


namespace NUMINAMATH_CALUDE_H_surjective_l3325_332511

def H (x : ℝ) : ℝ := |3 * x + 1| - |x - 2|

theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l3325_332511


namespace NUMINAMATH_CALUDE_hotel_guests_count_l3325_332595

theorem hotel_guests_count (oates_count hall_count both_count : ℕ) 
  (ho : oates_count = 40)
  (hh : hall_count = 70)
  (hb : both_count = 10) :
  oates_count + hall_count - both_count = 100 := by
  sorry

end NUMINAMATH_CALUDE_hotel_guests_count_l3325_332595


namespace NUMINAMATH_CALUDE_brick_height_calculation_l3325_332534

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    brick length and width, and the number of bricks needed. -/
theorem brick_height_calculation (wall_length wall_width wall_thickness : ℝ)
                                 (brick_length brick_width : ℝ)
                                 (num_bricks : ℝ) :
  wall_length = 200 →
  wall_width = 300 →
  wall_thickness = 2 →
  brick_length = 25 →
  brick_width = 11 →
  num_bricks = 72.72727272727273 →
  (wall_length * wall_width * wall_thickness) / (brick_length * brick_width * num_bricks) = 6 :=
by sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l3325_332534


namespace NUMINAMATH_CALUDE_camping_activities_count_l3325_332554

/-- The number of campers who went rowing and hiking in total, considering both morning and afternoon sessions -/
def total_rowing_and_hiking (total_campers : ℕ)
  (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_swimming : ℕ)
  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ) : ℕ :=
  morning_rowing + afternoon_rowing + morning_hiking + afternoon_hiking

/-- Theorem stating that the total number of campers who went rowing and hiking is 79 -/
theorem camping_activities_count
  (total_campers : ℕ)
  (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_swimming : ℕ)
  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ)
  (h1 : total_campers = 80)
  (h2 : morning_rowing = 41)
  (h3 : morning_hiking = 4)
  (h4 : morning_swimming = 15)
  (h5 : afternoon_rowing = 26)
  (h6 : afternoon_hiking = 8) :
  total_rowing_and_hiking total_campers morning_rowing morning_hiking morning_swimming afternoon_rowing afternoon_hiking = 79 := by
  sorry

#check camping_activities_count

end NUMINAMATH_CALUDE_camping_activities_count_l3325_332554


namespace NUMINAMATH_CALUDE_exists_x_y_inequality_l3325_332546

theorem exists_x_y_inequality (f : ℝ → ℝ) : ∃ x y : ℝ, f (x - f y) > y * f x + x := by
  sorry

end NUMINAMATH_CALUDE_exists_x_y_inequality_l3325_332546


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_squared_l3325_332510

theorem sqrt_49_times_sqrt_25_squared : (Real.sqrt (49 * Real.sqrt 25))^2 = 245 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_squared_l3325_332510


namespace NUMINAMATH_CALUDE_light_flash_duration_l3325_332500

/-- Proves that if a light flashes every 15 seconds and flashes 240 times, the time taken is exactly one hour. -/
theorem light_flash_duration (flash_interval : ℕ) (total_flashes : ℕ) (seconds_per_hour : ℕ) : 
  flash_interval = 15 → 
  total_flashes = 240 → 
  seconds_per_hour = 3600 → 
  flash_interval * total_flashes = seconds_per_hour :=
by
  sorry

#check light_flash_duration

end NUMINAMATH_CALUDE_light_flash_duration_l3325_332500


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_five_l3325_332523

theorem reciprocal_of_negative_five :
  ∀ x : ℚ, x * (-5) = 1 → x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_five_l3325_332523


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3325_332512

theorem pipe_filling_time (t_b t_ab : ℝ) (h_b : t_b = 20) (h_ab : t_ab = 20/3) :
  let t_a := (t_b * t_ab) / (t_b - t_ab)
  t_a = 10 := by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3325_332512


namespace NUMINAMATH_CALUDE_basketball_game_result_l3325_332590

/-- Represents a basketball team --/
structure Team where
  initial_score : ℕ
  baskets_scored : ℕ
  basket_value : ℕ

/-- Calculates the final score of a team --/
def final_score (team : Team) : ℕ := team.initial_score + team.baskets_scored * team.basket_value

/-- The basketball game scenario --/
def basketball_game_scenario : Prop :=
  let hornets : Team := { initial_score := 86, baskets_scored := 2, basket_value := 2 }
  let fireflies : Team := { initial_score := 74, baskets_scored := 7, basket_value := 3 }
  final_score fireflies - final_score hornets = 5

/-- Theorem stating the result of the basketball game --/
theorem basketball_game_result : basketball_game_scenario := by sorry

end NUMINAMATH_CALUDE_basketball_game_result_l3325_332590


namespace NUMINAMATH_CALUDE_range_of_product_l3325_332557

def f (x : ℝ) := |x^2 + 2*x - 1|

theorem range_of_product (a b : ℝ) 
  (h1 : a < b) (h2 : b < -1) (h3 : f a = f b) :
  ∃ y, y ∈ Set.Ioo 0 2 ∧ y = (a + 1) * (b + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_product_l3325_332557


namespace NUMINAMATH_CALUDE_olivia_coin_device_l3325_332553

def coin_change (start : ℕ) (change : ℕ) (target : ℕ) : Prop :=
  ∃ k : ℕ, start + k * (change - 1) = target

theorem olivia_coin_device (targets : List ℕ := [492, 776, 1248, 1520, 1984]) :
  ∀ t ∈ targets, (coin_change 1 80 t ↔ t = 1984) := by sorry

end NUMINAMATH_CALUDE_olivia_coin_device_l3325_332553


namespace NUMINAMATH_CALUDE_chips_count_proof_l3325_332588

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla : ℕ) : ℕ :=
  viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla

theorem chips_count_proof :
  ∀ (viviana_vanilla susana_chocolate : ℕ),
    viviana_vanilla = 20 →
    susana_chocolate = 25 →
    ∃ (viviana_chocolate susana_vanilla : ℕ),
      viviana_chocolate = susana_chocolate + 5 ∧
      susana_vanilla = (3 * viviana_vanilla) / 4 ∧
      total_chips viviana_chocolate viviana_vanilla susana_chocolate susana_vanilla = 90 := by
  sorry

end NUMINAMATH_CALUDE_chips_count_proof_l3325_332588


namespace NUMINAMATH_CALUDE_intersection_probability_theorem_l3325_332531

/-- The probability that two randomly chosen diagonals intersect in a convex polygon with 2n + 1 vertices. -/
def intersection_probability (n : ℕ) : ℚ :=
  if n > 0 then
    (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2))
  else
    0

/-- Theorem: In a convex polygon with 2n + 1 vertices (n > 0), the probability that two randomly
    chosen diagonals intersect is n(2n - 1) / (3(2n^2 - n - 2)). -/
theorem intersection_probability_theorem (n : ℕ) (h : n > 0) :
  intersection_probability n = (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_probability_theorem_l3325_332531


namespace NUMINAMATH_CALUDE_sam_drew_age_multiple_l3325_332594

/-- Proves that in five years, Sam's age divided by Drew's age equals 3 -/
theorem sam_drew_age_multiple (drew_current_age sam_current_age : ℕ) : 
  drew_current_age = 12 →
  sam_current_age = 46 →
  (sam_current_age + 5) / (drew_current_age + 5) = 3 := by
sorry

end NUMINAMATH_CALUDE_sam_drew_age_multiple_l3325_332594


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_9_20_plus_11_20_l3325_332549

theorem sum_of_last_two_digits_9_20_plus_11_20 :
  (9^20 + 11^20) % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_9_20_plus_11_20_l3325_332549


namespace NUMINAMATH_CALUDE_passing_marks_calculation_l3325_332567

theorem passing_marks_calculation (T : ℝ) (P : ℝ) : 
  (0.20 * T = P - 40) → 
  (0.30 * T = P + 20) → 
  P = 160 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_calculation_l3325_332567


namespace NUMINAMATH_CALUDE_max_handshakes_correct_l3325_332575

/-- The number of men shaking hands -/
def n : ℕ := 40

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- The maximum number of handshakes without cyclic handshakes -/
def maxHandshakes : ℕ := n.choose k

theorem max_handshakes_correct :
  maxHandshakes = 780 := by sorry

end NUMINAMATH_CALUDE_max_handshakes_correct_l3325_332575


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3325_332529

/-- Represents a quadratic equation of the form x^2 - (m-3)x - m = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (m-3)*x - m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (m-3)^2 - 4*(-m)

/-- Represents the condition on the roots of the quadratic equation -/
def root_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 - x₁*x₂ = 13

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂) ∧
  (∀ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ root_condition x₁ x₂ →
    m = 4 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3325_332529


namespace NUMINAMATH_CALUDE_car_repair_cost_l3325_332598

theorem car_repair_cost (total_cost : ℝ) (num_parts : ℕ) (labor_rate : ℝ) (work_hours : ℝ)
  (h1 : total_cost = 220)
  (h2 : num_parts = 2)
  (h3 : labor_rate = 0.5)
  (h4 : work_hours = 6) :
  (total_cost - labor_rate * work_hours * 60) / num_parts = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_l3325_332598


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l3325_332527

/-- Represents the earnings from the aqua park --/
def aqua_park_earnings 
  (admission_cost tour_cost meal_cost souvenir_cost : ℕ) 
  (group1_size group2_size group3_size : ℕ) : ℕ :=
  let group1_total := group1_size * (admission_cost + tour_cost + meal_cost + souvenir_cost)
  let group2_total := group2_size * (admission_cost + meal_cost)
  let group3_total := group3_size * (admission_cost + tour_cost + souvenir_cost)
  group1_total + group2_total + group3_total

/-- Theorem stating the total earnings from all groups --/
theorem total_earnings_theorem : 
  aqua_park_earnings 12 6 10 8 10 15 8 = 898 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l3325_332527


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_2km_l3325_332535

def hulk_jump (n : ℕ) : ℝ :=
  if n = 0 then 0.5 else 2^(n - 1)

theorem hulk_jump_exceeds_2km : 
  (∀ k < 13, hulk_jump k ≤ 2000) ∧ 
  hulk_jump 13 > 2000 := by sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_2km_l3325_332535


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3325_332584

/-- The equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-6)^2 + (y-4)^2) = 14

/-- The two focal points of the conic section --/
def focal_point1 : ℝ × ℝ := (0, -2)
def focal_point2 : ℝ × ℝ := (6, 4)

/-- Theorem stating that the given equation describes an ellipse --/
theorem conic_is_ellipse : ∃ (a b : ℝ) (center : ℝ × ℝ), 
  a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), conic_equation x y ↔ 
    ((x - center.1) / a)^2 + ((y - center.2) / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3325_332584


namespace NUMINAMATH_CALUDE_base8_145_equals_101_in_base10_l3325_332525

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the base-8 number 145
def base8Number : List Nat := [5, 4, 1]

-- State the theorem
theorem base8_145_equals_101_in_base10 :
  base8ToBase10 base8Number = 101 := by
  sorry

end NUMINAMATH_CALUDE_base8_145_equals_101_in_base10_l3325_332525


namespace NUMINAMATH_CALUDE_alternate_seating_l3325_332543

theorem alternate_seating (B : ℕ) :
  (∃ (G : ℕ), G = 1 ∧ B > 0 ∧ B - 1 = 24) → B = 25 := by
  sorry

end NUMINAMATH_CALUDE_alternate_seating_l3325_332543


namespace NUMINAMATH_CALUDE_no_three_digit_sum_product_l3325_332541

theorem no_three_digit_sum_product : ∀ a b c : ℕ, 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  ¬(a + b + c = 100 * a + 10 * b + c - a * b * c) :=
by sorry


end NUMINAMATH_CALUDE_no_three_digit_sum_product_l3325_332541


namespace NUMINAMATH_CALUDE_cylindrical_containers_radius_l3325_332573

theorem cylindrical_containers_radius (h : ℝ) (r : ℝ) :
  h > 0 →
  (π * (8^2) * (4 * h) = π * r^2 * h) →
  r = 16 := by
sorry

end NUMINAMATH_CALUDE_cylindrical_containers_radius_l3325_332573


namespace NUMINAMATH_CALUDE_range_of_f_l3325_332537

def f (x : ℤ) : ℤ := x^2 + 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 1}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3325_332537


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3325_332580

theorem quadratic_equation_solutions :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3325_332580


namespace NUMINAMATH_CALUDE_exists_asymmetric_but_rotational_invariant_figure_l3325_332507

/-- A convex figure in a 2D plane. -/
structure ConvexFigure where
  -- We don't need to fully define the structure, just declare it exists
  dummy : Unit

/-- Represents a rotation in 2D space. -/
structure Rotation where
  angle : ℝ

/-- Checks if a figure has an axis of symmetry. -/
def hasAxisOfSymmetry (figure : ConvexFigure) : Prop :=
  sorry

/-- Applies a rotation to a figure. -/
def applyRotation (figure : ConvexFigure) (rotation : Rotation) : ConvexFigure :=
  sorry

/-- Checks if a figure is invariant under a given rotation. -/
def isInvariantUnderRotation (figure : ConvexFigure) (rotation : Rotation) : Prop :=
  applyRotation figure rotation = figure

/-- The main theorem: There exists a convex figure with no axis of symmetry
    but invariant under 120° rotation. -/
theorem exists_asymmetric_but_rotational_invariant_figure :
  ∃ (figure : ConvexFigure),
    ¬(hasAxisOfSymmetry figure) ∧
    isInvariantUnderRotation figure ⟨2 * Real.pi / 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_exists_asymmetric_but_rotational_invariant_figure_l3325_332507


namespace NUMINAMATH_CALUDE_fermat_sum_divisibility_l3325_332585

theorem fermat_sum_divisibility (x y z : ℤ) 
  (hx : ¬ 7 ∣ x) (hy : ¬ 7 ∣ y) (hz : ¬ 7 ∣ z)
  (h_sum : (7:ℤ)^3 ∣ (x^7 + y^7 + z^7)) :
  (7:ℤ)^2 ∣ (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_fermat_sum_divisibility_l3325_332585


namespace NUMINAMATH_CALUDE_number_problem_l3325_332530

theorem number_problem (x : ℚ) (h : x - (3/5) * x = 62) : x = 155 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3325_332530


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3325_332581

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), (∀ z ∈ s, Complex.abs z < 20 ∧ Complex.exp z = (z - 1) / (z + 1)) ∧ Finset.card s = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3325_332581


namespace NUMINAMATH_CALUDE_song_book_cost_l3325_332589

theorem song_book_cost (flute_cost music_stand_cost total_spent : ℚ)
  (h1 : flute_cost = 142.46)
  (h2 : music_stand_cost = 8.89)
  (h3 : total_spent = 158.35) :
  total_spent - (flute_cost + music_stand_cost) = 7.00 := by
  sorry

end NUMINAMATH_CALUDE_song_book_cost_l3325_332589


namespace NUMINAMATH_CALUDE_number_of_boys_l3325_332520

theorem number_of_boys (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  girls = 120 →
  boys + girls = total →
  3 * total = 8 * boys →
  boys = 72 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3325_332520


namespace NUMINAMATH_CALUDE_international_data_daily_cost_l3325_332508

def regular_plan_cost : ℚ := 175
def total_charges : ℚ := 210
def stay_duration : ℕ := 10

theorem international_data_daily_cost : 
  (total_charges - regular_plan_cost) / stay_duration = (35 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_international_data_daily_cost_l3325_332508


namespace NUMINAMATH_CALUDE_smallest_average_l3325_332593

-- Define the set of digits
def digits : Finset Nat := Finset.range 9

-- Define the property of a valid selection
def valid_selection (single_digits double_digits : Finset Nat) : Prop :=
  single_digits.card = 3 ∧
  double_digits.card = 6 ∧
  (single_digits ∪ double_digits) = digits ∧
  single_digits ∩ double_digits = ∅

-- Define the average of the resulting set of numbers
def average (single_digits double_digits : Finset Nat) : ℚ :=
  let single_sum := single_digits.sum id
  let double_sum := (double_digits.filter (· ≤ 3)).sum (· * 10) +
                    (double_digits.filter (· > 3)).sum id
  (single_sum + double_sum : ℚ) / 6

-- Theorem statement
theorem smallest_average :
  ∀ single_digits double_digits : Finset Nat,
    valid_selection single_digits double_digits →
    average single_digits double_digits ≥ 33/2 :=
sorry

end NUMINAMATH_CALUDE_smallest_average_l3325_332593


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3325_332524

theorem quadratic_equation_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*m*x₁ - m - 1 = 0) ∧ 
  (x₂^2 - 2*m*x₂ - m - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3325_332524


namespace NUMINAMATH_CALUDE_simplify_expression_l3325_332592

theorem simplify_expression :
  (-2)^2006 + (-1)^3007 + 1^3010 - (-2)^2007 = -2^2006 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3325_332592


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3325_332544

theorem absolute_value_equation_solution :
  ∃! y : ℝ, (|y - 4| + 3 * y = 14) :=
by
  -- The unique solution is y = 4.5
  use 4.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3325_332544


namespace NUMINAMATH_CALUDE_proposition_condition_l3325_332577

theorem proposition_condition (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end NUMINAMATH_CALUDE_proposition_condition_l3325_332577


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l3325_332559

theorem quadratic_equation_integer_solutions :
  ∀ x y : ℤ, 
    x^2 + 2*x*y + 3*y^2 - 2*x + y + 1 = 0 ↔ 
    ((x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨ (x = 3 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l3325_332559


namespace NUMINAMATH_CALUDE_hot_dog_price_l3325_332560

/-- The cost of a hamburger -/
def hamburger_cost : ℝ := sorry

/-- The cost of a hot dog -/
def hot_dog_cost : ℝ := sorry

/-- First day's purchase equation -/
axiom day1_equation : 3 * hamburger_cost + 4 * hot_dog_cost = 10

/-- Second day's purchase equation -/
axiom day2_equation : 2 * hamburger_cost + 3 * hot_dog_cost = 7

/-- Theorem stating that a hot dog costs 1 dollar -/
theorem hot_dog_price : hot_dog_cost = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dog_price_l3325_332560


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3325_332564

theorem sufficient_condition_for_inequality (a : ℝ) (h : a ≥ 5) :
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3325_332564


namespace NUMINAMATH_CALUDE_midpoint_polar_coordinates_l3325_332556

/-- The polar coordinates of the midpoint of the chord intercepted by two curves -/
theorem midpoint_polar_coordinates (ρ θ : ℝ) :
  (ρ * (Real.cos θ - Real.sin θ) + 2 = 0) →  -- Curve C₁
  (ρ = 2) →  -- Curve C₂
  ∃ (r θ' : ℝ), (r = Real.sqrt 2 ∧ θ' = 3 * Real.pi / 4 ∧
    r * Real.cos θ' = -1 ∧ r * Real.sin θ' = 1) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_polar_coordinates_l3325_332556


namespace NUMINAMATH_CALUDE_coeff_bound_squared_poly_l3325_332591

/-- A polynomial with non-negative coefficients where no coefficient exceeds p(0) -/
structure NonNegPolynomial (n : ℕ) where
  p : Polynomial ℝ
  degree_eq : p.degree = n
  non_neg_coeff : ∀ i, 0 ≤ p.coeff i
  coeff_bound : ∀ i, p.coeff i ≤ p.coeff 0

/-- The coefficient of x^(n+1) in p(x)^2 is at most p(1)^2 / 2 -/
theorem coeff_bound_squared_poly {n : ℕ} (p : NonNegPolynomial n) :
  (p.p ^ 2).coeff (n + 1) ≤ (p.p.eval 1) ^ 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_coeff_bound_squared_poly_l3325_332591


namespace NUMINAMATH_CALUDE_f_neg_three_l3325_332569

def f (x : ℝ) : ℝ := x^2 + x

theorem f_neg_three : f (-3) = 6 := by sorry

end NUMINAMATH_CALUDE_f_neg_three_l3325_332569


namespace NUMINAMATH_CALUDE_distance_from_apex_l3325_332516

/-- A right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Area of the smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of the larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- Theorem about the distance of the larger cross section from the apex -/
theorem distance_from_apex (pyramid : OctagonalPyramid)
  (h_area_small : pyramid.area_small = 256 * Real.sqrt 2)
  (h_area_large : pyramid.area_large = 576 * Real.sqrt 2)
  (h_distance : pyramid.distance_between = 10) :
  ∃ (d : ℝ), d = 30 ∧ d > 0 ∧ 
  d * d * pyramid.area_small = (d - pyramid.distance_between) * (d - pyramid.distance_between) * pyramid.area_large :=
sorry

end NUMINAMATH_CALUDE_distance_from_apex_l3325_332516


namespace NUMINAMATH_CALUDE_sales_discount_effect_l3325_332596

theorem sales_discount_effect (discount : ℝ) 
  (h1 : discount = 10)
  (h2 : (1 - discount / 100) * 1.12 = 1.008) : 
  discount = 10 := by
sorry

end NUMINAMATH_CALUDE_sales_discount_effect_l3325_332596
