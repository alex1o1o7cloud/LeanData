import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_four_l453_45313

theorem absolute_value_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_four_l453_45313


namespace NUMINAMATH_CALUDE_complex_sum_powers_l453_45362

theorem complex_sum_powers (x : ℂ) (h : x^2 + x + 1 = 0) :
  x^49 + x^50 + x^51 + x^52 + x^53 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l453_45362


namespace NUMINAMATH_CALUDE_a_range_l453_45346

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

def is_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → seq n < seq m

theorem a_range (a : ℝ) :
  (∀ n : ℕ+, is_increasing (λ n => f a n)) →
  a ∈ Set.Ioo 2 3 :=
sorry

end NUMINAMATH_CALUDE_a_range_l453_45346


namespace NUMINAMATH_CALUDE_inequality_proof_l453_45377

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) :
  a < 2*b - b^2/a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l453_45377


namespace NUMINAMATH_CALUDE_bee_return_theorem_l453_45382

/-- Represents a position on the hexagonal grid -/
structure HexPosition where
  x : ℤ
  y : ℤ

/-- Represents a move on the hexagonal grid -/
structure HexMove where
  direction : Fin 6
  length : ℕ

/-- Applies a move to a position -/
def applyMove (pos : HexPosition) (move : HexMove) : HexPosition :=
  sorry

/-- Applies a sequence of moves to a position -/
def applyMoves (pos : HexPosition) (moves : List HexMove) : HexPosition :=
  sorry

/-- Generates a sequence of moves for a given N -/
def generateMoves (N : ℕ) : List HexMove :=
  sorry

theorem bee_return_theorem (N : ℕ) (h : N ≥ 3) :
  ∃ (startPos : HexPosition),
    applyMoves startPos (generateMoves N) = startPos :=
  sorry

end NUMINAMATH_CALUDE_bee_return_theorem_l453_45382


namespace NUMINAMATH_CALUDE_cistern_initial_water_fraction_l453_45302

theorem cistern_initial_water_fraction 
  (pipe_a_fill_time : ℝ) 
  (pipe_b_fill_time : ℝ) 
  (combined_fill_time : ℝ) 
  (h1 : pipe_a_fill_time = 12) 
  (h2 : pipe_b_fill_time = 8) 
  (h3 : combined_fill_time = 14.4) : 
  ∃ x : ℝ, x = 2/3 ∧ 
    (1 / combined_fill_time = (1 - x) / pipe_a_fill_time + (1 - x) / pipe_b_fill_time) :=
by sorry

end NUMINAMATH_CALUDE_cistern_initial_water_fraction_l453_45302


namespace NUMINAMATH_CALUDE_algebraic_notation_correctness_l453_45356

/-- Rules for algebraic notation --/
structure AlgebraicNotationRules where
  no_multiplication_sign : Bool
  number_before_variable : Bool
  proper_fraction : Bool
  correct_negative_placement : Bool

/-- Check if an expression follows algebraic notation rules --/
def follows_algebraic_notation (expr : String) (rules : AlgebraicNotationRules) : Bool :=
  rules.no_multiplication_sign ∧ 
  rules.number_before_variable ∧ 
  rules.proper_fraction ∧ 
  rules.correct_negative_placement

/-- Given expressions --/
def expr_A : String := "a×5"
def expr_B : String := "a7"
def expr_C : String := "3½x"
def expr_D : String := "-⅞x"

theorem algebraic_notation_correctness :
  follows_algebraic_notation expr_D 
    {no_multiplication_sign := true, 
     number_before_variable := true, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_A 
    {no_multiplication_sign := false, 
     number_before_variable := false, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_B
    {no_multiplication_sign := true, 
     number_before_variable := false, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_C
    {no_multiplication_sign := true, 
     number_before_variable := true, 
     proper_fraction := false, 
     correct_negative_placement := true} :=
by sorry

end NUMINAMATH_CALUDE_algebraic_notation_correctness_l453_45356


namespace NUMINAMATH_CALUDE_smallest_fraction_l453_45370

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) : a / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l453_45370


namespace NUMINAMATH_CALUDE_power_function_range_l453_45374

-- Define the power function f
def f (x : ℝ) : ℝ := x^2

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem power_function_range (m : ℝ) : 
  (f (Real.sqrt 3) = 3) → 
  (∀ x ∈ Set.Icc m 2, g x ∈ Set.Icc 1 5) → 
  (m ∈ Set.Icc (-2) 0) :=
by sorry

end NUMINAMATH_CALUDE_power_function_range_l453_45374


namespace NUMINAMATH_CALUDE_jones_pants_count_l453_45360

/-- Represents the number of pants Mr. Jones has -/
def num_pants : ℕ := 40

/-- Represents the number of shirts Mr. Jones has for each pair of pants -/
def shirts_per_pants : ℕ := 6

/-- Represents the total number of pieces of clothes Mr. Jones owns -/
def total_clothes : ℕ := 280

/-- Theorem stating that the number of pants Mr. Jones has is 40 -/
theorem jones_pants_count :
  num_pants * (shirts_per_pants + 1) = total_clothes :=
by sorry

end NUMINAMATH_CALUDE_jones_pants_count_l453_45360


namespace NUMINAMATH_CALUDE_misha_current_money_l453_45326

/-- The amount of money Misha needs to earn -/
def additional_money : ℕ := 13

/-- The total amount Misha would have after earning the additional money -/
def total_money : ℕ := 47

/-- Misha's current money amount -/
def current_money : ℕ := total_money - additional_money

theorem misha_current_money : current_money = 34 := by
  sorry

end NUMINAMATH_CALUDE_misha_current_money_l453_45326


namespace NUMINAMATH_CALUDE_inequality_proof_l453_45309

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l453_45309


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l453_45355

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 0 → a^2 > 0) ↔ (∀ a : ℝ, a^2 ≤ 0 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l453_45355


namespace NUMINAMATH_CALUDE_subset_iff_positive_l453_45345

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x : ℝ | x + a > 0}

theorem subset_iff_positive (a : ℝ) : A ⊆ B a ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_positive_l453_45345


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l453_45350

theorem sqrt_expressions_equality : 
  (2 * Real.sqrt (2/3) - 3 * Real.sqrt (3/2) + Real.sqrt 24 = (7 * Real.sqrt 6) / 6) ∧
  (Real.sqrt (25/2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2 = (11 * Real.sqrt 2) / 2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l453_45350


namespace NUMINAMATH_CALUDE_dons_average_speed_l453_45335

theorem dons_average_speed 
  (ambulance_speed : ℝ) 
  (ambulance_time : ℝ) 
  (don_time : ℝ) 
  (h1 : ambulance_speed = 60) 
  (h2 : ambulance_time = 1/4) 
  (h3 : don_time = 1/2) : 
  (ambulance_speed * ambulance_time) / don_time = 30 := by
sorry

end NUMINAMATH_CALUDE_dons_average_speed_l453_45335


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l453_45306

theorem largest_n_binomial_equality : ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l453_45306


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt2_minus3_l453_45347

theorem quadratic_root_sqrt2_minus3 :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 7
  f (Real.sqrt 2 - 3) = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt2_minus3_l453_45347


namespace NUMINAMATH_CALUDE_minnow_count_l453_45387

theorem minnow_count (total : ℕ) (red green white : ℕ) : 
  (red : ℚ) / total = 2/5 →
  (green : ℚ) / total = 3/10 →
  white + red + green = total →
  red = 20 →
  white = 15 := by
sorry

end NUMINAMATH_CALUDE_minnow_count_l453_45387


namespace NUMINAMATH_CALUDE_largest_expression_l453_45386

theorem largest_expression : 
  (100 - 0 > 0 / 100) ∧ (100 - 0 > 0 * 100) := by sorry

end NUMINAMATH_CALUDE_largest_expression_l453_45386


namespace NUMINAMATH_CALUDE_functional_equation_solution_l453_45365

/-- A polynomial of degree 2015 -/
def Polynomial2015 := Polynomial ℝ

/-- An odd polynomial of degree 2015 -/
def OddPolynomial2015 := {Q : Polynomial2015 // ∀ x, Q.eval (-x) = -Q.eval x}

/-- The functional equation P(x) + P(1-x) = 1 -/
def SatisfiesFunctionalEquation (P : Polynomial2015) : Prop :=
  ∀ x, P.eval x + P.eval (1 - x) = 1

theorem functional_equation_solution :
  ∀ P : Polynomial2015, SatisfiesFunctionalEquation P →
  ∃ Q : OddPolynomial2015, ∀ x, P.eval x = Q.val.eval (1/2 - x) + 1/2 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l453_45365


namespace NUMINAMATH_CALUDE_abc_relationship_l453_45375

theorem abc_relationship : ∀ (a b c : ℕ),
  a = 3^44 → b = 4^33 → c = 5^22 → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_abc_relationship_l453_45375


namespace NUMINAMATH_CALUDE_p_shape_points_l453_45319

/-- Represents a "P" shape formed from a square -/
structure PShape :=
  (side_length : ℕ)

/-- Counts the number of distinct points on a "P" shape -/
def count_points (p : PShape) : ℕ :=
  3 * (p.side_length + 1) - 2

/-- Theorem stating the number of points on a "P" shape with side length 10 -/
theorem p_shape_points :
  let p : PShape := { side_length := 10 }
  count_points p = 31 := by sorry

end NUMINAMATH_CALUDE_p_shape_points_l453_45319


namespace NUMINAMATH_CALUDE_three_tangent_planes_l453_45369

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle in 3D space -/
structure EquilateralTriangle where
  vertices : List (ℝ × ℝ × ℝ)
  side_length : ℝ

/-- Configuration of three spheres whose centers form an equilateral triangle -/
structure SphereConfiguration where
  spheres : List Sphere
  triangle : EquilateralTriangle

/-- Returns the number of planes tangent to all spheres in the configuration -/
def count_tangent_planes (config : SphereConfiguration) : ℕ :=
  sorry

theorem three_tangent_planes (config : SphereConfiguration) :
  (config.spheres.length = 3) →
  (config.triangle.side_length = 11) →
  (config.spheres.map Sphere.radius = [3, 4, 6]) →
  (count_tangent_planes config = 3) :=
sorry

end NUMINAMATH_CALUDE_three_tangent_planes_l453_45369


namespace NUMINAMATH_CALUDE_marble_problem_l453_45312

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) (emma : ℚ)
  (h1 : angela = a)
  (h2 : brian = 2 * a)
  (h3 : caden = 3 * brian)
  (h4 : daryl = 5 * caden)
  (h5 : emma = 2 * daryl)
  (h6 : angela + brian + caden + daryl + emma = 212) :
  a = 212 / 99 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l453_45312


namespace NUMINAMATH_CALUDE_least_k_divisible_by_1680_l453_45307

theorem least_k_divisible_by_1680 :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (a b c d : ℕ), k = 2^a * 3^b * 5^c * 7^d) ∧
  (1680 ∣ k^4) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (x y z w : ℕ), m = 2^x * 3^y * 5^z * 7^w) →
    (1680 ∣ m^4) →
    m ≥ k) ∧
  k = 210 :=
sorry

end NUMINAMATH_CALUDE_least_k_divisible_by_1680_l453_45307


namespace NUMINAMATH_CALUDE_exponent_simplification_l453_45381

theorem exponent_simplification (a b : ℝ) (m n : ℤ) 
    (ha : a > 0) (hb : b > 0) (hm : m ≠ 0) (hn : n ≠ 0) :
  (a^m)^(1/n) = a^(m/n) ∧
  (a^(1/n))^(n/m) = a^(1/m) ∧
  (a^n * b)^(1/n) = a * b^(1/n) ∧
  (a^n * b^m)^(1/(m*n)) = a^(1/m) * b^(1/n) ∧
  (a^n / b^m)^(1/(m*n)) = (a^(1/m)) / (b^(1/n)) :=
by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l453_45381


namespace NUMINAMATH_CALUDE_circle_radius_l453_45351

/-- Given a circle with area P and circumference Q, if P/Q = 25, then the radius is 50 -/
theorem circle_radius (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) (h3 : P / Q = 25) :
  ∃ (r : ℝ), P = π * r^2 ∧ Q = 2 * π * r ∧ r = 50 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l453_45351


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l453_45363

/-- The probability of drawing a yellow ball from a bag with yellow, red, and white balls -/
theorem yellow_ball_probability (yellow red white : ℕ) : 
  yellow = 5 → red = 8 → white = 7 → 
  (yellow : ℚ) / (yellow + red + white) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l453_45363


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l453_45352

theorem quadratic_equations_solutions :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (x₁^2 - 6*x₁ + 5 = 0 ∧ x₂^2 - 6*x₂ + 5 = 0 ∧ x₁ = 5 ∧ x₂ = 1) ∧
    (3*x₃*(2*x₃ - 1) = 4*x₃ - 2 ∧ 3*x₄*(2*x₄ - 1) = 4*x₄ - 2 ∧ x₃ = 1/2 ∧ x₄ = 2/3) ∧
    (x₅^2 - 2*Real.sqrt 2*x₅ - 2 = 0 ∧ x₆^2 - 2*Real.sqrt 2*x₆ - 2 = 0 ∧ 
     x₅ = Real.sqrt 2 + 2 ∧ x₆ = Real.sqrt 2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l453_45352


namespace NUMINAMATH_CALUDE_part_one_part_two_l453_45359

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Part 1
theorem part_one : (Aᶜ ∪ B 1) = {x : ℝ | x ≤ -2 ∨ x > 1} := by sorry

-- Part 2
theorem part_two : A ⊆ B a → a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l453_45359


namespace NUMINAMATH_CALUDE_average_fishes_is_45_2_l453_45325

-- Define the number of lakes
def num_lakes : ℕ := 5

-- Define the number of fishes caught in each lake
def lake_marion : ℕ := 38
def lake_norman : ℕ := 52
def lake_wateree : ℕ := 27
def lake_wylie : ℕ := 45
def lake_keowee : ℕ := 64

-- Define the total number of fishes caught
def total_fishes : ℕ := lake_marion + lake_norman + lake_wateree + lake_wylie + lake_keowee

-- Define the average number of fishes caught per lake
def average_fishes : ℚ := total_fishes / num_lakes

-- Theorem statement
theorem average_fishes_is_45_2 : average_fishes = 45.2 := by
  sorry

end NUMINAMATH_CALUDE_average_fishes_is_45_2_l453_45325


namespace NUMINAMATH_CALUDE_probability_of_card_sequence_l453_45358

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Calculates the probability of the specified card sequence -/
def probabilityOfSequence : ℚ :=
  (CardsPerSuit : ℚ) / StandardDeck *
  (CardsPerSuit - 1) / (StandardDeck - 1) *
  CardsPerSuit / (StandardDeck - 2) *
  CardsPerSuit / (StandardDeck - 3)

/-- Theorem stating that the probability of drawing two hearts, 
    followed by one diamond, and then one club from a standard 
    52-card deck is equal to 39/63875 -/
theorem probability_of_card_sequence :
  probabilityOfSequence = 39 / 63875 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_card_sequence_l453_45358


namespace NUMINAMATH_CALUDE_leg_head_difference_l453_45336

/-- Represents the number of legs for a buffalo -/
def buffalo_legs : ℕ := 4

/-- Represents the number of legs for a duck -/
def duck_legs : ℕ := 2

/-- Represents the number of heads for any animal -/
def animal_head : ℕ := 1

/-- The number of buffaloes in the group -/
def num_buffaloes : ℕ := 12

theorem leg_head_difference (num_ducks : ℕ) :
  (num_buffaloes * buffalo_legs + num_ducks * duck_legs) -
  2 * (num_buffaloes * animal_head + num_ducks * animal_head) = 24 :=
by sorry

end NUMINAMATH_CALUDE_leg_head_difference_l453_45336


namespace NUMINAMATH_CALUDE_smallest_box_for_vase_l453_45322

/-- Represents a cylindrical vase -/
structure Vase where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube-shaped box -/
structure CubeBox where
  sideLength : ℝ

/-- The volume of a cube-shaped box -/
def boxVolume (box : CubeBox) : ℝ := box.sideLength ^ 3

/-- Predicate to check if a vase fits upright in a box -/
def fitsUpright (v : Vase) (b : CubeBox) : Prop :=
  v.height ≤ b.sideLength ∧ v.baseDiameter ≤ b.sideLength

theorem smallest_box_for_vase (v : Vase) (h1 : v.height = 15) (h2 : v.baseDiameter = 8) :
  ∃ (b : CubeBox), fitsUpright v b ∧
    (∀ (b' : CubeBox), fitsUpright v b' → boxVolume b ≤ boxVolume b') ∧
    boxVolume b = 3375 := by
  sorry

end NUMINAMATH_CALUDE_smallest_box_for_vase_l453_45322


namespace NUMINAMATH_CALUDE_gcf_of_60_and_90_l453_45390

theorem gcf_of_60_and_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_90_l453_45390


namespace NUMINAMATH_CALUDE_isosceles_triangle_x_values_l453_45342

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the square of the distance between two points in 3D space -/
def distanceSquared (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- Theorem: In an isosceles triangle ABC with vertices A(4, 1, 9), B(10, -1, 6), 
    and C(x, 4, 3), where BC is the base, the possible values of x are 2 and 6 -/
theorem isosceles_triangle_x_values :
  let A : Point3D := ⟨4, 1, 9⟩
  let B : Point3D := ⟨10, -1, 6⟩
  let C : Point3D := ⟨x, 4, 3⟩
  (distanceSquared A B = distanceSquared A C) → (x = 2 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_x_values_l453_45342


namespace NUMINAMATH_CALUDE_angle_H_measure_l453_45379

/-- Pentagon MATHS with specific angle conditions -/
structure Pentagon where
  M : ℝ  -- Measure of angle M
  A : ℝ  -- Measure of angle A
  T : ℝ  -- Measure of angle T
  H : ℝ  -- Measure of angle H
  S : ℝ  -- Measure of angle S
  angles_sum : M + A + T + H + S = 540
  equal_angles : M = T ∧ T = H
  supplementary : A + S = 180

/-- The measure of angle H in the specified pentagon is 120° -/
theorem angle_H_measure (p : Pentagon) : p.H = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_H_measure_l453_45379


namespace NUMINAMATH_CALUDE_smallest_bound_is_two_l453_45361

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ≥ 0) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂) ∧
  f 0 = 0 ∧ f 1 = 1

/-- The theorem stating that 2 is the smallest positive number c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_bound_is_two (f : ℝ → ℝ) (h : SatisfyingFunction f) :
  (∀ c > 0, (∀ x ∈ Set.Icc 0 1, f x ≤ c * x) → c ≥ 2) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_is_two_l453_45361


namespace NUMINAMATH_CALUDE_problem_statement_l453_45366

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The equation given in the problem -/
def equation (z a : ℂ) : Prop := (2 + i) * z = 1 + a * i^3

/-- A complex number is in Quadrant IV if its real part is positive and imaginary part is negative -/
def inQuadrantIV (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem problem_statement (z a : ℂ) :
  isPurelyImaginary z → equation z a → inQuadrantIV (a + z) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l453_45366


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l453_45332

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₁ = -16 and a₄ = 8, prove that a₇ = -4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a1 : a 1 = -16) 
    (h_a4 : a 4 = 8) : 
  a 7 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l453_45332


namespace NUMINAMATH_CALUDE_population_growth_proof_l453_45337

/-- The annual population growth rate -/
def annual_growth_rate : ℝ := 0.10

/-- The population after 2 years -/
def final_population : ℝ := 18150

/-- The present population of the town -/
def present_population : ℝ := 15000

/-- Theorem stating that the present population results in the final population after 2 years of growth -/
theorem population_growth_proof :
  present_population * (1 + annual_growth_rate)^2 = final_population :=
by sorry

end NUMINAMATH_CALUDE_population_growth_proof_l453_45337


namespace NUMINAMATH_CALUDE_power_72_in_terms_of_m_and_n_l453_45316

theorem power_72_in_terms_of_m_and_n (a m n : ℝ) 
  (h1 : 2^a = m) (h2 : 3^a = n) : 72^a = m^3 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_72_in_terms_of_m_and_n_l453_45316


namespace NUMINAMATH_CALUDE_c_share_is_36_l453_45371

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the total ox-months for a usage -/
def oxMonths (u : Usage) : ℕ := u.oxen * u.months

/-- Represents the rental situation -/
structure RentalSituation where
  usageA : Usage
  usageB : Usage
  usageC : Usage
  totalRent : ℚ

/-- The specific rental situation from the problem -/
def problemSituation : RentalSituation := {
  usageA := { oxen := 10, months := 7 }
  usageB := { oxen := 12, months := 5 }
  usageC := { oxen := 15, months := 3 }
  totalRent := 140
}

/-- Calculates C's share of the rent -/
def cShare (s : RentalSituation) : ℚ :=
  let totalUsage := oxMonths s.usageA + oxMonths s.usageB + oxMonths s.usageC
  let costPerOxMonth := s.totalRent / totalUsage
  (oxMonths s.usageC : ℚ) * costPerOxMonth

/-- Theorem stating that C's share in the problem situation is 36 -/
theorem c_share_is_36 : cShare problemSituation = 36 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_36_l453_45371


namespace NUMINAMATH_CALUDE_children_count_proof_l453_45317

theorem children_count_proof :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 ∧ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_children_count_proof_l453_45317


namespace NUMINAMATH_CALUDE_adam_bought_more_cat_food_l453_45396

/-- Represents the number of packages of cat food Adam bought -/
def cat_packages : ℕ := 9

/-- Represents the number of packages of dog food Adam bought -/
def dog_packages : ℕ := 7

/-- Represents the number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := 10

/-- Represents the number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- Calculates the difference between the total number of cans of cat food and dog food -/
def cans_difference : ℕ := 
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package

theorem adam_bought_more_cat_food : cans_difference = 55 := by
  sorry

end NUMINAMATH_CALUDE_adam_bought_more_cat_food_l453_45396


namespace NUMINAMATH_CALUDE_simplify_expression_l453_45357

-- Define the trigonometric identity
axiom trig_identity (θ : Real) : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1

-- Define the theorem
theorem simplify_expression : 
  2 - Real.sin (21 * π / 180) ^ 2 - Real.cos (21 * π / 180) ^ 2 
  + Real.sin (17 * π / 180) ^ 4 + Real.sin (17 * π / 180) ^ 2 * Real.cos (17 * π / 180) ^ 2 
  + Real.cos (17 * π / 180) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l453_45357


namespace NUMINAMATH_CALUDE_trees_represents_41225_l453_45311

-- Define the type for our digit mapping
def DigitMapping := Char → Option Nat

-- Define our specific mapping
def greatSuccessMapping : DigitMapping := fun c =>
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'S' => some 5
  | 'U' => some 6
  | 'C' => some 7
  | _ => none

-- Function to convert a string to a number using the mapping
def stringToNumber (s : String) (m : DigitMapping) : Option Nat :=
  s.foldr (fun c acc =>
    match acc, m c with
    | some n, some d => some (n * 10 + d)
    | _, _ => none
  ) (some 0)

-- Theorem statement
theorem trees_represents_41225 :
  stringToNumber "TREES" greatSuccessMapping = some 41225 := by
  sorry

end NUMINAMATH_CALUDE_trees_represents_41225_l453_45311


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l453_45305

def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l453_45305


namespace NUMINAMATH_CALUDE_max_area_triangle_l453_45367

/-- Given points A and B, and a circle with two symmetric points M and N,
    prove that the maximum area of triangle PAB is 3 + √2 --/
theorem max_area_triangle (k : ℝ) : 
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 2)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + k*x = 0}
  let symmetry_line := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  ∃ (M N : ℝ × ℝ), M ∈ circle ∧ N ∈ circle ∧ M ≠ N ∧
    (∃ (c : ℝ × ℝ), c ∈ symmetry_line ∧ 
      (M.1 - c.1)^2 + (M.2 - c.2)^2 = (N.1 - c.1)^2 + (N.2 - c.2)^2) →
  (⨆ (P : ℝ × ℝ) (h : P ∈ circle), 
    abs ((P.1 - A.1) * (B.2 - A.2) - (P.2 - A.2) * (B.1 - A.1)) / 2) = 3 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l453_45367


namespace NUMINAMATH_CALUDE_halloween_candy_division_l453_45330

/-- Represents the fraction of candy taken by each person --/
def candy_fraction (total : ℚ) (remaining : ℚ) (ratio : ℚ) : ℚ :=
  ratio * remaining / total

/-- The problem of dividing Halloween candy --/
theorem halloween_candy_division :
  let total := 1
  let al_ratio := 4 / 10
  let bert_ratio := 3 / 10
  let carl_ratio := 2 / 10
  let dana_ratio := 1 / 10
  
  let al_takes := candy_fraction total total al_ratio
  let bert_takes := candy_fraction total (total - al_takes) bert_ratio
  let carl_takes := candy_fraction total (total - al_takes - bert_takes) carl_ratio
  let dana_takes := candy_fraction total (total - al_takes - bert_takes - carl_takes) dana_ratio
  
  total - (al_takes + bert_takes + carl_takes + dana_takes) = 27 / 125 :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_division_l453_45330


namespace NUMINAMATH_CALUDE_exponential_range_condition_l453_45385

theorem exponential_range_condition (a : ℝ) :
  (∀ x > 0, a^x > 1) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_exponential_range_condition_l453_45385


namespace NUMINAMATH_CALUDE_quadratic_factorization_l453_45383

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c) * (x - d)) →
  4*d - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l453_45383


namespace NUMINAMATH_CALUDE_store_profit_percentage_l453_45388

/-- Calculates the profit percentage on items sold in February given the markups and discount -/
theorem store_profit_percentage (initial_markup : ℝ) (new_year_markup : ℝ) (february_discount : ℝ) :
  initial_markup = 0.20 →
  new_year_markup = 0.25 →
  february_discount = 0.20 →
  (1 + initial_markup + new_year_markup * (1 + initial_markup)) * (1 - february_discount) - 1 = 0.20 := by
  sorry

#check store_profit_percentage

end NUMINAMATH_CALUDE_store_profit_percentage_l453_45388


namespace NUMINAMATH_CALUDE_total_amount_distributed_l453_45395

/-- Represents the share distribution among w, x, y, and z -/
structure ShareDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The given share distribution -/
def given_distribution : ShareDistribution :=
  { w := 2
    x := 1.5
    y := 2.5
    z := 1.7 }

/-- Y's share in rupees -/
def y_share : ℝ := 48.50

/-- Theorem stating the total amount distributed -/
theorem total_amount_distributed : 
  let d := given_distribution
  let unit_value := y_share / d.y
  let total_units := d.w + d.x + d.y + d.z
  total_units * unit_value = 188.08 := by
  sorry

#check total_amount_distributed

end NUMINAMATH_CALUDE_total_amount_distributed_l453_45395


namespace NUMINAMATH_CALUDE_temperature_data_inconsistency_l453_45300

theorem temperature_data_inconsistency 
  (x_bar : ℝ) 
  (m : ℝ) 
  (S_squared : ℝ) 
  (hx : x_bar = 0) 
  (hm : m = 4) 
  (hS : S_squared = 15.917) : 
  ¬(|x_bar - m| ≤ Real.sqrt S_squared) := by
sorry

end NUMINAMATH_CALUDE_temperature_data_inconsistency_l453_45300


namespace NUMINAMATH_CALUDE_silverware_cost_l453_45380

/-- The cost of silverware given the conditions in the problem -/
theorem silverware_cost : 
  ∀ (silverware_cost dinner_plates_cost : ℝ),
  dinner_plates_cost = 0.5 * silverware_cost →
  silverware_cost + dinner_plates_cost = 30 →
  silverware_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_silverware_cost_l453_45380


namespace NUMINAMATH_CALUDE_equation_solution_l453_45308

/-- Given two equations with the same solution for x, prove the value of an expression -/
theorem equation_solution (m n x : ℝ) : 
  (m + 3) * x^(|m| - 2) + 6 * m = 0 →  -- First equation
  n * x - 5 = x * (3 - n) →            -- Second equation
  (|m| - 2 = 0) →                      -- Condition for first-degree equation
  (m + x)^2000 * (-m^2 * n + x * n^2) + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l453_45308


namespace NUMINAMATH_CALUDE_bottle_production_time_l453_45320

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 5 such machines will take 4 minutes to produce 900 bottles. -/
theorem bottle_production_time (rate : ℕ) (h1 : 6 * rate = 270) : 
  (900 : ℕ) / (5 * rate) = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_time_l453_45320


namespace NUMINAMATH_CALUDE_max_value_F_H_surjective_implies_s_value_l453_45339

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x

def F (x : ℝ) : ℝ := x^2 - x * f x

def H (s : ℝ) (x : ℝ) : ℝ :=
  if x ≥ s then x / (2 * Real.exp 1) else f x

theorem max_value_F :
  ∃ (x : ℝ), x ∈ Set.Icc (1/2) 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (1/2) 2 → F x ≥ F y ∧
  F x = 4 - Real.log 2 := by sorry

theorem H_surjective_implies_s_value (s : ℝ) :
  (∀ (k : ℝ), ∃ (x : ℝ), H s x = k) →
  s = Real.sqrt (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_max_value_F_H_surjective_implies_s_value_l453_45339


namespace NUMINAMATH_CALUDE_center_is_three_l453_45378

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements on the main diagonal (top-left to bottom-right) -/
def mainDiagonalSum (g : Grid) : ℕ :=
  (g 0 0).val + (g 1 1).val + (g 2 2).val

/-- The sum of elements on the other diagonal (top-right to bottom-left) -/
def otherDiagonalSum (g : Grid) : ℕ :=
  (g 0 2).val + (g 1 1).val + (g 2 0).val

/-- All numbers in the grid are distinct and from 1 to 9 -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ (g i j).val ∧ (g i j).val ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

theorem center_is_three (g : Grid) 
  (h1 : isValidGrid g)
  (h2 : mainDiagonalSum g = 6)
  (h3 : otherDiagonalSum g = 20) :
  (g 1 1).val = 3 := by
  sorry

end NUMINAMATH_CALUDE_center_is_three_l453_45378


namespace NUMINAMATH_CALUDE_pudding_weight_l453_45315

theorem pudding_weight (w : ℝ) 
  (h1 : 9/11 * w + 4 = w - (w - (9/11 * w + 4)))
  (h2 : 9/11 * w + 52 = w + (w - (9/11 * w + 4))) :
  w = 154 := by sorry

end NUMINAMATH_CALUDE_pudding_weight_l453_45315


namespace NUMINAMATH_CALUDE_solve_equation_l453_45353

theorem solve_equation (x : ℝ) : ((17.28 / x) / (3.6 * 0.2) = 2) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l453_45353


namespace NUMINAMATH_CALUDE_problem_solution_l453_45394

theorem problem_solution (x y : ℝ) (h1 : 0.2 * x = 200) (h2 : 0.3 * y = 150) :
  (0.8 * x - 0.5 * y) + 0.4 * (x + y) = 1150 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l453_45394


namespace NUMINAMATH_CALUDE_father_seven_times_son_age_l453_45398

/-- 
Given a father who is currently 38 years old and a son who is currently 14 years old,
this theorem proves that 10 years ago, the father was seven times as old as his son.
-/
theorem father_seven_times_son_age (father_age : Nat) (son_age : Nat) (years_ago : Nat) : 
  father_age = 38 → son_age = 14 → 
  (father_age - years_ago) = 7 * (son_age - years_ago) → 
  years_ago = 10 := by
sorry

end NUMINAMATH_CALUDE_father_seven_times_son_age_l453_45398


namespace NUMINAMATH_CALUDE_simplify_product_l453_45349

theorem simplify_product : (625 : ℝ) ^ (1/4) * (343 : ℝ) ^ (1/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l453_45349


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_point_one_l453_45338

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 0.1 is -0.1 -/
theorem opposite_of_point_one : opposite 0.1 = -0.1 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_point_one_l453_45338


namespace NUMINAMATH_CALUDE_egg_price_calculation_l453_45327

/-- Proves that the price of each egg is $0.20 given the conditions of the problem --/
theorem egg_price_calculation (total_eggs : ℕ) (crate_cost : ℚ) (eggs_left : ℕ) : 
  total_eggs = 30 → crate_cost = 5 → eggs_left = 5 → 
  (crate_cost / (total_eggs - eggs_left : ℚ)) = 0.20 := by
  sorry

#check egg_price_calculation

end NUMINAMATH_CALUDE_egg_price_calculation_l453_45327


namespace NUMINAMATH_CALUDE_stamps_for_heavier_envelopes_l453_45397

/-- Represents the number of stamps required for each weight category --/
def stamps_required (weight : ℕ) : ℕ :=
  if weight < 5 then 2
  else if weight ≤ 10 then 5
  else 7

/-- The total number of stamps purchased --/
def total_stamps : ℕ := 126

/-- The number of envelopes weighing less than 5 pounds --/
def light_envelopes : ℕ := 6

/-- Theorem stating that the total number of stamps used for envelopes weighing 
    5-10 lbs and >10 lbs is 114 --/
theorem stamps_for_heavier_envelopes :
  ∃ (medium heavy : ℕ),
    total_stamps = 
      light_envelopes * stamps_required 4 + 
      medium * stamps_required 5 + 
      heavy * stamps_required 11 ∧
    medium * stamps_required 5 + heavy * stamps_required 11 = 114 :=
sorry

end NUMINAMATH_CALUDE_stamps_for_heavier_envelopes_l453_45397


namespace NUMINAMATH_CALUDE_total_toys_l453_45391

theorem total_toys (bill_toys : ℕ) (hash_toys : ℕ) : 
  bill_toys = 60 → 
  hash_toys = (bill_toys / 2) + 9 → 
  bill_toys + hash_toys = 99 :=
by sorry

end NUMINAMATH_CALUDE_total_toys_l453_45391


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l453_45318

theorem trigonometric_calculations :
  (2 * Real.cos (60 * π / 180) + |1 - 2 * Real.sin (45 * π / 180)| + (1/2)^0 = Real.sqrt 2 + 1) ∧
  (Real.sqrt (1 - 2 * Real.tan (60 * π / 180) + Real.tan (60 * π / 180)^2) - Real.tan (60 * π / 180) = -1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l453_45318


namespace NUMINAMATH_CALUDE_pascal_triangle_47_l453_45329

/-- Pascal's Triangle contains the number 47 in exactly one row -/
theorem pascal_triangle_47 (p : ℕ) (h_prime : Nat.Prime p) (h_p : p = 47) : 
  (∃! n : ℕ, ∃ k : ℕ, Nat.choose n k = p) :=
sorry

end NUMINAMATH_CALUDE_pascal_triangle_47_l453_45329


namespace NUMINAMATH_CALUDE_base_6_to_base_3_conversion_l453_45321

def base_6_to_decimal (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def base_3_to_decimal (n : ℕ) : ℕ :=
  2 * 3^3 + 2 * 3^2 + 2 * 3^1 + 0 * 3^0

theorem base_6_to_base_3_conversion :
  base_6_to_decimal 210 = base_3_to_decimal 2220 := by
  sorry

end NUMINAMATH_CALUDE_base_6_to_base_3_conversion_l453_45321


namespace NUMINAMATH_CALUDE_same_price_at_12_sheets_l453_45373

/-- The price per sheet for John's Photo World -/
def johns_price_per_sheet : ℚ := 275/100

/-- The sitting fee for John's Photo World -/
def johns_sitting_fee : ℚ := 125

/-- The price per sheet for Sam's Picture Emporium -/
def sams_price_per_sheet : ℚ := 150/100

/-- The sitting fee for Sam's Picture Emporium -/
def sams_sitting_fee : ℚ := 140

/-- The total cost for John's Photo World given a number of sheets -/
def johns_total_cost (sheets : ℚ) : ℚ := johns_price_per_sheet * sheets + johns_sitting_fee

/-- The total cost for Sam's Picture Emporium given a number of sheets -/
def sams_total_cost (sheets : ℚ) : ℚ := sams_price_per_sheet * sheets + sams_sitting_fee

theorem same_price_at_12_sheets :
  ∃ (sheets : ℚ), sheets = 12 ∧ johns_total_cost sheets = sams_total_cost sheets :=
by sorry

end NUMINAMATH_CALUDE_same_price_at_12_sheets_l453_45373


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l453_45393

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume :
  let thickness : ℝ := 1/2
  let diameter : ℝ := 12
  let num_slices : ℕ := 8
  let radius : ℝ := diameter / 2
  let pizza_volume : ℝ := π * radius^2 * thickness
  let slice_volume : ℝ := pizza_volume / num_slices
  slice_volume = 9*π/4 := by sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l453_45393


namespace NUMINAMATH_CALUDE_students_without_A_l453_45324

theorem students_without_A (total : ℕ) (english_A : ℕ) (math_A : ℕ) (both_A : ℕ) : 
  total = 40 →
  english_A = 10 →
  math_A = 18 →
  both_A = 6 →
  total - (english_A + math_A - both_A) = 18 := by
sorry

end NUMINAMATH_CALUDE_students_without_A_l453_45324


namespace NUMINAMATH_CALUDE_square_circle_relation_l453_45333

theorem square_circle_relation (s r : ℝ) (h : s > 0) :
  4 * s = π * r^2 → r = 2 * Real.sqrt 2 / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_relation_l453_45333


namespace NUMINAMATH_CALUDE_num_different_results_is_1024_l453_45304

/-- The expression as a list of integers -/
def expression : List Int := [1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024]

/-- The number of terms in the expression that can have their sign changed -/
def num_changeable_terms : Nat := expression.length - 1

/-- The number of different results obtainable by placing parentheses in the expression -/
def num_different_results : Nat := 2^num_changeable_terms

theorem num_different_results_is_1024 : num_different_results = 1024 := by
  sorry

end NUMINAMATH_CALUDE_num_different_results_is_1024_l453_45304


namespace NUMINAMATH_CALUDE_stick_markings_l453_45328

/-- The number of unique markings on a one-foot stick marked in both 1/4 and 1/5 portions -/
def num_markings : ℕ := 9

/-- The set of markings for 1/4 portions -/
def quarter_markings : Set ℚ :=
  {0, 1/4, 1/2, 3/4, 1}

/-- The set of markings for 1/5 portions -/
def fifth_markings : Set ℚ :=
  {0, 1/5, 2/5, 3/5, 4/5, 1}

/-- The theorem stating that the number of unique markings is 9 -/
theorem stick_markings :
  (quarter_markings ∪ fifth_markings).ncard = num_markings :=
sorry

end NUMINAMATH_CALUDE_stick_markings_l453_45328


namespace NUMINAMATH_CALUDE_pebble_count_l453_45384

theorem pebble_count (white_pebbles : ℕ) (red_pebbles : ℕ) : 
  white_pebbles = 20 → 
  red_pebbles = white_pebbles / 2 → 
  white_pebbles + red_pebbles = 30 := by
sorry

end NUMINAMATH_CALUDE_pebble_count_l453_45384


namespace NUMINAMATH_CALUDE_equation_solution_l453_45354

theorem equation_solution :
  let f : ℝ → ℝ := λ x => 1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1)
  ∀ x : ℝ, f x = 5 ↔ x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l453_45354


namespace NUMINAMATH_CALUDE_rectangle_area_l453_45323

/-- Theorem: Area of a rectangle with specific properties -/
theorem rectangle_area (length : ℝ) (width : ℝ) : 
  length = 12 →
  width * 1.2 = length →
  length * width = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l453_45323


namespace NUMINAMATH_CALUDE_bridget_erasers_l453_45389

theorem bridget_erasers (initial final given : ℕ) : 
  initial = 8 → final = 11 → given = final - initial :=
by
  sorry

end NUMINAMATH_CALUDE_bridget_erasers_l453_45389


namespace NUMINAMATH_CALUDE_sector_central_angle_sine_l453_45344

theorem sector_central_angle_sine (r : ℝ) (arc_length : ℝ) (h1 : r = 2) (h2 : arc_length = 8 * Real.pi / 3) :
  Real.sin (arc_length / r) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_sine_l453_45344


namespace NUMINAMATH_CALUDE_cleaner_solution_calculation_l453_45341

/-- Represents the amount of cleaner solution needed for each type of stain -/
structure StainCleaner where
  dog : Nat
  cat : Nat
  bird : Nat
  rabbit : Nat
  fish : Nat

/-- Represents the number of stains for each type -/
structure StainCount where
  dog : Nat
  cat : Nat
  bird : Nat
  rabbit : Nat
  fish : Nat

def cleaner : StainCleaner :=
  { dog := 6, cat := 4, bird := 3, rabbit := 1, fish := 2 }

def weeklyStains : StainCount :=
  { dog := 10, cat := 8, bird := 5, rabbit := 1, fish := 3 }

def bottleSize : Nat := 64

/-- Calculates the total amount of cleaner solution needed -/
def totalSolutionNeeded (c : StainCleaner) (s : StainCount) : Nat :=
  c.dog * s.dog + c.cat * s.cat + c.bird * s.bird + c.rabbit * s.rabbit + c.fish * s.fish

/-- Calculates the additional amount of cleaner solution needed -/
def additionalSolutionNeeded (total : Nat) (bottleSize : Nat) : Nat :=
  if total > bottleSize then total - bottleSize else 0

theorem cleaner_solution_calculation :
  totalSolutionNeeded cleaner weeklyStains = 114 ∧
  additionalSolutionNeeded (totalSolutionNeeded cleaner weeklyStains) bottleSize = 50 := by
  sorry

end NUMINAMATH_CALUDE_cleaner_solution_calculation_l453_45341


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l453_45331

/-- The number of games in a single-elimination tournament -/
def num_games (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 17 teams and no ties, 
    the number of games played is 16 -/
theorem single_elimination_tournament_games : 
  num_games 17 = 16 := by sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l453_45331


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l453_45392

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃! (s : Finset ℕ), ∀ x, x ∈ s ↔ ∃ (a' b' : ℕ+), 
    Nat.gcd a' b' = x ∧ Nat.gcd a' b' * Nat.lcm a' b' = 360 ∧ s.card = 12) := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l453_45392


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l453_45343

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l453_45343


namespace NUMINAMATH_CALUDE_correct_observation_value_l453_45348

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : corrected_mean = 36.5)
  (h4 : wrong_value = 23) :
  let total_sum := n * initial_mean
  let corrected_sum := n * corrected_mean
  corrected_sum = total_sum - wrong_value + (total_sum - wrong_value + corrected_sum - total_sum) := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l453_45348


namespace NUMINAMATH_CALUDE_prob_998th_toss_heads_l453_45314

/-- A fair coin is a coin where the probability of getting heads is 1/2. -/
def fair_coin (coin : Type) : Prop :=
  ∃ (p : coin → ℝ), (∀ c, p c = 1/2) ∧ (∀ c, 0 ≤ p c ∧ p c ≤ 1)

/-- An independent event is an event whose probability is not affected by other events. -/
def independent_event (event : Type) (p : event → ℝ) : Prop :=
  ∀ (e₁ e₂ : event), p e₁ = p e₂

/-- The probability of getting heads on the 998th toss of a fair coin in a sequence of 1000 tosses. -/
theorem prob_998th_toss_heads (coin : Type) (toss : ℕ → coin) :
  fair_coin coin →
  independent_event coin (λ c => 1/2) →
  (λ c => 1/2) (toss 998) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_998th_toss_heads_l453_45314


namespace NUMINAMATH_CALUDE_rhombus_side_length_l453_45303

theorem rhombus_side_length 
  (diag1 diag2 : ℝ) 
  (m : ℝ) 
  (h1 : diag1^2 - 10*diag1 + m = 0)
  (h2 : diag2^2 - 10*diag2 + m = 0)
  (h3 : diag1 * diag2 / 2 = 11) :
  ∃ (side : ℝ), side^2 = 14 ∧ 
    side = Real.sqrt ((diag1/2)^2 + (diag2/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l453_45303


namespace NUMINAMATH_CALUDE_books_from_library_l453_45368

/-- The number of books initially obtained from the library -/
def initial_books : ℕ := 54

/-- The number of additional books obtained from the library -/
def additional_books : ℕ := 23

/-- The total number of books obtained from the library -/
def total_books : ℕ := initial_books + additional_books

theorem books_from_library : total_books = 77 := by
  sorry

end NUMINAMATH_CALUDE_books_from_library_l453_45368


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l453_45376

theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 2323 → rate = 8 → time = 5 →
  (principal * rate * time) / 100 = 1861.84 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l453_45376


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l453_45340

theorem modular_congruence_unique_solution : ∃! m : ℤ, 0 ≤ m ∧ m < 31 ∧ 79453 ≡ m [ZMOD 31] := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l453_45340


namespace NUMINAMATH_CALUDE_a_power_b_equals_sixteen_l453_45364

theorem a_power_b_equals_sixteen (a b : ℝ) : (a - 4)^2 + |2 - b| = 0 → a^b = 16 := by
  sorry

end NUMINAMATH_CALUDE_a_power_b_equals_sixteen_l453_45364


namespace NUMINAMATH_CALUDE_team_games_count_l453_45334

/-- Proves that a team with the given win percentages played 175 games in total -/
theorem team_games_count (first_hundred_win_rate : Real) 
                          (remaining_win_rate : Real)
                          (total_win_rate : Real)
                          (h1 : first_hundred_win_rate = 0.85)
                          (h2 : remaining_win_rate = 0.5)
                          (h3 : total_win_rate = 0.7) : 
  ∃ (total_games : ℕ), total_games = 175 ∧ 
    (first_hundred_win_rate * 100 + remaining_win_rate * (total_games - 100)) / total_games = total_win_rate :=
by
  sorry


end NUMINAMATH_CALUDE_team_games_count_l453_45334


namespace NUMINAMATH_CALUDE_trigonometric_identities_l453_45399

theorem trigonometric_identities (θ : ℝ) :
  (2 * Real.cos ((3 / 2) * Real.pi + θ) + Real.cos (Real.pi + θ)) /
  (3 * Real.sin (Real.pi - θ) + 2 * Real.sin ((5 / 2) * Real.pi + θ)) = 1 / 5 →
  (Real.tan θ = 3 / 13 ∧
   Real.sin θ ^ 2 + 3 * Real.sin θ * Real.cos θ = 20160 / 28561) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l453_45399


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l453_45310

theorem simplify_fraction_multiplication : (123 : ℚ) / 9999 * 41 = 1681 / 3333 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l453_45310


namespace NUMINAMATH_CALUDE_problem_statement_l453_45372

theorem problem_statement : 
  |1 - Real.sqrt 3| + 2 * Real.cos (30 * π / 180) - Real.sqrt 12 - 2023 = -2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l453_45372


namespace NUMINAMATH_CALUDE_parking_space_savings_l453_45301

/-- The cost of renting a parking space for one week in dollars -/
def weekly_cost : ℕ := 10

/-- The cost of renting a parking space for one month in dollars -/
def monthly_cost : ℕ := 24

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The savings in dollars when renting a parking space by the month instead of by the week for a year -/
theorem parking_space_savings : 
  weeks_per_year * weekly_cost - months_per_year * monthly_cost = 232 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_savings_l453_45301
