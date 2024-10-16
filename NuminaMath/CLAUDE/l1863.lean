import Mathlib

namespace NUMINAMATH_CALUDE_product_evaluation_l1863_186326

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1863_186326


namespace NUMINAMATH_CALUDE_increase_by_fifty_percent_l1863_186320

theorem increase_by_fifty_percent (initial : ℕ) (increase : ℚ) (result : ℕ) : 
  initial = 80 → increase = 1/2 → result = initial + (initial * increase) → result = 120 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_fifty_percent_l1863_186320


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1863_186360

theorem division_multiplication_problem : (150 : ℚ) / ((30 : ℚ) / 3) * 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1863_186360


namespace NUMINAMATH_CALUDE_chromosome_replication_not_in_prophase_i_l1863_186352

-- Define the events that can occur during cell division
inductive CellDivisionEvent
  | ChromosomeReplication
  | ChromosomeShortening
  | HomologousPairing
  | CrossingOver

-- Define the phases of meiosis
inductive MeiosisPhase
  | Interphase
  | ProphaseI
  | OtherPhases

-- Define a function that determines if an event occurs in a given phase
def occurs_in (event : CellDivisionEvent) (phase : MeiosisPhase) : Prop := sorry

-- State the theorem
theorem chromosome_replication_not_in_prophase_i :
  occurs_in CellDivisionEvent.ChromosomeReplication MeiosisPhase.Interphase →
  occurs_in CellDivisionEvent.ChromosomeShortening MeiosisPhase.ProphaseI →
  occurs_in CellDivisionEvent.HomologousPairing MeiosisPhase.ProphaseI →
  occurs_in CellDivisionEvent.CrossingOver MeiosisPhase.ProphaseI →
  ¬ occurs_in CellDivisionEvent.ChromosomeReplication MeiosisPhase.ProphaseI :=
by
  sorry

end NUMINAMATH_CALUDE_chromosome_replication_not_in_prophase_i_l1863_186352


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_one_l1863_186382

-- Define the points
def A : ℝ × ℝ := (0, -3)
def B : ℝ × ℝ := (3, 3)
def C : ℝ → ℝ × ℝ := λ x ↦ (x, -1)

-- Define the vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC (x : ℝ) : ℝ × ℝ := ((C x).1 - A.1, (C x).2 - A.2)

-- Theorem statement
theorem parallel_vectors_imply_x_equals_one :
  ∀ x : ℝ, (∃ k : ℝ, AB = k • (AC x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_one_l1863_186382


namespace NUMINAMATH_CALUDE_remaining_coin_value_l1863_186334

/-- Represents the number and type of coins --/
structure Coins where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total value of coins in cents --/
def coinValue (c : Coins) : Nat :=
  c.quarters * 25 + c.dimes * 10 + c.nickels * 5

/-- Represents Olivia's initial coins --/
def initialCoins : Coins :=
  { quarters := 11, dimes := 15, nickels := 7 }

/-- Represents the coins spent on purchases --/
def purchasedCoins : Coins :=
  { quarters := 1, dimes := 8, nickels := 3 }

/-- Calculates the remaining coins after purchases --/
def remainingCoins (initial : Coins) (purchased : Coins) : Coins :=
  { quarters := initial.quarters - purchased.quarters,
    dimes := initial.dimes - purchased.dimes,
    nickels := initial.nickels - purchased.nickels }

theorem remaining_coin_value :
  coinValue (remainingCoins initialCoins purchasedCoins) = 340 := by
  sorry


end NUMINAMATH_CALUDE_remaining_coin_value_l1863_186334


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1863_186369

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1863_186369


namespace NUMINAMATH_CALUDE_extreme_points_count_l1863_186391

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- Define what an extreme point is
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≠ f x

-- State the theorem
theorem extreme_points_count :
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = f_prime x) ∧ 
  (∃ (a b : ℝ), a ≠ b ∧ 
    is_extreme_point f a ∧ 
    is_extreme_point f b ∧ 
    ∀ c, is_extreme_point f c → (c = a ∨ c = b)) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_count_l1863_186391


namespace NUMINAMATH_CALUDE_cafeteria_pies_l1863_186356

theorem cafeteria_pies (total_apples : Real) (handed_out : Real) (apples_per_pie : Real) 
  (h1 : total_apples = 135.5)
  (h2 : handed_out = 89.75)
  (h3 : apples_per_pie = 5.25) :
  ⌊(total_apples - handed_out) / apples_per_pie⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l1863_186356


namespace NUMINAMATH_CALUDE_smallest_number_l1863_186357

theorem smallest_number (S : Set ℕ) (h : S = {10, 11, 12, 13, 14}) : 
  ∃ n ∈ S, ∀ m ∈ S, n ≤ m ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1863_186357


namespace NUMINAMATH_CALUDE_clothing_pricing_solution_l1863_186380

/-- Represents the pricing strategy for a piece of clothing --/
structure ClothingPricing where
  markedPrice : ℝ
  costPrice : ℝ

/-- Defines the conditions for the clothing pricing problem --/
def validPricing (p : ClothingPricing) : Prop :=
  (0.5 * p.markedPrice + 20 = p.costPrice) ∧ 
  (0.8 * p.markedPrice - 40 = p.costPrice)

/-- Theorem stating the unique solution to the clothing pricing problem --/
theorem clothing_pricing_solution :
  ∃! p : ClothingPricing, validPricing p ∧ p.markedPrice = 200 ∧ p.costPrice = 120 := by
  sorry


end NUMINAMATH_CALUDE_clothing_pricing_solution_l1863_186380


namespace NUMINAMATH_CALUDE_evaluate_expression_l1863_186339

theorem evaluate_expression : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1863_186339


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l1863_186363

/-- A geometric sequence with a specific sum formula -/
structure GeometricSequence where
  a : ℕ → ℝ
  sum : ℕ → ℝ
  sum_formula : ∀ n, sum n = 3^(n + 1) + a 1
  is_geometric : ∀ n, a (n + 2) * a n = (a (n + 1))^2

/-- The value of 'a' in the sum formula is -3 -/
theorem geometric_sequence_sum_constant (seq : GeometricSequence) : seq.a 1 - 9 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l1863_186363


namespace NUMINAMATH_CALUDE_m_less_than_one_l1863_186337

/-- Given that the solution set of |x| + |x-1| > m is ℝ and 
    f(x) = -(7-3m)^x is decreasing on ℝ, prove that m < 1 -/
theorem m_less_than_one (m : ℝ) 
  (h1 : ∀ x : ℝ, |x| + |x - 1| > m)
  (h2 : Monotone (fun x => -(7 - 3*m)^x)) : 
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_one_l1863_186337


namespace NUMINAMATH_CALUDE_a_less_than_b_l1863_186346

-- Define the function f
def f (x m : ℝ) : ℝ := -4 * x^2 + 8 * x + m

-- State the theorem
theorem a_less_than_b (m : ℝ) (a b : ℝ) 
  (h1 : f (-2) m = a) 
  (h2 : f 3 m = b) : 
  a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l1863_186346


namespace NUMINAMATH_CALUDE_equilateral_triangle_roots_l1863_186398

theorem equilateral_triangle_roots (a b z₁ z₂ : ℂ) : 
  (z₁^2 + a*z₁ + b = 0) → 
  (z₂^2 + a*z₂ + b = 0) → 
  (∃ ω : ℂ, ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁) →
  a^2 / b = 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_roots_l1863_186398


namespace NUMINAMATH_CALUDE_area_of_triangle_A_l1863_186375

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents the folded state of the parallelogram -/
structure FoldedParallelogram :=
  (original : Parallelogram)
  (A' : Point)
  (K : Point)

/-- The area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := 27

/-- The ratio of BK to KC -/
def BK_KC_ratio : ℝ × ℝ := (3, 2)

/-- The area of triangle A'KC -/
def triangleA'KC_area (fp : FoldedParallelogram) : ℝ := sorry

theorem area_of_triangle_A'KC 
  (p : Parallelogram)
  (fp : FoldedParallelogram)
  (h1 : fp.original = p)
  (h2 : parallelogramArea p = 27)
  (h3 : BK_KC_ratio = (3, 2)) :
  triangleA'KC_area fp = 3.6 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_A_l1863_186375


namespace NUMINAMATH_CALUDE_woojung_high_school_students_l1863_186341

theorem woojung_high_school_students (first_year : ℕ) (non_first_year : ℕ) : 
  non_first_year = 954 → 
  first_year = non_first_year - 468 → 
  first_year + non_first_year = 1440 := by
sorry

end NUMINAMATH_CALUDE_woojung_high_school_students_l1863_186341


namespace NUMINAMATH_CALUDE_min_floor_sum_l1863_186319

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l1863_186319


namespace NUMINAMATH_CALUDE_complex_number_with_purely_imaginary_square_plus_three_l1863_186336

theorem complex_number_with_purely_imaginary_square_plus_three :
  ∃ (z : ℂ), (∀ (x : ℝ), (z^2 + 3).re = x → x = 0) ∧ z = (1 : ℂ) + (2 : ℂ) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_with_purely_imaginary_square_plus_three_l1863_186336


namespace NUMINAMATH_CALUDE_diagonal_sum_lower_bound_l1863_186318

/-- Given a convex quadrilateral ABCD with sides a, b, c, d and diagonals x, y,
    where a is the smallest side, prove that x + y ≥ (1 + √3)a -/
theorem diagonal_sum_lower_bound (a b c d x y : ℝ) :
  a > 0 →
  b ≥ a →
  c ≥ a →
  d ≥ a →
  x ≥ a →
  y ≥ a →
  x + y ≥ (1 + Real.sqrt 3) * a :=
by sorry

end NUMINAMATH_CALUDE_diagonal_sum_lower_bound_l1863_186318


namespace NUMINAMATH_CALUDE_probability_red_in_middle_l1863_186354

/- Define the types of rosebushes -/
inductive Rosebush
| Red
| White

/- Define a row of rosebushes -/
def Row := List Rosebush

/- Define a function to check if the middle two rosebushes are red -/
def middleTwoAreRed (row : Row) : Bool :=
  match row with
  | [_, Rosebush.Red, Rosebush.Red, _] => true
  | _ => false

/- Define a function to generate all possible arrangements -/
def allArrangements : List Row :=
  sorry

/- Define a function to count arrangements with red rosebushes in the middle -/
def countRedInMiddle (arrangements : List Row) : Nat :=
  sorry

/- Theorem statement -/
theorem probability_red_in_middle :
  let arrangements := allArrangements
  let total := arrangements.length
  let favorable := countRedInMiddle arrangements
  (favorable : ℚ) / total = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_in_middle_l1863_186354


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1863_186396

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1863_186396


namespace NUMINAMATH_CALUDE_system_solution_l1863_186340

theorem system_solution (x y : ℝ) : x = 1 ∧ y = -2 → x + y = -1 ∧ x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1863_186340


namespace NUMINAMATH_CALUDE_justina_tallest_l1863_186367

-- Define a type for people
inductive Person : Type
  | Gisa : Person
  | Henry : Person
  | Ivan : Person
  | Justina : Person
  | Katie : Person

-- Define a height function
variable (height : Person → ℝ)

-- Define the conditions
axiom gisa_taller_than_henry : height Person.Gisa > height Person.Henry
axiom gisa_shorter_than_justina : height Person.Gisa < height Person.Justina
axiom ivan_taller_than_katie : height Person.Ivan > height Person.Katie
axiom ivan_shorter_than_gisa : height Person.Ivan < height Person.Gisa

-- Theorem to prove
theorem justina_tallest : 
  ∀ p : Person, height Person.Justina ≥ height p :=
sorry

end NUMINAMATH_CALUDE_justina_tallest_l1863_186367


namespace NUMINAMATH_CALUDE_ellipse_region_area_l1863_186358

/-- The area of the region formed by all points on ellipses passing through (√3, 1) where y ≥ 1 -/
theorem ellipse_region_area :
  ∀ a b : ℝ,
  a ≥ b ∧ b > 0 →
  (3 / a^2) + (1 / b^2) = 1 →
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ y ≥ 1) →
  (∃ area : ℝ, area = 4 * Real.pi / 3 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_region_area_l1863_186358


namespace NUMINAMATH_CALUDE_inconsistent_equation_l1863_186344

theorem inconsistent_equation : ¬ (3 * (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2400.0000000000005) := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_equation_l1863_186344


namespace NUMINAMATH_CALUDE_xy_value_from_inequality_l1863_186370

theorem xy_value_from_inequality (x y : ℝ) :
  2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2) →
  x * y = -9/4 := by
sorry

end NUMINAMATH_CALUDE_xy_value_from_inequality_l1863_186370


namespace NUMINAMATH_CALUDE_number_puzzle_l1863_186338

theorem number_puzzle : ∃ x : ℤ, x + 2 - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1863_186338


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_first_term_l1863_186342

/-- An arithmetic sequence with common difference 2 where a_1, a_2, and a_4 form a geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ 
  (a 2)^2 = a 1 * a 4

theorem arithmetic_geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h : ArithmeticGeometricSequence a) : 
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_first_term_l1863_186342


namespace NUMINAMATH_CALUDE_late_secondary_spermatocyte_homomorphic_l1863_186390

-- Define the stages of meiosis
inductive MeiosisStage
  | PrimaryMidFirst
  | PrimaryLateFirst
  | SecondaryMidSecond
  | SecondaryLateSecond

-- Define the types of sex chromosome pairs
inductive SexChromosomePair
  | Heteromorphic
  | Homomorphic

-- Define a function that determines the sex chromosome pair for each stage
def sexChromosomePairAtStage (stage : MeiosisStage) : SexChromosomePair :=
  match stage with
  | MeiosisStage.PrimaryMidFirst => SexChromosomePair.Heteromorphic
  | MeiosisStage.PrimaryLateFirst => SexChromosomePair.Heteromorphic
  | MeiosisStage.SecondaryMidSecond => SexChromosomePair.Heteromorphic
  | MeiosisStage.SecondaryLateSecond => SexChromosomePair.Homomorphic

-- State the theorem
theorem late_secondary_spermatocyte_homomorphic :
  ∀ (stage : MeiosisStage),
    sexChromosomePairAtStage stage = SexChromosomePair.Homomorphic
    ↔ stage = MeiosisStage.SecondaryLateSecond :=
by sorry

end NUMINAMATH_CALUDE_late_secondary_spermatocyte_homomorphic_l1863_186390


namespace NUMINAMATH_CALUDE_benjamin_weekly_miles_l1863_186317

/-- Calculates the total miles Benjamin walks in a week --/
def total_miles_walked (work_distance : ℕ) (dog_walk_distance : ℕ) (friend_distance : ℕ) (store_distance : ℕ) : ℕ :=
  let work_trips := 2 * work_distance * 5
  let dog_walks := 2 * dog_walk_distance * 7
  let friend_visit := 2 * friend_distance
  let store_trips := 2 * store_distance * 2
  work_trips + dog_walks + friend_visit + store_trips

/-- Theorem stating that Benjamin walks 102 miles in a week --/
theorem benjamin_weekly_miles :
  total_miles_walked 6 2 1 3 = 102 := by
  sorry

#eval total_miles_walked 6 2 1 3

end NUMINAMATH_CALUDE_benjamin_weekly_miles_l1863_186317


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1863_186349

theorem min_value_quadratic_sum (a b c : ℝ) (h : 2*a + 2*b + c = 8) :
  (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ≥ 49/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1863_186349


namespace NUMINAMATH_CALUDE_halfDollarProbabilityIs3_16_l1863_186330

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | HalfDollar
  | Quarter

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.HalfDollar => 50
  | Coin.Quarter => 25

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 2000
  | Coin.HalfDollar => 3000
  | Coin.Quarter => 1500

/-- The number of coins of each type -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.HalfDollar + coinCount Coin.Quarter

/-- The probability of selecting a half-dollar -/
def halfDollarProbability : ℚ := coinCount Coin.HalfDollar / totalCoins

theorem halfDollarProbabilityIs3_16 : halfDollarProbability = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_halfDollarProbabilityIs3_16_l1863_186330


namespace NUMINAMATH_CALUDE_divisibility_implies_p_q_values_l1863_186312

/-- A polynomial is divisible by (x + 2)(x - 2) if and only if it equals zero when x = 2 and x = -2 -/
def is_divisible_by_x2_minus4 (f : ℝ → ℝ) : Prop :=
  f 2 = 0 ∧ f (-2) = 0

/-- The polynomial x^5 - x^4 + x^3 - px^2 + qx - 8 -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ :=
  x^5 - x^4 + x^3 - p*x^2 + q*x - 8

theorem divisibility_implies_p_q_values :
  ∀ p q : ℝ, is_divisible_by_x2_minus4 (polynomial p q) → p = -2 ∧ q = -12 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_p_q_values_l1863_186312


namespace NUMINAMATH_CALUDE_solution_mixture_l1863_186325

/-- Proves that 112 ounces of Solution B is needed to create a 140-ounce mixture
    that is 80% salt when combined with Solution A (40% salt) --/
theorem solution_mixture (solution_a_salt_percentage : ℝ) (solution_b_salt_percentage : ℝ)
  (total_mixture_ounces : ℝ) (target_salt_percentage : ℝ) :
  solution_a_salt_percentage = 0.4 →
  solution_b_salt_percentage = 0.9 →
  total_mixture_ounces = 140 →
  target_salt_percentage = 0.8 →
  ∃ (solution_b_ounces : ℝ),
    solution_b_ounces = 112 ∧
    solution_b_ounces + (total_mixture_ounces - solution_b_ounces) = total_mixture_ounces ∧
    solution_a_salt_percentage * (total_mixture_ounces - solution_b_ounces) +
      solution_b_salt_percentage * solution_b_ounces =
      target_salt_percentage * total_mixture_ounces :=
by sorry


end NUMINAMATH_CALUDE_solution_mixture_l1863_186325


namespace NUMINAMATH_CALUDE_min_distance_squared_l1863_186371

/-- Given real numbers a, b, c, and d satisfying certain conditions,
    the minimum value of (a-c)^2 + (b-d)^2 is 1. -/
theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (a' b' c' d' : ℝ), 
    Real.log (b' + 1) + a' - 3 * b' = 0 → 
    2 * d' - c' + Real.sqrt 5 = 0 → 
    (a' - c')^2 + (b' - d')^2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1863_186371


namespace NUMINAMATH_CALUDE_y_minus_x_value_l1863_186302

theorem y_minus_x_value (x y z : ℚ) 
  (eq1 : x + y + z = 12)
  (eq2 : x + y = 8)
  (eq3 : y - 3*x + z = 9) :
  y - x = 13/2 := by
sorry

end NUMINAMATH_CALUDE_y_minus_x_value_l1863_186302


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1863_186350

theorem complex_arithmetic_equality : 
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1863_186350


namespace NUMINAMATH_CALUDE_excircle_incircle_similarity_l1863_186333

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Represents a circle defined by its center and a point on the circumference -/
structure Circle :=
  (center : Point) (point : Point)

/-- Defines an excircle of a triangle -/
def excircle (T : Triangle) (vertex : Point) : Circle :=
  sorry

/-- Defines the incircle of a triangle -/
def incircle (T : Triangle) : Circle :=
  sorry

/-- Defines the circumcircle of a triangle -/
def circumcircle (T : Triangle) : Circle :=
  sorry

/-- Defines the point where a circle touches a line segment -/
def touchPoint (C : Circle) (A B : Point) : Point :=
  sorry

/-- Defines the intersection points of two circles -/
def circleIntersection (C1 C2 : Circle) : Set Point :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  sorry

theorem excircle_incircle_similarity
  (ABC : Triangle)
  (A' : Point) (B' : Point) (C' : Point)
  (C1 : Point) (A1 : Point) (B1 : Point) :
  A' = touchPoint (excircle ABC ABC.A) ABC.B ABC.C →
  B' = touchPoint (excircle ABC ABC.B) ABC.C ABC.A →
  C' = touchPoint (excircle ABC ABC.C) ABC.A ABC.B →
  C1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨A', B', C⟩) →
  A1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨ABC.A, B', C'⟩) →
  B1 ∈ circleIntersection (circumcircle ABC) (circumcircle ⟨A', ABC.B, C'⟩) →
  let incirclePoints := Triangle.mk
    (touchPoint (incircle ABC) ABC.B ABC.C)
    (touchPoint (incircle ABC) ABC.C ABC.A)
    (touchPoint (incircle ABC) ABC.A ABC.B)
  areSimilar ⟨A1, B1, C1⟩ incirclePoints :=
by
  sorry

end NUMINAMATH_CALUDE_excircle_incircle_similarity_l1863_186333


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l1863_186399

theorem complex_arithmetic_equation : 
  (22 / 3 : ℚ) - ((12 / 5 + 5 / 3 * 4) / (17 / 10)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l1863_186399


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l1863_186384

def is_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ,
    10000 ≤ n ∧ n < 100000 ∧
    is_abcba n ∧
    n % 13 = 0 →
    n ≤ 96769 :=
by sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l1863_186384


namespace NUMINAMATH_CALUDE_sin_675_degrees_l1863_186306

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l1863_186306


namespace NUMINAMATH_CALUDE_inflection_point_is_center_of_symmetry_l1863_186397

/-- Represents a cubic function of the form ax³ + bx² + cx + d -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_nonzero : a ≠ 0

/-- The given cubic function x³ - 3x² + 3x -/
def f : CubicFunction := {
  a := 1
  b := -3
  c := 3
  d := 0
  a_nonzero := by norm_num
}

/-- Evaluates a cubic function at a given x -/
def evaluate (f : CubicFunction) (x : ℝ) : ℝ :=
  f.a * x^3 + f.b * x^2 + f.c * x + f.d

/-- Computes the second derivative of a cubic function -/
def secondDerivative (f : CubicFunction) (x : ℝ) : ℝ :=
  6 * f.a * x + 2 * f.b

/-- An inflection point of a cubic function -/
structure InflectionPoint (f : CubicFunction) where
  x : ℝ
  y : ℝ
  is_inflection : secondDerivative f x = 0
  on_curve : y = evaluate f x

theorem inflection_point_is_center_of_symmetry :
  ∃ (p : InflectionPoint f), p.x = 1 ∧ p.y = 1 := by sorry

end NUMINAMATH_CALUDE_inflection_point_is_center_of_symmetry_l1863_186397


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1863_186304

theorem polynomial_divisibility (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℝ, x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 = (x - 1)^3 * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1863_186304


namespace NUMINAMATH_CALUDE_ellipse_trajectory_l1863_186361

/-- The trajectory of point Q given an ellipse and its properties -/
theorem ellipse_trajectory (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (x y : ℝ), 
  ((-x/2)^2 / a^2 + (-y/2)^2 / b^2 = 1) →
  (x^2 / (4*a^2) + y^2 / (4*b^2) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_trajectory_l1863_186361


namespace NUMINAMATH_CALUDE_unique_base_for_315_l1863_186310

theorem unique_base_for_315 :
  ∃! b : ℕ, b ≥ 2 ∧ b^4 ≤ 315 ∧ 315 < b^5 :=
by sorry

end NUMINAMATH_CALUDE_unique_base_for_315_l1863_186310


namespace NUMINAMATH_CALUDE_intersection_A_B_solution_set_a_eq_1_solution_set_a_gt_1_solution_set_a_lt_1_l1863_186379

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := x^2 - (a + 1)*x + a < 0

-- Theorems for the solution sets of the inequality
theorem solution_set_a_eq_1 : {x | inequality 1 x} = ∅ := by sorry

theorem solution_set_a_gt_1 (a : ℝ) (h : a > 1) : 
  {x | inequality a x} = {x | 1 < x ∧ x < a} := by sorry

theorem solution_set_a_lt_1 (a : ℝ) (h : a < 1) : 
  {x | inequality a x} = {x | a < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_solution_set_a_eq_1_solution_set_a_gt_1_solution_set_a_lt_1_l1863_186379


namespace NUMINAMATH_CALUDE_right_triangle_validity_and_area_l1863_186368

theorem right_triangle_validity_and_area :
  ∀ (a b c : ℝ),
  a = 5 ∧ c = 13 ∧ a < b ∧ b < c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 30 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_validity_and_area_l1863_186368


namespace NUMINAMATH_CALUDE_number_line_points_l1863_186389

theorem number_line_points (A B : ℝ) : 
  (|A - B| = 4 * Real.sqrt 2) → 
  (A = 3 * Real.sqrt 2) → 
  (B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_number_line_points_l1863_186389


namespace NUMINAMATH_CALUDE_cylinder_volume_l1863_186373

theorem cylinder_volume (r_cylinder r_cone h_cylinder h_cone v_cone : ℝ) :
  r_cylinder / r_cone = 2 / 3 →
  h_cylinder / h_cone = 4 / 3 →
  v_cone = 5.4 →
  (π * r_cylinder^2 * h_cylinder) = 3.2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1863_186373


namespace NUMINAMATH_CALUDE_exists_uncovered_cell_l1863_186385

/-- Represents a grid cell --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- Represents a rectangle --/
structure Rectangle :=
  (width : Nat) (height : Nat)

/-- The dimensions of the grid --/
def gridWidth : Nat := 11
def gridHeight : Nat := 1117

/-- The dimensions of the cutting rectangle --/
def cuttingRectangle : Rectangle := { width := 6, height := 1 }

/-- A function to check if a cell is covered by a rectangle --/
def isCovered (c : Cell) (r : Rectangle) (position : Cell) : Prop :=
  c.x ≥ position.x ∧ c.x < position.x + r.width ∧
  c.y ≥ position.y ∧ c.y < position.y + r.height

/-- The main theorem --/
theorem exists_uncovered_cell :
  ∃ (c : Cell), c.x < gridWidth ∧ c.y < gridHeight ∧
  ∀ (arrangements : List Cell),
    ∃ (p : Cell), p ∈ arrangements →
      ¬(isCovered c cuttingRectangle p) :=
sorry

end NUMINAMATH_CALUDE_exists_uncovered_cell_l1863_186385


namespace NUMINAMATH_CALUDE_work_distance_is_ten_l1863_186305

/-- Calculates the one-way distance to work given gas tank capacity, remaining fuel fraction, and fuel efficiency. -/
def distance_to_work (tank_capacity : ℚ) (remaining_fraction : ℚ) (miles_per_gallon : ℚ) : ℚ :=
  (tank_capacity * (1 - remaining_fraction) * miles_per_gallon) / 2

/-- Proves that given the specified conditions, Jim's work is 10 miles away from his house. -/
theorem work_distance_is_ten :
  let tank_capacity : ℚ := 12
  let remaining_fraction : ℚ := 2/3
  let miles_per_gallon : ℚ := 5
  distance_to_work tank_capacity remaining_fraction miles_per_gallon = 10 := by
  sorry


end NUMINAMATH_CALUDE_work_distance_is_ten_l1863_186305


namespace NUMINAMATH_CALUDE_a_5_value_l1863_186343

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence (λ n => 1 + a n) →
  (∀ n : ℕ, (1 + a (n + 1)) = 2 * (1 + a n)) →
  a 1 = 1 →
  a 5 = 31 := by
sorry

end NUMINAMATH_CALUDE_a_5_value_l1863_186343


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l1863_186307

/-- Calculates the total cost of a shopping trip and determines the additional amount needed --/
theorem shopping_cost_calculation (shirts_count sunglasses_count skirts_count sandals_count hats_count bags_count earrings_count : ℕ)
  (shirt_price sunglasses_price skirt_price sandal_price hat_price bag_price earring_price : ℚ)
  (discount_rate tax_rate : ℚ) (payment : ℚ) :
  let subtotal := shirts_count * shirt_price + sunglasses_count * sunglasses_price + 
                  skirts_count * skirt_price + sandals_count * sandal_price + 
                  hats_count * hat_price + bags_count * bag_price + 
                  earrings_count * earring_price
  let discounted_total := subtotal * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  let change_needed := final_total - payment
  shirts_count = 10 ∧ sunglasses_count = 2 ∧ skirts_count = 4 ∧
  sandals_count = 3 ∧ hats_count = 5 ∧ bags_count = 7 ∧ earrings_count = 6 ∧
  shirt_price = 5 ∧ sunglasses_price = 12 ∧ skirt_price = 18 ∧
  sandal_price = 3 ∧ hat_price = 8 ∧ bag_price = 14 ∧ earring_price = 6 ∧
  discount_rate = 1/10 ∧ tax_rate = 13/200 ∧ payment = 300 →
  change_needed = 307/20 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cost_calculation_l1863_186307


namespace NUMINAMATH_CALUDE_max_value_cube_plus_one_l1863_186335

/-- Given that x + y = 1, prove that (x³+1)(y³+1) achieves its maximum value
    when x = (1 ± √5)/2 and y = (1 ∓ √5)/2 -/
theorem max_value_cube_plus_one (x y : ℝ) (h : x + y = 1) :
  ∃ (max_x max_y : ℝ), 
    (max_x = (1 + Real.sqrt 5) / 2 ∧ max_y = (1 - Real.sqrt 5) / 2) ∨
    (max_x = (1 - Real.sqrt 5) / 2 ∧ max_y = (1 + Real.sqrt 5) / 2) ∧
    ∀ (a b : ℝ), a + b = 1 → 
      (x^3 + 1) * (y^3 + 1) ≤ (max_x^3 + 1) * (max_y^3 + 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_cube_plus_one_l1863_186335


namespace NUMINAMATH_CALUDE_phil_quarters_left_l1863_186303

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def jeans_cost : ℚ := 11.50
def quarter_value : ℚ := 0.25

theorem phil_quarters_left : 
  let total_spent := pizza_cost + soda_cost + jeans_cost
  let remaining_amount := initial_amount - total_spent
  (remaining_amount / quarter_value).floor = 97 := by sorry

end NUMINAMATH_CALUDE_phil_quarters_left_l1863_186303


namespace NUMINAMATH_CALUDE_fraction_order_l1863_186323

theorem fraction_order : 
  (23 : ℚ) / 18 < (21 : ℚ) / 16 ∧ (21 : ℚ) / 16 < (25 : ℚ) / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1863_186323


namespace NUMINAMATH_CALUDE_two_digit_product_8640_l1863_186331

theorem two_digit_product_8640 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8640 → 
  min a b = 60 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_8640_l1863_186331


namespace NUMINAMATH_CALUDE_valid_combinations_l1863_186327

/-- A combination is valid if it satisfies the given equation and range constraints. -/
def is_valid_combination (x y z : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 20 ∧
  10 ≤ y ∧ y ≤ 20 ∧
  10 ≤ z ∧ z ≤ 20 ∧
  3 * x^2 - y^2 - 7 * z = 99

/-- The theorem states that there are exactly three valid combinations. -/
theorem valid_combinations :
  (∀ x y z : ℕ, is_valid_combination x y z ↔ 
    ((x = 15 ∧ y = 10 ∧ z = 68) ∨ 
     (x = 16 ∧ y = 12 ∧ z = 75) ∨ 
     (x = 18 ∧ y = 15 ∧ z = 78))) :=
by sorry

end NUMINAMATH_CALUDE_valid_combinations_l1863_186327


namespace NUMINAMATH_CALUDE_wendys_washing_machine_capacity_l1863_186309

-- Define the number of shirts
def shirts : ℕ := 39

-- Define the number of sweaters
def sweaters : ℕ := 33

-- Define the number of loads
def loads : ℕ := 9

-- Define the function to calculate the washing machine capacity
def washing_machine_capacity (s : ℕ) (w : ℕ) (l : ℕ) : ℕ :=
  (s + w) / l

-- Theorem statement
theorem wendys_washing_machine_capacity :
  washing_machine_capacity shirts sweaters loads = 8 := by
  sorry

end NUMINAMATH_CALUDE_wendys_washing_machine_capacity_l1863_186309


namespace NUMINAMATH_CALUDE_parallelogram_existence_l1863_186353

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a table with marked cells -/
structure Table where
  size : Nat
  markedCells : Finset Cell

/-- Represents a parallelogram in the table -/
structure Parallelogram where
  v1 : Cell
  v2 : Cell
  v3 : Cell
  v4 : Cell

/-- Checks if a cell is within the table bounds -/
def Cell.isValid (c : Cell) (n : Nat) : Prop :=
  c.row < n ∧ c.col < n

/-- Checks if a parallelogram is valid (all vertices are marked and form a parallelogram) -/
def Parallelogram.isValid (p : Parallelogram) (t : Table) : Prop :=
  p.v1 ∈ t.markedCells ∧ p.v2 ∈ t.markedCells ∧ p.v3 ∈ t.markedCells ∧ p.v4 ∈ t.markedCells ∧
  (p.v1.row - p.v2.row = p.v4.row - p.v3.row) ∧
  (p.v1.col - p.v2.col = p.v4.col - p.v3.col)

/-- Main theorem: In an n × n table with 2n marked cells, there exists a valid parallelogram -/
theorem parallelogram_existence (t : Table) (h1 : t.markedCells.card = 2 * t.size) :
  ∃ p : Parallelogram, p.isValid t :=
sorry

end NUMINAMATH_CALUDE_parallelogram_existence_l1863_186353


namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_problem_l1863_186329

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (upstream_downstream_ratio : ℝ) : ℝ :=
  let stream_speed := (still_speed * (upstream_downstream_ratio - 1)) / (upstream_downstream_ratio + 1)
  stream_speed

/-- Proves that the speed of the stream is 0.5 km/h given the conditions -/
theorem stream_speed_problem : stream_speed 1.5 2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_problem_l1863_186329


namespace NUMINAMATH_CALUDE_unique_solution_system_l1863_186394

theorem unique_solution_system (x y : ℝ) :
  (2 * x - 3 * abs y = 1 ∧ abs x + 2 * y = 4) ↔ (x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1863_186394


namespace NUMINAMATH_CALUDE_rental_ratio_proof_l1863_186313

/-- Represents the ratio of dramas to action movies rented during a two-week period -/
def drama_action_ratio : ℚ := 37 / 8

/-- Theorem stating the ratio of dramas to action movies given the rental conditions -/
theorem rental_ratio_proof (T : ℝ) (a : ℝ) (h1 : T > 0) (h2 : a > 0) : 
  (0.64 * T = 10 * a) →  -- Condition: 64% of rentals are comedies, and comedies = 10a
  (∃ d : ℝ, d > 0 ∧ 0.36 * T = a + d) →  -- Condition: Remaining 36% are dramas and action movies
  (∃ s : ℝ, s > 0 ∧ ∃ d : ℝ, d = s * a) →  -- Condition: Dramas are some times action movies
  drama_action_ratio = 37 / 8 :=
sorry

end NUMINAMATH_CALUDE_rental_ratio_proof_l1863_186313


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1863_186311

/-- The slope of a line tangent to the circle x^2 + y^2 - 4x + 2 = 0 is either 1 or -1 -/
theorem tangent_line_slope (m : ℝ) :
  (∀ x y : ℝ, y = m * x → x^2 + y^2 - 4*x + 2 = 0 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → x'^2 + y'^2 - 4*x' + 2 > 0) →
  m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1863_186311


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1863_186376

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (has_square_base : base_side > 0)
  (has_equilateral_lateral_faces : True)

/-- A cube placed inside the pyramid -/
structure InsideCube :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The theorem stating the volume of the cube inside the pyramid -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) 
  (h1 : p.base_side = 2) : c.side_length ^ 3 = 3 * Real.sqrt 6 / 4 := by
  sorry

#check cube_volume_in_pyramid

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l1863_186376


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1863_186378

theorem tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 3/8) 
  (h2 : lily_win_prob = 3/10) 
  (h3 : amy_win_prob + lily_win_prob ≤ 1) :
  1 - (amy_win_prob + lily_win_prob) = 13/40 :=
by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l1863_186378


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1863_186365

theorem value_of_a_minus_b (a b : ℝ) 
  (ha : |a| = 5) 
  (hb : |b| = 4) 
  (hab : a + b < 0) : 
  a - b = -9 ∨ a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1863_186365


namespace NUMINAMATH_CALUDE_not_both_divisible_by_seven_l1863_186324

theorem not_both_divisible_by_seven (a b : ℝ) : 
  (¬ ∃ k : ℤ, a * b = 7 * k) → (¬ ∃ m : ℤ, a = 7 * m) ∧ (¬ ∃ n : ℤ, b = 7 * n) := by
  sorry

end NUMINAMATH_CALUDE_not_both_divisible_by_seven_l1863_186324


namespace NUMINAMATH_CALUDE_hundred_with_fewer_threes_l1863_186347

/-- An arithmetic expression using only the number 3 and basic operations -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of 3's in an expression -/
def count_threes : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2

/-- Theorem: There exists an expression that evaluates to 100 using fewer than 10 threes -/
theorem hundred_with_fewer_threes : ∃ e : Expr, eval e = 100 ∧ count_threes e < 10 := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_fewer_threes_l1863_186347


namespace NUMINAMATH_CALUDE_equation_solution_l1863_186300

theorem equation_solution :
  ∀ x : ℝ, (1 / 7 : ℝ) + 7 / x = 15 / x + (1 / 15 : ℝ) → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1863_186300


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l1863_186381

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l1863_186381


namespace NUMINAMATH_CALUDE_max_value_theorem_l1863_186374

theorem max_value_theorem (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_constraint : a^2 + b^2 + 4*c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ ∀ x, x = a*b + 2*a*c + 3*Real.sqrt 2*b*c → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1863_186374


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1863_186332

def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1863_186332


namespace NUMINAMATH_CALUDE_symmetry_correctness_l1863_186316

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry operations in 3D space -/
def symmetry_x_axis (p : Point3D) : Point3D := ⟨p.x, -p.y, -p.z⟩
def symmetry_yoz_plane (p : Point3D) : Point3D := ⟨-p.x, p.y, p.z⟩
def symmetry_y_axis (p : Point3D) : Point3D := ⟨-p.x, p.y, -p.z⟩
def symmetry_origin (p : Point3D) : Point3D := ⟨-p.x, -p.y, -p.z⟩

/-- The theorem to be proved -/
theorem symmetry_correctness (a b c : ℝ) : 
  let M : Point3D := ⟨a, b, c⟩
  (symmetry_x_axis M ≠ ⟨a, -b, c⟩) ∧ 
  (symmetry_yoz_plane M ≠ ⟨a, -b, -c⟩) ∧ 
  (symmetry_y_axis M ≠ ⟨a, -b, c⟩) ∧ 
  (symmetry_origin M = ⟨-a, -b, -c⟩) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_correctness_l1863_186316


namespace NUMINAMATH_CALUDE_sequence_properties_l1863_186322

def sequence_a (n : ℕ) : ℝ := sorry

def sequence_S (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (sequence_a 1 = 9) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_S n = 3 * (sequence_S (n - 1) + 3)) →
  (∀ n : ℕ, n > 0 → sequence_a n = 9 * 3^(n - 1)) ∧
  (∃ r : ℝ, r = 3 ∧ ∀ n : ℕ, n > 0 → 
    (sequence_S (n + 1) + 9/2) / (sequence_S n + 9/2) = r) ∧
  (sequence_S 1 + 9/2 = 27/2) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1863_186322


namespace NUMINAMATH_CALUDE_average_equation_solution_l1863_186386

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 69 → a = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1863_186386


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1863_186315

/-- Complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

/-- Predicate for z being purely imaginary -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem sufficient_not_necessary :
  (∀ (a : ℝ), a = -2 → isPurelyImaginary (z a)) ∧
  (∃ (a : ℝ), a ≠ -2 ∧ isPurelyImaginary (z a)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1863_186315


namespace NUMINAMATH_CALUDE_john_profit_calculation_l1863_186314

/-- Calculates John's profit from selling newspapers, magazines, and books --/
theorem john_profit_calculation :
  let newspaper_count : ℕ := 500
  let magazine_count : ℕ := 300
  let book_count : ℕ := 200
  let newspaper_price : ℚ := 2
  let magazine_price : ℚ := 4
  let book_price : ℚ := 10
  let newspaper_sold_ratio : ℚ := 0.80
  let magazine_sold_ratio : ℚ := 0.75
  let book_sold_ratio : ℚ := 0.60
  let newspaper_discount : ℚ := 0.75
  let magazine_discount : ℚ := 0.60
  let book_discount : ℚ := 0.45
  let tax_rate : ℚ := 0.08
  let shipping_fee : ℚ := 25
  let commission_rate : ℚ := 0.05

  let newspaper_cost := newspaper_price * (1 - newspaper_discount)
  let magazine_cost := magazine_price * (1 - magazine_discount)
  let book_cost := book_price * (1 - book_discount)

  let total_cost_before_tax := 
    newspaper_count * newspaper_cost +
    magazine_count * magazine_cost +
    book_count * book_cost

  let total_cost_after_tax_and_shipping :=
    total_cost_before_tax * (1 + tax_rate) + shipping_fee

  let total_revenue_before_commission :=
    newspaper_count * newspaper_sold_ratio * newspaper_price +
    magazine_count * magazine_sold_ratio * magazine_price +
    book_count * book_sold_ratio * book_price

  let total_revenue_after_commission :=
    total_revenue_before_commission * (1 - commission_rate)

  let profit := total_revenue_after_commission - total_cost_after_tax_and_shipping

  profit = 753.60 := by sorry

end NUMINAMATH_CALUDE_john_profit_calculation_l1863_186314


namespace NUMINAMATH_CALUDE_biased_coin_probability_l1863_186366

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  6 * p^2 * (1 - p)^2 = (1 : ℝ) / 6 →
  p = (3 - Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l1863_186366


namespace NUMINAMATH_CALUDE_pasha_mistake_l1863_186328

theorem pasha_mistake : 
  ¬ ∃ (K R O S C T A P : ℕ),
    (K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ A < 10 ∧ P < 10) ∧
    (K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ A ∧ K ≠ P ∧
     R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ A ∧ R ≠ P ∧
     O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ A ∧ O ≠ P ∧
     S ≠ C ∧ S ≠ T ∧ S ≠ A ∧ S ≠ P ∧
     C ≠ T ∧ C ≠ A ∧ C ≠ P ∧
     T ≠ A ∧ T ≠ P ∧
     A ≠ P) ∧
    (K * 10000 + R * 1000 + O * 100 + S * 10 + S + 2011 = 
     C * 10000 + T * 1000 + A * 100 + P * 10 + T) :=
by sorry

end NUMINAMATH_CALUDE_pasha_mistake_l1863_186328


namespace NUMINAMATH_CALUDE_pencils_per_row_l1863_186395

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 35)
  (h2 : num_rows = 7)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l1863_186395


namespace NUMINAMATH_CALUDE_product_equals_zero_l1863_186383

theorem product_equals_zero : (3 - 5) * (3 - 4) * (3 - 3) * (3 - 2) * (3 - 1) * 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1863_186383


namespace NUMINAMATH_CALUDE_cost_price_satisfies_profit_condition_l1863_186321

/-- The cost price of an article satisfies the given profit condition -/
theorem cost_price_satisfies_profit_condition (C : ℝ) : C > 0 → (0.27 * C) - (0.12 * C) = 108 ↔ C = 720 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_satisfies_profit_condition_l1863_186321


namespace NUMINAMATH_CALUDE_alice_favorite_number_l1863_186359

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  70 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  ¬(n % 3 = 0) ∧
  is_prime (digit_sum n)

theorem alice_favorite_number :
  ∀ n : ℕ, satisfies_conditions n ↔ n = 104 :=
sorry

end NUMINAMATH_CALUDE_alice_favorite_number_l1863_186359


namespace NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l1863_186388

/-- The cost of football equipment relative to shorts -/
def FootballEquipmentCost (x : ℝ) : Prop :=
  let shorts := x
  let tshirt := x
  let boots := 4 * x
  let shinguards := 2 * x
  (shorts + tshirt = 2 * x) ∧
  (shorts + boots = 5 * x) ∧
  (shorts + shinguards = 3 * x) ∧
  (shorts + tshirt + boots + shinguards = 8 * x)

/-- Theorem: The total cost of all items is 8 times the cost of shorts -/
theorem total_cost_is_eight_times_shorts (x : ℝ) (h : FootballEquipmentCost x) :
  ∃ (shorts tshirt boots shinguards : ℝ),
    shorts = x ∧
    shorts + tshirt + boots + shinguards = 8 * x :=
by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_eight_times_shorts_l1863_186388


namespace NUMINAMATH_CALUDE_percentage_problem_l1863_186348

-- Define the percentage P
def P : ℝ := sorry

-- Theorem to prove
theorem percentage_problem : P = 45 := by
  -- Define the conditions
  have h1 : P / 100 * 60 = 35 / 100 * 40 + 13 := sorry
  
  -- Prove that P equals 45
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1863_186348


namespace NUMINAMATH_CALUDE_dads_strawberry_weight_l1863_186377

/-- 
Given:
- The total initial weight of strawberries collected by Marco and his dad
- The weight of strawberries lost by Marco's dad
- The current weight of Marco's strawberries

Prove that the weight of Marco's dad's strawberries is equal to the difference between 
the total weight after loss and Marco's current weight of strawberries.
-/
theorem dads_strawberry_weight 
  (total_initial_weight : ℕ) 
  (weight_lost : ℕ) 
  (marcos_weight : ℕ) : 
  total_initial_weight - weight_lost - marcos_weight = 
    total_initial_weight - (weight_lost + marcos_weight) := by
  sorry

#check dads_strawberry_weight

end NUMINAMATH_CALUDE_dads_strawberry_weight_l1863_186377


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1863_186362

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n → n ≥ 104 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1863_186362


namespace NUMINAMATH_CALUDE_catenary_properties_l1863_186372

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + b * Real.exp (-x)

theorem catenary_properties :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  (∀ x, f 1 b x = f 1 b (-x) → b = 1) ∧
  (∃ a b, ∀ x y, x < y → f a b x < f a b y) ∧
  ((∃ a b, ∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) →
   (∀ a b, (∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) → a + b ≥ 2) ∧
   (∃ a b, (∀ x, f a b x ≥ 2 ∧ (∃ x₀, f a b x₀ = 2)) ∧ a + b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_catenary_properties_l1863_186372


namespace NUMINAMATH_CALUDE_stream_speed_l1863_186355

/-- Proves that the speed of a stream is 5 km/h, given a man's swimming speed in still water
    and the relative time taken to swim upstream vs downstream. -/
theorem stream_speed (man_speed : ℝ) (upstream_time_ratio : ℝ) 
  (h1 : man_speed = 15)
  (h2 : upstream_time_ratio = 2) : 
  ∃ (stream_speed : ℝ), stream_speed = 5 ∧
  (man_speed + stream_speed) * 1 = (man_speed - stream_speed) * upstream_time_ratio :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1863_186355


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1863_186393

theorem divisibility_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1863_186393


namespace NUMINAMATH_CALUDE_function_value_at_negative_five_hundred_l1863_186392

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + 2 * f x

theorem function_value_at_negative_five_hundred
  (f : ℝ → ℝ)
  (h1 : FunctionalEquation f)
  (h2 : f (-2) = 11) :
  f (-500) = -487 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_five_hundred_l1863_186392


namespace NUMINAMATH_CALUDE_arrangement_remainder_l1863_186387

/-- The number of green marbles -/
def green_marbles : ℕ := 5

/-- The maximum number of blue marbles that satisfies the arrangement condition -/
def max_blue_marbles : ℕ := 15

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + max_blue_marbles

/-- The number of ways to arrange the marbles satisfying the condition -/
def arrangement_count : ℕ := (Nat.choose (max_blue_marbles + green_marbles) green_marbles)

/-- Theorem stating that the remainder when dividing the number of arrangements by 1000 is 3 -/
theorem arrangement_remainder : arrangement_count % 1000 = 3 := by sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l1863_186387


namespace NUMINAMATH_CALUDE_course_selection_combinations_l1863_186301

theorem course_selection_combinations :
  let total_courses : ℕ := 7
  let required_courses : ℕ := 2
  let math_courses : ℕ := 2
  let program_size : ℕ := 5
  let remaining_courses : ℕ := total_courses - required_courses
  let remaining_selections : ℕ := program_size - required_courses

  (Nat.choose remaining_courses remaining_selections) -
  (Nat.choose (remaining_courses - math_courses) remaining_selections) = 9 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_combinations_l1863_186301


namespace NUMINAMATH_CALUDE_expression_equality_l1863_186351

theorem expression_equality : (45 + 15)^2 - 3 * (45^2 + 15^2 - 2 * 45 * 15) = 900 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1863_186351


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1863_186308

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : is_positive_geometric_sequence a)
  (h_2 : a 2 = 3)
  (h_8 : a 8 = 27) :
  a 5 = 9 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1863_186308


namespace NUMINAMATH_CALUDE_triangle_properties_l1863_186345

/-- Given a triangle ABC with the following properties:
    - B has coordinates (1, -2)
    - The median CM on side AB has equation 2x - y + 1 = 0
    - The angle bisector of ∠BAC has equation x + 7y - 12 = 0
    Prove that:
    1. A has coordinates (-2, 2)
    2. The equation of line AC is 3x - 4y + 14 = 0
-/
theorem triangle_properties (B : ℝ × ℝ) (median_CM : ℝ → ℝ → ℝ) (angle_bisector : ℝ → ℝ → ℝ) 
  (hB : B = (1, -2))
  (hmedian : ∀ x y, median_CM x y = 2 * x - y + 1)
  (hbisector : ∀ x y, angle_bisector x y = x + 7 * y - 12) :
  ∃ (A : ℝ × ℝ) (line_AC : ℝ → ℝ → ℝ),
    A = (-2, 2) ∧ 
    (∀ x y, line_AC x y = 3 * x - 4 * y + 14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1863_186345


namespace NUMINAMATH_CALUDE_v2_equals_22_at_neg_4_l1863_186364

/-- Horner's Rule for a specific polynomial -/
def horner_polynomial (x : ℝ) : ℝ := 
  ((((x * x + 6) * x + 9) * x + 0) * x + 0) * x + 208

/-- v2 calculation in Horner's Rule -/
def v2 (x : ℝ) : ℝ := 
  (1 * x * x) + 6

/-- Theorem: v2 equals 22 when x = -4 for the given polynomial -/
theorem v2_equals_22_at_neg_4 : 
  v2 (-4) = 22 := by sorry

end NUMINAMATH_CALUDE_v2_equals_22_at_neg_4_l1863_186364
