import Mathlib

namespace NUMINAMATH_CALUDE_platform_length_l4154_415412

/-- Given a train of length 300 meters that takes 42 seconds to cross a platform
    and 18 seconds to cross a signal pole, prove that the length of the platform
    is approximately 400.14 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 42)
  (h3 : pole_crossing_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 400.14) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l4154_415412


namespace NUMINAMATH_CALUDE_project_selection_count_l4154_415404

/-- The number of key projects -/
def num_key_projects : ℕ := 4

/-- The number of general projects -/
def num_general_projects : ℕ := 6

/-- The number of projects to be selected from each category -/
def projects_per_category : ℕ := 2

/-- Calculates the number of ways to select projects with the given conditions -/
def select_projects : ℕ :=
  Nat.choose num_key_projects projects_per_category *
  Nat.choose num_general_projects projects_per_category -
  Nat.choose (num_key_projects - 1) projects_per_category *
  Nat.choose (num_general_projects - 1) projects_per_category

theorem project_selection_count :
  select_projects = 60 := by sorry

end NUMINAMATH_CALUDE_project_selection_count_l4154_415404


namespace NUMINAMATH_CALUDE_symmetric_difference_of_M_and_N_l4154_415455

-- Define the symmetric difference operation
def symmetricDifference (A B : Set ℝ) : Set ℝ :=
  (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {y | ∃ x, 0 < x ∧ x < 2 ∧ y = -x^2 + 2*x}
def N : Set ℝ := {y | ∃ x, x > 0 ∧ y = 2^(x-1)}

-- State the theorem
theorem symmetric_difference_of_M_and_N :
  symmetricDifference M N = {y | (0 < y ∧ y ≤ 1/2) ∨ (1 < y)} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_M_and_N_l4154_415455


namespace NUMINAMATH_CALUDE_third_median_length_special_triangle_third_median_l4154_415479

/-- A triangle with specific median lengths and area -/
structure SpecialTriangle where
  -- Two medians of the triangle
  median1 : ℝ
  median2 : ℝ
  -- Area of the triangle
  area : ℝ
  -- Conditions on the medians and area
  median1_length : median1 = 4
  median2_length : median2 = 8
  triangle_area : area = 4 * Real.sqrt 15

/-- The third median of the special triangle has length 7 -/
theorem third_median_length (t : SpecialTriangle) : ℝ :=
  7

/-- The theorem stating that the third median of the special triangle has length 7 -/
theorem special_triangle_third_median (t : SpecialTriangle) : 
  third_median_length t = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_median_length_special_triangle_third_median_l4154_415479


namespace NUMINAMATH_CALUDE_polynomial_roots_problem_l4154_415416

theorem polynomial_roots_problem (r s t : ℝ) 
  (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + r*x + s = 0 ↔ x = s ∨ x = t)
  (h2 : (5 : ℝ)^2 + t*5 + r = 0) : 
  s = 29 := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_problem_l4154_415416


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4154_415444

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  (f 1 = 0 ∧ f 3 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4154_415444


namespace NUMINAMATH_CALUDE_special_ellipse_ratio_l4154_415410

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  -- Semi-major axis
  a : ℝ
  -- Semi-minor axis
  b : ℝ
  -- Distance from center to focus
  c : ℝ
  -- Ensure a > b > 0 and c > 0
  h1 : a > b
  h2 : b > 0
  h3 : c > 0
  -- Ellipse equation: a² = b² + c²
  h4 : a^2 = b^2 + c^2
  -- Special condition: |F1B2|² = |OF1| * |B1B2|
  h5 : (a + c)^2 = c * (2 * b)

/-- The ratio of semi-major axis to center-focus distance is 3:2 -/
theorem special_ellipse_ratio (e : SpecialEllipse) : a / c = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_ratio_l4154_415410


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l4154_415468

/-- For any real number p > 1, the minimum value of x + y, where x and y satisfy the equation
    (x + √(1 + x²))(y + √(1 + y²)) = p, is (p - 1) / √p. -/
theorem min_sum_with_constraint (p : ℝ) (hp : p > 1) :
  (∃ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p) →
  (∀ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p → 
    x + y ≥ (p - 1) / Real.sqrt p) ∧
  (∃ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p ∧
    x + y = (p - 1) / Real.sqrt p) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l4154_415468


namespace NUMINAMATH_CALUDE_expression_evaluation_l4154_415445

theorem expression_evaluation : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4154_415445


namespace NUMINAMATH_CALUDE_quadratic_roots_divisibility_l4154_415426

theorem quadratic_roots_divisibility
  (a b : ℤ) (u v : ℂ) (h1 : u^2 + a*u + b = 0)
  (h2 : v^2 + a*v + b = 0) (h3 : ∃ k : ℤ, a^2 = k * b) :
  ∀ n : ℕ, ∃ m : ℤ, u^(2*n) + v^(2*n) = m * b^n :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_divisibility_l4154_415426


namespace NUMINAMATH_CALUDE_set_union_problem_l4154_415422

-- Define the sets A and B as functions of x
def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

-- State the theorem
theorem set_union_problem (x : ℝ) :
  (A x ∩ B x = {9}) →
  (∃ y, A y ∪ B y = {-8, -7, -4, 4, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_union_problem_l4154_415422


namespace NUMINAMATH_CALUDE_smallest_solution_correct_l4154_415487

noncomputable def smallest_solution : ℝ := 4 - Real.sqrt 15 / 3

theorem smallest_solution_correct :
  let x := smallest_solution
  (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 5 / (y - 4)) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_correct_l4154_415487


namespace NUMINAMATH_CALUDE_houses_with_neither_amenity_l4154_415419

/-- Given a development with houses, some of which have a two-car garage and/or an in-the-ground swimming pool, 
    this theorem proves the number of houses with neither amenity. -/
theorem houses_with_neither_amenity 
  (total : ℕ) 
  (garage : ℕ) 
  (pool : ℕ) 
  (both : ℕ) 
  (h1 : total = 90) 
  (h2 : garage = 50) 
  (h3 : pool = 40) 
  (h4 : both = 35) : 
  total - (garage + pool - both) = 35 := by
  sorry


end NUMINAMATH_CALUDE_houses_with_neither_amenity_l4154_415419


namespace NUMINAMATH_CALUDE_larger_number_proof_l4154_415460

/-- Given two positive integers with HCF 23 and LCM factors 11 and 12, the larger is 276 -/
theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a.val b.val = 23) → 
  (∃ (k : ℕ+), Nat.lcm a.val b.val = 23 * 11 * 12 * k.val) → 
  max a b = 276 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4154_415460


namespace NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l4154_415443

theorem arithmetic_progression_divisibility 
  (a : ℕ → ℕ) 
  (h_ap : ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_div : ∀ n : ℕ, (a n * a (n + 31)) % 2005 = 0) : 
  ∀ n : ℕ, a n % 2005 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l4154_415443


namespace NUMINAMATH_CALUDE_negation_equivalence_l4154_415485

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 3 ∧ x^2 - 2*x + 3 < 0) ↔ (∀ x : ℝ, x ≥ 3 → x^2 - 2*x + 3 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4154_415485


namespace NUMINAMATH_CALUDE_kevin_distance_after_seven_leaps_l4154_415471

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Kevin's total distance hopped after n leaps -/
def kevinDistance (n : ℕ) : ℚ :=
  geometricSum (1/4) (3/4) n

/-- Theorem: Kevin's total distance after 7 leaps is 14197/16384 -/
theorem kevin_distance_after_seven_leaps :
  kevinDistance 7 = 14197 / 16384 := by
  sorry

end NUMINAMATH_CALUDE_kevin_distance_after_seven_leaps_l4154_415471


namespace NUMINAMATH_CALUDE_even_sum_probability_l4154_415429

/-- The set of the first twenty prime numbers -/
def first_twenty_primes : Finset ℕ := sorry

/-- The number of ways to select 6 numbers from a set of 20 -/
def total_selections : ℕ := Nat.choose 20 6

/-- The number of ways to select 6 odd numbers from the set of odd primes in first_twenty_primes -/
def odd_selections : ℕ := Nat.choose 19 6

/-- The probability of selecting six prime numbers from first_twenty_primes such that their sum is even -/
def prob_even_sum : ℚ := odd_selections / total_selections

theorem even_sum_probability : prob_even_sum = 354 / 505 := by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l4154_415429


namespace NUMINAMATH_CALUDE_total_balls_bought_l4154_415400

/-- Represents the total amount of money Mr. Li had --/
def total_money : ℚ := 1

/-- The cost of a plastic ball --/
def plastic_ball_cost : ℚ := 1 / 60

/-- The cost of a glass ball --/
def glass_ball_cost : ℚ := 1 / 36

/-- The cost of a wooden ball --/
def wooden_ball_cost : ℚ := 1 / 45

/-- The number of plastic balls Mr. Li bought --/
def plastic_balls_bought : ℕ := 10

/-- The number of glass balls Mr. Li bought --/
def glass_balls_bought : ℕ := 10

theorem total_balls_bought : ℕ := by
  -- The total number of balls Mr. Li bought is 45
  sorry

end NUMINAMATH_CALUDE_total_balls_bought_l4154_415400


namespace NUMINAMATH_CALUDE_imaginary_difference_condition_l4154_415452

def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_difference_condition (z₁ z₂ : ℂ) :
  (is_imaginary (z₁ - z₂) → (is_imaginary z₁ ∨ is_imaginary z₂)) ∧
  ∃ z₁ z₂ : ℂ, (is_imaginary z₁ ∨ is_imaginary z₂) ∧ ¬is_imaginary (z₁ - z₂) :=
sorry

end NUMINAMATH_CALUDE_imaginary_difference_condition_l4154_415452


namespace NUMINAMATH_CALUDE_probability_genuine_after_defective_l4154_415417

theorem probability_genuine_after_defective :
  ∀ (total genuine defective : ℕ),
    total = genuine + defective →
    total = 7 →
    genuine = 4 →
    defective = 3 →
    (genuine : ℚ) / (total - 1 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_genuine_after_defective_l4154_415417


namespace NUMINAMATH_CALUDE_max_draws_for_all_pairs_l4154_415489

/-- Represents the number of items of a specific color -/
structure ColorCount where
  total : Nat
  deriving Repr

/-- Calculates the maximum number of draws needed to guarantee a pair for a single color -/
def maxDrawsForColor (count : ColorCount) : Nat :=
  count.total + 1

/-- The box containing hats and gloves -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount
  blue : ColorCount
  yellow : ColorCount

/-- Calculates the total maximum draws needed for all colors -/
def totalMaxDraws (box : Box) : Nat :=
  maxDrawsForColor box.red +
  maxDrawsForColor box.green +
  maxDrawsForColor box.orange +
  maxDrawsForColor box.blue +
  maxDrawsForColor box.yellow

/-- The given box with the specified item counts -/
def givenBox : Box :=
  { red := { total := 41 },
    green := { total := 23 },
    orange := { total := 11 },
    blue := { total := 15 },
    yellow := { total := 10 } }

theorem max_draws_for_all_pairs (box : Box := givenBox) :
  totalMaxDraws box = 105 := by
  sorry

end NUMINAMATH_CALUDE_max_draws_for_all_pairs_l4154_415489


namespace NUMINAMATH_CALUDE_triple_equality_l4154_415435

theorem triple_equality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h1 : x * y * (x + y) = y * z * (y + z)) 
  (h2 : y * z * (y + z) = z * x * (z + x)) : 
  (x = y ∧ y = z) ∨ x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_triple_equality_l4154_415435


namespace NUMINAMATH_CALUDE_sam_spent_12_dimes_on_baseball_cards_l4154_415403

/-- The number of pennies Sam spent on ice cream -/
def ice_cream_pennies : ℕ := 2

/-- The total amount Sam spent in cents -/
def total_spent_cents : ℕ := 122

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes Sam spent on baseball cards -/
def baseball_card_dimes : ℕ := 12

theorem sam_spent_12_dimes_on_baseball_cards :
  (total_spent_cents - ice_cream_pennies * penny_value) / dime_value = baseball_card_dimes := by
  sorry

end NUMINAMATH_CALUDE_sam_spent_12_dimes_on_baseball_cards_l4154_415403


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l4154_415431

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l4154_415431


namespace NUMINAMATH_CALUDE_fruits_per_slice_l4154_415462

/-- Represents the number of fruits per dozen -/
def dozenSize : ℕ := 12

/-- Represents the number of dozens of Granny Smith apples -/
def grannySmithDozens : ℕ := 4

/-- Represents the number of dozens of Fuji apples -/
def fujiDozens : ℕ := 2

/-- Represents the number of dozens of Bartlett pears -/
def bartlettDozens : ℕ := 3

/-- Represents the number of Granny Smith apple pies -/
def grannySmithPies : ℕ := 4

/-- Represents the number of slices per Granny Smith apple pie -/
def grannySmithSlices : ℕ := 6

/-- Represents the number of Fuji apple pies -/
def fujiPies : ℕ := 3

/-- Represents the number of slices per Fuji apple pie -/
def fujiSlices : ℕ := 8

/-- Represents the number of pear tarts -/
def pearTarts : ℕ := 2

/-- Represents the number of slices per pear tart -/
def pearSlices : ℕ := 10

/-- Theorem stating the number of fruits per slice for each type of pie/tart -/
theorem fruits_per_slice :
  (grannySmithDozens * dozenSize) / (grannySmithPies * grannySmithSlices) = 2 ∧
  (fujiDozens * dozenSize) / (fujiPies * fujiSlices) = 1 ∧
  (bartlettDozens * dozenSize : ℚ) / (pearTarts * pearSlices) = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_fruits_per_slice_l4154_415462


namespace NUMINAMATH_CALUDE_function_properties_l4154_415421

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

-- State the theorem
theorem function_properties (a : ℝ) (h : a > 0) :
  -- 1. Domain of f(x) is (0, 2)
  (∀ x, f a x ≠ Real.log 0 → 0 < x ∧ x < 2) ∧
  -- 2. When a = 1, f(x) is increasing on (0, √2) and decreasing on (√2, 2)
  (a = 1 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt 2 → f 1 x₁ < f 1 x₂) ∧
    (∀ x₁ x₂, Real.sqrt 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f 1 x₁ > f 1 x₂)) ∧
  -- 3. If the maximum value of f(x) on (0, 1] is 1/2, then a = 1/2
  ((∃ x, 0 < x ∧ x ≤ 1 ∧ f a x = 1/2 ∧ ∀ y, 0 < y ∧ y ≤ 1 → f a y ≤ 1/2) → a = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4154_415421


namespace NUMINAMATH_CALUDE_edward_total_spent_l4154_415456

/-- The total amount Edward spent on a board game and action figures -/
def total_spent (board_game_cost : ℕ) (num_figures : ℕ) (figure_cost : ℕ) : ℕ :=
  board_game_cost + num_figures * figure_cost

/-- Theorem stating that Edward spent $30 in total -/
theorem edward_total_spent :
  total_spent 2 4 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_edward_total_spent_l4154_415456


namespace NUMINAMATH_CALUDE_functional_equation_identity_l4154_415414

theorem functional_equation_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (f m + f n) = m + n) : 
  ∀ x : ℕ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l4154_415414


namespace NUMINAMATH_CALUDE_normal_distribution_probability_bagged_rice_probability_l4154_415427

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability density function of the normal distribution with mean μ and variance σ² -/
noncomputable def normalPDF (μ σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability (μ σ x₁ x₂ : ℝ) (hσ : σ > 0) :
  (∫ x in x₁..x₂, normalPDF μ σ x) = Φ ((x₂ - μ) / σ) - Φ ((x₁ - μ) / σ) :=
sorry

/-- The probability that a value from N(10, 0.01) is between 9.8 and 10.2 -/
theorem bagged_rice_probability :
  (∫ x in 9.8..10.2, normalPDF 10 0.1 x) = 2 * Φ 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_bagged_rice_probability_l4154_415427


namespace NUMINAMATH_CALUDE_spending_recording_l4154_415423

/-- Represents the recording of a financial transaction -/
def record (amount : ℤ) : ℤ := amount

/-- Axiom: Depositing is recorded as a positive amount -/
axiom deposit_positive (amount : ℕ) : record amount = amount

/-- The main theorem: If depositing 300 is recorded as +300, then spending 500 should be recorded as -500 -/
theorem spending_recording :
  record 300 = 300 → record (-500) = -500 := by
  sorry

end NUMINAMATH_CALUDE_spending_recording_l4154_415423


namespace NUMINAMATH_CALUDE_trig_expression_value_l4154_415434

theorem trig_expression_value (x : Real) (h : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l4154_415434


namespace NUMINAMATH_CALUDE_iron_conductivity_is_deductive_l4154_415482

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define iron as a constant in our universe
variable (iron : U)

-- State the premises and conclusion
variable (all_metals_conduct : ∀ x, Metal x → ConductsElectricity x)
variable (iron_is_metal : Metal iron)
variable (iron_conducts : ConductsElectricity iron)

-- Define deductive reasoning
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- Theorem stating that the given reasoning is deductive
theorem iron_conductivity_is_deductive :
  is_deductive_reasoning 
    (∀ x, Metal x → ConductsElectricity x)
    (Metal iron)
    (ConductsElectricity iron) :=
by sorry

end NUMINAMATH_CALUDE_iron_conductivity_is_deductive_l4154_415482


namespace NUMINAMATH_CALUDE_existence_of_index_l4154_415498

theorem existence_of_index (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_x : ∀ i, i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1/4) * x 1 * (1 - x n) := by
sorry

end NUMINAMATH_CALUDE_existence_of_index_l4154_415498


namespace NUMINAMATH_CALUDE_exactly_four_valid_labelings_l4154_415475

/-- Represents a truncated 3x3 chessboard with 8 squares. -/
structure TruncatedChessboard :=
  (labels : Fin 8 → Fin 8)

/-- Checks if two positions on the board are connected (share a vertex). -/
def are_connected (p1 p2 : Fin 8) : Bool :=
  sorry

/-- Checks if a labeling is valid according to the problem rules. -/
def is_valid_labeling (board : TruncatedChessboard) : Prop :=
  (∀ p1 p2 : Fin 8, p1 ≠ p2 → board.labels p1 ≠ board.labels p2) ∧
  (∀ p1 p2 : Fin 8, are_connected p1 p2 → 
    (board.labels p1).val + 1 ≠ (board.labels p2).val ∧
    (board.labels p2).val + 1 ≠ (board.labels p1).val)

/-- The main theorem stating that there are exactly 4 valid labelings. -/
theorem exactly_four_valid_labelings :
  ∃! (valid_labelings : Finset TruncatedChessboard),
    (∀ board ∈ valid_labelings, is_valid_labeling board) ∧
    valid_labelings.card = 4 :=
  sorry

end NUMINAMATH_CALUDE_exactly_four_valid_labelings_l4154_415475


namespace NUMINAMATH_CALUDE_consecutive_palindrome_diff_l4154_415458

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The set of all five-digit palindromes -/
def five_digit_palindromes : Set ℕ :=
  {n : ℕ | is_five_digit_palindrome n}

/-- The theorem stating the possible differences between consecutive five-digit palindromes -/
theorem consecutive_palindrome_diff 
  (a b : ℕ) 
  (ha : a ∈ five_digit_palindromes) 
  (hb : b ∈ five_digit_palindromes)
  (hless : a < b)
  (hconsec : ∀ x, x ∈ five_digit_palindromes → a < x → x < b → False) :
  b - a = 100 ∨ b - a = 110 ∨ b - a = 11 :=
sorry

end NUMINAMATH_CALUDE_consecutive_palindrome_diff_l4154_415458


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l4154_415474

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  7 * (4 - 2*i) + 4*i*(7 - 3*i) + 2*(5 + i) = 50 + 16*i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l4154_415474


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l4154_415477

theorem coefficient_x_cubed_in_expansion : 
  let expression := (fun x : ℝ => (x^2 + 1)^2 * (x - 1)^6)
  ∃ (a b c d e f g : ℝ), 
    (∀ x, expression x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + f*x^4 + (-32)*x^3 + g) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l4154_415477


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4154_415442

theorem complex_equation_solution (m : ℝ) (i : ℂ) : 
  i * i = -1 → (m + 2 * i) * (2 - i) = 4 + 3 * i → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4154_415442


namespace NUMINAMATH_CALUDE_both_in_picture_probability_l4154_415465

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (sarah sam : Runner) (pictureWidth : ℝ) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem both_in_picture_probability 
  (sarah : Runner) 
  (sam : Runner) 
  (sarah_laptime : sarah.lapTime = 120)
  (sam_laptime : sam.lapTime = 75)
  (sarah_direction : sarah.direction = true)
  (sam_direction : sam.direction = false)
  (picture_width : ℝ)
  (picture_covers_third : picture_width = sarah.lapTime / 3) :
  probabilityBothInPicture sarah sam picture_width = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_both_in_picture_probability_l4154_415465


namespace NUMINAMATH_CALUDE_initial_amount_sufficient_l4154_415440

/-- Kanul's initial amount of money -/
def initial_amount : ℝ := 11058.82

/-- Raw materials cost -/
def raw_materials_cost : ℝ := 5000

/-- Machinery cost -/
def machinery_cost : ℝ := 200

/-- Employee wages -/
def employee_wages : ℝ := 1200

/-- Maintenance cost percentage -/
def maintenance_percentage : ℝ := 0.15

/-- Desired remaining balance -/
def desired_balance : ℝ := 3000

/-- Theorem: Given the expenses and conditions, the initial amount is sufficient -/
theorem initial_amount_sufficient :
  initial_amount - (raw_materials_cost + machinery_cost + employee_wages + maintenance_percentage * initial_amount) ≥ desired_balance := by
  sorry

#check initial_amount_sufficient

end NUMINAMATH_CALUDE_initial_amount_sufficient_l4154_415440


namespace NUMINAMATH_CALUDE_gecko_count_l4154_415451

theorem gecko_count : 
  ∀ (gecko_count : ℕ) (lizard_count : ℕ) (insects_per_gecko : ℕ) (total_insects : ℕ),
    lizard_count = 3 →
    insects_per_gecko = 6 →
    total_insects = 66 →
    total_insects = gecko_count * insects_per_gecko + lizard_count * (2 * insects_per_gecko) →
    gecko_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_gecko_count_l4154_415451


namespace NUMINAMATH_CALUDE_point_coordinates_l4154_415405

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the 2D plane -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating that a point in the second quadrant with given distances from axes has specific coordinates -/
theorem point_coordinates (A : Point) 
    (h1 : secondQuadrant A) 
    (h2 : distanceFromXAxis A = 5) 
    (h3 : distanceFromYAxis A = 6) : 
  A.x = -6 ∧ A.y = 5 := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l4154_415405


namespace NUMINAMATH_CALUDE_train_speed_l4154_415480

/-- Given a train and platform with the following properties:
  * The train and platform have equal length
  * The train is 750 meters long
  * The train crosses the platform in one minute
  Prove that the speed of the train is 90 km/hr -/
theorem train_speed (train_length : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 750 →
  platform_length = train_length →
  crossing_time = 1 →
  (train_length + platform_length) / crossing_time * 60 / 1000 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4154_415480


namespace NUMINAMATH_CALUDE_selling_price_for_loss_is_40_l4154_415432

/-- The selling price that yields the same loss as the profit for an article -/
def selling_price_for_loss (cost_price : ℕ) (profit_selling_price : ℕ) : ℕ :=
  cost_price - (profit_selling_price - cost_price)

/-- Proof that the selling price for loss is 40 given the conditions -/
theorem selling_price_for_loss_is_40 :
  selling_price_for_loss 47 54 = 40 := by
  sorry

#eval selling_price_for_loss 47 54

end NUMINAMATH_CALUDE_selling_price_for_loss_is_40_l4154_415432


namespace NUMINAMATH_CALUDE_student_count_l4154_415492

theorem student_count (average_decrease : ℝ) (weight_difference : ℝ) : 
  average_decrease = 8 → weight_difference = 32 → (weight_difference / average_decrease : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l4154_415492


namespace NUMINAMATH_CALUDE_three_pairs_product_l4154_415496

theorem three_pairs_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 1005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 1004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 1005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 1004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 1005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 1004) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/502 := by
  sorry

end NUMINAMATH_CALUDE_three_pairs_product_l4154_415496


namespace NUMINAMATH_CALUDE_sandwich_combinations_l4154_415497

/-- The number of different kinds of lunch meats -/
def num_meats : ℕ := 12

/-- The number of different kinds of cheeses -/
def num_cheeses : ℕ := 8

/-- The number of ways to choose meats for a sandwich -/
def meat_choices : ℕ := Nat.choose num_meats 1 + Nat.choose num_meats 2

/-- The number of ways to choose cheeses for a sandwich -/
def cheese_choices : ℕ := Nat.choose num_cheeses 2

/-- The total number of different sandwiches that can be made -/
def total_sandwiches : ℕ := meat_choices * cheese_choices

theorem sandwich_combinations : total_sandwiches = 2184 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l4154_415497


namespace NUMINAMATH_CALUDE_weekdays_wearing_one_shirt_to_school_l4154_415494

def shirts_for_two_weeks : ℕ := 22

def after_school_club_days_per_week : ℕ := 3
def saturdays_per_week : ℕ := 1
def sundays_per_week : ℕ := 1
def weeks : ℕ := 2

def shirts_for_after_school_club : ℕ := after_school_club_days_per_week * weeks
def shirts_for_saturdays : ℕ := saturdays_per_week * weeks
def shirts_for_sundays : ℕ := 2 * sundays_per_week * weeks

def shirts_for_other_activities : ℕ := 
  shirts_for_after_school_club + shirts_for_saturdays + shirts_for_sundays

theorem weekdays_wearing_one_shirt_to_school : 
  (shirts_for_two_weeks - shirts_for_other_activities) / weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_weekdays_wearing_one_shirt_to_school_l4154_415494


namespace NUMINAMATH_CALUDE_fraction_problem_l4154_415466

theorem fraction_problem (f : ℚ) : 
  (f * 20 + 5 = 15) → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4154_415466


namespace NUMINAMATH_CALUDE_grade_assignments_count_l4154_415437

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- The number of ways to assign grades to all students -/
def num_assignments : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 16777216 -/
theorem grade_assignments_count : num_assignments = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_count_l4154_415437


namespace NUMINAMATH_CALUDE_factorization_equality_l4154_415472

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4154_415472


namespace NUMINAMATH_CALUDE_olivias_wallet_problem_l4154_415402

/-- Given an initial amount of 78 dollars and a spending of 15 dollars,
    the remaining amount is 63 dollars. -/
theorem olivias_wallet_problem (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 78 ∧ spent_amount = 15 → remaining_amount = initial_amount - spent_amount → remaining_amount = 63 := by
  sorry

end NUMINAMATH_CALUDE_olivias_wallet_problem_l4154_415402


namespace NUMINAMATH_CALUDE_min_coins_for_alex_l4154_415486

/-- The minimum number of additional coins needed for distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for the given scenario. -/
theorem min_coins_for_alex : min_additional_coins 15 63 = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_for_alex_l4154_415486


namespace NUMINAMATH_CALUDE_existence_of_r_l4154_415478

/-- Two infinite sequences of rational numbers -/
def s : ℕ → ℚ := sorry
def t : ℕ → ℚ := sorry

/-- Neither sequence is constant -/
axiom not_constant_s : ∃ i j, s i ≠ s j
axiom not_constant_t : ∃ i j, t i ≠ t j

/-- For any integers i and j, (sᵢ - sⱼ)(tᵢ - tⱼ) is an integer -/
axiom product_is_integer : ∀ i j : ℕ, ∃ k : ℤ, (s i - s j) * (t i - t j) = k

/-- The main theorem to be proved -/
theorem existence_of_r : ∃ r : ℚ, 
  (∀ i j : ℕ, ∃ m : ℤ, (s i - s j) * r = m) ∧ 
  (∀ i j : ℕ, ∃ n : ℤ, (t i - t j) / r = n) :=
sorry

end NUMINAMATH_CALUDE_existence_of_r_l4154_415478


namespace NUMINAMATH_CALUDE_harveys_steak_sales_l4154_415428

/-- Given the initial number of steaks, the number left after the first sale,
    and the number of additional steaks sold, calculate the total number of steaks sold. -/
def total_steaks_sold (initial : ℕ) (left_after_first_sale : ℕ) (additional_sold : ℕ) : ℕ :=
  (initial - left_after_first_sale) + additional_sold

/-- Theorem stating that for Harvey's specific case, the total number of steaks sold is 17. -/
theorem harveys_steak_sales : total_steaks_sold 25 12 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_harveys_steak_sales_l4154_415428


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l4154_415441

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem greatest_common_divisor_with_digit_sum : 
  ∃ (n : ℕ), 
    n ∣ (6905 - 4665) ∧ 
    sum_of_digits n = 4 ∧ 
    ∀ (m : ℕ), m ∣ (6905 - 4665) ∧ sum_of_digits m = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l4154_415441


namespace NUMINAMATH_CALUDE_unique_factors_of_135135_l4154_415457

theorem unique_factors_of_135135 :
  ∃! (a b c d e f : ℕ),
    1 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    a * b * c * d * e * f = 135135 ∧
    a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 ∧ e = 11 ∧ f = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_factors_of_135135_l4154_415457


namespace NUMINAMATH_CALUDE_round_table_seating_arrangements_l4154_415446

def num_people : ℕ := 6
def num_specific_people : ℕ := 2

theorem round_table_seating_arrangements :
  let num_units := num_people - num_specific_people + 1
  (num_specific_people.factorial) * ((num_units - 1).factorial) = 48 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seating_arrangements_l4154_415446


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4154_415406

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0) → 
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ 
  (c > 0 ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4154_415406


namespace NUMINAMATH_CALUDE_ab_value_l4154_415450

theorem ab_value (a b c : ℝ) 
  (eq1 : a - b = 5)
  (eq2 : a^2 + b^2 = 34)
  (eq3 : a^3 - b^3 = 30)
  (eq4 : a^2 + b^2 - c^2 = 50) : 
  a * b = 4.5 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l4154_415450


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4154_415473

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4154_415473


namespace NUMINAMATH_CALUDE_edric_work_hours_l4154_415448

/-- Calculates the number of hours worked per day given monthly salary, days worked per week, and hourly rate -/
def hours_per_day (monthly_salary : ℕ) (days_per_week : ℕ) (hourly_rate : ℕ) : ℕ :=
  let days_per_month := days_per_week * 4
  let total_hours := monthly_salary / hourly_rate
  total_hours / days_per_month

theorem edric_work_hours :
  hours_per_day 576 6 3 = 8 := by
  sorry

#eval hours_per_day 576 6 3

end NUMINAMATH_CALUDE_edric_work_hours_l4154_415448


namespace NUMINAMATH_CALUDE_initial_peaches_proof_l4154_415499

/-- The number of peaches Mike picked from the orchard -/
def peaches_picked : ℕ := 52

/-- The total number of peaches Mike has now -/
def total_peaches : ℕ := 86

/-- The initial number of peaches at Mike's roadside fruit dish -/
def initial_peaches : ℕ := total_peaches - peaches_picked

theorem initial_peaches_proof : initial_peaches = 34 := by
  sorry

end NUMINAMATH_CALUDE_initial_peaches_proof_l4154_415499


namespace NUMINAMATH_CALUDE_shaded_area_four_circles_l4154_415453

/-- The area of the shaded region formed by the intersection of four circles -/
theorem shaded_area_four_circles (r : ℝ) (h : r = 5) : 
  let circle_area := π * r^2
  let quarter_circle_area := circle_area / 4
  let triangle_area := r^2 / 2
  let shaded_segment := quarter_circle_area - triangle_area
  4 * shaded_segment = 25 * π - 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_four_circles_l4154_415453


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l4154_415411

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Question 1
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Question 2
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l4154_415411


namespace NUMINAMATH_CALUDE_range_of_a_l4154_415461

/-- A function f is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The logarithm function with base a -/
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- Proposition p: log_a x is monotonically increasing for x > 0 -/
def p (a : ℝ) : Prop :=
  MonotonicallyIncreasing (fun x => log_base a x)

/-- Proposition q: x^2 + ax + 1 > 0 for all real x -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + 1 > 0

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Ioc (-2) 1 ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4154_415461


namespace NUMINAMATH_CALUDE_cinnamon_distribution_exists_l4154_415420

/-- Represents the number of cinnamon swirls eaten by each person -/
structure CinnamonDistribution where
  jane : ℕ
  siblings : Fin 2 → ℕ
  cousins : Fin 5 → ℕ

/-- Theorem stating the existence of a valid cinnamon swirl distribution -/
theorem cinnamon_distribution_exists : ∃ (d : CinnamonDistribution), 
  -- Each person eats a different number of pieces
  (∀ (i j : Fin 2), i ≠ j → d.siblings i ≠ d.siblings j) ∧ 
  (∀ (i j : Fin 5), i ≠ j → d.cousins i ≠ d.cousins j) ∧
  (∀ (i : Fin 2) (j : Fin 5), d.siblings i ≠ d.cousins j) ∧
  (∀ (i : Fin 2), d.jane ≠ d.siblings i) ∧
  (∀ (j : Fin 5), d.jane ≠ d.cousins j) ∧
  -- Jane eats 1 fewer piece than her youngest sibling
  (∃ (i : Fin 2), d.jane + 1 = d.siblings i ∧ ∀ (j : Fin 2), d.siblings j ≥ d.siblings i) ∧
  -- Jane's youngest sibling eats 2 pieces more than one of her cousins
  (∃ (i : Fin 2) (j : Fin 5), d.siblings i = d.cousins j + 2 ∧ ∀ (k : Fin 2), d.siblings k ≥ d.siblings i) ∧
  -- The sum of all pieces eaten equals 50
  d.jane + (Finset.sum (Finset.univ : Finset (Fin 2)) d.siblings) + (Finset.sum (Finset.univ : Finset (Fin 5)) d.cousins) = 50 :=
sorry

end NUMINAMATH_CALUDE_cinnamon_distribution_exists_l4154_415420


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l4154_415407

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + 1 + 2*m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + 1 + 2*m = 0 → y = x) → 
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l4154_415407


namespace NUMINAMATH_CALUDE_two_integers_sum_l4154_415467

theorem two_integers_sum (x y : ℕ+) : x - y = 4 → x * y = 63 → x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l4154_415467


namespace NUMINAMATH_CALUDE_cards_left_l4154_415454

/-- Given that Nell had 242 cards initially and gave away 136 cards,
    prove that she has 106 cards left. -/
theorem cards_left (initial_cards given_away_cards : ℕ) 
  (h1 : initial_cards = 242)
  (h2 : given_away_cards = 136) :
  initial_cards - given_away_cards = 106 := by
  sorry

end NUMINAMATH_CALUDE_cards_left_l4154_415454


namespace NUMINAMATH_CALUDE_basketball_percentage_l4154_415433

theorem basketball_percentage (total_students : ℕ) (chess_percent : ℚ) (chess_or_basketball : ℕ) : 
  total_students = 250 →
  chess_percent = 1/10 →
  chess_or_basketball = 125 →
  ∃ (basketball_percent : ℚ), 
    basketball_percent = 2/5 ∧ 
    (basketball_percent + chess_percent) * total_students = chess_or_basketball :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_percentage_l4154_415433


namespace NUMINAMATH_CALUDE_certain_number_problem_l4154_415463

theorem certain_number_problem (x : ℝ) : 45 * 7 = 0.35 * x → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4154_415463


namespace NUMINAMATH_CALUDE_problem_statement_l4154_415488

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  (y / x = 1) ∧
  (∃ (A B C : ℝ), Real.tan C = y / x ∧ 
    (∀ A' B' C' : ℝ, Real.tan C' = y / x → 
      Real.sin (2*A') + 2 * Real.cos B' ≤ Real.sin (2*A) + 2 * Real.cos B) ∧
    Real.sin (2*A) + 2 * Real.cos B = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4154_415488


namespace NUMINAMATH_CALUDE_student_count_l4154_415408

theorem student_count (n : ℕ) 
  (yellow : ℕ) (red : ℕ) (blue : ℕ) 
  (yellow_blue : ℕ) (yellow_red : ℕ) (blue_red : ℕ)
  (all_colors : ℕ) :
  yellow = 46 →
  red = 69 →
  blue = 104 →
  yellow_blue = 14 →
  yellow_red = 13 →
  blue_red = 19 →
  all_colors = 16 →
  n = yellow + red + blue - yellow_blue - yellow_red - blue_red + all_colors →
  n = 141 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l4154_415408


namespace NUMINAMATH_CALUDE_zelda_success_probability_l4154_415469

theorem zelda_success_probability 
  (p_xavier : ℝ) 
  (p_yvonne : ℝ) 
  (p_xy_not_z : ℝ) 
  (h1 : p_xavier = 1/4)
  (h2 : p_yvonne = 2/3)
  (h3 : p_xy_not_z = 0.0625)
  (h4 : p_xy_not_z = p_xavier * p_yvonne * (1 - p_zelda)) :
  p_zelda = 5/8 := by
sorry

end NUMINAMATH_CALUDE_zelda_success_probability_l4154_415469


namespace NUMINAMATH_CALUDE_units_digit_of_A_is_1_l4154_415438

-- Define the sequence of powers of 3
def powerOf3 : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * powerOf3 n

-- Define A
def A : ℕ := 2 * (3 + 1) * (powerOf3 2 + 1) * (powerOf3 4 + 1) + 1

-- Theorem statement
theorem units_digit_of_A_is_1 : A % 10 = 1 := by
  sorry


end NUMINAMATH_CALUDE_units_digit_of_A_is_1_l4154_415438


namespace NUMINAMATH_CALUDE_quadrilateral_area_with_diagonal_and_offsets_l4154_415459

/-- The area of a quadrilateral with a diagonal and its offsets -/
theorem quadrilateral_area_with_diagonal_and_offsets 
  (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) :
  diagonal = 40 → offset1 = 9 → offset2 = 6 →
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 300 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_with_diagonal_and_offsets_l4154_415459


namespace NUMINAMATH_CALUDE_fields_medal_stats_l4154_415481

def data_set : List ℕ := [29, 32, 33, 35, 35, 40]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem fields_medal_stats : 
  mode data_set = 35 ∧ median data_set = 34 := by sorry

end NUMINAMATH_CALUDE_fields_medal_stats_l4154_415481


namespace NUMINAMATH_CALUDE_max_sin_a_value_l4154_415436

open Real

theorem max_sin_a_value (a b c : ℝ) 
  (h1 : cos a = tan b) 
  (h2 : cos b = tan c) 
  (h3 : cos c = tan a) : 
  ∃ (max_sin_a : ℝ), (∀ a' b' c' : ℝ, cos a' = tan b' → cos b' = tan c' → cos c' = tan a' → sin a' ≤ max_sin_a) ∧ max_sin_a = (sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_sin_a_value_l4154_415436


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l4154_415491

/-- Parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {(x, y) | y = x^2}

/-- Point Q -/
def Q : ℝ × ℝ := (10, 4)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) := {(x, y) | y - 4 = m * (x - 10)}

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop := line_through_Q m ∩ P = ∅

theorem parabola_line_intersection (r s : ℝ) :
  (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l4154_415491


namespace NUMINAMATH_CALUDE_fraction_addition_l4154_415418

theorem fraction_addition (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) :
  (a + b) / b = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4154_415418


namespace NUMINAMATH_CALUDE_cube_volume_increase_l4154_415409

theorem cube_volume_increase (a : ℝ) (ha : a > 0) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l4154_415409


namespace NUMINAMATH_CALUDE_pet_store_puppies_l4154_415464

/-- The number of puppies sold -/
def puppies_sold : ℕ := 3

/-- The number of cages used -/
def cages_used : ℕ := 3

/-- The number of puppies in each cage -/
def puppies_per_cage : ℕ := 5

/-- The initial number of puppies in the pet store -/
def initial_puppies : ℕ := puppies_sold + cages_used * puppies_per_cage

theorem pet_store_puppies : initial_puppies = 18 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l4154_415464


namespace NUMINAMATH_CALUDE_minimal_disks_is_16_l4154_415493

-- Define the problem parameters
def total_files : ℕ := 42
def disk_capacity : ℚ := 2.88
def large_files : ℕ := 8
def medium_files : ℕ := 16
def large_file_size : ℚ := 1.6
def medium_file_size : ℚ := 1
def small_file_size : ℚ := 0.5

-- Define the function to calculate the minimal number of disks
def minimal_disks : ℕ := sorry

-- State the theorem
theorem minimal_disks_is_16 : minimal_disks = 16 := by sorry

end NUMINAMATH_CALUDE_minimal_disks_is_16_l4154_415493


namespace NUMINAMATH_CALUDE_last_digit_divisibility_l4154_415424

theorem last_digit_divisibility (n : ℕ) (h : n > 3) :
  let a := (2^n) % 10
  let b := 2^n - a
  6 ∣ (a * b) := by sorry

end NUMINAMATH_CALUDE_last_digit_divisibility_l4154_415424


namespace NUMINAMATH_CALUDE_only_consecutive_primes_fifth_power_difference_prime_l4154_415470

theorem only_consecutive_primes_fifth_power_difference_prime :
  ∀ p q : ℕ,
    Prime p → Prime q → p > q →
    Prime (p^5 - q^5) →
    p = 3 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_consecutive_primes_fifth_power_difference_prime_l4154_415470


namespace NUMINAMATH_CALUDE_man_birth_year_proof_l4154_415415

theorem man_birth_year_proof : ∃! x : ℤ,
  x^2 - x = 1640 ∧
  2*(x + 2*x) = 2*x ∧
  x^2 - x < 1825 := by
  sorry

end NUMINAMATH_CALUDE_man_birth_year_proof_l4154_415415


namespace NUMINAMATH_CALUDE_bouquet_carnations_fraction_l4154_415447

theorem bouquet_carnations_fraction (total_flowers : ℕ) 
  (blue_flowers red_flowers blue_roses red_roses blue_carnations red_carnations : ℕ) :
  (blue_flowers = red_flowers) →  -- Half of the flowers are blue
  (blue_flowers + red_flowers = total_flowers) →
  (blue_roses = 2 * blue_flowers / 5) →  -- Two-fifths of blue flowers are roses
  (red_carnations = 2 * red_flowers / 3) →  -- Two-thirds of red flowers are carnations
  (blue_carnations = blue_flowers - blue_roses) →
  (red_roses = red_flowers - red_carnations) →
  ((blue_carnations + red_carnations : ℚ) / total_flowers = 19 / 30) := by
sorry

end NUMINAMATH_CALUDE_bouquet_carnations_fraction_l4154_415447


namespace NUMINAMATH_CALUDE_ab_value_l4154_415401

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l4154_415401


namespace NUMINAMATH_CALUDE_scientific_notation_of_384000_l4154_415484

/-- Given a number 384000, prove that its scientific notation representation is 3.84 × 10^5 -/
theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * (10 : ℝ)^5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_384000_l4154_415484


namespace NUMINAMATH_CALUDE_candies_remaining_l4154_415425

def vasya_eat (n : ℕ) : ℕ := n - (1 + (n - 9) / 7)

def petya_eat (n : ℕ) : ℕ := n - (1 + (n - 7) / 9)

theorem candies_remaining (initial_candies : ℕ) : 
  initial_candies = 1000 → petya_eat (vasya_eat initial_candies) = 761 := by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_l4154_415425


namespace NUMINAMATH_CALUDE_max_greece_value_l4154_415495

/-- Represents a mapping from letters to digits -/
def LetterMap := Char → Nat

/-- Check if a LetterMap is valid according to the problem conditions -/
def isValidMapping (m : LetterMap) : Prop :=
  (∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧ 
  (∀ c, m c ≤ 9) ∧
  m 'G' ≠ 0 ∧ m 'E' ≠ 0 ∧ m 'V' ≠ 0 ∧ m 'I' ≠ 0

/-- Convert a string of letters to a number using the given mapping -/
def stringToNumber (m : LetterMap) (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + m c) 0

/-- Check if the equation holds for a given mapping -/
def equationHolds (m : LetterMap) : Prop :=
  (stringToNumber m "VER" - stringToNumber m "IA") = 
  (m 'G')^((m 'R')^(m 'E')) * (stringToNumber m "GRE" + stringToNumber m "ECE")

/-- The main theorem to be proved -/
theorem max_greece_value (m : LetterMap) :
  isValidMapping m →
  equationHolds m →
  (∀ m', isValidMapping m' → equationHolds m' → 
    stringToNumber m' "GREECE" ≤ stringToNumber m "GREECE") →
  stringToNumber m "GREECE" = 196646 := by
  sorry

end NUMINAMATH_CALUDE_max_greece_value_l4154_415495


namespace NUMINAMATH_CALUDE_exists_even_non_zero_from_step_two_l4154_415483

/-- Represents the state of the sequence at a given step -/
def SequenceState := ℤ → ℤ

/-- The initial state of the sequence -/
def initial_state : SequenceState :=
  fun i => if i = 0 then 1 else 0

/-- Updates the sequence for one step -/
def update_sequence (s : SequenceState) : SequenceState :=
  fun i => s (i - 1) + s i + s (i + 1)

/-- Checks if a number is even and non-zero -/
def is_even_non_zero (n : ℤ) : Prop :=
  n ≠ 0 ∧ n % 2 = 0

/-- The sequence after n steps -/
def sequence_at_step (n : ℕ) : SequenceState :=
  match n with
  | 0 => initial_state
  | n + 1 => update_sequence (sequence_at_step n)

/-- The main theorem to be proved -/
theorem exists_even_non_zero_from_step_two (n : ℕ) (h : n ≥ 2) :
  ∃ i : ℤ, is_even_non_zero ((sequence_at_step n) i) :=
sorry

end NUMINAMATH_CALUDE_exists_even_non_zero_from_step_two_l4154_415483


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4154_415439

-- Problem 1
theorem problem_1 : -4 * 9 = -36 := by sorry

-- Problem 2
theorem problem_2 : 10 - 14 - (-5) = 1 := by sorry

-- Problem 3
theorem problem_3 : -3 * (-1/3)^3 = 1/9 := by sorry

-- Problem 4
theorem problem_4 : -56 + (-8) * (1/8) = -57 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4154_415439


namespace NUMINAMATH_CALUDE_sophie_germain_identity_l4154_415490

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4*b^4 = (a^2 + 2*a*b + 2*b^2) * (a^2 - 2*a*b + 2*b^2) := by
  sorry

end NUMINAMATH_CALUDE_sophie_germain_identity_l4154_415490


namespace NUMINAMATH_CALUDE_bus_speed_is_40_l4154_415476

/-- Represents the scenario of a bus and cyclist traveling between points A, B, C, and D. -/
structure TravelScenario where
  distance_AB : ℝ
  time_to_C : ℝ
  distance_CD : ℝ
  bus_speed : ℝ
  cyclist_speed : ℝ

/-- The travel scenario satisfies the given conditions. -/
def satisfies_conditions (s : TravelScenario) : Prop :=
  s.distance_AB = 4 ∧
  s.time_to_C = 1/6 ∧
  s.distance_CD = 2/3 ∧
  s.bus_speed > 0 ∧
  s.cyclist_speed > 0 ∧
  s.bus_speed > s.cyclist_speed

/-- The theorem stating that under the given conditions, the bus speed is 40 km/h. -/
theorem bus_speed_is_40 (s : TravelScenario) (h : satisfies_conditions s) : 
  s.bus_speed = 40 := by
  sorry

#check bus_speed_is_40

end NUMINAMATH_CALUDE_bus_speed_is_40_l4154_415476


namespace NUMINAMATH_CALUDE_congruence_problem_l4154_415449

theorem congruence_problem (y : ℤ) 
  (h1 : (4 + y) % (2^4) = 3^2 % (2^4))
  (h2 : (6 + y) % (3^4) = 2^3 % (3^4))
  (h3 : (8 + y) % (5^4) = 7^2 % (5^4)) :
  y % 360 = 317 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4154_415449


namespace NUMINAMATH_CALUDE_basketball_games_l4154_415430

theorem basketball_games (c : ℕ) : 
  (3 * c / 4 : ℚ) = (7 * c / 10 : ℚ) - 5 ∧ 
  (c / 4 : ℚ) = (3 * c / 10 : ℚ) - 5 → 
  c = 100 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_l4154_415430


namespace NUMINAMATH_CALUDE_square_starts_with_self_l4154_415413

def starts_with (a b : ℕ) : Prop :=
  ∃ k, a = b * 10^k + (a % 10^k)

theorem square_starts_with_self (N : ℕ) :
  (N > 0) → (starts_with (N^2) N) → ∃ k, N = 10^(k-1) :=
sorry

end NUMINAMATH_CALUDE_square_starts_with_self_l4154_415413
