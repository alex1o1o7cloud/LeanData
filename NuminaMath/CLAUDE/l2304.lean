import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2304_230411

theorem quadratic_inequality_solution (b : ℝ) :
  (∀ x, x^2 - 3*x + 6 > 4 ↔ (x < 1 ∨ x > b)) →
  (b = 2 ∧
   ∀ c, 
     (c > 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ 2 < x ∧ x < c) ∧
     (c < 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ c < x ∧ x < 2) ∧
     (c = 2 → ∀ x, ¬(x^2 - (c+2)*x + 2*c < 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2304_230411


namespace NUMINAMATH_CALUDE_reflection_sum_l2304_230402

theorem reflection_sum (x : ℝ) : 
  let C : ℝ × ℝ := (x, -3)
  let D : ℝ × ℝ := (-x, -3)
  (C.1 + C.2 + D.1 + D.2) = -6 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l2304_230402


namespace NUMINAMATH_CALUDE_prime_product_divisors_l2304_230405

theorem prime_product_divisors (p q : ℕ) (n : ℕ) : 
  Prime p → Prime q → (Finset.card (Nat.divisors (p^n * q^7)) = 56) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_divisors_l2304_230405


namespace NUMINAMATH_CALUDE_problem_solution_l2304_230499

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2304_230499


namespace NUMINAMATH_CALUDE_platform_length_l2304_230457

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (train_length : ℝ)
  (h1 : train_speed = 72)  -- km/hr
  (h2 : crossing_time = 26)  -- seconds
  (h3 : train_length = 250)  -- meters
  : (train_speed * (5/18) * crossing_time) - train_length = 270 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2304_230457


namespace NUMINAMATH_CALUDE_arrangement_problem_l2304_230484

/-- The number of ways to arrange people in a row -/
def arrange (n : ℕ) (m : ℕ) : ℕ :=
  n.factorial * m.factorial * (n + 1).factorial

/-- The problem statement -/
theorem arrangement_problem : arrange 5 2 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_problem_l2304_230484


namespace NUMINAMATH_CALUDE_book_cost_calculation_l2304_230495

theorem book_cost_calculation (initial_amount : ℕ) (books_bought : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  books_bought = 9 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / books_bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l2304_230495


namespace NUMINAMATH_CALUDE_multiple_of_nine_between_15_and_30_l2304_230463

theorem multiple_of_nine_between_15_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 225)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_between_15_and_30_l2304_230463


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2304_230497

theorem quadratic_inequality_solution_sets
  (a b : ℝ)
  (h : Set.Ioo (-1 : ℝ) (1/2) = {x | a * x^2 + b * x + 3 > 0}) :
  Set.Ioo (-1 : ℝ) 2 = {x | 3 * x^2 + b * x + a < 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2304_230497


namespace NUMINAMATH_CALUDE_balls_sold_prove_balls_sold_l2304_230460

/-- The number of balls sold given the cost price, selling price, and loss condition. -/
theorem balls_sold (cost_price : ℕ) (selling_price : ℕ) (loss : ℕ) : ℕ :=
  let n : ℕ := (selling_price + loss) / cost_price
  n

/-- Prove that 13 balls were sold given the problem conditions. -/
theorem prove_balls_sold :
  balls_sold 90 720 (5 * 90) = 13 := by
  sorry

end NUMINAMATH_CALUDE_balls_sold_prove_balls_sold_l2304_230460


namespace NUMINAMATH_CALUDE_special_numbers_count_l2304_230489

/-- Sum of digits of a positive integer -/
def heartsuit (n : ℕ+) : ℕ :=
  sorry

/-- Counts the number of three-digit positive integers x such that heartsuit(heartsuit(x)) = 5 -/
def count_special_numbers : ℕ :=
  sorry

/-- Theorem stating that there are exactly 60 three-digit positive integers x 
    such that heartsuit(heartsuit(x)) = 5 -/
theorem special_numbers_count : count_special_numbers = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_count_l2304_230489


namespace NUMINAMATH_CALUDE_base7_305_eq_base5_1102_l2304_230486

/-- Converts a base-7 number to its decimal (base-10) representation -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal (base-10) number to its base-5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- States that the base-7 number 305 is equal to the base-5 number 1102 -/
theorem base7_305_eq_base5_1102 :
  decimalToBase5 (base7ToDecimal [5, 0, 3]) = [1, 1, 0, 2] := by
  sorry

#eval base7ToDecimal [5, 0, 3]
#eval decimalToBase5 152

end NUMINAMATH_CALUDE_base7_305_eq_base5_1102_l2304_230486


namespace NUMINAMATH_CALUDE_cell_division_chromosome_count_l2304_230427

/-- Represents the number of chromosomes in a fruit fly cell -/
def ChromosomeCount : ℕ := 8

/-- Represents the possible chromosome counts during cell division -/
def PossibleChromosomeCounts : Set ℕ := {8, 16}

/-- Represents a genotype with four alleles -/
structure Genotype :=
  (allele1 allele2 allele3 allele4 : Char)

/-- Represents a fruit fly cell -/
structure FruitFlyCell :=
  (genotype : Genotype)
  (chromosomeCount : ℕ)

/-- Axiom: Fruit flies have 2N=8 chromosomes -/
axiom fruit_fly_chromosome_count : ChromosomeCount = 8

/-- Axiom: Alleles A/a and B/b are inherited independently -/
axiom alleles_independent : True

/-- Theorem: A fruit fly cell with genotype AAaaBBbb during cell division
    contains either 8 or 16 chromosomes -/
theorem cell_division_chromosome_count
  (cell : FruitFlyCell)
  (h_genotype : cell.genotype = ⟨'A', 'A', 'B', 'B'⟩ ∨
                cell.genotype = ⟨'a', 'a', 'b', 'b'⟩) :
  cell.chromosomeCount ∈ PossibleChromosomeCounts := by
  sorry

end NUMINAMATH_CALUDE_cell_division_chromosome_count_l2304_230427


namespace NUMINAMATH_CALUDE_y_derivative_l2304_230440

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (1 - 3*x - 2*x^2) + (3 / (2 * Real.sqrt 2)) * Real.arcsin ((4*x + 3) / Real.sqrt 17)

theorem y_derivative (x : ℝ) : 
  deriv y x = -(2*x) / Real.sqrt (1 - 3*x - 2*x^2) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l2304_230440


namespace NUMINAMATH_CALUDE_cube_distance_to_plane_l2304_230433

theorem cube_distance_to_plane (cube_side : ℝ) (height1 height2 height3 : ℝ) 
  (r s t : ℕ+) (d : ℝ) :
  cube_side = 15 →
  height1 = 15 ∧ height2 = 16 ∧ height3 = 17 →
  d = (r : ℝ) - Real.sqrt s / (t : ℝ) →
  d = (48 - Real.sqrt 224) / 3 →
  r + s + t = 275 := by
sorry

end NUMINAMATH_CALUDE_cube_distance_to_plane_l2304_230433


namespace NUMINAMATH_CALUDE_log_meaningful_implies_t_range_p_sufficient_for_q_implies_a_range_l2304_230412

-- Define the propositions
def p (a t : ℝ) : Prop := -2 * t^2 + 7 * t - 5 > 0
def q (a t : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

theorem log_meaningful_implies_t_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ t : ℝ, p a t → 1 < t ∧ t < 5/2 :=
sorry

theorem p_sufficient_for_q_implies_a_range :
  ∀ a : ℝ, (∀ t : ℝ, 1 < t ∧ t < 5/2 → q a t) → a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_log_meaningful_implies_t_range_p_sufficient_for_q_implies_a_range_l2304_230412


namespace NUMINAMATH_CALUDE_f_positive_iff_a_in_range_l2304_230409

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.log (1 / (a * x + a)) - a

theorem f_positive_iff_a_in_range (a : ℝ) :
  (a > 0 ∧ ∀ x, f a x > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_a_in_range_l2304_230409


namespace NUMINAMATH_CALUDE_hcf_problem_l2304_230458

theorem hcf_problem (A B : ℕ) (H : ℕ) : 
  A = 900 → 
  A > B → 
  B > 0 →
  Nat.lcm A B = H * 11 * 15 →
  Nat.gcd A B = H →
  Nat.gcd A B = 165 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l2304_230458


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2304_230421

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) :=
sorry

end

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2304_230421


namespace NUMINAMATH_CALUDE_evaluate_expression_l2304_230474

theorem evaluate_expression : -(16 / 4 * 12 - 100 + 2^3 * 6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2304_230474


namespace NUMINAMATH_CALUDE_angle_relationships_l2304_230400

theorem angle_relationships (A B C : ℝ) : 
  A + B = 180 →  -- A and B are supplementary
  C = B / 2 →    -- C is half of B
  A = 6 * B →    -- A is 6 times B
  (A = 1080 / 7 ∧ B = 180 / 7 ∧ C = 90 / 7) := by
  sorry

end NUMINAMATH_CALUDE_angle_relationships_l2304_230400


namespace NUMINAMATH_CALUDE_solution_exists_l2304_230407

-- Define the vector type
def Vec2 := Fin 2 → ℝ

-- Define the constants a and b
variable (a b : ℝ)

-- Define the vectors
def v1 : Vec2 := ![1, 4]
def v2 : Vec2 := ![3, -2]
def result : Vec2 := ![5, 6]

-- Define vector addition and scalar multiplication
def add (u v : Vec2) : Vec2 := λ i => u i + v i
def smul (c : ℝ) (v : Vec2) : Vec2 := λ i => c * v i

-- State the theorem
theorem solution_exists :
  ∃ a b : ℝ, add (smul a v1) (smul b v2) = result ∧ a = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2304_230407


namespace NUMINAMATH_CALUDE_simple_interest_time_l2304_230418

/-- Simple interest calculation -/
theorem simple_interest_time (principal rate interest : ℝ) :
  principal > 0 →
  rate > 0 →
  interest > 0 →
  (interest * 100) / (principal * rate) = 2 →
  principal = 400 →
  rate = 12.5 →
  interest = 100 →
  (interest * 100) / (principal * rate) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_l2304_230418


namespace NUMINAMATH_CALUDE_min_value_expression_l2304_230416

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 3 * Real.sqrt 2 ∧
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) = 3 * Real.sqrt 2 ↔ y = x * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2304_230416


namespace NUMINAMATH_CALUDE_amc10_min_correct_problems_l2304_230428

/-- The AMC 10 scoring system and Sarah's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Nat
  unanswered_points : Nat
  target_score : Nat

/-- The minimum number of correctly solved problems to reach the target score -/
def min_correct_problems (amc : AMC10) : Nat :=
  let unanswered := amc.total_problems - amc.attempted_problems
  let unanswered_score := unanswered * amc.unanswered_points
  let required_score := amc.target_score - unanswered_score
  (required_score + amc.correct_points - 1) / amc.correct_points

/-- Theorem stating that for the given AMC 10 configuration, 
    the minimum number of correctly solved problems is 20 -/
theorem amc10_min_correct_problems :
  let amc : AMC10 := {
    total_problems := 30,
    attempted_problems := 25,
    correct_points := 7,
    unanswered_points := 2,
    target_score := 150
  }
  min_correct_problems amc = 20 := by
  sorry

end NUMINAMATH_CALUDE_amc10_min_correct_problems_l2304_230428


namespace NUMINAMATH_CALUDE_unique_relation_sum_l2304_230425

theorem unique_relation_sum (a b c : ℕ) : 
  ({a, b, c} : Set ℕ) = {1, 2, 3} →
  (((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∨ (a ≠ 3 ∧ b = 3 ∧ c ≠ 3) ∨ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∧ (a ≠ 3 ∧ b = 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b = 3 ∧ c ≠ 3) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3))) →
  100 * a + 10 * b + c = 312 := by
sorry

end NUMINAMATH_CALUDE_unique_relation_sum_l2304_230425


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2304_230420

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = 1 / x}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2304_230420


namespace NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l2304_230438

theorem modulus_of_5_minus_12i : Complex.abs (5 - 12 * Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l2304_230438


namespace NUMINAMATH_CALUDE_modulus_of_z_l2304_230465

theorem modulus_of_z (z : ℂ) (h : z / (1 - z) = Complex.I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2304_230465


namespace NUMINAMATH_CALUDE_elimination_theorem_l2304_230455

theorem elimination_theorem (x y a b c : ℝ) 
  (ha : a = x + y) 
  (hb : b = x^3 + y^3) 
  (hc : c = x^5 + y^5) : 
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) := by
  sorry

end NUMINAMATH_CALUDE_elimination_theorem_l2304_230455


namespace NUMINAMATH_CALUDE_quadratic_rewrite_proof_l2304_230476

theorem quadratic_rewrite_proof :
  ∃ (a b c : ℚ), 
    (∀ k, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧
    (c + 3 * b = -49 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_proof_l2304_230476


namespace NUMINAMATH_CALUDE_money_division_l2304_230432

theorem money_division (a b c : ℝ) (h1 : a = (1/2) * b) (h2 : b = (1/2) * c) (h3 : c = 208) :
  a + b + c = 364 := by sorry

end NUMINAMATH_CALUDE_money_division_l2304_230432


namespace NUMINAMATH_CALUDE_duty_roster_theorem_l2304_230423

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements where two specific people are adjacent -/
def adjacent_arrangements (n : ℕ) : ℕ := 2 * permutations (n - 1)

/-- The number of arrangements where both pairs of specific people are adjacent -/
def both_adjacent_arrangements (n : ℕ) : ℕ := 2 * 2 * permutations (n - 2)

/-- The number of valid arrangements for the duty roster problem -/
def duty_roster_arrangements (n : ℕ) : ℕ :=
  permutations n - 2 * adjacent_arrangements n + both_adjacent_arrangements n

theorem duty_roster_theorem :
  duty_roster_arrangements 6 = 336 := by sorry

end NUMINAMATH_CALUDE_duty_roster_theorem_l2304_230423


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2304_230464

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2304_230464


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2304_230435

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  (∃ x y : ℕ, n = 10 * x + y ∧ 
    10 ≤ x + y ∧ x + y < 100 ∧
    x = y / 4 ∧
    n = 28) :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2304_230435


namespace NUMINAMATH_CALUDE_parabola_tangent_problem_l2304_230456

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the point M
def point_M (p : ℝ) : ℝ × ℝ := (2, -2*p)

-- Define a line touching the parabola at two points
def touching_line (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  ∃ (m c : ℝ), A.2 = m * A.1 + c ∧ B.2 = m * B.1 + c ∧
  point_M p = (2, m * 2 + c)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  (A.2 + B.2) / 2 = 6

-- Theorem statement
theorem parabola_tangent_problem (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  touching_line p A B →
  midpoint_condition A B →
  p = 1 ∨ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_problem_l2304_230456


namespace NUMINAMATH_CALUDE_boys_from_clay_l2304_230422

/-- Represents the number of students from each school and gender --/
structure StudentCounts where
  total : Nat
  boys : Nat
  girls : Nat
  jonas : Nat
  clay : Nat
  pine : Nat
  jonasGirls : Nat
  pineBoys : Nat

/-- Theorem stating that the number of boys from Clay Middle School is 40 --/
theorem boys_from_clay (s : StudentCounts)
  (h_total : s.total = 120)
  (h_boys : s.boys = 70)
  (h_girls : s.girls = 50)
  (h_jonas : s.jonas = 50)
  (h_clay : s.clay = 40)
  (h_pine : s.pine = 30)
  (h_jonasGirls : s.jonasGirls = 30)
  (h_pineBoys : s.pineBoys = 10)
  : s.clay - (s.girls - s.jonasGirls - (s.pine - s.pineBoys)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_from_clay_l2304_230422


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2304_230468

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ f c) ∧
  f c = 23 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2304_230468


namespace NUMINAMATH_CALUDE_d_share_is_thirteen_sixtieths_l2304_230442

/-- Represents the capital shares of partners in a business. -/
structure CapitalShares where
  total : ℚ
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  a_share : a = (1 : ℚ) / 3 * total
  b_share : b = (1 : ℚ) / 4 * total
  c_share : c = (1 : ℚ) / 5 * total
  total_sum : a + b + c + d = total

/-- Represents the profit distribution in the business. -/
structure ProfitDistribution where
  total : ℚ
  a_profit : ℚ
  total_amount : total = 2490
  a_amount : a_profit = 830

/-- Theorem stating that given the capital shares and profit distribution,
    partner D's share of the capital is 13/60. -/
theorem d_share_is_thirteen_sixtieths
  (shares : CapitalShares) (profit : ProfitDistribution) :
  shares.d = (13 : ℚ) / 60 * shares.total :=
sorry

end NUMINAMATH_CALUDE_d_share_is_thirteen_sixtieths_l2304_230442


namespace NUMINAMATH_CALUDE_derivative_of_fraction_l2304_230436

open Real

theorem derivative_of_fraction (x : ℝ) (h : x > 0) :
  deriv (λ x => (1 - log x) / (1 + log x)) x = -2 / (x * (1 + log x)^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_fraction_l2304_230436


namespace NUMINAMATH_CALUDE_sue_falls_count_l2304_230448

structure Friend where
  name : String
  age : Nat
  falls : Nat

def steven : Friend := { name := "Steven", age := 20, falls := 3 }
def stephanie : Friend := { name := "Stephanie", age := 24, falls := steven.falls + 13 }
def sam : Friend := { name := "Sam", age := 24, falls := 1 }
def sue : Friend := { name := "Sue", age := 26, falls := 0 }  -- falls will be calculated

def sonya_falls : Nat := stephanie.falls / 2 - 2
def sophie_falls : Nat := sam.falls + 4

def youngest_age : Nat := min steven.age (min stephanie.age (min sam.age sue.age))

theorem sue_falls_count :
  sue.falls = sue.age - youngest_age :=
by sorry

end NUMINAMATH_CALUDE_sue_falls_count_l2304_230448


namespace NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2304_230475

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to the line y = 1 -/
  tangent_to_y1 : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_to_yaxis : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- The length of the major axis of the special ellipse -/
def majorAxisLength (e : SpecialEllipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 2 for the given ellipse -/
theorem special_ellipse_major_axis_length :
  ∀ (e : SpecialEllipse),
    e.tangent_to_y1 = true →
    e.tangent_to_yaxis = true →
    e.focus1 = (3, 2 + Real.sqrt 2) →
    e.focus2 = (3, 2 - Real.sqrt 2) →
    majorAxisLength e = 2 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2304_230475


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2304_230434

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2304_230434


namespace NUMINAMATH_CALUDE_add_25_to_number_l2304_230459

theorem add_25_to_number (x : ℤ) : 43 + x = 81 → x + 25 = 63 := by
  sorry

end NUMINAMATH_CALUDE_add_25_to_number_l2304_230459


namespace NUMINAMATH_CALUDE_inequality_proof_l2304_230477

theorem inequality_proof (x y z : ℤ) :
  (x^2 + y^2*z^2) * (y^2 + x^2*z^2) * (z^2 + x^2*y^2) ≥ 8*x*y^2*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2304_230477


namespace NUMINAMATH_CALUDE_square_of_9_divided_by_cube_root_of_125_remainder_l2304_230403

theorem square_of_9_divided_by_cube_root_of_125_remainder (n m q r : ℕ) : 
  n = 9^2 → 
  m = 5 → 
  n = m * q + r → 
  r < m → 
  r = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_square_of_9_divided_by_cube_root_of_125_remainder_l2304_230403


namespace NUMINAMATH_CALUDE_eunji_pocket_money_l2304_230479

theorem eunji_pocket_money (initial_money : ℕ) : 
  (initial_money / 4 : ℕ) + 
  ((3 * initial_money / 4) / 3 : ℕ) + 
  1600 = initial_money → 
  initial_money = 3200 := by
sorry

end NUMINAMATH_CALUDE_eunji_pocket_money_l2304_230479


namespace NUMINAMATH_CALUDE_gcd_459_357_l2304_230487

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2304_230487


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l2304_230466

theorem arithmetic_progression_first_term
  (d : ℝ)
  (a₁₂ : ℝ)
  (h₁ : d = 8)
  (h₂ : a₁₂ = 90)
  : ∃ (a₁ : ℝ), a₁₂ = a₁ + (12 - 1) * d ∧ a₁ = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l2304_230466


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2304_230473

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger
  (jogger_speed : Real)
  (train_speed : Real)
  (train_length : Real)
  (initial_distance : Real)
  (h1 : jogger_speed = 9 * (1000 / 3600))
  (h2 : train_speed = 45 * (1000 / 3600))
  (h3 : train_length = 120)
  (h4 : initial_distance = 250) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l2304_230473


namespace NUMINAMATH_CALUDE_inequality_holds_iff_theta_in_range_l2304_230404

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inequality_holds_iff_theta_in_range :
  ∀ x θ : ℝ,
  x ≥ 3/2 →
  0 < θ →
  θ < π →
  (f (x / Real.sin θ) - (4 * (Real.sin θ)^2 * f x) ≤ f (x - 1) + 4 * f (Real.sin θ))
  ↔
  π/3 ≤ θ ∧ θ ≤ 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_theta_in_range_l2304_230404


namespace NUMINAMATH_CALUDE_saree_ultimate_cost_l2304_230498

/-- Calculates the ultimate cost of a saree after discounts and commission -/
def ultimate_cost (initial_price : ℝ) (discount1 discount2 discount3 commission : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_discount3 := price_after_discount2 * (1 - discount3)
  let final_price := price_after_discount3 * (1 - commission)
  final_price

/-- Theorem stating the ultimate cost of the saree -/
theorem saree_ultimate_cost :
  ultimate_cost 340 0.2 0.15 0.1 0.05 = 197.676 := by
  sorry

end NUMINAMATH_CALUDE_saree_ultimate_cost_l2304_230498


namespace NUMINAMATH_CALUDE_slope_of_line_l2304_230446

/-- The slope of the line (x/4) + (y/5) = 1 is -5/4 -/
theorem slope_of_line (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l2304_230446


namespace NUMINAMATH_CALUDE_fraction_equality_l2304_230444

theorem fraction_equality (x z : ℚ) (hx : x = 4 / 7) (hz : z = 8 / 11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2304_230444


namespace NUMINAMATH_CALUDE_lemonade_sales_l2304_230462

theorem lemonade_sales (katya ricky tina : ℕ) : 
  ricky = 9 →
  tina = 2 * (katya + ricky) →
  tina = katya + 26 →
  katya = 8 := by sorry

end NUMINAMATH_CALUDE_lemonade_sales_l2304_230462


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2304_230424

/-- The area of the triangle formed by y = x, x = -5, and the x-axis --/
def triangle_area : ℝ := 12.5

/-- The x-coordinate of the vertical line --/
def vertical_line_x : ℝ := -5

/-- Theorem: The area of the triangle formed by y = x, x = -5, and the x-axis is 12.5 --/
theorem triangle_area_proof :
  let intersection_point := (vertical_line_x, vertical_line_x)
  let base := -vertical_line_x
  let height := -vertical_line_x
  (1/2 : ℝ) * base * height = triangle_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2304_230424


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l2304_230410

/-- The jumping distances of animals in a contest -/
structure JumpingContest where
  frog_jump : ℕ
  frog_grasshopper_diff : ℕ
  frog_mouse_diff : ℕ

/-- Theorem stating the grasshopper's jump distance given the conditions -/
theorem grasshopper_jump_distance (contest : JumpingContest) 
  (h1 : contest.frog_jump = 58)
  (h2 : contest.frog_grasshopper_diff = 39) :
  contest.frog_jump - contest.frog_grasshopper_diff = 19 := by
  sorry

#check grasshopper_jump_distance

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l2304_230410


namespace NUMINAMATH_CALUDE_melanie_catch_melanie_catch_is_ten_l2304_230414

def sara_catch : ℕ := 5
def melanie_multiplier : ℕ := 2

theorem melanie_catch : ℕ := sara_catch * melanie_multiplier

theorem melanie_catch_is_ten : melanie_catch = 10 := by
  sorry

end NUMINAMATH_CALUDE_melanie_catch_melanie_catch_is_ten_l2304_230414


namespace NUMINAMATH_CALUDE_min_value_problem_l2304_230449

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 8/y = 1) :
  x + y ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2304_230449


namespace NUMINAMATH_CALUDE_set_inclusion_condition_l2304_230445

/-- The necessary and sufficient condition for set inclusion -/
theorem set_inclusion_condition (a : ℝ) (h : a > 0) :
  ({p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 4)^2 ≤ 1} ⊆ 
   {p : ℝ × ℝ | |p.1 - 3| + 2 * |p.2 + 4| ≤ a}) ↔ 
  a ≥ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_condition_l2304_230445


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2304_230482

theorem fraction_evaluation : (3^4 - 3^2) / (3^(-2) + 3^(-4)) = 583.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2304_230482


namespace NUMINAMATH_CALUDE_orange_apple_cost_l2304_230469

/-- The cost of oranges and apples given specific quantities and price per kilo -/
theorem orange_apple_cost (orange_price apple_price : ℕ) 
  (orange_quantity apple_quantity : ℕ) : 
  orange_price = 29 → 
  apple_price = 29 → 
  orange_quantity = 6 → 
  apple_quantity = 5 → 
  orange_price * orange_quantity + apple_price * apple_quantity = 319 :=
by
  sorry

#check orange_apple_cost

end NUMINAMATH_CALUDE_orange_apple_cost_l2304_230469


namespace NUMINAMATH_CALUDE_apple_mango_equivalence_l2304_230417

theorem apple_mango_equivalence (apple_value mango_value : ℝ) :
  (5 / 4 * 16 * apple_value = 10 * mango_value) →
  (3 / 4 * 12 * apple_value = 4.5 * mango_value) := by
  sorry

end NUMINAMATH_CALUDE_apple_mango_equivalence_l2304_230417


namespace NUMINAMATH_CALUDE_union_complement_equals_reals_l2304_230413

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 > 2*x + 3}

noncomputable def B : Set ℝ := {x | Real.log x / Real.log 3 > 1}

theorem union_complement_equals_reals : A ∪ (U \ B) = U := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_reals_l2304_230413


namespace NUMINAMATH_CALUDE_carpet_area_and_cost_exceed_limits_l2304_230451

/-- Represents the dimensions of various room types in Jesse's house -/
structure RoomDimensions where
  rectangular_length : ℝ
  rectangular_width : ℝ
  square_side : ℝ
  triangular_base : ℝ
  triangular_height : ℝ
  trapezoidal_base1 : ℝ
  trapezoidal_base2 : ℝ
  trapezoidal_height : ℝ
  circular_radius : ℝ
  elliptical_major_axis : ℝ
  elliptical_minor_axis : ℝ

/-- Represents the number of each room type in Jesse's house -/
structure RoomCounts where
  rectangular : ℕ
  square : ℕ
  triangular : ℕ
  trapezoidal : ℕ
  circular : ℕ
  elliptical : ℕ

/-- Calculates the total carpet area needed and proves it exceeds 2000 square feet -/
def total_carpet_area_exceeds_2000 (dims : RoomDimensions) (counts : RoomCounts) : Prop :=
  let total_area := 
    counts.rectangular * (dims.rectangular_length * dims.rectangular_width) +
    counts.square * (dims.square_side * dims.square_side) +
    counts.triangular * (dims.triangular_base * dims.triangular_height / 2) +
    counts.trapezoidal * ((dims.trapezoidal_base1 + dims.trapezoidal_base2) / 2 * dims.trapezoidal_height) +
    counts.circular * (Real.pi * dims.circular_radius * dims.circular_radius) +
    counts.elliptical * (Real.pi * (dims.elliptical_major_axis / 2) * (dims.elliptical_minor_axis / 2))
  total_area > 2000

/-- Proves that the total cost exceeds $10,000 when carpet costs $5 per square foot -/
def total_cost_exceeds_budget (dims : RoomDimensions) (counts : RoomCounts) : Prop :=
  let total_area := 
    counts.rectangular * (dims.rectangular_length * dims.rectangular_width) +
    counts.square * (dims.square_side * dims.square_side) +
    counts.triangular * (dims.triangular_base * dims.triangular_height / 2) +
    counts.trapezoidal * ((dims.trapezoidal_base1 + dims.trapezoidal_base2) / 2 * dims.trapezoidal_height) +
    counts.circular * (Real.pi * dims.circular_radius * dims.circular_radius) +
    counts.elliptical * (Real.pi * (dims.elliptical_major_axis / 2) * (dims.elliptical_minor_axis / 2))
  total_area * 5 > 10000

/-- Main theorem combining both conditions -/
theorem carpet_area_and_cost_exceed_limits (dims : RoomDimensions) (counts : RoomCounts) :
  total_carpet_area_exceeds_2000 dims counts ∧ total_cost_exceeds_budget dims counts :=
sorry

end NUMINAMATH_CALUDE_carpet_area_and_cost_exceed_limits_l2304_230451


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2304_230450

/-- The age of the youngest sibling in a family of 6 siblings -/
def youngest_age : ℝ := 17.5

/-- The number of siblings in the family -/
def num_siblings : ℕ := 6

/-- The age differences between the siblings and the youngest sibling -/
def age_differences : List ℝ := [4, 5, 7, 9, 11]

/-- The average age of all siblings -/
def average_age : ℝ := 23.5

/-- Theorem stating that given the conditions, the age of the youngest sibling is 17.5 -/
theorem youngest_sibling_age :
  let ages := youngest_age :: (age_differences.map (· + youngest_age))
  (ages.sum / num_siblings) = average_age ∧
  ages.length = num_siblings :=
by sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2304_230450


namespace NUMINAMATH_CALUDE_inverse_difference_l2304_230453

theorem inverse_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x * y + 1) :
  1 / x - 1 / y = -1 - 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_l2304_230453


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2304_230419

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 56) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3808 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2304_230419


namespace NUMINAMATH_CALUDE_flowers_purchase_l2304_230447

theorem flowers_purchase (dozen_bought : ℕ) : 
  (∀ d : ℕ, 12 * d + 2 * d = 14 * d) →
  12 * dozen_bought + 2 * dozen_bought = 42 →
  dozen_bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_flowers_purchase_l2304_230447


namespace NUMINAMATH_CALUDE_max_value_expression_l2304_230481

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0) 
  (sum_condition : x + y + z = 2) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 256/243 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2304_230481


namespace NUMINAMATH_CALUDE_more_unrepresentable_ten_digit_numbers_l2304_230480

theorem more_unrepresentable_ten_digit_numbers :
  let total_ten_digit_numbers := 9 * (10 ^ 9)
  let five_digit_numbers := 9 * (10 ^ 4)
  let max_representable := five_digit_numbers * (five_digit_numbers + 1)
  max_representable < total_ten_digit_numbers / 2 := by
sorry

end NUMINAMATH_CALUDE_more_unrepresentable_ten_digit_numbers_l2304_230480


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2304_230483

theorem not_sufficient_not_necessary (a b : ℝ) : 
  ¬(∀ a b : ℝ, (a < 0 ∧ b < 0) → a * b * (a - b) > 0) ∧ 
  ¬(∀ a b : ℝ, a * b * (a - b) > 0 → (a < 0 ∧ b < 0)) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2304_230483


namespace NUMINAMATH_CALUDE_farm_has_eleven_goats_l2304_230470

/-- Represents the number of animals on a farm -/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Defines the conditions of the farm -/
def valid_farm (f : Farm) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows = f.goats + 4 ∧
  f.goats + f.cows + f.pigs = 56

/-- Theorem stating that a valid farm has 11 goats -/
theorem farm_has_eleven_goats (f : Farm) (h : valid_farm f) : f.goats = 11 := by
  sorry

#check farm_has_eleven_goats

end NUMINAMATH_CALUDE_farm_has_eleven_goats_l2304_230470


namespace NUMINAMATH_CALUDE_min_total_diff_three_students_l2304_230491

/-- Represents a student's ability characteristics as a list of 12 binary values -/
def Student := List Bool

/-- Calculates the number of different ability characteristics between two students -/
def diffCount (a b : Student) : Nat :=
  List.sum (List.map (fun (x, y) => if x = y then 0 else 1) (List.zip a b))

/-- Checks if two students have a significant comprehensive ability difference -/
def significantDiff (a b : Student) : Prop :=
  diffCount a b ≥ 7

/-- Calculates the total number of different ability characteristics among three students -/
def totalDiff (a b c : Student) : Nat :=
  diffCount a b + diffCount b c + diffCount c a

/-- Theorem: The minimum total number of different ability characteristics among three students
    with significant differences between each pair is 22 -/
theorem min_total_diff_three_students (a b c : Student) :
  (List.length a = 12 ∧ List.length b = 12 ∧ List.length c = 12) →
  (significantDiff a b ∧ significantDiff b c ∧ significantDiff c a) →
  totalDiff a b c ≥ 22 ∧ ∃ (x y z : Student), totalDiff x y z = 22 :=
sorry

end NUMINAMATH_CALUDE_min_total_diff_three_students_l2304_230491


namespace NUMINAMATH_CALUDE_claudia_weekend_earnings_l2304_230401

/-- Calculates the total earnings from weekend art classes -/
def weekend_earnings (cost_per_class : ℚ) (saturday_attendees : ℕ) : ℚ :=
  let sunday_attendees := saturday_attendees / 2
  let total_attendees := saturday_attendees + sunday_attendees
  cost_per_class * total_attendees

/-- Proves that Claudia's total earnings from her weekend art classes are $300.00 -/
theorem claudia_weekend_earnings :
  weekend_earnings 10 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_claudia_weekend_earnings_l2304_230401


namespace NUMINAMATH_CALUDE_tan_graph_problem_l2304_230406

theorem tan_graph_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = 3 → x = π / 4) →
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 3 * π / 4))) →
  a * b = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_problem_l2304_230406


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l2304_230471

/-- Given two rectangles with equal areas, where one rectangle measures 8 inches by 15 inches
    and the other is 4 inches long, prove that the width of the second rectangle is 30 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 8)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 4)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l2304_230471


namespace NUMINAMATH_CALUDE_function_value_sum_l2304_230429

def nondecreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_value_sum (f : ℝ → ℝ) :
  nondecreasing f 0 1 →
  f 0 = 0 →
  (∀ x, f (x / 3) = (1 / 2) * f x) →
  (∀ x, f (1 - x) = 1 - f x) →
  f 1 + f (1 / 2) + f (1 / 3) + f (1 / 6) + f (1 / 7) + f (1 / 8) = 11 / 4 := by
sorry

end NUMINAMATH_CALUDE_function_value_sum_l2304_230429


namespace NUMINAMATH_CALUDE_max_profit_at_16_l2304_230494

/-- Represents the annual profit function for a factory -/
def annual_profit (x : ℕ+) : ℚ :=
  if x ≤ 20 then -x^2 + 32*x - 100 else 160 - x

/-- Theorem stating that the maximum annual profit occurs at 16 units -/
theorem max_profit_at_16 :
  ∀ x : ℕ+, annual_profit 16 ≥ annual_profit x :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_16_l2304_230494


namespace NUMINAMATH_CALUDE_pumpkin_difference_l2304_230439

theorem pumpkin_difference (moonglow_pumpkins sunshine_pumpkins : ℕ) 
  (h1 : moonglow_pumpkins = 14)
  (h2 : sunshine_pumpkins = 54) :
  sunshine_pumpkins - 3 * moonglow_pumpkins = 12 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_difference_l2304_230439


namespace NUMINAMATH_CALUDE_ishaan_age_l2304_230478

/-- Proves that Ishaan is 6 years old given the conditions of the problem -/
theorem ishaan_age (daniel_age : ℕ) (future_years : ℕ) (future_ratio : ℕ) : 
  daniel_age = 69 → 
  future_years = 15 → 
  future_ratio = 4 → 
  ∃ (ishaan_age : ℕ), 
    daniel_age + future_years = future_ratio * (ishaan_age + future_years) ∧ 
    ishaan_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_ishaan_age_l2304_230478


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2_l2304_230452

theorem unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2 :
  ∃! n : ℕ+, 24 ∣ n ∧ 8 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 8.2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2_l2304_230452


namespace NUMINAMATH_CALUDE_scale_length_l2304_230461

/-- A scale is divided into equal parts -/
structure Scale :=
  (parts : ℕ)
  (part_length : ℕ)
  (total_length : ℕ)

/-- Convert inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

/-- Theorem: A scale with 4 parts, each 24 inches long, is 8 feet long -/
theorem scale_length (s : Scale) (h1 : s.parts = 4) (h2 : s.part_length = 24) :
  inches_to_feet s.total_length = 8 := by
  sorry

#check scale_length

end NUMINAMATH_CALUDE_scale_length_l2304_230461


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2304_230485

theorem sum_of_squares_of_roots (r s t : ℝ) : 
  (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
  (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
  (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
  (r ≠ s) → (r ≠ t) → (s ≠ t) →
  r^2 + s^2 + t^2 = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2304_230485


namespace NUMINAMATH_CALUDE_suit_tie_discount_cost_l2304_230492

/-- Represents the cost calculation for two discount options in a suit and tie sale. -/
theorem suit_tie_discount_cost 
  (suit_price : ℕ) 
  (tie_price : ℕ) 
  (num_suits : ℕ) 
  (num_ties : ℕ) 
  (h1 : suit_price = 500)
  (h2 : tie_price = 100)
  (h3 : num_suits = 20)
  (h4 : num_ties > 20) :
  (num_suits * suit_price + (num_ties - num_suits) * tie_price = 100 * num_ties + 8000) ∧ 
  (((num_suits * suit_price + num_ties * tie_price) * 90) / 100 = 90 * num_ties + 9000) := by
  sorry

end NUMINAMATH_CALUDE_suit_tie_discount_cost_l2304_230492


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l2304_230415

/-- The number of ways to select a president and a vice president from a group of 4 people -/
def select_president_and_vice_president (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select a president and a vice president from a group of 4 people is 12 -/
theorem president_vice_president_selection :
  select_president_and_vice_president 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l2304_230415


namespace NUMINAMATH_CALUDE_least_whole_number_subtraction_l2304_230488

theorem least_whole_number_subtraction (x : ℕ) : 
  x ≥ 3 ∧ 
  ∀ y : ℕ, y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21 ∧
  (6 - x : ℚ) / (7 - x) < 16 / 21 :=
sorry

end NUMINAMATH_CALUDE_least_whole_number_subtraction_l2304_230488


namespace NUMINAMATH_CALUDE_paula_karl_age_problem_l2304_230437

/-- Represents the ages and time in the problem about Paula and Karl --/
structure AgesProblem where
  paula_age : ℕ
  karl_age : ℕ
  years_until_double : ℕ

/-- The conditions of the problem are satisfied --/
def satisfies_conditions (ap : AgesProblem) : Prop :=
  (ap.paula_age - 5 = 3 * (ap.karl_age - 5)) ∧
  (ap.paula_age + ap.karl_age = 54) ∧
  (ap.paula_age + ap.years_until_double = 2 * (ap.karl_age + ap.years_until_double))

/-- The theorem stating that the solution to the problem is 6 years --/
theorem paula_karl_age_problem :
  ∃ (ap : AgesProblem), satisfies_conditions ap ∧ ap.years_until_double = 6 :=
by sorry

end NUMINAMATH_CALUDE_paula_karl_age_problem_l2304_230437


namespace NUMINAMATH_CALUDE_action_figure_cost_l2304_230472

theorem action_figure_cost (current : ℕ) (total : ℕ) (cost : ℕ) : current = 7 → total = 16 → cost = 72 → (cost / (total - current) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_cost_l2304_230472


namespace NUMINAMATH_CALUDE_sum_of_unknown_numbers_l2304_230408

def known_numbers : List ℕ := [690, 744, 745, 747, 748, 749, 752, 752, 753, 755, 760, 769]

theorem sum_of_unknown_numbers 
  (total_count : ℕ) 
  (average : ℕ) 
  (h1 : total_count = 15) 
  (h2 : average = 750) 
  (h3 : known_numbers.length = 12) : 
  (total_count * average) - known_numbers.sum = 2336 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_unknown_numbers_l2304_230408


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l2304_230490

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 78 →
  last_six_avg = 75 →
  sixth_number = 258 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 :=
by sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l2304_230490


namespace NUMINAMATH_CALUDE_boys_who_watched_l2304_230441

/-- The number of boys who went down the slide initially -/
def x : ℕ := 22

/-- The number of additional boys who went down the slide later -/
def y : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_slide : ℕ := x + y

/-- The ratio of boys who went down the slide to boys who watched -/
def ratio_slide_to_watch : Rat := 5 / 3

/-- The number of boys who watched but didn't go down the slide -/
def z : ℕ := (3 * total_slide) / 5

theorem boys_who_watched (h : ratio_slide_to_watch = 5 / 3) : z = 21 := by
  sorry

end NUMINAMATH_CALUDE_boys_who_watched_l2304_230441


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2304_230443

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.im ((1 + i) / (1 - i)) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2304_230443


namespace NUMINAMATH_CALUDE_largest_number_l2304_230467

theorem largest_number : 
  (1 : ℝ) ≥ Real.sqrt 29 - Real.sqrt 21 ∧ 
  (1 : ℝ) ≥ Real.pi / 3.142 ∧ 
  (1 : ℝ) ≥ 5.1 * Real.sqrt 0.0361 ∧ 
  (1 : ℝ) ≥ 6 / (Real.sqrt 13 + Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2304_230467


namespace NUMINAMATH_CALUDE_inequality_problem_l2304_230431

theorem inequality_problem :
  (∀ x : ℝ, |x + 7| + |x - 1| ≥ 8) ∧
  (¬ ∃ m : ℝ, m > 8 ∧ ∀ x : ℝ, |x + 7| + |x - 1| ≥ m) ∧
  (∀ x : ℝ, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2304_230431


namespace NUMINAMATH_CALUDE_roller_alignment_l2304_230430

/-- The number of rotations needed for alignment of two rollers -/
def alignmentRotations (r1 r2 : ℕ) : ℕ :=
  (Nat.lcm r1 r2) / r1

/-- Theorem: The number of rotations for alignment of rollers with radii 105 and 90 is 6 -/
theorem roller_alignment :
  alignmentRotations 105 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_alignment_l2304_230430


namespace NUMINAMATH_CALUDE_geometric_progression_x_value_l2304_230426

theorem geometric_progression_x_value : 
  ∀ (x : ℝ), 
  let a₁ := 2*x - 2
  let a₂ := 2*x + 2
  let a₃ := 4*x + 6
  (a₂ / a₁ = a₃ / a₂) → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_x_value_l2304_230426


namespace NUMINAMATH_CALUDE_negation_of_p_l2304_230493

-- Define the set M
def M : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the original proposition p
def p : Prop := ∃ x ∈ M, x^2 - x - 2 < 0

-- Statement: The negation of p is equivalent to ∀x ∈ M, x^2 - x - 2 ≥ 0
theorem negation_of_p : ¬p ↔ ∀ x ∈ M, x^2 - x - 2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l2304_230493


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2304_230496

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 2 * x + 15 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y - 2 * y + 15 = 0 → y = x) ↔ 
  (m = 6 * Real.sqrt 5 - 2 ∨ m = -6 * Real.sqrt 5 - 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2304_230496


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2304_230454

/-- 
In a Cartesian coordinate system, the coordinates of a point (2, -3) 
with respect to the origin are (2, -3).
-/
theorem point_coordinates_wrt_origin : 
  let point : ℝ × ℝ := (2, -3)
  point = (2, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2304_230454
