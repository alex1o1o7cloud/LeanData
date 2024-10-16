import Mathlib

namespace NUMINAMATH_CALUDE_system_three_solutions_l3754_375413

/-- The system of equations has exactly three solutions if and only if a = 49 or a = 40 - 4√51 -/
theorem system_three_solutions (a : ℝ) : 
  (∃! x y z : ℝ × ℝ, 
    ((abs (y.2 - 10) + abs (x.1 + 3) - 2) * (x.1^2 + x.2^2 - 6) = 0 ∧
     (x.1 + 3)^2 + (x.2 - 5)^2 = a) ∧
    ((abs (y.2 - 10) + abs (y.1 + 3) - 2) * (y.1^2 + y.2^2 - 6) = 0 ∧
     (y.1 + 3)^2 + (y.2 - 5)^2 = a) ∧
    ((abs (z.2 - 10) + abs (z.1 + 3) - 2) * (z.1^2 + z.2^2 - 6) = 0 ∧
     (z.1 + 3)^2 + (z.2 - 5)^2 = a) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ 
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end NUMINAMATH_CALUDE_system_three_solutions_l3754_375413


namespace NUMINAMATH_CALUDE_investment_value_proof_l3754_375443

theorem investment_value_proof (x : ℝ) : 
  (0.07 * x + 0.11 * 1500 = 0.10 * (x + 1500)) → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_proof_l3754_375443


namespace NUMINAMATH_CALUDE_problem_solution_l3754_375464

theorem problem_solution :
  (∀ x : ℝ, |x + 2| + |6 - x| ≥ 8) ∧
  (∃ x : ℝ, |x + 2| + |6 - x| = 8) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8 → 7 * a + 4 * b ≥ 9 / 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 8 / (5 * a + b) + 2 / (2 * a + 3 * b) = 8 ∧ 7 * a + 4 * b = 9 / 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3754_375464


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l3754_375420

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: The value of x in the O'Hara triple (49, 16, x) is 11 -/
theorem ohara_triple_49_16 :
  ∃ x : ℕ, is_ohara_triple 49 16 x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l3754_375420


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l3754_375456

/-- The angle of inclination of the line x - y - √3 = 0 is 45° -/
theorem line_inclination_45_degrees :
  let line := {(x, y) : ℝ × ℝ | x - y - Real.sqrt 3 = 0}
  ∃ θ : ℝ, θ = 45 * π / 180 ∧ ∀ (x y : ℝ), (x, y) ∈ line → y = Real.tan θ * x + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l3754_375456


namespace NUMINAMATH_CALUDE_x_range_l3754_375481

theorem x_range (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 
  -1 ≤ x ∧ x < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3754_375481


namespace NUMINAMATH_CALUDE_investment_problem_l3754_375442

theorem investment_problem (x y : ℝ) (h1 : x + y = 3000) 
  (h2 : 0.08 * x + 0.05 * y = 490 ∨ 0.08 * y + 0.05 * x = 490) : 
  x + y = 8000 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l3754_375442


namespace NUMINAMATH_CALUDE_curve_symmetry_l3754_375472

/-- A curve is symmetrical about the y-axis if substituting -x for x in its equation leaves it unchanged -/
def symmetrical_about_y_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f (-x) y

/-- The equation x^2 - y^2 = 1 -/
def curve_equation (x y : ℝ) : Prop := x^2 - y^2 = 1

theorem curve_symmetry : symmetrical_about_y_axis curve_equation :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l3754_375472


namespace NUMINAMATH_CALUDE_sum_of_divisors_882_prime_factors_l3754_375435

def sum_of_divisors (n : ℕ) : ℕ := sorry

def count_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_882_prime_factors :
  count_distinct_prime_factors (sum_of_divisors 882) = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_882_prime_factors_l3754_375435


namespace NUMINAMATH_CALUDE_benedicts_house_size_l3754_375485

theorem benedicts_house_size (kennedy_house : ℕ) (benedict_house : ℕ) : 
  kennedy_house = 10000 ∧ kennedy_house = 4 * benedict_house + 600 → benedict_house = 2350 := by
  sorry

end NUMINAMATH_CALUDE_benedicts_house_size_l3754_375485


namespace NUMINAMATH_CALUDE_female_student_fraction_l3754_375498

theorem female_student_fraction :
  ∀ (f m : ℝ),
  f + m = 1 →
  (5/6 : ℝ) * f + (2/3 : ℝ) * m = 0.7333333333333333 →
  f = 0.4 := by
sorry

end NUMINAMATH_CALUDE_female_student_fraction_l3754_375498


namespace NUMINAMATH_CALUDE_divisibility_condition_l3754_375426

theorem divisibility_condition (x y : ℕ+) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔
  (∃ a : ℕ+, x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3754_375426


namespace NUMINAMATH_CALUDE_extreme_values_and_range_l3754_375466

/-- The function f(x) with parameters a, b, and c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_range (a b c : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (∀ x ∈ Set.Icc 0 3, f a b c x < c^2) →
  (a = -3 ∧ b = 4 ∧ c < -1 ∨ c > 9) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_range_l3754_375466


namespace NUMINAMATH_CALUDE_expected_worth_of_coin_flip_l3754_375457

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Probability of getting heads on a single flip -/
def probHeads : ℚ := 2/3

/-- Probability of getting tails on a single flip -/
def probTails : ℚ := 1/3

/-- Reward for getting heads -/
def rewardHeads : ℚ := 5

/-- Penalty for getting tails -/
def penaltyTails : ℚ := -9

/-- Additional penalty for three consecutive tails -/
def penaltyThreeTails : ℚ := -10

/-- Expected value of a single coin flip -/
def expectedValueSingleFlip : ℚ := probHeads * rewardHeads + probTails * penaltyTails

/-- Probability of getting three consecutive tails -/
def probThreeTails : ℚ := probTails^3

/-- Additional expected loss from three consecutive tails -/
def expectedAdditionalLoss : ℚ := probThreeTails * penaltyThreeTails

/-- Total expected value of a coin flip -/
def totalExpectedValue : ℚ := expectedValueSingleFlip + expectedAdditionalLoss

theorem expected_worth_of_coin_flip :
  totalExpectedValue = -1/27 := by sorry

end NUMINAMATH_CALUDE_expected_worth_of_coin_flip_l3754_375457


namespace NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l3754_375438

theorem probability_of_two_in_three_elevenths : 
  let decimal_rep := (3 : ℚ) / 11
  let period := 2
  let count_of_two := 1
  (count_of_two : ℚ) / period = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l3754_375438


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3754_375400

/-- 
Given two runners a and b, where:
- a's speed is some multiple of b's speed
- a gives b a head start of 1/16 of the race length
- They finish at the same time (dead heat)
Then the ratio of a's speed to b's speed is 15/16
-/
theorem race_speed_ratio (v_a v_b : ℝ) (h : v_a > 0 ∧ v_b > 0) :
  (∃ k : ℝ, v_a = k * v_b) →
  (v_a * 1 = v_b * (15/16)) →
  v_a / v_b = 15/16 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l3754_375400


namespace NUMINAMATH_CALUDE_num_cars_in_parking_lot_l3754_375468

def num_bikes : ℕ := 10
def total_wheels : ℕ := 76
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem num_cars_in_parking_lot : 
  (total_wheels - num_bikes * wheels_per_bike) / wheels_per_car = 14 := by
  sorry

end NUMINAMATH_CALUDE_num_cars_in_parking_lot_l3754_375468


namespace NUMINAMATH_CALUDE_tangent_line_range_l3754_375463

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line is equal to the radius of the circle. -/
def is_tangent_line (m n : ℝ) : Prop :=
  1 = |(m + 1) + (n + 1) - 2| / Real.sqrt ((m + 1)^2 + (n + 1)^2)

/-- The range of m + n when the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1 -/
theorem tangent_line_range (m n : ℝ) :
  is_tangent_line m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_range_l3754_375463


namespace NUMINAMATH_CALUDE_probability_is_one_fifth_l3754_375451

/-- The probability of finding the last defective product on the fourth inspection -/
def probability_last_defective_fourth_inspection (total : ℕ) (qualified : ℕ) (defective : ℕ) : ℚ :=
  let p1 := qualified / total * (qualified - 1) / (total - 1) * defective / (total - 2) * 1 / (total - 3)
  let p2 := qualified / total * defective / (total - 1) * (qualified - 1) / (total - 2) * 1 / (total - 3)
  let p3 := defective / total * qualified / (total - 1) * (qualified - 1) / (total - 2) * 1 / (total - 3)
  p1 + p2 + p3

/-- Theorem stating that the probability is 1/5 for the given conditions -/
theorem probability_is_one_fifth :
  probability_last_defective_fourth_inspection 6 4 2 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_fifth_l3754_375451


namespace NUMINAMATH_CALUDE_aluminum_carbonate_weight_l3754_375477

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of moles of Aluminum carbonate -/
def moles : ℝ := 5

/-- The molecular weight of Aluminum carbonate (Al2(CO3)3) in g/mol -/
def Al2CO3_3_weight : ℝ := 2 * Al_weight + 3 * C_weight + 9 * O_weight

/-- The total weight of the given moles of Aluminum carbonate in grams -/
def total_weight : ℝ := moles * Al2CO3_3_weight

theorem aluminum_carbonate_weight : total_weight = 1169.95 := by sorry

end NUMINAMATH_CALUDE_aluminum_carbonate_weight_l3754_375477


namespace NUMINAMATH_CALUDE_find_genuine_stacks_l3754_375462

/-- Represents a stack of coins -/
structure CoinStack :=
  (count : Nat)
  (hasOddCoin : Bool)

/-- Represents the result of weighing two stacks -/
inductive WeighResult
  | Equal
  | Unequal

/-- Represents the state of the coin stacks -/
structure CoinStacks :=
  (stack1 : CoinStack)
  (stack2 : CoinStack)
  (stack3 : CoinStack)
  (stack4 : CoinStack)

/-- Represents a weighing action -/
def weigh (s1 s2 : CoinStack) : WeighResult :=
  if s1.hasOddCoin = s2.hasOddCoin then WeighResult.Equal else WeighResult.Unequal

/-- The main theorem -/
theorem find_genuine_stacks 
  (stacks : CoinStacks)
  (h1 : stacks.stack1.count = 5)
  (h2 : stacks.stack2.count = 6)
  (h3 : stacks.stack3.count = 7)
  (h4 : stacks.stack4.count = 19)
  (h5 : (stacks.stack1.hasOddCoin || stacks.stack2.hasOddCoin || stacks.stack3.hasOddCoin || stacks.stack4.hasOddCoin) ∧ 
        (¬stacks.stack1.hasOddCoin ∨ ¬stacks.stack2.hasOddCoin ∨ ¬stacks.stack3.hasOddCoin ∨ ¬stacks.stack4.hasOddCoin)) :
  ∃ (s1 s2 : CoinStack), s1 ∈ [stacks.stack1, stacks.stack2, stacks.stack3, stacks.stack4] ∧ 
                         s2 ∈ [stacks.stack1, stacks.stack2, stacks.stack3, stacks.stack4] ∧ 
                         s1 ≠ s2 ∧ 
                         ¬s1.hasOddCoin ∧ 
                         ¬s2.hasOddCoin := by
  sorry


end NUMINAMATH_CALUDE_find_genuine_stacks_l3754_375462


namespace NUMINAMATH_CALUDE_function_property_l3754_375424

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 19) ≤ f x + 19) 
  (h2 : ∀ x : ℝ, f (x + 94) ≥ f x + 94) : 
  ∀ x : ℝ, f (x + 1) = f x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3754_375424


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3754_375448

theorem inequalities_theorem :
  (∀ (a b c d : ℝ), a > b → c > d → a - d > b - c) ∧
  (∀ (a b : ℝ), 1/a < 1/b → 1/b < 0 → a*b < b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3754_375448


namespace NUMINAMATH_CALUDE_compound_proposition_negation_l3754_375444

theorem compound_proposition_negation (p q : Prop) : 
  ¬((p ∧ q → false) → (¬p → false) ∧ (¬q → false)) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_negation_l3754_375444


namespace NUMINAMATH_CALUDE_equation_solution_l3754_375459

def solution_set : Set ℝ := {-Real.sqrt 10, -Real.pi, -1, 1, Real.pi, Real.sqrt 10}

def domain (x : ℝ) : Prop :=
  (-Real.sqrt 10 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ Real.sqrt 10)

theorem equation_solution :
  ∀ x : ℝ, domain x →
    ((Real.sin (2 * x) - Real.pi * Real.sin x) * Real.sqrt (11 * x^2 - x^4 - 10) = 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3754_375459


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3754_375427

theorem perfect_square_trinomial : 120^2 - 40 * 120 + 20^2 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3754_375427


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l3754_375436

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l3754_375436


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3754_375496

/-- Given a square with perimeter 32 kilometers, its area is 64 square kilometers. -/
theorem square_area_from_perimeter :
  ∀ (s : ℝ), s > 0 → 4 * s = 32 → s * s = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3754_375496


namespace NUMINAMATH_CALUDE_shape_sum_theorem_l3754_375407

-- Define the shapes as real numbers
variable (triangle : ℝ) (circle : ℝ) (square : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := 2 * triangle + 2 * circle + square = 27
def condition2 : Prop := 2 * circle + triangle + square = 26
def condition3 : Prop := 2 * square + triangle + circle = 23

-- Define the theorem
theorem shape_sum_theorem 
  (h1 : condition1 triangle circle square)
  (h2 : condition2 triangle circle square)
  (h3 : condition3 triangle circle square) :
  2 * triangle + 3 * circle + square = 45.5 := by
  sorry

end NUMINAMATH_CALUDE_shape_sum_theorem_l3754_375407


namespace NUMINAMATH_CALUDE_round_robin_perfect_matching_l3754_375471

/-- Represents a round-robin tournament -/
structure RoundRobinTournament (n : ℕ) :=
  (teams : Finset (Fin (2*n)))
  (days : Finset (Fin (2*n - 1)))
  (winners : Fin (2*n - 1) → Finset (Fin (2*n)))
  (team_count : teams.card = 2*n)
  (day_count : days.card = 2*n - 1)
  (daily_winners : ∀ d, (winners d).card = n)
  (unique_games : ∀ t₁ t₂, t₁ ≠ t₂ → ∃! d, (t₁ ∈ winners d ∧ t₂ ∉ winners d) ∨ (t₂ ∈ winners d ∧ t₁ ∉ winners d))

/-- Perfect matching between days and winning teams -/
def perfect_matching {n : ℕ} (t : RoundRobinTournament n) :=
  ∃ f : Fin (2*n - 1) → Fin (2*n), Function.Injective f ∧ ∀ d, f d ∈ t.winners d

/-- Theorem stating the existence of a perfect matching in a round-robin tournament -/
theorem round_robin_perfect_matching {n : ℕ} (t : RoundRobinTournament n) :
  perfect_matching t :=
sorry

end NUMINAMATH_CALUDE_round_robin_perfect_matching_l3754_375471


namespace NUMINAMATH_CALUDE_xyz_value_l3754_375402

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 10)
  (eq5 : x + y + z = 6) :
  x * y * z = 14 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3754_375402


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_plus_one_l3754_375474

theorem cube_plus_reciprocal_cube_plus_one (m : ℝ) (h : m + 1/m = 10) : 
  m^3 + 1/m^3 + 1 = 971 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_plus_one_l3754_375474


namespace NUMINAMATH_CALUDE_painted_cube_probability_l3754_375475

theorem painted_cube_probability : 
  let cube_side : ℕ := 5
  let total_cubes : ℕ := cube_side ^ 3
  let painted_faces : ℕ := 2
  let two_face_painted : ℕ := 4 * (cube_side - 1)
  let no_face_painted : ℕ := total_cubes - 2 * cube_side^2 + two_face_painted
  let total_combinations : ℕ := total_cubes.choose 2
  let favorable_outcomes : ℕ := two_face_painted * no_face_painted
  (favorable_outcomes : ℚ) / total_combinations = 728 / 3875 := by
sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l3754_375475


namespace NUMINAMATH_CALUDE_oil_bill_ratio_change_l3754_375484

theorem oil_bill_ratio_change (january_bill : ℚ) (february_bill : ℚ) : 
  january_bill = 120 →
  february_bill / january_bill = 5 / 4 →
  (february_bill + 30) / january_bill = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_change_l3754_375484


namespace NUMINAMATH_CALUDE_nineteen_percent_female_officers_on_duty_l3754_375467

/-- Calculates the percentage of female officers on duty given the total officers on duty,
    the fraction of officers on duty who are female, and the total number of female officers. -/
def percentage_female_officers_on_duty (total_on_duty : ℕ) (fraction_female : ℚ) (total_female : ℕ) : ℚ :=
  (fraction_female * total_on_duty : ℚ) / total_female * 100

/-- Theorem stating that 19% of female officers were on duty that night. -/
theorem nineteen_percent_female_officers_on_duty :
  percentage_female_officers_on_duty 152 (1/2) 400 = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_percent_female_officers_on_duty_l3754_375467


namespace NUMINAMATH_CALUDE_greatest_common_length_l3754_375418

theorem greatest_common_length (rope1 rope2 rope3 rope4 : ℕ) 
  (h1 : rope1 = 48) (h2 : rope2 = 64) (h3 : rope3 = 80) (h4 : rope4 = 120) :
  Nat.gcd rope1 (Nat.gcd rope2 (Nat.gcd rope3 rope4)) = 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_length_l3754_375418


namespace NUMINAMATH_CALUDE_transmission_time_l3754_375408

/-- Proves that given the specified conditions, the transmission time is 5 minutes -/
theorem transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 80 →
  chunks_per_block = 640 →
  transmission_rate = 160 →
  (blocks * chunks_per_block : ℝ) / transmission_rate / 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_l3754_375408


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3754_375415

theorem line_tangent_to_circle (t θ : ℝ) (α : ℝ) : 
  (∃ t, ∀ θ, (t * Real.cos α - (4 + 2 * Real.cos θ))^2 + (t * Real.sin α - 2 * Real.sin θ)^2 = 4) →
  α = π / 6 ∨ α = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3754_375415


namespace NUMINAMATH_CALUDE_senior_policy_more_profitable_l3754_375479

/-- Represents a customer group with their characteristics -/
structure CustomerGroup where
  repaymentReliability : ℝ
  incomeStability : ℝ
  savingInclination : ℝ
  longTermPreference : ℝ

/-- Represents the bank's policy for a customer group -/
structure BankPolicy where
  depositRate : ℝ
  loanRate : ℝ

/-- Calculates the bank's profit from a customer group under a given policy -/
def bankProfit (group : CustomerGroup) (policy : BankPolicy) : ℝ :=
  group.repaymentReliability * policy.loanRate +
  group.savingInclination * group.longTermPreference * policy.depositRate

/-- Theorem: Under certain conditions, a bank can achieve higher profit 
    by offering better rates to seniors -/
theorem senior_policy_more_profitable 
  (seniors : CustomerGroup) 
  (others : CustomerGroup)
  (seniorPolicy : BankPolicy)
  (otherPolicy : BankPolicy)
  (h1 : seniors.repaymentReliability > others.repaymentReliability)
  (h2 : seniors.incomeStability > others.incomeStability)
  (h3 : seniors.savingInclination > others.savingInclination)
  (h4 : seniors.longTermPreference > others.longTermPreference)
  (h5 : seniorPolicy.depositRate > otherPolicy.depositRate)
  (h6 : seniorPolicy.loanRate < otherPolicy.loanRate) :
  bankProfit seniors seniorPolicy > bankProfit others otherPolicy :=
sorry

end NUMINAMATH_CALUDE_senior_policy_more_profitable_l3754_375479


namespace NUMINAMATH_CALUDE_product_remainder_ten_l3754_375421

theorem product_remainder_ten (a b c : ℕ) (ha : a = 2153) (hb : b = 3491) (hc : c = 925) :
  (a * b * c) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_ten_l3754_375421


namespace NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l3754_375449

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  Nat.log 9 n

theorem base_3_to_base_9_first_digit :
  let y : Nat := base_3_to_10 [1, 1, 2, 2, 1, 1]
  first_digit_base_9 y = 4 := by sorry

end NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l3754_375449


namespace NUMINAMATH_CALUDE_max_distance_symmetric_points_constant_sum_distances_to_foci_focal_length_to_minor_axis_ratio_no_perpendicular_lines_to_foci_l3754_375403

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := C p.1 p.2

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop := p.1 = -q.1 ∧ p.2 = -q.2

-- Statement 1: Maximum distance between symmetric points
theorem max_distance_symmetric_points :
  ∀ A B : ℝ × ℝ, on_ellipse A → on_ellipse B → symmetric_wrt_origin A B →
  ‖A - B‖ ≤ 10 :=
sorry

-- Statement 2: Constant sum of distances to foci
theorem constant_sum_distances_to_foci :
  ∀ A : ℝ × ℝ, on_ellipse A →
  ‖A - F₁‖ + ‖A - F₂‖ = 10 :=
sorry

-- Statement 3: Ratio of focal length to minor axis
theorem focal_length_to_minor_axis_ratio :
  ‖F₁ - F₂‖ / 8 = 3/4 :=
sorry

-- Statement 4: No point with perpendicular lines to foci
theorem no_perpendicular_lines_to_foci :
  ¬ ∃ A : ℝ × ℝ, on_ellipse A ∧ 
  (A - F₁) • (A - F₂) = 0 :=
sorry

end NUMINAMATH_CALUDE_max_distance_symmetric_points_constant_sum_distances_to_foci_focal_length_to_minor_axis_ratio_no_perpendicular_lines_to_foci_l3754_375403


namespace NUMINAMATH_CALUDE_probability_of_two_tails_in_three_flips_l3754_375428

def probability_of_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem probability_of_two_tails_in_three_flips :
  probability_of_k_successes 3 2 (1/2) = 0.375 := by
sorry

end NUMINAMATH_CALUDE_probability_of_two_tails_in_three_flips_l3754_375428


namespace NUMINAMATH_CALUDE_candy_distribution_l3754_375492

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) : 
  total_candy = 344 →
  num_students = 43 →
  total_candy = num_students * pieces_per_student →
  pieces_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3754_375492


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l3754_375417

theorem student_multiplication_problem (chosen_number : ℕ) (final_result : ℕ) (subtracted_amount : ℕ) :
  chosen_number = 125 →
  final_result = 112 →
  subtracted_amount = 138 →
  ∃ x : ℚ, chosen_number * x - subtracted_amount = final_result ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l3754_375417


namespace NUMINAMATH_CALUDE_ski_down_time_l3754_375465

-- Define the lift ride time
def lift_time : ℕ := 15

-- Define the number of round trips in 2 hours
def round_trips : ℕ := 6

-- Define the total time for 6 round trips in minutes
def total_time : ℕ := 2 * 60

-- Theorem: The time to ski down the mountain is 20 minutes
theorem ski_down_time : 
  (total_time / round_trips) - lift_time = 20 :=
sorry

end NUMINAMATH_CALUDE_ski_down_time_l3754_375465


namespace NUMINAMATH_CALUDE_problem_solution_l3754_375401

theorem problem_solution : 
  let a := (5 / 6) * 180
  let b := 0.70 * 250
  let diff := a - b
  let c := 0.35 * 480
  diff / c = -0.14880952381 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3754_375401


namespace NUMINAMATH_CALUDE_square_of_104_l3754_375450

theorem square_of_104 : (104 : ℕ)^2 = 10816 := by sorry

end NUMINAMATH_CALUDE_square_of_104_l3754_375450


namespace NUMINAMATH_CALUDE_shortest_dividing_line_l3754_375409

-- Define a circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a broken line
def BrokenLine := List (ℝ × ℝ)

-- Function to calculate the length of a broken line
def length (bl : BrokenLine) : ℝ := sorry

-- Function to check if a broken line divides the circle into two equal parts
def divides_equally (bl : BrokenLine) (c : Circle) : Prop := sorry

-- Define the diameter of a circle
def diameter (c : Circle) : ℝ := 2

-- Theorem statement
theorem shortest_dividing_line (c : Circle) (bl : BrokenLine) :
  divides_equally bl c → length bl ≥ diameter c ∧
  (length bl = diameter c ↔ ∃ a b : ℝ × ℝ, bl = [a, b] ∧ a.1^2 + a.2^2 = 1 ∧ b.1^2 + b.2^2 = 1 ∧ (a.1 + b.1 = 0 ∧ a.2 + b.2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_shortest_dividing_line_l3754_375409


namespace NUMINAMATH_CALUDE_frank_reading_time_l3754_375404

/-- Calculates the number of days needed to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Proves that it takes 569 days to read a book with 12518 pages at 22 pages per day -/
theorem frank_reading_time : days_to_read 12518 22 = 569 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_time_l3754_375404


namespace NUMINAMATH_CALUDE_subtract_negative_real_l3754_375483

theorem subtract_negative_real : 3.7 - (-1.45) = 5.15 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_real_l3754_375483


namespace NUMINAMATH_CALUDE_equation_solution_l3754_375473

theorem equation_solution : ∃ x : ℝ, 20 - 3 * (x + 4) = 2 * (x - 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3754_375473


namespace NUMINAMATH_CALUDE_product_543_7_base9_l3754_375440

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  sorry

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  sorry

/-- Multiplies two base-9 numbers and returns the result in base-9 --/
def multiplyBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem product_543_7_base9 :
  multiplyBase9 543 7 = 42333 :=
sorry

end NUMINAMATH_CALUDE_product_543_7_base9_l3754_375440


namespace NUMINAMATH_CALUDE_smartphone_price_l3754_375494

/-- The original sticker price of the smartphone -/
def sticker_price : ℝ := 950

/-- The price at store X after discount and rebate -/
def price_x (p : ℝ) : ℝ := 0.8 * p - 120

/-- The price at store Y after discount -/
def price_y (p : ℝ) : ℝ := 0.7 * p

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem smartphone_price :
  price_x sticker_price + 25 = price_y sticker_price :=
by sorry

end NUMINAMATH_CALUDE_smartphone_price_l3754_375494


namespace NUMINAMATH_CALUDE_max_product_distances_l3754_375433

/-- Two perpendicular lines passing through points A and B, intersecting at P -/
structure PerpendicularLines where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  perpendicular : (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

/-- The maximum value of |PA| * |PB| for perpendicular lines through A(0, 0) and B(1, 3) -/
theorem max_product_distances (l : PerpendicularLines) 
  (h_A : l.A = (0, 0)) (h_B : l.B = (1, 3)) : 
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ), 
    ((P.1 - l.A.1)^2 + (P.2 - l.A.2)^2) * ((P.1 - l.B.1)^2 + (P.2 - l.B.2)^2) ≤ max^2 ∧ 
    max = 5 :=
sorry

end NUMINAMATH_CALUDE_max_product_distances_l3754_375433


namespace NUMINAMATH_CALUDE_continuous_compound_interest_interest_rate_problem_l3754_375488

/-- The annual interest rate for a continuously compounded investment --/
noncomputable def annual_interest_rate (initial_investment : ℝ) (final_amount : ℝ) (years : ℝ) : ℝ :=
  (Real.log (final_amount / initial_investment)) / years

/-- Theorem stating the relationship between the initial investment, final amount, time, and interest rate --/
theorem continuous_compound_interest
  (initial_investment : ℝ)
  (final_amount : ℝ)
  (years : ℝ)
  (h1 : initial_investment > 0)
  (h2 : final_amount > initial_investment)
  (h3 : years > 0) :
  final_amount = initial_investment * Real.exp (years * annual_interest_rate initial_investment final_amount years) :=
by sorry

/-- The specific problem instance --/
theorem interest_rate_problem :
  let initial_investment : ℝ := 5000
  let final_amount : ℝ := 8500
  let years : ℝ := 10
  8500 = 5000 * Real.exp (10 * annual_interest_rate 5000 8500 10) :=
by sorry

end NUMINAMATH_CALUDE_continuous_compound_interest_interest_rate_problem_l3754_375488


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_hypotenuse_length_l3754_375480

theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a > 0 ∧ b > 0) :
  (∃ (m_a m_b : ℝ), 
    m_a^2 = b^2 + (a/2)^2 ∧
    m_b^2 = a^2 + (b/2)^2 ∧
    m_a = 6 ∧
    m_b = Real.sqrt 34) →
  a^2 + b^2 = 56 :=
by sorry

theorem hypotenuse_length (a b : ℝ) (h_right : a > 0 ∧ b > 0) :
  a^2 + b^2 = 56 →
  Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_hypotenuse_length_l3754_375480


namespace NUMINAMATH_CALUDE_variance_of_scores_l3754_375411

def scores : List ℝ := [87, 90, 90, 91, 91, 94, 94]

theorem variance_of_scores : 
  let n : ℕ := scores.length
  let mean : ℝ := scores.sum / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  variance = 36/7 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_scores_l3754_375411


namespace NUMINAMATH_CALUDE_smallest_self_repeating_square_end_l3754_375432

/-- A function that returns the last n digits of a natural number in base 10 -/
def lastNDigits (n : ℕ) (digits : ℕ) : ℕ :=
  n % (10 ^ digits)

/-- The theorem stating that 40625 is the smallest positive integer N such that
    N and N^2 end in the same sequence of five digits in base 10,
    with the first of these five digits being non-zero -/
theorem smallest_self_repeating_square_end : ∀ N : ℕ,
  N > 0 ∧ 
  lastNDigits N 5 = lastNDigits (N^2) 5 ∧
  N ≥ 10000 →
  N ≥ 40625 := by
  sorry

end NUMINAMATH_CALUDE_smallest_self_repeating_square_end_l3754_375432


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3754_375489

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3754_375489


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3754_375458

theorem sqrt_equation_solution (x : ℝ) : 
  (Real.sqrt (4 - 5*x + x^2) = 9) ↔ (x = (5 + Real.sqrt 333) / 2 ∨ x = (5 - Real.sqrt 333) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3754_375458


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3754_375434

theorem movie_ticket_cost 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (concession_cost : ℚ) 
  (total_cost : ℚ) 
  (child_ticket_cost : ℚ) : 
  num_adults = 5 → 
  num_children = 2 → 
  concession_cost = 12 → 
  total_cost = 76 → 
  child_ticket_cost = 7 → 
  (total_cost - concession_cost - num_children * child_ticket_cost) / num_adults = 10 :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l3754_375434


namespace NUMINAMATH_CALUDE_point_distance_product_l3754_375470

theorem point_distance_product : ∃ y₁ y₂ : ℝ,
  ((-1 - 4)^2 + (y₁ - 3)^2 = 8^2) ∧
  ((-1 - 4)^2 + (y₂ - 3)^2 = 8^2) ∧
  y₁ ≠ y₂ ∧
  y₁ * y₂ = -30 := by
sorry

end NUMINAMATH_CALUDE_point_distance_product_l3754_375470


namespace NUMINAMATH_CALUDE_distinct_power_differences_exist_l3754_375419

theorem distinct_power_differences_exist : ∃ (N : ℕ) (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ),
  (∃ (x₁ x₂ : ℕ), a₁ = x₁^2 ∧ a₂ = x₂^2) ∧
  (∃ (y₁ y₂ : ℕ), b₁ = y₁^3 ∧ b₂ = y₂^3) ∧
  (∃ (z₁ z₂ : ℕ), c₁ = z₁^5 ∧ c₂ = z₂^5) ∧
  (∃ (w₁ w₂ : ℕ), d₁ = w₁^7 ∧ d₂ = w₂^7) ∧
  N = a₁ - a₂ ∧
  N = b₁ - b₂ ∧
  N = c₁ - c₂ ∧
  N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by sorry

end NUMINAMATH_CALUDE_distinct_power_differences_exist_l3754_375419


namespace NUMINAMATH_CALUDE_child_growth_proof_l3754_375406

/-- Calculates the growth in height given current and previous heights -/
def heightGrowth (currentHeight previousHeight : Float) : Float :=
  currentHeight - previousHeight

theorem child_growth_proof :
  let currentHeight : Float := 41.5
  let previousHeight : Float := 38.5
  heightGrowth currentHeight previousHeight = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_growth_proof_l3754_375406


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l3754_375478

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l3754_375478


namespace NUMINAMATH_CALUDE_divisor_probability_l3754_375487

/-- The number of positive divisors of 10^99 -/
def total_divisors : ℕ := 10000

/-- The number of positive divisors of 10^99 that are multiples of 10^88 -/
def favorable_divisors : ℕ := 144

/-- The probability of a randomly chosen positive divisor of 10^99 being a multiple of 10^88 -/
def probability : ℚ := favorable_divisors / total_divisors

theorem divisor_probability :
  probability = 9 / 625 :=
sorry

end NUMINAMATH_CALUDE_divisor_probability_l3754_375487


namespace NUMINAMATH_CALUDE_palabras_bookstore_workers_l3754_375412

theorem palabras_bookstore_workers (W : ℕ) : 
  W / 2 = W / 2 ∧  -- Half of workers read Saramago's book
  W / 6 = W / 6 ∧  -- 1/6 of workers read Kureishi's book
  (∃ (n : ℕ), n = 12 ∧ n ≤ W / 2 ∧ n ≤ W / 6) ∧  -- 12 workers read both books
  (W - (W / 2 + W / 6 - 12)) = ((W / 2 - 12) - 1) →  -- Workers who read neither book
  W = 210 := by
sorry

end NUMINAMATH_CALUDE_palabras_bookstore_workers_l3754_375412


namespace NUMINAMATH_CALUDE_consecutive_squares_determinant_l3754_375469

theorem consecutive_squares_determinant (n : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    (n + (i.val * 3 + j.val : ℕ))^2
  Matrix.det M = -6^3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_determinant_l3754_375469


namespace NUMINAMATH_CALUDE_collinear_points_a_equals_9_l3754_375441

/-- Three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) are collinear if and only if
    (y₂ - y₁)*(x₃ - x₂) = (y₃ - y₂)*(x₂ - x₁) -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁)*(x₃ - x₂) = (y₃ - y₂)*(x₂ - x₁)

/-- If the points (1, 3), (2, 5), and (4, a) are collinear, then a = 9 -/
theorem collinear_points_a_equals_9 :
  collinear 1 3 2 5 4 a → a = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_collinear_points_a_equals_9_l3754_375441


namespace NUMINAMATH_CALUDE_altitude_intersection_angle_l3754_375425

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the altitude intersection point H
def H (t : Triangle) : Point := sorry

-- Define the angles of the triangle
def angle_BAC (t : Triangle) : ℝ := sorry
def angle_ABC (t : Triangle) : ℝ := sorry

-- Define the angle AHB
def angle_AHB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem altitude_intersection_angle (t : Triangle) 
  (h1 : angle_BAC t = 40)
  (h2 : angle_ABC t = 65) :
  angle_AHB t = 105 := by sorry

end NUMINAMATH_CALUDE_altitude_intersection_angle_l3754_375425


namespace NUMINAMATH_CALUDE_dime_difference_is_243_l3754_375410

/-- Represents the types of coins in the piggy bank --/
inductive Coin
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : Coin → Nat
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A configuration of coins in the piggy bank --/
structure CoinConfiguration where
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total number of coins in a configuration --/
def totalCoins (c : CoinConfiguration) : Nat :=
  c.nickels + c.dimes + c.quarters + c.halfDollars

/-- The total value of coins in a configuration in cents --/
def totalValue (c : CoinConfiguration) : Nat :=
  c.nickels * coinValue Coin.Nickel +
  c.dimes * coinValue Coin.Dime +
  c.quarters * coinValue Coin.Quarter +
  c.halfDollars * coinValue Coin.HalfDollar

/-- Predicate to check if a configuration is valid --/
def isValidConfiguration (c : CoinConfiguration) : Prop :=
  totalCoins c = 150 ∧ totalValue c = 2000

/-- The maximum number of dimes possible in a valid configuration --/
def maxDimes : Nat :=
  250

/-- The minimum number of dimes possible in a valid configuration --/
def minDimes : Nat :=
  7

theorem dime_difference_is_243 :
  ∃ (cMax cMin : CoinConfiguration),
    isValidConfiguration cMax ∧
    isValidConfiguration cMin ∧
    cMax.dimes = maxDimes ∧
    cMin.dimes = minDimes ∧
    maxDimes - minDimes = 243 :=
  sorry

end NUMINAMATH_CALUDE_dime_difference_is_243_l3754_375410


namespace NUMINAMATH_CALUDE_expand_product_l3754_375429

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3754_375429


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3754_375493

/-- Given a rectangle with side OA and diagonal OB, prove the value of k. -/
theorem rectangle_diagonal (OA OB : ℝ × ℝ) (k : ℝ) : 
  OA = (-3, 1) → 
  OB = (-2, k) → 
  (OA.1 * (OB.1 - OA.1) + OA.2 * (OB.2 - OA.2) = 0) → 
  k = 4 := by
  sorry

#check rectangle_diagonal

end NUMINAMATH_CALUDE_rectangle_diagonal_l3754_375493


namespace NUMINAMATH_CALUDE_solution_sum_l3754_375499

theorem solution_sum (x y : ℤ) : 
  (x : ℝ) * Real.log 27 * (Real.log 13)⁻¹ = 27 * Real.log y / Real.log 13 →
  y > 70 →
  ∀ z, z > 70 → z < y →
  x + y = 117 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l3754_375499


namespace NUMINAMATH_CALUDE_tan_beta_value_l3754_375491

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/7) 
  (h2 : Real.tan (α + β) = 1/3) : 
  Real.tan β = 2/11 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3754_375491


namespace NUMINAMATH_CALUDE_pool_filling_rounds_l3754_375486

/-- The number of buckets George can carry per round -/
def george_buckets : ℕ := 2

/-- The number of buckets Harry can carry per round -/
def harry_buckets : ℕ := 3

/-- The total number of buckets needed to fill the pool -/
def total_buckets : ℕ := 110

/-- The number of rounds needed to fill the pool -/
def rounds_to_fill : ℕ := total_buckets / (george_buckets + harry_buckets)

theorem pool_filling_rounds :
  rounds_to_fill = 22 := by sorry

end NUMINAMATH_CALUDE_pool_filling_rounds_l3754_375486


namespace NUMINAMATH_CALUDE_product_even_sum_undetermined_l3754_375455

theorem product_even_sum_undetermined (a b : ℤ) : 
  Even (a * b) → (Even (a + b) ∨ Odd (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_product_even_sum_undetermined_l3754_375455


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l3754_375497

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.2 * x := by
sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l3754_375497


namespace NUMINAMATH_CALUDE_union_A_complement_B_range_of_a_l3754_375445

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 3}

-- Theorem for part (1)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 4} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) (h : A ∩ C a = A) : 1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_range_of_a_l3754_375445


namespace NUMINAMATH_CALUDE_tom_total_money_l3754_375482

/-- Tom's initial amount of money in dollars -/
def initial_amount : ℕ := 74

/-- Amount Tom earned from washing cars in dollars -/
def car_wash_earnings : ℕ := 86

/-- Tom's total money after washing cars -/
def total_money : ℕ := initial_amount + car_wash_earnings

theorem tom_total_money :
  total_money = 160 := by sorry

end NUMINAMATH_CALUDE_tom_total_money_l3754_375482


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3754_375422

/-- The x-intercept of the line 2x - 4y = 12 is 6 -/
theorem x_intercept_of_line (x y : ℝ) : 2 * x - 4 * y = 12 → y = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3754_375422


namespace NUMINAMATH_CALUDE_ellipse_properties_l3754_375476

/-- Properties of an ellipse with given parameters -/
theorem ellipse_properties :
  let e : ℝ := 1/2  -- eccentricity
  let c : ℝ := 1    -- half the distance between foci
  let a : ℝ := 2    -- semi-major axis
  let b : ℝ := Real.sqrt 3  -- semi-minor axis
  let F₁ : ℝ × ℝ := (-1, 0)  -- left focus
  let A : ℝ × ℝ := (-2, 0)  -- left vertex
  ∀ x y : ℝ,
    (x^2 / 4 + y^2 / 3 = 1) →  -- point (x,y) is on the ellipse
    (0 ≤ (x + 1) * (x + 2) + y^2) ∧
    ((x + 1) * (x + 2) + y^2 ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3754_375476


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3754_375447

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3754_375447


namespace NUMINAMATH_CALUDE_farmer_seeds_sowed_l3754_375430

/-- The number of buckets of seeds sowed by a farmer -/
def seeds_sowed (initial final : ℝ) : ℝ := initial - final

/-- Theorem stating that the farmer sowed 2.75 buckets of seeds -/
theorem farmer_seeds_sowed :
  seeds_sowed 8.75 6 = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_farmer_seeds_sowed_l3754_375430


namespace NUMINAMATH_CALUDE_tank_filling_time_tank_filling_time_proof_l3754_375414

theorem tank_filling_time : ℝ → Prop :=
  fun T : ℝ =>
    let fill_rate_A : ℝ := 1 / 60
    let fill_rate_B : ℝ := 1 / 40
    let first_half : ℝ := T / 2 * fill_rate_B
    let second_half : ℝ := T / 2 * (fill_rate_A + fill_rate_B)
    (first_half + second_half = 1) → (T = 48)

-- The proof goes here
theorem tank_filling_time_proof : tank_filling_time 48 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_tank_filling_time_proof_l3754_375414


namespace NUMINAMATH_CALUDE_events_complementary_l3754_375405

-- Define the sample space for a fair die
def DieOutcome := Fin 6

-- Define Event 1: odd numbers
def Event1 (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 1

-- Define Event 2: even numbers
def Event2 (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 0

-- Theorem stating that Event1 and Event2 are complementary
theorem events_complementary :
  ∀ (outcome : DieOutcome), Event1 outcome ↔ ¬Event2 outcome :=
sorry

end NUMINAMATH_CALUDE_events_complementary_l3754_375405


namespace NUMINAMATH_CALUDE_expression_equality_l3754_375439

theorem expression_equality : -2^2 + (1 / (Real.sqrt 2 - 1))^0 - abs (2 * Real.sqrt 2 - 3) + Real.cos (π / 3) = -5 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3754_375439


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l3754_375495

theorem jordan_rectangle_width
  (carol_length : ℝ)
  (carol_width : ℝ)
  (jordan_length : ℝ)
  (jordan_width : ℝ)
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_length = 8)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 15 := by
sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l3754_375495


namespace NUMINAMATH_CALUDE_christinas_driving_time_l3754_375461

theorem christinas_driving_time 
  (total_distance : ℝ) 
  (christina_speed : ℝ) 
  (friend_speed : ℝ) 
  (friend_time : ℝ) 
  (h1 : total_distance = 210)
  (h2 : christina_speed = 30)
  (h3 : friend_speed = 40)
  (h4 : friend_time = 3) :
  (total_distance - friend_speed * friend_time) / christina_speed * 60 = 180 :=
by sorry

end NUMINAMATH_CALUDE_christinas_driving_time_l3754_375461


namespace NUMINAMATH_CALUDE_a_in_S_l3754_375490

def S : Set ℤ := {n | ∃ x y : ℤ, n = x^2 + 2*y^2}

theorem a_in_S (a : ℤ) (h : 3*a ∈ S) : a ∈ S := by
  sorry

end NUMINAMATH_CALUDE_a_in_S_l3754_375490


namespace NUMINAMATH_CALUDE_distance_between_centers_l3754_375446

/-- Given a triangle with sides 5, 12, and 13, the distance between the centers
    of its inscribed and circumscribed circles is √65/2 -/
theorem distance_between_centers (a b c : ℝ) (h_sides : a = 5 ∧ b = 12 ∧ c = 13) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt ((circumradius - inradius) ^ 2 + (area / (a * b * c) * (a + b - c) * (b + c - a) * (c + a - b))) = Real.sqrt 65 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3754_375446


namespace NUMINAMATH_CALUDE_cindy_calculation_l3754_375454

theorem cindy_calculation (x : ℚ) : (x - 7) / 5 = 25 → (x - 5) / 7 = 18 + 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l3754_375454


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_32_seconds_l3754_375452

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (train_length : ℝ) : ℝ :=
by
  -- Convert speeds from km/hr to m/s
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  
  -- Calculate relative speed
  let relative_speed := train_speed_ms - jogger_speed_ms
  
  -- Calculate total distance to be covered
  let total_distance := initial_distance + train_length
  
  -- Calculate time taken
  let time_taken := total_distance / relative_speed
  
  -- Prove that the time taken is 32 seconds
  sorry

/-- The main theorem stating that the train will pass the jogger in 32 seconds -/
theorem train_passes_jogger_in_32_seconds : 
  train_passing_jogger_time 9 45 200 120 = 32 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_32_seconds_l3754_375452


namespace NUMINAMATH_CALUDE_max_students_distribution_l3754_375437

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1340) (h_pencils : pencils = 1280) : 
  Nat.gcd pens pencils = 20 := by
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3754_375437


namespace NUMINAMATH_CALUDE_nell_card_difference_l3754_375460

/-- Given Nell's card collection information, prove the difference between
    her final Ace cards and baseball cards. -/
theorem nell_card_difference
  (initial_baseball : ℕ)
  (initial_ace : ℕ)
  (final_baseball : ℕ)
  (final_ace : ℕ)
  (h1 : initial_baseball = 239)
  (h2 : initial_ace = 38)
  (h3 : final_baseball = 111)
  (h4 : final_ace = 376) :
  final_ace - final_baseball = 265 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l3754_375460


namespace NUMINAMATH_CALUDE_hound_catches_hare_l3754_375431

/-- The number of jumps required for a hound to catch a hare -/
def catchHare (initialSeparation : ℕ) (hareJump : ℕ) (houndJump : ℕ) : ℕ :=
  initialSeparation / (houndJump - hareJump)

/-- Theorem stating that given the specific conditions, the hound catches the hare in 75 jumps -/
theorem hound_catches_hare :
  catchHare 150 7 9 = 75 := by
  sorry

#eval catchHare 150 7 9

end NUMINAMATH_CALUDE_hound_catches_hare_l3754_375431


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3754_375423

def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetric_points_sum_power (m n : ℝ) :
  symmetric_about_y_axis m 3 4 n →
  (m + n)^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3754_375423


namespace NUMINAMATH_CALUDE_some_number_value_l3754_375453

theorem some_number_value : ∃ (x : ℚ), 
  (1 / 2 : ℚ) + ((2 / 3 : ℚ) * (3 / 8 : ℚ) + x) - (8 / 16 : ℚ) = (17 / 4 : ℚ) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3754_375453


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3754_375416

theorem scientific_notation_equality : (58000000000 : ℝ) = 5.8 * (10 ^ 10) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3754_375416
