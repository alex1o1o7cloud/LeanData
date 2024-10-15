import Mathlib

namespace NUMINAMATH_CALUDE_winnie_repetitions_l1773_177377

/-- Calculates the number of repetitions completed today given yesterday's
    repetitions and the difference in performance. -/
def repetitions_today (yesterday : ℕ) (difference : ℕ) : ℕ :=
  yesterday - difference

/-- Proves that Winnie completed 73 repetitions today given the conditions. -/
theorem winnie_repetitions :
  repetitions_today 86 13 = 73 := by
  sorry

end NUMINAMATH_CALUDE_winnie_repetitions_l1773_177377


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1773_177384

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * (8 / y + 6 * y^3) = 6 / y + (9 * y^3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1773_177384


namespace NUMINAMATH_CALUDE_train_length_problem_l1773_177347

/-- Proves that given two trains moving in opposite directions with specified speeds and time to pass,
    the length of the first train is 150 meters. -/
theorem train_length_problem (v1 v2 l2 t : ℝ) (h1 : v1 = 80) (h2 : v2 = 70) (h3 : l2 = 100) 
    (h4 : t = 5.999520038396928) : ∃ l1 : ℝ, l1 = 150 ∧ (v1 + v2) * t * (5/18) = l1 + l2 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l1773_177347


namespace NUMINAMATH_CALUDE_minimize_expression_l1773_177306

theorem minimize_expression (a b : ℝ) (h1 : a + b = -2) (h2 : b < 0) :
  ∃ (min_a : ℝ), min_a = 2 ∧
  ∀ (x : ℝ), x + b = -2 → (1 / (2 * |x|) - |x| / b) ≥ (1 / (2 * |min_a|) - |min_a| / b) :=
sorry

end NUMINAMATH_CALUDE_minimize_expression_l1773_177306


namespace NUMINAMATH_CALUDE_equation_solution_l1773_177379

theorem equation_solution : ∃ x : ℚ, (5 * x + 9 * x = 570 - 12 * (x - 5)) ∧ (x = 315 / 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1773_177379


namespace NUMINAMATH_CALUDE_larger_integer_of_product_and_sum_l1773_177311

theorem larger_integer_of_product_and_sum (x y : ℤ) 
  (h_product : x * y = 30) 
  (h_sum : x + y = 13) : 
  max x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_of_product_and_sum_l1773_177311


namespace NUMINAMATH_CALUDE_gcd_50421_35343_l1773_177305

theorem gcd_50421_35343 : Nat.gcd 50421 35343 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50421_35343_l1773_177305


namespace NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l1773_177372

theorem modular_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l1773_177372


namespace NUMINAMATH_CALUDE_prime_power_congruence_l1773_177350

theorem prime_power_congruence (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  (p^(p+2) + (p+2)^p) % (2*p+2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_congruence_l1773_177350


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l1773_177334

/-- Rachel's apple picking problem -/
theorem rachel_apple_picking (num_trees : ℕ) (apples_left : ℕ) (initial_apples : ℕ) : 
  num_trees = 3 → apples_left = 9 → initial_apples = 33 → 
  (initial_apples - apples_left) / num_trees = 8 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l1773_177334


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l1773_177351

theorem triangle_sine_theorem (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ) :
  area = 30 →
  side = 10 →
  median = 9 →
  area = (1/2) * side * median * Real.sin θ →
  0 < θ →
  θ < π/2 →
  Real.sin θ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l1773_177351


namespace NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l1773_177333

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * (a - 3) * x + 1

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (lb : ℝ) : Prop :=
  ∀ x y, lb ≤ x → x < y → f y < f x

-- State the theorem
theorem function_decreasing_iff_a_in_range :
  ∀ a : ℝ, (is_decreasing_on (f a) (-2)) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l1773_177333


namespace NUMINAMATH_CALUDE_P_inter_Q_eq_interval_l1773_177322

/-- The set P defined by the inequality 3x - x^2 ≤ 0 -/
def P : Set ℝ := {x : ℝ | 3 * x - x^2 ≤ 0}

/-- The set Q defined by the inequality |x| ≤ 2 -/
def Q : Set ℝ := {x : ℝ | |x| ≤ 2}

/-- The theorem stating that the intersection of P and Q is equal to the set {x | -2 ≤ x ≤ 0} -/
theorem P_inter_Q_eq_interval : P ∩ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_P_inter_Q_eq_interval_l1773_177322


namespace NUMINAMATH_CALUDE_right_triangle_max_area_right_triangle_max_area_achieved_right_triangle_max_area_is_nine_l1773_177319

theorem right_triangle_max_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_hypotenuse : c = 6) :
  a * b ≤ 18 := by
  sorry

theorem right_triangle_max_area_achieved (a b : ℝ) (h_right : a^2 + b^2 = 36) (h_equal : a = b) :
  a * b = 18 := by
  sorry

theorem right_triangle_max_area_is_nine :
  ∃ (a b : ℝ), a^2 + b^2 = 36 ∧ a * b / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_area_right_triangle_max_area_achieved_right_triangle_max_area_is_nine_l1773_177319


namespace NUMINAMATH_CALUDE_pencils_misplaced_l1773_177314

theorem pencils_misplaced (initial : ℕ) (broken found bought final : ℕ) : 
  initial = 20 →
  broken = 3 →
  found = 4 →
  bought = 2 →
  final = 16 →
  initial - broken + found + bought - final = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencils_misplaced_l1773_177314


namespace NUMINAMATH_CALUDE_jackson_vacuum_count_l1773_177362

def chore_pay_rate : ℝ := 5
def vacuum_time : ℝ := 2
def dish_washing_time : ℝ := 0.5
def total_earnings : ℝ := 30

def total_chore_time (vacuum_count : ℝ) : ℝ :=
  vacuum_count * vacuum_time + dish_washing_time + 3 * dish_washing_time

theorem jackson_vacuum_count :
  ∃ (vacuum_count : ℝ), 
    total_chore_time vacuum_count * chore_pay_rate = total_earnings ∧ 
    vacuum_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_jackson_vacuum_count_l1773_177362


namespace NUMINAMATH_CALUDE_b_share_is_1540_l1773_177300

/-- Represents the share of profits for a partner in a partnership. -/
structure PartnerShare where
  investment : ℕ
  share : ℕ

/-- Calculates the share of a partner given the total profit and the investment ratios. -/
def calculateShare (totalProfit : ℕ) (investmentRatios : List ℕ) (partnerRatio : ℕ) : ℕ :=
  (totalProfit * partnerRatio) / (investmentRatios.sum)

/-- Theorem stating that given the investments and a's share, b's share is $1540. -/
theorem b_share_is_1540 (a b c : PartnerShare) 
  (h1 : a.investment = 15000)
  (h2 : b.investment = 21000)
  (h3 : c.investment = 27000)
  (h4 : a.share = 1100) : 
  b.share = 1540 := by
  sorry


end NUMINAMATH_CALUDE_b_share_is_1540_l1773_177300


namespace NUMINAMATH_CALUDE_triangle_problem_l1773_177380

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  (2 * c - b) / (Real.sqrt 3 * Real.sin C - Real.cos C) = a →
  -- b = 1
  b = 1 →
  -- Area condition
  (1 / 2) * b * c * Real.sin A = (3 / 4) * Real.tan A →
  -- Prove A = π/3 and a = √7
  A = π / 3 ∧ a = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1773_177380


namespace NUMINAMATH_CALUDE_women_who_left_l1773_177303

/-- Proves the number of women who left the room given the initial and final conditions --/
theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) 
  (h1 : initial_men * 5 = initial_women * 4)  -- Initial ratio of men to women is 4:5
  (h2 : initial_men + 2 = 14)  -- 2 men entered, final count is 14 men
  (h3 : 2 * (initial_women - women_left) = 24)  -- Women doubled after some left, final count is 24 women
  : women_left = 3 := by
  sorry

#check women_who_left

end NUMINAMATH_CALUDE_women_who_left_l1773_177303


namespace NUMINAMATH_CALUDE_exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1773_177385

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls -/
def sampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack : Set DrawOutcome := sorry

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack : Set DrawOutcome := sorry

/-- Definition of mutually exclusive events -/
def mutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Definition of complementary events -/
def complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = sampleSpace ∧ A ∩ B = ∅

/-- Main theorem: exactlyOneBlack and exactlyTwoBlack are mutually exclusive but not complementary -/
theorem exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoBlack ∧
  ¬complementary exactlyOneBlack exactlyTwoBlack := by
  sorry

end NUMINAMATH_CALUDE_exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1773_177385


namespace NUMINAMATH_CALUDE_subtraction_value_l1773_177321

theorem subtraction_value (N : ℝ) (h1 : (N - 24) / 10 = 3) : 
  ∃ x : ℝ, (N - x) / 7 = 7 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_value_l1773_177321


namespace NUMINAMATH_CALUDE_total_kids_l1773_177323

theorem total_kids (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 3) (h2 : boys = 6) : 
  girls + boys = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_l1773_177323


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1773_177348

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 5)) = Real.sqrt 10 → y = 41 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1773_177348


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_A_subset_B_iff_l1773_177346

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1
theorem intersection_A_complement_B (a : ℝ) (h : a = -2) :
  A a ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2
theorem A_subset_B_iff (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_A_subset_B_iff_l1773_177346


namespace NUMINAMATH_CALUDE_race_outcomes_l1773_177313

/-- The number of permutations of n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of positions to be filled -/
def positions_to_fill : ℕ := 4

theorem race_outcomes : permutations num_participants positions_to_fill = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l1773_177313


namespace NUMINAMATH_CALUDE_max_segment_for_quadrilateral_l1773_177336

theorem max_segment_for_quadrilateral
  (a b c d : ℝ)
  (total_length : a + b + c + d = 2)
  (ordered_segments : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (∃ (x : ℝ), x < 1 ∧
    (∀ (y : ℝ), y < x →
      (a + b > y ∧ a + c > y ∧ a + d > y ∧
       b + c > y ∧ b + d > y ∧ c + d > y))) ∧
  (∀ (z : ℝ), z ≥ 1 →
    ¬(a + b > z ∧ a + c > z ∧ a + d > z ∧
      b + c > z ∧ b + d > z ∧ c + d > z)) :=
by sorry

end NUMINAMATH_CALUDE_max_segment_for_quadrilateral_l1773_177336


namespace NUMINAMATH_CALUDE_connie_calculation_l1773_177381

theorem connie_calculation (x : ℝ) : 200 - x = 100 → 200 + x = 300 := by
  sorry

end NUMINAMATH_CALUDE_connie_calculation_l1773_177381


namespace NUMINAMATH_CALUDE_square_of_one_plus_i_l1773_177356

theorem square_of_one_plus_i :
  let z : ℂ := 1 + Complex.I
  z^2 = 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_square_of_one_plus_i_l1773_177356


namespace NUMINAMATH_CALUDE_angle_three_times_complement_l1773_177340

theorem angle_three_times_complement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_three_times_complement_l1773_177340


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l1773_177383

theorem complex_number_opposite_parts (a : ℝ) : 
  (∃ z : ℂ, z = (2 + a * Complex.I) * Complex.I ∧ 
   z.re = -z.im) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l1773_177383


namespace NUMINAMATH_CALUDE_age_difference_proof_l1773_177345

-- Define the ages of Betty, Mary, and Albert
def betty_age : ℕ := 11
def albert_age (betty_age : ℕ) : ℕ := 4 * betty_age
def mary_age (albert_age : ℕ) : ℕ := albert_age / 2

-- Theorem statement
theorem age_difference_proof :
  albert_age betty_age - mary_age (albert_age betty_age) = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1773_177345


namespace NUMINAMATH_CALUDE_max_profit_morel_purchase_l1773_177310

/-- Represents the purchase and profit calculation for Morel mushrooms. -/
structure MorelPurchase where
  freshPrice : ℝ  -- Purchase price of fresh Morel mushrooms (RMB/kg)
  driedPrice : ℝ  -- Purchase price of dried Morel mushrooms (RMB/kg)
  freshRetail : ℝ  -- Retail price of fresh Morel mushrooms (RMB/kg)
  driedRetail : ℝ  -- Retail price of dried Morel mushrooms (RMB/kg)
  totalQuantity : ℝ  -- Total quantity to purchase (kg)

/-- Calculates the profit for a given purchase plan. -/
def calculateProfit (p : MorelPurchase) (freshQuant : ℝ) : ℝ :=
  let driedQuant := p.totalQuantity - freshQuant
  (p.freshRetail - p.freshPrice) * freshQuant + (p.driedRetail - p.driedPrice) * driedQuant

/-- Theorem stating that the maximum profit is achieved with the specified quantities. -/
theorem max_profit_morel_purchase (p : MorelPurchase)
    (h1 : p.freshPrice = 80)
    (h2 : p.driedPrice = 240)
    (h3 : p.freshRetail = 100)
    (h4 : p.driedRetail = 280)
    (h5 : p.totalQuantity = 1500) :
    ∃ (maxProfit : ℝ) (optimalFresh : ℝ),
      maxProfit = 37500 ∧
      optimalFresh = 1125 ∧
      ∀ (freshQuant : ℝ), 0 ≤ freshQuant ∧ freshQuant ≤ p.totalQuantity ∧
        3 * (p.totalQuantity - freshQuant) ≤ freshQuant →
        calculateProfit p freshQuant ≤ maxProfit := by
  sorry


end NUMINAMATH_CALUDE_max_profit_morel_purchase_l1773_177310


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1773_177339

theorem contradiction_assumption (a b c d : ℝ) :
  (¬ (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0)) ↔ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1773_177339


namespace NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l1773_177318

/-- Given a set of four numbers with six pairwise sums, find the maximum value of x + y -/
theorem max_sum_of_pairwise_sums (a b c d : ℝ) : 
  let sums := [a + b, a + c, a + d, b + c, b + d, c + d]
  ∃ (perm : List ℝ), perm.Perm sums ∧ 
    perm.take 4 = [210, 336, 294, 252] →
  (∃ (x y : ℝ), x ∈ perm.drop 4 ∧ y ∈ perm.drop 4 ∧ x + y ≤ 798) ∧
  (∃ (a' b' c' d' : ℝ), 
    let sums' := [a' + b', a' + c', a' + d', b' + c', b' + d', c' + d']
    ∃ (perm' : List ℝ), perm'.Perm sums' ∧ 
      perm'.take 4 = [210, 336, 294, 252] ∧
      ∃ (x' y' : ℝ), x' ∈ perm'.drop 4 ∧ y' ∈ perm'.drop 4 ∧ x' + y' = 798) := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l1773_177318


namespace NUMINAMATH_CALUDE_alice_bob_meet_l1773_177304

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 13

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 9

theorem alice_bob_meet :
  (meeting_turns * alice_move) % n = (meeting_turns * (n - bob_move)) % n :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l1773_177304


namespace NUMINAMATH_CALUDE_expand_x_plus_y_seventh_third_to_fourth_term_ratio_p_plus_q_equals_three_p_and_q_positive_prove_p_value_l1773_177343

/-- The value of p in the expansion of (x+y)^7 -/
def p : ℚ :=
  30/13

/-- The value of q in the expansion of (x+y)^7 -/
def q : ℚ :=
  9/13

/-- The ratio of the third to fourth term in the expansion of (x+y)^7 when x=p and y=q -/
def ratio : ℚ :=
  2/1

theorem expand_x_plus_y_seventh (x y : ℚ) :
  (x + y)^7 = x^7 + 7*x^6*y + 21*x^5*y^2 + 35*x^4*y^3 + 35*x^3*y^4 + 21*x^2*y^5 + 7*x*y^6 + y^7 :=
sorry

theorem third_to_fourth_term_ratio :
  (21 * p^5 * q^2) / (35 * p^4 * q^3) = ratio :=
sorry

theorem p_plus_q_equals_three :
  p + q = 3 :=
sorry

theorem p_and_q_positive :
  p > 0 ∧ q > 0 :=
sorry

theorem prove_p_value :
  p = 30/13 :=
sorry

end NUMINAMATH_CALUDE_expand_x_plus_y_seventh_third_to_fourth_term_ratio_p_plus_q_equals_three_p_and_q_positive_prove_p_value_l1773_177343


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1773_177369

theorem negation_of_universal_proposition :
  (¬ ∀ (m : ℝ) (x : ℝ), m ∈ Set.Icc 0 1 → x + 1/x ≥ 2^m) ↔
  (∃ (m : ℝ) (x : ℝ), m ∈ Set.Icc 0 1 ∧ x + 1/x < 2^m) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1773_177369


namespace NUMINAMATH_CALUDE_square_triangle_circle_perimeter_l1773_177353

theorem square_triangle_circle_perimeter (x : ℝ) : 
  (4 * x) + (3 * x) = 2 * π * 4 → x = (8 * π) / 7 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_circle_perimeter_l1773_177353


namespace NUMINAMATH_CALUDE_max_distance_complex_numbers_l1773_177324

theorem max_distance_complex_numbers (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((1 + 2*Complex.I)*z - z^2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_complex_numbers_l1773_177324


namespace NUMINAMATH_CALUDE_max_sum_composite_shape_l1773_177389

/-- Represents a composite shape formed by adding a pyramid to a pentagonal prism --/
structure CompositePrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_faces : Nat
  pyramid_edges : Nat
  pyramid_vertices : Nat

/-- The total number of faces in the composite shape --/
def total_faces (shape : CompositePrismPyramid) : Nat :=
  shape.prism_faces + shape.pyramid_faces - 1

/-- The total number of edges in the composite shape --/
def total_edges (shape : CompositePrismPyramid) : Nat :=
  shape.prism_edges + shape.pyramid_edges

/-- The total number of vertices in the composite shape --/
def total_vertices (shape : CompositePrismPyramid) : Nat :=
  shape.prism_vertices + shape.pyramid_vertices

/-- The sum of faces, edges, and vertices in the composite shape --/
def total_sum (shape : CompositePrismPyramid) : Nat :=
  total_faces shape + total_edges shape + total_vertices shape

/-- Theorem stating the maximum sum of faces, edges, and vertices --/
theorem max_sum_composite_shape :
  ∃ (shape : CompositePrismPyramid),
    shape.prism_faces = 7 ∧
    shape.prism_edges = 15 ∧
    shape.prism_vertices = 10 ∧
    shape.pyramid_faces = 5 ∧
    shape.pyramid_edges = 5 ∧
    shape.pyramid_vertices = 1 ∧
    total_sum shape = 42 ∧
    ∀ (other : CompositePrismPyramid), total_sum other ≤ 42 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_composite_shape_l1773_177389


namespace NUMINAMATH_CALUDE_equation_solution_l1773_177359

theorem equation_solution : ∃ x : ℝ, 24 - 4 * 2 = 3 + x ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1773_177359


namespace NUMINAMATH_CALUDE_sine_product_ratio_l1773_177335

theorem sine_product_ratio (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (5 * c) * Real.sin (7 * c) * Real.sin (9 * c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_ratio_l1773_177335


namespace NUMINAMATH_CALUDE_vector_angle_difference_l1773_177338

theorem vector_angle_difference (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : a = λ i => if i = 0 then Real.cos α else Real.sin α)
  (h5 : b = λ i => if i = 0 then Real.cos β else Real.sin β)
  (h6 : ‖(2 : Real) • a + b‖ = ‖a - (2 : Real) • b‖) :
  β - α = π / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_angle_difference_l1773_177338


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1773_177370

theorem triangle_angle_calculation (y : ℝ) : 
  y > 0 ∧ y < 60 ∧ 45 + 3 * y + y = 180 → y = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1773_177370


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l1773_177371

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 81 * Real.pi := by sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l1773_177371


namespace NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l1773_177357

/-- Represents the number of students in a school -/
structure School where
  boarders : ℕ
  dayScholars : ℕ

/-- Represents a ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def School.ratio (s : School) : Ratio :=
  { numerator := s.boarders, denominator := s.dayScholars }

def School.addBoarders (s : School) (n : ℕ) : School :=
  { boarders := s.boarders + n, dayScholars := s.dayScholars }

theorem new_ratio_after_boarders_join
  (initialSchool : School)
  (initialRatio : Ratio)
  (newBoarders : ℕ) :
  initialSchool.ratio = initialRatio →
  initialSchool.boarders = 560 →
  initialRatio.numerator = 7 →
  initialRatio.denominator = 16 →
  newBoarders = 80 →
  (initialSchool.addBoarders newBoarders).ratio =
    { numerator := 1, denominator := 2 } :=
by
  sorry

end NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l1773_177357


namespace NUMINAMATH_CALUDE_tammy_haircuts_needed_l1773_177341

/-- Represents the haircut system for Tammy -/
structure HaircutSystem where
  total_haircuts : ℕ
  free_haircuts : ℕ
  haircuts_until_next_free : ℕ

/-- Calculates the number of haircuts needed for the next free one -/
def haircuts_needed (system : HaircutSystem) : ℕ :=
  system.haircuts_until_next_free

/-- Theorem stating that Tammy needs 5 more haircuts for her next free one -/
theorem tammy_haircuts_needed (system : HaircutSystem) 
  (h1 : system.total_haircuts = 79)
  (h2 : system.free_haircuts = 5)
  (h3 : system.haircuts_until_next_free = 5) :
  haircuts_needed system = 5 := by
  sorry

#eval haircuts_needed { total_haircuts := 79, free_haircuts := 5, haircuts_until_next_free := 5 }

end NUMINAMATH_CALUDE_tammy_haircuts_needed_l1773_177341


namespace NUMINAMATH_CALUDE_apple_pie_calculation_l1773_177364

theorem apple_pie_calculation (total_apples : ℕ) (unripe_apples : ℕ) (apples_per_pie : ℕ) :
  total_apples = 34 →
  unripe_apples = 6 →
  apples_per_pie = 4 →
  (total_apples - unripe_apples) / apples_per_pie = 7 :=
by sorry

end NUMINAMATH_CALUDE_apple_pie_calculation_l1773_177364


namespace NUMINAMATH_CALUDE_set_operation_result_l1773_177382

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem set_operation_result :
  (M ∩ N) ∪ (U \ N) = {0, 1, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1773_177382


namespace NUMINAMATH_CALUDE_triangle_max_area_l1773_177330

/-- Given a triangle ABC where c = 2 and b = √2 * a, 
    the maximum area of the triangle is 2√2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : c = 2) (h2 : b = Real.sqrt 2 * a) :
  ∃ (S : ℝ), S = (Real.sqrt 2 : ℝ) * 2 ∧ 
  (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2)/(2*a*b))) → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1773_177330


namespace NUMINAMATH_CALUDE_cubic_roots_expression_l1773_177374

theorem cubic_roots_expression (α β γ : ℂ) : 
  (α^3 - 3*α - 2 = 0) → 
  (β^3 - 3*β - 2 = 0) → 
  (γ^3 - 3*γ - 2 = 0) → 
  α*(β - γ)^2 + β*(γ - α)^2 + γ*(α - β)^2 = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_expression_l1773_177374


namespace NUMINAMATH_CALUDE_polygon_35_sides_5_restricted_l1773_177365

/-- The number of diagonals in a convex polygon with restricted vertices -/
def diagonals_with_restrictions (n : ℕ) (r : ℕ) : ℕ :=
  let effective_vertices := n - r
  (effective_vertices * (effective_vertices - 3)) / 2

/-- Theorem: A convex polygon with 35 sides and 5 restricted vertices has 405 diagonals -/
theorem polygon_35_sides_5_restricted : diagonals_with_restrictions 35 5 = 405 := by
  sorry

end NUMINAMATH_CALUDE_polygon_35_sides_5_restricted_l1773_177365


namespace NUMINAMATH_CALUDE_product_of_roots_l1773_177312

theorem product_of_roots (x : ℂ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  (∃ p q r : ℂ, x^3 - 15*x^2 + 75*x - 50 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 50) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1773_177312


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1773_177328

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointLiesOnPlane (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The greatest common divisor of the absolute values of the coefficients is 1 -/
def coefficientsAreCoprime (coeff : PlaneCoefficients) : Prop :=
  Nat.gcd (Int.natAbs coeff.A) (Nat.gcd (Int.natAbs coeff.B) (Nat.gcd (Int.natAbs coeff.C) (Int.natAbs coeff.D))) = 1

theorem plane_equation_proof (p1 p2 p3 : Point3D) (coeff : PlaneCoefficients) : 
  p1 = ⟨2, -1, 3⟩ →
  p2 = ⟨0, -1, 5⟩ →
  p3 = ⟨-1, -3, 4⟩ →
  coeff = ⟨1, 2, -1, 3⟩ →
  pointLiesOnPlane p1 coeff ∧
  pointLiesOnPlane p2 coeff ∧
  pointLiesOnPlane p3 coeff ∧
  coeff.A > 0 ∧
  coefficientsAreCoprime coeff :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l1773_177328


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1773_177308

theorem polynomial_divisibility (m n : ℕ+) :
  ∃ q : Polynomial ℚ, (X^2 + X + 1) * q = X^(3*m.val + 1) + X^(3*n.val + 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1773_177308


namespace NUMINAMATH_CALUDE_dark_tile_fraction_for_given_floor_l1773_177386

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor :=
  (pattern_size : Nat)
  (dark_tiles_in_quarter : Nat)

/-- Calculates the fraction of dark tiles in the entire floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  sorry

/-- Theorem stating the fraction of dark tiles in the given floor configuration -/
theorem dark_tile_fraction_for_given_floor :
  let floor := TiledFloor.mk 8 10
  dark_tile_fraction floor = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_for_given_floor_l1773_177386


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1773_177399

theorem max_perimeter_special_triangle :
  ∀ x : ℕ,
    x > 0 →
    x ≤ 20 →
    x + 4*x > 20 →
    x + 20 > 4*x →
    4*x + 20 > x →
    (∀ y : ℕ, 
      y > 0 →
      y ≤ 20 →
      y + 4*y > 20 →
      y + 20 > 4*y →
      4*y + 20 > y →
      x + 4*x + 20 ≥ y + 4*y + 20) →
    x + 4*x + 20 = 50 :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1773_177399


namespace NUMINAMATH_CALUDE_count_box_triples_l1773_177398

/-- The number of ordered triples (a, b, c) satisfying the box conditions -/
def box_triples : ℕ := 3

/-- Predicate defining the conditions for a valid box triple -/
def is_valid_box_triple (a b c : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (a * b * c : ℚ) = (2 / 3) * (a * b + b * c + c * a)

/-- Theorem stating that there are exactly 3 ordered triples satisfying the box conditions -/
theorem count_box_triples :
  (∃ (S : Finset (ℕ × ℕ × ℕ)), S.card = box_triples ∧
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_box_triple t.1 t.2.1 t.2.2)) :=
sorry

end NUMINAMATH_CALUDE_count_box_triples_l1773_177398


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1773_177378

-- Define the inequality
def inequality (x : ℝ) : Prop := 4 * x - 5 < 3

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1773_177378


namespace NUMINAMATH_CALUDE_quadruplet_equation_equivalence_l1773_177387

theorem quadruplet_equation_equivalence (x y z w : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  (1 + 1/x + 2*(x+1)/(x*y) + 3*(x+1)*(y+2)/(x*y*z) + 4*(x+1)*(y+2)*(z+3)/(x*y*z*w) = 0) ↔
  ((x+1)*(y+2)*(z+3)*(w+4) = 0) := by
sorry

end NUMINAMATH_CALUDE_quadruplet_equation_equivalence_l1773_177387


namespace NUMINAMATH_CALUDE_weekday_hours_are_six_l1773_177388

/-- Represents the daily weekday operation hours of Jean's business -/
def weekday_hours : ℝ := sorry

/-- The number of weekdays the business operates -/
def weekdays : ℕ := 5

/-- The number of hours the business operates each day on weekends -/
def weekend_daily_hours : ℕ := 4

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The total weekly operation hours -/
def total_weekly_hours : ℕ := 38

/-- Theorem stating that the daily weekday operation hours are 6 -/
theorem weekday_hours_are_six : weekday_hours = 6 := by sorry

end NUMINAMATH_CALUDE_weekday_hours_are_six_l1773_177388


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_hundred_l1773_177395

theorem sum_of_cubes_equals_hundred : (1 : ℕ)^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_hundred_l1773_177395


namespace NUMINAMATH_CALUDE_ab_minus_three_l1773_177326

theorem ab_minus_three (a b : ℤ) (h : a - b = -2) : a - b - 3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_ab_minus_three_l1773_177326


namespace NUMINAMATH_CALUDE_three_of_a_kind_probability_l1773_177355

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in a hand -/
def HandSize : ℕ := 5

/-- Represents the number of ranks in a standard deck -/
def NumRanks : ℕ := 13

/-- Represents the number of cards of each rank in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Calculates the probability of drawing a "three of a kind" hand with two other cards of different ranks -/
def probThreeOfAKind : ℚ :=
  let totalHands := Nat.choose StandardDeck HandSize
  let threeOfAKindHands := NumRanks * Nat.choose CardsPerRank 3 * (NumRanks - 1) * CardsPerRank * (NumRanks - 2) * CardsPerRank
  threeOfAKindHands / totalHands

theorem three_of_a_kind_probability : probThreeOfAKind = 1719 / 40921 := by
  sorry

end NUMINAMATH_CALUDE_three_of_a_kind_probability_l1773_177355


namespace NUMINAMATH_CALUDE_repeating_decimal_properties_l1773_177331

/-- Represents a repeating decimal with a 3-digit non-repeating part and a 4-digit repeating part -/
structure RepeatingDecimal where
  N : ℕ  -- Non-repeating part (3 digits)
  M : ℕ  -- Repeating part (4 digits)

variable (R : RepeatingDecimal)

/-- The decimal expansion of R -/
noncomputable def decimal_expansion (R : RepeatingDecimal) : ℝ := sorry

theorem repeating_decimal_properties (R : RepeatingDecimal) :
  -- 1. R = 0.NMM... is a correct representation
  decimal_expansion R = (R.N : ℝ) / 1000 + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 2. 10^3R = N.MMM... is a correct representation
  1000 * decimal_expansion R = R.N + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 3. 10^7R ≠ NMN.MMM...
  10000000 * decimal_expansion R ≠ (R.N * 1000000 + R.M * 100 + R.N) + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 4. 10^3(10^4-1)R ≠ 10^4N - M
  1000 * (10000 - 1) * decimal_expansion R ≠ 10000 * R.N - R.M :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_properties_l1773_177331


namespace NUMINAMATH_CALUDE_solution_value_l1773_177354

theorem solution_value (a b : ℝ) (h : 2 * (-3) - a + 2 * b = 0) : 
  2 * a - 4 * b + 1 = -11 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l1773_177354


namespace NUMINAMATH_CALUDE_complex_number_in_quadrant_III_l1773_177397

def complex_number : ℂ := (-2 + Complex.I) * Complex.I^5

theorem complex_number_in_quadrant_III : 
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_complex_number_in_quadrant_III_l1773_177397


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1773_177396

theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, s > 0 ∧ x^2 / 4 + y^2 / 8 = 1 ∧ x = s ∧ y = s) →
  (4 * s^2 = 32 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1773_177396


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l1773_177302

theorem smallest_divisible_number : ∃ (n : ℕ), 
  (n > 2014) ∧ 
  (∀ k : ℕ, k < 10 → n % k = 0) ∧
  (∀ m : ℕ, m > 2014 ∧ m < n → ∃ j : ℕ, j < 10 ∧ m % j ≠ 0) ∧
  n = 2014506 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l1773_177302


namespace NUMINAMATH_CALUDE_smallest_multiple_eight_is_solution_eight_is_smallest_l1773_177317

theorem smallest_multiple (x : ℕ+) : (450 * x : ℕ) % 720 = 0 → x ≥ 8 := by
  sorry

theorem eight_is_solution : (450 * 8 : ℕ) % 720 = 0 := by
  sorry

theorem eight_is_smallest : ∀ (x : ℕ+), (450 * x : ℕ) % 720 = 0 → x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_eight_is_solution_eight_is_smallest_l1773_177317


namespace NUMINAMATH_CALUDE_probability_at_least_one_pen_l1773_177315

theorem probability_at_least_one_pen
  (p_ball : ℝ)
  (p_ink : ℝ)
  (h_ball : p_ball = 3 / 5)
  (h_ink : p_ink = 2 / 3)
  (h_nonneg_ball : 0 ≤ p_ball)
  (h_nonneg_ink : 0 ≤ p_ink)
  (h_le_one_ball : p_ball ≤ 1)
  (h_le_one_ink : p_ink ≤ 1)
  (h_independent : True)  -- Assumption of independence
  : p_ball + p_ink - p_ball * p_ink = 13 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_pen_l1773_177315


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l1773_177358

/-- Given a square field with area 3136 sq m, barbed wire drawn 3 m around it,
    two 1 m wide gates, and a total cost of 1332 Rs, prove that the rate of
    drawing barbed wire per meter is 6 Rs/m. -/
theorem barbed_wire_rate (area : ℝ) (wire_distance : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ)
    (h_area : area = 3136)
    (h_wire_distance : wire_distance = 3)
    (h_gate_width : gate_width = 1)
    (h_num_gates : num_gates = 2)
    (h_total_cost : total_cost = 1332) :
    total_cost / (4 * Real.sqrt area - num_gates * gate_width) = 6 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l1773_177358


namespace NUMINAMATH_CALUDE_truck_to_car_ratio_l1773_177375

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The total number of people needed to lift 6 cars and 3 trucks -/
def total_people : ℕ := 60

/-- The number of cars that can be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks that can be lifted -/
def num_trucks : ℕ := 3

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := (total_people - num_cars * people_per_car) / num_trucks

theorem truck_to_car_ratio :
  (people_per_truck : ℚ) / people_per_car = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_truck_to_car_ratio_l1773_177375


namespace NUMINAMATH_CALUDE_jackies_pushup_count_l1773_177332

/-- Calculates the number of push-ups Jackie can do in one minute with breaks -/
def jackies_pushups (pushups_per_ten_seconds : ℕ) (total_time : ℕ) (break_duration : ℕ) (num_breaks : ℕ) : ℕ :=
  let total_break_time := break_duration * num_breaks
  let pushup_time := total_time - total_break_time
  let pushups_per_second := pushups_per_ten_seconds / 10
  pushup_time * pushups_per_second

/-- Proves that Jackie can do 22 push-ups in one minute with two 8-second breaks -/
theorem jackies_pushup_count : jackies_pushups 5 60 8 2 = 22 := by
  sorry

#eval jackies_pushups 5 60 8 2

end NUMINAMATH_CALUDE_jackies_pushup_count_l1773_177332


namespace NUMINAMATH_CALUDE_brothers_age_difference_l1773_177392

theorem brothers_age_difference (michael_age younger_brother_age older_brother_age : ℕ) : 
  younger_brother_age = 5 →
  older_brother_age = 3 * younger_brother_age →
  michael_age + older_brother_age + younger_brother_age = 28 →
  older_brother_age - 2 * (michael_age - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l1773_177392


namespace NUMINAMATH_CALUDE_inequality_proof_l1773_177390

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1773_177390


namespace NUMINAMATH_CALUDE_regular_star_polygon_points_l1773_177342

/-- A regular star polygon with n points -/
structure RegularStarPolygon where
  n : ℕ
  edges : Fin (2 * n) → ℝ
  angles_A : Fin n → ℝ
  angles_B : Fin n → ℝ
  edges_equal : ∀ i j, edges i = edges j
  angles_A_equal : ∀ i j, angles_A i = angles_A j
  angles_B_equal : ∀ i j, angles_B i = angles_B j
  angle_difference : ∀ i, angles_B i - angles_A i = 15

/-- The theorem stating that for a regular star polygon with the given conditions, n must be 24 -/
theorem regular_star_polygon_points (star : RegularStarPolygon) :
  (∀ i, star.angles_B i - star.angles_A i = 15) → star.n = 24 :=
by sorry

end NUMINAMATH_CALUDE_regular_star_polygon_points_l1773_177342


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l1773_177376

structure IsoscelesTrapezoid where
  smaller_base : ℝ
  larger_base : ℝ
  diagonal : ℝ
  altitude : ℝ
  is_isosceles : True
  smaller_base_half_diagonal : smaller_base = diagonal / 2
  altitude_half_larger_base : altitude = larger_base / 2

theorem isosceles_trapezoid_base_ratio 
  (t : IsoscelesTrapezoid) : t.smaller_base / t.larger_base = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l1773_177376


namespace NUMINAMATH_CALUDE_triangle_perimeter_32_l1773_177373

/-- Given a triangle ABC with vertices A(-3, 5), B(3, -3), and M(6, 1) as the midpoint of BC,
    prove that the perimeter of the triangle is 32. -/
theorem triangle_perimeter_32 :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (3, -3)
  let M : ℝ × ℝ := (6, 1)
  let C : ℝ × ℝ := (2 * M.1 - B.1, 2 * M.2 - B.2)  -- Derived from midpoint formula
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB + BC + AC = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_32_l1773_177373


namespace NUMINAMATH_CALUDE_greatest_x_4a_value_l1773_177366

theorem greatest_x_4a_value : 
  ∀ (x a b c : ℕ), 
    (100 ≤ x) ∧ (x < 1000) →  -- x is a 3-digit integer
    (x = 100*a + 10*b + c) →  -- a, b, c are hundreds, tens, and units digits
    (4*a = 2*b) ∧ (2*b = c) → -- 4a = 2b = c
    (a > 0) →                 -- a > 0
    (∃ (x₁ x₂ : ℕ), (100 ≤ x₁) ∧ (x₁ < 1000) ∧ (100 ≤ x₂) ∧ (x₂ < 1000) ∧ 
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) → y ≤ x₁) ∧ 
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) ∧ (y ≠ x₁) → y ≤ x₂) ∧
      (x₁ - x₂ = 124)) →     -- difference between two greatest values is 124
    (∃ (a_max : ℕ), (100 ≤ 100*a_max + 10*(2*a_max) + 4*a_max) ∧ 
      (100*a_max + 10*(2*a_max) + 4*a_max < 1000) ∧
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) → y ≤ 100*a_max + 10*(2*a_max) + 4*a_max) ∧
      (4*a_max = 8)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_4a_value_l1773_177366


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1773_177344

theorem polynomial_divisibility (x₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : x₀^4 + a₁*x₀^3 + a₂*x₀^2 + a₃*x₀ + a₄ = 0)
  (h2 : 4*x₀^3 + 3*a₁*x₀^2 + 2*a₂*x₀ + a₃ = 0) :
  ∃ g : ℝ → ℝ, ∀ x : ℝ, 
    x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x - x₀)^2 * g x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1773_177344


namespace NUMINAMATH_CALUDE_modified_bowling_tournament_distributions_l1773_177391

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := 5

/-- Theorem: The number of different prize distributions in the modified bowling tournament -/
theorem modified_bowling_tournament_distributions :
  (outcomes_per_match ^ num_matches : ℕ) = 32 :=
sorry

end NUMINAMATH_CALUDE_modified_bowling_tournament_distributions_l1773_177391


namespace NUMINAMATH_CALUDE_floor_plus_self_equals_seventeen_fourths_l1773_177352

theorem floor_plus_self_equals_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equals_seventeen_fourths_l1773_177352


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1994_l1773_177367

theorem rightmost_three_digits_of_7_to_1994 : 7^1994 % 1000 = 49 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1994_l1773_177367


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1773_177301

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 10th term is 3 and the 12th term is 9,
    the 5th term is -12. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_10th : a 10 = 3)
  (h_12th : a 12 = 9) :
  a 5 = -12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1773_177301


namespace NUMINAMATH_CALUDE_sqrt_6_irrational_l1773_177360

/-- A number is rational if it can be expressed as a ratio of two integers -/
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- A number is irrational if it is not rational -/
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

/-- √6 is irrational -/
theorem sqrt_6_irrational : IsIrrational (Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_sqrt_6_irrational_l1773_177360


namespace NUMINAMATH_CALUDE_special_pentagon_exists_l1773_177325

/-- A pentagon that can be divided into three parts by one straight cut,
    such that two of the parts can be combined to form the third part. -/
structure SpecialPentagon where
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The cut line that divides the pentagon -/
  cut_line : ℝ × ℝ → ℝ × ℝ → Prop
  /-- The three parts resulting from the cut -/
  parts : Fin 3 → Set (ℝ × ℝ)
  /-- Proof that the cut line divides the pentagon into exactly three parts -/
  valid_division : sorry
  /-- Proof that two of the parts can be combined to form the third part -/
  recombination : sorry

/-- Theorem stating the existence of a special pentagon -/
theorem special_pentagon_exists : ∃ (p : SpecialPentagon), True := by
  sorry

end NUMINAMATH_CALUDE_special_pentagon_exists_l1773_177325


namespace NUMINAMATH_CALUDE_base4_arithmetic_l1773_177337

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplication operation for base 4 numbers --/
def mulBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a * base4ToBase10 b)

/-- Division operation for base 4 numbers --/
def divBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a / base4ToBase10 b)

theorem base4_arithmetic : 
  divBase4 (mulBase4 231 21) 3 = 213 := by sorry

end NUMINAMATH_CALUDE_base4_arithmetic_l1773_177337


namespace NUMINAMATH_CALUDE_negative_of_negative_is_positive_l1773_177349

theorem negative_of_negative_is_positive (x : ℝ) : x < 0 → -x > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_is_positive_l1773_177349


namespace NUMINAMATH_CALUDE_special_point_is_zero_l1773_177393

/-- Definition of the polynomial p(x,y) -/
def p (b : Fin 14 → ℝ) (x y : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

/-- The theorem stating that (5/19, 16/19) is a zero of all polynomials p satisfying the given conditions -/
theorem special_point_is_zero (b : Fin 14 → ℝ) : 
  (p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧ p b 0 1 = 0 ∧ 
   p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧ p b (-1) (-1) = 0 ∧ 
   p b 2 2 = 0 ∧ p b 2 (-2) = 0 ∧ p b (-2) 2 = 0) → 
  p b (5/19) (16/19) = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_point_is_zero_l1773_177393


namespace NUMINAMATH_CALUDE_sum_equals_200_l1773_177368

theorem sum_equals_200 : 139 + 27 + 23 + 11 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_200_l1773_177368


namespace NUMINAMATH_CALUDE_smallest_k_inequality_k_is_smallest_l1773_177361

theorem smallest_k_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x * y) ^ (1/3 : ℝ) + (3/8 : ℝ) * (x - y)^2 ≥ (3/8 : ℝ) * (x + y) :=
sorry

theorem k_is_smallest :
  ∀ k : ℝ, k > 0 → k < 3/8 →
  ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ (x * y) ^ (1/3 : ℝ) + k * (x - y)^2 < (3/8 : ℝ) * (x + y) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_k_is_smallest_l1773_177361


namespace NUMINAMATH_CALUDE_system_solution_l1773_177320

theorem system_solution (x y k : ℝ) : 
  4 * x + 3 * y = 1 → 
  k * x + (k - 1) * y = 3 → 
  x = y → 
  k = 11 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1773_177320


namespace NUMINAMATH_CALUDE_equation_solution_l1773_177363

theorem equation_solution : ∃ (x : ℝ), x > 0 ∧ x = Real.sqrt (x - 1/x) + Real.sqrt (1 - 1/x) ∧ x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1773_177363


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1773_177309

/-- A rhombus with side length 65 and shorter diagonal 60 has a longer diagonal of 110 -/
theorem rhombus_longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ)
  (h1 : side_length = 65)
  (h2 : shorter_diagonal = 60)
  (h3 : longer_diagonal * longer_diagonal / 4 + shorter_diagonal * shorter_diagonal / 4 = side_length * side_length) :
  longer_diagonal = 110 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1773_177309


namespace NUMINAMATH_CALUDE_b_range_l1773_177394

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- Theorem stating the range of b given the conditions -/
theorem b_range :
  ∀ b : ℝ,
  (∀ x : ℝ, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  (∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → f b x < f b y) →
  b ∈ Set.Icc 3 4 := by
sorry

end NUMINAMATH_CALUDE_b_range_l1773_177394


namespace NUMINAMATH_CALUDE_largest_green_socks_l1773_177329

theorem largest_green_socks (g y : ℕ) :
  let t := g + y
  (t ≤ 2023) →
  ((g * (g - 1) + y * (y - 1)) / (t * (t - 1)) = 1/3) →
  g ≤ 990 ∧ ∃ (g' y' : ℕ), g' = 990 ∧ y' + g' ≤ 2023 ∧
    ((g' * (g' - 1) + y' * (y' - 1)) / ((g' + y') * (g' + y' - 1)) = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_largest_green_socks_l1773_177329


namespace NUMINAMATH_CALUDE_distance_to_destination_l1773_177307

/-- Proves that the distance to a destination is 144 km given specific rowing conditions --/
theorem distance_to_destination (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : 
  rowing_speed = 10 →
  current_speed = 2 →
  total_time = 30 →
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  ∃ (distance : ℝ), 
    distance / downstream_speed + distance / upstream_speed = total_time ∧
    distance = 144 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_destination_l1773_177307


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l1773_177316

theorem common_factor_of_polynomial (m a b : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3*m*a^2 - 6*m*a*b = m * (k₁*a^2 + k₂*a*b) :=
sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l1773_177316


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l1773_177327

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 1 ∧ min = -17 := by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l1773_177327
