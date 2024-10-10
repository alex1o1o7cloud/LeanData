import Mathlib

namespace grid_31_counts_l3254_325485

/-- Represents a grid with n horizontal and vertical lines -/
structure Grid (n : ℕ) where
  horizontal_lines : ℕ
  vertical_lines : ℕ
  h_lines : horizontal_lines = n
  v_lines : vertical_lines = n

/-- Counts the number of rectangles in a grid -/
def count_rectangles (g : Grid n) : ℕ :=
  (n.choose 2) * (n.choose 2)

/-- Counts the number of squares in a grid with 1:2 distance ratio -/
def count_squares (g : Grid n) : ℕ :=
  let S (k : ℕ) := k * (k + 1) * (2 * k + 1) / 6
  S n - 2 * S (n / 2)

/-- The main theorem about the 31x31 grid -/
theorem grid_31_counts :
  ∃ (g : Grid 31),
    count_rectangles g = 216225 ∧
    count_squares g = 6975 :=
by sorry

end grid_31_counts_l3254_325485


namespace percentage_85_89_is_40_3_l3254_325495

/-- Represents the frequency distribution of test scores in a class -/
structure ScoreDistribution where
  range_90_100 : Nat
  range_85_89 : Nat
  range_75_84 : Nat
  range_65_74 : Nat
  range_below_65 : Nat

/-- Calculates the percentage of students in a specific score range -/
def percentageInRange (dist : ScoreDistribution) (rangeCount : Nat) : Rat :=
  let totalStudents := dist.range_90_100 + dist.range_85_89 + dist.range_75_84 + 
                       dist.range_65_74 + dist.range_below_65
  (rangeCount : Rat) / (totalStudents : Rat) * 100

/-- The main theorem stating that the percentage of students in the 85%-89% range is 40/3% -/
theorem percentage_85_89_is_40_3 (dist : ScoreDistribution) 
    (h : dist = ScoreDistribution.mk 6 4 7 10 3) : 
    percentageInRange dist dist.range_85_89 = 40 / 3 := by
  sorry

end percentage_85_89_is_40_3_l3254_325495


namespace compound_interest_rate_l3254_325468

/-- Given a principal amount, final amount, and time period, 
    calculate the compound interest rate. -/
theorem compound_interest_rate 
  (P : ℝ) (A : ℝ) (n : ℕ) 
  (h_P : P = 453.51473922902494)
  (h_A : A = 500)
  (h_n : n = 2) :
  ∃ r : ℝ, A = P * (1 + r)^n := by
sorry

end compound_interest_rate_l3254_325468


namespace double_base_exponent_problem_l3254_325443

theorem double_base_exponent_problem (a b x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (2*a)^(2*b) = (a^2)^b * x^b → x = 4 := by
  sorry

end double_base_exponent_problem_l3254_325443


namespace right_triangle_and_modular_inverse_l3254_325494

theorem right_triangle_and_modular_inverse : 
  (80^2 + 150^2 = 170^2) ∧ 
  (320 * 642 % 2879 = 1) := by sorry

end right_triangle_and_modular_inverse_l3254_325494


namespace geometric_sequence_formula_l3254_325421

/-- A geometric sequence with positive terms, a₁ = 1, and a₁ + a₂ + a₃ = 7 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (a 1 + a 2 + a 3 = 7) ∧
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n)

/-- The general formula for the geometric sequence -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end geometric_sequence_formula_l3254_325421


namespace quadratic_inequality_nonnegative_l3254_325416

theorem quadratic_inequality_nonnegative (x : ℝ) : x^2 - x + 1 ≥ 0 := by
  sorry

end quadratic_inequality_nonnegative_l3254_325416


namespace triangle_ABC_properties_l3254_325499

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.sin C →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 19 →
  ((C = π / 6) ∨ (C = 5 * π / 6)) ∧
  (∃ (S : ℝ), (S = (7 * Real.sqrt 3) / 2) ∨ (S = Real.sqrt 3 / 2)) :=
by sorry

end triangle_ABC_properties_l3254_325499


namespace turn_duration_is_one_hour_l3254_325435

/-- Represents the time taken to complete the work individually -/
structure WorkTime where
  a : ℝ
  b : ℝ

/-- Represents the amount of work done per hour -/
structure WorkRate where
  a : ℝ
  b : ℝ

/-- The duration of each turn when working alternately -/
def turn_duration (wt : WorkTime) (wr : WorkRate) : ℝ :=
  sorry

/-- The theorem stating that the turn duration is 1 hour -/
theorem turn_duration_is_one_hour (wt : WorkTime) (wr : WorkRate) :
  wt.a = 4 →
  wt.b = 12 →
  wr.a = 1 / wt.a →
  wr.b = 1 / wt.b →
  (3 * wr.a * turn_duration wt wr + 3 * wr.b * turn_duration wt wr = 1) →
  turn_duration wt wr = 1 :=
sorry

end turn_duration_is_one_hour_l3254_325435


namespace sum_of_repeating_decimals_l3254_325470

/-- Represents a repeating decimal where the digit repeats infinitely after the decimal point. -/
def repeating_decimal (d : ℕ) := (d : ℚ) / 9

/-- The sum of the repeating decimals 0.333... and 0.222... is equal to 5/9. -/
theorem sum_of_repeating_decimals : 
  repeating_decimal 3 + repeating_decimal 2 = 5 / 9 := by sorry

end sum_of_repeating_decimals_l3254_325470


namespace simplify_fraction_l3254_325475

theorem simplify_fraction : (5^4 + 5^2 + 5) / (5^3 - 2 * 5) = 27 + 14 / 23 := by
  sorry

end simplify_fraction_l3254_325475


namespace frog_mouse_jump_difference_l3254_325460

/-- Represents the jumping contest between a grasshopper, a frog, and a mouse -/
def jumping_contest (grasshopper_jump mouse_jump frog_jump : ℕ) : Prop :=
  grasshopper_jump = 14 ∧
  frog_jump = grasshopper_jump + 37 ∧
  mouse_jump = grasshopper_jump + 21

/-- Theorem stating the difference between the frog's and mouse's jump distances -/
theorem frog_mouse_jump_difference 
  (grasshopper_jump mouse_jump frog_jump : ℕ)
  (h : jumping_contest grasshopper_jump mouse_jump frog_jump) :
  frog_jump - mouse_jump = 16 := by
  sorry

#check frog_mouse_jump_difference

end frog_mouse_jump_difference_l3254_325460


namespace student_rank_l3254_325467

theorem student_rank (total : ℕ) (left_rank : ℕ) (right_rank : ℕ) :
  total = 31 → left_rank = 11 → right_rank = total - left_rank + 1 → right_rank = 21 := by
  sorry

end student_rank_l3254_325467


namespace abc_is_cube_l3254_325410

theorem abc_is_cube (a b c : ℤ) (h : (a / b) + (b / c) + (c / a) = 3) : 
  ∃ n : ℤ, a * b * c = n^3 := by
sorry

end abc_is_cube_l3254_325410


namespace rectangle_length_l3254_325414

theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2*l + 2*w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end rectangle_length_l3254_325414


namespace constant_c_value_l3254_325465

theorem constant_c_value (b c : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) →
  c = 7 := by
sorry

end constant_c_value_l3254_325465


namespace positive_real_inequality_l3254_325477

theorem positive_real_inequality (a : ℝ) (h : a > 0) : a + 1/a ≥ 2 := by
  sorry

end positive_real_inequality_l3254_325477


namespace triangle_square_side_ratio_l3254_325411

theorem triangle_square_side_ratio : 
  ∀ (triangle_side square_side : ℚ),
    triangle_side * 3 = 60 →
    square_side * 4 = 60 →
    triangle_side / square_side = 4 / 3 := by
sorry

end triangle_square_side_ratio_l3254_325411


namespace min_rings_to_connect_five_links_l3254_325423

/-- Represents a chain link with a specific number of rings -/
structure ChainLink where
  rings : ℕ

/-- Represents a collection of chain links -/
structure ChainCollection where
  links : List ChainLink

/-- The minimum number of rings needed to connect a chain collection into a single chain -/
def minRingsToConnect (c : ChainCollection) : ℕ :=
  sorry

/-- The problem statement -/
theorem min_rings_to_connect_five_links :
  let links := List.replicate 5 (ChainLink.mk 3)
  let chain := ChainCollection.mk links
  minRingsToConnect chain = 3 := by
  sorry

end min_rings_to_connect_five_links_l3254_325423


namespace walkers_commute_l3254_325425

/-- Ms. Walker's commute problem -/
theorem walkers_commute
  (speed_to_work : ℝ)
  (speed_from_work : ℝ)
  (total_time : ℝ)
  (h1 : speed_to_work = 60)
  (h2 : speed_from_work = 40)
  (h3 : total_time = 1) :
  ∃ (distance : ℝ), 
    distance / speed_to_work + distance / speed_from_work = total_time ∧ 
    distance = 24 := by
  sorry

end walkers_commute_l3254_325425


namespace binomial_expansion_coefficient_l3254_325402

theorem binomial_expansion_coefficient (x : ℝ) :
  let expansion := (1 + 2*x)^5
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ : ℝ,
    expansion = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 ∧
    a₃ = 80 := by
  sorry

end binomial_expansion_coefficient_l3254_325402


namespace g_sum_symmetric_l3254_325433

-- Define the function g
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^8 + q * x^6 - r * x^4 + 5

-- State the theorem
theorem g_sum_symmetric (p q r : ℝ) :
  g p q r 12 = 3 → g p q r 12 + g p q r (-12) = 6 := by
  sorry

end g_sum_symmetric_l3254_325433


namespace election_outcomes_l3254_325413

/-- The number of students participating in the election -/
def total_students : ℕ := 4

/-- The number of students eligible for the entertainment committee member role -/
def eligible_for_entertainment : ℕ := total_students - 1

/-- The number of positions available for each role -/
def positions_per_role : ℕ := 1

/-- Theorem: The number of ways to select a class monitor and an entertainment committee member
    from 4 students, where one specific student cannot be the entertainment committee member,
    is equal to 9. -/
theorem election_outcomes :
  (eligible_for_entertainment.choose positions_per_role) *
  (eligible_for_entertainment.choose positions_per_role) = 9 := by
  sorry

end election_outcomes_l3254_325413


namespace wood_length_after_sawing_l3254_325448

/-- The new length of a piece of wood after sawing off a portion. -/
def new_wood_length (original_length saw_off_length : ℝ) : ℝ :=
  original_length - saw_off_length

/-- Theorem stating that the new length of the wood is 6.6 cm. -/
theorem wood_length_after_sawing :
  new_wood_length 8.9 2.3 = 6.6 := by
  sorry

end wood_length_after_sawing_l3254_325448


namespace station_entry_problem_l3254_325401

/-- The number of ways for n people to enter through k gates, where each gate must have at least one person -/
def enterWays (n k : ℕ) : ℕ :=
  sorry

/-- The condition that the number of people is greater than the number of gates -/
def validInput (n k : ℕ) : Prop :=
  n > k ∧ k > 0

theorem station_entry_problem :
  ∀ n k : ℕ, validInput n k → (n = 5 ∧ k = 3) → enterWays n k = 720 :=
sorry

end station_entry_problem_l3254_325401


namespace truck_loading_time_l3254_325403

theorem truck_loading_time 
  (worker1_rate : ℝ) 
  (worker2_rate : ℝ) 
  (h1 : worker1_rate = 1 / 6) 
  (h2 : worker2_rate = 1 / 4) : 
  1 / (worker1_rate + worker2_rate) = 12 / 5 := by
sorry

end truck_loading_time_l3254_325403


namespace complex_number_equality_l3254_325484

theorem complex_number_equality (Z : ℂ) (h : Z * (1 - Complex.I) = 3 - Complex.I) : 
  Z = 2 + Complex.I := by
sorry

end complex_number_equality_l3254_325484


namespace condition_relationship_l3254_325441

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end condition_relationship_l3254_325441


namespace sum_of_divisors_119_l3254_325405

/-- The sum of all positive integer divisors of 119 is 144. -/
theorem sum_of_divisors_119 : (Finset.filter (· ∣ 119) (Finset.range 120)).sum id = 144 := by
  sorry

end sum_of_divisors_119_l3254_325405


namespace percentage_of_defective_meters_l3254_325445

theorem percentage_of_defective_meters (total_meters examined_meters : ℕ) 
  (h1 : total_meters = 120) (h2 : examined_meters = 12) :
  (examined_meters : ℝ) / total_meters * 100 = 10 := by
  sorry

end percentage_of_defective_meters_l3254_325445


namespace polynomial_factorization_l3254_325432

theorem polynomial_factorization (x : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x + 3) * (x + 14) := by
  sorry

end polynomial_factorization_l3254_325432


namespace notebook_puzzle_l3254_325417

/-- Represents a set of statements where the i-th statement claims 
    "There are exactly i false statements in this set" --/
def StatementSet (n : ℕ) := Fin n → Prop

/-- The property that exactly one statement in the set is true --/
def ExactlyOneTrue (s : StatementSet n) : Prop :=
  ∃! i, s i

/-- The i-th statement claims there are exactly i false statements --/
def StatementClaim (s : StatementSet n) (i : Fin n) : Prop :=
  s i ↔ (∃ k : Fin n, k.val = n - i.val ∧ (∀ j : Fin n, s j ↔ j = k))

/-- The main theorem --/
theorem notebook_puzzle :
  ∀ (s : StatementSet 100),
    (∀ i, StatementClaim s i) →
    ExactlyOneTrue s →
    s ⟨99, by norm_num⟩ :=
by sorry

end notebook_puzzle_l3254_325417


namespace circular_table_dice_probability_l3254_325483

/-- The number of people sitting around the circular table -/
def num_people : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability that no two adjacent people roll the same number -/
def no_adjacent_same_prob : ℚ := 441 / 8192

theorem circular_table_dice_probability :
  let n := num_people
  let s := die_sides
  (n : ℚ) > 0 ∧ (s : ℚ) > 0 →
  no_adjacent_same_prob = 441 / 8192 := by
  sorry

end circular_table_dice_probability_l3254_325483


namespace roundness_of_hundred_billion_l3254_325430

/-- Roundness of a positive integer is the sum of exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 100,000,000,000 is 22 -/
theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by sorry

end roundness_of_hundred_billion_l3254_325430


namespace gas_pressure_change_l3254_325427

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
def pressure_volume_relation (p1 p2 v1 v2 : ℝ) : Prop :=
  p1 * v1 = p2 * v2

/-- Theorem: Given inverse proportionality of pressure and volume,
    if a gas initially at 8 kPa in a 3.5-liter container is transferred to a 7-liter container,
    its new pressure will be 4 kPa -/
theorem gas_pressure_change (p2 : ℝ) :
  pressure_volume_relation 8 p2 3.5 7 → p2 = 4 := by
  sorry

end gas_pressure_change_l3254_325427


namespace original_denominator_proof_l3254_325464

theorem original_denominator_proof (d : ℤ) : 
  (2 : ℚ) / d ≠ 0 →
  (5 : ℚ) / (d + 3) = 1 / 3 →
  d = 12 := by
sorry

end original_denominator_proof_l3254_325464


namespace parallel_vectors_x_value_l3254_325488

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (-2, 4)
  let n : ℝ × ℝ := (x, -1)
  parallel m n → x = 1/2 :=
by
  sorry

end parallel_vectors_x_value_l3254_325488


namespace tangent_slope_minimum_tangent_slope_minimum_achieved_l3254_325459

theorem tangent_slope_minimum (b : ℝ) (h : b > 0) : 
  (2 / b + b) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem tangent_slope_minimum_achieved (b : ℝ) (h : b > 0) : 
  (2 / b + b = 2 * Real.sqrt 2) ↔ (2 / b = b) :=
by sorry

end tangent_slope_minimum_tangent_slope_minimum_achieved_l3254_325459


namespace product_equality_l3254_325452

theorem product_equality : 469111 * 9999 = 4690428889 := by
  sorry

end product_equality_l3254_325452


namespace sum_of_solutions_prove_sum_of_solutions_l3254_325497

theorem sum_of_solutions : ℕ → Prop :=
  fun s => ∃ (S : Finset ℕ), 
    (∀ x ∈ S, (5 * x + 2 > 3 * (x - 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)) ∧
    (∀ x : ℕ, (5 * x + 2 > 3 * (x - 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x) → x ∈ S) ∧
    (Finset.sum S id = s) ∧
    s = 10

theorem prove_sum_of_solutions : sum_of_solutions 10 := by
  sorry

end sum_of_solutions_prove_sum_of_solutions_l3254_325497


namespace irrational_sqrt_three_rational_others_l3254_325424

-- Define a rational number
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define an irrational number
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem irrational_sqrt_three_rational_others : 
  is_irrational (Real.sqrt 3) ∧ 
  is_rational (-2) ∧ 
  is_rational (1/2) ∧ 
  is_rational 2 :=
sorry

end irrational_sqrt_three_rational_others_l3254_325424


namespace students_taking_german_prove_students_taking_german_l3254_325404

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  let students_taking_at_least_one := total - neither
  let students_taking_only_french := french - both
  let students_taking_german := students_taking_at_least_one - students_taking_only_french
  students_taking_german

/-- Given a class of 69 students, where 41 are taking French, 9 are taking both French and German,
    and 15 are not taking either course, prove that 22 students are taking German. -/
theorem prove_students_taking_german :
  students_taking_german 69 41 9 15 = 22 := by
  sorry

end students_taking_german_prove_students_taking_german_l3254_325404


namespace constant_width_from_circle_sum_l3254_325442

/-- A curve in 2D space -/
structure Curve where
  -- Add necessary fields here
  convex : Bool

/-- Rotation of a curve by 180 degrees -/
def rotate180 (K : Curve) : Curve :=
  sorry

/-- Sum of two curves -/
def curveSum (K1 K2 : Curve) : Curve :=
  sorry

/-- Check if a curve is a circle -/
def isCircle (K : Curve) : Prop :=
  sorry

/-- Check if a curve has constant width -/
def hasConstantWidth (K : Curve) : Prop :=
  sorry

/-- Main theorem -/
theorem constant_width_from_circle_sum (K : Curve) (h : K.convex) :
  let K' := rotate180 K
  let K_star := curveSum K K'
  isCircle K_star → hasConstantWidth K :=
by
  sorry

end constant_width_from_circle_sum_l3254_325442


namespace smallest_four_digit_mod_9_4_l3254_325478

theorem smallest_four_digit_mod_9_4 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n % 9 = 4) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m % 9 = 4 → m ≥ n) ∧ 
  (n = 1003) := by
sorry

end smallest_four_digit_mod_9_4_l3254_325478


namespace students_speaking_neither_language_l3254_325496

theorem students_speaking_neither_language (total : ℕ) (english : ℕ) (telugu : ℕ) (both : ℕ) :
  total = 150 →
  english = 55 →
  telugu = 85 →
  both = 20 →
  total - (english + telugu - both) = 30 :=
by sorry

end students_speaking_neither_language_l3254_325496


namespace circular_fields_area_comparison_l3254_325463

theorem circular_fields_area_comparison (r₁ r₂ : ℝ) (h : r₂ = (5/2) * r₁) :
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 2.25 := by
  sorry

end circular_fields_area_comparison_l3254_325463


namespace coin_denomination_l3254_325492

/-- Given a total bill of 285 pesos, paid with 11 20-peso bills and 11 coins of unknown denomination,
    prove that the denomination of the coins must be 5 pesos. -/
theorem coin_denomination (total_bill : ℕ) (bill_value : ℕ) (num_bills : ℕ) (num_coins : ℕ) 
  (h1 : total_bill = 285)
  (h2 : bill_value = 20)
  (h3 : num_bills = 11)
  (h4 : num_coins = 11) :
  ∃ (coin_value : ℕ), coin_value = 5 ∧ total_bill = num_bills * bill_value + num_coins * coin_value :=
by
  sorry

#check coin_denomination

end coin_denomination_l3254_325492


namespace solution_count_decrease_l3254_325419

/-- The system of equations has fewer than four solutions if and only if a = ±1 or a = ±√2 -/
theorem solution_count_decrease (a : ℝ) : 
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    (x₁^2 - y₁^2 = 0 ∧ (x₁ - a)^2 + y₁^2 = 1) → 
    (x₂^2 - y₂^2 = 0 ∧ (x₂ - a)^2 + y₂^2 = 1) → 
    (x₃^2 - y₃^2 = 0 ∧ (x₃ - a)^2 + y₃^2 = 1) → 
    (x₄^2 - y₄^2 = 0 ∧ (x₄ - a)^2 + y₄^2 = 1) → 
    (x₁ = x₂ ∧ y₁ = y₂) ∨ (x₁ = x₃ ∧ y₁ = y₃) ∨ (x₁ = x₄ ∧ y₁ = y₄) ∨ 
    (x₂ = x₃ ∧ y₂ = y₃) ∨ (x₂ = x₄ ∧ y₂ = y₄) ∨ (x₃ = x₄ ∧ y₃ = y₄)) ↔ 
  a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end solution_count_decrease_l3254_325419


namespace base_ten_is_only_solution_l3254_325481

/-- Represents the number in base b as a function of n -/
def number (b n : ℕ) : ℚ :=
  (b^(3*n) - b^(2*n+1) + 7 * b^(2*n) + b^(n+1) - 7 * b^n - 1) / (3 * (b - 1))

/-- Predicate to check if a rational number is a perfect cube -/
def is_perfect_cube (q : ℚ) : Prop :=
  ∃ m : ℤ, q = (m : ℚ)^3

theorem base_ten_is_only_solution :
  ∀ b : ℕ, b ≥ 9 →
  (∀ n : ℕ, ∃ N : ℕ, ∀ m : ℕ, m ≥ N → is_perfect_cube (number b m)) →
  b = 10 :=
sorry

end base_ten_is_only_solution_l3254_325481


namespace trig_simplification_l3254_325455

theorem trig_simplification (α : Real) (h : π < α ∧ α < 3*π/2) :
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) + Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = -2 / Real.sin α :=
by sorry

end trig_simplification_l3254_325455


namespace lemonade_syrup_parts_l3254_325479

/-- Given a solution with water and lemonade syrup, prove the original amount of syrup --/
theorem lemonade_syrup_parts (x : ℝ) : 
  x > 0 → -- Ensure x is positive
  x / (x + 8) ≠ 1/5 → -- Ensure the original solution is not already 20% syrup
  x / (x + 8 - 2.1428571428571423 + 2.1428571428571423) = 1/5 → -- After replacement, solution is 20% syrup
  x = 2 := by
  sorry

end lemonade_syrup_parts_l3254_325479


namespace stating_count_initial_sets_eq_720_l3254_325420

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The length of each set of initials -/
def set_length : ℕ := 3

/-- 
Calculates the number of different three-letter sets of initials 
using letters A through J, where no letter can be used more than once in each set.
-/
def count_initial_sets : ℕ :=
  (num_letters) * (num_letters - 1) * (num_letters - 2)

/-- 
Theorem stating that the number of different three-letter sets of initials 
using letters A through J, where no letter can be used more than once in each set, 
is equal to 720.
-/
theorem count_initial_sets_eq_720 : count_initial_sets = 720 := by
  sorry

end stating_count_initial_sets_eq_720_l3254_325420


namespace males_in_band_only_l3254_325447

/-- Represents the number of students in various musical groups and their intersections --/
structure MusicGroups where
  band_male : ℕ
  band_female : ℕ
  orchestra_male : ℕ
  orchestra_female : ℕ
  choir_male : ℕ
  choir_female : ℕ
  band_orchestra_male : ℕ
  band_orchestra_female : ℕ
  band_choir_male : ℕ
  band_choir_female : ℕ
  orchestra_choir_male : ℕ
  orchestra_choir_female : ℕ
  total_students : ℕ

/-- Theorem stating the number of males in the band who are not in the orchestra or choir --/
theorem males_in_band_only (g : MusicGroups)
  (h1 : g.band_male = 120)
  (h2 : g.band_female = 100)
  (h3 : g.orchestra_male = 90)
  (h4 : g.orchestra_female = 130)
  (h5 : g.choir_male = 40)
  (h6 : g.choir_female = 60)
  (h7 : g.band_orchestra_male = 50)
  (h8 : g.band_orchestra_female = 70)
  (h9 : g.band_choir_male = 30)
  (h10 : g.band_choir_female = 40)
  (h11 : g.orchestra_choir_male = 20)
  (h12 : g.orchestra_choir_female = 30)
  (h13 : g.total_students = 260) :
  g.band_male - (g.band_orchestra_male + g.band_choir_male - 20) = 60 := by
  sorry

end males_in_band_only_l3254_325447


namespace right_triangle_hypotenuse_l3254_325415

/-- 
Given a right triangle PQR with legs PQ and PR, where U is on PQ and V is on PR,
prove that if PU:UQ = PV:VR = 1:3, QU = 18 units, and RV = 45 units, 
then the length of the hypotenuse QR is 12√29 units.
-/
theorem right_triangle_hypotenuse (P Q R U V : ℝ × ℝ) : 
  let pq := ‖Q - P‖
  let pr := ‖R - P‖
  let qu := ‖U - Q‖
  let rv := ‖V - R‖
  let qr := ‖R - Q‖
  (P.1 - Q.1) * (R.2 - P.2) = (P.2 - Q.2) * (R.1 - P.1) → -- right angle at P
  (∃ t : ℝ, t > 0 ∧ t < 1 ∧ U = t • P + (1 - t) • Q) → -- U is on PQ
  (∃ s : ℝ, s > 0 ∧ s < 1 ∧ V = s • P + (1 - s) • R) → -- V is on PR
  ‖P - U‖ / ‖U - Q‖ = 1 / 3 → -- PU:UQ = 1:3
  ‖P - V‖ / ‖V - R‖ = 1 / 3 → -- PV:VR = 1:3
  qu = 18 →
  rv = 45 →
  qr = 12 * Real.sqrt 29 := by
sorry

end right_triangle_hypotenuse_l3254_325415


namespace matrix_equation_proof_l3254_325474

theorem matrix_equation_proof : 
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -39/14, 51/14]
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  N * A = B := by sorry

end matrix_equation_proof_l3254_325474


namespace N_equals_negative_fifteen_l3254_325473

/-- A grid with arithmetic sequences in rows and columns -/
structure ArithmeticGrid where
  row_start : ℤ
  col1_second : ℤ
  col1_third : ℤ
  col2_last : ℤ

/-- The value N we're trying to determine -/
def N (grid : ArithmeticGrid) : ℤ :=
  grid.col2_last + (grid.col1_third - grid.col1_second)

/-- Theorem stating that N equals -15 for the given grid -/
theorem N_equals_negative_fifteen (grid : ArithmeticGrid) 
  (h1 : grid.row_start = 25)
  (h2 : grid.col1_second = 10)
  (h3 : grid.col1_third = 18)
  (h4 : grid.col2_last = -23) :
  N grid = -15 := by
  sorry


end N_equals_negative_fifteen_l3254_325473


namespace m_range_l3254_325428

-- Define the plane region
def plane_region (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |2 * p.1 + p.2 + m| < 3}

-- Define the theorem
theorem m_range (m : ℝ) :
  ((0, 0) ∈ plane_region m) ∧ ((-1, 1) ∈ plane_region m) →
  -2 < m ∧ m < 3 :=
by sorry

end m_range_l3254_325428


namespace complement_of_M_in_U_l3254_325409

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_in_U_l3254_325409


namespace parabola_intersection_difference_l3254_325444

/-- The difference between the x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 3
  let g (x : ℝ) := -2 * x^2 - 4 * x + 5
  let a := (1 - Real.sqrt 11) / 5
  let c := (1 + Real.sqrt 11) / 5
  f a = g a ∧ f c = g c ∧ c ≥ a →
  c - a = 2 * Real.sqrt 11 / 5 := by
sorry

end parabola_intersection_difference_l3254_325444


namespace quadratic_functions_intersect_l3254_325436

/-- A quadratic function of the form f(x) = x^2 + px + q where p + q = 2002 -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : p + q = 2002

/-- The theorem stating that all quadratic functions satisfying the condition
    intersect at the point (1, 2003) -/
theorem quadratic_functions_intersect (f : QuadraticFunction) :
  f.p + f.q^2 + f.p + f.q = 2003 := by
  sorry

#check quadratic_functions_intersect

end quadratic_functions_intersect_l3254_325436


namespace min_value_sum_of_reciprocals_l3254_325461

theorem min_value_sum_of_reciprocals (a b c d e f g : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) 
  (pos_e : 0 < e) (pos_f : 0 < f) (pos_g : 0 < g)
  (sum_eq_8 : a + b + c + d + e + f + g = 8) :
  1/a + 4/b + 9/c + 16/d + 25/e + 36/f + 49/g ≥ 98 ∧ 
  ∃ (a' b' c' d' e' f' g' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧ 0 < g' ∧
    a' + b' + c' + d' + e' + f' + g' = 8 ∧
    1/a' + 4/b' + 9/c' + 16/d' + 25/e' + 36/f' + 49/g' = 98 :=
by sorry

end min_value_sum_of_reciprocals_l3254_325461


namespace citric_acid_molecular_weight_l3254_325466

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in a Citric acid molecule -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in a Citric acid molecule -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in a Citric acid molecule -/
def oxygen_count : ℕ := 7

/-- The molecular weight of Citric acid in g/mol -/
def citric_acid_weight : ℝ := 192.124

theorem citric_acid_molecular_weight :
  (carbon_count : ℝ) * carbon_weight +
  (hydrogen_count : ℝ) * hydrogen_weight +
  (oxygen_count : ℝ) * oxygen_weight =
  citric_acid_weight := by sorry

end citric_acid_molecular_weight_l3254_325466


namespace min_units_for_nonnegative_profit_l3254_325407

/-- Represents the profit function for ice powder sales -/
def profit : ℕ → ℤ
| 0 => -120
| 10 => -80
| 20 => -40
| 30 => 0
| 40 => 40
| 50 => 80
| _ => 0  -- Default case, not used in the proof

/-- Theorem: The minimum number of units to be sold for non-negative profit is 30 -/
theorem min_units_for_nonnegative_profit :
  (∀ x : ℕ, x < 30 → profit x < 0) ∧
  profit 30 = 0 ∧
  (∀ x : ℕ, x > 30 → profit x > 0) :=
by sorry


end min_units_for_nonnegative_profit_l3254_325407


namespace largest_n_for_triangle_inequality_l3254_325462

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    such that ∠A + ∠C = 2∠B, the largest positive integer n 
    for which a^n + c^n ≤ 2b^n holds is 4. -/
theorem largest_n_for_triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π → 
  A + C = 2 * B → 
  ∃ (n : ℕ), n > 0 ∧ a^n + c^n ≤ 2*b^n ∧ 
  ∀ (m : ℕ), m > n → ¬(a^m + c^m ≤ 2*b^m) → 
  n = 4 := by
sorry

end largest_n_for_triangle_inequality_l3254_325462


namespace powers_of_four_unit_digits_l3254_325457

theorem powers_of_four_unit_digits (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x = (4^a.val) % 10 ∧ 
    y = (4^b.val) % 10 ∧ 
    z = (4^c.val) % 10) := by
  sorry

end powers_of_four_unit_digits_l3254_325457


namespace minimum_value_theorem_l3254_325490

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (a 7 = a 6 + 2 * a 5) →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ k l : ℕ, 1 / k + 9 / l ≥ 11 / 4) :=
by sorry

end minimum_value_theorem_l3254_325490


namespace hiking_rate_proof_l3254_325439

-- Define the hiking scenario
def hiking_scenario (rate : ℝ) : Prop :=
  let initial_distance : ℝ := 2.5
  let total_distance : ℝ := 3.5
  let total_time : ℝ := 45
  let return_distance : ℝ := total_distance - initial_distance
  
  -- The time to hike the additional distance east
  (return_distance / rate) +
  -- The time to hike back the additional distance
  (return_distance / rate) +
  -- The time to hike back the initial distance
  (initial_distance / rate) = total_time

-- Theorem statement
theorem hiking_rate_proof :
  ∃ (rate : ℝ), hiking_scenario rate ∧ rate = 1/10 :=
sorry

end hiking_rate_proof_l3254_325439


namespace arithmetic_expression_equality_l3254_325471

theorem arithmetic_expression_equality :
  4^2 * 10 + 5 * 12 + 12 * 4 + 24 / 3 * 9 = 340 := by
  sorry

end arithmetic_expression_equality_l3254_325471


namespace similar_triangle_longest_side_l3254_325493

/-- Given a triangle with sides 5, 12, and 13, and a similar triangle with perimeter 150,
    the longest side of the similar triangle is 65. -/
theorem similar_triangle_longest_side : ∀ (a b c : ℝ) (x : ℝ),
  a = 5 → b = 12 → c = 13 →
  a * x + b * x + c * x = 150 →
  max (a * x) (max (b * x) (c * x)) = 65 := by
sorry

end similar_triangle_longest_side_l3254_325493


namespace max_moves_in_grid_fourteen_fits_grid_fifteen_exceeds_grid_max_moves_is_fourteen_l3254_325437

theorem max_moves_in_grid (n : ℕ) : n > 0 → n * (n + 1) ≤ 200 → n ≤ 14 := by
  sorry

theorem fourteen_fits_grid : 14 * (14 + 1) ≤ 200 := by
  sorry

theorem fifteen_exceeds_grid : 15 * (15 + 1) > 200 := by
  sorry

theorem max_moves_is_fourteen : 
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) ≤ 200 ∧ ∀ (m : ℕ), m > n → m * (m + 1) > 200 := by
  sorry

end max_moves_in_grid_fourteen_fits_grid_fifteen_exceeds_grid_max_moves_is_fourteen_l3254_325437


namespace odd_power_sum_divisible_l3254_325487

/-- A number is odd if it can be expressed as 2k + 1 for some integer k -/
def IsOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- A number is positive if it's greater than zero -/
def IsPositive (n : ℕ) : Prop := n > 0

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, IsPositive n → IsOdd n →
  ∃ k : ℤ, x^n + y^n = (x + y) * k :=
sorry

end odd_power_sum_divisible_l3254_325487


namespace product_decrease_theorem_l3254_325482

theorem product_decrease_theorem :
  ∃ (a b c d e : ℕ), 
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) = 15 * (a * b * c * d * e) := by
  sorry

end product_decrease_theorem_l3254_325482


namespace factor_of_x4_plus_12_l3254_325458

theorem factor_of_x4_plus_12 (x : ℝ) : ∃ (y : ℝ), x^4 + 12 = (x^2 - 3*x + 3) * y := by
  sorry

end factor_of_x4_plus_12_l3254_325458


namespace equation_solution_l3254_325422

theorem equation_solution : 
  ∃! x : ℝ, (Real.sqrt (x + 20) - 4 / Real.sqrt (x + 20) = 7) ∧ 
  (x = (114 + 14 * Real.sqrt 65) / 4 - 20) := by
  sorry

end equation_solution_l3254_325422


namespace book_price_problem_l3254_325451

theorem book_price_problem (original_price : ℝ) : 
  original_price * (1 - 0.25) + original_price * (1 - 0.40) = 66 → 
  original_price = 48.89 := by
sorry

end book_price_problem_l3254_325451


namespace smallest_repeating_block_7_11_l3254_325454

/-- The length of the smallest repeating block in the decimal expansion of 7/11 -/
def repeating_block_length_7_11 : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 7 / 11

theorem smallest_repeating_block_7_11 :
  repeating_block_length_7_11 = 2 ∧
  ∃ (a b : ℕ), fraction = (a : ℚ) / (10^repeating_block_length_7_11 - 1 : ℚ) +
                (b : ℚ) / (10^repeating_block_length_7_11 : ℚ) ∧
                0 ≤ a ∧ a < 10^repeating_block_length_7_11 - 1 ∧
                0 ≤ b ∧ b < 10^repeating_block_length_7_11 ∧
                ∀ (n : ℕ), n < repeating_block_length_7_11 →
                  ¬∃ (c d : ℕ), fraction = (c : ℚ) / (10^n - 1 : ℚ) +
                                (d : ℚ) / (10^n : ℚ) ∧
                                0 ≤ c ∧ c < 10^n - 1 ∧
                                0 ≤ d ∧ d < 10^n := by
  sorry

#eval repeating_block_length_7_11

end smallest_repeating_block_7_11_l3254_325454


namespace wait_ratio_l3254_325406

def total_time : ℕ := 180
def uber_to_house : ℕ := 10
def check_bag : ℕ := 15
def wait_for_boarding : ℕ := 20

def uber_to_airport : ℕ := 5 * uber_to_house
def security : ℕ := 3 * check_bag

def time_before_takeoff : ℕ := 
  uber_to_house + uber_to_airport + check_bag + security + wait_for_boarding

def wait_before_takeoff : ℕ := total_time - time_before_takeoff

theorem wait_ratio : 
  wait_before_takeoff = 2 * wait_for_boarding :=
sorry

end wait_ratio_l3254_325406


namespace cannot_finish_third_l3254_325486

-- Define the set of runners
inductive Runner : Type
| A | B | C | D | E | F

-- Define the finish order relation
def finishes_before (x y : Runner) : Prop := sorry

-- Define the race conditions
axiom race_condition1 : finishes_before Runner.A Runner.B ∧ finishes_before Runner.A Runner.D
axiom race_condition2 : finishes_before Runner.B Runner.C ∧ finishes_before Runner.B Runner.F
axiom race_condition3 : finishes_before Runner.C Runner.D
axiom race_condition4 : finishes_before Runner.E Runner.F ∧ finishes_before Runner.A Runner.E

-- Define a function to represent the finishing position of a runner
def finishing_position (r : Runner) : ℕ := sorry

-- Define what it means to finish in third place
def finishes_third (r : Runner) : Prop := finishing_position r = 3

-- Theorem to prove
theorem cannot_finish_third : 
  ¬(finishes_third Runner.A) ∧ ¬(finishes_third Runner.F) := sorry

end cannot_finish_third_l3254_325486


namespace nathan_nickels_l3254_325426

theorem nathan_nickels (n : ℕ) : 
  20 < n ∧ n < 200 ∧ 
  n % 4 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 → 
  n = 142 := by sorry

end nathan_nickels_l3254_325426


namespace truthful_dwarfs_count_l3254_325476

theorem truthful_dwarfs_count :
  ∀ (total_dwarfs : ℕ) 
    (vanilla_hands chocolate_hands fruit_hands : ℕ),
  total_dwarfs = 10 →
  vanilla_hands = total_dwarfs →
  chocolate_hands = total_dwarfs / 2 →
  fruit_hands = 1 →
  ∃ (truthful_dwarfs : ℕ),
    truthful_dwarfs = 4 ∧
    truthful_dwarfs + (total_dwarfs - truthful_dwarfs) = total_dwarfs ∧
    vanilla_hands + chocolate_hands + fruit_hands = 
      total_dwarfs + (total_dwarfs - truthful_dwarfs) :=
by sorry

end truthful_dwarfs_count_l3254_325476


namespace frank_and_friends_count_l3254_325456

/-- The number of people, including Frank, who can eat brownies -/
def num_people (columns rows brownies_per_person : ℕ) : ℕ :=
  (columns * rows) / brownies_per_person

theorem frank_and_friends_count :
  num_people 6 3 3 = 6 := by
  sorry

end frank_and_friends_count_l3254_325456


namespace friendship_ratio_theorem_l3254_325418

/-- Represents a boy in the school -/
structure Boy where
  id : Nat

/-- Represents a girl in the school -/
structure Girl where
  id : Nat

/-- The number of girls who know a given boy -/
def d_Boy (b : Boy) : ℕ := sorry

/-- The number of boys who know a given girl -/
def d_Girl (g : Girl) : ℕ := sorry

/-- Represents that a boy and a girl know each other -/
def knows (b : Boy) (g : Girl) : Prop := sorry

theorem friendship_ratio_theorem 
  (n m : ℕ) 
  (boys : Finset Boy) 
  (girls : Finset Girl) 
  (h_boys : boys.card = n) 
  (h_girls : girls.card = m) 
  (h_girls_know_boy : ∀ g : Girl, ∃ b : Boy, knows b g) :
  ∃ (b : Boy) (g : Girl), 
    knows b g ∧ (d_Boy b : ℚ) / (d_Girl g : ℚ) ≥ (m : ℚ) / (n : ℚ) := by
  sorry

end friendship_ratio_theorem_l3254_325418


namespace people_made_happy_l3254_325438

/-- The number of institutions made happy -/
def institutions : ℕ := 6

/-- The number of people in each institution -/
def people_per_institution : ℕ := 80

/-- The total number of people made happy -/
def total_people_happy : ℕ := institutions * people_per_institution

theorem people_made_happy : total_people_happy = 480 := by
  sorry

end people_made_happy_l3254_325438


namespace system_solution_l3254_325446

theorem system_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧
    (5 * x + 4 * y = 6) ∧
    (x + 2 * y = 2) ∧
    (x = 2/3) ∧
    (y = 2/3) := by
  sorry

end system_solution_l3254_325446


namespace tangent_lines_equal_implies_a_equals_one_l3254_325491

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2
def g (a : ℝ) (x : ℝ) : ℝ := 3 * Real.log x - a * x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := 2 * x
def g' (a : ℝ) (x : ℝ) : ℝ := 3 / x - a

-- Theorem statement
theorem tangent_lines_equal_implies_a_equals_one :
  ∃ (x : ℝ), x > 0 ∧ f x = g 1 x ∧ f' x = g' 1 x :=
sorry

end

end tangent_lines_equal_implies_a_equals_one_l3254_325491


namespace union_of_complements_is_certain_l3254_325412

-- Define the sample space
variable {Ω : Type}

-- Define events as sets of outcomes
variable (A B C D : Set Ω)

-- Define the properties of events
variable (h1 : A ∩ B = ∅)  -- A and B are mutually exclusive
variable (h2 : C = Aᶜ)     -- C is the complement of A
variable (h3 : D = Bᶜ)     -- D is the complement of B

-- Theorem statement
theorem union_of_complements_is_certain : C ∪ D = univ := by
  sorry

end union_of_complements_is_certain_l3254_325412


namespace work_completion_l3254_325440

/-- Given that 8 men complete a work in 80 days, prove that 20 men will complete the same work in 32 days. -/
theorem work_completion (work : ℕ) : 
  (8 * 80 = work) → (20 * 32 = work) := by
  sorry

end work_completion_l3254_325440


namespace symmetric_points_ab_value_l3254_325431

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry across y-axis
def symmetricAcrossYAxis (p q : Point2D) : Prop :=
  p.x = -q.x ∧ p.y = q.y

-- Theorem statement
theorem symmetric_points_ab_value :
  ∀ (a b : ℝ),
  let p : Point2D := ⟨3, -1⟩
  let q : Point2D := ⟨a, 1 - b⟩
  symmetricAcrossYAxis p q →
  a^b = 9 := by
sorry

end symmetric_points_ab_value_l3254_325431


namespace solution_analysis_l3254_325469

def system_of_equations (a x y z : ℝ) : Prop :=
  a^3 * x + a * y + z = a^2 ∧
  x + y + z = 1 ∧
  8 * x + 2 * y + z = 4

theorem solution_analysis :
  (∀ x y z : ℝ, system_of_equations 2 x y z ↔ x = 1/5 ∧ y = 8/5 ∧ z = -2/5) ∧
  (∃ x₁ y₁ z₁ x₂ y₂ z₂ : ℝ, x₁ ≠ x₂ ∧ system_of_equations 1 x₁ y₁ z₁ ∧ system_of_equations 1 x₂ y₂ z₂) ∧
  (¬∃ x y z : ℝ, system_of_equations (-3) x y z) :=
sorry

end solution_analysis_l3254_325469


namespace price_increase_condition_l3254_325450

/-- Represents the fruit purchase and sale scenario -/
structure FruitSale where
  quantity : ℝ  -- Initial quantity in kg
  price : ℝ     -- Purchase price per kg
  loss : ℝ      -- Fraction of loss during transportation
  profit : ℝ    -- Desired minimum profit fraction
  increase : ℝ  -- Fraction of price increase

/-- Theorem stating the condition for the required price increase -/
theorem price_increase_condition (sale : FruitSale) 
  (h1 : sale.quantity = 200)
  (h2 : sale.price = 5)
  (h3 : sale.loss = 0.05)
  (h4 : sale.profit = 0.2) :
  (1 - sale.loss) * (1 + sale.increase) ≥ (1 + sale.profit) :=
sorry

end price_increase_condition_l3254_325450


namespace sum_of_h_at_x_values_l3254_325408

def f (x : ℝ) : ℝ := |x| - 3

def g (x : ℝ) : ℝ := -x

def h (x : ℝ) : ℝ := f (g (f x))

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_h_at_x_values :
  (x_values.map h).sum = -17 := by sorry

end sum_of_h_at_x_values_l3254_325408


namespace naoh_equals_nano3_l3254_325472

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction between NH4NO3 and NaOH to produce NaNO3 -/
structure Reaction where
  nh4no3_initial : Moles
  naoh_combined : Moles
  nano3_formed : Moles

/-- The reaction has a 1:1 molar ratio between NH4NO3 and NaOH to produce NaNO3 -/
axiom molar_ratio (r : Reaction) : r.nh4no3_initial = r.naoh_combined

/-- The number of moles of NH4NO3 initially present equals the number of moles of NaNO3 formed -/
axiom conservation (r : Reaction) : r.nh4no3_initial = r.nano3_formed

/-- The number of moles of NaOH combined equals the number of moles of NaNO3 formed -/
theorem naoh_equals_nano3 (r : Reaction) : r.naoh_combined = r.nano3_formed := by
  sorry

end naoh_equals_nano3_l3254_325472


namespace francie_allowance_problem_l3254_325429

/-- Represents the number of weeks Francie received her increased allowance -/
def weeks_of_increased_allowance : ℕ := 6

/-- Initial savings from the first 8 weeks -/
def initial_savings : ℕ := 5 * 8

/-- Increased allowance per week -/
def increased_allowance : ℕ := 6

/-- Total savings including both initial and increased allowance periods -/
def total_savings (x : ℕ) : ℕ := initial_savings + increased_allowance * x

theorem francie_allowance_problem :
  total_savings weeks_of_increased_allowance / 2 = 35 + 3 ∧
  weeks_of_increased_allowance * increased_allowance = total_savings weeks_of_increased_allowance - initial_savings :=
by sorry

end francie_allowance_problem_l3254_325429


namespace total_earnings_calculation_l3254_325453

theorem total_earnings_calculation 
  (x y : ℝ) 
  (h1 : 4 * x * (5 * y / 100) = 3 * x * (6 * y / 100) + 350) 
  (h2 : x * y = 17500) : 
  (3 * x * (6 * y / 100) + 4 * x * (5 * y / 100) + 5 * x * (4 * y / 100)) = 10150 := by
  sorry

end total_earnings_calculation_l3254_325453


namespace quadratic_root_difference_l3254_325489

theorem quadratic_root_difference : ∀ (x₁ x₂ : ℝ),
  (7 + 4 * Real.sqrt 3) * x₁^2 + (2 + Real.sqrt 3) * x₁ - 2 = 0 →
  (7 + 4 * Real.sqrt 3) * x₂^2 + (2 + Real.sqrt 3) * x₂ - 2 = 0 →
  x₁ ≠ x₂ →
  max x₁ x₂ - min x₁ x₂ = 6 - 3 * Real.sqrt 3 :=
by sorry

end quadratic_root_difference_l3254_325489


namespace polly_cooking_time_l3254_325480

/-- The number of minutes Polly spends cooking breakfast each day -/
def breakfast_time : ℕ := 20

/-- The number of minutes Polly spends cooking lunch each day -/
def lunch_time : ℕ := 5

/-- The number of minutes Polly spends cooking dinner on 4 days of the week -/
def dinner_time_short : ℕ := 10

/-- The number of minutes Polly spends cooking dinner on the other 3 days of the week -/
def dinner_time_long : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days Polly spends less time cooking dinner -/
def short_dinner_days : ℕ := 4

/-- The number of days Polly spends more time cooking dinner -/
def long_dinner_days : ℕ := days_in_week - short_dinner_days

/-- The total time Polly spends cooking in a week -/
def total_cooking_time : ℕ :=
  breakfast_time * days_in_week +
  lunch_time * days_in_week +
  dinner_time_short * short_dinner_days +
  dinner_time_long * long_dinner_days

/-- Theorem stating that Polly spends 305 minutes cooking in a week -/
theorem polly_cooking_time : total_cooking_time = 305 := by
  sorry

end polly_cooking_time_l3254_325480


namespace unique_nonnegative_solution_l3254_325434

theorem unique_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -4*x := by sorry

end unique_nonnegative_solution_l3254_325434


namespace handshake_frames_remaining_l3254_325498

theorem handshake_frames_remaining (d₁ d₂ : ℕ) 
  (h₁ : d₁ % 9 = 4) 
  (h₂ : d₂ % 9 = 6) : 
  (d₁ * d₂) % 9 = 6 := by
sorry

end handshake_frames_remaining_l3254_325498


namespace sequence_non_positive_l3254_325400

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k : ℕ, k ∈ Finset.range (n - 1) → a k - 2 * a (k + 1) + a (k + 2) ≥ 0) : 
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
sorry

end sequence_non_positive_l3254_325400


namespace derivative_of_f_l3254_325449

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_of_f (x : ℝ) :
  deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end derivative_of_f_l3254_325449
