import Mathlib

namespace NUMINAMATH_CALUDE_root_transformation_l1003_100310

theorem root_transformation {b : ℝ} (a b c d : ℝ) :
  (a^4 - b*a - 3 = 0) ∧
  (b^4 - b*b - 3 = 0) ∧
  (c^4 - b*c - 3 = 0) ∧
  (d^4 - b*d - 3 = 0) →
  (3*(-1/a)^4 - b*(-1/a)^3 - 1 = 0) ∧
  (3*(-1/b)^4 - b*(-1/b)^3 - 1 = 0) ∧
  (3*(-1/c)^4 - b*(-1/c)^3 - 1 = 0) ∧
  (3*(-1/d)^4 - b*(-1/d)^3 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1003_100310


namespace NUMINAMATH_CALUDE_exist_special_pair_l1003_100306

theorem exist_special_pair : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧ 
  (((a.val = 18 ∧ b.val = 1) ∨ (a.val = 1 ∧ b.val = 18))) :=
by sorry

end NUMINAMATH_CALUDE_exist_special_pair_l1003_100306


namespace NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_perpendicular_line_to_planes_are_parallel_l1003_100345

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem perpendicular_lines_to_plane_are_parallel (m n : Line3D) (β : Plane3D) :
  perpendicular m β → perpendicular n β → parallel_lines m n :=
sorry

theorem perpendicular_line_to_planes_are_parallel (m : Line3D) (α β : Plane3D) :
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_perpendicular_line_to_planes_are_parallel_l1003_100345


namespace NUMINAMATH_CALUDE_exists_collatz_greater_than_2012x_l1003_100348

-- Define the Collatz function
def collatz (x : ℕ) : ℕ :=
  if x % 2 = 1 then 3 * x + 1 else x / 2

-- Define the iterated Collatz function
def collatz_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => collatz_iter n (collatz x)

-- State the theorem
theorem exists_collatz_greater_than_2012x : ∃ x : ℕ, x > 0 ∧ collatz_iter 40 x > 2012 * x := by
  sorry

end NUMINAMATH_CALUDE_exists_collatz_greater_than_2012x_l1003_100348


namespace NUMINAMATH_CALUDE_calculation_result_l1003_100305

theorem calculation_result : 2002 * 20032003 - 2003 * 20022002 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l1003_100305


namespace NUMINAMATH_CALUDE_captain_age_is_27_l1003_100308

/-- Represents the age of the cricket team captain -/
def captain_age : ℕ := sorry

/-- Represents the age of the wicket keeper -/
def wicket_keeper_age : ℕ := sorry

/-- The number of players in the cricket team -/
def team_size : ℕ := 11

/-- The average age of the whole team -/
def team_average_age : ℕ := 24

theorem captain_age_is_27 :
  captain_age = 27 ∧
  wicket_keeper_age = captain_age + 3 ∧
  team_size * team_average_age = captain_age + wicket_keeper_age + (team_size - 2) * (team_average_age - 1) :=
by sorry

end NUMINAMATH_CALUDE_captain_age_is_27_l1003_100308


namespace NUMINAMATH_CALUDE_triangle_shape_l1003_100314

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that if 
    a/cos(A) = b/cos(B) = c/cos(C) and sin(A) = 2sin(B)cos(C), then A = B = C. -/
theorem triangle_shape (a b c A B C : ℝ) 
    (h1 : a / Real.cos A = b / Real.cos B) 
    (h2 : b / Real.cos B = c / Real.cos C)
    (h3 : Real.sin A = 2 * Real.sin B * Real.cos C)
    (h4 : 0 < A ∧ A < π)
    (h5 : 0 < B ∧ B < π)
    (h6 : 0 < C ∧ C < π)
    (h7 : A + B + C = π) : 
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1003_100314


namespace NUMINAMATH_CALUDE_steve_jellybeans_l1003_100328

/-- Given the following conditions:
    - Matilda has half as many jellybeans as Matt
    - Matt has ten times as many jellybeans as Steve
    - Matilda has 420 jellybeans
    Prove that Steve has 84 jellybeans -/
theorem steve_jellybeans (steve matt matilda : ℕ) 
  (h1 : matilda = matt / 2)
  (h2 : matt = 10 * steve)
  (h3 : matilda = 420) :
  steve = 84 := by
  sorry

end NUMINAMATH_CALUDE_steve_jellybeans_l1003_100328


namespace NUMINAMATH_CALUDE_preceding_binary_l1003_100386

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (binary : List Bool) : ℕ :=
  binary.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

theorem preceding_binary (N : ℕ) (h : binaryToNat [true, true, false, false, false] = N) :
  natToBinary (N - 1) = [true, false, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_preceding_binary_l1003_100386


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l1003_100369

theorem power_function_not_through_origin (n : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (n^2 - 3*n + 3) * x^(n^2 - n - 2) ≠ 0) →
  n = 1 ∨ n = 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l1003_100369


namespace NUMINAMATH_CALUDE_range_of_x_l1003_100395

def is_meaningful (x : ℝ) : Prop := x ≠ 5

theorem range_of_x : ∀ x : ℝ, is_meaningful x ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l1003_100395


namespace NUMINAMATH_CALUDE_linda_sales_l1003_100347

/-- Calculates the total amount of money made from selling necklaces and rings -/
def total_money_made (num_necklaces : ℕ) (num_rings : ℕ) (cost_per_necklace : ℕ) (cost_per_ring : ℕ) : ℕ :=
  num_necklaces * cost_per_necklace + num_rings * cost_per_ring

/-- Theorem: The total money made from selling 4 necklaces at $12 each and 8 rings at $4 each is $80 -/
theorem linda_sales : total_money_made 4 8 12 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_linda_sales_l1003_100347


namespace NUMINAMATH_CALUDE_essay_pages_theorem_l1003_100331

/-- Calculates the number of pages needed for a given number of words -/
def pages_needed (words : ℕ) : ℕ := (words + 259) / 260

/-- Represents the essay writing scenario -/
def essay_pages : Prop :=
  let johnny_words : ℕ := 150
  let madeline_words : ℕ := 2 * johnny_words
  let timothy_words : ℕ := madeline_words + 30
  let total_pages : ℕ := pages_needed johnny_words + pages_needed madeline_words + pages_needed timothy_words
  total_pages = 5

theorem essay_pages_theorem : essay_pages := by
  sorry

end NUMINAMATH_CALUDE_essay_pages_theorem_l1003_100331


namespace NUMINAMATH_CALUDE_vojta_sum_problem_l1003_100344

theorem vojta_sum_problem (S A B C : ℕ) : 
  S + 10 * B + C = 2224 →
  S + 10 * A + B = 2198 →
  S + 10 * A + C = 2204 →
  A < 10 →
  B < 10 →
  C < 10 →
  S + 100 * A + 10 * B + C = 2324 :=
by sorry

end NUMINAMATH_CALUDE_vojta_sum_problem_l1003_100344


namespace NUMINAMATH_CALUDE_count_distinct_tetrahedrons_is_423_l1003_100365

/-- Represents a regular tetrahedron with its vertices and edge midpoints -/
structure RegularTetrahedron :=
  (vertices : Finset (Fin 4))
  (edge_midpoints : Finset (Fin 6))

/-- Represents a new tetrahedron formed from points of a regular tetrahedron -/
def NewTetrahedron (t : RegularTetrahedron) := Finset (Fin 4)

/-- Counts the number of distinct new tetrahedrons that can be formed -/
def count_distinct_tetrahedrons (t : RegularTetrahedron) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct tetrahedrons is 423 -/
theorem count_distinct_tetrahedrons_is_423 (t : RegularTetrahedron) :
  count_distinct_tetrahedrons t = 423 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_tetrahedrons_is_423_l1003_100365


namespace NUMINAMATH_CALUDE_fiftieth_ring_l1003_100360

/-- Represents the number of squares in the nth ring -/
def S (n : ℕ) : ℕ := 10 * n - 2

/-- The properties of the sequence of rings -/
axiom first_ring : S 1 = 8
axiom second_ring : S 2 = 18
axiom ring_increase (n : ℕ) : n ≥ 2 → S (n + 1) - S n = 10

/-- The theorem stating the number of squares in the 50th ring -/
theorem fiftieth_ring : S 50 = 498 := by sorry

end NUMINAMATH_CALUDE_fiftieth_ring_l1003_100360


namespace NUMINAMATH_CALUDE_profit_margin_increase_l1003_100334

theorem profit_margin_increase (initial_margin : ℝ) (final_margin : ℝ) 
  (price_increase : ℝ) : 
  initial_margin = 0.25 →
  final_margin = 0.40 →
  price_increase = (1 + final_margin) / (1 + initial_margin) - 1 →
  price_increase = 0.12 := by
  sorry

#check profit_margin_increase

end NUMINAMATH_CALUDE_profit_margin_increase_l1003_100334


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1003_100370

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The total number of possible outcomes when flipping 5 coins -/
def total_outcomes : ℕ := 32

/-- Predicate for the desired outcome (penny, nickel, and dime are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧ cs.nickel = CoinOutcome.Heads ∧ cs.dime = CoinOutcome.Heads

/-- The number of outcomes satisfying the desired condition -/
def successful_outcomes : ℕ := 4

/-- The probability of the desired outcome -/
def probability : ℚ := 1 / 8

theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1003_100370


namespace NUMINAMATH_CALUDE_games_attended_l1003_100385

def total_games : ℕ := 39
def missed_games : ℕ := 25

theorem games_attended : total_games - missed_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_games_attended_l1003_100385


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1003_100375

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the scenarios
structure Scenario where
  totalPopulation : ℕ
  sampleSize : ℕ
  hasSubgroups : Bool
  isLargeScale : Bool

-- Define the function to determine the appropriate sampling method
def appropriateSamplingMethod (scenario : Scenario) : SamplingMethod :=
  if scenario.hasSubgroups then
    SamplingMethod.Stratified
  else if scenario.isLargeScale then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

-- Define the three scenarios
def scenario1 : Scenario := ⟨60, 8, false, false⟩
def scenario2 : Scenario := ⟨0, 0, false, true⟩  -- We don't know exact numbers, but it's large scale
def scenario3 : Scenario := ⟨130, 13, true, false⟩

-- State the theorem
theorem appropriate_sampling_methods :
  (appropriateSamplingMethod scenario1 = SamplingMethod.SimpleRandom) ∧
  (appropriateSamplingMethod scenario2 = SamplingMethod.Systematic) ∧
  (appropriateSamplingMethod scenario3 = SamplingMethod.Stratified) :=
by sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1003_100375


namespace NUMINAMATH_CALUDE_circle_areas_and_square_l1003_100397

/-- Given two concentric circles with radii 23 and 33 units, prove that a third circle
    with area equal to the shaded area between the two original circles has a radius of 4√35,
    and when inscribed in a square, the square's side length is 8√35. -/
theorem circle_areas_and_square (r₁ r₂ r₃ : ℝ) (s : ℝ) : 
  r₁ = 23 →
  r₂ = 33 →
  π * r₃^2 = π * (r₂^2 - r₁^2) →
  s = 2 * r₃ →
  r₃ = 4 * Real.sqrt 35 ∧ s = 8 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_and_square_l1003_100397


namespace NUMINAMATH_CALUDE_tim_vocabulary_proof_l1003_100330

/-- Proves that given the conditions of Tim's word learning, his original vocabulary was 14600 words --/
theorem tim_vocabulary_proof (words_per_day : ℕ) (learning_days : ℕ) (increase_percentage : ℚ) : 
  words_per_day = 10 →
  learning_days = 730 →
  increase_percentage = 1/2 →
  (words_per_day * learning_days : ℚ) = increase_percentage * (words_per_day * learning_days + 14600) :=
by
  sorry

end NUMINAMATH_CALUDE_tim_vocabulary_proof_l1003_100330


namespace NUMINAMATH_CALUDE_range_of_a_l1003_100396

-- Define the * operation
def star (x y : ℝ) := x * (1 - y)

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, star x (x - a) > 0 → -1 ≤ x ∧ x ≤ 1) → 
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1003_100396


namespace NUMINAMATH_CALUDE_find_M_l1003_100379

theorem find_M (x y z M : ℝ) : 
  x + y + z = 120 ∧ 
  x - 10 = M ∧ 
  y + 10 = M ∧ 
  z / 10 = M 
  → M = 10 := by
sorry

end NUMINAMATH_CALUDE_find_M_l1003_100379


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1003_100319

theorem negation_of_proposition :
  (¬ ∀ n : ℕ, n^2 ≤ 2*n + 5) ↔ (∃ n : ℕ, n^2 > 2*n + 5) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1003_100319


namespace NUMINAMATH_CALUDE_solve_equation_l1003_100383

theorem solve_equation (x : ℚ) (h : x - 3*x + 5*x = 200) : x = 200/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1003_100383


namespace NUMINAMATH_CALUDE_only_prime_alternating_base14_l1003_100378

/-- Represents a number in base 14 with alternating 1s and 0s -/
def alternating_base14 (n : ℕ) : ℕ :=
  (14^(2*n) - 1) / 195

/-- Checks if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, 1 < m → m < p → ¬(p % m = 0)

theorem only_prime_alternating_base14 :
  ∀ n : ℕ, is_prime (alternating_base14 n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_only_prime_alternating_base14_l1003_100378


namespace NUMINAMATH_CALUDE_ella_coin_value_l1003_100340

/-- Represents the number of coins Ella has -/
def total_coins : ℕ := 18

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the number of nickels Ella has -/
def nickels : ℕ := sorry

/-- Represents the number of dimes Ella has -/
def dimes : ℕ := sorry

/-- The total number of coins is the sum of nickels and dimes -/
axiom coin_sum : nickels + dimes = total_coins

/-- If Ella had two more dimes, she would have an equal number of nickels and dimes -/
axiom equal_with_two_more : nickels = dimes + 2

/-- The theorem to be proved -/
theorem ella_coin_value : 
  nickels * nickel_value + dimes * dime_value = 130 :=
sorry

end NUMINAMATH_CALUDE_ella_coin_value_l1003_100340


namespace NUMINAMATH_CALUDE_opposite_abs_difference_l1003_100346

theorem opposite_abs_difference (a : ℤ) : a = -3 → |a - 2| = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_abs_difference_l1003_100346


namespace NUMINAMATH_CALUDE_hockey_games_played_l1003_100302

theorem hockey_games_played (layla_goals : ℕ) (kristin_goals_difference : ℕ) (average_goals : ℕ) 
  (h1 : layla_goals = 104)
  (h2 : kristin_goals_difference = 24)
  (h3 : average_goals = 92)
  (h4 : layla_goals - kristin_goals_difference = average_goals * 2) :
  2 = (layla_goals + (layla_goals - kristin_goals_difference)) / average_goals := by
  sorry

end NUMINAMATH_CALUDE_hockey_games_played_l1003_100302


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l1003_100318

theorem max_subjects_per_teacher (maths_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h1 : maths_teachers = 11)
  (h2 : physics_teachers = 8)
  (h3 : chemistry_teachers = 5)
  (h4 : min_teachers = 8) :
  ∃ (max_subjects : ℕ), max_subjects = 3 ∧
    min_teachers * max_subjects ≥ maths_teachers + physics_teachers + chemistry_teachers ∧
    ∀ (x : ℕ), x > max_subjects → min_teachers * x > maths_teachers + physics_teachers + chemistry_teachers :=
by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l1003_100318


namespace NUMINAMATH_CALUDE_voice_of_china_sampling_l1003_100388

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) (n : ℕ) : ℕ :=
  firstSample + (populationSize / sampleSize) * (n - 1)

/-- The Voice of China sampling theorem -/
theorem voice_of_china_sampling :
  let populationSize := 500
  let sampleSize := 20
  let firstSample := 3
  let fifthSample := 5
  systematicSample populationSize sampleSize firstSample fifthSample = 103 := by
sorry

end NUMINAMATH_CALUDE_voice_of_china_sampling_l1003_100388


namespace NUMINAMATH_CALUDE_little_john_money_l1003_100309

/-- Calculates the remaining money after spending on sweets and giving to friends -/
def remaining_money (initial : ℚ) (spent_on_sweets : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) : ℚ :=
  initial - spent_on_sweets - (given_to_each_friend * num_friends)

/-- Theorem stating that given the specific amounts, the remaining money is $2.05 -/
theorem little_john_money : 
  remaining_money 5.10 1.05 1.00 2 = 2.05 := by
  sorry

#eval remaining_money 5.10 1.05 1.00 2

end NUMINAMATH_CALUDE_little_john_money_l1003_100309


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1003_100352

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population with two equal-sized subgroups -/
structure Population where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  h1 : group1 + group2 = total
  h2 : group1 = group2

/-- Represents a sample drawn from a population -/
structure Sample where
  size : ℕ
  population : Population
  method : SamplingMethod

/-- Predicate to determine if a sampling method is appropriate for comparing subgroups -/
def is_appropriate_for_subgroup_comparison (s : Sample) : Prop :=
  s.method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method
    for comparing characteristics between two equal-sized subgroups -/
theorem stratified_sampling_most_appropriate
  (pop : Population)
  (sample_size : ℕ)
  (h_sample_size : sample_size > 0 ∧ sample_size < pop.total) :
  ∀ (s : Sample),
    s.population = pop →
    s.size = sample_size →
    is_appropriate_for_subgroup_comparison s ↔ s.method = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1003_100352


namespace NUMINAMATH_CALUDE_mandy_med_school_acceptances_l1003_100359

theorem mandy_med_school_acceptances
  (total_researched : ℕ)
  (applied_fraction : ℚ)
  (accepted_fraction : ℚ)
  (h1 : total_researched = 42)
  (h2 : applied_fraction = 1 / 3)
  (h3 : accepted_fraction = 1 / 2)
  : ℕ :=
  by
    sorry

#check mandy_med_school_acceptances

end NUMINAMATH_CALUDE_mandy_med_school_acceptances_l1003_100359


namespace NUMINAMATH_CALUDE_square_root_equation_l1003_100362

theorem square_root_equation (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1003_100362


namespace NUMINAMATH_CALUDE_binomial_10_choose_6_l1003_100342

theorem binomial_10_choose_6 : Nat.choose 10 6 = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_6_l1003_100342


namespace NUMINAMATH_CALUDE_family_c_members_l1003_100368

/-- Represents the number of members in each family in Indira Nagar --/
structure FamilyMembers where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat

/-- The initial number of family members before some left for the hostel --/
def initial_members : FamilyMembers := {
  a := 7,
  b := 8,
  c := 10,  -- This is what we want to prove
  d := 13,
  e := 6,
  f := 10
}

/-- The number of family members after one member from each family left for the hostel --/
def members_after_hostel (fm : FamilyMembers) : FamilyMembers :=
  { a := fm.a - 1,
    b := fm.b - 1,
    c := fm.c - 1,
    d := fm.d - 1,
    e := fm.e - 1,
    f := fm.f - 1 }

/-- The total number of families --/
def num_families : Nat := 6

/-- Theorem stating that the initial number of members in family c was 10 --/
theorem family_c_members :
  (members_after_hostel initial_members).a +
  (members_after_hostel initial_members).b +
  (members_after_hostel initial_members).c +
  (members_after_hostel initial_members).d +
  (members_after_hostel initial_members).e +
  (members_after_hostel initial_members).f =
  8 * num_families :=
by sorry

end NUMINAMATH_CALUDE_family_c_members_l1003_100368


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1003_100399

theorem consecutive_integers_sum (x : ℤ) :
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 75 →
  (x - 2) + (x + 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1003_100399


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l1003_100394

/-- A convex polygon with area, perimeter, and inscribed circle radius -/
structure ConvexPolygon where
  area : ℝ
  perimeter : ℝ
  inscribed_radius : ℝ
  area_pos : 0 < area
  perimeter_pos : 0 < perimeter
  inscribed_radius_pos : 0 < inscribed_radius

/-- The theorem stating that for any convex polygon, the ratio of its area to its perimeter
    is less than or equal to the radius of its inscribed circle -/
theorem inscribed_circle_radius_bound (poly : ConvexPolygon) :
  poly.area / poly.perimeter ≤ poly.inscribed_radius :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l1003_100394


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1003_100312

def is_geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = α n * r

theorem geometric_sequence_property (α : ℕ → ℝ) (h_geo : is_geometric_sequence α) 
  (h_prod : α 4 * α 5 * α 6 = 27) : α 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1003_100312


namespace NUMINAMATH_CALUDE_average_of_sqrt_equation_l1003_100329

theorem average_of_sqrt_equation (x : ℝ) :
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, Real.sqrt (3 * x^2 + 4 * x + 1) = Real.sqrt 37 ↔ x = x₁ ∨ x = x₂) ∧
  (x₁ + x₂) / 2 = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_average_of_sqrt_equation_l1003_100329


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l1003_100364

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -2 ∧ 
  (∀ (y : ℤ), y < 0 → 50 * y + 14 ≡ 10 [ZMOD 24] → y ≤ x) ∧
  50 * x + 14 ≡ 10 [ZMOD 24] := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l1003_100364


namespace NUMINAMATH_CALUDE_f_at_2_l1003_100376

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem f_at_2 : f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l1003_100376


namespace NUMINAMATH_CALUDE_class_size_from_marking_error_l1003_100343

/-- The number of pupils in a class where a marking error occurred -/
def number_of_pupils : ℕ := 16

/-- The incorrect mark entered for a pupil -/
def incorrect_mark : ℕ := 73

/-- The correct mark for the pupil -/
def correct_mark : ℕ := 65

/-- The increase in class average due to the error -/
def average_increase : ℚ := 1/2

theorem class_size_from_marking_error :
  (incorrect_mark - correct_mark : ℚ) = number_of_pupils * average_increase :=
sorry

end NUMINAMATH_CALUDE_class_size_from_marking_error_l1003_100343


namespace NUMINAMATH_CALUDE_square_area_calculation_l1003_100356

theorem square_area_calculation (s : ℝ) (r : ℝ) (l : ℝ) (b : ℝ) : 
  r = s →                -- radius of circle equals side of square
  l = (1 / 6) * r →      -- length of rectangle is one-sixth of circle radius
  l * b = 360 →          -- area of rectangle is 360 sq. units
  b = 10 →               -- breadth of rectangle is 10 units
  s^2 = 46656 :=         -- area of square is 46656 sq. units
by
  sorry

end NUMINAMATH_CALUDE_square_area_calculation_l1003_100356


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1003_100374

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 18) * (x / 162) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1003_100374


namespace NUMINAMATH_CALUDE_cans_left_to_load_l1003_100327

/-- Given a packing scenario for canned juice, prove the number of cans left to be loaded. -/
theorem cans_left_to_load 
  (cans_per_carton : ℕ) 
  (total_cartons : ℕ) 
  (loaded_cartons : ℕ) 
  (h1 : cans_per_carton = 20)
  (h2 : total_cartons = 50)
  (h3 : loaded_cartons = 40) :
  (total_cartons - loaded_cartons) * cans_per_carton = 200 :=
by sorry

end NUMINAMATH_CALUDE_cans_left_to_load_l1003_100327


namespace NUMINAMATH_CALUDE_aubriella_poured_gallons_l1003_100323

/-- Proves that Aubriella has poured 18 gallons into the fish tank -/
theorem aubriella_poured_gallons
  (tank_capacity : ℕ)
  (remaining_gallons : ℕ)
  (seconds_per_gallon : ℕ)
  (pouring_time_minutes : ℕ)
  (h1 : tank_capacity = 50)
  (h2 : remaining_gallons = 32)
  (h3 : seconds_per_gallon = 20)
  (h4 : pouring_time_minutes = 6) :
  tank_capacity - remaining_gallons = 18 :=
by sorry

end NUMINAMATH_CALUDE_aubriella_poured_gallons_l1003_100323


namespace NUMINAMATH_CALUDE_mary_sugar_addition_l1003_100361

/-- The amount of sugar Mary needs to add to her cake mix -/
def sugar_to_add (required_sugar : ℕ) (added_sugar : ℕ) : ℕ :=
  required_sugar - added_sugar

theorem mary_sugar_addition : sugar_to_add 11 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_addition_l1003_100361


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l1003_100338

def product_20_to_30 : ℕ := 20 * 21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29 * 30

theorem units_digit_of_fraction (h : product_20_to_30 % 8000 = 6) :
  (product_20_to_30 / 8000) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l1003_100338


namespace NUMINAMATH_CALUDE_unique_solution_iff_k_eq_neg_three_fourths_l1003_100320

/-- The equation (x + 3) / (kx - 2) = x has exactly one solution if and only if k = -3/4 -/
theorem unique_solution_iff_k_eq_neg_three_fourths (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_k_eq_neg_three_fourths_l1003_100320


namespace NUMINAMATH_CALUDE_darma_peanut_eating_l1003_100390

/-- Darma's peanut eating rate -/
def peanuts_per_15_seconds : ℕ := 20

/-- Convert minutes to seconds -/
def minutes_to_seconds (minutes : ℕ) : ℕ := minutes * 60

/-- Calculate peanuts eaten in a given time -/
def peanuts_eaten (seconds : ℕ) : ℕ :=
  (seconds / 15) * peanuts_per_15_seconds

theorem darma_peanut_eating (minutes : ℕ) (h : minutes = 6) :
  peanuts_eaten (minutes_to_seconds minutes) = 480 := by
  sorry

end NUMINAMATH_CALUDE_darma_peanut_eating_l1003_100390


namespace NUMINAMATH_CALUDE_elvis_recording_time_l1003_100321

theorem elvis_recording_time (total_songs : ℕ) (studio_hours : ℕ) (writing_time_per_song : ℕ) (total_editing_time : ℕ) :
  total_songs = 10 →
  studio_hours = 5 →
  writing_time_per_song = 15 →
  total_editing_time = 30 →
  (studio_hours * 60 - total_songs * writing_time_per_song - total_editing_time) / total_songs = 12 := by
  sorry

end NUMINAMATH_CALUDE_elvis_recording_time_l1003_100321


namespace NUMINAMATH_CALUDE_triangle_ABC_dot_product_l1003_100339

def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (5, 6)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_ABC_dot_product :
  dot_product vector_AB vector_AC = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_dot_product_l1003_100339


namespace NUMINAMATH_CALUDE_sale_markdown_l1003_100303

theorem sale_markdown (regular_price sale_price : ℝ) 
  (h : sale_price * (1 + 0.25) = regular_price) :
  (regular_price - sale_price) / regular_price = 0.2 := by
sorry

end NUMINAMATH_CALUDE_sale_markdown_l1003_100303


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l1003_100307

theorem quadratic_expression_equality : ∃ (a b c : ℝ), 
  (∀ x, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ 
  (10 * a - b - 4 * c = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l1003_100307


namespace NUMINAMATH_CALUDE_multiples_count_multiples_count_equals_188_l1003_100315

theorem multiples_count : ℕ :=
  let range_start := 1
  let range_end := 600
  let count_multiples_of (n : ℕ) := (range_end / n : ℕ)
  let multiples_of_5 := count_multiples_of 5
  let multiples_of_7 := count_multiples_of 7
  let multiples_of_35 := count_multiples_of 35
  multiples_of_5 + multiples_of_7 - multiples_of_35

theorem multiples_count_equals_188 : multiples_count = 188 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_multiples_count_equals_188_l1003_100315


namespace NUMINAMATH_CALUDE_sunscreen_discount_percentage_l1003_100384

/-- Calculate the discount percentage for Juanita's sunscreen purchase -/
theorem sunscreen_discount_percentage : 
  let bottles_per_year : ℕ := 12
  let cost_per_bottle : ℚ := 30
  let discounted_total_cost : ℚ := 252
  let original_total_cost : ℚ := bottles_per_year * cost_per_bottle
  let discount_amount : ℚ := original_total_cost - discounted_total_cost
  let discount_percentage : ℚ := (discount_amount / original_total_cost) * 100
  discount_percentage = 30 := by sorry

end NUMINAMATH_CALUDE_sunscreen_discount_percentage_l1003_100384


namespace NUMINAMATH_CALUDE_jan1_2010_is_sunday_l1003_100337

-- Define days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem jan1_2010_is_sunday :
  advanceDay DayOfWeek.Saturday 3653 = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_jan1_2010_is_sunday_l1003_100337


namespace NUMINAMATH_CALUDE_craft_supplies_ratio_l1003_100349

/-- Represents the craft supplies bought by a person -/
structure CraftSupplies :=
  (glueSticks : ℕ)
  (constructionPaper : ℕ)

/-- The ratio of two natural numbers -/
def ratio (a b : ℕ) : ℚ := a / b

theorem craft_supplies_ratio :
  ∀ (allison marie : CraftSupplies),
    allison.glueSticks = marie.glueSticks + 8 →
    marie.glueSticks = 15 →
    marie.constructionPaper = 30 →
    allison.glueSticks + allison.constructionPaper = 28 →
    ratio marie.constructionPaper allison.constructionPaper = 6 := by
  sorry

end NUMINAMATH_CALUDE_craft_supplies_ratio_l1003_100349


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l1003_100377

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l1003_100377


namespace NUMINAMATH_CALUDE_divisibility_statement_l1003_100353

theorem divisibility_statement (a : ℤ) :
  (∃! n : Fin 4, ¬ (
    (n = 0 → a % 2 = 0) ∧
    (n = 1 → a % 4 = 0) ∧
    (n = 2 → a % 12 = 0) ∧
    (n = 3 → a % 24 = 0)
  )) →
  ¬(a % 24 = 0) :=
by sorry


end NUMINAMATH_CALUDE_divisibility_statement_l1003_100353


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1003_100324

/-- Given a circle and a moving chord, prove the trajectory of the chord's midpoint -/
theorem midpoint_trajectory (x y : ℝ) :
  (∀ a b : ℝ, a^2 + b^2 = 25 → (x - a)^2 + (y - b)^2 ≤ 9) →
  x^2 + y^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1003_100324


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2023_l1003_100326

theorem sum_of_last_two_digits_of_9_pow_2023 : ∃ (a b : ℕ), 
  (9^2023 : ℕ) % 100 = 10 * a + b ∧ a + b = 11 := by sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2023_l1003_100326


namespace NUMINAMATH_CALUDE_flour_to_add_correct_l1003_100350

/-- Represents the recipe and baking constraints -/
structure BakingProblem where
  total_flour : ℝ  -- Total flour required by the recipe
  total_sugar : ℝ  -- Total sugar required by the recipe
  flour_sugar_diff : ℝ  -- Difference between remaining flour and sugar to be added

/-- Calculates the amount of flour that needs to be added -/
def flour_to_add (problem : BakingProblem) : ℝ :=
  problem.total_flour

/-- Theorem stating that the amount of flour to add is correct -/
theorem flour_to_add_correct (problem : BakingProblem) 
  (h1 : problem.total_flour = 6)
  (h2 : problem.total_sugar = 13)
  (h3 : problem.flour_sugar_diff = 8) :
  flour_to_add problem = 6 ∧ 
  flour_to_add problem = problem.total_sugar - problem.flour_sugar_diff + problem.flour_sugar_diff := by
  sorry

#eval flour_to_add { total_flour := 6, total_sugar := 13, flour_sugar_diff := 8 }

end NUMINAMATH_CALUDE_flour_to_add_correct_l1003_100350


namespace NUMINAMATH_CALUDE_distribute_6_balls_3_boxes_l1003_100398

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributeIndistinguishable (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_6_balls_3_boxes : distributeIndistinguishable 6 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_balls_3_boxes_l1003_100398


namespace NUMINAMATH_CALUDE_proposition_b_correct_l1003_100332

theorem proposition_b_correct :
  (∃ x : ℕ, x^3 ≤ x^2) ∧
  ((∀ x : ℝ, x > 1 → x^2 > 1) ∧ (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1)) ∧
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_b_correct_l1003_100332


namespace NUMINAMATH_CALUDE_four_color_plane_exists_l1003_100341

-- Define the color type
inductive Color
| Red | Blue | Green | Yellow | Purple

-- Define the space as a type of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the condition that each color appears at least once
axiom all_colors_present : ∀ c : Color, ∃ p : Point, coloring p = c

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if a point is on a plane
def on_plane (plane : Plane) (point : Point) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Define a function to count distinct colors on a plane
def count_colors_on_plane (plane : Plane) : ℕ := sorry

-- The main theorem
theorem four_color_plane_exists :
  ∃ plane : Plane, count_colors_on_plane plane ≥ 4 := sorry

end NUMINAMATH_CALUDE_four_color_plane_exists_l1003_100341


namespace NUMINAMATH_CALUDE_absolute_value_inequalities_l1003_100333

theorem absolute_value_inequalities (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| > a → a < 3) ∧
  (∀ x : ℝ, |x - 1| - |x + 3| < a → a > 4) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequalities_l1003_100333


namespace NUMINAMATH_CALUDE_f_properties_l1003_100372

def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x, c = 0 → f b c (-x) = -f b c x) ∧
  (∀ x y, b = 0 → x < y → f b c x < f b c y) ∧
  (∀ x, f b c x - c = -(f b c (-x) - c)) ∧
  ¬(∀ b c, ∃ x y, f b c x = 0 ∧ f b c y = 0 ∧ ∀ z, f b c z = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1003_100372


namespace NUMINAMATH_CALUDE_greatest_power_of_two_dividing_expression_l1003_100371

theorem greatest_power_of_two_dividing_expression : ∃ k : ℕ, 
  (k = 1007 ∧ 
   2^k ∣ (10^1004 - 4^502) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) := by
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_dividing_expression_l1003_100371


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1003_100391

/-- The number of terms in an arithmetic sequence from -3 to 53 -/
theorem arithmetic_sequence_length : ∀ (a d : ℤ), 
  a = -3 → 
  d = 4 → 
  ∃ n : ℕ, n > 0 ∧ a + (n - 1) * d = 53 → 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1003_100391


namespace NUMINAMATH_CALUDE_negation_equivalence_l1003_100358

def original_statement (a b : ℤ) : Prop :=
  (¬(Odd a ∧ Odd b)) → Even (a + b)

def proposed_negation (a b : ℤ) : Prop :=
  (¬(Odd a ∧ Odd b)) → ¬Even (a + b)

def correct_negation (a b : ℤ) : Prop :=
  ¬(Odd a ∧ Odd b) ∧ ¬Even (a + b)

theorem negation_equivalence :
  ∀ a b : ℤ, ¬(original_statement a b) ↔ correct_negation a b :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1003_100358


namespace NUMINAMATH_CALUDE_system_two_solutions_l1003_100366

/-- The system of equations has exactly two solutions if and only if a ∈ {49, 289} -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y z w : ℝ, 
    (abs (y + x + 8) + abs (y - x + 8) = 16 ∧
     (abs x - 15)^2 + (abs y - 8)^2 = a) ∧
    (abs (z + w + 8) + abs (z - w + 8) = 16 ∧
     (abs w - 15)^2 + (abs z - 8)^2 = a) ∧
    (x ≠ w ∨ y ≠ z)) ↔ 
  (a = 49 ∨ a = 289) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l1003_100366


namespace NUMINAMATH_CALUDE_max_a_for_inequality_l1003_100381

theorem max_a_for_inequality : 
  (∃ (a_max : ℝ), 
    (∀ (a : ℝ), (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a * Real.cos x ≥ 0) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a_max * Real.cos x ≥ 0)) ∧
  (∀ (a_max : ℝ), 
    ((∀ (a : ℝ), (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a * Real.cos x ≥ 0) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a_max * Real.cos x ≥ 0)) → 
    a_max = 1/3) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_l1003_100381


namespace NUMINAMATH_CALUDE_sebastians_age_l1003_100367

theorem sebastians_age (sebastian_age sister_age father_age : ℕ) : 
  (sebastian_age - 5) + (sister_age - 5) = 3 * (father_age - 5) / 4 →
  sebastian_age = sister_age + 10 →
  father_age = 85 →
  sebastian_age = 40 := by
sorry

end NUMINAMATH_CALUDE_sebastians_age_l1003_100367


namespace NUMINAMATH_CALUDE_max_plates_on_table_l1003_100300

/-- The radius of the table in meters -/
def table_radius : ℝ := 1

/-- The radius of each plate in meters -/
def plate_radius : ℝ := 0.15

/-- The maximum number of plates that can fit on the table -/
def max_plates : ℕ := 44

/-- Theorem stating that the maximum number of plates that can fit on the table is 44 -/
theorem max_plates_on_table :
  ∀ k : ℕ, 
    (k : ℝ) * π * plate_radius^2 ≤ π * table_radius^2 ↔ k ≤ max_plates :=
by sorry

end NUMINAMATH_CALUDE_max_plates_on_table_l1003_100300


namespace NUMINAMATH_CALUDE_phoebes_servings_is_one_l1003_100357

/-- The number of servings per jar of peanut butter -/
def servings_per_jar : ℕ := 15

/-- The number of jars needed -/
def jars_needed : ℕ := 4

/-- The number of days the peanut butter should last -/
def days_to_last : ℕ := 30

/-- Phoebe's serving amount equals her dog's serving amount -/
axiom phoebe_dog_equal_servings : True

/-- The number of servings Phoebe eats each night -/
def phoebes_nightly_servings : ℚ :=
  (servings_per_jar * jars_needed : ℚ) / (2 * days_to_last)

theorem phoebes_servings_is_one :
  phoebes_nightly_servings = 1 := by sorry

end NUMINAMATH_CALUDE_phoebes_servings_is_one_l1003_100357


namespace NUMINAMATH_CALUDE_rational_function_identity_l1003_100373

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n

theorem rational_function_identity 
  (f : ℚ → ℚ) 
  (h1 : ∃ a : ℚ, ¬is_integer (f a))
  (h2 : ∀ x y : ℚ, is_integer (f (x + y) - f x - f y))
  (h3 : ∀ x y : ℚ, is_integer (f (x * y) - f x * f y)) :
  ∀ x : ℚ, f x = x :=
sorry

end NUMINAMATH_CALUDE_rational_function_identity_l1003_100373


namespace NUMINAMATH_CALUDE_vertex_in_third_quadrant_l1003_100301

/-- Definition of the parabola --/
def parabola (x : ℝ) : ℝ := -2 * (x + 3)^2 - 21

/-- Definition of the vertex of the parabola --/
def vertex : ℝ × ℝ := (-3, parabola (-3))

/-- Definition of the third quadrant --/
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- Theorem: The vertex of the parabola is in the third quadrant --/
theorem vertex_in_third_quadrant : in_third_quadrant vertex := by
  sorry

end NUMINAMATH_CALUDE_vertex_in_third_quadrant_l1003_100301


namespace NUMINAMATH_CALUDE_mistaken_calculation_l1003_100392

theorem mistaken_calculation (x : ℚ) : x - 20 = 52 → x / 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l1003_100392


namespace NUMINAMATH_CALUDE_books_for_vacation_l1003_100380

/-- The number of books that can be read given reading speed, book parameters, and reading time -/
def books_to_read (reading_speed : ℕ) (words_per_page : ℕ) (pages_per_book : ℕ) (reading_time : ℕ) : ℕ :=
  (reading_speed * reading_time * 60) / (words_per_page * pages_per_book)

/-- Theorem stating that given the specific conditions, the number of books to read is 6 -/
theorem books_for_vacation : books_to_read 40 100 80 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_for_vacation_l1003_100380


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l1003_100322

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sumOfFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def lastTwoDigits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum_of_factorials_15 :
  lastTwoDigits (sumOfFactorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l1003_100322


namespace NUMINAMATH_CALUDE_three_card_selection_count_l1003_100336

/-- The number of ways to select 3 different cards in order from a set of 13 cards -/
def select_three_cards : ℕ := 13 * 12 * 11

/-- Theorem stating that selecting 3 different cards in order from 13 cards results in 1716 possibilities -/
theorem three_card_selection_count : select_three_cards = 1716 := by
  sorry

end NUMINAMATH_CALUDE_three_card_selection_count_l1003_100336


namespace NUMINAMATH_CALUDE_sum_to_term_ratio_l1003_100351

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  h1 : a 5 - a 3 = 12
  h2 : a 6 - a 4 = 24

/-- The sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the ratio of sum to nth term -/
theorem sum_to_term_ratio (seq : GeometricSequence) (n : ℕ) :
  sum_n seq n / seq.a n = 2 - 2^(1 - n) :=
sorry

end NUMINAMATH_CALUDE_sum_to_term_ratio_l1003_100351


namespace NUMINAMATH_CALUDE_range_of_a_l1003_100304

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x - 1 + 3*a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f a 0 < f a 1) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f a x = 0) →
  (1/7 < a ∧ a < 1/5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1003_100304


namespace NUMINAMATH_CALUDE_modified_factor_tree_l1003_100389

theorem modified_factor_tree : 
  ∀ A B C D E : ℕ,
  A = B * C →
  B = 3 * D →
  C = 7 * E →
  D = 5 * 2 →
  E = 7 * 3 →
  A = 4410 := by
  sorry

end NUMINAMATH_CALUDE_modified_factor_tree_l1003_100389


namespace NUMINAMATH_CALUDE_apple_sorting_probability_l1003_100355

/-- Ratio of large apples to small apples -/
def largeToSmallRatio : ℚ := 9/1

/-- Probability of sorting a large apple as a small apple -/
def largeSortedAsSmall : ℚ := 5/100

/-- Probability of sorting a small apple as a large apple -/
def smallSortedAsLarge : ℚ := 2/100

/-- The probability that a "large apple" selected after sorting is indeed a large apple -/
def probLargeGivenSortedLarge : ℚ := 855/857

theorem apple_sorting_probability :
  let totalApples : ℚ := 10
  let largeApples : ℚ := (largeToSmallRatio * totalApples) / (largeToSmallRatio + 1)
  let smallApples : ℚ := totalApples - largeApples
  let probLarge : ℚ := largeApples / totalApples
  let probSmall : ℚ := smallApples / totalApples
  let probLargeSortedLarge : ℚ := 1 - largeSortedAsSmall
  let probLargeAndSortedLarge : ℚ := probLarge * probLargeSortedLarge
  let probSmallAndSortedLarge : ℚ := probSmall * smallSortedAsLarge
  let probSortedLarge : ℚ := probLargeAndSortedLarge + probSmallAndSortedLarge
  probLargeGivenSortedLarge = probLargeAndSortedLarge / probSortedLarge :=
by sorry

end NUMINAMATH_CALUDE_apple_sorting_probability_l1003_100355


namespace NUMINAMATH_CALUDE_least_value_quadratic_l1003_100335

theorem least_value_quadratic (x : ℝ) :
  (4 * x^2 + 7 * x + 3 = 5) → x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l1003_100335


namespace NUMINAMATH_CALUDE_line_equidistant_point_value_l1003_100317

/-- A line passing through (4, 4) with slope 0.5, equidistant from (0, A) and (12, 8), implies A = 32 -/
theorem line_equidistant_point_value (A : ℝ) : 
  let line_point : ℝ × ℝ := (4, 4)
  let line_slope : ℝ := 0.5
  let point_P : ℝ × ℝ := (0, A)
  let point_Q : ℝ × ℝ := (12, 8)
  (∃ (line : ℝ → ℝ), 
    (line (line_point.1) = line_point.2) ∧ 
    ((line (line_point.1 + 1) - line (line_point.1)) / 1 = line_slope) ∧
    (∃ (midpoint : ℝ × ℝ), 
      (midpoint.1 = (point_P.1 + point_Q.1) / 2) ∧
      (midpoint.2 = (point_P.2 + point_Q.2) / 2) ∧
      (line midpoint.1 = midpoint.2))) →
  A = 32 := by
sorry

end NUMINAMATH_CALUDE_line_equidistant_point_value_l1003_100317


namespace NUMINAMATH_CALUDE_tyrah_sarah_pencil_ratio_l1003_100363

/-- Given that Tyrah has 12 pencils and Sarah has 2 pencils, 
    prove that the ratio of Tyrah's pencils to Sarah's pencils is 6. -/
theorem tyrah_sarah_pencil_ratio :
  ∀ (tyrah_pencils sarah_pencils : ℕ),
    tyrah_pencils = 12 →
    sarah_pencils = 2 →
    (tyrah_pencils : ℚ) / sarah_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_tyrah_sarah_pencil_ratio_l1003_100363


namespace NUMINAMATH_CALUDE_special_function_properties_l1003_100387

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y
  zero_map : f 0 = 0
  pi_half_map : f (Real.pi / 2) = 1

/-- The function is odd -/
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function is periodic with period 2π -/
def is_periodic_2pi (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x

/-- Main theorem: The special function is odd and periodic with period 2π -/
theorem special_function_properties (sf : SpecialFunction) :
    is_odd sf.f ∧ is_periodic_2pi sf.f := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l1003_100387


namespace NUMINAMATH_CALUDE_common_chord_length_l1003_100325

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem common_chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l1003_100325


namespace NUMINAMATH_CALUDE_derek_savings_l1003_100316

theorem derek_savings (a₁ a₂ : ℕ) (sum : ℕ) : 
  a₁ = 2 → a₂ = 4 → sum = 4096 → 
  ∃ (r : ℚ), r > 0 ∧ 
    (∀ n : ℕ, n > 0 → n ≤ 12 → a₁ * r^(n-1) = a₂ * r^(n-2)) ∧
    (sum = a₁ * (1 - r^12) / (1 - r)) →
  a₁ * r^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_derek_savings_l1003_100316


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1003_100313

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (acute_θ : 0 < θ ∧ θ < π/2) 
  (x_gt_1 : x > 1) 
  (h : Real.sin (θ/2) = Real.sqrt ((x + 1)/(2*x))) : 
  Real.tan θ = Real.sqrt (2*x - 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1003_100313


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1003_100311

/-- The area of the union of a square with side length 8 and a circle with radius 12
    centered at the center of the square is equal to 144π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let union_area : ℝ := max square_area circle_area
  union_area = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1003_100311


namespace NUMINAMATH_CALUDE_spaatz_frankie_relation_binkie_frankie_relation_binkie_has_24_gemstones_l1003_100382

/-- The number of gemstones on Spaatz's collar -/
def spaatz_gemstones : ℕ := 1

/-- The number of gemstones on Frankie's collar -/
def frankie_gemstones : ℕ := 6

/-- The relationship between Spaatz's and Frankie's gemstones -/
theorem spaatz_frankie_relation : spaatz_gemstones = frankie_gemstones / 2 - 2 := by sorry

/-- The relationship between Binkie's and Frankie's gemstones -/
theorem binkie_frankie_relation : ∃ (binkie_gemstones : ℕ), binkie_gemstones = 4 * frankie_gemstones := by sorry

/-- The main theorem: Binkie has 24 gemstones -/
theorem binkie_has_24_gemstones : ∃ (binkie_gemstones : ℕ), binkie_gemstones = 24 := by sorry

end NUMINAMATH_CALUDE_spaatz_frankie_relation_binkie_frankie_relation_binkie_has_24_gemstones_l1003_100382


namespace NUMINAMATH_CALUDE_rounded_number_problem_l1003_100393

theorem rounded_number_problem (x : ℝ) (n : ℤ) :
  x > 0 ∧ n = ⌈1.28 * x⌉ ∧ (n : ℝ) - 1 < x ∧ x ≤ (n : ℝ) →
  x = 25/32 ∨ x = 25/16 ∨ x = 75/32 ∨ x = 25/8 :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_problem_l1003_100393


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1003_100354

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  2 * Real.sin (7 * π / 6) * Real.sin (π / 6 + C) + Real.cos C = -1/2 →
  c = 2 * Real.sqrt 3 →
  C = π / 3 ∧ 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    1/2 * a' * b' * Real.sin C ≤ 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1003_100354
