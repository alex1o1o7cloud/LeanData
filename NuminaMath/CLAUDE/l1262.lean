import Mathlib

namespace NUMINAMATH_CALUDE_train_journey_speed_l1262_126264

/-- Given a train journey with the following conditions:
  - The total distance is 5x km
  - The first part of the journey is x km at 40 kmph
  - The second part of the journey is 2x km at speed v
  - The average speed for the entire journey is 40 kmph
  Prove that the speed v during the second part of the journey is 20 kmph -/
theorem train_journey_speed (x : ℝ) (v : ℝ) 
  (h1 : x > 0) 
  (h2 : x / 40 + 2 * x / v = 5 * x / 40) : v = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_speed_l1262_126264


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1262_126213

/-- Triangle ABC with vertices A(-3,0), B(2,1), and C(-2,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def ABC : Triangle := ⟨(-3, 0), (2, 1), (-2, 3)⟩

/-- The equation of line BC -/
def line_BC : LineEquation := ⟨1, 2, -4⟩

/-- The equation of the perpendicular bisector of BC -/
def perp_bisector_BC : LineEquation := ⟨2, -1, 2⟩

theorem triangle_ABC_properties :
  let t := ABC
  (line_BC.a * t.B.1 + line_BC.b * t.B.2 + line_BC.c = 0 ∧
   line_BC.a * t.C.1 + line_BC.b * t.C.2 + line_BC.c = 0) ∧
  (perp_bisector_BC.a * ((t.B.1 + t.C.1) / 2) + 
   perp_bisector_BC.b * ((t.B.2 + t.C.2) / 2) + 
   perp_bisector_BC.c = 0 ∧
   perp_bisector_BC.a * line_BC.b = -perp_bisector_BC.b * line_BC.a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1262_126213


namespace NUMINAMATH_CALUDE_independence_test_suitable_for_categorical_variables_l1262_126253

/-- Independence test is a statistical method used to determine the relationship between two variables -/
structure IndependenceTest where
  is_statistical_method : Bool
  determines_relationship : Bool
  between_two_variables : Bool

/-- Categorical variables are a type of variable -/
structure CategoricalVariable where
  is_variable : Bool

/-- The statement that the independence test is suitable for examining the relationship between categorical variables -/
theorem independence_test_suitable_for_categorical_variables 
  (test : IndependenceTest) 
  (cat_var : CategoricalVariable) : 
  test.is_statistical_method ∧ 
  test.determines_relationship ∧ 
  test.between_two_variables → 
  (∃ (relationship : CategoricalVariable → CategoricalVariable → Prop), 
    test.determines_relationship ∧ 
    ∀ (x y : CategoricalVariable), relationship x y) := by
  sorry

end NUMINAMATH_CALUDE_independence_test_suitable_for_categorical_variables_l1262_126253


namespace NUMINAMATH_CALUDE_garden_length_l1262_126221

/-- Proves that a rectangular garden with length twice its width and perimeter 300 yards has a length of 100 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- Length is twice the width
  2 * length + 2 * width = 300 →  -- Perimeter is 300 yards
  length = 100 := by  -- Prove that length is 100 yards
sorry

end NUMINAMATH_CALUDE_garden_length_l1262_126221


namespace NUMINAMATH_CALUDE_no_geometric_progression_l1262_126267

/-- The sequence a_n defined as 3^n - 2^n -/
def a (n : ℕ) : ℤ := 3^n - 2^n

/-- Theorem stating that no three consecutive terms of the sequence form a geometric progression -/
theorem no_geometric_progression (m n : ℕ) (h : m < n) :
  a m * a (2*n - m) < a n ^ 2 ∧ a n ^ 2 < a m * a (2*n - m + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_progression_l1262_126267


namespace NUMINAMATH_CALUDE_gabby_fruit_count_l1262_126247

def watermelons : ℕ := 1

def peaches (w : ℕ) : ℕ := w + 12

def plums (p : ℕ) : ℕ := 3 * p

def total_fruits (w p l : ℕ) : ℕ := w + p + l

theorem gabby_fruit_count :
  total_fruits watermelons (peaches watermelons) (plums (peaches watermelons)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_gabby_fruit_count_l1262_126247


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l1262_126251

theorem mayoral_election_votes (x y z : ℕ) : 
  x = y + y / 2 →
  y = z - 2 * z / 5 →
  x = 22500 →
  z = 25000 :=
by sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l1262_126251


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_one_l1262_126270

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = (n * (a 1 + a n)) / 2
  sum_condition : sum 2 = sum 6
  a4_condition : a 4 = 1

/-- The fifth term of the special arithmetic sequence is -1 -/
theorem fifth_term_is_negative_one (seq : ArithmeticSequence) : seq.a 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_one_l1262_126270


namespace NUMINAMATH_CALUDE_beads_per_necklace_is_20_l1262_126254

/-- The number of beads needed to make one necklace -/
def beads_per_necklace : ℕ := sorry

/-- The number of necklaces made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of necklaces made on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of bracelets made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of earrings made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed for one bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed for one earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads used -/
def total_beads : ℕ := 325

theorem beads_per_necklace_is_20 : 
  beads_per_necklace = 20 :=
by
  have h1 : monday_necklaces * beads_per_necklace + 
            tuesday_necklaces * beads_per_necklace + 
            wednesday_bracelets * beads_per_bracelet + 
            wednesday_earrings * beads_per_earring = total_beads := by sorry
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_is_20_l1262_126254


namespace NUMINAMATH_CALUDE_min_production_volume_for_break_even_l1262_126232

/-- The total cost function -/
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The revenue function -/
def revenue (x : ℝ) : ℝ := 25 * x

/-- The break-even condition -/
def break_even (x : ℝ) : Prop := revenue x ≥ total_cost x

theorem min_production_volume_for_break_even :
  ∃ (x : ℝ), x = 150 ∧ 0 < x ∧ x < 240 ∧ break_even x ∧
  ∀ (y : ℝ), 0 < y ∧ y < x → ¬(break_even y) := by
  sorry

end NUMINAMATH_CALUDE_min_production_volume_for_break_even_l1262_126232


namespace NUMINAMATH_CALUDE_correct_num_schools_l1262_126282

/-- The number of schools receiving soccer ball donations -/
def num_schools : ℕ := 2

/-- The number of classes per school -/
def classes_per_school : ℕ := 9

/-- The number of soccer balls per class -/
def balls_per_class : ℕ := 5

/-- The total number of soccer balls donated -/
def total_balls : ℕ := 90

/-- Theorem stating that the number of schools is correct -/
theorem correct_num_schools : 
  num_schools * classes_per_school * balls_per_class = total_balls :=
by sorry

end NUMINAMATH_CALUDE_correct_num_schools_l1262_126282


namespace NUMINAMATH_CALUDE_sin_750_degrees_l1262_126258

theorem sin_750_degrees (h : ∀ x, Real.sin (x + 2 * Real.pi) = Real.sin x) : 
  Real.sin (750 * Real.pi / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_750_degrees_l1262_126258


namespace NUMINAMATH_CALUDE_M_intersect_N_l1262_126235

def M : Set ℝ := {x | -2 ≤ x - 1 ∧ x - 1 ≤ 2}

def N : Set ℝ := {x | ∃ k : ℕ+, x = 2 * k - 1}

theorem M_intersect_N : M ∩ N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1262_126235


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1262_126241

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1262_126241


namespace NUMINAMATH_CALUDE_ott_final_fraction_l1262_126230

-- Define the friends
inductive Friend
| Moe
| Loki
| Nick
| Ott
| Pat

-- Define the function for initial money
def initialMoney (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 14
  | Friend.Loki => 10
  | Friend.Nick => 8
  | Friend.Pat => 12
  | Friend.Ott => 0

-- Define the function for the fraction given by each friend
def fractionGiven (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/7
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Pat => 1/6
  | Friend.Ott => 0

-- Define the amount given by each friend
def amountGiven : ℚ := 2

-- Theorem statement
theorem ott_final_fraction :
  let totalInitial := (initialMoney Friend.Moe) + (initialMoney Friend.Loki) + 
                      (initialMoney Friend.Nick) + (initialMoney Friend.Pat)
  let totalGiven := 4 * amountGiven
  (totalGiven / (totalInitial + totalGiven)) = 2/11 := by sorry

end NUMINAMATH_CALUDE_ott_final_fraction_l1262_126230


namespace NUMINAMATH_CALUDE_apple_ratio_l1262_126275

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := 2 * monday_apples

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := 9

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := total_apples - (monday_apples + tuesday_apples + wednesday_apples + friday_apples)

theorem apple_ratio : thursday_apples = 4 * friday_apples := by sorry

end NUMINAMATH_CALUDE_apple_ratio_l1262_126275


namespace NUMINAMATH_CALUDE_real_roots_condition_l1262_126271

theorem real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_condition_l1262_126271


namespace NUMINAMATH_CALUDE_exists_non_negative_sums_l1262_126260

/-- Represents a sign change operation on a matrix -/
inductive SignChange
| Row (i : Nat)
| Col (j : Nat)

/-- Applies a sequence of sign changes to a matrix -/
def applySignChanges (A : Matrix (Fin m) (Fin n) ℝ) (changes : List SignChange) : Matrix (Fin m) (Fin n) ℝ :=
  sorry

/-- Checks if all row sums and column sums are non-negative -/
def allSumsNonNegative (A : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  sorry

/-- Main theorem: For any matrix, there exists a sequence of sign changes that makes all sums non-negative -/
theorem exists_non_negative_sums (m n : Nat) (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ (changes : List SignChange), allSumsNonNegative (applySignChanges A changes) :=
by
  sorry

end NUMINAMATH_CALUDE_exists_non_negative_sums_l1262_126260


namespace NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l1262_126214

-- Define the set A
def A : Set ℕ := {x | ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a ≠ 0 ∧ b ≠ 0}

-- State the theorem
theorem prime_square_in_A_implies_prime_in_A (p : ℕ) (h_prime : Nat.Prime p) :
  p^2 ∈ A → p ∈ A := by
  sorry

end NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l1262_126214


namespace NUMINAMATH_CALUDE_count_x_values_l1262_126266

theorem count_x_values (x y z w : ℕ+) 
  (h1 : x > y ∧ y > z ∧ z > w)
  (h2 : x + y + z + w = 4020)
  (h3 : x^2 - y^2 + z^2 - w^2 = 4020) :
  ∃ (S : Finset ℕ+), (∀ a ∈ S, ∃ y z w : ℕ+, 
    x = a ∧ 
    a > y ∧ y > z ∧ z > w ∧
    a + y + z + w = 4020 ∧
    a^2 - y^2 + z^2 - w^2 = 4020) ∧ 
  S.card = 1003 :=
sorry

end NUMINAMATH_CALUDE_count_x_values_l1262_126266


namespace NUMINAMATH_CALUDE_solve_inequality_find_a_l1262_126262

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + 2 * |x - a|

-- Theorem for part I
theorem solve_inequality (x : ℝ) :
  f x 1 < 5 ↔ -2/3 < x ∧ x < 8/3 :=
sorry

-- Theorem for part II
theorem find_a :
  ∃ (a : ℝ), (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = -4 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_find_a_l1262_126262


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1262_126299

-- Define the sets P and Q
def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

-- State the theorem
theorem set_intersection_theorem : (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1262_126299


namespace NUMINAMATH_CALUDE_calculation_proof_l1262_126298

theorem calculation_proof :
  (2/3 - 1/4 - 1/6) * 24 = 6 ∧
  (-2)^3 + (-9 + (-3)^2 * (1/3)) = -14 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1262_126298


namespace NUMINAMATH_CALUDE_prob_truth_or_lie_classroom_l1262_126280

/-- Represents the characteristics of a student population -/
structure StudentPopulation where
  total : ℕ
  truth_tellers : ℕ
  liars : ℕ
  both : ℕ
  avoiders : ℕ
  serious_liars_ratio : ℚ

/-- Calculates the probability of a student speaking truth or lying in a serious situation -/
def prob_truth_or_lie (pop : StudentPopulation) : ℚ :=
  let serious_liars := (pop.both : ℚ) * pop.serious_liars_ratio
  (pop.truth_tellers + pop.liars + serious_liars) / pop.total

/-- Theorem stating the probability of a student speaking truth or lying in a serious situation -/
theorem prob_truth_or_lie_classroom (pop : StudentPopulation) 
  (h1 : pop.total = 100)
  (h2 : pop.truth_tellers = 40)
  (h3 : pop.liars = 25)
  (h4 : pop.both = 15)
  (h5 : pop.avoiders = 20)
  (h6 : pop.serious_liars_ratio = 70 / 100)
  (h7 : pop.truth_tellers + pop.liars + pop.both + pop.avoiders = pop.total) :
  prob_truth_or_lie pop = 76 / 100 := by
  sorry

end NUMINAMATH_CALUDE_prob_truth_or_lie_classroom_l1262_126280


namespace NUMINAMATH_CALUDE_anna_candy_distribution_l1262_126204

/-- Given a number of candies and friends, returns the minimum number of candies
    to remove for equal distribution -/
def min_candies_to_remove (candies : ℕ) (friends : ℕ) : ℕ :=
  candies % friends

theorem anna_candy_distribution :
  let total_candies : ℕ := 30
  let num_friends : ℕ := 4
  min_candies_to_remove total_candies num_friends = 2 := by
sorry

end NUMINAMATH_CALUDE_anna_candy_distribution_l1262_126204


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l1262_126292

/-- Calculates the total cost of John's purchase given the number of gum packs, candy bars, and the cost of a candy bar. -/
def total_cost (gum_packs : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℚ) : ℚ :=
  let gum_cost := candy_bar_cost / 2
  gum_packs * gum_cost + candy_bars * candy_bar_cost

/-- Proves that John's total cost for 2 packs of gum and 3 candy bars is $6, given that each candy bar costs $1.5 and gum costs half as much. -/
theorem johns_purchase_cost : total_cost 2 3 (3/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l1262_126292


namespace NUMINAMATH_CALUDE_correct_propositions_l1262_126223

theorem correct_propositions (a b : ℝ) : 
  ((a > |b| → a^2 > b^2) ∧ (a > b → a^3 > b^3)) := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l1262_126223


namespace NUMINAMATH_CALUDE_count_valid_paths_l1262_126245

/-- The number of paths from (0,1) to (n-1,n) that stay strictly above y=x -/
def validPaths (n : ℕ) : ℚ :=
  (1 : ℚ) / n * (Nat.choose (2*n - 2) (n - 1))

/-- Theorem stating the number of valid paths -/
theorem count_valid_paths (n : ℕ) (h : n > 0) :
  validPaths n = (1 : ℚ) / n * (Nat.choose (2*n - 2) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_count_valid_paths_l1262_126245


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1262_126290

theorem floor_equation_solutions :
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    ⌊(2.018 : ℝ) * p.1⌋ + ⌊(5.13 : ℝ) * p.2⌋ = 24) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1262_126290


namespace NUMINAMATH_CALUDE_parallel_segments_l1262_126269

/-- Given four points on a Cartesian plane, if AB is parallel to XY, then k = -6 -/
theorem parallel_segments (k : ℝ) : 
  let A : ℝ × ℝ := (-6, 0)
  let B : ℝ × ℝ := (0, -6)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (16, k)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_XY := (Y.2 - X.2) / (Y.1 - X.1)
  slope_AB = slope_XY → k = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_l1262_126269


namespace NUMINAMATH_CALUDE_alex_speed_l1262_126234

/-- Given the running speeds of Rick, Jen, Mark, and Alex, prove Alex's speed -/
theorem alex_speed (rick_speed : ℚ) (jen_ratio : ℚ) (mark_ratio : ℚ) (alex_ratio : ℚ)
  (h1 : rick_speed = 5)
  (h2 : jen_ratio = 3 / 4)
  (h3 : mark_ratio = 4 / 3)
  (h4 : alex_ratio = 5 / 6) :
  alex_ratio * mark_ratio * jen_ratio * rick_speed = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_alex_speed_l1262_126234


namespace NUMINAMATH_CALUDE_mrs_randall_teaching_years_l1262_126249

theorem mrs_randall_teaching_years (third_grade_years second_grade_years : ℕ) 
  (h1 : third_grade_years = 18) 
  (h2 : second_grade_years = 8) : 
  third_grade_years + second_grade_years = 26 := by
  sorry

end NUMINAMATH_CALUDE_mrs_randall_teaching_years_l1262_126249


namespace NUMINAMATH_CALUDE_florist_roses_l1262_126229

theorem florist_roses (initial_roses : ℕ) : 
  (initial_roses - 16 + 19 = 40) → initial_roses = 37 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l1262_126229


namespace NUMINAMATH_CALUDE_greatest_number_odd_factors_under_200_l1262_126285

theorem greatest_number_odd_factors_under_200 :
  ∃ (n : ℕ), n < 200 ∧ n = 196 ∧ 
  (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧
  (∃ k : ℕ, n = k^2) :=
sorry

end NUMINAMATH_CALUDE_greatest_number_odd_factors_under_200_l1262_126285


namespace NUMINAMATH_CALUDE_fishmonger_sales_l1262_126268

/-- The total amount of fish sold by a fishmonger in two weeks, given the first week's sales and a multiplier for the second week. -/
def total_fish_sales (first_week : ℕ) (multiplier : ℕ) : ℕ :=
  first_week + multiplier * first_week

/-- Theorem stating that if a fishmonger sold 50 kg of salmon in the first week and three times that amount in the second week, the total amount of fish sold in two weeks is 200 kg. -/
theorem fishmonger_sales : total_fish_sales 50 3 = 200 := by
  sorry

#eval total_fish_sales 50 3

end NUMINAMATH_CALUDE_fishmonger_sales_l1262_126268


namespace NUMINAMATH_CALUDE_max_sum_of_sides_l1262_126277

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

theorem max_sum_of_sides (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = Real.sqrt 3 →
  f A = 1 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b + c ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_sides_l1262_126277


namespace NUMINAMATH_CALUDE_grandpa_jungmin_age_ratio_l1262_126227

/-- The ratio of grandpa's age to Jung-min's age this year, given their ages last year -/
def age_ratio (grandpa_last_year : ℕ) (jungmin_last_year : ℕ) : ℚ :=
  (grandpa_last_year + 1) / (jungmin_last_year + 1)

/-- Theorem stating that the ratio of grandpa's age to Jung-min's age this year is 8 -/
theorem grandpa_jungmin_age_ratio : age_ratio 71 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_grandpa_jungmin_age_ratio_l1262_126227


namespace NUMINAMATH_CALUDE_full_face_time_l1262_126210

/-- Represents the time taken for Wendy's skincare routine and makeup application -/
def skincare_routine : List ℕ := [2, 3, 3, 4, 1, 3, 2, 5, 2, 2]

/-- The time taken for makeup application -/
def makeup_time : ℕ := 30

/-- Theorem stating that the total time for Wendy's "full face" routine is 57 minutes -/
theorem full_face_time : (skincare_routine.sum + makeup_time) = 57 := by
  sorry

end NUMINAMATH_CALUDE_full_face_time_l1262_126210


namespace NUMINAMATH_CALUDE_inequality_proof_l1262_126240

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x / (x + y*z)) + (y / (y + z*x)) + (z / (z + x*y)) ≤ 2 / (1 - 3*x*y*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1262_126240


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l1262_126202

/-- An isosceles triangle with two sides of length 4 and a base of length 3 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  is_isosceles : side = 4 ∧ base = 3

/-- A circle passing through the vertices of a triangle -/
structure CircumscribedCircle (t : IsoscelesTriangle) where
  radius : ℝ
  passes_through_vertices : True  -- This is a simplification, as we can't easily express this condition in Lean

/-- The theorem stating that the area of the circumscribed circle is 16π -/
theorem circumscribed_circle_area (t : IsoscelesTriangle) 
  (c : CircumscribedCircle t) : Real.pi * c.radius ^ 2 = 16 * Real.pi := by
  sorry

#check circumscribed_circle_area

end NUMINAMATH_CALUDE_circumscribed_circle_area_l1262_126202


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l1262_126284

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = -1 + 2*I ∧ 
  z₂ = -3 - 2*I ∧ 
  (z₁^2 + 2*z₁ = -3 + 4*I) ∧ 
  (z₂^2 + 2*z₂ = -3 + 4*I) := by
sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l1262_126284


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l1262_126208

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four sequentially meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshed : A.teeth * A.speed = B.teeth * B.speed ∧
           B.teeth * B.speed = C.teeth * C.speed ∧
           C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the proportion of angular speeds for the gear system -/
theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.A.speed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth) ∧
    sys.B.speed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth) ∧
    sys.C.speed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth) ∧
    sys.D.speed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth) :=
  sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l1262_126208


namespace NUMINAMATH_CALUDE_steven_jill_difference_l1262_126259

/-- The number of peaches each person has -/
structure PeachCounts where
  jake : ℕ
  steven : ℕ
  jill : ℕ

/-- The conditions given in the problem -/
def problem_conditions (p : PeachCounts) : Prop :=
  p.jake + 6 = p.steven ∧
  p.steven > p.jill ∧
  p.jill = 5 ∧
  p.jake = 17

/-- The theorem to be proved -/
theorem steven_jill_difference (p : PeachCounts) 
  (h : problem_conditions p) : p.steven - p.jill = 18 := by
  sorry

end NUMINAMATH_CALUDE_steven_jill_difference_l1262_126259


namespace NUMINAMATH_CALUDE_candy_bar_price_is_correct_l1262_126295

/-- The price of a candy bar in dollars -/
def candy_bar_price : ℝ := 2

/-- The price of a bag of chips in dollars -/
def chips_price : ℝ := 0.5

/-- The number of students -/
def num_students : ℕ := 5

/-- The total amount needed for all students in dollars -/
def total_amount : ℝ := 15

/-- The number of candy bars each student gets -/
def candy_bars_per_student : ℕ := 1

/-- The number of bags of chips each student gets -/
def chips_per_student : ℕ := 2

theorem candy_bar_price_is_correct : 
  candy_bar_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_price_is_correct_l1262_126295


namespace NUMINAMATH_CALUDE_ord₂_3n_minus_1_l1262_126293

-- Define ord₂ function
def ord₂ (i : ℤ) : ℕ :=
  if i = 0 then 0 else (i.natAbs.factors.filter (· = 2)).length

-- Main theorem
theorem ord₂_3n_minus_1 (n : ℕ) (h : n > 0) :
  (ord₂ (3^n - 1) = 1 ↔ n % 2 = 1) ∧
  (¬ ∃ n, ord₂ (3^n - 1) = 2) ∧
  (ord₂ (3^n - 1) = 3 ↔ n % 4 = 2) :=
sorry

-- Additional lemma to ensure ord₂(3ⁿ - 1) > 0 for n > 0
lemma ord₂_3n_minus_1_pos (n : ℕ) (h : n > 0) :
  ord₂ (3^n - 1) > 0 :=
sorry

end NUMINAMATH_CALUDE_ord₂_3n_minus_1_l1262_126293


namespace NUMINAMATH_CALUDE_solve_equation_l1262_126288

theorem solve_equation (x : ℚ) (h : (1 / 4 : ℚ) - (1 / 6 : ℚ) = 4 / x) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1262_126288


namespace NUMINAMATH_CALUDE_transformed_variance_l1262_126206

-- Define a type for our dataset
def Dataset := Fin 10 → ℝ

-- Define the variance of a dataset
noncomputable def variance (X : Dataset) : ℝ := sorry

-- State the theorem
theorem transformed_variance (X : Dataset) 
  (h : variance X = 3) : 
  variance (fun i => 2 * (X i) + 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_transformed_variance_l1262_126206


namespace NUMINAMATH_CALUDE_problem_solution_l1262_126248

def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, m = 4 → (p x ∧ q x m) → 4 < x ∧ x < 5) ∧
  ((∀ x : ℝ, ¬(q x m) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) → 5/3 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1262_126248


namespace NUMINAMATH_CALUDE_composite_shape_perimeter_l1262_126243

/-- A figure composed of two unit squares and one unit equilateral triangle. -/
structure CompositeShape where
  /-- The side length of each square -/
  square_side : ℝ
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- Assertion that both squares and the triangle have unit side length -/
  h_unit_sides : square_side = 1 ∧ triangle_side = 1

/-- The perimeter of the composite shape -/
def perimeter (shape : CompositeShape) : ℝ :=
  3 * shape.square_side + 2 * shape.triangle_side

/-- Theorem stating that the perimeter of the composite shape is 5 units -/
theorem composite_shape_perimeter (shape : CompositeShape) :
  perimeter shape = 5 :=
sorry

end NUMINAMATH_CALUDE_composite_shape_perimeter_l1262_126243


namespace NUMINAMATH_CALUDE_third_term_is_five_l1262_126273

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- second term
  d : ℝ  -- common difference

/-- The sum of the second and fourth terms is 10 -/
def sum_second_fourth (seq : ArithmeticSequence) : Prop :=
  seq.a + (seq.a + 2 * seq.d) = 10

/-- The third term of the sequence -/
def third_term (seq : ArithmeticSequence) : ℝ :=
  seq.a + seq.d

/-- Theorem: If the sum of the second and fourth terms of an arithmetic sequence is 10,
    then the third term is 5 -/
theorem third_term_is_five (seq : ArithmeticSequence) 
    (h : sum_second_fourth seq) : third_term seq = 5 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_five_l1262_126273


namespace NUMINAMATH_CALUDE_skyler_song_count_l1262_126238

/-- The number of songs Skyler wrote in total -/
def total_songs (hit_songs top_100_songs unreleased_songs : ℕ) : ℕ :=
  hit_songs + top_100_songs + unreleased_songs

/-- Theorem stating the total number of songs Skyler wrote -/
theorem skyler_song_count :
  ∀ (hit_songs : ℕ),
    hit_songs = 25 →
    ∀ (top_100_songs : ℕ),
      top_100_songs = hit_songs + 10 →
      ∀ (unreleased_songs : ℕ),
        unreleased_songs = hit_songs - 5 →
        total_songs hit_songs top_100_songs unreleased_songs = 80 := by
  sorry

end NUMINAMATH_CALUDE_skyler_song_count_l1262_126238


namespace NUMINAMATH_CALUDE_james_vegetable_consumption_l1262_126219

/-- Represents James' vegetable consumption --/
structure VegetableConsumption where
  asparagus : Real
  broccoli : Real
  kale : Real

/-- Calculates the total weekly consumption given daily consumption of asparagus and broccoli --/
def weekly_consumption (daily : VegetableConsumption) : Real :=
  (daily.asparagus + daily.broccoli) * 7

/-- James' initial daily consumption --/
def initial_daily : VegetableConsumption :=
  { asparagus := 0.25, broccoli := 0.25, kale := 0 }

/-- James' consumption after doubling asparagus and broccoli and adding kale --/
def final_weekly : VegetableConsumption :=
  { asparagus := initial_daily.asparagus * 2 * 7,
    broccoli := initial_daily.broccoli * 2 * 7,
    kale := 3 }

/-- Theorem stating James' final weekly vegetable consumption --/
theorem james_vegetable_consumption :
  final_weekly.asparagus + final_weekly.broccoli + final_weekly.kale = 10 := by
  sorry


end NUMINAMATH_CALUDE_james_vegetable_consumption_l1262_126219


namespace NUMINAMATH_CALUDE_smallest_z_value_l1262_126256

theorem smallest_z_value (w x y z : ℤ) : 
  (∀ n : ℤ, n ≥ 0 → (w + 2*n)^3 + (x + 2*n)^3 + (y + 2*n)^3 = (z + 2*n)^3) →
  (x = w + 2) →
  (y = x + 2) →
  (z = y + 2) →
  (w > 0) →
  (2 : ℤ) ≤ z :=
sorry

end NUMINAMATH_CALUDE_smallest_z_value_l1262_126256


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l1262_126220

theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 18 →
  length = 3 * breadth →
  area = length * breadth →
  area = 972 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l1262_126220


namespace NUMINAMATH_CALUDE_small_boxes_count_l1262_126276

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 525) 
  (h2 : chocolates_per_box = 25) : 
  total_chocolates / chocolates_per_box = 21 := by
  sorry

#check small_boxes_count

end NUMINAMATH_CALUDE_small_boxes_count_l1262_126276


namespace NUMINAMATH_CALUDE_pet_store_snake_distribution_l1262_126263

/-- Given a total number of snakes and cages, calculate the number of snakes per cage -/
def snakesPerCage (totalSnakes : ℕ) (totalCages : ℕ) : ℕ :=
  totalSnakes / totalCages

theorem pet_store_snake_distribution :
  let totalSnakes : ℕ := 4
  let totalCages : ℕ := 2
  snakesPerCage totalSnakes totalCages = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_snake_distribution_l1262_126263


namespace NUMINAMATH_CALUDE_fish_count_proof_l1262_126265

/-- The number of fish Jerk Tuna has -/
def jerk_tuna_fish : ℕ := 144

/-- The number of fish Tall Tuna has -/
def tall_tuna_fish : ℕ := 2 * jerk_tuna_fish

/-- The total number of fish Jerk Tuna and Tall Tuna have together -/
def total_fish : ℕ := jerk_tuna_fish + tall_tuna_fish

theorem fish_count_proof : total_fish = 432 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_proof_l1262_126265


namespace NUMINAMATH_CALUDE_not_prime_cubic_polynomial_l1262_126203

theorem not_prime_cubic_polynomial (n : ℕ+) : ¬ Prime (n.val^3 - 9*n.val^2 + 19*n.val - 13) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_cubic_polynomial_l1262_126203


namespace NUMINAMATH_CALUDE_tank_capacity_l1262_126294

theorem tank_capacity (x : ℚ) 
  (h1 : 2/3 * x - 15 = 1/3 * x) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1262_126294


namespace NUMINAMATH_CALUDE_kims_class_hours_l1262_126226

/-- Calculates the total class hours after dropping a class -/
def total_class_hours_after_drop (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

/-- Proves that Kim's total class hours after dropping a class is 6 -/
theorem kims_class_hours : total_class_hours_after_drop 4 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kims_class_hours_l1262_126226


namespace NUMINAMATH_CALUDE_exactville_running_difference_l1262_126200

/-- Represents the town layout with square blocks and streets --/
structure TownLayout where
  block_side : ℝ  -- Side length of a square block
  street_width : ℝ  -- Width of the streets

/-- Calculates the difference in running distance between outer and inner paths --/
def running_distance_difference (town : TownLayout) : ℝ :=
  4 * (town.block_side + 2 * town.street_width) - 4 * town.block_side

/-- Theorem stating the difference in running distance for Exactville --/
theorem exactville_running_difference :
  let town : TownLayout := { block_side := 500, street_width := 25 }
  running_distance_difference town = 200 := by
  sorry

end NUMINAMATH_CALUDE_exactville_running_difference_l1262_126200


namespace NUMINAMATH_CALUDE_decimal_addition_l1262_126287

theorem decimal_addition : (4.358 + 3.892 : ℝ) = 8.250 := by sorry

end NUMINAMATH_CALUDE_decimal_addition_l1262_126287


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_closed_interval_l1262_126222

/-- Set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

/-- Set B defined in terms of x ∈ A -/
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 + 4*x + 3}

/-- The intersection of A and B is equal to the closed interval [-1, 1] -/
theorem A_intersect_B_eq_closed_interval :
  A ∩ B = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_closed_interval_l1262_126222


namespace NUMINAMATH_CALUDE_animal_shelter_multiple_l1262_126217

theorem animal_shelter_multiple (puppies kittens : ℕ) (h1 : puppies = 32) (h2 : kittens = 78)
  (h3 : ∃ x : ℕ, kittens = x * puppies + 14) : 
  ∃ x : ℕ, x = 2 ∧ kittens = x * puppies + 14 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_multiple_l1262_126217


namespace NUMINAMATH_CALUDE_shape_D_symmetric_l1262_126297

-- Define the shape type
inductive Shape
| A
| B
| C
| D
| E

-- Define the property of being symmetric with respect to a horizontal line
def isSymmetric (s1 s2 : Shape) : Prop := sorry

-- Define the given shape
def givenShape : Shape := sorry

-- Theorem statement
theorem shape_D_symmetric : 
  isSymmetric givenShape Shape.D := by sorry

end NUMINAMATH_CALUDE_shape_D_symmetric_l1262_126297


namespace NUMINAMATH_CALUDE_soccer_team_selection_l1262_126228

theorem soccer_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  quad_starters = 2 →
  (quadruplets.choose quad_starters) * ((total_players - quadruplets).choose (starters - quad_starters)) = 2970 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l1262_126228


namespace NUMINAMATH_CALUDE_negation_equivalence_l1262_126218

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1262_126218


namespace NUMINAMATH_CALUDE_decimal_100_to_binary_l1262_126216

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_100_to_binary :
  decimal_to_binary 100 = [1, 1, 0, 0, 1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_100_to_binary_l1262_126216


namespace NUMINAMATH_CALUDE_sum_of_valid_a_l1262_126224

theorem sum_of_valid_a : ∃ (S : Finset Int), 
  (∀ a ∈ S, (∃ x : Int, x ≤ 2 ∧ x > a + 2) ∧ 
             (∃ x y : Nat, a * x + 2 * y = -4 ∧ x + y = 4)) ∧
  (∀ a : Int, (∃ x : Int, x ≤ 2 ∧ x > a + 2) ∧ 
              (∃ x y : Nat, a * x + 2 * y = -4 ∧ x + y = 4) → a ∈ S) ∧
  (S.sum id = -16) := by
sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_l1262_126224


namespace NUMINAMATH_CALUDE_largest_fraction_l1262_126225

theorem largest_fraction : 
  (101 : ℚ) / 199 > 5 / 11 ∧
  (101 : ℚ) / 199 > 6 / 13 ∧
  (101 : ℚ) / 199 > 19 / 39 ∧
  (101 : ℚ) / 199 > 159 / 319 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1262_126225


namespace NUMINAMATH_CALUDE_no_base_for_131_perfect_square_l1262_126261

theorem no_base_for_131_perfect_square :
  ¬ ∃ (b : ℕ), b ≥ 2 ∧ ∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_131_perfect_square_l1262_126261


namespace NUMINAMATH_CALUDE_gcd_lcm_product_8_12_l1262_126291

theorem gcd_lcm_product_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_8_12_l1262_126291


namespace NUMINAMATH_CALUDE_tank_truck_ratio_l1262_126250

theorem tank_truck_ratio (trucks : ℕ) (total : ℕ) : 
  trucks = 20 → total = 140 → (total - trucks) / trucks = 6 := by
  sorry

end NUMINAMATH_CALUDE_tank_truck_ratio_l1262_126250


namespace NUMINAMATH_CALUDE_log_equation_solution_l1262_126237

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 2 = 5/2 → x = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1262_126237


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1262_126201

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 2 * a * y - 1 = 0 ↔ (3 * a - 1) * x + y - 1 = 0) → 
  a = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1262_126201


namespace NUMINAMATH_CALUDE_queenie_earnings_l1262_126281

/-- Calculates the total earnings for a worker given their daily rate, overtime rate, 
    number of days worked, and number of overtime hours. -/
def total_earnings (daily_rate : ℕ) (overtime_rate : ℕ) (days_worked : ℕ) (overtime_hours : ℕ) : ℕ :=
  daily_rate * days_worked + overtime_rate * overtime_hours

/-- Proves that Queenie's total earnings for 5 days of work with 4 hours overtime
    are equal to $770, given her daily rate of $150 and overtime rate of $5 per hour. -/
theorem queenie_earnings : total_earnings 150 5 5 4 = 770 := by
  sorry

end NUMINAMATH_CALUDE_queenie_earnings_l1262_126281


namespace NUMINAMATH_CALUDE_janet_waterpark_cost_l1262_126205

/-- Calculates the total cost for a group visiting a waterpark with a discount -/
def waterpark_cost (adult_price : ℚ) (num_adults num_children : ℕ) (discount_percent : ℚ) (soda_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let total_admission := adult_price * num_adults + child_price * num_children
  let discounted_admission := total_admission * (1 - discount_percent / 100)
  discounted_admission + soda_price

/-- The total cost for Janet's group visit to the waterpark -/
theorem janet_waterpark_cost :
  waterpark_cost 30 6 4 20 5 = 197 := by
  sorry

end NUMINAMATH_CALUDE_janet_waterpark_cost_l1262_126205


namespace NUMINAMATH_CALUDE_students_above_115_l1262_126278

/-- Represents the score distribution of a math test -/
structure ScoreDistribution where
  mean : ℝ
  variance : ℝ
  normal : Bool

/-- Represents a class of students who took a math test -/
structure MathClass where
  size : ℕ
  scores : ScoreDistribution
  prob_95_to_105 : ℝ

/-- Calculates the number of students who scored above a given threshold -/
def students_above_threshold (c : MathClass) (threshold : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of students who scored above 115 in the given conditions -/
theorem students_above_115 (c : MathClass) 
  (h1 : c.size = 50)
  (h2 : c.scores.mean = 105)
  (h3 : c.scores.variance = 100)
  (h4 : c.scores.normal = true)
  (h5 : c.prob_95_to_105 = 0.32) :
  students_above_threshold c 115 = 9 :=
sorry

end NUMINAMATH_CALUDE_students_above_115_l1262_126278


namespace NUMINAMATH_CALUDE_no_indefinite_cutting_l1262_126212

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

/-- Defines the cutting procedure for rectangles -/
def cut_rectangle (T : Rectangle) : Option (Rectangle × Rectangle) :=
  sorry

/-- Checks if two rectangles are similar -/
def are_similar (T1 T2 : Rectangle) : Prop :=
  T1.length / T1.width = T2.length / T2.width

/-- Checks if two rectangles are congruent -/
def are_congruent (T1 T2 : Rectangle) : Prop :=
  T1.length = T2.length ∧ T1.width = T2.width

/-- Defines the property of indefinite cutting -/
def can_cut_indefinitely (T : Rectangle) : Prop :=
  ∀ n : ℕ, ∃ (T_seq : ℕ → Rectangle), 
    T_seq 0 = T ∧
    (∀ i < n, 
      ∃ T1 T2 : Rectangle, 
        cut_rectangle (T_seq i) = some (T1, T2) ∧
        are_similar T1 T2 ∧
        ¬are_congruent T1 T2 ∧
        T_seq (i + 1) = T1)

theorem no_indefinite_cutting : ¬∃ T : Rectangle, can_cut_indefinitely T := by
  sorry

end NUMINAMATH_CALUDE_no_indefinite_cutting_l1262_126212


namespace NUMINAMATH_CALUDE_base_8_4513_equals_2379_l1262_126236

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4513_equals_2379 :
  base_8_to_10 [3, 1, 5, 4] = 2379 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4513_equals_2379_l1262_126236


namespace NUMINAMATH_CALUDE_jogging_ninth_day_l1262_126289

def minutes_jogged_6_days : ℕ := 6 * 80
def minutes_jogged_2_days : ℕ := 2 * 105
def total_minutes_8_days : ℕ := minutes_jogged_6_days + minutes_jogged_2_days
def desired_average : ℕ := 100
def total_days : ℕ := 9

theorem jogging_ninth_day :
  desired_average * total_days - total_minutes_8_days = 210 := by
  sorry

end NUMINAMATH_CALUDE_jogging_ninth_day_l1262_126289


namespace NUMINAMATH_CALUDE_find_number_l1262_126211

theorem find_number : ∃ x : ℤ, (305 + x) / 16 = 31 ∧ x = 191 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1262_126211


namespace NUMINAMATH_CALUDE_sequence_2018th_term_l1262_126252

theorem sequence_2018th_term (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n, 3 * S n = 2 * a n - 3 * n) : 
  a 2018 = 2^2018 - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_2018th_term_l1262_126252


namespace NUMINAMATH_CALUDE_committee_probability_l1262_126209

/-- The probability of selecting a 5-person committee with at least one boy and one girl
    from a group of 25 members (10 boys and 15 girls) is equal to 475/506. -/
theorem committee_probability (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total_members = 25 →
  boys = 10 →
  girls = 15 →
  committee_size = 5 →
  (Nat.choose total_members committee_size - Nat.choose boys committee_size - Nat.choose girls committee_size) /
  Nat.choose total_members committee_size = 475 / 506 := by
  sorry

#eval Nat.choose 25 5
#eval Nat.choose 10 5
#eval Nat.choose 15 5

end NUMINAMATH_CALUDE_committee_probability_l1262_126209


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1262_126239

theorem fish_population_estimate 
  (tagged_june : ℕ) 
  (caught_october : ℕ) 
  (tagged_october : ℕ) 
  (death_migration_rate : ℚ) 
  (new_fish_rate : ℚ) 
  (h1 : tagged_june = 50) 
  (h2 : caught_october = 80) 
  (h3 : tagged_october = 4) 
  (h4 : death_migration_rate = 30/100) 
  (h5 : new_fish_rate = 35/100) : 
  ℕ := by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l1262_126239


namespace NUMINAMATH_CALUDE_total_cost_is_75_l1262_126231

/-- Calculates the total cost for two siblings attending a music school with a sibling discount -/
def total_cost_for_siblings (regular_tuition : ℕ) (sibling_discount : ℕ) : ℕ :=
  regular_tuition + (regular_tuition - sibling_discount)

/-- Theorem stating that the total cost for two siblings is $75 given the specific tuition and discount -/
theorem total_cost_is_75 :
  total_cost_for_siblings 45 15 = 75 := by
  sorry

#eval total_cost_for_siblings 45 15

end NUMINAMATH_CALUDE_total_cost_is_75_l1262_126231


namespace NUMINAMATH_CALUDE_max_value_xyz_expression_l1262_126286

theorem max_value_xyz_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z)) / ((x + z)^2 * (y + z)^2) ≤ (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_xyz_expression_l1262_126286


namespace NUMINAMATH_CALUDE_least_possible_b_l1262_126257

-- Define a structure for our triangle
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ
  is_prime_a : Nat.Prime a
  is_prime_b : Nat.Prime b
  a_gt_b : a > b
  angle_sum : a + 2 * b = 180

-- Define the theorem
theorem least_possible_b (t : IsoscelesTriangle) : 
  (∀ t' : IsoscelesTriangle, t'.b ≥ t.b) → t.b = 19 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_b_l1262_126257


namespace NUMINAMATH_CALUDE_debose_family_mean_age_l1262_126272

theorem debose_family_mean_age : 
  let ages : List ℕ := [8, 8, 16, 18]
  let num_children := ages.length
  let sum_ages := ages.sum
  (sum_ages : ℚ) / num_children = 25/2 := by sorry

end NUMINAMATH_CALUDE_debose_family_mean_age_l1262_126272


namespace NUMINAMATH_CALUDE_harry_travel_time_l1262_126255

theorem harry_travel_time (initial_bus_time remaining_bus_time : ℕ) 
  (h1 : initial_bus_time = 15)
  (h2 : remaining_bus_time = 25) : 
  let total_bus_time := initial_bus_time + remaining_bus_time
  let walking_time := total_bus_time / 2
  initial_bus_time + remaining_bus_time + walking_time = 60 := by
sorry

end NUMINAMATH_CALUDE_harry_travel_time_l1262_126255


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l1262_126274

theorem complex_arithmetic_equality : 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001 = 76802 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l1262_126274


namespace NUMINAMATH_CALUDE_problem_statement_l1262_126207

/-- The problem statement as a theorem -/
theorem problem_statement 
  (ω : ℝ) 
  (hω : ω > 0)
  (a : ℝ → ℝ × ℝ)
  (b : ℝ → ℝ × ℝ)
  (ha : ∀ x, a x = (Real.sin (ω * x) + Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x)))
  (hb : ∀ x, b x = (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x)))
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (hsymmetry : ∀ x, f (x + π / (2 * ω)) = f x)
  (A B C : ℝ)
  (hC : f C = 1)
  (c : ℝ)
  (hc : c = 2)
  (hsin : Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A))
  : ω = 1 ∧ 
    (Real.sqrt 3 / 3 * c ^ 2 = 2 * Real.sqrt 3 / 3 ∨ 
     Real.sqrt 3 / 3 * c ^ 2 = 3 * Real.sqrt 3 / 7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1262_126207


namespace NUMINAMATH_CALUDE_simplify_expression_l1262_126296

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1262_126296


namespace NUMINAMATH_CALUDE_egg_count_problem_l1262_126279

/-- Calculates the final number of eggs for a family given initial count and various changes --/
def final_egg_count (initial : ℕ) (mother_used : ℕ) (father_used : ℕ) 
  (chicken1_laid : ℕ) (chicken2_laid : ℕ) (chicken3_laid : ℕ) (child_took : ℕ) : ℕ :=
  initial - mother_used - father_used + chicken1_laid + chicken2_laid + chicken3_laid - child_took

/-- Theorem stating that given the specific values in the problem, the final egg count is 19 --/
theorem egg_count_problem : 
  final_egg_count 20 5 3 4 3 2 2 = 19 := by sorry

end NUMINAMATH_CALUDE_egg_count_problem_l1262_126279


namespace NUMINAMATH_CALUDE_valentines_distribution_l1262_126233

theorem valentines_distribution (initial_valentines : Real) (additional_valentines : Real) (num_students : Nat) :
  initial_valentines = 58.0 →
  additional_valentines = 16.0 →
  num_students = 74 →
  (initial_valentines + additional_valentines) / num_students = 1 := by
  sorry

end NUMINAMATH_CALUDE_valentines_distribution_l1262_126233


namespace NUMINAMATH_CALUDE_sum_of_five_decimals_theorem_l1262_126244

/-- Represents a two-digit number with a decimal point between the digits -/
structure TwoDigitDecimal where
  firstDigit : ℕ
  secondDigit : ℕ
  first_digit_valid : firstDigit < 10
  second_digit_valid : secondDigit < 10

/-- The sum of five TwoDigitDecimal numbers -/
def sumFiveDecimals (a b c d e : TwoDigitDecimal) : ℚ :=
  (a.firstDigit + a.secondDigit / 10 : ℚ) +
  (b.firstDigit + b.secondDigit / 10 : ℚ) +
  (c.firstDigit + c.secondDigit / 10 : ℚ) +
  (d.firstDigit + d.secondDigit / 10 : ℚ) +
  (e.firstDigit + e.secondDigit / 10 : ℚ)

/-- All digits are different -/
def allDifferent (a b c d e : TwoDigitDecimal) : Prop :=
  a.firstDigit ≠ b.firstDigit ∧ a.firstDigit ≠ c.firstDigit ∧ a.firstDigit ≠ d.firstDigit ∧ a.firstDigit ≠ e.firstDigit ∧
  a.firstDigit ≠ a.secondDigit ∧ a.firstDigit ≠ b.secondDigit ∧ a.firstDigit ≠ c.secondDigit ∧ a.firstDigit ≠ d.secondDigit ∧ a.firstDigit ≠ e.secondDigit ∧
  b.firstDigit ≠ c.firstDigit ∧ b.firstDigit ≠ d.firstDigit ∧ b.firstDigit ≠ e.firstDigit ∧
  b.firstDigit ≠ b.secondDigit ∧ b.firstDigit ≠ c.secondDigit ∧ b.firstDigit ≠ d.secondDigit ∧ b.firstDigit ≠ e.secondDigit ∧
  c.firstDigit ≠ d.firstDigit ∧ c.firstDigit ≠ e.firstDigit ∧
  c.firstDigit ≠ c.secondDigit ∧ c.firstDigit ≠ d.secondDigit ∧ c.firstDigit ≠ e.secondDigit ∧
  d.firstDigit ≠ e.firstDigit ∧
  d.firstDigit ≠ d.secondDigit ∧ d.firstDigit ≠ e.secondDigit ∧
  e.firstDigit ≠ e.secondDigit ∧
  a.secondDigit ≠ b.secondDigit ∧ a.secondDigit ≠ c.secondDigit ∧ a.secondDigit ≠ d.secondDigit ∧ a.secondDigit ≠ e.secondDigit ∧
  b.secondDigit ≠ c.secondDigit ∧ b.secondDigit ≠ d.secondDigit ∧ b.secondDigit ≠ e.secondDigit ∧
  c.secondDigit ≠ d.secondDigit ∧ c.secondDigit ≠ e.secondDigit ∧
  d.secondDigit ≠ e.secondDigit

theorem sum_of_five_decimals_theorem (a b c d e : TwoDigitDecimal) 
  (h1 : allDifferent a b c d e)
  (h2 : ∀ x ∈ [a, b, c, d, e], x.secondDigit ≠ 0) :
  sumFiveDecimals a b c d e = 27 ∨ sumFiveDecimals a b c d e = 18 :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_decimals_theorem_l1262_126244


namespace NUMINAMATH_CALUDE_salary_change_l1262_126283

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.4)
  let final_salary := decreased_salary * (1 + 0.4)
  (initial_salary - final_salary) / initial_salary = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l1262_126283


namespace NUMINAMATH_CALUDE_circus_performers_time_ratio_l1262_126215

theorem circus_performers_time_ratio :
  ∀ (polly_time pulsar_time petra_time : ℕ),
    pulsar_time = 10 →
    ∃ k : ℕ, polly_time = k * pulsar_time →
    petra_time = polly_time / 6 →
    pulsar_time + polly_time + petra_time = 45 →
    polly_time / pulsar_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_circus_performers_time_ratio_l1262_126215


namespace NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l1262_126246

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that checks if a given starting day results in equal Tuesdays and Thursdays in a 30-day month -/
def equalTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  sorry

/-- The number of days that can be the first day of a 30-day month with equal Tuesdays and Thursdays -/
def validStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_tuesdays_thursdays :
  validStartDays = 4 :=
sorry

end NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l1262_126246


namespace NUMINAMATH_CALUDE_sum_of_digits_c_plus_d_l1262_126242

/-- The sum of digits of c + d, where c and d are defined as follows:
    c = 10^1986 - 1
    d = 6(10^1986 - 1)/9 -/
theorem sum_of_digits_c_plus_d : ℕ :=
  let c : ℕ := 10^1986 - 1
  let d : ℕ := 6 * (10^1986 - 1) / 9
  9931

#check sum_of_digits_c_plus_d

end NUMINAMATH_CALUDE_sum_of_digits_c_plus_d_l1262_126242
