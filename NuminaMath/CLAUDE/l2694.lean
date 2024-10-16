import Mathlib

namespace NUMINAMATH_CALUDE_trip_time_difference_l2694_269453

def speed : ℝ := 40
def distance1 : ℝ := 360
def distance2 : ℝ := 400

theorem trip_time_difference : 
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l2694_269453


namespace NUMINAMATH_CALUDE_greatest_k_value_l2694_269424

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l2694_269424


namespace NUMINAMATH_CALUDE_sign_of_b_is_negative_l2694_269409

/-- Given that exactly two of a+b, a-b, ab, a/b are positive and the other two are negative, prove that b < 0 -/
theorem sign_of_b_is_negative (a b : ℝ) 
  (h : (a + b > 0 ∧ a - b > 0 ∧ a * b < 0 ∧ a / b < 0) ∨
       (a + b > 0 ∧ a - b < 0 ∧ a * b > 0 ∧ a / b < 0) ∨
       (a + b > 0 ∧ a - b < 0 ∧ a * b < 0 ∧ a / b > 0) ∨
       (a + b < 0 ∧ a - b > 0 ∧ a * b > 0 ∧ a / b < 0) ∨
       (a + b < 0 ∧ a - b > 0 ∧ a * b < 0 ∧ a / b > 0) ∨
       (a + b < 0 ∧ a - b < 0 ∧ a * b > 0 ∧ a / b > 0))
  (h_nonzero : b ≠ 0) : b < 0 := by
  sorry


end NUMINAMATH_CALUDE_sign_of_b_is_negative_l2694_269409


namespace NUMINAMATH_CALUDE_walk_distance_proof_l2694_269469

/-- Calculates the total distance walked given a constant speed and two walking durations -/
def total_distance (speed : ℝ) (duration1 : ℝ) (duration2 : ℝ) : ℝ :=
  speed * (duration1 + duration2)

/-- Proves that walking at 4 miles per hour for 2 hours and then 0.5 hours results in 10 miles -/
theorem walk_distance_proof :
  let speed := 4
  let duration1 := 2
  let duration2 := 0.5
  total_distance speed duration1 duration2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l2694_269469


namespace NUMINAMATH_CALUDE_amount_in_paise_l2694_269489

theorem amount_in_paise : 
  let a : ℝ := 130
  let percentage : ℝ := 0.5
  let amount_in_rupees : ℝ := (percentage / 100) * a
  let paise_per_rupee : ℝ := 100
  (percentage / 100 * a) * paise_per_rupee = 65 := by
  sorry

end NUMINAMATH_CALUDE_amount_in_paise_l2694_269489


namespace NUMINAMATH_CALUDE_calculate_fifth_subject_marks_l2694_269429

/-- Given a student's marks in four subjects and their average across five subjects,
    calculate the marks in the fifth subject. -/
theorem calculate_fifth_subject_marks
  (math_marks physics_marks chem_marks bio_marks : ℕ)
  (average_marks : ℚ)
  (h_math : math_marks = 65)
  (h_physics : physics_marks = 82)
  (h_chem : chem_marks = 67)
  (h_bio : bio_marks = 90)
  (h_average : average_marks = 75.6)
  (h_subjects : (math_marks + physics_marks + chem_marks + bio_marks : ℚ) + english_marks = average_marks * 5) :
  english_marks = 74 := by
sorry

end NUMINAMATH_CALUDE_calculate_fifth_subject_marks_l2694_269429


namespace NUMINAMATH_CALUDE_equation_negative_root_l2694_269450

theorem equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) ↔ -1/2 < a ∧ a ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_equation_negative_root_l2694_269450


namespace NUMINAMATH_CALUDE_visitor_increase_percentage_l2694_269472

/-- Represents the percentage increase in visitors after implementing discounts -/
def overallPercentageIncrease (initialChildren : ℕ) (initialSeniors : ℕ) (initialAdults : ℕ)
  (childrenIncrease : ℚ) (seniorsIncrease : ℚ) : ℚ :=
  let totalInitial := initialChildren + initialSeniors + initialAdults
  let totalAfter := 
    (initialChildren * (1 + childrenIncrease)) + 
    (initialSeniors * (1 + seniorsIncrease)) + 
    initialAdults
  (totalAfter - totalInitial) / totalInitial * 100

/-- Theorem stating that the overall percentage increase in visitors is approximately 13.33% -/
theorem visitor_increase_percentage : 
  ∀ (initialChildren initialSeniors initialAdults : ℕ),
  initialChildren > 0 → initialSeniors > 0 → initialAdults > 0 →
  let childrenIncrease : ℚ := 25 / 100
  let seniorsIncrease : ℚ := 15 / 100
  abs (overallPercentageIncrease initialChildren initialSeniors initialAdults childrenIncrease seniorsIncrease - 40 / 3) < 1 / 100 :=
by
  sorry

#eval overallPercentageIncrease 100 100 100 (25 / 100) (15 / 100)

end NUMINAMATH_CALUDE_visitor_increase_percentage_l2694_269472


namespace NUMINAMATH_CALUDE_original_number_proof_l2694_269467

theorem original_number_proof (x : ℝ) : x * 1.2 = 288 → x = 240 := by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2694_269467


namespace NUMINAMATH_CALUDE_becky_eddie_age_ratio_l2694_269463

/-- Given the ages of Eddie, Irene, and the relationship between Irene and Becky's ages,
    prove that the ratio of Becky's age to Eddie's age is 1:4. -/
theorem becky_eddie_age_ratio 
  (eddie_age : ℕ) 
  (irene_age : ℕ) 
  (becky_age : ℕ) 
  (h1 : eddie_age = 92) 
  (h2 : irene_age = 46) 
  (h3 : irene_age = 2 * becky_age) : 
  becky_age * 4 = eddie_age := by
  sorry

#check becky_eddie_age_ratio

end NUMINAMATH_CALUDE_becky_eddie_age_ratio_l2694_269463


namespace NUMINAMATH_CALUDE_sequence_conditions_diamonds_in_G20_l2694_269440

/-- The number of diamonds in figure n of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 9
  else 4 * n^2 + 4 * n - 7

/-- The sequence satisfies the given conditions -/
theorem sequence_conditions (n : ℕ) (h : n ≥ 3) :
  num_diamonds n = num_diamonds (n-1) + 8 * n :=
sorry

/-- The number of diamonds in G₂₀ is 1673 -/
theorem diamonds_in_G20 : num_diamonds 20 = 1673 :=
sorry

end NUMINAMATH_CALUDE_sequence_conditions_diamonds_in_G20_l2694_269440


namespace NUMINAMATH_CALUDE_conference_handshakes_l2694_269470

/-- Conference attendees are divided into three groups -/
structure ConferenceGroups where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ

/-- Calculate the number of handshakes in the conference -/
def calculate_handshakes (groups : ConferenceGroups) : ℕ :=
  let group3_handshakes := groups.group3 * (groups.total - groups.group3)
  let group2_handshakes := groups.group2 * (groups.group1 + groups.group3)
  (group3_handshakes + group2_handshakes) / 2

/-- Theorem stating that the number of handshakes is 237 -/
theorem conference_handshakes :
  ∃ (groups : ConferenceGroups),
    groups.total = 40 ∧
    groups.group1 = 25 ∧
    groups.group2 = 10 ∧
    groups.group3 = 5 ∧
    calculate_handshakes groups = 237 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_l2694_269470


namespace NUMINAMATH_CALUDE_prob_max_with_replacement_prob_max_without_replacement_l2694_269427

variable (M n k : ℕ)

-- Probability of drawing maximum k with replacement
def prob_with_replacement (M n k : ℕ) : ℚ :=
  (k^n - (k-1)^n) / M^n

-- Probability of drawing maximum k without replacement
def prob_without_replacement (M n k : ℕ) : ℚ :=
  (Nat.choose (k-1) (n-1)) / (Nat.choose M n)

-- Theorem for drawing with replacement
theorem prob_max_with_replacement (h1 : M > 0) (h2 : k > 0) (h3 : k ≤ M) :
  prob_with_replacement M n k = (k^n - (k-1)^n) / M^n :=
by sorry

-- Theorem for drawing without replacement
theorem prob_max_without_replacement (h1 : M > 0) (h2 : n > 0) (h3 : n ≤ k) (h4 : k ≤ M) :
  prob_without_replacement M n k = (Nat.choose (k-1) (n-1)) / (Nat.choose M n) :=
by sorry

end NUMINAMATH_CALUDE_prob_max_with_replacement_prob_max_without_replacement_l2694_269427


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2694_269408

theorem complex_equation_solution :
  ∀ (x y : ℝ), (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 3)) → (x = 2.5 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2694_269408


namespace NUMINAMATH_CALUDE_tree_watering_boys_l2694_269479

theorem tree_watering_boys (total_trees : ℕ) (trees_per_boy : ℕ) (h1 : total_trees = 29) (h2 : trees_per_boy = 3) :
  ∃ (num_boys : ℕ), num_boys * trees_per_boy ≥ total_trees ∧ (num_boys - 1) * trees_per_boy < total_trees ∧ num_boys = 10 :=
sorry

end NUMINAMATH_CALUDE_tree_watering_boys_l2694_269479


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_main_theorem_l2694_269455

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 - x - 1

/-- The property that f has exactly one zero in the interval (0,1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f a x = 0

/-- Theorem stating that if f has exactly one zero in (0,1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 :=
sorry

/-- The main theorem: if f has exactly one zero in (0,1), then a ∈ (1, +∞) -/
theorem main_theorem :
  ∀ a : ℝ, has_unique_zero_in_interval a → a ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_main_theorem_l2694_269455


namespace NUMINAMATH_CALUDE_shelf_filling_l2694_269442

theorem shelf_filling (P Q T N K : ℕ) (hP : P > 0) (hQ : Q > 0) (hT : T > 0) (hN : N > 0) (hK : K > 0)
  (hUnique : P ≠ Q ∧ P ≠ T ∧ P ≠ N ∧ P ≠ K ∧ Q ≠ T ∧ Q ≠ N ∧ Q ≠ K ∧ T ≠ N ∧ T ≠ K ∧ N ≠ K)
  (hThicker : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y > x ∧ P * x + Q * y = T * x + N * y ∧ K * x = P * x + Q * y) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end NUMINAMATH_CALUDE_shelf_filling_l2694_269442


namespace NUMINAMATH_CALUDE_flu_outbreak_theorem_l2694_269496

/-- Represents the state of a dwarf --/
inductive DwarfState
| Sick
| Healthy
| Immune

/-- Represents the population of dwarves --/
structure DwarfPopulation where
  sick : Set Nat
  healthy : Set Nat
  immune : Set Nat

/-- Represents the flu outbreak --/
structure FluOutbreak where
  initialVaccinated : Bool
  population : Nat → DwarfPopulation

/-- The flu lasts indefinitely if some dwarves are initially vaccinated --/
def fluLastsIndefinitely (outbreak : FluOutbreak) : Prop :=
  outbreak.initialVaccinated ∧
  ∀ n : Nat, ∃ i : Nat, i ∈ (outbreak.population n).sick

/-- The flu eventually ends if no dwarves are initially immune --/
def fluEventuallyEnds (outbreak : FluOutbreak) : Prop :=
  ¬outbreak.initialVaccinated ∧
  ∃ n : Nat, ∀ i : Nat, i ∉ (outbreak.population n).sick

theorem flu_outbreak_theorem (outbreak : FluOutbreak) :
  (outbreak.initialVaccinated → fluLastsIndefinitely outbreak) ∧
  (¬outbreak.initialVaccinated → fluEventuallyEnds outbreak) := by
  sorry


end NUMINAMATH_CALUDE_flu_outbreak_theorem_l2694_269496


namespace NUMINAMATH_CALUDE_unique_b_value_l2694_269481

/-- The value of 524123 in base 81 -/
def base_81_value : ℕ := 3 + 2 * 81 + 4 * 81^2 + 1 * 81^3 + 2 * 81^4 + 5 * 81^5

/-- Theorem stating that if b is an integer between 1 and 30 (inclusive),
    and base_81_value - b is divisible by 17, then b must equal 11 -/
theorem unique_b_value (b : ℤ) (h1 : 1 ≤ b) (h2 : b ≤ 30) 
    (h3 : (base_81_value : ℤ) - b ≡ 0 [ZMOD 17]) : b = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l2694_269481


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2694_269497

theorem sqrt_sum_fractions : 
  Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2694_269497


namespace NUMINAMATH_CALUDE_ballot_marking_combinations_l2694_269473

theorem ballot_marking_combinations : 
  ∀ n : ℕ, n = 10 → n.factorial = 3628800 :=
by
  sorry

end NUMINAMATH_CALUDE_ballot_marking_combinations_l2694_269473


namespace NUMINAMATH_CALUDE_system_solution_l2694_269457

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 6 ∧ 5*x - 4*y = 2) ↔ (x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2694_269457


namespace NUMINAMATH_CALUDE_pauls_erasers_l2694_269480

/-- Represents the number of crayons and erasers Paul has --/
structure PaulsSupplies where
  initialCrayons : ℕ
  finalCrayons : ℕ
  erasers : ℕ

/-- Defines the conditions of Paul's supplies --/
def validSupplies (s : PaulsSupplies) : Prop :=
  s.initialCrayons = 601 ∧
  s.finalCrayons = 336 ∧
  s.erasers = s.finalCrayons + 70

/-- Theorem stating the number of erasers Paul got for his birthday --/
theorem pauls_erasers (s : PaulsSupplies) (h : validSupplies s) : s.erasers = 406 := by
  sorry

end NUMINAMATH_CALUDE_pauls_erasers_l2694_269480


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l2694_269466

def biased_coin_prob : ℚ := 3/4
def die_sides : ℕ := 6

theorem coin_and_die_probability :
  let heads_prob := biased_coin_prob
  let three_prob := 1 / die_sides
  heads_prob * three_prob = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l2694_269466


namespace NUMINAMATH_CALUDE_f_min_max_l2694_269465

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_min_max :
  (∃ x ∈ domain, f x = 0) ∧
  (∀ x ∈ domain, f x ≥ 0) ∧
  (∃ x ∈ domain, f x = 9) ∧
  (∀ x ∈ domain, f x ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_f_min_max_l2694_269465


namespace NUMINAMATH_CALUDE_liam_speed_reduction_l2694_269492

/-- Proves that Liam should have driven 5 mph slower to arrive exactly on time -/
theorem liam_speed_reduction (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) :
  distance = 10 →
  actual_speed = 30 →
  early_time = 4 / 60 →
  let required_speed := distance / (distance / actual_speed + early_time)
  actual_speed - required_speed = 5 := by sorry

end NUMINAMATH_CALUDE_liam_speed_reduction_l2694_269492


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l2694_269486

/-- Given that 8 oranges weigh as much as 6 apples, prove that 32 oranges weigh as much as 24 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
  orange_weight > 0 →
  apple_weight > 0 →
  8 * orange_weight = 6 * apple_weight →
  32 * orange_weight = 24 * apple_weight :=
by
  sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l2694_269486


namespace NUMINAMATH_CALUDE_last_two_digits_of_2005_power_1989_l2694_269471

theorem last_two_digits_of_2005_power_1989 :
  (2005^1989) % 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_2005_power_1989_l2694_269471


namespace NUMINAMATH_CALUDE_line_through_specific_points_l2694_269438

/-- A line passes through points (2023, 0) and (-2021, 2024), and also through (1, c).
    This theorem proves that c must equal 1012. -/
theorem line_through_specific_points (c : ℚ) : 
  (∃ (m b : ℚ), (0 = m * 2023 + b) ∧ 
                 (2024 = m * (-2021) + b) ∧ 
                 (c = m * 1 + b)) → 
  c = 1012 := by sorry

end NUMINAMATH_CALUDE_line_through_specific_points_l2694_269438


namespace NUMINAMATH_CALUDE_grocery_weight_difference_l2694_269426

theorem grocery_weight_difference (rice sugar green_beans remaining_stock : ℝ) : 
  rice = green_beans - 30 →
  green_beans = 60 →
  remaining_stock = (2/3 * rice) + (4/5 * sugar) + green_beans →
  remaining_stock = 120 →
  green_beans - sugar = 10 := by
sorry

end NUMINAMATH_CALUDE_grocery_weight_difference_l2694_269426


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2694_269434

/-- Given a geometric sequence with first term 3 and second term -1/6,
    prove that its sixth term is -1/629856 -/
theorem sixth_term_of_geometric_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 3) (h₂ : a₂ = -1/6) :
  let r := a₂ / a₁
  a₁ * r^5 = -1/629856 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2694_269434


namespace NUMINAMATH_CALUDE_evaluate_expression_l2694_269437

theorem evaluate_expression : 3000 * (3000^1500 + 3000^1500) = 2 * 3000^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2694_269437


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l2694_269447

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l2694_269447


namespace NUMINAMATH_CALUDE_min_throws_for_repeat_sum_l2694_269412

/-- Represents a fair six-sided die -/
def Die : Type := Fin 6

/-- The sum of four dice rolls -/
def DiceSum : Type := Nat

/-- The minimum possible sum when rolling four dice -/
def minSum : Nat := 4

/-- The maximum possible sum when rolling four dice -/
def maxSum : Nat := 24

/-- The number of possible unique sums when rolling four dice -/
def uniqueSums : Nat := maxSum - minSum + 1

/-- 
  Theorem: The minimum number of throws needed to ensure the same sum 
  is rolled twice with four fair six-sided dice is 22.
-/
theorem min_throws_for_repeat_sum : 
  (uniqueSums + 1 : Nat) = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeat_sum_l2694_269412


namespace NUMINAMATH_CALUDE_probability_one_pair_l2694_269417

def total_gloves : ℕ := 10
def pairs_of_gloves : ℕ := 5
def gloves_picked : ℕ := 4

def total_ways : ℕ := Nat.choose total_gloves gloves_picked

def ways_one_pair : ℕ := 
  Nat.choose pairs_of_gloves 1 * Nat.choose 2 2 * Nat.choose (total_gloves - 2) (gloves_picked - 2)

theorem probability_one_pair :
  (ways_one_pair : ℚ) / total_ways = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_one_pair_l2694_269417


namespace NUMINAMATH_CALUDE_locus_is_finite_l2694_269494

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Definition of the right triangle -/
def rightTriangle (c : ℝ) : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ c ∧ 0 ≤ p.y ∧ p.y ≤ c ∧ p.x + p.y ≤ c}

/-- The set of points satisfying the given conditions -/
def locusSet (c : ℝ) : Set Point :=
  {p ∈ rightTriangle c |
    distanceSquared p ⟨0, 0⟩ + distanceSquared p ⟨c, 0⟩ = 2 * c^2 ∧
    distanceSquared p ⟨0, c⟩ = c^2}

theorem locus_is_finite (c : ℝ) (h : c > 0) : Set.Finite (locusSet c) :=
  sorry

end NUMINAMATH_CALUDE_locus_is_finite_l2694_269494


namespace NUMINAMATH_CALUDE_collinear_vectors_l2694_269443

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is collinear with c, then k = 1 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (h : a = (Real.sqrt 3, 1)) (h' : b = (0, -1)) 
    (h'' : c = (k, Real.sqrt 3)) (h''' : ∃ t : ℝ, a - 2 • b = t • c) : k = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2694_269443


namespace NUMINAMATH_CALUDE_maisie_flyers_l2694_269476

theorem maisie_flyers : 
  ∀ (maisie_flyers : ℕ), 
  (71 : ℕ) = 2 * maisie_flyers + 5 → 
  maisie_flyers = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_maisie_flyers_l2694_269476


namespace NUMINAMATH_CALUDE_arrangements_of_five_distinct_objects_l2694_269428

theorem arrangements_of_five_distinct_objects : 
  (Finset.univ : Finset (Equiv.Perm (Fin 5))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_of_five_distinct_objects_l2694_269428


namespace NUMINAMATH_CALUDE_distance_to_park_is_five_l2694_269402

/-- The distance from Talia's house to the park -/
def distance_to_park : ℝ := sorry

/-- The distance from the park to the grocery store -/
def park_to_grocery : ℝ := 3

/-- The distance from the grocery store to Talia's house -/
def grocery_to_house : ℝ := 8

/-- The total distance Talia drives -/
def total_distance : ℝ := 16

theorem distance_to_park_is_five :
  distance_to_park = 5 :=
by
  have h1 : distance_to_park + park_to_grocery + grocery_to_house = total_distance := by sorry
  sorry

end NUMINAMATH_CALUDE_distance_to_park_is_five_l2694_269402


namespace NUMINAMATH_CALUDE_multiple_of_72_digits_l2694_269459

theorem multiple_of_72_digits (n : ℕ) (x y : Fin 10) :
  (n = 320000000 + x * 10000000 + 35717 * 10 + y) →
  (n % 72 = 0) →
  (x * y = 12) :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_72_digits_l2694_269459


namespace NUMINAMATH_CALUDE_system_of_inequalities_l2694_269488

theorem system_of_inequalities (x : ℝ) : 
  (x + 1 < 5 ∧ (2 * x - 1) / 3 ≥ 1) ↔ 2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l2694_269488


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2694_269441

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (40 : ℤ) = Nat.gcd 40 ((12*x + 2)*(8*x + 14)*(10*x + 10)).natAbs := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2694_269441


namespace NUMINAMATH_CALUDE_dogsledding_race_speed_difference_l2694_269454

/-- The dogsledding race problem -/
theorem dogsledding_race_speed_difference
  (course_length : ℝ)
  (team_b_speed : ℝ)
  (time_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : team_b_speed = 20)
  (h3 : time_difference = 3)
  (h4 : team_b_speed > 0) :
  let team_b_time := course_length / team_b_speed
  let team_a_time := team_b_time - time_difference
  let team_a_speed := course_length / team_a_time
  team_a_speed - team_b_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogsledding_race_speed_difference_l2694_269454


namespace NUMINAMATH_CALUDE_inequality_proof_l2694_269478

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2694_269478


namespace NUMINAMATH_CALUDE_union_when_a_neg_two_subset_condition_l2694_269474

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a^2 + 1}

-- Statement for part (i)
theorem union_when_a_neg_two :
  A (-2) ∪ B (-2) = {x : ℝ | -5 < x ∧ x < 5} := by sorry

-- Statement for part (ii)
theorem subset_condition :
  ∀ a : ℝ, B a ⊆ A a ↔ a ∈ ({x : ℝ | 1 ≤ x ∧ x ≤ 3} ∪ {-1}) := by sorry

end NUMINAMATH_CALUDE_union_when_a_neg_two_subset_condition_l2694_269474


namespace NUMINAMATH_CALUDE_rectangle_arrangement_perimeter_bounds_l2694_269414

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents an arrangement of rectangles -/
structure Arrangement where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of an arrangement -/
def perimeter (a : Arrangement) : ℝ :=
  2 * (a.length + a.width)

/-- The set of all possible arrangements of four 7x5 rectangles -/
def possible_arrangements : Set Arrangement :=
  sorry

theorem rectangle_arrangement_perimeter_bounds :
  let r : Rectangle := { length := 7, width := 5 }
  let arrangements := possible_arrangements
  ∃ (max_arr min_arr : Arrangement),
    max_arr ∈ arrangements ∧
    min_arr ∈ arrangements ∧
    (∀ a ∈ arrangements, perimeter a ≤ perimeter max_arr) ∧
    (∀ a ∈ arrangements, perimeter a ≥ perimeter min_arr) ∧
    perimeter max_arr = 66 ∧
    perimeter min_arr = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_perimeter_bounds_l2694_269414


namespace NUMINAMATH_CALUDE_symmetric_line_l2694_269430

/-- Given a line L1 with equation x - 4y + 2 = 0 and an axis of symmetry x = -2,
    the symmetric line L2 has the equation x + 4y + 2 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (∃ L1 : Set (ℝ × ℝ), L1 = {(x, y) | x - 4*y + 2 = 0}) →
  (∃ axis : Set (ℝ × ℝ), axis = {(x, y) | x = -2}) →
  (∃ L2 : Set (ℝ × ℝ), L2 = {(x, y) | x + 4*y + 2 = 0}) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2694_269430


namespace NUMINAMATH_CALUDE_no_infinite_sequence_satisfying_recurrence_l2694_269407

theorem no_infinite_sequence_satisfying_recurrence : 
  ¬ ∃ (a : ℕ → ℕ+), ∀ (n : ℕ), 
    (a (n + 2) : ℝ) = (a (n + 1) : ℝ) + Real.sqrt ((a (n + 1) : ℝ) + (a n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_satisfying_recurrence_l2694_269407


namespace NUMINAMATH_CALUDE_x_range_for_decreasing_sequence_l2694_269493

def decreasing_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 1 - x
  | n + 1 => (1 - x) ^ (n + 2)

theorem x_range_for_decreasing_sequence (x : ℝ) :
  (∀ n : ℕ, decreasing_sequence x n > decreasing_sequence x (n + 1)) ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_decreasing_sequence_l2694_269493


namespace NUMINAMATH_CALUDE_estimate_comparison_l2694_269446

theorem estimate_comparison (x y a b : ℝ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : a > b) 
  (h4 : b > 0) : 
  (x + a) - (y - b) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_comparison_l2694_269446


namespace NUMINAMATH_CALUDE_ellen_smoothie_strawberries_l2694_269421

/-- The amount of strawberries used in Ellen's smoothie recipe. -/
def strawberries : ℝ := 0.5 - (0.1 + 0.2)

/-- Theorem stating that Ellen used 0.2 cups of strawberries in her smoothie. -/
theorem ellen_smoothie_strawberries :
  strawberries = 0.2 := by sorry

end NUMINAMATH_CALUDE_ellen_smoothie_strawberries_l2694_269421


namespace NUMINAMATH_CALUDE_cube_colorings_correct_dodecahedron_colorings_correct_l2694_269495

/-- The number of rotational symmetries of a cube -/
def cubeSymmetries : ℕ := 24

/-- The number of rotational symmetries of a dodecahedron -/
def dodecahedronSymmetries : ℕ := 60

/-- The number of geometrically distinct colorings of a cube with 6 different colors -/
def cubeColorings : ℕ := 30

/-- The number of geometrically distinct colorings of a dodecahedron with 12 different colors -/
def dodecahedronColorings : ℕ := (Nat.factorial 11) / 5

theorem cube_colorings_correct :
  cubeColorings = (Nat.factorial 6) / cubeSymmetries :=
sorry

theorem dodecahedron_colorings_correct :
  dodecahedronColorings = (Nat.factorial 12) / dodecahedronSymmetries :=
sorry

end NUMINAMATH_CALUDE_cube_colorings_correct_dodecahedron_colorings_correct_l2694_269495


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2694_269468

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem: The wet surface area of a cistern with given dimensions is 134 square meters -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 10 8 1.5 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2694_269468


namespace NUMINAMATH_CALUDE_factors_of_M_l2694_269432

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 7^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 : ℕ) * (4 : ℕ) * (2 : ℕ)

/-- M is defined as 2^4 · 3^3 · 7^1 -/
def M : ℕ := 2^4 * 3^3 * 7^1

theorem factors_of_M :
  num_factors M = 40 :=
by sorry

end NUMINAMATH_CALUDE_factors_of_M_l2694_269432


namespace NUMINAMATH_CALUDE_multiple_properties_l2694_269405

theorem multiple_properties (x y : ℤ) 
  (hx : ∃ k : ℤ, x = 4 * k) 
  (hy : ∃ m : ℤ, y = 8 * m) : 
  (∃ n : ℤ, y = 4 * n) ∧ 
  (∃ p : ℤ, x - y = 4 * p) ∧ 
  (∃ q : ℤ, x - y = 2 * q) := by
sorry

end NUMINAMATH_CALUDE_multiple_properties_l2694_269405


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2694_269444

theorem trigonometric_identity (x : ℝ) : 
  4 * Real.sin (5 * x) * Real.cos (5 * x) * (Real.cos x ^ 4 - Real.sin x ^ 4) = Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2694_269444


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2694_269449

/-- A complex number z tracing a circle centered at the origin with radius 3 -/
def z_on_circle (z : ℂ) : Prop := Complex.abs z = 3

/-- The locus of points (x, y) satisfying x + yi = z + 1/z -/
def locus (z : ℂ) (x y : ℝ) : Prop := x + y * Complex.I = z + 1 / z

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

theorem locus_is_ellipse :
  ∀ z : ℂ, z_on_circle z →
  ∀ x y : ℝ, locus z x y →
  is_ellipse x y :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l2694_269449


namespace NUMINAMATH_CALUDE_equilateral_max_altitude_sum_l2694_269425

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the total length of bisectors
def totalBisectorLength (t : Triangle) : ℝ := sorry

-- Define the total length of altitudes
def totalAltitudeLength (t : Triangle) : ℝ := sorry

-- Define an equilateral triangle
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem statement
theorem equilateral_max_altitude_sum 
  (t : Triangle) 
  (fixed_bisector_sum : ℝ) 
  (h_bisector_sum : totalBisectorLength t = fixed_bisector_sum) :
  ∀ t' : Triangle, 
    totalBisectorLength t' = fixed_bisector_sum → 
    totalAltitudeLength t' ≤ totalAltitudeLength t ↔ 
    isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_equilateral_max_altitude_sum_l2694_269425


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_distance_l2694_269431

/-- The distance between intersection points of an ellipse and a parabola -/
theorem ellipse_parabola_intersection_distance : 
  ∀ (ellipse : (ℝ × ℝ) → Prop) (parabola : (ℝ × ℝ) → Prop) 
    (focus : ℝ × ℝ) (directrix : ℝ → ℝ × ℝ),
  (∀ x y, ellipse (x, y) ↔ x^2 / 16 + y^2 / 36 = 1) →
  (∃ c, ∀ x, directrix x = (c, x)) →
  (∃ x₁ y₁ x₂ y₂, ellipse (x₁, y₁) ∧ parabola (x₁, y₁) ∧
                   ellipse (x₂, y₂) ∧ parabola (x₂, y₂) ∧
                   (x₁, y₁) ≠ (x₂, y₂)) →
  (∃ x y, focus = (x, y) ∧ parabola (x, y)) →
  (∃ x y, focus = (x, y) ∧ ellipse (x, y)) →
  ∃ x₁ y₁ x₂ y₂, 
    ellipse (x₁, y₁) ∧ parabola (x₁, y₁) ∧
    ellipse (x₂, y₂) ∧ parabola (x₂, y₂) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 24 * Real.sqrt 5 / Real.sqrt (9 + 5 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_distance_l2694_269431


namespace NUMINAMATH_CALUDE_set_problem_l2694_269422

def U : Set ℕ := {x | x ≤ 10}

theorem set_problem (A B : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {4,5,6})
  (h4 : (U \ B) ∩ A = {2,3})
  (h5 : (U \ A) ∩ (U \ B) = {7,8}) :
  A = {2,3,4,5,6} ∧ B = {4,5,6,9,10} := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l2694_269422


namespace NUMINAMATH_CALUDE_camera_and_lens_cost_l2694_269499

theorem camera_and_lens_cost
  (old_camera_cost : ℝ)
  (new_camera_percentage : ℝ)
  (lens_original_price : ℝ)
  (lens_discount : ℝ)
  (h1 : old_camera_cost = 4000)
  (h2 : new_camera_percentage = 1.3)
  (h3 : lens_original_price = 400)
  (h4 : lens_discount = 200) :
  old_camera_cost * new_camera_percentage + (lens_original_price - lens_discount) = 5400 :=
by sorry

end NUMINAMATH_CALUDE_camera_and_lens_cost_l2694_269499


namespace NUMINAMATH_CALUDE_f_derivative_at_two_l2694_269413

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  (∀ x, HasDerivAt (f a b) ((a * x + b) / x^2) x) →
  (HasDerivAt (f a b) 0 1) →
  (∀ x, HasDerivAt (f a b) ((-2 * x + 2) / x^2) x) →
  HasDerivAt (f a b) (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_two_l2694_269413


namespace NUMINAMATH_CALUDE_min_value_theorem_l2694_269487

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  ∃ (min : ℝ), min = 5 ∧ ∀ y, y > 1 → x + 4 / (x - 1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2694_269487


namespace NUMINAMATH_CALUDE_parabola_c_value_l2694_269491

/-- A parabola passing through two given points has a specific c value -/
theorem parabola_c_value :
  ∀ (b c : ℝ),
  (1^2 + b*1 + c = 5) →
  ((-2)^2 + b*(-2) + c = -8) →
  c = 4/3 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2694_269491


namespace NUMINAMATH_CALUDE_banana_cost_l2694_269439

theorem banana_cost (milk_cost banana_cost total_spent : ℝ) : 
  milk_cost = 3 →
  (1 + 0.2) * (milk_cost + banana_cost) = total_spent →
  total_spent = 6 →
  banana_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_cost_l2694_269439


namespace NUMINAMATH_CALUDE_photo_arrangements_l2694_269484

def teacher : ℕ := 1
def boys : ℕ := 4
def girls : ℕ := 2
def total_people : ℕ := teacher + boys + girls

theorem photo_arrangements :
  (∃ (arrangements_girls_together : ℕ), arrangements_girls_together = 1440) ∧
  (∃ (arrangements_boys_apart : ℕ), arrangements_boys_apart = 144) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2694_269484


namespace NUMINAMATH_CALUDE_president_savings_l2694_269451

theorem president_savings (total_funds : ℝ) (friends_percentage : ℝ) (family_percentage : ℝ) : 
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  let friends_contribution := (friends_percentage / 100) * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := (family_percentage / 100) * remaining_after_friends
  let president_savings := remaining_after_friends - family_contribution
  president_savings = 4200 := by
sorry

end NUMINAMATH_CALUDE_president_savings_l2694_269451


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l2694_269445

theorem ice_cream_combinations : Nat.choose 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l2694_269445


namespace NUMINAMATH_CALUDE_contest_end_time_l2694_269460

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60 % 24, minutes := totalMinutes % 60 }

-- Define the start time (3:00 p.m.)
def startTime : Time := { hours := 15, minutes := 0 }

-- Define the duration in minutes
def duration : Nat := 720

-- Theorem to prove
theorem contest_end_time :
  addMinutes startTime duration = { hours := 3, minutes := 0 } := by
  sorry

end NUMINAMATH_CALUDE_contest_end_time_l2694_269460


namespace NUMINAMATH_CALUDE_fifth_group_number_l2694_269498

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_elements : ℕ
  sample_size : ℕ
  first_drawn : ℕ
  h_positive : 0 < total_elements
  h_sample_size : 0 < sample_size
  h_first_drawn : first_drawn ≤ total_elements
  h_divisible : total_elements % sample_size = 0

/-- The number drawn in a specific group -/
def number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_drawn + (group - 1) * (s.total_elements / s.sample_size)

/-- Theorem stating the number drawn in the fifth group -/
theorem fifth_group_number (s : SystematicSampling)
  (h1 : s.total_elements = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.first_drawn = 3) :
  number_in_group s 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fifth_group_number_l2694_269498


namespace NUMINAMATH_CALUDE_product_expansion_sum_l2694_269436

theorem product_expansion_sum (a b c d : ℤ) : 
  (∀ x, (5 * x^2 - 8 * x + 3) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  10 * a + 5 * b + 2 * c + d = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l2694_269436


namespace NUMINAMATH_CALUDE_binary_1101001_plus_14_equals_119_l2694_269435

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 1101001₂ -/
def binary_1101001 : List Bool := [true, false, false, true, false, true, true]

theorem binary_1101001_plus_14_equals_119 :
  binary_to_decimal binary_1101001 + 14 = 119 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101001_plus_14_equals_119_l2694_269435


namespace NUMINAMATH_CALUDE_largest_integer_solution_l2694_269416

theorem largest_integer_solution : ∃ (x : ℤ), x ≤ 20 ∧ |x - 3| = 15 ∧ ∀ (y : ℤ), y ≤ 20 ∧ |y - 3| = 15 → y ≤ x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l2694_269416


namespace NUMINAMATH_CALUDE_correct_height_l2694_269483

theorem correct_height (n : ℕ) (initial_avg : ℝ) (incorrect_height : ℝ) (actual_avg : ℝ) :
  n = 30 ∧
  initial_avg = 175 ∧
  incorrect_height = 151 ∧
  actual_avg = 174.5 →
  ∃ (actual_height : ℝ),
    actual_height = 166 ∧
    n * actual_avg = (n - 1) * initial_avg + actual_height - incorrect_height :=
by sorry

end NUMINAMATH_CALUDE_correct_height_l2694_269483


namespace NUMINAMATH_CALUDE_missing_digit_is_4_l2694_269456

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem missing_digit_is_4 (n : ℕ) (h1 : n ≥ 35204 ∧ n < 35304) 
  (h2 : is_divisible_by_9 n) : 
  ∃ (d : ℕ), d < 10 ∧ n = 35204 + d * 10 ∧ d = 4 := by
  sorry

#check missing_digit_is_4

end NUMINAMATH_CALUDE_missing_digit_is_4_l2694_269456


namespace NUMINAMATH_CALUDE_perfect_square_consecutive_base_equation_l2694_269404

theorem perfect_square_consecutive_base_equation :
  ∀ (A B : ℕ),
    (∃ n : ℕ, A = n^2) →
    B = A + 1 →
    (1 * A^2 + 2 * A + 3) + (2 * B + 1) = 5 * (A + B) →
    (A : ℝ) + B = 7 + 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_consecutive_base_equation_l2694_269404


namespace NUMINAMATH_CALUDE_min_pizzas_correct_l2694_269410

/-- The cost of the car John bought -/
def car_cost : ℕ := 5000

/-- The amount John earns per pizza delivered -/
def earnings_per_pizza : ℕ := 10

/-- The amount John spends on gas per pizza delivered -/
def gas_cost_per_pizza : ℕ := 3

/-- The net profit John makes per pizza delivered -/
def net_profit_per_pizza : ℕ := earnings_per_pizza - gas_cost_per_pizza

/-- The minimum number of pizzas John must deliver to earn back the car cost -/
def min_pizzas : ℕ := (car_cost + net_profit_per_pizza - 1) / net_profit_per_pizza

theorem min_pizzas_correct :
  min_pizzas * net_profit_per_pizza ≥ car_cost ∧
  ∀ n : ℕ, n < min_pizzas → n * net_profit_per_pizza < car_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_correct_l2694_269410


namespace NUMINAMATH_CALUDE_vector_parallelism_l2694_269420

theorem vector_parallelism (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + 2 * b.1, a.2 + 2 * b.2) = k • (2 * a.1 - b.1, 2 * a.2 - b.2)) →
  x = (1 / 2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_vector_parallelism_l2694_269420


namespace NUMINAMATH_CALUDE_min_value_zero_iff_c_eq_four_l2694_269461

/-- The quadratic expression in x and y with parameter c -/
def f (c x y : ℝ) : ℝ :=
  5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

/-- The theorem stating that c = 4 is the unique value for which the minimum of f is 0 -/
theorem min_value_zero_iff_c_eq_four :
  (∃ (c : ℝ), ∀ (x y : ℝ), f c x y ≥ 0 ∧ (∃ (x₀ y₀ : ℝ), f c x₀ y₀ = 0)) ↔ c = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_iff_c_eq_four_l2694_269461


namespace NUMINAMATH_CALUDE_right_triangle_area_l2694_269482

theorem right_triangle_area (leg : ℝ) (altitude : ℝ) (area : ℝ) : 
  leg = 15 → altitude = 9 → area = 84.375 → 
  ∃ (hypotenuse : ℝ), 
    hypotenuse * altitude / 2 = area ∧ 
    leg^2 + altitude^2 = (hypotenuse / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2694_269482


namespace NUMINAMATH_CALUDE_equation_solution_l2694_269411

theorem equation_solution : 
  ∃! x : ℚ, (3 * x - 1) / (4 * x - 4) = 2 / 3 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2694_269411


namespace NUMINAMATH_CALUDE_parabola_vertex_trajectory_l2694_269448

/-- The trajectory of the vertex of a parabola -/
def vertex_trajectory (x y m : ℝ) : Prop :=
  y - 4*x - 4*m*y = 0

/-- The equation of the trajectory -/
def trajectory_equation (x y : ℝ) : Prop :=
  y^2 = -4*x

/-- Theorem: The trajectory of the vertex of the parabola y - 4x - 4my = 0 
    is described by the equation y^2 = -4x -/
theorem parabola_vertex_trajectory :
  ∀ x y : ℝ, (∃ m : ℝ, vertex_trajectory x y m) ↔ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_trajectory_l2694_269448


namespace NUMINAMATH_CALUDE_sets_theorem_l2694_269433

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

theorem sets_theorem :
  (∀ a : ℝ, (Aᶜ ∪ B a = U) ↔ a ≤ 0) ∧
  (∀ a : ℝ, (A ∩ B a ≠ B a) ↔ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_sets_theorem_l2694_269433


namespace NUMINAMATH_CALUDE_f_negative_t_zero_l2694_269464

theorem f_negative_t_zero (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f x = 3 * x + Real.sin x + 1) →
  f t = 2 →
  f (-t) = 0 := by
sorry

end NUMINAMATH_CALUDE_f_negative_t_zero_l2694_269464


namespace NUMINAMATH_CALUDE_fraction_simplification_l2694_269400

theorem fraction_simplification (x : ℝ) : 
  (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2694_269400


namespace NUMINAMATH_CALUDE_min_value_of_a_plus_2b_l2694_269419

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = a * b - 3) :
  ∀ x y, x > 0 → y > 0 → x + y = x * y - 3 → a + 2 * b ≤ x + 2 * y ∧
  ∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ - 3 ∧ a₀ + 2 * b₀ = 4 * Real.sqrt 2 + 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_plus_2b_l2694_269419


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2694_269452

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * q) →
  a 2 = 1 - a 1 →
  a 4 = 9 - a 3 →
  a 4 + a 5 = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2694_269452


namespace NUMINAMATH_CALUDE_circle_ratio_problem_l2694_269477

theorem circle_ratio_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 3 * (π * a^2) → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_problem_l2694_269477


namespace NUMINAMATH_CALUDE_game_score_theorem_l2694_269490

/-- Calculates the total points scored in a game with three tries --/
def total_points (first_try : ℕ) (second_try_difference : ℕ) : ℕ :=
  let second_try := first_try - second_try_difference
  let third_try := 2 * second_try
  first_try + second_try + third_try

/-- Theorem stating that under the given conditions, the total points scored is 1390 --/
theorem game_score_theorem :
  total_points 400 70 = 1390 := by
  sorry

end NUMINAMATH_CALUDE_game_score_theorem_l2694_269490


namespace NUMINAMATH_CALUDE_fraction_equation_l2694_269475

theorem fraction_equation : 45 / (8 - 3/7) = 315/53 := by sorry

end NUMINAMATH_CALUDE_fraction_equation_l2694_269475


namespace NUMINAMATH_CALUDE_cole_miles_l2694_269418

theorem cole_miles (xavier katie cole : ℕ) 
  (h1 : xavier = 3 * katie) 
  (h2 : katie = 4 * cole) 
  (h3 : xavier = 84) : 
  cole = 7 := by
  sorry

end NUMINAMATH_CALUDE_cole_miles_l2694_269418


namespace NUMINAMATH_CALUDE_average_sum_difference_l2694_269485

theorem average_sum_difference (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ a b c : ℝ) 
  (hx : (x₁ + x₂ + x₃) / 3 = a)
  (hy : (y₁ + y₂ + y₃) / 3 = b)
  (hz : (z₁ + z₂ + z₃) / 3 = c) :
  ((x₁ + y₁ - z₁) + (x₂ + y₂ - z₂) + (x₃ + y₃ - z₃)) / 3 = a + b - c := by
  sorry

end NUMINAMATH_CALUDE_average_sum_difference_l2694_269485


namespace NUMINAMATH_CALUDE_stability_comparison_l2694_269423

-- Define the Student type
structure Student where
  name : String
  average_score : ℝ
  variance : ℝ

-- Define the concept of stability
def more_stable (a b : Student) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem stability_comparison (A B : Student) 
  (h1 : A.average_score = B.average_score)
  (h2 : A.variance > B.variance) : 
  more_stable B A := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_l2694_269423


namespace NUMINAMATH_CALUDE_maintenance_team_journey_l2694_269462

def walking_records : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def fuel_consumption_rate : ℝ := 3
def initial_fuel : ℝ := 180

theorem maintenance_team_journey :
  let net_distance : Int := walking_records.sum
  let total_distance : ℕ := walking_records.map (Int.natAbs) |>.sum
  let total_fuel_consumption : ℝ := (total_distance : ℝ) * fuel_consumption_rate
  let fuel_needed : ℝ := total_fuel_consumption - initial_fuel
  (net_distance = 39) ∧ 
  (total_distance = 65) ∧ 
  (total_fuel_consumption = 195) ∧ 
  (fuel_needed = 15) := by
sorry

end NUMINAMATH_CALUDE_maintenance_team_journey_l2694_269462


namespace NUMINAMATH_CALUDE_victors_percentage_l2694_269458

/-- Given that Victor scored 184 marks out of a maximum of 200 marks,
    prove that the percentage of marks he got is 92%. -/
theorem victors_percentage (marks_obtained : ℕ) (maximum_marks : ℕ) 
  (h1 : marks_obtained = 184) (h2 : maximum_marks = 200) :
  (marks_obtained : ℚ) / maximum_marks * 100 = 92 := by
  sorry

end NUMINAMATH_CALUDE_victors_percentage_l2694_269458


namespace NUMINAMATH_CALUDE_water_bottles_total_l2694_269401

/-- Represents the number of water bottles filled for each team --/
structure TeamBottles where
  football : ℕ
  soccer : ℕ
  lacrosse : ℕ
  rugby : ℕ

/-- Calculate the total number of water bottles filled for all teams --/
def total_bottles (t : TeamBottles) : ℕ :=
  t.football + t.soccer + t.lacrosse + t.rugby

/-- Theorem stating the total number of water bottles filled for the teams --/
theorem water_bottles_total :
  ∃ (t : TeamBottles),
    t.football = 11 * 6 ∧
    t.soccer = 53 ∧
    t.lacrosse = t.football + 12 ∧
    t.rugby = 49 ∧
    total_bottles t = 246 :=
by
  sorry


end NUMINAMATH_CALUDE_water_bottles_total_l2694_269401


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2694_269415

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a (n + 3) = 39 →
  a (n + 1) + a (n + 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2694_269415


namespace NUMINAMATH_CALUDE_rational_terms_count_l2694_269403

/-- The number of rational terms in the expansion of (√2 + ∛3)^100 -/
def rational_terms_a : ℕ := 26

/-- The number of rational terms in the expansion of (√2 + ∜3)^300 -/
def rational_terms_b : ℕ := 13

/-- Theorem stating the number of rational terms in the expansions -/
theorem rational_terms_count :
  (rational_terms_a = 26) ∧ (rational_terms_b = 13) := by sorry

end NUMINAMATH_CALUDE_rational_terms_count_l2694_269403


namespace NUMINAMATH_CALUDE_pages_to_read_tonight_l2694_269406

def pages_three_nights_ago : ℕ := 20

def pages_two_nights_ago (p : ℕ) : ℕ := p^2 + 5

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + sum_of_digits (n / 10))

def pages_last_night (p : ℕ) : ℕ := 3 * sum_of_digits p

def total_pages : ℕ := 500

theorem pages_to_read_tonight :
  total_pages - (pages_three_nights_ago + 
    pages_two_nights_ago pages_three_nights_ago + 
    pages_last_night (pages_two_nights_ago pages_three_nights_ago)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_tonight_l2694_269406
