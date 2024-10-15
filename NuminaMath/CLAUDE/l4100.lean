import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_relation_l4100_410058

theorem quadratic_root_relation (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ = 4 * x₁ ∧ x₁^2 + a*x₁ + 2*a = 0 ∧ x₂^2 + a*x₂ + 2*a = 0) → 
  a = 25/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l4100_410058


namespace NUMINAMATH_CALUDE_family_weight_gain_l4100_410035

/-- The weight gained by Orlando, in pounds -/
def orlando_weight : ℕ := 5

/-- The weight gained by Jose, in pounds -/
def jose_weight : ℕ := 2 * orlando_weight + 2

/-- The weight gained by Fernando, in pounds -/
def fernando_weight : ℕ := jose_weight / 2 - 3

/-- The total weight gained by the three family members, in pounds -/
def total_weight : ℕ := orlando_weight + jose_weight + fernando_weight

theorem family_weight_gain : total_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_family_weight_gain_l4100_410035


namespace NUMINAMATH_CALUDE_function_bound_l4100_410085

theorem function_bound (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1)
  (h2 : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f x| ≤ 1) :
  ∀ x : ℝ, |f x| ≤ 2 + x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l4100_410085


namespace NUMINAMATH_CALUDE_simplify_expression_l4100_410051

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4100_410051


namespace NUMINAMATH_CALUDE_additional_black_balls_probability_l4100_410063

/-- Represents the contents of a bag of colored balls -/
structure BagContents where
  white : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the probability of drawing a black ball from the bag -/
def probBlack (bag : BagContents) : ℚ :=
  bag.black / (bag.white + bag.black + bag.red)

/-- The initial contents of the bag -/
def initialBag : BagContents :=
  { white := 2, black := 3, red := 5 }

/-- The number of additional black balls added -/
def additionalBlackBalls : ℕ := 18

/-- The final contents of the bag after adding black balls -/
def finalBag : BagContents :=
  { white := initialBag.white,
    black := initialBag.black + additionalBlackBalls,
    red := initialBag.red }

theorem additional_black_balls_probability :
  probBlack finalBag = 3/4 := by sorry

end NUMINAMATH_CALUDE_additional_black_balls_probability_l4100_410063


namespace NUMINAMATH_CALUDE_smallest_number_1755_more_than_sum_of_digits_l4100_410094

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_1755_more_than_sum_of_digits :
  (∀ m : ℕ, m < 1770 → m ≠ sum_of_digits m + 1755) ∧
  1770 = sum_of_digits 1770 + 1755 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_1755_more_than_sum_of_digits_l4100_410094


namespace NUMINAMATH_CALUDE_log_expression_equality_l4100_410002

theorem log_expression_equality : 2 * (Real.log 256 / Real.log 4) - (Real.log (1/16) / Real.log 4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l4100_410002


namespace NUMINAMATH_CALUDE_time_difference_l4100_410098

-- Define constants
def blocks : ℕ := 12
def walk_time_per_block : ℕ := 1  -- in minutes
def bike_time_per_block : ℕ := 20 -- in seconds

-- Define functions
def walk_time : ℕ := blocks * walk_time_per_block

def bike_time_seconds : ℕ := blocks * bike_time_per_block
def bike_time : ℕ := bike_time_seconds / 60

-- Theorem
theorem time_difference : walk_time - bike_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_time_difference_l4100_410098


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l4100_410093

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : ℝ
  area_ratio : ℚ

/-- The properties of the trapezoid as described in the problem -/
axiom trapezoid_properties (t : Trapezoid) :
  t.longer_base = t.shorter_base + t.base_difference ∧
  t.base_difference = 150 ∧
  t.midline_ratio = (t.shorter_base + t.longer_base) / 2 ∧
  t.area_ratio = 3 / 2 ∧
  (t.midline_ratio - t.shorter_base) / (t.longer_base - t.midline_ratio) = t.area_ratio

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) :
  ⌊(t.equal_area_segment ^ 2) / 150⌋ = 550 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l4100_410093


namespace NUMINAMATH_CALUDE_max_correct_answers_l4100_410092

/-- Represents the result of a math contest. -/
structure ContestResult where
  correct : ℕ
  blank : ℕ
  incorrect : ℕ
  deriving Repr

/-- Calculates the score for a given contest result. -/
def calculateScore (result : ContestResult) : ℤ :=
  5 * result.correct - 2 * result.incorrect

/-- Checks if a contest result is valid (total questions = 60). -/
def isValidResult (result : ContestResult) : Prop :=
  result.correct + result.blank + result.incorrect = 60

/-- Theorem stating the maximum number of correct answers Evelyn could have. -/
theorem max_correct_answers (result : ContestResult) 
  (h1 : isValidResult result) 
  (h2 : calculateScore result = 150) : 
  result.correct ≤ 38 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_answers_l4100_410092


namespace NUMINAMATH_CALUDE_range_of_g_l4100_410077

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1

-- Define the range of a function
def has_range (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y ∈ S, ∃ x, f x = y

-- State the theorem
theorem range_of_g (a : ℝ) :
  (has_range (f a) Set.univ) → (has_range (g a) {y | y ≥ 1}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l4100_410077


namespace NUMINAMATH_CALUDE_set_cardinality_lower_bound_l4100_410037

theorem set_cardinality_lower_bound (A : Finset ℤ) (m : ℕ) (hm : m ≥ 2) 
  (B : Fin m → Finset ℤ) (hB : ∀ i, B i ⊆ A) (hB_nonempty : ∀ i, (B i).Nonempty) 
  (hsum : ∀ i, (B i).sum id = m ^ (i : ℕ).succ) : 
  A.card ≥ m / 2 := by
  sorry

end NUMINAMATH_CALUDE_set_cardinality_lower_bound_l4100_410037


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l4100_410073

theorem fraction_sum_equals_one (m n : ℝ) (h : m ≠ n) :
  m / (m - n) + n / (n - m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l4100_410073


namespace NUMINAMATH_CALUDE_unique_base_solution_l4100_410096

/-- Convert a base-6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number in base b to decimal --/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The unique positive solution to 35₆ = 151ᵦ --/
theorem unique_base_solution : 
  ∃! (b : ℕ), b > 0 ∧ base6ToDecimal 35 = baseBToDecimal 151 b := by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l4100_410096


namespace NUMINAMATH_CALUDE_compound_proposition_true_l4100_410044

theorem compound_proposition_true : 
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∨ (∀ n : ℝ, ∃ m : ℝ, m^2 < n) :=
by sorry

end NUMINAMATH_CALUDE_compound_proposition_true_l4100_410044


namespace NUMINAMATH_CALUDE_shoe_matching_problem_l4100_410076

/-- Represents a collection of shoes -/
structure ShoeCollection :=
  (total_pairs : ℕ)
  (color_count : ℕ)
  (indistinguishable : Bool)

/-- 
Given a collection of shoes, returns the minimum number of shoes
needed to guarantee at least one matching pair of the same color
-/
def minShoesForMatch (collection : ShoeCollection) : ℕ :=
  collection.total_pairs + 1

/-- Theorem statement for the shoe matching problem -/
theorem shoe_matching_problem (collection : ShoeCollection) 
  (h1 : collection.total_pairs = 24)
  (h2 : collection.color_count = 2)
  (h3 : collection.indistinguishable = true) :
  minShoesForMatch collection = 25 := by
  sorry

#check shoe_matching_problem

end NUMINAMATH_CALUDE_shoe_matching_problem_l4100_410076


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l4100_410081

theorem inequality_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l4100_410081


namespace NUMINAMATH_CALUDE_balance_after_transfer_l4100_410045

/-- The initial balance in Christina's bank account before the transfer -/
def initial_balance : ℕ := 27004

/-- The amount Christina transferred out of her account -/
def transferred_amount : ℕ := 69

/-- The remaining balance in Christina's account after the transfer -/
def remaining_balance : ℕ := 26935

/-- Theorem stating that the initial balance minus the transferred amount equals the remaining balance -/
theorem balance_after_transfer : 
  initial_balance - transferred_amount = remaining_balance := by sorry

end NUMINAMATH_CALUDE_balance_after_transfer_l4100_410045


namespace NUMINAMATH_CALUDE_map_scale_l4100_410007

/-- Given a map where 12 cm represents 90 km, prove that 20 cm represents 150 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm / 12 = real_km / 90) :
  (20 * real_km) / map_cm = 150 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l4100_410007


namespace NUMINAMATH_CALUDE_bottles_per_case_l4100_410000

/-- Given a company that produces bottles of water and uses cases to hold them,
    this theorem proves the number of bottles per case. -/
theorem bottles_per_case
  (total_bottles : ℕ)
  (total_cases : ℕ)
  (h1 : total_bottles = 60000)
  (h2 : total_cases = 12000)
  : total_bottles / total_cases = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_l4100_410000


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4100_410038

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_5 - a_1 = 15 and a_4 - a_2 = 6, then q = 1/2 or q = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 5 - a 1 = 15 →              -- Condition 1
  a 4 - a 2 = 6 →               -- Condition 2
  q = 1/2 ∨ q = 2 :=            -- Conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4100_410038


namespace NUMINAMATH_CALUDE_area_ratio_extended_triangle_l4100_410071

-- Define the triangle ABC and its extensions
structure ExtendedTriangle where
  -- Original equilateral triangle
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Extended sides
  BB' : ℝ
  CC' : ℝ
  AA' : ℝ
  -- Conditions
  equilateral : AB = BC ∧ BC = CA
  extension_BB' : BB' = 2 * AB
  extension_CC' : CC' = 3 * BC
  extension_AA' : AA' = 4 * CA

-- Define the theorem
theorem area_ratio_extended_triangle (t : ExtendedTriangle) :
  (t.AB + t.BB')^2 + (t.BC + t.CC')^2 + (t.CA + t.AA')^2 = 25 * (t.AB^2 + t.BC^2 + t.CA^2) :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_extended_triangle_l4100_410071


namespace NUMINAMATH_CALUDE_stevens_skittles_l4100_410031

theorem stevens_skittles (erasers : ℕ) (groups : ℕ) (items_per_group : ℕ) (skittles : ℕ) :
  erasers = 4276 →
  groups = 154 →
  items_per_group = 57 →
  skittles + erasers = groups * items_per_group →
  skittles = 4502 := by
  sorry

end NUMINAMATH_CALUDE_stevens_skittles_l4100_410031


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l4100_410087

/-- A random variable with normal distribution -/
def normal_dist (μ σ : ℝ) : Type := ℝ

/-- Probability measure for the normal distribution -/
noncomputable def P (ξ : normal_dist (-1) σ) (s : Set ℝ) : ℝ := sorry

/-- The statement of the problem -/
theorem normal_distribution_probability (σ : ℝ) (ξ : normal_dist (-1) σ) 
  (h : P ξ {x | -3 ≤ x ∧ x ≤ -1} = 0.4) : 
  P ξ {x | x ≥ 1} = 0.1 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l4100_410087


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l4100_410097

theorem fraction_sum_equals_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : x - y + z = x * y * z) : 1 / x - 1 / y + 1 / z = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l4100_410097


namespace NUMINAMATH_CALUDE_sum_of_120_mod_980_l4100_410005

-- Define the sum of first n natural numbers
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- State the theorem
theorem sum_of_120_mod_980 : sum_of_first_n 120 % 980 = 320 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_120_mod_980_l4100_410005


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l4100_410047

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem chocolate_milk_probability :
  binomial_probability 7 3 (2/3) = 280/2187 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l4100_410047


namespace NUMINAMATH_CALUDE_total_people_on_bus_l4100_410046

/-- The number of people initially on the bus -/
def initial_people : ℕ := 4

/-- The number of people who got on the bus at the stop -/
def people_who_got_on : ℕ := 13

/-- Theorem stating the total number of people on the bus after the stop -/
theorem total_people_on_bus : initial_people + people_who_got_on = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_people_on_bus_l4100_410046


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_twice_l4100_410043

/-- A quadratic function parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 - (2*k - 1) * x + k

/-- The discriminant of the quadratic function f -/
def discriminant (k : ℝ) : ℝ := (2*k - 1)^2 - 4*k*(k - 2)

/-- The condition for f to have two distinct real roots -/
def has_two_distinct_roots (k : ℝ) : Prop :=
  discriminant k > 0 ∧ k ≠ 2

theorem quadratic_intersects_x_axis_twice (k : ℝ) :
  has_two_distinct_roots k ↔ k > -1/4 ∧ k ≠ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_twice_l4100_410043


namespace NUMINAMATH_CALUDE_cornbread_pieces_l4100_410069

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 20)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cornbread_pieces_l4100_410069


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4100_410086

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
                   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)
  numerator / denominator = 
    (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
    (2 * Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4100_410086


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l4100_410041

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 5 = 1 →
  a 9 = 81 →
  a 7 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l4100_410041


namespace NUMINAMATH_CALUDE_candy_problem_l4100_410020

theorem candy_problem (x : ℚ) : 
  (((3/4 * x - 3) * 3/4 - 5) = 10) → x = 336 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l4100_410020


namespace NUMINAMATH_CALUDE_watch_correction_proof_l4100_410042

/-- Represents the time loss of a watch in minutes per day -/
def timeLossPerDay : ℝ := 3

/-- Represents the number of days between April 1 at 12 noon and April 10 at 6 P.M. -/
def daysElapsed : ℝ := 9.25

/-- Calculates the positive correction in minutes for the watch -/
def watchCorrection (loss : ℝ) (days : ℝ) : ℝ := loss * days

theorem watch_correction_proof :
  watchCorrection timeLossPerDay daysElapsed = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_watch_correction_proof_l4100_410042


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4100_410053

theorem complex_equation_solution (z : ℂ) : (3 - z) * Complex.I = 1 - 3 * Complex.I → z = 6 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4100_410053


namespace NUMINAMATH_CALUDE_tiffany_lives_gained_l4100_410027

theorem tiffany_lives_gained (initial_lives lost_lives final_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : lost_lives = 14)
  (h3 : final_lives = 56) : 
  final_lives - (initial_lives - lost_lives) = 27 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_gained_l4100_410027


namespace NUMINAMATH_CALUDE_percentage_of_240_l4100_410015

theorem percentage_of_240 : (3 / 8 : ℚ) / 100 * 240 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_240_l4100_410015


namespace NUMINAMATH_CALUDE_sandy_puppies_count_l4100_410062

/-- Given that Sandy initially had 8 puppies and gave away 4,
    prove that she now has 4 puppies. -/
theorem sandy_puppies_count (initial_puppies : Nat) (given_away : Nat) :
  initial_puppies = 8 → given_away = 4 → initial_puppies - given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandy_puppies_count_l4100_410062


namespace NUMINAMATH_CALUDE_complex_sum_conjugate_l4100_410066

open Complex

theorem complex_sum_conjugate (α β γ : ℝ) 
  (h : exp (I * α) + exp (I * β) + exp (I * γ) = (1 / 3 : ℂ) + (1 / 2 : ℂ) * I) : 
  exp (-I * α) + exp (-I * β) + exp (-I * γ) = (1 / 3 : ℂ) - (1 / 2 : ℂ) * I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_conjugate_l4100_410066


namespace NUMINAMATH_CALUDE_least_n_for_length_50_l4100_410048

-- Define the points A_n on the x-axis
def A (n : ℕ) : ℝ × ℝ := (0, 0)  -- We only need A_0 for the statement

-- Define the points B_n on y = x^2
def B (n : ℕ) : ℝ × ℝ := sorry

-- Define the property that A_{n-1}B_nA_n is an equilateral triangle
def is_equilateral_triangle (n : ℕ) : Prop := sorry

-- Define the length of A_0A_n
def length_A0An (n : ℕ) : ℝ := sorry

-- The main theorem
theorem least_n_for_length_50 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → length_A0An m < 50) ∧ length_A0An n ≥ 50 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_length_50_l4100_410048


namespace NUMINAMATH_CALUDE_monomial_exponent_sum_l4100_410026

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m n : ℕ) : Prop := m = 3 ∧ n = 2

theorem monomial_exponent_sum (m n : ℕ) (h : like_terms m n) : m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_sum_l4100_410026


namespace NUMINAMATH_CALUDE_sheep_count_l4100_410033

/-- Given 3 herds of sheep with 20 sheep in each herd, the total number of sheep is 60. -/
theorem sheep_count (num_herds : ℕ) (sheep_per_herd : ℕ) 
  (h1 : num_herds = 3) 
  (h2 : sheep_per_herd = 20) : 
  num_herds * sheep_per_herd = 60 := by
  sorry

end NUMINAMATH_CALUDE_sheep_count_l4100_410033


namespace NUMINAMATH_CALUDE_smallest_max_sum_l4100_410057

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_eq : p + q + r + s + t = 3015) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∃ (min_N : ℕ), 
    (∀ (p' q' r' s' t' : ℕ+), 
      p' + q' + r' + s' + t' = 3015 → 
      max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) ≥ min_N) ∧
    N = min_N ∧
    min_N = 1508 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l4100_410057


namespace NUMINAMATH_CALUDE_total_balloon_cost_l4100_410001

def fred_balloons : ℕ := 10
def fred_cost : ℚ := 1

def sam_balloons : ℕ := 46
def sam_cost : ℚ := 3/2

def dan_balloons : ℕ := 16
def dan_cost : ℚ := 3/4

theorem total_balloon_cost :
  (fred_balloons : ℚ) * fred_cost +
  (sam_balloons : ℚ) * sam_cost +
  (dan_balloons : ℚ) * dan_cost = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_balloon_cost_l4100_410001


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l4100_410040

/-- The distance between point A and point B in kilometers -/
def distance_AB : ℝ := 120

/-- Xiao Zhang's speed in km/h -/
def speed_Zhang : ℝ := 60

/-- Xiao Wang's speed in km/h -/
def speed_Wang : ℝ := 40

/-- Time difference between Xiao Zhang and Xiao Wang's departures in hours -/
def time_difference : ℝ := 1

/-- Total travel time for both Xiao Zhang and Xiao Wang in hours -/
def total_time : ℝ := 4

/-- The meeting point of Xiao Zhang and Xiao Wang in km from point A -/
def meeting_point : ℝ := 96

theorem meeting_point_theorem :
  speed_Zhang * time_difference + 
  (speed_Zhang * speed_Wang / (speed_Zhang + speed_Wang)) * 
  (distance_AB - speed_Zhang * time_difference) = meeting_point :=
sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l4100_410040


namespace NUMINAMATH_CALUDE_factory_non_defective_percentage_l4100_410083

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : ℝ
  defective_percentage : ℝ

/-- The factory setup -/
def factory : List Machine := [
  { production_percentage := 0.25, defective_percentage := 0.02 },
  { production_percentage := 0.35, defective_percentage := 0.04 },
  { production_percentage := 0.40, defective_percentage := 0.05 }
]

/-- Calculate the percentage of non-defective products -/
def non_defective_percentage (machines : List Machine) : ℝ :=
  1 - (machines.map (λ m => m.production_percentage * m.defective_percentage)).sum

/-- Theorem stating that the percentage of non-defective products is 96.1% -/
theorem factory_non_defective_percentage :
  non_defective_percentage factory = 0.961 := by
  sorry

end NUMINAMATH_CALUDE_factory_non_defective_percentage_l4100_410083


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l4100_410010

theorem circle_ratio_after_increase (r : ℝ) (h : r > 0) :
  let new_radius := r + 2
  let new_circumference := 2 * Real.pi * new_radius
  let new_diameter := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l4100_410010


namespace NUMINAMATH_CALUDE_train_speed_problem_l4100_410034

/-- Proves that given a train journey of 5x km, where 4x km is traveled at 20 kmph,
    and the average speed for the entire journey is 40/3 kmph,
    the speed for the initial x km is 40/7 kmph. -/
theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let total_distance : ℝ := 5 * x
  let second_leg_distance : ℝ := 4 * x
  let second_leg_speed : ℝ := 20
  let average_speed : ℝ := 40 / 3
  let initial_speed : ℝ := 40 / 7
  (total_distance / (x / initial_speed + second_leg_distance / second_leg_speed) = average_speed) :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_problem_l4100_410034


namespace NUMINAMATH_CALUDE_interest_first_year_l4100_410013

def initial_deposit : ℝ := 5000
def balance_after_first_year : ℝ := 5500
def second_year_increase : ℝ := 0.1
def total_increase : ℝ := 0.21

theorem interest_first_year :
  balance_after_first_year - initial_deposit = 500 :=
sorry

end NUMINAMATH_CALUDE_interest_first_year_l4100_410013


namespace NUMINAMATH_CALUDE_smallest_possible_d_l4100_410088

theorem smallest_possible_d : 
  ∀ c d : ℝ, 
  (2 < c) → 
  (c < d) → 
  (2 + c ≤ d) → 
  (1 / c + 1 / d ≤ 2) → 
  (∀ d' : ℝ, 
    (∃ c' : ℝ, (2 < c') ∧ (c' < d') ∧ (2 + c' ≤ d') ∧ (1 / c' + 1 / d' ≤ 2)) → 
    d' ≥ 2 + Real.sqrt 3) → 
  d = 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l4100_410088


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l4100_410004

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem fifth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) :
  a 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l4100_410004


namespace NUMINAMATH_CALUDE_common_solution_y_values_l4100_410029

theorem common_solution_y_values (x y : ℝ) : 
  (x^2 + y^2 - 3 = 0 ∧ x^2 - 4*y + 6 = 0) →
  (y = -2 + Real.sqrt 13 ∨ y = -2 - Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_values_l4100_410029


namespace NUMINAMATH_CALUDE_triangle_properties_l4100_410006

/-- Triangle ABC with given properties -/
structure Triangle where
  b : ℝ
  c : ℝ
  cosC : ℝ
  h_b : b = 2
  h_c : c = 3
  h_cosC : cosC = 1/3

/-- Theorems about the triangle -/
theorem triangle_properties (t : Triangle) :
  ∃ (a : ℝ) (area : ℝ) (cosBminusC : ℝ),
    -- Side length a
    a = 3 ∧
    -- Area of the triangle
    area = 2 * Real.sqrt 2 ∧
    -- Cosine of B minus C
    cosBminusC = 23/27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4100_410006


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4100_410009

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  c^2 = a * Real.cos B + b * Real.cos A →
  a = 3 →
  b = 3 →
  a + b + c = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4100_410009


namespace NUMINAMATH_CALUDE_function_through_point_l4100_410061

/-- Given a function f(x) = x^α that passes through (2, √2), prove f(9) = 3 -/
theorem function_through_point (α : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x ^ α) → f 2 = Real.sqrt 2 → f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l4100_410061


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_four_l4100_410082

theorem three_digit_perfect_cube_divisible_by_four :
  ∃! n : ℕ, 100 ≤ 8 * n^3 ∧ 8 * n^3 ≤ 999 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_four_l4100_410082


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4100_410070

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + 2 * i) * i = b + i → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4100_410070


namespace NUMINAMATH_CALUDE_second_polygon_sides_l4100_410065

theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure positive side length
  50 * (3 * s) = n * s →  -- Same perimeter condition
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l4100_410065


namespace NUMINAMATH_CALUDE_equal_savings_after_25_weeks_l4100_410056

/-- Proves that saving 7 dollars per week results in equal savings after 25 weeks --/
theorem equal_savings_after_25_weeks 
  (your_initial_savings : ℕ := 160)
  (friend_initial_savings : ℕ := 210)
  (friend_weekly_savings : ℕ := 5)
  (weeks : ℕ := 25)
  (your_weekly_savings : ℕ := 7) : 
  your_initial_savings + weeks * your_weekly_savings = 
  friend_initial_savings + weeks * friend_weekly_savings := by
sorry

end NUMINAMATH_CALUDE_equal_savings_after_25_weeks_l4100_410056


namespace NUMINAMATH_CALUDE_base4_21202_equals_610_l4100_410017

def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem base4_21202_equals_610 :
  base4ToBase10 [2, 0, 2, 1, 2] = 610 := by
  sorry

end NUMINAMATH_CALUDE_base4_21202_equals_610_l4100_410017


namespace NUMINAMATH_CALUDE_colton_sticker_distribution_l4100_410050

/-- Proves that Colton gave 4 stickers to each of his 3 friends --/
theorem colton_sticker_distribution :
  ∀ (initial_stickers : ℕ) 
    (friends : ℕ) 
    (remaining_stickers : ℕ) 
    (stickers_to_friend : ℕ),
  initial_stickers = 72 →
  friends = 3 →
  remaining_stickers = 42 →
  initial_stickers = 
    remaining_stickers + 
    (friends * stickers_to_friend) + 
    (friends * stickers_to_friend + 2) + 
    (friends * stickers_to_friend - 8) →
  stickers_to_friend = 4 := by
sorry

end NUMINAMATH_CALUDE_colton_sticker_distribution_l4100_410050


namespace NUMINAMATH_CALUDE_grandma_age_l4100_410032

-- Define the number of grandchildren
def num_grandchildren : ℕ := 5

-- Define the average age of the entire group
def group_average_age : ℚ := 26

-- Define the average age of the grandchildren
def grandchildren_average_age : ℚ := 7

-- Define the age difference between grandpa and grandma
def age_difference : ℕ := 1

-- Theorem statement
theorem grandma_age :
  ∃ (grandpa_age grandma_age : ℕ),
    -- The average age of the group is 26
    (grandpa_age + grandma_age + num_grandchildren * grandchildren_average_age) / (2 + num_grandchildren : ℚ) = group_average_age ∧
    -- Grandma is one year younger than grandpa
    grandpa_age = grandma_age + age_difference ∧
    -- Grandma's age is 73
    grandma_age = 73 := by
  sorry

end NUMINAMATH_CALUDE_grandma_age_l4100_410032


namespace NUMINAMATH_CALUDE_no_integer_solution_l4100_410016

theorem no_integer_solution (a b c d : ℤ) : 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧
   a * 62^3 + b * 62^2 + c * 62 + d = 2) → False :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l4100_410016


namespace NUMINAMATH_CALUDE_cork_mass_proof_l4100_410025

/-- The density of platinum in kg/m^3 -/
def platinum_density : ℝ := 2.15e4

/-- The density of cork wood in kg/m^3 -/
def cork_density : ℝ := 2.4e2

/-- The density of the combined system in kg/m^3 -/
def system_density : ℝ := 4.8e2

/-- The mass of the piece of platinum in kg -/
def platinum_mass : ℝ := 86.94

/-- The mass of the piece of cork wood in kg -/
def cork_mass : ℝ := 85

theorem cork_mass_proof :
  ∃ (cork_volume platinum_volume : ℝ),
    cork_volume > 0 ∧
    platinum_volume > 0 ∧
    cork_density = cork_mass / cork_volume ∧
    platinum_density = platinum_mass / platinum_volume ∧
    system_density = (cork_mass + platinum_mass) / (cork_volume + platinum_volume) :=
by sorry

end NUMINAMATH_CALUDE_cork_mass_proof_l4100_410025


namespace NUMINAMATH_CALUDE_exists_left_identity_element_l4100_410090

variable {T : Type*} [Fintype T]

def LeftIdentityElement (star : T → T → T) (a : T) : Prop :=
  ∀ b : T, star a b = a

theorem exists_left_identity_element
  (star : T → T → T)
  (assoc : ∀ a b c : T, star (star a b) c = star a (star b c))
  (comm : ∀ a b : T, star a b = star b a) :
  ∃ a : T, LeftIdentityElement star a :=
by
  sorry

end NUMINAMATH_CALUDE_exists_left_identity_element_l4100_410090


namespace NUMINAMATH_CALUDE_copy_machines_total_output_l4100_410079

/-- 
Given two copy machines with constant rates:
- Machine 1 makes 35 copies per minute
- Machine 2 makes 65 copies per minute

Prove that they make 3000 copies together in 30 minutes.
-/
theorem copy_machines_total_output : 
  let machine1_rate : ℕ := 35
  let machine2_rate : ℕ := 65
  let time_in_minutes : ℕ := 30
  (machine1_rate * time_in_minutes) + (machine2_rate * time_in_minutes) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_copy_machines_total_output_l4100_410079


namespace NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l4100_410019

theorem not_prime_n4_plus_n2_plus_1 (n : ℕ) (h : n ≥ 2) :
  ¬ Nat.Prime (n^4 + n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l4100_410019


namespace NUMINAMATH_CALUDE_orange_painted_cubes_l4100_410089

/-- Represents a cube construction with small cubes -/
structure CubeConstruction where
  small_edge : ℝ
  large_edge : ℝ
  all_sides_painted : Bool

/-- Calculates the number of small cubes with only one side painted -/
def cubes_with_one_side_painted (c : CubeConstruction) : ℕ :=
  sorry

/-- Theorem stating the number of small cubes with one side painted in the given construction -/
theorem orange_painted_cubes (c : CubeConstruction) 
  (h1 : c.small_edge = 2)
  (h2 : c.large_edge = 10)
  (h3 : c.all_sides_painted = true) :
  cubes_with_one_side_painted c = 54 := by
  sorry

end NUMINAMATH_CALUDE_orange_painted_cubes_l4100_410089


namespace NUMINAMATH_CALUDE_cube_surface_covering_l4100_410011

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  side_length : ℝ
  height : ℝ
  area : ℝ

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  side_length : ℝ
  surface_area : ℝ

/-- A covering is a collection of shapes that cover a surface. -/
structure Covering where
  shapes : List Rhombus
  total_area : ℝ

/-- Theorem: It is possible to cover the surface of a cube with fewer than six rhombuses. -/
theorem cube_surface_covering (c : Cube) : 
  ∃ (cov : Covering), cov.shapes.length < 6 ∧ cov.total_area = c.surface_area := by
  sorry


end NUMINAMATH_CALUDE_cube_surface_covering_l4100_410011


namespace NUMINAMATH_CALUDE_cosA_sinB_value_l4100_410022

theorem cosA_sinB_value (A B : Real) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : (4 + Real.tan A ^ 2) * (5 + Real.tan B ^ 2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = 1 / Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_cosA_sinB_value_l4100_410022


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l4100_410068

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 →
  blue = 3 * red →
  red = 14 →
  total = red + blue + yellow →
  yellow = 29 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l4100_410068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_property_l4100_410039

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_property_l4100_410039


namespace NUMINAMATH_CALUDE_cone_base_radius_l4100_410003

theorem cone_base_radius (α : Real) (n : Nat) (r : Real) (h₁ : α = 30 * π / 180) (h₂ : n = 11) (h₃ : r = 3) :
  let R := r * (1 / Real.sin (π / n) - 1 / Real.tan (π / 4 + α / 2))
  R = r / Real.sin (π / n) - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l4100_410003


namespace NUMINAMATH_CALUDE_new_shoes_cost_approx_l4100_410018

/-- Cost of repairing used shoes in dollars -/
def repair_cost : ℝ := 13.50

/-- Duration of repaired shoes in years -/
def repaired_duration : ℝ := 1

/-- Duration of new shoes in years -/
def new_duration : ℝ := 2

/-- Percentage increase in average cost per year of new shoes compared to repaired shoes -/
def percentage_increase : ℝ := 0.1852

/-- Cost of purchasing new shoes -/
def new_shoes_cost : ℝ := 2 * (repair_cost + percentage_increase * repair_cost)

theorem new_shoes_cost_approx :
  ∃ ε > 0, |new_shoes_cost - 32| < ε :=
sorry

end NUMINAMATH_CALUDE_new_shoes_cost_approx_l4100_410018


namespace NUMINAMATH_CALUDE_multiplication_and_division_problem_l4100_410084

theorem multiplication_and_division_problem : (-12 * 3) + ((-15 * -5) / 3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_division_problem_l4100_410084


namespace NUMINAMATH_CALUDE_total_rats_l4100_410014

/-- The number of rats each person has -/
structure RatCounts where
  kenia : ℕ
  hunter : ℕ
  elodie : ℕ

/-- The conditions of the rat problem -/
def rat_problem (r : RatCounts) : Prop :=
  r.kenia = 3 * (r.hunter + r.elodie) ∧
  r.elodie = 30 ∧
  r.elodie = r.hunter + 10

theorem total_rats (r : RatCounts) (h : rat_problem r) : 
  r.kenia + r.hunter + r.elodie = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_rats_l4100_410014


namespace NUMINAMATH_CALUDE_brocard_and_interior_angle_bound_l4100_410030

/-- The Brocard angle of a triangle -/
def brocardAngle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def isInsideOrOnBoundary (M A B C : ℝ × ℝ) : Prop := sorry

theorem brocard_and_interior_angle_bound (A B C M : ℝ × ℝ) :
  isInsideOrOnBoundary M A B C →
  min (brocardAngle A B C) (min (angle A B M) (min (angle B C M) (angle C A M))) ≤ Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_brocard_and_interior_angle_bound_l4100_410030


namespace NUMINAMATH_CALUDE_urn_problem_l4100_410021

/-- Given two urns with different compositions of colored balls, 
    prove that the number of blue balls in the second urn is 15 --/
theorem urn_problem (N : ℕ) : 
  (5 : ℚ) / 10 * (10 : ℚ) / (10 + N) +  -- Probability of both balls being green
  (5 : ℚ) / 10 * (N : ℚ) / (10 + N) =   -- Probability of both balls being blue
  (52 : ℚ) / 100 →                      -- Total probability of same color
  N = 15 := by
sorry

end NUMINAMATH_CALUDE_urn_problem_l4100_410021


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4100_410059

theorem complex_fraction_simplification :
  let z : ℂ := (7 + 8*I) / (3 - 4*I)
  z = 53/25 + 52/25 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4100_410059


namespace NUMINAMATH_CALUDE_sequence_properties_l4100_410074

/-- Definition of the sequence and its partial sum -/
def sequence_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (2 * S n) / n + n = 2 * a n + 1

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

/-- Definition of geometric sequence for three terms -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- Main theorem -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence_condition a S →
  (is_arithmetic_sequence a ∧
   (is_geometric_sequence (a 4) (a 7) (a 9) →
    ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l4100_410074


namespace NUMINAMATH_CALUDE_digital_earth_definition_l4100_410028

/-- Definition of Digital Earth -/
def DigitalEarth : Type := Unit

/-- Property of Digital Earth being a digitized, informational virtual Earth -/
def is_digitized_informational_virtual_earth (de : DigitalEarth) : Prop :=
  -- This is left abstract as the problem doesn't provide specific criteria
  True

/-- Theorem stating that Digital Earth refers to a digitized, informational virtual Earth -/
theorem digital_earth_definition :
  ∀ (de : DigitalEarth), is_digitized_informational_virtual_earth de :=
sorry

end NUMINAMATH_CALUDE_digital_earth_definition_l4100_410028


namespace NUMINAMATH_CALUDE_randy_pictures_l4100_410024

theorem randy_pictures (peter_pictures quincy_pictures randy_pictures total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + randy_pictures →
  randy_pictures = 5 := by
sorry

end NUMINAMATH_CALUDE_randy_pictures_l4100_410024


namespace NUMINAMATH_CALUDE_function_composition_theorem_l4100_410060

/-- Given two functions f and g, with f(x) = Ax - 3B² and g(x) = Bx + C,
    where B ≠ 0 and f(g(1)) = 0, prove that A = 3B² / (B + C),
    assuming B + C ≠ 0. -/
theorem function_composition_theorem (A B C : ℝ) 
  (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x + C
  f (g 1) = 0 → A = 3 * B^2 / (B + C) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_theorem_l4100_410060


namespace NUMINAMATH_CALUDE_total_surfers_l4100_410008

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 50

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 30

/-- The number of surfers on Venice beach -/
def venice_surfers : ℕ := 20

/-- The ratio of surfers on Malibu beach -/
def malibu_ratio : ℕ := 5

/-- The ratio of surfers on Santa Monica beach -/
def santa_monica_ratio : ℕ := 3

/-- The ratio of surfers on Venice beach -/
def venice_ratio : ℕ := 2

theorem total_surfers : 
  malibu_surfers + santa_monica_surfers + venice_surfers = 100 ∧
  malibu_surfers * santa_monica_ratio = santa_monica_surfers * malibu_ratio ∧
  venice_surfers * santa_monica_ratio = santa_monica_surfers * venice_ratio :=
by sorry

end NUMINAMATH_CALUDE_total_surfers_l4100_410008


namespace NUMINAMATH_CALUDE_division_problem_l4100_410012

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 5/2) : 
  c / a = 2/15 := by sorry

end NUMINAMATH_CALUDE_division_problem_l4100_410012


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4100_410072

theorem solve_exponential_equation (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28 → n = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4100_410072


namespace NUMINAMATH_CALUDE_radio_station_survey_l4100_410095

theorem radio_station_survey (males_dont_listen : ℕ) (females_listen : ℕ) 
  (total_listeners : ℕ) (total_non_listeners : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listeners = 180)
  (h4 : total_non_listeners = 120) :
  total_listeners - females_listen = 105 := by
  sorry

end NUMINAMATH_CALUDE_radio_station_survey_l4100_410095


namespace NUMINAMATH_CALUDE_equation_solution_l4100_410036

theorem equation_solution : ∃ x : ℚ, (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 ∧ x = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4100_410036


namespace NUMINAMATH_CALUDE_elena_snow_removal_l4100_410080

/-- The volume of snow Elena removes from her pathway -/
def snow_volume (length width depth : ℝ) (compaction_factor : ℝ) : ℝ :=
  length * width * depth * compaction_factor

/-- Theorem stating the volume of snow Elena removes -/
theorem elena_snow_removal :
  snow_volume 30 3 0.75 0.9 = 60.75 := by
  sorry

end NUMINAMATH_CALUDE_elena_snow_removal_l4100_410080


namespace NUMINAMATH_CALUDE_car_speed_l4100_410052

/-- Proves that given the conditions, the speed of the car is 50 miles per hour -/
theorem car_speed (gasoline_consumption : Real) (tank_capacity : Real) (travel_time : Real) (gasoline_used_fraction : Real) :
  gasoline_consumption = 1 / 30 →
  tank_capacity = 10 →
  travel_time = 5 →
  gasoline_used_fraction = 0.8333333333333334 →
  (tank_capacity * gasoline_used_fraction) / (travel_time * gasoline_consumption) = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l4100_410052


namespace NUMINAMATH_CALUDE_number_problem_l4100_410064

theorem number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 6) (h4 : x / y = 6) :
  x * y - (x - y) = 6 / 49 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4100_410064


namespace NUMINAMATH_CALUDE_g_pow_6_eq_id_l4100_410091

/-- Definition of the function g -/
def g (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := v
  (a + b, b + c, a + c)

/-- Definition of g^n for n ≥ 2 -/
def g_pow (n : ℕ) : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) :=
  match n with
  | 0 => id
  | 1 => g
  | n + 2 => g ∘ (g_pow (n + 1))

/-- Main theorem -/
theorem g_pow_6_eq_id (v : ℝ × ℝ × ℝ) (h1 : v ≠ (0, 0, 0)) 
    (h2 : ∃ (n : ℕ+), g_pow n v = v) : 
  g_pow 6 v = v := by
  sorry

end NUMINAMATH_CALUDE_g_pow_6_eq_id_l4100_410091


namespace NUMINAMATH_CALUDE_max_value_log_product_l4100_410067

/-- The maximum value of lg a · lg c given the conditions -/
theorem max_value_log_product (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq1 : Real.log a / Real.log 10 + Real.log c / Real.log b = 3)
  (eq2 : Real.log b / Real.log 10 + Real.log c / Real.log a = 4) :
  (∃ (x : ℝ), (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ x) ∧
  (∀ (y : ℝ), (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ y → 16/3 ≤ y) :=
sorry

end NUMINAMATH_CALUDE_max_value_log_product_l4100_410067


namespace NUMINAMATH_CALUDE_relay_team_selection_l4100_410054

/-- The number of ways to select and arrange 4 sprinters out of 6 for a 4×100m relay, 
    given that one sprinter cannot run the first leg and another cannot run the fourth leg. -/
theorem relay_team_selection (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n.factorial / (n - k).factorial) -     -- Total arrangements without restrictions
  2 * ((n - 1).factorial / (n - k).factorial) +  -- Subtracting arrangements with A or B in wrong position
  ((n - 2).factorial / (n - k).factorial) -- Adding back arrangements with both A and B in wrong positions
  = 252 := by sorry

end NUMINAMATH_CALUDE_relay_team_selection_l4100_410054


namespace NUMINAMATH_CALUDE_jennifer_score_l4100_410099

/-- Calculates the score for a modified AMC 8 contest -/
def calculateScore (totalQuestions correctAnswers incorrectAnswers unansweredQuestions : ℕ) : ℤ :=
  2 * correctAnswers - incorrectAnswers

/-- Proves that Jennifer's score in the modified AMC 8 contest is 20 points -/
theorem jennifer_score :
  let totalQuestions : ℕ := 30
  let correctAnswers : ℕ := 15
  let incorrectAnswers : ℕ := 10
  let unansweredQuestions : ℕ := 5
  calculateScore totalQuestions correctAnswers incorrectAnswers unansweredQuestions = 20 := by
  sorry

#eval calculateScore 30 15 10 5

end NUMINAMATH_CALUDE_jennifer_score_l4100_410099


namespace NUMINAMATH_CALUDE_experts_win_probability_l4100_410049

/-- The probability of Experts winning a single round -/
def p_win : ℝ := 0.6

/-- The probability of Experts losing a single round -/
def p_lose : ℝ := 1 - p_win

/-- The current score of Experts -/
def experts_score : ℕ := 3

/-- The current score of Viewers -/
def viewers_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability that Experts will win the game from the current position -/
def experts_win_prob : ℝ :=
  p_win ^ 3 + 3 * p_win ^ 3 * p_lose

theorem experts_win_probability :
  experts_win_prob = 0.4752 :=
sorry

end NUMINAMATH_CALUDE_experts_win_probability_l4100_410049


namespace NUMINAMATH_CALUDE_initial_shirts_l4100_410023

theorem initial_shirts (S : ℕ) : S + 4 = 16 → S = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_shirts_l4100_410023


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l4100_410078

/-- Given two positive integers with a ratio of 4:5 and HCF of 4, their LCM is 80 -/
theorem lcm_of_ratio_and_hcf (a b : ℕ+) (h_ratio : a.val * 5 = b.val * 4) 
  (h_hcf : Nat.gcd a.val b.val = 4) : Nat.lcm a.val b.val = 80 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l4100_410078


namespace NUMINAMATH_CALUDE_tangerine_count_l4100_410055

theorem tangerine_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  initial = 10 → added = 6 → total = initial + added → total = 16 := by
sorry

end NUMINAMATH_CALUDE_tangerine_count_l4100_410055


namespace NUMINAMATH_CALUDE_washer_dryer_cost_washer_dryer_cost_proof_l4100_410075

/-- The total cost of a washer-dryer combination is 1200 dollars, given that the washer costs 710 dollars and is 220 dollars more expensive than the dryer. -/
theorem washer_dryer_cost : ℕ → ℕ → ℕ → Prop :=
  fun washer_cost dryer_cost total_cost =>
    washer_cost = 710 ∧
    washer_cost = dryer_cost + 220 ∧
    total_cost = washer_cost + dryer_cost →
    total_cost = 1200

/-- Proof of the washer-dryer cost theorem -/
theorem washer_dryer_cost_proof : washer_dryer_cost 710 490 1200 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_washer_dryer_cost_proof_l4100_410075
