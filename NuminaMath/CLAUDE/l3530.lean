import Mathlib

namespace NUMINAMATH_CALUDE_largest_term_is_115_div_3_l3530_353098

/-- An arithmetic sequence of 5 terms satisfying specific conditions -/
structure ArithmeticSequence where
  terms : Fin 5 → ℚ
  is_arithmetic : ∀ i j k : Fin 5, terms k - terms j = terms j - terms i
  sum_is_100 : (Finset.univ.sum terms) = 100
  ratio_condition : (terms 2 + terms 3 + terms 4) = (1/7) * (terms 0 + terms 1)

/-- The largest term in the arithmetic sequence is 115/3 -/
theorem largest_term_is_115_div_3 (seq : ArithmeticSequence) : seq.terms 4 = 115/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_term_is_115_div_3_l3530_353098


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3530_353041

/-- The function f(x) = x^4 - 4x -/
def f (x : ℝ) : ℝ := x^4 - 4*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 4*x^3 - 4

theorem tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), f x = y ∧ f_derivative x = 0 → x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l3530_353041


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3530_353018

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3530_353018


namespace NUMINAMATH_CALUDE_min_handshakes_35_people_l3530_353054

/-- Represents a gathering of people and their handshakes -/
structure Gathering where
  people : ℕ
  handshakes_per_person : ℕ

/-- Calculates the total number of handshakes in a gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  g.people * g.handshakes_per_person / 2

/-- Theorem: In a gathering of 35 people where each person shakes hands with 
    exactly 3 others, the minimum possible number of handshakes is 105 -/
theorem min_handshakes_35_people : 
  ∃ (g : Gathering), g.people = 35 ∧ g.handshakes_per_person = 6 ∧ total_handshakes g = 105 := by
  sorry

#check min_handshakes_35_people

end NUMINAMATH_CALUDE_min_handshakes_35_people_l3530_353054


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solvable_inequality_l3530_353070

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) < 8
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -5/2 < x ∧ x < 3/2} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_solvable_inequality :
  {m : ℝ | ∃ x, f x ≤ |3*m + 1|} = 
    {m : ℝ | m ≤ -5/3 ∨ m ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solvable_inequality_l3530_353070


namespace NUMINAMATH_CALUDE_A_equals_B_l3530_353097

def A : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def B : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l3530_353097


namespace NUMINAMATH_CALUDE_existence_of_x_y_satisfying_conditions_l3530_353047

theorem existence_of_x_y_satisfying_conditions : ∃ (x y : ℝ), x = y + 1 ∧ x^4 = y^4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_y_satisfying_conditions_l3530_353047


namespace NUMINAMATH_CALUDE_first_month_sale_l3530_353099

/-- Given the sales data for a grocer over 6 months, prove that the first month's sale was 5420 --/
theorem first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) 
  (h1 : sale2 = 5660)
  (h2 : sale3 = 6200)
  (h3 : sale4 = 6350)
  (h4 : sale5 = 6500)
  (h5 : sale6 = 6470)
  (h6 : average = 6100) :
  let total := 6 * average
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  total - known_sales = 5420 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l3530_353099


namespace NUMINAMATH_CALUDE_concert_friends_count_l3530_353089

theorem concert_friends_count : 
  ∀ (P : ℝ), P > 0 → 
  ∃ (F : ℕ), 
    (F : ℝ) * P = ((F + 1 : ℕ) : ℝ) * P * (1 - 0.25) ∧ 
    F = 3 := by
  sorry

end NUMINAMATH_CALUDE_concert_friends_count_l3530_353089


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l3530_353022

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30) 
  (h2 : coffee = 15) 
  (h3 : tea = 13) 
  (h4 : both = 8) : 
  total - (coffee + tea - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l3530_353022


namespace NUMINAMATH_CALUDE_school_sampling_theorem_l3530_353034

/-- Represents the types of schools --/
inductive SchoolType
  | Primary
  | Middle
  | University

/-- Represents the count of each school type --/
structure SchoolCounts where
  primary : Nat
  middle : Nat
  university : Nat

/-- Represents the result of stratified sampling --/
structure SamplingResult where
  primary : Nat
  middle : Nat
  university : Nat

/-- Calculates the stratified sampling result --/
def stratifiedSample (counts : SchoolCounts) (totalSample : Nat) : SamplingResult :=
  { primary := counts.primary * totalSample / (counts.primary + counts.middle + counts.university),
    middle := counts.middle * totalSample / (counts.primary + counts.middle + counts.university),
    university := counts.university * totalSample / (counts.primary + counts.middle + counts.university) }

/-- Calculates the probability of selecting two primary schools --/
def probabilityTwoPrimary (sample : SamplingResult) : Rat :=
  (sample.primary * (sample.primary - 1)) / (2 * (sample.primary + sample.middle + sample.university) * (sample.primary + sample.middle + sample.university - 1))

theorem school_sampling_theorem (counts : SchoolCounts) (h : counts = { primary := 21, middle := 14, university := 7 }) :
  let sample := stratifiedSample counts 6
  sample = { primary := 3, middle := 2, university := 1 } ∧
  probabilityTwoPrimary sample = 1/5 := by
  sorry

#check school_sampling_theorem

end NUMINAMATH_CALUDE_school_sampling_theorem_l3530_353034


namespace NUMINAMATH_CALUDE_function_to_polynomial_l3530_353094

-- Define the property that (f(x))^n is a polynomial for all n ≥ 2
def is_power_polynomial (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ p : Polynomial ℝ, ∀ x, (f x)^n = p.eval x

-- Theorem statement
theorem function_to_polynomial 
  (f : ℝ → ℝ) 
  (h : is_power_polynomial f) : 
  ∃ p : Polynomial ℝ, ∀ x, f x = p.eval x :=
sorry

end NUMINAMATH_CALUDE_function_to_polynomial_l3530_353094


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_each_set_l3530_353077

def total_cards : ℕ := 15
def cybil_cards : ℕ := 6
def ronda_cards : ℕ := 9
def cards_drawn : ℕ := 3

theorem probability_at_least_one_from_each_set :
  let p := 1 - (Nat.choose ronda_cards cards_drawn) / (Nat.choose total_cards cards_drawn) -
           (Nat.choose cybil_cards cards_drawn) / (Nat.choose total_cards cards_drawn)
  p = 351 / 455 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_each_set_l3530_353077


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l3530_353025

theorem sqrt_abs_sum_zero_implies_sum_power (a b : ℝ) :
  Real.sqrt (a - 2) + |b + 1| = 0 → (a + b)^2023 = 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l3530_353025


namespace NUMINAMATH_CALUDE_physical_exercise_test_results_l3530_353051

/-- Represents a school in the physical exercise test --/
structure School where
  name : String
  total_students : Nat
  sampled_students : Nat
  average_score : Float
  median_score : Float
  mode_score : Nat

/-- Represents the score distribution for a school --/
structure ScoreDistribution where
  school : School
  scores : List (Nat × Nat)  -- (score_range_start, count)

theorem physical_exercise_test_results 
  (school_a school_b : School)
  (dist_a : ScoreDistribution)
  (h1 : school_a.name = "School A")
  (h2 : school_b.name = "School B")
  (h3 : school_a.total_students = 180)
  (h4 : school_b.total_students = 180)
  (h5 : school_a.sampled_students = 30)
  (h6 : school_b.sampled_students = 30)
  (h7 : school_a.average_score = 96.35)
  (h8 : school_a.mode_score = 99)
  (h9 : school_b.average_score = 95.85)
  (h10 : school_b.median_score = 97.5)
  (h11 : school_b.mode_score = 99)
  (h12 : dist_a.school = school_a)
  (h13 : dist_a.scores = [(90, 2), (92, 3), (94, 5), (96, 10), (98, 10)]) :
  school_a.median_score = 96.5 ∧ 
  (((school_a.total_students * 20) / 30 : Nat) * 2 - 100 = 140) := by
  sorry

end NUMINAMATH_CALUDE_physical_exercise_test_results_l3530_353051


namespace NUMINAMATH_CALUDE_olivia_money_distribution_l3530_353076

theorem olivia_money_distribution (olivia_initial : ℕ) (sisters_count : ℕ) (sister_initial : ℕ) (amount_given : ℕ) :
  olivia_initial = 20 →
  sisters_count = 4 →
  sister_initial = 10 →
  amount_given = 2 →
  (olivia_initial - sisters_count * amount_given) = 
  (sister_initial + amount_given) :=
by
  sorry

end NUMINAMATH_CALUDE_olivia_money_distribution_l3530_353076


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3530_353017

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : a 1 = 2 * a 8 - 3 * a 4)
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) :
  let S : ℕ → ℝ := λ n ↦ (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2
  S 8 / S 16 = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3530_353017


namespace NUMINAMATH_CALUDE_three_teacher_student_pairs_arrangements_l3530_353043

def teacher_student_arrangements (n : ℕ) : ℕ :=
  n.factorial * (2^n)

theorem three_teacher_student_pairs_arrangements :
  teacher_student_arrangements 3 = 48 := by
sorry

end NUMINAMATH_CALUDE_three_teacher_student_pairs_arrangements_l3530_353043


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_200_l3530_353079

/-- The sum of divisors of a natural number n -/
def sumOfDivisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number n -/
def largestPrimeFactor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_200 :
  largestPrimeFactor (sumOfDivisors 200) = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_200_l3530_353079


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3530_353028

theorem two_numbers_problem :
  ∃ (x y : ℤ),
    x + y = 44 ∧
    y < 0 ∧
    (x - y) * 100 = y * y ∧
    x = 264 ∧
    y = -220 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3530_353028


namespace NUMINAMATH_CALUDE_y_value_l3530_353055

theorem y_value (y : ℝ) (h : (9 : ℝ) / y^2 = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3530_353055


namespace NUMINAMATH_CALUDE_optimal_bicycle_point_l3530_353005

/-- Represents the problem of finding the optimal point to leave a bicycle --/
theorem optimal_bicycle_point 
  (total_distance : ℝ) 
  (cycling_speed walking_speed : ℝ) 
  (h1 : total_distance = 30) 
  (h2 : cycling_speed = 20) 
  (h3 : walking_speed = 5) :
  ∃ (x : ℝ), 
    x = 5 ∧ 
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_distance → 
      max ((total_distance / 2 - y) / cycling_speed + y / walking_speed)
          (y / walking_speed + (total_distance / 2 - y) / cycling_speed)
      ≥ 
      max ((total_distance / 2 - x) / cycling_speed + x / walking_speed)
          (x / walking_speed + (total_distance / 2 - x) / cycling_speed)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_bicycle_point_l3530_353005


namespace NUMINAMATH_CALUDE_tesseract_parallel_edges_l3530_353009

/-- A tesseract is a four-dimensional hypercube -/
structure Tesseract where
  dim : Nat
  edges : Nat

/-- The number of pairs of parallel edges in a tesseract -/
def parallel_edge_pairs (t : Tesseract) : Nat :=
  sorry

/-- Theorem: A tesseract with 32 edges has 36 pairs of parallel edges -/
theorem tesseract_parallel_edges (t : Tesseract) (h1 : t.dim = 4) (h2 : t.edges = 32) :
  parallel_edge_pairs t = 36 := by
  sorry

end NUMINAMATH_CALUDE_tesseract_parallel_edges_l3530_353009


namespace NUMINAMATH_CALUDE_median_of_special_sequence_l3530_353065

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_sequence : 
  let N : ℕ := sequence_sum 200
  let median_index : ℕ := N / 2
  let cumulative_count (n : ℕ) := sequence_sum n
  ∃ (n : ℕ), 
    cumulative_count n ≥ median_index ∧ 
    cumulative_count (n - 1) < median_index ∧
    n = 141 :=
by sorry

end NUMINAMATH_CALUDE_median_of_special_sequence_l3530_353065


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3530_353033

-- Define the function f
def f (x t : ℝ) : ℝ := |x - 1| + |x - t|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 > 2} = {x : ℝ | x < (1/2) ∨ x > (5/2)} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ t x : ℝ, t ∈ [1, 2] → x ∈ [-1, 3] → f x t ≥ a + x) → a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3530_353033


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3530_353016

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence with positive terms, if the third term is 8 and the seventh term is 18, then the fifth term is 12. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_third : a 3 = 8)
  (h_seventh : a 7 = 18) :
  a 5 = 12 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3530_353016


namespace NUMINAMATH_CALUDE_infinitely_many_all_off_infinitely_many_never_all_off_l3530_353062

-- Define the lamp state as a list of booleans
def LampState := List Bool

-- Define the state modification function
def modifyState (state : LampState) : LampState :=
  sorry

-- Define the initial state
def initialState (n : Nat) : LampState :=
  sorry

-- Define a predicate to check if all lamps are off
def allLampsOff (state : LampState) : Prop :=
  sorry

-- Define a function to evolve the state
def evolveState (n : Nat) (steps : Nat) : LampState :=
  sorry

theorem infinitely_many_all_off :
  ∃ S : Set Nat, (∀ n ∈ S, n ≥ 2) ∧ Set.Infinite S ∧
  ∀ n ∈ S, ∃ k : Nat, allLampsOff (evolveState n k) :=
sorry

theorem infinitely_many_never_all_off :
  ∃ T : Set Nat, (∀ n ∈ T, n ≥ 2) ∧ Set.Infinite T ∧
  ∀ n ∈ T, ∀ k : Nat, ¬(allLampsOff (evolveState n k)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_all_off_infinitely_many_never_all_off_l3530_353062


namespace NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l3530_353012

theorem hcf_of_ratio_and_lcm (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → 
  Nat.lcm a b = 80 → 
  Nat.gcd a b = 2 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l3530_353012


namespace NUMINAMATH_CALUDE_three_digit_permutation_sum_divisible_by_37_l3530_353050

theorem three_digit_permutation_sum_divisible_by_37 (a b c : ℕ) 
  (h1 : 0 < a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) :
  37 ∣ (100*a + 10*b + c) + 
       (100*a + 10*c + b) + 
       (100*b + 10*a + c) + 
       (100*b + 10*c + a) + 
       (100*c + 10*a + b) + 
       (100*c + 10*b + a) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_permutation_sum_divisible_by_37_l3530_353050


namespace NUMINAMATH_CALUDE_total_raised_is_100_l3530_353088

/-- A fundraising event with raffle tickets and donations -/
structure Fundraiser where
  ticket_count : ℕ
  ticket_price : ℚ
  donation_15_count : ℕ
  donation_20_count : ℕ

/-- Calculate the total amount raised from a fundraiser -/
def total_raised (f : Fundraiser) : ℚ :=
  f.ticket_count * f.ticket_price +
  f.donation_15_count * 15 +
  f.donation_20_count * 20

/-- The specific fundraiser described in the problem -/
def charity_fundraiser : Fundraiser :=
  { ticket_count := 25
  , ticket_price := 2
  , donation_15_count := 2
  , donation_20_count := 1
  }

/-- Theorem stating that the total amount raised is $100.00 -/
theorem total_raised_is_100 :
  total_raised charity_fundraiser = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_raised_is_100_l3530_353088


namespace NUMINAMATH_CALUDE_exists_functions_satisfying_equations_l3530_353000

/-- A function defined on non-zero real numbers -/
def NonZeroRealFunction := {f : ℝ → ℝ // ∀ x ≠ 0, f x ≠ 0}

/-- The property that f and g satisfy the given equations -/
def SatisfiesEquations (f g : NonZeroRealFunction) : Prop :=
  ∀ x ≠ 0, f.val x + g.val (1/x) = x ∧ g.val x + f.val (1/x) = 1/x

theorem exists_functions_satisfying_equations :
  ∃ f g : NonZeroRealFunction, SatisfiesEquations f g ∧ f.val 1 = 1/2 ∧ g.val 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_exists_functions_satisfying_equations_l3530_353000


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3530_353048

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 4 →
  downstream_distance = 9.6 →
  downstream_time = 24 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 20 ∧ (boat_speed + current_speed) * downstream_time = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3530_353048


namespace NUMINAMATH_CALUDE_h3po4_naoh_reaction_results_l3530_353053

/-- Represents a chemical compound in a reaction --/
structure Compound where
  name : String
  moles : ℝ

/-- Represents a balanced chemical equation --/
structure BalancedEquation where
  reactant1 : Compound
  reactant2 : Compound
  product1 : Compound
  product2 : Compound
  stoichiometry : ℝ

/-- Determines the limiting reactant and calculates reaction results --/
def reactionResults (eq : BalancedEquation) : Compound × Compound × Compound := sorry

/-- Theorem stating the reaction results for H3PO4 and NaOH --/
theorem h3po4_naoh_reaction_results :
  let h3po4 := Compound.mk "H3PO4" 2.5
  let naoh := Compound.mk "NaOH" 3
  let equation := BalancedEquation.mk h3po4 naoh (Compound.mk "Na3PO4" 0) (Compound.mk "H2O" 0) 3
  let (h2o_formed, limiting_reactant, unreacted_h3po4) := reactionResults equation
  h2o_formed.moles = 3 ∧
  limiting_reactant.name = "NaOH" ∧
  unreacted_h3po4.moles = 1.5 := by sorry

end NUMINAMATH_CALUDE_h3po4_naoh_reaction_results_l3530_353053


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3530_353044

/-- The length of the major axis of an ellipse with given foci and tangent line -/
theorem ellipse_major_axis_length : ∀ (F₁ F₂ : ℝ × ℝ) (y₀ : ℝ),
  F₁ = (4, 10) →
  F₂ = (34, 40) →
  y₀ = -5 →
  ∃ (X : ℝ × ℝ), X.2 = y₀ ∧ 
    (∀ (P : ℝ × ℝ), P.2 = y₀ → 
      Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
      Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2) ≥ 
      Real.sqrt ((X.1 - F₁.1)^2 + (X.2 - F₁.2)^2) + 
      Real.sqrt ((X.1 - F₂.1)^2 + (X.2 - F₂.2)^2)) →
  30 * Real.sqrt 5 = 
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - (2 * y₀ - F₁.2))^2) := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3530_353044


namespace NUMINAMATH_CALUDE_symmetry_axis_of_sin_cos_function_l3530_353080

theorem symmetry_axis_of_sin_cos_function :
  ∃ (x : ℝ), x = π / 12 ∧
  ∀ (y : ℝ), y = Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) →
  (∀ (t : ℝ), y = Real.sin (2 * (x + t)) - Real.sqrt 3 * Real.cos (2 * (x + t)) ↔
               y = Real.sin (2 * (x - t)) - Real.sqrt 3 * Real.cos (2 * (x - t))) :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_sin_cos_function_l3530_353080


namespace NUMINAMATH_CALUDE_expand_expression_l3530_353090

theorem expand_expression (y : ℝ) : (7 * y + 12) * (3 * y) = 21 * y^2 + 36 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3530_353090


namespace NUMINAMATH_CALUDE_xy_value_l3530_353086

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3530_353086


namespace NUMINAMATH_CALUDE_car_wash_earnings_l3530_353032

/-- Proves that a car wash company cleaning 80 cars per day at $5 per car will earn $2000 in 5 days -/
theorem car_wash_earnings 
  (cars_per_day : ℕ) 
  (price_per_car : ℕ) 
  (num_days : ℕ) 
  (h1 : cars_per_day = 80) 
  (h2 : price_per_car = 5) 
  (h3 : num_days = 5) : 
  cars_per_day * price_per_car * num_days = 2000 := by
  sorry

#check car_wash_earnings

end NUMINAMATH_CALUDE_car_wash_earnings_l3530_353032


namespace NUMINAMATH_CALUDE_plumber_distribution_theorem_l3530_353046

/-- The number of ways to distribute n plumbers to k houses -/
def distribute_plumbers (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else Nat.choose n 2 * (Nat.factorial k)

theorem plumber_distribution_theorem :
  distribute_plumbers 4 3 = Nat.choose 4 2 * (Nat.factorial 3) :=
sorry

end NUMINAMATH_CALUDE_plumber_distribution_theorem_l3530_353046


namespace NUMINAMATH_CALUDE_joan_socks_total_l3530_353003

theorem joan_socks_total (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) :
  white_socks = (2 : ℚ) / 3 * total_socks →
  blue_socks = total_socks - white_socks →
  blue_socks = 60 →
  total_socks = 180 := by
  sorry

end NUMINAMATH_CALUDE_joan_socks_total_l3530_353003


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_bases_l3530_353021

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Diameter of the inscribed circle -/
  diameter : ℝ
  /-- Length of the leg (non-parallel side) -/
  leg : ℝ
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- The diameter is positive -/
  diameter_pos : diameter > 0
  /-- The leg is longer than the radius -/
  leg_gt_radius : leg > diameter / 2

/-- Theorem: For an isosceles trapezoid circumscribed around a circle with diameter 15 and leg length 17, the lengths of its bases are 25 and 9 -/
theorem isosceles_trapezoid_bases 
  (t : IsoscelesTrapezoid) 
  (h1 : t.diameter = 15) 
  (h2 : t.leg = 17) : 
  t.longerBase = 25 ∧ t.shorterBase = 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_bases_l3530_353021


namespace NUMINAMATH_CALUDE_leila_money_left_l3530_353083

def money_left_after_shopping (initial_money sweater_cost jewelry_cost : ℕ) : ℕ :=
  initial_money - (sweater_cost + jewelry_cost)

theorem leila_money_left :
  ∀ (sweater_cost : ℕ),
    sweater_cost = 40 →
    ∀ (initial_money : ℕ),
      initial_money = 4 * sweater_cost →
      ∀ (jewelry_cost : ℕ),
        jewelry_cost = sweater_cost + 60 →
        money_left_after_shopping initial_money sweater_cost jewelry_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_leila_money_left_l3530_353083


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l3530_353075

/-- Race finish time difference problem -/
theorem race_finish_time_difference 
  (malcolm_speed : ℕ) 
  (joshua_speed : ℕ) 
  (race_distance : ℕ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
sorry

end NUMINAMATH_CALUDE_race_finish_time_difference_l3530_353075


namespace NUMINAMATH_CALUDE_specific_divisors_of_20_pow_30_l3530_353031

def count_specific_divisors (n : ℕ) : ℕ :=
  let total_divisors := (60 + 1) * (30 + 1)
  let divisors_less_than_sqrt := (total_divisors - 1) / 2
  let divisors_of_sqrt := (30 + 1) * (15 + 1)
  divisors_less_than_sqrt - divisors_of_sqrt + 1

theorem specific_divisors_of_20_pow_30 :
  count_specific_divisors 20 = 450 := by
  sorry

end NUMINAMATH_CALUDE_specific_divisors_of_20_pow_30_l3530_353031


namespace NUMINAMATH_CALUDE_roots_and_m_value_l3530_353008

theorem roots_and_m_value (a b c m : ℝ) : 
  (a + b = 4 ∧ a * b = m) →  -- roots of x^2 - 4x + m = 0
  (b + c = 8 ∧ b * c = 5 * m) →  -- roots of x^2 - 8x + 5m = 0
  m = 0 ∨ m = 3 := by
sorry

end NUMINAMATH_CALUDE_roots_and_m_value_l3530_353008


namespace NUMINAMATH_CALUDE_chef_nut_purchase_l3530_353066

/-- The weight of almonds bought by the chef in kilograms -/
def almond_weight : ℝ := 0.14

/-- The weight of pecans bought by the chef in kilograms -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms -/
def total_weight : ℝ := almond_weight + pecan_weight

/-- Theorem stating that the total weight of nuts bought by the chef is 0.52 kilograms -/
theorem chef_nut_purchase : total_weight = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_chef_nut_purchase_l3530_353066


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l3530_353019

/-- The number of fruit types available to Joe -/
def num_fruits : ℕ := 4

/-- The number of meals Joe has per day -/
def num_meals : ℕ := 3

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruits

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_one_fruit ^ num_meals

/-- The probability of not eating at least two different kinds of fruits -/
def prob_not_varied : ℚ := num_fruits * prob_same_fruit

/-- The probability of eating at least two different kinds of fruits -/
def prob_varied : ℚ := 1 - prob_not_varied

theorem joe_fruit_probability : prob_varied = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l3530_353019


namespace NUMINAMATH_CALUDE_fourth_day_earning_l3530_353029

/-- Represents the daily earnings of a mechanic for a week -/
def MechanicEarnings : Type := Fin 7 → ℝ

/-- The average earning for the first 4 days is 18 -/
def avg_first_four (e : MechanicEarnings) : Prop :=
  (e 0 + e 1 + e 2 + e 3) / 4 = 18

/-- The average earning for the last 4 days is 22 -/
def avg_last_four (e : MechanicEarnings) : Prop :=
  (e 3 + e 4 + e 5 + e 6) / 4 = 22

/-- The average earning for the whole week is 21 -/
def avg_whole_week (e : MechanicEarnings) : Prop :=
  (e 0 + e 1 + e 2 + e 3 + e 4 + e 5 + e 6) / 7 = 21

/-- The theorem stating that given the conditions, the earning on the fourth day is 13 -/
theorem fourth_day_earning (e : MechanicEarnings) 
  (h1 : avg_first_four e) 
  (h2 : avg_last_four e) 
  (h3 : avg_whole_week e) : 
  e 3 = 13 := by sorry

end NUMINAMATH_CALUDE_fourth_day_earning_l3530_353029


namespace NUMINAMATH_CALUDE_debate_team_selection_l3530_353068

def total_students : ℕ := 9
def students_to_select : ℕ := 4
def specific_students : ℕ := 2

def select_with_condition (n k m : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - m) k

theorem debate_team_selection :
  select_with_condition total_students students_to_select specific_students = 91 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_selection_l3530_353068


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_l3530_353042

theorem sqrt_meaningful_iff (x : ℝ) : Real.sqrt (x - 1/2) ≥ 0 ↔ x ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_l3530_353042


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l3530_353087

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l3530_353087


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3530_353078

def i : ℂ := Complex.I

theorem z_in_first_quadrant : ∃ z : ℂ, 
  (1 + i) * z = 1 - 2 * i^3 ∧ 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3530_353078


namespace NUMINAMATH_CALUDE_a_2_value_a_general_formula_l3530_353027

def sequence_a (n : ℕ) : ℝ := sorry

def sequence_S (n : ℕ) : ℝ := sorry

axiom a_relation : ∀ n : ℕ, sequence_a (n + 1) = 2 * sequence_S n + 6

axiom a_1 : sequence_a 1 = 6

theorem a_2_value : sequence_a 2 = 18 := by sorry

theorem a_general_formula : ∀ n : ℕ, n ≥ 1 → sequence_a n = 2 * 3^n := by sorry

end NUMINAMATH_CALUDE_a_2_value_a_general_formula_l3530_353027


namespace NUMINAMATH_CALUDE_sum_of_ninth_powers_l3530_353060

theorem sum_of_ninth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^9 + b^9 = 76 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ninth_powers_l3530_353060


namespace NUMINAMATH_CALUDE_cars_return_to_start_l3530_353011

/-- Represents a car on a circular race track -/
structure Car where
  position : ℝ  -- Position on the track (0 ≤ position < track_length)
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race -/
structure RaceState where
  track_length : ℝ
  cars : Vector Car n
  time : ℝ

/-- The race system evolves over time -/
def evolve_race (initial_state : RaceState) (t : ℝ) : RaceState :=
  sorry

/-- Predicate to check if all cars are at their initial positions -/
def all_cars_at_initial_positions (initial_state : RaceState) (current_state : RaceState) : Prop :=
  sorry

/-- Main theorem: There exists a time when all cars return to their initial positions -/
theorem cars_return_to_start {n : ℕ} (initial_state : RaceState) :
  ∃ t : ℝ, all_cars_at_initial_positions initial_state (evolve_race initial_state t) :=
  sorry

end NUMINAMATH_CALUDE_cars_return_to_start_l3530_353011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3530_353064

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_common_difference 
  (a₁ : ℝ) 
  (d : ℝ) 
  (h_d_nonzero : d ≠ 0) 
  (h_sum : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 5 = 13)
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ 
    arithmetic_sequence a₁ d 2 = arithmetic_sequence a₁ d 1 * r ∧ 
    arithmetic_sequence a₁ d 5 = arithmetic_sequence a₁ d 2 * r) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3530_353064


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_l3530_353056

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / 3000 := by
  sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_l3530_353056


namespace NUMINAMATH_CALUDE_cody_game_expense_l3530_353036

theorem cody_game_expense (initial_amount birthday_gift final_amount : ℕ) 
  (h1 : initial_amount = 45)
  (h2 : birthday_gift = 9)
  (h3 : final_amount = 35) :
  initial_amount + birthday_gift - final_amount = 19 :=
by sorry

end NUMINAMATH_CALUDE_cody_game_expense_l3530_353036


namespace NUMINAMATH_CALUDE_some_ounce_glass_size_l3530_353020

theorem some_ounce_glass_size (total_water : ℕ) (five_ounce_filled : ℕ) (some_ounce_filled : ℕ) (four_ounce_remaining : ℕ) :
  total_water = 122 →
  five_ounce_filled = 6 →
  some_ounce_filled = 4 →
  four_ounce_remaining = 15 →
  ∃ (some_ounce_size : ℕ),
    total_water = five_ounce_filled * 5 + some_ounce_filled * some_ounce_size + four_ounce_remaining * 4 ∧
    some_ounce_size = 8 :=
by sorry

end NUMINAMATH_CALUDE_some_ounce_glass_size_l3530_353020


namespace NUMINAMATH_CALUDE_unique_matching_number_l3530_353038

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  t_range : tens ≥ 0 ∧ tens ≤ 9
  u_range : units ≥ 0 ∧ units ≤ 9

/-- Checks if two ThreeDigitNumbers match in exactly one digit place -/
def matchesOneDigit (a b : ThreeDigitNumber) : Prop :=
  (a.hundreds = b.hundreds ∧ a.tens ≠ b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens = b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens ≠ b.tens ∧ a.units = b.units)

/-- The theorem to be proved -/
theorem unique_matching_number : ∃! n : ThreeDigitNumber,
  matchesOneDigit n ⟨1, 0, 9, by sorry, by sorry, by sorry⟩ ∧
  matchesOneDigit n ⟨7, 0, 4, by sorry, by sorry, by sorry⟩ ∧
  matchesOneDigit n ⟨1, 2, 4, by sorry, by sorry, by sorry⟩ ∧
  n = ⟨7, 2, 9, by sorry, by sorry, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_unique_matching_number_l3530_353038


namespace NUMINAMATH_CALUDE_quadratic_intercept_l3530_353067

/-- A quadratic function with vertex (5,10) and one x-intercept at (0,0) has its other x-intercept at x = 10 -/
theorem quadratic_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 10 - a * (x - 5)^2) →  -- vertex form with vertex (5,10)
  (0^2 * a + 0 * b + c = 0) →                        -- (0,0) is an x-intercept
  (∃ x, x ≠ 0 ∧ x^2 * a + x * b + c = 0 ∧ x = 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_intercept_l3530_353067


namespace NUMINAMATH_CALUDE_binomial_seven_choose_two_l3530_353071

theorem binomial_seven_choose_two : (7 : ℕ).choose 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_choose_two_l3530_353071


namespace NUMINAMATH_CALUDE_fib_2n_eq_sum_squares_l3530_353095

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: For a Fibonacci sequence, f_{2n} = f_{n-1}^2 + f_n^2 -/
theorem fib_2n_eq_sum_squares (n : ℕ) : fib (2 * n) = (fib (n - 1))^2 + (fib n)^2 := by
  sorry

end NUMINAMATH_CALUDE_fib_2n_eq_sum_squares_l3530_353095


namespace NUMINAMATH_CALUDE_prob_one_or_two_pascal_l3530_353096

/-- Pascal's Triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Count of elements in first n rows of Pascal's Triangle -/
def count_elements (n : ℕ) : ℕ := (n * (n + 1)) / 2

/-- Count of ones in first n rows of Pascal's Triangle -/
def count_ones (n : ℕ) : ℕ := if n = 0 then 1 else 2 * n - 1

/-- Count of twos in first n rows of Pascal's Triangle -/
def count_twos (n : ℕ) : ℕ := if n ≤ 2 then 0 else 2 * (n - 2)

/-- Probability of selecting 1 or 2 from first 20 rows of Pascal's Triangle -/
theorem prob_one_or_two_pascal : 
  (count_ones 20 + count_twos 20 : ℚ) / count_elements 20 = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_prob_one_or_two_pascal_l3530_353096


namespace NUMINAMATH_CALUDE_tangent_parallel_points_tangent_equations_l3530_353023

/-- The function f(x) = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The slope of the line y = 4x - 1 -/
def m : ℝ := 4

/-- The set of points where the tangent line is parallel to y = 4x - 1 -/
def tangent_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f' p.1 = m ∧ p.2 = f p.1}

/-- The equation of the tangent line at a point (a, f(a)) -/
def tangent_line (a : ℝ) (x y : ℝ) : Prop :=
  y - f a = f' a * (x - a)

theorem tangent_parallel_points :
  tangent_points = {(1, 0), (-1, -4)} :=
sorry

theorem tangent_equations (a : ℝ) (h : (a, f a) ∈ tangent_points) :
  (∀ x y, tangent_line a x y ↔ (4 * x - y - 4 = 0 ∨ 4 * x - y = 0)) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_tangent_equations_l3530_353023


namespace NUMINAMATH_CALUDE_function_non_negative_l3530_353040

theorem function_non_negative 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * (deriv f x) > 0) : 
  ∀ x, f x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_non_negative_l3530_353040


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3530_353013

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3530_353013


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_two_lines_l3530_353058

/-- The maximum number of intersection points between circles -/
def max_circle_intersections (n : ℕ) : ℕ := n * (n - 1)

/-- The maximum number of intersection points between circles and lines -/
def max_circle_line_intersections (circles : ℕ) (lines : ℕ) : ℕ :=
  circles * lines * 2

/-- The maximum number of intersection points between lines -/
def max_line_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total maximum number of intersection points -/
def total_max_intersections (circles : ℕ) (lines : ℕ) : ℕ :=
  max_circle_intersections circles +
  max_circle_line_intersections circles lines +
  max_line_intersections lines

theorem max_intersections_three_circles_two_lines :
  total_max_intersections 3 2 = 19 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_two_lines_l3530_353058


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_sine_l3530_353061

/-- An isosceles triangle with a base angle tangent of 2/3 has a vertex angle sine of 12/13 -/
theorem isosceles_triangle_vertex_angle_sine (α β : Real) : 
  -- α is a base angle of the isosceles triangle
  -- β is the vertex angle of the isosceles triangle
  -- The triangle is isosceles
  β = π - 2 * α →
  -- The tangent of the base angle is 2/3
  Real.tan α = 2 / 3 →
  -- The sine of the vertex angle is 12/13
  Real.sin β = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_sine_l3530_353061


namespace NUMINAMATH_CALUDE_function_identification_l3530_353037

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- State the theorem
theorem function_identification (a b c : ℝ) :
  f a b c 0 = 1 ∧ 
  (∃ k m : ℝ, k = 4 * a * 1^3 + 2 * b * 1 ∧ 
              m = a * 1^4 + b * 1^2 + c ∧ 
              k = 1 ∧ 
              m = -1) →
  ∀ x, f a b c x = 5/2 * x^4 - 9/2 * x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_function_identification_l3530_353037


namespace NUMINAMATH_CALUDE_mean_age_of_friends_l3530_353045

theorem mean_age_of_friends (age_group1 : ℕ) (age_group2 : ℕ) 
  (h1 : age_group1 = 12 * 12 + 3)  -- 12 years and 3 months in months
  (h2 : age_group2 = 13 * 12 + 5)  -- 13 years and 5 months in months
  : (3 * age_group1 + 4 * age_group2) / 7 = 155 := by
  sorry

end NUMINAMATH_CALUDE_mean_age_of_friends_l3530_353045


namespace NUMINAMATH_CALUDE_pauls_vertical_distance_l3530_353063

/-- The number of feet Paul travels vertically in a week -/
def vertical_distance_per_week (story : ℕ) (trips_per_day : ℕ) (days_per_week : ℕ) (feet_per_story : ℕ) : ℕ :=
  2 * story * trips_per_day * days_per_week * feet_per_story

/-- Theorem stating the total vertical distance Paul travels in a week -/
theorem pauls_vertical_distance :
  vertical_distance_per_week 5 3 7 10 = 2100 := by
  sorry

#eval vertical_distance_per_week 5 3 7 10

end NUMINAMATH_CALUDE_pauls_vertical_distance_l3530_353063


namespace NUMINAMATH_CALUDE_equal_split_donation_l3530_353072

def total_donation : ℝ := 15000
def donation1 : ℝ := 3500
def donation2 : ℝ := 2750
def donation3 : ℝ := 3870
def donation4 : ℝ := 2475
def num_remaining_homes : ℕ := 4

theorem equal_split_donation :
  let donated_sum := donation1 + donation2 + donation3 + donation4
  let remaining := total_donation - donated_sum
  remaining / num_remaining_homes = 601.25 := by
sorry

end NUMINAMATH_CALUDE_equal_split_donation_l3530_353072


namespace NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l3530_353006

/-- Hyperbola C with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point on the right branch of hyperbola C -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  h_right_branch : 0 < x

/-- Equilateral triangle with vertices on hyperbola -/
structure EquilateralTriangleOnHyperbola (h : Hyperbola) where
  A : PointOnHyperbola h
  B : PointOnHyperbola h
  c : ℝ
  h_equilateral : c^2 = A.x^2 + A.y^2 ∧ c^2 = B.x^2 + B.y^2
  h_side_length : c^2 = h.a^2 + h.b^2

/-- Theorem: Asymptotes of hyperbola C are y = ±x -/
theorem asymptotes_of_hyperbola (h : Hyperbola) 
  (t : EquilateralTriangleOnHyperbola h) :
  ∃ (k : ℝ), k = 1 ∧ 
  (∀ (x y : ℝ), (y = k*x ∨ y = -k*x) ↔ 
    (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
      x'^2/h.a^2 - y'^2/h.b^2 = 1 → 
      x' > δ → |y'/x' - k| < ε ∨ |y'/x' + k| < ε)) :=
sorry

end NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l3530_353006


namespace NUMINAMATH_CALUDE_smallest_congruent_number_l3530_353024

theorem smallest_congruent_number : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 7 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  (n = 281) := by
sorry

end NUMINAMATH_CALUDE_smallest_congruent_number_l3530_353024


namespace NUMINAMATH_CALUDE_monday_temperature_l3530_353026

theorem monday_temperature
  (temp : Fin 5 → ℝ)  -- temperatures for 5 days
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 46)
  (friday_temp : temp 4 = 35)
  (exists_43 : ∃ i, temp i = 43)
  : temp 0 = 43 := by
  sorry

end NUMINAMATH_CALUDE_monday_temperature_l3530_353026


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l3530_353004

theorem company_picnic_attendance 
  (total_employees : ℝ) 
  (total_men : ℝ) 
  (total_women : ℝ) 
  (women_picnic_attendance : ℝ) 
  (total_picnic_attendance : ℝ) 
  (h1 : women_picnic_attendance = 0.4 * total_women)
  (h2 : total_men = 0.3 * total_employees)
  (h3 : total_women = total_employees - total_men)
  (h4 : total_picnic_attendance = 0.34 * total_employees)
  : (total_picnic_attendance - women_picnic_attendance) / total_men = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l3530_353004


namespace NUMINAMATH_CALUDE_probability_above_parabola_l3530_353010

def is_single_digit_positive (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def point_above_parabola (a b : ℕ) : Prop :=
  is_single_digit_positive a ∧ is_single_digit_positive b ∧ b > a * a + b * a

def total_combinations : ℕ := 81

def valid_combinations : ℕ := 7

theorem probability_above_parabola :
  (valid_combinations : ℚ) / total_combinations = 7 / 81 := by sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l3530_353010


namespace NUMINAMATH_CALUDE_unique_single_digit_square_l3530_353092

theorem unique_single_digit_square (A : ℕ) : A < 10 ∧ (10 * A + A) * (10 * A + A) = 5929 ↔ A = 7 := by sorry

end NUMINAMATH_CALUDE_unique_single_digit_square_l3530_353092


namespace NUMINAMATH_CALUDE_car_trip_equation_correct_l3530_353001

/-- Represents a car trip with a break -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  breakDuration : ℝ
  speedBefore : ℝ
  speedAfter : ℝ

/-- The equation representing the relationship between time before break and total distance -/
def tripEquation (trip : CarTrip) (t : ℝ) : Prop :=
  trip.speedBefore * t + trip.speedAfter * (trip.totalTime - trip.breakDuration / 60 - t) = trip.totalDistance

theorem car_trip_equation_correct (trip : CarTrip) : 
  trip.totalDistance = 295 ∧ 
  trip.totalTime = 3.25 ∧ 
  trip.breakDuration = 15 ∧ 
  trip.speedBefore = 85 ∧ 
  trip.speedAfter = 115 → 
  ∃ t, tripEquation trip t ∧ t > 0 ∧ t < trip.totalTime - trip.breakDuration / 60 :=
sorry

end NUMINAMATH_CALUDE_car_trip_equation_correct_l3530_353001


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3530_353007

theorem arithmetic_mean_of_fractions :
  (1 / 3 : ℚ) * (3 / 7 + 5 / 9 + 7 / 11) = 1123 / 2079 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3530_353007


namespace NUMINAMATH_CALUDE_tuesday_spending_multiple_l3530_353039

/-- Represents the spending on sneakers over three days -/
structure SneakerSpending where
  monday : ℕ
  tuesday_multiple : ℕ
  wednesday_multiple : ℕ
  total : ℕ

/-- The spending satisfies the given conditions -/
def valid_spending (s : SneakerSpending) : Prop :=
  s.monday = 60 ∧
  s.wednesday_multiple = 5 ∧
  s.total = 600 ∧
  s.monday + s.monday * s.tuesday_multiple + s.monday * s.wednesday_multiple = s.total

/-- The theorem to be proved -/
theorem tuesday_spending_multiple (s : SneakerSpending) :
  valid_spending s → s.tuesday_multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_spending_multiple_l3530_353039


namespace NUMINAMATH_CALUDE_original_number_proof_l3530_353084

theorem original_number_proof (x : ℕ) : (10 * x + 9) + 2 * x = 633 → x = 52 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3530_353084


namespace NUMINAMATH_CALUDE_decorations_count_l3530_353049

/-- The number of pieces of tinsel in each box -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box -/
def trees_per_box : ℕ := 1

/-- The number of snow globes in each box -/
def globes_per_box : ℕ := 5

/-- The number of families receiving a box -/
def families : ℕ := 11

/-- The number of boxes given to the community center -/
def community_boxes : ℕ := 1

/-- The total number of decorations handed out -/
def total_decorations : ℕ := (tinsel_per_box + trees_per_box + globes_per_box) * (families + community_boxes)

theorem decorations_count : total_decorations = 120 := by
  sorry

end NUMINAMATH_CALUDE_decorations_count_l3530_353049


namespace NUMINAMATH_CALUDE_denise_removed_five_bananas_l3530_353015

/-- The number of bananas Denise removed from the jar -/
def bananas_removed (original : ℕ) (remaining : ℕ) : ℕ := original - remaining

/-- Theorem stating that Denise removed 5 bananas -/
theorem denise_removed_five_bananas :
  bananas_removed 46 41 = 5 := by
  sorry

end NUMINAMATH_CALUDE_denise_removed_five_bananas_l3530_353015


namespace NUMINAMATH_CALUDE_simplify_expression_l3530_353073

theorem simplify_expression : (45 * 2^10) / (15 * 2^5) * 5 = 480 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3530_353073


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3530_353069

/-- Given a geometric sequence where the fourth term is 54 and the fifth term is 162,
    prove that the first term of the sequence is 2. -/
theorem geometric_sequence_first_term
  (a : ℝ)  -- First term of the sequence
  (r : ℝ)  -- Common ratio of the sequence
  (h1 : a * r^3 = 54)  -- Fourth term is 54
  (h2 : a * r^4 = 162)  -- Fifth term is 162
  : a = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3530_353069


namespace NUMINAMATH_CALUDE_lattice_points_count_l3530_353030

/-- A triangular lattice -/
structure TriangularLattice where
  /-- The distance between adjacent points is 1 -/
  adjacent_distance : ℝ
  adjacent_distance_eq : adjacent_distance = 1

/-- An equilateral triangle on a triangular lattice -/
structure EquilateralTriangle (L : ℝ) where
  /-- The side length of the triangle -/
  side_length : ℝ
  side_length_eq : side_length = L
  /-- The triangle has no lattice points on its sides -/
  no_lattice_points_on_sides : Prop

/-- The number of lattice points inside an equilateral triangle -/
def lattice_points_inside (L : ℝ) (triangle : EquilateralTriangle L) : ℕ :=
  sorry

theorem lattice_points_count (L : ℝ) (triangle : EquilateralTriangle L) :
  lattice_points_inside L triangle = (L^2 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_lattice_points_count_l3530_353030


namespace NUMINAMATH_CALUDE_french_not_english_speakers_l3530_353082

theorem french_not_english_speakers (total : ℕ) 
  (french_speakers : ℕ) (french_and_english : ℕ) :
  total = 200 →
  french_speakers = total * 45 / 100 →
  french_and_english = 25 →
  french_speakers - french_and_english = 65 := by
sorry

end NUMINAMATH_CALUDE_french_not_english_speakers_l3530_353082


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l3530_353074

/-- Calculates the total money earned by Katrina and her friends in the recycling program -/
def total_money_earned (initial_signup_bonus : ℕ) (referral_bonus : ℕ) (friends_day1 : ℕ) (friends_week : ℕ) : ℕ :=
  let katrina_earnings := initial_signup_bonus + referral_bonus * (friends_day1 + friends_week)
  let friends_earnings := referral_bonus * (friends_day1 + friends_week)
  katrina_earnings + friends_earnings

/-- Proves that the total money earned by Katrina and her friends is $125.00 -/
theorem recycling_program_earnings :
  total_money_earned 5 5 5 7 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_recycling_program_earnings_l3530_353074


namespace NUMINAMATH_CALUDE_school_enrollment_increase_l3530_353091

-- Define the variables and constants
def last_year_total : ℕ := 4000
def last_year_YY : ℕ := 2400
def XX_percent_increase : ℚ := 7 / 100
def extra_growth_XX : ℕ := 40

-- Define the theorem
theorem school_enrollment_increase : 
  ∃ (p : ℚ), 
    (p ≥ 0) ∧ 
    (p ≤ 1) ∧
    (XX_percent_increase * (last_year_total - last_year_YY) = 
     (p * last_year_YY) + extra_growth_XX) ∧
    (p = 3 / 100) := by
  sorry

end NUMINAMATH_CALUDE_school_enrollment_increase_l3530_353091


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3530_353014

def point (x y : ℝ) := (x, y)

def symmetric_point_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := point (-2) 1
  let Q : ℝ × ℝ := symmetric_point_x_axis P
  Q = point (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3530_353014


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l3530_353085

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hitting_ground_time : 
  ∃ t : ℝ, t = 1 + (Real.sqrt 19) / 2 ∧ 
  (∀ y : ℝ, y = -16 * t^2 + 32 * t + 60 → y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l3530_353085


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3530_353002

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * ((a + 2 * Complex.I) / (1 + Complex.I))).re = ((a + 2 * Complex.I) / (1 + Complex.I)).re → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3530_353002


namespace NUMINAMATH_CALUDE_polynomial_equality_l3530_353081

theorem polynomial_equality (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3530_353081


namespace NUMINAMATH_CALUDE_prime_square_sum_l3530_353093

theorem prime_square_sum (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
  (∃ (n : ℕ), p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_l3530_353093


namespace NUMINAMATH_CALUDE_at_least_one_nonnegative_l3530_353035

theorem at_least_one_nonnegative (x y z : ℝ) :
  max (x^2 + y + 1/4) (max (y^2 + z + 1/4) (z^2 + x + 1/4)) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_nonnegative_l3530_353035


namespace NUMINAMATH_CALUDE_correct_statements_count_l3530_353052

theorem correct_statements_count (x : ℝ) : 
  (((x > 0) → (x^2 > 0)) ∧ ((x^2 ≤ 0) → (x ≤ 0)) ∧ ¬((x ≤ 0) → (x^2 ≤ 0))) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_count_l3530_353052


namespace NUMINAMATH_CALUDE_enrique_commission_is_300_l3530_353059

-- Define the commission rate
def commission_rate : Real := 0.15

-- Define the sales data
def suits_sold : Nat := 2
def suit_price : Real := 700.00
def shirts_sold : Nat := 6
def shirt_price : Real := 50.00
def loafers_sold : Nat := 2
def loafer_price : Real := 150.00

-- Calculate total sales
def total_sales : Real :=
  (suits_sold : Real) * suit_price +
  (shirts_sold : Real) * shirt_price +
  (loafers_sold : Real) * loafer_price

-- Calculate Enrique's commission
def enrique_commission : Real := commission_rate * total_sales

-- Theorem to prove
theorem enrique_commission_is_300 :
  enrique_commission = 300.00 := by sorry

end NUMINAMATH_CALUDE_enrique_commission_is_300_l3530_353059


namespace NUMINAMATH_CALUDE_exam_pass_count_l3530_353057

theorem exam_pass_count (total_boys : ℕ) (overall_avg : ℚ) (pass_avg : ℚ) (fail_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 38 →
  pass_avg = 39 →
  fail_avg = 15 →
  ∃ (pass_count : ℕ), pass_count = 115 ∧ 
    pass_count * pass_avg + (total_boys - pass_count) * fail_avg = total_boys * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l3530_353057
