import Mathlib

namespace permutation_remainder_l877_87774

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid permutations of the 18-character string -/
def N : ℕ := sorry

/-- The sum of valid permutations for different arrangements -/
def permutation_sum : ℕ :=
  (choose 5 0) * (choose 5 0) * (choose 5 1) +
  (choose 5 1) * (choose 5 1) * (choose 5 2) +
  (choose 5 2) * (choose 5 2) * (choose 5 3) +
  (choose 5 3) * (choose 5 3) * (choose 5 4)

theorem permutation_remainder :
  N ≡ 755 [MOD 1000] :=
sorry

end permutation_remainder_l877_87774


namespace min_value_theorem_l877_87734

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (h_line : 3*m + n = 1) : 
  (∃ (x : ℝ), ∀ (m n : ℝ), m > 0 → n > 0 → 3*m + n = 1 → 3/m + 1/n ≥ x) ∧ 
  (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ 3*m + n = 1 ∧ 3/m + 1/n = 16) := by
  sorry

end min_value_theorem_l877_87734


namespace units_digit_of_power_l877_87746

theorem units_digit_of_power (n : ℕ) : n % 10 = 7 → (n^1997 % 10)^2999 % 10 = 3 := by
  sorry

end units_digit_of_power_l877_87746


namespace arithmetic_square_root_of_sqrt_16_l877_87782

theorem arithmetic_square_root_of_sqrt_16 :
  Real.sqrt (Real.sqrt 16) = 4 := by sorry

end arithmetic_square_root_of_sqrt_16_l877_87782


namespace initial_kibble_amount_l877_87741

/-- The amount of kibble Luna is supposed to eat daily -/
def daily_kibble : ℕ := 2

/-- The amount of kibble Mary gave Luna in the morning -/
def mary_morning : ℕ := 1

/-- The amount of kibble Mary gave Luna in the evening -/
def mary_evening : ℕ := 1

/-- The amount of kibble Frank gave Luna in the afternoon -/
def frank_afternoon : ℕ := 1

/-- The amount of kibble remaining in the bag the next morning -/
def remaining_kibble : ℕ := 7

/-- The theorem stating the initial amount of kibble in the bag -/
theorem initial_kibble_amount : 
  mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon + remaining_kibble = 12 := by
  sorry

#check initial_kibble_amount

end initial_kibble_amount_l877_87741


namespace sphere_hemisphere_volume_ratio_l877_87787

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) : 
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p)^3) = 1 / 13.5 := by
  sorry

end sphere_hemisphere_volume_ratio_l877_87787


namespace parallel_lines_a_value_l877_87764

/-- Given two lines that are parallel, prove that the value of 'a' is 1/2 -/
theorem parallel_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (2 * a * y - 1 = 0 ↔ y = 1 / (2 * a))) →
  (∀ x y : ℝ, ((3 * a - 1) * x + y - 1 = 0 ↔ y = -(3 * a - 1) * x + 1)) →
  (∀ x y : ℝ, 2 * a * y - 1 = 0 → (3 * a - 1) * x + y - 1 = 0 → x = 0) →
  a = 1 / 2 := by
sorry


end parallel_lines_a_value_l877_87764


namespace floor_sum_rationality_l877_87700

theorem floor_sum_rationality (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, ⌊p * n⌋ + ⌊q * n⌋ + ⌊r * n⌋ = n) :
  (∃ a b c : ℤ, p = a / b ∧ q = a / c) ∧ p + q + r = 1 := by
  sorry

end floor_sum_rationality_l877_87700


namespace bruce_payment_l877_87705

/-- Calculates the total amount Bruce paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1110 for his purchase -/
theorem bruce_payment : total_amount 8 70 10 55 = 1110 := by
  sorry

end bruce_payment_l877_87705


namespace range_of_b_length_of_AB_l877_87793

-- Define the line and ellipse
def line (x b : ℝ) : ℝ := x + b
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the intersection condition
def intersects (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  ellipse x₁ (line x₁ b) ∧ 
  ellipse x₂ (line x₂ b)

-- Theorem for the range of b
theorem range_of_b :
  ∀ b : ℝ, intersects b ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 :=
sorry

-- Theorem for the length of AB when b = 1
theorem length_of_AB :
  intersects 1 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧
    y₁ = line x₁ 1 ∧
    y₂ = line x₂ 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 :=
sorry

end range_of_b_length_of_AB_l877_87793


namespace proposition_relation_l877_87712

theorem proposition_relation :
  (∀ x y : ℝ, x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≤ 4 ∧ x^2 + y^2 > 2*x) :=
by sorry

end proposition_relation_l877_87712


namespace third_person_gets_max_median_l877_87714

/-- Represents the money distribution among three people -/
structure MoneyDistribution where
  person1 : ℕ
  person2 : ℕ
  person3 : ℕ

/-- The initial distribution of money -/
def initial_distribution : MoneyDistribution :=
  { person1 := 28, person2 := 72, person3 := 98 }

/-- The total amount of money -/
def total_money (d : MoneyDistribution) : ℕ :=
  d.person1 + d.person2 + d.person3

/-- Checks if a distribution is valid (sum equals total money) -/
def is_valid_distribution (d : MoneyDistribution) : Prop :=
  total_money d = total_money initial_distribution

/-- Checks if a number is the median of three numbers -/
def is_median (a b c m : ℕ) : Prop :=
  (a ≤ m ∧ m ≤ c) ∨ (c ≤ m ∧ m ≤ a)

/-- The maximum possible median after redistribution -/
def max_median : ℕ := 99

/-- Theorem: After redistribution to maximize the median, the third person ends up with $99 -/
theorem third_person_gets_max_median :
  ∃ (d : MoneyDistribution),
    is_valid_distribution d ∧
    is_median d.person1 d.person2 d.person3 max_median ∧
    d.person3 = max_median :=
  sorry

end third_person_gets_max_median_l877_87714


namespace alpha_integer_and_nonnegative_l877_87720

theorem alpha_integer_and_nonnegative (α : ℝ) 
  (h : ∀ n : ℕ+, ∃ k : ℤ, (n : ℝ) / α = k) : 
  0 ≤ α ∧ ∃ m : ℤ, α = m := by sorry

end alpha_integer_and_nonnegative_l877_87720


namespace mozzarella_amount_proof_l877_87781

/-- The cost of the special blend cheese in dollars per kilogram -/
def special_blend_cost : ℝ := 696.05

/-- The cost of mozzarella cheese in dollars per kilogram -/
def mozzarella_cost : ℝ := 504.35

/-- The cost of romano cheese in dollars per kilogram -/
def romano_cost : ℝ := 887.75

/-- The amount of romano cheese used in kilograms -/
def romano_amount : ℝ := 18.999999999999986

/-- The amount of mozzarella cheese used in kilograms -/
def mozzarella_amount : ℝ := 19

theorem mozzarella_amount_proof :
  ∃ (m : ℝ), abs (m - mozzarella_amount) < 0.1 ∧
  m * mozzarella_cost + romano_amount * romano_cost =
  (m + romano_amount) * special_blend_cost :=
sorry

end mozzarella_amount_proof_l877_87781


namespace arithmetic_sequence_general_term_l877_87785

/-- Proves the general term of an arithmetic sequence given specific conditions -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h3 : ∃ m : ℝ, ∀ n : ℕ, Real.sqrt (8 * S n + 2 * n) = m + (n - 1) * d) :
  ∀ n : ℕ, a n = 4 * n - 9 / 4 := by
  sorry

end arithmetic_sequence_general_term_l877_87785


namespace gummy_juice_time_proof_l877_87728

/-- Represents the time in hours since starting to mow the lawn -/
def DrinkTime : ℝ := 1.5

theorem gummy_juice_time_proof :
  let total_time : ℝ := 2.5 -- Total time spent mowing (from 10:00 AM to 12:30 PM)
  let normal_rate : ℝ := 1 / 3 -- Rate of mowing without juice (1/3 of lawn per hour)
  let boosted_rate : ℝ := 1 / 2 -- Rate of mowing with juice (1/2 of lawn per hour)
  let normal_portion : ℝ := DrinkTime * normal_rate -- Portion mowed without juice
  let boosted_portion : ℝ := (total_time - DrinkTime) * boosted_rate -- Portion mowed with juice
  normal_portion + boosted_portion = 1 -- Total lawn mowed equals 1
  ∧ DrinkTime = 1.5 -- Time when Bronquinha drank the juice (1.5 hours after 10:00 AM, which is 11:30 AM)
  := by sorry

#check gummy_juice_time_proof

end gummy_juice_time_proof_l877_87728


namespace probability_three_tails_one_head_probability_three_tails_one_head_proof_l877_87761

/-- The probability of getting exactly three tails and one head when four fair coins are tossed simultaneously -/
theorem probability_three_tails_one_head : ℚ :=
  1 / 4

/-- Proof that the probability of getting exactly three tails and one head when four fair coins are tossed simultaneously is 1/4 -/
theorem probability_three_tails_one_head_proof :
  probability_three_tails_one_head = 1 / 4 := by
  sorry

end probability_three_tails_one_head_probability_three_tails_one_head_proof_l877_87761


namespace inverse_composition_value_l877_87786

open Function

-- Define the functions h and k
variable (h k : ℝ → ℝ)

-- Define the condition given in the problem
axiom h_k_relation : ∀ x, h⁻¹ (k x) = 3 * x - 4

-- State the theorem to be proved
theorem inverse_composition_value : k⁻¹ (h 5) = 3 := by sorry

end inverse_composition_value_l877_87786


namespace four_squares_sum_l877_87747

theorem four_squares_sum (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a^2 + b^2 + c^2 + d^2 = 90 →
  a + b + c + d = 16 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 := by
sorry

end four_squares_sum_l877_87747


namespace salem_poem_lines_per_stanza_l877_87727

/-- Represents a poem with stanzas, lines, and words. -/
structure Poem where
  num_stanzas : ℕ
  words_per_line : ℕ
  total_words : ℕ

/-- Calculates the number of lines per stanza in a poem. -/
def lines_per_stanza (p : Poem) : ℕ :=
  (p.total_words / p.words_per_line) / p.num_stanzas

/-- Theorem stating that for a poem with 20 stanzas, 8 words per line, 
    and 1600 total words, each stanza has 10 lines. -/
theorem salem_poem_lines_per_stanza :
  let p : Poem := { num_stanzas := 20, words_per_line := 8, total_words := 1600 }
  lines_per_stanza p = 10 := by
  sorry

end salem_poem_lines_per_stanza_l877_87727


namespace solution_exists_l877_87744

/-- The number of primes less than or equal to n -/
def ν (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem solution_exists (m : ℕ) (hm : m > 2) :
  (∃ n : ℕ, n > 1 ∧ n / ν n = m) → (∃ n : ℕ, n > 1 ∧ n / ν n = m - 1) :=
sorry

end solution_exists_l877_87744


namespace arithmetic_sequences_ratio_l877_87757

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  h_sum_ratio : ∀ n : ℕ, S n / T n = 3 * n / (2 * n + 1)

/-- The main theorem -/
theorem arithmetic_sequences_ratio 
  (seq : ArithmeticSequences) : 
  (seq.a 1 + seq.a 2 + seq.a 14 + seq.a 19) / 
  (seq.b 1 + seq.b 3 + seq.b 17 + seq.b 19) = 17 / 13 := by
  sorry

end arithmetic_sequences_ratio_l877_87757


namespace factorization_2x_minus_x_squared_l877_87749

theorem factorization_2x_minus_x_squared (x : ℝ) : 2*x - x^2 = x*(2-x) := by
  sorry

end factorization_2x_minus_x_squared_l877_87749


namespace total_votes_l877_87751

theorem total_votes (votes_for votes_against total_votes : ℕ) : 
  votes_for = votes_against + 58 →
  votes_against = (40 * total_votes) / 100 →
  total_votes = votes_for + votes_against →
  total_votes = 290 := by
  sorry

end total_votes_l877_87751


namespace perpendicular_lines_and_intersection_l877_87797

-- Define the slopes and y-intercept of the lines
def m_s : ℚ := 4/3
def b_s : ℚ := -100
def m_t : ℚ := -3/4

-- Define the lines
def line_s (x : ℚ) : ℚ := m_s * x + b_s
def line_t (x : ℚ) : ℚ := m_t * x

-- Define the intersection point
def intersection_x : ℚ := 48
def intersection_y : ℚ := -36

theorem perpendicular_lines_and_intersection :
  -- Line t is perpendicular to line s
  m_s * m_t = -1 ∧
  -- Line t passes through (0, 0)
  line_t 0 = 0 ∧
  -- The intersection point satisfies both line equations
  line_s intersection_x = intersection_y ∧
  line_t intersection_x = intersection_y :=
sorry

end perpendicular_lines_and_intersection_l877_87797


namespace prob_max_with_replacement_prob_max_without_replacement_l877_87784

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

end prob_max_with_replacement_prob_max_without_replacement_l877_87784


namespace same_solution_implies_a_equals_seven_l877_87768

theorem same_solution_implies_a_equals_seven (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = 3 ∧ 3 - (a - x) / 3 = 1) →
  a = 7 := by
  sorry

end same_solution_implies_a_equals_seven_l877_87768


namespace sum_of_divisors_of_23_l877_87704

theorem sum_of_divisors_of_23 (h : Nat.Prime 23) : (Finset.sum (Nat.divisors 23) id) = 24 := by
  sorry

end sum_of_divisors_of_23_l877_87704


namespace inequality_solution_l877_87794

theorem inequality_solution (x : ℝ) : 
  x > 0 → |5 - 2*x| ≤ 8 → 0 ≤ x ∧ x ≤ 6.5 := by
  sorry

end inequality_solution_l877_87794


namespace constant_k_value_l877_87759

theorem constant_k_value (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) → k = -15 := by
sorry

end constant_k_value_l877_87759


namespace equation_represents_hyperbola_l877_87701

/-- The equation (x-y)^2 = 3(x^2 - y^2) represents a hyperbola -/
theorem equation_represents_hyperbola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b^2 - 4*a*c > 0 ∧
  ∀ (x y : ℝ), (x - y)^2 = 3*(x^2 - y^2) ↔ a*x^2 + b*x*y + c*y^2 = 0 :=
by sorry

end equation_represents_hyperbola_l877_87701


namespace composite_ratio_l877_87771

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]

def product_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_list first_six_composites : Rat) / (product_list next_six_composites) = 1 / 49 := by
  sorry

end composite_ratio_l877_87771


namespace inequality_proof_l877_87710

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 1) :
  Real.sqrt (x + y*z) + Real.sqrt (y + z*x) + Real.sqrt (z + x*y) ≥ 
  Real.sqrt (x*y*z) + Real.sqrt x + Real.sqrt y + Real.sqrt z :=
by sorry

end inequality_proof_l877_87710


namespace team_selection_ways_l877_87745

def number_of_combinations (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem team_selection_ways (total_boys total_girls team_boys team_girls : ℕ) 
  (h1 : total_boys = 7)
  (h2 : total_girls = 9)
  (h3 : team_boys = 3)
  (h4 : team_girls = 3) :
  (number_of_combinations total_boys team_boys) * (number_of_combinations total_girls team_girls) = 2940 :=
by sorry

end team_selection_ways_l877_87745


namespace nested_fraction_evaluation_l877_87748

theorem nested_fraction_evaluation :
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 ∧ 8 / 21 ≠ 3 / 5 := by
  sorry

end nested_fraction_evaluation_l877_87748


namespace equivalent_operation_l877_87738

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6)) / (2 / 3) = x * (5 / 4) := by
  sorry

end equivalent_operation_l877_87738


namespace bugs_and_flowers_l877_87739

/-- Given that 2.0 bugs ate 3.0 flowers in total, prove that each bug ate 1.5 flowers. -/
theorem bugs_and_flowers (total_bugs : ℝ) (total_flowers : ℝ) 
  (h1 : total_bugs = 2.0) 
  (h2 : total_flowers = 3.0) : 
  total_flowers / total_bugs = 1.5 := by
  sorry

end bugs_and_flowers_l877_87739


namespace min_cost_for_89_coins_l877_87725

/-- Represents the cost structure for the coin problem -/
structure CoinProblem where
  total_coins : Nat
  coin_cost : Nat
  yes_fee : Nat
  no_fee : Nat

/-- Calculates the minimum cost to guarantee obtaining the lucky coin -/
def min_cost_to_get_lucky_coin (problem : CoinProblem) : Nat :=
  sorry

/-- Theorem stating the minimum cost for the specific problem instance -/
theorem min_cost_for_89_coins :
  let problem : CoinProblem := {
    total_coins := 89,
    coin_cost := 30,
    yes_fee := 20,
    no_fee := 10
  }
  min_cost_to_get_lucky_coin problem = 130 := by
  sorry

end min_cost_for_89_coins_l877_87725


namespace ellipse_intersection_theorem_l877_87777

/-- Ellipse with foci on y-axis and center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line intersecting the ellipse -/
structure Line where
  k : ℝ
  m : ℝ

/-- Definition of the problem setup -/
def EllipseProblem (E : Ellipse) (l : Line) : Prop :=
  -- Eccentricity is √3/2
  E.a / (E.a ^ 2 - E.b ^ 2).sqrt = 2 / Real.sqrt 3 ∧
  -- Perimeter of quadrilateral is 4√5
  4 * (E.a ^ 2 + E.b ^ 2).sqrt = 4 * Real.sqrt 5 ∧
  -- Line intersects ellipse at two points
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (l.k * x₁ + l.m) ^ 2 / (4 : ℝ) + x₁ ^ 2 = 1 ∧
    (l.k * x₂ + l.m) ^ 2 / (4 : ℝ) + x₂ ^ 2 = 1 ∧
  -- AP = 3PB condition
  x₁ = -3 * x₂

/-- Main theorem to prove -/
theorem ellipse_intersection_theorem (E : Ellipse) (l : Line) 
  (h : EllipseProblem E l) : 
  1 < l.m ^ 2 ∧ l.m ^ 2 < 4 := by
  sorry

end ellipse_intersection_theorem_l877_87777


namespace common_y_intercept_l877_87706

theorem common_y_intercept (l₁ l₂ l₃ : ℝ → ℝ) (b : ℝ) :
  (∀ x, l₁ x = 1/2 * x + b) →
  (∀ x, l₂ x = 1/3 * x + b) →
  (∀ x, l₃ x = 1/4 * x + b) →
  ((-2*b) + (-3*b) + (-4*b) = 36) →
  b = -4 := by sorry

end common_y_intercept_l877_87706


namespace simplify_and_evaluate_l877_87790

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = -1) (h2 : y = 1) :
  2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 := by
  sorry

end simplify_and_evaluate_l877_87790


namespace milk_for_cookies_l877_87729

/-- Given that 18 cookies require 3 quarts of milk, and there are 2 pints in a quart,
    prove that 6 cookies require 2 pints of milk. -/
theorem milk_for_cookies (cookies_large : ℕ) (milk_quarts : ℕ) (cookies_small : ℕ) 
  (pints_per_quart : ℕ) (h1 : cookies_large = 18) (h2 : milk_quarts = 3) 
  (h3 : cookies_small = 6) (h4 : pints_per_quart = 2) : 
  (milk_quarts * pints_per_quart * cookies_small) / cookies_large = 2 :=
by sorry

end milk_for_cookies_l877_87729


namespace perfect_square_trinomial_condition_l877_87718

/-- 
If 4x² + (m-3)x + 1 is a perfect square trinomial, then m = 7 or m = -1.
-/
theorem perfect_square_trinomial_condition (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), 4*x^2 + (m-3)*x + 1 = (a*x + b)^2) → 
  (m = 7 ∨ m = -1) := by
sorry

end perfect_square_trinomial_condition_l877_87718


namespace leo_score_in_blackjack_l877_87779

/-- In a blackjack game, given the scores of Caroline and Anthony, 
    and the fact that Leo is the winner with the winning score, 
    prove that Leo's score is 21. -/
theorem leo_score_in_blackjack 
  (caroline_score : ℕ) 
  (anthony_score : ℕ) 
  (winning_score : ℕ) 
  (leo_is_winner : Bool) : ℕ :=
by
  -- Define the given conditions
  have h1 : caroline_score = 13 := by sorry
  have h2 : anthony_score = 19 := by sorry
  have h3 : winning_score = 21 := by sorry
  have h4 : leo_is_winner = true := by sorry

  -- Prove that Leo's score is equal to the winning score
  sorry

#check leo_score_in_blackjack

end leo_score_in_blackjack_l877_87779


namespace max_shadow_area_l877_87765

/-- Regular tetrahedron with edge length a -/
structure Tetrahedron where
  a : ℝ
  a_pos : a > 0

/-- Cube with edge length a -/
structure Cube where
  a : ℝ
  a_pos : a > 0

/-- The maximum shadow area of a regular tetrahedron and a cube -/
theorem max_shadow_area 
  (t : Tetrahedron) 
  (c : Cube) 
  (light_zenith : True) -- Light source is directly above
  : 
  (∃ (shadow_area_tetra : ℝ), 
    shadow_area_tetra ≤ t.a^2 / 2 ∧ 
    ∀ (other_area : ℝ), other_area ≤ shadow_area_tetra) ∧
  (∃ (shadow_area_cube : ℝ), 
    shadow_area_cube ≤ c.a^2 * Real.sqrt 3 / 3 ∧ 
    ∀ (other_area : ℝ), other_area ≤ shadow_area_cube) :=
sorry

end max_shadow_area_l877_87765


namespace line_equation_sum_of_squares_l877_87709

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 4 * x - 7

-- Define the point (2,1) that the line passes through
def point_on_line : Prop := line_l 2 1

-- Define the equation ax = by + c
def line_equation (a b c : ℤ) (x y : ℝ) : Prop := a * x = b * y + c

-- State that a, b, and c are positive integers with gcd 1
def abc_conditions (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ Int.gcd a (Int.gcd b c) = 1

-- Theorem statement
theorem line_equation_sum_of_squares :
  ∀ a b c : ℤ,
  (∀ x y : ℝ, line_l x y ↔ line_equation a b c x y) →
  abc_conditions a b c →
  a^2 + b^2 + c^2 = 66 := by sorry

end line_equation_sum_of_squares_l877_87709


namespace largest_angle_is_right_l877_87713

/-- Given a triangle ABC with side lengths a, b, and c, where c = 5 and 
    sqrt(a-4) + (b-3)^2 = 0, the largest interior angle of the triangle is 90°. -/
theorem largest_angle_is_right (a b c : ℝ) 
  (h1 : c = 5)
  (h2 : Real.sqrt (a - 4) + (b - 3)^2 = 0) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))
            (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
                 (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry

end largest_angle_is_right_l877_87713


namespace complex_number_problem_l877_87703

theorem complex_number_problem : ∃ (z : ℂ), 
  z.im = (3 * Complex.I).re ∧ 
  z.re = (-3 + Complex.I).im ∧ 
  z = 3 - 3 * Complex.I := by
  sorry

end complex_number_problem_l877_87703


namespace spoon_cost_l877_87717

theorem spoon_cost (num_plates : ℕ) (plate_cost : ℚ) (num_spoons : ℕ) (total_cost : ℚ) : 
  num_plates = 9 → 
  plate_cost = 2 → 
  num_spoons = 4 → 
  total_cost = 24 → 
  (total_cost - (↑num_plates * plate_cost)) / ↑num_spoons = 1.5 := by
sorry

end spoon_cost_l877_87717


namespace ellipse_and_line_theorem_l877_87763

-- Define the ellipse (C)
def ellipse (x y : ℝ) : Prop := x^2/12 + y^2/3 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 8*y^2 = 8

-- Define the line (l)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x+3)

-- Define the circle with PQ as diameter passing through origin
def circle_PQ_through_origin (P Q : ℝ × ℝ) : Prop :=
  (P.1 * Q.1 + P.2 * Q.2 = 0)

theorem ellipse_and_line_theorem :
  -- The ellipse passes through (-2,√2)
  ellipse (-2) (Real.sqrt 2) →
  -- The ellipse and hyperbola share the same foci
  (∀ x y, hyperbola x y ↔ x^2/8 - y^2 = 1) →
  -- For any k, if the line intersects the ellipse at P and Q
  -- and the circle with PQ as diameter passes through origin
  (∀ k P Q, 
    line k P.1 P.2 → 
    line k Q.1 Q.2 → 
    ellipse P.1 P.2 → 
    ellipse Q.1 Q.2 → 
    circle_PQ_through_origin P Q →
    -- Then k must be ±(2√11/11)
    (k = 2 * Real.sqrt 11 / 11 ∨ k = -2 * Real.sqrt 11 / 11)) :=
by sorry


end ellipse_and_line_theorem_l877_87763


namespace circus_tent_sections_l877_87780

theorem circus_tent_sections (section_capacity : ℕ) (total_capacity : ℕ) (h1 : section_capacity = 246) (h2 : total_capacity = 984) :
  total_capacity / section_capacity = 4 := by
  sorry

end circus_tent_sections_l877_87780


namespace polynomial_division_remainder_l877_87798

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^4 + 8 * X^3 - 27 * X^2 - 32 * X + 52 = 
  (X^2 + 5 * X + 2) * q + (52 * X + 80) := by
  sorry

end polynomial_division_remainder_l877_87798


namespace shortest_side_of_similar_triangle_l877_87719

/-- Given two similar right triangles, where the first triangle has a side length of 30 inches
    and a hypotenuse of 34 inches, and the second triangle has a hypotenuse of 102 inches,
    the shortest side of the second triangle is 48 inches. -/
theorem shortest_side_of_similar_triangle (a b c : ℝ) : 
  a^2 + b^2 = 34^2 →  -- Pythagorean theorem for the first triangle
  a = 30 →            -- Given side length of the first triangle
  b ≤ a →             -- b is the shortest side of the first triangle
  c^2 + (3*b)^2 = 102^2 →  -- Pythagorean theorem for the second triangle
  3*b = 48 :=         -- The shortest side of the second triangle is 48 inches
by sorry

end shortest_side_of_similar_triangle_l877_87719


namespace nested_radical_value_l877_87716

/-- The value of the infinite nested radical √(3 - √(3 - √(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + √13) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_radical_value_l877_87716


namespace parallel_vectors_iff_k_eq_neg_two_l877_87724

-- Define the vectors
def a : Fin 2 → ℝ := ![1, -2]
def b (k : ℝ) : Fin 2 → ℝ := ![k, 4]

-- Define parallel vectors
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ (∀ i, u i = c * v i)

-- Theorem statement
theorem parallel_vectors_iff_k_eq_neg_two :
  parallel a (b k) ↔ k = -2 := by
  sorry

end parallel_vectors_iff_k_eq_neg_two_l877_87724


namespace circle_center_l877_87740

/-- The center of the circle with equation x^2 + y^2 + 4x - 6y + 9 = 0 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 6*y + 9 = 0) ↔ ((x + 2)^2 + (y - 3)^2 = 4) :=
sorry

end circle_center_l877_87740


namespace boxes_filled_in_five_minutes_l877_87762

/-- A machine that fills boxes at a constant rate -/
structure BoxFillingMachine where
  boxes_per_hour : ℚ

/-- Given a machine that fills 24 boxes in 60 minutes, prove it fills 2 boxes in 5 minutes -/
theorem boxes_filled_in_five_minutes 
  (machine : BoxFillingMachine) 
  (h : machine.boxes_per_hour = 24 / 1) : 
  (machine.boxes_per_hour * 5 / 60 : ℚ) = 2 := by
  sorry


end boxes_filled_in_five_minutes_l877_87762


namespace ellipse_hyperbola_foci_l877_87796

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)  -- Condition for ellipse foci
  (h2 : a^2 + b^2 = 49)  -- Condition for hyperbola foci
  : a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 := by
  sorry

end ellipse_hyperbola_foci_l877_87796


namespace kaleb_ferris_wheel_spend_l877_87775

/-- Calculates the money spent on a ferris wheel ride given initial tickets, remaining tickets, and cost per ticket. -/
def money_spent_on_ride (initial_tickets remaining_tickets cost_per_ticket : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * cost_per_ticket

/-- Proves that Kaleb spent 27 dollars on the ferris wheel ride. -/
theorem kaleb_ferris_wheel_spend :
  let initial_tickets : ℕ := 6
  let remaining_tickets : ℕ := 3
  let cost_per_ticket : ℕ := 9
  money_spent_on_ride initial_tickets remaining_tickets cost_per_ticket = 27 := by
sorry

end kaleb_ferris_wheel_spend_l877_87775


namespace fourth_number_in_second_set_l877_87789

theorem fourth_number_in_second_set (x y : ℝ) : 
  ((28 + x + 42 + 78 + 104) / 5 = 90) →
  ((128 + 255 + 511 + y + x) / 5 = 423) →
  y = 1023 := by
sorry

end fourth_number_in_second_set_l877_87789


namespace trigonometric_problem_l877_87752

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (Real.tan (2*x) = -24/7) := by
  sorry

end trigonometric_problem_l877_87752


namespace percentage_equivalence_l877_87778

theorem percentage_equivalence : ∀ x : ℚ,
  (60 / 100) * 600 = (x / 100) * 720 → x = 50 := by
  sorry

end percentage_equivalence_l877_87778


namespace inequality_equivalence_l877_87766

theorem inequality_equivalence (x : ℝ) : (x - 8) / (x^2 - 4*x + 13) ≥ 0 ↔ x ≥ 8 := by
  sorry

end inequality_equivalence_l877_87766


namespace smallest_three_digit_power_of_two_plus_one_multiple_of_five_l877_87799

theorem smallest_three_digit_power_of_two_plus_one_multiple_of_five :
  ∃ (N : ℕ), 
    (100 ≤ N ∧ N ≤ 999) ∧ 
    (2^N + 1) % 5 = 0 ∧
    (∀ (M : ℕ), (100 ≤ M ∧ M < N) → (2^M + 1) % 5 ≠ 0) ∧
    N = 102 :=
by sorry

end smallest_three_digit_power_of_two_plus_one_multiple_of_five_l877_87799


namespace cube_sum_product_l877_87767

theorem cube_sum_product : (3^3 * 4^3) + (3^3 * 2^3) = 1944 := by
  sorry

end cube_sum_product_l877_87767


namespace kims_money_l877_87723

/-- Given the money relationships between Kim, Sal, Phil, and Alex, prove Kim's amount. -/
theorem kims_money (sal phil kim alex : ℝ) 
  (h1 : kim = 1.4 * sal)  -- Kim has 40% more money than Sal
  (h2 : sal = 0.8 * phil)  -- Sal has 20% less money than Phil
  (h3 : alex = 1.25 * (sal + kim))  -- Alex has 25% more money than Sal and Kim combined
  (h4 : sal + phil + alex = 3.6)  -- Sal, Phil, and Alex have a combined total of $3.60
  : kim = 0.96 := by
  sorry

end kims_money_l877_87723


namespace quadratic_inequality_l877_87742

/-- A quadratic function with axis of symmetry at x = 1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_of_symmetry : -b / (2 * a) = 1

/-- Theorem: For a quadratic function with axis of symmetry at x = 1, c < 2b -/
theorem quadratic_inequality (f : QuadraticFunction) : f.c < 2 * f.b := by
  sorry

end quadratic_inequality_l877_87742


namespace triangle_formation_l877_87730

/-- Function to check if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating which set of numbers can form a triangle -/
theorem triangle_formation :
  can_form_triangle 13 12 20 ∧
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 :=
sorry

end triangle_formation_l877_87730


namespace men_entered_room_l877_87750

theorem men_entered_room (initial_men initial_women : ℕ) 
  (men_entered women_left final_men : ℕ) :
  initial_men / initial_women = 4 / 5 →
  women_left = 3 →
  2 * (initial_women - women_left) = final_men →
  final_men = 14 →
  initial_men + men_entered = final_men →
  men_entered = 6 := by
sorry

end men_entered_room_l877_87750


namespace garden_internal_boundary_length_l877_87721

/-- Represents a square plot in the garden -/
structure Plot where
  side : ℕ
  deriving Repr

/-- Represents the garden configuration -/
structure Garden where
  width : ℕ
  height : ℕ
  plots : List Plot
  deriving Repr

/-- Calculates the total area of the garden -/
def gardenArea (g : Garden) : ℕ := g.width * g.height

/-- Calculates the area of a single plot -/
def plotArea (p : Plot) : ℕ := p.side * p.side

/-- Calculates the perimeter of a single plot -/
def plotPerimeter (p : Plot) : ℕ := 4 * p.side

/-- Calculates the external boundary of the garden -/
def externalBoundary (g : Garden) : ℕ := 2 * (g.width + g.height)

/-- The main theorem to prove -/
theorem garden_internal_boundary_length 
  (g : Garden) 
  (h1 : g.width = 6) 
  (h2 : g.height = 7) 
  (h3 : g.plots.length = 5) 
  (h4 : ∀ p ∈ g.plots, ∃ n : ℕ, p.side = n) 
  (h5 : (g.plots.map plotArea).sum = gardenArea g) : 
  ((g.plots.map plotPerimeter).sum - externalBoundary g) / 2 = 15 := by
  sorry

end garden_internal_boundary_length_l877_87721


namespace find_K_l877_87743

theorem find_K : ∃ K : ℕ, 32^5 * 4^5 = 2^K ∧ K = 35 := by
  sorry

end find_K_l877_87743


namespace exponent_multiplication_l877_87755

theorem exponent_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end exponent_multiplication_l877_87755


namespace monotonicity_of_g_range_of_a_for_two_extreme_points_inequality_for_extreme_points_l877_87776

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (1 - x) * Real.exp x + a

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.exp (2 * x) + (1 - x) * Real.exp x - Real.exp x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f' a x * Real.exp (2 - x)

-- Statement 1
theorem monotonicity_of_g :
  let a := Real.exp (-2) / 2
  ∀ x y, x < 2 → y > 2 → g a x > g a 2 ∧ g a 2 < g a y := by sorry

-- Statement 2
theorem range_of_a_for_two_extreme_points :
  ∀ a x₁ x₂, x₁ < x₂ →
  (f' a x₁ = 0 ∧ f' a x₂ = 0) →
  0 < a ∧ a < 1 / (2 * Real.exp 1) := by sorry

-- Statement 3
theorem inequality_for_extreme_points :
  ∀ a x₁ x₂, x₁ < x₂ →
  (f' a x₁ = 0 ∧ f' a x₂ = 0) →
  x₁ + 2 * x₂ > 3 := by sorry

end

end monotonicity_of_g_range_of_a_for_two_extreme_points_inequality_for_extreme_points_l877_87776


namespace eighteen_is_counterexample_l877_87711

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_counterexample (n : ℕ) : Prop :=
  ¬(is_prime n) ∧ ¬(is_prime (n - 3))

theorem eighteen_is_counterexample : is_counterexample 18 := by
  sorry

end eighteen_is_counterexample_l877_87711


namespace composition_of_even_function_is_even_l877_87773

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem composition_of_even_function_is_even (g : ℝ → ℝ) (h : is_even_function g) :
  is_even_function (fun x ↦ g (g (g x))) := by sorry

end composition_of_even_function_is_even_l877_87773


namespace existence_of_four_integers_l877_87736

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (abs a > 1000000) ∧ 
  (abs b > 1000000) ∧ 
  (abs c > 1000000) ∧ 
  (abs d > 1000000) ∧ 
  (1 / a + 1 / b + 1 / c + 1 / d : ℚ) = 1 / (a * b * c * d) := by
  sorry

end existence_of_four_integers_l877_87736


namespace dandelion_ratio_l877_87726

theorem dandelion_ratio : 
  ∀ (billy_initial george_initial billy_additional george_additional : ℕ) 
    (average : ℚ),
  billy_initial = 36 →
  billy_additional = 10 →
  george_additional = 10 →
  average = 34 →
  (billy_initial + george_initial + billy_additional + george_additional : ℚ) / 2 = average →
  george_initial * 3 = billy_initial :=
by sorry

end dandelion_ratio_l877_87726


namespace adult_price_calculation_l877_87795

/-- The daily price for adults at a public swimming pool -/
def adult_price (total_people : ℕ) (child_price : ℚ) (total_receipts : ℚ) (num_children : ℕ) : ℚ :=
  let num_adults : ℕ := total_people - num_children
  (total_receipts - (num_children : ℚ) * child_price) / (num_adults : ℚ)

/-- Theorem stating the adult price calculation for the given scenario -/
theorem adult_price_calculation :
  adult_price 754 (3/2) 1422 388 = 840 / 366 := by
  sorry

end adult_price_calculation_l877_87795


namespace angle_in_fourth_quadrant_l877_87731

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : Real.sin α < 0)
  (h3 : Real.cos α > 0) : 
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) := by
  sorry

end angle_in_fourth_quadrant_l877_87731


namespace second_train_length_l877_87753

/-- Calculates the length of the second train given the parameters of two trains approaching each other -/
theorem second_train_length 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (clear_time : ℝ) 
  (h1 : length_train1 = 100) 
  (h2 : speed_train1 = 42) 
  (h3 : speed_train2 = 30) 
  (h4 : clear_time = 12.998960083193344) :
  ∃ length_train2 : ℝ, abs (length_train2 - 159.98) < 0.01 :=
by
  sorry

end second_train_length_l877_87753


namespace perimeter_is_60_l877_87708

/-- Square with side length 9 inches -/
def square_side_length : ℝ := 9

/-- Equilateral triangle with side length equal to square's side length -/
def triangle_side_length : ℝ := square_side_length

/-- Figure ABFCE formed after translating the triangle -/
structure Figure where
  AB : ℝ := square_side_length
  BF : ℝ := triangle_side_length
  FC : ℝ := triangle_side_length
  CE : ℝ := square_side_length
  EA : ℝ := square_side_length

/-- Perimeter of the figure ABFCE -/
def perimeter (fig : Figure) : ℝ :=
  fig.AB + fig.BF + fig.FC + fig.CE + fig.EA

/-- Theorem: The perimeter of figure ABFCE is 60 inches -/
theorem perimeter_is_60 (fig : Figure) : perimeter fig = 60 := by
  sorry

end perimeter_is_60_l877_87708


namespace connie_marbles_l877_87737

/-- The number of marbles Connie started with -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℚ := 183.0

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 593

theorem connie_marbles :
  (initial_marbles : ℚ) - marbles_given = remaining_marbles :=
by sorry

end connie_marbles_l877_87737


namespace exam_average_theorem_l877_87756

def average_percentage (group1_count : ℕ) (group1_avg : ℚ) (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count : ℕ := group1_count + group2_count
  let total_points : ℚ := group1_count * group1_avg + group2_count * group2_avg
  total_points / total_count

theorem exam_average_theorem :
  average_percentage 15 (80/100) 10 (90/100) = 84/100 := by
  sorry

end exam_average_theorem_l877_87756


namespace lecture_arrangements_l877_87732

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of valid arrangements for k lecturers with ordering constraints --/
def valid_arrangements (k : ℕ) : ℕ :=
  (k - 1) * k / 2

/-- Calculates the number of ways to arrange the remaining lecturers --/
def remaining_arrangements (n k : ℕ) : ℕ :=
  Nat.factorial (n - k)

/-- Theorem stating the total number of possible lecture arrangements --/
theorem lecture_arrangements :
  valid_arrangements k * remaining_arrangements n k = 240 :=
sorry

end lecture_arrangements_l877_87732


namespace vinegar_mixture_percentage_l877_87707

/-- The percentage of the second vinegar solution -/
def P : ℝ := 40

/-- Volume of each initial solution in milliliters -/
def initial_volume : ℝ := 10

/-- Volume of the final mixture in milliliters -/
def final_volume : ℝ := 50

/-- Percentage of the first solution -/
def first_percentage : ℝ := 5

/-- Percentage of the final mixture -/
def final_percentage : ℝ := 9

theorem vinegar_mixture_percentage :
  initial_volume * (first_percentage / 100) +
  initial_volume * (P / 100) =
  final_volume * (final_percentage / 100) :=
sorry

end vinegar_mixture_percentage_l877_87707


namespace tetrahedron_division_l877_87772

-- Define a regular tetrahedron
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_positive : edge_length > 0)

-- Define the division of edges
def divide_edges (t : RegularTetrahedron) : ℕ := 3

-- Define the planes drawn through division points
structure DivisionPlanes :=
  (tetrahedron : RegularTetrahedron)
  (num_divisions : ℕ)
  (parallel_to_faces : Bool)

-- Define the number of parts the tetrahedron is divided into
def num_parts (t : RegularTetrahedron) (d : DivisionPlanes) : ℕ := 15

-- Theorem statement
theorem tetrahedron_division (t : RegularTetrahedron) :
  let d := DivisionPlanes.mk t (divide_edges t) true
  num_parts t d = 15 := by sorry

end tetrahedron_division_l877_87772


namespace prob_at_least_one_box_same_color_exact_l877_87760

/-- Represents the number of friends -/
def num_friends : ℕ := 4

/-- Represents the number of blocks each friend has -/
def num_blocks : ℕ := 6

/-- Represents the number of boxes -/
def num_boxes : ℕ := 6

/-- Represents the probability of a specific color being placed in a specific box by one friend -/
def prob_color_in_box : ℚ := 1 / num_blocks

/-- Represents the probability of three friends placing the same color in a specific box -/
def prob_three_same_color : ℚ := prob_color_in_box ^ 3

/-- Represents the probability of at least one box having all blocks of the same color -/
def prob_at_least_one_box_same_color : ℚ := 1 - (1 - num_blocks * prob_three_same_color) ^ num_boxes

theorem prob_at_least_one_box_same_color_exact : 
  prob_at_least_one_box_same_color = 517 / 7776 := by sorry

end prob_at_least_one_box_same_color_exact_l877_87760


namespace grocery_weight_difference_l877_87783

theorem grocery_weight_difference (rice sugar green_beans remaining_stock : ℝ) : 
  rice = green_beans - 30 →
  green_beans = 60 →
  remaining_stock = (2/3 * rice) + (4/5 * sugar) + green_beans →
  remaining_stock = 120 →
  green_beans - sugar = 10 := by
sorry

end grocery_weight_difference_l877_87783


namespace plane_equation_from_point_and_normal_l877_87754

/-- Given a point P₀ and a normal vector u⃗, prove that the equation
    ax + by + cz + d = 0 represents the plane passing through P₀ with normal vector u⃗. -/
theorem plane_equation_from_point_and_normal (P₀ : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ) 
  (a b c d : ℝ) :
  let (x₀, y₀, z₀) := P₀
  let (a', b', c') := u
  (a = 2 ∧ b = -1 ∧ c = -3 ∧ d = 3) →
  (x₀ = 1 ∧ y₀ = 2 ∧ z₀ = 1) →
  (a' = -2 ∧ b' = 1 ∧ c' = 3) →
  ∀ (x y z : ℝ), a*x + b*y + c*z + d = 0 ↔ 
    a'*(x - x₀) + b'*(y - y₀) + c'*(z - z₀) = 0 :=
by sorry

end plane_equation_from_point_and_normal_l877_87754


namespace equation_root_implies_z_value_l877_87792

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (x : ℂ) (a : ℝ) : Prop :=
  x^2 + (4 + i) * x + (4 : ℂ) + a * i = 0

-- Define the complex number z
def z (a b : ℝ) : ℂ := a + b * i

-- Theorem statement
theorem equation_root_implies_z_value (a b : ℝ) :
  equation b a → z a b = 2 - 2 * i :=
by
  sorry

end equation_root_implies_z_value_l877_87792


namespace solution_product_l877_87735

theorem solution_product (r s : ℝ) : 
  (r - 7) * (3 * r + 11) = r^2 - 16 * r + 63 →
  (s - 7) * (3 * s + 11) = s^2 - 16 * s + 63 →
  r ≠ s →
  (r + 4) * (s + 4) = -66 := by
sorry

end solution_product_l877_87735


namespace g_domain_all_reals_l877_87722

/-- The function g(x) = 1 / ((x-2)^2 + (x+2)^2 + 1) is defined for all real numbers. -/
theorem g_domain_all_reals :
  ∀ x : ℝ, (((x - 2)^2 + (x + 2)^2 + 1) ≠ 0) := by
  sorry

end g_domain_all_reals_l877_87722


namespace cab_speed_fraction_l877_87733

theorem cab_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 30 →
  delay = 6 →
  (usual_time / (usual_time + delay)) = 5/6 :=
by sorry

end cab_speed_fraction_l877_87733


namespace sams_books_l877_87770

theorem sams_books (tim_books sam_books total_books : ℕ) : 
  tim_books = 44 → 
  total_books = 96 → 
  total_books = tim_books + sam_books → 
  sam_books = 52 := by
sorry

end sams_books_l877_87770


namespace cookies_difference_l877_87788

def initial_cookies : ℕ := 41
def cookies_given : ℕ := 9
def cookies_eaten : ℕ := 18

theorem cookies_difference : cookies_eaten - cookies_given = 9 := by
  sorry

end cookies_difference_l877_87788


namespace shirt_discount_percentage_l877_87702

/-- Calculates the discount percentage for a shirt given its cost price, profit margin, and sale price. -/
theorem shirt_discount_percentage
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (sale_price : ℝ)
  (h1 : cost_price = 20)
  (h2 : profit_margin = 0.3)
  (h3 : sale_price = 13) :
  (cost_price * (1 + profit_margin) - sale_price) / (cost_price * (1 + profit_margin)) = 0.5 := by
  sorry

end shirt_discount_percentage_l877_87702


namespace pentagon_angle_measure_l877_87769

theorem pentagon_angle_measure (a b c d e : ℝ) : 
  -- Pentagon ABCDE is convex (sum of angles is 540°)
  a + b + c + d + e = 540 →
  -- Angle D is 30° more than angle A
  d = a + 30 →
  -- Angle E is 50° more than angle A
  e = a + 50 →
  -- Angles B and C are equal
  b = c →
  -- Angle A is 45° less than angle B
  a + 45 = b →
  -- Conclusion: Angle D measures 104°
  d = 104 := by
  sorry

end pentagon_angle_measure_l877_87769


namespace number_is_composite_l877_87758

theorem number_is_composite (n : ℕ) (h : n = 2^1000) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^n + 1 = a * b :=
sorry

end number_is_composite_l877_87758


namespace inequality_implies_m_upper_bound_l877_87715

theorem inequality_implies_m_upper_bound :
  (∀ x₁ : ℝ, ∃ x₂ ∈ Set.Icc 3 4, x₁^2 + x₁*x₂ + x₂^2 ≥ 2*x₁ + m*x₂ + 3) →
  m ≤ 3 :=
by sorry

end inequality_implies_m_upper_bound_l877_87715


namespace abs_eq_self_implies_nonnegative_l877_87791

theorem abs_eq_self_implies_nonnegative (a : ℝ) : |a| = a → a ≥ 0 := by
  sorry

end abs_eq_self_implies_nonnegative_l877_87791
