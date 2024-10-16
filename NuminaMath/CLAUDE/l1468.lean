import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l1468_146841

theorem simplify_expression (x : ℝ) : 3 * x^2 - 1 - 2*x - 5 + 3*x - x = 3 * x^2 - 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1468_146841


namespace NUMINAMATH_CALUDE_maria_test_scores_l1468_146806

def test_scores : List ℤ := [94, 92, 91, 75, 68]

theorem maria_test_scores :
  let scores := test_scores
  (scores.length = 5) ∧
  (scores.take 3 = [91, 75, 68]) ∧
  (scores.sum / scores.length = 84) ∧
  (∀ s ∈ scores, s < 95) ∧
  (∀ s ∈ scores, s ≥ 65) ∧
  scores.Nodup ∧
  scores.Sorted (· ≥ ·) :=
by sorry

end NUMINAMATH_CALUDE_maria_test_scores_l1468_146806


namespace NUMINAMATH_CALUDE_exists_divisible_by_n_l1468_146827

theorem exists_divisible_by_n (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k < n ∧ (n ∣ 2^k - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_n_l1468_146827


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l1468_146881

theorem cos_alpha_plus_5pi_12 (α : Real) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l1468_146881


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_two_squared_geq_four_l1468_146829

theorem negation_of_forall_geq_two_squared_geq_four :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x₀ : ℝ, x₀ ≥ 2 ∧ x₀^2 < 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_two_squared_geq_four_l1468_146829


namespace NUMINAMATH_CALUDE_smallest_special_number_l1468_146839

/-- A number is composite if it's not prime -/
def IsComposite (n : ℕ) : Prop := ¬ Nat.Prime n

/-- A number has no prime factor less than m if all its prime factors are greater than or equal to m -/
def NoPrimeFactorLessThan (n m : ℕ) : Prop :=
  ∀ p, p < m → Nat.Prime p → ¬(p ∣ n)

theorem smallest_special_number : ∃ n : ℕ,
  n > 3000 ∧
  IsComposite n ∧
  ¬(∃ k : ℕ, n = k^2) ∧
  NoPrimeFactorLessThan n 60 ∧
  (∀ m : ℕ, m > 3000 → IsComposite m → ¬(∃ k : ℕ, m = k^2) → NoPrimeFactorLessThan m 60 → m ≥ n) ∧
  n = 4087 := by
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l1468_146839


namespace NUMINAMATH_CALUDE_boric_acid_mixture_concentration_l1468_146861

/-- Given two boric acid solutions with concentrations and volumes, 
    calculate the concentration of the resulting mixture --/
theorem boric_acid_mixture_concentration 
  (c1 : ℝ) (c2 : ℝ) (v1 : ℝ) (v2 : ℝ) 
  (h1 : c1 = 0.01) -- 1% concentration
  (h2 : c2 = 0.05) -- 5% concentration
  (h3 : v1 = 15) -- 15 mL of first solution
  (h4 : v2 = 15) -- 15 mL of second solution
  : (c1 * v1 + c2 * v2) / (v1 + v2) = 0.03 := by
  sorry

#check boric_acid_mixture_concentration

end NUMINAMATH_CALUDE_boric_acid_mixture_concentration_l1468_146861


namespace NUMINAMATH_CALUDE_patio_length_l1468_146825

theorem patio_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 4 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 40 := by
sorry

end NUMINAMATH_CALUDE_patio_length_l1468_146825


namespace NUMINAMATH_CALUDE_division_problem_l1468_146882

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 131) (h2 : quotient = 9) (h3 : remainder = 5) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1468_146882


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1468_146883

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1468_146883


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1468_146876

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 140 → n = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1468_146876


namespace NUMINAMATH_CALUDE_consecutive_ranks_probability_l1468_146803

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Number of possible consecutive rank sets (A-2-3 to J-Q-K) -/
def ConsecutiveRankSets : ℕ := 10

/-- Number of suits in a standard deck -/
def Suits : ℕ := 4

/-- The probability of drawing three cards of consecutive ranks from a standard deck -/
theorem consecutive_ranks_probability :
  (ConsecutiveRankSets * Suits^CardsDrawn) / (StandardDeck.choose CardsDrawn) = 32 / 1105 := by
  sorry

#check consecutive_ranks_probability

end NUMINAMATH_CALUDE_consecutive_ranks_probability_l1468_146803


namespace NUMINAMATH_CALUDE_restaurant_order_l1468_146868

theorem restaurant_order (b h p s : ℕ) : 
  b = 30 → b = 2 * h → p = h + 5 → s = 3 * p → b + h + p + s = 125 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_order_l1468_146868


namespace NUMINAMATH_CALUDE_M_equals_N_l1468_146895

/-- Set M of integers defined as 12m + 8n + 4l where m, n, l are integers -/
def M : Set ℤ := {u : ℤ | ∃ (m n l : ℤ), u = 12*m + 8*n + 4*l}

/-- Set N of integers defined as 20p + 16q + 12r where p, q, r are integers -/
def N : Set ℤ := {u : ℤ | ∃ (p q r : ℤ), u = 20*p + 16*q + 12*r}

/-- Theorem stating that set M is equal to set N -/
theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l1468_146895


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l1468_146855

theorem simplify_radical_expression : 
  Real.sqrt 80 - 4 * Real.sqrt 5 + 3 * Real.sqrt (180 / 3) = Real.sqrt 540 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l1468_146855


namespace NUMINAMATH_CALUDE_bird_count_2003_l1468_146885

/-- The number of birds in the Weishui Development Zone over three years -/
structure BirdCount where
  year2001 : ℝ
  year2002 : ℝ
  year2003 : ℝ

/-- The conditions of the bird count problem -/
def bird_count_conditions (bc : BirdCount) : Prop :=
  bc.year2002 = 1.5 * bc.year2001 ∧ 
  bc.year2003 = 2 * bc.year2002

/-- Theorem stating that under the given conditions, the number of birds in 2003 is 3 times the number in 2001 -/
theorem bird_count_2003 (bc : BirdCount) (h : bird_count_conditions bc) : 
  bc.year2003 = 3 * bc.year2001 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_2003_l1468_146885


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l1468_146834

/-- Express 3.167167167... as a fraction -/
theorem recurring_decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ (3 : ℚ) + (167 : ℚ) / 1000 / (1 - 1 / 1000) = (n : ℚ) / d :=
by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l1468_146834


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1468_146843

/-- The line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∃ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x ∧
    ∀ x' y' : ℝ, y' = 3 * x' + c → y'^2 ≥ 12 * x') ↔ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1468_146843


namespace NUMINAMATH_CALUDE_equation_roots_sum_l1468_146844

theorem equation_roots_sum (a b c m : ℝ) : 
  (∃ x y : ℝ, 
    (x^2 - (b+1)*x) / (2*a*x - c) = (2*m-3) / (2*m+1) ∧
    (y^2 - (b+1)*y) / (2*a*y - c) = (2*m-3) / (2*m+1) ∧
    x + y = b + 1) →
  m = 1.5 := by sorry

end NUMINAMATH_CALUDE_equation_roots_sum_l1468_146844


namespace NUMINAMATH_CALUDE_jessica_probability_is_37_966_l1468_146816

/-- Represents the problem of distributing textbooks into boxes. -/
structure TextbookDistribution where
  total_books : Nat
  english_books : Nat
  box1_capacity : Nat
  box2_capacity : Nat
  box3_capacity : Nat
  box4_capacity : Nat

/-- The specific textbook distribution problem given in the question. -/
def jessica_distribution : TextbookDistribution :=
  { total_books := 15
  , english_books := 4
  , box1_capacity := 3
  , box2_capacity := 4
  , box3_capacity := 5
  , box4_capacity := 3
  }

/-- Calculates the probability of all English textbooks ending up in the third box. -/
def probability_all_english_in_third_box (d : TextbookDistribution) : Rat :=
  sorry

/-- Theorem stating that the probability for Jessica's distribution is 37/966. -/
theorem jessica_probability_is_37_966 :
  probability_all_english_in_third_box jessica_distribution = 37 / 966 := by
  sorry

end NUMINAMATH_CALUDE_jessica_probability_is_37_966_l1468_146816


namespace NUMINAMATH_CALUDE_josh_marbles_after_gift_l1468_146805

/-- The number of marbles Josh has after receiving marbles from Jack -/
theorem josh_marbles_after_gift (original : ℝ) (gift : ℝ) (total : ℝ)
  (h1 : original = 22.5)
  (h2 : gift = 20.75)
  (h3 : total = original + gift) :
  total = 43.25 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_after_gift_l1468_146805


namespace NUMINAMATH_CALUDE_garden_feet_count_l1468_146830

/-- Calculates the total number of feet in a garden with dogs and ducks -/
def total_feet (num_dogs : ℕ) (num_ducks : ℕ) (feet_per_dog : ℕ) (feet_per_duck : ℕ) : ℕ :=
  num_dogs * feet_per_dog + num_ducks * feet_per_duck

/-- Theorem: The total number of feet in a garden with 6 dogs and 2 ducks is 28 -/
theorem garden_feet_count : total_feet 6 2 4 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_feet_count_l1468_146830


namespace NUMINAMATH_CALUDE_abs_equation_solution_set_l1468_146812

theorem abs_equation_solution_set (x : ℝ) :
  |2*x - 1| = |x| + |x - 1| ↔ x ≤ 0 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_set_l1468_146812


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_tangent_circles_l1468_146870

/-- The parametric equation of a circle -/
def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

/-- The equation of the fixed circle -/
def fixed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_passes_through_fixed_point (a : ℝ) :
  circle_equation a 4 (-2) := by sorry

theorem tangent_circles (a : ℝ) :
  (∃ x y : ℝ, circle_equation a x y ∧ fixed_circle x y) ↔ 
  (a = 1 + Real.sqrt 5 / 5 ∨ a = 1 - Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_tangent_circles_l1468_146870


namespace NUMINAMATH_CALUDE_nth_equation_sum_l1468_146892

theorem nth_equation_sum (n : ℕ) (h : n > 0) :
  (Finset.range (2 * n - 1)).sum (λ i => n + i) = (2 * n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_sum_l1468_146892


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1468_146810

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * a * x + 3 > 0) → 0 ≤ a ∧ a < 3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1468_146810


namespace NUMINAMATH_CALUDE_lcm_5_6_8_18_l1468_146853

theorem lcm_5_6_8_18 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 18)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_8_18_l1468_146853


namespace NUMINAMATH_CALUDE_larger_number_proof_l1468_146858

theorem larger_number_proof (L S : ℕ) (hL : L > S) :
  L - S = 1365 → L = 6 * S + 20 → L = 1634 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1468_146858


namespace NUMINAMATH_CALUDE_expression_non_negative_lower_bound_achievable_l1468_146852

/-- The expression is always non-negative for real x and y -/
theorem expression_non_negative (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 := by
  sorry

/-- The lower bound of 0 is achievable -/
theorem lower_bound_achievable :
  ∃ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_non_negative_lower_bound_achievable_l1468_146852


namespace NUMINAMATH_CALUDE_gravel_path_cost_l1468_146809

/-- The cost of gravelling a path inside a rectangular plot -/
theorem gravel_path_cost 
  (length width path_width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_length : length = 110) 
  (h_width : width = 65) 
  (h_path_width : path_width = 2.5) 
  (h_cost_per_sqm : cost_per_sqm = 0.4) : 
  ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * cost_per_sqm = 360 := by
sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l1468_146809


namespace NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l1468_146877

theorem twenty_is_eighty_percent_of_twentyfive : ∃ y : ℝ, y > 0 ∧ 20 / y = 80 / 100 → y = 25 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l1468_146877


namespace NUMINAMATH_CALUDE_sequence_convergence_comparison_l1468_146838

/-- Given sequences (aₙ) and (bₙ) defined by the recurrence relations
    aₙ₊₁ = (aₙ + 1) / 2 and bₙ₊₁ = bₙᵏ, where 0 < k < 1/2 and
    a₀, b₀ ∈ (0, 1), there exists an N such that for all n ≥ N, aₙ < bₙ. -/
theorem sequence_convergence_comparison
  (k : ℝ) (h_k_pos : 0 < k) (h_k_bound : k < 1/2)
  (a₀ b₀ : ℝ) (h_a₀ : 0 < a₀ ∧ a₀ < 1) (h_b₀ : 0 < b₀ ∧ b₀ < 1)
  (a : ℕ → ℝ) (h_a : ∀ n, a (n + 1) = (a n + 1) / 2)
  (b : ℕ → ℝ) (h_b : ∀ n, b (n + 1) = (b n) ^ k)
  (h_a_init : a 0 = a₀) (h_b_init : b 0 = b₀) :
  ∃ N, ∀ n ≥ N, a n < b n :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_comparison_l1468_146838


namespace NUMINAMATH_CALUDE_exponential_max_greater_than_min_l1468_146818

theorem exponential_max_greater_than_min (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x ∈ Set.Icc 1 2 ∧ y ∈ Set.Icc 1 2 ∧ a^x > a^y :=
sorry

end NUMINAMATH_CALUDE_exponential_max_greater_than_min_l1468_146818


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1468_146884

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

-- State the theorem
theorem geometric_sequence_problem (a₁ a₄ : ℝ) (m : ℤ) :
  a₁ = 2 →
  a₄ = 1/4 →
  m = -15 →
  (∃ r : ℝ, ∀ n : ℕ, geometric_sequence a₁ r n = 2^(2 - n)) →
  m = 14 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l1468_146884


namespace NUMINAMATH_CALUDE_hostel_provisions_l1468_146840

-- Define the initial number of men
def initial_men : ℕ := 250

-- Define the number of days provisions last initially
def initial_days : ℕ := 36

-- Define the number of men who left
def men_left : ℕ := 50

-- Define the number of days provisions last after men left
def new_days : ℕ := 45

-- Theorem statement
theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_left) * new_days :=
by sorry

end NUMINAMATH_CALUDE_hostel_provisions_l1468_146840


namespace NUMINAMATH_CALUDE_boys_playing_marbles_count_l1468_146889

/-- The number of marbles Haley has -/
def total_marbles : ℕ := 26

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 2

/-- The number of boys who love to play marbles -/
def boys_playing_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_playing_marbles_count : boys_playing_marbles = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_playing_marbles_count_l1468_146889


namespace NUMINAMATH_CALUDE_train_cars_count_l1468_146842

-- Define the given conditions
def cars_counted : ℕ := 10
def initial_time : ℕ := 15
def total_time : ℕ := 210  -- 3 minutes and 30 seconds in seconds

-- Define the theorem
theorem train_cars_count :
  let rate : ℚ := cars_counted / initial_time
  rate * total_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l1468_146842


namespace NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l1468_146820

def total_population : ℕ := 180
def elderly_population : ℕ := 30
def sample_size : ℕ := 36

theorem stratified_sampling_elderly_count :
  (elderly_population * sample_size) / total_population = 6 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_elderly_count_l1468_146820


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1468_146862

-- Equation 1
theorem solve_equation_one (x : ℝ) : 4 * (x - 2) = 2 * x ↔ x = 4 := by
  sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x + 1) / 4 = 1 - (1 - x) / 3 ↔ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1468_146862


namespace NUMINAMATH_CALUDE_possible_regions_l1468_146811

/-- The number of lines dividing the plane -/
def num_lines : ℕ := 99

/-- The function that calculates the number of regions based on the number of parallel lines -/
def num_regions (k : ℕ) : ℕ := (k + 1) * (100 - k)

/-- The theorem stating the possible values of n less than 199 -/
theorem possible_regions :
  ∀ n : ℕ, n < 199 →
    (∃ k : ℕ, k ≤ num_lines ∧ n = num_regions k) →
    n = 100 ∨ n = 198 := by
  sorry

end NUMINAMATH_CALUDE_possible_regions_l1468_146811


namespace NUMINAMATH_CALUDE_candy_distribution_l1468_146846

/-- Represents the number of candies eaten by each person -/
structure CandyCount where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Andrey's rate / Boris's rate
  andrey_denis : ℚ  -- Andrey's rate / Denis's rate

/-- The theorem stating the correct number of candies eaten by each person -/
theorem candy_distribution (rates : EatingRates) (total : ℕ) : 
  rates.andrey_boris = 4/3 ∧ 
  rates.andrey_denis = 6/7 ∧ 
  total = 70 → 
  ∃ (count : CandyCount), 
    count.andrey = 24 ∧ 
    count.boris = 18 ∧ 
    count.denis = 28 ∧
    count.andrey + count.boris + count.denis = total ∧
    count.andrey / count.boris = rates.andrey_boris ∧
    count.andrey / count.denis = rates.andrey_denis :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1468_146846


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1468_146814

/-- An arithmetic sequence with sum of first n terms Sn -/
structure ArithmeticSequence where
  S : ℕ → ℤ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_{m-1} = -2, S_m = 0, and S_{m+1} = 3, then m = 5 -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
  (h1 : seq.S (m - 1) = -2)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 1) = 3) :
  m = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l1468_146814


namespace NUMINAMATH_CALUDE_mcpherson_contribution_l1468_146832

/-- Calculate Mr. McPherson's contribution to rent and expenses --/
theorem mcpherson_contribution
  (current_rent : ℝ)
  (rent_increase_rate : ℝ)
  (current_monthly_expenses : ℝ)
  (monthly_expenses_increase_rate : ℝ)
  (mrs_mcpherson_contribution_rate : ℝ)
  (h1 : current_rent = 1200)
  (h2 : rent_increase_rate = 0.05)
  (h3 : current_monthly_expenses = 100)
  (h4 : monthly_expenses_increase_rate = 0.03)
  (h5 : mrs_mcpherson_contribution_rate = 0.30) :
  ∃ (mr_mcpherson_contribution : ℝ),
    mr_mcpherson_contribution = 1747.20 ∧
    mr_mcpherson_contribution =
      (1 - mrs_mcpherson_contribution_rate) *
      (current_rent * (1 + rent_increase_rate) +
       12 * current_monthly_expenses * (1 + monthly_expenses_increase_rate)) :=
by
  sorry

end NUMINAMATH_CALUDE_mcpherson_contribution_l1468_146832


namespace NUMINAMATH_CALUDE_square_difference_l1468_146826

theorem square_difference (x y : ℝ) : (x - y) * (x - y) = x^2 - 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1468_146826


namespace NUMINAMATH_CALUDE_p_true_and_q_false_p_and_not_q_true_l1468_146823

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Theorem stating that p is true and q is false
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

-- Theorem stating that p ∧ ¬q is true
theorem p_and_not_q_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_p_and_not_q_true_l1468_146823


namespace NUMINAMATH_CALUDE_midline_triangle_area_sum_l1468_146867

/-- The sum of areas of an infinite series of triangles, where each triangle is constructed 
    from the midlines of the previous triangle, given the area of the original triangle. -/
theorem midline_triangle_area_sum (t : ℝ) (h : t > 0) : 
  ∃ (S : ℝ), S = (∑' n, t * (3/4)^n) ∧ S = 4 * t :=
sorry

end NUMINAMATH_CALUDE_midline_triangle_area_sum_l1468_146867


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1468_146854

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_second_term : a 1 = 3)
  (h_fourth_term : a 3 = 12) :
  a 0 = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1468_146854


namespace NUMINAMATH_CALUDE_polyline_intersection_theorem_l1468_146886

/-- A polyline is represented as a list of points -/
def Polyline := List (ℝ × ℝ)

/-- Calculate the length of a polyline -/
def polylineLength (p : Polyline) : ℝ :=
  sorry

/-- Check if a polyline is inside a unit square -/
def insideUnitSquare (p : Polyline) : Prop :=
  sorry

/-- Check if a polyline intersects itself -/
def selfIntersecting (p : Polyline) : Prop :=
  sorry

/-- Count the number of intersections between a line and a polyline -/
def intersectionCount (line : ℝ × ℝ → Prop) (p : Polyline) : ℕ :=
  sorry

/-- The main theorem -/
theorem polyline_intersection_theorem (p : Polyline) 
  (h1 : insideUnitSquare p)
  (h2 : ¬selfIntersecting p)
  (h3 : polylineLength p > 1000) :
  ∃ (line : ℝ × ℝ → Prop), 
    (∀ x y, line (x, y) ↔ (x = 0 ∨ x = 1 ∨ y = 0 ∨ y = 1)) ∧
    intersectionCount line p ≥ 501 :=
  sorry

end NUMINAMATH_CALUDE_polyline_intersection_theorem_l1468_146886


namespace NUMINAMATH_CALUDE_remainder_two_pow_33_mod_9_l1468_146828

theorem remainder_two_pow_33_mod_9 : 2^33 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_two_pow_33_mod_9_l1468_146828


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1468_146819

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
  x = 8 ∧ y = 2 * Real.sqrt 3 →
  r = Real.sqrt (x^2 + y^2) →
  θ = Real.arctan (y / x) →
  r > 0 →
  0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 2 * Real.sqrt 19 ∧ θ = Real.arctan (Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1468_146819


namespace NUMINAMATH_CALUDE_function_pair_properties_l1468_146899

/-- Real functions c and s defined on ℝ\{0} satisfying certain properties -/
def FunctionPair (c s : ℝ → ℝ) : Prop :=
  (∀ x, x ≠ 0 → c x ≠ 0 ∧ s x ≠ 0) ∧
  (∀ x y, x ≠ 0 → y ≠ 0 → c (x / y) = c x * c y - s x * s y)

/-- Properties of the function pair c and s -/
theorem function_pair_properties (c s : ℝ → ℝ) (h : FunctionPair c s) :
  (∀ x, x ≠ 0 → c (1 / x) = c x) ∧
  (∀ x, x ≠ 0 → s (1 / x) = -s x) ∧
  (c 1 = 1) ∧
  (s 1 = 0) ∧
  (s (-1) = 0) ∧
  ((∀ x, c (-x) = c x ∧ s (-x) = s x) ∨ (∀ x, c (-x) = -c x ∧ s (-x) = -s x)) :=
sorry


end NUMINAMATH_CALUDE_function_pair_properties_l1468_146899


namespace NUMINAMATH_CALUDE_expression_evaluation_l1468_146894

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1468_146894


namespace NUMINAMATH_CALUDE_ray_gave_25_cents_to_peter_l1468_146863

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the initial amount Ray had in cents -/
def initial_amount : ℕ := 95

/-- Represents the number of nickels Ray had left -/
def nickels_left : ℕ := 4

/-- Represents the amount given to Peter in cents -/
def amount_to_peter : ℕ := 25

/-- Proves that Ray gave 25 cents to Peter given the initial conditions -/
theorem ray_gave_25_cents_to_peter :
  let total_given := initial_amount - (nickels_left * nickel_value)
  let amount_to_randi := 2 * amount_to_peter
  total_given = amount_to_peter + amount_to_randi :=
by sorry

end NUMINAMATH_CALUDE_ray_gave_25_cents_to_peter_l1468_146863


namespace NUMINAMATH_CALUDE_dance_team_recruitment_l1468_146850

theorem dance_team_recruitment :
  ∀ (track_team choir dance_team : ℕ),
  track_team + choir + dance_team = 100 →
  choir = 2 * track_team →
  dance_team = choir + 10 →
  dance_team = 46 := by
sorry

end NUMINAMATH_CALUDE_dance_team_recruitment_l1468_146850


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l1468_146874

theorem fraction_of_powers_equals_500 : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l1468_146874


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1468_146802

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1468_146802


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l1468_146822

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 4 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l1468_146822


namespace NUMINAMATH_CALUDE_smallest_cube_divisor_l1468_146821

theorem smallest_cube_divisor (p q r s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → Nat.Prime s →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  (∀ m : ℕ, m > 0 → m^3 % (p * q^2 * r^4 * s^3) = 0 → m ≥ p * q * r^2 * s) ∧
  (p * q * r^2 * s)^3 % (p * q^2 * r^4 * s^3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_divisor_l1468_146821


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l1468_146845

/-- Given a rectangular room with specified dimensions and total paving cost,
    calculate the rate per square meter for paving the floor. -/
theorem paving_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 5.5)
    (h_width : width = 3.75)
    (h_total_cost : total_cost = 16500) : 
  total_cost / (length * width) = 800 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l1468_146845


namespace NUMINAMATH_CALUDE_xiaohui_wins_l1468_146878

/-- Represents a student with their scores -/
structure Student where
  name : String
  mandarin : ℕ
  sports : ℕ
  tourism : ℕ

/-- Calculates the weighted score for a student given the weights -/
def weightedScore (s : Student) (w1 w2 w3 : ℕ) : ℚ :=
  (s.mandarin * w1 + s.sports * w2 + s.tourism * w3 : ℚ) / (w1 + w2 + w3 : ℚ)

/-- The theorem stating that Xiaohui wins -/
theorem xiaohui_wins : 
  let xiaocong : Student := ⟨"Xiaocong", 80, 90, 72⟩
  let xiaohui : Student := ⟨"Xiaohui", 90, 80, 70⟩
  weightedScore xiaohui 4 3 3 > weightedScore xiaocong 4 3 3 := by
  sorry


end NUMINAMATH_CALUDE_xiaohui_wins_l1468_146878


namespace NUMINAMATH_CALUDE_line_circle_orthogonality_l1468_146859

/-- Given a line and a circle, prove that a specific value of 'a' ensures orthogonality of OA and OB -/
theorem line_circle_orthogonality (a : ℝ) (A B : ℝ × ℝ) :
  (∀ (x y : ℝ), x - 2*y + a = 0 → x^2 + y^2 = 2) →  -- Line intersects circle
  (A.1 - 2*A.2 + a = 0 ∧ A.1^2 + A.2^2 = 2) →        -- A satisfies both equations
  (B.1 - 2*B.2 + a = 0 ∧ B.1^2 + B.2^2 = 2) →        -- B satisfies both equations
  a = Real.sqrt 5 →                                  -- Specific value of a
  A.1 * B.1 + A.2 * B.2 = 0                          -- OA · OB = 0
  := by sorry

end NUMINAMATH_CALUDE_line_circle_orthogonality_l1468_146859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1468_146833

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the sequence is 8 -/
def a_15_eq_8 (a : ℕ → ℚ) : Prop := a 15 = 8

/-- The 60th term of the sequence is 20 -/
def a_60_eq_20 (a : ℕ → ℚ) : Prop := a 60 = 20

/-- Theorem: In an arithmetic sequence where a_15 = 8 and a_60 = 20, a_75 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a_15_eq_8 a) 
  (h3 : a_60_eq_20 a) : 
  a 75 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1468_146833


namespace NUMINAMATH_CALUDE_large_cube_volume_l1468_146869

/-- The volume of a cube constructed from smaller cubes -/
theorem large_cube_volume (n : ℕ) (edge : ℝ) (h : n = 125) (h_edge : edge = 2) :
  (n : ℝ) * (edge ^ 3) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_volume_l1468_146869


namespace NUMINAMATH_CALUDE_expo_arrangements_l1468_146865

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of volunteers. -/
def num_volunteers : ℕ := 5

/-- The number of foreign friends. -/
def num_foreign_friends : ℕ := 2

/-- The total number of people. -/
def total_people : ℕ := num_volunteers + num_foreign_friends

/-- The number of positions where the foreign friends can be placed. -/
def foreign_friend_positions : ℕ := total_people - num_foreign_friends - 1

theorem expo_arrangements : 
  choose foreign_friend_positions 1 * arrangements num_volunteers * arrangements num_foreign_friends = 960 := by
  sorry

end NUMINAMATH_CALUDE_expo_arrangements_l1468_146865


namespace NUMINAMATH_CALUDE_equivalent_operations_l1468_146880

theorem equivalent_operations (x : ℚ) : (x * (3/4)) / (3/5) = x * (5/4) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l1468_146880


namespace NUMINAMATH_CALUDE_value_of_x_l1468_146800

theorem value_of_x (w y z x : ℝ) 
  (hw : w = 90)
  (hz : z = 2/3 * w)
  (hy : y = 1/4 * z)
  (hx : x = 1/2 * y) : 
  x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1468_146800


namespace NUMINAMATH_CALUDE_share_of_y_l1468_146807

def total_amount : ℕ := 690
def ratio_x : ℕ := 5
def ratio_y : ℕ := 7
def ratio_z : ℕ := 11

theorem share_of_y : 
  (total_amount * ratio_y) / (ratio_x + ratio_y + ratio_z) = 210 := by
  sorry

end NUMINAMATH_CALUDE_share_of_y_l1468_146807


namespace NUMINAMATH_CALUDE_jumping_contest_l1468_146890

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : mouse_jump = 31)
  (h3 : mouse_jump + 26 = frog_jump)
  (h4 : frog_jump > grasshopper_jump) :
  frog_jump - grasshopper_jump = 32 := by
sorry


end NUMINAMATH_CALUDE_jumping_contest_l1468_146890


namespace NUMINAMATH_CALUDE_min_bullseyes_for_victory_l1468_146857

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : ℕ := 120
  halfway_point : ℕ := 60
  alex_lead_at_half : ℕ := 60
  bullseye_score : ℕ := 10
  alex_min_score : ℕ := 3

/-- Theorem stating the minimum number of bullseyes Alex needs to guarantee victory -/
theorem min_bullseyes_for_victory (comp : ArcheryCompetition) :
  ∃ n : ℕ, n = 52 ∧
  (∀ m : ℕ, -- m represents Alex's current score
    (comp.alex_lead_at_half + m = comp.halfway_point * comp.alex_min_score) →
    (m + n * comp.bullseye_score + (comp.halfway_point - n) * comp.alex_min_score >
     m - comp.alex_lead_at_half + comp.halfway_point * comp.bullseye_score) ∧
    (∀ k : ℕ, k < n →
      ∃ p : ℕ, p ≤ m - comp.alex_lead_at_half + comp.halfway_point * comp.bullseye_score ∧
      p ≥ m + k * comp.bullseye_score + (comp.halfway_point - k) * comp.alex_min_score)) :=
sorry

end NUMINAMATH_CALUDE_min_bullseyes_for_victory_l1468_146857


namespace NUMINAMATH_CALUDE_root_nature_depends_on_k_l1468_146804

theorem root_nature_depends_on_k :
  ∀ k : ℝ, ∃ Δ : ℝ, 
    (Δ = 1 + 4*k) ∧ 
    (Δ < 0 → (∀ x : ℝ, (x - 1) * (x - 2) ≠ k)) ∧
    (Δ = 0 → (∃! x : ℝ, (x - 1) * (x - 2) = k)) ∧
    (Δ > 0 → (∃ x y : ℝ, x ≠ y ∧ (x - 1) * (x - 2) = k ∧ (y - 1) * (y - 2) = k)) :=
by sorry


end NUMINAMATH_CALUDE_root_nature_depends_on_k_l1468_146804


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1468_146837

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1468_146837


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1468_146875

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement_equals_set : S ∩ (U \ T) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1468_146875


namespace NUMINAMATH_CALUDE_inverse_proportionality_l1468_146815

theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * (-10) = k) :
  x = 10 / 3 → y = -15 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l1468_146815


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1468_146898

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 6*y + k = 0 → y = x) → 
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1468_146898


namespace NUMINAMATH_CALUDE_mean_problem_l1468_146851

theorem mean_problem (x : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 → 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l1468_146851


namespace NUMINAMATH_CALUDE_rancher_loss_rancher_specific_loss_l1468_146871

/-- Calculates the total monetary loss for a rancher given specific conditions --/
theorem rancher_loss (initial_cattle : ℕ) (initial_rate : ℕ) (dead_cattle : ℕ) 
  (sick_cost : ℕ) (reduced_price : ℕ) : ℕ :=
  let expected_revenue := initial_cattle * initial_rate
  let remaining_cattle := initial_cattle - dead_cattle
  let revenue_remaining := remaining_cattle * reduced_price
  let additional_cost := dead_cattle * sick_cost
  let total_loss := (expected_revenue - revenue_remaining) + additional_cost
  total_loss

/-- Proves that the rancher's total monetary loss is $310,500 given the specific conditions --/
theorem rancher_specific_loss : 
  rancher_loss 500 700 350 80 450 = 310500 := by
  sorry

end NUMINAMATH_CALUDE_rancher_loss_rancher_specific_loss_l1468_146871


namespace NUMINAMATH_CALUDE_barry_votes_difference_l1468_146864

def election_votes (marcy_votes barry_votes joey_votes : ℕ) : Prop :=
  marcy_votes = 3 * barry_votes ∧
  ∃ x, barry_votes = 2 * (joey_votes + x) ∧
  marcy_votes = 66 ∧
  joey_votes = 8

theorem barry_votes_difference :
  ∀ marcy_votes barry_votes joey_votes,
  election_votes marcy_votes barry_votes joey_votes →
  barry_votes - joey_votes = 14 := by
sorry

end NUMINAMATH_CALUDE_barry_votes_difference_l1468_146864


namespace NUMINAMATH_CALUDE_pokemon_card_ratio_l1468_146879

theorem pokemon_card_ratio : ∀ (nicole cindy rex : ℕ),
  nicole = 400 →
  rex * 4 = 150 * 4 →
  2 * rex = nicole + cindy →
  cindy * 2 = nicole :=
by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_ratio_l1468_146879


namespace NUMINAMATH_CALUDE_largest_x_value_l1468_146801

theorem largest_x_value (x : ℝ) : 
  (((17 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 9 * x - 3) ∧ 
   (∀ y : ℝ, ((17 * y^2 - 40 * y + 15) / (4 * y - 3) + 7 * y = 9 * y - 3) → y ≤ x)) 
  → x = 2/3 := by sorry

end NUMINAMATH_CALUDE_largest_x_value_l1468_146801


namespace NUMINAMATH_CALUDE_f_minimum_at_negative_one_l1468_146848

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem f_minimum_at_negative_one :
  IsLocalMin f (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_f_minimum_at_negative_one_l1468_146848


namespace NUMINAMATH_CALUDE_magic_square_solution_l1468_146891

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  magic_sum : ℕ
  row_sum : a + b + c = magic_sum ∧ d + e + f = magic_sum ∧ g + h + i = magic_sum
  col_sum : a + d + g = magic_sum ∧ b + e + h = magic_sum ∧ c + f + i = magic_sum
  diag_sum : a + e + i = magic_sum ∧ c + e + g = magic_sum

/-- Theorem: In a 3x3 magic square with top row entries x, 23, 102 and middle-left entry 5, x must equal 208 -/
theorem magic_square_solution (ms : MagicSquare) (h1 : ms.b = 23) (h2 : ms.c = 102) (h3 : ms.d = 5) : ms.a = 208 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_solution_l1468_146891


namespace NUMINAMATH_CALUDE_cos_540_degrees_l1468_146866

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_540_degrees_l1468_146866


namespace NUMINAMATH_CALUDE_age_change_proof_l1468_146873

theorem age_change_proof (n : ℕ) (A : ℝ) : 
  ((n + 1) * (A + 7) = n * A + 39) →
  ((n + 1) * (A - 1) = n * A + 15) →
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_change_proof_l1468_146873


namespace NUMINAMATH_CALUDE_multiplication_properties_l1468_146860

theorem multiplication_properties (m n : ℕ) :
  let a := 6 * m + 1
  let b := 6 * n + 1
  let c := 6 * m + 5
  let d := 6 * n + 5
  (∃ k : ℕ, a * b = 6 * k + 1) ∧
  (∃ k : ℕ, c * d = 6 * k + 1) ∧
  (∃ k : ℕ, a * d = 6 * k + 5) :=
by sorry

end NUMINAMATH_CALUDE_multiplication_properties_l1468_146860


namespace NUMINAMATH_CALUDE_suzy_age_l1468_146847

theorem suzy_age (mary_age : ℕ) (suzy_age : ℕ) : 
  mary_age = 8 → 
  suzy_age + 4 = 2 * (mary_age + 4) → 
  suzy_age = 20 := by
sorry

end NUMINAMATH_CALUDE_suzy_age_l1468_146847


namespace NUMINAMATH_CALUDE_power_sum_of_i_l1468_146872

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^20 + i^39 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l1468_146872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1468_146836

/-- Given an arithmetic sequence where the sum of the first and third terms is 10,
    prove that the second term is 5. -/
theorem arithmetic_sequence_second_term 
  (a : ℝ) -- First term of the arithmetic sequence
  (d : ℝ) -- Common difference of the arithmetic sequence
  (h : a + (a + 2*d) = 10) -- Sum of first and third terms is 10
  : a + d = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1468_146836


namespace NUMINAMATH_CALUDE_cube_edge_length_is_ten_l1468_146835

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- Checks if a point is inside a cube -/
def isInside (p : Point3D) (c : Cube) : Prop :=
  0 < p.x ∧ p.x < c.edgeLength ∧
  0 < p.y ∧ p.y < c.edgeLength ∧
  0 < p.z ∧ p.z < c.edgeLength

/-- Theorem: If there exists an interior point with specific distances from four vertices of a cube,
    then the edge length of the cube is 10 -/
theorem cube_edge_length_is_ten (c : Cube) (p : Point3D) 
    (v1 v2 v3 v4 : Point3D) : 
    isInside p c →
    squaredDistance p v1 = 50 →
    squaredDistance p v2 = 70 →
    squaredDistance p v3 = 90 →
    squaredDistance p v4 = 110 →
    (v1.x = 0 ∨ v1.x = c.edgeLength) ∧ 
    (v1.y = 0 ∨ v1.y = c.edgeLength) ∧ 
    (v1.z = 0 ∨ v1.z = c.edgeLength) →
    (v2.x = 0 ∨ v2.x = c.edgeLength) ∧ 
    (v2.y = 0 ∨ v2.y = c.edgeLength) ∧ 
    (v2.z = 0 ∨ v2.z = c.edgeLength) →
    (v3.x = 0 ∨ v3.x = c.edgeLength) ∧ 
    (v3.y = 0 ∨ v3.y = c.edgeLength) ∧ 
    (v3.z = 0 ∨ v3.z = c.edgeLength) →
    (v4.x = 0 ∨ v4.x = c.edgeLength) ∧ 
    (v4.y = 0 ∨ v4.y = c.edgeLength) ∧ 
    (v4.z = 0 ∨ v4.z = c.edgeLength) →
    (v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4) →
    c.edgeLength = 10 := by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_is_ten_l1468_146835


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l1468_146817

/-- The equation represents two lines if it can be rewritten in the form of two linear equations -/
def represents_two_lines (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ x y, f x y = 0 ↔ (x = a * y + b ∨ x = c * y + d)

/-- The given equation -/
def equation (x y : ℝ) : ℝ :=
  x^2 - 25 * y^2 - 10 * x + 50

theorem equation_represents_two_lines :
  represents_two_lines equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l1468_146817


namespace NUMINAMATH_CALUDE_mixed_number_division_equality_l1468_146849

theorem mixed_number_division_equality :
  (4 + 2/3 + 5 + 1/4) / (3 + 1/2 - (2 + 3/5)) = 11 + 1/54 := by sorry

end NUMINAMATH_CALUDE_mixed_number_division_equality_l1468_146849


namespace NUMINAMATH_CALUDE_angle_value_l1468_146888

theorem angle_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan (α + β) = 3 →
  Real.tan β = 1/2 →
  α = π/4 := by sorry

end NUMINAMATH_CALUDE_angle_value_l1468_146888


namespace NUMINAMATH_CALUDE_max_area_rectangle_in_ellipse_l1468_146808

/-- Given an ellipse b² x² + a² y² = a² b², prove that the rectangle with the largest possible area
    inscribed in the ellipse has vertices at (±(a/2)√2, ±(b/2)√2) -/
theorem max_area_rectangle_in_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse := {p : ℝ × ℝ | b^2 * p.1^2 + a^2 * p.2^2 = a^2 * b^2}
  let inscribed_rectangle (p : ℝ × ℝ) := 
    {q : ℝ × ℝ | q ∈ ellipse ∧ |q.1| ≤ |p.1| ∧ |q.2| ≤ |p.2|}
  let area (p : ℝ × ℝ) := 4 * |p.1 * p.2|
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    (∀ q : ℝ × ℝ, q ∈ ellipse → area q ≤ area p) ∧
    p = (a/2 * Real.sqrt 2, b/2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_in_ellipse_l1468_146808


namespace NUMINAMATH_CALUDE_nancy_chips_l1468_146897

/-- Given that Nancy has 22 tortilla chips and gives 7 to her brother and 5 to her sister,
    prove that she keeps 10 chips for herself. -/
theorem nancy_chips (total : ℕ) (brother : ℕ) (sister : ℕ) 
    (h1 : total = 22) 
    (h2 : brother = 7) 
    (h3 : sister = 5) : 
  total - (brother + sister) = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancy_chips_l1468_146897


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1468_146813

def MonotonousFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ ∀ x y, x ≤ y → f x ≥ f y

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_mono : MonotonousFunction f)
  (h_eq : ∀ x, f (f x) = f (-f x) ∧ f (f x) = (f x)^2) :
  (∀ x, f x = 0) ∨ (∀ x, f x = 1) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1468_146813


namespace NUMINAMATH_CALUDE_unique_number_satisfies_equation_l1468_146893

theorem unique_number_satisfies_equation : ∃! x : ℝ, (60 + 12) / 3 = (x - 12) * 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfies_equation_l1468_146893


namespace NUMINAMATH_CALUDE_scaling_transformation_curve_l1468_146824

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem scaling_transformation_curve (x y x' y' : ℝ) :
  (x' = 5 * x) →
  (y' = 3 * y) →
  (x'^2 + y'^2 = 1) →
  (25 * x^2 + 9 * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_scaling_transformation_curve_l1468_146824


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1468_146831

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((100 : ℝ) * π / 180 - x * π / 180) =
    (Real.sin ((100 : ℝ) * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos ((100 : ℝ) * π / 180) - Real.cos (x * π / 180)) ∧
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1468_146831


namespace NUMINAMATH_CALUDE_max_value_of_f_l1468_146856

theorem max_value_of_f (x : ℝ) (h : x < 3) : 
  (4 / (x - 3) + x) ≤ -1 ∧ ∃ y < 3, 4 / (y - 3) + y = -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1468_146856


namespace NUMINAMATH_CALUDE_ps_length_approx_l1468_146896

/-- A quadrilateral with given side and diagonal segment lengths -/
structure Quadrilateral :=
  (QT TS PT TR PQ : ℝ)

/-- The length of PS in the quadrilateral -/
noncomputable def lengthPS (q : Quadrilateral) : ℝ :=
  Real.sqrt (q.PT^2 + q.TS^2 - 2 * q.PT * q.TS * (-((q.PQ^2 - q.PT^2 - q.QT^2) / (2 * q.PT * q.QT))))

/-- Theorem stating that for a quadrilateral with given measurements, PS ≈ 19.9 -/
theorem ps_length_approx (q : Quadrilateral) 
  (h1 : q.QT = 5) (h2 : q.TS = 7) (h3 : q.PT = 9) (h4 : q.TR = 4) (h5 : q.PQ = 7) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |lengthPS q - 19.9| < ε :=
sorry

end NUMINAMATH_CALUDE_ps_length_approx_l1468_146896


namespace NUMINAMATH_CALUDE_sum_of_u_and_v_l1468_146887

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 5 * u - 3 * v = 26)
  (eq2 : 3 * u + 5 * v = -19) : 
  u + v = -101 / 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_u_and_v_l1468_146887
