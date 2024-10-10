import Mathlib

namespace fraction_simplification_l437_43760

theorem fraction_simplification
  (a b c k : ℝ)
  (h1 : a * b = c * k)
  (h2 : c * k ≠ 0) :
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by sorry

end fraction_simplification_l437_43760


namespace f_decreasing_on_interval_l437_43767

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 6 * x^2 + 6 * x - 12

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 1, f' x < 0 :=
by sorry

end f_decreasing_on_interval_l437_43767


namespace practice_problems_count_l437_43785

theorem practice_problems_count (N : ℕ) 
  (h1 : N > 0)
  (h2 : (4 / 5 : ℚ) * (3 / 4 : ℚ) * (2 / 3 : ℚ) * N = 24) : N = 60 := by
  sorry

#check practice_problems_count

end practice_problems_count_l437_43785


namespace andrews_blue_balloons_l437_43782

/-- Given information about Andrew's balloons, prove the number of blue balloons. -/
theorem andrews_blue_balloons :
  ∀ (total_balloons remaining_balloons purple_balloons : ℕ),
    total_balloons = 2 * remaining_balloons →
    remaining_balloons = 378 →
    purple_balloons = 453 →
    total_balloons - purple_balloons = 303 := by
  sorry

end andrews_blue_balloons_l437_43782


namespace triple_minus_double_equals_eight_point_five_l437_43743

theorem triple_minus_double_equals_eight_point_five (x : ℝ) : 
  3 * x = 2 * x + 8.5 → 3 * x - 2 * x = 8.5 := by
  sorry

end triple_minus_double_equals_eight_point_five_l437_43743


namespace dan_marbles_count_l437_43715

/-- The total number of marbles Dan has after receiving red marbles from Mary -/
def total_marbles (violet_marbles red_marbles : ℕ) : ℕ :=
  violet_marbles + red_marbles

/-- Theorem stating that Dan has 78 marbles in total -/
theorem dan_marbles_count :
  total_marbles 64 14 = 78 := by
  sorry

end dan_marbles_count_l437_43715


namespace min_value_of_f_l437_43757

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -37) :=
by sorry

end min_value_of_f_l437_43757


namespace cube_division_theorem_l437_43778

/-- A point in 3D space represented by rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Represents that a point is inside the unit cube -/
def insideUnitCube (p : RationalPoint) : Prop :=
  0 < p.x ∧ p.x < 1 ∧ 0 < p.y ∧ p.y < 1 ∧ 0 < p.z ∧ p.z < 1

theorem cube_division_theorem (points : Finset RationalPoint) 
    (h : points.card = 2003) 
    (h_inside : ∀ p ∈ points, insideUnitCube p) :
    ∃ (n : ℕ), n > 2003 ∧ 
    ∀ p ∈ points, ∃ (i j k : ℕ), 
      i < n ∧ j < n ∧ k < n ∧
      (i : ℚ) / n < p.x ∧ p.x < ((i + 1) : ℚ) / n ∧
      (j : ℚ) / n < p.y ∧ p.y < ((j + 1) : ℚ) / n ∧
      (k : ℚ) / n < p.z ∧ p.z < ((k + 1) : ℚ) / n :=
by sorry


end cube_division_theorem_l437_43778


namespace polynomial_division_remainder_l437_43779

theorem polynomial_division_remainder (x : ℤ) : 
  x^1010 % ((x^2 - 1) * (x + 1)) = 1 := by sorry

end polynomial_division_remainder_l437_43779


namespace tens_digit_of_3_pow_405_l437_43773

theorem tens_digit_of_3_pow_405 : 3^405 % 100 = 43 := by
  sorry

end tens_digit_of_3_pow_405_l437_43773


namespace polynomial_simplification_l437_43741

theorem polynomial_simplification (x : ℝ) : 
  (3*x + 2) * (3*x - 2) - (3*x - 1)^2 = 6*x - 5 := by
  sorry

end polynomial_simplification_l437_43741


namespace fraction_calculation_l437_43792

theorem fraction_calculation : 
  (((1 / 2 : ℚ) + (1 / 3)) / ((2 / 7 : ℚ) + (1 / 4))) * (3 / 5) = 14 / 15 := by
  sorry

end fraction_calculation_l437_43792


namespace min_distance_B_to_M_l437_43716

-- Define the rectilinear distance function
def rectilinearDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the point B
def B : ℝ × ℝ := (1, 1)

-- Define the line on which M moves
def lineM (x y : ℝ) : Prop :=
  x - y + 4 = 0

-- Theorem statement
theorem min_distance_B_to_M :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (x y : ℝ), lineM x y →
    rectilinearDistance B.1 B.2 x y ≥ min :=
sorry

end min_distance_B_to_M_l437_43716


namespace pentagon_point_reconstruction_l437_43714

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D' -/
theorem pentagon_point_reconstruction (A B C D E A' B' C' D' : ℝ × ℝ) : 
  A'B = 2 * AB → 
  B'C = BC → 
  C'D = CD → 
  D'E = 2 * DE → 
  A = (1/9 : ℝ) • A' + (2/9 : ℝ) • B' + (4/9 : ℝ) • C' + (8/9 : ℝ) • D' := by
  sorry


end pentagon_point_reconstruction_l437_43714


namespace system_solution_ratio_l437_43719

theorem system_solution_ratio (a b x y : ℝ) : 
  8 * x - 6 * y = a →
  9 * x - 12 * y = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = 8 / 9 := by
sorry

end system_solution_ratio_l437_43719


namespace train_crossing_time_l437_43783

/-- Proves the time taken by a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) 
  (time_to_cross_pole : ℝ) (h1 : train_length = 300) 
  (h2 : platform_length = 675) (h3 : time_to_cross_pole = 12) : 
  (train_length + platform_length) / (train_length / time_to_cross_pole) = 39 := by
  sorry

#check train_crossing_time

end train_crossing_time_l437_43783


namespace divisor_problem_l437_43749

theorem divisor_problem : ∃ (x : ℕ), x > 0 ∧ 181 = 9 * x + 1 :=
by
  -- The proof goes here
  sorry

end divisor_problem_l437_43749


namespace tan_sum_minus_product_62_73_l437_43744

theorem tan_sum_minus_product_62_73 :
  Real.tan (62 * π / 180) + Real.tan (73 * π / 180) - 
  Real.tan (62 * π / 180) * Real.tan (73 * π / 180) = -1 := by
  sorry

end tan_sum_minus_product_62_73_l437_43744


namespace tangent_slope_implies_a_value_l437_43738

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a_value :
  ∀ a : ℝ, f' a (-1) = 8 → a = -6 :=
by
  sorry

#check tangent_slope_implies_a_value

end tangent_slope_implies_a_value_l437_43738


namespace max_notebook_price_l437_43771

def entrance_fee : ℕ := 3
def total_budget : ℕ := 160
def num_notebooks : ℕ := 15
def tax_rate : ℚ := 8 / 100

theorem max_notebook_price :
  ∃ (price : ℕ),
    price ≤ 9 ∧
    (price : ℚ) * (1 + tax_rate) * num_notebooks + entrance_fee ≤ total_budget ∧
    ∀ (p : ℕ), p > price →
      (p : ℚ) * (1 + tax_rate) * num_notebooks + entrance_fee > total_budget :=
by sorry

end max_notebook_price_l437_43771


namespace smallest_prime_dividing_sum_l437_43740

theorem smallest_prime_dividing_sum : ∃ (p : Nat), 
  Prime p ∧ 
  p ∣ (2^14 + 7^9) ∧ 
  ∀ (q : Nat), Prime q → q ∣ (2^14 + 7^9) → p ≤ q :=
by
  sorry

end smallest_prime_dividing_sum_l437_43740


namespace balloon_arrangement_count_l437_43761

def balloon_arrangements : ℕ :=
  let total_letters := 7
  let num_L := 2
  let num_O := 3
  let remaining_letters := 3  -- B, A, N
  let block_positions := 4    -- LLLOOO can be in 4 positions
  
  block_positions * remaining_letters.factorial * 
  (total_letters.factorial / (num_L.factorial * num_O.factorial))

theorem balloon_arrangement_count : balloon_arrangements = 10080 := by
  sorry

end balloon_arrangement_count_l437_43761


namespace percentage_relation_l437_43745

theorem percentage_relation (A B T : ℝ) 
  (h1 : A = 0.2 * B) 
  (h2 : B = 0.3 * T) : 
  A = 0.06 * T := by
  sorry

end percentage_relation_l437_43745


namespace expected_value_coin_flip_l437_43753

/-- The expected value of a coin flip game -/
theorem expected_value_coin_flip (p_heads : ℚ) (p_tails : ℚ) 
  (win_heads : ℚ) (lose_tails : ℚ) : 
  p_heads = 1/3 → p_tails = 2/3 → win_heads = 3 → lose_tails = 2 →
  p_heads * win_heads - p_tails * lose_tails = -1/3 := by
  sorry

#check expected_value_coin_flip

end expected_value_coin_flip_l437_43753


namespace triangle_area_l437_43734

theorem triangle_area (R : ℝ) (A : ℝ) (b c : ℝ) (h1 : R = 4) (h2 : A = π / 3) (h3 : b - c = 4) :
  let S := (1 / 2) * b * c * Real.sin A
  S = 8 * Real.sqrt 3 := by sorry

end triangle_area_l437_43734


namespace sum_of_roots_is_12_l437_43790

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (3 + x) = g (3 - x)

-- Define a proposition that g has exactly four distinct real roots
def has_four_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d))

-- The theorem statement
theorem sum_of_roots_is_12 (g : ℝ → ℝ) 
    (h1 : symmetric_about_3 g) 
    (h2 : has_four_distinct_roots g) : 
  ∃ a b c d : ℝ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (a + b + c + d = 12) :=
sorry

end sum_of_roots_is_12_l437_43790


namespace partition_exists_l437_43752

-- Define the type for our partition function
def PartitionFunction := ℕ+ → Fin 100

-- Define the property that the partition satisfies the required condition
def SatisfiesCondition (f : PartitionFunction) : Prop :=
  ∀ a b c : ℕ+, a + 99 * b = c → f a = f b ∨ f a = f c ∨ f b = f c

-- State the theorem
theorem partition_exists : ∃ f : PartitionFunction, SatisfiesCondition f := by
  sorry

end partition_exists_l437_43752


namespace root_sum_pow_l437_43794

theorem root_sum_pow (p q : ℝ) : 
  p^2 - 7*p + 12 = 0 → 
  q^2 - 7*q + 12 = 0 → 
  p^3 + p^4*q^2 + p^2*q^4 + q^3 = 3691 := by
sorry

end root_sum_pow_l437_43794


namespace g_limit_pos_infinity_g_limit_neg_infinity_g_behavior_l437_43729

/-- The polynomial function g(x) -/
def g (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + x - 9

/-- Theorem stating that g(x) approaches infinity as x approaches positive infinity -/
theorem g_limit_pos_infinity : 
  Filter.Tendsto g Filter.atTop Filter.atTop :=
sorry

/-- Theorem stating that g(x) approaches infinity as x approaches negative infinity -/
theorem g_limit_neg_infinity : 
  Filter.Tendsto g Filter.atBot Filter.atTop :=
sorry

/-- Main theorem combining both limits to show the behavior of g(x) -/
theorem g_behavior : 
  (Filter.Tendsto g Filter.atTop Filter.atTop) ∧ 
  (Filter.Tendsto g Filter.atBot Filter.atTop) :=
sorry

end g_limit_pos_infinity_g_limit_neg_infinity_g_behavior_l437_43729


namespace ivan_purchase_cost_l437_43798

/-- Calculates the total cost of a discounted purchase -/
def discounted_purchase_cost (original_price discount quantity : ℕ) : ℕ :=
  (original_price - discount) * quantity

/-- Proves that the total cost for Ivan's purchase is $100 -/
theorem ivan_purchase_cost :
  discounted_purchase_cost 12 2 10 = 100 := by
  sorry

end ivan_purchase_cost_l437_43798


namespace systematic_sample_theorem_l437_43733

/-- Systematic sampling function that returns the nth element of the sample -/
def systematicSample (populationSize sampleSize start n : ℕ) : ℕ :=
  start + (populationSize / sampleSize) * n

/-- Theorem: In a systematic sample of size 5 from a population of 55,
    if students 3, 25, and 47 are in the sample,
    then the other two students in the sample have numbers 14 and 36 -/
theorem systematic_sample_theorem :
  let populationSize : ℕ := 55
  let sampleSize : ℕ := 5
  let start : ℕ := 3
  (systematicSample populationSize sampleSize start 0 = 3) →
  (systematicSample populationSize sampleSize start 2 = 25) →
  (systematicSample populationSize sampleSize start 4 = 47) →
  (systematicSample populationSize sampleSize start 1 = 14) ∧
  (systematicSample populationSize sampleSize start 3 = 36) :=
by
  sorry


end systematic_sample_theorem_l437_43733


namespace class_size_l437_43713

/-- Proves that in a class where the number of girls is 0.4 of the number of boys
    and there are 10 girls, the total number of students is 35. -/
theorem class_size (boys girls : ℕ) : 
  girls = 10 → 
  girls = (2 / 5 : ℚ) * boys → 
  boys + girls = 35 := by
sorry

end class_size_l437_43713


namespace basketball_league_games_l437_43754

/-- The number of games played in a basketball league season -/
def total_games (n : ℕ) (games_per_pairing : ℕ) : ℕ :=
  n * (n - 1) * games_per_pairing / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180. -/
theorem basketball_league_games :
  total_games 10 4 = 180 := by
  sorry

end basketball_league_games_l437_43754


namespace banana_fraction_proof_l437_43786

theorem banana_fraction_proof (jefferson_bananas : ℕ) (walter_bananas : ℚ) (f : ℚ) :
  jefferson_bananas = 56 →
  walter_bananas = 56 - 56 * f →
  (56 + (56 - 56 * f)) / 2 = 49 →
  f = 1/4 := by
sorry

end banana_fraction_proof_l437_43786


namespace profit_and_marginal_profit_maxima_l437_43710

def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
def C (x : ℕ) : ℝ := 600 * x + 2000
def p (x : ℕ) : ℝ := R x - C x
def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_and_marginal_profit_maxima 
  (h : ∀ x : ℕ, 0 < x ∧ x ≤ 100) :
  (∃ x : ℕ, p x = 74000 ∧ ∀ y : ℕ, p y ≤ 74000) ∧
  (∃ x : ℕ, Mp x = 2340 ∧ ∀ y : ℕ, Mp y ≤ 2340) :=
sorry

end profit_and_marginal_profit_maxima_l437_43710


namespace expression_equality_l437_43775

theorem expression_equality : 
  Real.sqrt 32 + (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) - Real.sqrt 4 - 6 * Real.sqrt (1/2) = Real.sqrt 2 - 1 := by
  sorry

end expression_equality_l437_43775


namespace calculator_reciprocal_l437_43787

theorem calculator_reciprocal (x : ℝ) :
  (1 / (1/x - 1)) - 1 = -0.75 → x = 0.2 := by
  sorry

end calculator_reciprocal_l437_43787


namespace unique_a_for_cubic_property_l437_43781

theorem unique_a_for_cubic_property (a : ℕ+) :
  (∀ n : ℕ+, ∃ k : ℤ, 4 * (a.val ^ n.val + 1) = k ^ 3) →
  a = 1 :=
by sorry

end unique_a_for_cubic_property_l437_43781


namespace rap_song_requests_l437_43706

/-- Represents the number of song requests for different genres in a night --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests given the conditions --/
theorem rap_song_requests (s : SongRequests) : s.rap = 2 :=
  by
  have h1 : s.total = 30 := by sorry
  have h2 : s.electropop = s.total / 2 := by sorry
  have h3 : s.dance = s.electropop / 3 := by sorry
  have h4 : s.rock = 5 := by sorry
  have h5 : s.oldies = s.rock - 3 := by sorry
  have h6 : s.dj_choice = s.oldies / 2 := by sorry
  have h7 : s.total = s.electropop + s.dance + s.rock + s.oldies + s.dj_choice + s.rap := by sorry
  sorry

end rap_song_requests_l437_43706


namespace closest_point_parabola_to_line_l437_43725

/-- The point (1, 1) on the parabola y^2 = x is the closest point to the line x - 2y + 4 = 0 -/
theorem closest_point_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let line := {p : ℝ × ℝ | p.1 - 2*p.2 + 4 = 0}
  let distance (p : ℝ × ℝ) := |p.1 - 2*p.2 + 4| / Real.sqrt 5
  ∀ p ∈ parabola, distance (1, 1) ≤ distance p :=
by sorry

end closest_point_parabola_to_line_l437_43725


namespace num_pencils_is_75_l437_43717

/-- The number of pencils purchased given the conditions of the problem -/
def num_pencils : ℕ :=
  let num_pens : ℕ := 30
  let total_cost : ℕ := 570
  let pencil_price : ℕ := 2
  let pen_price : ℕ := 14
  let pen_cost : ℕ := num_pens * pen_price
  let pencil_cost : ℕ := total_cost - pen_cost
  pencil_cost / pencil_price

theorem num_pencils_is_75 : num_pencils = 75 := by
  sorry

end num_pencils_is_75_l437_43717


namespace smallest_two_digit_prime_with_reverse_property_l437_43756

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := n / 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_two_digit_prime_with_reverse_property : 
  ∀ n : ℕ, 
    n ≥ 20 ∧ n < 30 ∧ 
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (reverse_digits n % 3 = 0 ∨ reverse_digits n % 7 = 0) →
    n ≥ 21 :=
sorry

end smallest_two_digit_prime_with_reverse_property_l437_43756


namespace tangent_difference_l437_43750

/-- Given two circles in a plane, this theorem proves that the difference between
    the squares of their external and internal tangent lengths is 30. -/
theorem tangent_difference (r₁ r₂ x y A₁₀ : ℝ) : 
  r₁ > 0 → r₂ > 0 → A₁₀ > 0 →
  r₁ * r₂ = 15 / 2 →
  x^2 + (r₁ + r₂)^2 = A₁₀^2 →
  y^2 + (r₁ - r₂)^2 = A₁₀^2 →
  y^2 - x^2 = 30 := by
sorry

end tangent_difference_l437_43750


namespace monotonicity_and_slope_conditions_l437_43768

-- Define the function f
def f (a b x : ℝ) : ℝ := -x^3 + x^2 + a*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := -3*x^2 + 2*x + a

theorem monotonicity_and_slope_conditions (a b : ℝ) :
  -- Part 1: Monotonicity when a = 3
  (∀ x ∈ Set.Ioo (-1 : ℝ) 3, (f' 3 x > 0)) ∧
  (∀ x ∈ Set.Iic (-1 : ℝ), (f' 3 x < 0)) ∧
  (∀ x ∈ Set.Ici 3, (f' 3 x < 0)) ∧
  -- Part 2: Condition on a based on slope
  ((∀ x : ℝ, f' a x < 2*a^2) → (a > 1 ∨ a < -1/2)) :=
sorry

end monotonicity_and_slope_conditions_l437_43768


namespace science_fair_competition_l437_43721

theorem science_fair_competition (k h n : ℕ) : 
  h = (3 * k) / 5 →
  n = 2 * (k + h) →
  k + h + n = 240 →
  k = 50 ∧ h = 30 ∧ n = 160 := by
sorry

end science_fair_competition_l437_43721


namespace alyssa_puppies_left_l437_43759

/-- The number of puppies Alyssa has left after giving some away -/
def puppies_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Alyssa has 5 puppies left -/
theorem alyssa_puppies_left : puppies_left 12 7 = 5 := by
  sorry

end alyssa_puppies_left_l437_43759


namespace pascals_triangle_56th_row_second_to_last_l437_43700

theorem pascals_triangle_56th_row_second_to_last : Nat.choose 56 55 = 56 := by
  sorry

end pascals_triangle_56th_row_second_to_last_l437_43700


namespace quadratic_sum_l437_43718

/-- Given a quadratic function f(x) = 8x^2 - 48x - 320, prove that when written in the form a(x+b)^2+c, the sum a + b + c equals -387. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 8*x^2 - 48*x - 320) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = -387 := by
  sorry

end quadratic_sum_l437_43718


namespace coloring_scheme_exists_l437_43769

-- Define the color type
inductive Color
| White
| Red
| Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Statement of the theorem
theorem coloring_scheme_exists : ∃ (f : ColoringFunction),
  (∀ c : Color, ∃ (S : Set ℤ), Set.Infinite S ∧ 
    ∀ y ∈ S, Set.Infinite {x : ℤ | f (x, y) = c}) ∧
  (∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    f (x₁, y₁) = Color.White →
    f (x₂, y₂) = Color.Black →
    f (x₃, y₃) = Color.Red →
    f (x₁ + x₂ - x₃, y₁ + y₂ - y₃) = Color.Red) :=
by sorry


end coloring_scheme_exists_l437_43769


namespace f_of_3_equals_155_l437_43732

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

-- Theorem statement
theorem f_of_3_equals_155 : f 3 = 155 := by
  sorry

end f_of_3_equals_155_l437_43732


namespace percentage_of_375_l437_43784

theorem percentage_of_375 (x : ℝ) :
  (x / 100) * 375 = 5.4375 → x = 1.45 := by
  sorry

end percentage_of_375_l437_43784


namespace number_of_boys_in_class_l437_43762

/-- Given the conditions of a class of boys with height measurements:
    - initial_average: The initially calculated average height
    - wrong_height: The wrongly recorded height of one boy
    - correct_height: The correct height of the boy with the wrong measurement
    - actual_average: The actual average height after correction
    
    Prove that the number of boys in the class is equal to the given value.
-/
theorem number_of_boys_in_class 
  (initial_average : ℝ) 
  (wrong_height : ℝ) 
  (correct_height : ℝ) 
  (actual_average : ℝ) 
  (h1 : initial_average = 180) 
  (h2 : wrong_height = 156) 
  (h3 : correct_height = 106) 
  (h4 : actual_average = 178) : 
  ∃ n : ℕ, n * actual_average = n * initial_average - (wrong_height - correct_height) ∧ n = 25 :=
by sorry

end number_of_boys_in_class_l437_43762


namespace number_problem_l437_43796

theorem number_problem (x : ℝ) : 0.20 * x - 4 = 6 → x = 50 := by
  sorry

end number_problem_l437_43796


namespace parabola_y_intercepts_l437_43702

/-- The number of y-intercepts of the parabola x = 3y^2 - 6y + 3 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 6 * y + 3
  ∃! y : ℝ, f y = 0 := by sorry

end parabola_y_intercepts_l437_43702


namespace rectangle_perimeter_l437_43704

theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  let square_side := square_perimeter / 4
  let rect_length := square_side
  let rect_width := square_side / 4
  let rect_perimeter := 2 * (rect_length + rect_width)
  rect_perimeter = 100 := by sorry

end rectangle_perimeter_l437_43704


namespace volunteer_selection_theorem_l437_43764

/-- The number of volunteers --/
def n : ℕ := 20

/-- The number of volunteers to be selected --/
def k : ℕ := 4

/-- The number of the first specific volunteer that must be selected --/
def a : ℕ := 5

/-- The number of the second specific volunteer that must be selected --/
def b : ℕ := 14

/-- The number of volunteers with numbers less than the first specific volunteer --/
def m : ℕ := a - 1

/-- The number of volunteers with numbers greater than the second specific volunteer --/
def p : ℕ := n - b

/-- The total number of ways to select the volunteers under the given conditions --/
def total_ways : ℕ := Nat.choose m 2 + Nat.choose p 2

theorem volunteer_selection_theorem :
  total_ways = 21 := by sorry

end volunteer_selection_theorem_l437_43764


namespace triangle_properties_l437_43789

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (B = π / 4 → Real.sqrt 3 * b = Real.sqrt 2 * a) ∧
  (a = Real.sqrt 3 ∧ b + c = 3 → b * c = 2) :=
by sorry

end triangle_properties_l437_43789


namespace henry_age_is_27_l437_43755

/-- Henry's present age -/
def henry_age : ℕ := sorry

/-- Jill's present age -/
def jill_age : ℕ := sorry

/-- The sum of Henry and Jill's present ages is 43 -/
axiom sum_of_ages : henry_age + jill_age = 43

/-- 5 years ago, Henry was twice the age of Jill -/
axiom age_relation : henry_age - 5 = 2 * (jill_age - 5)

/-- Theorem: Henry's present age is 27 years -/
theorem henry_age_is_27 : henry_age = 27 := by sorry

end henry_age_is_27_l437_43755


namespace aquarium_fish_problem_l437_43723

theorem aquarium_fish_problem (initial_fish : ℕ) : 
  initial_fish > 0 → initial_fish + 3 = 13 :=
by
  sorry

end aquarium_fish_problem_l437_43723


namespace sqrt_three_inequality_l437_43742

theorem sqrt_three_inequality (n : ℕ+) :
  (n : ℝ) + 3 < n * Real.sqrt 3 ∧ n * Real.sqrt 3 < (n : ℝ) + 4 → n = 4 := by
  sorry

end sqrt_three_inequality_l437_43742


namespace coffee_shop_weekly_production_l437_43731

/-- Represents the brewing rate and operating hours of a coffee shop for a specific day type -/
structure DayType where
  brewingRate : ℕ
  operatingHours : ℕ

/-- Calculates the total number of coffee cups brewed in a week -/
def totalCupsPerWeek (weekday : DayType) (weekend : DayType) : ℕ :=
  weekday.brewingRate * weekday.operatingHours * 5 +
  weekend.brewingRate * weekend.operatingHours * 2

/-- Theorem: The coffee shop brews 400 cups in a week -/
theorem coffee_shop_weekly_production :
  let weekday : DayType := { brewingRate := 10, operatingHours := 5 }
  let weekend : DayType := { brewingRate := 15, operatingHours := 5 }
  totalCupsPerWeek weekday { weekend with operatingHours := 6 } = 400 :=
by sorry

end coffee_shop_weekly_production_l437_43731


namespace geometric_sum_problem_l437_43763

-- Define the sum of a geometric sequence
def GeometricSum (n : ℕ) := ℝ

-- State the theorem
theorem geometric_sum_problem :
  ∀ (S : ℕ → ℝ),
  (S 2 = 4) →
  (S 4 = 6) →
  (S 6 = 7) :=
by sorry

end geometric_sum_problem_l437_43763


namespace ellipse_eccentricity_condition_l437_43708

/-- The eccentricity of an ellipse with equation x^2 + y^2/m = 1 (m > 0) is greater than 1/2
    if and only if 0 < m < 4/3 or m > 3/4 -/
theorem ellipse_eccentricity_condition (m : ℝ) :
  (m > 0) →
  (∃ (x y : ℝ), x^2 + y^2/m = 1) →
  (∃ (e : ℝ), e > 1/2 ∧ e^2 = 1 - (min 1 m) / (max 1 m)) ↔
  (0 < m ∧ m < 4/3) ∨ m > 3/4 :=
by sorry

end ellipse_eccentricity_condition_l437_43708


namespace parabola_focus_coordinates_l437_43701

/-- The focus of the parabola y = 8x^2 has coordinates (0, 1/32) -/
theorem parabola_focus_coordinates :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - 8 * x^2
  ∃! p : ℝ × ℝ, p = (0, 1/32) ∧ ∀ x y, f (x, y) = 0 → (x - p.1)^2 = 4 * p.2 * (y - p.2) :=
sorry

end parabola_focus_coordinates_l437_43701


namespace weekly_running_distance_l437_43793

/-- Calculates the total distance run in a week given the track length, loops per day, and days per week. -/
def total_distance (track_length : ℕ) (loops_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  track_length * loops_per_day * days_per_week

/-- Theorem stating that running 10 loops per day on a 50-meter track for 7 days results in 3500 meters per week. -/
theorem weekly_running_distance :
  total_distance 50 10 7 = 3500 := by
  sorry

end weekly_running_distance_l437_43793


namespace five_consecutive_not_square_l437_43705

theorem five_consecutive_not_square (n : ℤ) : 
  ∃ (m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ≠ m ^ 2 := by
  sorry

end five_consecutive_not_square_l437_43705


namespace quartet_characterization_l437_43797

def is_valid_quartet (a b c d : ℕ+) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b = c * d ∧ a * b = c + d

def valid_quartets : List (ℕ+ × ℕ+ × ℕ+ × ℕ+) :=
  [(1, 5, 3, 2), (1, 5, 2, 3), (5, 1, 3, 2), (5, 1, 2, 3),
   (2, 3, 1, 5), (3, 2, 1, 5), (2, 3, 5, 1), (3, 2, 5, 1)]

theorem quartet_characterization (a b c d : ℕ+) :
  is_valid_quartet a b c d ↔ (a, b, c, d) ∈ valid_quartets :=
sorry

end quartet_characterization_l437_43797


namespace peytons_score_l437_43727

theorem peytons_score (n : ℕ) (avg_14 : ℚ) (avg_15 : ℚ) (peyton_score : ℚ) : 
  n = 15 → 
  avg_14 = 80 → 
  avg_15 = 81 → 
  (n - 1) * avg_14 + peyton_score = n * avg_15 →
  peyton_score = 95 := by
  sorry

end peytons_score_l437_43727


namespace area_BEDC_is_30_l437_43758

/-- Represents a parallelogram ABCD with a line DE parallel to AB -/
structure Parallelogram :=
  (AB : ℝ)
  (height : ℝ)
  (DE : ℝ)
  (is_parallelogram : Bool)
  (DE_parallel_AB : Bool)
  (E_midpoint_DC : Bool)

/-- Calculate the area of region BEDC in the given parallelogram -/
def area_BEDC (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area of region BEDC is 30 under given conditions -/
theorem area_BEDC_is_30 (p : Parallelogram) 
  (h1 : p.AB = 12)
  (h2 : p.height = 10)
  (h3 : p.DE = 6)
  (h4 : p.is_parallelogram = true)
  (h5 : p.DE_parallel_AB = true)
  (h6 : p.E_midpoint_DC = true) :
  area_BEDC p = 30 :=
sorry

end area_BEDC_is_30_l437_43758


namespace monotonic_decreasing_interval_l437_43724

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (∀ y : ℝ, y > x → f y < f x) :=
by sorry

end monotonic_decreasing_interval_l437_43724


namespace equilateral_triangle_segment_length_l437_43780

-- Define the triangle and points
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

def EquilateralTriangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  Triangle A B C ∧ dist A B = dist B C

def OnSegment (X P Q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • P + t • Q

-- State the theorem
theorem equilateral_triangle_segment_length 
  (A B C K L M : EuclideanSpace ℝ (Fin 2)) :
  EquilateralTriangle A B C →
  OnSegment K A B →
  OnSegment L B C →
  OnSegment M B C →
  OnSegment L B M →
  dist K L = dist K M →
  dist B L = 2 →
  dist A K = 3 →
  dist C M = 5 := by
sorry

end equilateral_triangle_segment_length_l437_43780


namespace product_of_divisors_60_has_three_prime_factors_l437_43766

def divisors (n : ℕ) : Finset ℕ :=
  sorry

def product_of_divisors (n : ℕ) : ℕ :=
  (divisors n).prod id

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  sorry

theorem product_of_divisors_60_has_three_prime_factors :
  num_distinct_prime_factors (product_of_divisors 60) = 3 :=
sorry

end product_of_divisors_60_has_three_prime_factors_l437_43766


namespace cos_two_alpha_value_l437_43748

theorem cos_two_alpha_value (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 := by
  sorry

end cos_two_alpha_value_l437_43748


namespace square_eq_neg_two_i_implies_a_eq_one_coordinates_of_z_over_one_plus_i_l437_43735

-- Define the complex number z
def z (a : ℝ) : ℂ := a - Complex.I

-- Theorem 1
theorem square_eq_neg_two_i_implies_a_eq_one :
  ∀ a : ℝ, (z a)^2 = -2 * Complex.I → a = 1 := by sorry

-- Theorem 2
theorem coordinates_of_z_over_one_plus_i :
  let z : ℂ := z 2
  (z / (1 + Complex.I)).re = 1/2 ∧ (z / (1 + Complex.I)).im = -3/2 := by sorry

end square_eq_neg_two_i_implies_a_eq_one_coordinates_of_z_over_one_plus_i_l437_43735


namespace plan_y_cheaper_at_601_l437_43737

/-- Represents an internet service plan with a flat fee and per-gigabyte charge -/
structure InternetPlan where
  flatFee : ℕ
  perGBCharge : ℕ

/-- Calculates the total cost in cents for a given plan and number of gigabytes -/
def totalCost (plan : InternetPlan) (gigabytes : ℕ) : ℕ :=
  plan.flatFee * 100 + plan.perGBCharge * gigabytes

theorem plan_y_cheaper_at_601 :
  let planX : InternetPlan := ⟨50, 15⟩
  let planY : InternetPlan := ⟨80, 10⟩
  ∀ g : ℕ, g ≥ 601 → totalCost planY g < totalCost planX g ∧
  ∀ g : ℕ, g < 601 → totalCost planX g ≤ totalCost planY g :=
by sorry

end plan_y_cheaper_at_601_l437_43737


namespace cubic_gp_roots_iff_a_60_l437_43703

/-- A cubic polynomial with parameter a -/
def cubic (a : ℝ) (x : ℝ) : ℝ := x^3 - 15*x^2 + a*x - 64

/-- Predicate for three distinct real roots in geometric progression -/
def has_three_distinct_gp_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    cubic a x₁ = 0 ∧ cubic a x₂ = 0 ∧ cubic a x₃ = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ x₂ = x₁ * q ∧ x₃ = x₂ * q

/-- The main theorem -/
theorem cubic_gp_roots_iff_a_60 :
  ∀ a : ℝ, has_three_distinct_gp_roots a ↔ a = 60 := by sorry

end cubic_gp_roots_iff_a_60_l437_43703


namespace intersection_value_l437_43720

theorem intersection_value (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | x ≤ 1} →
  B = {x : ℝ | x ≥ a} →
  A ∩ B = {1} →
  a = 1 := by
sorry

end intersection_value_l437_43720


namespace sum_of_four_real_numbers_l437_43776

theorem sum_of_four_real_numbers (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_eq : (a^2 + b^2 - 1)*(a + b) = (b^2 + c^2 - 1)*(b + c) ∧ 
          (b^2 + c^2 - 1)*(b + c) = (c^2 + d^2 - 1)*(c + d)) : 
  a + b + c + d = 0 := by
sorry

end sum_of_four_real_numbers_l437_43776


namespace solve_steak_problem_l437_43707

def steak_problem (cost_per_pound change_received : ℕ) : Prop :=
  let amount_paid : ℕ := 20
  let amount_spent : ℕ := amount_paid - change_received
  let pounds_bought : ℕ := amount_spent / cost_per_pound
  (cost_per_pound = 7 ∧ change_received = 6) → pounds_bought = 2

theorem solve_steak_problem :
  ∀ (cost_per_pound change_received : ℕ),
    steak_problem cost_per_pound change_received :=
by
  sorry

end solve_steak_problem_l437_43707


namespace final_bill_is_520_20_l437_43746

/-- The final bill amount after applying two consecutive 2% late charges to an original bill of $500 -/
def final_bill_amount (original_bill : ℝ) (late_charge_rate : ℝ) : ℝ :=
  original_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

/-- Theorem stating that the final bill amount is $520.20 -/
theorem final_bill_is_520_20 :
  final_bill_amount 500 0.02 = 520.20 := by sorry

end final_bill_is_520_20_l437_43746


namespace hyperbola_vertex_distance_l437_43722

/-- The distance between the vertices of a hyperbola with equation x^2/36 - y^2/25 = 1 is 12 -/
theorem hyperbola_vertex_distance : 
  let a : ℝ := Real.sqrt 36
  let b : ℝ := Real.sqrt 25
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 36 - y^2 / 25 = 1
  2 * a = 12 := by sorry

end hyperbola_vertex_distance_l437_43722


namespace no_primes_divisible_by_45_l437_43711

-- Definition of a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem statement
theorem no_primes_divisible_by_45 : ¬∃ p : ℕ, isPrime p ∧ 45 ∣ p := by
  sorry

end no_primes_divisible_by_45_l437_43711


namespace tangent_line_intersection_extreme_values_l437_43709

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x - 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem for the tangent line intersection points
theorem tangent_line_intersection (x₀ : ℝ) (b : ℝ) :
  f' x₀ = -9 ∧ f x₀ = -9 * x₀ + b → b = -3 ∨ b = -7 :=
sorry

-- Theorem for the extreme values of f(x)
theorem extreme_values :
  (∃ x : ℝ, f x = 2 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -30 ∧ ∀ y : ℝ, f y ≥ f x) :=
sorry

end tangent_line_intersection_extreme_values_l437_43709


namespace problem_2023_l437_43788

theorem problem_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end problem_2023_l437_43788


namespace num_divisors_2310_l437_43772

/-- The number of positive divisors of 2310 is 32 -/
theorem num_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end num_divisors_2310_l437_43772


namespace largest_N_for_dispersive_connective_perm_l437_43728

/-- The set of residues modulo 17 -/
def X : Set ℕ := {x | x < 17}

/-- Two numbers in X are adjacent if they differ by 1 or are 0 and 16 -/
def adjacent (a b : ℕ) : Prop :=
  (a ∈ X ∧ b ∈ X) ∧ ((a + 1 ≡ b [ZMOD 17]) ∨ (b + 1 ≡ a [ZMOD 17]))

/-- A permutation on X -/
def permutation_on_X (p : ℕ → ℕ) : Prop :=
  Function.Bijective p ∧ ∀ x, x ∈ X → p x ∈ X

/-- A permutation is dispersive if it never maps adjacent values to adjacent values -/
def dispersive (p : ℕ → ℕ) : Prop :=
  permutation_on_X p ∧ ∀ a b, adjacent a b → ¬adjacent (p a) (p b)

/-- A permutation is connective if it always maps adjacent values to adjacent values -/
def connective (p : ℕ → ℕ) : Prop :=
  permutation_on_X p ∧ ∀ a b, adjacent a b → adjacent (p a) (p b)

/-- The composition of a permutation with itself n times -/
def iterate_perm (p : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => id
  | n + 1 => p ∘ (iterate_perm p n)

/-- The theorem stating the largest N for which the described permutation exists -/
theorem largest_N_for_dispersive_connective_perm :
  ∃ (p : ℕ → ℕ), permutation_on_X p ∧
    (∀ k < 8, dispersive (iterate_perm p k)) ∧
    connective (iterate_perm p 8) ∧
    ∀ (q : ℕ → ℕ) (m : ℕ),
      (permutation_on_X q ∧
       (∀ k < m, dispersive (iterate_perm q k)) ∧
       connective (iterate_perm q m)) →
      m ≤ 8 :=
sorry

end largest_N_for_dispersive_connective_perm_l437_43728


namespace tunnel_crossing_possible_l437_43726

/-- Represents a friend with their crossing time -/
structure Friend where
  name : String
  time : Nat

/-- Represents a crossing of the tunnel -/
inductive Crossing
  | Forward : List Friend → Crossing
  | Backward : Friend → Crossing

/-- Calculates the time taken for a crossing -/
def crossingTime (c : Crossing) : Nat :=
  match c with
  | Crossing.Forward friends => friends.map Friend.time |>.maximum?.getD 0
  | Crossing.Backward friend => friend.time

/-- The tunnel crossing problem -/
def tunnelCrossing (friends : List Friend) : Prop :=
  ∃ (crossings : List Crossing),
    -- All friends have crossed
    (crossings.filter (λ c => match c with
      | Crossing.Forward _ => true
      | Crossing.Backward _ => false
    )).bind (λ c => match c with
      | Crossing.Forward fs => fs
      | Crossing.Backward _ => []
    ) = friends
    ∧
    -- The total time is exactly 17 minutes
    (crossings.map crossingTime).sum = 17
    ∧
    -- Each crossing involves at most two friends
    ∀ c ∈ crossings, match c with
      | Crossing.Forward fs => fs.length ≤ 2
      | Crossing.Backward _ => true

theorem tunnel_crossing_possible : 
  let friends := [
    { name := "One", time := 1 },
    { name := "Two", time := 2 },
    { name := "Five", time := 5 },
    { name := "Ten", time := 10 }
  ]
  tunnelCrossing friends :=
by
  sorry


end tunnel_crossing_possible_l437_43726


namespace boys_employed_is_50_l437_43747

/-- Represents the roadway construction scenario --/
structure RoadwayConstruction where
  totalLength : ℝ
  totalTime : ℝ
  initialLength : ℝ
  initialTime : ℝ
  initialMen : ℕ
  initialHours : ℝ
  overtimeHours : ℝ
  boyEfficiency : ℝ

/-- Calculates the number of boys employed in the roadway construction --/
def calculateBoysEmployed (rc : RoadwayConstruction) : ℕ :=
  sorry

/-- Theorem stating that the number of boys employed is 50 --/
theorem boys_employed_is_50 (rc : RoadwayConstruction) : 
  rc.totalLength = 15 ∧ 
  rc.totalTime = 40 ∧ 
  rc.initialLength = 3 ∧ 
  rc.initialTime = 10 ∧ 
  rc.initialMen = 180 ∧ 
  rc.initialHours = 8 ∧ 
  rc.overtimeHours = 1 ∧ 
  rc.boyEfficiency = 2/3 → 
  calculateBoysEmployed rc = 50 := by
  sorry

end boys_employed_is_50_l437_43747


namespace live_bargaining_theorem_l437_43730

/-- Represents the price reduction scenario in a live streaming bargaining event. -/
def live_bargaining_price_reduction (initial_price final_price : ℝ) (num_rounds : ℕ) (reduction_rate : ℝ) : Prop :=
  initial_price * (1 - reduction_rate) ^ num_rounds = final_price

/-- The live bargaining price reduction theorem. -/
theorem live_bargaining_theorem :
  ∃ (x : ℝ), live_bargaining_price_reduction 120 43.2 2 x :=
sorry

end live_bargaining_theorem_l437_43730


namespace modular_congruence_solution_l437_43791

theorem modular_congruence_solution : ∃! n : ℕ, 1 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end modular_congruence_solution_l437_43791


namespace jar_price_calculation_l437_43751

noncomputable def jar_price (d h p : ℝ) (d' h' : ℝ) : ℝ :=
  p * (d' / d)^2 * (h' / h)

theorem jar_price_calculation (d₁ h₁ p₁ d₂ h₂ : ℝ) 
  (hd₁ : d₁ = 2) (hh₁ : h₁ = 5) (hp₁ : p₁ = 0.75)
  (hd₂ : d₂ = 4) (hh₂ : h₂ = 8) :
  jar_price d₁ h₁ p₁ d₂ h₂ = 2.40 := by
  sorry

end jar_price_calculation_l437_43751


namespace ap_to_gp_ratio_is_positive_integer_l437_43739

/-- An arithmetic progression starting with 1 -/
def AP (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => AP x n + (x - 1)

/-- A geometric progression starting with 1 -/
def GP (a : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => GP a n * a

/-- The property that a GP is formed by deleting some terms from an AP -/
def isSubsequence (x a : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, GP a n = AP x m

theorem ap_to_gp_ratio_is_positive_integer (x : ℝ) (hx : x ≥ 1) (a : ℝ) (ha : a > 0)
    (h : isSubsequence x a) : ∃ k : ℕ+, a = k :=
  sorry

end ap_to_gp_ratio_is_positive_integer_l437_43739


namespace brian_chris_fishing_l437_43765

theorem brian_chris_fishing (brian_trips chris_trips : ℕ) 
  (brian_fish_per_trip : ℕ) (total_fish : ℕ) :
  brian_trips = 2 * chris_trips →
  brian_fish_per_trip = 400 →
  chris_trips = 10 →
  total_fish = 13600 →
  (1 - (brian_fish_per_trip : ℚ) / ((total_fish - brian_trips * brian_fish_per_trip) / chris_trips : ℚ)) = 2/7 := by
sorry

end brian_chris_fishing_l437_43765


namespace equation_solutions_l437_43774

theorem equation_solutions :
  (∃ x : ℝ, x - 0.4 * x = 120 ∧ x = 200) ∧
  (∃ x : ℝ, 5 * x - 5 / 6 = 5 / 4 ∧ x = 5 / 12) := by
  sorry

end equation_solutions_l437_43774


namespace copper_part_mass_l437_43795

/-- Given two parts with equal volume, one made of aluminum and one made of copper,
    prove that the mass of the copper part is approximately 0.086 kg. -/
theorem copper_part_mass
  (ρ_A : Real) -- density of aluminum
  (ρ_M : Real) -- density of copper
  (Δm : Real)  -- mass difference between parts
  (h1 : ρ_A = 2700) -- density of aluminum in kg/m³
  (h2 : ρ_M = 8900) -- density of copper in kg/m³
  (h3 : Δm = 0.06)  -- mass difference in kg
  : ∃ (m_M : Real), abs (m_M - 0.086) < 0.001 ∧ 
    ∃ (V : Real), V > 0 ∧ V = m_M / ρ_M ∧ V = (m_M - Δm) / ρ_A :=
by sorry

end copper_part_mass_l437_43795


namespace special_array_determination_l437_43770

/-- Represents an m×n array of positive integers -/
def SpecialArray (m n : ℕ) := Fin m → Fin n → ℕ+

/-- The condition that must hold for any four numbers in the array -/
def SpecialCondition (A : SpecialArray m n) : Prop :=
  ∀ (i₁ i₂ : Fin m) (j₁ j₂ : Fin n),
    A i₁ j₁ + A i₂ j₂ = A i₁ j₂ + A i₂ j₁

/-- The theorem stating that m+n-1 elements are sufficient to determine the entire array -/
theorem special_array_determination (m n : ℕ) (A : SpecialArray m n) 
  (hA : SpecialCondition A) :
  ∃ (S : Finset ((Fin m) × (Fin n))),
    S.card = m + n - 1 ∧ 
    (∀ (B : SpecialArray m n), 
      SpecialCondition B → 
      (∀ (p : (Fin m) × (Fin n)), p ∈ S → A p.1 p.2 = B p.1 p.2) → 
      A = B) :=
sorry

end special_array_determination_l437_43770


namespace class_gender_ratio_l437_43712

theorem class_gender_ratio (total_students : ℕ) (female_students : ℕ) : 
  total_students = 52 → female_students = 13 → 
  (total_students - female_students) / female_students = 3 := by
  sorry

end class_gender_ratio_l437_43712


namespace hyperbola_focal_length_l437_43799

/-- The focal length of the hyperbola y²/9 - x²/7 = 1 is 8 -/
theorem hyperbola_focal_length : ∃ (a b c : ℝ), 
  (a^2 = 9 ∧ b^2 = 7) → 
  (∀ (x y : ℝ), y^2 / 9 - x^2 / 7 = 1 → (x / a)^2 - (y / b)^2 = 1) →
  c^2 = a^2 + b^2 →
  2 * c = 8 := by sorry

end hyperbola_focal_length_l437_43799


namespace ratio_x_to_y_l437_43736

def total_amount : ℕ := 5000
def x_amount : ℕ := 1000

theorem ratio_x_to_y :
  (x_amount : ℚ) / (total_amount - x_amount : ℚ) = 1 / 4 := by
  sorry

end ratio_x_to_y_l437_43736


namespace complex_expression_value_l437_43777

theorem complex_expression_value : 
  ∃ (i : ℂ), i^2 = -1 ∧ i^3 * (1 + i)^2 = 2 := by
  sorry

end complex_expression_value_l437_43777
