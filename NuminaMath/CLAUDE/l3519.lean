import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptote_relation_l3519_351924

/-- A hyperbola with equation x^2/a + y^2/9 = 1 and asymptotes 3x ± 2y = 0 has a = -4 -/
theorem hyperbola_asymptote_relation (a : ℝ) :
  (∀ x y : ℝ, x^2/a + y^2/9 = 1 ↔ (3*x - 2*y = 0 ∨ 3*x + 2*y = 0)) →
  a = -4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_relation_l3519_351924


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l3519_351959

/-- The complex number z -/
def z : ℂ := (-8 + Complex.I) * Complex.I

/-- A complex number is in the third quadrant if its real part is negative and its imaginary part is negative -/
def is_in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

/-- Theorem: z is located in the third quadrant of the complex plane -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l3519_351959


namespace NUMINAMATH_CALUDE_third_group_men_count_l3519_351958

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (m w : WorkRate) (g : WorkGroup) : ℝ :=
  g.men * m.rate + g.women * w.rate

theorem third_group_men_count 
  (m w : WorkRate) 
  (g1 g2 : WorkGroup) 
  (h1 : totalWorkRate m w g1 = totalWorkRate m w g2)
  (h2 : g1.men = 3 ∧ g1.women = 8)
  (h3 : g2.men = 6 ∧ g2.women = 2)
  (g3 : WorkGroup)
  (h4 : g3.women = 3)
  (h5 : totalWorkRate m w g3 = 0.5 * totalWorkRate m w g1) :
  g3.men = 2 := by
sorry

end NUMINAMATH_CALUDE_third_group_men_count_l3519_351958


namespace NUMINAMATH_CALUDE_male_average_grade_l3519_351913

/-- Proves that the average grade of male students is 87 given the conditions of the problem -/
theorem male_average_grade (total_average : ℝ) (female_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : female_average = 92)
  (h3 : male_count = 8)
  (h4 : female_count = 12) :
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 87 := by
  sorry

end NUMINAMATH_CALUDE_male_average_grade_l3519_351913


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3519_351952

def p : ℕ := (List.range 34).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 15 ↔ (3^k : ℕ) ∣ p := by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3519_351952


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3519_351921

theorem quadratic_equation_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l3519_351921


namespace NUMINAMATH_CALUDE_big_stack_pancakes_l3519_351947

/-- The number of pancakes in a big stack at Hank's cafe. -/
def big_stack : ℕ := sorry

/-- The number of pancakes in a short stack at Hank's cafe. -/
def short_stack : ℕ := 3

/-- The number of customers who ordered short stack. -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack. -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes Hank needs to make. -/
def total_pancakes : ℕ := 57

/-- Theorem stating that the number of pancakes in a big stack is 5. -/
theorem big_stack_pancakes : 
  short_stack * short_stack_orders + big_stack * big_stack_orders = total_pancakes → 
  big_stack = 5 := by sorry

end NUMINAMATH_CALUDE_big_stack_pancakes_l3519_351947


namespace NUMINAMATH_CALUDE_bridge_bricks_l3519_351993

theorem bridge_bricks (type_a : ℕ) (type_b : ℕ) (other_types : ℕ) : 
  type_a ≥ 40 →
  type_b = type_a / 2 →
  type_a + type_b + other_types = 150 →
  other_types = 90 := by
sorry

end NUMINAMATH_CALUDE_bridge_bricks_l3519_351993


namespace NUMINAMATH_CALUDE_stationery_difference_is_fifty_l3519_351900

/-- The number of stationery pieces Georgia has -/
def georgia_stationery : ℕ := 25

/-- The number of stationery pieces Lorene has -/
def lorene_stationery : ℕ := 3 * georgia_stationery

/-- The difference in stationery pieces between Lorene and Georgia -/
def stationery_difference : ℕ := lorene_stationery - georgia_stationery

theorem stationery_difference_is_fifty : stationery_difference = 50 := by
  sorry

end NUMINAMATH_CALUDE_stationery_difference_is_fifty_l3519_351900


namespace NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l3519_351940

def T : Finset ℕ := Finset.range 11

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (fun i => S.sum (fun j => if i > j then 3^i - 3^j else 0))

theorem difference_sum_of_powers_of_three :
  difference_sum T = 783492 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l3519_351940


namespace NUMINAMATH_CALUDE_least_number_divisible_l3519_351966

def numbers : List ℕ := [52, 84, 114, 133, 221, 379]

def result : ℕ := 1097897218492

theorem least_number_divisible (n : ℕ) : n = result ↔ 
  (∀ m ∈ numbers, (n + 20) % m = 0) ∧ 
  (∀ k < n, ∃ m ∈ numbers, (k + 20) % m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_l3519_351966


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_30_l3519_351962

theorem sum_of_fifth_powers_divisible_by_30 (a b c : ℤ) (h : 30 ∣ (a + b + c)) :
  30 ∣ (a^5 + b^5 + c^5) := by sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_30_l3519_351962


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3519_351951

theorem sum_of_roots_quadratic (x : ℝ) : 
  x^2 - 6*x + 8 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x^2 - 6*x + 8 = (x - r₁) * (x - r₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3519_351951


namespace NUMINAMATH_CALUDE_num_fm_pairs_is_four_l3519_351936

/-- The number of possible (f,m) pairs for 7 people at a round table -/
def num_fm_pairs : ℕ :=
  let people : ℕ := 7
  4

/-- Theorem: The number of possible (f,m) pairs for 7 people at a round table is 4 -/
theorem num_fm_pairs_is_four :
  num_fm_pairs = 4 := by sorry

end NUMINAMATH_CALUDE_num_fm_pairs_is_four_l3519_351936


namespace NUMINAMATH_CALUDE_time_to_produce_one_item_l3519_351944

/-- Given a machine that can produce 300 items in 2 hours, 
    prove that it takes 0.4 minutes to produce one item. -/
theorem time_to_produce_one_item 
  (total_time : ℝ) 
  (total_items : ℕ) 
  (h1 : total_time = 2) 
  (h2 : total_items = 300) : 
  (total_time / total_items) * 60 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_time_to_produce_one_item_l3519_351944


namespace NUMINAMATH_CALUDE_football_group_size_l3519_351987

/-- The proportion of people who like football -/
def like_football_ratio : ℚ := 24 / 60

/-- The proportion of people who play football among those who like it -/
def play_football_ratio : ℚ := 1 / 2

/-- The number of people expected to play football -/
def expected_players : ℕ := 50

/-- The total number of people in the group -/
def total_people : ℕ := 250

theorem football_group_size :
  (↑expected_players : ℚ) = like_football_ratio * play_football_ratio * total_people :=
sorry

end NUMINAMATH_CALUDE_football_group_size_l3519_351987


namespace NUMINAMATH_CALUDE_tan_30_plus_4sin_30_l3519_351912

/-- The tangent of 30 degrees plus 4 times the sine of 30 degrees equals (√3)/3 + 2 -/
theorem tan_30_plus_4sin_30 : Real.tan (30 * π / 180) + 4 * Real.sin (30 * π / 180) = (Real.sqrt 3) / 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_4sin_30_l3519_351912


namespace NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l3519_351903

/-- A permutation of the first n natural numbers. -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Predicate to check if a permutation satisfies the average condition. -/
def SatisfiesAverageCondition (n : ℕ) (p : Permutation n) : Prop :=
  ∀ i j k : Fin n, i < j → j < k →
    (p i).val + (p k).val ≠ 2 * (p j).val

/-- Theorem stating that for any n, there exists a permutation satisfying the average condition. -/
theorem exists_permutation_satisfying_average_condition (n : ℕ) :
  ∃ p : Permutation n, SatisfiesAverageCondition n p :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l3519_351903


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3519_351970

theorem complex_equation_solution (z : ℂ) :
  (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I →
  z = 3/5 + 12/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3519_351970


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l3519_351968

/-- Represents the game board -/
def GameBoard := Fin 2020 → Fin 2020 → Option Bool

/-- Checks if there are k consecutive cells of the same color in a row or column -/
def has_k_consecutive (board : GameBoard) (k : ℕ) : Prop :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameBoard → Fin 2020 × Fin 2020

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (strategy : Strategy) (k : ℕ) : Prop :=
  sorry

/-- The main theorem stating the condition for the first player's winning strategy -/
theorem first_player_winning_strategy :
  ∀ k : ℕ, (∃ strategy : Strategy, is_winning_strategy strategy k) ↔ k ≤ 1011 :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l3519_351968


namespace NUMINAMATH_CALUDE_min_value_expression_l3519_351920

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ y₀ / x₀ + 4 / y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3519_351920


namespace NUMINAMATH_CALUDE_graces_age_fraction_l3519_351939

theorem graces_age_fraction (mother_age : ℕ) (grace_age : ℕ) :
  mother_age = 80 →
  grace_age = 60 →
  (grace_age : ℚ) / ((2 * mother_age) : ℚ) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_graces_age_fraction_l3519_351939


namespace NUMINAMATH_CALUDE_number_problem_l3519_351964

theorem number_problem : ∃ x : ℝ, x = 1/8 + 0.675 ∧ x = 0.800 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3519_351964


namespace NUMINAMATH_CALUDE_f_zero_and_no_extreme_value_l3519_351935

noncomputable section

/-- The function f(x) = (x+2)lnx + ax^2 - 4x + 7a -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) * Real.log x + a * x^2 - 4 * x + 7 * a

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x + (x + 2) / x + 2 * a * x - 4

theorem f_zero_and_no_extreme_value :
  (∀ x > 0, f (1/2) x = 0 ↔ x = 1) ∧
  (∀ a ≥ 1/2, ∀ x > 0, f_derivative a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_zero_and_no_extreme_value_l3519_351935


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3519_351933

/-- Prove that for given conditions, the interest rate is 4% --/
theorem interest_rate_problem (P t : ℝ) (diff : ℝ) (h1 : P = 2000) (h2 : t = 2) (h3 : diff = 3.20) :
  ∃ r : ℝ, r = 4 ∧ 
    P * ((1 + r / 100) ^ t - 1) - (P * r * t / 100) = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3519_351933


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3519_351909

theorem multiply_and_simplify (x : ℝ) : 
  (x^6 + 64*x^3 + 4096) * (x^3 - 64) = x^9 - 262144 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3519_351909


namespace NUMINAMATH_CALUDE_min_value_fraction_l3519_351905

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 2) :
  (x + y + z) / (x * y * z) ≥ 27 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3519_351905


namespace NUMINAMATH_CALUDE_mixtape_length_example_l3519_351998

/-- The length of a mixtape given the number of songs on each side and the length of each song. -/
def mixtape_length (side1_songs : ℕ) (side2_songs : ℕ) (song_length : ℕ) : ℕ :=
  (side1_songs + side2_songs) * song_length

/-- Theorem stating that a mixtape with 6 songs on the first side, 4 songs on the second side,
    and each song being 4 minutes long has a total length of 40 minutes. -/
theorem mixtape_length_example : mixtape_length 6 4 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_mixtape_length_example_l3519_351998


namespace NUMINAMATH_CALUDE_divisor_ratio_of_M_l3519_351946

def M : ℕ := 36 * 45 * 98 * 160

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- The ratio of sum of odd divisors to sum of even divisors -/
def divisor_ratio (n : ℕ) : ℚ :=
  (sum_odd_divisors n : ℚ) / (sum_even_divisors n : ℚ)

theorem divisor_ratio_of_M :
  divisor_ratio M = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_divisor_ratio_of_M_l3519_351946


namespace NUMINAMATH_CALUDE_fewer_servings_l3519_351985

def total_ounces : ℕ := 64
def old_serving_size : ℕ := 8
def new_serving_size : ℕ := 16

theorem fewer_servings :
  (total_ounces / old_serving_size) - (total_ounces / new_serving_size) = 4 :=
by sorry

end NUMINAMATH_CALUDE_fewer_servings_l3519_351985


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3519_351943

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 285600) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3519_351943


namespace NUMINAMATH_CALUDE_jamie_grape_juice_theorem_l3519_351955

/-- The amount of grape juice Jamie had at recess -/
def grape_juice_amount (max_liquid bathroom_threshold planned_water milk_amount : ℕ) : ℕ :=
  max_liquid - bathroom_threshold - planned_water - milk_amount

theorem jamie_grape_juice_theorem :
  grape_juice_amount 32 0 8 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jamie_grape_juice_theorem_l3519_351955


namespace NUMINAMATH_CALUDE_brads_running_speed_l3519_351980

/-- Prove Brad's running speed given the conditions of the problem -/
theorem brads_running_speed 
  (total_distance : ℝ) 
  (maxwells_speed : ℝ) 
  (maxwells_distance : ℝ) 
  (h1 : total_distance = 40) 
  (h2 : maxwells_speed = 3) 
  (h3 : maxwells_distance = 15) : 
  ∃ (brads_speed : ℝ), brads_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_brads_running_speed_l3519_351980


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3519_351919

/-- Given vectors a and b, function f, and triangle ABC, prove the perimeter range -/
theorem triangle_perimeter_range (x : ℝ) :
  let a : ℝ × ℝ := (1, Real.sin x)
  let b : ℝ × ℝ := (Real.cos (2*x + π/3), Real.sin x)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2 - (1/2) * Real.cos (2*x)
  let c : ℝ := Real.sqrt 3
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    f C = 0 ∧
    2 * Real.sqrt 3 < A + B + c ∧ A + B + c ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3519_351919


namespace NUMINAMATH_CALUDE_min_value_of_c_l3519_351942

theorem min_value_of_c (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 →
  c = b + 1 →
  d = c + 1 →
  e = d + 1 →
  ∃ n : ℕ, a + b + c + d + e = n^3 →
  ∃ m : ℕ, b + c + d = m^2 →
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
    b' = a' + 1 ∧ c' = b' + 1 ∧ d' = c' + 1 ∧ e' = d' + 1 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3 ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2) →
  c' ≥ c →
  c = 675 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_c_l3519_351942


namespace NUMINAMATH_CALUDE_flower_pots_theorem_l3519_351937

/-- Represents a set of items with increasing prices -/
structure IncreasingPriceSet where
  num_items : ℕ
  price_difference : ℚ
  total_cost : ℚ

/-- The cost of the most expensive item in the set -/
def most_expensive_item_cost (s : IncreasingPriceSet) : ℚ :=
  (s.total_cost - (s.num_items - 1) * s.num_items * s.price_difference / 2) / s.num_items + (s.num_items - 1) * s.price_difference

/-- Theorem: For a set of 6 items with $0.15 price difference and $8.25 total cost, 
    the most expensive item costs $1.75 -/
theorem flower_pots_theorem : 
  let s : IncreasingPriceSet := ⟨6, 15/100, 825/100⟩
  most_expensive_item_cost s = 175/100 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_theorem_l3519_351937


namespace NUMINAMATH_CALUDE_circle_radius_l3519_351954

theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 + 4*y + 13 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l3519_351954


namespace NUMINAMATH_CALUDE_boat_journey_distance_l3519_351963

-- Define the constants
def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the theorem
theorem boat_journey_distance :
  ∃ (x : ℝ), 
    (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time ∧ x = 20) ∨
    (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time ∧ x = 20/3) :=
by sorry


end NUMINAMATH_CALUDE_boat_journey_distance_l3519_351963


namespace NUMINAMATH_CALUDE_factorization_2a_squared_minus_2a_l3519_351994

theorem factorization_2a_squared_minus_2a (a : ℝ) : 2*a^2 - 2*a = 2*a*(a-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2a_squared_minus_2a_l3519_351994


namespace NUMINAMATH_CALUDE_meeting_impossible_l3519_351991

-- Define the type for people in the meeting
def Person : Type := ℕ

-- Define the relationship of knowing each other
def knows (p q : Person) : Prop := sorry

-- Define the number of people in the meeting
def num_people : ℕ := 65

-- State the conditions of the problem
axiom condition1 : ∀ p : Person, ∃ S : Finset Person, S.card ≥ 56 ∧ ∀ q ∈ S, ¬knows p q

axiom condition2 : ∀ p q : Person, p ≠ q → ∃ r : Person, r ≠ p ∧ r ≠ q ∧ knows r p ∧ knows r q

-- The theorem to be proved
theorem meeting_impossible : False := sorry

end NUMINAMATH_CALUDE_meeting_impossible_l3519_351991


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l3519_351923

theorem lcm_gcf_ratio : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l3519_351923


namespace NUMINAMATH_CALUDE_inequality_proof_l3519_351949

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / Real.sqrt (a^2 + a + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3519_351949


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_min_value_achievable_l3519_351992

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y :=
by sorry

theorem min_value_is_four (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  1 / m + 2 / n ≥ 4 :=
by sorry

theorem min_value_achievable (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_min_value_achievable_l3519_351992


namespace NUMINAMATH_CALUDE_vector_operation_l3519_351931

/-- Given two vectors a and b in R², prove that 2a - b equals (0,5) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, -1)) :
  (2 : ℝ) • a - b = (0, 5) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3519_351931


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3519_351971

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

theorem intersection_of_M_and_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3519_351971


namespace NUMINAMATH_CALUDE_materik_position_l3519_351988

def Alphabet : Finset Char := {'A', 'E', 'I', 'K', 'M', 'R', 'T'}

def Word := List Char

def isValidWord (w : Word) : Prop :=
  w.length = 7 ∧ w.toFinset = Alphabet

def alphabeticalOrder (order : List Char) : Prop :=
  order.length = 7 ∧ order.toFinset = Alphabet

def wordPosition (w : Word) (order : List Char) : ℕ :=
  sorry

theorem materik_position 
  (order : List Char) 
  (h_order : alphabeticalOrder order) 
  (h_metrika : wordPosition ['M', 'E', 'T', 'R', 'I', 'K', 'A'] order = 3634) :
  wordPosition ['M', 'A', 'T', 'E', 'R', 'I', 'K'] order = 3745 :=
sorry

end NUMINAMATH_CALUDE_materik_position_l3519_351988


namespace NUMINAMATH_CALUDE_second_man_speed_l3519_351928

/-- Given two men walking in the same direction for 1 hour, where one walks at 10 kmph
    and they end up 2 km apart, the speed of the second man is 8 kmph. -/
theorem second_man_speed (speed_first : ℝ) (distance_apart : ℝ) (time : ℝ) (speed_second : ℝ) :
  speed_first = 10 →
  distance_apart = 2 →
  time = 1 →
  speed_first - speed_second = distance_apart / time →
  speed_second = 8 := by
sorry

end NUMINAMATH_CALUDE_second_man_speed_l3519_351928


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l3519_351934

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l3519_351934


namespace NUMINAMATH_CALUDE_return_speed_theorem_l3519_351981

theorem return_speed_theorem (v : ℕ) : 
  v > 50 ∧ 
  v ≤ 100 ∧ 
  (∃ k : ℕ, k = (100 * v) / (50 + v)) → 
  v = 75 := by
sorry

end NUMINAMATH_CALUDE_return_speed_theorem_l3519_351981


namespace NUMINAMATH_CALUDE_coin_difference_l3519_351902

/-- Represents the denominations of coins available --/
inductive Coin
  | Five : Coin
  | Ten : Coin
  | Twenty : Coin

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
  | Coin.Five => 5
  | Coin.Ten => 10
  | Coin.Twenty => 20

/-- The target amount to be paid in cents --/
def targetAmount : Nat := 40

/-- A function that calculates the minimum number of coins needed --/
def minCoins : Nat := sorry

/-- A function that calculates the maximum number of coins needed --/
def maxCoins : Nat := sorry

/-- Theorem stating the difference between max and min number of coins --/
theorem coin_difference : maxCoins - minCoins = 6 := by sorry

end NUMINAMATH_CALUDE_coin_difference_l3519_351902


namespace NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l3519_351916

/-- Calculates the profit per meter of cloth given the total meters sold,
    total selling price, and cost price per meter. -/
def profit_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℚ :=
  (total_selling_price - total_meters * cost_price_per_meter : ℚ) / total_meters

/-- Proves that for a specific cloth sale, the profit per meter is 7 -/
theorem cloth_sale_profit_per_meter :
  profit_per_meter 80 10000 118 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l3519_351916


namespace NUMINAMATH_CALUDE_marble_capacity_l3519_351907

/-- Given a container of volume 24 cm³ holding 75 marbles, 
    prove that a container of volume 72 cm³ will hold 225 marbles, 
    assuming a linear relationship between volume and marble capacity. -/
theorem marble_capacity 
  (volume_small : ℝ) 
  (marbles_small : ℕ) 
  (volume_large : ℝ) 
  (h1 : volume_small = 24) 
  (h2 : marbles_small = 75) 
  (h3 : volume_large = 72) : 
  (volume_large / volume_small) * marbles_small = 225 := by
sorry

end NUMINAMATH_CALUDE_marble_capacity_l3519_351907


namespace NUMINAMATH_CALUDE_divisor_problem_l3519_351967

theorem divisor_problem (d : ℕ) : 
  (∃ q₁ q₂ : ℕ, 100 = q₁ * d + 4 ∧ 90 = q₂ * d + 18) → d = 24 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3519_351967


namespace NUMINAMATH_CALUDE_min_values_l3519_351978

-- Define the logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the condition from the problem
def condition (x y : ℝ) : Prop := lg (3 * x) + lg y = lg (x + y + 1)

-- Theorem statement
theorem min_values {x y : ℝ} (h : condition x y) :
  (∀ a b : ℝ, condition a b → x * y ≤ a * b) ∧
  (∀ c d : ℝ, condition c d → x + y ≤ c + d) :=
by sorry

end NUMINAMATH_CALUDE_min_values_l3519_351978


namespace NUMINAMATH_CALUDE_factorization_proof_l3519_351976

theorem factorization_proof (x : ℝ) : 4*x^3 - 8*x^2 + 4*x = 4*x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3519_351976


namespace NUMINAMATH_CALUDE_function_always_positive_l3519_351972

theorem function_always_positive (x : ℝ) : 
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ 
  (x < 1 ∨ x > 3) := by
sorry

end NUMINAMATH_CALUDE_function_always_positive_l3519_351972


namespace NUMINAMATH_CALUDE_table_height_l3519_351977

/-- Given three rectangular boxes with heights b, r, and g, and a table with height h,
    prove that h = 91 when the following conditions are met:
    1. h + b - g = 111
    2. h + r - b = 80
    3. h + g - r = 82 -/
theorem table_height (h b r g : ℝ) 
    (eq1 : h + b - g = 111)
    (eq2 : h + r - b = 80)
    (eq3 : h + g - r = 82) : h = 91 := by
  sorry

end NUMINAMATH_CALUDE_table_height_l3519_351977


namespace NUMINAMATH_CALUDE_circle_center_problem_l3519_351945

/-- A circle tangent to two parallel lines with its center on a third line --/
theorem circle_center_problem (x y : ℝ) :
  (6 * x - 5 * y = 15) ∧ 
  (3 * x + 2 * y = 0) →
  x = 10 / 3 ∧ y = -5 := by
  sorry

#check circle_center_problem

end NUMINAMATH_CALUDE_circle_center_problem_l3519_351945


namespace NUMINAMATH_CALUDE_impossible_chord_length_l3519_351999

theorem impossible_chord_length (r : ℝ) (chord_length : ℝ) : 
  r = 5 → chord_length = 11 → chord_length > 2 * r := by sorry

end NUMINAMATH_CALUDE_impossible_chord_length_l3519_351999


namespace NUMINAMATH_CALUDE_unity_digit_of_n_l3519_351997

theorem unity_digit_of_n (n : ℕ) (h : 3 * n = 999^1000) : n % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_unity_digit_of_n_l3519_351997


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l3519_351904

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∀ (x y : ℝ), perpendicular_bisector x y ↔
  (∃ (t : ℝ), (1 - t) • A.1 + t • B.1 = x ∧ (1 - t) • A.2 + t • B.2 = y) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l3519_351904


namespace NUMINAMATH_CALUDE_ellipse_equivalence_l3519_351957

/-- Given an ellipse with equation 9x^2 + 4y^2 = 36, prove that the ellipse with equation
    x^2/20 + y^2/25 = 1 has the same foci and a minor axis length of 4√5 -/
theorem ellipse_equivalence (x y : ℝ) : 
  (∃ (a b c : ℝ), 9 * x^2 + 4 * y^2 = 36 ∧ 
   c^2 = a^2 - b^2 ∧
   x^2 / 20 + y^2 / 25 = 1 ∧
   b = 2 * (5 : ℝ).sqrt) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equivalence_l3519_351957


namespace NUMINAMATH_CALUDE_problem_solution_l3519_351983

theorem problem_solution (a b x : ℝ) 
  (h1 : a * (x + 2) + b * (x + 2) = 60) 
  (h2 : a + b = 12) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3519_351983


namespace NUMINAMATH_CALUDE_city_population_ratio_l3519_351986

theorem city_population_ratio (x y z : ℕ) (hxy : ∃ k : ℕ, x = k * y) (hyz : y = 2 * z) (hxz : x = 14 * z) :
  x / y = 7 :=
by sorry

end NUMINAMATH_CALUDE_city_population_ratio_l3519_351986


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l3519_351906

theorem divisibility_by_twelve (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l3519_351906


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3519_351915

theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, q * x^2 - 8 * x + 16 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3519_351915


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l3519_351929

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let r := (d1 * d2) / (4 * a)
  r = (105 * Real.sqrt 274) / 274 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l3519_351929


namespace NUMINAMATH_CALUDE_initial_bushes_count_l3519_351901

/-- The number of orchid bushes planted today -/
def bushes_planted_today : ℕ := 37

/-- The number of orchid bushes planted tomorrow -/
def bushes_planted_tomorrow : ℕ := 25

/-- The total number of orchid bushes after planting -/
def total_bushes_after_planting : ℕ := 109

/-- The number of workers who finished the planting -/
def number_of_workers : ℕ := 35

/-- The initial number of orchid bushes in the park -/
def initial_bushes : ℕ := total_bushes_after_planting - (bushes_planted_today + bushes_planted_tomorrow)

theorem initial_bushes_count : initial_bushes = 47 := by
  sorry

end NUMINAMATH_CALUDE_initial_bushes_count_l3519_351901


namespace NUMINAMATH_CALUDE_car_speed_l3519_351925

/-- Calculates the speed of a car given distance and time -/
theorem car_speed (distance : ℝ) (time : ℝ) (h1 : distance = 624) (h2 : time = 2 + 2/5) :
  distance / time = 260 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3519_351925


namespace NUMINAMATH_CALUDE_no_intersection_l3519_351975

-- Define the two functions
def f (x : ℝ) : ℝ := |2 * x + 5|
def g (x : ℝ) : ℝ := -|3 * x - 2|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l3519_351975


namespace NUMINAMATH_CALUDE_limit_equals_six_l3519_351914

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_equals_six 
  (h : deriv f 2 = 3) : 
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ ≠ 0, |x₀| < δ → 
    |((f (2 + 2*x₀) - f 2) / x₀) - 6| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_equals_six_l3519_351914


namespace NUMINAMATH_CALUDE_five_player_four_stage_tournament_outcomes_l3519_351950

/-- Represents a tournament with a fixed number of players and stages. -/
structure Tournament :=
  (num_players : ℕ)
  (num_stages : ℕ)

/-- Calculates the number of possible outcomes in a tournament. -/
def tournament_outcomes (t : Tournament) : ℕ :=
  2^t.num_stages

/-- Theorem stating that a tournament with 5 players and 4 stages has 16 possible outcomes. -/
theorem five_player_four_stage_tournament_outcomes :
  ∀ t : Tournament, t.num_players = 5 → t.num_stages = 4 →
  tournament_outcomes t = 16 :=
by sorry

end NUMINAMATH_CALUDE_five_player_four_stage_tournament_outcomes_l3519_351950


namespace NUMINAMATH_CALUDE_circle_through_AB_with_center_on_line_l3519_351932

-- Define the points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (1, -3)

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := y = x + 3

-- Define the standard form of a circle
def isCircle (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- State the theorem
theorem circle_through_AB_with_center_on_line :
  ∃ (h k r : ℝ), 
    centerLine h k ∧
    isCircle h k r A.1 A.2 ∧
    isCircle h k r B.1 B.2 ∧
    h = -2 ∧ k = 1 ∧ r = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_through_AB_with_center_on_line_l3519_351932


namespace NUMINAMATH_CALUDE_mixed_groups_count_l3519_351995

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos) :=
by sorry

end NUMINAMATH_CALUDE_mixed_groups_count_l3519_351995


namespace NUMINAMATH_CALUDE_square_less_than_triple_l3519_351965

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l3519_351965


namespace NUMINAMATH_CALUDE_sine_identity_and_not_monotonicity_l3519_351941

theorem sine_identity_and_not_monotonicity : 
  (∀ x : ℝ, Real.sin (π - x) = Real.sin x) ∧ 
  ¬(∀ α β : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 → α > β → Real.sin α > Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_and_not_monotonicity_l3519_351941


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l3519_351973

/-- The number of rectangles in a 4x4 grid -/
def num_rectangles_4x4 : ℕ := 36

/-- The number of horizontal or vertical lines in a 4x4 grid -/
def grid_lines : ℕ := 4

/-- Theorem: The number of rectangles in a 4x4 grid is 36 -/
theorem rectangles_in_4x4_grid :
  (grid_lines.choose 2) * (grid_lines.choose 2) = num_rectangles_4x4 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l3519_351973


namespace NUMINAMATH_CALUDE_total_bread_slices_l3519_351960

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of bread slices needed for each sandwich -/
def slices_per_sandwich : ℕ := 3

/-- Theorem: The total number of bread slices needed for Ryan's sandwiches is 15 -/
theorem total_bread_slices : num_sandwiches * slices_per_sandwich = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_slices_l3519_351960


namespace NUMINAMATH_CALUDE_waiter_customers_l3519_351990

/-- The initial number of customers before 5 more arrived -/
def initial_customers : ℕ := 3

/-- The number of additional customers that arrived -/
def additional_customers : ℕ := 5

/-- The total number of customers after the additional customers arrived -/
def total_customers : ℕ := 8

theorem waiter_customers : 
  initial_customers + additional_customers = total_customers := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3519_351990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3519_351956

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3519_351956


namespace NUMINAMATH_CALUDE_students_taking_neither_subject_l3519_351908

-- Define the total number of students in the drama club
def total_students : ℕ := 60

-- Define the number of students taking mathematics
def math_students : ℕ := 40

-- Define the number of students taking physics
def physics_students : ℕ := 35

-- Define the number of students taking both mathematics and physics
def both_subjects : ℕ := 25

-- Theorem to prove
theorem students_taking_neither_subject : 
  total_students - (math_students + physics_students - both_subjects) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_subject_l3519_351908


namespace NUMINAMATH_CALUDE_garrison_problem_l3519_351974

/-- Represents the number of men in a garrison and their provisions --/
structure Garrison where
  initialMen : ℕ
  initialDays : ℕ
  reinforcementMen : ℕ
  remainingDays : ℕ
  reinforcementArrivalDay : ℕ

/-- Calculates the initial number of men in the garrison --/
def calculateInitialMen (g : Garrison) : ℕ :=
  (g.initialDays - g.reinforcementArrivalDay) * g.initialMen / 
  (g.initialDays - g.reinforcementArrivalDay - g.remainingDays)

/-- Theorem stating that given the conditions, the initial number of men is 2000 --/
theorem garrison_problem (g : Garrison) 
  (h1 : g.initialDays = 65)
  (h2 : g.reinforcementMen = 3000)
  (h3 : g.remainingDays = 20)
  (h4 : g.reinforcementArrivalDay = 15) :
  calculateInitialMen g = 2000 := by
  sorry

#eval calculateInitialMen { initialMen := 2000, initialDays := 65, reinforcementMen := 3000, remainingDays := 20, reinforcementArrivalDay := 15 }

end NUMINAMATH_CALUDE_garrison_problem_l3519_351974


namespace NUMINAMATH_CALUDE_range_of_m_for_increasing_function_l3519_351969

-- Define an increasing function on an open interval
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → f x < f y

-- State the theorem
theorem range_of_m_for_increasing_function 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : IncreasingFunction f) 
  (h_inequality : f (m - 1) < f (1 - 2*m)) :
  m ∈ Set.Ioo (-1/2) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_increasing_function_l3519_351969


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3519_351996

/-- The number of apples handed out to students -/
def apples_handed_out : ℕ := 5

/-- The number of pies made -/
def pies_made : ℕ := 9

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 5

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := apples_handed_out + pies_made * apples_per_pie

theorem cafeteria_apples : initial_apples = 50 := by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3519_351996


namespace NUMINAMATH_CALUDE_best_fit_highest_r_squared_model_1_best_fit_l3519_351927

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fit among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

/-- The model with the highest R² value has the best fit -/
theorem best_fit_highest_r_squared (models : List RegressionModel) (model : RegressionModel) 
    (h : model ∈ models) :
    has_best_fit model models ↔ ∀ m ∈ models, model.r_squared ≥ m.r_squared :=
  sorry

/-- Given four specific models, prove that Model ① has the best fit -/
theorem model_1_best_fit :
  let models : List RegressionModel := [
    ⟨"①", 0.976⟩,
    ⟨"②", 0.776⟩,
    ⟨"③", 0.076⟩,
    ⟨"④", 0.351⟩
  ]
  let model_1 : RegressionModel := ⟨"①", 0.976⟩
  has_best_fit model_1 models :=
  sorry

end NUMINAMATH_CALUDE_best_fit_highest_r_squared_model_1_best_fit_l3519_351927


namespace NUMINAMATH_CALUDE_number_division_l3519_351917

theorem number_division (x : ℤ) : (x - 39 = 54) → (x / 3 = 31) := by
  sorry

end NUMINAMATH_CALUDE_number_division_l3519_351917


namespace NUMINAMATH_CALUDE_three_horse_race_odds_l3519_351910

/-- Represents the odds against a horse winning as a pair of natural numbers -/
def Odds := ℕ × ℕ

/-- Calculates the probability of winning from given odds -/
def probability (odds : Odds) : ℚ :=
  (odds.2 : ℚ) / (odds.1 + odds.2)

theorem three_horse_race_odds
  (odds_X : Odds)
  (odds_Y : Odds)
  (h_X : odds_X = (5, 3))
  (h_Y : odds_Y = (8, 3))
  (h_no_ties : probability odds_X + probability odds_Y < 1) :
  ∃ (odds_Z : Odds), odds_Z = (57, 31) ∧
    probability odds_X + probability odds_Y + probability odds_Z = 1 :=
sorry

end NUMINAMATH_CALUDE_three_horse_race_odds_l3519_351910


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3519_351930

theorem sum_of_coefficients (a : ℝ) : 
  ((1 + a)^5 = -1) → (a = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3519_351930


namespace NUMINAMATH_CALUDE_count_negative_numbers_l3519_351982

def number_list : List ℚ := [-14, 7, 0, -2/3, -5/16]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l3519_351982


namespace NUMINAMATH_CALUDE_angle_measure_120_l3519_351984

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hsum : A + B + C = π)

-- State the theorem
theorem angle_measure_120 (t : Triangle) (h : t.a^2 = t.b^2 + t.b*t.c + t.c^2) :
  t.A = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_angle_measure_120_l3519_351984


namespace NUMINAMATH_CALUDE_savings_after_twelve_months_l3519_351989

def savings_sequence (n : ℕ) : ℕ := 2 ^ n

theorem savings_after_twelve_months :
  savings_sequence 12 = 4096 := by sorry

end NUMINAMATH_CALUDE_savings_after_twelve_months_l3519_351989


namespace NUMINAMATH_CALUDE_equation_solution_l3519_351953

theorem equation_solution :
  ∀ y : ℝ, (3 + 1.5 * y^2 = 0.5 * y^2 + 16) ↔ (y = Real.sqrt 13 ∨ y = -Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3519_351953


namespace NUMINAMATH_CALUDE_delta_flight_price_l3519_351922

theorem delta_flight_price (delta_discount : Real) (united_price : Real) (united_discount : Real) (price_difference : Real) :
  delta_discount = 0.20 →
  united_price = 1100 →
  united_discount = 0.30 →
  price_difference = 90 →
  ∃ (original_delta_price : Real),
    original_delta_price * (1 - delta_discount) = 
    united_price * (1 - united_discount) - price_difference ∧
    original_delta_price = 850 := by
  sorry

end NUMINAMATH_CALUDE_delta_flight_price_l3519_351922


namespace NUMINAMATH_CALUDE_ellipse_circle_area_relation_sum_of_x_values_l3519_351948

theorem ellipse_circle_area_relation (x : ℝ) : 
  let circle_radius := x - 2
  let ellipse_semi_major := x - 3
  let ellipse_semi_minor := x + 4
  π * ellipse_semi_major * ellipse_semi_minor = 2 * π * circle_radius^2 →
  x = 4 ∨ x = 5 :=
by
  sorry

theorem sum_of_x_values : 
  ∃ (x₁ x₂ : ℝ), 
    (let circle_radius := x₁ - 2
     let ellipse_semi_major := x₁ - 3
     let ellipse_semi_minor := x₁ + 4
     π * ellipse_semi_major * ellipse_semi_minor = 2 * π * circle_radius^2) ∧
    (let circle_radius := x₂ - 2
     let ellipse_semi_major := x₂ - 3
     let ellipse_semi_minor := x₂ + 4
     π * ellipse_semi_major * ellipse_semi_minor = 2 * π * circle_radius^2) ∧
    x₁ + x₂ = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_circle_area_relation_sum_of_x_values_l3519_351948


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l3519_351961

theorem fourth_root_equation_solution :
  ∃ x : ℝ, (x^(1/4) * (x^5)^(1/8) = 4) ∧ (x = 4^(8/7)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l3519_351961


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3519_351918

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 11) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3519_351918


namespace NUMINAMATH_CALUDE_election_vote_count_l3519_351938

theorem election_vote_count 
  (total_votes : ℕ) 
  (candidate_a_votes : ℕ) 
  (candidate_b_votes : ℕ) :
  (candidate_a_votes = candidate_b_votes + (15 * total_votes) / 100) →
  (candidate_b_votes = 3159) →
  ((80 * total_votes) / 100 = candidate_a_votes + candidate_b_votes) →
  (total_votes = 9720) :=
by sorry

end NUMINAMATH_CALUDE_election_vote_count_l3519_351938


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l3519_351911

theorem ordering_of_expressions : 3^(1/5) > 0.2^3 ∧ 0.2^3 > Real.log 0.1 / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l3519_351911


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3519_351979

/-- Given a cone with base radius 1 and lateral surface that unfolds to a 
    sector with a 90° central angle, its lateral surface area is 4π. -/
theorem cone_lateral_surface_area (r : Real) (θ : Real) : 
  r = 1 → θ = 90 → ∃ (l : Real), l * θ / 360 * (2 * Real.pi) = 2 * Real.pi ∧ 
    r * l * Real.pi = 4 * Real.pi := by
  sorry

#check cone_lateral_surface_area

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3519_351979


namespace NUMINAMATH_CALUDE_power_equation_solution_l3519_351926

theorem power_equation_solution (n b : ℝ) : n = 2^(1/4) → n^b = 16 → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3519_351926
