import Mathlib

namespace quadratic_roots_problem_l495_49548

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x : ℝ, x^2 - (2*m - 1)*x + m^2 = 0) →
  (x₁^2 - (2*m - 1)*x₁ + m^2 = 0) →
  (x₂^2 - (2*m - 1)*x₂ + m^2 = 0) →
  (x₁ ≠ x₂) →
  ((x₁ + 1) * (x₂ + 1) = 3) →
  m = -3 := by
sorry

end quadratic_roots_problem_l495_49548


namespace pie_division_l495_49501

theorem pie_division (initial_pie : ℚ) (scrooge_share : ℚ) (num_friends : ℕ) : 
  initial_pie = 4/5 → scrooge_share = 1/5 → num_friends = 3 →
  (initial_pie - scrooge_share * initial_pie) / num_friends = 1/5 := by
  sorry

end pie_division_l495_49501


namespace minutes_in_three_and_half_hours_l495_49583

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours -/
def hours : ℚ := 3.5

/-- Theorem: The number of minutes in 3.5 hours is 210 -/
theorem minutes_in_three_and_half_hours : 
  (hours * minutes_per_hour : ℚ) = 210 := by sorry

end minutes_in_three_and_half_hours_l495_49583


namespace eight_bead_bracelet_arrangements_l495_49529

/-- The number of unique arrangements of n distinct beads on a bracelet, 
    considering rotational and reflectional symmetry -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of unique arrangements of 8 distinct beads 
    on a bracelet, considering rotational and reflectional symmetry, is 2520 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 2520 := by
  sorry

end eight_bead_bracelet_arrangements_l495_49529


namespace cubic_expansion_simplification_l495_49596

theorem cubic_expansion_simplification :
  (30 + 5)^3 - (30^3 + 3*30^2*5 + 3*30*5^2 + 5^3 - 5^3) = 125 := by
  sorry

end cubic_expansion_simplification_l495_49596


namespace streetlight_problem_l495_49585

/-- The number of streetlights --/
def n : ℕ := 2020

/-- The number of lights to be turned off --/
def k : ℕ := 300

/-- The number of ways to select k non-adjacent positions from n-2 positions --/
def non_adjacent_selections (n k : ℕ) : ℕ := Nat.choose (n - k - 1) k

theorem streetlight_problem :
  non_adjacent_selections n k = Nat.choose 1710 300 :=
sorry

end streetlight_problem_l495_49585


namespace balloon_comparison_l495_49582

/-- The number of balloons Allan initially brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The number of additional balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℕ := jake_balloons - allan_total

theorem balloon_comparison : balloon_difference = 1 := by
  sorry

end balloon_comparison_l495_49582


namespace cost_of_fifty_roses_l495_49595

/-- Represents the cost of a bouquet of roses -/
def bouquetCost (roses : ℕ) : ℚ :=
  if roses ≤ 30 then
    (30 : ℚ) / 15 * roses
  else
    (30 : ℚ) / 15 * 30 + (30 : ℚ) / 15 / 2 * (roses - 30)

/-- The theorem stating the cost of a bouquet with 50 roses -/
theorem cost_of_fifty_roses : bouquetCost 50 = 80 := by
  sorry

end cost_of_fifty_roses_l495_49595


namespace knowledge_competition_probabilities_l495_49590

/-- Represents the probability of a team member answering correctly -/
structure TeamMember where
  prob_correct : ℝ
  prob_correct_nonneg : 0 ≤ prob_correct
  prob_correct_le_one : prob_correct ≤ 1

/-- Represents a team in the knowledge competition -/
structure Team where
  member_a : TeamMember
  member_b : TeamMember

/-- The total score of a team in the competition -/
inductive TotalScore
  | zero
  | ten
  | twenty
  | thirty

def prob_first_correct (team : Team) : ℝ :=
  team.member_a.prob_correct + (1 - team.member_a.prob_correct) * team.member_b.prob_correct

def prob_distribution (team : Team) : TotalScore → ℝ
  | TotalScore.zero => (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.ten => prob_first_correct team * (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.twenty => (prob_first_correct team)^2 * (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.thirty => (prob_first_correct team)^3

theorem knowledge_competition_probabilities (team : Team)
  (h_a : team.member_a.prob_correct = 2/5)
  (h_b : team.member_b.prob_correct = 2/3) :
  prob_first_correct team = 4/5 ∧
  prob_distribution team TotalScore.zero = 1/5 ∧
  prob_distribution team TotalScore.ten = 4/25 ∧
  prob_distribution team TotalScore.twenty = 16/125 ∧
  prob_distribution team TotalScore.thirty = 64/125 := by
  sorry

#check knowledge_competition_probabilities

end knowledge_competition_probabilities_l495_49590


namespace condition_satisfies_equation_l495_49569

theorem condition_satisfies_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end condition_satisfies_equation_l495_49569


namespace prob_five_heads_five_tails_l495_49573

/-- Represents the state of the coin after some number of flips. -/
structure CoinState where
  heads : ℕ
  tails : ℕ

/-- The probability of getting heads given the current state of the coin. -/
def prob_heads (state : CoinState) : ℚ :=
  (state.heads + 1) / (state.heads + state.tails + 2)

/-- The probability of a specific sequence of 10 flips resulting in exactly 5 heads and 5 tails. -/
def prob_sequence : ℚ := 1 / 39916800

/-- The number of ways to arrange 5 heads and 5 tails in 10 flips. -/
def num_sequences : ℕ := 252

/-- The theorem stating the probability of getting exactly 5 heads and 5 tails after 10 flips. -/
theorem prob_five_heads_five_tails :
  num_sequences * prob_sequence = 1 / 158760 := by sorry

end prob_five_heads_five_tails_l495_49573


namespace find_multiple_l495_49513

theorem find_multiple (n m : ℝ) (h1 : n + n + m * n + 4 * n = 104) (h2 : n = 13) : m = 2 := by
  sorry

end find_multiple_l495_49513


namespace parallel_vectors_m_value_l495_49553

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a = (1, m) and b = (3, 1), prove that if they are parallel, then m = 1/3 -/
theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (1, m) (3, 1) → m = 1/3 := by
  sorry

end parallel_vectors_m_value_l495_49553


namespace quadratic_equation_with_given_roots_l495_49591

theorem quadratic_equation_with_given_roots (x y : ℝ) 
  (h : x^2 - 6*x + 9 = -|y - 1|) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
    (∀ z : ℝ, a*z^2 + b*z + c = 0 ↔ z = x ∨ z = y) ∧
    a = 1 ∧ b = -4 ∧ c = -3 :=
by sorry

end quadratic_equation_with_given_roots_l495_49591


namespace grace_weekly_charge_l495_49557

/-- Grace's weekly charge given her total earnings and work duration -/
def weekly_charge (total_earnings : ℚ) (weeks : ℕ) : ℚ :=
  total_earnings / weeks

/-- Theorem: Grace's weekly charge is $300 -/
theorem grace_weekly_charge :
  let total_earnings : ℚ := 1800
  let weeks : ℕ := 6
  weekly_charge total_earnings weeks = 300 := by
  sorry

end grace_weekly_charge_l495_49557


namespace overtime_hours_calculation_l495_49554

/-- Calculates the number of overtime hours worked given the regular pay rate,
    regular hours limit, and total pay received. -/
def overtime_hours (regular_rate : ℚ) (regular_hours_limit : ℕ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours_limit
  let overtime_rate := 2 * regular_rate
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Proves that given the specified conditions, the number of overtime hours is 12. -/
theorem overtime_hours_calculation :
  let regular_rate : ℚ := 3
  let regular_hours_limit : ℕ := 40
  let total_pay : ℚ := 192
  overtime_hours regular_rate regular_hours_limit total_pay = 12 := by
  sorry

end overtime_hours_calculation_l495_49554


namespace power_equality_implies_x_equals_two_l495_49511

theorem power_equality_implies_x_equals_two :
  ∀ x : ℝ, (2 : ℝ)^10 = 32^x → x = 2 := by
  sorry

end power_equality_implies_x_equals_two_l495_49511


namespace bus_ride_difference_l495_49540

def tess_to_noah : ℝ := 0.75
def tess_noah_to_kayla : ℝ := 0.85
def tess_kayla_to_school : ℝ := 1.15

def oscar_to_charlie : ℝ := 0.25
def oscar_charlie_to_school : ℝ := 1.35

theorem bus_ride_difference : 
  (tess_to_noah + tess_noah_to_kayla + tess_kayla_to_school) - 
  (oscar_to_charlie + oscar_charlie_to_school) = 1.15 := by
  sorry

end bus_ride_difference_l495_49540


namespace sum_of_dot_products_l495_49502

/-- Given three points A, B, C on a plane, prove that the sum of their vector dot products is -25 -/
theorem sum_of_dot_products (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CA := (A.1 - C.1, A.2 - C.2)
  (AB.1^2 + AB.2^2 = 3^2) →
  (BC.1^2 + BC.2^2 = 4^2) →
  (CA.1^2 + CA.2^2 = 5^2) →
  (AB.1 * BC.1 + AB.2 * BC.2) + (BC.1 * CA.1 + BC.2 * CA.2) + (CA.1 * AB.1 + CA.2 * AB.2) = -25 :=
by
  sorry


end sum_of_dot_products_l495_49502


namespace mask_distribution_arrangements_l495_49551

/-- The number of ways to distribute n distinct objects among k distinct people,
    where each person must receive at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := sorry

theorem mask_distribution_arrangements :
  (distribute 7 4) * (permutations 4) = 8400 := by sorry

end mask_distribution_arrangements_l495_49551


namespace unknown_number_problem_l495_49508

theorem unknown_number_problem : ∃ x : ℝ, 0.5 * 56 = 0.3 * x + 13 ∧ x = 50 := by sorry

end unknown_number_problem_l495_49508


namespace square_remainder_l495_49579

theorem square_remainder (n : ℤ) : n % 5 = 3 → n^2 % 5 = 4 := by
  sorry

end square_remainder_l495_49579


namespace melanie_gave_27_apples_l495_49572

/-- The number of apples Joan picked from the orchard -/
def apples_picked : ℕ := 43

/-- The total number of apples Joan has now -/
def total_apples : ℕ := 70

/-- The number of apples Melanie gave to Joan -/
def apples_from_melanie : ℕ := total_apples - apples_picked

theorem melanie_gave_27_apples : apples_from_melanie = 27 := by
  sorry

end melanie_gave_27_apples_l495_49572


namespace expression_evaluation_l495_49531

theorem expression_evaluation :
  let a : ℤ := -1
  (2 - a)^2 - (1 + a)*(a - 1) - a*(a - 3) = 5 :=
by sorry

end expression_evaluation_l495_49531


namespace perpendicular_vectors_sum_magnitude_l495_49597

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem perpendicular_vectors_sum_magnitude (x : ℝ) :
  let a := vector_a x
  let b := vector_b
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- perpendicular condition
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 10 :=
by sorry

end perpendicular_vectors_sum_magnitude_l495_49597


namespace reciprocal_sum_one_triples_l495_49533

def reciprocal_sum_one (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1

def valid_triples : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(2, 3, 6), (2, 6, 3), (3, 2, 6), (3, 6, 2), (6, 2, 3), (6, 3, 2),
   (2, 4, 4), (4, 2, 4), (4, 4, 2), (3, 3, 3)}

theorem reciprocal_sum_one_triples :
  ∀ (a b c : ℕ+), reciprocal_sum_one a b c ↔ (a, b, c) ∈ valid_triples := by
  sorry

end reciprocal_sum_one_triples_l495_49533


namespace inequality_proof_l495_49567

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l495_49567


namespace range_of_m_for_quadratic_inequality_l495_49544

theorem range_of_m_for_quadratic_inequality (m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x > m) ∧ 
  (∃ x : ℝ, x > m ∧ x^2 - 3*x + 2 ≥ 0) → 
  m ≤ 1 :=
by sorry

end range_of_m_for_quadratic_inequality_l495_49544


namespace eleven_days_sufficiency_l495_49518

/-- Represents the amount of cat food in a package -/
structure CatFood where
  days : ℝ
  nonneg : days ≥ 0

/-- The amount of food in a large package -/
def large_package : CatFood := sorry

/-- The amount of food in a small package -/
def small_package : CatFood := sorry

/-- One large package and four small packages last for 14 days -/
axiom package_combination : large_package.days + 4 * small_package.days = 14

theorem eleven_days_sufficiency :
  large_package.days + 3 * small_package.days ≥ 11 := by
  sorry

end eleven_days_sufficiency_l495_49518


namespace rectangle_ratio_l495_49517

/-- A configuration of squares and rectangles -/
structure SquareRectConfig where
  /-- Side length of the inner square -/
  s : ℝ
  /-- Shorter side of each rectangle -/
  y : ℝ
  /-- Longer side of each rectangle -/
  x : ℝ
  /-- The shorter side of each rectangle is half the side of the inner square -/
  short_side_half : y = s / 2
  /-- The area of the outer square is 9 times that of the inner square -/
  area_ratio : (s + 2 * y)^2 = 9 * s^2
  /-- The longer side of the rectangle forms the side of the outer square with the inner square -/
  outer_square_side : x + s / 2 = 3 * s

/-- The ratio of the longer side to the shorter side of each rectangle is 5 -/
theorem rectangle_ratio (config : SquareRectConfig) : config.x / config.y = 5 := by
  sorry

end rectangle_ratio_l495_49517


namespace train_length_calculation_train2_length_l495_49581

/-- Calculates the length of a train given the conditions of two trains passing each other. -/
theorem train_length_calculation (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := speed_train1 * 1000 / 3600 + speed_train2 * 1000 / 3600
  let total_distance := relative_speed * time_to_cross
  total_distance - length_train1

/-- The length of Train 2 is approximately 269.95 meters. -/
theorem train2_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_length_calculation 230 120 80 9 - 269.95| < ε :=
sorry

end train_length_calculation_train2_length_l495_49581


namespace same_remainder_problem_l495_49559

theorem same_remainder_problem (x : ℕ+) : 
  (∃ q r : ℕ, 100 = q * x + r ∧ r < x) ∧ 
  (∃ p r : ℕ, 197 = p * x + r ∧ r < x) → 
  (∃ r : ℕ, 100 % x = r ∧ 197 % x = r ∧ r = 3) := by
sorry

end same_remainder_problem_l495_49559


namespace distribute_seven_balls_three_boxes_l495_49560

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_seven_balls_three_boxes : distribute_balls 7 3 = 365 := by sorry

end distribute_seven_balls_three_boxes_l495_49560


namespace stationery_store_profit_l495_49507

/-- Profit data for a week at a stationery store -/
structure WeekProfit :=
  (mon tue wed thu fri sat sun : ℝ)
  (total : ℝ)
  (sum_condition : mon + tue + wed + thu + fri + sat + sun = total)

/-- Theorem stating the properties of the profit data -/
theorem stationery_store_profit 
  (w : WeekProfit)
  (h1 : w.mon = -27.8)
  (h2 : w.tue = -70.3)
  (h3 : w.wed = 200)
  (h4 : w.thu = 138.1)
  (h5 : w.sun = 188)
  (h6 : w.total = 458) :
  (w.fri = -8 → w.sat = 38) ∧
  (w.sat = w.fri + 10 → w.sat = 20) ∧
  (w.fri < 0 → w.sat > 0 → w.sat > 30) :=
by sorry

end stationery_store_profit_l495_49507


namespace greatest_three_digit_divisible_by_3_6_5_l495_49578

theorem greatest_three_digit_divisible_by_3_6_5 :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
  n % 3 = 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧
  n = 990 ∧
  ∀ (m : ℕ), 100 ≤ m ∧ m ≤ 999 ∧ 
  m % 3 = 0 ∧ m % 6 = 0 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end greatest_three_digit_divisible_by_3_6_5_l495_49578


namespace project_completion_time_l495_49528

theorem project_completion_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 60) 
  (h2 : days_initial = 3) 
  (h3 : workers_new = 30) :
  workers_initial * days_initial = workers_new * (2 * days_initial) :=
by sorry

end project_completion_time_l495_49528


namespace polynomial_remainder_l495_49523

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 4*x^2 + 7) % (x - 1) = 4 := by
sorry

end polynomial_remainder_l495_49523


namespace ice_cream_flavors_count_l495_49588

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors created by combining 5 scoops from 3 basic flavors -/
def ice_cream_flavors : ℕ := distribute 5 3

/-- Theorem: The number of ice cream flavors is 21 -/
theorem ice_cream_flavors_count : ice_cream_flavors = 21 := by
  sorry

end ice_cream_flavors_count_l495_49588


namespace prob_three_primes_in_six_dice_l495_49598

-- Define a 12-sided die
def twelve_sided_die : Finset ℕ := Finset.range 12

-- Define prime numbers on a 12-sided die
def primes_on_die : Finset ℕ := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime on a single die
def prob_prime : ℚ := (primes_on_die.card : ℚ) / (twelve_sided_die.card : ℚ)

-- Define the probability of rolling a non-prime on a single die
def prob_non_prime : ℚ := 1 - prob_prime

-- Define the number of dice
def num_dice : ℕ := 6

-- Define the number of dice showing prime
def num_prime_dice : ℕ := 3

-- Theorem statement
theorem prob_three_primes_in_six_dice : 
  (Nat.choose num_dice num_prime_dice : ℚ) * 
  (prob_prime ^ num_prime_dice) * 
  (prob_non_prime ^ (num_dice - num_prime_dice)) = 857500 / 2985984 := by
  sorry

end prob_three_primes_in_six_dice_l495_49598


namespace divide_ten_with_difference_five_l495_49580

theorem divide_ten_with_difference_five :
  ∀ x y : ℝ, x + y = 10 ∧ y - x = 5 → x = (5 : ℝ) / 2 ∧ y = (15 : ℝ) / 2 := by
  sorry

end divide_ten_with_difference_five_l495_49580


namespace number_difference_l495_49535

theorem number_difference (x y : ℤ) (h1 : x > y) (h2 : x + y = 64) (h3 : y = 26) : x - y = 12 := by
  sorry

end number_difference_l495_49535


namespace cyclic_sum_factorization_l495_49565

theorem cyclic_sum_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + b*c + c*a) :=
by sorry

end cyclic_sum_factorization_l495_49565


namespace all_fruits_fall_on_day_14_l495_49561

/-- The number of fruits on the tree initially -/
def initial_fruits : ℕ := 60

/-- The number of fruits that fall on day n according to the original pattern -/
def fruits_falling (n : ℕ) : ℕ := n

/-- The sum of fruits that have fallen up to day n according to the original pattern -/
def sum_fallen (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of fruits remaining on the tree after n days of the original pattern -/
def fruits_remaining (n : ℕ) : ℕ := max 0 (initial_fruits - sum_fallen n)

/-- The day when the original pattern stops -/
def pattern_stop_day : ℕ := 10

/-- The number of days needed to finish the remaining fruits after the original pattern stops -/
def additional_days : ℕ := fruits_remaining pattern_stop_day

/-- The total number of days needed for all fruits to fall -/
def total_days : ℕ := pattern_stop_day + additional_days - 1

theorem all_fruits_fall_on_day_14 : total_days = 14 := by
  sorry

end all_fruits_fall_on_day_14_l495_49561


namespace hanks_fruit_purchase_cost_hanks_fruit_purchase_cost_is_1704_l495_49521

/-- Calculates the total cost of Hank's fruit purchase at Clark's Food Store --/
theorem hanks_fruit_purchase_cost : ℝ :=
  let apple_price_per_dozen : ℝ := 40
  let pear_price_per_dozen : ℝ := 50
  let orange_price_per_dozen : ℝ := 30
  let apple_dozens_bought : ℝ := 14
  let pear_dozens_bought : ℝ := 18
  let orange_dozens_bought : ℝ := 10
  let apple_discount_rate : ℝ := 0.1

  let apple_cost : ℝ := apple_price_per_dozen * apple_dozens_bought
  let discounted_apple_cost : ℝ := apple_cost * (1 - apple_discount_rate)
  let pear_cost : ℝ := pear_price_per_dozen * pear_dozens_bought
  let orange_cost : ℝ := orange_price_per_dozen * orange_dozens_bought

  let total_cost : ℝ := discounted_apple_cost + pear_cost + orange_cost

  1704

/-- Proves that Hank's total fruit purchase cost is 1704 dollars --/
theorem hanks_fruit_purchase_cost_is_1704 : hanks_fruit_purchase_cost = 1704 := by
  sorry

end hanks_fruit_purchase_cost_hanks_fruit_purchase_cost_is_1704_l495_49521


namespace system_solution_l495_49562

theorem system_solution (x y z : ℝ) 
  (eq1 : x^2 + 27 = -8*y + 10*z)
  (eq2 : y^2 + 196 = 18*z + 13*x)
  (eq3 : z^2 + 119 = -3*x + 30*y) :
  x + 3*y + 5*z = 127.5 := by
sorry

end system_solution_l495_49562


namespace emmanuel_regular_plan_cost_l495_49519

/-- Calculates the regular plan cost given the stay duration, international data cost per day, and total charges. -/
def regular_plan_cost (stay_duration : ℕ) (intl_data_cost_per_day : ℚ) (total_charges : ℚ) : ℚ :=
  total_charges - (stay_duration : ℚ) * intl_data_cost_per_day

/-- Proves that Emmanuel's regular plan cost is $175 given the problem conditions. -/
theorem emmanuel_regular_plan_cost :
  regular_plan_cost 10 (350/100) 210 = 175 := by
  sorry

end emmanuel_regular_plan_cost_l495_49519


namespace robot_competition_max_weight_l495_49584

/-- The weight of the standard robot in pounds -/
def standard_robot_weight : ℝ := 100

/-- The minimum additional weight above the standard robot weight -/
def min_additional_weight : ℝ := 5

/-- The minimum weight of a robot in the competition -/
def min_robot_weight : ℝ := standard_robot_weight + min_additional_weight

/-- The maximum weight multiplier relative to the minimum weight -/
def max_weight_multiplier : ℝ := 2

/-- The maximum weight of a robot in the competition -/
def max_robot_weight : ℝ := max_weight_multiplier * min_robot_weight

theorem robot_competition_max_weight :
  max_robot_weight = 210 := by sorry

end robot_competition_max_weight_l495_49584


namespace g_of_2_l495_49516

def g (x : ℝ) : ℝ := x^2 + 3*x - 1

theorem g_of_2 : g 2 = 9 := by sorry

end g_of_2_l495_49516


namespace second_sample_not_23_l495_49587

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total : ℕ  -- Total number of items
  sample_size : ℕ  -- Number of items to be sampled
  first_sample : ℕ  -- The first sampled item

/-- The second sample in a systematic sampling scheme -/
def second_sample (s : SystematicSampling) : ℕ :=
  s.first_sample + (s.total / s.sample_size)

/-- Theorem: The second sample cannot be 23 in the given systematic sampling scheme -/
theorem second_sample_not_23 (s : SystematicSampling) 
  (h1 : s.total > 0)
  (h2 : s.sample_size > 0)
  (h3 : s.sample_size ≤ s.total)
  (h4 : s.first_sample ≤ 10)
  (h5 : s.first_sample > 0)
  (h6 : s.sample_size = s.total / 10) :
  second_sample s ≠ 23 := by
  sorry

end second_sample_not_23_l495_49587


namespace dave_race_walking_time_l495_49570

theorem dave_race_walking_time 
  (total_time : ℕ) 
  (jogging_ratio : ℕ) 
  (walking_ratio : ℕ) 
  (h1 : total_time = 21)
  (h2 : jogging_ratio = 4)
  (h3 : walking_ratio = 3) :
  (walking_ratio * total_time) / (jogging_ratio + walking_ratio) = 9 := by
sorry


end dave_race_walking_time_l495_49570


namespace rectangle_perimeter_l495_49514

theorem rectangle_perimeter (square_perimeter : ℝ) (h : square_perimeter = 100) :
  let square_side := square_perimeter / 4
  let rectangle_length := square_side
  let rectangle_width := square_side / 2
  2 * (rectangle_length + rectangle_width) = 75 := by
sorry

end rectangle_perimeter_l495_49514


namespace opposite_of_negative_two_cubed_l495_49534

theorem opposite_of_negative_two_cubed : -((-2)^3) = 8 := by
  sorry

end opposite_of_negative_two_cubed_l495_49534


namespace largest_vertex_sum_l495_49563

/-- Represents a parabola passing through specific points -/
structure Parabola (P : ℤ) where
  a : ℤ
  b : ℤ
  c : ℤ
  pass_origin : a * 0 * 0 + b * 0 + c = 0
  pass_3P : a * (3 * P) * (3 * P) + b * (3 * P) + c = 0
  pass_3P_minus_1 : a * (3 * P - 1) * (3 * P - 1) + b * (3 * P - 1) + c = 45

/-- Calculates the sum of coordinates of the vertex of a parabola -/
def vertexSum (P : ℤ) (p : Parabola P) : ℚ :=
  3 * P / 2 - (p.a : ℚ) * (9 * P^2 : ℚ) / 4

/-- Theorem stating the largest possible vertex sum -/
theorem largest_vertex_sum :
  ∀ P : ℤ, P ≠ 0 → ∀ p : Parabola P, vertexSum P p ≤ 138 := by sorry

end largest_vertex_sum_l495_49563


namespace min_value_sqrt_plus_reciprocal_l495_49574

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 1 / x ≥ 4 ∧ ∃ y > 0, 3 * Real.sqrt y + 1 / y = 4 :=
by sorry

end min_value_sqrt_plus_reciprocal_l495_49574


namespace expression_bounds_l495_49556

theorem expression_bounds (p q r s : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2) (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (p^2 + (2-q)^2) + Real.sqrt (q^2 + (2-r)^2) + 
    Real.sqrt (r^2 + (2-s)^2) + Real.sqrt (s^2 + (2-p)^2) ∧
  Real.sqrt (p^2 + (2-q)^2) + Real.sqrt (q^2 + (2-r)^2) + 
    Real.sqrt (r^2 + (2-s)^2) + Real.sqrt (s^2 + (2-p)^2) ≤ 8 ∧
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 2 ∧
    4 * Real.sqrt (t^2 + (2-t)^2) = 4 * Real.sqrt 2 ∨
    4 * Real.sqrt (t^2 + (2-t)^2) = 8 :=
by sorry


end expression_bounds_l495_49556


namespace sum_of_powers_of_i_is_zero_l495_49541

theorem sum_of_powers_of_i_is_zero : Complex.I + Complex.I^2 + Complex.I^3 + Complex.I^4 = 0 := by
  sorry

end sum_of_powers_of_i_is_zero_l495_49541


namespace all_propositions_false_l495_49589

-- Define the basic types
variable (Line Plane : Type)

-- Define the parallel relation
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)

-- Define the "passes through" relation for planes and lines
variable (passes_through : Plane → Line → Prop)

-- Define the "within" relation for lines and planes
variable (within : Line → Plane → Prop)

-- Define the "has common points" relation
variable (has_common_points : Line → Line → Prop)

-- Define a proposition for "countless lines within a plane"
variable (countless_parallel_lines : Line → Plane → Prop)

-- State the theorem
theorem all_propositions_false :
  -- Proposition 1
  (∀ l1 l2 : Line, ∀ p : Plane, 
    parallel l1 l2 → passes_through p l2 → parallelLP l1 p) ∧
  -- Proposition 2
  (∀ l : Line, ∀ p : Plane,
    parallelLP l p → 
    (∀ l2 : Line, within l2 p → ¬(has_common_points l l2)) ∧
    (∀ l2 : Line, within l2 p → parallel l l2)) ∧
  -- Proposition 3
  (∀ l : Line, ∀ p : Plane,
    ¬(parallelLP l p) → ∀ l2 : Line, within l2 p → ¬(parallel l l2)) ∧
  -- Proposition 4
  (∀ l : Line, ∀ p : Plane,
    countless_parallel_lines l p → parallelLP l p)
  → False := by sorry

end all_propositions_false_l495_49589


namespace fair_die_weighted_coin_l495_49543

theorem fair_die_weighted_coin (n : ℕ) (p_heads : ℚ) : 
  n ≥ 7 →
  (p_heads = 1/3 ∨ p_heads = 2/3) →
  (1/n) * p_heads = 1/15 →
  n = 10 := by
  sorry

end fair_die_weighted_coin_l495_49543


namespace triangle_properties_l495_49594

/-- Triangle ABC with vertices A(1,3), B(3,1), and C(-1,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle := {
  A := (1, 3)
  B := (3, 1)
  C := (-1, 0)
}

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to get the line equation of side AB -/
def getLineAB (t : Triangle) : LineEquation := sorry

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) (h : t = triangleABC) : 
  getLineAB t = { a := 1, b := 1, c := -4 } ∧ triangleArea t = 5 := by sorry

end triangle_properties_l495_49594


namespace cats_and_fish_l495_49545

theorem cats_and_fish (c d : ℕ) : 
  (6 : ℕ) * (1 : ℕ) * (6 : ℕ) = (6 : ℕ) * (6 : ℕ) →  -- 6 cats eat 6 fish in 1 day
  c * d * (1 : ℕ) = (91 : ℕ) →                      -- c cats eat 91 fish in d days
  1 < c →                                           -- c is more than 1
  c < (10 : ℕ) →                                    -- c is less than 10
  c + d = (20 : ℕ) :=                               -- prove that c + d = 20
by sorry

end cats_and_fish_l495_49545


namespace minimum_shoeing_time_l495_49512

/-- The minimum time required for blacksmiths to shoe horses -/
theorem minimum_shoeing_time 
  (num_blacksmiths : ℕ) 
  (num_horses : ℕ) 
  (time_per_horseshoe : ℕ) 
  (horseshoes_per_horse : ℕ) : 
  num_blacksmiths = 48 → 
  num_horses = 60 → 
  time_per_horseshoe = 5 → 
  horseshoes_per_horse = 4 → 
  (num_horses * horseshoes_per_horse * time_per_horseshoe) / num_blacksmiths = 25 := by
  sorry

end minimum_shoeing_time_l495_49512


namespace man_mass_and_pressure_l495_49586

/-- Given a boat with specified dimensions and conditions, prove the mass and pressure exerted by a man --/
theorem man_mass_and_pressure (boat_length boat_breadth sink_depth : Real)
  (supplies_mass : Real) (water_density : Real) (gravity : Real)
  (h1 : boat_length = 6)
  (h2 : boat_breadth = 3)
  (h3 : sink_depth = 0.01)
  (h4 : supplies_mass = 15)
  (h5 : water_density = 1000)
  (h6 : gravity = 9.81) :
  ∃ (man_mass : Real) (pressure : Real),
    man_mass = 165 ∧
    pressure = 89.925 := by
  sorry

end man_mass_and_pressure_l495_49586


namespace triangle_angle_B_l495_49525

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_B (t : Triangle) :
  t.a = 2 ∧ t.b = 3 ∧ t.A = π/4 → t.B = π/6 := by
  sorry

end triangle_angle_B_l495_49525


namespace polynomial_simplification_l495_49526

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := by
  sorry

end polynomial_simplification_l495_49526


namespace work_completion_time_l495_49522

/-- Given a work that B can complete in 10 days, and when A and B work together,
    B's share of the total 5000 Rs wages is 3333 Rs, prove that A alone can do the work in 20 days. -/
theorem work_completion_time
  (b_time : ℝ)
  (total_wages : ℝ)
  (b_wages : ℝ)
  (h1 : b_time = 10)
  (h2 : total_wages = 5000)
  (h3 : b_wages = 3333)
  : ∃ (a_time : ℝ), a_time = 20 :=
by
  sorry

end work_completion_time_l495_49522


namespace simplification_and_sum_of_squares_l495_49550

/-- The polynomial expression to be simplified -/
def original_expression (x : ℝ) : ℝ :=
  5 * (2 * x^3 - 3 * x^2 + 4) - 6 * (x^4 - 2 * x^3 + 3 * x - 2)

/-- The simplified form of the polynomial expression -/
def simplified_expression (x : ℝ) : ℝ :=
  -6 * x^4 + 22 * x^3 - 15 * x^2 - 18 * x + 32

/-- The coefficients of the simplified expression -/
def coefficients : List ℝ := [-6, 22, -15, -18, 32]

/-- Sum of squares of the coefficients -/
def sum_of_squares : ℝ := (coefficients.map (λ c => c^2)).sum

theorem simplification_and_sum_of_squares :
  (∀ x, original_expression x = simplified_expression x) ∧
  sum_of_squares = 2093 := by
  sorry

end simplification_and_sum_of_squares_l495_49550


namespace least_subtraction_for_divisibility_l495_49505

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 1 ∧ 
  (5026 - x) % 5 = 0 ∧ 
  ∀ (y : ℕ), y < x → (5026 - y) % 5 ≠ 0 := by
  sorry

end least_subtraction_for_divisibility_l495_49505


namespace derivative_f_at_2_l495_49542

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 4 := by sorry

end derivative_f_at_2_l495_49542


namespace problem_statement_l495_49599

theorem problem_statement (x y : ℤ) (a b : ℤ) : 
  (∃ k₁ k₂ : ℤ, x - 5 = 7 * k₁ ∧ y + 7 = 7 * k₂) →
  (∃ k₃ : ℤ, x^2 + y^3 = 11 * k₃) →
  x = 7 * a + 5 →
  y = 7 * b - 7 →
  (y - x) / 13 = (7 * (b - a) - 12) / 13 :=
by sorry

end problem_statement_l495_49599


namespace average_of_six_numbers_l495_49537

theorem average_of_six_numbers (sequence : Fin 6 → ℝ) 
  (h1 : (sequence 0 + sequence 1 + sequence 2 + sequence 3) / 4 = 25)
  (h2 : (sequence 3 + sequence 4 + sequence 5) / 3 = 35)
  (h3 : sequence 3 = 25) :
  (sequence 0 + sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5) / 6 = 30 := by
sorry

end average_of_six_numbers_l495_49537


namespace fruit_basket_problem_l495_49530

theorem fruit_basket_problem :
  let oranges : ℕ := 15
  let peaches : ℕ := 9
  let pears : ℕ := 18
  let bananas : ℕ := 12
  let apples : ℕ := 24
  Nat.gcd oranges (Nat.gcd peaches (Nat.gcd pears (Nat.gcd bananas apples))) = 3 := by
  sorry

end fruit_basket_problem_l495_49530


namespace rectangle_area_l495_49575

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 6 →
  ratio = 3 →
  (2 * r) * (ratio * 2 * r) = 432 :=
by
  sorry

end rectangle_area_l495_49575


namespace multiply_72515_9999_l495_49549

theorem multiply_72515_9999 : 72515 * 9999 = 725077485 := by
  sorry

end multiply_72515_9999_l495_49549


namespace triangle_side_length_20_l495_49506

theorem triangle_side_length_20 :
  ∃ (T S : ℕ), 
    T = 20 ∧ 
    3 * T = 4 * S :=
by
  sorry

end triangle_side_length_20_l495_49506


namespace f_two_roots_iff_m_range_f_min_value_on_interval_l495_49576

/-- The function f(x) = x^2 - 4mx + 6m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*m*x + 6*m

theorem f_two_roots_iff_m_range (m : ℝ) :
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m < 0 ∨ m > 3/2 := by sorry

theorem f_min_value_on_interval (m : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f m x ≥ (
    if m ≤ 0 then 6*m
    else if m < 3/2 then -4*m^2 + 6*m
    else 9 - 6*m
  )) ∧
  (∃ x ∈ Set.Icc 0 3, f m x = (
    if m ≤ 0 then 6*m
    else if m < 3/2 then -4*m^2 + 6*m
    else 9 - 6*m
  )) := by sorry

end f_two_roots_iff_m_range_f_min_value_on_interval_l495_49576


namespace deepak_age_l495_49592

/-- Proves that Deepak's present age is 18 years given the conditions -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 2 = 26 →
  deepak_age = 18 := by
  sorry

end deepak_age_l495_49592


namespace equal_distance_point_sum_of_distances_equal_distance_time_l495_49552

def A : ℝ := -1
def B : ℝ := 3

theorem equal_distance_point (x : ℝ) : 
  |x - A| = |x - B| → x = 1 := by sorry

theorem sum_of_distances (x : ℝ) : 
  (|x - A| + |x - B| = 5) ↔ (x = -3/2 ∨ x = 7/2) := by sorry

def P (t : ℝ) : ℝ := -t
def A' (t : ℝ) : ℝ := -1 - 5*t
def B' (t : ℝ) : ℝ := 3 - 20*t

theorem equal_distance_time (t : ℝ) : 
  |P t - A' t| = |P t - B' t| ↔ (t = 4/15 ∨ t = 2/23) := by sorry

end equal_distance_point_sum_of_distances_equal_distance_time_l495_49552


namespace line_slope_one_m_value_l495_49539

/-- Given a line passing through points P (-2, m) and Q (m, 4) with a slope of 1,
    prove that the value of m is 1. -/
theorem line_slope_one_m_value (m : ℝ) : 
  (4 - m) / (m + 2) = 1 → m = 1 := by
sorry

end line_slope_one_m_value_l495_49539


namespace cashback_discount_percentage_l495_49566

theorem cashback_discount_percentage
  (iphone_price : ℝ)
  (iwatch_price : ℝ)
  (iphone_discount : ℝ)
  (iwatch_discount : ℝ)
  (total_after_cashback : ℝ)
  (h1 : iphone_price = 800)
  (h2 : iwatch_price = 300)
  (h3 : iphone_discount = 0.15)
  (h4 : iwatch_discount = 0.10)
  (h5 : total_after_cashback = 931) :
  let discounted_iphone := iphone_price * (1 - iphone_discount)
  let discounted_iwatch := iwatch_price * (1 - iwatch_discount)
  let total_after_discounts := discounted_iphone + discounted_iwatch
  let cashback_amount := total_after_discounts - total_after_cashback
  let cashback_percentage := cashback_amount / total_after_discounts * 100
  cashback_percentage = 2 := by sorry

end cashback_discount_percentage_l495_49566


namespace division_of_fractions_l495_49504

theorem division_of_fractions : 
  (5 / 6 : ℚ) / (7 / 9 : ℚ) / (11 / 13 : ℚ) = 195 / 154 := by sorry

end division_of_fractions_l495_49504


namespace total_apples_is_75_l495_49524

/-- The number of apples Benny picked from each tree -/
def benny_apples_per_tree : ℕ := 2

/-- The number of trees Benny picked from -/
def benny_trees : ℕ := 4

/-- The number of apples Dan picked from each tree -/
def dan_apples_per_tree : ℕ := 9

/-- The number of trees Dan picked from -/
def dan_trees : ℕ := 5

/-- Calculate the total number of apples picked by Benny -/
def benny_total : ℕ := benny_apples_per_tree * benny_trees

/-- Calculate the total number of apples picked by Dan -/
def dan_total : ℕ := dan_apples_per_tree * dan_trees

/-- Calculate the number of apples picked by Sarah (half of Dan's total, rounded down) -/
def sarah_total : ℕ := dan_total / 2

/-- The total number of apples picked by all three people -/
def total_apples : ℕ := benny_total + dan_total + sarah_total

theorem total_apples_is_75 : total_apples = 75 := by
  sorry

end total_apples_is_75_l495_49524


namespace simplify_expression_l495_49593

theorem simplify_expression (m n : ℝ) : m - n - (m - n) = 0 := by
  sorry

end simplify_expression_l495_49593


namespace westward_movement_l495_49558

-- Define the direction as an enumeration
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem westward_movement :
  movement 1000 Direction.West = -1000 :=
by sorry

end westward_movement_l495_49558


namespace sum_of_repeating_decimals_l495_49509

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem sum_of_repeating_decimals :
  (SingleDigitRepeatingDecimal 0 2) + (TwoDigitRepeatingDecimal 0 4) = 26 / 99 := by
  sorry

end sum_of_repeating_decimals_l495_49509


namespace inequality_proof_l495_49520

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 3) : 
  1 / (Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1)) + 
  1 / (Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1)) + 
  1 / (Real.sqrt (3 * z + 1) + Real.sqrt (3 * x + 1)) ≥ 3 / 4 := by
  sorry

end inequality_proof_l495_49520


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l495_49568

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l495_49568


namespace largest_c_for_4_in_range_l495_49532

/-- The quadratic function f(x) = x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- Theorem: The largest value of c such that 4 is in the range of f(x) = x^2 + 5x + c is 10.25 -/
theorem largest_c_for_4_in_range : 
  (∃ (x : ℝ), f 10.25 x = 4) ∧ 
  (∀ (c : ℝ), c > 10.25 → ¬∃ (x : ℝ), f c x = 4) := by
  sorry


end largest_c_for_4_in_range_l495_49532


namespace largest_two_digit_number_from_3_and_6_l495_49538

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ((n / 10 = 3 ∧ n % 10 = 6) ∨ (n / 10 = 6 ∧ n % 10 = 3))

theorem largest_two_digit_number_from_3_and_6 :
  ∀ n : ℕ, is_valid_number n → n ≤ 63 :=
sorry

end largest_two_digit_number_from_3_and_6_l495_49538


namespace cricket_team_size_l495_49571

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 0 →
  let captain_age : ℕ := 25
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 22
  let remaining_average_age : ℝ := team_average_age - 1
  (n : ℝ) * team_average_age = captain_age + wicket_keeper_age + (n - 2 : ℝ) * remaining_average_age →
  n = 11 :=
by sorry

end cricket_team_size_l495_49571


namespace binomial_12_choose_6_l495_49510

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end binomial_12_choose_6_l495_49510


namespace alcohol_percentage_solution_x_l495_49503

/-- Proves that the percentage of alcohol by volume in solution x is 10% -/
theorem alcohol_percentage_solution_x :
  ∀ (x y : ℝ),
  y = 0.30 →
  450 * y + 300 * x = 0.22 * (450 + 300) →
  x = 0.10 := by
sorry

end alcohol_percentage_solution_x_l495_49503


namespace equation_solution_l495_49536

theorem equation_solution : 
  {x : ℝ | (x^3 + 3*x^2 - x) / (x^2 + 4*x + 3) + x = -7} = {-5/2, -4} := by
  sorry

end equation_solution_l495_49536


namespace sears_tower_height_calculation_l495_49515

/-- The height of Burj Khalifa in meters -/
def burj_khalifa_height : ℕ := 830

/-- The difference in height between Burj Khalifa and Sears Tower in meters -/
def height_difference : ℕ := 303

/-- The height of Sears Tower in meters -/
def sears_tower_height : ℕ := burj_khalifa_height - height_difference

theorem sears_tower_height_calculation :
  sears_tower_height = 527 :=
by sorry

end sears_tower_height_calculation_l495_49515


namespace fraction_integer_iff_specific_p_l495_49547

theorem fraction_integer_iff_specific_p (p : ℕ+) :
  (∃ (k : ℕ+), (4 * p + 40 : ℚ) / (3 * p - 7 : ℚ) = k) ↔ p ∈ ({5, 8, 18, 50} : Set ℕ+) :=
sorry

end fraction_integer_iff_specific_p_l495_49547


namespace cistern_wet_surface_area_l495_49500

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width +  -- bottom area
  2 * length * depth +  -- longer sides area
  2 * width * depth  -- shorter sides area

/-- Theorem: The wet surface area of a 7m x 5m cistern with 1.40m water depth is 68.6 m² -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 7 5 1.40 = 68.6 := by
  sorry

#eval wetSurfaceArea 7 5 1.40

end cistern_wet_surface_area_l495_49500


namespace x_twelve_percent_greater_than_seventy_l495_49577

theorem x_twelve_percent_greater_than_seventy (x : ℝ) : 
  x = 70 * (1 + 12 / 100) → x = 78.4 := by
  sorry

end x_twelve_percent_greater_than_seventy_l495_49577


namespace favorite_fruit_strawberries_l495_49546

theorem favorite_fruit_strawberries (total students_oranges students_pears students_apples : ℕ) 
  (h_total : total = 450)
  (h_oranges : students_oranges = 70)
  (h_pears : students_pears = 120)
  (h_apples : students_apples = 147) :
  total - (students_oranges + students_pears + students_apples) = 113 := by
  sorry

end favorite_fruit_strawberries_l495_49546


namespace math_quiz_items_l495_49555

theorem math_quiz_items (score_percentage : ℚ) (num_mistakes : ℕ) : 
  score_percentage = 80 → num_mistakes = 5 → 
  (100 : ℚ) * num_mistakes / (100 - score_percentage) = 25 :=
by sorry

end math_quiz_items_l495_49555


namespace min_value_problem_l495_49527

theorem min_value_problem (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 4)
  (h5 : y^2 = x^2 + 2) (h6 : z^2 = y^2 + 2) :
  ∃ (min_val : ℝ), min_val = 4 - 2 * Real.sqrt 3 ∧ 
  ∀ (x' y' z' : ℝ), 0 ≤ x' ∧ x' ≤ y' ∧ y' ≤ z' ∧ z' ≤ 4 ∧ 
  y'^2 = x'^2 + 2 ∧ z'^2 = y'^2 + 2 → z' - x' ≥ min_val :=
by sorry

end min_value_problem_l495_49527


namespace books_bought_equals_difference_l495_49564

/-- Represents the number of books Melanie bought at the yard sale -/
def books_bought : ℕ := sorry

/-- Melanie's initial number of books -/
def initial_books : ℕ := 41

/-- Melanie's final number of books after the yard sale -/
def final_books : ℕ := 87

/-- Theorem stating that the number of books bought is the difference between final and initial books -/
theorem books_bought_equals_difference : 
  books_bought = final_books - initial_books :=
by sorry

end books_bought_equals_difference_l495_49564
