import Mathlib

namespace tenth_thousand_digit_is_seven_l4120_412045

def digit_sequence (n : ℕ) : ℕ :=
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  let digits_1_to_999 := digits_1_to_9 + digits_10_to_99 + digits_100_to_999
  let remaining_digits := n - digits_1_to_999
  let full_numbers_1000_onward := remaining_digits / 4
  let digits_from_full_numbers := full_numbers_1000_onward * 4
  let last_number := 1000 + full_numbers_1000_onward
  let remaining_digits_in_last_number := remaining_digits - digits_from_full_numbers
  if remaining_digits_in_last_number = 0 then
    (last_number - 1) % 10
  else
    (last_number / (10 ^ (4 - remaining_digits_in_last_number))) % 10

theorem tenth_thousand_digit_is_seven :
  digit_sequence 10000 = 7 := by
  sorry

end tenth_thousand_digit_is_seven_l4120_412045


namespace online_employees_probability_l4120_412003

/-- Probability of exactly k successes in n independent trials with probability p each -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ := sorry

/-- Probability of at least k successes in n independent trials with probability p each -/
def at_least_probability (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem online_employees_probability (n : ℕ) (p : ℝ) 
  (h_n : n = 6) (h_p : p = 0.5) : 
  at_least_probability n 3 p = 21/32 ∧ 
  (∀ k : ℕ, at_least_probability n k p < 0.3 ↔ k ≥ 4) := by sorry

end online_employees_probability_l4120_412003


namespace solution_set_equality_l4120_412064

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 9)) + (1 / (x^2 + 3*x - 9)) + (1 / (x^2 - 12*x - 9)) = 0

theorem solution_set_equality :
  {x : ℝ | equation x} = {1, -9, 3, -3} := by sorry

end solution_set_equality_l4120_412064


namespace expression_equals_zero_l4120_412046

theorem expression_equals_zero :
  (1 - Real.sqrt 2) ^ 0 + |2 - Real.sqrt 5| + (-1) ^ 2022 - (1/3) * Real.sqrt 45 = 0 := by
  sorry

end expression_equals_zero_l4120_412046


namespace another_hamiltonian_cycle_l4120_412057

/-- A graph with n vertices where each vertex has exactly 3 neighbors -/
structure ThreeRegularGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  degree_three : ∀ v : Fin n, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- A Hamiltonian cycle in a graph -/
def HamiltonianCycle {n : ℕ} (G : ThreeRegularGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.toFinset = G.vertices }

/-- Two Hamiltonian cycles are equivalent if one can be obtained from the other by rotation or reflection -/
def EquivalentCycles {n : ℕ} (G : ThreeRegularGraph n) (c1 c2 : HamiltonianCycle G) : Prop :=
  ∃ (k : ℕ) (reflect : Bool),
    c2.val = if reflect then c1.val.reverse.rotateRight k else c1.val.rotateRight k

theorem another_hamiltonian_cycle {n : ℕ} (G : ThreeRegularGraph n) (c : HamiltonianCycle G) :
  ∃ (c' : HamiltonianCycle G), ¬EquivalentCycles G c c' :=
sorry

end another_hamiltonian_cycle_l4120_412057


namespace total_digits_100000_l4120_412044

def total_digits (n : ℕ) : ℕ :=
  let d1 := 9
  let d2 := 90 * 2
  let d3 := 900 * 3
  let d4 := 9000 * 4
  let d5 := (n - 10000 + 1) * 5
  let d6 := if n = 100000 then 6 else 0
  d1 + d2 + d3 + d4 + d5 + d6

theorem total_digits_100000 :
  total_digits 100000 = 488895 := by
  sorry

end total_digits_100000_l4120_412044


namespace log_inequality_equiv_interval_l4120_412027

-- Define the logarithm function with base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_inequality_equiv_interval (x : ℝ) :
  (log2 (4 - x) > log2 (3 * x)) ↔ (0 < x ∧ x < 1) :=
by sorry

end log_inequality_equiv_interval_l4120_412027


namespace day_318_is_monday_l4120_412086

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a specific day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Given that the 45th day of 2003 is a Monday, 
    prove that the 318th day of 2003 is also a Monday -/
theorem day_318_is_monday (d45 d318 : DayInYear) 
  (h1 : d45.dayNumber = 45)
  (h2 : d45.dayOfWeek = DayOfWeek.Monday)
  (h3 : d318.dayNumber = 318) :
  d318.dayOfWeek = DayOfWeek.Monday := by
  sorry


end day_318_is_monday_l4120_412086


namespace min_value_absolute_sum_l4120_412032

theorem min_value_absolute_sum (x : ℝ) : 
  |x - 4| + |x + 6| + |x - 5| ≥ 1 ∧ ∃ y : ℝ, |y - 4| + |y + 6| + |y - 5| = 1 :=
by sorry

end min_value_absolute_sum_l4120_412032


namespace train_crossing_time_l4120_412084

/-- Given a train and a platform with specific dimensions, calculate the time taken for the train to cross the platform. -/
theorem train_crossing_time (train_length : ℝ) (signal_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : train_length = 450)
  (h2 : signal_crossing_time = 18)
  (h3 : platform_length = 525) :
  (train_length + platform_length) / (train_length / signal_crossing_time) = 39 := by
  sorry

end train_crossing_time_l4120_412084


namespace johns_arcade_spending_l4120_412017

/-- The fraction of John's allowance spent at the arcade -/
def arcade_fraction : ℚ := 3/5

/-- John's weekly allowance in dollars -/
def weekly_allowance : ℚ := 18/5

/-- The amount John had left after spending at the arcade and toy store, in dollars -/
def remaining_amount : ℚ := 24/25

theorem johns_arcade_spending :
  let remaining_after_arcade : ℚ := weekly_allowance * (1 - arcade_fraction)
  let spent_at_toy_store : ℚ := remaining_after_arcade * (1/3)
  remaining_after_arcade - spent_at_toy_store = remaining_amount :=
by sorry

end johns_arcade_spending_l4120_412017


namespace lilys_lottery_prize_l4120_412083

/-- The amount of money the lottery winner will receive -/
def lottery_prize (num_tickets : ℕ) (initial_price : ℕ) (price_increment : ℕ) (profit : ℕ) : ℕ :=
  let total_sales := (num_tickets * (2 * initial_price + (num_tickets - 1) * price_increment)) / 2
  total_sales - profit

/-- Theorem stating the lottery prize for Lily's specific scenario -/
theorem lilys_lottery_prize :
  lottery_prize 5 1 1 4 = 11 := by
  sorry

end lilys_lottery_prize_l4120_412083


namespace total_lockers_l4120_412061

/-- Represents the layout of lockers in a school -/
structure LockerLayout where
  left : ℕ  -- Number of lockers to the left of Yunjeong's locker
  right : ℕ  -- Number of lockers to the right of Yunjeong's locker
  front : ℕ  -- Number of lockers in front of Yunjeong's locker
  back : ℕ  -- Number of lockers behind Yunjeong's locker

/-- Theorem stating the total number of lockers given Yunjeong's locker position -/
theorem total_lockers (layout : LockerLayout) : 
  layout.left = 6 → 
  layout.right = 12 → 
  layout.front = 7 → 
  layout.back = 13 → 
  (layout.left + 1 + layout.right) * (layout.front + 1 + layout.back) = 399 := by
  sorry

#check total_lockers

end total_lockers_l4120_412061


namespace equation_solutions_l4120_412006

theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x * y + y * z + z * x = 2 * (x + y + z)} =
  {(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (2, 2, 2), (4, 1, 2), (4, 2, 1)} := by
sorry

end equation_solutions_l4120_412006


namespace mary_nickels_count_l4120_412055

/-- The number of nickels Mary has after receiving some from her dad and sister -/
def total_nickels (initial : ℕ) (from_dad : ℕ) (from_sister : ℕ) : ℕ :=
  initial + from_dad + from_sister

/-- Theorem stating that Mary's total nickels is the sum of her initial amount and what she received -/
theorem mary_nickels_count : total_nickels 7 12 9 = 28 := by
  sorry

end mary_nickels_count_l4120_412055


namespace no_savings_on_joint_purchase_l4120_412068

/-- Represents the cost of a window in dollars -/
def window_cost : ℕ := 100

/-- Represents the number of windows purchased to get free windows -/
def windows_for_offer : ℕ := 9

/-- Represents the number of free windows given in the offer -/
def free_windows : ℕ := 2

/-- Represents the number of windows Dave needs -/
def dave_windows : ℕ := 10

/-- Represents the number of windows Doug needs -/
def doug_windows : ℕ := 9

/-- Calculates the cost of purchasing windows with the special offer -/
def calculate_cost (num_windows : ℕ) : ℕ :=
  let paid_windows := num_windows - (num_windows / windows_for_offer) * free_windows
  paid_windows * window_cost

/-- Theorem stating that there are no savings when Dave and Doug purchase windows together -/
theorem no_savings_on_joint_purchase :
  calculate_cost dave_windows + calculate_cost doug_windows =
  calculate_cost (dave_windows + doug_windows) :=
sorry

end no_savings_on_joint_purchase_l4120_412068


namespace suraj_average_after_ninth_innings_l4120_412002

/-- Represents a cricket player's performance -/
structure CricketPerformance where
  innings : ℕ
  lowestScore : ℕ
  highestScore : ℕ
  fiftyPlusInnings : ℕ
  totalRuns : ℕ

/-- Calculates the average runs per innings -/
def average (cp : CricketPerformance) : ℚ :=
  cp.totalRuns / cp.innings

theorem suraj_average_after_ninth_innings 
  (suraj : CricketPerformance)
  (h1 : suraj.innings = 8)
  (h2 : suraj.lowestScore = 25)
  (h3 : suraj.highestScore = 80)
  (h4 : suraj.fiftyPlusInnings = 3)
  (h5 : average suraj + 6 = average { suraj with 
    innings := suraj.innings + 1, 
    totalRuns := suraj.totalRuns + 90 }) :
  average { suraj with 
    innings := suraj.innings + 1, 
    totalRuns := suraj.totalRuns + 90 } = 42 := by
  sorry


end suraj_average_after_ninth_innings_l4120_412002


namespace only_η_is_hypergeometric_l4120_412066

-- Define the types for balls and random variables
inductive BallColor
| Black
| White

structure Ball :=
  (color : BallColor)
  (number : Nat)

def TotalBalls : Nat := 10
def BlackBalls : Nat := 6
def WhiteBalls : Nat := 4
def DrawnBalls : Nat := 4

-- Define the random variables
def X (draw : Finset Ball) : Nat := sorry
def Y (draw : Finset Ball) : Nat := sorry
def ξ (draw : Finset Ball) : Nat := sorry
def η (draw : Finset Ball) : Nat := sorry

-- Define the hypergeometric distribution
def IsHypergeometric (f : (Finset Ball) → Nat) : Prop := sorry

-- State the theorem
theorem only_η_is_hypergeometric :
  IsHypergeometric η ∧
  ¬IsHypergeometric X ∧
  ¬IsHypergeometric Y ∧
  ¬IsHypergeometric ξ :=
sorry

end only_η_is_hypergeometric_l4120_412066


namespace distance_travelled_l4120_412025

-- Define the velocity function
def v (t : ℝ) : ℝ := 2 * t - 3

-- Define the theorem
theorem distance_travelled (t₀ t₁ : ℝ) (h : 0 ≤ t₀ ∧ t₁ = 5) :
  ∫ t in t₀..t₁, |v t| = 29/2 := by
  sorry

end distance_travelled_l4120_412025


namespace only_parallel_assertion_correct_l4120_412048

/-- Represents a line in 3D space -/
structure Line3D where
  -- This is just a placeholder definition
  dummy : Unit

/-- Perpendicular relation between two lines -/
def perpendicular (a b : Line3D) : Prop :=
  sorry

/-- Skew relation between two lines -/
def skew (a b : Line3D) : Prop :=
  sorry

/-- Intersection relation between two lines -/
def intersects (a b : Line3D) : Prop :=
  sorry

/-- Coplanar relation between two lines -/
def coplanar (a b : Line3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel (a b : Line3D) : Prop :=
  sorry

/-- Theorem stating that only the parallel assertion is correct -/
theorem only_parallel_assertion_correct (a b c : Line3D) :
  (¬ (∀ a b c, perpendicular a b → perpendicular b c → perpendicular a c)) ∧
  (¬ (∀ a b c, skew a b → skew b c → skew a c)) ∧
  (¬ (∀ a b c, intersects a b → intersects b c → intersects a c)) ∧
  (¬ (∀ a b c, coplanar a b → coplanar b c → coplanar a c)) ∧
  (∀ a b c, parallel a b → parallel b c → parallel a c) :=
by sorry

end only_parallel_assertion_correct_l4120_412048


namespace polynomial_division_l4120_412033

theorem polynomial_division (a : ℝ) (h : a ≠ 0) :
  (9 * a^6 - 12 * a^3) / (3 * a^3) = 3 * a^3 - 4 := by
  sorry

end polynomial_division_l4120_412033


namespace power_equality_l4120_412022

theorem power_equality (n : ℕ) : 4^n = 64^2 → n = 6 := by
  sorry

end power_equality_l4120_412022


namespace product_from_lcm_gcd_l4120_412074

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 60)
  (h_gcd : Nat.gcd x y = 10) : 
  x * y = 600 := by
sorry

end product_from_lcm_gcd_l4120_412074


namespace locus_of_q_l4120_412024

/-- The ellipse in the problem -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The hyperbola that is the locus of Q -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | (q.1^2 / a^2) - (q.2^2 / b^2) = 1}

/-- P and P' form a vertical chord of the ellipse -/
def VerticalChord (a b : ℝ) (p p' : ℝ × ℝ) : Prop :=
  p ∈ Ellipse a b ∧ p' ∈ Ellipse a b ∧ p.1 = p'.1

/-- Q is the intersection of A'P and AP' -/
def IntersectionPoint (a : ℝ) (p p' q : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ,
    q.1 = t * (p.1 + a) + (1 - t) * (-a) ∧
    q.2 = t * p.2 ∧
    q.1 = s * (p'.1 - a) + (1 - s) * a ∧
    q.2 = s * p'.2

/-- The main theorem -/
theorem locus_of_q (a b : ℝ) (p p' q : ℝ × ℝ) 
    (h_ab : a > 0 ∧ b > 0)
    (h_ellipse : p ∈ Ellipse a b ∧ p' ∈ Ellipse a b)
    (h_vertical : VerticalChord a b p p')
    (h_intersect : IntersectionPoint a p p' q) :
  q ∈ Hyperbola a b := by
  sorry

end locus_of_q_l4120_412024


namespace prob_one_letter_each_name_l4120_412071

/-- Probability of selecting one letter from each person's name -/
theorem prob_one_letter_each_name :
  let total_cards : ℕ := 14
  let elena_cards : ℕ := 5
  let mark_cards : ℕ := 4
  let julia_cards : ℕ := 5
  let num_permutations : ℕ := 6  -- 3! permutations of 3 items
  
  elena_cards + mark_cards + julia_cards = total_cards →
  
  (elena_cards : ℚ) / total_cards *
  (mark_cards : ℚ) / (total_cards - 1) *
  (julia_cards : ℚ) / (total_cards - 2) *
  num_permutations = 25 / 91 :=
by sorry

end prob_one_letter_each_name_l4120_412071


namespace sidney_cat_food_l4120_412047

/-- Represents the amount of food each adult cat eats per day -/
def adult_cat_food : ℝ := 1

theorem sidney_cat_food :
  let num_kittens : ℕ := 4
  let num_adult_cats : ℕ := 3
  let initial_food : ℕ := 7
  let kitten_food_per_day : ℚ := 3/4
  let additional_food : ℕ := 35
  let days : ℕ := 7
  
  (num_kittens : ℝ) * kitten_food_per_day * days +
  (num_adult_cats : ℝ) * adult_cat_food * days =
  (initial_food : ℝ) + additional_food :=
by sorry

#check sidney_cat_food

end sidney_cat_food_l4120_412047


namespace fraction_inequality_solution_set_l4120_412037

theorem fraction_inequality_solution_set (x : ℝ) :
  (x - 1) / (x - 3) < 0 ↔ 1 < x ∧ x < 3 :=
by sorry

end fraction_inequality_solution_set_l4120_412037


namespace stratified_sample_grade10_l4120_412008

theorem stratified_sample_grade10 (total_sample : ℕ) (grade12 : ℕ) (grade11 : ℕ) (grade10 : ℕ) :
  total_sample = 50 →
  grade12 = 750 →
  grade11 = 850 →
  grade10 = 900 →
  (grade10 * total_sample) / (grade12 + grade11 + grade10) = 18 := by
  sorry

end stratified_sample_grade10_l4120_412008


namespace hyperbola_asymptote_ratio_l4120_412072

theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.arctan (2 * (b / a) / (1 - (b / a)^2)) = π / 4) →
  a / b = 1 / (-1 + Real.sqrt 2) :=
by sorry

end hyperbola_asymptote_ratio_l4120_412072


namespace range_of_a_l4120_412011

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

-- Define the property that the quadratic is always positive
def always_positive (a : ℝ) : Prop := ∀ x : ℝ, f a x > 0

-- Theorem statement
theorem range_of_a : Set.Icc 0 3 = {a : ℝ | always_positive a} := by sorry

end range_of_a_l4120_412011


namespace gcd_repeating_six_digit_l4120_412060

def is_repeating_six_digit (n : ℕ) : Prop :=
  ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeating_six_digit :
  ∃ d : ℕ, d > 0 ∧ (∀ n : ℕ, is_repeating_six_digit n → d ∣ n) ∧
  (∀ d' : ℕ, d' > 0 → (∀ n : ℕ, is_repeating_six_digit n → d' ∣ n) → d' ≤ d) ∧
  d = 1001 :=
sorry

end gcd_repeating_six_digit_l4120_412060


namespace binomial_expectation_from_variance_l4120_412089

/-- 
Given a binomial distribution with 4 trials and probability p of success on each trial,
if the variance of the distribution is 1, then the expected value is 2.
-/
theorem binomial_expectation_from_variance 
  (p : ℝ) 
  (h_prob : 0 ≤ p ∧ p ≤ 1) 
  (h_var : 4 * p * (1 - p) = 1) : 
  4 * p = 2 := by
sorry

end binomial_expectation_from_variance_l4120_412089


namespace mother_three_times_daughter_age_l4120_412063

/-- Proves that in 9 years, a mother who is currently 42 years old will be three times as old as her daughter who is currently 8 years old. -/
theorem mother_three_times_daughter_age (mother_age : ℕ) (daughter_age : ℕ) (years : ℕ) : 
  mother_age = 42 → daughter_age = 8 → years = 9 → 
  mother_age + years = 3 * (daughter_age + years) :=
by sorry

end mother_three_times_daughter_age_l4120_412063


namespace store_transaction_loss_l4120_412091

def selling_price : ℝ := 60

theorem store_transaction_loss (cost_price_1 cost_price_2 : ℝ) 
  (h1 : (selling_price - cost_price_1) / cost_price_1 = 1/2)
  (h2 : (cost_price_2 - selling_price) / cost_price_2 = 1/2) :
  2 * selling_price - (cost_price_1 + cost_price_2) = -selling_price / 3 := by
  sorry

end store_transaction_loss_l4120_412091


namespace uphill_distance_l4120_412059

/-- Proves that the uphill distance is 45 km given the conditions of the problem -/
theorem uphill_distance (flat_speed : ℝ) (uphill_speed : ℝ) (extra_flat_distance : ℝ) :
  flat_speed = 20 →
  uphill_speed = 12 →
  extra_flat_distance = 30 →
  ∃ (uphill_distance : ℝ),
    uphill_distance / uphill_speed = (uphill_distance + extra_flat_distance) / flat_speed ∧
    uphill_distance = 45 :=
by sorry

end uphill_distance_l4120_412059


namespace sarah_toads_count_l4120_412026

/-- Proves that Sarah has 100 toads given the conditions of the problem -/
theorem sarah_toads_count : ∀ (tim_toads jim_toads sarah_toads : ℕ),
  tim_toads = 30 →
  jim_toads = tim_toads + 20 →
  sarah_toads = 2 * jim_toads →
  sarah_toads = 100 := by
sorry

end sarah_toads_count_l4120_412026


namespace lucy_grocery_cost_l4120_412013

/-- Represents the total cost of Lucy's grocery purchases in USD -/
def total_cost_usd (cookies_packs : ℕ) (cookies_price : ℚ)
                   (noodles_packs : ℕ) (noodles_price : ℚ)
                   (soup_cans : ℕ) (soup_price : ℚ)
                   (cereals_boxes : ℕ) (cereals_price : ℚ)
                   (crackers_packs : ℕ) (crackers_price : ℚ)
                   (usd_to_eur : ℚ) (usd_to_gbp : ℚ) : ℚ :=
  cookies_packs * cookies_price +
  (noodles_packs * noodles_price) / usd_to_eur +
  (soup_cans * soup_price) / usd_to_gbp +
  cereals_boxes * cereals_price +
  (crackers_packs * crackers_price) / usd_to_eur

/-- The theorem stating that Lucy's total grocery cost is $183.92 -/
theorem lucy_grocery_cost :
  total_cost_usd 12 (5/2) 16 (9/5) 28 (6/5) 5 (17/5) 45 (11/10) (17/20) (3/4) = 18392/100 := by
  sorry

end lucy_grocery_cost_l4120_412013


namespace tv_price_proof_l4120_412030

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem tv_price_proof (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  let total_price := a * 10000 + 6000 + 700 + 90 + b
  is_divisible_by total_price 72 →
  (total_price / 72 : ℚ) = 511 := by
  sorry

end tv_price_proof_l4120_412030


namespace solution_set_of_inequality_l4120_412077

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_point_A : f 0 = 3)
  (h_point_B : f 3 = -1) :
  {x : ℝ | |f (x + 1) - 1| < 2} = Set.Ioo (-1 : ℝ) 2 := by sorry

end solution_set_of_inequality_l4120_412077


namespace max_area_equilateral_triangle_in_rectangle_l4120_412052

/-- The maximum area of an equilateral triangle inscribed in a 12x17 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), A = 325 * Real.sqrt 3 - 612 ∧ 
  ∀ (triangle_area : ℝ), 
    (∃ (x y : ℝ), 
      0 ≤ x ∧ x ≤ 12 ∧ 
      0 ≤ y ∧ y ≤ 17 ∧ 
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ A :=
by sorry

end max_area_equilateral_triangle_in_rectangle_l4120_412052


namespace unique_solution_l4120_412056

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem unique_solution : 
  ∃! x : ℕ, 
    digit_product x = 44 * x - 86868 ∧ 
    is_perfect_cube (digit_sum x) ∧
    x = 1989 := by
  sorry

end unique_solution_l4120_412056


namespace max_students_equal_distribution_l4120_412098

/-- The maximum number of students among whom 781 pens and 710 pencils can be distributed equally -/
theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 781) (h_pencils : pencils = 710) :
  (∃ (students pen_per_student pencil_per_student : ℕ), 
    students * pen_per_student = pens ∧ 
    students * pencil_per_student = pencils ∧ 
    ∀ s : ℕ, s * pen_per_student = pens → s * pencil_per_student = pencils → s ≤ students) →
  Nat.gcd pens pencils = 71 :=
by sorry

end max_students_equal_distribution_l4120_412098


namespace probability_of_specific_distribution_l4120_412021

-- Define the number of balls and boxes
def num_balls : ℕ := 6
def num_boxes : ℕ := 4

-- Define the probability of a specific distribution
def prob_specific_distribution : ℚ := 45 / 128

-- Define the function to calculate the total number of ways to distribute balls
def total_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

-- Define the function to calculate the number of ways to distribute balls in the specific pattern
def specific_distribution_count (balls : ℕ) (boxes : ℕ) : ℕ :=
  (balls.choose 3) * ((balls - 3).choose 2) * (boxes.factorial)

-- Theorem statement
theorem probability_of_specific_distribution :
  (specific_distribution_count num_balls num_boxes : ℚ) / (total_distributions num_balls num_boxes : ℚ) = prob_specific_distribution :=
sorry

end probability_of_specific_distribution_l4120_412021


namespace least_x_value_l4120_412020

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Prime p) 
  (h3 : Prime (x / (9 * p))) (h4 : Odd (x / (9 * p))) :
  x ≥ 90 ∧ ∃ (x₀ : ℕ), x₀ = 90 ∧ 
    Prime (x₀ / (9 * p)) ∧ Odd (x₀ / (9 * p)) :=
sorry

end least_x_value_l4120_412020


namespace fraction_modification_l4120_412076

theorem fraction_modification (a b c d x : ℚ) : 
  a ≠ b →
  b ≠ 0 →
  (2 * a + x) / (3 * b + x) = c / d →
  ∃ (k₁ k₂ : ℚ), c = k₁ * x ∧ d = k₂ * x →
  x = (3 * b * c - 2 * a * d) / (d - c) := by
sorry

end fraction_modification_l4120_412076


namespace smallest_number_l4120_412014

theorem smallest_number (a b c d : ℝ) (ha : a = -2) (hb : b = 2) (hc : c = -4) (hd : d = -1) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end smallest_number_l4120_412014


namespace smallest_of_three_consecutive_even_numbers_l4120_412070

theorem smallest_of_three_consecutive_even_numbers (a b c : ℕ) : 
  (∃ n : ℕ, a = 2 * n ∧ b = 2 * n + 2 ∧ c = 2 * n + 4) →  -- consecutive even numbers
  a + b + c = 162 →                                      -- sum is 162
  a = 52 :=                                              -- smallest number is 52
by sorry

end smallest_of_three_consecutive_even_numbers_l4120_412070


namespace sine_inequality_l4120_412069

theorem sine_inequality : 
  let sin60 := Real.sqrt 3 / 2
  let sin62 := (Real.sqrt 2 / 2) * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let sin64 := 2 * (Real.cos (13 * π / 180))^2 - 1
  sin60 < sin62 ∧ sin62 < sin64 := by sorry

end sine_inequality_l4120_412069


namespace range_of_a_l4120_412018

/-- Proposition p -/
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1

/-- Proposition q -/
def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

/-- The set of x satisfying proposition p -/
def A : Set ℝ := {x | p x}

/-- The set of x satisfying proposition q -/
def B (a : ℝ) : Set ℝ := {x | q x a}

/-- The condition that ¬p is a necessary but not sufficient condition for ¬q -/
def condition (a : ℝ) : Prop := A ⊂ B a ∧ A ≠ B a

/-- The theorem stating the range of a -/
theorem range_of_a : ∀ a : ℝ, condition a ↔ 0 ≤ a ∧ a ≤ 1/2 ∧ a ≠ 1/2 :=
sorry

end range_of_a_l4120_412018


namespace sixth_year_fee_l4120_412043

def membership_fee (initial_fee : ℕ) (yearly_increase : ℕ) (year : ℕ) : ℕ :=
  initial_fee + (year - 1) * yearly_increase

theorem sixth_year_fee :
  membership_fee 80 10 6 = 130 := by
  sorry

end sixth_year_fee_l4120_412043


namespace binomial_10_choose_3_l4120_412097

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l4120_412097


namespace total_sticks_is_129_l4120_412051

/-- The number of sticks needed for Simon's raft -/
def simon_sticks : ℕ := 36

/-- The number of sticks needed for Gerry's raft -/
def gerry_sticks : ℕ := (2 * simon_sticks) / 3

/-- The number of sticks needed for Micky's raft -/
def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

/-- The total number of sticks needed for all three rafts -/
def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks

/-- Theorem stating that the total number of sticks needed is 129 -/
theorem total_sticks_is_129 : total_sticks = 129 := by
  sorry

#eval total_sticks

end total_sticks_is_129_l4120_412051


namespace gcd_of_powers_of_two_l4120_412007

theorem gcd_of_powers_of_two : Nat.gcd (2^1015 - 1) (2^1024 - 1) = 2^9 - 1 := by sorry

end gcd_of_powers_of_two_l4120_412007


namespace z_is_real_z_is_imaginary_z_is_pure_imaginary_l4120_412092

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 12) (a^2 - 5*a + 6)

-- Theorem for real values of z
theorem z_is_real (a : ℝ) : (z a).im = 0 ↔ a = 2 ∨ a = 3 := by sorry

-- Theorem for imaginary values of z
theorem z_is_imaginary (a : ℝ) : (z a).im ≠ 0 ↔ a ≠ 2 ∧ a ≠ 3 := by sorry

-- Theorem for pure imaginary values of z
theorem z_is_pure_imaginary (a : ℝ) : (z a).re = 0 ∧ (z a).im ≠ 0 ↔ a = 4 := by sorry

end z_is_real_z_is_imaginary_z_is_pure_imaginary_l4120_412092


namespace f_composed_with_g_l4120_412035

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x - 2

theorem f_composed_with_g : f (2 + g 3) = 5 := by
  sorry

end f_composed_with_g_l4120_412035


namespace skew_lines_projection_not_two_points_l4120_412054

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D

-- Define a type for points in 2D space (the projection plane)
structure Point2D where
  -- Add necessary fields to represent a point in 2D

-- Define a projection function from 3D to 2D
def project (l : Line3D) : Point2D :=
  sorry

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem skew_lines_projection_not_two_points 
  (l1 l2 : Line3D) (h : are_skew l1 l2) : 
  ¬(∃ (p1 p2 : Point2D), project l1 = p1 ∧ project l2 = p2 ∧ p1 ≠ p2) :=
sorry

end skew_lines_projection_not_two_points_l4120_412054


namespace expression_evaluation_l4120_412050

theorem expression_evaluation : 
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 := by sorry

end expression_evaluation_l4120_412050


namespace largest_of_three_consecutive_integers_l4120_412085

theorem largest_of_three_consecutive_integers (n : ℤ) 
  (h : (n - 1) + n + (n + 1) = 90) : 
  max (n - 1) (max n (n + 1)) = 31 := by
sorry

end largest_of_three_consecutive_integers_l4120_412085


namespace max_distance_from_origin_l4120_412010

/-- The post position -/
def post : ℝ × ℝ := (2, 5)

/-- The rope length -/
def rope_length : ℝ := 8

/-- The rectangle's vertices -/
def rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (0, 10), (10, 0), (10, 10)]

/-- Check if a point is within the rectangle -/
def in_rectangle (p : ℝ × ℝ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 10

/-- Check if a point is within the rope's reach -/
def in_rope_reach (p : ℝ × ℝ) : Prop :=
  (p.1 - post.1)^2 + (p.2 - post.2)^2 ≤ rope_length^2

/-- The maximum distance from origin theorem -/
theorem max_distance_from_origin :
  ∃ (p : ℝ × ℝ), in_rectangle p ∧ in_rope_reach p ∧
  ∀ (q : ℝ × ℝ), in_rectangle q → in_rope_reach q →
  p.1^2 + p.2^2 ≥ q.1^2 + q.2^2 ∧
  p.1^2 + p.2^2 = 125 :=
sorry

end max_distance_from_origin_l4120_412010


namespace minimum_seating_arrangement_l4120_412036

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement is valid -/
def is_valid_seating (s : CircularSeating) : Prop :=
  s.seated_people > 0 ∧ 
  s.seated_people ≤ s.total_chairs ∧
  s.total_chairs % s.seated_people = 0

/-- Checks if any additional person must sit next to someone -/
def forces_adjacent_seating (s : CircularSeating) : Prop :=
  s.total_chairs / s.seated_people ≤ 4

/-- The main theorem to prove -/
theorem minimum_seating_arrangement :
  ∃ (s : CircularSeating), 
    s.total_chairs = 75 ∧
    is_valid_seating s ∧
    forces_adjacent_seating s ∧
    (∀ (t : CircularSeating), 
      t.total_chairs = 75 → 
      is_valid_seating t → 
      forces_adjacent_seating t → 
      s.seated_people ≤ t.seated_people) ∧
    s.seated_people = 19 :=
  sorry

end minimum_seating_arrangement_l4120_412036


namespace container_capacity_l4120_412012

theorem container_capacity (x : ℝ) 
  (h1 : (1/4) * x + 300 = (3/4) * x) : x = 600 := by
  sorry

end container_capacity_l4120_412012


namespace line_perpendicular_and_tangent_l4120_412023

/-- The given line -/
def given_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

/-- The given curve -/
def given_curve (x y : ℝ) : Prop := y = x^3 + 3*x^2 - 5

/-- The line we want to prove is correct -/
def target_line (x y : ℝ) : Prop := 3*x + y + 6 = 0

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A line is tangent to a curve if it touches the curve at exactly one point -/
def tangent_to_curve (line : (ℝ → ℝ → Prop)) (curve : (ℝ → ℝ → Prop)) : Prop :=
  ∃! p : ℝ × ℝ, line p.1 p.2 ∧ curve p.1 p.2

theorem line_perpendicular_and_tangent :
  (∃ m₁ m₂ : ℝ, perpendicular m₁ m₂ ∧ 
    (∀ x y : ℝ, given_line x y → y = m₁*x + 1/6) ∧
    (∀ x y : ℝ, target_line x y → y = m₂*x - 2)) ∧
  tangent_to_curve target_line given_curve :=
sorry

end line_perpendicular_and_tangent_l4120_412023


namespace intersection_perpendicular_implies_a_value_l4120_412087

-- Define the line equation
def line_eq (x y a : ℝ) : Prop := x - y + a = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop := sorry

theorem intersection_perpendicular_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), line_eq A.1 A.2 a ∧ line_eq B.1 B.2 a ∧ 
                     circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
                     perpendicular A B circle_center) →
  a = 0 ∨ a = 6 := by
  sorry

end intersection_perpendicular_implies_a_value_l4120_412087


namespace dress_final_price_l4120_412034

/-- The final price of a dress after multiple discounts and tax -/
def finalPrice (d : ℝ) : ℝ :=
  let price1 := d * (1 - 0.45)  -- After first discount
  let price2 := price1 * (1 - 0.30)  -- After second discount
  let price3 := price2 * (1 - 0.25)  -- After third discount
  let price4 := price3 * (1 - 0.50)  -- After staff discount
  price4 * (1 + 0.10)  -- After sales tax

/-- Theorem stating the final price of the dress -/
theorem dress_final_price (d : ℝ) : finalPrice d = 0.1588125 * d := by
  sorry

end dress_final_price_l4120_412034


namespace mans_upstream_speed_l4120_412082

/-- Given a man's speed in still water and downstream speed, calculate his upstream speed -/
theorem mans_upstream_speed (v_still : ℝ) (v_downstream : ℝ) (h1 : v_still = 75) (h2 : v_downstream = 90) :
  v_still - (v_downstream - v_still) = 60 := by
  sorry

end mans_upstream_speed_l4120_412082


namespace geometric_sequence_sum_l4120_412067

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25 →
  a 4 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l4120_412067


namespace complement_of_M_in_U_l4120_412009

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end complement_of_M_in_U_l4120_412009


namespace problem_grid_square_count_l4120_412088

/-- Represents a grid with vertical and horizontal lines -/
structure Grid :=
  (vertical_lines : ℕ)
  (horizontal_lines : ℕ)
  (vertical_spacing : List ℕ)
  (horizontal_spacing : List ℕ)

/-- Counts the number of squares in a grid -/
def count_squares (g : Grid) : ℕ :=
  sorry

/-- The specific grid described in the problem -/
def problem_grid : Grid :=
  { vertical_lines := 5,
    horizontal_lines := 6,
    vertical_spacing := [1, 2, 1, 1],
    horizontal_spacing := [2, 1, 1, 1] }

/-- Theorem stating that the number of squares in the problem grid is 23 -/
theorem problem_grid_square_count :
  count_squares problem_grid = 23 :=
by sorry

end problem_grid_square_count_l4120_412088


namespace circle_symmetry_l4120_412000

/-- Given a circle and a line of symmetry, prove that another circle is symmetric to the given circle about the line. -/
theorem circle_symmetry (x y : ℝ) :
  let original_circle := (x - 1)^2 + (y - 2)^2 = 1
  let symmetry_line := x - y - 2 = 0
  let symmetric_circle := (x - 4)^2 + (y + 1)^2 = 1
  (∀ (x₀ y₀ : ℝ), original_circle → 
    ∃ (x₁ y₁ : ℝ), symmetric_circle ∧ 
    ((x₀ + x₁) / 2 - (y₀ + y₁) / 2 - 2 = 0)) :=
by sorry

end circle_symmetry_l4120_412000


namespace solve_for_a_l4120_412001

theorem solve_for_a : ∀ x a : ℝ, 2 * x + a - 9 = 0 → x = 2 → a = 5 := by
  sorry

end solve_for_a_l4120_412001


namespace ivy_collectors_edition_dolls_l4120_412041

theorem ivy_collectors_edition_dolls 
  (dina_dolls : ℕ)
  (ivy_dolls : ℕ)
  (h1 : dina_dolls = 60)
  (h2 : dina_dolls = 2 * ivy_dolls)
  (h3 : ivy_dolls > 0)
  : (2 : ℚ) / 3 * ivy_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l4120_412041


namespace seeds_per_flowerbed_l4120_412016

theorem seeds_per_flowerbed (total_seeds : ℕ) (num_flowerbeds : ℕ) (seeds_per_bed : ℕ) :
  total_seeds = 32 →
  num_flowerbeds = 8 →
  total_seeds = num_flowerbeds * seeds_per_bed →
  seeds_per_bed = 4 :=
by sorry

end seeds_per_flowerbed_l4120_412016


namespace AB_product_l4120_412039

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 0, -2]
def B_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, -1/2; 0, 2]

theorem AB_product :
  let B := B_inv⁻¹
  A * B = !![1, 5/4; 0, -1] := by sorry

end AB_product_l4120_412039


namespace trigonometric_problem_l4120_412065

theorem trigonometric_problem (α : Real) 
  (h1 : 3 * Real.pi / 4 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan α + 1 / Real.tan α = -10/3) : 
  Real.tan α = -1/3 ∧ 
  (5 * Real.sin (α/2)^2 + 8 * Real.sin (α/2) * Real.cos (α/2) + 11 * Real.cos (α/2)^2 - 8) / 
  (Real.sqrt 2 * Real.sin (α - Real.pi/4)) = -5/4 := by
  sorry

end trigonometric_problem_l4120_412065


namespace pencils_given_to_joyce_l4120_412080

theorem pencils_given_to_joyce (initial_pencils : ℝ) (remaining_pencils : ℕ) 
  (h1 : initial_pencils = 51.0)
  (h2 : remaining_pencils = 45) :
  initial_pencils - remaining_pencils = 6 := by
  sorry

end pencils_given_to_joyce_l4120_412080


namespace parabola_line_intersection_dot_product_l4120_412058

/-- Given a parabola y² = 4x and a line passing through (1,0) intersecting the parabola at A and B,
    prove that OB · OC = -5, where C is symmetric to A with respect to the y-axis -/
theorem parabola_line_intersection_dot_product :
  ∀ (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ),
  -- Line passes through (1,0)
  y₁ = k * (x₁ - 1) →
  y₂ = k * (x₂ - 1) →
  -- A and B are on the parabola
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  -- A and B are distinct points
  x₁ ≠ x₂ →
  -- C is symmetric to A with respect to y-axis
  let xc := -x₁
  let yc := y₁
  -- OB · OC = -5
  x₂ * xc + y₂ * yc = -5 :=
by sorry

end parabola_line_intersection_dot_product_l4120_412058


namespace number_multiplication_problem_l4120_412075

theorem number_multiplication_problem (x : ℝ) : 15 * x = x + 196 → 15 * x = 210 := by
  sorry

end number_multiplication_problem_l4120_412075


namespace shortest_distance_theorem_l4120_412031

theorem shortest_distance_theorem (a b c : ℝ) :
  a = 8 ∧ b = 6 ∧ c^2 = a^2 + b^2 → c = 10 := by
  sorry

end shortest_distance_theorem_l4120_412031


namespace chubby_checkerboard_black_squares_l4120_412093

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  rows : Nat
  cols : Nat

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  ((board.cols + 1) / 2) * board.rows

/-- Theorem: A 31x29 checkerboard has 465 black squares -/
theorem chubby_checkerboard_black_squares :
  let board : Checkerboard := ⟨31, 29⟩
  count_black_squares board = 465 := by
  sorry

#eval count_black_squares ⟨31, 29⟩

end chubby_checkerboard_black_squares_l4120_412093


namespace carpet_ratio_l4120_412029

theorem carpet_ratio (house1 house2 house3 total : ℕ) 
  (h1 : house1 = 12)
  (h2 : house2 = 20)
  (h3 : house3 = 10)
  (h_total : total = 62)
  (h_sum : house1 + house2 + house3 + (total - (house1 + house2 + house3)) = total) :
  (total - (house1 + house2 + house3)) / house3 = 2 := by
sorry

end carpet_ratio_l4120_412029


namespace middle_number_in_ratio_l4120_412053

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 3 / 2 ∧ 
  b / c = 2 / 5 ∧ 
  a^2 + b^2 + c^2 = 1862 → 
  b = 14 := by
sorry

end middle_number_in_ratio_l4120_412053


namespace rotate_from_one_to_six_l4120_412094

/-- Represents a face of a standard six-sided die -/
inductive DieFace
| one
| two
| three
| four
| five
| six

/-- Represents the state of a die with visible faces -/
structure DieState where
  top : DieFace
  front : DieFace
  right : DieFace

/-- Defines the opposite face relation for a standard die -/
def opposite_face (f : DieFace) : DieFace :=
  match f with
  | DieFace.one => DieFace.six
  | DieFace.two => DieFace.five
  | DieFace.three => DieFace.four
  | DieFace.four => DieFace.three
  | DieFace.five => DieFace.two
  | DieFace.six => DieFace.one

/-- Simulates a 90° clockwise rotation of the die -/
def rotate_90_clockwise (s : DieState) : DieState :=
  { top := s.right
  , front := s.top
  , right := opposite_face s.front }

/-- Theorem: After a 90° clockwise rotation from a state where 1 is visible,
    the opposite face (6) becomes visible -/
theorem rotate_from_one_to_six (initial : DieState) 
    (h : initial.top = DieFace.one) : 
    ∃ (rotated : DieState), rotated = rotate_90_clockwise initial ∧ 
    (rotated.top = DieFace.six ∨ rotated.front = DieFace.six ∨ rotated.right = DieFace.six) :=
  sorry


end rotate_from_one_to_six_l4120_412094


namespace vectors_parallel_iff_l4120_412004

def a (m : ℝ) : Fin 2 → ℝ := ![1, m + 1]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

theorem vectors_parallel_iff (m : ℝ) :
  (∃ (k : ℝ), a m = k • b m) ↔ m = -2 ∨ m = 1 := by
  sorry

end vectors_parallel_iff_l4120_412004


namespace polynomial_simplification_l4120_412042

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x + 11) - (x^6 + 2 * x^5 - 2 * x^4 + x^3 + 15) =
  x^6 - x^5 + 5 * x^4 - x^3 + x - 4 := by
  sorry

end polynomial_simplification_l4120_412042


namespace correct_average_weight_l4120_412028

/-- Given a class of 20 boys with an initial average weight and a misread weight,
    calculate the correct average weight. -/
theorem correct_average_weight
  (num_boys : ℕ)
  (initial_avg : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : num_boys = 20)
  (h2 : initial_avg = 58.4)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 62) :
  (num_boys : ℝ) * initial_avg + (correct_weight - misread_weight) = num_boys * 58.7 :=
by sorry

#check correct_average_weight

end correct_average_weight_l4120_412028


namespace divisibility_by_six_l4120_412015

theorem divisibility_by_six (n : ℕ) : 6 ∣ (n^3 - 7*n) := by
  sorry

end divisibility_by_six_l4120_412015


namespace smallest_coprime_to_180_l4120_412099

theorem smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ x < 7 → Nat.gcd x 180 ≠ 1 ∧ Nat.gcd 7 180 = 1 := by
  sorry

end smallest_coprime_to_180_l4120_412099


namespace fraction_equality_implies_numerator_equality_l4120_412081

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (hc : c ≠ 0) :
  a / c = b / c → a = b :=
by sorry

end fraction_equality_implies_numerator_equality_l4120_412081


namespace negative_fifteen_inequality_l4120_412096

theorem negative_fifteen_inequality (a b : ℝ) (h : a > b) : -15 * a < -15 * b := by
  sorry

end negative_fifteen_inequality_l4120_412096


namespace four_numbers_with_consecutive_sums_l4120_412005

theorem four_numbers_with_consecutive_sums : ∃ (a b c d : ℕ),
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) ∧
  (a + b = 2023) ∧
  (a + c = 2024) ∧
  (a + d = 2026) ∧
  (b + c = 2025) ∧
  (b + d = 2027) ∧
  (c + d = 2028) :=
by sorry

end four_numbers_with_consecutive_sums_l4120_412005


namespace fraction_simplification_l4120_412073

theorem fraction_simplification :
  (5 : ℝ) / (2 * Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 38 := by
  sorry

end fraction_simplification_l4120_412073


namespace labourer_income_l4120_412079

/-- Prove that the monthly income of a labourer is 75 --/
theorem labourer_income :
  ∀ (avg_expenditure_6m : ℝ) (debt : ℝ) (expenditure_4m : ℝ) (savings : ℝ),
    avg_expenditure_6m = 80 →
    debt > 0 →
    expenditure_4m = 60 →
    savings = 30 →
    ∃ (income : ℝ),
      income * 6 - debt + income * 4 = avg_expenditure_6m * 6 + expenditure_4m * 4 + debt + savings ∧
      income = 75 := by
  sorry

end labourer_income_l4120_412079


namespace words_with_vowels_l4120_412040

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels :
  total_words - words_without_vowels = 6752 := by sorry

end words_with_vowels_l4120_412040


namespace power_product_equals_sum_of_exponents_l4120_412019

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_product_equals_sum_of_exponents_l4120_412019


namespace emma_milk_containers_l4120_412062

/-- The number of weeks Emma buys milk -/
def weeks : ℕ := 3

/-- The number of school days in a week -/
def school_days_per_week : ℕ := 5

/-- The total number of milk containers Emma buys in 3 weeks -/
def total_containers : ℕ := 30

/-- The number of containers Emma buys each school day -/
def containers_per_day : ℚ := total_containers / (weeks * school_days_per_week)

theorem emma_milk_containers : containers_per_day = 2 := by
  sorry

end emma_milk_containers_l4120_412062


namespace train_distance_theorem_l4120_412038

-- Define the speeds of the trains
def speed_train1 : ℝ := 20
def speed_train2 : ℝ := 25

-- Define the difference in distance traveled
def distance_difference : ℝ := 50

-- Define the theorem
theorem train_distance_theorem :
  ∀ (t : ℝ), -- t represents the time taken for trains to meet
  t > 0 → -- time is positive
  speed_train1 * t + speed_train2 * t = -- total distance is sum of distances traveled by both trains
  speed_train1 * t + (speed_train1 * t + distance_difference) → -- one train travels 50 km more
  speed_train1 * t + (speed_train1 * t + distance_difference) = 450 -- total distance is 450 km
  := by sorry

end train_distance_theorem_l4120_412038


namespace intersection_of_A_and_B_l4120_412090

def A : Set ℤ := {x | x^2 + x - 6 ≤ 0}
def B : Set ℤ := {x | x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l4120_412090


namespace angle_A_is_60_l4120_412049

-- Define the triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the conditions
axiom acute_triangle : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90
axiom side_a : a = 2 * Real.sqrt 3
axiom side_b : b = 2 * Real.sqrt 2
axiom angle_B : B = 45

-- Theorem to prove
theorem angle_A_is_60 : A = 60 := by
  sorry

end angle_A_is_60_l4120_412049


namespace chess_tournament_ordering_l4120_412095

/-- A structure representing a chess tournament -/
structure ChessTournament (N : ℕ) where
  beats : Fin N → Fin N → Prop

/-- The tournament property described in the problem -/
def has_tournament_property {N : ℕ} (M : ℕ) (t : ChessTournament N) : Prop :=
  ∀ (players : Fin (M + 1) → Fin N),
    (∀ i : Fin M, t.beats (players i) (players (i + 1))) →
    t.beats (players 0) (players M)

/-- The theorem to be proved -/
theorem chess_tournament_ordering
  {N M : ℕ} (h_N : N > M) (h_M : M > 1)
  (t : ChessTournament N)
  (h_prop : has_tournament_property M t) :
  ∃ f : Fin N ≃ Fin N,
    ∀ a b : Fin N, (a : ℕ) ≥ (b : ℕ) + M - 1 → t.beats (f a) (f b) :=
sorry

end chess_tournament_ordering_l4120_412095


namespace decimal_representation_of_225_999_l4120_412078

theorem decimal_representation_of_225_999 :
  ∃ (d : ℕ → ℕ), 
    (∀ n, d n < 10) ∧ 
    (∀ n, d (n + 3) = d n) ∧
    (d 0 = 2 ∧ d 1 = 2 ∧ d 2 = 5) ∧
    (d 80 = 5) ∧
    (225 : ℚ) / 999 = ∑' n, (d n : ℚ) / 10 ^ (n + 1) := by
  sorry

end decimal_representation_of_225_999_l4120_412078
