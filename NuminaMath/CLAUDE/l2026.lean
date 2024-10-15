import Mathlib

namespace NUMINAMATH_CALUDE_winning_sequence_exists_l2026_202632

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

def last_digit (n : ℕ) : ℕ := n % 10

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length > 0 ∧
  ∀ n ∈ seq, is_prime n ∧ n ≤ 100 ∧
  ∀ i < seq.length - 1, last_digit (seq.get ⟨i, by sorry⟩) = first_digit (seq.get ⟨i+1, by sorry⟩) ∧
  ∀ i j, i ≠ j → seq.get ⟨i, by sorry⟩ ≠ seq.get ⟨j, by sorry⟩

theorem winning_sequence_exists :
  ∃ seq : List ℕ, valid_sequence seq ∧ seq.length = 3 ∧
  ∀ p : ℕ, is_prime p → p ≤ 100 → p ∉ seq →
    (seq.length > 0 → first_digit p ≠ last_digit (seq.getLast (by sorry))) :=
sorry

end NUMINAMATH_CALUDE_winning_sequence_exists_l2026_202632


namespace NUMINAMATH_CALUDE_only_lottery_is_random_l2026_202698

-- Define the events
inductive Event
| BasketballFall
| LotteryWin
| BirthdayMatch
| DrawBlackBall

-- Define the properties of events
def isCertain (e : Event) : Prop :=
  match e with
  | Event.BasketballFall => true
  | _ => false

def isImpossible (e : Event) : Prop :=
  match e with
  | Event.DrawBlackBall => true
  | _ => false

def isRandom (e : Event) : Prop :=
  ¬(isCertain e) ∧ ¬(isImpossible e)

-- Define the given conditions
axiom gravity_exists : isCertain Event.BasketballFall
axiom pigeonhole_principle : isCertain Event.BirthdayMatch
axiom bag_contents : isImpossible Event.DrawBlackBall

-- State the theorem
theorem only_lottery_is_random :
  ∀ e : Event, isRandom e ↔ e = Event.LotteryWin :=
sorry

end NUMINAMATH_CALUDE_only_lottery_is_random_l2026_202698


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2026_202684

/-- The radius of a circle inscribed in a rectangle and tangent to four circles -/
theorem inscribed_circle_radius (AB BC : ℝ) (h_AB : AB = 8) (h_BC : BC = 6) : ∃ r : ℝ,
  r > 0 ∧ r < 6 ∧
  (r + 4)^2 = r^2 + r^2 ∧
  (r + 3)^2 = (8 - r)^2 + r^2 ∧
  r = 11 - Real.sqrt 66 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2026_202684


namespace NUMINAMATH_CALUDE_range_of_a_l2026_202631

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 5*a| + |2*x + 1|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  a ≥ 0.4 ∨ a ≤ -0.8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2026_202631


namespace NUMINAMATH_CALUDE_volleyball_team_scoring_l2026_202626

/-- Volleyball team scoring problem -/
theorem volleyball_team_scoring 
  (lizzie_score : ℕ) 
  (nathalie_score : ℕ) 
  (aimee_score : ℕ) 
  (team_score : ℕ) 
  (h1 : lizzie_score = 4)
  (h2 : nathalie_score = lizzie_score + 3)
  (h3 : aimee_score = 2 * (lizzie_score + nathalie_score))
  (h4 : team_score = 50) :
  team_score - (lizzie_score + nathalie_score + aimee_score) = 17 := by
  sorry


end NUMINAMATH_CALUDE_volleyball_team_scoring_l2026_202626


namespace NUMINAMATH_CALUDE_gcd_79625_51575_l2026_202692

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_79625_51575_l2026_202692


namespace NUMINAMATH_CALUDE_amoeba_population_after_five_days_l2026_202613

/-- The number of amoebas after n days, given an initial population and daily split rate --/
def amoeba_population (initial_population : ℕ) (split_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * split_rate ^ days

/-- The theorem stating that after 5 days, the amoeba population will be 486 --/
theorem amoeba_population_after_five_days :
  amoeba_population 2 3 5 = 486 := by
  sorry

#eval amoeba_population 2 3 5

end NUMINAMATH_CALUDE_amoeba_population_after_five_days_l2026_202613


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_l2026_202669

def A (n : ℕ) : ℕ := (List.range n).foldl (λ acc k => acc * Nat.choose (k^2) k) 1

theorem smallest_n_divisible_by_ten : 
  (∀ m < 4, ¬(10 ∣ A m)) ∧ (10 ∣ A 4) := by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_l2026_202669


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l2026_202655

theorem pencil_buyers_difference : ∀ (pencil_cost : ℕ) 
  (eighth_graders fifth_graders : ℕ),
  pencil_cost > 0 ∧
  pencil_cost * eighth_graders = 234 ∧
  pencil_cost * fifth_graders = 285 ∧
  fifth_graders ≤ 25 →
  fifth_graders - eighth_graders = 17 := by
sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l2026_202655


namespace NUMINAMATH_CALUDE_max_k_value_l2026_202687

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a*x + 2 * log x

noncomputable def g (k : ℤ) (x : ℝ) : ℝ := (1/2) * x^2 + k*x + (2-x) * log x - k

theorem max_k_value :
  ∃ (k_max : ℤ),
    (∀ (k : ℤ), (∀ (x : ℝ), x > 1 → g k x < f 1 x) → k ≤ k_max) ∧
    (∀ (x : ℝ), x > 1 → g k_max x < f 1 x) ∧
    k_max = 3 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2026_202687


namespace NUMINAMATH_CALUDE_largest_two_digit_divisor_l2026_202615

def a : ℕ := 2^5 * 3^3 * 5^2 * 7

theorem largest_two_digit_divisor :
  (∀ d : ℕ, d > 96 → d < 100 → ¬(d ∣ a)) ∧ (96 ∣ a) := by sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisor_l2026_202615


namespace NUMINAMATH_CALUDE_arithmetic_sum_l2026_202661

theorem arithmetic_sum : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l2026_202661


namespace NUMINAMATH_CALUDE_product_ratio_equality_l2026_202664

theorem product_ratio_equality (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_equality_l2026_202664


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2026_202653

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2026_202653


namespace NUMINAMATH_CALUDE_sally_bread_consumption_l2026_202623

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The number of bread pieces used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of bread pieces Sally eats across Saturday and Sunday -/
def total_bread_pieces : ℕ := (saturday_sandwiches * bread_per_sandwich) + (sunday_sandwiches * bread_per_sandwich)

theorem sally_bread_consumption :
  total_bread_pieces = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_bread_consumption_l2026_202623


namespace NUMINAMATH_CALUDE_scout_troop_profit_is_480_l2026_202640

/-- Calculates the profit of a scout troop selling candy bars -/
def scout_troop_profit (total_bars : ℕ) (cost_per_six : ℚ) (discount_rate : ℚ)
  (price_first_tier : ℚ) (price_second_tier : ℚ) (first_tier_limit : ℕ) : ℚ :=
  let cost_per_bar := cost_per_six / 6
  let total_cost := total_bars * cost_per_bar
  let discounted_cost := total_cost * (1 - discount_rate)
  let revenue_first_tier := min first_tier_limit total_bars * price_first_tier
  let revenue_second_tier := max 0 (total_bars - first_tier_limit) * price_second_tier
  let total_revenue := revenue_first_tier + revenue_second_tier
  total_revenue - discounted_cost

theorem scout_troop_profit_is_480 :
  scout_troop_profit 1200 3 (5/100) 1 (3/4) 600 = 480 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_is_480_l2026_202640


namespace NUMINAMATH_CALUDE_max_product_sum_22_l2026_202652

/-- A list of distinct natural numbers -/
def DistinctNatList := List Nat

/-- Check if a list contains distinct elements -/
def isDistinct (l : List Nat) : Prop :=
  l.Nodup

/-- Sum of elements in a list -/
def listSum (l : List Nat) : Nat :=
  l.sum

/-- Product of elements in a list -/
def listProduct (l : List Nat) : Nat :=
  l.prod

/-- The maximum product of distinct natural numbers that sum to 22 -/
def maxProductSum22 : Nat :=
  1008

theorem max_product_sum_22 :
  ∀ (l : DistinctNatList), 
    isDistinct l → 
    listSum l = 22 → 
    listProduct l ≤ maxProductSum22 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_22_l2026_202652


namespace NUMINAMATH_CALUDE_equation_solution_exists_l2026_202656

theorem equation_solution_exists : ∃ (a b c d e : ℕ), 
  a ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  b ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  d ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  e ∈ ({1, 2, 3, 5, 6} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  (a + b - c) * d / e = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l2026_202656


namespace NUMINAMATH_CALUDE_middle_number_proof_l2026_202683

theorem middle_number_proof (x y z : ℕ) (h1 : x < y) (h2 : y < z) 
  (h3 : x + y = 16) (h4 : x + z = 21) (h5 : y + z = 23) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2026_202683


namespace NUMINAMATH_CALUDE_product_of_sums_of_squares_l2026_202650

theorem product_of_sums_of_squares (a b c d : ℤ) :
  ∃ x y : ℤ, (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_of_squares_l2026_202650


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2026_202646

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 11) ∣ (n^4 + 119) ∧ 
  ∀ (m : ℕ), m > n → ¬((m + 11) ∣ (m^4 + 119)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2026_202646


namespace NUMINAMATH_CALUDE_original_salary_calculation_l2026_202639

/-- Proves that if a salary S is increased by 2% to result in €10,200, then S equals €10,000. -/
theorem original_salary_calculation (S : ℝ) : S * 1.02 = 10200 → S = 10000 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_calculation_l2026_202639


namespace NUMINAMATH_CALUDE_max_salary_cricket_team_l2026_202689

/-- Represents a cricket team -/
structure CricketTeam where
  players : ℕ
  minSalary : ℕ
  salaryCap : ℕ

/-- Calculates the maximum possible salary for the highest-paid player in a cricket team -/
def maxSalary (team : CricketTeam) : ℕ :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem: The maximum possible salary for the highest-paid player in the given cricket team is 416000 -/
theorem max_salary_cricket_team :
  ∃ (team : CricketTeam),
    team.players = 18 ∧
    team.minSalary = 12000 ∧
    team.salaryCap = 620000 ∧
    maxSalary team = 416000 := by
  sorry

end NUMINAMATH_CALUDE_max_salary_cricket_team_l2026_202689


namespace NUMINAMATH_CALUDE_guest_lecturer_fee_l2026_202624

theorem guest_lecturer_fee (B : Nat) (h1 : B < 10) (h2 : (200 + 10 * B + 9) % 13 = 0) : B = 0 := by
  sorry

end NUMINAMATH_CALUDE_guest_lecturer_fee_l2026_202624


namespace NUMINAMATH_CALUDE_corresponds_to_zero_one_l2026_202603

/-- A mapping f from A to B where (x, y) in A corresponds to (x-1, 3-y) in B -/
def f (x y : ℝ) : ℝ × ℝ := (x - 1, 3 - y)

/-- Theorem stating that (1, 2) in A corresponds to (0, 1) in B under mapping f -/
theorem corresponds_to_zero_one : f 1 2 = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_corresponds_to_zero_one_l2026_202603


namespace NUMINAMATH_CALUDE_new_device_improvement_l2026_202610

/-- Represents the data for a device's products -/
structure DeviceData where
  mean : ℝ
  variance : ℝ

/-- Criterion for significant improvement -/
def significant_improvement (old new : DeviceData) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

/-- Theorem stating that the new device shows significant improvement -/
theorem new_device_improvement (old new : DeviceData)
  (h_old : old.mean = 10 ∧ old.variance = 0.036)
  (h_new : new.mean = 10.3 ∧ new.variance = 0.04) :
  significant_improvement old new :=
by sorry

end NUMINAMATH_CALUDE_new_device_improvement_l2026_202610


namespace NUMINAMATH_CALUDE_eleven_divides_difference_l2026_202670

/-- Represents a three-digit number ABC where A, B, and C are distinct digits and A ≠ 0 -/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C :Nat
  h1 : A ≠ 0
  h2 : A < 10
  h3 : B < 10
  h4 : C < 10
  h5 : A ≠ B
  h6 : B ≠ C
  h7 : A ≠ C

/-- Converts a ThreeDigitNumber to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

theorem eleven_divides_difference (n : ThreeDigitNumber) :
  11 ∣ (toNumber n - reverse n) := by
  sorry

#check eleven_divides_difference

end NUMINAMATH_CALUDE_eleven_divides_difference_l2026_202670


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l2026_202694

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/2 →
  diameter = 18 →
  num_pieces = 16 →
  (π * (diameter/2)^2 * thickness) / num_pieces = 2.53125 * π := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l2026_202694


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coefficient_quadratic_l2026_202619

theorem no_rational_solution_for_odd_coefficient_quadratic
  (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coefficient_quadratic_l2026_202619


namespace NUMINAMATH_CALUDE_problem_statement_l2026_202616

theorem problem_statement (a b x y : ℕ+) (P : ℕ) 
  (h1 : ∃ k : ℕ, (a * x + b * y : ℕ) = k * (a^2 + b^2))
  (h2 : P = x^2 + y^2)
  (h3 : Nat.Prime P) :
  (P ∣ (a^2 + b^2 : ℕ)) ∧ (a = x ∧ b = y) := by
sorry


end NUMINAMATH_CALUDE_problem_statement_l2026_202616


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l2026_202686

/-- A quadratic function passing through the origin -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

/-- The theorem stating that if the quadratic function passes through the origin, then m = -4 -/
theorem quadratic_through_origin (m : ℝ) :
  (∀ x, quadratic_function m x = 0 → x = 0) →
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l2026_202686


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l2026_202630

theorem square_sum_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l2026_202630


namespace NUMINAMATH_CALUDE_shirts_not_washed_l2026_202645

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 27 → washed = 20 → 
  short_sleeve + long_sleeve - washed = 16 := by
  sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l2026_202645


namespace NUMINAMATH_CALUDE_decagon_triangles_l2026_202679

/-- A regular decagon is a polygon with 10 vertices -/
def RegularDecagon : ℕ := 10

/-- No three vertices of a regular decagon are collinear -/
axiom decagon_vertices_not_collinear : True

/-- The number of triangles formed from vertices of a regular decagon -/
def num_triangles_from_decagon : ℕ := Nat.choose RegularDecagon 3

theorem decagon_triangles :
  num_triangles_from_decagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l2026_202679


namespace NUMINAMATH_CALUDE_division_with_remainder_l2026_202691

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 145 * q + r ∧ 0 ≤ r ∧ r < 145 ∧ r = 67 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2026_202691


namespace NUMINAMATH_CALUDE_scenario_one_scenario_two_scenario_three_scenario_four_l2026_202643

-- Define the probabilities for A and B
def prob_A : ℚ := 2/3
def prob_B : ℚ := 3/4

-- Define the complementary probabilities
def miss_A : ℚ := 1 - prob_A
def miss_B : ℚ := 1 - prob_B

-- Theorem 1
theorem scenario_one : 
  1 - prob_A^3 = 19/27 := by sorry

-- Theorem 2
theorem scenario_two :
  2 * prob_A^2 * miss_A * 2 * prob_B * miss_B = 1/6 := by sorry

-- Theorem 3
theorem scenario_three :
  miss_A^2 * prob_B^2 = 1/16 := by sorry

-- Theorem 4
theorem scenario_four :
  2 * prob_A * miss_A * 2 * prob_B * miss_B = 1/6 := by sorry

end NUMINAMATH_CALUDE_scenario_one_scenario_two_scenario_three_scenario_four_l2026_202643


namespace NUMINAMATH_CALUDE_third_beats_seventh_l2026_202642

/-- Represents a chess tournament with 8 players -/
structure ChessTournament where
  /-- List of player scores in descending order -/
  scores : List ℕ
  /-- Ensure there are exactly 8 scores -/
  score_count : scores.length = 8
  /-- Ensure all scores are different -/
  distinct_scores : scores.Nodup
  /-- Second place score equals sum of last four scores -/
  second_place_condition : scores[1]! = scores[4]! + scores[5]! + scores[6]! + scores[7]!

/-- Represents the result of a game between two players -/
inductive GameResult
  | Win
  | Loss

/-- Function to determine the game result between two players based on their positions -/
def gameResult (t : ChessTournament) (player1 : Fin 8) (player2 : Fin 8) : GameResult :=
  if player1 < player2 then GameResult.Win else GameResult.Loss

theorem third_beats_seventh (t : ChessTournament) :
  gameResult t 2 6 = GameResult.Win :=
sorry

end NUMINAMATH_CALUDE_third_beats_seventh_l2026_202642


namespace NUMINAMATH_CALUDE_total_eyes_l2026_202633

/-- The total number of eyes given the number of boys, girls, cats, and spiders -/
theorem total_eyes (boys girls cats spiders : ℕ) : 
  boys = 23 → 
  girls = 18 → 
  cats = 10 → 
  spiders = 5 → 
  boys * 2 + girls * 2 + cats * 2 + spiders * 8 = 142 := by
  sorry


end NUMINAMATH_CALUDE_total_eyes_l2026_202633


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2026_202673

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 8 / Real.sqrt 11) =
  2 * Real.sqrt 462 / 77 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2026_202673


namespace NUMINAMATH_CALUDE_chord_length_l2026_202625

/-- The length of the chord cut by a line on a circle in polar coordinates -/
theorem chord_length (ρ θ : ℝ) : 
  (ρ * Real.cos θ = 1/2) →  -- Line equation
  (ρ = 2 * Real.cos θ) →    -- Circle equation
  ∃ (chord_length : ℝ), chord_length = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_chord_length_l2026_202625


namespace NUMINAMATH_CALUDE_systematic_sampling_selection_l2026_202662

theorem systematic_sampling_selection (total_rooms : Nat) (sample_size : Nat) (first_room : Nat) : 
  total_rooms = 64 → 
  sample_size = 8 → 
  first_room = 5 → 
  ∃ k : Nat, k < sample_size ∧ (first_room + k * (total_rooms / sample_size)) % total_rooms = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_selection_l2026_202662


namespace NUMINAMATH_CALUDE_sin_150_degrees_l2026_202654

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l2026_202654


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2026_202621

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (n : ℕ) : ℚ := n / 99

/-- The sum of 0.5̅ and 0.07̅ is equal to 62/99 -/
theorem sum_of_repeating_decimals : 
  SingleDigitRepeatingDecimal 5 + TwoDigitRepeatingDecimal 7 = 62 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2026_202621


namespace NUMINAMATH_CALUDE_intersection_set_complement_l2026_202622

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x > 0}

theorem intersection_set_complement : A ∩ (𝒰 \ B) = {x | -1 ≤ x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_complement_l2026_202622


namespace NUMINAMATH_CALUDE_hall_length_is_18_l2026_202648

/-- Represents the dimensions of a rectangular hall -/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the hall dimensions satisfy the given conditions -/
def satisfiesConditions (d : HallDimensions) : Prop :=
  d.width = 9 ∧
  2 * (d.length * d.width) = 2 * (d.length * d.height + d.width * d.height) ∧
  d.length * d.width * d.height = 972

theorem hall_length_is_18 :
  ∃ (d : HallDimensions), satisfiesConditions d ∧ d.length = 18 :=
by sorry

end NUMINAMATH_CALUDE_hall_length_is_18_l2026_202648


namespace NUMINAMATH_CALUDE_initial_charge_calculation_l2026_202668

/-- A taxi company's pricing model -/
structure TaxiPricing where
  initial_charge : ℝ  -- Charge for the first 1/5 mile
  additional_charge : ℝ  -- Charge for each additional 1/5 mile
  total_charge : ℝ  -- Total charge for a specific ride
  ride_distance : ℝ  -- Distance of the ride in miles

/-- Theorem stating the initial charge for the first 1/5 mile -/
theorem initial_charge_calculation (tp : TaxiPricing) 
  (h1 : tp.additional_charge = 0.40)
  (h2 : tp.total_charge = 18.40)
  (h3 : tp.ride_distance = 8) :
  tp.initial_charge = 2.80 := by
  sorry

end NUMINAMATH_CALUDE_initial_charge_calculation_l2026_202668


namespace NUMINAMATH_CALUDE_marching_band_members_l2026_202641

theorem marching_band_members : ∃ n : ℕ,
  150 < n ∧ n < 250 ∧
  n % 3 = 1 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  (∀ m : ℕ, 150 < m ∧ m < n →
    ¬(m % 3 = 1 ∧ m % 6 = 2 ∧ m % 8 = 3)) ∧
  n = 203 :=
by sorry

end NUMINAMATH_CALUDE_marching_band_members_l2026_202641


namespace NUMINAMATH_CALUDE_solve_system_l2026_202690

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - y = 7) 
  (eq2 : x + 3 * y = 2) : 
  x = 23 / 10 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2026_202690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2026_202695

/-- An arithmetic sequence with sum Sn for first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of the arithmetic sequence is -2 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
  (h1 : seq.S 3 = 6)
  (h2 : seq.a 3 = 0) : 
  seq.d = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2026_202695


namespace NUMINAMATH_CALUDE_min_group_size_l2026_202618

/-- Represents the number of men in a group with various attributes -/
structure MenGroup where
  total : ℕ
  married : ℕ
  hasTV : ℕ
  hasRadio : ℕ
  hasAC : ℕ
  hasAll : ℕ

/-- The minimum number of men in the group is at least the maximum of any single category -/
theorem min_group_size (g : MenGroup) 
  (h1 : g.married = 81)
  (h2 : g.hasTV = 75)
  (h3 : g.hasRadio = 85)
  (h4 : g.hasAC = 70)
  (h5 : g.hasAll = 11)
  : g.total ≥ 85 := by
  sorry

#check min_group_size

end NUMINAMATH_CALUDE_min_group_size_l2026_202618


namespace NUMINAMATH_CALUDE_plane_equation_l2026_202680

/-- Given a parametric equation of a plane, prove its Cartesian equation. -/
theorem plane_equation (s t : ℝ) :
  let u : ℝ × ℝ × ℝ := (2 + 2*s - 3*t, 1 + s, 4 - s + t)
  ∃ (A B C D : ℤ), A > 0 ∧ 
    (∀ (x y z : ℝ), (x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ s t : ℝ, p = u} ↔ A*x + B*y + C*z + D = 0) ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 1 ∧ B = 4 ∧ C = 3 ∧ D = -18 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l2026_202680


namespace NUMINAMATH_CALUDE_faster_person_speed_l2026_202614

/-- Given two towns 45 km apart and two people traveling towards each other,
    where one person travels 1 km/h faster than the other and they meet after 5 hours,
    prove that the faster person's speed is 5 km/h. -/
theorem faster_person_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) :
  distance = 45 →
  time = 5 →
  speed_diff = 1 →
  ∃ (speed_slower : ℝ),
    speed_slower > 0 ∧
    speed_slower * time + (speed_slower + speed_diff) * time = distance ∧
    speed_slower + speed_diff = 5 := by
  sorry

end NUMINAMATH_CALUDE_faster_person_speed_l2026_202614


namespace NUMINAMATH_CALUDE_marble_distribution_correct_group_size_l2026_202601

/-- The number of marbles in the jar -/
def total_marbles : ℕ := 500

/-- The number of additional people that would join the group -/
def additional_people : ℕ := 5

/-- The number of marbles each person would receive less if additional people joined -/
def marbles_less : ℕ := 2

/-- The number of people in the group today -/
def group_size : ℕ := 33

theorem marble_distribution :
  (total_marbles = group_size * (total_marbles / group_size)) ∧
  (total_marbles = (group_size + additional_people) * (total_marbles / group_size - marbles_less)) :=
by sorry

theorem correct_group_size : group_size = 33 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_correct_group_size_l2026_202601


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2026_202658

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareArrangement where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The conditions of the arrangement -/
def valid_arrangement (a : RectangleSquareArrangement) : Prop :=
  -- The outer square's side length is 3 times the inner square's side length
  a.inner_square_side + 2 * a.rectangle_short_side = 3 * a.inner_square_side ∧
  -- The outer square's side length is also the sum of the long side and short side
  a.rectangle_long_side + a.rectangle_short_side = 3 * a.inner_square_side

/-- The theorem to be proved -/
theorem rectangle_ratio_is_two (a : RectangleSquareArrangement) 
    (h : valid_arrangement a) : 
    a.rectangle_long_side / a.rectangle_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2026_202658


namespace NUMINAMATH_CALUDE_right_triangle_leg_construction_l2026_202638

theorem right_triangle_leg_construction (c m : ℝ) (h_positive : c > 0) :
  ∃ (a b p : ℝ),
    a > 0 ∧ b > 0 ∧ p > 0 ∧
    a^2 + b^2 = c^2 ∧
    a^2 - b^2 = 4 * m^2 ∧
    p = (c * (1 + Real.sqrt 5)) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_construction_l2026_202638


namespace NUMINAMATH_CALUDE_old_clock_slower_l2026_202677

/-- Represents the number of minutes between hand overlaps on the old clock -/
def overlap_interval : ℕ := 66

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the number of hand overlaps in a 24-hour period -/
def overlaps_per_day : ℕ := hours_per_day - 2

/-- Calculates the total minutes on the old clock for 24 hours -/
def old_clock_minutes : ℕ := overlaps_per_day * overlap_interval

/-- Calculates the total minutes in a standard 24-hour period -/
def standard_clock_minutes : ℕ := hours_per_day * minutes_per_hour

/-- Theorem stating that the old clock is 12 minutes slower over 24 hours -/
theorem old_clock_slower :
  old_clock_minutes - standard_clock_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_old_clock_slower_l2026_202677


namespace NUMINAMATH_CALUDE_current_speed_l2026_202672

/-- Proves that the speed of the current is 8.5 kmph given the specified conditions -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  rowing_speed = 9.5 →
  distance = 45.5 →
  time = 9.099272058235341 →
  let downstream_speed := distance / 1000 / (time / 3600)
  downstream_speed = rowing_speed + 8.5 := by
sorry

end NUMINAMATH_CALUDE_current_speed_l2026_202672


namespace NUMINAMATH_CALUDE_fred_seashells_l2026_202647

/-- The number of seashells Fred found initially -/
def initial_seashells : ℝ := 47.5

/-- The number of seashells Fred gave to Jessica -/
def given_seashells : ℝ := 25.3

/-- The number of seashells Fred has now -/
def remaining_seashells : ℝ := initial_seashells - given_seashells

theorem fred_seashells : remaining_seashells = 22.2 := by sorry

end NUMINAMATH_CALUDE_fred_seashells_l2026_202647


namespace NUMINAMATH_CALUDE_complex_fourth_root_of_negative_sixteen_l2026_202617

theorem complex_fourth_root_of_negative_sixteen :
  let solutions := {z : ℂ | z^4 = -16 ∧ z.im ≥ 0}
  solutions = {Complex.mk (Real.sqrt 2) (Real.sqrt 2), Complex.mk (-Real.sqrt 2) (-Real.sqrt 2)} := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_root_of_negative_sixteen_l2026_202617


namespace NUMINAMATH_CALUDE_counterexample_exists_l2026_202674

theorem counterexample_exists : ∃ n : ℕ+, ¬(Nat.Prime (6 * n - 1)) ∧ ¬(Nat.Prime (6 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2026_202674


namespace NUMINAMATH_CALUDE_no_real_roots_l2026_202605

theorem no_real_roots : ¬∃ (x : ℝ), x + Real.sqrt (2*x - 6) = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2026_202605


namespace NUMINAMATH_CALUDE_aunt_age_proof_l2026_202612

def cori_age_today : ℕ := 3
def years_until_comparison : ℕ := 5

def aunt_age_today : ℕ := 19

theorem aunt_age_proof :
  (cori_age_today + years_until_comparison) * 3 = aunt_age_today + years_until_comparison :=
by sorry

end NUMINAMATH_CALUDE_aunt_age_proof_l2026_202612


namespace NUMINAMATH_CALUDE_grain_remaining_after_crash_l2026_202663

/-- The amount of grain remaining onboard after a ship crash -/
def remaining_grain (original : ℕ) (spilled : ℕ) : ℕ :=
  original - spilled

/-- Theorem stating the amount of grain remaining onboard after the specific crash -/
theorem grain_remaining_after_crash : 
  remaining_grain 50870 49952 = 918 := by
  sorry

end NUMINAMATH_CALUDE_grain_remaining_after_crash_l2026_202663


namespace NUMINAMATH_CALUDE_recipe_butter_amount_l2026_202665

/-- The amount of butter (in ounces) required per cup of baking mix -/
def butter_per_cup : ℚ := 4/3

/-- The number of cups of baking mix the chef planned to use -/
def planned_cups : ℕ := 6

/-- The amount of coconut oil (in ounces) the chef used as a substitute for butter -/
def coconut_oil_used : ℕ := 8

/-- Theorem stating that the recipe calls for 4/3 ounces of butter per cup of baking mix -/
theorem recipe_butter_amount :
  butter_per_cup * planned_cups = coconut_oil_used := by
  sorry

end NUMINAMATH_CALUDE_recipe_butter_amount_l2026_202665


namespace NUMINAMATH_CALUDE_apples_buyers_l2026_202666

theorem apples_buyers (men_apples : ℕ) (women_apples : ℕ) (total_apples : ℕ) :
  men_apples = 30 →
  women_apples = men_apples + 20 →
  total_apples = 210 →
  ∃ (num_men : ℕ), num_men * men_apples + 3 * women_apples = total_apples ∧ num_men = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_buyers_l2026_202666


namespace NUMINAMATH_CALUDE_f_properties_l2026_202649

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2 * Real.log x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x > 0, f a x = x^2 - a*x - a^2 * Real.log x) ∧
  (a = 1 → ∃ x > 0, ∀ y > 0, f a x ≤ f a y) ∧
  (a = 1 → ∃ x > 0, f a x = 0) ∧
  ((-2 ≤ a ∧ a ≤ 1) → ∀ x y, 1 < x ∧ x < y → f a x < f a y) ∧
  (a > 1 → ∀ x y, 1 < x ∧ x < y ∧ y < a → f a x > f a y) ∧
  (a > 1 → ∀ x y, a < x ∧ x < y → f a x < f a y) ∧
  (a < -2 → ∀ x y, 1 < x ∧ x < y ∧ y < -a/2 → f a x > f a y) ∧
  (a < -2 → ∀ x y, -a/2 < x ∧ x < y → f a x < f a y) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2026_202649


namespace NUMINAMATH_CALUDE_average_age_calculation_l2026_202629

/-- The average age of a group of fifth-graders, their parents, and teachers -/
theorem average_age_calculation (n_students : ℕ) (n_parents : ℕ) (n_teachers : ℕ)
  (avg_age_students : ℚ) (avg_age_parents : ℚ) (avg_age_teachers : ℚ)
  (h_students : n_students = 30)
  (h_parents : n_parents = 50)
  (h_teachers : n_teachers = 10)
  (h_avg_students : avg_age_students = 10)
  (h_avg_parents : avg_age_parents = 40)
  (h_avg_teachers : avg_age_teachers = 35) :
  (n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers) /
  (n_students + n_parents + n_teachers : ℚ) = 530 / 18 :=
by sorry

end NUMINAMATH_CALUDE_average_age_calculation_l2026_202629


namespace NUMINAMATH_CALUDE_not_divisible_by_11_l2026_202602

theorem not_divisible_by_11 : ¬(11 ∣ 98473092) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_11_l2026_202602


namespace NUMINAMATH_CALUDE_overlapping_part_length_l2026_202676

/-- Given three wooden planks of equal length and a total fence length,
    calculate the length of one overlapping part. -/
theorem overlapping_part_length
  (plank_length : ℝ)
  (num_planks : ℕ)
  (fence_length : ℝ)
  (h1 : plank_length = 217)
  (h2 : num_planks = 3)
  (h3 : fence_length = 627)
  (h4 : num_planks > 1) :
  let overlap_length := (num_planks * plank_length - fence_length) / (num_planks - 1)
  overlap_length = 12 := by
sorry

end NUMINAMATH_CALUDE_overlapping_part_length_l2026_202676


namespace NUMINAMATH_CALUDE_grade2_sample_count_l2026_202606

/-- Represents the number of students in a grade -/
def GradeCount := ℕ

/-- Represents a ratio of students across three grades -/
structure GradeRatio :=
  (grade1 : ℕ)
  (grade2 : ℕ)
  (grade3 : ℕ)

/-- Calculates the number of students in a stratified sample for a specific grade -/
def stratifiedSampleCount (totalSample : ℕ) (ratio : GradeRatio) (gradeRatio : ℕ) : ℕ :=
  (totalSample * gradeRatio) / (ratio.grade1 + ratio.grade2 + ratio.grade3)

/-- Theorem stating the number of Grade 2 students in the stratified sample -/
theorem grade2_sample_count 
  (totalSample : ℕ) 
  (ratio : GradeRatio) 
  (h1 : totalSample = 240) 
  (h2 : ratio = GradeRatio.mk 5 4 3) : 
  stratifiedSampleCount totalSample ratio ratio.grade2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_grade2_sample_count_l2026_202606


namespace NUMINAMATH_CALUDE_clock_equivalent_square_l2026_202681

theorem clock_equivalent_square : ∃ (n : ℕ), n > 9 ∧ n ≤ 13 ∧ (n ^ 2 - n) % 12 = 0 ∧ ∀ (m : ℕ), m > 9 ∧ m < n → (m ^ 2 - m) % 12 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_clock_equivalent_square_l2026_202681


namespace NUMINAMATH_CALUDE_triangle_lines_l2026_202608

structure Triangle where
  B : ℝ × ℝ
  altitude_AB : ℝ → ℝ → ℝ
  angle_bisector_A : ℝ → ℝ → ℝ

def line_AB (t : Triangle) : ℝ → ℝ → ℝ := fun x y => 2*x + y - 8
def line_AC (t : Triangle) : ℝ → ℝ → ℝ := fun x y => x + 2*y + 2

theorem triangle_lines (t : Triangle) 
  (hB : t.B = (3, 2))
  (hAlt : t.altitude_AB = fun x y => x - 2*y + 2)
  (hBis : t.angle_bisector_A = fun x y => x + y - 2) :
  (line_AB t = fun x y => 2*x + y - 8) ∧ 
  (line_AC t = fun x y => x + 2*y + 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_lines_l2026_202608


namespace NUMINAMATH_CALUDE_optimal_fraction_sum_l2026_202671

theorem optimal_fraction_sum (A B C D : ℕ) : 
  (A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9) →  -- A, B, C, D are digits
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →  -- A, B, C, D are different
  (C + D ≥ 5) →  -- C + D is at least 5
  (∃ k : ℕ, k * (C + D) = A + B) →  -- (A+B)/(C+D) is an integer
  (A + B ≤ 14) :=  -- The maximum possible value of A+B is 14
by sorry

end NUMINAMATH_CALUDE_optimal_fraction_sum_l2026_202671


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2026_202635

-- Define the sphere and its properties
def sphere_radius : ℝ := 13
def water_cross_section_radius : ℝ := 12
def submerged_depth : ℝ := 8

-- Theorem statement
theorem sphere_surface_area :
  (sphere_radius ^ 2 = water_cross_section_radius ^ 2 + (sphere_radius - submerged_depth) ^ 2) →
  (4 * π * sphere_radius ^ 2 = 676 * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2026_202635


namespace NUMINAMATH_CALUDE_volunteer_quota_allocation_l2026_202628

theorem volunteer_quota_allocation :
  let n : ℕ := 24  -- Total number of quotas
  let k : ℕ := 3   -- Number of venues
  let total_partitions : ℕ := Nat.choose (n - 1) (k - 1)
  let invalid_partitions : ℕ := (k - 1) * Nat.choose k 2 + 1
  total_partitions - invalid_partitions = 222 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_quota_allocation_l2026_202628


namespace NUMINAMATH_CALUDE_inequality_solution_l2026_202657

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x > 2 → y > 2 → x < y → f x > f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- Define the solution set
def solution_set (x : ℝ) := 4/3 < x ∧ x < 2

-- State the theorem
theorem inequality_solution :
  (∀ x, solution_set x ↔ f (2*x - 1) - f (x + 1) > 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2026_202657


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2026_202675

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The equation x² = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x² = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2026_202675


namespace NUMINAMATH_CALUDE_value_of_expression_l2026_202678

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x^2 + 2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2026_202678


namespace NUMINAMATH_CALUDE_jeans_business_weekly_hours_l2026_202636

/-- Represents the operating hours of a business for a single day -/
structure DailyHours where
  open_time : Nat
  close_time : Nat

/-- Calculates the number of hours a business is open in a day -/
def hours_open (dh : DailyHours) : Nat :=
  dh.close_time - dh.open_time

/-- Represents the operating hours of a business for a week -/
structure WeeklyHours where
  weekday : DailyHours
  weekend : DailyHours

/-- Calculates the total hours a business is open in a week -/
def total_weekly_hours (wh : WeeklyHours) : Nat :=
  (hours_open wh.weekday * 5) + (hours_open wh.weekend * 2)

/-- Jean's business hours -/
def jeans_business : WeeklyHours :=
  { weekday := { open_time := 16, close_time := 22 }
    weekend := { open_time := 18, close_time := 22 } }

theorem jeans_business_weekly_hours :
  total_weekly_hours jeans_business = 38 := by
  sorry

end NUMINAMATH_CALUDE_jeans_business_weekly_hours_l2026_202636


namespace NUMINAMATH_CALUDE_bg_length_is_two_l2026_202682

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 0 ∧ B.2 = Real.sqrt 12 ∧ C.1 = 2 ∧ C.2 = 0

-- Define the square BDEC
def Square (B D E C : ℝ × ℝ) : Prop :=
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2 ∧
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = (C.1 - E.1)^2 + (C.2 - E.2)^2 ∧
  (C.1 - E.1)^2 + (C.2 - E.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Define the center of the square
def CenterOfSquare (F B C : ℝ × ℝ) : Prop :=
  F.1 = (B.1 + C.1) / 2 ∧ F.2 = (B.2 + C.2) / 2

-- Define the intersection point G
def Intersection (A F B C G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G.1 = t * (F.1 - A.1) + A.1 ∧ G.2 = t * (F.2 - A.2) + A.2 ∧
  G.1 = B.1 + (C.1 - B.1) * ((G.2 - B.2) / (C.2 - B.2))

-- Main theorem
theorem bg_length_is_two 
  (A B C D E F G : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Square B D E C) 
  (h3 : CenterOfSquare F B C) 
  (h4 : Intersection A F B C G) : 
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bg_length_is_two_l2026_202682


namespace NUMINAMATH_CALUDE_car_down_payment_calculation_l2026_202627

/-- Calculates the down payment for a car purchase given the specified conditions. -/
theorem car_down_payment_calculation
  (car_cost : ℚ)
  (loan_term : ℕ)
  (monthly_payment : ℚ)
  (interest_rate : ℚ)
  (h_car_cost : car_cost = 32000)
  (h_loan_term : loan_term = 48)
  (h_monthly_payment : monthly_payment = 525)
  (h_interest_rate : interest_rate = 5 / 100)
  : ∃ (down_payment : ℚ),
    down_payment = car_cost - (loan_term * monthly_payment + loan_term * (interest_rate * monthly_payment)) ∧
    down_payment = 5540 :=
by sorry

end NUMINAMATH_CALUDE_car_down_payment_calculation_l2026_202627


namespace NUMINAMATH_CALUDE_old_cards_count_l2026_202696

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 6

theorem old_cards_count : 
  (total_pages * cards_per_page) - new_cards = 10 := by
  sorry

end NUMINAMATH_CALUDE_old_cards_count_l2026_202696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2026_202667

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2026_202667


namespace NUMINAMATH_CALUDE_desired_weather_probability_l2026_202604

def days : ℕ := 5
def p_sun : ℚ := 1/4
def p_rain : ℚ := 3/4

def probability_k_sunny_days (k : ℕ) : ℚ :=
  (Nat.choose days k) * (p_sun ^ k) * (p_rain ^ (days - k))

theorem desired_weather_probability : 
  probability_k_sunny_days 1 + probability_k_sunny_days 2 = 135/2048 := by
  sorry

end NUMINAMATH_CALUDE_desired_weather_probability_l2026_202604


namespace NUMINAMATH_CALUDE_red_cards_after_turning_l2026_202688

def is_divisible (n m : ℕ) : Prop := ∃ k, n = m * k

def count_red_cards (n : ℕ) : ℕ :=
  let initial_red := n
  let turned_by_2 := n / 2
  let odd_turned_by_3 := (n / 3 + 1) / 2
  let even_turned_by_3 := n / 6
  initial_red - turned_by_2 - odd_turned_by_3 + even_turned_by_3

theorem red_cards_after_turning (n : ℕ) (h : n = 100) : count_red_cards n = 49 := by
  sorry

end NUMINAMATH_CALUDE_red_cards_after_turning_l2026_202688


namespace NUMINAMATH_CALUDE_trapezoid_areas_l2026_202697

/-- Represents a trapezoid with given dimensions and a parallel line through the intersection of diagonals -/
structure Trapezoid :=
  (ad : ℝ) -- Length of base AD
  (bc : ℝ) -- Length of base BC
  (ab : ℝ) -- Length of side AB
  (cd : ℝ) -- Length of side CD

/-- Calculates the areas of the two resulting trapezoids formed by a line parallel to the bases through the diagonal intersection point -/
def calculate_areas (t : Trapezoid) : ℝ × ℝ := sorry

/-- Theorem stating the areas of the resulting trapezoids for the given dimensions -/
theorem trapezoid_areas (t : Trapezoid) 
  (h1 : t.ad = 84) (h2 : t.bc = 42) (h3 : t.ab = 39) (h4 : t.cd = 45) : 
  calculate_areas t = (588, 1680) := by sorry

end NUMINAMATH_CALUDE_trapezoid_areas_l2026_202697


namespace NUMINAMATH_CALUDE_four_numbers_product_sum_l2026_202620

theorem four_numbers_product_sum (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2 ∧
   x₂ + x₁ * x₃ * x₄ = 2 ∧
   x₃ + x₁ * x₂ * x₄ = 2 ∧
   x₄ + x₁ * x₂ * x₃ = 2) ↔
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) := by
sorry


end NUMINAMATH_CALUDE_four_numbers_product_sum_l2026_202620


namespace NUMINAMATH_CALUDE_fraction_equality_l2026_202607

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2026_202607


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2026_202644

theorem sqrt_sum_inequality (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) : 
  x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2026_202644


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2026_202685

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℝ)
  (group1_count : Nat)
  (group1_average : ℝ)
  (group2_count : Nat)
  (group2_average : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_count = 4)
  (h4 : group1_average = 14)
  (h5 : group2_count = 9)
  (h6 : group2_average = 16)
  (h7 : group1_count + group2_count + 1 = total_students) :
  ∃ (fifteenth_age : ℝ),
    fifteenth_age = total_students * average_age - (group1_count * group1_average + group2_count * group2_average) :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2026_202685


namespace NUMINAMATH_CALUDE_right_triangle_legs_l2026_202693

theorem right_triangle_legs (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a + b + c = 60 →   -- Perimeter condition
  h = 12 →           -- Altitude condition
  h = (a * b) / c →  -- Altitude formula
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l2026_202693


namespace NUMINAMATH_CALUDE_inverse_of_matrix_A_l2026_202659

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 15; 2, 6]

theorem inverse_of_matrix_A (h : Matrix.det A = 0) :
  A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_matrix_A_l2026_202659


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l2026_202660

-- Define the equations of the circles
def circle1_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 20
def circle2_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the lines
def line_y0 (x y : ℝ) : Prop := y = 0
def line_2x_y0 (x y : ℝ) : Prop := 2*x + y = 0
def line_tangent (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the points
def point_A : ℝ × ℝ := (1, 4)
def point_B : ℝ × ℝ := (3, 2)
def point_M : ℝ × ℝ := (2, -1)

-- Theorem for the first circle
theorem circle1_properties :
  ∃ (center_x : ℝ),
    (∀ (y : ℝ), circle1_equation center_x y → line_y0 center_x y) ∧
    circle1_equation point_A.1 point_A.2 ∧
    circle1_equation point_B.1 point_B.2 :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  ∃ (center_x center_y : ℝ),
    line_2x_y0 center_x center_y ∧
    (∀ (x y : ℝ), circle2_equation x y → 
      (x = point_M.1 ∧ y = point_M.2) → line_tangent x y) :=
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l2026_202660


namespace NUMINAMATH_CALUDE_average_equality_implies_z_l2026_202699

theorem average_equality_implies_z (z : ℝ) : 
  (8 + 11 + 20) / 3 = (14 + z) / 2 → z = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_equality_implies_z_l2026_202699


namespace NUMINAMATH_CALUDE_gcd_of_A_and_B_l2026_202651

def A : ℕ := 2 * 3 * 5
def B : ℕ := 2 * 2 * 5 * 7

theorem gcd_of_A_and_B : Nat.gcd A B = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_A_and_B_l2026_202651


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l2026_202637

theorem minimum_value_of_function (x : ℝ) (h : x > 0) : 
  (∀ y : ℝ, y > 0 → 4 / y + y ≥ 4) ∧ (∃ z : ℝ, z > 0 ∧ 4 / z + z = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l2026_202637


namespace NUMINAMATH_CALUDE_max_value_abc_l2026_202609

theorem max_value_abc (a b c : ℝ) (h : a^2 + b^2/4 + c^2/9 = 1) :
  a + b + c ≤ Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2026_202609


namespace NUMINAMATH_CALUDE_fourth_grade_students_at_end_l2026_202611

/-- Calculates the number of students remaining in fourth grade at the end of the year. -/
def studentsAtEnd (initial : Float) (left : Float) (transferred : Float) : Float :=
  initial - left - transferred

/-- Theorem stating that the number of students at the end of the year is 28.0 -/
theorem fourth_grade_students_at_end :
  studentsAtEnd 42.0 4.0 10.0 = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_at_end_l2026_202611


namespace NUMINAMATH_CALUDE_v_2002_equals_4_l2026_202634

-- Define the function g
def g : ℕ → ℕ
| 1 => 2
| 2 => 3
| 3 => 5
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

-- Theorem to prove
theorem v_2002_equals_4 : v 2002 = 4 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_4_l2026_202634


namespace NUMINAMATH_CALUDE_missing_digit_is_seven_l2026_202600

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem missing_digit_is_seven :
  ∃ (d : ℕ), d < 10 ∧ is_divisible_by_9 (365000 + d * 100 + 42) ∧ d = 7 :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_is_seven_l2026_202600
