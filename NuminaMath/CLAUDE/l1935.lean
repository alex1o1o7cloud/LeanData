import Mathlib

namespace position_of_negative_three_l1935_193500

theorem position_of_negative_three : 
  ∀ (x : ℝ), (x = 1 - 4) → (x = -3) :=
by
  sorry

end position_of_negative_three_l1935_193500


namespace extreme_value_condition_l1935_193521

-- Define the function f(x)
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x + m^2

-- Define the derivative of f(x)
def f_prime (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*m*x + n

-- Theorem statement
theorem extreme_value_condition (m n : ℝ) :
  f m n (-1) = 0 ∧ f_prime m n (-1) = 0 → m + n = 11 := by
  sorry

end extreme_value_condition_l1935_193521


namespace star_two_three_l1935_193510

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 * b^2 - a + 2

-- Theorem statement
theorem star_two_three : star 2 3 = 36 := by
  sorry

end star_two_three_l1935_193510


namespace inequality_not_always_holds_l1935_193522

theorem inequality_not_always_holds (a b : ℝ) (h : a > b) :
  ¬ ∀ c : ℝ, a * c > b * c := by
  sorry

end inequality_not_always_holds_l1935_193522


namespace outflow_symmetry_outflow_ratio_replacement_time_ratio_l1935_193542

/-- Represents the structure of a sewage purification tower -/
structure SewageTower where
  layers : Nat
  outlets : Nat
  flow_distribution : List (List Rat)

/-- Calculates the outflow for a given outlet -/
def outflow (tower : SewageTower) (outlet : Nat) : Rat :=
  sorry

/-- Theorem stating that outflows of outlet 2 and 4 are equal -/
theorem outflow_symmetry (tower : SewageTower) :
  tower.outlets = 5 → outflow tower 2 = outflow tower 4 :=
  sorry

/-- Theorem stating the ratio of outflows for outlets 1, 2, and 3 -/
theorem outflow_ratio (tower : SewageTower) :
  tower.outlets = 5 →
  ∃ (k : Rat), outflow tower 1 = k ∧ outflow tower 2 = 4*k ∧ outflow tower 3 = 6*k :=
  sorry

/-- Calculates the wear rate for a given triangle in the tower -/
def wear_rate (tower : SewageTower) (triangle : Nat) : Rat :=
  sorry

/-- Theorem stating the replacement time ratio for slowest and fastest wearing triangles -/
theorem replacement_time_ratio (tower : SewageTower) :
  ∃ (slow fast : Nat),
    wear_rate tower slow = (1/8 : Rat) * wear_rate tower fast ∧
    ∀ t, wear_rate tower t ≥ wear_rate tower slow ∧
         wear_rate tower t ≤ wear_rate tower fast :=
  sorry

end outflow_symmetry_outflow_ratio_replacement_time_ratio_l1935_193542


namespace minimum_cubes_for_valid_assembly_l1935_193578

/-- Represents a cube with either one or two snaps -/
inductive Cube
  | SingleSnap
  | DoubleSnap

/-- An assembly of cubes -/
def Assembly := List Cube

/-- Checks if an assembly is valid (all snaps covered, only receptacles exposed) -/
def isValidAssembly : Assembly → Bool := sorry

/-- Counts the number of cubes in an assembly -/
def countCubes : Assembly → Nat := sorry

theorem minimum_cubes_for_valid_assembly :
  ∃ (a : Assembly), isValidAssembly a ∧ countCubes a = 6 ∧
  ∀ (b : Assembly), isValidAssembly b → countCubes b ≥ 6 := by
  sorry

end minimum_cubes_for_valid_assembly_l1935_193578


namespace cup_production_decrease_rate_l1935_193597

theorem cup_production_decrease_rate 
  (initial_production : ℝ) 
  (final_production : ℝ) 
  (months : ℕ) 
  (h1 : initial_production = 1.6) 
  (h2 : final_production = 0.9) 
  (h3 : months = 2) :
  ∃ (rate : ℝ), 
    rate = 0.25 ∧ 
    final_production = initial_production * (1 - rate) ^ months :=
by sorry

end cup_production_decrease_rate_l1935_193597


namespace inverse_function_intersection_l1935_193513

def f (x : ℝ) : ℝ := 3 * x^2 - 8

theorem inverse_function_intersection (x : ℝ) : 
  f x = x ↔ x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6 := by
  sorry

end inverse_function_intersection_l1935_193513


namespace journey_distance_l1935_193503

theorem journey_distance (speed : ℝ) (time : ℝ) 
  (h1 : (speed + 1/2) * (3/4 * time) = speed * time)
  (h2 : (speed - 1/2) * (time + 3) = speed * time)
  : speed * time = 9 := by
  sorry

end journey_distance_l1935_193503


namespace tangent_length_to_circle_l1935_193528

/-- The length of the tangent segment from the origin to a circle passing through three given points -/
theorem tangent_length_to_circle (A B C : ℝ × ℝ) : 
  A = (2, 3) → B = (4, 6) → C = (3, 9) → 
  ∃ (circle : Set (ℝ × ℝ)), 
    (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle) ∧
    (∃ (T : ℝ × ℝ), T ∈ circle ∧ 
      (∀ (P : ℝ × ℝ), P ∈ circle → dist (0, 0) P ≥ dist (0, 0) T) ∧
      dist (0, 0) T = Real.sqrt (10 + 3 * Real.sqrt 5)) :=
by sorry

end tangent_length_to_circle_l1935_193528


namespace not_q_necessary_not_sufficient_for_not_p_l1935_193565

-- Define propositions p and q
def p (x : ℝ) : Prop := abs (x + 2) > 2
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬q is necessary but not sufficient for ¬p
theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  ¬(∀ x : ℝ, not_q x → not_p x) :=
sorry

end not_q_necessary_not_sufficient_for_not_p_l1935_193565


namespace last_four_average_l1935_193549

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 67.25 :=
by sorry

end last_four_average_l1935_193549


namespace probability_james_and_david_chosen_l1935_193534

-- Define the total number of workers
def total_workers : ℕ := 14

-- Define the number of workers to be chosen
def chosen_workers : ℕ := 2

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem probability_james_and_david_chosen :
  (1 : ℚ) / (combination total_workers chosen_workers) = 1 / 91 :=
sorry

end probability_james_and_david_chosen_l1935_193534


namespace permutation_combination_sum_l1935_193564

/-- Given that A(n,m) = 272 and C(n,m) = 136, prove that m + n = 19 -/
theorem permutation_combination_sum (m n : ℕ) : 
  (m.factorial * (n.choose m) = 272) → (n.choose m = 136) → m + n = 19 := by
  sorry

end permutation_combination_sum_l1935_193564


namespace divisibility_problem_l1935_193577

theorem divisibility_problem (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 18)
  (h2 : Nat.gcd b c = 45)
  (h3 : Nat.gcd c d = 75)
  (h4 : 80 < Nat.gcd d a)
  (h5 : Nat.gcd d a < 120) :
  7 ∣ a.val := by
  sorry

end divisibility_problem_l1935_193577


namespace range_of_a_l1935_193524

def P (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Q → x ∈ P a) → -1 ≤ a ∧ a ≤ 5 := by
  sorry

end range_of_a_l1935_193524


namespace negative_square_cubed_l1935_193526

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l1935_193526


namespace tower_heights_theorem_l1935_193599

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the possible height contributions of a brick -/
def HeightContributions : List ℕ := [4, 10, 19]

/-- The total number of bricks -/
def TotalBricks : ℕ := 94

/-- Calculates the number of distinct tower heights -/
def distinctTowerHeights (brickDims : BrickDimensions) (contributions : List ℕ) (totalBricks : ℕ) : ℕ :=
  sorry

theorem tower_heights_theorem (brickDims : BrickDimensions) 
    (h1 : brickDims.length = 4 ∧ brickDims.width = 10 ∧ brickDims.height = 19) :
    distinctTowerHeights brickDims HeightContributions TotalBricks = 465 := by
  sorry

end tower_heights_theorem_l1935_193599


namespace sqrt_inequality_l1935_193504

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a + d = b + c) : 
  Real.sqrt d + Real.sqrt a < Real.sqrt b + Real.sqrt c := by
sorry

end sqrt_inequality_l1935_193504


namespace profit_sharing_multiple_l1935_193595

/-- Given the conditions of a profit-sharing scenario, prove that the multiple of R's capital is 10. -/
theorem profit_sharing_multiple (P Q R k : ℚ) (total_profit : ℚ) : 
  4 * P = 6 * Q ∧ 
  4 * P = k * R ∧ 
  total_profit = 4340 ∧ 
  R * (total_profit / (P + Q + R)) = 840 →
  k = 10 := by
  sorry

end profit_sharing_multiple_l1935_193595


namespace fifth_term_value_l1935_193558

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = 6 →
  a 3 + a 5 + a 7 = 78 →
  a 5 = 18 := by
sorry

end fifth_term_value_l1935_193558


namespace outdoor_section_length_l1935_193583

/-- The length of a rectangular outdoor section, given its width and area -/
theorem outdoor_section_length (width : ℝ) (area : ℝ) (h1 : width = 4) (h2 : area = 24) :
  area / width = 6 := by
  sorry

end outdoor_section_length_l1935_193583


namespace proportional_segments_l1935_193535

theorem proportional_segments (a b c d : ℝ) : 
  a = 3 ∧ d = 4 ∧ c = 6 ∧ (a / b = c / d) → b = 2 := by
  sorry

end proportional_segments_l1935_193535


namespace smaller_rectangle_area_l1935_193519

theorem smaller_rectangle_area (length width : ℝ) (h1 : length = 40) (h2 : width = 20) :
  (length / 2) * (width / 2) = 200 :=
by
  sorry

end smaller_rectangle_area_l1935_193519


namespace polynomial_remainder_l1935_193585

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem polynomial_remainder : p 2 = 88 := by
  sorry

end polynomial_remainder_l1935_193585


namespace advance_tickets_sold_l1935_193579

theorem advance_tickets_sold (advance_cost same_day_cost total_tickets total_receipts : ℕ) 
  (h1 : advance_cost = 20)
  (h2 : same_day_cost = 30)
  (h3 : total_tickets = 60)
  (h4 : total_receipts = 1600) :
  ∃ (advance_sold : ℕ), 
    advance_sold * advance_cost + (total_tickets - advance_sold) * same_day_cost = total_receipts ∧ 
    advance_sold = 20 :=
by sorry

end advance_tickets_sold_l1935_193579


namespace inequality_solution_set_l1935_193537

theorem inequality_solution_set (x : ℝ) :
  (-2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2) ↔
  (x < -13.041 ∨ x > -0.959) :=
sorry

end inequality_solution_set_l1935_193537


namespace oatmeal_cookies_count_l1935_193551

/-- Represents the number of cookies in each baggie -/
def cookies_per_baggie : ℕ := 3

/-- Represents the number of chocolate chip cookies Maria had -/
def chocolate_chip_cookies : ℕ := 2

/-- Represents the number of baggies Maria could make -/
def total_baggies : ℕ := 6

/-- Theorem stating the number of oatmeal cookies Maria had -/
theorem oatmeal_cookies_count : 
  (total_baggies * cookies_per_baggie) - chocolate_chip_cookies = 16 := by
  sorry

end oatmeal_cookies_count_l1935_193551


namespace ratio_of_squares_l1935_193569

theorem ratio_of_squares (x y : ℝ) (h : x^2 = 8*y^2 - 224) :
  x/y = Real.sqrt (8 - 224/y^2) :=
by sorry

end ratio_of_squares_l1935_193569


namespace abs_geq_ax_implies_abs_a_leq_one_l1935_193570

theorem abs_geq_ax_implies_abs_a_leq_one (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end abs_geq_ax_implies_abs_a_leq_one_l1935_193570


namespace percentage_of_percentage_l1935_193587

theorem percentage_of_percentage (x : ℝ) : (10 / 100) * ((50 / 100) * 500) = 25 := by
  sorry

end percentage_of_percentage_l1935_193587


namespace lollipops_remaining_l1935_193557

def raspberry_lollipops : ℕ := 57
def mint_lollipops : ℕ := 98
def blueberry_lollipops : ℕ := 13
def cola_lollipops : ℕ := 167
def num_friends : ℕ := 13

theorem lollipops_remaining :
  (raspberry_lollipops + mint_lollipops + blueberry_lollipops + cola_lollipops) % num_friends = 10 :=
by sorry

end lollipops_remaining_l1935_193557


namespace two_numbers_difference_l1935_193536

theorem two_numbers_difference (x y : ℝ) : x + y = 55 ∧ x = 35 → x - y = 15 := by
  sorry

end two_numbers_difference_l1935_193536


namespace unique_odd_k_for_sum_1372_l1935_193527

theorem unique_odd_k_for_sum_1372 :
  ∃! (k : ℤ), ∃ (m : ℕ), 
    (k % 2 = 1) ∧ 
    (m > 0) ∧ 
    (k * m + 5 * (m * (m - 1) / 2) = 1372) ∧ 
    (k = 211) := by
  sorry

end unique_odd_k_for_sum_1372_l1935_193527


namespace inscribed_octagon_area_l1935_193574

theorem inscribed_octagon_area (circle_area : ℝ) (octagon_area : ℝ) : 
  circle_area = 64 * Real.pi →
  octagon_area = 8 * (1 / 2 * (circle_area / Real.pi) * Real.sin (Real.pi / 4)) →
  octagon_area = 128 * Real.sqrt 2 := by
  sorry

end inscribed_octagon_area_l1935_193574


namespace three_numbers_sum_6_product_4_l1935_193568

theorem three_numbers_sum_6_product_4 :
  ∀ a b c : ℕ,
  a + b + c = 6 →
  a * b * c = 4 →
  ((a = 1 ∧ b = 1 ∧ c = 4) ∨
   (a = 1 ∧ b = 4 ∧ c = 1) ∨
   (a = 4 ∧ b = 1 ∧ c = 1)) :=
by sorry

end three_numbers_sum_6_product_4_l1935_193568


namespace fahrenheit_to_celsius_l1935_193563

theorem fahrenheit_to_celsius (F C : ℝ) : F = 1.8 * C + 32 → F = 68 → C = 20 := by
  sorry

end fahrenheit_to_celsius_l1935_193563


namespace beth_marbles_l1935_193566

/-- The number of marbles Beth has initially -/
def initial_marbles : ℕ := 72

/-- The number of colors of marbles -/
def num_colors : ℕ := 3

/-- The number of red marbles Beth loses -/
def lost_red : ℕ := 5

/-- Calculates the number of marbles Beth has left after losing some -/
def marbles_left (initial : ℕ) (colors : ℕ) (lost_red : ℕ) : ℕ :=
  initial - (lost_red + 2 * lost_red + 3 * lost_red)

theorem beth_marbles :
  marbles_left initial_marbles num_colors lost_red = 42 := by
  sorry

end beth_marbles_l1935_193566


namespace same_parity_min_max_l1935_193586

/-- A set of elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def min_element (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def max_element (A : Set ℤ) : ℤ := sorry

/-- A predicate to check if a number is even -/
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_min_max : 
  is_even (min_element A_P) ↔ is_even (max_element A_P) := by sorry

end same_parity_min_max_l1935_193586


namespace normal_distribution_standard_deviations_l1935_193523

/-- Proves that for a normal distribution with mean 14.0 and standard deviation 1.5,
    the value 11 is exactly 2 standard deviations less than the mean. -/
theorem normal_distribution_standard_deviations (μ σ x : ℝ) 
  (h_mean : μ = 14.0)
  (h_std_dev : σ = 1.5)
  (h_value : x = 11.0) :
  (μ - x) / σ = 2 := by
  sorry

end normal_distribution_standard_deviations_l1935_193523


namespace factorable_implies_even_l1935_193518

-- Define the quadratic expression
def quadratic (a : ℤ) (x : ℝ) : ℝ := 21 * x^2 + a * x + 21

-- Define what it means for the quadratic to be factorable into linear binomials with integer coefficients
def is_factorable (a : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    ∀ (x : ℝ), quadratic a x = (m * x + n) * (p * x + q)

-- The theorem to prove
theorem factorable_implies_even (a : ℤ) : 
  is_factorable a → ∃ k : ℤ, a = 2 * k :=
sorry

end factorable_implies_even_l1935_193518


namespace production_cost_reduction_l1935_193588

/-- Represents the equation for production cost reduction over two years -/
theorem production_cost_reduction (initial_cost target_cost : ℝ) (x : ℝ) :
  initial_cost = 200000 →
  target_cost = 150000 →
  initial_cost * (1 - x)^2 = target_cost :=
by
  sorry

end production_cost_reduction_l1935_193588


namespace motorcyclist_meets_cyclist1_l1935_193501

/-- Represents the time in minutes for two entities to meet or overtake each other. -/
structure MeetingTime where
  time : ℝ
  time_positive : time > 0

/-- Represents an entity moving on the circular highway. -/
structure Entity where
  speed : ℝ
  direction : Bool  -- True for one direction, False for the opposite

/-- The circular highway setup with four entities. -/
structure CircularHighway where
  runner : Entity
  cyclist1 : Entity
  cyclist2 : Entity
  motorcyclist : Entity
  runner_cyclist2_meeting : MeetingTime
  runner_cyclist1_overtake : MeetingTime
  motorcyclist_cyclist2_overtake : MeetingTime
  highway_length : ℝ
  highway_length_positive : highway_length > 0

  runner_direction : runner.direction = true
  cyclist1_direction : cyclist1.direction = true
  cyclist2_direction : cyclist2.direction = false
  motorcyclist_direction : motorcyclist.direction = false

  runner_cyclist2_meeting_time : runner_cyclist2_meeting.time = 12
  runner_cyclist1_overtake_time : runner_cyclist1_overtake.time = 20
  motorcyclist_cyclist2_overtake_time : motorcyclist_cyclist2_overtake.time = 5

/-- The theorem stating that the motorcyclist meets the first cyclist every 3 minutes. -/
theorem motorcyclist_meets_cyclist1 (h : CircularHighway) :
  ∃ (t : MeetingTime), t.time = 3 ∧
    h.highway_length / t.time = h.motorcyclist.speed + h.cyclist1.speed :=
sorry

end motorcyclist_meets_cyclist1_l1935_193501


namespace walking_delay_l1935_193596

/-- Proves that walking at 3/4 of normal speed results in an 8-minute delay -/
theorem walking_delay (normal_speed : ℝ) (distance : ℝ) : 
  normal_speed > 0 → distance > 0 → 
  (distance / normal_speed = 24) → 
  (distance / (3/4 * normal_speed) - 24 = 8) := by
  sorry

end walking_delay_l1935_193596


namespace final_balance_is_correct_l1935_193515

/-- Represents a bank account with transactions and interest --/
structure BankAccount where
  initialBalance : ℝ
  annualInterestRate : ℝ
  monthlyInterestRate : ℝ
  shoeWithdrawalPercent : ℝ
  shoeDepositPercent : ℝ
  paycheckDepositPercent : ℝ
  giftWithdrawalPercent : ℝ

/-- Calculates the final balance after all transactions and interest --/
def finalBalance (account : BankAccount) : ℝ :=
  let shoeWithdrawal := account.initialBalance * account.shoeWithdrawalPercent
  let balanceAfterShoes := account.initialBalance - shoeWithdrawal
  let shoeDeposit := shoeWithdrawal * account.shoeDepositPercent
  let balanceAfterShoeDeposit := balanceAfterShoes + shoeDeposit
  let januaryInterest := balanceAfterShoeDeposit * account.monthlyInterestRate
  let balanceAfterJanuary := balanceAfterShoeDeposit + januaryInterest
  let paycheckDeposit := shoeWithdrawal * account.paycheckDepositPercent
  let balanceAfterPaycheck := balanceAfterJanuary + paycheckDeposit
  let februaryInterest := balanceAfterPaycheck * account.monthlyInterestRate
  let balanceAfterFebruary := balanceAfterPaycheck + februaryInterest
  let giftWithdrawal := balanceAfterFebruary * account.giftWithdrawalPercent
  let balanceAfterGift := balanceAfterFebruary - giftWithdrawal
  let marchInterest := balanceAfterGift * account.monthlyInterestRate
  balanceAfterGift + marchInterest

/-- Theorem stating that the final balance is correct --/
theorem final_balance_is_correct (account : BankAccount) : 
  account.initialBalance = 1200 ∧
  account.annualInterestRate = 0.03 ∧
  account.monthlyInterestRate = account.annualInterestRate / 12 ∧
  account.shoeWithdrawalPercent = 0.08 ∧
  account.shoeDepositPercent = 0.25 ∧
  account.paycheckDepositPercent = 1.5 ∧
  account.giftWithdrawalPercent = 0.05 →
  finalBalance account = 1217.15 := by
  sorry


end final_balance_is_correct_l1935_193515


namespace max_product_l1935_193502

def digits : List Nat := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 5 1 9 3 :=
by sorry

end max_product_l1935_193502


namespace arc_length_for_72_degrees_l1935_193548

theorem arc_length_for_72_degrees (d : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  d = 4 →  -- diameter is 4 cm
  θ_deg = 72 →  -- central angle is 72°
  l = d / 2 * (θ_deg * π / 180) →  -- arc length formula
  l = 4 * π / 5 :=  -- arc length is 4π/5 cm
by sorry

end arc_length_for_72_degrees_l1935_193548


namespace angle_complement_l1935_193571

theorem angle_complement (A : ℝ) : 
  A = 45 → 90 - A = 45 := by
sorry

end angle_complement_l1935_193571


namespace power_fraction_simplification_l1935_193553

theorem power_fraction_simplification :
  (3^2020 - 3^2018) / (3^2020 + 3^2018) = 4/5 := by
  sorry

end power_fraction_simplification_l1935_193553


namespace f_intersects_x_axis_iff_l1935_193581

/-- A function that represents (k-3)x^2+2x+1 --/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- Predicate to check if a function intersects the x-axis --/
def intersects_x_axis (g : ℝ → ℝ) : Prop :=
  ∃ x, g x = 0

/-- Theorem stating that f intersects the x-axis iff k ≤ 4 --/
theorem f_intersects_x_axis_iff (k : ℝ) :
  intersects_x_axis (f k) ↔ k ≤ 4 := by
  sorry

end f_intersects_x_axis_iff_l1935_193581


namespace millet_majority_on_friday_l1935_193509

/-- Represents the amount of millet in the feeder on a given day -/
def millet_amount (day : ℕ) : ℚ :=
  0.5 * (1 - (0.7 ^ day))

/-- Represents the total amount of seeds in the feeder on a given day -/
def total_seeds (day : ℕ) : ℚ :=
  0.5 * day

/-- Theorem stating that on the 5th day, more than two-thirds of the seeds are millet -/
theorem millet_majority_on_friday :
  (millet_amount 5) / (total_seeds 5) > 2/3 ∧
  ∀ d : ℕ, d < 5 → (millet_amount d) / (total_seeds d) ≤ 2/3 :=
sorry

end millet_majority_on_friday_l1935_193509


namespace simple_interest_rate_percent_l1935_193552

/-- Given simple interest conditions, prove the rate percent -/
theorem simple_interest_rate_percent 
  (P : ℝ) (SI : ℝ) (T : ℝ) 
  (h_P : P = 900) 
  (h_SI : SI = 160) 
  (h_T : T = 4) 
  (h_formula : SI = (P * R * T) / 100) : 
  R = 400 / 90 := by
  sorry

end simple_interest_rate_percent_l1935_193552


namespace perfect_square_implies_zero_l1935_193562

theorem perfect_square_implies_zero (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, a * 2013^n + b = k^2) → a = 0 := by
  sorry

end perfect_square_implies_zero_l1935_193562


namespace quiz_logic_l1935_193538

theorem quiz_logic (x y z w u v : ℝ) : 
  (x > y → z < w) → 
  (z > w → u < v) → 
  ¬((x < y → u < v) ∨ 
    (u < v → x < y) ∨ 
    (u > v → x > y) ∨ 
    (x > y → u > v)) := by
  sorry

end quiz_logic_l1935_193538


namespace isosceles_triangle_perimeter_l1935_193530

/-- An isosceles triangle with side lengths 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  a + b + c = 15 :=        -- The perimeter is 15
by
  sorry


end isosceles_triangle_perimeter_l1935_193530


namespace smallest_n_for_odd_ratio_l1935_193546

def concatenate_decimal_expansions (k : ℕ) : ℕ :=
  sorry

def X (n : ℕ) : ℕ := concatenate_decimal_expansions n

theorem smallest_n_for_odd_ratio :
  (∀ n : ℕ, n ≥ 2 → n < 5 → ¬(Odd (X n / 1024^n))) ∧
  (Odd (X 5 / 1024^5)) := by
  sorry

end smallest_n_for_odd_ratio_l1935_193546


namespace smallest_number_in_set_l1935_193576

def number_set : Set ℤ := {0, -2, 1, 5}

theorem smallest_number_in_set : 
  ∃ x ∈ number_set, ∀ y ∈ number_set, x ≤ y ∧ x = -2 := by
  sorry

end smallest_number_in_set_l1935_193576


namespace ratio_a_c_l1935_193508

-- Define the ratios
def ratio_a_b : ℚ := 5 / 3
def ratio_b_c : ℚ := 1 / 5

-- Theorem statement
theorem ratio_a_c (a b c : ℚ) (h1 : a / b = ratio_a_b) (h2 : b / c = ratio_b_c) : 
  a / c = 1 / 3 := by
  sorry

end ratio_a_c_l1935_193508


namespace matthew_cake_division_l1935_193517

/-- Given that Matthew has 30 cakes and 2 friends, prove that each friend receives 15 cakes when the cakes are divided equally. -/
theorem matthew_cake_division (total_cakes : ℕ) (num_friends : ℕ) (cakes_per_friend : ℕ) :
  total_cakes = 30 →
  num_friends = 2 →
  cakes_per_friend = total_cakes / num_friends →
  cakes_per_friend = 15 := by
  sorry

end matthew_cake_division_l1935_193517


namespace correct_multiple_choice_count_l1935_193591

/-- Represents the citizenship test with multiple-choice and fill-in-the-blank questions. -/
structure CitizenshipTest where
  totalQuestions : ℕ
  multipleChoiceTime : ℕ
  fillInBlankTime : ℕ
  totalStudyTime : ℕ

/-- Calculates the number of multiple-choice questions on the test. -/
def multipleChoiceCount (test : CitizenshipTest) : ℕ :=
  30

/-- Theorem stating that for the given test parameters, 
    the number of multiple-choice questions is 30. -/
theorem correct_multiple_choice_count 
  (test : CitizenshipTest)
  (h1 : test.totalQuestions = 60)
  (h2 : test.multipleChoiceTime = 15)
  (h3 : test.fillInBlankTime = 25)
  (h4 : test.totalStudyTime = 1200) :
  multipleChoiceCount test = 30 := by
  sorry

#eval multipleChoiceCount {
  totalQuestions := 60,
  multipleChoiceTime := 15,
  fillInBlankTime := 25,
  totalStudyTime := 1200
}

end correct_multiple_choice_count_l1935_193591


namespace game_show_probability_l1935_193539

theorem game_show_probability (total_doors : ℕ) (prize_doors : ℕ) 
  (opened_doors : ℕ) (opened_prize_doors : ℕ) :
  total_doors = 7 →
  prize_doors = 2 →
  opened_doors = 3 →
  opened_prize_doors = 1 →
  (total_doors - opened_doors - 1 : ℚ) / (total_doors - opened_doors) * 
  (prize_doors - opened_prize_doors) / (total_doors - opened_doors - 1) = 4 / 7 :=
by sorry

end game_show_probability_l1935_193539


namespace quadrilateral_side_length_l1935_193547

/-- Represents a quadrilateral with sides a, b, c, d --/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents the properties of the specific quadrilateral in the problem --/
def ProblemQuadrilateral (q : Quadrilateral) (x y : ℕ) : Prop :=
  q.a = 20 ∧ 
  q.a = x^2 + y^2 ∧ 
  q.b = x ∧ 
  q.c = y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  q.d ≥ q.a ∧ 
  q.d ≥ q.b ∧ 
  q.d ≥ q.c

theorem quadrilateral_side_length 
  (q : Quadrilateral) 
  (x y : ℕ) 
  (h : ProblemQuadrilateral q x y) : 
  q.d = 4 * Real.sqrt 5 := by
  sorry

end quadrilateral_side_length_l1935_193547


namespace max_k_value_l1935_193575

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-4 + Real.sqrt 29) / 13 := by
  sorry

end max_k_value_l1935_193575


namespace temperature_difference_l1935_193559

theorem temperature_difference 
  (highest_temp lowest_temp : ℝ) 
  (h_highest : highest_temp = 27) 
  (h_lowest : lowest_temp = 17) : 
  highest_temp - lowest_temp = 10 := by
sorry

end temperature_difference_l1935_193559


namespace reciprocal_problem_l1935_193505

theorem reciprocal_problem (x : ℚ) : (10 : ℚ) / 3 = 1 / x + 1 → x = 3 / 7 := by
  sorry

end reciprocal_problem_l1935_193505


namespace students_in_no_subjects_l1935_193550

theorem students_in_no_subjects (total : ℕ) (math chem bio : ℕ) (math_chem chem_bio math_bio : ℕ) (all_three : ℕ) : 
  total = 120 →
  math = 70 →
  chem = 50 →
  bio = 40 →
  math_chem = 30 →
  chem_bio = 20 →
  math_bio = 10 →
  all_three = 5 →
  total - (math + chem + bio - math_chem - chem_bio - math_bio + all_three) = 20 :=
by sorry

end students_in_no_subjects_l1935_193550


namespace intersection_constraint_l1935_193580

theorem intersection_constraint (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 := by
  sorry

end intersection_constraint_l1935_193580


namespace decreasing_linear_function_k_range_l1935_193556

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := (2*k - 4)*x - 1

-- State the theorem
theorem decreasing_linear_function_k_range (k : ℝ) :
  (∀ x y : ℝ, x < y → f k x > f k y) → k < 2 := by sorry

end decreasing_linear_function_k_range_l1935_193556


namespace quadratic_roots_condition_l1935_193560

/-- 
For a quadratic equation x^2 + 8x + q = 0 to have two distinct real roots,
q must be less than 16.
-/
theorem quadratic_roots_condition (q : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 8*x + q = 0 ∧ y^2 + 8*y + q = 0) ↔ q < 16 := by
  sorry

end quadratic_roots_condition_l1935_193560


namespace krishans_money_l1935_193516

/-- Proves that Krishan has Rs. 4335 given the conditions of the problem -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 735 →
  krishan = 4335 := by
  sorry

end krishans_money_l1935_193516


namespace coinciding_rest_days_l1935_193520

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 6

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Chris's rest days within his cycle -/
def chris_rest_days : List ℕ := [5, 6]

/-- Dana's rest days within her cycle -/
def dana_rest_days : List ℕ := [6, 7]

/-- The number of coinciding rest days for Chris and Dana in the first 1000 days -/
theorem coinciding_rest_days : 
  (List.filter (λ d : ℕ => 
    (d % chris_cycle ∈ chris_rest_days) ∧ 
    (d % dana_cycle ∈ dana_rest_days)) 
    (List.range total_days)).length = 23 := by
  sorry

end coinciding_rest_days_l1935_193520


namespace minimum_cost_is_74_l1935_193543

-- Define the box types
inductive BoxType
| A
| B

-- Define the problem parameters
def totalVolume : ℕ := 15
def boxCapacity : BoxType → ℕ
  | BoxType.A => 2
  | BoxType.B => 3
def boxPrice : BoxType → ℕ
  | BoxType.A => 13
  | BoxType.B => 15
def discountThreshold : ℕ := 3
def discountAmount : ℕ := 10

-- Define a purchase plan
def PurchasePlan := BoxType → ℕ

-- Calculate the total volume of a purchase plan
def totalVolumeOfPlan (plan : PurchasePlan) : ℕ :=
  (plan BoxType.A) * (boxCapacity BoxType.A) + (plan BoxType.B) * (boxCapacity BoxType.B)

-- Calculate the cost of a purchase plan
def costOfPlan (plan : PurchasePlan) : ℕ :=
  let basePrice := (plan BoxType.A) * (boxPrice BoxType.A) + (plan BoxType.B) * (boxPrice BoxType.B)
  if plan BoxType.A ≥ discountThreshold then basePrice - discountAmount else basePrice

-- Define a valid purchase plan
def isValidPlan (plan : PurchasePlan) : Prop :=
  totalVolumeOfPlan plan = totalVolume

-- Theorem to prove
theorem minimum_cost_is_74 :
  ∃ (plan : PurchasePlan), isValidPlan plan ∧
    ∀ (otherPlan : PurchasePlan), isValidPlan otherPlan → costOfPlan plan ≤ costOfPlan otherPlan ∧
    costOfPlan plan = 74 :=
  sorry

end minimum_cost_is_74_l1935_193543


namespace new_person_weight_l1935_193573

/-- Given a group of 10 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 7.2 kg,
    then the weight of the new person is 137 kg. -/
theorem new_person_weight
  (n : ℕ)
  (initial_weight : ℝ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 10)
  (h2 : weight_increase = 7.2)
  (h3 : replaced_weight = 65)
  : initial_weight + n * weight_increase = initial_weight - replaced_weight + 137 := by
  sorry

end new_person_weight_l1935_193573


namespace marble_difference_l1935_193594

/-- Given information about marbles owned by Amanda, Katrina, and Mabel -/
theorem marble_difference (amanda katrina mabel : ℕ) 
  (h1 : amanda + 12 = 2 * katrina)
  (h2 : mabel = 5 * katrina)
  (h3 : mabel = 85) :
  mabel - amanda = 63 := by
  sorry

end marble_difference_l1935_193594


namespace employee_pay_l1935_193540

/-- Given two employees A and B with a total weekly pay of 450 and A's pay being 150% of B's,
    prove that B's weekly pay is 180. -/
theorem employee_pay (total_pay : ℝ) (a_pay : ℝ) (b_pay : ℝ) 
  (h1 : total_pay = 450)
  (h2 : a_pay = 1.5 * b_pay)
  (h3 : total_pay = a_pay + b_pay) :
  b_pay = 180 := by
  sorry

end employee_pay_l1935_193540


namespace nuts_problem_l1935_193554

theorem nuts_problem (x y : ℕ) : 
  (70 ≤ x + y ∧ x + y ≤ 80) ∧ 
  (3 * x + 5 * y + x = 20 * x + 20) →
  x = 36 ∧ y = 41 :=
sorry

end nuts_problem_l1935_193554


namespace interest_rate_calculation_l1935_193572

/-- Proves that the annual interest rate is 0.1 given the initial investment,
    final amount, and time period. -/
theorem interest_rate_calculation (initial_investment : ℝ) (final_amount : ℝ) (years : ℕ) :
  initial_investment = 3000 →
  final_amount = 3630.0000000000005 →
  years = 2 →
  ∃ (r : ℝ), r = 0.1 ∧ final_amount = initial_investment * (1 + r) ^ years :=
by sorry


end interest_rate_calculation_l1935_193572


namespace alison_money_l1935_193589

def money_problem (kent_original brittany brooke kent alison : ℚ) : Prop :=
  let kent_after_lending := kent_original - 200
  alison = brittany / 2 ∧
  brittany = 4 * brooke ∧
  brooke = 2 * kent ∧
  kent = kent_after_lending ∧
  kent_original = 1000

theorem alison_money :
  ∀ kent_original brittany brooke kent alison,
    money_problem kent_original brittany brooke kent alison →
    alison = 3200 := by
  sorry

end alison_money_l1935_193589


namespace green_peaches_count_l1935_193592

/-- Given a basket of peaches with a total of 10 peaches and 4 red peaches,
    prove that there are 6 green peaches in the basket. -/
theorem green_peaches_count (total_peaches : ℕ) (red_peaches : ℕ) (baskets : ℕ) :
  total_peaches = 10 → red_peaches = 4 → baskets = 1 →
  total_peaches - red_peaches = 6 := by
  sorry

end green_peaches_count_l1935_193592


namespace length_breadth_difference_is_ten_l1935_193561

/-- Represents a rectangular plot with given dimensions and fencing costs. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fenceCostPerMeter : ℝ
  totalFenceCost : ℝ

/-- Calculates the difference between length and breadth of the plot. -/
def lengthBreadthDifference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- Theorem stating that for a rectangular plot with length 55 meters,
    where the cost of fencing at Rs. 26.50 per meter totals Rs. 5300,
    the length is 10 meters more than the breadth. -/
theorem length_breadth_difference_is_ten
  (plot : RectangularPlot)
  (h1 : plot.length = 55)
  (h2 : plot.fenceCostPerMeter = 26.5)
  (h3 : plot.totalFenceCost = 5300)
  (h4 : plot.totalFenceCost = plot.fenceCostPerMeter * (2 * (plot.length + plot.breadth))) :
  lengthBreadthDifference plot = 10 := by
  sorry

#eval lengthBreadthDifference { length := 55, breadth := 45, fenceCostPerMeter := 26.5, totalFenceCost := 5300 }

end length_breadth_difference_is_ten_l1935_193561


namespace sally_picked_42_peaches_l1935_193532

/-- The number of peaches Sally picked up at the orchard -/
def peaches_picked (initial current : ℕ) : ℕ := current - initial

/-- Proof that Sally picked up 42 peaches -/
theorem sally_picked_42_peaches (initial current : ℕ) 
  (h_initial : initial = 13)
  (h_current : current = 55) :
  peaches_picked initial current = 42 := by
  sorry

end sally_picked_42_peaches_l1935_193532


namespace odd_function_property_l1935_193507

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end odd_function_property_l1935_193507


namespace right_triangle_hypotenuse_l1935_193533

theorem right_triangle_hypotenuse (a b c : ℝ) :
  -- Right triangle condition
  c^2 = a^2 + b^2 →
  -- Area condition
  (1/2) * a * b = 48 →
  -- Geometric mean condition
  (a * b)^(1/2) = 8 →
  -- Conclusion: hypotenuse length
  c = 4 * (13 : ℝ)^(1/2) := by
  sorry

end right_triangle_hypotenuse_l1935_193533


namespace smallest_five_digit_congruent_to_11_mod_14_l1935_193598

theorem smallest_five_digit_congruent_to_11_mod_14 :
  ∃ n : ℕ, 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    n % 14 = 11 ∧
    (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 14 = 11 → n ≤ m) ∧
    n = 10007 :=
by sorry

end smallest_five_digit_congruent_to_11_mod_14_l1935_193598


namespace homework_time_theorem_l1935_193506

/-- The total time left for homework completion --/
def total_time (jacob_time greg_time patrick_time : ℕ) : ℕ :=
  jacob_time + greg_time + patrick_time

/-- Theorem stating the total time left for homework completion --/
theorem homework_time_theorem (jacob_time greg_time patrick_time : ℕ) 
  (h1 : jacob_time = 18)
  (h2 : greg_time = jacob_time - 6)
  (h3 : patrick_time = 2 * greg_time - 4) :
  total_time jacob_time greg_time patrick_time = 50 := by
  sorry

end homework_time_theorem_l1935_193506


namespace system_solution_l1935_193593

theorem system_solution : ∃ (x y : ℚ), 
  (x + 4*y = 14) ∧ 
  ((x - 3) / 4 - (y - 3) / 3 = 1 / 12) ∧ 
  (x = 3) ∧ 
  (y = 11 / 4) := by
  sorry

end system_solution_l1935_193593


namespace quadratic_point_condition_l1935_193544

/-- The quadratic function y = -(x-1)² + n -/
def f (x n : ℝ) : ℝ := -(x - 1)^2 + n

theorem quadratic_point_condition (m y₁ y₂ n : ℝ) :
  f m n = y₁ →
  f (m + 1) n = y₂ →
  y₁ > y₂ →
  m > 1/2 := by
  sorry

end quadratic_point_condition_l1935_193544


namespace seed_calculation_total_seed_gallons_l1935_193555

/-- Calculates the total gallons of seed used for a football field given the specified conditions -/
theorem seed_calculation (field_area : ℝ) (seed_ratio : ℝ) (combined_gallons : ℝ) (combined_area : ℝ) : ℝ :=
  let total_parts := seed_ratio + 1
  let seed_fraction := seed_ratio / total_parts
  let seed_per_combined_area := seed_fraction * combined_gallons
  let field_coverage_factor := field_area / combined_area
  field_coverage_factor * seed_per_combined_area

/-- Proves that the total gallons of seed used for the entire football field is 768 gallons -/
theorem total_seed_gallons :
  seed_calculation 8000 4 240 2000 = 768 := by
  sorry

end seed_calculation_total_seed_gallons_l1935_193555


namespace sequence_properties_l1935_193567

def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a (n + 1) = a n * q

theorem sequence_properties (a b : ℕ → ℕ) (k : ℝ) :
  geometric_sequence a →
  (a 1 = 3) →
  (2 * a 3 = a 2 + (3/4) * a 4) →
  (b 1 = 1) →
  (∀ n : ℕ, b (n + 1) = 2 * b n + 1) →
  (∀ n : ℕ, k * ((b n + 5) / 2) - a n ≥ 8 * n + 2 * k - 24) →
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, b n = 2^n - 1) ∧
  (k ≥ 4) :=
by sorry

end sequence_properties_l1935_193567


namespace room_population_change_l1935_193531

theorem room_population_change (initial_men initial_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  ∃ (current_women : ℕ),
    initial_men + 2 = 14 ∧
    current_women = 2 * (initial_women - 3) ∧
    current_women = 24 :=
by
  sorry

end room_population_change_l1935_193531


namespace fourth_term_of_sequence_l1935_193590

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = (16 : ℝ) ^ (1/4) →
  a 2 = (16 : ℝ) ^ (1/6) →
  a 3 = (16 : ℝ) ^ (1/8) →
  a 4 = (2 : ℝ) ^ (1/3) :=
sorry

end fourth_term_of_sequence_l1935_193590


namespace march_birth_percentage_l1935_193511

def total_people : ℕ := 100
def march_births : ℕ := 8

theorem march_birth_percentage :
  (march_births : ℚ) / (total_people : ℚ) * 100 = 8 := by
  sorry

end march_birth_percentage_l1935_193511


namespace farmer_land_usage_l1935_193512

/-- Represents the ratio of land used for beans, wheat, and corn -/
def land_ratio : Fin 3 → ℕ
  | 0 => 5  -- beans
  | 1 => 2  -- wheat
  | 2 => 4  -- corn
  | _ => 0

/-- The amount of land used for corn in acres -/
def corn_land : ℕ := 376

/-- The total amount of land used by the farmer in acres -/
def total_land : ℕ := 1034

/-- Theorem stating that given the land ratio and corn land usage, 
    the total land used by the farmer is 1034 acres -/
theorem farmer_land_usage : 
  (land_ratio 2 : ℚ) / (land_ratio 0 + land_ratio 1 + land_ratio 2 : ℚ) * total_land = corn_land :=
by sorry

end farmer_land_usage_l1935_193512


namespace equivalence_of_equations_l1935_193529

theorem equivalence_of_equations (p : ℕ) (hp : Nat.Prime p) :
  (∃ (x s : ℤ), x^2 - x + 3 - p * s = 0) ↔
  (∃ (y t : ℤ), y^2 - y + 25 - p * t = 0) := by
sorry

end equivalence_of_equations_l1935_193529


namespace nina_tomato_harvest_l1935_193584

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Represents the planting and yield information for tomatoes -/
structure TomatoInfo where
  plantsPerSquareFoot : ℝ
  tomatoesPerPlant : ℝ

/-- Calculates the total number of tomatoes expected from a garden -/
def expectedTomatoes (d : GardenDimensions) (t : TomatoInfo) : ℝ :=
  gardenArea d * t.plantsPerSquareFoot * t.tomatoesPerPlant

/-- Theorem stating the expected tomato harvest for Nina's garden -/
theorem nina_tomato_harvest :
  let garden := GardenDimensions.mk 10 20
  let tomato := TomatoInfo.mk 5 10
  expectedTomatoes garden tomato = 10000 := by
  sorry


end nina_tomato_harvest_l1935_193584


namespace total_worth_is_14000_l1935_193525

/-- The cost of the ring John gave to his fiancee -/
def ring_cost : ℕ := 4000

/-- The cost of the car John gave to his fiancee -/
def car_cost : ℕ := 2000

/-- The cost of the diamond brace John gave to his fiancee -/
def brace_cost : ℕ := 2 * ring_cost

/-- The total worth of the presents John gave to his fiancee -/
def total_worth : ℕ := ring_cost + car_cost + brace_cost

theorem total_worth_is_14000 : total_worth = 14000 := by
  sorry

end total_worth_is_14000_l1935_193525


namespace divisor_problem_l1935_193541

theorem divisor_problem (n : ℕ) (d : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n + 1 = k * d + 4) (h3 : ∃ m : ℕ, n = 2 * m + 1) : d = 6 := by
  sorry

end divisor_problem_l1935_193541


namespace residual_plot_vertical_axis_l1935_193545

/-- A residual plot used in residual analysis. -/
structure ResidualPlot where
  vertical_axis : Set ℝ
  horizontal_axis : Set ℝ

/-- The definition of a residual in the context of residual analysis. -/
def Residual : Type := ℝ

/-- Theorem stating that the vertical axis of a residual plot represents the residuals. -/
theorem residual_plot_vertical_axis (plot : ResidualPlot) :
  plot.vertical_axis = Set.range (λ r : Residual => r) :=
sorry

end residual_plot_vertical_axis_l1935_193545


namespace point_A_on_curve_l1935_193514

/-- The equation of the curve C is x^2 - xy + y - 5 = 0 -/
def curve_equation (x y : ℝ) : Prop := x^2 - x*y + y - 5 = 0

/-- Point A lies on curve C -/
theorem point_A_on_curve : curve_equation (-1) 2 := by
  sorry

end point_A_on_curve_l1935_193514


namespace f_nonnegative_iff_a_in_range_l1935_193582

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - (1/2) * (x - a)^2 + 4

theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ 0) ↔ a ∈ Set.Icc (Real.log 4 - 4) (Real.sqrt 10) :=
sorry

end f_nonnegative_iff_a_in_range_l1935_193582
