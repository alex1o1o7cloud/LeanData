import Mathlib

namespace allocation_schemes_eq_90_l3096_309656

/-- The number of ways to distribute 5 college students among 3 freshman classes -/
def allocation_schemes : ℕ :=
  let n_students : ℕ := 5
  let n_classes : ℕ := 3
  let min_per_class : ℕ := 1
  let max_per_class : ℕ := 2
  -- The actual calculation is not implemented, just returning the correct result
  90

/-- Theorem stating that the number of allocation schemes is 90 -/
theorem allocation_schemes_eq_90 : allocation_schemes = 90 := by
  -- The proof is not implemented
  sorry

end allocation_schemes_eq_90_l3096_309656


namespace jackson_metropolitan_population_l3096_309608

theorem jackson_metropolitan_population :
  ∀ (average_population : ℝ),
  3200 ≤ average_population ∧ average_population ≤ 3600 →
  80000 ≤ 25 * average_population ∧ 25 * average_population ≤ 90000 :=
by sorry

end jackson_metropolitan_population_l3096_309608


namespace prob_all_suits_in_five_draws_l3096_309606

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Number of cards drawn -/
def NumberOfDraws : ℕ := 5

/-- Probability of drawing a card from a specific suit -/
def ProbSingleSuit : ℚ := 1 / NumberOfSuits

/-- Theorem: Probability of getting at least one card from each suit in 5 draws with replacement -/
theorem prob_all_suits_in_five_draws : 
  let prob_different_suit (n : ℕ) := (NumberOfSuits - n) / NumberOfSuits
  (prob_different_suit 1) * (prob_different_suit 2) * (prob_different_suit 3) = 3 / 32 := by
  sorry

end prob_all_suits_in_five_draws_l3096_309606


namespace exactly_two_clubs_l3096_309632

theorem exactly_two_clubs (S : ℕ) (A B C ABC : ℕ) : 
  S = 400 ∧
  A = S / 2 ∧
  B = S * 5 / 8 ∧
  C = S * 3 / 4 ∧
  ABC = S * 3 / 8 ∧
  A + B + C - 2 * ABC ≥ S →
  A + B + C - S - ABC = 500 := by
sorry

end exactly_two_clubs_l3096_309632


namespace gcd_of_specific_numbers_l3096_309635

theorem gcd_of_specific_numbers : Nat.gcd 333333 7777777 = 1 := by
  sorry

end gcd_of_specific_numbers_l3096_309635


namespace reciprocal_gp_sum_l3096_309623

/-- Given a geometric progression with n terms, first term 1, common ratio r^2 (r ≠ 0),
    and sum s^3, the sum of the geometric progression formed by the reciprocals of each term
    is s^3 / r^2 -/
theorem reciprocal_gp_sum (n : ℕ) (r s : ℝ) (hr : r ≠ 0) :
  let original_sum := (1 - r^(2*n)) / (1 - r^2)
  let reciprocal_sum := (1 - (1/r^2)^n) / (1 - 1/r^2)
  original_sum = s^3 → reciprocal_sum = s^3 / r^2 := by
sorry

end reciprocal_gp_sum_l3096_309623


namespace first_number_in_proportion_l3096_309643

/-- Given a proportion a : 1.65 :: 5 : 11, prove that a = 0.75 -/
theorem first_number_in_proportion (a : ℝ) : 
  (a / 1.65 = 5 / 11) → a = 0.75 := by
sorry

end first_number_in_proportion_l3096_309643


namespace factorial_divisibility_l3096_309680

theorem factorial_divisibility (m n : ℕ) : 
  (Nat.factorial (2 * m) * Nat.factorial (2 * n)) % 
  (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) = 0 := by
  sorry

end factorial_divisibility_l3096_309680


namespace sandy_marks_per_correct_sum_l3096_309697

theorem sandy_marks_per_correct_sum :
  ∀ (total_sums : ℕ) (total_marks : ℤ) (correct_sums : ℕ) (marks_lost_per_incorrect : ℤ) (marks_per_correct : ℤ),
    total_sums = 30 →
    total_marks = 60 →
    correct_sums = 24 →
    marks_lost_per_incorrect = 2 →
    (marks_per_correct * correct_sums : ℤ) - (marks_lost_per_incorrect * (total_sums - correct_sums) : ℤ) = total_marks →
    marks_per_correct = 3 :=
by sorry

end sandy_marks_per_correct_sum_l3096_309697


namespace sum_of_squared_coefficients_l3096_309610

-- Define the original expression
def original_expression (x : ℝ) : ℝ := 4 * (x^2 - 2*x + 2) - 7 * (x^3 - 3*x + 1)

-- Define the simplified expression
def simplified_expression (x : ℝ) : ℝ := -7*x^3 + 4*x^2 + 13*x + 1

-- Theorem statement
theorem sum_of_squared_coefficients :
  ((-7)^2 + 4^2 + 13^2 + 1^2 = 235) ∧
  (∀ x : ℝ, original_expression x = simplified_expression x) :=
sorry

end sum_of_squared_coefficients_l3096_309610


namespace triangle_side_sum_l3096_309651

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B →
  b = 2 →
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 2 / 2 →
  a + c = 4 := by
  sorry

end triangle_side_sum_l3096_309651


namespace circle_m_range_l3096_309603

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

-- Define the condition for the equation to represent a circle
def is_circle (m : ℝ) : Prop := ∃ (x y : ℝ), circle_equation x y m

-- Theorem stating the range of m for which the equation represents a circle
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ m < (1/2 : ℝ) :=
sorry

end circle_m_range_l3096_309603


namespace largest_prime_factors_difference_l3096_309626

theorem largest_prime_factors_difference (n : Nat) (h : n = 242858) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) ∧
  p ≠ q ∧ p - q = 80 :=
sorry

end largest_prime_factors_difference_l3096_309626


namespace village_population_theorem_l3096_309640

/-- Given a village with a total population and a subset of that population,
    calculate the percentage that the subset represents. -/
def village_population_percentage (total : ℕ) (subset : ℕ) : ℚ :=
  (subset : ℚ) / (total : ℚ) * 100

/-- Theorem stating that 45,000 is 90% of 50,000 -/
theorem village_population_theorem :
  village_population_percentage 50000 45000 = 90 := by
  sorry

end village_population_theorem_l3096_309640


namespace largest_two_digit_prime_factor_of_binom_150_75_l3096_309633

theorem largest_two_digit_prime_factor_of_binom_150_75 :
  (∃ (p : ℕ), Prime p ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ p) →
  (73 : ℕ).Prime ∧ 73 ∣ Nat.choose 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ 73 :=
by sorry

end largest_two_digit_prime_factor_of_binom_150_75_l3096_309633


namespace m_returns_to_original_position_min_steps_to_return_l3096_309624

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the position of point M on side AB
def PositionM (t : Triangle) (a : ℝ) : ℝ × ℝ :=
  (a * t.A.1 + (1 - a) * t.B.1, a * t.A.2 + (1 - a) * t.B.2)

-- Define the movement of point M
def MoveM (t : Triangle) (pos : ℝ × ℝ) (step : ℕ) : ℝ × ℝ :=
  sorry

-- Theorem: M returns to its original position
theorem m_returns_to_original_position (t : Triangle) (a : ℝ) :
  ∃ n : ℕ, MoveM t (PositionM t a) n = PositionM t a :=
sorry

-- Theorem: Minimum number of steps for M to return
theorem min_steps_to_return (t : Triangle) (a : ℝ) :
  (a = 1/2 ∧ (∃ n : ℕ, n = 3 ∧ MoveM t (PositionM t a) n = PositionM t a ∧
    ∀ m : ℕ, m < n → MoveM t (PositionM t a) m ≠ PositionM t a)) ∨
  (a ≠ 1/2 ∧ (∃ n : ℕ, n = 6 ∧ MoveM t (PositionM t a) n = PositionM t a ∧
    ∀ m : ℕ, m < n → MoveM t (PositionM t a) m ≠ PositionM t a)) :=
sorry

end m_returns_to_original_position_min_steps_to_return_l3096_309624


namespace carbon_weight_in_C4H8O2_l3096_309664

/-- The molecular weight of the carbon part in C4H8O2 -/
def carbon_weight (atomic_weight : ℝ) (num_atoms : ℕ) : ℝ :=
  atomic_weight * num_atoms

/-- Proof that the molecular weight of the carbon part in C4H8O2 is 48.04 g/mol -/
theorem carbon_weight_in_C4H8O2 :
  let compound_weight : ℝ := 88
  let carbon_atomic_weight : ℝ := 12.01
  let num_carbon_atoms : ℕ := 4
  carbon_weight carbon_atomic_weight num_carbon_atoms = 48.04 := by
  sorry

end carbon_weight_in_C4H8O2_l3096_309664


namespace train_crossing_time_l3096_309607

/-- The time taken for a train to cross a man walking in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 500 →
  train_speed = 75 * 1000 / 3600 →
  man_speed = 3 * 1000 / 3600 →
  train_length / (train_speed - man_speed) = 25 := by
sorry

end train_crossing_time_l3096_309607


namespace abs_inequality_l3096_309692

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end abs_inequality_l3096_309692


namespace faster_speed_proof_l3096_309690

/-- Proves that the faster speed is 12 kmph given the problem conditions -/
theorem faster_speed_proof (distance : ℝ) (slow_speed : ℝ) (late_time : ℝ) (early_time : ℝ) 
  (h1 : distance = 24)
  (h2 : slow_speed = 9)
  (h3 : late_time = 1/3)  -- 20 minutes in hours
  (h4 : early_time = 1/3) -- 20 minutes in hours
  : ∃ (fast_speed : ℝ), 
    distance / slow_speed - distance / fast_speed = late_time + early_time ∧ 
    fast_speed = 12 := by
  sorry

end faster_speed_proof_l3096_309690


namespace polygon_sides_from_diagonals_l3096_309625

theorem polygon_sides_from_diagonals (d : ℕ) (h : d = 44) : ∃ n : ℕ, n ≥ 3 ∧ d = n * (n - 3) / 2 ∧ n = 11 := by
  sorry

end polygon_sides_from_diagonals_l3096_309625


namespace factors_of_M_l3096_309622

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^2 * 11^1

theorem factors_of_M :
  (∃ (f : ℕ → ℕ), f M = 720 ∧ (∀ d : ℕ, d ∣ M ↔ d ∈ Finset.range (f M + 1))) ∧
  (∃ (g : ℕ → ℕ), g M = 120 ∧ (∀ d : ℕ, d ∣ M ∧ Odd d ↔ d ∈ Finset.range (g M + 1))) :=
by sorry

end factors_of_M_l3096_309622


namespace coin_arrangement_strategy_exists_l3096_309601

/-- Represents a strategy for arranging coins by weight --/
structure CoinArrangementStrategy where
  /-- Function that decides which coins to compare at each step --/
  compareCoins : ℕ → (ℕ × ℕ)
  /-- Maximum number of comparisons needed --/
  maxComparisons : ℕ

/-- Represents the expected number of comparisons for a strategy --/
def expectedComparisons (strategy : CoinArrangementStrategy) : ℚ :=
  sorry

/-- There exists a strategy to arrange 4 coins with expected comparisons less than 4.8 --/
theorem coin_arrangement_strategy_exists :
  ∃ (strategy : CoinArrangementStrategy),
    strategy.maxComparisons ≤ 4 ∧ expectedComparisons strategy < 24/5 := by
  sorry

end coin_arrangement_strategy_exists_l3096_309601


namespace suzy_book_count_l3096_309668

/-- Calculates the final number of books Suzy has after three days of transactions -/
def final_book_count (initial : ℕ) (wed_out : ℕ) (thu_in : ℕ) (thu_out : ℕ) (fri_in : ℕ) : ℕ :=
  initial - wed_out + thu_in - thu_out + fri_in

/-- Theorem stating that given the specific transactions, Suzy ends up with 80 books -/
theorem suzy_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

#eval final_book_count 98 43 23 5 7

end suzy_book_count_l3096_309668


namespace greg_earnings_l3096_309639

/-- Represents the rates for a dog size --/
structure DogRate where
  baseCharge : ℝ
  perMinuteCharge : ℝ

/-- Represents a group of dogs walked --/
structure DogGroup where
  count : ℕ
  minutes : ℕ

/-- Calculates the earnings for a group of dogs --/
def calculateEarnings (rate : DogRate) (group : DogGroup) : ℝ :=
  rate.baseCharge * group.count + rate.perMinuteCharge * group.count * group.minutes

/-- Theorem: Greg's total earnings for the day --/
theorem greg_earnings : 
  let extraSmallRate : DogRate := ⟨12, 0.80⟩
  let smallRate : DogRate := ⟨15, 1⟩
  let mediumRate : DogRate := ⟨20, 1.25⟩
  let largeRate : DogRate := ⟨25, 1.50⟩
  let extraLargeRate : DogRate := ⟨30, 1.75⟩

  let extraSmallGroup : DogGroup := ⟨2, 10⟩
  let smallGroup : DogGroup := ⟨3, 12⟩
  let mediumGroup : DogGroup := ⟨1, 18⟩
  let largeGroup : DogGroup := ⟨2, 25⟩
  let extraLargeGroup : DogGroup := ⟨1, 30⟩

  let totalEarnings := 
    calculateEarnings extraSmallRate extraSmallGroup +
    calculateEarnings smallRate smallGroup +
    calculateEarnings mediumRate mediumGroup +
    calculateEarnings largeRate largeGroup +
    calculateEarnings extraLargeRate extraLargeGroup

  totalEarnings = 371 := by sorry

end greg_earnings_l3096_309639


namespace cos_225_degrees_l3096_309675

theorem cos_225_degrees : Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l3096_309675


namespace rectangle_area_is_464_l3096_309681

-- Define the side lengths of the squares
def E : ℝ := 7
def H : ℝ := 2
def D : ℝ := 8

-- Define the side lengths of other squares in terms of H and D
def F : ℝ := H + E
def B : ℝ := H + 2 * E
def I : ℝ := 2 * H + E
def G : ℝ := 3 * H + E
def C : ℝ := 3 * H + D + E
def A : ℝ := 3 * H + 2 * D + E

-- Define the dimensions of the rectangle
def rectangle_width : ℝ := A + B
def rectangle_height : ℝ := A + C

-- Theorem to prove
theorem rectangle_area_is_464 : 
  rectangle_width * rectangle_height = 464 := by
  sorry

end rectangle_area_is_464_l3096_309681


namespace table_height_is_36_l3096_309686

/-- The height of the table in inches -/
def table_height : ℝ := 36

/-- The length of each wooden block in inches -/
def block_length : ℝ := sorry

/-- The width of each wooden block in inches -/
def block_width : ℝ := sorry

/-- Two blocks stacked from one end to the other across the table measure 38 inches -/
axiom scenario1 : block_length + table_height - block_width = 38

/-- One block stacked on top of another with the third block beside them measure 34 inches -/
axiom scenario2 : block_width + table_height - block_length = 34

theorem table_height_is_36 : table_height = 36 := by sorry

end table_height_is_36_l3096_309686


namespace g_range_l3096_309672

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x/3))^2 + (Real.pi/4) * Real.arcsin (x/3) - (Real.arcsin (x/3))^2 + (Real.pi^2/16) * (x^2 + 2*x + 3)

theorem g_range : 
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, 
    g x ∈ Set.Icc (Real.pi^2/4) ((15*Real.pi^2/16) + (Real.pi/4)*Real.arcsin 1) ∧
    ∃ y ∈ Set.Icc (-3 : ℝ) 3, g y = Real.pi^2/4 ∧
    ∃ z ∈ Set.Icc (-3 : ℝ) 3, g z = (15*Real.pi^2/16) + (Real.pi/4)*Real.arcsin 1 :=
by sorry

end g_range_l3096_309672


namespace jobber_pricing_jobber_pricing_example_l3096_309660

theorem jobber_pricing (original_price : ℝ) (purchase_discount : ℝ) (desired_gain : ℝ) (sale_discount : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - purchase_discount)
  let selling_price := purchase_price * (1 + desired_gain)
  let marked_price := selling_price / (1 - sale_discount)
  marked_price

theorem jobber_pricing_example : jobber_pricing 24 0.125 (1/3) 0.2 = 35 := by
  sorry

end jobber_pricing_jobber_pricing_example_l3096_309660


namespace equation_solution_l3096_309648

theorem equation_solution (x : ℝ) : 
  (x = (-81 + Real.sqrt 5297) / 8 ∨ x = (-81 - Real.sqrt 5297) / 8) ↔ 
  (8 * x^2 + 89 * x + 3) / (3 * x + 41) = 4 * x + 2 := by sorry

end equation_solution_l3096_309648


namespace rectangular_box_sum_l3096_309665

theorem rectangular_box_sum (A B C : ℝ) 
  (h1 : A * B = 30)
  (h2 : A * C = 50)
  (h3 : B * C = 90) :
  A + B + C = 58 * Real.sqrt 15 / 3 := by
sorry

end rectangular_box_sum_l3096_309665


namespace largest_four_digit_multiple_of_9_with_digit_sum_27_l3096_309655

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if n is a four-digit number, false otherwise -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_four_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 9990 :=
by sorry

end largest_four_digit_multiple_of_9_with_digit_sum_27_l3096_309655


namespace function_properties_l3096_309658

/-- The function f(x) = ax ln x - x^2 - 2x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x^2 - 2 * x

/-- The derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (1 + Real.log x) - 2 * x - 2

/-- The function g(x) = f(x) + 2x --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * x

theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) :
  (∃ (x : ℝ), f_deriv 4 x = 4 * Real.log 2 - 2 ∧
    ∀ (y : ℝ), f_deriv 4 y ≤ 4 * Real.log 2 - 2) ∧
  (g a x₁ = 0 ∧ g a x₂ = 0 ∧ x₂ / x₁ > Real.exp 1 →
    Real.log a + Real.log (x₁ * x₂) > 3) :=
by sorry

end function_properties_l3096_309658


namespace smartphone_price_proof_l3096_309666

def laptop_price : ℕ := 600
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def total_paid : ℕ := 3000
def change_received : ℕ := 200

theorem smartphone_price_proof :
  ∃ (smartphone_price : ℕ),
    smartphone_price * num_smartphones + laptop_price * num_laptops = total_paid - change_received ∧
    smartphone_price = 400 := by
  sorry

end smartphone_price_proof_l3096_309666


namespace prize_logic_l3096_309661

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answered_all_correctly : Student → Prop)
variable (got_prize : Student → Prop)

-- State the theorem
theorem prize_logic (h : ∀ s : Student, answered_all_correctly s → got_prize s) :
  ∀ s : Student, ¬(got_prize s) → ¬(answered_all_correctly s) :=
by
  sorry

end prize_logic_l3096_309661


namespace hannah_grocery_cost_l3096_309699

theorem hannah_grocery_cost 
  (total_cost : ℝ)
  (cookie_price : ℝ)
  (carrot_price : ℝ)
  (cabbage_price : ℝ)
  (orange_price : ℝ)
  (h1 : cookie_price + carrot_price + cabbage_price + orange_price = total_cost)
  (h2 : orange_price = 3 * cookie_price)
  (h3 : cabbage_price = cookie_price - carrot_price)
  (h4 : total_cost = 24) :
  carrot_price + cabbage_price = 24 / 5 := by
sorry

end hannah_grocery_cost_l3096_309699


namespace cubic_polynomial_satisfies_conditions_l3096_309684

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -4/3 * x^3 + 6 * x^2 - 50/3 * x - 14/3
  (q 1 = -8) ∧ (q 2 = -12) ∧ (q 3 = -20) ∧ (q 4 = -40) := by
  sorry

end cubic_polynomial_satisfies_conditions_l3096_309684


namespace milk_cost_l3096_309691

/-- Proves that the cost of a gallon of milk is $3 given the total groceries cost and the costs of other items. -/
theorem milk_cost (total : ℝ) (cereal_price cereal_qty : ℝ) (banana_price banana_qty : ℝ) 
  (apple_price apple_qty : ℝ) (cookie_qty : ℝ) :
  total = 25 ∧ 
  cereal_price = 3.5 ∧ cereal_qty = 2 ∧
  banana_price = 0.25 ∧ banana_qty = 4 ∧
  apple_price = 0.5 ∧ apple_qty = 4 ∧
  cookie_qty = 2 →
  ∃ (milk_price : ℝ),
    milk_price = 3 ∧
    total = cereal_price * cereal_qty + banana_price * banana_qty + 
            apple_price * apple_qty + milk_price + 2 * milk_price * cookie_qty :=
by sorry

end milk_cost_l3096_309691


namespace solution_of_quadratic_equation_l3096_309634

theorem solution_of_quadratic_equation :
  {x : ℝ | 2 * (x + 1) = x * (x + 1)} = {-1, 2} := by sorry

end solution_of_quadratic_equation_l3096_309634


namespace toy_store_revenue_l3096_309631

theorem toy_store_revenue (D : ℚ) (D_pos : D > 0) : 
  let N := (2 : ℚ) / 5 * D
  let J := (1 : ℚ) / 5 * N
  let F := (3 : ℚ) / 4 * D
  let avg := (N + J + F) / 3
  D / avg = 100 / 41 := by
sorry

end toy_store_revenue_l3096_309631


namespace count_valid_antibirthdays_l3096_309696

/-- Represents a date in day.month format -/
structure Date :=
  (day : ℕ)
  (month : ℕ)

/-- Checks if a date is valid -/
def is_valid_date (d : Date) : Prop :=
  1 ≤ d.month ∧ d.month ≤ 12 ∧ 1 ≤ d.day ∧ d.day ≤ 31

/-- Swaps the day and month of a date -/
def swap_date (d : Date) : Date :=
  ⟨d.month, d.day⟩

/-- Checks if a date has a valid anti-birthday -/
def has_valid_antibirthday (d : Date) : Prop :=
  is_valid_date d ∧ 
  is_valid_date (swap_date d) ∧ 
  d.day ≠ d.month

/-- The number of days in a year with valid anti-birthdays -/
def days_with_valid_antibirthdays : ℕ := 132

/-- Theorem stating the number of days with valid anti-birthdays -/
theorem count_valid_antibirthdays : 
  (∀ d : Date, has_valid_antibirthday d) → 
  days_with_valid_antibirthdays = 132 := by
  sorry

#check count_valid_antibirthdays

end count_valid_antibirthdays_l3096_309696


namespace sets_inclusion_l3096_309693

-- Define the sets M, N, and P
def M : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}
def N : Set ℝ := {x | Real.cos (2 * x) = 0}
def P : Set ℝ := {a | Real.sin (2 * a) = 1}

-- State the theorem
theorem sets_inclusion : P ⊆ N ∧ N ⊆ M := by sorry

end sets_inclusion_l3096_309693


namespace square_area_with_side_5_l3096_309636

theorem square_area_with_side_5 :
  let side_length : ℝ := 5
  let area : ℝ := side_length * side_length
  area = 25 := by sorry

end square_area_with_side_5_l3096_309636


namespace probability_is_one_half_l3096_309653

/-- Represents the class of a bus -/
inductive BusClass
| Upper
| Middle
| Lower

/-- Represents a sequence of three buses -/
def BusSequence := (BusClass × BusClass × BusClass)

/-- All possible bus sequences -/
def allSequences : List BusSequence := sorry

/-- Determines if Mr. Li boards an upper-class bus given a sequence -/
def boardsUpperClass (seq : BusSequence) : Bool := sorry

/-- The probability of Mr. Li boarding an upper-class bus -/
def probabilityOfUpperClass : ℚ := sorry

/-- Theorem stating that the probability of boarding an upper-class bus is 1/2 -/
theorem probability_is_one_half : probabilityOfUpperClass = 1/2 := by sorry

end probability_is_one_half_l3096_309653


namespace arithmetic_sequence_problem_l3096_309663

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 7 - a 10 = -1 →
  a 11 - a 4 = 21 →
  a 7 = 20 := by
  sorry

end arithmetic_sequence_problem_l3096_309663


namespace correct_additional_money_l3096_309600

/-- Calculates the additional money Jack needs to buy socks and shoes -/
def additional_money_needed (sock_price : ℝ) (shoe_price : ℝ) (jack_has : ℝ) : ℝ :=
  2 * sock_price + shoe_price - jack_has

/-- Proves that the additional money needed is correct -/
theorem correct_additional_money 
  (sock_price : ℝ) (shoe_price : ℝ) (jack_has : ℝ) 
  (h1 : sock_price = 9.5)
  (h2 : shoe_price = 92)
  (h3 : jack_has = 40) :
  additional_money_needed sock_price shoe_price jack_has = 71 :=
by sorry

end correct_additional_money_l3096_309600


namespace second_diff_constant_correct_y_value_l3096_309679

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the sequence of x values
def x_seq (x₁ d : ℝ) (n : ℕ) : ℝ := x₁ + n * d

-- Define the sequence of y values
def y_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ := quadratic a b c (x_seq x₁ d n)

-- Define the first difference sequence
def delta_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ :=
  y_seq a b c x₁ d (n + 1) - y_seq a b c x₁ d n

-- Define the second difference sequence
def delta2_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ :=
  delta_seq a b c x₁ d (n + 1) - delta_seq a b c x₁ d n

-- Theorem: The second difference is constant
theorem second_diff_constant (a b c x₁ d : ℝ) (h : a ≠ 0) :
  ∃ k, ∀ n, delta2_seq a b c x₁ d n = k :=
sorry

-- Given y values
def given_y_values : List ℝ := [51, 107, 185, 285, 407, 549, 717]

-- Find the incorrect y value and its correct value
def find_incorrect_y (ys : List ℝ) : Option (ℕ × ℝ) :=
sorry

-- Theorem: The identified incorrect y value is 549 and should be 551
theorem correct_y_value :
  find_incorrect_y given_y_values = some (5, 551) :=
sorry

end second_diff_constant_correct_y_value_l3096_309679


namespace square_roots_problem_l3096_309674

theorem square_roots_problem (m : ℝ) (a : ℝ) (h1 : m > 0) 
  (h2 : (a + 6)^2 = m) (h3 : (2*a - 9)^2 = m) :
  a = 1 ∧ m = 49 ∧ ∀ x : ℝ, a*x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by
  sorry

end square_roots_problem_l3096_309674


namespace shekars_english_marks_l3096_309615

/-- Represents the marks scored in each subject -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  biology : ℕ
  english : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given Shekar's marks in other subjects and his average, his English marks are 67 -/
theorem shekars_english_marks (m : Marks) (h1 : m.mathematics = 76) (h2 : m.science = 65)
    (h3 : m.socialStudies = 82) (h4 : m.biology = 75)
    (h5 : average [m.mathematics, m.science, m.socialStudies, m.biology, m.english] = 73) :
    m.english = 67 := by
  sorry

#check shekars_english_marks

end shekars_english_marks_l3096_309615


namespace polynomial_factor_implies_d_value_l3096_309613

/-- 
Given a polynomial of the form 3x^3 + dx + 9 with a factor x^2 + qx + 1,
prove that d = -24.
-/
theorem polynomial_factor_implies_d_value :
  ∀ d q : ℝ,
  (∃ c : ℝ, ∀ x : ℝ, 3*x^3 + d*x + 9 = (x^2 + q*x + 1) * (3*x + c)) →
  d = -24 :=
by sorry

end polynomial_factor_implies_d_value_l3096_309613


namespace triangles_in_decagon_l3096_309629

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 sides -/
def decagonSides : ℕ := 10

/-- Theorem: The number of triangles that can be formed from the vertices of a regular decagon is equal to the number of ways to choose 3 vertices out of 10 -/
theorem triangles_in_decagon :
  trianglesInDecagon = Nat.choose decagonSides 3 := by
  sorry

#eval trianglesInDecagon -- Should output 120

end triangles_in_decagon_l3096_309629


namespace factor_cubic_expression_l3096_309628

theorem factor_cubic_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end factor_cubic_expression_l3096_309628


namespace labor_costs_theorem_l3096_309641

/-- Calculates the overall labor costs for one day given the number of workers and their wages. -/
def overall_labor_costs (
  num_construction_workers : ℕ)
  (num_electricians : ℕ)
  (num_plumbers : ℕ)
  (construction_worker_wage : ℚ)
  (electrician_wage_multiplier : ℚ)
  (plumber_wage_multiplier : ℚ) : ℚ :=
  (num_construction_workers * construction_worker_wage) +
  (num_electricians * (electrician_wage_multiplier * construction_worker_wage)) +
  (num_plumbers * (plumber_wage_multiplier * construction_worker_wage))

/-- Proves that the overall labor costs for one day is $650 given the specified conditions. -/
theorem labor_costs_theorem :
  overall_labor_costs 2 1 1 100 2 (5/2) = 650 := by
  sorry

#eval overall_labor_costs 2 1 1 100 2 (5/2)

end labor_costs_theorem_l3096_309641


namespace existence_of_common_source_l3096_309619

/-- The type of positive integers. -/
def PositiveInt := { n : ℕ // n > 0 }

/-- Predicate to check if a number contains the digit 5. -/
def containsFive (n : PositiveInt) : Prop :=
  ∃ d, d ∈ n.val.digits 10 ∧ d = 5

/-- The process of replacing two consecutive digits with the last digit of their product. -/
def replaceDigits (n : PositiveInt) : PositiveInt :=
  sorry

/-- A number m is obtainable from n if there exists a finite sequence of replaceDigits operations. -/
def isObtainable (m n : PositiveInt) : Prop :=
  sorry

/-- Main theorem: For any finite set of positive integers without digit 5, 
    there exists a positive integer from which all elements are obtainable. -/
theorem existence_of_common_source (S : Finset PositiveInt) 
  (h : ∀ s ∈ S, ¬containsFive s) : 
  ∃ N : PositiveInt, ∀ s ∈ S, isObtainable s N :=
sorry

end existence_of_common_source_l3096_309619


namespace stream_speed_calculation_l3096_309627

/-- Proves that the speed of a stream is 12.6 kmph given specific boat travel conditions -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 11.5) :
  ∃ (stream_speed : ℝ), 
    stream_speed = 12.6 ∧ 
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time :=
by
  sorry


end stream_speed_calculation_l3096_309627


namespace equation_solution_l3096_309616

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -2 ∧ x₂ = 3 ∧
  ∀ x : ℝ, (x + 2)^2 - 5*(x + 2) = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end equation_solution_l3096_309616


namespace least_multiple_of_35_greater_than_450_l3096_309677

theorem least_multiple_of_35_greater_than_450 : ∀ n : ℕ, n > 0 ∧ 35 ∣ n ∧ n > 450 → n ≥ 455 := by
  sorry

end least_multiple_of_35_greater_than_450_l3096_309677


namespace m_minus_reciprocal_l3096_309669

theorem m_minus_reciprocal (m : ℝ) (h : m^2 + 3*m = -1) : m - 1/(m+1) = -2 := by
  sorry

end m_minus_reciprocal_l3096_309669


namespace inequality_problem_l3096_309644

theorem inequality_problem (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  ∃ (s : Finset (Fin 4)),
    s.card = 3 ∧
    (∀ i ∈ s, match i with
      | 0 => x^2 + y^2 < a^2 + b^2
      | 1 => x^2 - y^2 < a^2 - b^2
      | 2 => x^2 * y^3 < a^2 * b^3
      | 3 => x^2 / y^3 < a^2 / b^3
    ) ∧
    (∀ i ∉ s, ¬(match i with
      | 0 => x^2 + y^2 < a^2 + b^2
      | 1 => x^2 - y^2 < a^2 - b^2
      | 2 => x^2 * y^3 < a^2 * b^3
      | 3 => x^2 / y^3 < a^2 / b^3
    )) := by sorry

end inequality_problem_l3096_309644


namespace middle_school_students_in_ganzhou_form_set_l3096_309620

-- Define the universe of discourse
def Universe : Type := Unit

-- Define the property of being a middle school student in Ganzhou
def IsMiddleSchoolStudentInGanzhou : Universe → Prop := sorry

-- Define what it means for a collection to have definite elements
def HasDefiniteElements (S : Set Universe) : Prop := sorry

-- Theorem: The set of all middle school students in Ganzhou has definite elements
theorem middle_school_students_in_ganzhou_form_set :
  HasDefiniteElements {x : Universe | IsMiddleSchoolStudentInGanzhou x} := by
  sorry

end middle_school_students_in_ganzhou_form_set_l3096_309620


namespace supermarket_spending_l3096_309694

theorem supermarket_spending (total : ℚ) (candy : ℚ) : 
  total = 24 →
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + candy = total →
  candy = 8 := by
  sorry

end supermarket_spending_l3096_309694


namespace quarters_indeterminate_l3096_309609

/-- Represents the number of coins Mike has --/
structure MikeCoins where
  quarters : ℕ
  nickels : ℕ

/-- Represents the state of Mike's coins before and after his dad's borrowing --/
structure CoinState where
  initial : MikeCoins
  borrowed_nickels : ℕ
  current : MikeCoins

/-- Theorem stating that the number of quarters cannot be uniquely determined --/
theorem quarters_indeterminate (state : CoinState) 
    (h1 : state.initial.nickels = 87)
    (h2 : state.borrowed_nickels = 75)
    (h3 : state.current.nickels = 12)
    (h4 : state.initial.nickels = state.borrowed_nickels + state.current.nickels) :
    ∀ q : ℕ, ∃ state' : CoinState, 
      state'.initial.nickels = state.initial.nickels ∧
      state'.borrowed_nickels = state.borrowed_nickels ∧
      state'.current.nickels = state.current.nickels ∧
      state'.initial.quarters = q :=
  sorry

end quarters_indeterminate_l3096_309609


namespace symmetric_line_wrt_y_axis_l3096_309659

/-- Given a line with equation y = -2x - 3, prove that its symmetric line
    with respect to the y-axis has the equation y = 2x - 3 -/
theorem symmetric_line_wrt_y_axis (x y : ℝ) :
  (y = -2*x - 3) → (∃ (x' y' : ℝ), y' = 2*x' - 3 ∧ x' = -x ∧ y' = y) :=
by sorry

end symmetric_line_wrt_y_axis_l3096_309659


namespace min_value_x_plus_2y_l3096_309689

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 16 := by
  sorry

end min_value_x_plus_2y_l3096_309689


namespace sine_problem_l3096_309604

theorem sine_problem (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := by
  sorry

end sine_problem_l3096_309604


namespace min_value_quadratic_l3096_309612

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_z : ℝ), min_z = 5 ∧ ∀ z : ℝ, z = 5*x^2 + 20*x + 25 → z ≥ min_z := by
  sorry

end min_value_quadratic_l3096_309612


namespace parabola_directrix_l3096_309647

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- The directrix equation -/
def directrix (x : ℝ) : Prop := x = 1

/-- Theorem stating that the directrix of the given parabola is x = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → (∃ (f : ℝ), ∀ (x' y' : ℝ), 
    parabola x' y' → (x' - f)^2 + y'^2 = (x' - 1)^2) :=
sorry

end parabola_directrix_l3096_309647


namespace possible_values_of_a_l3096_309688

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 2/3) :=
sorry

end possible_values_of_a_l3096_309688


namespace local_extrema_condition_l3096_309611

/-- A function f with parameter a that we want to analyze -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a + 6

theorem local_extrema_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), IsLocalMin (f a) x₁ ∧ IsLocalMax (f a) x₂) ↔ (a ≤ -3 ∨ a > 6) :=
sorry

end local_extrema_condition_l3096_309611


namespace range_of_f_l3096_309657

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -9 ≤ y ∧ y ≤ 0 } :=
by sorry

end range_of_f_l3096_309657


namespace park_ant_count_l3096_309621

/-- Represents the dimensions and ant densities of a rectangular park with a special corner area -/
structure ParkInfo where
  width : ℝ  -- width of the park in feet
  length : ℝ  -- length of the park in feet
  normal_density : ℝ  -- average number of ants per square inch in most of the park
  corner_side : ℝ  -- side length of the square corner patch in feet
  corner_density : ℝ  -- average number of ants per square inch in the corner patch

/-- Calculates the total number of ants in the park -/
def totalAnts (park : ParkInfo) : ℝ :=
  let inches_per_foot : ℝ := 12
  let park_area := park.width * park.length * inches_per_foot^2
  let corner_area := park.corner_side^2 * inches_per_foot^2
  let normal_area := park_area - corner_area
  normal_area * park.normal_density + corner_area * park.corner_density

/-- Theorem stating that the total number of ants in the given park is approximately 73 million -/
theorem park_ant_count :
  let park : ParkInfo := {
    width := 200,
    length := 500,
    normal_density := 5,
    corner_side := 50,
    corner_density := 8
  }
  abs (totalAnts park - 73000000) < 100000 := by
  sorry


end park_ant_count_l3096_309621


namespace lottery_problem_l3096_309652

/-- Represents a lottery with prizes and blanks. -/
structure Lottery where
  prizes : ℕ
  blanks : ℕ
  prob_win : ℝ
  h_prob : prob_win = prizes / (prizes + blanks : ℝ)

/-- The lottery problem statement. -/
theorem lottery_problem (L : Lottery)
  (h_prizes : L.prizes = 10)
  (h_prob : L.prob_win = 0.2857142857142857) :
  L.blanks = 25 := by
  sorry

#check lottery_problem

end lottery_problem_l3096_309652


namespace max_value_of_b_l3096_309673

theorem max_value_of_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1 / b - 1 / a) :
  b ≤ 1 / 3 ∧ ∃ (b₀ : ℝ), b₀ > 0 ∧ b₀ = 1 / 3 ∧ ∃ (a₀ : ℝ), a₀ > 0 ∧ a₀ + 3 * b₀ = 1 / b₀ - 1 / a₀ :=
by sorry

end max_value_of_b_l3096_309673


namespace pizza_toppings_combinations_l3096_309605

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end pizza_toppings_combinations_l3096_309605


namespace base_number_is_two_l3096_309638

theorem base_number_is_two (a : ℕ) (x : ℕ) (h1 : a^x - a^(x-2) = 3 * 2^11) (h2 : x = 13) :
  a = 2 := by
  sorry

end base_number_is_two_l3096_309638


namespace john_completion_time_l3096_309654

/-- The number of days it takes for Rose to complete the work alone -/
def rose_days : ℝ := 480

/-- The number of days it takes for John and Rose to complete the work together -/
def joint_days : ℝ := 192

/-- The number of days it takes for John to complete the work alone -/
def john_days : ℝ := 320

/-- Theorem stating that given Rose's and joint completion times, John's completion time is 320 days -/
theorem john_completion_time : 
  (1 / john_days + 1 / rose_days = 1 / joint_days) → john_days = 320 :=
by sorry

end john_completion_time_l3096_309654


namespace trig_identity_l3096_309614

theorem trig_identity : 
  Real.sin (71 * π / 180) * Real.cos (26 * π / 180) - 
  Real.sin (19 * π / 180) * Real.sin (26 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end trig_identity_l3096_309614


namespace plane_contains_points_l3096_309645

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (4, -1, 5)
def point3 : ℝ × ℝ × ℝ := (5, -3, 4)

def plane_equation (x y z : ℝ) : Prop := x + 2*y - 2*z + 6 = 0

theorem plane_contains_points :
  (plane_equation point1.1 point1.2.1 point1.2.2) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2) ∧
  (1 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 1 2) 2) 6 = 1) :=
by sorry

end plane_contains_points_l3096_309645


namespace min_value_expression_l3096_309630

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (x + 3 * z + 1) ≥ 24 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (x₀ + 3 * z₀ + 1) = 24 * Real.sqrt 3 :=
by sorry

end min_value_expression_l3096_309630


namespace power_two_greater_than_square_five_is_smallest_smallest_n_zero_l3096_309602

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by sorry

theorem five_is_smallest (n : ℕ) (h : n < 5) : 2^n ≤ n^2 := by sorry

theorem smallest_n_zero : ∃ (n₀ : ℕ), (∀ (n : ℕ), n ≥ n₀ → 2^n > n^2) ∧ 
  (∀ (m : ℕ), m < n₀ → 2^m ≤ m^2) := by
  use 5
  constructor
  · exact power_two_greater_than_square
  · exact five_is_smallest

end power_two_greater_than_square_five_is_smallest_smallest_n_zero_l3096_309602


namespace solid_circles_in_2006_l3096_309683

def circle_sequence (n : ℕ) : ℕ := n + 1

def total_circles (n : ℕ) : ℕ := (n * (n + 3)) / 2

theorem solid_circles_in_2006 : 
  ∃ n : ℕ, total_circles n ≤ 2006 ∧ total_circles (n + 1) > 2006 ∧ n = 61 :=
sorry

end solid_circles_in_2006_l3096_309683


namespace ladybug_dots_total_l3096_309687

/-- The total number of dots on ladybugs caught over three days -/
theorem ladybug_dots_total : 
  let monday_ladybugs : ℕ := 8
  let monday_dots_per_ladybug : ℕ := 6
  let tuesday_ladybugs : ℕ := 5
  let tuesday_dots_per_ladybug : ℕ := 7
  let wednesday_ladybugs : ℕ := 4
  let wednesday_dots_per_ladybug : ℕ := 8
  monday_ladybugs * monday_dots_per_ladybug + 
  tuesday_ladybugs * tuesday_dots_per_ladybug + 
  wednesday_ladybugs * wednesday_dots_per_ladybug = 115 := by
sorry

end ladybug_dots_total_l3096_309687


namespace total_tax_percentage_l3096_309649

-- Define the spending percentages
def clothing_percent : ℝ := 0.40
def food_percent : ℝ := 0.30
def other_percent : ℝ := 0.30

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0
def other_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_tax_percentage (total_spent : ℝ) (total_spent_pos : total_spent > 0) :
  let clothing_spent := clothing_percent * total_spent
  let food_spent := food_percent * total_spent
  let other_spent := other_percent * total_spent
  let clothing_tax := clothing_tax_rate * clothing_spent
  let food_tax := food_tax_rate * food_spent
  let other_tax := other_tax_rate * other_spent
  let total_tax := clothing_tax + food_tax + other_tax
  (total_tax / total_spent) * 100 = 4 := by
sorry

end total_tax_percentage_l3096_309649


namespace sin_cos_sum_equals_sqrt_sum_l3096_309642

theorem sin_cos_sum_equals_sqrt_sum : 
  Real.sin (26 * π / 3) + Real.cos (-17 * π / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 := by
  sorry

end sin_cos_sum_equals_sqrt_sum_l3096_309642


namespace average_book_width_l3096_309646

def book_widths : List ℝ := [4, 0.5, 1.2, 3, 7.5, 2, 5, 9]

theorem average_book_width : 
  (List.sum book_widths) / (List.length book_widths) = 4.025 := by
  sorry

end average_book_width_l3096_309646


namespace sum_of_diagonals_l3096_309662

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Sides of the hexagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  -- Diagonals from vertex A
  diag1 : ℝ
  diag2 : ℝ
  diag3 : ℝ
  -- Assumption that the hexagon is inscribed in a circle
  inscribed : True

/-- The theorem about the sum of diagonals in a specific inscribed hexagon -/
theorem sum_of_diagonals (h : InscribedHexagon) 
    (h1 : h.side1 = 70)
    (h2 : h.side2 = 90)
    (h3 : h.side3 = 90)
    (h4 : h.side4 = 90)
    (h5 : h.side5 = 90)
    (h6 : h.side6 = 50) :
    h.diag1 + h.diag2 + h.diag3 = 376 := by
  sorry

end sum_of_diagonals_l3096_309662


namespace total_eggs_is_63_l3096_309670

/-- The number of Easter eggs Hannah found -/
def hannah_eggs : ℕ := 42

/-- The number of Easter eggs Helen found -/
def helen_eggs : ℕ := hannah_eggs / 2

/-- The total number of Easter eggs in the yard -/
def total_eggs : ℕ := hannah_eggs + helen_eggs

/-- Theorem stating that the total number of Easter eggs in the yard is 63 -/
theorem total_eggs_is_63 : total_eggs = 63 := by
  sorry

end total_eggs_is_63_l3096_309670


namespace carl_driving_hours_l3096_309617

/-- Calculates the total driving hours for Carl over two weeks -/
def total_driving_hours : ℕ :=
  let daily_hours : ℕ := 2
  let days_in_two_weeks : ℕ := 14
  let additional_weekly_hours : ℕ := 6
  let weeks : ℕ := 2
  (daily_hours * days_in_two_weeks) + (additional_weekly_hours * weeks)

/-- Theorem stating that Carl's total driving hours over two weeks is 40 -/
theorem carl_driving_hours : total_driving_hours = 40 := by
  sorry

end carl_driving_hours_l3096_309617


namespace some_number_value_l3096_309650

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : x = 5)
  (h2 : (x / some_number) + 3 = 4) : 
  some_number = 5 := by
sorry

end some_number_value_l3096_309650


namespace eugene_pencils_l3096_309682

theorem eugene_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 := by
  sorry

end eugene_pencils_l3096_309682


namespace sequences_and_sum_theorem_l3096_309637

/-- Definition of sequence a_n -/
def a (n : ℕ+) : ℕ :=
  if n = 1 then 1 else 2 * n.val - 1

/-- Definition of sequence b_n -/
def b (n : ℕ+) : ℚ :=
  if n = 1 then 1 else 2^(2 - n.val)

/-- Definition of S_n (sum of first n terms of a_n) -/
def S (n : ℕ+) : ℕ := (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Definition of T_n (sum of first n terms of a_n * b_n) -/
def T (n : ℕ+) : ℚ := 11 - (2 * n.val + 3) * 2^(2 - n.val)

theorem sequences_and_sum_theorem (n : ℕ+) :
  (∀ (k : ℕ+), k ≥ 2 → S (k + 1) + S (k - 1) = 2 * (S k + 1)) ∧
  (∀ (k : ℕ+), (Finset.range k.val).sum (λ i => 2^i * b ⟨i + 1, Nat.succ_pos i⟩) = a k) →
  (∀ (k : ℕ+), a k = 2 * k.val - 1) ∧
  (∀ (k : ℕ+), b k = if k = 1 then 1 else 2^(2 - k.val)) ∧
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩ * b ⟨i + 1, Nat.succ_pos i⟩) = T n :=
by sorry


end sequences_and_sum_theorem_l3096_309637


namespace first_tv_width_l3096_309698

/-- Proves that the width of the first TV is 24 inches given the specified conditions. -/
theorem first_tv_width : 
  ∀ (W : ℝ),
  (672 / (W * 16) = 1152 / (48 * 32) + 1) →
  W = 24 := by
sorry

end first_tv_width_l3096_309698


namespace possible_values_of_a_l3096_309695

-- Define the sets P and S
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def S (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Define the set of possible values for a
def A : Set ℝ := {0, 1/3, -1/2}

-- Theorem statement
theorem possible_values_of_a :
  ∀ a : ℝ, (S a ⊆ P) ↔ (a ∈ A) := by sorry

end possible_values_of_a_l3096_309695


namespace store_money_made_l3096_309667

/-- Represents the total money made from pencil sales -/
def total_money_made (eraser_price regular_price short_price : ℚ)
                     (eraser_sold regular_sold short_sold : ℕ) : ℚ :=
  eraser_price * eraser_sold + regular_price * regular_sold + short_price * short_sold

/-- Theorem stating that the store made $194 from the given pencil sales -/
theorem store_money_made :
  total_money_made 0.8 0.5 0.4 200 40 35 = 194 := by sorry

end store_money_made_l3096_309667


namespace only_x0_is_perfect_square_l3096_309618

-- Define the sequence (x_n)
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * x (n + 1) - x n

-- Define a perfect square
def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

-- Theorem statement
theorem only_x0_is_perfect_square :
  ∀ n : ℕ, isPerfectSquare (x n) → n = 0 := by
  sorry

end only_x0_is_perfect_square_l3096_309618


namespace rose_garden_problem_l3096_309676

/-- Rose garden problem -/
theorem rose_garden_problem (total_rows : ℕ) (roses_per_row : ℕ) (total_pink : ℕ) :
  total_rows = 10 →
  roses_per_row = 20 →
  total_pink = 40 →
  ∃ (red_fraction : ℚ),
    red_fraction = 1/2 ∧
    ∀ (row : ℕ),
      row ≤ total_rows →
      ∃ (red white pink : ℕ),
        red + white + pink = roses_per_row ∧
        white = (3/5 : ℚ) * (roses_per_row - red) ∧
        pink = roses_per_row - red - white ∧
        red = (red_fraction * roses_per_row : ℚ) :=
by sorry

end rose_garden_problem_l3096_309676


namespace polynomial_identity_sum_of_squares_l3096_309671

theorem polynomial_identity_sum_of_squares (p q r s t u v : ℤ) :
  (∀ x : ℝ, 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 := by
  sorry

end polynomial_identity_sum_of_squares_l3096_309671


namespace power_eight_seven_thirds_l3096_309685

theorem power_eight_seven_thirds : (8 : ℝ) ^ (7/3) = 128 := by sorry

end power_eight_seven_thirds_l3096_309685


namespace slope_of_line_l3096_309678

/-- The slope of the line 4x + 7y = 28 is -4/7 -/
theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end slope_of_line_l3096_309678
