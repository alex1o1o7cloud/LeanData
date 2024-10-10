import Mathlib

namespace tangent_line_equation_l1736_173644

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem: The equation of the tangent line to f(x) at (1, 1) is 2x - y - 1 = 0
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2 * x - y - 1 = 0) :=
by sorry

end tangent_line_equation_l1736_173644


namespace painting_cost_tripled_l1736_173633

/-- Cost of painting a room's walls -/
structure PaintingCost where
  length : ℝ
  breadth : ℝ
  height : ℝ
  cost : ℝ

/-- Theorem: Cost of painting a room 3 times larger -/
theorem painting_cost_tripled (room : PaintingCost) (h : room.cost = 350) :
  let tripled_room := PaintingCost.mk (3 * room.length) (3 * room.breadth) (3 * room.height) 0
  tripled_room.cost = 6300 := by
  sorry


end painting_cost_tripled_l1736_173633


namespace divisibility_condition_l1736_173696

/-- Converts a base-9 number of the form 2d6d4₉ to base 10 --/
def base9_to_base10 (d : ℕ) : ℕ :=
  2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4

/-- Checks if a natural number is divisible by 13 --/
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

/-- States that 2d6d4₉ is divisible by 13 if and only if d = 4 --/
theorem divisibility_condition (d : ℕ) (h : d ≤ 8) :
  is_divisible_by_13 (base9_to_base10 d) ↔ d = 4 := by
  sorry

end divisibility_condition_l1736_173696


namespace three_numbers_sum_l1736_173648

theorem three_numbers_sum : ∀ (a b c : ℝ),
  (a ≤ b ∧ b ≤ c) →                             -- a, b, c are in ascending order
  ((a + b + c) / 3 = a + 15) →                  -- mean is 15 more than smallest
  ((a + b + c) / 3 = c - 20) →                  -- mean is 20 less than largest
  (b = 7) →                                     -- median is 7
  (a + b + c = 36) :=                           -- sum is 36
by
  sorry

end three_numbers_sum_l1736_173648


namespace lcm_18_28_l1736_173638

theorem lcm_18_28 : Nat.lcm 18 28 = 252 := by
  sorry

end lcm_18_28_l1736_173638


namespace division_remainder_l1736_173636

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 127 →
  divisor = 25 →
  quotient = 5 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end division_remainder_l1736_173636


namespace projection_of_a_on_b_l1736_173613

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 3, |b| = 2, and |a - b| = √19,
    prove that the projection of a onto b is -3/2 -/
theorem projection_of_a_on_b 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V)
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 2)
  (hab : ‖a - b‖ = Real.sqrt 19) :
  inner a b / ‖b‖ = -3/2 := by
  sorry

end projection_of_a_on_b_l1736_173613


namespace perfect_pair_122_14762_l1736_173687

/-- Two natural numbers form a perfect pair if their sum and product are perfect squares. -/
def isPerfectPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a + b = x^2 ∧ a * b = y^2

/-- Theorem stating that 122 and 14762 form a perfect pair. -/
theorem perfect_pair_122_14762 : isPerfectPair 122 14762 := by
  sorry

#check perfect_pair_122_14762

end perfect_pair_122_14762_l1736_173687


namespace inequality_of_powers_l1736_173617

theorem inequality_of_powers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^(2*a) * b^(2*b) * c^(2*c) ≥ a^(b+c) * b^(c+a) * c^(a+b) := by
  sorry

end inequality_of_powers_l1736_173617


namespace arithmetic_geometric_mean_inequality_l1736_173668

theorem arithmetic_geometric_mean_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a + b + c + d) / 4 ≥ (a * b * c * d) ^ (1/4) := by
  sorry

end arithmetic_geometric_mean_inequality_l1736_173668


namespace c_share_approximately_119_73_l1736_173657

-- Define the grazing capacity conversion rates
def horse_to_ox : ℝ := 2
def sheep_to_ox : ℝ := 0.5

-- Define the total rent
def total_rent : ℝ := 1200

-- Define the grazing capacities for each person
def a_capacity : ℝ := 10 * 7 + 4 * horse_to_ox * 3
def b_capacity : ℝ := 12 * 5
def c_capacity : ℝ := 15 * 3
def d_capacity : ℝ := 18 * 6 + 6 * sheep_to_ox * 8
def e_capacity : ℝ := 20 * 4
def f_capacity : ℝ := 5 * horse_to_ox * 2 + 10 * sheep_to_ox * 4

-- Define the total grazing capacity
def total_capacity : ℝ := a_capacity + b_capacity + c_capacity + d_capacity + e_capacity + f_capacity

-- Theorem to prove
theorem c_share_approximately_119_73 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((c_capacity / total_capacity * total_rent) - 119.73) < ε :=
sorry

end c_share_approximately_119_73_l1736_173657


namespace percentage_problem_l1736_173615

theorem percentage_problem : 
  ∃ (P : ℝ), (P / 100) * 40 = 0.25 * 16 + 2 ∧ P = 15 := by
  sorry

end percentage_problem_l1736_173615


namespace last_term_of_gp_l1736_173652

theorem last_term_of_gp (a : ℝ) (r : ℝ) (S : ℝ) (n : ℕ) :
  a = 9 →
  r = 1/3 →
  S = 40/3 →
  S = a * (1 - r^n) / (1 - r) →
  a * r^(n-1) = 3 :=
sorry

end last_term_of_gp_l1736_173652


namespace rational_root_iff_k_eq_neg_two_or_zero_l1736_173654

/-- The polynomial X^2017 - X^2016 + X^2 + kX + 1 has a rational root if and only if k = -2 or k = 0 -/
theorem rational_root_iff_k_eq_neg_two_or_zero (k : ℚ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k*x + 1 = 0) ↔ (k = -2 ∨ k = 0) := by
  sorry

end rational_root_iff_k_eq_neg_two_or_zero_l1736_173654


namespace store_b_earns_more_l1736_173672

/-- Represents the total value of goods sold by each store in yuan -/
def total_sales : ℕ := 1000000

/-- Represents the discount rate offered by store A -/
def discount_rate : ℚ := 1/10

/-- Represents the cost of a lottery ticket in yuan -/
def ticket_cost : ℕ := 100

/-- Represents the number of tickets in a batch -/
def tickets_per_batch : ℕ := 10000

/-- Represents the prize structure for store B -/
structure PrizeStructure where
  first_prize : ℕ × ℕ  -- (number of prizes, value of each prize)
  second_prize : ℕ × ℕ
  third_prize : ℕ × ℕ
  fourth_prize : ℕ × ℕ
  fifth_prize : ℕ × ℕ

/-- The actual prize structure used by store B -/
def store_b_prizes : PrizeStructure := {
  first_prize := (5, 1000),
  second_prize := (10, 500),
  third_prize := (20, 200),
  fourth_prize := (40, 100),
  fifth_prize := (5000, 10)
}

/-- Calculates the total prize value for a given prize structure -/
def total_prize_value (ps : PrizeStructure) : ℕ :=
  ps.first_prize.1 * ps.first_prize.2 +
  ps.second_prize.1 * ps.second_prize.2 +
  ps.third_prize.1 * ps.third_prize.2 +
  ps.fourth_prize.1 * ps.fourth_prize.2 +
  ps.fifth_prize.1 * ps.fifth_prize.2

/-- Theorem stating that store B earns at least 32,000 yuan more than store A -/
theorem store_b_earns_more :
  ∃ (x : ℕ), x ≥ 32000 ∧
  (total_sales - (total_prize_value store_b_prizes) * (total_sales / (tickets_per_batch * ticket_cost))) =
  (total_sales * (1 - discount_rate)).floor + x :=
by sorry


end store_b_earns_more_l1736_173672


namespace baseball_card_theorem_l1736_173699

/-- Represents the amount of money each person has and the cost of the baseball card. -/
structure BaseballCardProblem where
  patricia_money : ℝ
  lisa_money : ℝ
  charlotte_money : ℝ
  card_cost : ℝ

/-- Calculates the additional money required to buy the baseball card. -/
def additional_money_required (problem : BaseballCardProblem) : ℝ :=
  problem.card_cost - (problem.patricia_money + problem.lisa_money + problem.charlotte_money)

/-- Theorem stating the additional money required is $49 given the problem conditions. -/
theorem baseball_card_theorem (problem : BaseballCardProblem) 
  (h1 : problem.patricia_money = 6)
  (h2 : problem.lisa_money = 5 * problem.patricia_money)
  (h3 : problem.lisa_money = 2 * problem.charlotte_money)
  (h4 : problem.card_cost = 100) :
  additional_money_required problem = 49 := by
  sorry

#eval additional_money_required { 
  patricia_money := 6, 
  lisa_money := 30, 
  charlotte_money := 15, 
  card_cost := 100 
}

end baseball_card_theorem_l1736_173699


namespace cos_sum_plus_cos_diff_l1736_173604

theorem cos_sum_plus_cos_diff (x y : ℝ) : 
  Real.cos (x + y) + Real.cos (x - y) = 2 * Real.cos x * Real.cos y := by
  sorry

end cos_sum_plus_cos_diff_l1736_173604


namespace geometric_sequence_constant_l1736_173645

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_constant (a : ℕ → ℝ) (c : ℝ) :
  IsGeometric a → IsGeometric (fun n ↦ a n + c) → c = 0 := by
  sorry


end geometric_sequence_constant_l1736_173645


namespace candies_remaining_l1736_173619

/-- The number of candies remaining after Carlos ate all the yellow candies -/
def remaining_candies (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ :=
  red + blue

/-- Theorem stating the number of remaining candies given the problem conditions -/
theorem candies_remaining :
  ∀ (red : ℕ) (yellow : ℕ) (blue : ℕ),
  red = 40 →
  yellow = 3 * red - 20 →
  blue = yellow / 2 →
  remaining_candies red yellow blue = 90 := by
sorry

end candies_remaining_l1736_173619


namespace function_decreasing_and_inequality_l1736_173614

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / log x - a * x

theorem function_decreasing_and_inequality (e : ℝ) (h_e : exp 1 = e) :
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → (deriv (f a)) x ≤ 0) → a ≥ 1/4) ∧
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧
    f a x₁ - (deriv (f a)) x₂ ≤ a) → a ≥ 1/2 - 1/(4*e^2)) :=
by sorry

end function_decreasing_and_inequality_l1736_173614


namespace sine_function_value_l1736_173682

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0,
    if the distance between adjacent maximum and minimum points is 2√2,
    then f(1) = √3/2 -/
theorem sine_function_value (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 3)
  (∃ A B : ℝ × ℝ, 
    (A.2 = f A.1 ∧ B.2 = f B.1) ∧ 
    (∀ x ∈ Set.Icc A.1 B.1, f x ≤ A.2 ∧ f x ≥ B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2) →
  f 1 = Real.sqrt 3 / 2 := by
sorry

end sine_function_value_l1736_173682


namespace optimal_workers_theorem_l1736_173628

/-- The number of workers that should process part P to minimize processing time -/
def optimal_workers_for_P (total_P : ℕ) (total_Q : ℕ) (total_workers : ℕ) 
  (P_rate : ℚ) (Q_rate : ℚ) : ℕ :=
  137

/-- The theorem stating that 137 workers should process part P for optimal time -/
theorem optimal_workers_theorem (total_P : ℕ) (total_Q : ℕ) (total_workers : ℕ) 
  (P_rate : ℚ) (Q_rate : ℚ) :
  total_P = 6000 →
  total_Q = 2000 →
  total_workers = 214 →
  5 * P_rate = 3 * Q_rate →
  optimal_workers_for_P total_P total_Q total_workers P_rate Q_rate = 137 :=
by
  sorry

#check optimal_workers_theorem

end optimal_workers_theorem_l1736_173628


namespace agent_commission_proof_l1736_173660

/-- Calculate the commission for an agent given the commission rate and total sales -/
def calculate_commission (commission_rate : ℚ) (total_sales : ℚ) : ℚ :=
  commission_rate * total_sales

theorem agent_commission_proof :
  let commission_rate : ℚ := 5 / 100
  let total_sales : ℚ := 250
  calculate_commission commission_rate total_sales = 25 / 2 := by
  sorry

end agent_commission_proof_l1736_173660


namespace first_company_visit_charge_is_55_l1736_173691

/-- The visit charge of the first plumbing company -/
def first_company_visit_charge : ℝ := sorry

/-- The hourly rate of the first plumbing company -/
def first_company_hourly_rate : ℝ := 35

/-- The visit charge of Reliable Plumbing -/
def reliable_plumbing_visit_charge : ℝ := 75

/-- The hourly rate of Reliable Plumbing -/
def reliable_plumbing_hourly_rate : ℝ := 30

/-- The number of labor hours -/
def labor_hours : ℝ := 4

theorem first_company_visit_charge_is_55 :
  first_company_visit_charge = 55 :=
by
  have h1 : first_company_visit_charge + labor_hours * first_company_hourly_rate =
            reliable_plumbing_visit_charge + labor_hours * reliable_plumbing_hourly_rate :=
    sorry
  sorry

end first_company_visit_charge_is_55_l1736_173691


namespace pascal_parallelogram_sum_l1736_173625

/-- Represents a position in Pascal's triangle -/
structure Position :=
  (row : ℕ)
  (col : ℕ)
  (h : col ≤ row)

/-- The value at a given position in Pascal's triangle -/
def pascal_value (p : Position) : ℕ := sorry

/-- The parallelogram bounded by diagonals intersecting at a given position -/
def parallelogram (p : Position) : Set Position := sorry

/-- The sum of values in the parallelogram -/
def parallelogram_sum (p : Position) : ℕ := sorry

/-- The theorem stating the relationship between a number in Pascal's triangle
    and the sum of numbers in the parallelogram below it -/
theorem pascal_parallelogram_sum (p : Position) :
  pascal_value p - 1 = parallelogram_sum p := by sorry

end pascal_parallelogram_sum_l1736_173625


namespace large_stores_count_l1736_173683

/-- Represents the total number of stores -/
def total_stores : ℕ := 1500

/-- Represents the sample size -/
def sample_size : ℕ := 90

/-- Represents the ratio of large stores -/
def large_ratio : ℕ := 3

/-- Represents the ratio of medium stores -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores -/
def small_ratio : ℕ := 7

/-- Calculates the number of large stores in the sample -/
def large_stores_in_sample : ℕ :=
  (sample_size * large_ratio) / (large_ratio + medium_ratio + small_ratio)

theorem large_stores_count :
  large_stores_in_sample = 18 := by sorry

end large_stores_count_l1736_173683


namespace complex_sum_real_l1736_173635

theorem complex_sum_real (a : ℝ) (z₁ z₂ : ℂ) : 
  z₁ = (3 / (a + 5) : ℂ) + (10 - a^2 : ℂ) * Complex.I ∧
  z₂ = (2 / (1 - a) : ℂ) + (2*a - 5 : ℂ) * Complex.I ∧
  (z₁ + z₂).im = 0 →
  a = 3 :=
by sorry

end complex_sum_real_l1736_173635


namespace f_decreasing_f_odd_implies_m_zero_l1736_173688

/-- The function f(x) = -2x + m -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ -2 * x + m

theorem f_decreasing (m : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f m x₁ > f m x₂ := by sorry

theorem f_odd_implies_m_zero (m : ℝ) : 
  (∀ x : ℝ, f m (-x) = -(f m x)) → m = 0 := by sorry

end f_decreasing_f_odd_implies_m_zero_l1736_173688


namespace all_propositions_correct_l1736_173606

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

theorem all_propositions_correct :
  (double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011) ∧
  (double_factorial 2010 = 2^1005 * Nat.factorial 1005) ∧
  (double_factorial 2010 % 10 = 0) ∧
  (double_factorial 2011 % 10 = 5) := by
  sorry

end all_propositions_correct_l1736_173606


namespace power_tower_mod_500_l1736_173655

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by
  sorry

end power_tower_mod_500_l1736_173655


namespace race_distance_is_400_l1736_173634

/-- Represents the speed of a runner relative to others -/
structure RelativeSpeed where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculate the relative speeds based on race results -/
def calculate_relative_speeds : RelativeSpeed :=
  let ab_ratio := 500 / 450
  let bc_ratio := 500 / 475
  { a := ab_ratio * bc_ratio
  , b := bc_ratio
  , c := 1 }

/-- The race distance where A beats C by 58 meters -/
def race_distance (speeds : RelativeSpeed) : ℚ :=
  58 * speeds.a / (speeds.a - speeds.c)

/-- Theorem stating that the race distance is 400 meters -/
theorem race_distance_is_400 :
  race_distance calculate_relative_speeds = 400 := by sorry

end race_distance_is_400_l1736_173634


namespace geometric_series_ratio_l1736_173653

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h_conv : abs r < 1) :
  (a / (1 - r) = 16 * (a * r^2 / (1 - r))) → (r = 1/4 ∨ r = -1/4) := by
  sorry

end geometric_series_ratio_l1736_173653


namespace least_divisor_power_l1736_173637

theorem least_divisor_power (a : ℕ) (h1 : a > 1) (h2 : Odd a) :
  (∃ n : ℕ, n > 0 ∧ (2^2000 : ℕ) ∣ (a^n - 1)) ∧
  (∀ m : ℕ, 0 < m → m < 2^1998 → ¬((2^2000 : ℕ) ∣ (a^m - 1))) ∧
  ((2^2000 : ℕ) ∣ (a^(2^1998) - 1)) :=
sorry

end least_divisor_power_l1736_173637


namespace amounts_theorem_l1736_173663

/-- Represents the amounts held by individuals p, q, r, s, and t -/
structure Amounts where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ

/-- The total amount among all individuals is $24,000 -/
def total_amount : ℝ := 24000

/-- The conditions given in the problem -/
def satisfies_conditions (a : Amounts) : Prop :=
  a.p + a.q + a.r + a.s + a.t = total_amount ∧
  a.r = 3/5 * (a.p + a.q) ∧
  a.s = 0.45 * total_amount ∧
  a.t = 1/2 * a.r

/-- The theorem to be proved -/
theorem amounts_theorem (a : Amounts) (h : satisfies_conditions a) : 
  a.r = 4200 ∧ a.s = 10800 ∧ a.t = 2100 ∧ a.p + a.q = 7000 := by
  sorry

end amounts_theorem_l1736_173663


namespace betty_herb_garden_total_l1736_173602

/-- The number of basil plants in Betty's herb garden. -/
def basil_plants : ℕ := 5

/-- The number of oregano plants in Betty's herb garden. -/
def oregano_plants : ℕ := 2 * basil_plants + 2

/-- The total number of plants in Betty's herb garden. -/
def total_plants : ℕ := basil_plants + oregano_plants

/-- Theorem stating that the total number of plants in Betty's herb garden is 17. -/
theorem betty_herb_garden_total : total_plants = 17 := by
  sorry

end betty_herb_garden_total_l1736_173602


namespace square_sum_equality_l1736_173658

theorem square_sum_equality : 106 * 106 + 94 * 94 = 20072 := by
  sorry

end square_sum_equality_l1736_173658


namespace percentage_to_decimal_decimal_representation_of_208_percent_l1736_173674

/-- The decimal representation of a percentage is equal to the percentage divided by 100. -/
theorem percentage_to_decimal (p : ℝ) : p / 100 = p * (1 / 100) := by sorry

/-- The decimal representation of 208% is 2.08. -/
theorem decimal_representation_of_208_percent : (208 : ℝ) / 100 = 2.08 := by sorry

end percentage_to_decimal_decimal_representation_of_208_percent_l1736_173674


namespace geometric_sequence_property_l1736_173630

/-- Given a geometric sequence {a_n} where a₃a₅a₇a₉a₁₁ = 243, prove that a₁₀² / a₁₃ = 3 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_product : a 3 * a 5 * a 7 * a 9 * a 11 = 243) :
  a 10 ^ 2 / a 13 = 3 := by
  sorry

end geometric_sequence_property_l1736_173630


namespace point_on_x_axis_l1736_173667

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis is the set of all points with y-coordinate equal to 0 -/
def x_axis : Set Point := {p : Point | p.y = 0}

/-- Theorem: If a point A has y-coordinate equal to 0, then A lies on the x-axis -/
theorem point_on_x_axis (A : Point) (h : A.y = 0) : A ∈ x_axis := by
  sorry

end point_on_x_axis_l1736_173667


namespace bruce_pizza_production_l1736_173656

/-- The number of pizza doughs Bruce can make with one sack of flour -/
def pizzas_per_sack (sacks_per_day : ℕ) (pizzas_per_week : ℕ) (days_per_week : ℕ) : ℚ :=
  pizzas_per_week / (sacks_per_day * days_per_week)

/-- Proof that Bruce can make 15 pizza doughs with one sack of flour -/
theorem bruce_pizza_production :
  pizzas_per_sack 5 525 7 = 15 := by
  sorry

end bruce_pizza_production_l1736_173656


namespace tens_digit_of_3_pow_2016_l1736_173664

theorem tens_digit_of_3_pow_2016 :
  ∃ n : ℕ, 3^2016 = 100*n + 21 :=
sorry

end tens_digit_of_3_pow_2016_l1736_173664


namespace sector_central_angle_l1736_173693

/-- Given a sector with radius 2 cm and area 4 cm², 
    prove that the radian measure of its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r^2 * α → α = 2 := by sorry

end sector_central_angle_l1736_173693


namespace triple_sharp_of_30_l1736_173618

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem triple_sharp_of_30 : sharp (sharp (sharp 30)) = 7.25 := by
  sorry

end triple_sharp_of_30_l1736_173618


namespace least_bench_sections_l1736_173651

/-- Represents the capacity of a single bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Proves that the least positive integer N such that N bench sections can hold
    an equal number of adults and children is 3, given that one bench section
    holds 8 adults or 12 children. -/
theorem least_bench_sections (capacity : BenchCapacity)
    (h1 : capacity.adults = 8)
    (h2 : capacity.children = 12) :
    ∃ N : Nat, N > 0 ∧ N * capacity.adults = N * capacity.children ∧
    ∀ M : Nat, M > 0 → M * capacity.adults = M * capacity.children → N ≤ M :=
  by sorry

end least_bench_sections_l1736_173651


namespace expression_simplification_l1736_173624

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  (x^2 + 2 / x) * (y^2 + 2 / y) + (x^2 - 2 / y) * (y^2 - 2 / x) = 2 * x * y + 8 / (x * y) := by
  sorry

end expression_simplification_l1736_173624


namespace investment_ratio_from_profit_ratio_l1736_173607

/-- Represents an investment partner -/
structure Partner where
  investment : ℝ
  time : ℝ

/-- Theorem stating the relationship between profit ratio and investment ratio -/
theorem investment_ratio_from_profit_ratio
  (p q : Partner)
  (profit_ratio : ℝ × ℝ)
  (hp : p.time = 5)
  (hq : q.time = 12)
  (hprofit : profit_ratio = (7, 12)) :
  p.investment / q.investment = 7 / 5 := by
  sorry


end investment_ratio_from_profit_ratio_l1736_173607


namespace parabola_c_value_l1736_173686

/-- A parabola with equation y = ax^2 + bx + c, vertex at (3, -5), and passing through (0, -2) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := -5
  point_x : ℝ := 0
  point_y : ℝ := -2

/-- The c-value of the parabola is -2 -/
theorem parabola_c_value (p : Parabola) : p.c = -2 := by
  sorry

end parabola_c_value_l1736_173686


namespace tangent_line_points_l1736_173690

/-- The function f(x) = x³ + ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem tangent_line_points (a : ℝ) :
  ∃ (x₀ : ℝ), (f_deriv a x₀ = -1 ∧ x₀ + f a x₀ = 0) →
  ((x₀ = 1 ∧ f a x₀ = -1) ∨ (x₀ = -1 ∧ f a x₀ = 1)) :=
sorry

end tangent_line_points_l1736_173690


namespace inscribed_square_area_specific_inscribed_square_area_l1736_173695

/-- The area of a square inscribed in a right triangle -/
theorem inscribed_square_area (LP SN : ℝ) (h1 : LP > 0) (h2 : SN > 0) :
  let x := Real.sqrt (LP * SN)
  (x : ℝ) ^ 2 = LP * SN := by sorry

/-- The specific case where LP = 30 and SN = 70 -/
theorem specific_inscribed_square_area :
  let LP : ℝ := 30
  let SN : ℝ := 70
  let x := Real.sqrt (LP * SN)
  (x : ℝ) ^ 2 = 2100 := by sorry

end inscribed_square_area_specific_inscribed_square_area_l1736_173695


namespace garden_perimeter_l1736_173694

/-- The perimeter of a rectangular garden with width 12 meters and an area equal to that of a 16m × 12m playground is 56 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let garden_width : ℝ := 12
  let playground_area : ℝ := playground_length * playground_width
  let garden_length : ℝ := playground_area / garden_width
  let garden_perimeter : ℝ := 2 * (garden_length + garden_width)
  garden_perimeter = 56 := by sorry

end garden_perimeter_l1736_173694


namespace square_sum_of_solution_l1736_173622

theorem square_sum_of_solution (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 75) : 
  x^2 + y^2 = 3205 / 121 := by
sorry

end square_sum_of_solution_l1736_173622


namespace frobenius_coin_problem_l1736_173665

/-- Two natural numbers are coprime -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set M of integers that can be expressed as ax + by for non-negative x and y -/
def M (a b : ℕ) : Set ℤ := {z : ℤ | ∃ x y : ℕ, z = a * x + b * y}

/-- The greatest integer not in M -/
def c (a b : ℕ) : ℤ := a * b - a - b

theorem frobenius_coin_problem (a b : ℕ) (h : Coprime a b) :
  (∀ z : ℤ, z > c a b → z ∈ M a b) ∧
  (c a b ∉ M a b) ∧
  (∀ n : ℤ, (n ∈ M a b ∧ (c a b - n) ∉ M a b) ∨ (n ∉ M a b ∧ (c a b - n) ∈ M a b)) :=
sorry

end frobenius_coin_problem_l1736_173665


namespace ellipse_focus_distance_l1736_173631

/-- An ellipse with equation x²/4 + y² = 1 -/
structure Ellipse where
  eq : ∀ x y : ℝ, x^2/4 + y^2 = 1

/-- The left focus of the ellipse -/
def leftFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def rightFocus (e : Ellipse) : ℝ × ℝ := sorry

/-- A point on the ellipse where a line perpendicular to the x-axis passing through the left focus intersects the ellipse -/
def intersectionPoint (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_focus_distance (e : Ellipse) :
  distance (intersectionPoint e) (rightFocus e) = 7/2 := by sorry

end ellipse_focus_distance_l1736_173631


namespace modified_ohara_triple_27_8_l1736_173685

/-- Definition of a Modified O'Hara Triple -/
def is_modified_ohara_triple (a b x : ℕ+) : Prop :=
  (a.val : ℝ)^(1/3) - (b.val : ℝ)^(1/3) = x.val

/-- Theorem: If (27, 8, x) is a Modified O'Hara triple, then x = 1 -/
theorem modified_ohara_triple_27_8 (x : ℕ+) :
  is_modified_ohara_triple 27 8 x → x = 1 := by
  sorry

end modified_ohara_triple_27_8_l1736_173685


namespace min_dimension_sum_for_2310_volume_l1736_173698

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- The volume of a box given its dimensions -/
def volume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- The sum of the dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating that the minimum sum of dimensions for a box with volume 2310 is 52 -/
theorem min_dimension_sum_for_2310_volume :
  (∃ (d : BoxDimensions), volume d = 2310) →
  (∀ (d : BoxDimensions), volume d = 2310 → dimensionSum d ≥ 52) ∧
  (∃ (d : BoxDimensions), volume d = 2310 ∧ dimensionSum d = 52) :=
sorry

end min_dimension_sum_for_2310_volume_l1736_173698


namespace units_digit_of_sum_power_problem_solution_l1736_173640

theorem units_digit_of_sum_power (a b n : ℕ) : 
  (a + b) % 10 = 1 → ((a + b)^n) % 10 = 1 :=
by
  sorry

theorem problem_solution : ((5619 + 2272)^124) % 10 = 1 :=
by
  sorry

end units_digit_of_sum_power_problem_solution_l1736_173640


namespace product_inequality_solve_for_a_l1736_173609

-- Part I
theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 + 1/a) * (1 + 1/b) ≥ 9 := by sorry

-- Part II
theorem solve_for_a (a : ℝ) (h : ∀ x, |x + 3| - |x - a| ≥ 2 ↔ x ≥ 1) :
  a = 2 := by sorry

end product_inequality_solve_for_a_l1736_173609


namespace wills_breakfast_calories_l1736_173659

/-- Proves that Will's breakfast supplied him 900 calories of energy -/
theorem wills_breakfast_calories :
  ∀ (jog_duration : ℕ) (calories_per_minute : ℕ) (net_calories : ℕ),
    jog_duration = 30 →
    calories_per_minute = 10 →
    net_calories = 600 →
    jog_duration * calories_per_minute + net_calories = 900 :=
by
  sorry

end wills_breakfast_calories_l1736_173659


namespace sequence_always_terminates_l1736_173643

def last_digit (n : ℕ) : ℕ := n % 10

def next_term (n : ℕ) : ℕ :=
  if last_digit n ≤ 5 then n / 10 else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate next_term k a₀) = 0

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

end sequence_always_terminates_l1736_173643


namespace garden_minimum_cost_l1736_173673

/-- Represents the cost of a herb in dollars per square meter -/
structure HerbCost where
  cost : ℝ
  cost_positive : cost > 0

/-- Represents a region in the garden -/
structure Region where
  area : ℝ
  area_positive : area > 0

/-- Calculates the minimum cost for planting a garden given regions and herb costs -/
def minimum_garden_cost (regions : List Region) (herb_costs : List HerbCost) : ℝ :=
  sorry

/-- The main theorem stating the minimum cost for the given garden configuration -/
theorem garden_minimum_cost :
  let regions : List Region := [
    ⟨14, by norm_num⟩,
    ⟨35, by norm_num⟩,
    ⟨10, by norm_num⟩,
    ⟨21, by norm_num⟩,
    ⟨36, by norm_num⟩
  ]
  let herb_costs : List HerbCost := [
    ⟨1.00, by norm_num⟩,
    ⟨1.50, by norm_num⟩,
    ⟨2.00, by norm_num⟩,
    ⟨2.50, by norm_num⟩,
    ⟨3.00, by norm_num⟩
  ]
  minimum_garden_cost regions herb_costs = 195.50 := by
    sorry

end garden_minimum_cost_l1736_173673


namespace inequality_theorem_l1736_173661

theorem inequality_theorem (a b c m : ℝ) 
  (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c : ℝ, a > b → b > c → (1 / (a - b) + 1 / (b - c) ≥ m / (a - c))) : 
  m ≤ 4 := by
  sorry

end inequality_theorem_l1736_173661


namespace martian_right_angle_theorem_l1736_173675

/-- The number of clerts in a full circle in the Martian system -/
def full_circle_clerts : ℕ := 600

/-- The fraction of a full circle that represents a Martian right angle -/
def martian_right_angle_fraction : ℚ := 1/3

/-- The number of clerts in a Martian right angle -/
def martian_right_angle_clerts : ℕ := 200

/-- Theorem stating that the number of clerts in a Martian right angle is 200 -/
theorem martian_right_angle_theorem : 
  (↑full_circle_clerts : ℚ) * martian_right_angle_fraction = martian_right_angle_clerts := by
  sorry

end martian_right_angle_theorem_l1736_173675


namespace beaker_problem_solution_l1736_173641

/-- Represents the capacity of a beaker -/
structure Beaker where
  capacity : ℚ
  filled : ℚ
  h_filled_nonneg : 0 ≤ filled
  h_filled_le_capacity : filled ≤ capacity

/-- The fraction of a beaker that is filled -/
def fraction_filled (b : Beaker) : ℚ :=
  b.filled / b.capacity

/-- Represents the problem setup with two beakers -/
structure BeakerProblem where
  small : Beaker
  large : Beaker
  h_small_half_filled : fraction_filled small = 1/2
  h_large_capacity : large.capacity = 5 * small.capacity
  h_large_fifth_filled : fraction_filled large = 1/5

/-- The main theorem to prove -/
theorem beaker_problem_solution (problem : BeakerProblem) :
  let final_large := Beaker.mk
    problem.large.capacity
    (problem.large.filled + problem.small.filled)
    (by sorry) -- Proof that the new filled amount is non-negative
    (by sorry) -- Proof that the new filled amount is ≤ capacity
  fraction_filled final_large = 3/10 := by sorry


end beaker_problem_solution_l1736_173641


namespace arithmetic_sequence_sum_l1736_173684

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : a 4 + a 5 + a 6 = 42 := by
  sorry

end arithmetic_sequence_sum_l1736_173684


namespace sum_of_three_numbers_l1736_173662

theorem sum_of_three_numbers : 85.9 + 5.31 + (43 / 2 : ℝ) = 112.71 := by
  sorry

end sum_of_three_numbers_l1736_173662


namespace years_B_is_two_l1736_173611

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  principal_B : ℕ := 5000
  principal_C : ℕ := 3000
  years_C : ℕ := 4
  rate : ℚ := 1/10
  total_interest : ℕ := 2200

/-- Calculates the number of years A lent to B --/
def years_B (loan : LoanDetails) : ℚ :=
  (loan.total_interest - (loan.principal_C * loan.rate * loan.years_C)) / (loan.principal_B * loan.rate)

/-- Theorem stating that the number of years A lent to B is 2 --/
theorem years_B_is_two (loan : LoanDetails) : years_B loan = 2 := by
  sorry

end years_B_is_two_l1736_173611


namespace probability_three_blue_six_trials_l1736_173680

/-- The probability of drawing exactly k blue marbles in n trials,
    given b blue marbles and r red marbles in a bag,
    where each draw is independent and the marble is replaced after each draw. -/
def probability_k_blue (n k b r : ℕ) : ℚ :=
  (n.choose k) * ((b : ℚ) / (b + r : ℚ))^k * ((r : ℚ) / (b + r : ℚ))^(n - k)

/-- The main theorem stating the probability of drawing exactly three blue marbles
    in six trials from a bag with 8 blue marbles and 6 red marbles. -/
theorem probability_three_blue_six_trials :
  probability_k_blue 6 3 8 6 = 34560 / 117649 := by
  sorry

end probability_three_blue_six_trials_l1736_173680


namespace nested_bracket_calculation_l1736_173679

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Theorem statement
theorem nested_bracket_calculation :
  bracket (bracket 100 20 60) (bracket 7 2 3) (bracket 20 10 10) = 5 / 3 :=
by sorry

end nested_bracket_calculation_l1736_173679


namespace field_length_is_32_l1736_173697

/-- Proves that a rectangular field with specific properties has a length of 32 meters -/
theorem field_length_is_32 (l w : ℝ) (h1 : l = 2 * w) (h2 : (8 * 8 : ℝ) = (1 / 8) * (l * w)) : l = 32 :=
by sorry

end field_length_is_32_l1736_173697


namespace square_area_is_40_l1736_173681

/-- A parabola defined by y = x^2 + 4x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 1

/-- The y-coordinate of the line that coincides with one side of the square -/
def line_y : ℝ := 7

/-- The theorem stating that the area of the square is 40 -/
theorem square_area_is_40 :
  ∃ (x1 x2 : ℝ),
    parabola x1 = line_y ∧
    parabola x2 = line_y ∧
    x1 ≠ x2 ∧
    (x2 - x1)^2 = 40 :=
by sorry

end square_area_is_40_l1736_173681


namespace dentist_age_fraction_l1736_173676

theorem dentist_age_fraction (F : ℚ) : 
  let current_age : ℕ := 32
  let age_8_years_ago : ℕ := current_age - 8
  let age_8_years_hence : ℕ := current_age + 8
  F * age_8_years_ago = (1 : ℚ) / 10 * age_8_years_hence →
  F = (1 : ℚ) / 6 := by
  sorry

end dentist_age_fraction_l1736_173676


namespace first_three_valid_numbers_l1736_173647

def is_sum_of_consecutive (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : ℕ, n = k * a

def is_valid_number (n : ℕ) : Prop :=
  is_sum_of_consecutive n 5 ∧ is_sum_of_consecutive n 7

theorem first_three_valid_numbers :
  (is_valid_number 35 ∧ 
   is_valid_number 70 ∧ 
   is_valid_number 105) ∧ 
  (∀ m : ℕ, m < 35 → ¬is_valid_number m) ∧
  (∀ m : ℕ, 35 < m ∧ m < 70 → ¬is_valid_number m) ∧
  (∀ m : ℕ, 70 < m ∧ m < 105 → ¬is_valid_number m) :=
by sorry

end first_three_valid_numbers_l1736_173647


namespace isosceles_triangle_from_equation_l1736_173650

/-- Given a triangle ABC with sides a and b, and angles A and B,
    if the equation x^2 - (b cos A)x + a cos B = 0 has roots whose
    product equals their sum, then the triangle is isosceles. -/
theorem isosceles_triangle_from_equation (a b : ℝ) (A B : ℝ) :
  (∃ (x y : ℝ), x^2 - (b * Real.cos A) * x + a * Real.cos B = 0 ∧
                 x * y = x + y) →
  (a > 0 ∧ b > 0 ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π) →
  a = b ∨ A = B :=
by sorry

end isosceles_triangle_from_equation_l1736_173650


namespace line_slope_relation_l1736_173671

/-- Theorem: For a straight line y = kx + b passing through points A(-3, y₁) and B(4, y₂),
    if k < 0, then y₁ > y₂. -/
theorem line_slope_relation (k b y₁ y₂ : ℝ) : 
  k < 0 → 
  y₁ = k * (-3) + b →
  y₂ = k * 4 + b →
  y₁ > y₂ := by
  sorry

end line_slope_relation_l1736_173671


namespace train_crossing_time_l1736_173632

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 45 →
  train_speed_kmh = 108 →
  crossing_time = train_length / (train_speed_kmh * (1000 / 3600)) →
  crossing_time = 1.5 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1736_173632


namespace polygon_interior_angle_sum_l1736_173620

/-- A polygon where each exterior angle is 36° has a sum of interior angles equal to 1440°. -/
theorem polygon_interior_angle_sum (n : ℕ) (h : n * 36 = 360) : 
  (n - 2) * 180 = 1440 :=
sorry

end polygon_interior_angle_sum_l1736_173620


namespace divisible_by_24_l1736_173678

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, (n^4 : ℤ) + 2*(n^3 : ℤ) + 11*(n^2 : ℤ) + 10*(n : ℤ) = 24*k := by
  sorry

end divisible_by_24_l1736_173678


namespace parallel_vectors_expression_l1736_173601

theorem parallel_vectors_expression (α : Real) : 
  let a : Fin 2 → Real := ![2, Real.sin α]
  let b : Fin 2 → Real := ![1, Real.cos α]
  (∃ (k : Real), a = k • b) →
  (1 + Real.sin (2 * α)) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 5 / 3 := by
  sorry

end parallel_vectors_expression_l1736_173601


namespace range_of_m_for_negative_f_solution_sets_for_inequality_l1736_173612

-- Define the function f(x) = mx^2 - mx - 1
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m for which f(x) < 0 for all x ∈ ℝ
theorem range_of_m_for_negative_f :
  ∀ m : ℝ, (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution sets for the inequality f(x) < (1-m)x - 1
theorem solution_sets_for_inequality :
  ∀ m : ℝ,
    (m = 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | x > 0}) ∧
    (m > 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | 0 < x ∧ x < 1 / m}) ∧
    (m < 0 → {x : ℝ | f m x < (1 - m) * x - 1} = {x : ℝ | x < 1 / m ∨ x > 0}) :=
sorry

end range_of_m_for_negative_f_solution_sets_for_inequality_l1736_173612


namespace cube_volume_surface_area_l1736_173627

/-- A cube with volume 8x cubic units and surface area 4x square units has x = 5400 --/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 4*x) → x = 5400 := by
  sorry

end cube_volume_surface_area_l1736_173627


namespace product_mod_23_l1736_173666

theorem product_mod_23 : (191 * 193 * 197) % 23 = 14 := by
  sorry

end product_mod_23_l1736_173666


namespace fraction_repeating_block_length_l1736_173603

/-- The number of digits in the smallest repeating block of the decimal expansion of 3/11 -/
def smallest_repeating_block_length : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 3 / 11

theorem fraction_repeating_block_length :
  smallest_repeating_block_length = 2 ∧ 
  ∃ (a b : ℕ), fraction = (a : ℚ) / (10^smallest_repeating_block_length - 1 : ℚ) + (b : ℚ) / (10^smallest_repeating_block_length : ℚ) :=
sorry

end fraction_repeating_block_length_l1736_173603


namespace sum_of_base_8_digits_of_888_l1736_173669

def base_8_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_base_8_digits_of_888 :
  sum_of_digits (base_8_representation 888) = 13 := by
  sorry

end sum_of_base_8_digits_of_888_l1736_173669


namespace equation_solution_l1736_173605

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4 ↔ 
  x = (-3 + Real.sqrt 13) / 4 ∨ x = (-3 - Real.sqrt 13) / 4 :=
by sorry

end equation_solution_l1736_173605


namespace max_k_inequality_l1736_173677

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ k : ℝ, k ≤ 6 → (2 * (a^2 + k*a*b + b^2)) / ((k+2)*(a+b)) ≥ Real.sqrt (a*b)) ∧
  (∀ ε > 0, ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (2 * (a^2 + (6+ε)*a*b + b^2)) / ((6+ε+2)*(a+b)) < Real.sqrt (a*b)) :=
by sorry

end max_k_inequality_l1736_173677


namespace expedition_cans_required_l1736_173626

/-- The number of days between neighboring camps -/
def days_between_camps : ℕ := 1

/-- The number of days from base camp to destination camp -/
def days_to_destination : ℕ := 5

/-- The maximum number of cans a member can carry -/
def max_cans_per_member : ℕ := 3

/-- The number of cans consumed by a member per day -/
def cans_consumed_per_day : ℕ := 1

/-- Function to calculate the minimum number of cans required -/
def min_cans_required (n : ℕ) : ℕ := max_cans_per_member ^ n

/-- Theorem stating the minimum number of cans required for the expedition -/
theorem expedition_cans_required :
  min_cans_required days_to_destination = 243 :=
sorry

end expedition_cans_required_l1736_173626


namespace minimum_handshakes_l1736_173616

theorem minimum_handshakes (n : ℕ) (h : ℕ) (hn : n = 30) (hh : h = 3) :
  (n * h) / 2 = 45 := by
  sorry

end minimum_handshakes_l1736_173616


namespace factorization_of_x_squared_minus_one_l1736_173600

theorem factorization_of_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_of_x_squared_minus_one_l1736_173600


namespace third_term_value_l1736_173610

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 23 →
  a 6 = 53 →
  a 3 = 38 :=
by sorry

end third_term_value_l1736_173610


namespace plane_equation_satisfies_conditions_l1736_173692

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space --/
structure Point where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Check if a point lies on a plane --/
def Point.liesOn (p : Point) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

/-- Check if two planes are parallel --/
def Plane.isParallelTo (pl1 pl2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ pl1.a = k * pl2.a ∧ pl1.b = k * pl2.b ∧ pl1.c = k * pl2.c

theorem plane_equation_satisfies_conditions
  (given_plane : Plane)
  (given_point : Point)
  (parallel_plane : Plane)
  (h1 : given_plane.a = 3)
  (h2 : given_plane.b = 4)
  (h3 : given_plane.c = -2)
  (h4 : given_plane.d = 16)
  (h5 : given_point.x = 2)
  (h6 : given_point.y = -3)
  (h7 : given_point.z = 5)
  (h8 : parallel_plane.a = 3)
  (h9 : parallel_plane.b = 4)
  (h10 : parallel_plane.c = -2)
  (h11 : parallel_plane.d = 6)
  : given_point.liesOn given_plane ∧
    given_plane.isParallelTo parallel_plane ∧
    given_plane.a > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs given_plane.a) (Int.natAbs given_plane.b)) (Int.natAbs given_plane.c)) (Int.natAbs given_plane.d) = 1 :=
by sorry

end plane_equation_satisfies_conditions_l1736_173692


namespace expression_equivalence_l1736_173623

theorem expression_equivalence (a b c m n p : ℝ) 
  (h : a / m + (b * c + n * p) / (b * p + c * n) = 0) :
  b / n + (a * c + m * p) / (a * p + c * m) = 0 := by
  sorry

end expression_equivalence_l1736_173623


namespace inequality_solution_set_l1736_173649

def solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 0}

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x + 3) ≥ 0 ↔ x ∈ solution_set := by sorry

end inequality_solution_set_l1736_173649


namespace total_jeans_purchased_l1736_173621

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of pairs purchased
def fox_pairs : ℕ := 3
def pony_pairs : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 8.64

-- Define the sum of discount rates
def total_discount_rate : ℝ := 0.22

-- Define the Pony jeans discount rate
def pony_discount_rate : ℝ := 0.13999999999999993

-- Theorem statement
theorem total_jeans_purchased :
  fox_pairs + pony_pairs = 5 := by sorry

end total_jeans_purchased_l1736_173621


namespace solve_equation_l1736_173639

/-- A function representing the non-standard addition in the sequence -/
def nonStandardAdd (a b : ℕ) : ℕ := a + b - 1

/-- The theorem stating that if 8 + x = 16 in the non-standard addition, then x = 9 -/
theorem solve_equation (x : ℕ) : nonStandardAdd 8 x = 16 → x = 9 := by
  sorry

end solve_equation_l1736_173639


namespace train_speed_on_time_l1736_173629

/-- The speed at which a train arrives on time, given the journey length and late arrival information. -/
theorem train_speed_on_time 
  (journey_length : ℝ) 
  (late_speed : ℝ) 
  (late_time : ℝ) 
  (h1 : journey_length = 15) 
  (h2 : late_speed = 50) 
  (h3 : late_time = 0.25) : 
  (journey_length / ((journey_length / late_speed) - late_time) = 300) :=
by sorry

end train_speed_on_time_l1736_173629


namespace prob_three_in_seven_thirteenths_l1736_173689

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : List ℕ := sorry

/-- The probability of a digit occurring in a decimal representation -/
def digitProbability (d : ℕ) (q : ℚ) : ℚ := sorry

/-- Theorem: The probability of selecting 3 from the decimal representation of 7/13 is 1/6 -/
theorem prob_three_in_seven_thirteenths :
  digitProbability 3 (7/13) = 1/6 := by sorry

end prob_three_in_seven_thirteenths_l1736_173689


namespace function_difference_implies_m_value_l1736_173670

theorem function_difference_implies_m_value :
  ∀ (f g : ℝ → ℝ) (m : ℝ),
    (∀ x, f x = 4 * x^2 - 3 * x + 5) →
    (∀ x, g x = x^2 - m * x - 8) →
    f 5 - g 5 = 20 →
    m = -13.6 := by
  sorry

end function_difference_implies_m_value_l1736_173670


namespace product_xyz_l1736_173642

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 3) : 
  x * y * z = 1/11 := by
sorry

end product_xyz_l1736_173642


namespace range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1736_173646

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≤ 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 ≥ 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, q x ∧ ¬(p x a)) → 1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l1736_173646


namespace pirate_treasure_l1736_173608

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end pirate_treasure_l1736_173608
