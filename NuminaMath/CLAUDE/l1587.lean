import Mathlib

namespace product_remainder_l1587_158796

theorem product_remainder (a b : ℕ) (ha : a % 3 = 2) (hb : b % 3 = 2) : (a * b) % 3 = 1 := by
  sorry

end product_remainder_l1587_158796


namespace profit_starts_third_year_max_average_profit_at_six_option_i_more_cost_effective_l1587_158761

-- Define f(n) in ten thousand yuan
def f (n : ℕ) : ℤ := -2*n^2 + 40*n - 72

-- Question 1: Prove that the factory starts to make a profit from the third year
theorem profit_starts_third_year : 
  ∀ n : ℕ, n > 0 → (f n > 0 ↔ n ≥ 3) :=
sorry

-- Question 2: Prove that the annual average net profit reaches its maximum when n = 6
theorem max_average_profit_at_six :
  ∀ n : ℕ, n > 0 → f n / n ≤ f 6 / 6 :=
sorry

-- Question 3: Prove that option (i) is more cost-effective
theorem option_i_more_cost_effective :
  f 6 + 48 > f 10 + 10 :=
sorry

end profit_starts_third_year_max_average_profit_at_six_option_i_more_cost_effective_l1587_158761


namespace quadratic_rewrite_product_l1587_158714

theorem quadratic_rewrite_product (p q r : ℤ) : 
  (∀ x, 4 * x^2 - 20 * x - 32 = (p * x + q)^2 + r) → p * q = -10 := by
  sorry

end quadratic_rewrite_product_l1587_158714


namespace intersection_nonempty_implies_a_greater_than_neg_one_l1587_158704

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem intersection_nonempty_implies_a_greater_than_neg_one (a : ℝ) :
  (A ∩ B a).Nonempty → a > -1 := by
  sorry

end intersection_nonempty_implies_a_greater_than_neg_one_l1587_158704


namespace mittens_per_box_l1587_158727

theorem mittens_per_box (num_boxes : ℕ) (scarves_per_box : ℕ) (total_items : ℕ) 
  (h1 : num_boxes = 4)
  (h2 : scarves_per_box = 2)
  (h3 : total_items = 32) :
  (total_items - num_boxes * scarves_per_box) / num_boxes = 6 :=
by sorry

end mittens_per_box_l1587_158727


namespace distance_from_origin_l1587_158772

theorem distance_from_origin (a : ℝ) : |a| = 4 → a = 4 ∨ a = -4 := by sorry

end distance_from_origin_l1587_158772


namespace sufficient_not_necessary_condition_l1587_158721

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l1587_158721


namespace det_transformation_l1587_158726

/-- Given a 2x2 matrix with determinant 6, prove that a specific transformation of this matrix results in a determinant of 24. -/
theorem det_transformation (p q r s : ℝ) (h : Matrix.det !![p, q; r, s] = 6) :
  Matrix.det !![p, 8*p + 4*q; r, 8*r + 4*s] = 24 := by
  sorry

end det_transformation_l1587_158726


namespace unique_numbers_proof_l1587_158720

theorem unique_numbers_proof (a b : ℕ) : 
  a ≠ b →                 -- The numbers are distinct
  a > 11 →                -- a is greater than 11
  b > 11 →                -- b is greater than 11
  a + b = 28 →            -- Their sum is 28
  (Even a ∨ Even b) →     -- At least one of them is even
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) := by
sorry

end unique_numbers_proof_l1587_158720


namespace susie_piggy_bank_l1587_158703

theorem susie_piggy_bank (X : ℝ) : X + 0.2 * X = 240 → X = 200 := by
  sorry

end susie_piggy_bank_l1587_158703


namespace polynomial_perfect_square_l1587_158786

/-- The polynomial function P(x) = x^4 + 2x^3 - 2x^2 - 4x - 5 -/
def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 2*x^2 - 4*x - 5

/-- A function is a perfect square if there exists a real function g such that f(x) = g(x)^2 for all x -/
def is_perfect_square (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = (g x)^2

theorem polynomial_perfect_square :
  ∀ x : ℝ, is_perfect_square P ↔ (x = 3 ∨ x = -3) :=
by sorry

end polynomial_perfect_square_l1587_158786


namespace line_representation_l1587_158734

/-- A line in the xy-plane is represented by the equation y = k(x+1) -/
structure Line where
  k : ℝ

/-- The point (-1,0) in the xy-plane -/
def point : ℝ × ℝ := (-1, 0)

/-- A line passes through a point if the point satisfies the line's equation -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * (p.1 + 1)

/-- A line is perpendicular to the x-axis if its slope is undefined (i.e., infinite) -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  l.k = 0

/-- Main theorem: The equation y = k(x+1) represents all lines passing through
    the point (-1,0) and not perpendicular to the x-axis -/
theorem line_representation (l : Line) :
  (passes_through l point ∧ ¬perpendicular_to_x_axis l) ↔ 
  ∃ (k : ℝ), l = Line.mk k :=
sorry

end line_representation_l1587_158734


namespace candy_removal_time_l1587_158729

/-- Represents a position in a 3D grid -/
structure Position where
  x : Nat
  y : Nat
  z : Nat

/-- Calculates the layer sum for a given position -/
def layerSum (p : Position) : Nat :=
  p.x + p.y + p.z

/-- Represents the rectangular prism of candies -/
def CandyPrism :=
  {p : Position | p.x ≤ 3 ∧ p.y ≤ 4 ∧ p.z ≤ 5}

/-- Theorem stating that it takes 10 minutes to remove all candies -/
theorem candy_removal_time : 
  ∀ (start : Position), 
    start ∈ CandyPrism → 
    layerSum start = 3 → 
    (∀ (p : Position), p ∈ CandyPrism → layerSum p ≤ 12) →
    (∃ (p : Position), p ∈ CandyPrism ∧ layerSum p = 12) →
    10 = (12 - layerSum start + 1) :=
by sorry

end candy_removal_time_l1587_158729


namespace price_difference_in_cents_l1587_158775

-- Define the list price and discounts
def list_price : ℚ := 5999 / 100  -- $59.99 represented as a rational number
def tech_bargains_discount : ℚ := 15  -- $15 off
def budget_bytes_discount_rate : ℚ := 30 / 100  -- 30% off

-- Calculate the sale prices
def tech_bargains_price : ℚ := list_price - tech_bargains_discount
def budget_bytes_price : ℚ := list_price * (1 - budget_bytes_discount_rate)

-- Find the cheaper price
def cheaper_price : ℚ := min tech_bargains_price budget_bytes_price
def more_expensive_price : ℚ := max tech_bargains_price budget_bytes_price

-- Define the theorem
theorem price_difference_in_cents : 
  (more_expensive_price - cheaper_price) * 100 = 300 := by
  sorry

end price_difference_in_cents_l1587_158775


namespace parabola_and_tangent_circle_l1587_158752

noncomputable section

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the directrix l
def directrix_l (x : ℝ) : Prop := x = -2

-- Define point P on the directrix
def point_P (t : ℝ) : ℝ × ℝ := (-2, 3*t - 1/t)

-- Define point Q on the y-axis
def point_Q (t : ℝ) : ℝ × ℝ := (0, 2*t)

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x-2)^2 + y^2 = 4

-- Define a line through two points
def line_through (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Main theorem
theorem parabola_and_tangent_circle (t : ℝ) (ht : t ≠ 0) :
  (∀ x y, parabola_C x y ↔ y^2 = 8*x) ∧
  (∀ x y, line_through (point_P t) (point_Q t) x y →
    ∃ x0 y0, circle_M x0 y0 ∧
      ((x - x0)^2 + (y - y0)^2 = 4 ∧
       ((x - x0) * (x - x0) + (y - y0) * (y - y0) = 4))) :=
sorry

end parabola_and_tangent_circle_l1587_158752


namespace remainder_theorem_l1587_158766

theorem remainder_theorem (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 4 * u * y) % y = v := by
sorry

end remainder_theorem_l1587_158766


namespace lcm_of_numbers_with_given_hcf_and_product_l1587_158718

theorem lcm_of_numbers_with_given_hcf_and_product :
  ∀ a b : ℕ+,
  (Nat.gcd a.val b.val = 11) →
  (a * b = 1991) →
  (Nat.lcm a.val b.val = 181) :=
by
  sorry

end lcm_of_numbers_with_given_hcf_and_product_l1587_158718


namespace comparison_inequality_l1587_158763

theorem comparison_inequality : ∀ x : ℝ, (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry

end comparison_inequality_l1587_158763


namespace rancher_unique_solution_l1587_158792

/-- Represents the solution to the rancher's problem -/
structure RancherSolution where
  steers : ℕ
  cows : ℕ

/-- Checks if a given solution satisfies all conditions of the rancher's problem -/
def is_valid_solution (s : RancherSolution) : Prop :=
  s.steers > 0 ∧ 
  s.cows > 0 ∧ 
  30 * s.steers + 25 * s.cows = 1200

/-- Theorem stating that (5, 42) is the only valid solution to the rancher's problem -/
theorem rancher_unique_solution : 
  ∀ s : RancherSolution, is_valid_solution s ↔ s.steers = 5 ∧ s.cows = 42 := by
  sorry

#check rancher_unique_solution

end rancher_unique_solution_l1587_158792


namespace max_elevation_l1587_158736

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 160 * t - 16 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (u : ℝ), s u ≤ s t ∧ s t = 400 := by
  sorry

end max_elevation_l1587_158736


namespace problem_1_problem_2_l1587_158776

-- Problem 1
theorem problem_1 (a b : ℝ) : 4 * a^4 * b^3 / ((-2 * a * b)^2) = a^2 * b :=
by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = 5*y^2 - 6*x*y :=
by sorry

end problem_1_problem_2_l1587_158776


namespace cos_six_arccos_one_fourth_l1587_158785

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by
  sorry

end cos_six_arccos_one_fourth_l1587_158785


namespace four_stamps_cost_l1587_158700

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34/100

/-- The cost of n stamps in dollars -/
def n_stamps_cost (n : ℕ) : ℚ := n * stamp_cost

theorem four_stamps_cost :
  n_stamps_cost 4 = 136/100 :=
by sorry

end four_stamps_cost_l1587_158700


namespace flock_size_l1587_158779

/-- Represents the number of sheep in a flock -/
structure Flock :=
  (rams : ℕ)
  (ewes : ℕ)

/-- The ratio of rams to ewes after one ram runs away -/
def ratio_after_ram_leaves (f : Flock) : ℚ :=
  (f.rams - 1 : ℚ) / f.ewes

/-- The ratio of rams to ewes after the ram returns and one ewe runs away -/
def ratio_after_ewe_leaves (f : Flock) : ℚ :=
  (f.rams : ℚ) / (f.ewes - 1)

/-- The theorem stating the total number of sheep in the flock -/
theorem flock_size (f : Flock) :
  (ratio_after_ram_leaves f = 7/5) →
  (ratio_after_ewe_leaves f = 5/3) →
  f.rams + f.ewes = 25 := by
  sorry

end flock_size_l1587_158779


namespace max_table_sum_l1587_158709

def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_partition (l1 l2 : List ℕ) : Prop :=
  l1.length = 4 ∧ l2.length = 3 ∧ (l1 ++ l2).toFinset = numbers.toFinset

def table_sum (l1 l2 : List ℕ) : ℕ := (l1.sum * l2.sum)

theorem max_table_sum :
  ∃ (l1 l2 : List ℕ), is_valid_partition l1 l2 ∧
    (∀ (m1 m2 : List ℕ), is_valid_partition m1 m2 →
      table_sum m1 m2 ≤ table_sum l1 l2) ∧
    table_sum l1 l2 = 841 := by sorry

end max_table_sum_l1587_158709


namespace initial_value_proof_l1587_158770

theorem initial_value_proof (final_number : ℕ) (divisor : ℕ) (h1 : final_number = 859560) (h2 : divisor = 456) :
  ∃ (initial_value : ℕ) (added_number : ℕ),
    initial_value + added_number = final_number ∧
    final_number % divisor = 0 ∧
    initial_value = 859376 := by
  sorry

end initial_value_proof_l1587_158770


namespace grocery_total_l1587_158735

/-- The number of cookie packs Lucy bought -/
def cookie_packs : ℕ := 23

/-- The number of cake packs Lucy bought -/
def cake_packs : ℕ := 4

/-- The total number of grocery packs Lucy bought -/
def total_packs : ℕ := cookie_packs + cake_packs

theorem grocery_total : total_packs = 27 := by
  sorry

end grocery_total_l1587_158735


namespace smallest_solution_quadratic_equation_l1587_158731

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 10 * x^2 - 66 * x + 56
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = 8/5 := by
  sorry

end smallest_solution_quadratic_equation_l1587_158731


namespace exists_greater_than_product_l1587_158745

/-- A doubly infinite array of positive integers -/
def InfiniteArray := ℕ+ → ℕ+ → ℕ+

/-- The property that each positive integer appears exactly eight times in the array -/
def EightOccurrences (a : InfiniteArray) : Prop :=
  ∀ k : ℕ+, (∃ (S : Finset (ℕ+ × ℕ+)), S.card = 8 ∧ (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ a p.1 p.2 = k))

theorem exists_greater_than_product (a : InfiniteArray) (h : EightOccurrences a) :
  ∃ (m n : ℕ+), a m n > m * n := by
  sorry

end exists_greater_than_product_l1587_158745


namespace expression_value_at_three_l1587_158742

theorem expression_value_at_three :
  ∀ x : ℝ, x ≠ 2 → x = 3 → (x^2 - 5*x + 6) / (x - 2) = 0 := by
  sorry

end expression_value_at_three_l1587_158742


namespace base_five_digits_of_1250_l1587_158795

theorem base_five_digits_of_1250 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1250 ∧ 1250 < 5^n ∧ n = 5 := by
  sorry

end base_five_digits_of_1250_l1587_158795


namespace remainder_sum_of_powers_l1587_158760

theorem remainder_sum_of_powers (n : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_sum_of_powers_l1587_158760


namespace range_of_m_l1587_158751

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(¬(q m)) → m ≤ 1 :=
by sorry

end range_of_m_l1587_158751


namespace max_value_cos_sin_l1587_158771

theorem max_value_cos_sin (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ ≤ Real.sqrt (a^2 + b^2)) ∧ 
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ = Real.sqrt (a^2 + b^2)) := by
  sorry

end max_value_cos_sin_l1587_158771


namespace scout_troop_profit_scout_troop_profit_is_100_l1587_158744

/-- Calculates the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (total_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : ℚ :=
  let cost_per_bar := (2 : ℚ) / 5
  let sell_per_bar := (1 : ℚ) / 2
  let total_cost := total_bars * cost_per_bar
  let total_revenue := total_bars * sell_per_bar
  total_revenue - total_cost

/-- Proves that the scout troop's profit is $100 -/
theorem scout_troop_profit_is_100 :
  scout_troop_profit 1000 ((2 : ℚ) / 5) ((1 : ℚ) / 2) = 100 := by
  sorry

end scout_troop_profit_scout_troop_profit_is_100_l1587_158744


namespace equation_solutions_l1587_158733

theorem equation_solutions :
  ∀ a b c : ℕ+,
  (1 : ℚ) / a + (2 : ℚ) / b - (3 : ℚ) / c = 1 ↔
  (∃ n : ℕ+, a = 1 ∧ b = 2 * n ∧ c = 3 * n) ∨
  (a = 2 ∧ b = 1 ∧ c = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 18) :=
by sorry

end equation_solutions_l1587_158733


namespace banana_count_l1587_158740

/-- Represents the contents and costs of a fruit basket -/
structure FruitBasket where
  num_bananas : ℕ
  num_apples : ℕ
  num_strawberries : ℕ
  num_avocados : ℕ
  num_grape_bunches : ℕ
  banana_cost : ℚ
  apple_cost : ℚ
  strawberry_dozen_cost : ℚ
  avocado_cost : ℚ
  half_grape_bunch_cost : ℚ
  total_cost : ℚ

/-- Theorem stating the number of bananas in the fruit basket -/
theorem banana_count (basket : FruitBasket) 
  (h1 : basket.num_apples = 3)
  (h2 : basket.num_strawberries = 24)
  (h3 : basket.num_avocados = 2)
  (h4 : basket.num_grape_bunches = 1)
  (h5 : basket.banana_cost = 1)
  (h6 : basket.apple_cost = 2)
  (h7 : basket.strawberry_dozen_cost = 4)
  (h8 : basket.avocado_cost = 3)
  (h9 : basket.half_grape_bunch_cost = 2)
  (h10 : basket.total_cost = 28) :
  basket.num_bananas = 4 := by
  sorry

end banana_count_l1587_158740


namespace arithmetic_evaluation_l1587_158723

theorem arithmetic_evaluation : 6 + (3 * 6) - 12 = 12 := by
  sorry

end arithmetic_evaluation_l1587_158723


namespace C_power_50_l1587_158724

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_power_50 : C^50 = !![(-299), (-100); 800, 251] := by sorry

end C_power_50_l1587_158724


namespace no_primes_in_factorial_range_l1587_158788

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k ∈ Set.Ioo (n! + 1) (n! + n), ¬ Nat.Prime k := by
  sorry

end no_primes_in_factorial_range_l1587_158788


namespace rationalization_sum_l1587_158789

theorem rationalization_sum (A B C D : ℤ) : 
  (7 / (3 + Real.sqrt 8) = (A * Real.sqrt B + C) / D) →
  (Nat.gcd A.natAbs C.natAbs = 1) →
  (Nat.gcd A.natAbs D.natAbs = 1) →
  (Nat.gcd C.natAbs D.natAbs = 1) →
  A + B + C + D = 23 := by
sorry

end rationalization_sum_l1587_158789


namespace total_rainfall_2004_l1587_158783

/-- The average monthly rainfall in Mathborough in 2003 -/
def mathborough_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall in Mathborough from 2003 to 2004 -/
def mathborough_increase : ℝ := 5

/-- The average monthly rainfall in Hightown in 2003 -/
def hightown_2003 : ℝ := 38

/-- The increase in average monthly rainfall in Hightown from 2003 to 2004 -/
def hightown_increase : ℝ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem total_rainfall_2004 : 
  (mathborough_2003 + mathborough_increase) * months_in_year = 558 ∧
  (hightown_2003 + hightown_increase) * months_in_year = 492 := by
sorry

end total_rainfall_2004_l1587_158783


namespace point_sum_on_reciprocal_function_l1587_158717

theorem point_sum_on_reciprocal_function (p q : ℝ → ℝ) (h1 : p 4 = 8) (h2 : ∀ x, q x = 1 / p x) :
  4 + q 4 = 33 / 8 := by
  sorry

end point_sum_on_reciprocal_function_l1587_158717


namespace fish_brought_home_l1587_158777

/-- The number of fish Kendra caught -/
def kendras_catch : ℕ := 30

/-- The number of fish Ken released -/
def ken_released : ℕ := 3

/-- The number of fish Ken caught -/
def kens_catch : ℕ := 2 * kendras_catch

/-- The number of fish Ken brought home -/
def ken_brought_home : ℕ := kens_catch - ken_released

/-- The total number of fish brought home by Ken and Kendra -/
def total_brought_home : ℕ := ken_brought_home + kendras_catch

theorem fish_brought_home :
  total_brought_home = 87 :=
by sorry

end fish_brought_home_l1587_158777


namespace abc_inequality_l1587_158711

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a + b + c ≤ a^2 + b^2 + c^2 := by
  sorry

end abc_inequality_l1587_158711


namespace g_negative_six_l1587_158794

def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

theorem g_negative_six (h : g 6 = 12) : g (-6) = -28 := by
  sorry

end g_negative_six_l1587_158794


namespace thabo_book_difference_l1587_158728

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- The conditions of Thabo's book collection -/
def thabosBooks (b : BookCollection) : Prop :=
  b.paperbackFiction + b.paperbackNonfiction + b.hardcoverNonfiction = 200 ∧
  b.paperbackNonfiction > b.hardcoverNonfiction ∧
  b.paperbackFiction = 2 * b.paperbackNonfiction ∧
  b.hardcoverNonfiction = 35

theorem thabo_book_difference (b : BookCollection) 
  (h : thabosBooks b) : 
  b.paperbackNonfiction - b.hardcoverNonfiction = 20 := by
  sorry


end thabo_book_difference_l1587_158728


namespace four_numbers_problem_l1587_158719

theorem four_numbers_problem (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end four_numbers_problem_l1587_158719


namespace path_length_for_73_l1587_158712

/-- The length of a path around squares constructed on segments of a line -/
def path_length (segment_length : ℝ) : ℝ :=
  3 * segment_length

theorem path_length_for_73 :
  path_length 73 = 219 := by sorry

end path_length_for_73_l1587_158712


namespace inequality_proof_l1587_158758

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l1587_158758


namespace axisymmetric_shapes_l1587_158798

-- Define the basic shapes
inductive Shape
  | Triangle
  | Parallelogram
  | Rectangle
  | Circle

-- Define the property of being axisymmetric
def is_axisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.Circle => true
  | _ => false

-- Theorem statement
theorem axisymmetric_shapes :
  ∀ s : Shape, is_axisymmetric s ↔ (s = Shape.Rectangle ∨ s = Shape.Circle) :=
by sorry

end axisymmetric_shapes_l1587_158798


namespace y_investment_calculation_l1587_158759

/-- Represents the investment and profit sharing of two business partners -/
structure BusinessPartnership where
  /-- The amount invested by partner X -/
  x_investment : ℕ
  /-- The amount invested by partner Y -/
  y_investment : ℕ
  /-- The profit share ratio of partner X -/
  x_profit_ratio : ℕ
  /-- The profit share ratio of partner Y -/
  y_profit_ratio : ℕ

/-- Theorem stating that if the profit is shared in ratio 2:6 and X invested 5000, then Y invested 15000 -/
theorem y_investment_calculation (bp : BusinessPartnership) 
  (h1 : bp.x_investment = 5000)
  (h2 : bp.x_profit_ratio = 2)
  (h3 : bp.y_profit_ratio = 6) :
  bp.y_investment = 15000 := by
  sorry


end y_investment_calculation_l1587_158759


namespace cubic_roots_sum_l1587_158739

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - a - 2 = 0) → 
  (b^3 - b - 2 = 0) → 
  (c^3 - c - 2 = 0) → 
  2*a*(b - c)^2 + 2*b*(c - a)^2 + 2*c*(a - b)^2 = -36 := by
  sorry

end cubic_roots_sum_l1587_158739


namespace xy_inequality_l1587_158767

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2) : x * y + 3 ≥ 2 * x + 2 * y := by
  sorry

end xy_inequality_l1587_158767


namespace absolute_value_inequality_rational_inequality_l1587_158741

-- Problem 1
theorem absolute_value_inequality (x : ℝ) :
  (|x - 2| + |2*x - 3| < 4) ↔ (1/3 < x ∧ x < 3) := by sorry

-- Problem 2
theorem rational_inequality (x : ℝ) :
  ((x^2 - 3*x) / (x^2 - x - 2) ≤ x) ↔ 
  ((-1 < x ∧ x ≤ 0) ∨ x = 1 ∨ (2 < x)) := by sorry

end absolute_value_inequality_rational_inequality_l1587_158741


namespace matrix_equality_l1587_158706

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![2, 1; 4, 3]) : 
  B * A = !![2, 1; 4, 3] := by sorry

end matrix_equality_l1587_158706


namespace greatest_integer_radius_l1587_158782

theorem greatest_integer_radius (r : ℕ) (A : ℝ) : 
  A < 75 * Real.pi → A = Real.pi * (r : ℝ)^2 → r ≤ 8 ∧ ∃ (s : ℕ), s = 8 ∧ Real.pi * (s : ℝ)^2 < 75 * Real.pi := by
  sorry

end greatest_integer_radius_l1587_158782


namespace tinas_weekly_income_l1587_158762

/-- Calculates Tina's weekly income based on her work schedule and pay rates. -/
def calculate_weekly_income (hourly_wage : ℚ) (regular_hours : ℚ) (weekday_hours : ℚ) (weekend_hours : ℚ) : ℚ :=
  let overtime_rate := hourly_wage + hourly_wage / 2
  let double_overtime_rate := hourly_wage * 2
  let weekday_pay := (
    hourly_wage * regular_hours + 
    overtime_rate * (weekday_hours - regular_hours)
  ) * 5
  let weekend_pay := (
    hourly_wage * regular_hours + 
    overtime_rate * (regular_hours - regular_hours) +
    double_overtime_rate * (weekend_hours - regular_hours - (regular_hours - regular_hours))
  ) * 2
  weekday_pay + weekend_pay

/-- Theorem stating that Tina's weekly income is $1530.00 given her work schedule and pay rates. -/
theorem tinas_weekly_income :
  calculate_weekly_income 18 8 10 12 = 1530 := by
  sorry

end tinas_weekly_income_l1587_158762


namespace tan_product_equals_neg_one_fifth_l1587_158781

theorem tan_product_equals_neg_one_fifth 
  (α β : ℝ) (h : 2 * Real.cos (2 * α + β) - 3 * Real.cos β = 0) :
  Real.tan α * Real.tan (α + β) = -1/5 := by
  sorry

end tan_product_equals_neg_one_fifth_l1587_158781


namespace polygon_intersection_points_l1587_158768

/-- The number of intersection points between two regular polygons inscribed in a circle -/
def intersectionPoints (n m : ℕ) : ℕ := 2 * min n m

/-- The total number of intersection points for four regular polygons -/
def totalIntersectionPoints (a b c d : ℕ) : ℕ :=
  intersectionPoints a b + intersectionPoints a c + intersectionPoints a d +
  intersectionPoints b c + intersectionPoints b d + intersectionPoints c d

theorem polygon_intersection_points :
  totalIntersectionPoints 6 7 8 9 = 80 := by
  sorry

#eval totalIntersectionPoints 6 7 8 9

end polygon_intersection_points_l1587_158768


namespace inequality_solution_l1587_158716

/-- Given an inequality ax^2 - 3x + 2 < 0 with solution set {x | 1 < x < b}, prove a + b = 3 -/
theorem inequality_solution (a b : ℝ) 
  (h : ∀ x, ax^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < b) : 
  a + b = 3 := by
  sorry

end inequality_solution_l1587_158716


namespace value_of_A_l1587_158732

/-- Given the value of letters in words, find the value of A -/
theorem value_of_A (H M A T E : ℤ) : 
  H = 12 →
  M + A + T + H = 40 →
  T + E + A + M = 50 →
  M + E + E + T = 44 →
  A = 28 := by
sorry

end value_of_A_l1587_158732


namespace no_solution_implies_a_less_than_two_l1587_158748

theorem no_solution_implies_a_less_than_two (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - 1| > a) → a < 2 := by
  sorry

end no_solution_implies_a_less_than_two_l1587_158748


namespace locus_of_Q_is_ellipse_l1587_158738

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 16 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x / 12 + y / 8 = 1

/-- Point R is on ellipse C -/
def R_on_C (xR yR : ℝ) : Prop := C xR yR

/-- Point P is on line l -/
def P_on_l (xP yP : ℝ) : Prop := l xP yP

/-- Q is on OP and satisfies |OQ| * |OP| = |OR|² -/
def Q_condition (xQ yQ xP yP xR yR : ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ xQ = t * xP ∧ yQ = t * yP ∧
  t * (xP^2 + yP^2) = xR^2 + yR^2

/-- The resulting ellipse for Q -/
def Q_ellipse (x y : ℝ) : Prop := (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1

/-- Main theorem: The locus of Q is the ellipse (x-1)²/(5/2) + (y-1)²/(5/3) = 1 -/
theorem locus_of_Q_is_ellipse :
  ∀ (xQ yQ xP yP xR yR : ℝ),
    P_on_l xP yP →
    R_on_C xR yR →
    Q_condition xQ yQ xP yP xR yR →
    Q_ellipse xQ yQ :=
sorry

end locus_of_Q_is_ellipse_l1587_158738


namespace sticker_distribution_l1587_158778

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 29 := by sorry

end sticker_distribution_l1587_158778


namespace evaluate_expression_l1587_158702

theorem evaluate_expression : (16^24) / (64^8) = 16^8 := by
  sorry

end evaluate_expression_l1587_158702


namespace max_pie_pieces_l1587_158755

/-- Represents a five-digit number with distinct digits -/
def DistinctFiveDigitNumber (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000 ∧ (∀ i j, i ≠ j → (x / 10^i) % 10 ≠ (x / 10^j) % 10)

/-- The maximum number of pieces that can be obtained when dividing a pie -/
theorem max_pie_pieces :
  ∃ (n : ℕ) (pie piece : ℕ),
    n = 7 ∧
    DistinctFiveDigitNumber pie ∧
    10000 ≤ piece ∧ piece < 100000 ∧
    pie = piece * n ∧
    (∀ m > n, ¬∃ (p q : ℕ),
      DistinctFiveDigitNumber p ∧
      10000 ≤ q ∧ q < 100000 ∧
      p = q * m) :=
by sorry

end max_pie_pieces_l1587_158755


namespace max_distance_with_tire_swap_l1587_158754

/-- Represents the maximum distance a car can travel with tire swapping -/
def maxDistanceWithTireSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + (rearTireLife - frontTireLife)

/-- Theorem: The maximum distance a car can travel with given tire lifespans -/
theorem max_distance_with_tire_swap :
  maxDistanceWithTireSwap 20000 30000 = 30000 := by
  sorry

#eval maxDistanceWithTireSwap 20000 30000

end max_distance_with_tire_swap_l1587_158754


namespace football_group_stage_teams_l1587_158743

/-- The number of participating teams in the football group stage -/
def num_teams : ℕ := 16

/-- The number of stadiums used -/
def num_stadiums : ℕ := 6

/-- The number of games scheduled at each stadium per day -/
def games_per_stadium_per_day : ℕ := 4

/-- The number of consecutive days to complete all group stage matches -/
def num_days : ℕ := 10

theorem football_group_stage_teams :
  num_teams * (num_teams - 1) = num_stadiums * games_per_stadium_per_day * num_days :=
by sorry

end football_group_stage_teams_l1587_158743


namespace parakeets_per_cage_l1587_158707

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 8)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 7 := by
  sorry

end parakeets_per_cage_l1587_158707


namespace related_chord_midpoint_x_max_related_chord_length_l1587_158730

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- Checks if a chord AB is a "related chord" of point P -/
def isRelatedChord (A B P : Point) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ A ≠ B ∧
  P.y = 0 ∧
  ∃ (M : Point), M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2 ∧
  (M.y - P.y) * (A.x - B.x) = (P.x - M.x) * (A.y - B.y)

/-- The x-coordinate of the midpoint of any "related chord" of P(4,0) is 2 -/
theorem related_chord_midpoint_x (A B : Point) :
  isRelatedChord A B (Point.mk 4 0) → (A.x + B.x) / 2 = 2 := by sorry

/-- The maximum length of all "related chords" of P(4,0) is 6 -/
theorem max_related_chord_length (A B : Point) :
  isRelatedChord A B (Point.mk 4 0) →
  ∃ (max_length : ℝ), max_length = 6 ∧
  ∀ (C D : Point), isRelatedChord C D (Point.mk 4 0) →
  ((C.x - D.x)^2 + (C.y - D.y)^2)^(1/2) ≤ max_length := by sorry

end related_chord_midpoint_x_max_related_chord_length_l1587_158730


namespace percentage_problem_l1587_158725

theorem percentage_problem (P : ℝ) (N : ℝ) 
  (h1 : (P / 100) * N = 200)
  (h2 : 1.2 * N = 1200) : P = 20 := by
sorry

end percentage_problem_l1587_158725


namespace quadratic_coefficients_l1587_158715

/-- A quadratic function with vertex (h, k) and y-intercept c can be represented as f(x) = a(x - h)² + k,
    where a ≠ 0 and f(0) = c. -/
def quadratic_function (a h k c : ℝ) (ha : a ≠ 0) (f : ℝ → ℝ) :=
  ∀ x, f x = a * (x - h)^2 + k ∧ f 0 = c

theorem quadratic_coefficients (f : ℝ → ℝ) (a h k c : ℝ) (ha : a ≠ 0) :
  quadratic_function a h k c ha f →
  h = 2 ∧ k = -1 ∧ c = 11 →
  a = 3 ∧ 
  ∃ b, ∀ x, f x = 3 * x^2 + b * x + 11 ∧ b = -12 :=
by sorry

end quadratic_coefficients_l1587_158715


namespace segment_count_after_16_iterations_l1587_158753

/-- The number of segments after n iterations of the division process -/
def num_segments (n : ℕ) : ℕ := 2^n

/-- The length of each segment after n iterations of the division process -/
def segment_length (n : ℕ) : ℚ := (1 : ℚ) / 3^n

theorem segment_count_after_16_iterations :
  num_segments 16 = 2^16 := by sorry

end segment_count_after_16_iterations_l1587_158753


namespace sine_graph_shift_l1587_158708

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (3 * (x - 5 * π / 18) + π / 2) = 2 * Real.sin (3 * x - π / 3) := by
  sorry

end sine_graph_shift_l1587_158708


namespace complex_power_pure_integer_l1587_158710

def i : ℂ := Complex.I

theorem complex_power_pure_integer :
  ∃ (n : ℤ), ∃ (m : ℤ), (3 * n + 2 * i) ^ 6 = m := by
  sorry

end complex_power_pure_integer_l1587_158710


namespace fence_bricks_l1587_158799

/-- Calculates the number of bricks needed for a rectangular fence --/
def bricks_needed (length width height depth : ℕ) : ℕ :=
  4 * length * width * depth

theorem fence_bricks :
  bricks_needed 20 5 2 = 800 := by
  sorry

end fence_bricks_l1587_158799


namespace vectors_opposite_x_value_l1587_158722

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![2*x, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![4, x]

-- Define the condition that vectors are in opposite directions
def opposite_directions (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ (∀ i, v i = -k * w i)

-- Theorem statement
theorem vectors_opposite_x_value :
  ∀ x : ℝ, opposite_directions (a x) (b x) → x = -Real.sqrt 2 :=
by sorry

end vectors_opposite_x_value_l1587_158722


namespace intersection_P_Q_l1587_158797

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem intersection_P_Q : P ∩ Q = {1, 2, 3} := by sorry

end intersection_P_Q_l1587_158797


namespace quick_calculation_formula_l1587_158737

theorem quick_calculation_formula (a b : ℝ) :
  (100 + a) * (100 + b) = ((100 + a) + (100 + b) - 100) * 100 + a * b ∧
  (100 + a) * (100 - b) = ((100 + a) + (100 - b) - 100) * 100 + a * (-b) ∧
  (100 - a) * (100 + b) = ((100 - a) + (100 + b) - 100) * 100 + (-a) * b ∧
  (100 - a) * (100 - b) = ((100 - a) + (100 - b) - 100) * 100 + (-a) * (-b) :=
by sorry

end quick_calculation_formula_l1587_158737


namespace hyperbola_equation_l1587_158746

-- Define the general form of a hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the given hyperbola with known asymptotes
def given_hyperbola (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 = 1

-- Define the given ellipse with known foci
def given_ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 2 = 1

-- Theorem stating the equation of the hyperbola
theorem hyperbola_equation :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), hyperbola a b x y ↔ given_hyperbola x y) ∧
    (∀ (x : ℝ), x^2 = 18 → hyperbola a b x 0) ∧
    a = 4 ∧ b = Real.sqrt 2 :=
sorry

end hyperbola_equation_l1587_158746


namespace fraction_sum_integer_l1587_158790

theorem fraction_sum_integer (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : ∃ n : ℤ, (a * b) / c + (b * c) / a + (c * a) / b = n) : 
  (∃ n1 : ℤ, (a * b) / c = n1) ∧ (∃ n2 : ℤ, (b * c) / a = n2) ∧ (∃ n3 : ℤ, (c * a) / b = n3) :=
sorry

end fraction_sum_integer_l1587_158790


namespace university_theater_tickets_l1587_158705

/-- The total number of tickets sold at University Theater -/
def total_tickets (adult_price senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) : ℕ :=
  senior_tickets + ((total_receipts - senior_price * senior_tickets) / adult_price)

/-- Theorem stating that the total number of tickets sold is 509 -/
theorem university_theater_tickets :
  total_tickets 21 15 8748 327 = 509 := by
  sorry

end university_theater_tickets_l1587_158705


namespace rectangle_width_l1587_158749

/-- Given a rectangle ABCD with length 25 yards and an inscribed rhombus AFCE with perimeter 82 yards, 
    the width of the rectangle is equal to √(420.25 / 2) yards. -/
theorem rectangle_width (length : ℝ) (perimeter : ℝ) (width : ℝ) : 
  length = 25 →
  perimeter = 82 →
  width = Real.sqrt (420.25 / 2) :=
by sorry

end rectangle_width_l1587_158749


namespace solid_yellow_marbles_percentage_l1587_158793

theorem solid_yellow_marbles_percentage
  (total_marbles : ℝ)
  (solid_color_percentage : ℝ)
  (solid_color_not_yellow_percentage : ℝ)
  (h1 : solid_color_percentage = 90)
  (h2 : solid_color_not_yellow_percentage = 85)
  : (solid_color_percentage - solid_color_not_yellow_percentage) * total_marbles / 100 = 5 * total_marbles / 100 :=
by sorry

end solid_yellow_marbles_percentage_l1587_158793


namespace consecutive_even_integers_square_product_l1587_158713

theorem consecutive_even_integers_square_product : 
  ∀ (a b c : ℤ),
  (b = a + 2 ∧ c = b + 2) →  -- consecutive even integers
  (a * b * c = 12 * (a + b + c)) →  -- product is 12 times their sum
  (a^2 * b^2 * c^2 = 36864) :=  -- product of squares is 36864
by
  sorry

end consecutive_even_integers_square_product_l1587_158713


namespace phone_number_combinations_l1587_158773

def first_four_digits : ℕ := 12

def fifth_digit_options : ℕ := 2

def sixth_digit_options : ℕ := 10

theorem phone_number_combinations : 
  first_four_digits * fifth_digit_options * sixth_digit_options = 240 := by
  sorry

end phone_number_combinations_l1587_158773


namespace plant_supplier_remaining_money_l1587_158780

/-- Calculates the remaining money for a plant supplier after sales and expenses. -/
theorem plant_supplier_remaining_money
  (orchid_price : ℕ)
  (orchid_quantity : ℕ)
  (money_plant_price : ℕ)
  (money_plant_quantity : ℕ)
  (worker_pay : ℕ)
  (worker_count : ℕ)
  (pot_cost : ℕ)
  (h1 : orchid_price = 50)
  (h2 : orchid_quantity = 20)
  (h3 : money_plant_price = 25)
  (h4 : money_plant_quantity = 15)
  (h5 : worker_pay = 40)
  (h6 : worker_count = 2)
  (h7 : pot_cost = 150) :
  (orchid_price * orchid_quantity + money_plant_price * money_plant_quantity) -
  (worker_pay * worker_count + pot_cost) = 1145 := by
  sorry

end plant_supplier_remaining_money_l1587_158780


namespace vinnie_saturday_words_l1587_158784

/-- The number of words Vinnie wrote on Saturday -/
def saturday_words : ℕ := sorry

/-- The word limit -/
def word_limit : ℕ := 1000

/-- The number of words Vinnie wrote on Sunday -/
def sunday_words : ℕ := 650

/-- The number of words Vinnie exceeded the limit by -/
def excess_words : ℕ := 100

/-- Theorem stating that Vinnie wrote 450 words on Saturday -/
theorem vinnie_saturday_words :
  saturday_words = 450 ∧
  saturday_words + sunday_words = word_limit + excess_words :=
sorry

end vinnie_saturday_words_l1587_158784


namespace final_walnut_count_l1587_158765

/-- The number of walnuts left in the burrow after the squirrels' actions --/
def walnuts_left (initial_walnuts boy_walnuts boy_dropped girl_walnuts girl_eaten : ℕ) : ℕ :=
  initial_walnuts + (boy_walnuts - boy_dropped) + girl_walnuts - girl_eaten

/-- Theorem stating the final number of walnuts in the burrow --/
theorem final_walnut_count :
  walnuts_left 12 6 1 5 2 = 20 := by
  sorry

end final_walnut_count_l1587_158765


namespace simplified_expression_equals_one_l1587_158764

theorem simplified_expression_equals_one (a : ℚ) (h : a = 1/2) :
  (1 / (a + 2) + 1 / (a - 2)) / (1 / (a^2 - 4)) = 1 := by
  sorry

end simplified_expression_equals_one_l1587_158764


namespace unique_modular_solution_l1587_158701

theorem unique_modular_solution : 
  ∀ n : ℤ, (10 ≤ n ∧ n ≤ 20) ∧ (n ≡ 7882 [ZMOD 7]) → n = 14 := by
  sorry

end unique_modular_solution_l1587_158701


namespace not_p_sufficient_not_necessary_for_q_l1587_158769

-- Define p and q
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Define the negation of p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, not_p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(not_p x)) :=
by sorry

end not_p_sufficient_not_necessary_for_q_l1587_158769


namespace problem_solution_l1587_158750

theorem problem_solution : 
  let left_sum := 5 + 6 + 7 + 8 + 9
  let right_sum := 2005 + 2006 + 2007 + 2008 + 2009
  ∀ N : ℝ, (left_sum / 5 : ℝ) = (right_sum / N : ℝ) → N = 1433 :=
by
  sorry

end problem_solution_l1587_158750


namespace exists_four_axes_symmetry_l1587_158757

/-- A type representing a figure on a grid paper -/
structure GridFigure where
  cells : Set (ℤ × ℤ)

/-- A type representing an axis of symmetry -/
structure AxisOfSymmetry where
  -- Define properties of an axis of symmetry

/-- Function to count the number of axes of symmetry in a figure -/
def countAxesOfSymmetry (f : GridFigure) : ℕ := sorry

/-- Function to shade one more cell in a figure -/
def shadeOneMoreCell (f : GridFigure) : GridFigure := sorry

/-- Theorem stating that it's possible to create a figure with four axes of symmetry 
    by shading one more cell in a figure with no axes of symmetry -/
theorem exists_four_axes_symmetry :
  ∃ (f : GridFigure), 
    countAxesOfSymmetry f = 0 ∧ 
    ∃ (g : GridFigure), g = shadeOneMoreCell f ∧ countAxesOfSymmetry g = 4 := by
  sorry

end exists_four_axes_symmetry_l1587_158757


namespace factorial_500_properties_l1587_158774

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The highest power of 3 that divides n! -/
def highestPowerOfThree (n : ℕ) : ℕ := sorry

/-- Theorem about 500! -/
theorem factorial_500_properties :
  (trailingZeroes 500 = 124) ∧ (highestPowerOfThree 500 = 247) := by sorry

end factorial_500_properties_l1587_158774


namespace point_on_line_l1587_158756

theorem point_on_line : ∃ (x y : ℚ), x = 3 ∧ y = 16/7 ∧ 4*x + 7*y = 28 := by
  sorry

end point_on_line_l1587_158756


namespace square_side_length_l1587_158791

theorem square_side_length (area : ℚ) (h : area = 9/16) : 
  ∃ (side : ℚ), side * side = area ∧ side = 3/4 := by
  sorry

end square_side_length_l1587_158791


namespace clock_hand_overlaps_l1587_158787

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_hand : ℝ)
  (minute_hand : ℝ)

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of overlaps in a 12-hour period -/
def overlaps_per_half_day : ℕ := 11

/-- Calculates the number of times the hour and minute hands overlap in a day -/
def overlaps_per_day (c : Clock) : ℕ :=
  2 * overlaps_per_half_day

/-- Theorem: The number of times the hour and minute hands of a clock overlap in a 24-hour day is 22 -/
theorem clock_hand_overlaps :
  ∀ c : Clock, overlaps_per_day c = 22 :=
by
  sorry

end clock_hand_overlaps_l1587_158787


namespace cat_weight_sum_l1587_158747

/-- The combined weight of three cats -/
def combined_weight (w1 w2 w3 : ℕ) : ℕ := w1 + w2 + w3

/-- Theorem: The combined weight of three cats weighing 2, 7, and 4 pounds is 13 pounds -/
theorem cat_weight_sum : combined_weight 2 7 4 = 13 := by
  sorry

end cat_weight_sum_l1587_158747
