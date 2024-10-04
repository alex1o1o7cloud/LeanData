import Mathlib

namespace probability_of_selecting_cooking_l119_119349

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119349


namespace parabola_intersects_line_exactly_once_l119_119282

theorem parabola_intersects_line_exactly_once (p q : ℚ) : 
  (∀ x : ℝ, 2 * (x - p) ^ 2 = x - 4 ↔ p = 31 / 8) ∧ 
  (∀ x : ℝ, 2 * x ^ 2 - q = x - 4 ↔ q = 31 / 8) := 
by 
  sorry

end parabola_intersects_line_exactly_once_l119_119282


namespace probability_of_selecting_cooking_l119_119296

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119296


namespace normal_distribution_half_probability_l119_119004

variable (σ : ℝ)

theorem normal_distribution_half_probability 
  (ξ : ℝ → Prop)
  (h : ∀ (x : ℝ), ξ x ↔ true )
  (μ : ℝ)
  (hξ : Normal μ σ)
  (hx : μ = 2016):
  P (ξ < 2016) = 1 / 2 := by
  sorry

end normal_distribution_half_probability_l119_119004


namespace trigonometric_fraction_value_l119_119174

theorem trigonometric_fraction_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end trigonometric_fraction_value_l119_119174


namespace total_profit_equals_254000_l119_119391

-- Definitions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 6000
def investment_D : ℕ := 10000

def time_A : ℕ := 12
def time_B : ℕ := 8
def time_C : ℕ := 6
def time_D : ℕ := 9

def capital_months (investment : ℕ) (time : ℕ) : ℕ := investment * time

-- Given conditions
def A_capital_months := capital_months investment_A time_A
def B_capital_months := capital_months investment_B time_B
def C_capital_months := capital_months investment_C time_C
def D_capital_months := capital_months investment_D time_D

def total_capital_months : ℕ := A_capital_months + B_capital_months + C_capital_months + D_capital_months

def C_profit : ℕ := 36000

-- Proportion equation
def total_profit (C_capital_months : ℕ) (total_capital_months : ℕ) (C_profit : ℕ) : ℕ :=
  (C_profit * total_capital_months) / C_capital_months

-- Theorem statement
theorem total_profit_equals_254000 : total_profit C_capital_months total_capital_months C_profit = 254000 := by
  sorry

end total_profit_equals_254000_l119_119391


namespace range_of_a_1_range_of_a_2_l119_119435

-- Definitions based on conditions in a)

def func_domain (a : ℝ) (x : ℝ) : Prop := a * x^2 - 2 * x + 2 > 0

def Q (a : ℝ) : Set ℝ := {x | func_domain a x}

-- Problem 1
theorem range_of_a_1 (a : ℝ) (h1 : a > 0) (h2 : Disjoint {x | 2 ≤ x ∧ x ≤ 3} (Q a)) :
  0 < a ∧ a ≤ 4/9 := sorry

-- Problem 2
theorem range_of_a_2 (a : ℝ) (h : {x | 2 ≤ x ∧ x ≤ 3} ⊆ Q a) :
  a > 1/2 := sorry

end range_of_a_1_range_of_a_2_l119_119435


namespace repeating_decimals_sum_l119_119589

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l119_119589


namespace problem_l119_119778

noncomputable def x : ℕ := 5  -- Define x as the positive integer 5

theorem problem (hx : ∀ x, 1 ≤ x → 1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170 ↔ x = 5) : 1^(5+2) + 2^(5+1) + 3^(5-1) + 4^5 = 1170 :=
by {
  have : 1^(5+2) + 2^(5+1) + 3^(5-1) + 4^5 = 1^7 + 2^6 + 3^4 + 4^5 := by rfl,
  rw [this],
  norm_num,
}

end problem_l119_119778


namespace simplify_expression_l119_119176

theorem simplify_expression (a : ℝ) (h : a > 0) : 
  (a^2 / (a * (a^3) ^ (1 / 2)) ^ (1 / 3)) = a^(7 / 6) :=
sorry

end simplify_expression_l119_119176


namespace arithmetic_sequence_geometric_property_l119_119759

theorem arithmetic_sequence_geometric_property (a : ℕ → ℤ) (d : ℤ) (h_d : d = 2)
  (h_a3 : a 3 = a 1 + 4) (h_a4 : a 4 = a 1 + 6)
  (geo_seq : (a 1 + 4) * (a 1 + 4) = a 1 * (a 1 + 6)) :
  a 2 = -6 := sorry

end arithmetic_sequence_geometric_property_l119_119759


namespace least_four_digit_palindrome_divisible_by_5_l119_119120

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem least_four_digit_palindrome_divisible_by_5 : ∃ n, is_palindrome n ∧ is_divisible_by_5 n ∧ is_four_digit n ∧ ∀ m, is_palindrome m ∧ is_divisible_by_5 m ∧ is_four_digit m → n ≤ m :=
by
  -- proof steps will be here
  sorry

end least_four_digit_palindrome_divisible_by_5_l119_119120


namespace june_vs_christopher_l119_119405

namespace SwordLength

def christopher_length : ℕ := 15
def jameson_length : ℕ := 3 + 2 * christopher_length
def june_length : ℕ := 5 + jameson_length

theorem june_vs_christopher : june_length - christopher_length = 23 := by
  show 5 + (3 + 2 * christopher_length) - christopher_length = 23
  sorry

end SwordLength

end june_vs_christopher_l119_119405


namespace sum_of_ages_l119_119044

-- Define ages of Kiana and her twin brothers
variables (kiana_age : ℕ) (twin_age : ℕ)

-- Define conditions
def age_product_condition : Prop := twin_age * twin_age * kiana_age = 162
def age_less_than_condition : Prop := kiana_age < 10
def twins_older_condition : Prop := twin_age > kiana_age

-- The main problem statement
theorem sum_of_ages (h1 : age_product_condition twin_age kiana_age) (h2 : age_less_than_condition kiana_age) (h3 : twins_older_condition twin_age kiana_age) :
  twin_age * 2 + kiana_age = 20 :=
sorry

end sum_of_ages_l119_119044


namespace max_profit_at_grade_9_l119_119722

def profit (k : ℕ) : ℕ :=
  (8 + 2 * (k - 1)) * (60 - 3 * (k - 1))

theorem max_profit_at_grade_9 : ∀ k, 1 ≤ k ∧ k ≤ 10 → profit k ≤ profit 9 := 
by
  sorry

end max_profit_at_grade_9_l119_119722


namespace determine_k_l119_119620

theorem determine_k (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by
  sorry

end determine_k_l119_119620


namespace percentage_reduction_price_increase_l119_119536

open Real

-- Part 1: Finding the percentage reduction each time
theorem percentage_reduction (P₀ P₂ : ℝ) (x : ℝ) (h₀ : P₀ = 50) (h₁ : P₂ = 32) (h₂ : P₀ * (1 - x) ^ 2 = P₂) :
  x = 0.20 :=
by
  dsimp at h₀ h₁,
  rw h₀ at h₂,
  rw h₁ at h₂,
  simp at h₂,
  sorry

-- Part 2: Determining the price increase per kilogram
theorem price_increase (P y : ℝ) (profit_per_kg : ℝ) (initial_sales : ℝ) 
  (price_increase_limit : ℝ) (sales_decrease_rate : ℝ) (target_profit : ℝ)
  (h₀ : profit_per_kg = 10) (h₁ : initial_sales = 500) (h₂ : price_increase_limit = 8)
  (h₃ : sales_decrease_rate = 20) (h₄ : target_profit = 6000) (0 < y ∧ y ≤ price_increase_limit)
  (h₅ : (profit_per_kg + y) * (initial_sales - sales_decrease_rate * y) = target_profit) :
  y = 5 :=
by
  dsimp at h₀ h₁ h₂ h₃ h₄,
  rw [h₀, h₁, h₂, h₃, h₄] at h₅,
  sorry

end percentage_reduction_price_increase_l119_119536


namespace probability_of_selecting_cooking_l119_119339

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119339


namespace area_inequalities_l119_119764

noncomputable def f1 (x : ℝ) : ℝ := 1 - (1 / 2) * x
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def f3 (x : ℝ) : ℝ := 1 - (1 / 2) * x^2

noncomputable def S1 : ℝ := 1 - (1 / 4)
noncomputable def S2 : ℝ := Real.log 2
noncomputable def S3 : ℝ := (5 / 6)

theorem area_inequalities : S2 < S1 ∧ S1 < S3 := by
  sorry

end area_inequalities_l119_119764


namespace consumption_increase_l119_119876

variable (T C C' : ℝ)
variable (h1 : 0.8 * T * C' = 0.92 * T * C)

theorem consumption_increase (T C C' : ℝ) (h1 : 0.8 * T * C' = 0.92 * T * C) : C' = 1.15 * C :=
by
  sorry

end consumption_increase_l119_119876


namespace probability_of_selecting_cooking_l119_119295

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119295


namespace sum_of_repeating_decimals_l119_119585

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l119_119585


namespace repeating_decimals_sum_l119_119594

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l119_119594


namespace fare_collected_from_I_class_l119_119888

theorem fare_collected_from_I_class (x y : ℝ) 
  (h1 : ∀i, i = x → ∀ii, ii = 4 * x)
  (h2 : ∀f1, f1 = 3 * y)
  (h3 : ∀f2, f2 = y)
  (h4 : x * 3 * y + 4 * x * y = 224000) : 
  x * 3 * y = 96000 :=
by
  sorry

end fare_collected_from_I_class_l119_119888


namespace hyperbola_m_range_l119_119616

theorem hyperbola_m_range (m : ℝ) (h_eq : ∀ x y, (x^2 / m) - (y^2 / (2*m - 1)) = 1) : 
  0 < m ∧ m < 1/2 :=
sorry

end hyperbola_m_range_l119_119616


namespace possible_to_fill_array_l119_119997

open BigOperators

theorem possible_to_fill_array :
  ∃ (f : (Fin 10) × (Fin 10) → ℕ),
    (∀ i j : Fin 10, 
      (i ≠ 0 → f (i, j) ∣ f (i - 1, j) ∧ f (i, j) ≠ f (i - 1, j))) ∧
    (∀ i : Fin 10, ∃ n : ℕ, ∀ j : Fin 10, f (i, j) = n + j) :=
sorry

end possible_to_fill_array_l119_119997


namespace number_of_a_values_l119_119173

theorem number_of_a_values (a : ℝ) :
  (∃ x : ℝ, y = x + 2*a ∧ y = x^3 - 3*a*x + a^3) → a = 0 :=
by
  sorry

end number_of_a_values_l119_119173


namespace find_single_digit_number_l119_119650

theorem find_single_digit_number (n : ℕ) : 
  (5 < n ∧ n < 9 ∧ n > 7) ↔ n = 8 :=
by
  sorry

end find_single_digit_number_l119_119650


namespace coeff_x_squared_l119_119005

theorem coeff_x_squared (n : ℕ) (t h : ℕ)
  (h_t : t = 4^n) 
  (h_h : h = 2^n) 
  (h_sum : t + h = 272)
  (C : ℕ → ℕ → ℕ) -- binomial coefficient notation, we'll skip the direct proof of properties for simplicity
  : (C 4 4) * (3^0) = 1 := 
by 
  /-
  Proof steps (informal, not needed in Lean statement):
  Since the sum of coefficients is t, we have t = 4^n.
  For the sum of binomial coefficients, we have h = 2^n.
  Given t + h = 272, solve for n:
    4^n + 2^n = 272 
    implies 2^n = 16, so n = 4.
  Substitute into the general term (\(T_{r+1}\):
    T_{r+1} = C_4^r * 3^(4-r) * x^((8+r)/6)
  For x^2 term, set (8+r)/6 = 2, yielding r = 4.
  The coefficient is C_4^4 * 3^0 = 1.
  -/
  sorry

end coeff_x_squared_l119_119005


namespace arithmetic_sequence_sum_nine_l119_119613

variable {α : Type*} [LinearOrderedField α]

/-- An arithmetic sequence (a_n) is defined by a starting term a_1 and a common difference d. -/
def arithmetic_seq (a d n : α) : α := a + (n - 1) * d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmetic_sum (a d n : α) : α := n / 2 * (2 * a + (n - 1) * d)

/-- Prove that for a given arithmetic sequence where a_2 + a_4 + a_9 = 24, the sum of the first 9 terms is 72. -/
theorem arithmetic_sequence_sum_nine 
  {a d : α}
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 4 + arithmetic_seq a d 9 = 24) :
  arithmetic_sum a d 9 = 72 := 
by
  sorry

end arithmetic_sequence_sum_nine_l119_119613


namespace f_9_over_2_l119_119470

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end f_9_over_2_l119_119470


namespace customers_in_other_countries_l119_119135

-- Given 
def total_customers : ℕ := 7422
def customers_in_us : ℕ := 723

-- To Prove
theorem customers_in_other_countries : (total_customers - customers_in_us) = 6699 := 
by
  sorry

end customers_in_other_countries_l119_119135


namespace arithmetic_sequence_value_l119_119035

theorem arithmetic_sequence_value (a : ℕ → ℕ) (m : ℕ) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 4) 
  (h_a5 : a 5 = m) 
  (h_a7 : a 7 = 16) : 
  m = 10 := 
by
  sorry

end arithmetic_sequence_value_l119_119035


namespace multiply_expression_l119_119833

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l119_119833


namespace bakery_total_items_l119_119853

theorem bakery_total_items (total_money : ℝ) (cupcake_cost : ℝ) (pastry_cost : ℝ) (max_cupcakes : ℕ) (remaining_money : ℝ) (total_items : ℕ) :
  total_money = 50 ∧ cupcake_cost = 3 ∧ pastry_cost = 2.5 ∧ max_cupcakes = 16 ∧ remaining_money = 2 ∧ total_items = max_cupcakes + 0 → total_items = 16 :=
by
  sorry

end bakery_total_items_l119_119853


namespace rectangle_area_l119_119729

theorem rectangle_area
  (x y : ℝ) -- sides of the rectangle
  (h1 : 2 * x + 2 * y = 12)  -- perimeter
  (h2 : x^2 + y^2 = 25)  -- diagonal
  : x * y = 5.5 :=
sorry

end rectangle_area_l119_119729


namespace compound_interest_at_least_double_l119_119802

theorem compound_interest_at_least_double :
  ∀ t : ℕ, (0 < t) → (1.05 : ℝ)^t > 2 ↔ t ≥ 15 :=
by sorry

end compound_interest_at_least_double_l119_119802


namespace length_vector_eq_three_l119_119644

theorem length_vector_eq_three (A B : ℝ) (hA : A = -1) (hB : B = 2) : |B - A| = 3 :=
by
  sorry

end length_vector_eq_three_l119_119644


namespace find_f_at_9_over_2_l119_119472

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end find_f_at_9_over_2_l119_119472


namespace opposite_of_num_l119_119106

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l119_119106


namespace coefficients_square_sum_l119_119774

theorem coefficients_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1000 * x ^ 3 + 27 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end coefficients_square_sum_l119_119774


namespace candy_distribution_l119_119642

theorem candy_distribution :
  let bags := 4
  let candies := 9
  (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 :=
by
  -- define variables for bags and candies
  let bags := 4
  let candies := 9
  have h : (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 := sorry
  exact h

end candy_distribution_l119_119642


namespace blocks_for_sculpture_l119_119140

noncomputable def volume_block := 8 * 3 * 1
noncomputable def radius_cylinder := 3
noncomputable def height_cylinder := 8
noncomputable def volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder
noncomputable def blocks_needed := Nat.ceil (volume_cylinder / volume_block)

theorem blocks_for_sculpture : blocks_needed = 10 := by
  sorry

end blocks_for_sculpture_l119_119140


namespace men_entered_room_l119_119992

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l119_119992


namespace probability_selecting_cooking_l119_119300

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119300


namespace trains_clear_each_other_in_11_seconds_l119_119518

-- Define the lengths of the trains
def length_train1 := 100  -- in meters
def length_train2 := 120  -- in meters

-- Define the speeds of the trains (in km/h), converted to m/s
def speed_train1 := 42 * 1000 / 3600  -- 42 km/h to m/s
def speed_train2 := 30 * 1000 / 3600  -- 30 km/h to m/s

-- Calculate the total distance to be covered
def total_distance := length_train1 + length_train2  -- in meters

-- Calculate the relative speed when they are moving towards each other
def relative_speed := speed_train1 + speed_train2  -- in m/s

-- Calculate the time required for the trains to be clear of each other (in seconds)
noncomputable def clear_time := total_distance / relative_speed

-- Theorem stating the above
theorem trains_clear_each_other_in_11_seconds :
  clear_time = 11 :=
by
  -- Proof would go here
  sorry

end trains_clear_each_other_in_11_seconds_l119_119518


namespace sequence_general_term_l119_119185

-- Define the sequence and the sum of the sequence
def Sn (n : ℕ) : ℕ := 3 + 2^n

def an (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n - 1)

-- Proposition stating the equivalence
theorem sequence_general_term (n : ℕ) : 
  (n = 1 → an n = 5) ∧ (n ≠ 1 → an n = 2^(n - 1)) :=
by 
  sorry

end sequence_general_term_l119_119185


namespace correct_conclusions_l119_119475

variable (f : ℝ → ℝ)

def condition_1 := ∀ x : ℝ, f (x + 2) = f (2 - (x + 2))
def condition_2 := ∀ x : ℝ, f (-2*x - 1) = -f (2*x + 1)

theorem correct_conclusions 
  (h1 : condition_1 f) 
  (h2 : condition_2 f) : 
  f 1 = f 3 ∧ 
  f 2 + f 4 = 0 ∧ 
  f (-1 / 2) * f (11 / 2) ≤ 0 := 
by 
  sorry

end correct_conclusions_l119_119475


namespace rectangle_length_l119_119528

theorem rectangle_length
    (a : ℕ)
    (b : ℕ)
    (area_square : a * a = 81)
    (width_rect : b = 3)
    (area_equal : a * a = b * (27) )
    : b * 27 = 81 :=
by
  sorry

end rectangle_length_l119_119528


namespace specified_percentage_of_number_is_40_l119_119479

theorem specified_percentage_of_number_is_40 
  (N : ℝ) 
  (hN : (1 / 4) * (1 / 3) * (2 / 5) * N = 25) 
  (P : ℝ) 
  (hP : (P / 100) * N = 300) : 
  P = 40 := 
sorry

end specified_percentage_of_number_is_40_l119_119479


namespace complex_division_l119_119148

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (1 + i) / (1 - i) = i :=
by
  sorry

end complex_division_l119_119148


namespace team_game_probabilities_l119_119731

-- Define the given conditions
def P_A : ℝ := 1 / 3
def P_A_and_B : ℝ := 1 / 6
def P_B_and_C : ℝ := 1 / 5

-- Define probabilities we want to prove
def P_B : ℝ := 1 / 2
def P_C : ℝ := 2 / 5

-- Final probabilities for the team's score
def P_score_4 : ℝ := 3 / 10
def P_next_round : ℝ := 11 / 30

-- Main theorem to verify all the probabilities
theorem team_game_probabilities :
  (P(A) = 1 / 3) ∧ (P(A ∩ B) = 1 / 6) ∧ (P(B ∩ C) = 1 / 5) →
  (P(B) = 1 / 2) ∧ (P(C) = 2 / 5) ∧ (P(score_4) = 3 / 10) ∧ (P(next_round) = 11 / 30) :=
sorry

end team_game_probabilities_l119_119731


namespace repeating_decimal_sum_l119_119565

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l119_119565


namespace percentage_of_tip_l119_119553

-- Given conditions
def steak_cost : ℝ := 20
def drink_cost : ℝ := 5
def total_cost_before_tip : ℝ := 2 * (steak_cost + drink_cost)
def billy_tip_payment : ℝ := 8
def billy_tip_coverage : ℝ := 0.80

-- Required to prove
theorem percentage_of_tip : ∃ P : ℝ, (P = (billy_tip_payment / (billy_tip_coverage * total_cost_before_tip)) * 100) ∧ P = 20 := 
by {
  sorry
}

end percentage_of_tip_l119_119553


namespace min_matches_to_win_champion_min_total_matches_if_wins_11_l119_119689

-- Define the conditions and problem in Lean 4
def teams := ["A", "B", "C"]
def players_per_team : ℕ := 9
def initial_matches : ℕ := 0

-- The minimum number of matches the champion team must win
theorem min_matches_to_win_champion (H : ∀ t ∈ teams, t ≠ "Champion" → players_per_team = 0) :
  initial_matches + 19 = 19 :=
by
  sorry

-- The minimum total number of matches if the champion team wins 11 matches
theorem min_total_matches_if_wins_11 (wins_by_champion : ℕ := 11) (H : wins_by_champion = 11) :
  initial_matches + wins_by_champion + (players_per_team * 2 - wins_by_champion) + 4 = 24 :=
by
  sorry

end min_matches_to_win_champion_min_total_matches_if_wins_11_l119_119689


namespace range_y_minus_2x_l119_119025

theorem range_y_minus_2x (x y : ℝ) (hx : -2 ≤ x ∧ x ≤ 1) (hy : 2 ≤ y ∧ y ≤ 4) :
  0 ≤ y - 2 * x ∧ y - 2 * x ≤ 8 :=
sorry

end range_y_minus_2x_l119_119025


namespace thor_hammer_weight_exceeds_2000_l119_119654

/--  The Mighty Thor uses a hammer that doubles in weight each day as he trains.
      Starting on the first day with a hammer that weighs 7 pounds, prove that
      on the 10th day the hammer's weight exceeds 2000 pounds. 
-/
theorem thor_hammer_weight_exceeds_2000 :
  ∃ n : ℕ, 7 * 2^(n - 1) > 2000 ∧ n = 10 :=
by
  sorry

end thor_hammer_weight_exceeds_2000_l119_119654


namespace diagonal_AC_length_l119_119209

noncomputable def length_diagonal_AC (AB BC CD DA : ℝ) (angle_ADC : ℝ) : ℝ :=
  (CD^2 + DA^2 - 2 * CD * DA * Real.cos angle_ADC).sqrt

theorem diagonal_AC_length :
  ∀ (AB BC CD DA : ℝ) (angle_ADC : ℝ),
  AB = 10 → BC = 10 → CD = 17 → DA = 17 → angle_ADC = 2 * Real.pi / 3 →
  length_diagonal_AC AB BC CD DA angle_ADC = Real.sqrt 867 :=
begin
  intros AB BC CD DA angle_ADC hAB hBC hCD hDA hangle_ADC,
  rw [hCD, hDA, hangle_ADC],
  sorry
end

end diagonal_AC_length_l119_119209


namespace repeating_decimal_sum_l119_119562

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l119_119562


namespace jane_change_l119_119216

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l119_119216


namespace highland_high_students_highland_high_num_both_clubs_l119_119924

theorem highland_high_students (total_students drama_club science_club either_both both_clubs : ℕ)
  (h1 : total_students = 320)
  (h2 : drama_club = 90)
  (h3 : science_club = 140)
  (h4 : either_both = 200) : 
  both_clubs = drama_club + science_club - either_both :=
by
  sorry

noncomputable def num_both_clubs : ℕ :=
if h : 320 = 320 ∧ 90 = 90 ∧ 140 = 140 ∧ 200 = 200
then 90 + 140 - 200
else 0

theorem highland_high_num_both_clubs : num_both_clubs = 30 :=
by
  sorry

end highland_high_students_highland_high_num_both_clubs_l119_119924


namespace mika_initial_stickers_l119_119052

theorem mika_initial_stickers :
  let store_stickers := 26.0
  let birthday_stickers := 20.0 
  let sister_stickers := 6.0 
  let mother_stickers := 58.0 
  let total_stickers := 130.0 
  ∃ x : Real, x + store_stickers + birthday_stickers + sister_stickers + mother_stickers = total_stickers ∧ x = 20.0 := 
by 
  sorry

end mika_initial_stickers_l119_119052


namespace radius_of_circle_l119_119657

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l119_119657


namespace problem_set_equiv_l119_119395

def positive_nats (x : ℕ) : Prop := x > 0

def problem_set : Set ℕ := {x | positive_nats x ∧ x - 3 < 2}

theorem problem_set_equiv : problem_set = {1, 2, 3, 4} :=
by 
  sorry

end problem_set_equiv_l119_119395


namespace minimum_m_n_1978_l119_119760

-- Define the conditions given in the problem
variables (m n : ℕ) (h1 : n > m) (h2 : m > 1)
-- Define the condition that the last three digits of 1978^m and 1978^n are identical
def same_last_three_digits (a b : ℕ) : Prop :=
  (a % 1000 = b % 1000)

-- Define the problem statement: under the conditions, prove that m + n = 106 when minimized
theorem minimum_m_n_1978 (h : same_last_three_digits (1978^m) (1978^n)) : m + n = 106 :=
sorry   -- Proof will be provided here

end minimum_m_n_1978_l119_119760


namespace find_a_l119_119190

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a^2 = 12) : a = 3 :=
by sorry

end find_a_l119_119190


namespace smallest_square_side_length_paintings_l119_119914

theorem smallest_square_side_length_paintings (n : ℕ) :
  ∃ n : ℕ, (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2020 → 1 * i ≤ n * n) → n = 1430 :=
by
  sorry

end smallest_square_side_length_paintings_l119_119914


namespace repeating_decimal_sum_l119_119561

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l119_119561


namespace necessary_but_not_sufficient_l119_119443

   theorem necessary_but_not_sufficient (a : ℝ) : a^2 > a → (a > 1) :=
   by {
     sorry
   }
   
end necessary_but_not_sufficient_l119_119443


namespace find_k_l119_119186

-- Define the sequence and its sum
def Sn (k : ℝ) (n : ℕ) : ℝ := k + 3^n
def an (k : ℝ) (n : ℕ) : ℝ := Sn k n - (if n = 0 then 0 else Sn k (n - 1))

-- Define the condition that a sequence is geometric
def is_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n

theorem find_k (k : ℝ) :
  is_geometric (an k) (an k 1 / an k 0) → k = -1 := 
by sorry

end find_k_l119_119186


namespace find_chord_line_eq_l119_119024

theorem find_chord_line_eq (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ)
    (hP : P = (1, 1)) (hC : C = (3, 0)) (hr : r = 3)
    (circle_eq : ∀ (x y : ℝ), (x - 3)^2 + y^2 = r^2) :
    ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -1 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := by
  sorry

end find_chord_line_eq_l119_119024


namespace expected_winnings_l119_119911

-- Define the probabilities
def prob_heads : ℚ := 1/2
def prob_tails : ℚ := 1/3
def prob_edge : ℚ := 1/6

-- Define the winnings
def win_heads : ℚ := 1
def win_tails : ℚ := 3
def lose_edge : ℚ := -5

-- Define the expected value function
def expected_value (p1 p2 p3 : ℚ) (w1 w2 w3 : ℚ) : ℚ :=
  p1 * w1 + p2 * w2 + p3 * w3

-- The expected winnings from flipping this coin
theorem expected_winnings : expected_value prob_heads prob_tails prob_edge win_heads win_tails lose_edge = 2/3 :=
by
  sorry

end expected_winnings_l119_119911


namespace symmetric_y_axis_l119_119945

-- Definition of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of point symmetry with respect to the y-axis
def symmetric_about_y_axis (M : Point3D) : Point3D := 
  { x := -M.x, y := M.y, z := -M.z }

-- Theorem statement: proving the symmetry
theorem symmetric_y_axis (M : Point3D) : 
  symmetric_about_y_axis M = { x := -M.x, y := M.y, z := -M.z } := by
  sorry  -- Proof is left out as per instruction.

end symmetric_y_axis_l119_119945


namespace multiplication_identity_multiplication_l119_119839

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l119_119839


namespace conditional_probability_l119_119028

theorem conditional_probability :
  let P_B : ℝ := 0.15
  let P_A : ℝ := 0.05
  let P_A_and_B : ℝ := 0.03
  let P_B_given_A := P_A_and_B / P_A
  P_B_given_A = 0.6 :=
by
  sorry

end conditional_probability_l119_119028


namespace geometric_sequence_common_ratio_l119_119866

theorem geometric_sequence_common_ratio (r : ℝ) (a : ℝ) (a3 : ℝ) :
  a = 3 → a3 = 27 → r = 3 ∨ r = -3 :=
by
  intros ha ha3
  sorry

end geometric_sequence_common_ratio_l119_119866


namespace interval_length_implies_difference_l119_119501

variable (c d : ℝ)

theorem interval_length_implies_difference (h1 : ∀ x : ℝ, c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) (h2 : (d - c) / 3 = 15) : d - c = 45 := 
sorry

end interval_length_implies_difference_l119_119501


namespace matrix_B_pow_66_l119_119804

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0], 
    ![-1, 0, 0], 
    ![0, 0, 1]]

theorem matrix_B_pow_66 : B^66 = ![![-1, 0, 0], ![0, -1, 0], ![0, 0, 1]] := by
  sorry

end matrix_B_pow_66_l119_119804


namespace probability_of_selecting_cooking_l119_119286

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119286


namespace missing_fraction_l119_119254

-- Defining all the given fractions
def f1 : ℚ := 1 / 3
def f2 : ℚ := 1 / 2
def f3 : ℚ := 1 / 5
def f4 : ℚ := 1 / 4
def f5 : ℚ := -9 / 20
def f6 : ℚ := -5 / 6

-- Defining the total sum in decimal form
def total_sum : ℚ := 5 / 6  -- Since 0.8333333333333334 is equivalent to 5/6

-- Defining the sum of the given fractions
def given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6

-- The Lean 4 statement to prove the missing fraction
theorem missing_fraction : ∃ x : ℚ, (given_sum + x = total_sum) ∧ x = 5 / 6 :=
by
  use 5 / 6
  constructor
  . sorry
  . rfl

end missing_fraction_l119_119254


namespace find_positive_integer_l119_119154

theorem find_positive_integer (n : ℕ) (h1 : n % 14 = 0) (h2 : 676 ≤ n ∧ n ≤ 702) : n = 700 :=
sorry

end find_positive_integer_l119_119154


namespace Paige_team_players_l119_119281

/-- Paige's team won their dodgeball game and scored 41 points total.
    If Paige scored 11 points and everyone else scored 6 points each,
    prove that the total number of players on the team was 6. -/
theorem Paige_team_players (total_points paige_points other_points : ℕ) (x : ℕ) (H1 : total_points = 41) (H2 : paige_points = 11) (H3 : other_points = 6) (H4 : paige_points + other_points * x = total_points) : x + 1 = 6 :=
by {
  sorry
}

end Paige_team_players_l119_119281


namespace dynamic_load_L_value_l119_119548

theorem dynamic_load_L_value (T H : ℝ) (hT : T = 3) (hH : H = 6) : 
  (L : ℝ) = (50 * T^3) / (H^3) -> L = 6.25 := 
by 
  sorry 

end dynamic_load_L_value_l119_119548


namespace probability_distribution_xi_l119_119107

theorem probability_distribution_xi (a : ℝ) (ξ : ℕ → ℝ) (h1 : ξ 1 = a / (1 * 2))
  (h2 : ξ 2 = a / (2 * 3)) (h3 : ξ 3 = a / (3 * 4)) (h4 : ξ 4 = a / (4 * 5))
  (h5 : (ξ 1) + (ξ 2) + (ξ 3) + (ξ 4) = 1) :
  ξ 1 + ξ 2 = 5 / 6 :=
by
  sorry

end probability_distribution_xi_l119_119107


namespace opposite_of_num_l119_119103

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l119_119103


namespace max_min_difference_abc_l119_119011

theorem max_min_difference_abc (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
    let M := 1
    let m := -1/2
    M - m = 3/2 :=
by
  sorry

end max_min_difference_abc_l119_119011


namespace average_of_c_and_d_l119_119076

variable (c d e : ℝ)

theorem average_of_c_and_d
  (h1: (4 + 6 + 9 + c + d + e) / 6 = 20)
  (h2: e = c + 6) :
  (c + d) / 2 = 47.5 := by
sorry

end average_of_c_and_d_l119_119076


namespace compare_neg_rats_l119_119928

theorem compare_neg_rats : (-3/8 : ℚ) > (-4/9 : ℚ) :=
by sorry

end compare_neg_rats_l119_119928


namespace cos_double_angle_unit_circle_l119_119006

theorem cos_double_angle_unit_circle (α y₀ : ℝ) (h : (1/2)^2 + y₀^2 = 1) : 
  Real.cos (2 * α) = -1/2 :=
by 
  -- The proof is omitted
  sorry

end cos_double_angle_unit_circle_l119_119006


namespace probability_select_cooking_l119_119320

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119320


namespace root_bounds_l119_119647

noncomputable def sqrt (r : ℝ) (n : ℕ) := r^(1 / n)

theorem root_bounds (a b c d : ℝ) (n p x y : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hn : 0 < n) (hp : 0 < p) (hx : 0 < x) (hy : 0 < y) :
  sqrt d y < sqrt (a * b * c * d) (n + p + x + y) ∧
  sqrt (a * b * c * d) (n + p + x + y) < sqrt a n := 
sorry

end root_bounds_l119_119647


namespace vertex_of_quadratic_function_l119_119161

-- Define the function and constants
variables (p q : ℝ)
  (hp : p > 0)
  (hq : q > 0)

-- State the theorem
theorem vertex_of_quadratic_function : 
  ∀ p q : ℝ, p > 0 → q > 0 → 
  (∀ x : ℝ, x = - (2 * p) / (2 : ℝ) → x = -p) := 
sorry

end vertex_of_quadratic_function_l119_119161


namespace min_value_at_2_l119_119954

noncomputable def f (x : ℝ) : ℝ := (2 / (x^2)) + Real.log x

theorem min_value_at_2 : (∀ x ∈ Set.Ioi (0 : ℝ), f x ≥ f 2) ∧ (∃ x ∈ Set.Ioi (0 : ℝ), f x = f 2) :=
by
  sorry

end min_value_at_2_l119_119954


namespace opposite_of_num_l119_119105

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l119_119105


namespace multiply_expand_l119_119835

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l119_119835


namespace kendra_words_learned_l119_119043

theorem kendra_words_learned (Goal : ℕ) (WordsNeeded : ℕ) (WordsAlreadyLearned : ℕ) 
  (h1 : Goal = 60) (h2 : WordsNeeded = 24) :
  WordsAlreadyLearned = Goal - WordsNeeded :=
sorry

end kendra_words_learned_l119_119043


namespace frame_percentage_l119_119385

theorem frame_percentage : 
  let side_length := 80
  let frame_width := 4
  let total_area := side_length * side_length
  let picture_side_length := side_length - 2 * frame_width
  let picture_area := picture_side_length * picture_side_length
  let frame_area := total_area - picture_area
  let frame_percentage := (frame_area * 100) / total_area
  frame_percentage = 19 := 
by
  sorry

end frame_percentage_l119_119385


namespace find_number_l119_119717

theorem find_number (x : ℝ) :
  0.15 * x = 0.25 * 16 + 2 → x = 40 :=
by
  -- skipping the proof steps
  sorry

end find_number_l119_119717


namespace waitress_tips_fraction_l119_119527

theorem waitress_tips_fraction
  (S : ℝ) -- salary
  (T : ℝ) -- tips
  (hT : T = (11 / 4) * S) -- tips are 11/4 of salary
  (I : ℝ) -- total income
  (hI : I = S + T) -- total income is the sum of salary and tips
  : (T / I) = (11 / 15) := -- fraction of income from tips is 11/15
by
  sorry

end waitress_tips_fraction_l119_119527


namespace probability_of_selecting_cooking_l119_119340

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119340


namespace rotten_tomatoes_l119_119136

-- Conditions
def weight_per_crate := 20
def num_crates := 3
def total_cost := 330
def selling_price_per_kg := 6
def profit := 12

-- Derived data
def total_weight := num_crates * weight_per_crate
def total_revenue := profit + total_cost
def sold_weight := total_revenue / selling_price_per_kg

-- Proof statement
theorem rotten_tomatoes : total_weight - sold_weight = 3 := by
  sorry

end rotten_tomatoes_l119_119136


namespace probability_select_cooking_l119_119317

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119317


namespace find_k_l119_119468

-- Define the vector structures for i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define the vectors a and b based on i, j, and k
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Statement of the theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by sorry

end find_k_l119_119468


namespace opposite_of_neg_half_l119_119080

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l119_119080


namespace percentage_reduction_price_increase_l119_119539

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l119_119539


namespace probability_of_selecting_cooking_l119_119350

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119350


namespace sum_of_cubes_l119_119707

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l119_119707


namespace calculate_total_cost_l119_119799

theorem calculate_total_cost : 
  let piano_cost := 500
  let lesson_cost_per_lesson := 40
  let number_of_lessons := 20
  let discount_rate := 0.25
  let missed_lessons := 3
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_lesson_cost := number_of_lessons * lesson_cost_per_lesson
  let discount := total_lesson_cost * discount_rate
  let discounted_lesson_cost := total_lesson_cost - discount
  let cost_of_missed_lessons := missed_lessons * lesson_cost_per_lesson
  let effective_lesson_cost := discounted_lesson_cost + cost_of_missed_lessons
  let total_cost := piano_cost + effective_lesson_cost + sheet_music_cost + maintenance_fees
  total_cost = 1395 :=
by
  sorry

end calculate_total_cost_l119_119799


namespace gcd_154_and_90_l119_119603

theorem gcd_154_and_90 : Nat.gcd 154 90 = 2 := by
  sorry

end gcd_154_and_90_l119_119603


namespace polar_equation_graph_l119_119869

theorem polar_equation_graph :
  ∀ (ρ θ : ℝ), (ρ > 0) → ((ρ - 1) * (θ - π) = 0) ↔ (ρ = 1 ∨ θ = π) :=
by
  sorry

end polar_equation_graph_l119_119869


namespace coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l119_119495

theorem coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45 :
  let general_term (r : ℕ) := (Nat.choose 10 r) * (x^(10 - 3 * r)/2)
  ∃ r : ℕ, (general_term r) = 2 ∧ (Nat.choose 10 r) = 45 :=
by
  sorry

end coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l119_119495


namespace tan_pi_over_12_minus_tan_pi_over_6_l119_119648

theorem tan_pi_over_12_minus_tan_pi_over_6 :
  (Real.tan (Real.pi / 12) - Real.tan (Real.pi / 6)) = 7 - 4 * Real.sqrt 3 :=
  sorry

end tan_pi_over_12_minus_tan_pi_over_6_l119_119648


namespace smallest_number_gt_sum_digits_1755_l119_119605

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end smallest_number_gt_sum_digits_1755_l119_119605


namespace q_is_20_percent_less_than_p_l119_119890

theorem q_is_20_percent_less_than_p (p q : ℝ) (h : p = 1.25 * q) : (q - p) / p * 100 = -20 := by
  sorry

end q_is_20_percent_less_than_p_l119_119890


namespace circle_equation_m_l119_119782
open Real

theorem circle_equation_m (m : ℝ) : (x^2 + y^2 + 4 * x + 2 * y + m = 0 → m < 5) := sorry

end circle_equation_m_l119_119782


namespace multiply_polynomials_l119_119827

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l119_119827


namespace literate_employees_l119_119790

theorem literate_employees (num_illiterate : ℕ) (wage_decrease_per_illiterate : ℕ)
  (total_average_salary_decrease : ℕ) : num_illiterate = 35 → 
                                        wage_decrease_per_illiterate = 25 →
                                        total_average_salary_decrease = 15 →
                                        ∃ L : ℕ, L = 23 :=
by {
  -- given: num_illiterate = 35
  -- given: wage_decrease_per_illiterate = 25
  -- given: total_average_salary_decrease = 15
  sorry
}

end literate_employees_l119_119790


namespace probability_3_heads_5_tosses_l119_119880

noncomputable def probability_of_3_heads_in_5_tosses : ℚ :=
  (nat.choose 5 3) * ((1/2) ^ 3) * ((1/2) ^ 2)

theorem probability_3_heads_5_tosses : probability_of_3_heads_in_5_tosses = 5 / 16 := by
  sorry

end probability_3_heads_5_tosses_l119_119880


namespace probability_of_selecting_cooking_l119_119299

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119299


namespace angle_between_clock_hands_at_7_oclock_l119_119697

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l119_119697


namespace sum_squares_l119_119807

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end sum_squares_l119_119807


namespace maximum_bunnies_drum_l119_119899

-- Define the conditions as provided in the problem
def drumsticks := ℕ -- Natural number type for simplicity
def drum := ℕ -- Natural number type for simplicity

structure Bunny :=
(drum_size : drum)
(stick_length : drumsticks)

def max_drumming_bunnies (bunnies : List Bunny) : ℕ := 
  -- Actual implementation to find the maximum number of drumming bunnies
  sorry

theorem maximum_bunnies_drum (bunnies : List Bunny) (h_size : bunnies.length = 7) : max_drumming_bunnies bunnies = 6 :=
by
  -- Proof of the theorem
  sorry

end maximum_bunnies_drum_l119_119899


namespace average_visitors_on_Sundays_l119_119384

theorem average_visitors_on_Sundays (S : ℕ) 
  (h1 : 30 % 7 = 2)  -- The month begins with a Sunday
  (h2 : 25 = 30 - 5)  -- The month has 25 non-Sundays
  (h3 : (120 * 25) = 3000) -- Total visitors on non-Sundays
  (h4 : (125 * 30) = 3750) -- Total visitors for the month
  (h5 : 5 * 30 > 0) -- There are a positive number of Sundays
  : S = 150 :=
by
  sorry

end average_visitors_on_Sundays_l119_119384


namespace repeating_decimal_sum_in_lowest_terms_l119_119571

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l119_119571


namespace probability_selecting_cooking_l119_119304

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119304


namespace square_perimeter_l119_119892

theorem square_perimeter (area : ℝ) (h : area = 144) : ∃ perimeter : ℝ, perimeter = 48 :=
by
  sorry

end square_perimeter_l119_119892


namespace positive_solution_system_l119_119157

theorem positive_solution_system (x1 x2 x3 x4 x5 : ℝ) (h1 : (x3 + x4 + x5)^5 = 3 * x1)
  (h2 : (x4 + x5 + x1)^5 = 3 * x2) (h3 : (x5 + x1 + x2)^5 = 3 * x3)
  (h4 : (x1 + x2 + x3)^5 = 3 * x4) (h5 : (x2 + x3 + x4)^5 = 3 * x5) :
  x1 > 0 → x2 > 0 → x3 > 0 → x4 > 0 → x5 > 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 ∧ (x1 = 1/3) :=
by 
  intros hpos1 hpos2 hpos3 hpos4 hpos5
  sorry

end positive_solution_system_l119_119157


namespace opposite_neg_half_l119_119086

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l119_119086


namespace multiplication_identity_multiplication_l119_119840

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l119_119840


namespace students_neither_l119_119643

def total_students : ℕ := 150
def students_math : ℕ := 85
def students_physics : ℕ := 63
def students_chemistry : ℕ := 40
def students_math_physics : ℕ := 20
def students_physics_chemistry : ℕ := 15
def students_math_chemistry : ℕ := 10
def students_all_three : ℕ := 5

theorem students_neither:
  total_students - 
  (students_math + students_physics + students_chemistry 
  - students_math_physics - students_physics_chemistry 
  - students_math_chemistry + students_all_three) = 2 := 
by sorry

end students_neither_l119_119643


namespace simplest_form_of_expression_l119_119651

theorem simplest_form_of_expression (c : ℝ) : ((3 * c + 5 - 3 * c) / 2) = 5 / 2 :=
by 
  sorry

end simplest_form_of_expression_l119_119651


namespace average_speed_of_car_l119_119535

noncomputable def averageSpeed : ℚ := 
  let speed1 := 45     -- kph
  let distance1 := 15  -- km
  let speed2 := 55     -- kph
  let distance2 := 30  -- km
  let speed3 := 65     -- kph
  let time3 := 35 / 60 -- hours
  let speed4 := 52     -- kph
  let time4 := 20 / 60 -- hours
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  let totalDistance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let totalTime := time1 + time2 + time3 + time4
  totalDistance / totalTime

theorem average_speed_of_car :
  abs (averageSpeed - 55.85) < 0.01 := 
  sorry

end average_speed_of_car_l119_119535


namespace sum_of_roots_of_quadratic_eq_l119_119704

theorem sum_of_roots_of_quadratic_eq : 
  ∀ (a b c : ℝ), (x^2 - 6 * x + 8 = 0) → (a = 1 ∧ b = -6 ∧ c = 8) → -b / a = 6 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_eq_l119_119704


namespace vector_addition_correct_l119_119153

def vec1 : ℤ × ℤ := (5, -9)
def vec2 : ℤ × ℤ := (-8, 14)
def vec_sum (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

theorem vector_addition_correct :
  vec_sum vec1 vec2 = (-3, 5) :=
by
  -- Proof omitted
  sorry

end vector_addition_correct_l119_119153


namespace contrapositive_proposition_l119_119864

theorem contrapositive_proposition {a b : ℝ} :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) → (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_proposition_l119_119864


namespace tan_theta_half_l119_119946

theorem tan_theta_half (θ : ℝ) (a b : ℝ × ℝ) 
  (h₀ : a = (Real.sin θ, 1)) 
  (h₁ : b = (-2, Real.cos θ)) 
  (h₂ : a.1 * b.1 + a.2 * b.2 = 0) : Real.tan θ = 1 / 2 :=
sorry

end tan_theta_half_l119_119946


namespace sum_of_numbers_l119_119818

theorem sum_of_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000)
(h_eq : 100 * x + y = 7 * x * y) : x + y = 18 :=
sorry

end sum_of_numbers_l119_119818


namespace opposite_of_num_l119_119104

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l119_119104


namespace find_x_l119_119542

theorem find_x (t : ℤ) : 
∃ x : ℤ, (x % 7 = 3) ∧ (x^2 % 49 = 44) ∧ (x^3 % 343 = 111) ∧ (x = 343 * t + 17) :=
sorry

end find_x_l119_119542


namespace probability_of_selecting_cooking_l119_119342

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119342


namespace probability_of_selecting_cooking_l119_119334

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119334


namespace problem_statement_l119_119421

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y - x > 1) :
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 :=
sorry

end problem_statement_l119_119421


namespace inverse_proportion_incorrect_D_l119_119750

theorem inverse_proportion_incorrect_D :
  ∀ (x y x1 y1 x2 y2 : ℝ), (y = -3 / x) ∧ (y1 = -3 / x1) ∧ (y2 = -3 / x2) ∧ (x1 < x2) → ¬(y1 < y2) :=
by
  sorry

end inverse_proportion_incorrect_D_l119_119750


namespace xiaoming_selects_cooking_probability_l119_119329

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119329


namespace sampling_is_systematic_l119_119386

-- Conditions
def production_line (units_per_day : ℕ) : Prop := units_per_day = 128

def sampling_inspection (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  samples_per_day = 8 ∧ inspection_time = 30 ∧ inspection_days = 7

-- Question
def sampling_method (method : String) (units_per_day : ℕ) (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  production_line units_per_day ∧ sampling_inspection samples_per_day inspection_time inspection_days → method = "systematic sampling"

-- Theorem stating the question == answer given conditions
theorem sampling_is_systematic : sampling_method "systematic sampling" 128 8 30 7 :=
by
  sorry

end sampling_is_systematic_l119_119386


namespace student_B_speed_l119_119908

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l119_119908


namespace boy_real_name_is_kolya_l119_119896

variable (days_answers : Fin 6 → String)
variable (lies_on : Fin 6 → Bool)
variable (truth_days : List (Fin 6))

-- Define the conditions
def condition_truth_days : List (Fin 6) := [0, 1] -- Suppose Thursday is 0, Friday is 1.
def condition_lies_on (d : Fin 6) : Bool := d = 2 -- Suppose Tuesday is 2.

-- The sequence of answers
def condition_days_answers : Fin 6 → String := 
  fun d => match d with
    | 0 => "Kolya"
    | 1 => "Petya"
    | 2 => "Kolya"
    | 3 => "Petya"
    | 4 => "Vasya"
    | 5 => "Petya"
    | _ => "Unknown"

-- The proof problem statement
theorem boy_real_name_is_kolya : 
  ∀ (d : Fin 6), 
  (d ∈ condition_truth_days → condition_days_answers d = "Kolya") ∧
  (condition_lies_on d → condition_days_answers d ≠ "Vasya") ∧ 
  (¬(d ∈ condition_truth_days ∨ condition_lies_on d) → True) →
  "Kolya" = "Kolya" :=
by
  sorry

end boy_real_name_is_kolya_l119_119896


namespace find_number_l119_119895

variable (x : ℝ)

theorem find_number (hx : 5100 - (102 / x) = 5095) : x = 20.4 := 
by
  sorry

end find_number_l119_119895


namespace find_angle_between_vectors_l119_119183

variables {a b : EuclideanSpace ℝ (Fin 2)}

def vector_length (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (∑ i, (v i) ^ 2).sqrt

theorem find_angle_between_vectors
  (h1 : vector_length a = sqrt 3)
  (h2 : vector_length b = 4)
  (h3 : inner a (2 • a - b) = 0) :
  real.arccos (inner a b / (vector_length a * vector_length b)) = π / 6 :=
sorry

end find_angle_between_vectors_l119_119183


namespace max_unique_rankings_l119_119235

theorem max_unique_rankings (n : ℕ) : 
  ∃ (contestants : ℕ), 
    (∀ (scores : ℕ → ℕ), 
      (∀ i, 0 ≤ scores i ∧ scores i ≤ contestants) ∧
      (∀ i j, i ≠ j → scores i ≠ scores j)) 
    → contestants = 2^n := 
sorry

end max_unique_rankings_l119_119235


namespace repeating_decimal_sum_l119_119567

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l119_119567


namespace medal_ratio_l119_119464

theorem medal_ratio (total_medals : ℕ) (track_medals : ℕ) (badminton_medals : ℕ) (swimming_medals : ℕ) 
  (h1 : total_medals = 20) 
  (h2 : track_medals = 5) 
  (h3 : badminton_medals = 5) 
  (h4 : swimming_medals = total_medals - track_medals - badminton_medals) : 
  swimming_medals / track_medals = 2 := 
by 
  sorry

end medal_ratio_l119_119464


namespace repeating_decimals_sum_l119_119581

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l119_119581


namespace probability_of_selecting_cooking_l119_119351

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119351


namespace students_from_other_communities_eq_90_l119_119788

theorem students_from_other_communities_eq_90 {total_students : ℕ} 
  (muslims_percentage : ℕ)
  (hindus_percentage : ℕ)
  (sikhs_percentage : ℕ)
  (christians_percentage : ℕ)
  (buddhists_percentage : ℕ)
  : total_students = 1000 →
    muslims_percentage = 36 →
    hindus_percentage = 24 →
    sikhs_percentage = 15 →
    christians_percentage = 10 →
    buddhists_percentage = 6 →
    (total_students * (100 - (muslims_percentage + hindus_percentage + sikhs_percentage + christians_percentage + buddhists_percentage))) / 100 = 90 :=
by
  intros h_total h_muslims h_hindus h_sikhs h_christians h_buddhists
  -- Proof can be omitted as indicated
  sorry

end students_from_other_communities_eq_90_l119_119788


namespace some_number_proof_l119_119162

def g (n : ℕ) : ℕ :=
  if n < 3 then 1 else 
  if n % 2 = 0 then g (n - 1) else 
    g (n - 2) * n

theorem some_number_proof : g 106 - g 103 = 105 :=
by sorry

end some_number_proof_l119_119162


namespace repeating_decimal_sum_in_lowest_terms_l119_119576

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l119_119576


namespace probability_selecting_cooking_l119_119301

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119301


namespace proof_problem_l119_119623

theorem proof_problem (x : ℝ) (h1 : x = 3) (h2 : 2 * x ≠ 5) (h3 : x + 5 ≠ 3) 
                      (h4 : 7 - x ≠ 2) (h5 : 6 + 2 * x ≠ 14) :
    3 * x - 1 = 8 :=
by 
  sorry

end proof_problem_l119_119623


namespace probability_cooking_is_one_fourth_l119_119362
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119362


namespace apple_count_difference_l119_119687

theorem apple_count_difference
    (original_green : ℕ)
    (additional_green : ℕ)
    (red_more_than_green : ℕ)
    (green_now : ℕ := original_green + additional_green)
    (red_now : ℕ := original_green + red_more_than_green)
    (difference : ℕ := green_now - red_now)
    (h_original_green : original_green = 32)
    (h_additional_green : additional_green = 340)
    (h_red_more_than_green : red_more_than_green = 200) :
    difference = 140 :=
by
  sorry

end apple_count_difference_l119_119687


namespace clock_angle_at_seven_l119_119693

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l119_119693


namespace value_of_some_number_l119_119971

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l119_119971


namespace probability_selecting_cooking_l119_119302

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119302


namespace men_entered_l119_119990

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l119_119990


namespace remainder_of_b2_minus_3a_div_6_l119_119947

theorem remainder_of_b2_minus_3a_div_6 (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 5) : 
  (b^2 - 3 * a) % 6 = 1 := 
sorry

end remainder_of_b2_minus_3a_div_6_l119_119947


namespace angles_arithmetic_sequence_sides_l119_119007

theorem angles_arithmetic_sequence_sides (A B C a b c : ℝ)
  (h_angle_ABC : A + B + C = 180)
  (h_arithmetic_sequence : 2 * B = A + C)
  (h_cos_B : A * A + c * c - b * b = 2 * a * c)
  (angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < 180 ∧ B < 180 ∧ C < 180) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end angles_arithmetic_sequence_sides_l119_119007


namespace radius_of_circle_with_area_64pi_l119_119659

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l119_119659


namespace find_possible_y_values_l119_119047

noncomputable def validYValues (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) : Set ℝ :=
  { y | y = (x - 3)^2 * (x + 4) / (2 * x - 4) }

theorem find_possible_y_values (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) :
  validYValues x hx = {39, 6} :=
sorry

end find_possible_y_values_l119_119047


namespace age_difference_l119_119279

variable (x y z : ℝ)

def overall_age_condition (x y z : ℝ) : Prop := (x + y = y + z + 10)

theorem age_difference (x y z : ℝ) (h : overall_age_condition x y z) : (x - z) / 10 = 1 :=
  by
    sorry

end age_difference_l119_119279


namespace alicia_remaining_sets_l119_119916

def initial_sets : Nat := 600
def guggenheim_donation : Nat := 51
def met_fraction : Rat := 1 / 3
def louvre_fraction : Rat := 1 / 4
def damaged_sets : Nat := 30
def british_fraction : Rat := 40 / 100
def gallery_fraction : Rat := 1 / 8

theorem alicia_remaining_sets : 
  let after_guggenheim := initial_sets - guggenheim_donation in
  let after_met := after_guggenheim - Nat.floor (met_fraction * after_guggenheim) in
  let after_louvre := after_met - Nat.floor (louvre_fraction * after_met) in
  let after_damage := after_louvre - damaged_sets in
  let after_british := after_damage - Nat.floor (british_fraction * after_damage) in
  let after_gallery := after_british - Nat.floor (gallery_fraction * after_british) in
  after_gallery = 129 :=
by
  sorry

end alicia_remaining_sets_l119_119916


namespace fill_trough_time_l119_119919

theorem fill_trough_time 
  (old_pump_rate : ℝ := 1 / 600) 
  (new_pump_rate : ℝ := 1 / 200) : 
  1 / (old_pump_rate + new_pump_rate) = 150 := 
by 
  sorry

end fill_trough_time_l119_119919


namespace brick_height_l119_119418

variable {l w : ℕ} (SA : ℕ)

theorem brick_height (h : ℕ) (l_eq : l = 10) (w_eq : w = 4) (SA_eq : SA = 136) 
    (surface_area_eq : SA = 2 * (l * w + l * h + w * h)) : h = 2 :=
by
  sorry

end brick_height_l119_119418


namespace impossible_to_have_only_stacks_of_three_l119_119264

theorem impossible_to_have_only_stacks_of_three (n J : ℕ) (h_initial_n : n = 1) (h_initial_J : J = 1001) :
  (∀ n J, (n + J = 1002) → (∀ k : ℕ, 3 * k ≤ J → k + 3 * k ≠ 1002)) 
  :=
sorry

end impossible_to_have_only_stacks_of_three_l119_119264


namespace multiply_polynomials_l119_119828

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l119_119828


namespace toothpicks_in_12th_stage_l119_119113

def toothpicks_in_stage (n : ℕ) : ℕ :=
  3 * n

theorem toothpicks_in_12th_stage : toothpicks_in_stage 12 = 36 :=
by
  -- Proof steps would go here, including simplification and calculations, but are omitted with 'sorry'.
  sorry

end toothpicks_in_12th_stage_l119_119113


namespace probability_of_union_l119_119414

-- Define the range of two-digit numbers
def digit_count : ℕ := 90

-- Define events A and B
def event_a (n : ℕ) : Prop := n % 2 = 0
def event_b (n : ℕ) : Prop := n % 5 = 0

-- Define the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℚ := 45 / digit_count
def P_B : ℚ := 18 / digit_count
def P_A_and_B : ℚ := 9 / digit_count

-- Prove the final probability using inclusion-exclusion principle
theorem probability_of_union : P_A + P_B - P_A_and_B = 0.6 := by
  sorry

end probability_of_union_l119_119414


namespace gcd_221_195_l119_119074

-- Define the two numbers
def a := 221
def b := 195

-- Statement of the problem: the gcd of a and b is 13
theorem gcd_221_195 : Nat.gcd a b = 13 := 
by
  sorry

end gcd_221_195_l119_119074


namespace men_entered_l119_119995

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l119_119995


namespace Talia_father_age_l119_119460

def Talia_age (T : ℕ) : Prop := T + 7 = 20
def Talia_mom_age (M T : ℕ) : Prop := M = 3 * T
def Talia_father_age_in_3_years (F M : ℕ) : Prop := F + 3 = M

theorem Talia_father_age (T F M : ℕ) 
    (hT : Talia_age T)
    (hM : Talia_mom_age M T)
    (hF : Talia_father_age_in_3_years F M) :
    F = 36 :=
by 
  sorry

end Talia_father_age_l119_119460


namespace short_answer_question_time_l119_119042

-- Definitions from the conditions
def minutes_per_paragraph := 15
def minutes_per_essay := 60
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15
def total_minutes := 4 * 60

-- Auxiliary calculations
def total_minutes_essays := num_essays * minutes_per_essay
def total_minutes_paragraphs := num_paragraphs * minutes_per_paragraph
def total_minutes_used := total_minutes_essays + total_minutes_paragraphs

-- The time per short-answer question is 3 minutes
theorem short_answer_question_time (x : ℕ) : (total_minutes - total_minutes_used) / num_short_answer_questions = 3 :=
by
  -- x is defined as the time per short-answer question
  let x := (total_minutes - total_minutes_used) / num_short_answer_questions
  have time_for_short_answer_questions : total_minutes - total_minutes_used = 45 := by sorry
  have time_per_short_answer_question : 45 / num_short_answer_questions = 3 := by sorry
  have x_equals_3 : x = 3 := by sorry
  exact x_equals_3

end short_answer_question_time_l119_119042


namespace probability_of_selecting_cooking_l119_119346

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119346


namespace number_of_players_in_hockey_club_l119_119037

-- Defining the problem parameters
def cost_of_gloves : ℕ := 6
def cost_of_helmet := cost_of_gloves + 7
def total_cost_per_set := cost_of_gloves + cost_of_helmet
def total_cost_per_player := 2 * total_cost_per_set
def total_expenditure : ℕ := 3120

-- Defining the target number of players
def num_players : ℕ := total_expenditure / total_cost_per_player

theorem number_of_players_in_hockey_club : num_players = 82 := by
  sorry

end number_of_players_in_hockey_club_l119_119037


namespace smallest_n_is_1770_l119_119606

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end smallest_n_is_1770_l119_119606


namespace top_field_number_and_total_labelings_l119_119519

open Finset

-- Define the problem
def valid_labeling (fields : ℕ → ℕ) : Prop :=
  injective fields ∧
  (∀ i, fields i ∈ range 1 (9 + 1)) ∧
  (∃ x, ∀ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d → (fields a + fields b + fields c + fields d = x) ∧
       ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c → (fields a + fields b + fields c = x))

-- Prove the number in the top field is always 9 and there are exactly 48 such labellings
theorem top_field_number_and_total_labelings :
  (∀ fields, valid_labeling fields → fields 0 = 9) ∧
  (∃ count : ℕ, count = 48) :=
by
  sorry

end top_field_number_and_total_labelings_l119_119519


namespace clipping_per_friend_l119_119045

def GluePerClipping : Nat := 6
def TotalGlue : Nat := 126
def TotalFriends : Nat := 7

theorem clipping_per_friend :
  (TotalGlue / GluePerClipping) / TotalFriends = 3 := by
  sorry

end clipping_per_friend_l119_119045


namespace distribution_and_variance_of_X_optimal_strategy_changed_selection_l119_119785

-- Part 1: Distribution and variance of X
theorem distribution_and_variance_of_X :
  ∀ (X : ℕ),
  (X ∈ {0, 1, 2, 3} →
   let P_X := {0 ↦ (8 / 27 : ℚ), 1 ↦ (4 / 9 : ℚ), 2 ↦ (2 / 9 : ℚ), 3 ↦ (1 / 27 : ℚ)} in
     P_X X = match X with 
     | 0 => 8 / 27 
     | 1 => 4 / 9
     | 2 => 2 / 9
     | 3 => 1 / 27 
     | _ => 0) ∧
  (P_X.hasVariance (2 / 3)) := sorry

-- Part 2: Optimal strategy with changed rules
theorem optimal_strategy_changed_selection :
  let probability_win_after_change := 2 / 3 in
  let probability_lose_after_change := 1 / 3 in
  let expected_value_after_change := 400 * probability_win_after_change + 0 * probability_lose_after_change in
  let probability_win_initial := 1 / 3 in
  let probability_consolation_initial := 2 / 3 in
  let expected_value_initial := 200 * probability_win_initial + 50 * probability_consolation_initial in
  expected_value_after_change > expected_value_initial :=
sorry

end distribution_and_variance_of_X_optimal_strategy_changed_selection_l119_119785


namespace gcd_gx_x_l119_119950

noncomputable def g (x : ℤ) : ℤ :=
  (3 * x + 5) * (9 * x + 4) * (11 * x + 8) * (x + 11)

theorem gcd_gx_x (x : ℤ) (h : 34914 ∣ x) : Int.gcd (g x) x = 1760 :=
by
  sorry

end gcd_gx_x_l119_119950


namespace repeating_decimal_sum_l119_119560

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l119_119560


namespace B_and_C_together_l119_119732

-- Defining the variables and conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 500)
variable (h2 : A + C = 200)
variable (h3 : C = 50)

-- The theorem to prove that B + C = 350
theorem B_and_C_together : B + C = 350 :=
by
  -- Replacing with the actual proof steps
  sorry

end B_and_C_together_l119_119732


namespace inequality_cannot_hold_l119_119177

noncomputable def f (a b c x : ℝ) := a * x ^ 2 + b * x + c

theorem inequality_cannot_hold
  (a b c : ℝ)
  (h_symm : ∀ x, f a b c x = f a b c (2 - x)) :
  ¬ (f a b c (1 - a) < f a b c (1 - 2 * a) ∧ f a b c (1 - 2 * a) < f a b c 1) :=
by {
  sorry
}

end inequality_cannot_hold_l119_119177


namespace probability_select_cooking_l119_119321

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119321


namespace area_of_rectangular_field_l119_119503

theorem area_of_rectangular_field 
  (P L W : ℕ) 
  (hP : P = 120) 
  (hL : L = 3 * W) 
  (hPerimeter : 2 * L + 2 * W = P) : 
  (L * W = 675) :=
by 
  sorry

end area_of_rectangular_field_l119_119503


namespace number_of_teachers_at_Queen_Middle_School_l119_119850

-- Conditions
def num_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 25

-- Proof that the number of teachers is 72
theorem number_of_teachers_at_Queen_Middle_School :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by sorry

end number_of_teachers_at_Queen_Middle_School_l119_119850


namespace artwork_collection_l119_119681

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l119_119681


namespace mode_of_dataSet_is_3_l119_119248

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end mode_of_dataSet_is_3_l119_119248


namespace opposite_of_half_l119_119090

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l119_119090


namespace average_speed_of_train_l119_119712

theorem average_speed_of_train (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 125) (h2 : d2 = 270) (h3 : t1 = 2.5) (h4 : t2 = 3) :
  (d1 + d2) / (t1 + t2) = 71.82 :=
by
  sorry

end average_speed_of_train_l119_119712


namespace jane_change_l119_119218

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l119_119218


namespace repeating_decimal_sum_l119_119569

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l119_119569


namespace intersecting_lines_product_l119_119633

theorem intersecting_lines_product 
  (a b : ℝ)
  (T : Set (ℝ × ℝ)) (S : Set (ℝ × ℝ))
  (hT : T = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ a * x + y - 3 = 0})
  (hS : S = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x - y - b = 0})
  (h_intersect : (2, 1) ∈ T) (h_intersect_S : (2, 1) ∈ S) :
  a * b = 1 := 
by
  sorry

end intersecting_lines_product_l119_119633


namespace petya_oranges_l119_119059

theorem petya_oranges (m o : ℕ) (h1 : m + 6 * m + o = 20) (h2 : 6 * m > o) : o = 6 :=
by 
  sorry

end petya_oranges_l119_119059


namespace exists_c_same_digit_occurrences_l119_119761

theorem exists_c_same_digit_occurrences (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ c : ℕ, c > 0 ∧ ∀ d : ℕ, d ≠ 0 → 
    (Nat.digits 10 (c * m)).count d = (Nat.digits 10 (c * n)).count d := sorry

end exists_c_same_digit_occurrences_l119_119761


namespace participants_begin_competition_l119_119204

theorem participants_begin_competition (x : ℝ) 
  (h1 : 0.4 * x * (1 / 4) = 16) : 
  x = 160 := 
by
  sorry

end participants_begin_competition_l119_119204


namespace smaller_angle_at_seven_oclock_l119_119694

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l119_119694


namespace repeating_decimal_sum_in_lowest_terms_l119_119572

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l119_119572


namespace choosing_ways_president_vp_committee_l119_119458

theorem choosing_ways_president_vp_committee :
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  total_choices = 2520 := by
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  have : total_choices = 2520 := by
    sorry
  exact this

end choosing_ways_president_vp_committee_l119_119458


namespace intersection_complement_A_B_l119_119771

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def complement (S : Set ℝ) : Set ℝ := {x | x ∉ S}

theorem intersection_complement_A_B :
  U = Set.univ →
  A = {x | -1 < x ∧ x < 1} →
  B = {y | 0 < y} →
  (A ∩ complement B) = {x | -1 < x ∧ x ≤ 0} :=
by
  intros hU hA hB
  sorry

end intersection_complement_A_B_l119_119771


namespace green_apples_more_than_red_apples_l119_119685

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end green_apples_more_than_red_apples_l119_119685


namespace radius_of_circle_with_area_64pi_l119_119660

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l119_119660


namespace max_value_expr_l119_119473

open Real

noncomputable def expr (x : ℝ) : ℝ :=
  (x^4 + 3 * x^2 - sqrt (x^8 + 9)) / x^2

theorem max_value_expr : ∀ (x y : ℝ), (0 < x) → (y = x + 1 / x) → expr x = 15 / 7 :=
by
  intros x y hx hy
  sorry

end max_value_expr_l119_119473


namespace nina_homework_total_l119_119232

def ruby_math_homework : ℕ := 6

def ruby_reading_homework : ℕ := 2

def nina_math_homework : ℕ := ruby_math_homework * 4 + ruby_math_homework

def nina_reading_homework : ℕ := ruby_reading_homework * 8 + ruby_reading_homework

def nina_total_homework : ℕ := nina_math_homework + nina_reading_homework

theorem nina_homework_total :
  nina_total_homework = 48 :=
by
  unfold nina_total_homework
  unfold nina_math_homework
  unfold nina_reading_homework
  unfold ruby_math_homework
  unfold ruby_reading_homework
  sorry

end nina_homework_total_l119_119232


namespace arithmetic_sequence_a8_l119_119179

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_a8 (h : a 7 + a 9 = 8) : a 8 = 4 := 
by 
  -- proof steps would go here
  sorry

end arithmetic_sequence_a8_l119_119179


namespace opposite_of_neg_half_is_half_l119_119095

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l119_119095


namespace intersection_sets_l119_119619

def setA : set ℝ := {y | ∃ x : ℝ, y = x^2 - 2 * x + 3}
def setB : set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2 * x + 7}

theorem intersection_sets : setA ∩ setB = {y | 2 ≤ y ∧ y ≤ 8} :=
by sorry

end intersection_sets_l119_119619


namespace find_integer_l119_119549

theorem find_integer (n : ℤ) (h : 5 * (n - 2) = 85) : n = 19 :=
sorry

end find_integer_l119_119549


namespace number_of_boys_is_90_l119_119677

-- Define the conditions
variables (B G : ℕ)
axiom sum_condition : B + G = 150
axiom percentage_condition : G = (B / 150) * 100

-- State the theorem
theorem number_of_boys_is_90 : B = 90 :=
by
  -- We can skip the proof for now using sorry
  sorry

end number_of_boys_is_90_l119_119677


namespace q_alone_time_24_days_l119_119817

theorem q_alone_time_24_days:
  ∃ (Wq : ℝ), (∀ (Wp Ws : ℝ), 
    Wp = Wq + 1 / 60 → 
    Wp + Wq = 1 / 10 → 
    Wp + 1 / 60 + 2 * Wq = 1 / 6 → 
    1 / Wq = 24) :=
by
  sorry

end q_alone_time_24_days_l119_119817


namespace probability_selecting_cooking_l119_119303

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119303


namespace one_greater_others_less_l119_119672

theorem one_greater_others_less {a b c : ℝ} (h1 : a > 0 ∧ b > 0 ∧ c > 0) (h2 : a * b * c = 1) (h3 : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
by
  sorry

end one_greater_others_less_l119_119672


namespace repeating_decimal_sum_l119_119600

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l119_119600


namespace sum_of_repeating_decimals_l119_119587

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l119_119587


namespace solve_sine_equation_l119_119127

theorem solve_sine_equation (x : ℝ) (k : ℤ) (h : |Real.sin x| ≠ 1) :
  (8.477 * ((∑' n, Real.sin x ^ n) / (∑' n, ((-1 : ℝ) * Real.sin x) ^ n)) = 4 / (1 + Real.tan x ^ 2)) 
  ↔ (x = (-1)^k * (Real.pi / 6) + k * Real.pi) :=
by
  sorry

end solve_sine_equation_l119_119127


namespace opposite_of_neg_half_l119_119077

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l119_119077


namespace evaluate_expression_l119_119150

theorem evaluate_expression : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) :=
by
  sorry

end evaluate_expression_l119_119150


namespace triangle_ratio_l119_119733

theorem triangle_ratio (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let p := 12;
  let q := 8;
  let segment_length := L / p;
  let segment_width := W / q;
  let area_X := (segment_length * segment_width) / 2;
  let area_rectangle := L * W;
  (area_X / area_rectangle) = (1 / 192) :=
by 
  sorry

end triangle_ratio_l119_119733


namespace jane_received_change_l119_119215

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l119_119215


namespace find_coefficient_b_l119_119234

noncomputable def polynomial_f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_coefficient_b 
  (a b c d : ℝ)
  (h1 : polynomial_f a b c d (-2) = 0)
  (h2 : polynomial_f a b c d 0 = 0)
  (h3 : polynomial_f a b c d 2 = 0)
  (h4 : polynomial_f a b c d (-1) = 3) :
  b = 0 :=
sorry

end find_coefficient_b_l119_119234


namespace inequality_solution_set_l119_119506

theorem inequality_solution_set (x : ℝ) : ((x - 1) * (x^2 - x + 1) > 0) ↔ (x > 1) :=
by
  sorry

end inequality_solution_set_l119_119506


namespace intersection_is_correct_l119_119048

open Set

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_is_correct : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_is_correct_l119_119048


namespace gcd_lcm_lemma_l119_119634

theorem gcd_lcm_lemma (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 33) (h_lcm : Nat.lcm a b = 90) : Nat.gcd a b = 3 :=
by
  sorry

end gcd_lcm_lemma_l119_119634


namespace correct_sequence_is_A_l119_119272

def Step := String
def Sequence := List Step

def correct_sequence : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]

def option_A : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]
def option_B : Sequence :=
  ["Wait for the train", "Buy a ticket", "Board the train", "Check the ticket"]
def option_C : Sequence :=
  ["Buy a ticket", "Wait for the train", "Board the train", "Check the ticket"]
def option_D : Sequence :=
  ["Repair the train", "Buy a ticket", "Check the ticket", "Board the train"]

theorem correct_sequence_is_A :
  correct_sequence = option_A :=
sorry

end correct_sequence_is_A_l119_119272


namespace handshakes_13_people_l119_119984

-- Define the condition
def people_in_room : ℕ := 13

-- Using the combination formula to define the handshakes function
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

-- The theorem to prove the total number of handshakes
theorem handshakes_13_people : handshakes 13 = 78 := by
  sorry

end handshakes_13_people_l119_119984


namespace incorrect_conclusion_symmetry_l119_119189

/-- Given the function f(x) = sin(1/5 * x + 13/6 * π), we define another function g(x) as the
translated function of f rightward by 10/3 * π units. We need to show that the graph of g(x)
is not symmetrical about the line x = π/4. -/
theorem incorrect_conclusion_symmetry (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = Real.sin (1/5 * x + 13/6 * Real.pi))
  (h₂ : ∀ x, g x = f (x - 10/3 * Real.pi)) :
  ¬ (∀ x, g (2 * (Real.pi / 4) - x) = g x) :=
sorry

end incorrect_conclusion_symmetry_l119_119189


namespace sports_club_membership_l119_119455

theorem sports_club_membership :
  ∀ (total T B_and_T neither : ℕ),
    total = 30 → 
    T = 19 →
    B_and_T = 9 →
    neither = 2 →
  ∃ (B : ℕ), B = 18 :=
by
  intros total T B_and_T neither ht hT hBandT hNeither
  let B := total - neither - T + B_and_T
  use B
  sorry

end sports_club_membership_l119_119455


namespace opposite_of_neg_half_l119_119081

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l119_119081


namespace avg_terminals_used_l119_119029

noncomputable def avgTerminalsUsed (n : ℕ) (p : ℝ) := 
  if h : 0 ≤ p ∧ p ≤ 1 then n * p else 0 

theorem avg_terminals_used (n : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : avgTerminalsUsed n p = n * p := 
  by 
  unfold avgTerminalsUsed 
  simp [h] 
  sorry

end avg_terminals_used_l119_119029


namespace remainder_of_n_div_1000_l119_119224

noncomputable def setS : Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

def n : ℕ :=
  let T := {x | 4 ≤ x ∧ x ≤ 15}
  (3^12 - 2^12) / 2

theorem remainder_of_n_div_1000 : (n % 1000) = 672 := 
  by sorry

end remainder_of_n_div_1000_l119_119224


namespace ln_abs_x_minus_a_even_iff_a_zero_l119_119463

theorem ln_abs_x_minus_a_even_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, Real.log (|x - a|) = Real.log (|(-x) - a|)) ↔ a = 0 :=
sorry

end ln_abs_x_minus_a_even_iff_a_zero_l119_119463


namespace roof_collapse_l119_119552

theorem roof_collapse (roof_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) :
  roof_limit = 500 → leaves_per_day = 100 → leaves_per_pound = 1000 → 
  let d := (roof_limit * leaves_per_pound) / leaves_per_day in d = 5000 :=
by
  intros h₁ h₂ h₃
  sorry

end roof_collapse_l119_119552


namespace repeating_decimal_sum_l119_119595

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l119_119595


namespace average_rainfall_correct_l119_119147

-- Define the monthly rainfall
def january_rainfall := 150
def february_rainfall := 200
def july_rainfall := 366
def other_months_rainfall := 100

-- Calculate total yearly rainfall
def total_yearly_rainfall := 
  january_rainfall + 
  february_rainfall + 
  july_rainfall + 
  (9 * other_months_rainfall)

-- Calculate total hours in a year
def days_per_month := 30
def total_days_in_year := 12 * days_per_month
def hours_per_day := 24
def total_hours_in_year := total_days_in_year * hours_per_day

-- Calculate average rainfall per hour
def average_rainfall_per_hour := 
  total_yearly_rainfall / total_hours_in_year

theorem average_rainfall_correct :
  average_rainfall_per_hour = (101 / 540) := sorry

end average_rainfall_correct_l119_119147


namespace student_B_speed_l119_119905

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l119_119905


namespace digit_150_in_fraction_l119_119265

-- Define the decimal expansion repeating sequence for the fraction 31/198
def repeat_seq : List Nat := [1, 5, 6, 5, 6, 5]

-- Define a function to get the nth digit of the repeating sequence
def nth_digit (n : Nat) : Nat :=
  repeat_seq.get! ((n - 1) % repeat_seq.length)

-- State the theorem to be proved
theorem digit_150_in_fraction : nth_digit 150 = 5 := 
sorry

end digit_150_in_fraction_l119_119265


namespace repeating_decimals_sum_l119_119593

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l119_119593


namespace second_newly_inserted_number_eq_l119_119497

theorem second_newly_inserted_number_eq : 
  ∃ q : ℝ, (q ^ 12 = 2) ∧ (1 * (q ^ 2) = 2 ^ (1 / 6)) := 
by
  sorry

end second_newly_inserted_number_eq_l119_119497


namespace probability_cooking_is_one_fourth_l119_119363
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119363


namespace age_of_B_l119_119255

theorem age_of_B (A B C : ℕ) (h1 : A + B + C = 90)
                  (h2 : (A - 10) = (B - 10) / 2)
                  (h3 : (B - 10) / 2 = (C - 10) / 3) : 
                  B = 30 :=
by sorry

end age_of_B_l119_119255


namespace sufficient_but_not_necessary_condition_l119_119863

theorem sufficient_but_not_necessary_condition (x y : ℝ) : 
  (x > 3 ∧ y > 3 → x + y > 6) ∧ ¬(x + y > 6 → x > 3 ∧ y > 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l119_119863


namespace arithmetic_series_first_term_l119_119420

theorem arithmetic_series_first_term 
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1800)
  (h2 : 50 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by
  sorry

end arithmetic_series_first_term_l119_119420


namespace product_of_three_numbers_l119_119509

theorem product_of_three_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 200)
  (h2 : n = 8 * x)
  (h3 : n = y - 5)
  (h4 : n = z + 5) :
  x * y * z = 372462 :=
by sorry

end product_of_three_numbers_l119_119509


namespace highest_power_of_2_divides_l119_119941

def a : ℕ := 17
def b : ℕ := 15
def n : ℕ := a^5 - b^5

def highestPowerOf2Divides (k : ℕ) : ℕ :=
  -- Function to find the highest power of 2 that divides k, implementation is omitted
  sorry

theorem highest_power_of_2_divides :
  highestPowerOf2Divides n = 2^5 := by
    sorry

end highest_power_of_2_divides_l119_119941


namespace men_in_second_group_l119_119894

theorem men_in_second_group (M : ℕ) : 
    (18 * 20 = M * 24) → M = 15 :=
by
  intro h
  sorry

end men_in_second_group_l119_119894


namespace probability_not_blue_marble_l119_119979

-- Define the conditions
def odds_for_blue_marble : ℕ := 5
def odds_for_not_blue_marble : ℕ := 6
def total_outcomes := odds_for_blue_marble + odds_for_not_blue_marble

-- Define the question and statement to be proven
theorem probability_not_blue_marble :
  (odds_for_not_blue_marble : ℚ) / total_outcomes = 6 / 11 :=
by
  -- skipping the proof step as per instruction
  sorry

end probability_not_blue_marble_l119_119979


namespace percentage_reduction_price_increase_l119_119537

open Real

-- Part 1: Finding the percentage reduction each time
theorem percentage_reduction (P₀ P₂ : ℝ) (x : ℝ) (h₀ : P₀ = 50) (h₁ : P₂ = 32) (h₂ : P₀ * (1 - x) ^ 2 = P₂) :
  x = 0.20 :=
by
  dsimp at h₀ h₁,
  rw h₀ at h₂,
  rw h₁ at h₂,
  simp at h₂,
  sorry

-- Part 2: Determining the price increase per kilogram
theorem price_increase (P y : ℝ) (profit_per_kg : ℝ) (initial_sales : ℝ) 
  (price_increase_limit : ℝ) (sales_decrease_rate : ℝ) (target_profit : ℝ)
  (h₀ : profit_per_kg = 10) (h₁ : initial_sales = 500) (h₂ : price_increase_limit = 8)
  (h₃ : sales_decrease_rate = 20) (h₄ : target_profit = 6000) (0 < y ∧ y ≤ price_increase_limit)
  (h₅ : (profit_per_kg + y) * (initial_sales - sales_decrease_rate * y) = target_profit) :
  y = 5 :=
by
  dsimp at h₀ h₁ h₂ h₃ h₄,
  rw [h₀, h₁, h₂, h₃, h₄] at h₅,
  sorry

end percentage_reduction_price_increase_l119_119537


namespace area_covered_by_congruent_rectangles_l119_119815

-- Definitions of conditions
def length_AB : ℕ := 12
def width_AD : ℕ := 8
def area_rect (l w : ℕ) : ℕ := l * w

-- Center of the first rectangle
def center_ABCD : ℕ × ℕ := (length_AB / 2, width_AD / 2)

-- Proof statement
theorem area_covered_by_congruent_rectangles 
  (length_ABCD length_EFGH width_ABCD width_EFGH : ℕ)
  (congruent : length_ABCD = length_EFGH ∧ width_ABCD = width_EFGH)
  (center_E : ℕ × ℕ)
  (H_center_E : center_E = center_ABCD) :
  area_rect length_ABCD width_ABCD + area_rect length_EFGH width_EFGH - length_ABCD * width_ABCD / 2 = 168 := by
  sorry

end area_covered_by_congruent_rectangles_l119_119815


namespace arithmetic_seq_sum_l119_119462

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n+1) - a n = a (n+2) - a (n+1))
  (h2 : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l119_119462


namespace regression_line_fits_l119_119513

variables {x y : ℝ}

def points := [(1, 2), (2, 5), (4, 7), (5, 10)]

def regression_line (x : ℝ) : ℝ := x + 3

theorem regression_line_fits :
  (∀ p ∈ points, regression_line p.1 = p.2) ∧ (regression_line 3 = 6) :=
by
  sorry

end regression_line_fits_l119_119513


namespace median_length_of_right_triangle_l119_119036

theorem median_length_of_right_triangle (DE EF : ℝ) (hDE : DE = 5) (hEF : EF = 12) :
  let DF := Real.sqrt (DE^2 + EF^2)
  let N := (EF / 2)
  let DN := DF / 2
  DN = 6.5 :=
by
  sorry

end median_length_of_right_triangle_l119_119036


namespace translated_function_symmetry_center_l119_119515

theorem translated_function_symmetry_center :
  let f := fun x : ℝ => Real.sin (6 * x + π / 4)
  let g := fun x : ℝ => f (x / 3)
  let h := fun x : ℝ => g (x - π / 8)
  h π / 2 = 0 :=
by
  sorry

end translated_function_symmetry_center_l119_119515


namespace lindas_savings_l119_119819

theorem lindas_savings :
  ∃ S : ℝ, (3 / 4 * S) + 150 = S ∧ (S - 150) = 3 / 4 * S := 
sorry

end lindas_savings_l119_119819


namespace multiply_expand_l119_119838

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l119_119838


namespace solution_f_derivative_l119_119433

noncomputable def f (x : ℝ) := Real.sqrt x

theorem solution_f_derivative :
  (deriv f 1) = 1 / 2 :=
by
  -- This is where the proof would go, but for now, we just state sorry.
  sorry

end solution_f_derivative_l119_119433


namespace artworks_collected_l119_119683

theorem artworks_collected (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) :
  students = 15 →
  artworks_per_student_per_quarter = 2 →
  quarters_per_year = 4 →
  num_years = 2 →
  (students * artworks_per_student_per_quarter * quarters_per_year * num_years) = 240 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end artworks_collected_l119_119683


namespace check_3x5_board_cannot_be_covered_l119_119533

/-- Define the concept of a checkerboard with a given number of rows and columns. -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Define the number of squares on a checkerboard. -/
def num_squares (cb : Checkerboard) : ℕ :=
  cb.rows * cb.cols

/-- Define whether a board can be completely covered by dominoes. -/
def can_be_covered_by_dominoes (cb : Checkerboard) : Prop :=
  (num_squares cb) % 2 = 0

/-- Instantiate the specific checkerboard scenarios. -/
def board_3x4 := Checkerboard.mk 3 4
def board_3x5 := Checkerboard.mk 3 5
def board_4x4 := Checkerboard.mk 4 4
def board_4x5 := Checkerboard.mk 4 5
def board_6x3 := Checkerboard.mk 6 3

/-- Statement to prove which board cannot be covered completely by dominoes. -/
theorem check_3x5_board_cannot_be_covered : ¬ can_be_covered_by_dominoes board_3x5 :=
by
  /- We leave out the proof steps here as requested. -/
  sorry

end check_3x5_board_cannot_be_covered_l119_119533


namespace jose_share_of_profit_l119_119280

def investment_months (amount : ℕ) (months : ℕ) : ℕ := amount * months

def profit_share (investment_months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment_months * total_profit) / total_investment_months

theorem jose_share_of_profit :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 36000
  let tom_months := 12
  let jose_months := 10
  let tom_investment_months := investment_months tom_investment tom_months
  let jose_investment_months := investment_months jose_investment jose_months
  let total_investment_months := tom_investment_months + jose_investment_months
  profit_share jose_investment_months total_investment_months total_profit = 20000 :=
by
  sorry

end jose_share_of_profit_l119_119280


namespace matrix_equation_l119_119812

-- Definitions from conditions
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![ -1, 4], ![ -6, 3]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1  -- Identity matrix

-- Given calculation of N^2
def N_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![ -23, 8], ![ -12, -15]]

-- Goal: prove that N^2 = r*N + s*I for r = 2 and s = -21
theorem matrix_equation (r s : ℤ) (h_r : r = 2) (h_s : s = -21) : N_squared = r • N + s • I := by
  sorry

end matrix_equation_l119_119812


namespace total_leftover_tarts_l119_119139

def cherry_tarts := 0.08
def blueberry_tarts := 0.75
def peach_tarts := 0.08

theorem total_leftover_tarts : cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end total_leftover_tarts_l119_119139


namespace complement_union_l119_119012

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_union : U \ (A ∪ B) = {4} := by
  sorry

end complement_union_l119_119012


namespace find_xiao_li_compensation_l119_119526

-- Define the conditions
variable (total_days : ℕ) (extra_days : ℕ) (extra_compensation : ℕ)
variable (daily_work : ℕ) (daily_reward : ℕ) (xiao_li_days : ℕ)

-- Define the total compensation for Xiao Li
def xiao_li_compensation (xiao_li_days daily_reward : ℕ) : ℕ := xiao_li_days * daily_reward

-- The theorem statement asserting the final answer
theorem find_xiao_li_compensation
  (h1 : total_days = 12)
  (h2 : extra_days = 3)
  (h3 : extra_compensation = 2700)
  (h4 : daily_work = 1)
  (h5 : daily_reward = 225)
  (h6 : xiao_li_days = 2)
  (h7 : (total_days - extra_days) * daily_work = xiao_li_days * daily_work):
  xiao_li_compensation xiao_li_days daily_reward = 450 := 
sorry

end find_xiao_li_compensation_l119_119526


namespace multiply_expand_l119_119837

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l119_119837


namespace solve_expression_l119_119649

theorem solve_expression :
  2^3 + 2 * 5 - 3 + 6 = 21 :=
by
  sorry

end solve_expression_l119_119649


namespace prob_student_A_consecutive_days_l119_119652

/--
There are 4 students, and each student is assigned to participate in a 5-day volunteer activity.
- Student A participates for exactly 2 days.
- Each of the other 3 students participates for exactly 1 day.

Prove that the probability of student A participating in two consecutive days is 2/5.
-/
theorem prob_student_A_consecutive_days :
  let total_events := Nat.choose 5 2 * nat.perm 3 3,
      favorable_events := 4 * nat.perm 3 3
  in (favorable_events / total_events : ℚ) = 2 / 5 :=
by
  sorry

end prob_student_A_consecutive_days_l119_119652


namespace multiply_expression_l119_119831

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l119_119831


namespace sin_double_angle_l119_119755

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (Real.pi / 4 - α) = -3 / 5) :
  Real.sin (2 * α) = -7 / 25 := by
sorry

end sin_double_angle_l119_119755


namespace max_a3_b3_c3_d3_l119_119225

-- Define that a, b, c, d are real numbers that satisfy the given conditions.
theorem max_a3_b3_c3_d3 (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 16)
  (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 :=
sorry

end max_a3_b3_c3_d3_l119_119225


namespace cyclist_wait_time_l119_119544

noncomputable def hiker_speed : ℝ := 5 / 60
noncomputable def cyclist_speed : ℝ := 25 / 60
noncomputable def wait_time : ℝ := 5
noncomputable def distance_ahead : ℝ := cyclist_speed * wait_time
noncomputable def catching_time : ℝ := distance_ahead / hiker_speed

theorem cyclist_wait_time : catching_time = 25 := by
  sorry

end cyclist_wait_time_l119_119544


namespace total_jumps_l119_119488

-- Definitions based on given conditions
def Ronald_jumps : ℕ := 157
def Rupert_jumps : ℕ := Ronald_jumps + 86

-- The theorem we want to prove
theorem total_jumps : Ronald_jumps + Rupert_jumps = 400 :=
by
  sorry

end total_jumps_l119_119488


namespace solution_set_absolute_value_inequality_l119_119110

theorem solution_set_absolute_value_inequality (x : ℝ) :
  (|x-3| + |x-5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solution_set_absolute_value_inequality_l119_119110


namespace exists_k_undecided_l119_119635

def tournament (n : ℕ) : Type :=
  { T : Fin n → Fin n → Prop // ∀ i j, T i j = ¬T j i }

def k_undecided (n k : ℕ) (T : tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k → ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → T.1 p a

theorem exists_k_undecided (k : ℕ) (hk : 0 < k) : ∃ (n : ℕ), n > k ∧ ∃ (T : tournament n), k_undecided n k T :=
by
  sorry

end exists_k_undecided_l119_119635


namespace pencil_sharpening_and_breaking_l119_119041

/-- Isha's pencil initially has a length of 31 inches. After sharpening, it has a length of 14 inches.
Prove that:
1. The pencil was shortened by 17 inches.
2. Each half of the pencil, after being broken in half, is 7 inches long. -/
theorem pencil_sharpening_and_breaking 
  (initial_length : ℕ) 
  (length_after_sharpening : ℕ) 
  (sharpened_length : ℕ) 
  (half_length : ℕ) 
  (h1 : initial_length = 31) 
  (h2 : length_after_sharpening = 14) 
  (h3 : sharpened_length = initial_length - length_after_sharpening) 
  (h4 : half_length = length_after_sharpening / 2) : 
  sharpened_length = 17 ∧ half_length = 7 := 
by {
  sorry
}

end pencil_sharpening_and_breaking_l119_119041


namespace multiplication_identity_multiplication_l119_119841

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l119_119841


namespace sum_of_coordinates_of_other_endpoint_of_segment_l119_119233

theorem sum_of_coordinates_of_other_endpoint_of_segment {x y : ℝ}
  (h1 : (6 + x) / 2 = 3)
  (h2 : (1 + y) / 2 = 7) :
  x + y = 13 := by
  sorry

end sum_of_coordinates_of_other_endpoint_of_segment_l119_119233


namespace probability_of_selecting_cooking_is_one_fourth_l119_119376

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119376


namespace flyers_total_l119_119795

theorem flyers_total (jack_flyers : ℕ) (rose_flyers : ℕ) (left_flyers : ℕ) 
  (hj : jack_flyers = 120) (hr : rose_flyers = 320) (hl : left_flyers = 796) :
  jack_flyers + rose_flyers + left_flyers = 1236 :=
by {
  sorry
}

end flyers_total_l119_119795


namespace part1_part2_l119_119618

noncomputable def a (n : ℕ) : ℤ :=
  15 * n + 2 + (15 * n - 32) * 16^(n-1)

theorem part1 (n : ℕ) : 15^3 ∣ (a n) := by
  sorry

-- Correct answer for part (2) bundled in a formal statement:
theorem part2 (n k : ℕ) : 1991 ∣ (a n) ∧ 1991 ∣ (a (n + 1)) ∧
    1991 ∣ (a (n + 2)) ↔ n = 89595 * k := by
  sorry

end part1_part2_l119_119618


namespace right_triangle_other_acute_angle_l119_119454

theorem right_triangle_other_acute_angle (A B C : ℝ) (r : A + B + C = 180) (h : A = 90) (a : B = 30) :
  C = 60 :=
sorry

end right_triangle_other_acute_angle_l119_119454


namespace repeating_decimals_sum_l119_119590

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l119_119590


namespace probability_of_cooking_l119_119369

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119369


namespace total_raised_is_420_l119_119862

def pancake_cost : ℝ := 4.00
def bacon_cost : ℝ := 2.00
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90

theorem total_raised_is_420 : (pancake_cost * stacks_sold + bacon_cost * slices_sold) = 420.00 :=
by
  -- Proof goes here
  sorry

end total_raised_is_420_l119_119862


namespace probability_of_cooking_l119_119364

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119364


namespace P_is_necessary_but_not_sufficient_for_Q_l119_119424

def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem P_is_necessary_but_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
by
  sorry

end P_is_necessary_but_not_sufficient_for_Q_l119_119424


namespace opposite_of_neg_half_is_half_l119_119094

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l119_119094


namespace max_edges_convex_polyhedron_l119_119725

theorem max_edges_convex_polyhedron (n : ℕ) (c l e : ℕ) (h1 : c = n) (h2 : c + l = e + 2) (h3 : 2 * e ≥ 3 * l) : e ≤ 3 * n - 6 := 
sorry

end max_edges_convex_polyhedron_l119_119725


namespace hcf_of_numbers_l119_119448

theorem hcf_of_numbers (x y : ℕ) (hcf lcm : ℕ) 
    (h_sum : x + y = 45) 
    (h_lcm : lcm = 100)
    (h_reciprocal_sum : 1 / (x : ℝ) + 1 / (y : ℝ) = 0.3433333333333333) :
    hcf = 1 :=
by
  sorry

end hcf_of_numbers_l119_119448


namespace points_on_ellipse_l119_119609

noncomputable def x (t : ℝ) : ℝ := (3 - t^2) / (1 + t^2)
noncomputable def y (t : ℝ) : ℝ := 4 * t / (1 + t^2)

theorem points_on_ellipse : ∀ t : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (x t / a)^2 + (y t / b)^2 = 1 := 
sorry

end points_on_ellipse_l119_119609


namespace row_column_crossout_l119_119792

theorem row_column_crossout (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ rows : Finset (Fin 1000), rows.card = 990 ∧ ∀ j : Fin 1000, ∃ i ∈ rowsᶜ, M i j = 1) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 990 ∧ ∀ i : Fin 1000, ∃ j ∈ colsᶜ, M i j = 0) :=
by {
  sorry
}

end row_column_crossout_l119_119792


namespace geometric_sequence_sum_l119_119447

theorem geometric_sequence_sum (S : ℕ → ℝ) 
  (S5 : S 5 = 10)
  (S10 : S 10 = 50) :
  S 15 = 210 := 
by
  sorry

end geometric_sequence_sum_l119_119447


namespace num_digits_of_prime_started_numerals_l119_119258

theorem num_digits_of_prime_started_numerals (n : ℕ) (h : 4 * 10^(n-1) = 400) : n = 3 := 
  sorry

end num_digits_of_prime_started_numerals_l119_119258


namespace combined_mpg_l119_119851

theorem combined_mpg
  (R_eff : ℝ) (T_eff : ℝ)
  (R_dist : ℝ) (T_dist : ℝ)
  (H_R_eff : R_eff = 35)
  (H_T_eff : T_eff = 15)
  (H_R_dist : R_dist = 420)
  (H_T_dist : T_dist = 300)
  : (R_dist + T_dist) / (R_dist / R_eff + T_dist / T_eff) = 22.5 := 
by
  rw [H_R_eff, H_T_eff, H_R_dist, H_T_dist]
  -- Proof steps would go here, but we'll use sorry to skip it.
  sorry

end combined_mpg_l119_119851


namespace condition_suff_not_necess_l119_119813

theorem condition_suff_not_necess (x : ℝ) (h : |x - (1 / 2)| < 1 / 2) : x^3 < 1 :=
by
  have h1 : 0 < x := sorry
  have h2 : x < 1 := sorry
  sorry

end condition_suff_not_necess_l119_119813


namespace problem_proof_l119_119456

variable (A B C a b c : ℝ)
variable (ABC_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (sides_opposite : a = (b * sin A / sin B) ∧ b = (a * sin B / sin A))
variable (cos_eq : b + b * cos A = a * cos B)

theorem problem_proof :
  (A = 2 * B ∧ (π / 6 < B ∧ B < π / 4) ∧ a^2 = b^2 + b * c) :=
  sorry

end problem_proof_l119_119456


namespace range_of_sum_l119_119773

theorem range_of_sum (a b : ℝ) (h1 : -2 < a) (h2 : a < -1) (h3 : -1 < b) (h4 : b < 0) : 
  -3 < a + b ∧ a + b < -1 :=
by
  sorry

end range_of_sum_l119_119773


namespace probability_selecting_cooking_l119_119307

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119307


namespace average_weight_decrease_l119_119493

theorem average_weight_decrease 
  (A1 : ℝ) (new_person_weight : ℝ) (num_initial : ℕ) (num_total : ℕ) 
  (hA1 : A1 = 55) (hnew_person_weight : new_person_weight = 50) 
  (hnum_initial : num_initial = 20) (hnum_total : num_total = 21) :
  A1 - ((A1 * num_initial + new_person_weight) / num_total) = 0.24 :=
by
  rw [hA1, hnew_person_weight, hnum_initial, hnum_total]
  -- Further proof steps would go here
  sorry

end average_weight_decrease_l119_119493


namespace exists_triangle_free_not_4_colorable_l119_119611

/-- Define a graph as a structure with vertices and edges. -/
structure Graph (V : Type*) :=
  (adj : V → V → Prop)
  (symm : ∀ x y, adj x y → adj y x)
  (irreflexive : ∀ x, ¬adj x x)

/-- A definition of triangle-free graph. -/
def triangle_free {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c : V), G.adj a b → G.adj b c → G.adj c a → false

/-- A definition that a graph cannot be k-colored. -/
def not_k_colorable {V : Type*} (G : Graph V) (k : ℕ) : Prop :=
  ¬∃ (f : V → ℕ), (∀ (v : V), f v < k) ∧ (∀ (v w : V), G.adj v w → f v ≠ f w)

/-- There exists a triangle-free graph that is not 4-colorable. -/
theorem exists_triangle_free_not_4_colorable : ∃ (V : Type*) (G : Graph V), triangle_free G ∧ not_k_colorable G 4 := 
sorry

end exists_triangle_free_not_4_colorable_l119_119611


namespace max_students_l119_119494

-- Defining the problem's conditions
def cost_bus_rental : ℕ := 100
def max_capacity_students : ℕ := 25
def cost_per_student : ℕ := 10
def teacher_admission_cost : ℕ := 0
def total_budget : ℕ := 350

-- The Lean proof problem
theorem max_students (bus_cost : ℕ) (student_capacity : ℕ) (student_cost : ℕ) (teacher_cost : ℕ) (budget : ℕ) :
  bus_cost = cost_bus_rental → 
  student_capacity = max_capacity_students →
  student_cost = cost_per_student →
  teacher_cost = teacher_admission_cost →
  budget = total_budget →
  (student_capacity ≤ (budget - bus_cost) / student_cost) → 
  ∃ n : ℕ, n = student_capacity ∧ n ≤ (budget - bus_cost) / student_cost :=
by
  intros
  sorry

end max_students_l119_119494


namespace ratio_arithmetic_progression_l119_119206

theorem ratio_arithmetic_progression (a d : ℕ) 
  (h1 : a ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : 15 * (2 * a + 14 * d) = 3 * 8 * (2 * a + 7 * d)) :
  a / d = 7 / 3 :=
begin
  -- Proof is not needed as per instructions.
  sorry
end

end ratio_arithmetic_progression_l119_119206


namespace multiply_expression_l119_119830

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l119_119830


namespace volume_ratio_sphere_cylinder_inscribed_l119_119730

noncomputable def ratio_of_volumes (d : ℝ) : ℝ :=
  let Vs := (4 / 3) * Real.pi * (d / 2)^3
  let Vc := Real.pi * (d / 2)^2 * d
  Vs / Vc

theorem volume_ratio_sphere_cylinder_inscribed (d : ℝ) (h : d > 0) : 
  ratio_of_volumes d = 2 / 3 := 
by
  sorry

end volume_ratio_sphere_cylinder_inscribed_l119_119730


namespace boxes_calculation_l119_119273

theorem boxes_calculation (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) (boxes : ℕ) :
  total_bottles = 8640 → bottles_per_bag = 12 → bags_per_box = 6 → boxes = total_bottles / (bottles_per_bag * bags_per_box) → boxes = 120 :=
by
  intros h_total h_bottles_per_bag h_bags_per_box h_boxes
  rw [h_total, h_bottles_per_bag, h_bags_per_box] at h_boxes
  norm_num at h_boxes
  exact h_boxes

end boxes_calculation_l119_119273


namespace probability_of_selecting_cooking_l119_119287

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119287


namespace area_of_side_face_l119_119931

theorem area_of_side_face (l w h : ℝ)
  (h_front_top : w * h = 0.5 * (l * h))
  (h_top_side : l * h = 1.5 * (w * h))
  (h_volume : l * w * h = 3000) :
  w * h = 200 := 
sorry

end area_of_side_face_l119_119931


namespace fiona_received_59_l119_119394

theorem fiona_received_59 (Dan_riddles : ℕ) (Andy_riddles : ℕ) (Bella_riddles : ℕ) (Emma_riddles : ℕ) (Fiona_riddles : ℕ)
  (h1 : Dan_riddles = 21)
  (h2 : Andy_riddles = Dan_riddles + 12)
  (h3 : Bella_riddles = Andy_riddles - 7)
  (h4 : Emma_riddles = Bella_riddles / 2)
  (h5 : Fiona_riddles = Andy_riddles + Bella_riddles) :
  Fiona_riddles = 59 :=
by
  sorry

end fiona_received_59_l119_119394


namespace union_complement_subset_range_l119_119769

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | 2 * x ^ 2 - 3 * x - 2 < 0}

-- Define the complement of B
def complement_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- 1. The proof problem for A ∪ (complement of B) when a = 1
theorem union_complement (a : ℝ) (h : a = 1) :
  { x : ℝ | (-1/2 < x ∧ x ≤ 1) ∨ (x ≥ 2 ∨ x ≤ -1/2) } = 
  { x : ℝ | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

-- 2. The proof problem for A ⊆ B to find the range of a
theorem subset_range (a : ℝ) :
  (∀ x, A a x → B x) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end union_complement_subset_range_l119_119769


namespace bisect_area_of_trapezoid_l119_119071

-- Define the vertices of the quadrilateral
structure Point :=
  (x : ℤ)
  (y : ℤ)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 16, y := 0 }
def C : Point := { x := 8, y := 8 }
def D : Point := { x := 0, y := 8 }

-- Define the equation of a line
structure Line :=
  (slope : ℚ)
  (intercept : ℚ)

-- Define the condition for parallel lines
def parallel (L1 L2 : Line) : Prop :=
  L1.slope = L2.slope

-- Define the diagonal AC and the required line
def AC : Line := { slope := 1, intercept := 0 }
def bisecting_line : Line := { slope := 1, intercept := -4 }

-- The area of trapezoid
def trapezoid_area : ℚ := (8 * (16 + 8)) / 2

-- Proof that the required line is parallel to AC and bisects the area of the trapezoid
theorem bisect_area_of_trapezoid :
  parallel bisecting_line AC ∧ 
  (1 / 2) * (8 * (16 + bisecting_line.intercept)) = trapezoid_area / 2 :=
by
  sorry

end bisect_area_of_trapezoid_l119_119071


namespace river_length_l119_119998

theorem river_length :
  let still_water_speed := 10 -- Karen's paddling speed on still water in miles per hour
  let current_speed      := 4  -- River's current speed in miles per hour
  let time               := 2  -- Time it takes Karen to paddle up the river in hours
  let effective_speed    := still_water_speed - current_speed -- Karen's effective speed against the current
  effective_speed * time = 12 -- Length of the river in miles
:= by
  sorry

end river_length_l119_119998


namespace milan_billed_minutes_l119_119166

-- Define the conditions
def monthly_fee : ℝ := 2
def cost_per_minute : ℝ := 0.12
def total_bill : ℝ := 23.36

-- Define the number of minutes based on the above conditions
def minutes := (total_bill - monthly_fee) / cost_per_minute

-- Prove that the number of minutes is 178
theorem milan_billed_minutes : minutes = 178 := by
  -- Proof steps would go here, but as instructed, we use 'sorry' to skip the proof.
  sorry

end milan_billed_minutes_l119_119166


namespace opposite_of_neg_half_l119_119078

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l119_119078


namespace combined_exceeds_limit_l119_119057

-- Let Zone A, Zone B, and Zone C be zones on a road.
-- Let pA be the percentage of motorists exceeding the speed limit in Zone A.
-- Let pB be the percentage of motorists exceeding the speed limit in Zone B.
-- Let pC be the percentage of motorists exceeding the speed limit in Zone C.
-- Each zone has an equal amount of motorists.

def pA : ℝ := 15
def pB : ℝ := 20
def pC : ℝ := 10

/-
Prove that the combined percentage of motorists who exceed the speed limit
across all three zones is 15%.
-/
theorem combined_exceeds_limit :
  (pA + pB + pC) / 3 = 15 := 
by sorry

end combined_exceeds_limit_l119_119057


namespace aiden_nap_is_15_minutes_l119_119913

def aiden_nap_duration_in_minutes (nap_in_hours : ℚ) (minutes_per_hour : ℕ) : ℚ :=
  nap_in_hours * minutes_per_hour

theorem aiden_nap_is_15_minutes :
  aiden_nap_duration_in_minutes (1/4) 60 = 15 := by
  sorry

end aiden_nap_is_15_minutes_l119_119913


namespace integer_solution_exists_l119_119871

theorem integer_solution_exists
  (a b c : ℝ)
  (H1 : ∃ q1 : ℚ, a * b = q1)
  (H2 : ∃ q2 : ℚ, b * c = q2)
  (H3 : ∃ q3 : ℚ, c * a = q3)
  (H4 : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ x y z : ℤ, a * (x : ℝ) + b * (y : ℝ) + c * (z : ℝ) = 0 := 
sorry

end integer_solution_exists_l119_119871


namespace positive_number_representation_l119_119917

theorem positive_number_representation (a : ℝ) : 
  (a > 0) ↔ (a ≠ 0 ∧ a > 0 ∧ ¬(a < 0)) :=
by 
  sorry

end positive_number_representation_l119_119917


namespace rows_with_exactly_10_people_l119_119546

theorem rows_with_exactly_10_people (x : ℕ) (total_people : ℕ) (row_nine_seat : ℕ) (row_ten_seat : ℕ) 
    (H1 : row_nine_seat = 9) (H2 : row_ten_seat = 10) 
    (H3 : total_people = 55) 
    (H4 : total_people = x * row_ten_seat + (6 - x) * row_nine_seat) 
    : x = 1 :=
by
  sorry

end rows_with_exactly_10_people_l119_119546


namespace least_three_digit_divisible_by_2_5_7_3_l119_119268

theorem least_three_digit_divisible_by_2_5_7_3 : 
  ∃ n, n = 210 ∧ (100 ≤ n) ∧ 
           (n < 1000) ∧ 
           (n % 2 = 0) ∧ 
           (n % 5 = 0) ∧ 
           (n % 7 = 0) ∧ 
           (n % 3 = 0) :=
by
  use 210
  split
  rfl
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  norm_num

end least_three_digit_divisible_by_2_5_7_3_l119_119268


namespace garden_area_in_square_meters_l119_119887

def garden_width_cm : ℕ := 500
def garden_length_cm : ℕ := 800
def conversion_factor_cm2_to_m2 : ℕ := 10000

theorem garden_area_in_square_meters : (garden_length_cm * garden_width_cm) / conversion_factor_cm2_to_m2 = 40 :=
by
  sorry

end garden_area_in_square_meters_l119_119887


namespace number_of_oranges_l119_119060

theorem number_of_oranges :
  ∃ o m : ℕ, (m + 6 * m + o = 20) ∧ (6 * m > o) ∧ (2 ≤ m) ∧ (m ≤ 2) ∧ (o = 6) :=
begin
  -- instantiating variables m and o
  use [2, 6],
  -- prove and this would skip the proof,
  split, linarith, 
  split, linarith,
  split, linarith,
  linarith,
end

end number_of_oranges_l119_119060


namespace cos_5theta_l119_119196

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end cos_5theta_l119_119196


namespace sides_of_triangle_expr_negative_l119_119775

theorem sides_of_triangle_expr_negative (a b c : ℝ) 
(h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
(a - c)^2 - b^2 < 0 :=
sorry

end sides_of_triangle_expr_negative_l119_119775


namespace tangent_line_to_parabola_k_value_l119_119943

theorem tangent_line_to_parabola_k_value (k : ℝ) :
  (∀ x y : ℝ, 4 * x - 3 * y + k = 0 → y^2 = 16 * x → (4 * x - 3 * y + k = 0 ∧ y^2 = 16 * x) ∧ (144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_k_value_l119_119943


namespace value_of_a_l119_119629

-- Define the given conditions
def b := 2
def A := 45 * Real.Angle.deg
def C := 75 * Real.Angle.deg
noncomputable def sin := Real.sin
noncomputable def angle := Real.Angle

-- Define the calculated value of B
def B := 180 * angle.deg - A - C

-- Define sine values for specific angles
noncomputable def sin_45 := sin (45 * angle.deg)
noncomputable def sin_60 := sin (60 * angle.deg)

-- Define side a using the Sine Rule and given conditions
noncomputable def a := b * sin_45 / sin_60

-- Statement to prove
theorem value_of_a : a = (2 / 3) * Real.sqrt 6 := by sorry

end value_of_a_l119_119629


namespace probability_of_selecting_cooking_l119_119336

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119336


namespace log_base_10_of_50_eq_1_plus_log_base_10_of_5_l119_119738

variables {a b : ℝ}

theorem log_base_10_of_50_eq_1_plus_log_base_10_of_5 :
  log 10 50 = 1 + log 10 5 :=
by
  -- Using \(50 = 5 \times 10\)
  have h1 : 50 = 5 * 10 := by norm_num
  -- Applying the logarithmic product rule \(\log_{10}(a \times b) = \log_{10}(a) + \log_{10}(b)\)
  have h2 : log 10 50 = log 10 (5 * 10) := by rw h1
  have h3 : log 10 (5 * 10) = log 10 5 + log 10 10 := log_mul (by norm_num : 1 < 10)
  -- Knowing \(\log_{10}(10) = 1\)
  have h4 : log 10 10 = 1 := log_base_pow 10 (by norm_num : 1 < 10)
  -- Combining everything to establish the final equality
  rw [h2, h3, h4]
  sorry

end log_base_10_of_50_eq_1_plus_log_base_10_of_5_l119_119738


namespace Natasha_avg_speed_climb_l119_119231

-- Definitions for conditions
def distance_to_top : ℝ := sorry -- We need to find this
def time_up := 3 -- time in hours to climb up
def time_down := 2 -- time in hours to climb down
def avg_speed_journey := 3 -- avg speed in km/hr for the whole journey

-- Equivalent math proof problem statement
theorem Natasha_avg_speed_climb (distance_to_top : ℝ) 
  (h1 : time_up = 3)
  (h2 : time_down = 2)
  (h3 : avg_speed_journey = 3)
  (h4 : (2 * distance_to_top) / (time_up + time_down) = avg_speed_journey) : 
  (distance_to_top / time_up) = 2.5 :=
sorry -- Proof not required

end Natasha_avg_speed_climb_l119_119231


namespace ellipse_equation_and_max_area_l119_119765

noncomputable def ellipse_standard_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∃ (b' : ℝ) (e : ℝ), b' = b ∧ a > b' ∧ b' = sqrt 3 ∧ e = 0.5 ∧ a^2 = b'^2 + (e * a)^2

noncomputable def maximum_area_triangle (a b : ℝ) (ha : a > b) (hb : b > 0) : ℝ :=
  let c := sqrt (a^2 - b^2) in
  let f1 := (0, 0) in
  let f2 := (c, 0) in
  3

theorem ellipse_equation_and_max_area :
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ ellipse_standard_equation a b (by linarith [show 2 > 0 from dec_trivial]) (by linarith [show sqrt 3 > 0 from dec_trivial]) ∧ 
  maximum_area_triangle a b (by exact zero_lt_two) (by exact sqrt_pos_of_pos zero_lt_three) = 3 :=
begin
  use [2, sqrt 3],
  split,
  { refl },
  split,
  { refl },
  split,
  { use [sqrt 3, 0.5],
    exact ⟨rfl, by linarith, rfl, rfl, by linarith⟩ },
  { exact rfl }
end

end ellipse_equation_and_max_area_l119_119765


namespace exam_question_bound_l119_119532

theorem exam_question_bound (n_students : ℕ) (k_questions : ℕ) (n_answers : ℕ) 
    (H_students : n_students = 25) (H_answers : n_answers = 5) 
    (H_condition : ∀ (i j : ℕ) (H1 : i < n_students) (H2 : j < n_students) (H_neq : i ≠ j), 
      ∀ q : ℕ, q < k_questions → ∀ ai aj : ℕ, ai < n_answers → aj < n_answers → 
      ((ai = aj) → (i = j ∨ q' > 1))) : 
    k_questions ≤ 6 := 
sorry

end exam_question_bound_l119_119532


namespace milan_billed_minutes_l119_119164

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l119_119164


namespace apples_needed_per_month_l119_119404

theorem apples_needed_per_month (chandler_apples_per_week : ℕ) (lucy_apples_per_week : ℕ) (weeks_per_month : ℕ)
  (h1 : chandler_apples_per_week = 23)
  (h2 : lucy_apples_per_week = 19)
  (h3 : weeks_per_month = 4) :
  (chandler_apples_per_week + lucy_apples_per_week) * weeks_per_month = 168 :=
by
  sorry

end apples_needed_per_month_l119_119404


namespace find_width_l119_119646

theorem find_width (A : ℕ) (hA : A ≥ 120) (w : ℕ) (l : ℕ) (hl : l = w + 20) (h_area : w * l = A) : w = 4 :=
by sorry

end find_width_l119_119646


namespace problem_prove_divisibility_l119_119236

theorem problem_prove_divisibility (n : ℕ) : 11 ∣ (5^(2*n) + 3^(n+2) + 3^n) :=
sorry

end problem_prove_divisibility_l119_119236


namespace tap_B_time_l119_119918

-- Define the capacities and time variables
variable (A_rate B_rate : ℝ) -- rates in percentage per hour
variable (T_A T_B : ℝ) -- time in hours

-- Define the conditions as hypotheses
def conditions : Prop :=
  (4 * (A_rate + B_rate) = 50) ∧ (2 * A_rate = 15)

-- Define the question and the target time
def target_time := 7

-- Define the goal to prove
theorem tap_B_time (h : conditions A_rate B_rate) : T_B = target_time := by
  sorry

end tap_B_time_l119_119918


namespace polynomial_remainder_l119_119159

theorem polynomial_remainder (x : ℝ) : 
  let f := λ x : ℝ, x^3 - 4 * x + 6 in
  f (-3) = -9 :=
by
  let f := λ x : ℝ, x^3 - 4 * x + 6
  show f (-3) = -9
  sorry

end polynomial_remainder_l119_119159


namespace men_entered_room_l119_119991

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l119_119991


namespace workshop_c_defective_rate_l119_119625

theorem workshop_c_defective_rate :
  let P_B1 := 0.45 in
  let P_B2 := 0.35 in
  let P_B3 := 0.20 in
  let P_A_given_B1 := 0.02 in
  let P_A_given_B2 := 0.03 in
  let P_A := 0.0295 in
  ∃ m : ℝ, (m = 0.05) ∧ (P_A = P_A_given_B1 * P_B1 + P_A_given_B2 * P_B2 + m * P_B3) :=
by
  sorry

end workshop_c_defective_rate_l119_119625


namespace cos_5theta_l119_119197

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end cos_5theta_l119_119197


namespace probability_of_selecting_cooking_is_one_fourth_l119_119374

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119374


namespace probability_of_selecting_cooking_l119_119332

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119332


namespace jacket_initial_reduction_l119_119504

theorem jacket_initial_reduction (x : ℝ) :
  (1 - x / 100) * 1.53846 = 1 → x = 35 :=
by
  sorry

end jacket_initial_reduction_l119_119504


namespace factor_polynomial_l119_119152

theorem factor_polynomial (z : ℝ) : (70 * z ^ 20 + 154 * z ^ 40 + 224 * z ^ 60) = 14 * z ^ 20 * (5 + 11 * z ^ 20 + 16 * z ^ 40) := 
sorry

end factor_polynomial_l119_119152


namespace cost_price_percentage_of_marked_price_l119_119496

theorem cost_price_percentage_of_marked_price
  (MP : ℝ) -- Marked Price
  (CP : ℝ) -- Cost Price
  (discount_percent : ℝ) (gain_percent : ℝ)
  (H1 : CP = (x / 100) * MP) -- Cost Price is x percent of Marked Price
  (H2 : discount_percent = 13) -- Discount percentage
  (H3 : gain_percent = 55.35714285714286) -- Gain percentage
  : x = 56 :=
sorry

end cost_price_percentage_of_marked_price_l119_119496


namespace exists_n_divides_2022n_minus_n_l119_119810

theorem exists_n_divides_2022n_minus_n (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ n : ℕ, p ∣ (2022^n - n) :=
sorry

end exists_n_divides_2022n_minus_n_l119_119810


namespace range_of_c_extreme_values_l119_119977

noncomputable def f (c x : ℝ) : ℝ := x^3 - 2 * c * x^2 + x

theorem range_of_c_extreme_values 
  (c : ℝ) 
  (h : ∃ a b : ℝ, a ≠ b ∧ (3 * a^2 - 4 * c * a + 1 = 0) ∧ (3 * b^2 - 4 * c * b + 1 = 0)) :
  c < - (Real.sqrt 3 / 2) ∨ c > (Real.sqrt 3 / 2) :=
by sorry

end range_of_c_extreme_values_l119_119977


namespace binomial_coeff_and_coeff_of_x8_l119_119628

theorem binomial_coeff_and_coeff_of_x8 (x : ℂ) :
  let expr := (x^2 + 4*x + 4)^5
  let expansion := (x + 2)^10
  ∃ (binom_coeff_x8 coeff_x8 : ℤ),
    binom_coeff_x8 = 45 ∧ coeff_x8 = 180 :=
by
  sorry

end binomial_coeff_and_coeff_of_x8_l119_119628


namespace minimum_cost_is_correct_l119_119222

noncomputable def rectangular_area (length width : ℝ) : ℝ :=
  length * width

def flower_cost_per_sqft (flower : String) : ℝ :=
  match flower with
  | "Marigold" => 1.00
  | "Sunflower" => 1.75
  | "Tulip" => 1.25
  | "Orchid" => 2.75
  | "Iris" => 3.25
  | _ => 0.00

def min_garden_cost : ℝ :=
  let areas := [rectangular_area 5 2, rectangular_area 7 3, rectangular_area 5 5, rectangular_area 2 4, rectangular_area 5 4]
  let costs := [flower_cost_per_sqft "Orchid" * 8, 
                flower_cost_per_sqft "Iris" * 10, 
                flower_cost_per_sqft "Sunflower" * 20, 
                flower_cost_per_sqft "Tulip" * 21, 
                flower_cost_per_sqft "Marigold" * 25]
  costs.sum

theorem minimum_cost_is_correct :
  min_garden_cost = 140.75 :=
  by
    -- Proof omitted
    sorry

end minimum_cost_is_correct_l119_119222


namespace cost_of_blue_hat_is_six_l119_119263

-- Given conditions
def total_hats : ℕ := 85
def green_hats : ℕ := 40
def blue_hats : ℕ := total_hats - green_hats
def cost_green_hat : ℕ := 7
def total_cost : ℕ := 550
def total_cost_green_hats : ℕ := green_hats * cost_green_hat
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats
def cost_blue_hat : ℕ := total_cost_blue_hats / blue_hats

-- Proof statement
theorem cost_of_blue_hat_is_six : cost_blue_hat = 6 := sorry

end cost_of_blue_hat_is_six_l119_119263


namespace curves_intersection_four_points_l119_119740

theorem curves_intersection_four_points (b : ℝ) :
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
    x1^2 + y1^2 = b^2 ∧ y1 = x1^2 - b + 1 ∧
    x2^2 + y2^2 = b^2 ∧ y2 = x2^2 - b + 1 ∧
    x3^2 + y3^2 = b^2 ∧ y3 = x3^2 - b + 1 ∧
    x4^2 + y4^2 = b^2 ∧ y4 = x4^2 - b + 1 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧
    (x3, y3) ≠ (x4, y4)) →
  b > 2 :=
sorry

end curves_intersection_four_points_l119_119740


namespace probability_select_cooking_l119_119318

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119318


namespace x_increase_80_percent_l119_119673

noncomputable def percentage_increase (x1 x2 : ℝ) : ℝ :=
  ((x2 / x1) - 1) * 100

theorem x_increase_80_percent
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 * y1 = x2 * y2)
  (h2 : y2 = (5 / 9) * y1) :
  percentage_increase x1 x2 = 80 :=
by
  sorry

end x_increase_80_percent_l119_119673


namespace find_y_l119_119893

-- Define the problem conditions
variable (x y : ℕ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (rem_eq : x % y = 3)
variable (div_eq : (x : ℝ) / y = 96.12)

-- The theorem to prove
theorem find_y : y = 25 :=
sorry

end find_y_l119_119893


namespace jiwoo_magnets_two_digit_count_l119_119467

def num_magnets : List ℕ := [1, 2, 7]

theorem jiwoo_magnets_two_digit_count : 
  (∀ (x y : ℕ), x ≠ y → x ∈ num_magnets → y ∈ num_magnets → 2 * 3 = 6) := 
by {
  sorry
}

end jiwoo_magnets_two_digit_count_l119_119467


namespace repeating_decimal_sum_l119_119596

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l119_119596


namespace new_average_l119_119278

theorem new_average (avg : ℕ) (n : ℕ) (k : ℕ) (new_avg : ℕ) 
  (h1 : avg = 23) (h2 : n = 10) (h3 : k = 4) : 
  new_avg = (n * avg + n * k) / n → new_avg = 27 :=
by
  intro H
  sorry

end new_average_l119_119278


namespace radius_of_inscribed_circle_l119_119748

theorem radius_of_inscribed_circle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = a + b - c :=
sorry

end radius_of_inscribed_circle_l119_119748


namespace rectangle_area_increase_l119_119858

theorem rectangle_area_increase (x y : ℕ) 
  (hxy : x * y = 180) 
  (hperimeter : 2 * x + 2 * y = 54) : 
  (x + 6) * (y + 6) = 378 :=
by sorry

end rectangle_area_increase_l119_119858


namespace probability_cooking_is_one_fourth_l119_119359
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119359


namespace Tucker_last_number_l119_119261

-- Define the sequence of numbers said by Todd, Tadd, and Tucker
def game_sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 4
  else if n = 5 then 5
  else if n = 6 then 6
  else sorry -- Define recursively for subsequent rounds

-- Condition: The game ends when they reach the number 1000.
def game_end := 1000

-- Define the function to determine the last number said by Tucker
def last_number_said_by_Tucker (end_num : ℕ) : ℕ :=
  -- Assuming this function correctly calculates the last number said by Tucker
  if end_num = game_end then 1000 else sorry

-- Problem statement to prove
theorem Tucker_last_number : last_number_said_by_Tucker game_end = 1000 := by
  sorry

end Tucker_last_number_l119_119261


namespace clock_angle_at_seven_l119_119691

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l119_119691


namespace correct_equation_l119_119982

/-- Definitions and conditions used in the problem -/
def jan_revenue := 250
def feb_revenue (x : ℝ) := jan_revenue * (1 + x)
def mar_revenue (x : ℝ) := jan_revenue * (1 + x)^2
def first_quarter_target := 900

/-- Proof problem statement -/
theorem correct_equation (x : ℝ) : 
  jan_revenue + feb_revenue x + mar_revenue x = first_quarter_target := 
by
  sorry

end correct_equation_l119_119982


namespace math_problem_proof_l119_119805

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end math_problem_proof_l119_119805


namespace billy_initial_lemon_heads_l119_119927

theorem billy_initial_lemon_heads (n f : ℕ) (h_friends : f = 6) (h_eat : n = 12) :
  f * n = 72 := 
by
  -- Proceed by proving the statement using Lean
  sorry

end billy_initial_lemon_heads_l119_119927


namespace opposite_of_neg_half_is_half_l119_119093

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l119_119093


namespace rectangle_area_change_l119_119530

theorem rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  A' = 0.92 * A :=
by
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  show A' = 0.92 * A
  sorry

end rectangle_area_change_l119_119530


namespace inf_many_non_prime_additions_l119_119854

theorem inf_many_non_prime_additions :
  ∃ᶠ (a : ℕ) in at_top, ∀ n : ℕ, n > 0 → ¬ Prime (n^4 + a) :=
by {
  sorry -- proof to be provided
}

end inf_many_non_prime_additions_l119_119854


namespace circumference_of_smaller_circle_l119_119184

variable (R : ℝ)
variable (A_shaded : ℝ)

theorem circumference_of_smaller_circle :
  (A_shaded = (32 / π) ∧ 3 * (π * R ^ 2) - π * R ^ 2 = A_shaded) → 
  2 * π * R = 4 :=
by
  sorry

end circumference_of_smaller_circle_l119_119184


namespace num_A_is_9_l119_119445

-- Define the total number of animals
def total_animals : ℕ := 17

-- Define the number of animal B
def num_B : ℕ := 8

-- Define the number of animal A
def num_A : ℕ := total_animals - num_B

-- Statement to prove
theorem num_A_is_9 : num_A = 9 :=
by
  sorry

end num_A_is_9_l119_119445


namespace min_value_c_l119_119507

-- Define the problem using Lean
theorem min_value_c 
    (a b c d e : ℕ)
    (h1 : a + 1 = b) 
    (h2 : b + 1 = c)
    (h3 : c + 1 = d)
    (h4 : d + 1 = e)
    (h5 : ∃ n : ℕ, 5 * c = n ^ 3)
    (h6 : ∃ m : ℕ, 3 * c = m ^ 2) : 
    c = 675 := 
sorry

end min_value_c_l119_119507


namespace circle_radius_l119_119662

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l119_119662


namespace uniquely_identify_figure_l119_119015

structure Figure where
  is_curve : Bool
  has_axis_of_symmetry : Bool
  has_center_of_symmetry : Bool

def Circle : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Ellipse : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := false }
def Triangle : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }
def Square : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Rectangle : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Parallelogram : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := true }
def Trapezoid : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }

theorem uniquely_identify_figure (figures : List Figure) (q1 q2 q3 : Figure → Bool) :
  ∀ (f : Figure), ∃! (f' : Figure), 
    q1 f' = q1 f ∧ q2 f' = q2 f ∧ q3 f' = q3 f :=
by
  sorry

end uniquely_identify_figure_l119_119015


namespace local_minimum_f_eval_integral_part_f_l119_119223

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sin x * Real.sqrt (1 - Real.cos x))

theorem local_minimum_f :
  (0 < x) -> (x < π) -> f x >= 1 :=
  by sorry

theorem eval_integral_part_f :
  ∫ x in (↑(π / 2))..(↑(2 * π / 3)), f x = sorry :=
  by sorry

end local_minimum_f_eval_integral_part_f_l119_119223


namespace sandwich_cost_l119_119511

theorem sandwich_cost (S : ℝ) (h : 2 * S + 4 * 0.87 = 8.36) : S = 2.44 :=
by sorry

end sandwich_cost_l119_119511


namespace parallel_lines_implies_m_neg1_l119_119957

theorem parallel_lines_implies_m_neg1 (m : ℝ) :
  (∀ (x y : ℝ), x + m * y + 6 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + 3 * y + 2 * m = 0) ∧
  ∀ (l₁ l₂ : ℝ), l₁ = -(1 / m) ∧ l₂ = -((m - 2) / 3) ∧ l₁ = l₂ → m = -1 :=
by
  sorry

end parallel_lines_implies_m_neg1_l119_119957


namespace sum_of_coordinates_l119_119062

theorem sum_of_coordinates (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 20) : x + y = 2 :=
sorry

end sum_of_coordinates_l119_119062


namespace cubic_has_three_roots_l119_119195

noncomputable def solve_cubic (a b c α : ℚ) (x : ℚ) : Prop :=
  let poly := (Polynomial.X^3 + a * Polynomial.X^2 + b * Polynomial.X + c) in
  Polynomial.eval α poly = 0 ∧
  ∃ (q : Polynomial ℚ),
    poly = Polynomial.X - Polynomial.C α * q ∧
    q.degree = Polynomial.natDegree q - 1 ∧
    ∃ (roots : List ℚ), Polynomial.roots q = roots 

theorem cubic_has_three_roots (a b c α : ℚ) (x : ℚ) : solve_cubic a b c α x :=
by
  sorry

end cubic_has_three_roots_l119_119195


namespace some_number_value_correct_l119_119973

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l119_119973


namespace roots_sum_of_squares_l119_119816

theorem roots_sum_of_squares {r s : ℝ} (h : Polynomial.roots (X^2 - 3*X + 1) = {r, s}) : r^2 + s^2 = 7 :=
by
  sorry

end roots_sum_of_squares_l119_119816


namespace cos_5_theta_l119_119199

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end cos_5_theta_l119_119199


namespace prove_problem_statement_l119_119638

noncomputable def problem_statement : Prop :=
  let E := (0, 0)
  let F := (2, 4)
  let G := (6, 2)
  let H := (7, 0)
  let line_through_E x y := y = -2 * x + 14
  let intersection_x := 37 / 8
  let intersection_y := 19 / 4
  let intersection_point := (intersection_x, intersection_y)
  let u := 37
  let v := 8
  let w := 19
  let z := 4
  u + v + w + z = 68

theorem prove_problem_statement : problem_statement :=
  sorry

end prove_problem_statement_l119_119638


namespace machine_present_value_l119_119897

theorem machine_present_value
  (rate_of_decay : ℝ) (n_periods : ℕ) (final_value : ℝ) (initial_value : ℝ)
  (h_decay : rate_of_decay = 0.25)
  (h_periods : n_periods = 2)
  (h_final_value : final_value = 225) :
  initial_value = 400 :=
by
  -- The proof would go here. 
  sorry

end machine_present_value_l119_119897


namespace set_intersection_l119_119770

-- Definitions of sets M and N
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

-- The statement to prove that M ∩ N = {1, 2}
theorem set_intersection :
  M ∩ N = {1, 2} := by
  sorry

end set_intersection_l119_119770


namespace probability_of_selecting_cooking_l119_119298

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119298


namespace proof_problem_l119_119141

variable (A B C : ℕ)

-- Defining the conditions
def condition1 : Prop := A + B + C = 700
def condition2 : Prop := B + C = 600
def condition3 : Prop := C = 200

-- Stating the proof problem
theorem proof_problem (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 C) : A + C = 300 :=
sorry

end proof_problem_l119_119141


namespace no_integer_regular_pentagon_l119_119794

theorem no_integer_regular_pentagon 
  (x y : Fin 5 → ℤ) 
  (h_length : ∀ i j : Fin 5, i ≠ j → (x i - x j) ^ 2 + (y i - y j) ^ 2 = (x 0 - x 1) ^ 2 + (y 0 - y 1) ^ 2)
  : False :=
sorry

end no_integer_regular_pentagon_l119_119794


namespace det_of_A_squared_minus_3A_l119_119956

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 4], ![3, 2]]

theorem det_of_A_squared_minus_3A : det (A * A - 3 • A) = 140 := by
  sorry

end det_of_A_squared_minus_3A_l119_119956


namespace pythagorean_relationship_l119_119138

theorem pythagorean_relationship (a b c : ℝ) (h : c^2 = a^2 + b^2) : c^2 = a^2 + b^2 :=
by
  sorry

end pythagorean_relationship_l119_119138


namespace Veenapaniville_high_schools_l119_119039

theorem Veenapaniville_high_schools :
  ∃ (districtA districtB districtC : ℕ),
    districtA + districtB + districtC = 50 ∧
    (districtA + districtB + districtC = 50) ∧
    (∃ (publicB parochialB privateB : ℕ), 
      publicB + parochialB + privateB = 17 ∧ privateB = 2) ∧
    (∃ (publicC parochialC privateC : ℕ),
      publicC = 9 ∧ parochialC = 9 ∧ privateC = 9 ∧ publicC + parochialC + privateC = 27) ∧
    districtB = 17 ∧
    districtC = 27 →
    districtA = 6 := by
  sorry

end Veenapaniville_high_schools_l119_119039


namespace pages_per_day_l119_119131

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 63) (h2 : days = 3) : total_pages / days = 21 :=
by
  sorry

end pages_per_day_l119_119131


namespace probability_of_selecting_cooking_l119_119294

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119294


namespace train_length_correct_l119_119390

noncomputable def train_length (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_correct :
  train_length 2.49980001599872 144 = 99.9920006399488 :=
by
  dsimp [train_length]
  -- Convert 144 km/h to m/s => 40 m/s
  have speed_ms : ℝ := 144 * (1000 / 3600)
  norm_num [speed_ms]
  -- Compute distance: 40 m/s * 2.49980001599872 s => 99.9920006399488 m
  have distance : ℝ := speed_ms * 2.49980001599872
  norm_num [distance]
  -- Assert and verify the final result
  norm_num
  exact sorry

end train_length_correct_l119_119390


namespace bucket_full_weight_l119_119269

variable (p q r : ℚ)
variable (x y : ℚ)

-- Define the conditions
def condition1 : Prop := p = r + (3 / 4) * y
def condition2 : Prop := q = r + (1 / 3) * y
def condition3 : Prop := x = r

-- Define the conclusion
def conclusion : Prop := x + y = (4 * p - r) / 3

-- The theorem stating that the conclusion follows from the conditions
theorem bucket_full_weight (h1 : condition1 p r y) (h2 : condition2 q r y) (h3 : condition3 x r) : conclusion x y p r :=
by
  sorry

end bucket_full_weight_l119_119269


namespace total_handshakes_l119_119133

theorem total_handshakes (total_people : ℕ) (first_meeting_people : ℕ) (second_meeting_new_people : ℕ) (common_people : ℕ)
  (total_people_is : total_people = 12)
  (first_meeting_people_is : first_meeting_people = 7)
  (second_meeting_new_people_is : second_meeting_new_people = 5)
  (common_people_is : common_people = 2)
  (first_meeting_handshakes : ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2)
  (second_meeting_handshakes: ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2 - (common_people * (common_people - 1)) / 2):
  first_meeting_handshakes + second_meeting_handshakes = 41 := 
sorry

end total_handshakes_l119_119133


namespace draw_at_least_one_red_card_l119_119389

-- Define the deck and properties
def total_cards := 52
def red_cards := 26
def black_cards := 26

-- Define the calculation for drawing three cards sequentially
def total_ways_draw3 := total_cards * (total_cards - 1) * (total_cards - 2)
def black_only_ways_draw3 := black_cards * (black_cards - 1) * (black_cards - 2)

-- Define the main proof statement
theorem draw_at_least_one_red_card : 
    total_ways_draw3 - black_only_ways_draw3 = 117000 := by
    -- Proof is omitted
    sorry

end draw_at_least_one_red_card_l119_119389


namespace solve_system_l119_119856

def system_of_equations_solution : Prop :=
  ∃ (x y : ℚ), 4 * x - 7 * y = -9 ∧ 5 * x + 3 * y = -11 ∧ (x, y) = (-(104 : ℚ) / 47, (1 : ℚ) / 47)

theorem solve_system : system_of_equations_solution :=
sorry

end solve_system_l119_119856


namespace barrels_oil_total_l119_119143

theorem barrels_oil_total :
  let A := 3 / 4
  let B := A + 1 / 10
  A + B = 8 / 5 := by
  sorry

end barrels_oil_total_l119_119143


namespace average_of_175_results_l119_119608

theorem average_of_175_results (x y : ℕ) (hx : x = 100) (hy : y = 75) 
(a b : ℚ) (ha : a = 45) (hb : b = 65) :
  ((x * a + y * b) / (x + y) = 53.57) :=
sorry

end average_of_175_results_l119_119608


namespace probability_of_selecting_cooking_l119_119288

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119288


namespace art_club_artworks_l119_119678

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l119_119678


namespace no_solution_inequality_l119_119961

theorem no_solution_inequality (a b x : ℝ) (h : |a - b| > 2) : ¬(|x - a| + |x - b| ≤ 2) :=
sorry

end no_solution_inequality_l119_119961


namespace minimize_distance_l119_119070

-- Definitions of points and distances
structure Point where
  x : ℝ
  y : ℝ

def distanceSquared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition points A, B, and C
def A := Point.mk 7 3
def B := Point.mk 3 0

-- Mathematical problem: Find the value of k that minimizes the sum of distances squared
theorem minimize_distance : ∃ k : ℝ, ∀ k', 
  (distanceSquared A (Point.mk 0 k) + distanceSquared B (Point.mk 0 k) ≤ 
   distanceSquared A (Point.mk 0 k') + distanceSquared B (Point.mk 0 k')) → 
  k = 3 / 2 :=
by
  sorry

end minimize_distance_l119_119070


namespace solve_fractional_equation_l119_119253

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 0) : (x + 1) / x = 2 / 3 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l119_119253


namespace candy_cost_l119_119465

theorem candy_cost (J H C : ℕ) (h1 : J + 7 = C) (h2 : H + 1 = C) (h3 : J + H < C) : C = 7 :=
by
  sorry

end candy_cost_l119_119465


namespace oreo_shop_ways_l119_119920

theorem oreo_shop_ways (α β : ℕ) (products total_ways : ℕ) :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  (α + β = products) ∧ (products = 4) ∧ (total_ways = 2143) ∧ 
  (α ≤ 2 * total_flavors) ∧ (β ≤ 4 * oreo_flavors) →
  total_ways = 2143 :=
by sorry


end oreo_shop_ways_l119_119920


namespace max_f_l119_119490

open Real

noncomputable def f (x y z : ℝ) := (1 - y * z + z) * (1 - z * x + x) * (1 - x * y + y)

theorem max_f (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  f x y z ≤ 1 ∧ (x = 1 ∧ y = 1 ∧ z = 1 → f x y z = 1) := sorry

end max_f_l119_119490


namespace solution_set_of_inequality_l119_119873

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 2) * (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l119_119873


namespace initial_investors_and_contribution_l119_119724

theorem initial_investors_and_contribution :
  ∃ (x y : ℕ), 
    (x - 10) * (y + 1) = x * y ∧
    (x - 25) * (y + 3) = x * y ∧
    x = 100 ∧ 
    y = 9 :=
by
  sorry

end initial_investors_and_contribution_l119_119724


namespace mean_of_numbers_is_10_l119_119243

-- Define the list of numbers
def numbers : List ℕ := [6, 8, 9, 11, 16]

-- Define the length of the list
def n : ℕ := numbers.length

-- Define the sum of the list
def sum_numbers : ℕ := numbers.sum

-- Define the mean (average) calculation for the list
def average : ℕ := sum_numbers / n

-- Prove that the mean of the list is 10
theorem mean_of_numbers_is_10 : average = 10 := by
  sorry

end mean_of_numbers_is_10_l119_119243


namespace Ivan_can_safely_make_the_journey_l119_119260

def eruption_cycle_first_crater (t : ℕ) : Prop :=
  ∃ n : ℕ, t = 1 + 18 * n

def eruption_cycle_second_crater (t : ℕ) : Prop :=
  ∃ m : ℕ, t = 1 + 10 * m

def is_safe (start_time : ℕ) : Prop :=
  ∀ t, start_time ≤ t ∧ t < start_time + 16 → 
    ¬ eruption_cycle_first_crater t ∧ 
    ¬ (t ≥ start_time + 12 ∧ eruption_cycle_second_crater t)

theorem Ivan_can_safely_make_the_journey : ∃ t : ℕ, is_safe (38 + t) :=
sorry

end Ivan_can_safely_make_the_journey_l119_119260


namespace pi_sub_alpha_in_first_quadrant_l119_119020

theorem pi_sub_alpha_in_first_quadrant (α : ℝ) (h : π / 2 < α ∧ α < π) : 0 < π - α ∧ π - α < π / 2 :=
by
  sorry

end pi_sub_alpha_in_first_quadrant_l119_119020


namespace range_of_a_l119_119437

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 - x + (a - 4) = 0 ∧ y^2 - y + (a - 4) = 0 ∧ x > 0 ∧ y < 0) → a < 4 :=
by
  sorry

end range_of_a_l119_119437


namespace elder_age_is_twenty_l119_119491

-- Let e be the present age of the elder person
-- Let y be the present age of the younger person

def ages_diff_by_twelve (e y : ℕ) : Prop :=
  e = y + 12

def elder_five_years_ago (e y : ℕ) : Prop :=
  e - 5 = 5 * (y - 5)

theorem elder_age_is_twenty (e y : ℕ) (h1 : ages_diff_by_twelve e y) (h2 : elder_five_years_ago e y) :
  e = 20 :=
by
  sorry

end elder_age_is_twenty_l119_119491


namespace ajith_rana_meet_l119_119277

/--
Ajith and Rana walk around a circular course 115 km in circumference, starting together from the same point.
Ajith walks at 4 km/h, and Rana walks at 5 km/h in the same direction.
Prove that they will meet after 115 hours.
-/
theorem ajith_rana_meet 
  (course_circumference : ℕ)
  (ajith_speed : ℕ)
  (rana_speed : ℕ)
  (relative_speed : ℕ)
  (time : ℕ)
  (start_point : Point)
  (ajith : Person)
  (rana : Person)
  (walk_in_same_direction : Prop)
  (start_time : ℕ)
  (meet_time : ℕ) :
  course_circumference = 115 →
  ajith_speed = 4 →
  rana_speed = 5 →
  relative_speed = rana_speed - ajith_speed →
  time = course_circumference / relative_speed →
  meet_time = start_time + time →
  meet_time = 115 :=
by
  sorry

end ajith_rana_meet_l119_119277


namespace heartsuit_symmetric_solution_l119_119555

def heartsuit (a b : ℝ) : ℝ :=
  a^3 * b - a^2 * b^2 + a * b^3

theorem heartsuit_symmetric_solution :
  ∀ x y : ℝ, (heartsuit x y = heartsuit y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end heartsuit_symmetric_solution_l119_119555


namespace total_payment_correct_l119_119400

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end total_payment_correct_l119_119400


namespace abs_value_expression_l119_119962

theorem abs_value_expression (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 :=
sorry

end abs_value_expression_l119_119962


namespace repeating_decimals_sum_l119_119582

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l119_119582


namespace problem1_problem2_l119_119194

-- Definitions of the sets A and B based on the given conditions
def A : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - a) * (x - 3 * a) < 0 }

-- Proof statement for problem (1)
theorem problem1 (a : ℝ) : (∀ x, x ∈ A → x ∈ (B a)) ↔ (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

-- Proof statement for problem (2)
theorem problem2 (a : ℝ) : (∀ x, (x ∈ A ∧ x ∈ (B a)) ↔ (3 < x ∧ x < 4)) ↔ (a = 3) := by
  sorry

end problem1_problem2_l119_119194


namespace multiply_expression_l119_119832

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l119_119832


namespace three_star_five_l119_119754

-- Definitions based on conditions
def star (a b : ℕ) : ℕ := 2 * a^2 + 3 * a * b + 2 * b^2

-- Theorem statement to be proved
theorem three_star_five : star 3 5 = 113 := by
  sorry

end three_star_five_l119_119754


namespace probability_of_cooking_l119_119370

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119370


namespace extreme_values_l119_119617

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4 * x + 6

theorem extreme_values :
  (∃ x : ℝ, f x = 34/3 ∧ (x = -2 ∨ x = 4)) ∧
  (∃ x : ℝ, f x = 2/3 ∧ x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 4, f x ≤ 34/3 ∧ 2/3 ≤ f x) :=
by
  sorry

end extreme_values_l119_119617


namespace angle_EMF_90_l119_119630

open EuclideanGeometry

-- Define the triangle and points on it.
variable (A B C M E F : Point)

-- Define the conditions mathematically
def B_angle_120 (t : Triangle) : Prop :=
  t.angles B = 120

def midpoint_M (t : Triangle) (M : Point) : Prop :=
  M = midpoint t.A t.C

def E_and_F_conditions (t : Triangle) (E F : Point) : Prop :=
  dist t.A E = dist E F ∧ dist E F = dist F t.C

-- Main goal to prove
theorem angle_EMF_90 (t : Triangle)
  (hB : B_angle_120 t)
  (hM_mid : midpoint_M t M)
  (hE_F : E_and_F_conditions t E F) :
  angle E M F = 90 :=
by
  sorry

end angle_EMF_90_l119_119630


namespace fraction_of_percent_l119_119520

theorem fraction_of_percent (h : (1 / 8 * (1 / 100)) * 800 = 1) : true :=
by
  trivial

end fraction_of_percent_l119_119520


namespace gcd_18_n_eq_6_l119_119170

theorem gcd_18_n_eq_6 (num_valid_n : Nat) :
  (num_valid_n = (List.range 200).count (λ n, (1 ≤ n ∧ n ≤ 200) ∧ (6 ∣ n) ∧ ¬(9 ∣ n))) →
  num_valid_n = 22 := by
  sorry

end gcd_18_n_eq_6_l119_119170


namespace part_a_part_b_part_c_part_d_l119_119063

-- Part a
theorem part_a (x : ℝ) : 
  (5 / x - x / 3 = 1 / 6) ↔ x = 6 := 
by
  sorry

-- Part b
theorem part_b (a : ℝ) : 
  ¬ ∃ a, (1 / 2 + a / 4 = a / 4) := 
by
  sorry

-- Part c
theorem part_c (y : ℝ) : 
  (9 / y - y / 21 = 17 / 21) ↔ y = 7 := 
by
  sorry

-- Part d
theorem part_d (z : ℝ) : 
  (z / 8 - 1 / z = 3 / 8) ↔ z = 4 := 
by
  sorry

end part_a_part_b_part_c_part_d_l119_119063


namespace angle_between_clock_hands_at_7_oclock_l119_119698

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l119_119698


namespace largest_square_side_length_l119_119757

theorem largest_square_side_length (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) : 
  ∃ x : ℝ, x = (a * b) / (a + b) := 
sorry

end largest_square_side_length_l119_119757


namespace area_enclosed_by_circle_l119_119118

theorem area_enclosed_by_circle : 
  (∀ x y : ℝ, x^2 + y^2 + 10 * x + 24 * y = 0) → 
  (π * 13^2 = 169 * π):=
by
  intro h
  sorry

end area_enclosed_by_circle_l119_119118


namespace exists_unique_c_l119_119425

theorem exists_unique_c (a : ℝ) (h₁ : 1 < a) :
  (∃ (c : ℝ), ∀ (x : ℝ), x ∈ Set.Icc a (2 * a) → ∃ (y : ℝ), y ∈ Set.Icc a (a ^ 2) ∧ (Real.log x / Real.log a + Real.log y / Real.log a = c)) ↔ a = 2 :=
by
  sorry

end exists_unique_c_l119_119425


namespace some_number_value_correct_l119_119975

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l119_119975


namespace raisin_addition_l119_119631

theorem raisin_addition : 
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  yellow_raisins + black_raisins = 0.7 := 
by
  sorry

end raisin_addition_l119_119631


namespace rowing_speed_downstream_l119_119898

/--
A man can row upstream at 25 kmph and downstream at a certain speed. 
The speed of the man in still water is 30 kmph. 
Prove that the speed of the man rowing downstream is 35 kmph.
-/
theorem rowing_speed_downstream (V_u V_sw V_s V_d : ℝ)
  (h1 : V_u = 25) 
  (h2 : V_sw = 30) 
  (h3 : V_u = V_sw - V_s) 
  (h4 : V_d = V_sw + V_s) :
  V_d = 35 :=
by
  sorry

end rowing_speed_downstream_l119_119898


namespace clock_angle_at_7_oclock_l119_119701

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l119_119701


namespace radius_of_circle_l119_119656

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l119_119656


namespace max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l119_119003

-- Problem (1)
theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x*y ≤ 4 :=
sorry

-- Additional statement to show when the maximum is achieved
theorem max_xy_is_4 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x = 4 ∧ y = 1 ↔ x*y = 4 :=
sorry

-- Problem (2)
theorem min_x_plus_y (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x + y ≥ 9 :=
sorry

-- Additional statement to show when the minimum is achieved
theorem min_x_plus_y_is_9 (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x = 6 ∧ y = 3 ↔ x + y = 9 :=
sorry

end max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l119_119003


namespace ratio_sum_of_arithmetic_sequences_l119_119145

-- Definitions for the arithmetic sequences
def a_num := 3
def d_num := 3
def l_num := 99

def a_den := 4
def d_den := 4
def l_den := 96

-- Number of terms in each sequence
def n_num := (l_num - a_num) / d_num + 1
def n_den := (l_den - a_den) / d_den + 1

-- Sum of the sequences using the sum formula for arithmetic series
def S_num := n_num * (a_num + l_num) / 2
def S_den := n_den * (a_den + l_den) / 2

-- The theorem statement
theorem ratio_sum_of_arithmetic_sequences : S_num / S_den = 1683 / 1200 := by sorry

end ratio_sum_of_arithmetic_sequences_l119_119145


namespace incorrect_statement_D_l119_119751

theorem incorrect_statement_D (x1 x2 : ℝ) (hx : x1 < x2) :
  ¬ (y : ℝ) (y1 := -3 / x1) (y2 := -3 / x2) y1 < y2 :=
by
  sorry

end incorrect_statement_D_l119_119751


namespace find_rate_percent_l119_119130

theorem find_rate_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : 2420 = P * (1 + r / 100)^2) 
  (h2 : 3025 = P * (1 + r / 100)^3) : 
  r = 25 :=
by
  sorry

end find_rate_percent_l119_119130


namespace price_per_box_l119_119262

theorem price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : 
  total_apples = 10000 → apples_per_box = 50 → total_revenue = 7000 → 
  total_revenue / (total_apples / apples_per_box) = 35 :=
by
  intros h1 h2 h3
  -- we can skip the actual proof with sorry. This indicates that the proof is not provided,
  -- but the statement is what needs to be proven.
  sorry

end price_per_box_l119_119262


namespace find_coefficients_l119_119923

variables {x1 x2 x3 x4 x5 x6 x7 : ℝ}

theorem find_coefficients
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 14)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 30)
  (h4 : 16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 70) :
  25*x1 + 36*x2 + 49*x3 + 64*x4 + 81*x5 + 100*x6 + 121*x7 = 130 :=
sorry

end find_coefficients_l119_119923


namespace find_a_plus_b_find_range_of_c_l119_119874

-- For question (I)
theorem find_a_plus_b (a b : ℝ) (h_eq1 : a = 2 + 3) (h_eq2 : b = 2 * 3) : a + b = 11 :=
  sorry

-- For question (II)
theorem find_range_of_c (b c : ℝ) (h_b : b = 6) (h_empty : ∀ x, ¬(-x^2 + b * x + c > 0)) : c ≤ -9 :=
  sorry

end find_a_plus_b_find_range_of_c_l119_119874


namespace find_some_number_l119_119967

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l119_119967


namespace product_equivalence_l119_119023

theorem product_equivalence 
  (a b c d e f : ℝ) 
  (h1 : a + b + c + d + e + f = 0) 
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) : 
  (a + c) * (a + d) * (a + e) * (a + f) = (b + c) * (b + d) * (b + e) * (b + f) :=
by
  sorry

end product_equivalence_l119_119023


namespace men_entered_l119_119988

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l119_119988


namespace probability_cooking_selected_l119_119315

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119315


namespace total_profit_is_18900_l119_119912

-- Defining the conditions
variable (x : ℕ)  -- A's initial investment
variable (A_share : ℕ := 6300)  -- A's share in rupees

-- Total profit calculation
def total_annual_gain : ℕ :=
  (x * 12) + (2 * x * 6) + (3 * x * 4)

-- The main statement
theorem total_profit_is_18900 (x : ℕ) (A_share : ℕ := 6300) :
  3 * A_share = total_annual_gain x :=
by sorry

end total_profit_is_18900_l119_119912


namespace least_three_digit_with_factors_l119_119267

theorem least_three_digit_with_factors (n : ℕ) :
  (n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n) → n = 210 := by
  sorry

end least_three_digit_with_factors_l119_119267


namespace cubic_meters_to_cubic_feet_l119_119017

theorem cubic_meters_to_cubic_feet :
  (let feet_per_meter := 3.28084
  in (feet_per_meter ^ 3) * 2 = 70.6294) :=
by
  sorry

end cubic_meters_to_cubic_feet_l119_119017


namespace perfect_square_trinomial_k_l119_119776

theorem perfect_square_trinomial_k (a k : ℝ) : (∃ b : ℝ, (a - b)^2 = a^2 - ka + 25) ↔ k = 10 ∨ k = -10 := 
sorry

end perfect_square_trinomial_k_l119_119776


namespace vector_expression_result_l119_119013

structure Vector2 :=
(x : ℝ)
(y : ℝ)

def vector_dot_product (v1 v2 : Vector2) : ℝ :=
  v1.x * v1.y + v2.x * v2.y

def vector_scalar_mul (c : ℝ) (v : Vector2) : Vector2 :=
  { x := c * v.x, y := c * v.y }

def vector_sub (v1 v2 : Vector2) : Vector2 :=
  { x := v1.x - v2.x, y := v1.y - v2.y }

noncomputable def a : Vector2 := { x := 2, y := -1 }
noncomputable def b : Vector2 := { x := 3, y := -2 }

theorem vector_expression_result :
  vector_dot_product
    (vector_sub (vector_scalar_mul 3 a) b)
    (vector_sub a (vector_scalar_mul 2 b)) = -15 := by
  sorry

end vector_expression_result_l119_119013


namespace jane_change_l119_119220

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l119_119220


namespace crow_speed_l119_119726

/-- Definitions from conditions -/
def distance_between_nest_and_ditch : ℝ := 250 -- in meters
def total_trips : ℕ := 15
def total_hours : ℝ := 1.5 -- hours

/-- The statement to be proved -/
theorem crow_speed :
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000 -- convert to kilometers
  let speed := total_distance / total_hours
  speed = 5 := by
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000
  let speed := total_distance / total_hours
  sorry

end crow_speed_l119_119726


namespace maximum_cards_l119_119796

def total_budget : ℝ := 15
def card_cost : ℝ := 1.25
def transaction_fee : ℝ := 2
def desired_savings : ℝ := 3

theorem maximum_cards : ∃ n : ℕ, n ≤ 8 ∧ (card_cost * (n : ℝ) + transaction_fee ≤ total_budget - desired_savings) :=
by sorry

end maximum_cards_l119_119796


namespace multiply_expand_l119_119836

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l119_119836


namespace cos_tan_quadrant_l119_119959

theorem cos_tan_quadrant (α : ℝ) 
  (hcos : Real.cos α < 0) 
  (htan : Real.tan α > 0) : 
  (2 * π / 2 < α ∧ α < π) :=
by
  sorry

end cos_tan_quadrant_l119_119959


namespace complex_solution_l119_119780

theorem complex_solution (x : ℂ) (h : x^2 + 1 = 0) : x = Complex.I ∨ x = -Complex.I :=
by sorry

end complex_solution_l119_119780


namespace find_wsquared_l119_119963

theorem find_wsquared : 
  (2 * w + 10) ^ 2 = (5 * w + 15) * (w + 6) →
  w ^ 2 = (90 + 10 * Real.sqrt 65) / 4 := 
by 
  intro h₀
  sorry

end find_wsquared_l119_119963


namespace right_triangle_ratio_l119_119032

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def radius_circumscribed_circle (hypo : ℝ) : ℝ := hypo / 2

noncomputable def radius_inscribed_circle (a b hypo : ℝ) : ℝ := (a + b - hypo) / 2

theorem right_triangle_ratio 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 8) 
  (right_angle : a^2 + b^2 = hypotenuse a b ^ 2) :
  radius_inscribed_circle a b (hypotenuse a b) / radius_circumscribed_circle (hypotenuse a b) = 2 / 5 :=
by {
  -- The proof to be filled in
  sorry
}

end right_triangle_ratio_l119_119032


namespace probability_of_selecting_cooking_l119_119345

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119345


namespace jane_change_l119_119219

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l119_119219


namespace boss_contribution_l119_119641

variable (boss_contrib : ℕ) (todd_contrib : ℕ) (employees_contrib : ℕ)
variable (cost : ℕ) (n_employees : ℕ) (emp_payment : ℕ)
variable (total_payment : ℕ)

-- Conditions
def birthday_gift_conditions :=
  cost = 100 ∧
  todd_contrib = 2 * boss_contrib ∧
  employees_contrib = n_employees * emp_payment ∧
  n_employees = 5 ∧
  emp_payment = 11 ∧
  total_payment = boss_contrib + todd_contrib + employees_contrib

-- The proof goal
theorem boss_contribution
  (h : birthday_gift_conditions boss_contrib todd_contrib employees_contrib cost n_employees emp_payment total_payment) :
  boss_contrib = 15 :=
by
  sorry

end boss_contribution_l119_119641


namespace minimum_value_f_condition_f_geq_zero_l119_119952

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem minimum_value_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ f (Real.log a) a) ∧ f (Real.log a) a = a - a * Real.log a - 1 :=
by 
  sorry

theorem condition_f_geq_zero (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ a = 1 :=
by 
  sorry

end minimum_value_f_condition_f_geq_zero_l119_119952


namespace problem1_problem2_l119_119211

-- Definitions for the problem

/-- Definition of point P in Cartesian coordinate system -/
def P (x : ℝ) : ℝ × ℝ :=
  (x - 2, x)

-- First proof problem statement
theorem problem1 (x : ℝ) (h : (x - 2) * x < 0) : x = 1 :=
sorry

-- Second proof problem statement
theorem problem2 (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end problem1_problem2_l119_119211


namespace sequence_a1_l119_119230

variable (S : ℕ → ℤ) (a : ℕ → ℤ)

def Sn_formula (n : ℕ) (a₁ : ℤ) : ℤ := (a₁ * (4^n - 1)) / 3

theorem sequence_a1 (h1 : ∀ n : ℕ, S n = Sn_formula n (a 1))
                    (h2 : a 4 = 32) :
  a 1 = 1 / 2 :=
by
  sorry

end sequence_a1_l119_119230


namespace polynomial_divisibility_n_l119_119741

theorem polynomial_divisibility_n :
  ∀ (n : ℤ), (∀ x, x = 2 → 3 * x^2 - 4 * x + n = 0) → n = -4 :=
by
  intros n h
  have h2 : 3 * 2^2 - 4 * 2 + n = 0 := h 2 rfl
  linarith

end polynomial_divisibility_n_l119_119741


namespace repeating_decimal_sum_l119_119568

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l119_119568


namespace determine_d_iff_l119_119934

theorem determine_d_iff (x : ℝ) : 
  (x ∈ Set.Ioo (-5/2) 3) ↔ (x * (2 * x + 3) < 15) :=
by
  sorry

end determine_d_iff_l119_119934


namespace student_B_speed_l119_119907

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end student_B_speed_l119_119907


namespace repeating_decimal_sum_l119_119597

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l119_119597


namespace probability_of_selecting_cooking_l119_119297

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119297


namespace equation_solution_l119_119240

theorem equation_solution (x : ℝ) : 
  (x - 3)^4 = 16 → x = 5 :=
by
  sorry

end equation_solution_l119_119240


namespace math_problem_proof_l119_119806

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end math_problem_proof_l119_119806


namespace sum_of_repeating_decimals_l119_119588

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l119_119588


namespace sequence_term_500_l119_119453

theorem sequence_term_500 (a : ℕ → ℤ) (h1 : a 1 = 3009) (h2 : a 2 = 3010) 
  (h3 : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 500 = 3341 := 
sorry

end sequence_term_500_l119_119453


namespace part1_probability_part2_distribution_expectation_l119_119208

theorem part1_probability :
  ((P (C after B) = 3/4) ∧ (P (A after C) = 2/5))
  → (P (A on third) = 3/10) := by
  sorry

theorem part2_distribution_expectation :
  ((P (B after A) = 1/3) ∧ (P (C after A) = 2/3) ∧
   (P (A after B) = 1/4) ∧ (P (C after B) = 3/4) ∧
   (P (A after C) = 2/5) ∧ (P (B after C) = 3/5))
  → (P (X = 1) = 13/20) ∧ (P (X = 2) = 7/20) ∧ (E(X) = 27/20) := by
  sorry

end part1_probability_part2_distribution_expectation_l119_119208


namespace parallelogram_area_l119_119746

theorem parallelogram_area (base height : ℝ) (h_base : base = 12) (h_height : height = 10) :
  base * height = 120 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l119_119746


namespace jane_received_change_l119_119213

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l119_119213


namespace problem_f_of_3_l119_119431

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else -2 * x + 3

theorem problem_f_of_3 : f (f 3) = 10 := by
  sorry

end problem_f_of_3_l119_119431


namespace multiply_polynomials_l119_119825

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l119_119825


namespace value_of_some_number_l119_119968

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l119_119968


namespace find_number_l119_119283

theorem find_number (x : ℝ) (h : (3 / 4) * (1 / 2) * (2 / 5) * x = 753.0000000000001) : 
  x = 5020.000000000001 :=
by 
  sorry

end find_number_l119_119283


namespace bottle_capacity_l119_119716

theorem bottle_capacity
  (num_boxes : ℕ)
  (bottles_per_box : ℕ)
  (fill_fraction : ℚ)
  (total_volume : ℚ)
  (total_bottles : ℕ)
  (filled_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_fraction = 3 / 4 →
  total_volume = 4500 →
  total_bottles = num_boxes * bottles_per_box →
  filled_volume = (total_bottles : ℚ) * (fill_fraction * (12 : ℚ)) →
  12 = 4500 / (total_bottles * fill_fraction) := 
by 
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end bottle_capacity_l119_119716


namespace sum_of_repeating_decimals_l119_119586

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l119_119586


namespace opposite_neg_one_half_l119_119098

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l119_119098


namespace canal_depth_l119_119667

theorem canal_depth (A : ℝ) (w_top w_bottom : ℝ) (h : ℝ) 
    (hA : A = 10290) 
    (htop : w_top = 6) 
    (hbottom : w_bottom = 4) 
    (harea : A = 1 / 2 * (w_top + w_bottom) * h) : 
    h = 2058 :=
by
  -- here goes the proof steps
  sorry

end canal_depth_l119_119667


namespace price_reduction_percentage_price_increase_amount_l119_119541

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l119_119541


namespace milan_billed_minutes_l119_119167

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l119_119167


namespace sum_of_roots_l119_119706

theorem sum_of_roots (a b c: ℝ) (h: a ≠ 0) (h_eq : a = 1) (h_eq2 : b = -6) (h_eq3 : c = 8):
    let Δ := b ^ 2 - 4 * a * c in
    let root1 := (-b + real.sqrt Δ) / (2 * a) in
    let root2 := (-b - real.sqrt Δ) / (2 * a) in
    root1 + root2 = 6 :=
by
  sorry

end sum_of_roots_l119_119706


namespace translated_line_value_m_l119_119783

theorem translated_line_value_m :
  (∀ x y : ℝ, (y = x → y = x + 3) → y = 2 + 3 → ∃ m : ℝ, y = m) :=
by sorry

end translated_line_value_m_l119_119783


namespace find_some_number_l119_119966

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l119_119966


namespace Dan_picked_9_plums_l119_119050

-- Define the constants based on the problem
def M : ℕ := 4 -- Melanie's plums
def S : ℕ := 3 -- Sally's plums
def T : ℕ := 16 -- Total plums picked

-- The number of plums Dan picked
def D : ℕ := T - (M + S)

-- The theorem we want to prove
theorem Dan_picked_9_plums : D = 9 := by
  sorry

end Dan_picked_9_plums_l119_119050


namespace find_f_2014_l119_119423

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f x = f (x + 1) - f (x + 2)
axiom interval_def : ∀ x, 0 < x ∧ x < 3 → f x = x^2

theorem find_f_2014 : f 2014 = -1 := sorry

end find_f_2014_l119_119423


namespace probability_selecting_cooking_l119_119305

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119305


namespace required_speed_l119_119128

-- The car covers 504 km in 6 hours initially.
def distance : ℕ := 504
def initial_time : ℕ := 6
def initial_speed : ℕ := distance / initial_time

-- The time that is 3/2 times the initial time.
def factor : ℚ := 3 / 2
def new_time : ℚ := initial_time * factor

-- The speed required to cover the same distance in the new time.
def new_speed : ℚ := distance / new_time

-- The proof statement
theorem required_speed : new_speed = 56 := by
  sorry

end required_speed_l119_119128


namespace trig_identity_l119_119877

theorem trig_identity : 
  (2 * Real.sin (80 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l119_119877


namespace log_relationship_l119_119182

theorem log_relationship (a b c: ℝ) (ha : a = Real.log 3 / Real.log 2)
  (hb : b = Real.log 4 / Real.log 3) (hc : c = Real.log 11 / (2 * Real.log 2)) :
  b < a ∧ a < c :=
by
  sorry

end log_relationship_l119_119182


namespace hypotenuse_length_l119_119787

variables (a b c : ℝ)

-- Definitions from conditions
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def sum_of_squares_is_2000 (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 2000

def perimeter_is_60 (a b c : ℝ) : Prop :=
  a + b + c = 60

theorem hypotenuse_length (a b c : ℝ)
  (h1 : right_angled_triangle a b c)
  (h2 : sum_of_squares_is_2000 a b c)
  (h3 : perimeter_is_60 a b c) :
  c = 10 * Real.sqrt 10 :=
sorry

end hypotenuse_length_l119_119787


namespace jane_received_change_l119_119214

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l119_119214


namespace mail_difference_eq_15_l119_119803

variable (Monday Tuesday Wednesday Thursday : ℕ)
variable (total : ℕ)

theorem mail_difference_eq_15
  (h1 : Monday = 65)
  (h2 : Tuesday = Monday + 10)
  (h3 : Wednesday = Tuesday - 5)
  (h4 : total = 295)
  (h5 : total = Monday + Tuesday + Wednesday + Thursday) :
  Thursday - Wednesday = 15 := 
  by
  sorry

end mail_difference_eq_15_l119_119803


namespace value_of_a_l119_119944

noncomputable def a_solution (z : ℂ) : ℝ :=
if H : z = (λ a, a * complex.I / (1 + 2 * complex.I)) a ∧ |z| = real.sqrt 5 ∧ a < 0 
then -5 else 0

theorem value_of_a (a : ℝ) : z = a * complex.I / (1 + 2 * complex.I) → |z| = real.sqrt 5 → a < 0 → a = -5 :=
by sorry

end value_of_a_l119_119944


namespace together_time_l119_119129

theorem together_time (P_time Q_time : ℝ) (hP : P_time = 4) (hQ : Q_time = 6) : (1 / ((1 / P_time) + (1 / Q_time))) = 2.4 :=
by
  sorry

end together_time_l119_119129


namespace smallest_four_digit_number_l119_119415

theorem smallest_four_digit_number (N : ℕ) (a b : ℕ) (h1 : N = 100 * a + b) (h2 : N = (a + b)^2) (h3 : 1000 ≤ N) (h4 : N < 10000) : N = 2025 :=
sorry

end smallest_four_digit_number_l119_119415


namespace find_m_l119_119612

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) :
  (∀ n, S n = (n * (3 * n - 1)) / 2) →
  (a 1 = 1) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (a m = 3 * m - 2) →
  (a 4 * a 4 = a 1 * a m) →
  m = 34 :=
by
  intro hS h1 ha1 ha2 hgeom
  sorry

end find_m_l119_119612


namespace probability_select_cooking_l119_119316

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119316


namespace some_number_value_correct_l119_119972

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l119_119972


namespace general_formula_a_n_sum_first_n_b_l119_119811

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Sequence property
def seq_property (n : ℕ) (S_n : ℕ) : Prop :=
  a_n n ^ 2 + 2 * a_n n = 4 * S_n + 3

-- General formula for {a_n}
theorem general_formula_a_n (n : ℕ) (hpos : ∀ n, a_n n > 0) (S_n : ℕ) (hseq : seq_property n S_n) :
  a_n n = 2 * n + 1 :=
sorry

-- Sum of the first n terms of {b_n}
def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

def sum_b (n : ℕ) (T_n : ℚ) : Prop :=
  T_n = (1 / 2) * ((1 / (2 * n + 1)) - (1 / (2 * n + 3)))

theorem sum_first_n_b (n : ℕ) (hpos : ∀ n, a_n n > 0) (T_n : ℚ) :
  T_n = (n : ℚ) / (3 * (2 * n + 3)) :=
sorry

end general_formula_a_n_sum_first_n_b_l119_119811


namespace trigonometric_identity_l119_119933

noncomputable def trig_expr (θ φ : ℝ) : ℝ :=
  (Real.cos θ)^2 + (Real.cos φ)^2 + Real.cos θ * Real.cos φ

theorem trigonometric_identity :
  trig_expr (75 * Real.pi / 180) (15 * Real.pi / 180) = 5 / 4 :=
  sorry

end trigonometric_identity_l119_119933


namespace f_9_over_2_l119_119469

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end f_9_over_2_l119_119469


namespace total_votes_l119_119791

theorem total_votes (V : ℝ) (h : 0.60 * V - 0.40 * V = 1200) : V = 6000 :=
sorry

end total_votes_l119_119791


namespace range_of_m_to_satisfy_quadratic_l119_119441

def quadratic_positive_forall_m (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + m * x + 100 > 0

theorem range_of_m_to_satisfy_quadratic :
  {m : ℝ | quadratic_positive_forall_m m} = {m : ℝ | 0 ≤ m ∧ m < 400} :=
by
  sorry

end range_of_m_to_satisfy_quadratic_l119_119441


namespace pool_capacity_is_80_percent_l119_119242

noncomputable def current_capacity_percentage (width length depth rate time : ℝ) : ℝ :=
  let total_volume := width * length * depth
  let water_removed := rate * time
  (water_removed / total_volume) * 100

theorem pool_capacity_is_80_percent :
  current_capacity_percentage 50 150 10 60 1000 = 80 :=
by
  sorry

end pool_capacity_is_80_percent_l119_119242


namespace max_value_of_f_greater_than_2_pow_2018_l119_119407

-- Definitions of the Fibonacci sequence as given
def fib : ℕ → ℕ
| 0 := 1
| 1 := 2
| (n+2) := fib (n+1) + fib n

-- Given function f
def f (x : ℝ) : ℝ :=
  ∏ i in finset.range 3030, (x - fib i)

-- Statement to prove
theorem max_value_of_f_greater_than_2_pow_2018 :
  ∃ x_0 ∈ Ioo (fib 0) (fib 3030), |f x_0| = finset.univ.sup (λ x, |f x|) ∧ x_0 > 2^2018 :=
sorry

end max_value_of_f_greater_than_2_pow_2018_l119_119407


namespace find_divisor_l119_119072

theorem find_divisor : ∃ (divisor : ℕ), ∀ (quotient remainder dividend : ℕ), quotient = 14 ∧ remainder = 7 ∧ dividend = 301 → (dividend = divisor * quotient + remainder) ∧ divisor = 21 :=
by
  sorry

end find_divisor_l119_119072


namespace min_value_of_f_inequality_for_a_b_l119_119008

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  intro x
  sorry

theorem inequality_for_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : 1/a + 1/b = Real.sqrt 3) : 
  1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end min_value_of_f_inequality_for_a_b_l119_119008


namespace fourth_root_sq_eq_sixteen_l119_119270

theorem fourth_root_sq_eq_sixteen (x : ℝ) (h : (x^(1/4))^2 = 16) : x = 256 :=
sorry

end fourth_root_sq_eq_sixteen_l119_119270


namespace probability_cooking_selected_l119_119312

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119312


namespace functional_eq_solution_l119_119178

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x :=
sorry

end functional_eq_solution_l119_119178


namespace razorback_tshirt_profit_l119_119067

theorem razorback_tshirt_profit
  (total_tshirts_sold : ℕ)
  (tshirts_sold_arkansas_game : ℕ)
  (money_made_arkansas_game : ℕ) :
  total_tshirts_sold = 163 →
  tshirts_sold_arkansas_game = 89 →
  money_made_arkansas_game = 8722 →
  money_made_arkansas_game / tshirts_sold_arkansas_game = 98 :=
by 
  intros _ _ _
  sorry

end razorback_tshirt_profit_l119_119067


namespace problem_l119_119614

theorem problem (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := by
  sorry

end problem_l119_119614


namespace x_squared_plus_y_squared_l119_119018

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^3 = 8) (h2 : x * y = 5) : 
  x^2 + y^2 = -6 := by
  sorry

end x_squared_plus_y_squared_l119_119018


namespace min_fraction_value_l119_119010

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 2

theorem min_fraction_value : ∀ x ∈ (Set.Ici (7 / 4)), (f x)^2 + 2 / (f x) ≥ 81 / 28 :=
by
  sorry

end min_fraction_value_l119_119010


namespace probability_event_a_without_replacement_independence_of_events_with_replacement_l119_119718

open ProbabilityTheory MeasureTheory Set

-- Definitions corresponding to the conditions
def BallLabeled (i : ℕ) : Prop := i ∈ Finset.range 10

def EventA (second_ball : ℕ) : Prop := second_ball = 2

def EventB (first_ball second_ball : ℕ) (m : ℕ) : Prop := first_ball + second_ball = m

-- First Part: Probability without replacement
theorem probability_event_a_without_replacement :
  ∃ P_A : ℝ, P_A = 1 / 10 := sorry

-- Second Part: Independence with replacement
theorem independence_of_events_with_replacement (m : ℕ) :
  (EventA 2 → (∀ first_ball : ℕ, BallLabeled first_ball → EventB first_ball 2 m) ↔ m = 9) := sorry

end probability_event_a_without_replacement_independence_of_events_with_replacement_l119_119718


namespace fraction_sum_zero_implies_square_sum_zero_l119_119227

theorem fraction_sum_zero_implies_square_sum_zero (a b c : ℝ) (h₀ : a ≠ b) (h₁ : b ≠ c) (h₂ : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := 
by
  sorry

end fraction_sum_zero_implies_square_sum_zero_l119_119227


namespace probability_of_selecting_cooking_l119_119338

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119338


namespace total_fruits_in_bowl_l119_119784

theorem total_fruits_in_bowl (bananas apples oranges : ℕ) 
  (h1 : bananas = 2) 
  (h2 : apples = 2 * bananas) 
  (h3 : oranges = 6) : 
  bananas + apples + oranges = 12 := 
by 
  sorry

end total_fruits_in_bowl_l119_119784


namespace multiplication_identity_multiplication_l119_119842

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l119_119842


namespace patty_can_avoid_chores_l119_119481

theorem patty_can_avoid_chores (money_per_pack packs total_cookies_per_pack chores kid_cookies_cost packs_bought total_cookies total_weekly_cost weeks : ℕ)
    (h1 : money_per_pack = 3)
    (h2 : packs = 15 / money_per_pack)
    (h3 : total_cookies_per_pack = 24)
    (h4 : total_cookies = (p : ℕ) → packs * total_cookies_per_pack)
    (h5 : chores = 4)
    (h6 : kid_cookies_cost = 3)
    (h7 : total_weekly_cost = 2 * chores * kid_cookies_cost)
    (h8 : weeks = (total_cookies / total_weekly_cost)) : 
  weeks = 10 :=
by sorry

end patty_can_avoid_chores_l119_119481


namespace repeating_decimal_sum_l119_119599

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l119_119599


namespace cost_of_first_15_kgs_l119_119921

def cost_33_kg := 333
def cost_36_kg := 366
def kilo_33 := 33
def kilo_36 := 36
def first_limit := 30
def extra_3kg := 3  -- 33 - 30
def extra_6kg := 6  -- 36 - 30

theorem cost_of_first_15_kgs (l q : ℕ) 
  (h1 : first_limit * l + extra_3kg * q = cost_33_kg)
  (h2 : first_limit * l + extra_6kg * q = cost_36_kg) :
  15 * l = 150 :=
by
  sorry

end cost_of_first_15_kgs_l119_119921


namespace sum_midpoints_x_coordinates_is_15_l119_119508

theorem sum_midpoints_x_coordinates_is_15 :
  ∀ (a b : ℝ), a + 2 * b = 15 → 
  (a + 2 * b) = 15 :=
by
  intros a b h
  sorry

end sum_midpoints_x_coordinates_is_15_l119_119508


namespace repeating_decimal_sum_in_lowest_terms_l119_119574

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l119_119574


namespace find_principal_amount_l119_119708

theorem find_principal_amount (P r : ℝ) (A2 A3 : ℝ) (n2 n3 : ℕ) 
  (h1 : n2 = 2) (h2 : n3 = 3) 
  (h3 : A2 = 8820) 
  (h4 : A3 = 9261) 
  (h5 : r = 0.05) 
  (h6 : A2 = P * (1 + r)^n2) 
  (h7 : A3 = P * (1 + r)^n3) : 
  P = 8000 := 
by 
  sorry

end find_principal_amount_l119_119708


namespace probability_of_cooking_l119_119368

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119368


namespace probability_cooking_selected_l119_119313

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119313


namespace green_apples_more_than_red_apples_l119_119684

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end green_apples_more_than_red_apples_l119_119684


namespace perimeter_bisectors_concur_l119_119758

open Real

/-- Definition of a perimeter bisector. -/
def is_perimeter_bisector {A B C : Point} (P : Point) : Prop :=
  perimeter (triangle A B P) = perimeter (triangle A C P)

/-- In any triangle ABC, the three perimeter bisectors intersect at a single point. -/
theorem perimeter_bisectors_concur {A B C : Point} :
  ∃ P : Point, (is_perimeter_bisector A P) ∧ (is_perimeter_bisector B P) ∧ (is_perimeter_bisector C P) :=
sorry

end perimeter_bisectors_concur_l119_119758


namespace green_pill_cost_l119_119734

-- Define the conditions 
variables (pinkCost greenCost : ℝ)
variable (totalCost : ℝ := 819) -- total cost for three weeks
variable (days : ℝ := 21) -- number of days in three weeks

-- Establish relationships between pink and green pill costs
axiom greenIsMore : greenCost = pinkCost + 1
axiom dailyCost : 2 * greenCost + pinkCost = 39

-- Define the theorem to prove the cost of one green pill
theorem green_pill_cost : greenCost = 40/3 :=
by
  -- Proof would go here, but is omitted for now.
  sorry

end green_pill_cost_l119_119734


namespace sum_of_min_value_and_input_l119_119766

def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem sum_of_min_value_and_input : 
  let a := -1
  let b := 3 * a - a ^ 3
  a + b = -3 := 
by
  let a := -1
  let b := 3 * a - a ^ 3
  sorry

end sum_of_min_value_and_input_l119_119766


namespace minimum_value_l119_119670

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

theorem minimum_value : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≥ f (-1) :=
by
  sorry

end minimum_value_l119_119670


namespace min_washes_at_least_4_l119_119271

noncomputable def min_washes (x : ℕ) : Prop :=
  (1/4 : ℝ)^x ≤ 1/100

theorem min_washes_at_least_4 : ∃ x, min_washes x ∧ x = 4 :=
begin
  use 4,
  unfold min_washes,
  norm_num,
  have log_ineq : (4 : ℝ).log ≤ (100 : ℝ).log,
  { apply log_le_log,
    norm_num,
    apply pow_nonneg,
    norm_num },
  exact le_of_log_ineq log_ineq,
end

end min_washes_at_least_4_l119_119271


namespace circle_tangent_x_axis_l119_119413

theorem circle_tangent_x_axis (x y : ℝ) (h_center : (x, y) = (-3, 4)) (h_tangent : y = 4) :
  ∃ r : ℝ, r = 4 ∧ (∀ x y, (x + 3)^2 + (y - 4)^2 = 16) :=
sorry

end circle_tangent_x_axis_l119_119413


namespace inequality_proof_l119_119768

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : 
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l119_119768


namespace opposite_neg_one_half_l119_119097

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l119_119097


namespace number_of_triangles_l119_119451

open Nat

-- Define the number of combinations
def comb : Nat → Nat → Nat
  | n, k => if k > n then 0 else n.choose k

-- The given conditions
def points_on_OA := 5
def points_on_OB := 6
def point_O := 1
def total_points := points_on_OA + points_on_OB + point_O -- should equal 12

-- Lean proof problem statement
theorem number_of_triangles : comb total_points 3 - comb points_on_OA 3 - comb points_on_OB 3 = 165 := by
  sorry

end number_of_triangles_l119_119451


namespace difference_in_sword_length_l119_119406

variables (c_length : ℕ) (j_length : ℕ) (jn_length : ℕ)

def christopher_sword_length : ℕ := 15

def jameson_sword_length (c_length : ℕ) : ℕ := 2 * c_length + 3

def june_sword_length (j_length : ℕ) : ℕ := j_length + 5

theorem difference_in_sword_length :
  let c_length := christopher_sword_length in
  let j_length := jameson_sword_length c_length in
  let jn_length := june_sword_length j_length in
  jn_length = c_length + 23 :=
by
  sorry

end difference_in_sword_length_l119_119406


namespace ellipse_iff_k_range_l119_119781

theorem ellipse_iff_k_range (k : ℝ) :
  (∃ x y, (x ^ 2 / (1 - k)) + (y ^ 2 / (1 + k)) = 1) ↔ (-1 < k ∧ k < 1 ∧ k ≠ 0) :=
by
  sorry

end ellipse_iff_k_range_l119_119781


namespace camper_ratio_l119_119397

theorem camper_ratio (total_campers : ℕ) (G : ℕ) (B : ℕ)
  (h1: total_campers = 96) 
  (h2: G = total_campers / 3) 
  (h3: B = total_campers - G) 
  : B / total_campers = 2 / 3 :=
  by
    sorry

end camper_ratio_l119_119397


namespace fraction_addition_l119_119929

theorem fraction_addition : 
  (2 : ℚ) / 5 + (3 : ℚ) / 8 + 1 = 71 / 40 :=
by
  sorry

end fraction_addition_l119_119929


namespace probability_cooking_is_one_fourth_l119_119356
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119356


namespace arithmetic_expression_value_l119_119524

theorem arithmetic_expression_value :
  2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 :=
by {
  sorry
}

end arithmetic_expression_value_l119_119524


namespace least_four_digit_palindrome_div_by_5_l119_119119

noncomputable def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString in s = s.reverse

theorem least_four_digit_palindrome_div_by_5 : 
  ∃ n : ℕ, is_palindrome n ∧ 1000 ≤ n ∧ n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, is_palindrome m ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 5 = 0 → n ≤ m := 
sorry

end least_four_digit_palindrome_div_by_5_l119_119119


namespace average_of_second_set_l119_119068

open Real

theorem average_of_second_set 
  (avg6 : ℝ)
  (n1 n2 n3 n4 n5 n6 : ℝ)
  (avg1_set : ℝ)
  (avg3_set : ℝ)
  (h1 : avg6 = 3.95)
  (h2 : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = avg6)
  (h3 : (n1 + n2) / 2 = 3.6)
  (h4 : (n5 + n6) / 2 = 4.400000000000001) :
  (n3 + n4) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_l119_119068


namespace min_value_of_quadratic_form_l119_119951

theorem min_value_of_quadratic_form (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + 2 * y^2 + 3 * z^2 ≥ 1/3 :=
sorry

end min_value_of_quadratic_form_l119_119951


namespace circles_intersect_at_two_points_l119_119645

noncomputable def point_intersection_count (A B : ℝ × ℝ) (rA rB d : ℝ) : ℕ :=
  let distance := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2
  if rA + rB >= d ∧ d >= |rA - rB| then 2 else if d = rA + rB ∨ d = |rA - rB| then 1 else 0

theorem circles_intersect_at_two_points :
  point_intersection_count (0, 0) (8, 0) 3 6 8 = 2 :=
by 
  -- Proof for the statement will go here
  sorry

end circles_intersect_at_two_points_l119_119645


namespace least_positive_integer_added_to_575_multiple_4_l119_119521

theorem least_positive_integer_added_to_575_multiple_4 :
  ∃ n : ℕ, n > 0 ∧ (575 + n) % 4 = 0 ∧ 
           ∀ m : ℕ, (m > 0 ∧ (575 + m) % 4 = 0) → n ≤ m := by
  sorry

end least_positive_integer_added_to_575_multiple_4_l119_119521


namespace factor_tree_X_value_l119_119030

theorem factor_tree_X_value :
  let F := 2 * 5
  let G := 7 * 3
  let Y := 7 * F
  let Z := 11 * G
  let X := Y * Z
  X = 16170 := by
sorry

end factor_tree_X_value_l119_119030


namespace basketball_cricket_students_l119_119983

theorem basketball_cricket_students {A B : Finset ℕ} (hA : A.card = 7) (hB : B.card = 8) (hAB : (A ∩ B).card = 3) :
  (A ∪ B).card = 12 :=
by
  sorry

end basketball_cricket_students_l119_119983


namespace probability_of_selecting_cooking_is_one_fourth_l119_119375

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119375


namespace unique_3_digit_number_with_conditions_l119_119744

def valid_3_digit_number (n : ℕ) : Prop :=
  let d2 := n / 100
  let d1 := (n / 10) % 10
  let d0 := n % 10
  (d2 > 0) ∧ (d2 < 10) ∧ (d1 < 10) ∧ (d0 < 10) ∧ (d2 + d1 + d0 = 28) ∧ (d0 < 7) ∧ (d0 % 2 = 0)

theorem unique_3_digit_number_with_conditions :
  (∃! n : ℕ, valid_3_digit_number n) :=
sorry

end unique_3_digit_number_with_conditions_l119_119744


namespace prob_none_three_win_prob_at_least_two_not_win_l119_119380

-- Definitions for probabilities
def prob_win : ℚ := 1 / 6
def prob_not_win : ℚ := 1 - prob_win

-- Problem 1: Prove probability that none of the three students win
theorem prob_none_three_win : (prob_not_win ^ 3) = 125 / 216 := by
  sorry

-- Problem 2: Prove probability that at least two of the three students do not win
theorem prob_at_least_two_not_win : 1 - (3 * (prob_win ^ 2) * prob_not_win + prob_win ^ 3) = 25 / 27 := by
  sorry

end prob_none_three_win_prob_at_least_two_not_win_l119_119380


namespace max_area_of_triangle_l119_119745

noncomputable def maxAreaTriangle (m_a m_b m_c : ℝ) : ℝ :=
  1/3 * Real.sqrt (2 * (m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4))

theorem max_area_of_triangle (m_a m_b m_c : ℝ) (h1 : m_a ≤ 2) (h2 : m_b ≤ 3) (h3 : m_c ≤ 4) :
  maxAreaTriangle m_a m_b m_c ≤ 4 :=
sorry

end max_area_of_triangle_l119_119745


namespace minimum_number_of_colors_l119_119627

theorem minimum_number_of_colors (n : ℕ) (h_n : 2 ≤ n) :
  ∀ (f : (Fin n) → ℕ),
  (∀ i j : Fin n, i ≠ j → f i ≠ f j) →
  (∃ c : ℕ, c = n) :=
by sorry

end minimum_number_of_colors_l119_119627


namespace number_of_ways_to_choose_officers_l119_119381

-- Define the number of boys and girls.
def num_boys : ℕ := 12
def num_girls : ℕ := 13

-- Define the total number of boys and girls.
def num_members : ℕ := num_boys + num_girls

-- Calculate the number of ways to choose the president, vice-president, and secretary with given conditions.
theorem number_of_ways_to_choose_officers : 
  (num_boys * num_girls * (num_boys - 1)) + (num_girls * num_boys * (num_girls - 1)) = 3588 :=
by
  -- The first part calculates the ways when the president is a boy.
  -- The second part calculates the ways when the president is a girl.
  sorry

end number_of_ways_to_choose_officers_l119_119381


namespace fixed_point_of_function_l119_119867

theorem fixed_point_of_function :
  (4, 4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, 2^(x-4) + 3) } :=
by
  sorry

end fixed_point_of_function_l119_119867


namespace find_positive_integer_pairs_l119_119938

theorem find_positive_integer_pairs :
  ∀ (m n : ℕ), m > 0 ∧ n > 0 → ∃ k : ℕ, (2^n - 13^m = k^3) ↔ (m = 2 ∧ n = 9) :=
by
  sorry

end find_positive_integer_pairs_l119_119938


namespace find_f_inv_486_l119_119200

-- Assuming function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_cond1 : f 4 = 2
axiom f_cond2 : ∀ x : ℝ, f (3 * x) = 3 * f x

-- Proof problem: Prove that f⁻¹(486) = 972
theorem find_f_inv_486 : (∃ x : ℝ, f x = 486 ∧ x = 972) :=
sorry

end find_f_inv_486_l119_119200


namespace students_walk_fraction_l119_119737

theorem students_walk_fraction (h1 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/3))
                               (h2 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/5))
                               (h3 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/8))
                               (h4 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/10)) :
  ∃ (students : ℕ), (students - num1 - num2 - num3 - num4) / students = 29 / 120 :=
by
  sorry

end students_walk_fraction_l119_119737


namespace find_m_value_l119_119763

theorem find_m_value :
  ∃ (m : ℝ), (∃ (midpoint: ℝ × ℝ), midpoint = ((5 + m) / 2, 1) ∧ midpoint.1 - 2 * midpoint.2 = 0) -> m = -1 :=
by
  sorry

end find_m_value_l119_119763


namespace probability_selecting_cooking_l119_119306

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l119_119306


namespace range_of_a_l119_119026

theorem range_of_a (a : ℝ) (h : (∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2)) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l119_119026


namespace perimeter_result_l119_119910

-- Define the side length of the square
def side_length : ℕ := 100

-- Define the dimensions of the rectangle
def rectangle_dim1 : ℕ := side_length
def rectangle_dim2 : ℕ := side_length / 2

-- Perimeter calculation based on the arrangement
def perimeter : ℕ :=
  3 * rectangle_dim1 + 4 * rectangle_dim2

-- The statement of the problem
theorem perimeter_result :
  perimeter = 500 :=
by
  sorry

end perimeter_result_l119_119910


namespace tangent_line_parallel_range_a_l119_119669

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log x + 1/2 * x^2 + a * x

theorem tangent_line_parallel_range_a (a : ℝ) :
  (∃ x > 0, deriv (f a) x = 3) ↔ a ≤ 1 :=
by
  sorry

end tangent_line_parallel_range_a_l119_119669


namespace probability_cooking_is_one_fourth_l119_119360
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119360


namespace xiaoming_selects_cooking_probability_l119_119330

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119330


namespace circle_radius_l119_119661

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l119_119661


namespace multiply_by_11_l119_119514

theorem multiply_by_11 (A B : ℕ) (h : A + B < 10) : 
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B :=
by
  sorry

end multiply_by_11_l119_119514


namespace eval_fraction_expression_l119_119411
noncomputable def inner_expr := 2 + 2
noncomputable def middle_expr := 2 + (1 / inner_expr)
noncomputable def outer_expr := 2 + (1 / middle_expr)

theorem eval_fraction_expression : outer_expr = 22 / 9 := by
  sorry

end eval_fraction_expression_l119_119411


namespace trigonometric_identity_application_l119_119868

theorem trigonometric_identity_application :
  (1 / 2) * (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = (1 / 8) :=
by
  sorry

end trigonometric_identity_application_l119_119868


namespace problem1_problem2_l119_119953

def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

theorem problem1 (x : ℝ) : f x (-1) ≤ 2 ↔ -1 / 2 ≤ x ∧ x ≤ 1 / 2 :=
by sorry

theorem problem2 (a : ℝ) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 1, f x a ≤ |2 * x + 1|) → (0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l119_119953


namespace lucinda_jelly_beans_l119_119477

theorem lucinda_jelly_beans (g l : ℕ) 
  (h₁ : g = 3 * l) 
  (h₂ : g - 20 = 4 * (l - 20)) : 
  g = 180 := 
by 
  sorry

end lucinda_jelly_beans_l119_119477


namespace largest_sum_is_1173_l119_119266

def largest_sum_of_two_3digit_numbers : Prop :=
  ∃ a b c d e f : ℕ, 
  (a = 6 ∧ b = 5 ∧ c = 4 ∧ d = 3 ∧ e = 2 ∧ f = 1) ∧
  100 * (a + b) + 10 * (c + d) + (e + f) = 1173

theorem largest_sum_is_1173 : largest_sum_of_two_3digit_numbers :=
  by
  sorry

end largest_sum_is_1173_l119_119266


namespace opposite_of_half_l119_119087

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l119_119087


namespace intersection_M_S_l119_119438

namespace ProofProblem

def M : Set ℕ := { x | 0 < x ∧ x < 4 }

def S : Set ℕ := { 2, 3, 5 }

theorem intersection_M_S :
  M ∩ S = { 2, 3 } := by
  sorry

end ProofProblem

end intersection_M_S_l119_119438


namespace sum_of_coords_of_circle_center_l119_119417

theorem sum_of_coords_of_circle_center (x y : ℝ) :
  (x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by
  sorry

end sum_of_coords_of_circle_center_l119_119417


namespace xiaoming_selects_cooking_probability_l119_119326

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119326


namespace terminal_side_in_third_quadrant_l119_119505

def is_equivalent_angle (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

def in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

theorem terminal_side_in_third_quadrant : 
  ∀ θ, θ = 600 → in_third_quadrant (θ % 360) :=
by
  intro θ
  intro hθ
  sorry

end terminal_side_in_third_quadrant_l119_119505


namespace smallest_odd_factors_gt_100_l119_119932

theorem smallest_odd_factors_gt_100 : ∃ n : ℕ, n > 100 ∧ (∀ d : ℕ, d ∣ n → (∃ m : ℕ, n = m * m)) ∧ (∀ m : ℕ, m > 100 ∧ (∀ d : ℕ, d ∣ m → (∃ k : ℕ, m = k * k)) → n ≤ m) :=
by
  sorry

end smallest_odd_factors_gt_100_l119_119932


namespace probability_of_selecting_cooking_l119_119347

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119347


namespace postage_stamp_problem_l119_119720

theorem postage_stamp_problem
  (x y z : ℕ) (h1: y = 10 * x) (h2: x + 2 * y + 5 * z = 100) :
  x = 5 ∧ y = 50 ∧ z = 0 :=
by
  sorry

end postage_stamp_problem_l119_119720


namespace rainfall_on_tuesday_l119_119452

noncomputable def R_Tuesday (R_Sunday : ℝ) (D1 : ℝ) : ℝ := 
  R_Sunday + D1

noncomputable def R_Thursday (R_Tuesday : ℝ) (D2 : ℝ) : ℝ :=
  R_Tuesday + D2

noncomputable def total_rainfall (R_Sunday R_Tuesday R_Thursday : ℝ) : ℝ :=
  R_Sunday + R_Tuesday + R_Thursday

theorem rainfall_on_tuesday : R_Tuesday 2 3.75 = 5.75 := 
by 
  sorry -- Proof goes here

end rainfall_on_tuesday_l119_119452


namespace positive_diff_two_largest_prime_factors_l119_119703

theorem positive_diff_two_largest_prime_factors (a b c d : ℕ) (h : 178469 = a * b * c * d) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) 
  (hle1 : a ≤ b) (hle2 : b ≤ c) (hle3 : c ≤ d):
  d - c = 2 := by sorry

end positive_diff_two_largest_prime_factors_l119_119703


namespace repeating_decimal_sum_l119_119570

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l119_119570


namespace repeating_decimals_sum_l119_119591

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l119_119591


namespace multiply_and_simplify_l119_119846
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l119_119846


namespace distance_formula_example_l119_119121

variable (x1 y1 x2 y2 : ℝ)

theorem distance_formula_example : dist (3, -1) (-4, 3) = Real.sqrt 65 :=
by
  let x1 := 3
  let y1 := -1
  let x2 := -4
  let y2 := 3
  sorry

end distance_formula_example_l119_119121


namespace price_reduction_percentage_price_increase_amount_l119_119540

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l119_119540


namespace range_of_a_l119_119814

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) →
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l119_119814


namespace doctor_visit_cost_l119_119116

theorem doctor_visit_cost (cast_cost : ℝ) (insurance_coverage : ℝ) (out_of_pocket : ℝ) (visit_cost : ℝ) :
  cast_cost = 200 → insurance_coverage = 0.60 → out_of_pocket = 200 → 0.40 * (visit_cost + cast_cost) = out_of_pocket → visit_cost = 300 :=
by
  intros h_cast h_insurance h_out_of_pocket h_equation
  sorry

end doctor_visit_cost_l119_119116


namespace nth_number_eq_l119_119909

noncomputable def nth_number (n : Nat) : ℚ := n / (n^2 + 1)

theorem nth_number_eq (n : Nat) : nth_number n = n / (n^2 + 1) :=
by
  sorry

end nth_number_eq_l119_119909


namespace repeating_decimals_sum_l119_119579

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l119_119579


namespace probability_of_cooking_l119_119367

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119367


namespace decreased_value_l119_119987

noncomputable def original_expression (x y: ℝ) : ℝ :=
  x * y^2

noncomputable def decreased_expression (x y: ℝ) : ℝ :=
  (1 / 2) * x * (1 / 2 * y) ^ 2

theorem decreased_value (x y: ℝ) :
  decreased_expression x y = (1 / 8) * original_expression x y :=
by
  sorry

end decreased_value_l119_119987


namespace problem_l119_119061

-- Define the problem conditions and the statement that needs to be proved
theorem problem:
  ∀ (x : ℝ), (x ∈ Set.Icc (-1) m) ∧ ((1 - (-1)) / (m - (-1)) = 2 / 5) → m = 4 := by
  sorry

end problem_l119_119061


namespace bananas_needed_to_make_yogurts_l119_119743

theorem bananas_needed_to_make_yogurts 
    (slices_per_yogurt : ℕ) 
    (slices_per_banana: ℕ) 
    (number_of_yogurts: ℕ) 
    (total_needed_slices: ℕ) 
    (bananas_needed: ℕ) 
    (h1: slices_per_yogurt = 8)
    (h2: slices_per_banana = 10)
    (h3: number_of_yogurts = 5)
    (h4: total_needed_slices = number_of_yogurts * slices_per_yogurt)
    (h5: bananas_needed = total_needed_slices / slices_per_banana): 
    bananas_needed = 4 := 
by
    sorry

end bananas_needed_to_make_yogurts_l119_119743


namespace opposite_of_neg_half_l119_119079

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l119_119079


namespace total_amount_raised_l119_119860

-- Definitions based on conditions
def PancakeCost : ℕ := 4
def BaconCost : ℕ := 2
def NumPancakesSold : ℕ := 60
def NumBaconSold : ℕ := 90

-- Lean statement proving that the total amount raised is $420
theorem total_amount_raised : (NumPancakesSold * PancakeCost) + (NumBaconSold * BaconCost) = 420 := by
  -- Since we are not required to prove, we use sorry here
  sorry

end total_amount_raised_l119_119860


namespace repeating_decimal_sum_in_lowest_terms_l119_119575

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l119_119575


namespace work_days_of_A_and_B_l119_119383

theorem work_days_of_A_and_B (B : ℝ) (A : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 27) :
  1 / (A + B) = 9 :=
by
  sorry

end work_days_of_A_and_B_l119_119383


namespace shortest_travel_time_to_sunny_town_l119_119727

-- Definitions based on the given conditions
def highway_length : ℕ := 12

def railway_crossing_closed (t : ℕ) : Prop :=
  ∃ k : ℕ, t = 6 * k + 0 ∨ t = 6 * k + 1 ∨ t = 6 * k + 2

def traffic_light_red (t : ℕ) : Prop :=
  ∃ k1 : ℕ, t = 5 * k1 + 0 ∨ t = 5 * k1 + 1

def initial_conditions (t : ℕ) : Prop :=
  railway_crossing_closed 0 ∧ traffic_light_red 0

def shortest_time_to_sunny_town (time : ℕ) : Prop := 
  time = 24

-- The proof statement
theorem shortest_travel_time_to_sunny_town :
  ∃ time : ℕ, shortest_time_to_sunny_town time ∧
  (∀ t : ℕ, 0 ≤ t → t ≤ time → ¬railway_crossing_closed t ∧ ¬traffic_light_red t) :=
sorry

end shortest_travel_time_to_sunny_town_l119_119727


namespace probability_of_selecting_cooking_l119_119337

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119337


namespace find_integer_n_l119_119870

theorem find_integer_n : 
  ∃ n : ℤ, 50 ≤ n ∧ n ≤ 150 ∧ (n % 7 = 0) ∧ (n % 9 = 3) ∧ (n % 4 = 3) ∧ n = 147 :=
by 
  -- sorry is used here as a placeholder for the actual proof
  sorry

end find_integer_n_l119_119870


namespace emma_finishes_first_l119_119554

noncomputable def david_lawn_area : ℝ := sorry
noncomputable def emma_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 3
noncomputable def fiona_lawn_area (david_lawn_area : ℝ) : ℝ := david_lawn_area / 4

noncomputable def david_mowing_rate : ℝ := sorry
noncomputable def fiona_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 6
noncomputable def emma_mowing_rate (david_mowing_rate : ℝ) : ℝ := david_mowing_rate / 2

theorem emma_finishes_first (z w : ℝ) (hz : z > 0) (hw : w > 0) :
  (z / w) > (2 * z / (3 * w)) ∧ (3 * z / (2 * w)) > (2 * z / (3 * w)) :=
by
  sorry

end emma_finishes_first_l119_119554


namespace check_inequality_l119_119753

theorem check_inequality : 1.7^0.3 > 0.9^3.1 :=
sorry

end check_inequality_l119_119753


namespace clock_angle_at_7_oclock_l119_119700

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l119_119700


namespace probability_of_selecting_cooking_is_one_fourth_l119_119373

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119373


namespace f_half_and_minus_half_l119_119187

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem f_half_and_minus_half :
  f (1 / 2) + f (-1 / 2) = 2 := by
  sorry

end f_half_and_minus_half_l119_119187


namespace repeating_decimal_sum_l119_119563

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l119_119563


namespace num_values_x_satisfying_l119_119440

theorem num_values_x_satisfying (
  f : ℝ → ℝ → ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (x : ℝ)
  (h_eq : ∀ x, f (cos x) (sin x) = 2 ↔ (cos x) ^ 2 + 3 * (sin x) ^ 2 = 2)
  (h_interval : ∀ x, -20 < x ∧ x < 90)
  (h_cos_sin : ∀ x, cos x = cos (x) ∧ sin x = sin (x)) :
  ∃ n, n = 70 := sorry

end num_values_x_satisfying_l119_119440


namespace radius_of_circle_l119_119655

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end radius_of_circle_l119_119655


namespace probability_of_selecting_cooking_l119_119343

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119343


namespace ABC_three_digit_number_l119_119396

theorem ABC_three_digit_number : 
    ∃ (A B C : ℕ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    3 * C % 10 = 8 ∧ 
    3 * B + 1 % 10 = 8 ∧ 
    3 * A + 2 = 8 ∧ 
    100 * A + 10 * B + C = 296 := 
by
  sorry

end ABC_three_digit_number_l119_119396


namespace men_entered_l119_119989

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end men_entered_l119_119989


namespace solve_equation_l119_119855

theorem solve_equation (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 :=
sorry

end solve_equation_l119_119855


namespace equivalent_mod_l119_119181

theorem equivalent_mod (h : 5^300 ≡ 1 [MOD 1250]) : 5^9000 ≡ 1 [MOD 1000] :=
by 
  sorry

end equivalent_mod_l119_119181


namespace bicycle_speed_B_l119_119903

theorem bicycle_speed_B 
  (distance : ℝ := 12)
  (ratio : ℝ := 1.2)
  (time_diff : ℝ := 1 / 6) : 
  ∃ (B_speed : ℝ), B_speed = 12 :=
by
  let A_speed := ratio * B_speed
  have eqn : distance / B_speed - time_diff = distance / A_speed := sorry
  exact ⟨12, sorry⟩

end bicycle_speed_B_l119_119903


namespace notebook_cost_l119_119203

theorem notebook_cost (s n c : ℕ) (h1 : s > 17) (h2 : n > 2 ∧ n % 2 = 0) (h3 : c > n) (h4 : s * c * n = 2013) : c = 61 :=
sorry

end notebook_cost_l119_119203


namespace min_value_of_b1_plus_b2_l119_119510

theorem min_value_of_b1_plus_b2 (b : ℕ → ℕ) (h1 : ∀ n ≥ 1, b (n + 2) = (b n + 4030) / (1 + b (n + 1)))
  (h2 : ∀ n, b n > 0) : ∃ b1 b2, b1 * b2 = 4030 ∧ b1 + b2 = 127 :=
by {
  sorry
}

end min_value_of_b1_plus_b2_l119_119510


namespace correct_judgements_l119_119149

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_period_1 : ∀ x : ℝ, f (x + 1) = -f x
axiom f_increasing_0_1 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

theorem correct_judgements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ 
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧ 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧ 
  ¬(∀ x y : ℝ, -2 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y) :=
by 
  sorry

end correct_judgements_l119_119149


namespace car_travel_time_l119_119926

-- Definitions
def speed : ℝ := 50
def miles_per_gallon : ℝ := 30
def tank_capacity : ℝ := 15
def fraction_used : ℝ := 0.5555555555555556

-- Theorem statement
theorem car_travel_time : (fraction_used * tank_capacity * miles_per_gallon / speed) = 5 :=
sorry

end car_travel_time_l119_119926


namespace circle_radius_l119_119663

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l119_119663


namespace sum_of_k_l119_119073

theorem sum_of_k : ∃ (k_vals : List ℕ), 
  (∀ k ∈ k_vals, ∃ α β : ℤ, α + β = k ∧ α * β = -20) 
  ∧ k_vals.sum = 29 :=
by 
  sorry

end sum_of_k_l119_119073


namespace trig_identity_l119_119444

open Real

theorem trig_identity (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 6) (h : sin α ^ 6 + cos α ^ 6 = 7 / 12) : 1998 * cos α = 333 * Real.sqrt 30 :=
sorry

end trig_identity_l119_119444


namespace multiple_properties_l119_119857

variables (a b : ℤ)

-- Definitions of the conditions
def is_multiple_of_4 (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k
def is_multiple_of_8 (x : ℤ) : Prop := ∃ k : ℤ, x = 8 * k

-- Problem statement
theorem multiple_properties (h1 : is_multiple_of_4 a) (h2 : is_multiple_of_8 b) :
  is_multiple_of_4 b ∧ is_multiple_of_4 (a + b) ∧ (∃ k : ℤ, a + b = 2 * k) :=
by
  sorry

end multiple_properties_l119_119857


namespace probability_of_selecting_cooking_l119_119335

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119335


namespace P_sufficient_but_not_necessary_for_Q_l119_119949

variable (x : ℝ)

def P := x ≥ 0
def Q := 2 * x + 1 / (2 * x + 1) ≥ 1

theorem P_sufficient_but_not_necessary_for_Q : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l119_119949


namespace constant_S13_l119_119038

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |> List.sum

-- Defining the given conditions as hypotheses
variable {a : ℕ → ℤ} {d : ℤ}
variable (h_arith : arithmetic_sequence a d)
variable (constant_sum : a 2 + a 4 + a 15 = k)

-- Goal to prove: S_13 is a constant
theorem constant_S13 (k : ℤ) :
  sum_first_n_terms a 13 = k :=
  sorry

end constant_S13_l119_119038


namespace xiaoming_selects_cooking_probability_l119_119327

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119327


namespace moles_of_NaOH_combined_l119_119156

-- Given conditions
def moles_AgNO3 := 3
def moles_AgOH := 3
def balanced_ratio_AgNO3_NaOH := 1 -- 1:1 ratio as per the equation

-- Problem statement
theorem moles_of_NaOH_combined : 
  moles_AgOH = moles_AgNO3 → balanced_ratio_AgNO3_NaOH = 1 → 
  (∃ moles_NaOH, moles_NaOH = 3) := by
  sorry

end moles_of_NaOH_combined_l119_119156


namespace A_loses_240_l119_119823

def initial_house_value : ℝ := 12000
def house_value_after_A_sells : ℝ := initial_house_value * 0.85
def house_value_after_B_sells_back : ℝ := house_value_after_A_sells * 1.2

theorem A_loses_240 : house_value_after_B_sells_back - initial_house_value = 240 := by
  sorry

end A_loses_240_l119_119823


namespace initial_interest_rate_l119_119500

theorem initial_interest_rate
    (P R : ℝ) 
    (h1 : P * R = 10120) 
    (h2 : P * (R + 6) = 12144) : 
    R = 30 :=
sorry

end initial_interest_rate_l119_119500


namespace patty_weeks_without_chores_correct_l119_119483

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l119_119483


namespace common_z_values_l119_119408

theorem common_z_values (z : ℝ) :
  (∃ x : ℝ, x^2 + z^2 = 9 ∧ x^2 = 4*z - 5) ↔ (z = -2 + 3*Real.sqrt 2 ∨ z = -2 - 3*Real.sqrt 2) := 
sorry

end common_z_values_l119_119408


namespace multiplication_identity_multiplication_l119_119843

theorem multiplication_identity (x y : ℝ) :
    let a := 3 * x^2
    let b := 4 * y^3
    (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
by
  sorry

theorem multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  have h1 : (3 * x^2 - 4 * y^3) = a - b := rfl
  have h2 : (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = a^2 + a * b + b^2 := rfl
  have h := multiplication_identity x y
  rw [h1, h2] at h
  exact h

end multiplication_identity_multiplication_l119_119843


namespace no_such_real_numbers_l119_119409

noncomputable def have_integer_roots (a b c : ℝ) : Prop :=
  ∃ r s : ℤ, a * (r:ℝ)^2 + b * r + c = 0 ∧ a * (s:ℝ)^2 + b * s + c = 0

theorem no_such_real_numbers (a b c : ℝ) :
  have_integer_roots a b c → have_integer_roots (a + 1) (b + 1) (c + 1) → False :=
by
  -- proof will go here
  sorry

end no_such_real_numbers_l119_119409


namespace student_correct_answers_l119_119529

theorem student_correct_answers (c w : ℕ) 
  (h1 : c + w = 60)
  (h2 : 4 * c - w = 120) : 
  c = 36 :=
sorry

end student_correct_answers_l119_119529


namespace probability_of_selecting_cooking_l119_119344

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119344


namespace three_digit_number_increase_l119_119144

theorem three_digit_number_increase (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n * 1001 / n) = 1001 :=
by
  sorry

end three_digit_number_increase_l119_119144


namespace factor_expression_l119_119936

theorem factor_expression (x : ℝ) :
  80 * x ^ 5 - 250 * x ^ 9 = -10 * x ^ 5 * (25 * x ^ 4 - 8) :=
by
  sorry

end factor_expression_l119_119936


namespace more_crayons_given_to_Lea_than_Mae_l119_119055

-- Define the initial number of crayons
def initial_crayons : ℕ := 4 * 8

-- Condition: Nori gave 5 crayons to Mae
def crayons_given_to_Mae : ℕ := 5

-- Condition: Nori has 15 crayons left after giving some to Lea
def crayons_left_after_giving_to_Lea : ℕ := 15

-- Define the number of crayons after giving to Mae
def crayons_after_giving_to_Mae : ℕ := initial_crayons - crayons_given_to_Mae

-- Define the number of crayons given to Lea
def crayons_given_to_Lea : ℕ := crayons_after_giving_to_Mae - crayons_left_after_giving_to_Lea

-- Prove the number of more crayons given to Lea than Mae
theorem more_crayons_given_to_Lea_than_Mae : (crayons_given_to_Lea - crayons_given_to_Mae) = 7 := by
  sorry

end more_crayons_given_to_Lea_than_Mae_l119_119055


namespace average_percent_score_l119_119640

def num_students : ℕ := 180

def score_distrib : List (ℕ × ℕ) :=
[(95, 12), (85, 30), (75, 50), (65, 45), (55, 30), (45, 13)]

noncomputable def total_score : ℕ :=
(95 * 12) + (85 * 30) + (75 * 50) + (65 * 45) + (55 * 30) + (45 * 13)

noncomputable def average_score : ℕ :=
total_score / num_students

theorem average_percent_score : average_score = 70 :=
by 
  -- Here you would provide the proof, but for now we will leave it as:
  sorry

end average_percent_score_l119_119640


namespace solution_to_inequalities_l119_119939

theorem solution_to_inequalities (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end solution_to_inequalities_l119_119939


namespace repeating_decimal_to_fraction_l119_119412

theorem repeating_decimal_to_fraction :
  7.4646464646 = (739 / 99) :=
  sorry

end repeating_decimal_to_fraction_l119_119412


namespace probability_of_selecting_cooking_is_one_fourth_l119_119372

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119372


namespace quadratic_inequality_solution_range_l119_119980

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + m * x + 2 > 0) ↔ m > -3 := 
sorry

end quadratic_inequality_solution_range_l119_119980


namespace xiaoming_selects_cooking_probability_l119_119325

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119325


namespace similarity_transformation_result_l119_119985

-- Define the original coordinates of point A and the similarity ratio
def A : ℝ × ℝ := (2, 2)
def ratio : ℝ := 2

-- Define the similarity transformation that scales coordinates, optionally considering reflection
def similarity_transform (p : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  (r * p.1, r * p.2)

-- Use Lean to state the theorem based on the given conditions and expected answer
theorem similarity_transformation_result :
  similarity_transform A ratio = (4, 4) ∨ similarity_transform A (-ratio) = (-4, -4) :=
by
  sorry

end similarity_transformation_result_l119_119985


namespace alexander_eq_alice_l119_119676

-- Definitions and conditions
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.07

-- Calculation functions for Alexander and Alice
def alexander_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let taxed_price := price * (1 + tax)
  let discounted_price := taxed_price * (1 - discount)
  discounted_price

def alice_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

-- Proof that the difference between Alexander's and Alice's total is 0
theorem alexander_eq_alice : 
  alexander_total original_price discount_rate sales_tax_rate = 
  alice_total original_price discount_rate sales_tax_rate :=
by
  sorry

end alexander_eq_alice_l119_119676


namespace sequence_bound_l119_119636

theorem sequence_bound (a : ℕ → ℝ) (n : ℕ) 
  (h₁ : a 0 = 0) 
  (h₂ : a (n + 1) = 0)
  (h₃ : ∀ k, 1 ≤ k → k ≤ n → a (k - 1) - 2 * (a k) + (a (k + 1)) ≤ 1) 
  : ∀ k, 0 ≤ k → k ≤ n + 1 → a k ≤ (k * (n + 1 - k)) / 2 :=
sorry

end sequence_bound_l119_119636


namespace opposite_neg_half_l119_119084

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l119_119084


namespace probability_of_selecting_cooking_l119_119289

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119289


namespace sum_g_values_l119_119228

noncomputable def g (x : ℝ) : ℝ :=
if x > 3 then x^2 - 1 else
if x >= -3 then 3 * x + 2 else 4

theorem sum_g_values : g (-4) + g 0 + g 4 = 21 :=
by
  sorry

end sum_g_values_l119_119228


namespace angle_between_clock_hands_at_7_oclock_l119_119699

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l119_119699


namespace lcm_of_a_c_l119_119245

theorem lcm_of_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : Nat.lcm a c = 30 := by
  sorry

end lcm_of_a_c_l119_119245


namespace two_digit_number_reverse_sum_eq_99_l119_119668

theorem two_digit_number_reverse_sum_eq_99 :
  ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ ((10 * a + b) - (10 * b + a) = 5 * (a + b))
  ∧ (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_number_reverse_sum_eq_99_l119_119668


namespace opposite_neg_one_half_l119_119100

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l119_119100


namespace repeating_decimal_sum_l119_119559

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l119_119559


namespace line_condition_l119_119429

/-- Given a line l1 passing through points A(-2, m) and B(m, 4),
    a line l2 given by the equation 2x + y - 1 = 0,
    and a line l3 given by the equation x + ny + 1 = 0,
    if l1 is parallel to l2 and l2 is perpendicular to l3,
    then the value of m + n is -10. -/
theorem line_condition (m n : ℝ) (h1 : (4 - m) / (m + 2) = -2)
  (h2 : (2 * -1) * (-1 / n) = -1) : m + n = -10 := 
sorry

end line_condition_l119_119429


namespace probability_of_selecting_cooking_l119_119353

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119353


namespace men_entered_room_l119_119993

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l119_119993


namespace total_purchase_cost_l119_119466

-- Definitions for the quantities of the items
def quantity_chocolate_bars : ℕ := 10
def quantity_gummy_bears : ℕ := 10
def quantity_chocolate_chips : ℕ := 20

-- Definitions for the costs of the items
def cost_per_chocolate_bar : ℕ := 3
def cost_per_gummy_bear_pack : ℕ := 2
def cost_per_chocolate_chip_bag : ℕ := 5

-- Proof statement to be shown
theorem total_purchase_cost :
  (quantity_chocolate_bars * cost_per_chocolate_bar) + 
  (quantity_gummy_bears * cost_per_gummy_bear_pack) + 
  (quantity_chocolate_chips * cost_per_chocolate_chip_bag) = 150 :=
sorry

end total_purchase_cost_l119_119466


namespace probability_of_cooking_l119_119371

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119371


namespace line_through_midpoint_bisects_chord_eqn_l119_119430

theorem line_through_midpoint_bisects_chord_eqn :
  ∀ (x y : ℝ), (x^2 - 4*y^2 = 4) ∧ (∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 - 4 * y1^2 = 4) ∧ (x2^2 - 4 * y2^2 = 4) ∧ 
    (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = -1) → 
    3 * x + 4 * y - 5 = 0 :=
by
  intros x y h
  sorry

end line_through_midpoint_bisects_chord_eqn_l119_119430


namespace solve_for_x_l119_119777

-- Assume x is a positive integer
def pos_integer (x : ℕ) : Prop := 0 < x

-- Assume the equation holds for some x
def equation (x : ℕ) : Prop :=
  1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170

-- Proposition stating that if x satisfies the equation then x must be 5
theorem solve_for_x (x : ℕ) (h1 : pos_integer x) (h2 : equation x) : x = 5 :=
by
  sorry

end solve_for_x_l119_119777


namespace cans_in_third_bin_l119_119142

noncomputable def num_cans_in_bin (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | 4 => 11
  | 5 => 16
  | _ => sorry

theorem cans_in_third_bin :
  num_cans_in_bin 3 = 7 :=
sorry

end cans_in_third_bin_l119_119142


namespace monthly_salary_is_correct_l119_119545

noncomputable def man's_salary : ℝ :=
  let S : ℝ := 6500
  S

theorem monthly_salary_is_correct (S : ℝ) (h1 : S * 0.20 = S * 0.20) (h2 : S * 0.80 * 1.20 + 260 = S):
  S = man's_salary :=
by sorry

end monthly_salary_is_correct_l119_119545


namespace find_positive_number_l119_119713

theorem find_positive_number (x : ℝ) (h_pos : 0 < x) (h_eq : (2 / 3) * x = (49 / 216) * (1 / x)) : x = 24.5 :=
by
  sorry

end find_positive_number_l119_119713


namespace annual_income_from_investment_l119_119889

theorem annual_income_from_investment
  (I : ℝ) (P : ℝ) (R : ℝ)
  (hI : I = 6800) (hP : P = 136) (hR : R = 0.60) :
  (I / P) * 100 * R = 3000 := by
  sorry

end annual_income_from_investment_l119_119889


namespace total_raised_is_420_l119_119861

def pancake_cost : ℝ := 4.00
def bacon_cost : ℝ := 2.00
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90

theorem total_raised_is_420 : (pancake_cost * stacks_sold + bacon_cost * slices_sold) = 420.00 :=
by
  -- Proof goes here
  sorry

end total_raised_is_420_l119_119861


namespace find_x_l119_119075

theorem find_x (x : ℚ) : (8 + 12 + 24) / 3 = (16 + x) / 2 → x = 40 / 3 :=
by
  intro h
  sorry

end find_x_l119_119075


namespace value_of_some_number_l119_119970

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l119_119970


namespace final_reduced_price_l119_119387

noncomputable def original_price (P : ℝ) (Q : ℝ) : ℝ := 800 / Q

noncomputable def price_after_first_week (P : ℝ) : ℝ := 0.90 * P
noncomputable def price_after_second_week (price1 : ℝ) : ℝ := 0.85 * price1
noncomputable def price_after_third_week (price2 : ℝ) : ℝ := 0.80 * price2

noncomputable def reduced_price (P : ℝ) : ℝ :=
  let price1 := price_after_first_week P
  let price2 := price_after_second_week price1
  price_after_third_week price2

theorem final_reduced_price :
  ∃ P Q : ℝ, 
    800 = Q * P ∧
    800 = (Q + 5) * reduced_price P ∧
    abs (reduced_price P - 62.06) < 0.01 :=
by
  sorry

end final_reduced_price_l119_119387


namespace apple_distribution_l119_119053

theorem apple_distribution (total_apples : ℝ)
  (time_anya time_varya time_sveta total_time : ℝ)
  (work_anya work_varya work_sveta : ℝ) :
  total_apples = 10 →
  time_anya = 20 →
  time_varya = 35 →
  time_sveta = 45 →
  total_time = (time_anya + time_varya + time_sveta) →
  work_anya = (total_apples * time_anya / total_time) →
  work_varya = (total_apples * time_varya / total_time) →
  work_sveta = (total_apples * time_sveta / total_time) →
  work_anya = 2 ∧ work_varya = 3.5 ∧ work_sveta = 4.5 := by
  sorry

end apple_distribution_l119_119053


namespace largest_angle_of_triangle_l119_119246

theorem largest_angle_of_triangle 
  (α β γ : ℝ) 
  (h1 : α = 60) 
  (h2 : β = 70) 
  (h3 : α + β + γ = 180) : 
  max α (max β γ) = 70 := 
by 
  sorry

end largest_angle_of_triangle_l119_119246


namespace sum_sequence_a_b_eq_1033_l119_119251

def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

theorem sum_sequence_a_b_eq_1033 : 
  (a (b 1)) + (a (b 2)) + (a (b 3)) + (a (b 4)) + (a (b 5)) + 
  (a (b 6)) + (a (b 7)) + (a (b 8)) + (a (b 9)) + (a (b 10)) = 1033 := by
  sorry

end sum_sequence_a_b_eq_1033_l119_119251


namespace cylinder_surface_area_l119_119728

namespace SurfaceAreaProof

variables (a b : ℝ)

theorem cylinder_surface_area (a b : ℝ) :
  (2 * Real.pi * a * b) = (2 * Real.pi * a * b) :=
by sorry

end SurfaceAreaProof

end cylinder_surface_area_l119_119728


namespace multiply_expression_l119_119829

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l119_119829


namespace probability_of_selecting_cooking_l119_119285

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119285


namespace total_students_l119_119108

theorem total_students (N : ℕ) (num_provincial : ℕ) (sample_provincial : ℕ) 
(sample_experimental : ℕ) (sample_regular : ℕ) (sample_sino_canadian : ℕ) 
(ratio : ℕ) 
(h1 : num_provincial = 96) 
(h2 : sample_provincial = 12) 
(h3 : sample_experimental = 21) 
(h4 : sample_regular = 25) 
(h5 : sample_sino_canadian = 43) 
(h6 : ratio = num_provincial / sample_provincial) 
(h7 : ratio = 8) 
: N = ratio * (sample_provincial + sample_experimental + sample_regular + sample_sino_canadian) := 
by 
  sorry

end total_students_l119_119108


namespace remaining_area_l119_119137

theorem remaining_area (x : ℝ) :
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  A_large - A_hole = - x^2 + 22 * x + 52 := by
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  have hA_large : A_large = 2 * x^2 + 20 * x + 48 := by
    sorry
  have hA_hole : A_hole = 3 * x^2 - 2 * x - 4 := by
    sorry
  calc
    A_large - A_hole = (2 * x^2 + 20 * x + 48) - (3 * x^2 - 2 * x - 4) := by
      rw [hA_large, hA_hole]
    _ = -x^2 + 22 * x + 52 := by
      ring

end remaining_area_l119_119137


namespace solution_set_of_abs_inequality_l119_119875

theorem solution_set_of_abs_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < 2.5 :=
by
  sorry

end solution_set_of_abs_inequality_l119_119875


namespace side_length_irrational_l119_119109

theorem side_length_irrational (s : ℝ) (h : s^2 = 3) : ¬∃ (r : ℚ), s = r := by
  sorry

end side_length_irrational_l119_119109


namespace probability_of_selecting_cooking_l119_119341

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119341


namespace range_of_x_l119_119978

theorem range_of_x (x : ℝ) : 
  (∀ (m : ℝ), |m| ≤ 1 → x^2 - 2 > m * x) ↔ (x < -2 ∨ x > 2) :=
by 
  sorry

end range_of_x_l119_119978


namespace train_speed_in_m_per_s_l119_119275

-- Define the given train speed in kmph
def train_speed_kmph : ℕ := 72

-- Define the conversion factor from kmph to m/s
def km_per_hour_to_m_per_second (speed_in_kmph : ℕ) : ℕ := (speed_in_kmph * 1000) / 3600

-- State the theorem
theorem train_speed_in_m_per_s (h : train_speed_kmph = 72) : km_per_hour_to_m_per_second train_speed_kmph = 20 := by
  sorry

end train_speed_in_m_per_s_l119_119275


namespace prove_identical_numbers_l119_119485

variable {x y : ℝ}

theorem prove_identical_numbers (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + (1 / y^2) = y + (1 / x^2))
    (h2 : y^2 + (1 / x) = x^2 + (1 / y)) : x = y :=
by 
  sorry

end prove_identical_numbers_l119_119485


namespace max_xyz_l119_119637

theorem max_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) 
(h5 : x^2 + y^2 + z^2 = x * z + y * z + x * y) : xyz ≤ (8 / 27) :=
sorry

end max_xyz_l119_119637


namespace original_grain_amount_l119_119388

def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

theorem original_grain_amount : grain_spilled + grain_remaining = 50870 :=
by
  sorry

end original_grain_amount_l119_119388


namespace time_elephants_l119_119117

def total_time := 130
def time_seals := 13
def time_penguins := 8 * time_seals

theorem time_elephants : total_time - (time_seals + time_penguins) = 13 :=
by
  sorry

end time_elephants_l119_119117


namespace A_leaves_after_2_days_l119_119721

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 30
noncomputable def C_work_rate : ℚ := 1 / 10
noncomputable def C_days_work : ℚ := 4
noncomputable def total_days_work : ℚ := 15

theorem A_leaves_after_2_days (x : ℚ) : 
  2 / 5 + x / 12 + (15 - x) / 30 = 1 → x = 2 :=
by
  intro h
  sorry

end A_leaves_after_2_days_l119_119721


namespace difference_of_areas_l119_119940

-- Defining the side length of the square
def square_side_length : ℝ := 8

-- Defining the side lengths of the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 5

-- Defining the area functions
def area_of_square (side_length : ℝ) : ℝ := side_length * side_length
def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ := length * width

-- Stating the theorem
theorem difference_of_areas :
  area_of_square square_side_length - area_of_rectangle rectangle_length rectangle_width = 14 :=
by
  sorry

end difference_of_areas_l119_119940


namespace least_positive_integer_divisors_l119_119556

theorem least_positive_integer_divisors (n m k : ℕ) (h₁ : (∀ d : ℕ, d ∣ n ↔ d ≤ 2023))
(h₂ : n = m * 6^k) (h₃ : (∀ d : ℕ, d ∣ 6 → ¬(d ∣ m))) : m + k = 80 :=
sorry

end least_positive_integer_divisors_l119_119556


namespace part1_part2_l119_119809

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ := Real.tan ((x / 2) - (Real.pi / 3))

-- Part (1)
theorem part1 : f (5 * Real.pi / 2) = Real.sqrt 3 - 2 :=
by
  sorry

-- Part (2)
theorem part2 (k : ℤ) : { x : ℝ | f x ≤ Real.sqrt 3 } = 
  {x | ∃ (k : ℤ), 2 * k * Real.pi - Real.pi / 3 < x ∧ x ≤ 2 * k * Real.pi + 4 * Real.pi / 3} :=
by
  sorry

end ProofProblem

end part1_part2_l119_119809


namespace probability_of_selecting_cooking_is_one_fourth_l119_119377

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119377


namespace total_amount_raised_l119_119859

-- Definitions based on conditions
def PancakeCost : ℕ := 4
def BaconCost : ℕ := 2
def NumPancakesSold : ℕ := 60
def NumBaconSold : ℕ := 90

-- Lean statement proving that the total amount raised is $420
theorem total_amount_raised : (NumPancakesSold * PancakeCost) + (NumBaconSold * BaconCost) = 420 := by
  -- Since we are not required to prove, we use sorry here
  sorry

end total_amount_raised_l119_119859


namespace increasing_condition_l119_119428

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 2 * x - 2 * a

-- Prove that f is increasing on the interval [2, +∞) if and only if a ≤ 2
theorem increasing_condition (a : ℝ) : (∀ x ≥ 2, f' x a ≥ 0) ↔ (a ≤ 2) := 
sorry

end increasing_condition_l119_119428


namespace cook_stole_the_cookbook_l119_119525

-- Define the suspects
inductive Suspect
| CheshireCat
| Duchess
| Cook
deriving DecidableEq, Repr

-- Define the predicate for lying
def lied (s : Suspect) : Prop := sorry

-- Define the conditions
def conditions (thief : Suspect) : Prop :=
  lied thief ∧
  ((∀ s : Suspect, s ≠ thief → lied s) ∨ (∀ s : Suspect, s ≠ thief → ¬lied s))

-- Define the goal statement
theorem cook_stole_the_cookbook : conditions Suspect.Cook :=
sorry

end cook_stole_the_cookbook_l119_119525


namespace find_n_l119_119602

theorem find_n : ∃ n : ℕ, (∃ A B : ℕ, A ≠ B ∧ 10^(n-1) ≤ A ∧ A < 10^n ∧ 10^(n-1) ≤ B ∧ B < 10^n ∧ (10^n * A + B) % (10^n * B + A) = 0) ↔ n % 6 = 3 :=
by
  sorry

end find_n_l119_119602


namespace polynomials_common_zero_k_l119_119610

theorem polynomials_common_zero_k
  (k : ℝ) :
  (∃ x : ℝ, (1988 * x^2 + k * x + 8891 = 0) ∧ (8891 * x^2 + k * x + 1988 = 0)) ↔ (k = 10879 ∨ k = -10879) :=
sorry

end polynomials_common_zero_k_l119_119610


namespace opposite_neg_half_l119_119082

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l119_119082


namespace maximize_squares_l119_119878

theorem maximize_squares (a b : ℕ) (k : ℕ) :
  (a ≠ b) →
  ((∃ (k : ℤ), k ≠ 1 ∧ b = k^2) ↔ 
   (∃ (c₁ c₂ c₃ : ℕ), a * (b + 8) = c₁^2 ∧ b * (a + 8) = c₂^2 ∧ a * b = c₃^2 
     ∧ a = 1)) :=
by { sorry }

end maximize_squares_l119_119878


namespace max_n_positive_l119_119461

theorem max_n_positive (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : S 15 > 0)
  (h2 : S 16 < 0)
  (hs1 : S 15 = 15 * (a 8))
  (hs2 : S 16 = 8 * (a 8 + a 9)) :
  (∀ n, a n > 0 → n ≤ 8) :=
by {
    sorry
}

end max_n_positive_l119_119461


namespace some_number_value_correct_l119_119974

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l119_119974


namespace common_ratio_of_geometric_seq_l119_119666

variable {α : Type} [LinearOrderedField α] 
variables (a d : α) (h₁ : d ≠ 0) (h₂ : (a + 2 * d) / (a + d) = (a + 5 * d) / (a + 2 * d))

theorem common_ratio_of_geometric_seq : (a + 2 * d) / (a + d) = 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l119_119666


namespace part1_part2_l119_119436

-- Part 1: Prove that the range of values for k is k ≤ 1/4
theorem part1 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : ∀ x0 : ℝ, f x0 ≥ |k+3| - |k-2|)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  k ≤ 1/4 := 
sorry

-- Part 2: Show that the minimum value of m+n is 8/3
theorem part2 (f : ℝ → ℝ) (m n : ℝ) 
  (h1 : ∀ x : ℝ, f x ≥ 1/m + 1/n)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  m + n ≥ 8/3 := 
sorry

end part1_part2_l119_119436


namespace clock_angle_at_seven_l119_119692

/--
The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees.
-/
theorem clock_angle_at_seven : 
  let full_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle / hours_on_clock
  let hour_at_seven := 7
  let angle := hour_at_seven * degrees_per_hour
  in if angle <= full_circle / 2 then angle = 150 else full_circle - angle = 150 :=
begin
  -- Full circle in degrees
  let full_circle := 360,
  -- Hours on a clock
  let hours_on_clock := 12,
  -- Degrees per hour mark
  let degrees_per_hour := full_circle / hours_on_clock,
  -- Position of the hour hand at 7 o'clock
  let hour_at_seven := 7,
  -- Angle of the hour hand (clockwise)
  let angle := hour_at_seven * degrees_per_hour,
  -- The smaller angle is the one considered
  suffices h : full_circle - angle = 150,
  exact h,
  sorry
end

end clock_angle_at_seven_l119_119692


namespace opposite_of_neg_half_is_half_l119_119096

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l119_119096


namespace no_two_distinct_real_roots_l119_119115

-- Definitions of the conditions and question in Lean 4
theorem no_two_distinct_real_roots (a : ℝ) (h : a ≥ 1) : ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*x1 + a = 0) ∧ (x2^2 - 2*x2 + a = 0) :=
sorry

end no_two_distinct_real_roots_l119_119115


namespace multiply_expand_l119_119834

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l119_119834


namespace find_t_l119_119986

open Real

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem find_t (t : ℝ) 
  (area_eq_50 : area_of_triangle 3 15 15 0 0 t = 50) :
  t = 325 / 12 ∨ t = 125 / 12 := 
sorry

end find_t_l119_119986


namespace roof_collapse_days_l119_119551

-- Definitions based on the conditions
def roof_capacity_pounds : ℕ := 500
def leaves_per_pound : ℕ := 1000
def leaves_per_day : ℕ := 100

-- Statement of the problem and the result
theorem roof_collapse_days :
  let total_leaves := roof_capacity_pounds * leaves_per_pound in
  let days := total_leaves / leaves_per_day in
  days = 5000 :=
by
  -- To be proven, so we use sorry for now
  sorry

end roof_collapse_days_l119_119551


namespace probability_of_selecting_cooking_l119_119284

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119284


namespace smallest_number_gt_sum_digits_1755_l119_119604

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end smallest_number_gt_sum_digits_1755_l119_119604


namespace opposite_of_half_l119_119089

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l119_119089


namespace candy_problem_l119_119801

theorem candy_problem (n : ℕ) (h : n ∈ [2, 5, 9, 11, 14]) : ¬(23 - n) % 3 ≠ 0 → n = 9 := by
  sorry

end candy_problem_l119_119801


namespace player_A_min_score_l119_119056

theorem player_A_min_score (A B : ℕ) (hA_first_move : A = 1) (hB_next_move : B = 2) : 
  ∃ k : ℕ, k = 64 :=
by
  sorry

end player_A_min_score_l119_119056


namespace values_of_k_real_equal_roots_l119_119742

theorem values_of_k_real_equal_roots (k : ℝ) :
  (∀ x : ℝ, 3 * x^2 - (k + 2) * x + 12 = 0 → x * x = 0) ↔ (k = 10 ∨ k = -14) :=
by
  sorry

end values_of_k_real_equal_roots_l119_119742


namespace compare_sums_l119_119175

theorem compare_sums (a b c : ℝ) (h : a > b ∧ b > c) : a^2 * b + b^2 * c + c^2 * a > a * b^2 + b * c^2 + c * a^2 := by
  sorry

end compare_sums_l119_119175


namespace team_leaders_lcm_l119_119735

/-- Amanda, Brian, Carla, and Derek are team leaders rotating every
    5, 8, 10, and 12 weeks respectively. Given that this week they all are leading
    projects together, prove that they will all lead projects together again in 120 weeks. -/
theorem team_leaders_lcm :
  Nat.lcm (Nat.lcm 5 8) (Nat.lcm 10 12) = 120 := 
  by
  sorry

end team_leaders_lcm_l119_119735


namespace opposite_of_num_l119_119102

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l119_119102


namespace opposite_neg_half_l119_119083

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l119_119083


namespace negation_example_l119_119502

theorem negation_example : (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_example_l119_119502


namespace remainder_sum_of_squares_25_mod_6_l119_119122

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem remainder_sum_of_squares_25_mod_6 :
  (sum_of_squares 25) % 6 = 5 :=
by
  sorry

end remainder_sum_of_squares_25_mod_6_l119_119122


namespace value_of_some_number_l119_119969

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l119_119969


namespace probability_of_selecting_cooking_l119_119290

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119290


namespace arithmetic_sequence_50th_term_l119_119882

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 50 = 199 :=
by
  sorry

end arithmetic_sequence_50th_term_l119_119882


namespace cost_of_3600_pens_l119_119543

-- Define the conditions
def cost_per_200_pens : ℕ := 50
def pens_bought : ℕ := 3600

-- Define a theorem to encapsulate our question and provide the necessary definitions
theorem cost_of_3600_pens : cost_per_200_pens / 200 * pens_bought = 900 := by sorry

end cost_of_3600_pens_l119_119543


namespace algebraic_expression_value_zero_l119_119960

theorem algebraic_expression_value_zero (a b : ℝ) (h : a - b = 2) : (a^3 - 2 * a^2 * b + a * b^2 - 4 * a = 0) :=
sorry

end algebraic_expression_value_zero_l119_119960


namespace opposite_of_half_l119_119091

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l119_119091


namespace mark_repayment_l119_119639

noncomputable def totalDebt (days : ℕ) : ℝ :=
  if days < 3 then
    20 + (20 * 0.10 * days)
  else
    35 + (20 * 0.10 * 3) + (35 * 0.10 * (days - 3))

theorem mark_repayment :
  ∃ (x : ℕ), totalDebt x ≥ 70 ∧ x = 12 :=
by
  -- Use this theorem statement to prove the corresponding lean proof
  sorry

end mark_repayment_l119_119639


namespace mode_of_dataset_l119_119249

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end mode_of_dataset_l119_119249


namespace probability_of_selecting_cooking_l119_119292

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119292


namespace greatest_sum_first_quadrant_l119_119256

theorem greatest_sum_first_quadrant (x y : ℤ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_circle : x^2 + y^2 = 49) : x + y ≤ 7 :=
sorry

end greatest_sum_first_quadrant_l119_119256


namespace opposite_of_neg_half_is_half_l119_119092

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l119_119092


namespace original_number_of_men_l119_119276

theorem original_number_of_men (x : ℕ) (h : 10 * x = 7 * (x + 10)) : x = 24 := 
by 
  -- Add your proof here 
  sorry

end original_number_of_men_l119_119276


namespace student_B_speed_l119_119906

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l119_119906


namespace find_L_l119_119601

noncomputable def L_value : ℕ := 3

theorem find_L
  (a b : ℕ)
  (cows : ℕ := 5 * b)
  (chickens : ℕ := 5 * a + 7)
  (insects : ℕ := b ^ (a - 5))
  (legs_cows : ℕ := 4 * cows)
  (legs_chickens : ℕ := 2 * chickens)
  (legs_insects : ℕ :=  6 * insects)
  (total_legs : ℕ := legs_cows + legs_chickens + legs_insects) 
  (h1 : cows = insects)
  (h2 : total_legs = (L_value * 100 + L_value * 10 + L_value) + 1) :
  L_value = 3 := sorry

end find_L_l119_119601


namespace max_ballpoint_pens_l119_119534

def ballpoint_pen_cost : ℕ := 10
def gel_pen_cost : ℕ := 30
def fountain_pen_cost : ℕ := 60
def total_pens : ℕ := 20
def total_cost : ℕ := 500

theorem max_ballpoint_pens : ∃ (x y z : ℕ), 
  x + y + z = total_pens ∧ 
  ballpoint_pen_cost * x + gel_pen_cost * y + fountain_pen_cost * z = total_cost ∧ 
  1 ≤ x ∧ 
  1 ≤ y ∧
  1 ≤ z ∧
  ∀ x', ((∃ y' z', x' + y' + z' = total_pens ∧ 
                    ballpoint_pen_cost * x' + gel_pen_cost * y' + fountain_pen_cost * z' = total_cost ∧ 
                    1 ≤ x' ∧ 
                    1 ≤ y' ∧
                    1 ≤ z') → x' ≤ x) :=
  sorry

end max_ballpoint_pens_l119_119534


namespace area_of_ellipse_l119_119626

def endpoints_of_major_axis : (ℝ × ℝ) := (-10, 2)
def endpoints_of_major_axis' : (ℝ × ℝ) := (10, 2)
def passes_through_points_1 : (ℝ × ℝ) := (8, 6)
def passes_through_points_2 : (ℝ × ℝ) := (-8, -2)
def semi_major_axis_length : ℝ := 10
def semi_minor_axis_length : ℝ := 20 / 3

theorem area_of_ellipse 
  (a_eq : (endpoints_of_major_axis.fst + endpoints_of_major_axis'.fst) / 2 = 0)
  (b_eq : (endpoints_of_major_axis.snd + endpoints_of_major_axis'.snd) / 2 = 2)
  (ellipse_eq_1 : (8 - 0)^2 / semi_major_axis_length^2 + (6 - 2)^2 / semi_minor_axis_length^2 = 1)
  (ellipse_eq_2 : (-8 - 0)^2 / semi_major_axis_length^2 + (-2 - 2)^2 / semi_minor_axis_length^2 = 1) :
  let A := Real.pi * semi_major_axis_length * semi_minor_axis_length in
  A = 200 * Real.pi / 3 :=
by
  sorry

end area_of_ellipse_l119_119626


namespace tan_half_sum_pi_over_four_l119_119622

-- Define the problem conditions
variable (α : ℝ)
variable (h_cos : Real.cos α = -4 / 5)
variable (h_quad : α > π ∧ α < 3 * π / 2)

-- Define the theorem to prove
theorem tan_half_sum_pi_over_four (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_quad : α > π ∧ α < 3 * π / 2) :
  Real.tan (π / 4 + α / 2) = -1 / 2 := sorry

end tan_half_sum_pi_over_four_l119_119622


namespace SugarWeightLoss_l119_119820

noncomputable def sugar_fraction_lost : Prop :=
  let green_beans_weight := 60
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_lost := (1 / 3) * rice_weight
  let remaining_weight := 120
  let total_initial_weight := green_beans_weight + rice_weight + sugar_weight
  let total_lost := total_initial_weight - remaining_weight
  let sugar_lost := total_lost - rice_lost
  let expected_fraction := (sugar_lost / sugar_weight)
  expected_fraction = (1 / 5)

theorem SugarWeightLoss : sugar_fraction_lost := by
  sorry

end SugarWeightLoss_l119_119820


namespace sum_of_transformed_roots_equals_one_l119_119046

theorem sum_of_transformed_roots_equals_one 
  {α β γ : ℝ} 
  (hα : α^3 - α - 1 = 0) 
  (hβ : β^3 - β - 1 = 0) 
  (hγ : γ^3 - γ - 1 = 0) : 
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
sorry

end sum_of_transformed_roots_equals_one_l119_119046


namespace fraction_of_grid_covered_by_triangle_l119_119547

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))|

noncomputable def area_of_grid : ℝ := 7 * 6

noncomputable def fraction_covered : ℝ :=
  area_of_triangle (-1, 2) (3, 5) (2, 2) / area_of_grid

theorem fraction_of_grid_covered_by_triangle : fraction_covered = (3 / 28) :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l119_119547


namespace range_of_x_l119_119439

-- Defining the vectors as given in the conditions
def a (x : ℝ) : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (2, -1)

-- Defining the condition that the angle is obtuse
def is_obtuse (x : ℝ) : Prop := 
  let dot_product := (a x).1 * b.1 + (a x).2 * b.2
  dot_product < 0

-- Defining the condition that vectors are not in opposite directions
def not_opposite_directions (x : ℝ) : Prop := x ≠ -6

-- Proving the required range of x
theorem range_of_x (x : ℝ) :
  is_obtuse x → not_opposite_directions x → x < 3 / 2 :=
sorry

end range_of_x_l119_119439


namespace opposite_neg_half_l119_119085

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l119_119085


namespace find_other_person_money_l119_119051

noncomputable def other_person_money (mias_money : ℕ) : ℕ :=
  let x := (mias_money - 20) / 2
  x

theorem find_other_person_money (mias_money : ℕ) (h_mias_money : mias_money = 110) : 
  other_person_money mias_money = 45 := by
  sorry

end find_other_person_money_l119_119051


namespace parabola_equation_exists_line_m_equation_exists_l119_119901

noncomputable def problem_1 : Prop :=
  ∃ (p : ℝ), p > 0 ∧ (∀ (x y : ℝ), x^2 = 2 * p * y → y = x^2 / (2 * p)) ∧ 
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 = 2 * p * y1 → x2^2 = 2 * p * y2 → 
    (y1 + y2 = 8 - p) ∧ ((y1 + y2) / 2 = 3) → p = 2)

noncomputable def problem_2 : Prop :=
  ∃ (k : ℝ), (k^2 = 1 / 4) ∧ (∀ (x : ℝ), (x^2 - 4 * k * x - 24 = 0) → 
    (∃ (x1 x2 : ℝ), x1 + x2 = 4 * k ∧ x1 * x2 = -24)) ∧
  (∀ (x1 x2 : ℝ), x1^2 = 4 * (k * x1 + 6) ∧ x2^2 = 4 * (k * x2 + 6) → 
    ∀ (x3 x4 : ℝ), (x1 * x2) ^ 2 - 4 * ((x1 + x2) ^ 2 - 2 * x1 * x2) + 16 + 16 * x1 * x2 = 0 → 
    (k = 1 / 2 ∨ k = -1 / 2))

theorem parabola_equation_exists : problem_1 :=
by {
  sorry
}

theorem line_m_equation_exists : problem_2 :=
by {
  sorry
}

end parabola_equation_exists_line_m_equation_exists_l119_119901


namespace irene_overtime_pay_per_hour_l119_119212

def irene_base_pay : ℝ := 500
def irene_base_hours : ℕ := 40
def irene_total_hours_last_week : ℕ := 50
def irene_total_income_last_week : ℝ := 700

theorem irene_overtime_pay_per_hour :
  (irene_total_income_last_week - irene_base_pay) / (irene_total_hours_last_week - irene_base_hours) = 20 := 
by
  sorry

end irene_overtime_pay_per_hour_l119_119212


namespace functional_equation_solution_l119_119937

/-- For all functions f: ℝ → ℝ, that satisfy the given functional equation -/
def functional_equation (f: ℝ → ℝ) : Prop :=
  ∀ x y: ℝ, f (x + y * f (x + y)) = y ^ 2 + f (x * f (y + 1))

/-- The solution to the functional equation is f(x) = x -/
theorem functional_equation_solution :
  ∀ f: ℝ → ℝ, functional_equation f → (∀ x: ℝ, f x = x) :=
by
  intros f h x
  sorry

end functional_equation_solution_l119_119937


namespace repeating_decimal_sum_l119_119566

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end repeating_decimal_sum_l119_119566


namespace sum_squares_l119_119808

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end sum_squares_l119_119808


namespace arithmetic_sequence_sum_l119_119457

theorem arithmetic_sequence_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 9 = a_n 4 + a_n 5 + a_n 6 + 72)
  (h2 : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (h3 : ∀ n, a_n (n+1) - a_n n = d)
  (h4 : a_n 1 + a_n 9 = a_n 3 + a_n 7)
  (h5 : a_n 3 + a_n 7 = a_n 4 + a_n 6)
  (h6 : a_n 4 + a_n 6 = 2 * a_n 5) : 
  a_n 3 + a_n 7 = 24 := 
sorry

end arithmetic_sequence_sum_l119_119457


namespace find_k_minus_r_l119_119259

theorem find_k_minus_r : 
  ∃ (k r : ℕ), k > 1 ∧ r < k ∧ 
  (1177 % k = r) ∧ (1573 % k = r) ∧ (2552 % k = r) ∧ 
  (k - r = 11) :=
sorry

end find_k_minus_r_l119_119259


namespace probability_select_cooking_l119_119323

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119323


namespace three_a_greater_three_b_l119_119180

variable (a b : ℝ)

theorem three_a_greater_three_b (h : a > b) : 3 * a > 3 * b :=
  sorry

end three_a_greater_three_b_l119_119180


namespace find_room_length_l119_119797

theorem find_room_length (w : ℝ) (A : ℝ) (h_w : w = 8) (h_A : A = 96) : (A / w = 12) :=
by
  rw [h_w, h_A]
  norm_num

end find_room_length_l119_119797


namespace solve_for_x_l119_119621

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 :=
by
  sorry

end solve_for_x_l119_119621


namespace multiply_polynomials_l119_119824

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l119_119824


namespace f_zero_f_odd_solve_inequality_l119_119476

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom increasing_on_nonneg : ∀ {x y : ℝ}, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem f_zero : f 0 = 0 :=
by sorry

theorem f_odd (x : ℝ) : f (-x) = -f x :=
by sorry

theorem solve_inequality {x : ℝ} (h : 0 < x) : f (Real.log x / Real.log 10 - 1) < 0 ↔ 0 < x ∧ x < 10 :=
by sorry

end f_zero_f_odd_solve_inequality_l119_119476


namespace range_of_a_l119_119981

theorem range_of_a (a : ℝ) : (forall x : ℝ, (a-3) * x > 1 → x < 1 / (a-3)) → a < 3 :=
by
  sorry

end range_of_a_l119_119981


namespace probability_cooking_selected_l119_119314

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119314


namespace trip_duration_l119_119915

noncomputable def start_time : ℕ := 11 * 60 + 25 -- 11:25 a.m. in minutes
noncomputable def end_time : ℕ := 16 * 60 + 43 + 38 / 60 -- 4:43:38 p.m. in minutes

theorem trip_duration :
  end_time - start_time = 5 * 60 + 18 := 
sorry

end trip_duration_l119_119915


namespace fg_sum_at_2_l119_119226

noncomputable def f (x : ℚ) : ℚ := (5 * x^3 + 4 * x^2 - 2 * x + 3) / (x^3 - 2 * x^2 + 3 * x + 1)
noncomputable def g (x : ℚ) : ℚ := x^2 - 2

theorem fg_sum_at_2 : f (g 2) + g (f 2) = 468 / 7 := by
  sorry

end fg_sum_at_2_l119_119226


namespace multiply_and_simplify_l119_119847
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l119_119847


namespace number_of_classes_min_wins_for_class2101_l119_119114

-- Proof Problem for Q1
theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 := sorry

-- Proof Problem for Q2
theorem min_wins_for_class2101 (y : ℕ) (h : y + (9 - y) = 9 ∧ 2 * y + (9 - y) >= 14) : y >= 5 := sorry

end number_of_classes_min_wins_for_class2101_l119_119114


namespace mode_of_dataset_l119_119250

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end mode_of_dataset_l119_119250


namespace bicycle_speed_B_l119_119904

theorem bicycle_speed_B 
  (distance : ℝ := 12)
  (ratio : ℝ := 1.2)
  (time_diff : ℝ := 1 / 6) : 
  ∃ (B_speed : ℝ), B_speed = 12 :=
by
  let A_speed := ratio * B_speed
  have eqn : distance / B_speed - time_diff = distance / A_speed := sorry
  exact ⟨12, sorry⟩

end bicycle_speed_B_l119_119904


namespace empty_subset_singleton_l119_119393

theorem empty_subset_singleton : (∅ ⊆ ({0} : Set ℕ)) = true :=
by sorry

end empty_subset_singleton_l119_119393


namespace repeating_decimal_sum_in_lowest_terms_l119_119573

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l119_119573


namespace fishing_problem_l119_119031

theorem fishing_problem :
  ∃ (x y : ℕ), 
    (x + y = 70) ∧ 
    (∃ k : ℕ, x = 9 * k) ∧ 
    (∃ m : ℕ, y = 17 * m) ∧ 
    x = 36 ∧ 
    y = 34 := 
by
  sorry

end fishing_problem_l119_119031


namespace gcd_18_eq_6_l119_119169

theorem gcd_18_eq_6 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 200) : (nat.gcd 18 n = 6) ↔ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 22 ∧ n = 6 * k ∧ ¬(n = 18 * (n / 18))) :=
begin
  sorry
end

end gcd_18_eq_6_l119_119169


namespace opposite_of_half_l119_119088

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l119_119088


namespace combined_salaries_correct_l119_119675

noncomputable def combined_salaries_BCDE (A B C D E : ℕ) : Prop :=
  (A = 8000) →
  ((A + B + C + D + E) / 5 = 8600) →
  (B + C + D + E = 35000)

theorem combined_salaries_correct 
  (A B C D E : ℕ) 
  (hA : A = 8000) 
  (havg : (A + B + C + D + E) / 5 = 8600) : 
  B + C + D + E = 35000 :=
sorry

end combined_salaries_correct_l119_119675


namespace total_payment_correct_l119_119399

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end total_payment_correct_l119_119399


namespace probability_of_cooking_l119_119366

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119366


namespace xiaoming_selects_cooking_probability_l119_119328

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119328


namespace find_number_l119_119123

theorem find_number (x : ℝ) (h : 3034 - x / 200.4 = 3029) : x = 1002 :=
sorry

end find_number_l119_119123


namespace vertex_angle_of_cone_l119_119160

theorem vertex_angle_of_cone (α : ℝ) (h_α : 0 < α ∧ α < 2 * π) :
  let vertex_angle := 2 * real.arcsin (α / (2 * real.pi)) in 
  true := by
  sorry

end vertex_angle_of_cone_l119_119160


namespace ball_probability_l119_119034

theorem ball_probability :
  ∀ (total_balls red_balls white_balls : ℕ),
  total_balls = 10 → red_balls = 6 → white_balls = 4 →
  -- Given conditions: Total balls, red balls, and white balls.
  -- First ball drawn is red
  ∀ (first_ball_red : true),
  -- Prove that the probability of the second ball being red is 5/9.
  (red_balls - 1) / (total_balls - 1) = 5/9 :=
by
  intros total_balls red_balls white_balls h_total h_red h_white first_ball_red
  sorry

end ball_probability_l119_119034


namespace hermione_utility_l119_119014

theorem hermione_utility (h : ℕ) : (h * (10 - h) = (4 - h) * (h + 2)) ↔ h = 4 := by
  sorry

end hermione_utility_l119_119014


namespace atleast_one_genuine_l119_119392

noncomputable def products : ℕ := 12
noncomputable def genuine : ℕ := 10
noncomputable def defective : ℕ := 2
noncomputable def selected : ℕ := 3

theorem atleast_one_genuine :
  (selected = 3) →
  (genuine + defective = 12) →
  (genuine ≥ 3) →
  (selected ≥ 1) →
  ∃ g d : ℕ, g + d = 3 ∧ g > 0 ∧ d ≤ 2 :=
by
  -- Proof will go here.
  sorry

end atleast_one_genuine_l119_119392


namespace F_minimum_value_neg_inf_to_0_l119_119021

variable (f g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) := ∀ x, h (-x) = - (h x)

theorem F_minimum_value_neg_inf_to_0 
  (hf_odd : is_odd f) 
  (hg_odd : is_odd g)
  (hF_max : ∀ x > 0, f x + g x + 2 ≤ 8) 
  (hF_reaches_max : ∃ x > 0, f x + g x + 2 = 8) :
  ∀ x < 0, f x + g x + 2 ≥ -4 :=
by
  sorry

end F_minimum_value_neg_inf_to_0_l119_119021


namespace probability_cooking_selected_l119_119309

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119309


namespace parallelogram_ratio_l119_119664

-- Definitions based on given conditions
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_ratio (A : ℝ) (B : ℝ) (h : ℝ) (H1 : A = 242) (H2 : B = 11) (H3 : A = parallelogram_area B h) :
  h / B = 2 :=
by
  -- Proof goes here
  sorry

end parallelogram_ratio_l119_119664


namespace proof_problem_l119_119065

variables {a b c : Real}

theorem proof_problem (h1 : a < 0) (h2 : |a| < |b|) (h3 : |b| < |c|) (h4 : b < 0) :
  (|a * b| < |b * c|) ∧ (a * c < |b * c|) ∧ (|a + b| < |b + c|) :=
by
  sorry

end proof_problem_l119_119065


namespace prob_fourth_black_ball_is_half_l119_119719

-- Define the conditions
def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_black_balls

-- The theorem stating that the probability of drawing a black ball on the fourth draw is 1/2
theorem prob_fourth_black_ball_is_half : 
  (num_black_balls : ℚ) / (total_balls : ℚ) = 1 / 2 :=
by
  sorry

end prob_fourth_black_ball_is_half_l119_119719


namespace smallest_sum_ending_2050306_l119_119124

/--
Given nine consecutive natural numbers starting at n,
prove that the smallest sum of these nine numbers ending in 2050306 is 22050306.
-/
theorem smallest_sum_ending_2050306 
  (n : ℕ) 
  (hn : ∃ m : ℕ, 9 * m = (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ∧ 
                 (9 * m) % 10^7 = 2050306) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) = 22050306 := 
sorry

end smallest_sum_ending_2050306_l119_119124


namespace apple_count_difference_l119_119686

theorem apple_count_difference
    (original_green : ℕ)
    (additional_green : ℕ)
    (red_more_than_green : ℕ)
    (green_now : ℕ := original_green + additional_green)
    (red_now : ℕ := original_green + red_more_than_green)
    (difference : ℕ := green_now - red_now)
    (h_original_green : original_green = 32)
    (h_additional_green : additional_green = 340)
    (h_red_more_than_green : red_more_than_green = 200) :
    difference = 140 :=
by
  sorry

end apple_count_difference_l119_119686


namespace find_f_at_9_over_2_l119_119471

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end find_f_at_9_over_2_l119_119471


namespace jane_change_l119_119221

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l119_119221


namespace total_payment_is_53_l119_119402

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end total_payment_is_53_l119_119402


namespace add_pure_chocolate_to_achieve_percentage_l119_119958

/--
Given:
    Initial amount of chocolate topping: 620 ounces.
    Initial chocolate percentage: 10%.
    Desired total weight of the final mixture: 1000 ounces.
    Desired chocolate percentage in the final mixture: 70%.
Prove:
    The amount of pure chocolate to be added to achieve the desired mixture is 638 ounces.
-/
theorem add_pure_chocolate_to_achieve_percentage :
  ∃ x : ℝ,
    0.10 * 620 + x = 0.70 * 1000 ∧
    x = 638 :=
by
  sorry

end add_pure_chocolate_to_achieve_percentage_l119_119958


namespace artwork_collection_l119_119680

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l119_119680


namespace people_per_seat_l119_119653

def ferris_wheel_seats : ℕ := 4
def total_people_riding : ℕ := 20

theorem people_per_seat : total_people_riding / ferris_wheel_seats = 5 := by
  sorry

end people_per_seat_l119_119653


namespace artworks_collected_l119_119682

theorem artworks_collected (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) :
  students = 15 →
  artworks_per_student_per_quarter = 2 →
  quarters_per_year = 4 →
  num_years = 2 →
  (students * artworks_per_student_per_quarter * quarters_per_year * num_years) = 240 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end artworks_collected_l119_119682


namespace screen_time_morning_l119_119066

def total_screen_time : ℕ := 120
def evening_screen_time : ℕ := 75
def morning_screen_time : ℕ := 45

theorem screen_time_morning : total_screen_time - evening_screen_time = morning_screen_time := by
  sorry

end screen_time_morning_l119_119066


namespace geometric_sequences_identical_l119_119881

theorem geometric_sequences_identical
  (a_0 q r : ℝ)
  (a_n b_n c_n : ℕ → ℝ)
  (H₁ : ∀ n, a_n n = a_0 * q ^ n)
  (H₂ : ∀ n, b_n n = a_0 * r ^ n)
  (H₃ : ∀ n, c_n n = a_n n + b_n n)
  (H₄ : ∃ s : ℝ, ∀ n, c_n n = c_n 0 * s ^ n):
  ∀ n, a_n n = b_n n := sorry

end geometric_sequences_identical_l119_119881


namespace total_animal_eyes_l119_119786

def num_snakes := 18
def num_alligators := 10
def eyes_per_snake := 2
def eyes_per_alligator := 2

theorem total_animal_eyes : 
  (num_snakes * eyes_per_snake) + (num_alligators * eyes_per_alligator) = 56 :=
by 
  sorry

end total_animal_eyes_l119_119786


namespace math_problem_l119_119442

theorem math_problem (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a + b^2 + c^3 = 14 :=
by
  sorry

end math_problem_l119_119442


namespace dinosaur_book_cost_l119_119632

theorem dinosaur_book_cost (D : ℕ) : 
  (11 + D + 7 = 37) → (D = 19) := 
by 
  intro h
  sorry

end dinosaur_book_cost_l119_119632


namespace patty_weeks_without_chores_correct_l119_119484

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l119_119484


namespace unit_digit_of_12_pow_100_l119_119714

def unit_digit_pow (a: ℕ) (n: ℕ) : ℕ :=
  (a ^ n) % 10

theorem unit_digit_of_12_pow_100 : unit_digit_pow 12 100 = 6 := by
  sorry

end unit_digit_of_12_pow_100_l119_119714


namespace probability_of_cooking_l119_119365

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

def num_courses : ℕ := courses.length

def chosen_course : String := "cooking"

theorem probability_of_cooking : 1 / num_courses = 1 / 4 := by
  have : num_courses = 4 := rfl
  rw [this]
  norm_num

end probability_of_cooking_l119_119365


namespace sum_of_repeating_decimals_l119_119583

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l119_119583


namespace least_tablets_l119_119710

theorem least_tablets (num_A num_B : ℕ) (hA : num_A = 10) (hB : num_B = 14) :
  ∃ n, n = 12 ∧
  ∀ extracted_tablets, extracted_tablets > 0 →
    (∃ (a b : ℕ), a + b = extracted_tablets ∧ a ≥ 2 ∧ b ≥ 2) :=
by
  sorry

end least_tablets_l119_119710


namespace remainder_of_sum_l119_119821

theorem remainder_of_sum (a b : ℤ) (k m : ℤ)
  (h1 : a = 84 * k + 78)
  (h2 : b = 120 * m + 114) :
  (a + b) % 42 = 24 :=
  sorry

end remainder_of_sum_l119_119821


namespace hyperbola_equation_l119_119009

theorem hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (eccentricity : Real.sqrt 2 = b / a)
  (line_through_FP_parallel_to_asymptote : ∃ c : ℝ, c = Real.sqrt 2 * a ∧ ∀ P : ℝ × ℝ, P = (0, 4) → (P.2 - 0) / (P.1 + c) = 1) :
  (∃ (a b : ℝ), a = b ∧ (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2)) ∧
  (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2) → 
  (∃ x y : ℝ, ((x^2 / 8) - (y^2 / 8) = 1)) :=
by
  sorry

end hyperbola_equation_l119_119009


namespace smallest_prime_divisor_and_cube_root_l119_119237

theorem smallest_prime_divisor_and_cube_root (N : ℕ) (p : ℕ) (q : ℕ)
  (hN_composite : N > 1 ∧ ¬ (∃ p : ℕ, p > 1 ∧ p < N ∧ N = p))
  (h_divisor : N = p * q)
  (h_p_prime : Nat.Prime p)
  (h_min_prime : ∀ (d : ℕ), Nat.Prime d → d ∣ N → p ≤ d)
  (h_cube_root : p > Nat.sqrt (Nat.sqrt N)) :
  Nat.Prime q := 
sorry

end smallest_prime_divisor_and_cube_root_l119_119237


namespace probability_cooking_selected_l119_119310

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119310


namespace car_A_faster_than_car_B_l119_119146

noncomputable def car_A_speed := 
  let t_A1 := 50 / 60 -- time for the first 50 miles at 60 mph
  let t_A2 := 50 / 40 -- time for the next 50 miles at 40 mph
  let t_A := t_A1 + t_A2 -- total time for Car A
  100 / t_A -- average speed of Car A

noncomputable def car_B_speed := 
  let t_B := 1 + (1 / 4) + 1 -- total time for Car B, including a 15-minute stop
  100 / t_B -- average speed of Car B

theorem car_A_faster_than_car_B : car_A_speed > car_B_speed := 
by sorry

end car_A_faster_than_car_B_l119_119146


namespace find_some_number_l119_119965

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l119_119965


namespace find_ac_bc_val_l119_119615

variable (a b c d : ℚ)
variable (h_neq : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h1 : (a + c) * (a + d) = 1)
variable (h2 : (b + c) * (b + d) = 1)

theorem find_ac_bc_val : (a + c) * (b + c) = -1 := 
by 
  sorry

end find_ac_bc_val_l119_119615


namespace probability_cooking_is_one_fourth_l119_119358
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119358


namespace sarah_cupcakes_ratio_l119_119822

theorem sarah_cupcakes_ratio (total_cupcakes : ℕ) (cookies_from_michael : ℕ) 
    (final_desserts : ℕ) (cupcakes_given : ℕ) (h1 : total_cupcakes = 9) 
    (h2 : cookies_from_michael = 5) (h3 : final_desserts = 11) 
    (h4 : total_cupcakes - cupcakes_given + cookies_from_michael = final_desserts) : 
    cupcakes_given / total_cupcakes = 1 / 3 :=
by
  sorry

end sarah_cupcakes_ratio_l119_119822


namespace negation_of_proposition_l119_119671

theorem negation_of_proposition :
  (∀ x y : ℝ, (x * y = 0 → x = 0 ∨ y = 0)) →
  (∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
sorry

end negation_of_proposition_l119_119671


namespace northbound_vehicle_count_l119_119512

theorem northbound_vehicle_count :
  ∀ (southbound_speed northbound_speed : ℝ) (vehicles_passed : ℕ) 
  (time_minutes : ℝ) (section_length : ℝ), 
  southbound_speed = 70 → northbound_speed = 50 → vehicles_passed = 30 → time_minutes = 10
  → section_length = 150
  → (vehicles_passed / ((southbound_speed + northbound_speed) * (time_minutes / 60))) * section_length = 270 :=
by sorry

end northbound_vehicle_count_l119_119512


namespace smallest_part_division_l119_119019

theorem smallest_part_division (S : ℚ) (P1 P2 P3 : ℚ) (total : ℚ) :
  (P1, P2, P3) = (1, 2, 3) →
  total = 64 →
  S = total / (P1 + P2 + P3) →
  S = 10 + 2/3 :=
by
  sorry

end smallest_part_division_l119_119019


namespace milan_billed_minutes_l119_119163

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l119_119163


namespace repeating_decimal_sum_l119_119598

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l119_119598


namespace zeros_between_decimal_point_and_first_non_zero_digit_l119_119885

theorem zeros_between_decimal_point_and_first_non_zero_digit :
  ∀ n d : ℕ, (n = 7) → (d = 5000) → (real.to_rat ⟨(n : ℝ) / d, sorry⟩ = 7 / 5000) →
  (exists (k : ℕ), (7 / 5000 = 7 * 10^(-k)) ∧ k = 3) :=
by
  intros n d hn hd eq
  have h : d = 2^3 * 5^3 := by norm_num [hd, pow_succ, mul_comm]
  rw [hn, hd, h] at eq
  exact exists.intro 3 (by norm_num)

end zeros_between_decimal_point_and_first_non_zero_digit_l119_119885


namespace sum_of_roots_of_quadratic_eq_l119_119705

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l119_119705


namespace mark_sandwiches_l119_119558

/--
Each day of a 6-day workweek, Mark bought either an 80-cent donut or a $1.20 sandwich. 
His total expenditure for the week was an exact number of dollars.
Prove that Mark bought exactly 3 sandwiches.
-/
theorem mark_sandwiches (s d : ℕ) (h1 : s + d = 6) (h2 : ∃ k : ℤ, 120 * s + 80 * d = 100 * k) : s = 3 :=
by
  sorry

end mark_sandwiches_l119_119558


namespace sequence_difference_l119_119884

-- Definition of sequences sums
def odd_sum (n : ℕ) : ℕ := (n * n)
def even_sum (n : ℕ) : ℕ := n * (n + 1)

-- Main property to prove
theorem sequence_difference :
  odd_sum 1013 - even_sum 1011 = 3047 :=
by
  -- Definitions and assertions here
  sorry

end sequence_difference_l119_119884


namespace eighth_term_geometric_sequence_l119_119883

theorem eighth_term_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 12) (h_r : r = 1/4) (h_n : n = 8) :
  a * r^(n - 1) = 3 / 4096 := 
by 
  sorry

end eighth_term_geometric_sequence_l119_119883


namespace smallest_n_1999n_congruent_2001_mod_10000_l119_119416

theorem smallest_n_1999n_congruent_2001_mod_10000 : ∃ n : ℕ, n > 0 ∧ (1999 * n) % 10000 = 2001 ∧ ∀ m : ℕ, m > 0 ∧ 1999 * m % 10000 = 2001 → n ≤ m := by
  use 5999
  split
  - exact Nat.succ_pos'
  split
  - norm_num
  sorry

end smallest_n_1999n_congruent_2001_mod_10000_l119_119416


namespace least_positive_three_digit_multiple_of_7_l119_119522

theorem least_positive_three_digit_multiple_of_7 : ∃ n : ℕ, n % 7 = 0 ∧ n ≥ 100 ∧ n < 1000 ∧ ∀ m : ℕ, (m % 7 = 0 ∧ m ≥ 100 ∧ m < 1000) → n ≤ m := 
by
  sorry

end least_positive_three_digit_multiple_of_7_l119_119522


namespace museum_discount_l119_119557

theorem museum_discount
  (Dorothy_age : ℕ)
  (total_family_members : ℕ)
  (regular_ticket_cost : ℕ)
  (discountapplies_age : ℕ)
  (before_trip : ℕ)
  (after_trip : ℕ)
  (spend : ℕ := before_trip - after_trip)
  (adults_tickets : ℕ := total_family_members - 2)
  (youth_tickets : ℕ := 2)
  (total_cost := adults_tickets * regular_ticket_cost + youth_tickets * (regular_ticket_cost - regular_ticket_cost * discount))
  (discount : ℚ)
  (expected_spend : ℕ := 44) :
  total_cost = spend :=
by
  sorry

end museum_discount_l119_119557


namespace bryden_receives_amount_l119_119382

variable (q : ℝ) (p : ℝ) (num_quarters : ℝ)

-- Define the conditions
def face_value_of_quarter : Prop := q = 0.25
def percentage_offer : Prop := p = 25 * q
def number_of_quarters : Prop := num_quarters = 5

-- Define the theorem to be proved
theorem bryden_receives_amount (h1 : face_value_of_quarter q) (h2 : percentage_offer q p) (h3 : number_of_quarters num_quarters) :
  (p * num_quarters * q) = 31.25 :=
by
  sorry

end bryden_receives_amount_l119_119382


namespace lowest_temperature_l119_119478

-- Define the temperatures in the four cities.
def temp_Harbin := -20
def temp_Beijing := -10
def temp_Hangzhou := 0
def temp_Jinhua := 2

-- The proof statement asserting the lowest temperature.
theorem lowest_temperature :
  min temp_Harbin (min temp_Beijing (min temp_Hangzhou temp_Jinhua)) = -20 :=
by
  -- Proof omitted
  sorry

end lowest_temperature_l119_119478


namespace solid_could_be_rectangular_prism_or_cylinder_l119_119976

-- Definitions for the conditions
def is_rectangular_prism (solid : Type) : Prop := sorry
def is_cylinder (solid : Type) : Prop := sorry
def front_view_is_rectangle (solid : Type) : Prop := sorry
def side_view_is_rectangle (solid : Type) : Prop := sorry

-- Main statement
theorem solid_could_be_rectangular_prism_or_cylinder
  {solid : Type}
  (h1 : front_view_is_rectangle solid)
  (h2 : side_view_is_rectangle solid) :
  is_rectangular_prism solid ∨ is_cylinder solid :=
sorry

end solid_could_be_rectangular_prism_or_cylinder_l119_119976


namespace number_of_movies_in_series_l119_119688

variables (watched_movies remaining_movies total_movies : ℕ)

theorem number_of_movies_in_series 
  (h_watched : watched_movies = 4) 
  (h_remaining : remaining_movies = 4) :
  total_movies = watched_movies + remaining_movies :=
by
  sorry

end number_of_movies_in_series_l119_119688


namespace alpha_add_beta_eq_pi_div_two_l119_119427

open Real

theorem alpha_add_beta_eq_pi_div_two (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : (sin α) ^ 4 / (cos β) ^ 2 + (cos α) ^ 4 / (sin β) ^ 2 = 1) :
  α + β = π / 2 :=
sorry

end alpha_add_beta_eq_pi_div_two_l119_119427


namespace find_a_l119_119449

noncomputable def polynomial1 (x : ℝ) : ℝ := x^3 + 3 * x^2 - x - 3
noncomputable def polynomial2 (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem find_a (a : ℝ) (x : ℝ) (hx1 : polynomial1 x > 0)
  (hx2 : polynomial2 x a ≤ 0) (ha : a > 0) : 
  3 / 4 ≤ a ∧ a < 4 / 3 :=
sorry

end find_a_l119_119449


namespace jane_change_l119_119217

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l119_119217


namespace probability_of_selecting_cooking_l119_119333

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119333


namespace mode_of_dataSet_is_3_l119_119247

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end mode_of_dataSet_is_3_l119_119247


namespace relationship_between_b_and_g_l119_119550

-- Definitions based on the conditions
def n_th_boy_dances (n : ℕ) : ℕ := n + 5
def last_boy_dances_with_all : Prop := ∃ b g : ℕ, (n_th_boy_dances b = g)

-- The main theorem to prove the relationship between b and g
theorem relationship_between_b_and_g (b g : ℕ) (h : last_boy_dances_with_all) : b = g - 5 :=
by
  sorry

end relationship_between_b_and_g_l119_119550


namespace smallest_positive_period_of_h_l119_119229

-- Definitions of f and g with period 1
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ
axiom T1 : ℝ
axiom T2 : ℝ

-- Given conditions
@[simp] axiom f_periodic : ∀ x, f (x + T1) = f x
@[simp] axiom g_periodic : ∀ x, g (x + T2) = g x
@[simp] axiom T1_eq_one : T1 = 1
@[simp] axiom T2_eq_one : T2 = 1

-- Statement to prove the smallest positive period of h(x) = f(x) + g(x) is 1/k
theorem smallest_positive_period_of_h (k : ℕ) (h : ℝ → ℝ) (hk: k > 0) :
  (∀ x, h (x + 1) = h x) →
  (∀ T > 0, (∀ x, h (x + T) = h x) → (∃ k : ℕ, T = 1 / k)) :=
by sorry

end smallest_positive_period_of_h_l119_119229


namespace repeating_decimals_sum_l119_119578

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l119_119578


namespace ellipse_focus_and_axes_l119_119498

theorem ellipse_focus_and_axes (m : ℝ) :
  (∃ a b : ℝ, (a > b) ∧ (mx^2 + y^2 = 1) ∧ (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2 * a = 3 * 2 * b)) → 
  m = 4 / 9 :=
by
  intro h
  rcases h with ⟨a, b, hab, h_eq, ha, hb, ha_b_eq⟩
  sorry

end ellipse_focus_and_axes_l119_119498


namespace xiaoming_selects_cooking_probability_l119_119331

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119331


namespace min_expression_value_2023_l119_119523

noncomputable def min_expr_val := ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023

noncomputable def least_value : ℝ := 2023

theorem min_expression_value_2023 : min_expr_val ∧ (∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = least_value) := 
by sorry

end min_expression_value_2023_l119_119523


namespace sum_of_repeating_decimals_l119_119584

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l119_119584


namespace xy_diff_square_l119_119779

theorem xy_diff_square (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 :=
by
  sorry

end xy_diff_square_l119_119779


namespace length_of_each_train_l119_119517

theorem length_of_each_train (L : ℝ) 
  (speed_faster : ℝ := 45 * 5 / 18) -- converting 45 km/hr to m/s
  (speed_slower : ℝ := 36 * 5 / 18) -- converting 36 km/hr to m/s
  (time : ℝ := 36) 
  (relative_speed : ℝ := speed_faster - speed_slower) 
  (total_distance : ℝ := relative_speed * time) 
  (length_each_train : ℝ := total_distance / 2) 
  : length_each_train = 45 := 
by 
  sorry

end length_of_each_train_l119_119517


namespace smallest_n_is_1770_l119_119607

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end smallest_n_is_1770_l119_119607


namespace men_entered_l119_119994

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l119_119994


namespace trig_identity_l119_119752

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π + α)) / (Real.sin (π / 2 - α)) = -2 :=
by
  sorry

end trig_identity_l119_119752


namespace multiply_polynomials_l119_119826

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l119_119826


namespace inverse_of_217_mod_397_l119_119930

theorem inverse_of_217_mod_397 :
  ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ 217 * a % 397 = 1 :=
sorry

end inverse_of_217_mod_397_l119_119930


namespace largest_integer_x_divisible_l119_119942

theorem largest_integer_x_divisible (x : ℤ) : 
  (∃ x : ℤ, (x^2 + 3 * x + 8) % (x - 2) = 0 ∧ x ≤ 1) → x = 1 :=
sorry

end largest_integer_x_divisible_l119_119942


namespace probability_of_selecting_cooking_is_one_fourth_l119_119378

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119378


namespace peach_cost_l119_119849

theorem peach_cost 
  (total_fruits : ℕ := 32)
  (total_cost : ℕ := 52)
  (plum_cost : ℕ := 2)
  (num_plums : ℕ := 20)
  (cost_peach : ℕ) :
  (total_cost - (num_plums * plum_cost)) = cost_peach * (total_fruits - num_plums) →
  cost_peach = 1 :=
by
  intro h
  sorry

end peach_cost_l119_119849


namespace point_on_angle_describes_arc_of_circle_l119_119736

open EuclideanGeometry

theorem point_on_angle_describes_arc_of_circle
(angle BAC : ∀ A B C : Point, LinePoint B A → LinePoint A C → Sphere) 
(O1 O2 : Point) 
(r1 r2 : ℝ) 
(AB AC : Line)
(h₁ : Tangent O1 (angle B A C) AB r1)
(h₂ : Tangent O2 (angle B A C) AC r2) :
∃ A1 : Point, ArcCircle (segment O1 A1) (segment O2 A1) := 
sorry

end point_on_angle_describes_arc_of_circle_l119_119736


namespace number_of_subsets_of_set_l119_119022

theorem number_of_subsets_of_set (x y : ℝ) 
  (z : ℂ) (hz : z = (2 - (1 : ℂ) * Complex.I) / (1 + (2 : ℂ) * Complex.I))
  (hx : z.re = x) (hy : z.im = y) : 
  (Finset.powerset ({x, 2^x, y} : Finset ℝ)).card = 8 :=
by
  sorry

end number_of_subsets_of_set_l119_119022


namespace count_valid_triangles_l119_119772

def triangle_area (a b c : ℕ) : ℕ :=
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c)

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c < 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

theorem count_valid_triangles : { n : ℕ // n = 24 } :=
  sorry

end count_valid_triangles_l119_119772


namespace remaining_distance_l119_119798

theorem remaining_distance (speed time distance_covered total_distance remaining_distance : ℕ) 
  (h1 : speed = 60) 
  (h2 : time = 2) 
  (h3 : total_distance = 300)
  (h4 : distance_covered = speed * time) 
  (h5 : remaining_distance = total_distance - distance_covered) : 
  remaining_distance = 180 := 
by
  sorry

end remaining_distance_l119_119798


namespace men_entered_l119_119996

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l119_119996


namespace milan_billed_minutes_l119_119168

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l119_119168


namespace john_income_increase_l119_119800

noncomputable def net_percentage_increase (initial_income : ℝ) (final_income_before_bonus : ℝ) (monthly_bonus : ℝ) (tax_deduction_rate : ℝ) : ℝ :=
  let weekly_bonus := monthly_bonus / 4
  let final_income_before_taxes := final_income_before_bonus + weekly_bonus
  let tax_deduction := tax_deduction_rate * final_income_before_taxes
  let net_final_income := final_income_before_taxes - tax_deduction
  ((net_final_income - initial_income) / initial_income) * 100

theorem john_income_increase :
  net_percentage_increase 40 60 100 0.10 = 91.25 := by
  sorry

end john_income_increase_l119_119800


namespace ratio_of_cost_to_selling_price_l119_119058

-- Define the conditions in Lean
variable (C S : ℝ) -- C is the cost price per pencil, S is the selling price per pencil
variable (h : 90 * C - 40 * S = 90 * S)

-- Define the statement to be proved
theorem ratio_of_cost_to_selling_price (C S : ℝ) (h : 90 * C - 40 * S = 90 * S) : (90 * C) / (90 * S) = 13 :=
by
  sorry

end ratio_of_cost_to_selling_price_l119_119058


namespace range_of_a_l119_119955

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) ↔ a < 1004 := 
sorry

end range_of_a_l119_119955


namespace probability_select_cooking_l119_119319

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119319


namespace binary_catalan_count_l119_119422

open Nat

/-- The number of 2n-bit binary numbers consisting of n ones and n zeros
such that when scanned from left to right, the cumulative number of 1s is never less than the cumulative number of 0s is given by the n-th Catalan number. -/
theorem binary_catalan_count (n : ℕ) : 
  let p_2n := 1 / (n + 1) * (factorial (2 * n)) / (factorial n * factorial n)
  in p_2n = (factorial (2 * n)) / (factorial (n + 1) * factorial n) :=
by sorry

end binary_catalan_count_l119_119422


namespace distinct_real_solutions_l119_119192

open Real Nat

noncomputable def p_n : ℕ → ℝ → ℝ 
| 0, x => x
| (n+1), x => (p_n n (x^2 - 2))

theorem distinct_real_solutions (n : ℕ) : 
  ∃ S : Finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, p_n n x = x ∧ (∀ y ∈ S, x ≠ y → x ≠ y) := 
sorry

end distinct_real_solutions_l119_119192


namespace inscribed_to_circumscribed_ratio_l119_119033

theorem inscribed_to_circumscribed_ratio (a b : ℕ)
  (h1 : 6 = a) (h2 : 8 = b) :
  let c := (a^2 + b^2).sqrt
  let inscribed_radius := (a + b - c) / 2
  let circumscribed_radius := c / 2 in
  inscribed_radius / circumscribed_radius = 2 / 5 :=
by
  have ha : a = 6 := h1
  have hb : b = 8 := h2
  let c : ℕ := Int.sqrt ((a: ℤ)^2 + (b: ℤ)^2) -- Hypotenuse length
  let inscribed_radius := (a + b - c) / 2
  let circumscribed_radius := c / 2
  have hc : c = 10 := by sorry
  have h_inscribed : inscribed_radius = 2 := by sorry
  have h_circumscribed : circumscribed_radius = 5 := by sorry
  rw [h_inscribed, h_circumscribed]
  norm_num

end inscribed_to_circumscribed_ratio_l119_119033


namespace probability_of_selecting_cooking_l119_119354

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119354


namespace ral_current_age_l119_119238

variable (ral suri : ℕ)

-- Conditions
axiom age_relation : ral = 3 * suri
axiom suri_future_age : suri + 3 = 16

-- Statement
theorem ral_current_age : ral = 39 := by
  sorry

end ral_current_age_l119_119238


namespace range_of_m_l119_119432

noncomputable def f (m : ℝ) (x : ℝ) :=
  if x > 0 then m * x - Real.log x
  else m * x + Real.log (-x)

theorem range_of_m (m : ℝ) (x1 x2 : ℝ) (k : ℝ) 
  (h1 : ∀ x, f m x = if x > 0 then m * x - Real.log x else m * x + Real.log (-x))
  (h2 : Deriv f x1 = 0 ∧ Deriv f x2 = 0)
  (h3 : 0 < k ∧ k ≤ 2 * Real.exp 1)
  (h4 : k = (f m x2 - f m x1) / (x2 - x1)) :
  1 / Real.exp 1 < m ∧ m ≤ Real.exp 1 :=
sorry

end range_of_m_l119_119432


namespace assign_students_to_villages_l119_119419

theorem assign_students_to_villages (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  ∃ N : ℕ, N = 70 ∧ 
  (∃ (f : Fin n → Fin m), (∀ i j, f i = f j ↔ i = j) ∧ 
  (∀ x : Fin m, ∃ y : Fin n, f y = x)) :=
by
  sorry

end assign_students_to_villages_l119_119419


namespace probability_of_selecting_cooking_l119_119293

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l119_119293


namespace inequalities_not_equivalent_l119_119922

theorem inequalities_not_equivalent (x : ℝ) (h1 : x ≠ 1) :
  (x + 3 - (1 / (x - 1)) > -x + 2 - (1 / (x - 1))) ↔ (x + 3 > -x + 2) → False :=
by
  sorry

end inequalities_not_equivalent_l119_119922


namespace find_digit_l119_119879

theorem find_digit {x : ℕ} (hx : x = 7) : (10 * (x - 3) + x) = 47 :=
by
  sorry

end find_digit_l119_119879


namespace two_cubic_meters_to_cubic_feet_l119_119016

theorem two_cubic_meters_to_cubic_feet :
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  2 * cubic_meter_to_cubic_feet = 70.6294 :=
by
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  have h : 2 * cubic_meter_to_cubic_feet = 70.6294 := sorry
  exact h

end two_cubic_meters_to_cubic_feet_l119_119016


namespace number_of_math_students_l119_119925

-- Definitions for the problem conditions
variables (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
variable (total_students_eq : total_students = 100)
variable (both_classes_eq : both_classes = 10)
variable (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))

-- Theorem statement
theorem number_of_math_students (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
  (total_students_eq : total_students = 100)
  (both_classes_eq : both_classes = 10)
  (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))
  (total_students_eq : total_students = physics_class + math_class - both_classes) :
  math_class = 88 :=
sorry

end number_of_math_students_l119_119925


namespace inequality_for_abcd_one_l119_119489

theorem inequality_for_abcd_one (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) + (1 / (1 + d)) > 1 := 
by
  sorry

end inequality_for_abcd_one_l119_119489


namespace probability_cooking_is_one_fourth_l119_119361
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119361


namespace quadratic_solution_l119_119111

theorem quadratic_solution :
  ∀ x : ℝ, (3 * x - 1) * (2 * x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 :=
by
  sorry

end quadratic_solution_l119_119111


namespace gcd_18_n_eq_6_in_range_l119_119171

theorem gcd_18_n_eq_6_in_range :
  {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 18 n = 6}.card = 22 :=
by
  -- To skip the proof
  sorry

end gcd_18_n_eq_6_in_range_l119_119171


namespace milan_billed_minutes_l119_119165

-- Define the conditions
def monthly_fee : ℝ := 2
def cost_per_minute : ℝ := 0.12
def total_bill : ℝ := 23.36

-- Define the number of minutes based on the above conditions
def minutes := (total_bill - monthly_fee) / cost_per_minute

-- Prove that the number of minutes is 178
theorem milan_billed_minutes : minutes = 178 := by
  -- Proof steps would go here, but as instructed, we use 'sorry' to skip the proof.
  sorry

end milan_billed_minutes_l119_119165


namespace clock_angle_at_7_oclock_l119_119702

theorem clock_angle_at_7_oclock : 
  let degrees_per_hour := 360 / 12
  let hour_hand_position := 7
  let minute_hand_position := 12
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour
  smaller_angle = 150 :=
begin
  -- degrees_per_hour is 30
  let degrees_per_hour := 30,
  -- define the positions of hour and minute hands
  let hour_hand_position := 7,
  let minute_hand_position := 12,
  -- calculate the spaces from the minute hand (12) to hour hand (7)
  let spaces_from_minute_hand := if hour_hand_position ≥ minute_hand_position then hour_hand_position - minute_hand_position else hour_hand_position + (12 - minute_hand_position),
  -- spaces_from_minute_hand calculation shows 5 spaces (i.e., 5 hours)
  let smaller_angle := spaces_from_minute_hand * degrees_per_hour,
  -- therefore, the smaller angle should be 150 degrees
  exact calc smaller_angle = 5 * 30 : by rfl
                           ... = 150 : by norm_num,
end

end clock_angle_at_7_oclock_l119_119702


namespace possible_values_of_expr_l119_119756

-- Define conditions
variables (x y : ℝ)
axiom h1 : x + y = 2
axiom h2 : y > 0
axiom h3 : x ≠ 0

-- Define the expression we're investigating
noncomputable def expr : ℝ := (1 / (abs x)) + (abs x / (y + 2))

-- The statement of the problem
theorem possible_values_of_expr :
  expr x y = 3 / 4 ∨ expr x y = 5 / 4 :=
sorry

end possible_values_of_expr_l119_119756


namespace only_polyC_is_square_of_binomial_l119_119886

-- Defining the polynomials
def polyA (m n : ℤ) : ℤ := (-m + n) * (m - n)
def polyB (a b : ℤ) : ℤ := (1/2 * a + b) * (b - 1/2 * a)
def polyC (x : ℤ) : ℤ := (x + 5) * (x + 5)
def polyD (a b : ℤ) : ℤ := (3 * a - 4 * b) * (3 * b + 4 * a)

-- Proving that only polyC fits the square of a binomial formula
theorem only_polyC_is_square_of_binomial (x : ℤ) :
  (polyC x) = (x + 5) * (x + 5) ∧
  (∀ m n : ℤ, polyA m n ≠ (m - n)^2) ∧
  (∀ a b : ℤ, polyB a b ≠ (1/2 * a + b)^2) ∧
  (∀ a b : ℤ, polyD a b ≠ (3 * a - 4 * b)^2) :=
by
  sorry

end only_polyC_is_square_of_binomial_l119_119886


namespace f_sq_add_g_sq_eq_one_f_even_f_periodic_l119_119426

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom g_odd : ∀ x : ℝ, g (-x) = - g x
axiom f_0 : f 0 = 1
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y

theorem f_sq_add_g_sq_eq_one (x : ℝ) : f x ^ 2 + g x ^ 2 = 1 :=
sorry

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
sorry

theorem f_periodic (a : ℝ) (ha : a ≠ 0) (hfa : f a = 1) : ∀ x : ℝ, f (x + a) = f x :=
sorry

end f_sq_add_g_sq_eq_one_f_even_f_periodic_l119_119426


namespace find_a_l119_119767

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (2 * x) - (1 / 3) * Real.sin (3 * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  2 * a * Real.cos (2 * x) - Real.cos (3 * x)

theorem find_a (a : ℝ) (h : f_prime a (Real.pi / 3) = 0) : a = 1 :=
by
  sorry

end find_a_l119_119767


namespace cos_5_theta_l119_119198

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end cos_5_theta_l119_119198


namespace repeating_decimals_sum_l119_119577

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l119_119577


namespace smallest_int_a_for_inequality_l119_119446

theorem smallest_int_a_for_inequality (a : ℤ) : 
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → 
  Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) → 
  a = 1 := 
sorry

end smallest_int_a_for_inequality_l119_119446


namespace division_addition_correct_l119_119403

-- Define a function that performs the arithmetic operations described
def calculateResult : ℕ :=
  let division := 12 * 4 -- dividing 12 by 1/4 is the same as multiplying by 4
  division + 5 -- then add 5 to the result

-- The theorem statement to prove
theorem division_addition_correct : calculateResult = 53 := by
  sorry

end division_addition_correct_l119_119403


namespace closest_point_on_line_to_target_l119_119747

noncomputable def parametricPoint (s : ℝ) : ℝ × ℝ × ℝ :=
  (6 + 3 * s, 2 - 9 * s, 0 + 6 * s)

noncomputable def closestPoint : ℝ × ℝ × ℝ :=
  (249/42, 95/42, -1/7)

theorem closest_point_on_line_to_target :
  ∃ s : ℝ, parametricPoint s = closestPoint :=
by
  sorry

end closest_point_on_line_to_target_l119_119747


namespace chess_pieces_missing_l119_119398

theorem chess_pieces_missing (total_pieces present_pieces missing_pieces : ℕ) 
  (h1 : total_pieces = 32)
  (h2 : present_pieces = 22)
  (h3 : missing_pieces = total_pieces - present_pieces) :
  missing_pieces = 10 :=
by
  sorry

end chess_pieces_missing_l119_119398


namespace point_on_y_axis_coordinates_l119_119487

theorem point_on_y_axis_coordinates (m : ℤ) (P : ℤ × ℤ) (hP : P = (m - 1, m + 3)) (hY : P.1 = 0) : P = (0, 4) :=
sorry

end point_on_y_axis_coordinates_l119_119487


namespace probability_of_selecting_cooking_l119_119355

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119355


namespace probability_cooking_is_one_fourth_l119_119357
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l119_119357


namespace art_club_artworks_l119_119679

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end art_club_artworks_l119_119679


namespace train_speed_from_clicks_l119_119674

theorem train_speed_from_clicks (speed_mph : ℝ) (rail_length_ft : ℝ) (clicks_heard : ℝ) :
  rail_length_ft = 40 →
  clicks_heard = 1 →
  (60 * rail_length_ft * clicks_heard * speed_mph / 5280) = 27 :=
by
  intros h1 h2
  sorry

end train_speed_from_clicks_l119_119674


namespace largest_B_is_9_l119_119040

def is_divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

def is_divisible_by_four (n : ℕ) : Prop :=
  n % 4 = 0

def largest_B_divisible_by_3_and_4 (B : ℕ) : Prop :=
  is_divisible_by_three (21 + B) ∧ is_divisible_by_four 32

theorem largest_B_is_9 : largest_B_divisible_by_3_and_4 9 :=
by
  have h1 : is_divisible_by_three (21 + 9) := by sorry
  have h2 : is_divisible_by_four 32 := by sorry
  exact ⟨h1, h2⟩

end largest_B_is_9_l119_119040


namespace expand_polynomial_product_l119_119151

theorem expand_polynomial_product :
  (λ x : ℝ, (x^2 - 3*x + 3) * (x^2 + 3*x + 3)) = (λ x : ℝ, x^4 - 3*x^2 + 9) :=
by {
  funext x,
  -- The detailed proof steps would go here
  sorry
}

end expand_polynomial_product_l119_119151


namespace solution_set_l119_119486

theorem solution_set :
  {p : ℝ × ℝ | (p.1^2 + 3 * p.1 * p.2 + 2 * p.2^2) * (p.1^2 * p.2^2 - 1) = 0} =
  {p : ℝ × ℝ | p.2 = -p.1 / 2} ∪
  {p : ℝ × ℝ | p.2 = -p.1} ∪
  {p : ℝ × ℝ | p.2 = -1 / p.1} ∪
  {p : ℝ × ℝ | p.2 = 1 / p.1} :=
by sorry

end solution_set_l119_119486


namespace fraction_of_project_completed_in_one_hour_l119_119624

noncomputable def fraction_of_project_completed_together (a b : ℝ) : ℝ :=
  (1 / a) + (1 / b)

theorem fraction_of_project_completed_in_one_hour (a b : ℝ) :
  fraction_of_project_completed_together a b = (1 / a) + (1 / b) := by
  sorry

end fraction_of_project_completed_in_one_hour_l119_119624


namespace find_f_2018_l119_119000

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_functional_eq : ∀ x : ℝ, f x = - (1 / f (x + 3))
axiom f_at_4 : f 4 = -2018

theorem find_f_2018 : f 2018 = -2018 :=
  sorry

end find_f_2018_l119_119000


namespace simplify_expression_l119_119239

theorem simplify_expression (x : ℝ) :
  (3 * x)^3 - (4 * x^2) * (2 * x^3) = 27 * x^3 - 8 * x^5 :=
by
  sorry

end simplify_expression_l119_119239


namespace reflection_image_l119_119499

theorem reflection_image (m b : ℝ) 
  (h1 : ∀ x y : ℝ, (x, y) = (0, 1) → (4, 5) = (2 * ((x + (m * y - y + b))/ (1 + m^2)) - x, 2 * ((y + (m * x - x + b)) / (1 + m^2)) - y))
  : m + b = 4 :=
sorry

end reflection_image_l119_119499


namespace Nori_gave_more_to_Lea_l119_119054

noncomputable def Nori_crayons_initial := 4 * 8
def Mae_crayons := 5
def Nori_crayons_left := 15
def Crayons_given_to_Lea := Nori_crayons_initial - Mae_crayons - Nori_crayons_left
def Crayons_difference := Crayons_given_to_Lea - Mae_crayons

theorem Nori_gave_more_to_Lea : Crayons_difference = 7 := by
  sorry

end Nori_gave_more_to_Lea_l119_119054


namespace circumscribed_circle_radius_l119_119723

noncomputable def radius_of_circumcircle (a b c : ℚ) (h_a : a = 15/2) (h_b : b = 10) (h_c : c = 25/2) : ℚ :=
if h_triangle : a^2 + b^2 = c^2 then (c / 2) else 0

theorem circumscribed_circle_radius :
  radius_of_circumcircle (15/2 : ℚ) 10 (25/2 : ℚ) (by norm_num) (by norm_num) (by norm_num) = 25 / 4 := 
by
  sorry

end circumscribed_circle_radius_l119_119723


namespace average_salary_difference_l119_119891

theorem average_salary_difference :
  let total_payroll_factory := 30000
  let num_factory_workers := 15
  let total_payroll_office := 75000
  let num_office_workers := 30
  (total_payroll_office / num_office_workers) - (total_payroll_factory / num_factory_workers) = 500 :=
by
  sorry

end average_salary_difference_l119_119891


namespace problem_statement_l119_119207

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 := a 1 = 1
def condition2 := (a 3 + a 4) / (a 1 + a 2) = 4
def increasing := q > 0

-- Definition of S_n
def sum_geom (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)

theorem problem_statement (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_geom : increasing_geometric_sequence a q) 
  (h_condition1 : condition1 a) 
  (h_condition2 : condition2 a) 
  (h_increasing : increasing q)
  (h_sum_geom : sum_geom a q S) : 
  S 5 = 31 :=
sorry

end problem_statement_l119_119207


namespace problem_solution_l119_119872

theorem problem_solution (k : ℤ) : k ≤ 0 ∧ -2 < k → k = -1 ∨ k = 0 :=
by
  sorry

end problem_solution_l119_119872


namespace annual_income_correct_l119_119516

-- Define the principal amounts and interest rates
def principal_1 : ℝ := 3000
def rate_1 : ℝ := 0.085

def principal_2 : ℝ := 5000
def rate_2 : ℝ := 0.064

-- Define the interest calculations for each investment
def interest_1 : ℝ := principal_1 * rate_1
def interest_2 : ℝ := principal_2 * rate_2

-- Define the total annual income
def total_annual_income : ℝ := interest_1 + interest_2

-- Proof statement
theorem annual_income_correct : total_annual_income = 575 :=
by
  sorry

end annual_income_correct_l119_119516


namespace find_matches_in_second_set_l119_119492

-- Conditions defined as Lean variables
variables (x : ℕ)
variables (avg_first_20 : ℚ := 40)
variables (avg_second_x : ℚ := 20)
variables (avg_all_30 : ℚ := 100 / 3)
variables (total_first_20 : ℚ := 20 * avg_first_20)
variables (total_all_30 : ℚ := 30 * avg_all_30)

-- Proof statement (question) along with conditions
theorem find_matches_in_second_set (x_value : x = 10) :
  avg_first_20 = 40 ∧ avg_second_x = 20 ∧ avg_all_30 = 100 / 3 →
  20 * avg_first_20 + x * avg_second_x = 30 * avg_all_30 → x = 10 := 
sorry

end find_matches_in_second_set_l119_119492


namespace earnings_difference_l119_119480

theorem earnings_difference :
  let oula_deliveries := 96
  let tona_deliveries := oula_deliveries * 3 / 4
  let area_A_fee := 100
  let area_B_fee := 125
  let area_C_fee := 150
  let oula_area_A_deliveries := 48
  let oula_area_B_deliveries := 32
  let oula_area_C_deliveries := 16
  let tona_area_A_deliveries := 27
  let tona_area_B_deliveries := 18
  let tona_area_C_deliveries := 9
  let oula_total_earnings := oula_area_A_deliveries * area_A_fee + oula_area_B_deliveries * area_B_fee + oula_area_C_deliveries * area_C_fee
  let tona_total_earnings := tona_area_A_deliveries * area_A_fee + tona_area_B_deliveries * area_B_fee + tona_area_C_deliveries * area_C_fee
  oula_total_earnings - tona_total_earnings = 4900 := by
sorry

end earnings_difference_l119_119480


namespace opposite_neg_one_half_l119_119099

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l119_119099


namespace total_payment_is_53_l119_119401

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end total_payment_is_53_l119_119401


namespace units_digit_n_l119_119749

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31 ^ 6) (h2 : m % 10 = 9) : n % 10 = 2 := 
sorry

end units_digit_n_l119_119749


namespace extremum_range_k_l119_119434

noncomputable def f (x k : Real) : Real :=
  Real.exp x / x + k * (Real.log x - x)

/-- 
For the function f(x) = (exp(x) / x) + k * (log(x) - x), if x = 1 is the only extremum point, 
then k is in the interval (-∞, e].
-/
theorem extremum_range_k (k : Real) : 
  (∀ x : Real, (0 < x) → (f x k ≤ f 1 k)) → 
  k ≤ Real.exp 1 :=
sorry

end extremum_range_k_l119_119434


namespace convert_spherical_to_rectangular_l119_119739

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta, rho * Real.sin phi * Real.sin theta, rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 10 (4 * Real.pi / 3) (Real.pi / 3) = (-5 * Real.sqrt 3, -15 / 2, 5) :=
by 
  sorry

end convert_spherical_to_rectangular_l119_119739


namespace multiply_and_simplify_l119_119848
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l119_119848


namespace number_of_subsets_of_P_l119_119193

noncomputable def P : Set ℝ := {x | x^2 - 2*x + 1 = 0}

theorem number_of_subsets_of_P : ∃ (n : ℕ), n = 2 ∧ ∀ S : Set ℝ, S ⊆ P → S = ∅ ∨ S = {1} := by
  sorry

end number_of_subsets_of_P_l119_119193


namespace probability_of_selecting_cooking_l119_119348

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119348


namespace Emir_needs_more_money_l119_119410

theorem Emir_needs_more_money
  (cost_dictionary : ℝ)
  (cost_dinosaur_book : ℝ)
  (cost_cookbook : ℝ)
  (cost_science_kit : ℝ)
  (cost_colored_pencils : ℝ)
  (saved_amount : ℝ)
  (total_cost : ℝ := cost_dictionary + cost_dinosaur_book + cost_cookbook + cost_science_kit + cost_colored_pencils)
  (more_money_needed : ℝ := total_cost - saved_amount) :
  cost_dictionary = 5.50 →
  cost_dinosaur_book = 11.25 →
  cost_cookbook = 5.75 →
  cost_science_kit = 8.40 →
  cost_colored_pencils = 3.60 →
  saved_amount = 24.50 →
  more_money_needed = 10.00 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Emir_needs_more_money_l119_119410


namespace probability_cooking_selected_l119_119311

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119311


namespace patty_can_avoid_chores_l119_119482

theorem patty_can_avoid_chores (money_per_pack packs total_cookies_per_pack chores kid_cookies_cost packs_bought total_cookies total_weekly_cost weeks : ℕ)
    (h1 : money_per_pack = 3)
    (h2 : packs = 15 / money_per_pack)
    (h3 : total_cookies_per_pack = 24)
    (h4 : total_cookies = (p : ℕ) → packs * total_cookies_per_pack)
    (h5 : chores = 4)
    (h6 : kid_cookies_cost = 3)
    (h7 : total_weekly_cost = 2 * chores * kid_cookies_cost)
    (h8 : weeks = (total_cookies / total_weekly_cost)) : 
  weeks = 10 :=
by sorry

end patty_can_avoid_chores_l119_119482


namespace opposite_neg_one_half_l119_119101

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l119_119101


namespace evaluate_expression_l119_119935

theorem evaluate_expression :
  (2:ℝ) ^ ((0:ℝ) ^ (Real.sin (Real.pi / 2)) ^ 2) + ((3:ℝ) ^ 0) ^ 1 ^ 4 = 2 := by
  -- Given conditions
  have h1 : Real.sin (Real.pi / 2) = 1 := by sorry
  have h2 : (3:ℝ) ^ 0 = 1 := by sorry
  have h3 : (0:ℝ) ^ 1 = 0 := by sorry
  -- Proof omitted
  sorry

end evaluate_expression_l119_119935


namespace coefficient_term_without_x_in_expansion_l119_119155

theorem coefficient_term_without_x_in_expansion 
  (x y : ℝ) : 
  ∃ c : ℝ, (∀ r : ℕ, r = 4 → c = (-1)^r * (Nat.choose 8 r) * y^(8-r)) ∧ c = 70 :=
by {
  use (Nat.choose 8 4 : ℝ),
  split,
  { intros r hr,
    rw hr,
    simp, },
  { norm_num }
}

end coefficient_term_without_x_in_expansion_l119_119155


namespace marbles_remainder_l119_119852

theorem marbles_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 :=
by sorry

end marbles_remainder_l119_119852


namespace afb_leq_bfa_l119_119244

open Real

variable {f : ℝ → ℝ}

theorem afb_leq_bfa
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : ∀ x > 0, DifferentiableAt ℝ f x)
  (h_cond : ∀ x > 0, x * (deriv (deriv f) x) - f x ≤ 0)
  (a b : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_lt_b : a < b) :
  a * f b ≤ b * f a := 
sorry

end afb_leq_bfa_l119_119244


namespace length_of_AC_l119_119210

-- Definitions from the problem
variable (AB BC CD DA : ℝ)
variable (angle_ADC : ℝ)
variable (AC : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  AB = 10 ∧ BC = 10 ∧ CD = 17 ∧ DA = 17 ∧ angle_ADC = 120

-- The mathematically equivalent proof statement
theorem length_of_AC (h : conditions AB BC CD DA angle_ADC) : AC = Real.sqrt 867 := sorry

end length_of_AC_l119_119210


namespace smaller_angle_at_seven_oclock_l119_119696

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l119_119696


namespace wickets_before_last_match_l119_119274

theorem wickets_before_last_match (R W : ℝ) (h1 : R = 12.4 * W) (h2 : R + 26 = 12 * (W + 7)) :
  W = 145 := 
by 
  sorry

end wickets_before_last_match_l119_119274


namespace smaller_angle_at_seven_oclock_l119_119695

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l119_119695


namespace max_shapes_in_8x14_grid_l119_119690

def unit_squares := 3
def grid_8x14 := 8 * 14
def grid_points (m n : ℕ) := (m + 1) * (n + 1)
def shapes_grid_points := 8
def max_shapes (total_points shape_points : ℕ) := total_points / shape_points

theorem max_shapes_in_8x14_grid 
  (m n : ℕ) (shape_points : ℕ) 
  (h1 : m = 8) (h2 : n = 14)
  (h3 : shape_points = 8) :
  max_shapes (grid_points m n) shape_points = 16 := by
  sorry

end max_shapes_in_8x14_grid_l119_119690


namespace percent_increase_expenditure_l119_119027

theorem percent_increase_expenditure (cost_per_minute_2005 minutes_2005 minutes_2020 total_expenditure_2005 total_expenditure_2020 : ℕ)
  (h1 : cost_per_minute_2005 = 10)
  (h2 : minutes_2005 = 200)
  (h3 : minutes_2020 = 2 * minutes_2005)
  (h4 : total_expenditure_2005 = minutes_2005 * cost_per_minute_2005)
  (h5 : total_expenditure_2020 = minutes_2020 * cost_per_minute_2005) :
  ((total_expenditure_2020 - total_expenditure_2005) * 100 / total_expenditure_2005) = 100 :=
by
  sorry

end percent_increase_expenditure_l119_119027


namespace f_at_seven_l119_119002

variable {𝓡 : Type*} [CommRing 𝓡] [OrderedAddCommGroup 𝓡] [Module ℝ 𝓡]

-- Assuming f is a function from ℝ to ℝ with the given properties
variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function.
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Condition 2: f(x + 2) = -f(x) for all x.
def periodic_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = - f x 

-- Condition 3: f(x) = 2x^2 when x ∈ (0, 2)
def interval_definition (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_seven
  (h_odd : odd_function f)
  (h_periodic : periodic_negation f)
  (h_interval : interval_definition f) :
  f 7 = -2 :=
by
  sorry

end f_at_seven_l119_119002


namespace find_some_number_l119_119964

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l119_119964


namespace solve_eq1_solve_eq2_l119_119241

theorem solve_eq1 {x : ℝ} : 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5 := 
  sorry

theorem solve_eq2 {x : ℝ} : (x + 3)^3 = 64 ↔ x = 1 := 
  sorry

end solve_eq1_solve_eq2_l119_119241


namespace probability_cooking_selected_l119_119308

def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

theorem probability_cooking_selected : (1 / list.length courses) = 1 / 4 := by
  sorry

end probability_cooking_selected_l119_119308


namespace multiply_and_simplify_l119_119844
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l119_119844


namespace three_digit_sum_26_l119_119172

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_26 : 
  ∃! (n : ℕ), is_three_digit n ∧ digit_sum n = 26 := 
sorry

end three_digit_sum_26_l119_119172


namespace max_drumming_bunnies_l119_119900

structure Bunny where
  drum : ℕ
  drumsticks : ℕ

def can_drum (b1 b2 : Bunny) : Bool :=
  b1.drum > b2.drum ∧ b1.drumsticks > b2.drumsticks

theorem max_drumming_bunnies (bunnies : List Bunny) (h_size : bunnies.length = 7) : 
  ∃ (maxDrumming : ℕ), maxDrumming = 6 := 
by
  have h_drumming_limits : ∃ n, n ≤ 6 := 
    sorry -- Placeholder for the reasoning step
  use 6
  apply Eq.refl

-- Sorry is used to bypass the detailed proof, and placeholder comments indicate the steps needed for proof reasoning.

end max_drumming_bunnies_l119_119900


namespace temperature_on_tuesday_l119_119069

variable (T W Th F : ℕ)

-- Conditions
def cond1 : Prop := (T + W + Th) / 3 = 32
def cond2 : Prop := (W + Th + F) / 3 = 34
def cond3 : Prop := F = 44

-- Theorem statement
theorem temperature_on_tuesday : cond1 T W Th → cond2 W Th F → cond3 F → T = 38 :=
by
  sorry

end temperature_on_tuesday_l119_119069


namespace repeating_decimals_sum_l119_119592

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l119_119592


namespace no_ghost_not_multiple_of_p_l119_119999

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sequence_S (p : ℕ) (S : ℕ → ℕ) : Prop :=
  (is_prime p ∧ p % 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i < p → S i = i) ∧
  (∀ n, n ≥ p → (S n > S (n-1) ∧ 
    ∀ (a b c : ℕ), (a < b ∧ b < c ∧ c < n ∧ S a < S b ∧ S b < S c ∧
    S b - S a = S c - S b → false)))

def is_ghost (p : ℕ) (S : ℕ → ℕ) (g : ℕ) : Prop :=
  ∀ n : ℕ, S n ≠ g

theorem no_ghost_not_multiple_of_p (p : ℕ) (S : ℕ → ℕ) :
  (is_prime p ∧ p % 2 = 1) ∧ sequence_S p S → 
  ∀ g : ℕ, is_ghost p S g → p ∣ g :=
by 
  sorry

end no_ghost_not_multiple_of_p_l119_119999


namespace xiaoming_selects_cooking_probability_l119_119324

theorem xiaoming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"] in
  let num_courses := courses.length in
  num_courses = 4 → 
  (1 / num_courses : ℚ) = 1 / 4 :=
by
  assume courses : List String,
  assume num_courses : Nat,
  assume h : num_courses = 4,
  sorry

end xiaoming_selects_cooking_probability_l119_119324


namespace grandfather_age_l119_119125

theorem grandfather_age :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 10 * a + b = a + b^2 ∧ 10 * a + b = 89 :=
by
  sorry

end grandfather_age_l119_119125


namespace solution_set_of_inequality_l119_119252

theorem solution_set_of_inequality (x : ℝ) : 
  abs ((x + 2) / x) < 1 ↔ x < -1 :=
by
  sorry

end solution_set_of_inequality_l119_119252


namespace problem_correctness_l119_119001

theorem problem_correctness (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  (m = 2 ∨ m = -2) ∧ (m^2 + (a + b) / 2 + (- (x * y)) ^ 2023 = 3) := 
by
  sorry

end problem_correctness_l119_119001


namespace possible_values_of_n_l119_119126

-- Conditions: Definition of equilateral triangles and squares with side length 1
def equilateral_triangle_side_length_1 : Prop := ∀ (a : ℕ), 
  ∃ (triangle : ℕ), triangle * 60 = 180 * (a - 2)

def square_side_length_1 : Prop := ∀ (b : ℕ), 
  ∃ (square : ℕ), square * 90 = 180 * (b - 2)

-- Definition of convex n-sided polygon formed using these pieces
def convex_polygon_formed (n : ℕ) : Prop := 
  ∃ (a b c d : ℕ), 
    a + b + c + d = n ∧ 
    60 * a + 90 * b + 120 * c + 150 * d = 180 * (n - 2)

-- Equivalent proof problem
theorem possible_values_of_n :
  ∃ (n : ℕ), (5 ≤ n ∧ n ≤ 12) ∧ convex_polygon_formed n :=
sorry

end possible_values_of_n_l119_119126


namespace problem_statement_l119_119191

theorem problem_statement (m n : ℝ) :
  (m^2 - 1840 * m + 2009 = 0) → (n^2 - 1840 * n + 2009 = 0) → 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 := 
by
  intros h1 h2
  sorry

end problem_statement_l119_119191


namespace candles_must_be_odd_l119_119257

theorem candles_must_be_odd (n k : ℕ) (h : n * k = (n * (n + 1)) / 2) : n % 2 = 1 :=
by
  -- Given that the total burn time for all n candles = k * n
  -- And the sum of the first n natural numbers = (n * (n + 1)) / 2
  -- We have the hypothesis h: n * k = (n * (n + 1)) / 2
  -- We need to prove that n must be odd
  sorry

end candles_must_be_odd_l119_119257


namespace find_real_number_a_l119_119762

theorem find_real_number_a (a : ℝ) (h : (a^2 - 3*a + 2 = 0)) (h' : (a - 2) ≠ 0) : a = 1 :=
sorry

end find_real_number_a_l119_119762


namespace proof_problem_l119_119202

-- Defining a right triangle ΔABC with ∠BCA=90°
structure RightTriangle :=
(a b c : ℝ)  -- sides a, b, c with c as the hypotenuse
(hypotenuse_eq : c^2 = a^2 + b^2)  -- Pythagorean relation

-- Define the circles K1 and K2 with radii r1 and r2 respectively
structure CirclesOnTriangle (Δ : RightTriangle) :=
(r1 r2 : ℝ)  -- radii of the circles K1 and K2

-- Prove the relationship r1 + r2 = a + b - c
theorem proof_problem (Δ : RightTriangle) (C : CirclesOnTriangle Δ) :
  C.r1 + C.r2 = Δ.a + Δ.b - Δ.c := by
  sorry

end proof_problem_l119_119202


namespace talia_father_age_l119_119459

variable (talia_age : ℕ)
variable (mom_age : ℕ)
variable (dad_age : ℕ)

-- Conditions
def condition1 := talia_age + 7 = 20
def condition2 := mom_age = 3 * talia_age
def condition3 := dad_age + 3 = mom_age

-- Theorem to prove
theorem talia_father_age (h1 : condition1) (h2 : condition2) (h3 : condition3) : dad_age = 36 :=
by
  sorry

end talia_father_age_l119_119459


namespace tic_tac_toe_alex_wins_second_X_l119_119789

theorem tic_tac_toe_alex_wins_second_X :
  ∃ b : ℕ, b = 12 := 
sorry

end tic_tac_toe_alex_wins_second_X_l119_119789


namespace chicken_farm_l119_119112

def total_chickens (roosters hens : ℕ) : ℕ := roosters + hens

theorem chicken_farm (roosters hens : ℕ) (h1 : 2 * hens = roosters) (h2 : roosters = 6000) : 
  total_chickens roosters hens = 9000 :=
by
  sorry

end chicken_farm_l119_119112


namespace probability_of_selecting_cooking_l119_119352

theorem probability_of_selecting_cooking (courses : Finset String) (H : courses = {"planting", "cooking", "pottery", "carpentry"}) :
  (1 / (courses.card : ℝ)) = 1 / 4 :=
by
  have : courses.card = 4 := by rw [Finset.card_eq_four, H]
  sorry

end probability_of_selecting_cooking_l119_119352


namespace remainder_of_polynomial_l119_119158

-- Define the polynomial and the divisor
def f (x : ℝ) := x^3 - 4 * x + 6
def a := -3

-- State the theorem
theorem remainder_of_polynomial :
  f a = -9 := by
  sorry

end remainder_of_polynomial_l119_119158


namespace probability_select_cooking_l119_119322

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l119_119322


namespace eliminate_denominator_l119_119064

theorem eliminate_denominator (x : ℝ) : 6 - (x - 2) / 2 = x → 12 - x + 2 = 2 * x :=
by
  intro h
  sorry

end eliminate_denominator_l119_119064


namespace cubic_eq_one_real_root_l119_119948

-- Given a, b, c forming a geometric sequence
variables {a b c : ℝ}

-- Definition of a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Equation ax^3 + bx^2 + cx = 0
def cubic_eq (a b c x : ℝ) : Prop :=
  a * x^3 + b * x^2 + c * x = 0

-- Prove the number of real roots
theorem cubic_eq_one_real_root (h : geometric_sequence a b c) :
  ∃ x : ℝ, cubic_eq a b c x ∧ ¬∃ y ≠ x, cubic_eq a b c y :=
sorry

end cubic_eq_one_real_root_l119_119948


namespace person_birth_date_l119_119902

theorem person_birth_date
  (x : ℕ)
  (h1 : 1937 - x = x^2 - x)
  (d m : ℕ)
  (h2 : 44 + m = d^2)
  (h3 : 0 < m ∧ m < 13)
  (h4 : d = 7 ∧ m = 5) :
  (x = 44 ∧ 1937 - (x + x^2) = 1892) ∧  d = 7 ∧ m = 5 :=
by
  sorry

end person_birth_date_l119_119902


namespace range_of_x_l119_119188

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else Real.log (-x) / Real.log (1 / 2)

theorem range_of_x (x : ℝ) : f x > f (-x) ↔ (x > 1) ∨ (-1 < x ∧ x < 0) :=
by
  sorry

end range_of_x_l119_119188


namespace largest_common_value_less_than_500_for_sequences_l119_119665

-- Define the first arithmetic progression
def sequence1 (n : ℕ) : ℕ := 2 + 3 * n

-- Define the second arithmetic progression
def sequence2 (m : ℕ) : ℕ := 3 + 7 * m

-- Statement to prove the largest common value less than 500
theorem largest_common_value_less_than_500_for_sequences :
  ∃ (a : ℕ), a < 500 ∧ (∃ n, a = sequence1 n) ∧ (∃ m, a = sequence2 m) ∧ a = 479 :=
by
  sorry

end largest_common_value_less_than_500_for_sequences_l119_119665


namespace mary_screws_sections_l119_119049

def number_of_sections (initial_screws : Nat) (multiplier : Nat) (screws_per_section : Nat) : Nat :=
  let additional_screws := initial_screws * multiplier
  let total_screws := initial_screws + additional_screws
  total_screws / screws_per_section

theorem mary_screws_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end mary_screws_sections_l119_119049


namespace final_amoeba_is_blue_l119_119531

theorem final_amoeba_is_blue
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ)
  (merge : ∀ (a b : ℕ), a ≠ b → ∃ c, a + b - c = a ∧ a + b - c = b ∧ a + b - c = c)
  (initial_counts : n1 = 26 ∧ n2 = 31 ∧ n3 = 16)
  (final_count : ∃ a, a = 1) :
  ∃ color, color = "blue" := sorry

end final_amoeba_is_blue_l119_119531


namespace volume_of_new_cube_is_2744_l119_119711

-- Define the volume function for a cube given side length
def volume_of_cube (side : ℝ) : ℝ := side ^ 3

-- Given the original cube with a specific volume
def original_volume : ℝ := 343

-- Find the side length of the original cube by taking the cube root of the volume
def original_side_length := (original_volume : ℝ)^(1/3)

-- The side length of the new cube is twice the side length of the original cube
def new_side_length := 2 * original_side_length

-- The volume of the new cube should be calculated
def new_volume := volume_of_cube new_side_length

-- Theorem stating that the new volume is 2744 cubic feet
theorem volume_of_new_cube_is_2744 : new_volume = 2744 := sorry

end volume_of_new_cube_is_2744_l119_119711


namespace radius_of_circle_with_area_64pi_l119_119658

def circle_radius (A : ℝ) : ℝ := 
  real.sqrt (A / real.pi)

theorem radius_of_circle_with_area_64pi :
  circle_radius (64 * real.pi) = 8 :=
by sorry

end radius_of_circle_with_area_64pi_l119_119658


namespace repeating_decimals_sum_l119_119580

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l119_119580


namespace percentage_passed_eng_students_l119_119205

variable (total_male_students : ℕ := 120)
variable (total_female_students : ℕ := 100)
variable (total_international_students : ℕ := 70)
variable (total_disabilities_students : ℕ := 30)

variable (male_eng_percentage : ℕ := 25)
variable (female_eng_percentage : ℕ := 20)
variable (intern_eng_percentage : ℕ := 15)
variable (disab_eng_percentage : ℕ := 10)

variable (male_pass_percentage : ℕ := 20)
variable (female_pass_percentage : ℕ := 25)
variable (intern_pass_percentage : ℕ := 30)
variable (disab_pass_percentage : ℕ := 35)

def total_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100) +
  (total_female_students * female_eng_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100)

def total_passed_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100 * male_pass_percentage / 100) +
  (total_female_students * female_eng_percentage / 100 * female_pass_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100 * intern_pass_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100 * disab_pass_percentage / 100)

def passed_eng_students_percentage : ℕ :=
  total_passed_engineering_students * 100 / total_engineering_students

theorem percentage_passed_eng_students :
  passed_eng_students_percentage = 23 :=
sorry

end percentage_passed_eng_students_l119_119205


namespace repeating_decimal_sum_l119_119564

theorem repeating_decimal_sum :
  (let x := 2 / 9 in let y := 1 / 33 in x + y = 25 / 99) := sorry

end repeating_decimal_sum_l119_119564


namespace gcd_459_357_l119_119132

theorem gcd_459_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l119_119132


namespace slope_y_intercept_product_eq_neg_five_over_two_l119_119793

theorem slope_y_intercept_product_eq_neg_five_over_two :
  let A := (0, 10)
  let B := (0, 0)
  let C := (10, 0)
  let D := ((0 + 0) / 2, (10 + 0) / 2) -- midpoint of A and B
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope * y_intercept = -5 / 2 := 
by 
  sorry

end slope_y_intercept_product_eq_neg_five_over_two_l119_119793


namespace total_cans_collected_l119_119715

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end total_cans_collected_l119_119715


namespace chips_reach_end_l119_119474

theorem chips_reach_end (n k : ℕ) (h : n > k * 2^k) : True := sorry

end chips_reach_end_l119_119474


namespace probability_blue_or_purple_is_correct_l119_119134

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10 + 4

def blue_jelly_beans : ℕ := 10

def purple_jelly_beans : ℕ := 4

def blue_or_purple_jelly_beans : ℕ := blue_jelly_beans + purple_jelly_beans

def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_correct :
  probability_blue_or_purple = 7 / 19 :=
by
  sorry

end probability_blue_or_purple_is_correct_l119_119134


namespace probability_of_selecting_cooking_l119_119291

-- Define the four courses.
inductive Course
| planting
| cooking
| pottery
| carpentry

-- Define a random selection process.
def is_random_selection (s: List Course) : Prop :=
  ∀ x ∈ s, 1 = 1 -- This dummy definition should be replaced by appropriate randomness conditions.

-- Theorem statement
theorem probability_of_selecting_cooking :
  let courses := [Course.planting, Course.cooking, Course.pottery, Course.carpentry]
  in is_random_selection courses →
     (1 / (courses.length : ℝ)) = 1 / 4 := by
  intros courses h
  have h_length : courses.length = 4 := by simp
  rw h_length
  simp
  sorry -- Omit the proof steps

end probability_of_selecting_cooking_l119_119291


namespace sqrt_sum_l119_119450

theorem sqrt_sum (m n : ℝ) (h1 : m + n = 0) (h2 : m * n = -2023) : m + 2 * m * n + n = -4046 :=
by sorry

end sqrt_sum_l119_119450


namespace directrix_of_parabola_l119_119865

open Real

noncomputable def parabola_directrix (a : ℝ) : ℝ := -a / 4

theorem directrix_of_parabola (a : ℝ) (h : a = 4) : parabola_directrix a = -4 :=
by
  sorry

end directrix_of_parabola_l119_119865


namespace find_minimum_l119_119709

theorem find_minimum (a b c : ℝ) : ∃ (m : ℝ), m = min a (min b c) := 
  sorry

end find_minimum_l119_119709


namespace circle_line_distance_condition_l119_119201

theorem circle_line_distance_condition :
  ∀ (c : ℝ), 
    (∃ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 8 = 0 ∧ (x - y + c = 2 ∨ x - y + c = -2)) →
    -2*Real.sqrt 2 ≤ c ∧ c ≤ 2*Real.sqrt 2 := 
sorry

end circle_line_distance_condition_l119_119201


namespace multiply_and_simplify_l119_119845
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l119_119845


namespace percentage_reduction_price_increase_l119_119538

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l119_119538


namespace probability_of_selecting_cooking_is_one_fourth_l119_119379

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l119_119379
