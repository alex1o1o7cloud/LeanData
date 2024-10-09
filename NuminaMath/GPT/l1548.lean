import Mathlib

namespace min_value_of_m_l1548_154870

def ellipse (x y : ℝ) := (y^2 / 16) + (x^2 / 9) = 1
def line (x y m : ℝ) := y = x + m
def shortest_distance (d : ℝ) := d = Real.sqrt 2

theorem min_value_of_m :
  ∃ (m : ℝ), (∀ (x y : ℝ), ellipse x y → ∃ d, shortest_distance d ∧ line x y m) 
  ∧ ∀ m', m' < m → ¬(∃ (x y : ℝ), ellipse x y ∧ ∃ d, shortest_distance d ∧ line x y m') :=
sorry

end min_value_of_m_l1548_154870


namespace discount_percentage_l1548_154856

theorem discount_percentage (x : ℝ) : 
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  (marked_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount) = final_price) ↔ x = 20 := 
by
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  sorry

end discount_percentage_l1548_154856


namespace find_number_of_books_l1548_154879

-- Define the constants and equation based on the conditions
def price_paid_per_book : ℕ := 11
def price_sold_per_book : ℕ := 25
def total_difference : ℕ := 210

def books_equation (x : ℕ) : Prop :=
  (price_sold_per_book * x) - (price_paid_per_book * x) = total_difference

-- The theorem statement that needs to be proved
theorem find_number_of_books (x : ℕ) (h : books_equation x) : 
  x = 15 :=
sorry

end find_number_of_books_l1548_154879


namespace sum_first_15_odd_integers_l1548_154803

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l1548_154803


namespace parallel_vectors_l1548_154857

variable (y : ℝ)

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

theorem parallel_vectors (h : (-1 * y - 3 * 2) = 0) : y = -6 :=
by
  sorry

end parallel_vectors_l1548_154857


namespace prob_of_drawing_one_red_ball_distribution_of_X_l1548_154880

-- Definitions for conditions
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := red_balls + white_balls
def balls_drawn : ℕ := 3

-- Combinations 
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probabilities
noncomputable def prob_ex_one_red_ball : ℚ :=
  (combination red_balls 1 * combination white_balls 2) / combination total_balls balls_drawn

noncomputable def prob_X_0 : ℚ := (combination white_balls 3) / combination total_balls balls_drawn
noncomputable def prob_X_1 : ℚ := prob_ex_one_red_ball
noncomputable def prob_X_2 : ℚ := (combination red_balls 2 * combination white_balls 1) / combination total_balls balls_drawn

-- Theorem statements
theorem prob_of_drawing_one_red_ball : prob_ex_one_red_ball = 3/5 := by
  sorry

theorem distribution_of_X : prob_X_0 = 1/10 ∧ prob_X_1 = 3/5 ∧ prob_X_2 = 3/10 := by
  sorry

end prob_of_drawing_one_red_ball_distribution_of_X_l1548_154880


namespace num_positive_integers_satisfying_condition_l1548_154823

theorem num_positive_integers_satisfying_condition :
  ∃! (n : ℕ), 30 - 6 * n > 18 := by
  sorry

end num_positive_integers_satisfying_condition_l1548_154823


namespace find_softball_players_l1548_154862

def total_players : ℕ := 51
def cricket_players : ℕ := 10
def hockey_players : ℕ := 12
def football_players : ℕ := 16

def softball_players : ℕ := total_players - (cricket_players + hockey_players + football_players)

theorem find_softball_players : softball_players = 13 := 
by {
  sorry
}

end find_softball_players_l1548_154862


namespace rationalization_correct_l1548_154897

noncomputable def rationalize_denominator : Prop :=
  (∃ (a b : ℝ), a = (12:ℝ).sqrt + (5:ℝ).sqrt ∧ b = (3:ℝ).sqrt + (5:ℝ).sqrt ∧
                (a / b) = (((15:ℝ).sqrt - 1) / 2))

theorem rationalization_correct : rationalize_denominator :=
by {
  sorry
}

end rationalization_correct_l1548_154897


namespace length_of_platform_l1548_154891

variable (L : ℕ)

theorem length_of_platform
  (train_length : ℕ)
  (time_cross_post : ℕ)
  (time_cross_platform : ℕ)
  (train_length_eq : train_length = 300)
  (time_cross_post_eq : time_cross_post = 18)
  (time_cross_platform_eq : time_cross_platform = 39)
  : L = 350 := sorry

end length_of_platform_l1548_154891


namespace effective_annual_rate_l1548_154886

theorem effective_annual_rate (i : ℚ) (n : ℕ) (h_i : i = 0.16) (h_n : n = 2) :
  (1 + i / n) ^ n - 1 = 0.1664 :=
by {
  sorry
}

end effective_annual_rate_l1548_154886


namespace range_of_m_exacts_two_integers_l1548_154839

theorem range_of_m_exacts_two_integers (m : ℝ) :
  (∀ x : ℝ, (x - 2) / 4 < (x - 1) / 3 ∧ 2 * x - m ≤ 2 - x) ↔ -2 ≤ m ∧ m < 1 := 
sorry

end range_of_m_exacts_two_integers_l1548_154839


namespace nuts_division_pattern_l1548_154825

noncomputable def smallest_number_of_nuts : ℕ := 15621

theorem nuts_division_pattern :
  ∃ N : ℕ, N = smallest_number_of_nuts ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 
  (∃ M : ℕ, (N - k) % 4 = 0 ∧ (N - k) / 4 * 5 + 1 = N) := sorry

end nuts_division_pattern_l1548_154825


namespace max_plus_ten_min_eq_zero_l1548_154832

theorem max_plus_ten_min_eq_zero (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := max (x * y + x * z + y * z)
  let m := min (x * y + x * z + y * z)
  M + 10 * m = 0 :=
by
  sorry

end max_plus_ten_min_eq_zero_l1548_154832


namespace cost_of_plastering_l1548_154824

/-- 
Let's define the problem conditions
Length of the tank (in meters)
-/
def tank_length : ℕ := 25

/--
Width of the tank (in meters)
-/
def tank_width : ℕ := 12

/--
Depth of the tank (in meters)
-/
def tank_depth : ℕ := 6

/--
Cost of plastering per square meter (55 paise converted to rupees)
-/
def cost_per_sq_meter : ℝ := 0.55

/--
Prove that the cost of plastering the walls and bottom of the tank is 409.2 rupees
-/
theorem cost_of_plastering (total_cost : ℝ) : 
  total_cost = 409.2 :=
sorry

end cost_of_plastering_l1548_154824


namespace necessary_but_not_sufficient_l1548_154876

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x < 0 ∨ y < 0) → x + y < 0 :=
sorry

end necessary_but_not_sufficient_l1548_154876


namespace final_investment_amount_l1548_154830

noncomputable def final_amount (P1 P2 : ℝ) (r1 r2 t1 t2 n1 n2 : ℝ) : ℝ :=
  let A1 := P1 * (1 + r1 / n1) ^ (n1 * t1)
  let A2 := (A1 + P2) * (1 + r2 / n2) ^ (n2 * t2)
  A2

theorem final_investment_amount :
  final_amount 6000 2000 0.10 0.08 2 1.5 2 4 = 10467.05 :=
by
  sorry

end final_investment_amount_l1548_154830


namespace radius_of_given_spherical_circle_l1548_154850
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end radius_of_given_spherical_circle_l1548_154850


namespace arcsin_of_half_l1548_154842

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l1548_154842


namespace tens_digit_19_pow_1987_l1548_154833

theorem tens_digit_19_pow_1987 : (19 ^ 1987) % 100 / 10 = 3 := 
sorry

end tens_digit_19_pow_1987_l1548_154833


namespace sum_of_cubes_l1548_154892

theorem sum_of_cubes (x y : ℂ) (h1 : x + y = 1) (h2 : x * y = 1) : x^3 + y^3 = -2 := 
by 
  sorry

end sum_of_cubes_l1548_154892


namespace problem_l1548_154895

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * (Real.sqrt 3) * Real.cos x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let dot_product := (a x).fst * (b x).fst + (a x).snd * (b x).snd
  let magnitude_square_b := (b x).fst ^ 2 + (b x).snd ^ 2
  dot_product + magnitude_square_b

theorem problem :
  (∀ x, f x = 5 * Real.sin (2 * x + Real.pi / 6) + 7 / 2) ∧
  (∃ T, T = Real.pi) ∧ 
  (∃ x, f x = 17 / 2) ∧ 
  (∃ x, f x = -3 / 2) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), 0 ≤ x ∧ x ≤ Real.pi / 6) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, (2 * Real.pi / 3) ≤ x ∧ x ≤ Real.pi)
:= by
  sorry

end problem_l1548_154895


namespace unique_pair_prime_m_positive_l1548_154811

theorem unique_pair_prime_m_positive (p m : ℕ) (hp : Nat.Prime p) (hm : 0 < m) :
  p * (p + m) + p = (m + 1) ^ 3 → (p = 2 ∧ m = 1) :=
by
  sorry

end unique_pair_prime_m_positive_l1548_154811


namespace f_at_3_l1548_154890

noncomputable def f : ℝ → ℝ := sorry

lemma periodic (f : ℝ → ℝ) : ∀ x : ℝ, f (x + 4) = f x := sorry

lemma odd_function (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) + f x = 0 := sorry

lemma given_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = (x - 1)^2 := sorry

theorem f_at_3 : f 3 = 0 := 
by
  sorry

end f_at_3_l1548_154890


namespace simplify_expression_l1548_154840

theorem simplify_expression :
  (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end simplify_expression_l1548_154840


namespace unique_pegboard_arrangement_l1548_154819

/-- Conceptually, we will set up a function to count valid arrangements of pegs
based on the given conditions and prove that there is exactly one such arrangement. -/
def triangular_pegboard_arrangements (yellow red green blue orange black : ℕ) : ℕ :=
  if yellow = 6 ∧ red = 5 ∧ green = 4 ∧ blue = 3 ∧ orange = 2 ∧ black = 1 then 1 else 0

theorem unique_pegboard_arrangement :
  triangular_pegboard_arrangements 6 5 4 3 2 1 = 1 :=
by
  -- Placeholder for proof
  sorry

end unique_pegboard_arrangement_l1548_154819


namespace find_a_l1548_154812

def A : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem find_a (a : ℝ) (h : A ∩ B a = B a) : a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end find_a_l1548_154812


namespace asymptotes_of_hyperbola_l1548_154875

theorem asymptotes_of_hyperbola : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1 → y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l1548_154875


namespace geometric_sequence_identity_l1548_154802

variables {b : ℕ → ℝ} {m n p : ℕ}

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ i j k : ℕ, i < j → j < k → b j^2 = b i * b k

noncomputable def distinct_pos_ints (m n p : ℕ) :=
  0 < m ∧ 0 < n ∧ 0 < p ∧ m ≠ n ∧ n ≠ p ∧ p ≠ m

theorem geometric_sequence_identity 
  (h_geom : is_geometric_sequence b) 
  (h_distinct : distinct_pos_ints m n p) : 
  b p ^ (m - n) * b m ^ (n - p) * b n ^ (p - m) = 1 :=
sorry

end geometric_sequence_identity_l1548_154802


namespace A_investment_amount_l1548_154855

-- Conditions
variable (B_investment : ℝ) (C_investment : ℝ) (total_profit : ℝ) (A_profit : ℝ)
variable (B_investment_value : B_investment = 4200)
variable (C_investment_value : C_investment = 10500)
variable (total_profit_value : total_profit = 13600)
variable (A_profit_value : A_profit = 4080)

-- Proof statement
theorem A_investment_amount : 
  (∃ x : ℝ, x = 4410) :=
by
  sorry

end A_investment_amount_l1548_154855


namespace steve_total_payment_l1548_154807

def mike_dvd_cost : ℝ := 5
def steve_dvd_cost : ℝ := 2 * mike_dvd_cost
def additional_dvd_cost : ℝ := 7
def steve_additional_dvds : ℝ := 2 * additional_dvd_cost
def total_dvd_cost : ℝ := steve_dvd_cost + steve_additional_dvds
def shipping_cost : ℝ := 0.80 * total_dvd_cost
def subtotal_with_shipping : ℝ := total_dvd_cost + shipping_cost
def sales_tax : ℝ := 0.10 * subtotal_with_shipping
def total_amount_paid : ℝ := subtotal_with_shipping + sales_tax

theorem steve_total_payment : total_amount_paid = 47.52 := by
  sorry

end steve_total_payment_l1548_154807


namespace smallest_possible_abc_l1548_154805

open Nat

theorem smallest_possible_abc (a b c : ℕ)
  (h₁ : 5 * c ∣ a * b)
  (h₂ : 13 * a ∣ b * c)
  (h₃ : 31 * b ∣ a * c) :
  abc = 4060225 :=
by sorry

end smallest_possible_abc_l1548_154805


namespace complex_magnitude_difference_eq_one_l1548_154816

noncomputable def magnitude (z : Complex) : ℝ := Complex.abs z

/-- Lean 4 statement of the problem -/
theorem complex_magnitude_difference_eq_one (z₁ z₂ : Complex) (h₁ : magnitude z₁ = 1) (h₂ : magnitude z₂ = 1) (h₃ : magnitude (z₁ + z₂) = Real.sqrt 3) : magnitude (z₁ - z₂) = 1 := 
sorry

end complex_magnitude_difference_eq_one_l1548_154816


namespace complement_of_A_in_U_l1548_154809

-- Define the universal set U and the subset A
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 2, 5, 7}

-- Define the complement of A with respect to U
def complementU_A : Set Nat := {x ∈ U | x ∉ A}

-- Prove the complement of A in U is {3, 4, 6}
theorem complement_of_A_in_U :
  complementU_A = {3, 4, 6} :=
by
  sorry

end complement_of_A_in_U_l1548_154809


namespace hall_length_width_difference_l1548_154894

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L) 
  (h2 : L * W = 128) : 
  L - W = 8 :=
by
  sorry

end hall_length_width_difference_l1548_154894


namespace lorry_sand_capacity_l1548_154858

def cost_cement (bags : ℕ) (cost_per_bag : ℕ) : ℕ := bags * cost_per_bag
def total_cost (cement_cost : ℕ) (sand_cost : ℕ) : ℕ := cement_cost + sand_cost
def total_sand (sand_cost : ℕ) (cost_per_ton : ℕ) : ℕ := sand_cost / cost_per_ton
def sand_per_lorry (total_sand : ℕ) (lorries : ℕ) : ℕ := total_sand / lorries

theorem lorry_sand_capacity : 
  cost_cement 500 10 + (total_cost 5000 (total_sand 8000 40)) = 13000 ∧
  total_cost 5000 8000 = 13000 ∧
  total_sand 8000 40 = 200 ∧
  sand_per_lorry 200 20 = 10 :=
by
  sorry

end lorry_sand_capacity_l1548_154858


namespace max_price_per_unit_l1548_154896

-- Define the conditions
def original_price : ℝ := 25
def original_sales_volume : ℕ := 80000
def price_increase_effect (t : ℝ) : ℝ := 2000 * (t - original_price)
def new_sales_volume (t : ℝ) : ℝ := 130 - 2 * t

-- Define the condition for revenue
def revenue_condition (t : ℝ) : Prop :=
  t * new_sales_volume t ≥ original_price * original_sales_volume

-- Statement to prove the maximum price per unit
theorem max_price_per_unit : ∀ t : ℝ, revenue_condition t → t ≤ 40 := sorry

end max_price_per_unit_l1548_154896


namespace area_of_inscribed_square_l1548_154866

theorem area_of_inscribed_square (XY YZ : ℝ) (hXY : XY = 18) (hYZ : YZ = 30) :
  ∃ (s : ℝ), s^2 = 540 :=
by
  sorry

end area_of_inscribed_square_l1548_154866


namespace largest_power_of_2_dividing_n_l1548_154801

open Nat

-- Defining given expressions
def n : ℕ := 17^4 - 9^4 + 8 * 17^2

-- The theorem to prove
theorem largest_power_of_2_dividing_n : 2^3 ∣ n ∧ ∀ k, (k > 3 → ¬ 2^k ∣ n) :=
by
  sorry

end largest_power_of_2_dividing_n_l1548_154801


namespace Second_beats_Third_by_miles_l1548_154853

theorem Second_beats_Third_by_miles
  (v1 v2 v3 : ℝ) -- speeds of First, Second, and Third
  (H1 : (10 / v1) = (8 / v2)) -- First beats Second by 2 miles in 10-mile race
  (H2 : (10 / v1) = (6 / v3)) -- First beats Third by 4 miles in 10-mile race
  : (10 - (v3 * (10 / v2))) = 2.5 := 
sorry

end Second_beats_Third_by_miles_l1548_154853


namespace valid_license_plates_l1548_154826

def letters := 26
def digits := 10
def totalPlates := letters^3 * digits^4

theorem valid_license_plates : totalPlates = 175760000 := by
  sorry

end valid_license_plates_l1548_154826


namespace six_x_plus_four_eq_twenty_two_l1548_154827

theorem six_x_plus_four_eq_twenty_two (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := 
by
  sorry

end six_x_plus_four_eq_twenty_two_l1548_154827


namespace smallest_integer_solution_l1548_154863

theorem smallest_integer_solution (x : ℤ) : 
  (10 * x * x - 40 * x + 36 = 0) → x = 2 :=
sorry

end smallest_integer_solution_l1548_154863


namespace consecutive_odd_natural_numbers_sum_l1548_154851

theorem consecutive_odd_natural_numbers_sum (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : b = a + 6) 
  (h4 : c = a + 12) 
  (h5 : c = 27) 
  (h6 : a % 2 = 1) 
  (h7 : b % 2 = 1) 
  (h8 : c % 2 = 1) 
  (h9 : a % 3 = 0) 
  (h10 : b % 3 = 0) 
  (h11 : c % 3 = 0) 
  : a + b + c = 63 :=
by
  sorry

end consecutive_odd_natural_numbers_sum_l1548_154851


namespace factorization_of_polynomial_l1548_154818

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end factorization_of_polynomial_l1548_154818


namespace buy_tshirts_l1548_154882

theorem buy_tshirts
  (P T : ℕ)
  (h1 : 3 * P + 6 * T = 1500)
  (h2 : P + 12 * T = 1500)
  (budget : ℕ)
  (budget_eq : budget = 800) :
  (budget / T) = 8 := by
  sorry

end buy_tshirts_l1548_154882


namespace gcd_bezout_663_182_l1548_154828

theorem gcd_bezout_663_182 :
  let a := 182
  let b := 663
  ∃ d u v : ℤ, d = Int.gcd a b ∧ d = a * u + b * v ∧ d = 13 ∧ u = 11 ∧ v = -3 :=
by 
  let a := 182
  let b := 663
  use 13, 11, -3
  sorry

end gcd_bezout_663_182_l1548_154828


namespace adrien_winning_strategy_l1548_154810

/--
On the table, there are 2023 tokens. Adrien and Iris take turns removing at least one token and at most half of the remaining tokens at the time they play. The player who leaves a single token on the table loses the game. Adrien starts first. Prove that Adrien has a winning strategy.
-/
theorem adrien_winning_strategy : ∃ strategy : ℕ → ℕ, 
  ∀ n:ℕ, (n = 2023 ∧ 1 ≤ strategy n ∧ strategy n ≤ n / 2) → 
    (∀ u : ℕ, (u = n - strategy n) → (∃ strategy' : ℕ → ℕ , 
      ∀ m:ℕ, (m = u ∧ 1 ≤ strategy' m ∧ strategy' m ≤ m / 2) → 
        (∃ next_u : ℕ, (next_u = m - strategy' m → next_u ≠ 1 ∨ (m = 1 ∧ u ≠ 1 ∧ next_u = 1)))))
:= sorry

end adrien_winning_strategy_l1548_154810


namespace inequality_solution_set_no_positive_a_b_exists_l1548_154806

def f (x : ℝ) := abs (2 * x - 1) - abs (2 * x - 2)
def k := 1

theorem inequality_solution_set :
  { x : ℝ | f x ≥ x } = { x : ℝ | x ≤ -1 ∨ x = 1 } :=
sorry

theorem no_positive_a_b_exists (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ¬ (a + 2 * b = k ∧ 2 / a + 1 / b = 4 - 1 / (a * b)) :=
sorry

end inequality_solution_set_no_positive_a_b_exists_l1548_154806


namespace range_of_a_condition_l1548_154829

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem range_of_a_condition :
  range_of_a a → -1 < a ∧ a < 3 := sorry

end range_of_a_condition_l1548_154829


namespace coefficient_x2_term_l1548_154843

open Polynomial

noncomputable def poly1 : Polynomial ℝ := (X - 1)^3
noncomputable def poly2 : Polynomial ℝ := (X - 1)^4

theorem coefficient_x2_term :
  coeff (poly1 + poly2) 2 = 3 :=
sorry

end coefficient_x2_term_l1548_154843


namespace fraction_simplification_l1548_154814

theorem fraction_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) :
  (x / (x - 1) = 3 / (2 * x - 2) - 3) → (2 * x = 3 - 6 * x + 6) :=
by 
  intro h1
  -- Proof steps would be here, but we are using sorry
  sorry

end fraction_simplification_l1548_154814


namespace workshop_workers_transfer_l1548_154884

theorem workshop_workers_transfer (w d t : ℕ) (h_w : 63 ≤ w) (h_d : d ≤ 31) 
(h_prod : 1994 = 31 * w + t * (t + 1) / 2) : 
(d = 28 ∧ t = 10) ∨ (d = 30 ∧ t = 21) := sorry

end workshop_workers_transfer_l1548_154884


namespace additional_days_when_selling_5_goats_l1548_154898

variables (G D F X : ℕ)

def total_feed (num_goats days : ℕ) := G * num_goats * days

theorem additional_days_when_selling_5_goats
  (h1 : total_feed G 20 D = F)
  (h2 : total_feed G 15 (D + X) = F)
  (h3 : total_feed G 30 (D - 3) = F):
  X = 9 :=
by
  -- the exact proof is omitted and presented as 'sorry'
  sorry

end additional_days_when_selling_5_goats_l1548_154898


namespace oranges_in_bowl_l1548_154837

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end oranges_in_bowl_l1548_154837


namespace arithmetic_sequence_problem_l1548_154871

variable (a : ℕ → ℤ) -- The arithmetic sequence as a function from natural numbers to integers
variable (S : ℕ → ℤ) -- Sum of the first n terms of the sequence

-- Conditions
variable (h1 : S 8 = 4 * a 3) -- Sum of the first 8 terms is 4 times the third term
variable (h2 : a 7 = -2)      -- The seventh term is -2

-- Proven Goal
theorem arithmetic_sequence_problem : a 9 = -6 := 
by sorry -- This is a placeholder for the proof

end arithmetic_sequence_problem_l1548_154871


namespace integer_solutions_no_solutions_2891_l1548_154859

-- Define the main problem statement
-- Prove that if the equation x^3 - 3xy^2 + y^3 = n has a solution in integers x, y, then it has at least three such solutions.
theorem integer_solutions (n : ℕ) (x y : ℤ) (h : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x₁ y₁ x₂ y₂ : ℤ, x₁ ≠ x ∧ y₁ ≠ y ∧ x₂ ≠ x ∧ y₂ ≠ y ∧ 
  x₁^3 - 3 * x₁ * y₁^2 + y₁^3 = n ∧ 
  x₂^3 - 3 * x₂ * y₂^2 + y₂^3 = n := sorry

-- Prove that if n = 2891 then no such integer solutions exist.
theorem no_solutions_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) := sorry

end integer_solutions_no_solutions_2891_l1548_154859


namespace bala_age_difference_l1548_154813

theorem bala_age_difference 
  (a10 : ℕ) -- Anand's age 10 years ago.
  (b10 : ℕ) -- Bala's age 10 years ago.
  (h1 : a10 = b10 / 3) -- 10 years ago, Anand's age was one-third Bala's age.
  (h2 : a10 = 15 - 10) -- Anand was 5 years old 10 years ago, given his current age is 15.
  : (b10 + 10) - 15 = 10 := -- Bala is 10 years older than Anand.
sorry

end bala_age_difference_l1548_154813


namespace remainder_of_99_pow_36_mod_100_l1548_154854

theorem remainder_of_99_pow_36_mod_100 :
  (99 : ℤ)^36 % 100 = 1 := sorry

end remainder_of_99_pow_36_mod_100_l1548_154854


namespace how_many_tickets_left_l1548_154877

-- Define the conditions
def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define the total tickets won by Tom
def total_tickets : ℕ := tickets_from_whack_a_mole + tickets_from_skee_ball

-- State the theorem to be proved: how many tickets Tom has left
theorem how_many_tickets_left : total_tickets - tickets_spent_on_hat = 50 := by
  sorry

end how_many_tickets_left_l1548_154877


namespace simplify_fraction_l1548_154808

theorem simplify_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end simplify_fraction_l1548_154808


namespace total_distance_fourth_fifth_days_l1548_154878

theorem total_distance_fourth_fifth_days (d : ℕ) (total_distance : ℕ) (n : ℕ) (q : ℚ) 
  (S_6 : d * (1 - q^6) / (1 - q) = 378) (ratio : q = 1/2) (n_six : n = 6) : 
  (d * q^3) + (d * q^4) = 36 :=
by 
  sorry

end total_distance_fourth_fifth_days_l1548_154878


namespace cara_constant_speed_l1548_154815

noncomputable def cara_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

theorem cara_constant_speed
  ( distance : ℕ := 120 )
  ( dan_speed : ℕ := 40 )
  ( dan_time_offset : ℕ := 1 ) :
  cara_speed distance (3 + dan_time_offset) = 30 := 
by
  -- skip proof
  sorry

end cara_constant_speed_l1548_154815


namespace total_cost_of_video_games_l1548_154885

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end total_cost_of_video_games_l1548_154885


namespace last_digit_of_2_pow_2010_l1548_154836

-- Define the pattern of last digits of powers of 2
def last_digit_of_power_of_2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case is redundant as n % 4 ∈ {0, 1, 2, 3}

-- Main theorem stating the problem's assertion
theorem last_digit_of_2_pow_2010 : last_digit_of_power_of_2 2010 = 4 :=
by
  -- The proof is omitted
  sorry

end last_digit_of_2_pow_2010_l1548_154836


namespace sum_of_triangle_angles_l1548_154860

theorem sum_of_triangle_angles 
  (smallest largest middle : ℝ) 
  (h1 : smallest = 20) 
  (h2 : middle = 3 * smallest) 
  (h3 : largest = 5 * smallest) 
  (h4 : smallest + middle + largest = 180) :
  smallest + middle + largest = 180 :=
by sorry

end sum_of_triangle_angles_l1548_154860


namespace percent_decrease_in_hours_l1548_154838

variable {W H : ℝ} (W_nonzero : W ≠ 0) (H_nonzero : H ≠ 0)

theorem percent_decrease_in_hours
  (wage_increase : W' = 1.25 * W)
  (income_unchanged : W * H = W' * H')
  : (H' = 0.8 * H) → H' = H * (1 - 0.2) := by
  sorry

end percent_decrease_in_hours_l1548_154838


namespace prove_value_of_expression_l1548_154869

theorem prove_value_of_expression (x y a b : ℝ)
    (h1 : x = 2) 
    (h2 : y = 1)
    (h3 : 2 * a + b = 5)
    (h4 : a + 2 * b = 1) : 
    3 - a - b = 1 := 
by
    -- Skipping proof
    sorry

end prove_value_of_expression_l1548_154869


namespace carli_charlie_flute_ratio_l1548_154865

theorem carli_charlie_flute_ratio :
  let charlie_flutes := 1
  let charlie_horns := 2
  let charlie_harps := 1
  let carli_horns := charlie_horns / 2
  let total_instruments := 7
  ∃ (carli_flutes : ℕ), 
    (charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns = total_instruments) ∧ 
    (carli_flutes / charlie_flutes = 2) :=
by
  sorry

end carli_charlie_flute_ratio_l1548_154865


namespace find_solution_l1548_154834

-- Define the setup for the problem
variables (k x y : ℝ)

-- Conditions from the problem
def cond1 : Prop := x - y = 9 * k
def cond2 : Prop := x + y = 5 * k
def cond3 : Prop := 2 * x + 3 * y = 8

-- Proof statement combining all conditions to show the values of k, x, and y that satisfy them
theorem find_solution :
  cond1 k x y →
  cond2 k x y →
  cond3 x y →
  k = 1 ∧ x = 7 ∧ y = -2 := by
  sorry

end find_solution_l1548_154834


namespace mittens_in_each_box_l1548_154883

theorem mittens_in_each_box (boxes scarves_per_box total_clothing : ℕ) (h1 : boxes = 8) (h2 : scarves_per_box = 4) (h3 : total_clothing = 80) :
  ∃ (mittens_per_box : ℕ), mittens_per_box = 6 :=
by
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  let mittens_per_box := total_mittens / boxes
  use mittens_per_box
  sorry

end mittens_in_each_box_l1548_154883


namespace diophantine_soln_l1548_154800

-- Define the Diophantine equation as a predicate
def diophantine_eq (x y : ℤ) : Prop := x^3 - y^3 = 2 * x * y + 8

-- Theorem stating that the only solutions are (0, -2) and (2, 0)
theorem diophantine_soln :
  ∀ x y : ℤ, diophantine_eq x y ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end diophantine_soln_l1548_154800


namespace range_of_m_l1548_154841

open Set

def set_A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (3 * m - 2)}

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ m ≤ 4 :=
by sorry

end range_of_m_l1548_154841


namespace myOperation_identity_l1548_154873

variable {R : Type*} [LinearOrderedField R]

def myOperation (a b : R) : R := (a - b) ^ 2

theorem myOperation_identity (x y : R) : myOperation ((x - y) ^ 2) ((y - x) ^ 2) = 0 := 
by 
  sorry

end myOperation_identity_l1548_154873


namespace kim_gum_distribution_l1548_154849

theorem kim_gum_distribution (cousins : ℕ) (total_gum : ℕ) 
  (h1 : cousins = 4) (h2 : total_gum = 20) : 
  total_gum / cousins = 5 :=
by
  sorry

end kim_gum_distribution_l1548_154849


namespace distance_to_y_axis_l1548_154822

theorem distance_to_y_axis {x y : ℝ} (h : x = -3 ∧ y = 4) : abs x = 3 :=
by
  sorry

end distance_to_y_axis_l1548_154822


namespace smallest_sum_of_bases_l1548_154864

theorem smallest_sum_of_bases :
  ∃ (c d : ℕ), 8 * c + 9 = 9 * d + 8 ∧ c + d = 19 := 
by
  sorry

end smallest_sum_of_bases_l1548_154864


namespace cars_meet_time_l1548_154881

theorem cars_meet_time (t : ℝ) (highway_length : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ)
  (h1 : highway_length = 105) (h2 : speed_car1 = 15) (h3 : speed_car2 = 20) :
  15 * t + 20 * t = 105 → t = 3 := by
  sorry

end cars_meet_time_l1548_154881


namespace new_class_mean_l1548_154872

theorem new_class_mean (n1 n2 : ℕ) (mean1 mean2 : ℝ) (h1 : n1 = 45) (h2 : n2 = 5) (h3 : mean1 = 0.85) (h4 : mean2 = 0.90) : 
(n1 + n2 = 50) → 
((n1 * mean1 + n2 * mean2) / (n1 + n2) = 0.855) := 
by
  intro total_students
  sorry

end new_class_mean_l1548_154872


namespace time_to_Lake_Park_restaurant_l1548_154846

def time_to_Hidden_Lake := 15
def time_back_to_Park_Office := 7
def total_time_gone := 32

theorem time_to_Lake_Park_restaurant : 
  (total_time_gone = time_to_Hidden_Lake + time_back_to_Park_Office +
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office))) -> 
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office) = 10) := by
  intros 
  sorry

end time_to_Lake_Park_restaurant_l1548_154846


namespace evaluate_expression_l1548_154844

theorem evaluate_expression :
  (305^2 - 275^2) / 30 = 580 := 
by
  sorry

end evaluate_expression_l1548_154844


namespace solve_a_perpendicular_l1548_154889

theorem solve_a_perpendicular (a : ℝ) : 
  ((2 * a + 5) * (2 - a) + (a - 2) * (a + 3) = 0) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end solve_a_perpendicular_l1548_154889


namespace residue_11_pow_2021_mod_19_l1548_154804

theorem residue_11_pow_2021_mod_19 : (11^2021) % 19 = 17 := 
by
  -- this is to ensure the theorem is syntactically correct in Lean but skips the proof for now
  sorry

end residue_11_pow_2021_mod_19_l1548_154804


namespace longest_side_length_quadrilateral_l1548_154847

theorem longest_side_length_quadrilateral :
  (∀ (x y : ℝ),
    (x + y ≤ 4) ∧
    (2 * x + y ≥ 3) ∧
    (x ≥ 0) ∧
    (y ≥ 0)) →
  (∃ d : ℝ, d = 4 * Real.sqrt 2) :=
by sorry

end longest_side_length_quadrilateral_l1548_154847


namespace inequality_positive_reals_l1548_154899

theorem inequality_positive_reals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8) ∧ (1 / a^2 + 1 / b^2 + 8 * a * b = 8 → a = b ∧ a = 1/2) :=
by
  sorry

end inequality_positive_reals_l1548_154899


namespace pair_d_are_equal_l1548_154867

theorem pair_d_are_equal : -(2 ^ 3) = (-2) ^ 3 :=
by
  -- Detailed proof steps go here, but are omitted for this task.
  sorry

end pair_d_are_equal_l1548_154867


namespace peanuts_weight_l1548_154820

theorem peanuts_weight (total_snacks raisins : ℝ) (h_total : total_snacks = 0.5) (h_raisins : raisins = 0.4) : (total_snacks - raisins) = 0.1 :=
by
  rw [h_total, h_raisins]
  norm_num

end peanuts_weight_l1548_154820


namespace crucian_carps_heavier_l1548_154893

-- Variables representing the weights
variables (K O L : ℝ)

-- Given conditions
axiom weight_6K_lt_5O : 6 * K < 5 * O
axiom weight_6K_gt_10L : 6 * K > 10 * L

-- The proof statement
theorem crucian_carps_heavier : 2 * K > 3 * L :=
by
  -- Proof would go here
  sorry

end crucian_carps_heavier_l1548_154893


namespace brenda_distance_when_first_met_l1548_154835

theorem brenda_distance_when_first_met
  (opposite_points : ∀ (d : ℕ), d = 150) -- Starting at diametrically opposite points on a 300m track means distance is 150m
  (constant_speeds : ∀ (B S x : ℕ), B * x = S * x) -- Brenda/ Sally run at constant speed
  (meet_again : ∀ (d₁ d₂ : ℕ), d₁ + d₂ = 300 + 100) -- Together they run 400 meters when they meet again, additional 100m by Sally
  : ∃ (x : ℕ), x = 150 :=
  by
    sorry

end brenda_distance_when_first_met_l1548_154835


namespace compute_x_squared_first_compute_x_squared_second_l1548_154831

variable (x : ℝ)
variable (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1)

theorem compute_x_squared_first : 
  1 / (1 / x - 1 / (x + 1)) - x = x^2 :=
by
  sorry

theorem compute_x_squared_second : 
  1 / (1 / (x - 1) - 1 / x) + x = x^2 :=
by
  sorry

end compute_x_squared_first_compute_x_squared_second_l1548_154831


namespace prime_divides_factorial_difference_l1548_154874

theorem prime_divides_factorial_difference (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) : 
  p^5 ∣ (Nat.factorial p - p) := by
  sorry

end prime_divides_factorial_difference_l1548_154874


namespace total_difference_in_cups_l1548_154821

theorem total_difference_in_cups (h1: Nat) (h2: Nat) (h3: Nat) (hrs: Nat) : 
  h1 = 4 → h2 = 7 → h3 = 5 → hrs = 3 → 
  ((h2 * hrs - h1 * hrs) + (h3 * hrs - h1 * hrs) + (h2 * hrs - h3 * hrs)) = 18 :=
by
  intros h1_eq h2_eq h3_eq hrs_eq
  sorry

end total_difference_in_cups_l1548_154821


namespace point_in_third_quadrant_l1548_154887

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (-b < 0 ∧ a - 3 < 0) :=
by sorry

end point_in_third_quadrant_l1548_154887


namespace sum_of_digits_divisible_by_six_l1548_154852

theorem sum_of_digits_divisible_by_six (A B : ℕ) (h1 : 10 * A + B % 6 = 0) (h2 : A + B = 12) : A + B = 12 :=
by
  sorry

end sum_of_digits_divisible_by_six_l1548_154852


namespace product_increase_l1548_154845

theorem product_increase (a b c : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
  sorry

end product_increase_l1548_154845


namespace sum_gcd_lcm_75_4410_l1548_154848

theorem sum_gcd_lcm_75_4410 :
  Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end sum_gcd_lcm_75_4410_l1548_154848


namespace a_minus_b_eq_neg_9_or_neg_1_l1548_154888

theorem a_minus_b_eq_neg_9_or_neg_1 (a b : ℝ) (h₁ : |a| = 5) (h₂ : |b| = 4) (h₃ : a + b < 0) :
  a - b = -9 ∨ a - b = -1 :=
by
  sorry

end a_minus_b_eq_neg_9_or_neg_1_l1548_154888


namespace max_value_of_xy_l1548_154817

theorem max_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  xy ≤ 1 / 2 :=
sorry

end max_value_of_xy_l1548_154817


namespace no_integer_solutions_for_x2_minus_4y2_eq_2011_l1548_154861

theorem no_integer_solutions_for_x2_minus_4y2_eq_2011 :
  ∀ (x y : ℤ), x^2 - 4 * y^2 ≠ 2011 := by
sorry

end no_integer_solutions_for_x2_minus_4y2_eq_2011_l1548_154861


namespace carpet_width_l1548_154868

theorem carpet_width
  (carpet_percentage : ℝ)
  (living_room_area : ℝ)
  (carpet_length : ℝ) :
  carpet_percentage = 0.30 →
  living_room_area = 120 →
  carpet_length = 9 →
  carpet_percentage * living_room_area / carpet_length = 4 :=
by
  sorry

end carpet_width_l1548_154868
