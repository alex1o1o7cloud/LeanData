import Mathlib

namespace rink_rent_cost_l136_136185

theorem rink_rent_cost (admission_fee cost_new_skates visits : ℝ) (h1 : admission_fee = 5) 
(h2 : cost_new_skates = 65) (h3 : visits = 26) : 
  let x := (65 / 26) in $5 + (26 * x) = 130) :=
by
  sorry

end rink_rent_cost_l136_136185


namespace center_polar_coordinates_l136_136092

-- Assuming we have a circle defined in polar coordinates
def polar_circle_center (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

-- The goal is to prove that the center of this circle has the polar coordinates (sqrt 2, π/4)
theorem center_polar_coordinates : ∃ ρ θ, polar_circle_center ρ θ ∧ ρ = Real.sqrt 2 ∧ θ = Real.pi / 4 :=
sorry

end center_polar_coordinates_l136_136092


namespace problem1_problem2_l136_136953

-- 1. Definitions for arc length and area
def arc_length (α : ℝ) (R : ℝ) := α * R
def sector_area (α : ℝ) (R : ℝ) := (1/2) * α * R^2
def triangle_area (R : ℝ) (α : ℝ) := (1/2) * R^2 * Real.sin α
def segment_area (α : ℝ) (R : ℝ) := sector_area α R - triangle_area R α

-- 2. Problem 1: Prove the arc length and the segment area for given R = 10 and α = 60°
theorem problem1 (R : ℝ) (α : ℝ) (hα : α = π / 3) (hR : R = 10) :
  arc_length α R = (10 / 3) * π ∧ segment_area α R = 50 * (π / 3 - Real.sqrt 3 / 2) :=
by sorry

-- 3. Problem 2: Prove the α that maximizes the area of the sector with a given perimeter C
def perimeter (α : ℝ) (R : ℝ) := 2 * R + arc_length α R

theorem problem2 (C : ℝ) (hC : C > 0) :
  Exists (λ α : ℝ, α = 2 ∧ 
  ∀ (R : ℝ) (H : perimeter α R = C), sector_area α R = (C^2) / 16) :=
by sorry

end problem1_problem2_l136_136953


namespace solve_car_production_l136_136893

def car_production_problem : Prop :=
  ∃ (NorthAmericaCars : ℕ) (TotalCars : ℕ) (EuropeCars : ℕ),
    NorthAmericaCars = 3884 ∧
    TotalCars = 6755 ∧
    EuropeCars = TotalCars - NorthAmericaCars ∧
    EuropeCars = 2871

theorem solve_car_production : car_production_problem := by
  sorry

end solve_car_production_l136_136893


namespace evaluate_expression_l136_136927

theorem evaluate_expression : (Real.sqrt ((Real.sqrt 2)^4))^6 = 64 := by
  sorry

end evaluate_expression_l136_136927


namespace range_of_m_l136_136792

theorem range_of_m (m : ℝ) (h : 1 < (8 - m) / (m - 5)) : 5 < m ∧ m < 13 / 2 :=
sorry

end range_of_m_l136_136792


namespace marble_ratio_l136_136906

-- Let Allison, Angela, and Albert have some number of marbles denoted by variables.
variable (Albert Angela Allison : ℕ)

-- Given conditions.
axiom h1 : Angela = Allison + 8
axiom h2 : Allison = 28
axiom h3 : Albert + Allison = 136

-- Prove that the ratio of the number of marbles Albert has to the number of marbles Angela has is 3.
theorem marble_ratio : Albert / Angela = 3 := by
  sorry

end marble_ratio_l136_136906


namespace opposite_of_neg_2023_l136_136599

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136599


namespace number_of_k_pop_sequences_l136_136107

noncomputable def is_k_pop_sequence (k : ℕ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = ({a m | m < n+k}.to_finset.card : ℕ)

theorem number_of_k_pop_sequences (k : ℕ) (hk : 0 < k) :
  ∃ f : ℕ → ℕ, (∀ k, f k = 2^k) ∧ (∑ (_, _) in (finset.range (k) × finset.univ), f _) = f k :=
sorry

end number_of_k_pop_sequences_l136_136107


namespace cost_of_items_l136_136415

theorem cost_of_items {x y z : ℕ} (h1 : x + 3 * y + 2 * z = 98)
                      (h2 : 3 * x + y = 5 * z - 36)
                      (even_x : x % 2 = 0) :
  x = 4 ∧ y = 22 ∧ z = 14 := 
by
  sorry

end cost_of_items_l136_136415


namespace find_quarters_l136_136898

def num_pennies := 123
def num_nickels := 85
def num_dimes := 35
def cost_per_scoop_cents := 300  -- $3 = 300 cents
def num_family_members := 5
def leftover_cents := 48

def total_cost_cents := num_family_members * cost_per_scoop_cents
def total_initial_cents := total_cost_cents + leftover_cents

-- Values of coins in cents
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25

def total_pennies_value := num_pennies * penny_value
def total_nickels_value := num_nickels * nickel_value
def total_dimes_value := num_dimes * dime_value
def total_initial_excluding_quarters := total_pennies_value + total_nickels_value + total_dimes_value

def total_quarters_value := total_initial_cents - total_initial_excluding_quarters
def num_quarters := total_quarters_value / quarter_value

theorem find_quarters : num_quarters = 26 := by
  sorry

end find_quarters_l136_136898


namespace pascal_remaining_distance_l136_136416

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l136_136416


namespace min_value_proof_l136_136060

noncomputable def min_expr_value (x y : ℝ) : ℝ :=
  (1 / (2 * x)) + (1 / y)

theorem min_value_proof (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  min_expr_value x y = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_proof_l136_136060


namespace value_of_b_minus_d_squared_l136_136016

variable {a b c d : ℤ}

theorem value_of_b_minus_d_squared (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 3) : (b - d) ^ 2 = 25 := 
by
  sorry

end value_of_b_minus_d_squared_l136_136016


namespace coords_of_a_in_m_n_l136_136236

variable {R : Type} [Field R]

def coords_in_basis (a : R × R) (p q : R × R) (c1 c2 : R) : Prop :=
  a = c1 • p + c2 • q

theorem coords_of_a_in_m_n
  (a p q m n : R × R)
  (hp : p = (1, -1)) (hq : q = (2, 1)) (hm : m = (-1, 1)) (hn : n = (1, 2))
  (coords_pq : coords_in_basis a p q (-2) 2) :
  coords_in_basis a m n 0 2 :=
by
  sorry

end coords_of_a_in_m_n_l136_136236


namespace total_tickets_needed_l136_136846

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end total_tickets_needed_l136_136846


namespace opposite_of_neg_2023_l136_136672

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136672


namespace watch_loss_percentage_l136_136181

theorem watch_loss_percentage 
  (cost_price : ℕ) (gain_percent : ℕ) (extra_amount : ℕ) (selling_price_loss : ℕ)
  (h_cost_price : cost_price = 2500)
  (h_gain_percent : gain_percent = 10)
  (h_extra_amount : extra_amount = 500)
  (h_gain_condition : cost_price + gain_percent * cost_price / 100 = selling_price_loss + extra_amount) :
  (cost_price - selling_price_loss) * 100 / cost_price = 10 := 
by 
  sorry

end watch_loss_percentage_l136_136181


namespace expand_polynomial_l136_136774

theorem expand_polynomial :
  (2 * t^2 - 3 * t + 2) * (-3 * t^2 + t - 5) = -6 * t^4 + 11 * t^3 - 19 * t^2 + 17 * t - 10 :=
by sorry

end expand_polynomial_l136_136774


namespace square_area_increase_l136_136886

variable (s : ℝ)

theorem square_area_increase (h : s > 0) : 
  let s_new := 1.30 * s
  let A_original := s^2
  let A_new := s_new^2
  let percentage_increase := ((A_new - A_original) / A_original) * 100
  percentage_increase = 69 := by
sorry

end square_area_increase_l136_136886


namespace opposite_of_negative_2023_l136_136706

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136706


namespace dolphins_trained_next_month_l136_136290

theorem dolphins_trained_next_month
  (total_dolphins : ℕ) 
  (one_fourth_fully_trained : ℚ) 
  (two_thirds_in_training : ℚ)
  (h1 : total_dolphins = 20)
  (h2 : one_fourth_fully_trained = 1 / 4) 
  (h3 : two_thirds_in_training = 2 / 3) :
  (total_dolphins - total_dolphins * one_fourth_fully_trained) * two_thirds_in_training = 10 := 
by 
  sorry

end dolphins_trained_next_month_l136_136290


namespace opposite_of_neg_2023_l136_136671

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136671


namespace ratio_Sachin_Rahul_l136_136851

-- Definitions: Sachin's age (S) is 63, and Sachin is younger than Rahul by 18 years.
def Sachin_age : ℕ := 63
def Rahul_age : ℕ := Sachin_age + 18

-- The problem: Prove the ratio of Sachin's age to Rahul's age is 7/9.
theorem ratio_Sachin_Rahul : (Sachin_age : ℚ) / (Rahul_age : ℚ) = 7 / 9 :=
by 
  -- The proof will go here, but we are skipping the proof as per the instructions.
  sorry

end ratio_Sachin_Rahul_l136_136851


namespace find_six_digit_number_l136_136029

theorem find_six_digit_number (a b c d e f : ℕ) (N : ℕ) :
  a = 1 ∧ f = 7 ∧
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
  (f - 1) * 10^5 + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e = 5 * N →
  N = 142857 :=
by
  sorry

end find_six_digit_number_l136_136029


namespace fraction_problem_l136_136204

theorem fraction_problem :
  (1 / 4 + 3 / 8) - 1 / 8 = 1 / 2 :=
by
  -- The proof steps are skipped
  sorry

end fraction_problem_l136_136204


namespace opposite_of_neg_2023_l136_136687

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136687


namespace no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l136_136117

-- Define the context for real numbers and the main property P.
def property_P (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + f (x + 2) ≤ 2 * f (x + 1)

-- For part (1)
theorem no_exp_function_satisfies_P (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = a^x) ∧ property_P f :=
sorry

-- Define the context for natural numbers, d(x), and main properties related to P.
def d (f : ℕ → ℕ) (x : ℕ) : ℕ := f (x + 1) - f x

-- For part (2)(i)
theorem d_decreasing_nonnegative (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∀ x : ℕ, d f (x + 1) ≤ d f x ∧ d f x ≥ 0 :=
sorry

-- For part (2)(ii)
theorem exists_c_infinitely_many (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ c : ℕ, 0 ≤ c ∧ c ≤ d f 1 ∧ ∀ N : ℕ, ∃ n : ℕ, n > N ∧ d f n = c :=
sorry

end no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l136_136117


namespace initial_investment_l136_136326

theorem initial_investment
  (P r : ℝ)
  (h1 : P + (P * r * 2) / 100 = 600)
  (h2 : P + (P * r * 7) / 100 = 850) :
  P = 500 :=
sorry

end initial_investment_l136_136326


namespace paul_account_balance_l136_136431

variable (initial_balance : ℝ) (transfer1 : ℝ) (transfer2 : ℝ) (service_charge_rate : ℝ)

def final_balance (init_bal transfer1 transfer2 rate : ℝ) : ℝ :=
  let charge1 := transfer1 * rate
  let total_deduction := transfer1 + charge1
  init_bal - total_deduction

theorem paul_account_balance :
  initial_balance = 400 →
  transfer1 = 90 →
  transfer2 = 60 →
  service_charge_rate = 0.02 →
  final_balance 400 90 60 0.02 = 308.2 :=
by
  intros h1 h2 h3 h4
  rw [final_balance, h1, h2, h4]
  norm_num

end paul_account_balance_l136_136431


namespace smallest_triangle_perimeter_l136_136881

theorem smallest_triangle_perimeter : ∃ (a b c : ℕ), a = 3 ∧ b = a + 1 ∧ c = b + 1 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 12 := by
  sorry

end smallest_triangle_perimeter_l136_136881


namespace WillyLucyHaveMoreCrayons_l136_136307

-- Definitions from the conditions
def WillyCrayons : ℕ := 1400
def LucyCrayons : ℕ := 290
def MaxCrayons : ℕ := 650

-- Theorem statement
theorem WillyLucyHaveMoreCrayons : WillyCrayons + LucyCrayons - MaxCrayons = 1040 := 
by 
  sorry

end WillyLucyHaveMoreCrayons_l136_136307


namespace triangle_is_isosceles_right_triangle_l136_136721

theorem triangle_is_isosceles_right_triangle
  (a b c : ℝ)
  (h1 : (a - b)^2 + (Real.sqrt (2 * a - b - 3)) + (abs (c - 3 * Real.sqrt 2)) = 0) :
  (a = 3) ∧ (b = 3) ∧ (c = 3 * Real.sqrt 2) :=
by
  sorry

end triangle_is_isosceles_right_triangle_l136_136721


namespace problem_solution_l136_136989

theorem problem_solution (a b c d x : ℚ) 
  (h1 : 2 * a + 2 = x) 
  (h2 : 3 * b + 3 = x) 
  (h3 : 4 * c + 4 = x) 
  (h4 : 5 * d + 5 = x) 
  (h5 : 2 * a + 3 * b + 4 * c + 5 * d + 6 = x) 
  : 2 * a + 3 * b + 4 * c + 5 * d = -10 / 3 := 
by 
  sorry

end problem_solution_l136_136989


namespace perfect_square_trinomial_m_l136_136238

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2*(m-3)*x + 16) = (x + a)^2) ↔ (m = 7 ∨ m = -1) := 
sorry

end perfect_square_trinomial_m_l136_136238


namespace wilfred_carrots_on_tuesday_l136_136159

theorem wilfred_carrots_on_tuesday :
  ∀ (carrots_eaten_Wednesday carrots_eaten_Thursday total_carrots desired_total: ℕ),
    carrots_eaten_Wednesday = 6 →
    carrots_eaten_Thursday = 5 →
    desired_total = 15 →
    desired_total - (carrots_eaten_Wednesday + carrots_eaten_Thursday) = 4 :=
by
  intros
  sorry

end wilfred_carrots_on_tuesday_l136_136159


namespace opposite_of_neg_2023_l136_136695

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136695


namespace opposite_of_neg_2023_l136_136451

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136451


namespace pairing_probability_l136_136089

variable {students : Fin 28} (Alex Jamie : Fin 28)

theorem pairing_probability (h1 : ∀ (i j : Fin 28), i ≠ j) :
  ∃ p : ℚ, p = 1 / 27 ∧ 
  (∃ (A_J_pairs : Finset (Fin 28) × Finset (Fin 28)),
  A_J_pairs.1 = {Alex} ∧ A_J_pairs.2 = {Jamie}) -> p = 1 / 27
:= sorry

end pairing_probability_l136_136089


namespace max_value_of_sin_l136_136879

theorem max_value_of_sin (x : ℝ) : (2 * Real.sin x) ≤ 2 :=
by
  -- this theorem directly implies that 2sin(x) has a maximum value of 2.
  sorry

end max_value_of_sin_l136_136879


namespace digit_difference_one_l136_136030

theorem digit_difference_one (p q : ℕ) (h_pq : p < 10 ∧ q < 10) (h_diff : (10 * p + q) - (10 * q + p) = 9) :
  p - q = 1 :=
by
  sorry

end digit_difference_one_l136_136030


namespace opposite_neg_2023_l136_136537

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136537


namespace Delta_15_xDelta_eq_neg_15_l136_136784

-- Definitions of the operations based on conditions
def xDelta (x : ℝ) : ℝ := 9 - x
def Delta (x : ℝ) : ℝ := x - 9

-- Statement that we need to prove
theorem Delta_15_xDelta_eq_neg_15 : Delta (xDelta 15) = -15 :=
by
  -- The proof will go here
  sorry

end Delta_15_xDelta_eq_neg_15_l136_136784


namespace inequality_transitive_l136_136158

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by
  sorry

end inequality_transitive_l136_136158


namespace chip_price_reduction_equation_l136_136056

-- Define initial price
def initial_price : ℝ := 400

-- Define final price after reductions
def final_price : ℝ := 144

-- Define the price reduction percentage
variable (x : ℝ)

-- The equation we need to prove
theorem chip_price_reduction_equation :
  initial_price * (1 - x) ^ 2 = final_price :=
sorry

end chip_price_reduction_equation_l136_136056


namespace initial_tomatoes_l136_136175

theorem initial_tomatoes (T : ℕ) (picked : ℕ) (remaining_total : ℕ) (potatoes : ℕ) :
  potatoes = 12 →
  picked = 53 →
  remaining_total = 136 →
  T + picked = remaining_total - potatoes →
  T = 71 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_tomatoes_l136_136175


namespace opposite_of_neg_2023_l136_136477

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136477


namespace correct_operations_l136_136736

theorem correct_operations :
  (∀ {a b : ℝ}, -(-a + b) = a + b → False) ∧
  (∀ {a : ℝ}, 3 * a^3 - 3 * a^2 = a → False) ∧
  (∀ {x : ℝ}, (x^6)^2 = x^8 → False) ∧
  (∀ {z : ℝ}, 1 / (2 / 3 : ℝ)⁻¹ = 2 / 3) :=
by
  sorry

end correct_operations_l136_136736


namespace intersection_A_B_l136_136230

def A : Set ℝ := { y | ∃ x : ℝ, y = |x| }
def B : Set ℝ := { y | ∃ x : ℝ, y = 1 - 2*x - x^2 }

theorem intersection_A_B :
  A ∩ B = { y | 0 ≤ y ∧ y ≤ 2 } :=
sorry

end intersection_A_B_l136_136230


namespace alpha_necessary_not_sufficient_for_beta_l136_136968

def alpha (x : ℝ) : Prop := x^2 = 4
def beta (x : ℝ) : Prop := x = 2

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x : ℝ, beta x → alpha x) ∧ ¬(∀ x : ℝ, alpha x → beta x) :=
by
  sorry

end alpha_necessary_not_sufficient_for_beta_l136_136968


namespace math_club_partition_l136_136146

def is_played (team : Finset ℕ) (A B C : ℕ) : Bool :=
(A ∈ team ∧ B ∉ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∈ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∉ team ∧ C ∈ team) ∨ 
(A ∈ team ∧ B ∈ team ∧ C ∈ team)

theorem math_club_partition 
  (students : Finset ℕ) (A B C : ℕ) 
  (h_size : students.card = 24)
  (teams : List (Finset ℕ))
  (h_teams : teams.length = 4)
  (h_team_size : ∀ t ∈ teams, t.card = 6)
  (h_partition : ∀ t ∈ teams, t ⊆ students) :
  ∃ (teams_played : List (Finset ℕ)), teams_played.length = 1 ∨ teams_played.length = 3 :=
sorry

end math_club_partition_l136_136146


namespace arithmetic_expression_equiv_l136_136190

theorem arithmetic_expression_equiv :
  (-1:ℤ)^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end arithmetic_expression_equiv_l136_136190


namespace express_x_in_terms_of_y_l136_136787

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) : x = (4 * y + 8) / 3 :=
sorry

end express_x_in_terms_of_y_l136_136787


namespace cost_of_nuts_l136_136905

/--
Adam bought 3 kilograms of nuts and 2.5 kilograms of dried fruits at a store. 
One kilogram of nuts costs a certain amount N and one kilogram of dried fruit costs $8. 
His purchases cost $56. Prove that one kilogram of nuts costs $12.
-/
theorem cost_of_nuts (N : ℝ) 
  (h1 : 3 * N + 2.5 * 8 = 56) 
  : N = 12 := by
  sorry

end cost_of_nuts_l136_136905


namespace ant_prob_bottom_vertex_l136_136915

theorem ant_prob_bottom_vertex :
  let top := 1
  let first_layer := 4
  let second_layer := 4
  let bottom := 1
  let prob_first_layer := 1 / first_layer
  let prob_second_layer := 1 / second_layer
  let prob_bottom := 1 / (second_layer + bottom)
  prob_first_layer * prob_second_layer * prob_bottom = 1 / 80 :=
by
  sorry

end ant_prob_bottom_vertex_l136_136915


namespace abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l136_136075

variable (x b : ℝ)

theorem abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1 :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := sorry

end abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l136_136075


namespace masha_can_climb_10_steps_l136_136055

def ways_to_climb_stairs : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => ways_to_climb_stairs (n + 1) + ways_to_climb_stairs n

theorem masha_can_climb_10_steps : ways_to_climb_stairs 10 = 89 :=
by
  -- proof omitted here as per instruction
  sorry

end masha_can_climb_10_steps_l136_136055


namespace stickers_per_student_l136_136123

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end stickers_per_student_l136_136123


namespace evaluate_polynomial_at_2_l136_136928

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 1) = 31 := 
by 
  sorry

end evaluate_polynomial_at_2_l136_136928


namespace leading_coeff_of_100_degree_polynomial_l136_136847

open Polynomial

-- Define the Fibonacci sequence (using an existing library)
def fib : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Define the problem statement
theorem leading_coeff_of_100_degree_polynomial (p : Polynomial ℤ)
  (h_deg : p.degree = 100)
  (h_vals : ∀ n, 1 ≤ n → n ≤ 102 → p.eval n = fib n) :
  p.leading_coeff = 1 / (100.factorial : ℤ) := 
sorry

end leading_coeff_of_100_degree_polynomial_l136_136847


namespace meiosis_fertilization_stability_l136_136139

def maintains_chromosome_stability (x : String) : Prop :=
  x = "Meiosis and Fertilization"

theorem meiosis_fertilization_stability :
  maintains_chromosome_stability "Meiosis and Fertilization" :=
by
  sorry

end meiosis_fertilization_stability_l136_136139


namespace exists_divisible_by_2021_l136_136833

def concatenated_number (n m : ℕ) : ℕ := 
  -- This function should concatenate the digits from n to m inclusively
  sorry

theorem exists_divisible_by_2021 : ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concatenated_number n m :=
by
  sorry

end exists_divisible_by_2021_l136_136833


namespace sum_interest_l136_136759

noncomputable def simple_interest (P : ℝ) (R : ℝ) := P * R * 3 / 100

theorem sum_interest (P R : ℝ) (h : simple_interest P (R + 1) - simple_interest P R = 75) : P = 2500 :=
by
  sorry

end sum_interest_l136_136759


namespace arrangements_correctness_l136_136198

noncomputable def arrangements_of_groups (total mountaineers : ℕ) (familiar_with_route : ℕ) (required_in_each_group : ℕ) : ℕ :=
  sorry

theorem arrangements_correctness :
  arrangements_of_groups 10 4 2 = 120 :=
sorry

end arrangements_correctness_l136_136198


namespace opposite_of_neg_2023_l136_136586

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136586


namespace graph_paper_problem_l136_136137

theorem graph_paper_problem :
  let line_eq := ∀ x y : ℝ, 7 * x + 268 * y = 1876
  ∃ (n : ℕ), 
  (∀ x y : ℕ, 0 < x ∧ x ≤ 268 ∧ 0 < y ∧ y ≤ 7 ∧ (7 * (x:ℝ) + 268 * (y:ℝ)) < 1876) →
  n = 801 :=
by
  sorry

end graph_paper_problem_l136_136137


namespace opposite_of_neg_2023_l136_136691

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136691


namespace polynomial_coeff_sum_l136_136947

theorem polynomial_coeff_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) (h : (2 * x - 3) ^ 5 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 160 :=
sorry

end polynomial_coeff_sum_l136_136947


namespace opposite_of_neg_2023_l136_136522

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136522


namespace function_zero_solution_l136_136345

-- Define the statement of the problem
theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → ∀ y : ℝ, f (x ^ 2 + y) ≥ (1 / x + 1) * f y) →
  (∀ x : ℝ, f x = 0) :=
by
  -- The proof of this theorem will be inserted here.
  sorry

end function_zero_solution_l136_136345


namespace opposite_neg_2023_l136_136531

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136531


namespace positive_value_m_l136_136942

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l136_136942


namespace gumball_water_wednesday_l136_136805

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end gumball_water_wednesday_l136_136805


namespace relationship_between_abc_l136_136253

noncomputable def a : Real := (2 / 5) ^ (3 / 5)
noncomputable def b : Real := (2 / 5) ^ (2 / 5)
noncomputable def c : Real := (3 / 5) ^ (3 / 5)

theorem relationship_between_abc : a < b ∧ b < c := by
  sorry

end relationship_between_abc_l136_136253


namespace solve_for_N_l136_136718

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end solve_for_N_l136_136718


namespace compare_f_ln_l136_136115

variable {f : ℝ → ℝ}

theorem compare_f_ln (h : ∀ x : ℝ, deriv f x > f x) : 3 * f (Real.log 2) < 2 * f (Real.log 3) :=
by
  sorry

end compare_f_ln_l136_136115


namespace find_W_l136_136757

noncomputable def volumeOutsideCylinder (r_cylinder r_sphere : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (r_sphere^2 - r_cylinder^2)
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h
  V_sphere - V_cylinder

theorem find_W : 
  volumeOutsideCylinder 4 7 = (1372 / 3 - 32 * Real.sqrt 33) * Real.pi :=
by
  sorry

end find_W_l136_136757


namespace opposite_of_neg_2023_l136_136475

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136475


namespace opposite_of_neg_2023_l136_136507

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136507


namespace pascal_remaining_miles_l136_136421

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l136_136421


namespace pascal_remaining_miles_l136_136420

section PascalCyclingTrip

variable (D T : ℝ)
variable (V : ℝ := 8)
variable (V_reduced V_increased : ℝ)
variable (time_diff : ℝ := 16)

-- Definitions of reduced and increased speeds
def V_reduced := V - 4
def V_increased := V + 0.5 * V

-- Conditions for the distance
def distance_current := D = V * T
def distance_increased := D = V_increased * (T - time_diff)
def distance_reduced := D = V_reduced * (T + time_diff)

theorem pascal_remaining_miles (h1 : distance_current)
  (h2 : distance_increased)
  (h3 : distance_reduced) : D = 256 := 
sorry

end PascalCyclingTrip

end pascal_remaining_miles_l136_136420


namespace parabola_vertex_coordinates_l136_136858

theorem parabola_vertex_coordinates :
  ∃ (h k : ℝ), (∀ (x : ℝ), (y = (x - h)^2 + k) = (y = (x-1)^2 + 2)) ∧ h = 1 ∧ k = 2 :=
by
  sorry

end parabola_vertex_coordinates_l136_136858


namespace find_positive_m_l136_136939

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l136_136939


namespace opposite_of_neg2023_l136_136461

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136461


namespace anns_age_l136_136907

theorem anns_age (a b : ℕ)
  (h1 : a + b = 72)
  (h2 : ∃ y, y = a - b)
  (h3 : b = a / 3 + 2 * (a - b)) : a = 36 :=
by
  sorry

end anns_age_l136_136907


namespace johns_drive_distance_l136_136986

/-- John's driving problem -/
theorem johns_drive_distance
  (d t : ℝ)
  (h1 : d = 25 * (t + 1.5))
  (h2 : d = 25 + 45 * (t - 1.25)) :
  d = 123.4375 := 
sorry

end johns_drive_distance_l136_136986


namespace number_of_possible_orders_l136_136764

-- Define the total number of bowlers participating in the playoff
def num_bowlers : ℕ := 6

-- Define the number of games
def num_games : ℕ := 5

-- Define the number of possible outcomes per game
def outcomes_per_game : ℕ := 2

-- Prove the total number of possible orders for bowlers to receive prizes
theorem number_of_possible_orders : (outcomes_per_game ^ num_games) = 32 :=
by sorry

end number_of_possible_orders_l136_136764


namespace hagrid_divisible_by_three_l136_136021

def distinct_digits (n : ℕ) : Prop :=
  n < 10

theorem hagrid_divisible_by_three (H A G R I D : ℕ) (H_dist A_dist G_dist R_dist I_dist D_dist : distinct_digits H ∧ distinct_digits A ∧ distinct_digits G ∧ distinct_digits R ∧ distinct_digits I ∧ distinct_digits D)
  (distinct_letters: H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ R ≠ I ∧ R ≠ D ∧ I ≠ D) :
  3 ∣ (H * 100000 + A * 10000 + G * 1000 + R * 100 + I * 10 + D) * H * A * G * R * I * D :=
sorry

end hagrid_divisible_by_three_l136_136021


namespace find_d_l136_136765

theorem find_d (a b c d x : ℝ)
  (h1 : ∀ x, 2 ≤ a * (Real.cos (b * x + c)) + d ∧ a * (Real.cos (b * x + c)) + d ≤ 4)
  (h2 : Real.cos (b * 0 + c) = 1) :
  d = 3 :=
sorry

end find_d_l136_136765


namespace two_digit_plus_one_multiple_of_3_4_5_6_7_l136_136302

theorem two_digit_plus_one_multiple_of_3_4_5_6_7 (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) :
  (∃ m : ℕ, (m = n - 1 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 ∧ m % 7 = 0)) → False :=
sorry

end two_digit_plus_one_multiple_of_3_4_5_6_7_l136_136302


namespace opposite_of_negative_2023_l136_136710

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136710


namespace expression_value_zero_l136_136228

variables (a b c A B C : ℝ)

theorem expression_value_zero
  (h1 : a + b + c = 0)
  (h2 : A + B + C = 0)
  (h3 : a / A + b / B + c / C = 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 :=
by
  sorry

end expression_value_zero_l136_136228


namespace find_positive_m_has_exactly_single_solution_l136_136945

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l136_136945


namespace opposite_of_neg_2023_l136_136516

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136516


namespace opposite_of_neg_2023_l136_136638

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136638


namespace zeros_of_f_l136_136957

noncomputable def f (a : ℝ) (x : ℝ) :=
if x ≤ 1 then a + 2^x else (1/2) * x + a

theorem zeros_of_f (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ a ∈ Set.Ico (-2) (-1/2) :=
sorry

end zeros_of_f_l136_136957


namespace positive_value_m_l136_136940

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l136_136940


namespace opposite_of_neg_2023_l136_136539

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136539


namespace problem1_problem2_l136_136911

-- Problem 1
theorem problem1 (a : ℝ) : 3 * a ^ 2 - 2 * a + 1 + (3 * a - a ^ 2 + 2) = 2 * a ^ 2 + a + 3 :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : x - 2 * (x - 3 / 2 * y) + 3 * (x - x * y) = 2 * x + 3 * y - 3 * x * y :=
by
  sorry

end problem1_problem2_l136_136911


namespace silver_status_families_l136_136141

theorem silver_status_families 
  (goal : ℕ) 
  (remaining : ℕ) 
  (bronze_families : ℕ) 
  (bronze_donation : ℕ) 
  (gold_families : ℕ) 
  (gold_donation : ℕ) 
  (silver_donation : ℕ) 
  (total_raised_so_far : goal - remaining = 700)
  (amount_raised_by_bronze : bronze_families * bronze_donation = 250)
  (amount_raised_by_gold : gold_families * gold_donation = 100)
  (amount_raised_by_silver : 700 - 250 - 100 = 350) :
  ∃ (s : ℕ), s * silver_donation = 350 ∧ s = 7 :=
by
  sorry

end silver_status_families_l136_136141


namespace car_travel_time_l136_136999

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end car_travel_time_l136_136999


namespace find_length_of_chord_AB_l136_136900

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the coordinates of points A and B
variables (x1 x2 y1 y2 : ℝ)

-- Define the conditions
def conditions : Prop := 
  parabola x1 y1 ∧ parabola x2 y2 ∧ (x1 + x2 = 4 / 3)

-- Define the length of chord AB
def length_of_chord_AB : ℝ := 
  (x1 + 1) + (x2 + 1)

-- Prove the length of chord AB
theorem find_length_of_chord_AB (x1 x2 y1 y2 : ℝ) (h : conditions x1 x2 y1 y2) :
  length_of_chord_AB x1 x2 = 10 / 3 :=
by
  sorry -- Proof is not required

end find_length_of_chord_AB_l136_136900


namespace flagpole_height_l136_136897

theorem flagpole_height (h : ℕ) (shadow_flagpole : ℕ) (height_building : ℕ) (shadow_building : ℕ) (similar_conditions : Prop) 
  (H1 : shadow_flagpole = 45) 
  (H2 : height_building = 24) 
  (H3 : shadow_building = 60) 
  (H4 : similar_conditions) 
  (H5 : h / 45 = 24 / 60) : h = 18 := 
by 
sorry

end flagpole_height_l136_136897


namespace cubic_roots_result_l136_136009

theorem cubic_roots_result (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 64 + b * 16 + c * 4 + d = 0) (h₃ : a * (-27) + b * 9 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end cubic_roots_result_l136_136009


namespace isosceles_triangle_base_l136_136861

noncomputable def base_of_isosceles_triangle
  (height_to_base : ℝ)
  (height_to_side : ℝ)
  (is_isosceles : Bool) : ℝ :=
if is_isosceles then 7.5 else 0

theorem isosceles_triangle_base :
  base_of_isosceles_triangle 5 6 true = 7.5 :=
by
  -- The proof would go here, just placeholder for now
  sorry

end isosceles_triangle_base_l136_136861


namespace Grant_room_count_l136_136193

-- Defining the number of rooms in each person's apartments
def Danielle_rooms : ℕ := 6
def Heidi_rooms : ℕ := 3 * Danielle_rooms
def Jenny_rooms : ℕ := Danielle_rooms + 5

-- Combined total rooms
def Total_rooms : ℕ := Danielle_rooms + Heidi_rooms + Jenny_rooms

-- Division operation to determine Grant's room count
def Grant_rooms (total_rooms : ℕ) : ℕ := total_rooms / 9

-- Statement to be proved
theorem Grant_room_count : Grant_rooms Total_rooms = 3 := by
  sorry

end Grant_room_count_l136_136193


namespace complement_of_M_in_U_l136_136962

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M_in_U : compl M ∩ U = {1, 2, 3} :=
by
  sorry

end complement_of_M_in_U_l136_136962


namespace opposite_of_negative_2023_l136_136608

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136608


namespace neither_5_nor_6_nice_1200_l136_136349

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end neither_5_nor_6_nice_1200_l136_136349


namespace lcm_of_two_numbers_l136_136275

variable (a b hcf lcm : ℕ)

theorem lcm_of_two_numbers (ha : a = 330) (hb : b = 210) (hhcf : Nat.gcd a b = 30) :
  Nat.lcm a b = 2310 := by
  sorry

end lcm_of_two_numbers_l136_136275


namespace part_I_part_II_l136_136070

noncomputable def M : Set ℝ := { x | |x + 1| + |x - 1| ≤ 2 }

theorem part_I : M = Set.Icc (-1 : ℝ) (1 : ℝ) := 
sorry

theorem part_II (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ (1/6)) (hz : |z| ≤ (1/9)) :
  |x + 2 * y - 3 * z| ≤ (5/3) :=
by
  sorry

end part_I_part_II_l136_136070


namespace opposite_neg_2023_l136_136525

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136525


namespace one_def_and_two_def_mutually_exclusive_one_def_and_two_def_not_complementary_at_least_one_def_and_all_def_not_exclusive_nor_complementary_at_least_one_genuine_and_one_def_not_exclusive_nor_complementary_l136_136946

-- Definitions
def batch (genuine defective : ℕ) := genuine > 2 ∧ defective > 2

def select_two_items (genuine defective : ℕ) : list (ℕ × ℕ) := sorry  -- Placeholder, as the actual selection process is not specified

def exactly_one_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def exactly_two_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def at_least_one_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def all_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def at_least_one_genuine (selection : list (ℕ × ℕ)) : Prop := sorry

-- Propositions based on the problem conditions
theorem one_def_and_two_def_mutually_exclusive (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, exactly_one_defective selection → ¬exactly_two_defective selection :=
sorry

theorem one_def_and_two_def_not_complementary (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, exactly_one_defective selection ∨ exactly_two_defective selection → ¬(exactly_one_defective selection ↔ ¬exactly_two_defective selection) :=
sorry

theorem at_least_one_def_and_all_def_not_exclusive_nor_complementary (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, ¬(at_least_one_defective selection ∧ ¬all_defective selection ∨ all_defective selection ∧ ¬at_least_one_defective selection) :=
sorry

theorem at_least_one_genuine_and_one_def_not_exclusive_nor_complementary (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, ¬(at_least_one_genuine selection ∧ ¬at_least_one_defective selection ∨ at_least_one_genuine selection ∧ at_least_one_defective selection) :=
sorry

end one_def_and_two_def_mutually_exclusive_one_def_and_two_def_not_complementary_at_least_one_def_and_all_def_not_exclusive_nor_complementary_at_least_one_genuine_and_one_def_not_exclusive_nor_complementary_l136_136946


namespace jar_total_value_l136_136752

def total_value_in_jar (p n q : ℕ) (total_coins : ℕ) (value : ℝ) : Prop :=
  p + n + q = total_coins ∧
  n = 3 * p ∧
  q = 4 * n ∧
  value = p * 0.01 + n * 0.05 + q * 0.25

theorem jar_total_value (p : ℕ) (h₁ : 16 * p = 240) : 
  ∃ value, total_value_in_jar p (3 * p) (12 * p) 240 value ∧ value = 47.4 :=
by
  sorry

end jar_total_value_l136_136752


namespace age_of_oldest_child_l136_136276

theorem age_of_oldest_child
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 8)
  (h3 : c = 10)
  (h4 : (a + b + c + d) / 4 = 9) :
  d = 12 :=
sorry

end age_of_oldest_child_l136_136276


namespace opposite_of_neg_2023_l136_136632

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136632


namespace opposite_of_neg_2023_l136_136675

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136675


namespace bicycle_new_price_l136_136285

theorem bicycle_new_price (original_price : ℤ) (increase_rate : ℤ) (new_price : ℤ) : 
  original_price = 220 → increase_rate = 15 → new_price = original_price + (original_price * increase_rate / 100) → new_price = 253 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  have h4 : 220 * 15 / 100 = 33 := by norm_num
  rw [h4] at h3
  have h5 : 220 + 33 = 253 := by norm_num
  rw [h5] at h3
  exact h3

end bicycle_new_price_l136_136285


namespace reflection_transformation_l136_136265

structure Point (α : Type) :=
(x : α)
(y : α)

def reflect_x_axis (p : Point ℝ) : Point ℝ :=
  {x := p.x, y := -p.y}

def reflect_x_eq_3 (p : Point ℝ) : Point ℝ :=
  {x := 6 - p.x, y := p.y}

def D : Point ℝ := {x := 4, y := 1}

def D' := reflect_x_axis D

def D'' := reflect_x_eq_3 D'

theorem reflection_transformation :
  D'' = {x := 2, y := -1} :=
by
  -- We skip the proof here
  sorry

end reflection_transformation_l136_136265


namespace opposite_of_neg_2023_l136_136511

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136511


namespace game_show_possible_guesses_l136_136322

theorem game_show_possible_guesses : 
  (∃ A B C : ℕ, 
    A + B + C = 8 ∧ 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ 
    (A = 1 ∨ A = 4) ∧
    (B = 1 ∨ B = 4) ∧
    (C = 1 ∨ C = 4) ) →
  (number_of_possible_guesses : ℕ) = 210 :=
sorry

end game_show_possible_guesses_l136_136322


namespace opposite_of_neg_2023_l136_136653

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136653


namespace greg_and_earl_total_l136_136342

variable (Earl_initial Fred_initial Greg_initial : ℕ)
variable (Earl_to_Fred Fred_to_Greg Greg_to_Earl : ℕ)

def Earl_final := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final := Greg_initial + Fred_to_Greg - Greg_to_Earl

theorem greg_and_earl_total :
  Earl_initial = 90 → Fred_initial = 48 → Greg_initial = 36 →
  Earl_to_Fred = 28 → Fred_to_Greg = 32 → Greg_to_Earl = 40 →
  Greg_final + Earl_final = 130 :=
by
  intros h1 h2 h3 h4 h5 h6
  dsimp [Earl_final, Fred_final, Greg_final]
  rw [h1, h2, h3, h4, h5, h6]
  exact sorry

end greg_and_earl_total_l136_136342


namespace opposite_of_neg_2023_l136_136598

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136598


namespace negation_relation_l136_136398

def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

def not_p (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 1
def not_q (x : ℝ) : Prop := x ≥ -2 ∧ x ≤ 1

theorem negation_relation : (∀ x, not_p x → not_q x) ∧ ¬ (∀ x, not_q x → not_p x) :=
by 
  sorry

end negation_relation_l136_136398


namespace opposite_of_neg_2023_l136_136579

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136579


namespace div_decimals_l136_136913

theorem div_decimals : 0.45 / 0.005 = 90 := sorry

end div_decimals_l136_136913


namespace remove_vertex_preserves_connectivity_l136_136063

-- Definitions
variables {V : Type*} [Fintype V] [DecidableEq V]
variables (G : SimpleGraph V) (h_connected : G.Connected)

-- Theorem Statement
theorem remove_vertex_preserves_connectivity (G : SimpleGraph V) (h_connected : G.Connected) : 
  ∃ (v : V), ∀ w1 w2 ∈ (G.delete_vertex v).verts, (G.delete_vertex v).Connected :=
sorry

end remove_vertex_preserves_connectivity_l136_136063


namespace opposite_of_neg_2023_l136_136659

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136659


namespace opposite_of_neg_2023_l136_136458

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136458


namespace opposite_of_neg_2023_l136_136670

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136670


namespace solution_l136_136116

variable (x y z : ℝ)

noncomputable def problem := 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x^2 + x * y + y^2 = 48 →
  y^2 + y * z + z^2 = 25 →
  z^2 + z * x + x^2 = 73 →
  x * y + y * z + z * x = 40

theorem solution : problem := by
  intros
  sorry

end solution_l136_136116


namespace coefficient_x3y3_expansion_l136_136439

theorem coefficient_x3y3_expansion :
  (2 * (nat.choose 5 3) * (2:ℚ) = 20) :=
by sorry

end coefficient_x3y3_expansion_l136_136439


namespace factorization_of_polynomial_l136_136344

theorem factorization_of_polynomial :
  ∀ x : ℝ, (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) = (x - 1)^4 :=
by
  intro x
  sorry

end factorization_of_polynomial_l136_136344


namespace evaluate_expression_correct_l136_136767

noncomputable def evaluate_expression :=
  abs (-1) - ((-3.14 + Real.pi) ^ 0) + (2 ^ (-1 : ℤ)) + (Real.cos (Real.pi / 6)) ^ 2

theorem evaluate_expression_correct : evaluate_expression = 5 / 4 := by sorry

end evaluate_expression_correct_l136_136767


namespace loop_execution_count_l136_136288

theorem loop_execution_count : 
  ∀ (a b : ℤ), a = 2 → b = 20 → (b - a + 1) = 19 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- Here, we explicitly compute (20 - 2 + 1) = 19
  exact rfl

end loop_execution_count_l136_136288


namespace john_marbles_selection_l136_136383

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l136_136383


namespace opposite_of_neg_2023_l136_136680

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136680


namespace opposite_of_negative_2023_l136_136611

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136611


namespace scientific_notation_320000_l136_136271

theorem scientific_notation_320000 : 320000 = 3.2 * 10^5 :=
  by sorry

end scientific_notation_320000_l136_136271


namespace opposite_of_neg_2023_l136_136499

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136499


namespace opposite_of_neg_2023_l136_136542

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136542


namespace opposite_of_neg_2023_l136_136594

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136594


namespace counties_under_50k_perc_l136_136754

def percentage (s: String) : ℝ := match s with
  | "20k_to_49k" => 45
  | "less_than_20k" => 30
  | _ => 0

theorem counties_under_50k_perc : percentage "20k_to_49k" + percentage "less_than_20k" = 75 := by
  sorry

end counties_under_50k_perc_l136_136754


namespace find_positive_m_has_exactly_single_solution_l136_136944

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l136_136944


namespace simplify_expression_l136_136433

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end simplify_expression_l136_136433


namespace opposite_of_neg_2023_l136_136595

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136595


namespace opposite_of_neg_2023_l136_136452

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136452


namespace matrix_eigenvalue_problem_l136_136780

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l136_136780


namespace arithmetic_sequence_n_15_l136_136828

theorem arithmetic_sequence_n_15 (a : ℕ → ℤ) (n : ℕ)
  (h₁ : a 3 = 5)
  (h₂ : a 2 + a 5 = 12)
  (h₃ : a n = 29) :
  n = 15 :=
sorry

end arithmetic_sequence_n_15_l136_136828


namespace prove_N_value_l136_136244

theorem prove_N_value (x y N : ℝ) 
  (h1 : N = 4 * x + y) 
  (h2 : 3 * x - 4 * y = 5) 
  (h3 : 7 * x - 3 * y = 23) : 
  N = 86 / 3 := by
  sorry

end prove_N_value_l136_136244


namespace opposite_of_neg_2023_l136_136578

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136578


namespace log_x_inequality_l136_136064

noncomputable def log_x_over_x (x : ℝ) := (Real.log x) / x

theorem log_x_inequality {x : ℝ} (h1 : 1 < x) (h2 : x < 2) : 
  (log_x_over_x x) ^ 2 < log_x_over_x x ∧ log_x_over_x x < log_x_over_x (x * x) :=
by
  sorry

end log_x_inequality_l136_136064


namespace opposite_of_neg_2023_l136_136445

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136445


namespace no_real_solution_for_quadratic_eq_l136_136854

theorem no_real_solution_for_quadratic_eq (y : ℝ) :
  (8 * y^2 + 155 * y + 3) / (4 * y + 45) = 4 * y + 3 →  (¬ ∃ y : ℝ, (8 * y^2 + 37 * y + 33/2 = 0)) :=
by
  sorry

end no_real_solution_for_quadratic_eq_l136_136854


namespace common_difference_is_3_l136_136904

noncomputable def whale_plankton_frenzy (x : ℝ) (y : ℝ) : Prop :=
  (9 * x + 36 * y = 450) ∧
  (x + 5 * y = 53)

theorem common_difference_is_3 :
  ∃ (x y : ℝ), whale_plankton_frenzy x y ∧ y = 3 :=
by {
  sorry
}

end common_difference_is_3_l136_136904


namespace ellen_bought_chairs_l136_136773

-- Define the conditions
def cost_per_chair : ℕ := 15
def total_amount_spent : ℕ := 180

-- State the theorem to be proven
theorem ellen_bought_chairs :
  (total_amount_spent / cost_per_chair = 12) := 
sorry

end ellen_bought_chairs_l136_136773


namespace exists_k_for_A_mul_v_eq_k_mul_v_l136_136778

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l136_136778


namespace red_ballpoint_pens_count_l136_136747

theorem red_ballpoint_pens_count (R B : ℕ) (h1: R + B = 240) (h2: B = R - 2) : R = 121 :=
by
  sorry

end red_ballpoint_pens_count_l136_136747


namespace min_value_abs_expr_l136_136790

noncomputable def minExpr (a b : ℝ) : ℝ :=
  |a + b| + |(1 / (a + 1)) - b|

theorem min_value_abs_expr (a b : ℝ) (h₁ : a ≠ -1) : minExpr a b ≥ 1 ∧ (minExpr a b = 1 ↔ a = 0) :=
by
  sorry

end min_value_abs_expr_l136_136790


namespace coordinates_of_point_l136_136821

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l136_136821


namespace boxes_used_l136_136321

-- Definitions of the conditions
def total_oranges := 2650
def oranges_per_box := 10

-- Statement to prove
theorem boxes_used (total_oranges oranges_per_box : ℕ) : (total_oranges = 2650) → (oranges_per_box = 10) → (total_oranges / oranges_per_box = 265) :=
by
  intros h_total h_per_box
  rw [h_total, h_per_box]
  norm_num

end boxes_used_l136_136321


namespace triangle_PQR_area_l136_136829

/-

Define the points P, Q, and R.
Define a function to calculate the area of a triangle given three points.
Then write a theorem to state that the area of triangle PQR is 12.

-/

structure Point where
  x : ℕ
  y : ℕ

def P : Point := ⟨2, 6⟩
def Q : Point := ⟨2, 2⟩
def R : Point := ⟨8, 5⟩

def area (A B C : Point) : ℚ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

theorem triangle_PQR_area : area P Q R = 12 := by
  /- 
    The proof should involve calculating the area using the given points.
   -/
  sorry

end triangle_PQR_area_l136_136829


namespace kira_breakfast_time_l136_136104

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end kira_breakfast_time_l136_136104


namespace fraction_values_l136_136262

theorem fraction_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x^2 + 2 * y^2 = 5 * x * y) :
  ∃ k ∈ ({3, -3} : Set ℝ), (x + y) / (x - y) = k :=
by
  sorry

end fraction_values_l136_136262


namespace opposite_of_neg_2023_l136_136646

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136646


namespace sum_f_eq_29093_l136_136255

open Fintype

def f {n : ℕ} (π : Equiv.Perm (Fin n)) : ℕ :=
  (Finset.range n).find (λ i, Multiset.toFinset (Multiset.map (λ j, π (Fin.ofNat' j))
    (Multiset.range (i+1))) = (Finset.range (i+1))

theorem sum_f_eq_29093 :
  ∑ π in (@Finset.univ (equiv.Perm (Fin 7))), f π = 29093 :=
by
  sorry

end sum_f_eq_29093_l136_136255


namespace quadratic_roots_max_value_l136_136818

theorem quadratic_roots_max_value (t q u₁ u₂ : ℝ)
  (h1 : u₁ + u₂ = t)
  (h2 : u₁ * u₂ = q)
  (h3 : u₁ + u₂ = u₁^2 + u₂^2)
  (h4 : u₁ + u₂ = u₁^4 + u₂^4) :
  (1 / u₁^2009 + 1 / u₂^2009) ≤ 2 :=
sorry

-- Explaination: 
-- This theorem states that given the conditions on the roots u₁ and u₂ of the quadratic equation, 
-- the maximum possible value of the expression (1 / u₁^2009 + 1 / u₂^2009) is 2.

end quadratic_roots_max_value_l136_136818


namespace find_b_l136_136110

-- Define functions p and q
def p (x : ℝ) : ℝ := 3 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

-- Set the target value for p(q(3))
def target_val : ℝ := 9

-- Prove that b = 22/3
theorem find_b (b : ℝ) : p (q 3 b) = target_val → b = 22 / 3 := by
  intro h
  sorry

end find_b_l136_136110


namespace combination_sum_l136_136020

theorem combination_sum :
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) = 34 :=
by
  sorry

end combination_sum_l136_136020


namespace opposite_of_neg_2023_l136_136682

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136682


namespace kite_area_correct_l136_136266

-- Define the coordinates of the vertices
def vertex1 : (ℤ × ℤ) := (3, 0)
def vertex2 : (ℤ × ℤ) := (0, 5)
def vertex3 : (ℤ × ℤ) := (3, 7)
def vertex4 : (ℤ × ℤ) := (6, 5)

-- Define the area of a kite using the Shoelace formula for a quadrilateral
-- with given vertices
def kite_area (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))) / 2

theorem kite_area_correct : kite_area vertex1 vertex2 vertex3 vertex4 = 21 := 
  sorry

end kite_area_correct_l136_136266


namespace opposite_of_neg_2023_l136_136521

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136521


namespace greatest_third_term_of_arithmetic_sequence_l136_136868

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h : 4 * a + 6 * d = 46) : a + 2 * d ≤ 15 :=
sorry

end greatest_third_term_of_arithmetic_sequence_l136_136868


namespace time_to_travel_downstream_l136_136161

-- Definitions based on the conditions.
def speed_boat_still_water := 40 -- Speed of the boat in still water (km/hr)
def speed_stream := 5 -- Speed of the stream (km/hr)
def distance_downstream := 45 -- Distance to be traveled downstream (km)

-- The proof statement
theorem time_to_travel_downstream : (distance_downstream / (speed_boat_still_water + speed_stream)) = 1 :=
by
  -- This would be the place to include the proven steps, but it's omitted as per instructions.
  sorry

end time_to_travel_downstream_l136_136161


namespace area_ratio_equilateral_triangl_l136_136402

theorem area_ratio_equilateral_triangl (x : ℝ) :
  let sA : ℝ := x 
  let sB : ℝ := 3 * sA
  let sC : ℝ := 5 * sA
  let sD : ℝ := 4 * sA
  let area_ABC := (Real.sqrt 3 / 4) * (sA ^ 2)
  let s := (sB + sC + sD) / 2
  let area_A'B'C' := Real.sqrt (s * (s - sB) * (s - sC) * (s - sD))
  (area_A'B'C' / area_ABC) = 8 * Real.sqrt 3 := by
  sorry

end area_ratio_equilateral_triangl_l136_136402


namespace smallest_portion_is_two_l136_136131

theorem smallest_portion_is_two (a1 a2 a3 a4 a5 : ℕ) (d : ℕ) (h1 : a1 = a3 - 2 * d) (h2 : a2 = a3 - d) (h3 : a4 = a3 + d) (h4 : a5 = a3 + 2 * d) (h5 : a1 + a2 + a3 + a4 + a5 = 120) (h6 : a3 + a4 + a5 = 7 * (a1 + a2)) : a1 = 2 :=
by sorry

end smallest_portion_is_two_l136_136131


namespace minimum_unit_cubes_l136_136297

theorem minimum_unit_cubes (n : ℕ) (N : ℕ) : 
  (n ≥ 3) → (N = n^3) → ((n - 2)^3 > (1/2) * n^3) → 
  ∃ n : ℕ, N = n^3 ∧ (n - 2)^3 > (1/2) * n^3 ∧ N = 1000 :=
by
  intros
  sorry

end minimum_unit_cubes_l136_136297


namespace current_price_after_increase_and_decrease_l136_136894

-- Define constants and conditions
def initial_price_RAM : ℝ := 50
def percent_increase : ℝ := 0.30
def percent_decrease : ℝ := 0.20

-- Define intermediate and final values based on conditions
def increased_price_RAM : ℝ := initial_price_RAM * (1 + percent_increase)
def final_price_RAM : ℝ := increased_price_RAM * (1 - percent_decrease)

-- Theorem stating the final result
theorem current_price_after_increase_and_decrease 
  (init_price : ℝ) 
  (inc : ℝ) 
  (dec : ℝ) 
  (final_price : ℝ) :
  init_price = 50 ∧ inc = 0.30 ∧ dec = 0.20 → final_price = 52 := 
  sorry

end current_price_after_increase_and_decrease_l136_136894


namespace pencils_added_by_Nancy_l136_136148

def original_pencils : ℕ := 27
def total_pencils : ℕ := 72

theorem pencils_added_by_Nancy : ∃ x : ℕ, x = total_pencils - original_pencils := by
  sorry

end pencils_added_by_Nancy_l136_136148


namespace jelly_bean_count_l136_136191

variable (b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : b - 5 = 5 * (c - 15))

theorem jelly_bean_count : b = 105 := by
  sorry

end jelly_bean_count_l136_136191


namespace water_on_wednesday_l136_136807

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_l136_136807


namespace Peter_speed_is_correct_l136_136101

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end Peter_speed_is_correct_l136_136101


namespace opposite_of_neg2023_l136_136464

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136464


namespace min_value_frac_inv_l136_136389

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l136_136389


namespace coordinates_with_respect_to_origin_l136_136823

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l136_136823


namespace opposite_of_neg_2023_l136_136569

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136569


namespace gcd_90_252_eq_18_l136_136000

theorem gcd_90_252_eq_18 : Nat.gcd 90 252 = 18 := 
sorry

end gcd_90_252_eq_18_l136_136000


namespace range_of_a_l136_136167

noncomputable def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ (x : ℝ), a * x^2 - a * x - 1 < 0 

theorem range_of_a (a : ℝ) : quadratic_inequality_holds a ↔ -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l136_136167


namespace opposite_of_neg_2023_l136_136575

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136575


namespace coordinates_with_respect_to_origin_l136_136825

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l136_136825


namespace arithmetic_geometric_sequence_min_value_l136_136952

theorem arithmetic_geometric_sequence_min_value (x y a b c d : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (arithmetic_seq : a = (x + y) / 2) (geometric_seq : c * d = x * y) :
  ( (a + b) ^ 2 ) / (c * d) ≥ 4 := 
by
  sorry

end arithmetic_geometric_sequence_min_value_l136_136952


namespace symmetric_line_eq_l136_136347

theorem symmetric_line_eq (x y : ℝ) :
  (y = 2 * x + 3) → (y - 1 = x + 1) → (x - 2 * y = 0) :=
by
  intros h1 h2
  sorry

end symmetric_line_eq_l136_136347


namespace smallest_whole_number_larger_than_perimeter_l136_136733

-- Define the sides of the triangle
def side1 : ℕ := 7
def side2 : ℕ := 23

-- State the conditions using the triangle inequality theorem
def triangle_inequality_satisfied (s : ℕ) : Prop :=
  (side1 + side2 > s) ∧ (side1 + s > side2) ∧ (side2 + s > side1)

-- The proof statement
theorem smallest_whole_number_larger_than_perimeter
  (s : ℕ) (h : triangle_inequality_satisfied s) : 
  ∃ n : ℕ, n = 60 ∧ ∀ p : ℕ, (p > side1 + side2 + s) → (p ≥ n) :=
sorry

end smallest_whole_number_larger_than_perimeter_l136_136733


namespace opposite_of_neg_2023_l136_136519

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136519


namespace opposite_of_negative_2023_l136_136708

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136708


namespace special_discount_percentage_l136_136334

theorem special_discount_percentage (original_price discounted_price : ℝ) (h₀ : original_price = 80) (h₁ : discounted_price = 68) : 
  ((original_price - discounted_price) / original_price) * 100 = 15 :=
by 
  sorry

end special_discount_percentage_l136_136334


namespace seating_arrangements_count_is_134_l136_136291

theorem seating_arrangements_count_is_134 (front_row_seats : ℕ) (back_row_seats : ℕ) (valid_arrangements_with_no_next_to_each_other : ℕ) : 
  front_row_seats = 6 → back_row_seats = 7 → valid_arrangements_with_no_next_to_each_other = 134 :=
by
  intros h1 h2
  sorry

end seating_arrangements_count_is_134_l136_136291


namespace opposite_of_neg_2023_l136_136584

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136584


namespace find_values_of_expression_l136_136864

theorem find_values_of_expression (a b : ℝ) 
  (h : (2 * a) / (a + b) + b / (a - b) = 2) : 
  (∃ x : ℝ, x = (3 * a - b) / (a + 5 * b) ∧ (x = 3 ∨ x = 1)) :=
by 
  sorry

end find_values_of_expression_l136_136864


namespace find_x_when_z_64_l136_136883

-- Defining the conditions
def directly_proportional (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y^3
def inversely_proportional (y z : ℝ) : Prop := ∃ n : ℝ, y = n / z^2

theorem find_x_when_z_64 (x y z : ℝ) (m n : ℝ) (k : ℝ) (h1 : directly_proportional x y) 
    (h2 : inversely_proportional y z) (h3 : z = 64) (h4 : x = 8) (h5 : z = 16) : x = 1/256 := 
  sorry

end find_x_when_z_64_l136_136883


namespace opposite_of_neg_2023_l136_136683

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136683


namespace opposite_of_neg_2023_l136_136550

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136550


namespace sum_of_interior_angles_eq_1440_l136_136902

theorem sum_of_interior_angles_eq_1440 (h : ∀ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ)) : 
    (∃ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ) ∧ (n - 2) * 180 = 1440) :=
by
  sorry

end sum_of_interior_angles_eq_1440_l136_136902


namespace largest_valid_integer_l136_136209

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def satisfies_conditions (n : ℕ) : Prop :=
  (100 ≤ n ∧ n < 1000) ∧
  ∀ d ∈ n.digits 10, d ≠ 0 ∧ n % d = 0 ∧
  sum_of_digits n % 6 = 0

theorem largest_valid_integer : ∃ n : ℕ, satisfies_conditions n ∧ (∀ m : ℕ, satisfies_conditions m → m ≤ n) ∧ n = 936 :=
by
  sorry

end largest_valid_integer_l136_136209


namespace kite_minimum_area_correct_l136_136025

noncomputable def minimumKiteAreaAndSum (r : ℕ) (OP : ℕ) (h₁ : r = 60) (h₂ : OP < r) : ℕ × ℝ :=
  let d₁ := 2 * r
  let d₂ := 2 * Real.sqrt (r^2 - OP^2)
  let area := (d₁ * d₂) / 2
  (120 + 119, area)

theorem kite_minimum_area_correct {r OP : ℕ} (h₁ : r = 60) (h₂ : OP < r) :
  minimumKiteAreaAndSum r OP h₁ h₂ = (239, 120 * Real.sqrt 119) :=
by simp [minimumKiteAreaAndSum, h₁, h₂] ; sorry

end kite_minimum_area_correct_l136_136025


namespace f_properties_l136_136118

noncomputable def f : ℚ × ℚ → ℚ := sorry

theorem f_properties :
  (∀ (x y z : ℚ), f (x*y, z) = f (x, z) * f (y, z)) →
  (∀ (x y z : ℚ), f (z, x*y) = f (z, x) * f (z, y)) →
  (∀ (x : ℚ), f (x, 1 - x) = 1) →
  (∀ (x : ℚ), f (x, x) = 1) ∧
  (∀ (x : ℚ), f (x, -x) = 1) ∧
  (∀ (x y : ℚ), f (x, y) * f (y, x) = 1) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end f_properties_l136_136118


namespace total_tickets_needed_l136_136845

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end total_tickets_needed_l136_136845


namespace sequence_arithmetic_and_find_an_l136_136062

theorem sequence_arithmetic_and_find_an (a : ℕ → ℝ)
  (h1 : a 9 = 1 / 7)
  (h2 : ∀ n, a (n + 1) = a n / (3 * a n + 1)) :
  (∀ n, 1 / a (n + 1) = 3 + 1 / a n) ∧ (∀ n, a n = 1 / (3 * n - 20)) :=
by
  sorry

end sequence_arithmetic_and_find_an_l136_136062


namespace find_first_number_l136_136869

theorem find_first_number (x : ℕ) (h1 : x + 35 = 62) : x = 27 := by
  sorry

end find_first_number_l136_136869


namespace opposite_of_neg2023_l136_136469

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136469


namespace problem_statement_l136_136091

noncomputable def f (x : ℚ) : ℚ := (x^2 - x - 6) / (x^3 - 2 * x^2 - x + 2)

def a : ℕ := 1  -- number of holes
def b : ℕ := 2  -- number of vertical asymptotes
def c : ℕ := 1  -- number of horizontal asymptotes
def d : ℕ := 0  -- number of oblique asymptotes

theorem problem_statement : a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end problem_statement_l136_136091


namespace number_is_correct_l136_136023

theorem number_is_correct : (1 / 8) + 0.675 = 0.800 := 
by
  sorry

end number_is_correct_l136_136023


namespace min_value_frac_inv_l136_136390

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l136_136390


namespace opposite_of_neg_2023_l136_136654

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136654


namespace central_angle_correct_l136_136371

-- Define arc length, radius, and central angle
variables (l r α : ℝ)

-- Given conditions
def arc_length := 3
def radius := 2

-- Theorem to prove
theorem central_angle_correct : (l = arc_length) → (r = radius) → (l = r * α) → α = 3 / 2 :=
by
  intros h1 h2 h3
  sorry

end central_angle_correct_l136_136371


namespace opposite_of_neg_2023_l136_136483

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136483


namespace regular_octagon_area_l136_136848

theorem regular_octagon_area : ∀ (R : ℝ),
  let d_b := 2 * R,
      d_c := 2 * R * Real.sin (Real.pi / 8)
  in R^2 * 4 * Real.sin (Real.pi / 8) = d_b * d_c :=
by
  intros R d_b d_c
  dsimp [d_b, d_c]
  sorry

end regular_octagon_area_l136_136848


namespace range_of_x_max_y_over_x_l136_136794

-- Define the circle and point P(x,y) on the circle
def CircleEquation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9

theorem range_of_x (x y : ℝ) (h : CircleEquation x y) : 1 ≤ x ∧ x ≤ 7 :=
sorry

theorem max_y_over_x (x y : ℝ) (h : CircleEquation x y) : ∀ k : ℝ, (k = y / x) → 0 ≤ k ∧ k ≤ (24 / 7) :=
sorry

end range_of_x_max_y_over_x_l136_136794


namespace find_BD_in_triangle_l136_136830

theorem find_BD_in_triangle (A B C D : Type)
  (distance_AC : Float) (distance_BC : Float)
  (distance_AD : Float) (distance_CD : Float)
  (hAC : distance_AC = 10)
  (hBC : distance_BC = 10)
  (hAD : distance_AD = 12)
  (hCD : distance_CD = 5) :
  ∃ (BD : Float), BD = 6.85435 :=
by 
  sorry

end find_BD_in_triangle_l136_136830


namespace quadratic_roots_l136_136799

theorem quadratic_roots (m : ℝ) (h_eq : ∃ α β : ℝ, (α + β = -4) ∧ (α * β = m) ∧ (|α - β| = 2)) : m = 5 :=
sorry

end quadratic_roots_l136_136799


namespace mul_eight_neg_half_l136_136036

theorem mul_eight_neg_half : 8 * (- (1/2: ℚ)) = -4 := 
by 
  sorry

end mul_eight_neg_half_l136_136036


namespace polynomial_remainder_l136_136880

theorem polynomial_remainder (y : ℝ) : 
  let a := 3 ^ 50 - 2 ^ 50
  let b := 2 ^ 50 - 2 * 3 ^ 50 + 2 ^ 51
  (y ^ 50) % (y ^ 2 - 5 * y + 6) = a * y + b :=
by
  sorry

end polynomial_remainder_l136_136880


namespace opposite_of_neg_2023_l136_136666

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136666


namespace opposite_of_neg_2023_l136_136506

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136506


namespace opposite_of_neg_2023_l136_136449

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136449


namespace sharon_distance_l136_136771

noncomputable def usual_speed (x : ℝ) := x / 180
noncomputable def reduced_speed (x : ℝ) := (x / 180) - 0.5

theorem sharon_distance (x : ℝ) (h : 300 = (x / 2) / usual_speed x + (x / 2) / reduced_speed x) : x = 157.5 :=
by sorry

end sharon_distance_l136_136771


namespace f_odd_f_inequality_solution_l136_136800

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd: 
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = - f x := 
by
  sorry

theorem f_inequality_solution:
  { x : ℝ // -1 < x ∧ x < 1 ∧ f x < -1 } = { x : ℝ // -1 < x ∧ x < -1/3 } := 
by 
  sorry

end f_odd_f_inequality_solution_l136_136800


namespace triangle_abs_diff_l136_136218

theorem triangle_abs_diff (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b) :
  |a + b - c| - |a - b - c| = 2 * a - 2 * c := 
by sorry

end triangle_abs_diff_l136_136218


namespace ways_to_write_1800_as_sum_of_twos_and_threes_l136_136233

theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  ∃ (n : ℕ), n = 301 ∧ ∀ (x y : ℕ), 2 * x + 3 * y = 1800 → ∃ (a : ℕ), (x, y) = (3 * a, 300 - a) :=
sorry

end ways_to_write_1800_as_sum_of_twos_and_threes_l136_136233


namespace solution_to_equation_l136_136287

theorem solution_to_equation (x : ℝ) : x * (x - 2) = 2 * x ↔ (x = 0 ∨ x = 4) := by
  sorry

end solution_to_equation_l136_136287


namespace min_value_frac_inv_l136_136391

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l136_136391


namespace opposite_of_neg_2023_l136_136556

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136556


namespace opposite_of_negative_2023_l136_136707

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136707


namespace find_cos2α_l136_136954

noncomputable def cos2α (tanα : ℚ) : ℚ :=
  (1 - tanα^2) / (1 + tanα^2)

theorem find_cos2α (h : tanα = (3 / 4)) : cos2α tanα = (7 / 25) :=
by
  rw [cos2α, h]
  -- here the simplification steps would be performed
  sorry

end find_cos2α_l136_136954


namespace value_of_2m_plus_3n_l136_136360

theorem value_of_2m_plus_3n (m n : ℝ) (h : (m^2 + 4 * m + 5) * (n^2 - 2 * n + 6) = 5) : 2 * m + 3 * n = -1 :=
by
  sorry

end value_of_2m_plus_3n_l136_136360


namespace opposite_of_negative_2023_l136_136607

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136607


namespace polygon_interior_exterior_equal_l136_136084

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end polygon_interior_exterior_equal_l136_136084


namespace opposite_neg_2023_l136_136530

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136530


namespace find_x_l136_136222

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 :=
sorry

end find_x_l136_136222


namespace find_positive_m_l136_136938

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l136_136938


namespace opposite_of_neg_2023_l136_136663

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136663


namespace solve_for_x_l136_136853

-- Definitions of conditions
def sqrt_81_as_3sq : ℝ := (81 : ℝ)^(1/2)  -- sqrt(81)
def sqrt_81_as_3sq_simplified : ℝ := (3^4 : ℝ)^(1/2)  -- equivalent to (3^2) since 81 = 3^4

-- Theorem and goal statement
theorem solve_for_x :
  sqrt_81_as_3sq = sqrt_81_as_3sq_simplified →
  (3 : ℝ)^(3 * (2/3)) = sqrt_81_as_3sq :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l136_136853


namespace distance_between_vertices_of_hyperbola_l136_136932

theorem distance_between_vertices_of_hyperbola :
  ∀ (x y : ℝ), 16 * x^2 - 32 * x - y^2 + 10 * y + 19 = 0 → 
  2 * Real.sqrt (7 / 4) = Real.sqrt 7 :=
by
  intros x y h
  sorry

end distance_between_vertices_of_hyperbola_l136_136932


namespace solve_for_N_l136_136719

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end solve_for_N_l136_136719


namespace longest_path_is_critical_path_l136_136982

noncomputable def longest_path_in_workflow_diagram : String :=
"Critical Path"

theorem longest_path_is_critical_path :
  (longest_path_in_workflow_diagram = "Critical Path") :=
  by
  sorry

end longest_path_is_critical_path_l136_136982


namespace range_of_a_minus_b_l136_136955

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) : -3 < a - b ∧ a - b < 0 :=
by
  sorry

end range_of_a_minus_b_l136_136955


namespace cost_of_article_l136_136885

theorem cost_of_article (C G : ℝ) (h1 : C + G = 348) (h2 : C + 1.05 * G = 350) : C = 308 :=
by
  sorry

end cost_of_article_l136_136885


namespace opposite_of_neg_2023_l136_136565

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136565


namespace fixed_point_tangent_circle_l136_136857

theorem fixed_point_tangent_circle (x y a b t : ℝ) :
  (x ^ 2 + (y - 2) ^ 2 = 16) ∧ (a * 0 + b * 2 - 12 = 0) ∧ (y = -6) ∧ 
  (t * x - 8 * y = 0) → 
  (0, 0) = (0, 0) :=
by 
  sorry

end fixed_point_tangent_circle_l136_136857


namespace remaining_savings_after_purchase_l136_136407

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l136_136407


namespace line_segment_is_symmetric_l136_136184

def is_axial_symmetric (shape : Type) : Prop := sorry
def is_central_symmetric (shape : Type) : Prop := sorry

def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry
def parallelogram : Type := sorry
def line_segment : Type := sorry

theorem line_segment_is_symmetric : 
  is_axial_symmetric line_segment ∧ is_central_symmetric line_segment := 
by
  sorry

end line_segment_is_symmetric_l136_136184


namespace opposite_of_neg_2023_l136_136512

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136512


namespace opposite_of_neg_2023_l136_136491

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136491


namespace inverse_proportional_l136_136203

-- Define the variables and the condition
variables {R : Type*} [CommRing R] {x y k : R}
-- Assuming x and y are non-zero
variables (hx : x ≠ 0) (hy : y ≠ 0)

-- Define the constant product relationship
def product_constant (x y k : R) : Prop := x * y = k

-- The main statement that needs to be proved
theorem inverse_proportional (h : product_constant x y k) : 
  ∃ k, x * y = k :=
by sorry

end inverse_proportional_l136_136203


namespace eccentricities_proof_l136_136990

variable (e1 e2 m n c : ℝ)
variable (h1 : e1 = 2 * c / (m + n))
variable (h2 : e2 = 2 * c / (m - n))
variable (h3 : m ^ 2 + n ^ 2 = 4 * c ^ 2)

theorem eccentricities_proof :
  (e1 * e2) / (Real.sqrt (e1 ^ 2 + e2 ^ 2)) = (Real.sqrt 2) / 2 :=
by sorry

end eccentricities_proof_l136_136990


namespace paul_account_balance_after_transactions_l136_136426

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l136_136426


namespace opposite_of_neg_2023_l136_136564

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136564


namespace original_number_l136_136734

theorem original_number (N : ℕ) :
  (∃ k m n : ℕ, N - 6 = 5 * k + 3 ∧ N - 6 = 11 * m + 3 ∧ N - 6 = 13 * n + 3) → N = 724 :=
by
  sorry

end original_number_l136_136734


namespace min_value_x_plus_one_over_x_minus_one_max_value_sqrt_x_times_10_minus_x_l136_136156

-- Statement for problem A
theorem min_value_x_plus_one_over_x_minus_one (x : ℝ) (h : 1 < x) : 
  ∃ y, y = x + (1 / (x - 1)) ∧ y ≥ 3 := by
  sorry

-- Statement for problem C
theorem max_value_sqrt_x_times_10_minus_x (x : ℝ) (h1 : 0 < x) (h2 : x < 10) :
  ∃ y, y = sqrt (x * (10 - x)) ∧ y ≤ 5 := by
  sorry

end min_value_x_plus_one_over_x_minus_one_max_value_sqrt_x_times_10_minus_x_l136_136156


namespace number_of_real_zeros_l136_136936

def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

theorem number_of_real_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_real_zeros_l136_136936


namespace opposite_of_neg_2023_l136_136597

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136597


namespace max_subset_no_ap_l136_136268

theorem max_subset_no_ap (n : ℕ) (H : n ≥ 4) :
  ∃ (s : Finset ℝ), (s.card ≥ ⌊Real.sqrt (2 * n / 3)⌋₊ + 1) ∧
  ∀ (a b c : ℝ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → a ≠ c → b ≠ c → (a, b, c) ≠ (a + b - c, b, c) :=
sorry

end max_subset_no_ap_l136_136268


namespace cubic_inequality_solution_l136_136196

theorem cubic_inequality_solution (x : ℝ) :
  (x^3 - 2 * x^2 - x + 2 > 0) ∧ (x < 3) ↔ (x < -1 ∨ (1 < x ∧ x < 3)) := 
sorry

end cubic_inequality_solution_l136_136196


namespace opposite_of_neg_2023_l136_136602

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136602


namespace Jolene_cars_washed_proof_l136_136097

-- Definitions for conditions
def number_of_families : ℕ := 4
def babysitting_rate : ℕ := 30 -- in dollars
def car_wash_rate : ℕ := 12 -- in dollars
def total_money_raised : ℕ := 180 -- in dollars

-- Mathematical representation of the problem:
def babysitting_earnings : ℕ := number_of_families * babysitting_rate
def earnings_from_cars : ℕ := total_money_raised - babysitting_earnings
def number_of_cars_washed : ℕ := earnings_from_cars / car_wash_rate

-- The proof statement
theorem Jolene_cars_washed_proof : number_of_cars_washed = 5 := 
sorry

end Jolene_cars_washed_proof_l136_136097


namespace total_weight_is_correct_l136_136871

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end total_weight_is_correct_l136_136871


namespace square_diagonal_l136_136877

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (hA : A = 338) (hs : s^2 = A) (hd : d^2 = 2 * s^2) : d = 26 :=
by
  -- Proof goes here
  sorry

end square_diagonal_l136_136877


namespace opposite_of_neg_2023_l136_136591

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136591


namespace minimum_value_proof_l136_136392

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l136_136392


namespace equivalent_proof_problem_l136_136211

-- Define the real numbers x, y, z and the operation ⊗
variables {x y z : ℝ}

def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

theorem equivalent_proof_problem : otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x ^ 2 + 2 * x * z - y ^ 2 - 2 * z * y) ^ 2 :=
by sorry

end equivalent_proof_problem_l136_136211


namespace altitude_length_of_right_triangle_l136_136715

theorem altitude_length_of_right_triangle 
    (a b c : ℝ) 
    (h1 : a = 8) 
    (h2 : b = 15) 
    (h3 : c = 17) 
    (h4 : a^2 + b^2 = c^2) 
    : (2 * (1/2 * a * b))/c = 120/17 := 
by {
  sorry
}

end altitude_length_of_right_triangle_l136_136715


namespace opposite_of_neg_2023_l136_136444

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136444


namespace pencils_left_l136_136034

-- Define initial count of pencils
def initial_pencils : ℕ := 20

-- Define pencils misplaced
def misplaced_pencils : ℕ := 7

-- Define pencils broken and thrown away
def broken_pencils : ℕ := 3

-- Define pencils found
def found_pencils : ℕ := 4

-- Define pencils bought
def bought_pencils : ℕ := 2

-- Define the final number of pencils
def final_pencils: ℕ := initial_pencils - misplaced_pencils - broken_pencils + found_pencils + bought_pencils

-- Prove that the final number of pencils is 16
theorem pencils_left : final_pencils = 16 :=
by
  -- The proof steps are omitted here
  sorry

end pencils_left_l136_136034


namespace percentage_paid_l136_136731

theorem percentage_paid (X Y : ℝ) (h_sum : X + Y = 572) (h_Y : Y = 260) : (X / Y) * 100 = 120 :=
by
  -- We'll prove this result by using the conditions and solving for X.
  sorry

end percentage_paid_l136_136731


namespace order_of_arrival_l136_136210

noncomputable def position_order (P S O E R : ℕ) : Prop :=
  S = O - 10 ∧ S = R + 25 ∧ R = E - 5 ∧ E = P - 25

theorem order_of_arrival (P S O E R : ℕ) (h : position_order P S O E R) :
  P > (S + 10) ∧ S > (O - 10) ∧ O > (E + 5) ∧ E > R :=
sorry

end order_of_arrival_l136_136210


namespace boys_and_girls_arrangement_l136_136145

theorem boys_and_girls_arrangement : 
  ∃ (arrangements : ℕ), arrangements = 48 :=
  sorry

end boys_and_girls_arrangement_l136_136145


namespace opposite_of_neg_2023_l136_136629

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136629


namespace opposite_of_negative_2023_l136_136603

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136603


namespace coordinates_with_respect_to_origin_l136_136827

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l136_136827


namespace skittles_transfer_l136_136335

-- Define the initial number of Skittles Bridget and Henry have
def bridget_initial_skittles := 4
def henry_initial_skittles := 4

-- The main statement we want to prove
theorem skittles_transfer :
  bridget_initial_skittles + henry_initial_skittles = 8 :=
by
  sorry

end skittles_transfer_l136_136335


namespace prob_CD_l136_136176

variable (P : String → ℚ)
variable (x : ℚ)

axiom probA : P "A" = 1 / 3
axiom probB : P "B" = 1 / 4
axiom probC : P "C" = 2 * x
axiom probD : P "D" = x
axiom sumProb : P "A" + P "B" + P "C" + P "D" = 1

theorem prob_CD :
  P "D" = 5 / 36 ∧ P "C" = 5 / 18 := by
  sorry

end prob_CD_l136_136176


namespace distinct_arrangements_count_l136_136294

variable {M : Type} [DecidableEq M] [Fintype M] (male female : Finset M)

def males : Finset M := male
def females : Finset M := female
def total_students : Finset M := males ∪ females

axiom num_males_three : males.card = 3
axiom num_females_three : females.card = 3
axiom total_six : total_students.card = 6

def valid_arrangement (arr : List M) : Prop :=
  arr.length = 6 ∧
  arr.head ∈ males ∧
  arr.last ∈ males ∧
  ∀ i ∈ [1, 2, 3, 4],
    (arr.get i) ∈ females → ¬((arr.get (i-1)) = arr.get (i)) ∧ ¬((arr.get (i+1)) = arr.get (i)) 

theorem distinct_arrangements_count : 
  ∃ (arrangements : Finset (List M)), 
    (∀ arr ∈ arrangements, valid_arrangement arr) ∧ 
    arrangements.card = 144 :=
sorry

end distinct_arrangements_count_l136_136294


namespace not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l136_136013

theorem not_right_triangle_sqrt_3_sqrt_4_sqrt_5 :
  ¬ (Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2 :=
by
  -- Start constructing the proof here
  sorry

end not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l136_136013


namespace smallest_positive_a_l136_136215

/-- Define a function f satisfying the given conditions. -/
noncomputable def f : ℝ → ℝ :=
  sorry -- we'll define it later according to the problem

axiom condition1 : ∀ x > 0, f (2 * x) = 2 * f x

axiom condition2 : ∀ x, 1 < x ∧ x < 2 → f x = 2 - x

theorem smallest_positive_a :
  (∃ a > 0, f a = f 2020) ∧ ∀ b > 0, (f b = f 2020 → b ≥ 36) :=
  sorry

end smallest_positive_a_l136_136215


namespace alex_final_bill_l136_136746

def original_bill : ℝ := 500
def first_late_charge (bill : ℝ) : ℝ := bill * 1.02
def final_bill (bill : ℝ) : ℝ := first_late_charge bill * 1.03

theorem alex_final_bill : final_bill original_bill = 525.30 :=
by sorry

end alex_final_bill_l136_136746


namespace opposite_of_neg_2023_l136_136588

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136588


namespace calculate_expr_l136_136192

theorem calculate_expr (h1 : Real.sin (30 * Real.pi / 180) = 1 / 2)
    (h2 : Real.cos (30 * Real.pi / 180) = Real.sqrt (3) / 2) :
    3 * Real.tan (30 * Real.pi / 180) + 6 * Real.sin (30 * Real.pi / 180) = 3 + Real.sqrt 3 :=
  sorry

end calculate_expr_l136_136192


namespace opposite_of_neg_2023_l136_136576

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136576


namespace bread_remaining_is_26_85_l136_136076

noncomputable def bread_leftover (jimin_cm : ℕ) (taehyung_m original_length : ℝ) : ℝ :=
  original_length - (jimin_cm / 100 + taehyung_m)

theorem bread_remaining_is_26_85 :
  bread_leftover 150 1.65 30 = 26.85 :=
by
  sorry

end bread_remaining_is_26_85_l136_136076


namespace correct_exponent_calculation_l136_136304

theorem correct_exponent_calculation : 
(∀ (a b : ℝ), (a + b)^2 ≠ a^2 + b^2) ∧
(∀ (a : ℝ), a^9 / a^3 ≠ a^3) ∧
(∀ (a b : ℝ), (ab)^3 = a^3 * b^3) ∧
(∀ (a : ℝ), (a^5)^2 ≠ a^7) :=
by 
  sorry

end correct_exponent_calculation_l136_136304


namespace find_costs_compare_options_l136_136749

-- Definitions and theorems
def cost1 (x y : ℕ) : Prop := 2 * x + 4 * y = 350
def cost2 (x y : ℕ) : Prop := 6 * x + 3 * y = 420

def optionACost (m : ℕ) : ℕ := 70 * m + 35 * (80 - 2 * m)
def optionBCost (m : ℕ) : ℕ := (8 * (35 * m + 2800)) / 10

theorem find_costs (x y : ℕ) : 
  cost1 x y ∧ cost2 x y → (x = 35 ∧ y = 70) :=
by sorry

theorem compare_options (m : ℕ) (h : m < 41) : 
  if m < 20 then optionBCost m < optionACost m else 
  if m = 20 then optionBCost m = optionACost m 
  else optionBCost m > optionACost m :=
by sorry

end find_costs_compare_options_l136_136749


namespace basketball_probability_l136_136364

-- Define the probabilities of A and B making a shot
def prob_A : ℝ := 0.4
def prob_B : ℝ := 0.6

-- Define the probability that both miss their shots in one round
def prob_miss_one_round : ℝ := (1 - prob_A) * (1 - prob_B)

-- Define the probability that A takes k shots to make a basket
noncomputable def P_xi (k : ℕ) : ℝ := (prob_miss_one_round)^(k-1) * prob_A

-- State the theorem
theorem basketball_probability (k : ℕ) : 
  P_xi k = 0.24^(k-1) * 0.4 :=
by
  unfold P_xi
  unfold prob_miss_one_round
  sorry

end basketball_probability_l136_136364


namespace eigenvalues_of_2x2_matrix_l136_136776

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l136_136776


namespace opposite_of_neg_2023_l136_136648

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136648


namespace count_three_element_subsets_with_property_P_l136_136399

namespace ThreeElementSubsets

-- Define the set S
def S : Finset ℕ := Finset.range 101 \ Finset.singleton 0

-- Define a subset has property P if a + b = 3c for a subset {a, b, c}
def has_property_P (A : Finset ℕ) : Prop :=
  ∃ a b c, A = {a, b, c} ∧ a + b = 3 * c

-- Define the number of three-element subsets with property P
def count_property_P : ℕ :=
  (S.subsetsOfCard 3).count has_property_P

-- Prove that the count of such subsets is 1600
theorem count_three_element_subsets_with_property_P :
  count_property_P = 1600 :=
sorry

end ThreeElementSubsets

end count_three_element_subsets_with_property_P_l136_136399


namespace nellie_final_legos_l136_136841

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end nellie_final_legos_l136_136841


namespace initial_men_l136_136177

variable (x : ℕ)

-- Conditions
def condition1 (x : ℕ) : Prop :=
  -- The hostel had provisions for x men for 28 days.
  true

def condition2 (x : ℕ) : Prop :=
  -- If 50 men left, the food would last for 35 days for the remaining x - 50 men.
  (x - 50) * 35 = x * 28

-- Theorem to prove
theorem initial_men (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 250 :=
by
  sorry

end initial_men_l136_136177


namespace proof_abc_div_def_l136_136970

def abc_div_def (a b c d e f : ℚ) : Prop := 
  a / b = 1 / 3 ∧ b / c = 2 ∧ c / d = 1 / 2 ∧ d / e = 3 ∧ e / f = 1 / 8 → (a * b * c) / (d * e * f) = 1 / 16

theorem proof_abc_div_def (a b c d e f : ℚ) :
  abc_div_def a b c d e f :=
by 
  sorry

end proof_abc_div_def_l136_136970


namespace opposite_of_neg_2023_l136_136673

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136673


namespace Vihaan_more_nephews_than_Alden_l136_136330

theorem Vihaan_more_nephews_than_Alden :
  ∃ (a v : ℕ), (a = 100) ∧ (a + v = 260) ∧ (v - a = 60) := by
  sorry

end Vihaan_more_nephews_than_Alden_l136_136330


namespace pair_sum_ways_9_10_11_12_13_l136_136972

open Finset

def num_pairs_ways : Nat := 945

theorem pair_sum_ways_9_10_11_12_13 :
  ∃ pairs : Finset (Finset ℕ), 
    (pairs.card = 5) ∧ 
    (∀ pair ∈ pairs, pair.card = 2 ∧ Finset.sum pair ∈ {9, 10, 11, 12, 13}) ∧ 
    num_elements = univ.card 10
    := sorry

end pair_sum_ways_9_10_11_12_13_l136_136972


namespace log_expression_value_l136_136189

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_expression_value :
  log_base 3 32 * log_base 4 9 - log_base 2 (3/4) + log_base 2 6 = 8 := 
by 
  sorry

end log_expression_value_l136_136189


namespace range_of_m_for_hyperbola_l136_136226

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ u v : ℝ, (∀ x y : ℝ, x^2/(m+2) + y^2/(m+1) = 1) → (m > -2) ∧ (m < -1)) := by
  sorry

end range_of_m_for_hyperbola_l136_136226


namespace opposite_neg_2023_l136_136528

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136528


namespace sum_of_numbers_l136_136720

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : a^2 + b^2 + c^2 = 62) 
  (h₂ : ab + bc + ca = 131) : 
  a + b + c = 18 :=
sorry

end sum_of_numbers_l136_136720


namespace triangle_side_ratio_l136_136094

theorem triangle_side_ratio (A B C a b c : ℝ)
  (h1 : A + B + C = 180)
  (h2 : A = 30)
  (h3 : B = 60)
  (h4 : C = 90)
  (h5 : a / Real.sin (Real.pi / 6) = b / Real.sin (Real.pi / 3))
  (h6 : a / Real.sin (Real.pi / 6) = c / Real.sin (Real.pi / 2)) :
  a : b : c = 1 : Real.sqrt 3 : 2 :=
by sorry

end triangle_side_ratio_l136_136094


namespace smallest_total_marbles_l136_136187

-- Definitions based on conditions in a)
def urn_contains_marbles : Type := ℕ → ℕ
def red_marbles (u : urn_contains_marbles) := u 0
def white_marbles (u : urn_contains_marbles) := u 1
def blue_marbles (u : urn_contains_marbles) := u 2
def green_marbles (u : urn_contains_marbles) := u 3
def yellow_marbles (u : urn_contains_marbles) := u 4
def total_marbles (u : urn_contains_marbles) := u 0 + u 1 + u 2 + u 3 + u 4

-- Probabilities of selection events
def prob_event_a (u : urn_contains_marbles) := (red_marbles u).choose 5
def prob_event_b (u : urn_contains_marbles) := (white_marbles u).choose 1 * (red_marbles u).choose 4
def prob_event_c (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (red_marbles u).choose 3
def prob_event_d (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (red_marbles u).choose 2
def prob_event_e (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (yellow_marbles u).choose 1 * (red_marbles u).choose 1

-- Proof that the smallest total number of marbles satisfying the conditions is 33
theorem smallest_total_marbles : ∃ u : urn_contains_marbles, 
    (prob_event_a u = prob_event_b u) ∧ 
    (prob_event_b u = prob_event_c u) ∧ 
    (prob_event_c u = prob_event_d u) ∧ 
    (prob_event_d u = prob_event_e u) ∧ 
    total_marbles u = 33 := sorry

end smallest_total_marbles_l136_136187


namespace opposite_of_neg_2023_l136_136509

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136509


namespace opposite_of_negative_2023_l136_136701

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136701


namespace opposite_of_neg_2023_l136_136456

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136456


namespace opposite_of_neg_2023_l136_136639

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136639


namespace polygon_interior_exterior_eq_l136_136082

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_interior_exterior_eq_l136_136082


namespace probability_C_10000_equal_expected_prize_money_l136_136032

/-- Problem 1: Probability that C gets 10,000 yuan given P1 = P2 = 1/2 --/
theorem probability_C_10000 (P1 P2 : ℝ) (h : P1 = 1/2 ∧ P2 = 1/2) : 
  let A := 10000
  let B := 20000
  let prize := 40000
  P1 * (1 - P2) + (1 - P1) * P2 = 1/2 := by
  sorry

/-- Problem 2: Values of P1 and P2 for equal expected prize money --/
theorem equal_expected_prize_money (P1 P2 : ℝ) (h1 : P1 + P2 = 1) (h2 : 
  let eA := P1 + 2 * P2 
  let eB := P1 + 2 * P2
  let eC := 2 * P1^2 + 2 * P1 * P2
  eA = eB ∧ eB = eC) : P1 = 2 / 3 ∧ P2 = 1 / 3 := by
  sorry

end probability_C_10000_equal_expected_prize_money_l136_136032


namespace tan_theta_plus_pi_over_eight_sub_inv_l136_136367

/-- Given the trigonometric identity, we can prove the tangent calculation -/
theorem tan_theta_plus_pi_over_eight_sub_inv (θ : ℝ)
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) :
  Real.tan (θ + Real.pi / 8) - 1 / Real.tan (θ + Real.pi / 8) = -14 := 
sorry

end tan_theta_plus_pi_over_eight_sub_inv_l136_136367


namespace problem1_problem2_l136_136337

theorem problem1 : 6 + (-8) - (-5) = 3 := sorry

theorem problem2 : 18 / (-3) + (-2) * (-4) = 2 := sorry

end problem1_problem2_l136_136337


namespace general_term_formula_l136_136933

noncomputable def a (n : ℕ) : ℝ := 1 / (Real.sqrt n)

theorem general_term_formula :
  ∀ (n : ℕ), a n = 1 / Real.sqrt n :=
by
  intros
  rfl

end general_term_formula_l136_136933


namespace net_rate_of_pay_is_25_l136_136751

-- Define the conditions 
variables (hours : ℕ) (speed : ℕ) (efficiency : ℕ)
variables (pay_per_mile : ℝ) (cost_per_gallon : ℝ)
variables (total_distance : ℕ) (gas_used : ℕ)
variables (total_earnings : ℝ) (total_cost : ℝ) (net_earnings : ℝ) (net_rate_of_pay : ℝ)

-- Assume the given conditions are as stated in the problem
axiom hrs : hours = 3
axiom spd : speed = 50
axiom eff : efficiency = 25
axiom ppm : pay_per_mile = 0.60
axiom cpg : cost_per_gallon = 2.50

-- Assuming intermediate computations
axiom distance_calc : total_distance = speed * hours
axiom gas_calc : gas_used = total_distance / efficiency
axiom earnings_calc : total_earnings = pay_per_mile * total_distance
axiom cost_calc : total_cost = cost_per_gallon * gas_used
axiom net_earnings_calc : net_earnings = total_earnings - total_cost
axiom pay_rate_calc : net_rate_of_pay = net_earnings / hours

-- Proving the final result
theorem net_rate_of_pay_is_25 :
  net_rate_of_pay = 25 :=
by
  -- Proof goes here
  sorry

end net_rate_of_pay_is_25_l136_136751


namespace problem_statement_l136_136213

theorem problem_statement (x y : ℝ) (h : x - 2 * y = -2) : 3 + 2 * x - 4 * y = -1 :=
  sorry

end problem_statement_l136_136213


namespace advertising_department_size_l136_136317

-- Define the conditions provided in the problem.
def total_employees : Nat := 1000
def sample_size : Nat := 80
def advertising_sample_size : Nat := 4

-- Define the main theorem to prove the given problem.
theorem advertising_department_size :
  ∃ n : Nat, (advertising_sample_size : ℚ) / n = (sample_size : ℚ) / total_employees ∧ n = 50 :=
by
  sorry

end advertising_department_size_l136_136317


namespace solution_set_of_inequality_l136_136716

theorem solution_set_of_inequality (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := 
by
  sorry

end solution_set_of_inequality_l136_136716


namespace john_subtracts_79_l136_136728

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l136_136728


namespace train_meetings_between_stations_l136_136279

theorem train_meetings_between_stations
  (travel_time : ℕ := 3 * 60 + 30) -- Travel time in minutes
  (first_departure : ℕ := 6 * 60) -- First departure time in minutes from 0 (midnight)
  (departure_interval : ℕ := 60) -- Departure interval in minutes
  (A_departure_time : ℕ := 9 * 60) -- Departure time from Station A at 9:00 AM in minutes
  :
  ∃ n : ℕ, n = 7 :=
by
  sorry

end train_meetings_between_stations_l136_136279


namespace choir_population_l136_136182

theorem choir_population 
  (female_students : ℕ) 
  (male_students : ℕ) 
  (choir_multiple : ℕ) 
  (total_students_orchestra : ℕ := female_students + male_students)
  (total_students_choir : ℕ := choir_multiple * total_students_orchestra)
  (h_females : female_students = 18) 
  (h_males : male_students = 25) 
  (h_multiple : choir_multiple = 3) : 
  total_students_choir = 129 := 
by
  -- The proof of the theorem will be done here.
  sorry

end choir_population_l136_136182


namespace country_x_income_l136_136165

variable (income : ℝ)
variable (tax_paid : ℝ)
variable (income_first_40000_tax : ℝ := 40000 * 0.1)
variable (income_above_40000_tax_rate : ℝ := 0.2)
variable (total_tax_paid : ℝ := 8000)
variable (income_above_40000 : ℝ := (total_tax_paid - income_first_40000_tax) / income_above_40000_tax_rate)

theorem country_x_income : 
  income = 40000 + income_above_40000 → 
  total_tax_paid = tax_paid → 
  tax_paid = income_first_40000_tax + (income_above_40000 * income_above_40000_tax_rate) →
  income = 60000 :=
by sorry

end country_x_income_l136_136165


namespace domain_of_f_l136_136195

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - 2 * x) + Real.log (1 + 2 * x)

theorem domain_of_f : {x : ℝ | 1 - 2 * x > 0 ∧ 1 + 2 * x > 0} = {x : ℝ | -1 / 2 < x ∧ x < 1 / 2} :=
by
    sorry

end domain_of_f_l136_136195


namespace measure_angle_P_l136_136379

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end measure_angle_P_l136_136379


namespace max_area_of_triangle_l136_136934

theorem max_area_of_triangle (AB AC BC : ℝ) : 
  AB = 4 → AC = 2 * BC → 
  ∃ (S : ℝ), (∀ (S' : ℝ), S' ≤ S) ∧ S = 16 / 3 :=
by
  sorry

end max_area_of_triangle_l136_136934


namespace opposite_of_neg_2023_l136_136540

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136540


namespace H2O_production_l136_136052

theorem H2O_production (n : Nat) (m : Nat)
  (h1 : n = 3)
  (h2 : m = 3) :
  n = m → n = 3 := by
  sorry

end H2O_production_l136_136052


namespace final_selling_price_l136_136758

-- Define the conditions in Lean
def cost_price_A : ℝ := 150
def profit_A_rate : ℝ := 0.20
def profit_B_rate : ℝ := 0.25

-- Define the function to calculate selling price based on cost price and profit rate
def selling_price (cost_price : ℝ) (profit_rate : ℝ) : ℝ :=
  cost_price + (profit_rate * cost_price)

-- The theorem to be proved
theorem final_selling_price :
  selling_price (selling_price cost_price_A profit_A_rate) profit_B_rate = 225 :=
by
  -- The proof is omitted
  sorry

end final_selling_price_l136_136758


namespace rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l136_136783

-- Define the digit constraints and the RD sum function
def is_digit (n : ℕ) : Prop := n < 10
def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def rd_sum (A B C D : ℕ) : ℕ :=
  let abcd := 1000 * A + 100 * B + 10 * C + D
  let dcba := 1000 * D + 100 * C + 10 * B + A
  abcd + dcba

-- Problem (a)
theorem rd_sum_4281 : rd_sum 4 2 8 1 = 6105 := sorry

-- Problem (b)
theorem rd_sum_formula (A B C D : ℕ) (hA : is_nonzero_digit A) (hD : is_nonzero_digit D) :
  ∃ m n, m = 1001 ∧ n = 110 ∧ rd_sum A B C D = m * (A + D) + n * (B + C) :=
  sorry

-- Problem (c)
theorem rd_sum_count_3883 :
  ∃ n, n = 18 ∧ ∃ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D ∧ rd_sum A B C D = 3883 :=
  sorry

-- Problem (d)
theorem count_self_equal_rd_sum : 
  ∃ n, n = 143 ∧ ∀ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D → (1001 * (A + D) + 110 * (B + C) ≤ 9999 → (1000 * A + 100 * B + 10 * C + D = rd_sum A B C D → 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ D ∧ D ≤ 9)) :=
  sorry

end rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l136_136783


namespace opposite_of_neg2023_l136_136462

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136462


namespace modulo_problem_l136_136041

theorem modulo_problem :
  (47 ^ 2051 - 25 ^ 2051) % 5 = 3 := by
  sorry

end modulo_problem_l136_136041


namespace opposite_of_neg_2023_l136_136563

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136563


namespace polynomial_multiplication_l136_136839

noncomputable def multiply_polynomials (a b : ℤ) :=
  let p1 := 3 * a ^ 4 - 7 * b ^ 3
  let p2 := 9 * a ^ 8 + 21 * a ^ 4 * b ^ 3 + 49 * b ^ 6 + 6 * a ^ 2 * b ^ 2
  let result := 27 * a ^ 12 + 18 * a ^ 6 * b ^ 2 - 42 * a ^ 2 * b ^ 5 - 343 * b ^ 9
  p1 * p2 = result

-- The main statement to prove
theorem polynomial_multiplication (a b : ℤ) : multiply_polynomials a b :=
by
  sorry

end polynomial_multiplication_l136_136839


namespace opposite_of_negative_2023_l136_136700

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136700


namespace min_expression_l136_136259

theorem min_expression 
  (a b c : ℝ)
  (ha : -1 < a ∧ a < 1)
  (hb : -1 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 1) :
  ∃ m, m = 2 ∧ ∀ x y z, (-1 < x ∧ x < 1) → (-1 < y ∧ y < 1) → (-1 < z ∧ z < 1) → 
  ( 1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)) ) ≥ m :=
sorry

end min_expression_l136_136259


namespace sum_mod_20_l136_136054

theorem sum_mod_20 : 
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 20 = 15 :=
by 
  -- The proof goes here
  sorry

end sum_mod_20_l136_136054


namespace visited_neither_l136_136815

theorem visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) 
  (h1 : total = 100) 
  (h2 : iceland = 55) 
  (h3 : norway = 43) 
  (h4 : both = 61) : 
  (total - (iceland + norway - both)) = 63 := 
by 
  sorry

end visited_neither_l136_136815


namespace problem_I_problem_II_1_problem_II_2_l136_136976

section
variables (boys_A girls_A boys_B girls_B : ℕ)
variables (total_students : ℕ)

-- Define the conditions
def conditions : Prop :=
  boys_A = 2 ∧ girls_A = 1 ∧ boys_B = 3 ∧ girls_B = 2 ∧ total_students = boys_A + girls_A + boys_B + girls_B

-- Problem (I)
theorem problem_I (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ arrangements, arrangements = 14400 := sorry

-- Problem (II.1)
theorem problem_II_1 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 13 / 14 := sorry

-- Problem (II.2)
theorem problem_II_2 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 6 / 35 := sorry
end

end problem_I_problem_II_1_problem_II_2_l136_136976


namespace total_votes_l136_136738

-- Define the given conditions
def candidate_votes (V : ℝ) : ℝ := 0.35 * V
def rival_votes (V : ℝ) : ℝ := 0.35 * V + 1800

-- Prove the total number of votes cast
theorem total_votes (V : ℝ) (h : candidate_votes V + rival_votes V = V) : V = 6000 :=
by
  sorry

end total_votes_l136_136738


namespace opposite_of_negative_2023_l136_136616

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136616


namespace exponentiation_problem_l136_136301

theorem exponentiation_problem : 10^6 * (10^2)^3 / 10^4 = 10^8 := 
by 
  sorry

end exponentiation_problem_l136_136301


namespace number_of_green_balls_l136_136168

theorem number_of_green_balls (b g : ℕ) (h1 : b = 9) (h2 : (b : ℚ) / (b + g) = 3 / 10) : g = 21 :=
sorry

end number_of_green_balls_l136_136168


namespace range_of_a_l136_136958

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp x + x^2 + (3 * a + 2) * x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 0, ∀ y ∈ Set.Ioo (-1 : ℝ) 0, f a x ≤ f a y) →
  a ∈ Set.Ioo (-1 : ℝ) (-1 / (3 * Real.exp 1)) :=
sorry

end range_of_a_l136_136958


namespace distinct_ball_placement_l136_136072

def num_distributions (balls boxes : ℕ) : ℕ :=
  if boxes = 3 then 243 - 32 + 16 else 0

theorem distinct_ball_placement : num_distributions 5 3 = 227 :=
by
  sorry

end distinct_ball_placement_l136_136072


namespace opposite_of_neg_2023_l136_136658

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136658


namespace remainder_sum_first_150_mod_5000_l136_136299

theorem remainder_sum_first_150_mod_5000 : 
  (∑ i in Finset.range 151, i) % 5000 = 1325 := by
  sorry

end remainder_sum_first_150_mod_5000_l136_136299


namespace opposite_of_neg_2023_l136_136625

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136625


namespace banana_price_reduction_l136_136171

theorem banana_price_reduction (P_r : ℝ) (P : ℝ) (n : ℝ) (m : ℝ) (h1 : P_r = 3) (h2 : n = 40) (h3 : m = 64) 
  (h4 : 160 = (n / P_r) * 12) 
  (h5 : 96 = 160 - m) 
  (h6 : (40 / 8) = P) :
  (P - P_r) / P * 100 = 40 :=
by
  sorry

end banana_price_reduction_l136_136171


namespace quadratic_other_root_is_three_l136_136225

-- Steps for creating the Lean statement following the identified conditions
variable (b : ℝ)

theorem quadratic_other_root_is_three (h1 : ∀ x : ℝ, x^2 - 2 * x - b = 0 → (x = -1 ∨ x = 3)) : 
  ∀ x : ℝ, x^2 - 2 * x - b = 0 → x = -1 ∨ x = 3 :=
by
  -- The proof is omitted
  exact h1

end quadratic_other_root_is_three_l136_136225


namespace tangent_lines_from_point_to_circle_l136_136786

theorem tangent_lines_from_point_to_circle : 
  ∀ (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ), 
  P = (2, 3) → C = (1, 1) → r = 1 → 
  (∃ k : ℝ, ((3 : ℝ) * P.1 - (4 : ℝ) * P.2 + 6 = 0) ∨ (P.1 = 2)) :=
by
  intros P C r hP hC hr
  sorry

end tangent_lines_from_point_to_circle_l136_136786


namespace opposite_of_neg_2023_l136_136566

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136566


namespace opposite_of_neg2023_l136_136459

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136459


namespace inequality_holds_for_all_l136_136971

theorem inequality_holds_for_all (m : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 8 * x + 20) / (m * x^2 - m * x - 1) < 0) : -4 < m ∧ m ≤ 0 := 
sorry

end inequality_holds_for_all_l136_136971


namespace part1_part2_l136_136993

variable {a b : ℝ}

noncomputable def in_interval (x: ℝ) : Prop :=
  -1/2 < x ∧ x < 1/2

theorem part1 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1/3 * a + 1/6 * b) < 1/4 := 
by sorry

theorem part2 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1 - 4 * a * b) > 2 * abs (a - b) := 
by sorry

end part1_part2_l136_136993


namespace binary_111_is_7_l136_136878

def binary_to_decimal (b0 b1 b2 : ℕ) : ℕ :=
  b0 * (2^0) + b1 * (2^1) + b2 * (2^2)

theorem binary_111_is_7 : binary_to_decimal 1 1 1 = 7 :=
by
  -- We will provide the proof here.
  sorry

end binary_111_is_7_l136_136878


namespace opposite_of_neg_2023_l136_136637

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136637


namespace ratio_books_donated_l136_136119

theorem ratio_books_donated (initial_books: ℕ) (books_given_nephew: ℕ) (books_after_nephew: ℕ) 
  (books_final: ℕ) (books_purchased: ℕ) (books_donated_library: ℕ) (ratio: ℕ):
    initial_books = 40 → 
    books_given_nephew = initial_books / 4 → 
    books_after_nephew = initial_books - books_given_nephew →
    books_final = 23 →
    books_purchased = 3 →
    books_donated_library = books_after_nephew - (books_final - books_purchased) →
    ratio = books_donated_library / books_after_nephew →
    ratio = 1 / 3 := sorry

end ratio_books_donated_l136_136119


namespace min_value_of_box_l136_136073

theorem min_value_of_box (a b : ℤ) (h_ab : a * b = 30) : 
  ∃ (m : ℤ), m = 61 ∧ (∀ (c : ℤ), a * b = 30 → a^2 + b^2 = c → c ≥ m) := 
sorry

end min_value_of_box_l136_136073


namespace average_visitors_per_day_l136_136162

theorem average_visitors_per_day (avg_sunday_visitors : ℕ) (avg_otherday_visitors : ℕ) (days_in_month : ℕ)
  (starts_with_sunday : Bool) (num_sundays : ℕ) (num_otherdays : ℕ)
  (h1 : avg_sunday_visitors = 510)
  (h2 : avg_otherday_visitors = 240)
  (h3 : days_in_month = 30)
  (h4 : starts_with_sunday = true)
  (h5 : num_sundays = 5)
  (h6 : num_otherdays = 25) :
  (num_sundays * avg_sunday_visitors + num_otherdays * avg_otherday_visitors) / days_in_month = 285 :=
by 
  sorry

end average_visitors_per_day_l136_136162


namespace coordinates_of_point_l136_136820

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l136_136820


namespace least_possible_value_of_p_and_q_l136_136811

theorem least_possible_value_of_p_and_q 
  (p q : ℕ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 15 * (p + 1) = 29 * (q + 1)) : 
  p + q = 45 := 
sorry -- proof to be filled in

end least_possible_value_of_p_and_q_l136_136811


namespace kira_breakfast_time_l136_136102

theorem kira_breakfast_time (n_sausages : ℕ) (n_eggs : ℕ) (t_fry_per_sausage : ℕ) (t_scramble_per_egg : ℕ) (total_time : ℕ) :
  n_sausages = 3 → n_eggs = 6 → t_fry_per_sausage = 5 → t_scramble_per_egg = 4 → total_time = (n_sausages * t_fry_per_sausage + n_eggs * t_scramble_per_egg) →
  total_time = 39 :=
by
  intros h_sausages h_eggs h_fry h_scramble h_total
  rw [h_sausages, h_eggs, h_fry, h_scramble] at h_total
  exact h_total

end kira_breakfast_time_l136_136102


namespace pascal_remaining_distance_l136_136417

variable (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ)

def remaining_distance_proof : Prop :=
  v = 8 ∧
  v_red = 4 ∧
  v_inc = 12 ∧
  d = v * t ∧ 
  d = v_red * (t + 16) ∧ 
  d = v_inc * (t - 16) ∧ 
  d = 256

theorem pascal_remaining_distance (v : ℕ) (v_red : ℕ) (v_inc : ℕ) (t : ℕ) (d : ℕ) 
  (h1 : v = 8)
  (h2 : v_red = 4)
  (h3 : v_inc = 12)
  (h4 : d = v * t)
  (h5 : d = v_red * (t + 16))
  (h6 : d = v_inc * (t - 16)) : 
  d = 256 :=
by {
  -- Initial setup
  have h7 : d = 4 * (t + 16) := h5,
  have h8 : d = 12 * (t - 16) := h6,
  have h9 : 4 * (t + 16) = 12 * (t - 16), from sorry,
  -- Solve the equation 
  have h10 : 4 * t + 64 = 12 * t - 192, from sorry,
  have h11 : 4 * t + 256 = 12 * t, from sorry,
  have h12 : 8 * t = 256, from sorry,
  have ht : t = 32, from sorry,
  -- Compute d
  have hd : d = 8 * t, from h4,
  rw ht at hd,
  exact hd.symm,
}

end pascal_remaining_distance_l136_136417


namespace remaining_distance_l136_136422

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l136_136422


namespace opposite_of_neg_2023_l136_136571

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136571


namespace train_cross_time_l136_136903

noncomputable def train_length : ℝ := 130
noncomputable def train_speed_kph : ℝ := 45
noncomputable def total_length : ℝ := 375

noncomputable def speed_mps := train_speed_kph * 1000 / 3600
noncomputable def distance := train_length + total_length

theorem train_cross_time : (distance / speed_mps) = 30 := by
  sorry

end train_cross_time_l136_136903


namespace intersection_complement_A_B_l136_136059

open Set

theorem intersection_complement_A_B :
  let A := {x : ℝ | x + 1 > 0}
  let B := {-2, -1, 0, 1}
  (compl A ∩ B : Set ℝ) = {-2, -1} :=
by
  sorry

end intersection_complement_A_B_l136_136059


namespace rational_roots_of_second_equation_l136_136130

theorem rational_roots_of_second_equation (p q c : ℚ) (hpq : p + q = 1) (hc : p * q = c) : 
  ∀ (x : ℚ), is_root (x^2 + p * x - q) x :=
by sorry

end rational_roots_of_second_equation_l136_136130


namespace find_minimum_fuse_length_l136_136979

def safeZone : ℝ := 70
def fuseBurningSpeed : ℝ := 0.112
def personSpeed : ℝ := 7
def minimumFuseLength : ℝ := 1.1

theorem find_minimum_fuse_length (x : ℝ) (h1 : x ≥ 0):
  (safeZone / personSpeed) * fuseBurningSpeed ≤ x :=
by
  sorry

end find_minimum_fuse_length_l136_136979


namespace opposite_of_neg_2023_l136_136560

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136560


namespace students_meet_time_l136_136876

theorem students_meet_time :
  ∀ (distance rate1 rate2 : ℝ),
    distance = 350 ∧ rate1 = 1.6 ∧ rate2 = 1.9 →
    distance / (rate1 + rate2) = 100 := by
  sorry

end students_meet_time_l136_136876


namespace Nellie_legos_l136_136842

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end Nellie_legos_l136_136842


namespace solve_triangle_l136_136373

open Real

noncomputable def triangle_sides_angles (a b c A B C : ℝ) : Prop :=
  b^2 - (2 * (sqrt 3 / 3) * b * c * sin A) + c^2 = a^2

theorem solve_triangle 
  (b c : ℝ) (hb : b = 2) (hc : c = 3)
  (h : triangle_sides_angles a b c A B C) : 
  (A = π / 3) ∧ 
  (a = sqrt 7) ∧ 
  (sin (2 * B - A) = 3 * sqrt 3 / 14) := 
by
  sorry

end solve_triangle_l136_136373


namespace opposite_of_neg_2023_l136_136580

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136580


namespace angle_P_in_quadrilateral_l136_136378

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end angle_P_in_quadrilateral_l136_136378


namespace opposite_of_neg_2023_l136_136570

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136570


namespace opposite_of_negative_2023_l136_136709

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136709


namespace subtraction_calculation_l136_136724

theorem subtraction_calculation (a b : ℤ) (h : b = 40) (h1 : a = b - 1) : (a * a) = (b * b) - 79 := 
by
  -- Using the given condition
  have h2 : a * a = (b - 1) * (b - 1),
  from by rw [h1],
  -- Expanding using binomial theorem
  rw [mul_sub, sub_mul, mul_one, ← square_eq, sub_sub, one_mul, one_mul] at h2,
  -- Proving the theorem
  rw [sub_add] at h2,
  exact h2,
  sorry

end subtraction_calculation_l136_136724


namespace paint_leftover_l136_136121

theorem paint_leftover (containers total_walls tiles_wall paint_ceiling : ℕ) 
  (h_containers : containers = 16) 
  (h_total_walls : total_walls = 4) 
  (h_tiles_wall : tiles_wall = 1) 
  (h_paint_ceiling : paint_ceiling = 1) : 
  containers - ((total_walls - tiles_wall) * (containers / total_walls)) - paint_ceiling = 3 :=
by 
  sorry

end paint_leftover_l136_136121


namespace chocolate_eggs_total_weight_l136_136995

def total_weight_after_discarding_box_b : ℕ :=
  let weight_large := 14
  let weight_medium := 10
  let weight_small := 6
  let box_A_weight := 4 * weight_large + 2 * weight_medium
  let box_B_weight := 6 * weight_small + 2 * weight_large
  let box_C_weight := 4 * weight_large + 3 * weight_medium
  let box_D_weight := 4 * weight_medium + 4 * weight_small
  let box_E_weight := 4 * weight_small + 2 * weight_medium
  box_A_weight + box_C_weight + box_D_weight + box_E_weight

theorem chocolate_eggs_total_weight : total_weight_after_discarding_box_b = 270 := by
  sorry

end chocolate_eggs_total_weight_l136_136995


namespace cafe_purchase_max_items_l136_136813

theorem cafe_purchase_max_items (total_money sandwich_cost soft_drink_cost : ℝ) (total_money_pos sandwich_cost_pos soft_drink_cost_pos : total_money > 0 ∧ sandwich_cost > 0 ∧ soft_drink_cost > 0) :
    total_money = 40 ∧ sandwich_cost = 5 ∧ soft_drink_cost = 1.50 →
    ∃ s d : ℕ, s + d = 10 ∧ total_money = sandwich_cost * s + soft_drink_cost * d :=
by
  sorry

end cafe_purchase_max_items_l136_136813


namespace quadratic_inequality_real_solutions_l136_136205

theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∃ x : ℝ, x^2 - 10 * x + c < 0) ↔ c < 25 :=
sorry

end quadratic_inequality_real_solutions_l136_136205


namespace cube_inequality_l136_136789

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by
  sorry

end cube_inequality_l136_136789


namespace deductive_reasoning_correctness_l136_136305

theorem deductive_reasoning_correctness (major_premise minor_premise form_of_reasoning correct : Prop) 
  (h : major_premise ∧ minor_premise ∧ form_of_reasoning) : correct :=
  sorry

end deductive_reasoning_correctness_l136_136305


namespace tea_mixture_ratio_l136_136740

theorem tea_mixture_ratio
    (x y : ℝ)
    (h₁ : 62 * x + 72 * y = 64.5 * (x + y)) :
    x / y = 3 := by
  sorry

end tea_mixture_ratio_l136_136740


namespace opposite_of_neg_2023_l136_136622

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136622


namespace opposite_of_neg_2023_l136_136636

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136636


namespace min_value_of_ratio_l136_136216

noncomputable def min_ratio (a b c d : ℕ) : ℝ :=
  let num := 1000 * a + 100 * b + 10 * c + d
  let denom := a + b + c + d
  (num : ℝ) / (denom : ℝ)

theorem min_value_of_ratio : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  min_ratio a b c d = 60.5 :=
by
  sorry

end min_value_of_ratio_l136_136216


namespace remaining_distance_proof_l136_136419

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l136_136419


namespace opposite_of_neg_2023_l136_136515

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136515


namespace triangle_inequality_min_diff_l136_136295

theorem triangle_inequality_min_diff
  (DE EF FD : ℕ) 
  (h1 : DE + EF + FD = 398)
  (h2 : DE < EF ∧ EF ≤ FD) : 
  EF - DE = 1 :=
by
  sorry

end triangle_inequality_min_diff_l136_136295


namespace savings_after_purchase_l136_136404

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l136_136404


namespace pairs_of_managers_refusing_l136_136026

theorem pairs_of_managers_refusing (h_comb : (Nat.choose 8 4) = 70) (h_restriction : 55 = 70 - n * (Nat.choose 6 2)) : n = 1 :=
by
  have h1 : Nat.choose 8 4 = 70 := h_comb
  have h2 : Nat.choose 6 2 = 15 := by sorry -- skipped calculation for (6 choose 2), which is 15
  have h3 : 55 = 70 - n * 15 := h_restriction
  sorry -- proof steps to show n = 1

end pairs_of_managers_refusing_l136_136026


namespace no_yarn_earnings_l136_136044

noncomputable def yarn_cost : Prop :=
  let monday_yards := 20
  let tuesday_yards := 2 * monday_yards
  let wednesday_yards := (1 / 4) * tuesday_yards
  let total_yards := monday_yards + tuesday_yards + wednesday_yards
  let fabric_cost_per_yard := 2
  let total_fabric_earnings := total_yards * fabric_cost_per_yard
  let total_earnings := 140
  total_fabric_earnings = total_earnings

theorem no_yarn_earnings:
  yarn_cost :=
sorry

end no_yarn_earnings_l136_136044


namespace opposite_of_neg_2023_l136_136655

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136655


namespace minimize_feed_costs_l136_136296

theorem minimize_feed_costs 
  (x y : ℝ)
  (h1: 5 * x + 3 * y ≥ 30)
  (h2: 2.5 * x + 3 * y ≥ 22.5)
  (h3: x ≥ 0)
  (h4: y ≥ 0)
  : (x = 3 ∧ y = 5) ∧ (x + y = 8) := 
sorry

end minimize_feed_costs_l136_136296


namespace opposite_of_neg_2023_l136_136503

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136503


namespace consecutive_odd_split_l136_136352

theorem consecutive_odd_split (m : ℕ) (hm : m > 1) : (∃ n : ℕ, n = 2015 ∧ n < ((m + 2) * (m - 1)) / 2) → m = 45 :=
by
  sorry

end consecutive_odd_split_l136_136352


namespace range_of_a_l136_136067

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a > 0

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a) (h2 : q a) : a ≤ -2 :=
by
  sorry

end range_of_a_l136_136067


namespace derivative_of_f_l136_136440

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f :
  ∀ x ≠ 0, deriv f x = ((-x * Real.sin x - Real.cos x) / (x^2)) := sorry

end derivative_of_f_l136_136440


namespace max_value_2ab_2bc_2cd_2da_l136_136863

theorem max_value_2ab_2bc_2cd_2da {a b c d : ℕ} :
  (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 7) ∧
  (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 7) ∧
  (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 7) ∧
  (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
  (b ≠ c) ∧ (b ≠ d) ∧
  (c ≠ d)
  → 2 * (a * b + b * c + c * d + d * a) ≤ 144 :=
by
  sorry

end max_value_2ab_2bc_2cd_2da_l136_136863


namespace min_value_of_one_over_a_plus_one_over_b_l136_136397

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l136_136397


namespace opposite_of_neg_2023_l136_136447

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136447


namespace nine_cubed_expansion_l136_136153

theorem nine_cubed_expansion : 9^3 + 3 * 9^2 + 3 * 9 + 1 = 1000 := 
by 
  sorry

end nine_cubed_expansion_l136_136153


namespace opposite_of_neg_2023_l136_136585

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136585


namespace range_of_k_l136_136895

theorem range_of_k (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ (x, y) = (1, 2)) →
  (3 < k ∧ k < 7) :=
by
  intros hxy
  sorry

end range_of_k_l136_136895


namespace matrix_eigenvalue_problem_l136_136779

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l136_136779


namespace problem_1_problem_2_problem_3_l136_136065

-- Simplified and combined statements for clarity
theorem problem_1 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  f 3 + f (-1) = -3 := sorry

theorem problem_2 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  ∀ x, f x = if x ≤ 0 then Real.logb (1/2) (-x + 1) else Real.logb (1/2) (x + 1) := sorry

theorem problem_3 (f : ℝ → ℝ) (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1))
  (h_cond_ev : ∀ x, f x = f (-x)) (a : ℝ) : 
  f (a - 1) < -1 ↔ a ∈ ((Set.Iio 0) ∪ (Set.Ioi 2)) := sorry

end problem_1_problem_2_problem_3_l136_136065


namespace percentage_j_of_k_theorem_l136_136370

noncomputable def percentage_j_of_k 
  (j k l m : ℝ) (x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : Prop :=
  x = 500

theorem percentage_j_of_k_theorem 
  (j k l m : ℝ) (x : ℝ)
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : percentage_j_of_k j k l m x h1 h2 h3 h4 :=
by 
  sorry

end percentage_j_of_k_theorem_l136_136370


namespace nat_numbers_square_minus_one_power_of_prime_l136_136785

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ ∃ (k : ℕ), x = p ^ k

theorem nat_numbers_square_minus_one_power_of_prime (n : ℕ) (hn : 1 ≤ n) :
  is_power_of_prime (n ^ 2 - 1) ↔ (n = 2 ∨ n = 3) := by
  sorry

end nat_numbers_square_minus_one_power_of_prime_l136_136785


namespace conference_end_time_correct_l136_136024

-- Define the conference conditions
def conference_start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes
def conference_duration : ℕ := 450 -- 450 minutes duration
def daylight_saving_adjustment : ℕ := 60 -- clocks set forward by one hour

-- Define the end time computation
def end_time_without_daylight_saving : ℕ := conference_start_time + conference_duration
def end_time_with_daylight_saving : ℕ := end_time_without_daylight_saving + daylight_saving_adjustment

-- Prove that the conference ended at 11:30 p.m. (11:30 p.m. in minutes is 23 * 60 + 30)
theorem conference_end_time_correct : end_time_with_daylight_saving = 23 * 60 + 30 := by
  sorry

end conference_end_time_correct_l136_136024


namespace count_integers_between_3250_and_3500_with_increasing_digits_l136_136964

theorem count_integers_between_3250_and_3500_with_increasing_digits :
  ∃ n : ℕ, n = 20 ∧
    (∀ x : ℕ, 3250 ≤ x ∧ x ≤ 3500 →
      ∀ (d1 d2 d3 d4 : ℕ),
        d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧
        (x = d1 * 1000 + d2 * 100 + d3 * 10 + d4) →
        (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)) :=
  sorry

end count_integers_between_3250_and_3500_with_increasing_digits_l136_136964


namespace proof_probability_and_expectations_l136_136723

/-- Number of white balls drawn from two boxes --/
def X : ℕ := 1

/-- Number of red balls drawn from two boxes --/
def Y : ℕ := 1

/-- Given the conditions, the probability of drawing one white ball is 1/2, and
the expected value of white balls drawn is greater than the expected value of red balls drawn --/
theorem proof_probability_and_expectations :
  (∃ (P_X : ℚ), P_X = 1 / 2) ∧ (∃ (E_X E_Y : ℚ), E_X > E_Y) :=
by {
  sorry
}

end proof_probability_and_expectations_l136_136723


namespace arithmetic_sequence_properties_l136_136793

noncomputable def arithmetic_sequence (n : ℕ) : ℕ :=
  4 * n - 3

noncomputable def sum_of_first_n_terms (n : ℕ) : ℕ :=
  2 * n^2 - n

noncomputable def sum_of_reciprocal_sequence (n : ℕ) : ℝ :=
  n / (4 * n + 1)

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 3 = 9) →
  (arithmetic_sequence 8 = 29) →
  (∀ n, arithmetic_sequence n = 4 * n - 3) ∧
  (∀ n, sum_of_first_n_terms n = 2 * n^2 - n) ∧
  (∀ n, sum_of_reciprocal_sequence n = n / (4 * n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l136_136793


namespace committee_selection_license_plate_l136_136890

-- Problem 1
theorem committee_selection : 
  let members : Finset ℕ := (Finset.range 5) in
  let not_entertainment : Finset ℕ := {0, 1} in
  let eligible_for_entertainment : Finset ℕ := members \ not_entertainment in
  eligible_for_entertainment.card = 3 ∧ 
  (members \ {0}).card = 4 ∧ 
  (members \ {1}).card = 4 
  → Fintype.choose eligible_for_entertainment 1 * Fintype.perm (members \ {0}) 2 = 36
:= by sorry

-- Problem 2
theorem license_plate : 
  let english_letters : ℕ := 26 in
  let digits : ℕ := 10 in
  (english_letters ^ 2) * Fintype.arrangements digits 4 = 26^2 * Fintype.arrangements digits 4
:= by sorry

end committee_selection_license_plate_l136_136890


namespace households_without_car_or_bike_l136_136980

/--
In a neighborhood having 90 households, some did not have either a car or a bike.
If 16 households had both a car and a bike and 44 had a car, and
there were 35 households with a bike only.
Prove that there are 11 households that did not have either a car or a bike.
-/
theorem households_without_car_or_bike
  (total_households : ℕ)
  (both_car_and_bike : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : both_car_and_bike = 16)
  (H3 : car = 44)
  (H4 : bike_only = 35) :
  ∃ N : ℕ, N = total_households - (car - both_car_and_bike + bike_only + both_car_and_bike) ∧ N = 11 :=
by {
  sorry
}

end households_without_car_or_bike_l136_136980


namespace largest_number_l136_136923

theorem largest_number 
  (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)
  (hA : A = 0.986)
  (hB : B = 0.9851)
  (hC : C = 0.9869)
  (hD : D = 0.9807)
  (hE : E = 0.9819)
  : C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_l136_136923


namespace order_of_exponentials_l136_136949

theorem order_of_exponentials :
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  a < d ∧ d < b ∧ b < c :=
by
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  sorry

end order_of_exponentials_l136_136949


namespace remaining_savings_after_purchase_l136_136408

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l136_136408


namespace opposite_of_neg_2023_l136_136633

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136633


namespace opposite_of_neg_2023_l136_136679

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136679


namespace stickers_per_student_l136_136122

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end stickers_per_student_l136_136122


namespace opposite_of_neg2023_l136_136468

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136468


namespace arithmetic_geometric_sequence_l136_136219

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : d ≠ 0)
    (h₁ : a 3 = a 1 + 2 * d) (h₂ : a 9 = a 1 + 8 * d)
    (h₃ : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
    (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 := 
sorry

end arithmetic_geometric_sequence_l136_136219


namespace part1_part2_l136_136229

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := exp (a * x) * f x a + x

theorem part1 (a : ℝ) : 
  (a ≤ 0 → ∀ x, ∀ y, f x a ≤ y) ∧ (a > 0 → ∃ x, ∀ y, f x a ≤ y ∧ y = log (1 / a) - 2) :=
sorry

theorem part2 (a m : ℝ) (h_a : a > 0) (x1 x2 : ℝ) (h_x1 : 0 < x1) (h_x2 : x1 < x2) 
  (h_g1 : g x1 a = 0) (h_g2 : g x2 a = 0) : x1 * (x2 ^ 2) > exp m → m ≤ 3 :=
sorry

end part1_part2_l136_136229


namespace opposite_of_neg_2023_l136_136455

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136455


namespace problem1_problem2_l136_136109

-- Definitions
variables {a b z : ℝ}

-- Problem 1 translated to Lean
theorem problem1 (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) : -2 < a ∧ a < 1 := 
sorry

-- Problem 2 translated to Lean
theorem problem2 (h1 : a + 2 * b = 9) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  ∃ z : ℝ, z = a * b^2 ∧ ∀ w : ℝ, (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 2 * b = 9 ∧ w = a * b^2) → w ≤ 27 :=
sorry

end problem1_problem2_l136_136109


namespace opposite_of_neg_2023_l136_136446

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136446


namespace canteen_distances_l136_136748

theorem canteen_distances 
  (B G C : ℝ)
  (hB : B = 600)
  (hBG : G = 800)
  (hBC_eq_2x : ∃ x, C = 2 * x ∧ B = G + x + x) :
  G = 800 / 3 :=
by
  sorry

end canteen_distances_l136_136748


namespace opposite_of_neg_2023_l136_136669

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136669


namespace quadratic_equation_solution_l136_136284

noncomputable def findOrderPair (b d : ℝ) : Prop :=
  (b + d = 7) ∧ (b < d) ∧ (36 - 4 * b * d = 0)

theorem quadratic_equation_solution :
  ∃ b d : ℝ, findOrderPair b d ∧ (b, d) = ( (7 - Real.sqrt 13) / 2, (7 + Real.sqrt 13) / 2 ) :=
by
  sorry

end quadratic_equation_solution_l136_136284


namespace beijing_time_conversion_l136_136763

-- Define the conversion conditions
def new_clock_hours_in_day : Nat := 10
def new_clock_minutes_per_hour : Nat := 100
def new_clock_time_at_5_beijing_time : Nat := 12 * 60  -- converting 12 noon to minutes


-- Define the problem to prove the corresponding Beijing time 
theorem beijing_time_conversion :
  new_clock_minutes_per_hour * 5 = 500 → 
  new_clock_time_at_5_beijing_time = 720 →
  (720 + 175 * 1.44) = 4 * 60 + 12 :=
by
  intros h1 h2
  sorry

end beijing_time_conversion_l136_136763


namespace arithmetic_general_term_sum_b_terms_l136_136358

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

def b_sequence (b : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, b n = 2^(a n) + 2 * n

noncomputable def sum_b_sequence (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n, T n = (2 * 4^n + 3 * n^2 + 3 * n - 2) / 3

theorem arithmetic_general_term :
  ∀ (a : ℕ → ℤ) (S : ℕ → ℤ),
  (arithmetic_sequence a) →
  (sum_arithmetic_sequence a S) →
  a 3 = 5 →
  S 15 = 225 →
  ∀ n, a n = 2 * n - 1 :=
by
  intros a S ha hS ha3 hS15 n
  sorry

theorem sum_b_terms :
  ∀ (a b : ℕ → ℤ) (S T : ℕ → ℤ),
  (arithmetic_sequence a) →
  (sum_arithmetic_sequence a S) →
  a 3 = 5 →
  S 15 = 225 →
  (b_sequence b a) →
  (sum_b_sequence b T) →
  ∀ n, T n = (2 * 4^n + 3 * n^2 + 3 * n - 2) / 3 :=
by
  intros a b S T ha hS ha3 hS15 hb hT n
  sorry

end arithmetic_general_term_sum_b_terms_l136_136358


namespace opposite_of_neg2023_l136_136470

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136470


namespace line_point_relation_l136_136838

theorem line_point_relation (x1 y1 x2 y2 a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * x1 + b1 * y1 = c1)
  (h2 : a2 * x2 + b2 * y2 = c2)
  (h3 : a1 + b1 = c1)
  (h4 : a2 + b2 = 2 * c2)
  (h5 : dist (x1, y1) (x2, y2) ≥ (Real.sqrt 2) / 2) :
  c1 / a1 + a2 / c2 = 3 := 
sorry

end line_point_relation_l136_136838


namespace perfect_square_trinomial_m_l136_136241

theorem perfect_square_trinomial_m (m : ℝ) :
  (∀ x : ℝ, ∃ b : ℝ, x^2 + 2 * (m - 3) * x + 16 = (1 * x + b)^2) → (m = 7 ∨ m = -1) :=
by 
  intro h
  sorry

end perfect_square_trinomial_m_l136_136241


namespace find_x_l136_136922

theorem find_x (x : ℚ) : |x + 3| = |x - 4| → x = 1/2 := 
by 
-- Add appropriate content here
sorry

end find_x_l136_136922


namespace equation_of_line_passing_through_and_parallel_l136_136441

theorem equation_of_line_passing_through_and_parallel :
  ∀ (x y : ℝ), (x = -3 ∧ y = -1) → (∃ (C : ℝ), x - 2 * y + C = 0) → C = 1 :=
by
  intros x y h₁ h₂
  sorry

end equation_of_line_passing_through_and_parallel_l136_136441


namespace opposite_of_neg_2023_l136_136501

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136501


namespace mail_sorting_time_l136_136178

theorem mail_sorting_time :
  (1 / (1 / 3 + 1 / 6) = 2) :=
by
  sorry

end mail_sorting_time_l136_136178


namespace find_rate_percent_l136_136887

-- Definitions
def simpleInterest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Given conditions
def principal : ℕ := 900
def time : ℕ := 4
def simpleInterestValue : ℕ := 160

-- Rate percent
theorem find_rate_percent : 
  ∃ R : ℕ, simpleInterest principal R time = simpleInterestValue :=
by
  sorry

end find_rate_percent_l136_136887


namespace rectangles_on_grid_l136_136071

-- Define the grid dimensions
def m := 3
def n := 2

-- Define a function to count the total number of rectangles formed by the grid.
def count_rectangles (m n : ℕ) : ℕ := 
  (m * (m - 1) / 2 + n * (n - 1) / 2) * (n * (n - 1) / 2 + m * (m - 1) / 2) 

-- State the theorem we need to prove
theorem rectangles_on_grid : count_rectangles m n = 14 :=
  sorry

end rectangles_on_grid_l136_136071


namespace solve_simultaneous_equations_l136_136129

theorem solve_simultaneous_equations :
  (∃ x y : ℝ, x^2 + 3 * y = 10 ∧ 3 + y = 10 / x) ↔ 
  (x = 3 ∧ y = 1 / 3) ∨ 
  (x = 2 ∧ y = 2) ∨ 
  (x = -5 ∧ y = -5) := by sorry

end solve_simultaneous_equations_l136_136129


namespace opposite_of_neg_2023_l136_136487

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136487


namespace chess_tournament_games_l136_136088

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 2 * n * (n - 1) = 1200 :=
by
  sorry

end chess_tournament_games_l136_136088


namespace max_cut_length_l136_136745

theorem max_cut_length (board_size : ℕ) (total_pieces : ℕ) 
  (area_each : ℕ) 
  (total_area : ℕ)
  (total_perimeter : ℕ)
  (initial_perimeter : ℕ)
  (max_possible_length : ℕ)
  (h1 : board_size = 30) 
  (h2 : total_pieces = 225)
  (h3 : area_each = 4)
  (h4 : total_area = board_size * board_size)
  (h5 : total_perimeter = total_pieces * 10)
  (h6 : initial_perimeter = 4 * board_size)
  (h7 : max_possible_length = (total_perimeter - initial_perimeter) / 2) :
  max_possible_length = 1065 :=
by 
  -- Here, we do not include the proof as per the instructions
  sorry

end max_cut_length_l136_136745


namespace sum_of_first_150_mod_5000_l136_136298

theorem sum_of_first_150_mod_5000:
  let S := 150 * (150 + 1) / 2 in
  S % 5000 = 1325 :=
by
  sorry

end sum_of_first_150_mod_5000_l136_136298


namespace find_k_l136_136015

def condition (k : ℝ) : Prop := 24 / k = 4

theorem find_k (k : ℝ) (h : condition k) : k = 6 :=
sorry

end find_k_l136_136015


namespace initial_quantity_of_milk_in_container_A_l136_136164

variables {CA MB MC : ℝ}

theorem initial_quantity_of_milk_in_container_A (h1 : MB = 0.375 * CA)
    (h2 : MC = 0.625 * CA)
    (h_eq : MB + 156 = MC - 156) :
    CA = 1248 :=
by
  sorry

end initial_quantity_of_milk_in_container_A_l136_136164


namespace opposite_of_negative_2023_l136_136711

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136711


namespace solve_quadratic_solve_inequalities_l136_136888
open Classical

-- Define the equation for Part 1
theorem solve_quadratic (x : ℝ) : x^2 - 6 * x + 5 = 0 → (x = 1 ∨ x = 5) :=
by
  sorry

-- Define the inequalities for Part 2
theorem solve_inequalities (x : ℝ) : (x + 3 > 0) ∧ (2 * (x - 1) < 4) → (-3 < x ∧ x < 3) :=
by
  sorry

end solve_quadratic_solve_inequalities_l136_136888


namespace pieces_length_l136_136022

theorem pieces_length (L M S : ℝ) (h1 : L + M + S = 180)
  (h2 : L = M + S + 30)
  (h3 : M = L / 2 - 10) :
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end pieces_length_l136_136022


namespace opposite_of_neg2023_l136_136472

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136472


namespace boxes_with_neither_l136_136387

-- Definitions translating the conditions from the problem
def total_boxes : Nat := 15
def boxes_with_markers : Nat := 8
def boxes_with_crayons : Nat := 4
def boxes_with_both : Nat := 3

-- The theorem statement to prove
theorem boxes_with_neither : total_boxes - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 6 := by
  -- Proof will go here
  sorry

end boxes_with_neither_l136_136387


namespace opposite_of_neg_2023_l136_136549

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136549


namespace final_number_is_50_l136_136095

theorem final_number_is_50 (initial_ones initial_fours : ℕ) (h1 : initial_ones = 900) (h2 : initial_fours = 100) :
  ∃ (z : ℝ), (900 * (1:ℝ)^2 + 100 * (4:ℝ)^2) = z^2 ∧ z = 50 :=
by
  sorry

end final_number_is_50_l136_136095


namespace remaining_distance_l136_136423

variable (D : ℝ) -- remaining distance in miles

-- Current speed
def current_speed := 8.0
-- Increased speed by 50%
def increased_speed := 12.0
-- Reduced speed by 4 mph
def reduced_speed := 4.0

-- Trip time relations
def current_time := D / current_speed
def increased_time := D / increased_speed
def reduced_time := D / reduced_speed

-- Time difference condition
axiom condition : reduced_time = increased_time + 16.0

theorem remaining_distance : D = 96.0 :=
by
  -- proof placeholder
  sorry

end remaining_distance_l136_136423


namespace probability_D_l136_136173

section ProbabilityProof

variables (P : ℕ → ℚ)
variables (A B C D : ℕ)

-- Assume A = 1, B = 2, C = 3, D = 4 for region labels
-- Translation of conditions
axiom P_A: P A = 1 / 4
axiom P_B: P B = 1 / 3
axiom P_sum: P A + P B + P C + P D = 1

-- The theorem to prove
theorem probability_D : P D = 1 / 4 :=
sorry

end ProbabilityProof

end probability_D_l136_136173


namespace opposite_of_neg_2023_l136_136555

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136555


namespace opposite_of_neg_2023_l136_136450

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136450


namespace difference_of_squares_example_l136_136039

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end difference_of_squares_example_l136_136039


namespace value_range_f_l136_136870

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 2 * Real.cos x - Real.sin (2 * x) + 1

theorem value_range_f :
  ∀ x ∈ Set.Ico (-(5 * Real.pi) / 12) (Real.pi / 3), 
  f x ∈ Set.Icc ((3 : ℝ) / 2 - Real.sqrt 2) 3 :=
by
  sorry

end value_range_f_l136_136870


namespace opposite_of_neg_2023_l136_136583

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136583


namespace avg_weight_increase_l136_136277

theorem avg_weight_increase (A : ℝ) (X : ℝ) (hp1 : 8 * A - 65 + 105 = 8 * A + 40)
  (hp2 : 8 * (A + X) = 8 * A + 40) : X = 5 := 
by sorry

end avg_weight_increase_l136_136277


namespace opposite_of_neg_2023_l136_136692

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136692


namespace opposite_of_neg_2023_l136_136482

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136482


namespace sqrt32_plus_4sqrt_half_minus_sqrt18_l136_136768

theorem sqrt32_plus_4sqrt_half_minus_sqrt18 :
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18) = 3 * Real.sqrt 2 :=
sorry

end sqrt32_plus_4sqrt_half_minus_sqrt18_l136_136768


namespace travel_time_l136_136998

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end travel_time_l136_136998


namespace product_of_divisors_of_30_l136_136351

open Nat

def divisors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem product_of_divisors_of_30 :
  (divisors_of_30.foldr (· * ·) 1) = 810000 := by
  sorry

end product_of_divisors_of_30_l136_136351


namespace opposite_of_neg_2023_l136_136644

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136644


namespace marble_selection_l136_136385

theorem marble_selection : (∃ num_ways : ℕ, num_ways = 990 ∧ (∃ S : finset ℕ, S.card = 5 ∧ 
  (∃ subset_special : finset ℕ, subset_special.card = 2 ∧ subset_special ⊆ {0, 1, 2, 3} ∧ 
  ∃ subset_rest : finset ℕ, subset_rest.card = 3 ∧ subset_rest ⊆ (finset.range 15 \ {0, 1, 2, 3}) ∧ 
  subset_special ∪ subset_rest = S))) :=
sorry

end marble_selection_l136_136385


namespace prop_2_l136_136791

variables (m n : Plane → Prop) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop :=
  -- define perpendicular relationship between line and plane
  sorry

def parallel (m : Line) (n : Line) : Prop :=
  -- define parallel relationship between two lines
  sorry

-- The proof of proposition (2) converted into Lean 4 statement
theorem prop_2 (hm₁ : perpendicular m α) (hn₁ : perpendicular n α) : parallel m n :=
  sorry

end prop_2_l136_136791


namespace evaluate_expression_l136_136201

theorem evaluate_expression : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end evaluate_expression_l136_136201


namespace octagon_perimeter_l136_136293

-- Definitions based on conditions
def is_octagon (n : ℕ) : Prop := n = 8
def side_length : ℕ := 12

-- The proof problem statement
theorem octagon_perimeter (n : ℕ) (h : is_octagon n) : n * side_length = 96 := by
  sorry

end octagon_perimeter_l136_136293


namespace complement_intersection_example_l136_136363

open Set

variable (U A B : Set ℕ)

def C_U (A : Set ℕ) (U : Set ℕ) : Set ℕ := U \ A

theorem complement_intersection_example 
  (hU : U = {0, 1, 2, 3})
  (hA : A = {0, 1})
  (hB : B = {1, 2, 3}) :
  (C_U A U) ∩ B = {2, 3} :=
by
  sorry

end complement_intersection_example_l136_136363


namespace solution_set_of_inequality_l136_136717

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end solution_set_of_inequality_l136_136717


namespace restaurant_hamburgers_l136_136028

-- Define the conditions
def hamburgers_served : ℕ := 3
def hamburgers_left_over : ℕ := 6

-- Define the total hamburgers made
def hamburgers_made : ℕ := hamburgers_served + hamburgers_left_over

-- State and prove the theorem
theorem restaurant_hamburgers : hamburgers_made = 9 := by
  sorry

end restaurant_hamburgers_l136_136028


namespace line_through_point_with_opposite_intercepts_l136_136348

theorem line_through_point_with_opposite_intercepts :
  (∃ m : ℝ, (∀ x y : ℝ, y = m * x → (2,3) = (x, y)) ∧ ((∀ a : ℝ, a ≠ 0 → (x / a + y / (-a) = 1) → (2 - 3 = a ∧ a = -1)))) →
  ((∀ x y : ℝ, 3 * x - 2 * y = 0) ∨ (∀ x y : ℝ, x - y + 1 = 0)) :=
by
  sorry

end line_through_point_with_opposite_intercepts_l136_136348


namespace find_m_l136_136355

def line_eq (x y : ℝ) : Prop := x + 2 * y - 3 = 0

def circle_eq (x y m : ℝ) : Prop := x * x + y * y + x - 6 * y + m = 0

def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), line_eq x y ∧ line_eq (3 - 2 * y) y ∧ circle_eq x y m ∧ circle_eq (3 - 2 * y) y m) ∧
  (∃ (x1 y1 x2 y2 : ℝ), line_eq x1 y1 ∧ line_eq x2 y2 ∧ perpendicular_vectors x1 y1 x2 y2) → m = 3 :=
sorry

end find_m_l136_136355


namespace arthur_bought_2_hamburgers_on_second_day_l136_136908

theorem arthur_bought_2_hamburgers_on_second_day
  (H D X: ℕ)
  (h1: 3 * H + 4 * D = 10)
  (h2: D = 1)
  (h3: 2 * X + 3 * D = 7):
  X = 2 :=
by
  sorry

end arthur_bought_2_hamburgers_on_second_day_l136_136908


namespace peter_speed_l136_136099

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end peter_speed_l136_136099


namespace find_x_correct_l136_136017

theorem find_x_correct (x : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * 180 = 360 * x - 480)
  (h2 : (x + 4) + x + (x - 4) = 3 * x)
  (h3 : 100 = (360 * x - 480) / (3 * x)) : 
  x = 8 := 
sorry

end find_x_correct_l136_136017


namespace nellie_final_legos_l136_136840

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end nellie_final_legos_l136_136840


namespace distinct_real_numbers_a_l136_136206

theorem distinct_real_numbers_a (a x y z : ℝ) (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  (a = x + 1 / y ∧ a = y + 1 / z ∧ a = z + 1 / x) ↔ (a = 1 ∨ a = -1) :=
by sorry

end distinct_real_numbers_a_l136_136206


namespace danny_bottle_caps_l136_136920

theorem danny_bottle_caps 
  (wrappers_park : Nat := 46)
  (caps_park : Nat := 50)
  (wrappers_collection : Nat := 52)
  (more_caps_than_wrappers : Nat := 4)
  (h1 : caps_park = wrappers_park + more_caps_than_wrappers)
  (h2 : wrappers_collection = 52) : 
  (∃ initial_caps : Nat, initial_caps + caps_park = wrappers_collection + more_caps_than_wrappers) :=
by 
  use 6
  sorry

end danny_bottle_caps_l136_136920


namespace factorize_problem1_factorize_problem2_l136_136930

theorem factorize_problem1 (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

theorem factorize_problem2 (x y : ℝ) : 
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) :=
by sorry

end factorize_problem1_factorize_problem2_l136_136930


namespace income_M_l136_136856

variable (M N O : ℝ)

theorem income_M (h1 : (M + N) / 2 = 5050) 
                  (h2 : (N + O) / 2 = 6250) 
                  (h3 : (M + O) / 2 = 5200) : 
                  M = 2666.67 := 
by 
  sorry

end income_M_l136_136856


namespace opposite_of_neg_2023_l136_136678

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136678


namespace john_subtraction_number_l136_136727

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l136_136727


namespace opposite_of_negative_2023_l136_136609

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136609


namespace quad_function_intersects_x_axis_l136_136058

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quad_function_intersects_x_axis (m : ℝ) :
  (discriminant (2 * m) (8 * m + 1) (8 * m) ≥ 0) ↔ (m ≥ -1/16 ∧ m ≠ 0) :=
by
  sorry

end quad_function_intersects_x_axis_l136_136058


namespace fourth_quarter_points_sum_l136_136816

variable (a d b j : ℕ)

-- Conditions from the problem
def halftime_tied : Prop := 2 * a + d = 2 * b
def wildcats_won_by_four : Prop := 4 * a + 6 * d = 4 * b - 3 * j + 4

-- The proof goal to be established
theorem fourth_quarter_points_sum
  (h1 : halftime_tied a d b)
  (h2 : wildcats_won_by_four a d b j) :
  (a + 3 * d) + (b - 2 * j) = 28 :=
sorry

end fourth_quarter_points_sum_l136_136816


namespace collinear_points_sum_l136_136372

theorem collinear_points_sum (a b : ℝ) 
  (h_collin: ∃ k : ℝ, 
    (1 - a) / (a - a) = k * (a - b) / (b - b) ∧
    (a - a) / (2 - b) = k * (2 - 3) / (3 - 3) ∧
    (a - b) / (3 - 3) = k * (a - a) / (3 - b) ) : 
  a + b = 4 :=
by
  sorry

end collinear_points_sum_l136_136372


namespace sufficiency_and_necessity_of_p_and_q_l136_136795

noncomputable def p : Prop := ∀ k, k = Real.sqrt 3
noncomputable def q : Prop := ∀ k, ∃ y x, y = k * x + 2 ∧ x^2 + y^2 = 1

theorem sufficiency_and_necessity_of_p_and_q : (p → q) ∧ (¬ (q → p)) := by
  sorry

end sufficiency_and_necessity_of_p_and_q_l136_136795


namespace miles_run_by_harriet_l136_136782

def miles_run_by_all_runners := 285
def miles_run_by_katarina := 51
def miles_run_by_adriana := 74
def miles_run_by_tomas_tyler_harriet (total_run: ℝ) := (total_run - (miles_run_by_katarina + miles_run_by_adriana))

theorem miles_run_by_harriet : (miles_run_by_tomas_tyler_harriet miles_run_by_all_runners) / 3 = 53.33 := by
  sorry

end miles_run_by_harriet_l136_136782


namespace manuscript_typing_total_cost_is_1400_l136_136140

-- Defining the variables and constants based on given conditions
def cost_first_time_per_page := 10
def cost_revision_per_page := 5
def total_pages := 100
def pages_revised_once := 20
def pages_revised_twice := 30
def pages_no_revision := total_pages - pages_revised_once - pages_revised_twice

-- Calculations based on the given conditions
def cost_first_time :=
  total_pages * cost_first_time_per_page

def cost_revised_once :=
  pages_revised_once * cost_revision_per_page

def cost_revised_twice :=
  pages_revised_twice * cost_revision_per_page * 2

def total_cost :=
  cost_first_time + cost_revised_once + cost_revised_twice

-- Prove that the total cost equals the calculated value
theorem manuscript_typing_total_cost_is_1400 :
  total_cost = 1400 := by
  sorry

end manuscript_typing_total_cost_is_1400_l136_136140


namespace opposite_of_neg_2023_l136_136517

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136517


namespace opposite_of_neg_2023_l136_136621

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136621


namespace opposite_of_negative_2023_l136_136713

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136713


namespace opposite_of_neg_2023_l136_136640

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136640


namespace opposite_of_neg_2023_l136_136562

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136562


namespace positive_value_m_l136_136941

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l136_136941


namespace edward_spent_amount_l136_136924

-- Definitions based on the problem conditions
def initial_amount : ℕ := 18
def remaining_amount : ℕ := 2

-- The statement to prove: Edward spent $16
theorem edward_spent_amount : initial_amount - remaining_amount = 16 := by
  sorry

end edward_spent_amount_l136_136924


namespace difference_of_x_y_l136_136361

theorem difference_of_x_y :
  ∀ (x y : ℤ), x + y = 10 → x = 14 → x - y = 18 :=
by
  intros x y h1 h2
  sorry

end difference_of_x_y_l136_136361


namespace gu_xian_expression_right_triangle_l136_136018

-- Definitions for Part 1
def gu (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 - 1) / 2
def xian (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 + 1) / 2

-- Definitions for Part 2
def a (m : ℕ) (h : m > 1) : ℕ := m^2 - 1
def b (m : ℕ) (h : m > 1) : ℕ := 2 * m
def c (m : ℕ) (h : m > 1) : ℕ := m^2 + 1

-- Proof statement for Part 1
theorem gu_xian_expression (n : ℕ) (hn : n ≥ 3 ∧ n % 2 = 1) :
  gu n hn = (n^2 - 1) / 2 ∧ xian n hn = (n^2 + 1) / 2 :=
sorry

-- Proof statement for Part 2
theorem right_triangle (m : ℕ) (hm: m > 1) :
  (a m hm)^2 + (b m hm)^2 = (c m hm)^2 :=
sorry

end gu_xian_expression_right_triangle_l136_136018


namespace opposite_of_neg_2023_l136_136520

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136520


namespace probability_at_least_one_white_l136_136722

def total_number_of_pairs : ℕ := 10
def number_of_pairs_with_at_least_one_white_ball : ℕ := 7

theorem probability_at_least_one_white :
  (number_of_pairs_with_at_least_one_white_ball : ℚ) / (total_number_of_pairs : ℚ) = 7 / 10 :=
by
  sorry

end probability_at_least_one_white_l136_136722


namespace number_value_proof_l136_136368

theorem number_value_proof (x y : ℝ) (h1 : 0.5 * x = y + 20) (h2 : x - 2 * y = 40) : x = 40 := 
by
  sorry

end number_value_proof_l136_136368


namespace set_of_values_l136_136809

theorem set_of_values (a : ℝ) (h : 2 ∉ {x : ℝ | x - a < 0}) : a ≤ 2 := 
sorry

end set_of_values_l136_136809


namespace value_of_m_l136_136251

theorem value_of_m (a b c : ℤ) (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 26) 
  (h3 : (a + b + c) % 27 = m) (h4 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 :=
  by
  -- Proof is to be filled in
  sorry

end value_of_m_l136_136251


namespace eigenvalues_of_2x2_matrix_l136_136775

theorem eigenvalues_of_2x2_matrix :
  ∃ (k : ℝ), (k = 3 + 4 * Real.sqrt 6 ∨ k = 3 - 4 * Real.sqrt 6) ∧
            ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧
            ((3 : ℝ) * v.1 + 4 * v.2 = k * v.1 ∧ (6 : ℝ) * v.1 + 3 * v.2 = k * v.2) :=
begin
  sorry
end

end eigenvalues_of_2x2_matrix_l136_136775


namespace ratio_between_second_and_third_l136_136867

noncomputable def ratio_second_third : ℚ := sorry

theorem ratio_between_second_and_third (A B C : ℕ) (h₁ : A + B + C = 98) (h₂ : A * 3 = B * 2) (h₃ : B = 30) :
  ratio_second_third = 5 / 8 := sorry

end ratio_between_second_and_third_l136_136867


namespace evaluate_expression_l136_136199

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) : 
  x^3 * y^4 * z = 1/128 := 
by
  sorry

end evaluate_expression_l136_136199


namespace student_A_recruit_as_pilot_exactly_one_student_pass_l136_136270

noncomputable def student_A_recruit_prob : ℝ :=
  1 * 0.5 * 0.6 * 1

theorem student_A_recruit_as_pilot :
  student_A_recruit_prob = 0.3 :=
by
  sorry

noncomputable def one_student_pass_reinspection : ℝ :=
  0.5 * (1 - 0.6) * (1 - 0.75) +
  (1 - 0.5) * 0.6 * (1 - 0.75) +
  (1 - 0.5) * (1 - 0.6) * 0.75

theorem exactly_one_student_pass :
  one_student_pass_reinspection = 0.275 :=
by
  sorry

end student_A_recruit_as_pilot_exactly_one_student_pass_l136_136270


namespace total_cost_correct_l136_136090

-- Define the conditions
def total_employees : ℕ := 300
def emp_12_per_hour : ℕ := 200
def emp_14_per_hour : ℕ := 40
def emp_17_per_hour : ℕ := total_employees - emp_12_per_hour - emp_14_per_hour

def wage_12_per_hour : ℕ := 12
def wage_14_per_hour : ℕ := 14
def wage_17_per_hour : ℕ := 17

def hours_per_shift : ℕ := 8

-- Define the cost calculations
def cost_12 : ℕ := emp_12_per_hour * wage_12_per_hour * hours_per_shift
def cost_14 : ℕ := emp_14_per_hour * wage_14_per_hour * hours_per_shift
def cost_17 : ℕ := emp_17_per_hour * wage_17_per_hour * hours_per_shift

def total_cost : ℕ := cost_12 + cost_14 + cost_17

-- The theorem to be proved
theorem total_cost_correct :
  total_cost = 31840 :=
by
  sorry

end total_cost_correct_l136_136090


namespace sin_75_deg_l136_136035

theorem sin_75_deg : Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
by sorry

end sin_75_deg_l136_136035


namespace opposite_of_neg_2023_l136_136544

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136544


namespace shifted_function_correct_l136_136133

variable (x : ℝ)

/-- The original function -/
def original_function : ℝ := 3 * x - 4

/-- The function after shifting up by 2 units -/
def shifted_function : ℝ := original_function x + 2

theorem shifted_function_correct :
  shifted_function x = 3 * x - 2 :=
by
  sorry

end shifted_function_correct_l136_136133


namespace inequality_abc_l136_136837

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
sorry

end inequality_abc_l136_136837


namespace opposite_of_negative_2023_l136_136699

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136699


namespace Problem_statements_l136_136948

theorem Problem_statements (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = a * b) :
  (a + b ≥ 4) ∧
  ¬(a * b ≤ 4) ∧
  (a + 4 * b ≥ 9) ∧
  (1 / a ^ 2 + 2 / b ^ 2 ≥ 2 / 3) :=
by sorry

end Problem_statements_l136_136948


namespace opposite_neg_2023_l136_136534

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136534


namespace opposite_of_neg_2023_l136_136546

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136546


namespace opposite_of_neg_2023_l136_136443

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136443


namespace opposite_of_neg_2023_l136_136572

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136572


namespace opposite_of_neg_2023_l136_136696

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136696


namespace explicit_formula_solution_set_l136_136220

noncomputable def f : ℝ → ℝ 
| x => if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
       if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
       0

theorem explicit_formula (x : ℝ) :
  f x = if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
        if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
        0 := 
by 
  sorry 

theorem solution_set (x : ℝ) : 
  (0 < x ∧ x < 1 ∨ -4 < x ∧ x < -1) ↔ x * f x < 0 := 
by
  sorry

end explicit_formula_solution_set_l136_136220


namespace opposite_of_neg_2023_l136_136551

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136551


namespace find_PS_eq_13point625_l136_136249

theorem find_PS_eq_13point625 (PQ PR QR : ℝ) (h : ℝ) (QS SR : ℝ)
  (h_QS : QS^2 = 225 - h^2)
  (h_SR : SR^2 = 400 - h^2)
  (h_ratio : QS / SR = 3 / 7) :
  PS = 13.625 :=
by
  sorry

end find_PS_eq_13point625_l136_136249


namespace opposite_of_neg_2023_l136_136508

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136508


namespace maximum_xy_l136_136237

theorem maximum_xy (x y : ℕ) (h1 : 7 * x + 2 * y = 110) : ∃ x y, (7 * x + 2 * y = 110) ∧ (x > 0) ∧ (y > 0) ∧ (x * y = 216) :=
by
  sorry

end maximum_xy_l136_136237


namespace mean_proportional_234_104_l136_136741

theorem mean_proportional_234_104 : Real.sqrt (234 * 104) = 156 :=
by 
  sorry

end mean_proportional_234_104_l136_136741


namespace power_of_three_l136_136252

theorem power_of_three (a b : ℕ) (h1 : 360 = (2^3) * (3^2) * (5^1))
  (h2 : 2^a ∣ 360 ∧ ∀ n, 2^n ∣ 360 → n ≤ a)
  (h3 : 5^b ∣ 360 ∧ ∀ n, 5^n ∣ 360 → n ≤ b) :
  (1/3 : ℝ)^(b - a) = 9 :=
by sorry

end power_of_three_l136_136252


namespace factors_and_divisors_l136_136157

theorem factors_and_divisors :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (¬(∃ n : ℕ, 209 = 19 * n ∧ ¬ (∃ m : ℕ, 57 = 19 * m))) ∧
  (¬(¬(∃ n : ℕ, 90 = 30 * n) ∧ ¬(∃ m : ℕ, 75 = 30 * m))) ∧
  (¬(∃ n : ℕ, 51 = 17 * n ∧ ¬ (∃ m : ℕ, 68 = 17 * m))) ∧
  (∃ n : ℕ, 171 = 9 * n) :=
by {
  sorry
}

end factors_and_divisors_l136_136157


namespace opposite_of_neg_2023_l136_136677

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136677


namespace total_earnings_to_afford_car_l136_136163

-- Define the earnings per month
def monthlyEarnings : ℕ := 4000

-- Define the savings per month
def monthlySavings : ℕ := 500

-- Define the total amount needed to buy the car
def totalNeeded : ℕ := 45000

-- Define the number of months needed to save enough money
def monthsToSave : ℕ := totalNeeded / monthlySavings

-- Theorem stating the total money earned before he saves enough to buy the car
theorem total_earnings_to_afford_car : monthsToSave * monthlyEarnings = 360000 := by
  sorry

end total_earnings_to_afford_car_l136_136163


namespace perfect_square_trinomial_m_l136_136239

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2*(m-3)*x + 16) = (x + a)^2) ↔ (m = 7 ∨ m = -1) := 
sorry

end perfect_square_trinomial_m_l136_136239


namespace no_positive_integers_satisfy_condition_l136_136931

theorem no_positive_integers_satisfy_condition :
  ∀ (n : ℕ), n > 0 → ¬∃ (a b m : ℕ), a > 0 ∧ b > 0 ∧ m > 0 ∧ 
  (a + b * Real.sqrt n) ^ 2023 = Real.sqrt m + Real.sqrt (m + 2022) := by
  sorry

end no_positive_integers_satisfy_condition_l136_136931


namespace opposite_of_negative_2023_l136_136612

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136612


namespace opposite_of_neg2023_l136_136466

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136466


namespace vector_dot_product_proof_l136_136803

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_proof : 
  let a := (-1, 2)
  let b := (2, 3)
  vector_dot_product a (a.1 - b.1, a.2 - b.2) = 1 :=
by {
  sorry
}

end vector_dot_product_proof_l136_136803


namespace opposite_of_neg_2023_l136_136545

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136545


namespace opposite_of_neg_2023_l136_136498

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136498


namespace problem_l136_136068

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + Real.sin x) / (x ^ 2 + 1)

noncomputable def f' (x : ℝ) : ℝ := ((2 + Real.cos x) * (x ^ 2 + 1) - (2 * x + Real.sin x) * (2 * x)) / (x ^ 2 + 1) ^ 2

theorem problem : f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end problem_l136_136068


namespace Lauren_total_revenue_l136_136264

noncomputable def LaurenMondayEarnings (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.80)

noncomputable def LaurenTuesdayEarningsEUR (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.75)

noncomputable def convertEURtoUSD (eur : ℝ) : ℝ :=
  eur * (1 / 0.85)

noncomputable def convertGBPtoUSD (gbp : ℝ) : ℝ :=
  gbp * 1.38

noncomputable def LaurenWeekendEarnings (sales : ℝ) : ℝ :=
  (sales * 0.10)

theorem Lauren_total_revenue :
  let monday_views := 80
  let monday_subscriptions := 20
  let tuesday_views := 100
  let tuesday_subscriptions := 27
  let weekend_sales := 100

  let monday_earnings := LaurenMondayEarnings monday_views monday_subscriptions
  let tuesday_earnings_eur := LaurenTuesdayEarningsEUR tuesday_views tuesday_subscriptions
  let tuesday_earnings_usd := convertEURtoUSD tuesday_earnings_eur
  let weekend_earnings_gbp := LaurenWeekendEarnings weekend_sales
  let weekend_earnings_usd := convertGBPtoUSD weekend_earnings_gbp

  monday_earnings + tuesday_earnings_usd + weekend_earnings_usd = 132.68 :=
by
  sorry

end Lauren_total_revenue_l136_136264


namespace opposite_of_neg_2023_l136_136649

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136649


namespace mercers_theorem_gaussian_process_representation_verify_continuity_self_adjointness_l136_136019

noncomputable theory

-- Definition of the covariance function K
def covariance_function (a b : ℝ) := { K : ℝ × ℝ → ℝ // Continuous K }

-- Definition of the operator A in terms of K
def operator_A (a b : ℝ) (K : ℝ → ℝ → ℝ) (f : ℝ → ℝ) (s : ℝ) :=
  ∫ t in set.Icc a b, K s t * f t

-- Statement of Mercer's theorem
theorem mercers_theorem (a b : ℝ) (K : ℝ → ℝ → ℝ) (hK : ContinuousOn (λ p : ℝ × ℝ, K p.1 p.2) (set.Icc a b).prod (set.Icc a b)) :
  ∃ (λ : ℕ → ℝ) (φ : ℕ → ℝ → ℝ),
    (∀ n, λ n ≥ 0) ∧
    Orthonormal ℝ φ ∧
    (∀ n, ContinuousOn (λ s, φ n s) (set.Icc a b)) ∧
    (∀ s t, K s t = ∑' n, λ n * φ n s * φ n t) :=
sorry

-- Representation of the Gaussian process
theorem gaussian_process_representation (a b : ℝ) (K : ℝ → ℝ → ℝ) (λ : ℕ → ℝ) (φ : ℕ → ℝ → ℝ) (ξ : ℕ → ℝ) :
  (∀ n, ContinuousOn (λ s, φ n s) (set.Icc a b)) →
  (∀ n, ξ n ∼ 𝒩 0 1) →
  (∀ s t, K s t = ∑' n, λ n * φ n s * φ n t) →
  ∀ t, (X_t : ℝ) = ∑' n, ξ n * (λ n)^(1/2) * φ n t :=
sorry

-- Verify continuity and self-adjointness
theorem verify_continuity_self_adjointness (a b : ℝ) (K : ℝ → ℝ → ℝ) (hK : ContinuousOn (λ p : ℝ × ℝ, K p.1 p.2) (set.Icc a b).prod (set.Icc a b)) :
  ∃ (A : (ℝ → ℝ) → (ℝ → ℝ)),
    (∀ f g : ℝ → ℝ, ∥ A f - A g ∥_2 ≤ (b - a) * ∥K∥∞ * ∥ f - g ∥_2) ∧
    (∀ f g : ℝ → ℝ, ⟪ A f, g ⟫ = ⟪ f, A g ⟫) :=
sorry

end mercers_theorem_gaussian_process_representation_verify_continuity_self_adjointness_l136_136019


namespace stickers_per_student_l136_136124

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end stickers_per_student_l136_136124


namespace marble_selection_l136_136386

theorem marble_selection : (∃ num_ways : ℕ, num_ways = 990 ∧ (∃ S : finset ℕ, S.card = 5 ∧ 
  (∃ subset_special : finset ℕ, subset_special.card = 2 ∧ subset_special ⊆ {0, 1, 2, 3} ∧ 
  ∃ subset_rest : finset ℕ, subset_rest.card = 3 ∧ subset_rest ⊆ (finset.range 15 \ {0, 1, 2, 3}) ∧ 
  subset_special ∪ subset_rest = S))) :=
sorry

end marble_selection_l136_136386


namespace opposite_of_neg_2023_l136_136486

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136486


namespace ornamental_rings_remaining_l136_136925

theorem ornamental_rings_remaining :
  let r := 100 in
  let T := 200 + r in
  let rings_after_sale := T - (3 * T / 4) in
  let rings_after_mothers_purchase := rings_after_sale + 300 in
  rings_after_mothers_purchase - 150 = 225 :=
by
  sorry

end ornamental_rings_remaining_l136_136925


namespace number_of_type_one_triplets_l136_136744

-- Define the number of teams
def n_teams : ℕ := 15

-- Define the number of matches each team won
def matches_won_per_team : ℕ := 7

-- Define the total number of triplets
def total_triplets := Nat.choose n_teams 3

-- Define the number of triplets where each team in the trio won one match
def desired_triplets : ℕ := 140

-- Theorem statement
theorem number_of_type_one_triplets : total_triplets = 455 ∧ desired_triplets = 140 :=
  sorry

end number_of_type_one_triplets_l136_136744


namespace opposite_of_neg_2023_l136_136664

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136664


namespace price_reduction_to_achieve_profit_l136_136316

/-- 
A certain store sells clothing that cost $45$ yuan each to purchase for $65$ yuan each.
On average, they can sell $30$ pieces per day. For each $1$ yuan price reduction, 
an additional $5$ pieces can be sold per day. Given these conditions, 
prove that to achieve a daily profit of $800$ yuan, 
the price must be reduced by $10$ yuan per piece.
-/
theorem price_reduction_to_achieve_profit :
  ∃ x : ℝ, x = 10 ∧
    let original_cost := 45
    let original_price := 65
    let original_pieces_sold := 30
    let additional_pieces_per_yuan := 5
    let target_profit := 800
    let new_profit_per_piece := (original_price - original_cost) - x
    let new_pieces_sold := original_pieces_sold + additional_pieces_per_yuan * x
    new_profit_per_piece * new_pieces_sold = target_profit :=
by {
  sorry
}

end price_reduction_to_achieve_profit_l136_136316


namespace lattice_points_on_hyperbola_l136_136340

open Real

theorem lattice_points_on_hyperbola : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ S ↔ x^2 - y^2 = 65)) ∧ S.card = 8 :=
by
  sorry

end lattice_points_on_hyperbola_l136_136340


namespace sqrt_D_irrational_l136_136111

theorem sqrt_D_irrational (a b c : ℤ) (h : a + 1 = b) (h_c : c = a + b) : 
  Irrational (Real.sqrt ((a^2 : ℤ) + (b^2 : ℤ) + (c^2 : ℤ))) :=
  sorry

end sqrt_D_irrational_l136_136111


namespace desired_line_equation_exists_l136_136899

theorem desired_line_equation_exists :
  ∃ (a b c : ℝ), (a * 0 + b * 1 + c = 0) ∧
  (x - 3 * y + 10 = 0) ∧
  (2 * x + y - 8 = 0) ∧
  (a * x + b * y + c = 0) :=
by
  sorry

end desired_line_equation_exists_l136_136899


namespace solve_for_x_l136_136435

theorem solve_for_x (x : ℝ) : 3^(4 * x) = (81 : ℝ)^(1 / 4) → x = 1 / 4 :=
by
  intros
  sorry

end solve_for_x_l136_136435


namespace total_number_of_coins_l136_136382

theorem total_number_of_coins {N B : ℕ} 
    (h1 : B - 2 = Nat.floor (N / 9))
    (h2 : N - 6 * (B - 3) = 3) 
    : N = 45 :=
by
  sorry

end total_number_of_coins_l136_136382


namespace balance_difference_correct_l136_136762

def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

def angela_balance : ℝ :=
  compound_interest 5000 0.05 4 15

def bob_balance : ℝ :=
  simple_interest 7000 0.06 15

def balance_difference : ℝ :=
  bob_balance - angela_balance

theorem balance_difference_correct :
  abs balance_difference = 2732 :=
by
  sorry

end balance_difference_correct_l136_136762


namespace stephanie_falls_l136_136855

theorem stephanie_falls 
  (steven_falls : ℕ := 3)
  (sonya_falls : ℕ := 6)
  (h1 : sonya_falls = 6)
  (h2 : ∃ S : ℕ, sonya_falls = (S / 2) - 2 ∧ S > steven_falls) :
  ∃ S : ℕ, S - steven_falls = 13 :=
by
  sorry

end stephanie_falls_l136_136855


namespace opposite_of_neg_2023_l136_136626

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136626


namespace sequence_increasing_l136_136354

theorem sequence_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n + 3) : ∀ n, a (n + 1) > a n := 
by 
  sorry

end sequence_increasing_l136_136354


namespace sum_diff_l136_136138

-- Define the lengths of the ropes
def shortest_rope_length := 80
def ratio_shortest := 4
def ratio_middle := 5
def ratio_longest := 6

-- Use the given ratio to find the common multiple x.
def x := shortest_rope_length / ratio_shortest

-- Find the lengths of the other ropes
def middle_rope_length := ratio_middle * x
def longest_rope_length := ratio_longest * x

-- Define the sum of the longest and shortest ropes
def sum_of_longest_and_shortest := longest_rope_length + shortest_rope_length

-- Define the difference between the sum of the longest and shortest rope and the middle rope
def difference := sum_of_longest_and_shortest - middle_rope_length

-- Theorem statement
theorem sum_diff : difference = 100 := by
  sorry

end sum_diff_l136_136138


namespace opposite_of_neg_2023_l136_136485

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136485


namespace probability_of_non_perimeter_square_l136_136414

-- Defining the total number of squares on a 10x10 board
def total_squares : ℕ := 10 * 10

-- Defining the number of perimeter squares
def perimeter_squares : ℕ := 10 + 10 + (10 - 2) * 2

-- Defining the number of non-perimeter squares
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

-- Defining the probability of selecting a non-perimeter square
def probability_non_perimeter : ℚ := non_perimeter_squares / total_squares

-- The main theorem statement to be proved
theorem probability_of_non_perimeter_square:
  probability_non_perimeter = 16 / 25 := 
sorry

end probability_of_non_perimeter_square_l136_136414


namespace total_weight_remaining_eggs_l136_136994

theorem total_weight_remaining_eggs :
  let large_egg_weight := 14
  let medium_egg_weight := 10
  let small_egg_weight := 6

  let box_A_weight := 4 * large_egg_weight + 2 * medium_egg_weight
  let box_B_weight := 6 * small_egg_weight + 2 * large_egg_weight
  let box_C_weight := 4 * large_egg_weight + 3 * medium_egg_weight
  let box_D_weight := 4 * medium_egg_weight + 4 * small_egg_weight
  let box_E_weight := 4 * small_egg_weight + 2 * medium_egg_weight

  total_weight := box_A_weight + box_C_weight + box_D_weight + box_E_weight
  total_weight = 270 := 
by
  sorry

end total_weight_remaining_eggs_l136_136994


namespace total_time_to_fill_tank_with_leak_l136_136312

theorem total_time_to_fill_tank_with_leak
  (C : ℝ) -- Capacity of the tank
  (rate1 : ℝ := C / 20) -- Rate of pipe 1 filling the tank
  (rate2 : ℝ := C / 30) -- Rate of pipe 2 filling the tank
  (combined_rate : ℝ := rate1 + rate2) -- Combined rate of both pipes
  (effective_rate : ℝ := (2 / 3) * combined_rate) -- Effective rate considering the leak
  : (C / effective_rate = 18) :=
by
  -- The proof would go here but is removed per the instructions.
  sorry

end total_time_to_fill_tank_with_leak_l136_136312


namespace opposite_of_neg_2023_l136_136593

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136593


namespace contrapositive_of_implication_l136_136160

theorem contrapositive_of_implication (a : ℝ) (h : a > 0 → a > 1) : a ≤ 1 → a ≤ 0 :=
by
  sorry

end contrapositive_of_implication_l136_136160


namespace opposite_of_neg_2023_l136_136592

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136592


namespace hypotenuse_length_l136_136981

theorem hypotenuse_length (a b c : ℝ) 
  (h_right_angled : c^2 = a^2 + b^2) 
  (h_sum_squares : a^2 + b^2 + c^2 = 980) : 
  c = 70 :=
by
  sorry

end hypotenuse_length_l136_136981


namespace opposite_of_neg_2023_l136_136600

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136600


namespace exists_k_for_A_mul_v_eq_k_mul_v_l136_136777

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, 3]

theorem exists_k_for_A_mul_v_eq_k_mul_v (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (∃ k : ℝ, A.mul_vec v = k • v) →
  k = 3 + 2 * real.sqrt 6 ∨ k = 3 - 2 * real.sqrt 6 :=
by
  sorry

end exists_k_for_A_mul_v_eq_k_mul_v_l136_136777


namespace correct_choice_d_l136_136306

def is_quadrant_angle (alpha : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi

theorem correct_choice_d (alpha : ℝ) (k : ℤ) :
  is_quadrant_angle alpha k ↔ (2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi) := by
sorry

end correct_choice_d_l136_136306


namespace opposite_of_neg_2023_l136_136656

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136656


namespace opposite_of_neg_2023_l136_136697

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136697


namespace opposite_of_neg_2023_l136_136601

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136601


namespace opposite_of_neg_2023_l136_136698

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136698


namespace alcohol_percentage_first_solution_l136_136172

theorem alcohol_percentage_first_solution
  (x : ℝ)
  (h1 : 0 ≤ x ∧ x ≤ 1) -- since percentage in decimal form is between 0 and 1
  (h2 : 75 * x + 0.12 * 125 = 0.15 * 200) :
  x = 0.20 :=
by
  sorry

end alcohol_percentage_first_solution_l136_136172


namespace percentage_politics_not_local_politics_l136_136202

variables (total_reporters : ℝ) 
variables (reporters_cover_local_politics : ℝ) 
variables (reporters_not_cover_politics : ℝ)

theorem percentage_politics_not_local_politics :
  total_reporters = 100 → 
  reporters_cover_local_politics = 5 → 
  reporters_not_cover_politics = 92.85714285714286 → 
  (total_reporters - reporters_not_cover_politics) - reporters_cover_local_politics = 2.14285714285714 := 
by 
  intros ht hr hn
  rw [ht, hr, hn]
  norm_num


end percentage_politics_not_local_politics_l136_136202


namespace opposite_of_negative_2023_l136_136704

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136704


namespace opposite_of_neg_2023_l136_136481

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136481


namespace opposite_of_neg_2023_l136_136642

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136642


namespace inequality_sqrt_three_l136_136224

theorem inequality_sqrt_three (a b : ℤ) (h1 : a > b) (h2 : b > 1)
  (h3 : (a + b) ∣ (a * b + 1))
  (h4 : (a - b) ∣ (a * b - 1)) : a < Real.sqrt 3 * b := by
  sorry

end inequality_sqrt_three_l136_136224


namespace parrots_in_each_cage_l136_136027

theorem parrots_in_each_cage (P : ℕ) (h : 9 * P + 9 * 6 = 72) : P = 2 :=
sorry

end parrots_in_each_cage_l136_136027


namespace opposite_of_neg_2023_l136_136510

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136510


namespace my_inequality_l136_136114

open Real

variable {a b c : ℝ}

theorem my_inequality 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a * b + b * c + c * a = 1) :
  sqrt (a ^ 3 + a) + sqrt (b ^ 3 + b) + sqrt (c ^ 3 + c) ≥ 2 * sqrt (a + b + c) := 
  sorry

end my_inequality_l136_136114


namespace measure_angle_P_l136_136380

theorem measure_angle_P (P Q R S : ℝ) (hP : P = 3 * Q) (hR : 4 * R = P) (hS : 6 * S = P) (sum_angles : P + Q + R + S = 360) :
  P = 206 :=
by
  sorry

end measure_angle_P_l136_136380


namespace range_of_m_l136_136081

variable (x y m : ℝ)

def system_of_eq1 := 2 * x + y = -4 * m + 5
def system_of_eq2 := x + 2 * y = m + 4
def inequality1 := x - y > -6
def inequality2 := x + y < 8

theorem range_of_m:
  system_of_eq1 x y m → 
  system_of_eq2 x y m → 
  inequality1 x y → 
  inequality2 x y → 
  -5 < m ∧ m < 7/5 :=
by 
  intros h1 h2 h3 h4
  sorry

end range_of_m_l136_136081


namespace quadratic_roots_condition_l136_136079

theorem quadratic_roots_condition (k : ℝ) : 
  ((∃ x : ℝ, (k - 1) * x^2 + 4 * x + 1 = 0) ∧ ∃ x1 x2 : ℝ, x1 ≠ x2) ↔ (k < 5 ∧ k ≠ 1) :=
by {
  sorry  
}

end quadratic_roots_condition_l136_136079


namespace circle_tangent_area_l136_136770

noncomputable def circle_tangent_area_problem 
  (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) : ℝ :=
  if (radiusA = 1 ∧ radiusB = 1 ∧ radiusC = 2 ∧ tangent_midpoint) then 
    (4 * Real.pi) - (2 * Real.pi) 
  else 0

theorem circle_tangent_area (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) :
  radiusA = 1 → radiusB = 1 → radiusC = 2 → tangent_midpoint = true → 
  circle_tangent_area_problem radiusA radiusB radiusC tangent_midpoint = 2 * Real.pi :=
by
  intros
  simp [circle_tangent_area_problem]
  split_ifs
  · sorry
  · sorry

end circle_tangent_area_l136_136770


namespace opposite_of_neg2023_l136_136473

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136473


namespace parallelogram_base_length_l136_136755

theorem parallelogram_base_length (area height : ℝ) (h_area : area = 108) (h_height : height = 9) : 
  ∃ base : ℝ, base = area / height ∧ base = 12 := 
  by sorry

end parallelogram_base_length_l136_136755


namespace opposite_of_neg_2023_l136_136541

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136541


namespace exists_circle_with_exactly_n_integer_points_l136_136126

noncomputable def circle_with_n_integer_points (n : ℕ) : Prop :=
  ∃ r : ℤ, ∃ (xs ys : List ℤ), 
    xs.length = n ∧ ys.length = n ∧
    ∀ (x y : ℤ), x ∈ xs → y ∈ ys → x^2 + y^2 = r^2

theorem exists_circle_with_exactly_n_integer_points (n : ℕ) : 
  circle_with_n_integer_points n := 
sorry

end exists_circle_with_exactly_n_integer_points_l136_136126


namespace abc_inequality_l136_136057

theorem abc_inequality (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (h : a * b + a * c + b * c = a + b + c) : 
  a + b + c + 1 ≥ 4 * a * b * c :=
by 
  sorry

end abc_inequality_l136_136057


namespace kira_breakfast_time_l136_136103

theorem kira_breakfast_time (n_sausages : ℕ) (n_eggs : ℕ) (t_fry_per_sausage : ℕ) (t_scramble_per_egg : ℕ) (total_time : ℕ) :
  n_sausages = 3 → n_eggs = 6 → t_fry_per_sausage = 5 → t_scramble_per_egg = 4 → total_time = (n_sausages * t_fry_per_sausage + n_eggs * t_scramble_per_egg) →
  total_time = 39 :=
by
  intros h_sausages h_eggs h_fry h_scramble h_total
  rw [h_sausages, h_eggs, h_fry, h_scramble] at h_total
  exact h_total

end kira_breakfast_time_l136_136103


namespace tom_bike_rental_hours_calculation_l136_136413

variable (h : ℕ)
variable (base_cost : ℕ := 17)
variable (hourly_rate : ℕ := 7)
variable (total_paid : ℕ := 80)

theorem tom_bike_rental_hours_calculation (h : ℕ) 
  (base_cost : ℕ := 17) (hourly_rate : ℕ := 7) (total_paid : ℕ := 80) 
  (hours_eq : total_paid = base_cost + hourly_rate * h) : 
  h = 9 := 
by
  -- The proof is omitted.
  sorry

end tom_bike_rental_hours_calculation_l136_136413


namespace female_employees_l136_136166

theorem female_employees (E M F : ℕ) (h1 : 300 = 300) (h2 : (2/5 : ℚ) * E = (2/5 : ℚ) * M + 300) (h3 : E = M + F) : F = 750 := 
by
  sorry

end female_employees_l136_136166


namespace opposite_of_neg_2023_l136_136667

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136667


namespace bee_total_correct_l136_136144

def initial_bees : Nat := 16
def incoming_bees : Nat := 10
def total_bees : Nat := initial_bees + incoming_bees

theorem bee_total_correct : total_bees = 26 := by
  sorry

end bee_total_correct_l136_136144


namespace transportation_cost_l136_136261

theorem transportation_cost 
  (cost_per_kg : ℝ) 
  (weight_communication : ℝ) 
  (weight_sensor : ℝ) 
  (extra_sensor_cost_percentage : ℝ) 
  (cost_communication : ℝ)
  (basic_cost_sensor : ℝ)
  (extra_cost_sensor : ℝ)
  (total_cost : ℝ) : 
  cost_per_kg = 25000 → 
  weight_communication = 0.5 → 
  weight_sensor = 0.3 → 
  extra_sensor_cost_percentage = 0.10 →
  cost_communication = weight_communication * cost_per_kg →
  basic_cost_sensor = weight_sensor * cost_per_kg →
  extra_cost_sensor = extra_sensor_cost_percentage * basic_cost_sensor →
  total_cost = cost_communication + basic_cost_sensor + extra_cost_sensor →
  total_cost = 20750 :=
by sorry

end transportation_cost_l136_136261


namespace modulus_zero_l136_136256

/-- Given positive integers k and α such that 10k - α is also a positive integer, 
prove that the remainder when 8^(10k + α) + 6^(10k - α) - 7^(10k - α) - 2^(10k + α) is divided by 11 is 0. -/
theorem modulus_zero {k α : ℕ} (h₁ : 0 < k) (h₂ : 0 < α) (h₃ : 0 < 10 * k - α) :
  (8 ^ (10 * k + α) + 6 ^ (10 * k - α) - 7 ^ (10 * k - α) - 2 ^ (10 * k + α)) % 11 = 0 :=
by
  sorry

end modulus_zero_l136_136256


namespace opposite_of_neg_2023_l136_136494

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136494


namespace perfect_square_trinomial_m_l136_136240

theorem perfect_square_trinomial_m (m : ℝ) :
  (∀ x : ℝ, ∃ b : ℝ, x^2 + 2 * (m - 3) * x + 16 = (1 * x + b)^2) → (m = 7 ∨ m = -1) :=
by 
  intro h
  sorry

end perfect_square_trinomial_m_l136_136240


namespace opposite_of_neg_2023_l136_136635

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136635


namespace opposite_of_neg_2023_l136_136668

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136668


namespace opposite_neg_2023_l136_136529

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136529


namespace total_packages_of_gum_l136_136269

theorem total_packages_of_gum (R_total R_extra R_per_package A_total A_extra A_per_package : ℕ) 
  (hR1 : R_total = 41) (hR2 : R_extra = 6) (hR3 : R_per_package = 7)
  (hA1 : A_total = 23) (hA2 : A_extra = 3) (hA3 : A_per_package = 5) :
  (R_total - R_extra) / R_per_package + (A_total - A_extra) / A_per_package = 9 :=
by
  sorry

end total_packages_of_gum_l136_136269


namespace paddyfield_warblers_percentage_l136_136087

-- Definitions for the conditions
variables (B : ℝ) -- Total number of birds
variables (H : ℝ) -- Portion of hawks
variables (W : ℝ) -- Portion of paddyfield-warblers among non-hawks
variables (K : ℝ) -- Portion of kingfishers among non-hawks
variables (N : ℝ) -- Portion of non-hawks

-- Conditions
def hawks_portion : Prop := H = 0.30 * B
def non_hawks_portion : Prop := N = 0.70 * B
def paddyfield_warblers : Prop := W * N = W * (0.70 * B)
def kingfishers : Prop := K = 0.25 * W * N

-- Prove that the percentage of non-hawks that are paddyfield-warblers is 40%
theorem paddyfield_warblers_percentage :
  hawks_portion B H →
  non_hawks_portion B N →
  paddyfield_warblers B W N →
  kingfishers B W N K →
  W = 0.4 :=
by
  sorry

end paddyfield_warblers_percentage_l136_136087


namespace quadratic_relationship_l136_136798

variable (y_1 y_2 y_3 : ℝ)

-- Conditions
def vertex := (-2, 1)
def opens_downwards := true
def intersects_x_axis_at_two_points := true
def passes_through_points := [(1, y_1), (-1, y_2), (-4, y_3)]

-- Proof statement
theorem quadratic_relationship : y_1 < y_3 ∧ y_3 < y_2 :=
  sorry

end quadratic_relationship_l136_136798


namespace fraction_a_over_b_l136_136366

theorem fraction_a_over_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end fraction_a_over_b_l136_136366


namespace find_stiffnesses_l136_136147

def stiffnesses (m g x1 x2 k1 k2 : ℝ) : Prop :=
  (m = 3) ∧ (g = 10) ∧ (x1 = 0.4) ∧ (x2 = 0.075) ∧
  (k1 * k2 / (k1 + k2) * x1 = m * g) ∧
  ((k1 + k2) * x2 = m * g)

theorem find_stiffnesses (k1 k2 : ℝ) :
  stiffnesses 3 10 0.4 0.075 k1 k2 → 
  k1 = 300 ∧ k2 = 100 := 
sorry

end find_stiffnesses_l136_136147


namespace geometric_sequence_sum_l136_136956

theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : q ≠ 1)
    (hS2 : (a₁ * (1 - q^2)) / (1 - q) = 1)
    (hS4 : (a₁ * (1 - q^4)) / (1 - q) = 3) :
    (a₁ * (1 - q^8)) / (1 - q) = 15 := 
by
  sorry

end geometric_sequence_sum_l136_136956


namespace opposite_neg_2023_l136_136535

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136535


namespace opposite_of_neg_2023_l136_136694

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136694


namespace no_pos_int_solutions_l136_136267

theorem no_pos_int_solutions (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + 2^(2 * k) + 1 ≠ y^3 := by
  sorry

end no_pos_int_solutions_l136_136267


namespace find_p_l136_136341

variable (a b c p : ℚ)

theorem find_p (h1 : 5 / (a + b) = p / (a + c)) (h2 : p / (a + c) = 8 / (c - b)) : p = 13 := by
  sorry

end find_p_l136_136341


namespace maria_savings_l136_136412

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l136_136412


namespace union_M_N_l136_136889

noncomputable def M : Set ℝ := { x | x^2 - 3 * x = 0 }
noncomputable def N : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }

theorem union_M_N : M ∪ N = {0, 2, 3} :=
by {
  sorry
}

end union_M_N_l136_136889


namespace opposite_of_neg_2023_l136_136497

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136497


namespace opposite_of_negative_2023_l136_136610

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136610


namespace inequality_proof_l136_136835

theorem inequality_proof (a b c : ℝ) (ha1 : 0 ≤ a) (ha2 : a ≤ 1) (hb1 : 0 ≤ b) (hb2 : b ≤ 1) (hc1 : 0 ≤ c) (hc2 : c ≤ 1) :
  (a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by
  sorry

end inequality_proof_l136_136835


namespace find_f_two_l136_136061

-- Define the function f with the given properties
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1

-- Given conditions
variable (a b : ℝ)
axiom f_neg_two_zero : f (-2) a b = 0

-- Statement to be proven
theorem find_f_two : f 2 a b = 2 := 
by {
  sorry
}

end find_f_two_l136_136061


namespace find_b_l136_136282

theorem find_b (b : ℤ) (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : 21 * b = 160) : b = 9 := by
  sorry

end find_b_l136_136282


namespace nobel_prize_laureates_at_workshop_l136_136891

theorem nobel_prize_laureates_at_workshop :
  ∃ (T W W_and_N N_no_W X N : ℕ), 
    T = 50 ∧ 
    W = 31 ∧ 
    W_and_N = 16 ∧ 
    (N_no_W = X + 3) ∧ 
    (T - W = 19) ∧ 
    (N_no_W + X = 19) ∧ 
    (N = W_and_N + N_no_W) ∧ 
    N = 27 :=
by
  sorry

end nobel_prize_laureates_at_workshop_l136_136891


namespace opposite_of_neg_2023_l136_136553

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136553


namespace final_composite_score_is_correct_l136_136247

-- Defining scores
def written_exam_score : ℝ := 94
def interview_score : ℝ := 80
def practical_operation_score : ℝ := 90

-- Defining weights
def written_exam_weight : ℝ := 5
def interview_weight : ℝ := 2
def practical_operation_weight : ℝ := 3
def total_weight : ℝ := written_exam_weight + interview_weight + practical_operation_weight

-- Final composite score
noncomputable def composite_score : ℝ :=
  (written_exam_score * written_exam_weight + interview_score * interview_weight + practical_operation_score * practical_operation_weight)
  / total_weight

-- The theorem to be proved
theorem final_composite_score_is_correct : composite_score = 90 := by
  sorry

end final_composite_score_is_correct_l136_136247


namespace perfect_square_trinomial_l136_136243

theorem perfect_square_trinomial (m : ℤ) : (∃ a b : ℤ, a^2 = 1 ∧ b^2 = 16 ∧ (x : ℤ) → x^2 + 2*(m - 3)*x + 16 = (a*x + b)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_l136_136243


namespace complex_magnitude_l136_136273

-- Given definition of the complex number w with the condition provided
variables (w : ℂ) (h : w^2 = 48 - 14 * complex.I)

-- Statement of the problem to be proven
theorem complex_magnitude (w : ℂ) (h : w^2 = 48 - 14 * complex.I) : complex.abs w = 5 * real.sqrt 2 :=
sorry

end complex_magnitude_l136_136273


namespace opposite_of_neg2023_l136_136467

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136467


namespace opposite_of_neg_2023_l136_136681

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136681


namespace opposite_of_neg_2023_l136_136496

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136496


namespace orthocenter_circumradii_equal_l136_136313

-- Define a triangle with its orthocenter and circumradius
variables {A B C H : Point} (R r : ℝ)

-- Assume H is the orthocenter of triangle ABC
def is_orthocenter (H : Point) (A B C : Point) : Prop := 
  sorry -- This should state the definition or properties of an orthocenter

-- Assume the circumradius of triangle ABC is R 
def is_circumradius_ABC (A B C : Point) (R : ℝ) : Prop :=
  sorry -- This should capture the circumradius property

-- Assume circumradius of triangle BHC is r
def is_circumradius_BHC (B H C : Point) (r : ℝ) : Prop :=
  sorry -- This should capture the circumradius property
  
-- Prove that if H is the orthocenter of triangle ABC, the circumradius of ABC is R 
-- and the circumradius of BHC is r, then R = r
theorem orthocenter_circumradii_equal (h_orthocenter : is_orthocenter H A B C) 
  (h_circumradius_ABC : is_circumradius_ABC A B C R)
  (h_circumradius_BHC : is_circumradius_BHC B H C r) : R = r :=
  sorry

end orthocenter_circumradii_equal_l136_136313


namespace probability_of_king_then_queen_l136_136324

noncomputable def probability_top_king_second_queen : ℚ := 4 / 663

theorem probability_of_king_then_queen :
  let deck := (finset.range 52).image (λ i, (i / 13, i % 13)) in
  let shuffled_deck := deck.to_list in
  (∃ shuffled_deck : list (ℕ × ℕ), shuffled_deck = list.permutations deck.to_list) →
  let top_two := shuffled_deck.take 2 in
  (top_two.head = ⟨3, 12⟩ ∧ top_two.tail.head = ⟨2, 11⟩) →
  probability_top_king_second_queen = 4 / 663 :=
begin
  sorry

end probability_of_king_then_queen_l136_136324


namespace ratio_of_a_to_c_l136_136246

theorem ratio_of_a_to_c
  {a b c : ℕ}
  (h1 : a / b = 11 / 3)
  (h2 : b / c = 1 / 5) :
  a / c = 11 / 15 :=
by 
  sorry

end ratio_of_a_to_c_l136_136246


namespace circle_ratio_l136_136323

theorem circle_ratio (R r : ℝ) (h₁ : R > 0) (h₂ : r > 0) 
                     (h₃ : π * R^2 - π * r^2 = 3 * π * r^2) : R = 2 * r :=
by
  sorry

end circle_ratio_l136_136323


namespace opposite_of_negative_2023_l136_136618

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136618


namespace opposite_of_negative_2023_l136_136605

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136605


namespace min_red_chips_l136_136170

theorem min_red_chips (w b r : ℕ) (h1 : b ≥ w / 3) (h2 : b ≤ r / 4) (h3 : w + b ≥ 75) : r ≥ 76 :=
sorry

end min_red_chips_l136_136170


namespace opposite_of_neg_2023_l136_136624

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136624


namespace samuel_has_five_birds_l136_136892

theorem samuel_has_five_birds
  (birds_berries_per_day : ℕ)
  (total_berries_in_4_days : ℕ)
  (n_birds : ℕ)
  (h1 : birds_berries_per_day = 7)
  (h2 : total_berries_in_4_days = 140)
  (h3 : n_birds * birds_berries_per_day * 4 = total_berries_in_4_days) :
  n_birds = 5 := by
  sorry

end samuel_has_five_birds_l136_136892


namespace range_of_alpha_minus_beta_l136_136234

variable (α β : ℝ)

theorem range_of_alpha_minus_beta (h1 : -90 < α) (h2 : α < β) (h3 : β < 90) : -180 < α - β ∧ α - β < 0 := 
by
  sorry

end range_of_alpha_minus_beta_l136_136234


namespace opposite_of_negative_2023_l136_136703

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136703


namespace distance_to_second_picture_edge_l136_136328

/-- Given a wall of width 25 feet, with a first picture 5 feet wide centered on the wall,
and a second picture 3 feet wide centered in the remaining space, the distance 
from the nearest edge of the second picture to the end of the wall is 13.5 feet. -/
theorem distance_to_second_picture_edge :
  let wall_width := 25
  let first_picture_width := 5
  let second_picture_width := 3
  let side_space := (wall_width - first_picture_width) / 2
  let remaining_space := side_space
  let second_picture_side_space := (remaining_space - second_picture_width) / 2
  10 + 3.5 = 13.5 :=
by
  sorry

end distance_to_second_picture_edge_l136_136328


namespace remainder_when_divided_by_7_l136_136154

-- Definitions based on conditions
def k_condition (k : ℕ) : Prop :=
(k % 5 = 2) ∧ (k % 6 = 5) ∧ (k < 38)

-- Theorem based on the question and correct answer
theorem remainder_when_divided_by_7 {k : ℕ} (h : k_condition k) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l136_136154


namespace solution_set_f_inequality_l136_136227

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then 1 - 2^(-x)
else if x < 0 then 2^x - 1
else 0

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem solution_set_f_inequality : 
  is_odd_function f →
  {x | f x < -1/2} = {x | x < -1} := 
by
  sorry

end solution_set_f_inequality_l136_136227


namespace g_difference_l136_136991

def g (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

theorem g_difference (m : ℕ) : 
  g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) :=
by
  sorry

end g_difference_l136_136991


namespace perfect_square_trinomial_l136_136242

theorem perfect_square_trinomial (m : ℤ) : (∃ a b : ℤ, a^2 = 1 ∧ b^2 = 16 ∧ (x : ℤ) → x^2 + 2*(m - 3)*x + 16 = (a*x + b)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_l136_136242


namespace shaina_chocolate_l136_136987

-- Define the conditions
def total_chocolate : ℚ := 48 / 5
def number_of_piles : ℚ := 4

-- Define the assertion to prove
theorem shaina_chocolate : (total_chocolate / number_of_piles) = (12 / 5) := 
by 
  sorry

end shaina_chocolate_l136_136987


namespace john_marbles_selection_l136_136384

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end john_marbles_selection_l136_136384


namespace original_bullets_per_person_l136_136007

theorem original_bullets_per_person (x : ℕ) (h : 5 * (x - 4) = x) : x = 5 :=
by
  sorry

end original_bullets_per_person_l136_136007


namespace opposite_of_neg_2023_l136_136567

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136567


namespace opposite_of_neg_2023_l136_136643

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136643


namespace matrix_product_is_correct_l136_136350

-- Define the matrices A and B
def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 1, 1],
  ![2, 1, 2],
  ![1, 2, 3]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, 1, -1],
  ![2, -1, 1],
  ![1, 0, 1]
]

-- Define the expected product matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![6, 2, -1],
  ![6, 1, 1],
  ![8, -1, 4]
]

-- The statement of the problem
theorem matrix_product_is_correct : (A * B) = C := by
  sorry -- Proof is omitted as per instructions

end matrix_product_is_correct_l136_136350


namespace sum_of_angles_is_540_l136_136909

variables (angle1 angle2 angle3 angle4 angle5 angle6 angle7 : ℝ)

theorem sum_of_angles_is_540
  (h : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540 :=
sorry

end sum_of_angles_is_540_l136_136909


namespace opposite_of_neg_2023_l136_136502

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136502


namespace find_value_of_fraction_l136_136113

theorem find_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > x) (h : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 :=
by
  sorry

end find_value_of_fraction_l136_136113


namespace am_gm_inequality_l136_136257

theorem am_gm_inequality {a b : ℝ} (n : ℕ) (h₁ : n ≠ 1) (h₂ : a > b) (h₃ : b > 0) : 
  ( (a + b) / 2 )^n < (a^n + b^n) / 2 := 
sorry

end am_gm_inequality_l136_136257


namespace tangent_line_parallel_to_given_line_l136_136369

theorem tangent_line_parallel_to_given_line 
  (x : ℝ) (y : ℝ) (tangent_line : ℝ → ℝ) :
  (tangent_line y = x^2 - 1) → 
  (tangent_line = 4) → 
  (4 * x - y - 5 = 0) :=
by 
  sorry

end tangent_line_parallel_to_given_line_l136_136369


namespace polygon_interior_exterior_equal_l136_136085

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end polygon_interior_exterior_equal_l136_136085


namespace find_y_l136_136048

theorem find_y (y : ℕ) : (8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10) = 2 ^ y → y = 33 := 
by 
  sorry

end find_y_l136_136048


namespace smallest_n_for_factorable_quadratic_l136_136921

open Int

theorem smallest_n_for_factorable_quadratic : ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 72 → 3 * B + A = n) ∧ n = 35 :=
by
  sorry

end smallest_n_for_factorable_quadratic_l136_136921


namespace find_value_of_abc_cubed_l136_136400

-- Variables and conditions
variables {a b c : ℝ}
variables (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4)

-- The statement
theorem find_value_of_abc_cubed (ha : a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0) :
  a^3 + b^3 + c^3 = -3 * a * b * (a + b) :=
by
  sorry

end find_value_of_abc_cubed_l136_136400


namespace percentage_of_truth_speakers_l136_136975

theorem percentage_of_truth_speakers
  (L : ℝ) (hL: L = 0.2)
  (B : ℝ) (hB: B = 0.1)
  (prob_truth_or_lies : ℝ) (hProb: prob_truth_or_lies = 0.4)
  (T : ℝ)
: T = prob_truth_or_lies - L + B :=
sorry

end percentage_of_truth_speakers_l136_136975


namespace additional_grassy_ground_l136_136311

theorem additional_grassy_ground (r₁ r₂ : ℝ) (π : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 23) :
  π * r₂ ^ 2 - π * r₁ ^ 2 = 385 * π :=
  by
  subst h₁ h₂
  sorry

end additional_grassy_ground_l136_136311


namespace opposite_neg_2023_l136_136527

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136527


namespace sqrt_expr_evaluation_l136_136336

theorem sqrt_expr_evaluation : 
  (Real.sqrt 24) - 3 * (Real.sqrt (1 / 6)) + (Real.sqrt 6) = (5 * Real.sqrt 6) / 2 :=
by
  sorry

end sqrt_expr_evaluation_l136_136336


namespace compute_product_l136_136872

variables (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop :=
  x^3 - 3 * x * y^2 = 2017

def condition2 (x y : ℝ) : Prop :=
  y^3 - 3 * x^2 * y = 2016

theorem compute_product :
  condition1 x1 y1 → condition2 x1 y1 →
  condition1 x2 y2 → condition2 x2 y2 →
  condition1 x3 y3 → condition2 x3 y3 →
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1008 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end compute_product_l136_136872


namespace diana_erasers_l136_136197

theorem diana_erasers (number_of_friends : ℕ) (erasers_per_friend : ℕ) (total_erasers : ℕ) :
  number_of_friends = 48 →
  erasers_per_friend = 80 →
  total_erasers = number_of_friends * erasers_per_friend →
  total_erasers = 3840 :=
by
  intros h_friends h_erasers h_total
  sorry

end diana_erasers_l136_136197


namespace quadratic_distinct_real_roots_l136_136078

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) ≠ 0) ∧ ((4^2 - 4 * (k - 1) * 1) > 0) → k < 5 ∧ k ≠ 1 :=
by
  -- We state the problem conditions directly and prove the intended result.
  intro h
  cases h with hk hΔ
  sorry

end quadratic_distinct_real_roots_l136_136078


namespace range_x_y_l136_136788

variable (x y : ℝ)

theorem range_x_y (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) : 
  27 < x - y ∧ x - y < 56 :=
sorry

end range_x_y_l136_136788


namespace grant_room_proof_l136_136045

/-- Danielle's apartment has 6 rooms -/
def danielle_rooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment -/
def heidi_rooms : ℕ := 3 * danielle_rooms

/-- Jenny's apartment has 5 more rooms than Danielle's apartment -/
def jenny_rooms : ℕ := danielle_rooms + 5

/-- Lina's apartment has 7 rooms -/
def lina_rooms : ℕ := 7

/-- The total number of rooms from Danielle, Heidi, Jenny,
    and Lina's apartments -/
def total_rooms : ℕ := danielle_rooms + heidi_rooms + jenny_rooms + lina_rooms

/-- Grant's apartment has 1/3 less rooms than 1/9 of the
    combined total of rooms from Danielle's, Heidi's, Jenny's, and Lina's apartments -/
def grant_rooms : ℕ := (total_rooms / 9) - (total_rooms / 9) / 3

/-- Prove that Grant's apartment has 3 rooms -/
theorem grant_room_proof : grant_rooms = 3 :=
by
  sorry

end grant_room_proof_l136_136045


namespace find_m_n_sum_l136_136753

def P : ℕ × ℕ → ℚ
| (0, 0)       := 1
| (x+1, 0)     := 0
| (0, y+1)     := 0
| (x+1, y+1)   := (1 / 3) * P (x, y+1) + (1 / 3) * P (x+1, y) + (1 / 3) * P (x, y)

theorem find_m_n_sum : ∃ m n : ℕ, P (5, 5) = m / 3^n ∧ m % 3 ≠ 0 ∧ m + n = 6 := sorry

end find_m_n_sum_l136_136753


namespace opposite_of_neg_2023_l136_136620

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136620


namespace travel_time_l136_136997

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end travel_time_l136_136997


namespace paul_final_balance_l136_136428

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l136_136428


namespace average_fruits_per_basket_is_correct_l136_136135

noncomputable def average_fruits_per_basket : ℕ :=
  let basket_A := 15
  let basket_B := 30
  let basket_C := 20
  let basket_D := 25
  let basket_E := 35
  let total_fruits := basket_A + basket_B + basket_C + basket_D + basket_E
  let number_of_baskets := 5
  total_fruits / number_of_baskets

theorem average_fruits_per_basket_is_correct : average_fruits_per_basket = 25 := by
  unfold average_fruits_per_basket
  rfl

end average_fruits_per_basket_is_correct_l136_136135


namespace tabby_swimming_speed_l136_136438

theorem tabby_swimming_speed :
  ∃ (S : ℝ), S = 4.125 ∧ (∀ (D : ℝ), 6 = (2 * D) / ((D / S) + (D / 11))) :=
by {
 sorry
}

end tabby_swimming_speed_l136_136438


namespace angle_P_in_quadrilateral_l136_136377

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end angle_P_in_quadrilateral_l136_136377


namespace part1_part2_part3_l136_136951

variable {x y z : ℝ}

-- Given condition
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)

theorem part1 : 
  (x / y + y / z + z / x) / 3 ≥ 1 := sorry

theorem part2 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3 := sorry

theorem part3 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x := sorry

end part1_part2_part3_l136_136951


namespace range_of_t_for_point_in_upper_left_side_l136_136381

def point_in_upper_left_side_condition (x y : ℝ) : Prop :=
  x - y + 4 < 0

theorem range_of_t_for_point_in_upper_left_side :
  ∀ t : ℝ, point_in_upper_left_side_condition (-2) t ↔ t > 2 :=
by
  intros t
  unfold point_in_upper_left_side_condition
  simp
  sorry

end range_of_t_for_point_in_upper_left_side_l136_136381


namespace opposite_of_neg_2023_l136_136518

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136518


namespace opposite_of_neg_2023_l136_136647

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136647


namespace gumball_water_wednesday_l136_136804

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end gumball_water_wednesday_l136_136804


namespace find_A_d_minus_B_d_l136_136274

variable {d : ℕ} (A B : ℕ) (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2)

theorem find_A_d_minus_B_d (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2) :
  A - B = 3 :=
sorry

end find_A_d_minus_B_d_l136_136274


namespace opposite_of_negative_2023_l136_136615

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136615


namespace gcd_poly_l136_136066

theorem gcd_poly {b : ℕ} (h : 1116 ∣ b) : Nat.gcd (b^2 + 11 * b + 36) (b + 6) = 6 :=
by
  sorry

end gcd_poly_l136_136066


namespace children_tickets_l136_136332

-- Definition of the problem
variables (A C t : ℕ) (h_eq_people : A + C = t) (h_eq_money : 9 * A + 5 * C = 190)

-- The main statement we need to prove
theorem children_tickets (h_t : t = 30) : C = 20 :=
by {
  -- Proof will go here eventually
  sorry
}

end children_tickets_l136_136332


namespace original_average_of_numbers_l136_136096

theorem original_average_of_numbers 
  (A : ℝ) 
  (h : (A * 15) + (11 * 15) = 51 * 15) : 
  A = 40 :=
sorry

end original_average_of_numbers_l136_136096


namespace opposite_neg_2023_l136_136536

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136536


namespace g_54_l136_136836

def g : ℕ → ℤ := sorry

axiom g_multiplicative (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_6 : g 6 = 10
axiom g_18 : g 18 = 14

theorem g_54 : g 54 = 18 := by
  sorry

end g_54_l136_136836


namespace opposite_of_neg_2023_l136_136478

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136478


namespace opposite_of_neg_2023_l136_136558

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136558


namespace santa_chocolate_candies_l136_136432

theorem santa_chocolate_candies (C M : ℕ) (h₁ : C + M = 2023) (h₂ : C = 3 * M / 4) : C = 867 :=
sorry

end santa_chocolate_candies_l136_136432


namespace polygon_interior_exterior_eq_l136_136083

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_interior_exterior_eq_l136_136083


namespace number_of_rods_in_one_mile_l136_136797

theorem number_of_rods_in_one_mile :
  (1 : ℤ) * 6 * 60 = 360 :=
by
  sorry

end number_of_rods_in_one_mile_l136_136797


namespace opposite_of_neg_2023_l136_136641

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136641


namespace opposite_of_neg_2023_l136_136574

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136574


namespace opposite_of_neg_2023_l136_136582

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136582


namespace remaining_savings_after_purchase_l136_136409

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l136_136409


namespace profit_share_of_B_l136_136329

theorem profit_share_of_B (P : ℝ) (A_share B_share C_share : ℝ) :
  let A_initial := 8000
  let B_initial := 10000
  let C_initial := 12000
  let total_capital := A_initial + B_initial + C_initial
  let investment_ratio_A := A_initial / total_capital
  let investment_ratio_B := B_initial / total_capital
  let investment_ratio_C := C_initial / total_capital
  let total_profit := 4200
  let diff_AC := 560
  A_share = (investment_ratio_A * total_profit) →
  B_share = (investment_ratio_B * total_profit) →
  C_share = (investment_ratio_C * total_profit) →
  C_share - A_share = diff_AC →
  B_share = 1400 :=
by
  intros
  sorry

end profit_share_of_B_l136_136329


namespace minimum_n_required_l136_136258

def A_0 : (ℝ × ℝ) := (0, 0)

def is_on_x_axis (A : ℝ × ℝ) : Prop := A.snd = 0
def is_on_y_equals_x_squared (B : ℝ × ℝ) : Prop := B.snd = B.fst ^ 2
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := sorry

def A_n (n : ℕ) : ℝ × ℝ := sorry
def B_n (n : ℕ) : ℝ × ℝ := sorry

def euclidean_distance (P Q : ℝ × ℝ) : ℝ :=
  ((Q.fst - P.fst) ^ 2 + (Q.snd - P.snd) ^ 2) ^ (1/2)

theorem minimum_n_required (n : ℕ) (h1 : ∀ n, is_on_x_axis (A_n n))
    (h2 : ∀ n, is_on_y_equals_x_squared (B_n n))
    (h3 : ∀ n, is_equilateral_triangle (A_n (n-1)) (B_n n) (A_n n)) :
    (euclidean_distance A_0 (A_n n) ≥ 50) → n ≥ 17 :=
by sorry

end minimum_n_required_l136_136258


namespace remainder_24_l136_136303

-- Statement of the problem in Lean 4
theorem remainder_24 (y : ℤ) (h : y % 288 = 45) : y % 24 = 21 :=
by
  sorry

end remainder_24_l136_136303


namespace opposite_of_negative_2023_l136_136614

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136614


namespace opposite_of_neg_2023_l136_136484

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136484


namespace opposite_of_neg_2023_l136_136693

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136693


namespace fred_earnings_over_weekend_l136_136106

-- Fred's earning from delivering newspapers
def earnings_from_newspapers : ℕ := 16

-- Fred's earning from washing cars
def earnings_from_cars : ℕ := 74

-- Fred's total earnings over the weekend
def total_earnings : ℕ := earnings_from_newspapers + earnings_from_cars

-- Proof that total earnings is 90
theorem fred_earnings_over_weekend : total_earnings = 90 :=
by 
  -- sorry statement to skip the proof steps
  sorry

end fred_earnings_over_weekend_l136_136106


namespace find_sides_of_triangle_l136_136865

noncomputable def right_triangle_sides 
  (ρ : ℝ) (α : ℝ) 
  (hρ : ρ = 10) (hα : α = 23 + 14/60 * 1/3600 * 360) : 
  ℝ × ℝ × ℝ :=
let a := ρ * (1 + real.cot ((33 + 23/60) * real.pi / 180))
let b := ρ * (1 + real.cot ((11 + 37/60) * real.pi / 180))
let c := ρ * (real.cot ((33 + 23/60) * real.pi / 180) + real.cot ((11 + 37/60) * real.pi / 180))
in (a, b, c)

theorem find_sides_of_triangle : 
  ∀ (a b c : ℝ), 
  ∀ (ρ : ℝ) (α : ℝ)
  (hρ : ρ = 10) (hα : α = 23 + 14 / 60 * 1 / 3600 * 360), 
  right_triangle_sides ρ α hρ hα = (a, b, c) → 
  a = 25.18 ∧ b = 58.65 ∧ c = 63.82 := 
  by
  intros a b c ρ α hρ hα
  intro h
  rw right_triangle_sides at h
  sorry

end find_sides_of_triangle_l136_136865


namespace work_done_by_b_l136_136014

theorem work_done_by_b (x : ℝ) (h1 : (1/6) + (1/13) = (1/x)) : x = 78/7 :=
  sorry

end work_done_by_b_l136_136014


namespace find_abc_l136_136314

theorem find_abc (a b c : ℕ) (h_coprime_ab : gcd a b = 1) (h_coprime_ac : gcd a c = 1) 
  (h_coprime_bc : gcd b c = 1) (h1 : ab + bc + ac = 431) (h2 : a + b + c = 39) 
  (h3 : a + b + (ab / c) = 18) : 
  a = 7 ∧ b = 9 ∧ c = 23 := 
sorry

end find_abc_l136_136314


namespace base_addition_l136_136046

theorem base_addition (b : ℕ) (h : b > 1) :
  (2 * b^3 + 3 * b^2 + 8 * b + 4) + (3 * b^3 + 4 * b^2 + 1 * b + 7) = 
  1 * b^4 + 0 * b^3 + 2 * b^2 + 0 * b + 1 → b = 10 :=
by
  intro H
  -- skipping the detailed proof steps
  sorry

end base_addition_l136_136046


namespace total_doors_needed_correct_l136_136319

-- Define the conditions
def buildings : ℕ := 2
def floors_per_building : ℕ := 12
def apartments_per_floor : ℕ := 6
def doors_per_apartment : ℕ := 7

-- Define the total number of doors needed
def total_doors_needed : ℕ := buildings * floors_per_building * apartments_per_floor * doors_per_apartment

-- State the theorem to prove the total number of doors needed is 1008
theorem total_doors_needed_correct : total_doors_needed = 1008 := by
  sorry

end total_doors_needed_correct_l136_136319


namespace difference_of_squares_example_l136_136040

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end difference_of_squares_example_l136_136040


namespace Greg_and_Earl_together_l136_136343

-- Conditions
def Earl_initial : ℕ := 90
def Fred_initial : ℕ := 48
def Greg_initial : ℕ := 36

def Earl_to_Fred : ℕ := 28
def Fred_to_Greg : ℕ := 32
def Greg_to_Earl : ℕ := 40

def Earl_final : ℕ := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final : ℕ := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final : ℕ := Greg_initial + Fred_to_Greg - Greg_to_Earl

-- Theorem statement
theorem Greg_and_Earl_together : Greg_final + Earl_final = 130 := by
  sorry

end Greg_and_Earl_together_l136_136343


namespace opposite_of_neg_2023_l136_136684

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136684


namespace compound_interest_time_period_l136_136732

theorem compound_interest_time_period (P r I : ℝ) (n A t : ℝ) 
(hP : P = 6000) 
(hr : r = 0.10) 
(hI : I = 1260.000000000001) 
(hn : n = 1)
(hA : A = P + I)
(ht_eqn: (A / P) = (1 + r / n) ^ t) :
t = 2 := 
by sorry

end compound_interest_time_period_l136_136732


namespace opposite_of_neg_2023_l136_136665

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136665


namespace simplify_expression_l136_136434

variable {a : ℝ}

theorem simplify_expression (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by
  sorry

end simplify_expression_l136_136434


namespace gaussian_guardians_total_points_l136_136142

theorem gaussian_guardians_total_points :
  let daniel := 7
  let curtis := 8
  let sid := 2
  let emily := 11
  let kalyn := 6
  let hyojeong := 12
  let ty := 1
  let winston := 7
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston = 54 := by
  sorry

end gaussian_guardians_total_points_l136_136142


namespace probability_correct_l136_136742

-- Definitions of given conditions
def P_AB := 2 / 3
def P_BC := 1 / 2

-- Probability that at least one road is at least 5 miles long
def probability_at_least_one_road_is_5_miles_long : ℚ :=
  1 - (1 - P_AB) * (1 - P_BC)

theorem probability_correct :
  probability_at_least_one_road_is_5_miles_long = 5 / 6 :=
by
  -- Proof goes here
  sorry

end probability_correct_l136_136742


namespace decreasing_interval_range_of_a_l136_136959

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem decreasing_interval :
  (∀ x > 0, deriv f x = 1 + log x) →
  { x : ℝ | 0 < x ∧ x < 1/e } = { x | 0 < x ∧ deriv f x < 0 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a * x - 6) →
  a ≤ 5 + log 2 :=
sorry

end decreasing_interval_range_of_a_l136_136959


namespace white_balls_count_l136_136983

theorem white_balls_count (a : ℕ) (h : 3 / (3 + a) = 3 / 7) : a = 4 :=
by sorry

end white_balls_count_l136_136983


namespace opposite_of_neg_2023_l136_136554

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136554


namespace sixteen_pow_five_eq_four_pow_p_l136_136808

theorem sixteen_pow_five_eq_four_pow_p (p : ℕ) (h : 16^5 = 4^p) : p = 10 := 
  sorry

end sixteen_pow_five_eq_four_pow_p_l136_136808


namespace subtraction_calculation_l136_136725

theorem subtraction_calculation (a b : ℤ) (h : b = 40) (h1 : a = b - 1) : (a * a) = (b * b) - 79 := 
by
  -- Using the given condition
  have h2 : a * a = (b - 1) * (b - 1),
  from by rw [h1],
  -- Expanding using binomial theorem
  rw [mul_sub, sub_mul, mul_one, ← square_eq, sub_sub, one_mul, one_mul] at h2,
  -- Proving the theorem
  rw [sub_add] at h2,
  exact h2,
  sorry

end subtraction_calculation_l136_136725


namespace opposite_of_neg_2023_l136_136568

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136568


namespace opposite_of_neg_2023_l136_136493

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136493


namespace construction_doors_needed_l136_136318

theorem construction_doors_needed :
  (let number_of_buildings := 2 in
   let floors_per_building := 12 in
   let apartments_per_floor := 6 in
   let doors_per_apartment := 7 in
   let apartments_per_building := floors_per_building * apartments_per_floor in
   let total_apartments := apartments_per_building * number_of_buildings in
   let total_doors := total_apartments * doors_per_apartment in
   total_doors = 1008) :=
begin
  sorry
end

end construction_doors_needed_l136_136318


namespace peter_speed_l136_136098

theorem peter_speed (P : ℝ) (h1 : P >= 0) (h2 : 1.5 * P + 1.5 * (P + 3) = 19.5) : P = 5 := by
  sorry

end peter_speed_l136_136098


namespace new_price_after_increase_l136_136286

def original_price : ℝ := 220
def percentage_increase : ℝ := 0.15

def new_price (original_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  original_price + (original_price * percentage_increase)

theorem new_price_after_increase : new_price original_price percentage_increase = 253 := 
by
  sorry

end new_price_after_increase_l136_136286


namespace tangent_line_at_point_e_tangent_line_from_origin_l136_136960

-- Problem 1
theorem tangent_line_at_point_e (x y : ℝ) (h : y = Real.exp x) (h_e : x = Real.exp 1) :
    (Real.exp x) * x - y - Real.exp (x + 1) = 0 :=
sorry

-- Problem 2
theorem tangent_line_from_origin (x y : ℝ) (h : y = Real.exp x) :
    x = 1 →  Real.exp x * x - y = 0 :=
sorry

end tangent_line_at_point_e_tangent_line_from_origin_l136_136960


namespace opposite_of_neg2023_l136_136463

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136463


namespace state_B_more_candidates_l136_136375

theorem state_B_more_candidates (appeared : ℕ) (selected_A_pct selected_B_pct : ℝ)
  (h1 : appeared = 8000)
  (h2 : selected_A_pct = 0.06)
  (h3 : selected_B_pct = 0.07) :
  (selected_B_pct * appeared - selected_A_pct * appeared = 80) :=
by
  sorry

end state_B_more_candidates_l136_136375


namespace isosceles_triangle_side_length_l136_136033

theorem isosceles_triangle_side_length (a b : ℝ) (h : a < b) : 
  ∃ l : ℝ, l = (b - a) / 2 := 
sorry

end isosceles_triangle_side_length_l136_136033


namespace minimum_value_proof_l136_136394

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l136_136394


namespace value_of_x_l136_136973

theorem value_of_x :
  ∃ x : ℝ, x = 1.13 * 80 :=
sorry

end value_of_x_l136_136973


namespace opposite_of_negative_2023_l136_136714

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136714


namespace bike_cost_l136_136769

theorem bike_cost (h1: 8 > 0) (h2: 35 > 0) (weeks_in_month: ℕ := 4) (saved: ℕ := 720):
  let hourly_wage := 8
  let weekly_hours := 35
  let weekly_earnings := weekly_hours * hourly_wage
  let monthly_earnings := weekly_earnings * weeks_in_month
  let cost_of_bike := monthly_earnings - saved
  cost_of_bike = 400 :=
by
  sorry

end bike_cost_l136_136769


namespace opposite_of_neg_2023_l136_136589

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136589


namespace inequality_xyz_equality_condition_l136_136967

theorem inequality_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ 2 + x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  (x + y + z = 2 + x * y * z) ↔ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨
                                                  (x = 0 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 0 ∧ z = 1) ∨
                                                  (x = -1 ∧ y = 1 ∧ z = 0) :=
sorry

end inequality_xyz_equality_condition_l136_136967


namespace minimum_slope_tangent_point_coordinates_l136_136362

theorem minimum_slope_tangent_point_coordinates :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, (2 * x + a / x ≥ 4) ∧ (2 * x + a / x = 4 ↔ x = 1)) → 
  (1, 1) = (1, 1) := by
sorry

end minimum_slope_tangent_point_coordinates_l136_136362


namespace total_handshakes_at_convention_l136_136008

theorem total_handshakes_at_convention :
  let gremlins := 25
  let imps := 18
  let specific_gremlins := 5
  let friendly_gremlins := gremlins - specific_gremlins
  let handshakes_among_gremlins := (friendly_gremlins * (friendly_gremlins - 1)) / 2
  let handshakes_between_imps_and_gremlins := imps * gremlins
  handshakes_among_gremlins + handshakes_between_imps_and_gremlins = 640 := by
  sorry

end total_handshakes_at_convention_l136_136008


namespace find_a_l136_136077

-- Given conditions and definitions
def circle_eq (x y : ℝ) : Prop := (x^2 + y^2 - 2*x - 2*y + 1 = 0)
def line_eq (x y a : ℝ) : Prop := (x - 2*y + a = 0)
def chord_length (r : ℝ) : ℝ := 2 * r

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y) → 
  (∀ x y : ℝ, line_eq x y a) → 
  (∃ x y : ℝ, (x = 1 ∧ y = 1) ∧ (line_eq x y a ∧ chord_length 1 = 2)) → 
  a = 1 := by sorry

end find_a_l136_136077


namespace parabola_vertex_in_fourth_quadrant_l136_136961

theorem parabola_vertex_in_fourth_quadrant (a c : ℝ) (h : -a > 0 ∧ c < 0) :
  a < 0 ∧ c < 0 :=
by
  sorry

end parabola_vertex_in_fourth_quadrant_l136_136961


namespace opposite_of_neg_2023_l136_136552

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136552


namespace ratio_of_steps_l136_136761

-- Defining the conditions of the problem
def andrew_steps : ℕ := 150
def jeffrey_steps : ℕ := 200

-- Stating the theorem that we need to prove
theorem ratio_of_steps : andrew_steps / Nat.gcd andrew_steps jeffrey_steps = 3 ∧ jeffrey_steps / Nat.gcd andrew_steps jeffrey_steps = 4 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_steps_l136_136761


namespace min_n_satisfies_inequality_l136_136112

theorem min_n_satisfies_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ n * (x^4 + y^4 + z^4)) ∧ (n = 3) :=
by
  sorry

end min_n_satisfies_inequality_l136_136112


namespace ice_rink_rental_fee_l136_136186

/-!
  # Problem:
  An ice skating rink charges $5 for admission and a certain amount to rent skates. 
  Jill can purchase a new pair of skates for $65. She would need to go to the rink 26 times 
  to justify buying the skates rather than renting a pair. How much does the rink charge to rent skates?
-/

/-- Lean statement of the problem. --/
theorem ice_rink_rental_fee 
  (admission_fee : ℝ) (skates_cost : ℝ) (num_visits : ℕ)
  (total_buying_cost : ℝ) (total_renting_cost : ℝ)
  (rental_fee : ℝ) :
  admission_fee = 5 ∧
  skates_cost = 65 ∧
  num_visits = 26 ∧
  total_buying_cost = skates_cost + (admission_fee * num_visits) ∧
  total_renting_cost = (admission_fee + rental_fee) * num_visits ∧
  total_buying_cost = total_renting_cost →
  rental_fee = 2.50 :=
by
  intros h
  sorry

end ice_rink_rental_fee_l136_136186


namespace polynomial_factorization_l136_136860

theorem polynomial_factorization :
  ∀ (a b c : ℝ),
    a * (b - c) ^ 4 + b * (c - a) ^ 4 + c * (a - b) ^ 4 =
    (a - b) * (b - c) * (c - a) * (a + b + c) :=
  by
    intro a b c
    sorry

end polynomial_factorization_l136_136860


namespace opposite_of_neg_2023_l136_136495

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136495


namespace probability_interval_contains_q_l136_136339

theorem probability_interval_contains_q (P_C P_D : ℝ) (q : ℝ)
    (hC : P_C = 5 / 7) (hD : P_D = 3 / 4) :
    (5 / 28 ≤ q ∧ q ≤ 5 / 7) ↔ (max (P_C + P_D - 1) 0 ≤ q ∧ q ≤ min P_C P_D) :=
by
  sorry

end probability_interval_contains_q_l136_136339


namespace sum_of_fractions_eq_two_l136_136005

theorem sum_of_fractions_eq_two : 
  (1 / 2) + (2 / 4) + (4 / 8) + (8 / 16) = 2 :=
by sorry

end sum_of_fractions_eq_two_l136_136005


namespace triangles_with_positive_integer_area_count_l136_136917

theorem triangles_with_positive_integer_area_count :
  let points := { p : (ℕ × ℕ) // 41 * p.1 + p.2 = 2017 }
  ∃ count, count = 600 ∧ ∀ (P Q : points), P ≠ Q →
    let area := (P.val.1 * Q.val.2 - Q.val.1 * P.val.2 : ℤ)
    0 < area ∧ (area % 2 = 0) := sorry

end triangles_with_positive_integer_area_count_l136_136917


namespace paul_account_balance_l136_136430

variable (initial_balance : ℝ) (transfer1 : ℝ) (transfer2 : ℝ) (service_charge_rate : ℝ)

def final_balance (init_bal transfer1 transfer2 rate : ℝ) : ℝ :=
  let charge1 := transfer1 * rate
  let total_deduction := transfer1 + charge1
  init_bal - total_deduction

theorem paul_account_balance :
  initial_balance = 400 →
  transfer1 = 90 →
  transfer2 = 60 →
  service_charge_rate = 0.02 →
  final_balance 400 90 60 0.02 = 308.2 :=
by
  intros h1 h2 h3 h4
  rw [final_balance, h1, h2, h4]
  norm_num

end paul_account_balance_l136_136430


namespace opposite_of_neg_2023_l136_136476

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136476


namespace Malou_score_third_quiz_l136_136403

-- Defining the conditions as Lean definitions
def score1 : ℕ := 91
def score2 : ℕ := 92
def average : ℕ := 91
def num_quizzes : ℕ := 3

-- Proving that score3 equals 90
theorem Malou_score_third_quiz :
  ∃ score3 : ℕ, (score1 + score2 + score3) / num_quizzes = average ∧ score3 = 90 :=
by
  use (90 : ℕ)
  sorry

end Malou_score_third_quiz_l136_136403


namespace opposite_of_neg_2023_l136_136490

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136490


namespace divisor_inequality_l136_136108

-- Definition of our main inequality theorem
theorem divisor_inequality (n : ℕ) (h1 : n > 0) (h2 : n % 8 = 4)
    (divisors : List ℕ) (h3 : divisors = (List.range (n + 1)).filter (λ x => n % x = 0)) 
    (i : ℕ) (h4 : i < divisors.length - 1) (h5 : i % 3 ≠ 0) : 
    divisors[i + 1] ≤ 2 * divisors[i] := sorry

end divisor_inequality_l136_136108


namespace interest_rate_l136_136325

theorem interest_rate (SI P T R : ℝ) (h1 : SI = 100) (h2 : P = 500) (h3 : T = 4) (h4 : SI = (P * R * T) / 100) :
  R = 5 :=
by
  sorry

end interest_rate_l136_136325


namespace opposite_neg_2023_l136_136538

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136538


namespace number_of_dolls_of_jane_l136_136150

-- Given conditions
def total_dolls (J D : ℕ) := J + D = 32
def jill_has_more (J D : ℕ) := D = J + 6

-- Statement to prove
theorem number_of_dolls_of_jane (J D : ℕ) (h1 : total_dolls J D) (h2 : jill_has_more J D) : J = 13 :=
by
  sorry

end number_of_dolls_of_jane_l136_136150


namespace commute_distance_l136_136760

noncomputable def distance_to_work (total_time : ℕ) (speed_to_work : ℕ) (speed_to_home : ℕ) : ℕ :=
  let d := (speed_to_work * speed_to_home * total_time) / (speed_to_work + speed_to_home)
  d

-- Given conditions
def speed_to_work : ℕ := 45
def speed_to_home : ℕ := 30
def total_time : ℕ := 1

-- Proof problem statement
theorem commute_distance : distance_to_work total_time speed_to_work speed_to_home = 18 :=
by
  sorry

end commute_distance_l136_136760


namespace opposite_neg_2023_l136_136523

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136523


namespace solve_quadratic_equation_solve_linear_equation_l136_136128

-- Equation (1)
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 8 * x + 1 = 0 → (x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) :=
by
  sorry

-- Equation (2)
theorem solve_linear_equation :
  ∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x → (x = 1 ∨ x = -2/3) :=
by
  sorry

end solve_quadratic_equation_solve_linear_equation_l136_136128


namespace num_valid_subsets_is_2380_l136_136004

def S : Finset ℕ := Finset.range 20
def is_valid_subset (T : Finset ℕ) : Prop :=
  T.card = 4 ∧ ∀ (x y : ℕ), x ∈ T → y ∈ T → x ≠ y → |x - y| ≠ 1

theorem num_valid_subsets_is_2380 :
  (Finset.filter is_valid_subset (Finset.powersetLen 4 S)).card = 2380 := 
sorry

end num_valid_subsets_is_2380_l136_136004


namespace simplify_expression_l136_136852

theorem simplify_expression :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := 
by
  sorry

end simplify_expression_l136_136852


namespace opposite_of_negative_2023_l136_136705

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136705


namespace opposite_of_neg2023_l136_136474

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136474


namespace cost_price_perc_of_selling_price_l136_136278

theorem cost_price_perc_of_selling_price
  (SP : ℝ) (CP : ℝ) (P : ℝ)
  (h1 : P = SP - CP)
  (h2 : P = (4.166666666666666 / 100) * SP) :
  CP = SP * 0.9583333333333334 :=
by
  sorry

end cost_price_perc_of_selling_price_l136_136278


namespace symmetric_polynomial_evaluation_l136_136950

theorem symmetric_polynomial_evaluation :
  ∃ (a b : ℝ), (∀ x : ℝ, (x^2 + 3 * x) * (x^2 + a * x + b) = ((2 - x)^2 + 3 * (2 - x)) * ((2 - x)^2 + a * (2 - x) + b)) ∧
  ((3^2 + 3 * 3) * (3^2 + (-6) * 3 + 8) = -18) :=
sorry

end symmetric_polynomial_evaluation_l136_136950


namespace rectangle_same_color_l136_136049

theorem rectangle_same_color (colors : ℕ → ℕ → ℕ) (h_colors : ∀ x y, colors x y < 3) :
  ∃ (x1 x2 y1 y2 : ℕ), x1 < x2 ∧ y1 < y2 ∧ colors x1 y1 = colors x1 y2 ∧ colors x1 y1 = colors x2 y1 ∧ colors x1 y1 = colors x2 y2 := 
sorry

end rectangle_same_color_l136_136049


namespace quadratic_equation_with_given_means_l136_136245

theorem quadratic_equation_with_given_means (α β : ℝ)
  (h1 : (α + β) / 2 = 8) 
  (h2 : Real.sqrt (α * β) = 12) : 
  x ^ 2 - 16 * x + 144 = 0 :=
sorry

end quadratic_equation_with_given_means_l136_136245


namespace remaining_gift_card_value_correct_l136_136831

def initial_best_buy := 5
def initial_target := 3
def initial_walmart := 7
def initial_amazon := 2

def value_best_buy := 500
def value_target := 250
def value_walmart := 100
def value_amazon := 1000

def sent_best_buy := 1
def sent_walmart := 2
def sent_amazon := 1

def remaining_dollars : Nat :=
  (initial_best_buy - sent_best_buy) * value_best_buy +
  initial_target * value_target +
  (initial_walmart - sent_walmart) * value_walmart +
  (initial_amazon - sent_amazon) * value_amazon

theorem remaining_gift_card_value_correct : remaining_dollars = 4250 :=
  sorry

end remaining_gift_card_value_correct_l136_136831


namespace divisibility_condition_l136_136194

theorem divisibility_condition (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1) ∣ ((a + 1)^n) ↔ (a = 1 ∧ 1 ≤ m ∧ 1 ≤ n) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n) := 
by 
  sorry

end divisibility_condition_l136_136194


namespace opposite_of_neg_2023_l136_136504

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136504


namespace opposite_of_neg_2023_l136_136674

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136674


namespace largest_value_x_y_l136_136254

theorem largest_value_x_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 11 / 4 :=
sorry

end largest_value_x_y_l136_136254


namespace opposite_of_neg_2023_l136_136657

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136657


namespace ratio_of_perimeters_l136_136743

theorem ratio_of_perimeters (s d s' d': ℝ) (h1 : d = s * Real.sqrt 2) (h2 : d' = 2.5 * d) (h3 : d' = s' * Real.sqrt 2) : (4 * s') / (4 * s) = 5 / 2 :=
by
  -- Additional tactical details for completion, proof is omitted as per instructions
  sorry

end ratio_of_perimeters_l136_136743


namespace derivatives_at_zero_l136_136988

noncomputable def f : ℝ → ℝ := sorry

axiom diff_f : ∀ n : ℕ, f (1 / (n + 1)) = (n + 1)^2 / ((n + 1)^2 + 1)

theorem derivatives_at_zero :
  f 0 = 1 ∧ 
  deriv f 0 = 0 ∧ 
  deriv (deriv f) 0 = -2 ∧ 
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by
  sorry

end derivatives_at_zero_l136_136988


namespace find_positive_m_has_exactly_single_solution_l136_136943

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l136_136943


namespace area_expression_l136_136093

noncomputable def overlapping_area (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) : ℝ :=
if h : m ≤ 2 * Real.sqrt 2 then
  6 - Real.sqrt 2 * m
else
  (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8

theorem area_expression (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) :
  let y := overlapping_area m h1 h2
  (if h : m ≤ 2 * Real.sqrt 2 then y = 6 - Real.sqrt 2 * m
   else y = (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8) := 
sorry

end area_expression_l136_136093


namespace largest_element_sum_of_digits_in_E_l136_136042
open BigOperators
open Nat

def E : Set ℕ := { n | ∃ (r₉ r₁₀ r₁₁ : ℕ), 0 < r₉ ∧ r₉ ≤ 9 ∧ 0 < r₁₀ ∧ r₁₀ ≤ 10 ∧ 0 < r₁₁ ∧ r₁₁ ≤ 11 ∧
  r₉ = n % 9 ∧ r₁₀ = n % 10 ∧ r₁₁ = n % 11 ∧
  (r₉ > 1) ∧ (r₁₀ > 1) ∧ (r₁₁ > 1) ∧
  ∃ (a : ℕ) (b : ℕ) (c : ℕ), r₉ = a ∧ r₁₀ = a * b ∧ r₁₁ = a * b * c ∧ b ≠ 1 ∧ c ≠ 1 }

noncomputable def N : ℕ := 
  max (max (74 % 990) (134 % 990)) (526 % 990)

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem largest_element_sum_of_digits_in_E :
  sum_of_digits N = 13 :=
sorry

end largest_element_sum_of_digits_in_E_l136_136042


namespace opposite_of_neg_2023_l136_136623

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136623


namespace road_width_l136_136120

theorem road_width
  (road_length : ℝ) 
  (truckload_area : ℝ) 
  (truckload_cost : ℝ) 
  (sales_tax : ℝ) 
  (total_cost : ℝ) :
  road_length = 2000 ∧
  truckload_area = 800 ∧
  truckload_cost = 75 ∧
  sales_tax = 0.20 ∧
  total_cost = 4500 →
  ∃ width : ℝ, width = 20 :=
by
  sorry

end road_width_l136_136120


namespace jordan_trapezoid_height_l136_136912

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end jordan_trapezoid_height_l136_136912


namespace amit_work_days_l136_136183

theorem amit_work_days (x : ℕ) (h : 2 * (1 / x : ℚ) + 16 * (1 / 20 : ℚ) = 1) : x = 10 :=
by {
  sorry
}

end amit_work_days_l136_136183


namespace prove_a_is_perfect_square_l136_136388

-- Definition of a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Main theorem statement
theorem prove_a_is_perfect_square 
  (a b : ℕ) 
  (hb_odd : b % 2 = 1) 
  (h_integer : ∃ k : ℕ, ((a + b) * (a + b) + 4 * a) = k * a * b) :
  is_perfect_square a :=
sorry

end prove_a_is_perfect_square_l136_136388


namespace arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l136_136365

-- Statement for Question 1
theorem arrangement_ways_13_books : 
  (Nat.factorial 13) = 6227020800 := 
sorry

-- Statement for Question 2
theorem arrangement_ways_13_books_with_4_arithmetic_together :
  (Nat.factorial 10) * (Nat.factorial 4) = 87091200 := 
sorry

-- Statement for Question 3
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_together :
  (Nat.factorial 5) * (Nat.factorial 4) * (Nat.factorial 6) = 2073600 := 
sorry

-- Statement for Question 4
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together :
  (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 6) * (Nat.factorial 3) = 622080 := 
sorry

end arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l136_136365


namespace rook_arrangement_count_l136_136935

theorem rook_arrangement_count : 
  ∃ (count : ℕ), count = 576 ∧ 
  (∀ (rooks : Fin 8 → Fin 8 × Fin 8),
    (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) ∧
    (∀ k : Fin 8, (rooks k).1 % 2 = (rooks k).2 % 2 → False)) := 
begin
  use 576,
  sorry,
end

end rook_arrangement_count_l136_136935


namespace quadratic_root_condition_l136_136002

theorem quadratic_root_condition (b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + b*x1 + c = 0) ∧ (x2^2 + b*x2 + c = 0)) ↔ (b^2 - 4*c ≥ 0) :=
by
  sorry

end quadratic_root_condition_l136_136002


namespace cost_hour_excess_is_1point75_l136_136859

noncomputable def cost_per_hour_excess (x : ℝ) : Prop :=
  let total_hours := 9
  let initial_cost := 15
  let excess_hours := total_hours - 2
  let total_cost := initial_cost + excess_hours * x
  let average_cost_per_hour := 3.0277777777777777
  (total_cost / total_hours) = average_cost_per_hour

theorem cost_hour_excess_is_1point75 : cost_per_hour_excess 1.75 :=
by
  sorry

end cost_hour_excess_is_1point75_l136_136859


namespace cuboid_volume_l136_136134

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) : a * b * c = 6 := by
  sorry

end cuboid_volume_l136_136134


namespace part1_part2_part3_l136_136223

open Real

variables {a b x0 y0 x1 y1 x2 y2 : ℝ} 

-- Hypothesis definitions

def is_on_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def focal_length (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

def relation_op (x0 x1 x2 y0 y1 y2 : ℝ) : Prop :=
  3 * (x0, y0) = (x1, y1) + 2 * (x2, y2)

def l2_x_axis (a x0 : ℝ) : ℝ := a^2 / x0

theorem part1 (h_hyperbola : is_on_hyperbola x0 y0 a b)
    (h_focal_length : focal_length a b = 4 * sqrt 2)
    (h_relation : relation_op x0 x1 x2 y0 y1 y2) :
  x1 * x2 - y1 * y2 = 9 := 
  sorry

theorem part2 (h_hyperbola : is_on_hyperbola x0 y0 a b)
    (h_focal_length : focal_length a b = 4 * sqrt 2)
    (h_relation : relation_op x0 x1 x2 y0 y1 y2)
    {area_max : ℝ} :
  area_max = 9 / 2 ∧ 
  (4 * is_on_hyperbola a^2 / x0 0 a 4 - 1) = 4 := 
  sorry

theorem part3 (h1 : (0, -b^2 / y0))
    (h2 : (0, 8 * y0 / b^2))
    {fixed_point1 fixed_point2 : ℝ} :
  (fixed_point1, 0) = (0, 2 * sqrt 2) ∧
  (fixed_point2, 0) = (0, -2 * sqrt 2) := 
  sorry

end part1_part2_part3_l136_136223


namespace opposite_of_neg_2023_l136_136596

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136596


namespace sum_real_roots_eq_neg4_l136_136047

-- Define the equation condition
def equation_condition (x : ℝ) : Prop :=
  (2 * x / (x^2 + 5 * x + 3) + 3 * x / (x^2 + x + 3) = 1)

-- Define the statement that sums the real roots
theorem sum_real_roots_eq_neg4 : 
  ∃ S : ℝ, (∀ x : ℝ, equation_condition x → x = -1 ∨ x = -3) ∧ (S = -4) :=
sorry

end sum_real_roots_eq_neg4_l136_136047


namespace stickers_per_student_l136_136125

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end stickers_per_student_l136_136125


namespace problem_statement_l136_136802

variables {x y z w p q : Prop}

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q :=
by
  sorry

end problem_statement_l136_136802


namespace line_equation_l136_136283

-- Define the structure of a point
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the projection condition
def projection_condition (P : Point) (l : ℤ → ℤ → Prop) : Prop :=
  l P.x P.y ∧ ∀ (Q : Point), l Q.x Q.y → (Q.x ^ 2 + Q.y ^ 2) ≥ (P.x ^ 2 + P.y ^ 2)

-- Define the point P(-2, 1)
def P : Point := ⟨ -2, 1 ⟩

-- Define line l
def line_l (x y : ℤ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem line_equation :
  projection_condition P line_l → ∀ (x y : ℤ), line_l x y ↔ 2 * x - y + 5 = 0 :=
by
  sorry

end line_equation_l136_136283


namespace find_center_and_radius_find_lines_passing_through_P_find_trajectory_l136_136221

noncomputable def circle_center : ℝ × ℝ := (-2, 6)
noncomputable def circle_radius : ℝ := 4

def line1 : AffineLine ℝ ℝ := AffineLine.mk (3 : ℝ) (-4, 20)
def line2 : AffineLine ℝ ℝ := AffineLine.mk (0 : ℝ) -- x = 0 is a vertical line
def lines : List (AffineLine ℝ ℝ) := [line1, line2]

noncomputable def circle_equation (x y : ℝ) : Prop := 
  x^2 + y^2 + 4 * x - 12 * y + 24 = 0

noncomputable def trajectory_equation (x y : ℝ) : Prop := 
  x^2 + y^2 + 2 * x - 11 * y + 30 = 0

theorem find_center_and_radius :
  ∃ center radius, circle C center radius ∧ center = circle_center ∧ radius = circle_radius := sorry

theorem find_lines_passing_through_P :
  ∃ l ∈ lines, line_passes_through_point l (0, 5) ∧
    intercepted_segment_length l circle_center circle_radius = 4 * Real.sqrt 3 := sorry

theorem find_trajectory :
  ∃ trajectory, (∀ x y, trajectory x y ↔ trajectory_equation x y) := sorry

end find_center_and_radius_find_lines_passing_through_P_find_trajectory_l136_136221


namespace computation_result_l136_136080

theorem computation_result :
  let a := -6
  let b := 25
  let c := -39
  let d := 40
  9 * a + 3 * b + 6 * c + d = -173 := by
  sorry

end computation_result_l136_136080


namespace opposite_of_neg_2023_l136_136543

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136543


namespace savings_after_purchase_l136_136406

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l136_136406


namespace debate_team_group_size_l136_136866

theorem debate_team_group_size (boys girls groups : ℕ) (h_boys : boys = 11) (h_girls : girls = 45) (h_groups : groups = 8) : 
  (boys + girls) / groups = 7 := by
  sorry

end debate_team_group_size_l136_136866


namespace swimming_championship_l136_136376

theorem swimming_championship (num_swimmers : ℕ) (lanes : ℕ) (advance : ℕ) (eliminated : ℕ) (total_races : ℕ) : 
  num_swimmers = 300 → 
  lanes = 8 → 
  advance = 2 → 
  eliminated = 6 → 
  total_races = 53 :=
by
  intros
  sorry

end swimming_championship_l136_136376


namespace routes_from_M_to_N_l136_136010

structure Paths where
  -- Specify the paths between nodes
  C_to_N : ℕ
  D_to_N : ℕ
  A_to_C : ℕ
  A_to_D : ℕ
  B_to_N : ℕ
  B_to_A : ℕ
  B_to_C : ℕ
  M_to_B : ℕ
  M_to_A : ℕ

theorem routes_from_M_to_N (p : Paths) : 
  p.C_to_N = 1 → 
  p.D_to_N = 1 →
  p.A_to_C = 1 →
  p.A_to_D = 1 →
  p.B_to_N = 1 →
  p.B_to_A = 1 →
  p.B_to_C = 1 →
  p.M_to_B = 1 →
  p.M_to_A = 1 →
  (p.M_to_B * (p.B_to_N + (p.B_to_A * (p.A_to_C + p.A_to_D)) + p.B_to_C)) + 
  (p.M_to_A * (p.A_to_C + p.A_to_D)) = 6 
:= by
  sorry

end routes_from_M_to_N_l136_136010


namespace total_students_in_faculty_l136_136984

theorem total_students_in_faculty :
  (let sec_year_num := 230
   let sec_year_auto := 423
   let both_subj := 134
   let sec_year_total := 0.80
   let at_least_one_subj := sec_year_num + sec_year_auto - both_subj
   ∃ (T : ℝ), sec_year_total * T = at_least_one_subj ∧ T = 649) := by
  sorry

end total_students_in_faculty_l136_136984


namespace value_of_expression_l136_136812

theorem value_of_expression
  (m n : ℝ)
  (h1 : n = -2 * m + 3) :
  4 * m + 2 * n + 1 = 7 :=
sorry

end value_of_expression_l136_136812


namespace inscribed_sphere_radius_l136_136356

noncomputable def radius_inscribed_sphere (S1 S2 S3 S4 V : ℝ) : ℝ :=
  3 * V / (S1 + S2 + S3 + S4)

theorem inscribed_sphere_radius (S1 S2 S3 S4 V R : ℝ) :
  R = radius_inscribed_sphere S1 S2 S3 S4 V :=
by
  sorry

end inscribed_sphere_radius_l136_136356


namespace Peter_speed_is_correct_l136_136100

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end Peter_speed_is_correct_l136_136100


namespace inclination_angle_of_line_l136_136862

-- Definitions drawn from the condition.
def line_equation (x y : ℝ) := x - y + 1 = 0

-- The statement of the theorem (equivalent proof problem).
theorem inclination_angle_of_line : ∀ x y : ℝ, line_equation x y → θ = π / 4 :=
sorry

end inclination_angle_of_line_l136_136862


namespace opposite_of_neg_2023_l136_136561

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136561


namespace correct_factorization_l136_136735

theorem correct_factorization (a b : ℝ) : a^2 - 4 * a * b + 4 * b^2 = (a - 2 * b)^2 :=
by sorry

end correct_factorization_l136_136735


namespace product_of_numbers_l136_136289

theorem product_of_numbers (a b c m : ℚ) (h_sum : a + b + c = 240)
    (h_m_a : 6 * a = m) (h_m_b : m = b - 12) (h_m_c : m = c + 12) :
    a * b * c = 490108320 / 2197 :=
by 
  sorry

end product_of_numbers_l136_136289


namespace solve_for_x_l136_136143

theorem solve_for_x (x : ℝ) (h : 3 / (x + 2) = 2 / (x - 1)) : x = 7 :=
sorry

end solve_for_x_l136_136143


namespace opposite_of_neg_2023_l136_136689

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136689


namespace relationship_between_a_b_l136_136436

theorem relationship_between_a_b (a b x : ℝ) 
  (h₁ : x = (a + b) / 2)
  (h₂ : x^2 = (a^2 - b^2) / 2):
  a = -b ∨ a = 3 * b :=
sorry

end relationship_between_a_b_l136_136436


namespace milk_production_l136_136437

theorem milk_production 
  (initial_cows : ℕ)
  (initial_milk : ℕ)
  (initial_days : ℕ)
  (max_milk_per_cow_per_day : ℕ)
  (available_cows : ℕ)
  (days : ℕ)
  (H_initial : initial_cows = 10)
  (H_initial_milk : initial_milk = 40)
  (H_initial_days : initial_days = 5)
  (H_max_milk : max_milk_per_cow_per_day = 2)
  (H_available_cows : available_cows = 15)
  (H_days : days = 8) :
  available_cows * initial_milk / (initial_cows * initial_days) * days = 96 := 
by 
  sorry

end milk_production_l136_136437


namespace coordinates_with_respect_to_origin_l136_136824

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l136_136824


namespace rectangle_area_divisible_by_12_l136_136001

theorem rectangle_area_divisible_by_12
  (x y z : ℤ)
  (h : x^2 + y^2 = z^2) :
  12 ∣ (x * y) :=
sorry

end rectangle_area_divisible_by_12_l136_136001


namespace roots_square_sum_l136_136969

theorem roots_square_sum (r s p q : ℝ) 
  (root_cond : ∀ x : ℝ, x^2 - 2 * p * x + 3 * q = 0 → (x = r ∨ x = s)) :
  r^2 + s^2 = 4 * p^2 - 6 * q :=
by
  sorry

end roots_square_sum_l136_136969


namespace opposite_of_neg_2023_l136_136480

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136480


namespace opposite_of_neg_2023_l136_136547

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136547


namespace reduced_expression_none_of_these_l136_136127

theorem reduced_expression_none_of_these (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : b ≠ a^2) (h4 : ab ≠ a^3) :
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 1 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (b^2 + b) / (b - a^2) ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ 0 ∧
  ((a^2 - b^2) / ab + (ab + b^2) / (ab - a^3)) ≠ (a^2 + b) / (a^2 - b) :=
by
  sorry

end reduced_expression_none_of_these_l136_136127


namespace opposite_of_neg_2023_l136_136573

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136573


namespace part_I_part_II_l136_136359

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1

theorem part_I (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eccentricity : ℝ := c / a) (h3 : eccentricity = Real.sqrt 2 / 2) (vertex : ℝ × ℝ := (0, 1)) (h4 : vertex = (0, b)) 
  : ellipse_equation (Real.sqrt 2) 1 (0:ℝ) 1 :=
sorry

theorem part_II (a b k : ℝ) (x y : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 1)
  (line_eq : ℝ → ℝ := fun x => k * x + 1) 
  (h3 : (1 + 2 * k^2) * x^2 + 4 * k * x = 0) 
  (distance_AB : ℝ := Real.sqrt 2 * 4 / 3) 
  (h4 : Real.sqrt (1 + k^2) * abs ((-4 * k) / (2 * k^2 + 1)) = distance_AB) 
  : (x, y) = (4/3, -1/3) ∨ (x, y) = (-4/3, -1/3) :=
sorry

end part_I_part_II_l136_136359


namespace base_conversion_subtraction_l136_136051

theorem base_conversion_subtraction :
  (4 * 6^4 + 3 * 6^3 + 2 * 6^2 + 1 * 6^1 + 0 * 6^0) - (3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0) = 4776 :=
by {
  sorry
}

end base_conversion_subtraction_l136_136051


namespace opposite_of_neg_2023_l136_136690

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136690


namespace parallelogram_base_is_36_l136_136207

def parallelogram_base (area height : ℕ) : ℕ :=
  area / height

theorem parallelogram_base_is_36 (h : parallelogram_base 864 24 = 36) : True :=
by
  trivial

end parallelogram_base_is_36_l136_136207


namespace opposite_of_negative_2023_l136_136613

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136613


namespace area_of_L_shaped_figure_l136_136292

theorem area_of_L_shaped_figure :
  let large_rect_area := 10 * 7
  let small_rect_area := 4 * 3
  large_rect_area - small_rect_area = 58 := by
  sorry

end area_of_L_shaped_figure_l136_136292


namespace eccentricity_squared_l136_136801

-- Define the hyperbola and its properties
variables (a b c e : ℝ) (x₁ y₁ x₂ y₂ : ℝ)

-- Define the hyperbola equation and conditions
def hyperbola_eq (a b x y : ℝ) := (x^2)/(a^2) - (y^2)/(b^2) = 1

def midpoint_eq (x₁ y₁ x₂ y₂ : ℝ) := x₁ + x₂ = -4 ∧ y₁ + y₂ = 2

def slope_eq (a b c : ℝ) := -b / c = (b^2 * (-4)) / (a^2 * 2)

-- Define the proof
theorem eccentricity_squared :
  a > 0 → b > 0 → hyperbola_eq a b x₁ y₁ → hyperbola_eq a b x₂ y₂ → midpoint_eq x₁ y₁ x₂ y₂ →
  slope_eq a b c → c^2 = a^2 + b^2 → (e = c / a) → e^2 = (Real.sqrt 2 + 1) / 2 :=
by
  intro ha hb h1 h2 h3 h4 h5 he
  sorry

end eccentricity_squared_l136_136801


namespace solution_set_f_gt_x_l136_136069

noncomputable def f : ℝ → ℝ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_deriv_gt_one (x : ℝ) : deriv f x > 1

theorem solution_set_f_gt_x :
  { x : ℝ | f x > x } = set.Ioi 1 :=
begin
  sorry
end

end solution_set_f_gt_x_l136_136069


namespace marina_drive_l136_136996

theorem marina_drive (a b c : ℕ) (x : ℕ) 
  (h1 : 1 ≤ a) 
  (h2 : a + b + c ≤ 9)
  (h3 : 90 * (b - a) = 60 * x)
  (h4 : x = 3 * (b - a) / 2) :
  a = 1 ∧ b = 3 ∧ c = 5 ∧ a^2 + b^2 + c^2 = 35 :=
by {
  sorry
}

end marina_drive_l136_136996


namespace opposite_of_neg_2023_l136_136685

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136685


namespace king_and_queen_ages_l136_136442

variable (K Q : ℕ)

theorem king_and_queen_ages (h1 : K = 2 * (Q - (K - Q)))
                            (h2 : K + (K + (K - Q)) = 63) :
                            K = 28 ∧ Q = 21 := by
  sorry

end king_and_queen_ages_l136_136442


namespace area_of_regular_octagon_l136_136849

-- Define a regular octagon with given diagonals
structure RegularOctagon where
  d_max : ℝ  -- length of the longest diagonal
  d_min : ℝ  -- length of the shortest diagonal

-- Theorem stating that the area of the regular octagon
-- is the product of its longest and shortest diagonals
theorem area_of_regular_octagon (O : RegularOctagon) : 
  let A := O.d_max * O.d_min
  A = O.d_max * O.d_min :=
by
  -- Proof to be filled in
  sorry

end area_of_regular_octagon_l136_136849


namespace opposite_of_neg_2023_l136_136634

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136634


namespace cats_left_l136_136901

def initial_siamese_cats : ℕ := 12
def initial_house_cats : ℕ := 20
def cats_sold : ℕ := 20

theorem cats_left : (initial_siamese_cats + initial_house_cats - cats_sold) = 12 :=
by
sorry

end cats_left_l136_136901


namespace area_triangle_BQW_l136_136248

theorem area_triangle_BQW (ABCD : Rectangle) (AZ WC : ℝ) (AB : ℝ)
    (area_trapezoid_ZWCD : ℝ) :
    AZ = WC ∧ AZ = 6 ∧ AB = 12 ∧ area_trapezoid_ZWCD = 120 →
    (1/2) * ((120) - (1/2) * 6 * 12) = 42 :=
by
  intros
  sorry

end area_triangle_BQW_l136_136248


namespace paul_final_balance_l136_136429

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l136_136429


namespace opposite_of_neg_2023_l136_136631

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136631


namespace distance_to_nearest_lattice_point_l136_136756

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem distance_to_nearest_lattice_point :
  ∃ d : ℝ, (∃ (p : ℝ), p = 7 / 16 / 2560000 * 2500 * 2500 * 4 / π) ∧ 
  sqrt (7 / 16 / π) = 0.4 :=
by sorry

end distance_to_nearest_lattice_point_l136_136756


namespace speed_of_stream_l136_136169

theorem speed_of_stream (v : ℝ) : (13 + v) * 4 = 68 → v = 4 :=
by
  intro h
  sorry

end speed_of_stream_l136_136169


namespace cutting_wire_random_event_l136_136919

noncomputable def length : ℝ := sorry

def is_random_event (a : ℝ) : Prop := sorry

theorem cutting_wire_random_event (a : ℝ) (h : a > 0) :
  is_random_event a := 
by
  sorry

end cutting_wire_random_event_l136_136919


namespace minimum_value_proof_l136_136393

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l136_136393


namespace opposite_of_neg_2023_l136_136500

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136500


namespace branches_on_main_stem_l136_136772

theorem branches_on_main_stem (x : ℕ) (h : 1 + x + x^2 = 57) : x = 7 :=
  sorry

end branches_on_main_stem_l136_136772


namespace opposite_of_neg_2023_l136_136489

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136489


namespace find_lambda_l136_136231

noncomputable def vec_length (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

theorem find_lambda {a b : ℝ × ℝ} (lambda : ℝ) 
  (ha : vec_length a = 1) (hb : vec_length b = 2)
  (hab_angle : dot_product a b = -1) 
  (h_perp : dot_product (lambda • a + b) (a - 2 • b) = 0) : 
  lambda = 3 := 
sorry

end find_lambda_l136_136231


namespace opposite_of_neg_2023_l136_136457

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136457


namespace opposite_of_negative_2023_l136_136617

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136617


namespace savings_after_purchase_l136_136405

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l136_136405


namespace fireflies_win_l136_136308

theorem fireflies_win 
  (initial_hornets : ℕ) (initial_fireflies : ℕ) 
  (hornets_scored : ℕ) (fireflies_scored : ℕ) 
  (three_point_baskets : ℕ) (two_point_baskets : ℕ)
  (h1 : initial_hornets = 86)
  (h2 : initial_fireflies = 74)
  (h3 : three_point_baskets = 7)
  (h4 : two_point_baskets = 2)
  (h5 : fireflies_scored = three_point_baskets * 3)
  (h6 : hornets_scored = two_point_baskets * 2)
  : initial_fireflies + fireflies_scored - (initial_hornets + hornets_scored) = 5 := 
sorry

end fireflies_win_l136_136308


namespace opposite_of_neg_2023_l136_136505

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136505


namespace john_subtracts_79_l136_136729

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l136_136729


namespace remaining_distance_proof_l136_136418

-- Define the conditions
def pascal_current_speed : ℝ := 8
def pascal_reduced_speed : ℝ := pascal_current_speed - 4
def pascal_increased_speed : ℝ := pascal_current_speed * 1.5

-- Define the remaining distance in terms of the current speed and time taken
noncomputable def remaining_distance (T : ℝ) : ℝ := pascal_current_speed * T

-- Define the times with the increased and reduced speeds
noncomputable def time_with_increased_speed (T : ℝ) : ℝ := T - 16
noncomputable def time_with_reduced_speed (T : ℝ) : ℝ := T + 16

-- Define the distances using increased and reduced speeds
noncomputable def distance_increased_speed (T : ℝ) : ℝ := pascal_increased_speed * (time_with_increased_speed T)
noncomputable def distance_reduced_speed (T : ℝ) : ℝ := pascal_reduced_speed * (time_with_reduced_speed T)

-- Main theorem stating that the remaining distance is 256 miles
theorem remaining_distance_proof (T : ℝ) (ht_eq: pascal_current_speed * T = 256) : 
  remaining_distance T = 256 := by
  sorry

end remaining_distance_proof_l136_136418


namespace opposite_of_neg_2023_l136_136559

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136559


namespace opposite_neg_2023_l136_136533

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136533


namespace coordinates_with_respect_to_origin_l136_136826

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l136_136826


namespace Pascal_remaining_distance_l136_136425

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l136_136425


namespace percentage_decrease_hours_with_assistant_l136_136985

theorem percentage_decrease_hours_with_assistant :
  ∀ (B H H_new : ℝ), H_new = 0.9 * H → (H - H_new) / H * 100 = 10 :=
by
  intros B H H_new h_new_def
  sorry

end percentage_decrease_hours_with_assistant_l136_136985


namespace satisfying_lines_l136_136882

theorem satisfying_lines (x y : ℝ) : (y^2 - 2*y = x^2 + 2*x) ↔ (y = x + 2 ∨ y = -x) :=
by
  sorry

end satisfying_lines_l136_136882


namespace min_value_of_one_over_a_plus_one_over_b_l136_136396

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l136_136396


namespace evaluate_expression_l136_136929

theorem evaluate_expression : 3 - 5 * (6 - 2^3) / 2 = 8 :=
by
  sorry

end evaluate_expression_l136_136929


namespace opposite_of_neg2023_l136_136465

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136465


namespace copper_production_is_correct_l136_136766

-- Define the percentages of copper production for each mine
def percentage_copper_mine_a : ℝ := 0.05
def percentage_copper_mine_b : ℝ := 0.10
def percentage_copper_mine_c : ℝ := 0.15

-- Define the daily production of each mine in tons
def daily_production_mine_a : ℕ := 3000
def daily_production_mine_b : ℕ := 4000
def daily_production_mine_c : ℕ := 3500

-- Define the total copper produced from all mines
def total_copper_produced : ℝ :=
  percentage_copper_mine_a * daily_production_mine_a +
  percentage_copper_mine_b * daily_production_mine_b +
  percentage_copper_mine_c * daily_production_mine_c

-- Prove that the total daily copper production is 1075 tons
theorem copper_production_is_correct :
  total_copper_produced = 1075 := 
sorry

end copper_production_is_correct_l136_136766


namespace common_ratio_of_arithmetic_seq_l136_136357

theorem common_ratio_of_arithmetic_seq (a_1 q : ℝ) 
  (h1 : a_1 + a_1 * q^2 = 10) 
  (h2 : a_1 * q^3 + a_1 * q^5 = 5 / 4) : 
  q = 1 / 2 := 
by 
  sorry

end common_ratio_of_arithmetic_seq_l136_136357


namespace find_y_l136_136235

noncomputable def G (a b c d : ℝ) : ℝ := a ^ b + c ^ d

theorem find_y (h : G 3 y 2 5 = 100) : y = Real.log 68 / Real.log 3 := 
by
  have hG : G 3 y 2 5 = 3 ^ y + 2 ^ 5 := rfl
  sorry

end find_y_l136_136235


namespace tap_filling_time_l136_136896

theorem tap_filling_time
  (T : ℝ)
  (H1 : 10 > 0) -- Second tap can empty the cistern in 10 hours
  (H2 : T > 0)  -- First tap's time must be positive
  (H3 : (1 / T) - (1 / 10) = (3 / 20))  -- Both taps together fill the cistern in 6.666... hours
  : T = 4 := sorry

end tap_filling_time_l136_136896


namespace constant_term_eq_160_l136_136053

-- Define the binomial coefficients and the binomial theorem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term of (2x + 1/x)^6 expansion
def general_term_expansion (r : ℕ) : ℤ :=
  2^(6 - r) * binom 6 r

-- Define the proof statement for the required constant term
theorem constant_term_eq_160 : general_term_expansion 3 = 160 := 
by
  sorry

end constant_term_eq_160_l136_136053


namespace work_rate_l136_136737

theorem work_rate (x : ℝ) (h : (1 / x + 1 / 15 = 1 / 6)) : x = 10 :=
sorry

end work_rate_l136_136737


namespace opposite_of_neg_2023_l136_136686

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136686


namespace leaves_fall_total_l136_136817

theorem leaves_fall_total : 
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  actual_cherry_trees * leaves_per_cherry_tree + actual_maple_trees * leaves_per_maple_tree = 3650 :=
by
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  sorry

end leaves_fall_total_l136_136817


namespace ornamental_rings_remaining_l136_136926

-- Definitions based on conditions
variable (initial_stock : ℕ) (final_stock : ℕ)

-- Condition 1
def condition1 := initial_stock + 200 = 3 * initial_stock

-- Condition 2
def condition2 := final_stock = (200 + initial_stock) * 1 / 4 - (200 + initial_stock) / 4 + 300 - 150

-- Theorem statement to prove the final stock is 225
theorem ornamental_rings_remaining
  (h1 : condition1 initial_stock)
  (h2 : condition2 initial_stock final_stock) :
  final_stock = 225 :=
sorry

end ornamental_rings_remaining_l136_136926


namespace books_on_shelf_l136_136180

theorem books_on_shelf (total_books : ℕ) (sold_books : ℕ) (shelves : ℕ) (remaining_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 27 → sold_books = 6 → shelves = 3 → remaining_books = total_books - sold_books → books_per_shelf = remaining_books / shelves → books_per_shelf = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_on_shelf_l136_136180


namespace magnitude_of_z_l136_136401

open Complex

theorem magnitude_of_z (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 7 * Complex.I) : 
  Complex.normSq z = 65 / 8 := 
by
  sorry

end magnitude_of_z_l136_136401


namespace tank_capacity_l136_136327

theorem tank_capacity :
  let rateA := 40  -- Pipe A fills at 40 liters per minute
  let rateB := 30  -- Pipe B fills at 30 liters per minute
  let rateC := -20  -- Pipe C (drains) at 20 liters per minute, thus negative contribution
  let cycle_duration := 3  -- The cycle duration is 3 minutes
  let total_duration := 51  -- The tank gets full in 51 minutes
  let net_per_cycle := rateA + rateB + rateC  -- Net fill per cycle of 3 minutes
  let num_cycles := total_duration / cycle_duration  -- Number of complete cycles
  let tank_capacity := net_per_cycle * num_cycles  -- Tank capacity in liters
  tank_capacity = 850  -- Assertion that needs to be proven
:= by
  let rateA := 40
  let rateB := 30
  let rateC := -20
  let cycle_duration := 3
  let total_duration := 51
  let net_per_cycle := rateA + rateB + rateC
  let num_cycles := total_duration / cycle_duration
  let tank_capacity := net_per_cycle * num_cycles
  have : tank_capacity = 850 := by
    sorry
  assumption

end tank_capacity_l136_136327


namespace maximize_side_area_of_cylinder_l136_136750

noncomputable def radius_of_cylinder (x : ℝ) : ℝ :=
  (6 - x) / 3

noncomputable def side_area_of_cylinder (x : ℝ) : ℝ :=
  2 * Real.pi * (radius_of_cylinder x) * x

theorem maximize_side_area_of_cylinder :
  ∃ x : ℝ, (0 < x ∧ x < 6) ∧ (∀ y : ℝ, (0 < y ∧ y < 6) → (side_area_of_cylinder y ≤ side_area_of_cylinder x)) ∧ x = 3 :=
by
  sorry

end maximize_side_area_of_cylinder_l136_136750


namespace opposite_of_neg_2023_l136_136557

theorem opposite_of_neg_2023 : ∃ x : ℤ, -2023 + x = 0 ∧ x = 2023 := 
by 
  use 2023
  constructor
  · exact rfl
  · exact rfl

end opposite_of_neg_2023_l136_136557


namespace opposite_of_neg_2023_l136_136651

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136651


namespace opposite_of_neg_2023_l136_136650

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136650


namespace find_matrix_A_l136_136217

theorem find_matrix_A (a b c d : ℝ) 
  (h1 : a - 3 * b = -1)
  (h2 : c - 3 * d = 3)
  (h3 : a + b = 3)
  (h4 : c + d = 3) :
  a = 2 ∧ b = 1 ∧ c = 3 ∧ d = 0 := by
  sorry

end find_matrix_A_l136_136217


namespace ferry_heading_to_cross_perpendicularly_l136_136188

theorem ferry_heading_to_cross_perpendicularly (river_speed ferry_speed : ℝ) (river_speed_val : river_speed = 12.5) (ferry_speed_val : ferry_speed = 25) : 
  angle_to_cross = 30 :=
by
  -- Definitions for the problem
  let river_velocity : ℝ := river_speed
  let ferry_velocity : ℝ := ferry_speed
  have river_velocity_def : river_velocity = 12.5 := river_speed_val
  have ferry_velocity_def : ferry_velocity = 25 := ferry_speed_val
  -- The actual proof would go here
  sorry

end ferry_heading_to_cross_perpendicularly_l136_136188


namespace simplify_and_evaluate_expression_l136_136272

theorem simplify_and_evaluate_expression (x : ℝ) (h : x^2 - 2 * x - 2 = 0) :
    ( ( (x - 1)/x - (x - 2)/(x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) = 1 / 2 ) :=
by
    -- sorry to skip the proof
    sorry

end simplify_and_evaluate_expression_l136_136272


namespace opposite_of_neg_2023_l136_136645

-- Let x be the opposite of -2023
def is_opposite (x : ℤ) (n : ℤ) : Prop := n + x = 0

theorem opposite_of_neg_2023 : ∃ x : ℤ, is_opposite x (-2023) ∧ x = 2023 :=
by
  exists 2023
  unfold is_opposite
  sorry

end opposite_of_neg_2023_l136_136645


namespace opposite_of_neg_2023_l136_136688

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l136_136688


namespace weekly_allowance_l136_136963

theorem weekly_allowance (A : ℝ) (H1 : A - (3/5) * A = (2/5) * A)
(H2 : (2/5) * A - (1/3) * ((2/5) * A) = (4/15) * A)
(H3 : (4/15) * A = 0.96) : A = 3.6 := 
sorry

end weekly_allowance_l136_136963


namespace xiaoming_minimum_time_l136_136310

theorem xiaoming_minimum_time :
  let review_time := 30
  let rest_time := 30
  let boil_time := 15
  let homework_time := 25
  (boil_time ≤ rest_time) → 
  (review_time + rest_time + homework_time = 85) :=
by
  intros review_time rest_time boil_time homework_time h_boil_le_rest
  sorry

end xiaoming_minimum_time_l136_136310


namespace sheryll_paid_total_l136_136974

-- Variables/conditions
variables (cost_per_book : ℝ) (num_books : ℕ) (discount_per_book : ℝ)

-- Given conditions
def assumption1 : cost_per_book = 5 := by sorry
def assumption2 : num_books = 10 := by sorry
def assumption3 : discount_per_book = 0.5 := by sorry

-- Theorem statement
theorem sheryll_paid_total : cost_per_book = 5 → num_books = 10 → discount_per_book = 0.5 → 
  (cost_per_book - discount_per_book) * num_books = 45 := by
  sorry

end sheryll_paid_total_l136_136974


namespace opposite_of_negative_2023_l136_136702

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136702


namespace percent_paddyfield_warblers_l136_136086

variable (B : ℝ) -- The total number of birds.
variable (N_h : ℝ := 0.30 * B) -- Number of hawks.
variable (N_non_hawks : ℝ := 0.70 * B) -- Number of non-hawks.
variable (N_not_hpwk : ℝ := 0.35 * B) -- 35% are not hawks, paddyfield-warblers, or kingfishers.
variable (N_hpwk : ℝ := 0.65 * B) -- 65% are hawks, paddyfield-warblers, or kingfishers.
variable (P : ℝ) -- Percentage of non-hawks that are paddyfield-warblers, to be found.
variable (N_pw : ℝ := P * 0.70 * B) -- Number of paddyfield-warblers.
variable (N_k : ℝ := 0.25 * N_pw) -- Number of kingfishers.

theorem percent_paddyfield_warblers (h_eq : N_h + N_pw + N_k = 0.65 * B) : P = 0.5714 := by
  sorry

end percent_paddyfield_warblers_l136_136086


namespace kira_breakfast_time_l136_136105

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end kira_breakfast_time_l136_136105


namespace geometric_sequence_term_eq_l136_136978

theorem geometric_sequence_term_eq (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 / 2 → q = 1 / 2 → a₁ * q ^ (n - 1) = 1 / 32 → n = 5 :=
by
  intros ha₁ hq han
  sorry

end geometric_sequence_term_eq_l136_136978


namespace range_of_a_l136_136796

def A (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + a * x - y + 2 = 0}
def B : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x - y + 1 = 0 ∧ x > 0}

theorem range_of_a (a : ℝ) : (∃ p, p ∈ A a ∧ p ∈ B) ↔ a ∈ Set.Iic 0 := by
  sorry

end range_of_a_l136_136796


namespace find_c_l136_136834

def p (x : ℝ) := 4 * x - 9
def q (x : ℝ) (c : ℝ) := 5 * x - c

theorem find_c : ∃ (c : ℝ), p (q 3 c) = 14 ∧ c = 9.25 :=
by
  sorry

end find_c_l136_136834


namespace problem_a_l136_136884

theorem problem_a (k l m : ℝ) : 
  (k + l + m) ^ 2 >= 3 * (k * l + l * m + m * k) :=
by sorry

end problem_a_l136_136884


namespace magnitude_of_complex_l136_136200

noncomputable def z : ℂ := (2 / 3 : ℝ) - (4 / 5 : ℝ) * Complex.I

theorem magnitude_of_complex :
  Complex.abs z = (2 * Real.sqrt 61) / 15 :=
by
  sorry

end magnitude_of_complex_l136_136200


namespace find_positive_m_l136_136937

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l136_136937


namespace Nellie_legos_l136_136843

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end Nellie_legos_l136_136843


namespace dealers_profit_percentage_l136_136174

theorem dealers_profit_percentage 
  (articles_purchased : ℕ)
  (total_cost_price : ℝ)
  (articles_sold : ℕ)
  (total_selling_price : ℝ)
  (CP_per_article : ℝ := total_cost_price / articles_purchased)
  (SP_per_article : ℝ := total_selling_price / articles_sold)
  (profit_per_article : ℝ := SP_per_article - CP_per_article)
  (profit_percentage : ℝ := (profit_per_article / CP_per_article) * 100) :
  articles_purchased = 15 →
  total_cost_price = 25 →
  articles_sold = 12 →
  total_selling_price = 32 →
  profit_percentage = 60 :=
by
  intros h1 h2 h3 h4
  sorry

end dealers_profit_percentage_l136_136174


namespace digit_B_divisibility_l136_136280

theorem digit_B_divisibility (B : ℕ) (h : 4 * 1000 + B * 100 + B * 10 + 6 % 11 = 0) : B = 5 :=
sorry

end digit_B_divisibility_l136_136280


namespace opposite_of_neg_2023_l136_136662

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136662


namespace triangle_perimeter_l136_136031

theorem triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 6)
  (c1 c2 : ℝ) (h3 : (c1 - 2) * (c1 - 4) = 0) (h4 : (c2 - 2) * (c2 - 4) = 0) :
  c1 = 2 ∨ c1 = 4 → c2 = 2 ∨ c2 = 4 → 
  (c1 ≠ 2 ∧ c1 = 4 ∨ c2 ≠ 2 ∧ c2 = 4) → 
  (a + b + c1 = 13 ∨ a + b + c2 = 13) :=
by
  sorry

end triangle_perimeter_l136_136031


namespace greatest_constant_triangle_l136_136208

theorem greatest_constant_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  ∃ N : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → c + a > b → (a^2 + b^2 + a * b) / c^2 > N) ∧ N = 3 / 4 :=
  sorry

end greatest_constant_triangle_l136_136208


namespace broken_pieces_correct_l136_136132

variable (pieces_transported : ℕ)
variable (shipping_cost_per_piece : ℝ)
variable (compensation_per_broken_piece : ℝ)
variable (total_profit : ℝ)
variable (broken_pieces : ℕ)

def logistics_profit (pieces_transported : ℕ) (shipping_cost_per_piece : ℝ) 
                     (compensation_per_broken_piece : ℝ) (broken_pieces : ℕ) : ℝ :=
  shipping_cost_per_piece * (pieces_transported - broken_pieces) - compensation_per_broken_piece * broken_pieces

theorem broken_pieces_correct :
  pieces_transported = 2000 →
  shipping_cost_per_piece = 0.2 →
  compensation_per_broken_piece = 2.3 →
  total_profit = 390 →
  logistics_profit pieces_transported shipping_cost_per_piece compensation_per_broken_piece broken_pieces = total_profit →
  broken_pieces = 4 :=
by
  intros
  sorry

end broken_pieces_correct_l136_136132


namespace valid_votes_for_candidate_a_l136_136739

theorem valid_votes_for_candidate_a (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_a_percentage : ℝ) (valid_votes_a : ℝ) :
  total_votes = 560000 ∧ invalid_percentage = 0.15 ∧ candidate_a_percentage = 0.80 →
  valid_votes_a = (candidate_a_percentage * (1 - invalid_percentage) * total_votes) := 
sorry

end valid_votes_for_candidate_a_l136_136739


namespace probability_of_one_in_pascals_triangle_first_20_rows_l136_136331

noncomputable def number_of_elements (n : ℕ) : ℕ := n * (n + 1) / 2
noncomputable def ones_in_rows (n : ℕ) : ℕ := if n = 0 then 1 else 2 * (n - 1) + 1

theorem probability_of_one_in_pascals_triangle_first_20_rows : 
  let total_elements := number_of_elements 20
  let total_ones := ones_in_rows 20
  in total_ones / total_elements = 13 / 70 :=
by
  sorry

end probability_of_one_in_pascals_triangle_first_20_rows_l136_136331


namespace correct_conclusion_l136_136214

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^3 - 6*x^2 + 9*x - a*b*c

-- The statement to be proven, without providing the actual proof.
theorem correct_conclusion 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : f a a b c = 0) 
  (h4 : f b a b c = 0) 
  (h5 : f c a b c = 0) :
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
sorry

end correct_conclusion_l136_136214


namespace opposite_of_negative_2023_l136_136606

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136606


namespace find_n_l136_136346

theorem find_n (n : ℕ) :
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) ≥ 1991) ∧ (∃ k : ℕ, k > 0 ∧ k^2 + (n / k^2) < 1992) ↔ 967 * 1024 ≤ n ∧ n < 968 * 1024 :=
by
  sorry

end find_n_l136_136346


namespace probability_Xavier_Yvonne_not_Zelda_l136_136309

-- Define the probabilities of success for Xavier, Yvonne, and Zelda
def pXavier := 1 / 5
def pYvonne := 1 / 2
def pZelda := 5 / 8

-- Define the probability that Zelda does not solve the problem
def pNotZelda := 1 - pZelda

-- The desired probability that we want to prove equals 3/80
def desiredProbability := (pXavier * pYvonne * pNotZelda) = (3 / 80)

-- The statement of the problem in Lean
theorem probability_Xavier_Yvonne_not_Zelda :
  desiredProbability := by
  sorry

end probability_Xavier_Yvonne_not_Zelda_l136_136309


namespace opposite_of_neg_2023_l136_136590

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136590


namespace number_of_boys_l136_136338

theorem number_of_boys
  (x y : ℕ) 
  (h1 : x + y = 43)
  (h2 : 24 * x + 27 * y = 1101) : 
  x = 20 := by
  sorry

end number_of_boys_l136_136338


namespace Janet_pages_per_day_l136_136810

variable (J : ℕ)

-- Conditions
def belinda_pages_per_day : ℕ := 30
def janet_extra_pages_per_6_weeks : ℕ := 2100
def days_in_6_weeks : ℕ := 42

-- Prove that Janet reads 80 pages a day
theorem Janet_pages_per_day (h : J * days_in_6_weeks = (belinda_pages_per_day * days_in_6_weeks) + janet_extra_pages_per_6_weeks) : J = 80 := 
by sorry

end Janet_pages_per_day_l136_136810


namespace base5_to_base4_last_digit_l136_136918

theorem base5_to_base4_last_digit (n : ℕ) (h : n = 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4) : (n % 4 = 2) :=
by sorry

end base5_to_base4_last_digit_l136_136918


namespace find_a_l136_136814

theorem find_a
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 2 * (b * Real.cos A + a * Real.cos B) = c^2)
  (h2 : b = 3)
  (h3 : 3 * Real.cos A = 1) :
  a = 3 :=
sorry

end find_a_l136_136814


namespace count_implications_l136_136043

theorem count_implications (p q r : Prop) :
  ((p ∧ q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (¬p ∧ ¬q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (p ∧ ¬q ∧ r → ¬ ((q → p) → ¬r)) ∧ 
   (¬p ∧ q ∧ ¬r → ((q → p) → ¬r))) →
   (3 = 3) := sorry

end count_implications_l136_136043


namespace opposite_of_neg_2023_l136_136454

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136454


namespace opposite_of_neg_2023_l136_136587

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l136_136587


namespace opposite_of_neg_2023_l136_136630

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136630


namespace mgp_inequality_l136_136992

theorem mgp_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (1 / Real.sqrt (1 / 2 + a + a * b + a * b * c) + 
   1 / Real.sqrt (1 / 2 + b + b * c + b * c * d) + 
   1 / Real.sqrt (1 / 2 + c + c * d + c * d * a) + 
   1 / Real.sqrt (1 / 2 + d + d * a + d * a * b)) 
  ≥ Real.sqrt 2 := 
sorry

end mgp_inequality_l136_136992


namespace total_time_naomi_30webs_l136_136333

-- Define the constants based on the given conditions
def time_katherine : ℕ := 20
def factor_naomi : ℚ := 5/4
def websites : ℕ := 30

-- Define the time taken by Naomi to build one website based on the conditions
def time_naomi (time_katherine : ℕ) (factor_naomi : ℚ) : ℚ :=
  factor_naomi * time_katherine

-- Define the total time Naomi took to build all websites
def total_time_naomi (time_naomi : ℚ) (websites : ℕ) : ℚ :=
  time_naomi * websites

-- Statement: Proving that the total number of hours Naomi took to create 30 websites is 750
theorem total_time_naomi_30webs : 
  total_time_naomi (time_naomi time_katherine factor_naomi) websites = 750 := 
sorry

end total_time_naomi_30webs_l136_136333


namespace find_natural_number_l136_136874

theorem find_natural_number :
  ∃ x : ℕ, (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → d2 - d1 = 4) ∧
           (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → x - d2 = 308) ∧
           x = 385 :=
by
  sorry

end find_natural_number_l136_136874


namespace opposite_of_negative_2023_l136_136712

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l136_136712


namespace number_of_special_three_digit_numbers_l136_136965

theorem number_of_special_three_digit_numbers : ∃ (n : ℕ), n = 3 ∧
  (∀ (A B C : ℕ), 
    (100 * A + 10 * B + C < 1000 ∧ 100 * A + 10 * B + C ≥ 100) ∧
    B = 2 * C ∧
    B = (A + C) / 2 → 
    (A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 312 ∨ 
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 642 ∨
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 963))
:= 
sorry

end number_of_special_three_digit_numbers_l136_136965


namespace olivia_time_spent_l136_136263

theorem olivia_time_spent :
  ∀ (x : ℕ), 7 * x + 3 = 31 → x = 4 :=
by
  intro x
  intro h
  sorry

end olivia_time_spent_l136_136263


namespace pen_cost_l136_136179

variable (p i : ℝ)

theorem pen_cost (h1 : p + i = 1.10) (h2 : p = 1 + i) : p = 1.05 :=
by 
  -- proof steps here
  sorry

end pen_cost_l136_136179


namespace find_value_of_A_l136_136281

theorem find_value_of_A (M T A E : ℕ) (H : ℕ := 8) 
  (h1 : M + A + T + H = 28) 
  (h2 : T + E + A + M = 34) 
  (h3 : M + E + E + T = 30) : 
  A = 16 :=
by 
  sorry

end find_value_of_A_l136_136281


namespace Pascal_remaining_distance_l136_136424

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l136_136424


namespace opposite_of_neg_2023_l136_136548

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136548


namespace boxes_used_l136_136320

-- Define the given conditions
def oranges_per_box : ℕ := 10
def total_oranges : ℕ := 2650

-- Define the proof statement
theorem boxes_used : total_oranges / oranges_per_box = 265 :=
by
  -- Proof goes here
  sorry

end boxes_used_l136_136320


namespace coordinates_with_respect_to_origin_l136_136822

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l136_136822


namespace opposite_of_neg_2023_l136_136577

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136577


namespace opposite_of_neg_2023_l136_136627

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136627


namespace number_of_terms_geometric_seq_l136_136003

-- Given conditions
variables (a1 q : ℝ)  -- First term and common ratio of the sequence
variable  (n : ℕ)     -- Number of terms in the sequence

-- The product of the first three terms
axiom condition1 : a1^3 * q^3 = 3

-- The product of the last three terms
axiom condition2 : a1^3 * q^(3 * n - 6) = 9

-- The product of all terms
axiom condition3 : a1^n * q^(n * (n - 1) / 2) = 729

-- Proving the number of terms in the sequence
theorem number_of_terms_geometric_seq : n = 12 := by
  sorry

end number_of_terms_geometric_seq_l136_136003


namespace julians_girls_percentage_l136_136832

theorem julians_girls_percentage
  (julian_friends : ℕ)
  (julian_boys_percentage : ℚ)
  (boyd_friends : ℕ)
  (boyd_girls_multiple : ℕ)
  (boyd_boys_percentage : ℚ)
  (h_julian_friends : julian_friends = 80)
  (h_julian_boys_percentage : julian_boys_percentage = 0.60)
  (h_boyd_friends : boyd_friends = 100)
  (h_boyd_girls_multiple : boyd_girls_multiple = 2)
  (h_boyd_boys_percentage : boyd_boys_percentage = 0.36)
  : ((julian_friends * (1 - julian_boys_percentage)) / julian_friends) * 100 = 40 := by
    sorry

end julians_girls_percentage_l136_136832


namespace opposite_of_negative_2023_l136_136604

theorem opposite_of_negative_2023 : (- -2023) = 2023 := 
by {
sorry
}

end opposite_of_negative_2023_l136_136604


namespace div_decimals_l136_136914

theorem div_decimals : 0.45 / 0.005 = 90 := sorry

end div_decimals_l136_136914


namespace opposite_of_neg_2023_l136_136492

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l136_136492


namespace opposite_of_neg_2023_l136_136514

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136514


namespace multiple_of_distance_l136_136850

namespace WalkProof

variable (H R M : ℕ)

/-- Rajesh walked 10 kilometers less than a certain multiple of the distance that Hiro walked. 
    Together they walked 25 kilometers. Rajesh walked 18 kilometers. 
    Prove that the multiple of the distance Hiro walked that Rajesh walked less than is 4. -/
theorem multiple_of_distance (h1 : R = M * H - 10) 
                             (h2 : H + R = 25)
                             (h3 : R = 18) :
                             M = 4 :=
by
  sorry

end WalkProof

end multiple_of_distance_l136_136850


namespace probability_properties_l136_136012

noncomputable def P1 : ℝ := 1 / 4
noncomputable def P2 : ℝ := 1 / 4
noncomputable def P3 : ℝ := 1 / 2

theorem probability_properties :
  (P1 ≠ P3) ∧
  (P1 + P2 = P3) ∧
  (P1 + P2 + P3 = 1) ∧
  (P3 = 2 * P1) ∧
  (P3 = 2 * P2) :=
by
  sorry

end probability_properties_l136_136012


namespace opposite_of_neg_2023_l136_136488

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136488


namespace total_distance_traveled_by_children_l136_136050

theorem total_distance_traveled_by_children :
  let ap := 50
  let dist_1_vertex_skip := (50 : ℝ) * Real.sqrt 2
  let dist_2_vertices_skip := (50 : ℝ) * Real.sqrt (2 + 2 * Real.sqrt 2)
  let dist_diameter := (2 : ℝ) * 50
  let single_child_distance := 2 * dist_1_vertex_skip + 2 * dist_2_vertices_skip + dist_diameter
  8 * single_child_distance = 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + 2 * Real.sqrt 2) + 800 :=
sorry

end total_distance_traveled_by_children_l136_136050


namespace opposite_of_neg_2023_l136_136479

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l136_136479


namespace coordinates_of_point_l136_136819

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l136_136819


namespace opposite_of_neg2023_l136_136471

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136471


namespace maria_savings_l136_136410

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l136_136410


namespace water_on_wednesday_l136_136806

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_l136_136806


namespace parabola_equation_l136_136136

-- Defining the point F and the line
def F : ℝ × ℝ := (0, 4)

def line_eq (y : ℝ) : Prop := y = -5

-- Defining the condition that point M is closer to F(0, 4) than to the line y = -5 by less than 1
def condition (M : ℝ × ℝ) : Prop :=
  let dist_to_F := (M.1 - F.1)^2 + (M.2 - F.2)^2
  let dist_to_line := abs (M.2 - (-5))
  abs (dist_to_F - dist_to_line) < 1

-- The equation we need to prove under the given condition
theorem parabola_equation (M : ℝ × ℝ) (h : condition M) : M.1^2 = 16 * M.2 := 
sorry

end parabola_equation_l136_136136


namespace count_whole_numbers_between_cuberoots_l136_136966

theorem count_whole_numbers_between_cuberoots : 
  ∃ (n : ℕ), n = 7 ∧ 
      ∀ x : ℝ, (3 < x ∧ x < 4 → ∃ k : ℕ, k = 4) ∧ 
                (9 < x ∧ x ≤ 10 → ∃ k : ℕ, k = 10) :=
sorry

end count_whole_numbers_between_cuberoots_l136_136966


namespace union_of_sets_l136_136260

def setA := {x : ℝ | x^2 < 4}
def setB := {y : ℝ | ∃ x ∈ setA, y = x^2 - 2 * x - 1}

theorem union_of_sets : (setA ∪ setB) = {x : ℝ | -2 ≤ x ∧ x < 7} :=
by sorry

end union_of_sets_l136_136260


namespace initial_salt_percentage_l136_136730

theorem initial_salt_percentage (initial_mass : ℝ) (added_salt_mass : ℝ) (final_solution_percentage : ℝ) (final_mass : ℝ) 
  (h1 : initial_mass = 100) 
  (h2 : added_salt_mass = 38.46153846153846) 
  (h3 : final_solution_percentage = 0.35) 
  (h4 : final_mass = 138.46153846153846) : 
  ((10 / 100) * 100) = 10 := 
sorry

end initial_salt_percentage_l136_136730


namespace road_length_in_km_l136_136844

/-- The actual length of the road in kilometers is 7.5, given the scale of 1:50000 
    and the map length of 15 cm. -/

theorem road_length_in_km (s : ℕ) (map_length_cm : ℕ) (actual_length_cm : ℕ) (actual_length_km : ℝ) 
  (h_scale : s = 50000) (h_map_length : map_length_cm = 15) (h_conversion : actual_length_km = actual_length_cm / 100000) :
  actual_length_km = 7.5 :=
  sorry

end road_length_in_km_l136_136844


namespace opposite_neg_2023_l136_136526

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136526


namespace john_subtraction_number_l136_136726

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l136_136726


namespace locate_z_in_fourth_quadrant_l136_136212

def z_in_quadrant_fourth (z : ℂ) : Prop :=
  (z.re > 0) ∧ (z.im < 0)

theorem locate_z_in_fourth_quadrant (z : ℂ) (i : ℂ) (h : i * i = -1) 
(hz : z * (1 + i) = 1) : z_in_quadrant_fourth z :=
sorry

end locate_z_in_fourth_quadrant_l136_136212


namespace probability_at_least_one_task_expectation_of_X_l136_136977

open ProbabilityTheory

def pA : ℚ := 3 / 4
def pB : ℚ := 3 / 4
def pC : ℚ := 2 / 3

-- Define the event that Jia completes at least one task
def P_at_least_one : ℚ :=
  1 - ((1 - pA) * (1 - pB) * (1 - pC))

theorem probability_at_least_one_task : P_at_least_one = 47 / 48 := by
  sorry

-- Define the random variable for the points earned
def X : Fin₄ → ℚ
| Fin₄.fz := 0
| Fin₄.fin1 := 1
| Fin₄.fin2 := 3
| Fin₄.fin3 := 6

-- Define the probabilities for the points
def pX (x : ℚ) : ℚ :=
  match x with
  | 0   => 7 / 16
  | 1   => 63 / 256
  | 3   => 21 / 256
  | 6   => 15 / 64
  | _   => 0

-- Calculate the expected value of X
def E_X : ℚ :=
  ∑ i, (X i) * (pX (X i))

theorem expectation_of_X : E_X = 243 / 128 := by
  sorry

end probability_at_least_one_task_expectation_of_X_l136_136977


namespace distance_from_tangency_to_tangent_l136_136875

theorem distance_from_tangency_to_tangent 
  (R r : ℝ)
  (hR : R = 3)
  (hr : r = 1)
  (externally_tangent : true) :
  ∃ d : ℝ, (d = 0 ∨ d = 7/3) :=
by
  sorry

end distance_from_tangency_to_tangent_l136_136875


namespace opposite_of_neg_2023_l136_136581

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136581


namespace remainder_of_sum_of_first_150_numbers_l136_136300

def sum_of_first_n_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem remainder_of_sum_of_first_150_numbers :
  (sum_of_first_n_natural_numbers 150) % 5000 = 1275 :=
by
  sorry

end remainder_of_sum_of_first_150_numbers_l136_136300


namespace calc_difference_of_squares_l136_136038

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end calc_difference_of_squares_l136_136038


namespace find_m_l136_136916

namespace MathProof

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

-- State the problem
theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -15 := by
  sorry

end MathProof

end find_m_l136_136916


namespace opposite_of_neg_2023_l136_136676

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136676


namespace remaining_area_after_cut_l136_136353

theorem remaining_area_after_cut
  (cell_side_length : ℝ)
  (grid_side_length : ℕ)
  (total_area : ℝ)
  (removed_area : ℝ)
  (hyp1 : cell_side_length = 1)
  (hyp2 : grid_side_length = 6)
  (hyp3 : total_area = (grid_side_length * grid_side_length) * cell_side_length * cell_side_length) 
  (hyp4 : removed_area = 9) :
  total_area - removed_area = 27 := by
  sorry

end remaining_area_after_cut_l136_136353


namespace maria_savings_l136_136411

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end maria_savings_l136_136411


namespace power_of_point_l136_136152

namespace ChordsIntersect

variables (A B C D P : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]

def AP := 4
def CP := 9

theorem power_of_point (BP DP : ℕ) :
  AP * BP = CP * DP -> (BP / DP) = 9 / 4 :=
by
  sorry

end ChordsIntersect

end power_of_point_l136_136152


namespace average_speed_of_trip_l136_136315

noncomputable def total_distance (d1 d2 : ℝ) : ℝ :=
  d1 + d2

noncomputable def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem average_speed_of_trip :
  let d1 := 60
  let s1 := 20
  let d2 := 120
  let s2 := 60
  let total_d := total_distance d1 d2
  let time1 := travel_time d1 s1
  let time2 := travel_time d2 s2
  let total_t := time1 + time2
  average_speed total_d total_t = 36 :=
by
  sorry

end average_speed_of_trip_l136_136315


namespace opposite_of_neg_2023_l136_136628

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136628


namespace smallest_positive_value_l136_136074

theorem smallest_positive_value (c d : ℤ) (h : c^2 > d^2) : 
  ∃ m > 0, m = (c^2 + d^2) / (c^2 - d^2) + (c^2 - d^2) / (c^2 + d^2) ∧ m = 2 :=
by
  sorry

end smallest_positive_value_l136_136074


namespace ratio_of_third_to_second_is_four_l136_136006

theorem ratio_of_third_to_second_is_four
  (x y z k : ℕ)
  (h1 : y = 2 * x)
  (h2 : z = k * y)
  (h3 : (x + y + z) / 3 = 165)
  (h4 : y = 90) :
  z / y = 4 :=
by
  sorry

end ratio_of_third_to_second_is_four_l136_136006


namespace percentage_increase_of_kim_l136_136250

variables (S P K : ℝ)
variables (h1 : S = 0.80 * P) (h2 : S + P = 1.80) (h3 : K = 1.12)

theorem percentage_increase_of_kim (hK : K = 1.12) (hS : S = 0.80 * P) (hSP : S + P = 1.80) :
  ((K - S) / S * 100) = 40 :=
sorry

end percentage_increase_of_kim_l136_136250


namespace opposite_of_neg_2023_l136_136448

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136448


namespace hannah_total_cost_l136_136232

def price_per_kg : ℝ := 5
def discount_rate : ℝ := 0.4
def kilograms : ℝ := 10

theorem hannah_total_cost :
  (price_per_kg * (1 - discount_rate)) * kilograms = 30 := 
by
  sorry

end hannah_total_cost_l136_136232


namespace gcd_of_three_numbers_l136_136781

theorem gcd_of_three_numbers : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end gcd_of_three_numbers_l136_136781


namespace trailing_zeroes_500_fact_l136_136910

theorem trailing_zeroes_500_fact : 
  let count_multiples (n m : ℕ) := n / m 
  let count_5 := count_multiples 500 5
  let count_25 := count_multiples 500 25
  let count_125 := count_multiples 500 125
-- We don't count multiples of 625 because 625 > 500, thus its count is 0. 
-- Therefore: total trailing zeroes = count_5 + count_25 + count_125
  count_5 + count_25 + count_125 = 124 := sorry

end trailing_zeroes_500_fact_l136_136910


namespace find_d_l136_136873

-- Defining the basic points and their corresponding conditions
structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def a : Point ℝ := ⟨1, 0, 1⟩
def b : Point ℝ := ⟨0, 1, 0⟩
def c : Point ℝ := ⟨0, 1, 1⟩

-- introducing k as a positive integer
variables (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1)

def d (k : ℤ) : Point ℝ := ⟨k*d, k*d, -d⟩ where d := -(k / (k-1))

-- The proof statement
theorem find_d (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1) :
∃ d: ℝ, d = - (k / (k-1)) :=
sorry

end find_d_l136_136873


namespace opposite_of_neg_2023_l136_136660

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136660


namespace paul_account_balance_after_transactions_l136_136427

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l136_136427


namespace opposite_of_neg_2023_l136_136513

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136513


namespace opposite_neg_2023_l136_136532

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136532


namespace prop_A_l136_136155

theorem prop_A (x : ℝ) (h : x > 1) : (x + (1 / (x - 1)) >= 3) :=
sorry

end prop_A_l136_136155


namespace opposite_of_neg_2023_l136_136652

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136652


namespace opposite_neg_2023_l136_136524

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l136_136524


namespace opposite_of_neg_2023_l136_136619

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l136_136619


namespace calc_difference_of_squares_l136_136037

theorem calc_difference_of_squares :
  625^2 - 375^2 = 250000 :=
by sorry

end calc_difference_of_squares_l136_136037


namespace import_tax_paid_l136_136011

theorem import_tax_paid (total_value excess_value tax_rate tax_paid : ℝ)
  (h₁ : total_value = 2590)
  (h₂ : excess_value = total_value - 1000)
  (h₃ : tax_rate = 0.07)
  (h₄ : tax_paid = excess_value * tax_rate) : 
  tax_paid = 111.30 := by
  -- variables
  sorry

end import_tax_paid_l136_136011


namespace opposite_of_neg_2023_l136_136453

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l136_136453


namespace total_fiscal_revenue_scientific_notation_l136_136374

theorem total_fiscal_revenue_scientific_notation : 
  ∃ a n, (1073 * 10^8 : ℝ) = a * 10^n ∧ (1 ≤ |a| ∧ |a| < 10) ∧ a = 1.07 ∧ n = 11 :=
by
  use 1.07, 11
  simp
  sorry

end total_fiscal_revenue_scientific_notation_l136_136374


namespace opposite_of_neg_2023_l136_136661

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l136_136661


namespace min_value_of_one_over_a_plus_one_over_b_l136_136395

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l136_136395


namespace opposite_of_neg2023_l136_136460

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l136_136460


namespace extended_ohara_triple_example_l136_136149

theorem extended_ohara_triple_example : 
  (2 * Real.sqrt 49 + Real.sqrt 64 = 22) :=
by
  -- We are stating the conditions and required proof here.
  sorry

end extended_ohara_triple_example_l136_136149


namespace twenty_four_is_eighty_percent_of_what_number_l136_136151

theorem twenty_four_is_eighty_percent_of_what_number (x : ℝ) (hx : 24 = 0.8 * x) : x = 30 :=
  sorry

end twenty_four_is_eighty_percent_of_what_number_l136_136151
