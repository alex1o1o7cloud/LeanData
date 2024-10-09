import Mathlib

namespace tan_alpha_value_l1299_129990

noncomputable def f (x : ℝ) := 3 * Real.sin x + 4 * Real.cos x

theorem tan_alpha_value (α : ℝ) (h : ∀ x : ℝ, f x ≥ f α) : Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l1299_129990


namespace blankets_first_day_l1299_129964

-- Definition of the conditions
def num_people := 15
def blankets_day_three := 22
def total_blankets := 142

-- The problem statement
theorem blankets_first_day (B : ℕ) : 
  (num_people * B) + (3 * (num_people * B)) + blankets_day_three = total_blankets → 
  B = 2 :=
by sorry

end blankets_first_day_l1299_129964


namespace water_consumption_150_litres_per_household_4_months_6000_litres_l1299_129978

def number_of_households (household_water_use_per_month : ℕ) (water_supply : ℕ) (duration_months : ℕ) : ℕ :=
  water_supply / (household_water_use_per_month * duration_months)

theorem water_consumption_150_litres_per_household_4_months_6000_litres : 
  number_of_households 150 6000 4 = 10 :=
by
  sorry

end water_consumption_150_litres_per_household_4_months_6000_litres_l1299_129978


namespace tony_average_time_to_store_l1299_129967

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l1299_129967


namespace expansion_of_product_l1299_129981

theorem expansion_of_product (x : ℝ) :
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := 
by
  sorry

end expansion_of_product_l1299_129981


namespace all_integers_equal_l1299_129949

theorem all_integers_equal (k : ℕ) (a : Fin (2 * k + 1) → ℤ)
(h : ∀ b : Fin (2 * k + 1) → ℤ,
  (∀ i : Fin (2 * k + 1), b i = (a ((i : ℕ) % (2 * k + 1)) + a ((i + 1) % (2 * k + 1))) / 2) →
  ∀ i : Fin (2 * k + 1), ↑(b i) % 2 = 0) :
∀ i j : Fin (2 * k + 1), a i = a j :=
by
  sorry

end all_integers_equal_l1299_129949


namespace solve_for_x_l1299_129965

variable (x y z a b w : ℝ)
variable (angle_DEB : ℝ)

def angle_sum_D (x y z angle_DEB : ℝ) : Prop := x + y + z + angle_DEB = 360
def angle_sum_E (a b w angle_DEB : ℝ) : Prop := a + b + w + angle_DEB = 360

theorem solve_for_x 
  (h1 : angle_sum_D x y z angle_DEB) 
  (h2 : angle_sum_E a b w angle_DEB) : 
  x = a + b + w - y - z :=
by
  -- Proof not required
  sorry

end solve_for_x_l1299_129965


namespace minimum_value_of_quadratic_function_l1299_129941

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2

theorem minimum_value_of_quadratic_function :
  ∃ m : ℝ, (∀ x : ℝ, quadratic_function x ≥ m) ∧ (∀ ε > 0, ∃ x : ℝ, quadratic_function x < m + ε) ∧ m = 2 :=
by
  sorry

end minimum_value_of_quadratic_function_l1299_129941


namespace aira_fewer_bands_than_joe_l1299_129940

-- Define initial conditions
variables (samantha_bands aira_bands joe_bands : ℕ)
variables (shares_each : ℕ) (total_bands: ℕ)

-- Conditions from the problem
axiom h1 : shares_each = 6
axiom h2 : samantha_bands = aira_bands + 5
axiom h3 : total_bands = shares_each * 3
axiom h4 : aira_bands = 4
axiom h5 : samantha_bands + aira_bands + joe_bands = total_bands

-- The statement to be proven
theorem aira_fewer_bands_than_joe : joe_bands - aira_bands = 1 :=
sorry

end aira_fewer_bands_than_joe_l1299_129940


namespace eggs_in_each_basket_is_15_l1299_129996
open Nat

theorem eggs_in_each_basket_is_15 :
  ∃ n : ℕ, (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧ (n = 15) :=
sorry

end eggs_in_each_basket_is_15_l1299_129996


namespace complex_expression_l1299_129918

theorem complex_expression (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3 / 2 :=
by 
  sorry

end complex_expression_l1299_129918


namespace find_angle_C_l1299_129984

theorem find_angle_C (a b c : ℝ) (h : a ^ 2 + b ^ 2 - c ^ 2 + a * b = 0) : 
  C = 2 * pi / 3 := 
sorry

end find_angle_C_l1299_129984


namespace pencil_distribution_l1299_129950

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) 
  (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) : 
  total_pencils / max_students = 10 :=
by
  sorry

end pencil_distribution_l1299_129950


namespace real_roots_a_set_t_inequality_l1299_129927

noncomputable def set_of_a : Set ℝ := {a | -1 ≤ a ∧ a ≤ 7}

theorem real_roots_a_set (x a : ℝ) :
  (∃ x, x^2 - 4 * x + abs (a - 3) = 0) ↔ a ∈ set_of_a := 
by
  sorry

theorem t_inequality (t a : ℝ) (h : ∀ a ∈ set_of_a, t^2 - 2 * a * t + 12 < 0) :
  3 < t ∧ t < 4 := 
by
  sorry

end real_roots_a_set_t_inequality_l1299_129927


namespace parabola_equation_focus_l1299_129901

theorem parabola_equation_focus (p : ℝ) (h₀ : p > 0)
  (h₁ : (p / 2 = 2)) : (y^2 = 2 * p * x) :=
  sorry

end parabola_equation_focus_l1299_129901


namespace inequality_proof_l1299_129917

noncomputable def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

theorem inequality_proof (a b : ℝ) (ha : 1 < a) (hb : 0 < b) : 
  f a (a + b) > f a 1 → g (a / b) < g 0 → 1 / (a + b) < Real.log (a + b) / b ∧ Real.log (a + b) / b < a / b := 
by
  sorry

end inequality_proof_l1299_129917


namespace max_value_g_l1299_129930

def g : ℕ → ℤ
| n => if n < 5 then n + 10 else g (n - 3)

theorem max_value_g : ∃ x, (∀ n : ℕ, g n ≤ x) ∧ (∃ y, g y = x) ∧ x = 14 := 
by
  sorry

end max_value_g_l1299_129930


namespace wilson_hamburgers_l1299_129902

def hamburger_cost (H : ℕ) := 5 * H
def cola_cost := 6
def discount := 4
def total_cost (H : ℕ) := hamburger_cost H + cola_cost - discount

theorem wilson_hamburgers (H : ℕ) (h : total_cost H = 12) : H = 2 :=
sorry

end wilson_hamburgers_l1299_129902


namespace roofing_cost_per_foot_l1299_129921

theorem roofing_cost_per_foot:
  ∀ (total_feet needed_feet free_feet : ℕ) (total_cost : ℕ),
  needed_feet = 300 →
  free_feet = 250 →
  total_cost = 400 →
  needed_feet - free_feet = 50 →
  total_cost / (needed_feet - free_feet) = 8 :=
by sorry

end roofing_cost_per_foot_l1299_129921


namespace hyperbola_standard_eq_l1299_129997

theorem hyperbola_standard_eq (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  (∃ b, b^2 = c^2 - a^2 ∧ (1 = (x^2 / a^2 - y^2 / b^2) ∨ 1 = (y^2 / a^2 - x^2 / b^2))) := by
  sorry

end hyperbola_standard_eq_l1299_129997


namespace calculate_expr_l1299_129982

theorem calculate_expr : (2023^0 + (-1/3) = 2/3) := by
  sorry

end calculate_expr_l1299_129982


namespace no_non_integer_point_exists_l1299_129980

variable (b0 b1 b2 b3 b4 b5 u v : ℝ)

def q (x y : ℝ) : ℝ := b0 + b1 * x + b2 * y + b3 * x^2 + b4 * x * y + b5 * y^2

theorem no_non_integer_point_exists
    (h₀ : q b0 b1 b2 b3 b4 b5 0 0 = 0)
    (h₁ : q b0 b1 b2 b3 b4 b5 1 0 = 0)
    (h₂ : q b0 b1 b2 b3 b4 b5 (-1) 0 = 0)
    (h₃ : q b0 b1 b2 b3 b4 b5 0 1 = 0)
    (h₄ : q b0 b1 b2 b3 b4 b5 0 (-1) = 0)
    (h₅ : q b0 b1 b2 b3 b4 b5 1 1 = 0) :
  ∀ u v : ℝ, (¬ ∃ (n m : ℤ), u = n ∧ v = m) → q b0 b1 b2 b3 b4 b5 u v ≠ 0 :=
by
  sorry

end no_non_integer_point_exists_l1299_129980


namespace num_remainders_prime_squares_mod_210_l1299_129975

theorem num_remainders_prime_squares_mod_210 :
  (∃ (p : ℕ) (hp : p > 7) (hprime : Prime p), 
    ∀ r : Finset ℕ, 
      (∀ q ∈ r, (∃ (k : ℕ), p = 210 * k + q)) 
      → r.card = 8) :=
sorry

end num_remainders_prime_squares_mod_210_l1299_129975


namespace directrix_of_parabola_l1299_129935

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l1299_129935


namespace min_area_circle_equation_l1299_129915

theorem min_area_circle_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : (x - 4)^2 + (y - 4)^2 = 256 :=
sorry

end min_area_circle_equation_l1299_129915


namespace blue_tshirts_in_pack_l1299_129931

theorem blue_tshirts_in_pack
  (packs_white : ℕ := 2) 
  (white_per_pack : ℕ := 5) 
  (packs_blue : ℕ := 4)
  (cost_per_tshirt : ℕ := 3)
  (total_cost : ℕ := 66)
  (B : ℕ := 3) :
  (packs_white * white_per_pack * cost_per_tshirt) + (packs_blue * B * cost_per_tshirt) = total_cost := 
by
  sorry

end blue_tshirts_in_pack_l1299_129931


namespace isosceles_triangle_l1299_129989

def triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B → (B = C)

theorem isosceles_triangle (a b c A B C : ℝ) (h : a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B) : B = C :=
  sorry

end isosceles_triangle_l1299_129989


namespace winston_cents_left_l1299_129904

-- Definitions based on the conditions in the problem
def quarters := 14
def cents_per_quarter := 25
def half_dollar_in_cents := 50

-- Formulation of the problem statement in Lean
theorem winston_cents_left : (quarters * cents_per_quarter) - half_dollar_in_cents = 300 :=
by sorry

end winston_cents_left_l1299_129904


namespace intersection_of_sets_l1299_129922

noncomputable def setA : Set ℝ := { x | (x + 2) / (x - 2) ≤ 0 }
noncomputable def setB : Set ℝ := { x | x ≥ 1 }
noncomputable def expectedSet : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_of_sets : (setA ∩ setB) = expectedSet := by
  sorry

end intersection_of_sets_l1299_129922


namespace simplified_expression_evaluates_to_2_l1299_129914

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end simplified_expression_evaluates_to_2_l1299_129914


namespace albert_horses_l1299_129966

variable {H C : ℝ}

theorem albert_horses :
  (2000 * H + 9 * C = 13400) ∧ (200 * H + 0.20 * 9 * C = 1880) ∧ (∀ x : ℝ, x = 2000) → H = 4 := 
by
  sorry

end albert_horses_l1299_129966


namespace cricket_innings_l1299_129928

theorem cricket_innings (n : ℕ) (h1 : (36 * n) / n = 36) (h2 : (36 * n + 80) / (n + 1) = 40) : n = 10 := by
  -- The proof goes here
  sorry

end cricket_innings_l1299_129928


namespace numeric_puzzle_AB_eq_B_pow_V_l1299_129988

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l1299_129988


namespace flowers_sold_difference_l1299_129957

def number_of_daisies_sold_on_second_day (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) : Prop :=
  d3 = 2 * d2 - 10 ∧
  d_sum = 45 + d2 + d3 + 120

theorem flowers_sold_difference (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) 
  (h : number_of_daisies_sold_on_second_day d2 d3 d_sum) :
  45 + d2 + d3 + 120 = 350 → 
  d2 - 45 = 20 := 
by
  sorry

end flowers_sold_difference_l1299_129957


namespace find_trajectory_l1299_129926

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (y - 1) * (y + 1) / ((x + 1) * (x - 1)) = -1 / 3

theorem find_trajectory (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  trajectory_equation x y → x^2 + 3 * y^2 = 4 :=
by
  sorry

end find_trajectory_l1299_129926


namespace triangle_inequality_l1299_129963

variables {α β γ a b c : ℝ}
variable {n : ℕ}

theorem triangle_inequality (h_sum_angles : α + β + γ = Real.pi) (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.pi / 3) ^ n ≤ (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) ∧ 
  (a * α ^ n + b * β ^ n + c * γ ^ n) / (a + b + c) < (Real.pi ^ n / 2) :=
by
  sorry

end triangle_inequality_l1299_129963


namespace solve_inequality1_solve_inequality2_l1299_129969

-- Problem 1: Solve the inequality (1)
theorem solve_inequality1 (x : ℝ) (h : x ≠ -4) : 
  (2 - x) / (x + 4) ≤ 0 ↔ (x ≥ 2 ∨ x < -4) := sorry

-- Problem 2: Solve the inequality (2) for different cases of a
theorem solve_inequality2 (x a : ℝ) : 
  (x^2 - 3 * a * x + 2 * a^2 ≥ 0) ↔
  (if a > 0 then (x ≥ 2 * a ∨ x ≤ a) 
   else if a < 0 then (x ≥ a ∨ x ≤ 2 * a) 
   else true) := sorry

end solve_inequality1_solve_inequality2_l1299_129969


namespace triangle_lengths_relationship_l1299_129948

-- Given data
variables {a b c f_a f_b f_c t_a t_b t_c : ℝ}
-- Conditions/assumptions
variables (h1 : f_a * t_a = b * c)
variables (h2 : f_b * t_b = a * c)
variables (h3 : f_c * t_c = a * b)

-- Theorem to prove
theorem triangle_lengths_relationship :
  a^2 * b^2 * c^2 = f_a * f_b * f_c * t_a * t_b * t_c :=
by sorry

end triangle_lengths_relationship_l1299_129948


namespace solve_for_x_l1299_129944

theorem solve_for_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 := by
  sorry

end solve_for_x_l1299_129944


namespace minimize_square_sum_l1299_129909

theorem minimize_square_sum (x1 x2 x3 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) 
  (h4 : x1 + 3 * x2 + 5 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 ≥ 2000 / 7 :=
sorry

end minimize_square_sum_l1299_129909


namespace total_peaches_l1299_129983

variable (numberOfBaskets : ℕ)
variable (redPeachesPerBasket : ℕ)
variable (greenPeachesPerBasket : ℕ)

theorem total_peaches (h1 : numberOfBaskets = 1) 
                      (h2 : redPeachesPerBasket = 4)
                      (h3 : greenPeachesPerBasket = 3) :
  numberOfBaskets * (redPeachesPerBasket + greenPeachesPerBasket) = 7 := 
by
  sorry

end total_peaches_l1299_129983


namespace Sophie_Spends_72_80_l1299_129938

noncomputable def SophieTotalCost : ℝ :=
  let cupcakesCost := 5 * 2
  let doughnutsCost := 6 * 1
  let applePieCost := 4 * 2
  let cookiesCost := 15 * 0.60
  let chocolateBarsCost := 8 * 1.50
  let sodaCost := 12 * 1.20
  let gumCost := 3 * 0.80
  let chipsCost := 10 * 1.10
  cupcakesCost + doughnutsCost + applePieCost + cookiesCost + chocolateBarsCost + sodaCost + gumCost + chipsCost

theorem Sophie_Spends_72_80 : SophieTotalCost = 72.80 :=
by
  sorry

end Sophie_Spends_72_80_l1299_129938


namespace monotonic_decreasing_range_of_a_l1299_129923

-- Define the given function
def f (a x : ℝ) := a * x^2 - 3 * x + 4

-- State the proof problem
theorem monotonic_decreasing_range_of_a (a : ℝ) : (∀ x : ℝ, x < 6 → deriv (f a) x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
sorry

end monotonic_decreasing_range_of_a_l1299_129923


namespace speed_of_first_train_l1299_129910

-- Define the problem conditions
def distance_between_stations : ℝ := 20
def speed_of_second_train : ℝ := 25
def meet_time : ℝ := 8
def start_time_first_train : ℝ := 7
def start_time_second_train : ℝ := 8
def travel_time_first_train : ℝ := meet_time - start_time_first_train

-- The actual proof statement in Lean
theorem speed_of_first_train : ∀ (v : ℝ),
  v * travel_time_first_train = distance_between_stations → v = 20 :=
by
  intro v
  intro h
  sorry

end speed_of_first_train_l1299_129910


namespace probability_divisible_by_5_l1299_129929

def is_three_digit_integer (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

theorem probability_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit_integer N) 
  (h2 : ends_with_five N) : 
  ∃ (p : ℚ), p = 1 := 
sorry

end probability_divisible_by_5_l1299_129929


namespace roberto_outfits_l1299_129911

theorem roberto_outfits : 
  let trousers := 5
  let shirts := 5
  let jackets := 3
  (trousers * shirts * jackets = 75) :=
by sorry

end roberto_outfits_l1299_129911


namespace intersection_A_B_l1299_129958

open Set

variable (x : ℝ)

def setA : Set ℝ := {x | x^2 - 3 * x ≤ 0}
def setB : Set ℝ := {1, 2}

theorem intersection_A_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_A_B_l1299_129958


namespace turtles_order_l1299_129962

-- Define variables for each turtle as real numbers representing their positions
variables (O P S E R : ℝ)

-- Define the conditions given in the problem
def condition1 := S = O - 10
def condition2 := S = R + 25
def condition3 := R = E - 5
def condition4 := E = P - 25

-- Define the order of arrival
def order_of_arrival (O P S E R : ℝ) := 
     O = 0 ∧ 
     P = -5 ∧
     S = -10 ∧
     E = -30 ∧
     R = -35

-- Theorem to show the given conditions imply the order of arrival
theorem turtles_order (h1 : condition1 S O)
                     (h2 : condition2 S R)
                     (h3 : condition3 R E)
                     (h4 : condition4 E P) :
  order_of_arrival O P S E R :=
by sorry

end turtles_order_l1299_129962


namespace evaluate_three_squared_raised_four_l1299_129977

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l1299_129977


namespace sum_even_integers_602_to_700_l1299_129937

-- Definitions based on the conditions and the problem statement
def sum_first_50_even_integers := 2550
def n_even_602_700 := 50
def first_term_602_to_700 := 602
def last_term_602_to_700 := 700

-- Theorem statement
theorem sum_even_integers_602_to_700 : 
  sum_first_50_even_integers = 2550 → 
  n_even_602_700 = 50 →
  (n_even_602_700 / 2) * (first_term_602_to_700 + last_term_602_to_700) = 32550 :=
by
  sorry

end sum_even_integers_602_to_700_l1299_129937


namespace steve_break_even_l1299_129999

noncomputable def break_even_performances
  (fixed_overhead : ℕ)
  (min_production_cost max_production_cost : ℕ)
  (venue_capacity percentage_occupied : ℕ)
  (ticket_price : ℕ) : ℕ :=
(fixed_overhead + (percentage_occupied / 100 * venue_capacity * ticket_price)) / (percentage_occupied / 100 * venue_capacity * ticket_price)

theorem steve_break_even
  (fixed_overhead : ℕ := 81000)
  (min_production_cost : ℕ := 5000)
  (max_production_cost : ℕ := 9000)
  (venue_capacity : ℕ := 500)
  (percentage_occupied : ℕ := 80)
  (ticket_price : ℕ := 40)
  (avg_production_cost : ℕ := (min_production_cost + max_production_cost) / 2) :
  break_even_performances fixed_overhead min_production_cost max_production_cost venue_capacity percentage_occupied ticket_price = 9 :=
by
  sorry

end steve_break_even_l1299_129999


namespace line_equation_cartesian_circle_equation_cartesian_l1299_129936

theorem line_equation_cartesian (t : ℝ) (x y : ℝ) : 
  (x = 3 - (Real.sqrt 2 / 2) * t ∧ y = Real.sqrt 5 + (Real.sqrt 2 / 2) * t) -> 
  y = -2 * x + 6 + Real.sqrt 5 :=
sorry

theorem circle_equation_cartesian (ρ θ x y : ℝ) : 
  (ρ = 2 * Real.sqrt 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) -> 
  x^2 = 0 :=
sorry

end line_equation_cartesian_circle_equation_cartesian_l1299_129936


namespace unique_integer_for_P5_l1299_129913

-- Define the polynomial P with integer coefficients
variable (P : ℤ → ℤ)

-- The conditions given in the problem
variable (x1 x2 x3 : ℤ)
variable (Hx1 : P x1 = 1)
variable (Hx2 : P x2 = 2)
variable (Hx3 : P x3 = 3)

-- The main theorem to prove
theorem unique_integer_for_P5 {P : ℤ → ℤ} {x1 x2 x3 : ℤ}
(Hx1 : P x1 = 1) (Hx2 : P x2 = 2) (Hx3 : P x3 = 3) :
  ∃!(x : ℤ), P x = 5 := sorry

end unique_integer_for_P5_l1299_129913


namespace bullfinches_are_50_l1299_129998

theorem bullfinches_are_50 :
  ∃ N : ℕ, (N > 50 ∨ N < 50 ∨ N ≥ 1) ∧ (¬(N > 50) ∨ ¬(N < 50) ∨ ¬(N ≥ 1)) ∧
  (N > 50 ∧ ¬(N < 50) ∨ N < 50 ∧ ¬(N > 50) ∨ N ≥ 1 ∧ (¬(N > 50) ∧ ¬(N < 50))) ∧
  N = 50 :=
by
  sorry

end bullfinches_are_50_l1299_129998


namespace negation_of_existential_prop_l1299_129943

open Real

theorem negation_of_existential_prop :
  ¬ (∃ x, x ≥ π / 2 ∧ sin x > 1) ↔ ∀ x, x < π / 2 → sin x ≤ 1 :=
by
  sorry

end negation_of_existential_prop_l1299_129943


namespace days_to_cover_half_lake_l1299_129934

-- Define the problem conditions in Lean
def doubles_every_day (size: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def takes_25_days_to_cover_lake (size: ℕ → ℝ) (lake_size: ℝ) : Prop :=
  size 25 = lake_size

-- Define the main theorem
theorem days_to_cover_half_lake (size: ℕ → ℝ) (lake_size: ℝ) 
  (h1: doubles_every_day size) (h2: takes_25_days_to_cover_lake size lake_size) : 
  size 24 = lake_size / 2 :=
sorry

end days_to_cover_half_lake_l1299_129934


namespace temple_shop_total_cost_l1299_129955

theorem temple_shop_total_cost :
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  total_cost = 374 :=
by
  let price_per_object := 11
  let num_people := 5
  let items_per_person := 4
  let extra_items := 4
  let total_objects := num_people * items_per_person + extra_items
  let total_cost := total_objects * price_per_object
  show total_cost = 374
  sorry

end temple_shop_total_cost_l1299_129955


namespace lucy_total_packs_l1299_129959

-- Define the number of packs of cookies Lucy bought
def packs_of_cookies : ℕ := 12

-- Define the number of packs of noodles Lucy bought
def packs_of_noodles : ℕ := 16

-- Define the total number of packs of groceries Lucy bought
def total_packs_of_groceries : ℕ := packs_of_cookies + packs_of_noodles

-- Proof statement: The total number of packs of groceries Lucy bought is 28
theorem lucy_total_packs : total_packs_of_groceries = 28 := by
  sorry

end lucy_total_packs_l1299_129959


namespace probability_one_hits_correct_l1299_129907

-- Define the probabilities for A hitting and B hitting
noncomputable def P_A : ℝ := 0.4
noncomputable def P_B : ℝ := 0.5

-- Calculate the required probability
noncomputable def probability_one_hits : ℝ :=
  P_A * (1 - P_B) + (1 - P_A) * P_B

-- Statement of the theorem
theorem probability_one_hits_correct :
  probability_one_hits = 0.5 := by 
  sorry

end probability_one_hits_correct_l1299_129907


namespace milk_removal_replacement_l1299_129994

theorem milk_removal_replacement (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 45) :
  (45 - x) * (45 - x) / 45 = 28.8 → x = 9 :=
by
  -- skipping the proof for now
  sorry

end milk_removal_replacement_l1299_129994


namespace inequality_example_l1299_129912

variable {a b c : ℝ} -- Declare a, b, c as real numbers

theorem inequality_example
  (ha : 0 < a)  -- Condition: a is positive
  (hb : 0 < b)  -- Condition: b is positive
  (hc : 0 < c) :  -- Condition: c is positive
  (ab * (a + b) + ac * (a + c) + bc * (b + c)) / (abc) ≥ 6 := 
sorry  -- Proof is skipped

end inequality_example_l1299_129912


namespace grayson_vs_rudy_distance_l1299_129951

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end grayson_vs_rudy_distance_l1299_129951


namespace find_remainder_l1299_129953

theorem find_remainder :
  ∀ (D d q r : ℕ), 
    D = 18972 → 
    d = 526 → 
    q = 36 → 
    D = d * q + r → 
    r = 36 :=
by 
  intros D d q r hD hd hq hEq
  sorry

end find_remainder_l1299_129953


namespace function_d_has_no_boundary_point_l1299_129974

def is_boundary_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∃ x₁ < x₀, f x₁ = 0) ∧ (∃ x₂ > x₀, f x₂ = 0)

def f_a (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x - 2
def f_b (x : ℝ) : ℝ := abs (x^2 - 3)
def f_c (x : ℝ) : ℝ := 1 - abs (x - 2)
def f_d (x : ℝ) : ℝ := x^3 + x

theorem function_d_has_no_boundary_point :
  ¬ ∃ x₀ : ℝ, is_boundary_point f_d x₀ :=
sorry

end function_d_has_no_boundary_point_l1299_129974


namespace inverse_function_log_base_two_l1299_129916

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_log_base_two (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : f a (a^2) = a) : f a = fun x => Real.log x / Real.log 2 := 
by
  sorry

end inverse_function_log_base_two_l1299_129916


namespace waiter_date_trick_l1299_129985

theorem waiter_date_trick :
  ∃ d₂ : ℕ, ∃ x : ℝ, 
  (∀ d₁ : ℕ, ∀ x : ℝ, x + d₁ = 168) ∧
  3 * x + d₂ = 486 ∧
  3 * (x + d₂) = 516 ∧
  d₂ = 15 :=
by
  sorry

end waiter_date_trick_l1299_129985


namespace minimum_value_of_f_range_of_x_l1299_129971

noncomputable def f (x : ℝ) := |2*x + 1| + |2*x - 1|

-- Problem 1
theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 2 :=
by
  intro x
  sorry

-- Problem 2
theorem range_of_x (a b : ℝ) (h : |2*a + b| + |a| - (1/2) * |a + b| * f x ≥ 0) : 
  - (1/2) ≤ x ∧ x ≤ 1/2 :=
by
  sorry

end minimum_value_of_f_range_of_x_l1299_129971


namespace xy_squared_sum_l1299_129993

theorem xy_squared_sum {x y : ℝ} (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 :=
by
  sorry

end xy_squared_sum_l1299_129993


namespace number_of_ways_to_make_78_rubles_l1299_129995

theorem number_of_ways_to_make_78_rubles : ∃ n, n = 5 ∧ ∃ x y : ℕ, 78 = 5 * x + 3 * y := sorry

end number_of_ways_to_make_78_rubles_l1299_129995


namespace steven_has_19_peaches_l1299_129973

-- Conditions
def jill_peaches : ℕ := 6
def steven_peaches : ℕ := jill_peaches + 13

-- Statement to prove
theorem steven_has_19_peaches : steven_peaches = 19 :=
by {
    -- Proof steps would go here
    sorry
}

end steven_has_19_peaches_l1299_129973


namespace solve_equation1_solve_equation2_l1299_129932

-- Define the two equations
def equation1 (x : ℝ) := 3 * x - 4 = -2 * (x - 1)
def equation2 (x : ℝ) := 1 + (2 * x + 1) / 3 = (3 * x - 2) / 2

-- The statements to prove
theorem solve_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1.2 :=
by
  sorry

theorem solve_equation2 : ∃ x : ℝ, equation2 x ∧ x = 2.8 :=
by
  sorry

end solve_equation1_solve_equation2_l1299_129932


namespace triangle_side_length_l1299_129987

theorem triangle_side_length (a b p : ℝ) (H_perimeter : a + b + 10 = p) (H_a : a = 7) (H_b : b = 15) (H_p : p = 32) : 10 = 10 :=
by
  sorry

end triangle_side_length_l1299_129987


namespace probability_differ_by_three_is_one_sixth_l1299_129905

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l1299_129905


namespace f_le_g_for_a_eq_neg1_l1299_129946

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * Real.exp x

noncomputable def g (t : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * x - Real.log x + t

theorem f_le_g_for_a_eq_neg1 (t : ℝ) :
  let b := 3
  ∃ x ∈ Set.Ioi 0, f (-1) b x ≤ g t x ↔ t ≤ Real.exp 2 - 1 / 2 :=
by
  sorry

end f_le_g_for_a_eq_neg1_l1299_129946


namespace distinct_solutions_abs_eq_l1299_129960

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 3| = |x + 5|) → x = -1 :=
by
  sorry

end distinct_solutions_abs_eq_l1299_129960


namespace max_edges_convex_polyhedron_l1299_129906

theorem max_edges_convex_polyhedron (n : ℕ) (c l e : ℕ) (h1 : c = n) (h2 : c + l = e + 2) (h3 : 2 * e ≥ 3 * l) : e ≤ 3 * n - 6 := 
sorry

end max_edges_convex_polyhedron_l1299_129906


namespace find_a_b_l1299_129972

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 1) → (x^2 + a * x + b > 0)) →
  (a = 1 ∧ b = -2) :=
by
  sorry

end find_a_b_l1299_129972


namespace xiao_ming_error_step_l1299_129952

theorem xiao_ming_error_step (x : ℝ) :
  (1 / (x + 1) = (2 * x) / (3 * x + 3) - 1) → 
  3 = 2 * x - (3 * x + 3) → 
  (3 = 2 * x - 3 * x + 3) ↔ false := by
  sorry

end xiao_ming_error_step_l1299_129952


namespace min_boat_trips_l1299_129954
-- Import Mathlib to include necessary libraries

-- Define the problem using noncomputable theory if necessary
theorem min_boat_trips (students boat_capacity : ℕ) (h1 : students = 37) (h2 : boat_capacity = 5) : ∃ x : ℕ, x ≥ 9 :=
by
  -- Here we need to prove the assumption and goal, hence adding sorry
  sorry

end min_boat_trips_l1299_129954


namespace determine_x_2y_l1299_129924

theorem determine_x_2y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : (x + y) / 3 = 5 / 3) : x + 2 * y = 8 :=
sorry

end determine_x_2y_l1299_129924


namespace leah_daily_savings_l1299_129925

theorem leah_daily_savings 
  (L : ℝ)
  (h1 : 0.25 * 24 = 6)
  (h2 : ∀ (L : ℝ), (L * 20) = 20 * L)
  (h3 : ∀ (L : ℝ), 2 * L * 12 = 24 * L)
  (h4 :  6 + 20 * L + 24 * L = 28) 
: L = 0.5 :=
by
  sorry

end leah_daily_savings_l1299_129925


namespace inequality_4th_power_l1299_129970

theorem inequality_4th_power (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 :=
sorry

end inequality_4th_power_l1299_129970


namespace car_return_speed_l1299_129979

theorem car_return_speed (d : ℕ) (r : ℕ) (h₁ : d = 180) (h₂ : (2 * d) / ((d / 90) + (d / r)) = 60) : r = 45 :=
by
  rw [h₁] at h₂
  have h3 : 2 * 180 / ((180 / 90) + (180 / r)) = 60 := h₂
  -- The rest of the proof involves solving for r, but here we only need the statement
  sorry

end car_return_speed_l1299_129979


namespace smallest_perfect_cube_divisor_l1299_129991

theorem smallest_perfect_cube_divisor (p q r : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] [hr : Fact (Nat.Prime r)] (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ k : ℕ, (k = (p * q * r^2)^3) ∧ (∃ n, n = p * q^3 * r^4 ∧ n ∣ k) := 
sorry

end smallest_perfect_cube_divisor_l1299_129991


namespace canoe_vs_kayak_l1299_129986

theorem canoe_vs_kayak (
  C K : ℕ 
) (h1 : 14 * C + 15 * K = 288) 
  (h2 : C = (3 * K) / 2) : 
  C - K = 4 := 
sorry

end canoe_vs_kayak_l1299_129986


namespace diane_owes_money_l1299_129947

theorem diane_owes_money (initial_amount winnings total_losses : ℤ) (h_initial : initial_amount = 100) (h_winnings : winnings = 65) (h_losses : total_losses = 215) : 
  initial_amount + winnings - total_losses = -50 := by
  sorry

end diane_owes_money_l1299_129947


namespace julie_savings_fraction_l1299_129956

variables (S : ℝ) (x : ℝ)
theorem julie_savings_fraction (h : 12 * S * x = 4 * S * (1 - x)) : 1 - x = 3 / 4 :=
sorry

end julie_savings_fraction_l1299_129956


namespace no_square_has_units_digit_seven_l1299_129920

theorem no_square_has_units_digit_seven :
  ¬ ∃ n : ℕ, n ≤ 9 ∧ (n^2 % 10) = 7 := by
  sorry

end no_square_has_units_digit_seven_l1299_129920


namespace pastries_left_l1299_129908

def pastries_baked : ℕ := 4 + 29
def pastries_sold : ℕ := 9

theorem pastries_left : pastries_baked - pastries_sold = 24 :=
by
  -- assume pastries_baked = 33
  -- assume pastries_sold = 9
  -- prove 33 - 9 = 24
  sorry

end pastries_left_l1299_129908


namespace oldest_child_age_l1299_129939

theorem oldest_child_age 
  (avg_age : ℕ) (child1 : ℕ) (child2 : ℕ) (child3 : ℕ) (child4 : ℕ)
  (h_avg : avg_age = 8) 
  (h_child1 : child1 = 5) 
  (h_child2 : child2 = 7) 
  (h_child3 : child3 = 10)
  (h_avg_eq : (child1 + child2 + child3 + child4) / 4 = avg_age) :
  child4 = 10 := 
by 
  sorry

end oldest_child_age_l1299_129939


namespace calc_6_4_3_199_plus_100_l1299_129992

theorem calc_6_4_3_199_plus_100 (a b : ℕ) (h_a : a = 199) (h_b : b = 100) :
  6 * a + 4 * a + 3 * a + a + b = 2886 :=
by
  sorry

end calc_6_4_3_199_plus_100_l1299_129992


namespace top_and_bottom_edges_same_color_l1299_129903

-- Define the vertices for top and bottom pentagonal faces
inductive Vertex
| A1 | A2 | A3 | A4 | A5
| B1 | B2 | B3 | B4 | B5

-- Define the edges
inductive Edge : Type
| TopEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) : Edge
| BottomEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge
| SideEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge

-- Define colors
inductive Color
| Red | Blue

-- Define a function that assigns a color to each edge
def edgeColor : Edge → Color := sorry

-- Define a function that checks if a triangle is monochromatic
def isMonochromatic (e1 e2 e3 : Edge) : Prop :=
  edgeColor e1 = edgeColor e2 ∧ edgeColor e2 = edgeColor e3

-- Define our main theorem statement
theorem top_and_bottom_edges_same_color (h : ∀ v1 v2 v3 : Vertex, ¬ isMonochromatic (Edge.TopEdge v1 v2 sorry sorry) (Edge.SideEdge v1 v3 sorry sorry) (Edge.BottomEdge v2 v3 sorry sorry)) : 
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → edgeColor (Edge.TopEdge v1 v2 sorry sorry) = edgeColor (Edge.TopEdge Vertex.A1 Vertex.A2 sorry sorry)) ∧
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → edgeColor (Edge.BottomEdge v1 v2 sorry sorry) = edgeColor (Edge.BottomEdge Vertex.B1 Vertex.B2 sorry sorry)) :=
sorry

end top_and_bottom_edges_same_color_l1299_129903


namespace area_of_right_triangle_l1299_129968

-- Define the conditions
def hypotenuse : ℝ := 9
def angle : ℝ := 30

-- Define the Lean statement for the proof problem
theorem area_of_right_triangle : 
  ∃ (area : ℝ), area = 10.125 * Real.sqrt 3 ∧
  ∃ (shorter_leg : ℝ) (longer_leg : ℝ),
    shorter_leg = hypotenuse / 2 ∧
    longer_leg = shorter_leg * Real.sqrt 3 ∧
    area = (shorter_leg * longer_leg) / 2 :=
by {
  -- The proof would go here, but we only need to state the problem for this task.
  sorry
}

end area_of_right_triangle_l1299_129968


namespace ratio_of_books_to_pens_l1299_129900

theorem ratio_of_books_to_pens (total_stationery : ℕ) (books : ℕ) (pens : ℕ) 
    (h1 : total_stationery = 400) (h2 : books = 280) (h3 : pens = total_stationery - books) : 
    books / (Nat.gcd books pens) = 7 ∧ pens / (Nat.gcd books pens) = 3 := 
by 
  -- proof steps would go here
  sorry

end ratio_of_books_to_pens_l1299_129900


namespace number_of_sarees_l1299_129942

-- Define variables representing the prices of one saree and one shirt
variables (X S T : ℕ)

-- Define the conditions 
def condition1 := X * S + 4 * T = 1600
def condition2 := S + 6 * T = 1600
def condition3 := 12 * T = 2400

-- The proof problem (statement only, without proof)
theorem number_of_sarees (X S T : ℕ) (h1 : condition1 X S T) (h2 : condition2 S T) (h3 : condition3 T) : X = 2 := by
  sorry

end number_of_sarees_l1299_129942


namespace a_4_is_4_l1299_129961

-- Define the general term formula of the sequence
def a (n : ℕ) : ℤ := (-1)^n * n

-- State the desired proof goal
theorem a_4_is_4 : a 4 = 4 :=
by
  -- Proof to be provided here,
  -- adding 'sorry' as we are only defining the statement, not solving it
  sorry

end a_4_is_4_l1299_129961


namespace first_number_in_sum_l1299_129933

theorem first_number_in_sum (a b c : ℝ) (h : a + b + c = 3.622) : a = 3.15 :=
by
  -- Assume the given values of b and c
  have hb : b = 0.014 := sorry
  have hc : c = 0.458 := sorry
  -- From the assumption h and hb, hc, we deduce a = 3.15
  sorry

end first_number_in_sum_l1299_129933


namespace range_a_range_b_l1299_129976

def set_A : Set ℝ := {x | Real.log x / Real.log 2 > 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}
def set_C (b : ℝ) : Set ℝ := {x | b + 1 < x ∧ x < 2 * b + 1}

-- Part (1)
theorem range_a (a : ℝ) : (∀ x, x ∈ set_A → x ∈ set_B a) ↔ a ∈ Set.Iic 4 := sorry

-- Part (2)
theorem range_b (b : ℝ) : (set_A ∪ set_C b = set_A) ↔ b ∈ Set.Iic 0 ∪ Set.Ici 3 := sorry

end range_a_range_b_l1299_129976


namespace xy_yz_zx_nonzero_l1299_129945

theorem xy_yz_zx_nonzero (x y z : ℝ)
  (h1 : 1 / |x^2 + 2 * y * z| + 1 / |y^2 + 2 * z * x| > 1 / |z^2 + 2 * x * y|)
  (h2 : 1 / |y^2 + 2 * z * x| + 1 / |z^2 + 2 * x * y| > 1 / |x^2 + 2 * y * z|)
  (h3 : 1 / |z^2 + 2 * x * y| + 1 / |x^2 + 2 * y * z| > 1 / |y^2 + 2 * z * x|) :
  x * y + y * z + z * x ≠ 0 := by
  sorry

end xy_yz_zx_nonzero_l1299_129945


namespace cost_of_shoes_l1299_129919

-- Define the conditions
def saved : Nat := 30
def earn_per_lawn : Nat := 5
def lawns_per_weekend : Nat := 3
def weekends_needed : Nat := 6

-- Prove the total amount saved is the cost of the shoes
theorem cost_of_shoes : saved + (earn_per_lawn * lawns_per_weekend * weekends_needed) = 120 := by
  sorry

end cost_of_shoes_l1299_129919
