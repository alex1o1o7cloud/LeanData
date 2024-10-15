import Mathlib

namespace NUMINAMATH_GPT_equivalent_form_l1166_116628

theorem equivalent_form (x y : ℝ) (h : y = x + 1/x) :
  (x^4 + x^3 - 3*x^2 + x + 2 = 0) ↔ (x^2 * (y^2 + y - 5) = 0) :=
sorry

end NUMINAMATH_GPT_equivalent_form_l1166_116628


namespace NUMINAMATH_GPT_distance_between_points_is_sqrt_5_l1166_116671

noncomputable def distance_between_polar_points : ℝ :=
  let xA := 1 * Real.cos (3/4 * Real.pi)
  let yA := 1 * Real.sin (3/4 * Real.pi)
  let xB := 2 * Real.cos (Real.pi / 4)
  let yB := 2 * Real.sin (Real.pi / 4)
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

theorem distance_between_points_is_sqrt_5 :
  distance_between_polar_points = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_is_sqrt_5_l1166_116671


namespace NUMINAMATH_GPT_double_angle_cosine_l1166_116635

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_double_angle_cosine_l1166_116635


namespace NUMINAMATH_GPT_g_of_3_l1166_116691

theorem g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 4 * g x + 3 * g (1 / x) = 2 * x) :
  g 3 = 22 / 7 :=
sorry

end NUMINAMATH_GPT_g_of_3_l1166_116691


namespace NUMINAMATH_GPT_general_term_of_sequence_l1166_116673

theorem general_term_of_sequence (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 4 * n) : 
  a n = 2 * n - 5 :=
by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1166_116673


namespace NUMINAMATH_GPT_team_sports_competed_l1166_116625

theorem team_sports_competed (x : ℕ) (n : ℕ) 
  (h1 : (97 + n) / x = 90) 
  (h2 : (73 + n) / x = 87) : 
  x = 8 := 
by sorry

end NUMINAMATH_GPT_team_sports_competed_l1166_116625


namespace NUMINAMATH_GPT_shirts_per_minute_l1166_116604

theorem shirts_per_minute (S : ℕ) 
  (h1 : 12 * S + 14 = 156) : S = 11 := 
by
  sorry

end NUMINAMATH_GPT_shirts_per_minute_l1166_116604


namespace NUMINAMATH_GPT_calc_sqrt_expr_l1166_116684

theorem calc_sqrt_expr :
  (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end NUMINAMATH_GPT_calc_sqrt_expr_l1166_116684


namespace NUMINAMATH_GPT_notebooks_difference_l1166_116680

noncomputable def price_more_than_dime (p : ℝ) : Prop := p > 0.10
noncomputable def payment_equation (nL nN : ℕ) (p : ℝ) : Prop :=
  (nL * p = 2.10 ∧ nN * p = 2.80)

theorem notebooks_difference (nL nN : ℕ) (p : ℝ) (h1 : price_more_than_dime p) (h2 : payment_equation nL nN p) :
  nN - nL = 2 :=
by sorry

end NUMINAMATH_GPT_notebooks_difference_l1166_116680


namespace NUMINAMATH_GPT_remainder_3x_minus_6_divides_P_l1166_116666

def P(x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 8 * x^4 + 3 * x^3 - 5
def D(x : ℝ) : ℝ := 3 * x - 6

theorem remainder_3x_minus_6_divides_P :
  P 2 = 915 :=
by
  sorry

end NUMINAMATH_GPT_remainder_3x_minus_6_divides_P_l1166_116666


namespace NUMINAMATH_GPT_maximum_area_l1166_116649

variable {l w : ℝ}

theorem maximum_area (h1 : l + w = 200) (h2 : l ≥ 90) (h3 : w ≥ 50) (h4 : l ≤ 2 * w) : l * w ≤ 10000 :=
sorry

end NUMINAMATH_GPT_maximum_area_l1166_116649


namespace NUMINAMATH_GPT_remainder_identity_l1166_116600

variable {n : ℕ}

theorem remainder_identity
  (a b a_1 b_1 a_2 b_2 : ℕ)
  (ha : a = a_1 + a_2 * n)
  (hb : b = b_1 + b_2 * n) :
  (((a + b) % n = (a_1 + b_1) % n) ∧ ((a - b) % n = (a_1 - b_1) % n)) ∧ ((a * b) % n = (a_1 * b_1) % n) := by
  sorry

end NUMINAMATH_GPT_remainder_identity_l1166_116600


namespace NUMINAMATH_GPT_cubic_inequality_l1166_116643

theorem cubic_inequality (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0) (hne : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end NUMINAMATH_GPT_cubic_inequality_l1166_116643


namespace NUMINAMATH_GPT_remainder_and_division_l1166_116623

theorem remainder_and_division (n : ℕ) (h1 : n = 1680) (h2 : n % 9 = 0) : 
  1680 % 1677 = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_and_division_l1166_116623


namespace NUMINAMATH_GPT_union_of_P_and_Q_l1166_116682

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem union_of_P_and_Q : (P ∪ Q) = {x | -1 < x ∧ x < 3} := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_union_of_P_and_Q_l1166_116682


namespace NUMINAMATH_GPT_min_colors_correctness_l1166_116646

noncomputable def min_colors_no_monochromatic_cycle (n : ℕ) : ℕ :=
if n ≤ 2 then 1 else 2

theorem min_colors_correctness (n : ℕ) (h₀ : n > 0) :
  (min_colors_no_monochromatic_cycle n = 1 ∧ n ≤ 2) ∨
  (min_colors_no_monochromatic_cycle n = 2 ∧ n ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_min_colors_correctness_l1166_116646


namespace NUMINAMATH_GPT_smallest_number_of_fruits_l1166_116614

theorem smallest_number_of_fruits 
  (n_apple_slices : ℕ) (n_grapes : ℕ) (n_orange_wedges : ℕ) (n_cherries : ℕ)
  (h_apple : n_apple_slices = 18)
  (h_grape : n_grapes = 9)
  (h_orange : n_orange_wedges = 12)
  (h_cherry : n_cherries = 6)
  : ∃ (n : ℕ), n = 36 ∧ (n % n_apple_slices = 0) ∧ (n % n_grapes = 0) ∧ (n % n_orange_wedges = 0) ∧ (n % n_cherries = 0) :=
sorry

end NUMINAMATH_GPT_smallest_number_of_fruits_l1166_116614


namespace NUMINAMATH_GPT_quadratic_roots_solution_l1166_116636

theorem quadratic_roots_solution (x : ℝ) (h : x > 0) (h_roots : 7 * x^2 - 8 * x - 6 = 0) : (x = 6 / 7) ∨ (x = 1) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_solution_l1166_116636


namespace NUMINAMATH_GPT_flower_beds_fraction_l1166_116652

noncomputable def area_triangle (leg: ℝ) : ℝ := (leg * leg) / 2
noncomputable def area_rectangle (length width: ℝ) : ℝ := length * width
noncomputable def area_trapezoid (a b height: ℝ) : ℝ := ((a + b) * height) / 2

theorem flower_beds_fraction : 
  ∀ (leg len width a b height total_length: ℝ),
    a = 30 →
    b = 40 →
    height = 6 →
    total_length = 60 →
    leg = 5 →
    len = 20 →
    width = 5 →
    (area_rectangle len width + 2 * area_triangle leg) / (area_trapezoid a b height + area_rectangle len width) = 125 / 310 :=
by
  intros
  sorry

end NUMINAMATH_GPT_flower_beds_fraction_l1166_116652


namespace NUMINAMATH_GPT_solution_set_inequality_l1166_116678

theorem solution_set_inequality (a b x : ℝ) (h₀ : {x : ℝ | ax - b < 0} = {x : ℝ | 1 < x}) :
  {x : ℝ | (ax + b) * (x - 3) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1166_116678


namespace NUMINAMATH_GPT_sum_first_13_terms_l1166_116612

theorem sum_first_13_terms
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₁ : a 4 + a 10 - (a 7)^2 + 15 = 0)
  (h₂ : ∀ n : ℕ, a n > 0) :
  S 13 = 65 :=
sorry

end NUMINAMATH_GPT_sum_first_13_terms_l1166_116612


namespace NUMINAMATH_GPT_irr_sqrt6_l1166_116648

open Real

theorem irr_sqrt6 : ¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt 6 := by
  sorry

end NUMINAMATH_GPT_irr_sqrt6_l1166_116648


namespace NUMINAMATH_GPT_john_total_spending_l1166_116660

def t_shirt_price : ℝ := 20
def num_t_shirts : ℝ := 3
def t_shirt_offer_discount : ℝ := 0.50
def t_shirt_total_cost : ℝ := (2 * t_shirt_price) + (t_shirt_price * t_shirt_offer_discount)

def pants_price : ℝ := 50
def num_pants : ℝ := 2
def pants_total_cost : ℝ := pants_price * num_pants

def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def jacket_total_cost : ℝ := jacket_original_price * (1 - jacket_discount)

def hat_price : ℝ := 15

def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10
def shoes_total_cost : ℝ := shoes_original_price * (1 - shoes_discount)

def clothes_tax_rate : ℝ := 0.05
def shoes_tax_rate : ℝ := 0.08

def clothes_total_cost : ℝ := t_shirt_total_cost + pants_total_cost + jacket_total_cost + hat_price
def total_cost_before_tax : ℝ := clothes_total_cost + shoes_total_cost

def clothes_tax : ℝ := clothes_total_cost * clothes_tax_rate
def shoes_tax : ℝ := shoes_total_cost * shoes_tax_rate

def total_cost_including_tax : ℝ := total_cost_before_tax + clothes_tax + shoes_tax

theorem john_total_spending :
  total_cost_including_tax = 294.57 := by
  sorry

end NUMINAMATH_GPT_john_total_spending_l1166_116660


namespace NUMINAMATH_GPT_girls_dropped_out_l1166_116686

theorem girls_dropped_out (B_initial G_initial B_dropped G_remaining S_remaining : ℕ)
  (hB_initial : B_initial = 14)
  (hG_initial : G_initial = 10)
  (hB_dropped : B_dropped = 4)
  (hS_remaining : S_remaining = 17)
  (hB_remaining : B_initial - B_dropped = B_remaining)
  (hG_remaining : G_remaining = S_remaining - B_remaining) :
  (G_initial - G_remaining) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_girls_dropped_out_l1166_116686


namespace NUMINAMATH_GPT_infinite_expressible_terms_l1166_116663

theorem infinite_expressible_terms
  (a : ℕ → ℕ)
  (h1 : ∀ n, a n < a (n + 1)) :
  ∃ f : ℕ → ℕ, (∀ n, a (f n) = (f n).succ * a 1 + (f n).succ.succ * a 2) ∧
    ∀ i j, i ≠ j → f i ≠ f j :=
by
  sorry

end NUMINAMATH_GPT_infinite_expressible_terms_l1166_116663


namespace NUMINAMATH_GPT_range_of_a_l1166_116616

theorem range_of_a (a m : ℝ) (hp : 3 * a < m ∧ m < 4 * a) 
  (hq : 1 < m ∧ m < 3 / 2) :
  1 / 3 ≤ a ∧ a ≤ 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1166_116616


namespace NUMINAMATH_GPT_walking_time_l1166_116659

theorem walking_time (v : ℕ) (d : ℕ) (h1 : v = 10) (h2 : d = 4) : 
    ∃ (T : ℕ), T = 24 := 
by
  sorry

end NUMINAMATH_GPT_walking_time_l1166_116659


namespace NUMINAMATH_GPT_train_length_approx_500_l1166_116655

noncomputable def length_of_train (speed_km_per_hr : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  speed_m_per_s * time_sec

theorem train_length_approx_500 :
  length_of_train 120 15 = 500 :=
by
  sorry

end NUMINAMATH_GPT_train_length_approx_500_l1166_116655


namespace NUMINAMATH_GPT_arithmetic_common_difference_l1166_116639

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_common_difference_l1166_116639


namespace NUMINAMATH_GPT_divide_two_equal_parts_divide_four_equal_parts_l1166_116690

-- the figure is bounded by three semicircles
def figure_bounded_by_semicircles 
-- two have the same radius r1 
(r1 r2 r3 : ℝ) 
-- the third has twice the radius r3 = 2 * r1
(h_eq : r3 = 2 * r1) 
-- Let's denote the figure as F
(F : Type) :=
-- conditions for r1 and r2
r1 > 0 ∧ r2 = r1 ∧ r3 = 2 * r1

-- Prove the figure can be divided into two equal parts.
theorem divide_two_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 : F), H1 ≠ H2 ∧ H1 = H2 :=
sorry

-- Prove the figure can be divided into four equal parts.
theorem divide_four_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 H3 H4 : F), H1 ≠ H2 ∧ H2 ≠ H3 ∧ H3 ≠ H4 ∧ H1 = H2 ∧ H2 = H3 ∧ H3 = H4 :=
sorry

end NUMINAMATH_GPT_divide_two_equal_parts_divide_four_equal_parts_l1166_116690


namespace NUMINAMATH_GPT_souvenirs_expenses_l1166_116689

/--
  Given:
  1. K = T + 146.00
  2. T + K = 548.00
  Prove: 
  - K = 347.00
-/
theorem souvenirs_expenses (T K : ℝ) (h1 : K = T + 146) (h2 : T + K = 548) : K = 347 :=
  sorry

end NUMINAMATH_GPT_souvenirs_expenses_l1166_116689


namespace NUMINAMATH_GPT_probability_distribution_m_l1166_116607

theorem probability_distribution_m (m : ℚ) : 
  (m + m / 2 + m / 3 + m / 4 = 1) → m = 12 / 25 :=
by sorry

end NUMINAMATH_GPT_probability_distribution_m_l1166_116607


namespace NUMINAMATH_GPT_discrim_of_quadratic_eqn_l1166_116634

theorem discrim_of_quadratic_eqn : 
  let a := 3
  let b := -2
  let c := -1
  b^2 - 4 * a * c = 16 := 
by
  sorry

end NUMINAMATH_GPT_discrim_of_quadratic_eqn_l1166_116634


namespace NUMINAMATH_GPT_andy_cavity_per_candy_cane_l1166_116617

theorem andy_cavity_per_candy_cane 
  (cavities_per_candy_cane : ℝ)
  (candy_caned_from_parents : ℝ := 2)
  (candy_caned_each_teacher : ℝ := 3)
  (num_teachers : ℝ := 4)
  (allowance_factor : ℝ := 1/7)
  (total_cavities : ℝ := 16) :
  let total_given_candy : ℝ := candy_caned_from_parents + candy_caned_each_teacher * num_teachers
  let total_bought_candy : ℝ := allowance_factor * total_given_candy
  let total_candy : ℝ := total_given_candy + total_bought_candy
  total_candy / total_cavities = cavities_per_candy_cane :=
by
  sorry

end NUMINAMATH_GPT_andy_cavity_per_candy_cane_l1166_116617


namespace NUMINAMATH_GPT_bill_new_profit_percentage_l1166_116632

theorem bill_new_profit_percentage 
  (original_SP : ℝ)
  (profit_percent : ℝ)
  (increment : ℝ)
  (CP : ℝ)
  (CP_new : ℝ)
  (SP_new : ℝ)
  (Profit_new : ℝ)
  (new_profit_percent : ℝ) :
  original_SP = 439.99999999999966 →
  profit_percent = 0.10 →
  increment = 28 →
  CP = original_SP / (1 + profit_percent) →
  CP_new = CP * (1 - profit_percent) →
  SP_new = original_SP + increment →
  Profit_new = SP_new - CP_new →
  new_profit_percent = (Profit_new / CP_new) * 100 →
  new_profit_percent = 30 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_bill_new_profit_percentage_l1166_116632


namespace NUMINAMATH_GPT_visitors_surveyed_l1166_116657

-- Given definitions
def total_visitors : ℕ := 400
def visitors_not_enjoyed_nor_understood : ℕ := 100
def E := total_visitors / 2
def U := total_visitors / 2

-- Using condition that 3/4th visitors enjoyed and understood
def enjoys_and_understands := (3 * total_visitors) / 4

-- Assert the equivalence of total number of visitors calculation
theorem visitors_surveyed:
  total_visitors = enjoys_and_understands + visitors_not_enjoyed_nor_understood :=
by
  sorry

end NUMINAMATH_GPT_visitors_surveyed_l1166_116657


namespace NUMINAMATH_GPT_eggs_problem_solution_l1166_116624

theorem eggs_problem_solution :
  ∃ (n x : ℕ), 
  (120 * n = 206 * x) ∧
  (n = 103) ∧
  (x = 60) :=
by sorry

end NUMINAMATH_GPT_eggs_problem_solution_l1166_116624


namespace NUMINAMATH_GPT_circle_intersection_range_l1166_116654

theorem circle_intersection_range (a : ℝ) :
  (0 < a ∧ a < 2 * Real.sqrt 2) ∨ (-2 * Real.sqrt 2 < a ∧ a < 0) ↔
  (let C := { p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - a) ^ 2 = 4 };
   let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4 };
   ∀ p, p ∈ C → p ∈ O) :=
sorry

end NUMINAMATH_GPT_circle_intersection_range_l1166_116654


namespace NUMINAMATH_GPT_quadratic_completeness_l1166_116674

noncomputable def quad_eqn : Prop :=
  ∃ b c : ℤ, (∀ x : ℝ, (x^2 - 10 * x + 15 = 0) ↔ ((x + b)^2 = c)) ∧ b + c = 5

theorem quadratic_completeness : quad_eqn :=
sorry

end NUMINAMATH_GPT_quadratic_completeness_l1166_116674


namespace NUMINAMATH_GPT_sam_gave_2_puppies_l1166_116687

theorem sam_gave_2_puppies (original_puppies given_puppies remaining_puppies : ℕ) 
  (h1 : original_puppies = 6) (h2 : remaining_puppies = 4) :
  given_puppies = original_puppies - remaining_puppies := by 
  sorry

end NUMINAMATH_GPT_sam_gave_2_puppies_l1166_116687


namespace NUMINAMATH_GPT_find_square_tiles_l1166_116615

variable {s p : ℕ}

theorem find_square_tiles (h1 : s + p = 30) (h2 : 4 * s + 5 * p = 110) : s = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_square_tiles_l1166_116615


namespace NUMINAMATH_GPT_remainder_of_division_l1166_116641

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end NUMINAMATH_GPT_remainder_of_division_l1166_116641


namespace NUMINAMATH_GPT_blue_crayons_l1166_116662

variables (B G : ℕ)

theorem blue_crayons (h1 : 24 = 8 + B + G + 6) (h2 : G = (2 / 3) * B) : B = 6 :=
by 
-- This is where the proof would go
sorry

end NUMINAMATH_GPT_blue_crayons_l1166_116662


namespace NUMINAMATH_GPT_bicycle_stock_decrease_l1166_116672

-- Define the conditions and the problem
theorem bicycle_stock_decrease (m : ℕ) (jan_to_oct_decrease june_to_oct_decrease monthly_decrease : ℕ) 
  (h1: monthly_decrease = 4)
  (h2: jan_to_oct_decrease = 36)
  (h3: june_to_oct_decrease = 4 * monthly_decrease):
  m * monthly_decrease = jan_to_oct_decrease - june_to_oct_decrease → m = 5 := 
by
  sorry

end NUMINAMATH_GPT_bicycle_stock_decrease_l1166_116672


namespace NUMINAMATH_GPT_sequences_of_length_15_l1166_116609

def odd_runs_of_A_even_runs_of_B (n : ℕ) : ℕ :=
  (if n = 1 then 1 else 0) + (if n = 2 then 1 else 0)

theorem sequences_of_length_15 : 
  odd_runs_of_A_even_runs_of_B 15 = 47260 :=
  sorry

end NUMINAMATH_GPT_sequences_of_length_15_l1166_116609


namespace NUMINAMATH_GPT_dan_blue_marbles_l1166_116602

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end NUMINAMATH_GPT_dan_blue_marbles_l1166_116602


namespace NUMINAMATH_GPT_annette_weights_more_l1166_116699

variable (A C S B : ℝ)

theorem annette_weights_more :
  A + C = 95 ∧
  C + S = 87 ∧
  A + S = 97 ∧
  C + B = 100 ∧
  A + C + B = 155 →
  A - S = 8 := by
  sorry

end NUMINAMATH_GPT_annette_weights_more_l1166_116699


namespace NUMINAMATH_GPT_valid_three_digit_numbers_count_l1166_116668

def count_three_digit_numbers : ℕ := 900

def count_invalid_numbers : ℕ := (90 + 90 - 9)

def count_valid_three_digit_numbers : ℕ := 900 - (90 + 90 - 9)

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 729 :=
by
  show 900 - (90 + 90 - 9) = 729
  sorry

end NUMINAMATH_GPT_valid_three_digit_numbers_count_l1166_116668


namespace NUMINAMATH_GPT_hansel_album_duration_l1166_116619

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end NUMINAMATH_GPT_hansel_album_duration_l1166_116619


namespace NUMINAMATH_GPT_range_of_m_l1166_116603

-- Definitions used to state conditions of the problem.
def fractional_equation (m x : ℝ) : Prop := (m / (2 * x - 1)) + 2 = 0
def positive_solution (x : ℝ) : Prop := x > 0

-- The Lean 4 theorem statement
theorem range_of_m (m x : ℝ) (h : fractional_equation m x) (hx : positive_solution x) : m < 2 ∧ m ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1166_116603


namespace NUMINAMATH_GPT_hulk_jump_distance_exceeds_1000_l1166_116681

theorem hulk_jump_distance_exceeds_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 3^m ≤ 1000) ∧ 3^n > 1000 :=
sorry

end NUMINAMATH_GPT_hulk_jump_distance_exceeds_1000_l1166_116681


namespace NUMINAMATH_GPT_find_a1_plus_a2_l1166_116670

theorem find_a1_plus_a2 (x : ℝ) (a0 a1 a2 a3 : ℝ) 
  (h : (1 - 2/x)^3 = a0 + a1 * (1/x) + a2 * (1/x)^2 + a3 * (1/x)^3) : 
  a1 + a2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_plus_a2_l1166_116670


namespace NUMINAMATH_GPT_percentage_of_salt_in_second_solution_l1166_116664

-- Define the data and initial conditions
def original_solution_salt_percentage := 0.15
def replaced_solution_salt_percentage (x: ℝ) := x
def resulting_solution_salt_percentage := 0.16

-- State the question as a theorem
theorem percentage_of_salt_in_second_solution (S : ℝ) (x : ℝ) :
  0.15 * S - 0.0375 * S + x * (S / 4) = 0.16 * S → x = 0.19 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_of_salt_in_second_solution_l1166_116664


namespace NUMINAMATH_GPT_simplify_one_simplify_two_simplify_three_simplify_four_l1166_116695

-- (1) Prove that (1 / 2) * sqrt(4 / 7) = sqrt(7) / 7
theorem simplify_one : (1 / 2) * Real.sqrt (4 / 7) = Real.sqrt 7 / 7 := sorry

-- (2) Prove that sqrt(20 ^ 2 - 15 ^ 2) = 5 * sqrt(7)
theorem simplify_two : Real.sqrt (20 ^ 2 - 15 ^ 2) = 5 * Real.sqrt 7 := sorry

-- (3) Prove that sqrt((32 * 9) / 25) = (12 * sqrt(2)) / 5
theorem simplify_three : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := sorry

-- (4) Prove that sqrt(22.5) = (3 * sqrt(10)) / 2
theorem simplify_four : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := sorry

end NUMINAMATH_GPT_simplify_one_simplify_two_simplify_three_simplify_four_l1166_116695


namespace NUMINAMATH_GPT_car_speed_first_hour_l1166_116633

theorem car_speed_first_hour (speed1 speed2 avg_speed : ℕ) (h1 : speed2 = 70) (h2 : avg_speed = 95) :
  (2 * avg_speed) = speed1 + speed2 → speed1 = 120 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l1166_116633


namespace NUMINAMATH_GPT_evaluate_g_at_3_l1166_116631

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem evaluate_g_at_3 : g 3 = 79 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l1166_116631


namespace NUMINAMATH_GPT_integer_divisibility_l1166_116650

theorem integer_divisibility (m n : ℕ) (hm : m > 1) (hn : n > 1) (h1 : n ∣ 4^m - 1) (h2 : 2^m ∣ n - 1) : n = 2^m + 1 :=
by sorry

end NUMINAMATH_GPT_integer_divisibility_l1166_116650


namespace NUMINAMATH_GPT_binomial_inequality_l1166_116658

theorem binomial_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end NUMINAMATH_GPT_binomial_inequality_l1166_116658


namespace NUMINAMATH_GPT_solution_inequality_set_l1166_116618

-- Define the inequality condition
def inequality (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

-- Define the interval solution set
def solution_set := Set.Icc (-2 : ℝ) 5

-- The statement that we want to prove
theorem solution_inequality_set : {x : ℝ | inequality x} = solution_set :=
  sorry

end NUMINAMATH_GPT_solution_inequality_set_l1166_116618


namespace NUMINAMATH_GPT_each_spider_eats_seven_bugs_l1166_116630

theorem each_spider_eats_seven_bugs (initial_bugs : ℕ) (reduction_rate : ℚ) (spiders_introduced : ℕ) (bugs_left : ℕ) (result : ℕ)
  (h1 : initial_bugs = 400)
  (h2 : reduction_rate = 0.80)
  (h3 : spiders_introduced = 12)
  (h4 : bugs_left = 236)
  (h5 : result = initial_bugs * (4 / 5) - bugs_left) :
  (result / spiders_introduced) = 7 :=
by
  sorry

end NUMINAMATH_GPT_each_spider_eats_seven_bugs_l1166_116630


namespace NUMINAMATH_GPT_cost_difference_per_square_inch_l1166_116676

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end NUMINAMATH_GPT_cost_difference_per_square_inch_l1166_116676


namespace NUMINAMATH_GPT_striped_octopus_has_eight_legs_l1166_116688

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_striped_octopus_has_eight_legs_l1166_116688


namespace NUMINAMATH_GPT_remaining_lemon_heads_after_eating_l1166_116621

-- Assume initial number of lemon heads is given
variables (initial_lemon_heads : ℕ)

-- Patricia eats 15 lemon heads
def remaining_lemon_heads (initial_lemon_heads : ℕ) : ℕ :=
  initial_lemon_heads - 15

theorem remaining_lemon_heads_after_eating :
  ∀ (initial_lemon_heads : ℕ), remaining_lemon_heads initial_lemon_heads = initial_lemon_heads - 15 :=
by
  intros
  rfl

end NUMINAMATH_GPT_remaining_lemon_heads_after_eating_l1166_116621


namespace NUMINAMATH_GPT_nadine_white_pebbles_l1166_116679

variable (W R : ℝ)

theorem nadine_white_pebbles :
  (R = 1/2 * W) →
  (W + R = 30) →
  W = 20 :=
by
  sorry

end NUMINAMATH_GPT_nadine_white_pebbles_l1166_116679


namespace NUMINAMATH_GPT_repeating_decimal_fraction_value_l1166_116613

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d

theorem repeating_decimal_fraction_value :
  repeating_decimal_to_fraction (73 / 100 + 246 / 999000) = 731514 / 999900 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_value_l1166_116613


namespace NUMINAMATH_GPT_find_p_current_age_l1166_116683

theorem find_p_current_age (x p q : ℕ) (h1 : p - 3 = 4 * x) (h2 : q - 3 = 3 * x) (h3 : (p + 6) / (q + 6) = 7 / 6) : p = 15 := 
sorry

end NUMINAMATH_GPT_find_p_current_age_l1166_116683


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_q_l1166_116626

theorem p_necessary_not_sufficient_q (x : ℝ) : (|x| = 2) → (x = 2) → (|x| = 2 ∧ (x ≠ 2 ∨ x = -2)) := by
  intros h_p h_q
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_q_l1166_116626


namespace NUMINAMATH_GPT_simple_interest_years_l1166_116685

theorem simple_interest_years
  (CI : ℝ)
  (SI : ℝ)
  (p1 : ℝ := 4000) (r1 : ℝ := 0.10) (t1 : ℝ := 2)
  (p2 : ℝ := 1750) (r2 : ℝ := 0.08)
  (h1 : CI = p1 * (1 + r1) ^ t1 - p1)
  (h2 : SI = CI / 2)
  (h3 : SI = p2 * r2 * t2) :
  t2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_years_l1166_116685


namespace NUMINAMATH_GPT_max_fraction_sum_l1166_116606

theorem max_fraction_sum (a b c : ℝ) 
  (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum: a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_fraction_sum_l1166_116606


namespace NUMINAMATH_GPT_income_day_3_is_750_l1166_116656

-- Define the given incomes for the specific days
def income_day_1 : ℝ := 250
def income_day_2 : ℝ := 400
def income_day_4 : ℝ := 400
def income_day_5 : ℝ := 500

-- Define the total number of days and the average income over these days
def total_days : ℝ := 5
def average_income : ℝ := 460

-- Define the total income based on the average
def total_income : ℝ := total_days * average_income

-- Define the income on the third day
def income_day_3 : ℝ := total_income - (income_day_1 + income_day_2 + income_day_4 + income_day_5)

-- Claim: The income on the third day is $750
theorem income_day_3_is_750 : income_day_3 = 750 := by
  sorry

end NUMINAMATH_GPT_income_day_3_is_750_l1166_116656


namespace NUMINAMATH_GPT_greatest_integer_difference_l1166_116677

theorem greatest_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  ∃ d : ℤ, d = y - x ∧ ∀ z, 4 < z ∧ z < 8 ∧ 8 < y ∧ y < 12 → (y - z ≤ d) :=
sorry

end NUMINAMATH_GPT_greatest_integer_difference_l1166_116677


namespace NUMINAMATH_GPT_divide_money_equally_l1166_116692

-- Length of the road built by companies A, B, and total length of the road
def length_A : ℕ := 6
def length_B : ℕ := 10
def total_length : ℕ := 16

-- Money contributed by company C
def money_C : ℕ := 16 * 10^6

-- The equal contribution each company should finance
def equal_contribution := total_length / 3

-- Deviations from the expected length for firms A and B
def deviation_A := length_A - (total_length / 3)
def deviation_B := length_B - (total_length / 3)

-- The ratio based on the deviations to divide the money
def ratio_A := deviation_A * (total_length / (deviation_A + deviation_B))
def ratio_B := deviation_B * (total_length / (deviation_A + deviation_B))

-- The amount of money firms A and B should receive, respectively
def money_A := money_C * ratio_A / total_length
def money_B := money_C * ratio_B / total_length

-- Theorem statement
theorem divide_money_equally : money_A = 2 * 10^6 ∧ money_B = 14 * 10^6 :=
by 
  sorry

end NUMINAMATH_GPT_divide_money_equally_l1166_116692


namespace NUMINAMATH_GPT_remainder_sum_div7_l1166_116620

theorem remainder_sum_div7 (a b c : ℕ) (h1 : a * b * c ≡ 2 [MOD 7])
  (h2 : 3 * c ≡ 4 [MOD 7])
  (h3 : 4 * b ≡ 2 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_div7_l1166_116620


namespace NUMINAMATH_GPT_reachable_target_l1166_116696

-- Define the initial state of the urn
def initial_urn_state : (ℕ × ℕ) := (150, 50)

-- Define the operations as changes in counts of black and white marbles
def operation1 (state : ℕ × ℕ) := (state.1 - 2, state.2)
def operation2 (state : ℕ × ℕ) := (state.1 - 1, state.2)
def operation3 (state : ℕ × ℕ) := (state.1, state.2 - 2)
def operation4 (state : ℕ × ℕ) := (state.1 + 2, state.2 - 3)

-- Define a predicate that a state can be reached from the initial state
def reachable (target : ℕ × ℕ) : Prop :=
  ∃ n1 n2 n3 n4 : ℕ, 
    operation1^[n1] (operation2^[n2] (operation3^[n3] (operation4^[n4] initial_urn_state))) = target

-- The theorem to be proved
theorem reachable_target : reachable (1, 2) :=
sorry

end NUMINAMATH_GPT_reachable_target_l1166_116696


namespace NUMINAMATH_GPT_find_percentage_decrease_in_fourth_month_l1166_116608

theorem find_percentage_decrease_in_fourth_month
  (P0 : ℝ) (P1 : ℝ) (P2 : ℝ) (P3 : ℝ) (x : ℝ) :
  (P0 = 100) →
  (P1 = P0 + 0.30 * P0) →
  (P2 = P1 - 0.15 * P1) →
  (P3 = P2 + 0.10 * P2) →
  (P0 = P3 - x / 100 * P3) →
  x = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_decrease_in_fourth_month_l1166_116608


namespace NUMINAMATH_GPT_quadratic_value_at_point_a_l1166_116667

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

open Real

theorem quadratic_value_at_point_a
  (a b c : ℝ)
  (axis : ℝ)
  (sym : ∀ x, quadratic a b c (2 * axis - x) = quadratic a b c x)
  (at_zero : quadratic a b c 0 = -3) :
  quadratic a b c 20 = -3 := by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_quadratic_value_at_point_a_l1166_116667


namespace NUMINAMATH_GPT_intersection_eq_N_l1166_116693

def U := Set ℝ                                        -- Universal set U = ℝ
def M : Set ℝ := {x | x ≥ 0}                         -- Set M = {x | x ≥ 0}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}                 -- Set N = {x | 0 ≤ x ≤ 1}

theorem intersection_eq_N : M ∩ N = N := by
  sorry

end NUMINAMATH_GPT_intersection_eq_N_l1166_116693


namespace NUMINAMATH_GPT_smallest_n_l1166_116675

theorem smallest_n (n : ℕ) (h : 0 < n) (h1 : 813 * n % 30 = 1224 * n % 30) : n = 10 := 
sorry

end NUMINAMATH_GPT_smallest_n_l1166_116675


namespace NUMINAMATH_GPT_frog_vertical_boundary_prob_l1166_116694

-- Define the type of points on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define the type of the rectangle
structure Rectangle where
  left_bottom : Point
  right_top : Point

-- Conditions
def start_point : Point := ⟨2, 3⟩
def boundary : Rectangle := ⟨⟨0, 0⟩, ⟨5, 5⟩⟩

-- Define the probability function
noncomputable def P (p : Point) : ℚ := sorry

-- Symmetry relations and recursive relations
axiom symmetry_P23 : P ⟨2, 3⟩ = P ⟨3, 3⟩
axiom symmetry_P22 : P ⟨2, 2⟩ = P ⟨3, 2⟩
axiom recursive_P23 : P ⟨2, 3⟩ = 1 / 4 + 1 / 4 * P ⟨2, 2⟩ + 1 / 4 * P ⟨1, 3⟩ + 1 / 4 * P ⟨3, 3⟩

-- Main Theorem
theorem frog_vertical_boundary_prob :
  P start_point = 2 / 3 := sorry

end NUMINAMATH_GPT_frog_vertical_boundary_prob_l1166_116694


namespace NUMINAMATH_GPT_convert_base_8_to_7_l1166_116645

def convert_base_8_to_10 (n : Nat) : Nat :=
  let d2 := n / 100 % 10
  let d1 := n / 10 % 10
  let d0 := n % 10
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def convert_base_10_to_7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else 
    let rec helper (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else helper (n / 7) ((n % 7) :: acc)
    helper n []

def represent_in_base_7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem convert_base_8_to_7 :
  represent_in_base_7 (convert_base_10_to_7 (convert_base_8_to_10 653)) = 1150 :=
by
  sorry

end NUMINAMATH_GPT_convert_base_8_to_7_l1166_116645


namespace NUMINAMATH_GPT_initial_rate_of_commission_is_4_l1166_116698

noncomputable def initial_commission_rate (B : ℝ) (x : ℝ) : Prop :=
  B * (x / 100) = 0.8 * B * (5 / 100)

theorem initial_rate_of_commission_is_4 (B : ℝ) (hB : B > 0) :
  initial_commission_rate B 4 :=
by
  unfold initial_commission_rate
  sorry

end NUMINAMATH_GPT_initial_rate_of_commission_is_4_l1166_116698


namespace NUMINAMATH_GPT_smallest_angle_of_triangle_l1166_116601

theorem smallest_angle_of_triangle (y : ℝ) (h : 40 + 70 + y = 180) : 
  ∃ smallest_angle : ℝ, smallest_angle = 40 ∧ smallest_angle = min 40 (min 70 y) := 
by
  use 40
  sorry

end NUMINAMATH_GPT_smallest_angle_of_triangle_l1166_116601


namespace NUMINAMATH_GPT_simplify_expr_l1166_116605

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1166_116605


namespace NUMINAMATH_GPT_total_lives_after_third_level_l1166_116653

def initial_lives : ℕ := 2

def extra_lives_first_level : ℕ := 6
def modifier_first_level (lives : ℕ) : ℕ := lives / 2

def extra_lives_second_level : ℕ := 11
def challenge_second_level (lives : ℕ) : ℕ := lives - 3

def reward_third_level (lives_first_two_levels : ℕ) : ℕ := 2 * lives_first_two_levels

theorem total_lives_after_third_level :
  let lives_first_level := modifier_first_level extra_lives_first_level
  let lives_after_first_level := initial_lives + lives_first_level
  let lives_second_level := challenge_second_level extra_lives_second_level
  let lives_after_second_level := lives_after_first_level + lives_second_level
  let total_gained_lives_first_two_levels := lives_first_level + lives_second_level
  let third_level_reward := reward_third_level total_gained_lives_first_two_levels
  lives_after_second_level + third_level_reward = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_lives_after_third_level_l1166_116653


namespace NUMINAMATH_GPT_deepak_present_age_l1166_116661

theorem deepak_present_age (x : ℕ) (rahul deepak rohan : ℕ) 
  (h_ratio : rahul = 5 * x ∧ deepak = 2 * x ∧ rohan = 3 * x)
  (h_rahul_future_age : rahul + 8 = 28) :
  deepak = 8 := 
by
  sorry

end NUMINAMATH_GPT_deepak_present_age_l1166_116661


namespace NUMINAMATH_GPT_fraction_of_students_with_buddy_l1166_116622

theorem fraction_of_students_with_buddy (t s : ℕ) (h1 : (t / 4) = (3 * s / 5)) :
  (t / 4 + 3 * s / 5) / (t + s) = 6 / 17 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_buddy_l1166_116622


namespace NUMINAMATH_GPT_residue_of_11_pow_2048_mod_19_l1166_116644

theorem residue_of_11_pow_2048_mod_19 :
  (11 ^ 2048) % 19 = 16 := 
by
  sorry

end NUMINAMATH_GPT_residue_of_11_pow_2048_mod_19_l1166_116644


namespace NUMINAMATH_GPT_remainder_of_76_pow_k_mod_7_is_6_l1166_116629

theorem remainder_of_76_pow_k_mod_7_is_6 (k : ℕ) (hk : k % 2 = 1) : (76 ^ k) % 7 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_of_76_pow_k_mod_7_is_6_l1166_116629


namespace NUMINAMATH_GPT_unique_root_exists_maximum_value_lnx_l1166_116640

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x

theorem unique_root_exists (k : ℝ) :
  ∃ a, a = 1 ∧ (∃ x ∈ Set.Ioo k (k+1), f x = g x) :=
sorry

theorem maximum_value_lnx (p q : ℝ) :
  (∃ x, (x = min p q) ∧ Real.log x = ( 4 / Real.exp 2 )) :=
sorry

end NUMINAMATH_GPT_unique_root_exists_maximum_value_lnx_l1166_116640


namespace NUMINAMATH_GPT_total_ice_cream_sales_l1166_116610

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end NUMINAMATH_GPT_total_ice_cream_sales_l1166_116610


namespace NUMINAMATH_GPT_unique_combined_friends_count_l1166_116637

theorem unique_combined_friends_count 
  (james_friends : ℕ)
  (susan_friends : ℕ)
  (john_multiplier : ℕ)
  (shared_friends : ℕ)
  (maria_shared_friends : ℕ)
  (maria_friends : ℕ)
  (h_james : james_friends = 90)
  (h_susan : susan_friends = 50)
  (h_john : ∃ (john_friends : ℕ), john_friends = john_multiplier * susan_friends ∧ john_multiplier = 4)
  (h_shared : shared_friends = 35)
  (h_maria_shared : maria_shared_friends = 10)
  (h_maria : maria_friends = 80) :
  ∃ (total_unique_friends : ℕ), total_unique_friends = 325 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_unique_combined_friends_count_l1166_116637


namespace NUMINAMATH_GPT_remainder_127_14_l1166_116642

theorem remainder_127_14 : ∃ r : ℤ, r = 127 - (14 * 9) ∧ r = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_127_14_l1166_116642


namespace NUMINAMATH_GPT_find_k_l1166_116647

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k

theorem find_k (k : ℝ) : 
  (f 3 - g 3 k = 6) → k = -23/3 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l1166_116647


namespace NUMINAMATH_GPT_find_x_l1166_116627

theorem find_x (x : ℝ) : x * 2.25 - (5 * 0.85) / 2.5 = 5.5 → x = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1166_116627


namespace NUMINAMATH_GPT_damaged_books_l1166_116697

theorem damaged_books (O D : ℕ) (h1 : O = 6 * D - 8) (h2 : D + O = 69) : D = 11 :=
by
  sorry

end NUMINAMATH_GPT_damaged_books_l1166_116697


namespace NUMINAMATH_GPT_base_conversion_l1166_116638

theorem base_conversion (b2_to_b10_step : 101101 = 1 * 2 ^ 5 + 0 * 2 ^ 4 + 1 * 2 ^ 3 + 1 * 2 ^ 2 + 0 * 2 + 1)
  (b10_to_b7_step1 : 45 / 7 = 6) (b10_to_b7_step2 : 45 % 7 = 3) (b10_to_b7_step3 : 6 / 7 = 0) (b10_to_b7_step4 : 6 % 7 = 6) :
  101101 = 45 ∧ 45 = 63 :=
by {
  -- Conversion steps from the proof will be filled in here
  sorry
}

end NUMINAMATH_GPT_base_conversion_l1166_116638


namespace NUMINAMATH_GPT_car_speed_l1166_116669

theorem car_speed (t_60 : ℝ := 60) (t_12 : ℝ := 12) (t_dist : ℝ := 1) :
  ∃ v : ℝ, v = 50 ∧ (t_60 / 60 + t_12 = 3600 / v) := 
by
  sorry

end NUMINAMATH_GPT_car_speed_l1166_116669


namespace NUMINAMATH_GPT_remainder_of_sum_l1166_116651

theorem remainder_of_sum (h1 : 9375 % 5 = 0) (h2 : 9376 % 5 = 1) (h3 : 9377 % 5 = 2) (h4 : 9378 % 5 = 3) :
  (9375 + 9376 + 9377 + 9378) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l1166_116651


namespace NUMINAMATH_GPT_payment_to_Y_is_227_27_l1166_116665

-- Define the conditions
def total_payment_per_week (x y : ℝ) : Prop :=
  x + y = 500

def x_payment_is_120_percent_of_y (x y : ℝ) : Prop :=
  x = 1.2 * y

-- Formulate the problem as a theorem to be proven
theorem payment_to_Y_is_227_27 (Y : ℝ) (X : ℝ) 
  (h1 : total_payment_per_week X Y) 
  (h2 : x_payment_is_120_percent_of_y X Y) : 
  Y = 227.27 :=
by
  sorry

end NUMINAMATH_GPT_payment_to_Y_is_227_27_l1166_116665


namespace NUMINAMATH_GPT_radius_of_circle_l1166_116611

theorem radius_of_circle (A C : ℝ) (h1 : A = π * (r : ℝ)^2) (h2 : C = 2 * π * r) (h3 : A / C = 10) :
  r = 20 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1166_116611
