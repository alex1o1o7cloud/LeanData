import Mathlib
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Combinatorics.Combination
import Mathlib.Combinatorics.Combinations
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Choose.Factorization
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob.Prob
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Induction
import Mathlib.Tactic.Linarith

namespace triangle_cotangent_identity_l102_102386

theorem triangle_cotangent_identity {A B C : ℝ} 
  {a b c : ℝ} (hA : 0 < A ∧ A < π) 
  (hB : 0 < B ∧ B < π) 
  (hC : 0 < C ∧ C < π) 
  (h_sum: A + B + C = π) 
  (h_sides: 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_cos_a: a^2 = b^2 + c^2 - 2*b*c*cos A) 
  (h_cos_c: c^2 = a^2 + b^2 - 2*a*b*cos C):
  a^2 + c^2 = 2 * b^2 ↔ cot A + cot C = 2 * cot B := 
by
  sorry

end triangle_cotangent_identity_l102_102386


namespace lewis_speed_l102_102390

theorem lewis_speed
  (v : ℕ)
  (john_speed : ℕ := 40)
  (distance_AB : ℕ := 240)
  (meeting_distance : ℕ := 160)
  (time_john_to_meeting : ℕ := meeting_distance / john_speed)
  (distance_lewis_traveled : ℕ := distance_AB + (distance_AB - meeting_distance))
  (v_eq : v = distance_lewis_traveled / time_john_to_meeting) :
  v = 80 :=
by
  sorry

end lewis_speed_l102_102390


namespace opposite_number_of_sqrt_of_9_is_neg3_l102_102863

theorem opposite_number_of_sqrt_of_9_is_neg3 :
  - (Real.sqrt 9) = -3 :=
by
  -- The proof is omitted as required.
  sorry

end opposite_number_of_sqrt_of_9_is_neg3_l102_102863


namespace gamma_donuts_received_l102_102245

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end gamma_donuts_received_l102_102245


namespace probability_union_l102_102878

noncomputable def faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def odd_faces : Finset ℕ := {1, 3, 5}
noncomputable def faces_not_exceeding_3 : Finset ℕ := {1, 2, 3}

def event (s : Finset ℕ) : ℙ := Finset.card s / Finset.card faces

def P_A : ℙ := event odd_faces
def P_B : ℙ := event faces_not_exceeding_3
def P_A_inter_B : ℙ := event (odd_faces ∩ faces_not_exceeding_3)

theorem probability_union (A B : Finset ℕ) (P_A : ℙ) (P_B : ℙ) (P_A_inter_B : ℙ) :
  P_A = 1 / 2 ∧ P_B = 1 / 2 ∧ P_A_inter_B = 1 / 3 → event (odd_faces ∪ faces_not_exceeding_3) = 2 / 3 :=
by
  sorry

end probability_union_l102_102878


namespace levi_to_juwan_ratio_l102_102816

-- Let's define the noncomputable nature where necessary
noncomputable def total_cans := 99
noncomputable def solomon_cans := 66

-- Given conditions
def juwan_cans := solomon_cans / 3
def levi_cans := total_cans - solomon_cans - juwan_cans

-- Expected ratio
def expected_ratio := 1 / 2

-- Proof goal statement
theorem levi_to_juwan_ratio :
  (levi_cans : ℚ) / (juwan_cans : ℚ) = expected_ratio :=
by
  sorry -- Proof omitted as instructed

end levi_to_juwan_ratio_l102_102816


namespace range_of_t_l102_102408

noncomputable def f : ℝ → ℝ := sorry

axiom f_diff : ∀ x : ℝ, differentiable_at ℝ f x
axiom f_condition : ∀ x : ℝ, f(x) - f(-x) = 2 * real.sin x
axiom f_prime_condition : ∀ x ∈ set.Ici (0 : ℝ), deriv f x > real.cos x

theorem range_of_t (t : ℝ) : f(π/2 - t) - f(t) > real.cos t - real.sin t → t < π/4 :=
begin
  sorry
end

end range_of_t_l102_102408


namespace rowing_upstream_speed_l102_102923

theorem rowing_upstream_speed 
  (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)
  (hyp1 : V_m = 30)
  (hyp2 : V_downstream = 35) :
  V_upstream = V_m - (V_downstream - V_m) := 
  sorry

end rowing_upstream_speed_l102_102923


namespace selling_price_before_brokerage_l102_102908

-- Definitions for conditions
def cash_realized : ℝ := 106.25
def brokerage_rate : ℝ := 1 / 400

-- The statement to prove
theorem selling_price_before_brokerage :
  let SP := 106.25 * (400 / 399)
  SP = 106.65 :=
by
  sorry

end selling_price_before_brokerage_l102_102908


namespace problem1_problem2_problem3_l102_102474

-- Definitions
def num_students := 8
def num_males := 4
def num_females := 4

-- Problem statements
-- (1) Number of arrangements where male student A and female student B stand next to each other is 10080
theorem problem1 (A B : Fin num_students) (h1 : A < 4) (h2 : 4 ≤ B ∧ B < 8) : 
    ((num_males + num_females) - 1)! * 2 = 10080 :=
by sorry

-- (2) Number of arrangements where the order of male student A and female student B is fixed is 20160
theorem problem2 : 
    num_students! / 2 = 20160 :=
by sorry

-- (3) Number of arrangements where female student A does not stand at either end, and among the 4 male students, exactly two of them stand next to each other is 13824
theorem problem3 (A : Fin num_females) :
    3! * C(num_males, 2) * 2! * (num_males - 1)! * 5 = 13824 :=
by sorry

end problem1_problem2_problem3_l102_102474


namespace max_sqrt_sum_l102_102401

theorem max_sqrt_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a + b + 9*c^2 = 1) : 
  (sqrt a + sqrt b + sqrt (3) * c) ≤ (sqrt (7 / 3)) :=
sorry

end max_sqrt_sum_l102_102401


namespace correct_expression_l102_102950

variables {a b c : ℝ}

theorem correct_expression :
  -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b :=
by
  sorry

end correct_expression_l102_102950


namespace Janine_total_pages_read_l102_102755

theorem Janine_total_pages_read :
  let last_month_books := [(3, 12), (2, 15)]
  let this_month_books := [(1, 20), (4, 25), (2, 30), (1, 35)]
  let total_last_month := last_month_books.sum (λ bp, bp.1 * bp.2)
  let total_this_month := this_month_books.sum (λ bp, bp.1 * bp.2)
  total_last_month + total_this_month = 281 :=
by
  sorry

end Janine_total_pages_read_l102_102755


namespace transformed_quadratic_h_eq_three_l102_102123

noncomputable def given_quadratic (a b c : ℝ) (x : ℝ) :=
  a * x^2 + b * x + c

noncomputable def transformed_quadratic (a b c : ℝ) (x : ℝ) :=
  4 * a * x^2 + 4 * b * x + 4 * c

theorem transformed_quadratic_h_eq_three :
  ∀ (a b c : ℝ),
    given_quadratic a b c = (5 * (x - 3)^2 + 15) →
    (exists (n h k : ℝ), transformed_quadratic a b c = n * (x - h)^2 + k ∧ h = 3) :=
by
  intros a b c h_eq

  sorry

end transformed_quadratic_h_eq_three_l102_102123


namespace pre_bought_ticket_price_l102_102881

variable (P : ℕ)

theorem pre_bought_ticket_price :
  (20 * P = 6000 - 2900) → P = 155 :=
by
  intro h
  sorry

end pre_bought_ticket_price_l102_102881


namespace trapezoid_diagonal_length_l102_102748

variable (A B C D : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [HasDist A] [HasDist B] [HasDist C] [HasDist D]

def is_trapezoid (AD BC AB CD : ℝ) : Prop := 
  AD = 2 ∧ BC = 1 ∧ AB = 1 ∧ CD = 1

theorem trapezoid_diagonal_length (h : is_trapezoid dist(A, D) dist(B, C) dist(A, B) dist(C, D)) :
  dist(A, C) = real.sqrt 3 :=
sorry

end trapezoid_diagonal_length_l102_102748


namespace max_value_of_f_angle_C_magnitude_l102_102323

open Real

def f (x : ℝ) := sin x + cos (x - π / 6)

theorem max_value_of_f : ∀ x ∈ Icc (-π / 2) (π / 2), f x ≤ sqrt 3 :=
sorry

noncomputable def triangle (A B C a b : ℝ) :=
  B = 2 * A ∧
  b = 2 * a * f (A - π / 6)

theorem angle_C_magnitude (A B C a b : ℝ) (h1 : triangle A B C a b) : C = π / 2 :=
sorry

end max_value_of_f_angle_C_magnitude_l102_102323


namespace ellipse_standard_eq_l102_102678

theorem ellipse_standard_eq
  (e : ℝ) (a b : ℝ) (h1 : e = 1 / 2) (h2 : 2 * a = 4) (h3 : b^2 = a^2 - (a * e)^2)
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) ↔
    ( ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ) :=
by
  sorry

end ellipse_standard_eq_l102_102678


namespace internship_company_selection_exactly_one_same_l102_102882

theorem internship_company_selection_exactly_one_same :
  let companies : Finset ℕ := {1, 2, 3, 4},
      choices := companies.powerset.filter (λ s, s.card = 2) in
  (∑ A in choices, ∑ B in choices.filter (λ b, (A ∩ b).card = 1), 1) = 24 :=
by sorry

end internship_company_selection_exactly_one_same_l102_102882


namespace strategy_winner_l102_102870

def hasWinningStrategy (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) (h3 : ¬ (∃a, ∃b, ∃c, ∃d, (a,b,c,d : ℝ^3))) : Prop :=
  (m % 2 = 1 ∧ n % 2 = 1) → "Geoff" = "winning" ∨ (m % 2 = 0 ∨ n % 2 = 0) → "Nazar" = "winning".

theorem strategy_winner (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) (h3 : ¬ (∃a, ∃b, ∃c, ∃d, (a,b,c,d : ℝ^3))) :
  hasWinningStrategy m n h1 h2 h3 :=
by
  sorry

end strategy_winner_l102_102870


namespace selling_price_solution_l102_102509

def profit_eq (x : ℝ) : Prop :=
  (x - 40) * (500 - 10 * (x - 50)) = 8000

def cost_le_10000 (x : ℝ) : Prop :=
  40 * (500 - 10 * (x - 50)) ≤ 10000

theorem selling_price_solution :
  ∃ x : ℝ, profit_eq x ∧ cost_le_10000 x :=
by
  use 80
  split
  -- Prove (80 - 40) * (500 - 10 * (80 - 50)) = 8000
  { sorry }
  -- Prove 40 * (500 - 10 * (80 - 50)) ≤ 10000
  { sorry }

end selling_price_solution_l102_102509


namespace cash_sales_is_48_l102_102097

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end cash_sales_is_48_l102_102097


namespace sum_of_two_integers_l102_102122

theorem sum_of_two_integers (a b : ℕ) (h₁ : a * b + a + b = 135) (h₂ : Nat.gcd a b = 1) (h₃ : a < 30) (h₄ : b < 30) : a + b = 23 :=
sorry

end sum_of_two_integers_l102_102122


namespace cash_sales_is_48_l102_102096

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end cash_sales_is_48_l102_102096


namespace mr_caiden_payment_l102_102065

theorem mr_caiden_payment (total_feet_needed : ℕ) (cost_per_foot : ℕ) (free_feet_supplied : ℕ) : 
  total_feet_needed = 300 → cost_per_foot = 8 → free_feet_supplied = 250 → 
  (total_feet_needed - free_feet_supplied) * cost_per_foot = 400 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end mr_caiden_payment_l102_102065


namespace find_x_l102_102203

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem find_x :
  {x : ℝ | distance (3, -2) (x, 10) = 15} = {12, -6} :=
by
  sorry

end find_x_l102_102203


namespace find_sales_tax_rate_l102_102022

-- The statement of the problem including conditions
def cost_of_milk : ℝ := 3
def cost_of_bananas : ℝ := 2
def total_spent : ℝ := 6

-- Define the problem to prove the sales tax rate
theorem find_sales_tax_rate 
  (cost_of_milk : ℝ)
  (cost_of_bananas : ℝ)
  (total_spent : ℝ)
  (sales_tax_rate : ℝ) :
  cost_of_milk = 3 →
  cost_of_bananas = 2 →
  total_spent = 6 →
  sales_tax_rate = ((total_spent - (cost_of_milk + cost_of_bananas)) / (cost_of_milk + cost_of_bananas)) * 100 →
  sales_tax_rate = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end find_sales_tax_rate_l102_102022


namespace allocation_methods_count_l102_102546

/-- The number of ways to allocate 24 quotas to 3 venues such that:
1. Each venue gets at least one quota.
2. Each venue gets a different number of quotas.
is equal to 222. -/
theorem allocation_methods_count : 
  ∃ n : ℕ, n = 222 ∧ 
  ∃ (a b c: ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a + b + c = 24 ∧ 
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c := 
sorry

end allocation_methods_count_l102_102546


namespace lines_are_parallel_and_not_coincident_l102_102859

theorem lines_are_parallel_and_not_coincident (a : ℝ) :
  (a * (a - 1) - 3 * 2 = 0) ∧ (3 * (a - 7) - a * 3 * a ≠ 0) ↔ a = 3 :=
by
  sorry

end lines_are_parallel_and_not_coincident_l102_102859


namespace monotonicity_of_f_range_of_a_f_geq_exp_negx_l102_102319

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → deriv (λ x, a * exp x - x) x < 0) ∧
  (∀ x : ℝ, 0 < a → ((x < Real.log a → deriv (λ x, a * exp x - x) x < 0) ∧ 
                    (x > Real.log a → deriv (λ x, a * exp x - x) x > 0))) :=
by sorry

theorem range_of_a_f_geq_exp_negx (a : ℝ) :
  (∀ x ∈ Icc 1 2, a * exp x - x ≥ exp (-x)) ↔ a ≥ (1 / (exp 2) + 1 / exp 1) :=
by sorry

end monotonicity_of_f_range_of_a_f_geq_exp_negx_l102_102319


namespace part_a_part_b_part_c_l102_102618

noncomputable def expected_total_dice_rolls : ℕ := 30

noncomputable def expected_total_points_rolled : ℕ := 105

noncomputable def expected_number_of_salvos : ℚ := 13.02

theorem part_a :
  (let total_dice := 5
   in ∀ (salvo_rolls : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → salvo_rolls die = 6)
   → (total_dice * 6) = expected_total_dice_rolls) :=
by {
  sorry
}

theorem part_b :
  (let total_dice := 5
   in ∀ (points_rolled : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → points_rolled die = 21)
   → (total_dice * 21) = expected_total_points_rolled) :=
by {
  sorry
}

theorem part_c :
  (let total_dice := 5
   in ∀ (salvos : ℕ → ℚ),
   (∀ die : ℕ, die < total_dice → salvos die = 13.02)
   → (total_dice * 13.02) = expected_number_of_salvos) :=
by {
  sorry
}

end part_a_part_b_part_c_l102_102618


namespace distance_from_P_to_AB_l102_102928

theorem distance_from_P_to_AB (hP : ℝ) (h_triangle : Triangle ABC) 
  (h_parallel : is_parallel (line_through P base AB)) 
  (h_equal_area : divides_into_equal_areas (line_through P base AB))
  (altitude_C_AB : length (altitude_from C to AB) = 2) 
  (length_AB : length (base AB) = 8) :
  distance_from P to AB = 1 :=
sorry

end distance_from_P_to_AB_l102_102928


namespace composite_surface_area_is_39_l102_102575

def radius1 := 1.5
def radius2 := 1
def radius3 := 0.5
def height := 1
def pi := 3

def surface_area (r : ℝ) (h : ℝ) (π : ℝ) : ℝ :=
  let lateral_surface_area := 2 * π * r * h
  let base_area := π * r^2
  lateral_surface_area + 2 * base_area

def combined_surface_area : ℝ :=
  surface_area radius1 height pi +
  surface_area radius2 height pi +
  surface_area radius3 height pi

theorem composite_surface_area_is_39 :
  combined_surface_area = 39 :=
by
  sorry

end composite_surface_area_is_39_l102_102575


namespace sum_of_consecutive_integers_l102_102351

theorem sum_of_consecutive_integers:
  ∃ a b : ℤ, (a < (sqrt 5 + 1) / 2 ∧ (sqrt 5 + 1) / 2 < b) ∧ a + b = 3 :=
by
  use [1, 2]
  split
  sorry

end sum_of_consecutive_integers_l102_102351


namespace range_of_a_l102_102674

variable (f : ℝ → ℝ) (a : ℝ)
variable (H1 : ∀ x, deriv f x = a * (x - 1) * (x - a))
variable (H2 : is_local_max f a)

theorem range_of_a (H1 : ∀ x, deriv f x = a * (x - 1) * (x - a))
                   (H2 : is_local_max f a) : 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l102_102674


namespace minimal_n_value_l102_102381

noncomputable def minimal_sum_n (a : ℕ → ℝ) : ℕ :=
  let d := a 6 - a 5 in
  let S (n : ℕ) := (n / 2) * (2 * a 1 + (n - 1) * d) in
  if 7 * a 5 + 5 * a 9 = 0 ∧ a 5 < a 9 then 6 else 0 -- 6 if conditions hold otherwise 0

theorem minimal_n_value (a : ℕ → ℝ) (h1 : 7 * a 5 + 5 * a 9 = 0) (h2 : a 5 < a 9) :
  minimal_sum_n a = 6 :=
by {
  sorry
}

end minimal_n_value_l102_102381


namespace cash_sales_amount_l102_102098

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end cash_sales_amount_l102_102098


namespace cone_height_ratio_proof_l102_102932

noncomputable def coneHeightRatio : ℚ :=
  let r := 10
  let originalHeight := 40
  let newVolume := 800 * Real.pi
  let newHeight := (3 * newVolume) / (100 * Real.pi)
  newHeight / originalHeight

theorem cone_height_ratio_proof (circumference : ℝ) (originalHeight : ℝ) (newVolume : ℝ) (r : ℝ) (h : ℝ) :
  circumference = 20 * Real.pi → r = 10 → originalHeight = 40 →
  newVolume = 800 * Real.pi → h = (3 * newVolume) / (100 * Real.pi) →
  (h / originalHeight = (3/5) : ℚ) := by
  intros h1 h2 h3 h4 h5
  have : h = 24 := by
    rw [h5, h4]
    field_simp
    norm_num
  rw [this, h3]
  norm_num
  done

end cone_height_ratio_proof_l102_102932


namespace dividend_from_tonys_stock_l102_102910

theorem dividend_from_tonys_stock (investment price_per_share total_income : ℝ) 
  (h1 : investment = 3200) (h2 : price_per_share = 85) (h3 : total_income = 250) : 
  (total_income / (investment / price_per_share)) = 6.76 :=
by 
  sorry

end dividend_from_tonys_stock_l102_102910


namespace quadratic_integers_roots_l102_102274

theorem quadratic_integers_roots {a : ℝ} : 
  (∀ x ∈ ℝ, x^2 - a*x + a = 0 → x ∈ ℤ) → a = 0 ∨ a = 4 :=
sorry

end quadratic_integers_roots_l102_102274


namespace driver_net_rate_of_pay_l102_102200

theorem driver_net_rate_of_pay 
  (hours_traveled : ℝ) (speed : ℝ) (mpg : ℝ) (pay_per_mile : ℝ) (gas_cost_per_gallon : ℝ)
  (h_hours : hours_traveled = 2) (h_speed : speed = 60) (h_mpg : mpg = 30) 
  (h_pay_per_mile : pay_per_mile = 0.5) (h_gas_cost_per_gallon : gas_cost_per_gallon = 2) :
  (pay_per_mile * speed * hours_traveled - gas_cost_per_gallon * (speed * hours_traveled / mpg)) / hours_traveled = 26 :=
by 
  rw [h_hours, h_speed, h_mpg, h_pay_per_mile, h_gas_cost_per_gallon]
  norm_num
  simp
  sorry

end driver_net_rate_of_pay_l102_102200


namespace functional_equation_solution_l102_102041

theorem functional_equation_solution :
  ∃ f : ℝ → ℝ,
  (f 1 = 1 ∧ (∀ x y : ℝ, f (x * y + f x) = x * f y + f x)) ∧ f (1/2) = 1/2 :=
by
  sorry

end functional_equation_solution_l102_102041


namespace factories_checked_by_second_group_l102_102602

theorem factories_checked_by_second_group 
(T : ℕ) (G1 : ℕ) (R : ℕ) 
(hT : T = 169) 
(hG1 : G1 = 69) 
(hR : R = 48) : 
T - (G1 + R) = 52 :=
by {
  sorry
}

end factories_checked_by_second_group_l102_102602


namespace customers_in_each_car_l102_102471

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end customers_in_each_car_l102_102471


namespace no_int_a_divisible_289_l102_102437

theorem no_int_a_divisible_289 : ¬ ∃ a : ℤ, ∃ k : ℤ, a^2 - 3 * a - 19 = 289 * k :=
by
  sorry

end no_int_a_divisible_289_l102_102437


namespace single_shot_decrease_l102_102907

theorem single_shot_decrease (S : ℝ) (r1 r2 r3 : ℝ) (h1 : r1 = 0.05) (h2 : r2 = 0.10) (h3 : r3 = 0.15) :
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100 = 27.325 := 
by
  sorry

end single_shot_decrease_l102_102907


namespace ratio_3_2_l102_102018

theorem ratio_3_2 (m n : ℕ) (h1 : m + n = 300) (h2 : m > 100) (h3 : n > 100) : m / n = 3 / 2 := by
  sorry

end ratio_3_2_l102_102018


namespace length_of_EC_l102_102339

-- Definitions based on the problem conditions.
variables {A B C D E : Type*}
variables [is_triangle A B C] (angleA : ∠A = 45) (BC : dist B C = 8)
variables (midpoint_D : is_midpoint D A C) (perpendicular_BD_AC : ⊥ BD AC)
variables (perpendicular_CE_AB : ⊥ CE AB) (angle_relation : ∠DBC = 3 * ∠ECB)

-- The theorem we need to prove.
theorem length_of_EC
  (angleA : ∠A = 45) (BC : dist B C = 8)
  (midpoint_D : is_midpoint D A C) (perpendicular_BD_AC : ⊥ BD AC)
  (perpendicular_CE_AB : ⊥ CE AB) (angle_relation : ∠DBC = 3 * ∠ECB) :
  dist E C = 4 * real.sqrt 2 :=
sorry

end length_of_EC_l102_102339


namespace ratio_GH_BD_l102_102074

theorem ratio_GH_BD 
  (A B C D E F G H : Point)
  (x : ℝ)
  (h_square : is_square A B C D)
  (h_ratio_BE_EA : BE / EA = 2022 / 2023)
  (h_ratio_AF_FD : AF / FD = 2022 / 2023)
  (h_E_on_AB : on_line E A B)
  (h_F_on_AD : on_line F A D)
  (h_intercept_G : intersect EC BD = G)
  (h_intercept_H : intersect FC BD = H) :
  GH / BD = 12271519 / 36814556 := 
sorry

end ratio_GH_BD_l102_102074


namespace fourth_guard_distance_l102_102167

open Real

theorem fourth_guard_distance
  (length width : ℝ)
  (hl : length = 200)
  (hw : width = 300)
  (d3_sum : ℝ)
  (h3 : d3_sum = 850) :
  let P := 2 * (length + width) in
  let total_distance := P in
  let d4 := total_distance - d3_sum in
  d4 = 150 :=
by
  sorry

end fourth_guard_distance_l102_102167


namespace students_overlap_difference_l102_102730

theorem students_overlap_difference : 
  ∀ (total students geometry biology : ℕ),
  total = 232 → geometry = 144 → biology = 119 →
  (geometry + biology - total = 31) ∧ (min geometry biology = 119) →
  (min geometry biology - (geometry + biology - total) = 88) :=
by
  intros total geometry biology htotal hgeometry hbiology hconds,
  exact sorry

end students_overlap_difference_l102_102730


namespace eq_of_op_star_l102_102701

theorem eq_of_op_star (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^b^2)^n = a^(bn)^2 ↔ n = 1 := by
sorry

end eq_of_op_star_l102_102701


namespace octahedron_side_length_in_unit_cube_l102_102588

theorem octahedron_side_length_in_unit_cube :
  let V1 := (0 : ℝ, 0 : ℝ, 0 : ℝ),
      V1' := (1 : ℝ, 1 : ℝ, 1 : ℝ),
      x := 3 / 4,
      y := 3 / 4,
      z := 3 / 4,
      A := (x, 0, 0),
      B := (0, y, 0),
      C := (0, 0, z),
      A' := (1, 1 - y, 1),
      B' := (1 - x, 1, 1),
      C' := (1, 1, 1 - z) in
  dist A B = dist B C ∧ dist B C = dist C A :=
  sorry

end octahedron_side_length_in_unit_cube_l102_102588


namespace sum_of_exponents_l102_102088

theorem sum_of_exponents (x y z : ℕ) : 
  let expr := 40 * x^5 * y^9 * z^{14} in
  let simplified_expr := 2 * x * y * z^3 in
  ∑ i in [(1 : ℕ), 1, 3], i = 5 :=
by
  sorry

end sum_of_exponents_l102_102088


namespace circle_circumference_to_diameter_ratio_invariant_l102_102723

theorem circle_circumference_to_diameter_ratio_invariant (r : ℝ) (h : r + 2 ≠ 0) : 
  let new_radius := r + 2 
  let new_diameter := 2 * new_radius 
  let new_circumference := 2 * real.pi * new_radius
  (new_circumference / new_diameter) = real.pi := 
by
  let new_radius := r + 2 
  let new_diameter := 2 * new_radius 
  let new_circumference := 2 * real.pi * new_radius
  have h1 : new_circumference = 2 * real.pi * (r + 2) := by sorry
  have h2 : new_diameter = 2 * (r + 2) := by sorry
  have h3 : new_circumference / new_diameter = (2 * real.pi * (r + 2)) / (2 * (r + 2)) := by sorry
  have h4 : (2 * real.pi * (r + 2)) / (2 * (r + 2)) = real.pi := by sorry
  exact h4

end circle_circumference_to_diameter_ratio_invariant_l102_102723


namespace tangency_concurrence_l102_102769

theorem tangency_concurrence
  {A1 A2 A3 A4 : Type}
  (no_parallel_sides : ¬(A1 ≠ A3 ∧ A2 ≠ A4))
  (ω1 ω2 ω3 ω4 : circle)
  (tangent_circles : ∀ i : fin 4, is_tangent (ω i) (side A_{i-1}, side A_i, side (A_{i+1})))
  (T1 T2 T3 T4 : point)
  (tangency_points : ∀ i : fin 4, T i = point_of_tangency (ω i) (side A_i A_{i+1})) :
  concurrent (line A1 A2) (line A3 A4) (line T2 T4) ↔ concurrent (line A2 A3) (line A4 A1) (line T1 T3) :=
begin
  sorry
end

end tangency_concurrence_l102_102769


namespace log_sqrt_pi_simplification_l102_102176

theorem log_sqrt_pi_simplification:
  2 * Real.log 4 + Real.log (5 / 8) + Real.sqrt ((Real.sqrt 3 - Real.pi) ^ 2) = 1 + Real.pi - Real.sqrt 3 :=
sorry

end log_sqrt_pi_simplification_l102_102176


namespace population_multiple_of_18_l102_102121

theorem population_multiple_of_18
  (a b c P : ℕ)
  (ha : P = a^2)
  (hb : P + 200 = b^2 + 1)
  (hc : b^2 + 301 = c^2) :
  ∃ k, P = 18 * k := 
sorry

end population_multiple_of_18_l102_102121


namespace remainder_of_division_l102_102973

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l102_102973


namespace log_monotonicity_interval_l102_102992

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3) / Real.log (1 / 2)

theorem log_monotonicity_interval :
  ∀ x, (x < -1) → (∀ y, (y < -1) → (x ≤ y → f(x) ≤ f(y))) ∧ (x > 1) → (∀ y, (y > 1) → (x ≤ y → f(x) ≤ f(y))) :=
by
  sorry

end log_monotonicity_interval_l102_102992


namespace roots_sum_l102_102776

theorem roots_sum (a b : ℝ) 
  (h1 : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
          (roots_of (λ x, x^3 - 8 * x^2 + a * x - b) = {r1, r2, r3}) ) :
  a + b = 31 :=
sorry

end roots_sum_l102_102776


namespace rectangle_to_square_y_l102_102544

theorem rectangle_to_square_y (y : ℝ) (a b : ℝ) (s : ℝ) (h1 : a = 7) (h2 : b = 21)
  (h3 : s^2 = a * b) (h4 : y = s / 2) : y = 7 * Real.sqrt 3 / 2 :=
by
  -- proof skipped
  sorry

end rectangle_to_square_y_l102_102544


namespace floor_of_fraction_is_2006_l102_102979

theorem floor_of_fraction_is_2006 :
  ⌊(nat.factorial 2007 + nat.factorial 2004 : ℚ) / (nat.factorial 2006 + nat.factorial 2005)⌋ = 2006 := 
by
  sorry

end floor_of_fraction_is_2006_l102_102979


namespace distance_inequality_l102_102011

-- Define basic points and vectors in the plane
variables (O A B C P Q : Type) [plane : Plane O A B C]

-- Declare the conditions
open Plane
open EuclideanGeometry

-- Condition (O is the center of triangle ABC)
def is_center_triangle (O A B C : point) [equilateral O A B C] : Prop :=
is_center O A B C

-- Condition (OQ = 2*PO)
def scaled_vector (O P Q : point) : Prop :=
vector.eq (O.to_vector Q) (2 * (O.to_vector P))

-- The statement to be proved
theorem distance_inequality (O A B C P Q : point) 
  [is_center_triangle O A B C] [scaled_vector O P Q] :
  (distance P A + distance P B + distance P C) ≤ (distance Q A + distance Q B + distance Q C) :=
sorry

end distance_inequality_l102_102011


namespace highest_frequency_fruit_l102_102475

def tangerines (boxes : Nat) (tangerines_per_box : Nat) : Nat := boxes * tangerines_per_box
def apples (boxes : Nat) (apples_per_box : Nat) : Nat := boxes * apples_per_box
def pears (boxes : Nat) (pears_per_box : Nat) : Nat := boxes * pears_per_box

theorem highest_frequency_fruit :
    let num_tangerines := tangerines 5 30 in
    let num_apples := apples 3 20 in
    let num_pears := pears 4 15 in
    num_tangerines > num_apples ∧ num_tangerines > num_pears :=
by
    intros
    -- proof goes here
    sorry

end highest_frequency_fruit_l102_102475


namespace dilworth_theorem_l102_102742

def poset (P : Type) := P → P → Prop

def is_antichain {P : Type} (leq : poset P) (A : set P) : Prop :=
∀ a b ∈ A, a ≠ b → ¬ leq a b ∧ ¬ leq b a

def max_antichain_size {P : Type} (leq : poset P) : ℕ :=
  max (cardinal.mk {A : set P // is_antichain leq A})

def min_chain_cover {P : Type} (leq : poset P) : ℕ :=
  sorry -- Formalize the minimum chain cover size here

theorem dilworth_theorem (P : Type) (leq : poset P) [fintype P] :
  min_chain_cover leq = max_antichain_size leq :=
sorry

end dilworth_theorem_l102_102742


namespace cost_of_notebook_l102_102726

theorem cost_of_notebook (num_students : ℕ) (more_than_half_bought : ℕ) (num_notebooks : ℕ) 
                         (cost_per_notebook : ℕ) (total_cost : ℕ) 
                         (half_students : more_than_half_bought > 18) 
                         (more_than_one_notebook : num_notebooks > 1) 
                         (cost_gt_notebooks : cost_per_notebook > num_notebooks) 
                         (calc_total_cost : more_than_half_bought * cost_per_notebook * num_notebooks = 2310) :
  cost_per_notebook = 11 := 
sorry

end cost_of_notebook_l102_102726


namespace people_on_bus_now_l102_102717

variable (x : ℕ)

def original_people_on_bus : ℕ := 38
def people_got_on_bus (x : ℕ) : ℕ := x
def people_left_bus (x : ℕ) : ℕ := x + 9

theorem people_on_bus_now (x : ℕ) : original_people_on_bus - people_left_bus x + people_got_on_bus x = 29 := 
by
  sorry

end people_on_bus_now_l102_102717


namespace rectangle_max_area_l102_102545

noncomputable def max_rectangle_area : ℝ :=
  let x_vals := [16, 9]
  let r := real.sqrt (x_vals.max' _)
  let l := (2/5) * r
  let ratio := 5
  let b := l / ratio
  l * b

theorem rectangle_max_area (x_vals : Set ℝ) (h1 : ∀ x ∈ x_vals, x^2 - 25 * x + 144 = 0) (h_pos : ∀ x ∈ x_vals, 0 < x)
  (ratio_vals : Set ℝ) (h2 : ∀ x ∈ ratio_vals, x^2 - 3 * x - 10 = 0) (h_pos_ratio : ∀ x ∈ ratio_vals, 0 < x) :
  max_rectangle_area = 0.512 :=
by
  sorry

end rectangle_max_area_l102_102545


namespace minimum_weights_for_balance_l102_102021

theorem minimum_weights_for_balance (k : ℕ) :
  (∀ m : ℤ, (1 ≤ m ∧ m ≤ 2009) → 
  ∃ (ws : fin k → ℤ), (∀ (σ : fin k → ℤ), σ ∈ {1, -1, 0}^fin k → m = (∑ i, ws i * σ i))) → k = 8 :=
begin
  sorry
end

end minimum_weights_for_balance_l102_102021


namespace log_eq_exp_l102_102817

theorem log_eq_exp (x : ℝ) (h : log 3 x + log 9 x = 5) : 
  x = 3 ^ (10 / 3) := 
by
  sorry

end log_eq_exp_l102_102817


namespace inscribe_hexagon_no_intersect_l102_102524

theorem inscribe_hexagon_no_intersect (r : ℝ) (chords : list ℝ)
  (hr : r = 1) (hlen : chords.sum = 1) : 
  ∃ hexagon : list (real × real), 
  is_regular_hexagon hexagon ∧ ∀ side, side ∈ sides hexagon → ∀ chord, chord ∈ chords → ¬ (intersects side chord) :=
by sorry

-- Definitions used as examples (the actual definitions may vary based on the implementation)
def is_regular_hexagon (hexagon : list (real × real)) : Prop := sorry
def sides (hexagon : list (real × real)) : list (real × real) := sorry
def intersects (side : (real × real)) (chord : real) : Prop := sorry

end inscribe_hexagon_no_intersect_l102_102524


namespace find_complex_z_l102_102691

-- Define the imaginary unit i
def i := Complex.I

-- Define the given complex equation (1 + i) * z = i
def given_equation (z : Complex) : Prop :=
  (1 + i) * z = i

-- Define the target value of z
def target_value : Complex :=
  1 / 2 - (1 / 2) * i

-- Formalize the theorem statement
theorem find_complex_z (z : Complex) (h : given_equation z) : 
  z = target_value := 
sorry

end find_complex_z_l102_102691


namespace find_a_b_range_m_proving_inequality_l102_102318

variable {x : ℝ} (f : ℝ → ℝ) (e : ℝ) (a b : ℝ)

-- Define the function f
def func (x : ℝ) : ℝ := (a + b * log x) / (x + 1)

-- Condition: The equation of the tangent line at the point (1, f(1)) is x + y = 2
def tangent_condition_at_1 : Prop := func 1 = 1

-- Condition: The range for m such that x * f(x) < m
def inequality (m : ℝ) (x : ℝ) : Prop := x * func x < m

-- Condition: Inequality for 3 - (x + 1) * f(x)
def inequality_proof (x : ℝ) : Prop := 3 - (x + 1) * func x > 1 / exp x - 2 / (exp x * x)

theorem find_a_b :
  tangent_condition_at_1 → a = 2 ∧ b = -1 := sorry

theorem range_m (m : ℝ) :
  (∀ x > 0, x * func x < m) → 1 < m := sorry

theorem proving_inequality :
  (∀ x > 0, inequality_proof x) := sorry

end find_a_b_range_m_proving_inequality_l102_102318


namespace least_subtracted_12702_is_26_l102_102166

theorem least_subtracted_12702_is_26 : 12702 % 99 = 26 :=
by
  sorry

end least_subtracted_12702_is_26_l102_102166


namespace median_song_length_l102_102131

-- Define the list of song lengths in seconds
def song_lengths : List ℕ := [32, 43, 58, 65, 70, 72, 75, 80, 145, 150, 175, 180, 195, 210, 215, 225, 250, 252]

-- Define the statement that the median length of the songs is 147.5 seconds
theorem median_song_length : ∃ median : ℕ, median = 147 ∧ (median : ℚ) + 0.5 = 147.5 := by
  sorry

end median_song_length_l102_102131


namespace total_theme_parks_l102_102753

-- Define the constants based on the problem's conditions
def Jamestown := 20
def Venice := Jamestown + 25
def MarinaDelRay := Jamestown + 50

-- Theorem statement: Total number of theme parks in all three towns is 135
theorem total_theme_parks : Jamestown + Venice + MarinaDelRay = 135 := by
  sorry

end total_theme_parks_l102_102753


namespace divide_into_equal_areas_l102_102915

-- Define the figure having all right angles
inductive RightAngleFigure
| rectangle (a b: ℝ) : RightAngleFigure
| composed_of_rectangles (figs : list RightAngleFigure) : RightAngleFigure

-- Function to compute the area of the RightAngleFigure
def area : RightAngleFigure → ℝ
| RightAngleFigure.rectangle a b := a * b
| RightAngleFigure.composed_of_rectangles figs := figs.sum (λ f, area f)

-- Property of the diagonals of a rectangle
axiom diagonals_intersection_point 
  {a b : ℝ} : 
  ∃ (p : ℝ × ℝ), 
  (fst p = a / 2) ∧ (snd p = b / 2)

-- Proving that a line passing through the intersection point will divide the figure into two equal areas
theorem divide_into_equal_areas (fig : RightAngleFigure) :
  ∃ (line : (ℝ × ℝ) × (ℝ × ℝ)),
  (line.1 = line.2) → ∀ (d : diag_intersection_point fig), 
  (area (RightAngleFigure.composed_of_rectangles [subfigure1, subfigure2]) = area fig / 2) :=
sorry

end divide_into_equal_areas_l102_102915


namespace company_employees_after_reduction_l102_102214

theorem company_employees_after_reduction :
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  current_number = 195 :=
by
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  sorry

end company_employees_after_reduction_l102_102214


namespace sum_of_exponents_l102_102087

theorem sum_of_exponents (x y z : ℕ) : 
  let expr := 40 * x^5 * y^9 * z^{14} in
  let simplified_expr := 2 * x * y * z^3 in
  ∑ i in [(1 : ℕ), 1, 3], i = 5 :=
by
  sorry

end sum_of_exponents_l102_102087


namespace average_volume_correct_l102_102962

-- Define the volumes of the milk bottles in liters.
def volumes : List ℝ := [2.35, 1.75, 0.9, 0.75, 0.5, 0.325, 0.25]

-- Total volume is the sum of the volumes.
def total_volume (vols : List ℝ) : ℝ := vols.sum

-- Calculate the average volume per bottle.
def average_volume (vols : List ℝ) : ℝ := total_volume vols / vols.length

-- The proof statement.
theorem average_volume_correct :
  total_volume volumes = 6.825 ∧ average_volume volumes = 0.975 :=
by
  sorry

end average_volume_correct_l102_102962


namespace no_seven_brown_l102_102886

def cockroach_board (board : Matrix (Fin 4) (Fin 4) ℕ) (brown : ℕ) (black : ℕ) : Prop :=
  ∀ i j, (board i j = brown → 
  ((i > 0 → board (i-1) j = brown) ∧ 
   (i < 3 → board (i+1) j = brown) ∧ 
   (j > 0 → board i (j-1) = brown) ∧ 
   (j < 3 → board i (j+1) = brown))) ∧ 
  (board i j = black → 
  ((i > 0 → board (i-1) j ≠ black) ∧ 
   (i < 3 → board (i+1) j ≠ black) ∧ 
   (j > 0 → board i (j-1) ≠ black) ∧ 
   (j < 3 → board i (j+1) ≠ black)))

theorem no_seven_brown (board : Matrix (Fin 4) (Fin 4) ℕ): 
  (∃ brown black, (∑ i j, if (board i j = brown) then 1 else 0 = 7 ∧ 
  ∑ i j, if (board i j = black) then 1 else 0 = 9) ∧ 
  cockroach_board board brown black) → false := 
begin
  sorry 
end

end no_seven_brown_l102_102886


namespace number_of_true_propositions_l102_102590

variable (a b c : ℝ^3) -- Assume vectors are in 3-dimensional space
variable (x : ℝ) -- Assume x is a real number

lemma proposition1 : (|a| = |b| → a = b) ↔ false := sorry

lemma proposition2 (hx : x ∈ set.Icc 0 real.pi) (hcos : real.cos x = -2/3):
  x = real.pi - real.arccos (2/3) := sorry

lemma proposition3 : (a = b ∧ b = c) → a = c := sorry

lemma proposition4 : (a = b → (|a| = |b| ∧ vector.parallel a b)) := sorry

theorem number_of_true_propositions : 
  ((|a| = |b| → a = b) = false) ∧
  ((∃ x, x ∈ set.Icc 0 real.pi ∧ real.cos x = -2/3 ∧ x = real.pi - real.arccos 2/3)) ∧
  ((a = b ∧ b = c) → a = c) ∧
  ((a = b → (|a| = |b| ∧ vector.parallel a b)))
  → ∃ n, n = 3 := 
sorry

end number_of_true_propositions_l102_102590


namespace compare_values_l102_102040

theorem compare_values (a b c : ℝ) (h₁ : a = 1/2) (h₂ : b = log 3 2) (h₃ : c = 2^(1/3)) : c > b ∧ b > a :=
by
  sorry

end compare_values_l102_102040


namespace prob_xy_x_y_div_by_3_eq_2_7_l102_102883

theorem prob_xy_x_y_div_by_3_eq_2_7 :
  let S := { x | 1 ≤ x ∧ x ≤ 15 ∧ x ∈ ℕ }
  let pairs := (S.product S).filter (λ p, p.1 ≠ p.2)
  let favorable := pairs.filter (λ p, (p.1 * p.2 - p.1 - p.2) % 3 = 0)
  (favorable.card : ℚ) / (pairs.card : ℚ) = 2 / 7 := by
  sorry

end prob_xy_x_y_div_by_3_eq_2_7_l102_102883


namespace partition_subsets_with_equal_sum_l102_102787

theorem partition_subsets_with_equal_sum :
  ∀ (a : Fin 32 → ℕ), (∀ i, 1 ≤ a i) → (∑ i, a i = 120) → (∀ i, a i ≤ 60) →
  ∃ (S1 S2 : Finset (Fin 32)), S1 ∩ S2 = ∅ ∧ S1 ∪ S2 = Finset.univ ∧ (S1.sum a = S2.sum a) :=
by
  intro a h1 h2 h3
  sorry

end partition_subsets_with_equal_sum_l102_102787


namespace find_angle_beta_l102_102608

theorem find_angle_beta (α β : ℝ)
  (h1 : (π / 2) < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19)
  (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := 
sorry

end find_angle_beta_l102_102608


namespace cos_expression_sum_l102_102833

theorem cos_expression_sum :
  ∃ (a b c d : ℕ), a = 4 ∧ b = 5 ∧ c = 2 ∧ d = 1 ∧
                   (cos (2 * x) + cos (4 * x) + cos (6 * x) + cos (8 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) ∧
                   (a + b + c + d = 12) :=
by {
  use [4, 5, 2, 1],
  -- You may leave the proof steps as sorry for now.
  sorry
}

end cos_expression_sum_l102_102833


namespace range_of_a_l102_102648

theorem range_of_a (x : ℝ) (a : ℝ) (hx : x ∈ Ioc 0 real.pi) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * sin (x₁ + real.pi / 3) = a ∧ 2 * sin (x₂ + real.pi / 3) = a) →
  a ∈ Ioo (real.sqrt 3) 2 :=
sorry

end range_of_a_l102_102648


namespace union_of_A_and_B_l102_102358

open Set

variable {α : Type*} [LinearOrder α]

noncomputable def A : Set α := {x | x < 2}
noncomputable def B : Set α := {x | -1 < x ∧ x < 3}

theorem union_of_A_and_B : A ∪ B = {x | x < 3} := by
  sorry

end union_of_A_and_B_l102_102358


namespace remainder_div_1234567_256_l102_102965

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l102_102965


namespace james_profit_l102_102023

theorem james_profit
  (tickets_bought : ℕ)
  (cost_per_ticket : ℕ)
  (percentage_winning : ℕ)
  (winning_tickets_percentage_5dollars : ℕ)
  (grand_prize : ℕ)
  (average_other_prizes : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (winning_tickets : ℕ)
  (tickets_prize_5dollars : ℕ)
  (amount_won_5dollars : ℕ)
  (other_winning_tickets : ℕ)
  (other_tickets_prize : ℕ)
  (total_winning_amount : ℕ)
  (profit : ℕ) :

  tickets_bought = 200 →
  cost_per_ticket = 2 →
  percentage_winning = 20 →
  winning_tickets_percentage_5dollars = 80 →
  grand_prize = 5000 →
  average_other_prizes = 10 →
  total_tickets = tickets_bought →
  total_cost = total_tickets * cost_per_ticket →
  winning_tickets = (percentage_winning * total_tickets) / 100 →
  tickets_prize_5dollars = (winning_tickets_percentage_5dollars * winning_tickets) / 100 →
  amount_won_5dollars = tickets_prize_5dollars * 5 →
  other_winning_tickets = winning_tickets - 1 →
  other_tickets_prize = (other_winning_tickets - tickets_prize_5dollars) * average_other_prizes →
  total_winning_amount = amount_won_5dollars + grand_prize + other_tickets_prize →
  profit = total_winning_amount - total_cost →
  profit = 4830 := 
sorry

end james_profit_l102_102023


namespace customers_in_each_car_l102_102472

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end customers_in_each_car_l102_102472


namespace gala_handshakes_l102_102961

def num_handshakes (couples men women : ℕ) (handshakes_among_men handshakes_between_men_and_women : ℕ): ℕ :=
  handshakes_among_men + handshakes_between_men_and_women

theorem gala_handshakes
  (n : ℕ) -- number of married couples
  (men women : ℕ) -- number of men and women
  (handshakes_among_men handshakes_between_men_and_women total_handshakes : ℕ)
  (h1 : n = 15) -- 15 married couples
  (h2 : men = 15) -- 15 men
  (h3 : women = 15) -- 15 women
  (h4 : handshakes_among_men = (men * (men - 1)) / 2) -- handshakes among men
  (h5 : handshakes_between_men_and_women = men * women) -- handshakes between each man and women (including his spouse)
  (h6 : total_handshakes = handshakes_among_men + handshakes_between_men_and_women) :
  total_handshakes = 330 :=
by
  rw [h1, h2, h3] at h4 h5
  rw h4 at h6
  rw h5 at h6
  norm_num at h6
  sorry

end gala_handshakes_l102_102961


namespace total_shaded_area_l102_102745

/-
There are two circles:
- One larger circle with center \( O \) and area \( 64\pi \), divided into two equal regions.
- One smaller circle, also divided into two equal regions, whose diameter equals the radius of the larger circle.

We aim to prove the total area of the shaded regions is \( 40\pi \).
-/

def radius_of_larger_circle (area : ℝ) : ℝ := Real.sqrt (area / Real.pi)

noncomputable def shaded_area (r : ℝ) : ℝ := (Real.pi * r^2) / 2

theorem total_shaded_area (area_larger_circle : ℝ)
  (h_area_larger : area_larger_circle = 64 * Real.pi) :
  let r1 := radius_of_larger_circle area_larger_circle,
      r2 := r1 / 2 in
  shaded_area r1 + shaded_area r2 = 40 * Real.pi :=
by
  sorry

end total_shaded_area_l102_102745


namespace determinant_is_zero_l102_102404

noncomputable def determinant : ℝ := 
  Matrix.det ![
    ![a, b, c],
    ![c, a, b],
    ![b, c, a]
  ]

theorem determinant_is_zero (a b c p : ℝ) (h : ∀ x, Polynomial.eval x (Polynomial.C (-p) + Polynomial.C (3 * p) * Polynomial.X - Polynomial.C p * Polynomial.X ^ 3) = 0 → x = a ∨ x = b ∨ x = c) : 
  determinant a b c = 0 :=
sorry

end determinant_is_zero_l102_102404


namespace seq_term_and_first_l102_102310

theorem seq_term_and_first {a : ℕ → ℕ} {S : ℕ → ℕ} 
  (h_pos : ∀ n, 0 < a n)
  (hS : ∀ n, S n = (finset.range n).sum a)
  (h_relation : ∀ n, a n * a n + a n = 2 * S n) :
  a 0 = 1 ∧ (∀ n, a n = n) := 
by
  sorry

end seq_term_and_first_l102_102310


namespace part1_part2_l102_102685

variables {n : ℕ} (a : ℕ → ℚ) (T : ℕ → ℚ)

-- Given conditions
def prod_n (a : ℕ → ℚ) (n : ℕ) : ℚ := 
  (List.prod (List.map a (List.range (n + 1))))

axiom cond1 (n : ℕ) : a n + T n = 1

axiom prod_condition (n : ℕ) : T n = prod_n a n

-- Question 1: Proving the arithmetic sequence property
theorem part1 (n : ℕ) :
  (∀ n, T (n + 1) = a (n + 1) * T n) → 
  (∀ n, (1 / T (n + 1)) - (1 / T n) = 1) :=
sorry

-- Question 2: Proving the inequality for the sum
theorem part2 (n : ℕ) :
  (∀ n, T n = prod_n a n) →
  (∀ n, a n + T n = 1) →
  ∑ k in List.range (n + 1), ((a (k + 1) - a k) / a k) < 3 / 4 :=
sorry

end part1_part2_l102_102685


namespace probability_of_exactly_one_match_l102_102301

-- Define the problem context
def balls_and_boxes := {red, yellow, blue, green}
def is_match (ball box : _root_.balls_and_boxes) : Prop := ball = box

-- Define the total number of permutations
noncomputable def total_permutations : ℕ := 4.factorial

-- Define valid permutations count for exactly one match
def valid_permutations := 8

-- Calculate the probability
def probability_one_match := (valid_permutations : ℚ) / total_permutations

-- Create the theorem to prove
theorem probability_of_exactly_one_match :
  probability_one_match = 1 / 3 :=
by
  sorry

end probability_of_exactly_one_match_l102_102301


namespace calculate_expression_l102_102963

theorem calculate_expression :
  (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := 
by sorry

end calculate_expression_l102_102963


namespace box_white_balls_count_l102_102517

/--
A box has exactly 100 balls, and each ball is either red, blue, or white.
Given that the box has 12 more blue balls than white balls,
and twice as many red balls as blue balls,
prove that the number of white balls is 16.
-/
theorem box_white_balls_count (W B R : ℕ) 
  (h1 : B = W + 12) 
  (h2 : R = 2 * B) 
  (h3 : W + B + R = 100) : 
  W = 16 := 
sorry

end box_white_balls_count_l102_102517


namespace monotonic_decreasing_interval_sin_neg2x_plus_pi_over_6_l102_102116
open Real

theorem monotonic_decreasing_interval_sin_neg2x_plus_pi_over_6 :
  ∃ (k : ℤ), ∀ x : ℝ, y = sin (-2 * x + π / 6) -> x ∈ Icc (k * π - π / 6) (k * π + π / 3) :=
sorry

end monotonic_decreasing_interval_sin_neg2x_plus_pi_over_6_l102_102116


namespace solve_686_l102_102818

theorem solve_686 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 686 := 
by
  sorry

end solve_686_l102_102818


namespace incorrect_proposition_l102_102952

variable (p q : Prop)
variable (a b c x : ℝ)
variable (f : ℝ → ℝ)

-- Definition for even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- The conditions provided in Lean-compatible definitions
def condition1 : Prop := (¬ (p → q)) = (p ∧ ¬ q)
def condition2 : Prop := (¬ ∀ x > 2, x^2 - 2 * x > 0) = ∃ x ≤ 2, x^2 - 2 * x ≤ 0
def condition3 : Prop := (p ∧ q) → (p ∨ q) ∧ ((p ∧ q) ↔ (¬ (p ∨ ¬ q)))
def condition4 : Prop := (b = 0 → even_function (λ x, a * x^2 + b * x + c)) → (even_function (λ x, a * x^2 + b * x + c) → b = 0)

-- The main statement to prove
theorem incorrect_proposition :
  condition1 → condition2 → condition3 → condition4 → ¬ condition2 :=
by sorry

end incorrect_proposition_l102_102952


namespace solution_l102_102052

def two_adic_valuation (n : ℕ) : ℕ :=
  n.trailingZeros

noncomputable def problem (n : ℕ) (p : ℕ) [Prime p] : Prop :=
  n > 1 ∧ p ∣ 2^(2^n) + 1 → two_adic_valuation (p - 1) ≥ n + 2

theorem solution (n : ℕ) (p : ℕ) [Prime p] (hn : n > 1) (hp : p ∣ 2^(2^n) + 1) :
  two_adic_valuation (p - 1) ≥ n + 2 := sorry

end solution_l102_102052


namespace parabola_vertex_l102_102825

theorem parabola_vertex : ∃ h k : ℝ, (∀ x : ℝ, 2 * (x - h)^2 + k = 2 * (x - 5)^2 + 3) ∧ h = 5 ∧ k = 3 :=
by {
  use 5,
  use 3,
  split,
  { intro x,
    simp },
  exact ⟨rfl, rfl⟩,
}

end parabola_vertex_l102_102825


namespace pure_imaginary_complex_number_l102_102313

variable (a : ℝ)

theorem pure_imaginary_complex_number:
  (a^2 + 2*a - 3 = 0) ∧ (a^2 + a - 6 ≠ 0) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l102_102313


namespace lines_concurrent_or_parallel_l102_102840

variable (A B C A1 B1 C1 A2 B2 C2 P Q : Type)

-- condition 1: Ratios of segments are preserved
axiom BA2_over_A2C_eq_A1C_over_BA1 :
  (A2 : BC.α) → (A1 : BC.α) → 
  (ratio BA2 A2C = ratio A1C BA1)
axiom CB2_over_B2A_eq_B1A_over_CB1 :
  (B2 : CA.α) → (B1 : CA.α) → 
  (ratio CB2 B2A = ratio B1A CB1)
axiom AC2_over_C2B_eq_C1B_over_AC1 :
  (C2 : AB.α) → (C1 : AB.α) → 
  (ratio AC2 C2B = ratio C1B AC1)

-- Goal: prove that the lines AA2, BB2, and CC2 are concurrent or parallel
theorem lines_concurrent_or_parallel 
  (AA2 : line A A2)
  (BB2 : line B B2)
  (CC2 : line C C2)  
  (concur_or_parallel : (concurrent AA2 BB2 CC2) ∨ (parallel AA2 BB2 CC2)) :
  concurrent_or_parallel AA2 BB2 CC2 :=
by sorry

end lines_concurrent_or_parallel_l102_102840


namespace range_of_a_l102_102299

theorem range_of_a (x a : ℝ) (p : (x + 1)^2 > 4) (q : x > a) 
  (h : (¬((x + 1)^2 > 4)) → (¬(x > a)))
  (sufficient_but_not_necessary : (¬((x + 1)^2 > 4)) → (¬(x > a))) : a ≥ 1 :=
sorry

end range_of_a_l102_102299


namespace number_of_real_z12_l102_102871

theorem number_of_real_z12 (z : ℂ) (h1 : z^48 = 1) : 
  {z | z^48 = 1}.count (λ z, ∃ r : ℝ, z^12 = r) = 24 := 
sorry

end number_of_real_z12_l102_102871


namespace cos_double_angle_l102_102352

theorem cos_double_angle (α : ℝ) (h1 : sin (π / 4 - α) = 3 / 5) (h2 : -π / 4 < α ∧ α < 0) :
  cos (2 * α) = 24 / 25 := 
sorry

end cos_double_angle_l102_102352


namespace smallest_period_2cos2x_div2_plus_1_l102_102254

def smallest_positive_period (f : ℝ → ℝ) : ℝ := 
  Inf {T : ℝ | T > 0 ∧ ∀ x, f (x + T) = f x }

theorem smallest_period_2cos2x_div2_plus_1 :
  smallest_positive_period (λ x, 2 * (Real.cos (x / 2)) ^ 2 + 1) = 2 * Real.pi := 
sorry

end smallest_period_2cos2x_div2_plus_1_l102_102254


namespace ratio_of_chocolate_to_regular_milk_l102_102535

def total_cartons : Nat := 24
def regular_milk_cartons : Nat := 3
def chocolate_milk_cartons : Nat := total_cartons - regular_milk_cartons

theorem ratio_of_chocolate_to_regular_milk (h1 : total_cartons = 24) (h2 : regular_milk_cartons = 3) :
  chocolate_milk_cartons / regular_milk_cartons = 7 :=
by 
  -- Skipping proof with sorry
  sorry

end ratio_of_chocolate_to_regular_milk_l102_102535


namespace pages_per_day_difference_l102_102515

theorem pages_per_day_difference :
  ∀ (total_pages_Ryan : ℕ) (days : ℕ) (pages_per_book_brother : ℕ) (books_per_day_brother : ℕ),
    total_pages_Ryan = 2100 →
    days = 7 →
    pages_per_book_brother = 200 →
    books_per_day_brother = 1 →
    (total_pages_Ryan / days) - (books_per_day_brother * pages_per_book_brother) = 100 := 
by
  intros total_pages_Ryan days pages_per_book_brother books_per_day_brother
  intros h_total_pages_Ryan h_days h_pages_per_book_brother h_books_per_day_brother
  have h1 : total_pages_Ryan / days = 300 := by sorry
  have h2 : books_per_day_brother * pages_per_book_brother = 200 := by sorry
  rw [h1, h2]
  exact rfl

end pages_per_day_difference_l102_102515


namespace largest_independent_amount_l102_102002

theorem largest_independent_amount (n : ℕ) :
  ∃ s, ¬∃ a b c d e f g h i j : ℕ, s = a * (3^n) + b * (3^(n-1) * 5) + c * (3^(n-2) * 5^2) + d * (3^(n-3) * 5^3) + 
        e * (3^(n-4) * 5^4) + f * (3^(n-5) * 5^5) + g * (3^(n-6) * 5^6) + h * (3^(n-7) * 5^7) + i * (3^(n-8) * 5^8) + 
        j * (5^n) := (5^(n+1)) - 2 * (3^(n+1)) :=
sorry

end largest_independent_amount_l102_102002


namespace sum_put_at_simple_interest_l102_102217

theorem sum_put_at_simple_interest (P R : ℝ) 
  (h : ((P * (R + 3) * 2) / 100) - ((P * R * 2) / 100) = 300) : 
  P = 5000 :=
by
  sorry

end sum_put_at_simple_interest_l102_102217


namespace min_max_values_on_interval_l102_102842

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l102_102842


namespace traveler_distance_from_start_l102_102553

def net_travel_distance (north south west east : ℕ) : ℝ :=
  let net_north := north - south
  let net_west := west - east
  real.sqrt ((net_north ^ 2) + (net_west ^ 2))

theorem traveler_distance_from_start :
  net_travel_distance 15 9 7 3 = 2 * real.sqrt 13 :=
by
  sorry

end traveler_distance_from_start_l102_102553


namespace net_rate_26_dollars_per_hour_l102_102198

/-
Conditions:
1. The driver travels for 2 hours.
2. The speed of the car is 60 miles per hour.
3. The car gets 30 miles per gallon of gasoline.
4. She is paid $0.50 per mile.
5. The cost of gasoline is $2.00 per gallon.
-/

/-
Question:
Prove that the net rate of pay, in dollars per hour, after the gasoline expense is 26 dollars.
-/

def net_rate_of_pay 
  (travel_hours : ℕ) 
  (speed_mph : ℕ) 
  (efficiency_mpg : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ) : ℚ :=
  let distance := travel_hours * speed_mph,
      gallons := distance / efficiency_mpg,
      earnings := distance * pay_per_mile,
      gas_expense := gallons * gas_cost_per_gallon,
      net_earnings := earnings - gas_expense,
      net_rate := net_earnings / travel_hours
  in net_rate

theorem net_rate_26_dollars_per_hour :
  net_rate_of_pay 2 60 30 (1/2) 2 = 26 := by
  sorry

end net_rate_26_dollars_per_hour_l102_102198


namespace bicycle_license_count_l102_102939

/-- The number of possible bicycle licenses where the license consists of 
    a letter (either 'B' or 'C') followed by six digits and ends with the digit 5. -/
theorem bicycle_license_count : 
  let letter_choices := 2 in
  let digit_choices := 10 in
  let fixed_last_digit := 1 in
  letter_choices * (digit_choices ^ 5) * fixed_last_digit = 200000 := 
by
  simp only [letter_choices, digit_choices, fixed_last_digit]
  norm_num
  sorry

end bicycle_license_count_l102_102939


namespace triangle_area_polar_coords_l102_102383

theorem triangle_area_polar_coords (rA θA rB θB : ℝ) (hA : rA = 2 ∧ θA = π / 6) (hB : rB = 4 ∧ θB = π / 3) :
  1 / 2 * rA * rB * sin (θB - θA) = 2 :=
  sorry

end triangle_area_polar_coords_l102_102383


namespace effective_annual_rate_nominal_8_percent_l102_102831

theorem effective_annual_rate_nominal_8_percent : 
  ∃ (n : ℕ), (1 + 0.08 / n) ^ n - 1 = 0.0816 ∧ n = 2 :=
sorry

end effective_annual_rate_nominal_8_percent_l102_102831


namespace minimal_area_of_inscribed_equilateral_triangle_l102_102495

theorem minimal_area_of_inscribed_equilateral_triangle 
  (a T : ℝ) 
  (h1 : T > 0) 
  (h2 : a > 0)
  (h3 : isosceles_triangle ABC a T) -- Assuming isosceles_triangle is defined
  (h4 : equilateral_triangle_inscribed ABC A1B1C1) -- Assuming equilateral_triangle_inscribed is defined
  : ∃ t : ℝ, t = (2 * a * T / (4 * T + a^2 * real.sqrt 3))^2 * real.sqrt 3 :=
sorry

end minimal_area_of_inscribed_equilateral_triangle_l102_102495


namespace sum_common_elements_in_arith_progressions_l102_102571

theorem sum_common_elements_in_arith_progressions :
  let
    a1 := 3; d1 := 4;
    b1 := 2; d2 := 7;
    first_ap (n : ℕ) := a1 + (n - 1) * d1;
    second_ap (m : ℕ) := b1 + (m - 1) * d2;
    common_elements :=
      {x : ℕ | ∃ n m, n ≤ 100 ∧ m ≤ 100 ∧ first_ap n = x ∧ second_ap m = x};
    sum_common := Finset.sum (common_elements.to_finset) id
  in sum_common = 2870 :=
by
  -- proof goes here
  sorry

end sum_common_elements_in_arith_progressions_l102_102571


namespace sum_of_inscribed_angles_in_pentagon_l102_102927

theorem sum_of_inscribed_angles_in_pentagon (h : ∀ (p : ℕ), p = 5) : 
  ∑ i in finset.range 5, ∠ (circle 1 0).arc (pentagon_side i) = 180 := sorry

end sum_of_inscribed_angles_in_pentagon_l102_102927


namespace triangle_area_is_18_l102_102888

-- Define the vertices of the triangle
def A := (2 : ℝ, 1 : ℝ)
def B := (2 : ℝ, 7 : ℝ)
def C := (8 : ℝ, 4 : ℝ)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)|

-- Problem statement: Prove that the area of the given triangle is 18 square units
theorem triangle_area_is_18 : triangle_area A B C = 18 := by
  sorry

end triangle_area_is_18_l102_102888


namespace amara_remaining_clothes_l102_102560

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end amara_remaining_clothes_l102_102560


namespace sector_central_angle_l102_102308

theorem sector_central_angle (S l : ℝ) (hS : S = 4) (hl : l = 4) : 
  let r := 2 in
  l = r * 2 :=
by
  sorry

end sector_central_angle_l102_102308


namespace ellipse1_standard_eq_area_quadrilateral_constant_l102_102300

-- Definitions and conditions based on the original problem
def Ellipse1 (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

def Ellipse2 (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Given points P, A, B and their relationships
def PointOnEllipse2 (P : ℝ × ℝ) : Prop :=
  Ellipse2 P.1 P.2

def PointA (P A : ℝ × ℝ) : Prop :=
  A.1 = 2 * P.1 ∧ A.2 = 2 * P.2

def LineOPIntersects (O P A : ℝ × ℝ) (C1 : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ B : ℝ × ℝ, C1 (2 * P.1) (2 * P.2) A.1 A.2 ∧ C1 B.1 B.2 ∧
  A = (2 * P.1, 2 * P.2)

-- The proof statement to check the standard equation of ellipse C1
theorem ellipse1_standard_eq (a b : ℝ) (P A : ℝ × ℝ) :
  Ellipse1 (2 * P.1) (2 * P.2) 4 2 → PointOnEllipse2 P → PointA P A →
  (∀ x y, Ellipse1 x y 16 4 ↔ Ellipse1 x y a b) →
  (x y : ℝ) = true :=
by
  sorry

-- The proof statement to check the area of quadrilateral ACBD
theorem area_quadrilateral_constant (a b : ℝ) (P A C D : ℝ × ℝ) :
  Ellipse1 (2 * P.1) (2 * P.2) 4 2 → PointOnEllipse2 P → PointA P A →
  C.2 ≠ (8 * P.2 / √(P.1^2 + 16 * P.2^2) + 4 * A.2 / √(P.1^2 + 16 * A.2^2)) →
  D.2 ≠ (12 * P.2 / √(P.1^2 + 16 * P.2^2) + 1) →
  (8 * math.sqrt(3)) = (P A C D : ℝ) :=
by
  sorry

end ellipse1_standard_eq_area_quadrilateral_constant_l102_102300


namespace probability_smallest_divides_l102_102876

theorem probability_smallest_divides (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  let all_sets := S.powerset.filter (λ s, s.card = 3) in
  let valid_sets := all_sets.filter (λ s, ∃ a b c, s = {a, b, c} ∧ (a < b ∧ a < c) ∧ (b % a = 0 ∧ c % a = 0)) in
  (valid_sets.card : ℚ) / (all_sets.card : ℚ) = 11 / 20 :=
by {
  sorry
}

end probability_smallest_divides_l102_102876


namespace sum_first_12_terms_l102_102706

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 ^ (n - 1) else 2 * n - 1

def S (n : ℕ) : ℕ := 
  (Finset.range n).sum a

theorem sum_first_12_terms : S 12 = 1443 :=
by
  sorry

end sum_first_12_terms_l102_102706


namespace percentage_markdown_l102_102547

variable (P : ℝ) (X : ℝ)

noncomputable def original_price (P : ℝ) := P
noncomputable def sale_price (P : ℝ) := 0.80 * P
noncomputable def final_price (P : ℝ) := 0.64 * P

theorem percentage_markdown : 
  ∀ (P : ℝ), 
  P > 0 → 
  let S := 0.80 * P in 
  let F := 0.64 * P in
  F = S * (1 - X / 100) → 
  X = 20 
  := by
  intros P P_pos S F H
  rw [H]
  sorry

end percentage_markdown_l102_102547


namespace can_divide_and_reassemble_l102_102425

open Set

variables (N : ℕ) (assumption1 : ∃ (fig : Set (ℕ × ℕ)), area fig = N) (assumption2 : ∃ k : ℕ, k * k = N)

theorem can_divide_and_reassemble : 
  (∃ (triangles : list (Set (ℕ × ℕ))), length triangles = 5 ∧ 
  (∀ t ∈ triangles, area t = (N / 5)) ∧ 
  (union_of_triangles triangles = fig N) ∧ 
  (exists (square : Set (ℕ × ℕ)), area square = N ∧ reassemble_from_triangles triangles square)) :=
sorry

end can_divide_and_reassemble_l102_102425


namespace rectangle_area_l102_102162

theorem rectangle_area {A B C D : ℝ × ℝ}
  (hA : A = (-3, 1)) (hB : B = (1, 1)) (hC : C = (1, -2)) (hD : D = (-3, -2)) :
  let length := B.1 - A.1,
      width := A.2 - D.2 in
  length * width = 12 := by
  sorry

end rectangle_area_l102_102162


namespace grid_star_value_l102_102727

theorem grid_star_value (a b c d e h i l m n o p : ℕ) (H1 : a ≠ 0) (H2 : b ≠ 0) (H3 : c ≠ 0)
  (H4 : d ≠ 0) (H5 : e ≠ 0) (H6 : h ≠ 0) (H7 : i ≠ 0) (H8 : l ≠ 0) (H9 : m ≠ 0) (H10 : n ≠ 0)
  (H11 : o ≠ 0) (H12 : p ≠ 0) (H13 : e * 2 * 16 * h = i * 8 * 32 * l) :
    (g : ℕ) (H14 : g ≠ 0) : g = 1 :=
begin
  sorry
end

end grid_star_value_l102_102727


namespace polynomial_root_check_l102_102237

def polynomial (x : ℝ) : ℝ := x^3 - 5 * x^2 + 7 * x - 12

theorem polynomial_root_check : polynomial 4 = 0 :=
by
  simp [polynomial]
  norm_num
  sorry

end polynomial_root_check_l102_102237


namespace inversely_proportional_decrease_l102_102093

theorem inversely_proportional_decrease :
  ∀ {x y q c : ℝ}, 
  0 < x ∧ 0 < y ∧ 0 < c ∧ 0 < q →
  (x * y = c) →
  (((1 + q / 100) * x) * ((100 / (100 + q)) * y) = c) →
  ((y - (100 / (100 + q)) * y) / y) * 100 = 100 * q / (100 + q) :=
by
  intros x y q c hb hxy hxy'
  sorry

end inversely_proportional_decrease_l102_102093


namespace expression_evaluation_l102_102263

theorem expression_evaluation (x y : ℕ) (h1 : x = 2) (h2 : y = 3) :
  (x^(2*y - 3)) / (2^(-2) + 4^(-1)) = 16 := by
  sorry

end expression_evaluation_l102_102263


namespace compute_expression_l102_102046

noncomputable def roots : List ℂ := [r, s, t]

def polynomial := Polynomial.C 13 + Polynomial.X * Polynomial.C (-11) + Polynomial.X^2 * Polynomial.C 7 - Polynomial.X^3

theorem compute_expression (r s t : ℂ) 
  (hr : polynomial.eval r = 0) 
  (hs : polynomial.eval s = 0) 
  (ht : polynomial.eval t = 0) : 
  (r + s) / t + (s + t) / r + (t + r) / s = 38 / 13 := 
by 
  sorry

end compute_expression_l102_102046


namespace y_coordinate_of_equidistant_point_l102_102147

theorem y_coordinate_of_equidistant_point
  (y : ℝ)
  (h1 : dist (0, y) (3, 0) = dist (0, y) (-4, 5)) :
  y = 16 / 5 :=
by
  have h_dist1 := dist_eq (0, y) (3, 0)
  have h_dist2 := dist_eq (0, y) (-4, 5)
  calc
    dist (0, y) (3, 0)     = sqrt (3^2 + y^2) : by simp [dist]
                       ... = sqrt (9 + y^2) : by norm_num
                       ... = sqrt ((-4)^2 + (5 - y)^2) : by rw h1
                       ... = sqrt (16 + (5 - y)^2) : by norm_num
  sorry

end y_coordinate_of_equidistant_point_l102_102147


namespace area_of_enclosed_region_l102_102496

open Real

noncomputable def enclosedRegionArea : ℝ :=
  let s := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = sqrt (abs (p.1 - 1)) + sqrt (abs (p.2 - 1)) }
  1 -- This is the given answer

theorem area_of_enclosed_region :
  let S := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = sqrt (abs (p.1 - 1)) + sqrt (abs (p.2 - 1)) } in
  measure_theory.measure_theory.measure.measure_S S = 1 :=
sorry

end area_of_enclosed_region_l102_102496


namespace find_value_of_expression_l102_102714

theorem find_value_of_expression (a b : ℝ) (h : a + 2 * b - 1 = 0) : 3 * a + 6 * b = 3 :=
by
  sorry

end find_value_of_expression_l102_102714


namespace trig_identity_l102_102175

theorem trig_identity : 
  Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + 
  Real.cos (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 :=
by 
  sorry

end trig_identity_l102_102175


namespace remainder_1234567_div_256_l102_102971

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l102_102971


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102621

-- Conditions and Probability Definitions
noncomputable def prob_indicator (k : ℕ) : ℝ := (5 / 6) ^ (k - 1)

-- a) Expected total number of dice rolled
theorem expected_total_dice_rolled : 5 * (1 / (1 - 5/6)) = 30 := by
  sorry

-- b) Expected total number of points rolled
theorem expected_total_points_rolled :
  let expected_one_die_points : ℝ := 3.5 / (1 - 5/6)
  in 5 * expected_one_die_points = 105 := by
  sorry

-- c) Expected number of salvos
theorem expected_number_of_salvos :
  let expected_salvos (n : ℕ) : ℝ := 
    ∑ k in Finset.range n, (n.choose k * 5^(n - k) * (1/ (1 - 5/6))) / (6^n)
  in expected_salvos 5 ≈ 13.02 := by
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102621


namespace f_on_interval_neg1_0_l102_102989

noncomputable def f : ℝ → ℝ := λ x, if (2 ≤ x ∧ x ≤ 3) then x else sorry

theorem f_on_interval_neg1_0 : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) -> f x = 2 - x :=
by
  intros x h
  have h1 : f (2 - x) = 2 - x := sorry
  have h2 : f x = f (2 - x) := sorry
  rw [←h2, h1]
  done

end f_on_interval_neg1_0_l102_102989


namespace sum_of_exponents_l102_102086

theorem sum_of_exponents (x y z : ℕ) : 
  let expr := 40 * x^5 * y^9 * z^{14} in
  let simplified_expr := 2 * x * y * z^3 in
  ∑ i in [(1 : ℕ), 1, 3], i = 5 :=
by
  sorry

end sum_of_exponents_l102_102086


namespace sum_of_natural_numbers_not_equal_to_power_sums_l102_102820

theorem sum_of_natural_numbers_not_equal_to_power_sums (n : ℕ) (m : ℤ) : 
  n * (n + 1) / 2 ≠ 2^m + 3^m :=
sorry

end sum_of_natural_numbers_not_equal_to_power_sums_l102_102820


namespace find_M_range_of_f_range_of_b_and_roots_number_of_real_roots_l102_102452

-- Define the domain M
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4^x - 2^(x + 1)

-- Prove the first part: M is {x | x < 1 or x > 3}
theorem find_M : {x : ℝ | ∃ (x : ℝ), domain x} = {x : ℝ | x < 1 ∨ x > 3} := sorry

-- Prove the second part: Range of f(x)
theorem range_of_f : set.image (f) {x : ℝ | domain x} = {y : ℝ | (-1 ≤ y ∧ y < 0) ∨ (48 < y)} := sorry

-- Prove the third part: Range of b and number of real roots
theorem range_of_b_and_roots (b : ℝ) :
  (∃ x, domain x ∧ f x = b) ↔ (-1 ≤ b ∧ b < 0) ∨ (48 < b) := sorry

theorem number_of_real_roots (b : ℝ) : 
  (-∞ < b ∧ b < -1) ∨ (0 ≤ b ∧ b ≤ 48) → (∃! x, domain x ∧ f x = b) := sorry

end find_M_range_of_f_range_of_b_and_roots_number_of_real_roots_l102_102452


namespace sum_of_first_8_terms_log_seq_l102_102009

def geometric_sequence (n : ℕ) : ℝ

variable {a₄ a₅ : ℝ}
variable (log_seq_sum : ℝ)

-- Conditions
def cond1 := a₄ = 2
def cond2 := a₅ = 5

-- Definition of the sequence in terms of the provided conditions
axiom geom_seq_def : ∀ n : ℕ, geometric_sequence n = if n = 4 then a₄ else if n = 5 then a₅ else geometric_sequence n

theorem sum_of_first_8_terms_log_seq:
  cond1 → cond2 → log_seq_sum = 4 :=
begin
  sorry
end

end sum_of_first_8_terms_log_seq_l102_102009


namespace count_ordered_triplets_l102_102344

theorem count_ordered_triplets :
  { (a, b, c) : ℕ × ℕ × ℕ // a > 0 ∧ b > 0 ∧ c > 0 ∧ 30 * a + 50 * b + 70 * c ≤ 343 }.finite.to_finset.card = 30 :=
by
  -- Proof will be provided here
  sorry

end count_ordered_triplets_l102_102344


namespace initial_bacteria_count_is_42_l102_102822

-- Define conditions
def bacteria_tripling (t : ℕ) : ℕ := 3 ^ (t / 30)

def bacteria_after_5_minutes : ℕ := 1_239_220

def half_life_at_halfway (n : ℕ) : Prop :=
  n * bacteria_tripling (150) / 2 * bacteria_tripling (150) = bacteria_after_5_minutes

-- Define theorem to be proved
theorem initial_bacteria_count_is_42 (n : ℕ) (t : ℕ) (h1 : t = 300) (h2 : half_life_at_halfway n) :
  n = 42 :=
sorry

end initial_bacteria_count_is_42_l102_102822


namespace part1_part2_l102_102293

variable (a b c x : ℝ)

-- Condition: lengths of the sides of the triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Quadratic equation
def quadratic_eq (x : ℝ) : ℝ := (a + c) * x^2 - 2 * b * x + (a - c)

-- Proof problem 1: If x = 1 is a root, then triangle ABC is isosceles
theorem part1 (h : quadratic_eq a b c 1 = 0) : a = b :=
by
  sorry

-- Proof problem 2: If triangle ABC is equilateral, then roots of the quadratic equation are 0 and 1
theorem part2 (h_eq : a = b ∧ b = c) :
  (quadratic_eq a a a 0 = 0) ∧ (quadratic_eq a a a 1 = 0) :=
by
  sorry

end part1_part2_l102_102293


namespace min_max_f_on_interval_l102_102856

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_f_on_interval :
  ∃ min max, min = - (3 * Real.pi) / 2 ∧ max = (Real.pi / 2) + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min ∧ f x ≤ max) :=
sorry

end min_max_f_on_interval_l102_102856


namespace John_new_weekly_earnings_l102_102391

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end John_new_weekly_earnings_l102_102391


namespace loss_percentage_l102_102520

theorem loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : ((C - S) / C) * 100 = 50 := by
  -- We know that 8C = 16S
  have hC : C = 2 * S := by linarith [h]
  
  -- Substitute C = 2S in the loss percentage formula
  calc
    ((C - S) / C) * 100
      = (((2 * S) - S) / (2 * S)) * 100 : by rw [hC]
  ... = (S / (2 * S)) * 100 : by congr
  ... = (1 / 2) * 100 : by field_simp
  ... = 50 : by norm_num

end loss_percentage_l102_102520


namespace carpet_area_l102_102212

theorem carpet_area (L W A : ℝ) (hL : L = 12) (h1 : (L + sqrt (L^2 + W^2)) = 5 * W) (h2 : A = L * W) : 
  A ≈ 29.4 :=
by
  -- the proof will go here
  sorry

end carpet_area_l102_102212


namespace third_term_expansion_l102_102290

noncomputable def binom_expansion_third_term (n : ℕ) (a b : ℂ) : ℂ := ∑ k in finset.range (n + 1), nat.choose n k * a^(n - k) * b^k

-- The imaginary unit i in ℂ
def i : ℂ := complex.I

theorem third_term_expansion : binom_expansion_third_term 6 1 i = -15 :=
by
  have h0 : nat.choose 6 2 = 15 := by norm_num,
  have h1 : 1 ^ (6 - 2) = 1 := by norm_num,
  have h2 : i ^ 2 = -1 := by simp [i, complex.I, complex.pow_two, complex.I_mul_I],
  show 15 * 1 * (-1) = -15,
  rw [h0, h1, h2],
  norm_num

end third_term_expansion_l102_102290


namespace gina_minutes_of_netflix_l102_102642

-- Define the conditions given in the problem
def gina_chooses_three_times_as_often (g s : ℕ) : Prop :=
  g = 3 * s

def total_shows_watched (g s : ℕ) : Prop :=
  g + s = 24

def duration_per_show : ℕ := 50

-- The theorem that encapsulates the problem statement and the correct answer
theorem gina_minutes_of_netflix (g s : ℕ) (h1 : gina_chooses_three_times_as_often g s) 
    (h2 : total_shows_watched g s) :
    g * duration_per_show = 900 :=
by
  sorry

end gina_minutes_of_netflix_l102_102642


namespace geom_bio_students_difference_l102_102736

theorem geom_bio_students_difference (total_students geometry_students biology_students : ℕ)
  (h1 : total_students = 232) (h2 : geometry_students = 144) (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students,
      min_overlap := geometry_students + biology_students - total_students
  in max_overlap - min_overlap = 88 :=
by
  sorry

end geom_bio_students_difference_l102_102736


namespace hexagon_side_length_is_correct_l102_102894

-- Given definitions
def perimeter : ℝ := 43.56
def number_of_sides : ℕ := 6

-- Stating the theorem
theorem hexagon_side_length_is_correct : 
  (perimeter / number_of_sides) = 7.26 := 
begin
  sorry
end

end hexagon_side_length_is_correct_l102_102894


namespace certain_number_value_l102_102357

theorem certain_number_value :
  ∃ n : ℚ, 9 - (4 / 6) = 7 + (n / 6) ∧ n = 8 := by
sorry

end certain_number_value_l102_102357


namespace angle_of_inclination_of_tangent_at_1_l102_102278

def f (x : ℝ) : ℝ := 1 / x + 2 * x

theorem angle_of_inclination_of_tangent_at_1 :
  let df := deriv f 1 in
  ∃ α : ℝ, tan α = df ∧ α = π / 4 :=
by
  sorry

end angle_of_inclination_of_tangent_at_1_l102_102278


namespace paula_karl_age_sum_l102_102181

theorem paula_karl_age_sum :
  ∃ (P K : ℕ), (P - 5 = 3 * (K - 5)) ∧ (P + 6 = 2 * (K + 6)) ∧ (P + K = 54) :=
by
  sorry

end paula_karl_age_sum_l102_102181


namespace relationship_y_l102_102676

-- Definitions of the points based on the given problem conditions
def A : ℝ × ℝ := (-2, 4 + b)
def B : ℝ × ℝ := (-1, 2 + b)
def C : ℝ × ℝ := (1, -2 + b)

-- The proof problem translated into Lean 4
theorem relationship_y (b : ℝ) : 
  (4 + b > 2 + b) ∧ (2 + b > -2 + b) := 
by
  sorry

end relationship_y_l102_102676


namespace time_for_second_half_is_24_hours_l102_102934

-- Initial definitions
noncomputable def total_distance : ℝ := 40
noncomputable def halfway_distance : ℝ := total_distance / 2
noncomputable def initial_speed (v : ℝ) : ℝ := v
noncomputable def injured_speed (v : ℝ) : ℝ := v / 2
noncomputable def time_first_half (v : ℝ) : ℝ := halfway_distance / initial_speed v
noncomputable def time_second_half (v : ℝ) : ℝ := halfway_distance / injured_speed v

-- Given condition that second half takes 12 hours longer than first half
axiom condition (v : ℝ) : time_second_half v = time_first_half v + 12

-- Define what we need to prove
theorem time_for_second_half_is_24_hours (v : ℝ) (h : condition v) : time_second_half v = 24 :=
by
  sorry

end time_for_second_half_is_24_hours_l102_102934


namespace min_sum_OA_OB_l102_102202

theorem min_sum_OA_OB (a b : ℝ) (h₁ : 1 / a + 3 / b = 1) (h₂ : a > 0) (h₃ : b > 0) : 
  a + b ≥ 4 + 2 * real.sqrt 3 := 
sorry

end min_sum_OA_OB_l102_102202


namespace min_value_of_expression_l102_102287

theorem min_value_of_expression (c : ℝ) :
  (∀ x : ℝ, (1/3) * x^2 + 6 * x + 4 ≥ (1/3) * (-9:ℝ)^2 + 6 * (-9:ℝ) + 4) := 
begin
  sorry
end

end min_value_of_expression_l102_102287


namespace side_length_is_prime_l102_102865

-- Define the integer side length of the square
variable (a : ℕ)

-- Define the conditions
def impossible_rectangle (m n : ℕ) : Prop :=
  m * n = a^2 ∧ m ≠ 1 ∧ n ≠ 1

-- Declare the theorem to be proved
theorem side_length_is_prime (h : ∀ m n : ℕ, impossible_rectangle a m n → false) : Nat.Prime a := sorry

end side_length_is_prime_l102_102865


namespace increase_in_average_l102_102192

theorem increase_in_average 
  (total_runs_10 : ℕ) (runs_11th_innings : ℕ) (total_runs_11 : ℕ) 
  (h1 : total_runs_10 = 10 * 37)
  (h2 : runs_11th_innings = 81)
  (h3 : total_runs_11 = total_runs_10 + runs_11th_innings) :
  11 * (37 + 4) = total_runs_11 :=
by
  rw [h1, h2, h3]
  sorry

end increase_in_average_l102_102192


namespace chosen_number_eq_l102_102207

-- Given a number x, if (x / 2) - 100 = 4, then x = 208.
theorem chosen_number_eq (x : ℝ) (h : (x / 2) - 100 = 4) : x = 208 := 
by
  sorry

end chosen_number_eq_l102_102207


namespace closest_approx_w_l102_102595

noncomputable def w : ℝ := ((69.28 * 123.57 * 0.004) - (42.67 * 3.12)) / (0.03 * 8.94 * 1.25)

theorem closest_approx_w : |w + 296.073| < 0.001 :=
by
  sorry

end closest_approx_w_l102_102595


namespace complex_number_coordinates_l102_102315

noncomputable def z := (3 - 2 * complex.I) / (complex.I ^ 2015)

theorem complex_number_coordinates :
  (complex.re z, complex.im z) = (2, 3) := 
by
  sorry

end complex_number_coordinates_l102_102315


namespace correct_position_of_neg3_l102_102105

theorem correct_position_of_neg3 (A B C D : Prop) :
  A = (-3 < -4) ∧
  B = (-3 > -2 ∧ -3 < 0) ∧
  C = (1 - 4 = -3) ∧
  D = (| -3 | = -3) →
  (C) :=
by 
  intros h,
  sorry

end correct_position_of_neg3_l102_102105


namespace even_numbers_count_l102_102127
-- Import necessary libraries

-- Define the sequence {a_n}
def seq_a : ℕ → ℕ
| 0       := 1
| (n + 1) := ∑ k in Finset.range(⌊Real.sqrt (n + 1)⌋₊ + 1), seq_a (n + 1 - k * k)

-- Define the main theorem
theorem even_numbers_count : 
  500 ≤ Finset.card (Finset.filter (λ n, seq_a n % 2 = 0) (Finset.range (10^6))) :=
sorry

end even_numbers_count_l102_102127


namespace floor_length_is_twelve_l102_102213

-- Definitions based on the conditions
def floor_width := 10
def strip_width := 3
def rug_area := 24

-- Problem statement
theorem floor_length_is_twelve (L : ℕ) 
  (h1 : rug_area = (L - 2 * strip_width) * (floor_width - 2 * strip_width)) :
  L = 12 := 
sorry

end floor_length_is_twelve_l102_102213


namespace chandra_monster_hunt_l102_102236

theorem chandra_monster_hunt : 
  let num_monsters : ℕ → ℕ := λ n, 2 * (2 ^ n)
  (num_monsters 0 + num_monsters 1 + num_monsters 2 + num_monsters 3 + num_monsters 4) = 62 :=
by
  sorry

end chandra_monster_hunt_l102_102236


namespace percent_asian_west_is_53_l102_102577

def population_table := {
  NE : ℕ × ℕ × ℕ × ℕ, -- (White, Black, Asian, Other)
  MW : ℕ × ℕ × ℕ × ℕ,
  South : ℕ × ℕ × ℕ × ℕ,
  West : ℕ × ℕ × ℕ × ℕ 
}

def table : population_table := {
  NE := (50, 8, 2, 2),
  MW := (60, 7, 3, 2),
  South := (70, 22, 4, 3),
  West := (40, 3, 10, 5)
}

def total_asian_population (t : population_table) : ℕ :=
  (t.NE.2.2) + (t.MW.2.2) + (t.South.2.2) + (t.West.2.2)

def asian_population_in_west (t : population_table) : ℕ :=
  t.West.2.2

def percent_asian_in_west (t : population_table) : ℚ :=
  (asian_population_in_west t : ℚ) / (total_asian_population t : ℚ) * 100

theorem percent_asian_west_is_53 : percent_asian_in_west table ≈ 53 := begin
  sorry
end

end percent_asian_west_is_53_l102_102577


namespace asymptotes_of_hyperbola_l102_102012

noncomputable def hyperbola_asymptotes (a c : ℝ) :=
  let b := real.sqrt (c ^ 2 - a ^ 2) in
  let slope := b / a in
  (λ x : ℝ, slope * x, λ x : ℝ, -slope * x)

theorem asymptotes_of_hyperbola :
  let f := (const : ℝ → ℝ) 0 in
  let p := λ y, y ^ 2 = -12 * const x in
  let h := λ x y, (x ^ 2 / (2 * real.sqrt 2) ^ 2 - y ^ 2 = 1) in
  let focus := (-3, 0) in
  let a := 2 * real.sqrt 2 in
  hyperbola_asymptotes a 3 =
  (λ x : ℝ, (real.sqrt 2 / 4) * x, λ x : ℝ, -(real.sqrt 2 / 4) * x) ↔
  bijective (h - focus) :=
begin
  sorry
end

end asymptotes_of_hyperbola_l102_102012


namespace amara_clothing_remaining_l102_102559

theorem amara_clothing_remaining :
  (initial_clothing - donated_first - donated_second - thrown_away = remaining_clothing) :=
by
  let initial_clothing := 100
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining_clothing := 65
  sorry

end amara_clothing_remaining_l102_102559


namespace find_a_for_purely_imaginary_l102_102690

noncomputable def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_a_for_purely_imaginary (a : ℝ) :
  isPurelyImaginary (⟨(2 + a, 0), (2 - a, -1)⟩ / ⟨1, 1⟩) → a = -2 := by
  simp [isPurelyImaginary]
  sorry

end find_a_for_purely_imaginary_l102_102690


namespace range_of_k_l102_102335

theorem range_of_k (x y k : ℝ) (h1 : 2 * x - 3 * y = 5) (h2 : 2 * x - y = k) (h3 : x > y) : k > -5 :=
sorry

end range_of_k_l102_102335


namespace fence_length_l102_102288

noncomputable def horizontal_distance (lower_base upper_base : ℝ) : ℝ :=
  (lower_base - upper_base) / 2

noncomputable def slant_height (height horizontal_distance : ℝ) : ℝ :=
  real.sqrt (height^2 + horizontal_distance^2)

noncomputable def perimeter (lower_base upper_base slant_height : ℝ) : ℝ :=
  lower_base + upper_base + 2 * slant_height

-- Given the conditions
def lower_base := 40.0
def upper_base := 20.0
def height := 10.0

-- Translate these to perimeter calculation
def horiz_dist := horizontal_distance lower_base upper_base
def slant_ht := slant_height height horiz_dist
def yard_perimeter := perimeter lower_base upper_base slant_ht

-- Theorem stating the required proof
theorem fence_length : yard_perimeter ≈ 88.28 := by
  sorry

end fence_length_l102_102288


namespace problem1_problem2_l102_102042

noncomputable def f (x a b : ℝ) := |x + a^2| + |x - b^2|

theorem problem1 (a b x : ℝ) (h : a^2 + b^2 - 2 * a + 2 * b + 2 = 0) :
  f x a b >= 3 ↔ x <= -0.5 ∨ x >= 1.5 :=
sorry

theorem problem2 (a b x : ℝ) (h : a + b = 4) :
  f x a b >= 8 :=
sorry

end problem1_problem2_l102_102042


namespace find_C_coordinates_l102_102430

open Real

noncomputable def pointC_coordinates (A B : ℝ × ℝ) (hA : A = (-1, 0)) (hB : B = (3, 8)) (hdist : dist A C = 2 * dist C B) : ℝ × ℝ :=
  (⟨7 / 3, 20 / 3⟩)

theorem find_C_coordinates :
  ∀ (A B C : ℝ × ℝ), 
  A = (-1, 0) → B = (3, 8) → dist A C = 2 * dist C B →
  C = (7 / 3, 20 / 3) :=
by 
  intros A B C hA hB hdist
  -- We will use the given conditions and definitions to find the coordinates of C
  sorry

end find_C_coordinates_l102_102430


namespace roots_sum_l102_102777

theorem roots_sum (a b : ℝ) 
  (h1 : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
          (roots_of (λ x, x^3 - 8 * x^2 + a * x - b) = {r1, r2, r3}) ) :
  a + b = 31 :=
sorry

end roots_sum_l102_102777


namespace range_of_func_l102_102124

def func (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_func : set.range (λ x, func x) ∩ set.Icc (-1 : ℝ) 2 = set.Icc 3 6 :=
by
sorry

end range_of_func_l102_102124


namespace find_rates_l102_102056

theorem find_rates
  (d b p t_p t_b t_w: ℕ)
  (rp rb rw: ℚ)
  (h1: d = b + 10)
  (h2: b = 3 * p)
  (h3: p = 50)
  (h4: t_p = 4)
  (h5: t_b = 2)
  (h6: t_w = 5)
  (h7: rp = p / t_p)
  (h8: rb = b / t_b)
  (h9: rw = d / t_w):
  rp = 12.5 ∧ rb = 75 ∧ rw = 32 := by
  sorry

end find_rates_l102_102056


namespace value_of_T_l102_102255

theorem value_of_T : (∏ k in Finset.range 11, (3^2^k + 1)) = (1 / 2) * (3^2048 - 1) :=
by sorry

end value_of_T_l102_102255


namespace exists_monic_polynomial_with_root_l102_102269

theorem exists_monic_polynomial_with_root (r : ℝ) :
  ∃ p : Polynomial ℚ, Polynomial.monic p ∧ Polynomial.degree p = 4 ∧
  Polynomial.coeff p (numer (RootSpaceOfRoot r (Polynomial.map Polynomial.C Polynomial.X)) = 1 ∧
  ∀ (x : ℝ), Polynomial.eval x p = 0 ↔ x = r ∨ x = -r := 
  sorry

end exists_monic_polynomial_with_root_l102_102269


namespace find_angle_A_l102_102367

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c) :
  A = Real.pi / 3 :=
sorry

end find_angle_A_l102_102367


namespace solution_set_inequality_l102_102284

theorem solution_set_inequality (x : ℝ) (h : x - 3 / x > 2) :
    -1 < x ∧ x < 0 ∨ x > 3 :=
  sorry

end solution_set_inequality_l102_102284


namespace minimum_value_l102_102054

open Classical

variable {a b c : ℝ}

theorem minimum_value (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a + b + c = 4) :
  36 ≤ (9 / a) + (16 / b) + (25 / c) :=
sorry

end minimum_value_l102_102054


namespace find_r_l102_102349

def k (r : ℝ) : ℝ := 5 / (3^r)
def k' (r : ℝ) : ℝ := 45 / (9^r)

theorem find_r (r : ℝ) : 5 = k r ∧ 45 = k' r → r = 2 :=
by
  intro h
  cases h with h1 h2
  sorry

end find_r_l102_102349


namespace complex_z_pow_2017_l102_102303

noncomputable def complex_number_z : ℂ := (1 + Complex.I) / (1 - Complex.I)

theorem complex_z_pow_2017 :
  (complex_number_z * (1 - Complex.I) = 1 + Complex.I) → (complex_number_z ^ 2017 = Complex.I) :=
by
  intro h
  sorry

end complex_z_pow_2017_l102_102303


namespace lines_parallel_l102_102340

def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_parallel : ∀ x1 x2 : ℝ, l1 x1 = l2 x2 → false := 
by
  intros x1 x2 h
  rw [l1, l2] at h
  sorry

end lines_parallel_l102_102340


namespace initial_cells_count_l102_102875

theorem initial_cells_count (cells_after_8_hours : ℕ) (death_rate : ℕ) (division_factor : ℕ) :
    cells_after_8_hours = 1284 → death_rate = 2 → division_factor = 2 → initial_cells = 9 :=
begin
    sorry
end

end initial_cells_count_l102_102875


namespace sum_of_first_17_terms_arithmetic_sequence_l102_102689

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem sum_of_first_17_terms_arithmetic_sequence
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 3 + a 9 + a 15 = 9) :
  sum_of_first_n_terms a 17 = 51 :=
sorry

end sum_of_first_17_terms_arithmetic_sequence_l102_102689


namespace num_tangent_lines_l102_102483

theorem num_tangent_lines : 
  (∀ (x x0 : ℝ), (x, 0) = (2, 0) → y = x^2 * exp x → y' = (x^2 + 2*x) * exp x → 
  (y - (x0^2 * exp x0)) = ((x0^2 + 2*x0) * exp x0) * (x - x0) → 
  (2 - x0) * (x0^2 - x0 - 4) = 0 →
  {x0 : ℝ | x0 = 0 ∨ (x0^2 - x0 - 4 = 0)}.card = 3) := 
by sorry

end num_tangent_lines_l102_102483


namespace binary_addition_correct_l102_102146

def bin_to_nat : list ℕ → ℕ
| []       := 0
| (b :: bs) := b + 2 * bin_to_nat bs

-- Define the two binary numbers in list form
def bin1 : list ℕ := [1, 0, 0, 1, 1, 0, 1]
def bin2 : list ℕ := [1, 0, 1, 1, 0, 1]

-- Prove the statement
theorem binary_addition_correct :
  bin_to_nat bin1 + bin_to_nat bin2 = 122 :=
by
  calc
    bin_to_nat bin1 + bin_to_nat bin2 = 77 + 45 : by simp [bin_to_nat, bin1, bin2]
    ... = 122 : by norm_num

end binary_addition_correct_l102_102146


namespace g_of_4_equals_neg_75_l102_102414

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 3

def g (x : ℝ) : ℝ :=
  let r := complex.roots_of_unity 3 in
  let s := complex.roots_of_unity 3 in
  let t := complex.roots_of_unity 3 in
  -1/3 * (x - r^2) * (x - s^2) * (x - t^2)

theorem g_of_4_equals_neg_75 (h : g 0 = 3) : g 4 = -75 :=
  sorry

end g_of_4_equals_neg_75_l102_102414


namespace find_inverse_function_l102_102361

noncomputable def inverse_of_exponential {a : ℝ} (ha : a > 0) (ha_ne_one : a ≠ 1) : (ℝ → ℝ) :=
  λ y, Real.log y / Real.log a

theorem find_inverse_function
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = Real.log x / Real.log a)
  (h4 : f 4 = 2) :
  f = (λ x, Real.log x / Real.log 2) :=
by
  -- Proof steps would go here
  sorry

end find_inverse_function_l102_102361


namespace sin_double_angle_l102_102350

theorem sin_double_angle (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2)
  (h : cos (π / 4 - θ) * cos (π / 4 + θ) = sqrt 2 / 6) : sin (2 * θ) = sqrt 7 / 3 :=
by
  sorry

end sin_double_angle_l102_102350


namespace movie_screening_guests_l102_102479

theorem movie_screening_guests
  (total_guests : ℕ)
  (women_percentage : ℝ)
  (men_count : ℕ)
  (men_left_fraction : ℝ)
  (children_left_percentage : ℝ)
  (children_count : ℕ)
  (people_left : ℕ) :
  total_guests = 75 →
  women_percentage = 0.40 →
  men_count = 25 →
  men_left_fraction = 1/3 →
  children_left_percentage = 0.20 →
  children_count = total_guests - (round (women_percentage * total_guests) + men_count) →
  people_left = (round (men_left_fraction * men_count)) + (round (children_left_percentage * children_count)) →
  (total_guests - people_left) = 63 :=
by
  intros ht hw hm hf hc hc_count hl
  sorry

end movie_screening_guests_l102_102479


namespace age_problem_l102_102138

theorem age_problem (d f : ℕ) (h₀ : d = 4) (h₁ : f = 28) : ∃ a : ℕ, f + a = 3 * (d + a) ∧ a = 8 :=
by
  use 8
  split
  . rw [←h₀, ←h₁]
    norm_num
  . rfl

end age_problem_l102_102138


namespace cube_volume_inscribed_in_sphere_l102_102195

variable (R : ℝ)

theorem cube_volume_inscribed_in_sphere (R : ℝ) (hR : R > 0) :
  ∃ (V : ℝ), V = (8 / 9) * real.sqrt 3 * R^3 :=
by
  use (8 / 9) * real.sqrt 3 * R^3
  sorry

end cube_volume_inscribed_in_sphere_l102_102195


namespace hiker_speed_is_correct_l102_102540

-- Definitions for the conditions
def cyclist_speed : ℝ := 20
def cyclist_time_passed : ℝ := 5 / 60 -- 5 minutes in hours
def hiker_time_to_catch_up : ℝ := 15 / 60 -- 15 minutes in hours

-- Distance covered by the cyclist in the time passed
def distance_cyclist : ℝ := cyclist_speed * cyclist_time_passed

-- Distance covered by the hiker in the time to catch up
def hiker_speed : ℝ := distance_cyclist / hiker_time_to_catch_up

-- Theorem statement
theorem hiker_speed_is_correct : hiker_speed = 20 / 3 := 
by
  sorry

end hiker_speed_is_correct_l102_102540


namespace function_increasing_iff_m_le_one_contrapositive_function_decreasing_if_m_gt_one_l102_102703

theorem function_increasing_iff_m_le_one (m : ℝ) :
  (∀ x > 0, (exp x - m) ≥ 0) → m ≤ 1 :=
sorry

theorem contrapositive_function_decreasing_if_m_gt_one (m : ℝ) :
  m > 1 → ¬ (∀ x > 0, (exp x - m) ≥ 0) :=
sorry

end function_increasing_iff_m_le_one_contrapositive_function_decreasing_if_m_gt_one_l102_102703


namespace person_A_catches_person_B_first_time_l102_102960

noncomputable def person_A_speed : ℝ := 100 / 60
noncomputable def person_B_speed : ℝ := 80 / 60
noncomputable def side_length : ℝ := 200
noncomputable def delay : ℝ := 15
noncomputable def perimeter : ℝ := 3 * side_length
noncomputable def lap_time (speed : ℝ) : ℝ :=
  perimeter / speed + 3 * delay

theorem person_A_catches_person_B_first_time:
  ∃ t : ℝ, t = 1470 ∧ 
    let lta := lap_time person_A_speed,
        ltb := lap_time person_B_speed in
    ∃ n : ℕ, n > 0 ∧
      t = n * lta + (t - n * ltb) / (person_A_speed - person_B_speed) := 
sorry

end person_A_catches_person_B_first_time_l102_102960


namespace honey_teas_l102_102444

-- Definitions corresponding to the conditions
def evening_cups := 2
def evening_servings_per_cup := 2
def morning_cups := 1
def morning_servings_per_cup := 1
def afternoon_cups := 1
def afternoon_servings_per_cup := 1
def servings_per_ounce := 6
def container_ounces := 16

-- Calculation for total servings of honey per day and total days until the container is empty
theorem honey_teas :
  (container_ounces * servings_per_ounce) / 
  (evening_cups * evening_servings_per_cup +
   morning_cups * morning_servings_per_cup +
   afternoon_cups * afternoon_servings_per_cup) = 16 :=
by
  sorry

end honey_teas_l102_102444


namespace unique_y_star_l102_102592

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_y_star :
  ∃! y : ℝ, star 4 y = 20 :=
by 
  sorry

end unique_y_star_l102_102592


namespace sum_of_ages_l102_102221

theorem sum_of_ages (a b c : ℕ) 
  (h1 : a = 18 + b + c) 
  (h2 : a^2 = 2016 + (b + c)^2) : 
  a + b + c = 112 := 
sorry

end sum_of_ages_l102_102221


namespace decreasing_interval_of_f_l102_102858

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem decreasing_interval_of_f :
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → f'(x) ≤ 0) → ∃ a b, f x = a ∧ f y = b :=
by
  sorry

end decreasing_interval_of_f_l102_102858


namespace number_of_valid_numbers_l102_102343

-- Define the conditions of the problem
def is_valid_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 600 ∧ (n % 10 = 4 ∨ n % 10 = 6) ∧ 
  let digits := (n / 100, (n / 10) % 10, n % 10) in 
  (digits.1 ≠ digits.2 ∧ digits.1 ≠ digits.3 ∧ digits.2 ≠ digits.3)

-- The proof statement
theorem number_of_valid_numbers : 
  ({ n : ℕ | is_valid_number n}.to_finset.card = 48) :=
sorry

end number_of_valid_numbers_l102_102343


namespace min_max_values_on_interval_l102_102845

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l102_102845


namespace evaluate_expression_l102_102505

theorem evaluate_expression : (3^1 - 6 + 4^2 - 3)⁻¹ * 4 = (2 : ℤ) / 5 :=
by
  sorry

end evaluate_expression_l102_102505


namespace number_of_divisors_not_multiples_of_35_l102_102045

theorem number_of_divisors_not_multiples_of_35 :
  let n := 2^30 * 3^15 * 5^10 * 7^7 in
  let total_divisors := (30 + 1) * (15 + 1) * (10 + 1) * (7 + 1) in
  let divisors_multiples_of_35 := (30 + 1) * (15 + 1) * (9 + 1) * (6 + 1) in
  total_divisors - divisors_multiples_of_35 = 8288 := 
by 
  let n := 2^30 * 3^15 * 5^10 * 7^7
  let total_divisors := (30 + 1) * (15 + 1) * (10 + 1) * (7 + 1)
  let divisors_multiples_of_35 := (30 + 1) * (15 + 1) * (9 + 1) * (6 + 1)
  sorry

end number_of_divisors_not_multiples_of_35_l102_102045


namespace remainder_div_1234567_256_l102_102966

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l102_102966


namespace part_a_part_b_part_c_l102_102617

noncomputable def expected_total_dice_rolls : ℕ := 30

noncomputable def expected_total_points_rolled : ℕ := 105

noncomputable def expected_number_of_salvos : ℚ := 13.02

theorem part_a :
  (let total_dice := 5
   in ∀ (salvo_rolls : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → salvo_rolls die = 6)
   → (total_dice * 6) = expected_total_dice_rolls) :=
by {
  sorry
}

theorem part_b :
  (let total_dice := 5
   in ∀ (points_rolled : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → points_rolled die = 21)
   → (total_dice * 21) = expected_total_points_rolled) :=
by {
  sorry
}

theorem part_c :
  (let total_dice := 5
   in ∀ (salvos : ℕ → ℚ),
   (∀ die : ℕ, die < total_dice → salvos die = 13.02)
   → (total_dice * 13.02) = expected_number_of_salvos) :=
by {
  sorry
}

end part_a_part_b_part_c_l102_102617


namespace filling_time_without_leak_is_12_l102_102189

variable (T : ℝ)

def filling_rate_without_leak (T : ℝ) : ℝ := 1 / T
def filling_rate_with_leak (T : ℝ) : ℝ := 1 / (T + 2)
def leak_rate : ℝ := 1 / 84

theorem filling_time_without_leak_is_12 (h1 : filling_rate_without_leak T - leak_rate = filling_rate_with_leak T) : T = 12 := by
  sorry

end filling_time_without_leak_is_12_l102_102189


namespace correct_statements_count_l102_102693

-- Definitions based on the conditions in the problem
def statement1 : Prop :=
  "A figure formed by three line segments is called a triangle"

def statement2 : Prop :=
  "The angle bisector of a triangle is a ray"

def statement3 : Prop :=
  "The lines containing the altitudes of a triangle intersect at a point, which is either inside or outside the triangle"

def statement4 : Prop :=
  "Every triangle has three altitudes, three medians, and three angle bisectors"

def statement5 : Prop :=
  "The three angle bisectors of a triangle intersect at a point, and this point is inside the triangle"

-- Translate the original problem into a Lean theorem statement
theorem correct_statements_count :
  let correct_statements := [statement4, statement5]
  in correct_statements.length = 2 :=
by sorry

end correct_statements_count_l102_102693


namespace ratio_of_boys_to_girls_l102_102548

theorem ratio_of_boys_to_girls (boys : ℕ) (students : ℕ) (h1 : boys = 42) (h2 : students = 48) : (boys : ℚ) / (students - boys : ℚ) = 7 / 1 := 
by
  sorry

end ratio_of_boys_to_girls_l102_102548


namespace largest_in_sequence_l102_102385

theorem largest_in_sequence : ∀ n : ℕ, n < 4 → [2, 35, 26, 1].nth n ≤ 35 := by
  intros n hn
  fin_cases n
  · simp
  · simp
  · simp
  · simp
  sorry

end largest_in_sequence_l102_102385


namespace total_number_of_eggs_l102_102473

theorem total_number_of_eggs 
  (cartons : ℕ) 
  (eggs_per_carton_length : ℕ) 
  (eggs_per_carton_width : ℕ)
  (egg_position_from_front : ℕ)
  (egg_position_from_back : ℕ)
  (egg_position_from_left : ℕ)
  (egg_position_from_right : ℕ) :
  cartons = 28 →
  egg_position_from_front = 14 →
  egg_position_from_back = 20 →
  egg_position_from_left = 3 →
  egg_position_from_right = 2 →
  eggs_per_carton_length = egg_position_from_front + egg_position_from_back - 1 →
  eggs_per_carton_width = egg_position_from_left + egg_position_from_right - 1 →
  cartons * (eggs_per_carton_length * eggs_per_carton_width) = 3696 := 
  by 
  intros
  sorry

end total_number_of_eggs_l102_102473


namespace S_div_T_is_one_half_l102_102773

def T (x y z : ℝ) := x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x + y + z = 1

def supports (a b c x y z : ℝ) := 
  (x >= a ∧ y >= b ∧ z < c) ∨ 
  (x >= a ∧ z >= c ∧ y < b) ∨ 
  (y >= b ∧ z >= c ∧ x < a)

def S (x y z : ℝ) := T x y z ∧ supports (1/4) (1/4) (1/2) x y z

theorem S_div_T_is_one_half :
  let area_T := 1 -- Normalizing since area of T is in fact √3 / 2 but we care about ratios
  let area_S := 1/2 * area_T -- Given by the problem solution
  area_S / area_T = 1/2 := 
sorry

end S_div_T_is_one_half_l102_102773


namespace necklace_ratio_l102_102709

variable {J Q H : ℕ}

theorem necklace_ratio (h1 : H = J + 5) (h2 : H = 25) (h3 : H = Q + 15) : Q / J = 1 / 2 := by
  sorry

end necklace_ratio_l102_102709


namespace projection_of_a_onto_b_l102_102342

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (3, -4)

-- Function to compute the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Function to compute the projection of vector u onto vector v
def projection (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / magnitude v

-- The goal statement proving that the projection of a onto b is -2
theorem projection_of_a_onto_b :
  projection a b = -2 :=
by
  sorry

end projection_of_a_onto_b_l102_102342


namespace omega_values_l102_102834

theorem omega_values (ω : ℝ) : 
  (∀ x, 0 ≤ x → x ≤ π → (sin (ω * x)) ≤ sin (ω * (x + ε))) ∧ 
  (sin (4 * π * ω) = 0) → 
  (ω = 1 / 4 ∨ ω = 1 / 2) :=
begin
  sorry
end

end omega_values_l102_102834


namespace find_y_l102_102268

-- Define points A, B, C, D, and O
variables {A B C D O : Type}

-- Define the distances from O to other points
variables (AO BO CO DO BD : ℝ)
variables (φ : ℝ)

-- Given conditions
axiom hAO : AO = 5
axiom hBO : BO = 7
axiom hCO : CO = 12
axiom hDO : DO = 5
axiom hBD : BD = 9

-- Given φ = angle between AO and CO, and BO and DO
axiom hφ1 : φ = ∠ A O C
axiom hφ2 : φ = ∠ B O D

-- Law of Cosines for triangle BOD
axiom hcosφ : cos φ = -1/10

-- Now the proof statement
theorem find_y : ∃ y : ℝ, y = 13 :=
  sorry

end find_y_l102_102268


namespace even_function_f_l102_102680

-- Problem statement:
-- Given that f is an even function and that for x < 0, f(x) = x^2 - 1/x,
-- prove that f(1) = 2.

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - 1/x else 0

theorem even_function_f {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x))
  (h_neg_def : ∀ x, x < 0 → f x = x^2 - 1/x) : f 1 = 2 :=
by
  -- Proof body (to be completed)
  sorry

end even_function_f_l102_102680


namespace handshake_total_l102_102229

noncomputable def men := { x : ℝ | x ∈ {1, 2, 3, 4} } -- Heights 1 to 4 for men (example)
noncomputable def women := { x : ℝ | x ∈ {1, 2, 3, 4} } -- Heights 1 to 4 for women (example)

def handshake_count (n : ℕ) : ℕ := n * (n - 1) / 2

theorem handshake_total : handshake_count 4 + handshake_count 4 = 12 :=
by
  sorry

end handshake_total_l102_102229


namespace correct_factorization_l102_102898

variable (a b : ℝ)

def expr1 := (a + 1) * (a - 1) = a^2 - 1
def expr2 := a * (a - b) = a^2 - ab
def expr3 := 4 * a^2 - 2 * a = 2 * a * (2 * a - 1)
def expr4 := a^2 + 2 * a + 3 = (a + 1)^2 + 2

theorem correct_factorization : expr3 := by
  sorry

end correct_factorization_l102_102898


namespace circumcircles_concur_of_acute_scalene_l102_102953

noncomputable def acute_scalene_triangle (A B C D E F K L : Point) (Γ : Circle) : Prop :=
  is_acute_triangle A B C ∧
  is_scalene_triangle A B C ∧
  is_in_circle A B C Γ ∧
  altitude A D B C ∧
  altitude B E A C ∧
  altitude C F A B ∧
  is_median B K A B C Γ ∧
  is_median C L A B C Γ

noncomputable def circumcircle_concurrence (B F K C E L D : Point) : Prop :=
  concurrent (circumcircle B F K) (circumcircle C E L) (circumcircle D E F)

theorem circumcircles_concur_of_acute_scalene {A B C D E F K L : Point} {Γ : Circle} :
  acute_scalene_triangle A B C D E F K L Γ → circumcircle_concurrence B F K C E L D :=
by
  sorry

end circumcircles_concur_of_acute_scalene_l102_102953


namespace pay_nineteen_rubles_l102_102543

/-- 
Given a purchase cost of 19 rubles, a customer with only three-ruble bills, 
and a cashier with only five-ruble bills, both having 15 bills each,
prove that it is possible for the customer to pay exactly 19 rubles.
-/
theorem pay_nineteen_rubles (purchase_cost : ℕ) (customer_bills cashier_bills : ℕ) 
  (customer_denomination cashier_denomination : ℕ) (customer_count cashier_count : ℕ) :
  purchase_cost = 19 →
  customer_denomination = 3 →
  cashier_denomination = 5 →
  customer_count = 15 →
  cashier_count = 15 →
  (∃ m n : ℕ, m * customer_denomination - n * cashier_denomination = purchase_cost 
  ∧ m ≤ customer_count ∧ n ≤ cashier_count) :=
by
  intros
  sorry

end pay_nineteen_rubles_l102_102543


namespace gas_cost_correct_l102_102453

def cost_to_fill_remaining_quarter (initial_fill : ℚ) (final_fill : ℚ) (added_gas : ℚ) (cost_per_litre : ℚ) : ℚ :=
  let tank_capacity := (added_gas * (1 / (final_fill - initial_fill)))
  let remaining_quarter_cost := (tank_capacity * (1 / 4)) * cost_per_litre
  remaining_quarter_cost

theorem gas_cost_correct :
  cost_to_fill_remaining_quarter (1/8) (3/4) 30 1.38 = 16.56 :=
by
  sorry

end gas_cost_correct_l102_102453


namespace final_value_A_eq_B_pow_N_l102_102887

-- Definitions of conditions
def compute_A (A B : ℕ) (N : ℕ) : ℕ :=
    if N ≤ 0 then 
        1 
    else 
        let rec compute_loop (A' B' N' : ℕ) : ℕ :=
            if N' = 0 then A' 
            else 
                let B'' := B' * B'
                let N'' := N' / 2
                let A'' := if N' % 2 = 1 then A' * B' else A'
                compute_loop A'' B'' N'' 
        compute_loop A B N

-- Theorem statement
theorem final_value_A_eq_B_pow_N (A B N : ℕ) : compute_A A B N = B ^ N :=
    sorry

end final_value_A_eq_B_pow_N_l102_102887


namespace closest_number_is_2100_l102_102598

def closest_number_to_850_over_0_42 : ℕ :=
  let options := [500, 1000, 2000, 2100, 4000]
  let target := 850 / 0.42
  options.minBy (λ n => abs (n - target))

theorem closest_number_is_2100 : closest_number_to_850_over_0_42 = 2100 :=
  sorry

end closest_number_is_2100_l102_102598


namespace function_even_and_decreasing_l102_102695

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp 1 + x) + Real.log (Real.exp 1 - x)

theorem function_even_and_decreasing : 
  (∀ x : ℝ, f(-x) = f(x)) ∧ (∀ x ∈ Set.Ioo 0 (Real.exp 1), f' x < 0) := by
  sorry

end function_even_and_decreasing_l102_102695


namespace contrapositive_negation_l102_102713

-- Define the main condition of the problem
def statement_p (x y : ℝ) : Prop :=
  (x - 1) * (y + 2) = 0 → (x = 1 ∨ y = -2)

-- Prove the contrapositive of statement_p
theorem contrapositive (x y : ℝ) : 
  (x ≠ 1 ∧ y ≠ -2) → ¬ ((x - 1) * (y + 2) = 0) :=
by 
  sorry

-- Prove the negation of statement_p
theorem negation (x y : ℝ) : 
  ((x - 1) * (y + 2) = 0) → ¬ (x = 1 ∨ y = -2) :=
by 
  sorry

end contrapositive_negation_l102_102713


namespace calculate_allie_distance_l102_102884

noncomputable def allie_skate_distance : ℝ :=
  let deg_to_rad (d : ℝ) : ℝ := d * real.pi / 180
  let AB := 150
  let vA := 10
  let vB := 9
  let angle := deg_to_rad 45 -- convert 45 degrees to radians
  let t := ((3000 * real.sqrt 2) - 4036.05) / 38 
  vA * t 

theorem calculate_allie_distance :
  allie_skate_distance = 238.7 := by
  sorry

end calculate_allie_distance_l102_102884


namespace num_tangent_lines_l102_102484

theorem num_tangent_lines : 
  (∀ (x x0 : ℝ), (x, 0) = (2, 0) → y = x^2 * exp x → y' = (x^2 + 2*x) * exp x → 
  (y - (x0^2 * exp x0)) = ((x0^2 + 2*x0) * exp x0) * (x - x0) → 
  (2 - x0) * (x0^2 - x0 - 4) = 0 →
  {x0 : ℝ | x0 = 0 ∨ (x0^2 - x0 - 4 = 0)}.card = 3) := 
by sorry

end num_tangent_lines_l102_102484


namespace solve_for_x_l102_102090

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end solve_for_x_l102_102090


namespace bn_geometric_an_general_formula_inequality_holds_min_positive_m_l102_102294

noncomputable def a : ℕ → ℚ
| 0     := 1
| (n+1) := 1 + 2 / (a n)

def b (n : ℕ) : ℚ := (a n - 2) / (a n + 1)

noncomputable def c (n : ℕ) : ℚ := n * b n

noncomputable def S (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), c k

theorem bn_geometric : ∀ n : ℕ, b n = (-1 / 2) ^ n :=
sorry

theorem an_general_formula : ∀ n : ℕ, a n = (2 ^ (n + 1) + (-1) ^ n) / (2 ^ n + (-1) ^ (n - 1)) :=
sorry

theorem inequality_holds (n m : ℕ) (hn : 0 < n) : 
  (m : ℚ) / 32 + 3 / 2 * S n + n * (-1 / 2) ^ (n + 1) - 1 / 3 * (-1 / 2) ^ n > 0 :=
sorry

theorem min_positive_m : ∃ m : ℕ, 0 < m ∧ ∀ n : ℕ, 0 < n →
  (m : ℚ) / 32 + 3 / 2 * S n + n * (-1 / 2) ^ (n + 1) - 1 / 3 * (-1 / 2) ^ n > 0 :=
begin
  use 11,
  sorry
end

end bn_geometric_an_general_formula_inequality_holds_min_positive_m_l102_102294


namespace existence_of_monic_poly_degree_4_with_root_sqrt2_add_sqrt5_l102_102272

theorem existence_of_monic_poly_degree_4_with_root_sqrt2_add_sqrt5 :
  ∃ p : Polynomial ℚ, Polynomial.monic p ∧ p.degree = 4 ∧ p.eval (Real.sqrt 2 + Real.sqrt 5) = 0 :=
by
  sorry

end existence_of_monic_poly_degree_4_with_root_sqrt2_add_sqrt5_l102_102272


namespace students_overlap_difference_l102_102731

theorem students_overlap_difference : 
  ∀ (total students geometry biology : ℕ),
  total = 232 → geometry = 144 → biology = 119 →
  (geometry + biology - total = 31) ∧ (min geometry biology = 119) →
  (min geometry biology - (geometry + biology - total) = 88) :=
by
  intros total geometry biology htotal hgeometry hbiology hconds,
  exact sorry

end students_overlap_difference_l102_102731


namespace rational_function_form_l102_102417

open Set Function

-- Let's define the positive rational numbers
def Q_plus := { x : ℚ // 0 < x }

-- The function f: Q_plus → Q_plus
variables (f : Q_plus → Q_plus)

-- Conditions
axiom cond1 : ∀ x : Q_plus, f (⟨1/x.1, by exact div_pos zero_lt_one x.2⟩) = f x
axiom cond2 : ∀ x : Q_plus, (1 + 1 / x.1) * f x = f ⟨x.1 + 1, add_pos x.2 zero_lt_one⟩

-- Given conditions, prove the form of f
theorem rational_function_form : ∃ a : ℚ, 0 < a ∧ ∀ p q : ℤ, (0 < p ∧ 0 < q ∧ gcd p q = 1) → f ⟨(p : ℚ) / (q : ℚ), div_pos (int.cast_pos.mpr ‹0 < p›) (int.cast_pos.mpr ‹0 < q›)⟩ = a * p * q := 
sorry

end rational_function_form_l102_102417


namespace min_max_values_l102_102848

noncomputable def f (x : ℝ) := cos x + (x + 1) * sin x + 1

theorem min_max_values :
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≥ - (3 * π) / 2) ∧
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≤ (π / 2 + 2)) :=
sorry

end min_max_values_l102_102848


namespace length_AF_parabola_l102_102330

theorem length_AF_parabola {y : ℝ} (hyp_A: y^2 = 4 * x) (hyp_F: F = (1, 0)) 
    (hyp_A_y_axis: A = (0, a)) (hyp_B: B = ((fst A + 1) / 2, (snd A + 0) / 2)) 
    (hyp_B_on_parabola: snd B ^ 2 = 2 * fst B):
    ∃ a : ℝ, dist A F = 3 := sorry

end length_AF_parabola_l102_102330


namespace solution_interval_l102_102275

theorem solution_interval (x : ℝ) (h1: x ≥ 2)
  (h2: sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2)
  : 11 ≤ x ∧ x ≤ 27 :=
sorry

end solution_interval_l102_102275


namespace min_sum_all_numbers_l102_102387

noncomputable def least_possible_sum : ℕ :=
  let x := 50
      y := 100 in
  -880 - 5 * x + 25 * y

theorem min_sum_all_numbers (y x z : ℕ)
  (h1: 1 + x - z = 100)
  (h2: x = y - 100)
  (h3: y = 100)
  (h4: z = 50) : least_possible_sum = 1370 :=
by
  sorry

end min_sum_all_numbers_l102_102387


namespace weight_of_mixture_is_3_52_kg_l102_102522

-- Definitions of weights and volumes
def weight_of_brand_a_per_liter : ℝ := 900 -- in gm
def weight_of_brand_b_per_liter : ℝ := 850 -- in gm
def total_volume : ℝ := 4 -- in liters
def ratio_a : ℝ := 3 -- ratio part of brand 'a'
def ratio_b : ℝ := 2 -- ratio part of brand 'b'
def total_parts : ℝ := ratio_a + ratio_b -- total parts in the mixture

-- Calculate volumes of each brand in the mixture
def volume_of_brand_a_in_mixture : ℝ := (ratio_a / total_parts) * total_volume
def volume_of_brand_b_in_mixture : ℝ := (ratio_b / total_parts) * total_volume

-- Calculate weights of each brand in the mixture
def weight_of_brand_a_in_mixture : ℝ := weight_of_brand_a_per_liter * volume_of_brand_a_in_mixture
def weight_of_brand_b_in_mixture : ℝ := weight_of_brand_b_per_liter * volume_of_brand_b_in_mixture

-- Calculate the total weight of the mixture in gm
def total_weight_in_gm : ℝ := weight_of_brand_a_in_mixture + weight_of_brand_b_in_mixture

-- Calculate the total weight of the mixture in kg
def total_weight_in_kg : ℝ := total_weight_in_gm / 1000

-- Theorem statement for the proof
theorem weight_of_mixture_is_3_52_kg :
  total_weight_in_kg = 3.52 := by
  sorry

end weight_of_mixture_is_3_52_kg_l102_102522


namespace isosceles_triangle_perimeter_l102_102669

theorem isosceles_triangle_perimeter (m : ℝ) (h : polynomial.eval 2 (polynomial.C 1 * polynomial.X^2 - polynomial.C (2 * m) * polynomial.X + polynomial.C (3 * m)) = 0)
  (h_roots_length : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧
    polynomial.C 1 * polynomial.X^2 - polynomial.C (2 * m) * polynomial.X + polynomial.C (3 * m) = polynomial.C 1 * (polynomial.X - polynomial.C x1) * (polynomial.X - polynomial.C x2) ∧
    (x1 = 2 ∨ x1 = 6) ∧ (x2 = 2 ∨ x2 = 6)) :
  2 * 6 + 2 = 14 :=
by
  sorry

end isosceles_triangle_perimeter_l102_102669


namespace number_of_matches_is_85_l102_102872

open Nat

/-- This definition calculates combinations of n taken k at a time. -/
def binom (n k : ℕ) : ℕ := n.choose k

/-- The calculation of total number of matches in the entire tournament. -/
def total_matches (groups teams_per_group : ℕ) : ℕ :=
  let matches_per_group := binom teams_per_group 2
  let total_matches_first_round := groups * matches_per_group
  let matches_final_round := binom groups 2
  total_matches_first_round + matches_final_round

/-- Theorem proving the total number of matches played is 85, given 5 groups with 6 teams each. -/
theorem number_of_matches_is_85 : total_matches 5 6 = 85 :=
  by
  sorry

end number_of_matches_is_85_l102_102872


namespace abs_inequality_proof_by_contradiction_l102_102143

theorem abs_inequality_proof_by_contradiction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  |a| > |b| :=
by
  let h := |a| ≤ |b|
  sorry

end abs_inequality_proof_by_contradiction_l102_102143


namespace AE_equals_EC_l102_102305

-- Definitions of isosceles and right triangles
structure IsoscelesRightTriangle (A B C : Point) :=
(hypotenuse : Bdist(A, C) = Bdist(A, B) + Bdist(B, C))
(right_angle : angle A B C = 90)

-- Points and lines
variables (A B C D E : Point) -- Points A,B,C form the triangle, D and E are additional points

-- Definitions
def BC_hypotenuse (h : IsoscelesRightTriangle A B C) : line BC := sorry -- Line BC is the hypotenuse
def D_on_BC := Bdist(B, D) + Bdist(D, C) = Bdist(B, C) -- D is on line BC
def DC_one_third : Bdist(D, C) = (1/3) * Bdist(B, C) := sorry -- DC is 1/3 of BC
def BE_perp_AD (h : ∃ p : Point, Bdist(B, p) = Bdist(A, p)) : Bperpendicular (A E D) (B E) := sorry -- BE is perpendicular to AD
def E_on_AC := Bdist(A, E) + Bdist(E, C) = Bdist(A, C) -- E is on line AC

-- Main theorem to prove
theorem AE_equals_EC (ABC_right : IsoscelesRightTriangle A B C) (D_on_BC : D_on_BC) (DC_one_third : DC_one_third) (BE_perp : BE_perp_AD ABC_right) (E_on_AC : E_on_AC):
  Bdist(A, E) = Bdist(E, C) :=
sorry

end AE_equals_EC_l102_102305


namespace amara_remaining_clothes_l102_102561

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end amara_remaining_clothes_l102_102561


namespace regular_pay_per_hour_l102_102924

theorem regular_pay_per_hour (R : ℝ) (h : 40 * R + 11 * (2 * R) = 186) : R = 3 :=
by
  sorry

end regular_pay_per_hour_l102_102924


namespace number_of_weeks_in_a_single_harvest_season_l102_102790

-- Define constants based on conditions
def weeklyEarnings : ℕ := 1357
def totalHarvestSeasons : ℕ := 73
def totalEarnings : ℕ := 22090603

-- Prove the number of weeks in a single harvest season
theorem number_of_weeks_in_a_single_harvest_season :
  (totalEarnings / weeklyEarnings) / totalHarvestSeasons = 223 := 
  by
    sorry

end number_of_weeks_in_a_single_harvest_season_l102_102790


namespace necessarily_positive_y_plus_z_l102_102078

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end necessarily_positive_y_plus_z_l102_102078


namespace find_angle_B_find_shortest_side_l102_102725

-- Definitions for the sides and angles in triangle ABC
variables {a b c : ℝ} {A B C : ℝ}

-- Given conditions
axiom triangle_abc 
  (h1 : b = Real.sqrt 14) 
  (h2 : sin A = 2 * sin C) 
  (h3 : (a + c) / (a + b) = (b - a) / c)

-- Proof statements to be proved
theorem find_angle_B 
  (h_cos_rule : cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (h_cos_formula : cos B = -1/2) : 
  B = 2 * Real.pi / 3 := 
by
  -- Proof of the first part - to be completed
  sorry

theorem find_shortest_side 
  (h_cos_B : B = 2 * Real.pi / 3) 
  (h_side_rule : ∀ (a c : ℝ), a = 2 * c) 
  (h_b_square : b^2 = a^2 + c^2 - 2 * a * c * (-1/2)) 
  (c_squared_eq2 : c^2 = 2) : 
  c = Real.sqrt 2 := 
by
  -- Proof of the second part - to be completed
  sorry

end find_angle_B_find_shortest_side_l102_102725


namespace problem_1_problem_2_problem_3_l102_102227

-- Define the coordinates and properties based on the given conditions
structure Point (α : Type*) := (x : α) (y : α) (z : α)
def midpoint (p1 p2 : Point ℝ) : Point ℝ :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2,
  z := (p1.z + p2.z) / 2
}

noncomputable def AO_perpendicular_Plane_BCD (A B D C G H O : Point ℝ) : Prop :=
  let BD := {x := B.x - D.x, y := B.y - D.y, z := B.z - D.z} in
  let AO := {x := A.x - O.x, y := A.y - O.y, z := A.z - O.z} in
  let N := (BD.x * AO.x + BD.y * AO.y + BD.z * AO.z) = 0 in
  N

noncomputable def cosine_dihedral_angle_GHB_BCD (G H B D C : Point ℝ) : ℝ :=
  let BG := {x := G.x - B.x, y := G.y - B.y, z := G.z - B.z} in
  let BH := {x := H.x - B.x, y := H.y - B.y, z := H.z - B.z} in
  let normal_GHB := {x := 3 * BG.x + BH.z, y := BG.x + 4 * BH.y + BH.z, z := -6} in
  let normal_BCD := {x := 0, y := 0, z := 1} in
  let dot_product := abs (normal_GHB.x * normal_BCD.x + normal_GHB.y * normal_BCD.y + normal_GHB.z * normal_BCD.z) in
  let magnitude_GHB := ((normal_GHB.x ^ 2 + normal_GHB.y ^ 2 + normal_GHB.z ^ 2) ^ (1 / 2)) in
  dot_product / magnitude_GHB

def existence_point_E (B C D : Point ℝ) : Prop :=
  let E := {x := B.x, y := (B.y + C.y) / 2, z := B.z} in
  let DE := {x := E.x - D.x, y := E.y - D.y, z := E.z - D.z} in
  let parallel := DE.x * 2 + DE.y * 1 + DE.z * (-6) = 0 in
  not parallel

-- The statements of the problems
theorem problem_1 (A B D C O : Point ℝ) (h1 : AO_perpendicular_Plane_BCD A B D C O) : true := sorry

theorem problem_2 (G H B D C : Point ℝ) (h2 : cosine_dihedral_angle_GHB_BCD G H B D C = 6 * real.sqrt(41) / 41) : true := sorry

theorem problem_3 (B C D : Point ℝ) (h3 : existence_point_E B C D) : false := sorry

end problem_1_problem_2_problem_3_l102_102227


namespace grid_no_stranded_black_square_closed_formula_l102_102662

-- Define the function N which represents the number of 2x(n) grids with no stranded black squares.
def N : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * 3^(n + 1) - 2^(n + 1)

-- We want to prove the closed formula for N.
theorem grid_no_stranded_black_square_closed_formula (n : ℕ) :
  N(n) = 2 * 3^n - 2^n :=
sorry

end grid_no_stranded_black_square_closed_formula_l102_102662


namespace area_of_triangle_F1PF2_l102_102670

open real

noncomputable def hyperbola_area_theorem : Prop :=
  ∀ (F₁ F₂ P : ℝ × ℝ),
  (F₁ ≠ F₂) →
  (P ≠ F₁ ∧ P ≠ F₂) →
  ((P.1)^2 - (P.2)^2 = 2) →
  (angle F₁ P F₂ = π / 3) →
  let a := dist F₁ P,
      b := dist F₂ P,
      c := dist F₁ F₂ in
  a * b * sin (π / 3) = 4 * sqrt 3
  
theorem area_of_triangle_F1PF2 : hyperbola_area_theorem :=
sorry

end area_of_triangle_F1PF2_l102_102670


namespace geometric_sequence_l102_102652

variable {α : Type*}

-- Given conditions
variables (a : ℕ → ℝ) (h1 : a 1 * a 2 * a 3 = 5) (h2 : a 7 * a 8 * a 9 = 10)
variable (positive_a : ∀ n, 0 < a n)

-- Main goal to prove
theorem geometric_sequence (a : ℕ → ℝ) (h1 : a 1 * a 2 * a 3 = 5) (h2 : a 7 * a 8 * a 9 = 10) (positive_a : ∀ n, 0 < a n) :
  a 4 * a 5 * a 6 = 5 * real.sqrt 2 :=
sorry

end geometric_sequence_l102_102652


namespace exists_prime_with_composite_sequence_l102_102410

theorem exists_prime_with_composite_sequence (n : ℕ) (hn : n ≠ 0) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬ Nat.Prime (p + k) :=
sorry

end exists_prime_with_composite_sequence_l102_102410


namespace jay_sarah_distance_proof_l102_102026

/-- Jay and Sarah decide to hike in opposite directions around a circular lake.
    Jay hikes at a rate of 1 mile every 20 minutes,
    while Sarah jogs at a rate of 3 miles every 40 minutes.
    How far apart will they be after 2 hours? -/
def distance_apart_after_two_hours : ℝ :=
  let jay_rate := 1 / 20 -- miles per minute
  let sarah_rate := 3 / 40 -- miles per minute
  let time_in_minutes := 120 -- 2 hours
  let jay_distance := jay_rate * time_in_minutes
  let sarah_distance := sarah_rate * time_in_minutes
  jay_distance + sarah_distance

theorem jay_sarah_distance_proof :
  distance_apart_after_two_hours = 15 := by
  -- Discard detailed calculations
  sorry

end jay_sarah_distance_proof_l102_102026


namespace correct_statements_l102_102636

noncomputable def F (a b : ℝ) : ℝ := (1 / 2) * (a + b - |a - b|)
def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.cos x
def G (x : ℝ) : ℝ := F (f x) (g x)

-- Prove that G(x) defined as above gives that statements (2), (4) and (5) are correct.
theorem correct_statements : 
  (∀ x, G x < 0 ↔ ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < x ∧ x < 2 * (k + 1) * Real.pi) ∧  -- statement (2)
  (∀ x : ℝ, G (5 * Real.pi / 4 - x) = G (5 * Real.pi / 4 + x)) ∧  -- statement (5)
  (let adj_highest_distance := 
    {l : ℝ × ℝ | ∃ k : ℤ, l = (2 * k * Real.pi + Real.pi / 2, 2 * (k + 1) * Real.pi + Real.pi / 2)}
  let adj_lowest_distance :=
    {l : ℝ × ℝ | ∃ k : ℤ, l = (2 * k * Real.pi, 2 * (k + 1) * Real.pi)}
  4 * (classical.some (adj_lowest_distance 1) - classical.some (adj_lowest_distance 0)) = 
  classical.some (adj_highest_distance 1) - classical.some (adj_highest_distance 0))  -- statement (4)
  := 
sorry

end correct_statements_l102_102636


namespace max_value_of_function_l102_102327

open Real

theorem max_value_of_function :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), (x + cos x) ≤ π / 2 ∧ (∃ y ∈ Icc (0 : ℝ) (π / 2), y + cos y = π / 2) := by
  sorry

end max_value_of_function_l102_102327


namespace length_of_box_l102_102812

theorem length_of_box (rate : ℕ) (width : ℕ) (depth : ℕ) (time : ℕ) (volume : ℕ) (length : ℕ) :
  rate = 4 →
  width = 6 →
  depth = 2 →
  time = 21 →
  volume = rate * time →
  length = volume / (width * depth) →
  length = 7 :=
by
  intros
  sorry

end length_of_box_l102_102812


namespace value_of_x_l102_102665

theorem value_of_x (f : ℤ → ℤ) (h : ∀ x, f(x) = 2 * x - 3) : 
  (2 * (f 8) - 21 = f (8 - 4)) :=
by
  -- Conditions given in the problem
  let x := 8
  -- Substituting the conditions
  have h1 : f x = 2 * x - 3 := h x
  have h2 : f (x - 4 ) = 2 * (x - 4 ) - 3 := h (x - 4)
  
  -- Showing the result
  suffices : 2 * (2 * x - 3) - 21 = 2 * (x - 4) - 3
   by sorry
  rw [← h1, ← h2]
  -- equation simplifying steps
  tactic.skip -- This is a placeholder for additional simplification steps

end value_of_x_l102_102665


namespace drawing_blue_ball_probability_l102_102741

noncomputable def probability_of_blue_ball : ℚ :=
  let total_balls := 10
  let blue_balls := 6
  blue_balls / total_balls

theorem drawing_blue_ball_probability :
  probability_of_blue_ball = 3 / 5 :=
by
  sorry -- Proof is omitted as per instructions.

end drawing_blue_ball_probability_l102_102741


namespace six_roles_assignment_l102_102542

def number_of_ways_to_assign_roles 
  (n_men : ℕ) (n_women : ℕ) (n_male_roles : ℕ) (n_female_roles : ℕ) (n_either_roles : ℕ) : ℕ :=
  let male_ways := Nat.permutations n_men n_male_roles
  let female_ways := Nat.permutations n_women n_female_roles
  let remaining := n_men + n_women - n_male_roles - n_female_roles
  let either_ways := remaining
  male_ways * female_ways * either_ways

theorem six_roles_assignment 
  (n_men : ℕ) (n_women : ℕ) (n_male_roles : ℕ) (n_female_roles : ℕ) (n_either_roles : ℕ) :
  n_men = 6 →
  n_women = 5 →
  n_male_roles = 3 →
  n_female_roles = 2 →
  n_either_roles = 1 →
  number_of_ways_to_assign_roles n_men n_women n_male_roles n_female_roles n_either_roles = 14400 :=
by
  intros h_men h_women h_male_roles h_female_roles h_either_roles
  rw [h_men, h_women, h_male_roles, h_female_roles, h_either_roles]
  simp [number_of_ways_to_assign_roles]
  sorry

end six_roles_assignment_l102_102542


namespace turnerSyndromeIsChromosomalNumberVariation_l102_102489

-- Definitions derived from conditions
def TurnerSyndrome (absenceOfSexChromosome : Prop) : Prop := absenceOfSexChromosome
def ChromosomalMutation (changesInChromosomes : Prop) : Prop := changesInChromosomes
def ChromosomalStructureVariation (deletionDuplicationInversionTranslocation : Prop) : Prop :=
  deletionDuplicationInversionTranslocation
def ChromosomalNumberVariation (individualChromosomesChanged : Prop) : Prop :=
  individualChromosomesChanged

-- Condition instantiations
def absenceOfSexChromosome : Prop := true -- Assume the condition given
def changesInChromosomes : Prop := true -- Assume the condition given
def deletionDuplicationInversionTranslocation : Prop := true -- Assume the condition given
def individualChromosomesChanged : Prop := true -- Assume the condition given

-- The theorem to prove
theorem turnerSyndromeIsChromosomalNumberVariation :
  TurnerSyndrome absenceOfSexChromosome →
  ChromosomalMutation changesInChromosomes →
  ChromosomalNumberVariation individualChromosomesChanged :=
by {
  intro h1,
  intro h2,
  sorry -- Proof to be filled in
}

end turnerSyndromeIsChromosomalNumberVariation_l102_102489


namespace qu_arrangements_l102_102436

theorem qu_arrangements :
  ∃ n : ℕ, n = 480 ∧ 
    ∀ (s : Finset Char), s = {'e', 'q', 'u', 'a', 't', 'i', 'o', 'n'} → 
    ∃ (t : Finset (Finset Char)), t.card = 1 ∧ t ⊆ s →
    ∃ u : Finset Char, u ⊆ (s \ {'q', 'u'}) ∧ u.card = 3 →
    (factorial 4) * (Finset.choose (s \ {'q', 'u'}) 3).card = n :=
by
  sorry

end qu_arrangements_l102_102436


namespace binom_307_307_l102_102980

theorem binom_307_307 : nat.choose 307 307 = 1 := 
by
  -- Proof skipped
  sorry

end binom_307_307_l102_102980


namespace EF_parallel_tangent_at_D_BFDE_is_rectangle_l102_102525

-- Definitions of given conditions

def point (x y : ℝ) : Type := { x : ℝ, y : ℝ } -- defining points in a 2D plane

-- Defining segments and semicircles as per the problem
def segment (A B : point) := line A B

def semicircle (A B : point) :=
  {p | dist A p = dist B p} -- A simplified way of representing semicircles but defining accurately is complex

-- Given points
variable (A B C D : point)
variable (S1 S2 S3 : semicircle)
variable (F E : point)

-- Conditions from the problem in Lean terms
axiom AC : segment A C
axiom B_on_AC : point_on_segment B AC
axiom S1_on_AB : point_on_semicircle S1 A B
axiom S2_on_BC : point_on_semicircle S2 B C
axiom S3_on_CA : point_on_semicircle S3 C A
axiom BD_perp_AC : perpendicular B D AC
axiom tangent_S1_S2 : tangent_point S1 F S2 E

-- Facts to prove
theorem EF_parallel_tangent_at_D :
  is_parallel (line F E) (tangent_at_point S3 D) := sorry

theorem BFDE_is_rectangle :
  is_rectangle B F D E := sorry

end EF_parallel_tangent_at_D_BFDE_is_rectangle_l102_102525


namespace largest_prime_factor_binom_300_150_l102_102890

theorem largest_prime_factor_binom_300_150 :
  let n := Nat.choose 300 150 in
  ∃ p : ℕ, Nat.Prime p ∧ 10 ≤ p ∧ p < 100 ∧
  (p > 75 → 3 * p < 300 ∧ p ∣ n) ∧
  ∀ q : ℕ, Nat.Prime q ∧ 10 ≤ q ∧ q < 100 → q ∣ n → q ≤ p :=
  sorry

end largest_prime_factor_binom_300_150_l102_102890


namespace arrange_three_digit_numbers_l102_102019

theorem arrange_three_digit_numbers :
  (∃ f : ℕ → ℕ,
     (∀ n : ℕ,
        let a := (f n) / 100,
            b := ((f n) % 100) / 10,
            c := (f n) % 10 in 
        a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ c = (f (n+1)) / 100
     )
  ) :=
sorry

end arrange_three_digit_numbers_l102_102019


namespace product_of_primes_l102_102464

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m | n, m = 1 ∨ m = n

theorem product_of_primes (p q : Nat) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h : 2 * p + 5 * q = 36) : 
  p * q = 26 :=
by
  sorry

end product_of_primes_l102_102464


namespace number_exists_ab_cd_mod_r_l102_102415

theorem number_exists_ab_cd_mod_r (p : ℕ) (A : finset ℕ) 
(hp : p > 7) (hp_prime : nat.prime p) (hA_size : (A.card : ℤ) ≥ ((p - 1) / 2)) 
(r : ℤ) : ∃ a b c d ∈ A, ((a * b - c * d) % p : ℤ) = r % p := 
sorry

end number_exists_ab_cd_mod_r_l102_102415


namespace distance_F_to_midpoint_DE_l102_102377

-- Define the parameters of the right triangle
structure Triangle :=
  (DE DF EF: ℝ)
  (is_right_triangle : DE^2 = DF^2 + EF^2)
  (DE_pos : DE > 0)
  (DF_pos : DF > 0)
  (EF_pos : EF > 0)

-- Given the specific right triangle DEF
def DEF : Triangle := {
  DE := 15,
  DF := 9,
  EF := 12,
  is_right_triangle := by {
    rw [pow_two, pow_two, pow_two],
    norm_num,
  },
  DE_pos := by norm_num,
  DF_pos := by norm_num,
  EF_pos := by norm_num,
}

-- Statement of the theorem to be proved
theorem distance_F_to_midpoint_DE : 
  ∀ (T : Triangle), T = DEF → 
  (T.DE / 2 = 7.5) := 
by {
  intro T,
  intro h,
  rw h,
  norm_num,
}

end distance_F_to_midpoint_DE_l102_102377


namespace sum_of_exponents_outside_radical_l102_102085

theorem sum_of_exponents_outside_radical (x y z : ℝ) :
  let expr := 40 * x^5 * y^9 * z^14
  in let simplified_expr := 2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3)
  in (if (simplified_expr = (2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3))) then (1 + 1 + 3 = 5) else false) := sorry

end sum_of_exponents_outside_radical_l102_102085


namespace platform_length_l102_102516

/-- Mathematical proof problem:
The problem is to prove that given the train's length, time taken to cross a signal pole and 
time taken to cross a platform, the length of the platform is 525 meters.
-/
theorem platform_length 
    (train_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) (P : ℕ) 
    (h_train_length : train_length = 450) (h_time_pole : time_pole = 18) 
    (h_time_platform : time_platform = 39) (h_P : P = 525) : 
    P = 525 := 
  sorry

end platform_length_l102_102516


namespace planeAPB_perpendicular_planeAPD_planeAPB_perpendicular_planeBPC_planeAPD_perpendicular_planeDPC_l102_102072

variables {A B C D P : Point}
variables {AB AC AD BC BD CD: Line}
variables {planeABCD planeAPB planeAPD planeBPC planeDPC: Plane}

-- Given: AB, AC, AD, BC, BD, and CD are sides of the rectangle ABCD.
-- P lies on a perpendicular from A to plane ABCD.

-- Let's assume we have the points and their relations captured in the conditions
def isRectangle : Prop :=
  collinear A B D → collinear B C D →
  dist A B = dist C D ∧
  dist B C = dist A D ∧
  dist A C = dist B D ∧
  ∠A B = ∠B C ∧ ∠A D = ∠D C

-- Point P lies on the line perpendicular from A to the plane ABCD
def pointPperpendicular : Prop :=
  ∃ l, (line_perpendicular_to_plane l planeABCD) ∧
       (on_line A l) ∧ (on_line P l) ∧ A ≠ P

-- The required theorems
theorem planeAPB_perpendicular_planeAPD :
  isRectangle ABCD → pointPperpendicular P planeABCD → 
  perpendicular planeAPB planeAPD := 
sorry

theorem planeAPB_perpendicular_planeBPC :
  isRectangle ABCD → pointPperpendicular P planeABCD →
  perpendicular planeAPB planeBPC := 
sorry

theorem planeAPD_perpendicular_planeDPC :
  isRectangle ABCD → pointPperpendicular P planeABCD → 
  perpendicular planeAPD planeDPC := 
sorry

end planeAPB_perpendicular_planeAPD_planeAPB_perpendicular_planeBPC_planeAPD_perpendicular_planeDPC_l102_102072


namespace shifted_function_is_cosine_l102_102699

-- Define the function f with the given conditions
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + (Real.pi / 6))

-- Condition: ω is determined such that the period T satisfies T = π
def ω_is_two : Prop := (2 : ℝ)

-- The shifted function to the left by π/6
def g (x : ℝ) : ℝ := cos (2 * x)

theorem shifted_function_is_cosine (x : ℝ) : g (x) = cos (2 * x) :=
by
  -- Shift operation on the function f should lead to g(x) = cos(2 * x)
  sorry

end shifted_function_is_cosine_l102_102699


namespace total_weight_of_10_moles_AlBr3_l102_102895

theorem total_weight_of_10_moles_AlBr3:
  let atomic_weight_Al := 26.98
  let atomic_weight_Br := 79.90
  let num_moles := 10
  let num_Br_in_AlBr3 := 3
  let molecular_weight_AlBr3 := atomic_weight_Al + (num_Br_in_AlBr3 * atomic_weight_Br)
  let total_weight := molecular_weight_AlBr3 * num_moles
  in total_weight = 2666.8 :=
by
  let atomic_weight_Al := 26.98
  let atomic_weight_Br := 79.90
  let num_moles := 10
  let num_Br_in_AlBr3 := 3
  let molecular_weight_AlBr3 := atomic_weight_Al + (num_Br_in_AlBr3 * atomic_weight_Br)
  let total_weight := molecular_weight_AlBr3 * num_moles
  have h : molecular_weight_AlBr3 = 266.68 := sorry
  have h2 : total_weight = 2666.8 := sorry
  exact h2

end total_weight_of_10_moles_AlBr3_l102_102895


namespace Mr_Caiden_payment_l102_102068

-- Defining the conditions as variables and constants
def total_roofing_needed : ℕ := 300
def cost_per_foot : ℕ := 8
def free_roofing : ℕ := 250

-- Define the remaining roofing needed and the total cost
def remaining_roofing : ℕ := total_roofing_needed - free_roofing
def total_cost : ℕ := remaining_roofing * cost_per_foot

-- The proof statement: 
theorem Mr_Caiden_payment : total_cost = 400 := 
by
  -- Proof omitted
  sorry

end Mr_Caiden_payment_l102_102068


namespace gamma_received_eight_donuts_l102_102247

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end gamma_received_eight_donuts_l102_102247


namespace set_of_points_M_l102_102292

open EuclideanGeometry

variables {Point : Type*} [MetricSpace Point]

-- Define the basic elements of the problem
def circle (O : Point) (R : ℝ) : set Point := setOf (λ P : Point, dist P O = R)
def tangent_line (N : Point) (P : Point) : set Point := setOf (λ Q : Point, dist N Q > 0 ∧ dist N Q = dist N P ∧∃ P' : Point, dist P' N = dist P N ∧ is_perpendicular (P' - N) (Q - N))

variables (O N : Point) (R : ℝ) (A B : Point)

-- M is the intersection point of the line AB and tangent at N
def M (O N A B : Point) (R : ℝ) : set Point := 
  { M : Point |
    (∃ t : ℝ, M = A + t • (B - A))
    ∧ (M ∈ tangent_line N (circle O R))
  }

-- The statement to prove
theorem set_of_points_M (O N : Point) (R : ℝ) :
  ∀ (A B : Point), 
  ∃ (line : set Point),
    (∀ (M : Point), M ∈ M O N A B R → M ∈ line)
    ∧ is_perpendicular (O - N) (line : Point).
sorry

end set_of_points_M_l102_102292


namespace expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102631

open ProbabilityTheory

/-- The expected total number of dice rolls until all five dice show sixes is 30 --/
theorem expected_total_number_of_dice_rolls (n : ℕ) (H : n = 5) : 
  (∑ k in range(n : ℕ), (E[(indicator(event_of_five_dice_roll)) k])) = 30 :=
sorry

/-- The expected total number of points rolled until all five dice show sixes is 105 --/
theorem expected_total_number_of_points (n : ℕ) (H : n = 5) : 
  (E[∑ i in range(n), points_rolled_until_six]) = 105 :=
sorry

/-- The expected number of salvos until all five dice show sixes is approximately 13.02 --/
theorem expected_number_of_salvos (n : ℕ) (H : n = 5) : 
  (E[number_of_salvos_to_get_all_sixes]) = 13.02 :=
sorry

end expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102631


namespace sticks_can_be_paired_l102_102427

noncomputable def can_pair_sticks (red_lengths blue_lengths : List ℝ) : Prop :=
  (red_lengths.Sum = 30) ∧ (blue_lengths.Sum = 30) ∧
  (red_lengths.Nodup = true) ∧ (blue_lengths.Nodup = true) ∧
  (red_lengths.length = 3) ∧ (blue_lengths.length = 5) →
  ∃ cut_reds cut_blues, 
    (cut_reds.Sum = 30) ∧ (cut_blues.Sum = 30) ∧ 
    (cut_reds.length = cut_blues.length) ∧
    (∀ (i : ℕ), i < cut_reds.length → cut_reds[i] = cut_blues[i])

theorem sticks_can_be_paired 
  (red_lengths blue_lengths : List ℝ) 
  (h : can_pair_sticks red_lengths blue_lengths) : 
  ∃ cut_reds cut_blues, 
    (cut_reds.Sum = 30) ∧ (cut_blues.Sum = 30) ∧ 
    (cut_reds.length = cut_blues.length) ∧
    (∀ (i : ℕ), i < cut_reds.length → cut_reds[i] = cut_blues[i]) :=
sorry

end sticks_can_be_paired_l102_102427


namespace abs_m_minus_1_greater_eq_abs_m_minus_1_l102_102633

theorem abs_m_minus_1_greater_eq_abs_m_minus_1 (m : ℝ) : |m - 1| ≥ |m| - 1 := 
sorry

end abs_m_minus_1_greater_eq_abs_m_minus_1_l102_102633


namespace tens_digit_of_13_pow_3007_l102_102498

theorem tens_digit_of_13_pow_3007 : 
  (13 ^ 3007 / 10) % 10 = 1 :=
sorry

end tens_digit_of_13_pow_3007_l102_102498


namespace last_ball_probability_l102_102186

variables (p q : ℕ)

def probability_white_last_ball (p : ℕ) : ℝ :=
  if p % 2 = 0 then 0 else 1

theorem last_ball_probability :
  ∀ {p q : ℕ},
    probability_white_last_ball p = if p % 2 = 0 then 0 else 1 :=
by
  intros
  sorry

end last_ball_probability_l102_102186


namespace Gamma_trajectory_midpoint_on_fixed_line_l102_102379

section Problem1

-- Define the points A, B, and the line x = 6 where point C is located
def A := (-2 : ℝ, 0 : ℝ)
def B := ( 6 : ℝ, 0 : ℝ)
def Lx : ℝ → ℝ × ℝ := λ n, (6 : ℝ, n)

-- Define the midpoint of AB
def D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the equation of the ellipse
def ellipse_eq (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

theorem Gamma_trajectory : 
  ∃ P : ℝ × ℝ, ∀ (C : ℝ × ℝ), C ∈ (Lx '' set.univ) → ellipse_eq P.1 P.2 :=
sorry

end Problem1

section Problem2

-- Define Q and the fixed line
def Q := (2 : ℝ, real.sqrt 3)
def fixed_line (x y : ℝ) := (real.sqrt 3 * x + 2 * y - 2 * real.sqrt 3 = 0)

-- Define the conditions for points E and F based on intersecting with Γ and line l
def on_ellipse (E : ℝ × ℝ) := ellipse_eq E.1 E.2
def line_l_intersects (x1 y1 x2 y2: ℝ) (t : ℝ) :=
  E1 ∈ (set_of (λ t, y1 = t * (Q.2 - y1) ∧ x1 = Q.1 + (y1 - Q.2))),
  E2 ∈ (set_of (λ t, y2 = t * (Q.2 - y2) ∧ x2 = Q.1 + (y2 - Q.2)))

theorem midpoint_on_fixed_line :
  ∀ (x0 y0 : ℝ) (x1 y1 x2 y2 : ℝ), 
    on_ellipse (x1, y1) →
    on_ellipse (x2, y2) → 
    line_l_intersects x1 y1 x2 y2 → 
    let M := ((x0 + x2) / 2, (y0 + y2) / 2) in 
    fixed_line M.1 M.2 :=
sorry

end Problem2

end Gamma_trajectory_midpoint_on_fixed_line_l102_102379


namespace range_of_x_l102_102649

theorem range_of_x (x : ℝ) : (|x + 1| + |x - 1| = 2) → (-1 ≤ x ∧ x ≤ 1) :=
by
  intro h
  sorry

end range_of_x_l102_102649


namespace monotonic_intervals_and_maximum_value_l102_102696

noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (a * x^2 + x + 1)

theorem monotonic_intervals_and_maximum_value (a : ℝ) (h : 0 < a) : 
  (a = 1/2 → (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ∧ ¬∃ x, ∀ y, f a y ≤ f a x) ∧ 
  (0 < a ∧ a < 1/2 → 
    (∀ x₁ x₂ : ℝ, x₁ < -2 ∨ x₁ > -1/a → x₂ < -2 ∨ x₂ > -1/a → x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ∧ 
    (∀ x₃ x₄ : ℝ, -1/a < x₃ ∧ x₃ < -2 → -1/a < x₄ ∧ x₄ < -2 → x₃ ≤ x₄ → f a x₄ ≤ f a x₃) ∧ 
    ∃ x = -1/a, f a x = Real.exp (-1/a)) ∧ 
  (a > 1/2 → 
    (∀ x₁ x₂ : ℝ, x₁ < -2 ∨ x₁ > -1/a → x₂ < -2 ∨ x₂ > -1/a → x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ∧ 
    (∀ x₃ x₄ : ℝ, -2 < x₃ ∧ x₃ < -1/a → -2 < x₄ ∧ x₄ < -1/a → x₃ ≤ x₄ → f a x₄ ≤ f a x₃) ∧ 
    ∃ x = -2, f a x = Real.exp (-2) * (4*a - 1))
 := sorry

end monotonic_intervals_and_maximum_value_l102_102696


namespace number_of_distinct_arrangements_l102_102422

theorem number_of_distinct_arrangements : 
  let grid := (6, 4)
  let domino := (2, 1)
  let path_length := 8 
  let right_moves := 5
  let down_moves := 3
  ∃ (arrangements : ℕ), arrangements = nat.choose path_length down_moves ∧ arrangements = 56 :=
begin
  let grid := (6, 4),
  let domino := (2, 1),
  let path_length := 8,
  let right_moves := 5,
  let down_moves := 3,
  use nat.choose path_length down_moves,
  split,
  { refl },
  { sorry }
end

end number_of_distinct_arrangements_l102_102422


namespace find_range_of_a_l102_102677

variable {f : ℝ → ℝ}
noncomputable def domain_f : Set ℝ := {x | 7 ≤ x ∧ x < 15}
noncomputable def domain_f_2x_plus_1 : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}
noncomputable def A_or_B_eq_r (a : ℝ) : Prop := domain_f_2x_plus_1 ∪ B a = Set.univ

theorem find_range_of_a (a : ℝ) : 
  A_or_B_eq_r a → 3 ≤ a ∧ a < 6 := 
sorry

end find_range_of_a_l102_102677


namespace min_max_values_of_f_l102_102853

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x >= - (3 * Real.pi / 2)) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = - (3 * Real.pi / 2)) ∧
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x <= Real.pi / 2 + 2) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = Real.pi / 2 + 2) :=
by {
  -- Proof omitted
  sorry
}

end min_max_values_of_f_l102_102853


namespace binom_sum_mod_prime_l102_102118

theorem binom_sum_mod_prime (S : ℕ) : 
  S = ∑ k in finset.range (63), nat.choose 2014 k →
  nat.prime 2017 →
  S % 2017 = 1024 :=
begin
  sorry
end

end binom_sum_mod_prime_l102_102118


namespace equal_parabolic_segments_l102_102431

theorem equal_parabolic_segments (a m₁ m₂ : ℝ) :
  let M₁ := (m₁, m₁^2)
  let N₁ := (m₁ + a, (m₁ + a)^2)
  let M₂ := (m₂, m₂^2)
  let N₂ := (m₂ + a, (m₂ + a)^2)
  let area_parabolic_segment := λ m, (a^3) / 6 in
  area_parabolic_segment m₁ = area_parabolic_segment m₂ :=
by 
  sorry

end equal_parabolic_segments_l102_102431


namespace max_watertown_marching_band_l102_102133

theorem max_watertown_marching_band (n : ℤ) (h1 : 25 * n % 29 = 6) (h2 : 25 * n < 1200) : 
  25 * n ≤ 1050 :=
by {
  have h3 : n = 42,
  sorry,
  rw h3,
  exact le_refl (25 * 42),
}

end max_watertown_marching_band_l102_102133


namespace problem_statement_l102_102406

theorem problem_statement (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) →
  (∀ x : ℝ, x > 1 → f x < 0) →

  -- Conclusion 1: f(1) = 0, f(-1) = 0
  f 1 = 0 ∧ f (-1) = 0 ∧

  -- Conclusion 2: f(x) is an odd function: f(-x) = -f(x)
  (∀ x : ℝ, f (-x) = -f x) ∧

  -- Conclusion 3: f(x) is decreasing on (1, +∞)
  (∀ x1 x2 : ℝ, x1 > 1 → x2 > 1 → x1 < x2 → f x1 < f x2) := sorry

end problem_statement_l102_102406


namespace find_f_2015_l102_102407

noncomputable def f : ℝ → ℝ :=
sorry

lemma problem_conditions : (∀ x : ℝ, f (x + 1) = 1 / 2 + sqrt (f x - (f x)^2)) ∧ (f (-1) = 1 / 2) :=
sorry

theorem find_f_2015 (h : ∀ x : ℝ, f (x + 1) = 1 / 2 + sqrt (f x - (f x)^2)) (h0 : f (-1) = 1 / 2) :
  f 2015 = 1 / 2 :=
sorry

end find_f_2015_l102_102407


namespace B_starts_cycling_after_A_l102_102946

theorem B_starts_cycling_after_A (t : ℝ) : 10 * t + 20 * (2 - t) = 60 → t = 2 :=
by
  intro h
  sorry

end B_starts_cycling_after_A_l102_102946


namespace unique_solution_positivity_condition_l102_102038

theorem unique_solution_positivity_condition
  (n k : ℕ) (a : Fin n → ℝ)
  (h1 : n > k)
  (h2 : k > 1)
  (h3 : ∀ i, 0 < a i)
  (h4 : ∑ i, a i = n)
  (h5 : ∑ i, real.sqrt (k * a i ^ k / ((k - 1) * a i ^ k + 1)) = n) :
  ∀ i, a i = 1 := 
sorry

end unique_solution_positivity_condition_l102_102038


namespace vertex_of_parabola_l102_102869

def f (x : ℝ) : ℝ := 2 - (2*x + 1)^2

theorem vertex_of_parabola :
  (∀ x : ℝ, f x ≤ 2) ∧ (f (-1/2) = 2) :=
by
  sorry

end vertex_of_parabola_l102_102869


namespace sum_f_1_to_1023_eq_45_over_2_l102_102635

noncomputable def f (n : ℕ) : ℝ := if rational (𝕡 (log 4 n)) then log 4 n else 0

theorem sum_f_1_to_1023_eq_45_over_2 : (∑ n in finset.range 1023, f n.succ) = 45 / 2 :=
sorry

end sum_f_1_to_1023_eq_45_over_2_l102_102635


namespace problem1_solution_problem2_solution_l102_102523

-- Problem 1 Definitions and Theorem
noncomputable def de1 := λ (y : ℝ → ℝ) x, y x = -3 * exp(-2 * x) * (cos x + 2 * sin x)

theorem problem1_solution (y : ℝ → ℝ) :
  (∀ x, eqn1 y x = 0) ∧ y 0 = -3 ∧ (derivative y) 0 = 0 → y = de1 := by
  sorry

-- Problem 2 Definitions and Theorem
noncomputable def de2 := λ (y : ℝ → ℝ) x, y x = exp(-x) * (2 * x^2 + x - 1)

theorem problem2_solution (y : ℝ → ℝ) :
  (∀ x, eqn2 y x = 0) ∧ y 0 = -1 ∧ (derivative y) 0 = 2 ∧ (derivative₂ y) 0 = 3 → y = de2 := by
  sorry

-- Definitions for the differential equations
def eqn1 (y : ℝ → ℝ) (x : ℝ) := 
  (derivative₂ y) x + 4 * (derivative y) x + 5 * y x

def eqn2 (y : ℝ → ℝ) (x : ℝ) :=
  (derivative₃ y) x + 3 * (derivative₂ y) x + 3 * (derivative y) x + y x

end problem1_solution_problem2_solution_l102_102523


namespace focus_of_parabola_l102_102101

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end focus_of_parabola_l102_102101


namespace num_divisors_of_16n2_l102_102044

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

def divisor_count (n : ℕ) : ℕ := 
  ∏ d in (finset.range (n + 1)).filter (λ d, d ∣ n), 1 + (nat.factors n).count d

theorem num_divisors_of_16n2 {n p q : ℕ} (hp : is_prime p) (hq : is_prime q) (hpnq : p ≠ q)
  (hn : n = p^2 * q^4 ∨ n = p^4 * q^2) (hdv : divisor_count n = 15) :
  divisor_count (16 * n^2) = 225 :=
by sorry

end num_divisors_of_16n2_l102_102044


namespace jeremy_sticker_distribution_l102_102028

def number_of_ways_to_distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  (Nat.choose (total_stickers - 1) (sheets - 1))

theorem jeremy_sticker_distribution : number_of_ways_to_distribute_stickers 10 3 = 36 :=
by
  sorry

end jeremy_sticker_distribution_l102_102028


namespace cd_eq_2_of_l102_102458

variables (A B C D O F : Point)
variables [circular_order A B C D]
variables [circle_center O]
variables [perpendicular AC BD]
variables [foot OF AB]

theorem cd_eq_2_of (hA : circle_point A O)
  (hB : circle_point B O)
  (hC : circle_point C O)
  (hD : circle_point D O)
  (h_perpendicular : perpendicular AC BD)
  (h_foot : foot OF AB) :
  distance C D = 2 * distance O F :=
begin
  sorry
end

end cd_eq_2_of_l102_102458


namespace find_total_children_l102_102423

-- Define conditions as a Lean structure
structure SchoolDistribution where
  B : ℕ     -- Total number of bananas
  C : ℕ     -- Total number of children
  absent : ℕ := 160      -- Number of absent children (constant)
  bananas_per_child : ℕ := 2 -- Bananas per child originally (constant)
  bananas_extra : ℕ := 2      -- Extra bananas given to present children (constant)

-- Define the theorem we want to prove
theorem find_total_children (dist : SchoolDistribution) 
  (h1 : dist.B = 2 * dist.C) 
  (h2 : dist.B = 4 * (dist.C - dist.absent)) :
  dist.C = 320 := by
  sorry

end find_total_children_l102_102423


namespace unique_non_zero_in_rows_and_cols_l102_102395

variable (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)

theorem unique_non_zero_in_rows_and_cols
  (non_neg_A : ∀ i j, 0 ≤ A i j)
  (non_sing_A : Invertible A)
  (non_neg_A_inv : ∀ i j, 0 ≤ (A⁻¹) i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end unique_non_zero_in_rows_and_cols_l102_102395


namespace angle_PTQ_is_90_degrees_l102_102913

-- Definitions according to conditions
variables {Γ1 Γ2 : Circle} -- Γ1 and Γ2 are circles
variables {T P Q : Point} -- T is the tangency point, P and Q are tangency points on Γ1 and Γ2

-- Given conditions
variable [h1 : ExternallyTangent Γ1 Γ2 T] -- Γ1 and Γ2 are externally tangent at T
variable [h2 : CommonTangentWithoutPassingThroughT Γ1 Γ2 P Q T] -- A common tangent touches Γ1 at P and Γ2 at Q, not passing through T

-- The proof statement
theorem angle_PTQ_is_90_degrees : ∠PTQ = 90 :=
by
  sorry

end angle_PTQ_is_90_degrees_l102_102913


namespace twelve_women_reseated_l102_102140

def T : ℕ → ℕ 
| 0 := 1 -- base case, define for convenience though not explicitly given
| 1 := 1
| 2 := 2
| n := (if n ≥ 3 then (T (n - 1) + T (n - 2) + if n ≥ 4 then T (n - 3) else 0) else 0)

theorem twelve_women_reseated : T 12 = 233 :=
by
  -- Recurrence calculations will be checked by Lean automation
  sorry

end twelve_women_reseated_l102_102140


namespace max_sin_A_sin_B_plus_sin2_A_l102_102948

-- Define the conditions of the problem
def right_triangle (A B C : ℝ) : Prop :=
  C = π / 2

def sin_A_sin_B_plus_sin2_A (A B : ℝ) : ℝ :=
  sin A + sin B + sin A ^ 2

-- State the theorem
theorem max_sin_A_sin_B_plus_sin2_A (A B C : ℝ) (h : right_triangle A B C)
  (h1 : B = π / 2 - A) : 
  sin_A_sin_B_plus_sin2_A A B = sqrt 2 + 1 / 2 :=
sorry

end max_sin_A_sin_B_plus_sin2_A_l102_102948


namespace certain_number_l102_102182

theorem certain_number (x : ℝ) : 
  0.55 * x = (4/5 : ℝ) * 25 + 2 → 
  x = 40 :=
by
  sorry

end certain_number_l102_102182


namespace integer_solution_exists_l102_102607

theorem integer_solution_exists :
  ∃ (a b : ℤ), (a^2 * b^2 + a^2 + b^2 + 1 = 2005) := by
  use (2 : ℤ)
  use (20 : ℤ)
  simp
  norm_num
  sorry

end integer_solution_exists_l102_102607


namespace space_divided_into_7_parts_l102_102480

-- Definitions based on the given conditions
def plane (α : Type) := α → α → Prop
def intersection_line (α : Type) := α → α → Prop

variable (α : Type)

-- Conditions
variable (P1 P2 P3 : plane α)
variable (L1 L2 L3 : intersection_line α)

-- Assuming pairwise intersections
axiom planes_intersect_pairwise : (∀ a b : α, P1 a b = L1 a b) ∧
                                  (∀ a b : α, P2 a b = L2 a b) ∧
                                  (∀ a b : α, P3 a b = L3 a b)

-- Assuming lines of intersection are parallel
axiom lines_are_parallel : (∀ a b : α, L1 a b → L2 a b ∧ L3 a b)

-- Proposition
theorem space_divided_into_7_parts : ∃ n : ℕ, n = 7 := 
begin
  sorry
end

end space_divided_into_7_parts_l102_102480


namespace largest_indecomposable_amount_l102_102005

theorem largest_indecomposable_amount (n : ℕ) : 
  ∀ s, ¬(∃ k : ℕ, s = k * 5 ^ (n + 1) - 2 * 3 ^ (n + 1)) → 
       ¬(∃ (m : ℕ), m < 5 ∧ ∃ (r : ℕ), s = 5 * r + m * 3) :=
by
  intro s h_decomposable
  sorry

end largest_indecomposable_amount_l102_102005


namespace interval_length_l102_102253

noncomputable def g (x : ℝ) : ℝ := log 4 (log 16 (log 4 (log (1 / 16) (log 16 x))))

theorem interval_length : ∀ x ∈ set.Ioo 1 16, g x = log 4 (log 16 (log 4 (log (1 / 16) (log 16 x)))) :=
sorry

end interval_length_l102_102253


namespace percentage_of_men_speak_french_l102_102368

theorem percentage_of_men_speak_french 
  (E : ℕ) 
  (h1 : 0.35 * E = E * 35 / 100)
  (h2 : 0.40 * E = E * 40 / 100)
  (h3 : 0.7077 * (0.65 * E) = (0.65 * E) * 70.77 / 100) :
  (∀ P : ℝ, 0.2923 * 0.65 * E + P * 0.35 * E = 0.40 * E → P = 0.6) :=
by {
  intros,
  sorry
}

end percentage_of_men_speak_french_l102_102368


namespace isosceles_triangle_construction_l102_102388
-- Definitions and preconditions
variables {Point : Type} [IncidencePlane Point]
variables (O M N P Q A B C : Point)
variables (OM ON : Line Point)
variables (ω : ℝ)
variables (Hacute : 0 < ω ∧ ω < 90)

-- Define the acute angle MON
def acute_angle (M O N : Point) (ω : ℝ) : Prop :=
  angle M O N = ω ∧ ω < 90

-- Define the conditions for the isosceles triangle construction
def isosceles_triangle (A B C : Point) (OM ON : Line Point) (P Q : Point) : Prop :=
  A ∈ ON ∧
  B ∈ OM ∧
  C ∈ OM ∧
  P ∈ (line_through A B) ∧
  Q ∈ (line_through A C) ∧
  distance A B = distance A C

-- The Lean 4 statement with the equivalent proof problem
theorem isosceles_triangle_construction :
  ∀ (O M N P Q : Point) (OM ON : Line Point) (ω : ℝ),
  acute_angle M O N ω → 
  ∃ A B C : Point, isosceles_triangle A B C OM ON P Q :=
begin
  intros,
  sorry -- Proof goes here
end

end isosceles_triangle_construction_l102_102388


namespace prove_relationship_l102_102697

noncomputable def problem_statement : Prop :=
  let f : ℝ → ℝ := sorry; -- Assume some real-valued function
  let a := 3^(0.3) * f(3^(0.3));
  let b := (Real.log 3 / Real.log π) * f(Real.log 3 / Real.log π);
  let c := (Real.log (1 / 9) / Real.log 3) * f(Real.log (1 / 9) / Real.log 3);
  -- Assume conditions
  (∀ x : ℝ, f(x-1) = f (1-x)) ∧ -- The graph is symmetric about (1,0)
  (∀ x : ℝ, x < 0 → f(x) + x * (f der x) < 0) ∧ -- The given inequality condition
  -- Prove the relationship among a, b, and c
  c > a ∧ a > b

theorem prove_relationship : problem_statement := sorry

end prove_relationship_l102_102697


namespace combinedHeightOfBuildingsIsCorrect_l102_102393

-- Define the heights to the top floor of the buildings (in feet)
def empireStateBuildingHeightFeet : Float := 1250
def willisTowerHeightFeet : Float := 1450
def oneWorldTradeCenterHeightFeet : Float := 1368

-- Define the antenna heights of the buildings (in feet)
def empireStateBuildingAntennaFeet : Float := 204
def willisTowerAntennaFeet : Float := 280
def oneWorldTradeCenterAntennaFeet : Float := 408

-- Define the conversion factor from feet to meters
def feetToMeters : Float := 0.3048

-- Calculate the total heights of the buildings in meters
def empireStateBuildingTotalHeightMeters : Float := (empireStateBuildingHeightFeet + empireStateBuildingAntennaFeet) * feetToMeters
def willisTowerTotalHeightMeters : Float := (willisTowerHeightFeet + willisTowerAntennaFeet) * feetToMeters
def oneWorldTradeCenterTotalHeightMeters : Float := (oneWorldTradeCenterHeightFeet + oneWorldTradeCenterAntennaFeet) * feetToMeters

-- Calculate the combined total height of the three buildings in meters
def combinedTotalHeightMeters : Float :=
  empireStateBuildingTotalHeightMeters + willisTowerTotalHeightMeters + oneWorldTradeCenterTotalHeightMeters

-- The statement to prove
theorem combinedHeightOfBuildingsIsCorrect : combinedTotalHeightMeters = 1511.8164 := by
  sorry

end combinedHeightOfBuildingsIsCorrect_l102_102393


namespace digit_two_not_in_mean_l102_102242

namespace MeanDigits

def set_of_numbers := {8, 88, 888, 8888, 88888, 888888, 8888888, 88888888}

def arithmetic_mean : ℕ := 11111111

theorem digit_two_not_in_mean : ∀ (d : ℕ), (d = 2) → ¬ (d ∈ (list.of_fn (λ i, (arithmetic_mean / (10^i % 10)) % 10)).erase_dup)
:= sorry

end MeanDigits

end digit_two_not_in_mean_l102_102242


namespace compute_fraction_l102_102583

theorem compute_fraction :
  (∏ i in (1 : Finset ℕ).range 17, 1 + (15 / i)) /
  (∏ j in (1 : Finset ℕ).range 15, 1 + (17 / j)) = 272 :=
by
  sorry

end compute_fraction_l102_102583


namespace complex_sum_l102_102416

noncomputable def z := sorry -- as we are skipping the proof, we define z but leave its construction pending

theorem complex_sum (z : ℂ) (h : z^2 - z + 1 = 0) : z^107 + z^108 + z^109 + z^110 + z^111 = z := by
  sorry

end complex_sum_l102_102416


namespace part_a_part_b_l102_102397

noncomputable def a : ℕ → ℝ
| 0     := 0
| 1     := 1
| (n+2) := a (n+1) + 1 / (2 * a (n+1))

theorem part_a (n : ℕ) (hn : n ≥ 1) : 
  (n : ℝ) ≤ a (n + 1) ^ 2 ∧ a (n + 1) ^ 2 < (n : ℝ) + (n : ℝ)^(1/3) := sorry

theorem part_b : 
  filter.tendsto (fun (n : ℕ) => a (n + 1) - real.sqrt (n : ℝ)) filter.at_top (𝓝 0) := sorry

end part_a_part_b_l102_102397


namespace combination_symmetry_pascal_identity_permutation_combination_relation_l102_102632

variables (m n : ℕ)
include m_pos n_pos : m > 0, n > 0

theorem combination_symmetry : (nat.choose n m = nat.choose n (n - m)) :=
sorry

theorem pascal_identity : (nat.choose (n + 1) m = nat.choose n (m - 1) + nat.choose n m) :=
sorry

theorem permutation_combination_relation : (nat.perm n m = nat.choose n m * nat.factorial m) :=
sorry

end combination_symmetry_pascal_identity_permutation_combination_relation_l102_102632


namespace students_failed_in_english_l102_102740

variable (H : ℝ) (E : ℝ) (B : ℝ) (P : ℝ)

theorem students_failed_in_english
  (hH : H = 34 / 100) 
  (hB : B = 22 / 100)
  (hP : P = 44 / 100)
  (hIE : (1 - P) = H + E - B) :
  E = 44 / 100 := 
sorry

end students_failed_in_english_l102_102740


namespace john_pack_count_l102_102029

-- Defining the conditions
def utensilsInPack : Nat := 30
def knivesInPack : Nat := utensilsInPack / 3
def forksInPack : Nat := utensilsInPack / 3
def spoonsInPack : Nat := utensilsInPack / 3
def requiredKnivesRatio : Nat := 2
def requiredForksRatio : Nat := 3
def requiredSpoonsRatio : Nat := 5
def minimumSpoons : Nat := 50

-- Proving the solution
theorem john_pack_count : 
  ∃ packs : Nat, 
    (packs * spoonsInPack >= minimumSpoons) ∧
    (packs * foonsInPack / packs * knivesInPack = requiredForksRatio / requiredKnivesRatio) ∧
    (packs * spoonsInPack / packs * forksInPack = requiredForksRatio / requiredSpoonsRatio) ∧
    (packs * spoonsInPack / packs * knivesInPack = requiredSpoonsRatio / requiredKnivesRatio) ∧
    packs = 5 :=
sorry

end john_pack_count_l102_102029


namespace find_root_D_l102_102306

/-- Given C and D are roots of the polynomial k x^2 + 2 x + 5 = 0, 
    and k = -1/4 and C = 10, then D must be -2. -/
theorem find_root_D 
  (k : ℚ) (C D : ℚ)
  (h1 : k = -1/4)
  (h2 : C = 10)
  (h3 : C^2 * k + 2 * C + 5 = 0)
  (h4 : D^2 * k + 2 * D + 5 = 0) : 
  D = -2 :=
by
  sorry

end find_root_D_l102_102306


namespace squirrel_population_difference_l102_102134

theorem squirrel_population_difference :
  ∀ (total_population scotland_population rest_uk_population : ℕ), 
  scotland_population = 120000 →
  120000 = 75 * total_population / 100 →
  rest_uk_population = total_population - scotland_population →
  scotland_population - rest_uk_population = 80000 :=
by
  intros total_population scotland_population rest_uk_population h1 h2 h3
  sorry

end squirrel_population_difference_l102_102134


namespace regions_first_two_sets_regions_all_sets_l102_102037

-- Definitions for the problem
def triangle_regions_first_two_sets (n : ℕ) : ℕ :=
  (n + 1) * (n + 1)

def triangle_regions_all_sets (n : ℕ) : ℕ :=
  3 * n * n + 3 * n + 1

-- Proof Problem 1: Given n points on AB and AC, prove the regions are (n + 1)^2
theorem regions_first_two_sets (n : ℕ) :
  (n * (n + 1) + (n + 1)) = (n + 1) * (n + 1) :=
by sorry

-- Proof Problem 2: Given n points on AB, AC, and BC, prove the regions are 3n^2 + 3n + 1
theorem regions_all_sets (n : ℕ) :
  ((n + 1) * (n + 1) + n * (2 * n + 1)) = 3 * n * n + 3 * n + 1 :=
by sorry

end regions_first_two_sets_regions_all_sets_l102_102037


namespace total_interest_correct_l102_102059

-- Initial conditions
def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.08
def additional_deposit : ℝ := 500
def first_period : ℕ := 2
def second_period : ℕ := 2

-- Calculate the accumulated value after the first period
def first_accumulated_value : ℝ := initial_investment * (1 + annual_interest_rate)^first_period

-- Calculate the new principal after additional deposit
def new_principal := first_accumulated_value + additional_deposit

-- Calculate the accumulated value after the second period
def final_value := new_principal * (1 + annual_interest_rate)^second_period

-- Calculate the total interest earned after 4 years
def total_interest_earned := final_value - initial_investment - additional_deposit

-- Final theorem statement to be proven
theorem total_interest_correct : total_interest_earned = 515.26 :=
by sorry

end total_interest_correct_l102_102059


namespace distance_between_houses_l102_102455

theorem distance_between_houses :
  (∀ (A B V G : ℝ), 
     (abs (A - B) = 600) ∧ 
     (abs (V - G) = 600) ∧ 
     (abs (A - G) = 3 * abs (B - V)) → 
     (abs (A - G) = 900 ∨ abs (A - G) = 1800)) :=
begin
  sorry
end

end distance_between_houses_l102_102455


namespace domain_of_f_l102_102830

def f (x : ℝ) : ℝ := (Real.sqrt (x + 1)) / x

theorem domain_of_f :
  {x : ℝ | (x + 1 ≥ 0) ∧ (x ≠ 0)} = {x : ℝ | x ∈ Set.Ico (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ)} := by
  sorry

end domain_of_f_l102_102830


namespace gcd_in_range_l102_102836

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end gcd_in_range_l102_102836


namespace polygon_intersection_count_l102_102079

theorem polygon_intersection_count :
  let m := [4, 5, 6, 7] in
  ∀ n ∈ m, ∀ k ∈ m, n ≠ k →
  ¬ ∃ v v', v ∈ n.vertices ∧ v' ∈ k.vertices ∧ v = v' →
  (∀ p, exactly_two_sides_intersect p m → p.count = 56)
  := sorry

end polygon_intersection_count_l102_102079


namespace unique_integral_root_of_equation_l102_102450

theorem unique_integral_root_of_equation :
  ∀ x : ℤ, (x - 9 / (x - 5) = 7 - 9 / (x - 5)) ↔ (x = 7) :=
by
  sorry

end unique_integral_root_of_equation_l102_102450


namespace probability_a_b_equal_3a_probability_a_squared_plus_b_minus_five_squared_leq_nine_l102_102554

theorem probability_a_b_equal_3a :
  let Ω := ({(a, b) | a ∈ Finset.range 6 ∧ b ∈ Finset.range 4 ∧ b + 6 = a + a + a}) in
  (Finset.filter (λ (p: ℕ × ℕ), p.snd = 3 * p.fst) (Finset.product (Finset.range 6) (Finset.range 10))).card.to_real / Ω.card.to_real = 1 / 12 := 
  by sorry

theorem probability_a_squared_plus_b_minus_five_squared_leq_nine :
  let Ω := ({(a, b) | a ∈ Finset.range 6 ∧ b ∈ Finset.range 10}) in
  let E := (Finset.filter (λ (p: ℕ × ℕ), p.fst^2 + (p.snd - 5)^2 ≤ 9) Ω) in
  E.card.to_real / Ω.card.to_real = 7 / 24 := 
  by sorry

end probability_a_b_equal_3a_probability_a_squared_plus_b_minus_five_squared_leq_nine_l102_102554


namespace find_real_number_l102_102152

noncomputable def recursive_sequence : ℝ :=
  let rec_seq : ℝ := 3 + 4 / (1 + 4 / (3 + 4 / (1/2 + rec_seq)))
  rec_seq

theorem find_real_number :
  recursive_sequence = (43 + Real.sqrt 4049) / 22 :=
by
  sorry

end find_real_number_l102_102152


namespace students_overlap_difference_l102_102729

theorem students_overlap_difference : 
  ∀ (total students geometry biology : ℕ),
  total = 232 → geometry = 144 → biology = 119 →
  (geometry + biology - total = 31) ∧ (min geometry biology = 119) →
  (min geometry biology - (geometry + biology - total) = 88) :=
by
  intros total geometry biology htotal hgeometry hbiology hconds,
  exact sorry

end students_overlap_difference_l102_102729


namespace expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102629

open ProbabilityTheory

/-- The expected total number of dice rolls until all five dice show sixes is 30 --/
theorem expected_total_number_of_dice_rolls (n : ℕ) (H : n = 5) : 
  (∑ k in range(n : ℕ), (E[(indicator(event_of_five_dice_roll)) k])) = 30 :=
sorry

/-- The expected total number of points rolled until all five dice show sixes is 105 --/
theorem expected_total_number_of_points (n : ℕ) (H : n = 5) : 
  (E[∑ i in range(n), points_rolled_until_six]) = 105 :=
sorry

/-- The expected number of salvos until all five dice show sixes is approximately 13.02 --/
theorem expected_number_of_salvos (n : ℕ) (H : n = 5) : 
  (E[number_of_salvos_to_get_all_sixes]) = 13.02 :=
sorry

end expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102629


namespace complement_U_P_l102_102645

def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

theorem complement_U_P :
  (U \ P) = Set.Ici (1 / 2) := 
by
  sorry

end complement_U_P_l102_102645


namespace tree_arrangement_exists_l102_102370
-- Since we need a broad import to cover necessary libraries

-- Definitions for the problem
def Tree := Type
def AppleTree : Tree := sorry
def PearTree : Tree := sorry

def Garden := {t : Tree // t = AppleTree ∨ t = PearTree}

-- Conditions given in the problem
def num_trees (g: list Garden) : Prop := g.length = 7
def has_both_types (g: list Garden) : Prop :=
  ∃ a p, a ∈ g ∧ p ∈ g ∧ a.val = AppleTree ∧ p.val = PearTree
def closest_same_kind (g: list Garden) : Prop := sorry -- needs definition based on tree proximity
def farthest_same_kind (g: list Garden) : Prop := sorry -- needs definition based on tree proximity

-- The theorem that needs to be proven
theorem tree_arrangement_exists : 
  ∃ g : list Garden, num_trees g ∧ has_both_types g ∧ closest_same_kind g ∧ farthest_same_kind g :=
sorry

end tree_arrangement_exists_l102_102370


namespace expression_undefined_at_12_l102_102593

theorem expression_undefined_at_12 : ∀ x : ℝ, (x^2 - 24*x + 144) = 0 → x = 12 :=
by
  intro x h
  have : (x - 12)^2 = 0,
  { rw [← h] }
  exact eq_of_sq_eq_sq x 12
  sorry

end expression_undefined_at_12_l102_102593


namespace circle_area_l102_102499

-- Define the diameter and radius
def diameter : Real := 8
def radius : Real := diameter / 2

-- The main theorem to prove the area of the circle
theorem circle_area : 
    (π * radius^2 = 16 * π) := 
by
  -- Placeholder for the proof
  sorry

end circle_area_l102_102499


namespace eq_perp_bisector_BC_area_triangle_ABC_l102_102316

section Triangle_ABC

open Real

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the equation of the perpendicular bisector
theorem eq_perp_bisector_BC : ∀ x y : ℝ, 2 * x + y - 4 = 0 :=
sorry

-- Define the area of the triangle ABC
noncomputable def triangle_area : ℝ :=
1 / 2 * (abs ((-1 * 3 + 3 * (-2) + 3 * 4) - (3 * 4 + 1 * (-2) + 3*(-1))))

theorem area_triangle_ABC : triangle_area = 7 :=
sorry

end Triangle_ABC

end eq_perp_bisector_BC_area_triangle_ABC_l102_102316


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102625

-- Conditions
constant NumDice : ℕ := 5

-- Define the probability of a die not showing six on each roll
def prob_not_six := (5 : ℚ) / 6

-- Question a: Expected total number of dice rolled
theorem expected_total_dice_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 30
  sorry

-- Question b: Expected total number of points rolled
theorem expected_total_points_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 105
  sorry

-- Question c: Expected number of salvos
theorem expected_number_of_salvos :
  True := by
  -- The proof would calculate the expected number of salvos as derived, resulting in 13.02
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102625


namespace time_before_Car_Y_started_in_minutes_l102_102235

noncomputable def timeBeforeCarYStarted (speedX speedY distanceX : ℝ) : ℝ :=
  let t := distanceX / speedX
  (speedY * t - distanceX) / speedX

theorem time_before_Car_Y_started_in_minutes 
  (speedX speedY distanceX : ℝ)
  (h_speedX : speedX = 35)
  (h_speedY : speedY = 70)
  (h_distanceX : distanceX = 42) : 
  (timeBeforeCarYStarted speedX speedY distanceX) * 60 = 72 :=
by
  sorry

end time_before_Car_Y_started_in_minutes_l102_102235


namespace second_largest_div_second_smallest_l102_102154

theorem second_largest_div_second_smallest : 
  let a := 10
  let b := 11
  let c := 12
  ∃ second_smallest second_largest, 
    second_smallest = b ∧ second_largest = b ∧ second_largest / second_smallest = 1 := 
by
  let a := 10
  let b := 11
  let c := 12
  use b
  use b
  exact ⟨rfl, rfl, rfl⟩

end second_largest_div_second_smallest_l102_102154


namespace determine_radius_of_sphere_l102_102144

noncomputable def radius_of_sphere (A M N P : ℝ) (M'N'P' : ℝ) 
  (distance_1 distance_2 distance_3 circumference : ℝ) (perpendicular distance radius diameter : ℝ) : Prop :=
radius = diameter / 2

theorem determine_radius_of_sphere (A M N P : ℝ) (M'N'P' : ℝ)
  (distance_1 distance_2 distance_3 circumference : ℝ) (perpendicular distance radius diameter : ℝ)
  (hyp_1 : type 1 condition here)
  (hyp_2 : type 2 condition here)
  (hyp_3 : type 3 condition here)
  (hyp_4 : type 4 condition here)
  (hyp_5 : type 5 condition here)
  (hyp_6 : type 6 condition here)
  (hyp_7 : type 7 condition here)
  (hyp_8 : type 8 condition here)
  : radius_of_sphere A M N P M'N'P' distance_1 distance_2 distance_3 circumference perpendicular distance radius diameter :=
sorry

end determine_radius_of_sphere_l102_102144


namespace remainder_div_1234567_256_l102_102967

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l102_102967


namespace min_direction_changes_l102_102225

theorem min_direction_changes (n : ℕ) : 
  ∀ (path : Finset (ℕ × ℕ)), 
    (path.card = (n + 1) * (n + 2) / 2) → 
    (∀ (v : ℕ × ℕ), v ∈ path) →
    ∃ changes, (changes ≥ n) :=
by sorry

end min_direction_changes_l102_102225


namespace distance_product_l102_102302

-- Define the condition that P lies on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (y^2) / 4 - (x^2) = 1

-- Define the distance product property to be proved
theorem distance_product (P : ℝ × ℝ) (h : on_hyperbola P) :
  let (x₀, y₀) := P in
  |(2 * x₀ - y₀) / (real.sqrt 5)| * |(2 * x₀ + y₀) / (real.sqrt 5)| = 4 / 5 :=
by
  sorry

end distance_product_l102_102302


namespace fraction_numerator_greater_than_denominator_l102_102862

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5 / 3 → (8 / 11 < x ∧ x < 5 / 3) ∨ (5 / 3 < x ∧ x ≤ 3) ↔ (8 * x - 3 > 5 - 3 * x) := by
  sorry

end fraction_numerator_greater_than_denominator_l102_102862


namespace sum_of_first_n_terms_l102_102657

theorem sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + 2 * a 2 = 3)
  (h2 : ∀ n, a (n + 1) = a n + 2) :
  ∀ n, S n = n * (n - 4 / 3) := 
sorry

end sum_of_first_n_terms_l102_102657


namespace triangle_area_l102_102941

theorem triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ ∃ (A : ℝ), 
  A = Real.sqrt (6 * (6 - a) * (6 - b) * (6 - c)) ∧ A = 6 := by
  sorry

end triangle_area_l102_102941


namespace A_eq_B_l102_102036

noncomputable def A := Real.sqrt 5 + Real.sqrt (22 + 2 * Real.sqrt 5)
noncomputable def B := Real.sqrt (11 + 2 * Real.sqrt 29) 
                      + Real.sqrt (16 - 2 * Real.sqrt 29 
                                   + 2 * Real.sqrt (55 - 10 * Real.sqrt 29))

theorem A_eq_B : A = B := 
  sorry

end A_eq_B_l102_102036


namespace students_water_count_l102_102738

-- Define the given conditions
def pct_students_juice (total_students : ℕ) : ℕ := 70 * total_students / 100
def pct_students_water (total_students : ℕ) : ℕ := 30 * total_students / 100
def students_juice (total_students : ℕ) : Prop := pct_students_juice total_students = 140

-- Define the proposition that needs to be proven
theorem students_water_count (total_students : ℕ) (h1 : students_juice total_students) : 
  pct_students_water total_students = 60 := 
by
  sorry


end students_water_count_l102_102738


namespace parts_stock_cannot_be_exactly_used_up_l102_102168

theorem parts_stock_cannot_be_exactly_used_up
  (p q r : ℕ)
  (units_A units_B units_C : ℕ)
  (alpha_units_A alpha_units_B beta_units_B beta_units_C gamma_units_A gamma_units_C : ℕ)
  (remaining_A remaining_B remaining_C : ℕ) :
  alpha_units_A = 2 → 
  alpha_units_B = 2 →
  beta_units_B = 1 → 
  beta_units_C = 1 → 
  gamma_units_A = 2 → 
  gamma_units_C = 1 →
  remaining_A = 2 →
  remaining_B = 1 →
  remaining_C = 0 →
  (units_A - (alpha_units_A * p + gamma_units_A * r) = remaining_A) →
  (units_B - (alpha_units_B * p + beta_units_B * q) = remaining_B) →
  (units_C - (beta_units_C * q + gamma_units_C * r) = remaining_C) →
  units_A - (alpha_units_A * p + gamma_units_A * r) ≠ 0 ∨ 
  units_B - (alpha_units_B * p + beta_units_B * q) ≠ 0 ∨ 
  units_C - (beta_units_C * q + gamma_units_C * r) ≠ 0 :=
begin
  -- The proof will go here
  sorry
end

end parts_stock_cannot_be_exactly_used_up_l102_102168


namespace maria_visits_exhibits_l102_102061

theorem maria_visits_exhibits :
  ∃ (n : ℕ), n = 5! ∧ n = 120 :=
by {
  have h : 5! = 120,
  { exact dec_trivial, },
  use 120,
  exact ⟨h, rfl⟩,
}

end maria_visits_exhibits_l102_102061


namespace volume_of_snow_l102_102539

theorem volume_of_snow (L W H : ℝ) (hL : L = 30) (hW : W = 3) (hH : H = 0.75) :
  L * W * H = 67.5 := by
  sorry

end volume_of_snow_l102_102539


namespace exists_moment_distance_ge_L_l102_102481

noncomputable def track_length : ℝ := 3 * L

structure Runner :=
  (name : String)
  (speed : ℝ)

def runners : List Runner := [
  {name := "A", speed := vA},
  {name := "B", speed := vB},
  {name := "C", speed := vC}
]

def distance_around_track (pos1 pos2 : ℝ) : ℝ :=
  let dist := abs (pos1 - pos2) % track_length
  min dist (track_length - dist)

theorem exists_moment_distance_ge_L (L vA vB vC : ℝ) (h_diff_speeds : vA ≠ vB ∧ vA ≠ vC ∧ vB ≠ vC) :
  ∃ t : ℝ, ∀ (i j k : Fin 3), 
    let positions := [vA * t % track_length, vB * t % track_length, vC * t % track_length] in 
    distance_around_track (positions.nth i) (positions.nth j) ≥ L ∧ 
    distance_around_track (positions.nth i) (positions.nth k) ≥ L := 
sorry

end exists_moment_distance_ge_L_l102_102481


namespace steps_to_return_l102_102346

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0
def forward_steps : ℕ := 2
def backward_steps : ℕ := 3

theorem steps_to_return : 
  let primes := {n | 2 ≤ n ∧ n ≤ 30 ∧ is_prime n} in
  let composites := {n | 2 ≤ n ∧ n ≤ 30 ∧ ¬is_prime n} in
  let forward_total := forward_steps * primes.to_finset.card in
  let backward_total := backward_steps * composites.to_finset.card in
  forward_total - backward_total = -37 →
  - (forward_total - backward_total) = 37 :=
by
  sorry

end steps_to_return_l102_102346


namespace domain_of_v_l102_102250

noncomputable def v (x : ℝ) : ℝ := 1 / (x ^ (1/3) + x^2 - 1)

theorem domain_of_v : ∀ x, x ≠ 1 → x ^ (1/3) + x^2 - 1 ≠ 0 :=
by
  sorry

end domain_of_v_l102_102250


namespace rectangle_area_from_conditions_l102_102909

variable (r : ℝ) (l : ℝ) (b : ℝ) (s : ℝ)

-- Define the relationships and conditions
def square_side_given_area (A : ℝ) : ℝ := real.sqrt A
def rectangle_length_from_radius (r : ℝ) : ℝ := (2 / 5) * r
def rectangle_area (l : ℝ) (b : ℝ) : ℝ := l * b

theorem rectangle_area_from_conditions
  (s : ℝ) (hs : s = square_side_given_area 2500)
  (r : ℝ) (hr : r = s)
  (l : ℝ) (hl : l= rectangle_length_from_radius r)
  (b : ℝ) (hb : b = 10) :
  rectangle_area l b = 200 := by
  sorry

end rectangle_area_from_conditions_l102_102909


namespace equivalent_proof_problem_l102_102724

-- Define the conditions as Lean 4 definitions
variable (x₁ x₂ : ℝ)

-- The conditions given in the problem
def condition1 : Prop := x₁ * Real.logb 2 x₁ = 1008
def condition2 : Prop := x₂ * 2^x₂ = 1008

-- The problem to be proved
theorem equivalent_proof_problem (hx₁ : condition1 x₁) (hx₂ : condition2 x₂) : 
  x₁ * x₂ = 1008 := 
sorry

end equivalent_proof_problem_l102_102724


namespace smallest_number_of_rectangles_needed_l102_102503

theorem smallest_number_of_rectangles_needed :
  ∃ n, (n * 12 = 144) ∧ (∀ k, (k * 12 = 144) → k ≥ n) := by
  sorry

end smallest_number_of_rectangles_needed_l102_102503


namespace sum_of_coefficients_l102_102615

-- Definition of the polynomial
def polynomial := 3 * (x^8 - 2*x^5 + x^3 - 6) - 5 * (2*x^4 + 3*x^2) + 2 * (x^6 - 5)

-- Theorem statement
theorem sum_of_coefficients : (polynomial 1) = -51 := by
  sorry

end sum_of_coefficients_l102_102615


namespace compute_a_plus_b_l102_102778

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end compute_a_plus_b_l102_102778


namespace shortest_altitude_of_right_triangle_l102_102866

theorem shortest_altitude_of_right_triangle :
  ∀ (a b c : ℕ) (ha : a = 8) (hb : b = 15) (hc : c = 17) (h_right : a^2 + b^2 = c^2), ∃ h : ℚ, 
  2 * 60 = c * h ∧ h = 120 / 17 := 
by
  intros a b c ha hb hc h_right
  use 120 / 17
  split
  { calc 
    2 * 60 = ↑c * (120 / 17) : by norm_num[hc, ha, hb]
  }
  { norm_num[sorry]
  }

end shortest_altitude_of_right_triangle_l102_102866


namespace inradius_inequality_l102_102712

theorem inradius_inequality (a b c r : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) (h4 : r = √((s * (s - a) * (s - b) * (s - c))) / s) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≤ 1 / (4 * r^2) :=
sorry

end inradius_inequality_l102_102712


namespace avgPairsTriplets_l102_102497

open Finset

noncomputable def numPairsTriplets (s : Finset ℕ) : ℕ :=
  let pairs := s.subsets.card.filter (λ t, t.card = 2 ∧ (t.max' (by simp [t.nonempty]) - t.min' (by simp [t.nonempty]) = 1))
  let triplets := s.subsets.card.filter (λ t, t.card = 3 ∧ (t.max' (by simp [t.nonempty]) - t.min' (by simp [t.nonempty]) = 2))
  pairs + triplets

theorem avgPairsTriplets : 
  let sets := (powerset (range 1 21)).filter (λ s, s.card = 4) in
  (∑ s in sets, numPairsTriplets s : ℝ) / sets.card = 0.2 :=
  sorry

end avgPairsTriplets_l102_102497


namespace solve_expression_l102_102494

theorem solve_expression : 6 / 3 - 2 - 8 + 2 * 8 = 8 := 
by 
  sorry

end solve_expression_l102_102494


namespace function_evaluation_l102_102789

def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else 2 * Real.sqrt 2 * Real.cos x

theorem function_evaluation : f (f (-π/4)) = 4 :=
by
  sorry

end function_evaluation_l102_102789


namespace find_prices_l102_102196

variables (C S : ℕ) -- Using natural numbers to represent rubles

theorem find_prices (h1 : C + S = 2500) (h2 : 4 * C + 3 * S = 8870) :
  C = 1370 ∧ S = 1130 :=
by
  sorry

end find_prices_l102_102196


namespace minimum_matches_l102_102215

theorem minimum_matches (n : ℕ) (h1 : n > 3) 
  (matches : set (ℕ × ℕ)) 
  (h2 : ∀ {x y : ℕ}, (x, y) ∈ matches → (y, x) ∉ matches)
  (h3 : ∀ {x y : ℕ}, x ≠ y → ∃ z : ℕ, (z, x) ∈ matches ∧ (z, y) ∈ matches ∨ (x, z) ∈ matches ∧ (y, z) ∈ matches)
  (h4 : ∃ t : ℕ, ∀ {x : ℕ}, (t, x) ∈ matches → x = t) :
  ∃ (k : ℕ), k = 2 * n - 2 ∧ finite_but_non_empty (matches) :=
by
  sorry

end minimum_matches_l102_102215


namespace amara_remaining_clothes_l102_102562

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end amara_remaining_clothes_l102_102562


namespace perpendicular_condition_l102_102653

def line := Type
def plane := Type

variables {α : plane} {a b : line}

-- Conditions: define parallelism and perpendicularity
def parallel (a : line) (α : plane) : Prop := sorry
def perpendicular (a : line) (α : plane) : Prop := sorry
def perpendicular_lines (a b : line) : Prop := sorry

-- Given Hypotheses
variable (h1 : parallel a α)
variable (h2 : perpendicular b α)

-- Statement to prove
theorem perpendicular_condition (h1 : parallel a α) (h2 : perpendicular b α) :
  (perpendicular_lines b a) ∧ (¬ (perpendicular_lines b a → perpendicular b α)) := 
sorry

end perpendicular_condition_l102_102653


namespace remainder_1234567_div_256_l102_102970

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l102_102970


namespace taehyung_math_score_l102_102445

theorem taehyung_math_score
  (avg_before : ℝ)
  (drop_in_avg : ℝ)
  (num_subjects_before : ℕ)
  (num_subjects_after : ℕ)
  (avg_after : ℝ)
  (total_before : ℝ)
  (total_after : ℝ)
  (math_score : ℝ) :
  avg_before = 95 →
  drop_in_avg = 3 →
  num_subjects_before = 3 →
  num_subjects_after = 4 →
  avg_after = avg_before - drop_in_avg →
  total_before = avg_before * num_subjects_before →
  total_after = avg_after * num_subjects_after →
  math_score = total_after - total_before →
  math_score = 83 :=
by
  intros
  sorry

end taehyung_math_score_l102_102445


namespace monotonicity_of_f_f_leq_g_implies_a_leq_zero_l102_102666

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x ^ 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 
  x * Real.exp x - (1 / 2) * a * x ^ 2 - (a + 1) * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → f x a ≥ f (x - 1) a) → 
  (a ≤ 0 → ∀ x > 0, deriv (f x a) x ≥ 0) ∧ 
  (a > 0 → ∀ x, (0 < x < (Real.sqrt a / a)) → deriv (f x a) x ≥ 0 ∧ 
                 deriv (f x a) x < 0) :=
sorry

theorem f_leq_g_implies_a_leq_zero :
  (∀ x : ℝ, 0 < x → f x a ≤ g x a) → (a ≤ 0) :=
sorry

end monotonicity_of_f_f_leq_g_implies_a_leq_zero_l102_102666


namespace f_n_solution_l102_102634

theorem f_n_solution (n : ℕ) (h_pos : 0 < n) : 
  (∀ n, f (n + 1) = f n + n) ∧ f 1 = 1 → f n = (n^2 - n + 2) / 2 :=
by sorry

end f_n_solution_l102_102634


namespace wages_problem_l102_102940

variable {S W_y W_x : ℝ}
variable {D_x : ℝ}

theorem wages_problem
  (h1 : S = 45 * W_y)
  (h2 : S = 20 * (W_x + W_y))
  (h3 : S = D_x * W_x) :
  D_x = 36 :=
sorry

end wages_problem_l102_102940


namespace distribution_count_l102_102955

theorem distribution_count (n : ℕ) : 
  ∃ f : Fin (2 * n) → Bool, 
  (∀ k, 1 ≤ k ∧ k ≤ n → (f (⟨k, by linarith⟩) ≠ f (⟨2*k, by linarith⟩))) ∧ 
  (2^n = 2^n) := 
begin
  sorry
end

end distribution_count_l102_102955


namespace area_ratio_of_inscribed_quadrilateral_l102_102805

theorem area_ratio_of_inscribed_quadrilateral (r : ℝ) :
  let AC := 2 * r
  let angle_DAC := real.pi / 6  -- 30 degrees in radians
  let angle_BAC := real.pi / 4  -- 45 degrees in radians
  let area_ABCD := (r^2 * (real.sqrt 3 / 2 + 1))
  let area_circle := real.pi * r^2
  area_ABCD / area_circle = (real.sqrt 3 + 2) / (2 * real.pi)
→ 7 :=
by
  sorry

end area_ratio_of_inscribed_quadrilateral_l102_102805


namespace parallel_lines_slope_l102_102506

-- Define the equations of the lines in Lean
def line1 (x : ℝ) : ℝ := 7 * x + 3
def line2 (c : ℝ) (x : ℝ) : ℝ := (3 * c) * x + 5

-- State the theorem: if the lines are parallel, then c = 7/3
theorem parallel_lines_slope (c : ℝ) :
  (∀ x : ℝ, (7 * x + 3 = (3 * c) * x + 5)) → c = (7/3) :=
by
  sorry

end parallel_lines_slope_l102_102506


namespace sum_of_solutions_l102_102897

theorem sum_of_solutions (x : ℝ) (h : ∀ x, sqrt ((x + 5) ^ 2) = 8 → x ∈ {3, -13}) : 3 + (-13) = -10 := by
  have h₁ : sqrt ((3 + 5) ^ 2) = 8 := by norm_num
  have h₂ : sqrt ((-13 + 5) ^ 2) = 8 := by norm_num
  exact eq_of_beq (3 + (-13)) (-10) sorry

end sum_of_solutions_l102_102897


namespace ellipse_equation_l102_102380

theorem ellipse_equation
  (h1 : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 2 * a + 2 * c = 12)
  (h2 : ∃ a c : ℝ, a = 2 * c)
  (h3 : ∃ a b c : ℝ, b^2 = a^2 - c^2) :
  ∃ (a b : ℝ), a = 4 ∧ b = real.sqrt (16 - 4) ∧ (∀ x y : ℝ, x^2 / 16 + y^2 / 12 = 1) :=
by
  sorry

end ellipse_equation_l102_102380


namespace spider_wear_order_count_l102_102216

-- Definitions based on the conditions
def spider_legs : Nat := 8
def items_per_leg : Nat := 3
def total_items : Nat := spider_legs * items_per_leg

-- The theorem statement
theorem spider_wear_order_count : 
  (∑ (perms : Finset (Fin total_items → Fin total_items)), 
    let valid_mask (f : Fin total_items → Fin total_items) := 
      (∀ i < spider_legs, 
        let base := i * items_per_leg in
        f base < f (base + 1) ∧ f base < f (base + 2) ∧ 
        (f (base + 1) < f (base + 2) ∨ f (base + 2) < f (base + 1))) in 
    if valid_mask perms then 1 else 0) 
  = ((24.factorial * (4 ^ spider_legs)) / (6 ^ spider_legs)) := sorry

end spider_wear_order_count_l102_102216


namespace vector_magnitude_l102_102708

variables (a b : ℝ × ℝ)
def is_parallel (v u : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem vector_magnitude (y : ℝ) 
  (hyp_a : a = (1, 1)) 
  (hyp_b : b = (3, y)) 
  (hyp_parallel : is_parallel (a.1 + b.1, a.2 + b.2) a) : 
  |b.1 - a.1, b.2 - a.2| = 2 * real.sqrt 2 := 
sorry

end vector_magnitude_l102_102708


namespace fill_tank_with_two_pipes_l102_102877

def Pipe (Rate : Type) := Rate

theorem fill_tank_with_two_pipes
  (capacity : ℝ)
  (three_pipes_fill_time : ℝ)
  (h1 : three_pipes_fill_time = 12)
  (pipe_rate : ℝ)
  (h2 : pipe_rate = capacity / 36) :
  2 * pipe_rate * 18 = capacity := 
by 
  sorry

end fill_tank_with_two_pipes_l102_102877


namespace min_value_of_f_l102_102281

def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 2|

theorem min_value_of_f : ∃ x : ℝ, (f x = 1) ∧ ∀ y : ℝ, f y ≥ f x :=
begin
  sorry
end

end min_value_of_f_l102_102281


namespace integer_roots_of_polynomial_l102_102276

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 4 * x^2 - 7 * x + 10 = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_of_polynomial_l102_102276


namespace equal_study_time_l102_102792

theorem equal_study_time (total_time : ℝ) (num_question_types : ℕ) (hours_per_type : ℝ) :
  total_time = 50 → num_question_types = 3 → hours_per_type = total_time / num_question_types →
  hours_per_type ≈ 16.67 :=
begin
  intros h1 h2 h3,
  sorry
end

end equal_study_time_l102_102792


namespace isosceles_triangle_perimeter_l102_102297

-- Define an isosceles triangle structure
structure IsoscelesTriangle where
  (a b c : ℝ) 
  (isosceles : a = b ∨ a = c ∨ b = c)
  (side_lengths : (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧ (c = 2 ∨ c = 3))
  (valid_triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem to prove the perimeter
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.a + t.b + t.c = 7 ∨ t.a + t.b + t.c = 8 :=
sorry

end isosceles_triangle_perimeter_l102_102297


namespace arithmetic_sequence_sum_l102_102663

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (d : ℚ)
  (h1 : a 3 + a 4 = a 12)
  (h2 : a 1 + a 2 = 10)
  (h_arith : ∀ n, a (n + 1) = a n + d) :
  ∑ k in finset.range 50, a (2 * k + 2) = 28000 / 13 :=
by
  sorry

end arithmetic_sequence_sum_l102_102663


namespace range_of_a_l102_102317

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ (¬ ∃ x : ℝ, x < 0 ∧ |x| = a * x - a) ↔ (a > 1 ∨ a ≤ -1) :=
sorry

end range_of_a_l102_102317


namespace probability_divisible_by_4_l102_102155

-- Definitions and assumptions
def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def fair_8_sided_die : Finset ℕ :=
  {n | n ∈ (Finset.range 9) \ {0}}

-- Main statement
theorem probability_divisible_by_4 :
  let P := (finset.filter (λ a, is_divisible_by_4 a) fair_8_sided_die).card.toRat / fair_8_sided_die.card.toRat in
  let P2 := P * P in
  (P2 = (1 / 16 : ℚ)) :=
sorry

end probability_divisible_by_4_l102_102155


namespace geometric_seq_log_sum_l102_102008

theorem geometric_seq_log_sum (a : ℕ → ℝ) (h : a 4 * a 5 = 2) :
  log 4 (a 1) + log 4 (a 2) + log 4 (a 3) + log 4 (a 4) + log 4 (a 5) + log 4 (a 6) + log 4 (a 7) + log 4 (a 8) = 2 := by
  sorry

end geometric_seq_log_sum_l102_102008


namespace object_distance_traveled_l102_102720

theorem object_distance_traveled
  (t : ℕ) (v_mph : ℝ) (mile_to_feet : ℕ)
  (h_t : t = 2)
  (h_v : v_mph = 68.18181818181819)
  (h_mile : mile_to_feet = 5280) :
  ∃ d : ℝ, d = 200 :=
by {
  sorry
}

end object_distance_traveled_l102_102720


namespace area_ratio_bdf_abc_l102_102169

-- Defining the conditions in terms of points and segments

variable (A B C D E F : Type)
variable [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint D] [IsPoint E] [IsPoint F]
variable (α β : ℝ)

-- Condition 1: AD:AB = α
axiom ad_ab_ratio : ∀ (A B D : IsPoint), segment_ratio AD AB = α 

-- Condition 2: EF:BE = β
axiom ef_be_ratio : ∀ (E B F : IsPoint), segment_ratio EF BE = β

-- Condition 3: E is the midpoint of BC
axiom e_midpoint_bc : midpoint E B C

-- Theorem: The ratio of the areas of triangles BDF and ABC is 1/4 (α + 1) (β + 1)
theorem area_ratio_bdf_abc : 
  (area (triangle B D F)) / (area (triangle A B C)) = (1/4) * (α + 1) * (β + 1) := 
by sorry

end area_ratio_bdf_abc_l102_102169


namespace find_dividend_l102_102609

-- Define the given conditions
def quotient : ℝ := 0.0012000000000000001
def divisor : ℝ := 17

-- State the problem: Prove that the dividend is the product of the quotient and the divisor
theorem find_dividend (q : ℝ) (d : ℝ) (hq : q = 0.0012000000000000001) (hd : d = 17) : 
  q * d = 0.0204000000000000027 :=
sorry

end find_dividend_l102_102609


namespace extremum_minimum_l102_102716

def f (x a : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

theorem extremum_minimum (a : ℝ) (h : ∃ x : ℝ, f x a = (x^2 + a*x - 1) * Real.exp (x - 1) ∧ x = -2) :
  ∃ x : ℝ, f x a = -1 :=
sorry

end extremum_minimum_l102_102716


namespace find_savings_l102_102163

-- Define the problem statement
def income_expenditure_problem (income expenditure : ℝ) (ratio : ℝ) : Prop :=
  (income / ratio = expenditure) ∧ (income = 20000)

-- Define the theorem for savings
theorem find_savings (income expenditure : ℝ) (ratio : ℝ) (h_ratio : ratio = 4 / 5) (h_income : income = 20000) : 
  income_expenditure_problem income expenditure ratio → income - expenditure = 4000 :=
by
  sorry

end find_savings_l102_102163


namespace packages_needed_l102_102763

/-- Kelly puts string cheeses in her kids' lunches 5 days per week. Her oldest wants 2 every day and her youngest will only eat 1.
The packages come with 30 string cheeses per pack. Prove that Kelly will need 2 packages of string cheese to fill her kids' lunches for 4 weeks. -/
theorem packages_needed (days_per_week : ℕ) (oldest_per_day : ℕ) (youngest_per_day : ℕ) (package_size : ℕ) (weeks : ℕ) :
  days_per_week = 5 →
  oldest_per_day = 2 →
  youngest_per_day = 1 →
  package_size = 30 →
  weeks = 4 →
  (2 * days_per_week + 1 * days_per_week) * weeks / package_size = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end packages_needed_l102_102763


namespace bottle_caps_proof_l102_102986

def bottle_caps_difference (found thrown : ℕ) := found - thrown

theorem bottle_caps_proof : bottle_caps_difference 50 6 = 44 := by
  sorry

end bottle_caps_proof_l102_102986


namespace simplify_and_evaluate_expression_l102_102815

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end simplify_and_evaluate_expression_l102_102815


namespace seeds_per_watermelon_l102_102258

theorem seeds_per_watermelon (total_seeds : ℕ) (num_watermelons : ℕ) (h : total_seeds = 400 ∧ num_watermelons = 4) : total_seeds / num_watermelons = 100 :=
by
  sorry

end seeds_per_watermelon_l102_102258


namespace compute_expression_l102_102783

def f (x : ℝ) := x + 3
def g (x : ℝ) := 2 * x
def f_inv (x : ℝ) := x - 3
def g_inv (x : ℝ) := x / 2

theorem compute_expression : f(g_inv(f_inv(f_inv(g(f(23)))))) = 26 :=
by
  sorry

end compute_expression_l102_102783


namespace sqrt_expression_l102_102998

theorem sqrt_expression (y : ℝ) (hy : y < 0) : 
  Real.sqrt (y / (1 - ((y - 2) / y))) = -y / Real.sqrt 2 := 
sorry

end sqrt_expression_l102_102998


namespace hyperbola_equation_l102_102328

-- Definitions for a given hyperbola
variables {a b : ℝ}
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Definitions for the asymptote condition
axiom point_on_asymptote : (4 : ℝ) = (b / a) * 3

-- Definitions for the focal distance condition
axiom point_circle_intersect : (3 : ℝ)^2 + 4^2 = (a^2 + b^2)

-- The goal is to prove the hyperbola's specific equation
theorem hyperbola_equation : 
  (a^2 = 9 ∧ b^2 = 16) →
  (∃ a b : ℝ, (4 : ℝ)^2 + 3^2 = (a^2 + b^2) ∧ 
               (4 : ℝ) = (b / a) * 3 ∧ 
               ((a^2 = 9) ∧ (b^2 = 16)) ∧ (a > 0) ∧ (b > 0)) :=
sorry

end hyperbola_equation_l102_102328


namespace proposition_p_is_true_negation_of_prop_p_l102_102668

open Real Classical

-- Proposition definition
def prop_p : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + 3 > 0

-- Negation of the proposition
def neg_prop_p : Prop :=
  ∃ x : ℝ, x^2 + 2 * x + 3 ≤ 0

-- Proof statements
theorem proposition_p_is_true : prop_p := 
begin
  sorry
end

theorem negation_of_prop_p : ¬prop_p ↔ neg_prop_p :=
by
  sorry

end proposition_p_is_true_negation_of_prop_p_l102_102668


namespace polynomial_root_l102_102286

theorem polynomial_root (r : ℝ) (P : ℝ[X])
    (hP : P = polynomial.C 8 * polynomial.X ^ 3 - polynomial.C 4 * polynomial.X ^ 2 -
                  polynomial.C 42 * polynomial.X + polynomial.C 45) :
    (∃ Q : ℝ[X], P = (polynomial.X - polynomial.C r) ^ 2 * Q) ↔ r = 3/2 := by
  sorry

end polynomial_root_l102_102286


namespace orange_bin_count_l102_102173

theorem orange_bin_count (initial_count throw_away add_new : ℕ) 
  (h1 : initial_count = 40) 
  (h2 : throw_away = 37) 
  (h3 : add_new = 7) : 
  initial_count - throw_away + add_new = 10 := 
by 
  sorry

end orange_bin_count_l102_102173


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102622

-- Conditions and Probability Definitions
noncomputable def prob_indicator (k : ℕ) : ℝ := (5 / 6) ^ (k - 1)

-- a) Expected total number of dice rolled
theorem expected_total_dice_rolled : 5 * (1 / (1 - 5/6)) = 30 := by
  sorry

-- b) Expected total number of points rolled
theorem expected_total_points_rolled :
  let expected_one_die_points : ℝ := 3.5 / (1 - 5/6)
  in 5 * expected_one_die_points = 105 := by
  sorry

-- c) Expected number of salvos
theorem expected_number_of_salvos :
  let expected_salvos (n : ℕ) : ℝ := 
    ∑ k in Finset.range n, (n.choose k * 5^(n - k) * (1/ (1 - 5/6))) / (6^n)
  in expected_salvos 5 ≈ 13.02 := by
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102622


namespace distance_between_parallel_lines_correct_l102_102981

open Real

noncomputable def distance_between_parallel_lines : ℝ :=
  let a := (3, 1)
  let b := (2, 4)
  let d := (4, -6)
  let v := (b.1 - a.1, b.2 - a.2)
  let d_perp := (6, 4) -- a vector perpendicular to d
  let v_dot_d_perp := v.1 * d_perp.1 + v.2 * d_perp.2
  let d_perp_dot_d_perp := d_perp.1 * d_perp.1 + d_perp.2 * d_perp.2
  let proj_v_onto_d_perp := (v_dot_d_perp / d_perp_dot_d_perp * d_perp.1, v_dot_d_perp / d_perp_dot_d_perp * d_perp.2)
  sqrt (proj_v_onto_d_perp.1 * proj_v_onto_d_perp.1 + proj_v_onto_d_perp.2 * proj_v_onto_d_perp.2)

theorem distance_between_parallel_lines_correct :
  distance_between_parallel_lines = (3 * sqrt 13) / 13 := by
  sorry

end distance_between_parallel_lines_correct_l102_102981


namespace quadratic_inequalities_l102_102721

-- Definitions based on conditions
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b * x + c

-- The theorem we need to prove
theorem quadratic_inequalities (b c : ℝ) :
  let f := quadratic_function b c in
  f 2 > f 1 ∧ f 1 > f 4 :=
by sorry

end quadratic_inequalities_l102_102721


namespace sum_common_elements_eq_2870_l102_102568

def AP1 (n : ℕ) : ℕ :=
  3 + (n - 1) * 4

def AP2 (m : ℕ) : ℕ :=
  2 + (m - 1) * 7

noncomputable def CommonElements : List ℕ :=
  List.filter (λ x, x ≤ 399 && x % 4 == 3) (List.range 400)

theorem sum_common_elements_eq_2870 :
  List.sum CommonElements = 2870 :=
by
  sorry

end sum_common_elements_eq_2870_l102_102568


namespace problem_statement_l102_102900

theorem problem_statement
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy : x ≠ y)
  (h_xy : x ≠ -y) :
  (∛(x^9 - x^6 * y^3) - y^2 * ∛(8 * x^6 / y^3 - 8 * x^3) + x * y^3 * ∛(y^3 - y^6 / x^3)) / (∛(x^8) * (x^2 - 2 * y^2) + ∛(x^2 * y^12)) =
  ∛(x - y) / (x + y) :=
  sorry

end problem_statement_l102_102900


namespace factorization_example_l102_102868

theorem factorization_example (x : ℝ) : (x^2 - 4 * x + 4) = (x - 2)^2 :=
by sorry

end factorization_example_l102_102868


namespace Kat_training_hours_l102_102759

-- Define the total number of hours spent on strength training per week
def strength_training_hours_per_week : ℕ := 1 * 3

-- Define the total number of hours spent on boxing training per week
def boxing_training_hours_per_week : ℕ := 1.5 * 4

-- Statement to prove the number of hours per week
theorem Kat_training_hours : strength_training_hours_per_week + boxing_training_hours_per_week = 9 := by
  sorry

end Kat_training_hours_l102_102759


namespace probability_at_least_two_heads_l102_102566

theorem probability_at_least_two_heads (tosses : ℕ) (p_heads : ℚ) (p_tails : ℚ) (n_heads : ℕ) :
  tosses = 5 → p_heads = 1/2 → p_tails = 1/2 → n_heads ≥ 2 →
  ∑ k in finset.range (n_heads), (nat.choose 5 k) * (p_heads ^ k) * (p_tails ^ (5 - k)) = 13/16 :=
by
  sorry

end probability_at_least_two_heads_l102_102566


namespace time_to_meet_l102_102108

variable (distance : ℕ)
variable (speed1 speed2 time : ℕ)

-- Given conditions
def distanceAB := 480
def speedPassengerCar := 65
def speedCargoTruck := 55

-- Sum of the speeds of the two vehicles
def sumSpeeds := speedPassengerCar + speedCargoTruck

-- Prove that the time it takes for the two vehicles to meet is 4 hours
theorem time_to_meet : sumSpeeds * time = distanceAB → time = 4 :=
by
  sorry

end time_to_meet_l102_102108


namespace parabola_vertex_l102_102826

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ y : ℝ, y = 2 * (x - 5)^2 + 3) → (5, 3) = (5, 3) :=
by
  intros x y_eq
  sorry

end parabola_vertex_l102_102826


namespace range_of_a_l102_102364

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a < -1 ∨ a > 3 :=
by
  sorry

end range_of_a_l102_102364


namespace log_equation_solution_l102_102603

theorem log_equation_solution :
  ∃ x : ℝ, log x 8 = log 81 3 ∧ x = 4096 :=
by
  sorry

end log_equation_solution_l102_102603


namespace find_n_that_solves_arctan_sum_l102_102282

theorem find_n_that_solves_arctan_sum :
  ∃ (n : ℕ), 0 < n ∧ arctan (1/3) + arctan (1/5) + arctan (1/7) + arctan (1/n) = π / 4 ∧ n = 8 :=
by
  use 8
  split
  . exact Nat.zero_lt_succ 7
  split
  . have h_ts : arctan (1/3) + arctan (1/5) = arctan (4/7) := by sorry
    have h_ts' : arctan (4/7) + arctan (1/7) = arctan (7/9) := by sorry
    rw [h_ts, h_ts']
    have h : arctan (7/9) + arctan (1/8) = π / 4 := by sorry
    exact h
  . rfl

end find_n_that_solves_arctan_sum_l102_102282


namespace final_number_not_2013cubed_l102_102426

def initial_set : Finset ℕ := Finset.range 26  -- {0, 1, ..., 25}
def initial_sum : ℕ := initial_set.sum id
def final_number (n : ℕ) : Prop := initial_sum ≡ 1 [MOD 3] ∧ ∀ a ∈ initial_set, ∀ b ∈ initial_set, ∀ c ∈ initial_set, 
a ≠ b → a ≠ c → b ≠ c → ∃ S : Finset ℕ, initial_set.erase a |> Finset.erase b |> Finset.erase c |>.insert (a^3 + b^3 + c^3) = S

theorem final_number_not_2013cubed : ¬ final_number (2013^3) := by
  sorry

end final_number_not_2013cubed_l102_102426


namespace complement_A_complement_A_inter_B_l102_102419

def U := Set ℝ
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B : Set ℝ := {x | x < 3 ∨ x ≥ 15}

theorem complement_A :
  U \ A = {x | x ≤ -1 ∨ x > 5} := 
by sorry

theorem complement_A_inter_B :
  U \ (A ∩ B) = {x | x ≤ -1 ∨ x ≥ 3} := 
by sorry

end complement_A_complement_A_inter_B_l102_102419


namespace solve_matrix_equality_l102_102597

def determinant_matrix := λ (x : ℝ), 12 * x^2 - 4 * x

theorem solve_matrix_equality (x : ℝ) : determinant_matrix x = 5 ↔ (x = 5 / 6 ∨ x = -1 / 2) := sorry

end solve_matrix_equality_l102_102597


namespace range_of_x_l102_102113

-- Define the function and its properties.
variable {X : Type} [LinearOrder X] [OrderedAddCommGroup X] 

-- Define the function f with the given properties.
variable (f : X → X)
    (monotonic_decreasing : ∀ ⦃x y: X⦄, x ≤ y → f y ≤ f x)
    (odd_function : ∀ x : X, f (-x) = -f x)
    (f_one : f 1 = -1)

-- Prove the range of x satisfying the given inequality.
theorem range_of_x (x : X) : 
  -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 :=
sorry

end range_of_x_l102_102113


namespace cleaner_flow_rate_after_second_unclogging_l102_102209

theorem cleaner_flow_rate_after_second_unclogging
  (rate1 rate2 : ℕ) (time1 time2 total_time total_cleaner : ℕ)
  (used_cleaner1 used_cleaner2 : ℕ)
  (final_rate : ℕ)
  (H1 : rate1 = 2)
  (H2 : rate2 = 3)
  (H3 : time1 = 15)
  (H4 : time2 = 10)
  (H5 : total_time = 30)
  (H6 : total_cleaner = 80)
  (H7 : used_cleaner1 = rate1 * time1)
  (H8 : used_cleaner2 = rate2 * time2)
  (H9 : used_cleaner1 + used_cleaner2 ≤ total_cleaner)
  (H10 : final_rate = (total_cleaner - (used_cleaner1 + used_cleaner2)) / (total_time - (time1 + time2))) :
  final_rate = 4 := by
  sorry

end cleaner_flow_rate_after_second_unclogging_l102_102209


namespace find_matrix_N_l102_102611

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
  ![\[3, 0\], \[0, -2\]]

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N * !\[⟨1, 0⟩\] = 3 * !\[⟨1, 0⟩\])  
  (h2 : N * !\[⟨0, 1⟩\] = -2 * !\[⟨0, 1⟩\]) : 
  N = matrix_N :=
by 
  sorry

end find_matrix_N_l102_102611


namespace boys_playing_both_sports_l102_102904

theorem boys_playing_both_sports : 
  ∀ (total boys basketball football neither both : ℕ), 
  total = 22 → boys = 22 → basketball = 13 → football = 15 → neither = 3 → 
  boys = basketball + football - both + neither → 
  both = 9 :=
by
  intros total boys basketball football neither both
  intros h_total h_boys h_basketball h_football h_neither h_formula
  sorry

end boys_playing_both_sports_l102_102904


namespace total_drink_volume_l102_102901

theorem total_drink_volume (oj wj gj : ℕ) (hoj : oj = 25) (hwj : wj = 40) (hgj : gj = 70) : (gj * 100) / (100 - oj - wj) = 200 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_drink_volume_l102_102901


namespace cash_sales_amount_l102_102099

-- Definitions for conditions
def total_sales : ℕ := 80
def credit_sales : ℕ := (2 * total_sales) / 5

-- Statement of the proof problem
theorem cash_sales_amount :
  ∃ cash_sales : ℕ, cash_sales = total_sales - credit_sales ∧ cash_sales = 48 :=
by
  sorry

end cash_sales_amount_l102_102099


namespace coating_profit_l102_102917

theorem coating_profit (x y : ℝ) (h1 : 0.6 * x + 0.9 * (150 - x) ≤ 120)
  (h2 : 0.7 * x + 0.4 * (150 - x) ≤ 90) :
  (50 ≤ x ∧ x ≤ 100) → (y = -50 * x + 75000) → (x = 50 → y = 72500) :=
by
  intros hx hy hx_val
  sorry

end coating_profit_l102_102917


namespace sum_diagonal_elements_l102_102135

theorem sum_diagonal_elements (n : ℕ) (h : n ≥ 4) 
  (a : ℕ → ℕ → ℝ) 
  (arithmetic_sequence : ∀ i, ∃ d, ∀ j, a i (j + 1) = a i j + d)
  (geometric_sequence : ∀ j, ∃ q, ∀ i, a (i + 1) j = a i j * q)
  (h_a24 : a 2 4 = 1)
  (h_a42 : a 4 2 = 1 / 8)
  (h_a43 : a 4 3 = 3 / 16) :
  (∑ k in Finset.range n, a k k) = 2 - (n + 2) / (2 ^ n) :=
sorry

end sum_diagonal_elements_l102_102135


namespace inequality_ordering_l102_102400

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 3

def mam : ℚ := (a + 3 * b) / 4
def mgm : ℚ := (a^2 * b)^(1/3 : ℝ)
def mhm : ℚ := ((a + 3 * b)^2) / (4 * (a + b))

theorem inequality_ordering :
  mam < mgm ∧ mgm < mhm :=
sorry

end inequality_ordering_l102_102400


namespace neg_mod_eq_1998_l102_102353

theorem neg_mod_eq_1998 {a : ℤ} (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end neg_mod_eq_1998_l102_102353


namespace volterra_solution_l102_102081

noncomputable def phi (x : ℝ) : ℝ := 1 / (1 + x^2)^(3/2 : ℝ)

theorem volterra_solution (x : ℝ) : 
  phi x = (1 / (1 + x^2)) - ∫ t in 0 .. x, (t / (1 + x^2)) * phi t :=
by
  sorry

end volterra_solution_l102_102081


namespace diamonds_balance_emerald_l102_102958

theorem diamonds_balance_emerald (D E : ℝ) (h1 : 9 * D = 4 * E) (h2 : 9 * D + E = 4 * E) : 3 * D = E := by
  sorry

end diamonds_balance_emerald_l102_102958


namespace compute_expression_l102_102239

theorem compute_expression : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end compute_expression_l102_102239


namespace false_propositions_count_l102_102360

def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

def f_prime (x : ℝ) (a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem false_propositions_count :
  ∃ (a b c : ℝ),
    (f 0 a b c = 0) ∧
    (f_prime 1 a b = -1) ∧
    (f_prime (-1) a b = -1) ∧
    -- Analytical expression of f(x) = x³ - 4x
    ((f = λ x, x^3 - 4 * x) ↔ true) ∧
    -- One and only one critical point
    ((∀ x, f_prime x a b = 0 → false) ∧ false ∨ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f_prime x1 a b = 0 ∧ f_prime x2 a b = 0)) ∧
    -- Sum of max and min values of f(x) is 0
    (let c_points := [-2, -1, 1, 2] in
    ∃ (max min : ℝ), max + min = 0 ∧
      max = max (list.map (λ x, f x a b c) c_points) ∧
      min = min (list.map (λ x, f x a b c) c_points)) →
    1 = 1 :=
by
  sorry

end false_propositions_count_l102_102360


namespace profit_percentage_B_l102_102937

theorem profit_percentage_B (price_A price_C : ℝ) (profit_A_percentage : ℝ) : 
  price_A = 150 → 
  price_C = 225 → 
  profit_A_percentage = 20 →
  let price_B := price_A + (profit_A_percentage / 100 * price_A) in
  let profit_B := price_C - price_B in
  let profit_B_percentage := (profit_B / price_B) * 100 in
  profit_B_percentage = 25 := 
by
  intros
  simp only
  sorry

end profit_percentage_B_l102_102937


namespace park_area_is_20000_l102_102929

noncomputable def park_area : ℝ :=
  let speed := 6    -- km/hr
  let time := 6 / 60 -- hrs
  let ratio := (1, 2)
  let distance := speed * time * 1000 -- meters
  let length := distance / (2 * (ratio.1 + ratio.2)) * ratio.1
  let breadth := distance / (2 * (ratio.1 + ratio.2)) * ratio.2
  length * breadth

theorem park_area_is_20000 :
  park_area = 20000 :=
by
  -- Calculations according to the conditions
  let speed := 6    -- km/hr
  let time := 6 / 60 -- hrs
  let ratio := (1, 2)
  let distance := speed * time * 1000 -- meters
  have h_distance : distance = 600 := by
    unfold distance
    norm_num
  let length := distance / (2 * (ratio.1 + ratio.2)) * ratio.1
  let breadth := distance / (2 * (ratio.1 + ratio.2)) * ratio.2
  have h_length : length = 100 := by
    unfold length distance ratio
    norm_num
  have h_breadth : breadth = 200 := by
    unfold breadth distance ratio
    norm_num
  unfold park_area length breadth
  norm_num
  sorry

end park_area_is_20000_l102_102929


namespace find_a_l102_102132

theorem find_a (a x : ℝ) (n : ℕ) (h1 : (∑ k in Finset.range (n + 1), (Nat.choose n k) * (a / x)^k * 3^(n - k)) = 256) 
               (h2 : x = 1) : a = -1 ∨ a = -5 :=
by 
  -- Variables and conditions setup according to the problem statement
  have h_sum_coeffs : (a + 3)^n = 256, from sorry,
  -- Proof that the only possible values of a are -1 or -5
  sorry

end find_a_l102_102132


namespace triangle_union_area_l102_102219

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_union_area :
  let A := (1, 2) in
  let B := (4, 3) in
  let C := (5, 5) in
  let A' := (2, 1) in
  let B' := (3, 4) in
  let C' := (5, 5) in
  triangle_area A B C + triangle_area A' B' C' = 5 :=
by
  sorry

end triangle_union_area_l102_102219


namespace amara_clothing_remaining_l102_102558

theorem amara_clothing_remaining :
  (initial_clothing - donated_first - donated_second - thrown_away = remaining_clothing) :=
by
  let initial_clothing := 100
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining_clothing := 65
  sorry

end amara_clothing_remaining_l102_102558


namespace set_intersection_l102_102531

def U := {1, 2, 3, 4, 5}
def A := {1, 2}
def B := {2, 3}
def comp_U_B := U \ B

theorem set_intersection :
  A ∩ comp_U_B = {1} :=
by
  sorry

end set_intersection_l102_102531


namespace carolyn_sum_of_removed_numbers_l102_102582

def initial_list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolynFirstMove := 5

theorem carolyn_sum_of_removed_numbers : ∑ (n : ℕ) in [5, 10, 9], n = 24 := by
  sorry

end carolyn_sum_of_removed_numbers_l102_102582


namespace geom_bio_students_difference_l102_102737

theorem geom_bio_students_difference (total_students geometry_students biology_students : ℕ)
  (h1 : total_students = 232) (h2 : geometry_students = 144) (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students,
      min_overlap := geometry_students + biology_students - total_students
  in max_overlap - min_overlap = 88 :=
by
  sorry

end geom_bio_students_difference_l102_102737


namespace remainder_1234567_div_256_l102_102969

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l102_102969


namespace pattern_continues_for_max_8_years_l102_102508

def is_adult_age (age : ℕ) := 18 ≤ age ∧ age < 40

def fits_pattern (p1 p2 n : ℕ) : Prop := 
  is_adult_age p1 ∧
  is_adult_age p2 ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 
    (k % (p1 + k) = 0 ∨ k % (p2 + k) = 0) ∧ ¬ (k % (p1 + k) = 0 ∧ k % (p2 + k) = 0))

theorem pattern_continues_for_max_8_years (p1 p2 : ℕ) : 
  fits_pattern p1 p2 8 := 
sorry

end pattern_continues_for_max_8_years_l102_102508


namespace bags_sold_on_friday_l102_102032

theorem bags_sold_on_friday (total_stock : ℕ) (sold_mon sold_tue sold_wed sold_thu : ℕ) (not_sold_percentage : ℚ) :
  total_stock = 600 →
  sold_mon = 25 →
  sold_tue = 70 →
  sold_wed = 100 →
  sold_thu = 110 →
  not_sold_percentage = 0.25 →
  let not_sold := (not_sold_percentage * total_stock : ℕ),
      sold_mon_to_thu := sold_mon + sold_tue + sold_wed + sold_thu,
      sold_fri := total_stock - (sold_mon_to_thu + not_sold) in
  sold_fri = 145 :=
by
  intros h_total_stock h_sold_mon h_sold_tue h_sold_wed h_sold_thu h_not_sold_percentage
  sorry

end bags_sold_on_friday_l102_102032


namespace part_a_part_b_part_c_l102_102616

noncomputable def expected_total_dice_rolls : ℕ := 30

noncomputable def expected_total_points_rolled : ℕ := 105

noncomputable def expected_number_of_salvos : ℚ := 13.02

theorem part_a :
  (let total_dice := 5
   in ∀ (salvo_rolls : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → salvo_rolls die = 6)
   → (total_dice * 6) = expected_total_dice_rolls) :=
by {
  sorry
}

theorem part_b :
  (let total_dice := 5
   in ∀ (points_rolled : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → points_rolled die = 21)
   → (total_dice * 21) = expected_total_points_rolled) :=
by {
  sorry
}

theorem part_c :
  (let total_dice := 5
   in ∀ (salvos : ℕ → ℚ),
   (∀ die : ℕ, die < total_dice → salvos die = 13.02)
   → (total_dice * 13.02) = expected_number_of_salvos) :=
by {
  sorry
}

end part_a_part_b_part_c_l102_102616


namespace work_done_by_force_l102_102184

noncomputable def F (x : ℝ) : ℝ :=
  if (0 ≤ x ∧ x ≤ 2) then 10
  else if (x > 2) then 3 * x + 4
  else 0 -- To ensure F is defined for all x ∈ ℝ

theorem work_done_by_force :
  (∫ x in 0..4, F x) = 46 := by
  sorry

end work_done_by_force_l102_102184


namespace mayo_bottles_count_l102_102034

theorem mayo_bottles_count
  (ketchup_ratio mayo_ratio : ℕ) 
  (ratio_multiplier ketchup_bottles : ℕ)
  (h_ratio_eq : 3 = ketchup_ratio)
  (h_mayo_ratio_eq : 2 = mayo_ratio)
  (h_ketchup_bottles_eq : 6 = ketchup_bottles)
  (h_ratio_condition : ketchup_bottles * mayo_ratio = ketchup_ratio * ratio_multiplier) :
  ratio_multiplier = 4 := 
by 
  sorry

end mayo_bottles_count_l102_102034


namespace max_area_of_triangle_ABC_l102_102412

-- Definitions for the problem conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (5, 4)
def parabola (x : ℝ) : ℝ := x^2 - 3 * x
def C (r : ℝ) : ℝ × ℝ := (r, parabola r)

-- Function to compute the Shoelace Theorem area of ABC
def shoelace_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Proof statement
theorem max_area_of_triangle_ABC : ∃ (r : ℝ), -2 ≤ r ∧ r ≤ 5 ∧ shoelace_area A B (C r) = 39 := 
  sorry

end max_area_of_triangle_ABC_l102_102412


namespace profit_ratio_l102_102461

-- Define the investment ratios and time periods as given in the conditions
variable (x : ℕ)
variable (p_investment_ratio q_investment_ratio : ℕ)
variable (p_time q_time : ℕ)

-- Given conditions
def investment_ratio_condition : p_investment_ratio = 7 ∧ q_investment_ratio = 5 := sorry
def time_period_condition : p_time = 7 ∧ q_time = 14 := sorry

-- Prove that the profit ratio of p to q is 7:10
theorem profit_ratio (p_investment_ratio q_investment_ratio p_time q_time : ℕ) :
  p_investment_ratio * p_time / q_investment_ratio / q_time = 7 / 10 := 
by
  -- Use given conditions to fill in the gaps
  have h1 : p_investment_ratio = 7 := (investment_ratio_condition).1
  have h2 : q_investment_ratio = 5 := (investment_ratio_condition).2
  have h3 : p_time = 7 := (time_period_condition).1
  have h4 : q_time = 14 := (time_period_condition).2
  -- Sorry to skip the proof
  sorry

end profit_ratio_l102_102461


namespace lateral_surface_area_inclined_prism_l102_102823

noncomputable def lateralSurfaceArea (a b : ℝ) : ℝ := 
  a * b * (1 + Real.sqrt 2)

theorem lateral_surface_area_inclined_prism (a b : ℝ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) :
    let base_is_equilateral : Prop = ∀ (x y z : ℝ), x = y ∧ y = z ∧ z = a
    let lateral_edge_length := b
    let angle_with_base_sides := 45
    lateralSurfaceArea a b = a * b * (1 + Real.sqrt 2) :=
by
  sorry

end lateral_surface_area_inclined_prism_l102_102823


namespace original_two_digit_number_l102_102943

theorem original_two_digit_number :
  ∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ 500 + x = 9 * x - 12 ∧ x = 64 :=
by
  have h₁ : ∀ (x : ℕ), 500 + x = 9 * x - 12 → x = 64 := sorry
  use 64
  split
  all_goals { sorry }

end original_two_digit_number_l102_102943


namespace max_area_segment_l102_102507

noncomputable def area_segment (r : ℝ) (α : ℝ) : ℝ :=
  0.5 * r^2 * (α - Real.sin α)

def radius_from_arc_length (α : ℝ) : ℝ :=
  1 / α

theorem max_area_segment :
  ∀ α : ℝ, (0 < α ∧ α ≤ 2 * Real.pi) →
  0.5 * (1 / α)^2 * (α - Real.sin α) ≤ 0.5 * (1 / Real.pi)^2 * (Real.pi - Real.sin Real.pi) :=
by
  intro α h
  sorry

end max_area_segment_l102_102507


namespace frog_mutation_percentages_l102_102739

theorem frog_mutation_percentages 
  (total_frogs : ℕ)
  (extra_legs_frogs two_heads_frogs bright_red_frogs skin_abnormalities_frogs extra_eyes_frogs : ℕ)
  (h_total_frogs : total_frogs = 150)
  (h_extra_legs_frogs : extra_legs_frogs = 25)
  (h_two_heads_frogs : two_heads_frogs = 15)
  (h_bright_red_frogs : bright_red_frogs = 10)
  (h_skin_abnormalities_frogs : skin_abnormalities_frogs = 8)
  (h_extra_eyes_frogs : extra_eyes_frogs = 5) :
  (25 * 100 / 150).nat_abs = 17 ∧
  (15 * 100 / 150).nat_abs = 10 ∧
  (10 * 100 / 150).nat_abs = 7 ∧
  (8 * 100 / 150).nat_abs = 5 ∧
  (5 * 100 / 150).nat_abs = 3 :=
  by sorry

end frog_mutation_percentages_l102_102739


namespace matching_pair_probability_l102_102265

/-- Proof goal: To prove that Emily picking two socks randomly from her drawer, 
which has 12 gray-bottomed socks, 10 white-bottomed socks, and 6 black-bottomed socks, 
gives a probability of 1/3 that she picks a matching pair. -/

theorem matching_pair_probability :
  let gray_count := 12
  let white_count := 10
  let black_count := 6
  let total_socks := gray_count + white_count + black_count
  let total_ways := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_count * (gray_count - 1) / 2
  let white_pairs := white_count * (white_count - 1) / 2
  let black_pairs := black_count * (black_count - 1) / 2
  let matching_pairs := gray_pairs + white_pairs + black_pairs
  let probability := matching_pairs / total_ways
  probability = 1/3 := 
sorry

end matching_pair_probability_l102_102265


namespace lines_perpendicular_to_same_plane_are_parallel_l102_102804

variables {Point Line Plane : Type}
variable (α : Plane)
variable (m n : Line)
variable perpendicular : Line → Plane → Prop
variable parallel : Line → Line → Prop

theorem lines_perpendicular_to_same_plane_are_parallel
  (hm : perpendicular m α)
  (hn : perpendicular n α) :
  parallel m n :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l102_102804


namespace matrix_computation_l102_102772

variable (N : Matrix (Fin 2) (Fin 2) ℝ)

def vec1 : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-2]]
def vec2 : Matrix (Fin 2) (Fin 1) ℝ := ![![4], ![1]]
def vec3 : Matrix (Fin 2) (Fin 1) ℝ := ![!-4, ![6]]
def vec4 : Matrix (Fin 2) (Fin 1) ℝ := ![!-2, ![0]]
def vec5 : Matrix (Fin 2) (Fin 1) ℝ := ![![7], ![0]]
def result : Matrix (Fin 2) (Fin 1) ℝ := ![![6], ![2]]

theorem matrix_computation :
  N.mul vec1 = vec2 → N.mul vec3 = vec4 → N.mul vec5 = result :=
by
  intros h1 h2
  sorry

end matrix_computation_l102_102772


namespace percentage_increase_in_combined_cost_l102_102035

theorem percentage_increase_in_combined_cost
  (bicycle_cost_last_year : ℝ) (helmet_cost_last_year : ℝ) (gloves_cost_last_year : ℝ)
  (bicycle_increase : ℝ) (helmet_increase : ℝ) (gloves_increase : ℝ)
  (h1 : bicycle_cost_last_year = 200) (h2 : helmet_cost_last_year = 50) (h3 : gloves_cost_last_year = 30)
  (h4 : bicycle_increase = 0.08) (h5 : helmet_increase = 0.15) (h6 : gloves_increase = 0.20) :
  let new_bicycle_cost := bicycle_cost_last_year * (1 + bicycle_increase)
      new_helmet_cost := helmet_cost_last_year * (1 + helmet_increase)
      new_gloves_cost := gloves_cost_last_year * (1 + gloves_increase)
      total_cost_last_year := bicycle_cost_last_year + helmet_cost_last_year + gloves_cost_last_year
      total_cost_this_year := new_bicycle_cost + new_helmet_cost + new_gloves_cost
      increase := total_cost_this_year - total_cost_last_year
      percentage_increase := (increase / total_cost_last_year) * 100 in
  percentage_increase ≈ 10.54 :=
by
  sorry

end percentage_increase_in_combined_cost_l102_102035


namespace runs_scored_in_match_l102_102076

variable {average_before : ℕ} (average_after : ℕ) (matches_before : ℕ) (runs_in_match : ℕ)

def total_runs_before := average_before * matches_before
def total_runs_after := average_after * (matches_before + 1)

theorem runs_scored_in_match 
  (h1 : average_before = 52)
  (h2 : average_after = 54)
  (h3 : matches_before = 12) :
  runs_in_match = total_runs_after average_before average_after matches_before - total_runs_before average_before matches_before := 
by
  sorry

#eval (total_runs_after 52 54 12) - (total_runs_before 52 12) -- Expected: 78

end runs_scored_in_match_l102_102076


namespace square_area_l102_102016

noncomputable def area_of_square (q : ℝ) : ℝ :=
  (144 * q^2) / (25 - 24 * real.sqrt(1 - q^2))

theorem square_area (a D N A M O : ℝ) (q : ℝ) 
  (hDN : D N = 4) (hAM : A M = 3) (hcos_DOA : real.cos (angle D O A) = q) :
  a^2 = area_of_square q := 
sorry

end square_area_l102_102016


namespace arithmetic_geometric_sequence_exists_l102_102139

theorem arithmetic_geometric_sequence_exists (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z) 
(h4 : x + y + z = 6) (h5 : ∃ (a d : ℝ), x = a - d ∧ y = a ∧ z = a + d) 
(h6 : ∃ (b c : ℝ), x = b ∧ y = c ∧ z = b * c) :
  {x, y, z} = {8.0, 2.0, -4.0} ∨ {x, y, z} = {-4.0, 2.0, 8.0} :=
sorry

end arithmetic_geometric_sequence_exists_l102_102139


namespace each_shopper_will_receive_amount_l102_102020

/-- Definitions of the given conditions -/
def isabella_has_more_than_sam : ℕ := 45
def isabella_has_more_than_giselle : ℕ := 15
def giselle_money : ℕ := 120

/-- Calculation based on the provided conditions -/
def isabella_money : ℕ := giselle_money + isabella_has_more_than_giselle
def sam_money : ℕ := isabella_money - isabella_has_more_than_sam
def total_money : ℕ := isabella_money + sam_money + giselle_money

/-- The total amount each shopper will receive when the donation is shared equally -/
def money_each_shopper_receives : ℕ := total_money / 3

/-- Main theorem to prove the statement derived from the problem -/
theorem each_shopper_will_receive_amount :
  money_each_shopper_receives = 115 := by
  sorry

end each_shopper_will_receive_amount_l102_102020


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102620

-- Conditions and Probability Definitions
noncomputable def prob_indicator (k : ℕ) : ℝ := (5 / 6) ^ (k - 1)

-- a) Expected total number of dice rolled
theorem expected_total_dice_rolled : 5 * (1 / (1 - 5/6)) = 30 := by
  sorry

-- b) Expected total number of points rolled
theorem expected_total_points_rolled :
  let expected_one_die_points : ℝ := 3.5 / (1 - 5/6)
  in 5 * expected_one_die_points = 105 := by
  sorry

-- c) Expected number of salvos
theorem expected_number_of_salvos :
  let expected_salvos (n : ℕ) : ℝ := 
    ∑ k in Finset.range n, (n.choose k * 5^(n - k) * (1/ (1 - 5/6))) / (6^n)
  in expected_salvos 5 ≈ 13.02 := by
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102620


namespace function_is_monotonic_and_odd_l102_102673

   variable (a : ℝ) (x : ℝ)

   noncomputable def f : ℝ := (a^x - a^(-x))

   theorem function_is_monotonic_and_odd (h1 : a > 0) (h2 : a ≠ 1) : 
     (∀ x : ℝ, f (-x) = -f (x)) ∧ ((a > 1 → ∀ x y : ℝ, x < y → f x < f y) ∧ (0 < a ∧ a < 1 → ∀ x y : ℝ, x < y → f x > f y)) :=
   by
         sorry
   
end function_is_monotonic_and_odd_l102_102673


namespace minimum_distance_l102_102671

-- Define the ellipse equation
def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 25) + (P.2^2 / 16) = 1

-- Define the first circle equation
def circle1 (M : ℝ × ℝ) : Prop :=
  ((M.1 + 3)^2) + (M.2^2) = 1

-- Define the second circle equation
def circle2 (N : ℝ × ℝ) : Prop :=
  ((N.1 - 3)^2) + (N.2^2) = 4

-- Define distances from points P, M, N
def distance (A B : ℝ × ℝ) : ℝ :=
  (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

-- Main theorem statement
theorem minimum_distance (P M N : ℝ × ℝ) (hP : ellipse P) (hM : circle1 M) (hN : circle2 N) :
  ∃ (r : ℝ), r = 7 ∧ (distance P M + distance P N) ≥ r :=
sorry

end minimum_distance_l102_102671


namespace transformation_population_growth_decline_mortality_rate_l102_102467

theorem transformation_population_growth_decline_mortality_rate
  (birth_rate decline: Bool) (mortality_rate decline: Bool) (natural_growth_rate decline increase: Bool) : 
  decline in mortality_rate -> 
  (decline in birth_rate ∨ decline in natural_growth_rate ∨ increase in natural_growth_rate ∨ decline in mortality_rate) := 
begin
  sorry
end

end transformation_population_growth_decline_mortality_rate_l102_102467


namespace find_four_digit_number_l102_102606

def digits_sum (n : ℕ) : ℕ := (n / 1000) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)
def digits_product (n : ℕ) : ℕ := (n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)

theorem find_four_digit_number :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (digits_sum n) * (digits_product n) = 3990 :=
by
  -- The proof is omitted as instructed.
  sorry

end find_four_digit_number_l102_102606


namespace number_of_special_four_digit_numbers_l102_102071

theorem number_of_special_four_digit_numbers : 
  ∃ n, n = 1296 ∧ ∀ num : ℕ,
    num >= 1000 ∧ num < 2000 ∧ -- Four-digit number starting with '1'
    ((num / 1000 = 1) ∧  -- The first digit is '1'
    ((exists d, d ∈ {1,2,3,4,5,6,7,8,9} ∧ 
    (num % 10 == d ∧ (num % 100 / 10) = d ∨ (num % 1000 / 100) = d ∨ (num % 10000 / 1000) = d))) -- Exactly two identical digits
    → 
    count four-digit numbers with given properties) = n :=
sorry

end number_of_special_four_digit_numbers_l102_102071


namespace jenny_first_year_spending_l102_102756

def adoption_fee : ℝ := 50
def vet_visits_cost : ℝ := 500
def monthly_food_cost : ℝ := 25
def toys_cost : ℝ := 200

def total_cost_first_year : ℝ :=
  adoption_fee + vet_visits_cost + (monthly_food_cost * 12)

def jenny_share : ℝ :=
  total_cost_first_year / 2

theorem jenny_first_year_spending :
  jenny_share + toys_cost = 625 := by
  sorry

end jenny_first_year_spending_l102_102756


namespace solve_quadratic_l102_102439

theorem solve_quadratic : 
  ∀ x : ℝ, (x - 1) ^ 2 = 64 → (x = 9 ∨ x = -7) :=
by
  sorry

end solve_quadratic_l102_102439


namespace quadratic_roots_integer_sum_eq_198_l102_102692

theorem quadratic_roots_integer_sum_eq_198 (x p q x1 x2 : ℤ) 
  (h_eqn : x^2 + p * x + q = 0)
  (h_roots : (x - x1) * (x - x2) = 0)
  (h_pq_sum : p + q = 198) :
  (x1 = 2 ∧ x2 = 200) ∨ (x1 = 0 ∧ x2 = -198) :=
sorry

end quadratic_roots_integer_sum_eq_198_l102_102692


namespace count_six_digit_numbers_with_sum_52_l102_102567

/-- Definition to check if a number is a six-digit number within a certain range. -/
def is_six_digit_number (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- Function to compute the sum of the digits of a number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Theorem stating that there are exactly 21 six-digit numbers with digit sum equal to 52. -/
theorem count_six_digit_numbers_with_sum_52 : 
  (∑ n in (Finset.filter (λ n, is_six_digit_number n ∧ digit_sum n = 52) (Finset.range 1000000)), 1) = 21 := 
sorry

end count_six_digit_numbers_with_sum_52_l102_102567


namespace number_of_common_points_l102_102119

open Real

theorem number_of_common_points (k : ℝ) (hk : k ≠ 0) :
  let line := λ y, y = 2 * k in
  let curve := λ x y, 9 * k^2 * x^2 + y^2 = 18 * k^2 * (2 * abs x - x) in
  ∃ (p : ℝ × ℝ), line p.2 ∧ curve p.1 p.2 ∧
  ∃ (q : ℝ × ℝ), line q.2 ∧ curve q.1 q.2 ∧
  ∃ (r : ℝ × ℝ), line r.2 ∧ curve r.1 r.2 ∧
  ∃ (s : ℝ × ℝ), line s.2 ∧ curve s.1 s.2 ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s :=
sorry

end number_of_common_points_l102_102119


namespace point_B_in_fourth_quadrant_l102_102000

theorem point_B_in_fourth_quadrant (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a > 0) ∧ (-b < 0) → (quadrant (a, -b) = 4) :=
by
  sorry

end point_B_in_fourth_quadrant_l102_102000


namespace range_of_a_l102_102664

variable {x a : ℝ}

def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := |x| > a

theorem range_of_a (h : ¬p x → ¬q x a) : a ≤ 1 :=
sorry

end range_of_a_l102_102664


namespace edge_length_of_colored_paper_l102_102728

theorem edge_length_of_colored_paper (a : ℝ) (n : ℕ) (h_box : a = 12) (h_paper : n = 54) :
  ∀ l, (6 * a^2 = n * l^2) → l = 4 :=
by
  -- Define the edge length of the cube and the number of pieces of paper
  assume h_box h_paper
  sorry

end edge_length_of_colored_paper_l102_102728


namespace correct_choice_l102_102951

theorem correct_choice :
  ¬(∃ r : ℝ, r * 0 = 1) → 
  ¬(∃ q : ℚ, (↑π / 2 : ℝ) = (q : ℝ)) →
  3 < real.sqrt 15 ∧ real.sqrt 15 < 4 →
  ¬ (real.sqrt 9 = -3 ∨ real.sqrt 9 = 3) →
  (3 < real.sqrt 15 ∧ real.sqrt 15 < 4) :=
by 
  intro h1 h2 h3 h4 
  exact h3

end correct_choice_l102_102951


namespace smallest_rectangles_needed_l102_102502

theorem smallest_rectangles_needed {a b : ℕ} (h1 : a = 3) (h2 : b = 4) :
  ∃ (n : ℕ), ∀ (s : ℕ), square_side s ∧ (∃ m, m * (a * b) = s * s) → n = 12 :=
begin
  sorry

end smallest_rectangles_needed_l102_102502


namespace five_hash_neg_one_l102_102987

def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

theorem five_hash_neg_one : hash 5 (-1) = 0 :=
by
  sorry

end five_hash_neg_one_l102_102987


namespace John_new_weekly_earnings_l102_102392

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end John_new_weekly_earnings_l102_102392


namespace min_max_f_on_interval_l102_102857

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_f_on_interval :
  ∃ min max, min = - (3 * Real.pi) / 2 ∧ max = (Real.pi / 2) + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min ∧ f x ≤ max) :=
sorry

end min_max_f_on_interval_l102_102857


namespace poly_a_c_sum_l102_102781

theorem poly_a_c_sum {a b c d : ℝ} (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 + a * x + b)
  (hg : ∀ x, g x = x^2 + c * x + d)
  (hv_f_root_g : g (-a / 2) = 0)
  (hv_g_root_f : f (-c / 2) = 0)
  (f_min : ∀ x, f x ≥ -25)
  (g_min : ∀ x, g x ≥ -25)
  (f_g_intersect : f 50 = -25 ∧ g 50 = -25) : a + c = -101 :=
by
  sorry

end poly_a_c_sum_l102_102781


namespace base_coat_drying_time_l102_102025

/-- Jane is painting her fingernails. She applies a base coat that takes some time to dry,
  two color coats that take 3 minutes each to dry, and a clear top coat that takes 5 minutes to dry.
  Jane spends a total of 13 minutes waiting for her nail polish to dry.
  Prove that the base coat takes 2 minutes to dry. --/
theorem base_coat_drying_time :
  let total_time := 13 in
  let color_coats_time := 2 * 3 in
  let top_coat_time := 5 in
  let x := total_time - (color_coats_time + top_coat_time) in
  x = 2 := by
    /- Proof goes here -/
    sorry

end base_coat_drying_time_l102_102025


namespace anna_correct_percentage_l102_102226

theorem anna_correct_percentage :
  let first_test_probs := 20
  let first_test_score := 0.85
  let second_test_probs := 50
  let second_test_score := 0.92
  let third_test_probs := 30
  let third_test_score := 0.75
  let fourth_test_probs := 15
  let fourth_test_score := 0.60
  let fifth_test_probs := 40
  let fifth_test_score := 0.95
  let sixth_test_probs := 5
  let sixth_test_score := 0.80
  let total_probs := first_test_probs + second_test_probs + third_test_probs + fourth_test_probs + fifth_test_probs + sixth_test_probs
  let total_correct := (first_test_score * first_test_probs) + (second_test_score * second_test_probs) + (third_test_score * third_test_probs) + 
                        (fourth_test_score * fourth_test_probs) + (fifth_test_score * fifth_test_probs) + (sixth_test_score * sixth_test_probs)
  let correct_percentage := total_correct / total_probs * 100
  correct_percentage = 85.31 :=
by
  let first_test_probs := 20
  let first_test_score := 0.85
  let second_test_probs := 50
  let second_test_score := 0.92
  let third_test_probs := 30
  let third_test_score := 0.75
  let fourth_test_probs := 15
  let fourth_test_score := 0.60
  let fifth_test_probs := 40
  let fifth_test_score := 0.95
  let sixth_test_probs := 5
  let sixth_test_score := 0.80
  let total_probs := first_test_probs + second_test_probs + third_test_probs + fourth_test_probs + fifth_test_probs + sixth_test_probs
  let total_correct := (first_test_score * first_test_probs) + (second_test_score * second_test_probs) + (third_test_score * third_test_probs) + 
                        (fourth_test_score * fourth_test_probs) + (fifth_test_score * fifth_test_probs) + (sixth_test_score * sixth_test_probs)
  let correct_percentage := total_correct / total_probs * 100
  have : correct_percentage = 85.3125 := by sorry
  exact this

end anna_correct_percentage_l102_102226


namespace sector_area_correct_l102_102448

def central_angle := (Real.pi / 3)
def chord_length := 3
def radius := 3
def area_of_sector := (1 / 2) * radius * central_angle * radius

theorem sector_area_correct : area_of_sector = (3 * Real.pi / 2) := 
by
  -- Placeholder for proof
  sorry

end sector_area_correct_l102_102448


namespace area_triangle_ABC_is_correct_l102_102482

noncomputable def radius : ℝ := 4

noncomputable def angleABDiameter : ℝ := 30

noncomputable def ratioAM_MB : ℝ := 2 / 3

theorem area_triangle_ABC_is_correct :
  ∃ (area : ℝ), area = (180 * Real.sqrt 3) / 19 :=
by sorry

end area_triangle_ABC_is_correct_l102_102482


namespace sum_of_exponents_outside_radical_l102_102083

theorem sum_of_exponents_outside_radical (x y z : ℝ) :
  let expr := 40 * x^5 * y^9 * z^14
  in let simplified_expr := 2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3)
  in (if (simplified_expr = (2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3))) then (1 + 1 + 3 = 5) else false) := sorry

end sum_of_exponents_outside_radical_l102_102083


namespace find_n_gt_101_same_rightmost_digit_l102_102125

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def rightmost_nonzero_digit (n : ℕ) : ℕ :=
let fac := factorial n in
(fac / (10 ^ (nat.find_greatest (λ m, 10 ^ m ∣ fac) fac))) % 10

theorem find_n_gt_101_same_rightmost_digit :
  ∃ n : ℕ, n > 101 ∧ rightmost_nonzero_digit 101 = rightmost_nonzero_digit n ∧
           ∀ m : ℕ, m > 101 ∧ m < n → rightmost_nonzero_digit 101 ≠ rightmost_nonzero_digit m :=
by
  sorry

end find_n_gt_101_same_rightmost_digit_l102_102125


namespace largest_angle_in_triangle_l102_102465

theorem largest_angle_in_triangle : 
  ∀ (A B C : ℝ), A + B + C = 180 ∧ A + B = 105 ∧ (A = B + 40)
  → (C = 75) :=
by
  sorry

end largest_angle_in_triangle_l102_102465


namespace largest_prime_factor_binom_300_150_l102_102891

theorem largest_prime_factor_binom_300_150 :
  let n := Nat.choose 300 150 in
  ∃ p : ℕ, Nat.Prime p ∧ 10 ≤ p ∧ p < 100 ∧
  (p > 75 → 3 * p < 300 ∧ p ∣ n) ∧
  ∀ q : ℕ, Nat.Prime q ∧ 10 ≤ q ∧ q < 100 → q ∣ n → q ≤ p :=
  sorry

end largest_prime_factor_binom_300_150_l102_102891


namespace find_n_l102_102348

theorem find_n (n : ℕ) (h : 2 * 2^2 * 2^n = 2^10) : n = 7 :=
sorry

end find_n_l102_102348


namespace phi_sum_geq_inverse_sum_l102_102784

def is_injective (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, f a = f b → a = b

theorem phi_sum_geq_inverse_sum (φ : ℕ → ℕ) (hφ : is_injective φ) :
  ∀ n : ℕ, (∑ k in finset.range (n + 1), (φ k) / (k + 1) ^ 2) ≥ (∑ k in finset.range (n + 1), 1 / (k + 1)) := 
sorry

end phi_sum_geq_inverse_sum_l102_102784


namespace intercept_difference_l102_102405

theorem intercept_difference (f g : ℝ → ℝ) 
  (h1 : ∀ x, g x = -f (120 - x))
  (h2 : ∃ v, v ∈ set.range g ∧ v ∈ set.range f) 
  (x1 x2 x3 x4 : ℝ)
  (intercepts_order : x1 < x2 ∧ x2 < x3 ∧ x3 < x4)
  (intercepts_difference : x3 - x2 = 180) :
  x4 - x1 = 540 + 360 * real.sqrt 2 :=
sorry

end intercept_difference_l102_102405


namespace coeff_x3_in_1_minus_x_pow_10_l102_102100

theorem coeff_x3_in_1_minus_x_pow_10 :
  coefficient (polynomial.of_fn (λ i, if i = 3 then -120 else 0)) (1 - x) ^ 10 = -120 :=
by
  sorry

end coeff_x3_in_1_minus_x_pow_10_l102_102100


namespace marketing_firm_ratio_l102_102925

theorem marketing_firm_ratio (total_households : ℕ)
  (neither_soap : ℕ)
  (only_A : ℕ)
  (both_A_and_B : ℕ)
  (only_B : ℕ)
  (households_surveyed : total_households = 200)
  (households_neither : neither_soap = 80)
  (households_only_A : only_A = 60)
  (households_both_A_and_B : both_A_and_B = 5)
  (households_total : neither_soap + only_A + both_A_and_B + only_B = total_households) :
  only_B / both_A_and_B = 11 :=
by
  have h1 : 80 + 60 + 5 + only_B = 200,
  { rw [households_neither, households_only_A, households_both_A_and_B, households_total], },
  have h2 : 145 + only_B = 200,
  { rw [add_assoc, add_assoc, add_comm 5, add_comm 60, add_comm 80, h1], },
  have h3 : only_B = 55,
  { linarith, },
  show only_B / both_A_and_B = 11,
  { rw [h3, households_both_A_and_B], norm_num, }

end marketing_firm_ratio_l102_102925


namespace coloring_count_l102_102985

noncomputable def count_colorings : ℕ :=
  let colors := {red, white, blue},
      dots := {dot1, dot2, dot3, dot4, dot5, dot6, dot7, dot8, dot9, dot10},
      edges := {(dot1, dot2), (dot2, dot3), (dot3, dot1), -- first triangle
                (dot3, dot4), (dot4, dot5), (dot5, dot3), -- second triangle
                (dot5, dot6), (dot6, dot7), (dot7, dot5)}, -- third triangle
  {c : dots → colors // ∀ (d1 d2 : dots), (d1, d2) ∈ edges → c d1 ≠ c d2}.to_finset.card

theorem coloring_count : count_colorings = 216 :=
  sorry

end coloring_count_l102_102985


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102624

-- Conditions
constant NumDice : ℕ := 5

-- Define the probability of a die not showing six on each roll
def prob_not_six := (5 : ℚ) / 6

-- Question a: Expected total number of dice rolled
theorem expected_total_dice_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 30
  sorry

-- Question b: Expected total number of points rolled
theorem expected_total_points_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 105
  sorry

-- Question c: Expected number of salvos
theorem expected_number_of_salvos :
  True := by
  -- The proof would calculate the expected number of salvos as derived, resulting in 13.02
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102624


namespace gina_minutes_of_netflix_l102_102641

-- Define the conditions given in the problem
def gina_chooses_three_times_as_often (g s : ℕ) : Prop :=
  g = 3 * s

def total_shows_watched (g s : ℕ) : Prop :=
  g + s = 24

def duration_per_show : ℕ := 50

-- The theorem that encapsulates the problem statement and the correct answer
theorem gina_minutes_of_netflix (g s : ℕ) (h1 : gina_chooses_three_times_as_often g s) 
    (h2 : total_shows_watched g s) :
    g * duration_per_show = 900 :=
by
  sorry

end gina_minutes_of_netflix_l102_102641


namespace eggs_per_week_is_84_l102_102757

-- Define the number of pens
def number_of_pens : Nat := 4

-- Define the number of emus per pen
def emus_per_pen : Nat := 6

-- Define the number of days in a week
def days_in_week : Nat := 7

-- Define the number of eggs per female emu per day
def eggs_per_female_emu_per_day : Nat := 1

-- Calculate the total number of emus
def total_emus : Nat := number_of_pens * emus_per_pen

-- Calculate the number of female emus
def female_emus : Nat := total_emus / 2

-- Calculate the number of eggs per day
def eggs_per_day : Nat := female_emus * eggs_per_female_emu_per_day

-- Calculate the number of eggs per week
def eggs_per_week : Nat := eggs_per_day * days_in_week

-- The theorem to prove
theorem eggs_per_week_is_84 : eggs_per_week = 84 := by
  sorry

end eggs_per_week_is_84_l102_102757


namespace health_risk_problem_correct_l102_102574

noncomputable def health_risk_problem : ℕ :=
let p := 21
let q := 55
in p + q

theorem health_risk_problem_correct (p q : ℕ) (hpq_coprime : Nat.coprime p q) (hp : p = 21) (hq : q = 55) :
  p + q = 76 :=
by
  rw [hp, hq]
  exact Nat.add_trivial 21 55

end health_risk_problem_correct_l102_102574


namespace gina_netflix_time_l102_102643

theorem gina_netflix_time (sister_shows : ℕ) (show_length : ℕ) (ratio : ℕ) (sister_ratio : ℕ) :
sister_shows = 24 →
show_length = 50 →
ratio = 3 →
sister_ratio = 1 →
(ratio * sister_shows * show_length = 3600) :=
begin
  intros hs hl hr hsr,
  rw hs,
  rw hl,
  rw hr,
  rw hsr,
  norm_num,
  sorry
end

end gina_netflix_time_l102_102643


namespace canoe_upstream_speed_l102_102188

noncomputable def speed_of_canoe_upstream (S downstream_speed : ℝ) : ℝ :=
  let C := downstream_speed - S in C - S

theorem canoe_upstream_speed : ∀ S downstream_speed : ℝ, S = 4 → downstream_speed = 12 → speed_of_canoe_upstream S downstream_speed = 4 :=
by
  intros S downstream_speed hS hD
  unfold speed_of_canoe_upstream
  rw [hS, hD]
  norm_num
  sorry

end canoe_upstream_speed_l102_102188


namespace gcd_lcm_product_l102_102283

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 3 * 5^2) (h2 : b = 5^3) : 
  Nat.gcd a b * Nat.lcm a b = 9375 := by
  sorry

end gcd_lcm_product_l102_102283


namespace proof_warehouse_l102_102555

variables (x y : ℝ)

def cost_eq : Prop := 40 * x + 2 * 45 * y + 20 * x * y = 3200

def domain_x : Prop := 0 < x ∧ x < 80

def y_function_eq (x y : ℝ) (h_cost : cost_eq x y) (h_domain : domain_x x) : y = (320 - 4 * x) / (9 + 2 * x) :=
sorry

def max_surface_area : Prop := 
  maximize_surface_area : x = 15 ∧ 100 ≤ 3200 + 20 * x * y :=
sorry

theorem proof_warehouse : 
  ∀ (x : ℝ), domain_x x → ∀ (y : ℝ), 
    cost_eq x y →
    (y = (320 - 4 * x) / (9 + 2 * x)) → 
    (maximize_surface_area x) :=
sorry

end proof_warehouse_l102_102555


namespace ratio_largest_to_sum_of_others_l102_102241

theorem ratio_largest_to_sum_of_others :
  let S := ∑ i in finsupp.range (11 + 1), (2: ℤ) ^ (i + 1)
  let ratio := (2: ℤ) ^ 12 / S
  ratio ≈ 1 :=
by
  let a := 2
  let r := 2
  let n := 11
  let S := a * (r^n - 1) / (r - 1)
  let ratio := (2: ℤ) ^ 12 / S
  -- The required proof steps would be filled in here.
  sorry

end ratio_largest_to_sum_of_others_l102_102241


namespace sin_alpha_trigonometric_expression_l102_102687

variable (α : ℝ)

-- Given condition: α such that its terminal side passes through point (4/5, -3/5) on the unit circle
def point_on_unit_circle (P : ℝ × ℝ) := ∀ α, P = (Real.cos α, Real.sin α)

noncomputable def point_P : ℝ × ℝ := (4/5, -3/5)

-- Conjecture 1: To prove sin α = -3/5
theorem sin_alpha : point_on_unit_circle point_P → Real.sin α = -3/5 := 
by
  intro h
  sorry

-- Conjecture 2: To prove the given trigonometric expression equals -15/16
theorem trigonometric_expression :
  (point_on_unit_circle point_P) → 
  (Real.cos(α + Real.pi / 2) / Real.sin(Real.pi + α) * Real.tan(Real.pi + α) / Real.cos(3 * Real.pi - α)) = -15 / 16 :=
by
  intro h
  sorry


end sin_alpha_trigonometric_expression_l102_102687


namespace chess_tournament_games_l102_102179

theorem chess_tournament_games (n : ℕ) (h1 : n = 20) (h2 : ∀ i j, i ≠ j -> (number_of_games i j = 2)) : number_of_total_games = 760 :=
by
  sorry

end chess_tournament_games_l102_102179


namespace range_of_f_l102_102722

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) * Real.exp x

theorem range_of_f'_over_f (f : ℝ → ℝ)
  (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x)
  (h2 : f 0 = 1)
  (x : ℝ) (hx : 0 < x) :
  1 < (deriv f x) / (f x) ∧ (deriv f x) / (f x) ≤ 2 :=
by
  sorry

end range_of_f_l102_102722


namespace trader_profit_percentage_l102_102903

-- Definitions used in the problem
variables (C : ℝ) (hC : 0 < C)
def initial_selling_price := 1.2 * C
def new_selling_price := 2 * initial_selling_price

-- Proof problem statement
theorem trader_profit_percentage :
  let new_profit_percentage := ((new_selling_price - C) / C) * 100 in
  new_profit_percentage = 140 :=
by
  let initial_selling_price := 1.2 * C
  let new_selling_price := 2 * initial_selling_price
  let new_profit_percentage := ((new_selling_price - C) / C) * 100
  calc 
    new_profit_percentage 
        = ((2 * (1.2 * C) - C) / C) * 100 : by refl
    ... = ((2.4 * C - C) / C) * 100 : by rw [mul_assoc]
    ... = (1.4 * C / C) * 100 : by { congr, ring }
    ... = 1.4 * 100 : by rw div_self hC.ne'
    ... = 140 : by norm_num

end trader_profit_percentage_l102_102903


namespace number_of_monomials_is_3_l102_102743

def expr1 := 3 * a ^ 2 + b
def expr2 := -2
def expr3 := (3 * x * y ^ 3) / 5
def expr4 := (a ^ 2 * b) / 3 + 1
def expr5 := a ^ 2 - 3 * b ^ 2
def expr6 := 2 * a * b * c

def isMonomial (e : Expr) : Prop :=
  match e with
  | Expr.monomial _ => true
  | _ => false

theorem number_of_monomials_is_3 : 
  (∃ expr1 = e1, isMonomial e1 = false) ∧ 
  (∃ expr2 = e2, isMonomial e2 = true) ∧ 
  (∃ expr3 = e3, isMonomial e3 = true) ∧ 
  (∃ expr4 = e4, isMonomial e4 = false) ∧ 
  (∃ expr5 = e5, isMonomial e5 = false) ∧ 
  (∃ expr6 = e6, isMonomial e6 = true) → 
  3 sorry

end number_of_monomials_is_3_l102_102743


namespace cannot_be_covered_l102_102587

-- Define the checkerboard dimensions and conditions
def checkerboard_5x3_missing_one_corner := 5 * 3 - 1 = 14
def checkerboard_4x4 := 4 * 4 = 16
def checkerboard_5x5_missing_two_diagonally := 5 * 5 - 2 = 23
def checkerboard_4x6 := 4 * 6 = 24
def checkerboard_7x2 := 7 * 2 = 14

-- Define a theorem to prove which checkerboard configuration cannot be covered completely by dominos
theorem cannot_be_covered (c : ℕ) : 
  ((c ≠ checkerboard_5x3_missing_one_corner) ∧
   (c ≠ checkerboard_4x4) ∧
   (c ≠ checkerboard_4x6) ∧
   (c ≠ checkerboard_7x2) ∧
   (c = checkerboard_5x5_missing_two_diagonally)) :=
by sorry

end cannot_be_covered_l102_102587


namespace Ryan_reads_more_l102_102513

theorem Ryan_reads_more 
  (total_pages_Ryan : ℕ)
  (days_in_week : ℕ)
  (pages_per_book_brother : ℕ)
  (books_per_day_brother : ℕ)
  (total_pages_brother : ℕ)
  (Ryan_books : ℕ)
  (Ryan_weeks : ℕ)
  (Brother_weeks : ℕ)
  (days_in_week_def : days_in_week = 7)
  (total_pages_Ryan_def : total_pages_Ryan = 2100)
  (pages_per_book_brother_def : pages_per_book_brother = 200)
  (books_per_day_brother_def : books_per_day_brother = 1)
  (Ryan_weeks_def : Ryan_weeks = 1)
  (Brother_weeks_def : Brother_weeks = 1)
  (total_pages_brother_def : total_pages_brother = pages_per_book_brother * days_in_week)
  : ((total_pages_Ryan / days_in_week) - (total_pages_brother / days_in_week) = 100) :=
by
  -- We provide the proof steps
  sorry

end Ryan_reads_more_l102_102513


namespace part1_a_n_formula_part1_S_n_formula_part1_b_n_formula_part2_T_n_bound_l102_102867

open Nat

-- Conditions from the problem
def S (n : ℕ) : ℕ :=
  ∑ i in range (n + 1), a i

def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 ^ (n - 1)

def b (n : ℕ) : ℕ :=
  2 * n - 1

def c (n : ℕ) : ℝ :=
  1 / ((b n) * (b (n + 1)))

def T (n : ℕ) : ℝ :=
  ∑ i in range n, c i

-- The required proof problem with conditions
theorem part1_a_n_formula (n : ℕ) : a n = 2^(n-1) :=
sorry

theorem part1_S_n_formula (n : ℕ) : S n = 2^n - 1 :=
sorry

theorem part1_b_n_formula (n : ℕ) : b n = 2 * n - 1 :=
sorry

theorem part2_T_n_bound (n : ℕ) : T n < 1 / 2 :=
sorry

end part1_a_n_formula_part1_S_n_formula_part1_b_n_formula_part2_T_n_bound_l102_102867


namespace largest_indecomposable_amount_l102_102004

theorem largest_indecomposable_amount (n : ℕ) : 
  ∀ s, ¬(∃ k : ℕ, s = k * 5 ^ (n + 1) - 2 * 3 ^ (n + 1)) → 
       ¬(∃ (m : ℕ), m < 5 ∧ ∃ (r : ℕ), s = 5 * r + m * 3) :=
by
  intro s h_decomposable
  sorry

end largest_indecomposable_amount_l102_102004


namespace ram_birthday_l102_102264

theorem ram_birthday
    (L : ℕ) (L1 : ℕ) (Llast : ℕ) (d : ℕ) (languages_learned_per_day : ℕ) (days_in_month : ℕ) :
    (L = 1000) →
    (L1 = 820) →
    (Llast = 1100) →
    (days_in_month = 28 ∨ days_in_month = 29 ∨ days_in_month = 30 ∨ days_in_month = 31) →
    (d = days_in_month - 1) →
    (languages_learned_per_day = (Llast - L1) / d) →
    ∃ n : ℕ, n = 19 :=
by
  intros hL hL1 hLlast hDays hm_d hLearned
  existsi 19
  sorry

end ram_birthday_l102_102264


namespace largest_among_abcd_l102_102718

theorem largest_among_abcd (a b c d k : ℤ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = k + 3 ∧
  a = k + 1 ∧
  b = k - 2 ∧
  d = k - 4 ∧
  c > a ∧
  c > b ∧
  c > d :=
by
  sorry

end largest_among_abcd_l102_102718


namespace existence_of_monic_poly_degree_4_with_root_sqrt2_add_sqrt5_l102_102271

theorem existence_of_monic_poly_degree_4_with_root_sqrt2_add_sqrt5 :
  ∃ p : Polynomial ℚ, Polynomial.monic p ∧ p.degree = 4 ∧ p.eval (Real.sqrt 2 + Real.sqrt 5) = 0 :=
by
  sorry

end existence_of_monic_poly_degree_4_with_root_sqrt2_add_sqrt5_l102_102271


namespace fuel_capacity_ratio_l102_102024

noncomputable def oldCost : ℝ := 200
noncomputable def newCost : ℝ := 480
noncomputable def priceIncreaseFactor : ℝ := 1.20

theorem fuel_capacity_ratio (C C_new : ℝ) (h1 : newCost = C_new * oldCost * priceIncreaseFactor / C) : 
  C_new / C = 2 :=
sorry

end fuel_capacity_ratio_l102_102024


namespace cylindrical_to_rectangular_l102_102591

theorem cylindrical_to_rectangular (r θ z : ℝ) (h₁ : r = 10) (h₂ : θ = Real.pi / 6) (h₃ : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5 * Real.sqrt 3, 5, 2) := 
by
  sorry

end cylindrical_to_rectangular_l102_102591


namespace Jean_had_41_candies_at_first_l102_102027

-- Let total_candies be the initial number of candies Jean had
variable (total_candies : ℕ)
-- Jean gave 18 pieces to a friend
def given_away := 18
-- Jean ate 7 pieces
def eaten := 7
-- Jean has 16 pieces left now
def remaining := 16

-- Calculate the total number of candies initially
def candy_initial (total_candies given_away eaten remaining : ℕ) : Prop :=
  total_candies = remaining + (given_away + eaten)

-- Prove that Jean had 41 pieces of candy initially
theorem Jean_had_41_candies_at_first : candy_initial 41 given_away eaten remaining :=
by
  -- Skipping the proof for now
  sorry

end Jean_had_41_candies_at_first_l102_102027


namespace min_max_values_l102_102849

noncomputable def f (x : ℝ) := cos x + (x + 1) * sin x + 1

theorem min_max_values :
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≥ - (3 * π) / 2) ∧
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≤ (π / 2 + 2)) :=
sorry

end min_max_values_l102_102849


namespace sequence_exists_l102_102572

def pawn_move (n : ℕ) (start : ℕ × ℕ) : Prop :=
  ∃ seq : List (ℕ × ℕ), seq.length = n^2 ∧
    List.Nodup seq ∧
    seq.head = some start ∧ 
    List.last seq sorry = some start ∧
    (∀ i < n - 1, (seq.nth i).1 = (seq.nth (i + 1)).2)

theorem sequence_exists (n : ℕ) (start : ℕ × ℕ) :
  ∃ seq : List (ℕ × ℕ), pawn_move n start :=
sorry

end sequence_exists_l102_102572


namespace prime_factors_of_M_are_one_l102_102594

-- Given condition
def condition (M : ℝ) : Prop :=
  log 3 (log 7 (log 11 (log 13 M))) = 7

-- Desired property/statement
theorem prime_factors_of_M_are_one (M : ℝ) (h : condition M) : 
  ∃! p : ℝ, prime p ∧ p.factor_of M := sorry

end prime_factors_of_M_are_one_l102_102594


namespace count_valid_arrangements_l102_102190

-- Definitions based on conditions
def total_chairs : Nat := 48

def valid_factor_pairs (n : Nat) : List (Nat × Nat) :=
  [ (2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2) ]

def count_valid_arrays : Nat := valid_factor_pairs total_chairs |>.length

-- The theorem we want to prove
theorem count_valid_arrangements : count_valid_arrays = 8 := 
  by
    -- proof should be provided here
    sorry

end count_valid_arrangements_l102_102190


namespace exists_monic_polynomial_with_root_l102_102270

theorem exists_monic_polynomial_with_root (r : ℝ) :
  ∃ p : Polynomial ℚ, Polynomial.monic p ∧ Polynomial.degree p = 4 ∧
  Polynomial.coeff p (numer (RootSpaceOfRoot r (Polynomial.map Polynomial.C Polynomial.X)) = 1 ∧
  ∀ (x : ℝ), Polynomial.eval x p = 0 ↔ x = r ∨ x = -r := 
  sorry

end exists_monic_polynomial_with_root_l102_102270


namespace pages_per_day_difference_l102_102514

theorem pages_per_day_difference :
  ∀ (total_pages_Ryan : ℕ) (days : ℕ) (pages_per_book_brother : ℕ) (books_per_day_brother : ℕ),
    total_pages_Ryan = 2100 →
    days = 7 →
    pages_per_book_brother = 200 →
    books_per_day_brother = 1 →
    (total_pages_Ryan / days) - (books_per_day_brother * pages_per_book_brother) = 100 := 
by
  intros total_pages_Ryan days pages_per_book_brother books_per_day_brother
  intros h_total_pages_Ryan h_days h_pages_per_book_brother h_books_per_day_brother
  have h1 : total_pages_Ryan / days = 300 := by sorry
  have h2 : books_per_day_brother * pages_per_book_brother = 200 := by sorry
  rw [h1, h2]
  exact rfl

end pages_per_day_difference_l102_102514


namespace pig_problem_l102_102382

theorem pig_problem (x y : ℕ) (h₁ : y - 100 = 100 * x) (h₂ : y = 90 * x) : x = 10 ∧ y = 900 := 
by
  sorry

end pig_problem_l102_102382


namespace correct_statements_about_f_l102_102681

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x * Real.cos x - 1 / 2

theorem correct_statements_about_f :
  (∀ x : ℝ, f x ≤ (√2 + 1) / 2) → 
  (∀ x, f (x + π) = f x) → 
  (f (-π/8) = 0) → 
  (∀ x, f (2 * x - π/4) = (√2 / 2) * Real.sin x) → 
  ( B && C && D ) :=
by
  sorry

end correct_statements_about_f_l102_102681


namespace parabola_vertex_l102_102463

theorem parabola_vertex (c d : ℝ)
  (h1 : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ∈ set.Icc (-6) (-1) ∨ x ≥ 4))
  : ∃ vx vy : ℝ, vx = 7 / 2 ∧ vy = -171 / 4 :=
by
  sorry

end parabola_vertex_l102_102463


namespace intersection_M_N_l102_102334

open Set

def M := { x : ℝ | 0 < x ∧ x < 3 }
def N := { x : ℝ | x^2 - 5 * x + 4 ≥ 0 }

theorem intersection_M_N :
  { x | x ∈ M ∧ x ∈ N } = { x | 0 < x ∧ x ≤ 1 } :=
sorry

end intersection_M_N_l102_102334


namespace poly_expansion_l102_102601

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end poly_expansion_l102_102601


namespace remainder_of_division_l102_102976

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l102_102976


namespace grade_above_B_l102_102069

theorem grade_above_B (total_students : ℕ) (percentage_below_B : ℕ) (students_above_B : ℕ) :
  total_students = 60 ∧ percentage_below_B = 40 ∧ students_above_B = total_students * (100 - percentage_below_B) / 100 →
  students_above_B = 36 :=
by
  sorry

end grade_above_B_l102_102069


namespace tangent_lines_count_l102_102486

theorem tangent_lines_count (C : ℝ → ℝ) (hC : ∀ x, C x = x^2 * Real.exp x) :
  ∃ n, n = 3 ∧ ∀ P, P = (2, 0) → ∃ k : ℕ, k = n ∧ ∀ i < k, ∃ x₀, (P.1 - x₀, C P.1 - C x₀) ∈ tangent_to P C :=
sorry

end tangent_lines_count_l102_102486


namespace solve_for_x_l102_102091

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end solve_for_x_l102_102091


namespace correct_statement_l102_102807

def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_statement :
  ¬ (∀ x, f x = 4 * Real.cos (2 * x + Real.pi / 3)) ∧ 
  (∃ x, f (-x - Real.pi / 6) = f (x + Real.pi / 6)) ∧ 
  ¬ (∃ T > 0, T < 2 * Real.pi ∧ ∀ x, f (x) = f (x + T)) ∧ 
  ¬ (∃ x, f x = f (-x - Real.pi / 6)) := 
sorry

end correct_statement_l102_102807


namespace planes_intersect_or_parallel_l102_102659

-- Define the tetrahedron and the point P
variables {A₁ A₂ A₃ A₄ P : Point}
variables {P₁₂ P₂₃ P₃₁ P₁₄ P₂₄ P₃₄ : Plane}

-- Set up the conditions for the planes being symmetric to P₁₂, P₂₃, etc.
axiom symmetric_planes (A₁ A₂ A₃ A₄ P : Point) (P₁₂ P₂₃ P₃₁ P₁₄ P₂₄ P₃₄ : Plane) : 
(⟦ (P₁₂ = symmetric_plane (P, A₁, A₂)) ⟧ ∧ 
 ⟦ (P₂₃ = symmetric_plane (P, A₂, A₃)) ⟧ ∧ 
 ⟦ (P₃₁ = symmetric_plane (P, A₃, A₁)) ⟧ ∧ 
 ⟦ (P₁₄ = symmetric_plane (P, A₁, A₄)) ⟧ ∧ 
 ⟦ (P₂₄ = symmetric_plane (P, A₂, A₄)) ⟧ ∧ 
 ⟦ (P₃₄ = symmetric_plane (P, A₃, A₄)) ⟧)

noncomputable def intersection_or_parallel : Prop :=
(∃ P_star : Point, 
 P₁₂.contains P_star ∧ P₂₃.contains P_star ∧ P₃₁.contains P_star ∧ P₁₄.contains P_star ∧ P₂₄.contains P_star ∧ P₃₄.contains P_star) ∨
(∃ l : Line, 
 P₁₂ ∥ l ∧ P₂₃ ∥ l ∧ P₃₁ ∥ l ∧ P₁₄ ∥ l ∧ P₂₄ ∥ l ∧ P₃₄ ∥ l)

theorem planes_intersect_or_parallel :
 symmetric_planes A₁ A₂ A₃ A₄ P P₁₂ P₂₃ P₃₁ P₁₄ P₂₄ P₃₄ → intersection_or_parallel :=
by sorry

end planes_intersect_or_parallel_l102_102659


namespace car_speed_decrease_l102_102094

theorem car_speed_decrease (d : ℝ) (speed_first : ℝ) (distance_fifth : ℝ) (time_interval : ℝ) :
  speed_first = 45 ∧ distance_fifth = 4.4 ∧ time_interval = 8 / 60 ∧ speed_first - 4 * d = distance_fifth / time_interval -> d = 3 :=
by
  intros h
  obtain ⟨_, _, _, h_eq⟩ := h
  sorry

end car_speed_decrease_l102_102094


namespace division_remainder_l102_102613

-- let f(r) = r^15 + r + 1
def f (r : ℝ) : ℝ := r^15 + r + 1

-- let g(r) = r^2 - 1
def g (r : ℝ) : ℝ := r^2 - 1

-- remainder polynomial b(r)
def b (r : ℝ) : ℝ := r + 1

-- Lean statement to prove that polynomial division of f(r) by g(r) 
-- yields the remainder b(r)
theorem division_remainder (r : ℝ) : (f r) % (g r) = b r :=
  sorry

end division_remainder_l102_102613


namespace part_a_part_b_part_c_l102_102619

noncomputable def expected_total_dice_rolls : ℕ := 30

noncomputable def expected_total_points_rolled : ℕ := 105

noncomputable def expected_number_of_salvos : ℚ := 13.02

theorem part_a :
  (let total_dice := 5
   in ∀ (salvo_rolls : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → salvo_rolls die = 6)
   → (total_dice * 6) = expected_total_dice_rolls) :=
by {
  sorry
}

theorem part_b :
  (let total_dice := 5
   in ∀ (points_rolled : ℕ → ℕ),
   (∀ die : ℕ, die < total_dice → points_rolled die = 21)
   → (total_dice * 21) = expected_total_points_rolled) :=
by {
  sorry
}

theorem part_c :
  (let total_dice := 5
   in ∀ (salvos : ℕ → ℚ),
   (∀ die : ℕ, die < total_dice → salvos die = 13.02)
   → (total_dice * 13.02) = expected_number_of_salvos) :=
by {
  sorry
}

end part_a_part_b_part_c_l102_102619


namespace compute_expression_l102_102982

theorem compute_expression : 7^2 + 4 * 5 - 2^3 = 61 := by
  calc
    7^2 + 4 * 5 - 2^3 = 49 + 20 - 8 : by
      have h1: 7^2 = 49 := by norm_num
      have h2: 4 * 5 = 20 := by norm_num
      have h3: 2^3 = 8 := by norm_num
      rw [h1, h2, h3]
    ... = 69 - 8 : by norm_num
    ... = 61    : by norm_num


end compute_expression_l102_102982


namespace polygon_num_sides_l102_102931

-- Define the given conditions
def perimeter : ℕ := 150
def side_length : ℕ := 15

-- State the theorem to prove the number of sides of the polygon
theorem polygon_num_sides (P : ℕ) (s : ℕ) (hP : P = perimeter) (hs : s = side_length) : P / s = 10 :=
by
  sorry

end polygon_num_sides_l102_102931


namespace amara_clothing_remaining_l102_102557

theorem amara_clothing_remaining :
  (initial_clothing - donated_first - donated_second - thrown_away = remaining_clothing) :=
by
  let initial_clothing := 100
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining_clothing := 65
  sorry

end amara_clothing_remaining_l102_102557


namespace age_of_b_l102_102159

variable {a b c d Y : ℝ}

-- Conditions
def condition1 (a b : ℝ) := a = b + 2
def condition2 (b c : ℝ) := b = 2 * c
def condition3 (a d : ℝ) := d = a / 2
def condition4 (a b c d Y : ℝ) := a + b + c + d = Y

-- Theorem to prove
theorem age_of_b (a b c d Y : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 b c) 
  (h3 : condition3 a d) 
  (h4 : condition4 a b c d Y) : 
  b = Y / 3 - 1 := 
sorry

end age_of_b_l102_102159


namespace set_intersection_complement_l102_102337

open Set

variable (U P Q: Set ℕ)

theorem set_intersection_complement (hU: U = {1, 2, 3, 4}) (hP: P = {1, 2}) (hQ: Q = {2, 3}) :
  P ∩ (U \ Q) = {1} :=
by
  sorry

end set_intersection_complement_l102_102337


namespace decreasing_interval_l102_102841

noncomputable def f (a b x : ℝ) := x^3 - 3 * a * x + b

theorem decreasing_interval (a b : ℝ) (h_a : a > 0) (h_max : ∀ x, f a b x ≤ 6) (h_min : ∀ x, f a b x ≥ 2) :
  ∃ I : set ℝ, I = { x : ℝ | -1 < x ∧ x < 1 } ∧ ∀ x ∈ I, has_deriv_at (f a b x) (3 * x^2 - 3 * a) x ∧ 3 * x^2 - 3 * a < 0 :=
by
  sorry

end decreasing_interval_l102_102841


namespace mady_2500th_step_total_balls_l102_102060

theorem mady_2500th_step_total_balls :
  let step := 2500
  let base := 9
  let balls_in_boxes := 20
  ∑ i in (step.digits base), i = balls_in_boxes := by
  sorry

end mady_2500th_step_total_balls_l102_102060


namespace string_cheese_packages_l102_102761

theorem string_cheese_packages (days_per_week : ℕ) (weeks : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) (pack_size : ℕ) 
    (H1 : days_per_week = 5)
    (H2 : weeks = 4)
    (H3 : oldest_daily = 2)
    (H4 : youngest_daily = 1)
    (H5 : pack_size = 30) 
  : (oldest_daily * days_per_week + youngest_daily * days_per_week) * weeks / pack_size = 2 :=
  sorry

end string_cheese_packages_l102_102761


namespace saber_toothed_frog_tails_l102_102466

def tails_saber_toothed_frog (n k : ℕ) (x : ℕ) : Prop :=
  5 * n + 4 * k = 100 ∧ n + x * k = 64

theorem saber_toothed_frog_tails : ∃ x, ∃ n k : ℕ, tails_saber_toothed_frog n k x ∧ x = 3 := 
by
  sorry

end saber_toothed_frog_tails_l102_102466


namespace find_ab_l102_102672

noncomputable def validate_ab : Prop :=
  let n : ℕ := 8
  let a : ℕ := n^2 - 1
  let b : ℕ := n
  a = 63 ∧ b = 8

theorem find_ab : validate_ab :=
by
  sorry

end find_ab_l102_102672


namespace find_intersection_points_max_AB_distance_l102_102746

-- Define the conditions for the curves C2 and C3 in rectangular coordinates
def curve_C2 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * y

def curve_C3 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * sqrt 3 * x

-- Prove the intersection points of C2 and C3
def intersection_points : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ curve_C2 x y ∧ curve_C3 x y }

theorem find_intersection_points :
  intersection_points = {(0, 0), (sqrt 3 / 2, 3 / 2)} :=
by
  -- Skipping the detailed proof here
  sorry

-- Define the rectangular and polar coordinates conversion
def polar_to_rect (rho theta : ℝ) : ℝ × ℝ :=
  (rho * cos theta, rho * sin theta)

-- Define the curve C1 in terms of alpha
def curve_C1 (t α : ℝ) (h : 0 ≤ α ∧ α < π) : ℝ × ℝ :=
  (t * cos α, t * sin α)

-- Define the AB distance
def AB_distance (α : ℝ) : ℝ :=
  |2 * sin α - 2 * sqrt 3 * cos α|

-- Prove the maximum value of |AB|
theorem max_AB_distance : 
  ∃ α, 0 ≤ α ∧ α < π ∧ AB_distance α = 4 :=
by
  -- Skipping the detailed proof here
  sorry

end find_intersection_points_max_AB_distance_l102_102746


namespace probability_of_committee_correct_l102_102446

noncomputable def probability_of_committee_with_at_least_one_boy_and_one_girl : ℚ :=
  let total_members := 24
  let boys := 14
  let girls := 10
  let committee_size := 5
  let total_ways := Nat.choose total_members committee_size
  let ways_all_boys := Nat.choose boys committee_size
  let ways_all_girls := Nat.choose girls committee_size
  let ways_unwanted := ways_all_boys + ways_all_girls
  let desirable_ways := total_ways - ways_unwanted in
  (desirable_ways : ℚ) / total_ways

theorem probability_of_committee_correct :
  probability_of_committee_with_at_least_one_boy_and_one_girl = 40250 / 42504 := by
  sorry

end probability_of_committee_correct_l102_102446


namespace sequence_arithmetic_and_sum_inequality_l102_102683

theorem sequence_arithmetic_and_sum_inequality 
  (a : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, T n = (∏ k in finset.range n, a k))
  (h2 : ∀ n, a n + T n = 1) :
  (∀ n, (1 / T n) - (1 / T (n + 1)) = 1) ∧ 
  (∑ k in finset.range(n + 1), (a (k + 1) - a k) / a k < 3 / 4) :=
by
  sorry

end sequence_arithmetic_and_sum_inequality_l102_102683


namespace roots_cubic_eq_l102_102780

noncomputable def polynomial : Polynomial ℂ := Polynomial.x^3 - 2 * Polynomial.x^2 + 3 * Polynomial.x - 4

theorem roots_cubic_eq (a b c : ℂ) : 
  (Polynomial.aeval a polynomial = 0) ∧
  (Polynomial.aeval b polynomial = 0) ∧
  (Polynomial.aeval c polynomial = 0) ∧
  (a + b + c = 2) ∧
  (a * b + a * c + b * c = 3) ∧
  (a * b * c = 4) → a^3 + b^3 + c^3 = 2 := 
by
  sorry

end roots_cubic_eq_l102_102780


namespace tangent_line_P_eq_trajectory_eq_l102_102298

-- Definitions based on conditions
def circleC : set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 + 2 * p.1 - 4 * p.2 + 1 = 0}
def originO : ℝ × ℝ := (0, 0)
def isOutside (P : ℝ × ℝ) : Prop := ∀ x y, (x, y) ∉ circleC
def isTangentPoint (P M : ℝ × ℝ) : Prop := ∃ k : ℝ, ∃ b : ℝ, ∀ p : ℝ × ℝ, p ∈ circleC → p.2 = k * p.1 + b ∧ (p.1, p.2) = M

-- Problem 1
theorem tangent_line_P_eq : 
  ∀ (P : ℝ × ℝ), 
  P = (1, 3) →
  isOutside P →
  ∃ (k : ℝ) (b : ℝ),
  (∀ x y, isTangentPoint P (x, y) → (y = k * x + b ∨ x = 1 ∨ 3 * x + 4 * y - 15 = 0)) := 
sorry

-- Problem 2
theorem trajectory_eq :
  ∀ (P : ℝ × ℝ),
  isOutside P →
  ∃ (M : ℝ × ℝ),
  isTangentPoint P M → 
  dist P M = dist P originO →
  ∀ x y, (x, y) = P → (2 * x - 4 * y + 1 = 0) := 
sorry

end tangent_line_P_eq_trajectory_eq_l102_102298


namespace smallest_eight_digit_with_four_fours_l102_102149

theorem smallest_eight_digit_with_four_fours :
  ∃ n : ℕ, (10000000 ≤ n ∧ n < 100000000) ∧ (count_digit 4 n = 4) ∧ (∀ m : ℕ, (10000000 ≤ m ∧ m < 100000000) ∧ (count_digit 4 m = 4) → n ≤ m) → n = 10004444 := sorry

/-- Helper function to count occurrences of a digit in a number -/
def count_digit (d : ℕ) (n : ℕ) : ℕ :=
  (n.digits 10).count d


end smallest_eight_digit_with_four_fours_l102_102149


namespace same_orientation_equilateral_centroids_l102_102047

-- Definitions for complex numbers and triangles
noncomputable def triangle_oriented (A B C : ℂ) : Prop :=
  ∃ j : ℂ, j * j * j = 1 ∧ B = j * A ∧ C = j^2 * A

def centroid (A B C : ℂ) : ℂ :=
  (A + B + C) / 3

def is_equilateral_triangle (A B C : ℂ) : Prop :=
  abs (B - A) = abs (C - B) ∧ abs (A - C) = abs (B - A)

-- Problem statement part 1: Prove that triangles ABC and MNP have the same orientation
theorem same_orientation (A B C M N P : ℂ)
  (h1 : triangle_oriented B M C)
  (h2 : triangle_oriented C N A)
  (h3 : triangle_oriented A P B)
  (h4 : M = -complex.exp(2 * real.pi * I / 3) * B - complex.exp(4 * real.pi * I / 3) * C)
  (h5 : N = -complex.exp(2 * real.pi * I / 3) * C - complex.exp(4 * real.pi * I / 3) * A)
  (h6 : P = -complex.exp(2 * real.pi * I / 3) * A - complex.exp(4 * real.pi * I / 3) * B) : 
  centroid A B C = centroid M N P :=
sorry

-- Problem statement part 2: Prove that G_1G_2G_3 forms an equilateral triangle with direct orientation
theorem equilateral_centroids (A B C M N P G1 G2 G3 : ℂ)
  (h1 : triangle_oriented B M C)
  (h2 : triangle_oriented C N A)
  (h3 : triangle_oriented A P B)
  -- derivable from centroids of BMC, CNA, APB respectively
  (hG1 : G1 = (B + M + C) / 3)
  (hG2 : G2 = (C + N + A) / 3)
  (hG3 : G3 = (A + P + B) / 3) :
  is_equilateral_triangle G1 G2 G3 :=
sorry

end same_orientation_equilateral_centroids_l102_102047


namespace eight_boys_travel_distance_l102_102260

theorem eight_boys_travel_distance :
  let radius := 50
  let distance := 1600 + 800 * Real.sqrt 2
  ∃ (n : ℕ), 
    n = 8 ∧ 
    (∀ (boy_walk: ℕ), boy_walk ∈ Set.range (Finset.range n) →
    ∃ θ : List ℝ, 
      List.Perm θ [Real.pi / 4, 3 * Real.pi / 4, 5 * Real.pi / 4, 7 * Real.pi / 4] ∧ 
      ∀ angle ∈ θ, Real.dist (Real.cos (2 * Real.pi * boy_walk / n) * radius + 
                                Real.sin (2 * Real.pi * boy_walk / n) * radius)
                                (Real.cos (2 * Real.pi * (boy_walk + angle) / n) * 
                                distance) + 
                                Real.sin (2 * Real.pi * (boy_walk + angle) / n) * 
                                distance = distance) :=
  sorry

end eight_boys_travel_distance_l102_102260


namespace limit_sum_binom_inv_l102_102596

-- Define the function for the sum S_n
def S (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (nat.choose n k)⁻¹

-- The theorem to prove
theorem limit_sum_binom_inv : Tendsto (λ n : ℕ, S n) at_top (𝓝 2) :=
sorry

end limit_sum_binom_inv_l102_102596


namespace calculation_correct_l102_102234

theorem calculation_correct : 
    (Real.sqrt 2 + Real.abs (Real.cbrt (-27)) + 2 * Real.sqrt 9 + Real.sqrt 2 * (Real.sqrt 2 - 1) = 11) :=
by
  sorry

end calculation_correct_l102_102234


namespace direction_vector_of_line_l102_102107

theorem direction_vector_of_line : ∀ (m : ℝ), ∃ (v : ℝ × ℝ), v = (3, 4) ∧ v ∈ set_of (λ w : ℝ × ℝ, ∃ k : ℝ, w = (k * 1, k * (4/3))) :=
by
  sorry

end direction_vector_of_line_l102_102107


namespace angle_C_measure_l102_102660

theorem angle_C_measure (A B C : ℝ) (a b c : ℝ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : A + B + C = π)
  (h3 : ∀ x y z, sin x / sin y = z / y) -- Law of Sines assumption
  (h4 : (sin A - sin C) * (a + c) = b * (sin A - sin B)) : C = π / 3 :=
sorry

end angle_C_measure_l102_102660


namespace stratified_sampling_l102_102879

-- Define population and sample sizes and their ratios
def num_male_students := 500
def num_female_students := 400
def sample_male_students := 25
def sample_female_students := 20

-- Define ratios
def population_ratio := num_male_students / num_female_students
def sample_ratio := sample_male_students / sample_female_students

-- Prove that the sampling method is stratified based on ratio equality
theorem stratified_sampling :
  population_ratio = sample_ratio →
  "Stratified Sampling Method" :=
by
  sorry

end stratified_sampling_l102_102879


namespace cricket_scores_l102_102193

-- Define the conditions
variable (X : ℝ) (A B C D E average10 average6 : ℝ)
variable (matches10 matches6 : ℕ)

-- Set the given constants
axiom average_runs_10 : average10 = 38.9
axiom matches_10 : matches10 = 10
axiom average_runs_6 : average6 = 42
axiom matches_6 : matches6 = 6

-- Define the equations based on the conditions
axiom eq1 : X = average10 * matches10
axiom eq2 : A + B + C + D = X - (average6 * matches6)
axiom eq3 : E = (A + B + C + D) / 4

-- The target statement
theorem cricket_scores : X = 389 ∧ A + B + C + D = 137 ∧ E = 34.25 :=
  by
    sorry

end cricket_scores_l102_102193


namespace baskets_needed_l102_102033
open Nat

theorem baskets_needed : 
    (let keith_turnips := 6 in
     let alyssa_turnips := 9 in
     let sean_carrots := 5 in
     let turnip_basket_capacity := 5 in
     let carrot_basket_capacity := 4 in

     let total_turnips := keith_turnips + alyssa_turnips in
     let turnip_baskets := (total_turnips + turnip_basket_capacity - 1) / turnip_basket_capacity in
     let carrot_baskets := (sean_carrots + carrot_basket_capacity - 1) / carrot_basket_capacity in
     let total_baskets := turnip_baskets + carrot_baskets in

     total_baskets = 5) := by
    sorry

end baskets_needed_l102_102033


namespace Mr_Caiden_payment_l102_102067

-- Defining the conditions as variables and constants
def total_roofing_needed : ℕ := 300
def cost_per_foot : ℕ := 8
def free_roofing : ℕ := 250

-- Define the remaining roofing needed and the total cost
def remaining_roofing : ℕ := total_roofing_needed - free_roofing
def total_cost : ℕ := remaining_roofing * cost_per_foot

-- The proof statement: 
theorem Mr_Caiden_payment : total_cost = 400 := 
by
  -- Proof omitted
  sorry

end Mr_Caiden_payment_l102_102067


namespace range_of_m_union_num_non_empty_proper_subsets_of_A_range_of_m_intersection_l102_102333

open Set

-- Definitions of sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Problem 1: Prove the range of m given A ∪ B = A
theorem range_of_m_union (m : ℝ) : (A ∪ B m = A) → m ∈ Iic 3 :=
by
  sorry

-- Problem 2: When x is an integer, find the number of non-empty proper subsets of A
theorem num_non_empty_proper_subsets_of_A : 
  (∃ (S : Set ℤ), ∀ x, x ∈ S → x ∈ A) → finite S → S.card - 1 = 254 :=
by
  sorry

-- Problem 3: Prove the range of m given A ∩ B = ∅
theorem range_of_m_intersection (m : ℝ) : (A ∩ B m = ∅) → m ∈ (Iio 2 ∪ Ioi 4) :=
by
  sorry

end range_of_m_union_num_non_empty_proper_subsets_of_A_range_of_m_intersection_l102_102333


namespace sun_city_population_greater_than_twice_roseville_l102_102442

-- Conditions
def willowdale_population : ℕ := 2000
def roseville_population : ℕ := 3 * willowdale_population - 500
def sun_city_population : ℕ := 12000

-- Theorem
theorem sun_city_population_greater_than_twice_roseville :
  sun_city_population = 2 * roseville_population + 1000 :=
by
  -- The proof is omitted as per the problem statement
  sorry

end sun_city_population_greater_than_twice_roseville_l102_102442


namespace minimum_weighings_to_find_genuine_l102_102949

-- Definitions from conditions:
constant A B C D E : Type -- coins
constant is_genuine : A → Prop
constant is_counterfeit : A → Prop
constant equal_weight : A → A → Prop

axiom three_genuine : ∃ a b c : A, is_genuine a ∧ is_genuine b ∧ is_genuine c
axiom two_counterfeit : ∃ d e : A, is_counterfeit d ∧ is_counterfeit e
axiom counterfeit_same_weight : ∀ d1 d2 : A, is_counterfeit d1 → is_counterfeit d2 → equal_weight d1 d2
axiom same_or_diff_weight : ∀ x y : A, equal_weight x y ∨ ¬ equal_weight x y

-- Theorem to prove:
theorem minimum_weighings_to_find_genuine : ∃ (n : ℕ), n ≤ 2 ∧ ∀ coins, (is_genuine ∨ is_counterfeit) coins → ∃ c : A, is_genuine c :=
by 
  sorry

end minimum_weighings_to_find_genuine_l102_102949


namespace length_of_crease_correct_l102_102930

noncomputable def length_of_crease (theta : ℝ) : ℝ := Real.sqrt (40 + 24 * Real.cos theta)

theorem length_of_crease_correct (theta : ℝ) : 
  length_of_crease theta = Real.sqrt (40 + 24 * Real.cos theta) := 
by 
  sorry

end length_of_crease_correct_l102_102930


namespace opposite_of_one_sixth_l102_102864

theorem opposite_of_one_sixth : (-(1 / 6) : ℚ) = -1 / 6 := 
by
  sorry

end opposite_of_one_sixth_l102_102864


namespace cos_18_polynomial_has_integer_coeffs_root_l102_102605

theorem cos_18_polynomial_has_integer_coeffs_root :
  ∃ x : ℝ, x = real.cos (real.pi / 10) ∧ (16 * x ^ 4 - 20 * x ^ 2 + 5 = 0) :=
by
  sorry

end cos_18_polynomial_has_integer_coeffs_root_l102_102605


namespace complete_square_solution_l102_102510

theorem complete_square_solution (x : ℝ) :
  (x^2 + 6 * x - 4 = 0) → ((x + 3)^2 = 13) :=
by
  sorry

end complete_square_solution_l102_102510


namespace radish_patch_size_l102_102541

theorem radish_patch_size (R P : ℕ) (h1 : P = 2 * R) (h2 : P / 6 = 5) : R = 15 := by
  sorry

end radish_patch_size_l102_102541


namespace minimum_distinct_numbers_l102_102803

theorem minimum_distinct_numbers (a : ℕ → ℕ) (h_pos : ∀ i, 1 ≤ i → a i > 0)
  (h_distinct_ratios : ∀ i j : ℕ, 1 ≤ i ∧ i < 2006 ∧ 1 ≤ j ∧ j < 2006 ∧ i ≠ j → a i / a (i + 1) ≠ a j / a (j + 1)) :
  ∃ (n : ℕ), n = 46 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2006 ∧ 1 ≤ j ∧ j ≤ i ∧ (a i = a j → i = j) :=
sorry

end minimum_distinct_numbers_l102_102803


namespace tracy_feeds_dogs_times_per_day_l102_102880

theorem tracy_feeds_dogs_times_per_day : 
  let cups_per_meal_per_dog := 1.5
  let dogs := 2
  let total_pounds_per_day := 4
  let cups_per_pound := 2.25
  (total_pounds_per_day * cups_per_pound) / (dogs * cups_per_meal_per_dog) = 3 :=
by
  sorry

end tracy_feeds_dogs_times_per_day_l102_102880


namespace feuerbach_circles_common_point_l102_102704

theorem feuerbach_circles_common_point (A B C D : Point) :
  ∃ P, P ∈ feuerbach_circle A B C ∧ P ∈ feuerbach_circle A B D ∧ P ∈ feuerbach_circle A C D ∧ P ∈ feuerbach_circle B C D :=
sorry

end feuerbach_circles_common_point_l102_102704


namespace profit_percentage_B_l102_102938

theorem profit_percentage_B (price_A price_C : ℝ) (profit_A_percentage : ℝ) : 
  price_A = 150 → 
  price_C = 225 → 
  profit_A_percentage = 20 →
  let price_B := price_A + (profit_A_percentage / 100 * price_A) in
  let profit_B := price_C - price_B in
  let profit_B_percentage := (profit_B / price_B) * 100 in
  profit_B_percentage = 25 := 
by
  intros
  simp only
  sorry

end profit_percentage_B_l102_102938


namespace solve_system_of_equations_l102_102440

-- Define the system of equations
def system_of_equations (a b c : ℝ) : Prop :=
  a + b + c = 0 ∧
  a^2 + b^2 + c^2 = 1 ∧
  a^3 + b^3 + c^3 = 4 * a * b * c

-- Solutions to the system
def is_solution (a b c : ℝ) : Prop :=
  (a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 ∧ c = 0) ∨
  (a = -sqrt 2 / 2 ∧ b = sqrt 2 / 2 ∧ c = 0) ∨
  (a = sqrt 2 / 2 ∧ b = 0 ∧ c = -sqrt 2 / 2) ∨
  (a = -sqrt 2 / 2 ∧ b = 0 ∧ c = sqrt 2 / 2) ∨
  (a = 0 ∧ b = sqrt 2 / 2 ∧ c = -sqrt 2 / 2) ∨
  (a = 0 ∧ b = -sqrt 2 / 2 ∧ c = sqrt 2 / 2)

-- The proof statement
theorem solve_system_of_equations (a b c : ℝ) :
  system_of_equations a b c → is_solution a b c :=
by sorry

end solve_system_of_equations_l102_102440


namespace correct_subtraction_result_l102_102899

theorem correct_subtraction_result (n : ℕ) (h : 40 / n = 5) : 20 - n = 12 := by
sorry

end correct_subtraction_result_l102_102899


namespace monotone_increasing_interval_of_tan_shifted_l102_102117

theorem monotone_increasing_interval_of_tan_shifted (k : ℤ) :
  ∃ I : set ℝ, I = set.Ioo (k * π - 3 * π / 4) (k * π + π / 4) ∧ 
  ∀ x ∈ I, ∃ y ∈ I, (y = tan (x + π / 4)) :=
by
  sorry

end monotone_increasing_interval_of_tan_shifted_l102_102117


namespace vova_winning_strategy_l102_102428

-- Define the conditions of the game
def game_conditions := 
  ∃ (board : Matrix Bool (Fin 3) (Fin 101)),
    board 1 50 = false ∧  -- The central cell (second row, 51st column) is initially crossed out
    (∀ (move : Fin 3 → Fin 101 → Bool),  -- Move is a function that marks cells on the board
      (∃ i j, move i j = false) →  -- Each move crosses out at least one new cell
      (∃ (diagonal_length : Fin 3),  -- The diagonal can have lengths 1, 2, or 3
        ∀ (i j : Fin 3) (k : Fin diagonal_length),
          move (i + k) (j + k) = false → move i j = false)) ∧
    (∀ number_of_moves,  -- Total number of moves
      number_of_moves = (3 * 101) - 1 →  -- Initial cell is crossed out, so 302 remaining cells
      even number_of_moves)

-- Formulate the theorem to prove
theorem vova_winning_strategy : game_conditions → (∃ strategy, strategy = "Vova wins") :=
begin
  sorry
end

end vova_winning_strategy_l102_102428


namespace B_completion_time_l102_102174

theorem B_completion_time (A_days : ℕ) (A_efficiency_multiple : ℝ) (B_days_correct : ℝ) :
  A_days = 15 →
  A_efficiency_multiple = 1.8 →
  B_days_correct = 4 + 1 / 6 →
  B_days_correct = 25 / 6 :=
sorry

end B_completion_time_l102_102174


namespace no_valid_n_l102_102285

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n.minFac

theorem no_valid_n (n : ℕ) (h1 : n > 1)
  (h2 : is_prime (greatest_prime_factor n))
  (h3 : greatest_prime_factor n = Nat.sqrt n)
  (h4 : is_prime (greatest_prime_factor (n + 36)))
  (h5 : greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) :
  false :=
sorry

end no_valid_n_l102_102285


namespace collinear_circumcenters_l102_102526

-- Define the geometrical setup and needed points
variable {Ω : Type*} [metric_space Ω] [inner_product_space ℝ Ω] [normed_space ℝ Ω]
variables (O I A B C : Ω)
variable (tangent_to_circle : ∀ P, P = A ∨ P = B → ∃ (T : set Ω), is_tangent T Ω ∧ P ∈ T)
variable (I_center : is_center I Ω)
variable (C_minor_arc : minor_arc A B C Ω ∧ C ≠ midpoint_arc A B Ω)

-- Define lines and their intersections
variables (D E : Ω)
variable (AC : line A C)
variable (OB : line O B)
variable (BC : line B C)
variable (OA : line O A)

noncomputable def point_D := intersection_point AC OB
noncomputable def point_E := intersection_point BC OA

-- Circumcenters
noncomputable def circumcenter_ACE := circumcenter (triangle.mk A C point_E)
noncomputable def circumcenter_BCD := circumcenter (triangle.mk B C point_D)
noncomputable def circumcenter_OCI := circumcenter (triangle.mk O C I)

-- Collinearity of circumcenters
theorem collinear_circumcenters :
  collinear [circumcenter_ACE, circumcenter_BCD, circumcenter_OCI] :=
sorry

end collinear_circumcenters_l102_102526


namespace side_length_of_largest_square_l102_102194

theorem side_length_of_largest_square (S : ℝ) 
  (h1 : 2 * (S / 2)^2 + 2 * (S / 4)^2 = 810) : S = 36 :=
by
  -- proof steps go here
  sorry

end side_length_of_largest_square_l102_102194


namespace length_of_picture_frame_l102_102363

theorem length_of_picture_frame (P W : ℕ) (hP : P = 30) (hW : W = 10) : ∃ L : ℕ, 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end length_of_picture_frame_l102_102363


namespace Alibaba_gems_selection_l102_102511

theorem Alibaba_gems_selection :
  ∃ (R B G : ℕ), R + B + G = 10 ∧ 2 ≤ R ∧ 2 ≤ B ∧ G ≤ 3 ∧ R ≤ 9 ∧ B ≤ 5 ∧ G ≤ 6 ∧ 
    (nat.choose (R, 2, 9 + 1 - R) * nat.choose (B, 2, 5 + 1 - B) * nat.choose (G, 0, 6 + 1 - G)) = 22 :=
by
  sorry

end Alibaba_gems_selection_l102_102511


namespace player_A_wins_l102_102490

/-!
# Lean 4 statement for the game problem

Given:
1. Player A chooses 1000 odd primes (not necessarily different), all congruent to 1 mod 4.
2. Player B then selects 500 of these primes and writes them on the blackboard.

In each turn:
- A player chooses a positive integer n, erases some primes p1, p2, ..., pn from the blackboard,
  and writes all the prime factors of p1 * p2 * ... * pn - 2 instead.
- Player A starts, and the player whose move leaves the blackboard empty loses the game.

Prove:
Player A has a winning strategy.
-/

theorem player_A_wins (primes_A : Multiset ℕ) (h1 : primes_A.card = 1000)
  (h2 : ∀ p ∈ primes_A, Prime p ∧ p % 4 = 1)
  (primes_B : Multiset ℕ) (h3 : primes_B.card = 500) (h4 : primes_B ⊆ primes_A) :
  ∃ strategy : (Multiset ℕ → ℕ) → Prop, (∀ (move : Multiset ℕ) (h : move ⊆ primes_B), strategy move) ∧
  (strategy (to_finset primes_B) = ∅) →
  (strategy (to_finset primes_B) = ∅ → ¬ winning_move for player B) :=
sorry

end player_A_wins_l102_102490


namespace bailey_credit_cards_l102_102578

theorem bailey_credit_cards (dog_treats : ℕ) (chew_toys : ℕ) (rawhide_bones : ℕ) (items_per_charge : ℕ) (total_items : ℕ) (credit_cards : ℕ)
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : rawhide_bones = 10)
  (h4 : items_per_charge = 5)
  (h5 : total_items = dog_treats + chew_toys + rawhide_bones)
  (h6 : credit_cards = total_items / items_per_charge) :
  credit_cards = 4 :=
by
  sorry

end bailey_credit_cards_l102_102578


namespace remainder_div_1234567_256_l102_102968

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l102_102968


namespace negation_correct_l102_102457

variable (x : Real)

def original_proposition : Prop :=
  x > 0 → x^2 > 0

def negation_proposition : Prop :=
  x ≤ 0 → x^2 ≤ 0

theorem negation_correct :
  ¬ original_proposition x = negation_proposition x :=
by 
  sorry

end negation_correct_l102_102457


namespace sum_of_digits_of_N_l102_102964

noncomputable def N : ℕ := (10^20 - 1)^2 / 81

theorem sum_of_digits_of_N : 
  let digits_sum := (List.range 20).sum + (List.range 20).reverse.sum in
  digits_sum = 400 :=
by sorry

end sum_of_digits_of_N_l102_102964


namespace perp_AK_AB_l102_102372

variables (A B C H K : Point)
variables (angle : Point → Point → Point → ℝ)
variables (altitude : Point → Point → Point → Prop)
variables (perpendicular : Point → Point → Point → Prop)
variables (acute_triangle : Point → Point → Point → Prop)
variables (congruent_triangles : Point → Point → Point → Point → Point → Point → Prop)

-- Conditions
variables (acute_ABC : acute_triangle A B C)
variables (BH_altitude : altitude B H C)
variables (AB_eq_CH : dist A B = dist C H)
variables (angle_BKC_eq_BCK : angle B K C = angle B C K)
variables (angle_ABK_eq_ACB : angle A B K = angle A C B)

-- To Prove
theorem perp_AK_AB (h1 : acute_ABC) (h2 : BH_altitude) 
(h3 : AB_eq_CH) (h4 : angle_BKC_eq_BCK) (h5 : angle_ABK_eq_ACB) : perpendicular A K B := 
sorry

end perp_AK_AB_l102_102372


namespace min_max_values_of_f_l102_102851

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x >= - (3 * Real.pi / 2)) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = - (3 * Real.pi / 2)) ∧
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x <= Real.pi / 2 + 2) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = Real.pi / 2 + 2) :=
by {
  -- Proof omitted
  sorry
}

end min_max_values_of_f_l102_102851


namespace speed_back_home_l102_102204

-- Define the necessary variables and conditions
variables (D : ℝ) (T₁ T₂ S : ℝ)

-- Given conditions
def condition1 : Prop := T₁ = D / 20
def condition2 : Prop := T₂ = D / S
def condition3 : Prop := (2 * D) / (T₁ + T₂) = 24

-- The theorem that needs to be proven
theorem speed_back_home (hD : D > 0) :
  condition1 ∧ condition2 ∧ condition3 → S = 30 :=
by
  intro ⟨h1, h2, h3⟩
  -- The proof part is omitted as per the problem statement
  sorry

end speed_back_home_l102_102204


namespace distance_traveled_l102_102160

theorem distance_traveled (speed time : ℕ) (h_speed : speed = 25) (h_time : time = 5) :
  speed * time = 125 := by
    -- Given conditions as assumptions
    rw [h_speed, h_time]
    -- Evaluate the left-hand side
    exact rfl

end distance_traveled_l102_102160


namespace MN_bisects_BC_iff_MN_perp_EF_l102_102959

-- Definition of the problem context
variables (A B C D E F G: Point) (AB AC BC: Line) 
  (ACDE ABGF: Square) (M N: Point) (MN EF: Line)

-- Definition of geometric conditions
axiom ACDE_is_square : is_square ACDE
axiom ABGF_is_square : is_square ABGF
axiom MN_passes_through_A : passes_through MN A
axiom M_is_on_BC : lies_on M BC
axiom N_is_on_EF : lies_on N EF

-- Theorem statement
theorem MN_bisects_BC_iff_MN_perp_EF : 
  (bisects MN BC M) ↔ (perpendicular MN EF) :=
sorry

end MN_bisects_BC_iff_MN_perp_EF_l102_102959


namespace min_max_f_on_interval_l102_102854

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_f_on_interval :
  ∃ min max, min = - (3 * Real.pi) / 2 ∧ max = (Real.pi / 2) + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min ∧ f x ≤ max) :=
sorry

end min_max_f_on_interval_l102_102854


namespace ratio_unit_price_brand_x_to_brand_y_l102_102231

-- Definitions based on the conditions in the problem
def volume_brand_y (v : ℝ) := v
def price_brand_y (p : ℝ) := p
def volume_brand_x (v : ℝ) := 1.3 * v
def price_brand_x (p : ℝ) := 0.85 * p
noncomputable def unit_price (volume : ℝ) (price : ℝ) := price / volume

-- Theorems to prove the ratio of unit price of Brand X to Brand Y is 17/26
theorem ratio_unit_price_brand_x_to_brand_y (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0) : 
  (unit_price (volume_brand_x v) (price_brand_x p)) / (unit_price (volume_brand_y v) (price_brand_y p)) = 17 / 26 := by
  sorry

end ratio_unit_price_brand_x_to_brand_y_l102_102231


namespace smallest_a_undefined_mod_39_70_l102_102500

def is_undefined_mod (a n : ℕ) : Prop := Nat.gcd a n ≠ 1

theorem smallest_a_undefined_mod_39_70 :
  ∃ a : ℕ, a > 0 ∧ is_undefined_mod a 39 ∧ is_undefined_mod a 70 ∧
  (∀ b : ℕ, b > 0 ∧ is_undefined_mod b 39 ∧ is_undefined_mod b 70 → a ≤ b) :=
begin
  use 21,
  split,
  { norm_num, },
  split,
  { unfold is_undefined_mod,
    norm_num, },
  split,
  { unfold is_undefined_mod,
    norm_num, },
  { intros b hb_pos hb_39 hb_70,
    cases hb_39,
    cases hb_70,
    have h_a39 : Nat.dvd 39 b ∨ Nat.dvd 70 b,
    { sorry }, -- detailed arithmetic check
    have h_a21 : Nat.dvd b 21,
    { sorry }, -- detailed arithmetic check
    exact h_a21,
  }
end

end smallest_a_undefined_mod_39_70_l102_102500


namespace volume_of_tetrahedron_l102_102296

-- Given conditions
variables {A B C S H: Type}
variables (is_equilateral_triangle : ABC.equilateral)
variables (A_on_SBC : A ∈ S \and \ S ∈ set.univ H)
variables (H_is_orthocenter_SBC : H.orthocenter \triangle SBC)
variables (dihedral_angle_H_AB_BC : ∠(H-AB) (BC) = 30°)
variables (SA_length : SA = 2 * sqrt 3)

-- Concluding statement with the known volume
theorem volume_of_tetrahedron (is_equilateral_triangle : ABC.equilateral)
  (A_on_SBC : A ∈ S \and \ S ∈ set.univ H)
  (H_is_orthocenter_SBC : H.orthocenter \triangle SBC)
  (dihedral_angle_H_AB_BC : ∠(H-AB) (BC) = 30°)
  (SA_length : SA = 2 * sqrt 3) :
  volume S ABC = (9 / 4) * sqrt 3 :=
by
  sorry

end volume_of_tetrahedron_l102_102296


namespace angle_BMC_not_obtuse_angle_BAC_is_120_l102_102447

theorem angle_BMC_not_obtuse (α β γ : ℝ) (h : α + β + γ = 180) :
  0 < 90 - α / 2 ∧ 90 - α / 2 < 90 :=
sorry

theorem angle_BAC_is_120 (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : 90 - α / 2 = α / 2) : α = 120 :=
sorry

end angle_BMC_not_obtuse_angle_BAC_is_120_l102_102447


namespace amara_clothing_remaining_l102_102564

theorem amara_clothing_remaining :
  ∀ (initial donation_one donation_factor discard : ℕ),
    initial = 100 →
    donation_one = 5 →
    donation_factor = 3 →
    discard = 15 →
    let total_donated := donation_one + (donation_factor * donation_one) in
    let remaining_after_donation := initial - total_donated in
    let final_remaining := remaining_after_donation - discard in
    final_remaining = 65 := 
by
  sorry

end amara_clothing_remaining_l102_102564


namespace period_of_cos_3x_l102_102148

-- Define the function
def cos_3x (x : ℝ) : ℝ := Real.cos (3 * x)

-- Define the standard period of the cosine function
def standard_cos_period : ℝ := 2 * Real.pi

-- Theorem statement
theorem period_of_cos_3x : ∃ T : ℝ, (∀ x : ℝ, cos_3x (x + T) = cos_3x x) ∧ T = standard_cos_period / 3 :=
by
  sorry

end period_of_cos_3x_l102_102148


namespace symmetric_line_b_value_l102_102679

theorem symmetric_line_b_value (b : ℝ) : 
  (∃ l1 l2 : ℝ × ℝ → Prop, 
    (∀ (x y : ℝ), l1 (x, y) ↔ y = -2 * x + b) ∧ 
    (∃ p2 : ℝ × ℝ, p2 = (1, 6) ∧ l2 p2) ∧
    l2 (-1, 6) ∧ 
    (∀ (x y : ℝ), l1 (x, y) ↔ l2 (-x, y))) →
  b = 4 := 
by
  sorry

end symmetric_line_b_value_l102_102679


namespace lambda_range_l102_102341

-- Definitions and parameters
variables (m n θ λ : ℝ)
def a := (m, n)
def b := (Real.cos θ, Real.sin θ)

-- Condition: Magnitude of a is 4 times the magnitude of b
def mag_condition : Prop := Real.sqrt (m^2 + n^2) = 4

-- Dot product of a and b
def dot_product : ℝ := m * Real.cos θ + n * Real.sin θ

-- The proof goal
theorem lambda_range (h : mag_condition) : dot_product < λ^2 ↔ (λ > 2 ∨ λ < -2) :=
sorry

end lambda_range_l102_102341


namespace part1_part2_l102_102177

-- Part (I)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, 3 * x - abs (-2 * x + 1) ≥ a ↔ 2 ≤ x) → a = 3 :=
by
  sorry

-- Part (II)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x - abs (x - a) ≤ 1)) → (a ≤ 1 ∨ 3 ≤ a) :=
by
  sorry

end part1_part2_l102_102177


namespace three_digit_solution_count_l102_102711

theorem three_digit_solution_count :
  ({ x : ℕ | x >= 100 ∧ x <= 999 ∧ (4573 * x + 502) % 23 = 1307 % 23 }.finite.toFinset.card = 39) := sorry

end three_digit_solution_count_l102_102711


namespace find_C_coordinates_l102_102010

noncomputable def is_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

noncomputable def oa_length (A : ℝ × ℝ) : ℝ := real.sqrt ((-1)^2 + 0^2)

theorem find_C_coordinates :
  ∀ (A B C : ℝ × ℝ),
  A = (-1, 0) →
  B = (-3, -4) →
  C.2 = -4 →
  (is_parallel (C.1 - B.1, C.2 - B.2) (A.1, A.2)) →
  real.sqrt (((C.1 - B.1) * (C.1 - B.1)) + ((C.2 - B.2) * (C.2 - B.2))) = 4 * oa_length A →
  (C = (-7, -4) ∨ C = (1, -4)) :=
begin
  intros A B C hA hB hC hParallel hDist,
  sorry
end

end find_C_coordinates_l102_102010


namespace oxygen_atoms_in_compound_l102_102191

theorem oxygen_atoms_in_compound (K_weight Br_weight O_weight molecular_weight : ℕ) 
    (hK : K_weight = 39) (hBr : Br_weight = 80) (hO : O_weight = 16) (hMW : molecular_weight = 168) 
    (n : ℕ) :
    168 = 39 + 80 + n * 16 → n = 3 :=
by
  intros h
  sorry

end oxygen_atoms_in_compound_l102_102191


namespace gcd_in_range_l102_102835

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end gcd_in_range_l102_102835


namespace maxwell_meets_brad_time_l102_102906

-- Definitions of the initial conditions
def distance_between_homes : ℝ := 34
def maxwell_speed : ℝ := 4
def brad_speed : ℝ := 6
def additional_time_for_maxwell : ℝ := 1

-- The theorem to prove the total time Maxwell walks before he meets Brad
theorem maxwell_meets_brad_time :
  ∃ t : ℝ, (maxwell_speed * (t + additional_time_for_maxwell) + brad_speed * t = distance_between_homes) ∧ t + additional_time_for_maxwell = 4 :=
begin
  sorry
end

end maxwell_meets_brad_time_l102_102906


namespace alice_telephone_numbers_l102_102222

theorem alice_telephone_numbers :
  let usable_digits := {2, 3, 4, 5, 6, 7, 8, 9}
  let even_digits := {2, 4, 6, 8}
  let odd_digits := {3, 5, 7, 9}
  let num_pairs := 4 * 4
  let remaining_combinations := Nat.choose 6 3
  let total_combinations := num_pairs * remaining_combinations
  total_combinations = 320 := sorry

end alice_telephone_numbers_l102_102222


namespace sequences_of_length_15_l102_102983

def odd_runs_of_A_even_runs_of_B (n : ℕ) : ℕ :=
  (if n = 1 then 1 else 0) + (if n = 2 then 1 else 0)

theorem sequences_of_length_15 : 
  odd_runs_of_A_even_runs_of_B 15 = 47260 :=
  sorry

end sequences_of_length_15_l102_102983


namespace circle_radius_decrease_l102_102359

theorem circle_radius_decrease (r r' : ℝ) (h1 : r > 0)
  (h2 : 0.58 * (real.pi * r^2) = real.pi * r'^2) : 
  (r' = r * real.sqrt 0.58) →
  (1 - real.sqrt 0.58) * 100 ≈ 23.84 :=
by
  sorry

end circle_radius_decrease_l102_102359


namespace cartesian_equations_minimum_distance_point_l102_102532

section CoordinateSystems

variable (θ ϕ : ℝ)

/-- Definitions of the equations in polar coordinates --/
def polar_equation_line := ∀ ρ θ, ρ * (Math.cos θ + 2 * Math.sin θ) = 10
def parametric_curve := ∀ θ, (3 * Math.cos θ, 2 * Math.sin θ)

/-- Cartesian equations derived from the provided conditions --/
def cartesian_equation_line := ∀ x y, x + 2*y - 10 = 0
def cartesian_equation_curve := ∀ x y, (x^2 / 9 + y^2 / 4 = 1)

/-- Problem 1: Verifying Cartesian Equations --/
theorem cartesian_equations :
    (∃ ρ θ, polar_equation_line ρ θ) →
    (∀ θ, parametric_curve θ) →
    cartesian_equation_line ∧ cartesian_equation_curve := 
by sorry

/-- Problem 2: Minimum distance from point M on C to line l --/
theorem minimum_distance_point :
    (∀ θ, parametric_curve θ) →
    (3 * Math.cos ϕ, 2 * Math.sin ϕ) = (9/5, 8/5) →
    (dist (9/5, 8/5) (λ xy, x + 2 * y - 10 = 0) = sqrt 5) := 
by sorry

end CoordinateSystems

end cartesian_equations_minimum_distance_point_l102_102532


namespace slope_l3_l102_102058

noncomputable def pointA : ℝ × ℝ := (2, -2)
noncomputable def pointB : ℝ × ℝ := (8 / 5, 2)
noncomputable def pointC : ℝ × ℝ := (19 / 5, 2)

noncomputable def line1 (x y : ℝ) := 5 * x - 3 * y = 2
noncomputable def line2 (y : ℝ) := y = 2
noncomputable def l1_passes_through_A := line1 2 (-2)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(fst B - fst A) * (snd C - snd A) - 
         (fst C - fst A) * (snd B - snd A)|

axiom meet_at_pointB : line1 (fst pointB) (snd pointB) ∧ line2 (snd pointB)
axiom positive_slope : 0 < fst pointC - fst pointA
axiom triangle_area_ABC : triangle_area pointA pointB pointC = 5

theorem slope_l3 : (snd pointC - snd pointA) / (fst pointC - fst pointA) = 20 / 9 :=
by
  sorry

end slope_l3_l102_102058


namespace correct_car_selection_l102_102228

-- Define the production volumes
def production_emgrand : ℕ := 1600
def production_king_kong : ℕ := 6000
def production_freedom_ship : ℕ := 2000

-- Define the total number of cars produced
def total_production : ℕ := production_emgrand + production_king_kong + production_freedom_ship

-- Define the number of cars selected for inspection
def cars_selected_for_inspection : ℕ := 48

-- Calculate the sampling ratio
def sampling_ratio : ℚ := cars_selected_for_inspection / total_production

-- Define the expected number of cars to be selected from each model using the sampling ratio
def cars_selected_emgrand : ℚ := sampling_ratio * production_emgrand
def cars_selected_king_kong : ℚ := sampling_ratio * production_king_kong
def cars_selected_freedom_ship : ℚ := sampling_ratio * production_freedom_ship

theorem correct_car_selection :
  cars_selected_emgrand = 8 ∧ cars_selected_king_kong = 30 ∧ cars_selected_freedom_ship = 10 := by
  sorry

end correct_car_selection_l102_102228


namespace geom_bio_students_difference_l102_102735

theorem geom_bio_students_difference (total_students geometry_students biology_students : ℕ)
  (h1 : total_students = 232) (h2 : geometry_students = 144) (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students,
      min_overlap := geometry_students + biology_students - total_students
  in max_overlap - min_overlap = 88 :=
by
  sorry

end geom_bio_students_difference_l102_102735


namespace gina_netflix_time_l102_102644

theorem gina_netflix_time (sister_shows : ℕ) (show_length : ℕ) (ratio : ℕ) (sister_ratio : ℕ) :
sister_shows = 24 →
show_length = 50 →
ratio = 3 →
sister_ratio = 1 →
(ratio * sister_shows * show_length = 3600) :=
begin
  intros hs hl hr hsr,
  rw hs,
  rw hl,
  rw hr,
  rw hsr,
  norm_num,
  sorry
end

end gina_netflix_time_l102_102644


namespace tangent_line_at_1_l102_102111

noncomputable def f (x : ℝ) (f3_1 : ℝ) : ℝ := (f3_1 / real.exp 1) * real.exp x - (f3_1 / real.exp 1) * x + (1/2 : ℝ) * x^2

theorem tangent_line_at_1 (f3_1 : ℝ) :
  let f := f f3_1 in
  let slope_at_1 := real.exp 1 in
  let point_1 := (1 : ℝ, f 1) in
  let tangent_line := λ x, slope_at_1 * x - (1/2 : ℝ) in
  tangent_line 1 = f 1 :=
by 
  sorry

end tangent_line_at_1_l102_102111


namespace point_on_parabola_l102_102208

noncomputable def P : ℝ × ℝ := (10 * Real.sqrt 2, 50)

def parabola_vertex := (0, 0) : ℝ × ℝ
def parabola_focus := (0, 1) : ℝ × ℝ

def is_on_parabola (P : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), P = (x, y) ∧ Real.sqrt (x^2 + (y - 1)^2) = y + 1

theorem point_on_parabola (P = (10 * Real.sqrt 2, 50)) (vertex = (0,0)) (focus = (0, 1)) (PF = 51) :
  is_on_parabola P :=
sorry

end point_on_parabola_l102_102208


namespace letter_arrangement_3x2_l102_102957

theorem letter_arrangement_3x2 : ∃ (n : ℕ), n = 12 ↔
  ∃ (arrangement : fin 3 → fin 2 → char), 
    (∀ i : fin 3, arrangement i 0 ≠ arrangement i 1) ∧
    (∀ j : fin 2, arrangement 0 j ≠ arrangement 1 j) ∧
    (∀ j : fin 2, arrangement 0 j ≠ arrangement 2 j) ∧
    (∀ j : fin 2, arrangement 1 j ≠ arrangement 2 j) :=
by
  sorry

end letter_arrangement_3x2_l102_102957


namespace union_complement_equals_set_l102_102336

universe u

variable {I A B : Set ℕ}

def universal_set : Set ℕ := {0, 1, 2, 3, 4}
def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {2, 3, 4}
def complement_B : Set ℕ := { x ∈ universal_set | x ∉ set_B }

theorem union_complement_equals_set :
  set_A ∪ complement_B = {0, 1, 2} := by
  sorry

end union_complement_equals_set_l102_102336


namespace binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l102_102244

def binary_to_decimal (b : ℕ) : ℕ :=
  32 + 0 + 8 + 4 + 2 + 1 -- Calculated manually for simplicity

def decimal_to_octal (d : ℕ) : ℕ :=
  (5 * 10) + 7 -- Manually converting decimal 47 to octal 57 for simplicity

theorem binary_101111_to_decimal_is_47 : binary_to_decimal 0b101111 = 47 := 
by sorry

theorem decimal_47_to_octal_is_57 : decimal_to_octal 47 = 57 := 
by sorry

end binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l102_102244


namespace number_of_zeros_of_f_l102_102861

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin (2 * x)

theorem number_of_zeros_of_f : (∃ l : List ℝ, (∀ x ∈ l, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧ l.length = 4) := 
by
  sorry

end number_of_zeros_of_f_l102_102861


namespace probability_f_le_0_is_3_over_4_l102_102675

def f (x : ℝ) : ℝ := Real.sin (x - 1)

def probability_f_le_0 : ℚ :=
  let p_values := {1, 3, 5, 7}
  let valid_values := p_values.filter (λ p, f p ≤ 0)
  valid_values.card / p_values.card

theorem probability_f_le_0_is_3_over_4 :
  probability_f_le_0 = 3 / 4 :=
by 
  sorry

end probability_f_le_0_is_3_over_4_l102_102675


namespace books_in_bargain_bin_now_l102_102185

-- Definitions based on conditions
def initial_books : ℕ := 4
def books_sold : ℕ := 3
def additional_percentage : ℝ := 0.5
def new_shipment : ℕ := 10

-- Function to calculate the number of book after selling, adding percentage, and receiving shipment
def books_in_bin_after_operations (initial books_sold new_shipment : ℕ) (additional_percentage : ℝ) : ℕ :=
  let books_after_sale := initial - books_sold
  let additional_books := (additional_percentage * books_after_sale).to_nat
  books_after_sale + additional_books + new_shipment

-- Theorem stating the final number of books in the bargain bin
theorem books_in_bargain_bin_now : books_in_bin_after_operations initial_books books_sold new_shipment additional_percentage = 12 := by
  sorry

end books_in_bargain_bin_now_l102_102185


namespace tan_A_sub_tan_B_div_tan_A_add_tan_B_eq_c_sub_b_div_c_l102_102650

theorem tan_A_sub_tan_B_div_tan_A_add_tan_B_eq_c_sub_b_div_c
  (A B C : Triangle)
  (angle_A : A.angle = 60)
  (BC a : ℝ)
  (CA b : ℝ)
  (AB c : ℝ) 
  (h_BC : BC = a)
  (h_CA : CA = b)
  (h_AB : AB = c) :
  (tan A - tan B) / (tan A + tan B) = (c - b) / c :=
by
  sorry

end tan_A_sub_tan_B_div_tan_A_add_tan_B_eq_c_sub_b_div_c_l102_102650


namespace find_angle_C_l102_102420

-- Definitions and conditions
variables (l m : Type) [IsParallel l m]
variables (CD : Type) [IsParallel CD l] [IsParallel CD m]
def angle_A : ℝ := 110
def angle_B : ℝ := 140

-- Proof statement
theorem find_angle_C (h1 : Parallel l m) (h2 : Parallel CD l) (h3 : Parallel CD m) 
                     (hA : angle_A = 110) (hB : angle_B = 140) : mangleC = 70 :=
by sorry

end find_angle_C_l102_102420


namespace remaining_fruits_total_l102_102224

theorem remaining_fruits_total :
  let apples := 180 in
  let plums := apples / 3 in
  let pears := 2 * plums in
  let cherries := 4 * apples in
  let apples_picked := (13 / 15 : ℚ) * apples in
  let plums_picked := (5 / 6 : ℚ) * plums in
  let pears_picked := (3 / 4 : ℚ) * pears in
  let cherries_picked := (37 / 50 : ℚ) * cherries in
  let apples_remaining := apples - apples_picked in
  let plums_remaining := plums - plums_picked in
  let pears_remaining := pears - pears_picked in
  let cherries_remaining := cherries - cherries_picked in
  apples_remaining + plums_remaining + pears_remaining + cherries_remaining = 251 :=
by
  sorry

end remaining_fruits_total_l102_102224


namespace exists_nat_no_perfect_square_after_complications_l102_102057

theorem exists_nat_no_perfect_square_after_complications :
  ∃ n : ℕ, ∀ (complications : List (ℕ → ℕ)), 
    (complications.length ≤ 100) →
    ¬(∃ m : ℕ, is_perfect_square (apply_complications n complications)) :=
by
  sorry

-- Definitions for the sake of completeness within the theorem context.
def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

def apply_complications (n : ℕ) (complications : List (ℕ → ℕ)) : ℕ :=
  complications.foldl (fun x f => f x) n

end exists_nat_no_perfect_square_after_complications_l102_102057


namespace last_two_nonzero_digits_of_factorial_l102_102860

theorem last_two_nonzero_digits_of_factorial (n : ℕ) (h : n = 100) : 
  let num_of_factors_5 := (100 / 5) + (100 / 25) + (100 / 125)
  let num_of_factors_10 := num_of_factors_5
  let N := 100! / 10^num_of_factors_10
  (N % 100) = 0 :=
by
  sorry

end last_two_nonzero_digits_of_factorial_l102_102860


namespace lisa_photos_calculation_l102_102791

-- Definitions based on conditions
variable (k : ℕ) (claire_photos : ℕ) (lisa_photos : ℕ)
variable (claire_taken : claire_photos = 10)
variable (lisa_taken : lisa_photos = k * claire_photos)

-- Theorem we need to prove
theorem lisa_photos_calculation (claire_photos = 10) : lisa_photos = k * 10 :=
by
  -- sorry means proof skipped
  sorry

end lisa_photos_calculation_l102_102791


namespace crayons_per_unit_l102_102389

theorem crayons_per_unit :
  ∀ (units : ℕ) (cost_per_crayon : ℕ) (total_cost : ℕ),
    units = 4 →
    cost_per_crayon = 2 →
    total_cost = 48 →
    (total_cost / cost_per_crayon) / units = 6 :=
by
  intros units cost_per_crayon total_cost h_units h_cost_per_crayon h_total_cost
  sorry

end crayons_per_unit_l102_102389


namespace new_stats_l102_102656

open Real

noncomputable def initial_sample_size : ℕ := 8
noncomputable def initial_mean : ℝ := 5
noncomputable def initial_variance : ℝ := 2
noncomputable def new_data_point : ℝ := 5
noncomputable def new_sample_size : ℕ := 9
noncomputable def new_mean (old_size : ℕ) (old_mean : ℝ) (new_point : ℝ): ℝ :=
  (old_size * old_mean + new_point) / (old_size + 1)
noncomputable def new_variance (old_size : ℕ) (old_variance : ℝ) (old_mean: ℝ) (new_point : ℝ): ℝ :=
  (old_size * old_variance + (new_point - old_mean)^2) / (old_size + 1)

theorem new_stats:
  (new_mean initial_sample_size initial_mean new_data_point = 5) ∧
  (new_variance initial_sample_size initial_variance initial_mean new_data_point < 2) :=
begin
  sorry
end

end new_stats_l102_102656


namespace find_intersection_points_l102_102331

noncomputable def C1_parametric (t : ℝ) : ℝ × ℝ :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

def C2_polar (θ : ℝ) : ℝ := 2 * Real.sin θ

def C1_polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 8 * ρ * Real.cos θ - 10 * ρ * Real.sin θ + 16 = 0

def C2_cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

theorem find_intersection_points :
  ∃ ρ θ, ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ C1_polar_equation ρ θ ∧ C2_polar θ = ρ ∧
    ((ρ = Real.sqrt 2 ∧ θ = Real.pi / 4) ∨ (ρ = 2 ∧ θ = Real.pi / 2)) :=
  sorry

end find_intersection_points_l102_102331


namespace no_1968_classes_l102_102752

theorem no_1968_classes :
  ∀ (classes : Finset (Finset ℕ)), 
    (∀ n ∈ classes, n.nonempty) →
    classes.card = 1968 →
    (∀ (m n : ℕ), (∃ p q : ℕ, (m = p*100 + q) ∨ (m = p*10 + q)) → 
    (∃ c ∈ classes, n ∈ c ∧ m ∈ c)) →
  False :=
begin
  intros classes nonempty_classes card_1968 transform_preservation,
  -- Since we don't need to prove it, we leave it as sorry
  sorry
end

end no_1968_classes_l102_102752


namespace angles_of_DEF_are_60_degrees_l102_102456

theorem angles_of_DEF_are_60_degrees
  (A B C D E F : Type)
  (h1 : triangle A B C)
  (h2 : is_equilateral A B C)
  (h3 : is_midpoint D B C)
  (h4 : is_midpoint E C A)
  (h5 : is_midpoint F A B)
  (h6 : triangle D E F) :
  angles_are_equal_to_60 D E F := 
sorry

end angles_of_DEF_are_60_degrees_l102_102456


namespace simplified_expression_is_one_l102_102233

-- Define the specific mathematical expressions
def expr1 := -1 ^ 2023
def expr2 := (-2) ^ 3
def expr3 := (-2) * (-3)

-- Construct the full expression
def full_expr := expr1 - expr2 - expr3

-- State the theorem that this full expression equals 1
theorem simplified_expression_is_one : full_expr = 1 := by
  sorry

end simplified_expression_is_one_l102_102233


namespace variables_related_with_90_confidence_l102_102307

noncomputable def chi_squared_value : ℝ := 3.528

def df (R C : ℕ) : ℕ := (R - 1) * (C - 1)

def is_related_at_confidence_level (chi_squared : ℝ) (df : ℕ) (confidence_level : ℝ) : Prop :=
  let critical_value := if confidence_level = 0.90 then 2.706 
                        else if confidence_level = 0.95 then 3.841
                        else if confidence_level = 0.99 then 6.635
                        else 0
  in chi_squared > critical_value

theorem variables_related_with_90_confidence :
  is_related_at_confidence_level chi_squared_value (df 2 2) 0.90 :=
by
  sorry

end variables_related_with_90_confidence_l102_102307


namespace rhombus_perimeter_area_l102_102828

theorem rhombus_perimeter_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (right_angle : ∀ (x : ℝ), x = d1 / 2 ∧ x = d2 / 2 → x * x + x * x = (d1 / 2)^2 + (d2 / 2)^2) : 
  ∃ (P A : ℝ), P = 52 ∧ A = 120 :=
by
  sorry

end rhombus_perimeter_area_l102_102828


namespace polynomial_integer_root_l102_102459

theorem polynomial_integer_root 
  (b c : ℚ) (p : Polynomial ℚ) 
  (h_poly : p = Polynomial.C c + Polynomial.X ^ 3 + Polynomial.C b * Polynomial.X)
  (h_root1 : p.eval (5 - real.sqrt 21) = 0) :
  ∃ r : ℤ, p.eval (r : ℚ) = 0 ∧ r = -10 := 
sorry

end polynomial_integer_root_l102_102459


namespace cost_of_jeans_l102_102180

    variable (J S : ℝ)

    def condition1 := 3 * J + 6 * S = 104.25
    def condition2 := 4 * J + 5 * S = 112.15

    theorem cost_of_jeans (h1 : condition1 J S) (h2 : condition2 J S) : J = 16.85 := by
      sorry
    
end cost_of_jeans_l102_102180


namespace circles_intersect_if_and_only_if_l102_102311

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6 * x - 8 * y - 11 = 0) ↔ (1 < m ∧ m < 121) :=
by
  sorry

end circles_intersect_if_and_only_if_l102_102311


namespace total_cost_of_taxi_ride_downtown_l102_102488

def uberCost := 22
def lyftCost := uberCost - 3
def taxiCost := lyftCost - 4
def detourCost := taxiCost + 0.15 * taxiCost
def discountedTaxiCost := detourCost - 0.10 * detourCost
def originalTaxiCost := taxiCost
def tip := 0.20 * originalTaxiCost
def totalCost := discountedTaxiCost + tip

theorem total_cost_of_taxi_ride_downtown : totalCost = 18.53 :=
by
  sorry

end total_cost_of_taxi_ride_downtown_l102_102488


namespace car_gasoline_tank_capacity_l102_102230

theorem car_gasoline_tank_capacity
    (speed : ℝ)
    (usage_rate : ℝ)
    (travel_time : ℝ)
    (fraction_used : ℝ)
    (tank_capacity : ℝ)
    (gallons_used : ℝ)
    (distance_traveled : ℝ) :
  speed = 50 →
  usage_rate = 1 / 30 →
  travel_time = 5 →
  fraction_used = 0.5555555555555556 →
  distance_traveled = speed * travel_time →
  gallons_used = distance_traveled * usage_rate →
  gallon_used = tank_capacity * fraction_used →
  tank_capacity = 15 :=
by
  intros hs hr ht hf hd hu hf
  sorry

end car_gasoline_tank_capacity_l102_102230


namespace nested_function_evaluation_l102_102320

def f (x : ℝ) : ℝ :=
if x > 0 then log 4 x else 3^x

theorem nested_function_evaluation :
  f (f (1/4)) = 1/3 :=
by
  sorry

end nested_function_evaluation_l102_102320


namespace person_A_catches_person_B_after_15_minutes_l102_102802

noncomputable def distance := ℝ
def time (x : ℝ) := ℝ

-- Given conditions
def A_to_B_distance : distance := sorry -- let distance be S (unassigned)
def A_speed (d : distance) := d / 0.5 -- given Person A takes 30 minutes; speed = S / 0.5 hours
def B_speed (d : distance) := d / (2/3) -- given Person B takes 40 minutes; speed = S / (2/3) hours

def B_earlier_departure := 5 / 60 -- 5 minutes earlier in hours

-- Equivalent proof problem
theorem person_A_catches_person_B_after_15_minutes (d : distance) :
  let t := 15 / 60 in -- 15 minutes converted to hours
  (A_speed d) * t = (B_speed d) * (t + B_earlier_departure) :=
sorry

end person_A_catches_person_B_after_15_minutes_l102_102802


namespace parabola_integer_points_l102_102120

/-- 
Given a parabola Q with focus (0,0) and passing through points (3,4) and (-3,-4),
prove there are exactly 24 integer coordinate points (x,y) on Q such that |3x + 4y| ≤ 900.
-/
theorem parabola_integer_points :
  let Q := {p : ℝ × ℝ | ∃ (a b : ℝ), let f : ℝ × ℝ := (0, 0) in 
                                    p = (a, b) ∧ 
                                    b = (1/20) * (a^2 - 25) } in
  let points := {p : ℤ × ℤ | ∃ (x y : ℤ), Q (x, y) ∧ abs (3 * x + 4 * y) ≤ 900} in
  points.card = 24 := sorry

end parabola_integer_points_l102_102120


namespace sum_first_50_odd_numbers_sum_first_n_odd_numbers_sum_odd_numbers_from_201_to_399_l102_102266

theorem sum_first_50_odd_numbers : (1 + 3 + 5 + 7 + 9 + ... + 99) = 2500 := by
  sorry

theorem sum_first_n_odd_numbers (n : ℕ) (hn : n ≥ 1) : (1 + 3 + 5 + ... + (2*n - 1)) = n^2 := by
  sorry

theorem sum_odd_numbers_from_201_to_399 : (201 + 203 + 205 + ... + 399) = 30000 := by
  sorry

end sum_first_50_odd_numbers_sum_first_n_odd_numbers_sum_odd_numbers_from_201_to_399_l102_102266


namespace roger_toys_l102_102809

theorem roger_toys (initial_money spent_money toy_cost remaining_money toys : ℕ) 
  (h1 : initial_money = 63) 
  (h2 : spent_money = 48) 
  (h3 : toy_cost = 3) 
  (h4 : remaining_money = initial_money - spent_money) 
  (h5 : toys = remaining_money / toy_cost) : 
  toys = 5 := 
by 
  sorry

end roger_toys_l102_102809


namespace problem_l102_102338

-- Definitions of sets U, A, B
def U := set ℝ

def A : set ℝ := {x | abs (x + 3) - abs (x - 3) > 3}

def B : set ℝ := {x | ∃ t > 0, x = (t^2 - 4 * t + 1) / t}

def complement_U (A : set ℝ) : set ℝ := {x | x ∈ U ∧ ¬ (x ∈ A)}

-- The proof statement
theorem problem : (B ∩ (complement_U A)) = {x | -2 ≤ x ∧ x ≤ 3/2} :=
sorry

end problem_l102_102338


namespace length_of_T3_l102_102954

theorem length_of_T3 (a : ℝ) (h1 : a = 50) (h2 : ∑' n, 3 * (a / (2 ^ n)) = 300) : T3 = 12.5 := by
  have h3 : T3 = a / 4
  { sorry }

  rw [h1, h3] 
  norm_num

end length_of_T3_l102_102954


namespace population_after_two_years_l102_102460

def initial_population : ℕ := 6000

def decrease_rate_year_1 : ℝ := 0.07
def decrease_rate_year_2 : ℝ := 0.12

def migration_out_year_1 : ℕ := 200
def migration_in_year_1 : ℕ := 100
def migration_out_year_2 : ℕ := 150
def migration_in_year_2 : ℕ := 80

def birth_rate : ℝ := 0.025
def death_rate : ℝ := 0.015

theorem population_after_two_years :
  let decrease_1 := initial_population * decrease_rate_year_1
  let net_migration_1 := migration_in_year_1 - migration_out_year_1
  let births_1 := initial_population * birth_rate
  let deaths_1 := initial_population * death_rate
  let pop_end_year_1 := initial_population - decrease_1 + net_migration_1 + births_1 - deaths_1

  let decrease_2 := pop_end_year_1 * decrease_rate_year_2
  let net_migration_2 := migration_in_year_2 - migration_out_year_2
  let births_2 := pop_end_year_1 * birth_rate
  let deaths_2 := pop_end_year_1 * death_rate
  let pop_end_year_2 := pop_end_year_1 - decrease_2 + net_migration_2 + births_2 - deaths_2
  in
  pop_end_year_2 = 4914 := 
sorry

end population_after_two_years_l102_102460


namespace max_volume_tetrahedron_l102_102661

-- Definitions to mirror the conditions
def side_lengths (a b c : ℝ) (h : a = 4 ∧ b = 5 ∧ c = 6) : Prop := h

def circumcircle_is_great_circle (circum_radius : ℝ → Prop) (radius_sphere : ℝ) : Prop :=
    ∀ r, circum_radius r ↔ r = radius_sphere

def point_on_sphere_surface (P : Type) (on_surface : P → Prop) : Prop := ∀ p, on_surface p

-- The theorem statement
theorem max_volume_tetrahedron 
    (a b c : ℝ) (h₁ : side_lengths a b c (and.intro rfl (and.intro rfl rfl)))
    (circum_radius : ℝ → Prop) (radius_sphere : ℝ) 
    (h₂ : circumcircle_is_great_circle circum_radius radius_sphere)
    (P : Type) (on_surface : P → Prop) (h₃ : point_on_sphere_surface P on_surface) :
    ∃ V : ℝ, V = 10 :=
begin
  sorry -- Proof omitted
end

end max_volume_tetrahedron_l102_102661


namespace fred_initial_cards_l102_102639

/-- 
Fred had some baseball cards. Keith bought 22 of Fred's baseball cards, 
and now Fred has 18 baseball cards left. How many baseball cards did 
Fred have initially?
-/
theorem fred_initial_cards (cards_bought_by_keith : ℕ) (cards_left : ℕ) 
(h1 : cards_bought_by_keith = 22) (h2 : cards_left = 18) :
  ∃ initial_cards : ℕ, initial_cards = cards_left + cards_bought_by_keith :=
begin
  sorry
end

end fred_initial_cards_l102_102639


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102623

-- Conditions and Probability Definitions
noncomputable def prob_indicator (k : ℕ) : ℝ := (5 / 6) ^ (k - 1)

-- a) Expected total number of dice rolled
theorem expected_total_dice_rolled : 5 * (1 / (1 - 5/6)) = 30 := by
  sorry

-- b) Expected total number of points rolled
theorem expected_total_points_rolled :
  let expected_one_die_points : ℝ := 3.5 / (1 - 5/6)
  in 5 * expected_one_die_points = 105 := by
  sorry

-- c) Expected number of salvos
theorem expected_number_of_salvos :
  let expected_salvos (n : ℕ) : ℝ := 
    ∑ k in Finset.range n, (n.choose k * 5^(n - k) * (1/ (1 - 5/6))) / (6^n)
  in expected_salvos 5 ≈ 13.02 := by
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102623


namespace shaded_areas_comparison_l102_102995

theorem shaded_areas_comparison :
  let size := (λ (square : String), 1) -- assuming each square is normalized to size 1
  let area_I := 2 * (1 / 4) -- shaded area in Square I
  let area_II := 3 * (1 / 9) -- shaded area in Square II
  let area_III := 4 * (1 / 12) -- shaded area in Square III
  (area_I = area_II ∨ area_I = area_III ∨ area_II = area_III) ∧
  ¬(area_I = area_II ∧ area_I = area_III ) :=
by
  have hI : area_I = 1 / 2 := by rfl
  have hII : area_II = 1 / 3 := by rfl
  have hIII : area_III = 1 / 3 := by rfl
  left
  exact hII.symm.trans hIII
  exact hI.symm.trans hII
  sorry -- Skipping final proof steps

end shaded_areas_comparison_l102_102995


namespace smallest_rectangles_needed_l102_102501

theorem smallest_rectangles_needed {a b : ℕ} (h1 : a = 3) (h2 : b = 4) :
  ∃ (n : ℕ), ∀ (s : ℕ), square_side s ∧ (∃ m, m * (a * b) = s * s) → n = 12 :=
begin
  sorry

end smallest_rectangles_needed_l102_102501


namespace exists_complex_z0_l102_102396

theorem exists_complex_z0 {a : ℂ} (ha : a.re > 0 ∧ a.re < 1) :
  ∀ (z : ℂ), |z| ≥ 1 → 
  ∃ z0 : ℂ, |z0| = 1 ∧ |(z0^2 - z0 + a)| ≤ |(z^2 - z + a)| :=
by sorry

end exists_complex_z0_l102_102396


namespace part_a_part_b_KM_max_part_b_LN_max_l102_102771

-- Define the points and midpoints in a convex quadrilateral
namespace Quadrilateral

variables {A B C D K L M N : Type*} [MetricSpace A] [MetricSpace B] 
          [MetricSpace C] [MetricSpace D] [MetricSpace K]
          [MetricSpace L] [MetricSpace M] [MetricSpace N]

-- Assume K, L, M, N are midpoints of AB, BC, CD, and DA respectively
def midpoint (a b : Type*) [MetricSpace a] [MetricSpace b] : Type* :=
  sorry -- Define a midpoint properly in your context

axiom is_midpoint_AB_K : midpoint A B = K
axiom is_midpoint_BC_L : midpoint B C = L
axiom is_midpoint_CD_M : midpoint C D = M
axiom is_midpoint_DA_N : midpoint D A = N

-- We need to prove
theorem part_a : (dist K M) ≤ (dist B C + dist D A) / 2 :=
by sorry -- Proof needed

-- Given fixed side lengths
variables (a b c d : ℝ)

axiom length_AB : dist A B = a
axiom length_BC : dist B C = b
axiom length_CD : dist C D = c
axiom length_DA : dist D A = d

-- Conclusions about maximum lengths
theorem part_b_KM_max : (dist K M) = (dist B C + dist D A) / 2 :=
by sorry -- Proof needed

theorem part_b_LN_max : (dist L N) ≤ (Real.sqrt (2 * (a^2 + c^2) - (b - d)^2)) / 2 :=
by sorry -- Proof needed

end Quadrilateral

end part_a_part_b_KM_max_part_b_LN_max_l102_102771


namespace combined_loss_l102_102798

variable (initial : ℕ) (donation : ℕ) (prize : ℕ) (final : ℕ) (lottery_winning : ℕ) (X : ℕ)

theorem combined_loss (h1 : initial = 10) (h2 : donation = 4) (h3 : prize = 90) 
                      (h4 : final = 94) (h5 : lottery_winning = 65) :
                      (initial - donation + prize - X + lottery_winning = final) ↔ (X = 67) :=
by
  -- proof steps will go here
  sorry

end combined_loss_l102_102798


namespace max_similar_triangles_five_points_l102_102073

-- Let P be a finite set of points on a plane with exactly 5 elements.
def max_similar_triangles(P : Finset (ℝ × ℝ)) : ℕ :=
  if h : P.card = 5 then
    8
  else
    0 -- This is irrelevant for the problem statement, but we need to define it.

-- The main theorem statement
theorem max_similar_triangles_five_points {P : Finset (ℝ × ℝ)} (h : P.card = 5) :
  max_similar_triangles P = 8 :=
sorry

end max_similar_triangles_five_points_l102_102073


namespace line_TK_is_bisector_of_angle_MTN_l102_102774

variables {T K M N : Point}
variables {ω1 ω2 : Circle}
variables (AB CD : Line)

-- Conditions
-- ω1 and ω2 are tangent at T
axiom ω1_tangent_ω2_at_T : ω1.is_tangent_at T ω2

-- M and N are distinct points on ω1, different from T
axiom M_on_ω1 : ω1.contains M
axiom N_on_ω1 : ω1.contains N
axiom M_neq_T : M ≠ T
axiom N_neq_T : N ≠ T

-- AB and CD are chords of ω2 passing through M and N respectively
axiom AB_chord_through_M : AB.is_chord_of ω2 M
axiom CD_chord_through_N : CD.is_chord_of ω2 N

-- BD, AC, and MN intersect at K
axiom BD_intersects_AC_and_MN_at_K : 
  (BD : Line).intersects(AC : Line).intersects(MN : Line) = K

-- Problem statement: Show that the line (TK) is the bisector of the angle ∠MTN
theorem line_TK_is_bisector_of_angle_MTN (h: ω1_tangent_ω2_at_T):
  bisector_of_angle (∠ M T N) (Line.mk T K) := sorry

end line_TK_is_bisector_of_angle_MTN_l102_102774


namespace angle_x_in_triangle_l102_102896

theorem angle_x_in_triangle :
  ∃ x : ℝ, (x + 3 * x + 40 = 180) ∧ x = 35 :=
by
  use 35
  split
  sorry
  sorry

end angle_x_in_triangle_l102_102896


namespace Jovana_final_addition_l102_102394

theorem Jovana_final_addition 
  (initial_amount added_initial removed final_amount x : ℕ)
  (h1 : initial_amount = 5)
  (h2 : added_initial = 9)
  (h3 : removed = 2)
  (h4 : final_amount = 28) :
  final_amount = initial_amount + added_initial - removed + x → x = 16 :=
by
  intros h
  sorry

end Jovana_final_addition_l102_102394


namespace evaluate_expression_l102_102262

theorem evaluate_expression : 2 * (2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3) = 24240542 :=
by
  let a := 2009
  let b := 2010
  sorry

end evaluate_expression_l102_102262


namespace initial_friends_online_l102_102137

theorem initial_friends_online (F : ℕ) 
  (h1 : 8 + F = 13) 
  (h2 : 6 * F = 30) : 
  F = 5 :=
by
  sorry

end initial_friends_online_l102_102137


namespace infinite_B_with_1980_solutions_l102_102432

theorem infinite_B_with_1980_solutions :
  ∃ᶠ B in at_top, ∃ (S : finset (ℤ × ℤ)), S.card ≥ 1980 ∧ 
  (∀ (x y : ℤ), (x, y) ∈ S ↔ ⌊x^3 / 2⌋ + ⌊y^3 / 2⌋ = B) :=
sorry

end infinite_B_with_1980_solutions_l102_102432


namespace greatest_matching_pairs_left_l102_102905

-- Define the initial number of pairs and lost individual shoes
def initial_pairs : ℕ := 26
def lost_ind_shoes : ℕ := 9

-- The statement to be proved
theorem greatest_matching_pairs_left : 
  (initial_pairs * 2 - lost_ind_shoes) / 2 + (initial_pairs - (initial_pairs * 2 - lost_ind_shoes) / 2) / 1 = 17 := 
by 
  sorry

end greatest_matching_pairs_left_l102_102905


namespace ellipse_equation_max_area_triangle_l102_102309

-- Ellipse parameters
def ellipse_focus_on_y_axis (c : ℝ) (e : ℝ) (a : ℝ) : Prop := 
  c = a * e

-- Ellipse equation proof
theorem ellipse_equation (a b : ℝ) (h1 : a = 2) (h2 : b^2 = a^2 - (a * (√3 / 2))^2) : 
  (h2 : b^2 = 4 - 3) →
  ellipse_focus_on_y_axis (√3) (√3 / 2) 2 →
  (∀ x y : ℝ, (x^2 / b^2) + (y^2 / a^2) = 1) →
  ∃ (eq : ℝ → ℝ → Prop), eq = (λ x y, x^2 + y^2 / 4 = 1) := sorry

-- Maximum area proof
theorem max_area_triangle (P : ℝ × ℝ) (line_eq : ℝ → ℝ) (a : ℝ) (b : ℝ) 
  (h1 : P = (0, -3)) 
  (h2 : ∀ k x : ℝ, line_eq x = k * x + 1)
  (h3 : a = 2) 
  (h4 : b = 1) 
  (h5 : ∀ k : ℝ, ∃ t : ℝ, t = sqrt(3 + k^2)) :
  let area := λ t : ℝ, (8 * t) / (t^2 + 1) 
  in Sup (set.range area) = 2 * (√3) := sorry

end ellipse_equation_max_area_triangle_l102_102309


namespace find_f3_l102_102322

def f (x a b : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1)) + a*x^7 + b*x^3 - 4

theorem find_f3 (a b : ℝ) (h : f (-3) a b = 4) : f 3 a b = -12 := by
  sorry

end find_f3_l102_102322


namespace equal_ratios_of_perpendiculars_l102_102768

/--
Given:
- ABC is an acute triangle.
- D is the foot of the perpendicular from A to side BC.
- P is a point on segment AD.
- Lines BP and CP intersect sides AC and AB at E and F, respectively.
- J and K are the feet of the perpendiculars from E and F to AD, respectively.

Prove that FK / KD = EJ / JD.
-/
theorem equal_ratios_of_perpendiculars 
  (A B C D P E F J K : Point)
  (h_acute : acute_triangle A B C)
  (h_perp_D : foot_perpendicular A B C D)
  (h_P_on_AD : P ∈ segment A D)
  (h_intersect_E : BP ∩ AC = {E})
  (h_intersect_F : CP ∩ AB = {F})
  (h_perp_J : foot_perpendicular E A D J)
  (h_perp_K : foot_perpendicular F A D K)
  : FK / KD = EJ / JD := sorry

end equal_ratios_of_perpendiculars_l102_102768


namespace number_of_possible_values_for_x_is_3_l102_102218

noncomputable def num_possible_values (x : ℕ) : ℕ :=
  if h : x ∈ {d | d ∣ 36 ∧ d ∣ 63} then
    ({d | d ∣ 9}.to_finset.card)
  else
    0

theorem number_of_possible_values_for_x_is_3 : num_possible_values 3 = 3 :=
by {
  unfold num_possible_values,
  simp,
  sorry
}

end number_of_possible_values_for_x_is_3_l102_102218


namespace number_of_dogs_l102_102070

theorem number_of_dogs (dogs chickens total_legs: ℕ) (dogs_have_4_legs: ∀ d, d ∈ dogs → d.legs = 4)
  (chickens_have_2_legs: ∀ c, c ∈ chickens → c.legs = 2)
  (chickens_eq_2 : chickens = 2)
  (total_legs_eq_12 : total_legs = 12)
  (legs_count_eq : total_legs = 4 * dogs + 2 * chickens):
  dogs = 2 :=
by {
  have h : 4 * dogs + 4 = 12, from calc
    4 * dogs + 4 = total_legs :
      by sorry, -- Need to prove this step.
    total_legs = 12 : by sorry;
  have h1 : 4 * dogs = 8, from by {
    rw h,
    ring,
  };
  have h2 : dogs = 2, from eq_of_mul_eq_mul_left
    (show 4 ≠ 0, from dec_trivial) h;
  exact h2,
}

end number_of_dogs_l102_102070


namespace bag_of_food_costs_two_dollars_l102_102062

theorem bag_of_food_costs_two_dollars
  (cost_puppy : ℕ)
  (total_cost : ℕ)
  (daily_food : ℚ)
  (bag_food_quantity : ℚ)
  (weeks : ℕ)
  (h1 : cost_puppy = 10)
  (h2 : total_cost = 14)
  (h3 : daily_food = 1/3)
  (h4 : bag_food_quantity = 3.5)
  (h5 : weeks = 3) :
  (total_cost - cost_puppy) / (21 * daily_food / bag_food_quantity) = 2 := 
  by sorry

end bag_of_food_costs_two_dollars_l102_102062


namespace carla_greatest_possible_speed_l102_102977

def is_palindrome (n : Nat) : Prop :=
  let s := n.digits 10
  s = s.reverse

noncomputable def greatest_possible_average_speed (start_palindrome : Nat) (end_palindrome : Nat) (time_hours : Nat) (max_speed : Nat) (distance_traveled : Nat) : Nat :=
  if is_palindrome start_palindrome ∧ is_palindrome end_palindrome ∧ max_speed = 80 ∧ time_hours = 4 ∧ end_palindrome - start_palindrome = distance_traveled ∧ distance_traveled ≤ time_hours * max_speed then
    distance_traveled / time_hours
  else
    0

theorem carla_greatest_possible_speed : 
  ∃ start_palindrome end_palindrome distance_traveled, 
  is_palindrome start_palindrome ∧ 
  is_palindrome end_palindrome ∧ 
  max_speed = 80 ∧ 
  time_hours = 4 ∧
  start_palindrome = 12321 ∧
  greatest_possible_average_speed start_palindrome end_palindrome time_hours max_speed distance_traveled = 75 :=
sorry

end carla_greatest_possible_speed_l102_102977


namespace verify_complex_expression_l102_102999

-- Definition of the problem
def complex_expression (z1 z2 z3 : ℂ) : ℂ :=
  z1 * z2 + z3

-- Specific instances in the problem
def z1 := (7 : ℂ) - 8 * complex.I
def z2 := (-6 : ℂ) + 3 * complex.I
def z3 := (2 : ℂ) + complex.I
  
-- Correct answer
def target := (-16 : ℂ) + 70 * complex.I

-- Proof statement 
theorem verify_complex_expression : 
  complex_expression z1 z2 z3 = target :=
by 
  sorry

end verify_complex_expression_l102_102999


namespace overall_average_tickets_sold_l102_102537

variable {M : ℕ} -- number of male members
variable {F : ℕ} -- number of female members
variable (male_to_female_ratio : M * 2 = F) -- 1:2 ratio
variable (average_female : ℕ) (average_male : ℕ) -- average tickets sold by female/male members
variable (total_tickets_female : F * average_female = 70 * F) -- Total tickets sold by female members
variable (total_tickets_male : M * average_male = 58 * M) -- Total tickets sold by male members

-- The overall average number of raffle tickets sold per member is 66.
theorem overall_average_tickets_sold 
  (h1 : 70 * F + 58 * M = 198 * M) -- total tickets sold
  (h2 : M + F = 3 * M) -- total number of members
  : (70 * F + 58 * M) / (M + F) = 66 := by
  sorry

end overall_average_tickets_sold_l102_102537


namespace sum_first_1000_setS_l102_102411

def setS : Set ℕ := { k | ∃ n : ℕ, k = 8 * n + 5 }

def nth_number (n : ℕ) : ℕ := 8 * n + 5

theorem sum_first_1000_setS :
  (Finset.range 1000).sum (λ n, nth_number n) = 4003500 := 
sorry

end sum_first_1000_setS_l102_102411


namespace parabola_proof_l102_102702

def parabola_relation (a b : ℝ) : Prop :=
  (a * 3^2 + b * 3 + 3 = 0) ∧ (a * 4^2 + b * 4 + 3 = 3)

theorem parabola_proof : ∃ a b : ℝ, parabola_relation a b ∧ 
  (∀ x : ℝ, (y = a * x^2 + b * x + 3 ↔ y = x^2 - 4x + 3)) ∧ 
  (a > 0 ∧ x = 2 ∧ (∀ x, -((b + x)^2) + 3 = -1)) :=
by {
  sorry
}

end parabola_proof_l102_102702


namespace perpendicular_lines_l102_102651

variables {A B C D E1 F1 E2 F2 : Type}
variables [convex_quadrilateral A B C D]
variables (ratio_eq : (AE1 / ED = BF1 / FC = AE2 / EC = BF2 / FD = AB / CD))
variables (no_coincide : ∀ (X Y : Type), X ≠ Y)

theorem perpendicular_lines (A B C D E1 F1 E2 F2 : Type)
  [convex_quadrilateral A B C D]
  (ratio_eq : (AE1 / ED = BF1 / FC = AE2 / EC = BF2 / FD = AB / CD))
  (no_coincide : ∀ (X Y : Type), X ≠ Y) :
  are_perpendicular E1F1 E2F2 :=
sorry

end perpendicular_lines_l102_102651


namespace people_own_only_cats_and_dogs_l102_102476

-- Define the given conditions
def total_people : ℕ := 59
def only_dogs : ℕ := 15
def only_cats : ℕ := 10
def cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 29

-- Define the proof problem
theorem people_own_only_cats_and_dogs : ∃ x : ℕ, 15 + 10 + x + 3 + (29 - 3) = 59 ∧ x = 5 :=
by {
  sorry
}

end people_own_only_cats_and_dogs_l102_102476


namespace find_a_and_sequence_l102_102366

theorem find_a_and_sequence : ∃ a : ℤ,
  (a = 6 ∧ sequence = [2, 8, 14]) ∨ 
  (a = 9 ∧ sequence = [5, 8, 11]) ∨ 
  (a = 12 ∧ sequence = [2, 8, 14]) := 
sorry

end find_a_and_sequence_l102_102366


namespace min_max_values_of_f_l102_102850

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x >= - (3 * Real.pi / 2)) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = - (3 * Real.pi / 2)) ∧
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x <= Real.pi / 2 + 2) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = Real.pi / 2 + 2) :=
by {
  -- Proof omitted
  sorry
}

end min_max_values_of_f_l102_102850


namespace cannot_factor_polynomial_l102_102519

theorem cannot_factor_polynomial (a b c d : ℤ) :
  ¬(x^4 + 3 * x^3 + 6 * x^2 + 9 * x + 12 = (x^2 + a * x + b) * (x^2 + c * x + d)) := 
by {
  sorry
}

end cannot_factor_polynomial_l102_102519


namespace rectangles_on_4x4_grid_l102_102586

theorem rectangles_on_4x4_grid : 
  (∃ (grid : finset (fin 4 × fin 4)), 
    (∀ (p1 p2 : fin 4 × fin 4), p1 ≠ p2 ∧ 
    ((p1.1 = p2.1 → p1.2 ≠ p2.2) ∨ (p1.2 = p2.2 → p1.1 ≠ p2.1))) ∧ 
    (∃ (rectangles : ℕ), rectangles = 100)) :=
begin
  sorry
end

end rectangles_on_4x4_grid_l102_102586


namespace games_played_so_far_l102_102201

variable {G : ℕ} -- Initial number of games
variable {A : ℕ} -- Number of additional games

-- Conditions
def won_40_percent_initial_games (W : ℕ) := W = 0.40 * G
def additional_games_and_wins (A : ℕ) (add_wins : ℕ) := A = 20 ∧ add_wins = 0.80 * A
def final_performance (W : ℕ) (total_wins : ℕ) (total_games : ℕ) := total_wins = 0.60 * total_games

theorem games_played_so_far 
    (W : ℕ)
    (add_wins : ℕ)
    (total_wins : ℕ)
    (total_games : ℕ)
    (h1 : won_40_percent_initial_games W)
    (h2 : additional_games_and_wins A add_wins)
    (h3 : final_performance W total_wins total_games)
    (G : ℕ := 20) :
    G = 20 :=
    sorry


end games_played_so_far_l102_102201


namespace sequence_is_palindromic_sum_of_7_consecutive_is_minus_1_sum_of_11_consecutive_is_plus_1_l102_102604

def sequence_length : ℕ := 16
def sequence : ℕ → ℤ
| 0 := 5
| 1 := 5
| 2 := -13
| 3 := 5
| 4 := 5
| 5 := 5
| 6 := -13
| 7 := 5
| 8 := 5
| 9 := -13
| 10 := 5
| 11 := 5
| 12 := 5
| 13 := -13
| 14 := 5
| 15 := 5
| _ := 0

theorem sequence_is_palindromic :
  ∀ i, i < 8 → sequence i = sequence (15 - i) :=
by sorry

theorem sum_of_7_consecutive_is_minus_1 :
  ∀ i, i ≤ 9 → (sequence i + sequence (i + 1) + sequence (i + 2) + sequence (i + 3) + sequence (i + 4) + sequence (i + 5) + sequence (i + 6)) = -1 :=
by sorry

theorem sum_of_11_consecutive_is_plus_1 :
  ∀ i, i ≤ 5 → (sequence i + sequence (i + 1) + sequence (i + 2) + sequence (i + 3) + sequence (i + 4) + sequence (i + 5) + sequence (i + 6) + sequence (i + 7) + sequence (i + 8) + sequence (i + 9) + sequence (i + 10)) = 1 :=
by sorry

end sequence_is_palindromic_sum_of_7_consecutive_is_minus_1_sum_of_11_consecutive_is_plus_1_l102_102604


namespace simplification_cos_squares_l102_102438

open Real

theorem simplification_cos_squares (x y : ℝ) :
  cos^2 (x + π / 4) + cos^2 (x + y + π / 2) - 2 * cos (x + π / 4) * cos (y + π / 4) * cos (x + y + π / 2) = 1 :=
by
  sorry

end simplification_cos_squares_l102_102438


namespace sum_tens_ones_digit_l102_102150

theorem sum_tens_ones_digit (a : ℕ) (b : ℕ) (n : ℕ) (h : a - b = 3) :
  let d := (3^n)
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  ones_digit + tens_digit = 9 :=
by 
  let d := 3^17
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  sorry

end sum_tens_ones_digit_l102_102150


namespace ed_more_marbles_than_doug_initially_l102_102259

noncomputable def ed_initial_marbles := 37
noncomputable def doug_marbles := 5

theorem ed_more_marbles_than_doug_initially :
  ed_initial_marbles - doug_marbles = 32 := by
  sorry

end ed_more_marbles_than_doug_initially_l102_102259


namespace minimum_value_expr_l102_102289

theorem minimum_value_expr (a : ℝ) (h₀ : 0 < a) (h₁ : a < 2) : 
  ∃ (m : ℝ), m = (4 / a + 1 / (2 - a)) ∧ m = 9 / 2 :=
by
  sorry

end minimum_value_expr_l102_102289


namespace max_sum_b_l102_102528

theorem max_sum_b (a : Fin 101 → ℝ) (b : Fin 101 → ℝ)
  (h0 : a 100 ≤ a 0)
  (h1 : ∀ n : Fin 100, (a (n + 1) = a n / 2 ∧ b (n + 1) = 1 / 2 - a n) ∨ (a (n + 1) = 2 * (a n)^2 ∧ b (n + 1) = a n)) :
  (∑ n in Finset.range 100, b (n + 1)) ≤ 50 :=
sorry

end max_sum_b_l102_102528


namespace triangle_probability_is_9_over_35_l102_102813

-- Define the stick lengths
def stick_lengths : List ℕ := [2, 3, 5, 7, 11, 13, 17]

-- Define the function that checks the triangle inequality for three lengths
def can_form_triangle (a b c : ℕ) : Bool :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Get all combinations of three stick lengths
def all_combinations : List (ℕ × ℕ × ℕ) :=
  List.combinations 3 stick_lengths |>.map (λ l => (l[0], l[1], l[2]))

-- Get the valid combinations that can form a triangle
def valid_combinations : List (ℕ × ℕ × ℕ) :=
  all_combinations.filter (λ ⟨a, b, c⟩ => can_form_triangle a b c)

-- Calculate the probability as a rational number
def triangle_probability : ℚ :=
  valid_combinations.length / all_combinations.length

-- The theorem to prove
theorem triangle_probability_is_9_over_35 : triangle_probability = 9 / 35 :=
  sorry

end triangle_probability_is_9_over_35_l102_102813


namespace amino_inequality_l102_102172

theorem amino_inequality
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h : x + y + z = x * y * z) :
  ( (x^2 - 1) / x )^2 + ( (y^2 - 1) / y )^2 + ( (z^2 - 1) / z )^2 ≥ 4 := by
  sorry

end amino_inequality_l102_102172


namespace range_of_k_for_ellipse_l102_102832

theorem range_of_k_for_ellipse (k : ℝ) :
  (∃ f : ℝ → ℝ → ℝ, (∀ x y, f x y = (x^2 / (2 - k) + y^2 / (2*k - 1)) = 1) ∧
  (2 - k > 0) ∧ (2*k - 1 > 0) ∧ (2 - k > 2*k - 1)) ↔ (1 / 2 < k ∧ k < 1) :=
sorry

end range_of_k_for_ellipse_l102_102832


namespace focus_of_parabola_l102_102102

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end focus_of_parabola_l102_102102


namespace smallest_group_size_l102_102801

theorem smallest_group_size (n : ℕ) (k : ℕ) (hk : k > 2) (h1 : n % 2 = 0) (h2 : n % k = 0) :
  n = 6 :=
sorry

end smallest_group_size_l102_102801


namespace prime_expression_integer_value_l102_102273

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_expression_integer_value (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ n, (p * q + p^p + q^q) % (p + q) = 0 → n = 3 :=
by
  sorry

end prime_expression_integer_value_l102_102273


namespace rational_terms_binomial_expansion_a_rational_terms_binomial_expansion_b_l102_102345

-- Definitions required for Part (a):
def rational_terms_count_binomial_expansion_a : ℕ :=
  let n := 100
  let a := 2^(1/2 : ℝ)
  let b := 3^(1/4 : ℝ)
  let valid_k_count := (100 / 4).to_nat + 1 -- Total terms are 26 since k is a multiple of 4
  valid_k_count

-- Definitions required for Part (b):
def rational_terms_count_binomial_expansion_b : ℕ :=
  let n := 300
  let a := 2^(1/2 : ℝ)
  let b := 3^(1/8 : ℝ)
  let valid_k_count := (300 / 24).to_nat + 1 -- Total terms are 13 since k is a multiple of 24
  valid_k_count

-- Proof statements

-- Part (a)
theorem rational_terms_binomial_expansion_a :
  rational_terms_count_binomial_expansion_a = 26 :=
by sorry

-- Part (b)
theorem rational_terms_binomial_expansion_b :
  rational_terms_count_binomial_expansion_b = 13 :=
by sorry

end rational_terms_binomial_expansion_a_rational_terms_binomial_expansion_b_l102_102345


namespace other_x_intercept_l102_102638

-- Given conditions: quadratic equation and specific points
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c

-- Vertex of the quadratic equation
def vertex (a b c : ℝ) := (3 : ℝ, 7 : ℝ)

-- One x-intercept of the quadratic equation
def x_intercept1 (a b c : ℝ) := (-2 : ℝ, 0 : ℝ)

-- Definition of the problem: Proving the other x-intercept
theorem other_x_intercept (a b c : ℝ) 
  (hv : vertex a b c = (3, 7)) 
  (hi : x_intercept1 a b c = (-2, 0)) : 
  (3 + (3 - (-2))) = 8 := 
by 
  sorry

end other_x_intercept_l102_102638


namespace days_worked_l102_102187

theorem days_worked (A_days B_days : ℕ) (work_left : ℝ) (hA : A_days = 15) (hB : B_days = 30) (hW : work_left = 0.6) :
  let d := 4 in
  d = (1 - work_left) * 10 :=
by
  subst hA
  subst hB
  subst hW
  suffices h : 4 = (1 - 0.6) * 10 by
    exact h
  sorry

end days_worked_l102_102187


namespace find_polynomial_P_l102_102399

noncomputable def P (x : ℝ) : ℝ :=
  - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1

theorem find_polynomial_P 
  (α β γ : ℝ)
  (h_roots : ∀ {x: ℝ}, x^3 - 4 * x^2 + 6 * x + 8 = 0 → x = α ∨ x = β ∨ x = γ)
  (h1 : P α = β + γ)
  (h2 : P β = α + γ)
  (h3 : P γ = α + β)
  (h4 : P (α + β + γ) = -20) :
  P x = - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1 :=
by sorry

end find_polynomial_P_l102_102399


namespace ellen_breakfast_calories_l102_102261

theorem ellen_breakfast_calories :
  (let total_calories := 2200 in
   let lunch_calories := 885 in
   let snack_calories := 130 in
   let dinner_remaining := 832 in
   total_calories - lunch_calories - snack_calories - dinner_remaining = 353) := 
by
  -- Proof would go here
  sorry

end ellen_breakfast_calories_l102_102261


namespace x_1998_eq_6_l102_102384

def units_digit (n : ℕ) : ℕ := n % 10

def sequence (x : ℕ → ℕ) : ℕ → ℕ
| 0     := 2
| 1     := 7
| (n+2) := units_digit (x n * x (n+1))

theorem x_1998_eq_6 (x : ℕ → ℕ) (h : ∀ n, x = sequence x) : x 1998 = 6 :=
sorry

end x_1998_eq_6_l102_102384


namespace subsets_count_of_three_element_set_l102_102332

theorem subsets_count_of_three_element_set :
  ∀ (A : set ℕ), A = {0, 1, 2} → (set.powerset A).card = 8 :=
by
  intros A hA,
  sorry

end subsets_count_of_three_element_set_l102_102332


namespace original_two_digit_number_l102_102942

theorem original_two_digit_number :
  ∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ 500 + x = 9 * x - 12 ∧ x = 64 :=
by
  have h₁ : ∀ (x : ℕ), 500 + x = 9 * x - 12 → x = 64 := sorry
  use 64
  split
  all_goals { sorry }

end original_two_digit_number_l102_102942


namespace find_pots_l102_102747

def num_pots := 46
def cost_green_lily := 9
def cost_spider_plant := 6
def total_cost := 390

theorem find_pots (x y : ℕ) (h1 : x + y = num_pots) (h2 : cost_green_lily * x + cost_spider_plant * y = total_cost) :
  x = 38 ∧ y = 8 :=
by
  sorry

end find_pots_l102_102747


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102626

-- Conditions
constant NumDice : ℕ := 5

-- Define the probability of a die not showing six on each roll
def prob_not_six := (5 : ℚ) / 6

-- Question a: Expected total number of dice rolled
theorem expected_total_dice_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 30
  sorry

-- Question b: Expected total number of points rolled
theorem expected_total_points_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 105
  sorry

-- Question c: Expected number of salvos
theorem expected_number_of_salvos :
  True := by
  -- The proof would calculate the expected number of salvos as derived, resulting in 13.02
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102626


namespace graph_contains_k_star_or_k_matching_l102_102048

open SimpleGraph

/-- Let G be a graph and k be a positive integer. If G has strictly more than 2(k-1)^2 edges, 
then G contains a k-star or a k-matching. -/
theorem graph_contains_k_star_or_k_matching 
  (G : SimpleGraph V) (k : ℕ)
  (h1 : 0 < k)
  (h2 : G.edge_finset.card > 2 * (k-1)^2) :
  ∃ (S : Finset (Sym2 V)), 
    ((∃ v : V, S.card = k ∧ ∀ e ∈ S, v ∈ e) ∨ (S.card = k ∧ ∀ e f ∈ S, e ≠ f ∧ Sym2.card e ∩ f = 0)) :=
sorry

end graph_contains_k_star_or_k_matching_l102_102048


namespace a_2022_eq_4043_l102_102128

noncomputable def a_seq (n : ℕ) : ℕ 
| 0       := 0 -- By convention, not relevant here.
| (n + 1) := match n with
                | 0 => 1
                | n' + 1 => ((n' + 2) * a_seq (n' + 1) + 1) / (n' + 1)

theorem a_2022_eq_4043 : a_seq 2022 = 4043 :=
  sorry

end a_2022_eq_4043_l102_102128


namespace continuous_piecewise_l102_102055

def g (x : ℝ) (c d : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if -1 ≤ x ∧ x ≤ 1 then 2 * x - 3
  else 3 * x - d

theorem continuous_piecewise (c d : ℝ) (h1 : c + 2 = -1) (h2 : -5 = -3 - d) : c + d = -1 :=
by
  sorry

end continuous_piecewise_l102_102055


namespace math_problem_l102_102580

-- Define the variables and their conditions
def a : ℝ := (-2)^2
def b : ℝ := real.sqrt ((-3)^2)
def c : ℝ := real.cbrt 27
def d : ℝ := abs (real.sqrt 3 - 2)

-- The main statement that needs to be proved
theorem math_problem : a + b - c + d = 6 - real.sqrt 3 := by
  -- Definitions of a, b, c, and d
  have ha : a = 4 := sorry
  have hb : b = 3 := sorry
  have hc : c = 3 := sorry
  have hd : d = 2 - real.sqrt 3 := sorry
  -- Utilizing the definitions
  rw [ha, hb, hc, hd]
  -- Simplify the expression
  sorry

end math_problem_l102_102580


namespace find_a_n_l102_102373

noncomputable def arithmetic_sequence_condition (a : ℕ → ℝ) (n : ℕ) : Prop :=
  let odds_sum := ∑ i in finset.range n, a (2 * i + 1)
  let evens_sum := ∑ i in finset.range (n - 1), a (2 * (i + 1))
  odds_sum = 105 ∧ evens_sum = 87

theorem find_a_n (a : ℕ → ℝ) (n : ℕ) (h : arithmetic_sequence_condition a n) : a n = 18 :=
by {
  sorry
}

end find_a_n_l102_102373


namespace driver_net_rate_of_pay_l102_102199

theorem driver_net_rate_of_pay 
  (hours_traveled : ℝ) (speed : ℝ) (mpg : ℝ) (pay_per_mile : ℝ) (gas_cost_per_gallon : ℝ)
  (h_hours : hours_traveled = 2) (h_speed : speed = 60) (h_mpg : mpg = 30) 
  (h_pay_per_mile : pay_per_mile = 0.5) (h_gas_cost_per_gallon : gas_cost_per_gallon = 2) :
  (pay_per_mile * speed * hours_traveled - gas_cost_per_gallon * (speed * hours_traveled / mpg)) / hours_traveled = 26 :=
by 
  rw [h_hours, h_speed, h_mpg, h_pay_per_mile, h_gas_cost_per_gallon]
  norm_num
  simp
  sorry

end driver_net_rate_of_pay_l102_102199


namespace ladder_length_l102_102921

/-- The length of the ladder leaning against a wall when it forms
    a 60 degree angle with the ground and the foot of the ladder 
    is 9.493063650744542 m from the wall is 18.986127301489084 m. -/
theorem ladder_length (L : ℝ) (adjacent : ℝ) (θ : ℝ) (cosθ : ℝ) :
  θ = Real.pi / 3 ∧ adjacent = 9.493063650744542 ∧ cosθ = Real.cos θ →
  L = 18.986127301489084 :=
by
  intro h
  sorry

end ladder_length_l102_102921


namespace probability_of_rolling_multiple_of_3_l102_102156

theorem probability_of_rolling_multiple_of_3 (n : ℕ) (hn : n = 6) :
  let multiples_of_3 := {3, 6}
  let outcomes := finset.range (n + 1) \ {0}
  let count := multiples_of_3.card
  let total := outcomes.card
  (count / total : ℚ) = 1 / 3 :=
by
  let multiples_of_3 := {3, 6}
  let outcomes := finset.range (n + 1) \ {0}
  let count := multiples_of_3.card
  let total := outcomes.card
  have : count = 2 := by sorry
  have : total = 6 := by sorry
  calc (count / total : ℚ)
        = 2 / 6 : by rw [this, this_1]
    ... = 1 / 3 : by norm_num

end probability_of_rolling_multiple_of_3_l102_102156


namespace journey_speed_l102_102158

theorem journey_speed (v : ℝ) 
  (h1 : 3 * v + 60 * 2 = 240)
  (h2 : 3 + 2 = 5) :
  v = 40 :=
by
  sorry

end journey_speed_l102_102158


namespace point_in_region_l102_102223

-- Definition of points
def A := (0, 2)
def B := (-2, 0)
def C := (0, -2)
def D := (2, 0)

-- Conditions as functions
def cond1 (x y : Int) := x + y - 1 < 0
def cond2 (x y : Int) := x - y + 1 > 0

-- Main theorem to prove
theorem point_in_region : ∃ (p : Int × Int), p = C ∧ cond1 p.1 p.2 ∧ cond2 p.1 p.2 ∧ 
  ¬ (cond1 A.1 A.2 ∧ cond2 A.1 A.2) ∧
  ¬ (cond1 B.1 B.2 ∧ cond2 B.1 B.2) ∧
  ¬ (cond1 D.1 D.2 ∧ cond2 D.1 D.2) :=
by {
  let A := A,
  let B := B,
  let C := C,
  let D := D,
  let cond1 := cond1,
  let cond2 := cond2,
  use C,
  split,
  -- proof parts go here, if needed
  sorry,
}

end point_in_region_l102_102223


namespace verify_sine_function_l102_102279

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ t : ℝ, f (t + T) = f t

def symmetric_about_line (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x : ℝ, f (2 * c - x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y

def center_of_symmetry (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  let (x_0, y_0) := p in f x_0 = y_0

theorem verify_sine_function : 
  (∀ x : ℝ, smallest_positive_period (λ x : ℝ, Real.sin (2 * x - π / 6)) π) ∧
  (∀ x : ℝ, symmetric_about_line (λ x : ℝ, Real.sin (2 * x - π / 6)) (π / 3)) ∧
  (∀ x : ℝ, increasing_on_interval (λ x : ℝ, Real.sin (2 * x - π / 6)) (-π / 6) (π / 3)) ∧
  center_of_symmetry (λ x : ℝ, Real.sin (2 * x - π / 6)) (π / 12, 0) := 
sorry

end verify_sine_function_l102_102279


namespace circle_radius_square_l102_102538

-- Definition of the problem setup
variables {EF GH ER RF GS SH R S : ℝ}

-- Given conditions
def condition1 : ER = 23 := by sorry
def condition2 : RF = 23 := by sorry
def condition3 : GS = 31 := by sorry
def condition4 : SH = 15 := by sorry

-- Circle radius to be proven
def radius_squared : ℝ := 706

-- Lean 4 theorem statement
theorem circle_radius_square (h1 : ER = 23) (h2 : RF = 23) (h3 : GS = 31) (h4 : SH = 15) :
  (r : ℝ) ^ 2 = 706 := sorry

end circle_radius_square_l102_102538


namespace root_in_interval_l102_102114

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem root_in_interval : ∃ c ∈ Set.Ioo 0 1, f c = 0 := by
  have f0 : f 0 < 0 := by
    simp [f, Real.exp_zero]
    norm_num
  have f1 : 0 < f 1 := by
    simp [f, Real.exp_one_lt]
  sorry

end root_in_interval_l102_102114


namespace amara_clothing_remaining_l102_102563

theorem amara_clothing_remaining :
  ∀ (initial donation_one donation_factor discard : ℕ),
    initial = 100 →
    donation_one = 5 →
    donation_factor = 3 →
    discard = 15 →
    let total_donated := donation_one + (donation_factor * donation_one) in
    let remaining_after_donation := initial - total_donated in
    let final_remaining := remaining_after_donation - discard in
    final_remaining = 65 := 
by
  sorry

end amara_clothing_remaining_l102_102563


namespace geometric_series_sum_l102_102599

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
  a / (1 - r)

theorem geometric_series_sum :
  infinite_geometric_series_sum (5 / 3) (1 / 3) (by norm_num : |1 / 3| < 1) = 5 / 2 :=
by
  sorry

end geometric_series_sum_l102_102599


namespace problem1_problem2_l102_102581

-- Define the conditions and goal for Problem 1
def log_expr1 : ℝ :=
  (Real.logb 2 24) + (Real.logb 10 (1/2)) - (Real.logb 3 (Real.sqrt 27)) + (Real.logb 10 2) - (Real.logb 2 3)

theorem problem1 :
  log_expr1 = 2.25 :=
sorry

-- Define the conditions and goal for Problem 2
def exp_expr1 : ℝ :=
  (33 * Real.sqrt 2)^6 - (1 / 9)^(-3/2) - (-8)^0

theorem problem2 :
  exp_expr1 = 33^6 * 8 - 28 :=
sorry

end problem1_problem2_l102_102581


namespace train_crossing_time_l102_102518

theorem train_crossing_time
  (length_train : ℝ)
  (length_bridge : ℝ)
  (speed_train_kmph : ℝ)
  (conversion_factor : ℝ)
  : (length_train = 100) →
    (length_bridge = 150) →
    (speed_train_kmph = 18) →
    (conversion_factor = 1000 / 3600) →
    let speed_train_mps := speed_train_kmph * conversion_factor in
    let total_distance := length_train + length_bridge in
    let time_to_cross := total_distance / speed_train_mps in
    time_to_cross = 50 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  dsimp [speed_train_mps, total_distance, time_to_cross]
  norm_num
  rfl


end train_crossing_time_l102_102518


namespace prove_problem_statement_l102_102050

noncomputable def problem_statement (n : ℕ) (a : Finₓ n → ℝ) : Prop :=
  (∀ i, 0 < a i) ∧ (Finₓ.sum a = 1) →
  2 * Finₓ.sum (λ i, let j := if i < n - 1 then i + 1 else 0;
                  (a i)^2 / (a i + a (j)))  ≥ 1

theorem prove_problem_statement (n : ℕ) (a : Finₓ n → ℝ)
  (h1 : ∀ i, 0 < a i)
  (h2 : Finₓ.sum a = 1) :
  2 * Finₓ.sum (λ i, let j := if i < n - 1 then i + 1 else 0;
                  (a i)^2 / (a i + a (j)))  ≥ 1 :=
sorry

end prove_problem_statement_l102_102050


namespace Ryan_reads_more_l102_102512

theorem Ryan_reads_more 
  (total_pages_Ryan : ℕ)
  (days_in_week : ℕ)
  (pages_per_book_brother : ℕ)
  (books_per_day_brother : ℕ)
  (total_pages_brother : ℕ)
  (Ryan_books : ℕ)
  (Ryan_weeks : ℕ)
  (Brother_weeks : ℕ)
  (days_in_week_def : days_in_week = 7)
  (total_pages_Ryan_def : total_pages_Ryan = 2100)
  (pages_per_book_brother_def : pages_per_book_brother = 200)
  (books_per_day_brother_def : books_per_day_brother = 1)
  (Ryan_weeks_def : Ryan_weeks = 1)
  (Brother_weeks_def : Brother_weeks = 1)
  (total_pages_brother_def : total_pages_brother = pages_per_book_brother * days_in_week)
  : ((total_pages_Ryan / days_in_week) - (total_pages_brother / days_in_week) = 100) :=
by
  -- We provide the proof steps
  sorry

end Ryan_reads_more_l102_102512


namespace range_of_a_iff_condition_l102_102365

theorem range_of_a_iff_condition (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3 * a) ↔ (a ≥ -2 ∧ a ≤ 5) :=
by
  sorry

end range_of_a_iff_condition_l102_102365


namespace swimming_lane_length_l102_102794

theorem swimming_lane_length (round_trips : ℕ) (total_distance : ℕ) (lane_length : ℕ) 
  (h1 : round_trips = 4) (h2 : total_distance = 800) 
  (h3 : total_distance = lane_length * (round_trips * 2)) : 
  lane_length = 100 := 
by
  sorry

end swimming_lane_length_l102_102794


namespace Vlad_height_extra_inches_l102_102492

/--
Vlad is 6 feet, some inches tall. His younger sister is 2 feet, 10 inches tall.
Vlad is 41 inches taller than his sister. Prove that Vlad is 3 inches taller
than 6 feet.
-/
theorem Vlad_height_extra_inches :
  ∀ (vlad_feet vlad_extra sister_feet sister_inches vlad_sister_diff : ℕ), 
  vlad_feet = 6 →
  sister_feet = 2 →
  sister_inches = 10 →
  vlad_sister_diff = 41 →
  vlad_extra = ((sister_feet * 12 + sister_inches) + vlad_sister_diff) - (vlad_feet * 12) →
  vlad_extra = 3 :=
by
  intros vlad_feet vlad_extra sister_feet sister_inches vlad_sister_diff
  assume h1 h2 h3 h4 h5
  sorry

end Vlad_height_extra_inches_l102_102492


namespace dentist_work_hours_l102_102919

theorem dentist_work_hours :
  (∀(b : ℕ), b = 2) →
  (∀(h : ℝ), h = 0.5) →
  (∀(w : ℕ), w = 5) →
  (∀(t : ℕ), t = 160) →
  ∃ (d : ℝ), d = 8 :=
by
  intros b b_prop h h_prop w w_prop t t_prop
  use 8
  sorry

end dentist_work_hours_l102_102919


namespace monika_movie_ticket_cost_l102_102064

theorem monika_movie_ticket_cost 
  (spent_mall : ℝ)
  (num_movies : ℕ)
  (cost_bag : ℝ)
  (num_bags : ℕ)
  (total_spent : ℝ)
  (spent_mall = 250)
  (num_movies = 3)
  (cost_bag = 1.25)
  (num_bags = 20)
  (total_spent = 347) 
  : (total_spent - (spent_mall + (num_bags * cost_bag))) / num_movies = 24 :=
by 
  sorry

end monika_movie_ticket_cost_l102_102064


namespace sufficient_but_not_necessary_condition_l102_102291

open Real

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (x = y → |x| = |y|) ∧ (|x| = |y| → x = y) = false :=
by
  sorry

end sufficient_but_not_necessary_condition_l102_102291


namespace quadratic_decreasing_on_nonneg_real_l102_102362

theorem quadratic_decreasing_on_nonneg_real (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) : 
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → (a * x^2 + b * x + c) ≥ (a * y^2 + b * y + c) :=
by
  sorry

end quadratic_decreasing_on_nonneg_real_l102_102362


namespace set_1234_excellent_no_proper_subset_excellent_l102_102493

open Set

namespace StepLength

def excellent_set (D : Set ℤ) : Prop :=
∀ A : Set ℤ, ∃ a d : ℤ, d ∈ D → ({a - d, a, a + d} ⊆ A ∨ {a - d, a, a + d} ⊆ (univ \ A))

noncomputable def S : Set (Set ℤ) := {{1}, {2}, {3}, {4}}

theorem set_1234_excellent : excellent_set {1, 2, 3, 4} := sorry

theorem no_proper_subset_excellent :
  ¬ (excellent_set {1, 3, 4} ∨ excellent_set {1, 2, 3} ∨ excellent_set {1, 2, 4} ∨ excellent_set {2, 3, 4}) := sorry

end StepLength

end set_1234_excellent_no_proper_subset_excellent_l102_102493


namespace sqrt_D_irrational_l102_102770

-- Define the conditions
def D (x : ℤ) : ℤ := 
  let a := x
  let b := x + 2
  let c := a + b
  a^2 + b^2 + c^2

-- The theorem statement
theorem sqrt_D_irrational (x : ℤ) : (real.sqrt (D x)).irrationals :=
by sorry

end sqrt_D_irrational_l102_102770


namespace eval_f_at_4_l102_102715

def f (x : ℕ) : ℕ := 5 * x + 2

theorem eval_f_at_4 : f 4 = 22 :=
by
  sorry

end eval_f_at_4_l102_102715


namespace total_students_in_class_l102_102797

theorem total_students_in_class (total_jelly_beans : ℕ) (remaining_jelly_beans : ℕ) (boys_more_than_girls : ℕ) (g b : ℕ) :
  total_jelly_beans = 500 →
  remaining_jelly_beans = 10 →
  boys_more_than_girls = 4 →
  b = g + boys_more_than_girls →
  total_jelly_beans - remaining_jelly_beans = g * g + b * b →
  g + b = 32 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end total_students_in_class_l102_102797


namespace uncle_bob_can_park_l102_102926

def parking_lot_problem : Prop :=
  ∃ (n m k l : ℕ), 
    n = 20 ∧ 
    m = 15 ∧ 
    k = 5 ∧ 
    l = 2 ∧ 
    (nat.choose n m) = (nat.choose n k) ∧ 
    (nat.choose 16 k) = 4368 ∧ 
    (nat.choose n k) = 15504 ∧ 
    ((nat.choose 16 k) / (nat.choose n k)) = (273 / 969) ∧ 
    (1 - (273 / 969)) = (232 / 323)

theorem uncle_bob_can_park : parking_lot_problem :=
  by {
  -- sorry to skip the proof
  sorry
}

end uncle_bob_can_park_l102_102926


namespace center_of_mass_eq_l102_102918

noncomputable def center_mass_height (h_v : ℝ) : ℝ := 
  let r := 3.3  -- cm
  let R := 3.45 -- cm
  let h := 10   -- cm
  let h_t := 0.3 -- cm   
  let ρ_water := 1 -- conventionally taken as 1 g/cm³
  let ρ_glass := 2 * ρ_water   
  let m_wall := 2 * pi * h * (R^2 - r^2) * ρ_glass
  let m_bottom := pi * (R^2 - r^2) * h_t * ρ_glass
  let m_water := pi * r^2 * h_v * ρ_water
  let h_A := h_t + h / 2
  let h_B := h_t / 2
  let h_water := h_t + h_v / 2
  let y := ( (h_t + h_v / 2) * pi * r^2 * h_v * ρ_water + h_A * m_wall + h_B * m_bottom ) / ( pi * r^2 * h_v * ρ_water + m_wall + m_bottom )
  y

theorem center_of_mass_eq (h_v : ℝ) :
  center_mass_height h_v = (17 * h_v^2 + 340.5) / (34.21 * h_v + 86.05) :=
by sorry

end center_of_mass_eq_l102_102918


namespace profit_percentage_B_l102_102935

-- Definitions based on conditions:
def CP_A : ℝ := 150  -- Cost price for A
def profit_percentage_A : ℝ := 0.20  -- Profit percentage for A
def SP_C : ℝ := 225  -- Selling price for C

-- Lean statement for the problem:
theorem profit_percentage_B : (SP_C - (CP_A * (1 + profit_percentage_A))) / (CP_A * (1 + profit_percentage_A)) * 100 = 25 := 
by 
  sorry

end profit_percentage_B_l102_102935


namespace sum_of_inserted_numbers_l102_102750

theorem sum_of_inserted_numbers (x y : ℝ) (h1 : x^2 = 2 * y) (h2 : 2 * y = x + 20) :
  x + y = 4 ∨ x + y = 17.5 :=
sorry

end sum_of_inserted_numbers_l102_102750


namespace complex_conjugate_l102_102785

-- Defining the imaginary unit i
def i : ℂ := complex.I

-- Given condition: z = i / (1 - i^3)
def z : ℂ := i / (1 - i^3)

-- The goal is to prove that the conjugate of z is (1/2 - 1/2 * i)
theorem complex_conjugate (z : ℂ) (h : z = i / (1 - i^3)) : complex.conj z = 1/2 - 1/2 * i :=
by {
    subst h,
    sorry
}

end complex_conjugate_l102_102785


namespace coefficient_x4_in_expansion_binomial_coefficients_equality_l102_102312

-- Problem 1: Coefficient of the term containing \(x^4\) in the expansion of \((x - 2/sqrt(x))^{10}\)
theorem coefficient_x4_in_expansion: 
  ∃ c: ℝ, c = 3360 ∧ ∃ k: ℕ, (k = 4 ∧ 
    ( (x - (2 / real.sqrt x)) ^ 10 ).coeff (10 - (3 / 2) * k) = c) :=
    sorry

-- Problem 2: If binomial coefficients of the \(3r\)-th term and \((r+2)\)-th term are equal, then r = 1
theorem binomial_coefficients_equality :
  ∀ r: ℕ, (binomial 10 (3 * r - 1) = binomial 10 (r + 1)) → (r = 1) :=
  sorry

end coefficient_x4_in_expansion_binomial_coefficients_equality_l102_102312


namespace g_at_50_l102_102451

variable (g : ℝ → ℝ)

axiom g_functional_eq (x y : ℝ) : g (x * y) = x * g y
axiom g_at_1 : g 1 = 40

theorem g_at_50 : g 50 = 2000 :=
by
  -- Placeholder for proof
  sorry

end g_at_50_l102_102451


namespace log_base49_7_1_4_equals_1_8_l102_102997

-- Definitions corresponding to conditions
def base_identity : ℕ := 49
def root_base : ℕ := 7
def logarithm_input : ℚ := 1/4

-- The mathematical proof problem statement
theorem log_base49_7_1_4_equals_1_8 
  (h1 : base_identity = root_base^2) : 
  Real.logBase (root_base^2) (root_base^((1/4))) = (1/8) :=
by
  sorry

end log_base49_7_1_4_equals_1_8_l102_102997


namespace number_of_convex_quadrilaterals_l102_102141

theorem number_of_convex_quadrilaterals :
  ∀ (n : ℕ), n = 20 → (Nat.choose n 4 = 4845) :=
by
  intros n hn
  rw hn
  norm_num
  sorry

end number_of_convex_quadrilaterals_l102_102141


namespace principal_amount_l102_102829

theorem principal_amount (P : ℝ) (h : (P * 0.1236) - (P * 0.12) = 36) : P = 10000 := 
sorry

end principal_amount_l102_102829


namespace circumcircles_coaxial_l102_102914

variable (A B C D O E F S E' F' X Y M: Type)
variables [Field K]
variables {circumcircle_OAM circumcircle_OXY diameter_OS : circle K}
variables {AB CD AD BC AC BD : line K}
variables {A B C D O E F S E' F' X Y M: point K}

-- Note that the definitions for "circumscribed", "inscribed", "midpoint" etc. would have to be defined appropriately 
-- in the context of Lean's geometry library if they are not already available.

-- Conditions
def is_circumscribed_and_inscribed (ABCD : quad K) : Prop := 
  circumscribed ABCD ∧ inscribed ABCD ∧ circumcenter ABCD = O ∧
  intersection (line A B) (line C D) = E ∧
  intersection (line A D) (line B C) = F ∧
  intersection (line A C) (line B D) = S ∧
  on_line E' (line A D) ∧ on_line F' (line A B) ∧
  angle A E E' = angle E' E D ∧ angle A F F' = angle F' F B ∧
  on_line X (line O E') ∧ on_line Y (line O F') ∧
  ratio XA XD = ratio EA ED ∧ ratio YA YB = ratio FA FB ∧
  midpoint_arc M (BD : arc K) containing A 

-- Theorem to prove
theorem circumcircles_coaxial (h : is_circumscribed_and_inscribed ABCD) : 
  coaxial circumcircle_OXY circumcircle_OAM diameter_OS :=
sorry

end circumcircles_coaxial_l102_102914


namespace max_male_students_l102_102533

theorem max_male_students (students : Fin 1650 → Prop) (rows : Fin 22) (cols : Fin 75)
  (condition: ∀(c1 c2 : Fin 75) (c1 ≠ c2), 
    ∑ i in Finset.univ, if students (rows i, c1) = students (rows i, c2) then 1 else 0 ≤ 11) : 
  ∑ i in Finset.univ, if students i then 1 else 0 ≤ 928 :=
sorry

end max_male_students_l102_102533


namespace value_of_f_l102_102647

def f (α : ℝ) : ℝ := (sin (π - α) * cos (2 * π - α)) / (cos (-π - α) * tan α)

theorem value_of_f : f (-31 * π / 3) = -1 / 2 :=
by 
  -- Sorry is used to skip the proof
  sorry

end value_of_f_l102_102647


namespace avg_weight_additional_friends_l102_102095

noncomputable def average_weight_additional (A B : ℝ) (h1 : B = A + 10) : ℝ := B + 10

theorem avg_weight_additional_friends
  (A B : ℝ) 
  (initial_avg_increase : A + 10 = B)
  (post_addition_avg : B = A + 10)
  (final_avg_weight : B) :
  average_weight_additional A B post_addition_avg = final_avg_weight + 10 :=
by
  sorry

end avg_weight_additional_friends_l102_102095


namespace area_EFC_l102_102527

-- We define the geometrical setup and given areas as assumptions.
variables {A B C D E F : Type} [AffineSpace ℝ A B C D E F]
variables (x y a b : ℝ)  -- lengths for the rectangle sides and segments
variables (area_EBC area_EAF area_FDC : ℝ)

-- Rectangle ABCD with A at origin
def is_rectangle (A B D C : A) : Prop :=
  -- Insert appropriate conditions for rectangle ABCD ensuring it is a rectangle

-- Define the areas
def area (T : A) : ℝ := sorry -- Definition of area of triangle T

axiom given_areas :
  area (Triangle E B C) = 16 ∧
  area (Triangle E A F) = 12 ∧
  area (Triangle F D C) = 30

-- Main theorem: area of triangle EFC
theorem area_EFC :
  area (Triangle E F C) = 38 :=
sorry

end area_EFC_l102_102527


namespace necessarily_positive_y_plus_z_l102_102077

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end necessarily_positive_y_plus_z_l102_102077


namespace part1_part2_l102_102686

variables {n : ℕ} (a : ℕ → ℚ) (T : ℕ → ℚ)

-- Given conditions
def prod_n (a : ℕ → ℚ) (n : ℕ) : ℚ := 
  (List.prod (List.map a (List.range (n + 1))))

axiom cond1 (n : ℕ) : a n + T n = 1

axiom prod_condition (n : ℕ) : T n = prod_n a n

-- Question 1: Proving the arithmetic sequence property
theorem part1 (n : ℕ) :
  (∀ n, T (n + 1) = a (n + 1) * T n) → 
  (∀ n, (1 / T (n + 1)) - (1 / T n) = 1) :=
sorry

-- Question 2: Proving the inequality for the sum
theorem part2 (n : ℕ) :
  (∀ n, T n = prod_n a n) →
  (∀ n, a n + T n = 1) →
  ∑ k in List.range (n + 1), ((a (k + 1) - a k) / a k) < 3 / 4 :=
sorry

end part1_part2_l102_102686


namespace distance_from_point_to_line_l102_102109

theorem distance_from_point_to_line : 
  let p := (1 : ℝ, -2 : ℝ)
  let A := 1
  let B := -1
  let C := -1
  let distance := (|A * p.1 + B * p.2 + C|) / (Real.sqrt (A^2 + B^2))
  distance = Real.sqrt 2 :=
by
  let p := (1 : ℝ, -2 : ℝ)
  let A := 1
  let B := -1
  let C := -1
  let distance := (|A * p.1 + B * p.2 + C|) / (Real.sqrt (A^2 + B^2))
  sorry

end distance_from_point_to_line_l102_102109


namespace profit_percentage_B_l102_102936

-- Definitions based on conditions:
def CP_A : ℝ := 150  -- Cost price for A
def profit_percentage_A : ℝ := 0.20  -- Profit percentage for A
def SP_C : ℝ := 225  -- Selling price for C

-- Lean statement for the problem:
theorem profit_percentage_B : (SP_C - (CP_A * (1 + profit_percentage_A))) / (CP_A * (1 + profit_percentage_A)) * 100 = 25 := 
by 
  sorry

end profit_percentage_B_l102_102936


namespace min_max_values_l102_102847

noncomputable def f (x : ℝ) := cos x + (x + 1) * sin x + 1

theorem min_max_values :
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≥ - (3 * π) / 2) ∧
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≤ (π / 2 + 2)) :=
sorry

end min_max_values_l102_102847


namespace ratio_of_refurb_to_new_tshirt_l102_102758

def cost_of_new_tshirt : ℤ := 5
def cost_of_pants : ℤ := 4
def cost_of_skirt : ℤ := 6

-- Total income from selling two new T-shirts, one pair of pants, four skirts, and six refurbished T-shirts is $53.
def total_income : ℤ := 53

-- Total income from selling new items.
def income_from_new_items : ℤ :=
  2 * cost_of_new_tshirt + cost_of_pants + 4 * cost_of_skirt

-- Income from refurbished T-shirts.
def income_from_refurb_tshirts : ℤ :=
  total_income - income_from_new_items

-- Number of refurbished T-shirts sold.
def num_refurb_tshirts_sold : ℤ := 6

-- Price of one refurbished T-shirt.
def cost_of_refurb_tshirt : ℤ :=
  income_from_refurb_tshirts / num_refurb_tshirts_sold

-- Prove the ratio of the price of a refurbished T-shirt to a new T-shirt is 0.5
theorem ratio_of_refurb_to_new_tshirt :
  (cost_of_refurb_tshirt : ℚ) / cost_of_new_tshirt = 0.5 := 
sorry

end ratio_of_refurb_to_new_tshirt_l102_102758


namespace find_divisor_l102_102164

theorem find_divisor (d : ℕ) : ((23 = (d * 7) + 2) → d = 3) :=
by
  sorry

end find_divisor_l102_102164


namespace find_real_m_l102_102314

theorem find_real_m (m : ℝ) : 
  let z := ⟨Real.log (m^2 + 2 * m - 14), m^2 - m - 6⟩ in
  z.im = 0 → m^2 + 2 * m - 14 > 0 → m = 3 :=
by
  let z := ⟨Real.log (m^2 + 2 * m - 14), m^2 - m - 6⟩
  intros hzIm hPos
  sorry

end find_real_m_l102_102314


namespace cylinder_height_l102_102534

theorem cylinder_height
    (P : ℝ)
    (d : ℝ)
    (h : ℝ) 
    (hP : P = 6) 
    (hd : d = 10):
    h = 8 :=
by
    have r : ℝ := P / (2 * Real.pi)
    have hsq : h^2 = d^2 - P^2 / (Real.pi^2)
    rw [hP, hd] at hsq
    field_simp [Real.pi_ne_zero] at hsq
    norm_num at hsq
    sorry

end cylinder_height_l102_102534


namespace probability_three_digit_multiple_of_4_l102_102183

theorem probability_three_digit_multiple_of_4 :
  let digits := {1, 2, 3, 4, 5}
  let total_ways := 5 * 4 * 3
  let valid_ways := 15
  valid_ways / total_ways = 1/4 :=
by
  let digits := {1, 2, 3, 4, 5}
  let total_ways := 5 * 4 * 3
  let valid_ways := 15
  have h1 : valid_ways / total_ways = 1/4 := sorry
  exact h1

end probability_three_digit_multiple_of_4_l102_102183


namespace find_alpha_l102_102007

-- Definitions in Lean
variables (A B C D E : Type) 

noncomputable def square (a b c d : Type) : Prop :=
  -- Condition that ABCD is a square
  sorry

noncomputable def equilateral (a d e : Type) : Prop :=
  -- Condition that ADE is an equilateral triangle
  sorry

noncomputable def outside_square (e a b c d : Type) : Prop :=
  -- Condition that E is outside the square ABCD
  sorry

noncomputable def angle_AEB (e a b : Type) (alpha : ℝ) : Prop :=
  α = 30 -- Given that angle AEB is 30 degrees

-- The proof statement
theorem find_alpha (A B C D E : Type) (α : ℝ)
  (h1 : square A B C D)
  (h2 : equilateral A D E)
  (h3 : outside_square E A B C D)
  (h4 : angle_AEB E A B α) :
  α = 15 :=
sorry

end find_alpha_l102_102007


namespace integer_segments_from_vertex_E_l102_102080

theorem integer_segments_from_vertex_E :
  ∀ (DE EF : ℕ), (DE = 12) → (EF = 16) →
  (∃ (n : ℕ), n = 5) :=
by
intros DE EF hDE hEF
exists 5
sorry

end integer_segments_from_vertex_E_l102_102080


namespace ratio_m_n_is_3_over_19_l102_102719

def is_linear (a b : ℚ → ℚ) := ∀ x y, a x = 1 ∧ b y = 1

def constants_eq (m n : ℚ) : Prop := 
  3 * m + 5 * n + 9 = 1 ∧ 4 * m - 2 * n - 1 = 1

theorem ratio_m_n_is_3_over_19 (m n : ℚ) (h : constants_eq m n) :
  m / n = 3 / 19 :=
sorry

end ratio_m_n_is_3_over_19_l102_102719


namespace largest_two_digit_prime_factor_of_binom_300_150_l102_102893

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := Nat.choose 300 150 in
  ∃ p, Nat.Prime p ∧ (10 ≤ p ∧ p < 100) ∧ p ∣ n ∧ 
       (∀ q, Nat.Prime q → (10 ≤ q ∧ q < 100) → q ∣ n → q ≤ p) ∧ p = 89 := 
by
  -- Define n as the binomial coefficient
  let n := Nat.choose 300 150
  -- Proof will be filled here
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l102_102893


namespace original_number_is_64_l102_102945

theorem original_number_is_64 (x : ℕ) : 500 + x = 9 * x - 12 → x = 64 :=
by
  sorry

end original_number_is_64_l102_102945


namespace sum_of_a_b_l102_102356

theorem sum_of_a_b (a b : ℤ) :
  (x^3 + a * x^2 + b * x + 8).has_factor (x + 1) ∧ (x^3 + a * x^2 + b * x + 8).has_factor (x + 2) → 
  a + b = 21 :=
by
  sorry

end sum_of_a_b_l102_102356


namespace diamond_of_2_and_3_l102_102988

def diamond (a b : ℕ) : ℕ := a^3 * b^2 - b + 2

theorem diamond_of_2_and_3 : diamond 2 3 = 71 := by
  sorry

end diamond_of_2_and_3_l102_102988


namespace quadratic_equation_divisible_by_x_minus_one_l102_102655

theorem quadratic_equation_divisible_by_x_minus_one (a b c : ℝ) (h1 : (x - 1) ∣ (a * x * x + b * x + c)) (h2 : c = 2) :
  (a = 1 ∧ b = -3 ∧ c = 2) → a * x * x + b * x + c = x^2 - 3 * x + 2 :=
by
  sorry

end quadratic_equation_divisible_by_x_minus_one_l102_102655


namespace focus_of_parabola_l102_102103

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end focus_of_parabola_l102_102103


namespace fuchs_can_relocate_all_cards_l102_102911

def shuffled_deck : Type := fin 52 → option (fin 52)

def is_free_spot (deck : shuffled_deck) (i : fin 52) : Prop :=
deck i = none

def move_card (deck : shuffled_deck) (card : fin 52) : shuffled_deck :=
deck.modify (λ i, if i = card - 1 ∨ i = card + 1 then deck card else none) card

theorem fuchs_can_relocate_all_cards :
  ∀ (deck : shuffled_deck), ∃ (moves : ℕ → fin 52),
  (∀ (n : ℕ), is_free_spot (move_card (deck ∘ moves) (moves n)) (moves n)) →
  (∃ (k : ℕ), ∀ i, deck i ≠ deck ((fin.of_nat (i + k)) % 52)) :=
begin
  sorry
end

end fuchs_can_relocate_all_cards_l102_102911


namespace double_average_l102_102165

theorem double_average (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : new_avg * n = 2 * (initial_avg * n)) : new_avg = 140 :=
sorry

end double_average_l102_102165


namespace even_positive_factors_count_l102_102354

def n : ℕ := 2^4 * 3^2 * 5 * 7

theorem even_positive_factors_count (n = 2^4 * 3^2 * 5 * 7) : ∃ count, count = 48 ∧ even_positive_factors_count n = count :=
by
  sorry

end even_positive_factors_count_l102_102354


namespace roots_are_distinct_and_negative_l102_102667

theorem roots_are_distinct_and_negative : 
  (∀ x : ℝ, x^2 + m * x + 1 = 0 → ∃! (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2) ↔ m > 2 :=
by
  sorry

end roots_are_distinct_and_negative_l102_102667


namespace evaluate_expression_l102_102996

theorem evaluate_expression (a b : ℕ) (ha : a = 64) (hb : b = 81) : 
  (a : ℝ) ^ (-1/2) + (b : ℝ) ^ (-1/4) = 11/24 := by
  sorry

end evaluate_expression_l102_102996


namespace ratio_of_male_alligators_l102_102767

variable (female_alligators adult_females male_alligators : ℕ)

def total_female_alligators (adult_females : ℕ) : ℕ :=
  adult_females / 0.60

def total_alligators (female_alligators male_alligators : ℕ) : ℕ :=
  female_alligators + male_alligators

noncomputable def male_to_total_ratio (male_alligators total_alligators : ℕ) : ℚ :=
  male_alligators / total_alligators

theorem ratio_of_male_alligators :
  (60% of female_alligators = adult_females) →
  40% ≥ 0 →
  (15 adult female = adult_females) →
  (25 male alligator = male_alligators) →
  male_to_total_ratio male_alligators (total_alligators (total_female_alligators adult_females) male_alligators) = 1 / 2 :=
by {
  sorry
}

end ratio_of_male_alligators_l102_102767


namespace problem_l102_102990

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

theorem problem :
  (∀ x, f (-x) = -f x) → -- f is odd
  (∀ x, f (x + 2) = -1 / f x) → -- Functional equation
  (∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) → -- Definition on interval (0,1)
  f (Real.log (54) / Real.log 3) = -3 / 2 := sorry

end problem_l102_102990


namespace find_a_and_intervals_l102_102304

def f (a x : ℝ) := a * Real.log (1 + x) + x^2 - 10 * x

def f_deriv (a x : ℝ) := (a / (1 + x)) + 2 * x - 10

theorem find_a_and_intervals (a : ℝ) (h : f_deriv a 3 = 0) :
  a = 16 ∧ 
  (∀ x : ℝ, 1 + x > 0 → (2 * x^2 - 8 * x + 6 > 0 ↔ x ∈ (-∞, 1) ∪ (3, ∞)) ∧
                        (2 * x^2 - 8 * x + 6 < 0 ↔ x ∈ (1, 3))) :=
sorry

end find_a_and_intervals_l102_102304


namespace largest_two_digit_prime_factor_of_binom_300_150_l102_102892

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  let n := Nat.choose 300 150 in
  ∃ p, Nat.Prime p ∧ (10 ≤ p ∧ p < 100) ∧ p ∣ n ∧ 
       (∀ q, Nat.Prime q → (10 ≤ q ∧ q < 100) → q ∣ n → q ≤ p) ∧ p = 89 := 
by
  -- Define n as the binomial coefficient
  let n := Nat.choose 300 150
  -- Proof will be filled here
  sorry

end largest_two_digit_prime_factor_of_binom_300_150_l102_102892


namespace minimum_circles_for_tangents_to_2011_gon_l102_102145

theorem minimum_circles_for_tangents_to_2011_gon :
  ∃ (n : ℕ), (∀ (C : set (set ℝ × ℝ)) (tangents : ℕ → set ℝ × ℝ),
    (∀ (i : ℕ), 1 ≤ i ∧ i < 2012 → tangents i ∈ C ∧ 
    ∃ (p1 p2: ℝ × ℝ), p1 ∈ C ∧ p2 ∈ C ∧ tangents i = (p1, p2)) → 
    (C.size ≥ 504 ∧ ∀ k, C.size_min k)) :=
begin
  sorry
end

end minimum_circles_for_tangents_to_2011_gon_l102_102145


namespace parabola_vertex_l102_102827

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ y : ℝ, y = 2 * (x - 5)^2 + 3) → (5, 3) = (5, 3) :=
by
  intros x y_eq
  sorry

end parabola_vertex_l102_102827


namespace matrix_solution_l102_102589

theorem matrix_solution (k : ℝ) (x : ℝ) :
  let a := (3 * x)
  let b := (5 * x)
  let c := 2
  let d := (2 * x)
  (a * b - c * d = k) ↔ 
  (∃ k : ℝ, k >= -4/15 ∧ x = (4 + Real.sqrt(16 + 60 * k)) / 30 ∨
                                x = (4 - Real.sqrt(16 + 60 * k)) / 30) :=
by
  sorry

end matrix_solution_l102_102589


namespace kicks_before_break_l102_102808

def total_kicks : ℕ := 98
def kicks_after_break : ℕ := 36
def kicks_needed_to_goal : ℕ := 19

theorem kicks_before_break :
  total_kicks - (kicks_after_break + kicks_needed_to_goal) = 43 := 
by
  -- proof wanted
  sorry

end kicks_before_break_l102_102808


namespace infinite_pairs_satisfying_eq_l102_102170

/-- Definition of the sequence f according to the recurrence relation. -/
noncomputable def f_seq (c : ℤ) (n : ℕ) : ℤ :=
  if h : n = 0 then 0 else
  if h : n = 1 then 1 else
  have h0 : n > 1 := nat.succ_le_succ_iff.mp (nat.succ_le_succ_iff.mp h),
  f_seq c (n - 1) * c - f_seq c (n - 2)

/-- Main theorem statement. -/
theorem infinite_pairs_satisfying_eq :
  (∃ c : ℤ, c > 2) → 
  (∃∞ (a b : ℤ), 
     a = f_seq c 2012 ∧ 
     b = -f_seq c 2011 ∧ 
     ∃ x y : ℝ, 
       x ≠ y ∧ 
       x * y = 1 ∧ 
       x^2012 = a * x + b ∧ 
       y^2012 = a * y + b) := by
  sorry

end infinite_pairs_satisfying_eq_l102_102170


namespace weight_of_box_is_correct_l102_102106

-- Definitions for outer dimensions and thickness of the box
def outer_length := 50  -- in cm
def outer_width := 40  -- in cm
def outer_height := 23  -- in cm
def thickness := 2  -- in cm
def weight_per_cm3 := 0.5  -- in grams

-- Definitions for inner dimensions
def inner_length := outer_length - 2 * thickness
def inner_width := outer_width - 2 * thickness
def inner_height := outer_height - thickness

-- Definitions for volumes
def outer_volume := outer_length * outer_width * outer_height
def inner_volume := inner_length * inner_width * inner_height
def volume_of_metal := outer_volume - inner_volume

-- Definition for the total weight of the metal used
def weight_of_box := volume_of_metal * weight_per_cm3

-- Statement to be proved
theorem weight_of_box_is_correct : weight_of_box = 5504 := by
  sorry

end weight_of_box_is_correct_l102_102106


namespace find_quotient_l102_102449

theorem find_quotient :
  ∃ q : ℕ, ∀ L S : ℕ, L = 1584 ∧ S = 249 ∧ (L - S = 1335) ∧ (L = S * q + 15) → q = 6 :=
by
  sorry

end find_quotient_l102_102449


namespace student_overlap_difference_l102_102732

theorem student_overlap_difference :
  ∀ (total geom bio : ℕ), total = 232 → geom = 144 → bio = 119 → 
  (∀ max_overlap, max_overlap = min geom bio) →
  (∀ min_overlap, min_overlap = max 0 (geom + bio - total)) →
  max_overlap = 119 → min_overlap = 31 → 
  max_overlap - min_overlap = 88 :=
by
  intros total geom bio ht hg hb hmax hmin hmaxval hminval
  rw [ht, hg, hb] at *
  have hmax’ := hmax 119
  rw [min_eq_left (le_of_lt (nat.lt_of_lt_of_le (nat.lt_of_succ_lt_succ $ nat.lt_of_succ_le $ nat.le_of_lt_succ $ nat.succ_lt_succ $ nat.lt_succ_iff.2 n119.le) $ match 144 with | 119 ⇒ dec_trivial | _ ⇒ all_alts_or_fail applies)).1]
  rw [hmaxval] at hmax’
  have hmin’ := hmin 31
  rw [max_eq_right (nat.sub_le _ _)] at hmin’
  rw [sub_eq_add_neg, nat.add_comm, hmax’, hminval]
  simp only [nat.add_sub_cancel, nat.add_comm, sub_self]
  exact rfl

end student_overlap_difference_l102_102732


namespace statement_1_statement_2_statement_4_l102_102991

theorem statement_1 :
  (¬ ∀ x : ℝ, x^2 - 2 < 3 * x) ↔ (∃ x : ℝ, x^2 - 2 ≥ 3 * x) :=
sorry

theorem statement_2 (a b : ℝ) :
  (2 ^ a < 2 ^ b) → (∀ a b, ∃ (c < 0), ¬ (c < 0)) → (log (1 / 2) a > log (1 / 2) b) ∧ log (1 / 2) a < 1 → 0 < a ∧ a < 3 ∧ 2 ^ a < 2 ^ b :=
sorry

theorem statement_4 (a b : ℝ) (θ : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hθ : θ = 2 * Real.pi / 3) :
  (|a + b| = real.sqrt 3) :=
sorry

end statement_1_statement_2_statement_4_l102_102991


namespace cos_pi_minus_alpha_l102_102646

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 7) : Real.cos (Real.pi - α) = - (1 / 7) := by
  sorry

end cos_pi_minus_alpha_l102_102646


namespace tangent_lines_count_l102_102485

theorem tangent_lines_count (C : ℝ → ℝ) (hC : ∀ x, C x = x^2 * Real.exp x) :
  ∃ n, n = 3 ∧ ∀ P, P = (2, 0) → ∃ k : ℕ, k = n ∧ ∀ i < k, ∃ x₀, (P.1 - x₀, C P.1 - C x₀) ∈ tangent_to P C :=
sorry

end tangent_lines_count_l102_102485


namespace polynomial_simplification_l102_102089

theorem polynomial_simplification (x : ℤ) :
  (5 * x ^ 12 + 8 * x ^ 11 + 10 * x ^ 9) + (3 * x ^ 13 + 2 * x ^ 12 + x ^ 11 + 6 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9) =
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 16 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9 :=
by
  sorry

end polynomial_simplification_l102_102089


namespace total_opponent_runs_l102_102916

theorem total_opponent_runs : 
  let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      lost_two_runs := [1, 3, 5, 7, 9, 11],
      won_double_runs := [2, 4, 6, 8, 10, 12]
  in  
  (list.sum (lost_two_runs.map (λ x, x + 2))
   + list.sum (won_double_runs.map (λ x, x / 2))) = 69 :=
by
  let team_scores := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let lost_two_runs := [1, 3, 5, 7, 9, 11]
  let won_double_runs := [2, 4, 6, 8, 10, 12]
  have H_lost : (list.sum (lost_two_runs.map (λ x, x + 2))) = 48 := sorry
  have H_won : (list.sum (won_double_runs.map (λ x, x / 2))) = 21 := sorry
  show (48 + 21) = 69 from
    by rw [H_lost, H_won]
    rfl

end total_opponent_runs_l102_102916


namespace compute_difference_of_squares_l102_102584

theorem compute_difference_of_squares :
  let a := 305
  let b := 295
  (a^2 - b^2) = 6000 :=
by
  let a := 305
  let b := 295
  have h1 : a + b = 600 := by
    sorry
  have h2 : a - b = 10 := by
    sorry
  calc
    a^2 - b^2 = (a + b) * (a - b) : by
      sorry
          ... = 600 * 10       : by
      sorry
          ... = 6000           : by
      sorry

end compute_difference_of_squares_l102_102584


namespace segment_ratio_ae_ad_l102_102418

/-- Given points B, C, and E lie on line segment AD, and the following conditions:
  1. The length of segment AB is twice the length of segment BD.
  2. The length of segment AC is 5 times the length of segment CD.
  3. The length of segment BE is one-third the length of segment EC.
Prove that the fraction of the length of segment AD that segment AE represents is 17/24. -/
theorem segment_ratio_ae_ad (AB BD AC CD BE EC AD AE : ℝ)
    (h1 : AB = 2 * BD)
    (h2 : AC = 5 * CD)
    (h3 : BE = (1/3) * EC)
    (h4 : AD = 6 * CD)
    (h5 : AE = 4.25 * CD) :
    AE / AD = 17 / 24 := 
  by 
  sorry

end segment_ratio_ae_ad_l102_102418


namespace every_positive_integer_has_good_multiple_l102_102211

def is_good (n : ℕ) : Prop :=
  ∃ (D : Finset ℕ), (D.sum id = n) ∧ (1 ∈ D) ∧ (∀ d ∈ D, d ∣ n)

theorem every_positive_integer_has_good_multiple (n : ℕ) (hn : n > 0) : ∃ m : ℕ, (m % n = 0) ∧ is_good m :=
  sorry

end every_positive_integer_has_good_multiple_l102_102211


namespace age_difference_l102_102205

variable (S M : ℕ)

theorem age_difference (hS : S = 28) (hM : M + 2 = 2 * (S + 2)) : M - S = 30 :=
by
  sorry

end age_difference_l102_102205


namespace vector_add_sub_l102_102530

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V] (A B C D : V)

theorem vector_add_sub : (C - B) + (A - D) - (A - B) = C - D := 
sorry

end vector_add_sub_l102_102530


namespace simplify_div_c1_c2_l102_102082

-- Define the complex numbers c1 as 7 + 18i and c2 as 4 - 5i
def c1 : ℂ := 7 + 18 * Complex.i
def c2 : ℂ := 4 - 5 * Complex.i
def answer : ℂ := - (62 : ℝ) / 41 + ((107 : ℝ) / 41) * Complex.i

-- Statement of the theorem
theorem simplify_div_c1_c2 : (c1 / c2) = answer := by
  sorry

end simplify_div_c1_c2_l102_102082


namespace least_subtraction_and_divisor_l102_102151

theorem least_subtraction_and_divisor (k : ℕ) (h : k = 4) :
  ∃ d, d = 675 ∧ (9679 - k) % d = 0 :=
by {
  use 675,
  split,
  { refl },
  {
    rw h,
    norm_num,
  }
}

end least_subtraction_and_divisor_l102_102151


namespace perfect_square_formula_l102_102157

theorem perfect_square_formula (x y : ℝ) :
  ¬∃ a b : ℝ, (x^2 + (1/4)*x + (1/4)) = (a + b)^2 ∧
  ¬∃ c d : ℝ, (x^2 + 2*x*y - y^2) = (c + d)^2 ∧
  ¬∃ e f : ℝ, (x^2 + x*y + y^2) = (e + f)^2 ∧
  ∃ g h : ℝ, (4*x^2 + 4*x + 1) = (g + h)^2 :=
sorry

end perfect_square_formula_l102_102157


namespace num_sweet_numbers_up_to_60_l102_102795

def double_or_subtract_15 (n : ℕ) : ℕ :=
  if n <= 30 then 2 * n else n - 15

def sequence (start : ℕ) : ℕ → ℕ
| 0       => start
| (n + 1) => double_or_subtract_15 (sequence n)

def is_sweet_number (G : ℕ) : Prop :=
  ∀ n, sequence G n ≠ 18

def count_sweet_numbers (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).filter (λ G => is_sweet_number G).card

theorem num_sweet_numbers_up_to_60 : count_sweet_numbers 60 = 56 :=
by
  sorry

end num_sweet_numbers_up_to_60_l102_102795


namespace incorrect_statement_D_l102_102326

def f (x : ℝ) : ℝ := sin x / (1 + cos x)
def g (x : ℝ) : ℝ := tan (x / 2)

theorem incorrect_statement_D :
  ∀ x, f x = g x →
  monotone_on g (Ioo π (2 * π)) → ¬monotone_on f (Ioo π (2 * π)) :=
by
  intros x h1 h2
  sorry

end incorrect_statement_D_l102_102326


namespace no_such_function_exists_l102_102413

-- Define the set S of integers greater than or equal to 2
def S : Set ℕ := {n | 2 ≤ n}

-- Define the functional property
def functional_property (f : ℕ → ℕ) : Prop :=
  ∀ a b ∈ S, a ≠ b → f(a) * f(b) = f(a^2 * b^2)

-- State the theorem
theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ a b ∈ S, a ≠ b → f(a) * f(b) = f(a^2 * b^2) := 
sorry

end no_such_function_exists_l102_102413


namespace rectangular_field_area_l102_102129

noncomputable def length (c : ℚ) : ℚ := 3 * c / 2
noncomputable def width (c : ℚ) : ℚ := 4 * c / 2
noncomputable def area (c : ℚ) : ℚ := (length c) * (width c)
noncomputable def field_area (c1 : ℚ) (c2 : ℚ) : ℚ :=
  let l := length c1
  let w := width c1
  if 25 * c2 = 101.5 * 100 then
    area c1
  else
    0

theorem rectangular_field_area :
  ∃ (c : ℚ), field_area c 25 = 10092 := by
  sorry

end rectangular_field_area_l102_102129


namespace proof_PA_sqrt_PB_QC_l102_102640

variable {T A B C P Q : Type}
variables [InCircle T A B C]

-- Given structure of tangent and secant
structure TangentSecant (T A B C : Type) where
  tangent_eq : ∀ (TA : T → A), secant_eq : ∀ (TB TC : T → B → C)

-- Given bisector condition
structure Bisector (A T C P Q : Type) where
  angle_bisector : ∀ (bisects : ∠ ATC / 2), intersects : ∀ (P Q : A → T → C)

-- Define segments and their properties
variable (PA PB QC : Type)

-- Main theorem to prove
theorem proof_PA_sqrt_PB_QC (h1 : TangentSecant T A B C)
  (h2 : Bisector A T C P Q)
  : ∀ PA PB QC, PA = Math.sqrt (PB * QC) :=
by
  sorry

end proof_PA_sqrt_PB_QC_l102_102640


namespace total_theme_parks_l102_102754

-- Define the constants based on the problem's conditions
def Jamestown := 20
def Venice := Jamestown + 25
def MarinaDelRay := Jamestown + 50

-- Theorem statement: Total number of theme parks in all three towns is 135
theorem total_theme_parks : Jamestown + Venice + MarinaDelRay = 135 := by
  sorry

end total_theme_parks_l102_102754


namespace sum_common_elements_eq_2870_l102_102569

def AP1 (n : ℕ) : ℕ :=
  3 + (n - 1) * 4

def AP2 (m : ℕ) : ℕ :=
  2 + (m - 1) * 7

noncomputable def CommonElements : List ℕ :=
  List.filter (λ x, x ≤ 399 && x % 4 == 3) (List.range 400)

theorem sum_common_elements_eq_2870 :
  List.sum CommonElements = 2870 :=
by
  sorry

end sum_common_elements_eq_2870_l102_102569


namespace mart_income_percentage_of_juan_l102_102063

theorem mart_income_percentage_of_juan
  (J T M : ℝ)
  (h1 : T = 0.60 * J)
  (h2 : M = 1.60 * T) :
  M = 0.96 * J :=
by 
  sorry

end mart_income_percentage_of_juan_l102_102063


namespace right_triangle_inscribed_circles_segments_length_l102_102751

theorem right_triangle_inscribed_circles_segments_length 
  (a b c : ℝ) (h_triangle: a^2 + b^2 = c^2) (r : ℝ) (h_r : r = 2) 
  (h_a : a = 10) (h_b : b = 24) (h_c : c = 26) :
  let inradius := (a * b) / (a + b + c),
  let total_length := (a - r) + (b - r) + (c - 2 * r) in
  inradius = r ∧ total_length = 52 := by
  sorry

end right_triangle_inscribed_circles_segments_length_l102_102751


namespace sum_of_exponents_outside_radical_l102_102084

theorem sum_of_exponents_outside_radical (x y z : ℝ) :
  let expr := 40 * x^5 * y^9 * z^14
  in let simplified_expr := 2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3)
  in (if (simplified_expr = (2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3))) then (1 + 1 + 3 = 5) else false) := sorry

end sum_of_exponents_outside_radical_l102_102084


namespace problem_a_problem_b_l102_102573

-- Part (a)
theorem problem_a {A B C : Point} :
  (∀ M : Point, M ∈ triangle ABC → MA + MB > MC ∧ MB + MC > MA ∧ MC + MA > MB) →
  (is_equilateral ABC) :=
by
  sorry

-- Part (b)
theorem problem_b {A B C M : Point} (h : is_equilateral ABC):
  (M ∈ triangle ABC → MA + MB > MC ∧ MB + MC > MA ∧ MC + MA > MB) :=
by
  sorry

end problem_a_problem_b_l102_102573


namespace incorrect_locus_statement_B_l102_102256

-- Definitions of conditions and locus
variable {α : Type*} (on_locus : α → Prop) (satisfies_conditions : α → Prop)

-- Statements considered
def Stmt_A := (∀ x, ¬ (on_locus x → satisfies_conditions x)) ∧ (∀ x, (on_locus x → satisfies_conditions x))
def Stmt_B := (∀ x, (satisfies_conditions x → on_locus x)) ∧ (∃ x, (on_locus x ∧ satisfies_conditions x))
def Stmt_C := (∀ x, (on_locus x → satisfies_conditions x)) ∧ (∀ x, (satisfies_conditions x → on_locus x))
def Stmt_D := (∀ x, ¬ (satisfies_conditions x → on_locus x)) ∧ (¬ ∃ x, satisfies_conditions x ∧ ¬ on_locus x)

-- Proof that Statement B is incorrect.
theorem incorrect_locus_statement_B : ¬ Stmt_B :=
sorry -- Proof omitted

end incorrect_locus_statement_B_l102_102256


namespace expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102627

-- Conditions
constant NumDice : ℕ := 5

-- Define the probability of a die not showing six on each roll
def prob_not_six := (5 : ℚ) / 6

-- Question a: Expected total number of dice rolled
theorem expected_total_dice_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 30
  sorry

-- Question b: Expected total number of points rolled
theorem expected_total_points_rolled :
  True := by
  -- The proof would calculate the expected value as derived, resulting in 105
  sorry

-- Question c: Expected number of salvos
theorem expected_number_of_salvos :
  True := by
  -- The proof would calculate the expected number of salvos as derived, resulting in 13.02
  sorry

end expected_total_dice_rolled_expected_total_points_rolled_expected_number_of_salvos_l102_102627


namespace arithmetic_sequence_a8_l102_102001

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 15 = 90) :
  a 8 = 6 :=
by
  sorry

end arithmetic_sequence_a8_l102_102001


namespace part1_minimum_value_part2_inequality_l102_102324

def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem part1_minimum_value : ∃ x, f x = 1 :=
sorry

theorem part2_inequality (x : ℝ) (h : 0 < x) : 
  x * Real.exp x * f x + (x * Real.exp x - Real.exp x) * Real.log x - Real.exp x + 1/2 > 0 :=
sorry

end part1_minimum_value_part2_inequality_l102_102324


namespace seats_per_bus_l102_102126

theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) : students / buses = 3 := by
  sorry

end seats_per_bus_l102_102126


namespace length_of_longer_train_l102_102885

noncomputable def speed_of_first_train_kph : ℝ := 60
noncomputable def speed_of_second_train_kph : ℝ := 40
noncomputable def length_of_shorter_train_m : ℝ := 250
noncomputable def time_to_cross_s : ℝ := 26.99784017278618

noncomputable def kph_to_mps (v_kph : ℝ) := v_kph * 5 / 18
noncomputable def speed_of_first_train_mps := kph_to_mps speed_of_first_train_kph
noncomputable def speed_of_second_train_mps := kph_to_mps speed_of_second_train_kph
noncomputable def relative_speed := speed_of_first_train_mps + speed_of_second_train_mps

theorem length_of_longer_train :
  let L2 := (time_to_cross_s * relative_speed) - length_of_shorter_train_m in
  L2 = 500 :=
by
  sorry

end length_of_longer_train_l102_102885


namespace soybean_cornmeal_proof_l102_102821

theorem soybean_cornmeal_proof :
  ∃ (x y : ℝ), 
    (0.14 * x + 0.07 * y = 0.13 * 280) ∧
    (x + y = 280) ∧
    (x = 240) ∧
    (y = 40) :=
by
  sorry

end soybean_cornmeal_proof_l102_102821


namespace johns_show_episodes_l102_102030

def episodes_in_show 
    (days : ℕ)
    (hours_per_day : ℕ)
    (minutes_per_episode : ℕ) : ℕ :=
by
    have total_minutes := days * hours_per_day * 60
    exact total_minutes / minutes_per_episode

theorem johns_show_episodes :
  episodes_in_show 5 2 30 = 20 := 
by
    unfold episodes_in_show
    rw [mul_assoc, Nat.mul_div_cancel_left, mul_comm (2 * 5), mul_assoc, mul_comm _ 60]
    simp
    sorry

end johns_show_episodes_l102_102030


namespace unique_laptop_ways_l102_102257

theorem unique_laptop_ways : 
  ∃ (ways : ℕ), ways = 15 * 14 * 13 ∧ ways = 2730 :=
begin
  use 15 * 14 * 13,
  split,
  { refl },
  { norm_num }
end

end unique_laptop_ways_l102_102257


namespace gamma_received_eight_donuts_l102_102248

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end gamma_received_eight_donuts_l102_102248


namespace smallest_positive_period_of_f_l102_102614

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x + cos x

theorem smallest_positive_period_of_f : ∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T = 2 * π :=
by
  sorry

end smallest_positive_period_of_f_l102_102614


namespace find_a_value_l102_102402

theorem find_a_value (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c)
  (h3 : a^2 - b^2 - c^2 + a * b = 2035)
  (h4 : a^2 + 3 * b^2 + 3 * c^2 - 3 * a * b - 2 * a * c - 2 * b * c = -2067) :
  a = 255 :=
by
  sorry

end find_a_value_l102_102402


namespace ratio_a7_b7_l102_102112

variable {α : Type*}
variables {a_n b_n : ℕ → α} [AddGroup α] [Field α]
variables {S_n T_n : ℕ → α}

-- Define the sum of the first n terms for sequences a_n and b_n
def sum_of_first_terms_a (n : ℕ) := S_n n = (n * (a_n n + a_n (n-1))) / 2
def sum_of_first_terms_b (n : ℕ) := T_n n = (n * (b_n n + b_n (n-1))) / 2

-- Given condition about the ratio of sums
axiom ratio_condition (n : ℕ) : S_n n / T_n n = (3 * n - 2) / (2 * n + 1)

-- The statement to be proved
theorem ratio_a7_b7 : (a_n 7 / b_n 7) = (37 / 27) := sorry

end ratio_a7_b7_l102_102112


namespace balls_in_boxes_l102_102477

theorem balls_in_boxes (balls : Fin 4) (boxes : Fin 3) :
  ∃ (n : ℕ), n = 42 ∧ 
  (∃ emp_box : boxes, ∀ b ∈ balls, b ≠ emp_box) := 
sorry

end balls_in_boxes_l102_102477


namespace student_overlap_difference_l102_102733

theorem student_overlap_difference :
  ∀ (total geom bio : ℕ), total = 232 → geom = 144 → bio = 119 → 
  (∀ max_overlap, max_overlap = min geom bio) →
  (∀ min_overlap, min_overlap = max 0 (geom + bio - total)) →
  max_overlap = 119 → min_overlap = 31 → 
  max_overlap - min_overlap = 88 :=
by
  intros total geom bio ht hg hb hmax hmin hmaxval hminval
  rw [ht, hg, hb] at *
  have hmax’ := hmax 119
  rw [min_eq_left (le_of_lt (nat.lt_of_lt_of_le (nat.lt_of_succ_lt_succ $ nat.lt_of_succ_le $ nat.le_of_lt_succ $ nat.succ_lt_succ $ nat.lt_succ_iff.2 n119.le) $ match 144 with | 119 ⇒ dec_trivial | _ ⇒ all_alts_or_fail applies)).1]
  rw [hmaxval] at hmax’
  have hmin’ := hmin 31
  rw [max_eq_right (nat.sub_le _ _)] at hmin’
  rw [sub_eq_add_neg, nat.add_comm, hmax’, hminval]
  simp only [nat.add_sub_cancel, nat.add_comm, sub_self]
  exact rfl

end student_overlap_difference_l102_102733


namespace packages_needed_l102_102765

/-- Kelly puts string cheeses in her kids' lunches 5 days per week. Her oldest wants 2 every day and her youngest will only eat 1.
The packages come with 30 string cheeses per pack. Prove that Kelly will need 2 packages of string cheese to fill her kids' lunches for 4 weeks. -/
theorem packages_needed (days_per_week : ℕ) (oldest_per_day : ℕ) (youngest_per_day : ℕ) (package_size : ℕ) (weeks : ℕ) :
  days_per_week = 5 →
  oldest_per_day = 2 →
  youngest_per_day = 1 →
  package_size = 30 →
  weeks = 4 →
  (2 * days_per_week + 1 * days_per_week) * weeks / package_size = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end packages_needed_l102_102765


namespace max_apartment_size_l102_102956

/-- Define the rental rate and the maximum rent Michael can afford. -/
def rental_rate : ℝ := 1.20
def max_rent : ℝ := 720

/-- State the problem in Lean: Prove that the maximum apartment size Michael should consider is 600 square feet. -/
theorem max_apartment_size :
  ∃ s : ℝ, rental_rate * s = max_rent ∧ s = 600 := by
  sorry

end max_apartment_size_l102_102956


namespace compute_f_l102_102782

noncomputable def f (x : ℝ) (hx : 1 < x) : ℝ :=
∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem compute_f (x : ℝ) (hx : 1 < x) : f x hx = 1 / (x - 1) :=
by
  sorry

end compute_f_l102_102782


namespace arccos_lt_arcsin_plus_pi_over_six_l102_102277

theorem arccos_lt_arcsin_plus_pi_over_six (x : ℝ) : 
    (arccos x < arcsin x + π / 6) ↔ (x ∈ Ioc (1 / sqrt 2) 1) :=
sorry

end arccos_lt_arcsin_plus_pi_over_six_l102_102277


namespace silk_per_dress_l102_102947

theorem silk_per_dress (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (total_dresses : ℕ)
  (h1 : initial_silk = 600)
  (h2 : friends = 5)
  (h3 : silk_per_friend = 20)
  (h4 : total_dresses = 100)
  (remaining_silk := initial_silk - friends * silk_per_friend) :
  remaining_silk / total_dresses = 5 :=
by
  -- proof goes here
  sorry

end silk_per_dress_l102_102947


namespace triangle_inequality_l102_102049

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l102_102049


namespace a_2009_eq_1_l102_102705

def a : ℕ → ℕ 
| 0 := 2
| 1 := 3
| (n+2) := abs (a (n+1) - a n)

theorem a_2009_eq_1 : a 2009 = 1 :=
by
  sorry

end a_2009_eq_1_l102_102705


namespace triangle_construction_l102_102243

theorem triangle_construction (a b c : ℝ) (α : ℝ) (h : (a^2 - c^2)^2 = b^2 * (2 * c^2 - b^2)) (ha : a = 5) (hα : α = 72) : 
  ∃ (β γ : ℝ), β + γ = 180 - α ∧ sin β = b/a ∧ sin γ = c/a :=
by
  sorry

end triangle_construction_l102_102243


namespace water_tank_capacity_l102_102153

-- Define the variables and conditions
variables (T : ℝ) (h : 0.35 * T = 36)

-- State the theorem
theorem water_tank_capacity : T = 103 :=
by
  -- Placeholder for proof
  sorry

end water_tank_capacity_l102_102153


namespace trapezoid_proof_l102_102800

noncomputable def distance (a b : Point) : ℝ := sorry

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Trapezoid :=
  (A B C D : Point)
  (AD_parallel_BC : (A.x = D.x → B.x = C.x))

def perimeter (a b c : Point) : ℝ :=
  distance a b + distance b c + distance c a

noncomputable def trapezoid_with_point (T : Trapezoid) (E : Point) (hE : T.A.x ≤ E.x ∧ E.x ≤ T.D.x) :=
  perimeter T.A T.B E = perimeter T.B T.C E ∧ perimeter T.C T.D E = perimeter T.B T.C E

theorem trapezoid_proof (T : Trapezoid) (E : Point) (hE : T.A.x ≤ E.x ∧ E.x ≤ T.D.x)
  (h_perim : trapezoid_with_point T E hE) : distance T.B T.C = (distance T.A T.D) / 2 :=
  sorry

end trapezoid_proof_l102_102800


namespace surface_area_with_holes_l102_102220

-- Define the cube and holes properties
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def number_faces_cube : ℕ := 6

-- Define areas
def area_face_cube := edge_length_cube ^ 2
def area_face_hole := side_length_hole ^ 2
def original_surface_area := number_faces_cube * area_face_cube
def total_hole_area := number_faces_cube * area_face_hole
def new_exposed_area := number_faces_cube * 4 * area_face_hole

-- Calculate the total surface area including holes
def total_surface_area := original_surface_area - total_hole_area + new_exposed_area

-- Lean statement for the proof
theorem surface_area_with_holes :
  total_surface_area = 168 := by
  sorry

end surface_area_with_holes_l102_102220


namespace sum_common_elements_in_arith_progressions_l102_102570

theorem sum_common_elements_in_arith_progressions :
  let
    a1 := 3; d1 := 4;
    b1 := 2; d2 := 7;
    first_ap (n : ℕ) := a1 + (n - 1) * d1;
    second_ap (m : ℕ) := b1 + (m - 1) * d2;
    common_elements :=
      {x : ℕ | ∃ n m, n ≤ 100 ∧ m ≤ 100 ∧ first_ap n = x ∧ second_ap m = x};
    sum_common := Finset.sum (common_elements.to_finset) id
  in sum_common = 2870 :=
by
  -- proof goes here
  sorry

end sum_common_elements_in_arith_progressions_l102_102570


namespace parabola_vertex_l102_102824

theorem parabola_vertex : ∃ h k : ℝ, (∀ x : ℝ, 2 * (x - h)^2 + k = 2 * (x - 5)^2 + 3) ∧ h = 5 ∧ k = 3 :=
by {
  use 5,
  use 3,
  split,
  { intro x,
    simp },
  exact ⟨rfl, rfl⟩,
}

end parabola_vertex_l102_102824


namespace mr_caiden_payment_l102_102066

theorem mr_caiden_payment (total_feet_needed : ℕ) (cost_per_foot : ℕ) (free_feet_supplied : ℕ) : 
  total_feet_needed = 300 → cost_per_foot = 8 → free_feet_supplied = 250 → 
  (total_feet_needed - free_feet_supplied) * cost_per_foot = 400 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end mr_caiden_payment_l102_102066


namespace constant_term_in_binomial_expansion_l102_102249

theorem constant_term_in_binomial_expansion :
  let T := (λ r, Nat.choose 6 r * (1 / 2) ^ (6 - r) * x^0) in
  T 4 = 15 / 4 :=
by
  -- Proof omitted, as per instructions
  sorry

end constant_term_in_binomial_expansion_l102_102249


namespace girls_in_school_play_l102_102873

theorem girls_in_school_play (G : ℕ) (boys : ℕ) (total_parents : ℕ)
  (h1 : boys = 8) (h2 : total_parents = 28) (h3 : 2 * boys + 2 * G = total_parents) : 
  G = 6 :=
sorry

end girls_in_school_play_l102_102873


namespace calculate_dot_product_l102_102321

noncomputable def function_f (x: ℝ) : ℝ := 2 * sin (π / 6 * x + π / 3)

def is_point_A (p: ℝ × ℝ) : Prop := p = (4, 0)

def are_points_B_C_symmetric (A B C: ℝ × ℝ) : Prop :=
  B.1 + C.1 = 8 ∧ B.2 + C.2 = 0

def dot_product (v1 v2: ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem calculate_dot_product (A B C: ℝ × ℝ)
  (hA: is_point_A A)
  (hBC_symm: are_points_B_C_symmetric A B C)
  (OB: ℝ × ℝ := B)
  (OC: ℝ × ℝ := C)
  (OA: ℝ × ℝ := A) :
  dot_product ((OB.1 + OC.1, OB.2 + OC.2)) OA = 32 := 
sorry

end calculate_dot_product_l102_102321


namespace expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102628

open ProbabilityTheory

/-- The expected total number of dice rolls until all five dice show sixes is 30 --/
theorem expected_total_number_of_dice_rolls (n : ℕ) (H : n = 5) : 
  (∑ k in range(n : ℕ), (E[(indicator(event_of_five_dice_roll)) k])) = 30 :=
sorry

/-- The expected total number of points rolled until all five dice show sixes is 105 --/
theorem expected_total_number_of_points (n : ℕ) (H : n = 5) : 
  (E[∑ i in range(n), points_rolled_until_six]) = 105 :=
sorry

/-- The expected number of salvos until all five dice show sixes is approximately 13.02 --/
theorem expected_number_of_salvos (n : ℕ) (H : n = 5) : 
  (E[number_of_salvos_to_get_all_sixes]) = 13.02 :=
sorry

end expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102628


namespace customers_in_each_car_l102_102470

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end customers_in_each_car_l102_102470


namespace total_pages_in_book_l102_102443

theorem total_pages_in_book (pages_monday pages_tuesday total_pages_read total_pages_book : ℝ)
    (h1 : pages_monday = 15.5)
    (h2 : pages_tuesday = 1.5 * pages_monday + 16)
    (h3 : total_pages_read = pages_monday + pages_tuesday)
    (h4 : total_pages_book = 2 * total_pages_read) :
    total_pages_book = 109.5 :=
by
  sorry

end total_pages_in_book_l102_102443


namespace hearts_per_card_l102_102478

-- Definitions of the given conditions
def num_suits := 4
def num_cards_total := 52
def num_cards_per_suit := num_cards_total / num_suits
def cost_per_cow := 200
def total_cost := 83200
def num_cows := total_cost / cost_per_cow

-- The mathematical proof problem translated to Lean 4:
theorem hearts_per_card :
    (2 * (num_cards_total / num_suits) = num_cows) → (num_cows = 416) → (num_cards_total / num_suits = 208) :=
by
  intros h1 h2
  sorry

end hearts_per_card_l102_102478


namespace gcd_5280_2155_l102_102839

theorem gcd_5280_2155 :
  let d1 := 5280 % 2155,
      q1 := 5280 / 2155,
      d2 := 2155 % d1,
      q2 := 2155 / d1,
      d3 := d1 % d2,
      q3 := d1 / d2,
      d4 := d2 % d3,
      q4 := d2 / d3,
      d5 := d3 % d4,
      q5 := d3 / d4,
      d6 := d4 % d5,
      q6 := d4 / d5
  in d1 = 970 ∧ q1 = 2 ∧
     d2 = 215 ∧ q2 = 2 ∧
     d3 = 110 ∧ q3 = 4 ∧
     d4 = 105 ∧ q4 = 1 ∧
     d5 = 5 ∧ q5 = 1 ∧
     d6 = 0 ∧ q6 = 21 →
     Nat.gcd 5280 2155 = 5 := by
  sorry

end gcd_5280_2155_l102_102839


namespace elgin_money_l102_102576

theorem elgin_money {A B C D E : ℤ} 
  (h1 : |A - B| = 19) 
  (h2 : |B - C| = 9) 
  (h3 : |C - D| = 5) 
  (h4 : |D - E| = 4) 
  (h5 : |E - A| = 11) 
  (h6 : A + B + C + D + E = 60) : 
  E = 10 := 
sorry

end elgin_money_l102_102576


namespace Quentin_chickens_l102_102433

variable (C S Q : ℕ)

theorem Quentin_chickens (h1 : C = 37)
    (h2 : S = 3 * C - 4)
    (h3 : Q + S + C = 383) :
    (Q = 2 * S + 32) :=
by
  sorry

end Quentin_chickens_l102_102433


namespace lucas_quiz_scores_l102_102421

theorem lucas_quiz_scores :
  ∃ (scores : List ℕ), 
   scores.sorted (≥) ∧
   scores.length = 5 ∧
   scores.sum = 84 * 5 ∧
   scores.nodup ∧
   (∀ score ∈ scores, score < 95) ∧
   scores = [94, 92, 91, 78, 65] :=
by
  sorry

end lucas_quiz_scores_l102_102421


namespace robot_distance_covered_l102_102933

theorem robot_distance_covered :
  let start1 := -3
  let end1 := -8
  let end2 := 6
  let distance1 := abs (end1 - start1)
  let distance2 := abs (end2 - end1)
  distance1 + distance2 = 19 := by
  sorry

end robot_distance_covered_l102_102933


namespace solve_medium_apple_cost_l102_102978

def cost_small_apple : ℝ := 1.5
def cost_big_apple : ℝ := 3.0
def num_small_apples : ℕ := 6
def num_medium_apples : ℕ := 6
def num_big_apples : ℕ := 8
def total_cost : ℝ := 45

noncomputable def cost_medium_apple (M : ℝ) : Prop :=
  (6 * cost_small_apple) + (6 * M) + (8 * cost_big_apple) = total_cost

theorem solve_medium_apple_cost : ∃ M : ℝ, cost_medium_apple M ∧ M = 2 := by
  sorry

end solve_medium_apple_cost_l102_102978


namespace max_value_of_f_on_S_l102_102811

noncomputable def S : Set ℝ := { x | x^4 - 13 * x^2 + 36 ≤ 0 }
noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_of_f_on_S : ∃ x ∈ S, ∀ y ∈ S, f y ≤ f x ∧ f x = 18 :=
by
  sorry

end max_value_of_f_on_S_l102_102811


namespace girls_attending_picnic_l102_102710

theorem girls_attending_picnic (g b : ℕ) (h1 : g + b = 1200) (h2 : (2 * g) / 3 + b / 2 = 730) : (2 * g) / 3 = 520 :=
by
  -- The proof steps would go here.
  sorry

end girls_attending_picnic_l102_102710


namespace proof_problem_l102_102015

-- Define natural numbers
def N := { n : ℕ | true }

-- Define the heaps [r]
def heap (r : ℕ) : set ℕ := { n | ∃ k ∈ N, n = 3 * k + r }

-- Define the conclusions
def conclusion1 := 2011 ∈ heap 1

def conclusion2 (a b : ℕ) : a ∈ heap 1 → b ∈ heap 2 → a + b ∈ heap 0 :=
  λ ha hb, exists.elim ha $ λ k hk, exists.elim hb $ λ m hm,
    ⟨k + m + 1, ⟨0, trivial⟩, by rw [add_assoc, add_comm, ←hm.2, ←hk.2, mul_add, add_mul, mul_one, mul_one]⟩

def conclusion3 := ∀ n ∈ N, n ∈ heap 0 ∨ n ∈ heap 1 ∨ n ∈ heap 2

def conclusion4 (a b : ℕ) : (∀ r, a ∈ heap r → b ∈ heap r → ¬(a - b) ∈ heap r) :=
  λ r ha hb, exists.elim ha $ λ ka hka', exists.elim hb $ λ kb hkb',
    match r with
    | 0 => (λ h, by rw [←hkb'.2, ←hka'.2, sub_eq_add_neg, add_mul, mul_one] at h; exact nat.not_lt_zero 1 h)
    | 1 => (λ h, by rw [←hkb'.2, ←hka'.2, sub_eq_add_neg, add_mul, mul_one] at h; exact nat.not_lt_zero 1 h)
    | 2 => (λ h, by rw [←hkb'.2, ←hka'.2, sub_eq_add_neg, add_mul, mul_one] at h; exact nat.not_lt_zero 1 h)
    end

theorem proof_problem : (∃ (C : ℕ), C = 3) :=
  by { existsi 3, sorry }


end proof_problem_l102_102015


namespace find_b_and_c_find_a_l102_102694

-- Define the function f(x)
def f (x a b c : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c

-- Hypotheses
variables {a : ℝ} {b : ℝ} {c : ℝ}

-- Condition: a > 0
def a_pos (h_a : a > 0) := h_a

-- Condition at point (0, f(0)), tangent line y = 1
def tangent_at_zero (h_tangent : f 0 a b c = 1 ∧ (x : ℝ) → (f x a b c)' = x^2 - a * x + b ∧ ((f x a b c)' x).eval 0 = 0) := h_tangent

-- Function has exactly two distinct zeros
def exactly_two_distinct_zeros (h_zeros : ∃! x1 x2 : ℝ, f x1 a 0 1 = 0 ∧ f x2 a 0 1 = 0 ∧ x1 ≠ x2) := h_zeros

-- Goals to prove
theorem find_b_and_c (h_tangent : tangent_at_zero (f 0 a b c = 1 ∧ f' 0 = 0)) : b = 0 ∧ c = 1 :=
by
  sorry

theorem find_a (h_zeros : exactly_two_distinct_zeros (∃! x1 x2 : ℝ, f x1 a 0 1 = 0 ∧ f x2 a 0 1 = 0 ∧ x1 ≠ x2)) : a = 36 :=
by
  sorry

end find_b_and_c_find_a_l102_102694


namespace pipe_fill_time_with_leak_l102_102429

theorem pipe_fill_time_with_leak (A L : ℝ) (hA : A = 1 / 2) (hL : L = 1 / 6) :
  (1 / (A - L)) = 3 :=
by
  sorry

end pipe_fill_time_with_leak_l102_102429


namespace remainder_of_division_l102_102975

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l102_102975


namespace mikey_initial_leaves_l102_102793

theorem mikey_initial_leaves (current_leaves : ℕ) (additional_leaves : ℕ) : 
  current_leaves = 468 → additional_leaves = 112 → current_leaves - additional_leaves = 356 :=
by
  intros hcurrent hadditional
  rw [hcurrent, hadditional]
  exact rfl

end mikey_initial_leaves_l102_102793


namespace odd_periodic_function_l102_102043

noncomputable def f (x : ℝ) := 
  if 0 < x ∧ x < 1 then x * (x + 1)
  else if -1 < x ∧ x < 0 then x * (1 - x)
  else if 5 < x ∧ x < 6 then (x - 6) * (7 - x)
  else if -6 < x ∧ x < -5 then (x + 6) * (7 + x)
  else 0 -- defining other values for completeness

theorem odd_periodic_function (x : ℝ) (h1 : 0 < x ∧ x < 1) (h2 : f(x) = x * (x + 1))
  (h3 : f(-x) = -f(x)) (h4 : f(x+2) = f(x) ∧ f(x-2) = f(x)) :
  ∀ (x : ℝ), 5 < x ∧ x < 6 → f(x) = (x-6)*(7-x) := 
by
  sorry

end odd_periodic_function_l102_102043


namespace drive_can_store_12_videos_l102_102210

theorem drive_can_store_12_videos :
  ∀ (storage_per_photo storage_per_video total_photos used_photos : ℕ), 
  storage_per_photo = 3 / 2 ∧ storage_per_video = 200 ∧ total_photos = 2000 ∧ used_photos = 400 →
  (let total_storage := storage_per_photo * total_photos in
   let used_storage := storage_per_photo * used_photos in
   let remaining_storage := total_storage - used_storage in
   let num_videos := remaining_storage / storage_per_video in
   num_videos) = 12 :=
begin
  intros storage_per_photo storage_per_video total_photos used_photos h,
  have h1 : storage_per_photo = 3 / 2, from h.1,
  have h2 : storage_per_video = 200, from h.2.1,
  have h3 : total_photos = 2000, from h.2.2.1,
  have h4 : used_photos = 400, from h.2.2.2,
  let total_storage := storage_per_photo * total_photos,
  let used_storage := storage_per_photo * used_photos,
  let remaining_storage := total_storage - used_storage,
  let num_videos := remaining_storage / storage_per_video,
  calc
    num_videos = (storage_per_photo * total_photos - storage_per_photo * used_photos) / storage_per_video
             : by rw [remaining_storage]
         ... = (3 / 2 * total_photos - 3 / 2 * used_photos) / storage_per_video
             : by rw [h1]
         ... = (3 / 2 * 2000 - 3 / 2 * 400) / 200
             : by rw [h3, h4, h2]
         ... = 2400 / 200
             : by norm_num
         ... = 12
             : by norm_num,
end

end drive_can_store_12_videos_l102_102210


namespace union_of_setA_and_setB_l102_102788

def setA : Set ℕ := {1, 2, 4}
def setB : Set ℕ := {2, 6}

theorem union_of_setA_and_setB :
  setA ∪ setB = {1, 2, 4, 6} :=
by sorry

end union_of_setA_and_setB_l102_102788


namespace second_batch_jelly_beans_weight_l102_102766

theorem second_batch_jelly_beans_weight (J : ℝ) (h1 : 2 * 3 + J > 0) (h2 : (6 + J) * 2 = 16) : J = 2 :=
sorry

end second_batch_jelly_beans_weight_l102_102766


namespace relationship_a_b_range_of_c_l102_102325

variable {a b c : ℝ}

-- Problem I: Relationship between a and b
theorem relationship_a_b (x₀ : ℝ) (h_tangent : (3 * x₀ ^ 2) - (2 * a * x₀) + b = 0) : a^2 ≥ 3 * b := 
  sorry

-- Problem II: Range of c
theorem range_of_c (h_extreme_values : ∀ x, x = -1 ∨ x = 3 → f' x = 0) 
  (h_intersects_x_axis : ∃ x1 x2 x3, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) 
  : -5 < c ∧ c < 27 := 
  sorry

-- Definition of the function f and its derivative f'
def f (x : ℝ) : ℝ := x^3 - a * x^2 + b * x + c
def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

end relationship_a_b_range_of_c_l102_102325


namespace sum_of_bn_l102_102775

noncomputable def geometric_sequence_sum (a1 a2 a3 : ℕ) (q : ℕ) := 
  ((a1 + a2 + a3 = 7) ∧ ((a1 + 3) + (a3 + 4)) / 2 = 3 * a2) → 
  a1 = 1 ∧ a2 = 2 ∧ a3 = 4 ∧ q = 2 ∧ 
  (∀ n : ℕ, a_n = 2^(n-1))

theorem sum_of_bn (b n : ℕ) := 
  let a := λ n, 2^(n-1) in
  let b := λ n, 1/(n*(n+1)) + a (2*n) in
  (∀ k : ℕ, T_n = ∑ i in i.range k, b i) ∧ 
  T_n = 2^(2*n + 1)/3 + 1/3 - 1/(n + 1) := 
  sorry

end sum_of_bn_l102_102775


namespace part1_part2_l102_102698

-- Define the function f(x)
def f (x : ℝ) : ℝ := (16 * x + 7) / (4 * x + 4)

-- Define sequences aₙ and bₙ
noncomputable def a (n : ℕ) : ℝ := if n = 0 then 3 else f (a (n - 1))
noncomputable def b (n : ℕ) : ℝ := if n = 0 then 4 else f (b (n - 1))

-- Part (1): Prove the range of a₁ such that ∀ n ∈ ℕ, a_{n+1} > aₙ
theorem part1 (a1 : ℝ) (h₁ : 0 < a1 ∧ a1 < 3) : ∀ (n : ℕ), a (n + 1) > a n :=
sorry

-- Part (2): Given a₁ = 3 and b₁ = 4, prove 0 < bₙ - aₙ ≤ 1 / 8^(n-1) for n = 1, 2, 3, ...
theorem part2 : ∀ (n : ℕ), 0 < b n - a n ∧ b n - a n ≤ 1 / 8^(n-1) :=
sorry

end part1_part2_l102_102698


namespace donation_amount_l102_102462

theorem donation_amount 
  (total_needed : ℕ) (bronze_amount : ℕ) (silver_amount : ℕ) (raised_so_far : ℕ)
  (bronze_families : ℕ) (silver_families : ℕ) (other_family_donation : ℕ)
  (final_push_needed : ℕ) 
  (h1 : total_needed = 750) 
  (h2 : bronze_amount = 25)
  (h3 : silver_amount = 50)
  (h4 : bronze_families = 10)
  (h5 : silver_families = 7)
  (h6 : raised_so_far = 600)
  (h7 : final_push_needed = 50)
  (h8 : raised_so_far = bronze_families * bronze_amount + silver_families * silver_amount)
  (h9 : total_needed - raised_so_far - other_family_donation = final_push_needed) : 
  other_family_donation = 100 :=
by
  sorry

end donation_amount_l102_102462


namespace problem_1_problem_2_l102_102409

-- Problem 1 conditions
def f (x a : ℝ) := exp(x) - a * x
def condition_1 (x0 a : ℝ) := (1 <= x0) ∧ f x0 a < exp(1) - a

-- Problem 1 statement
theorem problem_1 (a : ℝ) :
  (∃ x0 : ℝ, condition_1 x0 a) → a > Real.exp 1 :=
sorry

-- Problem 2 definitions
def p (x : ℝ) := exp(1 : ℝ) / x - log x
def q (x a : ℝ) := exp(x - 1) + a - log x
def closer_to_ln_x (s t x : ℝ) := abs (s - log x) ≤ abs (t - log x)

-- Problem 2 statement
theorem problem_2 (a : ℝ) (x : ℝ) :
  a > Real.exp 1 → x ≥ 1 → closer_to_ln_x (exp(1 : ℝ) / x) (exp(x - 1) + a) x :=
sorry

end problem_1_problem_2_l102_102409


namespace magic_trick_min_value_l102_102922

theorem magic_trick_min_value (n : ℕ) : ∃ n ≥ 2018, 
  ∀ (colors : Finset ℕ) (a : ℕ), 
    (∀ c ∈ colors, c ≤ 2017) → card_correct_strategy (a, colors.card) := sorry

end magic_trick_min_value_l102_102922


namespace prism_with_nonagon_base_has_27_edges_l102_102874

-- Define a nonagon as a polygon with 9 sides.
def nonagon : Prop := true

-- A prism with a nonagon base has two nonagon bases and 9 vertical edges
theorem prism_with_nonagon_base_has_27_edges (h : nonagon) : 
  ∃ vertical_edges top_edges bottom_edges, vertical_edges = 9 ∧ top_edges = 9 ∧ bottom_edges = 9 ∧ 
  (vertical_edges + top_edges + bottom_edges = 27) := 
by {
  existsi 9, -- vertical_edges
  existsi 9, -- top_edges
  existsi 9, -- bottom_edges
  split,
  { reflexivity },
  split,
  { reflexivity },
  split,
  { reflexivity },
  { reflexivity }, -- (9 + 9 + 9 = 27)
}

end prism_with_nonagon_base_has_27_edges_l102_102874


namespace packages_needed_l102_102764

/-- Kelly puts string cheeses in her kids' lunches 5 days per week. Her oldest wants 2 every day and her youngest will only eat 1.
The packages come with 30 string cheeses per pack. Prove that Kelly will need 2 packages of string cheese to fill her kids' lunches for 4 weeks. -/
theorem packages_needed (days_per_week : ℕ) (oldest_per_day : ℕ) (youngest_per_day : ℕ) (package_size : ℕ) (weeks : ℕ) :
  days_per_week = 5 →
  oldest_per_day = 2 →
  youngest_per_day = 1 →
  package_size = 30 →
  weeks = 4 →
  (2 * days_per_week + 1 * days_per_week) * weeks / package_size = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end packages_needed_l102_102764


namespace focus_of_parabola_l102_102104

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end focus_of_parabola_l102_102104


namespace trapezoid_area_YZRS_l102_102529

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

noncomputable def area_of_trapezoid (A B C D : Point) : ℝ :=
((A.x - B.x).abs * (A.y - B.y).abs + (C.x - D.x).abs * (C.y - D.y).abs) / 2

theorem trapezoid_area_YZRS :
  let P : Point := {x := 0, y := 0},
      Q : Point := {x := 8, y := 0},
      R : Point := {x := 8, y := 8},
      S : Point := {x := 0, y := 8},
      X : Point := midpoint P Q,
      Y : Point := midpoint X S,
      Z : Point := midpoint X R
  in area_of_trapezoid Y Z R S = 24 :=
by sorry

end trapezoid_area_YZRS_l102_102529


namespace valid_license_plates_count_l102_102556

theorem valid_license_plates_count :
  ∃ (n : ℕ), n = 26^3 * 10^4 ∧ n = 175760000 :=
by {
  use 175760000,
  split,
  { norm_num },
  { refl },
}

end valid_license_plates_count_l102_102556


namespace shooting_prob_l102_102550

theorem shooting_prob (p : ℝ) (h₁ : (1 / 3) * (1 / 2) * (1 - p) + (1 / 3) * (1 / 2) * p + (2 / 3) * (1 / 2) * p = 7 / 18) :
  p = 2 / 3 :=
sorry

end shooting_prob_l102_102550


namespace exists_number_between_70_and_80_with_gcd_10_l102_102838

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end exists_number_between_70_and_80_with_gcd_10_l102_102838


namespace circles_intersect_l102_102142

theorem circles_intersect (m c : ℝ) (h1 : (1:ℝ) = (5 + (-m))) (h2 : (3:ℝ) = (5 + (c - (-2)))) :
  m + c = 3 :=
sorry

end circles_intersect_l102_102142


namespace greatest_integer_value_l102_102889

theorem greatest_integer_value (x : ℤ) (h : ∃ x : ℤ, x = 29 ∧ ∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*(x+6) + 26)) :
  (∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*k + 26) → x = 29) :=
by
  sorry

end greatest_integer_value_l102_102889


namespace geom_seq_solution_l102_102398

-- Definition of the given conditions
def geom_seq (a₁: ℤ) (a₄: ℤ) (q: ℚ) : Prop :=
  a₄ = a₁ * q^3

-- Proposition to prove the common ratio q and the maximum n with conditions
theorem geom_seq_solution :
  ∃ (q: ℚ) (n : ℕ), geom_seq (-6) (-3/4) q ∧ q = 1/2 ∧ n = 4 :=
by
  exists 1/2
  exists 4
  dsimp [geom_seq]
  split
  sorry
  split
  refl
  simp

end geom_seq_solution_l102_102398


namespace sum_of_possible_values_l102_102355

theorem sum_of_possible_values (x : ℝ) (h : x^2 - 4 * x + 4 = 0) : x = 2 :=
sorry

end sum_of_possible_values_l102_102355


namespace largest_independent_amount_l102_102003

theorem largest_independent_amount (n : ℕ) :
  ∃ s, ¬∃ a b c d e f g h i j : ℕ, s = a * (3^n) + b * (3^(n-1) * 5) + c * (3^(n-2) * 5^2) + d * (3^(n-3) * 5^3) + 
        e * (3^(n-4) * 5^4) + f * (3^(n-5) * 5^5) + g * (3^(n-6) * 5^6) + h * (3^(n-7) * 5^7) + i * (3^(n-8) * 5^8) + 
        j * (5^n) := (5^(n+1)) - 2 * (3^(n+1)) :=
sorry

end largest_independent_amount_l102_102003


namespace min_max_values_on_interval_l102_102844

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l102_102844


namespace sqrt_nested_expression_l102_102585

theorem sqrt_nested_expression (x : ℝ) (hx : 0 ≤ x) : 
    (sqrt (x * sqrt (x * sqrt (x * sqrt x )))) = x * (x ^ (3 / 4)) :=
sorry

end sqrt_nested_expression_l102_102585


namespace system_of_equations_solution_l102_102819

theorem system_of_equations_solution (x y z : ℝ) :
  (x * y + x * z = 8 - x^2) →
  (x * y + y * z = 12 - y^2) →
  (y * z + z * x = -4 - z^2) →
  (x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1) :=
by
  sorry

end system_of_equations_solution_l102_102819


namespace BQ_not_perpendicular_to_PD_l102_102014

def S_ABCD_edges_length_2 : Prop := ∀ (A B C D S : ℝ × ℝ × ℝ), 
  dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧
  dist S A = 2 ∧ dist S B = 2 ∧ dist S C = 2 ∧ dist S D = 2

def midpoint (P A S : ℝ × ℝ × ℝ) : Prop := 
  P = ((A.1 + S.1) / 2, (A.2 + S.2) / 2, (A.3 + S.3) / 2)

def on_edge (Q S C : ℝ × ℝ × ℝ) (m : ℝ) : Prop := 
  Q = (S.1 * (1 - m/2) + C.1 * (m/2), S.2 * (1 - m/2) + C.2 * (m/2), S.3 * (1 - m/2) + C.3 * (m/2))

theorem BQ_not_perpendicular_to_PD 
  (A B C D S P Q : ℝ × ℝ × ℝ) (m : ℝ)
  (h_edges : S_ABCD_edges_length_2)
  (h_midpoint : midpoint P A S)
  (h_on_edge : on_edge Q S C m) 
  (h_range : 0 ≤ m ∧ m ≤ 2) :
  ¬(vector.dot (P - D) (Q - B) = 0) := sorry

end BQ_not_perpendicular_to_PD_l102_102014


namespace min_max_values_of_f_l102_102852

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x >= - (3 * Real.pi / 2)) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = - (3 * Real.pi / 2)) ∧
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x <= Real.pi / 2 + 2) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = Real.pi / 2 + 2) :=
by {
  -- Proof omitted
  sorry
}

end min_max_values_of_f_l102_102852


namespace min_factors_to_erase_l102_102110

theorem min_factors_to_erase (x : ℝ) : 
  ( ∀ (f : ℝ → ℝ), 
    f = (fun x => (x - 1) * (x - 2) * ... * (x - 2016)) →
    ( ∃ (n : ℕ), n = 2016 ∧ 
      ( ∀ (g : ℝ → ℝ), 
        g = (fun x => ( (x - (some el1)) * (x - (some el2)) * ... * (x - (some el(m-n))) )) 
        → ¬ ∃ (r : ℝ), g r = 0 ) )
  ) := 
  sorry

end min_factors_to_erase_l102_102110


namespace area_ratio_l102_102552

-- Definitions of points and distances as given in the problem
variables {A B C D A₁ B₁ C₁ D₁ : Type}
variables (radius : ℝ) (Omega : circle ℝ) (AD BC : ℝ)

-- Conditions explicitly stated in the problem
axiom trapezoid_inscribed : inscribed_in_trapezoid Omega A B C D
axiom rectangle_inscribed : inscribed_in_rectangle Omega A₁ B₁ C₁ D₁
axiom parallel_AD_BC : AD ∥ BC
axiom parallel_AC_B₁D₁ : AC ∥ B₁D₁
axiom parallel_BD_A₁C₁ : BD ∥ A₁C₁
axiom AD_length : AD = 24
axiom BC_length : BC = 10
axiom circle_radius : Omega.radius = 13

-- Assert the ratio of the areas
theorem area_ratio :
  let area_trapezoid := 0.5 * (AD + BC) * height_of_trapezoid AD BC radius
  let area_rectangle := length_diagonal₁ Omega * length_diagonal₂ Omega
  in area_trapezoid / area_rectangle = 1 / 2 := sorry

end area_ratio_l102_102552


namespace triangle_range_of_ab_cos_C_l102_102749

theorem triangle_range_of_ab_cos_C 
(ABC : ∀ {α β γ : Type}, (α → β → γ) → Prop)
(A B C : Type)
(a b : Real)
(h1 : a + b = 10)
(h2 : AB = 6)
(h3 : 2 < a ∧ a < 8) :
7 ≤ a * b * Real.cos (angle ABC A B C) ∧ a * b * Real.cos (angle ABC A B C) < 16 :=
sorry

end triangle_range_of_ab_cos_C_l102_102749


namespace value_of_t_for_x_equals_y_l102_102161

theorem value_of_t_for_x_equals_y (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : 
    t = 1 / 2 → x = y :=
by 
  intro ht
  rw [ht] at h1 h2
  sorry

end value_of_t_for_x_equals_y_l102_102161


namespace exists_number_between_70_and_80_with_gcd_10_l102_102837

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end exists_number_between_70_and_80_with_gcd_10_l102_102837


namespace gamma_donuts_received_l102_102246

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end gamma_donuts_received_l102_102246


namespace smallest_number_of_rectangles_needed_l102_102504

theorem smallest_number_of_rectangles_needed :
  ∃ n, (n * 12 = 144) ∧ (∀ k, (k * 12 = 144) → k ≥ n) := by
  sorry

end smallest_number_of_rectangles_needed_l102_102504


namespace customers_in_each_car_l102_102469

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end customers_in_each_car_l102_102469


namespace tetrahedron_dihedral_angle_l102_102295

noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x

theorem tetrahedron_dihedral_angle
  (α β : ℝ)
  (hα1 : 0 < α)
  (hα2 : α < Real.pi / 2)
  (hβ1 : 0 < β)
  (hβ2 : β < Real.pi / 2) :
  let θ := Real.pi - Real.arccos (cot α * cot β) in
  θ = Real.pi - Real.arccos (cot α * cot β) :=
by
  sorry

end tetrahedron_dihedral_angle_l102_102295


namespace min_distance_from_origin_to_line_l102_102376

/-- Prove that the minimum distance from the origin to the line described by 
  ρ sin θ = ρ cos θ + 2 is √2 in polar coordinates. -/
theorem min_distance_from_origin_to_line (ρ θ : ℝ) : 
  ∀ θ, ∃ A, A ∈ { A : ℝ × ℝ | A.2 = A.1 + 2 } ∧ 
         |(A.1, A.2)| = √2 :=
by sorry

end min_distance_from_origin_to_line_l102_102376


namespace min_max_f_on_interval_l102_102855

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_f_on_interval :
  ∃ min max, min = - (3 * Real.pi) / 2 ∧ max = (Real.pi / 2) + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min ∧ f x ≤ max) :=
sorry

end min_max_f_on_interval_l102_102855


namespace cevian_ratio_l102_102006

theorem cevian_ratio
  (A B C D E F : Type*)
  (h1 : dist A B = 150)
  (h2 : dist A C = 150)
  (h3 : dist A D = 50)
  (h4 : dist C F = 100)
  (h5 : collinear A B C)
  (h6 : collinear A C F)
  (h7 : collinear A B D)
  (h8 : collinear C D F) :
  let [CEF] := area (triangle C E F)
  let [DBE] := area (triangle D B E) 
  in [CEF] / [DBE] = 5 := 
sorry

end cevian_ratio_l102_102006


namespace sum_sequence_eq_l102_102658

noncomputable def sequence (a_n : ℕ → ℝ) : Prop :=
∀ n, (3 - a_n (n + 1)) * (6 + a_n n) = 18

theorem sum_sequence_eq (a : ℕ → ℝ) (h_seq : sequence a) (h_init : a 0 = 3) :
  ∀ n, ∑ i in Finset.range (n+1), 1 / a i = 1 / 3 * (2^(n+2) - n - 3) :=
by
  sorry

end sum_sequence_eq_l102_102658


namespace expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102630

open ProbabilityTheory

/-- The expected total number of dice rolls until all five dice show sixes is 30 --/
theorem expected_total_number_of_dice_rolls (n : ℕ) (H : n = 5) : 
  (∑ k in range(n : ℕ), (E[(indicator(event_of_five_dice_roll)) k])) = 30 :=
sorry

/-- The expected total number of points rolled until all five dice show sixes is 105 --/
theorem expected_total_number_of_points (n : ℕ) (H : n = 5) : 
  (E[∑ i in range(n), points_rolled_until_six]) = 105 :=
sorry

/-- The expected number of salvos until all five dice show sixes is approximately 13.02 --/
theorem expected_number_of_salvos (n : ℕ) (H : n = 5) : 
  (E[number_of_salvos_to_get_all_sixes]) = 13.02 :=
sorry

end expected_total_number_of_dice_rolls_expected_total_number_of_points_expected_number_of_salvos_l102_102630


namespace compute_a_plus_b_l102_102779

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end compute_a_plus_b_l102_102779


namespace introspective_for_every_element_l102_102814

noncomputable def introspective (m : ℕ) (P : polynomial ℤ) : Prop := sorry
variable (G1 H1 : set ℕ)
variables (n λ : ℕ)
def P (a : ℕ) [h : 1 ≤ a ∧ a ≤ λ] : polynomial ℤ := polynomial.X + polynomial.C a

theorem introspective_for_every_element :
  (∀ g ∈ G1, ∀ h ∈ H1, introspective g h) :=
begin
  -- Given conditions
  assume g hg,
  assume h hh,
  -- Conditions and polynomial form P
  have hypothesis_n : ∀ a, 1 ≤ a → a ≤ λ → introspective n (P a),
  { sorry },
  sorry
end

end introspective_for_every_element_l102_102814


namespace intersection_A_complementB_l102_102688

def R : Set ℝ := set.univ

def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}

def B : Set ℝ := {x | x + 1 < 0}

def complement_B : Set ℝ := {x | x ≥ -1}

theorem intersection_A_complementB :
  A ∩ complement_B = {x | -1 ≤ x ∧ x ≤ 2} :=
sorry

end intersection_A_complementB_l102_102688


namespace planet_not_observed_l102_102799

theorem planet_not_observed (k : ℕ) (d : Fin (2*k+1) → Fin (2*k+1) → ℝ) 
  (h_d : ∀ i j : Fin (2*k+1), i ≠ j → d i i = 0 ∧ d i j ≠ d i i) 
  (h_astronomer : ∀ i : Fin (2*k+1), ∃ j : Fin (2*k+1), j ≠ i ∧ ∀ k : Fin (2*k+1), k ≠ i → d i j < d i k) : 
  ∃ i : Fin (2*k+1), ∀ j : Fin (2*k+1), i ≠ j → ∃ l : Fin (2*k+1), (j ≠ l ∧ d l i < d l j) → false :=
  sorry

end planet_not_observed_l102_102799


namespace original_number_is_64_l102_102944

theorem original_number_is_64 (x : ℕ) : 500 + x = 9 * x - 12 → x = 64 :=
by
  sorry

end original_number_is_64_l102_102944


namespace place_numbers_sequentially_l102_102912

def is_arithmetic_mean (A : List ℤ) (i j k : ℕ) : Prop :=
  2 * A.nth_le j (by linarith) = A.nth_le i (by linarith) + A.nth_le k (by linarith)

def regular_nonagon_condition (A : List ℤ) : Prop :=
  (∀ i, is_arithmetic_mean A i ((i+3) % 9) ((i+6) % 9))

theorem place_numbers_sequentially : 
  regular_nonagon_condition [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] :=
by
  sorry

end place_numbers_sequentially_l102_102912


namespace min_max_values_l102_102846

noncomputable def f (x : ℝ) := cos x + (x + 1) * sin x + 1

theorem min_max_values :
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≥ - (3 * π) / 2) ∧
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≤ (π / 2 + 2)) :=
sorry

end min_max_values_l102_102846


namespace find_b_l102_102707

noncomputable def z1 : ℂ := 1 + complex.i
noncomputable def z2 (b : ℂ) : ℂ := 2 + b * complex.i

-- Theorem: If the product of the complex numbers z1 and z2 is a real number, then b = -2.
theorem find_b (b : ℝ) (h : (z1 * (z2 b)).im = 0) : b = -2 :=
sorry

end find_b_l102_102707


namespace sin_sgn_eq_sin_abs_l102_102786

def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x = 0 then 0 else -1

theorem sin_sgn_eq_sin_abs (x : ℝ) : (Real.sin x) * (sgn x) = Real.sin (abs x) := by
  sorry

end sin_sgn_eq_sin_abs_l102_102786


namespace infinite_series_sum_eq_l102_102240

theorem infinite_series_sum_eq : 
  (∑' n : ℕ, if n = 0 then 0 else ((1 : ℝ) / (n * (n + 3)))) = (11 / 18 : ℝ) :=
sorry

end infinite_series_sum_eq_l102_102240


namespace marching_band_members_l102_102115

theorem marching_band_members (x : ℕ) : 
  (150 < x ∧ x < 250) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 2) → 
  x = 163 :=
by
  intros,
  sorry

end marching_band_members_l102_102115


namespace smallest_number_of_coins_l102_102491

theorem smallest_number_of_coins :
  ∃ pennies nickels dimes quarters half_dollars : ℕ,
    pennies + nickels + dimes + quarters + half_dollars = 6 ∧
    (∀ amount : ℕ, amount < 100 →
      ∃ p n d q h : ℕ,
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧ h ≤ half_dollars ∧
        1 * p + 5 * n + 10 * d + 25 * q + 50 * h = amount) :=
sorry

end smallest_number_of_coins_l102_102491


namespace fourier_budan_l102_102039

def numSignChanges (s : List ℝ) : ℕ :=
  s.pairwise (λ x y, x * y < 0) |> count id

def N (f : ℝ → ℝ) (x : ℝ) : ℕ :=
  numSignChanges (List.map (λ m, (derivative^[m]) f x) (List.range (polynomial.natDegree f)))

theorem fourier_budan (f : ℝ → ℝ) (a b : ℝ) (h₁ : a < b) (h₂ : f a ≠ 0) (h₃ : f b ≠ 0) :
  ∃ k : ℕ, k ≤ N f a - N f b ∧ ∃ m : ℕ, k = 2*m :=
sorry

end fourier_budan_l102_102039


namespace woman_working_days_l102_102902

-- Define the conditions
def man_work_rate := 1 / 6
def boy_work_rate := 1 / 18
def combined_work_rate := 1 / 4

-- Question statement in Lean 4
theorem woman_working_days :
  ∃ W : ℚ, (man_work_rate + W + boy_work_rate = combined_work_rate) ∧ (1 / W = 1296) :=
sorry

end woman_working_days_l102_102902


namespace unique_function_solution_l102_102612

noncomputable def f (x : ℝ) : ℝ := sorry

theorem unique_function_solution :
  (∀ x y : ℝ, f(x + y) * f(x - y) = (f(x) + f(y))^2 - 4 * x^2 * f(y) + 4 * y^2) →
  (∀ x : ℝ, f(x) = x^2) :=
by sorry

end unique_function_solution_l102_102612


namespace print_shop_Y_charge_l102_102637

variable (y : ℝ)
variable (X_charge_per_copy : ℝ := 1.25)
variable (Y_charge_for_60_color_copies : ℝ := 60 * y)
variable (X_charge_for_60_color_copies : ℝ := 60 * X_charge_per_copy)
variable (additional_cost_at_Y : ℝ := 90)

theorem print_shop_Y_charge :
  Y_charge_for_60_color_copies = X_charge_for_60_color_copies + additional_cost_at_Y → y = 2.75 := by
  sorry

end print_shop_Y_charge_l102_102637


namespace solve_for_q_l102_102267

theorem solve_for_q (q : ℕ) : 16^4 = (8^3 / 2 : ℕ) * 2^(16 * q) → q = 1 / 2 :=
by
  sorry

end solve_for_q_l102_102267


namespace find_expression_value_l102_102806

variable (x y z : ℚ)
variable (h1 : x - y + 2 * z = 1)
variable (h2 : x + y + 4 * z = 3)

theorem find_expression_value : x + 2 * y + 5 * z = 4 := 
by {
  sorry
}

end find_expression_value_l102_102806


namespace width_of_green_arms_l102_102549

theorem width_of_green_arms (s : ℝ) (a_cross : ℝ) (a_green : ℝ) : 
  s = 1 ∧ a_cross = 0.4 ∧ a_green = 0.32 → 
  let w := 0.5 - (sqrt (a_cross - a_green) / 2) in
  abs (w - 0.3586) < 0.0001 :=
by
  sorry

end width_of_green_arms_l102_102549


namespace area_of_curve_l102_102579

theorem area_of_curve (a : ℝ) (ha : a > 0) :
  let curve := {p : ℝ × ℝ | (p.1^2 + p.2^2)^2 = 4 * a * p.2^3} in
  (area curve) = (5 * π * a^2 / 2) := 
sorry

end area_of_curve_l102_102579


namespace combined_alloy_tin_amount_l102_102178

theorem combined_alloy_tin_amount
  (weight_A weight_B weight_C : ℝ)
  (ratio_lead_tin_A : ℝ)
  (ratio_tin_copper_B : ℝ)
  (ratio_copper_tin_C : ℝ)
  (amount_tin : ℝ) :
  weight_A = 150 → weight_B = 200 → weight_C = 250 →
  ratio_lead_tin_A = 5/3 → ratio_tin_copper_B = 2/3 → ratio_copper_tin_C = 4 →
  amount_tin = ((3/8) * weight_A) + ((2/5) * weight_B) + ((1/5) * weight_C) →
  amount_tin = 186.25 :=
by sorry

end combined_alloy_tin_amount_l102_102178


namespace hyperbola_eccentricity_l102_102700

-- Conditions
def hyperbola (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

def circle (r : ℝ) (h_r_pos : 0 < r) : Prop :=
  ∀ (x y : ℝ), ((x - 3)^2 + y^2 = r^2)

-- Given condition
def asymptote_intersects (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : Prop :=
  ∃ (x y : ℝ), (b * x - a * y = 0 ∧ (x-3)^2 + y^2 = 9)

-- Proof statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_rel : b = sqrt 2 * a)
  (h_asymptote : asymptote_intersects a b h_a_pos h_b_pos): 
  eccentricity a (3 * a) = 3 :=
sorry

-- Eccentricity definition
def eccentricity (a c : ℝ) (h_a_pos : 0 < a) (h_c_pos : 0 < c) : ℝ :=
c / a

end hyperbola_eccentricity_l102_102700


namespace student_overlap_difference_l102_102734

theorem student_overlap_difference :
  ∀ (total geom bio : ℕ), total = 232 → geom = 144 → bio = 119 → 
  (∀ max_overlap, max_overlap = min geom bio) →
  (∀ min_overlap, min_overlap = max 0 (geom + bio - total)) →
  max_overlap = 119 → min_overlap = 31 → 
  max_overlap - min_overlap = 88 :=
by
  intros total geom bio ht hg hb hmax hmin hmaxval hminval
  rw [ht, hg, hb] at *
  have hmax’ := hmax 119
  rw [min_eq_left (le_of_lt (nat.lt_of_lt_of_le (nat.lt_of_succ_lt_succ $ nat.lt_of_succ_le $ nat.le_of_lt_succ $ nat.succ_lt_succ $ nat.lt_succ_iff.2 n119.le) $ match 144 with | 119 ⇒ dec_trivial | _ ⇒ all_alts_or_fail applies)).1]
  rw [hmaxval] at hmax’
  have hmin’ := hmin 31
  rw [max_eq_right (nat.sub_le _ _)] at hmin’
  rw [sub_eq_add_neg, nat.add_comm, hmax’, hminval]
  simp only [nat.add_sub_cancel, nat.add_comm, sub_self]
  exact rfl

end student_overlap_difference_l102_102734


namespace sum_alpha_eq_n_l102_102347

theorem sum_alpha_eq_n {n : ℕ} (α : fin n → ℝ) (h : ∑ i, real.arccos (α i) = 0) : 
  ∑ i, α i = n := 
by 
  sorry

end sum_alpha_eq_n_l102_102347


namespace count_correct_statements_l102_102454

def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem count_correct_statements :
  let s1 := f (11 * Real.pi / 12) = -3
  let s2 := f (2 * Real.pi / 3) = 0
  let s3 := (∀ x : ℝ, 3 * Real.sin (2 * x) = f (x - Real.pi / 3) → false)
  let s4 := ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ 5 * Real.pi / 12 → f x ≤ f (x + 1) in
  (cond (s1 ∧ s2 ∧ s4) → (count true [s1, s2, s3, s4] = 3))

where count := λ (b : Bool) (lst : List Bool), lst.foldr (λ t acc, if b then acc + 1 else acc) 0 := sorry

end count_correct_statements_l102_102454


namespace christine_makes_two_cakes_l102_102238

theorem christine_makes_two_cakes (tbsp_per_egg_white : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (total_tbsp_aquafaba : ℕ)
  (h1 : tbsp_per_egg_white = 2) 
  (h2 : egg_whites_per_cake = 8) 
  (h3 : total_tbsp_aquafaba = 32) : 
  total_tbsp_aquafaba / tbsp_per_egg_white / egg_whites_per_cake = 2 := by 
  sorry

end christine_makes_two_cakes_l102_102238


namespace g_of_12_div_35_l102_102051

theorem g_of_12_div_35 (g : ℚ+ → ℤ)
  (h₁: ∀ a b : ℚ+, g (a * b) = g a + g b)
  (h₂: ∀ p : ℚ+, p.1.is_prime → g p = 2 * p) :
  g (⟨12, by norm_num⟩ / ⟨35, by norm_num⟩) = -10 :=
sorry

end g_of_12_div_35_l102_102051


namespace string_cheese_packages_l102_102762

theorem string_cheese_packages (days_per_week : ℕ) (weeks : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) (pack_size : ℕ) 
    (H1 : days_per_week = 5)
    (H2 : weeks = 4)
    (H3 : oldest_daily = 2)
    (H4 : youngest_daily = 1)
    (H5 : pack_size = 30) 
  : (oldest_daily * days_per_week + youngest_daily * days_per_week) * weeks / pack_size = 2 :=
  sorry

end string_cheese_packages_l102_102762


namespace angle_bisector_AC_l102_102171

-- Definitions of the given conditions in the problem
variable {O O' A B C D : Point} -- O is the center of the smaller circle, O' the center of the larger circle.
variable {r r' : ℝ} -- r is the radius of the smaller circle, r' the radius of the larger circle
variable {A_onCircle : PointOnCircle A O r} -- A is a point on the smaller circle
variable {B_onCircle : PointOnCircle B O' r'} -- B is a point on the larger circle
variable {D_onCircle : PointOnCircle D O' r'} -- D is a point on the larger circle
variable {BD_chord : ChordOnCircle B D O' r'} -- BD is a chord of the larger circle
variable {C_tangent : TangentToCircleAt C O r BD} -- BD touches the smaller circle at C

-- Statement to prove
theorem angle_bisector_AC : 
  isAngleBisector A C (angle B A D) :=
sorry

end angle_bisector_AC_l102_102171


namespace denise_spending_l102_102031

theorem denise_spending : 
  ∃ D : ℕ, (D = 14 ∨ D = 17) ∧ (∃ J : ℕ, (J = D + 6) ∧ (
    (∃ D_i : ℕ, D_i ∈ {7, 11, 14} ∧ ∃ V_j : ℕ, V_j ∈ {6, 7, 9} ∧ J = D_i + V_j) ∧
    (∃ D_k : ℕ, D_k ∈ {7, 11, 14} ∧ ∃ V_l : ℕ, V_l ∈ {6, 7, 9} ∧ D = D_k + V_l)
  )) :=
by
  sorry

end denise_spending_l102_102031


namespace max_value_of_m_l102_102403

theorem max_value_of_m {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 20) :
  ∃ m, m = min (a * b) (min (b * c) (c * a)) ∧ m = 12 :=
by
  sorry

end max_value_of_m_l102_102403


namespace range_a_A_intersect_B_not_empty_l102_102329
noncomputable def range_of_a (a : ℝ) (x : ℝ) :=
  (0 < x) → (a - 5) / x < |1 + 1 / x| - |1 - 2 / x| ∧ |1 + 1 / x| - |1 - 2 / x| < (a + 2) / x

theorem range_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → (range_of_a a x)) : 1 < a ∧ a < 8 :=
begin
  sorry
end

def A (a x : ℝ) := |x - 1| + |x + 1| ≤ a
def B (x : ℝ) := 4 ≤ 2 ^ x ∧ 2 ^ x ≤ 8

theorem A_intersect_B_not_empty (a : ℝ) (h1 : ∀ x : ℝ, A a x) (h2 : ∀ x : ℝ, B x) :
  ¬(∀ x : ℝ, A a x ∧ B x → false) :=
begin
  sorry
end

end range_a_A_intersect_B_not_empty_l102_102329


namespace sequence_arithmetic_and_sum_inequality_l102_102684

theorem sequence_arithmetic_and_sum_inequality 
  (a : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, T n = (∏ k in finset.range n, a k))
  (h2 : ∀ n, a n + T n = 1) :
  (∀ n, (1 / T n) - (1 / T (n + 1)) = 1) ∧ 
  (∑ k in finset.range(n + 1), (a (k + 1) - a k) / a k < 3 / 4) :=
by
  sorry

end sequence_arithmetic_and_sum_inequality_l102_102684


namespace coordinates_B_l102_102682

theorem coordinates_B (A B : ℝ × ℝ) (distance : ℝ) (A_coords : A = (-1, 3)) 
  (AB_parallel_x : A.snd = B.snd) (AB_distance : abs (A.fst - B.fst) = distance) :
  (B = (-6, 3) ∨ B = (4, 3)) :=
by
  sorry

end coordinates_B_l102_102682


namespace distance_from_C_to_tangency_point_l102_102017

-- Define the given conditions
def triangle_ABC_right (A B C : Point) : Prop :=
  right_angle (angle A B C) ∧ angle B = 30°

def inscribed_circle (ABC : Triangle) (r : ℝ) : Prop :=
  inscribed_circle_radius ABC r = sqrt 3

-- State the problem
theorem distance_from_C_to_tangency_point
  {A B C : Point}
  (h₁ : triangle_ABC_right A B C)
  (h₂ : inscribed_circle ⟨A, B, C⟩ (sqrt 3)) :
  distance C tangency_point_AB = sqrt (15 + 6 * sqrt 3) :=
sorry

end distance_from_C_to_tangency_point_l102_102017


namespace plane_equation_l102_102610

noncomputable def point_on_plane (p : ℝ × ℝ × ℝ) : Prop :=
p = (2, 1, -1)

def line_through_plane (p : ℝ × ℝ × ℝ) : Prop :=
∃ t : ℝ, (p.1 = 4 * t + 2) ∧ (p.2 = -t + 2) ∧ (p.3 = 3 * t - 1)

def equation_of_plane (p : ℝ × ℝ × ℝ) : Prop :=
3 * p.1 + 0 * p.2 - 4 * p.3 - 10 = 0

theorem plane_equation :
  (∀ p : ℝ × ℝ × ℝ, point_on_plane p → equation_of_plane p) ∧ 
  (∀ p : ℝ × ℝ × ℝ, line_through_plane p → equation_of_plane p) :=
by
  sorry

end plane_equation_l102_102610


namespace circles_must_be_odd_l102_102468

theorem circles_must_be_odd (n : ℕ) (h1 : ∀ (c1 c2 : ℕ), c1 ≠ c2 → ∃ p, c1 ≠ p ∧ c2 ≠ p) 
  (h2 : ∀ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 → ∃ p1 p2, p1 ≠ p2 ∧ (c1, c2) ≠ (p1, p2)) 
  (h3 : ∃ path, (∀ circle, circle ∈ path)) : n % 2 = 1 :=
by
  sorry

end circles_must_be_odd_l102_102468


namespace complex_multiplication_l102_102744

theorem complex_multiplication (z : ℂ) (i : ℂ) (hz : z = 1 + 2 * i) (hi : i = complex.I) : 
  i * z = -2 + i := 
by
  sorry

end complex_multiplication_l102_102744


namespace geometric_series_sum_l102_102600

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
  a / (1 - r)

theorem geometric_series_sum :
  infinite_geometric_series_sum (5 / 3) (1 / 3) (by norm_num : |1 / 3| < 1) = 5 / 2 :=
by
  sorry

end geometric_series_sum_l102_102600


namespace remainder_of_division_l102_102974

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l102_102974


namespace binomial_equality_and_value_l102_102232

open Nat

def binomial (n k : ℕ) : ℕ :=
  n.choose k

theorem binomial_equality_and_value :
  binomial 10 8 = binomial 10 2 ∧ binomial 10 8 = 45 :=
by
  -- binomial coefficient symmetry property
  have h_symmetry : binomial 10 8 = binomial 10 2 := by 
    rw Nat.choose_symm 
    rfl
  -- compute binomial coefficient binomial(10,8) = 45
  have h_value : binomial 10 8 = 45 := by
    rw [←Nat.choose_succ_right_eq_choose, ←Nat.choose_self (10 - 8)]
    sorry
  exact ⟨h_symmetry, h_value⟩

end binomial_equality_and_value_l102_102232


namespace excluded_sum_l102_102251

theorem excluded_sum (f : ℝ → ℝ) (hf : ∀ x, f x = 5 * x / (3 * x^2 - 9 * x + 6)) :
  let domain_exclude := {x | 3 * x^2 - 9 * x + 6 = 0}
  finset.univ.filter (λ x, x ∈ domain_exclude).sum = 3 :=
by
  sorry

end excluded_sum_l102_102251


namespace number_of_zeros_of_f_l102_102252

def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else x^2 - 1

theorem number_of_zeros_of_f : ∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
  sorry

end number_of_zeros_of_f_l102_102252


namespace angle_AMD_is_73_16_l102_102435

noncomputable def rectangle_ABCD := 
  ∃ (A B C D : Point), 
    (dist A B = 8) ∧ 
    (dist B C = 4) ∧ 
    (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) ∧ 
    (A ≠ C ∧ B ≠ D) ∧ 
    (dist A D = 4) ∧ 
    (dist C D = 8)

def point_M_on_AB (A B M : Point) := 
  dist A M = (2 / 3) * dist A B

axiom angle_AMD_eq_angle_CMD 
  (A B C D M : Point)
  (h1 : rectangle_ABCD A B C D)
  (h2 : point_M_on_AB A B M) :
  angle A M D = angle C M D

theorem angle_AMD_is_73_16 :
  ∀ (A B C D M : Point)
  (h1 : rectangle_ABCD A B C D)
  (h2 : point_M_on_AB A B M)
  (h3 : angle A M D = angle C M D), 
  angle A M D = 73.16 :=
by sorry

end angle_AMD_is_73_16_l102_102435


namespace quadratic_roots_solution_l102_102984

noncomputable def quadratic_roots_differ_by_2 (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) : Prop :=
  let root1 := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let root2 := (-p - Real.sqrt (p^2 - 4*q)) / 2
  abs (root1 - root2) = 2

theorem quadratic_roots_solution (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) :
  quadratic_roots_differ_by_2 p q hq_pos hp_pos →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end quadratic_roots_solution_l102_102984


namespace sum_f_l102_102920

noncomputable def f (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x < -1 then -(x+2)^(2)
else if -1 ≤ x ∧ x < 3 then x
else f (x - 6)

theorem sum_f (n : ℕ) :
  (∑ k in Finset.range n, f k) = 336 :=
by
  sorry

end sum_f_l102_102920


namespace gcd_40304_30203_eq_1_l102_102280

theorem gcd_40304_30203_eq_1 : Nat.gcd 40304 30203 = 1 := 
by 
  sorry

end gcd_40304_30203_eq_1_l102_102280


namespace inequality_proof_l102_102053

theorem inequality_proof (n : ℕ) (r : Fin n → ℝ) 
  (hr : ∀ i, 1 ≤ r i) : 
  (∑ i, 1 / (r i + 1)) ≥ n / (Real.sqrt (Finset.univ.prod r) + 1) :=
sorry

end inequality_proof_l102_102053


namespace cartesian_eq_line_l_min_distance_PQ_l102_102378

variables {θ : ℝ} (x y : ℝ)

noncomputable def curve_C := ∃ θ : ℝ, θ ∈ Ioo (-π/2) (π/2) ∧
  (x = 8 * (Real.tan θ)^2 ∧ y = 8 * Real.tan θ)

noncomputable def line_l := ∀ (ρ θ : ℝ), ρ * Real.cos (θ - π / 4) = -4 * Real.sqrt 2 → 
  (∃ x y : ℝ, ρ^2 = x^2 + y^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
    x + y + 8 = 0)

theorem cartesian_eq_line_l : line_l
 := sorry

theorem min_distance_PQ : ∀ P Q : (ℝ × ℝ), 
  (curve_C P.1 P.2 → line_l Q.1 Q.2 → 
  ∃ θ : ℝ, (Real.tan θ = - 1/2) ∧ (|P.1 - Q.1|^2 + |P.2 - Q.2|^2 = (3 * Real.sqrt 2)^2)) 
 := sorry

end cartesian_eq_line_l_min_distance_PQ_l102_102378


namespace translate_sine_graph_l102_102487

theorem translate_sine_graph (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) :
  (∀ x, sqrt 2 * sin (2 * (x - ϕ) + π / 3) = 2 * sin x * (sin x - cos x) - 1) →
  ϕ = 13 * π / 24 :=
by
  sorry

end translate_sine_graph_l102_102487


namespace min_max_values_on_interval_l102_102843

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l102_102843


namespace amara_clothing_remaining_l102_102565

theorem amara_clothing_remaining :
  ∀ (initial donation_one donation_factor discard : ℕ),
    initial = 100 →
    donation_one = 5 →
    donation_factor = 3 →
    discard = 15 →
    let total_donated := donation_one + (donation_factor * donation_one) in
    let remaining_after_donation := initial - total_donated in
    let final_remaining := remaining_after_donation - discard in
    final_remaining = 65 := 
by
  sorry

end amara_clothing_remaining_l102_102565


namespace sum_of_digits_of_N_l102_102206

theorem sum_of_digits_of_N :
  (∃ N : ℕ, 3 * N * (N + 1) / 2 = 3825 ∧ (N.digits 10).sum = 5) :=
by
  sorry

end sum_of_digits_of_N_l102_102206


namespace problem_statement_l102_102013

noncomputable def points_collinear (O A B C : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (C.1 - A.1) = k * (B.1 - A.1) ∧ (C.2 - A.2) = k * (B.2 - A.2)

noncomputable def f (x m : ℝ) : ℝ :=
  let OA := (1, Real.cos x)
  let OB := (1 + Real.sin x, Real.cos x)
  let OC := (1 + (2 / 3) * Real.sin x, Real.cos x)
  let AB := (Real.sin x, 0)
  let dot_product := (1 + (2 / 3) * Real.sin x + (Real.cos x) ^ 2)
  let abs_AB := Real.sin x
  dot_product + (2 * m + (1 / 3)) * abs_AB + m ^ 2

theorem problem_statement (O A B C : ℝ × ℝ) (x : ℝ) (m : ℝ) (h1 : O = (0, 0)) 
    (h2 : C = (1 + (2 / 3) * (B.1 - 1), B.2))
    (h3 : A = (1, Real.cos x)) (h4 : B = (1 + Real.sin x, Real.cos x))
    (h5 : 0 ≤ x ∧ x ≤ Real.pi / 2)
    (h6 : ∀ x, f x m = 5) : 
  points_collinear O A B C ∧ (m = -3 ∨ m = Real.sqrt 3) :=
begin
  sorry
end


end problem_statement_l102_102013


namespace solve_system_of_equations_l102_102441

theorem solve_system_of_equations :
  ∃ x y : ℝ, x + 2 * y = 8 ∧ 3 * x + y = 9 ∧ x = 2 ∧ y = 3 :=
by
  use 2, 3
  split
  { norm_num }
  split
  { norm_num }
  split
  { refl }
  { refl }

end solve_system_of_equations_l102_102441


namespace number_of_prime_sums_l102_102092

open Nat

noncomputable def generate_sums : Nat → List Nat
| 0 => [3]
| n + 1 => let sums := generate_sums n
           sums ++ [sums.last!.get! + nth_prime (n + 2)]

def is_prime_sum (sums : List Nat) : Nat :=
sums.count isPrime

theorem number_of_prime_sums : is_prime_sum (generate_sums 14) = 3 := 
by
  sorry

end number_of_prime_sums_l102_102092


namespace new_shoes_cost_percentage_increase_l102_102536

noncomputable def repair_cost : ℝ := 14.50
noncomputable def repair_tax : ℝ := 0.10
noncomputable def repair_lifespan : ℝ := 1.0

noncomputable def new_shoes_cost : ℝ := 32.00
noncomputable def new_shoes_discount : ℝ := 0.075
noncomputable def new_shoes_tax : ℝ := 0.125
noncomputable def new_shoes_lifespan : ℝ := 2.0

def total_repair_cost : ℝ := repair_cost * (1 + repair_tax)
def total_new_shoes_cost : ℝ := (new_shoes_cost * (1 - new_shoes_discount)) * (1 + new_shoes_tax)

def avg_repair_cost_per_year : ℝ := total_repair_cost / repair_lifespan
def avg_new_shoes_cost_per_year : ℝ := total_new_shoes_cost / new_shoes_lifespan

def cost_increase_percentage : ℝ := ((avg_new_shoes_cost_per_year - avg_repair_cost_per_year) / avg_repair_cost_per_year) * 100

theorem new_shoes_cost_percentage_increase :
  cost_increase_percentage ≈ 4.39 :=
by
  sorry

end new_shoes_cost_percentage_increase_l102_102536


namespace correct_sampling_methods_l102_102375

def reporter_A_sampling : String :=
  "systematic sampling"

def reporter_B_sampling : String :=
  "systematic sampling"

theorem correct_sampling_methods (constant_flow : Prop)
  (A_interview_method : ∀ t : ℕ, t % 10 = 0)
  (B_interview_method : ∀ n : ℕ, n % 1000 = 0) :
  reporter_A_sampling = "systematic sampling" ∧ reporter_B_sampling = "systematic sampling" :=
by
  sorry

end correct_sampling_methods_l102_102375


namespace polygon_area_l102_102994

def point := (ℕ, ℕ)

def polygon : list point := 
  [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0),
   (0, 10), (10, 10), (20, 10), (30, 10), (40, 10),
   (0, 20), (10, 20), (20, 20), (30, 20), (40, 20),
   (0, 30), (10, 30), (20, 30), (30, 30), (40, 30),
   (0, 40), (10, 40), (20, 40), (30, 40), (40, 40)]

def is_polygon (pts: list point) : Prop := 
  pts.head = (0, 0) ∧ pts.last = (0, 0) ∧ pts.length ≥ 3

def area_of_polygon (pts: list point) : ℝ := 
  -- Define the function to compute the area of a polygon, skipped for brevity
  sorry

theorem polygon_area : 
  is_polygon polygon ∧ area_of_polygon polygon = 31.5 :=
by
  sorry

end polygon_area_l102_102994


namespace y_intercepts_parabola_l102_102993

theorem y_intercepts_parabola : 
  ∀ (y : ℝ), ¬(0 = 3 * y^2 - 5 * y + 12) :=
by 
  -- Given x = 0, we have the equation 3 * y^2 - 5 * y + 12 = 0.
  -- The discriminant ∆ = b^2 - 4ac = (-5)^2 - 4 * 3 * 12 = 25 - 144 = -119 which is less than 0.
  -- Since the discriminant is negative, the quadratic equation has no real roots.
  sorry

end y_intercepts_parabola_l102_102993


namespace maxTestTubesC_is_73_l102_102136

noncomputable def maxTestTubesC : ℕ :=
  let a b c : ℕ 
  in if 0.1 * a + 0.2 * b + 0.9 * c = 0.2017 * (a + b + c) 
  ∧ a + b + c = 1000 
  ∧ a + 2 * b + 9 * c = 2017 
  ∧ 7 * c ≤ 517 
  ∧ 8 * c ≥ 518 
  ∧ c ≤ 500 
  then c else 0

theorem maxTestTubesC_is_73 : maxTestTubesC = 73 := 
by {
  sorry 
}

end maxTestTubesC_is_73_l102_102136


namespace solve_inequalities_l102_102130

theorem solve_inequalities (x : ℝ) : (x + 1 > 0 ∧ x - 3 < 2) ↔ (-1 < x ∧ x < 5) :=
by sorry

end solve_inequalities_l102_102130


namespace student_total_marks_l102_102374

theorem student_total_marks (total_questions correct_answers : ℕ) 
  (marks_per_correct marks_per_wrong : ℤ) : 
  total_questions = 120 ∧ correct_answers = 75 ∧ marks_per_correct = 3 ∧ marks_per_wrong = -1 →
  let wrong_answers := total_questions - correct_answers in
  let total_marks := correct_answers * marks_per_correct + wrong_answers * marks_per_wrong in
  total_marks = 180 :=
by {
  intro h,
  cases h with h_tot_ques h1,
  cases h1 with h_corr_ans h2,
  cases h2 with h_marks_corr h_marks_wrong,
  simp [wrong_answers, total_marks, h_tot_ques, h_corr_ans, h_marks_corr, h_marks_wrong],
  sorry,
}

end student_total_marks_l102_102374


namespace freq_dist_histogram_tallest_rectangle_is_mode_l102_102369

/-- 
Given a frequency distribution histogram, prove that the characteristic of the data 
corresponding to the middle position of the tallest rectangle is Mode.
-/
theorem freq_dist_histogram_tallest_rectangle_is_mode :
  ∀ (data : List ℕ), (∀ median mode mean stddev : ℝ,
  let histogram : ℝ → ℝ := frequency_distribution_histogram data in
  let tallest_rectangle_x : ℝ := midpoint_of_tallest_rectangle_base histogram in
  (tallest_rectangle_x = mode)) :=
begin
  sorry,
end

end freq_dist_histogram_tallest_rectangle_is_mode_l102_102369


namespace remainder_1234567_div_256_l102_102972

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l102_102972


namespace expected_value_of_winnings_l102_102796

theorem expected_value_of_winnings :
  let primes := [2, 3, 5, 7]
  let composites := [4, 6, 8]
  let p_prime := 4/8
  let p_composite := 3/8
  let p_one := 1/8
  let winnings_primes := 2*2 + 2*3 + 2*5 + 2*7
  let loss_composite := -1
  let loss_one := -3
  let E := p_prime * winnings_primes + p_composite * loss_composite + p_one * loss_one
  E = 16.25 :=
by
  let primes := [2, 3, 5, 7]
  let composites := [4, 6, 8]
  let p_prime := 4/8
  let p_composite := 3/8
  let p_one := 1/8
  let winnings_primes := 2*2 + 2*3 + 2*5 + 2*7
  let loss_composite := -1
  let loss_one := -3
  let E := p_prime * winnings_primes + p_composite * loss_composite + p_one * loss_one
  sorry

end expected_value_of_winnings_l102_102796


namespace hour_hand_degrees_noon_to_2_30_l102_102521

def degrees_moved (hours: ℕ) : ℝ := (hours * 30)

theorem hour_hand_degrees_noon_to_2_30 :
  degrees_moved 2 + degrees_moved 1 / 2 = 75 :=
sorry

end hour_hand_degrees_noon_to_2_30_l102_102521


namespace max_value_of_x_plus_y_l102_102654

theorem max_value_of_x_plus_y (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : x + y ≤ 5 :=
sorry

end max_value_of_x_plus_y_l102_102654


namespace share_of_C_l102_102810

theorem share_of_C (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 578) : 
  C = 408 :=
by
  -- Proof goes here
  sorry

end share_of_C_l102_102810


namespace spaces_per_row_l102_102075

theorem spaces_per_row 
  (kind_of_tomatoes : ℕ)
  (tomatoes_per_kind : ℕ)
  (kind_of_cucumbers : ℕ)
  (cucumbers_per_kind : ℕ)
  (potatoes : ℕ)
  (rows : ℕ)
  (additional_spaces : ℕ)
  (h1 : kind_of_tomatoes = 3)
  (h2 : tomatoes_per_kind = 5)
  (h3 : kind_of_cucumbers = 5)
  (h4 : cucumbers_per_kind = 4)
  (h5 : potatoes = 30)
  (h6 : rows = 10)
  (h7 : additional_spaces = 85) :
  (kind_of_tomatoes * tomatoes_per_kind + kind_of_cucumbers * cucumbers_per_kind + potatoes + additional_spaces) / rows = 15 :=
by
  sorry

end spaces_per_row_l102_102075


namespace simplify_fraction_sqrt_l102_102434

theorem simplify_fraction_sqrt (x y : ℝ) (hx : x = √3) (hy : y = √2) : 
  1 / (x + y) = x - y :=
by
  sorry

end simplify_fraction_sqrt_l102_102434


namespace highest_avg_speed_2_to_3_l102_102424

-- Define the time periods and distances traveled in those periods
def distance_8_to_9 : ℕ := 50
def distance_9_to_10 : ℕ := 70
def distance_10_to_11 : ℕ := 60
def distance_2_to_3 : ℕ := 80
def distance_3_to_4 : ℕ := 40

-- Define the average speed calculation for each period
def avg_speed (distance : ℕ) (hours : ℕ) : ℕ := distance / hours

-- Proposition stating that the highest average speed is from 2 pm to 3 pm
theorem highest_avg_speed_2_to_3 : 
  avg_speed distance_2_to_3 1 > avg_speed distance_8_to_9 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_9_to_10 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_10_to_11 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_3_to_4 1 := 
by 
  sorry

end highest_avg_speed_2_to_3_l102_102424


namespace pond_eyes_l102_102371

theorem pond_eyes : 
  let frogs := 20 in
  let crocodiles := 10 in
  let spiders := 15 in
  let cyclops := 5 in
  let eyes_per_frog := 2 in
  let eyes_per_crocodile := 2 in
  let eyes_per_spider := 8 in
  let eyes_per_cyclops := 1 in
  (frogs * eyes_per_frog + crocodiles * eyes_per_crocodile + spiders * eyes_per_spider + cyclops * eyes_per_cyclops) = 185 :=
by
  sorry

end pond_eyes_l102_102371


namespace string_cheese_packages_l102_102760

theorem string_cheese_packages (days_per_week : ℕ) (weeks : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) (pack_size : ℕ) 
    (H1 : days_per_week = 5)
    (H2 : weeks = 4)
    (H3 : oldest_daily = 2)
    (H4 : youngest_daily = 1)
    (H5 : pack_size = 30) 
  : (oldest_daily * days_per_week + youngest_daily * days_per_week) * weeks / pack_size = 2 :=
  sorry

end string_cheese_packages_l102_102760


namespace net_rate_26_dollars_per_hour_l102_102197

/-
Conditions:
1. The driver travels for 2 hours.
2. The speed of the car is 60 miles per hour.
3. The car gets 30 miles per gallon of gasoline.
4. She is paid $0.50 per mile.
5. The cost of gasoline is $2.00 per gallon.
-/

/-
Question:
Prove that the net rate of pay, in dollars per hour, after the gasoline expense is 26 dollars.
-/

def net_rate_of_pay 
  (travel_hours : ℕ) 
  (speed_mph : ℕ) 
  (efficiency_mpg : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ) : ℚ :=
  let distance := travel_hours * speed_mph,
      gallons := distance / efficiency_mpg,
      earnings := distance * pay_per_mile,
      gas_expense := gallons * gas_cost_per_gallon,
      net_earnings := earnings - gas_expense,
      net_rate := net_earnings / travel_hours
  in net_rate

theorem net_rate_26_dollars_per_hour :
  net_rate_of_pay 2 60 30 (1/2) 2 = 26 := by
  sorry

end net_rate_26_dollars_per_hour_l102_102197


namespace max_tourism_consumption_l102_102551

def p (x : ℕ) (h : 1 ≤ x ∧ x ≤ 12) : ℕ := -3 * x^2 + 40 * x

def q (x : ℕ) (h : 1 ≤ x ∧ x ≤ 6 ∨ 7 ≤ x ∧ x ≤ 12) : ℕ :=
  if h.1 then 35 - 2 * x else 160 / x

def g (x : ℕ) (h : 1 ≤ x ∧ x ≤ 12) : ℕ :=
  if x ≤ 6 then 6 * x^3 - 185 * x^2 + 1400 * x else -480 * x + 6400

theorem max_tourism_consumption : ∃ x, 1 ≤ x ∧ x ≤ 12 ∧ g x (and.intro sorry sorry) = 3125 :=
sorry

end max_tourism_consumption_l102_102551
