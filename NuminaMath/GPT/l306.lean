import Mathlib

namespace value_of_M_l306_306711

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end value_of_M_l306_306711


namespace solve_for_m_l306_306239

noncomputable def has_positive_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (m / (x - 3) - 1 / (3 - x) = 2)

theorem solve_for_m (m : ℝ) : has_positive_root m → m = -1 :=
sorry

end solve_for_m_l306_306239


namespace solve_for_x_l306_306096

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end solve_for_x_l306_306096


namespace minimum_sum_of_distances_l306_306492

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem minimum_sum_of_distances :
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (1, 1)
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (∀ Q : ℝ × ℝ, Q.2 = 0 → distance Q A + distance Q B ≥ sqrt 10) :=
begin
  sorry
end

end minimum_sum_of_distances_l306_306492


namespace equilateral_triangle_side_length_l306_306888

theorem equilateral_triangle_side_length :
  ∃ s : ℕ, (3 * s = 2 * (125 + 115)) ∧ s = 160 :=
by
  use 160
  split
  · calc
    3 * 160 = 480 := by norm_num
    2 * (125 + 115) = 2 * 240 := by norm_num
    2 * 240 = 480 := by norm_num
  · refl

end equilateral_triangle_side_length_l306_306888


namespace angle_A_value_l306_306266

/-- 
In triangle ABC, the sides opposite to angles A, B, C are a, b, and c respectively.
Given:
  - C = π / 3,
  - b = √6,
  - c = 3,
Prove that A = 5π / 12.
-/
theorem angle_A_value (a b c : ℝ) (A B C : ℝ) (hC : C = Real.pi / 3) (hb : b = Real.sqrt 6) (hc : c = 3) :
  A = 5 * Real.pi / 12 :=
sorry

end angle_A_value_l306_306266


namespace randi_peter_ratio_l306_306846

-- Given conditions
def ray_cents := 175
def cents_per_nickel := 5
def peter_cents := 30
def randi_extra_nickels := 6

-- Define the nickels Ray has
def ray_nickels := ray_cents / cents_per_nickel
-- Define the nickels Peter receives
def peter_nickels := peter_cents / cents_per_nickel
-- Define the nickels Randi receives
def randi_nickels := peter_nickels + randi_extra_nickels
-- Define the cents Randi receives
def randi_cents := randi_nickels * cents_per_nickel

-- The goal is to prove the ratio of the cents given to Randi to the cents given to Peter is 2.
theorem randi_peter_ratio : randi_cents / peter_cents = 2 := by
  sorry

end randi_peter_ratio_l306_306846


namespace complement_union_l306_306619

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306619


namespace average_of_thirteen_numbers_l306_306326

theorem average_of_thirteen_numbers (a b : ℕ) (middle : ℕ) (numbers : list ℕ)
  (h1 : a = 30) (h2 : b = 42) (h3 : middle = 45)
  (h4 : numbers.length = 13)
  (h5 : (∀ l1 l2, l1 ++ [middle] ++ l2 = numbers → l1.length = 6 → l2.length = 6 → list.sum l1 = a ∧ list.sum l2 = b)) :
  list.sum numbers / numbers.length = 9 :=
by
  sorry

end average_of_thirteen_numbers_l306_306326


namespace divisors_of_9_fact_greater_than_8_fact_l306_306225

noncomputable def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

def divisor (d n : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem divisors_of_9_fact_greater_than_8_fact : 
  { d : ℕ | divisor d (fact 9) ∧ d > fact 8 }.toFinset.card = 8 := by
  sorry

end divisors_of_9_fact_greater_than_8_fact_l306_306225


namespace M_intersection_N_eq_M_l306_306214

def is_element_of_M (y : ℝ) : Prop := ∃ x : ℝ, y = 2^x
def is_element_of_N (y : ℝ) : Prop := ∃ x : ℝ, y = x^2

theorem M_intersection_N_eq_M : {y | is_element_of_M y} ∩ {y | is_element_of_N y} = {y | is_element_of_M y} :=
by
  sorry

end M_intersection_N_eq_M_l306_306214


namespace part1_part2_l306_306187

-- Definition of sets A and B
def A (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem (Ⅰ)
theorem part1 (a : ℝ) : (A a ∩ B = ∅ ∧ A a ∪ B = Set.univ) → a = 2 :=
by
  sorry

-- Theorem (Ⅱ)
theorem part2 (a : ℝ) : (A a ⊆ B ∧ A a ≠ ∅) → (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l306_306187


namespace price_after_reductions_l306_306870

theorem price_after_reductions (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 
  let p := 2 in
  let y := p * (1 - x) * (1 - x) in
  y = 2 * (1 - x) ^ 2 :=
by
  sorry

end price_after_reductions_l306_306870


namespace ratio_amounts_l306_306311

-- Define the variables and values for the problem
variable (total amount_J : ℕ := 6000)
variable (amount_John : ℕ := 2000)

-- The proof statement to show that the ratio is 1 : y : z given the conditions.
theorem ratio_amounts (y z : ℕ) (h_total : total = 6000) (h_john : amount_John = 2000) 
                      (h_sum : amount_John + y + z = total) :
  1 * (amount_John / 2000) : y : z :=
sorry

end ratio_amounts_l306_306311


namespace solve_for_x_l306_306852

theorem solve_for_x (x : ℝ) : 64 = 4 * (16:ℝ)^(x - 2) → x = 3 :=
by 
  intro h
  sorry

end solve_for_x_l306_306852


namespace find_missing_number_l306_306914

theorem find_missing_number (x : ℕ) :
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 :=
by
  sorry

end find_missing_number_l306_306914


namespace find_greatest_k_l306_306454

theorem find_greatest_k (x : ℕ → ℝ) :
  (x 0 = 0) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 100 → 1 ≤ x i - x (i-1) ∧ x i - x (i-1) ≤ 2) ∧
  (∃ k, k ≤ 100 ∧ (∀ k_1 ≤ k, x k_1 + ∑ i in finset.range (101 - k_1), x (k_1 + i) ≥ ∑ i in finset.range (k_1 - 1), x i)) :=
begin
  let k := 67,
  sorry
end

end find_greatest_k_l306_306454


namespace coo_coo_count_correct_l306_306105

theorem coo_coo_count_correct :
  let monday_coos := 89
  let tuesday_coos := 179
  let wednesday_coos := 21
  let total_coos := monday_coos + tuesday_coos + wednesday_coos
  total_coos = 289 :=
by
  sorry

end coo_coo_count_correct_l306_306105


namespace top_four_teams_points_l306_306752

theorem top_four_teams_points (total_teams : ℕ) (total_games : ℕ) (total_points : ℕ)
  (points_per_win : ℕ) (points_per_draw : ℕ) (points_per_loss : ℕ) (games_between_teams : total_teams = 8)
  (games_each_other_twice : total_games = nat.choose 8 2 * 2)
  (points_distributed : total_points = total_games * points_per_win) :
  let p := 33 in
  total_teams = 8 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  ∃ (A B C D : ℕ), A = p ∧ B = p ∧ C = p ∧ D = p :=
by {
  intros h0 h1 h2 h3,
  sorry
}

end top_four_teams_points_l306_306752


namespace range_of_a_l306_306550

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a * x - 1 else a / x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

def func_increasing_on_R (a : ℝ) : Prop :=
  is_increasing_on (f a) Set.univ

theorem range_of_a (a : ℝ) : func_increasing_on_R a ↔ a < -2 :=
sorry

end range_of_a_l306_306550


namespace fixed_cost_to_break_even_l306_306832

def cost_per_handle : ℝ := 0.6
def selling_price_per_handle : ℝ := 4.6
def num_handles_to_break_even : ℕ := 1910

theorem fixed_cost_to_break_even (F : ℝ) (h : F = num_handles_to_break_even * (selling_price_per_handle - cost_per_handle)) :
  F = 7640 := by
  sorry

end fixed_cost_to_break_even_l306_306832


namespace complement_union_l306_306652

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306652


namespace degree_not_determined_from_characteristic_l306_306787

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l306_306787


namespace complement_of_union_is_singleton_five_l306_306640

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306640


namespace walking_speed_l306_306079

theorem walking_speed (A : ℝ) (T : ℝ) (s_diagonal : ∀ s : ℝ, s = real.sqrt A) :
  (∃ d : ℝ, d = real.sqrt (2 * A)) →
  (∃ v : ℝ, v = 15 / T) →
  ((15 / T) * 3.6 = 6) :=
by
  intros h_d h_v
  exact sorry 

end walking_speed_l306_306079


namespace range_of_m_l306_306333

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (h_even : ∀ x, f x = f (-x)) 
 (h_decreasing : ∀ {x y}, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x)
 (h_condition : ∀ x, 1 ≤ x → x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (Real.log x + 3 - 2 * m * x)) :
  m ∈ Set.Icc (1 / (2 * Real.exp 1)) ((Real.log 3 + 6) / 6) :=
sorry

end range_of_m_l306_306333


namespace selection_events_mutually_exclusive_not_complementary_l306_306746

-- Definitions representing the individuals
def boys : Finset ℕ := {1, 2}
def girls : Finset ℕ := {3, 4}

-- Event definitions
def event_exactly_one_girl (selection : Finset ℕ) : Prop :=
  selection.card = 2 ∧ ∃ girl ∈ selection, ∃ boy ∈ selection, boy ∉ girls

def event_exactly_two_girls (selection : Finset ℕ) : Prop :=
  selection.card = 2 ∧ ∀ x ∈ selection, x ∈ girls

-- Sample space of selecting 2 people out of 4
def sample_space : Finset (Finset ℕ) := (Finset.powerset (boys ∪ girls)).filter (λ s, s.card = 2)

-- The theorem statement
theorem selection_events_mutually_exclusive_not_complementary :
  ∀ selection ∈ sample_space,
  (event_exactly_one_girl selection ∧ event_exactly_two_girls selection) = false :=
begin
  sorry -- Proof to be filled in
end

end selection_events_mutually_exclusive_not_complementary_l306_306746


namespace limit_computation_l306_306053

noncomputable def compute_limit : ℝ :=
  (4 : ℝ)^(5 * (0 : ℝ)) - (9 : ℝ)^(-2 * (0 : ℝ)) / (sin (0 : ℝ) - tan ((0 : ℝ)^3))

theorem limit_computation :
  ∃ l : ℝ, (∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < abs x ∧ abs x < δ → abs ((4^(5*x) - 9^(-2*x)) / (sin x - tan(x^3)) - l) < ε)) ∧ l = ln(1024 * 81)
:= 
sorry

end limit_computation_l306_306053


namespace eccentricity_of_hyperbola_l306_306263

theorem eccentricity_of_hyperbola (a : ℝ) (h : a > 0) :
  (∃ b : ℝ, b^2 = 1) ∧ (∃ f : ℝ × ℝ, f = (2, 0)) ∧ (let c := Math.sqrt(4 + 1) in 
  ∀ e : ℝ, e = c / a ↔ e = Math.sqrt(5) / 2) := 
sorry

end eccentricity_of_hyperbola_l306_306263


namespace sqrt_inequality_l306_306394

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) :
  sqrt (a * b^2 + a^2 * b) + sqrt ((1 - a) * (1 - b)^2 + (1 - a)^2 * (1 - b)) < sqrt 2 := 
sorry

end sqrt_inequality_l306_306394


namespace complement_union_eq_singleton_five_l306_306567

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306567


namespace prove_a_range_for_f_l306_306204

def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then -x^2 + 4*x - 3 else Real.log x

theorem prove_a_range_for_f (a : ℝ) :
  (∀ x : ℝ, (|f x| + 1 ≥ a * x)) ↔ (-8 ≤ a ∧ a ≤ 1) :=
by
  sorry

end prove_a_range_for_f_l306_306204


namespace complement_union_l306_306610

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306610


namespace sum_of_N_digits_l306_306342

-- Defining the conditions and the problem statement
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_N_digits (N : ℕ) (h : N^2 = 25^64 * 64^25) : sum_of_digits N = 14 := by
  sorry

end sum_of_N_digits_l306_306342


namespace algebraic_expression_values_l306_306698

-- Defining the given condition
def condition (x y : ℝ) : Prop :=
  x^4 + 6 * x^2 * y + 9 * y^2 + 2 * x^2 + 6 * y + 4 = 7

-- Defining the target expression
def target_expression (x y : ℝ) : ℝ :=
  x^4 + 6 * x^2 * y + 9 * y^2 - 2 * x^2 - 6 * y - 1

-- Stating the theorem to be proved
theorem algebraic_expression_values (x y : ℝ) (h : condition x y) :
  target_expression x y = -2 ∨ target_expression x y = 14 :=
by
  sorry

end algebraic_expression_values_l306_306698


namespace probability_either_but_not_both_l306_306361

open Classical

def event_chile := Ω → Prop
def event_madagascar := Ω → Prop

variable {Ω : Type}

axiom P_chile : ProbabilityTheory ℙ event_chile
axiom P_madagascar : ProbabilityTheory ℙ event_madagascar

axiom P_chile_given : ℙ[event_chile] = 0.5
axiom P_madagascar_given : ℙ[event_madagascar] = 0.5

theorem probability_either_but_not_both :
  ℙ[event_chile ∧ ¬event_madagascar] + ℙ[¬event_chile ∧ event_madagascar] = 0.5 :=
  sorry

end probability_either_but_not_both_l306_306361


namespace BP_eq_CQ_MN_parallel_AD_l306_306526

-- Given conditions
variables {A B C D M P Q N : Type}
variables (triangleABC : Triangle A B C) (circumcircleO : Circle) (circumscribed ⦃triangleABC⦄)
variables (angleBisectorBAC : AngleBisector ∠BAC intersects BC at D)
variables (midpointM : MidPoint B C M)
variables (circumcircleADM : Circle) (circumscribed ⦃triangleADM⦄ intersects AB at P intersects AC at Q)
variables (midpointN : MidPoint P Q N)

-- Proof statements to be provided
theorem BP_eq_CQ (triangleABC : Triangle A B C) 
  (circumcircleO : Circle) 
  (angleBisectorBAC : AngleBisector ∠BAC intersects BC at D) 
  (midpointM : MidPoint B C M) 
  (circumcircleADM : Circle intersects AB at P intersects AC at Q) 
  : IsEqualLength (Segment B P) (Segment C Q) :=
sorry

theorem MN_parallel_AD (triangleABC : Triangle A B C) 
  (circumcircleO : Circle) 
  (angleBisectorBAC : AngleBisector ∠BAC intersects BC at D) 
  (midpointM : MidPoint B C M) 
  (circumcircleADM : Circle intersects AB at P intersects AC at Q) 
  (midpointN : MidPoint P Q N) 
  : Parallel (Segment M N) (Segment A D) :=
sorry

end BP_eq_CQ_MN_parallel_AD_l306_306526


namespace plum_purchase_l306_306368

theorem plum_purchase
    (x : ℕ)
    (h1 : ∃ x, 5 * (6 * (4 * x) / 5) - 6 * ((5 * x) / 6) = -30) :
    2 * x = 60 := sorry

end plum_purchase_l306_306368


namespace total_walnut_trees_l306_306354

theorem total_walnut_trees (current_trees new_trees : ℕ) (h1 : current_trees = 107) (h2 : new_trees = 104) :
  current_trees + new_trees = 211 :=
by
  rw [h1, h2]
  rfl

end total_walnut_trees_l306_306354


namespace area_increase_correct_l306_306940

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l306_306940


namespace garden_area_difference_l306_306953

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l306_306953


namespace binom_16_12_eq_1820_l306_306128

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l306_306128


namespace greatest_groups_of_stuffed_animals_l306_306136

def stuffed_animals_grouping : Prop :=
  let cats := 26
  let dogs := 14
  let bears := 18
  let giraffes := 22
  gcd (gcd (gcd cats dogs) bears) giraffes = 2

theorem greatest_groups_of_stuffed_animals : stuffed_animals_grouping :=
by sorry

end greatest_groups_of_stuffed_animals_l306_306136


namespace divisors_of_9_fact_greater_than_8_fact_l306_306226

noncomputable def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

def divisor (d n : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem divisors_of_9_fact_greater_than_8_fact : 
  { d : ℕ | divisor d (fact 9) ∧ d > fact 8 }.toFinset.card = 8 := by
  sorry

end divisors_of_9_fact_greater_than_8_fact_l306_306226


namespace expression_evaluation_l306_306913

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) : -a - b^4 + a * b = -28 := 
by 
  sorry

end expression_evaluation_l306_306913


namespace find_f1_increasing_on_positive_solve_inequality_l306_306539

-- Given conditions
axiom f : ℝ → ℝ
axiom domain : ∀ x, 0 < x → true
axiom f4 : f 4 = 1
axiom multiplicative : ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y
axiom less_than_zero : ∀ x, 0 < x ∧ x < 1 → f x < 0

-- Required proofs
theorem find_f1 : f 1 = 0 := sorry

theorem increasing_on_positive : ∀ x y, 0 < x → 0 < y → x < y → f x < f y := sorry

theorem solve_inequality : {x : ℝ // 3 < x ∧ x ≤ 5} := sorry

end find_f1_increasing_on_positive_solve_inequality_l306_306539


namespace complement_of_union_is_singleton_five_l306_306647

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306647


namespace knight_tour_count_l306_306308

-- Define the 3x4 chessboard as a finite set of (x, y) coordinates 
def chessboard : Finset (Fin 3 × Fin 4) :=
  { (x, y) | x < 3 ∧ y < 4 }

-- Definition of a knight move on the chessboard
def knight_moves (p : Fin 3 × Fin 4) : Finset (Fin 3 × Fin 4) :=
  let (x, y) := p in
  (chessboard.filter (λ (p' : Fin 3 × Fin 4), 
    let (x', y') := p' in
    (abs (x' - x) = 2 ∧ abs (y' - y) = 1) ∨ (abs (x' - x) = 1 ∧ abs (y' - y) = 2)))

-- Definition of Hamiltonian paths on the given chessboard
def is_hamiltonian_path (path : List (Fin 3 × Fin 4)) : Prop :=
  (path.nodup ∧ path.length = chessboard.card ∧ 
  ∀ i < path.length - 1, List.nth path i ∈ knight_moves (List.nth path (i + 1)))

-- Main assertion: the number of Hamiltonian paths (not returning to the start) is exactly 8
theorem knight_tour_count : 
  ∃ paths : Finset (List (Fin 3 × Fin 4)), 
  (paths.filter (λ path, is_hamiltonian_path path ∧ path.head ≠ path.tail.head)).card = 8 :=
sorry

end knight_tour_count_l306_306308


namespace average_age_union_l306_306397

theorem average_age_union
    (A B C : Set Person)
    (a b c : ℕ)
    (sum_A sum_B sum_C : ℝ)
    (h_disjoint_AB : Disjoint A B)
    (h_disjoint_AC : Disjoint A C)
    (h_disjoint_BC : Disjoint B C)
    (h_avg_A : sum_A / a = 40)
    (h_avg_B : sum_B / b = 25)
    (h_avg_C : sum_C / c = 35)
    (h_avg_AB : (sum_A + sum_B) / (a + b) = 33)
    (h_avg_AC : (sum_A + sum_C) / (a + c) = 37.5)
    (h_avg_BC : (sum_B + sum_C) / (b + c) = 30) :
  (sum_A + sum_B + sum_C) / (a + b + c) = 51.6 :=
sorry

end average_age_union_l306_306397


namespace complement_union_of_M_and_N_l306_306694

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306694


namespace degree_not_determined_by_A_P_l306_306778

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l306_306778


namespace area_of_B_l306_306457

/-- B is the region in the complex plane such that both z/50 and 50/conjugate(z) have real and imaginary parts between 0 and 1 inclusive -/
def region_B (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im in
  0 ≤ x / 50 ∧ x / 50 ≤ 1 ∧ 
  0 ≤ y / 50 ∧ y / 50 ≤ 1 ∧ 
  0 ≤ 50 * x / (x^2 + y^2) ∧ 50 * x / (x^2 + y^2) ≤ 1 ∧ 
  0 ≤ 50 * y / (x^2 + y^2) ∧ 50 * y / (x^2 + y^2) ≤ 1

/-- The area of the region B defined in region_B is 2500 - 625 * π / 4 -/
theorem area_of_B :
  (∫ x : ℝ in 0..50, ∫ y : ℝ in 0..50, char_fun (region_B (x + y * I))) = 2500 - 625 * π / 4 :=
sorry

end area_of_B_l306_306457


namespace total_cost_21_l306_306047

theorem total_cost_21 (cost_app : ℝ) (monthly_cost_online : ℝ) (months_played : ℝ) (total_cost : ℝ) :
  cost_app = 5 → 
  monthly_cost_online = 8 → 
  months_played = 2 → 
  total_cost = cost_app + (monthly_cost_online * months_played) → 
  total_cost = 21 := by
  intros h_app h_monthly h_months h_total
  rw [h_app, h_monthly, h_months, h_total]
  norm_num
  sorry

end total_cost_21_l306_306047


namespace two_polyhedra_share_interior_point_l306_306102

-- Definitions for the geometric entities involved in the problem
variable {P : Type}
variable [convex_polyhedron P]

-- Assume a convex polyhedron P1 with 9 vertices A1, A2, ..., A9
variables (A : Fin 9 → P)

-- Definition of polyhedron Pi obtained by translating P1 such that A1 is mapped to Ai
def translated_polyhedron (i : Fin 9) : P :=
  sorry -- The exact translation mechanism needs to be defined

-- Statement of the proof problem
theorem two_polyhedra_share_interior_point :
  ∃ (i j : Fin 9), i ≠ j ∧
  ∃ x : P, interior_point_of (translated_polyhedron i) x ∧ interior_point_of (translated_polyhedron j) x :=
sorry

end two_polyhedra_share_interior_point_l306_306102


namespace total_amount_spent_l306_306366

def price_per_deck (n : ℕ) : ℝ :=
if n <= 3 then 8 else if n <= 6 then 7 else 6

def promotion_price (price : ℝ) : ℝ :=
price * 0.5

def total_cost (decks_victor decks_friend : ℕ) : ℝ :=
let cost_victor :=
  if decks_victor % 2 = 0 then
    let pairs := decks_victor / 2
    price_per_deck decks_victor * pairs + promotion_price (price_per_deck decks_victor) * pairs
  else sorry
let cost_friend :=
  if decks_friend = 2 then
    price_per_deck decks_friend + promotion_price (price_per_deck decks_friend)
  else sorry
cost_victor + cost_friend

theorem total_amount_spent : total_cost 6 2 = 43.5 := sorry

end total_amount_spent_l306_306366


namespace solve_n_l306_306432

def arithmetic_sequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  has_sum (λ k : ℕ, if k % 2 = 0 then a (k + 1) else 0) n / 2 * a 0

def sum_of_odds (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  finset.sum (finset.range (2 * n + 1)) (λ k, if k % 2 = 0 then 0 else a k)

def sum_of_evens (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  finset.sum (finset.range (2 * n)) (λ k, if k % 2 = 0 then a k else 0)

theorem solve_n (a : ℕ → ℕ) (n : ℕ) (h1 : sum_of_odds a n = 4) (h2 : sum_of_evens a n = 3) : n = 3 :=
by
  sorry

end solve_n_l306_306432


namespace regular_polygon_sides_l306_306474

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end regular_polygon_sides_l306_306474


namespace sum_of_solutions_eq_l306_306281

def g (x : ℝ) : ℝ := 20 * x - 4
def g_inv (x : ℝ) : ℝ := (x + 4) / 20
def h (x : ℝ) : ℝ := g (1 / (2 * x + 1))

theorem sum_of_solutions_eq :
  let sol := { x | g_inv x = h x } in
  (∑ x in sol, x) = -84.5 :=
by
  sorry

end sum_of_solutions_eq_l306_306281


namespace area_of_triangle_correct_l306_306325

noncomputable def area_of_triangle : Real := 
  let x := -Real.pi / 2
  let y := x * Real.sin x
  let f := λ x : Real, x * Real.sin x
  let f' := λ x : Real, Real.sin x + x * Real.cos x
  have derivative_at_x : Real := f' x
  let tangent_line := λ x : Real, y + derivative_at_x * (x - x)
  let area_of_intersection := 1 / 2 * Real.pi * Real.pi 
  area_of_intersection

theorem area_of_triangle_correct : area_of_triangle = Real.pi^2 / 2 := 
  sorry

end area_of_triangle_correct_l306_306325


namespace coloring_possible_if_n_div_by_3_n_div_by_3_if_coloring_possible_l306_306052
noncomputable def is_colored (n : ℕ) (coloring : fin 2n → ℕ) : Prop :=
  ∀ i : fin 2n, ∃ j₁ j₂ : fin 2n, j₁ ≠ j₂ ∧ coloring j₁ ≠ coloring i ∧ coloring j₂ ≠ coloring i

theorem coloring_possible_if_n_div_by_3 (n : ℕ) (h : n % 3 = 0) : ∃ coloring : fin 2n → ℕ, is_colored n coloring :=
  sorry

theorem n_div_by_3_if_coloring_possible (n : ℕ) (coloring : fin 2n → ℕ) (h : is_colored n coloring) : n % 3 = 0 :=
  sorry

end coloring_possible_if_n_div_by_3_n_div_by_3_if_coloring_possible_l306_306052


namespace complement_union_eq_singleton_five_l306_306570

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306570


namespace pentagon_exceeds_ninety_total_degrees_l306_306144

-- Begin the problem translation
theorem pentagon_exceeds_ninety_total_degrees :
  let n := 5 in
  let sum_of_angles := (n - 2) * 180 in
  let each_angle := sum_of_angles / n in
  let excess := each_angle - 90 in
  let total_excess := n * excess in
  total_excess = 90 := by
  -- Proof not required, hence 'sorry'
  sorry

end pentagon_exceeds_ninety_total_degrees_l306_306144


namespace infinite_series_sum_l306_306118

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end infinite_series_sum_l306_306118


namespace ship_sinks_l306_306976

noncomputable def time_to_shore (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

noncomputable def leak_rate (leak_tonnes : ℝ) (leak_time_min : ℝ) : ℝ :=
  leak_tonnes / (leak_time_min / 60)

noncomputable def net_rate (leak_rate : ℝ) (pump_rate : ℝ) : ℝ :=
  leak_rate - pump_rate

noncomputable def total_water (net_rate : ℝ) (time : ℝ) : ℝ :=
  net_rate * time

theorem ship_sinks (distance : ℝ) (speed : ℝ) (leak_tonnes : ℝ) (leak_time_min : ℝ) (pump_rate : ℝ) : 
  total_water (net_rate (leak_rate leak_tonnes leak_time_min) pump_rate) (time_to_shore distance speed) ≈ 91.98 :=
by
  sorry

-- Given conditions
def distance := 77
def speed := 10.5
def leak_tonnes := 9/4
def leak_time_min := 11/2
def pump_rate := 12

-- Ship sinks theorem with given conditions
#eval ship_sinks distance speed leak_tonnes leak_time_min pump_rate

end ship_sinks_l306_306976


namespace complement_union_l306_306625

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306625


namespace complement_union_eq_l306_306588

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306588


namespace smallest_n_probability_less_than_half_l306_306062

theorem smallest_n_probability_less_than_half :
  ∃ n : ℕ, (n > 0 ∧ (10 - n + 1) / 11 < 0.5) ∧ ∀ m : ℕ, (m > 0 ∧ (10 - m + 1) / 11 < 0.5) → (n ≤ m) :=
begin
  use 6,
  split,
  { split,
    { exact nat.succ_pos' 5 },
    { norm_num } -- This checks if the fraction 5/11 < 0.5
  },
  { intros m hm,
    by_cases h1 : m ≤ 6,
    { exact h1 },
    { exfalso,
      simp at h1,
      exact not_lt.mpr h1 (by norm_num [hm.2]) }
  }
end

end smallest_n_probability_less_than_half_l306_306062


namespace horizontal_length_of_rectangle_l306_306917

theorem horizontal_length_of_rectangle
  (P : ℕ)
  (h v : ℕ)
  (hP : P = 54)
  (hv : v = h - 3) :
  2*h + 2*v = 54 → h = 15 :=
by sorry

end horizontal_length_of_rectangle_l306_306917


namespace particle_position_1989_l306_306966

theorem particle_position_1989 : 
  let initial_position := (0, 0)
  let positions :=
    [(2, 0), (2, 2), (0, 2), (0, 0), (4, 0)] ++
    (List.cycle [
      (4, 2), (2, 2), (2, 0), (3, 0),
      (3, 1), (1, 1), (1, 0), (0, 0)])
  in positions[1988] = (0, 0) :=
by 
  sorry

end particle_position_1989_l306_306966


namespace find_x_l306_306761

theorem find_x 
  (angle_ACD angle_BCD angle_BAC angle_ACB angle_ABC : ℝ)
  (h_supp : angle_ACD + angle_BCD = 180)
  (h_BCD : angle_BCD = 130)
  (h_isosceles : angle_BAC = angle_ACB) :
  angle_ABC = 80 :=
by
  have h_ACD : angle_ACD = 180 - angle_BCD, from sorry
  have h_angle_ACD_value : angle_ACD = 50, from sorry
  have h_eq : angle_ACD = angle_ACB, from sorry
  have h_sum_of_angles : angle_BAC + angle_ACB + angle_ABC = 180, from sorry
  have h_total : 50 + 50 + angle_ABC = 180, from sorry
  have h_final : angle_ABC = 80, from sorry
  exact h_final

end find_x_l306_306761


namespace max_fa_triangle_a_l306_306219

/-- Given vectors a and b and the function f(a), show that f(a) has the maximum value 4√2 + 2 --/
theorem max_fa (a : ℝ) : 
  let vec_a := (Real.sin a, Real.cos a),
      vec_b := (6 * Real.sin a + Real.cos a, 7 * Real.sin a - 2 * Real.cos a),
      f := (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2)
  in f ≤ 4 * Real.sqrt 2 + 2 := sorry

/-- In an acute triangle ABC with given conditions, show that side a has the value √10 --/
theorem triangle_a (A : ℝ) (b c : ℝ) (area : ℝ) :
  let f := 4 * Real.sqrt 2 * Real.sin (2 * A - Real.pi / 4) + 2,
      area_condition := area = 3,
      side_condition := b + c = 2 + 3 * Real.sqrt 2,
      acute_condition := A > 0 ∧ A < Real.pi / 2,
      bc := 6 * Real.sqrt 2
  in f = 6 → 
     area_condition →
     side_condition →
     acute_condition →
     bc = b * c →
     (a : ℝ), a = Real.sqrt 10 := sorry

end max_fa_triangle_a_l306_306219


namespace f_monotonically_increasing_on_0_pi_div_2_f_symmetric_about_negative_pi_div_4_l306_306211

def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem f_monotonically_increasing_on_0_pi_div_2 :
  ∀ x y ∈ Icc 0 (Real.pi / 2), x ≤ y → f x ≤ f y := by
  sorry

theorem f_symmetric_about_negative_pi_div_4:
  ∀ x : ℝ, f (-x - Real.pi / 4) = f (x - Real.pi / 4) := by
  sorry

end f_monotonically_increasing_on_0_pi_div_2_f_symmetric_about_negative_pi_div_4_l306_306211


namespace complement_union_l306_306679

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306679


namespace largest_prime_divisor_31_sq_plus_54_sq_l306_306490

/-- 
Given the number 3877, which equals 31^2 + 54^2, 
prove that the largest prime divisor of 3877 is 19.
-/
theorem largest_prime_divisor_31_sq_plus_54_sq :
  let n := 31^2 + 54^2
  n = 3877 → (∀ p : ℕ, prime p ∧ p ∣ 3877 → p ≤ 19) ∧ ∃ (p : ℕ), prime p ∧ p ∣ 3877 ∧ p = 19 :=
by
  sorry

end largest_prime_divisor_31_sq_plus_54_sq_l306_306490


namespace range_of_a_l306_306563

def set_A : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
def set_B (a : ℝ) : Set ℝ := { x : ℝ | x^2 - 2 * a * x + a + 2 = 0 }

def is_subset (A B : Set ℝ) := ∀ x, x ∈ B → x ∈ A

theorem range_of_a (a : ℝ) : is_subset set_A (set_B a) ↔ a ∈ Ioo (-1 : ℝ) (2 : ℝ) :=
  sorry

end range_of_a_l306_306563


namespace magnitude_of_z_l306_306516

noncomputable def z : ℂ := (1 + complex.i) / (2 - 2 * complex.i)

theorem magnitude_of_z : complex.abs z = 1 / 2 := 
by
  -- proof goes here
  sorry

end magnitude_of_z_l306_306516


namespace pyramid_circumscribed_sphere_area_l306_306093

noncomputable def circumscribed_sphere_surface_area (a b c : ℝ) : ℝ :=
  let diameter := real.sqrt (a^2 + b^2 + c^2)
  let radius := diameter / 2
  4 * real.pi * radius^2

theorem pyramid_circumscribed_sphere_area :
  circumscribed_sphere_surface_area (real.sqrt 3) (real.sqrt 2) 1 = 6 * real.pi :=
by sorry

end pyramid_circumscribed_sphere_area_l306_306093


namespace garden_area_increase_l306_306945

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l306_306945


namespace remainder_of_num_constant_function_compositions_mod_1000_l306_306806

noncomputable def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_constant_function_of_composition (f : ℕ → ℕ) : Prop :=
  ∃ c ∈ A, ∀ x ∈ A, f(f(x)) = c

def num_constant_function_compositions : ℕ :=
  (7 * (∑ k in Finset.range 6 \{0}, (Nat.choose 6 k) * k^(6-k)))

theorem remainder_of_num_constant_function_compositions_mod_1000 : 
  (num_constant_function_compositions % 1000) = 399 :=
by
  sorry

end remainder_of_num_constant_function_compositions_mod_1000_l306_306806


namespace limit_problem_l306_306055

noncomputable def limit_quot : ℝ :=
  real.log (1024 * 81)

theorem limit_problem :
  ∀ f : ℝ → ℝ, ∀ g : ℝ → ℝ,
  (∀ x, f x = 4^(5 * x) - 9^(-2 * x)) →
  (∀ x, g x = real.sin x - real.tan (x^3)) →
  filter.tendsto (λ x, (f x) / (g x)) filter.at_top (nhds limit_quot) :=
begin
  intros f g hf hg,
  sorry
end

end limit_problem_l306_306055


namespace cos_diff_expression_eq_half_l306_306145

theorem cos_diff_expression_eq_half :
  (Real.cos (Real.pi * 24 / 180) * Real.cos (Real.pi * 36 / 180) -
   Real.cos (Real.pi * 66 / 180) * Real.cos (Real.pi * 54 / 180)) = 1 / 2 := by
sorry

end cos_diff_expression_eq_half_l306_306145


namespace total_questions_on_test_l306_306978

/-- A teacher grades students' tests by subtracting twice the number of incorrect responses
    from the number of correct responses. Given that a student received a score of 64
    and answered 88 questions correctly, prove that the total number of questions on the test is 100. -/
theorem total_questions_on_test (score correct_responses : ℕ) (grading_system : ℕ → ℕ → ℕ)
  (h1 : score = grading_system correct_responses (88 - 2 * 12))
  (h2 : correct_responses = 88)
  (h3 : score = 64) : correct_responses + (88 - 2 * 12) = 100 :=
by
  sorry

end total_questions_on_test_l306_306978


namespace area_of_square_l306_306838

noncomputable def sqr_area_proof (P : Point) (A B C D : Point) (AP BP PC PD : ℝ)
  (hAPB : angle A P B = 3 * pi / 4) (hPC : dist P C = 12) (hPD : dist P D = 15) : ℝ :=
  123 + 6 * sqrt 119

theorem area_of_square
  (P A B C D : Point) 
  (hAP : dist A P = AP) 
  (hBP : dist B P = BP) 
  (hAPB : angle A P B = 3 * pi / 4) 
  (hPC : dist P C = 12) 
  (hPD : dist P D = 15) :
  sqr_area_proof P A B C D AP BP 12 15 = 123 + 6 * sqrt 119 :=
sorry

end area_of_square_l306_306838


namespace BigDigMiningCopperOutput_l306_306438

theorem BigDigMiningCopperOutput :
  (∀ (total_output : ℝ) (nickel_percentage : ℝ) (iron_percentage : ℝ) (amount_of_nickel : ℝ),
      nickel_percentage = 0.10 → 
      iron_percentage = 0.60 → 
      amount_of_nickel = 720 →
      total_output = amount_of_nickel / nickel_percentage →
      (1 - nickel_percentage - iron_percentage) * total_output = 2160) :=
sorry

end BigDigMiningCopperOutput_l306_306438


namespace necessary_conditions_l306_306191

theorem necessary_conditions (a b c d e : ℝ) (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) :
  a = c ∨ a + b + c + d + e = 0 :=
by
  sorry

end necessary_conditions_l306_306191


namespace equilateral_intersections_l306_306288

-- Definitions: Let ABCDEF be an inscribed hexagon with specific side lengths.
def is_regular_hexagon (A B C D E F : Point) (O : Point) (R : ℝ) : Prop :=
  is_on_circle A O R ∧ is_on_circle B O R ∧ is_on_circle C O R ∧
  is_on_circle D O R ∧ is_on_circle E O R ∧ is_on_circle F O R ∧
  dist A B = R ∧ dist C D = R ∧ dist E F = R

-- The theorem to prove: The intersection points of the circles form an equilateral triangle.
theorem equilateral_intersections
  (A B C D E F O K L M : Point) (R : ℝ)
  (Hhex : is_regular_hexagon A B C D E F O R)
  (Hcirc1 : is_circumscribed_by (triangle B O C) (circle O K))
  (Hcirc2 : is_circumscribed_by (triangle D O E) (circle O L))
  (Hcirc3 : is_circumscribed_by (triangle F O A) (circle O M))
  (Hint1 : K ≠ O)
  (Hint2 : L ≠ O)
  (Hint3 : M ≠ O) :
  dist K L = R ∧ dist L M = R ∧ dist M K = R :=
sorry

end equilateral_intersections_l306_306288


namespace complement_union_eq_l306_306579

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306579


namespace complement_union_M_N_l306_306630

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306630


namespace Anne_wander_time_l306_306232

theorem Anne_wander_time (distance speed : ℝ) (h1 : distance = 3.0) (h2 : speed = 2.0) : distance / speed = 1.5 := by
  -- Given conditions
  sorry

end Anne_wander_time_l306_306232


namespace geom_seq_common_ratio_l306_306762

theorem geom_seq_common_ratio (S_3 S_6 : ℕ) (h1 : S_3 = 7) (h2 : S_6 = 63) : 
  ∃ q : ℕ, q = 2 := 
by
  sorry

end geom_seq_common_ratio_l306_306762


namespace cannot_determine_degree_from_char_set_l306_306770

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l306_306770


namespace find_Cs_other_three_balls_l306_306986

-- Definitions
def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def sum_to_26 := 26

-- Conditions
def conditions_met (A B C : Finset ℕ) : Prop :=
  A ∪ B ∪ C = balls ∧
  A.disjoint B ∧ A.disjoint C ∧ B.disjoint C ∧
  A.sum id = sum_to_26 ∧
  B.sum id = sum_to_26 ∧
  C.sum id = sum_to_26 ∧
  {6, 11} ⊆ A  ∧
  {4, 8} ⊆ B ∧
  1 ∈ C

-- Theorem to prove
theorem find_Cs_other_three_balls (A B C : Finset ℕ) (h : conditions_met A B C) : 
  {3, 10, 12} ⊆ C :=
sorry

end find_Cs_other_three_balls_l306_306986


namespace cannot_determine_degree_from_A_P_l306_306780

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l306_306780


namespace length_WX_eq_cos18_sqrt_2_1_sub_sin18_l306_306291

noncomputable def cos : ℝ → ℝ := sorry
noncomputable def sin : ℝ → ℝ := sorry

theorem length_WX_eq_cos18_sqrt_2_1_sub_sin18 :
  let WZ := 2
  let angle_WXY := 72
  let angle_XYW := 72
  let angle_WZY := 180 - 2 * angle_WXY
  let cos := λ θ, sorry -- Assume existence of cosine function
  let sin := λ θ, sorry -- Assume existence of sine function
  let WZ_diameter := WZ
  let WX := cos 18 * Real.sqrt(2 * (1 - sin 18))
  ∀ (W X Y Z: ℝ) 
    (quad: ∀ (A B C D: ℝ), A = W → B = X → C = Y → D = Z → circle A B C D)
    (length_WZ: ∀ (W Z: ℝ), WZ_diameter = 2),
    (angle_WXY_eq : ∀ (W X Y : ℝ), ∠WXY = angle_WXY),
    (XZ_eq_YW : ∀ (X Z Y W : ℝ), XZ = YW) 
    (length_WX_proved : WX = cos 18 * Real.sqrt(2 * (1 - sin 18))),
    true :=
λ W X Y Z quad length_WZ angle_WXY_eq XZ_eq_YW length_WX_proved, sorry

end length_WX_eq_cos18_sqrt_2_1_sub_sin18_l306_306291


namespace complement_union_l306_306623

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306623


namespace complex_number_corresponding_to_OB_l306_306258

theorem complex_number_corresponding_to_OB :
  let OA : ℂ := 6 + 5 * Complex.I
  let AB : ℂ := 4 + 5 * Complex.I
  OB = OA + AB -> OB = 10 + 10 * Complex.I := by
  sorry

end complex_number_corresponding_to_OB_l306_306258


namespace garden_area_increase_l306_306944

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l306_306944


namespace unique_zero_function_l306_306486

theorem unique_zero_function 
  (f : ℕ → ℝ) 
  (h : ∀ n m : ℕ, n ≥ m → f(n + m) + f(n - m) = f(3 * n))
  : ∀ n : ℕ, f n = 0 :=
begin
  sorry
end

end unique_zero_function_l306_306486


namespace corporate_event_handshakes_l306_306437

def GroupHandshakes (A B C : Nat) (knows_all_A : Nat) (knows_none : Nat) (C_knows_none : Nat) : Nat :=
  -- Handshakes between Group A and Group B
  let handshakes_AB := knows_none * A
  -- Handshakes within Group B
  let handshakes_B := (knows_none * (knows_none - 1)) / 2
  -- Handshakes between Group B and Group C
  let handshakes_BC := B * C_knows_none
  -- Total handshakes
  handshakes_AB + handshakes_B + handshakes_BC

theorem corporate_event_handshakes : GroupHandshakes 15 20 5 5 15 = 430 :=
by
  sorry

end corporate_event_handshakes_l306_306437


namespace event_distance_l306_306108

noncomputable def distance_to_event (cost_per_mile : ℝ) (days : ℕ) (rides_per_day : ℕ) (total_cost : ℝ) : ℝ :=
  total_cost / (days * rides_per_day * cost_per_mile)

theorem event_distance 
  (cost_per_mile : ℝ)
  (days : ℕ)
  (rides_per_day : ℕ)
  (total_cost : ℝ)
  (h1 : cost_per_mile = 2.5)
  (h2 : days = 7)
  (h3 : rides_per_day = 2)
  (h4 : total_cost = 7000) : 
  distance_to_event cost_per_mile days rides_per_day total_cost = 200 :=
by {
  sorry
}

end event_distance_l306_306108


namespace sodas_drunk_robin_sodas_l306_306310

theorem sodas_drunk (total_sodas : ℕ) (extra_sodas : ℕ) (drunk_sodas : ℕ) 
  (h1 : total_sodas = 11)
  (h2 : extra_sodas = 8)
  : drunk_sodas = total_sodas - extra_sodas :=
by
  -- Proof omitted
  sorry

theorem robin_sodas : (11 - 8) = 3 :=
by
  exact calc
    11 - 8 = 3 : by rfl

end sodas_drunk_robin_sodas_l306_306310


namespace A_less_B_C_A_relationship_l306_306507

variable (a : ℝ)
def A := a + 2
def B := 2 * a^2 - 3 * a + 10
def C := a^2 + 5 * a - 3

theorem A_less_B : A a - B a < 0 := by
  sorry

theorem C_A_relationship :
  if a < -5 then C a > A a
  else if a = -5 then C a = A a
  else if a < 1 then C a < A a
  else if a = 1 then C a = A a
  else C a > A a := by
  sorry

end A_less_B_C_A_relationship_l306_306507


namespace choir_third_verse_joiners_l306_306409

theorem choir_third_verse_joiners:
  let total_singers := 30 in
  let first_verse_singers := total_singers / 2 in
  let remaining_after_first := total_singers - first_verse_singers in
  let second_verse_singers := remaining_after_first / 3 in
  let remaining_after_second := remaining_after_first - second_verse_singers in
  let third_verse_singers := remaining_after_second in
  third_verse_singers = 10 := 
by
  sorry

end choir_third_verse_joiners_l306_306409


namespace part1_part2_part3_l306_306332

noncomputable def f : ℝ → ℝ :=
sorry  -- Define f with the properties given in the problem

-- Conditions
axiom Dom (x : ℝ) : 0 < x → f x ∈ ℝ -- Domain of f is (0, +∞)
axiom Prop1 (x y : ℝ) : 0 < x → 0 < y → f (x / y) = f x - f y -- Functional equation property
axiom Prop2 (x : ℝ) : 1 < x → 0 < f x -- When x > 1, f(x) > 0

-- Questions with given solutions
theorem part1 : f 1 = 0 := sorry

theorem part2 (x₁ x₂ : ℝ) (h₁: 0 < x₁) (h₂: 0 < x₂) (h: x₁ < x₂) : f x₁ < f x₂ := sorry

theorem part3 (f4 : f 4 = 2) : set.range (λ x, f x) (set.Icc 1 16) = set.Icc 0 4 := sorry

end part1_part2_part3_l306_306332


namespace arithmetic_mean_a_X_l306_306499

theorem arithmetic_mean_a_X (M : Set ℕ) (hM : M = {i | 1 ≤ i ∧ i ≤ 2021}) :
  let a_X (X : Set ℕ) := X.max' (by { sorry }) + X.min' (by { sorry }) in
  (∑ X in (Set.powerset M).filter (λ X, X ≠ ∅), a_X X) / ((2^2021) - 1) = 2 :=
by
  sorry

end arithmetic_mean_a_X_l306_306499


namespace three_room_partition_l306_306076

open Finset

noncomputable def possible_partition (G : Type) [Fintype G] (knows : G → G → Prop) {h : ∀ a b, knows a b → knows b a}
  (no_four_chain : ∀ a b c d, ¬(knows a b ∧ knows b c ∧ knows c d)): Prop :=
  ∃ (rooms : G → Fin 3), ∀ a b, rooms a = rooms b → ¬knows a b

-- Proof is omitted.
theorem three_room_partition (G : Type) [Fintype G] (knows : G → G → Prop) {h : ∀ a b, knows a b → knows b a}
  (no_four_chain : ∀ a b c d, ¬(knows a b ∧ knows b c ∧ knows c d)) : possible_partition G knows :=
sorry

end three_room_partition_l306_306076


namespace original_number_of_matchsticks_l306_306831

-- Define the conditions
def matchsticks_per_house : ℕ := 10
def houses_created : ℕ := 30
def total_matchsticks_used := houses_created * matchsticks_per_house

-- Define the question and the proof goal
theorem original_number_of_matchsticks (h : total_matchsticks_used = (Michael's_original_matchsticks / 2)) :
  (Michael's_original_matchsticks = 600) :=
by
  sorry

end original_number_of_matchsticks_l306_306831


namespace length_of_PQ_l306_306131

open Real

theorem length_of_PQ (r : ℝ) (h₁ : 1.5 < r) (h₂ : r < 2.5) : 
  let A := (0, 0)
  let B := (3, 0)
  let D := (0, 3)
  let P := (1.5, sqrt (r^2 - 2.25))
  let Q := (sqrt (r^2 - 2.25), 1.5)
  in dist P Q = 3 :=
begin
  -- Given conditions
  sorry
end

end length_of_PQ_l306_306131


namespace greatest_value_of_n_l306_306921

theorem greatest_value_of_n (n : ℤ) (h : 101 * n ^ 2 ≤ 3600) : n ≤ 5 :=
by
  sorry

end greatest_value_of_n_l306_306921


namespace sum_of_first_60_terms_l306_306182

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) + (-1)^n * a n = 3 * n - 1

theorem sum_of_first_60_terms (a : ℕ → ℤ) (h : sequence a) :
  (Finset.range 60).sum a = 2760 :=
sorry

end sum_of_first_60_terms_l306_306182


namespace line_tangent_to_ellipse_l306_306733

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 → ∃! y : ℝ, 3 * x^2 + 6 * y^2 = 6) →
  m^2 = 3 / 2 :=
by
  sorry

end line_tangent_to_ellipse_l306_306733


namespace complement_union_l306_306617

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306617


namespace circle_tangent_radius_l306_306218

theorem circle_tangent_radius (r : ℝ) (hr_pos : r > 0) :
  let O1_eq := ∀ x y : ℝ, x^2 + y^2 + 4 * x - 8 * y - 5 = 0
  let O2_eq := ∀ x y : ℝ, (x + 2)^2 + y^2 = r^2
  (∃ P : ℝ × ℝ , O1_eq P.1 P.2 ∧ O2_eq P.1 P.2) ∧
  (¬ ∃ Q₁ Q₂ : ℝ × ℝ, O1_eq Q₁.1 Q₁.2 ∧ O2_eq Q₂.1 Q₂.2 ∧ Q₁ ≠ Q₂) →
  r = 1 ∨ r = 9 :=
begin
  intros O1_eq O2_eq H,
  sorry
end

end circle_tangent_radius_l306_306218


namespace a_correct_T_correct_l306_306561

-- Given the sequence and sum conditions
def S (n : ℕ) : ℕ := n^2 + 2 * n + 3

-- Define the sequence a_n based on conditions
def a : ℕ → ℕ
| 1     := 6
| (n+2) := 2 * (n + 2) + 1

-- Define T_n based on conditions and a_n
def T (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2^i.succ) * a (n - i)

theorem a_correct :
  a 1 = 6 ∧ (∀ n ≥ 2, a n = 2 * n + 1) := by
  -- Proof not required
  sorry

theorem T_correct (n : ℕ) :
  T n = 13 * 2^n - 4 * n - 10 := by
  -- Proof not required
  sorry

end a_correct_T_correct_l306_306561


namespace sum_of_k_for_double_root_eq_seven_l306_306013

theorem sum_of_k_for_double_root_eq_seven :
  let discriminant (a b c : ℝ) := b^2 - 4 * a * c
  in {k : ℝ | discriminant 1 (2*k) (7*k - 10) = 0}.sum = 7 :=
by
  let discriminant (a b c : ℝ) := b^2 - 4 * a * c
  sorry

end sum_of_k_for_double_root_eq_seven_l306_306013


namespace range_of_a_l306_306731

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → deriv (f a x) ≤ 0) ↔ a ≥ Real.exp 1 + 1 :=
by
  sorry

end range_of_a_l306_306731


namespace complement_of_union_is_singleton_five_l306_306642

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306642


namespace smallest_root_of_polynomial_l306_306906

theorem smallest_root_of_polynomial :
  ∃ x : ℝ, (24 * x^3 - 106 * x^2 + 116 * x - 70 = 0) ∧ x = 0.67 :=
by
  sorry

end smallest_root_of_polynomial_l306_306906


namespace domain_of_f_l306_306489

def f (x : ℝ) : ℝ := sqrt (2 * sin x - 1) + sqrt (-x^2 + 6 * x)

theorem domain_of_f :
  {x : ℝ | 0 ≤ 2 * sin x - 1 ∧ 0 ≤ -x^2 + 6 * x} = set.Icc (real.pi / 6) (5 * real.pi / 6) :=
by
  sorry

end domain_of_f_l306_306489


namespace complement_of_union_is_singleton_five_l306_306643

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306643


namespace complement_union_eq_singleton_five_l306_306572

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306572


namespace complement_union_M_N_l306_306633

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306633


namespace right_triangle_third_side_l306_306732

theorem right_triangle_third_side (a b c : ℝ) (ha : a = 8) (hb : b = 6) (h_right_triangle : a^2 + b^2 = c^2) :
  c = 10 :=
by
  sorry

end right_triangle_third_side_l306_306732


namespace min_positive_S_l306_306152

noncomputable def main : ℕ :=
  let a : fin 150 → ℤ := fun i => if i.val < 82 then 1 else -1 
  let sum_a := finset.univ.sum (λ i : fin 150, a i)
  let sum_a_squared := sum_a ^ 2
  (sum_a_squared - 150) / 2

theorem min_positive_S :
  ∃ S : ℕ, S = 23 ∧ ∀ (a : fin 150 → ℤ), (∀ i, a i = 1 ∨ a i = -1) →
    S ≤ (∑ i in finset.univ, ∑ j in finset.Ico 0 i, a i * a j) :=
sorry

end min_positive_S_l306_306152


namespace complement_of_union_is_singleton_five_l306_306641

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306641


namespace distance_between_chords_l306_306889

-- Definitions based on the conditions
structure CircleGeometry where
  radius: ℝ
  d1: ℝ -- distance from the center to the closest chord (34 units)
  d2: ℝ -- distance from the center to the second chord (38 units)
  d3: ℝ -- distance from the center to the outermost chord (38 units)

-- The problem itself
theorem distance_between_chords (circle: CircleGeometry) (h1: circle.d2 = 3) (h2: circle.d1 = 3 * circle.d2) (h3: circle.d3 = circle.d2) :
  2 * circle.d2 = 6 :=
by
  sorry

end distance_between_chords_l306_306889


namespace water_calculation_l306_306367

theorem water_calculation : 
  let W := 40 in
  let C := W / 2 in
  let C_current := (7/8 : ℝ) * C in
  let W_current := (3/4 : ℝ) * W in
  let A := (3/2 : ℝ) * W in
  let A_current := (2/3 : ℝ) * A - 5 in
  let B := C / 2 in
  let B_current := (5/8 : ℝ) * B in
  C_current + W_current + A_current + B_current = 89 :=
by
  sorry

end water_calculation_l306_306367


namespace complement_union_l306_306621

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306621


namespace final_price_of_purchases_l306_306146

theorem final_price_of_purchases :
  let electronic_discount := 0.20
  let clothing_discount := 0.15
  let bundle_discount := 10
  let voucher_threshold := 200
  let voucher_value := 20
  let voucher_limit := 2
  let delivery_charge := 15
  let tax_rate := 0.08

  let electronic_original_price := 150
  let clothing_original_price := 80
  let num_clothing := 2

  -- Calculate discounts
  let electronic_discount_amount := electronic_original_price * electronic_discount
  let electronic_discount_price := electronic_original_price - electronic_discount_amount
  let clothing_discount_amount := clothing_original_price * clothing_discount
  let clothing_discount_price := clothing_original_price - clothing_discount_amount

  -- Sum of discounted clothing items
  let total_clothing_discount_price := clothing_discount_price * num_clothing

  -- Calculate bundle discount
  let total_before_bundle_discount := electronic_discount_price + total_clothing_discount_price
  let total_after_bundle_discount := total_before_bundle_discount - bundle_discount

  -- Calculate vouchers
  let num_vouchers := if total_after_bundle_discount >= voucher_threshold * 2 then voucher_limit else 
                      if total_after_bundle_discount >= voucher_threshold then 1 else 0
  let total_voucher_amount := num_vouchers * voucher_value
  let total_after_voucher_discount := total_after_bundle_discount - total_voucher_amount

  -- Add delivery charge
  let total_before_tax := total_after_voucher_discount + delivery_charge

  -- Calculate tax
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount

  final_price = 260.28 :=
by
  -- the actual proof will be included here
  sorry

end final_price_of_purchases_l306_306146


namespace problems_per_page_l306_306928

theorem problems_per_page (total_problems finished_problems remaining_pages : Nat) (h1 : total_problems = 101) 
  (h2 : finished_problems = 47) (h3 : remaining_pages = 6) :
  (total_problems - finished_problems) / remaining_pages = 9 :=
by
  sorry

end problems_per_page_l306_306928


namespace exists_arith_prog_perfect_square_l306_306167

theorem exists_arith_prog_perfect_square:
  ∃ (a d : ℤ), 
    (∃ k : ℤ, (a - d) + a = k^2) ∧ 
    (∃ m : ℤ, a + (a + d) = m^2) ∧ 
    (∃ n : ℤ, (a - d) + (a + d) = n^2) ∧ 
    (a = 3362) ∧ 
    (d = 2880) :=
by
  -- Definitions based on conditions:
  let a := 3362
  let d := 2880
  exists a, d
  split
  -- Condition 1: (a - d) + a = k^2
  { use 62
    rw [←Int.add_sub_assoc, ←Int.add_assoc]
    norm_num },
  split
  -- Condition 2: a + (a + d) = m^2
  { use 82
    rw [←Int.add_assoc, ←Int.add_assoc]
    norm_num },
  split
  -- Condition 3: (a - d) + (a + d) = n^2
  { use 98
    simp [Int.add_sub_cancel]
    norm_num },
  -- Ensure correct values for a and d
  { norm_num },
  { norm_num }

end exists_arith_prog_perfect_square_l306_306167


namespace installation_time_l306_306396

theorem installation_time (total_windows installed_windows time_per_window : ℕ)
                          (h1 : total_windows = 9)
                          (h2 : installed_windows = 6)
                          (h3 : time_per_window = 6) :
    (total_windows - installed_windows) * time_per_window = 18 :=
by
    rw [h1, h2, h3]
    show (9 - 6) * 6 = 18
    simp
    sorry

end installation_time_l306_306396


namespace complement_union_M_N_l306_306627

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306627


namespace x_value_when_y_is_two_l306_306858

noncomputable def k : ℝ := 4

theorem x_value_when_y_is_two : ∃ x : ℝ, (x * (2 : ℝ)^3) = k ∧ (k = 4) := 
by
  use 1/2
  split
  · norm_num
  · exact rfl

end x_value_when_y_is_two_l306_306858


namespace problem1_solution_problem2_solution_l306_306441

noncomputable def problem1 : Real :=
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2)

noncomputable def problem2 : Real :=
  (2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6)

theorem problem1_solution : problem1 = 0 := by
  sorry

theorem problem2_solution : problem2 = 6 := by
  sorry

end problem1_solution_problem2_solution_l306_306441


namespace selected_six_numbers_have_two_correct_statements_l306_306174

def selection := {n : ℕ // 1 ≤ n ∧ n ≤ 11}

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_multiple (a b : ℕ) : Prop := a ≠ b ∧ (b % a = 0 ∨ a % b = 0)

def is_double_multiple (a b : ℕ) : Prop := a ≠ b ∧ (2 * a = b ∨ 2 * b = a)

theorem selected_six_numbers_have_two_correct_statements (s : Finset selection) (h : s.card = 6) :
  ∃ n1 n2 : selection, is_coprime n1.1 n2.1 ∧ ∃ n1 n2 : selection, is_double_multiple n1.1 n2.1 :=
by
  -- The detailed proof is omitted.
  sorry

end selected_six_numbers_have_two_correct_statements_l306_306174


namespace sum_of_x_where_median_equals_mean_and_x_less_than_9_l306_306033

theorem sum_of_x_where_median_equals_mean_and_x_less_than_9 :
  let numbers := [3, 7, 9, 20] in 
  ∃ x : ℝ, 
    (x < 9) ∧ 
    let mean := (3 + 7 + 9 + 20 + x) / 5 in 
    let median := if x < 7 then 7 else if x < 9 then x else 9 in
    median = mean ∧ 
    x = -4 :=
by 
  sorry

end sum_of_x_where_median_equals_mean_and_x_less_than_9_l306_306033


namespace range_of_lambda_over_m_l306_306699

variable (λ m α : ℝ)
def vector_a := (λ + 2, λ^2 - sqrt 3 * cos (2 * α))
def vector_b := (m, m / 2 + sin α * cos α)

theorem range_of_lambda_over_m :
  vector_a λ m α = (2 * fst (vector_b m α), 2 * snd (vector_b m α)) →
  -6 ≤ λ / m ∧ λ / m ≤ 1 := sorry

end range_of_lambda_over_m_l306_306699


namespace max_blocks_fit_l306_306898

-- Define the dimensions of the block
def block_length := 2
def block_width := 3
def block_height := 1

-- Define the dimensions of the container box
def box_length := 4
def box_width := 3
def box_height := 3

-- Define the volume calculations
def volume (length width height : ℕ) : ℕ := length * width * height

def block_volume := volume block_length block_width block_height
def box_volume := volume box_length box_width box_height

-- The theorem to prove
theorem max_blocks_fit : (box_volume / block_volume) = 6 :=
by
  sorry

end max_blocks_fit_l306_306898


namespace total_products_produced_by_B_l306_306353

-- Definitions based on conditions
variables (total_products : ℕ) (sample_size : ℕ)
variables (sampled_A : ℕ) (produced_A produced_B : ℕ)

-- Given conditions
def condition_1 : total_products = 4800 := sorry 
def condition_2 : sample_size = 80 := sorry
def condition_3 : sampled_A = 50 := sorry

-- Statement to prove
theorem total_products_produced_by_B : produced_B = 1800 :=
by
  -- conditions
  have h1: total_products = 4800 := sorry,
  have h2: sample_size = 80 := sorry,
  have h3: sampled_A = 50 := sorry,
  -- calculate B's output
  have ratio : (50 + (sample_size - sampled_A)) = 80 := sorry,
  sorry

end total_products_produced_by_B_l306_306353


namespace binom_16_12_eq_1820_l306_306127

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l306_306127


namespace permutation_exists_l306_306282

def D : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_bijection (f : ℕ → ℕ) : Prop :=
  ∀ x ∈ D, f(x) ∈ D ∧ ∀ y ∈ D, ∃ x' ∈ D, f(x') = y

def f_iteration (f : ℕ → ℕ) : ℕ → (ℕ → ℕ)
| 0 => id
| (n+1) => f ∘ (f_iteration f n)

theorem permutation_exists (f : ℕ → ℕ) (h_bij : is_bijection f) :
  ∃ (σ : Permutation { x // x ∈ D }),
  sum (D.attach.map (λ i, (λ⟨i, hi⟩, σ i * f_iteration f 2520 i))) = 220 :=
sorry

end permutation_exists_l306_306282


namespace garden_area_increase_l306_306948

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l306_306948


namespace area_increase_correct_l306_306937

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l306_306937


namespace trajectory_translation_sin_cos_l306_306878

theorem trajectory_translation_sin_cos
  (f : ℝ → ℝ)
  (c1 : ∀ x, f x = sin x)
  (d : ℝ := 1/2 * real.sqrt (π^2 + 4))
  (v : ℝ × ℝ := (π/4, -1/2)) :
  ∀ x, (∃ t, ((x - t) = d * (π/4)) ∧ (f t = sin (x - d * (π/4)) - d * (-1/2)))
    → f x = -2 * cos^2 (x / 2) :=
sorry

end trajectory_translation_sin_cos_l306_306878


namespace sum_of_n_conditions_l306_306277

theorem sum_of_n_conditions (T : ℕ) (h : T = ∑ n in (finset.filter (λ n, ∃ m : ℕ, n^2 + 14 * n - 2009 = m^2) (finset.Icc 1 2009)), n) : T % 1000 = 59 :=
by
sorry

end sum_of_n_conditions_l306_306277


namespace third_quadrant_angles_l306_306348

theorem third_quadrant_angles :
  {α : ℝ | ∃ k : ℤ, π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π} =
  {α | π < α ∧ α < 3 * π / 2} :=
sorry

end third_quadrant_angles_l306_306348


namespace probability_of_x_gt_5y_l306_306836

noncomputable theory

open Set

def rectangle : Set (ℝ × ℝ) := 
  { p : ℝ × ℝ | 
  (0 ≤ p.1 ∧ p.1 ≤ 500) ∧ 
  (0 ≤ p.2 ∧ p.2 ≤ 600) }

def triangle : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | 
  (0 ≤ p.1 ∧ p.1 ≤ 500) ∧ 
  (0 ≤ p.2 ∧ p.2 ≤ 600) ∧ 
  p.2 < 1/5 * p.1 }

theorem probability_of_x_gt_5y :
  (μ (triangle) / μ (rectangle) = 1 / 12) :=
sorry

end probability_of_x_gt_5y_l306_306836


namespace regular_polygon_sides_l306_306476

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end regular_polygon_sides_l306_306476


namespace semicircle_perimeter_calc_l306_306392

noncomputable def pi_approx : ℝ := 3.14

def radius : ℝ := 3.1

def perimeter_of_semicircle (r : ℝ) : ℝ :=
  pi_approx * r + 2 * r

theorem semicircle_perimeter_calc :
  radius = 3.1 → perimeter_of_semicircle radius = 15.934 :=
begin
  intro h,
  unfold perimeter_of_semicircle,
  rw h,
  simp [pi_approx],
  norm_num,
end

end semicircle_perimeter_calc_l306_306392


namespace square_of_binomial_is_25_l306_306717

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l306_306717


namespace complement_union_eq_l306_306589

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306589


namespace complement_union_eq_singleton_five_l306_306571

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306571


namespace sqrt_sum_leq_sqrt_quad_l306_306276

theorem sqrt_sum_leq_sqrt_quad (E : Point) (A B C D : Point) (F1 F2 F3 : ℝ) 
  (h1 : E ∈ Line (A, C)) (h2 : E ∈ Line (B, D)) 
  (hF1 : F1 = Area (Triangle A B E)) 
  (hF2 : F2 = Area (Triangle C D E)) 
  (hF3 : F3 = Area (Quadrilateral A B C D)) :
  sqrt F1 + sqrt F2 ≤ sqrt F3 :=
sorry

end sqrt_sum_leq_sqrt_quad_l306_306276


namespace joe_left_pocket_initial_l306_306390

-- Definitions from conditions
def total_money : ℕ := 200
def initial_left_pocket (L : ℕ) : ℕ := L
def initial_right_pocket (R : ℕ) : ℕ := R
def transfer_one_fourth (L : ℕ) : ℕ := L - L / 4
def add_to_right (R : ℕ) (L : ℕ) : ℕ := R + L / 4
def transfer_20 (L : ℕ) : ℕ := transfer_one_fourth L - 20
def add_20_to_right (R : ℕ) (L : ℕ) : ℕ := add_to_right R L + 20

-- Statement to prove
theorem joe_left_pocket_initial (L R : ℕ) (h₁ : L + R = total_money) 
  (h₂ : transfer_20 L = add_20_to_right R L) : 
  initial_left_pocket L = 160 :=
by
  sorry

end joe_left_pocket_initial_l306_306390


namespace solve_diamond_op_l306_306404

theorem solve_diamond_op (a b c x : ℝ) (h₁ : ∀ a b c, a ≠ 0 → b ≠ 0 → c ≠ 0 → a ◇ (b ◇ c) = (a ◇ b) * c) (h₂ : ∀ a, a ≠ 0 → a ◇ a = 1)
  (h_eq : 504 ◇ (12 ◇ x) = 50) :
  x = 25 / 21 :=
begin
  sorry
end

end solve_diamond_op_l306_306404


namespace mul_sqrt_sqrt_2_mul_sqrt_6_eq_2_sqrt_3_l306_306130

-- Defining the multiplication rule for square roots 
theorem mul_sqrt {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) : (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) := 
  Real.sqrt_mul ha hb

-- Defining the main theorem to be proved
theorem sqrt_2_mul_sqrt_6_eq_2_sqrt_3 : (Real.sqrt 2) * (Real.sqrt 6) = 2 * (Real.sqrt 3) := 
by
  -- Apply the multiplication rule for square roots
  have h := mul_sqrt (by linarith) (by linarith)
  rw [←Real.mul_self_inj_of_nonneg (Real.sqrt_nonneg 12) (by norm_num : (2:ℝ) * Real.sqrt 3 ≥ 0), ←Real.sqrt_mul]
  rw [Real.sqrt_mul (by norm_num : 4 ≥ 0) (by norm_num : 3 ≥ 0), Real.sqrt_sq (by norm_num : (2:ℝ).pow 2 = 4), mul_comm]
  exact h
sorry

end mul_sqrt_sqrt_2_mul_sqrt_6_eq_2_sqrt_3_l306_306130


namespace quantiville_jacket_junction_l306_306763

theorem quantiville_jacket_junction :
  let sales_tax_rate := 0.07
  let original_price := 120.0
  let discount := 0.25
  let amy_total := (original_price * (1 + sales_tax_rate)) * (1 - discount)
  let bob_total := (original_price * (1 - discount)) * (1 + sales_tax_rate)
  let carla_total := ((original_price * (1 + sales_tax_rate)) * (1 - discount)) * (1 + sales_tax_rate)
  (carla_total - amy_total) = 6.744 :=
by
  sorry

end quantiville_jacket_junction_l306_306763


namespace estimate_passed_students_l306_306173

-- Definitions for the given conditions
def total_papers_in_city : ℕ := 5000
def papers_selected : ℕ := 400
def papers_passed : ℕ := 360

-- The theorem stating the problem in Lean
theorem estimate_passed_students : 
    (5000:ℕ) * ((360:ℕ) / (400:ℕ)) = (4500:ℕ) :=
by
  -- Providing a trivial sorry to skip the proof.
  sorry

end estimate_passed_students_l306_306173


namespace xiangming_payment_methods_count_l306_306046

def xiangming_payment_methods : Prop :=
  ∃ x y z : ℕ, 
    x + y + z ≤ 10 ∧ 
    x + 2 * y + 5 * z = 18 ∧ 
    ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0))

theorem xiangming_payment_methods_count : 
  xiangming_payment_methods → ∃! n, n = 11 :=
by sorry

end xiangming_payment_methods_count_l306_306046


namespace christen_potatoes_peeled_l306_306222

-- Define the initial conditions and setup
def initial_potatoes := 50
def homer_rate := 4
def christen_rate := 6
def time_homer_alone := 5
def combined_rate := homer_rate + christen_rate

-- Calculate the number of potatoes peeled by Homer alone in the first 5 minutes
def potatoes_peeled_by_homer_alone := time_homer_alone * homer_rate

-- Calculate the remaining potatoes after Homer peeled alone
def remaining_potatoes := initial_potatoes - potatoes_peeled_by_homer_alone

-- Calculate the time taken for Homer and Christen to peel the remaining potatoes together
def time_to_finish_together := remaining_potatoes / combined_rate

-- Calculate the number of potatoes peeled by Christen during the shared work period
def potatoes_peeled_by_christen := christen_rate * time_to_finish_together

-- The final theorem we need to prove
theorem christen_potatoes_peeled : potatoes_peeled_by_christen = 18 := by
  sorry

end christen_potatoes_peeled_l306_306222


namespace area_of_region_b_l306_306456

open Complex Real Set

noncomputable def region_in_complex_plane : Set ℂ :=
  { z : ℂ | ∀ (x y : ℝ), z = x + y * Complex.I ∧ 
    (0 ≤ x / 50) ∧ (x / 50 ≤ 1) ∧ (0 ≤ y / 50) ∧ (y / 50 ≤ 1) ∧
    (0 ≤ 50 * x / (x^2 + y^2)) ∧ (50 * x / (x^2 + y^2) ≤ 1) ∧ 
    (0 ≤ 50 * y / (x^2 + y^2)) ∧ (50 * y / (x^2 + y^2) ≤ 1) }

theorem area_of_region_b : measure_theory.measure.restrict measure_theory.lebesgue region_in_complex_plane = 1875 - 312.5 * real.pi :=
sorry

end area_of_region_b_l306_306456


namespace find_hidden_cards_l306_306884

def card_positions (cards : Fin 9 → ℕ) : Prop :=
  ∃ A B C : Fin 9, 
  (cards A = 5) ∧ (cards B = 2) ∧ (cards C = 9) ∧
  ∀ i j k : Fin 9, i < j → j < k →
  (cards i < cards j → cards j < cards k) ∨ (cards i > cards j → cards j > cards k) →
  {1, 3, 4, 6, 7, 8} ⊆ cards '' univ.to_finset
  
theorem find_hidden_cards (cards : Fin 9 → ℕ) : card_positions cards :=
sorry

end find_hidden_cards_l306_306884


namespace min_ab_ge_one_min_a_plus_2b_ge_value_l306_306176

noncomputable def min_ab (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_cond : 1/a + 1/b = 2) : Prop :=
  ab : ℝ

theorem min_ab_ge_one (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_cond : 1/a + 1/b = 2) : ab := 
  sorry

noncomputable def min_a_plus_2b (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_cond : 1/a + 1/b = 2) : ℝ :=
  a + 2b

theorem min_a_plus_2b_ge_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_cond : 1/a + 1/b = 2) : a + 2b := 
  sorry

end min_ab_ge_one_min_a_plus_2b_ge_value_l306_306176


namespace complement_union_l306_306620

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306620


namespace hexagon_bc_de_eq_14_l306_306248

theorem hexagon_bc_de_eq_14
  (α β γ δ ε ζ : ℝ)
  (angle_cond : α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ)
  (AB BC CD DE EF FA : ℝ)
  (sum_AB_BC : AB + BC = 11)
  (diff_FA_CD : FA - CD = 3)
  : BC + DE = 14 := sorry

end hexagon_bc_de_eq_14_l306_306248


namespace percentage_of_flutes_got_in_l306_306296

theorem percentage_of_flutes_got_in :
  let total_flutes := 20
  let total_clarinets := 30
  let total_trumpets := 60
  let total_pianists := 20
  let total_band := 53
  let clarinets_got_in := total_clarinets / 2
  let trumpets_got_in := total_trumpets / 3
  let pianists_got_in := total_pianists / 10
  let other_musicians := clarinets_got_in + trumpets_got_in + pianists_got_in
  let flutes_got_in := total_band - other_musicians
  (flutes_got_in / total_flutes) * 100 = 80 := by
begin
  -- Definitions and step-by-step calculations would go here
  sorry
end

end percentage_of_flutes_got_in_l306_306296


namespace cannot_determine_degree_from_A_P_l306_306783

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l306_306783


namespace complement_union_eq_singleton_five_l306_306577

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306577


namespace inequality_proof_equality_iff_l306_306822

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) 
  (hx_nonneg : ∀ i, 0 ≤ x i) 
  (a : ℝ) (ha_def : a = Finset.min' (Finset.univ.image x) (Finset.card_pos.mpr $ by simp)) :
  (∑ j in Finset.range n, (1 + x j) / (1 + x ((j + 1) % n))) 
  ≤ n + (1 / (1 + a) ^ 2) * ∑ j in Finset.range n, (x j - a) ^ 2 :=
sorry

theorem equality_iff (n : ℕ) (x : Fin n → ℝ) 
  (hx_nonneg : ∀ i, 0 ≤ x i) 
  (a : ℝ) (ha_def : a = Finset.min' (Finset.univ.image x) (Finset.card_pos.mpr $ by simp)) :
  (∑ j in Finset.range n, (1 + x j) / (1 + x ((j + 1) % n))) = n + (1 / (1 + a) ^ 2) * ∑ j in Finset.range n, (x j - a) ^ 2 
    ↔ (∀ i j, x i = x j) :=
sorry

end inequality_proof_equality_iff_l306_306822


namespace correct_statements_l306_306357

theorem correct_statements :
  (∀ (a : ℝ), (∀ x ∈ ℝ, f x = a * sin x + cos x) → (is_symmetrical f (π/6) → a = √(1 + (1/a)^2))) ∧
  (∀ {m : ℝ}, angle_obtuse ⟨1, 2⟩ ⟨-2, m⟩ → m < 1) ∧
  (∀ {α : ℝ}, (5 * π / 2 < α ∧ α < 9 * π / 2) → ∀ a : ℝ, ∃ x₁ x₂ x₃, f x = sin x - log a x) ∧
  (∀ x : ℝ, (f x = x * sin x) →
    (∀ x ∈ interval_left_closed_right_open (-π/2) 0, derivative f x ≤ 0) ∧
    (∀ x ∈ interval_open_right_closed 0 (π/2), derivative f x ≥ 0)) →
    (statements_correct ∈ [1, 4] and statements_incorrect ∈ [2, 3])


end correct_statements_l306_306357


namespace find_x_value_l306_306339

noncomputable def x_value := 92

open BigOperators

section

variables (data : list ℝ) (x : ℝ)

theorem find_x_value :
  (mean data = x) ∧ (median data = x) ∧ (mode data = x) ∧ (frequency x data ≥ 2) ↔ x = 92 :=
begin
  sorry
end

end

end find_x_value_l306_306339


namespace FK_approx_two_cubed_root_l306_306482

noncomputable def unit_square : ℝ := 1

noncomputable def BE : ℝ := 1
noncomputable def AF : ℝ := 5 / 9

noncomputable def FC : ℝ := Real.sqrt (1 + (1 - 5 / 9)^2) / 9
noncomputable def FE : ℝ := Real.sqrt (4 + (5 / 9)^2) / 9

theorem FK_approx_two_cubed_root :
    let FK := (FE^2) / FC
    | FK - Real.cbrt 2 | < 0.00001 :=
by
  sorry

end FK_approx_two_cubed_root_l306_306482


namespace range_of_m_l306_306702

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - m * x + 3/2 > 0) ∨ (foci_on_x_axis m) → -Real.sqrt 6 < m ∧ m < 3 :=
by
  sorry

-- Additional definitions based on the conditions in a)
def condition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m * x + 3/2 > 0

def foci_on_x_axis (m : ℝ) : Prop := (m - 1 > 0) ∧ (3 - m > 0)

end range_of_m_l306_306702


namespace least_value_y_l306_306029

theorem least_value_y : ∃ y : ℝ, (3 * y ^ 3 + 3 * y ^ 2 + 5 * y + 1 = 5) ∧ ∀ z : ℝ, (3 * z ^ 3 + 3 * z ^ 2 + 5 * z + 1 = 5) → y ≤ z :=
sorry

end least_value_y_l306_306029


namespace find_number_l306_306166

noncomputable def calc1 : Float := 0.47 * 1442
noncomputable def calc2 : Float := 0.36 * 1412
noncomputable def diff : Float := calc1 - calc2

theorem find_number :
  ∃ (n : Float), (diff + n = 6) :=
sorry

end find_number_l306_306166


namespace soaked_part_solution_l306_306383

theorem soaked_part_solution 
  (a b : ℝ) (c : ℝ) 
  (h : c * (2/3) * a * b = 2 * a^2 * b^3 + (1/3) * a^3 * b^2) :
  c = 3 * a * b^2 + (1/2) * a^2 * b :=
by
  sorry

end soaked_part_solution_l306_306383


namespace decimal_equiv_half_squared_l306_306923

theorem decimal_equiv_half_squared :
  ((1 / 2 : ℝ) ^ 2) = 0.25 := by
  sorry

end decimal_equiv_half_squared_l306_306923


namespace fraction_to_decimal_l306_306042

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end fraction_to_decimal_l306_306042


namespace complement_union_eq_singleton_five_l306_306569

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306569


namespace mutually_exclusive_A_mutually_exclusive_C_l306_306378

-- Definitions based on conditions:
def hitsNineRings (a : Athlete) : Prop := a.score = 9
def hitsEightRings (a : Athlete) : Prop := a.score = 8

def bothHitTarget (a b : Athlete) : Prop := a.hits && b.hits
def neitherHitTarget (a b : Athlete) : Prop := ¬a.hits && ¬b.hits

-- Athlete type for generality
structure Athlete where
  score : ℕ
  hits : Prop

-- The problem to prove those events are mutually exclusive
theorem mutually_exclusive_A (a : Athlete) : hitsNineRings a ∧ hitsEightRings a → False := sorry

theorem mutually_exclusive_C (a b : Athlete) : bothHitTarget a b ∧ neitherHitTarget a b → False := sorry

end mutually_exclusive_A_mutually_exclusive_C_l306_306378


namespace intersection_points_l306_306374

-- Define the four line equations
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℝ) : Prop := 5 * x - 15 * y = 15

-- State the theorem for intersection points
theorem intersection_points : 
  (line1 (18/11) (13/11) ∧ line2 (18/11) (13/11)) ∧ 
  (line2 (21/11) (8/11) ∧ line3 (21/11) (8/11)) :=
by
  sorry

end intersection_points_l306_306374


namespace max_increase_flow_rate_l306_306101

theorem max_increase_flow_rate :
  let initial_pipes_AB := 10
  let initial_pipes_BC := 10
  let flow_increase_per_swap := 40 in
  -- Maximum swap operations is half of initial_pipes_AB
  -- since we need to balance the sections
  let max_swaps := initial_pipes_AB / 2 in
  max_swaps * flow_increase_per_swap = 200 :=
by
  sorry

end max_increase_flow_rate_l306_306101


namespace complement_union_l306_306602

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306602


namespace complement_of_union_is_singleton_five_l306_306638

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306638


namespace complement_union_l306_306600

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306600


namespace problem_conditions_equation_right_triangle_vertex_coordinates_l306_306201

theorem problem_conditions_equation : 
  ∃ (a b c : ℝ), a = -1 ∧ b = -2 ∧ c = 3 ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - (-(x + 1))^2 + 4) ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - x^2 - 2 * x + 3)
:= sorry

theorem right_triangle_vertex_coordinates :
  ∀ x y : ℝ, x = -1 ∧ 
  (y = -2 ∨ y = 4 ∨ y = (3 + (17:ℝ).sqrt) / 2 ∨ y = (3 - (17:ℝ).sqrt) / 2)
  ∧ 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (-3, 0)
  let C : ℝ × ℝ := (0, 3)
  let P : ℝ × ℝ := (x, y)
  let BC : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2
  let PB : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let PC : ℝ := (P.1 - C.1)^2 + (P.2 - C.2)^2
  (BC + PB = PC ∨ BC + PC = PB ∨ PB + PC = BC)
:= sorry

end problem_conditions_equation_right_triangle_vertex_coordinates_l306_306201


namespace Linda_cookies_l306_306294

/-- 
Given the following:
- Linda has 24 classmates.
- She baked 2 batches of chocolate chip cookies.
- She baked 1 batch of oatmeal raisin cookies.
- She plans to bake 2 more batches of cookies.
- Each batch yields exactly 4 dozen cookies.

Prove that Linda can give each student exactly 10 cookies.
-/
theorem Linda_cookies (n_classmates : ℕ) (n_choco_batches : ℕ) (n_oatmeal_batches : ℕ) (n_more_batches : ℕ) (dozen : ℕ) :
    n_classmates = 24 →
    n_choco_batches = 2 →
    n_oatmeal_batches = 1 →
    n_more_batches = 2 →
    dozen = 12 →
    (n_choco_batches + n_oatmeal_batches + n_more_batches) * 4 * dozen / n_classmates = 10 :=
by
  intros hc ho hm hz
  refine eq.trans _ (eq_of_div_eq (nat_of_int 240) (nat.succ_pos' 23))
  ring
  sorry
  sorry

end Linda_cookies_l306_306294


namespace sandwich_cost_l306_306908

theorem sandwich_cost (soda_cost sandwich_cost total_cost : ℝ) (h1 : soda_cost = 0.87) (h2 : total_cost = 10.46) (h3 : 4 * soda_cost + 2 * sandwich_cost = total_cost) :
  sandwich_cost = 3.49 :=
by
  sorry

end sandwich_cost_l306_306908


namespace complement_union_l306_306651

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306651


namespace quadratic_solution_range_l306_306462

noncomputable def quadratic_inequality_real_solution (c : ℝ) : Prop :=
  0 < c ∧ c < 16

theorem quadratic_solution_range :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 8 * x + c < 0) ↔ quadratic_inequality_real_solution c :=
by
  intro c
  simp only [quadratic_inequality_real_solution]
  sorry

end quadratic_solution_range_l306_306462


namespace complement_union_eq_l306_306585

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306585


namespace point_inside_circle_chord_l306_306016

theorem point_inside_circle_chord (M : Point) (S : Circle) (A B P Q : Point)
  (hM_inside : M ∈ interior_of(S))
  (hAB_chord : chord(S, A, B))
  (hMP_perp : perpendicular(M, P, tangent_through(A)))
  (hMQ_perp : perpendicular(M, Q, tangent_through(B)))
  (R : ℝ) (hR : radius(S) = R) :
  ∃ K : ℝ, (∀ AB : chord(S, A, B), (1 / PM + 1 / QM = K)) :=
by 
  sorry

end point_inside_circle_chord_l306_306016


namespace line_intersects_circle_and_passes_through_center_l306_306139

-- Define the line and circle as given conditions
def line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 5 = 0

def circle (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 4 * x - 2 * y + 1 = 0

-- Define the center and radius of the circle
def center : ℝ × ℝ := (1, 1 / 2)
def radius : ℝ := real.sqrt 3 / 2

-- Define the distance function from a point to a line
def distance (p : ℝ × ℝ) : ℝ :=
  |3 * p.1 + 4 * p.2 - 5| / real.sqrt (3^2 + 4^2)

-- The theorem stating the problem
theorem line_intersects_circle_and_passes_through_center : 
  ∃ p : ℝ × ℝ, line p.1 p.2 ∧ circle p.1 p.2 ∧ distance center = 0 := 
sorry

end line_intersects_circle_and_passes_through_center_l306_306139


namespace four_digit_multiples_of_3_count_l306_306704

/-- Number of four-digit positive integers that are multiples of 3. -/
theorem four_digit_multiples_of_3_count : 
  let four_digit_numbers := {n | 1000 ≤ n ∧ n ≤ 9999},
      multiples_of_3 := {n | n % 3 = 0} in
  (four_digit_numbers ∩ multiples_of_3).card = 3000 := 
by
  sorry

end four_digit_multiples_of_3_count_l306_306704


namespace segments_divide_each_other_into_three_equal_parts_l306_306153

-- Define the points of the convex quadrilateral
variables (A B C D P1 P2 Q1 Q2 R1 R2 S1 S2 : Point)

-- Assume each side of the quadrilateral is divided into three equal parts
axiom AP1_P1P2_P2B : SegmentEquality (A, P1) (P1, P2) ∧ SegmentEquality (P1, P2) (P2, B)
axiom BQ1_Q1Q2_Q2C : SegmentEquality (B, Q1) (Q1, Q2) ∧ SegmentEquality (Q1, Q2) (Q2, C)
axiom CR1_R1R2_R2D : SegmentEquality (C, R1) (R1, R2) ∧ SegmentEquality (R1, R2) (R2, D)
axiom DS1_S1S2_S2A : SegmentEquality (D, S1) (S1, S2) ∧ SegmentEquality (S1, S2) (S2, A)

-- Connecting corresponding division points on opposite sides
def segment1 := Segment (P1, R1)
def segment2 := Segment (Q1, S1)

-- Define points at which these segments intersect
variable (IntersectionPoint : Point)
axiom Intersection_Segment1_Segment2 : Intersect segment1 segment2 IntersectionPoint

-- Proof statement declaration
theorem segments_divide_each_other_into_three_equal_parts :
  ProportionalDivision segment1 segment2 IntersectionPoint 1 2 3 :=
sorry

end segments_divide_each_other_into_three_equal_parts_l306_306153


namespace angle_ACB_eq_150_l306_306097

-- Define the conditions as constants

-- Point A near Quito, Ecuador
constant A_latitude : ℝ := 0
constant A_longitude : ℝ := -78

-- Point B near Vladivostok, Russia
constant B_latitude : ℝ := 43
constant B_longitude : ℝ := 132

-- The Earth's center
def C : Type := ℝ -- Assume C is the center of a perfect sphere of type real numbers

-- Define the spherical angle calculation function
constant calc_spherical_angle : ℝ → ℝ → ℝ → ℝ → ℝ

-- The Lean statement for the degree measure of the angle
theorem angle_ACB_eq_150 :
  calc_spherical_angle A_latitude A_longitude B_latitude B_longitude = 150 := sorry

end angle_ACB_eq_150_l306_306097


namespace max_m_value_l306_306511

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 1 / b = 1 / 4) : ∃ m : ℝ, (∀ a b : ℝ,  a > 0 ∧ b > 0 ∧ (2 / a + 1 / b = 1 / 4) → 2 * a + b ≥ 4 * m) ∧ m = 7 / 4 :=
sorry

end max_m_value_l306_306511


namespace complement_union_eq_singleton_five_l306_306566

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306566


namespace polynomial_e_value_l306_306869

theorem polynomial_e_value : 
  ∀ (a b c d e : ℤ), 
  ∃ (p : polynomial ℤ), 
  p.coeff 4 = a ∧ p.coeff 3 = b ∧ p.coeff 2 = c ∧ p.coeff 1 = d ∧ p.coeff 0 = e ∧ 
  (p.eval (-3) = 0 ∧ p.eval 6 = 0 ∧ p.eval 10 = 0 ∧ p.eval (-1 / 4 : ℚ) = 0) ∧ 
  0 < e ∧ 
  e = 180 :=
begin
  sorry
end

end polynomial_e_value_l306_306869


namespace surface_area_of_box_l306_306292

variable {l w h : ℝ}

def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * h + w * h + l * w)

theorem surface_area_of_box (l w h : ℝ) : surfaceArea l w h = 2 * (l * h + w * h + l * w) :=
by
  sorry

end surface_area_of_box_l306_306292


namespace bug_returns_to_A_at_8_l306_306275

noncomputable def P : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 3) * (1 - P n)

theorem bug_returns_to_A_at_8 : P 8 = 547 / 2187 := by
sorry

end bug_returns_to_A_at_8_l306_306275


namespace complement_union_of_M_and_N_l306_306689

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306689


namespace complement_union_of_M_and_N_l306_306697

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306697


namespace range_of_a_l306_306037

-- Given conditions definitions
variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

-- Goal statement
theorem range_of_a (h : ∀ x ∈ ℝ, 4 * f(a, x) > f(a, 0)) : a ∈ set.Icc (0 : ℝ) 1 ∪ set.Ici (4 : ℝ) :=
sorry

end range_of_a_l306_306037


namespace work_done_l306_306406

noncomputable def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 3

theorem work_done (W : ℝ) (h : W = ∫ x in (1:ℝ)..(5:ℝ), F x) : W = 112 :=
by sorry

end work_done_l306_306406


namespace closest_point_on_curve_is_correct_l306_306161

noncomputable def closest_point_to_curve : ℝ × ℝ :=
  ⟨(real.sqrt 2) / 2, 5 / 2⟩

theorem closest_point_on_curve_is_correct :
  ∀ (x y : ℝ), (x > 0) → (y = 3 - x^2) →
  (dist (x, y) (0, 2)) = dist closest_point_to_curve (0, 2) → 
  (x, y) = closest_point_to_curve := 
by sorry

end closest_point_on_curve_is_correct_l306_306161


namespace same_full_name_exists_l306_306412

/-
Given:
- A class of 33 students.
- Each student reports two numbers: the number of other students with the same first name and the same last name.
- Each number from 0 to 10 appears exactly once among these answers.

Prove:
- There are at least 2 students with the same full name.
-/

theorem same_full_name_exists :
  ∀ (students : Fin 33 → (Fin 11 × Fin 11)),
  (∀ n : Fin 11, ∃ i : Fin 33, students i.fst = n ∨ students i.snd = n) →
      ∃ (i j : Fin 33), i ≠ j ∧ students i = students j :=
begin
  sorry
end

end same_full_name_exists_l306_306412


namespace xy_sum_l306_306847

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x + y = 2 :=
sorry

end xy_sum_l306_306847


namespace area_of_region_R_l306_306319

theorem area_of_region_R (A B C D : Point) (square : is_square A B C D) (side_length : dist A B = 3)
  (angle_ABC : angle A B C = 120) : area (region_R A B C D) = 1.125 := 
by 
    sorry

end area_of_region_R_l306_306319


namespace bacteria_population_l306_306327

theorem bacteria_population (n : ℕ) 
  (h1 : ∀ t : ℕ, t = 300 → n * 2 ^ (t / 30) = 1_310_720) : 
  n = 1280 :=
by
  have h2 : 2 ^ 10 = 1024 := by norm_num
  have h3 : 300 / 30 = 10 := by norm_num
  have h4 : n * 1024 = 1_310_720 := by rw [←h2, ←h3, h1 300 rfl]
  sorry

end bacteria_population_l306_306327


namespace sphere_radius_in_cube_l306_306154

theorem sphere_radius_in_cube :
  (∃ (spheres : Finset (EuclideanSpace ℝ (Fin 3))),
     spheres.card = 8 ∧
     ∀ (s ∈ spheres), (∃ r : ℝ, r = 1 / 4 ∧
       ∀ (i : Fin 3), ∃ k : ℝ, k = 0 ∨ k = 1 ∧
         dist (s i) (k - r) = r)) :=
sorry

end sphere_radius_in_cube_l306_306154


namespace elective_courses_combination_l306_306015

theorem elective_courses_combination :
  let courses := 4
  let A_choices := nat.choose courses 2
  let B_choices := nat.choose courses 3
  let C_choices := nat.choose courses 3
  (A_choices * B_choices * B_choices) = 96 :=
by
  let courses := 4
  let A_choices := nat.choose courses 2
  let B_choices := nat.choose courses 3
  let C_choices := nat.choose courses 3
  have combA := computeCombinations 4 2 -- where computeCombinations n k = nat.choose n k
  have combBC := computeCombinations 4 3
  calc
    A_choices * B_choices * B_choices = combA.val * combBC.val * combBC.val : by sorry
                           ... = 96 : by sorry

end elective_courses_combination_l306_306015


namespace tangent_line_y_intercept_l306_306070
  
theorem tangent_line_y_intercept (O₁ O₂ : ℝ × ℝ) (r₁ r₂ : ℝ) (hO₁ : O₁ = (2, 0)) 
(hO₂ : O₂ = (5, 0)) (hr₁ : r₁ = 2) (hr₂ : r₂ = 1) :
  ∃ m c : ℝ, (∀ A : ℝ × ℝ, A ∈ tangent_points O₁ r₁ O₂ r₂ (line_eqn m c) → A.2 = m * A.1 + c)
  ∧ c = 2 * real.sqrt 2 :=
by sorry

end tangent_line_y_intercept_l306_306070


namespace range_of_a_l306_306553

noncomputable def f (x a : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem range_of_a (a : ℝ) :
  let x0 := -a/3 in
  (x0 > 0 ∧ ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < - (3 * (2 : ℝ)^(1 / 3) / 2) := 
by 
   sorry

end range_of_a_l306_306553


namespace cannot_determine_degree_from_char_set_l306_306771

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l306_306771


namespace example_theorem_l306_306668

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306668


namespace value_of_expression_l306_306815

theorem value_of_expression (x : ℤ) (h : x = 2017) : 
  | |x| + x - |x| | + x = 4034 :=
by
  rw h
  sorry

end value_of_expression_l306_306815


namespace complement_union_l306_306590

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306590


namespace sufficient_but_not_necessary_condition_l306_306000

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l306_306000


namespace garden_area_increase_l306_306943

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l306_306943


namespace project_completion_time_l306_306422

theorem project_completion_time :
  let A_work_rate := (1 / 30) * (2 / 3)
  let B_work_rate := (1 / 60) * (3 / 4)
  let C_work_rate := (1 / 40) * (5 / 6)
  let combined_work_rate_per_12_days := 12 * (A_work_rate + B_work_rate + C_work_rate)
  let remaining_work_after_12_days := 1 - (2 / 3)
  let additional_work_rates_over_5_days := 
        5 * A_work_rate + 
        5 * B_work_rate + 
        5 * C_work_rate
  let remaining_work_after_5_days := remaining_work_after_12_days - additional_work_rates_over_5_days
  let B_additional_time := remaining_work_after_5_days / B_work_rate
  12 + 5 + B_additional_time = 17.5 :=
sorry

end project_completion_time_l306_306422


namespace probability_even_sum_l306_306505

open Finset

def set_nums : Finset ℕ := {1, 2, 3, 4, 5}

theorem probability_even_sum :
  let pairs := set_nums.ssubsets_len 2
  let valid_pairs := pairs.filter (λ s, (s.sum (λ x, x) % 2 = 0))
  let total_combinations := pairs.card
  let favorable_combinations := valid_pairs.card
  ∃ proportion : ℚ, (proportion = favorable_combinations / total_combinations) ∧ proportion = 2 / 5 :=
by
  sorry

end probability_even_sum_l306_306505


namespace complement_of_union_is_singleton_five_l306_306645

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306645


namespace sum_of_a_b_l306_306519

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_b_l306_306519


namespace exists_triangle_cut_into_2005_congruent_l306_306845

theorem exists_triangle_cut_into_2005_congruent :
  ∃ (Δ : Type) (a b c : Δ → ℝ )
  (h₁ : a^2 + b^2 = c^2) (h₂ : a * b / 2 = 2005 / 2),
  true :=
sorry

end exists_triangle_cut_into_2005_congruent_l306_306845


namespace meters_examined_l306_306992

theorem meters_examined (x : ℝ) (h1 : 0.07 / 100 * x = 2) : x = 2857 :=
by
  -- using the given setup and simplification
  sorry

end meters_examined_l306_306992


namespace f_zero_f_odd_f_inequality_solution_l306_306533

open Real

-- Given definitions
variables {f : ℝ → ℝ}
variable (h_inc : ∀ x y, x < y → f x < f y)
variable (h_eq : ∀ x y, y * f x - x * f y = x * y * (x^2 - y^2))

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := 
sorry

-- Prove that f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := 
sorry

-- Prove the range of x satisfying the given inequality
theorem f_inequality_solution : {x : ℝ | f (x^2 + 1) + f (3 * x - 5) < 0} = {x : ℝ | -4 < x ∧ x < 1} :=
sorry

end f_zero_f_odd_f_inequality_solution_l306_306533


namespace flash_catches_up_to_arrow_l306_306497

theorem flash_catches_up_to_arrow (z k w u : ℝ) (hz : z > 1) :
  let distance := z * (k * u + w) / (z - 1)
  in distance = (z * (k * u + w)) / (z - 1) :=
by
  -- The proof goes here
  sorry

end flash_catches_up_to_arrow_l306_306497


namespace arithmetic_series_sum_l306_306440

def a := 5
def l := 20
def n := 16
def S := (n / 2) * (a + l)

theorem arithmetic_series_sum :
  S = 200 :=
by
  sorry

end arithmetic_series_sum_l306_306440


namespace area_triangle_EOF_l306_306078

-- Define the line and circle
def line : ℝ → ℝ → Prop := λ x y, x - 2 * y - 3 = 0
def circle : ℝ → ℝ → Prop := λ x y, (x - 2)^2 + (y + 3)^2 = 9

-- Define the origin
def origin := (0:ℝ, 0:ℝ)

-- Define points E and F as intersections of line and circle
def is_intersection (E : ℝ × ℝ) := line E.1 E.2 ∧ circle E.1 E.2
def is_intersection (F : ℝ × ℝ) := line F.1 F.2 ∧ circle F.1 F.2

-- Define the area formula proof
theorem area_triangle_EOF (E F : ℝ × ℝ) (hE : is_intersection E) (hF : is_intersection F) :
  let EO := euclidean_distance (E.1, E.2) origin
      O := origin
      h := line O.1 O.2
      area_EOF := (1 / 2) * (euclidean_distance E F) * (euclidean_distance O (foot O line)) in
  area_EOF = (6 * real.sqrt 5) / 5 := 
begin
  -- Using sorry for now as proof is not needed
  sorry
end

end area_triangle_EOF_l306_306078


namespace wait_time_probability_l306_306069

theorem wait_time_probability
  (P_B1_8_00 : ℚ)
  (P_B1_8_20 : ℚ)
  (P_B1_8_40 : ℚ)
  (P_B2_9_00 : ℚ)
  (P_B2_9_20 : ℚ)
  (P_B2_9_40 : ℚ)
  (h_independent : true)
  (h_employee_arrival : true)
  (h_P_B1 : P_B1_8_00 = 1/4 ∧ P_B1_8_20 = 1/2 ∧ P_B1_8_40 = 1/4)
  (h_P_B2 : P_B2_9_00 = 1/4 ∧ P_B2_9_20 = 1/2 ∧ P_B2_9_40 = 1/4) :
  (P_B1_8_00 * P_B2_9_20 + P_B1_8_00 * P_B2_9_40 = 3/16) :=
sorry

end wait_time_probability_l306_306069


namespace decagon_perimeter_l306_306109

theorem decagon_perimeter (n : ℕ) (s : ℕ) (h₁ : n = 10) (h₂ : s = 3) : n * s = 30 :=
by
  rw [h₁, h₂]
  sorry

end decagon_perimeter_l306_306109


namespace range_of_a_l306_306730

noncomputable def f (x a : ℝ) : ℝ := (x + 1) ^ 2 - a * log x

def g (x : ℝ) : ℝ := 2 * x ^ 2 + x

theorem range_of_a 
  (x1 x2 : ℝ) (a : ℝ) 
  (h₁ : x1 ∈ Set.Ioi (0 : ℝ)) (h₂ : x2 ∈ Set.Ioi (0 : ℝ)) 
  (h₃ : x1 ≠ x2)
  (h₄ : ∀ x1 x2, (f (x1 + 1) a - f (x2 + 1) a) / (x1 - x2) > 1) :
  a ∈ Set.Iic (3 : ℝ) :=
sorry

end range_of_a_l306_306730


namespace limit_computation_l306_306054

noncomputable def compute_limit : ℝ :=
  (4 : ℝ)^(5 * (0 : ℝ)) - (9 : ℝ)^(-2 * (0 : ℝ)) / (sin (0 : ℝ) - tan ((0 : ℝ)^3))

theorem limit_computation :
  ∃ l : ℝ, (∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < abs x ∧ abs x < δ → abs ((4^(5*x) - 9^(-2*x)) / (sin x - tan(x^3)) - l) < ε)) ∧ l = ln(1024 * 81)
:= 
sorry

end limit_computation_l306_306054


namespace vacation_expenses_split_l306_306988

theorem vacation_expenses_split
  (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ)
  (hA : A = 180)
  (hB : B = 240)
  (hC : C = 120)
  (ha : a = 0)
  (hb : b = 0)
  : a - b = 0 := 
by
  sorry

end vacation_expenses_split_l306_306988


namespace count_integer_radii_l306_306446

theorem count_integer_radii (r : ℕ) (h : r < 150) :
  (∃ n : ℕ, n = 11 ∧ (∀ r, 0 < r ∧ r < 150 → (150 % r = 0)) ∧ (r ≠ 150)) := sorry

end count_integer_radii_l306_306446


namespace card_at_position_53_l306_306465

theorem card_at_position_53 (seq : ℕ → string)
  (h_cycle : ∀ n, seq (n + 13) = seq n)
  (h_initial : seq 0 = "A") :
  seq 52 = "A" :=
sorry

end card_at_position_53_l306_306465


namespace regular_polygon_sides_l306_306477

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end regular_polygon_sides_l306_306477


namespace total_snakes_owned_l306_306747

theorem total_snakes_owned 
  (total_people : ℕ)
  (only_dogs only_cats only_birds only_snakes : ℕ)
  (cats_and_dogs birds_and_dogs birds_and_cats snakes_and_dogs snakes_and_cats snakes_and_birds : ℕ)
  (cats_dogs_snakes cats_dogs_birds cats_birds_snakes dogs_birds_snakes all_four_pets : ℕ)
  (h1 : total_people = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four_pets = 10) : 
  7 + 3 + 4 + 2 + 5 + 6 + 9 + 10 = 46 := 
sorry

end total_snakes_owned_l306_306747


namespace complement_union_l306_306675

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306675


namespace degree_not_determined_by_A_P_l306_306795

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l306_306795


namespace functions_are_equal_l306_306916

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- Statement to be proven
theorem functions_are_equal : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_equal_l306_306916


namespace danny_marks_in_english_l306_306135

theorem danny_marks_in_english
  (M : ℕ := 65)    -- Mathematics marks
  (Ph : ℕ := 82)   -- Physics marks
  (C : ℕ := 67)    -- Chemistry marks
  (B : ℕ := 75)    -- Biology marks
  (avg : ℕ := 73)  -- Average marks
  (subjects : ℕ := 5)  -- Number of subjects 
  (total_marks : ℕ := avg * subjects := 365)
  (total_known_marks : ℕ := M + Ph + C + B := 289) :
  total_marks - total_known_marks = 76 :=
by
  sorry

end danny_marks_in_english_l306_306135


namespace range_of_function_l306_306141

theorem range_of_function : 
  ∀ y : ℝ, 
  (∃ x : ℝ, y = x^2 + 1) ↔ (y ≥ 1) :=
by
  sorry

end range_of_function_l306_306141


namespace train_length_l306_306090

theorem train_length (t1 t2 P : ℝ) (L : ℝ) (h1 : t1 = 18) (h2 : t2 = 51) (h3 : P = 550) :
  ∃ (V : ℝ), L = 300 ∧ L = V * t1 ∧ L + P = V * t2 :=
by
  use L / t1  -- Define V as L / t1
  split
  . sorry   -- L = 300
  . exact eq.symm (div_mul_cancel L (ne_of_gt (show t1 > 0, by simp [h1])))  -- L = V * t1
  . sorry   -- L + P = V * t2

end train_length_l306_306090


namespace distance_to_lightning_l306_306302

def speed_of_sound := 1100   -- (feet per second)
def time_elapsed := 8        -- (seconds)
def feet_per_mile := 5280    -- (feet)

def distance_in_feet := speed_of_sound * time_elapsed
def distance_in_miles := distance_in_feet / feet_per_mile

theorem distance_to_lightning : distance_in_miles = 1.75 :=
by
  have : distance_in_feet = 1100 * 8 := rfl
  have : distance_in_feet = 8800 := by norm_num
  have : distance_in_miles = 8800 / 5280 := rfl
  have : distance_in_miles = 1.66666667 := by norm_num
  have : distance_in_miles = 1.75 := sorry
  exact this

end distance_to_lightning_l306_306302


namespace inscribed_circle_radius_l306_306371

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaUsingHeron (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  let K := areaUsingHeron a b c
  K / s

theorem inscribed_circle_radius : inscribedCircleRadius 26 18 20 = Real.sqrt 31 :=
  sorry

end inscribed_circle_radius_l306_306371


namespace complement_union_l306_306683

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306683


namespace right_triangle_curvilinear_area_ratio_l306_306252

noncomputable def ratio_of_curvilinear_triangle_areas (α : ℝ) : ℝ :=
  (Real.tan α - α) / (Real.cot α - (Real.pi / 2) + α)

theorem right_triangle_curvilinear_area_ratio (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) :
    ratio_of_curvilinear_triangle_areas α =
    (Real.tan α - α) / (Real.cot α - (Real.pi / 2) + α) := sorry

end right_triangle_curvilinear_area_ratio_l306_306252


namespace knights_hamiltonian_cycle_l306_306428

open Classical

noncomputable theory

-- Definition for a graph with knights and its properties
variables {V : Type} [Fintype V]
variables (G : SimpleGraph V)

-- Number of knights (vertices) in the graph
variable [Fintype V]

def less_than_half_degrees : Prop :=
  ∀ (v : V), G.degree v ≥ ⌈(Fintype.card V - 1) / 2⌉

theorem knights_hamiltonian_cycle (n : ℕ) (hn : n ≥ 3) (hG : less_than_half_degrees G) : 
  ∃ (cycle : List V), (cycle.Nodup) ∧ (cycle.length = Fintype.card V) ∧ (G.isHamiltonianCycle cycle) :=
sorry

end knights_hamiltonian_cycle_l306_306428


namespace minimum_positive_sum_example_l306_306150

noncomputable def min_positive_sum (n : ℕ) (a : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n, ∑ j in Finset.Icc (i + 1) (n - 1), a i * a j

theorem minimum_positive_sum_example :
  ∀ (a : Fin 150 → ℝ), (∀ i, a i = 1 ∨ a i = -1) →
  min_positive_sum 150 a = 53 :=
by
  intro a ha
  have S : ℝ := min_positive_sum 150 a
  have h_sum_eq_16 : (∑ i in Finset.range 150, a i) = 16 := sorry
  have h_sq_sum : (∑ i in Finset.range 150, (a i)^2) = 150 := sorry
  -- Goal is to prove that 2S = 256 - 150
  have h_2S_eq : 2 * S = (∑ i in Finset.range 150, a i)^2 - 150 := by
    calc 2 * S = (∑ i in Finset.range 150, ∑ j in Finset.Icc (i + 1) (150 - 1), a i * a j) : sorry
    ... = (∑ i in Finset.range 150, a i)^2 - (∑ i in Finset.range 150, (a i)^2) : sorry
    ... = 256 - 150 : by rw [h_sum_eq_16, h_sq_sum]
  have S_eq_53 : S = (256 - 150) / 2 := by
    rw [h_2S_eq]
    norm_num
  rw [S_eq_53]
  norm_num
  done

end minimum_positive_sum_example_l306_306150


namespace length_of_AB_l306_306180

def point := (ℤ × ℤ × ℤ)

def A : point := (1, 3, -5)
def B : point := (4, -2, 3)

noncomputable def distance (p₁ p₂ : point) : ℝ :=
  real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2 + (p₂.3 - p₁.3)^2)

theorem length_of_AB : distance A B = 7 * real.sqrt 2 := 
by sorry

end length_of_AB_l306_306180


namespace xyz_values_l306_306816

theorem xyz_values (x y z : ℝ)
  (h1 : x * y - 5 * y = 20)
  (h2 : y * z - 5 * z = 20)
  (h3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := 
by sorry

end xyz_values_l306_306816


namespace probability_of_stack_height_48_l306_306018

open Nat

theorem probability_of_stack_height_48 :
  let crates := 12
  let height := 48
  let dims := {3, 4, 6}
  let ways_to_stack := 37522
  let total_ways := 3 ^ 12
  let gcd_of_ways := gcd ways_to_stack total_ways
  gcd_of_ways = 1 →
  (ways_to_stack, total_ways).1 = 37522 :=
by sorry

end probability_of_stack_height_48_l306_306018


namespace incorrect_propositions_l306_306990

open Plane
open Point
open Line

-- Definitions of the propositions
def proposition1 (α β : Plane) : Prop :=
  (∃ lines : set Line, ∀ l ∈ lines, l ∷ α ∧ l ∸ β) → α ∸ β

def proposition2 (α β γ : Plane) (l : Line) : Prop :=
  (l ∷ α ∧ l ∷ β ∧ l ∷ γ) → α ∸ β

def proposition3 (α : Plane) (p q : Point) : Prop :=
  (p ∉ α ∧ q ∉ α) → ∃ β : Plane, β ∸ α ∧ (p ∈ β) ∧ (q ∈ β)

def proposition4 (α β γ : Plane) : Prop :=
  (α ∸ β ∧ β ∸ γ) → α ∸ γ

-- Main theorem stating the correctness of each proposition
theorem incorrect_propositions : 
  (¬ (∀ α β : Plane, proposition1 α β) ∧
   ¬ (∀ α β γ : Plane, ∀ l : Line, proposition2 α β γ l) ∧
   ¬ (∀ α : Plane, ∀ p q : Point, proposition3 α p q) ∧
   (∀ α β γ : Plane, proposition4 α β γ)) :=
by sorry

-- Lean will require the base axioms and definitions
-- Each additional proposition requires thorough definitions 
-- and metamathematical assumptions about space, points, lines, and planes.

end incorrect_propositions_l306_306990


namespace stones_in_10th_pattern_l306_306436

def stones_in_nth_pattern (n : ℕ) : ℕ :=
n * (3 * n - 1) / 2 + 1

theorem stones_in_10th_pattern : stones_in_nth_pattern 10 = 145 :=
by
  sorry

end stones_in_10th_pattern_l306_306436


namespace females_in_town_l306_306931

theorem females_in_town (total_population males_in_town females_with_glasses : ℕ) 
    (h1 : total_population = 5000) 
    (h2 : males_in_town = 2000) 
    (h3 : females_with_glasses = 900) 
    (h4 : 0.30 * (total_population - males_in_town) = females_with_glasses) : 
    total_population - males_in_town = 3000 :=
by sorry

end females_in_town_l306_306931


namespace problem_part1_problem_part2_l306_306512

theorem problem_part1 (a : ℝ) (P : ∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - a ≥ 0) : a ≤ 1 :=
sorry

theorem problem_part2 (a : ℝ) 
  (P_or_Q : (∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - a ≥ 0) ∨ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0))
  (not_P_and_Q : ¬ ((∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0))) :
  a > 1 ∨ (-2 < a ∧ a < 1) :=
sorry

end problem_part1_problem_part2_l306_306512


namespace possible_value_of_n_l306_306329

open Nat

def coefficient_is_rational (n r : ℕ) : Prop :=
  (n - r) % 2 = 0 ∧ r % 3 = 0

theorem possible_value_of_n :
  ∃ n : ℕ, n > 0 ∧ (∀ r : ℕ, r ≤ n → coefficient_is_rational n r) ↔ n = 9 :=
sorry

end possible_value_of_n_l306_306329


namespace midpoint_equidistant_l306_306337

-- Definitions based on the conditions
variables {A B C B1 C1 B2 C2 : Type}
-- Assuming appropriate types and operations for points and lines
-- for simplicity we use dummy types above, these should ideally be points and line structures from geometry

def incircle_touches_at (A B C : Type) (B1 C1 : Type) : Prop := -- placeholder for actual geometric definition
sorry

def excircle_touches_at_extensions (A B C : Type) (B2 C2 : Type) : Prop := -- placeholder for actual geometric definition
sorry

def midpoint (B C : Type) : Type := -- placeholder for actual midpoint definition
sorry

def equidistant_from_lines (M : Type) (line1 line2 : Type) : Prop := -- placeholder for equidistant definition
sorry

-- Theorem statement
theorem midpoint_equidistant
  (A B C M B1 C1 B2 C2 : Type)
  (h1 : incircle_touches_at A B C B1 C1)
  (h2 : excircle_touches_at_extensions A B C B2 C2)
  (hM : M = midpoint B C)
  : equidistant_from_lines M (line B1 C1) (line B2 C2)
:= sorry

end midpoint_equidistant_l306_306337


namespace tan_alpha_eq_l306_306250

theorem tan_alpha_eq :
  ∀ (A B C : Type) [AddGroup A] [AddGroup B] [AddGroup C]
  (a : ℝ) (n : ℕ) (h : ℝ) (α : ℝ),
  (a > 0) →
  (n > 0) → (Nat.Odd n) →
  (tan α = (4 * n * h) / ((n^2 - 1) * a)) :=
by
  intros,
  sorry

end tan_alpha_eq_l306_306250


namespace complement_of_union_is_singleton_five_l306_306646

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306646


namespace problem_statement_l306_306709

theorem problem_statement (x : ℝ) 
  (h : x + sqrt (x^2 - 4) + (1 / (x - sqrt (x^2 - 4))) = 10) : 
  x^2 + sqrt (x^4 - 4) + (1 / (x^2 + sqrt (x^4 - 4))) = 841 / 100 := 
sorry

end problem_statement_l306_306709


namespace length_BC_125_l306_306246

-- Given conditions:
structure Triangle :=
(AB : ℝ) (AC : ℝ) (BC : ℝ)
(BX : ℝ) (CX : ℝ)
(center_A_in_circle : BX + CX = BC)
(radius_AB : AB = 75)
(radius_AC : AC = 100)
(BX_CX_integers : BX ∈ Int ∧ CX ∈ Int)

-- Prove that the length of BC is 125
theorem length_BC_125 (T : Triangle) : T.BC = 125 :=
by
  sorry

end length_BC_125_l306_306246


namespace find_x_plus_z_l306_306012

variable (u v w x y z : ℂ)

theorem find_x_plus_z
  (h1: v = 5)
  (h2: y = -u - w)
  (h3: u + (5 : ℂ) * complex.I + w + x * complex.I + y + z * complex.I = 4 * complex.I) : 
  x + z = -1 := 
by
  sorry

end find_x_plus_z_l306_306012


namespace number_of_n_with_odd_tens_digit_in_square_l306_306170

def ends_in_3_or_7 (n : ℕ) : Prop :=
  n % 10 = 3 ∨ n % 10 = 7

def tens_digit_odd (n : ℕ) : Prop :=
  ((n * n / 10) % 10) % 2 = 1

theorem number_of_n_with_odd_tens_digit_in_square :
  ∀ n ∈ {n : ℕ | n ≤ 50 ∧ ends_in_3_or_7 n}, ¬tens_digit_odd n :=
by 
  sorry

end number_of_n_with_odd_tens_digit_in_square_l306_306170


namespace find_balcony_seat_cost_l306_306979

def orchestra_seat_price := 12
def total_tickets_sold := 355
def total_revenue := 3320
def more_balcony_than_orchestra := 115
def orchestra_tickets_sold := (λ O : ℕ, 2 * O + more_balcony_than_orchestra = total_tickets_sold)
def total_revenue_from_balcony_tickets := (λ O : ℕ, total_revenue - (O * orchestra_seat_price))
def balcony_tickets_sold := (λ O : ℕ, O + more_balcony_than_orchestra)
def cost_of_balcony_seat := (λ B O : ℕ, total_revenue_from_balcony_tickets O = balcony_tickets_sold O * B)

theorem find_balcony_seat_cost (O B : ℕ) :
  orchestra_tickets_sold O →
  cost_of_balcony_seat 8 O :=
by
  sorry

end find_balcony_seat_cost_l306_306979


namespace compute_expression_l306_306449

theorem compute_expression : (3 + 5) ^ 2 + (3 ^ 2 + 5 ^ 2) = 98 := by
  sorry

end compute_expression_l306_306449


namespace min_value_of_abs_diff_l306_306548
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem min_value_of_abs_diff (x1 x2 x : ℝ) (h1 : f x1 ≤ f x) (h2: f x ≤ f x2) : |x1 - x2| = π := by
  sorry

end min_value_of_abs_diff_l306_306548


namespace complement_union_eq_singleton_five_l306_306576

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306576


namespace equation_solutions_exist_l306_306306

theorem equation_solutions_exist (d x y : ℤ) (hx : Odd x) (hy : Odd y)
  (hxy : x^2 - d * y^2 = -4) : ∃ X Y : ℕ, X^2 - d * Y^2 = -1 :=
by
  sorry  -- Proof is omitted as per the instructions

end equation_solutions_exist_l306_306306


namespace circle_area_l306_306857

theorem circle_area (r : ℝ) (h : 5 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 5 := 
by
  sorry -- Proof is not required, placeholder for the actual proof

end circle_area_l306_306857


namespace ellipse_parameters_sum_l306_306142

def ellipse_sum (h k a b : ℝ) : ℝ :=
  h + k + a + b

theorem ellipse_parameters_sum :
  let h := 5
  let k := -3
  let a := 7
  let b := 4
  ellipse_sum h k a b = 13 := by
  sorry

end ellipse_parameters_sum_l306_306142


namespace find_a_l306_306722

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l306_306722


namespace complement_union_M_N_l306_306626

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306626


namespace garden_area_increase_l306_306935

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l306_306935


namespace example_theorem_l306_306672

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306672


namespace area_increase_correct_l306_306941

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l306_306941


namespace question1_question2_l306_306450

theorem question1 :
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + 1.5^(-2) = 1/2 :=
by
  sorry

theorem question2 :
  log 3 (real.sqrt 27 / 3) + log 10 25 + log 10 4 + 7^(log 7 2) = 15/4 :=
by
  sorry

end question1_question2_l306_306450


namespace complement_union_eq_l306_306582

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306582


namespace probability_heads_at_most_two_l306_306401

noncomputable def coin_flip_probability : ℚ :=
  let total_outcomes := 2 ^ 8 in
  let favorable_outcomes := Nat.choose 8 0 + Nat.choose 8 1 + Nat.choose 8 2 in
  favorable_outcomes / total_outcomes

theorem probability_heads_at_most_two :
  coin_flip_probability = 37 / 256 :=
by
  sorry

end probability_heads_at_most_two_l306_306401


namespace min_m_plus_n_l306_306915

theorem min_m_plus_n (m n : ℕ) (h₁ : m > n) (h₂ : 4^m + 4^n % 100 = 0) : m + n = 7 :=
sorry

end min_m_plus_n_l306_306915


namespace number_of_ways_to_test_l306_306498

/-- For a certain product, there are 5 different genuine items and 4 different defective items tested one by one until all defective items are identified. If all defective items are exactly discovered after five tests, then there are 480 ways to conduct such a test. -/
theorem number_of_ways_to_test 
  (genuine : Finset (Fin 5)) 
  (defective : Finset (Fin 4)) 
  (tests : List (Fin 9)) :
  tests.length = 5 ∧
  (∀ t ∈ tests.take 4, t ∈ genuine ∨ t ∈ defective) ∧
  (∃ t ∈ tests.drop 4, t ∈ defective) →
  tests.nodup →
  (count_defective tests = 4) →
  (count_genuine tests = 1) →
  ∃ n: ℕ, n = 480 := sorry

end number_of_ways_to_test_l306_306498


namespace complement_union_l306_306597

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306597


namespace boxes_needed_l306_306313

theorem boxes_needed (total_oranges : ℕ) (oranges_per_box : ℕ) (h : total_oranges = 56) (k : oranges_per_box = 7) : total_oranges / oranges_per_box = 8 :=
by {
  rw [h, k], -- Using the given conditions
  norm_num,
}

end boxes_needed_l306_306313


namespace complement_union_l306_306685

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306685


namespace household_item_pricing_l306_306068

theorem household_item_pricing :
  ∀ (x : ℕ), 8 ≤ x ∧ x ≤ 15 →
  (x - 8) * (-5 * x + 150) = 425 → x = 13 :=
by {
  sorry,
}

end household_item_pricing_l306_306068


namespace largest_divisor_n4_minus_n2_l306_306467

theorem largest_divisor_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_n4_minus_n2_l306_306467


namespace units_digit_of_24_pow_4_add_42_pow_4_l306_306912

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end units_digit_of_24_pow_4_add_42_pow_4_l306_306912


namespace students_passed_both_l306_306253

noncomputable def F_H : ℝ := 32
noncomputable def F_E : ℝ := 56
noncomputable def F_HE : ℝ := 12
noncomputable def total_percentage : ℝ := 100

theorem students_passed_both : (total_percentage - (F_H + F_E - F_HE)) = 24 := by
  sorry

end students_passed_both_l306_306253


namespace area_of_triangle_XYZ_is_24_l306_306766

variables {α : Type*} [LinearOrderedField α]

noncomputable def area_of_triangle_XYZ (XY YZ YM YW tan_ZYW tan_MYW tan_XYW cot_MYW cot_ZYW cot_ZYM : α) 
  (h1 : XY = YZ)
  (h2 : YM^2 + (XY / 2)^2 = YZ^2)
  (h3 : YW = 12)
  (geom_progression : (tan_ZYW * tan_MYW * tan_XYW = tan_MYW^3))
  (arith_progression : (cot_MYW - cot_ZYW) + (cot_ZYW - cot_ZYM) = cot_MYW - cot_ZYM) : α := 
  if h : XY = 0 then 0 else
    let XM := sqrt (YZ^2 - YM^2) in
    let area := (XY * YM) / 2 in
    area

-- Main theorem stating the problem
theorem area_of_triangle_XYZ_is_24 
  (XY YZ YM YW tan_ZYW tan_MYW tan_XYW cot_MYW cot_ZYW cot_ZYM : α) 
  (h1 : XY = YZ)
  (h2 : YM^2 + (XY / 2)^2 = YZ^2)
  (h3 : YW = 12)
  (geom_progression : (tan_ZYW * tan_MYW * tan_XYW = tan_MYW^3))
  (arith_progression : (cot_MYW - cot_ZYW) + (cot_ZYW - cot_ZYM) = cot_MYW - cot_ZYM) :
  area_of_triangle_XYZ XY YZ YM YW tan_ZYW tan_MYW tan_XYW cot_MYW cot_ZYW cot_ZYM h1 h2 h3 geom_progression arith_progression = 24 := 
sorry

end area_of_triangle_XYZ_is_24_l306_306766


namespace exists_triangle_cut_into_2005_congruent_l306_306844

theorem exists_triangle_cut_into_2005_congruent :
  ∃ (Δ : Type) (a b c : Δ → ℝ )
  (h₁ : a^2 + b^2 = c^2) (h₂ : a * b / 2 = 2005 / 2),
  true :=
sorry

end exists_triangle_cut_into_2005_congruent_l306_306844


namespace sum_of_three_numbers_from_1_to_100_is_odd_l306_306919

open Probability

noncomputable def probability_sum_odd : ℚ :=
  let numbers := finset.range 101
  let balls := numbers.filter (λ n, n > 0)
  let odd_count := (balls.filter (λ n, n % 2 = 1)).card
  let even_count := (balls.filter (λ n, n % 2 = 0)).card
  let odd_prob := (odd_count : ℚ) / balls.card
  let even_prob := (even_count : ℚ) / balls.card
  let odd_sum_prob := (even_prob * even_prob * odd_prob) + (odd_prob * odd_prob * odd_prob)
  odd_sum_prob

theorem sum_of_three_numbers_from_1_to_100_is_odd : probability_sum_odd = 1 / 2 :=
sorry

end sum_of_three_numbers_from_1_to_100_is_odd_l306_306919


namespace true_proposition_l306_306529

open classical

variables (ϕ : ℝ) (x : ℝ)

def p := ϕ = π / 2 → ¬(∃ k : ℤ, ϕ ≠ k * π + π / 2)
def q := ∀ x ∈ Ioo 0 (π / 2), sin x = 1 / 2

theorem true_proposition (hp : p ϕ) (hq : q) : p ϕ ∧ q :=
by sorry

end true_proposition_l306_306529


namespace garden_area_increase_l306_306949

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l306_306949


namespace cos_value_l306_306190

theorem cos_value (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (2 * π / 3 - α) = 1 / 3 :=
by
  sorry

end cos_value_l306_306190


namespace complement_union_l306_306591

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306591


namespace subset_contains_power_of_two_or_sum_power_of_two_l306_306426

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def contains_sum_power_of_two (A : set ℕ) : Prop :=
  ∃ x y ∈ A, x ≠ y ∧ is_power_of_two (x + y)

theorem subset_contains_power_of_two_or_sum_power_of_two (A : set ℕ) (hA : A ⊆ (set.Icc 0 1997)) (h_size : (A.card : ℕ) > 1000) :
  (∃ x ∈ A, is_power_of_two x) ∨ contains_sum_power_of_two A :=
sorry

end subset_contains_power_of_two_or_sum_power_of_two_l306_306426


namespace triangle_is_isosceles_triangle_area_l306_306510

-- Part 1: Proving the triangle is isosceles
theorem triangle_is_isosceles
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) -- Positive side lengths
  (h_triangle : A + B + C = π) -- Sum of angles in a triangle
  (h_sin_condition : a * Real.sin A = b * Real.sin B) :
  a = b :=
by
  sorry

-- Part 2: Finding the area of the triangle
theorem triangle_area
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a + b = a * b)
  (h2 : c = 2)
  (h3 : C = π / 3)
  (ha : a > 0) (hb : b > 0) -- Positive side lengths
  (h_triangle : A + B + C = π) -- Sum of angles in a triangle
  :
  (1 / 2) * a * b * Real.sin C = real.sqrt 3 :=
by
  sorry

end triangle_is_isosceles_triangle_area_l306_306510


namespace right_triangle_angle_bisector_tan_l306_306092

theorem right_triangle_angle_bisector_tan
  (a b c : ℝ)
  (ha : a = 5)
  (hb : b = 12)
  (hc : c = 13)
  (h_perimeter : a + b + c = 30)
  (h_perimeter_bisect : ∀ x y, x + y = 15)
  (h_area : (1 / 2) * a * b = 30)
  (h_area_bisect : ∀ x y, (1 / 2) * x * y = 15) :
  ∃ φ : ℝ, φ < π / 2 ∧ tan φ = ?  := -- Here you would place the correct tan φ value based on calculated steps

sorry

end right_triangle_angle_bisector_tan_l306_306092


namespace possible_trajectory_l306_306259

-- Given problem conditions
variable (A A1 B B1 C C1 D D1 E E1 P: Type)
variable [Group A]
variable [Group A1]
variable [Group B]
variable [Group B1]
variable [Group C]
variable [Group C1]
variable [Group D]
variable [Group D1]
variable [Group E]
variable [Group E1]
variable [Group P]
variable [MetricSpace A]
variable [MetricSpace A1]
variable [MetricSpace B]
variable [MetricSpace B1]
variable [MetricSpace C]
variable [MetricSpace C1]
variable [MetricSpace D]
variable [MetricSpace D1]
variable [MetricSpace E]
variable [MetricSpace E1]
variable [MetricSpace P]

-- Midpoint definitions
def is_midpoint (x y z: Type) [MetricSpace x] [MetricSpace y] [MetricSpace z]: Prop :=
  dist x z = dist y z

axiom AA1_midpoint: is_midpoint A A1 E
axiom CC1_midpoint: is_midpoint C C1 E1

-- Line parallel and perpendicular conditions
axiom EE1_parallel_AC: parallel EE1 AC
axiom EE1_perpendicular_plane_DBB1D1: perp_to_plane EE1 (plane D B B1 D1)

-- Movement of P and angle conditions
noncomputable def angle_with_line (x y z: Type) [MetricSpace x] [MetricSpace y] [MetricSpace z]: Angle := sorry

axiom equal_angle_EP_AC: ∀ lim, (angle_with_line E P AC) = lim ∧ (angle_with_line E P EE1) = lim

-- Prove the possible trajectory of point P
theorem possible_trajectory : 
  ∀ lim, is_circle (pos_trajectory E P (angle_with_line E P EE1) (plane D B B1 D1)) :=
by sorry

end possible_trajectory_l306_306259


namespace intersection_area_ratio_p_plus_q_correct_l306_306895

-- Define points and line segments with the given lengths
def A : Point := sorry
def B : Point := sorry
def E : Point := sorry
def F : Point := sorry

axiom AB_length : dist A B = 8
axiom BE_length : dist B E = 15
axiom EA_length : dist E A = 20
axiom AF_length : dist A F = 15
axiom FB_length : dist F B = 20
axiom congruent_triangles : triangle A B E ≅ triangle B A F

-- Define the intersection area with relation to rational numbers
def intersection_area : ℚ := 21.24

noncomputable def p : ℚ := 21.24
def q : ℚ := 1

theorem intersection_area_ratio : intersection_area = (p / q) := sorry
theorem p_plus_q_correct : p + q = 22.24 := sorry

end intersection_area_ratio_p_plus_q_correct_l306_306895


namespace correct_comprehensive_survey_l306_306381

-- Definitions for the types of surveys.
inductive Survey
| A : Survey
| B : Survey
| C : Survey
| D : Survey

-- Function that identifies the survey suitable for a comprehensive survey.
def is_comprehensive_survey (s : Survey) : Prop :=
  match s with
  | Survey.A => False            -- A is for sampling, not comprehensive
  | Survey.B => False            -- B is for sampling, not comprehensive
  | Survey.C => False            -- C is for sampling, not comprehensive
  | Survey.D => True             -- D is suitable for comprehensive survey

-- The theorem to prove that D is the correct answer.
theorem correct_comprehensive_survey : is_comprehensive_survey Survey.D = True := by
  sorry

end correct_comprehensive_survey_l306_306381


namespace similar_triangles_length_l306_306398

variable (PQR STU : Type) [triangle PQR] [triangle STU]
variables (PQ QR ST TU : ℝ)

theorem similar_triangles_length
  (h_sim : similar PQR STU)
  (h_PQ : PQ = 12)
  (h_QR : QR = 10)
  (h_ST : ST = 18)
  : TU = 15 := by
  sorry

end similar_triangles_length_l306_306398


namespace max_permutations_l306_306254

def permutation (n : ℕ) : Type := list (fin n)

noncomputable def swap_non_adj {n : ℕ} (p : permutation n) (i j : ℕ) : permutation n :=
if abs (i - j) = 1 then p else list.update_nth (list.update_nth p i (p.nth_le j (nat.lt_trans i j))) j (p.nth_le i (nat.lt_trans j i))

def S (p : permutation 100) : string :=
(list.range 99).map (λ i, if p.nth_le i sorry < p.nth_le (i + 1) sorry then '<' else '>').as_string

theorem max_permutations (G : Type) [fintype G] (exists_S_index : permutation 100 → string) :
  (∀ p1 p2 : permutation 100, (∃ u v : permutation 100, swap_non_adj u = v) →
    (S p1 = S p2 ↔ ( ∃ x y : G, x = y))) →
  ∃ n : ℕ, ∀ grid : vector (permutation 100) n, (∀ i j, i ≠ j → grid.nth i ≠ grid.nth j) → 
    ∃ k, k = 2 ^ 99 := sorry

end max_permutations_l306_306254


namespace determine_value_of_product_l306_306287

theorem determine_value_of_product (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := 
by 
  sorry

end determine_value_of_product_l306_306287


namespace q_minus_p_897_l306_306819

def smallest_three_digit_integer_congruent_7_mod_13 := ∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7
def smallest_four_digit_integer_congruent_7_mod_13 := ∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7

theorem q_minus_p_897 : 
  (∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7) → 
  (∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7) → 
  ∀ p q : ℕ, 
    (p = 8*13+7) → 
    (q = 77*13+7) → 
    q - p = 897 :=
by
  intros h1 h2 p q hp hq
  sorry

end q_minus_p_897_l306_306819


namespace combined_weight_is_18442_l306_306271

noncomputable def combined_weight_proof : ℝ :=
  let elephant_weight_tons := 3
  let donkey_weight_percentage := 0.1
  let giraffe_weight_tons := 1.5
  let hippopotamus_weight_kg := 4000
  let elephant_food_oz := 16
  let donkey_food_lbs := 5
  let giraffe_food_kg := 3
  let hippopotamus_food_g := 5000

  let ton_to_pounds := 2000
  let kg_to_pounds := 2.20462
  let oz_to_pounds := 1 / 16
  let g_to_pounds := 0.00220462

  let elephant_weight_pounds := elephant_weight_tons * ton_to_pounds
  let donkey_weight_pounds := (1 - donkey_weight_percentage) * elephant_weight_pounds
  let giraffe_weight_pounds := giraffe_weight_tons * ton_to_pounds
  let hippopotamus_weight_pounds := hippopotamus_weight_kg * kg_to_pounds

  let elephant_food_pounds := elephant_food_oz * oz_to_pounds
  let giraffe_food_pounds := giraffe_food_kg * kg_to_pounds
  let hippopotamus_food_pounds := hippopotamus_food_g * g_to_pounds

  elephant_weight_pounds + donkey_weight_pounds + giraffe_weight_pounds + hippopotamus_weight_pounds +
  elephant_food_pounds + donkey_food_lbs + giraffe_food_pounds + hippopotamus_food_pounds

theorem combined_weight_is_18442 : combined_weight_proof = 18442 := by
  sorry

end combined_weight_is_18442_l306_306271


namespace maximum_angle_at_vertex_l306_306834

structure Sphere := 
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

structure Cone := 
  (vertex : ℝ × ℝ × ℝ)
  (radius : ℝ)
  (slant_height : ℝ)

def spheres_touch_externally (s1 s2 : Sphere) : Prop :=
  dist s1.center s2.center = s1.radius + s2.radius

def cone_touches_spheres_and_table (c : Cone) (s1 s2 : Sphere) (table_height : ℝ) : Prop :=
  c.vertex.2 = table_height ∧
  dist c.vertex s1.center = c.radius + s1.radius ∧ 
  dist c.vertex s2.center = c.radius + s2.radius

def vertex_on_segment (c : Cone) (s1 s2 : Sphere) : Prop :=
  ∃ A : ℝ × ℝ × ℝ, A = (s1.center.1 + s2.center.1) / 2, 
  dist c.vertex A = dist s1.center s2.center / 2

def rays_form_equal_angles (c : Cone) (s1 s2 : Sphere) : Prop :=
  let φ := angle_between_rays c.vertex s1.center s2.center
  angle_between_rays c.vertex s1.center (0, 0, 0) = φ ∧
  angle_between_rays c.vertex s2.center (0, 0, 0) = φ

theorem maximum_angle_at_vertex (s1 s2 : Sphere) (c : Cone) (table_height : ℝ)
  (h1: spheres_touch_externally s1 s2)
  (h2: cone_touches_spheres_and_table c s1 s2 table_height)
  (h3: vertex_on_segment c s1 s2)
  (h4: rays_form_equal_angles c s1 s2) :
  cone_apex_angle c = 2 * real.arctan (1 / 2) := 
sorry

end maximum_angle_at_vertex_l306_306834


namespace complement_union_l306_306595

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306595


namespace complement_union_of_M_and_N_l306_306693

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306693


namespace ratio_problem_l306_306715

variable (a b c d : ℚ)

theorem ratio_problem
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end ratio_problem_l306_306715


namespace complement_union_eq_l306_306586

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306586


namespace count_valid_numbers_l306_306707

def first_digit (n : ℕ) : ℕ := n / 100
def second_digit (n : ℕ) : ℕ := (n / 10) % 10
def third_digit (n : ℕ) : ℕ := n % 10
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def valid_number (n : ℕ) : Prop :=
  is_three_digit n ∧
  first_digit n = third_digit n ∧
  (2 * first_digit n + second_digit n) < 15 ∧
  (2 * first_digit n + second_digit n) % 3 ≠ 0

theorem count_valid_numbers : 
  ∃ N : ℕ, (N = (set_of valid_number).card) ∧ N = 52 :=
sorry

end count_valid_numbers_l306_306707


namespace car_rental_cost_per_mile_l306_306064

theorem car_rental_cost_per_mile:
  ∀ (daily_rental_cost total_budget distance : ℝ),
  daily_rental_cost = 30 →
  total_budget = 75 →
  distance = 250 →
  (total_budget - daily_rental_cost) / distance = 0.18 := by
  intros daily_rental_cost total_budget distance h1 h2 h3
  calc
    (total_budget - daily_rental_cost) / distance
        = (75 - 30) / 250 : by rw [h1, h2, h3]
    ... = 45 / 250 : by norm_num
    ... = 0.18 : by norm_num

end car_rental_cost_per_mile_l306_306064


namespace cube_edges_l306_306918

-- Definitions representing the conditions
def is_cube (shape : Type) : Prop := 
  shape = Cube

-- The statement of the problem in Lean 4
theorem cube_edges (shape : Type) (h : is_cube shape) : num_edges shape = 12 :=
sorry

end cube_edges_l306_306918


namespace lateral_surface_area_of_cone_l306_306735

theorem lateral_surface_area_of_cone (m r : ℝ) (hm : 0 < m) (hr : 0 < r) : 
  let area := π * r * m
  in area = π * r * m :=
by
  sorry

end lateral_surface_area_of_cone_l306_306735


namespace complement_union_l306_306606

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306606


namespace find_set_A_l306_306216

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4, 5})
variable (h1 : (U \ A) ∩ B = {0, 4})
variable (h2 : (U \ A) ∩ (U \ B) = {3, 5})

theorem find_set_A :
  A = {1, 2} :=
by
  sorry

end find_set_A_l306_306216


namespace exists_triangle_cut_into_2005_congruent_l306_306842

theorem exists_triangle_cut_into_2005_congruent (n : ℕ) (hn : n = 2005) : 
  ∃ (Δ : Type) [triangle Δ], ∃ (cut : Δ → list Δ), list.all (congruent Δ) (cut Δ) ∧ list.length (cut Δ) = n := 
sorry

end exists_triangle_cut_into_2005_congruent_l306_306842


namespace money_difference_l306_306977

def share_ratio (w x y z : ℝ) (k : ℝ) : Prop :=
  w = k ∧ x = 6 * k ∧ y = 2 * k ∧ z = 4 * k

theorem money_difference (k : ℝ) (h : k = 375) : 
  ∀ w x y z : ℝ, share_ratio w x y z k → (x - y) = 1500 := 
by
  intros w x y z h_ratio
  rw [share_ratio] at h_ratio
  have h_w : w = k := h_ratio.1
  have h_x : x = 6 * k := h_ratio.2.1
  have h_y : y = 2 * k := h_ratio.2.2.1
  rw [h_x, h_y]
  rw [h] at h_x h_y
  sorry

end money_difference_l306_306977


namespace triangle_is_equilateral_l306_306963

-- Definitions and conditions of the problem
variables (A B C O X Y Z : Type) [triangle A B C]
variables (BX : altitude B A C) (CY : bisector C A B) (AZ : median A B C)
variables (intersect : BX ∩ CY ∩ AZ = O)
variables (CO_eq_BO : CO = BO)

-- The goal is to prove that the triangle ABC is equilateral
theorem triangle_is_equilateral (h1 : BX ∩ CY ∩ AZ = O)
                                (h2 : CO = BO) : equilateral A B C :=
by sorry

end triangle_is_equilateral_l306_306963


namespace am_gm_inequality_l306_306700

variable {x y z : ℝ}

theorem am_gm_inequality (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y + z) / 3 ≥ Real.sqrt (Real.sqrt (x * y) * Real.sqrt z) :=
by
  sorry

end am_gm_inequality_l306_306700


namespace complement_union_l306_306596

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306596


namespace Zoila_friends_13_l306_306019

def P : Type := ℕ

noncomputable def friends : P → P → Prop := sorry -- Replace with appropriate function

-- Condition 1: P_1 has 1 friend, P_2 has 2 friends, ... P_25 has 25 friends
axiom A1 : ∀ n : ℕ, n ∈ {1, 2, ..., 25} → ∃ p : P, p ≠ n ∧ (friends n p)

-- Condition 2: Friendship is mutual.
axiom A2 : ∀ {a b : P}, friends a b → friends b a

-- To prove: Zoila (P_26) has 13 friends.
theorem Zoila_friends_13 : (∃ l : list P, l.length = 13 ∧ ∀ p ∈ l, friends 26 p) :=
  sorry

end Zoila_friends_13_l306_306019


namespace function_translation_equivalence_l306_306466

noncomputable def f (x : ℝ) (h : x ≠ 3) : ℝ := (2 * x - 1) / (x - 3)

theorem function_translation_equivalence : 
  ∀ x, x ≠ 3 → f x (by assumption) = (5 / (x - 3) + 2) :=
by
  intros x hx
  dsimp [f]
  field_simp [hx]
  ring
  sorry

end function_translation_equivalence_l306_306466


namespace inequality_solution_min_value_of_a2_b2_c2_min_achieved_l306_306207

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (x - 1)

theorem inequality_solution :
  ∀ x : ℝ, (f x ≥ 3) ↔ (x ≤ -1 ∨ x ≥ 1) :=
by sorry

theorem min_value_of_a2_b2_c2 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  a^2 + b^2 + c^2 ≥ 3/7 :=
by sorry

theorem min_achieved (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  (2*a = b) ∧ (b = c/2) ∧ (a^2 + b^2 + c^2 = 3/7) :=
by sorry

end inequality_solution_min_value_of_a2_b2_c2_min_achieved_l306_306207


namespace sum_of_ages_l306_306002

theorem sum_of_ages (Petra_age : ℕ) (Mother_age : ℕ)
  (h_petra : Petra_age = 11)
  (h_mother : Mother_age = 36) :
  Petra_age + Mother_age = 47 :=
by
  -- Using the given conditions:
  -- Petra_age = 11
  -- Mother_age = 36
  sorry

end sum_of_ages_l306_306002


namespace infinite_series_sum_l306_306117

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end infinite_series_sum_l306_306117


namespace distinct_differences_permutation_iff_l306_306171

theorem distinct_differences_permutation_iff (n : ℕ) (hn : 0 < n) :
  (∃ σ : equiv.perm (fin n), (finset.image (λ k : fin n, (σ k).val - k.val) finset.univ).card = n) ↔ (n % 4 = 0 ∨ n % 4 = 1) :=
by
  sorry

end distinct_differences_permutation_iff_l306_306171


namespace complement_union_l306_306598

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306598


namespace infinite_series_value_l306_306120

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end infinite_series_value_l306_306120


namespace triangle_area_proof_l306_306363

noncomputable def area_of_triangle_ABC : ℝ :=
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  let area := 3 / 11
  area

theorem triangle_area_proof :
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  area_of_triangle_ABC = 3 / 11 :=
by
  sorry

end triangle_area_proof_l306_306363


namespace garden_area_increase_l306_306951

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l306_306951


namespace problem_solution_l306_306188

universe u

variable {R : Type u} [LinearOrderedField R]

def p (a m : R) : Prop :=
  ∀ (x1 x2 : R), x1 + x2 = m ∧ x1 * x2 = -2 → a^2 - 5 * a - 3 ≥ |x1 - x2|

def q (a : R) : Prop :=
  ∃ x : R, a * x^2 + 2 * x - 1 > 0

theorem problem_solution (a m : R) (h : ∀ m ∈ Icc (-1 : R) 1, p a m) (h_q_false : ¬q a) : a ≤ -1 :=
sorry

end problem_solution_l306_306188


namespace fisherman_multiple_l306_306081

theorem fisherman_multiple (pelican_fish : ℕ) (kingfisher_more : ℕ) (fisherman_more : ℕ)
    (h_pelican : pelican_fish = 13) (h_kingfisher_more : kingfisher_more = 7) 
    (h_fisherman_more : fisherman_more = 86) : 
    let kingfisher_fish := pelican_fish + kingfisher_more in
    let total_fish := pelican_fish + kingfisher_fish in
    let fisherman_fish := pelican_fish + fisherman_more in
    fisherman_fish = 3 * total_fish :=
by 
  sorry

end fisherman_multiple_l306_306081


namespace sum_of_cubes_div_xyz_l306_306821

-- Given: x, y, z are non-zero real numbers, and x + y + z = 0.
-- Prove: (x^3 + y^3 + z^3) / (xyz) = 3.
theorem sum_of_cubes_div_xyz (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by
  sorry

end sum_of_cubes_div_xyz_l306_306821


namespace equal_arcs_correspond_to_equal_chords_l306_306431

variable {α : Type*} [MetricSpace α]

/-- In the same circle or congruent circles: -/
theorem equal_arcs_correspond_to_equal_chords
  (C : Circle α) (arc1 arc2 : Subset α) (chord1 chord2 : Segment α)
  (h_eq_arc : arc1 = arc2) :
  chord1 = chord2 ↔ arc1 = arc2 :=
by
  sorry

end equal_arcs_correspond_to_equal_chords_l306_306431


namespace binomial_coefficients_sum_l306_306543

theorem binomial_coefficients_sum (x : ℝ) (n : ℕ) (h : x ≠ 0) 
  (h_sum : (2 * x + 1 / x^(1/2))^n = 64) :
  ∑ i in (finset.range (n.succ)), (nat.choose n i * (2 * x) ^ (n - i) * ((1 / x^(1/2)) ^ i)) = (3 : ℝ) ^ 6 := 
by 
  sorry

end binomial_coefficients_sum_l306_306543


namespace place_rooks_l306_306328

open Function

theorem place_rooks {n : ℕ} (hn_even : Even n) (hn_gt_two : 2 < n) (colors : Fin (n^2) → Fin (n^2/2)) 
  (colors_used_twice : ∀ c : Fin (n^2/2), (colors (c.castAdd 0)) == c ∧ (colors (c.castAdd 1)) == c) :
  ∃ (rooks : Fin n → Fin n), (injective rooks) ∧ (∀ i j, (i ≠ j) → (colors ((rooks i) * n + i)) ≠ (colors ((rooks j) * n + j ))) :=
sorry

end place_rooks_l306_306328


namespace max_correct_answers_l306_306088

theorem max_correct_answers
  (c w b : ℕ) -- Natural numbers for correct, wrong, and unanswered questions
  (h1 : c + w + b = 25) -- Total number of questions
  (h2 : 4 * c - 3 * w = 52) -- John's score
  (h3 : c + w ≥ 20) -- Minimum attempted questions
  : c ≤ 18 :=
begin
  -- Starting new goal, we can add contextual work here if needs proof strategy to be drawn
  iterate_cases 
    sorry
end

end max_correct_answers_l306_306088


namespace height_of_cylinder_eq_2_l306_306415

-- Define the height of the cylinder
def h_cylinder : ℝ := 2

-- Cylinder's radius and cone's given radius and height
def r_cylinder : ℝ := 8
def r_cone : ℝ := 8
def h_cone : ℝ := 6

-- Volume formula for the cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Volume formula for the cone
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- The proof statement asserting that the volumes are equal given the heights and radii
theorem height_of_cylinder_eq_2 :
    volume_cylinder r_cylinder h_cylinder = volume_cone r_cone h_cone := by
  sorry

end height_of_cylinder_eq_2_l306_306415


namespace otimes_value_l306_306500

def otimes (a b c : ℝ) : ℝ :=
  a / (b - c)

theorem otimes_value :
  otimes (otimes (otimes 2 5 3) 4 (otimes 5 1 2)) (otimes (otimes 3 7 2) 8 (otimes 4 9 5)) (otimes (otimes 1 6 5) (otimes 6 3 7) 2) = 5 / 17 :=
by
  sorry

end otimes_value_l306_306500


namespace area_WXYZ_l306_306824

variable {α : Type*} [Field α] {A B C D E W X Y Z : α}
variable (cyclic_quad : cyclic_quad A B C D)
variable (AC_BD_intersection : intersect_segment AC BD E)
variable (altitudes : ∀ {W Y}, is_foot W E DA ∧ is_foot Y E BC)
variable (midpoints : ∀ {X Z}, is_midpoint X AB ∧ is_midpoint Z CD)
variable (area_AED : area_of_triangle A E D = 9)
variable (area_BEC : area_of_triangle B E C = 25)
variable (angle_relation : ∠EBC - ∠ECB = 30)

theorem area_WXYZ :
  area_of_quad W X Y Z = 17 + (15 / 2) * Real.sqrt 3 := by
  sorry

end area_WXYZ_l306_306824


namespace ratio_a_to_c_l306_306873

-- Declaring the variables a, b, c, and d as real numbers.
variables (a b c d : ℝ)

-- Define the conditions given in the problem.
def ratio_conditions : Prop :=
  (a / b = 5 / 4) ∧ (c / d = 4 / 3) ∧ (d / b = 1 / 5)

-- State the theorem we need to prove based on the conditions.
theorem ratio_a_to_c (h : ratio_conditions a b c d) : a / c = 75 / 16 :=
by
  sorry

end ratio_a_to_c_l306_306873


namespace circle_intersection_unique_point_l306_306759

open Complex

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem circle_intersection_unique_point :
  ∃ k : ℝ, (distance (0, 0) (-5 / 2, 0) - 3 / 2 = k ∨ distance (0, 0) (-5 / 2, 0) + 3 / 2 = k)
  ↔ (k = 2 ∨ k = 5) := sorry

end circle_intersection_unique_point_l306_306759


namespace no_integer_solutions_l306_306399

theorem no_integer_solutions (x y z : ℤ) (h₀ : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solutions_l306_306399


namespace area_triangle_ABG_l306_306559

theorem area_triangle_ABG
  (A B F G : ℝ × ℝ)
  (l : ℝ → ℝ × ℝ → Prop)
  (M : ℝ × ℝ)
  (h1 : ∀ x y, l x y ↔ y^2 = 4 * x ∧ y = x - 1)
  (h2 : B.2 = -1 / 3 * A.2)
  (h3 : F = (1, 0))
  (h4 : l F.1 F.2)
  (h5 : l A.1 A.2 ∧ l B.1 B.2)
  (h6 : M = (1 / 2 * (A.1 + B.1), 1 / 2 * (A.2 + B.2)))
  (l' : ℝ → ℝ)
  (h7 : ∀ x, l' x ↔ x = 0 → M.1 * x + M.2)
  (hG : G.2 = 0 ∧ ∃ x, l' x = G)
  : let area := 1 / 2 * ((G.1 - F.1) * (A.2 + B.2))
    in area = 32 * real.sqrt 3 / 9 := 
sorry

end area_triangle_ABG_l306_306559


namespace find_n_and_coefficient_l306_306203

-- Problem restated in Lean 4 format

theorem find_n_and_coefficient (n : ℕ) :
  (∀ x > 0, let c1 := (sqrt x - 1 / (2 * x ^ (1 / 4))) ^ n in
   let c2 := (n : ℝ) / 2 * c1 in
   let c3 := (n * (n - 1) : ℝ) / 8 * c1 in
   c1 + c3 = 2 * c2) →
  n = 8 ∧
  let term := (2:ℝ) / 4 in -- The exponent part is 1/4
  (∃ r : ℕ, (16 - 3 * r) / 4 = 1) ∧
  (r = 4) ∧ (term ^ r * (8.choose r) = 35 / 8) :=
begin
  sorry
end

end find_n_and_coefficient_l306_306203


namespace problem_1_A_cap_B_problem_1_A_union_complement_B_problem_2_l306_306562

-- Define the sets A and B
def A (a : ℝ) : set ℝ := {x | (2 - a) ≤ x ∧ x ≤ (2 + a)}
def B : set ℝ := {x | x^2 - 5 * x + 4 ≥ 0}

-- Problem (1)
theorem problem_1_A_cap_B :
  A 3 ∩ B = {x : ℝ | (-1 : ℝ) ≤ x ∧ x ≤ 1 ∨ 4 ≤ x ∧ x ≤ 5} :=
sorry

theorem problem_1_A_union_complement_B (U : Type*) [univ : set U] :
  A 3 ∪ (univ \ B) = {x : ℝ | (-1 : ℝ) ≤ x ∧ x ≤ 5} :=
sorry

-- Problem (2)
theorem problem_2 (a : ℝ) :
  (A a ∩ B = ∅) → a < 1 :=
sorry

end problem_1_A_cap_B_problem_1_A_union_complement_B_problem_2_l306_306562


namespace separation_sequence_exists_l306_306112

theorem separation_sequence_exists :
  ∃ (l : List ℕ), (l.count 1 = 2 ∧ l.count 2 = 2 ∧ l.count 3 = 2 ∧
                   l.count 4 = 2 ∧ l.count 5 = 2 ∧ l.count 6 = 2 ∧
                   l.count 7 = 2 ∧ l.count 8 = 2 ∧ l.count 9 = 2 ∧
                   l.count 10 = 2 ∧ l.count 11 = 2) ∧
                  ((List.indexOf 1 l) + 2 = List.indexOfFrom 1 (List.indexOf 1 l + 1) l + 1) ∧
                  ((List.indexOf 2 l) + 3 = List.indexOfFrom 2 (List.indexOf 2 l + 1) l + 2) ∧
                  ((List.indexOf 3 l) + 4 = List.indexOfFrom 3 (List.indexOf 3 l + 1) l + 3) ∧
                  ((List.indexOf 4 l) + 5 = List.indexOfFrom 4 (List.indexOf 4 l + 1) l + 4) ∧
                  ((List.indexOf 5 l) + 6 = List.indexOfFrom 5 (List.indexOf 5 l + 1) l + 5) ∧
                  ((List.indexOf 6 l) + 7 = List.indexOfFrom 6 (List.indexOf 6 l + 1) l + 6) ∧
                  ((List.indexOf 7 l) + 8 = List.indexOfFrom 7 (List.indexOf 7 l + 1) l + 7) ∧
                  ((List.indexOf 8 l) + 9 = List.indexOfFrom 8 (List.indexOf 8 l + 1) l + 8) ∧
                  ((List.indexOf 9 l) + 10 = List.indexOfFrom 9 (List.indexOf 9 l + 1) l + 9) ∧
                  ((List.indexOf 10 l) + 11 = List.indexOfFrom 10 (List.indexOf 10 l + 1) l + 10) ∧
                  ((List.indexOf 11 l) + 12 = List.indexOfFrom 11 (List.indexOf 11 l + 1) l + 11) :=
sorry

end separation_sequence_exists_l306_306112


namespace right_handed_players_total_l306_306833

-- Definitions of the given quantities
def total_players : ℕ := 70
def throwers : ℕ := 49
def non_throwers : ℕ := total_players - throwers
def one_third_non_throwers : ℕ := non_throwers / 3
def left_handed_non_throwers : ℕ := one_third_non_throwers
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers

-- The theorem stating the main proof goal
theorem right_handed_players_total (h1 : total_players = 70)
                                   (h2 : throwers = 49)
                                   (h3 : total_players - throwers = non_throwers)
                                   (h4 : non_throwers = 21) -- derived from the above
                                   (h5 : non_throwers / 3 = left_handed_non_throwers)
                                   (h6 : non_throwers - left_handed_non_throwers = right_handed_non_throwers)
                                   (h7 : right_handed_throwers = throwers)
                                   (h8 : total_right_handed = right_handed_throwers + right_handed_non_throwers) :
  total_right_handed = 63 := sorry

end right_handed_players_total_l306_306833


namespace angle_equality_l306_306893

-- Definitions to establish the problem conditions
variables {Point : Type*} [AffineSpace ℝ Point]

def two_intersecting_circles (P Q A B C D : Point) : Prop :=
  -- Define the conditions for two intersecting circles intersecting at P and Q 
  -- and a line intersecting them at points A, B, C, and D
  sorry

-- Example of an angle declaration in the Lean theorem
def angle (A B C : Point) : ℝ := sorry

-- Define the theorem based on the proof problem
theorem angle_equality {P Q A B C D : Point} 
  (h : two_intersecting_circles P Q A B C D) :
  angle A P B = angle C Q D :=
sorry

end angle_equality_l306_306893


namespace equal_shipments_by_truck_l306_306362

theorem equal_shipments_by_truck (T : ℕ) (hT1 : 120 % T = 0) (hT2 : T ≠ 5) : T = 2 :=
by
  sorry

end equal_shipments_by_truck_l306_306362


namespace circle_area_and_circumference_l306_306839

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_area_and_circumference : 
  let Cx := 2; let Cy := 3; let Dx := 8; let Dy := 9 in
  let d := distance Cx Cy Dx Dy in
  let r := d / 2 in
  d = 6 * real.sqrt 2 ∧
  r = 3 * real.sqrt 2 ∧
  (π * r^2 = 18 * π) ∧
  (2 * π * r = 6 * real.sqrt 2 * π) :=
by
  let Cx := 2
  let Cy := 3
  let Dx := 8
  let Dy := 9
  let d := distance Cx Cy Dx Dy
  let r := d / 2
  have h1: d = 6 * real.sqrt 2 := sorry
  have h2: r = 3 * real.sqrt 2 := sorry
  have h3: π * r^2 = 18 * π := sorry
  have h4: 2 * π * r = 6 * real.sqrt 2 * π := sorry
  exact ⟨h1, h2, h3, h4⟩

end circle_area_and_circumference_l306_306839


namespace parabola_standard_equations_l306_306196

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end parabola_standard_equations_l306_306196


namespace problem1_problem2a_problem2b_l306_306061

theorem problem1 : ∃ l : ℝ → ℝ → Prop, 
  (l 0 0) ∧ (∃ p, (2 * p.1 + 3 * p.2 + 8 = 0) ∧ (p.1 - p.2 - 1 = 0) ∧ l p.1 p.2) ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) :=
by
  sorry

theorem problem2a : ∃ l : ℝ → ℝ → Prop, 
  (l 2 3) ∧ (∀ x y, l x y ↔ x + y = 5) :=
by
  sorry

theorem problem2b : ∃ l : ℝ → ℝ → Prop, 
  (l 2 3) ∧ (∀ x y, l x y ↔ 3 * x - 2 * y = 0) :=
by
  sorry

end problem1_problem2a_problem2b_l306_306061


namespace perimeter_of_ABFCDE_l306_306134

theorem perimeter_of_ABFCDE {side : ℝ} (h : side = 12) : 
  ∃ perimeter : ℝ, perimeter = 84 :=
by
  sorry

end perimeter_of_ABFCDE_l306_306134


namespace fraction_to_decimal_l306_306044

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end fraction_to_decimal_l306_306044


namespace margaret_mean_score_l306_306168

theorem margaret_mean_score : 
  let all_scores_sum := 832
  let cyprian_scores_count := 5
  let margaret_scores_count := 4
  let cyprian_mean_score := 92
  let cyprian_scores_sum := cyprian_scores_count * cyprian_mean_score
  (all_scores_sum - cyprian_scores_sum) / margaret_scores_count = 93 := by
  sorry

end margaret_mean_score_l306_306168


namespace compute_expression_l306_306382

-- Lean 4 statement for the mathematic equivalence proof problem
theorem compute_expression:
  (1004^2 - 996^2 - 1002^2 + 998^2) = 8000 := by
  sorry

end compute_expression_l306_306382


namespace complement_union_of_M_and_N_l306_306691

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306691


namespace mean_exercise_days_l306_306300

theorem mean_exercise_days :
  let students := [2, 4, 5, 3, 7, 2]
  let days := [0.5, 1, 3, 4, 6, 7]
  let total_days := (students.zip days).map (λ p => p.1 * p.2)
  let total_students := students.sum
  let mean_days := total_days.sum / total_students
  Float.round (mean_days, 2) = 3.83 :=
by
  sorry

end mean_exercise_days_l306_306300


namespace percent_married_women_below_30_insufficient_l306_306419

-- Definitions of the conditions
def num_employees := 7520
def percent_women := 0.58
def percent_men := 0.42
def percent_married := 0.60
def fraction_single_men := 2/3
def fraction_single_women_below_30 := 1/5
def fraction_married_women_above_50 := 1/3

-- Number calculations based on conditions
def num_women := (percent_women * num_employees).toNat
def num_men := (percent_men * num_employees).toNat
def num_married_emp := (percent_married * num_employees).toNat
def num_married_men := ((1/3) * num_men).toNat
def num_married_women := num_married_emp - num_married_men

-- Essential question translated into a theorem
theorem percent_married_women_below_30_insufficient :
  -- Given the stated conditions
  0 < num_employees → 
  0 < percent_women →
  0 < percent_men →
  0 < percent_married →
  0 < fraction_single_men →
  0 < fraction_single_women_below_30 →
  0 < fraction_married_women_above_50 →
  -- It is insufficient to determine the percentage of married women below the age of 30
  true := 
by 
  sorry

end percent_married_women_below_30_insufficient_l306_306419


namespace percentage_of_180_out_of_360_equals_50_l306_306036

theorem percentage_of_180_out_of_360_equals_50 :
  (180 / 360 : ℚ) * 100 = 50 := 
sorry

end percentage_of_180_out_of_360_equals_50_l306_306036


namespace garden_area_increase_l306_306947

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l306_306947


namespace standard_equation_of_parabola_l306_306198

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end standard_equation_of_parabola_l306_306198


namespace min_omega_symmetry_l306_306205

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem min_omega_symmetry :
  ∃ (omega : ℝ), omega > 0 ∧ 
  (∀ x : ℝ, Real.cos (omega * (x - π / 12)) = Real.cos (omega * (2 * (π / 4) - x) - omega * π / 12) ) ∧ 
  (∀ ω_, ω_ > 0 → 
  (∀ x : ℝ, Real.cos (ω_ * (x - π / 12)) = Real.cos (ω_ * (2 * (π / 4) - x) - ω_ * π / 12) → 
  omega ≤ ω_)) ∧ omega = 6 :=
sorry

end min_omega_symmetry_l306_306205


namespace value_of_fraction_l306_306034

theorem value_of_fraction :
  (16.factorial / (7.factorial * 9.factorial) = 5720 / 3) := by
  sorry

end value_of_fraction_l306_306034


namespace rhombus_perimeter_approx_equal_64_04_l306_306051

theorem rhombus_perimeter_approx_equal_64_04 
  (width length : ℝ) 
  (rhombus_inscribed : Prop) 
  (bf_eq_de: Prop) 
  (width_20 : width = 20) 
  (length_25 : length = 25) 
  (perimeter : ℝ) 
  (fence_def : perimeter = 4 * 16.01) 
:
  perimeter ≈ 64.04 :=
by 
  unfold perimeter
  rw [width_20, length_25]
  sorry -- the proof steps would go here

end rhombus_perimeter_approx_equal_64_04_l306_306051


namespace complement_union_eq_l306_306583

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306583


namespace shortest_leg_length_l306_306091

theorem shortest_leg_length (x : ℝ) (h : 15^2 + x^2 = 20^2) : x ≈ 13.23 :=
by {
  let y := Real.sqrt 175,
  have y_def : y = Real.sqrt 175 := rfl,
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow],
  sorry
}

end shortest_leg_length_l306_306091


namespace unique_solution_f_eq_x_l306_306485

theorem unique_solution_f_eq_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + y + f y) = 2 * y + f x ^ 2) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_eq_x_l306_306485


namespace find_equidistant_point_l306_306493

-- Define the points in 3D space
def point1 : ℝ × ℝ × ℝ := (0, 2, 1)
def point2 : ℝ × ℝ × ℝ := (1, 0, -1)
def point3 : ℝ × ℝ × ℝ := (-1, -1, 2)

-- Define the point in the xy-plane form
def point_in_xy_plane (x y : ℝ) : ℝ × ℝ × ℝ := (x, y, 0)

-- Define the distance squared between two points in 3D
def dist_squared (p q : ℝ × ℝ × ℝ) : ℝ := (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2

-- The point we want to prove is equidistant
def target_point : ℝ × ℝ × ℝ := (5/2, 1/2, 0)

-- The statement we need to prove
theorem find_equidistant_point : 
  dist_squared target_point point1 = dist_squared target_point point2 ∧
  dist_squared target_point point1 = dist_squared target_point point3 := 
  sorry

end find_equidistant_point_l306_306493


namespace quadratic_residue_l306_306280

theorem quadratic_residue (a : ℤ) (p : ℕ) (hp : p > 2) (ha_nonzero : a ≠ 0) :
  (∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ 1 [ZMOD p]) ∧
  (¬ ∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ -1 [ZMOD p]) :=
sorry

end quadratic_residue_l306_306280


namespace find_ns_with_polynomial_conditions_l306_306137

theorem find_ns_with_polynomial_conditions:
  {n : ℕ} (hn_pos : n > 0) 
  (h_poly : ∃ p : Polynomial ℤ, 
    degree p = n ∧ 
    p.eval 0 = 0 ∧ 
    (∃ S : Finset ℤ, S.card = n ∧ ∀ x ∈ S, p.eval x = n)) → 
  n = 1 ∨ n = 2 ∨ n = 6 ∨ n = 12 ∨ n = 24 :=
by
  intros n hn_pos h_poly
  sorry

end find_ns_with_polynomial_conditions_l306_306137


namespace complement_union_of_M_and_N_l306_306692

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306692


namespace largest_three_digit_geometric_sequence_l306_306028

theorem largest_three_digit_geometric_sequence :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧
             (∀ (d1 d2 d3 : ℕ), d1 = 8 → 
                                d2 = 8 * r →
                                d3 = 8 * r^2 →
                                (d1, d2, d3) ∈ digits n →
                                n = 842) :=
begin
  sorry
end

end largest_three_digit_geometric_sequence_l306_306028


namespace square_of_binomial_is_25_l306_306719

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l306_306719


namespace sum_sqrt_inequality_l306_306059

theorem sum_sqrt_inequality (n : ℕ) (x : Fin n → ℝ)
  (h0 : ∀ i, 0 < x i)
  (h1 : ∑ i, x i = 1) :
  (∑ i, Real.sqrt (x i)) * (∑ i, 1 / Real.sqrt (1 + x i)) ≤ n^2 / Real.sqrt (n + 1) := 
sorry

end sum_sqrt_inequality_l306_306059


namespace find_f_2015_l306_306048

theorem find_f_2015 (f : ℕ → ℕ) (h1 : ∀ n, f(f(n)) + f(n) = 2n + 3) (h2 : f(0) = 1) : f(2015) = 2016 :=
sorry

end find_f_2015_l306_306048


namespace lcm_inequality_l306_306305

theorem lcm_inequality (k m n : ℕ) (hk : k > 0) (hm : m > 0) (hn : n > 0) :
  Nat.lcm k m * Nat.lcm m n * Nat.lcm n k ≥ Nat.lcm k m n * Nat.lcm k m n := 
by 
  sorry

end lcm_inequality_l306_306305


namespace roots_odd_even_l306_306729

theorem roots_odd_even (n : ℤ) (x1 x2 : ℤ) (h_eqn : x1^2 + (4 * n + 1) * x1 + 2 * n = 0) (h_eqn' : x2^2 + (4 * n + 1) * x2 + 2 * n = 0) :
  ((x1 % 2 = 0 ∧ x2 % 2 ≠ 0) ∨ (x1 % 2 ≠ 0 ∧ x2 % 2 = 0)) :=
sorry

end roots_odd_even_l306_306729


namespace example_theorem_l306_306667

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306667


namespace students_in_cafeteria_final_l306_306882

-- Define the initial conditions
def total_students : ℕ := 300
def indoor_cafeteria_initial : ℕ := (2 * total_students) / 5
def covered_picnic_outside_initial : ℕ := (3 * total_students) / 10
def classroom_initial : ℕ := total_students - indoor_cafeteria_initial - covered_picnic_outside_initial

def moved_from_outside_to_cafeteria : ℕ := (4 * covered_picnic_outside_initial) / 10
def lost_from_cafeteria_to_outside : ℕ := 5
def joined_from_classroom_to_cafeteria : ℕ := (15 * classroom_initial) / 100
def moved_from_outside_to_classroom : ℕ := 2

-- Final count for the cafeteria
def final_cafeteria_count : ℕ :=
  indoor_cafeteria_initial + moved_from_outside_to_cafeteria - lost_from_cafeteria_to_outside + joined_from_classroom_to_cafeteria

-- Main theorem statement
theorem students_in_cafeteria_final : final_cafeteria_count = 165 :=
by
  -- Definitions
  let indoor_cafeteria_initial := 120
  let covered_picnic_outside_initial := 90
  let classroom_initial := 90
  let moved_from_outside_to_cafeteria := 36
  let lost_from_cafeteria_to_outside := 5
  let joined_from_classroom_to_cafeteria := 14
  let final_cafeteria_count := indoor_cafeteria_initial + moved_from_outside_to_cafeteria - lost_from_cafeteria_to_outside + joined_from_classroom_to_cafeteria
  -- Use expected final value
  show final_cafeteria_count = 165 from rfl

end students_in_cafeteria_final_l306_306882


namespace isosceles_triangle_BC_length_l306_306255

theorem isosceles_triangle_BC_length (A B C H : Type) 
  (AB AC : ℝ) 
  (AB_eq_AC : AB = AC) 
  (AC_val : AC = 5) 
  (AH HC : ℝ) 
  (AH_eq_4HC : AH = 4 * HC) 
  (H_on_AC : H ∈ line_segment A C) 
  (altitude_BH : is_altitude B H) : 
  ∃ (BC : ℝ), BC = sqrt 10 :=
by
  sorry

end isosceles_triangle_BC_length_l306_306255


namespace number_at_2004th_position_l306_306830

theorem number_at_2004th_position :
  ∃ (k : ℕ), 
  (1 ≤ k ∧ k ≤ 10000) ∧ 
  (∀ n : ℕ, (1 ≤ n ≤ 10000) → (n % 5 = 0 ∨ n % 11 = 0) → ((List.filter (λ x, x % 5 = 0 ∨ x % 11 = 0) (List.range' 1 10000)).nth 2003 = some k)) ∧ 
  k = 7348 :=
sorry

end number_at_2004th_position_l306_306830


namespace angle_A_30_side_b_sqrt2_l306_306247

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the dot product of vectors AB and AC is 2√3 times the area S, 
    then angle A equals 30 degrees --/
theorem angle_A_30 {a b c S : ℝ} (h : (a * b * Real.sqrt 3 * c * Real.sin (π / 6)) = 2 * Real.sqrt 3 * S) : 
  A = π / 6 :=
sorry

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the tangent of angles A, B, C are in the ratio 1:2:3 and c equals 1, 
    then side b equals √2 --/
theorem side_b_sqrt2 {A B C : ℝ} (a b c : ℝ) (h_tan_ratio : Real.tan A / Real.tan B = 1 / 2 ∧ Real.tan B / Real.tan C = 2 / 3)
  (h_c : c = 1) : b = Real.sqrt 2 :=
sorry

end angle_A_30_side_b_sqrt2_l306_306247


namespace diff_course_choices_l306_306113

/-- Chongqing No.8 Middle School offers 6 different math elective courses. -/
def elective_courses : ℕ := 6

/-- A student can choose to study 1 or 2 courses. -/
def student_choice := { n : ℕ | n = 1 ∨ n = 2 }

/-- The number of different ways that students A, B, and C can choose courses such that their choices are different
 is 1290 -/
theorem diff_course_choices (A B C : { s : set (fin 6) | card s ∈ student_choice }) :
  A ≠ B → B ≠ C → A ≠ C → (number_of_ways A B C = 1290) :=
sorry

end diff_course_choices_l306_306113


namespace tom_has_7_blue_tickets_l306_306890

def number_of_blue_tickets_needed_for_bible := 10 * 10 * 10
def toms_current_yellow_tickets := 8
def toms_current_red_tickets := 3
def toms_needed_blue_tickets := 163

theorem tom_has_7_blue_tickets : 
  (number_of_blue_tickets_needed_for_bible - 
    (toms_current_yellow_tickets * 10 * 10 + 
     toms_current_red_tickets * 10 + 
     toms_needed_blue_tickets)) = 7 :=
by
  -- Proof can be provided here
  sorry

end tom_has_7_blue_tickets_l306_306890


namespace horner_eval_hex_to_decimal_l306_306929

-- Problem 1: Evaluate the polynomial using Horner's method
theorem horner_eval (x : ℤ) (f : ℤ → ℤ) (v3 : ℤ) :
  (f x = 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12) →
  x = -4 →
  v3 = (((((3 * x + 5) * x + 6) * x + 79) * x - 8) * x + 35) * x + 12 →
  v3 = -57 :=
by
  intros hf hx hv
  sorry

-- Problem 2: Convert hexadecimal base-6 to decimal
theorem hex_to_decimal (hex : ℕ) (dec : ℕ) :
  hex = 210 →
  dec = 0 * 6^0 + 1 * 6^1 + 2 * 6^2 →
  dec = 78 :=
by
  intros hhex hdec
  sorry

end horner_eval_hex_to_decimal_l306_306929


namespace complement_union_l306_306611

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306611


namespace initial_time_for_train_l306_306980

theorem initial_time_for_train (S : ℝ)
  (length_initial : ℝ := 12 * 15)
  (length_detached : ℝ := 11 * 15)
  (time_detached : ℝ := 16.5)
  (speed_constant : S = length_detached / time_detached) :
  (length_initial / S = 18) :=
by
  sorry

end initial_time_for_train_l306_306980


namespace count_polynomials_l306_306808

/-- Let H be the set of polynomials of the form P(z) = z^n + c_(n-1)z^(n-1) + ... + c_2z^2 + c_1z + 36,
  where c_1, c_2, ..., c_(n-1) are integers, and P(z) has distinct roots of the form a + ib with 
  a and b being integers. The number of such polynomials is N. -/
theorem count_polynomials (H : set (polynomial ℤ)) (N : ℕ) :
  (∀ P ∈ H, ∃ (n : ℕ) (c : fin n → ℤ), P = polynomial.mk_fin n c (λ (i : fin n), c i) ∧ 
  (∀ z ∈ (P.roots), ∃ (a b : ℤ), z = a + b * I ∧ b ≠ 0)) →
  H.card = N :=
sorry

end count_polynomials_l306_306808


namespace permutations_with_ea_l306_306859

theorem permutations_with_ea :
  let letters := ['C', 'N', 'D', 'r', 'e', 'a', 'm']
  ∃ (n : ℕ), n = 600 ∧ 
  (n = (choose 5 4) * (fact 5)) := 
begin
  let letters := ['C', 'N', 'D', 'r', 'e', 'a', 'm'],
  have h1 : choose 5 4 = 5,
  { exact nat.choose_eq_nat_cho (by norm_num : 5=5) (by norm_num : 4=4) },
  have h2 : fact 5 = 120,
  { exact nat.factorial_five },
  use 600,
  split,
  { norm_num },
  { rw [h1, h2],
    norm_num }
end

end permutations_with_ea_l306_306859


namespace solve_equation_l306_306854

def solutions (x : ℝ) (n : ℤ) : Prop :=
  x = -ℝ.pi / 3 + 2 * ℝ.pi * n ∨ x = -ℝ.pi / 4 + 2 * ℝ.pi * n

theorem solve_equation (x : ℝ) (n : ℤ) : 
  (cos x + sin x ≠ -sqrt 3 / 2 ∧ sqrt (sqrt 3 * cos x - sin x) ≠ 0) →
  ((sin (2 * x) - cos (2 * x) + sqrt 3 * cos x + sqrt 3 * sin x + 1) / sqrt (sqrt 3 * cos x - sin x) = 0)
  →
  solutions x n :=
by
  sorry

end solve_equation_l306_306854


namespace triangles_from_points_l306_306749

theorem triangles_from_points (P : Finset (ℕ × ℕ)) (h_card : P.card = 12) (C : Finset (ℕ × ℕ)) (hC_card : C.card = 4) 
(hC_collinear : ∀ p1 p2 p3 ∈ C, collinear ({p1, p2, p3} : Set (ℕ × ℕ)))
(h_non_collinear : ∀ (p1 p2 p3 : Finset (ℕ × ℕ)), p1 ⊆ P \ C → p2 ⊆ P \ C → p3 ⊆ P \ C → p1 ∪ p2 ∪ p3 = P \ C → ¬ collinear ({p1, p2, p3} : Set (ℕ × ℕ))):
  ∃ T : Finset {S : Finset (ℕ × ℕ) | S.card = 3}, T.card = 216 :=
sorry

end triangles_from_points_l306_306749


namespace g_is_odd_l306_306267

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end g_is_odd_l306_306267


namespace range_of_y_l306_306179

theorem range_of_y :
  ∀ (y x : ℝ), x = 4 - y → (-2 ≤ x ∧ x ≤ -1) → (5 ≤ y ∧ y ≤ 6) :=
by
  intros y x h1 h2
  sorry

end range_of_y_l306_306179


namespace tangent_line_at_point_l306_306513

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, x^3 + a * x^2 - 2 * x

theorem tangent_line_at_point (a : ℝ) (h : ∀ x, f a (-x) = -f a x) : 
  (1:ℝ) - (-1:ℝ) - 2 = 0 :=
by {
  have h1 : a = 0,
  { sorry }, -- Comes from proving odd function condition...
  have f1 : f 0 1 = -1,
  { rw h1, simp [f], },
  have f_prime : (λ (x : ℝ), f 0 x)' (1 : ℝ) = 1,
  { sorry }, -- Calculating derivative
  show (1:ℝ) - (-1:ℝ) - 2 = 0,
  { simp, ring }
}

end tangent_line_at_point_l306_306513


namespace ratio_of_sum_to_first_term_l306_306542

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - (2 ^ n)) / (1 - 2)

-- Main statement to be proven
theorem ratio_of_sum_to_first_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geo : geometric_sequence a 2) (h_sum : sum_of_first_n_terms a S) :
  S 3 / a 0 = 7 :=
sorry

end ratio_of_sum_to_first_term_l306_306542


namespace rope_cut_probability_l306_306085

theorem rope_cut_probability :
  let rope_length : ℝ := 3
  let condition (cut_position : ℝ) : Prop := 0 ≤ cut_position ∧ cut_position ≤ rope_length
  let segment1_length (cut_position : ℝ) : ℝ := cut_position
  let segment2_length (cut_position : ℝ) : ℝ := rope_length - cut_position
  let event_A (cut_position : ℝ) : Prop := 1 ≤ segment1_length cut_position ∧ 1 ≤ segment2_length cut_position
  let probability_A : ℝ := {
    {cut_position | condition cut_position ∧ event_A cut_position}.measure / {cut_position | condition cut_position}.measure
  }
  in probability_A = 1 / 3 := 
begin
  sorry
end

end rope_cut_probability_l306_306085


namespace correct_inequality_l306_306417

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_increasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0

theorem correct_inequality : f (-2) < f 1 ∧ f 1 < f 3 :=
by 
  sorry

end correct_inequality_l306_306417


namespace degree_not_determined_from_characteristic_l306_306786

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l306_306786


namespace math_problem_l306_306823

theorem math_problem {n : ℕ} (x y : Fin n → ℝ) (m : ℕ) (h_pos: 0 < m)
  (h_nonneg : ∀ i, 0 ≤ x i ∧ 0 ≤ y i)
  (h_sum: ∀ i, x i + y i = 1) :
  (1 - ∏ i, x i) ^ m + (∏ i, 1 - (y i) ^ m) ≥ 1 := by
  sorry

end math_problem_l306_306823


namespace units_digit_24_pow_4_plus_42_pow_4_l306_306909

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end units_digit_24_pow_4_plus_42_pow_4_l306_306909


namespace ellipse_area_quadrants_eq_zero_l306_306880

theorem ellipse_area_quadrants_eq_zero 
(E : Type)
(x y : E → ℝ) 
(h_ellipse : ∀ (x y : ℝ), (x - 19)^2 / (19 * 1998) + (y - 98)^2 / (98 * 1998) = 1998) 
(R1 R2 R3 R4 : ℝ)
(H1 : ∀ (R1 R2 R3 R4 : ℝ), R1 = R_ellipse / 4 ∧ R2 = R_ellipse / 4 ∧ R3 = R_ellipse / 4 ∧ R4 = R_ellipse / 4)
: R1 - R2 + R3 - R4 = 0 := 
by 
sorry

end ellipse_area_quadrants_eq_zero_l306_306880


namespace polygon_interior_angles_l306_306736

theorem polygon_interior_angles (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 := 
by 
-- Given, the sum of the interior angles is (n - 2) * 180 = 1080
-- So, solve for n
-- simplified as n - 2 = 1080 / 180
-- hence, n - 2 = 6
-- therefore, n = 6 + 2
sory

end polygon_interior_angles_l306_306736


namespace tangent_line_at_point_a_eq_1_range_of_a_for_positivity_l306_306206

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x + 1) - a * Real.log x + a

theorem tangent_line_at_point_a_eq_1 :
  let a := 1 in let f := f x a in
  ∀ x, ((f 1 '(a := 1)), (1, f 1))
  (∀ x : ℝ, (e ^ 2 - 1) * x - y - 2 = 0) := 
  sorry

theorem range_of_a_for_positivity :
  ∀ a : ℝ, (0 < a ∧ a < Real.exp 2) →
  ∀ x : ℝ, (0 < x → f x a > 0) := 
  sorry


end tangent_line_at_point_a_eq_1_range_of_a_for_positivity_l306_306206


namespace choir_singers_joined_final_verse_l306_306411

theorem choir_singers_joined_final_verse (total_singers : ℕ) (first_verse_fraction : ℚ)
  (second_verse_fraction : ℚ) (initial_remaining : ℕ) (second_verse_joined : ℕ) : 
  total_singers = 30 → 
  first_verse_fraction = 1 / 2 → 
  second_verse_fraction = 1 / 3 → 
  initial_remaining = total_singers / 2 → 
  second_verse_joined = initial_remaining / 3 → 
  (total_singers - (initial_remaining + second_verse_joined)) = 10 := 
by
  intros
  sorry

end choir_singers_joined_final_verse_l306_306411


namespace third_median_length_l306_306020

theorem third_median_length (a b: ℝ) (h_a: a = 5) (h_b: b = 8)
  (area: ℝ) (h_area: area = 6 * Real.sqrt 15) (m: ℝ):
  m = 3 * Real.sqrt 6 :=
sorry

end third_median_length_l306_306020


namespace regular_polygon_sides_with_interior_angle_162_l306_306975

theorem regular_polygon_sides_with_interior_angle_162 {n : ℕ} :
  let interior_angle := 162 in
  let exterior_angle := 180 - interior_angle in
  let total_exterior_angles := 360 in
  180 * (n - 2) = 162 * n → (total_exterior_angles / exterior_angle) = 20 :=
by
  intros
  sorry

end regular_polygon_sides_with_interior_angle_162_l306_306975


namespace complement_union_l306_306653

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306653


namespace area_under_pressure_l306_306237

theorem area_under_pressure (F : ℝ) (S : ℝ) (p : ℝ) (hF : F = 100) (hp : p > 1000) (hpressure : p = F / S) :
  S < 0.1 :=
by
  sorry

end area_under_pressure_l306_306237


namespace initial_distance_l306_306896

def relative_speed (v1 v2 : ℝ) : ℝ := v1 + v2

def total_distance (rel_speed time : ℝ) : ℝ := rel_speed * time

theorem initial_distance (v1 v2 time : ℝ) : (v1 = 1.6) → (v2 = 1.9) → 
                                            (time = 100) →
                                            total_distance (relative_speed v1 v2) time = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [relative_speed, total_distance]
  sorry

end initial_distance_l306_306896


namespace complement_union_M_N_l306_306634

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306634


namespace example_theorem_l306_306671

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306671


namespace parameter_exists_solution_l306_306488

theorem parameter_exists_solution (b : ℝ) (h : b ≥ -2 * Real.sqrt 2 - 1 / 4) :
  ∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y) :=
by
  sorry

end parameter_exists_solution_l306_306488


namespace regular_pentagon_min_rotation_overlap_l306_306423

/-- The minimum degree of rotation for a regular pentagon to overlap with itself is 72 degrees. -/
theorem regular_pentagon_min_rotation_overlap : 
  ∀ (P : Type), is_regular_pentagon P → (min_rotation_degrees P ≥ 72) :=
sorry

end regular_pentagon_min_rotation_overlap_l306_306423


namespace exists_points_XY_l306_306522

open Set

noncomputable def circle_with_diameter_AB (A B : Point) : Circle := sorry

axiom point_on_AB (A B C : Point) (h : C ∈ LineSegment A B) : Prop

theorem exists_points_XY (A B C : Point) (hC : point_on_AB A B C) :
  ∃ (X Y : Point), X ∈ circle_with_diameter_AB A B ∧ 
  Y ∈ circle_with_diameter_AB A B ∧ 
  (Y = reflect_over_line_AB X A B) ∧ 
  is_perpendicular (line_through C Y) (line_through X A) :=
sorry

end exists_points_XY_l306_306522


namespace S_n_leq_n_sq_sub_14_exists_n_satisfying_S_eq_n_sq_sub_14_infinite_n_satisfying_S_eq_n_sq_sub_14_l306_306501

def S (n : ℕ) : ℕ := -- Definition from problem context for S(n)
  sorry -- Placeholder for the function definition since we don't have the exact formula.

theorem S_n_leq_n_sq_sub_14 (n : ℕ) (h : n ≥ 4) : S(n) ≤ n^2 - 14 := sorry

theorem exists_n_satisfying_S_eq_n_sq_sub_14 : ∃ (n : ℕ), S(n) = n^2 - 14 := sorry

theorem infinite_n_satisfying_S_eq_n_sq_sub_14 : ∃ᶠ (n : ℕ) in Filter.at_top, S(n) = n^2 - 14 := sorry

end S_n_leq_n_sq_sub_14_exists_n_satisfying_S_eq_n_sq_sub_14_infinite_n_satisfying_S_eq_n_sq_sub_14_l306_306501


namespace hexagon_diagonal_ratio_collinear_l306_306470

theorem hexagon_diagonal_ratio_collinear 
  (A B C D E F M N : ℝ × ℝ)
  (h_regular : regular_hexagon A B C D E F)
  (h_division_AC : ∃ λ : ℝ, (λ > 0) ∧ AM / AC = λ)
  (h_division_CE : ∃ λ : ℝ, (λ > 0) ∧ CN / CE = λ)
  (h_collinear : collinear {B, M, N}) :
  λ = 1 / real.sqrt 3 :=
sorry

end hexagon_diagonal_ratio_collinear_l306_306470


namespace sin_expression_eq_one_l306_306818

theorem sin_expression_eq_one (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (sin (4 * c) * sin (8 * c) * sin (12 * c) * sin (16 * c) * sin (20 * c)) /
  (sin c * sin (2 * c) * sin (3 * c) * sin (5 * c) * sin (6 * c)) = 1 :=
by
  rw [h]
  sorry

end sin_expression_eq_one_l306_306818


namespace similarity_coordinates_l306_306256

theorem similarity_coordinates {B B1 : ℝ × ℝ} 
  (h₁ : ∃ (k : ℝ), k = 2 ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → x₁ = x / k ∨ x₁ = x / -k) ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → y₁ = y / k ∨ y₁ = y / -k))
  (h₂ : B = (-4, -2)) :
  B1 = (-2, -1) ∨ B1 = (2, 1) :=
sorry

end similarity_coordinates_l306_306256


namespace weight_removal_l306_306011

theorem weight_removal {n k : ℕ} (w : Fin n → ℕ) (h : ∃ S : Finset (Fin n), S.card = k ∧ ∀ x ∈ S, ∀ i j : ℕ, i < j → sum (w '' S.filter (λ x, x = i ∨ x = j)) = sum (w '' S.filter (λ x, x = j ∨ x = i))) :
  ∃ T : Finset (Fin n), T.card ≥ k ∧ ∀ t ∈ T, 
    ¬(∃ U : Finset (Fin (n-1)), U.card = k ∧ U ∪ {t} = S ∧ sum (w '' U) = sum (w '' U)) :=
sorry

end weight_removal_l306_306011


namespace Matthew_shares_less_valuable_stock_l306_306297

def number_of_shares_less_valuable_stock
  (shares_more : ℕ) (price_more : ℚ) (total_assets : ℚ) (ratio : ℚ) : ℕ :=
  let value_more := shares_more * price_more
  let price_less := price_more / ratio
  let value_less := total_assets - value_more
  let shares_less := value_less / price_less
  shares_less.toNat 

theorem Matthew_shares_less_valuable_stock : 
  ∀ (shares_more : ℕ) (price_more : ℚ) (total_assets : ℚ) (ratio : ℚ), 
    shares_more = 14 → 
    price_more = 78 → 
    total_assets = 2106 → 
    ratio = 2 → 
    number_of_shares_less_valuable_stock shares_more price_more total_assets ratio = 26 :=
by
  intros;
  sorry

end Matthew_shares_less_valuable_stock_l306_306297


namespace sum_of_three_numbers_l306_306877

theorem sum_of_three_numbers
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 252)
  (h2 : ab + bc + ca = 116) :
  a + b + c = 22 :=
by
  sorry

end sum_of_three_numbers_l306_306877


namespace value_of_a_plus_b_2023_l306_306215

theorem value_of_a_plus_b_2023 
    (x y a b : ℤ)
    (h1 : 4*x + 3*y = 11)
    (h2 : 2*x - y = 3)
    (h3 : a*x + b*y = -2)
    (h4 : b*x - a*y = 6)
    (hx : x = 2)
    (hy : y = 1) :
    (a + b) ^ 2023 = 0 := 
sorry

end value_of_a_plus_b_2023_l306_306215


namespace denomination_is_100_l306_306298

-- Define the initial conditions
def num_bills : ℕ := 8
def total_savings : ℕ := 800

-- Define the denomination of the bills
def denomination_bills (num_bills : ℕ) (total_savings : ℕ) : ℕ := 
  total_savings / num_bills

-- The theorem stating the denomination is $100
theorem denomination_is_100 :
  denomination_bills num_bills total_savings = 100 := by
  sorry

end denomination_is_100_l306_306298


namespace exists_symmetric_points_l306_306210

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m
noncomputable def g (x : ℝ) : ℝ := -log (1 / x) - 3 * x
noncomputable def h (x : ℝ) : ℝ := -log x + 3 * x - x^2

theorem exists_symmetric_points (m : ℝ) :
  (∃ x ∈ Icc (1/2 : ℝ) 2, f x m = g x) ↔ 2 - log 2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end exists_symmetric_points_l306_306210


namespace ln_b_over_a_range_l306_306217

-- Given three positive real numbers a, b, c
variables (a b c : ℝ)

-- Conditions
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom ratio_cond : (1 / real.exp 1) ≤ (c / a) ∧ (c / a) ≤ 2
axiom eq_cond : c * real.log b = a + c * real.log c

-- Statement of the problem
theorem ln_b_over_a_range : 1 ≤ real.log (b / a) ∧ real.log (b / a) ≤ real.exp 1 - 1 :=
by
  sorry

end ln_b_over_a_range_l306_306217


namespace general_form_of_quadratic_l306_306334

theorem general_form_of_quadratic (x : ℝ) : 4 * x = x ^ 2 - 8 ↔ x ^ 2 - 4 * x - 8 = 0 :=
by {
  have : (4 * x = x ^ 2 - 8) = (x ^ 2 - 4 * x - 8 = 0),
  {
    split,
    {
      intro h,
      rw [h],
      exact rfl,
    },
    {
      intro h,
      rw [←h],
      exact rfl,
    },
  },
  exact this,
}

end general_form_of_quadratic_l306_306334


namespace sum_of_possible_n_values_l306_306003

theorem sum_of_possible_n_values :
  ∀ (n a_1: ℕ),
  n > 1 → 
  (∃ a_1: ℤ, (∑ i in finset.range n, (a_1 + i * 2)) = 2000) →
  (n ≠ 1) →
  (sum (x ∈ finset.filter (λ n, ∃ a_1: ℤ, ( ∑ i in finset.range n, (a_1 + i * 2)) = 2000) (finset.range (2000 + 1))) id) = 4835
    :=
by
  sorry

end sum_of_possible_n_values_l306_306003


namespace number_of_paths_A_to_B_l306_306985

def labeled_points : Type := {A, B, C, D, E, F, G}

def is_connected (a b : labeled_points) : Prop :=
  (a = A ∧ b = C) ∨ (a = A ∧ b = D) ∨ (a = C ∧ b = B) ∨ (a = D ∧ b = C) ∨
  (a = D ∧ b = E) ∨ (a = C ∧ b = F) ∨ (a = D ∧ b = F) ∨ (a = E ∧ b = F) ∨
  (a = F ∧ b = B) ∨ (a = D ∧ b = G) ∨ (a = F ∧ b = G) ∨ (a = E ∧ b = G)

def is_valid_path (path : list labeled_points) : Prop :=
  (path.head = some A) ∧ (path.last = some B) ∧
  (path.nodup) ∧
  (∀ (a b : labeled_points), (a, b) ∈ path.zip path.tail → is_connected a b)

noncomputable def count_valid_paths (start end : labeled_points) : ℕ :=
  (if (start = A ∧ end = B) then 15 else 0) -- Based on provided solution

theorem number_of_paths_A_to_B :
  count_valid_paths A B = 15 :=
by sorry

end number_of_paths_A_to_B_l306_306985


namespace degree_not_determined_by_A_P_l306_306777

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l306_306777


namespace stream_speed_correct_l306_306920

-- Conditions
def downstream_speed : ℝ := 15
def upstream_speed : ℝ := 8

-- Define the stream speed according to the given conditions
def stream_speed : ℝ := (downstream_speed - upstream_speed) / 2

-- Statement to prove
theorem stream_speed_correct : stream_speed = 3.5 := by
  -- Currently proof not required, thus we use sorry to skip the proof steps
  sorry

end stream_speed_correct_l306_306920


namespace fraction_to_decimal_l306_306043

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end fraction_to_decimal_l306_306043


namespace complement_union_l306_306616

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306616


namespace sequence_formula_l306_306038

-- Define the problem when n >= 2
theorem sequence_formula (n : ℕ) (h : n ≥ 2) : 
  1 / (n^2 - 1) = (1 / 2) * (1 / (n - 1) - 1 / (n + 1)) := 
by {
  sorry
}

end sequence_formula_l306_306038


namespace linear_function_no_first_quadrant_l306_306336

theorem linear_function_no_first_quadrant : 
  ¬ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = -3 * x - 2 := by
  sorry

end linear_function_no_first_quadrant_l306_306336


namespace complement_union_l306_306678

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306678


namespace ratio_of_number_to_ten_l306_306080

theorem ratio_of_number_to_ten (n : ℕ) (h : n = 200) : n / 10 = 20 :=
by
  sorry

end ratio_of_number_to_ten_l306_306080


namespace fabian_cards_l306_306158

theorem fabian_cards : ∃ (g y b r : ℕ),
  (g > 0 ∧ g < 10) ∧ (y > 0 ∧ y < 10) ∧ (b > 0 ∧ b < 10) ∧ (r > 0 ∧ r < 10) ∧
  (g * y = g) ∧
  (b = r) ∧
  (b * r = 10 * g + y) ∧ 
  (g = 8) ∧
  (y = 1) ∧
  (b = 9) ∧
  (r = 9) :=
by
  sorry

end fabian_cards_l306_306158


namespace verify_rhombus_properties_l306_306084

noncomputable def area_of_rhombus (d1 d2 : ℕ) : ℕ :=
(d1 * d2) / 2

noncomputable def side_length (d1 d2 : ℕ) : ℝ :=
real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

def perimeter (side_length : ℝ) : ℝ :=
4 * side_length

theorem verify_rhombus_properties :
  let d1 := 30, d2 := 18, P := 80 in
  let calculated_area := area_of_rhombus d1 d2 in
  let s := side_length d1 d2 in
  let calculated_perimeter := perimeter s in
  calculated_area = 270 ∧ calculated_perimeter ≠ P :=
by {
  sorry
}

end verify_rhombus_properties_l306_306084


namespace find_line_equation_l306_306162

theorem find_line_equation : 
  ∃ (m : ℝ), (∀ (x y : ℝ), (2 * x + y - 5 = 0) → (m = -2)) → 
  ∀ (x₀ y₀ : ℝ), (x₀ = -2) ∧ (y₀ = 3) → 
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * x₀ + b * y₀ + c = 0) ∧ (a = 1 ∧ b = -2 ∧ c = 8) := 
by
  sorry

end find_line_equation_l306_306162


namespace find_range_of_m_l306_306530

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  m^2 - 4 > 0

def inequality_holds_for_all_real_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * (m + 1) * x + m * (m + 1) > 0

def p (m : ℝ) : Prop := has_two_distinct_real_roots m
def q (m : ℝ) : Prop := inequality_holds_for_all_real_x m

theorem find_range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > 2 ∨ (-2 ≤ m ∧ m < -1)) :=
sorry

end find_range_of_m_l306_306530


namespace find_AB_length_find_AB_equation_l306_306521

-- Given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8
def P0 := (-1, 2 : ℝ)
def is_point_inside_circle (p : ℝ × ℝ) : Prop := circle_eq p.fst p.snd ∧ (p.fst^2 + p.snd^2 < 8)
def is_chord (A B P0 : ℝ × ℝ) (α : ℝ) : Prop := 
  let line_through_P0 := α
  -- Aligning with your input, but actual condition for line definition might slightly change
  in α = 3 * real.pi / 4 ∨ (A = P0 ∨ B = P0) -- Dummy condition, may differ in actual scenario

-- Theorems to be proved
theorem find_AB_length (α : ℝ) (A B : ℝ × ℝ) :
  α = 3 * real.pi / 4 → is_chord A B P0 α → is_point_inside_circle P0 → 
  ∃ AB : ℝ, AB = real.sqrt 30 := 
sorry

theorem find_AB_equation (A B : ℝ × ℝ) :
  (A + B) / 2 = (P0.fst, P0.snd) → is_point_inside_circle P0 →
  ∃ f : ℝ → ℝ, ∀ x, f(x) = 2 * x + 5 :=
sorry

end find_AB_length_find_AB_equation_l306_306521


namespace red_numbers_le_totient_l306_306927

variable (n : ℕ)
variable (is_red : ℕ → Prop)

-- Conditions
hypothesis (h1 : ∀ a, is_red a → a ≠ 1 → ∀ k, 1 ≤ k → k * a ≤ n → is_red (k * a))

-- Euler's totient function definition
noncomputable def euler_totient (n : ℕ) : ℕ := n * (∏ p in finset.prime_divisors n, 1 - 1 / p.to_nat)

-- Statement to prove
theorem red_numbers_le_totient (n : ℕ) (is_red : ℕ → Prop) :
  (∃ red_count ≤ euler_totient n, ∀ k, 1 ≤ k → k ≤ n → (is_red k → k ≠ 1 → ∀ m, 1 ≤ m → m * k ≤ n → is_red (m * k))) :=
sorry

end red_numbers_le_totient_l306_306927


namespace complement_union_l306_306677

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306677


namespace soldiers_same_schedule_l306_306881

-- Define the statement of the problem in Lean 4
theorem soldiers_same_schedule (soldiers : Fin 85 → ℕ × ℕ) (color : Fin 10 → Fin 85 → Prop) :
  (∃ i j : Fin 85, i ≠ j ∧ ∀ d : Fin 10, color d i ↔ color d j) :=
begin
  sorry
end

end soldiers_same_schedule_l306_306881


namespace minimum_positive_sum_example_l306_306149

noncomputable def min_positive_sum (n : ℕ) (a : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n, ∑ j in Finset.Icc (i + 1) (n - 1), a i * a j

theorem minimum_positive_sum_example :
  ∀ (a : Fin 150 → ℝ), (∀ i, a i = 1 ∨ a i = -1) →
  min_positive_sum 150 a = 53 :=
by
  intro a ha
  have S : ℝ := min_positive_sum 150 a
  have h_sum_eq_16 : (∑ i in Finset.range 150, a i) = 16 := sorry
  have h_sq_sum : (∑ i in Finset.range 150, (a i)^2) = 150 := sorry
  -- Goal is to prove that 2S = 256 - 150
  have h_2S_eq : 2 * S = (∑ i in Finset.range 150, a i)^2 - 150 := by
    calc 2 * S = (∑ i in Finset.range 150, ∑ j in Finset.Icc (i + 1) (150 - 1), a i * a j) : sorry
    ... = (∑ i in Finset.range 150, a i)^2 - (∑ i in Finset.range 150, (a i)^2) : sorry
    ... = 256 - 150 : by rw [h_sum_eq_16, h_sq_sum]
  have S_eq_53 : S = (256 - 150) / 2 := by
    rw [h_2S_eq]
    norm_num
  rw [S_eq_53]
  norm_num
  done

end minimum_positive_sum_example_l306_306149


namespace operating_system_overhead_cost_l306_306331

theorem operating_system_overhead_cost :
  ∃ (O : ℝ), 
    (0.023 * 1.5 * 1000 + 5.35 + O = 40.92) ∧ 
    O = 1.07 :=
by {
  use 1.07,
  split,
  sorry,
  refl,
}

end operating_system_overhead_cost_l306_306331


namespace complement_union_l306_306681

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306681


namespace garden_area_increase_l306_306950

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l306_306950


namespace equation_1_solve_equation_2_solve_l306_306317

-- The first equation
theorem equation_1_solve (x : ℝ) (h : 4 * (x - 2) = 2 * x) : x = 4 :=
by
  sorry

-- The second equation
theorem equation_2_solve (x : ℝ) (h : (x + 1) / 4 = 1 - (1 - x) / 3) : x = -5 :=
by
  sorry

end equation_1_solve_equation_2_solve_l306_306317


namespace cube_face_sum_equals_27point5_l306_306344

theorem cube_face_sum_equals_27point5 (numbers_at_vertices : Fin 8 → ℕ) (h₁ : ∀ i, 1 ≤ numbers_at_vertices i ∧ numbers_at_vertices i ≤ 10) 
  (h₂ : ∑ i, numbers_at_vertices i = 55) : 
  (∃ common_sum : ℚ, common_sum = 27.5 ∧ 
  ∀ faces : Fin 6 → Fin 4 → Fin 8, 
  (∀ f, (∑ j, numbers_at_vertices (faces f j) : ℚ) = common_sum)) :=
begin
  sorry
end

end cube_face_sum_equals_27point5_l306_306344


namespace pencil_length_l306_306082

theorem pencil_length
  (R P L : ℕ)
  (h1 : P = R + 3)
  (h2 : P = L - 2)
  (h3 : R + P + L = 29) :
  L = 12 :=
by
  sorry

end pencil_length_l306_306082


namespace line_through_A_B_l306_306200

open Real

theorem line_through_A_B (a : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  let A := (2, -1)
  let B := (1, 1)
  let f := λ x, log a (x - 1) - 1
  let line_eq := λ x y, (m + 1) * x + (m - 1) * y - 2 * m = 0
  (2 = ↑A.1) ∧ (-1 = ↑A.2) → 
  (1 = ↑B.1) ∧ (1 = ↑B.2) →
  ((f 2 = -1) ∧ (line_eq 1 1)) → 
  ∀ x y, (y - 1 = -2 * (x - 1)) ↔ (2 * x + y - 3 = 0) :=
by
  intros
  sorry

end line_through_A_B_l306_306200


namespace smallest_positive_integer_l306_306373

theorem smallest_positive_integer (N : ℕ) :
  (N % 2 = 1) ∧
  (N % 3 = 2) ∧
  (N % 4 = 3) ∧
  (N % 5 = 4) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ↔ 
  N = 2519 := by {
  sorry
}

end smallest_positive_integer_l306_306373


namespace geometric_sequence_S4_l306_306261

/-
In the geometric sequence {a_n}, S_2 = 7, S_6 = 91. Prove that S_4 = 28.
-/

theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 7) 
  (h_S6 : S 6 = 91) :
  S 4 = 28 := 
sorry

end geometric_sequence_S4_l306_306261


namespace complement_union_l306_306660

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306660


namespace matrix_vector_subtraction_l306_306279

variable {R : Type} [Field R] {n m : Type} [Fintype n] [Fintype m] [DecidableEq n] [DecidableEq m]

variable (M : Matrix n m R)
variable (v w : Matrix m (Fin 1) R)

theorem matrix_vector_subtraction (h1 : M.mulVec v = ![5, 1]) : M.mulVec w = ![0, 4] → 
  M.mulVec (v - (2 : R) • w) = ![5, -7] :=
by
  intro h2
  rw [Matrix.mulVec, Matrix.mulVec]
  rw [h1, h2]
  simp
  sorry

end matrix_vector_subtraction_l306_306279


namespace range_of_f_max_omega_increasing_f_l306_306546

noncomputable def f (ω x : ℝ) : ℝ :=
  4 * cos (ω * x - π / 6) * sin (π - ω * x) - sin (2 * ω * x - π / 2)

theorem range_of_f (ω : ℝ) (h : ω > 0) : 
  (set.range (f ω)) = set.Icc (1 - real.sqrt 3) (1 + real.sqrt 3) := 
sorry

theorem max_omega_increasing_f : 
  ∃ ω : ℝ, (0 < ω ∧ ω ≤ 1 / 6) ∧ 
  ∀ x ∈ set.Icc (-3 * π / 2) (π / 2), 
  ∃ δ > 0, ∀ h, (h > 0 ∧ h < δ → f ω (x + h) > f ω x) :=
sorry

end range_of_f_max_omega_increasing_f_l306_306546


namespace find_f_of_pi_l306_306552

def f (x : ℝ) : ℝ := (Real.sin (x / 2))^2

theorem find_f_of_pi :
  f (9 * Real.pi / 4) = 1 / 2 - Real.sqrt 2 / 4 := by
  sorry

end find_f_of_pi_l306_306552


namespace eval_expression_l306_306157

def base8_to_base10 (n : Nat) : Nat :=
  2 * 8^2 + 4 * 8^1 + 5 * 8^0

def base4_to_base10 (n : Nat) : Nat :=
  1 * 4^1 + 5 * 4^0

def base5_to_base10 (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 2 * 5^0

def base6_to_base10 (n : Nat) : Nat :=
  3 * 6^1 + 2 * 6^0

theorem eval_expression : 
  base8_to_base10 245 / base4_to_base10 15 - base5_to_base10 232 / base6_to_base10 32 = 15 :=
by sorry

end eval_expression_l306_306157


namespace price_of_turban_l306_306221

theorem price_of_turban (T : ℝ) (h1 : 90 + T = S) (h2 : (3/4) * S = 60 + T) : T = 30 :=
by
  have h3 : S = 90 + T, from h1
  rw [h3] at h2
  sorry

end price_of_turban_l306_306221


namespace part_a_part_b_l306_306802

noncomputable def f (g n : ℕ) : ℕ := g^n + 1

theorem part_a (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → f g n ∣ f g (3*n) ∧ f g n ∣ f g (5*n) ∧ f g n ∣ f g (7*n) :=
sorry

theorem part_b (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → ∀ k : ℕ, 1 ≤ k → gcd (f g n) (f g (2*k*n)) = 1 :=
sorry

end part_a_part_b_l306_306802


namespace complement_union_l306_306624

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306624


namespace work_time_l306_306172

-- Definitions and conditions
variables (A B C D h : ℝ)
variable (h_def : ℝ := 1 / (1 / A + 1 / B + 1 / D))

-- Conditions
axiom cond1 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (A - 8)
axiom cond2 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (B - 2)
axiom cond3 : 1 / A + 1 / B + 1 / C + 1 / D = 3 / C
axiom cond4 : 1 / A + 1 / B + 1 / D = 2 / C

-- The statement to prove
theorem work_time : h_def = 16 / 11 := by
  sorry

end work_time_l306_306172


namespace perfect_square_of_c_perfect_square_c_l306_306377

def prime_factorization_4 : ℕ := 2^2
def prime_factorization_5 : ℕ := 5^1
def prime_factorization_6 : ℕ := 2 * 3

theorem perfect_square_of_c :
  (4^5 * 5^4 * 6^6) = (2^(2*5) * 5^(1*4) * (2 * 3)^6) :=
by
  -- Expand the prime factorizations
  have h1 : 4^5 = (2^2)^5 := by sorry
  have h2 : 5^4 = (5^1)^4 := by sorry
  have h3 : 6^6 = (2 * 3)^6 := by sorry
  -- Combine and prove equality
  sorry

theorem perfect_square_c :
  ∀ (n : ℕ),
    (n = 4^5 * 5^4 * 6^6 →
    ∃ (m : ℕ), n = m^2) :=
by
  intros n h
  rw h
  use 2 * 5^2 * (2 * 3)^3
  exact perfect_square_of_c

end perfect_square_of_c_perfect_square_c_l306_306377


namespace min_value_f1_range_of_a_l306_306547

-- Define the function f(x) for problem (1)
def f1 (x : ℝ) : ℝ := (x^2 + 2*x + 1/2) / x
-- State the minimum value of f1(x) on [1, +∞) is 7/2
theorem min_value_f1 : (∀ x : ℝ, 1 ≤ x → f1 x ≥ 7/2) ∧ (f1 1 = 7/2) :=
sorry

-- Define the function f(x) for problem (2)
def f2 (x a : ℝ) : ℝ := (x^2 + 2*x + a) / x
-- State the range of a for f(x) being always positive on [1, +∞)
theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → f2 x a > 0) ↔ a > -3 :=
sorry

end min_value_f1_range_of_a_l306_306547


namespace homologous_chromosomes_correct_l306_306379

def homologous_chromosomes_property (A B C D : Prop) : Prop :=
  -- conditions representing the definitions of homologous chromosomes and the descriptions
  (A ↔ (∀ (chr1 chr2 : Chromosome), (is_result_of_replication chr1 chr2) → is_sister_chromatid chr1 chr2)) ∧
  (B ↔ (∀ (chr1 chr2 : Chromosome), (is_paternal_maternal_pair chr1 chr2) → homologous_check_incomplete chr1 chr2)) ∧
  (C ↔ (∀ (chr1 chr2 : Chromosome), (synapse_during_meiosis chr1 chr2) → is_homologous chr1 chr2)) ∧
  (D ↔ (∀ (chr1 chr2 : Chromosome), (similar_shape_size chr1 chr2) → homologous_check_partial chr1 chr2))

theorem homologous_chromosomes_correct (A B C D : Prop) : 
  homologous_chromosomes_property A B C D → C :=
sorry

end homologous_chromosomes_correct_l306_306379


namespace candy_count_l306_306359

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end candy_count_l306_306359


namespace simplify_polynomial_l306_306851

variable (r : ℝ)

theorem simplify_polynomial : (2 * r^2 + 5 * r - 7) - (r^2 + 9 * r - 3) = r^2 - 4 * r - 4 := by
  sorry

end simplify_polynomial_l306_306851


namespace complement_union_eq_l306_306581

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306581


namespace number_of_lines_l306_306365

-- Definitions and conditions based on the problem statement
def point (α : Type) := α × α

variable (α : Type) [Real α]

def distance (p1 p2 : point α) : α := Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def circle (center : point α) (radius : α) : set (point α) :=
  { p | distance α p center = radius }

variable (A B : point α)

axiom distance_AB : distance α A B = 8

-- Main theorem to prove the number of lines
theorem number_of_lines (r1 r2 : α) (h1 : r1 = 5) (h2 : r2 = 3) : 
  ∃ n, n = 3 :=
by {
  sorry
}

end number_of_lines_l306_306365


namespace complement_union_l306_306613

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306613


namespace sum_seven_terms_l306_306527

-- Define the arithmetic sequence and sum of first n terms
variable {a : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S : ℕ → ℝ} -- The sum of the first n terms S_n

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition: a_4 = 4
def a_4_eq_4 (a : ℕ → ℝ) : Prop :=
  a 4 = 4

-- Proposition we want to prove: S_7 = 28 given a_4 = 4
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (ha : is_arithmetic_sequence a)
  (hS : sum_of_arithmetic_sequence a S)
  (h : a_4_eq_4 a) : 
  S 7 = 28 := 
sorry

end sum_seven_terms_l306_306527


namespace saltwater_concentration_l306_306072

theorem saltwater_concentration (m w : ℝ)
  (h1 : m + w = 200)
  (h2 : (m + 25) / (m + w + 25) = 2 * m / (m + w)) :
  (m / (m + w)) * 100 = 10 := 
begin
  sorry
end

end saltwater_concentration_l306_306072


namespace sum_binary_digit_expression_l306_306169

def s (n : ℕ) : ℕ :=
  n.binaryDigits.sum

theorem sum_binary_digit_expression (s : ℕ → ℕ) (n k : ℕ) (h1 : ∀ n, s n = n.binaryDigits.sum) (h2 : k = 2^2022 - 1) :
  (∑ n in finset.range k, (-1)^(s n) / (n + 2022)) > 0 :=
by
  sorry

end sum_binary_digit_expression_l306_306169


namespace probability_three_one_painted_faces_l306_306959

/-
  A cube with 5 units on each side is composed of 125 unit cubes.
  Three faces of the larger cube that meet at one corner are painted red.
  The cube is disassembled into 125 unit cubes.
  Two unit cubes are selected uniformly at random.
-/
def total_unit_cubes : ℕ := 125

def three_painted_faces : ℕ := 8

def two_painted_faces: ℕ := 18

def one_painted_face : ℕ := 27

def total_ways_select_two_cubes : ℕ := nat.choose total_unit_cubes 2

def successful_outcomes : ℕ := three_painted_faces * one_painted_face

def probability : ℚ := (successful_outcomes : ℚ) / (total_ways_select_two_cubes : ℚ)

/-- The probability that one of the two selected unit cubes will have exactly
  three painted faces while the other cube has exactly one painted face is 216/7750.
-/
theorem probability_three_one_painted_faces :
  probability = 216 / 7750 := 
sorry

end probability_three_one_painted_faces_l306_306959


namespace find_values_of_k_l306_306758

noncomputable def complex_distance (z w : ℂ) : ℝ := complex.abs (z - w)

theorem find_values_of_k (k : ℝ) :
  (∀ z : ℂ, (complex_distance z 2 = 3 * complex_distance z (-2)) ↔ (complex.abs z = k)) ->
  k = 1.5 ∨ k = 4.5 ∨ k = 5.5 :=
by sorry

end find_values_of_k_l306_306758


namespace degree_not_determined_from_characteristic_l306_306788

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l306_306788


namespace T_n_sum_absolute_values_l306_306257

theorem T_n_sum_absolute_values (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) :
  (∀ n, a n = a1 + n * d) →
  (a 15 + a 16 + a 17 = -36) →
  ∀ n, T_n n = if n ≤ 21 then - (3 / 2) * n^2 + (123 / 2) * n else (3 / 2) * n^2 - (123 / 2) * n + 1260 := 
sorry

end T_n_sum_absolute_values_l306_306257


namespace volume_ratio_divided_tetrahedron_l306_306969

-- Define the given conditions
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Defining the medians for the faces ABC, ACD, and ABD
def median := ... -- Placeholder for median definitions
def plane_divides (A: Type) (ratio1 ratio2 ratio3 : ℚ) : Prop := 
  median ABC == 1/3 ∧ median ACD == 1/2 ∧ median ABD == 1/3

-- The proof problem statement
theorem volume_ratio_divided_tetrahedron (h : plane_divides A 1 1 1) : 
  ratio_volumes (tetrahedron A B C D) = (1 : ℚ) / 15 := 
sorry

end volume_ratio_divided_tetrahedron_l306_306969


namespace area_computation_l306_306060

-- Definitions from the problem conditions
variable (AB CD : ℝ) (E F : ℝ)
variable (BE CF : ℝ) (theta : ℝ)
variable (sin_theta : ℝ)

-- Given conditions
axiom AB_CD_rectangular : AB * CD = (AB * CD) -- ABCD is a rectangle
axiom BE_CF_points : BE < CF -- BE < CF
axiom AB_E_angle : angle AB' C' = angle B' E A -- ∠AB'C' ≅ ∠B'EA
axiom AB_BE_values : AB' = 7 ∧ BE = 17 -- AB' = 7 and BE = 17

-- Trigonometric identity and calculations from the proof steps
axiom sin_theta_value: sin(theta) = 1 / 17

-- Final result (answer): the area in the desired form
def area_ABCD (A B C : ℝ) : ℝ :=
  (1372 + 833 * sqrt 2) / 6

-- The proof statement
theorem area_computation (AB CD BE CF : ℝ) (theta : ℝ) (sin_theta : ℝ) :
  AB_CD_rectangular →
  BE_CF_points →
  AB_E_angle →
  AB_BE_values →
  sin_theta_value →
  area_ABCD AB CD BE CF theta sin_theta = (1372 + 833 * sqrt 2) / 6 :=
sorry

end area_computation_l306_306060


namespace largest_divisor_36_l306_306177

open Nat

noncomputable def f (n : ℕ) : ℕ := (2 * n + 7) * 3^(n + 9)

theorem largest_divisor_36 (m : ℕ) : (∀ n : ℕ, 0 < n → m ∣ f n) ↔ m = 36 :=
by
  -- Left part: assume hypothesis and prove m = 36 (proof omitted)
  -- Right part: assume m = 36 and prove hypothesis (proof omitted)
  sorry

end largest_divisor_36_l306_306177


namespace num_diagonals_increase_by_n_l306_306413

-- Definitions of the conditions
def num_diagonals (n : ℕ) : ℕ := sorry  -- Consider f(n) to be a function that calculates diagonals for n-sided polygon

-- Lean 4 proof problem statement
theorem num_diagonals_increase_by_n (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n :=
sorry

end num_diagonals_increase_by_n_l306_306413


namespace cameron_greater_probability_l306_306442

-- Define the conditions based on the cubes
def cameron_cube : list ℕ := [6, 6, 6, 6, 6, 6]
def dean_cube : list ℕ := [1, 1, 2, 2, 3, 3]
def olivia_cube : list ℕ := [3, 3, 3, 3, 6, 6]

-- Define a function to calculate the favorable probability
def probability_less_than (values : list ℕ) (n : ℕ) := 
  (values.filter (λ x, x < n)).length / values.length.to_real

-- Calculate probabilities
def dean_probability := probability_less_than dean_cube 6
def olivia_probability := probability_less_than olivia_cube 6

-- The final statement to prove the correct probability
theorem cameron_greater_probability : 
  (dean_probability * olivia_probability) = 2 / 3 :=
by
  have dean_prob : dean_probability = 1 := by sorry
  have olivia_prob : olivia_probability = 2 / 3 := by sorry
  rw [dean_prob, olivia_prob]
  norm_num -- simplifies the expression

end cameron_greater_probability_l306_306442


namespace complement_union_l306_306680

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306680


namespace binomial_sixteen_twelve_eq_l306_306124

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l306_306124


namespace triangle_cosine_condition_l306_306183

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Definitions according to the problem conditions
def law_of_sines (a b : ℝ) (A B : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B

theorem triangle_cosine_condition (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : law_of_sines a b A B)
  (h1 : a > b) : Real.cos (2 * A) < Real.cos (2 * B) ↔ a > b :=
by
  sorry

end triangle_cosine_condition_l306_306183


namespace boat_speed_in_still_water_l306_306876

-- Define the conditions
def current_rate : ℝ := 3
def distance_downstream : ℝ := 7.2
def time_downstream_hours : ℝ := 24 / 60 -- Convert 24 minutes to hours
def downstream_speed (x : ℝ) : ℝ := x + current_rate

-- The main theorem to prove
theorem boat_speed_in_still_water (x : ℝ) :
  distance_downstream = downstream_speed x * time_downstream_hours → x = 15 :=
by
  sorry -- Proof is not required

end boat_speed_in_still_water_l306_306876


namespace square_of_binomial_is_25_l306_306718

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l306_306718


namespace decimal_to_base_5_digits_l306_306262

theorem decimal_to_base_5_digits (n : ℕ) (h : n = 124) : 
  let base_5_digits := 4 * 5^2 + 4 * 5^1 + 4 * 5^0 in
  base_5_digits = 124 ∧
  (base_5_digits = 4 * 25 + 4 * 5 + 4) ∧
  true := -- just to make the logic connect correctly to the original problem
begin
  sorry
end

end decimal_to_base_5_digits_l306_306262


namespace apples_left_l306_306270

theorem apples_left (initial_apples : Nat)
                    (sold_to_jill_percent : Nat)
                    (sold_to_june_percent : Nat)
                    (sold_to_judy_percent : Nat)
                    (given_to_teacher : Nat) :
  initial_apples = 150 →
  sold_to_jill_percent = 30 →
  sold_to_june_percent = 20 →
  sold_to_judy_percent = 10 →
  given_to_teacher = 1 →
  let apples_after_jill := initial_apples - (initial_apples * sold_to_jill_percent / 100)
  let apples_after_june := apples_after_jill - (apples_after_jill * sold_to_june_percent / 100)
  let apples_after_judy := apples_after_june - (apples_after_june * sold_to_judy_percent / 100)
  let apples_final := apples_after_judy - given_to_teacher in
  apples_final = 75 :=
by
  intros h1 h2 h3 h4 h5
  let apples_after_jill := 150 - (150 * 30 / 100)
  let apples_after_june := apples_after_jill - (apples_after_jill * 20 / 100)
  let apples_after_judy := apples_after_june - (apples_after_june * 10 / 100)
  let apples_final := apples_after_judy - 1
  show apples_final = 75
  sorry

end apples_left_l306_306270


namespace complement_union_l306_306599

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306599


namespace total_salary_under_unkind_manager_l306_306745

noncomputable theory
open_locale classical

-- Definitions and conditions
variables (x y n S_1 S_2 : ℕ) (S : ℕ)
variables (h1 : S = 10000) (h2 : x + y = n) (h3 : S_1 + S_2 = 10000)
variables (h4 : 3 * S_1 + S_2 + 1000 * y = 24000)

-- Theorem: Prove the total salary under the unkind manager's proposal
theorem total_salary_under_unkind_manager : 
  S_1 + 500 * y = 7000 :=
sorry

end total_salary_under_unkind_manager_l306_306745


namespace adam_tshirts_count_l306_306984

-- Given conditions
variable (T : ℕ)
variable (pants jumpers pajamas half_clothes : ℕ)
variable (friends_clothes : ℕ)
variable (total_donated : ℕ)

-- Initial counts
def pants := 4
def jumpers := 4
def pajamas := 8
def initial_clothes := pants + jumpers + pajamas + T

-- Friends' contributions
def friends_clothes := 3 * initial_clothes

-- Adam keeps half of his clothes
def half_clothes := initial_clothes / 2

-- Total donated
def total_donated := half_clothes + friends_clothes

-- The proof statement
theorem adam_tshirts_count : total_donated = 126 -> T = 20 :=
by
  sorry

end adam_tshirts_count_l306_306984


namespace radical_center_fixed_circle_l306_306982

-- Define the given elements and conditions
variables {ABC : Type} [triangle ABC]
variables {I O S N M : point}
variables {r R : ℝ}

-- Given conditions
axiom incircle_fixed (h1 : incircle ABC = (I, r))
axiom circumcircle_fixed (h2 : circumcircle ABC = (O, R))
axiom spieker_point_radical_center (h3 : spieker_point ABC = S)
axiom nagel_point_definition (h4 : nagel_point ABC = N)
axiom nagel_incenter_reflection (h5 : reflection_point S I = N)
axiom distance_ON (h6 : distance O N = R - 2 * r)

-- Problem statement
theorem radical_center_fixed_circle :
  ∃ (M : point) (k : ℝ), midpoint I O = M ∧ distance S M = k ∧ k = (1 / 2) * R - r :=
sorry

end radical_center_fixed_circle_l306_306982


namespace complement_union_eq_singleton_five_l306_306575

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306575


namespace jellybeans_original_count_l306_306800

noncomputable def original_jellybeans (remaining_jellybeans : ℕ) : ℕ :=
  remaining_jellybeans / (0.75 ^ 3)

theorem jellybeans_original_count (remaining_jellybeans : ℕ) :
  remaining_jellybeans = 27 → original_jellybeans remaining_jellybeans = 64 :=
by
  intro h
  rw [h]
  norm_num
  sorry

end jellybeans_original_count_l306_306800


namespace min_triangles_groups_four_color_possible_l306_306525

noncomputable def min_num_triangles (n m : ℕ) : ℕ :=
∑ i in (range m).attach, (choose (1994 // m + if i < 1994 % m then 1 else 0) 3)

theorem min_triangles_groups :
  min_num_triangles 1994 83 = 168544 := sorry

theorem four_color_possible (G : graph) (hG : ∃ k, num_triangles G k ∧ k = 168544) :
  ∃ f : G.edge_set → fin 4, ∀ t : triangle, ∃ c₁ c₂ c₃ : fin 4, c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ := sorry

end min_triangles_groups_four_color_possible_l306_306525


namespace int_pairs_satisfy_eq_l306_306706

theorem int_pairs_satisfy_eq :
  {p : ℤ × ℤ // let x := p.1; let y := p.2 in x^4 + y^2 = 4 * y}.to_finset.card = 2 :=
by
  sorry

end int_pairs_satisfy_eq_l306_306706


namespace length_of_room_is_21_l306_306865

variable (L : ℕ) -- The length of the room
variable (W : ℕ) -- The width of the room (given as 12)
variable (Vw : ℕ) -- The width of the veranda (given as 2)
variable (Va : ℕ) -- The area of the veranda (given as 148)

-- Given conditions
axiom width_of_room : W = 12
axiom width_of_veranda : Vw = 2
axiom area_of_veranda : Va = 148

-- Proof problem statement: Prove that the length of the room is 21 meters.
theorem length_of_room_is_21 : L = 21 := by {
  sorry,
}

end length_of_room_is_21_l306_306865


namespace complement_union_l306_306654

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306654


namespace product_of_two_numbers_l306_306004

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 7) :
  x * y = 144 := 
by
  sorry

end product_of_two_numbers_l306_306004


namespace spinner_prob_multiple_of_3_l306_306021

theorem spinner_prob_multiple_of_3 :
  let spinner_C := {2, 5, 7, 6}
  let spinner_D := {11, 3, 6}
  (∃ (count_successful : ℕ), count_successful = 
    (∑ c in spinner_C, ∑ d in spinner_D, if (c * d) % 3 = 0 then 1 else 0))
  →  (count_successful : ℕ) / (4 * 3) = (9 / 12) := by
  sorry

end spinner_prob_multiple_of_3_l306_306021


namespace count_triangles_l306_306751

theorem count_triangles :
  let points := 12
  let collinear_points := 4
  let non_collinear_points := points - collinear_points
  (coll_tris := ((collinear_points * (collinear_points - 1)) / 2) * non_collinear_points) 
  (one_coll_tris := collinear_points * ((non_collinear_points * (non_collinear_points - 1)) / 2))
  (no_coll_tris := (non_collinear_points * (non_collinear_points - 1) * (non_collinear_points - 2)) / 6)
  coll_tris + one_coll_tris + no_coll_tris = 216 := 
by
  unfold points collinear_points non_collinear_points coll_tris one_coll_tris no_coll_tris
  norm_num
  sorry

end count_triangles_l306_306751


namespace find_I_value_l306_306764

-- Definitions based on the conditions
def E : ℕ := 4
def G := ∀ (x : ℕ), x ∈ (Set.toFinset {1, 3, 5, 7, 9}) → Prop
def validI (i : ℕ) := 2

-- Main theorem statement
theorem find_I_value (I : ℕ) (E_condition : E = 4) (G_condition : G 3) : validI I = 2 := 
sorry

end find_I_value_l306_306764


namespace average_cost_per_hour_for_12_hours_l306_306330

/-- Given the cost structure of a parking garage, we want to prove that the
average cost per hour to park a car for 12 hours is $2.77. --/
theorem average_cost_per_hour_for_12_hours :
  let first_2_hours_cost := 10.00 in
  let next_3_hours_cost := 3 * 1.75 in
  let next_3_hours_cost_after_5 := 3 * 2.00 in
  let next_4_hours_cost_after_8 := 4 * 3.00 in
  let total_cost := first_2_hours_cost + next_3_hours_cost + next_3_hours_cost_after_5 + next_4_hours_cost_after_8 in
  total_cost / 12 = 2.77 :=
by
  sorry

end average_cost_per_hour_for_12_hours_l306_306330


namespace average_visitors_per_day_l306_306962

theorem average_visitors_per_day:
  (∃ (Sundays OtherDays: ℕ) (visitors_per_sunday visitors_per_other_day: ℕ),
    Sundays = 4 ∧
    OtherDays = 26 ∧
    visitors_per_sunday = 600 ∧
    visitors_per_other_day = 240 ∧
    (Sundays + OtherDays = 30) ∧
    (Sundays * visitors_per_sunday + OtherDays * visitors_per_other_day) / 30 = 288) :=
sorry

end average_visitors_per_day_l306_306962


namespace total_kayaks_built_l306_306107

/-- Geometric sequence sum definition -/
def geom_sum (a r : ℕ) (n : ℕ) : ℕ :=
  if r = 1 then n * a
  else a * (r ^ n - 1) / (r - 1)

/-- Problem statement: Prove that the total number of kayaks built by the end of June is 726 -/
theorem total_kayaks_built : geom_sum 6 3 5 = 726 :=
  sorry

end total_kayaks_built_l306_306107


namespace max_a_minus_b_l306_306236

variable (z : ℂ)
def a := (z^2 - (conj z)^2) / (2 * complex.I)
def b := z * conj z

theorem max_a_minus_b : ∃ z : ℂ, a z - b z ≤ 0 ∧ (∀ (w : ℂ), a w - b w ≤ 0) := by
  sorry

end max_a_minus_b_l306_306236


namespace box_inscribed_in_sphere_radius_correct_l306_306974

noncomputable theory
open Real

def box_inscribed_in_sphere_radius (a b c r : ℝ) : Prop :=
  (a + b + c = 40) ∧ 
  (2 * a * b + 2 * b * c + 2 * c * a = 600) ∧ 
  (2 * r = sqrt (40^2 - 2 * 300))

theorem box_inscribed_in_sphere_radius_correct :
  ∃ (r : ℝ), ∀ (a b c : ℝ), box_inscribed_in_sphere_radius a b c r → r = 5 * sqrt 10 :=
begin
  sorry
end

end box_inscribed_in_sphere_radius_correct_l306_306974


namespace example_theorem_l306_306665

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306665


namespace order_y1_y2_y3_l306_306213

-- Defining the parabolic function and the points A, B, C
def parabola (a x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

-- Points A, B, C
def y1 (a : ℝ) : ℝ := parabola a (-1)
def y2 (a : ℝ) : ℝ := parabola a 2
def y3 (a : ℝ) : ℝ := parabola a 4

-- Assumption: a > 0
variables (a : ℝ) (h : a > 0)

-- The theorem to prove
theorem order_y1_y2_y3 : 
  y2 a < y1 a ∧ y1 a < y3 a :=
sorry

end order_y1_y2_y3_l306_306213


namespace product_of_two_numbers_l306_306005

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 7) :
  x * y = 144 := 
by
  sorry

end product_of_two_numbers_l306_306005


namespace complement_of_union_is_singleton_five_l306_306639

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306639


namespace degree_not_determined_by_A_P_l306_306792

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l306_306792


namespace value_of_M_l306_306712

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end value_of_M_l306_306712


namespace parallel_vector_line_m_l306_306558

def line (m : ℝ) := λ x y : ℝ, m * x + 2 * y + 6 = 0

def slope_vector (m : ℝ) := (1 - m, 1)

theorem parallel_vector_line_m (m : ℝ) :
  let slope_line := -m / 2
  let slope_vec := 1 / (1 - m)
  (slope_vec = slope_line) → (m = -1 ∨ m = 2) :=
by
  intros slope_line slope_vec h
  sorry

end parallel_vector_line_m_l306_306558


namespace bill_left_with_money_l306_306996

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end bill_left_with_money_l306_306996


namespace price_and_max_units_proof_l306_306473

/-- 
Given the conditions of purchasing epidemic prevention supplies: 
- 60 units of type A and 45 units of type B costing 1140 yuan
- 45 units of type A and 30 units of type B costing 840 yuan
- A total of 600 units with a cost not exceeding 8000 yuan

Prove:
1. The price of each unit of type A is 16 yuan, and type B is 4 yuan.
2. The maximum number of units of type A that can be purchased is 466.
--/
theorem price_and_max_units_proof 
  (x y : ℕ) 
  (m : ℕ)
  (h1 : 60 * x + 45 * y = 1140) 
  (h2 : 45 * x + 30 * y = 840) 
  (h3 : 16 * m + 4 * (600 - m) ≤ 8000) 
  (h4 : m ≤ 600) :
  x = 16 ∧ y = 4 ∧ m = 466 := 
by 
  sorry

end price_and_max_units_proof_l306_306473


namespace degree_not_determined_by_A_P_l306_306776

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l306_306776


namespace complement_union_l306_306676

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306676


namespace regular_polygon_property_l306_306307

variables {n : ℕ}
variables {r : ℝ} -- r is the radius of the circumscribed circle
variables {t_2n : ℝ} -- t_2n is the area of the 2n-gon
variables {k_n : ℝ} -- k_n is the perimeter of the n-gon

theorem regular_polygon_property
  (h1 : t_2n = (n * k_n * r) / 2)
  (h2 : k_n = n * a_n) :
  (t_2n / r^2) = (k_n / (2 * r)) :=
by sorry

end regular_polygon_property_l306_306307


namespace prime_lonely_infinitely_many_non_lonely_l306_306972

-- Definition of a lonely number
def lonely (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n → (∑ d in (Nat.divisors n), (1 : ℚ) / d) ≠ (∑ d in (Nat.divisors m), (1 : ℚ) / d)

-- Part a: Show that any prime number is lonely
theorem prime_lonely (p : ℕ) (hp : p.Prime) : lonely p := sorry

-- Part b: Prove that there are infinitely many numbers that are not lonely
theorem infinitely_many_non_lonely : ∃ᶠ n in (Filter.atTop : Filter ℕ), ¬ lonely n := sorry

end prime_lonely_infinitely_many_non_lonely_l306_306972


namespace divisors_larger_than_8_factorial_l306_306223

theorem divisors_larger_than_8_factorial (divisor_g9 : ℕ → Prop) :
  (∀ d, d ∣ fact 9 → d > fact 8 → divisor_g9 d) →
  (∀ d₁ d₂, divisor_g9 d₁ → divisor_g9 d₂ → d₁ = d₂ → false) →
  ∑ d in {d | d ∣ fact 9 ∧ d > fact 8}.toFinset, 1 = 8 :=
by
  sorry

end divisors_larger_than_8_factorial_l306_306223


namespace train_length_in_meters_l306_306089

-- We define the given conditions in Lean
def speed_km_hr : ℝ := 40
def time_seconds : ℝ := 17.1

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Converted speed in m/s
def speed_m_s : ℝ := speed_km_hr * conversion_factor

-- Length of the train
def train_length : ℝ := speed_m_s * time_seconds

-- Theorem stating the length of the train
theorem train_length_in_meters : train_length ≈ 190 := by
  sorry

end train_length_in_meters_l306_306089


namespace garden_area_increase_l306_306934

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l306_306934


namespace probability_prime_multiple_of_11_l306_306316

/-- Given 60 cards numbered from 1 to 60, we are to find the probability of selecting a card where 
    the number on the card is both prime and a multiple of 11. -/
theorem probability_prime_multiple_of_11 : 
  let n := 60 in
  let cards := finset.range (n + 1) in
  let prime_multiple_of_11 := {k ∈ cards | nat.prime k ∧ (11 ∣ k)} in
  (prime_multiple_of_11.card : ℚ) / n = 1 / 60 :=
by
  sorry

end probability_prime_multiple_of_11_l306_306316


namespace minimum_distance_intersection_l306_306867

theorem minimum_distance_intersection : 
  ∀ (a : ℝ), 
  let x2 := (λ a : ℝ, classical.some (real.exists_log (a - 1))) in
  let x1 := 1 / 2 * (x2 a + real.log (x2 a)) - 1 in
  let |AB| := x2 a - x1 in
  (∃ a, (|AB| = real.min ((1 / 2) * a - real.log a + 1))) → (|AB| = 3 / 2) :=
sorry

end minimum_distance_intersection_l306_306867


namespace piggy_bank_dimes_l306_306968

theorem piggy_bank_dimes (q d : ℕ) 
  (h1 : q + d = 100) 
  (h2 : 25 * q + 10 * d = 1975) : 
  d = 35 :=
by
  -- skipping the proof
  sorry

end piggy_bank_dimes_l306_306968


namespace S_sum_l306_306285

def S_n (n : ℕ) : ℤ :=
  -- defining S_n based on the condition given
  if even n then -n / 2
  else (n + 1) / 2

theorem S_sum : S_n 17 + S_n 33 + S_n 50 = 1 := 
by 
  -- include the proof body here
  sorry

end S_sum_l306_306285


namespace focal_distance_of_conic_l306_306230

theorem focal_distance_of_conic (m : ℝ) (hm : m = sqrt (2 * 8)) :
  (m = -4) → (∀ x y : ℝ, x^2 - (y^2 / 4) = 1 → 2 * sqrt (1 + 4) = 2 * sqrt 5) :=
by
  intro h
  sorry

end focal_distance_of_conic_l306_306230


namespace unique_solution_m_n_l306_306504

theorem unique_solution_m_n (m n : ℕ) (h1 : m > 1) (h2 : (n - 1) % (m - 1) = 0) 
  (h3 : ¬ ∃ k : ℕ, n = m ^ k) :
  ∃! (a b c : ℕ), a + m * b = n ∧ a + b = m * c := 
sorry

end unique_solution_m_n_l306_306504


namespace find_a_even_function_l306_306540

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f x = (x + 1) * (x + a))  
  (h2 : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_even_function_l306_306540


namespace monotonic_intervals_when_a_eq_2_extreme_value_point_in_interval_l306_306554

-- Definition of the function f and its derivative
def f (a x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1
def f' (a x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

-- Problem Statement: Part I
theorem monotonic_intervals_when_a_eq_2 :
  ∀ x : ℝ, (x < 2 - real.sqrt 3 ∨ x > 2 + real.sqrt 3 → f' 2 x > 0) ∧ 
            (2 - real.sqrt 3 < x ∧ x < 2 + real.sqrt 3 → f' 2 x < 0) := 
by
  intro x
  sorry

-- Problem Statement: Part II
theorem extreme_value_point_in_interval :
  (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f' a x = 0) → (5 / 4 < a ∧ a < 5 / 3) := 
by
  intro h
  sorry

end monotonic_intervals_when_a_eq_2_extreme_value_point_in_interval_l306_306554


namespace PropositionA_PropositionB_PropositionC_PropositionD_l306_306040

-- Proposition A (Incorrect)
theorem PropositionA : ¬(∀ a b c : ℝ, a > b ∧ b > 0 → a * c^2 > b * c^2) :=
sorry

-- Proposition B (Correct)
theorem PropositionB : ∀ a b : ℝ, -2 < a ∧ a < 3 ∧ 1 < b ∧ b < 2 → -4 < a - b ∧ a - b < 2 :=
sorry

-- Proposition C (Correct)
theorem PropositionC : ∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2) :=
sorry

-- Proposition D (Incorrect)
theorem PropositionD : ¬(∀ a b c : ℝ, c > a ∧ a > b → a / (c - a) > b / (c - b)) :=
sorry

end PropositionA_PropositionB_PropositionC_PropositionD_l306_306040


namespace max_value_of_expression_l306_306557

variables (a x1 x2 : ℝ)

theorem max_value_of_expression :
  (x1 < 0) → (0 < x2) → (∀ x, x^2 - a * x + a - 2 > 0 ↔ (x < x1) ∨ (x > x2)) →
  (x1 * x2 = a - 2) → 
  x1 + x2 + 2 / x1 + 2 / x2 ≤ 0 :=
by
  intros h1 h2 h3 h4
  -- Proof goes here
  sorry

end max_value_of_expression_l306_306557


namespace limit_problem_l306_306056

noncomputable def limit_quot : ℝ :=
  real.log (1024 * 81)

theorem limit_problem :
  ∀ f : ℝ → ℝ, ∀ g : ℝ → ℝ,
  (∀ x, f x = 4^(5 * x) - 9^(-2 * x)) →
  (∀ x, g x = real.sin x - real.tan (x^3)) →
  filter.tendsto (λ x, (f x) / (g x)) filter.at_top (nhds limit_quot) :=
begin
  intros f g hf hg,
  sorry
end

end limit_problem_l306_306056


namespace complement_union_eq_singleton_five_l306_306574

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306574


namespace speeds_ratio_correct_l306_306894

structure Cyclists (distance : ℝ) (time_same_dir : ℝ) (time_towards : ℝ) :=
(v_A v_B : ℝ)

def speeds_ration (c : Cyclists 10 5 1) : ℝ :=
c.v_A / c.v_B

theorem speeds_ratio_correct (c : Cyclists 10 5 1) : speeds_ration c = 61 / 29 :=
by
  have eq1 : 10 = (c.v_B - c.v_A) * (5 - 0.5) := sorry
  have eq2 : 10 = (c.v_A + c.v_B) * 1 := sorry
  have vB_val : c.v_B = (10 + 10 / 4.5) / 2 := sorry
  have vA_val : c.v_A = 10 - vB_val := sorry
  have ratio : speeds_ration c = c.v_A / c.v_B := sorry
  show speeds_ration c = 61 / 29 from sorry

end speeds_ratio_correct_l306_306894


namespace degree_not_determined_by_A_P_l306_306797

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l306_306797


namespace integer_satisfies_mod_and_range_l306_306025

theorem integer_satisfies_mod_and_range :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-150 ≡ n [ZMOD 25]) → n = 0 :=
by
  sorry

end integer_satisfies_mod_and_range_l306_306025


namespace complement_union_l306_306655

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306655


namespace union_sets_l306_306809

open Set

variable (a b : ℝ)
def M := {a, b}
def N := {a + 1, 3}

theorem union_sets (h : M = {2}) : M ∪ N = {1, 2, 3} := by
  sorry

end union_sets_l306_306809


namespace garden_area_increase_l306_306942

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l306_306942


namespace angle_A_in_triangle_ABC_l306_306741

noncomputable def find_angle_A (a b : ℝ) (B : ℝ) := (asin (a * sin B / b) * 180 / Real.pi)

theorem angle_A_in_triangle_ABC (a b : ℝ) (B : ℝ) (h₁ : B = Real.pi / 4) (h₂ : b = Real.sqrt 2) (h₃ : a = 1) : 
  find_angle_A a b B = 30 := by
  sorry

end angle_A_in_triangle_ABC_l306_306741


namespace infinite_series_value_l306_306119

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end infinite_series_value_l306_306119


namespace sequence_divisibility_l306_306461

theorem sequence_divisibility (a : ℕ → ℕ)
  (h : ∀ n, 2 * n = ∑ d in divisors n, a d) :
  ∀ n, n ∣ a n :=
by {
  sorry
}

end sequence_divisibility_l306_306461


namespace bill_left_with_money_l306_306995

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end bill_left_with_money_l306_306995


namespace log_prop_example_l306_306110

theorem log_prop_example : 1g2 + (Real.sqrt 2 - 1)^0 + lg5 = 2 :=
by
  -- Proof goes here
  sorry

end log_prop_example_l306_306110


namespace g_xh_minus_g_x_l306_306725

theorem g_xh_minus_g_x (x h : ℝ) :
  let g := λ x, 3 * x^2 + 5 * Real.sin x - 4 * x in
  g (x + h) - g x = h * (6 * x + 3 * h + 10 * Real.cos (x + h/2) * Real.sin (h/2) - 4) :=
by
  let g := λ x:ℝ, 3 * x^2 + 5 * Real.sin x - 4 * x
  sorry

end g_xh_minus_g_x_l306_306725


namespace sqrt_equation_l306_306234

theorem sqrt_equation (n : ℕ) (h : sqrt (25 - sqrt n) = 3) : n = 256 := by
  sorry

end sqrt_equation_l306_306234


namespace polygonal_chain_length_l306_306767

theorem polygonal_chain_length (exists_chain : ∀ l < 2, ∃ chain, length(chain) = l ∧ (∀ line_parallel_to_side, (line_parallel_to_side ∩ chain).points ≤ 1)) (square_side_length : 1) :
  (∀ chain, (drawn_in_square chain square_side_length ∧ (∀ line_parallel_to_side, (line_parallel_to_side ∩ chain).points ≤ 1)) → length(chain) < 2) ∧ 
  (∀ l, l < 2 → ∃ chain, length(chain) = l ∧ (drawn_in_square chain square_side_length ∧ (∀ line_parallel_to_side, (line_parallel_to_side ∩ chain).points ≤ 1))) :=
by
  sorry

end polygonal_chain_length_l306_306767


namespace coefficient_x2_in_expansion_l306_306863

/- Given the polynomial expansion (1 + 2x)^5, prove that the coefficient of x^2 is 40 -/
theorem coefficient_x2_in_expansion : 
  (polynomial.coeff ((1 + 2 * polynomial.X) ^ 5) 2) = 40 := 
by
  sorry

end coefficient_x2_in_expansion_l306_306863


namespace angle_C_is_30_degrees_l306_306264

theorem angle_C_is_30_degrees
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (A_rad: 0 ≤ A ∧ A ≤ Real.pi)
  (B_rad: 0 ≤ B ∧ B ≤ Real.pi)
  (C_rad : 0 ≤ C ∧ C ≤ Real.pi)
  (triangle_condition: A + B + C = Real.pi) :
  C = Real.pi / 6 :=
sorry

end angle_C_is_30_degrees_l306_306264


namespace proof_problem_l306_306202

noncomputable def g (x : ℝ) : ℝ := 2^(2*x - 1) + x - 1

theorem proof_problem
  (x1 x2 : ℝ)
  (h1 : g x1 = 0)  -- x1 is the root of the equation
  (h2 : 2 * x2 - 1 = 0)  -- x2 is the zero point of f(x) = 2x - 1
  : |x1 - x2| ≤ 1/4 :=
sorry

end proof_problem_l306_306202


namespace value_of_x_plus_y_l306_306564

theorem value_of_x_plus_y (x y : ℝ) 
  (h1 : 2 * x - y = -1) 
  (h2 : x + 4 * y = 22) : 
  x + y = 7 :=
sorry

end value_of_x_plus_y_l306_306564


namespace isosceles_triangle_l306_306272

variable {A B C D E : Type} [triangle : triangle A B C]
variable (foot_D : foot D A B C) (foot_E : foot E B A C)
variable (acute : acute_triangle A B C)
variable (ineq1 : area B D E ≤ area D E A)
variable (ineq2 : area D E A ≤ area E A B)
variable (ineq3 : area E A B ≤ area A B D)

theorem isosceles_triangle
  (h1 : foot D A B C)
  (h2 : foot E B A C)
  (h3 : acute_triangle A B C)
  (h4 : area B D E ≤ area D E A)
  (h5 : area D E A ≤ area E A B)
  (h6 : area E A B ≤ area A B D) : 
  isosceles_triangle A B C :=
sorry

end isosceles_triangle_l306_306272


namespace charlotte_total_dog_walking_time_l306_306445

def poodles_monday : ℕ := 4
def chihuahuas_monday : ℕ := 2
def poodles_tuesday : ℕ := 4
def chihuahuas_tuesday : ℕ := 2
def labradors_wednesday : ℕ := 4

def time_poodle : ℕ := 2
def time_chihuahua : ℕ := 1
def time_labrador : ℕ := 3

def total_time_monday : ℕ := poodles_monday * time_poodle + chihuahuas_monday * time_chihuahua
def total_time_tuesday : ℕ := poodles_tuesday * time_poodle + chihuahuas_tuesday * time_chihuahua
def total_time_wednesday : ℕ := labradors_wednesday * time_labrador

def total_time_week : ℕ := total_time_monday + total_time_tuesday + total_time_wednesday

theorem charlotte_total_dog_walking_time : total_time_week = 32 := by
  -- Lean allows us to state the theorem without proving it.
  sorry

end charlotte_total_dog_walking_time_l306_306445


namespace garden_area_increase_l306_306936

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l306_306936


namespace cos_pi_over_3_plus_double_alpha_l306_306716

theorem cos_pi_over_3_plus_double_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 3 + 2 * α) = -7 / 8 :=
sorry

end cos_pi_over_3_plus_double_alpha_l306_306716


namespace colleagues_seniority_order_l306_306460

/-- Define the rankings for the three colleagues. -/
inductive Rank
| most_senior
| middle
| most_junior

/-- Define colleagues for simplicity. -/
def Dan : Prop := True
def Emma : Prop := True
def Fred : Prop := True

/-- Define the conditions of the problem. -/
def Statements := 
  (¬ (Rank.most_junior Dan) ∨ 
  (Rank.most_senior Emma) ∨ 
  (¬ (Rank.most_senior Fred)))

/-- The proposition to be proved: The correct order of seniority from most senior to most junior. -/
theorem colleagues_seniority_order
  (h1 : (¬ (Rank.most_junior Dan) = true)) 
  (h2 : (Rank.most_senior Emma) = false)
  (h3 : (¬ (Rank.most_senior Fred) = false)): 
  (Rank.most_senior Fred ∧ Rank.middle Dan ∧ Rank.most_junior Emma) :=
by
  sorry

end colleagues_seniority_order_l306_306460


namespace circumference_tank_M_l306_306322

-- Define the problem based on given conditions
def right_circular_cylinder (height circumference : ℝ) := 
  (height > 0) ∧ (circumference > 0)

-- Define the relationship between radius and circumference
def circumference_of_radius (r: ℝ) : ℝ := 2 * Real.pi * r

-- Define the volume of a cylinder given height and radius
def volume_of_cylinder (r h: ℝ): ℝ := Real.pi * r^2 * h

-- Given values
def height_M : ℝ := 10
def height_B : ℝ := 8
def circumference_B : ℝ := 10
def volume_ratio : ℝ := 0.8

-- The radius of tank B derived from circumference
def radius_B : ℝ := circumference_B / (2 * Real.pi)

-- The volume of tank M is 80% of the volume of tank B
theorem circumference_tank_M :
  let radius_M := (Real.sqrt (volume_ratio * volume_of_cylinder radius_B height_B) / height_M)
  circumference_of_radius radius_M = 8 :=
by sorry

end circumference_tank_M_l306_306322


namespace abc_value_l306_306723

theorem abc_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_ab : a * b = 15 * real.sqrt 3) (h_bc : b * c = 21 * real.sqrt 3) (h_ac : a * c = 10 * real.sqrt 3) :
  a * b * c = 15 * real.sqrt 42 :=
sorry

end abc_value_l306_306723


namespace no_natural_numbers_satisfy_conditions_l306_306471

theorem no_natural_numbers_satisfy_conditions : 
  ¬ ∃ (a b : ℕ), 
    (∃ (k : ℕ), k^2 = a^2 + 2 * b^2) ∧ 
    (∃ (m : ℕ), m^2 = b^2 + 2 * a) :=
by {
  -- Proof steps and logical deductions can be written here.
  sorry
}

end no_natural_numbers_satisfy_conditions_l306_306471


namespace months_in_season_l306_306014

/-- Definitions for conditions in the problem --/
def total_games_per_month : ℝ := 323.0
def total_games_season : ℝ := 5491.0

/-- The statement to be proven: The number of months in the season --/
theorem months_in_season (x : ℝ) (h : x = total_games_season / total_games_per_month) : x = 17.0 := by
  sorry

end months_in_season_l306_306014


namespace complement_union_l306_306658

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306658


namespace jerseys_sold_l306_306860

theorem jerseys_sold (unit_price_jersey : ℕ) (total_revenue_jersey : ℕ) (n : ℕ) 
  (h_unit_price : unit_price_jersey = 165) 
  (h_total_revenue : total_revenue_jersey = 25740) 
  (h_eq : n * unit_price_jersey = total_revenue_jersey) : 
  n = 156 :=
by
  rw [h_unit_price, h_total_revenue] at h_eq
  sorry

end jerseys_sold_l306_306860


namespace infinite_n_satisfying_f_k_min_l306_306820

def sum_digits (x : ℤ) : ℤ := -- implement the sum of digits function
sorry

def f_k (k n : ℤ) : ℤ := 
  (sum_digits (k * n^2)) / (sum_digits (n^3))

theorem infinite_n_satisfying_f_k_min {k : ℤ} (hk : k > 0) : 
  ∃ᶠ n : ℤ in filter.at_top, n ≥ 2 ∧ f_k k n < min (finset.range (n-1)).image (f_k k) :=
sorry

end infinite_n_satisfying_f_k_min_l306_306820


namespace slope_of_parallel_line_l306_306032

theorem slope_of_parallel_line (a b c : ℝ) (x y : ℝ) (h : 3 * x + 6 * y = -12):
  (∀ m : ℝ, (∀ (x y : ℝ), (3 * x + 6 * y = -12) → y = m * x + (-(12 / 6) / 6)) → m = -1/2) :=
sorry

end slope_of_parallel_line_l306_306032


namespace fraction_to_decimal_l306_306041

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end fraction_to_decimal_l306_306041


namespace find_a_l306_306720

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l306_306720


namespace complement_union_l306_306612

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306612


namespace inequality_proof_l306_306817

open Real

variable {a : ℕ → ℝ}

-- Conditions
def condition1 (a : ℕ → ℝ) : Prop :=
  ∀ (i j : ℕ), 1 ≤ i → 1 ≤ j → a (i + j) ≤ a i + a j

-- Statement
theorem inequality_proof (n : ℕ) (h_pos : 0 < n) (h_seq_pos : ∀ k, 0 < a k) (h_condition: condition1 a) :
  (∑ i in Finset.range n, (a (i + 1) / (i + 1))) ≥ a n :=
by
  sorry

end inequality_proof_l306_306817


namespace degree_not_determined_by_A_P_l306_306794

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l306_306794


namespace exactly_one_female_student_l306_306883

-- Definitions directly from the conditions
def groupA_males : ℕ := 5
def groupA_females : ℕ := 3
def groupB_males : ℕ := 6
def groupB_females : ℕ := 2

-- The number of ways to choose 1 female student and the remaining students accordingly
def scenario1 : ℕ := Nat.choose 3 1 * Nat.choose 5 1 * Nat.choose 6 2
def scenario2 : ℕ := Nat.choose 2 1 * Nat.choose 5 2 * Nat.choose 6 1

-- The total number of ways
def total_ways : ℕ := scenario1 + scenario2

-- Lean statement for the proof problem
theorem exactly_one_female_student : total_ways = 345 := by
  sorry

end exactly_one_female_student_l306_306883


namespace part_a_part_b_l306_306801

variables {a r : ℝ} (f : ℝ → ℝ)

namespace MathProof

-- Hypotheses
variables (ha : a > 1) (hr : r > 1)
    (hf1 : ∀ x > 0, f x^2 ≤ a * x^r * f (x / a))
    (hf2 : ∀ x, x < 1 / (2^2000) → f x < 2^2000)

/-- Part (a): Prove that for all x > 0, f(x) ≤ x^r * a^(1 - r) -/
theorem part_a : ∀ x > 0, f x ≤ x^r * a^(1 - r) :=
    sorry

/-- Part (b): Construct a function f such that:
  (i) f : ℝ → ℝ
  (ii) ∀ x > 0, f x^2 ≤ a * x^r * f (x / a)
  (iii) ∀ x > 0, f x > x^r * a^(1 - r)
-/
noncomputable def f_construction (x : ℝ) : ℝ := x^r * a^(1 - r) + 1

theorem part_b : (∀ x > 0, (f_construction x)^2 ≤ a * x^r * (f_construction (x / a)) ∧ 
                  ∀ x > 0, f_construction x > x^r * a^(1 - r)) :=
    sorry

end MathProof

end part_a_part_b_l306_306801


namespace find_a_l306_306721

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end find_a_l306_306721


namespace atleast_one_red_prob_l306_306429

/-- There are 5 balls: 2 are red and 3 are white. If 2 balls are randomly selected,
the probability that at least one of the selected balls is red is 7/10. -/
theorem atleast_one_red_prob (total_balls red_balls white_balls select_balls : ℕ)
  (h1 : total_balls = 5) (h2 : red_balls = 2) (h3 : white_balls = 3) (h4 : select_balls = 2) :
  (∃ favorable_ways : ℕ, favorable_ways = 7) ∧ 
  (∃ total_ways : ℕ, total_ways = 10) →
  (favorable_ways : ℕ) = 7 → (total_ways : ℕ) = 10 →
  7 / 10 := by
  sorry

end atleast_one_red_prob_l306_306429


namespace y2_odd_increasing_l306_306989

def y1 (x : ℝ) := 4 * x + 1 / x
def y2 (x : ℝ) := exp x - exp (- x)
def y3 (x : ℝ) := exp x + exp (- x)
def y4 (x : ℝ) := (x - x^2) / (1 - x)

theorem y2_odd_increasing :
  (∀ x, y2 (-x) = -y2 x) ∧ (∀ x, y2' x > 0) := sorry

end y2_odd_increasing_l306_306989


namespace sequence_xn_converges_l306_306290

variable {α : Type*} [LinearOrderedField α]

noncomputable def sequence_convergence (x y : ℕ → α) : Prop :=
  (∀ n > 1, y n = x (n-1) + 2 * x n) → (filter.tendsto y filter.at_top (nhds 0) →
  filter.tendsto x filter.at_top (nhds 0))

theorem sequence_xn_converges (x y : ℕ → α) (h : ∀ n > 1, y n = x (n-1) + 2 * x n) :
  filter.tendsto y filter.at_top (nhds 0) → filter.tendsto x filter.at_top (nhds 0) := 
begin
  sorry,
end

end sequence_xn_converges_l306_306290


namespace value_of_expression_l306_306006

theorem value_of_expression : (5 * 6 - 3 * 4) / (6 + 3) = 2 := 
by 
    have h_numerator := 5 * 6 - 3 * 4
    have h_denominator := 6 + 3
    have h_fraction := h_numerator / h_denominator
    have h1 : h_numerator = 18 := by sorry
    have h2 : h_denominator = 9 := by sorry
    have h3 : h_fraction = 2 := by sorry
    exact h3

end value_of_expression_l306_306006


namespace evaluate_expression_l306_306481

def f : ℕ → ℚ
| 1       := 1 / 2
| (n + 1) := 1 / (2 - f n)

theorem evaluate_expression : f 2013 = 2013 / 2014 := by
  sorry

end evaluate_expression_l306_306481


namespace degree_not_determined_from_characteristic_l306_306790

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l306_306790


namespace perpendicular_tangents_l306_306335

theorem perpendicular_tangents (m : ℝ) : 
  let F1 := λ x : ℝ, x^2
  let F2 := λ x : ℝ, (x - m)^2 - 1
  let slope_F1 := λ x : ℝ, 2 * x
  let slope_F2 := λ x : ℝ, 2 * (x - m)
  tangent_perpendicular : slope_F1 1 * slope_F2 1 = -1
  in m = 5 / 4 :=
by
  -- Define F1, F2, slope_F1, and slope_F2
  let F1 := λ x : ℝ, x^2
  let F2 := λ x : ℝ, (x - m)^2 - 1
  let slope_F1 := λ x : ℝ, 2 * x
  let slope_F2 := λ x : ℝ, 2 * (x - m)

  -- Define the condition that the tangents are perpendicular
  have tangent_perpendicular : slope_F1 1 * slope_F2 1 = -1 := sorry
  
  -- Show that m = 5 / 4
  exact calc 
    m = 5 / 4 : sorry

end perpendicular_tangents_l306_306335


namespace find_shaded_area_l306_306007

noncomputable def area_shaded_region (side : ℝ) (radius : ℝ) : ℝ := 
  let area_square := side ^ 2
  let area_circular_sector := (1 / 4) * π * radius ^ 2
  let leg1 := side / 2
  let leg2 := (radius ^ 2 - leg1 ^ 2) ^ (1/2)
  let area_triangle := (1 / 2) * leg1 * leg2
  area_square - 4 * area_circular_sector - 8 * area_triangle

theorem find_shaded_area : 
  ∀ (side : ℝ) (radius : ℝ), 
  side = 10 → radius = 3 * Real.sqrt 3 → 
  area_shaded_region side radius = 100 - 27 * π - 20 * Real.sqrt 2 :=
by
  intros side radius h_side h_radius
  rw [h_side, h_radius]
  unfold area_shaded_region
  sorry

end find_shaded_area_l306_306007


namespace candy_count_l306_306360

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end candy_count_l306_306360


namespace B_travelled_path_length_l306_306100

noncomputable def arc_AC_quarter_circle_center_B (BC_radius : ℝ) := 
  ∃ (arc: ℝ), arc = 2 * π * (BC_radius / π) / 4

noncomputable def region_ABC_roll_along_PQ (BC_radius : ℝ) :=
  arc_AC_quarter_circle_center_B BC_radius ∧ BC_radius = 3 / π

theorem B_travelled_path_length (BC_radius : ℝ) 
  (h1: region_ABC_roll_along_PQ BC_radius):
  ∃ length, length = 4.5 := 
by
  sorry

end B_travelled_path_length_l306_306100


namespace transformed_sine_function_l306_306238

theorem transformed_sine_function :
  ∀ (x : ℝ), (∃ y : ℝ, y = sin (2 * x - π / 3)) :=
by
  sorry

end transformed_sine_function_l306_306238


namespace total_plant_count_l306_306961

-- Definitions for conditions.
def total_rows : ℕ := 96
def columns_per_row : ℕ := 24
def divided_rows : ℕ := total_rows / 3
def undivided_rows : ℕ := total_rows - divided_rows
def beans_in_undivided_row : ℕ := columns_per_row
def corn_in_divided_row : ℕ := columns_per_row / 2
def tomatoes_in_divided_row : ℕ := columns_per_row / 2

-- Total number of plants calculation.
def total_bean_plants : ℕ := undivided_rows * beans_in_undivided_row
def total_corn_plants : ℕ := divided_rows * corn_in_divided_row
def total_tomato_plants : ℕ := divided_rows * tomatoes_in_divided_row

def total_plants : ℕ := total_bean_plants + total_corn_plants + total_tomato_plants

-- Proof statement.
theorem total_plant_count : total_plants = 2304 :=
by
  sorry

end total_plant_count_l306_306961


namespace four_digit_numbers_count_l306_306138

theorem four_digit_numbers_count:
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  ∃ nums : Finset (Fin 10), 
    (∀ num ∈ nums, (∃ a b c d : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
    b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    ∀ x ∈ {a, b, c, d}, x ∈ digits ∧ 
    (x = (a + b) / 2 ∨ x = (a + c) / 2 ∨ x = (a + d) / 2 ∨ 
    x = (b + c) / 2 ∨ x = (b + d) / 2 ∨ x = (c + d) / 2))) ∧
    nums.card = 2880 := sorry

end four_digit_numbers_count_l306_306138


namespace average_speed_correct_l306_306073

def distance1 : ℝ := 8
def speed1 : ℝ := 10
def distance2 : ℝ := 10
def speed2 : ℝ := 8
def total_distance : ℝ := distance1 + distance2
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2
def total_time : ℝ := time1 + time2
def average_speed : ℝ := total_distance / total_time

theorem average_speed_correct :
  abs (average_speed - 8.78) < 0.01 :=
by
  sorry

end average_speed_correct_l306_306073


namespace slope_of_line_through_midpoints_l306_306901

-- Define points and calculate midpoints
structure Point where
  x : ℝ
  y : ℝ

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def slope (p1 p2 : Point) : ℝ :=
  if h : p1.x ≠ p2.x then (p2.y - p1.y) / (p2.x - p1.x) else 0

theorem slope_of_line_through_midpoints
  (A B C D : Point)
  (h1 : A.x = 0) (h2 : A.y = 0)
  (h3 : B.x = 3) (h4 : B.y = 4)
  (h5 : C.x = 6) (h6 : C.y = 0)
  (h7 : D.x = 7) (h8 : D.y = 4) :
  slope (midpoint A B) (midpoint C D) = 0 :=
by
  sorry

end slope_of_line_through_midpoints_l306_306901


namespace min_value_of_f_inequality_l306_306208

open Real

noncomputable theory

-- Definition of the function f(x).
def f (x : ℝ) : ℝ := 2 * abs(x + 1) + abs(x + 2)

-- First statement for minimum value of the function.
theorem min_value_of_f : ∃ x₀, (∀ x, f x ≥ f x₀) ∧ f x₀ = 1 := sorry

-- Second statement for the inequality given a + b + c = 1.
theorem inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b + c = 1) :
  (a^2 + b^2) / c + (c^2 + a^2) / b + (b^2 + c^2) / a ≥ 2 := sorry

end min_value_of_f_inequality_l306_306208


namespace complement_union_l306_306603

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306603


namespace problem_proof_l306_306448

-- Definitions for conditions
variables {O₁ O₂ : Type} [circle O₁] [circle O₂]
variables {B C A E F H G D : Point}
variables (h_intersect : intersect O₁ O₂ B C)
variables (h_diameter : diameter O₁ B C)
variables (h_tangent : tangent O₁ C A)
variables (h_ABE : intersects_again (line A B) O₁ E)
variables (h_CEF : intersects_again (extended_line C E) O₂ F)
variables (h_H_on_AF : lies_on H (segment A F))
variables (h_HEG : intersects_again (extended_line H E) O₁ G)
variables (h_BGAD : intersects_extended_line (line B G) (extended_line A C) D)

-- Statement of the proof problem
theorem problem_proof :
  ∀ (H : Point),
  lies_on H (segment A F) →
  ∀ (G D : Point),
  intersects_again (extended_line H E) O₁ G →
  intersects_extended_line (line B G) (extended_line A C) D →
  (AH / HF) = (AC / CD) :=
by { sorry }

end problem_proof_l306_306448


namespace sum_of_denominators_of_fractions_l306_306358

theorem sum_of_denominators_of_fractions {a b : ℕ} (ha : 3 * a / 5 * b + 2 * a / 9 * b + 4 * a / 15 * b = 28 / 45) (gcd_ab : Nat.gcd a b = 1) :
  5 * b + 9 * b + 15 * b = 203 := sorry

end sum_of_denominators_of_fractions_l306_306358


namespace pipe_b_filling_time_l306_306385

theorem pipe_b_filling_time :
  (∃ (c : ℝ), c > 0 ∧ (4 * c + 2 * c + c = 1 / 16)) →
  (1 / (2 * classical.some _) = 56) :=
by
  intro h
  sorry

end pipe_b_filling_time_l306_306385


namespace probability_log3_integer_l306_306971

noncomputable theory
open BigOperators

def log3_is_int (N : ℕ) : Prop :=
  ∃ k : ℕ, N = 3^k

theorem probability_log3_integer (N : ℕ) (hN : 100 ≤ N ∧ N ≤ 999)
  (H : ∀ n, 100 ≤ n ∧ n ≤ 999 → uniform_probability n) : 
  1 / 450 :=
by
  sorry

end probability_log3_integer_l306_306971


namespace number_of_real_solutions_l306_306468

theorem number_of_real_solutions (x : ℝ) (n : ℤ) : 
  (3 : ℝ) * x^2 - 27 * (n : ℝ) + 29 = 0 → n = ⌊x⌋ →  ∃! x, (3 : ℝ) * x^2 - 27 * (⌊x⌋ : ℝ) + 29 = 0 := 
sorry

end number_of_real_solutions_l306_306468


namespace example_theorem_l306_306673

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306673


namespace fraction_of_number_is_three_quarters_l306_306066

theorem fraction_of_number_is_three_quarters 
  (f : ℚ) 
  (h1 : 76 ≠ 0) 
  (h2 : f * 76 = 76 - 19) : 
  f = 3 / 4 :=
by
  sorry

end fraction_of_number_is_three_quarters_l306_306066


namespace right_triangle_equal_ratios_l306_306826

theorem right_triangle_equal_ratios
  (XYZ : Triangle)
  (h_right : XYZ.angle Y Z X = 90)
  (sequence : ℕ → Triangle)
  (h_sequence : sequence 0 = XYZ ∧ 
                (∀ i, (sequence i).angle Y Z X = 90) ∧
                (∀ i, let t := sequence i in t.vertex X lies_on segment Y t.opposite_side T) ∧
                (∀ i, let t := sequence i in t.vertex Y lies_on segment Z t.opposite_side T) ∧
                (∀ i, sequence (i + 1).vertex X lies_on segment (sequence i).vertex Y sequence(i).opposite_side T)) :
  (area (⋃ i, sequence i) = area XYZ) →
  (XY / YZ = 1) :=
sorry

end right_triangle_equal_ratios_l306_306826


namespace systematic_sample_seat_number_l306_306071

theorem systematic_sample_seat_number (total_students : ℕ) (sample_size : ℕ) 
  (seat_numbers : Fin 60 → ℕ) (sample : Set ℕ)
  (h1 : total_students = 60)
  (h2 : sample_size = 4)
  (h3 : sample = {3, 18, 48, 33})
  (h4 : ∀ x ∈ sample, x ∈ seat_numbers '' (Finset.range total_students))
  (h5 : ∃ f : ℕ, f = total_students / sample_size)
  (h6 : 18 - 3 = 15)
  (h7 : 48 - 18 = 2 * 15)
  : 33 ∈ sample :=
sorry

end systematic_sample_seat_number_l306_306071


namespace square_diagonal_length_l306_306324

noncomputable def diagonal_of_square (area : ℝ) : ℝ :=
  let side := Real.sqrt area in
  let diagonal := Real.sqrt (2 * side^2) in
  diagonal

theorem square_diagonal_length (area : ℝ) (h : area = 4802) : diagonal_of_square area ≈ 98 :=
by {
  rw [diagonal_of_square, h],
  norm_num,
  sorry  -- Proof of approximation to be provided
}

end square_diagonal_length_l306_306324


namespace proof_problem_l306_306538

theorem proof_problem 
  (a b c : ℤ) 
  (h1 : (a - 4)^(1/3 : ℝ) = 1) 
  (h2 : (b : ℝ)^(1/2 : ℝ) = 2) 
  (h3 : ⌊(11 : ℝ)^(1/2 : ℝ)⌋ = c) : 
  a = 5 ∧ 
  b = 4 ∧ 
  c = 3 ∧ 
  ((2 * a - 3 * b + c)^(1/2 : ℝ) = 1 ∨ (2 * a - 3 * b + c)^(1/2 : ℝ) = -1) :=
sorry

end proof_problem_l306_306538


namespace complement_union_of_M_and_N_l306_306686

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306686


namespace find_t_l306_306541

theorem find_t (t M N : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = (t * x^2 + 2 * x + t^2 + Real.sin x) / (x^2 + t))
  (h2 : t > 0) (h3 : ∀ x, IsMax f x ↔ f x = M) (h4 : ∀ x, IsMin f x ↔ f x = N)
  (h5 : M + N = 4) : t = 2 := by
  sorry

end find_t_l306_306541


namespace min_tetrahedrons_to_partition_cube_l306_306902

theorem min_tetrahedrons_to_partition_cube : ∃ n : ℕ, n = 5 ∧ (∀ m : ℕ, m < 5 → ¬partitions_cube_into_tetrahedra m) :=
by
  sorry

end min_tetrahedrons_to_partition_cube_l306_306902


namespace find_number_of_As_l306_306260

variables (M L S : ℕ)

def number_of_As (M L S : ℕ) : Prop :=
  M + L = 23 ∧ S + M = 18 ∧ S + L = 15

theorem find_number_of_As (M L S : ℕ) (h : number_of_As M L S) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end find_number_of_As_l306_306260


namespace complement_union_l306_306605

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306605


namespace log_8_4000_l306_306027

theorem log_8_4000 : ∃ (n : ℤ), 8^3 = 512 ∧ 8^4 = 4096 ∧ 512 < 4000 ∧ 4000 < 4096 ∧ n = 4 :=
by
  sorry

end log_8_4000_l306_306027


namespace arithmetic_sequence_inequality_l306_306812

variable {a1 a2 a3 : ℝ}
variable {d : ℝ}

-- Definition of an arithmetic sequence in Lean
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a n + d = a (n + 1)

theorem arithmetic_sequence_inequality (h_seq : is_arithmetic_sequence (λ n, a1 + n * d))
  (h_pos1 : 0 < a1) (h_pos2 : a1 < a2) :
  a2 = (a1 + a3) / 2 ∧ a2 > Real.sqrt (a1 * a3) :=
sorry

end arithmetic_sequence_inequality_l306_306812


namespace max_cables_used_l306_306434

/--
An organization has 40 employees, 25 of whom have brand A computers while the other 15 have brand B computers.
Computers can only be connected from brand A to brand B. Every employee should be able to communicate either
directly or indirectly. Prove that the maximum number of cables that could be used is 361.
-/
theorem max_cables_used
  (employees : ℕ)
  (brandA : ℕ)
  (brandB : ℕ)
  (con : brandA + brandB = employees)
  (cables : ℕ)
  (condition : ∀ a b, a ∈ finset.range brandA → b ∈ finset.range brandB → connected a b) :
  cables ≤ 361 :=
sorry

end max_cables_used_l306_306434


namespace area_of_triangle_ABC_l306_306058

-- Definitions based on conditions
variables {A B C : Point}
variable [Triangle ABC]

-- Provided conditions
variables (BC AC : ℝ)
variables (h1 : BC = 6)
variables (h2 : AC = 9)
variables (O : Point)
variable (h3 : O ∈ segment A C)
variables (R : ℝ)
variable (h4 : circle_center_touches_B_pasces_through_A (Circle O R) BC B AC A O)

-- Question and the final proof statement
theorem area_of_triangle_ABC (h₁ : BC = 6) (h₂ : AC = 9) (h₃ : O ∈ segment A C) (h₄ : circle_center_touches_B_pasces_through_A (Circle O R) BC B AC A O) 
    : area ABC = 135 / 13 :=
sorry

end area_of_triangle_ABC_l306_306058


namespace not_perfect_square_for_n_values_l306_306314

def is_not_perfect_square (x : ℕ) : Prop := ∀ y : ℕ, y * y ≠ x

noncomputable def sum_of_squares (a n : ℕ) : ℕ :=
  ∑ k in Finset.range (2 * n + 1), (a + k - n) ^ 2

theorem not_perfect_square_for_n_values :
  ∀ a, ∀ n ∈ {1, 2, 3, 4}, is_not_perfect_square (sum_of_squares a n) := by
  sorry

end not_perfect_square_for_n_values_l306_306314


namespace bisect_segment_l306_306321

theorem bisect_segment (O S T1 T2 M K L : Point) (circleO : Circle O)
  (h1 : Tangent S T1 circleO) (h2 : Tangent S T2 circleO)
  (h3 : OnChord M T1 T2 circleO)
  (h4 : Perpendicular (LineThrough M) (OM)) :
  SegmentBisectedByM K L M :=
sorry

end bisect_segment_l306_306321


namespace first_quartile_is_294_l306_306545

theorem first_quartile_is_294 :
  let data := [296, 301, 305, 293, 293, 305, 302, 303, 306, 294] in
  let sorted_data := data.sort (≤) in
  first_quartile sorted_data = 294 :=
by
  sorry

end first_quartile_is_294_l306_306545


namespace example_theorem_l306_306662

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306662


namespace number_of_roses_per_set_l306_306891

-- Define the given conditions
def total_days : ℕ := 7
def sets_per_day : ℕ := 2
def total_roses : ℕ := 168

-- Define the statement to be proven
theorem number_of_roses_per_set : 
  (sets_per_day * total_days * (total_roses / (sets_per_day * total_days)) = total_roses) ∧ 
  (total_roses / (sets_per_day * total_days) = 12) :=
by 
  sorry

end number_of_roses_per_set_l306_306891


namespace garden_area_difference_l306_306952

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l306_306952


namespace incorrect_statement_d_l306_306380

theorem incorrect_statement_d :
  (¬(abs 2 = -2)) :=
by sorry

end incorrect_statement_d_l306_306380


namespace sum_geometric_sequence_l306_306874

theorem sum_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a 1 + ∑ i in finset.range (n - 1), 3^i * a (i + 2) = n / 2) →
  (∀ n : ℕ, S n = (3 / 4) * (1 - (1 / 3^n))) :=
by
  sorry

end sum_geometric_sequence_l306_306874


namespace area_of_triangle_ABC_l306_306245

-- Given conditions
variable {A B C : Type} [EuclideanGeometry A B C]
variable (triangle_ABC : EuclideanGeometry.Triangle A B C)
variable (angle_A : Real) (AC BC : Real)

-- Set the given conditions
axiom angle_A_equals_60 : angle_A = 60
axiom AC_equals_4 : AC = 4
axiom BC_equals_2_sqrt_3 : BC = 2 * Real.sqrt 3

-- Proof statement that the area of triangle ABC is 2√3
theorem area_of_triangle_ABC : EuclideanGeometry.area triangle_ABC = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_ABC_l306_306245


namespace binomial_16_12_l306_306123

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l306_306123


namespace eat_cereal_together_l306_306299

theorem eat_cereal_together (fat_rate : ℝ := 1 / 25) (thin_rate : ℝ := 1 / 40) (combined_rate : ℝ := fat_rate + thin_rate) :
  5 / combined_rate = 1000 / 13 :=
by
  have h1 : combined_rate = 1 / 25 + 1 / 40 := rfl
  have h2 : combined_rate = 13 / 200 := by field_simp [h1]; norm_num
  have h3 : 5 / combined_rate = 5 / (13 / 200) := by rw h2
  have h4 : 5 / (13 / 200) = 5 * (200 / 13) := by field_simp
  have h5 : 5 * (200 / 13) = (5 * 200) / 13 := mul_div_assoc 5 200 13
  norm_num at h5
  rw h5
  exact rfl

end eat_cereal_together_l306_306299


namespace max_sum_abs_values_l306_306813

-- Define the main problem in Lean
theorem max_sum_abs_values (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 :=
by
  intros h
  sorry

end max_sum_abs_values_l306_306813


namespace min_time_to_complete_tasks_l306_306705

theorem min_time_to_complete_tasks :
  ∀ (W_rice C_porridge W_veg C_veg : ℕ), 
    W_rice = 2 → 
    C_porridge = 10 → 
    W_veg = 3 → 
    C_veg = 5 → 
    (W_rice + C_porridge) = 12 := 
by 
  intros W_rice C_porridge W_veg C_veg 
  assume h1 : W_rice = 2 
  assume h2 : C_porridge = 10 
  assume h3 : W_veg = 3 
  assume h4 : C_veg = 5 
  rw [h1, h2]
  exact rfl

end min_time_to_complete_tasks_l306_306705


namespace circle_intersection_unique_point_l306_306760

open Complex

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem circle_intersection_unique_point :
  ∃ k : ℝ, (distance (0, 0) (-5 / 2, 0) - 3 / 2 = k ∨ distance (0, 0) (-5 / 2, 0) + 3 / 2 = k)
  ↔ (k = 2 ∨ k = 5) := sorry

end circle_intersection_unique_point_l306_306760


namespace sqrt_7_irrational_l306_306850

theorem sqrt_7_irrational : ¬ ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧ (Nat.gcd p q = 1) ∧ (7 = p * p) ∧ (q * q) := sorry

end sqrt_7_irrational_l306_306850


namespace unique_b_intersects_vertex_l306_306463

-- Define the equation of the line: y = 2x + b
def line (b x : ℝ) : ℝ := 2 * x + b

-- Define the equation of the parabola: y = x^2 + 2bx
def parabola (b x : ℝ) : ℝ := x^2 + 2 * b * x

-- Define the vertex of the parabola y = x^2 + 2bx
def parabola_vertex (b : ℝ) : ℝ × ℝ := (-b, -b^2)

-- Define the condition that the vertex of the parabola lies on the line
def intersects_vertex (b : ℝ) : Prop :=
  let (vx, vy) := parabola_vertex b in line b vx = vy

-- Define the theorem to prove: There is exactly one b such that line intersects the vertex of the parabola
theorem unique_b_intersects_vertex : ∃! b : ℝ, intersects_vertex b := sorry

end unique_b_intersects_vertex_l306_306463


namespace complement_union_eq_l306_306580

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306580


namespace complement_union_M_N_l306_306632

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306632


namespace find_a_tangent_perpendicular_l306_306496

theorem find_a_tangent_perpendicular :
  let f := λ x : ℝ, (x + 1) / (x - 1)
  let point := (3 : ℝ, 2 : ℝ)
  let line := λ a : ℝ, λ x : ℝ, λ y : ℝ, a * x + y + 1 = 0
  let derivative_f := λ x : ℝ, -2 / (x - 1) ^ 2 in
  (derivative_f 3) * (-a) = -1 → a = -2 :=
by
  sorry

end find_a_tangent_perpendicular_l306_306496


namespace complement_union_M_N_l306_306636

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306636


namespace at_least_two_same_handshakes_l306_306356

theorem at_least_two_same_handshakes (n : ℕ) (handshakes : Finset (Finset ℕ)) (H : ∀ h ∈ handshakes, h.card ≤ n - 1) : 
  2 ≤ n → ∃ (students : Finset ℕ), ∃ (h1 h2 ∈ handshakes), h1 ≠ h2 ∧ h1.card = h2.card :=
by 
  sorry

end at_least_two_same_handshakes_l306_306356


namespace circles_intersect_l306_306346

-- Definitions
def circle1 (x y : ℝ) := x^2 + y^2 = 1
def circle2 (x y : ℝ) := (x - 2)^2 + (y - 2)^2 = 5

-- Distance between centers of the circles
def distance_between_centers := Real.sqrt ((2 - 0)^2 + (2 - 0)^2)

-- Radii of the circles
def radius_circle1 := 1
def radius_circle2 := Real.sqrt 5

-- Conditions for circles to intersect
def circles_intersect_condition := 
  radius_circle2 - radius_circle1 < distance_between_centers ∧ 
  distance_between_centers < radius_circle2 + radius_circle1

-- Main theorem statement
theorem circles_intersect : 
  circles_intersect_condition := 
by
  -- Proof omitted, sorry used to indicate incomplete proof.
  sorry

end circles_intersect_l306_306346


namespace regular_polygon_sides_l306_306475

noncomputable def interiorAngle (n : ℕ) : ℝ :=
  if n ≥ 3 then (180 * (n - 2) / n) else 0

noncomputable def exteriorAngle (n : ℕ) : ℝ :=
  180 - interiorAngle n

theorem regular_polygon_sides (n : ℕ) (h : interiorAngle n = 160) : n = 18 :=
by sorry

end regular_polygon_sides_l306_306475


namespace shortest_distance_l306_306405

-- Define the displacements along x and y axes as given by the conditions
definition displacement_x (t : Nat) : Real :=
  if t < 2 then 2.5 * t
  else if t < 4 then 5
  else if t < 5 then 4 - (t - 4)
  else 4

definition displacement_y (t : Nat) : Real :=
  if t < 2 then 0
  else if t < 4 then 2 * (t - 2)
  else if t < 5 then 4
  else 4 - (t - 4)

-- Define the total displacement in both axes
definition total_displacement_x : Real :=
  displacement_x 5

definition total_displacement_y : Real :=
  displacement_y 5

-- Prove that the shortest distance is 5 meters
theorem shortest_distance :
  sqrt ((total_displacement_x) ^ 2 + (total_displacement_y) ^ 2) = 5 :=
by
  sorry

end shortest_distance_l306_306405


namespace choir_singers_joined_final_verse_l306_306410

theorem choir_singers_joined_final_verse (total_singers : ℕ) (first_verse_fraction : ℚ)
  (second_verse_fraction : ℚ) (initial_remaining : ℕ) (second_verse_joined : ℕ) : 
  total_singers = 30 → 
  first_verse_fraction = 1 / 2 → 
  second_verse_fraction = 1 / 3 → 
  initial_remaining = total_singers / 2 → 
  second_verse_joined = initial_remaining / 3 → 
  (total_singers - (initial_remaining + second_verse_joined)) = 10 := 
by
  intros
  sorry

end choir_singers_joined_final_verse_l306_306410


namespace complement_union_eq_singleton_five_l306_306573

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306573


namespace torn_pages_sum_odd_l306_306868

theorem torn_pages_sum_odd (pages : ℕ → ℕ) :
  (∀ n, pages n = n + 1) → (∃ L : list ℕ, L.length = 50 ∧ list.sum L = 2020) → false :=
by
  assume h_pages h_portion
  sorry

end torn_pages_sum_odd_l306_306868


namespace sum_of_cubes_of_roots_l306_306284

theorem sum_of_cubes_of_roots (P : Polynomial ℝ)
  (hP : P = Polynomial.C (-1) + Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X) 
  (x1 x2 x3 : ℝ) 
  (hr : P.eval x1 = 0 ∧ P.eval x2 = 0 ∧ P.eval x3 = 0) :
  x1^3 + x2^3 + x3^3 = 3 := 
sorry

end sum_of_cubes_of_roots_l306_306284


namespace complement_union_M_N_l306_306629

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306629


namespace smallest_n_mod_equiv_l306_306372

open Int

theorem smallest_n_mod_equiv : ∃ n : ℕ, 0 < n ∧ 29 * n ≡ 5678 [MOD 11] ∧ ∀ m : ℕ, (0 < m ∧ 29 * m ≡ 5678 [MOD 11]) → n ≤ m := by
  use 9
  split
  · exact Nat.succ_pos' _
  split
  · calc 
      29 * 9 ≡ 261 [MOD 11] := by norm_num
      5678 ≡ 2 [MOD 11] := by norm_num
      _ ≡ 2 [MOD 11] := by ring
  · intro m hm
    obtain ⟨_, hm⟩ := hm
    have : ∃ k : ℤ, 29 * m = 5678 + 11 * k := by 
      cases hm with x hx
      use x
      norm_cast at hx
    sorry

end smallest_n_mod_equiv_l306_306372


namespace find_inverse_at_2_l306_306240

noncomputable def f (x : ℝ) : ℝ := x ^ (-1 / 2)
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x ^ 2)

theorem find_inverse_at_2 :
  (∃ α : ℝ, f(2) = 2 ^ α ∧ f (2) = (√2) / 2) → f_inv(2) = 1 / 4 :=
by
  intro h
  have α := -1 / 2
  have heq : f (2) = 2 ^ α := by sorry
  have inv : f (x) = x ^ (-1 / 2) := by sorry
  have h : f_inv (2) = 1 / (2^2) := by sorry
  rw ← h
  exact heq

end find_inverse_at_2_l306_306240


namespace cannot_determine_degree_from_char_set_l306_306768

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l306_306768


namespace unit_vector_perpendicular_to_a_l306_306095

theorem unit_vector_perpendicular_to_a :
  ∃ x y : ℝ, (x - sqrt 3 * y = 0) ∧ (x^2 + y^2 = 1) ∧ (x = sqrt 3 / 2) ∧ (y = 1 / 2) :=
by 
  use sqrt 3 / 2, 1 / 2
  split
  · sorry
  split
  · sorry
  split
  · rfl
  · rfl

end unit_vector_perpendicular_to_a_l306_306095


namespace maximum_area_of_triangle_ABC_l306_306805

-- Given points and conditions
def A := (2 : ℝ, 0 : ℝ)
def B := (5 : ℝ, 3 : ℝ)
def parabola (x : ℝ) : ℝ := x^2 - 6 * x + 11
def C (p : ℝ) := (p, parabola p)

noncomputable def area_triangle (p : ℝ) : ℝ :=
  abs((A.1 * B.2 + B.1 * (parabola p) + p * A.2 - A.2 * B.1 - B.2 * p - (parabola p) * A.1) / 2)

theorem maximum_area_of_triangle_ABC :
  ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ area_triangle p = 4.5 :=
sorry

end maximum_area_of_triangle_ABC_l306_306805


namespace problem_statement_l306_306532

noncomputable def sequence_def (a : ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  (a ≠ 0) ∧
  (S 1 = a) ∧
  (S 2 = 2 / S 1) ∧
  (∀ n, n ≥ 3 → S n = 2 / S (n - 1))

theorem problem_statement (a : ℝ) (S : ℕ → ℝ) (h : sequence_def a S 2018) : 
  S 2018 = 2 / a := 
by 
  sorry

end problem_statement_l306_306532


namespace find_n_l306_306803

theorem find_n (n : ℕ) (m : ℕ) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_div : (2^n - 1) ∣ (m^2 + 81)) : 
  ∃ k : ℕ, n = 2^k := 
sorry

end find_n_l306_306803


namespace garden_area_difference_l306_306954

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l306_306954


namespace probability_A_not_losing_is_80_percent_l306_306403

def probability_A_winning : ℝ := 0.30
def probability_draw : ℝ := 0.50
def probability_A_not_losing : ℝ := probability_A_winning + probability_draw

theorem probability_A_not_losing_is_80_percent : probability_A_not_losing = 0.80 :=
by 
  sorry

end probability_A_not_losing_is_80_percent_l306_306403


namespace rectangle_area_increase_l306_306050

theorem rectangle_area_increase (L B : ℝ) :
  let new_length := 1.05 * L,
      new_breadth := 1.15 * B,
      original_area := L * B,
      new_area := new_length * new_breadth,
      area_increase := new_area - original_area
  in (area_increase / original_area) * 100 = 20.75 :=
by
  sorry

end rectangle_area_increase_l306_306050


namespace minimum_tetrahedra_partition_l306_306905

-- Definitions for the problem conditions
def cube_faces : ℕ := 6
def tetrahedron_faces : ℕ := 4

def face_constraint (cube_faces : ℕ) (tetrahedral_faces : ℕ) : Prop :=
  cube_faces * 2 = 12

def volume_constraint (cube_volume : ℝ) (tetrahedron_volume : ℝ) : Prop :=
  tetrahedron_volume < cube_volume / 6

-- Main proof statement
theorem minimum_tetrahedra_partition (cube_faces tetrahedron_faces : ℕ) (cube_volume tetrahedron_volume : ℝ) :
  face_constraint cube_faces tetrahedron_faces →
  volume_constraint cube_volume tetrahedron_volume →
  5 ≤ cube_faces * 2 / 3 :=
  sorry

end minimum_tetrahedra_partition_l306_306905


namespace world_cup_edition_l306_306323

theorem world_cup_edition (a_n : ℕ → ℕ) (n : ℕ) :
  (∀ n, a_n n = 1950 + 4 * (n - 4)) ∧ a_n 21 = 2018 → n = 21 :=
by
  have h : 2018 = 1950 + 4 * (21 - 4) := rfl
  sorry

end world_cup_edition_l306_306323


namespace complement_union_of_M_and_N_l306_306687

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306687


namespace probability_hat_given_sunglasses_l306_306303

theorem probability_hat_given_sunglasses : 
  let total_sunglasses := 80 in
  let total_hats := 60 in
  let prob_hat_given_sunglasses := (1:ℝ) / 3 in
  let both := prob_hat_given_sunglasses * total_hats in
  (both / total_sunglasses = (1:ℝ) / 4) :=
by
  sorry

end probability_hat_given_sunglasses_l306_306303


namespace arrangement_equivalence_l306_306241

noncomputable def arrangements_single_row (n : ℕ) : ℕ := (2 * n)!

noncomputable def arrangements_two_rows (n : ℕ) : ℕ := (2 * n)! / (n! * n!)

theorem arrangement_equivalence (n : ℕ) : 
  arrangements_single_row n = arrangements_two_rows n := by
    sorry

end arrangement_equivalence_l306_306241


namespace log_a_b_eq_neg_one_l306_306286

variable (a b : ℝ)

-- Assuming a and b are positive real numbers
variable (ha : 0 < a)
variable (hb : 0 < b)

-- Condition 1: 1/a + 1/b ≤ 2√2
variable (h1 : 1/a + 1/b ≤ 2 * Real.sqrt 2)

-- Condition 2: (a - b)^2 = 4(ab)^3
variable (h2 : (a - b)^2 = 4 * (a * b)^3)

-- Goal: log_a(b) = -1
theorem log_a_b_eq_neg_one (ha : 0 < a) (hb : 0 < b) (h1 : 1/a + 1/b ≤ 2 * Real.sqrt 2) (h2 : (a - b)^2 = 4 * (a * b)^3) :
  Real.log a b = -1 :=
sorry

end log_a_b_eq_neg_one_l306_306286


namespace find_a_l306_306728

theorem find_a (a : ℝ) : (dist_point_line 2 2 3 (-4) a) = a → a = 1 / 3 := by
  intro h
  have H : (|3 * 2 + (-4) * 2 + a| / sqrt (3 ^ 2 + 4 ^ 2)) = a from h
  sorry

end find_a_l306_306728


namespace max_list_element_l306_306420

theorem max_list_element 
  (l : List ℕ) 
  (hl_pos : ∀ x ∈ l, x > 0) 
  (hl_len : l.length = 5) 
  (hl_median : l.nth_le 2 (by linarith [hl_len]) = 5) 
  (hl_mean : (l.sum : ℝ) / l.length = 12) : 
  l.maximum_attained (by linarith [hl_len]) = 44 := 
sorry

end max_list_element_l306_306420


namespace optionA_is_square_difference_l306_306039

theorem optionA_is_square_difference (x y : ℝ) : 
  (-x + y) * (x + y) = -(x + y) * (x - y) :=
by sorry

end optionA_is_square_difference_l306_306039


namespace find_a_l306_306701

theorem find_a (a : ℝ) 
  (h1 : ∀ x y : ℝ, 2*x + y - 2 = 0)
  (h2 : ∀ x y : ℝ, a*x + 4*y + 1 = 0)
  (perpendicular : ∀ (m1 m2 : ℝ), m1 = -2 → m2 = -a/4 → m1 * m2 = -1) :
  a = -2 :=
sorry

end find_a_l306_306701


namespace least_number_subtracted_divisible_by_5_l306_306163

theorem least_number_subtracted_divisible_by_5 : 
  ∃ (n : ℕ), (n = 4) ∧ (568219 - n) % 5 = 0 :=
by {
  use 4,
  split,
  { refl },
  { 
    have h : 568219 % 5 = 4 := rfl,
    rw [←nat.sub_add_cancel (le_of_lt (nat.mod_lt 568219 (zero_lt_of_pos dec_trivial))), h, nat.add_mod],
    norm_num
  }
}

end least_number_subtracted_divisible_by_5_l306_306163


namespace volume_ratio_l306_306866

-- Define the structure of the cube, planes, and necessary points
structure Point := (x y z : ℝ)
structure Cube := (edge_length : ℝ) (A B C D E F G H : Point)

axiom midpoint (p1 p2 : Point) : Point

-- Define specific points in the cube based on edge length
def default_cube : Cube :=
  { edge_length := 2,
    A := ⟨0, 0, 0⟩,
    B := ⟨2, 0, 0⟩,
    C := ⟨2, 2, 0⟩,
    D := ⟨0, 2, 0⟩,
    E := ⟨0, 0, 2⟩,
    F := ⟨2, 0, 2⟩,
    G := ⟨2, 2, 2⟩,
    H := ⟨0, 2, 2⟩ }

-- Defining the conditions of the problem
def plane1 := (midpoint default_cube.B default_cube.G)
def plane2 := (midpoint default_cube.A default_cube.B, midpoint default_cube.A default_cube.D, midpoint default_cube.H default_cube.E, midpoint default_cube.G default_cube.H)

-- Proof statement
theorem volume_ratio (cube : Cube) (plane1: Point) (plane2: (Point × Point × Point × Point)) :
  (7/17) = volume_smallest_piece / volume_largest_piece := by
  sorry 

end volume_ratio_l306_306866


namespace vertical_bisecting_line_of_circles_l306_306022

theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x + 6 * y + 2 = 0 ∨ x^2 + y^2 + 4 * x - 2 * y - 4 = 0) →
  (4 * x + 3 * y + 5 = 0) :=
sorry

end vertical_bisecting_line_of_circles_l306_306022


namespace fruits_fall_in_14_days_l306_306886

theorem fruits_fall_in_14_days :
  ∃ D : ℕ, (∃ (n1 n2 : ℕ), 
    ((∑ k in Finset.range (n1 + 1), k + 1 = 55) ∧
     (∑ k in Finset.range (n2 + 1), (if k < 5 then 1 else 0) = 5) ∧ 
     (n1 + n2 = 14))) := sorry

end fruits_fall_in_14_days_l306_306886


namespace angle_MPF_proof_l306_306765

-- Define the setup and conditions
variables {A B C P E D Q F M : Type}
variable triangle_ABC : Triangle A B C
variable is_point_interior : P ∈ interior (Triangle A B C)
variable is_midpoint : midpoint B C = M
variables (AB AC : ℝ) (angle_PBC : ℝ)
variables (on_AB : E ∈ segment A B) (on_AC : D ∈ segment A C)
variables (angle_BPE angle_EPA angle_APD angle_DPC : ℝ)
variables (BD CE : Line) (BD_meets_CE_at_Q : Q ∈ (intersection (Line B D) (Line C E))) 
variables (AQ_meets_BC_at_F : F ∈ (intersection (Line A Q) (Line B C))) 

-- Assign given values
axiom AB_val : AB = 1
axiom AC_val : AC = 2
axiom angle_PBC_val : angle_PBC = 70
axiom angle_BPE_val : angle_BPE = 75
axiom angle_EPA_val : angle_EPA = 75
axiom angle_APD_val : angle_APD = 60
axiom angle_DPC_val : angle_DPC = 60

-- Define the problem to prove: angle MPF = 15 degrees
theorem angle_MPF_proof : ang MPF = 15 := by
  sorry

end angle_MPF_proof_l306_306765


namespace complement_union_of_M_and_N_l306_306688

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306688


namespace trapezoid_circle_center_l306_306892

theorem trapezoid_circle_center 
  (EF GH : ℝ)
  (FG HE : ℝ)
  (p q : ℕ) 
  (rel_prime : Nat.gcd p q = 1)
  (EQ GH : ℝ)
  (h1 : EF = 105)
  (h2 : FG = 57)
  (h3 : GH = 22)
  (h4 : HE = 80)
  (h5 : EQ = p / q)
  (h6 : p = 10)
  (h7 : q = 1) :
  p + q = 11 :=
by
  sorry

end trapezoid_circle_center_l306_306892


namespace value_of_M_l306_306714

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end value_of_M_l306_306714


namespace complement_union_eq_l306_306584

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306584


namespace solve_fraction_equation_l306_306487

theorem solve_fraction_equation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  1 / (x + 1) + 1 / (x + 2) = 1 / x ↔ x = sqrt 2 ∨ x = -sqrt 2 :=
by
  sorry

end solve_fraction_equation_l306_306487


namespace second_worker_time_on_DE_l306_306251

def pavingTime (total_time_hours : ℕ) (speed_ratio : ℝ) (d1 : ℝ) : Prop :=
  let total_time_minutes := total_time_hours * 60
  let d2 := speed_ratio * d1
  let time_spent_DE := (total_time_minutes / d2) * (d1 * 0.1)
  time_spent_DE = 45

theorem second_worker_time_on_DE (t : ℕ) (v1 v2 d1 d2 : ℝ) (h_t : t = 9) (h_v2 : v2 = 1.2 * v1) (h_d2 : d2 = 1.2 * d1) (h_totaltime : 9 * 60 = t * 60) : pavingTime 9 1.2 d1 :=
by
  have total_time_minutes := 9 * 60
  have d2 := 1.2 * d1
  have time_spent_DE := (total_time_minutes / d2) * (d1 * 0.1)
  show time_spent_DE = 45
  sorry

end second_worker_time_on_DE_l306_306251


namespace cannot_determine_degree_from_A_P_l306_306781

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l306_306781


namespace axis_of_symmetry_range_of_t_l306_306755

section
variables (a b m n p t : ℝ)

-- Assume the given conditions
def parabola (x : ℝ) : ℝ := a * x ^ 2 + b * x

-- Part (1): Find the axis of symmetry
theorem axis_of_symmetry (h_a_pos : a > 0) 
    (hM : parabola a b 2 = m) 
    (hN : parabola a b 4 = n) 
    (hmn : m = n) : 
    -b / (2 * a) = 3 := 
  sorry

-- Part (2): Find the range of values for t
theorem range_of_t (h_a_pos : a > 0) 
    (hP : parabola a b (-1) = p)
    (axis : -b / (2 * a) = t) 
    (hmn_neg : m * n < 0) 
    (hmpn : m < p ∧ p < n) :
    1 < t ∧ t < 3 / 2 := 
  sorry
end

end axis_of_symmetry_range_of_t_l306_306755


namespace monotonically_decreasing_interval_l306_306551

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 6) + Real.cos (2 * x)

theorem monotonically_decreasing_interval :
  ∃ (a b : ℝ), a = π / 12 ∧ b = 7 * π / 12 ∧ 
  ∀ x y : ℝ, (a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y) → (f y ≤ f x) :=
begin
  sorry
end

end monotonically_decreasing_interval_l306_306551


namespace complement_union_l306_306618

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306618


namespace sum_a_leq_9900_l306_306273

-- Define the function p(x)
noncomputable def p (x : ℤ) : ℤ :=
  x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 7) *
  (x - 8) * (x - 9) * (x - 10) * (x - 11) * (x - 12) * (x - 13) * (x - 14) * (x - 15) * (x - 16) *
  (x - 17) * (x - 18) * (x - 19) * (x - 20)

-- Define the given constants
def C : ℤ := 100 * 99 * 98 * 97 * 96 * 95 * 94 * 93 * 92 * 91 * 90 * 89 * 88 * 87 * 86 * 85 * 84 * 83 * 82 * 81 * 80 * 79

-- The given conditions and the statement to prove
theorem sum_a_leq_9900 (a : ℕ → ℤ) (h_nonneg : ∀ i, 1 ≤ i → i ≤ 100 → 0 ≤ a i)
  (h_inequality : ∑ i in Finset.range 100, p (a (i + 1)) ≤ C ) : 
  ∑ i in Finset.range 100, a (i + 1) ≤ 9900 :=
sorry

end sum_a_leq_9900_l306_306273


namespace complement_union_of_M_and_N_l306_306695

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306695


namespace calculate_p_op_l306_306827

def op (x y : ℝ) := x * y^2 - x

theorem calculate_p_op (p : ℝ) : op p (op p p) = p^7 - 2*p^5 + p^3 - p :=
by
  sorry

end calculate_p_op_l306_306827


namespace jesters_on_stilts_count_l306_306104

theorem jesters_on_stilts_count :
  ∃ j e : ℕ, 3 * j + 4 * e = 50 ∧ j + e = 18 ∧ j = 22 :=
by 
  sorry

end jesters_on_stilts_count_l306_306104


namespace Aarti_work_days_l306_306983

theorem Aarti_work_days (single_work_days : ℕ) (factor : ℕ) : single_work_days = 8 → factor = 3 → (factor * single_work_days) = 24 :=
begin
  intros h1 h2,
  rw [h1, h2],
  norm_num,
end

end Aarti_work_days_l306_306983


namespace angle_between_vectors_l306_306186

noncomputable def vec_m := (-1 : ℝ, real.sqrt 3)
noncomputable def vec_n (n : ℝ × ℝ) := ∃ n1 n2 : ℝ, n = (n1, n2)

theorem angle_between_vectors {θ : ℝ} (n : ℝ × ℝ) 
  (h1 : vec_m = (-1, real.sqrt 3)) 
  (h2 : vec_m.1 * (vec_m.1 - n.1) + (vec_m.2) * (vec_m.2 - n.2) = 5)
  (h3 : n.1 * (vec_m.1 + n.1) + n.2 * (vec_m.2 + n.2) = 0) :
  θ = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_l306_306186


namespace total_perimeter_is_8_l306_306452

def side_length_square := 4 / Real.pi
def radius_semicircle := side_length_square / 2
def full_circumference := 2 * Real.pi * radius_semicircle
def semicircle_circumference := full_circumference / 2
def total_perimeter := 4 * semicircle_circumference

theorem total_perimeter_is_8 :
  total_perimeter = 8 :=
by
  unfold side_length_square radius_semicircle full_circumference semicircle_circumference total_perimeter
  sorry

end total_perimeter_is_8_l306_306452


namespace quadratic_polynomial_correction_l306_306494

noncomputable def q (x : ℝ) : ℝ := -25 / 2 * x^2 + 161 / 2 * x + 371

theorem quadratic_polynomial_correction : 
  (q (-3) = 17) ∧ (q 2 = -1) ∧ (q 4 = 10) :=
by
  -- Definitions put in place
  let q := fun x : ℝ => -25 / 2 * x^2 + 161 / 2 * x + 371
  
  -- Proof conditions
  have h1 : q (-3) = 17 :=
    by
    calc q (-3) = -25 / 2 * (-3)^2 + 161 / 2 * (-3) + 371 : by rfl
             ... = -25 / 2 * 9 + 161 / 2 * (-3) + 371  : by norm_num
             ... = -225 / 2 - 483 / 2 + 371            : by norm_num
             ... = 17                                   : by norm_num
  have h2 : q 2 = -1 :=
    by
    calc q 2 = -25 / 2 * 2^2 + 161 / 2 * 2 + 371 : by rfl
            ... = -25 / 2 * 4 + 161 / 2 * 2 + 371  : by norm_num
            ... = -100 / 2 + 161 / 2 * 2 + 371     : by norm_num
            ... = -50 + 161 + 371                  : by norm_num
            ... = -1                                : by norm_num  
  have h3 : q 4 = 10 :=
    by
    calc q 4 = -25 / 2 * 4^2 + 161 / 2 * 4 + 371 : by rfl
            ... = -25 / 2 * 16 + 161 / 2 * 4 + 371 : by norm_num
            ... = -400 / 2 + 161 + 371             : by norm_num
            ... = -200 + 161 + 371                 : by norm_num
            ... = 10                               : by norm_num

  exact ⟨ h1, h2, h3 ⟩

end quadratic_polynomial_correction_l306_306494


namespace binomial_16_12_l306_306122

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l306_306122


namespace find_roots_and_m_l306_306535

theorem find_roots_and_m (m a : ℝ) (h_root : (-2)^2 - 4 * (-2) + m = 0) :
  m = -12 ∧ a = 6 :=
by
  sorry

end find_roots_and_m_l306_306535


namespace b_seq_geometric_S_seq_geometric_find_smallest_a_seq_l306_306524

-- Definitions for sequences and terms
def a_seq (a : ℝ) : ℕ → ℝ
| 1       := 2 * a + 1
| (n + 1) := 2 * a_seq n + (n + 1) * (n + 1) - 4 * (n + 1) + 2

def b_seq (a : ℝ) : ℕ → ℝ
| 1       := a
| (n + 1) := a_seq a (n + 1) + (n + 1) * (n + 1)

def S_seq (a : ℝ) : ℕ → ℝ
| 1       := a
| (n + 1) := S_seq n + b_seq a (n + 1)

-- Problem Statements
-- (1) Proving b_n is a geometric sequence with ratio 2 starting from the second term
theorem b_seq_geometric {a : ℝ} {n : ℕ} (h : n ≥ 2) : b_seq a (n + 1) = 2 * b_seq a n :=
sorry

-- (2) Proving a = -4/3 for S_n to be a geometric sequence
theorem S_seq_geometric (a : ℝ) (h : ∀ n ≥ 2, S_seq a n / S_seq a (n - 1) = 4 / 3) : a = -4 / 3 :=
sorry

-- (3) Finding the smallest term of a_seq when a > 0, categorized by intervals
theorem find_smallest_a_seq (a : ℝ) (h : a > 0) : 
  ∃ n : ℕ, a_seq a n = min (min (8 * a - 1) (4 * a)) (2 * a + 1) :=
sorry

end b_seq_geometric_S_seq_geometric_find_smallest_a_seq_l306_306524


namespace fractions_problem_l306_306724

theorem fractions_problem (x y : ℚ) (hx : x = 2 / 3) (hy : y = 3 / 2) :
  (1 / 3) * x^5 * y^6 = 3 / 2 := by
  sorry

end fractions_problem_l306_306724


namespace correct_option_l306_306991

-- Definitions corresponding to the conditions
def option_A : Prop := ¬(∃ S, ∀ x ∈ S, abs x < ε) -- "very small" is not definite (Note: This is a more formal interpretation regarding the ambiguity)
def option_B : Prop := {x | x * (x - 1)^2 = 0} = {1, 0, 1}
def option_C : Prop := {1, a, b, c} = {c, b, a, 1}
def option_D : Prop := ∀ S, ∅ ⊊ S

-- The proof problem we need to solve
theorem correct_option : ¬option_A ∧ ¬option_B ∧ option_C ∧ ¬option_D := 
by
  -- Insert proof here
  sorry

end correct_option_l306_306991


namespace normal_dist_symmetry_l306_306242

noncomputable theory

open ProbabilityTheory MeasureTheory

variables {X : Type} [MeasureSpace X]

def X_dist : ProbabilityMassFunction X := 
  classical.arbitrary (ProbabilityMassFunction X)

axiom hX : X_dist.hasLeftIntegral (set.univ X)
  (ProbabilityDensityFunction.normal X_dist 1 2)

/-- Given a random variable X that follows a normal distribution N(1,4),
and probabilities P(0 < X < 3) = m and P(-1 < X < 2) = n,
we prove that m = n. -/
theorem normal_dist_symmetry (X : ℝ) (hX : NormalDist X 1 2)
    (hm : P (0 < X ∧ X < 3) = m) (hn : P (-1 < X ∧ X < 2) = n) : m = n := 
sorry

end normal_dist_symmetry_l306_306242


namespace complement_union_of_M_and_N_l306_306696

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306696


namespace increasing_sequences_modulo_l306_306343

theorem increasing_sequences_modulo : 
  let m := 515
  in let remainder := m % 1000
  in remainder = 515 :=
by
  sorry

end increasing_sequences_modulo_l306_306343


namespace sin2theta_plus_cos2theta_one_l306_306509

theorem sin2theta_plus_cos2theta_one (theta : ℝ) :
  let a := (1, -2)
      b := (Real.sin theta, Real.cos theta) in
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.sin (2 * theta) + Real.cos theta ^ 2 = 1 := 
by
  intros a b hab
  let sin_theta := b.1
  let cos_theta := b.2
  have h1 : sin_theta = 2 * cos_theta, from sorry
  have h2 : Real.sin (2 * theta) = 2 * sin_theta * cos_theta, from sorry
  have h3 : Real.sin_theta ^ 2 + cos_theta ^ 2 = 1, from sorry
  sorry

end sin2theta_plus_cos2theta_one_l306_306509


namespace vertex_of_quadratic_l306_306879

theorem vertex_of_quadratic : 
  (∃ (h k : ℝ), ∀ x : ℝ, (y = (1/2) * (x - 4) ^ 2 + 5) → (h = 4 ∧ k = 5)) := 
begin
  sorry
end

end vertex_of_quadratic_l306_306879


namespace magnitude_of_z_l306_306517

noncomputable def z : ℂ := (1 + complex.i) / (2 - 2 * complex.i)

theorem magnitude_of_z : complex.abs z = 1 / 2 := 
by
  -- proof goes here
  sorry

end magnitude_of_z_l306_306517


namespace area_of_region_b_l306_306455

open Complex Real Set

noncomputable def region_in_complex_plane : Set ℂ :=
  { z : ℂ | ∀ (x y : ℝ), z = x + y * Complex.I ∧ 
    (0 ≤ x / 50) ∧ (x / 50 ≤ 1) ∧ (0 ≤ y / 50) ∧ (y / 50 ≤ 1) ∧
    (0 ≤ 50 * x / (x^2 + y^2)) ∧ (50 * x / (x^2 + y^2) ≤ 1) ∧ 
    (0 ≤ 50 * y / (x^2 + y^2)) ∧ (50 * y / (x^2 + y^2) ≤ 1) }

theorem area_of_region_b : measure_theory.measure.restrict measure_theory.lebesgue region_in_complex_plane = 1875 - 312.5 * real.pi :=
sorry

end area_of_region_b_l306_306455


namespace garden_area_difference_l306_306955

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l306_306955


namespace prove_new_rate_of_interest_l306_306393

variables (P R1 T1 SI T2 R2 : ℝ)
variables (simple_interest_calculator : ℝ → ℝ → ℝ → ℝ)

-- Conditions
def condition1 := (SI = 840)
def condition2 := (R1 = 5)
def condition3 := (T1 = 8)
def condition4 := (T2 = 5)
def condition5 := (simple_interest_calculator P R1 T1 = SI)

-- New Rate Calculation
noncomputable def new_rate_of_interest (SI P T2 : ℝ) : ℝ :=
  (SI * 100) / (P * T2)

theorem prove_new_rate_of_interest :
  condition1 → condition2 → condition3 → condition4 → condition5 →
  new_rate_of_interest SI P T2 = 8 :=
by
  intros _ _ _ _ _
  sorry

end prove_new_rate_of_interest_l306_306393


namespace max_value_of_quadratic_expression_l306_306899

theorem max_value_of_quadratic_expression (s : ℝ) : ∃ x : ℝ, -3 * s^2 + 24 * s - 8 ≤ x ∧ x = 40 :=
sorry

end max_value_of_quadratic_expression_l306_306899


namespace perimeter_of_smaller_rectangle_l306_306352

def large_square_perimeter := 256
def small_square_side_length : ℕ := 64 / 4
def diagonal_length := real.sqrt (small_square_side_length^2 + small_square_side_length^2)
def small_rectangle_perimeter := 2 * (small_square_side_length + diagonal_length / 2)

theorem perimeter_of_smaller_rectangle :
  small_rectangle_perimeter = 32 + 16 * real.sqrt 2 := by
  sorry

end perimeter_of_smaller_rectangle_l306_306352


namespace max_subset_elements_l306_306710

theorem max_subset_elements (p : ℕ) (h_prime : Prime p) (h_p : p = 2^16 + 1) :
  ∃ S : Finset ℕ, (∀ a b ∈ S, a ≠ b → a^2 ≠ b [MOD p]) ∧ S.card = 43691 :=
by
  sorry

end max_subset_elements_l306_306710


namespace problem_condition_f_phi_summation_l306_306549

theorem problem_condition_f_phi_summation (A ω φ : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = A * (Real.sin (ω * x + φ))^2) ∧
  A > 0 ∧ ω > 0 ∧ 0 < φ ∧ φ < π / 2 ∧
  (∃ x, f x = 2) ∧
  (∀ n : ℤ, n ≠ 0 → Mod (2 * π / ω) = 2) ∧
  f 1 = 2 →
  φ = π / 4 ∧ (Finset.sum (Finset.range 2008) f) = 2008 := 
by sorry

end problem_condition_f_phi_summation_l306_306549


namespace problem_k_value_maximum_profit_l306_306416

def C (x : ℝ) : ℝ := 3 + x

noncomputable def S (x : ℝ) (k : ℝ) : ℝ := 
  if h : (0 < x ∧ x < 6) then 3 * x + k / (x - 8) + 5
  else if 6 ≤ x then 14
  else 0

def L (x : ℝ) (k : ℝ) : ℝ := S x k - C x

theorem problem_k_value :
  (L 2 k = 3) → k = 18 := sorry

theorem maximum_profit :
  ∃ x_max : ℝ, (0 < x_max) ∧ (x_max < 6) ∧ L x_max 18 = 6 := sorry

end problem_k_value_maximum_profit_l306_306416


namespace problem_I_problem_II_problem_III_l306_306508

theorem problem_I (θ : ℝ) (h₁: cos θ = sqrt 5 / 5) (h₂: 0 < θ ∧ θ < π / 2) : 
  sin θ = 2 * sqrt 5 / 5 :=
sorry

theorem problem_II (θ : ℝ) (h₁: cos θ = sqrt 5 / 5) (h₂ : 0 < θ ∧ θ < π / 2) : 
  cos (2 * θ) = -3 / 5 :=
sorry

theorem problem_III (θ φ : ℝ) (h₁: cos θ = sqrt 5 / 5) (h₂: 0 < θ ∧ θ < π / 2) 
  (h₃: sin (θ - φ) = sqrt 10 / 10) (h₄: 0 < φ ∧ φ < π / 2) : 
  cos φ = sqrt 2 / 2 :=
sorry

end problem_I_problem_II_problem_III_l306_306508


namespace min_d1_plus_d2_l306_306536

noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ :=
(abs (a * px + b * py + c)) / (sqrt (a ^ 2 + b ^ 2))

theorem min_d1_plus_d2 :
  let P := (2, 2) in -- let P be a point on the parabola y^2 = 8x
  let F := (2, 0) in -- focus of the parabola y^2 = 8x is (2, 0)
  let d1 := abs (2 - 2) / sqrt (1 + 0 ^ 2) in -- distance from P to axis x
  let d2 := distance_point_to_line 2 2 4 3 8 in -- distance from P to the line 4x + 3y+ 8 = 0
  (d1 + d2) = 16 / 5 := 
by
  sorry

end min_d1_plus_d2_l306_306536


namespace range_of_magnitude_l306_306220

def vector_magnitude {α : Type} [NormedAddMonoid α] (u v : α) : ℝ :=
  ∥u + v∥

def a : ℝ × ℝ := (1, 0)
def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def θ_in_range (θ : ℝ) : Prop := θ ∈ Icc (-Real.pi / 2) (Real.pi / 2)

theorem range_of_magnitude (θ : ℝ) (h : θ_in_range θ) :
  ∃ lower upper, ∀ θ, θ_in_range θ →
  vector_magnitude a (b θ) ∈ Icc lower upper ∧ lower = Real.sqrt 2 ∧ upper = 2 :=
by sorry

end range_of_magnitude_l306_306220


namespace complement_union_M_N_l306_306631

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306631


namespace hyperbola_eccentricity_is_two_l306_306726

-- Define the focus of the parabola y^2 = 8x
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the hyperbola equation parameters
variables (a b c m : ℝ) (h_hyperbola : 1 = c^2 / a^2 + 1 / b^2)
           (h_focus : c = 2)    -- Since one focus of the hyperbola coincides with that of the parabola
           (h_b_squared : b^2 = 3)
           (h_m : m = 4)

-- Define the eccentricity of the hyperbola 
def hyperbola_eccentricity (a c : ℝ) : ℝ := c / a

-- The theorem to prove that the eccentricity of the hyperbola is 2
theorem hyperbola_eccentricity_is_two (h_cond : hyperbola_eccentricity a c = 2) : 
    hyperbola_eccentricity a c = 2 :=
by
  sorry

end hyperbola_eccentricity_is_two_l306_306726


namespace min_positive_S_l306_306151

noncomputable def main : ℕ :=
  let a : fin 150 → ℤ := fun i => if i.val < 82 then 1 else -1 
  let sum_a := finset.univ.sum (λ i : fin 150, a i)
  let sum_a_squared := sum_a ^ 2
  (sum_a_squared - 150) / 2

theorem min_positive_S :
  ∃ S : ℕ, S = 23 ∧ ∀ (a : fin 150 → ℤ), (∀ i, a i = 1 ∨ a i = -1) →
    S ≤ (∑ i in finset.univ, ∑ j in finset.Ico 0 i, a i * a j) :=
sorry

end min_positive_S_l306_306151


namespace probability_rolls_divisible_by_4_and_ab_divisible_by_4_l306_306965

-- Definition of a fair 8-sided die and the possible outcomes
def fair_eight_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Definition of the probabilities of rolling certain values
def prob_divisible_by_4 : ℝ := 1 / 4

-- Definition of numbers divisible by 4 on an 8-sided die
def divisible_by_4 : Set ℕ := fair_eight_sided_die.filter (λ n, n % 4 = 0)

-- Define probability calculation
noncomputable def combined_prob : ℝ := prob_divisible_by_4 * prob_divisible_by_4

-- The Lean 4 statement to prove the probability question
theorem probability_rolls_divisible_by_4_and_ab_divisible_by_4 :
  combined_prob = 1 / 16 :=
by
  sorry

end probability_rolls_divisible_by_4_and_ab_divisible_by_4_l306_306965


namespace problem_a_problem_b_problem_c_l306_306925

variables (n k : ℕ) (P Q : set (vector (fin (k + 1)) n))

-- Condition for elements in P and Q
def element_condition (p q : vector (fin (k + 1)) n) : Prop :=
  ∃ m : fin n, p.nth m = q.nth m

-- Problem (a) for k = 2 and any natural n
theorem problem_a (hPk: ∀ p ∈ P, p.val.forall (λ ai, ai < 2))
  (hQk: ∀ q ∈ Q, q.val.forall (λ ai, ai < 2))
  (h : ∀ p ∈ P, ∀ q ∈ Q, element_condition n 2 p q) :
  (∃ s : set (vector (fin 2) n), (s = P ∨ s = Q) ∧ s.card ≤ 2^(n-1)) :=
  sorry

-- Problem (b) for n = 2 and any natural k > 1
theorem problem_b (k_gt1 : 1 < k)
  (hP: ∀ p ∈ P, p.val.forall (λ ai, ai < k + 1))
  (hQ: ∀ q ∈ Q, q.val.forall (λ ai, ai < k + 1))
  (h : ∀ p ∈ P, ∀ q ∈ Q, element_condition 2 k p q) :
  (∃ s : set (vector (fin (k + 1)) 2), (s = P ∨ s = Q) ∧ s.card ≤ k) :=
  sorry

-- Problem (c) for arbitrary natural n and k > 1
theorem problem_c (k_gt1 : 1 < k)
  (hP: ∀ p ∈ P, p.val.forall (λ ai, ai < k + 1))
  (hQ: ∀ q ∈ Q, q.val.forall (λ ai, ai < k + 1))
  (h : ∀ p ∈ P, ∀ q ∈ Q, element_condition n k p q) :
  (∃ s : set (vector (fin (k + 1)) n), (s = P ∨ s = Q) ∧ s.card ≤ 2*k^(n-1)) :=
  sorry

end problem_a_problem_b_problem_c_l306_306925


namespace sum_of_side_lengths_l306_306349

theorem sum_of_side_lengths (A B C : ℕ) (hA : A = 10) (h_nat_B : B > 0) (h_nat_C : C > 0)
(h_eq_area : B^2 + C^2 = A^2) : B + C = 14 :=
sorry

end sum_of_side_lengths_l306_306349


namespace complement_union_l306_306604

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306604


namespace lcm_condition_proof_l306_306515

theorem lcm_condition_proof (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h3 : ∀ i, 1 ≤ i → i ≤ n → a i ≤ 2 * n)
  (h4 : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) :
  a 1 > n * 2 / 3 := 
sorry

end lcm_condition_proof_l306_306515


namespace find_students_just_passed_l306_306049

theorem find_students_just_passed (total_students : ℕ) (first_division_percent : ℕ) (second_division_percent : ℕ) (no_students_failed : Prop) :
  total_students = 300 →
  first_division_percent = 30 →
  second_division_percent = 54 →
  no_students_failed →
  let first_division := total_students * first_division_percent / 100 in
  let second_division := total_students * second_division_percent / 100 in
  total_students - (first_division + second_division) = 48 :=
begin
  sorry
end

end find_students_just_passed_l306_306049


namespace midpoint_correct_l306_306756

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

theorem midpoint_correct :
  let A := { x := 1, y := -2 }
  let B := { x := 3, y := 0 }
  midpoint A B = { x := 2, y := -1 } := by
  sorry

end midpoint_correct_l306_306756


namespace remainder_3012_div_96_l306_306031

theorem remainder_3012_div_96 : 3012 % 96 = 36 :=
by 
  sorry

end remainder_3012_div_96_l306_306031


namespace math_problem_l306_306231

variables {R : Type*} [Ring R] (x y z : R)

theorem math_problem (h : x * y + y * z + z * x = 0) : 
  3 * x * y * z + x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = 0 :=
by 
  sorry

end math_problem_l306_306231


namespace percentage_error_l306_306973

theorem percentage_error (x : ℝ) (hx : 0 < x) : 
  let correct_result := 4 * x
  let erroneous_result := x / 8
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  round error_percentage = 97 := 
by
  have correct_result_eq : correct_result = 4 * x := rfl
  have erroneous_result_eq : erroneous_result = x / 8 := rfl
  have error_eq : error = abs (4 * x - x / 8) := rfl
  have error_eq_simplified : error = 31 * x / 8 := by 
    calc
      abs (4 * x - x / 8)
          = abs ((32 * x / 8) - x / 8)      : by rw [four_eq_32_div_8, sub_div]
      ... = abs (31 * x / 8)                : by rw [sub_eq_add_neg, add_comm, add_neg_thm]
      ... = 31 * x / 8                      : abs_of_nonneg (by linarith)
  have error_percentage_eq : error_percentage = (31 / 32) * 100 := by
    calc
      error / correct_result * 100
          = (31 * x / 8) / (4 * x) * 100   : by rw [error_eq_simplified, correct_result_eq]
      ... = (31 / 32) * 100                : by field_simp
  have error_percentage_exact : error_percentage = 96.875 := by 
    rw [error_percentage_eq]
    norm_num
  exact_mod_cast error_percentage
  sorry

end percentage_error_l306_306973


namespace students_who_like_both_channels_l306_306067

theorem students_who_like_both_channels (total_students : ℕ) 
    (sports_channel : ℕ) (arts_channel : ℕ) (neither_channel : ℕ)
    (h_total : total_students = 100) (h_sports : sports_channel = 68) 
    (h_arts : arts_channel = 55) (h_neither : neither_channel = 3) :
    ∃ x, (x = 26) :=
by
  have h_at_least_one := total_students - neither_channel
  have h_A_union_B := sports_channel + arts_channel - h_at_least_one
  use h_A_union_B
  sorry

end students_who_like_both_channels_l306_306067


namespace solve_for_x_l306_306024

-- conditions
def dimensions (x : ℝ) : ℝ × ℝ := (x - 3, 3x + 4)
def area (x : ℝ) : ℝ := 10 * x

-- proof we want to carry out
theorem solve_for_x (x : ℝ) :
  (dimensions x).fst * (dimensions x).snd = area x → x ≈ 5.7 :=
by
  -- given (x - 3)(3x + 4) = 10x
  -- Expanding and simplifying the equation:
  -- 3x^2 + 4x - 9x - 12 = 10x
  -- 3x^2 - 5x - 12 = 10x
  -- 3x^2 - 15x - 12 = 0
  -- Solving the quadratic equation yields x ≈ 5.7
  sorry

end solve_for_x_l306_306024


namespace families_cross_river_l306_306269

/-
Define the families and their rowing ability.
-/
inductive Person
| Ivan | Irina | Mikhail | Maria | Petr | Polina
| I | i | M | m | P | p

open Person

def canRow : Person → Prop
| Ivan | Mikhail | Polina | I | M | p := true
| _ := false

def family (p : Person) : ℕ
| I | i := 1
| M | m := 2
| P | p := 3
| _ := 0

def isMale : Person → Prop
| I | M | P := true
| _ := false

/-
Define the initial and final states.
-/
inductive Bank
| Left | Right

open Bank

structure State :=
(left : list Person)
(right : list Person)

def initialState : State :=
{ left := [I, i, M, m, P, p], right := [] }

def finalState : State :=
{ left := [], right := [I, i, m, M, P, p] }

/- 
State the conditions as constraints.
Constraints: A man cannot stay on shore or in the boat alone with another man's wife.
-/

def isSafe (bank : list Person) : Prop :=
  ∀ wives husbands, (wives ⊆ bank ∧ husbands ⊆ bank ∧ (∀ w, w ∈ wives → ¬isMale w)) → 
  (∀ w h, w ∈ wives → h ∈ husbands → isMale h → family w = family h)

def move (s : State) (p1 p2 : Person) (from to : Bank) (cR1 cR2 : Prop) (cS : Prop) : State :=
if from = Left ∧ to = Right ∧ cR1 ∧ cR2 ∧ cS then
{ left := list.erase (p1::(list.erase s.left p2)) p1, right := p1::(p2::s.right) }
else if from = Right ∧ to = Left ∧ cR1 ∧ cR2 ∧ cS then
{ left := p1::(p2::s.left), right := list.erase (p1::(list.erase s.right p2)) p1 }
else s

noncomputable def crossRiver : State -> Prop
| initialState := true -- initial state is initial
| finalState := true  -- final state is final
| s := sorry -- To define all the possible steps and intermediate states.

theorem families_cross_river (initialState : State) (finalState : State) : crossRiver initialState = finalState :=
sorry

end families_cross_river_l306_306269


namespace concurrency_KH_EM_BC_l306_306807

-- Definitions for points, lines, and circles
variables {ABC : Triangle} [isAcute ABC]
variables {Γ : Circle} (circ_ABC : Γ = circumcircle ABC)
variables {H : Point} (H_is_orthocenter : H = orthocenter ABC)
variables {K : Point} (K_on_circ : K ∈ Γ) (K_not_on_arc : ¬contains_arc Γ A K)
variables {L : Point} (L_reflect_AB : L = reflect K (line AB))
variables {M : Point} (M_reflect_BC : M = reflect K (line BC))
variables {E : Point} (E_second_intersect : second_intersection_point Γ (circle_through_points B L M) = E)

-- The theorem to prove
theorem concurrency_KH_EM_BC :
  concurrent (line_through_points K H) (line_through_points E M) (line BC) :=
sorry

end concurrency_KH_EM_BC_l306_306807


namespace length_of_chord_of_intersection_l306_306491

theorem length_of_chord_of_intersection
  (line_eq : ∀ x y : ℝ, x - 3 * y + 3 = 0)
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 10) :
  ∃ l : ℝ, l = sqrt 30 :=
by
  sorry

end length_of_chord_of_intersection_l306_306491


namespace garden_area_increase_l306_306946

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l306_306946


namespace binomial_sixteen_twelve_eq_l306_306125

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l306_306125


namespace sequence_integer_count_l306_306098

theorem sequence_integer_count (a₀ : ℕ) (h₀ : a₀ = 8505) (h₁ : ∀ n, a₀ / 5^n ∈ ℕ) : 
  ∃ n, n = 3 ∧ (a₀ / 5^n) ∉ ℕ :=
by
  sorry

end sequence_integer_count_l306_306098


namespace painted_cubes_l306_306133

-- conditions
def unit_cube : Type := fin 1 ⊗ fin 1 ⊗ fin 1 -- each side of 1-inch cube
def larger_cube (n : ℕ) : Type := fin n ⊗ fin n ⊗ fin n -- general n-side cube

-- problem definitions
noncomputable def total_cubes_in_larger_cube (n : ℕ) : ℕ := n^3

-- given 22 cubes have no paint
def interior_cubes_unpainted : ℕ := 22

-- proof problem
theorem painted_cubes (large_side_len n : ℕ) (h : n^3 = 22) (interior_len : ℕ) (H : large_side_len = interior_len + 2) :
  total_cubes_in_larger_cube large_side_len - interior_cubes_unpainted = 42 :=
begin
  -- Problem statement only, the steps are handled in the proof
  sorry
end

end painted_cubes_l306_306133


namespace sum_of_distinct_prime_factors_296352_l306_306907

theorem sum_of_distinct_prime_factors_296352 :
  let p1 := 2,
      p2 := 3,
      p3 := 7,
      n := 296352 in
  (∃ (a b c d : ℕ), n = p1^a * (p3 * (p2^b * p3^c * (p3^d))))
  → p1 + p2 + p3 = 12 :=
by
  intros p1 p2 p3 n h
  simp [p1, p2, p3, h]
  sorry

end sum_of_distinct_prime_factors_296352_l306_306907


namespace magnitude_of_v_l306_306320

noncomputable theory
open Complex Real

theorem magnitude_of_v {u v : ℂ} (h1 : u * v = 20 - 15 * I) (h2 : abs u = sqrt 34) : 
  abs v = (25 * Real.sqrt 34) / 34 :=
by
  sorry

end magnitude_of_v_l306_306320


namespace units_digit_of_24_pow_4_add_42_pow_4_l306_306911

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end units_digit_of_24_pow_4_add_42_pow_4_l306_306911


namespace complement_union_l306_306615

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306615


namespace explicit_formula_a_value_exists_l306_306199

section
variables (a e : ℝ) (f : ℝ → ℝ)

-- Condition 1: f is defined to be odd on [-e, 0) ∪ (0, e]
-- Explicit formula for f(x) when x ∈ (0, e] is f(x) = ax + ln x.
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def fx_pos (x : ℝ) : ℝ := a * x + Real.log x

def fx_neg (x : ℝ) : ℝ := a * x - Real.log (-x)

-- Goal 1: Prove the explicit formula for f(x)
theorem explicit_formula (h : is_odd_function f)
  (hx_pos : ∀ x ∈ Ioo 0 e, f x = fx_pos a x)
  (hx_neg : ∀ x ∈ Ioo (-e) 0, f x = fx_neg a x) :
  ∀ x ∈ Ioo (-e) e, f x =
  if x < 0 then fx_neg a x else if 0 < x then fx_pos a x else f x :=
begin
  intros x hx,
  cases lt_or_ge x 0 with h₁ h₂,
  { -- case x < 0
    rw hx_neg x ⟨h₁, hx.2⟩,
    simp only [if_true, if_pos h₁] },
  cases lt_or_eq_of_le h₂ with h₃ h₄,
  { -- case 0 < x
    rw hx_pos x ⟨le_of_not_lt h₁, h₃⟩,
    simp only [if_false, not_false_iff, if_true, h₃] },
  { -- case x = 0, not needed since x ∈ (-e, e)
    linarith }
end

-- Goal 2: Prove there exists an a = -e^2 s.t. f(x) on [-e, 0) has min value 3.
noncomputable def derivative (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  a - (1 / x)

theorem a_value_exists (h : is_odd_function f)
  (hx : ∀ x ∈ Ioo 0 e, f x = fx_pos a x) :
  (∃ a : ℝ, 
      (a = -Real.exp 2) ∧ 
      (∀ x ∈ Ioo (-e) 0, f x = fx_neg a x) ∧ 
      (∃ y ∈ Ioc (-e) 0, f y = 3)) :=
begin
  use -Real.exp 2,
  split,
  { refl },
  split,
  { intros x hx_neg,
    rw fx_neg,
    sorry },
  { use (1 / -Real.exp 2),
    split;
    simp only [fx_neg];
    sorry }
end

end

end explicit_formula_a_value_exists_l306_306199


namespace g_is_odd_l306_306464

def g (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  intros x
  -- Provide the steps towards the proof which we skip here
  sorry

end g_is_odd_l306_306464


namespace smallest_initial_coins_l306_306075

-- Definitions based on conditions:
def pirates_count := 15

def pirate_fraction (k : Nat) : Rational := (k + 1) / (pirates_count + k)

def coins_left_for_cabin_boy := 5

def is_whole_number (n : Rational) : Prop := n.denom = 1

-- Main lean statement:
theorem smallest_initial_coins (x : Nat) (positive_number_of_coins : ∀ k, is_whole_number ((x - coins_left_for_cabin_boy) * pirate_fraction k))
  (received_coins_whole : is_whole_number ((x - coins_left_for_cabin_boy) * pirate_fraction pirates_count)) : 
  x = 24 :=
begin
  sorry
end

end smallest_initial_coins_l306_306075


namespace sum_of_a_b_l306_306518

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_b_l306_306518


namespace seating_arrangement_l306_306155

noncomputable def number_of_ways_to_seat_six_people (people : Finset ℕ) : ℕ :=
let n := people.card in
if h : n = 8 then
  let k := 6 in
  Nat.choose n (n - k) * (Nat.factorial k / k)
else
  0

theorem seating_arrangement (people : Finset ℕ) (h : people.card = 8) : 
  number_of_ways_to_seat_six_people people = 3360 := by
sorry

end seating_arrangement_l306_306155


namespace visible_factor_numbers_count_l306_306421

-- Define a function to check if a number is a visible factor number
def isVisibleFactorNumber (n : ℕ) : Prop :=
  let digits := [n / 100 % 10, n / 10 % 10, n % 10]
  digits.filter (λ d => d ≠ 0).all (λ d => n % d = 0)

-- Define a set of numbers from 200 to 250
def visibleFactorNumbers : List ℕ :=
  List.filter isVisibleFactorNumber (List.range' 200 51)

-- State the theorem
theorem visible_factor_numbers_count : visibleFactorNumbers.length = 16 := by
  sorry

end visible_factor_numbers_count_l306_306421


namespace set_intersection_complement_l306_306459

theorem set_intersection_complement :
  let U := {1, 2, 3, 4, 5}
  let M := {1, 4}
  let N := {1, 3, 5}
  let complement_U_M := {2, 3, 5} := by
  ∀ x : ℕ, x ∈ N ∩ (U \ M) ↔ x ∈ {3, 5} := by
    sorry

end set_intersection_complement_l306_306459


namespace area_increase_correct_l306_306938

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l306_306938


namespace cannot_determine_degree_from_char_set_l306_306772

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l306_306772


namespace anna_distance_in_6_minutes_l306_306435

def constant_rate_walk (d t : ℕ) : Prop :=
  ∃ r, r * t = d

theorem anna_distance_in_6_minutes (d : ℕ) (t₁ t₂ : ℕ)
  (h_const_rate : constant_rate_walk d t₁)
  (h_walk_600_4 : d = 600 ∧ t₁ = 4)
  (h_time_6 : t₂ = 6) :
  ∃ d₂, d₂ = 900 := 
by
  -- Definitions and conditions
  cases h_const_rate with r hr,
  cases h_walk_600_4 with hd ht,
  subst hd,
  subst ht,
  subst h_time_6,
  -- Calculate the distance
  use r * t₂,
  -- Show that the calculated distance is 900 meters
  -- Proof is omitted
  sorry

end anna_distance_in_6_minutes_l306_306435


namespace example_theorem_l306_306663

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306663


namespace cannot_determine_degree_from_char_set_l306_306773

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l306_306773


namespace esther_commute_time_l306_306479

theorem esther_commute_time :
  ∀ (distance_to_work distance_from_work : ℝ) (speed_to_work speed_from_work : ℝ),
  distance_to_work = 18 →
  speed_to_work = 45 →
  distance_from_work = 18 →
  speed_from_work = 30 →
  (distance_to_work / speed_to_work + distance_from_work / speed_from_work) = 1 :=
by
  intros distance_to_work distance_from_work speed_to_work speed_from_work h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end esther_commute_time_l306_306479


namespace acute_triangles_bound_l306_306506

theorem acute_triangles_bound (P : Finset Point) (hP : P.card = 100) (h_collinear : ∀ p1 p2 p3 ∈ P, ¬Collinear ℝ ({p1, p2, p3} : Set Point)) :
  ∃ K, K ≤ 0.70 ∧ ∀ S ⊆ P, (S.card = 3) → isAcuteTriangle S → (count {T | T ∈ P.triangles ∧ isAcuteTriangle T}.count / P.triangles.count ≤ K) :=
by
  sorry

end acute_triangles_bound_l306_306506


namespace surface_area_is_correct_l306_306453

structure CubicSolid where
  base_layer : ℕ
  second_layer : ℕ
  third_layer : ℕ
  top_layer : ℕ

def conditions : CubicSolid := ⟨4, 4, 3, 1⟩

theorem surface_area_is_correct : 
  (conditions.base_layer + conditions.second_layer + conditions.third_layer + conditions.top_layer + 7 + 7 + 3 + 3) = 28 := 
  by
  sorry

end surface_area_is_correct_l306_306453


namespace initial_number_2008_l306_306964

theorem initial_number_2008 
  (numbers_on_blackboard : ℕ → Prop)
  (x : ℕ)
  (Ops : ∀ x, numbers_on_blackboard x → (numbers_on_blackboard (2 * x + 1) ∨ numbers_on_blackboard (x / (x + 2)))) 
  (initial_apearing : numbers_on_blackboard 2008) :
  numbers_on_blackboard 2008 = true :=
sorry

end initial_number_2008_l306_306964


namespace cannot_determine_degree_from_A_P_l306_306782

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l306_306782


namespace binomial_sixteen_twelve_eq_l306_306126

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l306_306126


namespace checkerboard_black_squares_l306_306114

theorem checkerboard_black_squares (n : ℕ) (hn : n = 33) :
  let checkerboard := λ i j : ℕ, (i + j) % 2 = 0 in
  finset.card {ij : finset (ℕ × ℕ) | checkerboard ij.1 ij.2 ∧ ij.1 < n ∧ ij.2 < n } = 545 :=
by {
  apply finset.card,
  sorry
}

end checkerboard_black_squares_l306_306114


namespace eval_quadratic_abs_l306_306480

theorem eval_quadratic_abs (z : ℂ) (hz : z = 10 + 3 * Complex.i) :
  abs (z^2 + 8 * z + 85) = 4 * Real.sqrt 3922 := by
  -- Here you should include the proof
  sorry

end eval_quadratic_abs_l306_306480


namespace sum_a_greater_than_zero_l306_306274

theorem sum_a_greater_than_zero (a : ℕ → ℤ) (n : ℕ)
  (h1 : ∃ i, i ≤ n ∧ a i ≠ 0) 
  (h2 : ∀ i, i ≤ n → a i ≥ -1)
  (h3 : ∑ i in Finset.range (n + 1), 2^i * a i = 0) : 
  ∑ i in Finset.range (n + 1), a i > 0 :=
sorry

end sum_a_greater_than_zero_l306_306274


namespace ratio_is_1_over_14_l306_306433

noncomputable def side_length_large_triangle : ℝ := 12
noncomputable def side_length_small_triangle : ℝ := 3

noncomputable def area_large_triangle : ℝ :=
  (sqrt 3 / 4) * side_length_large_triangle ^ 2

noncomputable def area_small_triangle : ℝ :=
  (sqrt 3 / 4) * side_length_small_triangle ^ 2

noncomputable def area_remaining_polygon : ℝ :=
  area_large_triangle - 2 * area_small_triangle

noncomputable def ratio_of_areas : ℝ :=
  area_small_triangle / area_remaining_polygon

theorem ratio_is_1_over_14 :
  ratio_of_areas = 1 / 14 :=
by
  sorry

end ratio_is_1_over_14_l306_306433


namespace total_sticks_used_l306_306057

-- Definitions based on the conditions
def hexagons : Nat := 800
def sticks_for_first_hexagon : Nat := 6
def sticks_per_additional_hexagon : Nat := 5

-- The theorem to prove
theorem total_sticks_used :
  sticks_for_first_hexagon + (hexagons - 1) * sticks_per_additional_hexagon = 4001 := by
  sorry

end total_sticks_used_l306_306057


namespace find_m_digit_divisible_by_9_l306_306734

theorem find_m_digit_divisible_by_9 :
  ∃ (m : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ (7 + 4 + 6 + m + 8 + 1 + 3) % 9 = 0 ∧ m = 7 :=
begin
  sorry
end

end find_m_digit_divisible_by_9_l306_306734


namespace function_equation_l306_306484

noncomputable def f (n : ℕ) : ℕ := sorry

theorem function_equation (h : ∀ m n : ℕ, m > 0 → n > 0 →
  f (f (f m) + 2 * f (f n)) = m^2 + 2 * n^2) : 
  ∀ n : ℕ, n > 0 → f n = n := 
sorry

end function_equation_l306_306484


namespace stickers_count_l306_306009

variable (pages : ℕ) (stickers_A stickers_B stickers_C stickers_D : ℕ)

def total_stickers (stickers_per_page : ℕ) : ℕ := 
  stickers_per_page * pages

theorem stickers_count
  (h_pages : pages = 22)
  (h_stickers_A : stickers_A = 5)
  (h_stickers_B : stickers_B = 3)
  (h_stickers_C : stickers_C = 2)
  (h_stickers_D : stickers_D = 1) :
  total_stickers stickers_A pages = 110 ∧
  total_stickers stickers_B pages = 66 ∧
  total_stickers stickers_C pages = 44 ∧
  total_stickers stickers_D pages = 22 := 
by
  simp [total_stickers, h_pages, h_stickers_A, h_stickers_B, h_stickers_C, h_stickers_D]
  sorry

end stickers_count_l306_306009


namespace ship_distance_graph_l306_306425

-- Definitions of points and paths
structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def distance (p q : Point) : ℝ :=
  real.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2)

def is_circular_path (A B D : Point) (X : Point) (r : ℝ) : Prop :=
  distance A X = r ∧ distance B X = r ∧ distance D X = r

def is_straight_path (B C : Point) (X : Point) : Prop :=
  ∀ t : ℝ, distance B X < distance C X → distance (Point.mk (B.x + t * (C.x - B.x)) (B.y + t * (C.y - B.y))) X > distance B X

-- The main theorem
theorem ship_distance_graph
  (A B D C X : Point)
  (r : ℝ)
  (h_circular_path : is_circular_path A B D X r)
  (h_straight_path : is_straight_path B C X) :
  ∃ graph : ℝ → ℝ, (∀ t < distance A X, graph t = r) ∧ (∀ t ≥ distance A X, graph t > r) := 
sorry

end ship_distance_graph_l306_306425


namespace garden_area_difference_l306_306956

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l306_306956


namespace median_moons_per_celestial_body_is_correct_l306_306900

theorem median_moons_per_celestial_body_is_correct :
  let moons := [0, 0, 0, 1, 2, 5, 14, 27, 67, 82] in
  let sorted_moons := List.sort moons in
  let n := List.length sorted_moons in
  let median := if Even n then (sorted_moons.get! (n / 2 - 1) + sorted_moons.get! (n / 2)) / 2 else sorted_moons.get! (n / 2) 
  median = 3.5 :=
by
  sorry

end median_moons_per_celestial_body_is_correct_l306_306900


namespace irrational_prime_concatenation_l306_306386

theorem irrational_prime_concatenation :
  (∀ n : ℕ, prime p_n) ∧ (∑ n, 1 / (p_n : ℝ) = ⊤) → ¬(rational (digits 10 (0.p1p2p3...))) :=
by
  sorry

end irrational_prime_concatenation_l306_306386


namespace expected_difference_l306_306999

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def outcomes := {1, 2, 3, 4, 5, 6, 7, 8}

def is_prime_roll (n : ℕ) : Prop :=
  n ∈ {2, 3, 5, 7}

def is_perfect_square_roll (n : ℕ) : Prop :=
  n ∈ {4}

def is_composite_non_perfect_square_roll (n : ℕ) : Prop :=
  n ∈ {6, 8}

def rolls_again (n : ℕ) : Prop :=
  n = 1

def unsweetened_cereal_roll (n : ℕ) : Prop :=
  is_prime_roll n ∨ is_perfect_square_roll n

def sweetened_cereal_roll (n : ℕ) : Prop :=
  is_composite_non_perfect_square_roll n

def num_days_non_leap_year := 365

theorem expected_difference:
  let unsweetened_prob := (5 : ℚ) / 7
  let sweetened_prob := (2 : ℚ) / 7
  let expected_unsweetened := unsweetened_prob * 365
  let expected_sweetened := sweetened_prob * 365
  let expected_diff := expected_unsweetened - expected_sweetened
  in expected_diff = 156 :=
by
  sorry

end expected_difference_l306_306999


namespace parallel_lines_slope_l306_306469

theorem parallel_lines_slope (c : ℚ) 
  (h1 : ∀ x : ℚ, (12 * x + 5) = 12 * x + 5) 
  (h2 : ∀ x : ℚ, ((3 * c - 1) * x - 7) = (3 * c - 1) * x - 7) : 
  (3 * c - 1 = 12) → c = 13 / 3 :=
by
  intro h
  have : 3 * c = 13 := by linarith
  rw this
  sorry

end parallel_lines_slope_l306_306469


namespace value_of_a_daily_profit_1600_profit_equal_l306_306407

-- Defines the relationship and terms, ensuring they align with the conditions.
def data :=
  { price_per_item : ℕ → ℕ // ∀ x y, (x = 130 ∧ y = 70) ∨ (x = 135 ∧ y = 65) ∨ (x = 140 ∧ y = 60) → x + y = 200 }

-- Prove that the sales volume for selling price 180 is 20.
theorem value_of_a : data → ∃ (a : ℕ), (180 + a = 200) :=
by {
  intro d,
  use 20,
  exact (180 + 20 = 200)
}

-- Prove that the selling price per item to achieve a daily profit of 1600 yuan is 160.
theorem daily_profit_1600 (cost price : ℕ) (profit : ℕ) (h_cost : cost = 120) (h_profit : profit = 1600) :
  (∃ x, (x - cost) * (200 - x) = profit) :=
by {
  intro h_cost,
  intro h_profit,
  use 160,
  have : (160 - 120) * (200 - 160) = 1600,
  linarith,
}

-- Given that profit for selling m items equals that of selling n items (m ≠ n), prove m + n = 80.
theorem profit_equal (m n : ℕ) (h_cost : ∀ x, (200 - x - 120) * x = (200 - n - 120) * n) (h_mn : m ≠ n) :
  m + n = 80 :=
by {
  intro h_cost,
  intro h_mn,
  have : (m - n) * (m + n - 80) = 0,
  by linarith,
  finish,
}


end value_of_a_daily_profit_1600_profit_equal_l306_306407


namespace no_more_overcrowded_apartments_l306_306744

/--
In a building with 120 apartments and 119 tenants, an apartment is considered overcrowded if at least 15 people live in it. Each day, if there is an overcrowded apartment, all the tenants from that apartment move to different apartments.
We want to prove that eventually, there will be no more overcrowded apartments.
-/
theorem no_more_overcrowded_apartments (apartments tenants : ℕ) (overcrowded_threshold : ℕ) :
  apartments = 120 →
  tenants = 119 →
  overcrowded_threshold = 15 →
  (∃ f : ℕ → ℕ → ℕ, ∀ d : ℕ, ∀ a : ℕ, 1 ≤ a ∧ a ≤ apartments → f d a ≤ tenants ∧
    (∃ b : ℕ, b ≤ apartments ∧ f d b ≥ overcrowded_threshold →
    ∀ k : ℕ, k ≤ apartments → ∃ n : ℕ, 1 ≤ n ∧ n ≤ tenants ∧ f (d+1) k = f d k + n)) →
  ∃ d_final : ℕ, ∀ a : ℕ, 1 ≤ a ∧ a ≤ apartments → f d_final a < overcrowded_threshold :=
begin
  sorry
end

end no_more_overcrowded_apartments_l306_306744


namespace min_value_frac_l306_306164

theorem min_value_frac (x : ℝ) : 
  let u := cos x ^ 2 in
  min (frac (sin x ^ 8 + cos x ^ 8 + 2) (sin x ^ 6 + cos x ^ 6 + 2))
  = (14:ℚ)/27 :=
by
  sorry

end min_value_frac_l306_306164


namespace ratio_a024_a13_l306_306189

theorem ratio_a024_a13 :
  let p := (2 - x)^5
  let a_0 := p.coeff 0
  let a_1 := p.coeff 1
  let a_2 := p.coeff 2
  let a_3 := p.coeff 3
  let a_4 := p.coeff 4
  let a_5 := p.coeff 5
  (a_0 + a_2 + a_4) / (a_1 + a_3) = - 61 / 60 :=
by
  sorry

end ratio_a024_a13_l306_306189


namespace Matrix_inverse_condition_l306_306212

theorem Matrix_inverse_condition (a d : ℝ) :
    (∀ ⦃e⦄, e = Matrix.of 2 2 ![![a, 4], ![-9, d]] → e.mul e = Matrix.one 2) → 
      (a = Real.sqrt 37 ∧ d = -Real.sqrt 37) ∨ 
      (a = -Real.sqrt 37 ∧ d = Real.sqrt 37) := 
by {
  intro h,
  -- Proof goes here (not required for this task)
  sorry
}

end Matrix_inverse_condition_l306_306212


namespace smallest_n_log_series_l306_306451

theorem smallest_n_log_series :
  ∃ n : ℕ, n > 0 ∧ (∑ k in Finset.range (n + 1), Real.log (1 + 1 / (3 ^ (3 ^ k))) ≥ Real.log (500 / 501)) ∧ n = 1 :=
begin
  -- Proof omitted
  sorry
end

end smallest_n_log_series_l306_306451


namespace max_good_isosceles_triangles_l306_306283

-- Define the problem conditions and the corresponding theorem statement
def is_regular_2006_gon (P : polygon) : Prop :=
  P.sides = 2006 ∧ P.regular

def is_good_diagonal (P : polygon) (d : diagonal) : Prop :=
  let (v1, v2) = d.endpoints in
  d.sides_partition = (odd, odd)

-- Non-intersecting diagonals
def non_intersecting_diagonals (P : polygon) (diags : set diagonal) : Prop :=
  diags.size = 2003 ∧ ∀ d1 d2 ∈ diags, d1 ≠ d2 → ¬intersect d1 d2

-- Maximum number of isosceles triangles with two "good diagonals"
theorem max_good_isosceles_triangles (P : polygon) (diags : set diagonal)
  (h1 : is_regular_2006_gon P)
  (h2 : ∀ d ∈ diags, is_good_diagonal P d)
  (h3 : non_intersecting_diagonals P diags) :
  ∃ n, n = 1003 :=
sorry

end max_good_isosceles_triangles_l306_306283


namespace harvest_duration_l306_306829

theorem harvest_duration (total_earnings earnings_per_week : ℕ) (h1 : total_earnings = 1216) (h2 : earnings_per_week = 16) :
  total_earnings / earnings_per_week = 76 :=
by
  sorry

end harvest_duration_l306_306829


namespace geometric_series_expr_l306_306116

theorem geometric_series_expr :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4)))))))))) + 100 = 5592504 := 
sorry

end geometric_series_expr_l306_306116


namespace cannot_determine_degree_from_A_P_l306_306784

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l306_306784


namespace calculate_final_number_l306_306035

def increaseByPercentage (x : ℕ) (p : ℝ) : ℕ :=
  (x : ℝ) * (p / 100) |> Int.ofNat.floor

def reduceByPercentage (x : ℕ) (p : ℝ) : ℕ :=
  (x : ℝ) * (p / 100) |> Int.ofNat.floor

theorem calculate_final_number (initial : ℕ) (increase_percent decrease_percent : ℝ) :
  increaseByPercentage initial 150 = 120 →
  increaseByPercentage initial 150 + initial = 200 →
  reduceByPercentage (increaseByPercentage initial 150 + initial) 20 = 40 →
  (increaseByPercentage initial 150 + initial) - reduceByPercentage (increaseByPercentage initial 150 + initial) 20 = 160 :=
by
  sorry

end calculate_final_number_l306_306035


namespace commodity_price_difference_l306_306871

theorem commodity_price_difference :
  ∀ (n : ℕ),
  (initial_P : ℚ) (increase_P : ℚ)
  (initial_Q : ℚ) (increase_Q : ℚ),
  initial_P = 4.20 →
  increase_P = 0.40 →
  initial_Q = 6.30 →
  increase_Q = 0.15 →
  (initial_P + increase_P * n = initial_Q + increase_Q * n + 0.40) →
  (2001 + n = 2011) :=
by
  intros n initial_P increase_P initial_Q increase_Q h1 h2 h3 h4 h5
  sorry

end commodity_price_difference_l306_306871


namespace degree_not_determined_by_A_P_l306_306775

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l306_306775


namespace train_speed_l306_306981

/--
Given:
  Length of the train = 500 m
  Length of the bridge = 350 m
  The train takes 60 seconds to completely cross the bridge.

Prove:
  The speed of the train is exactly 14.1667 m/s
-/
theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 500) (h_bridge : length_bridge = 350) (h_time : time = 60) :
  (length_train + length_bridge) / time = 14.1667 :=
by
  rw [h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_l306_306981


namespace cyclic_quadrilateral_tangency_l306_306414

theorem cyclic_quadrilateral_tangency (a b c d x y : ℝ) (h_cyclic : a = 80 ∧ b = 100 ∧ c = 140 ∧ d = 120) 
  (h_tangency: x + y = 140) : |x - y| = 5 := 
sorry

end cyclic_quadrilateral_tangency_l306_306414


namespace max_blue_points_l306_306478

theorem max_blue_points (n : ℕ) (r b : ℕ)
  (h1 : n = 2009)
  (h2 : b + r = n)
  (h3 : ∀(k : ℕ), b ≤ k * (k - 1) / 2 → r ≥ k) :
  b = 1964 :=
by
  sorry

end max_blue_points_l306_306478


namespace count_triangles_l306_306750

theorem count_triangles :
  let points := 12
  let collinear_points := 4
  let non_collinear_points := points - collinear_points
  (coll_tris := ((collinear_points * (collinear_points - 1)) / 2) * non_collinear_points) 
  (one_coll_tris := collinear_points * ((non_collinear_points * (non_collinear_points - 1)) / 2))
  (no_coll_tris := (non_collinear_points * (non_collinear_points - 1) * (non_collinear_points - 2)) / 6)
  coll_tris + one_coll_tris + no_coll_tris = 216 := 
by
  unfold points collinear_points non_collinear_points coll_tris one_coll_tris no_coll_tris
  norm_num
  sorry

end count_triangles_l306_306750


namespace complement_union_l306_306656

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306656


namespace find_triangle_sides_angles_l306_306175

noncomputable def triangle_sides_angles (a c sinA sinB : ℝ) : Prop :=
  ∃ (b : ℝ) (C : ℝ), a = 4 ∧ c = Real.sqrt 13 ∧ sinA = 4 * sinB ∧ b = 1 ∧ C = 60

theorem find_triangle_sides_angles :
  triangle_sides_angles 4 (Real.sqrt 13) (4 * (sin 0)) (sin 0) :=
begin
  -- Proof omitted
  sorry
end

end find_triangle_sides_angles_l306_306175


namespace find_x_l306_306165

theorem find_x (x : ℚ) : x * 9999 = 724827405 → x = 72492.75 :=
by
  sorry

end find_x_l306_306165


namespace number_of_BMWs_sold_l306_306074

theorem number_of_BMWs_sold (total_cars : ℕ) (ford_percentage nissan_percentage volkswagen_percentage : ℝ) 
    (h1 : total_cars = 300)
    (h2 : ford_percentage = 0.2)
    (h3 : nissan_percentage = 0.25)
    (h4 : volkswagen_percentage = 0.1) :
    ∃ (bmw_percentage : ℝ) (bmw_cars : ℕ), bmw_percentage = 0.45 ∧ bmw_cars = 135 :=
by 
    sorry

end number_of_BMWs_sold_l306_306074


namespace sum_of_diameters_equals_BC_normalized_sum_equals_one_l306_306753

-- Given: Acute-angled triangle ABC
variables {A B C A1 B1 C1 K N L: Type}

-- Given: The altitudes of the triangle
axiom altitude_AA1 : A1 = altitude A B C
axiom altitude_BB1 : B1 = altitude B A C
axiom altitude_CC1 : C1 = altitude C A B

-- Given: Intersection point of lines AA1 and B1C1
axiom intersection_K : K = intersection_point (line A A1) (line B1 C1)

-- Given: The circumcircles of triangles A1KC1 and A1KB1 intersect AB and AC at points N and L respectively
axiom circ_A1KC1_inter : N = intersection_point (circumcircle (triangle A1 K C1)) (line A B)
axiom circ_A1KB1_inter : L = intersection_point (circumcircle (triangle A1 K B1)) (line A C)

-- First part to prove: The sum of the diameters of these circles is equal to side BC
theorem sum_of_diameters_equals_BC : (diameter (circumcircle (triangle A1 K C1))) + (diameter (circumcircle (triangle A1 K B1))) = side B C :=
sorry

-- Second part to prove: (A1N / BB1) + (A1L / CC1) = 1
theorem normalized_sum_equals_one : (length (segment A1 N) / length (segment B B1)) + (length (segment A1 L) / length (segment C C1)) = 1 :=
sorry

end sum_of_diameters_equals_BC_normalized_sum_equals_one_l306_306753


namespace nat_divisibility_l306_306840

theorem nat_divisibility (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
  sorry

end nat_divisibility_l306_306840


namespace prove_lines_intersect_or_parallel_l306_306387

variables {P Q K L M N A : Type*}
variables [IncidenceGeometry P Q K L M N A]

def lines_intersect_or_parallel : Prop :=
    ∃ (R : Type*), is_intersection_of R KL MN ∧ is_intersection_of R AC MN ∧ is_intersection_of R AC KL

theorem prove_lines_intersect_or_parallel (h1 : Pappus P Q K L M N A) :
  lines_intersect_or_parallel :=
by sorry

end prove_lines_intersect_or_parallel_l306_306387


namespace find_line_equation_l306_306534

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  (∀ x, l x 0) ∨ (∀ x, l x (sqrt 3 / 3 * x)) :=
by
  sorry

end find_line_equation_l306_306534


namespace complement_union_l306_306592

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306592


namespace unique_two_digit_integer_solution_l306_306887

variable {s : ℕ}

-- Conditions
def is_two_digit_positive_integer (s : ℕ) : Prop :=
  10 ≤ s ∧ s < 100

def last_two_digits_of_13s_are_52 (s : ℕ) : Prop :=
  13 * s % 100 = 52

-- Theorem statement
theorem unique_two_digit_integer_solution (h1 : is_two_digit_positive_integer s)
                                          (h2 : last_two_digits_of_13s_are_52 s) :
  s = 4 :=
sorry

end unique_two_digit_integer_solution_l306_306887


namespace printing_time_is_23_minutes_l306_306083

-- Define the conditions
def pagesPerMinute := 20
def maintenanceBreak := 5
def maintenanceInterval := 150
def pagesToPrint := 350

-- Define the total printing time function
noncomputable def totalPrintingTime (p : ℕ) : ℝ :=
  let fullIntervals := p / maintenanceInterval
  let remainingPages := p % maintenanceInterval
  let intervalTime := maintenanceInterval / pagesPerMinute
  let remainingTime := remainingPages / pagesPerMinute
  (fullIntervals * intervalTime + (fullIntervals * maintenanceBreak) + remainingTime)

-- Prove the total time to print 350 pages is 23 minutes
theorem printing_time_is_23_minutes :
  totalPrintingTime pagesToPrint = 23 := by
  calc totalPrintingTime pagesToPrint
    = (2 * 7.5 + (2 * 5) + 2.5) : by sorry -- use actual calculation steps here
    ... = 22.5 : by sorry
    ... = 23 : by sorry -- rounding to the nearest whole number

end printing_time_is_23_minutes_l306_306083


namespace sequence_converges_l306_306181

theorem sequence_converges (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n) (h_condition : ∀ m n, a (n + m) ≤ a n * a m) : 
    ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(a n)^ (1/n) - l| < ε :=
by
  sorry

end sequence_converges_l306_306181


namespace degree_not_determined_from_characteristic_l306_306791

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l306_306791


namespace cannot_determine_degree_from_A_P_l306_306785

def A_P : (ℚ[X] → Type) := sorry -- some characteristic of polynomials

theorem cannot_determine_degree_from_A_P (P₁ P₂ : ℚ[X]) (h₁ : P₁ = X) (h₂ : P₂ = X ^ 3)
  (h_A_P : A_P P₁ = A_P P₂) : degree P₁ ≠ degree P₂ :=
by {
  sorry -- since proof is omitted, use sorry.
}

end cannot_determine_degree_from_A_P_l306_306785


namespace total_pawns_left_is_10_l306_306856

noncomputable def total_pawns_left_in_game 
    (initial_pawns : ℕ)
    (sophia_lost : ℕ)
    (chloe_lost : ℕ) : ℕ :=
  initial_pawns - sophia_lost + (initial_pawns - chloe_lost)

theorem total_pawns_left_is_10 :
  total_pawns_left_in_game 8 5 1 = 10 := by
  sorry

end total_pawns_left_is_10_l306_306856


namespace complement_union_l306_306650

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306650


namespace clock_angle_at_8_15_l306_306030

noncomputable def hour_angle_at (hour : ℕ) (minutes : ℕ) : ℝ :=
  (↑hour + ↑minutes / 60) * 30

noncomputable def minute_angle_at (minutes : ℕ) : ℝ :=
  ↑minutes * 6

theorem clock_angle_at_8_15 :
  let hour_hand_angle := hour_angle_at 8 15,
      minute_hand_angle := minute_angle_at 15 in
  abs (hour_hand_angle - minute_hand_angle) = 157.5 :=
begin
  sorry
end

end clock_angle_at_8_15_l306_306030


namespace symmetric_line_equation_proof_l306_306369

-- We state that the line equation is given
def given_line (x y : ℝ) : Prop :=
  3 * x - 4 * y + 5 = 0

-- We define the symmetric line equation
def symmetric_line (x y : ℝ) : Prop :=
  3 * x + 4 * y + 5 = 0

-- The theorem statement that we need to prove
theorem symmetric_line_equation_proof (x y : ℝ) :
  given_line x (-y) → symmetric_line x y :=
by
  intro h,
  sorry

end symmetric_line_equation_proof_l306_306369


namespace crease_set_equation_l306_306967

def crease_set (R a : ℝ) (ha : 0 ≤ a ∧ a < R) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ x y, p = (x, y) ∧
    ((2 * x - a)^2 / R^2 + 4 * y^2 / (R^2 - a^2) = 1) }

theorem crease_set_equation (R a : ℝ) (ha : 0 ≤ a ∧ a < R) :
  ∀ p : ℝ × ℝ, p ∈ crease_set R a ha ↔ ∃ x y, p = (x, y) ∧
    ((2 * x - a)^2 / R^2 + 4 * y^2 / (R^2 - a^2) = 1) :=
sorry

end crease_set_equation_l306_306967


namespace distance_at_40_kmph_l306_306063

theorem distance_at_40_kmph (x y : ℕ) 
  (h1 : x + y = 250) 
  (h2 : x / 40 + y / 60 = 6) : 
  x = 220 :=
by
  sorry

end distance_at_40_kmph_l306_306063


namespace convert_234_to_base_13_l306_306132

-- Define the base and the number to be converted
def base := 13
def num_decimal := 234

-- Define a function to convert a number from decimal to another base
def convert_to_base (n : ℕ) (b : ℕ) : List ℕ := 
  if b ≤ 1 then [] else
  let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    match n with
    | 0 => acc
    | _ => aux (n / b) ((n % b) :: acc)
  aux n []

-- Define what is meant by an equal base 13 representation
noncomputable def base_13_representation := (15 : ℕ)

-- Proof statement that num_decimal in base 13 is equal to base_13_representation
theorem convert_234_to_base_13 : convert_to_base num_decimal base = [1, 5] := by
  sorry

end convert_234_to_base_13_l306_306132


namespace root_interval_l306_306514

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h₁ : f 1 < 0) (h₂ : f 1.5 > 0) (h₃ : f 1.25 < 0) (h₄ : f 2 > 0) :
  ∃ x, 1.25 < x ∧ x < 1.5 ∧ f x = 0 :=
sorry

end root_interval_l306_306514


namespace product_of_two_equal_numbers_l306_306861

-- Definitions and conditions
def arithmetic_mean (xs : List ℚ) : ℚ :=
  xs.sum / xs.length

-- Theorem stating the product of the two equal numbers
theorem product_of_two_equal_numbers (a b c : ℚ) (x : ℚ) :
  arithmetic_mean [a, b, c, x, x] = 20 → a = 22 → b = 18 → c = 32 → x * x = 196 :=
by
  intros h_mean h_a h_b h_c
  sorry

end product_of_two_equal_numbers_l306_306861


namespace problem_statement_l306_306192

noncomputable def f (x : ℝ) : ℝ := if x > 0 then Real.log x / Real.log 6 else 0

-- Even function property
axiom even_f : ∀ x : ℝ, f(-x) = f(x)

-- Problem statement: proving that f(-4) + f(9) = 2
theorem problem_statement : f (-4) + f 9 = 2 := by
  sorry

end problem_statement_l306_306192


namespace number_of_valid_rectangles_l306_306301

-- Defining the integer side lengths and perimeter condition
def isValidRectangle (l w : ℕ) : Prop :=
  2 * l + 2 * w = 60 ∧ l * w ≥ 150

-- Counting the number of valid rectangles
def countValidRectangles : ℕ :=
  Finset.card (Finset.filter (λ pair : ℕ × ℕ, isValidRectangle pair.1 pair.2)
                             (Finset.Icc (1, 1) (29, 29))) -- providing bounds based on perimeter constraint

theorem number_of_valid_rectangles :
  countValidRectangles = 17 :=
  sorry

end number_of_valid_rectangles_l306_306301


namespace math_problem_proof_l306_306159

noncomputable def problemExpr1 : ℤ := Int.ceil (15 / 7 * (-27 / 3)) 
noncomputable def problemExpr2 : ℤ := Int.floor (15 / 7 * Int.ceil (-27 / 3)) 

theorem math_problem_proof : problemExpr1 - problemExpr2 = 1 := 
by 
  have h₁ : 15 / 7 * (-27 / 3) = (-405) / 21 := by norm_num
  have h₂ : (-405) / 21 = -135 / 7 := by norm_num
  have h₃ : Int.ceil (-135 / 7) = -19 := by norm_num
  have h₄ : Int.ceil (-27 / 3) = -9 := by norm_num
  have h₅ : 15 / 7 * -9 = -135 / 7 := by norm_num
  have h₆ : Int.floor (-135 / 7) = -20 := by norm_num
  rw [problemExpr1, problemExpr2, h₃, h₆]
  norm_num
  sorry

end math_problem_proof_l306_306159


namespace intersection_A_B_eq_union_A_B_eq_range_of_a_l306_306565

open Set

variables {U : Type*} [OrderedField U]

def A : Set U := {x : U | 3 ≤ x ∧ x < 10}
def B : Set U := {x : U | 2 < x ∧ x ≤ 7}
def C (a : U) : Set U := {x : U | a < x ∧ x < 2 * a + 6}

theorem intersection_A_B_eq :
  A ∩ B = {x : U | 3 ≤ x ∧ x ≤ 7} := by
  sorry

theorem union_A_B_eq :
  A ∪ B = {x : U | 2 < x ∧ x < 10} := by
  sorry

theorem range_of_a (a : U) (hA_C : A ⊆ C a) :
  2 ≤ a ∧ a < 3 := by
  sorry

end intersection_A_B_eq_union_A_B_eq_range_of_a_l306_306565


namespace simplify_imaginary_sum_l306_306315

noncomputable def i : ℂ := complex.I

-- Theorem statement
theorem simplify_imaginary_sum : 
  (i^10 + i^11 + ⋯ + i^2023 = -1 - 2 * i) :=
by 
  sorry

end simplify_imaginary_sum_l306_306315


namespace at_least_one_travels_l306_306147

-- Define the probabilities of A and B traveling
def P_A := 1 / 3
def P_B := 1 / 4

-- Define the probability that person A does not travel
def P_not_A := 1 - P_A

-- Define the probability that person B does not travel
def P_not_B := 1 - P_B

-- Define the probability that neither person A nor person B travels
def P_neither := P_not_A * P_not_B

-- Define the probability that at least one of them travels
def P_at_least_one := 1 - P_neither

theorem at_least_one_travels : P_at_least_one = 1 / 2 := by
  sorry

end at_least_one_travels_l306_306147


namespace range_of_a_l306_306738

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by
  sorry

end range_of_a_l306_306738


namespace area_of_quadrilateral_l306_306825

-- Definitions used in conditions
variables {A B C D : Type} [convex_quadrilateral A B C D]
variables {M : Type} -- Intersection of diagonals
variables {P Q : Type} -- Midpoints of BC and AD
variable [midpoint P B C]
variable [midpoint Q A D]
variable {d : ℝ} -- Distance condition
axiom distance_condition : ∀ (AP AQ : ℝ), AP + AQ = sqrt 2

-- The statement to prove
theorem area_of_quadrilateral (h1 : convex_quadrilateral A B C D)
                               (h2 : M = intersection_diagonals A B C D)
                               (h3 : midpoint P B C)
                               (h4 : midpoint Q A D)
                               (h5 : ∀ (AP AQ : ℝ), AP + AQ = sqrt 2) :
  ∃ (area : ℝ), area_of_quadrilateral A B C D < 1 :=
sorry

end area_of_quadrilateral_l306_306825


namespace complement_union_M_N_l306_306635

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306635


namespace uniform_mixture_exists_l306_306743

theorem uniform_mixture_exists (n : ℕ) (h_n : n ≥ 1) :
  ∃ f : fin (n+1) → (fin (n+1) → ℝ), 
    (∀ i j : fin (n+1), 0 ≤ f i j ∧ f i j ≤ 1) ∧
    (∀ i : fin (n+1), ∑ j, f i j = 1) ∧
    (∃ i0 : fin (n+1), ∀ j : fin (n+1), f i0 j = 0) :=
sorry

end uniform_mixture_exists_l306_306743


namespace tavern_keeper_pays_for_beer_l306_306742

/-- 
  Initially, the dollars in Anchuria and Gvaiasuela are valued equally.
  One day, the government of Gvaiasuela decided to equate the Anchurian dollar 
  to ninety Gvaiasuelan cents. The next day, a similar exchange rate was set 
  for the Gvaiasuelan dollar in Anchuria.
  Near the border in Anchuria, a man drinks a beer for 10 cents. He pays with 
  an Anchurian dollar and receives change in Gvaiasuelan currency. Then, he 
  crosses the border, buys another beer, pays with a Gvaiasuelan dollar, 
  and receives change in Anchurian currency. After returning home, he has 
  the same amount of money he started with.
  Prove that the tavern keeper pays for the beers.
-/
theorem tavern_keeper_pays_for_beer 
  (initial_value_equal : ∀ (A G : ℝ), A = G)
  (exchange_rate1 : ∀ (A Gc : ℝ), A = 90 * Gc)
  (beer_cost : ℝ := 10) :
  ∃ (T : Prop), T = "The tavern keeper pays for the beer" :=
  sorry

end tavern_keeper_pays_for_beer_l306_306742


namespace degree_not_determined_by_A_P_l306_306793

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l306_306793


namespace domain_f_minus_g_range_of_x_l306_306556

-- Definitions for functions f and g
def f (a x : ℝ) : ℝ := log a (x - 1)
def g (a x : ℝ) : ℝ := log a (4 - 2 * x)

-- Part Ⅰ: The domain of f(x) - g(x)
theorem domain_f_minus_g (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 
  (∀ x, (1 < x ∧ x < 2) ↔ (0 < f a x ∧ 0 < g a x)) :=
sorry

-- Part Ⅱ: The range of x given f(x) > g(x)
theorem range_of_x (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, f a x > g a x → 
    ((a > 1 → (5/3 < x ∧ x < 2)) ∧ (0 < a ∧ a < 1 → (1 < x ∧ x < 5/3)))) :=
sorry

end domain_f_minus_g_range_of_x_l306_306556


namespace sum_first_n_terms_l306_306560

noncomputable def a_n (n : ℕ) : ℕ := n * 3^n

noncomputable def S (n : ℕ) : ℕ := ∑ i in finset.range(n), a_n (i + 1)

theorem sum_first_n_terms (n : ℕ) :
  S n = (2 * n - 1) / 4 * 3^(n + 1) + 3 / 4 :=
by
  sorry

end sum_first_n_terms_l306_306560


namespace maximum_conformists_in_circle_l306_306400

-- Lean 4 statement for the transformed problem 
noncomputable def max_conformists (n : ℕ) : ℕ :=
  if n = 200 then 150 else 0

-- The theorem to express the problem
theorem maximum_conformists_in_circle :
  ∀ (people : ℕ) (C L : ℕ) 
    (h1 : people = 200)
    (h2 : C + L = 200)
    (h3 : ∀ (i : ℕ), i < 200 → (if i % 2 = 0 then true else true))
    (h4 : ∃ (conformist_liars : ℕ), conformist_liars = 100 ∧ conformist_liars <= L)
    (h5 : ∀ (i : ℕ), i < 100 → conforms_condition (i + 1) (i + 2) (i + 3)),
  max_conformists people = 150 := sorry

end maximum_conformists_in_circle_l306_306400


namespace salary_increase_correct_l306_306099

def new_salary : ℝ := 25000
def percent_increase : ℝ := 0.80
def original_salary (new_salary percent_increase : ℝ) : ℝ := new_salary / (1 + percent_increase)
def salary_increase (new_salary original_salary : ℝ) : ℝ := new_salary - original_salary new_salary percent_increase

theorem salary_increase_correct :
  salary_increase new_salary (original_salary new_salary percent_increase) = 11111.11 := by
  sorry

end salary_increase_correct_l306_306099


namespace find_values_of_k_l306_306757

noncomputable def complex_distance (z w : ℂ) : ℝ := complex.abs (z - w)

theorem find_values_of_k (k : ℝ) :
  (∀ z : ℂ, (complex_distance z 2 = 3 * complex_distance z (-2)) ↔ (complex.abs z = k)) ->
  k = 1.5 ∨ k = 4.5 ∨ k = 5.5 :=
by sorry

end find_values_of_k_l306_306757


namespace num_solutions_of_abs_eq_l306_306875

noncomputable def count_solutions : ℝ → ℝ → ℝ → ℤ
| x y z := if |x| - |y| = z then 1 else 0

theorem num_solutions_of_abs_eq :
  ∃! x y : ℝ, count_solutions (x+5) (3*x-7) 1 = 1 :=
by
  -- Declare variables to match the original equation | x + 5 | - | 3x - 7 | = 1
  let f := λ x : ℝ, |x + 5| - |3 * x - 7| = 1,
  -- Establish the main theorem
  have h : ∀ f, ∃ x1 x2 : ℝ, f x1 ∧ f x2 ∧ x1 ≠ x2,
    { use [4.5, 0.75], sorry },  -- Just provide the proven solutions with sorry for simplicity.
  exact ⟨f, h⟩

end num_solutions_of_abs_eq_l306_306875


namespace complement_union_l306_306593

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306593


namespace garden_area_increase_l306_306933

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l306_306933


namespace range_of_fx_l306_306872

-- Define the function y = x - sqrt(x + 1)
def fx (x : ℝ) : ℝ := x - real.sqrt (x + 1)

-- Define the target property to prove: the range of the function is [-5/4, ∞)
theorem range_of_fx : ∀ y : ℝ, y ∈ set.Ici (-5/4) ↔ (∃ x : ℝ, fx x = y) :=
by
  sorry

end range_of_fx_l306_306872


namespace min_tetrahedrons_to_partition_cube_l306_306903

theorem min_tetrahedrons_to_partition_cube : ∃ n : ℕ, n = 5 ∧ (∀ m : ℕ, m < 5 → ¬partitions_cube_into_tetrahedra m) :=
by
  sorry

end min_tetrahedrons_to_partition_cube_l306_306903


namespace weight_not_qualified_l306_306094
noncomputable def acceptable_weight (x : ℝ) : Prop :=
  x ≥ 24.75 ∧ x ≤ 25.25

theorem weight_not_qualified (weight : ℝ) : weight = 25.26 → ¬ acceptable_weight weight :=
by
  intro h
  rw [h]
  apply and.intro
  sorry -- skipped proof, confirmation out of bounds

end weight_not_qualified_l306_306094


namespace value_of_f5_f_neg5_l306_306178

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 - a * x^3 + b * x + 2

-- Given conditions
variable (a b : ℝ)
axiom h1 : f (-5) a b = 3

-- The proposition to prove
theorem value_of_f5_f_neg5 : f 5 a b + f (-5) a b = 4 :=
by
  -- Include the result of the proof
  sorry

end value_of_f5_f_neg5_l306_306178


namespace complement_union_l306_306674

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306674


namespace polynomial_divisibility_l306_306841

theorem polynomial_divisibility (p : ℤ[X]) (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, 0 < n ∧ (∑ i in Finset.range (n + 1), p.eval (i + 1)) % k = 0 :=
sorry

end polynomial_divisibility_l306_306841


namespace train_passed_indicated_segment_l306_306340

-- Possibility of more context needed for the variables, but it looks like this:
-- Assuming we have a way to represent the graph and the movement of the train
variable (metro_system : Type)
variable (station : metro_system)
variable (segment : metro_system → metro_system → Prop )
variable (transfer_station : metro_system → Prop)
variable (train_trajectory : ℕ → metro_system)

-- Conditions: 
variable (A B : metro_system) -- Initial and final stations
variable (time_minutes : ℕ) -- Total time
variable (no_of_lines: ℕ) -- Number of lines in city N
variable (line : metro_system → ℕ) -- Line function
variable (is_terminal : metro_system → Prop)

-- Given conditions
axiom train_conditions : 
  train_trajectory 0 = A ∧
  train_trajectory time_minutes = B ∧
  no_of_lines = 3 ∧
  (∀ n, segment (train_trajectory n) (train_trajectory (n+1))) ∧
  (∀ n, line (train_trajectory n) ≠ line (train_trajectory (n+1)) ↔ transfer_station (train_trajectory n)) ∧
  (∀ n, (is_terminal (train_trajectory n) → (line (train_trajectory (n+1)) = line (train_trajectory n)) ∨ is_terminal (train_trajectory (n+1))))

-- Prove the train traveled through the marked segment
theorem train_passed_indicated_segment : ∃ n, segment (train_trajectory n) (train_trajectory (n+1)) sorry :=
sorry

end train_passed_indicated_segment_l306_306340


namespace find_radius_of_circle_B_l306_306447

noncomputable def radius_of_circle_B : Real :=
  sorry

theorem find_radius_of_circle_B :
  let A := 2
  let R := 4
  -- Define x as the horizontal distance (FG) and y as the vertical distance (GH)
  ∃ (x y : Real), 
  (y = x + (x^2 / 2)) ∧
  (y = 2 - (x^2 / 4)) ∧
  (5 * x^2 + 4 * x - 8 = 0) ∧
  -- Contains only the positive solution among possible valid radii
  (radius_of_circle_B = (22 / 25) + (2 * Real.sqrt 11 / 25))
:= 
sorry

end find_radius_of_circle_B_l306_306447


namespace b5_b9_equal_16_l306_306523

-- Define the arithmetic sequence and conditions
variables {a : ℕ → ℝ} (h_arith : ∀ n m, a m = a n + (m - n) * (a 1 - a 0))
variable (h_non_zero : ∀ n, a n ≠ 0)
variable (h_cond : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)

-- Define the geometric sequence and condition
variables {b : ℕ → ℝ} (h_geom : ∀ n, b (n + 1) = b n * (b 1 / b 0))
variable (h_b7 : b 7 = a 7)

-- State the theorem to prove
theorem b5_b9_equal_16 : b 5 * b 9 = 16 :=
sorry

end b5_b9_equal_16_l306_306523


namespace complement_union_eq_l306_306587

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306587


namespace ratio_of_shoppers_l306_306001

theorem ratio_of_shoppers (boxes ordered_of_yams: ℕ) (packages_per_box shoppers total_shoppers: ℕ)
  (h1 : packages_per_box = 25)
  (h2 : ordered_of_yams = 5)
  (h3 : total_shoppers = 375)
  (h4 : shoppers = ordered_of_yams * packages_per_box):
  (shoppers : ℕ) / total_shoppers = 1 / 3 := 
sorry

end ratio_of_shoppers_l306_306001


namespace degree_not_determined_by_A_P_l306_306779

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l306_306779


namespace AP_times_AQ_equals_a_square_l306_306341

-- Define the conditions as Lean 4 definitions.
def ellipse (F1 F2 : Point) (a b : ℝ) : Ellipse := sorry
def is_on_ellipse (A : Point) (e : Ellipse) : Prop := sorry
def normal_to_ellipse (A : Point) (e : Ellipse) : Line := sorry
def intersects_minor_axis (l : Line) (e : Ellipse) : Point := sorry
def projection_on_normal (center A : Point) (n : Line) : Point := sorry
def major_axis_length (e : Ellipse) : ℝ := sorry
def AP_distance (A P : Point) : ℝ := sorry
def AQ_distance (A Q : Point) : ℝ := sorry

-- Use the above definitions in the theorem statement.
theorem AP_times_AQ_equals_a_square 
(F1 F2 A C Q P : Point) 
(e : Ellipse) 
(a : ℝ) :
e = ellipse F1 F2 a b →
is_on_ellipse A e →
Q = intersects_minor_axis (normal_to_ellipse A e) e →
P = projection_on_normal C A (normal_to_ellipse A e) →
AP_distance A P * AQ_distance A Q = a^2 :=
sorry

end AP_times_AQ_equals_a_square_l306_306341


namespace example_theorem_l306_306666

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306666


namespace largest_mersenne_prime_less_than_500_l306_306957

open Nat

/-- A Mersenne prime is a prime number of the form 2^n - 1, where n is prime -/
def isMersennePrime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1

/-- The largest Mersenne prime less than 500 -/
theorem largest_mersenne_prime_less_than_500 : ∃ p, isMersennePrime p ∧ p < 500 ∧ ∀ q, isMersennePrime q ∧ q < 500 → q ≤ p :=
by
  use 127
  have h_prime_7 : prime 7 := by norm_num
  have h_127 : 127 = 2^7 - 1 := by norm_num
  split
  use 7
  exact h_prime_7
  exact h_127
  split
  exact by norm_num
  intros q hq
  sorry

end largest_mersenne_prime_less_than_500_l306_306957


namespace integer_satisfies_mod_and_range_l306_306026

theorem integer_satisfies_mod_and_range :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-150 ≡ n [ZMOD 25]) → n = 0 :=
by
  sorry

end integer_satisfies_mod_and_range_l306_306026


namespace percentage_reduction_in_price_l306_306077

-- Definitions for the conditions in the problem
def reduced_price_per_kg : ℕ := 30
def extra_oil_obtained_kg : ℕ := 10
def total_money_spent : ℕ := 1500

-- Definition of the original price per kg of oil
def original_price_per_kg : ℕ := 75

-- Statement to prove the percentage reduction
theorem percentage_reduction_in_price : 
  (original_price_per_kg - reduced_price_per_kg) * 100 / original_price_per_kg = 60 := by
  sorry

end percentage_reduction_in_price_l306_306077


namespace excenter_square_equal_excenter_square_sum_equal_l306_306395
-- Load the entire Mathlib library

-- Semiperimeter definition
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Excircle radius opposite to vertex A, its definition
def exradius (s a b c : ℝ) : ℝ := (s - a) * (s - b) * (s - c) / s

-- The given problem reformulated in Lean
theorem excenter_square_equal (a b c R r : ℝ) (s : ℝ) (h_s : s = semi_perimeter a b c) :
  let AO' := (s - a) ^ 2 + (exradius s a b c) ^ 2 in AO' = (s - a) / s * b * c := 
sorry

theorem excenter_square_sum_equal (a b c R r : ℝ) (s : ℝ) (h_s : s = semi_perimeter a b c) :
  let AO' := (s - a) / s * b * c
  let BO' := (s - b) / s * a * c
  let CO' := (s - c) / s * a * b
  AO' + BO' + CO' = b * c + c * a + a * b - 12 * R * r :=
sorry

end excenter_square_equal_excenter_square_sum_equal_l306_306395


namespace milk_in_jugs_l306_306265

theorem milk_in_jugs (x y : ℝ) (h1 : x + y = 70) (h2 : y + 0.125 * x = 0.875 * x) :
  x = 40 ∧ y = 30 := 
sorry

end milk_in_jugs_l306_306265


namespace central_angle_radian_measure_l306_306537

namespace SectorProof

variables (R l : ℝ)
variables (α : ℝ)

-- Given conditions
def condition1 : Prop := 2 * R + l = 20
def condition2 : Prop := 1 / 2 * l * R = 9
def α_definition : Prop := α = l / R

-- Central angle result
theorem central_angle_radian_measure (h1 : condition1 R l) (h2 : condition2 R l) :
  α_definition α l R → α = 2 / 9 :=
by
  intro h_α
  -- proof steps would be here, but we skip them with sorry
  sorry

end SectorProof

end central_angle_radian_measure_l306_306537


namespace triangle_perimeter_l306_306017

theorem triangle_perimeter {DE EF FD : ℝ} (h1 : DE = 160) (h2 : EF = 280) (h3 : FD = 240)
  (m_D m_E m_F : ℝ) (h4 : m_D = 70) (h5 : m_E = 60) (h6 : m_F = 30) :
  let DK := (70 / 280) * 160,
      DL := (60 / 160) * 240 in
  DK + DL + m_D = 200 :=
by sorry

end triangle_perimeter_l306_306017


namespace cannot_determine_degree_from_char_set_l306_306769

noncomputable def characteristic_set (P : Polynomial ℝ) : SomeType := sorry  -- Define the type and function for characteristic set here

-- Define two polynomials P1 and P2
def P1 : Polynomial ℝ := Polynomial.Coeff 1 1 
def P2 : Polynomial ℝ := Polynomial.Coeff 1 3

-- Assume the characteristic sets are equal but degrees are different
theorem cannot_determine_degree_from_char_set
  (A_P1 := characteristic_set P1)
  (A_P2 := characteristic_set P2)
  (h_eq : A_P1 = A_P2)
  (h_deg_neq : Polynomial.degree P1 ≠ Polynomial.degree P2) :
  False :=
begin
  sorry,
end

end cannot_determine_degree_from_char_set_l306_306769


namespace circle_and_line_intersection_l306_306194

theorem circle_and_line_intersection :
  (∀ (x y : ℝ), (y = 2 * x) → (x - 1)^2 + (y - 2)^2 = 8) ∧
  (∀ (m : ℝ), (∀ (x y : ℝ), mx + y - 3m - 1 = 0 → 2 * sqrt 3 = 2 * sqrt (8 - ((abs (1 - 2 * m)) / sqrt (1 + m^2))^2)) → m = -2) := 
  sorry

end circle_and_line_intersection_l306_306194


namespace find_real_root_l306_306495

theorem find_real_root (x : ℝ) (hx : sqrt x + sqrt (x + 3) = 8) : x = 3721 / 256 := 
by sorry

end find_real_root_l306_306495


namespace math_proof_problem_l306_306544

noncomputable def ellipse_equation : Type :=
  { a > b > 0 ∧ e = sqrt 3 / 2 ∧ (∀ x y, x^2 + y^2 = 12 → ∃ x y, ellipse a b x y) |
    (a, b) : (Real, Real) ∧
    (h : a = 2 * sqrt 3 ∧ b^2 = a^2 - a^2 * e^2)
  }

def find_m_value (x1 y1 x2 y2 : Real) : set Real :=
  { m : Real | (∃ M N : Real, line_eq M N x y m x1 x2 y1 y2 ∧ circle MN diameter x y ∧ passes_origin x y) ∧ 
    (m = sqrt 11 / 2 ∨ m = -sqrt 11 / 2)
  }

def max_area_triangle_PNM (m : Real) : Real :=
  { max_area | (∃ PMN : Real, y_surface y_surface_y x_line x_line_y m ∧ area_triangle PMN = max_area) }
  
theorem math_proof_problem:
  (∀ (a b : Real) (e : Real), e = sqrt 3 / 2 ∧ ∃ (x y : Real), x^2 + y^2 = 12 → ∃ (x y : Real), ellipse_equation x y →
    find_m_value x y y y ∧ max_area_triangle_PNM y) :=
begin
  sorry
end

end math_proof_problem_l306_306544


namespace simple_interest_principal_l306_306087

-- Define the simple interest problem
theorem simple_interest_principal
  (SI : ℝ) (R : ℝ) (T : ℝ)
  (hSI : SI = 4016.25)
  (hR : R = 0.1)
  (hT : T = 5) :
  let P := SI / (R * T) in
  P = 8032.5 :=
by
  -- Conditions already defined and simplified to simplified proof outline
  rw [hSI, hR, hT]
  have : P = SI / (R * T) := rfl
  rw this
  sorry

end simple_interest_principal_l306_306087


namespace complement_union_l306_306657

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306657


namespace angle_of_inclination_tangent_l306_306160

theorem angle_of_inclination_tangent :
  let curve (x : ℝ) := (1 / 2) * x^2 - 2 * x
  let point := (1, -(3 / 2) : ℝ × ℝ)
  let slope (x : ℝ) := x - 2
  let theta := Real.arctan
  ∃ θ : ℝ, θ = 135 ∧ θ = θ (slope 1) := 
by
  sorry

end angle_of_inclination_tangent_l306_306160


namespace problem_proof_l306_306233

theorem problem_proof (f : ℝ → ℝ) (x : ℝ) (a b : ℝ) 
  (hf : ∀ x, f(x) = 4 * x + 1) (ha : a > 0) (hb : b > 0) 
  (hxb : |x + 1| < b) :
  |f(x) + 4| < a ↔ b ≤ a / 4 :=
by
  sorry

end problem_proof_l306_306233


namespace example_theorem_l306_306664

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306664


namespace inversion_range_10_volumes_l306_306855

/-- This structure defines the notion of inversion (disarrangement) in a sequence. -/
structure InversionProblem where
  n : ℕ
  arrangement : ℕ → ℕ  -- A function to represent the arrangement of volumes
  count_inversions : ((ℕ × ℕ) → Bool) → ℕ  -- A function to count inversions satisfying a given predicate

/-- This predicate checks if a pair of indices form an inversion. -/
def is_inversion {n : ℕ} (arrangement : ℕ → ℕ) (i j : ℕ) : Bool :=
  i < j ∧ arrangement i > arrangement j

noncomputable def count_inversions {n : ℕ} (arrangement : ℕ → ℕ) : ℕ :=
  (Finset.univ.product Finset.univ).count (λ p => is_inversion arrangement p.1 p.2)

theorem inversion_range_10_volumes : 
  let n := 10 in
  ∀ S : ℕ, S ≤ (n * (n - 1)) / 2 ↔ ∃ arrangement : ℕ → ℕ, count_inversions arrangement = S :=
  by
  let n := 10
  sorry

end inversion_range_10_volumes_l306_306855


namespace complement_union_l306_306608

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306608


namespace original_trash_cans_in_certain_park_l306_306444

variable (x : ℝ)

def centralParkTrashCans (x : ℝ) : ℝ := (1 / 2) * x + 8
def trashCansMoved (x : ℝ) : ℝ := (1 / 2) * (centralParkTrashCans x)
def finalTrashCansInCertainPark (x : ℝ) : ℝ := x + trashCansMoved x

theorem original_trash_cans_in_certain_park (x : ℝ) (h : finalTrashCansInCertainPark x = 34) : x = 24 :=
sorry

end original_trash_cans_in_certain_park_l306_306444


namespace factorable_iff_some_even_b_l306_306864

open Int

theorem factorable_iff_some_even_b (b : ℤ) :
  (∃ m n p q : ℤ,
    (35 : ℤ) = m * p ∧
    (35 : ℤ) = n * q ∧
    b = m * q + n * p) →
  (∃ k : ℤ, b = 2 * k) :=
by
  sorry

end factorable_iff_some_even_b_l306_306864


namespace parabola_standard_equations_l306_306195

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end parabola_standard_equations_l306_306195


namespace area_of_B_l306_306458

/-- B is the region in the complex plane such that both z/50 and 50/conjugate(z) have real and imaginary parts between 0 and 1 inclusive -/
def region_B (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im in
  0 ≤ x / 50 ∧ x / 50 ≤ 1 ∧ 
  0 ≤ y / 50 ∧ y / 50 ≤ 1 ∧ 
  0 ≤ 50 * x / (x^2 + y^2) ∧ 50 * x / (x^2 + y^2) ≤ 1 ∧ 
  0 ≤ 50 * y / (x^2 + y^2) ∧ 50 * y / (x^2 + y^2) ≤ 1

/-- The area of the region B defined in region_B is 2500 - 625 * π / 4 -/
theorem area_of_B :
  (∫ x : ℝ in 0..50, ∫ y : ℝ in 0..50, char_fun (region_B (x + y * I))) = 2500 - 625 * π / 4 :=
sorry

end area_of_B_l306_306458


namespace no_divisibility_among_permutations_l306_306885

def is_seven_digit_number (n : ℕ) : Prop :=
  n > 10^6 ∧ n < 10^7

def permutations_of_1_to_7 : Finset ℕ :=
  (Finset.univ.perm 7).filter is_seven_digit_number

theorem no_divisibility_among_permutations :
  ∀ a b : ℕ, a ∈ permutations_of_1_to_7 → b ∈ permutations_of_1_to_7 → a ≠ b → ¬ (b ∣ a) := 
by
  intros a b ha hb hneq hdiv
  sorry

end no_divisibility_among_permutations_l306_306885


namespace polynomial_b_cond_l306_306970

theorem polynomial_b_cond (b : Fin 32 → ℕ) (h: ∀ n > 32, coeff (expand_poly b) n = 0) :
  b 31 = 2^27 - 2^11 :=
sorry

def expand_poly (b : Fin 32 → ℕ) : Polynomial ℤ :=
  ∏ i in Finset.range 32, (1 - Polynomial.C (z ^ (i + 1))) ^ b i

lemma coeff (p : Polynomial ℤ) (n : ℕ) : ℤ := 
  -- coefficient of z^n in polynomial p
sorry

end polynomial_b_cond_l306_306970


namespace number_of_triangles_l306_306243

theorem number_of_triangles (x : ℕ) (h₁ : 2 + x > 3) (h₂ : x + 3 > 2) (h₃ : 2 + 3 > x) : finset.card ({n ∈ finset.range 5 | 1 < n}) = 3 :=
by
  sorry

end number_of_triangles_l306_306243


namespace complement_union_l306_306682

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306682


namespace complement_union_eq_singleton_five_l306_306568

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5})
variable (M_def : M = {1, 2})
variable (N_def : N = {3, 4})

theorem complement_union_eq_singleton_five :
  U \ (M ∪ N) = {5} :=
by
  rw [U_def, M_def, N_def]
  simp
  sorry

end complement_union_eq_singleton_five_l306_306568


namespace area_increase_correct_l306_306939

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l306_306939


namespace binom_16_12_eq_1820_l306_306129

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l306_306129


namespace total_dolls_in_2_years_l306_306309

variables (G : ℝ)

def Gina_dolls (G : ℝ) := G
def Susan_dolls (G : ℝ) := G + 2
def Rene_dolls (G : ℝ) := 3 * (G + 2)
def Natalie_dolls (G : ℝ) := (5 / 2) * G
def Emily_dolls (G : ℝ) := real.sqrt G - 2

def Gina_dolls_in_2_years (G: ℝ) := G + 6
def Rene_dolls_in_2_years (G : ℝ) := 3 * (G + 2 + 6)
def Susan_dolls_in_2_years (G : ℝ) := (G + 2) + 6
def Natalie_dolls_in_2_years (G : ℝ) := (5 / 2) * (G + 6)
def Emily_dolls_in_2_years (G : ℝ) := real.sqrt (G + 6) - 2

theorem total_dolls_in_2_years 
(G : ℝ):
  (Gina_dolls_in_2_years G) +
  (Rene_dolls_in_2_years G) + 
  (Susan_dolls_in_2_years G) +
  (Natalie_dolls_in_2_years G) +
  (Emily_dolls_in_2_years G) = 
  (3 * G + 24) + (G + 8) + ((5 / 2) * G + 15) + (real.sqrt (G + 6) - 2) + (G + 6) :=
sorry

end total_dolls_in_2_years_l306_306309


namespace logician1_max_gain_l306_306418

noncomputable def maxCoinsDistribution (logician1 logician2 logician3 : ℕ) := (logician1, logician2, logician3)

theorem logician1_max_gain 
  (total_coins : ℕ) 
  (coins1 coins2 coins3 : ℕ) 
  (H : total_coins = 10)
  (H1 : ¬ (coins1 = 9 ∧ coins2 = 0 ∧ coins3 = 1) → coins1 = 2):
  maxCoinsDistribution coins1 coins2 coins3 = (9, 0, 1) :=
by
  sorry

end logician1_max_gain_l306_306418


namespace fraction_identity_l306_306235

theorem fraction_identity (a b : ℝ) (h : a ≠ b) (h₁ : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end fraction_identity_l306_306235


namespace proportion_equal_l306_306388

theorem proportion_equal (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 :=
by
  sorry

end proportion_equal_l306_306388


namespace example_theorem_l306_306670

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306670


namespace smallest_t_l306_306008

def degree (G : SimpleGraph V) (v : V) : ℕ :=
  G.incidenceSet v

def significance (G : SimpleGraph V) (v : V) : ℝ :=
  let x := degree G v
  let y := (G.neighbor v).count (λ u, degree G u < degree G v)
  y / x

theorem smallest_t (n : ℕ) (hn : n > 1) (G : SimpleGraph (Fin n)) [G.IsConnected] [G.IsAcyclic] :
  ∑ v in G.vertices, significance G v < (3 / 8) * n :=
sorry

end smallest_t_l306_306008


namespace exists_triangle_cut_into_2005_congruent_l306_306843

theorem exists_triangle_cut_into_2005_congruent (n : ℕ) (hn : n = 2005) : 
  ∃ (Δ : Type) [triangle Δ], ∃ (cut : Δ → list Δ), list.all (congruent Δ) (cut Δ) ∧ list.length (cut Δ) = n := 
sorry

end exists_triangle_cut_into_2005_congruent_l306_306843


namespace favorite_food_sandwiches_l306_306862

theorem favorite_food_sandwiches (total_students : ℕ) (cookies_percent pizza_percent pasta_percent : ℝ)
  (h_total : total_students = 200)
  (h_cookies : cookies_percent = 0.25)
  (h_pizza : pizza_percent = 0.30)
  (h_pasta : pasta_percent = 0.35) :
  let sandwiches_percent := 1 - (cookies_percent + pizza_percent + pasta_percent)
  sandwiches_percent * total_students = 20 :=
by
  sorry

end favorite_food_sandwiches_l306_306862


namespace Sara_sister_notebooks_l306_306312

theorem Sara_sister_notebooks :
  let initial_notebooks := 4 
  let ordered_notebooks := (3 / 2) * initial_notebooks -- 150% more notebooks
  let notebooks_after_order := initial_notebooks + ordered_notebooks
  let notebooks_after_loss := notebooks_after_order - 2 -- lost 2 notebooks
  let sold_notebooks := (1 / 4) * notebooks_after_loss -- sold 25% of remaining notebooks
  let notebooks_after_sales := notebooks_after_loss - sold_notebooks
  let notebooks_after_giveaway := notebooks_after_sales - 3 -- gave away 3 notebooks
  notebooks_after_giveaway = 3 := 
by {
  sorry
}

end Sara_sister_notebooks_l306_306312


namespace sum_of_reciprocals_lt_three_l306_306804

theorem sum_of_reciprocals_lt_three (x : ℕ → ℕ) (n : ℕ) 
  (hx_pos : ∀ i, i < n → 0 < x i)
  (hx_nofragment : ∀ i j, i < n → j < n → ¬(to_digit_seq (x i) ➔ x j)) :
  (∑ i in finset.range n, 1 / x i) < 3 := 
sorry

end sum_of_reciprocals_lt_three_l306_306804


namespace complement_union_l306_306614

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306614


namespace solve_equation_simplify_expression_l306_306111

-- Problem (1)
theorem solve_equation : ∀ x : ℝ, x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by
  sorry

-- Problem (2)
theorem simplify_expression : ∀ a b : ℝ, a ≠ b → (a ≠ 0 ∧ b ≠ 0) →
  (3 * a ^ 2 - 3 * b ^ 2) / (a ^ 2 * b + a * b ^ 2) /
  (1 - (a ^ 2 + b ^ 2) / (2 * a * b)) = -6 / (a - b) := by
  sorry

end solve_equation_simplify_expression_l306_306111


namespace discount_percentage_l306_306798

theorem discount_percentage (shirts : ℕ) (total_cost : ℕ) (price_after_discount : ℕ) 
  (h1 : shirts = 3) (h2 : total_cost = 60) (h3 : price_after_discount = 12) : 
  ∃ discount_percentage : ℕ, discount_percentage = 40 := 
by 
  sorry

end discount_percentage_l306_306798


namespace cover9x9BoardWith48Moves_l306_306849

-- Define the board size
def boardSize : ℕ := 9

-- Define the checkered pattern condition
def isCheckeredPattern (x y : ℕ) : Prop := (x + y) % 2 = 1

-- The board is represented as a 2D array where the bottom left (1,1) is black
def bottomLeftBlack : Prop := isCheckeredPattern 1 1 = false

-- Define the coverage by a domino as a pair of adjacent squares
def coversByDomino (x1 y1 x2 y2 : ℕ) : Prop := 
  (x1 + 1 = x2 ∧ y1 = y2) ∨ (x1 = x2 ∧ y1 + 1 = y2)

-- A domino covers two adjacent squares with one black and one white
def isValidDomino (x1 y1 x2 y2 : ℕ) : Prop := 
  coversByDomino x1 y1 x2 y2 ∧ isCheckeredPattern x1 y1 ≠ isCheckeredPattern x2 y2

-- State the final proof statement: 
theorem cover9x9BoardWith48Moves : 
  bottomLeftBlack → 
  (∃ (dominos : list ((ℕ × ℕ) × (ℕ × ℕ))),
    dominos.length = 48 ∧ 
    ∀ ((x1, y1), (x2, y2)) ∈ dominos, isValidDomino x1 y1 x2 y2) := 
sorry

end cover9x9BoardWith48Moves_l306_306849


namespace num_elements_satisfying_f_l306_306814

def f (x : ℕ) := x^2 + 3 * x + 2

def S := Finset.range 26

def is_multiple_of_6 (n : ℕ) := n % 6 = 0

def count_elements (l : List ℕ) (p : ℕ → Bool) : ℕ :=
  l.count p

theorem num_elements_satisfying_f :
  count_elements S.toList (λ s => is_multiple_of_6 (f s)) = 17 :=
by sorry

end num_elements_satisfying_f_l306_306814


namespace degree_not_determined_by_A_P_l306_306774

variable {R : Type} [CommRing R]

def A_P {R : Type} [CommRing R] (P : R[X]) : Type := sorry

noncomputable def P1 : R[X] := X
noncomputable def P2 : R[X] := X^3

theorem degree_not_determined_by_A_P {R : Type} [CommRing R] :
  (A_P P1 = A_P P2) → ¬ (∀ P : R[X], A_P P → degree P) := sorry

end degree_not_determined_by_A_P_l306_306774


namespace cost_of_pens_l306_306293

-- Definitions from conditions
variables (notebook_cost pencil_cost total_spent : ℕ)
variables (notebooks : ℕ)

-- Given values in the conditions
def notebook_cost := 120 -- $1.20 in cents
def notebooks := 3
def pencil_cost := 150 -- $1.50 in cents
def total_spent := 680 -- $6.80 in cents

-- The question we need to prove:
theorem cost_of_pens : (total_spent - (notebooks * notebook_cost + pencil_cost) = 170) :=
by
  sorry

end cost_of_pens_l306_306293


namespace least_number_to_make_divisible_by_3_l306_306924

theorem least_number_to_make_divisible_by_3 :
  let n := 625573
  let sum_digits := (6 + 2 + 5 + 5 + 7 + 3 : Nat)
  let remainder := sum_digits % 3
  let least_number := 3 - remainder
  625573 + least_number ≡ n + 2 [MOD 3] :=
  by
  -- define the initial values
  let n := 625573
  let sum_digits := 6 + 2 + 5 + 5 + 7 + 3
  let remainder := sum_digits % 3
  let least_number := 3 - remainder
  have h1 : remainder = 1, by norm_num
  have h2 : least_number = 2, by norm_num
  sorry

end least_number_to_make_divisible_by_3_l306_306924


namespace value_of_t_l306_306727

theorem value_of_t (k : ℤ) (t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 68) : t = 20 :=
by
  sorry

end value_of_t_l306_306727


namespace maximize_3digit_number_l306_306897

theorem maximize_3digit_number :
  ∃ a b c d e : ℕ, 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    ({a, b, c, d, e} = {3, 5, 8, 6, 1}) ∧ 
    (100 * a + 10 * b + c) * (10 * d + e) = max
    [((100 * a₁ + 10 * b₁ + c₁) * (10 * d₁ + e₁) |
      {a₁, b₁, c₁, d₁, e₁} = {3, 5, 8, 6, 1} ∧
      a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ a₁ ≠ e₁ ∧
      b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ b₁ ≠ e₁ ∧
      c₁ ≠ d₁ ∧ c₁ ≠ e₁ ∧
      d₁ ≠ e₁)] :=
begin
  sorry
end

end maximize_3digit_number_l306_306897


namespace input_language_is_input_l306_306376

def is_print_statement (statement : String) : Prop := 
  statement = "PRINT"

def is_input_statement (statement : String) : Prop := 
  statement = "INPUT"

def is_conditional_statement (statement : String) : Prop := 
  statement = "IF"

theorem input_language_is_input :
  is_input_statement "INPUT" := 
by
  -- Here we need to show "INPUT" is an input statement
  sorry

end input_language_is_input_l306_306376


namespace sum_of_coordinates_l306_306345

noncomputable def endpoint_x (x : ℤ) := (-3 + x) / 2 = 2
noncomputable def endpoint_y (y : ℤ) := (-15 + y) / 2 = -5

theorem sum_of_coordinates : ∃ x y : ℤ, endpoint_x x ∧ endpoint_y y ∧ x + y = 12 :=
by
  sorry

end sum_of_coordinates_l306_306345


namespace crayon_count_l306_306106

theorem crayon_count (initial_crayons eaten_crayons : ℕ) (h1 : initial_crayons = 62) (h2 : eaten_crayons = 52) : initial_crayons - eaten_crayons = 10 := 
by 
  sorry

end crayon_count_l306_306106


namespace inscribed_sphere_radius_l306_306350

def radius_of_inscribed_sphere (a : ℝ) : ℝ := a * (Real.sqrt 5 - 1) / Real.sqrt 15

theorem inscribed_sphere_radius
  (a : ℝ)
  (base_dihedral_angle : ℝ)
  (h_base_dihedral_angle : base_dihedral_angle = 60) :
  radius_of_inscribed_sphere a = a * (Real.sqrt 5 - 1) / Real.sqrt 15 := 
by {
  sorry
}

end inscribed_sphere_radius_l306_306350


namespace math_problem_l306_306209
noncomputable theory

open Real

def f (ω : ℝ) (x : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x)

theorem math_problem (
  ω : ℝ) (hx1 : 0 < ω) (hx2 : ω < 3)
  (h_periodicity : ∀ x, f ω (x + π / 2) = -f ω x)
  (s : ℕ) (hs : 0 < s) (g : ℝ → ℝ)
  (h_shift : ∀ x, g x = f ω (x - s))
  (h_monotonicity : ∀ x, -π / 6 ≤ x ∧ x ≤ π / 6 → g x ≤ g (x + 1)) :
  ω = 2 ∧ (∀ x, f ω (x + 5 * π / 12) = f ω (5 * π / 12)) ∧ (∃ x, s = 5) ∧ (∃ s_min, s_min = 2) :=
sorry

end math_problem_l306_306209


namespace base5_divisibility_l306_306503

theorem base5_divisibility (d : ℕ) (h : d ∈ {0, 1, 2, 3, 4}) : (379 + 30 * d) % 7 = 0 ↔ d = 3 :=
by {
  sorry
}

end base5_divisibility_l306_306503


namespace number_of_regular_washes_l306_306427

open nat

theorem number_of_regular_washes
  (HeavyWash RegularWash LightWash BleachExtraWash : ℕ)
  (heavy_washes light_wash bleaches total_water : ℕ) :
  HeavyWash = 20 →
  RegularWash = 10 →
  LightWash = 2 →
  BleachExtraWash = 2 →
  heavy_washes = 2 →
  light_wash = 1 →
  bleaches = 2 →
  total_water = 76 →
  ∃ R : ℕ, 2 * HeavyWash + R * RegularWash + light_wash * LightWash + bleaches * BleachExtraWash = total_water ∧ R = 3 :=
by
  intros hHW hRW hLW hBEW hhw hlw hb tw
  use 3
  split
  { rw [hHW, hRW, hLW, hBEW, hhw, hlw, hb],
    norm_num },
  { refl }

end number_of_regular_washes_l306_306427


namespace binomial_16_12_l306_306121

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l306_306121


namespace concur_lines_l306_306289

variables {α : Type*} [InnerProductSpace ℝ α]

-- Definitions
def is_acute_angle_triangle (A B C : α) : Prop :=
  ∀ (angle1 angle2 angle3 : RealAngle), angle1 < π / 2 ∧ angle2 < π / 2 ∧ angle3 < π / 2

def midpoint (A B : α) : α := (1/2 : ℝ) • A + (1/2 : ℝ) • B

def orthocenter (A B C : α) : α := sorry -- This is to be defined

def altitude (A B C : α) : α := sorry -- Foot of the altitude

def perpendicular_from_point_to_line (P L₁ L₂ : α) : α := sorry -- Perpendicular from point P to line through L₁ and L₂

-- Main statement
theorem concur_lines (A B C M H D E : α)
  (h_acute : is_acute_angle_triangle A B C)
  (h_mid : M = midpoint A B)
  (h_orth : H = orthocenter A B C)
  (h_D : D = altitude A C B)
  (h_E : E = altitude B A C) :
  ∃ S : α, S ∈ line_through A B ∧ S ∈ line_through D E ∧ S ∈ line_through C (perpendicular_from_point_to_line C M H) :=
sorry

end concur_lines_l306_306289


namespace total_goals_in_5_matches_l306_306384

theorem total_goals_in_5_matches 
  (x : ℝ) 
  (h1 : 4 * x + 3 = 5 * (x + 0.2)) 
  : 4 * x + 3 = 11 :=
by
  -- The proof is omitted here
  sorry

end total_goals_in_5_matches_l306_306384


namespace daily_profit_at_13_max_profit_l306_306065

noncomputable def daily_profit (selling_price cost_price units_sold_price10 decrease_rate : ℝ) : ℝ :=
  let decrease := (selling_price - 10) * decrease_rate
  let units_sold := units_sold_price10 - decrease
  (selling_price - cost_price) * units_sold

theorem daily_profit_at_13 :
  (daily_profit 13 8 100 10) = 350 := 
by
  unfold daily_profit
  sorry

noncomputable def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - (x - 10) * 10)

lemma profit_function_simplified :
  profit_function = λ x, -10 * x^2 + 280 * x - 1600 :=
by
  intro x
  unfold profit_function
  sorry

theorem max_profit :
  ∃ x : ℝ, 10 ≤ x ∧ x ≤ 20 ∧ profit_function x = 360 ∧ ∀ y : ℝ, 10 ≤ y ∧ y ≤ 20 → profit_function y ≤ 360 :=
by
  use 14
  split
  { linarith }
  split
  { linarith }
  split
  { unfold profit_function
    sorry }
  {
    intros y hy
    unfold profit_function
    sorry
  }

end daily_profit_at_13_max_profit_l306_306065


namespace value_of_M_l306_306713

theorem value_of_M (M : ℝ) (h : (20 / 100) * M = (60 / 100) * 1500) : M = 4500 :=
by {
    sorry
}

end value_of_M_l306_306713


namespace complement_union_M_N_l306_306628

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306628


namespace complement_union_l306_306661

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306661


namespace ae_length_l306_306958

noncomputable def length_of_AE (AB CD AC : ℝ) (area_ratio : ℝ) : ℝ :=
  let x := 3 * AC / (2 + 3) in x

theorem ae_length :
  ∀ (AB CD AC : ℝ) (h_AB : AB = 10) (h_CD : CD = 15) (h_AC : AC = 17)
  (h_area_ratio : (2 / 3) = (10 / 15)),
  length_of_AE AB CD AC (2 / 3) = 34 / 5 :=
by
  intros
  sorry

end ae_length_l306_306958


namespace chord_length_of_circle_l306_306520

variables {M : Type} {a x y : ℝ}

/-- Define the center of circle M moving along the parabola y^2 = 4x --/
def center_of_circle (a : ℝ) := (a^2 / 4, a)

/-- Define the property that circle M passes through the fixed point (2, 0) --/
def passes_through_fixed_point (x y : ℝ) : Prop := (x - 2)^2 + y^2 = (x - a^2 / 4)^2 + (y - a)^2

/-- Define the length of the chord intercepted by the y-axis --/
def chord_length {a : ℝ} := ∀ y1 y2 : ℝ, y1 + y2 = 2a ∧ y1 * y2 = a^2 - 4 → |y1 - y2| = 4

-- The statement to prove
theorem chord_length_of_circle : ∀ a : ℝ, (∃ x y : ℝ, passes_through_fixed_point x y) →
  chord_length :=
sorry

end chord_length_of_circle_l306_306520


namespace calculate_absolute_expression_l306_306739

def polynomial := 3 * x^4 - m * x^2 + n * x + p

axiom factor1 : polynomial 3 = 0
axiom factor2 : polynomial (-1) = 0
axiom factor3 : polynomial 2 = 0

theorem calculate_absolute_expression (m n : ℝ) : 
  |3 * m - 2 * n| = 25 :=
by
  sorry

end calculate_absolute_expression_l306_306739


namespace num_three_person_subcommittees_l306_306703

theorem num_three_person_subcommittees (n k : ℕ) (h : n = 8) (hk : k = 3) (chair_included : k > 0) : 
  ∃ r : ℕ, r = Nat.choose 7 2 ∧ r = 21 :=
by
  refine ⟨_, _, _⟩
  sorry

end num_three_person_subcommittees_l306_306703


namespace remainder_proof_l306_306304

def remainder_when_dividing_161_by_16 : Prop :=
  ∃ (remainder : ℕ), 161 = (16 * 10) + remainder ∧ remainder = 1

theorem remainder_proof : remainder_when_dividing_161_by_16 :=
by
  use 1
  split
  { exact rfl }
  { exact rfl }

end remainder_proof_l306_306304


namespace greatest_value_x_plus_y_l306_306370

variable {R : Type*} [Real R]

theorem greatest_value_x_plus_y (x y : R) (h1 : x^2 + y^2 = 98) (h2 : x * y = 36) : x + y = sqrt 170 := by
  sorry

end greatest_value_x_plus_y_l306_306370


namespace complement_union_l306_306684

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_l306_306684


namespace find_equation_of_ellipse_find_area_of_triangle_maximize_area_of_triangle_through_vertices_l306_306185

noncomputable def ellipse_equation (a b: ℝ) : QuadraticEquation :=
  { h : ∀ x y: ℝ, x^2 / a^2 + y^2 / b^2 = 1 }

noncomputable def foci_eccentricity (a c: ℝ) (e: ℝ): Prop :=
  a > 0 ∧ c > 0 ∧ e = c / a

noncomputable def perimeter_triangle (Q R: Point) (F₂: Point) (perimeter: ℝ): Prop :=
  dist Q F₂ + dist R F₂ + dist Q R = perimeter

variable (Q: Point)
axiom Q_on_ellipse: ∀ Q : ℝ, (∀ a b : ℝ, ellipse_equation a b → Q ∈ C)

theorem find_equation_of_ellipse (a b: ℝ)
  (h₁ : ellipse_equation a b)
  (h₂ : foci_eccentricity a sqrt(3/4) e)
  (h₃ : perimeter_triangle Q R F₂ 8)
  (h₄ : Q_on_ellipse Q):
  ∃ a b: ℝ, h₁ = (ellipse_equation 2 1) :=
sorry

theorem find_area_of_triangle (Q R S: Point)
  (h₁ : ellipse_equation 2 1)
  (h₂ : Q_on_ellipse Q)
  (h₃ : Q.x = 0 ∧ Q.y = 1):
  ∃ area: ℝ, area = (64 * sqrt(3) / 49) :=
sorry

theorem maximize_area_of_triangle_through_vertices
  (Q R S: Point)
  (h₁ : ellipse_equation 2 1)
  (h₂ : Q_on_ellipse Q)
  (h₃ : Q.x = 0 ∧ Q.y = 1):
  ∀ Q R S, maximize_area_through_vertices Q R S →
  ∃ vertices (C: Ellipse) (area: ℝ), area = (64 * sqrt(3) / 49) :=
sorry

end find_equation_of_ellipse_find_area_of_triangle_maximize_area_of_triangle_through_vertices_l306_306185


namespace seq_properties_l306_306184

theorem seq_properties (a b : ℕ) (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (S_n : ℕ → ℚ) :
  -- Conditions and given information
  (∀ x : ℝ, a * x^2 - 3 * x + 2 = 0 → (x = 1 ∨ x = (b : ℝ))) →
  a = 1 →
  b = 2 →
  (∀ n : ℕ, a_seq n = 2 * n - 1) →
  (∀ n : ℕ, b_seq n = (1 : ℚ) / (a_seq n * a_seq (n + 1))) →
  -- Proof goal
  (∀ n : ℕ, S_n n = (b_seq 1 + b_seq 2 + ... + b_seq n)) →
  S_n n = (n : ℚ) / (2 * n + 1) :=
sorry

end seq_properties_l306_306184


namespace student_teacher_arrangements_l306_306010

theorem student_teacher_arrangements :
  let total_arrangements := (choose 5 2) * 2 * (perm 4 4)
  in total_arrangements = 960 :=
begin
  -- Defining parameters and the number of ways to choose the 2 students
  let num_students := 5,
  let num_teachers := 2,
  let students_between := 2,
  
  -- Calculating the number of ways to choose 2 students out of 5
  let choose_students : ℕ := (choose num_students students_between),

  -- Considering the order of the teachers (2 options)
  let order_teachers : ℕ := 2,

  -- Treating the group of 4 individuals as one element and arranging them with the other 3 individuals
  let arrange_others : ℕ := (perm 4 4),

  -- Calculating the total number of arrangements
  let total_arrangements := choose_students * order_teachers * arrange_others,

  -- The equality we need to prove
  show total_arrangements = 960, from sorry,
end

end student_teacher_arrangements_l306_306010


namespace find_k_value_l306_306193

theorem find_k_value (x k : ℝ) (h : x = -3) (h_eq : k * (x - 2) - 4 = k - 2 * x) : k = -5/3 := by
  sorry

end find_k_value_l306_306193


namespace geometric_series_limit_l306_306249

noncomputable def geometric_series_sum (a_1 a_2 a_3 : ℝ) 
  (a1_pos : 0 < a_1) (a2_pos : 0 < a_2) (a3_pos : 0 < a_3) 
  (h1 : a_1 * a_3 = 1) (h2 : a_2 + a_3 = 4 / 3) : ℝ :=
  let q := a_2 / a_1 in
  a_1 / (1 - q)

theorem geometric_series_limit (a_1 a_2 a_3 : ℝ) 
  (a1_pos : 0 < a_1) (a2_pos : 0 < a_2) (a3_pos : 0 < a_3)
  (h1 : a_1 * a_3 = 1) (h2 : a_2 + a_3 = 4 / 3) :
  (geometric_series_sum a_1 a_2 a_3 a1_pos a2_pos a3_pos h1 h2) = 9 / 2 :=
sorry

end geometric_series_limit_l306_306249


namespace complement_union_l306_306607

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306607


namespace complement_union_l306_306659

def U := {1, 2, 3, 4, 5}
def M := {1, 2}
def N := {3, 4}

theorem complement_union : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_l306_306659


namespace bill_left_with_22_l306_306997

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end bill_left_with_22_l306_306997


namespace final_price_lower_than_budget_l306_306295

theorem final_price_lower_than_budget :
  let budget := 1500
  let T := 750 -- budget equally split for TV
  let S := 750 -- budget equally split for Sound System
  let TV_price_with_discount := (T - 150) * 0.80
  let SoundSystem_price_with_discount := S * 0.85
  let combined_price_before_tax := TV_price_with_discount + SoundSystem_price_with_discount
  let final_price_with_tax := combined_price_before_tax * 1.08
  budget - final_price_with_tax = 293.10 :=
by
  sorry

end final_price_lower_than_budget_l306_306295


namespace complement_union_l306_306609

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l306_306609


namespace age_sum_proof_l306_306987

theorem age_sum_proof (a b c : ℕ) (h1 : a - (b + c) = 16) (h2 : a^2 - (b + c)^2 = 1632) : a + b + c = 102 :=
by
  sorry

end age_sum_proof_l306_306987


namespace tan_sum_l306_306228

theorem tan_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 96 / 65)
  (h2 : Real.cos x + Real.cos y = 72 / 65) :
  Real.tan x + Real.tan y = 507 / 112 := 
sorry

end tan_sum_l306_306228


namespace classify_abc_l306_306835

theorem classify_abc (a b c : ℝ) 
  (h1 : (a > 0 ∨ a < 0 ∨ a = 0) ∧ (b > 0 ∨ b < 0 ∨ b = 0) ∧ (c > 0 ∨ c < 0 ∨ c = 0))
  (h2 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ (a > 0 ∧ b = 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c = 0) ∨
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ b > 0 ∧ c < 0) ∨ (a = 0 ∧ b < 0 ∧ c > 0))
  (h3 : |a| = b^2 * (b - c)) : 
  a < 0 ∧ b > 0 ∧ c = 0 :=
by 
  sorry

end classify_abc_l306_306835


namespace natasha_average_speed_while_climbing_l306_306922

/-
Definitions:
- Natashen climbs up a hill (using time t_up) and descends with the same path (using time t_down).
- Total time, average speed, distance D are defined.
- We are required to find average speed when climbing.
-/

noncomputable def average_speed_while_climbing (D : ℝ) (t_up : ℝ) (t_down : ℝ) (avg_speed_total : ℝ) :=
  D / t_up

theorem natasha_average_speed_while_climbing :
  ∀ (D t_up t_down avg_speed_total : ℝ),
    D = 12 ∧ t_up = 4 ∧ t_down = 2 ∧ avg_speed_total = 4 →
    average_speed_while_climbing D t_up t_down avg_speed_total = 3 :=
by
  intros D t_up t_down avg_speed_total h
  cases' h with hD ht
  cases' ht with ht_up ht
  cases' ht with ht_down h_avg_speed
  rw [← hD, ← ht_up, ← ht_down, ← h_avg_speed]
  sorry

end natasha_average_speed_while_climbing_l306_306922


namespace degree_not_determined_from_characteristic_l306_306789

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l306_306789


namespace factorize_difference_of_squares_l306_306483

theorem factorize_difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := 
by
  sorry

end factorize_difference_of_squares_l306_306483


namespace proof_problem_l306_306926

variables {ℝ : Type*} [field ℝ] [normed_space ℝ ℝ]

variables {a b c d : ℝ}
variables {c1 c2 d1 d2 : ℝ}

-- Hypotheses
variables (h1 : a ≠ b) (ha_perp_b : a * b = 0)
variables (ha_norm : ∥a∥ = 1) (hb_norm : ∥b∥ = 1)
variables (h2 : c = c1 * a + c2 * b)
variables (h3 : d = d1 * a + d2 * b)
variables (hc_perp_d : c * d = 0) (hc_norm : ∥c∥ = 1) (hd_norm : ∥d∥ = 1)

-- Proofs to be shown
theorem proof_problem (h1 : a ≠ b) (ha_perp_b : a * b = 0)
                      (ha_norm : ∥a∥ = 1) (hb_norm : ∥b∥ = 1)
                      (h2 : c = c1 * a + c2 * b)
                      (h3 : d = d1 * a + d2 * b)
                      (hc_perp_d : c * d = 0) (hc_norm : ∥c∥ = 1) (hd_norm : ∥d∥ = 1) : 
                      c1^2 + d1^2 = 1 ∧ c2^2 + d2^2 = 1 ∧ c1 * c2 + d1 * d2 = 0 ∧ 
                      a = c1 * c + d1 * d ∧ b = c2 * c + d2 * d := 
    by {
        sorry
    }

end proof_problem_l306_306926


namespace choir_third_verse_joiners_l306_306408

theorem choir_third_verse_joiners:
  let total_singers := 30 in
  let first_verse_singers := total_singers / 2 in
  let remaining_after_first := total_singers - first_verse_singers in
  let second_verse_singers := remaining_after_first / 3 in
  let remaining_after_second := remaining_after_first - second_verse_singers in
  let third_verse_singers := remaining_after_second in
  third_verse_singers = 10 := 
by
  sorry

end choir_third_verse_joiners_l306_306408


namespace complement_of_union_is_singleton_five_l306_306648

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306648


namespace complement_union_eq_l306_306578

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l306_306578


namespace douglas_won_40_percent_in_y_l306_306389

-- Define conditions: votes percentages in counties X and Y, and the ratio of voters between counties X and Y.
variables (V : ℕ) (vx vy : ℕ)
hypothesis (h1 : vx = 2 * V) -- County X voters
hypothesis (h2 : vy = V) -- County Y voters
hypothesis (h3 : ∀ (d_votes_x d_votes_y : ℕ), d_votes_x = 76 * vx / 100 ∧ d_votes_y = 64 * (vx + vy) / 100 - d_votes_x)

-- Define the theorem to prove that candidate Douglas won 40 percent of the vote in County Y.
theorem douglas_won_40_percent_in_y :
  ∃ P : ℕ, (∀ (d_votes_x d_votes_y : ℕ), d_votes_x = 76 * vx / 100 ∧ d_votes_y = 64 * (vx + vy) / 100 - d_votes_x) ∧  (P = d_votes_y * 100 / vy) ∧ P = 40 :=
sorry

end douglas_won_40_percent_in_y_l306_306389


namespace sum_of_faces_l306_306156

variable (a d b c e f : ℕ)
variable (pos_a : a > 0) (pos_d : d > 0) (pos_b : b > 0) (pos_c : c > 0) 
variable (pos_e : e > 0) (pos_f : f > 0)
variable (h : a * b * e + a * b * f + a * c * e + a * c * f + d * b * e + d * b * f + d * c * e + d * c * f = 1176)

theorem sum_of_faces : a + d + b + c + e + f = 33 := by
  sorry

end sum_of_faces_l306_306156


namespace perp_of_circle_lines_l306_306278

open EuclideanGeometry

variable {P : Type} [EuclideanSpace P] 

def perp_of_two_intersecting_circles (Γ₁ Γ₂ : Circle P) (A B : P) (O : P) (C : P) (D E : P) : Prop :=
(A ∈ Γ₁) ∧ (A ∈ Γ₂) ∧ (B ∈ Γ₁) ∧ (B ∈ Γ₂) ∧
(O ∈ Center Γ₁) ∧ (C ∈ Γ₁) ∧ (C ≠ A) ∧ (C ≠ B) ∧
(D ∈ Γ₂) ∧ (E ∈ Γ₂) ∧
(Line_through A C D) ∧ (Line_through B C E) ∧
Perpendicular (Line_through O C) (Line_through D E)

theorem perp_of_circle_lines
  {Γ₁ Γ₂ : Circle P} {A B O C D E : P}
  (h : perp_of_two_intersecting_circles Γ₁ Γ₂ A B O C D E) :
  Perpendicular (Line_through O C) (Line_through D E)
:=
sorry

end perp_of_circle_lines_l306_306278


namespace complement_union_l306_306622

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l306_306622


namespace garden_area_increase_l306_306932

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l306_306932


namespace sum_of_intervals_l306_306502

-- Define the floor function and the function f(x)
def floor (x : ℝ) : ℤ := int.floor x

def f (x : ℝ) : ℝ := (floor x).to_real * (2013 ^ (x - (floor x).to_real) - 1)

-- Define the main theorem statement
theorem sum_of_intervals (S : Set Real) (hS : S = {x | 1 ≤ x ∧ x < 2013 ∧ f x ≤ 1}) :
  (∑ k in (Finset.range 2012).map Finset.cast, real.log (2013 / k.to_real)) = 1 :=
sorry

end sum_of_intervals_l306_306502


namespace reading_numbers_proof_l306_306268

theorem reading_numbers_proof :
  ∃ S Z I M L : ℕ, 
    L = 32 ∧
    S + Z = 64 ∧
    I = Z + 5 ∧
    M = S - 8 ∧
    I = (M + Z) / 2 ∧
    S = 41 ∧
    Z = 23 ∧
    I = 28 ∧
    M = 33 ∧
    L = 32 := 
by {
  existsi (41 : ℕ),
  existsi (23 : ℕ),
  existsi (28 : ℕ),
  existsi (33 : ℕ),
  existsi (32 : ℕ),
  sorry
}

end reading_numbers_proof_l306_306268


namespace month_with_fewest_relatively_prime_dates_is_august_l306_306424

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def relatively_prime_dates_count (month_num : ℕ) (days_in_month : ℕ) : ℕ :=
  (List.range (days_in_month + 1)).count (is_coprime month_num)

noncomputable def month_with_fewest_relatively_prime_dates : ℕ :=
  let months_with_31_days := [1, 3, 5, 7, 8, 10, 12]
  let month_counts := months_with_31_days.map (λ m => relatively_prime_dates_count m 31)
  months_with_31_days.getI (month_counts.indexOf (month_counts.minimum.getD 0))

theorem month_with_fewest_relatively_prime_dates_is_august :
  month_with_fewest_relatively_prime_dates = 8 :=
by
  sorry

end month_with_fewest_relatively_prime_dates_is_august_l306_306424


namespace complement_union_of_M_and_N_l306_306690

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l306_306690


namespace complement_of_union_is_singleton_five_l306_306644

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306644


namespace complement_union_l306_306594

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306594


namespace solution_is_111_l306_306318

-- Define the system of equations
def system_of_equations (x y z : ℝ) :=
  (x^2 + 7 * y + 2 = 2 * z + 4 * Real.sqrt (7 * x - 3)) ∧
  (y^2 + 7 * z + 2 = 2 * x + 4 * Real.sqrt (7 * y - 3)) ∧
  (z^2 + 7 * x + 2 = 2 * y + 4 * Real.sqrt (7 * z - 3))

-- Prove that x = 1, y = 1, z = 1 is a solution to the system of equations
theorem solution_is_111 : system_of_equations 1 1 1 :=
by
  sorry

end solution_is_111_l306_306318


namespace AC_length_is_sqrt_481_l306_306993

noncomputable def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (26, 0)
def C : ℝ × ℝ := (17, 12)
def D : ℝ × ℝ := (9, 12)

def length_AC : ℝ := segment_length A C

theorem AC_length_is_sqrt_481 : length_AC = real.sqrt 481 := by
  sorry

end AC_length_is_sqrt_481_l306_306993


namespace mirror_area_l306_306799

theorem mirror_area (outer_length outer_width frame_width : ℕ)
  (h_outer_length : outer_length = 70)
  (h_outer_width : outer_width = 90)
  (h_frame_width : frame_width = 15) :
  (outer_length - 2 * frame_width) * (outer_width - 2 * frame_width) = 2400 := 
by
  rw [h_outer_length, h_outer_width, h_frame_width]
  norm_num
  sorry

end mirror_area_l306_306799


namespace triangles_from_points_l306_306748

theorem triangles_from_points (P : Finset (ℕ × ℕ)) (h_card : P.card = 12) (C : Finset (ℕ × ℕ)) (hC_card : C.card = 4) 
(hC_collinear : ∀ p1 p2 p3 ∈ C, collinear ({p1, p2, p3} : Set (ℕ × ℕ)))
(h_non_collinear : ∀ (p1 p2 p3 : Finset (ℕ × ℕ)), p1 ⊆ P \ C → p2 ⊆ P \ C → p3 ⊆ P \ C → p1 ∪ p2 ∪ p3 = P \ C → ¬ collinear ({p1, p2, p3} : Set (ℕ × ℕ))):
  ∃ T : Finset {S : Finset (ℕ × ℕ) | S.card = 3}, T.card = 216 :=
sorry

end triangles_from_points_l306_306748


namespace find_line_equation_find_triangle_area_l306_306528

-- Define the ellipse condition
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 4) = 1

-- Define the midpoint condition for the chord AB
def midpoint_chord (A B : ℝ × ℝ) : Prop := (A.1 + B.1) / 2 = sqrt 3 ∧ (A.2 + B.2) / 2 = sqrt 3 / 2

-- Given the midpoint condition, find the slope of the line l
def slope_line (l : ℝ → ℝ) : Prop := 
  ∀ A B : ℝ × ℝ, (ellipse A.1 A.2) → (ellipse B.1 B.2) → midpoint_chord A B →
  l = λ x, - (1 / 2) * (x - sqrt 3) + (sqrt 3 / 2)

-- Given the conditions, find the equation of line l
theorem find_line_equation : 
  ∀ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 → 
    ellipse B.1 B.2 → 
    midpoint_chord A B →
    ∃ a b c : ℝ, a * sqrt 3 + b * (sqrt 3 / 2) = c ∧
      ∀ (x y : ℝ), l x = y ↔ a * x + b * y = c :=
sorry

-- Given conditions, find the area of triangle F1AB 
noncomputable def F1 := (-2 * sqrt 3, 0 : ℝ × ℝ)

def line_intersects_ellipse (x y : ℝ) : Prop :=
  ellipse x y ∧ (1 / 2 * x + y - sqrt 3 = 0)

theorem find_triangle_area :
  ∃ A B : ℝ × ℝ, 
    line_intersects_ellipse A.1 A.2 ∧ 
    line_intersects_ellipse B.1 B.2 ∧ 
    ∃ (area : ℝ), area = 2 * sqrt 15 :=
sorry

end find_line_equation_find_triangle_area_l306_306528


namespace integer_squared_equals_product_l306_306227

theorem integer_squared_equals_product : 
  3^8 * 3^12 * 2^5 * 2^10 = 1889568^2 :=
by
  sorry

end integer_squared_equals_product_l306_306227


namespace kids_meals_sold_l306_306103

theorem kids_meals_sold (x y : ℕ) (h1 : x / y = 2) (h2 : x + y = 12) : x = 8 :=
by
  sorry

end kids_meals_sold_l306_306103


namespace avg_of_first_5_multiples_of_5_l306_306391

theorem avg_of_first_5_multiples_of_5 : (5 + 10 + 15 + 20 + 25) / 5 = 15 := 
by {
  sorry
}

end avg_of_first_5_multiples_of_5_l306_306391


namespace tens_digit_of_13_pow_2021_l306_306143

theorem tens_digit_of_13_pow_2021 :
  let p := 2021
  let base := 13
  let mod_val := 100
  let digit := (base^p % mod_val) / 10
  digit = 1 := by
  sorry

end tens_digit_of_13_pow_2021_l306_306143


namespace solution_l306_306229

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if x + 2 ≤ 1 then f (x + 2)
else - f (-x)

theorem solution : f 2017.5 = -0.5 :=
by
  sorry

end solution_l306_306229


namespace sum_of_consecutive_page_numbers_l306_306347

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20250) : n + (n + 1) = 285 := 
sorry

end sum_of_consecutive_page_numbers_l306_306347


namespace equilateral_if_two_coincide_l306_306810

-- Define the roles of the points in the triangle
variables (A B C O I M H : Type)

-- Assumptions: Definitions of circumcenter, incenter, centroid, and orthocenter
variables [circumcenter O A B C] [incenter I A B C]
          [centroid M A B C] [orthocenter H A B C]

-- Prove that if any two points out of O, I, M, and H coincide, the triangle is equilateral
theorem equilateral_if_two_coincide :
  (O = I ∨ O = M ∨ O = H ∨ I = M ∨ I = H ∨ M = H) → equilateral A B C :=
by
  sorry

end equilateral_if_two_coincide_l306_306810


namespace Marcus_pretzels_l306_306355

theorem Marcus_pretzels (John_pretzels : ℕ) (Marcus_more_than_John : ℕ) (h1 : John_pretzels = 28) (h2 : Marcus_more_than_John = 12) : Marcus_more_than_John + John_pretzels = 40 :=
by
  sorry

end Marcus_pretzels_l306_306355


namespace rectangle_area_change_area_analysis_l306_306708

noncomputable def original_area (a b : ℝ) : ℝ := a * b

noncomputable def new_area (a b : ℝ) : ℝ := (a - 3) * (b + 3)

theorem rectangle_area_change (a b : ℝ) :
  let S := original_area a b
  let S₁ := new_area a b
  S₁ - S = 3 * (a - b - 3) :=
by
  sorry

theorem area_analysis (a b : ℝ) :
  if a - b - 3 = 0 then new_area a b = original_area a b
  else if a - b - 3 > 0 then new_area a b > original_area a b
  else new_area a b < original_area a b :=
by
  sorry

end rectangle_area_change_area_analysis_l306_306708


namespace complement_union_M_N_l306_306637

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l306_306637


namespace range_of_phi_l306_306555

theorem range_of_phi (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ) 
  (h1 : ω > 0)
  (h2 : |φ| < (Real.pi / 2))
  (h3 : ∀ x, f x = Real.sin (ω * x + φ))
  (h4 : ∀ x, f (x + (Real.pi / ω)) = f x)
  (h5 : ∀ x y, (x ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) ∧
                  (y ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) → 
                  (x < y → f x ≤ f y)) :
  (φ ∈ Set.Icc (- Real.pi / 6) (- Real.pi / 10)) :=
by
  sorry

end range_of_phi_l306_306555


namespace percentage_less_than_l306_306244

theorem percentage_less_than (x y : ℝ) (P : ℝ) (h1 : y = 1.6667 * x) (h2 : x = (1 - P / 100) * y) : P = 66.67 :=
sorry

end percentage_less_than_l306_306244


namespace max_value_xyz_infinite_rational_triples_l306_306930

-- Definitions for the problem conditions
variables {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (H : 16 * x * y * z = (x + y)^2 * (x + z)^2)

-- Maximum M value part (a)
theorem max_value_xyz : (x + y + z) ≤ 4 :=
sorry

-- Existence of infinitely many rational triples part (b)
theorem infinite_rational_triples : 
  ∃ (S : set (ℚ × ℚ × ℚ)), infinite S ∧ ∀ (t : ℚ × ℚ × ℚ), t ∈ S → 
  (16 * (t.1 : ℝ) * (t.2.1 : ℝ) * (t.2.2 : ℝ) = ((t.1 + t.2.1 : ℝ)^2 * (t.1 + t.2.2 : ℝ)^2) ∧ ((t.1 : ℝ) + (t.2.1 : ℝ) + (t.2.2 : ℝ) = 4)) :=
sorry

end max_value_xyz_infinite_rational_triples_l306_306930


namespace total_preparation_and_cooking_time_l306_306439

def time_to_chop_pepper := 3
def time_to_chop_onion := 4
def time_to_slice_mushroom := 2
def time_to_dice_tomato := 3
def time_to_grate_cheese := 1
def time_to_assemble_and_cook_omelet := 6

def num_peppers := 8
def num_onions := 4
def num_mushrooms := 6
def num_tomatoes := 6
def num_omelets := 10

theorem total_preparation_and_cooking_time :
  (num_peppers * time_to_chop_pepper) +
  (num_onions * time_to_chop_onion) +
  (num_mushrooms * time_to_slice_mushroom) +
  (num_tomatoes * time_to_dice_tomato) +
  (num_omelets * time_to_grate_cheese) +
  (num_omelets * time_to_assemble_and_cook_omelet) = 140 :=
by
  sorry

end total_preparation_and_cooking_time_l306_306439


namespace tan_beta_minus_2alpha_l306_306531

variable {α β : ℝ}

-- Condition definitions
def conditions := α ∈ Icc 0 (π / 2) ∧ (sin α ^ 2 + sin α * cos α = 3 / 5) ∧ (tan (α - β) = -3 / 2)

-- Theorem statement
theorem tan_beta_minus_2alpha (h : conditions) : tan (β - 2 * α) = 4 / 7 :=
sorry

end tan_beta_minus_2alpha_l306_306531


namespace BA_is_AB_l306_306811

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) ℝ := by
  exact sorry
noncomputable def matrixB : Matrix (Fin 2) (Fin 2) ℝ := by
  exact sorry
def AB : Matrix (Fin 2) (Fin 2) ℝ := ![[5, 2], [-2, 4]]

-- The conditions
axiom condition1 : matrixA + matrixB = matrixA * matrixB
axiom condition2 : matrixA * matrixB = AB

-- The proof statement
theorem BA_is_AB : (matrixB * matrixA) = AB :=
by
  sorry

end BA_is_AB_l306_306811


namespace cost_five_dvds_l306_306364

theorem cost_five_dvds (cost_two_dvds : ℝ) (h : cost_two_dvds = 40) : (5 / 2 * cost_two_dvds) = 100 :=
by
  rw h
  norm_num
  sorry

end cost_five_dvds_l306_306364


namespace existsBalancedCellsCount_l306_306148

def isBalanced (board : Array (Array Bool)) (i j : Nat) : Bool := sorry

def countBalancedCells (board : Array (Array Bool)) : Nat :=
  (Array.sum <| board.mapIdx (fun i row =>
    Array.sum (row.mapIdx (fun j _ =>
      if isBalanced board i j then 1 else 0))))

theorem existsBalancedCellsCount :
  ∀ n : Nat, 0 ≤ n ∧ n ≤ 9608 →
  ∃ board : Array (Array Bool), countBalancedCells board = n := sorry

end existsBalancedCellsCount_l306_306148


namespace solve_for_x_l306_306853

theorem solve_for_x (x : ℝ) : 64 = 4 * (16:ℝ)^(x - 2) → x = 3 :=
by 
  intro h
  sorry

end solve_for_x_l306_306853


namespace circumcircle_tangent_to_omega_l306_306837

noncomputable theory

open_locale classical

variables {A B C M D E X Y : Type*}

-- Define points and circle properties
def point := Type*
def circle (ω : point → Prop) (A B : point) : Prop := sorry -- Placeholder definition

-- Midpoints
def midpoint (M : point) (B C : point) : Prop :=
  M = (B + C) / 2  -- Assume some usual definition of midpoint

-- Tangency of circle to line
def tangent (ω : point → Prop) (M : point) : Prop := sorry -- Placeholder definition

-- Midpoint property of points X and Y
def midpoints (X Y : point) (E D B C : point) : Prop :=
  X = (B + E) / 2 ∧ Y = (C + D) / 2  -- Assume usual definition

-- Circle through tangent point with given properties
def circle_through_tangent (ω : point → Prop) (A M : point) : Prop :=
  (ω A) ∧ tangent ω M

-- Proof statement
theorem circumcircle_tangent_to_omega
  (M B C A D E X Y : point)
  (H1 : midpoint M B C)
  (H2 : circle_through_tangent (λ p, sorry) A M)
  (H3 : midpoints X Y E D B C)
  : tangent (λ p, sorry) (circumcircle M X Y) :=
sorry

end circumcircle_tangent_to_omega_l306_306837


namespace taylor_tan_at_0_taylor_tan_at_pi_over_4_l306_306045

open Real

noncomputable def taylor_tan_0 : ℝ → ℝ :=
  λ x, x + x^3 / 3 + 2 * x^5 / 15

theorem taylor_tan_at_0 :
  ∀ x : ℝ, abs x < π / 2 → tan x = taylor_tan_0 x + O(λ (y : ℝ), y^6) x :=
sorry

noncomputable def taylor_tan_pi_over_4 : ℝ → ℝ :=
  λ x, 1 + 2 * (x - π/4) + 2 * (x - π/4)^2 + 8 * (x - π/4)^3 / 3 + 10 * (x - π/4)^4 / 3 + 64 * (x - π/4)^5 / 15

theorem taylor_tan_at_pi_over_4 :
  ∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x = taylor_tan_pi_over_4 x + O(λ (y : ℝ), (y - π/4)^6) x :=
sorry

end taylor_tan_at_0_taylor_tan_at_pi_over_4_l306_306045


namespace complement_of_union_is_singleton_five_l306_306649

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end complement_of_union_is_singleton_five_l306_306649


namespace consecutive_numbers_l306_306338

theorem consecutive_numbers (a b c d e : ℕ) 
  (h1 : max a (max b (max c (max d e))) = 59)
  (h2 : min a (min b (min c (min d e))) = 7)
  (h3 : (a + b + c + d + e) / 5 = 27)
  (h4 : ∃ n : ℕ, {a, b, c, d, e} \ {59, 7} = {n - 1, n, n + 1}) :
  {a, b, c, d, e} \ {59, 7} = {22, 23, 24} :=
by
  sorry  -- Proof to be provided

end consecutive_numbers_l306_306338


namespace bill_left_with_22_l306_306998

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end bill_left_with_22_l306_306998


namespace triangle_centroid_equality_l306_306023

variables {x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 : ℝ}

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

def centroid (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ × ℝ :=
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

theorem triangle_centroid_equality :
  let G := midpoint x2 y2 x4 y4 in
  let H := midpoint x3 y3 x5 y5 in
  let I := midpoint x1 y1 x6 y6 in
  let K := midpoint x3 y3 x4 y4 in
  let L := midpoint x1 y1 x5 y5 in
  let M := midpoint x2 y2 x6 y6 in
  centroid (G.fst) (G.snd) (H.fst) (H.snd) (I.fst) (I.snd) =
  centroid (K.fst) (K.snd) (L.fst) (L.snd) (M.fst) (M.snd) :=
sorry

end triangle_centroid_equality_l306_306023


namespace shaded_area_in_6x6_grid_l306_306402

def total_shaded_area (grid_size : ℕ) (triangle_squares : ℕ) (num_triangles : ℕ)
  (trapezoid_squares : ℕ) (num_trapezoids : ℕ) : ℕ :=
  (triangle_squares * num_triangles) + (trapezoid_squares * num_trapezoids)

theorem shaded_area_in_6x6_grid :
  total_shaded_area 6 2 2 3 4 = 16 :=
by
  -- Proof omitted for demonstration purposes
  sorry

end shaded_area_in_6x6_grid_l306_306402


namespace distance_between_closest_points_correct_l306_306115

noncomputable def circle_1_center : ℝ × ℝ := (3, 3)
noncomputable def circle_2_center : ℝ × ℝ := (20, 12)
noncomputable def circle_1_radius : ℝ := circle_1_center.2
noncomputable def circle_2_radius : ℝ := circle_2_center.2
noncomputable def distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (12 - 3)^2)
noncomputable def distance_between_closest_points : ℝ := distance_between_centers - (circle_1_radius + circle_2_radius)

theorem distance_between_closest_points_correct :
  distance_between_closest_points = Real.sqrt 370 - 15 :=
sorry

end distance_between_closest_points_correct_l306_306115


namespace fourth_root_of_m4_eq2_l306_306754

-- Given conditions
variable (x a : ℝ)
variable (m : ℝ)
variable (h_ge_0 : a ≥ 0)
variable (h_x4_eq_a : x^4 = a)
variable (h_sqrt4_m4_eq_2 : real.sqrt (real.sqrt (m^4)) = 2)

-- Formal statement to prove in Lean 4
theorem fourth_root_of_m4_eq2 (h : real.sqrt (real.sqrt (m^4)) = 2) : m = 2 ∨ m = -2 :=
sorry

end fourth_root_of_m4_eq2_l306_306754


namespace sam_total_hours_studying_l306_306848

def sam_minutes_spent_studying_science := 60
def sam_minutes_spent_studying_math := 80
def sam_minutes_spent_studying_literature := 40
def sam_minutes_spent_studying_history := 100
def sam_minutes_spent_studying_geography := 30

def sam_seconds_spent_studying_physical_education := 1500
def sam_minutes_spent_studying_physical_education := sam_seconds_spent_studying_physical_education / 60

def total_minutes_spent_studying := sam_minutes_spent_studying_science + 
                                    sam_minutes_spent_studying_math + 
                                    sam_minutes_spent_studying_literature + 
                                    sam_minutes_spent_studying_history + 
                                    sam_minutes_spent_studying_geography + 
                                    sam_minutes_spent_studying_physical_education

def total_hours_spent_studying := total_minutes_spent_studying / 60

theorem sam_total_hours_studying : total_hours_spent_studying = 5.5833 := by sorry

end sam_total_hours_studying_l306_306848


namespace each_person_owes_29_01_l306_306994

noncomputable def bill_amount_before_tax : ℝ := 461.79
noncomputable def tax_rate : ℝ := 0.0925
noncomputable def tip_rate : ℝ := 0.15
noncomputable def number_of_people : ℝ := 20

def tax_amount : ℝ := bill_amount_before_tax * tax_rate
def total_amount_before_tip : ℝ := bill_amount_before_tax + tax_amount
def tip_amount : ℝ := total_amount_before_tip * tip_rate
def final_total_amount : ℝ := total_amount_before_tip + tip_amount
def amount_owed_per_person : ℝ := final_total_amount / number_of_people

theorem each_person_owes_29_01 : Real.round (amount_owed_per_person * 100) / 100 = 29.01 := by
  sorry

end each_person_owes_29_01_l306_306994


namespace units_digit_24_pow_4_plus_42_pow_4_l306_306910

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end units_digit_24_pow_4_plus_42_pow_4_l306_306910


namespace proof_of_non_control_l306_306430

def experiment_A_is_not_control : Prop := 
  ¬designed_based_on_principle_of_control "Investigating the changes in the population of yeast in the culture medium"

def experiment_B_is_control : Prop := 
  designed_based_on_principle_of_control "Investigating the effect of pH on the activity of catalase"

def experiment_C_is_control : Prop := 
  designed_based_on_principle_of_control "Investigating the method of respiration in yeast cells"

def experiment_D_is_control : Prop := 
  designed_based_on_principle_of_control "Investigating the optimal concentration of auxin analogs to promote rooting in cuttings"

theorem proof_of_non_control : experiment_A_is_not_control ∧ experiment_B_is_control ∧ experiment_C_is_control ∧ experiment_D_is_control → 
  ¬designed_based_on_principle_of_control "Investigating the changes in the population of yeast in the culture medium" :=
by
  sorry

end proof_of_non_control_l306_306430


namespace terminal_side_alpha_value_zero_l306_306737

theorem terminal_side_alpha_value_zero (α : ℝ):
    (∃ x y : ℝ, x + y = 0 ∧ (x = cos α ∧ y = sin α)) → 
    (cos α + sin α = 0) :=
begin
  sorry
end

end terminal_side_alpha_value_zero_l306_306737


namespace sequence_bounds_l306_306828

noncomputable theory

def sequence (n : ℕ) : ℕ → ℝ
| 0       := 1 / 2
| (k + 1) := sequence k + (1 / n) * (sequence k)^2

theorem sequence_bounds (n : ℕ) (hn : 0 < n) : 1 - (1 / n) < sequence n n ∧ sequence n n < 1 := 
sorry

end sequence_bounds_l306_306828


namespace example_theorem_l306_306669

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l306_306669


namespace quadratic_inequality_solution_l306_306140

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 5*x + 6 > 0) ↔ (x < -3 ∨ x > -2) :=
  by
    sorry

end quadratic_inequality_solution_l306_306140


namespace standard_equation_of_parabola_l306_306197

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end standard_equation_of_parabola_l306_306197


namespace area_ABC_360_l306_306740

variables {R : Type*} [OrderedSemiring R] 

structure Triangle (α : Type*) [OrderedSemiring α] :=
(A B C D E F : Point)
(midpoint_D : D = midpoint B C)
(ratio_AE_EC : ∃ (k : α), k > 0 ∧ AE = k * EC ∧ k = 2 / 3)
(ratio_AF_FD : ∃ (m : α), m > 0 ∧ AF = m * FD ∧ m = 2 / 1)
(area_DEF : ∃ (area : α), area = 24)

def Triangle.area_ABC (T : Triangle R) : R :=
  sorry

theorem area_ABC_360 (T : Triangle R) : T.area_ABC = 360 :=
  sorry

end area_ABC_360_l306_306740


namespace complement_union_l306_306601

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l306_306601


namespace SUVs_total_l306_306472

variable afternoon_suvs : ℕ := 10
variable evening_suvs : ℕ := 5
def total_suvs := afternoon_suvs + evening_suvs

theorem SUVs_total : total_suvs = 15 := by
  sorry

end SUVs_total_l306_306472


namespace minimum_tetrahedra_partition_l306_306904

-- Definitions for the problem conditions
def cube_faces : ℕ := 6
def tetrahedron_faces : ℕ := 4

def face_constraint (cube_faces : ℕ) (tetrahedral_faces : ℕ) : Prop :=
  cube_faces * 2 = 12

def volume_constraint (cube_volume : ℝ) (tetrahedron_volume : ℝ) : Prop :=
  tetrahedron_volume < cube_volume / 6

-- Main proof statement
theorem minimum_tetrahedra_partition (cube_faces tetrahedron_faces : ℕ) (cube_volume tetrahedron_volume : ℝ) :
  face_constraint cube_faces tetrahedron_faces →
  volume_constraint cube_volume tetrahedron_volume →
  5 ≤ cube_faces * 2 / 3 :=
  sorry

end minimum_tetrahedra_partition_l306_306904


namespace solution_set_of_x_x_plus_2_lt_3_l306_306351

theorem solution_set_of_x_x_plus_2_lt_3 :
  {x : ℝ | x*(x + 2) < 3} = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_x_x_plus_2_lt_3_l306_306351


namespace divisors_larger_than_8_factorial_l306_306224

theorem divisors_larger_than_8_factorial (divisor_g9 : ℕ → Prop) :
  (∀ d, d ∣ fact 9 → d > fact 8 → divisor_g9 d) →
  (∀ d₁ d₂, divisor_g9 d₁ → divisor_g9 d₂ → d₁ = d₂ → false) →
  ∑ d in {d | d ∣ fact 9 ∧ d > fact 8}.toFinset, 1 = 8 :=
by
  sorry

end divisors_larger_than_8_factorial_l306_306224


namespace distance_from_ground_at_speed_25_is_137_5_l306_306086
noncomputable section

-- Define the initial conditions and givens
def buildingHeight : ℝ := 200
def speedProportionalityConstant : ℝ := 10
def distanceProportionalityConstant : ℝ := 10

-- Define the speed function and distance function
def speed (t : ℝ) : ℝ := speedProportionalityConstant * t
def distance (t : ℝ) : ℝ := distanceProportionalityConstant * (t * t)

-- Define the specific time when speed is 25 m/sec
def timeWhenSpeedIs25 : ℝ := 25 / speedProportionalityConstant

-- Define the distance traveled at this specific time
def distanceTraveledAtTime : ℝ := distance timeWhenSpeedIs25

-- Calculate the distance from the ground
def distanceFromGroundAtSpeed25 : ℝ := buildingHeight - distanceTraveledAtTime

-- State the theorem
theorem distance_from_ground_at_speed_25_is_137_5 :
  distanceFromGroundAtSpeed25 = 137.5 :=
sorry

end distance_from_ground_at_speed_25_is_137_5_l306_306086


namespace drop_in_price_is_20_percent_l306_306960

noncomputable def percentage_drop_proof : Prop :=
  ∀ (P N : ℝ), 
  let N' := 1.5 * N in
  let gross_increase := 1.20000000000000014 in
  let P' := (gross_increase * P * N) / (1.5 * N) in
  let price_drop := 1 - (P' / P) in
  price_drop = 0.20

theorem drop_in_price_is_20_percent : percentage_drop_proof :=
by
  sorry

end drop_in_price_is_20_percent_l306_306960


namespace degree_not_determined_by_A_P_l306_306796

-- Define the polynomial type
noncomputable def A_P (P : Polynomial ℚ) : Prop := 
  -- Suppose some characteristic computation from the polynomial's coefficients.
  sorry

theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℚ), A_P P1 = A_P P2 ∧ Polynomial.degree P1 ≠ Polynomial.degree P2 :=
by
  -- Example polynomials P1(x) = x and P2(x) = x^3
  let P1 := Polynomial.X
  let P2 := Polynomial.X ^ 3
  use P1, P2
  -- Assume given characteristic computation results in the same A_P for both polynomials
  have h1 : A_P P1 = A_P P2 := sorry
  -- Show P1 and P2 have different degrees
  have h2 : Polynomial.degree P1 ≠ Polynomial.degree P2 := by
    simp[Polynomial.degree] -- degree of P1 = 1 and degree of P2 = 3
  exact ⟨h1, h2⟩

end degree_not_determined_by_A_P_l306_306796


namespace sum_first_35_digits_l306_306375

theorem sum_first_35_digits (cyclic_142857 : is_cyclic 142857) :
  (decimal_digits_sum (1 / 142857) 35 = 35) :=
sorry

end sum_first_35_digits_l306_306375


namespace figure_not_tilable_with_1x3_strips_l306_306443

-- Define the problem conditions as the hypotheses for the theorem
def figure_has_seven_colored_cells (figure : Type) [Fintype figure] [DecidablePred (∈ figure)] : Prop :=
  Fintype.card {c ∈ figure | is_colored c} = 7

noncomputable def can_tile_with_strips (figure : Type) [Fintype figure] [DecidablePred (∈ figure)] : Prop :=
  ∃ (tiling : figure → Prop), (∀ (cell : figure), tiling cell) ∧ (Fintype.card {c ∈ figure | tiling c} = 3)

-- The actual theorem to be proved
theorem figure_not_tilable_with_1x3_strips
  (figure : Type) [Fintype figure] [DecidablePred (∈ figure)]
  (h : figure_has_seven_colored_cells figure) :
  ¬ can_tile_with_strips figure :=
  sorry

end figure_not_tilable_with_1x3_strips_l306_306443
