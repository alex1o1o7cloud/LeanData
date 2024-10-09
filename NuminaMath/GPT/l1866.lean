import Mathlib

namespace part1_part2_l1866_186617

open Set

variable {α : Type*} [LinearOrderedField α]

def A : Set α := { x | abs (x - 1) ≤ 1 }
def B (a : α) : Set α := { x | x ≥ a }

theorem part1 {x : α} : x ∈ (A ∩ B 1) ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 {a : α} : (A ⊆ B a) ↔ a ≤ 0 := by
  sorry

end part1_part2_l1866_186617


namespace three_digit_permuted_mean_l1866_186689

theorem three_digit_permuted_mean (N : ℕ) :
  (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
    (N = 111 ∨ N = 222 ∨ N = 333 ∨ N = 444 ∨ N = 555 ∨ N = 666 ∨ N = 777 ∨ N = 888 ∨ N = 999 ∨
     N = 407 ∨ N = 518 ∨ N = 629 ∨ N = 370 ∨ N = 481 ∨ N = 592)) ↔
    (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ 7 * x = 3 * y + 4 * z) := by
sorry

end three_digit_permuted_mean_l1866_186689


namespace lieutenant_age_l1866_186611

variables (n x : ℕ) 

-- Condition 1: Number of soldiers is the same in both formations
def total_soldiers_initial (n : ℕ) : ℕ := n * (n + 5)
def total_soldiers_new (n x : ℕ) : ℕ := x * (n + 9)

-- Condition 2: The number of soldiers is the same 
-- and Condition 3: Equations relating n and x
theorem lieutenant_age (n x : ℕ) (h1: total_soldiers_initial n = total_soldiers_new n x) (h2 : x = 24) : 
  x = 24 :=
by {
  sorry
}

end lieutenant_age_l1866_186611


namespace problem1_problem2_l1866_186612

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

variable (a : ℝ) (x : ℝ)

-- Problem 1: Proving intersection of sets when a = 2
theorem problem1 (ha : a = 2) : (A a ∩ B a) = {x | 4 < x ∧ x < 5} :=
sorry

-- Problem 2: Proving the range of a for which B is a subset of A
theorem problem2 : {a | B a ⊆ A a} = {a | (1 < a ∧ a ≤ 3) ∨ a = -1} :=
sorry

end problem1_problem2_l1866_186612


namespace intersection_of_A_and_B_l1866_186695

def A := Set.Ioo 1 3
def B := Set.Ioo 2 4

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 3 :=
by
  sorry

end intersection_of_A_and_B_l1866_186695


namespace dan_bought_18_stickers_l1866_186643

variable (S D : ℕ)

-- Given conditions
def stickers_initially_same : Prop := S = S -- Cindy and Dan have the same number of stickers initially
def cindy_used_15_stickers : Prop := true -- Cindy used 15 of her stickers
def dan_bought_D_stickers : Prop := true -- Dan bought D stickers
def dan_has_33_more_stickers_than_cindy : Prop := (S + D) = (S - 15 + 33)

-- Question: Prove that the number of stickers Dan bought is 18
theorem dan_bought_18_stickers (h1 : stickers_initially_same S)
                               (h2 : cindy_used_15_stickers)
                               (h3 : dan_bought_D_stickers)
                               (h4 : dan_has_33_more_stickers_than_cindy S D) : D = 18 :=
sorry

end dan_bought_18_stickers_l1866_186643


namespace smallest_difference_of_sides_l1866_186664

/-- Triangle PQR has a perimeter of 2021 units. The sides have lengths that are integer values with PQ < QR ≤ PR. 
The smallest possible value of QR - PQ is 1. -/
theorem smallest_difference_of_sides :
  ∃ (PQ QR PR : ℕ), PQ < QR ∧ QR ≤ PR ∧ PQ + QR + PR = 2021 ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ QR + PR > PQ ∧ QR - PQ = 1 :=
sorry

end smallest_difference_of_sides_l1866_186664


namespace possible_values_of_C_l1866_186603

theorem possible_values_of_C {a b C : ℤ} :
  (C = a * (a - 5) ∧ C = b * (b - 8)) ↔ (C = 0 ∨ C = 84) :=
sorry

end possible_values_of_C_l1866_186603


namespace total_gain_loss_is_correct_l1866_186642

noncomputable def total_gain_loss_percentage 
    (cost1 cost2 cost3 : ℝ) 
    (gain1 gain2 gain3 : ℝ) : ℝ :=
  let total_cost := cost1 + cost2 + cost3
  let gain_amount1 := cost1 * gain1
  let loss_amount2 := cost2 * gain2
  let gain_amount3 := cost3 * gain3
  let net_gain_loss := (gain_amount1 + gain_amount3) - loss_amount2
  (net_gain_loss / total_cost) * 100

theorem total_gain_loss_is_correct :
  total_gain_loss_percentage 
    675958 995320 837492 0.11 (-0.11) 0.15 = 3.608 := 
sorry

end total_gain_loss_is_correct_l1866_186642


namespace remainder_of_exponentiation_l1866_186678

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end remainder_of_exponentiation_l1866_186678


namespace Lincoln_High_School_max_principals_l1866_186629

def max_principals (total_years : ℕ) (term_length : ℕ) (max_principals_count : ℕ) : Prop :=
  ∀ (period : ℕ), period = total_years → 
                  term_length = 4 → 
                  max_principals_count = 3

theorem Lincoln_High_School_max_principals 
  (total_years term_length max_principals_count : ℕ) :
  max_principals total_years term_length max_principals_count :=
by 
  intros period h1 h2
  have h3 : period = 10 := sorry
  have h4 : term_length = 4 := sorry
  have h5 : max_principals_count = 3 := sorry
  sorry

end Lincoln_High_School_max_principals_l1866_186629


namespace geometric_sequence_sixth_term_l1866_186646

/-- 
The statement: 
The first term of a geometric sequence is 1000, and the 8th term is 125. Prove that the positive,
real value for the 6th term is 31.25.
-/
theorem geometric_sequence_sixth_term :
  ∀ (a1 a8 a6 : ℝ) (r : ℝ),
    a1 = 1000 →
    a8 = 125 →
    a8 = a1 * r^7 →
    a6 = a1 * r^5 →
    a6 = 31.25 :=
by
  intros a1 a8 a6 r h1 h2 h3 h4
  sorry

end geometric_sequence_sixth_term_l1866_186646


namespace has_local_maximum_l1866_186679

noncomputable def func (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem has_local_maximum :
  ∃ x, x = -2 ∧ func x = 28 / 3 :=
by
  sorry

end has_local_maximum_l1866_186679


namespace remainder_div_5_l1866_186622

theorem remainder_div_5 (n : ℕ): (∃ k : ℤ, n = 10 * k + 7) → (∃ m : ℤ, n = 5 * m + 2) :=
by
  sorry

end remainder_div_5_l1866_186622


namespace sugar_amount_indeterminate_l1866_186616

-- Define the variables and conditions
variable (cups_of_flour_needed : ℕ) (cups_of_sugar_needed : ℕ)
variable (cups_of_flour_put_in : ℕ) (cups_of_flour_to_add : ℕ)

-- Conditions
axiom H1 : cups_of_flour_needed = 8
axiom H2 : cups_of_flour_put_in = 4
axiom H3 : cups_of_flour_to_add = 4

-- Problem statement: Prove that the amount of sugar cannot be determined
theorem sugar_amount_indeterminate (h : cups_of_sugar_needed > 0) :
  cups_of_flour_needed = 8 → cups_of_flour_put_in = 4 → cups_of_flour_to_add = 4 → cups_of_sugar_needed > 0 :=
by
  intros
  sorry

end sugar_amount_indeterminate_l1866_186616


namespace least_common_multiple_1008_672_l1866_186627

theorem least_common_multiple_1008_672 : Nat.lcm 1008 672 = 2016 := by
  -- Add the prime factorizations and show the LCM calculation
  have h1 : 1008 = 2^4 * 3^2 * 7 := by sorry
  have h2 : 672 = 2^5 * 3 * 7 := by sorry
  -- Utilize the factorizations to compute LCM
  have calc1 : Nat.lcm (2^4 * 3^2 * 7) (2^5 * 3 * 7) = 2^5 * 3^2 * 7 := by sorry
  -- Show the calculation of 2^5 * 3^2 * 7
  have calc2 : 2^5 * 3^2 * 7 = 2016 := by sorry
  -- Therefore, LCM of 1008 and 672 is 2016
  exact calc2

end least_common_multiple_1008_672_l1866_186627


namespace smallest_n_containing_375_consecutively_l1866_186697

theorem smallest_n_containing_375_consecutively :
  ∃ (m n : ℕ), m < n ∧ Nat.gcd m n = 1 ∧ (n = 8) ∧ (∀ (d : ℕ), d < 1000 →
  ∃ (k : ℕ), k * d % n = m ∧ (d / 100) % 10 = 3 ∧ (d / 10) % 10 = 7 ∧ d % 10 = 5) :=
sorry

end smallest_n_containing_375_consecutively_l1866_186697


namespace natural_number_with_property_l1866_186641

theorem natural_number_with_property :
  ∃ n a b c : ℕ, (n = 10 * a + b) ∧ (100 * a + 10 * c + b = 6 * n) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (n = 18) :=
sorry

end natural_number_with_property_l1866_186641


namespace smallest_multiple_l1866_186613

theorem smallest_multiple (x : ℕ) (h1 : x % 24 = 0) (h2 : x % 36 = 0) (h3 : x % 20 ≠ 0) :
  x = 72 :=
by
  sorry

end smallest_multiple_l1866_186613


namespace find_h_l1866_186672

theorem find_h: 
  ∃ h k, (∀ x, 2 * x ^ 2 + 6 * x + 11 = 2 * (x - h) ^ 2 + k) ∧ h = -3 / 2 :=
by
  sorry

end find_h_l1866_186672


namespace max_value_of_fraction_l1866_186669

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l1866_186669


namespace total_length_of_intervals_l1866_186657

theorem total_length_of_intervals :
  (∀ (x : ℝ), |x| < 1 → Real.tan (Real.log x / Real.log 5) < 0) →
  ∃ (length : ℝ), length = (2 * (5 ^ (Real.pi / 2))) / (1 + (5 ^ (Real.pi / 2))) :=
sorry

end total_length_of_intervals_l1866_186657


namespace car_mpg_city_l1866_186624

theorem car_mpg_city
  (h c T : ℝ)
  (h1 : h * T = 480)
  (h2 : c * T = 336)
  (h3 : c = h - 6) :
  c = 14 :=
by
  sorry

end car_mpg_city_l1866_186624


namespace rotate_parabola_180deg_l1866_186610

theorem rotate_parabola_180deg (x y : ℝ) :
  (∀ x, y = 2 * x^2 - 12 * x + 16) →
  (∀ x, y = -2 * x^2 + 12 * x - 20) :=
sorry

end rotate_parabola_180deg_l1866_186610


namespace major_snow_shadow_length_l1866_186636

theorem major_snow_shadow_length :
  ∃ (a1 d : ℝ), 
  (3 * a1 + 12 * d = 16.5) ∧ 
  (12 * a1 + 66 * d = 84) ∧
  (a1 + 11 * d = 12.5) := 
sorry

end major_snow_shadow_length_l1866_186636


namespace project_completion_days_l1866_186696

theorem project_completion_days (A B C : ℝ) (h1 : 1/A + 1/B = 1/2) (h2 : 1/B + 1/C = 1/4) (h3 : 1/C + 1/A = 1/2.4) : A = 3 :=
by
sorry

end project_completion_days_l1866_186696


namespace hyperbola_and_line_properties_l1866_186674

open Real

def hyperbola (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote1 (x y : ℝ) : Prop := y = sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -sqrt 3 * x
def line (x y t : ℝ) : Prop := y = x + t

theorem hyperbola_and_line_properties :
  ∃ a b t : ℝ,
  a > 0 ∧ b > 0 ∧ a = 1 ∧ b^2 = 3 ∧
  (∀ x y, hyperbola x y a b ↔ (x^2 - y^2 / 3 = 1)) ∧
  (∀ x y, asymptote1 x y ↔ y = sqrt 3 * x) ∧
  (∀ x y, asymptote2 x y ↔ y = -sqrt 3 * x) ∧
  (∀ x y, (line x y t ↔ (y = x + sqrt 3) ∨ (y = x - sqrt 3))) := sorry

end hyperbola_and_line_properties_l1866_186674


namespace exp_decreasing_iff_a_in_interval_l1866_186638

theorem exp_decreasing_iff_a_in_interval (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 - a)^x > (2 - a)^y) ↔ 1 < a ∧ a < 2 :=
by 
  sorry

end exp_decreasing_iff_a_in_interval_l1866_186638


namespace g_of_neg3_l1866_186686

def g (x : ℝ) : ℝ := x^2 + 2 * x

theorem g_of_neg3 : g (-3) = 3 :=
by
  sorry

end g_of_neg3_l1866_186686


namespace solve_sqrt_equation_l1866_186602

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end solve_sqrt_equation_l1866_186602


namespace z_amount_per_rupee_l1866_186675

theorem z_amount_per_rupee (x y z : ℝ) 
  (h1 : ∀ rupees_x, y = 0.45 * rupees_x)
  (h2 : y = 36)
  (h3 : x + y + z = 156)
  (h4 : ∀ rupees_x, x = rupees_x) :
  ∃ a : ℝ, z = a * x ∧ a = 0.5 := 
by
  -- Placeholder for the actual proof
  sorry

end z_amount_per_rupee_l1866_186675


namespace number_of_friends_l1866_186699

-- Define the conditions
def kendra_packs : ℕ := 7
def tony_packs : ℕ := 5
def pens_per_kendra_pack : ℕ := 4
def pens_per_tony_pack : ℕ := 6
def pens_kendra_keep : ℕ := 3
def pens_tony_keep : ℕ := 3

-- Define the theorem to be proved
theorem number_of_friends 
  (packs_k : ℕ := kendra_packs)
  (packs_t : ℕ := tony_packs)
  (pens_per_pack_k : ℕ := pens_per_kendra_pack)
  (pens_per_pack_t : ℕ := pens_per_tony_pack)
  (kept_k : ℕ := pens_kendra_keep)
  (kept_t : ℕ := pens_tony_keep) :
  packs_k * pens_per_pack_k + packs_t * pens_per_pack_t - (kept_k + kept_t) = 52 :=
by
  sorry

end number_of_friends_l1866_186699


namespace g_of_12_l1866_186658

def g (n : ℕ) : ℕ := n^2 - n + 23

theorem g_of_12 : g 12 = 155 :=
by
  sorry

end g_of_12_l1866_186658


namespace volume_of_soup_in_hemisphere_half_height_l1866_186659

theorem volume_of_soup_in_hemisphere_half_height 
  (V_hemisphere : ℝ)
  (hV_hemisphere : V_hemisphere = 8)
  (V_cap : ℝ) :
  V_cap = 2.5 :=
sorry

end volume_of_soup_in_hemisphere_half_height_l1866_186659


namespace solution_to_fraction_l1866_186666

theorem solution_to_fraction (x : ℝ) (h_fraction : (x^2 - 4) / (x + 4) = 0) (h_denom : x ≠ -4) : x = 2 ∨ x = -2 :=
sorry

end solution_to_fraction_l1866_186666


namespace cube_volume_ratio_l1866_186681

theorem cube_volume_ratio
  (a : ℕ) (b : ℕ)
  (h₁ : a = 5)
  (h₂ : b = 24)
  : (a^3 : ℚ) / (b^3 : ℚ) = 125 / 13824 := by
  sorry

end cube_volume_ratio_l1866_186681


namespace fraction_after_adding_liters_l1866_186670

-- Given conditions
variables (c w : ℕ)
variables (h1 : w = c / 3)
variables (h2 : (w + 5) / c = 2 / 5)

-- The proof statement
theorem fraction_after_adding_liters (h1 : w = c / 3) (h2 : (w + 5) / c = 2 / 5) : 
  (w + 9) / c = 34 / 75 :=
sorry -- Proof omitted

end fraction_after_adding_liters_l1866_186670


namespace division_of_decimals_l1866_186677

theorem division_of_decimals : (0.05 / 0.002) = 25 :=
by
  -- Proof will be filled here
  sorry

end division_of_decimals_l1866_186677


namespace james_new_fuel_cost_l1866_186684

def original_cost : ℕ := 200
def price_increase_rate : ℕ := 20
def extra_tank_factor : ℕ := 2

theorem james_new_fuel_cost :
  let new_price := original_cost + (price_increase_rate * original_cost / 100)
  let total_cost := extra_tank_factor * new_price
  total_cost = 480 :=
by
  sorry

end james_new_fuel_cost_l1866_186684


namespace smallest_distance_proof_l1866_186662

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  Complex.abs (z - w)

theorem smallest_distance_proof (z w : ℂ) 
  (h1 : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (h2 : Complex.abs (w - (-5 + 6*Complex.I)) = 4) :
  smallest_distance z w ≥ Real.sqrt 149 - 6 :=
by
  sorry

end smallest_distance_proof_l1866_186662


namespace problem_statement_l1866_186634

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 1988 a b α β = 3) : f 2013 a b α β = 5 :=
by 
  sorry

end problem_statement_l1866_186634


namespace find_second_offset_l1866_186671

variable (d : ℕ) (o₁ : ℕ) (A : ℕ)

theorem find_second_offset (hd : d = 20) (ho₁ : o₁ = 5) (hA : A = 90) : ∃ (o₂ : ℕ), o₂ = 4 :=
by
  sorry

end find_second_offset_l1866_186671


namespace union_of_S_and_T_l1866_186668

-- Declare sets S and T
def S : Set ℕ := {3, 4, 5}
def T : Set ℕ := {4, 7, 8}

-- Statement about their union
theorem union_of_S_and_T : S ∪ T = {3, 4, 5, 7, 8} :=
sorry

end union_of_S_and_T_l1866_186668


namespace age_difference_l1866_186682

variable (S R : ℝ)

theorem age_difference (h1 : S = 38.5) (h2 : S / R = 11 / 9) : S - R = 7 :=
by
  sorry

end age_difference_l1866_186682


namespace two_times_sum_of_squares_l1866_186633

theorem two_times_sum_of_squares (P a b : ℤ) (h : P = a^2 + b^2) : 
  ∃ x y : ℤ, 2 * P = x^2 + y^2 := 
by 
  sorry

end two_times_sum_of_squares_l1866_186633


namespace sheep_to_cow_ratio_l1866_186620

theorem sheep_to_cow_ratio : 
  ∀ (cows sheep : ℕ) (cow_water sheep_water : ℕ),
  cows = 40 →
  cow_water = 80 →
  sheep_water = cow_water / 4 →
  7 * (cows * cow_water + sheep * sheep_water) = 78400 →
  sheep / cows = 10 :=
by
  intros cows sheep cow_water sheep_water hcows hcow_water hsheep_water htotal
  sorry

end sheep_to_cow_ratio_l1866_186620


namespace minimize_relative_waiting_time_l1866_186608

-- Definitions of task times in seconds
def task_U : ℕ := 10
def task_V : ℕ := 120
def task_W : ℕ := 900

-- Definition of relative waiting time given a sequence of task execution times
def relative_waiting_time (times : List ℕ) : ℚ :=
  (times.head! : ℚ) / (times.head! : ℚ) + 
  (times.head! + times.tail.head! : ℚ) / (times.tail.head! : ℚ) + 
  (times.head! + times.tail.head! + times.tail.tail.head! : ℚ) / (times.tail.tail.head! : ℚ)

-- Sequences
def sequence_A : List ℕ := [task_U, task_V, task_W]
def sequence_B : List ℕ := [task_V, task_W, task_U]
def sequence_C : List ℕ := [task_W, task_U, task_V]
def sequence_D : List ℕ := [task_U, task_W, task_V]

-- Sum of relative waiting times for each sequence
def S_A := relative_waiting_time sequence_A
def S_B := relative_waiting_time sequence_B
def S_C := relative_waiting_time sequence_C
def S_D := relative_waiting_time sequence_D

-- Theorem to prove that sequence A has the minimum sum of relative waiting times
theorem minimize_relative_waiting_time : S_A < S_B ∧ S_A < S_C ∧ S_A < S_D := 
  by sorry

end minimize_relative_waiting_time_l1866_186608


namespace train_length_l1866_186673

theorem train_length (L : ℝ) (h1 : L + 110 / 15 = (L + 250) / 20) : L = 310 := 
sorry

end train_length_l1866_186673


namespace ages_of_father_and_daughter_l1866_186656

variable (F D : ℕ)

-- Conditions
def condition1 : Prop := F = 4 * D
def condition2 : Prop := F + 20 = 2 * (D + 20)

-- Main statement
theorem ages_of_father_and_daughter (h1 : condition1 F D) (h2 : condition2 F D) : D = 10 ∧ F = 40 := by
  sorry

end ages_of_father_and_daughter_l1866_186656


namespace range_of_set_l1866_186665

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l1866_186665


namespace angle_P_measure_l1866_186693

theorem angle_P_measure (P Q : ℝ) (h1 : P + Q = 180) (h2 : P = 5 * Q) : P = 150 := by
  sorry

end angle_P_measure_l1866_186693


namespace find_x_l1866_186663

theorem find_x (x : ℝ) (h : 0.5 * x = 0.05 * 500 - 20) : x = 10 :=
by
  sorry

end find_x_l1866_186663


namespace range_of_g_l1866_186676

noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^4 + (Real.sin x)^2

theorem range_of_g : Set.Icc (3 / 4) 1 = Set.range g :=
by
  sorry

end range_of_g_l1866_186676


namespace complete_the_square_example_l1866_186609

theorem complete_the_square_example : ∀ x m n : ℝ, (x^2 - 12 * x + 33 = 0) → 
  (x + m)^2 = n → m = -6 ∧ n = 3 :=
by
  sorry

end complete_the_square_example_l1866_186609


namespace opera_house_earnings_l1866_186644

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end opera_house_earnings_l1866_186644


namespace highest_number_of_years_of_service_l1866_186600

theorem highest_number_of_years_of_service
  (years_of_service : Fin 8 → ℕ)
  (h_range : ∃ L, ∃ H, H - L = 14)
  (h_second_highest : ∃ second_highest, second_highest = 16) :
  ∃ highest, highest = 17 := by
  sorry

end highest_number_of_years_of_service_l1866_186600


namespace range_of_a_l1866_186688

-- Definitions of conditions
def is_odd_function {A : Type} [AddGroup A] (f : A → A) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing {A : Type} [LinearOrderedAddCommGroup A] (f : A → A) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Main statement
theorem range_of_a 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_monotone_dec : is_monotonically_decreasing f)
  (h_domain : ∀ x, -7 < x ∧ x < 7 → -7 < f x ∧ f x < 7)
  (h_cond : ∀ a, f (1 - a) + f (2 * a - 5) < 0): 
  ∀ a, 4 < a → a < 6 :=
sorry

end range_of_a_l1866_186688


namespace boolean_logic_problem_l1866_186661

theorem boolean_logic_problem (p q : Prop) (h₁ : ¬(p ∧ q)) (h₂ : ¬(¬p)) : ¬q :=
by {
  sorry
}

end boolean_logic_problem_l1866_186661


namespace solve_equation_l1866_186639

theorem solve_equation (x : ℚ) (h : x ≠ 1) : (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by 
  sorry

end solve_equation_l1866_186639


namespace exists_polynomial_P_l1866_186626

open Int Nat

/-- Define a predicate for a value is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Define the polynomial P(x, y, z) --/
noncomputable def P (x y z : ℕ) : ℤ := 
  (1 - 2013 * (z - 1) * (z - 2)) * 
  ((x + y - 1) * (x + y - 1) + 2 * y - 2 + z)

/-- The main theorem to prove --/
theorem exists_polynomial_P :
  ∃ (P : ℕ → ℕ → ℕ → ℤ), 
  (∀ n : ℕ, (¬ is_square n) ↔ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) := 
sorry

end exists_polynomial_P_l1866_186626


namespace sum_first_six_terms_l1866_186628

variable (a1 q : ℤ)
variable (n : ℕ)

noncomputable def geometric_sum (a1 q : ℤ) (n : ℕ) : ℤ :=
  a1 * (1 - q^n) / (1 - q)

theorem sum_first_six_terms :
  geometric_sum (-1) 2 6 = 63 :=
sorry

end sum_first_six_terms_l1866_186628


namespace problem_27_integer_greater_than_B_over_pi_l1866_186604

noncomputable def B : ℕ := 22

theorem problem_27_integer_greater_than_B_over_pi :
  Nat.ceil (B / Real.pi) = 8 := sorry

end problem_27_integer_greater_than_B_over_pi_l1866_186604


namespace tortoise_age_l1866_186618

-- Definitions based on the given problem conditions
variables (a b c : ℕ)

-- The conditions as provided in the problem
def condition1 (a b : ℕ) : Prop := a / 4 = 2 * a - b
def condition2 (b c : ℕ) : Prop := b / 7 = 2 * b - c
def condition3 (a b c : ℕ) : Prop := a + b + c = 264

-- The main theorem to prove
theorem tortoise_age (a b c : ℕ) (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 a b c) : b = 77 :=
sorry

end tortoise_age_l1866_186618


namespace fibonacci_units_digit_l1866_186680

def fibonacci (n : ℕ) : ℕ :=
match n with
| 0     => 4
| 1     => 3
| (n+2) => fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem fibonacci_units_digit : units_digit (fibonacci (fibonacci 10)) = 3 := by
  sorry

end fibonacci_units_digit_l1866_186680


namespace savings_percentage_l1866_186653

variable {I S : ℝ}
variable (h1 : 1.30 * I - 2 * S + I - S = 2 * (I - S))

theorem savings_percentage (h : 1.30 * I - 2 * S + I - S = 2 * (I - S)) : S = 0.30 * I :=
  by
    sorry

end savings_percentage_l1866_186653


namespace lateral_surface_area_cone_l1866_186649

theorem lateral_surface_area_cone (r l : ℝ) (h₀ : r = 6) (h₁ : l = 10) : π * r * l = 60 * π := by 
  sorry

end lateral_surface_area_cone_l1866_186649


namespace find_x_y_l1866_186694

theorem find_x_y (x y : ℝ) (h1 : x + Real.cos y = 2023) (h2 : x + 2023 * Real.sin y = 2022) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 :=
sorry

end find_x_y_l1866_186694


namespace number_of_mappings_n_elements_l1866_186625

theorem number_of_mappings_n_elements
  (A : Type) [Fintype A] [DecidableEq A] (n : ℕ) (h : 3 ≤ n) (f : A → A)
  (H1 : ∀ x : A, ∃ c : A, ∀ (i : ℕ), i ≥ n - 2 → f^[i] x = c)
  (H2 : ∃ x₁ x₂ : A, f^[n] x₁ ≠ f^[n] x₂) :
  ∃ m : ℕ, m = (2 * n - 5) * (n.factorial) / 2 :=
sorry

end number_of_mappings_n_elements_l1866_186625


namespace solve_x_division_l1866_186651

theorem solve_x_division :
  ∀ x : ℝ, (3 / x + 4 / x / (8 / x) = 1.5) → x = 3 := 
by
  intro x
  intro h
  sorry

end solve_x_division_l1866_186651


namespace binomial_10_3_eq_120_l1866_186652

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l1866_186652


namespace a_finishes_work_in_four_days_l1866_186690

theorem a_finishes_work_in_four_days (x : ℝ) 
  (B_work_rate : ℝ) 
  (work_done_together : ℝ) 
  (work_done_by_B_alone : ℝ) : 
  B_work_rate = 1 / 16 → 
  work_done_together = 2 * (1 / x + 1 / 16) → 
  work_done_by_B_alone = 6 * (1 / 16) → 
  work_done_together + work_done_by_B_alone = 1 → 
  x = 4 :=
by
  intros hB hTogether hBAlone hTotal
  sorry

end a_finishes_work_in_four_days_l1866_186690


namespace determine_radius_of_semicircle_l1866_186621

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem determine_radius_of_semicircle :
  radius_of_semicircle 32.392033717615696 = 6.3 :=
by
  sorry

end determine_radius_of_semicircle_l1866_186621


namespace range_of_4a_minus_2b_l1866_186687

theorem range_of_4a_minus_2b (a b : ℝ) (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := 
sorry

end range_of_4a_minus_2b_l1866_186687


namespace value_of_abc_l1866_186630

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c
noncomputable def f_inv (a b c x : ℝ) := c * x^2 + b * x + a

-- The main theorem statement
theorem value_of_abc (a b c : ℝ) (h : ∀ x : ℝ, f a b c (f_inv a b c x) = x) : a + b + c = 1 :=
sorry

end value_of_abc_l1866_186630


namespace roots_expression_value_l1866_186691

theorem roots_expression_value {a b : ℝ} 
  (h₁ : a^2 + a - 3 = 0) 
  (h₂ : b^2 + b - 3 = 0) 
  (ha_ne_hb : a ≠ b) : 
  a * b - 2023 * a - 2023 * b = 2020 :=
by 
  sorry

end roots_expression_value_l1866_186691


namespace circuit_disconnected_scenarios_l1866_186605

def num_scenarios_solder_points_fall_off (n : Nat) : Nat :=
  2 ^ n - 1

theorem circuit_disconnected_scenarios : num_scenarios_solder_points_fall_off 6 = 63 :=
by
  sorry

end circuit_disconnected_scenarios_l1866_186605


namespace gcd_of_differences_l1866_186601

theorem gcd_of_differences (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 1351) : 
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a) = 4 :=
by
  sorry

end gcd_of_differences_l1866_186601


namespace business_proof_l1866_186647

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end business_proof_l1866_186647


namespace count_1000_pointed_stars_l1866_186637

/--
A regular n-pointed star is defined by:
1. The points P_1, P_2, ..., P_n are coplanar and no three of them are collinear.
2. Each of the n line segments intersects at least one other segment at a point other than an endpoint.
3. All of the angles at P_1, P_2, ..., P_n are congruent.
4. All of the n line segments P_2P_3, ..., P_nP_1 are congruent.
5. The path P_1P_2, P_2P_3, ..., P_nP_1 turns counterclockwise at an angle of less than 180 degrees at each vertex.

There are no regular 3-pointed, 4-pointed, or 6-pointed stars.
All regular 5-pointed stars are similar.
There are two non-similar regular 7-pointed stars.

Prove that the number of non-similar regular 1000-pointed stars is 199.
-/
theorem count_1000_pointed_stars : ∀ (n : ℕ), n = 1000 → 
  -- Points P_1, P_2, ..., P_1000 are coplanar, no three are collinear.
  -- Each of the 1000 segments intersects at least one other segment not at an endpoint.
  -- Angles at P_1, P_2, ..., P_1000 are congruent.
  -- Line segments P_2P_3, ..., P_1000P_1 are congruent.
  -- Path P_1P_2, P_2P_3, ..., P_1000P_1 turns counterclockwise at < 180 degrees each.
  -- No 3-pointed, 4-pointed, or 6-pointed regular stars.
  -- All regular 5-pointed stars are similar.
  -- There are two non-similar regular 7-pointed stars.
  -- Proven: The number of non-similar regular 1000-pointed stars is 199.
  n = 1000 ∧ (∀ m : ℕ, 1 ≤ m ∧ m < 1000 → (gcd m 1000 = 1 → (m ≠ 1 ∧ m ≠ 999))) → 
    -- Because 1000 = 2^3 * 5^3 and we exclude 1 and 999.
    (2 * 5 * 2 * 5 * 2 * 5) / 2 - 1 - 1 / 2 = 199 :=
by
  -- Pseudo-proof steps for the problem.
  sorry

end count_1000_pointed_stars_l1866_186637


namespace general_inequality_l1866_186635

theorem general_inequality (x : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_n : 0 < n) : 
  x + n^n / x^n ≥ n + 1 := by 
  sorry

end general_inequality_l1866_186635


namespace proof_equiv_l1866_186692

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ Real.sqrt (3 + 2 * x - x ^ 2) }
noncomputable def N : Set ℝ := { x | ∃ y : ℝ, y = Real.log (x - 2) }
def I : Set ℝ := Set.univ
def complement_N : Set ℝ := I \ N

theorem proof_equiv : M ∩ complement_N = { y | 1 ≤ y ∧ y ≤ 2 } :=
sorry

end proof_equiv_l1866_186692


namespace second_grade_girls_l1866_186650

theorem second_grade_girls (G : ℕ) 
  (h1 : ∃ boys_2nd : ℕ, boys_2nd = 20)
  (h2 : ∃ students_3rd : ℕ, students_3rd = 2 * (20 + G))
  (h3 : 20 + G + (2 * (20 + G)) = 93) :
  G = 11 :=
by
  sorry

end second_grade_girls_l1866_186650


namespace gain_percent_l1866_186655

variable (C S : ℝ)
variable (h : 65 * C = 50 * S)

theorem gain_percent (h : 65 * C = 50 * S) : (S - C) / C * 100 = 30 :=
by
  sorry

end gain_percent_l1866_186655


namespace sam_age_l1866_186631

-- Definitions
variables (B J S : ℕ)
axiom H1 : B = 2 * J
axiom H2 : B + J = 60
axiom H3 : S = (B + J) / 2

-- Problem statement
theorem sam_age : S = 30 :=
sorry

end sam_age_l1866_186631


namespace side_length_of_square_l1866_186645

theorem side_length_of_square (A : ℝ) (h : A = 81) : ∃ s : ℝ, s^2 = A ∧ s = 9 :=
by
  sorry

end side_length_of_square_l1866_186645


namespace citrus_grove_total_orchards_l1866_186623

theorem citrus_grove_total_orchards (lemons_orchards oranges_orchards grapefruits_orchards limes_orchards total_orchards : ℕ) 
  (h1 : lemons_orchards = 8) 
  (h2 : oranges_orchards = lemons_orchards / 2) 
  (h3 : grapefruits_orchards = 2) 
  (h4 : limes_orchards = grapefruits_orchards) 
  (h5 : total_orchards = lemons_orchards + oranges_orchards + grapefruits_orchards + limes_orchards) : 
  total_orchards = 16 :=
by 
  sorry

end citrus_grove_total_orchards_l1866_186623


namespace part1_part2_l1866_186640

-- Part 1: Expression simplification
theorem part1 (a : ℝ) : (a - 3)^2 + a * (4 - a) = -2 * a + 9 := 
by
  sorry

-- Part 2: Solution set of inequalities
theorem part2 (x : ℝ) : 
  (3 * x - 5 < x + 1) ∧ (2 * (2 * x - 1) ≥ 3 * x - 4) ↔ (-2 ≤ x ∧ x < 3) := 
by
  sorry

end part1_part2_l1866_186640


namespace comparison_abc_l1866_186654

noncomputable def a : ℝ := (Real.exp 1 + 2) / Real.log (Real.exp 1 + 2)
noncomputable def b : ℝ := 2 / Real.log 2
noncomputable def c : ℝ := (Real.exp 1)^2 / (4 - Real.log 4)

theorem comparison_abc : c < b ∧ b < a :=
by {
  sorry
}

end comparison_abc_l1866_186654


namespace anna_gets_more_candy_l1866_186632

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end anna_gets_more_candy_l1866_186632


namespace dons_profit_l1866_186660

-- Definitions from the conditions
def bundles_jamie_bought := 20
def bundles_jamie_sold := 15
def profit_jamie := 60

def bundles_linda_bought := 34
def bundles_linda_sold := 24
def profit_linda := 69

def bundles_don_bought := 40
def bundles_don_sold := 36

-- Variables representing the unknown prices
variables (b s : ℝ)

-- Conditions written as equalities
axiom eq_jamie : bundles_jamie_sold * s - bundles_jamie_bought * b = profit_jamie
axiom eq_linda : bundles_linda_sold * s - bundles_linda_bought * b = profit_linda

-- Statement to prove Don's profit
theorem dons_profit : bundles_don_sold * s - bundles_don_bought * b = 252 :=
by {
  sorry -- proof goes here
}

end dons_profit_l1866_186660


namespace unique_solution_l1866_186614

theorem unique_solution (a n : ℕ) (h₁ : 0 < a) (h₂ : 0 < n) (h₃ : 3^n = a^2 - 16) : a = 5 ∧ n = 2 :=
by
sorry

end unique_solution_l1866_186614


namespace quadratic_inequality_solution_set_l1866_186619

/- Given a quadratic function with specific roots and coefficients, prove a quadratic inequality. -/
theorem quadratic_inequality_solution_set :
  ∀ (a b : ℝ),
    (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 + a*x + b < 0) →
    a = -3 →
    b = 2 →
    ∀ x : ℝ, (x < 1/2 ∨ x > 1) ↔ (2*x^2 - 3*x + 1 > 0) :=
by
  intros a b h cond_a cond_b x
  sorry

end quadratic_inequality_solution_set_l1866_186619


namespace polynomial_expansion_a5_l1866_186683

theorem polynomial_expansion_a5 :
  (x - 1) ^ 8 = (1 : ℤ) + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 →
  a₅ = -56 :=
by
  intro h
  -- The proof is omitted.
  sorry

end polynomial_expansion_a5_l1866_186683


namespace central_angle_of_cone_l1866_186685

theorem central_angle_of_cone (A : ℝ) (l : ℝ) (r : ℝ) (θ : ℝ)
  (hA : A = (1 / 2) * 2 * Real.pi * r)
  (hl : l = 1)
  (ha : A = (3 / 8) * Real.pi) :
  θ = (3 / 4) * Real.pi :=
by
  sorry

end central_angle_of_cone_l1866_186685


namespace roger_owes_correct_amount_l1866_186607

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end roger_owes_correct_amount_l1866_186607


namespace find_number_l1866_186606

theorem find_number (x : ℝ) (h : 0.50 * x = 48 + 180) : x = 456 :=
sorry

end find_number_l1866_186606


namespace find_unknown_polynomial_l1866_186648

theorem find_unknown_polynomial (m : ℤ) : 
  ∃ q : ℤ, (q + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1) → q = 2 * m^2 + 3 * m - 4 :=
by {
  sorry
}

end find_unknown_polynomial_l1866_186648


namespace simplify_expression_l1866_186667

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end simplify_expression_l1866_186667


namespace equivalent_expression_l1866_186698

theorem equivalent_expression : 8^8 * 4^4 / 2^28 = 16 := by
  -- Here, we're stating the equivalency directly
  sorry

end equivalent_expression_l1866_186698


namespace cost_price_of_watch_l1866_186615

theorem cost_price_of_watch (CP : ℝ) (h1 : SP1 = CP * 0.64) (h2 : SP2 = CP * 1.04) (h3 : SP2 = SP1 + 140) : CP = 350 :=
by
  sorry

end cost_price_of_watch_l1866_186615
