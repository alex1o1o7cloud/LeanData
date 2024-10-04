import Mathlib

namespace polygon_contains_center_l333_333273

theorem polygon_contains_center (n : ℕ) (a : ℝ) :
  let M1 := regular_polygon (2 * n) a in
  let M2 := regular_polygon (2 * n) (2 * a) in
  inside_polygon M1 M2 →
  contains_center M1 (center M2) :=
by
  sorry

end polygon_contains_center_l333_333273


namespace problem1_problem2_l333_333403

-- Problem 1
theorem problem1 (x: ℝ) (h: x = π / 3) :
  let m := (sqrt 3 * cos x, -1)
  let n := (sin x, cos x ^ 2)
  (m.1 * n.1 + m.2 * n.2) = 1 / 2 :=
by
  rw h
  let m := (sqrt 3 * cos (π / 3), -1)
  let n := (sin (π / 3), cos (π / 3) ^ 2)
  calc
    (m.1 * n.1 + m.2 * n.2) = (√3 / 2) * (√3 / 2) + (-1) * (1 / 4) : sorry

-- Problem 2
theorem problem2 (x: ℝ) (hx: 0 ≤ x ∧ x ≤ π / 4)
  (h: (let m := (sqrt 3 * cos x, -1)
       let n := (sin x, cos x ^ 2)
       (m.1 * n.1 + m.2 * n.2) = sqrt 3 / 3 - 1 / 2)) :
  cos (2 * x) = (3 * sqrt 2 - sqrt 3) / 6 :=
by
  have : 2 * x - π / 6 ∈ [-π / 6, π / 3] := sorry
  -- proceed with proof of cos (2 * x) transformation
  sorry

end problem1_problem2_l333_333403


namespace problem_l333_333285

variables (A B C D E : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := E > B ∧ B > D
def condition3 := D > A
def condition4 := C > B

-- Proof goal: Dana (D) and Beth (B) have the same amount of money
theorem problem (h1 : condition1 A C) (h2 : condition2 E B D) (h3 : condition3 D A) (h4 : condition4 C B) : D = B :=
sorry

end problem_l333_333285


namespace ratio_EP_PD_l333_333293

-- Definitions based on given conditions in the problem statement
variables {A B C F E D : Type}
variables [triangle : Triangle A B C]
variables [points_A F B : Point]
variables [points_C E A : Point]
variables [points_C D B : Point]

-- Conditions provided in the problem
def condition_AF_BF (AF BF : ℝ) : Prop := AF = 2 * BF
def condition_CE_AE (CE AE : ℝ) : Prop := CE = 3 * AE
def condition_CD_BD (CD BD : ℝ) : Prop := CD = 4 * BD

-- Given conditions in triangle ABC
axiom given_cond_AF_BF : condition_AF_BF (AF B) (BF A)
axiom given_cond_CE_AE : condition_CE_AE (CE A) (AE C)
axiom given_cond_CD_BD : condition_CD_BD (CD B) (BD D)

-- Prove the ratio EP : PD = 15 / 8 when intersecting at DE
theorem ratio_EP_PD : ∀ (EP PD : ℝ), EP / PD = 15 / 8 :=
by
  sorry

end ratio_EP_PD_l333_333293


namespace compute_expression_l333_333231

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := 
by
  sorry

end compute_expression_l333_333231


namespace divisible_by_11_and_smallest_n_implies_77_l333_333688

theorem divisible_by_11_and_smallest_n_implies_77 (n : ℕ) (h₁ : n = 7) : ∃ m : ℕ, m = 11 * n := 
sorry

end divisible_by_11_and_smallest_n_implies_77_l333_333688


namespace number_of_square_free_odd_integers_between_1_and_200_l333_333049

def count_square_free_odd_integers (a b : ℕ) (squares : List ℕ) : ℕ :=
  (b - (a + 1)) / 2 + 1 - List.foldl (λ acc sq => acc + ((b - 1) / sq).div 2 + 1) 0 squares

theorem number_of_square_free_odd_integers_between_1_and_200 :
  count_square_free_odd_integers 1 200 [9, 25, 49, 81, 121] = 81 :=
by
  apply sorry

end number_of_square_free_odd_integers_between_1_and_200_l333_333049


namespace ticket_price_difference_l333_333144

-- Defining the problem context and necessary conditions
def number_of_tickets := (x y : ℕ) : Prop :=
  x + y = 30 ∧ 10 * x + 20 * y = 500

-- Stating the theorem to be proved
theorem ticket_price_difference (x y : ℕ) (h : number_of_tickets x y) : y - x = 10 :=
by
  sorry

end ticket_price_difference_l333_333144


namespace sum_of_midpoint_coordinates_in_meters_l333_333519

-- Define the points
def point1 : ℝ × ℝ := (5, -2)
def point2 : ℝ × ℝ := (-3, 6)

-- Define the conversion factor
def unit_to_meters : ℝ := 4

-- Define the midpoint calculation function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the conversion of coordinates to meters
def convert_to_meters (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * unit_to_meters, p.2 * unit_to_meters)

-- State the problem
theorem sum_of_midpoint_coordinates_in_meters :
  let m := convert_to_meters (midpoint point1 point2)
  in m.1 + m.2 = 12 :=
by
  sorry

end sum_of_midpoint_coordinates_in_meters_l333_333519


namespace ducks_in_smaller_pond_l333_333424

theorem ducks_in_smaller_pond (x : ℕ) (h1 : ∃ (x : ℕ), x = 45)
  (h2 : 55 = 55)
  (h3 : ∀ x, 0.20 * x = 0.20 * x)
  (h4 : 0.40 * 55 = 22)
  (h5 : 31% * (x + 55) = 0.31 * (x + 55)) :
  x = 45 :=
by {
  sorry
}

end ducks_in_smaller_pond_l333_333424


namespace parabola_directrix_focus_x0_l333_333934

-- Definition of the given parabola equation
def parabola : (ℝ × ℝ) → Prop := λ (x0 y0 : ℝ) => y0 ^ 2 = 2 * x0

-- Definition of the focus F of the parabola y^2 = 2x 
def focus : (ℝ × ℝ) := (1 / 2, 0)

-- Definition of the directrix equation
def directrix_eq : ℝ → Prop := λ x => x = -1 / 2

-- Definition of point M satisfying the given parabola equation and distance
def point_M (x0 y0 : ℝ) := parabola x0 y0 ∧ dist (x0, y0) focus = 5 / 2

-- Proof goal
theorem parabola_directrix_focus_x0 : 
    (∀ (x0 y0 : ℝ), point_M x0 y0 → x0 = 2) ∧ directrix_eq (-1 / 2) :=
by
  sorry

end parabola_directrix_focus_x0_l333_333934


namespace minimum_value_fraction_l333_333831

theorem minimum_value_fraction (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : 2 * a + b - 6 = 0) :
  (1 / (a - 1) + 2 / (b - 2)) = 4 := 
  sorry

end minimum_value_fraction_l333_333831


namespace work_done_l333_333670

def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 5 else 3 * x + 4

theorem work_done :
  ∫ x in 0..4, F x = 36 := by
  sorry

end work_done_l333_333670


namespace exists_subset_T_l333_333497

variables {S : Finset ℕ} (n : ℕ) (s : ℕ) [fintype S] 
          (S_i : fin 1066 → Finset ℕ)
          (hs : |S| = n)
          (hS_i : ∀ i, S_i i ∈ S ∧ S_i i.card > n / 2)

theorem exists_subset_T (S : Finset ℕ) (hs : |S| = n)
    (hS_i : ∀ i, (S_i i) ⊆ S ∧ (S_i i).card > n / 2) :
  ∃ T : Finset ℕ, T ⊆ S ∧ T.card ≤ 10 ∧ ∀ i, (S_i i ∩ T).nonempty :=
sorry

end exists_subset_T_l333_333497


namespace room_length_l333_333514

theorem room_length (L : ℝ) (width height door_area window_area cost_per_sq_ft total_cost : ℝ) 
    (num_windows : ℕ) (door_w window_w door_h window_h : ℝ)
    (h_width : width = 15) (h_height : height = 12) 
    (h_cost_per_sq_ft : cost_per_sq_ft = 9)
    (h_door_area : door_area = door_w * door_h)
    (h_window_area : window_area = window_w * window_h)
    (h_num_windows : num_windows = 3)
    (h_door_dim : door_w = 6 ∧ door_h = 3)
    (h_window_dim : window_w = 4 ∧ window_h = 3)
    (h_total_cost : total_cost = 8154) :
    (2 * height * (L + width) - (door_area + num_windows * window_area)) * cost_per_sq_ft = total_cost →
    L = 25 := 
by
  intros h_cost_eq
  sorry

end room_length_l333_333514


namespace total_savings_l333_333687

def chlorine_price := 10
def chlorine_discount := 0.20
def soap_price := 16
def soap_discount := 0.25
def quantity_chlorine := 3
def quantity_soap := 5

theorem total_savings :
  (chlorine_price * chlorine_discount * quantity_chlorine) +
  (soap_price * soap_discount * quantity_soap) = 26 := 
by
  sorry

end total_savings_l333_333687


namespace total_floor_area_covered_l333_333975

theorem total_floor_area_covered (A B C : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : B = 24) 
  (h3 : C = 19) : 
  A - (B - C) - 2 * C = 138 := 
by sorry

end total_floor_area_covered_l333_333975


namespace log_convex_proof_sqrt_convex_proof_inequality_1_inequality_2_inequality_1_equality_inequality_2_equality_l333_333921

noncomputable def log_convex (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : Prop := 
∀ x y : ℝ, log(1 + a^x) * (1/2) + log(1 + a^y) * (1/2) ≥ log(1 + a^((x + y) / 2))

noncomputable def sqrt_convex : Prop := 
∀ x y : ℝ, sqrt(1 + x^2) * (1/2) + sqrt(1 + y^2) * (1/2) ≥ sqrt(1 + ((x + y) / 2)^2)

theorem log_convex_proof (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : log_convex a h1 h2 :=
sorry

theorem sqrt_convex_proof : sqrt_convex :=
sorry

theorem inequality_1 (k : ℕ) (a : Fin k → ℝ) (h : ∀ i, 0 < a i) :
  (∏ i, (1 + a i)) ^ (1 / k : ℝ) ≥ (1 + (∏ i, a i) ^ (1 / k : ℝ)) :=
sorry

theorem inequality_2 (k : ℕ) (a : Fin k → ℝ) :
  sqrt(k^2 + (∑ i, a i)^2) ≤ ∑ i, sqrt(1 + (a i)^2) :=
sorry

theorem inequality_1_equality (k : ℕ) (a : Fin k → ℝ) (h : ∀ i j, a i = a j) :
  (∏ i, (1 + a i)) ^ (1 / k : ℝ) = (1 + (∏ i, a i) ^ (1 / k : ℝ)) :=
sorry

theorem inequality_2_equality (k : ℕ) (a : Fin k → ℝ) (h : ∀ i j, a i = a j) :
  sqrt(k^2 + (∑ i, a i)^2) = ∑ i, sqrt(1 + (a i)^2) :=
sorry

end log_convex_proof_sqrt_convex_proof_inequality_1_inequality_2_inequality_1_equality_inequality_2_equality_l333_333921


namespace compute_sqrt_factorial_square_l333_333622

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333622


namespace contrapositive_l333_333177

variable {α : Type*} [Mul α] [Zero α]

-- Variables for the components a and b
variable (a b : α)

-- Main statement to prove
theorem contrapositive (h : a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) : (a = 0 ∨ b = 0) → a * b = 0 := 
begin
  sorry
end

end contrapositive_l333_333177


namespace sqrt_factorial_mul_square_l333_333643

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333643


namespace legos_needed_l333_333107

theorem legos_needed (total_legos : ℕ) (legos_per_airplane : ℕ) (airplanes_needed : ℕ) 
  (current_legos : ℕ) : total_legos = 2 * legos_per_airplane → current_legos = 400 → 
  airplanes_needed = 2 → legos_needed = total_legos - current_legos → 
  total_legos = 480 → legos_needed = 80 :=
by
  sorry

end legos_needed_l333_333107


namespace find_a_for_parallel_lines_l333_333766

-- Definition of the lines l1 and l2
def line1 (a : ℝ) := (3 + a) * x + 4 * y = 5 - 3 * a
def line2 (a : ℝ) := 2 * x + (5 + a) * y = 8

-- Parallel condition for the lines
def are_parallel (a : ℝ) : Prop :=
  (3 + a) / 2 = 4 / (5 + a)

-- The main theorem to prove the value of a
theorem find_a_for_parallel_lines : ∃ (a : ℝ), are_parallel a ∧ a = -7 :=
by
  existsi (-7 : ℝ)
  split
  exact sorry   -- Proof of the parallel condition for a = -7
  refl          -- Proof that a = -7

end find_a_for_parallel_lines_l333_333766


namespace sum_of_series_l333_333300

theorem sum_of_series : 
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := by
  sorry

end sum_of_series_l333_333300


namespace cosine_smallest_angle_l333_333729

theorem cosine_smallest_angle 
    (n : ℕ) 
    (h1 : n % 2 = 0) 
    (h2 : ∀ a b c : ℕ, a = n ∧ b = n+2 ∧ c = n+4 → angle b a c = 1.5 * angle a b c)
    : cos (angle n (n+2) (n+4)) = 45/58 :=
by
  sorry

end cosine_smallest_angle_l333_333729


namespace odd_square_free_count_l333_333047

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 64 ∧ ∀ k : ℕ, (1 < k ∧ k < 200 ∧ k % 2 = 1 ∧ 
  (∀ m : ℕ, m * m ∣ k → m = 1)) ↔ k ∈ {3, 5, 7, ..., 199} :=
sorry

end odd_square_free_count_l333_333047


namespace celine_buys_two_laptops_l333_333701

variable (number_of_laptops : ℕ)
variable (laptop_cost : ℕ := 600)
variable (smartphone_cost : ℕ := 400)
variable (number_of_smartphones : ℕ := 4)
variable (total_money_spent : ℕ := 3000)
variable (change_back : ℕ := 200)

def total_spent : ℕ := total_money_spent - change_back

def cost_of_laptops (n : ℕ) : ℕ := n * laptop_cost

def cost_of_smartphones (n : ℕ) : ℕ := n * smartphone_cost

theorem celine_buys_two_laptops :
  cost_of_laptops number_of_laptops + cost_of_smartphones number_of_smartphones = total_spent →
  number_of_laptops = 2 := by
  sorry

end celine_buys_two_laptops_l333_333701


namespace Mitzi_leftover_money_l333_333141

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end Mitzi_leftover_money_l333_333141


namespace factorial_sqrt_sq_l333_333619

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333619


namespace susan_remaining_spaces_l333_333509

def susan_first_turn_spaces : ℕ := 15
def susan_second_turn_spaces : ℕ := 7 - 5
def susan_third_turn_spaces : ℕ := 20
def susan_fourth_turn_spaces : ℕ := 0
def susan_fifth_turn_spaces : ℕ := 10 - 8
def susan_sixth_turn_spaces : ℕ := 0
def susan_seventh_turn_roll : ℕ := 6
def susan_seventh_turn_spaces : ℕ := susan_seventh_turn_roll * 2
def susan_total_moved_spaces : ℕ := susan_first_turn_spaces + susan_second_turn_spaces + susan_third_turn_spaces + susan_fourth_turn_spaces + susan_fifth_turn_spaces + susan_sixth_turn_spaces + susan_seventh_turn_spaces
def game_total_spaces : ℕ := 100

theorem susan_remaining_spaces : susan_total_moved_spaces = 51 ∧ (game_total_spaces - susan_total_moved_spaces) = 49 := by
  sorry

end susan_remaining_spaces_l333_333509


namespace sqrt_sum_geq_10_l333_333887

theorem sqrt_sum_geq_10
  (n : ℕ)
  (x : ℕ → ℝ)
  (hx_sorted : ∀ i j, i < j → x i ≥ x j)
  (hx_nonneg : ∀ i, 0 ≤ x i)
  (hx_sum_leq_400 : (∑ i in Finset.range n, x i) ≤ 400)
  (hx_sq_sum_geq_10k : (∑ i in Finset.range n, (x i)^2) ≥ 10000)
  : sqrt (x 0) + sqrt (x 1) ≥ 10 := by
    sorry

end sqrt_sum_geq_10_l333_333887


namespace correct_average_of_15_numbers_l333_333173

theorem correct_average_of_15_numbers :
  ∀ (incorrect_ave : ℕ) (n : ℕ) (readings adjustments : ℕ → ℕ → ℕ),
  incorrect_ave = 62 →
  n = 15 →
  readings 1 = 30 ∧ readings 2 = 60 ∧ readings 3 = 25 →
  adjustments 1 = 90 ∧ adjustments 2 = 120 ∧ adjustments 3 = 75 →
  (let incorrect_sum := incorrect_ave * n in
  let total_diff := (adjustments 1 - readings 1) + 
                    (adjustments 2 - readings 2) + 
                    (adjustments 3 - readings 3) in
  let correct_sum := incorrect_sum + total_diff in
  correct_sum / n = 73.33) :=
by
  intros incorrect_ave n readings adjustments h_ave h_n h_readings h_adjustments
  let incorrect_sum := incorrect_ave * n
  let total_diff := (adjustments 1 - readings 1) + 
                    (adjustments 2 - readings 2) + 
                    (adjustments 3 - readings 3)
  let correct_sum := incorrect_sum + total_diff
  have : correct_sum / n = 73.33
  exact this
  sorry

end correct_average_of_15_numbers_l333_333173


namespace sin_cos_identity_l333_333302

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) 
  - Real.cos (200 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 := 
by
  -- This would be where the proof goes
  sorry

end sin_cos_identity_l333_333302


namespace exists_k_with_no_carry_l333_333738

def no_carry_sum (n : ℕ) : Prop :=
  ∀ m k : ℕ, ∀ b : nat, (b > 0) ∧ n * b = m  -> (n * b ).digitSum = 9 * (nat.log10(n * b ) + 1)

theorem exists_k_with_no_carry :
  ∃ k : ℕ, k > 0 ∧ no_carry_sum 3993 :=
begin
  sorry
end

end exists_k_with_no_carry_l333_333738


namespace pat_stickers_at_end_of_week_l333_333910

def initial_stickers : ℕ := 39
def monday_transaction : ℕ := 15
def tuesday_transaction : ℕ := 22
def wednesday_transaction : ℕ := 10
def thursday_trade_net_loss : ℕ := 4
def friday_find : ℕ := 5

def final_stickers (initial : ℕ) (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ) : ℕ :=
  initial + mon - tue + wed - thu + fri

theorem pat_stickers_at_end_of_week :
  final_stickers initial_stickers 
                 monday_transaction 
                 tuesday_transaction 
                 wednesday_transaction 
                 thursday_trade_net_loss 
                 friday_find = 43 :=
by
  sorry

end pat_stickers_at_end_of_week_l333_333910


namespace general_formula_arithmetic_seq_sum_transformed_seq_l333_333783

-- Define the arithmetic sequence {a_n} with given conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) (a1 : ℕ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Define the transformed sequence {b_n}
def transformed_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = 2 ^ (a n) + 1

-- Define the sum of the first n terms of the transformed sequence
def sum_transformed_sequence (T : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, T n = (∑ k in finset.range n, b (k + 1))

-- Given conditions
axiom a3 : a 3 = 3
axiom S4 : S 4 = 10

-- Proof goals
theorem general_formula_arithmetic_seq (a : ℕ → ℕ) (d a1 : ℕ) 
  (ha : arithmetic_sequence a d a1) (ha3 : a 3 = 3) (hS4 : S 4 = 10) :
  ∀ n, a n = n :=
begin
  sorry
end

theorem sum_transformed_seq (a b : ℕ → ℕ) (T : ℕ → ℕ) (ha : ∀ n, a n = n) 
  (hb : transformed_sequence b a) (hT : sum_transformed_sequence T b) :
  ∀ n, T n = 2^(n+1) + n - 2 :=
begin
  sorry
end

end general_formula_arithmetic_seq_sum_transformed_seq_l333_333783


namespace dot_product_magnitude_l333_333454

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

variables (c d : ℝ^3)
variable (theta : ℝ)

-- Conditions
def norm_c : real := 3
def norm_d : real := 4
def norm_cross_c_d : real := 6

-- Proof statement
theorem dot_product_magnitude : 
  ‖c‖ = norm_c → ‖d‖ = norm_d → ‖c ×ᵥ d‖ = norm_cross_c_d → | (inner_product_space.dot c d) | = 6 * real.sqrt 3 :=
begin
  intros,
  sorry
end

end dot_product_magnitude_l333_333454


namespace no_real_set_exists_l333_333320

theorem no_real_set_exists 
  (n : ℕ)
  (x : Fin n → ℝ)
  (h1 : (∑ i, x i) = 2)
  (h2 : (∑ i, (x i) ^ 2) = 3)
  (h3 : (∑ i, (x i) ^ 3) = 4)
  (h4 : (∑ i, (x i) ^ 4) = 5)
  (h5 : (∑ i, (x i) ^ 5) = 6)
  (h6 : (∑ i, (x i) ^ 6) = 7)
  (h7 : (∑ i, (x i) ^ 7) = 8)
  (h8 : (∑ i, (x i) ^ 8) = 9)
  (h9 : (∑ i, (x i) ^ 9) = 10) : 
  False :=
sorry

end no_real_set_exists_l333_333320


namespace second_year_students_fraction_l333_333838

variable (F S : ℚ)

-- Conditions as given in the problem
axiom h1 : F + S = 1
axiom h2 : 4 / 5 * F -- Fraction of first-year who have not declared
axiom h3 : (1 / 3) * (1 / 5 * F) = 1 / 15 * S -- Fraction of second-year who have declared
axiom h4 : 7 / 15 = 14 / 15 * S -- Given fraction of second-year not declared

theorem second_year_students_fraction (F S : ℚ) (h1 : F + S = 1) (h2 : 4 / 5 * F) 
  (h3 : (1 / 3) * (1 / 5 * F) = 1 / 15 * S) (h4 : 7 / 15 = 14 / 15 * S) : 
    S = 1 / 2 := 
by 
  sorry

end second_year_students_fraction_l333_333838


namespace definite_integral_result_l333_333238

theorem definite_integral_result :
  ∫ x in (2 * arctan (1 / 2)) .. (π / 2), (cos x) / (1 - cos x)^3 = 1.3 :=
by
  sorry

end definite_integral_result_l333_333238


namespace factorial_sqrt_sq_l333_333615

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333615


namespace angle_ABG_in_regular_octagon_l333_333486

theorem angle_ABG_in_regular_octagon (N : ℕ) (hN : N = 8) (regular_octagon : RegularPolygon N) : 
  angle ABG = 22.5 :=
by
  sorry

end angle_ABG_in_regular_octagon_l333_333486


namespace maximum_of_fraction_l333_333058

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end maximum_of_fraction_l333_333058


namespace regular_pre_tax_price_l333_333559

theorem regular_pre_tax_price (p : ℚ) (h : (p / 5) * 1.15 = 8) : p = 34.78 :=
by
  have h1 : p / 5 = 8 / 1.15 := by sorry
  have h2 : p = (8 / 1.15) * 5 := by sorry
  have h3 : p = 34.7825 := by sorry
  exact rat.cast_round p

end regular_pre_tax_price_l333_333559


namespace range_of_λ_l333_333401

noncomputable def OriginalInequality (x λ : ℝ) : Prop :=
  (exp x + 1) * x > (log x - log λ) * (x / λ + 1)

theorem range_of_λ (λ : ℝ) :
  (∀ x : ℝ, x > 0 → λ > 0 → OriginalInequality x λ) → λ > 1 / exp 1 :=
by
  sorry

end range_of_λ_l333_333401


namespace calculate_expression_l333_333303

theorem calculate_expression :
  2 * real.sin (real.pi / 3) + real.sqrt 12 + abs (-5) - (real.pi - real.sqrt 2) ^ 0 = 3 * real.sqrt 3 + 4 :=
by
  sorry

end calculate_expression_l333_333303


namespace jar_and_beans_weight_percentage_l333_333972

-- Define the conditions
variables (J B : ℝ)

-- Given conditions
def condition1 : Prop := J = 0.20 * (J + B)
def condition2 : Prop := (J + 0.5 * B)

-- Question translated into proof problem
theorem jar_and_beans_weight_percentage (h1 : condition1 J B) : 
  (J + 0.5 * B) / (J + B) * 100 = 60 :=
by
  sorry

end jar_and_beans_weight_percentage_l333_333972


namespace count_multiples_6_not_12_l333_333033

theorem count_multiples_6_not_12 (n: ℕ) : 
  ∃ (count : ℕ), count = 25 ∧ 
                  count = (finset.filter (λ m, (m < 300) ∧ (6 ∣ m) ∧ ¬ (12 ∣ m)) (finset.range 300)).card :=
by
  sorry

end count_multiples_6_not_12_l333_333033


namespace distance_AB_is_10_l333_333194

noncomputable def distance_AB (perimeter_smaller_square : ℝ) (area_larger_square : ℝ) : ℝ :=
  let side_smaller_square := perimeter_smaller_square / 4
  let side_larger_square := Real.sqrt area_larger_square
  let horizontal_distance := side_smaller_square + side_larger_square
  let vertical_distance := side_larger_square
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2)

theorem distance_AB_is_10 :
  distance_AB 8 36 = 10 :=
by
  unfold distance_AB
  simp
  rw [Real.mul_self_sqrt (le_of_lt (by norm_num : 0 < 36)),
      Real.sqrt_eq_rpow]
  norm_num
  sorry

end distance_AB_is_10_l333_333194


namespace two_correct_inferences_l333_333765

def custom_operation (a b : ℝ) := (a - b)^2

-- Conditions as definitions
def commutativity (a b : ℝ) : Prop := custom_operation a b = custom_operation b a
def quadrature (a b : ℝ) : Prop := (custom_operation a b)^2 = custom_operation (a^2) (b^2)
def negation (a b : ℝ) : Prop := custom_operation (-a) b = custom_operation a (-b)

-- Main theorem stating that exactly two inferences are correct
theorem two_correct_inferences (a b : ℝ) : 
    (if commutativity a b then 1 else 0) + (if quadrature a b then 1 else 0) + (if negation a b then 1 else 0) = 2 := 
by
  sorry

end two_correct_inferences_l333_333765


namespace find_first_part_time_l333_333899

variables {x y z T D : ℝ} -- speeds in km/h, total distance in km, total time in hours
variables {t1 t2 t3 : ℝ} -- times for each part of the journey

-- Define the journey conditions
def journey_conditions := T = t1 + t2 + t3 ∧ D = x * t1 + y * t2 + z * t3

-- The goal is to find t1 given the conditions
theorem find_first_part_time (h : journey_conditions) : ∃ t1, ∃ t2, ∃ t3, h :=
by { sorry } -- proof steps are omitted

end find_first_part_time_l333_333899


namespace committee_ways_l333_333852

-- Given conditions
def total_people : ℕ := 12
def people_to_choose : ℕ := 6
def special_person_on_committee : Prop := true
def remaining_people : ℕ := 11
def people_needed : ℕ := 5

-- Number of ways to choose a subset
def choose : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k := 0
| (n+1), (k+1) := (choose n k + choose n (k+1))

-- Proof statement
theorem committee_ways : choose remaining_people people_needed = 462 :=
by
  sorry

end committee_ways_l333_333852


namespace exists_periodic_sequence_of_period_ge_two_l333_333958

noncomputable def periodic_sequence (x : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, x (n + p) = x n

theorem exists_periodic_sequence_of_period_ge_two :
  ∀ (p : ℕ), p ≥ 2 →
  ∃ (x : ℕ → ℝ), periodic_sequence x p ∧ 
  ∀ n, x (n + 1) = x n - (1 / x n) :=
by {
  sorry
}

end exists_periodic_sequence_of_period_ge_two_l333_333958


namespace yearly_income_is_130_l333_333920

-- Definitions of the given conditions
def total_amount : ℝ := 2500
def amount_at_5_percent : ℝ := 2000
def interest_rate_5_percent : ℝ := 5 / 100
def interest_rate_6_percent : ℝ := 6 / 100

-- Prove that the yearly annual income is Rs. 130
theorem yearly_income_is_130 (total_amount_eq : total_amount = 2500)
                             (amount_at_5_percent_eq : amount_at_5_percent = 2000)
                             (interest_rate_5_percent_eq : interest_rate_5_percent = 5 / 100)
                             (interest_rate_6_percent_eq : interest_rate_6_percent = 6 / 100) :
  let amount_at_6_percent := total_amount - amount_at_5_percent,
      interest_from_5 := (amount_at_5_percent * interest_rate_5_percent * 1),
      interest_from_6 := (amount_at_6_percent * interest_rate_6_percent * 1),
      total_yearly_income := interest_from_5 + interest_from_6 in
  total_yearly_income = 130 :=
by
  sorry

end yearly_income_is_130_l333_333920


namespace tangent_line_at_point_l333_333378

-- Define the odd function f based on the given conditions
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 3*x else -x^2 - 3*x

-- Define the derivative of f (manually derived)
def f_prime (x : ℝ) : ℝ :=
  if x ≤ 0 then 2*x - 3 else -2*x - 3

-- Define the theorem to prove the equation of the tangent line at the point (1, -4)
theorem tangent_line_at_point (x y : ℝ) (h : x = 1 ∧ y = -4):
  5*x + y - 1 = 0 :=
by
  sorry

end tangent_line_at_point_l333_333378


namespace circle_division_exists_pentagon_and_quadrilaterals_l333_333700

theorem circle_division_exists_pentagon_and_quadrilaterals :
  ∃ (circle : EuclideanSpace ℝ (Fin 2)) 
    (points : Fin 5 → circle), 
    let segments := {s : set (Fin 5 × Fin 5) | (∃ a b, s = set.insert (points a, points b) ∅)}
    ∧ (segments.connects_points ([]) points) -- Define segments to connect provided points
    ∧ (∃ polygon1 polygon2 polygon3, 
          polygon1.is_pentagon
        ∧ polygon2.is_quadrilateral
        ∧ polygon3.is_quadrilateral) :=
  sorry

end circle_division_exists_pentagon_and_quadrilaterals_l333_333700


namespace robin_albums_l333_333495

theorem robin_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums_created : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : pics_per_album = 8)
  (h4 : total_pics = phone_pics + camera_pics)
  (h5 : albums_created = total_pics / pics_per_album) : albums_created = 5 := 
sorry

end robin_albums_l333_333495


namespace find_smallest_angle_b1_l333_333676

-- Definitions and conditions
def smallest_angle_in_sector (b1 e : ℕ) (k : ℕ := 5) : Prop :=
  2 * b1 + (k - 1) * k * e = 360 ∧ b1 + 2 * e = 36

theorem find_smallest_angle_b1 (b1 e : ℕ) : smallest_angle_in_sector b1 e → b1 = 30 :=
  sorry

end find_smallest_angle_b1_l333_333676


namespace value_of_Q_l333_333006

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 + 5 * x ^ 2 + 24 * x + 11
noncomputable def g (x : ℝ) : ℝ := x ^ 3 + 7 * x - 22

theorem value_of_Q (a b : ℝ) (h : ∃ (p : ℝ), (x : ℝ) →  f x = (x ^ 2 + a * x + b) * (p * x + 1) ∧ g x = (x ^ 2 + a * x + b) * (x - 2)) :
  Q = a + b := 13 :=
by
  sorry

end value_of_Q_l333_333006


namespace hyperbola_equation_and_perpendicularity_l333_333366

theorem hyperbola_equation_and_perpendicularity
    (eccentricity : ℝ)
    (h_eccentricity : eccentricity = sqrt 2)
    (h_passing_point : ∃ (λ : ℝ), λ ≠ 0 ∧ ∀ (x y : ℝ), (x, y) = (4, -sqrt 10) → x^2 - y^2 = λ)
    (point_on_hyperbola : ∀ (m : ℝ), (3^2 - m^2 = 6) → m = sqrt 3 ∨ m = -sqrt 3)
    (line_perpendicularity : ∀ (m : ℝ), (F1M_slope m) * (F2M_slope m) = -1 ∨ (F1M_slope m) * (F2M_slope m) = -1 
        where F1M_slope (m : ℝ) := m / (3 + 2 * sqrt 3)
              F2M_slope (m : ℝ) := m / (3 - 2 * sqrt 3)) :
    ∃ λ : ℝ, λ = 6 :=
by
  sorry

end hyperbola_equation_and_perpendicularity_l333_333366


namespace distance_between_parallel_lines_l333_333099

/-- Given a circle with four equally spaced parallel lines creating chords of lengths 44, 44, 40 and 40,
    the distance between two adjacent parallel lines is 8 / √23. -/
theorem distance_between_parallel_lines :
  let d := 4 * (Real.sqrt (16 / 23)),
  let r := Real.sqrt (44 / 4 + (1 / 4) * (d * d)),
  let s := Real.sqrt (40 / 4 + (27 / 16) * (d * d)),
  d = 8 / Real.sqrt 23 := by
  have h1 : 44 + (1 / 4) * d^2 = r^2 := sorry
  have h2 : 40 + (27 / 16) * d^2 = s^2 := sorry
  have h3 : r = s := sorry
  have d_value : d^2 = 64 / 23 := by
    linarith [h1, h2, h3]
  exact Real.sqrt_eq_iff.mpr ⟨d_value, by norm_num⟩

end distance_between_parallel_lines_l333_333099


namespace tank_filling_time_l333_333667

theorem tank_filling_time :
  let tank_capacity : ℝ := 1 -- in kiloliters
  let initial_volume : ℝ := tank_capacity / 2
  let fill_rate : ℝ := 1 / 2 -- in kiloliters per minute
  let drain_rate1 : ℝ := 1 / 4 -- in kiloliters per minute
  let drain_rate2 : ℝ := 1 / 6 -- in kiloliters per minute
  let net_rate : ℝ := fill_rate - (drain_rate1 + drain_rate2)
  let remaining_volume : ℝ := tank_capacity / 2 -- 0.5 kiloliters remaining
  let time_to_fill : ℝ := remaining_volume / net_rate
  time_to_fill ≈ 6.0048 := 
sorry

end tank_filling_time_l333_333667


namespace no_real_roots_of_quad_eq_l333_333767

theorem no_real_roots_of_quad_eq (k : ℝ) : ¬(k ≠ 0 ∧ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0) :=
by
  sorry

end no_real_roots_of_quad_eq_l333_333767


namespace simplify_expression_l333_333769

theorem simplify_expression :
  ((3 + 4 + 5 + 6) ^ 2 / 4) + ((3 * 6 + 9) ^ 2 / 3) = 324 := 
  sorry

end simplify_expression_l333_333769


namespace compute_ab_l333_333937

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end compute_ab_l333_333937


namespace ferris_wheel_capacity_l333_333172

theorem ferris_wheel_capacity :
  let seats := 14
  let people_per_seat := 6
  seats * people_per_seat = 84 := by
  let seats := 14
  let people_per_seat := 6
  calc
    seats * people_per_seat = 14 * 6 : by rfl
    ... = 84 : by norm_num

end ferris_wheel_capacity_l333_333172


namespace value_of_expression_l333_333198

theorem value_of_expression :
  4 * 5 + 5 * 4 = 40 :=
sorry

end value_of_expression_l333_333198


namespace max_M_is_7524_l333_333063

-- Define the conditions
def is_valid_t (t : ℕ) : Prop :=
  let a := t / 1000
  let b := (t % 1000) / 100
  let c := (t % 100) / 10
  let d := t % 10
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  (2 * (2 * a + d)) % (2 * b + c) = 0

-- Define function M
def M (a b c d : ℕ) : ℕ := 2000 * a + 100 * b + 10 * c + d

-- Define the maximum value of M
def max_valid_M : ℕ :=
  let m_values := [5544, 7221, 7322, 7524]
  m_values.foldl max 0

theorem max_M_is_7524 : max_valid_M = 7524 := by
  -- The proof would be written here. For now, we indicate the theorem as
  -- not yet proven.
  sorry

end max_M_is_7524_l333_333063


namespace circumscribed_circle_radius_l333_333698

variables (θ : ℝ)

theorem circumscribed_circle_radius (hθ : θ > 0 ∧ θ < 2 * π) :
  ∃ R : ℝ, R = 5 * (sec (θ / 2)) :=
sorry

end circumscribed_circle_radius_l333_333698


namespace approximate_value_exists_l333_333835

variables {α : Type*} [LinearOrderedField α] {f : α → α} {x_i x_{i+1} ξ_i : α}

theorem approximate_value_exists (h_interval : x_i ≤ x_{i+1}) (h_ξ_i : ξ_i ∈ set.Icc x_i x_{i+1}) :
  ∃ ξ_i ∈ set.Icc x_i x_{i+1}, f ξ_i = f ξ_i :=
begin
  use ξ_i,
  split,
  { exact h_ξ_i, },
  { refl, }
end

end approximate_value_exists_l333_333835


namespace square_division_l333_333492

theorem square_division (n : ℕ) (h : n ≥ 6) : ∃ squares : finset (set (ℝ × ℝ)), finset.card squares = n ∧ (⋃ s ∈ squares, s) = (set.univ : set (ℝ × ℝ)) :=
by
  have base_6 := sorry -- Base case verification for n = 6 (detailed construction is skipped here)
  have base_7 := sorry -- Base case verification for n = 7 (detailed construction is skipped here)
  have base_8 := sorry -- Base case verification for n = 8 (detailed construction is skipped here)
  have inductive_step := sorry -- Inductive step showing (k -> k + 1) (detailed construction is skipped here)
  sorry -- Conclude using induction from the base cases and inductive step

end square_division_l333_333492


namespace trains_at_start_2016_l333_333946

def traversal_time_red := 7
def traversal_time_blue := 8
def traversal_time_green := 9

def return_period_red := 2 * traversal_time_red
def return_period_blue := 2 * traversal_time_blue
def return_period_green := 2 * traversal_time_green

def train_start_pos_time := 2016
noncomputable def lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)

theorem trains_at_start_2016 :
  train_start_pos_time % lcm_period = 0 :=
by
  have return_period_red := 2 * traversal_time_red
  have return_period_blue := 2 * traversal_time_blue
  have return_period_green := 2 * traversal_time_green
  have lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)
  have train_start_pos_time := 2016
  exact sorry

end trains_at_start_2016_l333_333946


namespace pentagon_area_ratio_l333_333209

theorem pentagon_area_ratio 
  (s : ℝ)
  (area_AJICB : ℝ)
  (total_square_area : ℝ)
  (H1 : ∀ (E F G H : (ℝ × ℝ)), dist E G = s ∧ dist F H = s ∧ dist G H = s)
  (H2 : ∃ C D, dist C (IH_s) = s / 3 ∧ dist D (HE_s) = 2 * s / 3)
  (H3 : area_AJICB = 1)
  (H4 : total_square_area = 3 * s * s) :
  (area_AJICB / total_square_area) = 1 / 3 :=
by
  sorry

end pentagon_area_ratio_l333_333209


namespace chess_team_boys_l333_333253

variable (B G : ℕ)

theorem chess_team_boys (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 20) : B = 15 := by
  sorry

end chess_team_boys_l333_333253


namespace find_number_l333_333510

theorem find_number (A : ℕ) (B : ℕ) (H1 : B = 300) (H2 : Nat.lcm A B = 2310) (H3 : Nat.gcd A B = 30) : A = 231 := 
by 
  sorry

end find_number_l333_333510


namespace find_real_pairs_l333_333756

theorem find_real_pairs (x y : ℝ) (h : 2 * x / (1 + x^2) = (1 + y^2) / (2 * y)) : 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end find_real_pairs_l333_333756


namespace solve_inequality_l333_333924

theorem solve_inequality (x : ℝ) :
  10^(7 * x - 1) + 6 * 10^(1 - 7 * x) - 5 ≤ 0 ↔
  (1 + Real.log 2) / 7 ≤ x ∧ x ≤ (1 + Real.log 3) / 7 :=
by
    sorry

end solve_inequality_l333_333924


namespace sum_first_30_terms_l333_333097

def a (n : ℕ) : ℚ := if n = 0 then 1/2 else n / ((n+1) * (n + 2))

theorem sum_first_30_terms (S : ℚ) :
  ( ∀ n, a 0 = 1/2 ∧ (n + 2) * a (n + 1) = n * a n ) →
  S = ∑ i in range 30, a i → 
  S = 30 / 31 :=
by
  intros h1 h2
  sorry

end sum_first_30_terms_l333_333097


namespace morse_code_sequences_count_l333_333077

theorem morse_code_sequences_count : 
  (∑ n in (finset.range 5).map nat.succ, 2^n) = 62 := by
  sorry

end morse_code_sequences_count_l333_333077


namespace molecular_weight_correct_l333_333724

-- Definition of atomic weights for the elements
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Number of atoms in Ascorbic acid (C6H8O6)
def count_C : ℕ := 6
def count_H : ℕ := 8
def count_O : ℕ := 6

-- Calculation of molecular weight
def molecular_weight_ascorbic_acid : ℝ :=
  (count_C * atomic_weight_C) +
  (count_H * atomic_weight_H) +
  (count_O * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_ascorbic_acid = 176.124 :=
by sorry


end molecular_weight_correct_l333_333724


namespace identify_fake_coin_l333_333996

def is_fake_coin (coin_idx : Fin 7) (weights : Fin 7 → ℝ) : Prop :=
  ∃ real_weight counter_weight, (∀ i, i ≠ coin_idx → weights i = real_weight) ∧ weights coin_idx = counter_weight ∧ counter_weight < real_weight

def balance_scale (left_pan right_pan : Finset (Fin 7)) (weights : Fin 7 → ℝ) : Ordering :=
  let left_weight := left_pan.sum weights
  let right_weight := right_pan.sum weights
  if right_weight = 3 * left_weight then Ordering.eq
  else if right_weight > 3 * left_weight then Ordering.gt
  else Ordering.lt

theorem identify_fake_coin (weights : Fin 7 → ℝ) (h_fake : ∃ coin_idx, is_fake_coin coin_idx weights) :
  ∃ coin_idx, is_fake_coin coin_idx weights ∧ ∀ pan1 pan2 (h1 : ∑ i in pan1, weights i = ∑ i in pan2, 3 * weights i) (h2 : ∑ i in {i | i ∈ pan1} = ∑ i in {i | i ∈ pan2} 3 * (λ i, if i = coin_idx then counter_weight else real_weight)), sorry :=
sorry

end identify_fake_coin_l333_333996


namespace sqrt_factorial_product_squared_l333_333583

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333583


namespace sqrt_factorial_mul_square_l333_333644

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333644


namespace field_area_l333_333696

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end field_area_l333_333696


namespace area_of_garden_l333_333089

variables (P : ℕ) (S : ℕ) (A : ℕ)

-- Condition: Julia must walk 1500 meters to cover the distance.
axiom distance_walked : 1500

-- Condition: Julia can walk its perimeter 15 times.
axiom perimeter : P = distance_walked / 15

-- Condition: The garden is square.
axiom square_garden : P = 4 * S

-- Question: What is the area of Julia's garden in square meters?
theorem area_of_garden : A = S * S :=
sorry

end area_of_garden_l333_333089


namespace general_formula_for_a_general_formula_for_b_sum_T_n_l333_333457

variables (a b : ℕ → ℕ) (d q : ℕ) (S : ℕ → ℕ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, b n = b 1 * q ^ (n - 1)

def sum_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

axiom a1 : a 1 = 1
axiom b1 : b 1 = 1
axiom condition1 : a 4 + b 4 = 15
axiom condition2 : S 4 - b 4 = 8
axiom arithmetic_def : arithmetic_sequence a d
axiom geometric_def : geometric_sequence b q

-- Proof goals
theorem general_formula_for_a (h_d : d = 2) : ∀ n, a n = 2 * n - 1 := sorry
theorem general_formula_for_b (h_q : q = 2) : ∀ n, b n = 2 ^ (n - 1) := sorry
theorem sum_T_n (h_d : d = 2) (h_q : q = 2) : 
  ∀ n, (range n).sum (λ i, a (i + 1) / b (i + 1)) = 6 - (2 * n - 3) / 2 ^ (n - 1) := sorry

end general_formula_for_a_general_formula_for_b_sum_T_n_l333_333457


namespace proportional_parts_l333_333267

theorem proportional_parts (A B C D : ℕ) (number : ℕ) (h1 : A = 5 * x) (h2 : B = 7 * x) (h3 : C = 4 * x) (h4 : D = 8 * x) (h5 : C = 60) : number = 360 := by
  sorry

end proportional_parts_l333_333267


namespace number_of_nickels_l333_333174

-- Define the conditions
variable (m : ℕ) -- Total number of coins initially
variable (v : ℕ) -- Total value of coins initially in cents
variable (n : ℕ) -- Number of nickels

-- State the conditions in terms of mathematical equations
-- Condition 1: Average value is 25 cents
axiom avg_value_initial : v = 25 * m

-- Condition 2: Adding one half-dollar (50 cents) results in average of 26 cents
axiom avg_value_after_half_dollar : v + 50 = 26 * (m + 1)

-- Define the relationship between the number of each type of coin and the total value
-- We sum the individual products of the count of each type and their respective values
axiom total_value_definition : v = 5 * n  -- since the problem already validates with total_value == 25m

-- Question to prove
theorem number_of_nickels : n = 30 :=
by
  -- Since we are not providing proof, we will use sorry to indicate the proof is omitted
  sorry

end number_of_nickels_l333_333174


namespace sqrt_factorial_product_squared_l333_333579

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333579


namespace sqrt_factorial_product_squared_l333_333631

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333631


namespace find_a_l333_333018

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then real.sqrt x else real.sqrt (-x)

theorem find_a (a : ℝ)
  (h : f a + f (-1) = 2) :
  a = 1 ∨ a = -1 :=
by sorry

end find_a_l333_333018


namespace vector_perpendicular_vector_parallel_l333_333406


variables {x : ℝ}
def vector_a : ℝ × ℝ × ℝ := (2, -1, 5)
def vector_b : ℝ × ℝ × ℝ := (-4, 2, x)

-- Prove that if vectors are perpendicular, x = 2
theorem vector_perpendicular : (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 + vector_a.3 * vector_b.3 = 0) → x = 2 :=
by sorry

-- Prove that if vectors are parallel, x = -10
theorem vector_parallel : (vector_b.1 / vector_a.1 = vector_b.2 / vector_a.2 ∧ vector_b.1 / vector_a.1 = vector_b.3 / vector_a.3) → x = -10 :=
by sorry

end vector_perpendicular_vector_parallel_l333_333406


namespace brownies_pieces_l333_333249

theorem brownies_pieces (tray_length : ℕ) (tray_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (h_tray_dim : tray_length = 24) (h_tray_wid : tray_width = 30)
  (h_piece_len : piece_length = 3) (h_piece_wid : piece_width = 4) :
  tray_length * tray_width / (piece_length * piece_width) = 60 :=
by
  rw [h_tray_dim, h_tray_wid, h_piece_len, h_piece_wid]
  norm_num
  sorry

end brownies_pieces_l333_333249


namespace water_height_is_correct_l333_333685

def full_cone_radius : ℝ := 20
def full_cone_height : ℝ := 100
def water_occupancy : ℝ := 0.4

def cone_volume (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h

def full_volume := cone_volume full_cone_radius full_cone_height
def water_volume := water_occupancy * full_volume

noncomputable def x : ℝ := real.cbrt (water_volume / full_volume)
noncomputable def water_height : ℝ := full_cone_height * x

theorem water_height_is_correct :
  water_height = 50 * real.cbrt 3.2 :=
by {
  -- Proof should go here
  sorry
}

end water_height_is_correct_l333_333685


namespace delta_y_over_delta_x_l333_333070

noncomputable def f (x : ℝ) : ℝ := -x^2 + x

theorem delta_y_over_delta_x (Δx : ℝ) (h₁ : f (-1) = -2) (h_adj : f (-1 + Δx) = -(-1 + Δx)^2 + (-1 + Δx)) :
  (f (-1 + Δx) - f (-1)) / Δx = 3 - Δx :=
by
  have h₂ : f (-1) = -(-1)^2 + (-1), from h₁,
  have h₃ : f (-1 + Δx) = -((-1 + Δx)^2) + (-1 + Δx), from h_adj,
  sorry

end delta_y_over_delta_x_l333_333070


namespace Ronald_sessions_needed_l333_333157

/-
Conditions:
1. Ronald can grill 27.5 hamburgers per session.
2. Ronald needs to cook 578 hamburgers in total.
3. Ronald has already cooked 163 hamburgers.

Question: How many more sessions will it take Ronald to finish cooking all 578 hamburgers?
Answer: 16 sessions.
-/

theorem Ronald_sessions_needed : let hamburgers_per_session := 27.5;
                                    let total_hamburgers := 578;
                                    let cooked_hamburgers := 163;
                                    let remaining_hamburgers := total_hamburgers - cooked_hamburgers
                                    let sessions_needed := (remaining_hamburgers / hamburgers_per_session).ceil
                                in sessions_needed = 16 :=
by 
  let hamburgers_per_session := 27.5
  let total_hamburgers := 578
  let cooked_hamburgers := 163
  let remaining_hamburgers := total_hamburgers - cooked_hamburgers
  let sessions_needed := (remaining_hamburgers / hamburgers_per_session).ceil
  exact Eq.refl 16

end Ronald_sessions_needed_l333_333157


namespace three_digit_multiples_of_25_but_not_45_l333_333813

theorem three_digit_multiples_of_25_but_not_45 : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ n % 25 = 0 ∧ n % 45 ≠ 0 }.card = 32 := 
sorry

end three_digit_multiples_of_25_but_not_45_l333_333813


namespace clock_time_combinations_l333_333248

theorem clock_time_combinations :
  (∑ h in Finset.range 24, ∑ m in Finset.range 60, ∑ s in Finset.range 60, if h + m = s then 1 else 0) = 1164 :=
by
  sorry

end clock_time_combinations_l333_333248


namespace largest_product_of_three_l333_333651

theorem largest_product_of_three :
  ∃ a b c ∈ ({-5, -4, -1, 3, 7, 9} : set ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 189 ∧
  ∀ x y z ∈ ({-5, -4, -1, 3, 7, 9} : set ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z →
  x * y * z ≤ 189 :=
begin
  sorry
end

end largest_product_of_three_l333_333651


namespace cows_with_no_spots_l333_333903

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end cows_with_no_spots_l333_333903


namespace greatest_integer_property_l333_333334

theorem greatest_integer_property :
  ∃ n : ℤ, n < 1000 ∧ (∃ m : ℤ, 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧ 
  (∀ k : ℤ, k < 1000 ∧ (∃ m : ℤ, 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) → k ≤ n) := by
  -- skipped the proof with sorry
  sorry

end greatest_integer_property_l333_333334


namespace f_exactly_six_zeros_l333_333458

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then cos (2 * Real.pi * (x - a))
  else x^2 - 2 * (a + 1) * x + a^2 + 5

def has_six_zeros (a : ℝ) : Prop :=
  let zero_intervals := ((2, 9 / 4], (5 / 2, 11 / 4] : Set ℝ)
  set.in zero_intervals a

theorem f_exactly_six_zeros (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x = 0) :
  has_six_zeros a :=
sorry

end f_exactly_six_zeros_l333_333458


namespace sum_of_roots_l333_333822

theorem sum_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -3) (hx1 : x1 + x2 = 2) :
  x1 + x2 = 2 :=
by {
  sorry
}

end sum_of_roots_l333_333822


namespace interval_comparison_l333_333774

theorem interval_comparison (x : ℝ) :
  ((x - 1) * (x + 3) < 0) → ¬((x + 1) * (x - 3) < 0) ∧ ¬((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0) :=
by
  sorry

end interval_comparison_l333_333774


namespace min_dist_ellipse_l333_333780

noncomputable def min_dist (x y : ℝ) : ℝ :=
  sqrt(x^2 + y^2 + 4*y + 4) - x/2

theorem min_dist_ellipse :
  ∀ x y : ℝ, 
  (x^2 / 4 + y^2 / 3 = 1) →
  min_dist x y ≥ 2 * sqrt 2 - 1 := 
by
  sorry

end min_dist_ellipse_l333_333780


namespace find_T_l333_333478

theorem find_T (T : ℝ) : (1 / 2) * (1 / 7) * T = (1 / 3) * (1 / 5) * 90 → T = 84 :=
by sorry

end find_T_l333_333478


namespace locker_combinations_count_l333_333445

theorem locker_combinations_count : 
  let primes := {x ∈ (Finset.range 51) | Nat.prime x}.card
  let evens := {x ∈ (Finset.range 51) | x % 2 = 0}.card
  let multiples_of_5 := {x ∈ (Finset.range 51) | x % 5 = 0}.card
  primes = 15 ∧ evens = 25 ∧ multiples_of_5 = 10 →
  primes * evens * multiples_of_5 = 3750 := 
by
  sorry

end locker_combinations_count_l333_333445


namespace fraction_orange_juice_in_large_container_l333_333981

-- Definitions according to the conditions
def pitcher1_capacity : ℕ := 800
def pitcher2_capacity : ℕ := 500
def pitcher1_fraction_orange_juice : ℚ := 1 / 4
def pitcher2_fraction_orange_juice : ℚ := 3 / 5

-- Prove the fraction of orange juice
theorem fraction_orange_juice_in_large_container :
  ( (pitcher1_capacity * pitcher1_fraction_orange_juice + pitcher2_capacity * pitcher2_fraction_orange_juice) / 
    (pitcher1_capacity + pitcher2_capacity) ) = 5 / 13 :=
by
  sorry

end fraction_orange_juice_in_large_container_l333_333981


namespace roots_in_interval_l333_333358

def P (x : ℝ) : ℝ := x^2014 - 100 * x + 1

theorem roots_in_interval : 
  ∀ x : ℝ, P x = 0 → (1/100) ≤ x ∧ x ≤ 100^(1 / 2013) := 
  sorry

end roots_in_interval_l333_333358


namespace sqrt_factorial_product_squared_l333_333630

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333630


namespace b_share_1500_l333_333999

theorem b_share_1500 (total_amount : ℕ) (parts_A parts_B parts_C : ℕ)
  (h_total_amount : total_amount = 4500)
  (h_ratio : (parts_A, parts_B, parts_C) = (2, 3, 4)) :
  parts_B * (total_amount / (parts_A + parts_B + parts_C)) = 1500 :=
by
  sorry

end b_share_1500_l333_333999


namespace circle_tangent_radius_sum_correct_sum_of_all_possible_values_radius_l333_333257

noncomputable def circle_tangent_radius_sum : ℝ :=
  sorry

theorem circle_tangent_radius_sum_correct :
  ∀ (r : ℝ), (r > 0 ∧ (r - 5)^2 + r^2 = (r + 2)^2) → 
  (∃ r1 r2 : ℝ, r = r1 + r2 ∧ 7 + 2 * real.sqrt 7 = r1 ∧ 7 - 2 * real.sqrt 7 = r2) :=
sorry

theorem sum_of_all_possible_values_radius :
  circle_tangent_radius_sum = 14 :=
sorry

end circle_tangent_radius_sum_correct_sum_of_all_possible_values_radius_l333_333257


namespace final_price_correct_l333_333150

-- Definitions of the conditions
def initial_price := 2000
def first_discount_rate := 0.15
def second_discount_rate := 0.10
def gift_card := 200

-- Calculate the price after the first discount
def price_after_first_discount := initial_price - (initial_price * first_discount_rate)

-- Calculate the price after the second discount
def price_after_second_discount := price_after_first_discount - (price_after_first_discount * second_discount_rate)

-- Final price after applying gift card
def final_price := price_after_second_discount - gift_card

-- Prove that the final price is $1330
theorem final_price_correct : final_price = 1330 := by
  sorry

end final_price_correct_l333_333150


namespace num_valid_arrangements_l333_333918

-- Define the conditions
def no_two_adjacent (l : List Bool) : Prop :=
  ∀ (n : ℕ), n < l.length - 1 → l.nth n ≠ l.nth (n + 1)

def engraved_condition (l : List Bool) : Prop :=
  ∀ (n : ℕ), n < l.length - 1 → not (l.nth n = some true ∧ l.nth (n + 1) = some true)

def at_least_three_consecutive_engraved (l : List Bool) : Prop :=
  ∃ (n : ℕ), n < l.length - 2 ∧ l.nth n = some true ∧ l.nth (n + 1) = some true ∧ l.nth (n + 2) = some true

-- Define the main theorem
theorem num_valid_arrangements : 
  (∃ (l : List Bool), 
    l.length = 10 ∧ 
    l.filter (λ b => b = true) = 5 ∧ 
    l.filter (λ b => b = false) = 5 ∧ 
    no_two_adjacent l ∧ 
    engraved_condition l ∧ 
    at_least_three_consecutive_engraved l) → 
  48 := 
sorry

end num_valid_arrangements_l333_333918


namespace find_angle_B_l333_333372

noncomputable def triangle_angle (a b A : ℝ) : Set ℝ :=
  {B : ℝ | ∃ C, ∃ (c : ℝ), tan (B / 2 + A / 2) = tan (A / 2 + C / 2) ∧ a / sin A = b / sin B}

theorem find_angle_B :
  triangle_angle (Real.sqrt 3) (Real.sqrt 6) (Real.pi / 6) = {Real.pi / 4, 3 * Real.pi / 4} :=
by 
  sorry

end find_angle_B_l333_333372


namespace factorial_sqrt_sq_l333_333611

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333611


namespace common_chord_length_l333_333212

theorem common_chord_length (r : ℝ) (h : r = 12) 
  (condition : ∀ (C₁ C₂ : Set (ℝ × ℝ)), 
      ((C₁ = {p : ℝ × ℝ | dist p (0, 0) = r}) ∧ 
       (C₂ = {p : ℝ × ℝ | dist p (12, 0) = r}) ∧
       (C₂ ∩ C₁ ≠ ∅))) : 
  ∃ chord_len : ℝ, chord_len = 12 * Real.sqrt 3 :=
by
  sorry

end common_chord_length_l333_333212


namespace perimeter_of_resulting_polygon_l333_333692

theorem perimeter_of_resulting_polygon :
  ∀ (triangle_leg1 triangle_leg2 rec_longer_side : ℕ),
  triangle_leg1 = 3 →
  triangle_leg2 = 4 →
  rec_longer_side = 10 →
  2 * (rec_longer_side + (rec_longer_side - triangle_leg2)) - triangle_leg1 = 29 :=
by
  intros triangle_leg1 triangle_leg2 rec_longer_side h1 h2 h3
  rw [h1, h2, h3]
  sorry

end perimeter_of_resulting_polygon_l333_333692


namespace volume_of_P2_seq_l333_333370

theorem volume_of_P2_seq {P : ℕ → ℝ} (h0 : P 0 = 2) 
    (h_rec : ∀ i, P (i + 1) = P i + 4 * ((1/2) ^ 3 ^ (i + 1))) : 
    P 2 = 3.5 :=
by 
    have hP1: P 1 = 2 + 4 * (1/8), from by rw [h0]; ring,
    have hP2: P 2 = (2 + 0.5) + 16 * (1/16); ring,
    exact hP2

end volume_of_P2_seq_l333_333370


namespace equation_of_curve_area_of_triangle_l333_333856

variables (a b : ℝ)
variables {α β γ : ℝ}

-- Conditions
axiom h_cond1 : a > b
axiom h_cond2 : b > 0

-- Proof for part (1)
theorem equation_of_curve : ∀ x y : ℝ, (x = a * Real.cos α) ∧ (y = b * Real.sin α) → 
  (x^2 / a^2) + (y^2 / b^2) = 1 := 
sorry

-- Proof for part (2)
theorem area_of_triangle (A B C: ℝ × ℝ) (hA: A.1 = a * Real.cos α ∧ A.2 = b * Real.sin α)
  (hB: B.1 = a * Real.cos β ∧ B.2 = b * Real.sin β)
  (hC: C.1 = a * Real.cos γ ∧ C.2 = b * Real.sin γ)
  (h_sum : A.1 + B.1 + C.1 = 0 ∧ A.2 + B.2 + C.2 = 0 ) :
  Real.abs (1 / 2 * ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))) = 
    (3 * Real.sqrt 3 / 4 * a * b) :=
sorry

end equation_of_curve_area_of_triangle_l333_333856


namespace part1_part2_part3_l333_333396

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + 1

theorem part1 (a : ℝ) : 
  (∀ x, x = 1 → (∃ b, tangent_line (f a) 1 = 4 * x + b)) → a = 6 ∧ ∃ b, tangent_line (f a) 1 = 4 * 1 + b :=
sorry

theorem part2 (a : ℝ) : 
  a ≤ 0 → ∀ x, x > 0 → ∀ y, y > 0 → (∃ m, monotone (f a) (4 * x - y + b = 0)) :=
sorry

theorem part3 (a : ℝ) : 
  a < 0 → (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → (|f a x1 - f a x2| ≥ |x1 - x2|)) → a ≤ -1/8 :=
sorry

end part1_part2_part3_l333_333396


namespace no_valid_real_solutions_for_x_l333_333960

theorem no_valid_real_solutions_for_x (x : ℝ) :
  (∃ (x : ℝ), 2 * (x - 1)^2 = (x + 2) * (x - 5)) → False :=
by
  intro h,
  cases h with x hx,
  have eq1 : 2 * (x - 1)^2 = (x + 2) * (x - 5) := hx,
  sorry

end no_valid_real_solutions_for_x_l333_333960


namespace measure_angle_ABG_l333_333487

-- Formalizing the conditions
def is_regular_octagon (polygon : Fin 8 → ℝ × ℝ) : Prop :=
  let vertices := [polygon 0, polygon 1, polygon 2, polygon 3, polygon 4, polygon 5, polygon 6, polygon 7]
  (∀ i, ∥vertices ((i + 1) % 8) - vertices i∥ = ∥vertices 1 - vertices 0∥) ∧ 
  (∀ i, ∠ (vertices (i + 1) % 8) (vertices i) (vertices (i - 1 + 8) % 8) = 135)

-- Define angle_measure, considering the numbering polygon Fin 8 from 0 to 7
def angle_measure_polygon (polygon : Fin 8 → ℝ × ℝ) (i j k : Fin 8) : ℝ :=
  ∠ (polygon j) (polygon i) (polygon k)

-- The proof problem statement
theorem measure_angle_ABG (polygon : Fin 8 → ℝ × ℝ) (h : is_regular_octagon polygon) : 
  angle_measure_polygon polygon 0 1 6 = 22.5 :=
sorry

end measure_angle_ABG_l333_333487


namespace find_vector_b_coordinates_l333_333797

theorem find_vector_b_coordinates 
  (a b : ℝ × ℝ) 
  (h₁ : a = (-3, 4)) 
  (h₂ : ∃ m : ℝ, m < 0 ∧ b = (-3 * m, 4 * m)) 
  (h₃ : ‖b‖ = 10) : 
  b = (6, -8) := 
by
  sorry

end find_vector_b_coordinates_l333_333797


namespace roadway_deck_needs_1600_tons_l333_333104

theorem roadway_deck_needs_1600_tons (concrete_anchor : ℕ) (total_concrete : ℕ) (concrete_pillars : ℕ) :
  (concrete_anchor = 700) →
  (total_concrete = 4800) →
  (concrete_pillars = 1800) →
  (2 * concrete_anchor + concrete_pillars + (total_concrete - (2 * concrete_anchor + concrete_pillars))) = total_concrete →
  (total_concrete - (2 * concrete_anchor + concrete_pillars) = 1600) :=
by {
  intros h1 h2 h3 h4,
  calc
    total_concrete - (2 * concrete_anchor + concrete_pillars)
        = 4800 - (2 * 700 + 1800) : by rw [h1, h2, h3]
    ... = 1600 : by norm_num,
}

end roadway_deck_needs_1600_tons_l333_333104


namespace M_inter_N_eq_interval_l333_333898

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem M_inter_N_eq_interval : (M ∩ N) = {x | 0 ≤ x ∧ x < 1} := 
  sorry

end M_inter_N_eq_interval_l333_333898


namespace conjugate_of_z_given_condition_l333_333068

def complex_conjugate (z : ℂ) : ℂ := conj z

theorem conjugate_of_z_given_condition : 
  ∀ (z : ℂ), (z * complex.i = 1 + 2 * complex.i) → complex_conjugate z = 2 + complex.i :=
by
  intros z hz
  sorry

end conjugate_of_z_given_condition_l333_333068


namespace prove_length_square_qp_l333_333423

noncomputable def length_square_qp (r1 r2 d : ℝ) (x : ℝ) : Prop :=
  r1 = 10 ∧ r2 = 8 ∧ d = 15 ∧ (2*r1*x - (x^2 + r2^2 - d^2) = 0) → x^2 = 164

theorem prove_length_square_qp : length_square_qp 10 8 15 x :=
sorry

end prove_length_square_qp_l333_333423


namespace value_of_f_at_1_l333_333015

def f (x : ℝ) : ℝ := 2^x + 2

theorem value_of_f_at_1 : f 1 = 4 := by
  sorry

end value_of_f_at_1_l333_333015


namespace alice_painted_cuboids_l333_333287

theorem alice_painted_cuboids (total_faces : ℕ) (faces_per_cuboid : ℕ) (h1 : faces_per_cuboid = 6) (h2 : total_faces = 36) :
  ∃ n : ℕ, total_faces = faces_per_cuboid * n ∧ n = 6 :=
by
  have h3 : 6 * 6 = 36 := by norm_num
  use 6
  split
  · rw [h1, h2]
    exact h3
  · rfl

end alice_painted_cuboids_l333_333287


namespace tangent_line_passes_through_origin_l333_333834

noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

theorem tangent_line_passes_through_origin (α : ℝ)
  (h_tangent : ∀ (x : ℝ), curve α 1 + (α * (x - 1)) - 2 = curve α x) :
  α = 2 :=
sorry

end tangent_line_passes_through_origin_l333_333834


namespace intersecting_point_value_l333_333521

theorem intersecting_point_value (c d : ℤ) (h1 : d = 5 * (-5) + c) (h2 : -5 = 5 * d + c) : 
  d = -5 := 
sorry

end intersecting_point_value_l333_333521


namespace AM_eq_CD_l333_333365

variable {α : Type*} [Field α] [CharZero α]

-- Define the points and the circle
variables (A B C D M : α)

-- Assume the given cyclic quadrilateral properties and intersections
variable (h_cyclic : is_cyclic Quad ABCD)
variable (h_ratio : AB / BC = AD / DC)
variable (mid_AC : midpoint A C B)
variable (M_on_circle: on_circle M ABCD)
variable (M_diff_B: M ≠ B)
variable (line_through_B_mid_AC: line_through B mid_AC M)

-- Proof statement
theorem AM_eq_CD :
  AM = CD := 
by
  sorry

end AM_eq_CD_l333_333365


namespace daisies_bought_l333_333292

theorem daisies_bought (cost_per_flower roses total_cost : ℕ) 
  (h1 : cost_per_flower = 3) 
  (h2 : roses = 8) 
  (h3 : total_cost = 30) : 
  (total_cost - (roses * cost_per_flower)) / cost_per_flower = 2 :=
by
  sorry

end daisies_bought_l333_333292


namespace binomial_expansion_fifth_term_constant_l333_333832

open Classical -- Allows the use of classical logic

noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (x ^ (n - r) / (x ^ r * (2 ^ r / x ^ r)))

theorem binomial_expansion_fifth_term_constant (n : ℕ) :
  (binomial_term n 4 x = (x ^ (n - 3 * 4) * (-2) ^ 4)) → n = 12 := by
  intro h
  sorry

end binomial_expansion_fifth_term_constant_l333_333832


namespace integer_solution_abs_lt_sqrt2_l333_333364

theorem integer_solution_abs_lt_sqrt2 (x : ℤ) (h : |x| < Real.sqrt 2) : x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end integer_solution_abs_lt_sqrt2_l333_333364


namespace find_f_2013_l333_333004

-- Given assumptions
variables {R : Type*} [TopologicalSpace R]

def is_even_function (f : R → R) : Prop := ∀ x, f x = f (-x)

def satisfies_recurrence (f : R → R) : Prop := ∀ x, f (x + 4) = f x + 2 * f 2

-- Given conditions
variable (f : R → R)
variable (hf_even : is_even_function f)
variable (hf_recurrence : satisfies_recurrence f)
variable (hf_neg1 : f (-1) = 2)

-- Required proof
theorem find_f_2013 : f 2013 = 2 := 
by
  sorry

end find_f_2013_l333_333004


namespace number_of_students_playing_soccer_l333_333860

-- Definitions of the conditions
def total_students : ℕ := 500
def total_boys : ℕ := 350
def percent_boys_playing_soccer : ℚ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- To be proved
theorem number_of_students_playing_soccer :
  ∃ (S : ℕ), S = 250 ∧ 0.14 * (S : ℚ) = 35 :=
sorry

end number_of_students_playing_soccer_l333_333860


namespace problem_inequality_1_problem_inequality_2_l333_333881

theorem problem_inequality_1 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : 0 < α ∧ α < 1) : 
  (1 + x) ^ α ≤ 1 + α * x :=
sorry

theorem problem_inequality_2 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : α < 0 ∨ α > 1) : 
  (1 + x) ^ α ≥ 1 + α * x :=
sorry

end problem_inequality_1_problem_inequality_2_l333_333881


namespace circle_radius_l333_333761

theorem circle_radius (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y + 16 = 0 → sqrt ((4:ℝ)^2) = 2 :=
by
  sorry

end circle_radius_l333_333761


namespace function_translation_symmetry_l333_333941

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem function_translation_symmetry :
  (∀ x : ℝ, f(x - 1) = e^(-x)) → ∀ x : ℝ, f x = e^(-(x + 1)) :=
by
  intros h x
  sorry

end function_translation_symmetry_l333_333941


namespace largest_sum_is_8_over_15_l333_333726

theorem largest_sum_is_8_over_15 :
  let s1 := 1/5 + 1/6,
      s2 := 1/5 + 1/7,
      s3 := 1/5 + 1/3,
      s4 := 1/5 + 1/8,
      s5 := 1/5 + 1/9 in
  max s1 (max s2 (max s3 (max s4 s5))) = 8/15 :=
by
  let s1 := 1/5 + 1/6
  let s2 := 1/5 + 1/7
  let s3 := 1/5 + 1/3
  let s4 := 1/5 + 1/8
  let s5 := 1/5 + 1/9
  sorry

end largest_sum_is_8_over_15_l333_333726


namespace slope_of_tangent_is_less_than_one_tangent_lines_with_slope_zero_l333_333390

def curve (x : ℝ) : ℝ := x + 1/x

theorem slope_of_tangent_is_less_than_one (x : ℝ) (hx : x ≠ 0) : 
  derivative (λ x, curve x) x < 1 := sorry

theorem tangent_lines_with_slope_zero : 
  ∃ (x : ℝ), curve x = 2 ∧ x = 1 ∨ curve x = -2 ∧ x = -1 := sorry

end slope_of_tangent_is_less_than_one_tangent_lines_with_slope_zero_l333_333390


namespace good_coloring_l333_333928

noncomputable def smallest_k (n : ℕ) : ℕ :=
if n % 3 = 2 then n - 1 else n

theorem good_coloring (n : ℕ) (E : Finset ℕ) (h₁ : 3 ≤ n) (h₂ : E.card = 2 * n - 1) : 
  ∃ k, ∀ (C : Finset ℕ) (hC : C ⊆ E) (h3 : C.card = k), 
    ∃ b1 b2 ∈ C, (↑(b2 - b1) / (E.card) ) = n →
  k = smallest_k n :=
sorry

end good_coloring_l333_333928


namespace sqrt_factorial_squared_l333_333573

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333573


namespace hexagon_inequality_l333_333446

noncomputable def ABCDEF := 3 * Real.sqrt 3 / 2
noncomputable def ACE := Real.sqrt 3
noncomputable def BDF := Real.sqrt 3
noncomputable def R₁ := Real.sqrt 3 / 4
noncomputable def R₂ := -Real.sqrt 3 / 4

theorem hexagon_inequality :
  min ACE BDF + R₂ - R₁ ≤ 3 * Real.sqrt 3 / 4 :=
by
  sorry

end hexagon_inequality_l333_333446


namespace least_positive_number_of_linear_combination_of_24_20_l333_333075

-- Define the conditions as integers
def problem_statement (x y : ℤ) : Prop := 24 * x + 20 * y = 4

theorem least_positive_number_of_linear_combination_of_24_20 :
  ∃ (x y : ℤ), (24 * x + 20 * y = 4) := 
by
  sorry

end least_positive_number_of_linear_combination_of_24_20_l333_333075


namespace function_periodic_l333_333884

open Real

def periodic (f : ℝ → ℝ) := ∃ T > 0, ∀ x, f (x + T) = f x

theorem function_periodic (a : ℚ) (b d c : ℝ) (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, f (x + ↑a + b) - f (x + b) = c * (x + 2 * ↑a + ⌊x⌋ - 2 * ⌊x + ↑a⌋ - ⌊b⌋) + d) : 
    periodic f :=
sorry

end function_periodic_l333_333884


namespace sqrt_factorial_mul_square_l333_333639

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333639


namespace speed_of_man_l333_333706

-- Define all given conditions and constants

def trainLength : ℝ := 110 -- in meters
def trainSpeed : ℝ := 40 -- in km/hr
def timeToPass : ℝ := 8.799296056315494 -- in seconds

-- We want to prove that the speed of the man is approximately 4.9968 km/hr
theorem speed_of_man :
  let trainSpeedMS := trainSpeed * (1000 / 3600)
  let relativeSpeed := trainLength / timeToPass
  let manSpeedMS := relativeSpeed - trainSpeedMS
  let manSpeedKMH := manSpeedMS * (3600 / 1000)
  abs (manSpeedKMH - 4.9968) < 0.01 := sorry

end speed_of_man_l333_333706


namespace pet_store_inventory_l333_333297

def initial_inventory := (7, 6, 4, 5, 3, 2) -- (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

def morning_transactions := 
  (7 - 2, 6 - 1, 4 - 1, 5 - 1, 3 - 1 + 1, 2 - 1 + 1) -- Adjusted (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

def afternoon_transactions := 
  (5 - 1, 5 - 2, 3 - 1, 4 - 2, 3 + 1, 2 - 1) -- Further adjusted (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

def returns :=
  (4, 3 + 1, 2, 2, 4 + 1, 1 + 1) -- Further adjusted (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

def new_order :=
  (4 + 3, 3 + 2, 2 + 1, 2, 5, 2) -- Further adjusted (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

def evening_transactions :=
  (7 - 1, 5 - 1, 3 - 1, 2 - 1, 5, 1 + 1 - 1) -- Adjusted (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

def family_purchases := 
  (6 - 1, 4 - 1, 2 - 1, 1 - 1, 5 - 1, 1 - 1) -- Adjusted for family purchases (puppies, kittens, rabbits, guinea pigs, chameleons, parrots)

theorem pet_store_inventory :
  let final_inventory := (6 - 1, 4 - 1, 2 - 1, 1 - 1, 5 - 1, 1 - 1) in  -- Subtracted for exact family purchases
  (let (puppies, kittens, rabbits, guinea_pigs, chameleons, parrots) := final_inventory in 
    puppies + kittens + rabbits + guinea_pigs + chameleons + parrots) = 16 :=
by
  sorry

end pet_store_inventory_l333_333297


namespace spring_deformation_horizontal_l333_333552

-- Definitions and conditions
def x1 : ℝ := 8 / 100  -- converting cm to meters
def x2 : ℝ := 15 / 100 -- converting cm to meters
def m1 : ℝ := ... -- placeholder for mass m1 (in kg)
def m2 : ℝ := ... -- placeholder for mass m2 (in kg)
def k : ℝ := ... -- placeholder for spring constant (in N/m)
def g : ℝ := 9.81 -- gravitational acceleration in m/s^2

-- Theorem statement
theorem spring_deformation_horizontal (m1 m2 k : ℝ) : ¬¬(m1 > 0 ∧ m2 > 0 ∧ k > 0) →
  let x := (2 * x1) in x = 16 / 100 :=    -- converting cm to meters
by
  intros
  sorry

end spring_deformation_horizontal_l333_333552


namespace second_train_speed_is_correct_l333_333554

noncomputable def speed_second_train 
  (length_train1 length_train2 : ℝ) 
  (speed_train1 : ℝ) 
  (time_clearing : ℝ) : ℝ :=
  let total_distance := length_train1 + length_train2
  let time_hours := time_clearing / 3600
  let relative_speed := total_distance / time_hours / 1000
  relative_speed - speed_train1

theorem second_train_speed_is_correct 
  (length_train1 : ℝ := 111) 
  (length_train2 : ℝ := 165) 
  (speed_train1 : ℝ := 60) 
  (time_clearing : ℝ := 6.623470122390208) : 
  speed_second_train length_train1 length_train2 speed_train1 time_clearing ≈ 89.916 :=
by 
  sorry

end second_train_speed_is_correct_l333_333554


namespace correct_answer_l333_333711

-- Conditions in Lean 4
def impossible_event (e : Prop) := ¬ e
def certain_event (e : Prop) := e
def random_event (e : Prop) := ¬ certain_event e ∧ ¬ impossible_event e

-- Given events
def event1 := impossible_event (∃ t : Type, t = "Water turns into oil")
def event2 := random_event (∃ t : Type, t = "It will rain tomorrow")
def event3 := random_event (∃ t : Type, t = "Xiao Ming scores a 10 in shooting")
def event4 := certain_event (∃ t : Type, t = "Ice melts under normal temperature and pressure")
def event5 := certain_event (∃ t : Type, t = "January of year 13 has 31 days")

-- Definition to count non-random events
def non_random_event_count : Nat :=
  List.length (List.filter (λ e, e) [event1, ¬ event2, ¬ event3, event4, event5])

-- The theorem stating the correct answer 
theorem correct_answer : non_random_event_count = 3 :=
by 
  unfold non_random_event_count
  unfold event1 event2 event3 event4 event5
  simp
  exact sorry

end correct_answer_l333_333711


namespace sum_of_first_four_terms_of_geometric_sequence_l333_333386

noncomputable def geometric_sum_first_four (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : q > 0) 
  (h3 : a 2 = 1) 
  (h4 : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  geometric_sum_first_four a q = 15 / 2 :=
sorry

end sum_of_first_four_terms_of_geometric_sequence_l333_333386


namespace interval_of_decrease_for_f_max_min_values_for_f_on_interval_l333_333016

open Real

noncomputable def f (x : ℝ) : ℝ :=
  4 * sin x * cos (x + π / 6)

theorem interval_of_decrease_for_f :
  ∀ k : ℤ, ∀ x : ℝ, (k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) ↔ (∃ t : ℤ, x ∈ Icc (t * π + π / 6) (t * π + 2 * π / 3)) := sorry

theorem max_min_values_for_f_on_interval :
  ∃ max min : ℝ, max = 1 ∧ min = -2 ∧ (∀ x : ℝ, x ∈ Icc 0 (π / 2) → f x ≤ max ∧ f x ≥ min) := sorry

end interval_of_decrease_for_f_max_min_values_for_f_on_interval_l333_333016


namespace fold_creates_bisector_l333_333147

-- Define an angle α with its vertex located outside the drawing (hence inaccessible)
structure Angle :=
  (theta1 theta2 : ℝ) -- theta1 and theta2 are the measures of the two angle sides

-- Define the condition: there exists an angle on transparent paper
variable (a: Angle)

-- Prove that folding such that the sides of the angle coincide results in the crease formed being the bisector
theorem fold_creates_bisector (a: Angle) :
  ∃ crease, crease = (a.theta1 + a.theta2) / 2 := 
sorry

end fold_creates_bisector_l333_333147


namespace smallest_element_in_A_l333_333005

def is_imaginary_unit (i : ℂ) : Prop :=
  i = complex.I

def A (i : ℂ) : set ℕ := {n | n > 0 ∧ i^n = -1}

theorem smallest_element_in_A (i : ℂ) (h : is_imaginary_unit i) : ∃ n ∈ A i, ∀ m ∈ A i, n ≤ m :=
  sorry

end smallest_element_in_A_l333_333005


namespace first_player_wins_l333_333430

-- Define the chessboard and the initial setup of knights on opposite corners
structure Chessboard where
  size : ℕ
  initial_k1_pos : ℕ × ℕ
  initial_k2_pos : ℕ × ℕ

-- Define the game conditions
def opposite_corners (board : Chessboard) : Prop :=
  board.initial_k1_pos.1 ≠ board.initial_k2_pos.1 ∧
  board.initial_k1_pos.2 ≠ board.initial_k2_pos.2

def knight_moves (pos : ℕ × ℕ) : list (ℕ × ℕ) := 
  [(pos.1 + 2, pos.2 + 1), (pos.1 + 2, pos.2 - 1), (pos.1 - 2, pos.2 + 1), (pos.1 - 2, pos.2 - 1),
   (pos.1 + 1, pos.2 + 2), (pos.1 + 1, pos.2 - 2), (pos.1 - 1, pos.2 + 2), (pos.1 - 1, pos.2 - 2)]

noncomputable def player_can_win (board : Chessboard) : Prop :=
  ∃ strategy : list (ℕ × ℕ), ∀ k1_pos k2_pos, 
  (k1_pos ≠ k2_pos → (knight_moves k1_pos).any (λ pos, pos ≠ k2_pos)) → 
  (strategy.all (λ move, move ∈ knight_moves k1_pos ∧ move ≠ k2_pos))

-- Lean 4 statement to prove that the first player wins
theorem first_player_wins (board : Chessboard) (h : opposite_corners board) : player_can_win board :=
  sorry

end first_player_wins_l333_333430


namespace max_value_of_1_minus_sin_l333_333945

-- Define the maximum value of a function y = 1 - sin x
def max_value_y : ℝ :=
  2

theorem max_value_of_1_minus_sin (x : ℝ) :
  ∃ x, y = 1 - sin x ∧ ∀ x : ℝ, 1 - sin x ≤ 2 ∧ (∃ x, y = 2) :=
begin
  sorry
end

end max_value_of_1_minus_sin_l333_333945


namespace g_values_l333_333128

def g (x : ℝ) : ℝ :=
if x < 0 then 2 * x + 4
else if x < 5 then x^2 - 3 * x + 1
else 10 - x

theorem g_values :
  g 3 = 1 ∧ g 7 = 3 :=
by
  split
  { -- Proof for g(3)
    have h1 : 0 ≤ (3 : ℝ) := by norm_num,
    have h2 : (3 : ℝ) < 5 := by norm_num,
    simp [g, h1, h2] }
  { -- Proof for g(7)
    have h : (7 : ℝ) ≥ 5 := by norm_num,
    simp [g, h] }

end g_values_l333_333128


namespace min_socks_to_guarantee_15_pairs_l333_333422

theorem min_socks_to_guarantee_15_pairs :
  let red_socks := 120
  let green_socks := 100
  let blue_socks := 80
  let yellow_socks := 60
  let purple_socks := 40
  let total_pairs_needed := 15
  (* total pairs needed to guarantee 15 pairs *)
  ∀ (selected_socks : ℕ), selected_socks >= 35 → 
  ∃ (r g b y p : ℕ), (r + g + b + y + p = selected_socks
                      ∧ r + g + b + y + p >= 2 * total_pairs_needed
                      ∧ r ≤ red_socks ∧ g ≤ green_socks 
                      ∧ b ≤ blue_socks ∧ y ≤ yellow_socks 
                      ∧ p ≤ purple_socks) :=
sorry

#check min_socks_to_guarantee_15_pairs

end min_socks_to_guarantee_15_pairs_l333_333422


namespace min_calls_required_l333_333321

-- Define the set of people involved in the communication
inductive Person
| A | B | C | D | E | F

-- Function to calculate the minimum number of calls for everyone to know all pieces of gossip
def minCalls : ℕ :=
  9

-- Theorem stating the minimum number of calls required
theorem min_calls_required : minCalls = 9 := by
  sorry

end min_calls_required_l333_333321


namespace triangle_angle_B_l333_333848

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end triangle_angle_B_l333_333848


namespace compute_sqrt_factorial_square_l333_333625

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333625


namespace prob_correct_l333_333280

noncomputable def prob_train_there_when_sam_arrives : ℚ :=
  let total_area := (60 : ℚ) * 60
  let triangle_area := (1 / 2 : ℚ) * 15 * 15
  let parallelogram_area := (30 : ℚ) * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem prob_correct : prob_train_there_when_sam_arrives = 25 / 160 :=
  sorry

end prob_correct_l333_333280


namespace power_sum_ge_three_l333_333127

theorem power_sum_ge_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a ^ a + b ^ b + c ^ c ≥ 3 :=
by
  sorry

end power_sum_ge_three_l333_333127


namespace problem_satisfying_value_l333_333284

theorem problem_satisfying_value (b c x : ℝ) (h : c = 0) : x = b → x^3 + c^2 = (b - x)^2 :=
by
  assume hb : x = b
  rw [h, hb]
  sorry

end problem_satisfying_value_l333_333284


namespace base8_add_sub_l333_333565

-- Definition to convert numbers from base 8 to decimal
def from_base8 (n : ℕ) : ℕ :=
  let d0 := n % 10 in
  let d1 := (n / 10) % 10 in
  let d2 := (n / 100) % 10 in
  d0 * 8^0 + d1 * 8^1 + d2 * 8^2

-- Specific numbers in base 8
def number1 := from_base8 176
def number2 := from_base8 45
def number3 := from_base8 63
def result := from_base8 151

theorem base8_add_sub:
  (number1 + number2 - number3) = result :=
by
  -- skipping the proof
  sorry

end base8_add_sub_l333_333565


namespace sqrt_factorial_sq_l333_333590

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333590


namespace triangle_incircle_relationship_l333_333877

theorem triangle_incircle_relationship (A B C O' O D : Type*)
  [incircle ABC O]
  [circumcircle ABC O']
  (hD : extends (line_segment AO) to (cut_sphere O' D)) :
  CD = OD ∧ OD = BD :=
by
  sorry

end triangle_incircle_relationship_l333_333877


namespace derivative_of_even_function_is_odd_l333_333907

theorem derivative_of_even_function_is_odd
  (f : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f (-x) = f x) 
  (g : ℝ → ℝ) 
  (hg : ∀ x : ℝ, g x = deriv f x) 
  : ∀ x : ℝ, g (-x) = - g x :=
begin
  sorry
end

end derivative_of_even_function_is_odd_l333_333907


namespace find_b_plus_d_l333_333282

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 3⟩
def B : Point := ⟨-12, -15⟩
def C : Point := ⟨6, -5⟩

noncomputable def bisector_of_angle_A (A B C : Point) : ℝ × ℝ :=
sorry -- This function should determine the coefficients b and d for the bisector equation

theorem find_b_plus_d (A B C : Point) : 
  b + d = some_value := 
by
  let (b, d) := bisector_of_angle_A A B C
  exact sorry

end find_b_plus_d_l333_333282


namespace find_number_l333_333824

theorem find_number (X : ℝ) (h : 30 = 0.50 * X + 10) : X = 40 :=
by
  sorry

end find_number_l333_333824


namespace arithmetic_sequence_property_l333_333699

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n - a (n - 1)

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, 1 / (Real.sqrt (a i) + Real.sqrt (a (i+1)))) = n / (Real.sqrt (a 0) + Real.sqrt (a n))) 
  (h2 : ∀ n : ℕ, a n > 0)
  : is_arithmetic_sequence a :=
sorry

end arithmetic_sequence_property_l333_333699


namespace increased_contact_area_effect_l333_333349

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l333_333349


namespace sum_of_digits_of_d_l333_333476

-- Given conditions
def exchange_rate := 8 / 5
def spent_euros := 80

-- Prove that the sum of the digits of d' is 7
theorem sum_of_digits_of_d' (d' : ℚ) :
  ((exchange_rate * d' - spent_euros = d') → (d'.ceil.digits.sum = 7)) :=
by
  sorry

end sum_of_digits_of_d_l333_333476


namespace subcommittee_ways_l333_333686

theorem subcommittee_ways :
  let R := 10  -- Number of Republicans
  let D := 7   -- Number of Democrats
  let r := 4   -- Republicans to choose
  let d := 3   -- Democrats to choose
  (Nat.choose R r) * (Nat.choose D d) = 7350 := 
by
  let R := 10
  let D := 7
  let r := 4
  let d := 3
  show (Nat.choose R r) * (Nat.choose D d) = 7350 from sorry

end subcommittee_ways_l333_333686


namespace final_price_of_bedroom_set_l333_333152

def original_price : ℕ := 2000
def gift_card_amount : ℕ := 200
def store_discount : ℝ := 0.15
def credit_card_discount : ℝ := 0.10

theorem final_price_of_bedroom_set :
  let store_discount_amount := original_price * store_discount
  let after_store_discount := original_price - store_discount_amount.to_nat
  let credit_card_discount_amount := after_store_discount * credit_card_discount
  let after_credit_card_discount := after_store_discount - credit_card_discount_amount.to_nat
  let final_price := after_credit_card_discount - gift_card_amount
  final_price = 1330 :=
by
  -- Here we have the placeholder 'sorry' to indicate the proof step.
  sorry

end final_price_of_bedroom_set_l333_333152


namespace problem1_problem2_l333_333451

noncomputable def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

/- Prove that the general formula for a_n is n given the conditions -/
theorem problem1 (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_eq : ∀ n, (a n) ^ 2 + a n = 2 * S_n a n) :
  ∀ n, a n = n := 
sorry

noncomputable def b_n (n : ℕ) : ℝ := n / (2 ^ (n - 1))

noncomputable def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n (i + 1)

/- Prove that the sum of the first n terms of b_n, T_n, is 4 - (n + 2) / 2^(n - 1) -/
theorem problem2 (n : ℕ) :
  T_n n = 4 - (n + 2) / (2 ^ (n - 1)) :=
sorry

end problem1_problem2_l333_333451


namespace portion_left_l333_333443

theorem portion_left (john_portion emma_portion final_portion : ℝ) (H1 : john_portion = 0.6) (H2 : emma_portion = 0.5 * (1 - john_portion)) :
  final_portion = 1 - john_portion - emma_portion :=
by
  sorry

end portion_left_l333_333443


namespace perp_vec_m_l333_333027

theorem perp_vec_m (m : ℝ) : (1 : ℝ) * (-1 : ℝ) + 2 * m = 0 → m = 1 / 2 :=
by 
  intro h
  -- Translate the given condition directly
  sorry

end perp_vec_m_l333_333027


namespace minimize_tangent_y_intercept_l333_333777

noncomputable theory

def circle_eq (x y : ℝ) := x^2 + (y - 1)^2 = 1

def is_tangent (line : ℝ → ℝ → Prop) :=
  ∃ A B : ℝ × ℝ, (A.1 > 0 ∧ A.2 = 0) ∧ (B.1 = 0 ∧ B.2 > 0) ∧
  line A.1 A.2 ∧ line B.1 B.2

def tangent_line_eq (a b x y : ℝ) := b * x + a * y - a * b = 0

def minimized_AB_y_intercept (b : ℝ) := b = (3 + Real.sqrt 5) / 2

theorem minimize_tangent_y_intercept :
  (∃ l : ℝ → ℝ → Prop, is_tangent l ∧ 
    ∀ x y, l x y ↔ tangent_line_eq (Real.sqrt ((3 + Real.sqrt 5) / (Real.sqrt 5) + 2)) (3 + Real.sqrt 5) / 2 x y) →
  ∃ b, minimized_AB_y_intercept b :=
begin
  sorry
end

end minimize_tangent_y_intercept_l333_333777


namespace find_a_l333_333073

-- Definitions and conditions
def pointA (a : ℝ) : ℝ × ℝ := (a, -1)
def pointB : ℝ × ℝ := (2, 3)

def slope (A B : ℝ × ℝ) : ℝ :=
  (B.snd - A.snd) / (B.fst - A.fst)

theorem find_a (a : ℝ) (h : slope (pointA a) pointB = 2) : a = 0 :=
by
  sorry

end find_a_l333_333073


namespace sequence_contains_irrational_l333_333873

theorem sequence_contains_irrational (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_recurrence : ∀ n, a (n + 1)^2 = a n + 1) : ∃ n, ¬ is_rat (a n) :=
sorry

end sequence_contains_irrational_l333_333873


namespace solve_abs_eq_l333_333977

theorem solve_abs_eq (x a : ℝ) (h1 : f x = abs (3 * x - 2)) (h2 : ¬ x = 3) (h3 : ¬ x = 0) : 
  (|3 * x - 2| = |x + a|) ↔ (a = -2/3 ∨ a = 2) :=
sorry

end solve_abs_eq_l333_333977


namespace isosceles_triangle_angle_AMF_l333_333374

theorem isosceles_triangle_angle_AMF
  (A B C M N F : Point)
  (h_iso : is_isosceles ABC AB BC)
  (h_M_N_on_BC : lies_on_segment M B C ∧ lies_on_segment N B C)
  (h_M_between_B_N : lies_on_segment M B N)
  (h_AN_MN : dist A N = dist M N)
  (h_angles : ∠BAM = ∠NAC)
  (h_MF_to_AC : dist_to_line M AC = MF)
  : ∠AMF = 30° := 
sorry

end isosceles_triangle_angle_AMF_l333_333374


namespace vasya_read_entire_book_l333_333215

theorem vasya_read_entire_book :
  let day1 := 1 / 2
  let day2 := 1 / 3 * (1 - day1)
  let days12 := day1 + day2
  let day3 := 1 / 2 * days12
  (days12 + day3) = 1 :=
by
  sorry

end vasya_read_entire_book_l333_333215


namespace sqrt_factorial_product_squared_l333_333575

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333575


namespace tan_div_formula_22_5_l333_333223

theorem tan_div_formula_22_5 :
  (tan (Real.pi / 8)) / (1 - (tan (Real.pi / 8))^2) = 1/2 := sorry

end tan_div_formula_22_5_l333_333223


namespace number_of_members_l333_333674

-- Define the setup and conditions
variable (n : ℕ)

-- Define the probability condition
def probability_condition (n : ℕ) : Prop :=
  (2 / (n - 1 : ℕ)) = 0.2

-- State the main theorem
theorem number_of_members (h : probability_condition n) : n = 11 := by
  sorry

end number_of_members_l333_333674


namespace right_triangle_determination_l333_333863

theorem right_triangle_determination (A B C : ℕ) (a b c : ℕ) :
  (angle_A_eq_angle_C_sub_angle_B : A = C - B) ∨
  (a_squared_eq_b_squared_sub_c_squared : a^2 = b^2 - c^2) ∨
  (a_eq_3_b_eq_5_c_eq_4 : a = 3 ∧ b = 5 ∧ c = 4) ∨
  (a_b_c_ratio : a / gcd (gcd a b) c = 2 / 3 ∧ b / gcd (gcd a b) c = 3 / 4 ∧ c / gcd (gcd a b) c = 4) → 
  (¬ right_triangle (a b c)) :=
sorry

end right_triangle_determination_l333_333863


namespace west_movement_representation_l333_333812

theorem west_movement_representation :
  (east : ℤ) → (direction : ℕ) → (x : ℤ) →
  east = 50 → 
  direction = 60 →
  (west : ℤ)
  (h : west = -direction) → x = -west :=
by
  intros east direction x heast hdirection west hwest
  rw [heast, hdirection, hwest]
  sorry

end west_movement_representation_l333_333812


namespace value_of_square_of_sum_l333_333411

theorem value_of_square_of_sum (x y: ℝ) 
(h1: 2 * x * (x + y) = 58) 
(h2: 3 * y * (x + y) = 111):
  (x + y)^2 = (169/5)^2 := by
  sorry

end value_of_square_of_sum_l333_333411


namespace ninety_eight_times_ninety_eight_l333_333324

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 := 
by
  sorry

end ninety_eight_times_ninety_eight_l333_333324


namespace fraction_S9_T9_equals_2_l333_333450

-- Definitions for the arithmetic sequences and their sums
variable {α : Type*} [linear_ordered_field α]
def sum_arith_seq (a : ℕ → α) (n : ℕ) : α := (n * (a 1 + a n)) / 2

-- Conditions
axiom a_sequence (a : ℕ → α)
axiom b_sequence (b : ℕ → α)
axiom a5_equals_2b5 : a_sequence 5 = 2 * b_sequence 5

-- Problem statement
theorem fraction_S9_T9_equals_2 :
  (sum_arith_seq a_sequence 9) / (sum_arith_seq b_sequence 9) = 2 := 
sorry

end fraction_S9_T9_equals_2_l333_333450


namespace matthew_hotdogs_l333_333137

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end matthew_hotdogs_l333_333137


namespace explanatory_variable_is_fertilizer_l333_333279

theorem explanatory_variable_is_fertilizer 
  (predict_yield_based_on_fertilizer : Prop) 
  (explanatory_variable : Prop) :
  predict_yield_based_on_fertilizer → explanatory_variable = "amount_of_fertilizer" :=
by
  intro h
  -- Here you would complete the proof.
  sorry

end explanatory_variable_is_fertilizer_l333_333279


namespace solution_set_for_inequality_l333_333125

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

theorem solution_set_for_inequality
  (h_deriv : ∀ x ∈ set.Iio 0, deriv f x = f' x)
  (h_ineq : ∀ x ∈ set.Iio 0, 2 * f x + x * f' x > x^2) :
  ∀ x < -2018, (x + 2017) ^ 2 * f (x + 2017) - f (-1) > 0 := sorry

end solution_set_for_inequality_l333_333125


namespace find_prices_min_volleyballs_l333_333210

namespace SchoolPurchasing

-- Definitions based on the conditions given in the problem.
variable (a b : ℕ)

-- Condition 1: 3a + 2b = 520
axiom cond1 : 3 * a + 2 * b = 520

-- Condition 2: 2a + 5b = 640
axiom cond2 : 2 * a + 5 * b = 640

-- Statement to prove: 
-- Given cond1 and cond2, a = 120 and b = 80
theorem find_prices (a b : ℕ) (cond1 : 3 * a + 2 * b = 520) (cond2 : 2 * a + 5 * b = 640) : 
  a = 120 ∧ b = 80 := 
sorry

-- Given that the total purchasing funds do not exceed 5500 yuan for 50 balls, 
-- prove that at least 13 volleyballs can be purchased.
-- We do not need to check the computations of a and b here; we just use their values
variable (x : ℕ)  -- x represents the number of basketballs

theorem min_volleyballs (x : ℕ) (cond1 : 3 * a + 2 * b = 520) (cond2 : 2 * a + 5 * b = 640)
  (price_a : a = 120) (price_b : b = 80) (total_balls : x + (50 - x)) (cost_limit : 120 * x + 80 * (50 - x) ≤ 5500) : 
  (50 - x) ≥ 13 :=
sorry

end SchoolPurchasing

end find_prices_min_volleyballs_l333_333210


namespace collection_of_propositions_l333_333025

variable {α : Type*}
variables (A B : set α)

theorem collection_of_propositions (A B : set α) :
  ((A ∪ B ≠ B) → (A ∩ B ≠ A)) ∧
  ((A ∪ B = B) → (A ∩ B = A)) ∧
  ((A ∩ B ≠ A) → (A ∪ B ≠ B)) :=
by
  sorry

end collection_of_propositions_l333_333025


namespace clara_plays_fewer_songs_per_day_l333_333558

theorem clara_plays_fewer_songs_per_day :
  ∀ (v d T T_v T_c c : ℕ),
     v = 10 →
     d = 22 →
     T = 396 →
     T_v = v * d →
     T_c = T - T_v →
     c = T_c / d →
     v - c = 2 :=
by {
  intros v d T T_v T_c c,
  intros hv hd hT hTv hTc hc,
  rw [hv, hd, hT, hTv, hTc, hc],
  sorry
}

end clara_plays_fewer_songs_per_day_l333_333558


namespace sqrt_factorial_product_squared_l333_333629

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333629


namespace number_of_square_free_odd_integers_between_1_and_200_l333_333050

def count_square_free_odd_integers (a b : ℕ) (squares : List ℕ) : ℕ :=
  (b - (a + 1)) / 2 + 1 - List.foldl (λ acc sq => acc + ((b - 1) / sq).div 2 + 1) 0 squares

theorem number_of_square_free_odd_integers_between_1_and_200 :
  count_square_free_odd_integers 1 200 [9, 25, 49, 81, 121] = 81 :=
by
  apply sorry

end number_of_square_free_odd_integers_between_1_and_200_l333_333050


namespace circle_equation_l333_333776

theorem circle_equation 
  (center_symmetric : ∃ (M : ℝ × ℝ), M = (1, 1) ∧ 
    ∃ (C : ℝ × ℝ), C = (0, 2) ∧ symmetric_about_line C M (x - y + 1 = 0))
  (tangent_to_asymptotes : ∀ (h : ℝ × ℝ), h = (x^2 / 3 - y^2 = 1) 
    → tangent_to_circle C (x / sqrt 3 ± y = 0)) :
  ∃ (circle : ℝ), circle = (x^2 + (y-2)^2 = 3) :=
by
  sorry

end circle_equation_l333_333776


namespace faster_train_overtakes_after_time_to_cross_opposite_direction_l333_333214

def train_length1 : ℝ := 140
def train_length2 : ℝ := 180
def time_to_cross1 : ℝ := 16
def time_to_cross2 : ℝ := 20

def speed1 : ℝ := train_length1 / time_to_cross1
def speed2 : ℝ := train_length2 / time_to_cross2

def relative_speed_same_direction : ℝ := speed2 - speed1
def relative_speed_opposite_direction : ℝ := speed1 + speed2

def distance_to_overtake : ℝ := train_length1 + train_length2
def total_length : ℝ := train_length1 + train_length2

theorem faster_train_overtakes_after (t_same_direction : ℝ) : 
  t_same_direction = distance_to_overtake / relative_speed_same_direction := 
sorry

theorem time_to_cross_opposite_direction (t_opposite_direction : ℝ) : 
  t_opposite_direction = total_length / relative_speed_opposite_direction :=
sorry

end faster_train_overtakes_after_time_to_cross_opposite_direction_l333_333214


namespace percentage_of_b_over_a_eq_61_8_l333_333235

variable {R : Type*} [LinearOrderedCommRing R]

theorem percentage_of_b_over_a_eq_61_8
  (a b : R)
  (h : b / a = a / (a + b)) :
  b / a = 0.618 := by
  sorry

end percentage_of_b_over_a_eq_61_8_l333_333235


namespace angle_ABG_in_regular_octagon_l333_333484

theorem angle_ABG_in_regular_octagon (N : ℕ) (hN : N = 8) (regular_octagon : RegularPolygon N) : 
  angle ABG = 22.5 :=
by
  sorry

end angle_ABG_in_regular_octagon_l333_333484


namespace circle_radius_l333_333758

theorem circle_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ (∀ x y, x^2 - 8*x + y^2 - 4*y + 16 = 0 → r = 2)) :=
sorry

end circle_radius_l333_333758


namespace passing_marks_required_l333_333252

theorem passing_marks_required (T : ℝ)
  (h1 : 0.30 * T + 60 = 0.40 * T)
  (h2 : 0.40 * T = passing_mark)
  (h3 : 0.50 * T - 40 = passing_mark) :
  passing_mark = 240 := by
  sorry

end passing_marks_required_l333_333252


namespace sqrt_factorial_sq_l333_333587

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333587


namespace number_of_pairs_count_number_of_pairs_l333_333313

theorem number_of_pairs (a b : ℝ) :
  (∃ x y : ℤ, a * x + b * y = 2 ∧ x^2 + y^2 = 65) ↔
  (a, b) ∈ { (a, b) | -- set of all pairs (a, b) such that there are integer solutions (x, y)
    let solutions := { (x, y) | (x : ℤ) * x + (y : ℤ) * y = 65 } in
    (∃ x y : ℤ, (a * x + b * y = 2) ∧ ((x, y) ∈ solutions)) } := by
  sorry

theorem count_number_of_pairs :
  (card {(a, b) | ∃ x y : ℤ, a * x + b * y = 2 ∧ x^2 + y^2 = 65}) = 128 := by
  sorry

end number_of_pairs_count_number_of_pairs_l333_333313


namespace solution_l333_333537

def visitors_not_enjoy_nor_understand (V E U : ℕ) (h1 : E = U) (h2 : E = (3/4) * V) (h3 : V = 560) : ℕ :=
  V - E

theorem solution :
  let V := 560
  let E := (3 / 4 : ℝ) * V
  let U := E
  visitors_not_enjoy_nor_understand V (E.toNat) (U.toNat) (by rw [← E, ← U]) (by norm_num1) (by norm_num1) = 140 :=
begin
  sorry
end

end solution_l333_333537


namespace min_value_ratio_l333_333782

variable {α : Type*} [LinearOrderedField α]

theorem min_value_ratio (a : ℕ → α) (h1 : a 7 = a 6 + 2 * a 5) (h2 : ∃ m n : ℕ, a m * a n = 8 * a 1^2) :
  ∃ m n : ℕ, (1 / m + 4 / n = 11 / 6) :=
by
  sorry

end min_value_ratio_l333_333782


namespace system_equations_sum_14_l333_333925

theorem system_equations_sum_14 (a b c d : ℝ) 
  (h1 : a + c = 4) 
  (h2 : a * d + b * c = 5) 
  (h3 : a * c + b + d = 8) 
  (h4 : b * d = 1) :
  a + b + c + d = 7 ∨ a + b + c + d = 7 → (a + b + c + d) * 2 = 14 := 
by {
  sorry
}

end system_equations_sum_14_l333_333925


namespace perpendicular_line_parallel_line_implies_perpendicular_plane_l333_333113

variables {α : Type*} [Plane α] {a b : Line α}

-- Defining the conditions
def line_parallel_plane (a : Line α) (α : Plane α) : Prop := ∀ {p : Point α}, p ∈ a → p ∈ α
def line_perpendicular_plane (a : Line α) (α : Plane α) : Prop := ∀ {p q : Point α}, p ∈ a → q ∈ α → p ⟂ q
def line_parallel_line (a b : Line α) : Prop := ∀ {p q : Point α}, p ∈ a → q ∈ b → p ∥ q

-- Stating the proof problem
theorem perpendicular_line_parallel_line_implies_perpendicular_plane
  (h1 : line_perpendicular_plane a α)
  (h2 : line_parallel_line a b) : line_perpendicular_plane b α :=
sorry

end perpendicular_line_parallel_line_implies_perpendicular_plane_l333_333113


namespace problem_statement_l333_333885

-- Define the primary function f(x)
def f (x : ℝ) (h : x ≠ 0) : ℝ := (x^2 + 1) / (2 * x)

-- Define the initial function f_0(x)
def f_0 (x : ℝ) : ℝ := x

-- Define the recursive function f_{n+1}(x)
def f_rec (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0   => f_0 x
  | (k+1) => f (f_rec k x) (by {dsimp [f_rec], exact h})

-- The main theorem to prove
theorem problem_statement
  (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) (h3 : x ≠ 1)
  (n : ℕ) :
  let y := (x + 1) / (x - 1) in
  let N := 2^n in
  let y_exp := y ^ N in
  f_rec n x / f_rec (n + 1) x = 1 + 1 / f y_exp (by simp [h2]) :=
sorry

end problem_statement_l333_333885


namespace angle_ACE_is_40_degrees_l333_333950

/--
Given a pentagon ABCDE inscribed around a circle,
with the angles at vertices A, C, and E being 100 degrees each,
prove that the angle ∠ACE is 40 degrees.
-/
theorem angle_ACE_is_40_degrees
  (A B C D E O : Point)
  (h_circumscribed : inscribed_around_circle O A B C D E)
  (h_angles : angle_at_vertex A = 100 ∧ angle_at_vertex C = 100 ∧ angle_at_vertex E = 100) :
  angle_ACE = 40 :=
sorry

end angle_ACE_is_40_degrees_l333_333950


namespace parabola_equation_l333_333796

-- Define the conditions and given facts
variables {C : Type} [parabola : Parabola C]
variables (A B : Point)
variables (P : Point) (vertex : Point)
variables [Midpoint P A B]
variables [LineIntersection (y = x) C A B]
variables [VertexAtOrigin C]
variables [FocusOnXAxis C]

-- The mathematically equivalent proof problem statement
theorem parabola_equation :
  P = Point.mk 2 2 →
  vertex = Point.mk 0 0 →
  (∃ p : ℝ, equation C = (λ x y, y^2 = 2 * p * x)) →
  equation C = (λ x y, y^2 = 4 * x) :=
by
  -- Proof omitted
  sorry

end parabola_equation_l333_333796


namespace possible_triple_roots_l333_333693

theorem possible_triple_roots (b4 b3 b2 b1 r : ℤ) :
  (∀ (x : ℤ), (x - r)^3 ∣ x^5 + b4 * x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 24) →
  r ∈ {-2, -1, 1, 2} :=
sorry

end possible_triple_roots_l333_333693


namespace min_value_expression_ge_072_l333_333891

theorem min_value_expression_ge_072 (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 0.5) 
  (hy : |y| ≤ 0.5) 
  (hz : 0 ≤ z ∧ z < 1) :
  ((1 / ((1 - x) * (1 - y) * (1 - z))) - (1 / ((2 + x) * (2 + y) * (2 + z)))) ≥ 0.72 := sorry

end min_value_expression_ge_072_l333_333891


namespace distance_squared_between_centers_of_circles_l333_333845

theorem distance_squared_between_centers_of_circles 
  (a b : ℝ) (h₁ : a = 3) (h₂ : b = 4) : 
  let c := Real.sqrt (a^2 + b^2),
      s := (a + b + c) / 2,
      r₁ := 2 * a * b * (a + b - c) / (4 * s),
      r₂ := c / 2,
      O₁O₂_sq := r₂^2 - r₁^2 + (a^2 / (a + b - c)^2) in
    O₁O₂_sq = 1.25 := 
by 
  let a := a,
  let b := b,
  have hyp : c = 5 := by simp [Real.sqrt_eq_rpow, *],
  have s_eq : s = 6 := by simp [s],
  have r₁_eq : r₁ = 1 := by simp [r₁, *],
  have r₂_eq : r₂ = 2.5 := by simp [r₂, *],
  have O₁O₂_sq_eq : O₁O₂_sq = 1.25 := by simp [O₁O₂_sq_eq, *],
  exact O₁O₂_sq_eq

end distance_squared_between_centers_of_circles_l333_333845


namespace factorial_expression_value_l333_333596

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333596


namespace smallest_number_l333_333206

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end smallest_number_l333_333206


namespace mass_of_man_l333_333233

-- Definitions of the given conditions
def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def sink_depth : Float := 0.01 -- 1 cm converted to meters
def water_density : Float := 1000.0 -- Density of water in kg/m³

-- Define the proof goal as the mass of the man
theorem mass_of_man : Float :=
by
  let volume_displaced := boat_length * boat_breadth * sink_depth
  let weight_displaced := volume_displaced * water_density
  exact weight_displaced

end mass_of_man_l333_333233


namespace hyperbola_eccentricity_l333_333382

-- Define the conditions of the hyperbola and points
variables {a b c m d : ℝ} (ha : a > 0) (hb : b > 0) 
noncomputable def F1 : ℝ := sorry -- Placeholder for focus F1
noncomputable def F2 : ℝ := sorry -- Placeholder for focus F2
noncomputable def P : ℝ := sorry  -- Placeholder for point P

-- Define the sides of the triangle in terms of an arithmetic progression
def PF2 (m d : ℝ) : ℝ := m - d
def PF1 (m : ℝ) : ℝ := m
def F1F2 (m d : ℝ) : ℝ := m + d

-- Prove that the eccentricity is 5 given the conditions
theorem hyperbola_eccentricity 
  (m d : ℝ) (hc : c = (5 / 2) * d )  
  (h1 : PF1 m = 2 * a)
  (h2 : F1F2 m d = 2 * c)
  (h3 : (PF2 m d)^2 + (PF1 m)^2 = (F1F2 m d)^2 ) :
  (c / a) = 5 := 
sorry

end hyperbola_eccentricity_l333_333382


namespace sqrt_factorial_product_squared_l333_333637

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333637


namespace sum_first_100_terms_periodic_sequence_l333_333436

noncomputable def periodic_sequence (seq : ℕ → ℚ) (T : ℕ) :=
  ∀ n : ℕ, seq (n + T) = seq n

noncomputable def x_sequence (n : ℕ) : ℕ → ℚ :=
  match n with
  | 1 => 1
  | 2 => 0
  | k + 2 => abs (x_sequence k 1 - x_sequence k 0)

theorem sum_first_100_terms_periodic_sequence (a : ℚ)
    (h_periodic : periodic_sequence x_sequence 3)
    (h_x1 : x_sequence 1 = 1)
    (h_x2 : x_sequence 2 = a)
    (h_a_le_1 : a ≤ 1)
  :
    (∑ i in finset.range 100, x_sequence i) = 67 := sorry

end sum_first_100_terms_periodic_sequence_l333_333436


namespace line_through_point_with_equal_intercepts_l333_333277

theorem line_through_point_with_equal_intercepts (x y : ℝ) :
  (∃ b : ℝ, 3 * x + y = 0) ∨ (∃ b : ℝ, x - y + 4 = 0) ∨ (∃ b : ℝ, x + y - 2 = 0) :=
  sorry

end line_through_point_with_equal_intercepts_l333_333277


namespace students_in_band_l333_333421

def total_students : ℕ := 50
def students_chorus : ℕ := 18
def both_chorus_band : ℕ := 2
def neither_chorus_band : ℕ := 8

theorem students_in_band : 
  let only_chorus := students_chorus - both_chorus_band in
  let at_least_one := total_students - neither_chorus_band in
  let only_one_activity := at_least_one - both_chorus_band in
  let only_band := only_one_activity - only_chorus in
  let total_band := only_band + both_chorus_band in
  total_band = 26 :=
by
  sorry

end students_in_band_l333_333421


namespace expected_plain_zongzi_l333_333322

theorem expected_plain_zongzi :
  let total_zongzi := 10
  let red_bean_zongzi := 3
  let meat_zongzi := 3
  let plain_zongzi := 4
  let selected_zongzi := 3
  -- Let X be a random variable representing the number of plain zongzi taken.
  -- We need to prove that the expected value of X is 6/5
  ∃ X : ℕ → ℕ, -- Define a random variable X
  (X = λ n, if n = 0 then (plain_zongzi * (plain_zongzi - 1) * (plain_zongzi - 2)) / (total_zongzi * (total_zongzi - 1) * (total_zongzi - 2)) else
           if n = 1 then (plain_zongzi * (plain_zongzi - 1) * selected_zongzi) / (total_zongzi * (total_zongzi - 1)) else 
           if n = 2 then (plain_zongzi * selected_zongzi * selected_zongzi) / (total_zongzi * (total_zongzi - 1)) else 
           if n = 3 then (plain_zongzi * plain_zongzi * plain_zongzi) / (total_zongzi * selected_zongzi) else 0)
  -- Expected value calculation
  ∑ i in {0, 1, 2, 3}, i * X i = 6 / 5 :=
sorry

end expected_plain_zongzi_l333_333322


namespace circles_exist_with_line_intersection_points_l333_333923

noncomputable def circle1 (x y : ℝ) := x^2 + y^2 - 2 * x - 14 * y + 25 = 0
noncomputable def circle2 (x y : ℝ) := x^2 + y^2 - 34 * x + 210 * y - 711 = 0

theorem circles_exist_with_line_intersection_points :
  (∃ A B : ℝ × ℝ, A = (-2, 3) ∧ B = (5, 4)) →
  (∃ e : ℝ → ℝ, ∀ x, e x = (1 / 2) * x + 9) →
  (∃ d : ℝ, d = 4 * real.sqrt 5) →
  (∃ x1 x2 y1 y2 : ℝ, 
    circle1 x1 y1 ∧ circle1 x2 y2 ∧
    y1 = (1 / 2) * x1 + 9 ∧ y2 = (1 / 2) * x2 + 9 ∧
    abs (x2 - x1) = 8) ∧
  (∃ x1' x2' y1' y2' : ℝ, 
    circle2 x1' y1' ∧ circle2 x2' y2' ∧
    y1' = (1 / 2) * x1' + 9 ∧ y2' = (1 / 2) * x2' + 9 ∧
    abs (x2' - x1') = 8) := by
  sorry

end circles_exist_with_line_intersection_points_l333_333923


namespace cows_with_no_spot_l333_333905

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end cows_with_no_spot_l333_333905


namespace minimum_clients_visiting_l333_333275

theorem minimum_clients_visiting (C : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ 18 → 4 ≤ price i → price i ≤ 25)
  (h2 : selections = 3 * C)
  (h3 : ∀ i, 1 ≤ i → i ≤ 18 → selected i ≥ 1)
  (h4 : ∃ k, selected k = 6 ∧ k ≤ 18) :
  8 ≤ C :=
by
  sorry

end minimum_clients_visiting_l333_333275


namespace total_legs_among_tables_l333_333682

noncomputable def total_legs (total_tables four_legged_tables: ℕ) : ℕ :=
  let three_legged_tables := total_tables - four_legged_tables
  4 * four_legged_tables + 3 * three_legged_tables

theorem total_legs_among_tables : total_legs 36 16 = 124 := by
  sorry

end total_legs_among_tables_l333_333682


namespace inequality_solution_l333_333329

noncomputable def g (x : ℝ) : ℝ := (3 * x - 4) * (x - 5) / (2 * x)

theorem inequality_solution :
  {x : ℝ | g x ≥ 0} = set.Union (set.Ioo (-∞) 0) (set.Ioo 0 (4 / 3)) ∪ set.Ici (5) :=
by
  sorry

end inequality_solution_l333_333329


namespace orchids_cut_l333_333544

-- Define initial and final number of orchids in the vase
def initialOrchids : ℕ := 2
def finalOrchids : ℕ := 21

-- Formulate the claim to prove the number of orchids Jessica cut
theorem orchids_cut : finalOrchids - initialOrchids = 19 := by
  sorry

end orchids_cut_l333_333544


namespace sally_took_home_pens_l333_333159

theorem sally_took_home_pens
    (initial_pens : ℕ)
    (students : ℕ)
    (pens_per_student : ℕ)
    (locker_fraction : ℕ)
    (total_pens_given : ℕ)
    (remainder : ℕ)
    (locker_pens : ℕ)
    (home_pens : ℕ) :
    initial_pens = 5230 →
    students = 89 →
    pens_per_student = 58 →
    locker_fraction = 2 →
    total_pens_given = students * pens_per_student →
    remainder = initial_pens - total_pens_given →
    locker_pens = remainder / locker_fraction →
    home_pens = locker_pens →
    home_pens = 34 :=
by {
  sorry
}

end sally_took_home_pens_l333_333159


namespace theo_eggs_needed_l333_333294

def customers_first_hour : ℕ := 5
def customers_second_hour : ℕ := 7
def customers_third_hour : ℕ := 3
def customers_fourth_hour : ℕ := 8
def eggs_per_3_egg_omelette : ℕ := 3
def eggs_per_4_egg_omelette : ℕ := 4

theorem theo_eggs_needed :
  (customers_first_hour * eggs_per_3_egg_omelette) +
  (customers_second_hour * eggs_per_4_egg_omelette) +
  (customers_third_hour * eggs_per_3_egg_omelette) +
  (customers_fourth_hour * eggs_per_4_egg_omelette) = 84 := by
  sorry

end theo_eggs_needed_l333_333294


namespace probability_at_B_after_6_steps_l333_333715

-- Define the hexagonal lattice and properties
constant HexagonalLattice : Type
constant A : HexagonalLattice
constant B : HexagonalLattice
constant neighbors : HexagonalLattice → Set HexagonalLattice
axiom hex_neighbors : ∀ x : HexagonalLattice, neighbors x ≠ ∅

-- Define the random walk of the ant on the lattice
noncomputable def random_walk (start : HexagonalLattice) (steps : ℕ) : Set HexagonalLattice :=
sorry  -- Detailed implementation of random walk is abstracted

-- The theorem to prove
theorem probability_at_B_after_6_steps :
  let prob := 1/6 in
  (random_walk A 6).card = 1 → 
  ∃! outcome : HexagonalLattice, outcome = B ∧
  (neighbors B).card = 6 → 
  True :=
sorry

end probability_at_B_after_6_steps_l333_333715


namespace julian_needs_more_legos_l333_333106

theorem julian_needs_more_legos : ∀ (lego_count airplane_model_count legos_per_model : ℕ) 
                                  (H1 : lego_count = 400) 
                                  (H2 : airplane_model_count = 2) 
                                  (H3 : legos_per_model = 240), 
                                  (airplane_model_count * legos_per_model) - lego_count = 80 :=
by
  intros lego_count airplane_model_count legos_per_model H1 H2 H3
  rw [H1, H2, H3]
  sorry

end julian_needs_more_legos_l333_333106


namespace dart_board_center_square_probability_l333_333678

-- Definitions given by the conditions
def is_regular_octagon (A : Type) [NormedRing A] (s : A) : Prop :=
  true -- Placeholder definition for a regular octagon with side length s

-- Assume the dart lands equally likely anywhere on the dart board
def dart_lands_equally_likely (A : Type) [NormedRing A] : Prop :=
  true -- Placeholder for uniformly random landing

-- Statement to be proved
theorem dart_board_center_square_probability {A : Type} [NormedRing A] (s : A) :
  is_regular_octagon A s → dart_lands_equally_likely A → 
  (let center_square_area := s^2 in let total_area := 2 * s^2 in center_square_area / total_area = 1 / 2) :=
by
  intros h1 h2
  sorry

end dart_board_center_square_probability_l333_333678


namespace range_of_a_l333_333069

noncomputable def is_increasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x > 1 then a^x else (4 - a / 2) * x + 2

theorem range_of_a (a : ℝ) :
  (is_increasing_on_ℝ (f a)) ↔ (4 ≤ a ∧ a < 8) :=
begin
  sorry
end

end range_of_a_l333_333069


namespace tangent_line_at_x_is_1_l333_333021

-- Define the function f(x)
def f (x : ℝ) := Real.log x + x^2 - 1/x

-- Define the statement that we need to prove
theorem tangent_line_at_x_is_1 : 
  ∃ m b, m = 4 ∧ b = -4 ∧ ∀ x y, y = m * (x - 1) + f 1 ↔ (y = 4 * x - 4) := by
  -- proof to be filled in
  sorry

end tangent_line_at_x_is_1_l333_333021


namespace find_k_l333_333114

variable x : ℝ
variable k : ℝ

-- Assume necessary conditions
axiom log_base_5_of_2_eq_x : log 5 2 = x
axiom log_base_10_of_32_eq_ky : log 10 32 = k * y

theorem find_k (h1 : log 5 2 = x) (h2 : log 10 32 = k * (log 5 2)) : k = 5 * log 2 5 := by
  sorry

end find_k_l333_333114


namespace y_coordinate_in_fourth_quadrant_l333_333071
-- Importing the necessary libraries

-- Definition of the problem statement
theorem y_coordinate_in_fourth_quadrant (x y : ℝ) (h : x = 5 ∧ y < 0) : y < 0 :=
by 
  sorry

end y_coordinate_in_fourth_quadrant_l333_333071


namespace triangle_inequality_inequality_l333_333468

theorem triangle_inequality_inequality (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  ( (a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b) )^(1/3) > (a^2 + b^2 + c^2) / 2 :=
by {
  have h₁ : a + b > c := h.1,
  have h₂ : b + c > a := h.2.1,
  have h₃ : c + a > b := h.2.2,
  sorry
}

end triangle_inequality_inequality_l333_333468


namespace binomial_sum_l333_333763

theorem binomial_sum (n m : ℕ) (h : 1 ≤ m ∧ m ≤ n) :
  (∑ k in Finset.range (n + 1), if m ≤ k ∧ k ≤ n then Nat.choose n k * Nat.choose k m else 0)
  = Nat.choose n m * 2 ^ (n - m) := by
  sorry

end binomial_sum_l333_333763


namespace part_one_solution_set_part_two_min_value_l333_333397

noncomputable def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem part_one_solution_set :
  {x : ℝ | f x ≤ 1} = set.Ici (-1) :=
by
  sorry

theorem part_two_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 3) :
  Real.Inf ({3 / a + a / b} : set ℝ) = 3 :=
by
  sorry

end part_one_solution_set_part_two_min_value_l333_333397


namespace simplify_trig_expression_l333_333922

theorem simplify_trig_expression :
  (tan 40 + tan 50 + tan 60 + tan 70) / cos 30 =
  2 * (cos 60 * cos 70 + sin 50 * cos 40 * cos 50) / (sqrt 3 * cos 40 * cos 50 * cos 60 * cos 70) :=
by
  sorry

end simplify_trig_expression_l333_333922


namespace solve_problem_l333_333299

noncomputable def problem_statement : ℤ :=
  (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2

theorem solve_problem : problem_statement = -166 :=
by 
  -- Proof omitted
  sorry

end solve_problem_l333_333299


namespace sum_of_two_numbers_eq_l333_333953

theorem sum_of_two_numbers_eq (x y : ℝ) (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 :=
by sorry

end sum_of_two_numbers_eq_l333_333953


namespace sqrt_factorial_sq_l333_333591

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333591


namespace smallest_period_of_f_decreasing_interval_of_f_min_value_of_f_on_interval_l333_333017

open Real Set

def f (x : ℝ) : ℝ := sin x - 2 * sqrt 3 * sin (x / 2) ^ 2

theorem smallest_period_of_f : is_periodic f (2 * π) :=
sorry

theorem decreasing_interval_of_f :
  ∃ k : ℤ, ∀ x : ℝ, x ∈ Icc (π / 6 + 2 * k * π) (7 * π / 6 + 2 * k * π) → is_decreasing_on f (Icc (π / 6 + 2 * k * π) (7 * π / 6 + 2 * k * π)) :=
sorry

theorem min_value_of_f_on_interval : 
  ∃ x ∈ Icc 0 (2 * π / 3), f x = -sqrt 3 :=
sorry

end smallest_period_of_f_decreasing_interval_of_f_min_value_of_f_on_interval_l333_333017


namespace length_AD_l333_333115

-- Definitions and conditions from part (a)
variable (A B C D : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable (right_triangle_ABC : triangle A B C)
variable (is_right_angle : angle A B C = 90)
variable (circle_with_diameter_BC : ∃ O, circle O (distance B C / 2) ∧ diameter O B C)
variable (intersects_AB_at_D : point D ∈ line_segment A B)
variable (BD : ℝ) (CD : ℝ)
variable (BD_val : BD = 3)
variable (CD_val : CD = 2)

-- Prove the length of AD
theorem length_AD : ∀ (A B C D : point),
  right_triangle_ABC A B C ->
  is_right_angle A B C ->
  circle_with_diameter_BC A B C ->
  intersects_AB_at_D A B C D ->
  BD A B C D = 3 ->
  CD A B C D = 2 ->
  AD A B C D = 4.5 := by
  sorry

end length_AD_l333_333115


namespace range_fm_plus_fn_max_fm_minus_fn_l333_333460

open Real

noncomputable def f (x a : ℝ) := log x + (1/2) * x^2 - (a+2) * x

theorem range_fm_plus_fn (m n a : ℝ) (h1 : m < n) (h2 : ∀ x, DiffableAt ℝ (f x a) m ∧ f m a = 0)
  (h3 : ∀ x, DiffableAt ℝ (f x a) n ∧ f n a = 0) : f m a + f n a < -3 := by
  sorry

theorem max_fm_minus_fn (m n a : ℝ) (h1 : m < n) (h2 : a ≥ sqrt e + 1 / sqrt e - 2)
  (h3 : ∀ x, DiffableAt ℝ (f x a) m ∧ f m a = 0)
  (h4 : ∀ x, DiffableAt ℝ (f x a) n ∧ f n a = 0) :
  max (f n a - f m a) = 1 - (e / 2) + (1 / (2 * e)) := by
  sorry

end range_fm_plus_fn_max_fm_minus_fn_l333_333460


namespace solve_system_of_equations_l333_333926

theorem solve_system_of_equations (x1 x2 x3 x4 x5 y : ℝ) :
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x5 →
  (y = 2 ∧ x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∨
  (y ≠ 2 ∧ (y^2 + y - 1 ≠ 0 ∧ x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (y^2 + y - 1 = 0 ∧ y = (1 / 2) * (-1 + Real.sqrt 5) ∨ y = (1 / 2) * (-1 - Real.sqrt 5) ∧
    ∃ a b : ℝ, x1 = a ∧ x2 = b ∧ x3 = y * b - a ∧ x4 = - y * (a + b) ∧ x5 = y * a - b))
:=
sorry

end solve_system_of_equations_l333_333926


namespace intersection_A_B_l333_333377

def A : Set ℝ := { x | 2 * x^2 - 5 * x < 0 }
def B : Set ℝ := { x | 3^(x - 1) ≥ Real.sqrt 3 }

theorem intersection_A_B : A ∩ B = Set.Ico (3 / 2) (5 / 2) := 
by
  sorry

end intersection_A_B_l333_333377


namespace marble_probability_l333_333656

theorem marble_probability 
  (total_marbles : ℕ)
  (P_white : ℚ)
  (P_green : ℚ)
  (white_marbles green_marbles : ℕ)
  (P_red_or_blue : ℚ)
  (h1 : total_marbles = 100)
  (h2 : P_white = 1 / 4)
  (h3 : P_green = 1 / 5)
  (h4 : white_marbles = (P_white * total_marbles).toNat)
  (h5 : green_marbles = (P_green * total_marbles).toNat)
  (h6 : P_red_or_blue = ((total_marbles - white_marbles - green_marbles) : ℚ) / total_marbles) :
  P_red_or_blue = 11 / 20 :=
sorry

end marble_probability_l333_333656


namespace points_concyclic_l333_333111

-- Define the main entities and their relationships
variable {A B C D M N I : Type}
variable [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq M] [DecidableEq N] [DecidableEq I]

-- Define a triangle
variable (triangle_ABC : Triangle A B C)

-- Define D as the intersection of internal bisector of ∠BAC with [BC]
variable (D : Point)
variable (bisector_A : IsAngleBisector A B C)
variable (intersection_BC : Intersects D (Segment B C))

-- Define M as the intersection of the perpendicular bisector of [AD] with the angle bisector from B
variable (M : Point)
variable (perpendicular_bisector_AD : IsPerpendicularBisector D A)
variable (intersection_B : Intersects M bisector_A.bisector_B)

-- Define N as the intersection of the perpendicular bisector of [AD] with the angle bisector from C
variable (N : Point)
variable (intersection_C : Intersects N bisector_A.bisector_C)

-- Define I as the incenter of triangle ABC
variable (I : Point)
variable (incenter_ABC : IsIncenter I A B C)

-- The theorem to be proved
theorem points_concyclic (triangle_ABC : Triangle A B C) (bisector_A : IsAngleBisector A B C) 
  (intersection_BC : Intersects D (Segment B C)) (perpendicular_bisector_AD : IsPerpendicularBisector D A)
  (intersection_B : Intersects M bisector_A.bisector_B) (intersection_C : Intersects N bisector_A.bisector_C)
  (incenter_ABC : IsIncenter I A B C) : Concyclic A M N I := 
by sorry

end points_concyclic_l333_333111


namespace find_coordinates_of_D_l333_333480

theorem find_coordinates_of_D :
  ∃ (D : ℝ × ℝ),
    (∃ (λ : ℝ), 0 ≤ λ ∧ λ ≤ 1 ∧ 
    D = (P.1 + λ * (Q.1 - P.1), P.2 + λ * (Q.2 - P.2))
    ∧ dist D P = 2 * dist D Q) ∧
    D = (3, 7) :=
by
  let P : ℝ × ℝ := (-3, -2)
  let Q : ℝ × ℝ := (5, 10)
  use (3, 7)
  use 0.75 -- because PD = 2DQ happens when λ = 0.75
  sorry

end find_coordinates_of_D_l333_333480


namespace sqrt_factorial_product_squared_l333_333580

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333580


namespace fourth_root_of_expression_l333_333052

theorem fourth_root_of_expression (x : ℝ) (h : 0 < x) : Real.sqrt (x^3 * Real.sqrt (x^2)) ^ (1 / 4) = x := sorry

end fourth_root_of_expression_l333_333052


namespace solution_to_equation_l333_333502

noncomputable def solve_equation (x : ℝ) : Prop :=
  x + 2 = 1 / (x - 2) ∧ x ≠ 2

theorem solution_to_equation (x : ℝ) (h : solve_equation x) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 :=
sorry

end solution_to_equation_l333_333502


namespace distance_between_lines_l333_333154

noncomputable def point_m : ℝ × ℝ := (0, 2)
def circle_c (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 25
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 6 = 0
def line_l' (x y : ℝ) : Prop := 4 * x - 3 * y + 2 = 0

theorem distance_between_lines : 
  ∀ x₁ y₁ x₂ y₂, 
    circle_c x₁ y₁ → 
    x₁ = 0 → 
    y₁ = 2 → 
    line_l 0 2 → 
    line_l' x₂ y₂ → 
    (|6 - 2|) / real.sqrt (16 + 9) = 4 / 5 := 
by {
  intros x₁ y₁ x₂ y₂ hc hx₁ hy₁ hl hl',
  sorry
}

end distance_between_lines_l333_333154


namespace length_of_bridge_l333_333707

-- Define the conditions
def length_of_train : ℝ := 750
def speed_of_train_kmh : ℝ := 120
def crossing_time : ℝ := 45
def wind_resistance_factor : ℝ := 0.10

-- Define the conversion from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 0.27778

-- Define the actual speed considering wind resistance
def actual_speed_ms (v : ℝ) (resistance : ℝ) : ℝ := (kmh_to_ms v) * (1 - resistance)

-- Define the total distance covered
def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem: Length of the bridge
theorem length_of_bridge : total_distance (actual_speed_ms speed_of_train_kmh wind_resistance_factor) crossing_time - length_of_train = 600 := by
  sorry

end length_of_bridge_l333_333707


namespace train_length_eq_210_l333_333281

-- Define the conditions
def train_speed_kmh : ℝ := 108 -- Speed in km/hr
def crossing_time_s : ℝ := 7 -- Time in seconds

-- Convert the speed to m/s
def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600

-- State the proposition: The length of the train equals 210 meters
theorem train_length_eq_210 :
  let length_of_train := train_speed_ms * crossing_time_s in
  length_of_train = 210 :=
by
  sorry

end train_length_eq_210_l333_333281


namespace equivalency_of_4an_fn_l333_333186

noncomputable def is_kth_order_difference_sequence (a : ℕ → ℝ) (k : ℕ) : Prop :=
∀ n, a (n + k + 1) - (k + 1) * a (n + k) + (k * (k + 1) / 2) * a (n + k - 1) = 0

noncomputable def is_kth_degree_polynomial (f : ℕ → ℝ) (k : ℕ) : Prop :=
∃ P : polynomial ℝ, P.degree = k ∧ ∀ n, f n = P.eval n

theorem equivalency_of_4an_fn (a f : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, 4 * a n = f n) (h2 : is_kth_degree_polynomial f k) :
  is_kth_order_difference_sequence a k ↔ is_kth_degree_polynomial a k :=
sorry

end equivalency_of_4an_fn_l333_333186


namespace problem1_solution_set_problem2_min_value_l333_333399

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem problem1_solution_set : 
  {x : ℝ | f x ≤ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 ≤ x} :=
sorry

theorem problem2_min_value : 
  let m := 3 in
  ∀ (a b : ℝ), a + b = m → 0 < a → 0 < b → (3 / a) + (a / b) ≥ 3 :=
sorry

end problem1_solution_set_problem2_min_value_l333_333399


namespace cube_root_fraction_l333_333326

open Real

theorem cube_root_fraction (h : 16.2 = (81 / 5)) : 
  (∛(6 / 16.2)) = (∛10 / 3) :=
by
  sorry

end cube_root_fraction_l333_333326


namespace october_percentage_l333_333520

theorem october_percentage (total_writers october_writers july_writers : ℝ) 
  (h1 : total_writers = 200) (h2 : october_writers = 15) (h3 : july_writers = 14) : 
  (october_writers / total_writers * 100 = 7.5) ∧ (october_writers / total_writers * 100 > july_writers / total_writers * 100) := by
  -- Calculate the percentage for October
  calc 
    let october_percentage := october_writers / total_writers * 100;
  -- Calculate the percentage for July
    let july_percentage := july_writers / total_writers * 100;
  sorry

end october_percentage_l333_333520


namespace part_one_unique_root_part_two_max_min_part_three_range_a_l333_333375

-- Definitions from the problem conditions
def f (a b x : ℝ) := a * x ^ 2 + b * x

section part_one
variables (a b : ℝ)
-- Given conditions
axiom a_ne_zero : a ≠ 0
axiom f_at_2_zero : f a b 2 = 0

-- To prove:
theorem part_one_unique_root : 
  f a b x - x = 0 → f a b x = -1/2 * x ^ 2 + x :=
sorry
end part_one

section part_two
-- Given conditions for specific case a = 1
def f_special (x : ℝ) := x ^ 2 - 2 * x
def I := set.Icc (-1 : ℝ) (2 : ℝ)

-- To prove max and min values
theorem part_two_max_min : 
  (∀ x ∈ I, f_special x ≤ 3) ∧ (∀ x ∈ I, -1 ≤ f_special x) :=
sorry
end part_two

section part_three
variables (a : ℝ)
-- To prove range of a
theorem part_three_range_a : 
  (∀ x, x ≥ 2 → f a 0 x ≥ 2 - a) → a ≥ 2 :=
sorry
end part_three

end part_one_unique_root_part_two_max_min_part_three_range_a_l333_333375


namespace digit_rounding_problem_l333_333684

theorem digit_rounding_problem : 
  let A_candidates := {A | A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 73*1000 + A*100 + 94 rounded_to_nearest_thousand = 74000}
  count A_candidates = 5 :=
by
  sorry -- The proof is skipped, only the statement is provided

end digit_rounding_problem_l333_333684


namespace part1_part2_part3_l333_333773

def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^(x - 1) - 1) / (2^(x + 1) + a)

theorem part1 (h_odd : ∀ x, f a x = -f a (-x)) (h_a_gt_zero : a > 0) : a = 2 := sorry

def f_updated (x : ℝ) : ℝ := (2^x - 1) / (2 * (2^x + 1))

theorem part2 : ∀ x1 x2 : ℝ, x1 > x2 → f_updated x1 > f_updated x2 := sorry

theorem part3 (h_ineq : ∀ x : ℝ, f_updated(2 * m - m * Real.sin x) + f_updated((Real.cos x) ^ 2) ≥ 0) : 0 ≤ m := sorry

end part1_part2_part3_l333_333773


namespace minimum_distance_circle_to_line_l333_333368

def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 4 = 0
def line (x y : ℝ) : Prop := x - 2 * y - 5 = 0

theorem minimum_distance_circle_to_line :
  ∀ P : ℝ × ℝ, circle P.1 P.2 → ∃ d : ℝ, d = sqrt 5 - 1 :=
sorry

end minimum_distance_circle_to_line_l333_333368


namespace range_of_a_l333_333381

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 4 → deriv (λ x, x^2 + 2*(a-1)*x + 2) x ≤ 0) ∧
  (∀ x : ℝ, x ≥ 5 → deriv (λ x, x^2 + 2*(a-1)*x + 2) x ≥ 0) ↔
  -4 ≤ a ∧ a ≤ -3 :=
by sorry

end range_of_a_l333_333381


namespace find_guest_towel_price_l333_333721

-- Let's define the conditions as hypotheses
variable (G : ℝ) -- Price of each set of towels for the guest bathroom
variable (price_master : ℝ := 50) -- Price of each set of towels for the master bathroom
variable (num_guest : ℕ := 2) -- Number of sets for the guest bathroom
variable (num_master : ℕ := 4) -- Number of sets for the master bathroom
variable (discount : ℝ := 0.80) -- Store discount (20% off means Bailey pays 80%)
variable (total_spent : ℝ := 224) -- Total amount Bailey will spend

-- The equation representing the total amount spent
def towel_price_condition (G : ℝ) : Prop :=
  discount * (num_guest * G + num_master * price_master) = total_spent

-- The statement we need to prove
theorem find_guest_towel_price : towel_price_condition G → G = 40 := by
  sorry

end find_guest_towel_price_l333_333721


namespace find_minimum_a_l333_333024

theorem find_minimum_a (a x : ℤ) : 
  (x - a < 0) → 
  (x > -3 / 2) → 
  (∃ n : ℤ, ∀ y : ℤ, y ∈ {k | -1 ≤ k ∧ k ≤ n} ∧ y < a) → 
  a = 3 := sorry

end find_minimum_a_l333_333024


namespace rectangle_width_l333_333185

-- Conditions
def length (w : Real) : Real := 4 * w
def area (w : Real) : Real := w * length w

-- Theorem stating that the width of the rectangle is 5 inches if the area is 100 square inches
theorem rectangle_width (h : area w = 100) : w = 5 :=
sorry

end rectangle_width_l333_333185


namespace translate_quadratic_function_l333_333549

theorem translate_quadratic_function :
  ∀ x : ℝ, (y = (1 / 3) * x^2) →
          (y₂ = (1 / 3) * (x - 1)^2) →
          (y₃ = y₂ + 3) →
          y₃ = (1 / 3) * (x - 1)^2 + 3 := 
by 
  intros x h₁ h₂ h₃ 
  sorry

end translate_quadratic_function_l333_333549


namespace line_curve_intersection_l333_333091
open ComplexConjugate

noncomputable def x_t (t : ℝ) := 1 + 1/2 * t
noncomputable def y_t (t : ℝ) := (√3 / 2) * t
def polar_to_cartesian (ρ θ : ℝ) := ρ * (θ.sin)^2 - 4 * θ.cos = 0

-- At F(1, 0), prove the desired result
theorem line_curve_intersection :
  ∀ t1 t2 : ℝ,
  x_t t1 = x_t t2 ∧ y_t t1 = y_t t2 ∧ (t1 + t2 = 8 / 3) ∧ (t1 * t2 = -16 / 3) →
  ∃ A B : ℝ × ℝ,
  (A.fst = x_t t1 ∧ A.snd = y_t t1) ∧ 
  (B.fst = x_t t2 ∧ B.snd = y_t t2) ∧ 
  let FA := (Math.sqrt((1 - A.fst)^2 + (0 - A.snd)^2)) in
  let FB := (Math.sqrt((1 - B.fst)^2 + (0 - B.snd)^2)) in
  (1/FA + 1/FB) = 1 :=
sorry

end line_curve_intersection_l333_333091


namespace fourth_vertex_of_square_l333_333546

theorem fourth_vertex_of_square :
  ∃ z : ℂ, z = (1 / 6) - (5 / 6) * I ∧
         ∃ A B C : ℂ, A = (2 + 3 * I) ∧ B = (-3 + 2 * I) ∧ C = (2 - 3 * I) ∧
         is_square A B C z :=
sorry

end fourth_vertex_of_square_l333_333546


namespace range_of_f2_intersection_points_l333_333391

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + a * x^2 + 1 - a

theorem range_of_f2 (a : ℝ) (h : a > 3 / 2) : 
  ∃ r : Set ℝ, r = {y : ℝ | y > -5 / 2} ∧ f 2 a ∈ r := sorry

theorem intersection_points' (a : ℝ) (h : a > 3 / 2) : 
  let δ := (1 - a) ^ 2 - 4 * (2 - a)
  in if δ < 0 then true -- one intersection
  else if δ = 0 then true -- two intersections
  else a ≠ 2 → true -- three intersections := sorry

end range_of_f2_intersection_points_l333_333391


namespace triangle_area_is_zero_l333_333332

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def scalar_vector_mult (k : ℝ) (v : Point3D) : Point3D := {
  x := k * v.x,
  y := k * v.y,
  z := k * v.z
}

theorem triangle_area_is_zero : 
  let u := Point3D.mk 2 1 (-1)
  let v := Point3D.mk 5 4 1
  let w := Point3D.mk 11 10 5
  vector_sub w u = scalar_vector_mult 3 (vector_sub v u) →
-- If the points u, v, w are collinear, the area of the triangle formed by these points is zero:
  ∃ area : ℝ, area = 0 :=
by {
  sorry
}

end triangle_area_is_zero_l333_333332


namespace problem_statement_l333_333218

theorem problem_statement (x y : ℂ) (hx : x = 6 - 3 * complex.I) (hy : y = 2 + 3 * complex.I) :
  x - 3 * y = -12 * complex.I :=
by
  -- x and y are defined by the conditions hx and hy
  sorry

end problem_statement_l333_333218


namespace ellipse_properties_l333_333003

def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 4 = 1

def foci (c : ℝ) : Prop :=
  (0 < c ∧ c < 2 * sqrt 2 ∧ 2 * sqrt 2 < 3)

def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  P = (2, sqrt 2)

def distance_from_origin (d : ℝ) : Prop :=
  d = 2 * sqrt 6 / 3

theorem ellipse_properties :
  ∃ c a b : ℝ,
    0 < b ∧ b < a ∧ a < 3 ∧
    foci c ∧
    a = 2 * sqrt 2 ∧
    b^2 = a^2 - c^2 ∧
    ellipse_equation (2, sqrt 2).fst (2, sqrt 2).snd ∧
    point_on_ellipse (2, sqrt 2) ∧
    ∃ d : ℝ, distance_from_origin d :=
by
  sorry

end ellipse_properties_l333_333003


namespace greatest_k_dividing_n_l333_333694

noncomputable def num_divisors (n : ℕ) : ℕ :=
  n.divisors.card

theorem greatest_k_dividing_n (n : ℕ) (h_pos : n > 0)
  (h_n_divisors : num_divisors n = 120)
  (h_5n_divisors : num_divisors (5 * n) = 144) :
  ∃ k : ℕ, 5^k ∣ n ∧ (∀ m : ℕ, 5^m ∣ n → m ≤ k) ∧ k = 4 :=
by sorry

end greatest_k_dividing_n_l333_333694


namespace estimate_blue_cards_l333_333837

theorem estimate_blue_cards (total_red_cards : ℕ) (blue_card_freq : ℝ) (h : total_red_cards = 10) (h_freq : blue_card_freq = 0.8) : 
    ∃ (total_blue_cards : ℕ), total_blue_cards = 40 :=
begin
  use 40,
  sorry
end

end estimate_blue_cards_l333_333837


namespace sqrt_factorial_product_squared_l333_333581

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333581


namespace qingdao_mock_exam_2015_l333_333665

theorem qingdao_mock_exam_2015 (m : ℝ) (h : (tan m + sin m + 2015 = 2)) :
  (tan (-m) + sin (-m) + 2015 = 4028) :=
by
  sorry

end qingdao_mock_exam_2015_l333_333665


namespace parabola_vertex_value_of_a_l333_333182

-- Define the conditions as given in the math problem
variables (a b c : ℤ)
def quadratic_fun (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Given conditions about the vertex and a point on the parabola
def vertex_condition : Prop := (quadratic_fun a b c 2 = 3)
def point_condition : Prop := (quadratic_fun a b c 1 = 0)

-- Statement to prove
theorem parabola_vertex_value_of_a : vertex_condition a b c ∧ point_condition a b c → a = -3 :=
sorry

end parabola_vertex_value_of_a_l333_333182


namespace matthew_hotdogs_needed_l333_333135

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end matthew_hotdogs_needed_l333_333135


namespace sum_remainder_eq_zero_or_one_l333_333886

theorem sum_remainder_eq_zero_or_one {p q : ℕ} (h_coprime : Nat.coprime p q) :
  (∑ n in Finset.range (p * q), (-1) ^ (n % p + n % q)) =
    if (p * q) % 2 = 0 then 0 else 1 :=
sorry

end sum_remainder_eq_zero_or_one_l333_333886


namespace range_of_a_l333_333189

theorem range_of_a (a b : ℝ) : 
  (∀ x ∈ Icc (-1 : ℝ) (∞), dydx (x^2 + 2*a*x + b) ≥ 0) →
  ∃ a_range : Set ℝ, a_range = {a : ℝ | a ≥ 1} := 
by
  sorry

end range_of_a_l333_333189


namespace sqrt_factorial_sq_l333_333584

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333584


namespace factorial_sqrt_sq_l333_333617

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333617


namespace part_a_part_b_l333_333477

variables {A B C A1 B1 C1 M P Q : Type}
variables (BC CA AB : Triangle) (line1 line2 line3 : Line)

-- Assuming the conditions given in the problem
-- Point A1 on BC, B1 on CA, C1 on AB
-- Lines B1C1, BB1, and CC1 intersect AA1 at points M, P, and Q respectively

-- Prove part (a)
theorem part_a (H : Triangle A B C)
  (H1 : Point_on_Side BC A1)
  (H2 : Point_on_Side CA B1)
  (H3 : Point_on_Side AB C1)
  (H4 : Intersect B1 C1 AA1 M)
  (H5 : Intersect B B1 AA1 P)
  (H6 : Intersect C C1 AA1 Q) :
  (A1M / MA = (A1P / PA) + (A1Q / QA)) :=
sorry

-- Prove part (b)
theorem part_b (H : Triangle A B C)
  (H1 : Point_on_Side BC A1)
  (H2 : Point_on_Side CA B1)
  (H3 : Point_on_Side AB C1)
  (H4 : Intersect B1 C1 AA1 M)
  (H5 : Intersect B B1 AA1 P)
  (H6 : Intersect C C1 AA1 Q)
  (H7 : P = Q) :
  (MC1 / MB1 = (BC1 / AB) / (CB1 / AC)) :=
sorry

end part_a_part_b_l333_333477


namespace selfie_ratio_l333_333426

theorem selfie_ratio (x y : ℕ) (total : ℕ) (extra : ℕ) (h1 : total = x + y) (h2 : y = x + extra) 
                     (h3 : total = 2430) (h4 : extra = 630) :
                     (x : ℚ) / y = 10 / 17 :=
by
  have : x + (x + extra) = total, from sorry
  have : x + x + extra = total, from sorry
  have : 2 * x + extra = total, from sorry
  have : 2 * x + 630 = 2430, from sorry
  have : 2 * x = 1800, from sorry
  have : x = 900, from sorry
  have : y = 900 + 630, from sorry
  have : y = 1530, from sorry
  have : (900 : ℚ) / 1530 = 10 / 17, from sorry
  show (x : ℚ) / y = 10 / 17, from this

end selfie_ratio_l333_333426


namespace find_x_l333_333956

-- Define the given conditions
def constant_ratio (k : ℚ) : Prop :=
  ∀ (x y : ℚ), (3 * x - 4) / (y + 15) = k

def initial_condition (k : ℚ) : Prop :=
  (3 * 5 - 4) / (4 + 15) = k

def new_condition (k : ℚ) (x : ℚ) : Prop :=
  (3 * x - 4) / 30 = k

-- Prove that x = 406/57 given the conditions
theorem find_x (k : ℚ) (x : ℚ) :
  constant_ratio k →
  initial_condition k →
  new_condition k x →
  x = 406 / 57 :=
  sorry

end find_x_l333_333956


namespace tet_surface_area_l333_333959

theorem tet_surface_area (side_len : ℝ) (midpoint1 midpoint2 : ℝ) (folding_pts: ℝ) : 
  (∃ (S : ℝ), S = 8 * real.pi) :=
begin
  sorry
end

end tet_surface_area_l333_333959


namespace cube_root_simplification_l333_333162

theorem cube_root_simplification :
  ∛(8 + 27) * (∛(27 + ∛27)) = ∛1050 :=
by
  sorry

end cube_root_simplification_l333_333162


namespace centroid_on_parabola_l333_333827

noncomputable def parabola (a b c : ℝ) (x : ℝ) := a*x^2 + b*x + c

theorem centroid_on_parabola :
  ∃ x_c : ℝ,
  let A := (6 : ℝ, 6 : ℝ)
  let B := (-6 : ℝ, 6 : ℝ)
  let C := (x_c, 0 : ℝ)
  let G := ((6 - 6 + x_c) / 3, (6 + 6 + 0) / 3)
  (∀ x y : ℝ, (parabola (1/6) 0 0 x = y ↔ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = 6)) ∧ 
   G.2 = parabola (1/6) 0 0 G.1)
  → (G = (2*Real.sqrt 6, 4)) ∨ (G = (-2*Real.sqrt 6, 4)) :=
by
  sorry

end centroid_on_parabola_l333_333827


namespace julian_needs_more_legos_l333_333105

theorem julian_needs_more_legos : ∀ (lego_count airplane_model_count legos_per_model : ℕ) 
                                  (H1 : lego_count = 400) 
                                  (H2 : airplane_model_count = 2) 
                                  (H3 : legos_per_model = 240), 
                                  (airplane_model_count * legos_per_model) - lego_count = 80 :=
by
  intros lego_count airplane_model_count legos_per_model H1 H2 H3
  rw [H1, H2, H3]
  sorry

end julian_needs_more_legos_l333_333105


namespace continued_fraction_identity_l333_333508

noncomputable def alpha (a : ℕ → ℕ) : ℝ :=
  let cf : List ℕ := List.zipWith (fun a b => a / (b : ℝ)) (List (0::List.repeat 1 a.length)) (List.init a.length) 
  let s_n (n : ℕ) : ℝ := sum (List.map (fun i => (-1)^i * (1 / (cf.nth i * cf.nth (i + 1)))) (List.range n))
  a.head + s_n ((a.length - 1))

theorem continued_fraction_identity (a : ℕ → ℕ) (Q : ℕ → ℕ) : 
  alpha a = a.head + (∑ n, (-1:ℝ)^n * (1 / (Q n * Q (n + 1)))) :=
sorry

end continued_fraction_identity_l333_333508


namespace angle_east_northwest_l333_333258

def num_spokes : ℕ := 12
def central_angle : ℕ := 360 / num_spokes
def angle_between (start_dir end_dir : ℕ) : ℕ := (end_dir - start_dir) * central_angle

theorem angle_east_northwest : angle_between 3 9 = 90 := sorry

end angle_east_northwest_l333_333258


namespace lines_intersect_not_perpendicular_l333_333793

noncomputable def slopes_are_roots (m k1 k2 : ℝ) : Prop :=
  k1^2 + m*k1 - 2 = 0 ∧ k2^2 + m*k2 - 2 = 0

theorem lines_intersect_not_perpendicular (m k1 k2 : ℝ) (h : slopes_are_roots m k1 k2) : (k1 * k2 = -2 ∧ k1 ≠ k2) → ∃ l1 l2 : ℝ, l1 ≠ l2 ∧ l1 = k1 ∧ l2 = k2 :=
by
  sorry

end lines_intersect_not_perpendicular_l333_333793


namespace area_of_triangle_union_reflection_l333_333708

def area_of_union_of_reflected_triangle : ℝ :=
  let A := (2:ℝ, 3:ℝ)
  let B := (4:ℝ, 1:ℝ)
  let C := (6:ℝ, 6:ℝ)
  let A' := (2:ℝ, 1:ℝ)
  let B' := (4:ℝ, 3:ℝ)
  let C' := (6:ℝ, -2:ℝ)
  let area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
    (0.5) * float.abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

  let area_original := area 2 3 4 1 6 6
  let area_reflected := area 2 1 4 3 6 (-2)

  area_original + area_reflected

theorem area_of_triangle_union_reflection :
  area_of_union_of_reflected_triangle = 14 := sorry

end area_of_triangle_union_reflection_l333_333708


namespace norm_of_scaled_vector_l333_333771

variables {u : EuclideanSpace ℝ (Fin 2)}

-- Given that the norm of vector u is 7
theorem norm_of_scaled_vector (h : ∥u∥ = 7) : ∥5 • u∥ = 35 :=
sorry

end norm_of_scaled_vector_l333_333771


namespace determine_first_quarter_points_l333_333082

-- Define the given conditions
variables {a d b r : ℕ}
variables {total_pts_q4 : ℤ}

-- Condition 1: Raiders' quarterly scores form an arithmetic sequence
def raiders_scores : list ℕ := [a, a+d, a+2d, a+3d]

-- Condition 2: Wildcats' quarterly scores form a geometric sequence
def wildcats_scores : list ℕ := [b, b*r, b*(r^2), b*(r^3)]

-- Condition 3: Scores at halftime are equal
def halftime_tied : Prop := a + (a + d) = b + (b * r)

-- Condition 4: Combined score in the fourth quarter is half of the total combined score
def combined_q4 : Prop :=
  (a + 3*d) + (b * r^3) = 1/2 * (4*a + 6*d + b*(1 + r + (r^2) + (r^3)))

-- Condition 5: Neither team scored more than 100 points in total
def raiders_under_100 : Prop := a + (a+d) + (a+2*d) + (a+3*d) < 100
def wildcats_under_100 : Prop := b + (b*r) + (b*(r^2)) + (b*(r^3)) < 100

-- Theorem: Determine the first quarter points
theorem determine_first_quarter_points : ∃ (q1_points : ℕ),
  halftime_tied ∧ combined_q4 ∧ raiders_under_100 ∧ wildcats_under_100 →
  (q1_points = a + b) :=
begin
  sorry
end

end determine_first_quarter_points_l333_333082


namespace angle_measure_l333_333535

theorem angle_measure (x : ℝ) 
  (h1 : 5 * x + 12 = 180 - x) : x = 28 := by
  sorry

end angle_measure_l333_333535


namespace total_distance_l333_333900

/-- 
Given:
1. Mary ran 3/8 of the total distance D.
2. Edna ran 2/3 of the distance Mary ran.
3. Lucy ran 5/6 of the distance Edna ran.
4. Lucy should run 4 more kilometers to cover the same distance as Mary.

Prove that the total distance of the field D is 24 kilometers.
-/
theorem total_distance (D : ℝ) (Mary Edna Lucy : ℝ) 
  (hMary : Mary = (3/8) * D)
  (hEdna : Edna = (2/3) * Mary)
  (hLucy : Lucy = (5/6) * Edna)
  (hLucy_more : Lucy + 4 = Mary) :
  D = 24 := 
begin
  sorry
end

end total_distance_l333_333900


namespace problem1_solution_set_problem2_min_value_l333_333400

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem problem1_solution_set : 
  {x : ℝ | f x ≤ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 ≤ x} :=
sorry

theorem problem2_min_value : 
  let m := 3 in
  ∀ (a b : ℝ), a + b = m → 0 < a → 0 < b → (3 / a) + (a / b) ≥ 3 :=
sorry

end problem1_solution_set_problem2_min_value_l333_333400


namespace least_tiles_needed_l333_333655

theorem least_tiles_needed 
  (length_cm width_cm : ℕ) 
  (h1 : length_cm = 720) 
  (h2 : width_cm = 432) : 
  let gcd_length_width := Nat.gcd length_cm width_cm in
  gcd_length_width = 144 ∧ (length_cm / gcd_length_width) * (width_cm / gcd_length_width) = 15 :=
by
  sorry

end least_tiles_needed_l333_333655


namespace first_player_wins_l333_333201

theorem first_player_wins (
  (points : Finset Point) (h_points_card : points.card = 1993)
  (segments : Finset (Point × Point)) 
  (h_non_intersecting : ∀ (s1 s2 : Point × Point), s1 ∈ segments → s2 ∈ segments → s1 ≠ s2 → 
                        ¬ (intersect s1 s2)) 
  (h_no_cycle : ∀ (polygon : Finset (Point × Point)), polygon ⊆ segments → 
                ¬ (is_closed_polygonal_chain polygon))
  (players_take_turns : ∀ (p1 p2 : Player), turn p1 → ¬ turn p2)
  (h_adjacent_move : ∀ (p : Point), last_move.adjacent_to p → next_move_on p)
  (h_losing_condition : ∀ (p : Player), ¬ next_move_on.available_moves p → loses p)
) : 
  ∃ (first_player : Player), has_winning_strategy first_player :=
sorry

end first_player_wins_l333_333201


namespace sqrt_factorial_product_squared_l333_333577

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333577


namespace coefficient_x4_in_f_f_x_l333_333394

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then (x + 1)^5 else x^2 + 2

theorem coefficient_x4_in_f_f_x 
  (h0 : 0 < x)
  (h1 : x < 1) : 
  let ff := f (f x) in 
  (polynomial.expand (ff) : polynomial ℝ).coeff 4 = 270 :=
sorry

end coefficient_x4_in_f_f_x_l333_333394


namespace ratio_of_nuts_to_raisins_l333_333304

theorem ratio_of_nuts_to_raisins 
  (R N : ℝ) 
  (h_ratio : 3 * R = 0.2727272727272727 * (3 * R + 4 * N)) : 
  N = 2 * R := 
sorry

end ratio_of_nuts_to_raisins_l333_333304


namespace polar_equation_correct_l333_333951

noncomputable def polar_equation_of_line (x y : ℝ) : Prop :=
  (∃ (θ : ℝ), θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6) ∧ (x = y * (Real.sqrt 3 / 3))

theorem polar_equation_correct :
  ∀ (x y : ℝ), polar_equation_of_line x y ↔ (∃ (θ : ℝ), θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6) ∧ (x = (Real.sqrt 3 / 3) * y) :=
by
  intros x y
  split
  case mp =>
    intro h
    exact h
  case mpr =>
    intro h
    exact h
    sorry

end polar_equation_correct_l333_333951


namespace initial_roses_eq_six_l333_333974

-- Define the initial conditions
def initial_roses (x : ℕ) : Prop := 
  ∃ y : ℕ, y = 10 ∧ x + y = 16

-- The main theorem we need to prove
theorem initial_roses_eq_six : initial_roses 6 :=
by
  unfold initial_roses
  exists 10
  apply And.intro 
  . { refl }
  . { exact Nat.add_eq_right 6 10 sorry }

end initial_roses_eq_six_l333_333974


namespace prove_phi_shift_left_symmetric_l333_333417

theorem prove_phi_shift_left_symmetric (
  φ : ℝ,
  h1 : -π < φ,
  h2 : φ < 0
) :
  φ = -π / 4 := 
sorry

end prove_phi_shift_left_symmetric_l333_333417


namespace statement_D_incorrect_l333_333132

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem statement_D_incorrect : ¬ (∀ x y : ℝ, x ∈ Ioo (Real.pi / 2) Real.pi → y ∈ Ioo (Real.pi / 2) Real.pi → x < y → f x > f y) := 
sorry

end statement_D_incorrect_l333_333132


namespace sin_sum_to_product_l333_333325

theorem sin_sum_to_product (x : ℝ) : sin (4 * x) + sin (6 * x) = 2 * sin (5 * x) * cos x := 
  sorry

end sin_sum_to_product_l333_333325


namespace tony_needs_23_gallons_l333_333548
noncomputable def gallons_of_paint_needed 
  (num_columns : ℕ) (height : ℕ) (diameter : ℕ) (coverage_per_gallon : ℕ) : ℕ :=
  let r := diameter / 2
  let lateral_surface_area := 2 * Float.pi * r * height
  let total_area := lateral_surface_area * num_columns
  let gallons := total_area / coverage_per_gallon
  Float.ceil gallons

theorem tony_needs_23_gallons :
  gallons_of_paint_needed 20 12 12 400 = 23 :=
by
  sorry

end tony_needs_23_gallons_l333_333548


namespace student_error_difference_l333_333083

theorem student_error_difference (num : ℤ) (num_val : num = 480) : 
  (5 / 6 * num - 5 / 16 * num) = 250 := 
by 
  sorry

end student_error_difference_l333_333083


namespace intersection_of_A_and_B_l333_333001

-- Definitions of the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Statement to prove the intersection of sets A and B is {3}
theorem intersection_of_A_and_B : A ∩ B = {3} :=
sorry

end intersection_of_A_and_B_l333_333001


namespace can_create_program_a_cannot_create_program_b_l333_333658

-- Define the operation φ for part a
def φ (x y : ℝ) := x * y + x + y + 1

-- Define the polynomials sequence for part a
def f_seq_a (f : ℕ → ℝ → ℝ) : Prop :=
  f 1 = λ x, x ∧
  (∀ i : ℕ, 2 ≤ i → f i = (λ x, ∃ j k < i, f j x * f k x + f j x + f k x + 1)) ∧
  f 1983 = λ x, ∑ i in range (1983), x ^ i + 1

-- Define the theorem for part a
theorem can_create_program_a : ∃ (f : ℕ → ℝ → ℝ), f_seq_a f :=
sorry

-- Define the operation φ for part b
def φ' (x y : ℝ) := x * y + x + y

-- Define the polynomials sequence for part b
def f_seq_b (f : ℕ → ℝ → ℝ) : Prop :=
  f 1 = λ x, x ∧
  (∀ i : ℕ, 2 ≤ i → f i = (λ x, ∃ j k < i, f j x * f k x + f j x + f k x)) ∧
  f 1983 = λ x, ∑ i in range (1983), x ^ i + 1

-- Define the theorem for part b
theorem cannot_create_program_b : ¬ ∃ (f : ℕ → ℝ → ℝ), f_seq_b f :=
sorry

end can_create_program_a_cannot_create_program_b_l333_333658


namespace num_multiples_6_not_12_lt_300_l333_333038

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l333_333038


namespace measure_angle_B_max_area_triangle_l333_333007

theorem measure_angle_B (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < B) (h2 : B < real.pi) 
  (h3 : b * real.sin A + a * real.cos B = 0) :
  B = 3 * real.pi / 4 :=
sorry

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < B) (h2 : B < real.pi) 
  (h3 : b = 2) (h4 : b * real.sin A + a * real.cos B = 0) :
  (∃ (S : ℝ), S = (sqrt 2) - 1) :=
sorry

end measure_angle_B_max_area_triangle_l333_333007


namespace number_of_correct_statements_l333_333712

theorem number_of_correct_statements : (∑ i in [false, true, true, true], if i then 1 else 0) = 3 := by
  trivial

end number_of_correct_statements_l333_333712


namespace original_number_contains_digit_geq_5_l333_333545

theorem original_number_contains_digit_geq_5
  {n : ℕ} (h_zero_free : ∀ d ∈ (n.digits 10), d ≠ 0)
  (h_rearranged_sum : let rearranged_sums := (finset.univ : finset (perm n.digits 10)).sum (λ perm, (list.of_digits 10 (perm n.digits 10))) in 
                        ∀ d ∈ rearranged_sums.digits 10, d = 1) : 
  ∃ d ∈ (n.digits 10), 5 ≤ d :=
sorry

end original_number_contains_digit_geq_5_l333_333545


namespace find_a_g_range_l333_333805

noncomputable def f (x a : ℝ) : ℝ := x^2 + 4 * a * x + 2 * a + 6
noncomputable def g (a : ℝ) : ℝ := 2 - a * |a - 1|

theorem find_a (x a : ℝ) :
  (∀ x, f x a ≥ 0) ∧ (∀ x, f x a = 0 → x^2 + 4 * a * x + 2 * a + 6 = 0) ↔ (a = -1 ∨ a = 3 / 2) :=
  sorry

theorem g_range :
  (∀ x, f x a ≥ 0) ∧ (-1 ≤ a ∧ a ≤ 3/2) → (∀ a, (5 / 4 ≤ g a ∧ g a ≤ 4)) :=
  sorry

end find_a_g_range_l333_333805


namespace inverse_variation_l333_333916

theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 :=
by 
  sorry

end inverse_variation_l333_333916


namespace circle_center_polar_coordinates_l333_333095

theorem circle_center_polar_coordinates :
  ∀ (θ : ℝ), (x = cos θ ∧ y = -1 + sin θ) → 
  ((0, -1) = (1, -(real.pi / 2))) :=
by
  sorry

end circle_center_polar_coordinates_l333_333095


namespace student_average_greater_l333_333278

variable (x y z : ℝ)

def A : ℝ := (x + y + z) / 3

def B : ℝ := ((y + z) / 2 + x) / 2

theorem student_average_greater (hxy : x > y) (hyz : y > z) : B x y z > A x y z := 
by
  sorry

end student_average_greater_l333_333278


namespace count_valid_n_divisible_7_l333_333880

theorem count_valid_n_divisible_7 :
  let n_valid := λ (n : ℕ), 
    let q := n / 100 in
    let r := n % 100 in
    10000 ≤ n ∧ n ≤ 99999 ∧ (q - r) % 7 = 0 in
  (Finset.filter n_valid (Finset.range 100000)).card = N := by
  sorry

end count_valid_n_divisible_7_l333_333880


namespace force_on_dam_l333_333239

noncomputable theory

-- Definitions
def trapezoidForce (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * ((b / 2) - ((b - a) / 3))

-- Given conditions
constants (ρ g a b h : ℝ)
axiom rho_val : ρ = 1000
axiom g_val : g = 10
axiom a_val : a = 5.1
axiom b_val : b = 7.8
axiom h_val : h = 3.0

-- Goal
theorem force_on_dam : trapezoidForce ρ g a b h = 270000 := by
  rw [rho_val, g_val, a_val, b_val, h_val]
  norm_num
  sorry

end force_on_dam_l333_333239


namespace sequence_cycles_iff_odd_l333_333506

noncomputable def symbol := {A, B, C}
def transform : (Fin n → symbol) → (Fin n → symbol)
| R => λ i, if R i = R (i + 1) % n then R i else (symbol.diff {R i, R (i + 1) % n}).choose

theorem sequence_cycles_iff_odd (n : ℕ) (h : n > 1) :
  (∃ m > 0, ∃ R0 : Fin n→symbol, transform^[m] R0 = R0) ↔ n % 2 = 1 :=
by
  sorry

end sequence_cycles_iff_odd_l333_333506


namespace algebra_expression_value_l333_333418

theorem algebra_expression_value (x y : ℤ) (h : x - 2 * y + 2 = 5) : 2 * x - 4 * y - 1 = 5 :=
by
  sorry

end algebra_expression_value_l333_333418


namespace value_range_l333_333199

-- Step to ensure proofs about sine and real numbers are within scope
open Real

noncomputable def y (x : ℝ) : ℝ := 2 * sin x * cos x - 1

theorem value_range (x : ℝ) : -2 ≤ y x ∧ y x ≤ 0 :=
by sorry

end value_range_l333_333199


namespace find_omega_dot_product_BA_BC_l333_333029

noncomputable def m (ω x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin (ω * x), (Real.cos (ω * x))^2 - (Real.sin (ω * x))^2)

noncomputable def n (ω x : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos (ω * x), 1)

noncomputable def f (ω x : ℝ) : ℝ :=
  m ω x.1 * n ω x.1.1 + m ω x.2 * n ω x.2.2

theorem find_omega (ω : ℝ) (hω : 0 < ω) : f ω x = 2 * Real.sin(2 * ω * x + π / 6) → 
  ∃ (ω : ℝ), 2 * π / (2 * ω) = π → ω = 1 :=
by sorry

theorem dot_product_BA_BC (ω : ℝ) (A B : ℝ) (a b c : ℝ) (h1 : ω = 1) (h2 : f B = -2) 
(h3 : a = Real.sqrt 3) (h4 : b = 3) 
(h5 : c = Real.sqrt 3) :
  ∃ (dot_product : ℝ), dot_product = a * c * Real.cos ((2*π) / 3) → dot_product = -3 / 2 :=
by sorry

end find_omega_dot_product_BA_BC_l333_333029


namespace find_lambda_l333_333809

open Real

variables (a b : ℝ)
variable (λ : ℝ)
variables (a_vec b_vec : ℝ × ℝ)
variables
  (H₁ : ‖a_vec‖ = 2)
  (H₂ : ‖b_vec‖ = sqrt 2)
  (H₃ : acos ((a_vec.1 * b_vec.1 + a_vec.2 * b_vec.2) / (‖a_vec‖ * ‖b_vec‖)) = π / 4)
  (H₄ : (λ * b_vec.1 - a_vec.1) * a_vec.1 + (λ * b_vec.2 - a_vec.2) * a_vec.2 = 0)

theorem find_lambda : λ = 2 := 
sorry

end find_lambda_l333_333809


namespace megan_initial_cupcakes_l333_333139

noncomputable def initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) : Nat :=
  packages * cupcakes_per_package + cupcakes_eaten

theorem megan_initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) :
  packages = 4 → cupcakes_per_package = 7 → cupcakes_eaten = 43 →
  initial_cupcakes packages cupcakes_per_package cupcakes_eaten = 71 :=
by
  intros
  simp [initial_cupcakes]
  sorry

end megan_initial_cupcakes_l333_333139


namespace smallest_sum_two_3digit_numbers_l333_333989

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem smallest_sum_two_3digit_numbers :
  ∀ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f) ∧
  (c ≠ d ∧ c ≠ e ∧ c ≠ f) ∧
  (d ≠ e ∧ d ≠ f) ∧
  (e ≠ f) ∧
  ({a, b, c, d, e, f} = {3, 4, 5, 6, 7, 8}) ∧
  (is_odd (100 * a + 10 * b + c) ∨ is_odd (100 * d + 10 * e + f)) ∧
  (is_even (100 * a + 10 * b + c) ∨ is_even (100 * d + 10 * e + f))
  → min ((100 * a + 10 * b + c) + (100 * d + 10 * e + f))
       ((100 * d + 10 * e + f) + (100 * a + 10 * b + c)) = 1257 :=
sorry

end smallest_sum_two_3digit_numbers_l333_333989


namespace num_three_digit_numbers_l333_333818

theorem num_three_digit_numbers (a b c : ℕ) :
  a ≠ 0 →
  b = (a + c) / 2 →
  c = a - b →
  ∃ n1 n2 n3 : ℕ, 
    (n1 = 100 * 3 + 10 * 2 + 1) ∧
    (n2 = 100 * 9 + 10 * 6 + 3) ∧
    (n3 = 100 * 6 + 10 * 4 + 2) ∧ 
    3 = 3 := 
sorry  

end num_three_digit_numbers_l333_333818


namespace non_isosceles_count_l333_333927

def n : ℕ := 20

def total_triangles : ℕ := Nat.choose n 3

def isosceles_triangles_per_vertex : ℕ := 9

def total_isosceles_triangles : ℕ := n * isosceles_triangles_per_vertex

def non_isosceles_triangles : ℕ := total_triangles - total_isosceles_triangles

theorem non_isosceles_count :
  non_isosceles_triangles = 960 := 
  by 
    -- proof details would go here
    sorry

end non_isosceles_count_l333_333927


namespace prove_area_and_sum_l333_333695

-- Define the coordinates of the vertices of the quadrilateral.
variables (a b : ℤ)

-- Define the non-computable requirements related to the problem.
noncomputable def problem_statement : Prop :=
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a > b ∧ (4 * a * b = 32) ∧ (a + b = 5)

theorem prove_area_and_sum : problem_statement := 
sorry

end prove_area_and_sum_l333_333695


namespace remainder_12345678901_mod_101_l333_333564

theorem remainder_12345678901_mod_101 : 12345678901 % 101 = 24 :=
by
  sorry

end remainder_12345678901_mod_101_l333_333564


namespace max_value_fraction_l333_333055

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end max_value_fraction_l333_333055


namespace cows_with_no_spot_l333_333906

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end cows_with_no_spot_l333_333906


namespace parallelepiped_ratio_l333_333305

-- Define vectors as per the problem conditions
def v : ℝ × ℝ × ℝ := (2, 1, 0)
def w : ℝ × ℝ × ℝ := (0, 1, 2)
def u : ℝ × ℝ × ℝ := (2, 0, 1)

-- Define norms as the squares of Euclidean distances
noncomputable def norm_sq (a : ℝ × ℝ × ℝ) : ℝ :=
  a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2

noncomputable def qr_sq := norm_sq (v.1 - w.1, v.2 - w.2, v.3 - w.3)
noncomputable def ps_sq := norm_sq u
noncomputable def qt_sq := norm_sq (v.1 - u.1 - w.1, v.2 - u.2 - w.2, v.3 - u.3 - w.3)
noncomputable def rs_sq := norm_sq (w.1 - u.1, w.2 - u.2, w.3 - u.3)

noncomputable def pq_sq := norm_sq v
noncomputable def pr_sq := norm_sq w
noncomputable def ps_sq2 := norm_sq u

theorem parallelepiped_ratio : (qr_sq + ps_sq + qt_sq + rs_sq) / (pq_sq + pr_sq + ps_sq2) = 28 / 15 := by
  sorry

end parallelepiped_ratio_l333_333305


namespace two_perfect_squares_not_two_perfect_cubes_l333_333498

-- Define the initial conditions as Lean assertions
def isSumOfTwoPerfectSquares (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^2

def isSumOfTwoPerfectCubes (n : ℕ) := ∃ a b : ℕ, n = a^3 + b^3

-- Lean 4 statement to show 2005^2005 is a sum of two perfect squares
theorem two_perfect_squares :
  isSumOfTwoPerfectSquares (2005^2005) :=
sorry

-- Lean 4 statement to show 2005^2005 is not a sum of two perfect cubes
theorem not_two_perfect_cubes :
  ¬ isSumOfTwoPerfectCubes (2005^2005) :=
sorry

end two_perfect_squares_not_two_perfect_cubes_l333_333498


namespace find_varphi_l333_333775

open Real

noncomputable def y (x ϕ : ℝ) : ℝ := sin (3 * x + ϕ)

def axis_of_symmetry (x : ℝ) : ℝ := (3 / 4) * π

def axis_of_symmetry_condition (x ϕ : ℝ) : Prop :=
  x = (k : ℤ) * π / 3 + π / 6 - ϕ / 3

def condition_on_phi (ϕ : ℝ) : Prop :=
  abs ϕ < π / 2

theorem find_varphi :
  ∃ ϕ, (ϕ = π / 4) ∧ axis_of_symmetry (3 / 4 * π) = 3 / 4 * π ∧ condition_on_phi ϕ :=
begin
  sorry
end

end find_varphi_l333_333775


namespace find_x_l333_333810

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (x, 2)
def b : vector := (1, -1)

-- Dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Orthogonality condition rewritten in terms of dot product
def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

-- Main theorem to prove
theorem find_x (x : ℝ) (h : orthogonal ((a x).1 - b.1, (a x).2 - b.2) b) : x = 4 :=
by sorry

end find_x_l333_333810


namespace shaded_area_l333_333093

-- Define the necessary conditions and terms in Lean
def radius : ℝ := 6
def sector_angle_degrees : ℝ := 60
def pi : ℝ := real.pi

-- The formula for the area should be a ratio of the full circle area multiplied by the angle ratio in radians
def sector_area (r : ℝ) (angle_degrees : ℝ) : ℝ :=
  (angle_degrees / 360) * pi * r^2

-- The formula for the area of the right-angled triangle inscribed by the radius meeting at center by the given angle
def triangle_area (r : ℝ) (angle_degrees : ℝ) : ℝ :=
  0.5 * r * r * real.sin (real.of_nat angle_degrees * real.pi / 180)

-- Total area of shaded region as per problem description (2 triangles and 2 sectors)
def total_area (r : ℝ) (angle_degrees : ℝ) : ℝ :=
  2 * sector_area r angle_degrees + 2 * triangle_area r angle_degrees

-- Statement in Lean 4 proving the equivalent mathematics problem
theorem shaded_area : total_area radius sector_angle_degrees = 36 * real.sqrt 3 + 12 * pi :=
by
  have h1 : triangle_area radius sector_angle_degrees = 18 * real.sqrt 3,
  {
    sorry
  },
  have h2 : sector_area radius sector_angle_degrees = 6 * pi,
  {
    sorry
  },
  have h3 : total_area radius sector_angle_degrees = 2 * (18 * real.sqrt 3) + 2 * (6 * pi),
  {
    sorry
  },
  rw [h1, h2],
  norm_num,
  ring

end shaded_area_l333_333093


namespace arithmetic_sequence_general_term_and_max_sum_l333_333791

theorem arithmetic_sequence_general_term_and_max_sum (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) 
(a2_eq : a_n 2 = 1) (a5_eq : a_n 5 = -5) :
  (∀ n, a_n n = -2 * n + 5) ∧ (∀ n, S_n n = n * 3 + (n * (n - 1)) / 2 * (-2)) ∧ 
  (S_n = λ n, -n^2 + 4 * n) ∧ (argmax_n = 2 ∧ max_S_n = 4) :=
sorry

end arithmetic_sequence_general_term_and_max_sum_l333_333791


namespace find_dot_product_magnitude_l333_333455

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def vector_cross (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2 * w.3 - v.3 * w.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2 - v.2 * w.1)

def vector_dot (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def abs (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

theorem find_dot_product_magnitude (c d : ℝ × ℝ × ℝ)
  (h1 : vector_magnitude c = 3)
  (h2 : vector_magnitude d = 4)
  (h3 : vector_magnitude (vector_cross c d) = 6) :
  abs (vector_dot c d) = 6 * real.sqrt 3 :=
by
  sorry -- proof omitted

end find_dot_product_magnitude_l333_333455


namespace sum_999_is_1998_l333_333536

theorem sum_999_is_1998 : 999 + 999 = 1998 :=
by
  sorry

end sum_999_is_1998_l333_333536


namespace evaluate_square_of_sum_l333_333380

theorem evaluate_square_of_sum (x y : ℕ) (h1 : x + y = 20) (h2 : 2 * x + y = 27) : (x + y) ^ 2 = 400 :=
by
  sorry

end evaluate_square_of_sum_l333_333380


namespace find_multiple_of_b_l333_333825

-- Define the conditions
variables (a b : ℝ) (x : ℝ)
hypothesis h1 : a = 4 * b
hypothesis h2 : (a - x * b) / (2 * a - b) = 0.14285714285714285

-- Define the theorem to prove
theorem find_multiple_of_b (h1 : a = 4 * b) (h2 : (a - x * b) / (2 * a - b) = 0.14285714285714285) : x = 3 := 
by
  sorry

end find_multiple_of_b_l333_333825


namespace mitzi_money_left_l333_333142

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end mitzi_money_left_l333_333142


namespace factorial_expression_value_l333_333598

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333598


namespace factorial_expression_value_l333_333599

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333599


namespace region_area_correct_l333_333331

noncomputable def region_area : ℝ :=
  { (x, y) | x ≥ 0 ∧ y ≥ 0 ∧ 50 * fractional_part x ≥ 2 * floor x + floor y }.to_finset.measure ℝ

theorem region_area_correct : region_area = 1 :=
by
  sorry

end region_area_correct_l333_333331


namespace projection_magnitude_eq_sqrt_three_l333_333811

noncomputable def a : EuclideanSpace ℝ (Fin 2) := ![1, Real.sqrt 3]
noncomputable def b : EuclideanSpace ℝ (Fin 2) := ![Real.sqrt 3, 1]

theorem projection_magnitude_eq_sqrt_three :
  Real.sqrt (((a ⬝ b) / (∥b∥^2))^2 * ∥b∥^2) = Real.sqrt 3 :=
by
  sorry

end projection_magnitude_eq_sqrt_three_l333_333811


namespace probability_of_region_C_l333_333250

theorem probability_of_region_C :
  let a := 5 / 12;
  let b := 1 / 6;
  let x := (1 - (a + b)) / 3 in
  x = 5 / 36 :=
by {
  sorry
}

end probability_of_region_C_l333_333250


namespace college_selection_problem_l333_333683

theorem college_selection_problem (A B : Type) (hA : fintype A) (hB : fintype B) 
  (h_eq : fintype.card A = 2) (h_eqB : fintype.card B = 4) : 
  fintype.card {S : finset (A ⊕ B) // S.card = 3 ∧ (∃ a ∈ S, a ∈ A)} + 
  fintype.card {S : finset B // S.card = 3} = 16 :=
by sorry

end college_selection_problem_l333_333683


namespace matthew_hotdogs_needed_l333_333136

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end matthew_hotdogs_needed_l333_333136


namespace compute_sqrt_factorial_square_l333_333628

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333628


namespace tripling_base_exponent_l333_333890

variables (a b x : ℝ)

theorem tripling_base_exponent (b_ne_zero : b ≠ 0) (r_def : (3 * a)^(3 * b) = a^b * x^b) : x = 27 * a^2 :=
by
  -- Proof omitted as requested
  sorry

end tripling_base_exponent_l333_333890


namespace sarah_total_shampoo_conditioner_in_two_weeks_l333_333160

def total_volume_shampoo_conditioner (shampoo_daily conditioner_multiple days : ℕ) : ℕ :=
  let shampoo_total := shampoo_daily * days
  let conditioner_total := (conditioner_multiple * shampoo_daily) * days
  shampoo_total + conditioner_total

theorem sarah_total_shampoo_conditioner_in_two_weeks :
  total_volume_shampoo_conditioner 1 0.5 14 = 21 :=
  by sorry

end sarah_total_shampoo_conditioner_in_two_weeks_l333_333160


namespace distinct_integers_count_l333_333452

def floor_div (x y : ℝ) : ℤ := Int.floor (x / y)

def a_k (k : ℕ) : ℤ :=
  floor_div 2009 k

def num_distinct_a_k : ℕ :=
  (Finset.image a_k (Finset.range 1 101)).card

theorem distinct_integers_count :
  num_distinct_a_k = 69 := 
sorry

end distinct_integers_count_l333_333452


namespace part1_part2_l333_333243

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x + 2) * exp x

theorem part1 (x : ℝ) (hx : 0 < x) : (x - 2) * exp x + x + 2 > 0 :=
sorry

noncomputable def g (x a : ℝ) : ℝ := (exp x - a * x - a) / x^2

noncomputable def h (a : ℝ) : ℝ := sorry -- (requires minimization solution steps)

theorem part2 (a : ℝ) (ha : 0 ≤ a ∧ a < 1) :
  ∃ x, 0 < x ∧ ∀ y > 0, g y a ≥ g x a ∧ h a = g x a ∧ h a ∈ (1/2, exp 2 / 4] :=
sorry

end part1_part2_l333_333243


namespace sqrt_factorial_mul_squared_l333_333606

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333606


namespace greatest_root_of_gx_l333_333335

theorem greatest_root_of_gx :
  ∃ x : ℝ, (10 * x^4 - 16 * x^2 + 3 = 0) ∧ (∀ y : ℝ, (10 * y^4 - 16 * y^2 + 3 = 0) → x ≥ y) ∧ x = Real.sqrt (3 / 5) := 
sorry

end greatest_root_of_gx_l333_333335


namespace orthocenter_of_triangle_is_vertex_l333_333523

-- Definitions based on conditions
def parabola (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 4 * p.2

def line_l (p : ℝ × ℝ) : Prop := p.2 = -3

def point_O (t1 t2 : ℝ) : ℝ × ℝ := (t1 + t2, -2)

def is_tangent (O A B : ℝ × ℝ) : Prop := 
  O.1 = A.1 + B.1 ∧
  O.2 = -2

def is_orthocenter (O A B C : ℝ × ℝ) : Prop := sorry -- orthocenter definition can be complex

-- Main statement based on the proof problem
theorem orthocenter_of_triangle_is_vertex (t1 t2 : ℝ) (h1 : t1 * t2 = -2) :
  ∃ (A B O : ℝ × ℝ),
  parabola A ∧ parabola B ∧ 
  point_O t1 t2 = O ∧
  is_tangent O A B ∧
  is_orthocenter (0, 0) A B O :=
begin
  sorry
end

end orthocenter_of_triangle_is_vertex_l333_333523


namespace evaluate_101_times_101_l333_333746

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end evaluate_101_times_101_l333_333746


namespace measure_angle_ABG_l333_333488

-- Formalizing the conditions
def is_regular_octagon (polygon : Fin 8 → ℝ × ℝ) : Prop :=
  let vertices := [polygon 0, polygon 1, polygon 2, polygon 3, polygon 4, polygon 5, polygon 6, polygon 7]
  (∀ i, ∥vertices ((i + 1) % 8) - vertices i∥ = ∥vertices 1 - vertices 0∥) ∧ 
  (∀ i, ∠ (vertices (i + 1) % 8) (vertices i) (vertices (i - 1 + 8) % 8) = 135)

-- Define angle_measure, considering the numbering polygon Fin 8 from 0 to 7
def angle_measure_polygon (polygon : Fin 8 → ℝ × ℝ) (i j k : Fin 8) : ℝ :=
  ∠ (polygon j) (polygon i) (polygon k)

-- The proof problem statement
theorem measure_angle_ABG (polygon : Fin 8 → ℝ × ℝ) (h : is_regular_octagon polygon) : 
  angle_measure_polygon polygon 0 1 6 = 22.5 :=
sorry

end measure_angle_ABG_l333_333488


namespace abs_diff_31st_terms_l333_333550

/-- Sequence C is an arithmetic sequence with a starting term 100 and a common difference 15. --/
def seqC (n : ℕ) : ℤ :=
  100 + 15 * (n - 1)

/-- Sequence D is an arithmetic sequence with a starting term 100 and a common difference -20. --/
def seqD (n : ℕ) : ℤ :=
  100 - 20 * (n - 1)

/-- Absolute value of the difference between the 31st terms of sequences C and D is 1050. --/
theorem abs_diff_31st_terms : |seqC 31 - seqD 31| = 1050 := by
  sorry

end abs_diff_31st_terms_l333_333550


namespace amount_p_l333_333237

variable (P : ℚ)

/-- p has $42 more than what q and r together would have had if both q and r had 1/8 of what p has.
    We need to prove that P = 56. -/
theorem amount_p (h : P = (1/8 : ℚ) * P + (1/8) * P + 42) : P = 56 :=
by
  sorry

end amount_p_l333_333237


namespace camel_path_divisible_by_3_l333_333673

theorem camel_path_divisible_by_3 (n : ℕ) (h : camel path takes n steps and returns to starting vertex) :
  n % 3 = 0 :=
sorry

end camel_path_divisible_by_3_l333_333673


namespace math_majors_consecutive_probability_l333_333742

-- Defining the setup and requirements of the problem
variables (total_people math_majors physics_majors chemistry_majors : ℕ)
variables (total_ways consecutive_ways : ℕ)

-- Stating the conditions
def conditions := total_people = 11 ∧ math_majors = 5 ∧ physics_majors = 3 ∧ chemistry_majors = 3
def total_ways := 10.factorial
def consecutive_ways := 7 * (5.factorial)

-- Stating the probability result
def probability_math_majors_consecutive := consecutive_ways / total_ways

-- The theorem to prove
theorem math_majors_consecutive_probability :
  conditions → probability_math_majors_consecutive = 1 / 4320 :=
by
  intros h
  rw [conditions] at h
  cases h with h_total h_math_phys_chem
  rw [h_total, h_math_phys_chem.left, h_math_phys_chem.right.left, h_math_phys_chem.right.right]
  sorry

end math_majors_consecutive_probability_l333_333742


namespace complex_modulus_sum_correct_l333_333745

noncomputable def complex_modulus_sum : ℝ :=
  complex.abs ⟨3, -5⟩ + complex.abs ⟨3, 5⟩ + complex.abs ⟨1, 5⟩

theorem complex_modulus_sum_correct :
  complex_modulus_sum = 2 * Real.sqrt 34 + Real.sqrt 26 :=
by
  sorry

end complex_modulus_sum_correct_l333_333745


namespace largest_digit_2n_eq_5n_is_3_l333_333367

def largest_common_digit (n : ℕ) :=
  let a := 3  -- The largest digit satisfying the conditions is found to be 3
  in a^2 < 10 ∧ 10 < (a + 1)^2

theorem largest_digit_2n_eq_5n_is_3 (n : ℕ) (h1 : largest_common_digit n)
  (h2 : ∀ m : ℕ, (m^2 < 10 ∧ 10 < (m+1)^2 → m = 3)) : 
  ∃ a, a = 3 ∧ largest_common_digit n :=
by 
  use 3
  split
  { refl }
  { exact h1 }

end largest_digit_2n_eq_5n_is_3_l333_333367


namespace line_intersects_circle_l333_333190

theorem line_intersects_circle (k : ℝ) (h1 : k = 2) (radius : ℝ) (center_distance : ℝ) (eq_roots : ∀ x, x^2 - k * x + 1 = 0) :
  radius = 5 → center_distance = k → k < radius :=
by
  intros hradius hdistance
  have h_root_eq : k = 2 := h1
  have h_rad : radius = 5 := hradius
  have h_dist : center_distance = k := hdistance
  have kval : k = 2 := h1
  simp [kval, hradius, hdistance, h_rad, h_dist]
  sorry

end line_intersects_circle_l333_333190


namespace convex_1976_polyhedron_non_zero_vector_sum_l333_333737

theorem convex_1976_polyhedron_non_zero_vector_sum :
  ∃ P : Polyhedron, convex P ∧ P.faces = 1976 ∧
    ∀ arrows : P.edges → Vector, (∑ e : P.edges, arrows e) ≠ 0 := 
sorry

end convex_1976_polyhedron_non_zero_vector_sum_l333_333737


namespace min_max_change_eq_10_l333_333296

theorem min_max_change_eq_10 :
    let initial_yes := 30
    let initial_no := 40
    let initial_undecided := 30
    let final_yes := 50
    let final_no := 20
    let final_undecided := 30
    let total_students := 100
    let min_change := initial_no - final_no -- Minimally shifting students from No to Yes
    let max_change := (total_students - initial_undecided - final_yes - final_no) + (initial_yes - (final_yes - min_change))
    let x_min := (min_change * 100) / total_students
    let x_max := (max_change * 100) / total_students
    x := x_max - x_min in
    x = 10
:= 
by
sorrry

end min_max_change_eq_10_l333_333296


namespace s1_lt_s2_l333_333448

-- Definitions of points and distances to represent conditions
variables {A B C O : Type} 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space O]

-- Assuming metric space distances corresponding to line segments
variable (dist : A → B → ℝ)

-- Definitions corresponding to conditions
def is_centroid (O : A) (A B C : A) : Prop :=
  ∃ M N P : A, -- Medians meeting at centroid O
  dist A M = dist B M ∧ dist B N = dist C N ∧ dist C P = dist A P ∧
  dist O M = 2 / 3 * dist A M ∧
  dist O N = 2 / 3 * dist B N ∧
  dist O P = 2 / 3 * dist C P

def s1 (O A B C : A) : ℝ := 
  dist O A + dist O B + dist O C

def s2 (A B C : A) : ℝ := 
  3 * (dist A B + dist B C + dist C A)

-- The main theorem statement
theorem s1_lt_s2 (O A B C : A) (h : is_centroid O A B C) :
  s1 O A B C < s2 A B C := 
sorry

end s1_lt_s2_l333_333448


namespace amy_files_count_l333_333714

theorem amy_files_count : 
  let initial_music_files := 26 in
  let initial_video_files := 36 in
  let deleted_files := 48 in
  let new_downloaded_files := 15 in
  let initial_total_files := initial_music_files + initial_video_files in
  let files_after_deletion := initial_total_files - deleted_files in
  let final_total_files := files_after_deletion + new_downloaded_files in
  final_total_files = 29 :=
by
  norm_num

end amy_files_count_l333_333714


namespace concrete_order_l333_333276

-- Definitions of the given conditions
def width_feet := 4
def length_feet := 80
def thickness_inches := 4

-- Conversion factors
def feet_to_yards : Float := 1 / 3
def inches_to_yards : Float := 1 / 36

-- Converting dimensions to yards
def width_yards := width_feet.toFloat * feet_to_yards
def length_yards := length_feet.toFloat * feet_to_yards
def thickness_yards := thickness_inches.toFloat * inches_to_yards

-- Calculating the volume in cubic yards
def volume_yards := width_yards * length_yards * thickness_yards

-- Function to compute the required cubic yards of concrete rounded up
def required_cubic_yards (v : Float) : Int :=
  if v.ceil.toInt > v.toInt then v.ceil.toInt else v.toInt

theorem concrete_order : required_cubic_yards volume_yards = 4 :=
  by
    have h : volume_yards = (320.0 / 81.0)
    -- further arithmetic steps showing volume_yards calculations
    -- ...
    /-
    Calculations would show that (320.0 / 81.0) ≈ 3.9506, so 
    required_cubic_yards(volume_yards) = 4
    -/
    sorry  -- Assume correctness of volume_yards calculations.

end concrete_order_l333_333276


namespace series_converges_uniformly_l333_333467

noncomputable def xi (ω : Ω) (n : ℕ) : ℝ := sorry -- Suppose xi is defined as needed

theorem series_converges_uniformly (h_iid : ∀ n, i.i.d. (xi ω n)) (h_vals : ∀ n, xi ω n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
(h_prob : ∀ n, prob {ω | xi ω n = k} = 1 / 10):
  ∃ X, (∀ ω, (∀ ε > 0, ∃ N, ∀ n ≥ N, |(∑ i in finset.range n, xi ω i / (10 ^ i)) - X ω| < ε)) ∧ (uniform_distribution X [0, 1]) :=
begin
  sorry
end

end series_converges_uniformly_l333_333467


namespace op_comm_op_not_assoc_calc_op_2_3_l333_333764

def op (x y : ℝ) : ℝ :=
  (x * y + x + y) / (x + y + 1)

theorem op_comm (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  op x y = op y x := by
  sorry

theorem op_not_assoc (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (op (op x y) z) ≠ (op x (op y z)) := by
  sorry

theorem calc_op_2_3 : 
  op 2 3 = 11 / 6 := by
  sorry

end op_comm_op_not_assoc_calc_op_2_3_l333_333764


namespace measure_angle_ABG_l333_333489

-- Formalizing the conditions
def is_regular_octagon (polygon : Fin 8 → ℝ × ℝ) : Prop :=
  let vertices := [polygon 0, polygon 1, polygon 2, polygon 3, polygon 4, polygon 5, polygon 6, polygon 7]
  (∀ i, ∥vertices ((i + 1) % 8) - vertices i∥ = ∥vertices 1 - vertices 0∥) ∧ 
  (∀ i, ∠ (vertices (i + 1) % 8) (vertices i) (vertices (i - 1 + 8) % 8) = 135)

-- Define angle_measure, considering the numbering polygon Fin 8 from 0 to 7
def angle_measure_polygon (polygon : Fin 8 → ℝ × ℝ) (i j k : Fin 8) : ℝ :=
  ∠ (polygon j) (polygon i) (polygon k)

-- The proof problem statement
theorem measure_angle_ABG (polygon : Fin 8 → ℝ × ℝ) (h : is_regular_octagon polygon) : 
  angle_measure_polygon polygon 0 1 6 = 22.5 :=
sorry

end measure_angle_ABG_l333_333489


namespace average_minutes_run_per_day_l333_333719

theorem average_minutes_run_per_day 
  (f : ℕ)
  (third_graders : ℕ := 3 * f)
  (fourth_graders : ℕ := (3 / 2) * f)
  (fifth_graders : ℕ := f)
  (third_grade_minutes : ℝ := 14)
  (fourth_grade_minutes : ℝ := 17)
  (fifth_grade_minutes : ℝ := 12)
  (total_third_grade_minutes := third_grade_minutes * third_graders : ℝ)
  (total_fourth_grade_minutes := fourth_grade_minutes * fourth_graders : ℝ)
  (total_fifth_grade_minutes := fifth_grade_minutes * fifth_graders : ℝ)
  (total_minutes_run := total_third_grade_minutes + total_fourth_grade_minutes + total_fifth_grade_minutes : ℝ)
  (total_students := third_graders + fourth_graders + fifth_graders : ℝ) :
  (total_minutes_run / total_students = 159 / 11) := sorry

end average_minutes_run_per_day_l333_333719


namespace tan_alpha_equals_one_l333_333412

theorem tan_alpha_equals_one {α β : ℝ}
  (h0 : 0 < α ∧ α < π/2)
  (h1 : 0 < β ∧ β < π/2)
  (h2 : cos (α + β) = sin (α - β)) :
  tan α = 1 :=
by
  sorry

end tan_alpha_equals_one_l333_333412


namespace sum_of_roots_l333_333821

theorem sum_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -3) (hx1 : x1 + x2 = 2) :
  x1 + x2 = 2 :=
by {
  sorry
}

end sum_of_roots_l333_333821


namespace binom_inequality1_binom_inequality2_l333_333768

theorem binom_inequality1 (n : ℕ) (h : n ≥ 82) : choose (2 * n) n < 4^(n - 2) :=
sorry

theorem binom_inequality2 (n : ℕ) (h : n ≥ 1305) : choose (2 * n) n < 4^(n - 3) :=
sorry

end binom_inequality1_binom_inequality2_l333_333768


namespace solve_firm_problem_l333_333997

def firm_problem : Prop :=
  ∃ (P A : ℕ), 
    (P / A = 2 / 63) ∧ 
    (P / (A + 50) = 1 / 34) ∧ 
    (P = 20)

theorem solve_firm_problem : firm_problem :=
  sorry

end solve_firm_problem_l333_333997


namespace intersection_point_exists_l333_333507

def h : ℝ → ℝ := sorry  -- placeholder for the function h
def j : ℝ → ℝ := sorry  -- placeholder for the function j

-- Conditions
axiom h_3_eq : h 3 = 3
axiom j_3_eq : j 3 = 3
axiom h_6_eq : h 6 = 9
axiom j_6_eq : j 6 = 9
axiom h_9_eq : h 9 = 18
axiom j_9_eq : j 9 = 18

-- Theorem
theorem intersection_point_exists :
  ∃ a b : ℝ, a = 2 ∧ h (3 * a) = 3 * j (a) ∧ h (3 * a) = b ∧ 3 * j (a) = b ∧ a + b = 11 :=
  sorry

end intersection_point_exists_l333_333507


namespace total_discount_is_52_5_percent_l333_333703

def original_price := 1
def sale_price := (2 / 3) * original_price
def price_after_coupon := 0.75 * sale_price
def final_price := 0.95 * price_after_coupon

theorem total_discount_is_52_5_percent :
  1 - final_price = 0.525 :=
by
  unfold original_price sale_price price_after_coupon final_price
  sorry

end total_discount_is_52_5_percent_l333_333703


namespace max_cubes_l333_333222

variables (length width height cube_volume num_cubes_fit : ℕ)

def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def max_cubes_fit (box_volume cube_volume : ℕ) : ℕ :=
  box_volume / cube_volume

axiom box_dimensions : length = 9 ∧ width = 8 ∧ height = 12
axiom cube_spec : cube_volume = 27
axiom cube_count_constraint : num_cubes_fit = 24

theorem max_cubes : max_cubes_fit (box_volume length width height) cube_volume = num_cubes_fit :=
by {
  cases box_dimensions with h_length h_width_height,
  cases h_width_height with h_width h_height,
  rw [h_length, h_width, h_height],
  rw cube_spec,
  rw cube_count_constraint,
  sorry
}

end max_cubes_l333_333222


namespace constant_term_expansion_l333_333220

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_expansion :
  let sqr_x := (x ^ (1 / 2 : ℚ)) 
  let inv_x := (3 * (x ^ (-1 : ℚ)))
  let expr := (sqr_x + inv_x) ^ 12
  (binom 12 4) * 3^4 = 40095 := 
by {
  let x := x -- treating x as a variable in the expression
  sorry
}

end constant_term_expansion_l333_333220


namespace neznaika_cut_l333_333145

-- Definitions according to the conditions
def cell : Type := (ℕ × ℕ)
def grid : Type := fin 5 × fin 5
def is_adjacent (a b : cell) : Prop := (a.fst = b.fst ∧ (a.snd = b.snd + 1 ∨ a.snd + 1 = b.snd)) ∨ 
                                       (a.snd = b.snd ∧ (a.fst = b.fst + 1 ∨ a.fst + 1 = b.fst))

-- A shape is contiguous if all cells are connected
def is_contiguous (shape : set cell) : Prop :=
  ∀ (a ∈ shape) (b ∈ shape), ∃ (p : list cell), p.head = some a ∧ p.last = some b ∧ ∀ (x y ∈ p), is_adjacent x y

-- A valid cut configuration
def valid_cut (cuts : list (cell × cell)) (shapes : list (set cell)) : Prop :=
  ∀ cut ∈ cuts, cut.fst ∈ grid ∧ cut.snd ∈ grid ∧ is_adjacent cut.fst cut.snd ∧
  (∀ shape ∈ shapes, is_contiguous shape) ∧
  list.length shapes = 8

-- Given the conditions, prove that there exists a valid cut
theorem neznaika_cut :
  ∃ (cuts : list (cell × cell)) (shapes : list (set cell)), valid_cut cuts shapes :=
by
  sorry

end neznaika_cut_l333_333145


namespace stickers_per_student_l333_333472

theorem stickers_per_student (G S B N: ℕ) (hG: G = 50) (hS: S = 2 * G) (hB: B = S - 20) (hN: N = 5) : 
  (G + S + B) / N = 46 := by
  sorry

end stickers_per_student_l333_333472


namespace Black_Queen_thought_Black_King_asleep_l333_333987

theorem Black_Queen_thought_Black_King_asleep (BK_awake : Prop) (BQ_awake : Prop) :
  (∃ t : ℕ, t = 10 * 60 + 55 → 
  ∀ (BK : Prop) (BQ : Prop),
    ((BK_awake ↔ ¬BK) ∧ (BQ_awake ↔ ¬BQ)) ∧
    (BK → BQ → BQ_awake) ∧
    (¬BK → ¬BQ → BK_awake)) →
  ((BQ ↔ BK) ∧ (BQ_awake ↔ ¬BQ)) →
  (∃ (BQ_thought : Prop), BQ_thought ↔ BK) := 
sorry

end Black_Queen_thought_Black_King_asleep_l333_333987


namespace probability_of_product_divisible_by_8_l333_333982

noncomputable def dice_probability_is_divisible_by_8 : Prop :=
  let is_standard_die (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}
  let roll_dice (n : ℕ) : ℕ := n % 6 + 1
  let dice_product_d8 := (list.map roll_dice (list.range 8)).foldl (*) 1
  classical.some (exists_decidable (dice_product_d8 % 8 = 0)) = (1697 / 1728)

theorem probability_of_product_divisible_by_8 :
  dice_probability_is_divisible_by_8 :=
sorry

end probability_of_product_divisible_by_8_l333_333982


namespace even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l333_333882

theorem even_product_implies_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 = 0 → ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

theorem odd_product_implies_no_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 ≠ 0 → ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

end even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l333_333882


namespace lem_intersection_l333_333944

noncomputable def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }
def line_through_focus (p : ℝ × ℝ) : Prop := ∃ m : ℚ, m = 4 / 3 ∧ ∃ (x y : ℝ), p = (x, y) ∧ y = m * (x - 1)
def focus : ℝ × ℝ := (1, 0)
def intersects_parabola (p1 p2 : ℝ × ℝ) : Prop := p1 ∈ parabola ∧ p2 ∈ parabola ∧ ∃ m : ℚ, m = 4 / 3 ∧ line_through_focus p1 ∧ line_through_focus p2
def lam_equal_four (p1 p2 : ℝ × ℝ) (λ : ℝ) : Prop := λ > 1 ∧ (p1.1 - focus.1) = λ * (focus.1 - p2.1) ∧ (p1.2 - focus.2) = λ * (focus.2 - p2.2) ∧ λ = 4

theorem lem_intersection (p1 p2 : ℝ × ℝ) (λ : ℝ) :
  intersects_parabola p1 p2 ∧ lam_equal_four p1 p2 λ → λ = 4 := by
  sorry

end lem_intersection_l333_333944


namespace problem_solution_final_solution_count_l333_333757

def equation_solutions_count : ℕ :=
  ∃ (x y : ℤ), 2^(2*x) - 5^(2*y) = 75

theorem problem_solution : ∃ (x y : ℤ), 2^(2*x) - 5^(2*y) = 75 :=
by sorry

theorem final_solution_count : {s // equation_solutions_count s} = 1 :=
by sorry

end problem_solution_final_solution_count_l333_333757


namespace JohnIncome_l333_333102

variable (J : ℝ)
axiom JohnTaxRate : ℝ
axiom IngridTaxRate : ℝ
axiom IngridIncome : ℝ
axiom CombinedTaxRate : ℝ

-- Conditions
def JohnTax := JohnTaxRate * J
def IngridTax := IngridTaxRate * IngridIncome
def CombinedTax := CombinedTaxRate * (J + IngridIncome)

-- Hypothesis
axiom h_combined_tax : JohnTax + IngridTax = CombinedTax

-- The proof problem
theorem JohnIncome :
  J = 57936.462 := by
  sorry

-- Specific values for the conditions
def JohnTaxRate := 0.30
def IngridTaxRate := 0.40
def IngridIncome := 72000
def CombinedTaxRate := 0.3554

#eval JohnIncome

end JohnIncome_l333_333102


namespace exists_three_digit_primes_l333_333660

theorem exists_three_digit_primes : ∃ (S : Finset ℕ), 
  (S.card ≥ 1 ∧ S.card ≤ 10) 
  ∧ ∀ n ∈ S, Nat.Prime n 
  ∧ ∀ n ∈ S, 100 ≤ n ∧ n ≤ 999 := 
by
  sorry

end exists_three_digit_primes_l333_333660


namespace hyperbola_expression_l333_333330

theorem hyperbola_expression (P : ℝ × ℝ) (k : ℝ) 
  (h1 : P.y = 2) 
  (h2 : P.y = (1/2) * P.x + 1)
  (h3 : P.y = k / P.x) : k = 4 :=
by sorry

end hyperbola_expression_l333_333330


namespace next_consecutive_time_l333_333680

theorem next_consecutive_time (current_hour : ℕ) (current_minute : ℕ) 
  (valid_minutes : 0 ≤ current_minute ∧ current_minute < 60) 
  (valid_hours : 0 ≤ current_hour ∧ current_hour < 24) : 
  current_hour = 4 ∧ current_minute = 56 →
  ∃ next_hour next_minute : ℕ, 
    (0 ≤ next_minute ∧ next_minute < 60) ∧ 
    (0 ≤ next_hour ∧ next_hour < 24) ∧
    (next_hour, next_minute) = (12, 34) ∧ 
    (next_hour * 60 + next_minute) - (current_hour * 60 + current_minute) = 458 := 
by sorry

end next_consecutive_time_l333_333680


namespace sqrt_factorial_product_squared_l333_333578

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333578


namespace ratio_O_n_E_n_l333_333657

def S_n (n : ℕ) (xs : list ℕ) : ℕ :=
(xs.chunk 2).map (λ g, g.head! * g.get! 1).sum

def O_n (n : ℕ) : ℕ :=
(list.replicate (2*n) [0, 1]).product.filter (λ xs, S_n n xs % 2 = 1).length

def E_n (n : ℕ) : ℕ :=
(list.replicate (2*n) [0, 1]).product.filter (λ xs, S_n n xs % 2 = 0).length

theorem ratio_O_n_E_n (n : ℕ) (hn : n > 0) :
  (O_n n : ℝ) / (E_n n : ℝ) = (2^n - 1) / (2^n + 1) :=
sorry

end ratio_O_n_E_n_l333_333657


namespace russian_pairing_probability_l333_333929

-- Definitions based on conditions
def total_players : ℕ := 10
def russian_players : ℕ := 4
def non_russian_players : ℕ := total_players - russian_players

-- Probability calculation as a hypothesis
noncomputable def pairing_probability (rs: ℕ) (ns: ℕ) : ℚ :=
  (rs * (rs - 1)) / (total_players * (total_players - 1))

theorem russian_pairing_probability :
  pairing_probability russian_players non_russian_players = 1 / 21 :=
sorry

end russian_pairing_probability_l333_333929


namespace evaluate_six_fold_l333_333879

def g (x : ℕ) : ℕ := x^2 - 6 * x + 8

theorem evaluate_six_fold : g(g(g(g(g(g(2)))))) = 36468425032 := by
  sorry

end evaluate_six_fold_l333_333879


namespace sqrt_factorial_product_squared_l333_333582

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333582


namespace problem_statement_l333_333176

-- Mathematical Conditions
variables (a : ℝ)

-- Sufficient but not necessary condition proof statement
def sufficient_but_not_necessary : Prop :=
  (∀ a : ℝ, a > 0 → a^2 + a ≥ 0) ∧ ¬(∀ a : ℝ, a^2 + a ≥ 0 → a > 0)

-- Main problem to be proved
theorem problem_statement : sufficient_but_not_necessary :=
by
  sorry

end problem_statement_l333_333176


namespace find_certain_number_l333_333561

theorem find_certain_number (x : ℝ) : 136 - 0.35 * x = 31 -> x = 300 :=
by
  intro h
  sorry

end find_certain_number_l333_333561


namespace triangular_difference_30_28_l333_333649

noncomputable def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_difference_30_28 : triangular 30 - triangular 28 = 59 :=
by
  sorry

end triangular_difference_30_28_l333_333649


namespace pamTotalApples_l333_333148

-- Define the given conditions
def applesPerGeraldBag : Nat := 40
def applesPerPamBag := 3 * applesPerGeraldBag
def pamBags : Nat := 10

-- Statement to prove
theorem pamTotalApples : pamBags * applesPerPamBag = 1200 :=
by
  sorry

end pamTotalApples_l333_333148


namespace least_value_in_set_T_l333_333112

theorem least_value_in_set_T :
  ∃ (T : Set ℕ), T ⊆ {n | 1 ≤ n ∧ n ≤ 18} ∧ ∀ x ∈ T, ¬ Prime x ∧ ∀ a b ∈ T, a < b → ¬ (b % a = 0) ∧ T.card = 8 ∧ ∀ c ∈ T, c ≥ 4 ∧ (4 ∈ T) := 
sorry

end least_value_in_set_T_l333_333112


namespace monotonicity_interval_extreme_value_intervals_product_of_extreme_values_gt_e_l333_333803

open Real

def f (x : ℝ) : ℝ := 2 * log x - x - 1

theorem monotonicity_interval (h : (∀ x, f' x = (2 - x) / x)) : 
  (∀ x ∈ set.Ioo 0 2, f' x > 0) ∧ (∀ x ∈ set.Ici 2, f' x < 0) :=
sorry

def b (x m : ℝ) : ℝ := 2 * log x - m * x - 1

def g (x m : ℝ) : ℝ := x * b x m

theorem extreme_value_intervals (D : set ℝ) (hD : D = set.Ioo (exp (-1/2)) (exp (3/2))) :
  (∀ m, g' x m = (2 * log x - 2 * m * x + 1)) → 
  (∃ x₁ x₂ ∈ D, g' x₁ m = 0 ∧ g' x₂ m = 0) ↔ (m ∈ Ioo (2 * exp (-3/2)) (3 / (2 * exp 1))) :=
sorry

theorem product_of_extreme_values_gt_e (x₁ x₂ : ℝ) (D : set ℝ)
  (hx₁ : x₁ ∈ D) (hx₂ : x₂ ∈ D) (hD : D = set.Ioo (exp (-1/2)) (exp (3/2))) :
  g' x₁ m = 0 → g' x₂ m = 0 → (x₁ * x₂ > exp 1) :=
sorry

end monotonicity_interval_extreme_value_intervals_product_of_extreme_values_gt_e_l333_333803


namespace maximum_of_fraction_l333_333057

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end maximum_of_fraction_l333_333057


namespace sqrt_factorial_mul_squared_l333_333603

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333603


namespace problem1_problem2_l333_333895

open Nat

def seq (a : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → a n < a (n + 1) ∧ a n > 0

def b_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n)

def c_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n + 1)

theorem problem1 (a : ℕ → ℕ) (h_seq : seq a) (h_bseq : ∀ n, n > 0 → b_seq a n = 3 * n) : a 1 = 2 ∧ c_seq a 1 = 6 :=
  sorry

theorem problem2 (a : ℕ → ℕ) (h_seq : seq a) (h_cseq : ∀ n, n > 0 → c_seq a (n + 1) - c_seq a n = 1) : 
  ∀ n, n > 0 → a (n + 1) - a n = 1 :=
  sorry

end problem1_problem2_l333_333895


namespace simplify_expr_l333_333163

variable (a b c : ℤ)

theorem simplify_expr :
  (15 * a + 45 * b + 20 * c) + (25 * a - 35 * b - 10 * c) - (10 * a + 55 * b + 30 * c) = 30 * a - 45 * b - 20 * c := 
by
  sorry

end simplify_expr_l333_333163


namespace power_function_monotonic_incr_l333_333072

theorem power_function_monotonic_incr (m : ℝ) (h₁ : m^2 - 5 * m + 7 = 1) (h₂ : m^2 - 6 > 0) : m = 3 := 
by
  sorry

end power_function_monotonic_incr_l333_333072


namespace geometric_series_common_ratio_l333_333291

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by
  rw [ha, hS] at hS_eq
  -- This statement follows from algebraic manipulation outlined in the solution steps.
  sorry

end geometric_series_common_ratio_l333_333291


namespace increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l333_333355

variables {k: ℝ} -- Thermal conductivity
variables {A A': ℝ} -- Original and increased contact area
variables {dT: ℝ} -- Temperature difference
variables {dx: ℝ} -- Thickness of the skillet handle

-- Define the heat flow rate according to Fourier's law of heat conduction
def heat_flow_rate (k: ℝ) (A: ℝ) (dT: ℝ) (dx: ℝ) : ℝ :=
  -k * A * (dT / dx)

theorem increased_contact_area_increases_heat_flow 
  (h₁: A' > A) -- Increased contact area
  (h₂: dT / dx > 0) -- Positive temperature gradient
  : heat_flow_rate k A' dT dx > heat_flow_rate k A dT dx :=
by
  -- Proof to show that increased area increases heat flow rate
  sorry

theorem handle_felt_hotter_no_thermodynamic_contradiction 
  (h₁: A' > A)
  (h₂: dT / dx > 0)
  : ¬(heat_flow_rate k A' dT dx contradicts thermodynamic laws) :=
by
  -- Proof to show no contradiction with the laws of thermodynamics
  sorry

end increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l333_333355


namespace geometric_sequence_sum_squared_l333_333387

theorem geometric_sequence_sum_squared (a : ℕ → ℕ) (n : ℕ) (q : ℕ) 
    (h_geometric: ∀ n, a (n + 1) = a n * q)
    (h_a1 : a 1 = 2)
    (h_a3 : a 3 = 4) :
    (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 = 1020 :=
by
  sorry

end geometric_sequence_sum_squared_l333_333387


namespace find_distance_between_foci_l333_333204

noncomputable def distance_between_foci (pts : List (ℝ × ℝ)) : ℝ :=
  let c := (1, -1)  -- center of the ellipse
  let x1 := (1, 3)
  let x2 := (1, -5)
  let y := (7, -5)
  let b := 4       -- semi-minor axis length
  let a := 2 * Real.sqrt 13  -- semi-major axis length
  let foci_distance := 2 * Real.sqrt (a^2 - b^2)
  foci_distance

theorem find_distance_between_foci :
  distance_between_foci [(1, 3), (7, -5), (1, -5)] = 12 :=
by
  sorry

end find_distance_between_foci_l333_333204


namespace number_of_zeros_inequality_l333_333801

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * log x - a * x + 1

theorem number_of_zeros (a : ℝ) (h : a ≤ 1) : 
  (a = 1 → ∃! x, f x a = 0) ∧ (a < 1 → ¬∃ x, f x a = 0) :=
sorry

theorem inequality (x : ℝ) (a : ℝ) (h : a ≤ 1): 
  f x a + a * x + 2 / exp x > log (2 * exp 1) :=
sorry

end number_of_zeros_inequality_l333_333801


namespace honey_production_increase_l333_333103

/-- Definition stating that John has two hives -/
def john_hives : ℕ := 2

/-- The first hive has 1000 bees and produces 500 liters of honey -/
def first_hive_bees : ℕ := 1000
def first_hive_honey : ℝ := 500

/-- The second hive has 20% fewer bees and produces 2460 liters of honey -/
def second_hive_bees : ℕ := (first_hive_bees - first_hive_bees * 20 / 100).to_nat
def second_hive_honey : ℝ := 2460

/-- Honey production per bee in both hives -/
def honey_per_bee_first_hive : ℝ := first_hive_honey / first_hive_bees
def honey_per_bee_second_hive : ℝ := second_hive_honey / second_hive_bees

/-- Calculate the percentage increase in honey production per bee -/
def percentage_increase : ℝ := ((honey_per_bee_second_hive - honey_per_bee_first_hive) / honey_per_bee_first_hive) * 100

/-- Proof that the percentage increase in honey production per bee in the second hive compared to the first hive is 515%  -/
theorem honey_production_increase :
  percentage_increase = 515 :=
by
  -- Assuming all conditions (import and definitions are correct, we skip the proof)
  sorry

end honey_production_increase_l333_333103


namespace ratio_of_c_and_b_l333_333736

variable (x b c : ℝ)

-- Conditions as definitions
def quadratic := x^2 + 1500 * x + 2400
def completed_square := (x + b)^2 + c
def b_value := 750
def c_value := -560100

-- Statement only without the proof
theorem ratio_of_c_and_b : quadratic x = completed_square x b_value c_value → 
  c_value / b_value = -746.8 :=
by
  sorry

end ratio_of_c_and_b_l333_333736


namespace problem_proof_l333_333862

open Real

noncomputable def angle_B (A C : ℝ) : ℝ := π / 3

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  (1/2) * a * c * (sqrt 3 / 2)

theorem problem_proof (A B C a b c : ℝ)
  (h1 : 2 * cos A * cos C * (tan A * tan C - 1) = 1)
  (h2 : a + c = sqrt 15)
  (h3 : b = sqrt 3)
  (h4 : B = π / 3) :
  (B = angle_B A C) ∧ 
  (area_triangle a b c = sqrt 3) :=
by
  sorry

end problem_proof_l333_333862


namespace jensens_inequality_l333_333242

-- Define the context and assumptions
variables {α : Type*} [MeasurableSpace α] {μ : MeasureTheory.Measure α}
variables {𝔽 : Type*} [IsROrC 𝔽]

-- Define ξ as a random variable
noncomputable def ξ (ω : α) : 𝔽 := sorry

-- Define the σ-algebra 𝒢
variable (𝒢 : MeasurableSpace α)

-- Define g as a convex downward Borel function
noncomputable def g (x : 𝔽) : 𝔽 := sorry
axiom g_convex_downward : ConvexOn ℝ Set.univ g
axiom g_borel_measurable : Measurable g

-- Define the expected value condition
axiom integrable_g_ξ : Integrable (λ a, abs (g (ξ a))) μ

-- Theorem statement
theorem jensens_inequality 
  (ξ : α → 𝔽)
  (𝒢 : MeasurableSpace α) 
  (g : 𝔽 → 𝔽)
  [g_convex_downward : convex_on ℝ (set.univ) g]
  [g_borel_measurable : measurable g]
  (integrable_g_ξ : integrable (λ a, abs (g (ξ a))) μ) :
  (g (conditional_expectation 𝒢 ξ) ≤ conditional_expectation 𝒢 (λ a, g (ξ a))) ∧
  (g (expectation ξ) ≤ expectation (λ a, g (conditional_expectation 𝒢 ξ)) ∧
   expectation (λ a, g (conditional_expectation 𝒢 ξ)) ≤ expectation (λ a, g (ξ a))) :=
begin
  sorry,
end

end jensens_inequality_l333_333242


namespace sqrt_factorial_mul_squared_l333_333609

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333609


namespace log_expression_as_product_l333_333751

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_as_product (A m n p : ℝ) (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hA : 0 < A) :
  log m A * log n A + log n A * log p A + log p A * log m A =
  log A (m * n * p) * log p A * log n A * log m A :=
by
  sorry

end log_expression_as_product_l333_333751


namespace find_quotient_l333_333080

theorem find_quotient (divisor remainder dividend : ℕ) (h_dvd : divisor = 72) (h_rem : remainder = 64) (h_div : dividend = 2944) :
  ∃ Q : ℕ, dividend = (divisor * Q) + remainder ∧ Q = 40 :=
begin
  sorry
end

end find_quotient_l333_333080


namespace evaluate_fraction_sum_l333_333786

variable (a b c : ℝ)

theorem evaluate_fraction_sum
  (h : (a / (30 - a)) + (b / (70 - b)) + (c / (80 - c)) = 9) :
  (6 / (30 - a)) + (14 / (70 - b)) + (16 / (80 - c)) = 2.4 :=
by
  sorry

end evaluate_fraction_sum_l333_333786


namespace domain_f_l333_333516

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x / real.log 2)

theorem domain_f :
  {x : ℝ | f x ≥ 0} = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by sorry

end domain_f_l333_333516


namespace regular_octagon_angle_ABG_l333_333481

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end regular_octagon_angle_ABG_l333_333481


namespace factorial_sqrt_sq_l333_333614

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333614


namespace num_real_solutions_abs_eq_l333_333815

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end num_real_solutions_abs_eq_l333_333815


namespace sum_of_integers_square_plus_224_equals_l333_333965

theorem sum_of_integers_square_plus_224_equals (x : ℤ) :
  (x^2 = x + 224) → (({x : ℤ | x^2 = x + 224}.sum id) = 2) :=
sorry

end sum_of_integers_square_plus_224_equals_l333_333965


namespace probability_even_digits_non_adjacent_l333_333731

theorem probability_even_digits_non_adjacent :
  let digits := {0, 1, 2, 3}
  let non_div_by_10 (n : Nat) : Prop := n % 10 ≠ 0
  in (prob_two_even_digits_non_adjacent digits non_div_by_10) = 4 / 9 :=
by
  sorry

noncomputable def prob_two_even_digits_non_adjacent (digits : Set Nat) (filter_condition : Nat -> Prop) : ℚ :=
  let four_digit_numbers := {n : Nat | n ∈ permutations digits ∧ filter_condition n}
  let favorable_outcomes := {n ∈ four_digit_numbers | not_adjacent n 0 2}
  (favorable_outcomes.card : ℚ) / (four_digit_numbers.card : ℚ)

def permutations (s : Set Nat) : Set Nat :=
  -- Function to generate all permutations of a set of digits (not implemented here)
  sorry

def not_adjacent (n : Nat) (d1 d2 : Nat) : Prop :=
  -- Function to check that digits d1 and d2 are not adjacent in the number n (not implemented here)
  sorry

end probability_even_digits_non_adjacent_l333_333731


namespace factorial_sqrt_sq_l333_333613

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333613


namespace polynomial_B_value_l333_333710

open Polynomial

theorem polynomial_B_value (A B C D : ℤ) :
  (∀ z : ℤ, (root P z) → 0 < z) →
  (roots_sum P = 10) →
  B = -108 :=
by sorry

end polynomial_B_value_l333_333710


namespace no_propositions_deducible_l333_333371

def grib := Type
def nool := Type

variables (G : set grib) (N : set nool)
variables (belongs_to : nool → grib → Prop)

noncomputable def Q1 (g : grib) : set nool := {n | belongs_to n g}
noncomputable def Q2 (g1 g2 : grib) (h : g1 ≠ g2) : set nool := {n | belongs_to n g1 ∧ belongs_to n g2}
noncomputable def Q3 (n : nool) : set grib := {g | belongs_to n g}
constant Q4 : G.card = 5
noncomputable def R1 : Prop := N.card = 10
noncomputable def R2 : Prop := ∀ g ∈ G, (Q1 g).card = 4
noncomputable def R3 : Prop := ∀ n ∈ N, ∃ n' ∈ N, n ≠ n' ∧ ∀ g ∈ G, ¬ (belongs_to n g ∧ belongs_to n' g)

theorem no_propositions_deducible (hQ1 : ∀ g ∈ G, ∃ nools, Q1 g = nools ∧ nools ⊆ N) 
                                 (hQ2 : ∀ g1 g2 ∈ G, g1 ≠ g2 → (Q2 g1 g2 (by assumption)).card = 2)
                                 (hQ3 : ∀ n ∈ N, (Q3 n).card = 3)
                                 (hQ4 : G.card = 5) :
  ¬ (R1 ∨ R2 ∨ R3) :=
sorry

end no_propositions_deducible_l333_333371


namespace product_mean_median_eq_l333_333065

noncomputable def s : Set ℝ := {7.5, 15, 23.5, 31, 39.5, 48, 56.5, 65}

noncomputable def mean (s : Set ℝ) : ℝ :=
  s.toFinset.sum / s.toFinset.card

noncomputable def median (s : Set ℝ) : ℝ :=
  if h : s.toFinset.card % 2 = 0 then
    let l := s.toFinset.sort (by apply_instance)
    (l.get ⟨(s.toFinset.card / 2) - 1, sorry⟩ + l.get ⟨s.toFinset.card / 2, sorry⟩) / 2
  else
    let l := s.toFinset.sort (by apply_instance)
    l.get ⟨s.toFinset.card / 2, sorry⟩

noncomputable def product_of_mean_and_median (s : Set ℝ) : ℝ :=
  mean s * median s

theorem product_mean_median_eq : product_of_mean_and_median s = 1260.0625 := 
  sorry

end product_mean_median_eq_l333_333065


namespace arrangement_problem_l333_333433

theorem arrangement_problem :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'];
  let valid_arrangements := {arr : list char // arr ~ letters ∧
                             (∃ l r, arr = l ++ 'C'::r ∧ 'A' ∈ l ∧ 'B' ∈ l)
                             ∨ (∃ l r, arr = l ++ 'C'::r ∧ 'A' ∈ r ∧ 'B' ∈ r)}
  in card valid_arrangements = 480 :=
by sorry

end arrangement_problem_l333_333433


namespace count_multiples_l333_333046

theorem count_multiples (n : ℕ) (h_n : n = 300) : 
  let m := 6 in 
  let m' := 12 in 
  (finset.card (finset.filter (λ x, x % m = 0 ∧ x % m' ≠ 0) (finset.range n))) = 24 :=
by
  sorry

end count_multiples_l333_333046


namespace log_custom_op_l333_333345

noncomputable def custom_op (a b : ℝ) : ℝ :=
  if a < b then (b - 1) / a else (a + 1) / b

theorem log_custom_op :
  custom_op (Real.log 10000 / Real.log 10) (1 / 2)^(-2) = 5 / 4 :=
by
  let a := Real.log 10000 / Real.log 10
  let b := (1 / 2) ^ (-2 : ℤ)
  have ha : a = 4 := by norm_num [Real.log, Real.log10_eq_log]
  have hb : b = 4 := by norm_num
  rw [ha, hb]
  exact rfl

end log_custom_op_l333_333345


namespace range_of_f_l333_333529

noncomputable def f (x : ℝ) : ℝ := -x^2 + 6 * x - 3

theorem range_of_f : set_of (λ y : ℝ, ∃ (x : ℝ), 2 ≤ x ∧ x < 5 ∧ y = f x) = set.Ioo 2 6 ∪ {6} :=
by
  sorry

end range_of_f_l333_333529


namespace find_m_l333_333312

def otimes (a b : ℝ) : ℝ := a * b + a + b^2

theorem find_m {m : ℝ} (h : 1 ⊗ m = 3) (hm : m > 0) : m = 1 := by
sorry

end find_m_l333_333312


namespace exercise_l333_333074

-- Define the given expression.
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- Define the general form expression.
def g (x h k : ℝ) (a : ℝ) := a * (x - h)^2 + k

-- Prove that a + h + k = 6 when expressing f(x) in the form a(x-h)^2 + k.
theorem exercise : ∃ a h k : ℝ, (∀ x : ℝ, f x = g x h k a) ∧ (a + h + k = 6) :=
by
  sorry

end exercise_l333_333074


namespace multiply_101_self_l333_333749

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end multiply_101_self_l333_333749


namespace find_a_extreme_value_at_1_l333_333180

theorem find_a_extreme_value_at_1 (a : ℝ) (f : ℝ → ℝ) 
  (h : f = fun x => a * Real.log x + x) 
  (h_extreme : ∀ x, f' x = (a / x) + 1) 
  (h_f_prime_at_1 : f' 1 = 0) : a = -1 := 
by
  sorry

end find_a_extreme_value_at_1_l333_333180


namespace proof_problem_l333_333369

noncomputable def a_n (n : ℕ) : ℕ := n + 2
noncomputable def b_n (n : ℕ) : ℕ := 2 * n + 3
noncomputable def C_n (n : ℕ) : ℚ := 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))
noncomputable def T_n (n : ℕ) : ℚ := (1/4) * (1 - (1/(2 * n + 1)))

theorem proof_problem :
  (∀ n, a_n n = n + 2) ∧
  (∀ n, b_n n = 2 * n + 3) ∧
  (∀ n, C_n n = 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))) ∧
  (∀ n, T_n n = (1/4) * (1 - (1/(2 * n + 1)))) ∧
  (∀ n, (T_n n > k / 54) ↔ k < 9) :=
by
  sorry

end proof_problem_l333_333369


namespace num_multiples_6_not_12_lt_300_l333_333036

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l333_333036


namespace part1_part2_part3_l333_333779

-- Definitions for Part (1)
def A1 := {1, 3}
def S1 : Set ℕ := {x | ∃ a b ∈ A1, x = a + b}
def T1 : Set ℕ := {x | ∃ a b ∈ A1, x = (a - b).natAbs}

-- Prove the sets S and T for A = {1, 3}
theorem part1 : S1 = {2, 4, 6} ∧ T1 = {0, 2} := sorry

-- Definitions for Part (2)
def A2 := {x1, x2, x3, x4}
def T2 : Set ℕ := {x | ∃ a b ∈ A2, x = (a - b).natAbs}

-- Prove x1 + x4 = x2 + x3 for specific A and T = A assumptions
theorem part2 (hA : A2 = {x1, x2, x3, x4})
              (hT : T2 = A2)
              (hOrdered : x1 < x2 ∧ x2 < x3 ∧ x3 < x4) :
  x1 + x4 = x2 + x3 := sorry

-- Definitions for Part (3)
def A3 := {x : ℕ | 0 ≤ x ∧ x ≤ 2021}
def S3 (A : Set ℕ) : Set ℕ := {x | ∃ a b ∈ A, x = a + b}
def T3 (A : Set ℕ) : Set ℕ := {x | ∃ a b ∈ A, x = (a - b).natAbs}

-- Prove the maximum size of A such that S ∩ T = ∅
theorem part3 (A : Set ℕ)
              (hSubset : A ⊆ A3)
              (hDisjoint : S3 A ∩ T3 A = ∅) :
  |A| ≤ 1348 := sorry

end part1_part2_part3_l333_333779


namespace inequality_satisfied_equality_condition_l333_333661

theorem inequality_satisfied (x y : ℝ) : x^2 + y^2 + 1 ≥ 2 * (x * y - x + y) :=
sorry

theorem equality_condition (x y : ℝ) : (x^2 + y^2 + 1 = 2 * (x * y - x + y)) ↔ (x = y - 1) :=
sorry

end inequality_satisfied_equality_condition_l333_333661


namespace oldest_child_age_l333_333930

theorem oldest_child_age (ages : Fin 7 → ℝ) (h_distinct : Function.Injective ages)
  (h_diff : ∀ i : Fin 6, ages i.succ - ages i = 3)
  (h_avg : (∑ i, ages i) / 7 = 8) :
  ∃ i : Fin 7, ages i = 19 := 
by 
  sorry

end oldest_child_age_l333_333930


namespace max_bc_lemma_l333_333893

noncomputable def max_bc (a b c : ℝ) [h_0a : 0 < a] [h_0b : 0 < b] [h_0c : 0 < c] (h : (a+c)*(b^2 + a*c) = 4*a) : ℝ :=
  2

theorem max_bc_lemma (a b c : ℝ) [h_0a : 0 < a] [h_0b : 0 < b] [h_0c : 0 < c] (h : (a+c)*(b^2 + a*c) = 4*a) : b + c ≤ max_bc a b c h :=
by
  have h_max_bc : max_bc a b c h = 2 := rfl
  rw h_max_bc
  sorry

end max_bc_lemma_l333_333893


namespace neg_ex_iff_forall_geq_0_l333_333026

theorem neg_ex_iff_forall_geq_0 :
  ¬(∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≥ 0 :=
by
  sorry

end neg_ex_iff_forall_geq_0_l333_333026


namespace period_f_range_x_l333_333404

def OP (x : ℝ) : ℝ × ℝ := (2 * Real.cos x + 1, Real.cos (2 * x) - Real.sin x + 1)
def OQ (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)
def f (x : ℝ) : ℝ := (OP x).fst * (OQ x).fst + (OP x).snd * (OQ x).snd

theorem period_f : Real.T (Real.T f x = 2 * Real.pi) := sorry
theorem range_x (x : ℝ) (h : 0 < x ∧ x < 2 * Real.pi) : f x < -1 ↔ Real.pi < x ∧ x < (3 * Real.pi) / 2 := sorry

end period_f_range_x_l333_333404


namespace frustum_surface_area_l333_333264

noncomputable def total_surface_area_of_frustum
  (R r h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (R - r)^2)
  let A_lateral := Real.pi * (R + r) * s
  let A_top := Real.pi * r^2
  let A_bottom := Real.pi * R^2
  A_lateral + A_top + A_bottom

theorem frustum_surface_area :
  total_surface_area_of_frustum 8 2 5 = 10 * Real.pi * Real.sqrt 61 + 68 * Real.pi :=
  sorry

end frustum_surface_area_l333_333264


namespace compute_abs_ab_eq_2_sqrt_111_l333_333940

theorem compute_abs_ab_eq_2_sqrt_111 (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = 2 * Real.sqrt 111 := 
sorry

end compute_abs_ab_eq_2_sqrt_111_l333_333940


namespace cannot_form_consecutive_lcms_l333_333538

theorem cannot_form_consecutive_lcms (n : ℕ) (h : n = 10^1000) 
  (a : Fin n → ℕ) (b : Fin n → ℕ)
  (hb : ∀ i : Fin n, b i = Nat.lcm (a i) (a (i + 1) % n)) :
  ¬ ∃ f : Fin n → ℕ, (∀ i : Fin n, f i = b (Fin.ofNat (i + f 0))) ∧ 
  (∀ i j : Fin n, i ≠ j → f i ≠ f j) ∧ ∃ k : ℕ, ∀ i : Fin n, f i = k + i :=
sorry

end cannot_form_consecutive_lcms_l333_333538


namespace sqrt_factorial_sq_l333_333585

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333585


namespace sqrt_factorial_product_squared_l333_333634

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333634


namespace rectangles_cover_triangle_interior_l333_333203

theorem rectangles_cover_triangle_interior
  (T : Type) [triangle T]
  (l : line)
  (rectangles : fin 3 → rectangle)
  (cover_sides : ∀ (s : side T), ∃ (r : fin 3), rectangle.contains_side rectangles[r] s)
  (parallel_sides : ∀ (r : fin 3), (rectangle.side1_parallel rectangles[r] l) ∨ (rectangle.side2_parallel rectangles[r] l)) :
  ∀ (P : point T), P ∈ triangle.interior T → ∃ (r : fin 3), rectangle.contains_point rectangles[r] P := 
by
  sorry

end rectangles_cover_triangle_interior_l333_333203


namespace find_k_tangent_l333_333010

-- Conditions
def y_curve (x : ℝ) : ℝ := 1 / x
def y_line (k x : ℝ) : ℝ := k * x + 1

-- Problem Statement: Prove that the value of k is -1/4 such that y_line is tangent to y_curve.
theorem find_k_tangent : ∃ k : ℝ, (∀ x : ℝ, y_line k x = y_curve x) → k = -1/4 :=
by
  sorry

end find_k_tangent_l333_333010


namespace largest_two_digit_integer_l333_333563

theorem largest_two_digit_integer
  (a b : ℕ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 3 * (10 * a + b) = 10 * b + a + 5) :
  10 * a + b = 13 :=
by {
  -- Sorry is placed here to indicate that the proof is not provided
  sorry
}

end largest_two_digit_integer_l333_333563


namespace new_problem_l333_333170

theorem new_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := 
by
  sorry

end new_problem_l333_333170


namespace find_a_l333_333346

noncomputable def a (u v w : ℝ) (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) 
  (h4 : u ≠ v) (h5 : v ≠ w) (h6 : u ≠ w)
  (h7 : log 3 u + log 3 v + log 3 w = 5)
  (h8 : 16 ∗ x^3 + 8 ∗ a ∗ x^2 + 7 ∗ b ∗ x + 2 ∗ a = 0) : ℝ :=
-1944

theorem find_a (b : ℝ) (u v w : ℝ) 
  (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) 
  (h4 : u ≠ v) (h5 : v ≠ w) (h6 : u ≠ w)
  (h7 : log 3 u + log 3 v + log 3 w = 5)
  (h8 : ∀ x, 16 * x^3 + 8 * a u v w h1 h2 h3 h4 h5 h6 h7 b * x^2 + 7 * b * x + 2 * a u v w h1 h2 h3 h4 h5 h6 h7 b = 0) : 
  a u v w h1 h2 h3 h4 h5 h6 h7 b = -1944 :=
sorry

end find_a_l333_333346


namespace factorial_expression_value_l333_333600

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333600


namespace count_multiples_of_6_not_12_lt_300_l333_333041

theorem count_multiples_of_6_not_12_lt_300 : 
  {N : ℕ // 0 < N ∧ N < 300 ∧ (6 ∣ N) ∧ ¬(12 ∣ N)}.toFinset.card = 25 := sorry

end count_multiples_of_6_not_12_lt_300_l333_333041


namespace find_x2_value_l333_333543

noncomputable def x := Classical.some (Exists.intro 
  (has_arctan.domain (∃ x : ℝ, 0 < x ∧ cos (arctan x) = x)))

theorem find_x2_value : ∃ x : ℝ, 0 < x ∧ cos (arctan x) = x → x^2 = (-1 + Real.sqrt 5) / 2 :=
by
  apply Exists.intro 
  use Classical.choose (Exists.intro x _) -- Introduce some positive real x satisfying the condition.
  sorry

end find_x2_value_l333_333543


namespace statement_B_statement_C_l333_333864

variable (α β γ : ℝ)
variable (a b c: ℝ)
variable (A B C: ℝ)
variable [triangle_ABC : triangle α β γ]
variable [oppositeSidesA : side A = a]
variable [oppositeSidesB : side B = b]
variable [oppositeSidesC : side C = c]

theorem statement_B (h_AB: A > B) : cos (2 * A) < cos (2 * B) :=
  sorry

theorem statement_C (h_C: C > π / 2) : sin C ^ 2 > sin A ^ 2 + sin B ^ 2 :=
  sorry

end statement_B_statement_C_l333_333864


namespace digit_at_position_2021_l333_333120

def sequence_digit (n : ℕ) : ℕ :=
  let seq := (List.range' 1 999).bind (λ i => i.toString.data.toList)
  seq.nth! (n - 1)

theorem digit_at_position_2021 : sequence_digit 2021 = 1 := 
by
  -- We skip the proof details for now
  sorry

end digit_at_position_2021_l333_333120


namespace tan_2A_cos_pi3_minus_A_l333_333471

variable (A : ℝ)

def line_equation (A : ℝ) : Prop :=
  (4 * Real.tan A = 3)

theorem tan_2A : line_equation A → Real.tan (2 * A) = -24 / 7 :=
by
  intro h 
  sorry

theorem cos_pi3_minus_A : (0 < A ∧ A < Real.pi) →
    Real.tan A = 4 / 3 →
    Real.cos (Real.pi / 3 - A) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  intro h1 h2
  sorry

end tan_2A_cos_pi3_minus_A_l333_333471


namespace twentieth_number_base_8_l333_333090

theorem twentieth_number_base_8 : ∀ (n : ℕ), n = 20 → (to_base n 8) = 24 :=
by
  intro n hn
  sorry

end twentieth_number_base_8_l333_333090


namespace pen_and_notebook_cost_l333_333690

theorem pen_and_notebook_cost (pen_cost : ℝ) (notebook_cost : ℝ) 
  (h1 : pen_cost = 4.5) 
  (h2 : pen_cost = notebook_cost + 1.8) : 
  pen_cost + notebook_cost = 7.2 := 
  by
    sorry

end pen_and_notebook_cost_l333_333690


namespace smallest_number_am_median_l333_333208

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end smallest_number_am_median_l333_333208


namespace angle_B_in_triangle_l333_333846

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end angle_B_in_triangle_l333_333846


namespace smallest_formed_number_is_40678_l333_333338

def smallest_five_digit_number (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
if h : {d1, d2, d3, d4, d5} = {0, 4, 6, 7, 8} then
  nat.join_digits [d1, d2, d3, d4, d5]
else
  0

theorem smallest_formed_number_is_40678 :
  smallest_five_digit_number 4 0 6 7 8 = 40678 :=
by
  sorry

end smallest_formed_number_is_40678_l333_333338


namespace tens_digit_mod_power_tens_digit_of_13_pow_2017_l333_333666

theorem tens_digit_mod_power (k : ℕ) : 
    (13^k % 100) = (13 ^ (k % 20) % 100) :=
  by sorry

theorem tens_digit_of_13_pow_2017 :
    ((13 ^ 2017) % 100) / 10 % 10 = 3 :=
  by 
    have h1 : 2017 % 20 = 17 := by rfl
    have h2 : 13 ^ 2017 % 100 = 13 ^ 17 % 100 := by 
      rw [tens_digit_mod_power 2017, h1]
    calc 
      (13 ^ 17 % 100) / 10 % 10 = 33 / 10 % 10 := by sorry
                            ... = 3 := by norm_num

end tens_digit_mod_power_tens_digit_of_13_pow_2017_l333_333666


namespace find_value_of_x_l333_333340

theorem find_value_of_x (x : ℝ) (h : sqrt (4 * x + 9) = 13) : x = 40 := 
sorry

end find_value_of_x_l333_333340


namespace problem1_l333_333245

theorem problem1 : 
  let S := set.Icc (0 : ℝ) 4,
      event := {x ∈ S | -1 ≤ log (x + 0.5) / log (0.5) ∧ log (x + 0.5) / log (0.5) ≤ 1} in
  (measure_theory.measure.count (by {haveI := classical.prop_decidable, exact S}) event) / (measure_theory.measure.count (by {haveI := classical.prop_decidable, exact S}) univ) = 3 / 8 :=
sorry

end problem1_l333_333245


namespace triangle_with_angle_ratio_obtuse_l333_333850

theorem triangle_with_angle_ratio_obtuse 
  (a b c : ℝ) 
  (h_sum : a + b + c = 180) 
  (h_ratio : a = 2 * d ∧ b = 2 * d ∧ c = 5 * d) : 
  90 < c :=
by
  sorry

end triangle_with_angle_ratio_obtuse_l333_333850


namespace second_quadrant_implies_value_of_m_l333_333858

theorem second_quadrant_implies_value_of_m (m : ℝ) : 4 - m < 0 → m = 5 := by
  intro h
  have ineq : m > 4 := by
    linarith
  sorry

end second_quadrant_implies_value_of_m_l333_333858


namespace last_three_digits_of_sum_l333_333195

noncomputable def sum_term (n : ℕ) : ℚ :=
  if n = 0 then 0 else (n^2 - 2) / n.factorial

def sum_series (m : ℕ) : ℚ :=
  (Finset.range (m + 1)).sum (λ n, sum_term n)

theorem last_three_digits_of_sum :
  (2021.factorial * sum_series 2021) % 1000 = 977 :=
sorry

end last_three_digits_of_sum_l333_333195


namespace infinite_circles_intersect_no_more_than_two_l333_333438

theorem infinite_circles_intersect_no_more_than_two :
  ∃ (C : ℕ → ℝ × ℝ) (r : ℝ), (0 < r) ∧
  (∀ (m c : ℝ) (L : ℝ → ℝ), L = λ x, m * x + c →
    ∃ (ℕ : ℕ → bool),
    ∀ x y : ℕ, (x ≠ y) → (L (fst (C x)) = snd (C x) → L (fst (C y)) = snd (C y) → false)) :=
sorry

end infinite_circles_intersect_no_more_than_two_l333_333438


namespace quadrilateral_is_rectangle_if_three_right_angles_l333_333547

-- Definition of a quadrilateral
structure quadrilateral (A B C D : Type) :=
(a b c d : A)

-- Definition of a right angle in the context of the quadrilateral
def is_right_angle {A : Type} (α : A) : Prop := sorry

-- Definition of the internal angles of a quadrilateral
def internal_angles_sum_360 {A : Type} [angle : quadrilateral A B C D] (x y z w: A) 
  (hx : is_right_angle x) (hy : is_right_angle y) (hz : is_right_angle z) : 
  is_right_angle w := sorry

-- The main statement: a quadrilateral is a rectangle if and only if three of its internal angles are right angles
theorem quadrilateral_is_rectangle_if_three_right_angles 
  {A B C D : Type} [angle : quadrilateral A B C D] 
  (x y z w : A) 
  (hx : is_right_angle x) (hy : is_right_angle y) (hz : is_right_angle z) : 
  is_right_angle w := 
sorry

end quadrilateral_is_rectangle_if_three_right_angles_l333_333547


namespace compute_sqrt_factorial_square_l333_333621

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333621


namespace total_area_l333_333717

variable (A : ℝ)

-- Defining the conditions
def first_carpet : Prop := 0.55 * A = 36
def second_carpet : Prop := 0.25 * A = A * 0.25
def third_carpet : Prop := 0.15 * A = 18 + 6
def remaining_floor : Prop := 0.05 * A + 0.55 * A + 0.25 * A + 0.15 * A = A

-- Main theorem to prove the total area
theorem total_area : first_carpet A → second_carpet A → third_carpet A → remaining_floor A → A = 65.45 :=
by
  sorry

end total_area_l333_333717


namespace find_value_l333_333465

theorem find_value (x : ℝ) (hx : x + 1/x = 4) : x^3 + 1/x^3 = 52 := 
by 
  sorry

end find_value_l333_333465


namespace solution_set_of_inequality_l333_333961

theorem solution_set_of_inequality :
  { x : ℝ | (x - 5) / (x + 1) ≤ 0 } = { x : ℝ | -1 < x ∧ x ≤ 5 } :=
sorry

end solution_set_of_inequality_l333_333961


namespace clock_spoke_angle_l333_333259

-- Define the parameters of the clock face and the problem.
def num_spokes := 10
def total_degrees := 360
def degrees_per_spoke := total_degrees / num_spokes
def position_3_oclock := 3 -- the third spoke
def halfway_45_oclock := 5 -- approximately the fifth spoke
def spokes_between := halfway_45_oclock - position_3_oclock
def smaller_angle := spokes_between * degrees_per_spoke
def expected_angle := 72

-- Statement of the problem
theorem clock_spoke_angle :
  smaller_angle = expected_angle := by
    -- Proof is omitted
    sorry

end clock_spoke_angle_l333_333259


namespace sqrt_factorial_squared_l333_333568

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333568


namespace count_multiples_of_6_not_12_lt_300_l333_333042

theorem count_multiples_of_6_not_12_lt_300 : 
  {N : ℕ // 0 < N ∧ N < 300 ∧ (6 ∣ N) ∧ ¬(12 ∣ N)}.toFinset.card = 25 := sorry

end count_multiples_of_6_not_12_lt_300_l333_333042


namespace shaded_fraction_of_square_is_three_fourths_l333_333425

/--
In a square, points A and B are midpoints of two adjacent sides.
A line segment is drawn from point A to the opposite vertex of the side that does not contain B, forming a triangle.
Prove that the fraction of the interior of the square that is shaded (not occupied by the triangle) is 3/4.
-/
theorem shaded_fraction_of_square_is_three_fourths (s : ℝ) : 
  let A := s / 2,
  let B := s,
  let triangle_area := (1 / 2) * (s / 2) * s,
  let square_area := s * s,
  (square_area - triangle_area) / square_area = 3 / 4 :=
by
  let A := s / 2
  let B := s
  let triangle_area := (1 / 2) * (s / 2) * s
  let square_area := s * s
  calc
    (square_area - triangle_area) / square_area = sorry

end shaded_fraction_of_square_is_three_fourths_l333_333425


namespace factorial_sqrt_sq_l333_333618

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333618


namespace count_real_solutions_l333_333817

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end count_real_solutions_l333_333817


namespace card_numbers_satisfy_conditions_l333_333909

-- Definitions for assumptions stated in the conditions
variables {a b c d : ℕ} (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d)
 (habc : a + b < 9 ∧ a + c < 9 ∧ b + c = 9 ∧ a + d = 9 ∧ b + d > 9 ∧ c + d > 9)

-- The target sets of numbers on the cards
def valid_sets := [
  (1, 2, 7, 8), 
  (1, 3, 6, 8), 
  (1, 4, 5, 8), 
  (2, 3, 6, 7), 
  (2, 4, 5, 7), 
  (3, 4, 5, 6)
]

-- The target proof statement.
theorem card_numbers_satisfy_conditions :
  (a, b, c, d) ∈ valid_sets :=
begin
  sorry
end

end card_numbers_satisfy_conditions_l333_333909


namespace smallest_positive_period_monotonically_decreasing_minimum_value_in_interval_l333_333395

noncomputable def f (x : ℝ) : ℝ := (sin (x / 2) * cos (x / 2)) + (cos (x / 2))^2 - 1

theorem smallest_positive_period : (∀ x : ℝ, f (x) = f (x + 2 * π)) :=
sorry

theorem monotonically_decreasing (k : ℤ) :
  ∀ x : ℝ, 2 * k * π + π / 4 ≤ x ∧ x ≤ 2 * k * π + 5 * π / 4 → f x ≤ f (x + ε) :=
sorry

theorem minimum_value_in_interval :
  ∃ x : ℝ, (π / 4 ≤ x ∧ x ≤ 3 * π / 2) ∧ f x = - (√2 + 1) / 2 :=
sorry

end smallest_positive_period_monotonically_decreasing_minimum_value_in_interval_l333_333395


namespace sum_of_binaries_l333_333969

theorem sum_of_binaries : 
  let bin1 := 2^2 + 0*2^1 + 1*2^0,
      bin2 := 1*2^2 + 1*2^1 + 0*2^0
  in bin1 + bin2 = 11 := 
by
  let bin1 := 2^2 + 0*2^1 + 1*2^0
  let bin2 := 1*2^2 + 1*2^1 + 0*2^0
  sorry

end sum_of_binaries_l333_333969


namespace reverse_difference_198_l333_333985

theorem reverse_difference_198 (a : ℤ) : 
  let N := 100 * (a - 1) + 10 * a + (a + 1)
  let M := 100 * (a + 1) + 10 * a + (a - 1)
  M - N = 198 := 
by
  sorry

end reverse_difference_198_l333_333985


namespace chocolates_for_sister_l333_333869
-- Importing necessary library

-- Lean 4 statement of the problem
theorem chocolates_for_sister (S : ℕ) 
  (herself_chocolates_per_saturday : ℕ := 2)
  (birthday_gift_chocolates : ℕ := 10)
  (saturdays_in_month : ℕ := 4)
  (total_chocolates : ℕ := 22) 
  (monthly_chocolates_herself := saturdays_in_month * herself_chocolates_per_saturday) 
  (equation : saturdays_in_month * S + monthly_chocolates_herself + birthday_gift_chocolates = total_chocolates) : 
  S = 1 :=
  sorry

end chocolates_for_sister_l333_333869


namespace first_player_wins_l333_333429

-- Define the chessboard and the initial setup of knights on opposite corners
structure Chessboard where
  size : ℕ
  initial_k1_pos : ℕ × ℕ
  initial_k2_pos : ℕ × ℕ

-- Define the game conditions
def opposite_corners (board : Chessboard) : Prop :=
  board.initial_k1_pos.1 ≠ board.initial_k2_pos.1 ∧
  board.initial_k1_pos.2 ≠ board.initial_k2_pos.2

def knight_moves (pos : ℕ × ℕ) : list (ℕ × ℕ) := 
  [(pos.1 + 2, pos.2 + 1), (pos.1 + 2, pos.2 - 1), (pos.1 - 2, pos.2 + 1), (pos.1 - 2, pos.2 - 1),
   (pos.1 + 1, pos.2 + 2), (pos.1 + 1, pos.2 - 2), (pos.1 - 1, pos.2 + 2), (pos.1 - 1, pos.2 - 2)]

noncomputable def player_can_win (board : Chessboard) : Prop :=
  ∃ strategy : list (ℕ × ℕ), ∀ k1_pos k2_pos, 
  (k1_pos ≠ k2_pos → (knight_moves k1_pos).any (λ pos, pos ≠ k2_pos)) → 
  (strategy.all (λ move, move ∈ knight_moves k1_pos ∧ move ≠ k2_pos))

-- Lean 4 statement to prove that the first player wins
theorem first_player_wins (board : Chessboard) (h : opposite_corners board) : player_can_win board :=
  sorry

end first_player_wins_l333_333429


namespace arrange_children_in_car_alpha_l333_333420

noncomputable def num_ways_to_arrange_children : ℕ :=
24

theorem arrange_children_in_car_alpha :
  ∀ (A B C D : Type)
    (children_A children_B children_C children_D : A → β) 
    -- Assume children_A, children_B, children_C, children_D specify families' children
    (car_alpha car_beta : set β) 
    -- cars alpha and beta
    (seat_lim_car_alpha seat_lim_car_beta : ℕ) 
    -- seating limits for cars
    (twin_sisters_A : children_A),
    -- twin sisters from family A must ride in the same car
    car_alpha.union car_beta = set.univ →
    seat_lim_car_alpha = 4 →
    seat_lim_car_beta = 4 →
    (@set.to_finset _ (children_A) (twin_sisters_A) ∩ car_alpha.to_finset = (twin_sisters_A).to_finset) ∨
    (@set.to_finset _ (children_A) (twin_sisters_A) ∩ car_beta.to_finset = (twin_sisters_A).to_finset) →
    (@set.to_finset _ (children_A) (twin_sisters_A) ∩ car_alpha.to_finset ≠ ∅ ∧
    @set.to_finset _ (children_A) (twin_sisters_A) ∩ car_beta.to_finset = ∅ ∨
    @set.to_finset _ (children_A) (twin_sisters_A) ∩ car_alpha.to_finset = ∅ ∧
    @set.to_finset _ (children_A) (twin_sisters_A) ∩ car_beta.to_finset ≠ ∅) ∧
    (exists (family_func : (B | C | D) → A × B × C × D), 
    ∑ i in family_func.range, (if i ∈ car_alpha.to_finset.to_set then 1 else 0) = 2) →
    num_ways_to_arrange_children = 24 :=
sorry

end arrange_children_in_car_alpha_l333_333420


namespace factorization_of_expression_l333_333410

theorem factorization_of_expression
  (a b c : ℝ)
  (expansion : (b+c)*(c+a)*(a+b) + abc = (a+b+c)*(ab+ac+bc)) : 
  ∃ (m l : ℝ), (m = 0 ∧ l = a + b + c ∧ 
  (b+c)*(c+a)*(a+b) + abc = m*(a^2 + b^2 + c^2) + l*(ab + ac + bc)) :=
by
  sorry

end factorization_of_expression_l333_333410


namespace parallel_tangent_a_eq_neg_e_extreme_values_l333_333802

noncomputable def f (a x : ℝ) : ℝ := a / Real.exp x + x + 1

def derivative_f (a : ℝ) (x : ℝ) : ℝ := 1 - a / Real.exp x

theorem parallel_tangent_a_eq_neg_e (a : ℝ) (h : derivative_f a 1 = 2) : a = -Real.exp 1 := by
  sorry

theorem extreme_values (a : ℝ) (h1 : 0 < a) : 
  ∃ x_min : ℝ, x_min = Real.log a ∧ f a x_min = Real.log a + 2 ∧ 
  ∀ x : ℝ, f a x ≥ f a x_min := by
  sorry

end parallel_tangent_a_eq_neg_e_extreme_values_l333_333802


namespace increased_contact_area_effect_l333_333348

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l333_333348


namespace find_x_range_l333_333788

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 ≥ 0 → x2 ≥ 0 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

theorem find_x_range (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : decreasing_on_nonnegative f)
  (h3 : f (1/3) = 3/4)
  (h4 : ∀ x : ℝ, 4 * f (Real.logb (1/8) x) > 3) :
  ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ True := sorry

end find_x_range_l333_333788


namespace opposite_of_neg_nine_l333_333949

theorem opposite_of_neg_nine : -(-9) = 9 :=
by
  sorry

end opposite_of_neg_nine_l333_333949


namespace angle_A_measure_l333_333839

theorem angle_A_measure 
  (B : ℝ) 
  (angle_in_smaller_triangle : ℝ) 
  (sum_of_triangle_angles_eq_180 : ∀ (x y z : ℝ), x + y + z = 180)
  (C : ℝ) 
  (angle_pair_linear : ∀ (x y : ℝ), x + y = 180) 
  (A : ℝ) 
  (C_eq_180_minus_B : C = 180 - B) 
  (A_eq_180_minus_angle_in_smaller_triangle_minus_C : 
    A = 180 - angle_in_smaller_triangle - C) :
  A = 70 :=
by
  sorry

end angle_A_measure_l333_333839


namespace find_y_l333_333650

theorem find_y : ∀ (x y : ℤ), x > 0 ∧ y > 0 ∧ x % y = 9 ∧ (x:ℝ) / (y:ℝ) = 96.15 → y = 60 :=
by
  intros x y h
  sorry

end find_y_l333_333650


namespace count_multiples_of_6_not_12_lt_300_l333_333039

theorem count_multiples_of_6_not_12_lt_300 : 
  {N : ℕ // 0 < N ∧ N < 300 ∧ (6 ∣ N) ∧ ¬(12 ∣ N)}.toFinset.card = 25 := sorry

end count_multiples_of_6_not_12_lt_300_l333_333039


namespace semicircle_perimeter_l333_333234

/-- Lean code for proving the approximate perimeter of a semicircular cubicle -/
def perimeter_semicircle_approx (r : ℝ) (pi_approx : ℝ) : ℝ :=
  let diameter := 2 * r
  let half_circumference := pi_approx * r
  diameter + half_circumference

theorem semicircle_perimeter (r : ℝ) (pi_approx : ℝ) (approx_perimeter : ℝ) :
  r = 14 → pi_approx = 3.14 → approx_perimeter = 71.96 →
  perimeter_semicircle_approx r pi_approx = approx_perimeter := by
  intros hr hpi happ
  rw [hr, hpi, happ]
  unfold perimeter_semicircle_approx
  norm_num
  sorry

end semicircle_perimeter_l333_333234


namespace johns_donation_l333_333236

theorem johns_donation
  (n : ℕ)
  (new_average : ℝ)
  (percent_increase : ℝ)
  (previous_contributions : n = 6)
  (new_average_equals_75 : new_average = 75)
  (percent_increase_equals_50 : percent_increase = 50) :
  let initial_total_contribution := 50 * 6 in
  let total_donation := initial_total_contribution + 225 in
  75 = (total_donation / 7) :=
by
  sorry

end johns_donation_l333_333236


namespace range_of_m_l333_333663

theorem range_of_m (x m : ℝ) (h1: |x - m| < 1) (h2: x^2 - 8 * x + 12 < 0) (h3: ∀ x, (x^2 - 8 * x + 12 < 0) → ((m - 1) < x ∧ x < (m + 1))) : 
  3 ≤ m ∧ m ≤ 5 := 
sorry

end range_of_m_l333_333663


namespace find_dot_product_magnitude_l333_333456

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def vector_cross (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2 * w.3 - v.3 * w.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2 - v.2 * w.1)

def vector_dot (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def abs (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

theorem find_dot_product_magnitude (c d : ℝ × ℝ × ℝ)
  (h1 : vector_magnitude c = 3)
  (h2 : vector_magnitude d = 4)
  (h3 : vector_magnitude (vector_cross c d) = 6) :
  abs (vector_dot c d) = 6 * real.sqrt 3 :=
by
  sorry -- proof omitted

end find_dot_product_magnitude_l333_333456


namespace factorial_expression_value_l333_333594

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333594


namespace complex_eq_l333_333130

theorem complex_eq (z1 z2 A : ℂ) (hA : A ≠ 0) 
  (h : z1 * conj z2 + conj A * z1 + A * conj z2 = 0) : 
  (|z1 + A| * |z2 + A| = |A|^2) ∧ 
  (z1 + A) / (z2 + A) = |(z1 + A) / (z2 + A)| :=
sorry

end complex_eq_l333_333130


namespace cube_height_sum_l333_333261

-- Define the problem conditions as Lean hypotheses and definitions
variables (r s t d : ℝ) (a b c : ℝ)
variables (ha : a^2 + b^2 + c^2 = 1)
variables (h1 : 8 * a + d = 8)
variables (h2 : 8 * b + d = 9)
variables (h3 : 8 * c + d = 11)
variables (d_eq : d = 14 / 3)
variables (r_eq : r = 42)
variables (s_eq : s = 9)
variables (t_eq : t = 3)
variables (hdist : ∀ x y z, ∀ d, a * x + b * y + c * z + d = (14 - (14/3) * sqrt(3)) / 3)

-- State the proof goal that the sum r + s + t equals 54
theorem cube_height_sum : r + s + t = 54 :=
by 
  sorry

end cube_height_sum_l333_333261


namespace final_price_correct_l333_333149

-- Definitions of the conditions
def initial_price := 2000
def first_discount_rate := 0.15
def second_discount_rate := 0.10
def gift_card := 200

-- Calculate the price after the first discount
def price_after_first_discount := initial_price - (initial_price * first_discount_rate)

-- Calculate the price after the second discount
def price_after_second_discount := price_after_first_discount - (price_after_first_discount * second_discount_rate)

-- Final price after applying gift card
def final_price := price_after_second_discount - gift_card

-- Prove that the final price is $1330
theorem final_price_correct : final_price = 1330 := by
  sorry

end final_price_correct_l333_333149


namespace sm_parallel_ac_iff_om_perp_bs_l333_333853

-- Variables
variables (A B C M O S : Point)
variables [TriangleIsosceles ABC AB]
variables [OnLine M BC]
variables [Circumcenter O ABC]
variables [Incenter S ABC]

-- Theorem statement
theorem sm_parallel_ac_iff_om_perp_bs
    (SM_parallel_AC : Parallel SM AC)
    (OM_perpendicular_BS : Perpendicular OM BS) :
    (SM_parallel_AC ↔ OM_perpendicular_BS) :=
sorry

end sm_parallel_ac_iff_om_perp_bs_l333_333853


namespace rectangles_are_squares_l333_333840

variable (n : ℕ)
variable (k l : ℕ)
variable (a b c : Fin n → ℕ)

theorem rectangles_are_squares
  (h1 : 1 < n)
  (h2 : ∀ i : Fin n, a i * b i * c i = k * l)
  (h3 : (∀ i : Fin n, l = b i * (a i)) ∧ (∀ i : Fin n, k = c i * (a i)))
  (h4 : Prime (n * ∑ i, b i * c i)) :
  k = l :=
sorry

end rectangles_are_squares_l333_333840


namespace sum_elements_AB_eq_14_l333_333311

def set_operation (A B : Set ℕ) : Set ℕ := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2}
def AB : Set ℕ := set_operation A B

theorem sum_elements_AB_eq_14 : (Finset.univ.filter (λ x, x ∈ AB)).sum = 14 :=
by
  -- Placeholder for the proof
  sorry

end sum_elements_AB_eq_14_l333_333311


namespace gcd_51457_37958_l333_333333

theorem gcd_51457_37958 : Nat.gcd 51457 37958 = 1 := 
  sorry

end gcd_51457_37958_l333_333333


namespace not_function_of_x_l333_333229

theorem not_function_of_x : 
  ∃ x : ℝ, ∃ y1 y2 : ℝ, (|y1| = 2 * x ∧ |y2| = 2 * x ∧ y1 ≠ y2) := sorry

end not_function_of_x_l333_333229


namespace product_fraction_l333_333753

theorem product_fraction :
  (∏ k in Finset.range (50 - 2) + 3, (1 - (1 : ℝ) / k)) = (1 : ℝ) / 25 := by
  sorry

end product_fraction_l333_333753


namespace stratified_sampling_third_grade_l333_333867

theorem stratified_sampling_third_grade:
  ∀ (students_first students_second students_third total_sample : ℕ),
  students_first = 400 →
  students_second = 400 →
  students_third = 500 →
  total_sample = 65 →
  let sampling_ratio := total_sample / (students_first + students_second + students_third : ℕ)
  in students_third * sampling_ratio = 25 :=
by
  intros students_first students_second students_third total_sample
  intros hf hs ht htots
  rw [hf, hs, ht, htots]
  let sampling_ratio := total_sample / (students_first + students_second + students_third : ℕ)
  rw [sampling_ratio]
  rw [total_sample, students_first, students_second, students_third]
  norm_num
  sorry

end stratified_sampling_third_grade_l333_333867


namespace center_of_rotation_l333_333314

noncomputable def f (z : ℂ) : ℂ := ((-1 - (Complex.I * Real.sqrt 3)) * z + (2 * Real.sqrt 3 - 12 * Complex.I)) / 2

theorem center_of_rotation :
  ∃ c : ℂ, f c = c ∧ c = -5 * Real.sqrt 3 / 2 - 7 / 2 * Complex.I :=
by
  sorry

end center_of_rotation_l333_333314


namespace max_ab_value_l333_333830

theorem max_ab_value (a b c : ℝ) (h : ∀ x : ℝ, 2 * x + 2 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 2 * x^2 - 2 * x + 4) :
  ∃ (a b : ℝ), (a * (2 - 2 * a)) = 1 / 2 :=
begin
  sorry
end

end max_ab_value_l333_333830


namespace sqrt_factorial_sq_l333_333592

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333592


namespace max_value_f_range_of_g_l333_333028

-- Definitions for given vectors and function f
def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1 / 2)
def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos (2 * x))
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Theorem statements
theorem max_value_f : ∀ x : ℝ, f x ≤ 1 :=
by sorry

theorem range_of_g : ∀ x ∈ Set.Icc 0 (Real.pi / 2),
  -1 / 2 ≤ (f (x + Real.pi / 6)) ≤ 1 :=
by sorry

end max_value_f_range_of_g_l333_333028


namespace quadrilaterals_count_l333_333171

theorem quadrilaterals_count (n : ℕ) (hn : n = 10) : 
  (nat.choose n 4) = 300 :=
by
  rw hn
  exact nat.choose_spec 10 4 sorry

end quadrilaterals_count_l333_333171


namespace eccentricity_correct_l333_333013

def hyperbola_eq : Prop := (y : ℝ), (x: ℝ), (y^2 / 4 - x^2 / 9 = 1)

noncomputable def eccentricity : ℝ :=
  real.sqrt(1 + 9 / 4)

theorem eccentricity_correct : eccentricity = real.sqrt(13) / 2 :=
by
  sorry

end eccentricity_correct_l333_333013


namespace calculate_inverse_expression_l333_333301

theorem calculate_inverse_expression :
  (3 - 4 * (4 - 6)⁻¹ + 2)⁻¹ = 1 / 7 :=
by
  have h1 : (4 - 6)⁻¹ = -1 / 2 := by 
    sorry
  sorry

end calculate_inverse_expression_l333_333301


namespace only_solution_n_sigma_over_p_minus_1_eq_n_l333_333755

theorem only_solution_n_sigma_over_p_minus_1_eq_n (n : ℕ) (h1 : n ≥ 2) (h2 : σ n / (p n - 1) = n) : n = 6 :=
sorry

end only_solution_n_sigma_over_p_minus_1_eq_n_l333_333755


namespace count_multiples_l333_333045

theorem count_multiples (n : ℕ) (h_n : n = 300) : 
  let m := 6 in 
  let m' := 12 in 
  (finset.card (finset.filter (λ x, x % m = 0 ∧ x % m' ≠ 0) (finset.range n))) = 24 :=
by
  sorry

end count_multiples_l333_333045


namespace prime_sum_divisible_l333_333307

theorem prime_sum_divisible (p : Fin 2021 → ℕ) (prime : ∀ i, Nat.Prime (p i))
  (h : 6060 ∣ Finset.univ.sum (fun i => (p i)^4)) : 4 ≤ Finset.card (Finset.univ.filter (fun i => p i < 2021)) :=
sorry

end prime_sum_divisible_l333_333307


namespace sqrt_factorial_squared_l333_333572

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333572


namespace Cubs_home_runs_third_inning_l333_333078

variable (X : ℕ)

theorem Cubs_home_runs_third_inning 
  (h : X + 1 + 2 = 2 + 3) : 
  X = 2 :=
by 
  sorry

end Cubs_home_runs_third_inning_l333_333078


namespace creeping_jennies_per_planter_l333_333986

theorem creeping_jennies_per_planter (c : ℕ) :
  (4 * (15 + 4 * c + 4 * 3.50) = 180) → (c = 4) :=
by
  -- introduce c
  assume h : 4 * (15 + 4 * c + 4 * 3.50) = 180
  -- we do not need to perform the steps, just proving the statement is sufficient
  sorry

end creeping_jennies_per_planter_l333_333986


namespace probability_of_number_less_than_three_l333_333976

theorem probability_of_number_less_than_three :
  let faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Finset ℕ := {1, 2}
  (favorable_outcomes.card : ℚ) / (faces.card : ℚ) = 1 / 3 :=
by
  -- This is the placeholder for the actual proof.
  sorry

end probability_of_number_less_than_three_l333_333976


namespace count_integers_with_digit_sum_18_l333_333030

def digit_sum (n : ℕ) : ℕ :=
  n / 100 + (n % 100) / 10 + n % 10

theorem count_integers_with_digit_sum_18 :
  (Finset.filter (λ n, digit_sum n = 18) (Finset.Icc 500 800)).card = 21 :=
by
  sorry

end count_integers_with_digit_sum_18_l333_333030


namespace max_sum_possible_l333_333741

theorem max_sum_possible (a b c d e f : ℕ) 
  (h1 : a + b + e = c + d + f) 
  (h2 : a + c = b + d) 
  (h3 : a + c = e + f) 
  (h4 : b + d = e + f) 
  (h5 : {a, b, c, d, e, f} ⊆ {2, 5, 8, 11, 14}) :
  ∃ sum_max : ℕ, sum_max = 27 :=
begin
  -- proof here
  sorry
end

end max_sum_possible_l333_333741


namespace point_P_on_x_axis_l333_333416

noncomputable def point_on_x_axis (m : ℝ) : ℝ × ℝ := (4, m + 1)

theorem point_P_on_x_axis (m : ℝ) (h : point_on_x_axis m = (4, 0)) : m = -1 := 
by
  sorry

end point_P_on_x_axis_l333_333416


namespace johns_earnings_l333_333442

def trees_planted (rows cols : ℕ) : ℕ := rows * cols
def apples_produced (trees apples_per_tree : ℕ) : ℕ := trees * apples_per_tree
def money_earned (apples price_per_apple : ℕ) : ℝ := apples * price_per_apple

theorem johns_earnings : 
  let rows := 3
  let cols := 4
  let trees := trees_planted rows cols
  let apples_per_tree := 5
  let apples := apples_produced trees apples_per_tree
  let price_per_apple := 0.5
  money_earned apples price_per_apple = 30 :=
by
  sorry

end johns_earnings_l333_333442


namespace smallest_value_l333_333363

theorem smallest_value {a b : ℤ} 
  (h : |a - 2| + (b + 3)^2 = 0) : 
  min (a + b) (min (a - b) (min (b^a) (a * b))) = a * b := 
sorry

end smallest_value_l333_333363


namespace sqrt_factorial_squared_l333_333570

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333570


namespace fg_of_3_l333_333118

def f (x : ℕ) : ℕ := 5 * x - 1
def g (x : ℕ) : ℕ := (x + 2)^2 + 3

theorem fg_of_3 : f(g(3)) = 139 := by
  sorry

end fg_of_3_l333_333118


namespace find_number_l333_333336

-- Define the problem conditions
def problem_condition (x : ℝ) : Prop := 2 * x - x / 2 = 45

-- Main theorem statement
theorem find_number : ∃ (x : ℝ), problem_condition x ∧ x = 30 :=
by
  existsi 30
  -- Include the problem condition and the solution check
  unfold problem_condition
  -- We are skipping the proof using sorry to just provide the statement
  sorry

end find_number_l333_333336


namespace elizabeth_money_l333_333323

theorem elizabeth_money :
  (∀ (P N : ℝ), P = 5 → N = 6 → 
    (P * 1.60 + N * 2.00) = 20.00) :=
by
  sorry

end elizabeth_money_l333_333323


namespace factorial_sqrt_sq_l333_333616

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333616


namespace sixth_number_is_188_l333_333511

/--
Given 11 numbers such that:
1. The average of the 11 numbers is 50.
2. The average of the first 6 numbers is 58.
3. The average of the last 6 numbers is 65.

Prove that the 6th number is 188.
-/

theorem sixth_number_is_188
  (A : Fin 11 → ℝ)
  (h_avg_11 : (∑ i, A i) = 11 * 50)
  (h_avg_first6 : (∑ i in Finset.range 6, A i) = 6 * 58)
  (h_avg_last6 : (∑ i in Finset.Ico 5 11, A i) = 6 * 65) :
  A 5 = 188 :=
by
  sorry

end sixth_number_is_188_l333_333511


namespace cube_surface_area_l333_333262

noncomputable def volume_cylinder (r : ℝ) (h : ℝ) : ℝ := π * r ^ 2 * h

noncomputable def edge_length_cube (V : ℝ) : ℝ := (V)^(1/3)

noncomputable def surface_area_cube (s : ℝ) : ℝ := 6 * s ^ 2

theorem cube_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 12) :
  surface_area_cube (edge_length_cube (volume_cylinder r h)) ≈ 882.7392 :=
by
  sorry

end cube_surface_area_l333_333262


namespace solve_monetary_prize_problem_l333_333266

def monetary_prize_problem : Prop :=
  ∃ (P x y : ℝ), 
    P = x + y + 30000 ∧
    x = (1/2) * P - (3/22) * (y + 30000) ∧
    y = (1/4) * P + (1/56) * x ∧
    P = 95000 ∧
    x = 40000 ∧
    y = 25000

theorem solve_monetary_prize_problem : monetary_prize_problem :=
  sorry

end solve_monetary_prize_problem_l333_333266


namespace chips_total_calories_l333_333441

theorem chips_total_calories:
  ∀ (c : ℝ),
  (10 * c + 8 * c = 108) →
  (10 * c = 60) :=
begin
  intros c h₁,
  sorry -- proof not required
end

end chips_total_calories_l333_333441


namespace main_theorem_l333_333462

variable {n : ℕ}
variable {a : Fin n → ℝ}
noncomputable def b (k : ℕ) : ℝ := (Finset.range k).Sum (fun i => a ⟨i, i < k⟩) / k

noncomputable def C : ℝ := (Finset.range n).Sum (fun i => (a ⟨i, i < n⟩ - b (i+1))^2)

noncomputable def D : ℝ := (Finset.range n).Sum (fun i => (a ⟨i, i < n⟩ - b n)^2)

theorem main_theorem (n : ℕ) {a : Fin n → ℝ} :
  C ≤ D ∧ D ≤ 2 * C :=
  sorry

end main_theorem_l333_333462


namespace investment_calculation_l333_333984

theorem investment_calculation
    (R Trishul Vishal Alok Harshit : ℝ)
    (hTrishul : Trishul = 0.9 * R)
    (hVishal : Vishal = 0.99 * R)
    (hAlok : Alok = 1.035 * Trishul)
    (hHarshit : Harshit = 0.95 * Vishal)
    (hTotal : R + Trishul + Vishal + Alok + Harshit = 22000) :
  R = 22000 / 3.8655 ∧
  Trishul = 0.9 * R ∧
  Vishal = 0.99 * R ∧
  Alok = 1.035 * Trishul ∧
  Harshit = 0.95 * Vishal ∧
  R + Trishul + Vishal + Alok + Harshit = 22000 :=
sorry

end investment_calculation_l333_333984


namespace sin_A_over_tanA_plus_1_over_tanB_range_l333_333836

variables {A B C : ℝ} {a b c : ℝ}

-- Given: a, b, c form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- The main theorem to prove
theorem sin_A_over_tanA_plus_1_over_tanB_range 
  (h1 : ∀ A B C : ℝ, a b c : ℝ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧
    is_geometric_sequence a b c) : 
  ∃ q : ℝ, (sqrt 5 - 1) / 2 < q ∧ q < (sqrt 5 + 1) / 2 :=
sorry

end sin_A_over_tanA_plus_1_over_tanB_range_l333_333836


namespace arithmetic_progression_contains_sixth_power_l333_333829

theorem arithmetic_progression_contains_sixth_power
  (a h : ℕ) (a_pos : 0 < a) (h_pos : 0 < h)
  (sq : ∃ n : ℕ, a + n * h = k^2)
  (cube : ∃ m : ℕ, a + m * h = l^3) :
  ∃ p : ℕ, ∃ q : ℕ, a + q * h = p^6 := sorry

end arithmetic_progression_contains_sixth_power_l333_333829


namespace lcm_of_4_9_10_27_l333_333221

theorem lcm_of_4_9_10_27 : Nat.lcm (Nat.lcm 4 9) (Nat.lcm 10 27) = 540 :=
by
  sorry

end lcm_of_4_9_10_27_l333_333221


namespace three_million_times_three_million_l333_333244

theorem three_million_times_three_million : 
  (3 * 10^6) * (3 * 10^6) = 9 * 10^12 := 
by
  sorry

end three_million_times_three_million_l333_333244


namespace bacterium_radius_in_scientific_notation_l333_333527

theorem bacterium_radius_in_scientific_notation :
  ∃ a n, (0.000012 : ℝ) = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ (a = 1.2 ∧ n = -5) :=
by
  sorry

end bacterium_radius_in_scientific_notation_l333_333527


namespace integral_value_l333_333175

theorem integral_value (a : ℝ)
  (h: ∀ x: ℝ, (a = 1 ∨ a = -1) ∧ 
    ((|a| * x - sqrt 3 / 6)^3 = 
    C(3, 1) * |a|^2 * (-sqrt 3 / 6) = -sqrt 3 / 2)) :
  ∫ x in -2..a, x^2 = 3 ∨ ∫ x in -2..a, x^2 = 7 / 3 := by 
sorry

end integral_value_l333_333175


namespace sqrt_factorial_mul_squared_l333_333604

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333604


namespace sum_AP_squared_PB_PC_l333_333124

-- Given conditions and definitions
def point := {x : ℝ // 0 ≤ x}
variables (P : fin 41 → point) (A B C : point)
variables (d_AB : A = ⟨7, by norm_num⟩)
variables (d_AC : A = ⟨7, by norm_num⟩)

-- The question to prove
theorem sum_AP_squared_PB_PC {A B C : point} (P : fin 41 → point)
  (hAB : ∀ i : fin 41, dist A B = 7)
  (hAC : ∀ i : fin 41, dist A C = 7) :
  ∑ i : fin 41, (dist A (P i))^2 + ((dist (P i) B) * (dist (P i) C)) = 2009 :=
by sorry

end sum_AP_squared_PB_PC_l333_333124


namespace finite_set_independent_subset_dominance_l333_333491

theorem finite_set_independent_subset_dominance (A : Finset ℕ) (hA : ∀ a ∈ A, 0 < a) :
  ∃ B ⊆ A, (∀ b₁ b₂ ∈ B, b₁ ≠ b₂ → ¬(b₁ ∣ b₂ ∨ (b₂ ∣ b₁) ∨ (b₁ + 1 ∣ b₂ + 1) ∨ (b₂ + 1 ∣ b₁ + 1))) ∧
           (∀ a ∈ A, ∃ b ∈ B, a ∣ b ∨ (b + 1 ∣ a + 1)) :=
by
  sorry

end finite_set_independent_subset_dominance_l333_333491


namespace sum_of_digits_of_fraction_repeating_decimal_l333_333192

theorem sum_of_digits_of_fraction_repeating_decimal :
  (exists (c d : ℕ), (4 / 13 : ℚ) = c * 0.1 + d * 0.01 ∧ (c + d) = 3) :=
sorry

end sum_of_digits_of_fraction_repeating_decimal_l333_333192


namespace part_I_part_II_l333_333772

-- Part (I)
theorem part_I (n : ℕ) (hn : n > 0) (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (hₙ : ∀ n ∈ ℕ, a (n + 1) = Real.sqrt ((a n) ^ 2 - 2 * (a n) + 2) + 1):
  a n = Real.sqrt (n - 1) + 1 := sorry

-- Part (II)
theorem part_II (n : ℕ) (hn : n > 0) (a : ℕ → ℝ) (c : ℝ)
  (h₀ : a 1 = 1)
  (hₙ : ∀ n ∈ ℕ, a (n + 1) = Real.sqrt ((a n) ^ 2 - 2 * (a n) + 2) - 1)
  (hc : c = 1 / 4):
  (a (2 * n) < c ∧ c < a (2 * n + 1)) := sorry

end part_I_part_II_l333_333772


namespace cannot_form_triangle_l333_333153

noncomputable def point_on_segment (M N P : ℝ) : Prop :=
  M < P ∧ P < N

theorem cannot_form_triangle
  (a x : ℝ)
  (h1 : point_on_segment 0 a x)
  (h2 : ((a - x) / x) = (x / a)) :
  (let mpsq := (a - x) ^ 2;
       pnsq := x ^ 2;
       mnsq := a ^ 2 in
   mpsq + pnsq < mnsq) :=
by sorry

end cannot_form_triangle_l333_333153


namespace largest_seven_consecutive_3003_l333_333968

/-- The sum of seven consecutive positive integers is 3003. -/
def largest_of_seven_consecutive_integers (n : ℕ) (h : 7 * n + 21 = 3003) : ℕ :=
  let largest := n + 6
  largest

theorem largest_seven_consecutive_3003 : ∃ n, (7 * n + 21 = 3003) ∧ (largest_of_seven_consecutive_integers n (by sorry) = 432) :=
begin
  use 426,
  split,
  { exact (by norm_num), },
  { unfold largest_of_seven_consecutive_integers,
    norm_num, }
end

end largest_seven_consecutive_3003_l333_333968


namespace ordered_pair_l333_333449

variables {C D Q : Type} [AddCommGroup C] [AddCommGroup D] [AddCommGroup Q]
variables {ratio : ℚ} (h_ratio : 3/5 = ratio)
variables (point_Q : Q) (vector_C : C) (vector_D : D)

noncomputable def x : ℚ := 5 / 8
noncomputable def y : ℚ := 3 / 8

theorem ordered_pair (h : point_Q = (5 / 8) • vector_C + (3 / 8) • vector_D) : 
  (x, y) = (5 / 8, 3 / 8) :=
by
  rw [h]
  exact ⟨rfl, rfl⟩

end ordered_pair_l333_333449


namespace relationship_of_y_values_l333_333376

theorem relationship_of_y_values 
  (k : ℝ) (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_pos : k > 0) 
  (hA : y1 = k / x1) 
  (hB : y2 = k / x2) 
  (hC : y3 = k / x3) 
  (h_order : x1 < 0 ∧ 0 < x2 ∧ x2 < x3) : y1 < y3 ∧ y3 < y2 := 
by
  sorry

end relationship_of_y_values_l333_333376


namespace percentage_error_l333_333716

theorem percentage_error (e : ℝ) : (1 + e / 100)^2 = 1.1025 → e = 5.125 := 
by sorry

end percentage_error_l333_333716


namespace david_ate_more_than_emma_l333_333084

-- Definitions and conditions
def contestants : Nat := 8
def pies_david_ate : Nat := 8
def pies_emma_ate : Nat := 2
def pies_by_david (contestants pies_david_ate: Nat) : Prop := pies_david_ate = 8
def pies_by_emma (contestants pies_emma_ate: Nat) : Prop := pies_emma_ate = 2

-- Theorem statement
theorem david_ate_more_than_emma (contestants pies_david_ate pies_emma_ate : Nat) (h_david : pies_by_david contestants pies_david_ate) (h_emma : pies_by_emma contestants pies_emma_ate) : pies_david_ate - pies_emma_ate = 6 :=
by
  sorry

end david_ate_more_than_emma_l333_333084


namespace factorial_expression_value_l333_333595

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333595


namespace sum_prime_factors_l333_333990

-- Define the number 2010
def n : ℕ := 2010

-- Define the prime factors of 2010
def prime_factors_2010 : List ℕ := [2, 3, 5, 67]

-- Prove that the prime factors of 2010 sum to 77
theorem sum_prime_factors (n = 2010) : prime_factors_2010.sum = 77 :=
by sorry

end sum_prime_factors_l333_333990


namespace Q_at_7_l333_333892

noncomputable def Q (x : ℝ) : ℝ :=
  (3 * x^4 - 30 * x^3 + g * x^2 + h * x + k) *
  (4 * x^4 - 60 * x^3 + l * x^2 + m * x + n)

theorem Q_at_7 :
  (∀ x : ℂ, x ∈ {2, 3, 4, 5, 6}) →
  Q 7 = 28800 :=
sorry

end Q_at_7_l333_333892


namespace infinite_integer_solutions_l333_333933

variable (x : ℤ)

theorem infinite_integer_solutions (x : ℤ) : 
  ∃ (k : ℤ), ∀ n : ℤ, n > 2 → k = n :=
by {
  sorry
}

end infinite_integer_solutions_l333_333933


namespace quadratic_root_n_value_l333_333532

theorem quadratic_root_n_value :
  (∃ m p : ℕ, ∀ x : ℝ, (2 * x^2 - 5 * x - 4 = 0) ↔ (x = (m + sqrt 57) / p) ∨ (x = (m - sqrt 57) / p)) :=
sorry

end quadratic_root_n_value_l333_333532


namespace sqrt_factorial_sq_l333_333589

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333589


namespace nancy_weight_l333_333473

theorem nancy_weight (R E W : ℝ) (h1 : W = R - 2) (h2 : 0.05 * (W + 2) = E) (h3 : R - E = 64) : W ≈ 65.37 :=
by
  sorry

end nancy_weight_l333_333473


namespace angle_ABG_in_regular_octagon_l333_333485

theorem angle_ABG_in_regular_octagon (N : ℕ) (hN : N = 8) (regular_octagon : RegularPolygon N) : 
  angle ABG = 22.5 :=
by
  sorry

end angle_ABG_in_regular_octagon_l333_333485


namespace calculate_m_minus_n_plus_p_l333_333474

def cos2a (α : ℝ) : ℝ := 2 * (Real.cos α)^2 - 1
def cos4a (α : ℝ) : ℝ := 8 * (Real.cos α)^4 - 8 * (Real.cos α)^2 + 1
def cos6a (α : ℝ) : ℝ := 32 * (Real.cos α)^6 - 48 * (Real.cos α)^4 + 18 * (Real.cos α)^2 - 1
def cos8a (α : ℝ) : ℝ := 128 * (Real.cos α)^8 - 256 * (Real.cos α)^6 + 160 * (Real.cos α)^4 - 32 * (Real.cos α)^2 + 1
def cos10a (α m n p : ℝ) : ℝ := m * (Real.cos α)^10 - 1280 * (Real.cos α)^8 + 1120 * (Real.cos α)^6 + n * (Real.cos α)^4 + p * (Real.cos α)^2 - 1

theorem calculate_m_minus_n_plus_p
  (α m n p : ℝ)
  (h1 : cos2a α = Real.cos 2 * α)
  (h2 : cos4a α = Real.cos 4 * α)
  (h3 : cos6a α = Real.cos 6 * α)
  (h4 : cos8a α = Real.cos 8 * α)
  (h5 : cos10a α m n p = Real.cos 10 * α) :
  m - n + p = 962 := 
sorry

end calculate_m_minus_n_plus_p_l333_333474


namespace sum_of_ten_smallest_multiples_of_12_l333_333991

theorem sum_of_ten_smallest_multiples_of_12 : (Finset.range 10).sum (λ n, 12 * (n + 1)) = 660 := by
  sorry

end sum_of_ten_smallest_multiples_of_12_l333_333991


namespace space_exploration_funding_support_l333_333679

variables (men women supporters_surveyed total_surveyed : ℕ)
variables (percent_support_men percent_support_women overall_support : ℝ)

def num_men : ℕ := 200
def num_women : ℕ := 600
def percent_men_support : ℝ := 0.7
def percent_women_support : ℝ := 0.75

def overall_percentage : ℝ :=
  (percent_men_support * num_men + percent_women_support * num_women) / (num_men + num_women) * 100

theorem space_exploration_funding_support :
  overall_percentage = 73.75 :=
by
  sorry

end space_exploration_funding_support_l333_333679


namespace problem_solution_l333_333799

theorem problem_solution (a b x : ℝ) (h : b ≥ 0) :
  (b ∈ set.Ico 0 1 ∨ b ∈ set.Ioo 1 ∞) ↔
  (x = 0 ∨ (b = 1 ∧ x ∈ set.Icc (-2 : ℝ) 2)) :=
begin
  sorry
end

end problem_solution_l333_333799


namespace geometric_sequence_first_term_l333_333533

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 5) 
  (h2 : a * r^3 = 45) : 
  a = 5 / (3^(2/3)) := 
by
  -- proof steps to be filled here
  sorry

end geometric_sequence_first_term_l333_333533


namespace average_annual_growth_rate_reforested_area_in_2003_l333_333079

/- Problem Constants -/
def initial_area : ℝ := 637680
def converted_2000 : ℝ := 80000
def total_converted_2002 : ℝ := 291200
def ref_area_const : ℝ := 11.52

/- Problem Statement -/
theorem average_annual_growth_rate : ∃ x : ℝ, 80_000 * (1 + x) + 80_000 * (1 + x)^2 = 291200 - 80_000 := by
  sorry

theorem reforested_area_in_2003 : ∀ y : ℝ, (14.4 ≤ y) → ∃ x : ℝ, 0.25 ≤ x ∧ x ≤ 2 ∧ y = 11.52 * x + 11.52 := by
  sorry

end average_annual_growth_rate_reforested_area_in_2003_l333_333079


namespace circle_radius_l333_333759

theorem circle_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ (∀ x y, x^2 - 8*x + y^2 - 4*y + 16 = 0 → r = 2)) :=
sorry

end circle_radius_l333_333759


namespace coefficient_x4_expansion_eq_7_l333_333067

theorem coefficient_x4_expansion_eq_7 (a : ℝ) : 
  (∀ r : ℕ, 8 - (4 * r) / 3 = 4 → (a ^ r) * (Nat.choose 8 r) = 7) → a = 1 / 2 :=
by
  sorry

end coefficient_x4_expansion_eq_7_l333_333067


namespace quadratic_roots_equation_l333_333066

theorem quadratic_roots_equation (α β : ℝ) 
  (h1 : (α + β) / 2 = 8) 
  (h2 : Real.sqrt (α * β) = 15) : 
  x^2 - (α + β) * x + α * β = 0 ↔ x^2 - 16 * x + 225 = 0 := 
by
  sorry

end quadratic_roots_equation_l333_333066


namespace chemical_reactions_proof_l333_333014

-- Define the chemical reaction products
def reaction1 (CaCO3 moles : ℕ) : ℕ :=
  CaCO3 moles  -- 1 mole of CaCO3 produces 1 mole of CaO

def reaction2 (CaCO3 moles : ℕ) (HCl moles : ℕ) : ℕ :=
  0  -- No CaCO3 remaining means no CaCl2 is produced

-- Define the molar masses
def molar_mass_CaO : ℝ := 56.08
def molar_mass_CO2 : ℝ := 44.01

-- Calculate the weight of the products from reaction 1
def weight_reaction1 (CaCO3 moles : ℕ) : ℝ :=
  CaCO3 moles * molar_mass_CaO + CaCO3 moles * molar_mass_CO2

theorem chemical_reactions_proof :
  (reaction1 8 = 8) ∧
  (reaction2 0 12 = 0) ∧
  (weight_reaction1 8 = 800.72) :=
by
  -- Statements are true by conditions and problem definition, hence proof is omitted here.
  sorry

end chemical_reactions_proof_l333_333014


namespace valid_transformation_b_l333_333230

theorem valid_transformation_b (a b : ℚ) : ((-a - b) / (a + b) = -1) := sorry

end valid_transformation_b_l333_333230


namespace recycling_cans_l333_333501

noncomputable def total_new_cans (initial_cans : ℕ) (recycle_factor : ℕ → ℕ) : ℕ :=
let rec aux (remaining_cans total_new : ℕ) : ℕ :=
  let new_cans := recycle_factor remaining_cans in
  if new_cans = 0 then total_new else aux new_cans (total_new + new_cans)
in aux initial_cans 0

def recycle_factor (old_cans : ℕ) : ℕ :=
(old_cans * 2) / 6

theorem recycling_cans : total_new_cans 388 recycle_factor = 193 :=
by
  sorry

end recycling_cans_l333_333501


namespace max_value_fraction_l333_333056

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end max_value_fraction_l333_333056


namespace general_formula_a_n_sum_b_n_l333_333784

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

def b_n (n : ℕ) : ℕ := a_n n * Int.sqrt(3^(a_n n + 1))

def T_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, b_n (i + 1)

theorem general_formula_a_n :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
begin
  sorry
end

theorem sum_b_n :
  ∀ n : ℕ, T_n n = 3 + (n - 1) * 3^(n + 1) :=
begin
  sorry
end

end general_formula_a_n_sum_b_n_l333_333784


namespace distance_between_parallel_lines_l333_333384

theorem distance_between_parallel_lines (A B C1 C2 : ℝ) (hA : A = 2) (hB : B = 4)
  (hC1 : C1 = -8) (hC2 : C2 = 7) : 
  (|C2 - C1| / (Real.sqrt (A^2 + B^2)) = 3 * Real.sqrt 5 / 2) :=
by
  rw [hA, hB, hC1, hC2]
  sorry

end distance_between_parallel_lines_l333_333384


namespace train_cross_platform_time_l333_333668

/-- Definitions of the problem -/
def L_train : ℝ := 300
def L_platform : ℝ := 366.67
def T_pole : ℝ := 18
def v : ℝ := L_train / T_pole
def T_platform : ℝ := (L_train + L_platform) / v

/-- Theorem to be proved -/
theorem train_cross_platform_time :
  T_platform = 40 := 
by
  sorry

end train_cross_platform_time_l333_333668


namespace sqrt_factorial_mul_squared_l333_333605

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333605


namespace sqrt_factorial_sq_l333_333586

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333586


namespace number_of_teams_l333_333675

theorem number_of_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : x = 8 :=
sorry

end number_of_teams_l333_333675


namespace negation_of_exists_irrational_square_rational_l333_333187

open Classical

-- Definitions and assumptions
def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

theorem negation_of_exists_irrational_square_rational :
  (¬ (∃ (x : ℝ), irrational x ∧ rational (x * x))) ↔
  (∀ (x : ℝ), irrational x → ¬ rational (x * x)) :=
begin
  sorry,
end

end negation_of_exists_irrational_square_rational_l333_333187


namespace increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l333_333354

variables {k: ℝ} -- Thermal conductivity
variables {A A': ℝ} -- Original and increased contact area
variables {dT: ℝ} -- Temperature difference
variables {dx: ℝ} -- Thickness of the skillet handle

-- Define the heat flow rate according to Fourier's law of heat conduction
def heat_flow_rate (k: ℝ) (A: ℝ) (dT: ℝ) (dx: ℝ) : ℝ :=
  -k * A * (dT / dx)

theorem increased_contact_area_increases_heat_flow 
  (h₁: A' > A) -- Increased contact area
  (h₂: dT / dx > 0) -- Positive temperature gradient
  : heat_flow_rate k A' dT dx > heat_flow_rate k A dT dx :=
by
  -- Proof to show that increased area increases heat flow rate
  sorry

theorem handle_felt_hotter_no_thermodynamic_contradiction 
  (h₁: A' > A)
  (h₂: dT / dx > 0)
  : ¬(heat_flow_rate k A' dT dx contradicts thermodynamic laws) :=
by
  -- Proof to show no contradiction with the laws of thermodynamics
  sorry

end increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l333_333354


namespace part_one_solution_set_part_two_min_value_l333_333398

noncomputable def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem part_one_solution_set :
  {x : ℝ | f x ≤ 1} = set.Ici (-1) :=
by
  sorry

theorem part_two_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 3) :
  Real.Inf ({3 / a + a / b} : set ℝ) = 3 :=
by
  sorry

end part_one_solution_set_part_two_min_value_l333_333398


namespace convert_quadratic_l333_333732

theorem convert_quadratic :
  ∀ x : ℝ, (x^2 + 2*x + 4) = ((x + 1)^2 + 3) :=
by
  sorry

end convert_quadratic_l333_333732


namespace solution_quadratic_inequality_l333_333794

noncomputable def prop_inequality (a b : ℝ) : Prop :=
  (∀ x : ℝ, x ∈ set.Icc (-2) 1 → a * x ^ 2 - x + b ≥ 0) ∧ (∀ x : ℝ, (b * x ^ 2 - x + a ≤ 0) ↔ (x ∈ set.Icc (-1 / 2) 1))

theorem solution_quadratic_inequality (a b : ℝ) (h_roots : -2, 1 are roots of a * x ^ 2 - x + b = 0) 
  (h_solution_set : (∀ x : ℝ, x ∈ set.Icc (-2) 1 → a * x ^ 2 - x + b ≥ 0)) :
  (∀ x : ℝ, (b * x ^ 2 - x + a ≤ 0) ↔ (x ∈ set.Icc (-1 / 2) 1)) :=
sorry

end solution_quadratic_inequality_l333_333794


namespace train_speed_clicks_l333_333528

theorem train_speed_clicks (x : ℝ) (rail_length_feet : ℝ := 40) (clicks_per_mile : ℝ := 5280/ 40) :
  15 ≤ (2400/5280) * 60  * clicks_per_mile ∧ (2400/5280) * 60 * clicks_per_mile ≤ 30 :=
by {
  sorry
}

end train_speed_clicks_l333_333528


namespace area_of_circle_B_l333_333727

noncomputable def radius_circle_A (area : ℝ) : ℝ :=
  real.sqrt (area / real.pi)

noncomputable def radius_circle_B (radius_A : ℝ) : ℝ :=
  radius_A / 2

noncomputable def area_circle (radius : ℝ) : ℝ :=
  real.pi * radius ^ 2

theorem area_of_circle_B :
  let radius_A := radius_circle_A 16 in
  let radius_B := radius_circle_B radius_A in
  area_circle radius_B = 4 := by
  sorry

end area_of_circle_B_l333_333727


namespace quadratic_c_over_b_l333_333954

theorem quadratic_c_over_b :
  ∃ (b c : ℤ), (x^2 + 500 * x + 1000 = (x + b)^2 + c) ∧ (c / b = -246) :=
by sorry

end quadratic_c_over_b_l333_333954


namespace totalPoundsOfFoodConsumed_l333_333720

def maxConsumptionPerGuest : ℝ := 2.5
def minNumberOfGuests : ℕ := 165

theorem totalPoundsOfFoodConsumed : 
    maxConsumptionPerGuest * (minNumberOfGuests : ℝ) = 412.5 := by
  sorry

end totalPoundsOfFoodConsumed_l333_333720


namespace find_digit_A_l333_333121

theorem find_digit_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) (h4 : (100 * A + 10 * M + C) * (A + M + C) = 2008) : 
  A = 2 :=
sorry

end find_digit_A_l333_333121


namespace passengers_count_l333_333146

def total_passengers (P : ℕ) :=
  let females := 0.40 * P
  let first_class := 0.10 * P
  let coach_class := 0.90 * P
  let males_first_class := (1/3 : ℝ) * first_class
  let females_first_class := (2/3 : ℝ) * first_class
  females - females_first_class = 40

theorem passengers_count : ∃ (P : ℕ), total_passengers P ∧ P = 120 :=
by
  sorry

end passengers_count_l333_333146


namespace sqrt_of_nine_l333_333967

theorem sqrt_of_nine : sqrt 9 = 3 ∨ sqrt 9 = -3 := 
sorry

end sqrt_of_nine_l333_333967


namespace independence_test_l333_333861

-- Define the statement and the conditions
def probability_related (H0 : Prop) (K2 : ℝ → ℝ) (P : Set ℝ → ℝ) := 
  (H0 → P ({x | K2 x ≥ 6.635}) ≈ 0.010 → 
  P ({x | ¬H0}) ≈ 0.99)

-- Assume H0 that variables X and Y are unrelated
axiom H0_unrelated : Prop

-- Function K^2 representing test statistic
noncomputable def K2 : ℝ → ℝ := sorry

-- Function P representing probability measure
noncomputable def P : Set ℝ → ℝ := sorry

-- Statement translating the math problem
theorem independence_test :
  probability_related H0_unrelated K2 P :=
sorry

end independence_test_l333_333861


namespace no_contradiction_to_thermodynamics_l333_333353

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l333_333353


namespace distinct_primes_sum_reciprocal_l333_333316

open Classical

theorem distinct_primes_sum_reciprocal (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (hineq: (1 / p : ℚ) + (1 / q) + (1 / r) ≥ 1) 
  : (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨
    (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) := 
sorry

end distinct_primes_sum_reciprocal_l333_333316


namespace find_ordered_pair_l333_333883

noncomputable def find_ab : ℝ × ℝ :=
  let a := 23
  let b := -11
  (a, b)

theorem find_ordered_pair (a b : ℝ) :
  let root1 := a + 3 * complex.I
  let root2 := b + 6 * complex.I in
  (root1 + root2 = 12 + 15 * complex.I) →
  (root1 * root2 = 52 + 105 * complex.I) →
  (a, b) = (23, -11) :=
by 
  sorry

end find_ordered_pair_l333_333883


namespace find_N_l333_333059

theorem find_N (N p q : ℝ) 
  (h1 : N / p = 4) 
  (h2 : N / q = 18) 
  (h3 : p - q = 0.5833333333333334) :
  N = 3 := 
sorry

end find_N_l333_333059


namespace sqrt_factorial_product_squared_l333_333636

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333636


namespace perpendicular_implies_lambda_neg_one_l333_333402

variable (λ : ℝ)
variable (a : ℝ × ℝ := (1, -3))
variable (b : ℝ × ℝ := (4, -2))

theorem perpendicular_implies_lambda_neg_one
  (h : (λ * a.1 + b.1, λ * a.2 + b.2).fst * a.1 + (λ * a.1 + b.1, λ * a.2 + b.2).snd * a.2 = 0) : 
  λ = -1 := 
  sorry

end perpendicular_implies_lambda_neg_one_l333_333402


namespace no_contradiction_to_thermodynamics_l333_333352

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l333_333352


namespace possible_values_of_c_l333_333126

-- Definition of c(S) based on the problem conditions
def c (S : String) (m : ℕ) : ℕ := sorry

-- Condition: m > 1
variable {m : ℕ} (hm : m > 1)

-- Goal: To prove the possible values that c(S) can take
theorem possible_values_of_c (S : String) : ∃ n : ℕ, c S m = 0 ∨ c S m = 2^n :=
sorry

end possible_values_of_c_l333_333126


namespace hundredth_digit_of_fraction_l333_333219

theorem hundredth_digit_of_fraction :
  let decimal_extension := "17".cycle.take 100
  decimal_extension.get_last = '7' :=
by
  sorry

end hundredth_digit_of_fraction_l333_333219


namespace probability_A_more_than_B_sum_m_n_l333_333161

noncomputable def prob_A_more_than_B : ℚ :=
  0.6 + 0.4 * (1 / 2) * (1 - (63 / 512))

theorem probability_A_more_than_B : prob_A_more_than_B = 779 / 1024 := sorry

theorem sum_m_n : 779 + 1024 = 1803 := sorry

end probability_A_more_than_B_sum_m_n_l333_333161


namespace trapezoid_with_slopes_p_plus_q_l333_333931

-- Define the points A and D
def A : ℤ × ℤ := (-2, 1)
def D : ℤ × ℤ := (-1, 3)

-- Define an isosceles trapezoid as a structure
structure IsoscelesTrapezoid (A B C D : ℤ × ℤ) : Prop :=
  (integers : A.1.toInt = A.1 ∧ A.2.toInt = A.2 ∧ D.1.toInt = D.1 ∧ D.2.toInt = D.2 ∧ B.1.toInt = B.1 ∧ B.2.toInt = B.2 ∧ C.1.toInt = C.1 ∧ C.2.toInt = C.2)
  (no_horizontal_or_vertical_sides : A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ D.1)
  (parallel_sides : ¬(A.2 = B.2) ∧ ¬(C.2 = D.2))

-- Define valid slopes
def valid_slopes : List ℚ := [2, -2, 1, -1, (1/3), (-1/3)]

-- Calculate the sum of absolute values of the slopes
def sum_of_abs_slopes (slopes : List ℚ) : ℚ :=
  slopes.map (λ x, abs x) |>.sum

-- Prove the problem statement
theorem trapezoid_with_slopes (A B C D : ℤ × ℤ) (trapezoid : IsoscelesTrapezoid A B C D)
  (slopes : List ℚ) (valid_slopes : slopes = [2, -2, 1, -1, (1/3), (-1/3)]) : 
  sum_of_abs_slopes slopes = 20 / 3 :=
by
  sorry

theorem p_plus_q : 20 + 3 = 23 := by
  rfl

end trapezoid_with_slopes_p_plus_q_l333_333931


namespace find_a7_l333_333434

-- We define a geometric sequence with initial term a₁ and common ratio q
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ
| 0     => a₁
| (n+1) => geometric_sequence a₁ q n * q

-- Define the conditions given in the problem
axiom a3_def (a₁ q : ℝ) : geometric_sequence a₁ q 2 = 4
axiom condition (a₁ q : ℝ) : geometric_sequence a₁ q 6 - 2 * geometric_sequence a₁ q 4 = 32

-- The statement to be proved
theorem find_a7 (a₁ q : ℝ) (h1 : a3_def a₁ q) (h2 : condition a₁ q) : 
  geometric_sequence a₁ q 6 = 64 := sorry

end find_a7_l333_333434


namespace triangle_angle_B_l333_333849

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end triangle_angle_B_l333_333849


namespace original_quantity_of_ghee_l333_333432

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end original_quantity_of_ghee_l333_333432


namespace sequence_sum_periodic_l333_333088

theorem sequence_sum_periodic (a : ℕ → ℕ) (a1 a8 : ℕ) :
  a 1 = 11 →
  a 8 = 12 →
  (∀ i, 1 ≤ i → i ≤ 6 → a i + a (i + 1) + a (i + 2) = 50) →
  (a 1 = 11 ∧ a 2 = 12 ∧ a 3 = 27 ∧ a 4 = 11 ∧ a 5 = 12 ∧ a 6 = 27 ∧ a 7 = 11 ∧ a 8 = 12) :=
by
  intros h1 h8 hsum
  sorry

end sequence_sum_periodic_l333_333088


namespace S_11_equals_neg55_l333_333859

noncomputable def sum_of_arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a_1 + (n - 1) * d) / 2

theorem S_11_equals_neg55 (a_1 d : ℝ) (a_5 a_7 : ℝ) :
  (a_5 + a_7 = -10) → 
  (a_5 = a_1 + 4 * d) → 
  (a_7 = a_1 + 6 * d) → 
  (sum_of_arithmetic_sequence a_1 d 11 = -55) :=
by { intros, sorry }

end S_11_equals_neg55_l333_333859


namespace part_one_part_two_l333_333022

-- Part 1
theorem part_one (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, e ≤ x → f x = a * x + x * Real.log x) 
  (h_increasing : ∀ x y : ℝ, e ≤ x → x ≤ y → y ≤ +∞ → f x ≤ f y) : 
  a ≥ -2 :=
sorry

-- Part 2
theorem part_two (k : ℤ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, 1 < x → f x = x + x * Real.log x) 
  (h_ineq : ∀ x : ℝ, 1 < x → k * (x - 1) < f x) : 
  k ≤ 3 :=
sorry

end part_one_part_two_l333_333022


namespace solve_system_of_equations_l333_333503

theorem solve_system_of_equations (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h1 : log 4 x + log 4 y = 1 + log 4 9)
  (h2 : x + y = 20) :
  (x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18) :=
by
  -- Proof here, but we'll use sorry for now to skip the proof.
  sorry

end solve_system_of_equations_l333_333503


namespace complex_magnitude_eq_one_l333_333008

open Complex

theorem complex_magnitude_eq_one
  (z : ℂ) (p : ℕ)
  (h : 11 * z^10 + 10 * Complex.I * z^p + 10 * Complex.I * z - 11 = 0) :
  |z| = 1 :=
sorry

end complex_magnitude_eq_one_l333_333008


namespace angle_EAF_equals_2_angle_BEC_l333_333789

-- Definitions based on conditions
variables {A B C D E F : Point}
variable {l : Line}
variable {circle1 circle2 : Circle}

-- Assuming our initial conditions
axiom tangent_to_circle1 : Tangent l circle1
axiom chord_of_circle2 : Chord l circle2
axiom connect_AE : Line_through A E
axiom connect_BE : Line_through B E
axiom DF_perp_BE : Perpendicular D F (Line_through B E)
axiom connect_AF : Line_through A F

theorem angle_EAF_equals_2_angle_BEC :
  ∠EAF = 2 * ∠BEC :=
sorry

end angle_EAF_equals_2_angle_BEC_l333_333789


namespace sin_addition_l333_333770

variables (α β : ℝ)

-- Given conditions
axiom sin_alpha (h : sin α = -3 / 5)
axiom cos_beta (h : cos β = 1)

-- The math proof problem statement to prove
theorem sin_addition : sin (α + β) = -3 / 5 :=
by sorry

end sin_addition_l333_333770


namespace sum_of_integers_square_eq_224_plus_self_sum_solution_of_square_eq_224_plus_self_l333_333964

theorem sum_of_integers_square_eq_224_plus_self (x : ℤ) (h : x^2 = 224 + x) : 
  x = 16 ∨ x = -14 :=
begin
  sorry
end

theorem sum_solution_of_square_eq_224_plus_self :
  (∑ x in {y : ℤ | y^2 = 224 + y}.to_finset, x) = 2 :=
begin
  have h1 : 16 ∈ {y : ℤ | y^2 = 224 + y}.to_finset, 
  from sorry, -- We need to show that 16 is in the set of solutions
  have h2 : -14 ∈ {y : ℤ | y^2 = 224 + y}.to_finset, 
  from sorry, -- We need to show that -14 is in the set of solutions
  have h3 : {y : ℤ | y^2 = 224 + y}.to_finset = {16, -14}, 
  from sorry, -- We need to show that these are the only solutions
  rw h3,
  norm_num, -- This concludes the sum is 2
end

end sum_of_integers_square_eq_224_plus_self_sum_solution_of_square_eq_224_plus_self_l333_333964


namespace length_B_to_B_l333_333978

noncomputable def distance (p q : ℝ×ℝ) :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

def point_B : ℝ×ℝ := (1, -5)
def point_B' : ℝ×ℝ := (-1, -5)

theorem length_B_to_B' : distance point_B point_B' = 2 := by
  sorry

end length_B_to_B_l333_333978


namespace find_new_bottle_caps_l333_333309

theorem find_new_bottle_caps (initial caps_thrown current : ℕ) (h_initial : initial = 69)
  (h_thrown : caps_thrown = 60) (h_current : current = 67) :
  ∃ n, initial - caps_thrown + n = current ∧ n = 58 := by
sorry

end find_new_bottle_caps_l333_333309


namespace tissues_available_for_regular_use_l333_333542

theorem tissues_available_for_regular_use :
  let 
    students_group1 := 15,
    students_group2 := 20,
    students_group3 := 18,
    students_group4 := 22,
    students_group5 := 25,
    tissues_per_student := 70,
    reserve_percentage := 0.3
  in
  let 
    total_students := students_group1 + students_group2 + students_group3 + students_group4 + students_group5,
    total_tissues := total_students * tissues_per_student,
    reserve_tissues := total_tissues * reserve_percentage,
    tissues_for_regular_use := total_tissues - reserve_tissues
  in
  tissues_for_regular_use = 4900 :=
by
  sorry

end tissues_available_for_regular_use_l333_333542


namespace smallest_number_am_median_l333_333207

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end smallest_number_am_median_l333_333207


namespace original_quantity_of_ghee_l333_333431

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end original_quantity_of_ghee_l333_333431


namespace batsman_average_increase_l333_333669

theorem batsman_average_increase 
    (runs_in_17th_inning : ℕ)
    (avg_after_17th : ℕ) :
    runs_in_17th_inning = 82 → avg_after_17th = 34 → 
    ∃ (avg_increase : ℕ), avg_increase = 3 :=
by {
    intros h1 h2,
    use 3,
    sorry
}

end batsman_average_increase_l333_333669


namespace zeros_at_end_of_product_l333_333819

open Nat

def prime_factors (n : ℕ) : List ℕ := sorry -- assume we have a way to get the prime factors

theorem zeros_at_end_of_product :
  prime_factors 30 = [2, 3, 5] →
  prime_factors 450 = [2, 2, 3, 3, 5, 5] →
  (30 * 450).factor_multiplicity 10 = 3 :=
by
  sorry

end zeros_at_end_of_product_l333_333819


namespace find_price_saturday_l333_333908

variables (price_friday : ℕ) (visitors_saturday : ℕ) (visitors_friday : ℕ)  
          (revenue_friday : ℕ) (revenue_saturday : ℕ) (k : ℕ)

-- Conditions
def conditions :=
  price_friday = 9 ∧
  visitors_saturday = 200 ∧
  visitors_saturday = 2 * visitors_friday ∧
  revenue_saturday = (4 * revenue_friday) / 3 ∧
  revenue_friday = price_friday * visitors_friday ∧
  revenue_saturday = k * visitors_saturday

-- Question and expected answer
theorem find_price_saturday (h : conditions) : k = 6 :=
sorry

end find_price_saturday_l333_333908


namespace legos_needed_l333_333108

theorem legos_needed (total_legos : ℕ) (legos_per_airplane : ℕ) (airplanes_needed : ℕ) 
  (current_legos : ℕ) : total_legos = 2 * legos_per_airplane → current_legos = 400 → 
  airplanes_needed = 2 → legos_needed = total_legos - current_legos → 
  total_legos = 480 → legos_needed = 80 :=
by
  sorry

end legos_needed_l333_333108


namespace lucy_packs_sold_l333_333156

theorem lucy_packs_sold (robyn_sold : ℕ) (total_sold : ℕ) (h1 : robyn_sold = 47) (h2 : total_sold = 76) : ∃ lucy_sold : ℕ, lucy_sold = total_sold - robyn_sold ∧ lucy_sold = 29 :=
by
  use 29
  split
  . exact rfl
  . sorry

end lucy_packs_sold_l333_333156


namespace evaluate_101_times_101_l333_333747

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end evaluate_101_times_101_l333_333747


namespace domain_correct_l333_333517

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | (4 - x^2 > 0) ∧ (1 - tan x ≥ 0)}

theorem domain_correct :
  domain_of_function = 
    {x : ℝ | -π/2 < x ∧ x ≤ π/4} ∪ 
    {x : ℝ | π/2 < x ∧ x < 2} :=
by sorry

end domain_correct_l333_333517


namespace find_magnitude_l333_333168

-- Define the given condition in Lean
variable (w : ℂ) (h : w^2 = 45 - 21 * complex.I)

-- Define the statement of the problem in Lean
theorem find_magnitude : |w| = real.sqrt (real.sqrt 2466) := by sorry

end find_magnitude_l333_333168


namespace meeting_point_l333_333555

-- Given conditions
def speed_A_kmph : ℝ := 162
def speed_B_kmph : ℝ := 120
def distance_km : ℝ := 450

-- Conversion factors
def km_to_m (km : ℝ) : ℝ := km * 1000
def hr_to_s : ℝ := 3600

-- Speed conversions
def speed_A_mps : ℝ := (speed_A_kmph * 1000) / 3600
def speed_B_mps : ℝ := (speed_B_kmph * 1000) / 3600

-- Relative speed
def relative_speed_mps : ℝ := speed_A_mps + speed_B_mps

-- Distance conversion
def distance_m : ℝ := km_to_m distance_km

-- Time to meet
def time_to_meet_s : ℝ := distance_m / relative_speed_mps

-- Distance covered by Train A when they meet
def distance_covered_by_A_km : ℝ := (speed_A_mps * time_to_meet_s) / 1000

theorem meeting_point :
  speed_A_mps = 45 ∧
  speed_B_mps ≈ 33.33 ∧
  distance_covered_by_A_km ≈ 258.5691 :=
by
  -- Step through the calculations given the problem conditions
  sorry

end meeting_point_l333_333555


namespace select_1011_numbers_l333_333490

theorem select_1011_numbers :
  ∃! (S : Finset ℕ), S.card = 1011 ∧ (∀ a b ∈ S, a ≠ b → a + b ≠ 2021 ∧ a + b ≠ 2022) ∧ (∀ x ∈ S, x ∈ Finset.range 2022) :=
sorry

end select_1011_numbers_l333_333490


namespace sin_C_in_right_triangle_l333_333854

theorem sin_C_in_right_triangle (A B C : ℝ) (hABC : ∠A + ∠B + ∠C = 180)
  (hA : sin A = 3/5) (hB : sin B = 1) (right_angle_B : ∠B = 90) : 
  sin C = 4/5 := 
begin
  sorry
end

end sin_C_in_right_triangle_l333_333854


namespace find_BC_l333_333098

-- Given data and conditions for the geometric problem
variables (A B C D O : Type) 
variables (OA OB OC OD : ℝ) -- Assuming lengths are real numbers
variables (r a : ℝ)
variables (AB AD BC: ℝ)

-- Geometric constraints
axiom AB_perp_BC : (BC - AB = 0)
axiom BC_perp_CD : (BC - CD = 0)
axiom AC_perp_BD : (AC - BD = 0)
axiom AB_length : AB = real.sqrt 11
axiom AD_length : AD = real.sqrt 1001

-- Final statement to prove
theorem find_BC : BC = real.sqrt 110 :=
sorry

end find_BC_l333_333098


namespace problem_9_1_problem_9_2_l333_333246

theorem problem_9_1 (n : ℕ) (h : n > 2) : ∀ a_n : ℕ, a_n = 10^n - 2 → 
  (¬ (∃ x y : ℕ, a_n = x^2 + y^2) ∧ ¬ (∃ x y : ℕ, a_n = x^2 - y^2)) → a_n = 10^n - 2 :=
sorry

theorem problem_9_2 : ∃ n : ℕ, (∀ d ∈ Nat.digits 10 n, d ≥ 0 ∧ d < 10) ∧ 
  Nat.isPerfectSquare (∑ d in Nat.digits 10 66, d^2) ∧ n = 66 :=
sorry

end problem_9_1_problem_9_2_l333_333246


namespace problem1_l333_333362

open Real

noncomputable def a : ℝ := (16 / 5) ^ (2 / 5) - (-2) ^ 0 - (9 / 4) ^ (-3 / 4) + (6 / 5) ^ (-3)
noncomputable def b : ℝ := (log (sqrt[2] (64 / 2))) / (log 2) + log 10 + log 5 + (log 3) / (log 5)

theorem problem1 : a + b = 283 / 120 :=
by
  sorry

end problem1_l333_333362


namespace distance_AD_between_20_and_21_l333_333914

noncomputable def distance_AD (A B C D : ℝ × ℝ) : ℝ :=
  let AB := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 in
  let BC := (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 in
  let CD := (C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 in
  let AD := AB + CD in
  real.sqrt AD

theorem distance_AD_between_20_and_21 (A B C D : ℝ × ℝ) :
  B = (A.1 + 5, A.2) → C = (B.1, B.2 + 5) → D = (C.1, C.2 + 15) →
  real.sqrt ((A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2) = real.sqrt 425 →
  20 < real.sqrt 425 ∧ real.sqrt 425 < 21 :=
by
  intros h1 h2 h3 h4
  sorry

end distance_AD_between_20_and_21_l333_333914


namespace tangent_line_y_intercept_l333_333256

def circle1Center: ℝ × ℝ := (3, 0)
def circle1Radius: ℝ := 3
def circle2Center: ℝ × ℝ := (7, 0)
def circle2Radius: ℝ := 2

theorem tangent_line_y_intercept
    (tangent_line: ℝ × ℝ -> ℝ) 
    (P : tangent_line (circle1Center.1, circle1Center.2 + circle1Radius) = 0) -- Tangent condition for Circle 1
    (Q : tangent_line (circle2Center.1, circle2Center.2 + circle2Radius) = 0) -- Tangent condition for Circle 2
    :
    tangent_line (0, 4.5) = 0 := 
sorry

end tangent_line_y_intercept_l333_333256


namespace product_of_four_consecutive_numbers_l333_333952

theorem product_of_four_consecutive_numbers (n : ℕ) (h : n = 6) :
  n * (n + 1) * (n + 2) * (n + 3) = 3024 :=
by
  rw h
  -- skip the proof
  sorry

end product_of_four_consecutive_numbers_l333_333952


namespace largest_number_divisible_by_six_l333_333988

theorem largest_number_divisible_by_six : ∃ n < 9000, (6 ∣ n) ∧ ∀ m < 9000, 6 ∣ m → m <= n :=
  exists.intro 8994
  (and.intro
    (by norm_num)
    (by
      intros m hm hdiv
      sorry))

end largest_number_divisible_by_six_l333_333988


namespace fourth_largest_divisor_l333_333935

def n : ℕ := 1234560000
def prime_factors (n : ℕ) : List ℕ := [2, 5, 3, 643] -- Simplified representation of prime factors
def factors := [1234560000, 617280000, 308640000, 154320000] -- Calculation based on the solution steps

theorem fourth_largest_divisor :
  n = 2^10 * 3 * 5^4 * 643 →
  factors.nth 3 = some 154320000 :=
by
  intros h,
  -- Proof to be filled in
  sorry

end fourth_largest_divisor_l333_333935


namespace factorial_expression_value_l333_333593

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333593


namespace compare_A_B_l333_333875

variable (x : ℝ) (n : ℕ) (h_pos : x > 0)

def A : ℝ := x^n + x^(-n)
def B : ℝ := x^(n-1) + x^(1-n)

theorem compare_A_B (hx : 0 < x) : A x n ≥ B x n := sorry

end compare_A_B_l333_333875


namespace find_x_given_parallel_vectors_l333_333359

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_x_given_parallel_vectors (x : ℝ) :
  vector_parallel (x, 2) (1, 6) → x = 1 / 3 :=
by
  intro h
  cases h with k hk
  have h1 : x = k := by rw [hk]
  have h2 : 2 = 6 * k := by rw [hk]
  sorry

end find_x_given_parallel_vectors_l333_333359


namespace sum_of_integers_square_eq_224_plus_self_sum_solution_of_square_eq_224_plus_self_l333_333963

theorem sum_of_integers_square_eq_224_plus_self (x : ℤ) (h : x^2 = 224 + x) : 
  x = 16 ∨ x = -14 :=
begin
  sorry
end

theorem sum_solution_of_square_eq_224_plus_self :
  (∑ x in {y : ℤ | y^2 = 224 + y}.to_finset, x) = 2 :=
begin
  have h1 : 16 ∈ {y : ℤ | y^2 = 224 + y}.to_finset, 
  from sorry, -- We need to show that 16 is in the set of solutions
  have h2 : -14 ∈ {y : ℤ | y^2 = 224 + y}.to_finset, 
  from sorry, -- We need to show that -14 is in the set of solutions
  have h3 : {y : ℤ | y^2 = 224 + y}.to_finset = {16, -14}, 
  from sorry, -- We need to show that these are the only solutions
  rw h3,
  norm_num, -- This concludes the sum is 2
end

end sum_of_integers_square_eq_224_plus_self_sum_solution_of_square_eq_224_plus_self_l333_333963


namespace AF_over_FB_l333_333437

variables {A B C D F P : Type}
variables [AffineSpace ℝ (A B C D F P)]

-- Condition: AP/PD = 4/3
def ratio_AP_PD (P A D : ℝ) : Prop := (A - P) / (P - D) = 4 / 3

-- Condition: FP/PC = 1/2
def ratio_FP_PC (P F C : ℝ) : Prop := (F - P) / (P - C) = 1 / 2

-- Theorem: AF/FB = 5/9 given the conditions
theorem AF_over_FB (P A D F C B : ℝ) (h1 : ratio_AP_PD P A D) (h2 : ratio_FP_PC P F C) : (F - A) / (B - F) = 5 / 9 := by
  sorry

end AF_over_FB_l333_333437


namespace non_zero_weight_zero_l333_333866

noncomputable def vertex_weight_problem (G : Type*) [graph G] (a : G → G → ℤ) : Prop :=
  ∀ (v : G), (∑ u in G, a u v - ∑ w in G, a v w) % 2012 = 0

noncomputable def exists_arc_weights (G : Type*) [graph G] (a : G → G → ℤ) : Prop :=
  ∃ (x : G → G → ℤ), 
    (∀ u v, x u v ≠ 0 ∧ |x u v| ≤ 2012 ∧ ¬ (x u v % 2012 = 0)) ∧ 
    vertex_weight_problem G x

theorem non_zero_weight_zero
  (G : Type*) [graph G] (a : G → G → ℤ) 
  (h₁ : ∀ (u v : G), ¬(a u v % 2012 = 0)) 
  (h₂ : vertex_weight_problem G a) :
  exists_arc_weights G a :=
sorry

end non_zero_weight_zero_l333_333866


namespace factorial_expression_value_l333_333601

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333601


namespace history_paper_pages_l333_333504

theorem history_paper_pages (days: ℕ) (pages_per_day: ℕ) (h₁: days = 3) (h₂: pages_per_day = 27) : days * pages_per_day = 81 := 
by
  sorry

end history_paper_pages_l333_333504


namespace fold_length_not_possible_l333_333828

theorem fold_length_not_possible {a b : ℕ} (h₁ : a = 6) (h₂ : b = 5) :
  ¬ (∃ l : ℝ, l = 8 ∧ l ≤ real.sqrt (a^2 + b^2)) :=
by {
  -- Proof steps would go here
  sorry
}

end fold_length_not_possible_l333_333828


namespace sum_of_integers_square_plus_224_equals_l333_333966

theorem sum_of_integers_square_plus_224_equals (x : ℤ) :
  (x^2 = x + 224) → (({x : ℤ | x^2 = x + 224}.sum id) = 2) :=
sorry

end sum_of_integers_square_plus_224_equals_l333_333966


namespace range_of_quadratic_function_l333_333530

theorem range_of_quadratic_function :
  ∀ (x : ℝ), x ∈ Icc (-1 : ℝ) 2 → let y := -x^2 + 3*x + 1 in y ∈ Icc (-3 : ℝ) (13/4 : ℝ) :=
sorry

end range_of_quadratic_function_l333_333530


namespace sqrt_factorial_squared_l333_333569

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333569


namespace general_term_a_b_geometric_sum_c_n_l333_333792

noncomputable section

-- Definitions for arithmetic sequence
def a (n : ℕ) : ℝ :=
  let a1 := 2
  let d := 4
  a1 + d * (n - 1)
  
-- Given conditions
axiom a_2_eq_6 : a 2 = 6
axiom a_5_eq_18 : a 5 = 18

-- Definitions for sequence b and its sum T
def T (n : ℕ) : ℝ := sorry -- T's exact formula not given in problem
def b (n : ℕ) : ℝ := sorry -- b's exact formula to be determined from given conditions

axiom Tn_plus_b_n_eq_1 (n : ℕ) : T n + 1/2 * b n = 1

-- Proving the general term for a_n
theorem general_term_a (n : ℕ) : a n = 4 * n - 2 :=
sorry

-- Proving that b_n is a geometric sequence
theorem b_geometric (b1 r : ℝ) (n : ℕ) (h1 : b 1 = b1) (h2 : ∀ n, b (n + 1) = r * b n) : 
  b1 = 2/3 ∧ r = 1/3 :=
sorry

-- Proving the sum of the first n terms of sequence c_n
def c (n : ℕ) : ℝ := a n * b n

theorem sum_c_n (Sn : ℕ → ℝ) (n : ℕ) :
  Sn n = ∑ i in finset.range (n + 1), c i → 
  Sn n = 4 - 4 * (n + 1) * (1 / 3) ^ n :=
sorry

end general_term_a_b_geometric_sum_c_n_l333_333792


namespace games_lost_percentage_l333_333191

theorem games_lost_percentage (x : ℕ) (hx : x ≠ 0)
  (won_games lost_games total_games : ℕ)
  (hwon : won_games = 11 * x) (hlost : lost_games = 4 * x)
  (htotal : total_games = won_games + lost_games) :
  (lost_games * 100 / total_games).toNat = 27 :=
by
  sorry

end games_lost_percentage_l333_333191


namespace solution_set_of_inequality_l333_333790

/-- Given the conditions:
 1) The domain of the function f(x) is ℝ
 2) f(2 - x) = f(2 + x)
 3) f(5) = 2
 4) For all x₁, x₂ ∈ (-∞, 2], when x₁ ≠ x₂, the expression (f(x₁) - f(x₂)) / (x₁ - x₂) > 0 -/
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x ∈ set.univ) 
  (h2 : ∀ x, f (2 - x) = f (2 + x)) 
  (h3 : f 5 = 2) 
  (h4 : ∀ x₁ x₂ : ℝ, x₁ ≤ 2 → x₂ ≤ 2 → x₁ ≠ x₂ → ((f x₁ - f x₂) / (x₁ - x₂) > 0)) :
  {x : ℝ | f x + 4 * x + 3 > x ^ 2} = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

end solution_set_of_inequality_l333_333790


namespace compute_sqrt_factorial_square_l333_333624

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333624


namespace quadratic_condition_l333_333512

theorem quadratic_condition (a b c : ℝ) : (a ≠ 0) ↔ ∃ (x : ℝ), ax^2 + bx + c = 0 :=
by sorry

end quadratic_condition_l333_333512


namespace one_third_percent_of_200_l333_333911

theorem one_third_percent_of_200 : ((1206 / 3) / 200) * 100 = 201 := by
  sorry

end one_third_percent_of_200_l333_333911


namespace original_number_l333_333268

theorem original_number (N : ℤ) (h1 : 4 * N = 108) (h2 : odd N) (h3 : 9 ∣ N) : N = 27 :=
by
  sorry

end original_number_l333_333268


namespace transformed_sine_function_l333_333936

theorem transformed_sine_function :
  ∀ (x : ℝ), (∃ f : ℝ → ℝ, f = λ x, sin x) ∧
  (∃ g : ℝ → ℝ, g = λ x, sin (x - π/8)) ∧
  (∃ h : ℝ → ℝ, h = λ x, sin (x / 2 - π / 8)) →
  (h = λ x, sin (x / 2 - π / 8)) :=
sorry

end transformed_sine_function_l333_333936


namespace min_cardinality_sets_l333_333785

open Nat

theorem min_cardinality_sets (n : ℕ) (A B : Finset ℕ) (h_n : 4 ≤ n)
  (h_A : ∀ a ∈ A, a ∈ (Finset.range (n + 1)).filter (λ x, 1 ≤ x))
  (h_B : ∀ b ∈ B, b ∈ (Finset.range (n + 1)).filter (λ x, 1 ≤ x))
  (h_perfect_square : ∀ a b, a ∈ A → b ∈ B → ∃ k : ℕ, a * b + 1 = k * k) :
  min (A.card) (B.card) ≤ log 2 n :=
sorry

end min_cardinality_sets_l333_333785


namespace transformed_roots_l333_333119

theorem transformed_roots {p q r s : ℝ} (hpqrs : Set {z : ℝ | z^4 - 5 * z^2 + 6 = 0} ⊆ {p, q, r, s}) :
  (by apply (polynomial.associated_eq.mp : (p + q) * (r + s) = 0) { 
    let p_q_r_s_roots : list ℝ := [p, q, r, s],
    let transformations := [p + q, (r + s), p + r, (q + s), p + s, (q + r)],
    (by ring : (rhs.instances.global.zero_smul @']) : ℝ}) :=
sorry

end transformed_roots_l333_333119


namespace monotonically_increasing_interval_l333_333525

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^3

theorem monotonically_increasing_interval : ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

end monotonically_increasing_interval_l333_333525


namespace exists_finite_group_with_normal_subgroup_GT_Aut_l333_333319

noncomputable def finite_group_G (n : ℕ) : Type := sorry -- Specific construction details omitted
noncomputable def normal_subgroup_H (n : ℕ) : Type := sorry -- Specific construction details omitted

def Aut_G (n : ℕ) : ℕ := sorry -- Number of automorphisms of G
def Aut_H (n : ℕ) : ℕ := sorry -- Number of automorphisms of H

theorem exists_finite_group_with_normal_subgroup_GT_Aut (n : ℕ) :
  ∃ G H, finite_group_G n = G ∧ normal_subgroup_H n = H ∧ Aut_H n > Aut_G n := sorry

end exists_finite_group_with_normal_subgroup_GT_Aut_l333_333319


namespace num_multiples_6_not_12_lt_300_l333_333035

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l333_333035


namespace least_Nice_Group_l333_333841
-- Definitions for conditions
def isNiceGroup (students : List α) : Prop :=
  -- A definition to determine if a group of students is nice based on the problem conditions
  sorry

def distinctMarks (students : List α) : Prop :=
  -- A definition to ensure any two students have distinct marks in all four areas
  sorry

-- The theorem statement
theorem least_Nice_Group {α : Type*} (students : List α) [inhabited α] :
  ∃ N, (∀ (s : List α), length s = N → distinctMarks s → existsNiceGroupOfTen s) :=
  exists.intro 730 (by sorry)

end least_Nice_Group_l333_333841


namespace factorial_expression_value_l333_333597

theorem factorial_expression_value : (sqrt (5! * 4!))^2 = 2880 := sorry

end factorial_expression_value_l333_333597


namespace smallest_positive_period_of_y_l333_333317

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.sin (-x / 2 + Real.pi / 4)

-- Statement we need to prove
theorem smallest_positive_period_of_y :
  ∃ T > 0, ∀ x : ℝ, y (x + T) = y x ∧ T = 4 * Real.pi := sorry

end smallest_positive_period_of_y_l333_333317


namespace surface_area_of_cube_given_sphere_surface_area_l333_333833

noncomputable def edge_length_of_cube (sphere_surface_area : ℝ) : ℝ :=
  let a_square := 2
  Real.sqrt a_square

def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

theorem surface_area_of_cube_given_sphere_surface_area (sphere_surface_area : ℝ) :
  sphere_surface_area = 6 * Real.pi → 
  surface_area_of_cube (edge_length_of_cube sphere_surface_area) = 12 :=
by
  sorry

end surface_area_of_cube_given_sphere_surface_area_l333_333833


namespace integral_f_l333_333361

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 1 then x^2
else if h : 1 < x ∧ x ≤ Real.exp 1 then 1 / x
else 0

theorem integral_f : ∫ x in 0..Real.exp 1, f x = 4 / 3 :=
by
  sorry

end integral_f_l333_333361


namespace smallest_piece_length_l333_333283

theorem smallest_piece_length (x : ℕ) :
  (9 - x) + (14 - x) ≤ (16 - x) → x ≥ 7 :=
by
  sorry

end smallest_piece_length_l333_333283


namespace stock_price_after_two_years_l333_333740

def initial_price : ℝ := 120

def first_year_increase (p : ℝ) : ℝ := p * 2

def second_year_decrease (p : ℝ) : ℝ := p * 0.30

def final_price (initial : ℝ) : ℝ :=
  let after_first_year := first_year_increase initial
  after_first_year - second_year_decrease after_first_year

theorem stock_price_after_two_years : final_price initial_price = 168 :=
by
  sorry

end stock_price_after_two_years_l333_333740


namespace fraction_to_decimal_l333_333659

/-- The decimal equivalent of 1/4 is 0.25. -/
theorem fraction_to_decimal : (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end fraction_to_decimal_l333_333659


namespace stans_average_speed_l333_333505

noncomputable def average_speed (distance1 distance2 distance3 : ℝ) (time1_hrs time1_mins time2 time3_hrs time3_mins : ℝ) : ℝ :=
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1_hrs + time1_mins / 60 + time2 + time3_hrs + time3_mins / 60
  total_distance / total_time

theorem stans_average_speed  :
  average_speed 350 420 330 5 40 7 5 30 = 60.54 :=
by
  -- sorry block indicates missing proof
  sorry

end stans_average_speed_l333_333505


namespace factory_material_equation_correct_l333_333754

variable (a b x : ℝ)
variable (h_a : a = 180)
variable (h_b : b = 120)
variable (h_condition : (a - 2 * x) - (b + x) = 30)

theorem factory_material_equation_correct : (180 - 2 * x) - (120 + x) = 30 := by
  rw [←h_a, ←h_b]
  exact h_condition

end factory_material_equation_correct_l333_333754


namespace complex_number_problem_l333_333379

noncomputable def z : ℂ := 1 + 2 * Complex.I

theorem complex_number_problem (i : ℂ) (h_i : i = Complex.I) (h_eq : (2 / (1 + i) = Complex.conj z + i)) : 
  z = 1 + 2 * i :=
by
  sorry

end complex_number_problem_l333_333379


namespace significant_figures_accuracy_120_million_l333_333713

def is_significant_figures (n : ℕ) (x : ℕ) : Prop :=
x = 2

def is_accurate_to (p : ℕ) (x : ℕ) : Prop :=
x = 6

theorem significant_figures_accuracy_120_million :
  ∀ n, (n = 120000000) → (is_significant_figures n 2) ∧ (is_accurate_to 1000000 n 6) :=
  by
    intro n
    assume h : n = 120000000
    split
    case left =>
      unfold is_significant_figures
      rw h
      exact rfl
    case right =>
      unfold is_accurate_to
      rw h
      exact rfl

end significant_figures_accuracy_120_million_l333_333713


namespace people_in_room_l333_333407

variable (total_chairs occupied_chairs people_present : ℕ)
variable (h1 : total_chairs = 28)
variable (h2 : occupied_chairs = 14)
variable (h3 : (2 / 3 : ℚ) * people_present = 14)
variable (h4 : total_chairs = 2 * occupied_chairs)

theorem people_in_room : people_present = 21 := 
by 
  --proof will be here
  sorry

end people_in_room_l333_333407


namespace dot_product_magnitude_l333_333453

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

variables (c d : ℝ^3)
variable (theta : ℝ)

-- Conditions
def norm_c : real := 3
def norm_d : real := 4
def norm_cross_c_d : real := 6

-- Proof statement
theorem dot_product_magnitude : 
  ‖c‖ = norm_c → ‖d‖ = norm_d → ‖c ×ᵥ d‖ = norm_cross_c_d → | (inner_product_space.dot c d) | = 6 * real.sqrt 3 :=
begin
  intros,
  sorry
end

end dot_product_magnitude_l333_333453


namespace factorial_sqrt_sq_l333_333612

theorem factorial_sqrt_sq (h : ∀ (n : ℕ), ∃ (m : ℕ), nat.factorial n = m) : 
  (real.sqrt (nat.factorial 5 * nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end factorial_sqrt_sq_l333_333612


namespace no_two_friends_contribute_less_than_one_third_l333_333342

def totalCost (a1 a2 a3 a4 a5 : ℝ) : ℝ := a1 + a2 + a3 + a4 + a5

theorem no_two_friends_contribute_less_than_one_third 
  (a1 a2 a3 a4 a5 : ℝ) (T : ℝ) 
  (h_sum : totalCost a1 a2 a3 a4 a5 = T) :
  ¬ (∀ i j : ℕ, i ≠ j → (i < 5) → (j < 5) → (a1 + a2 + a3 + a4 + a5)[i] + (a1 + a2 + a3 + a4 + a5)[j] < T / 3) := 
sorry

end no_two_friends_contribute_less_than_one_third_l333_333342


namespace sqrt_factorial_mul_square_l333_333640

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333640


namespace average_value_of_T_is_45_25_l333_333166

noncomputable def average_value_of_T (T : Finset ℕ) (m b₁ bₘ : ℕ) : ℚ :=
  let S := (T \ {b₁, bₘ}).sum id
  (42 * (m - 1) + b₁ + 81) / m

theorem average_value_of_T_is_45_25 (T : Finset ℕ) (m b₁ bₘ : ℕ) 
  (h1 : ∑ i in (T \ {bₘ}).sum id = 42 * (m - 1))
  (h2 : ∑ i in (T \ {b₁, bₘ}).sum id = 46 * (m - 2))
  (h3 : ∑ i in ((T \ {b₁}) ∪ {bₘ}).sum id = 49 * (m - 1))
  (h4 : bₘ = b₁ + 81) :
  average_value_of_T T m b₁ bₘ = 45.25 := 
by
  sorry

end average_value_of_T_is_45_25_l333_333166


namespace mitzi_money_left_l333_333143

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end mitzi_money_left_l333_333143


namespace translation_upward_6_units_l333_333092

def l1 : ℝ → ℝ := λ x, -2 * x - 2
def l2 : ℝ → ℝ := λ x, -2 * x + 4

theorem translation_upward_6_units :
  ∃ k : ℝ, (∀ x : ℝ, l2 x = l1 x + k) ∧ k = 6 :=
by {
  use 6,
  split,
  {
    intro x,
    simp [l1, l2],
    ring,
  },
  {
    refl,
  }
}

end translation_upward_6_units_l333_333092


namespace number_of_subsets_A_union_B_l333_333897

def A : Set ℕ := {x | x^2 - 2 * x = 0}
def B : Set ℕ := {0, 1}

theorem number_of_subsets_A_union_B :
  Finset.card (Finset.powerset (Finset.filter (λ x, x ∈ (A ∪ B)) (Finset.range 3))) = 8 := by
sorry

end number_of_subsets_A_union_B_l333_333897


namespace first_player_wins_l333_333427

-- Define the initial condition for the placement of knights on the board
def initial_knight_positions : Prop :=
  ∃ (p1 p2 : (ℕ × ℕ)), p1 = (0, 0) ∧ p2 = (7, 7)

-- Define the rules for removing squares in terms of the game's progression.
def remove_square (board : set (ℕ × ℕ)) (square : ℕ × ℕ) : set (ℕ × ℕ) :=
  board \ {square}

-- Define the condition that a knight can move from one position to another
def knight_can_reach (pos1 pos2 : ℕ × ℕ) (board : set (ℕ × ℕ)) : Prop :=
  -- This would need actual implementation of knight move reachability, simplified here:
  ∀ k ∈ board, (k = pos2) → (∃ p ∈ board, p = pos1)

-- Define the losing condition: no path between two knights
def losing_condition (pos1 pos2 : ℕ × ℕ) (board : set (ℕ × ℕ)) : Prop :=
  ¬ knight_can_reach pos1 pos2 board

-- Statement of the problem: the first player can guarantee a win
theorem first_player_wins : ∀ (board : set (ℕ × ℕ)),
  initial_knight_positions →
  (∀ p1 p2, ¬ losing_condition p1 p2 board) →
  (∃ k1 k2, ∀ board', remove_square board k1 ⊆ board' → remove_square board' k2 ⊆ board → 
    losing_condition k1 k2 board) :=
sorry

end first_player_wins_l333_333427


namespace sqrt_factorial_mul_squared_l333_333610

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333610


namespace range_of_a_l333_333800

theorem range_of_a (a : ℝ) (x : ℝ) : (-8 ≤ a ∧ a ≤ 0) → (a * x^2 - a * x - 2 ≤ 0) :=
begin
  sorry
end

end range_of_a_l333_333800


namespace reeya_fourth_subject_score_l333_333917

theorem reeya_fourth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℕ) (n : ℕ)
  (h_avg : avg = 75) (h_n : n = 4) (h_s1 : s1 = 65) (h_s2 : s2 = 67) (h_s3 : s3 = 76)
  (h_total_sum : avg * n = s1 + s2 + s3 + s4) : s4 = 92 := by
  sorry

end reeya_fourth_subject_score_l333_333917


namespace profit_A_after_upgrade_profit_B_constrained_l333_333677

-- Part Ⅰ
theorem profit_A_after_upgrade (x : ℝ) (h : x^2 - 300 * x ≤ 0) : 0 < x ∧ x ≤ 300 := sorry

-- Part Ⅱ
theorem profit_B_constrained (a x : ℝ) (h1 : a ≤ (x/125 + 500/x + 3/2)) (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := sorry

end profit_A_after_upgrade_profit_B_constrained_l333_333677


namespace increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l333_333356

variables {k: ℝ} -- Thermal conductivity
variables {A A': ℝ} -- Original and increased contact area
variables {dT: ℝ} -- Temperature difference
variables {dx: ℝ} -- Thickness of the skillet handle

-- Define the heat flow rate according to Fourier's law of heat conduction
def heat_flow_rate (k: ℝ) (A: ℝ) (dT: ℝ) (dx: ℝ) : ℝ :=
  -k * A * (dT / dx)

theorem increased_contact_area_increases_heat_flow 
  (h₁: A' > A) -- Increased contact area
  (h₂: dT / dx > 0) -- Positive temperature gradient
  : heat_flow_rate k A' dT dx > heat_flow_rate k A dT dx :=
by
  -- Proof to show that increased area increases heat flow rate
  sorry

theorem handle_felt_hotter_no_thermodynamic_contradiction 
  (h₁: A' > A)
  (h₂: dT / dx > 0)
  : ¬(heat_flow_rate k A' dT dx contradicts thermodynamic laws) :=
by
  -- Proof to show no contradiction with the laws of thermodynamics
  sorry

end increased_contact_area_increases_heat_flow_handle_felt_hotter_no_thermodynamic_contradiction_l333_333356


namespace inevitable_answer_l333_333409

-- Define the conditions
inductive Guest
| sane_human
| mindless_werewolf

-- Define the questions
def question1 : Guest → Prop :=
λ guest, guest = Guest.sane_human ∨ guest = Guest.mindless_werewolf

def question2 : Guest → Prop :=
λ guest, (guest = Guest.sane_human) ↔ ("бaл" = "да")

-- Prove the equivalence
theorem inevitable_answer (guest : Guest) : question1 guest ∨ question2 guest → ("бaл" = "бaл") :=
by sorry

end inevitable_answer_l333_333409


namespace fold_lines_hyperbola_l333_333254

theorem fold_lines_hyperbola (R a : ℝ) (hR : R > 0) (ha : 0 < a ∧ a < R) : 
  ∀ (x y : ℝ), 
  (∃ A' : ℝ × ℝ, 
    (A'.1 ^ 2 + A'.2 ^ 2 = R ^ 2 ∧ 
     ((x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) = 1))) → 
  ((x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) = 1) :=
begin
  sorry
end

end fold_lines_hyperbola_l333_333254


namespace second_trial_amount_is_809_l333_333993

-- The problem conditions
def element_range := (500 : ℝ, 1000 : ℝ)
def golden_ratio_method := 0.618

-- Definition for the second trial amount calculation using golden ratio method
def second_trial_amount (element_range : ℝ × ℝ) (method_coef : ℝ) : ℝ :=
  element_range.1 + (element_range.2 - element_range.1) * method_coef

-- Proof statement
theorem second_trial_amount_is_809 :
  second_trial_amount element_range golden_ratio_method = 809 :=
by
  -- Proof to be provided
  sorry

end second_trial_amount_is_809_l333_333993


namespace at_most_one_perfect_square_l333_333133

theorem at_most_one_perfect_square (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n ^ 3 + 103) →
  (∃ n1, ∃ n2, a n1 = k1^2 ∧ a n2 = k2^2) → n1 = n2 
    ∨ (∀ n, a n ≠ k1^2) 
    ∨ (∀ n, a n ≠ k2^2) :=
sorry

end at_most_one_perfect_square_l333_333133


namespace factorization_correct_l333_333228

theorem factorization_correct (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by 
  sorry

end factorization_correct_l333_333228


namespace sunday_saturday_ratio_is_two_to_one_l333_333744

-- Define the conditions as given in the problem
def total_pages : ℕ := 360
def saturday_morning_read : ℕ := 40
def saturday_night_read : ℕ := 10
def remaining_pages : ℕ := 210

-- Define Ethan's total pages read so far
def total_read : ℕ := total_pages - remaining_pages

-- Define pages read on Saturday
def saturday_total_read : ℕ := saturday_morning_read + saturday_night_read

-- Define pages read on Sunday
def sunday_total_read : ℕ := total_read - saturday_total_read

-- Define the ratio of pages read on Sunday to pages read on Saturday
def sunday_to_saturday_ratio : ℕ := sunday_total_read / saturday_total_read

-- Theorem statement: ratio of pages read on Sunday to pages read on Saturday is 2:1
theorem sunday_saturday_ratio_is_two_to_one : sunday_to_saturday_ratio = 2 :=
by
  -- This part should contain the detailed proof
  sorry

end sunday_saturday_ratio_is_two_to_one_l333_333744


namespace incorrect_statements_l333_333876

-- Define planes and a line
variables {α β : Plane} {l : Line}

-- Define conditions
def condition_1 : Prop := ∀ l α β, l ⊥ α ∧ α ⊥ β → ¬(l ⊂ β)
def condition_2 : Prop := ∀ l α β, l ∥ α ∧ α ∥ β → ¬(l ⊂ β)
def condition_3 : Prop := ∀ l α β, l ⊥ α ∧ α ∥ β → l ⊥ β
def condition_4 : Prop := ∀ l α β, l ∥ α ∧ α ⊥ β → ¬(l ⊥ β)

-- Define the theorem
theorem incorrect_statements : condition_1 ∧ condition_2 ∧ condition_4 :=
by
  sorry

end incorrect_statements_l333_333876


namespace concurrency_of_lines_l333_333888

open EuclideanGeometry
open Real

theorem concurrency_of_lines
  (A B C D P Q : EuclideanGeometry.Point)
  (h_parallel : parallelogram A B C D)
  (hP_on_BC : collinear B C P)
  (hQ_on_CD : collinear C D Q)
  (h_sim : similar (triangle.mk A B P) (triangle.mk A D Q)) :
  concurrent (line_through B D) (line_through P Q) (tangent_to_circumcircle (triangle.mk A P Q)) :=
  sorry

end concurrency_of_lines_l333_333888


namespace cookie_revenue_l333_333496

theorem cookie_revenue :
  let robyn_day1_packs := 25
  let robyn_day1_price := 4.0
  let lucy_day1_packs := 17
  let lucy_day1_price := 5.0
  let robyn_day2_packs := 15
  let robyn_day2_price := 3.5
  let lucy_day2_packs := 9
  let lucy_day2_price := 4.5
  let robyn_day3_packs := 23
  let robyn_day3_price := 4.5
  let lucy_day3_packs := 20
  let lucy_day3_price := 3.5
  let robyn_day1_revenue := robyn_day1_packs * robyn_day1_price
  let lucy_day1_revenue := lucy_day1_packs * lucy_day1_price
  let robyn_day2_revenue := robyn_day2_packs * robyn_day2_price
  let lucy_day2_revenue := lucy_day2_packs * lucy_day2_price
  let robyn_day3_revenue := robyn_day3_packs * robyn_day3_price
  let lucy_day3_revenue := lucy_day3_packs * lucy_day3_price
  let robyn_total_revenue := robyn_day1_revenue + robyn_day2_revenue + robyn_day3_revenue
  let lucy_total_revenue := lucy_day1_revenue + lucy_day2_revenue + lucy_day3_revenue
  let total_revenue := robyn_total_revenue + lucy_total_revenue
  total_revenue = 451.5 := 
by
  sorry

end cookie_revenue_l333_333496


namespace count_multiples_of_6_not_12_lt_300_l333_333040

theorem count_multiples_of_6_not_12_lt_300 : 
  {N : ℕ // 0 < N ∧ N < 300 ∧ (6 ∣ N) ∧ ¬(12 ∣ N)}.toFinset.card = 25 := sorry

end count_multiples_of_6_not_12_lt_300_l333_333040


namespace compute_sqrt_factorial_square_l333_333620

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333620


namespace odd_square_free_count_l333_333048

theorem odd_square_free_count : 
  ∃ n : ℕ, n = 64 ∧ ∀ k : ℕ, (1 < k ∧ k < 200 ∧ k % 2 = 1 ∧ 
  (∀ m : ℕ, m * m ∣ k → m = 1)) ↔ k ∈ {3, 5, 7, ..., 199} :=
sorry

end odd_square_free_count_l333_333048


namespace sqrt_factorial_product_squared_l333_333635

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333635


namespace sum_of_first_five_betas_l333_333306

noncomputable def Q (x : ℂ) : ℂ := (∑ i in finset.range 21, x^i)^2 - x^19

theorem sum_of_first_five_betas :
  let roots := {z : ℂ | Q z = 0}
  -- assuming roots are ordered and extracted as described in the problem.
  let betas := [1/21, 1/2, 2/21, 1/21, 3/21] in
  list.sum betas = 9 / 7 :=
by
  sorry

end sum_of_first_five_betas_l333_333306


namespace series_equivalence_l333_333889

variable (c d : ℝ)

noncomputable def sum_of_series (x : ℝ) := x / (x + 2 * d) + x / (x + 2 * d)^2 + x / (x + 2 * d)^3 + (⨏ : ℝ)

theorem series_equivalence 
  (h : c / d + c / d^2 + c / d^3 + (⨏ : ℝ) = 6) : 
  sum_of_series c d = (6 * d - 6) / (8 * d - 7) := 
sorry

end series_equivalence_l333_333889


namespace relationship_between_m_and_n_l333_333129

variables (a b : ℝ)
#check (0 : ℝ) -- accessing real numbers

noncomputable def m : ℝ := sqrt a - sqrt b
noncomputable def n : ℝ := sqrt (a - b)

theorem relationship_between_m_and_n (h₀ : a > b) (h₁ : b > 0) : m a b < n a b := 
by sorry

end relationship_between_m_and_n_l333_333129


namespace distance_from_O_to_AB_half_CD_l333_333662

-- Definitions for points, line segments, and distances
variables {O A B C D : Point}
variables {AB CD : Line}
variables {circumcircle : Circle}

-- Definitions for properties of the quadrilateral
def cyclic_quadrilateral (A B C D : Point) (circumcircle : Circle) : Prop :=
  On_Circle A circumcircle ∧ On_Circle B circumcircle ∧ 
  On_Circle C circumcircle ∧ On_Circle D circumcircle

def perpendicular_diagonals (A B C D : Point) : Prop :=
  Perpendicular (Line A C) (Line B D)

def circumcenter (O : Point) (A B C D : Point) (circumcircle : Circle) : Prop :=
  Center circumsircle = O

-- The main theorem
theorem distance_from_O_to_AB_half_CD
  (h_quadrilateral : cyclic_quadrilateral A B C D circumcircle)
  (h_perpendicular : perpendicular_diagonals A B C D)
  (h_circumcenter : circumcenter O A B C D circumcircle) :
  distance (orthogonal_projection O AB) O = (length CD) / 2 :=
sorry

end distance_from_O_to_AB_half_CD_l333_333662


namespace work_efficiency_ratio_l333_333998

theorem work_efficiency_ratio (a b k : ℝ) (ha : a = k * b) (hb : b = 1/15)
  (hab : a + b = 1/5) : k = 2 :=
by sorry

end work_efficiency_ratio_l333_333998


namespace compute_sqrt_factorial_square_l333_333626

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333626


namespace sequence_sum_periodic_l333_333087

theorem sequence_sum_periodic (a : ℕ → ℕ) (a1 a8 : ℕ) :
  a 1 = 11 →
  a 8 = 12 →
  (∀ i, 1 ≤ i → i ≤ 6 → a i + a (i + 1) + a (i + 2) = 50) →
  (a 1 = 11 ∧ a 2 = 12 ∧ a 3 = 27 ∧ a 4 = 11 ∧ a 5 = 12 ∧ a 6 = 27 ∧ a 7 = 11 ∧ a 8 = 12) :=
by
  intros h1 h8 hsum
  sorry

end sequence_sum_periodic_l333_333087


namespace first_player_wins_l333_333428

-- Define the initial condition for the placement of knights on the board
def initial_knight_positions : Prop :=
  ∃ (p1 p2 : (ℕ × ℕ)), p1 = (0, 0) ∧ p2 = (7, 7)

-- Define the rules for removing squares in terms of the game's progression.
def remove_square (board : set (ℕ × ℕ)) (square : ℕ × ℕ) : set (ℕ × ℕ) :=
  board \ {square}

-- Define the condition that a knight can move from one position to another
def knight_can_reach (pos1 pos2 : ℕ × ℕ) (board : set (ℕ × ℕ)) : Prop :=
  -- This would need actual implementation of knight move reachability, simplified here:
  ∀ k ∈ board, (k = pos2) → (∃ p ∈ board, p = pos1)

-- Define the losing condition: no path between two knights
def losing_condition (pos1 pos2 : ℕ × ℕ) (board : set (ℕ × ℕ)) : Prop :=
  ¬ knight_can_reach pos1 pos2 board

-- Statement of the problem: the first player can guarantee a win
theorem first_player_wins : ∀ (board : set (ℕ × ℕ)),
  initial_knight_positions →
  (∀ p1 p2, ¬ losing_condition p1 p2 board) →
  (∃ k1 k2, ∀ board', remove_square board k1 ⊆ board' → remove_square board' k2 ⊆ board → 
    losing_condition k1 k2 board) :=
sorry

end first_player_wins_l333_333428


namespace distinct_real_roots_range_l333_333781

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 - a = 0) ∧ (x2^2 - 4*x2 - a = 0)) ↔ a > -4 :=
by
  sorry

end distinct_real_roots_range_l333_333781


namespace lana_nickels_per_stack_l333_333871

theorem lana_nickels_per_stack (total_nickels stacks : ℕ) (h1 : total_nickels = 72) (h2 : stacks = 9) :
  total_nickels / stacks = 8 :=
by {
  rw [h1, h2],
  exact Nat.div_eq_of_eq_mul (by norm_num) (by norm_num)
}

end lana_nickels_per_stack_l333_333871


namespace factorization_left_to_right_l333_333225

-- Definitions (conditions)
def exprD_lhs : ℝ → ℝ := λ x, x^2 - 9
def exprD_rhs : ℝ → ℝ := λ x, (x + 3) * (x - 3)

-- Statement
theorem factorization_left_to_right (x : ℝ) :
  exprD_lhs x = exprD_rhs x := 
by sorry

end factorization_left_to_right_l333_333225


namespace graph_not_in_fourth_quadrant_l333_333647

-- Define the power function y = x^α and the quadrants in the plane
def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The statement to prove
theorem graph_not_in_fourth_quadrant (α : ℝ) : ¬ ∃ x : ℝ, fourth_quadrant x (power_function α x) :=
sorry

end graph_not_in_fourth_quadrant_l333_333647


namespace foci_distance_equal_l333_333518

open Real

noncomputable def foci_distance_of_ellipse (a b : ℝ) : ℝ :=
  2 * sqrt (a^2 - b^2)

theorem foci_distance_equal :
  ∀ (k : ℝ),
    0 < k → k < 9 →
    (foci_distance_of_ellipse 5 3 = 8 ∧ 
     foci_distance_of_ellipse (sqrt (25 - k)) (sqrt (9 - k)) = 8) :=
begin
  intros k h1 h2,
  split,
  { -- First ellipse foci distance
    simp [foci_distance_of_ellipse, sqr_abs, sqrt, abs_of_nonneg],
  },
  { -- Second ellipse foci distance
    have h3 : sqrt (25 - k) > 0, by { sorry },
    have h4 : sqrt (9 - k) > 0, by { sorry },
    rw ← sqr_abs,
    sorry,
  },
end

end foci_distance_equal_l333_333518


namespace king_to_h8_l333_333942

theorem king_to_h8 :
  ∃ (move : ℕ × ℕ → ℕ × ℕ), 
    (∀ initial first_move, initial = (1, 1) 
      → first_move = (2, 2) 
      → (move first_move = (8, 8) ∨ ∃ next_move, move first_move = next_move ∧ move next_move = (8, 8))) :=
begin
  sorry
end

end king_to_h8_l333_333942


namespace division_remainder_l333_333648

theorem division_remainder (A : ℕ) :
  13 = 7 * 1 + A → A = 6 :=
by
  intro h
  rw [mul_one, add_comm] at h
  rw [h]
  rfl

end division_remainder_l333_333648


namespace has_zero_in_interval_l333_333459

noncomputable def f (x : ℝ) : ℝ := 3^x - x^2

theorem has_zero_in_interval :
  ∃ x ∈ set.Ioc (-1 : ℝ) (0 : ℝ), f x = 0 :=
sorry

end has_zero_in_interval_l333_333459


namespace no_infinite_arith_seq_with_digit_sum_arith_seq_l333_333439

-- Define the problem as a Lean proposition.
theorem no_infinite_arith_seq_with_digit_sum_arith_seq :
  ¬ (∃ (a : ℕ) (d : ℕ), 
      (∀ n m : ℕ, n ≠ m → (a + n * d) ≠ (a + m * d)) ∧  -- The sequence is composed of distinct terms
      (∀ n : ℕ, n > 0 → ∃ k : ℕ, a + n * d = k) ∧  -- The sequence is infinite
      (∃ b c : ℕ, ∀ n : ℕ, sum_of_digits (a + n * d) = b + n * c)) :=  -- The sum of digits forms an arithmetic sequence
sorry

end no_infinite_arith_seq_with_digit_sum_arith_seq_l333_333439


namespace difference_of_largest_and_smallest_divisible_by_3_l333_333562

theorem difference_of_largest_and_smallest_divisible_by_3 : 
  let digits := [1, 2, 3, 4, 5]
  let largest := 54321
  let smallest := 12345
  let diff := largest - smallest
  (∑ d in digits, d) % 3 = 0 → diff = 41976 :=
by
  -- Details of the proof are omitted 
  sorry

end difference_of_largest_and_smallest_divisible_by_3_l333_333562


namespace digit_divisibility_by_7_l333_333743

theorem digit_divisibility_by_7 (d : ℕ) (h : d < 10) : (10000 + 100 * d + 10) % 7 = 0 ↔ d = 5 :=
by
  sorry

end digit_divisibility_by_7_l333_333743


namespace speed_of_man_upstream_l333_333265

def speed_of_man_in_still_water : ℝ := 32
def speed_of_man_downstream : ℝ := 39

theorem speed_of_man_upstream (V_m V_s : ℝ) :
  V_m = speed_of_man_in_still_water →
  V_m + V_s = speed_of_man_downstream →
  V_m - V_s = 25 :=
sorry

end speed_of_man_upstream_l333_333265


namespace sqrt_factorial_product_squared_l333_333576

theorem sqrt_factorial_product_squared (n m : ℕ) (h1: n = 5) (h2: m = 4) : (Real.sqrt (Nat.fact n * Nat.fact m))^2 = 2880 := by
  sorry

end sqrt_factorial_product_squared_l333_333576


namespace number_of_boys_and_girls_l333_333295

theorem number_of_boys_and_girls (b g : ℕ) 
    (h1 : ∀ n : ℕ, (n ≥ 1) → ∃ (a_n : ℕ), a_n = 2 * n + 1)
    (h2 : (2 * b + 1 = g))
    : b = (g - 1) / 2 :=
by
  sorry

end number_of_boys_and_girls_l333_333295


namespace ticket_cost_possible_values_l333_333705

theorem ticket_cost_possible_values : 
  (∃ x : ℕ, ∀ n : ℕ, (n = 60 ∨ n = 90) → n % x = 0) →
  (finset.card (finset.filter (λ d, d ∣ 30) (finset.range 31)) = 8) :=
begin
  intro hx,
  have h_gcd : nat.gcd 60 90 = 30, sorry,
  rw ←h_gcd,
  apply congr_arg,
  ext,
  split;
  intro h,
  {
    simp only [finset.mem_filter, finset.mem_range] at h,
    exact h.1,
  },
  {
    simp only [finset.mem_filter, finset.mem_range],
    split,
    { exact h },
    {
      rw nat.gcd_dvd_left 60 90,
      rw nat.gcd_dvd_right 60 90,
    }
  },
  exact nat.dvd_gcd h hx,
end

end ticket_cost_possible_values_l333_333705


namespace probability_top_four_cards_is_2_over_95_l333_333702

noncomputable def probability_top_four_hearts :
  ℚ :=
  ((13 * 12 * 11 * 10) : ℚ) / ((52 * 51 * 50 * 49) : ℚ)

theorem probability_top_four_cards_is_2_over_95 :
  probability_top_four_hearts = 2 / 95 :=
by
  -- This space is intentionally left without proof
  sorry

end probability_top_four_cards_is_2_over_95_l333_333702


namespace sqrt_factorial_mul_square_l333_333646

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333646


namespace translation_invariant_line_l333_333240

theorem translation_invariant_line (k : ℝ) :
  (∀ x : ℝ, k * (x - 2) + 5 = k * x + 2) → k = 3 / 2 :=
by
  sorry

end translation_invariant_line_l333_333240


namespace compute_sqrt_factorial_square_l333_333627

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333627


namespace train1_speed_l333_333556

noncomputable def total_distance_in_kilometers : ℝ :=
  (630 + 100 + 200) / 1000

noncomputable def time_in_hours : ℝ :=
  13.998880089592832 / 3600

noncomputable def relative_speed : ℝ :=
  total_distance_in_kilometers / time_in_hours

noncomputable def speed_of_train2 : ℝ :=
  72

noncomputable def speed_of_train1 : ℝ :=
  relative_speed - speed_of_train2

theorem train1_speed : speed_of_train1 = 167.076 := by 
  sorry

end train1_speed_l333_333556


namespace sqrt_factorial_mul_square_l333_333638

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333638


namespace time_45_minutes_after_10_20_is_11_05_l333_333992

def time := Nat × Nat -- Represents time as (hours, minutes)

noncomputable def add_minutes (t : time) (m : Nat) : time :=
  let (hours, minutes) := t
  let total_minutes := minutes + m
  let new_hours := hours + total_minutes / 60
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_45_minutes_after_10_20_is_11_05 :
  add_minutes (10, 20) 45 = (11, 5) :=
  sorry

end time_45_minutes_after_10_20_is_11_05_l333_333992


namespace angle_B_in_triangle_l333_333847

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end angle_B_in_triangle_l333_333847


namespace drowning_ratio_l333_333251

variable (total_sheep total_cows total_dogs drowned_sheep drowned_cows total_animals : ℕ)

-- Conditions provided
variable (initial_conditions : total_sheep = 20 ∧ total_cows = 10 ∧ total_dogs = 14)
variable (sheep_drowned_condition : drowned_sheep = 3)
variable (dogs_shore_condition : total_dogs = 14)
variable (total_made_it_shore : total_animals = 35)

theorem drowning_ratio (h1 : total_sheep = 20) (h2 : total_cows = 10) (h3 : total_dogs = 14) 
    (h4 : drowned_sheep = 3) (h5 : total_animals = 35) 
    : (drowned_cows = 2 * drowned_sheep) :=
by
  sorry

end drowning_ratio_l333_333251


namespace additional_tiles_needed_l333_333271

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end additional_tiles_needed_l333_333271


namespace max_whole_nine_one_number_l333_333061

def is_non_zero_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def whole_nine_one_number (a b c d : ℕ) : Prop :=
  is_non_zero_digit a ∧ is_non_zero_digit b ∧ is_non_zero_digit c ∧ is_non_zero_digit d ∧ 
  (a + c = 9) ∧ (b = d + 1) ∧ ((2 * (2 * a + d) : ℚ) / (2 * b + c : ℚ)).denom = 1

def M (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d

theorem max_whole_nine_one_number : 
  ∃ (a b c d : ℕ), whole_nine_one_number a b c d ∧ M a b c d = 7524 :=
begin
  sorry
end

end max_whole_nine_one_number_l333_333061


namespace ab_ac_bc_range_l333_333116

-- Define the variables and conditions
variables (a b c : ℝ)
hypothesis (h : a + b + c = 3)

-- Statement to prove that the set of all possible values of ab + ac + bc is (-∞, 3]
theorem ab_ac_bc_range : ∃ S, S = set.Iic 3 ∧ (ab + ac + bc) ∈ S :=
by
  sorry

end ab_ac_bc_range_l333_333116


namespace find_a_l333_333857

/- Definitions -/
def C1_parametric (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a * real.cos t, 1 + a * real.sin t)

def C2_polar (theta : ℝ) : ℝ :=
  4 * real.cos theta

def C3_line (alpha0 : ℝ) : ℝ × ℝ :=
  (real.cos alpha0, real.sin alpha0)

def is_common_point (pt : ℝ × ℝ) (a : ℝ) (theta : ℝ) (alpha0 : ℝ) : Prop :=
  (∃ t, pt = C1_parametric a t) ∧
  (pt = ⟨C2_polar theta * real.cos theta, C2_polar theta * real.sin theta⟩) ∧
  (∃ x, pt = (x, 2 * x))

theorem find_a (theta alpha0 : ℝ) (h_alpha : real.tan alpha0 = 2) (pt : ℝ × ℝ) (a : ℝ) (h_common : is_common_point pt a theta alpha0) :
  a = 1 :=
sorry

end find_a_l333_333857


namespace part_one_solution_set_part_two_lower_bound_l333_333392

def f (x a b : ℝ) : ℝ := abs (x - a) + abs (x + b)

-- Part (I)
theorem part_one_solution_set (a b x : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (f x a b ≤ 5) ↔ -3 ≤ x ∧ x ≤ 2 := by
  rw [h1, h2]
  sorry

-- Part (II)
theorem part_two_lower_bound (a b x : ℝ) (h : a > 0) (h' : b > 0) (h'' : a + 4 * b = 2 * a * b) :
  f x a b ≥ 9 / 2 := by
  sorry

end part_one_solution_set_part_two_lower_bound_l333_333392


namespace fixed_point_through_M1M2_l333_333730

variable {p a b : ℝ}
variable {M M1 M2 A B : ℝ × ℝ}

def on_parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

def is_fixed_point (p a b : ℝ) (x y : ℝ) : Prop :=
  x = a ∧ y = 2 * p * a / b

theorem fixed_point_through_M1M2
  (h_parabola : ∀ x1 y1, on_parabola p x1 y1 → on_parabola p (M.fst) (M.snd))
  (h_A : A = (a, b)) 
  (h_B : B = (-a, 0))
  (h_ab_ne_zero : a ≠ 0 ∧ b ≠ 0)
  (h_b2_ne_2pa : b^2 ≠ 2 * p * a) 
  (h_M_on_parabola : on_parabola p (M.fst) (M.snd)) 
  (h_AM_intersects_parabola_at : ∃ y1, on_parabola p ((y1^2) / (2*p)) y1 = M1)
  (h_BM_intersects_parabola_at : ∃ y2, on_parabola p ((y2^2) / (2*p)) y2 = M2) :
  ∃ (x y : ℝ), on_parabola p x y ∧ is_fixed_point p a b x y := 
sorry

end fixed_point_through_M1M2_l333_333730


namespace no_valid_circle_arrangement_l333_333100

theorem no_valid_circle_arrangement : ¬ ∃ (circle : list ℕ), 
  (∀ (n : ℕ), n ∈ circle → 1 ≤ n ∧ n ≤ 12) ∧
  (∀ (i : ℕ), i < 12 → (circle.nth i).isSome) ∧
  (∀ (i : ℕ), i < 12 →
    ∃ (j : ℕ), j < 12 ∧ circle.nth i = some (j + 1) → 
    |(circle.nth i).get_or_else 0 - (circle.nth ((i + 1) % 12)).get_or_else 0| = 3 ∨ 
    |(circle.nth i).get_or_else 0 - (circle.nth ((i + 1) % 12)).get_or_else 0| = 4 ∨ 
    |(circle.nth i).get_or_else 0 - (circle.nth ((i + 1) % 12)).get_or_else 0| = 5
  ) ∧ 
  (circle.erase_dup.length = 12) := sorry

end no_valid_circle_arrangement_l333_333100


namespace lambda_range_l333_333000

open Real

theorem lambda_range {x y λ : ℝ} (h : 3 * x^2 + 4 * y^2 = 1) :
  (∀ x y : ℝ, |3 * x + 4 * y - λ| + |λ + 7 - 3 * x - 4 * y| = k) ↔ 
  (√7 - 7 ≤ λ ∧ λ ≤ -√7) :=
sorry

end lambda_range_l333_333000


namespace range_p_in_interval_l333_333463

def h (x : ℝ) : ℝ := 4 * x - 3
def p (x : ℝ) : ℝ := h(h(h(x)))

theorem range_p_in_interval (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) : 
  1 ≤ p(x) ∧ p(x) ≤ 129 :=
by
  sorry

end range_p_in_interval_l333_333463


namespace dessert_menu_count_l333_333263

def desserts := ["cake", "pie", "ice cream", "pudding"]

def menu_possible (desserts : List String) (days : ℕ) : ℕ := sorry

theorem dessert_menu_count : menu_possible desserts 7 = 972 := sorry

end dessert_menu_count_l333_333263


namespace intersection_P_Q_l333_333894

def P : Set ℝ := {x | |x| > 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x | -2 ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_P_Q_l333_333894


namespace num_divisors_of_power_l333_333408

theorem num_divisors_of_power (a b c : ℕ) (h1 : 2028 = 2^2 * 3^2 * 13^2)
  (h2 : 0 ≤ a ∧ a ≤ 4008)
  (h3 : 0 ≤ b ∧ b ≤ 4008)
  (h4 : 0 ≤ c ∧ c ≤ 4008)
  (h5 : (a + 1) * (b + 1) * (c + 1) = 2028) :
  (∏ d in {d | d.divisors 2028^2004 ∧ d % 2028 = 0}, 1) = 216 :=
sorry

end num_divisors_of_power_l333_333408


namespace symmetric_point_of_A_is_correct_l333_333178

def symmetric_point_with_respect_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_of_A_is_correct :
  symmetric_point_with_respect_to_x_axis (3, 4) = (3, -4) :=
by
  sorry

end symmetric_point_of_A_is_correct_l333_333178


namespace Mitzi_leftover_money_l333_333140

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end Mitzi_leftover_money_l333_333140


namespace range_of_x2_plus_4y2_l333_333385

theorem range_of_x2_plus_4y2 (x y : ℝ) (h : 4 * x^2 - 2 * real.sqrt 3 * x * y + 4 * y^2 = 13) :
  10 - 4 * real.sqrt 3 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 10 + 4 * real.sqrt 3 :=
by sorry

end range_of_x2_plus_4y2_l333_333385


namespace matthew_hotdogs_l333_333138

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end matthew_hotdogs_l333_333138


namespace polynomial_irreducible_l333_333464

variable {R : Type*} [CommRing R] (n : ℕ)

def poly (n : ℕ) : Polynomial R := X ^ n + (5 : R) * X ^ (n - 1) + (3 : R)

theorem polynomial_irreducible (hn : 2 ≤ n) : Irreducible (poly n) :=
by
  sorry

end polynomial_irreducible_l333_333464


namespace rainfall_mode_l333_333722

open List

-- Define the rainfall amounts for each day as given in the problem.
def weekly_rainfall : List ℕ := [6, 15, 3, 6, 3, 3, 9]

-- Define the mode function.
def mode (l : List ℕ) : ℕ :=
  l.groupBy id <|
    λ x y => x = y
  |>.map (λ g => (g.head!, g.length))
  |>.maxBy (λ p => p.2)
  |>.1

-- Problem statement:
theorem rainfall_mode : mode weekly_rainfall = 3 := 
  sorry

end rainfall_mode_l333_333722


namespace sum_of_squares_11_to_20_eq_2485_l333_333341

theorem sum_of_squares_11_to_20_eq_2485 :
  (∑ i in Finset.range (20 + 1), if 11 ≤ i then i ^ 2 else 0) = 2485 :=
by
  sorry

end sum_of_squares_11_to_20_eq_2485_l333_333341


namespace upper_limit_of_sixth_powers_l333_333269

theorem upper_limit_of_sixth_powers :
  ∃ b : ℕ, (∀ n : ℕ, (∃ a : ℕ, a^6 = n) ∧ n ≤ b → n = 46656) :=
by
  sorry

end upper_limit_of_sixth_powers_l333_333269


namespace sqrt_factorial_mul_squared_l333_333608

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333608


namespace cost_of_six_books_l333_333980

-- Define the regular cost of two books
def cost_of_two_books : ℝ := 36

-- Define the cost of one book, derived from the cost of two books
def cost_of_one_book : ℝ := cost_of_two_books / 2

-- Define the number of books
def num_books : ℕ := 6

-- Define the regular cost of the given number of books
def regular_cost (n : ℕ) : ℝ := (n : ℝ) * cost_of_one_book

-- Define the discount rate for buying 5 or more books
def discount_rate : ℝ := 0.10

-- Define the discounted cost calculation
def discounted_cost (n : ℕ) : ℝ :=
  if n >= 5 then
    regular_cost n - discount_rate * regular_cost n
  else
    regular_cost n

-- Theorem statement: cost of six books with discount should be 97.2 dollars
theorem cost_of_six_books : discounted_cost num_books = 97.2 :=
by
  -- Proof will go here
  sorry

end cost_of_six_books_l333_333980


namespace regular_octagon_angle_ABG_l333_333482

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end regular_octagon_angle_ABG_l333_333482


namespace camille_model_height_l333_333725

theorem camille_model_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) 
  (h_real_height : real_height = 50) (h_real_volume : real_volume = 200000) 
  (h_model_volume : model_volume = 0.05) : 
  real_height / (real_volume / model_volume)^(1/3) ≈ 0.315 :=
by
  sorry

end camille_model_height_l333_333725


namespace compute_abs_ab_eq_2_sqrt_111_l333_333939

theorem compute_abs_ab_eq_2_sqrt_111 (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = 2 * Real.sqrt 111 := 
sorry

end compute_abs_ab_eq_2_sqrt_111_l333_333939


namespace phil_coins_correct_l333_333913

def phil_coins_final (initial : ℕ) (years : ℕ) (weekly_collect : ℕ) (half_days_collect : ℕ) (daily_collect : ℕ) (loss_fraction : ℚ) : ℕ :=
  let after_three_years := initial * 3
  let fourth_year := weekly_collect * 52
  let fifth_year_days := 365 / 2 -- integer division
  let fifth_year := 2 * fifth_year_days
  let sixth_year := daily_collect * 365
  let before_loss := after_three_years + fourth_year + fifth_year + sixth_year
  let loss := before_loss / loss_fraction
  let loss_rounded := loss.to_nat -- rounding down
  before_loss - loss_rounded

theorem phil_coins_correct :
  phil_coins_final 250 3 5 182 1 (1/3 : ℚ) = 1160 := 
by
  sorry

end phil_coins_correct_l333_333913


namespace cone_base_radius_l333_333388

theorem cone_base_radius (L l r : ℝ) (h₁ : l = 5) (h₂ : L = 15 * real.pi) :
    2 * real.pi * r * l / 2 = L → r = 3 :=
by
  intro h
  simp at h
  sorry

end cone_base_radius_l333_333388


namespace length_n_squared_l333_333167

-- Define the linear functions a(x), b(x), c(x)
def a (x : ℝ) : ℝ := -x + 1
def b (x : ℝ) : ℝ := 1
def c (x : ℝ) : ℝ := x - 1

-- Define m(x) as the maximum of the linear functions
def m (x : ℝ) : ℝ := max (max (a x) (b x)) (c x)

-- Define n(x) as the minimum of the linear functions
def n (x : ℝ) : ℝ := min (min (a x) (b x)) (c x)

-- Compute the length of the graph of n(x) for the interval -4 to 4
noncomputable def length_n : ℝ :=
  let l1 := real.sqrt (((-2) - (-4))^2 + ((1 - 3)^2))
  let l2 := 4 -- length from -2 to 2 is just 4 for b(x)
  let l3 := real.sqrt (((4) - (2))^2 + ((3 - 1)^2))
  l1 + l2 + l3

-- Verify the square of the length of the graph of n(x)
theorem length_n_squared : length_n^2 = 48 + 32 * real.sqrt 2 :=
by
  sorry

end length_n_squared_l333_333167


namespace simplify_expression_l333_333164

theorem simplify_expression (x : ℝ) (h : x ≠ -2) : (4 / (x + 2) + x - 2) = (x^2 / (x + 2)) :=
by
  intro x h
  -- steps are omitted
  sorry

end simplify_expression_l333_333164


namespace problem_solution_l333_333466

noncomputable def compute_p_plus_q : ℤ :=
  let u := rat_cubrt x
  let v := rat_cubrt (30 - x)
  let h1 : u + v = 2 := sorry
  let h2 : u ^ 3 + v ^ 3 = 30 := sorry
  let h3 : u = 1 + (sqrt 42) / 3 := sorry
  let h4 : x = u ^ 3 := sorry
  let p : ℤ := 6 := sorry
  let q : ℤ := 42 := sorry
  p + q

theorem problem_solution (x : ℝ) (h : real.cbrt x + real.cbrt (30 - x) = 2) : 
  ∃ p q : ℤ, x = p - (real.sqrt q) ∧ p + q = compute_p_plus_q :=
by
  use 6
  use 42
  split
  sorry
  refl

end problem_solution_l333_333466


namespace find_a_l333_333383

theorem find_a (a : ℝ) (x : ℝ) : 
  let expr := (x + 1) * (a * x - 1 / x)^5 in
  expr.mk_const_term = -40 → a = 2 ∨ a = -2 := by
  sorry

def expr.mk_const_term : expression → ℝ 

end find_a_l333_333383


namespace geom_seq_value_l333_333011

variable (a_n : ℕ → ℝ)
variable (r : ℝ)
variable (π : ℝ)

-- Define the conditions
axiom geom_seq : ∀ n, a_n (n + 1) = a_n n * r
axiom sum_pi : a_n 3 + a_n 5 = π

-- Statement to prove
theorem geom_seq_value : a_n 4 * (a_n 2 + 2 * a_n 4 + a_n 6) = π^2 :=
by
  sorry

end geom_seq_value_l333_333011


namespace final_price_of_bedroom_set_l333_333151

def original_price : ℕ := 2000
def gift_card_amount : ℕ := 200
def store_discount : ℝ := 0.15
def credit_card_discount : ℝ := 0.10

theorem final_price_of_bedroom_set :
  let store_discount_amount := original_price * store_discount
  let after_store_discount := original_price - store_discount_amount.to_nat
  let credit_card_discount_amount := after_store_discount * credit_card_discount
  let after_credit_card_discount := after_store_discount - credit_card_discount_amount.to_nat
  let final_price := after_credit_card_discount - gift_card_amount
  final_price = 1330 :=
by
  -- Here we have the placeholder 'sorry' to indicate the proof step.
  sorry

end final_price_of_bedroom_set_l333_333151


namespace range_of_f_l333_333955

noncomputable def f (x : ℝ) : ℝ := 2 * x - x^2

theorem range_of_f : set.image f {x : ℝ | 1 < x ∧ x < 3} = set.Ioo (-3) 1 := by
  sorry

end range_of_f_l333_333955


namespace domain_h_l333_333169

theorem domain_h (f : ℝ → ℝ) (domain_f : ∀ x, (-10 ≤ x ∧ x ≤ 6) → x ∈ set.univ) :
  set_of (λ x, (-(5:ℝ)/3 ≤ x ∧ x ≤ 11/3)) = set_of (λ x, ∃ y, y = -3*x + 1 ∧ (-10 ≤ y ∧ y ≤ 6)) :=
sorry

end domain_h_l333_333169


namespace positive_difference_between_loans_l333_333444

noncomputable def loan_amount : ℝ := 12000

noncomputable def option1_interest_rate : ℝ := 0.08
noncomputable def option1_years_1 : ℕ := 3
noncomputable def option1_years_2 : ℕ := 9

noncomputable def option2_interest_rate : ℝ := 0.09
noncomputable def option2_years : ℕ := 12

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate)^years

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal + principal * rate * years

noncomputable def payment_at_year_3 : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 / 3

noncomputable def remaining_balance_after_3_years : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 - payment_at_year_3

noncomputable def total_payment_option1 : ℝ :=
  payment_at_year_3 + compound_interest remaining_balance_after_3_years option1_interest_rate option1_years_2

noncomputable def total_payment_option2 : ℝ :=
  simple_interest loan_amount option2_interest_rate option2_years

noncomputable def positive_difference : ℝ :=
  abs (total_payment_option1 - total_payment_option2)

theorem positive_difference_between_loans : positive_difference = 1731 := by
  sorry

end positive_difference_between_loans_l333_333444


namespace point_X_trajectory_star_shape_l333_333344

variables {n : ℕ} (O A B : Point)

-- Assume the following conditions:
-- 1. Regular n-gon with n ≥ 5 centered at point O.
-- 2. A and B are two adjacent vertices.
-- 3. △XYZ is congruent to △OAB.
-- 4. Initially, △XYZ is aligned with △OAB.
-- 5. Points Y and Z move along perimeter of n-gon, point X remains inside.

-- Definitions (should directly appear in the conditions)
def regular_ngon (n : ℕ) (O : Point) : Prop := n ≥ 5 ∧ 
  ∀ i j, i ≠ j → dist (vertex O i n) (vertex O j n) = dist (vertex O 0 n) (vertex O 1 n)

def congruent_triangles (XYZ OAB : Triangle) : Prop :=
  XYZ ≅ OAB

def moves_along_perimeter (Y Z : Point) (n : ℕ) : Prop :=
  ∃ k, Y = vertex O k n ∧ ∃ l, Z = vertex O l n

def X_inside (X : Point) (O : Point) (n : ℕ) : Prop :=
  ∀ i, dist O X < dist O (vertex O i n)

-- Determine trajectory of point X
theorem point_X_trajectory_star_shape
  (h_n : regular_ngon n O)
  (h_adj : adjacent O A B n)
  (h_congr : congruent_triangles (triangle X Y Z) (triangle O A B))
  (h_move : moves_along_perimeter Y Z n)
  (h_inside : X_inside X O n) :
  locus X = star_shape O n :=
sorry

end point_X_trajectory_star_shape_l333_333344


namespace no_polynomial_is_even_function_l333_333664

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, let S := {x : ℝ | f x = a} in S.finite ∧ S.card % 2 = 0

theorem no_polynomial_is_even_function (f : ℝ[X]) : ¬ is_even_function f :=
  sorry

end no_polynomial_is_even_function_l333_333664


namespace sum_of_three_consecutive_is_50_l333_333085

-- Define a sequence of integers
def seq : Fin 8 → ℕ
| 0 => 11
| 1 => 12
| 2 => 27
| 3 => 11
| 4 => 12
| 5 => 27
| 6 => 11
| 7 => 12

-- Theorem: The sum of any three consecutive numbers in the sequence is 50
theorem sum_of_three_consecutive_is_50 (n : Fin (8 - 2)) : 
  seq n + seq (n + 1) + seq (n + 2) = 50 :=
by
  fin_cases n
  case 0 => simp [seq]
  case 1 => simp [seq]
  case 2 => simp [seq]
  case 3 => simp [seq]
  case 4 => simp [seq]
  case 5 => simp [seq]
  -- This skips the proof details; the proof should show the sum matches 50 in all cases.
  sorry

-- Use sorry to skip the detailed proof

end sum_of_three_consecutive_is_50_l333_333085


namespace sqrt_factorial_mul_square_l333_333642

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333642


namespace alpha_plus_beta_l333_333541

theorem alpha_plus_beta (α β : ℤ) (h : ∀ x : ℤ, (x - α) / (x + β) = (x^2 - 96 * x + 2209) / (x^2 + 66 * x - 3969)) :
  α + β = 128 :=
begin
  sorry
end

end alpha_plus_beta_l333_333541


namespace minimize_dist_sum_to_sides_is_lemoine_l333_333494

-- Given data: triangle ABC with sides a, b, and c, and a point M inside the triangle.
variables {A B C M : Type} {triangle_ABC : Triangle A B C}
variables {a b c : ℝ} -- sides of the triangle
variables {x y z : ℝ} -- distances from point M to sides BC, CA, AB, respectively
variables (h1 : a = side_length BC)
variables (h2 : b = side_length CA)
variables (h3 : c = side_length AB)
variables (h4 : x = distance_to_side M BC)
variables (h5 : y = distance_to_side M CA)
variables (h6 : z = distance_to_side M AB)
variables (area_ABC : ℝ) (h7 : area_ABC = triangle_area A B C)

-- The proof statement to be formalized in Lean 4:
theorem minimize_dist_sum_to_sides_is_lemoine :
  ∃ (M : Type), (x^2 + y^2 + z^2) = (2 * area_ABC * a / (a^2 + b^2 + c^2))^2 +
                              (2 * area_ABC * b / (a^2 + b^2 + c^2))^2 + 
                              (2 * area_ABC * c / (a^2 + b^2 + c^2))^2 
                  → M = Lemoine_point triangle_ABC :=
by
  sorry

end minimize_dist_sum_to_sides_is_lemoine_l333_333494


namespace find_a_l333_333393

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then sin x else x^3 - 9*x^2 + 25*x + a

theorem find_a (a : ℝ) : 
  (∃ x1 x2 x3, x1 < 1 ∧ f x1 a = x1 ∧ 1 ≤ x2 ∧ f x2 a = x2 ∧ 1 ≤ x3 ∧ f x3 a = x3 ∧ x2 ≠ x3) ↔ a = -20 ∨ a = -16 :=
  sorry

end find_a_l333_333393


namespace initial_assumption_for_contradiction_l333_333994

-- Definitions corresponding to the conditions in the problem
def triangle (A B C : Type) := -- Placeholder definition for representing a triangle
  sorry 

def obtuse (angle : Type) : Prop := -- Placeholder definition for an obtuse angle
  sorry 

def has_at_most_one_obtuse_angle (T : Type) [triangle T] : Prop :=
  sorry 

-- The proof problem: Prove that for a triangle to have at most one obtuse angle, the contradiction assumption is that it has at least two obtuse angles.
theorem initial_assumption_for_contradiction (T : Type) [triangle T] :
  has_at_most_one_obtuse_angle T → ∀ (A B : Type), obtuse A → obtuse B → false :=
begin
  sorry
end

end initial_assumption_for_contradiction_l333_333994


namespace bombardment_deaths_l333_333851

variable (initial_population final_population : ℕ)
variable (fear_factor death_percentage : ℝ)

theorem bombardment_deaths (h1 : initial_population = 4200)
                           (h2 : final_population = 3213)
                           (h3 : fear_factor = 0.15)
                           (h4 : ∃ x, death_percentage = x / 100 ∧ 
                                       4200 - (x / 100) * 4200 - fear_factor * (4200 - (x / 100) * 4200) = 3213) :
                           death_percentage = 0.1 :=
by
  sorry

end bombardment_deaths_l333_333851


namespace reasoning_is_inductive_l333_333531

-- Define conditions
def conducts_electricity (metal : String) : Prop :=
  metal = "copper" ∨ metal = "iron" ∨ metal = "aluminum" ∨ metal = "gold" ∨ metal = "silver"

-- Define the inductive reasoning type
def is_inductive_reasoning : Prop := 
  ∀ metals, conducts_electricity metals → (∀ m : String, conducts_electricity m → conducts_electricity m)

-- The theorem to prove
theorem reasoning_is_inductive : is_inductive_reasoning :=
by
  sorry

end reasoning_is_inductive_l333_333531


namespace tommy_can_ride_north_proof_l333_333211

variable (n : ℕ)

-- Conditions
def tommy_can_ride_north : Prop := true
def tommy_can_ride_east : ℕ := 3
def tommy_can_ride_west : ℕ := 2
def tommy_can_ride_south : ℕ := 2
def friends_area : ℕ := 80
def area_relation : Prop := friends_area = 4 * (5 * (n + 2))

-- Problem statement
theorem tommy_can_ride_north_proof (h1 : tommy_can_ride_north) (h2 : area_relation) : n = 2 :=
sorry

end tommy_can_ride_north_proof_l333_333211


namespace solution_l333_333123

def sequenceProblem : Prop :=
  ∃ (X : ℕ), 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 1023 → (a : ℕ) (a = a * 2 + a * 2 + 1)) ∧
  (∀ a_i : ℕ, a_i = 0 ∨ ∃ k : ℕ, a_i = 2^k) ∧
  (a1 = 1024) ∧
  (X % 100 = 15)

theorem solution : sequenceProblem :=
begin
  sorry
end

end solution_l333_333123


namespace additional_tiles_needed_l333_333272

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end additional_tiles_needed_l333_333272


namespace sqrt_factorial_product_squared_l333_333632

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333632


namespace negation_of_cos_ge_one_l333_333526

theorem negation_of_cos_ge_one :
  ¬∀ x : ℝ, cos x ≥ 1 ↔ ∃ x0 : ℝ, cos x0 < 1 := by
  sorry

end negation_of_cos_ge_one_l333_333526


namespace even_four_digit_numbers_count_l333_333948

-- Definition of digits set
def digits : Set ℕ := {1, 2, 3, 4, 5}

-- Property that no digits are repeated and number must be four-digit and even
def valid_number (n : ℕ) : Prop :=
  let ds := digits.toList
  let ds_u := List.erase ds (n % 10)
  let ds_ss : List ℕ := ds_u.filter (λ x, x ≠ (n / 10 % 10) ∧ x ≠ (n / 100 % 10) ∧ x ≠ (n / 1000 % 10))
  (n % 2 = 0) ∧ ((1000 ≤ n) ∧ (n < 10000)) ∧ List.length ds_ss = 2 ∧ List.perm ds_u [((n / 10 % 10)), (n / 100 % 10), (n / 1000 % 10)]

-- Theorem stating the number of valid four-digit even numbers
theorem even_four_digit_numbers_count : 
  {n : ℕ | valid_number n}.toFinset.card = 48 := by
  sorry

end even_four_digit_numbers_count_l333_333948


namespace ratio_of_square_side_lengths_l333_333957

theorem ratio_of_square_side_lengths (a b c : ℕ) :
  (∀ (r : ℚ), r = 192 / 80 → 
  (∃ (x y : ℕ), x = a * y ∧ y * y = b * c ∧ r = a / c)) → 
  a + b + c = 22 :=
begin
  -- Proof to be provided
  sorry
end

end ratio_of_square_side_lengths_l333_333957


namespace regular_octagon_angle_ABG_l333_333483

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end regular_octagon_angle_ABG_l333_333483


namespace only_prime_alternating_1_0_l333_333288

def is_alternating_1_0 (n : ℕ) : Prop :=
  let bits := Nat.digits 2 n
  bits.headD 0 = 1 ∧ bits.tailD [] = List.map (λ x, (x + 1) % 2) (bits.tailD [])

theorem only_prime_alternating_1_0 :
  ∀ n : ℕ, is_prime n ∧ is_alternating_1_0 n → n = 101 := by
  sorry

end only_prime_alternating_1_0_l333_333288


namespace diagonal_rectangle_is_correct_l333_333188

noncomputable def length_of_diagonal (P : ℕ) (ratio_length width_ratio: ℕ) : ℚ :=
let k := (P / (2 * (ratio_length + width_ratio))) in
let length := ratio_length * k in
let width := width_ratio * k in
let diagonal := Real.sqrt (length^2 + width^2) in diagonal

theorem diagonal_rectangle_is_correct :
  length_of_diagonal 72 5 2 = 194 / 7 := by
  sorry

end diagonal_rectangle_is_correct_l333_333188


namespace count_multiples_l333_333044

theorem count_multiples (n : ℕ) (h_n : n = 300) : 
  let m := 6 in 
  let m' := 12 in 
  (finset.card (finset.filter (λ x, x % m = 0 ∧ x % m' ≠ 0) (finset.range n))) = 24 :=
by
  sorry

end count_multiples_l333_333044


namespace factorization_correct_l333_333227

theorem factorization_correct (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by 
  sorry

end factorization_correct_l333_333227


namespace magnitude_of_complex_number_l333_333315

theorem magnitude_of_complex_number : 
  let i : ℂ := complex.I in
  let z : ℂ := 1 / (i - 1) in
  complex.abs z = real.sqrt 2 / 2 :=
by
  sorry

end magnitude_of_complex_number_l333_333315


namespace angle_B_in_triangle_ABC_range_of_f_in_interval_l333_333019

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 2 * sin x * cos (x + π / 4)

/- Part I -/
theorem angle_B_in_triangle_ABC (A B C : ℝ) (BC AB : ℝ) 
  (hBC : BC = 2) (hAB : AB = sqrt 2) (h : f (A - π / 4) = 0) :
  B = π / 4 ∨ B = 7 * π / 12 :=
sorry

/- Part II -/
theorem range_of_f_in_interval :
  ∀ x ∈ set.Icc (π / 2) (17 * π / 24), f x ∈ set.Icc (-sqrt 2 - 1) (-2) :=
sorry

end angle_B_in_triangle_ABC_range_of_f_in_interval_l333_333019


namespace total_amount_paid_l333_333110

def toy_organizers: ℕ := 3
def cost_per_toy_organizer: ℝ := 78
def gaming_chairs: ℕ := 2
def cost_per_gaming_chair: ℝ := 83
def desk_cost: ℝ := 120
def bookshelf_cost: ℝ := 95
def total_sales (a b c d: ℝ) : ℝ := a + b + c + d
def delivery_fee (total: ℝ) : ℝ :=
  if total <= 300 then total * 0.03
  else if total ≤ 600 then total * 0.05
  else total * 0.07

theorem total_amount_paid:
  let total_items_cost := total_sales (toy_organizers * cost_per_toy_organizer)
                                       (gaming_chairs * cost_per_gaming_chair)
                                       desk_cost
                                       bookshelf_cost in
  let fee := delivery_fee total_items_cost in
  total_items_cost + fee = 658.05 := by
  sorry

end total_amount_paid_l333_333110


namespace sqrt_factorial_squared_l333_333574

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333574


namespace sqrt_factorial_squared_l333_333571

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333571


namespace intersecting_circles_l333_333654

theorem intersecting_circles (circles : List ℝ) (h1 : ∑ r in circles, r = 0.51) (h2 : ∀ r ∈ circles, r < 0.5) :
  ∃ l, (∀r ∈ circles, r ≤ 1) → (1 < ∑d in circles.map (λ r, 2 * r), d) := 
sorry

end intersecting_circles_l333_333654


namespace cannot_be_expressed_l333_333461

open Nat

theorem cannot_be_expressed (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_coprime_ab : coprime a b) (h_coprime_bc : coprime b c) (h_coprime_ca : coprime c a) :
    ¬ ∃ (x y z : ℕ), 2 * a * b * c - a * b - b * c - c * a = x * b * c + y * c * a + z * a * b :=
sorry

end cannot_be_expressed_l333_333461


namespace number_of_boys_is_10_l333_333723

-- Definitions based on given conditions
def num_children := 20
def has_blue_neighbor_clockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition
def has_red_neighbor_counterclockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition

axiom boys_and_girls_exist : ∃ b g : ℤ, b + g = num_children ∧ b > 0 ∧ g > 0

-- Theorem based on the problem statement
theorem number_of_boys_is_10 (b g : ℤ) 
  (total_children: b + g = num_children)
  (boys_exist: b > 0)
  (girls_exist: g > 0)
  (each_boy_has_blue_neighbor: ∀ i, has_blue_neighbor_clockwise i → true)
  (each_girl_has_red_neighbor: ∀ i, has_red_neighbor_counterclockwise i → true): 
  b = 10 :=
by
  sorry

end number_of_boys_is_10_l333_333723


namespace principal_amount_is_10000_l333_333337

-- Definitions of the conditions in the problem
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Given conditions
def given_SI : ℝ := 400
def given_R : ℝ := 4 / 100
def given_T : ℝ := 1.0

-- Prove the principal amount
theorem principal_amount_is_10000 : 
    ∃ P : ℝ, P = 10000 ∧ simple_interest P given_R given_T = given_SI := 
by 
  sorry

end principal_amount_is_10000_l333_333337


namespace num_multiples_6_not_12_lt_300_l333_333037

theorem num_multiples_6_not_12_lt_300 : 
  ∃ n : ℕ, n = 25 ∧ ∀ k : ℕ, k < 300 ∧ k % 6 = 0 ∧ k % 12 ≠ 0 → ∃ m : ℕ, k = 6 * (2 * m - 1) ∧ 1 ≤ m ∧ m ≤ 25 := 
by
  sorry

end num_multiples_6_not_12_lt_300_l333_333037


namespace find_f_prime_two_l333_333806

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * (deriv f 1) * x

theorem find_f_prime_two : deriv f 2 = 1 := sorry

end find_f_prime_two_l333_333806


namespace triangle_construction_possible_l333_333308

noncomputable theory

variables (a r : ℝ) (A : ℝ) 

-- Define the conditions for constructing triangle ABC
def can_construct_triangle :=
  ∃ (BC : ℝ) (inscribed_circle_radius : ℝ) (angle_A : ℝ),
    BC = a ∧ angle_A = A ∧ inscribed_circle_radius = r ∧
    (∃ (O : Type) (B C : Type), 
      -- Geometry constrains ensuring triangle existence with the given properties
      ∃ (inscribed_circle : Type) (tangent_from_B tangent_from_C : Type), 
        BC = a ∧ 
        angle_A = A ∧ 
        inscribed_circle_radius = r ∧
        -- Conditions to satisfy the construction of the triangle
        true)

theorem triangle_construction_possible :
  can_construct_triangle a r A :=
sorry

end triangle_construction_possible_l333_333308


namespace range_of_a_l333_333415

theorem range_of_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2)
    (hxy : x * y = 2) (h : ∀ x y, 2 - x ≥ a / (4 - y)) : a ≤ 0 :=
sorry

end range_of_a_l333_333415


namespace ants_meet_again_at_point_P_l333_333213

-- Definitions
def circle_tangent_at (P : Point) := ∃ c₁ c₂ : Circle, c₁.radius = 6 ∧ c₂.radius = 3 ∧ tangent c₁ c₂ P

def ant_speed := ∀ (c : Circle) (P : Point), 
  (c.radius = 6 → speed c P = 4 * π) ∧
  (c.radius = 3 → speed c P = 3 * π)

-- Theorem Statement
theorem ants_meet_again_at_point_P  (P : Point) 
  (C₁ C₂ : Circle) 
  (h_tangent : tangent C₁ C₂ P) 
  (hC₁ : C₁.radius = 6) 
  (hC₂ : C₂.radius = 3) 
  (v₁ : speed C₁ P = 4 * π) 
  (v₂ : speed C₂ P = 3 * π) 
  : meet_again_at P (6 : ℕ) := 
by 
  sorry

end ants_meet_again_at_point_P_l333_333213


namespace no_third_degree_polynomial_exists_l333_333241

theorem no_third_degree_polynomial_exists (a b c d : ℤ) (h : a ≠ 0) :
  ¬(p 15 = 3 ∧ p 21 = 12 ∧ p = λ x => a * x ^ 3 + b * x ^ 2 + c * x + d) :=
sorry

end no_third_degree_polynomial_exists_l333_333241


namespace find_values_f_l333_333020

open Real

noncomputable def f (ω A x : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) + 2 * A * (cos (ω * x))^2 - A

theorem find_values_f (θ : ℝ) (A : ℝ) (ω : ℝ) (hA : A > 0) (hω : ω = 1)
  (h1 : π / 6 < θ) (h2 : θ < π / 3) (h3 : f ω A θ = 2 / 3) :
  f ω A (π / 3 - θ) = (1 + 2 * sqrt 6) / 3 :=
  sorry

end find_values_f_l333_333020


namespace ticket_distribution_l333_333540

noncomputable def num_dist_methods (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : ℕ := sorry

theorem ticket_distribution :
  num_dist_methods 18 5 6 7 10 = 140 := sorry

end ticket_distribution_l333_333540


namespace how_many_pints_did_Annie_pick_l333_333739

theorem how_many_pints_did_Annie_pick (x : ℕ) (h1 : Kathryn = x + 2)
                                      (h2 : Ben = Kathryn - 3)
                                      (h3 : x + Kathryn + Ben = 25) : x = 8 :=
  sorry

end how_many_pints_did_Annie_pick_l333_333739


namespace area_of_triangle_l333_333002

noncomputable def F1 : ℝ × ℝ := (-√2, 0)
noncomputable def F2 : ℝ × ℝ := (√2, 0)
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2
def is_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def perp_vectors (P : ℝ × ℝ) : Prop := dot_product (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2) = 0
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem area_of_triangle (P : ℝ × ℝ)
  (hC : is_on_hyperbola P)
  (hPerp : perp_vectors P) :
  (1/2) * distance P F1 * distance P F2 = 2 :=
sorry

end area_of_triangle_l333_333002


namespace find_third_line_l333_333327

theorem find_third_line (a b : ℝ) : 
  let x := -a / 2 + sqrt (a^2 / 4 + b^2) in 
  x * (x + a) = b^2 :=
by
  sorry

end find_third_line_l333_333327


namespace first_student_stickers_l333_333971

theorem first_student_stickers (
  a : ℕ → ℕ,
  h1 : a 2 = 35,
  h2 : a 3 = 41,
  h3 : a 4 = 47,
  h4 : a 5 = 53,
  h5 : a 6 = 59,
  h_pattern : ∀ n, a (n + 1) = a n + 6
) : a 1 = 29 :=
sorry

end first_student_stickers_l333_333971


namespace quadrilateral_is_rhombus_l333_333122

theorem quadrilateral_is_rhombus {A B C D O : Type*} 
  (hO: ∃ {A B C D : Point}, Point O ∧ Intersection (Diags A B C D) O)
  (hEqualRadii: ∃ r, inscribed_circle_radius (Triangle A B O) = r ∧
                       inscribed_circle_radius (Triangle B C O) = r ∧
                       inscribed_circle_radius (Triangle C D O) = r ∧
                       inscribed_circle_radius (Triangle D A O) = r ) :
  is_rhombus (Quadrilateral A B C D) :=
sorry

end quadrilateral_is_rhombus_l333_333122


namespace probability_absolute_value_l333_333155

-- Defines the events specified in the problem.
variable (roll_die : ℕ → ℝ)
variable (x y : ℝ)
variable (prob : ℝ → ℝ → ℝ)

-- Conditions
axiom die_roll_1_2 : ∀ n, (roll_die n = 0) ↔ (n = 1 ∨ n = 2)
axiom die_roll_3_4 : ∀ n, (roll_die n = 1) ↔ (n = 3 ∨ n = 4)
axiom die_roll_5_6 : ∀ n, 5 ≤ n ∧ n ≤ 6 → ∃ z, (0 ≤ z ∧ z ≤ 1 ∧ roll_die n = z)
axiom x_chosen : (x = roll_die 1)
axiom y_chosen : (y = roll_die 1 ∨ y = roll_die 2)

-- Question to prove:
theorem probability_absolute_value (h : prob x y = | x - y | > 1/2): prob x y = 7/12 := 
sorry

end probability_absolute_value_l333_333155


namespace james_initial_amount_l333_333868

-- Define the costs and leftover money
def ticket1_cost : ℕ := 150
def ticket2_cost : ℕ := 150
def ticket3_fraction : ℚ := 1 / 3
def remaining_money : ℕ := 325

-- Define the costs of the individual tickets
def ticket3_cost : ℕ := (ticket1_cost : ℚ * ticket3_fraction).natAbs
def total_cost : ℕ := ticket1_cost + ticket2_cost + ticket3_cost
def roommate_payment_fraction : ℚ := 1 / 2
def james_payment : ℕ := (total_cost * roommate_payment_fraction).natAbs

-- Lean statement for required proof
theorem james_initial_amount : (remaining_money + james_payment) = 500 :=
by
  sorry

end james_initial_amount_l333_333868


namespace monotonically_increasing_interval_l333_333051

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * real.log x + b * x^2 + x
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := (a / x) + 2 * b * x + 1

theorem monotonically_increasing_interval (a b : ℝ) (h1 : a + 2 * b + 1 = 0) (h2 : a / 2 + 4 * b + 1 = 0) :
 ∃ I : Set ℝ, I = !(1, 2) ∧ ∀ x ∈ I, f' a b x > 0 :=
by
  sorry

end monotonically_increasing_interval_l333_333051


namespace height_of_cylinder_l333_333274

theorem height_of_cylinder
  (R r : ℝ)
  (h : ℝ)
  (hr : r = 3)
  (hR : R = 6) :
  h = 3 * real.sqrt 3 :=
sorry

end height_of_cylinder_l333_333274


namespace sum_of_repeating_decimals_l333_333752

theorem sum_of_repeating_decimals : (0.\overline{6} + 0.\overline{3} = 1) :=
by sorry

end sum_of_repeating_decimals_l333_333752


namespace pages_for_15_dollars_l333_333440

theorem pages_for_15_dollars 
  (cpg : ℚ) -- cost per 5 pages in cents
  (budget : ℚ) -- budget in cents
  (h_cpg_pos : cpg = 7 * 1) -- 7 cents for 5 pages
  (h_budget_pos : budget = 1500 * 1) -- $15 = 1500 cents
  : (budget * (5 / cpg)).floor = 1071 :=
by {
  sorry
}

end pages_for_15_dollars_l333_333440


namespace four_m_plus_one_2013_eq_neg_one_l333_333823

theorem four_m_plus_one_2013_eq_neg_one (m : ℝ) (h : |m| = m + 1) : (4 * m + 1) ^ 2013 = -1 := 
sorry

end four_m_plus_one_2013_eq_neg_one_l333_333823


namespace solution_set_of_quadratic_inequality_l333_333795

theorem solution_set_of_quadratic_inequality 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x < 0 ↔ x < -1 ∨ x > 1 / 3)
  (h₂ : ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) : 
  ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3 := 
by
  intro x
  exact h₂ x

end solution_set_of_quadratic_inequality_l333_333795


namespace initial_dogs_l333_333691

theorem initial_dogs (D : ℕ) (h : D + 5 + 3 = 10) : D = 2 :=
by sorry

end initial_dogs_l333_333691


namespace bug_returns_starting_vertex_eighth_move_l333_333672

/-- Initial Probability Definition -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => 1 / 3 * (1 - P n)

-- Define the theorem to prove
theorem bug_returns_starting_vertex_eighth_move :
  let m := 547
  let n := 2187
  P 8 = m / n ∧ m + n = 2734 :=
by
  -- sorry to defer the actual proof
  sorry

end bug_returns_starting_vertex_eighth_move_l333_333672


namespace percent_sum_l333_333414

theorem percent_sum (A B C : ℝ)
  (hA : 0.45 * A = 270)
  (hB : 0.35 * B = 210)
  (hC : 0.25 * C = 150) :
  0.75 * A + 0.65 * B + 0.45 * C = 1110 := by
  sorry

end percent_sum_l333_333414


namespace car_speed_second_hour_l333_333962

theorem car_speed_second_hour (s₁ s_avg t₁ t₂ : ℝ) (h₁ : s₁ = 90) (h_avg : s_avg = 66) (h_t : t₁ + t₂ = 2) :
  (2 * s_avg - s₁ = 42) :=
by
  rw [h₁, h_avg, h_t]
  sorry

end car_speed_second_hour_l333_333962


namespace pentagon_area_increase_l333_333217

theorem pentagon_area_increase (P : Type) [convex_pentagon P] : 
  let Q = move_sides_outward_by P 4
  in area(Q) >= area(P) + 50 :=
sorry

end pentagon_area_increase_l333_333217


namespace correct_conditions_for_cubic_eq_single_root_l333_333469

noncomputable def hasSingleRealRoot (a b : ℝ) : Prop :=
  let f := λ x : ℝ => x^3 - a * x + b
  let f' := λ x : ℝ => 3 * x^2 - a
  ∀ (x y : ℝ), f' x = 0 → f' y = 0 → x = y

theorem correct_conditions_for_cubic_eq_single_root :
  (hasSingleRealRoot 0 2) ∧ 
  (hasSingleRealRoot (-3) 2) ∧ 
  (hasSingleRealRoot 3 (-3)) :=
  by 
    sorry

end correct_conditions_for_cubic_eq_single_root_l333_333469


namespace sum_of_interior_angles_of_polyhedron_l333_333778

theorem sum_of_interior_angles_of_polyhedron (V E : ℕ) (hV : V = 20) (hE : E = 30) (F : ℕ) :
  (V - E + F = 2) → (12 * 540 = 6480) :=
by
  intros hEuler
  have hF : F = 12,
  { sorry },  -- This is where you would solve for F using Euler's formula
  rw hF,
  simp,
  exact 6480,

end sum_of_interior_angles_of_polyhedron_l333_333778


namespace number_of_factors_b_pow_n_eq_46_l333_333787

theorem number_of_factors_b_pow_n_eq_46 (b n : ℕ) (hb : b = 8) (hn : n = 15) : 
  (∃ k : ℕ, k = (b^n).factors.length + 1) := 
by
  have h1 : b = 8 := hb
  have h2 : n = 15 := hn
  have h3 : b^n = 8^15 := by rw [h1, h2]
  have h4 : (8^15).prime_factors = (2^45).prime_factors := by sorry -- Substitute the prime factorization
  have h5 : (2^45).factors.length = 45 := by sorry -- Number of factors for prime factor
  rw [h3, h4, h5]
  use 46
  exact sorry -- Finalize with the result 45 + 1 = 46

end number_of_factors_b_pow_n_eq_46_l333_333787


namespace range_ffx_l333_333470

def f (x : ℝ) : ℝ :=
if x < 1 then x - 1 else 2 ^ x

theorem range_ffx : set.range (f ∘ f) = set.Ici (2 / 3) := by
  sorry

end range_ffx_l333_333470


namespace two_op_neg_two_op_eq_one_implies_x_l333_333347

-- Define the custom operation ※
def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

-- Proof statement 1: 2 ※ (-2) = -1
theorem two_op_neg_two : op 2 (-2) = -1 := sorry

-- Proof statement 2: (2 ※ (2x - 1) = 1) implies x = 5/6
theorem op_eq_one_implies_x (x : ℝ) (hx : op 2 (2 * x - 1) = 1) : x = 5 / 6 := sorry

end two_op_neg_two_op_eq_one_implies_x_l333_333347


namespace compute_fraction_l333_333728

theorem compute_fraction :
  (∏ n in (finset.range 51).filter (λ k, k ≠ 0 ∧ k % 2 = 1), int.floor (real.sqrt (k + 1))) /
  (∏ n in (finset.range 51).filter (λ k, k % 2 = 0), int.floor (real.sqrt (k + 1))) = 21 / 64 := by
    sorry

end compute_fraction_l333_333728


namespace linear_function_no_pass_quadrant_I_l333_333183

theorem linear_function_no_pass_quadrant_I (x y : ℝ) (h : y = -2 * x - 1) : 
  ¬ (0 < x ∧ 0 < y) :=
by 
  sorry

end linear_function_no_pass_quadrant_I_l333_333183


namespace max_M_is_7524_l333_333062

-- Define the conditions
def is_valid_t (t : ℕ) : Prop :=
  let a := t / 1000
  let b := (t % 1000) / 100
  let c := (t % 100) / 10
  let d := t % 10
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  (2 * (2 * a + d)) % (2 * b + c) = 0

-- Define function M
def M (a b c d : ℕ) : ℕ := 2000 * a + 100 * b + 10 * c + d

-- Define the maximum value of M
def max_valid_M : ℕ :=
  let m_values := [5544, 7221, 7322, 7524]
  m_values.foldl max 0

theorem max_M_is_7524 : max_valid_M = 7524 := by
  -- The proof would be written here. For now, we indicate the theorem as
  -- not yet proven.
  sorry

end max_M_is_7524_l333_333062


namespace find_polynomial_l333_333270

noncomputable def polynomial (m : ℝ) : ℝ := 4 * m^3 + m^2 + 5

theorem find_polynomial (m : ℝ) :
  (∃ p : ℝ → ℝ, ∀ x : ℝ, p x - polynomial x = 3 * x^4 - 4 * x^3 - x^2 + x - 8) →
  (∃ q : ℝ → ℝ, q = fun x => 3 * x^4 + x - 3) :=
begin
  intro h,
  cases h with p hp,
  use (fun x => 3 * x^4 + x - 3),
  sorry
end

end find_polynomial_l333_333270


namespace emptying_time_proof_l333_333704

def cubic_inches_to_cubic_feet (cubic_inches: ℝ): ℝ := cubic_inches / 1728

def total_outlet_rate (outlet1 outlet2 outlet3: ℝ): ℝ := 
  cubic_inches_to_cubic_feet outlet1 + cubic_inches_to_cubic_feet outlet2 + cubic_inches_to_cubic_feet outlet3

def total_inlet_rate (inlet1 inlet2: ℝ): ℝ :=
  inlet1 + inlet2

def net_emptying_rate (inlet_rate outlet_rate: ℝ): ℝ :=
  inlet_rate - outlet_rate

def time_to_empty (volume rate: ℝ): ℝ :=
  volume / rate

theorem emptying_time_proof :
  let tank_volume := 30
  let inlet_rate1 := 5
  let inlet_rate2 := 2
  let outlet_rate1 := 9
  let outlet_rate2 := 8
  let outlet_rate3 := 6
  let outlet_total := total_outlet_rate outlet_rate1 outlet_rate2 outlet_rate3
  let inlet_total := total_inlet_rate inlet_rate1 inlet_rate2
  let net_rate := net_emptying_rate inlet_total outlet_total
  time_to_empty tank_volume net_rate ≈ 4.294 := by
    -- Placeholder for proof
    sorry

end emptying_time_proof_l333_333704


namespace k_points_if_one_point_l333_333184

-- The notion of an inner point being observable from a side.
def observable (X : Point) (YZ : LineSegment) : Prop :=
  -- Definition that the perpendicular to YZ from X meets YZ in the closed interval [YZ].
  -- This definition needs to be devised based on the specific geometry setup.
  sorry

-- A point in a quadrilateral is a k-point if it is observable from exactly k sides.
def k_point (Q : Quadrilateral) (P : Point) (k : ℕ) : Prop :=
  -- Definition that P is observable from exactly k sides of Q.
  sorry

-- Main theorem statement
theorem k_points_if_one_point (Q : Quadrilateral) :
  (∃ P : Point, k_point Q P 1) → (∀ k : ℕ, k ∈ {2, 3, 4} → ∃ P : Point, k_point Q P k) :=
by
  sorry

end k_points_if_one_point_l333_333184


namespace decreasing_interval_f_l333_333947

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (f (x) = (1 / 2) * x^2 - Real.log x) →
  (∃ a b : ℝ, 0 < a ∧ a ≤ b ∧ b = 1 ∧ ∀ y, a < y ∧ y ≤ b → f (y) ≤ f (y+1)) := sorry

end decreasing_interval_f_l333_333947


namespace count_multiples_l333_333043

theorem count_multiples (n : ℕ) (h_n : n = 300) : 
  let m := 6 in 
  let m' := 12 in 
  (finset.card (finset.filter (λ x, x % m = 0 ∧ x % m' ≠ 0) (finset.range n))) = 24 :=
by
  sorry

end count_multiples_l333_333043


namespace smallest_number_l333_333205

theorem smallest_number (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) = 90) (h4 : b = 28) (h5 : b = c - 6) : a = 28 :=
by 
  sorry

end smallest_number_l333_333205


namespace even_function_has_a_square_equal_1_l333_333413

/- 
Define the function f and its property of being even, then prove a^2 = 1.
-/
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + (a^2 - 1) * x + 6

theorem even_function_has_a_square_equal_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a) → a^2 = 1 :=
by
  intro h
  have hx : ∀ x : ℝ, x^2 - (a^2 - 1) * x + 6 = x^2 + (a^2 - 1) * x + 6 := by {
    intro x,
    specialize h x,
    simp [f] at h,
    exact h,
  }
  sorry

end even_function_has_a_square_equal_1_l333_333413


namespace ounces_of_wax_for_car_l333_333101

noncomputable def ounces_wax_for_SUV : ℕ := 4
noncomputable def initial_wax_amount : ℕ := 11
noncomputable def wax_spilled : ℕ := 2
noncomputable def wax_left_after_detailing : ℕ := 2
noncomputable def total_wax_used : ℕ := initial_wax_amount - wax_spilled - wax_left_after_detailing

theorem ounces_of_wax_for_car :
  (initial_wax_amount - wax_spilled - wax_left_after_detailing) - ounces_wax_for_SUV = 3 :=
by
  sorry

end ounces_of_wax_for_car_l333_333101


namespace sum_of_distinct_x_values_l333_333896

def complex_eq_system (x y z : ℂ) : Prop :=
  x + y * z = 9 ∧ y + x * z = 12 ∧ z + x * y = 12

theorem sum_of_distinct_x_values :
  (finset.univ.filter (λ p : ℂ × ℂ × ℂ, complex_eq_system p.1 p.2.1 p.2.2)).sum (λ p, p.1) = 9 :=
sorry

end sum_of_distinct_x_values_l333_333896


namespace increased_contact_area_effect_l333_333350

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end increased_contact_area_effect_l333_333350


namespace value_of_b_div_a_l333_333053

theorem value_of_b_div_a (a b : ℝ) (h : |5 - a| + (b + 3)^2 = 0) : b / a = -3 / 5 :=
by
  sorry

end value_of_b_div_a_l333_333053


namespace trigonometric_identity_l333_333500

theorem trigonometric_identity :
  sin (119 * Real.pi / 180) * cos (91 * Real.pi / 180) - sin (91 * Real.pi / 180) * sin (29 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l333_333500


namespace quadratic_inequality_solution_l333_333339

noncomputable def solve_inequality (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x > -1/2 ∧ x < 1/3) → (a * x^2 + b * x + 2 > 0)) →
  (a = -12) ∧ (b = -2)

theorem quadratic_inequality_solution :
   solve_inequality (-12) (-2) :=
by
  intro h
  sorry

end quadratic_inequality_solution_l333_333339


namespace find_periodic_increasing_function_l333_333289

theorem find_periodic_increasing_function : 
  ∃ (f : ℝ → ℝ), (function_period f π) ∧ (∀ x, 0 < x ∧ x < π/2 → increasing_on f (set.Ioo 0 (π/2))) ∧ (f = (λ x, -cos (2*x))) :=
by
  -- Following definitions as per Lean standards for keeping the proof statement valid.
  def function_period (f : ℝ → ℝ) (p : ℝ) : Prop := 
    ∀ x, f(x + p) = f(x) 

  def increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop := 
    ∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f a < f b 

  sorry

end find_periodic_increasing_function_l333_333289


namespace sqrt_factorial_mul_squared_l333_333607

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333607


namespace sqrt_factorial_mul_squared_l333_333602

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_squared_l333_333602


namespace marked_prices_l333_333298

noncomputable def marked_price_A (CP : ℝ) (profit_percent : ℝ) (deduction_percent : ℝ) : ℝ :=
  let SP := CP + CP * profit_percent / 100
  let MP := SP / ((100 - deduction_percent) / 100)
  MP

theorem marked_prices :
  (marked_price_A 47.50 25 6 ≈ 63.15) ∧
  (marked_price_A 82.00 30 8 ≈ 115.87) ∧
  (marked_price_A 120.00 20 5 ≈ 151.58) :=
by {
  sorry
}

end marked_prices_l333_333298


namespace count_real_solutions_l333_333816

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end count_real_solutions_l333_333816


namespace subtraction_example_l333_333224

theorem subtraction_example : 6102 - 2016 = 4086 := by
  sorry

end subtraction_example_l333_333224


namespace A_n_expression_b_geometric_sequence_l333_333878

noncomputable section

variable (C q B : ℝ) (n : ℕ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)

axiom h₁ : ∀ n, a n = (1 - q^n) / (1 - q)
axiom h₂ : b 1 + b 2 + ... + b n = B
axiom h₃ : A n = n - (q + q^2 + ... + q^n)
axiom h₄ : A n = n - q * (1 - q^n) / (1 - q)
axiom h₅ : A n = (1 - q^n) - q * (n - 1)

theorem A_n_expression : A n = C * (n * (1 - q) - q + q^(n + 1)) / (1 - q)^2 :=
  sorry

theorem b_geometric_sequence : ∀ n > 1, b n = q * b (n - 1) :=
  sorry

end A_n_expression_b_geometric_sequence_l333_333878


namespace domain_f_l333_333179

noncomputable def f (x : ℝ) : ℝ := log 2 (x - 1) + sqrt (4 - 2^x)

theorem domain_f : {x : ℝ | x - 1 > 0 ∧ 4 - 2^x ≥ 0} = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_f_l333_333179


namespace restricted_bijection_of_f_l333_333820

variable {α : Type*} (f : α → α) (A_n : Set α)

theorem restricted_bijection_of_f (h_iter : ∀ m : ℕ, f^[m] = f) (h_fix : ∀ x ∈ f '' A_n, f x ∈ f '' A_n) :
  Function.Bijective (f ∘ fun x => x ∈ (f '' A_n)) :=
sorry

end restricted_bijection_of_f_l333_333820


namespace range_of_a_l333_333181

def quadratic_function (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l333_333181


namespace determine_a_l333_333804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 / (3 ^ x + 1)) - a

theorem determine_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end determine_a_l333_333804


namespace oc_value_l333_333255

theorem oc_value (O A B C : Point) (theta : Real) (s' c' : Real) 
  (h1 : circle O 1 A)
  (h2 : tangent_segment AB O A)
  (h3 : angle AOB = theta / 2)
  (h4 : lies_on_line C O A)
  (h5 : bisects_line_angle_segment BC ABO)
  (s'_def : s' = Real.sin (theta / 2))
  (c'_def : c' = Real.cos (theta / 2)) :
  OC = 1 / (1 + s') := 
begin
  sorry,
end

end oc_value_l333_333255


namespace person_A_days_l333_333479

-- Introduce assumptions and variables
variable (A : ℕ) -- Number of days it takes for Person A to complete the work alone
variable (T_b : ℕ) := 60 -- Number of days it takes for Person B to complete the work alone
variable (combined_rate : ℚ := 0.25 / 6) -- The combined rate of work completion for A and B in 6 days

-- Define the rate of work for each person
def rate_a := (1 : ℚ) / A -- Person A's rate
def rate_b := (1 : ℚ) / T_b -- Person B's rate

-- The main theorem statement
theorem person_A_days : rate_a + rate_b = combined_rate → A = 40 :=
by
  -- Proof can be filled in later
  sorry

end person_A_days_l333_333479


namespace square_field_side_length_l333_333671

theorem square_field_side_length (time_sec : ℕ) (speed_kmh : ℕ) (perimeter : ℕ) (side_length : ℕ)
  (h1 : time_sec = 96)
  (h2 : speed_kmh = 9)
  (h3 : perimeter = (9 * 1000 / 3600 : ℕ) * 96)
  (h4 : perimeter = 4 * side_length) :
  side_length = 60 :=
by
  sorry

end square_field_side_length_l333_333671


namespace count_multiples_6_not_12_l333_333032

theorem count_multiples_6_not_12 (n: ℕ) : 
  ∃ (count : ℕ), count = 25 ∧ 
                  count = (finset.filter (λ m, (m < 300) ∧ (6 ∣ m) ∧ ¬ (12 ∣ m)) (finset.range 300)).card :=
by
  sorry

end count_multiples_6_not_12_l333_333032


namespace Roy_trip_distance_l333_333158

-- Define the total distance of the trip
variable (d : ℝ)

-- Conditions given in the problem
def first_segment_distance := 60
def second_segment_distance := 100
def second_segment_fuel_rate := 0.03
def third_segment_fuel_rate := 0.015
def overall_efficiency := 75

-- Define the gasoline usage according to the given conditions
def gasoline_used : ℝ := second_segment_fuel_rate * second_segment_distance + third_segment_fuel_rate * (d - first_segment_distance - second_segment_distance)

-- Define the equation representing the overall average fuel efficiency
def avg_fuel_efficiency_equation : Prop := d / gasoline_used = overall_efficiency

-- State the goal (proof problem)
theorem Roy_trip_distance : avg_fuel_efficiency_equation d → d = 360 :=
by
  sorry

end Roy_trip_distance_l333_333158


namespace distance_between_parallel_lines_l333_333515

/-- The distance between two parallel lines 3x + 4y - 5 = 0 and 6x + 8y - 5 = 0 is 1/2. -/
theorem distance_between_parallel_lines :
  let l1 := (3 : ℝ, 4 : ℝ, -5 : ℝ)
  let l2 := (3 * 2 : ℝ, 4 * 2 : ℝ, -5 : ℝ)
  let a := l1.1
  let b := l1.2
  let c1 := l1.3
  let c2 := l2.3
  dist := (|c2 - c1| : ℝ) / real.sqrt (a ^ 2 + b ^ 2)
  dist = (1 / 2 : ℝ) := sorry

end distance_between_parallel_lines_l333_333515


namespace cows_with_no_spots_l333_333904

-- Definitions of conditions
def total_cows : Nat := 140
def cows_with_red_spot : Nat := (40 * total_cows) / 100
def cows_without_red_spot : Nat := total_cows - cows_with_red_spot
def cows_with_blue_spot : Nat := (25 * cows_without_red_spot) / 100

-- Theorem statement asserting the number of cows with no spots
theorem cows_with_no_spots : (total_cows - cows_with_red_spot - cows_with_blue_spot) = 63 := by
  -- Proof would go here
  sorry

end cows_with_no_spots_l333_333904


namespace tan_theta_is_minus_five_divided_by_twelve_l333_333360

noncomputable def tan_theta_given_conditions (m : ℝ) (θ : ℝ) : ℝ :=
  let sinθ := (m - 3) / (m + 5)
  let cosθ := (4 - 2 * m) / (m + 5)
  sinθ / cosθ

theorem tan_theta_is_minus_five_divided_by_twelve (m θ : ℝ) 
  (h1 : sin θ = (m - 3) / (m + 5))
  (h2 : cos θ = (4 - 2 * m) / (m + 5))
  (h3 : π / 2 < θ ∧ θ < π) :
  tan θ = -5 / 12 :=
sorry

end tan_theta_is_minus_five_divided_by_twelve_l333_333360


namespace percentage_difference_l333_333689

theorem percentage_difference (n z x y y_decreased : ℝ)
  (h1 : x = 8 * y)
  (h2 : y = 2 * |z - n|)
  (h3 : z = 1.1 * n)
  (h4 : y_decreased = 0.75 * y) :
  (x - y_decreased) / x * 100 = 90.625 := by
sorry

end percentage_difference_l333_333689


namespace astronomy_club_officer_selection_l333_333697

def members : Finset ℕ := Finset.range 25  -- Representing the 25 members (numbered from 0 to 24).

def alice : ℕ := 0 -- Let's assume Alice is member 0.
def bob : ℕ := 1 -- Let's assume Bob is member 1.
def charles : ℕ := 2 -- Let's assume Charles is member 2.
def diana : ℕ := 3 -- Let's assume Diana is member 3.

def alice_and_bob_condition (S : Finset ℕ) : Prop := (alice ∈ S ↔ bob ∈ S)  -- Alice and Bob both should be officers or neither.
def charles_and_diana_condition (S : Finset ℕ) : Prop := (charles ∈ S ↔ diana ∈ S)  -- Charles and Diana both should be officers or neither.

def valid_officer_selection (S : Finset ℕ) : Prop :=
  S.card = 3 ∧ alice_and_bob_condition S ∧ charles_and_diana_condition S  -- The selection must have all the members of the conditions.

theorem astronomy_club_officer_selection : 
  ∑ S in Finset.powersetLen 3 members, if valid_officer_selection S then (S.card.factorial : ℕ) else 0 = 8232 := 
by
  sorry

end astronomy_club_officer_selection_l333_333697


namespace new_box_volume_eq_5_76_m3_l333_333973

-- Given conditions:
def original_width_cm := 80
def original_length_cm := 75
def original_height_cm := 120
def conversion_factor_cm3_to_m3 := 1000000

-- New dimensions after doubling
def new_width_cm := 2 * original_width_cm
def new_length_cm := 2 * original_length_cm
def new_height_cm := 2 * original_height_cm

-- Statement of the problem
theorem new_box_volume_eq_5_76_m3 :
  (new_width_cm * new_length_cm * new_height_cm : ℝ) / conversion_factor_cm3_to_m3 = 5.76 := 
  sorry

end new_box_volume_eq_5_76_m3_l333_333973


namespace least_m_eq_2_pow_1990_minus_1_l333_333874

def num_factors_of_2 (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, n / (2 ^ (k + 1)))

def s_2 (n : ℕ) : ℕ :=
  (nat.digits 2 n).sum

theorem least_m_eq_2_pow_1990_minus_1 :
  ∃ m : ℕ, (m - num_factors_of_2 m = 1990) ∧ m = 2 ^ 1990 - 1 :=
by
  sorry

end least_m_eq_2_pow_1990_minus_1_l333_333874


namespace ending_number_div_by_3_l333_333200

theorem ending_number_div_by_3 (n : ℤ) (h1 : n = 10) (h2 : ∃ m : ℤ, ∀ k : ℤ, k ≥ 0 ∧ k < 13 → (10 + 3 * k) = 10 + 3 * k ∧ (10 + 3 * k) is exactly divisible by 3) : 
  ∃ l, l = 48 := 
by sorry

end ending_number_div_by_3_l333_333200


namespace minimum_a_for_cube_in_tetrahedron_l333_333260

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  (Real.sqrt 6 / 12) * a

theorem minimum_a_for_cube_in_tetrahedron (a : ℝ) (r : ℝ) 
  (h_radius : r = radius_of_circumscribed_sphere a)
  (h_diag : Real.sqrt 3 = 2 * r) :
  a = 3 * Real.sqrt 2 :=
by
  sorry

end minimum_a_for_cube_in_tetrahedron_l333_333260


namespace time_to_pass_platform_l333_333247

theorem time_to_pass_platform (length_train : ℝ) (time_to_cross_tree : ℝ) (length_platform : ℝ) 
  (length_train_eq : length_train = 1200) (time_to_cross_tree_eq : time_to_cross_tree = 120) 
  (length_platform_eq : length_platform = 500) : 
  let speed := length_train / time_to_cross_tree in
  let total_distance := length_train + length_platform in
  let time_to_pass_platform := total_distance / speed in
  time_to_pass_platform = 170 := 
by
  rw [length_train_eq, time_to_cross_tree_eq, length_platform_eq]
  have speed_eq : speed = length_train / time_to_cross_tree := rfl
  rw [speed_eq]
  have speed_val : speed = 10 := by
    simp [length_train_eq, time_to_cross_tree_eq]
  rw [speed_val]
  have total_dist_eq : total_distance = length_train + length_platform := rfl
  rw [total_dist_eq]
  have total_distance_val : total_distance = 1700 := by
    simp [length_train_eq, length_platform_eq]
  rw [total_distance_val]
  have time_pass_eq : time_to_pass_platform = total_distance / speed := rfl
  rw [time_pass_eq]
  simp [total_distance_val, speed_val]
  sorry

end time_to_pass_platform_l333_333247


namespace find_m_l333_333808

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b : V)
variables (m : ℝ)
-- Magnitude of vectors
axiom a_norm_one : ∥a∥ = 1
axiom b_norm_two : ∥b∥ = 2
-- The angle between a and b is 60 degrees which implies their dot product
axiom angle_60 : ⟪a, b⟫ = 1

-- Lean theorem statement corresponding to the math problem
theorem find_m (h : ⟪3 • a + 5 • b, m • a - b⟫ = 0) : m = 23 / 8 := 
sorry

end find_m_l333_333808


namespace sum_of_first_10_terms_l333_333389

variable (a : ℕ → ℕ) (d : ℕ)
def aₙ (n : ℕ) := 3 + n * d

theorem sum_of_first_10_terms :
  (a 1 = 3) →
  (a 2 + a 5 = 36) →
  (∑ i in finset.range 10, a i) = 300 :=
by
  sorry

end sum_of_first_10_terms_l333_333389


namespace triangle_is_right_l333_333373

-- Definitions and conditions
variable (m n : ℝ) (x y : ℝ)
variable (F1 F2 P : Point)
-- Assumptions
axiom h1 : m > 1
axiom h2 : n > 0
axiom h3 : P ∈ ellipse (m) ∧ P ∈ hyperbola (n)
axiom h4 : shared_foci (ellipse (m)) (hyperbola (n)) F1 F2

-- Proof statement
theorem triangle_is_right :
  ∠(F1, P, F2) = 90 :=
  sorry

end triangle_is_right_l333_333373


namespace andy_2023rd_turn_l333_333718

-- Andy's initial position and direction.
structure InitialState :=
  (position : ℤ × ℤ)
  (facing : ℤ × ℤ)

def initial_state : InitialState :=
  { position := (10, -10), facing := (0, 1) }  -- facing north

-- Define the movement after n steps
def next_position (n : ℕ) (initial : InitialState) : ℤ × ℤ :=
  let directions := [ (0, 1), (1, 0), (0, -1), (-1, 0) ] in
  let move_dir := directions.get! ((n - 1) % 4) in
  let step := n + 1 in
  (initial.position.1 + move_dir.1 * step, initial.position.2 + move_dir.2 * step)

-- Define the position after 2023 moves.
def andy_position_after_turns (n : ℕ) : ℤ × ℤ :=
  (foldl (λ pos i, (next_position i { position := pos, facing := initial_state.facing }).position) initial_state.position (List.range n))

theorem andy_2023rd_turn : andy_position_after_turns 2023 = (1022, 1) :=
  sorry

end andy_2023rd_turn_l333_333718


namespace sqrt_factorial_product_squared_l333_333633

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l333_333633


namespace positive_integer_divisors_l333_333328

theorem positive_integer_divisors (n : ℕ) (p : ℕ → ℕ) (k : ℕ)
  (hp : ∀ i, i < k → Nat.Prime (p i))
  (h_factorization : ∏ i in Finset.range k, p i = n)
  (h_divisibility : n ∣ ∏ i in Finset.range k, (p i + 1)) :
  ∃ r s : ℕ, n = 2^r * 3^s ∧ s ≤ r ∧ r ≤ 2*s := by
  sorry

end positive_integer_divisors_l333_333328


namespace regular_polygon_num_sides_l333_333290

theorem regular_polygon_num_sides (angle : ℝ) (h : angle = 45) : 
  (∃ n : ℕ, n = 360 / angle ∧ n ≠ 0) → n = 8 :=
by
  sorry

end regular_polygon_num_sides_l333_333290


namespace unique_reachable_pair_l333_333216

-- Axiomatize the operations of the calculator
inductive CalcOp (x y : ℕ) : ℕ × ℕ → Prop
| add (x y : ℕ) : CalcOp x y (x + y, x)
| special (x y : ℕ) : CalcOp x y (2 * x + y + 1, x + y + 1)

-- Define the reachability relation
inductive Reachable : ℕ × ℕ → ℕ × ℕ → Prop
| start (x y : ℕ) : Reachable (x, y) (x, y)
| step (x y z w : ℕ) (p : ℕ × ℕ) :
    CalcOp x y (z, w) → Reachable (z, w) p → Reachable (x, y) p

-- The main theorem statement
theorem unique_reachable_pair (n : ℕ) :
  ∃! (k : ℕ), Reachable (1, 1) (n, k) :=
sorry

end unique_reachable_pair_l333_333216


namespace max_whole_nine_one_number_l333_333060

def is_non_zero_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def whole_nine_one_number (a b c d : ℕ) : Prop :=
  is_non_zero_digit a ∧ is_non_zero_digit b ∧ is_non_zero_digit c ∧ is_non_zero_digit d ∧ 
  (a + c = 9) ∧ (b = d + 1) ∧ ((2 * (2 * a + d) : ℚ) / (2 * b + c : ℚ)).denom = 1

def M (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d

theorem max_whole_nine_one_number : 
  ∃ (a b c d : ℕ), whole_nine_one_number a b c d ∧ M a b c d = 7524 :=
begin
  sorry
end

end max_whole_nine_one_number_l333_333060


namespace probability_ending_at_point_a_probability_passing_segment_b_probability_passing_circle_c_l333_333681

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial_coefficient (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def fly_probability_ending_at (x y steps : ℕ) : ℕ :=
  binomial_coefficient (steps) (x) / (2 ^ steps)

def fly_probability_passing_segment (x1 y1 x2 y2 x y steps1 steps2 steps3 : ℕ) : ℕ :=
  (binomial_coefficient steps1 x1 * binomial_coefficient steps2 x2) / (2 ^ steps3)

def fly_probability_passing_circle (circle_points : List (ℕ × ℕ)) (x y steps : ℕ) : ℕ :=
  (circle_points.sum (λ (i, j), binomial_coefficient 9 i * binomial_coefficient 9 j)) / (2 ^ steps)

theorem probability_ending_at_point_a :
  fly_probability_ending_at 8 10 18 = (binomial_coefficient 18 8 / (2^18)) := by sorry

theorem probability_passing_segment_b :
  fly_probability_passing_segment 5 6 6 6 8 10 11 6 18 = (binomial_coefficient 11 5 * binomial_coefficient 6 2 / (2^18)) := by sorry

theorem probability_passing_circle_c :
  fly_probability_passing_circle [(2, 11), (3, 10), (4, 9), (5, 8), (6, 7)] 8 10 18 = (2 * binomial_coefficient 9 2 * binomial_coefficient 9 6 + 2 * binomial_coefficient 9 3 * binomial_coefficient 9 5 + binomial_coefficient 9 4 * binomial_coefficient 9 4) / (2^18) := by sorry

end probability_ending_at_point_a_probability_passing_segment_b_probability_passing_circle_c_l333_333681


namespace find_a_find_omega_max_l333_333405

variables {x : ℝ} {ω : ℝ} {a : ℝ}

def a_vec (ω x : ℝ) : ℝ × ℝ := (1 + real.cos (ω * x), 1)
def b_vec (a ω x : ℝ) : ℝ × ℝ := (1, a + sqrt 3 * real.sin (ω * x))

def f (a ω x : ℝ) := (a_vec ω x).1 * (b_vec a ω x).1 + (a_vec ω x).2 * (b_vec a ω x).2

def g (a ω x : ℝ) := f a ω (x - π / (6 * ω))

noncomputable def g' (a ω x : ℝ) := -ω * real.sin (ω * (x - π / (6 * ω))) + sqrt 3 * ω * real.cos (ω * (x - π / (6 * ω)))

axiom ω_pos (ω : ℝ) : ω > 0
axiom f_max (a ω : ℝ) : ∀ x : ℝ, f a ω x ≤ 2
axiom g_increasing (a ω : ℝ) : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 4 → 0 ≤ g' a ω x

theorem find_a : a = 0 :=
sorry

theorem find_omega_max : ω ≤ 3 :=
sorry

end find_a_find_omega_max_l333_333405


namespace quadratic_inequality_solution_l333_333012

theorem quadratic_inequality_solution (a m : ℝ) (h : a < 0) :
  (∀ x : ℝ, ax^2 + 6*x - a^2 < 0 ↔ (x < 1 ∨ x > m)) → m = 2 :=
by
  sorry

end quadratic_inequality_solution_l333_333012


namespace number_of_correct_coverings_l333_333475

def is_correct_covering (covering : matrix (fin 8) (fin 8) (fin 4)) : Prop :=
  ∀ i j : fin 8, 
    -- Check that any two triangles sharing a side are of different colors
    ∃ c : fin 4, covering i j ≠ c

theorem number_of_correct_coverings : ∃ n : nat, n = 2^16 ∧
  ∃ covering : matrix (fin 8) (fin 8) (fin 4) → Prop, is_correct_covering covering :=
sorry

end number_of_correct_coverings_l333_333475


namespace part1_part2_l333_333343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  ∫ t in x..(a * x), real.sqrt (1 - t^2)

theorem part1 (a : ℝ) (h : a > 1) :
  f a (1 / a) + ∫ t in 0..(1 / a), real.sqrt (1 - t^2) = π / 4 :=
by sorry

theorem part2 (a : ℝ) (h : a > 1) :
  (∃ M, ∀ x, f a x ≤ M) ∧
  (f a (real.sqrt 3 / (2 * a)) = π / 12) :=
by sorry

end part1_part2_l333_333343


namespace length_one_side_triangle_ABO_l333_333915

theorem length_one_side_triangle_ABO :
  ∃ (x1 : ℝ), 
  let A := (x1, -x1^2) in
  let B := (-x1, -x1^2) in
  let O := (0, 0) in
  (x1^2 = 3) →
  (abs (2 * x1) = 4 * sqrt 3) ∧
  ∀ x1 y1, 
  y1 = -x1^2 →
  sqrt (x1^2 + y1^2) = 2 * sqrt 3 :=
by
  sorry

end length_one_side_triangle_ABO_l333_333915


namespace percentage_below_cost_l333_333557

variable (CP SP : ℝ)

-- Given conditions
def cost_price : ℝ := 5625
def more_for_profit : ℝ := 1800
def profit_percentage : ℝ := 0.16
def expected_SP : ℝ := cost_price + (cost_price * profit_percentage)
def actual_SP : ℝ := expected_SP - more_for_profit

-- Statement to prove
theorem percentage_below_cost (h1 : CP = cost_price) (h2 : SP = actual_SP) :
  (CP - SP) / CP * 100 = 16 := by
sorry

end percentage_below_cost_l333_333557


namespace Kelvin_frog_likes_5_digit_numbers_l333_333870

open Nat

def kelvinLikes (n : Nat) : Prop :=
  ∀ i j, 0 ≤ i → i < j → j < 5 → digit n i ≥ digit n j

def kelvinAcceptable (n : Nat) : Prop :=
  let violations := (List.range 4).countp (λ i => digit n i < digit n (i + 1))
  violations ≤ 1

def isFiveDigits (n : Nat) : Prop := 10000 ≤ n ∧ n < 100000

theorem Kelvin_frog_likes_5_digit_numbers : 
  (finset.range 100000).filter (λ n => isFiveDigits n ∧ (kelvinLikes n ∨ kelvinAcceptable n)).card = 14034 := 
by 
  sorry

end Kelvin_frog_likes_5_digit_numbers_l333_333870


namespace add_decimals_l333_333286

theorem add_decimals (a b : ℝ) (h_a : a = 4.358) (h_b : b = 3.892) : a + b = 8.250 := 
by
  rw [h_a, h_b]
  norm_num
  sorry

end add_decimals_l333_333286


namespace exists_composite_value_in_polynomial_factorial_l333_333447

open Nat

def is_composite (n : ℕ) : Prop :=
  ∃ x y, 1 < x ∧ x < n ∧ 1 < y ∧ y < n ∧ x * y = n

theorem exists_composite_value_in_polynomial_factorial (P : ℕ → ℕ)
  (a : ℕ → ℤ) (n : ℕ)
  (hP : ∀ x, P(x) = ∑ i in range (n + 1), a i * x^i)
  (ha_nonzero : a n > 0)
  (hn_ge_two : n ≥ 2) :
  ∃ m : ℕ, is_composite(P(m!)) :=
by
  sorry

end exists_composite_value_in_polynomial_factorial_l333_333447


namespace find_num_yoYos_l333_333109

variables (x y z w : ℕ)

def stuffed_animals_frisbees_puzzles := x + y + w = 80
def total_prizes := x + y + z + w + 180 + 60
def cars_and_robots := 180 + 60 = x + y + z + w + 15

theorem find_num_yoYos 
(h1 : stuffed_animals_frisbees_puzzles x y w)
(h2 : total_prizes = 300)
(h3 : cars_and_robots x y z w) : z = 145 :=
sorry

end find_num_yoYos_l333_333109


namespace count_multiples_6_not_12_l333_333031

theorem count_multiples_6_not_12 (n: ℕ) : 
  ∃ (count : ℕ), count = 25 ∧ 
                  count = (finset.filter (λ m, (m < 300) ∧ (6 ∣ m) ∧ ¬ (12 ∣ m)) (finset.range 300)).card :=
by
  sorry

end count_multiples_6_not_12_l333_333031


namespace area_of_rectangle_ABCD_is_12_2_l333_333318

noncomputable def area_of_rectangle_ABCD : ℝ :=
let DB := 5 in -- Diagonal
let AE := Real.sqrt 6 in -- Altitude from geometric mean theorem
let area_triangle_ABD := (1 / 2) * DB * AE in -- Area of triangle ABD
let area_ABCD := 2 * area_triangle_ABD in -- Area of rectangle ABCD
area_ABCD

theorem area_of_rectangle_ABCD_is_12_2 : area_of_rectangle_ABCD = 12.2 :=
by
  have h1 : 5 * Real.sqrt 6 = 12.245 := sorry
  rw [←h1]
  norm_num

end area_of_rectangle_ABCD_is_12_2_l333_333318


namespace average_growth_rate_l333_333193

theorem average_growth_rate (x : ℝ) (hx : (1 + x)^2 = 1.44) : x < 0.22 :=
sorry

end average_growth_rate_l333_333193


namespace series_sum_eq_l333_333995

theorem series_sum_eq : 
  (∑ k in (finset.range (50)), (1 : ℚ) / (2 * k + 1) * (1 : ℚ) / (2 * (k+1) + 1)) = (50:ℚ) / 101 := 
sorry

end series_sum_eq_l333_333995


namespace multiply_101_self_l333_333748

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end multiply_101_self_l333_333748


namespace rectangle_area_given_conditions_l333_333522

theorem rectangle_area_given_conditions
  (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_given_conditions_l333_333522


namespace range_F_is_one_l333_333009

-- Defining the function f_M given a set M and a real number x
def f_M (M : set ℝ) (x : ℝ) : ℝ :=
  if x ∈ M then 1 else 0

-- Defining the function F given sets A and B, and a real number x
def F (A B : set ℝ) (x : ℝ) : ℝ :=
  (f_M (A ∪ B) x + 1) / (f_M A x + f_M B x + 1)

-- The theorem we want to prove
theorem range_F_is_one (A B : set ℝ) (hA : A ≠ ∅) (hB : B ≠ ∅) (hAB : A ∩ B = ∅) :
  set.range (F A B) = {1} :=
sorry

end range_F_is_one_l333_333009


namespace hyperbola_equation_l333_333807

-- Define the hyperbola equation with conditions
def hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a * b / Math.sqrt (a^2 + b^2)) = 12 / 5 ∧ 2 * b = 8

-- Statement of the theorem
theorem hyperbola_equation : ∃ a b : ℝ, hyperbola a b ∧ (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1) :=
by
  sorry

end hyperbola_equation_l333_333807


namespace find_x_l333_333094

theorem find_x (B C D : Point) (A : Point) (x y : ℝ)
  (h1 : B ≠ A) (h2 : C ≠ A) (h3 : D ≠ A)
  (hBCD : Collinear B C D)
  (hACD : ∠ A C D = 100)
  (hADB : ∠ A D B = x)
  (hABD : ∠ A B D = 2 * x)
  (hDAC : ∠ D A C = y)
  (hBAC : ∠ B A C = y)
  : x = 20 := by
  sorry

end find_x_l333_333094


namespace compute_sqrt_factorial_square_l333_333623

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_sqrt_factorial_square :
  (sqrt ((factorial 5) * (factorial 4)))^2 = 2880 :=
by
  sorry

end compute_sqrt_factorial_square_l333_333623


namespace perfect_square_k_values_l333_333534

def y_sequence (k : ℤ) : ℕ → ℤ
| 0       => 1
| 1       => 1
| (n + 2) => (4 * k - 5) * y_sequence k (n + 1) - y_sequence k n + 4 - 2 * k

theorem perfect_square_k_values :
  ∀ k : ℤ, (∀ n : ℕ, ∃ m : ℤ, y_sequence k n = m * m) ↔ (k = 1 ∨ k = 3) :=
begin
  sorry
end

end perfect_square_k_values_l333_333534


namespace potato_yield_l333_333902

/-- Mr. Green's gardening problem -/
theorem potato_yield
  (steps_length : ℝ)
  (steps_width : ℝ)
  (step_size : ℝ)
  (yield_rate : ℝ)
  (feet_length := steps_length * step_size)
  (feet_width := steps_width * step_size)
  (area := feet_length * feet_width)
  (yield := area * yield_rate) :
  steps_length = 18 →
  steps_width = 25 →
  step_size = 2.5 →
  yield_rate = 0.75 →
  yield = 2109.375 :=
by
  sorry

end potato_yield_l333_333902


namespace total_amount_spent_is_300_l333_333653

-- Definitions of conditions
def S : ℕ := 97
def H : ℕ := 2 * S + 9

-- The total amount spent
def total_spent : ℕ := S + H

-- Proof statement
theorem total_amount_spent_is_300 : total_spent = 300 :=
by
  sorry

end total_amount_spent_is_300_l333_333653


namespace apples_in_blue_basket_l333_333539

-- Define the number of bananas in the blue basket
def bananas := 12

-- Define the total number of fruits in the blue basket
def totalFruits := 20

-- Define the number of apples as total fruits minus bananas
def apples := totalFruits - bananas

-- Prove that the number of apples in the blue basket is 8
theorem apples_in_blue_basket : apples = 8 := by
  sorry

end apples_in_blue_basket_l333_333539


namespace peter_reads_one_book_18_hours_l333_333912

-- Definitions of conditions given in the problem
variables (P : ℕ)

-- Condition: Peter can read three times as fast as Kristin
def reads_three_times_as_fast (P : ℕ) : Prop :=
  ∀ (K : ℕ), K = 3 * P

-- Condition: Kristin reads half of her 20 books in 540 hours
def half_books_in_540_hours (K : ℕ) : Prop :=
  K = 54

-- Theorem stating the main proof problem: proving P equals 18 hours
theorem peter_reads_one_book_18_hours
  (H1 : reads_three_times_as_fast P)
  (H2 : half_books_in_540_hours (3 * P)) :
  P = 18 :=
sorry

end peter_reads_one_book_18_hours_l333_333912


namespace problem_solution_l333_333798

theorem problem_solution (a b x : ℝ) (h : b ≥ 0) :
  (b ∈ set.Ico 0 1 ∨ b ∈ set.Ioo 1 ∞) ↔
  (x = 0 ∨ (b = 1 ∧ x ∈ set.Icc (-2 : ℝ) 2)) :=
begin
  sorry
end

end problem_solution_l333_333798


namespace percent_in_second_part_l333_333054

-- Defining the conditions and the proof statement
theorem percent_in_second_part (x y P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.25 * x) : 
  P = 15 :=
by
  sorry

end percent_in_second_part_l333_333054


namespace arithmetic_geometric_sequence_problem_l333_333435

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (q : ℚ)
  (h1 : ∀ n m : ℕ, a (n + m) = a n * (q ^ m))
  (h2 : a 2 * a 3 * a 4 = 27 / 64)
  (h3 : q = 2)
  (h4 : ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h5 : b 7 = a 5) : 
  b 3 + b 11 = 6 := 
sorry

end arithmetic_geometric_sequence_problem_l333_333435


namespace probability_of_odd_perfect_ten_three_digit_number_l333_333064

-- Define a "perfect ten three-digit number" 
def is_perfect_ten_three_digit_number (n : Nat) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits[0] ≠ digits[1] ∧ digits[0] ≠ digits[2] ∧ digits[1] ≠ digits[2] ∧
  digits.sum = 10

-- Define a three-digit number to be odd if its last digit is odd
def is_odd (n : Nat) : Prop :=
  (n % 10) % 2 = 1

-- Number of "perfect ten three-digit numbers"
def count_perfect_ten_three_digit_numbers : Nat :=
  ([109, 190, 901, 910, 127, 172, 271, 217, 721, 712, 136, 163, 316, 361, 613, 631,
    145, 154, 451, 415, 514, 541, 208, 280, 802, 820, 235, 253, 352, 325, 523, 532,
    307, 370, 703, 730, 406, 460, 604, 640].length)

-- Number of odd "perfect ten three-digit numbers"
def count_odd_perfect_ten_three_digit_numbers : Nat :=
  ([109, 127, 271, 217, 721, 712, 136, 163, 316, 361, 613, 631, 145, 154, 451, 415,
    514, 541, 235, 253, 523, 532, 307, 703].length)

theorem probability_of_odd_perfect_ten_three_digit_number : 
  ∀ (n : Nat), (n ∈ [109, 190, 901, 910, 127, 172, 271, 217, 721, 712, 136, 163,
  316, 361, 613, 631, 145, 154, 451, 415, 514, 541, 208, 280, 802, 820, 235, 253,
  352, 325, 523, 532, 307, 370, 703, 730, 406, 460, 604, 640]) →
  P((is_odd n)) = 1 / 2 :=
by
  sorry

end probability_of_odd_perfect_ten_three_digit_number_l333_333064


namespace Miki_last_observation_l333_333901

theorem Miki_last_observation :
  ∃ n : ℕ, 
    n = 18 ∧ 
    (∀ (P : ℕ → ℕ), 
      (P 5 = 2 * P 3 ∧ 
      P 8 = P 3 + 100 ∧ 
      P 9 = P 4 + P 7 ∧ 
      ∀ k, even (P k) ∧ ¬ divisible_by 3 (P k) ∧ 
      ∃ m, P n = m * m ∧ 
      ∀ d, (P 3 - P 2 = d) ∧ 
      ∀ k > n, ¬∃ m, P k = m * m) :=
  sorry

end Miki_last_observation_l333_333901


namespace sqrt_factorial_mul_square_l333_333641

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333641


namespace genevieve_coffee_drink_l333_333357

theorem genevieve_coffee_drink :
  let gallons := 4.5
  let small_thermos_count := 12
  let small_thermos_capacity_ml := 250
  let large_thermos_count := 6
  let large_thermos_capacity_ml := 500
  let genevieve_small_thermos_drink_count := 2
  let genevieve_large_thermos_drink_count := 1
  let ounces_per_gallon := 128
  let mls_per_ounce := 29.5735
  let total_mls := (gallons * ounces_per_gallon) * mls_per_ounce
  let genevieve_ml_drink := (genevieve_small_thermos_drink_count * small_thermos_capacity_ml) 
                            + (genevieve_large_thermos_drink_count * large_thermos_capacity_ml)
  let genevieve_ounces_drink := genevieve_ml_drink / mls_per_ounce
  genevieve_ounces_drink = 33.814 :=
by sorry

end genevieve_coffee_drink_l333_333357


namespace correct_transformation_l333_333652

theorem correct_transformation (a b m : ℝ) (h : m ≠ 0) : (am / bm) = (a / b) :=
by sorry

end correct_transformation_l333_333652


namespace find_x_l333_333196

theorem find_x (x : ℝ) (h : (1 / 2) * x + (1 / 3) * x = (1 / 4) * x + 7) : x = 12 :=
by
  sorry

end find_x_l333_333196


namespace domino_cover_l333_333553

-- Definitions
def is_valid_domino_cover (squares_removed: set (ℕ, ℕ)) : Prop :=
  ∃ doms : set (set (ℕ, ℕ)), 
    (∀ d ∈ doms, d.card = 2) ∧  -- Each domino covers exactly two squares
    (∀ d ∈ doms, ∃ b w: (ℕ, ℕ), b ∈ d ∧ w ∈ d ∧ b ≠ w ∧ (b.1 + b.2) % 2 ≠ (w.1 + w.2) % 2) ∧ -- Each domino consists of one black and one white square
    (disjoint doms.unions) ∧ -- Dominoes do not overlap
    (doms.unions = {(i, j) | i < 8 ∧ j < 8} \ squares_removed) -- Remaining squares must be exactly all and only those not in squares_removed

-- Theorem
theorem domino_cover (squares_removed: set (ℕ, ℕ)) (h: squares_removed.card = 2): 
  is_valid_domino_cover squares_removed ↔ 
  ∃ b w : (ℕ, ℕ), b ∈ squares_removed ∧ w ∈ squares_removed ∧ 
    (b.1 + b.2) % 2 ≠ (w.1 + w.2) % 2 := -- Squares removed are of different colors
by
  sorry

end domino_cover_l333_333553


namespace larger_triangle_legs_sum_l333_333551

theorem larger_triangle_legs_sum (a b A: ℕ) (c A1  A2 : ℝ) 
(ha : A = 10) (hA : A1 = 360) (hc : c = 6) 
(h_similar: (A2 / A = 36) ): √30) )

(h_alt : (0.5 * a * b = 10 )) :
  (a + b = 16*sqrt 30) ):

end larger_triangle_legs_sum_l333_333551


namespace subtraction_identity_l333_333560

theorem subtraction_identity : 3.57 - 1.14 - 0.23 = 2.20 := sorry

end subtraction_identity_l333_333560


namespace euclidean_steps_arbitrarily_large_l333_333493

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

theorem euclidean_steps_arbitrarily_large (n : ℕ) (h : n ≥ 2) :
  gcd (fib (n+1)) (fib n) = gcd (fib 1) (fib 0) := 
sorry

end euclidean_steps_arbitrarily_large_l333_333493


namespace number_of_partners_l333_333524

def total_profit : ℝ := 80000
def majority_owner_share := 0.25 * total_profit
def remaining_profit := total_profit - majority_owner_share
def partner_share := 0.25 * remaining_profit
def combined_share := majority_owner_share + 2 * partner_share

theorem number_of_partners : combined_share = 50000 → remaining_profit / partner_share = 4 := by
  intro h1
  have h_majority : majority_owner_share = 0.25 * total_profit := by sorry
  have h_remaining : remaining_profit = total_profit - majority_owner_share := by sorry
  have h_partner : partner_share = 0.25 * remaining_profit := by sorry
  have h_combined : combined_share = majority_owner_share + 2 * partner_share := by sorry
  calc
    remaining_profit / partner_share = _ := by sorry
    4 = 4 := by sorry

end number_of_partners_l333_333524


namespace sqrt_factorial_squared_l333_333566

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333566


namespace average_speed_trip_l333_333826

-- Define the speeds
def speed1 := 80 -- km/h
def speed2 := 24 -- km/h
def speed3 := 54 -- km/h
def speed4 := 36 -- km/h

-- Define the total distance and the proportions of the distance
variable (D : ℝ) (D_pos : D > 0)

-- Define the times taken for each portion of the trip
def time1 := (D / 4) / speed1
def time2 := (D / 4) / speed2
def time3 := (D / 4) / speed3
def time4 := (D / 4) / speed4

-- Define the total time for the trip
def total_time := time1 + time2 + time3 + time4

-- Define the average speed
def average_speed := D / total_time

-- The theorem stating the average speed of the trip
theorem average_speed_trip : average_speed = 39.84 := by
  sorry

end average_speed_trip_l333_333826


namespace area_enclosed_by_equation_l333_333735

noncomputable def area_of_bounded_region : ℝ :=
  let f (x y : ℝ) : ℝ := y ^ 2 + 3 * x * y + 30 * |x|
  if h : f x y = 300 then 400 else 0

theorem area_enclosed_by_equation : area_of_bounded_region = 400 :=
by
  -- Proof will go here
  sorry

end area_enclosed_by_equation_l333_333735


namespace line_passes_through_fixed_point_l333_333943

theorem line_passes_through_fixed_point (m : ℝ) : ∃ (x y : ℝ), (2 * x + m * (x - y) - 1 = 0) ∧ x = 1 / 2 ∧ y = 1 / 2 :=
by
  use 1 / 2, 1 / 2
  split
  · calc
    2 * (1 / 2) + m * ((1 / 2) - (1 / 2)) - 1 = 1 + m * 0 - 1 : by ring
                                 ... = 0 : by ring
                                   
  split
  · rfl
  · rfl

end line_passes_through_fixed_point_l333_333943


namespace range_of_a_l333_333023

theorem range_of_a (a : ℝ) (h : a > 0) (g : ℝ → ℝ := λ x, a * x + 2) (f : ℝ → ℝ := λ x, x^2 + 2 * x) :
  (∀ x1 ∈ Icc (-1 : ℝ) 1, ∃ x0 ∈ Icc (-2 : ℝ) 1, g x1 = f x0) → a ∈ Ioc 0 1 :=
by
  sorry

end range_of_a_l333_333023


namespace rosencrantz_guildenstern_prob_l333_333919

-- Define the sequence a_n
def a : ℕ → ℕ
| 1       := 4
| 2       := 3
| n + 1   := a n + a (n - 1) /* definition starts from n = 2 */

-- Define the probabilities for heads and tails in flipping a fair coin
def fair_coin_prob := 1 / 2

-- Definition of the probability calculation function for the given problem
def calc_prob (n : ℕ) : ℚ := 1 / 2 - 1 / (2^(n+1))

-- Theorem statement to prove the probability calculation for Rosencrantz's game
theorem rosencrantz_guildenstern_prob :
  calc_prob 1340 = 1 / 2 - 1 / (2^1341) :=
  sorry

end rosencrantz_guildenstern_prob_l333_333919


namespace find_natural_numbers_l333_333979

theorem find_natural_numbers (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 3^5) : 
  (x = 6 ∧ y = 3) := 
sorry

end find_natural_numbers_l333_333979


namespace circle_radius_l333_333760

theorem circle_radius (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y + 16 = 0 → sqrt ((4:ℝ)^2) = 2 :=
by
  sorry

end circle_radius_l333_333760


namespace Mike_gave_20_pens_l333_333232

theorem Mike_gave_20_pens (M : ℕ) : 
  let initial_pens := 5 in
  let after_M_pens := initial_pens + M in
  let after_doubling := 2 * after_M_pens in
  let after_giving_sharon := after_doubling - 19 in
  after_giving_sharon = 31 → M = 20 :=
by
  intro h
  sorry

end Mike_gave_20_pens_l333_333232


namespace area_enclosed_by_circle_l333_333734

theorem area_enclosed_by_circle :
  let center := (3, -10)
  let radius := 3
  let equation := ∀ (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2
  ∃ enclosed_area : ℝ, enclosed_area = 9 * Real.pi :=
by
  sorry

end area_enclosed_by_circle_l333_333734


namespace cousins_in_rooms_l333_333134

theorem cousins_in_rooms (cousins rooms : ℕ) (garden_view non_view : ℕ)
  (cond1 : cousins = 5) (cond2 : rooms = 4) (cond3 : garden_view = 2)
  (cond4 : non_view = 2) (cond5 : ∃ g, g ≥ 1 ∧ g ≤ 5 := true) :
  ∃ n, n = 30 := sorry

end cousins_in_rooms_l333_333134


namespace solve_triangle_AC_eq_3_l333_333419

noncomputable def triangle_AC_eq_3 (BC AB : ℝ) (cos_C : ℝ) : Prop :=
  BC = 2 ∧ AB = 4 ∧ cos_C = -1/4 → AC = 3

theorem solve_triangle_AC_eq_3 : triangle_AC_eq_3 2 4 (-1/4) :=
by
  sorry

end solve_triangle_AC_eq_3_l333_333419


namespace acute_angle_comparison_l333_333733

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem acute_angle_comparison (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (f_even : even_function f)
  (f_periodic : ∀ x, f (x + 1) + f x = 0)
  (f_increasing : increasing_on_interval f 3 4) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end acute_angle_comparison_l333_333733


namespace selection_representatives_count_l333_333202

theorem selection_representatives_count (boys girls : ℕ) (specific_girl specific_boy : ℕ) 
  (chinese_rep : specific_girl ∈ girls)
  (math_rep : specific_boy ∈ boys ∧ specific_boy ≠ specific_girl)
  (girls_less_than_boys : ∀ g b, g ∈ girls → b ∈ boys → specific_girl ≠ g ∧ specific_boy ≠ b) :
  ∑ (g : ℕ) (h : g < boys), binom girls 2 * binom boys 3 * perm (boys - 1) 5 + 
  ∑ (g : ℕ) (h : g < boys), (binom girls 1 * binom boys 4 + binom girls 2 * binom boys 3) * perm boys 4 = 360 := 
by
  sorry

end selection_representatives_count_l333_333202


namespace triangles_might_not_be_congruent_l333_333865

theorem triangles_might_not_be_congruent
  (A B C A1 B1 C1 : Point)
  (angle_A angle_B angle_C angle_A1 angle_B1 angle_C1 : ℝ)
  (AB A1B1 BC B1C1 : ℝ)
  (h_angleA : ∠ A B C = ∠ A1 B1 C1)
  (h_angleB : ∠ B C A = ∠ B1 C1 A1)
  (h_angleC : ∠ C A B = ∠ C1 A1 B1)
  (h_AB_len : dist A B = dist A1 B1)
  (h_BC_len : dist B C = dist B1 C1) :
  ∃ (a b c a1 b1 c1 : ℝ), a ≠ a1 ∧ (a / a1 = b / b1 ∧ b / b1 = c / c1) :=
by
  sorry

end triangles_might_not_be_congruent_l333_333865


namespace sqrt_factorial_mul_square_l333_333645

theorem sqrt_factorial_mul_square (h1 : fact 5 = 120) (h2 : fact 4 = 24) : (sqrt (fact 5 * fact 4))^2 = 2880 :=
by
  sorry

end sqrt_factorial_mul_square_l333_333645


namespace constant_speed_40_min_fuel_consumption_l333_333709

-- Define the fuel consumption function
def fuel_consumption (x : ℝ) : ℝ :=
  (1 / 128000) * x^3 - (3 / 80) * x + 8

-- Define the total fuel consumption over the journey
def total_fuel_consumption (x : ℝ) : ℝ :=
  fuel_consumption(x) * 100 / x

-- The distance between A and B
def distance_A_to_B : ℝ := 100

-- Prove that the fuel needed to travel 100 km at 40 km/h is 17.5 liters
theorem constant_speed_40 :
  total_fuel_consumption 40 = 17.5 :=
  sorry

-- Prove that the speed minimizing fuel consumption is 80 km/h, with minimum fuel consumption being 11.25 liters
theorem min_fuel_consumption :
  ∃ x : ℝ, 0 < x ∧ x ≤ 120 ∧ (total_fuel_consumption x = 11.25) :=
  sorry

end constant_speed_40_min_fuel_consumption_l333_333709


namespace compute_ab_l333_333938

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end compute_ab_l333_333938


namespace right_triangle_cosine_l333_333844

theorem right_triangle_cosine (R P T Q : Type) 
  (h1 : ∠ RTP = 90)
  (h2 : sin ∠ PRQ = 3/5)
  (h3 : ∠ PRQ = 180 - ∠ RPT) :
  cos ∠ RPQ = 4/5 := 
sorry

end right_triangle_cosine_l333_333844


namespace sqrt_factorial_sq_l333_333588

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l333_333588


namespace ab_div_10_eq_401_l333_333310

def double_factorial : ℕ → ℕ 
| 0     := 1
| 1     := 1
| (n+2) := (n + 2) * double_factorial n

def S :=
  ∑ i in finset.range 2009, (nat.choose (2*i) i / 2^(2*i))

theorem ab_div_10_eq_401 :
  let a := 4010 in
  let b := 1 in
  (a * b) / 10 = 401 :=
by
  sorry

end ab_div_10_eq_401_l333_333310


namespace set_equality_l333_333131

theorem set_equality (a : ℝ) : 
  ({1, -2, a^2 - 1} : Set ℝ) = {1, a^2 - 3a, 0} → a = 1 :=
by
  sorry

end set_equality_l333_333131


namespace common_difference_arithmetic_sequence_l333_333081

theorem common_difference_arithmetic_sequence
  (a : ℕ) (a_n : ℕ) (S_n : ℕ) (d : ℕ) (n : ℕ)
  (h1 : a = 3) (h2 : a_n = 48) (h3 : S_n = 255)
  (hnth : a_n = a + (n - 1) * d) (hsum : S_n = n * (a + a_n) / 2) :
  d = 5 :=
by
  -- Given data
  have h_an : 48 = 3 + (n - 1) * d := by rw [h1, h2, a_n_eq]
  have h_sum : 255 = n * (3 + 48) / 2 := by rw [h1, h2, h3]
  
  -- From h_sum calculate n
  have h_n : 510 = 51 * n := by linarith
  have hn : n = 10 := by linarith
  
  -- Combine with h_an
  linarith

end common_difference_arithmetic_sequence_l333_333081


namespace hous_alkali_process_developer_l333_333076

-- Define the condition that "Hou's Alkali Process" was successfully developed in March 1941.
def developed_in_march_1941 : Prop :=
  "Hou's Alkali Process was developed in March 1941"

-- Define the condition that the process is praised as "Hou's Alkali Process".
def praised_as_hous_alkali_process : Prop :=
  "The process is praised as 'Hou's Alkali Process'"

-- Define the four potential developers.
inductive Developer
| Hou_Debang
| Hou_Guangtian
| Hou_Xianglin
| Hou_Xueyu

-- State the problem as a theorem.
theorem hous_alkali_process_developer
  (d1 : developed_in_march_1941)
  (d2 : praised_as_hous_alkali_process) :
  Developer.Hou_Debang = true :=
sorry

end hous_alkali_process_developer_l333_333076


namespace sum_of_reciprocals_l333_333970

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) : 
  1/x + 1/y = 14/45 := 
sorry

end sum_of_reciprocals_l333_333970


namespace number_of_children_l333_333165

theorem number_of_children (x : ℕ) : 3 * x + 12 = 5 * x - 10 → x = 11 :=
by
  intros h
  have : 3 * x + 12 = 5 * x - 10 := h
  sorry

end number_of_children_l333_333165


namespace log_relation_l333_333117

noncomputable def a := Real.log 3 / Real.log 4
noncomputable def b := Real.log 3 / Real.log 0.4
def c := (1 / 2) ^ 2

theorem log_relation (h1 : a = Real.log 3 / Real.log 4)
                     (h2 : b = Real.log 3 / Real.log 0.4)
                     (h3 : c = (1 / 2) ^ 2) : a > c ∧ c > b :=
by
  sorry

end log_relation_l333_333117


namespace find_p_l333_333983

theorem find_p (P Q R S T : ℕ)
  (h1 : {P, Q, R, S, T} ⊆ {1, 2, 3, 4, 5})
  (h2 : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T)
  (h3 : Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
  (h4 : (10 * P + Q) * 10 + R ≡ 0 [MOD 4])
  (h5 : (10 * Q + R) * 10 + S ≡ 0 [MOD 5])
  (h6 : (10 * R + S) * 10 + T ≡ 0 [MOD 3]) :
  P = 1 :=
sorry

end find_p_l333_333983


namespace num_real_solutions_abs_eq_l333_333814

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end num_real_solutions_abs_eq_l333_333814


namespace sum_of_three_consecutive_is_50_l333_333086

-- Define a sequence of integers
def seq : Fin 8 → ℕ
| 0 => 11
| 1 => 12
| 2 => 27
| 3 => 11
| 4 => 12
| 5 => 27
| 6 => 11
| 7 => 12

-- Theorem: The sum of any three consecutive numbers in the sequence is 50
theorem sum_of_three_consecutive_is_50 (n : Fin (8 - 2)) : 
  seq n + seq (n + 1) + seq (n + 2) = 50 :=
by
  fin_cases n
  case 0 => simp [seq]
  case 1 => simp [seq]
  case 2 => simp [seq]
  case 3 => simp [seq]
  case 4 => simp [seq]
  case 5 => simp [seq]
  -- This skips the proof details; the proof should show the sum matches 50 in all cases.
  sorry

-- Use sorry to skip the detailed proof

end sum_of_three_consecutive_is_50_l333_333086


namespace factorization_left_to_right_l333_333226

-- Definitions (conditions)
def exprD_lhs : ℝ → ℝ := λ x, x^2 - 9
def exprD_rhs : ℝ → ℝ := λ x, (x + 3) * (x - 3)

-- Statement
theorem factorization_left_to_right (x : ℝ) :
  exprD_lhs x = exprD_rhs x := 
by sorry

end factorization_left_to_right_l333_333226


namespace limit_f_k_div_k_l333_333762

-- Define f_k based on the problem condition.
def f_k (k : ℕ) : ℕ := (finset.Icc 0 1).filter (λ x, (Real.sin (k * Real.pi * x / 2)) = 1).card

-- Prove the limit
theorem limit_f_k_div_k : 
  tendsto (λ k: ℕ, (f_k k : ℝ) / (k : ℝ)) atTop (𝓝 (1 / 4)) :=
sorry

end limit_f_k_div_k_l333_333762


namespace volume_of_rectangular_prism_l333_333096

-- Given conditions translated into Lean definitions
variables (AB AD AC1 AA1 : ℕ)

def rectangular_prism_properties : Prop :=
  AB = 2 ∧ AD = 2 ∧ AC1 = 3 ∧ AA1 = 1

-- The mathematical volume of the rectangular prism
def volume (AB AD AA1 : ℕ) := AB * AD * AA1

-- Prove that given the conditions, the volume of the rectangular prism is 4
theorem volume_of_rectangular_prism (h : rectangular_prism_properties AB AD AC1 AA1) : volume AB AD AA1 = 4 :=
by
  sorry

#check volume_of_rectangular_prism

end volume_of_rectangular_prism_l333_333096


namespace volume_of_cube_in_pyramid_l333_333843

open Real

noncomputable def side_length_of_base := 2
noncomputable def height_of_equilateral_triangle := sqrt 6
noncomputable def cube_side_length := sqrt 6 / 3
noncomputable def volume_of_cube := cube_side_length ^ 3

theorem volume_of_cube_in_pyramid 
  (side_length_of_base : ℝ) (height_of_equilateral_triangle : ℝ) (cube_side_length : ℝ) :
  volume_of_cube = 2 * sqrt 6 / 9 := 
by
  sorry

end volume_of_cube_in_pyramid_l333_333843


namespace shaded_region_correct_area_l333_333842

-- Define the side length of the regular octagon
def side_length : ℝ := 3

-- Define the area of the regular octagon with side length 3
def octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * side_length ^ 2

-- Define the radius of the semicircles as half the side length
def semicircle_radius : ℝ := side_length / 2

-- Define the area of one semicircle
def semicircle_area : ℝ := (1 / 2) * Real.pi * semicircle_radius ^ 2

-- Define the total area of the 8 semicircles
def total_semicircles_area : ℝ := 8 * semicircle_area

-- The goal is to calculate the area of the octagon not covered by semicircles
def shaded_region_area (oct_area : ℝ) (semi_area : ℝ) : ℝ := oct_area - semi_area

-- Now express the final goal
theorem shaded_region_correct_area : 
  shaded_region_area octagon_area total_semicircles_area = 18 * (1 + Real.sqrt 2) - 9 * Real.pi :=
by
  sorry

end shaded_region_correct_area_l333_333842


namespace count_multiples_6_not_12_l333_333034

theorem count_multiples_6_not_12 (n: ℕ) : 
  ∃ (count : ℕ), count = 25 ∧ 
                  count = (finset.filter (λ m, (m < 300) ∧ (6 ∣ m) ∧ ¬ (12 ∣ m)) (finset.range 300)).card :=
by
  sorry

end count_multiples_6_not_12_l333_333034


namespace positive_difference_proof_l333_333197

/-- Definitions used in the conditions -/
variables {x y : ℝ}
def sum_eq_40 (x y : ℝ) : Prop := x + y = 40
def triple_subtract_eq_10 (x y : ℝ) : Prop := 3 * y - 4 * x = 10
def positive_difference (x y : ℝ) : ℝ := abs (y - x)

/-- Lean 4 theorem statement -/
theorem positive_difference_proof (hx : sum_eq_40 x y) (hy : triple_subtract_eq_10 x y) :
  positive_difference x y = 8.58 :=
sorry

end positive_difference_proof_l333_333197


namespace sqrt_factorial_squared_l333_333567

theorem sqrt_factorial_squared (h5fac: fact 5) (h4fac: fact 4) :
  (real.sqrt (h5fac * h4fac))^2 = 2880 := by
  sorry

end sqrt_factorial_squared_l333_333567


namespace pythagorean_ratio_l333_333932

variables (a b : ℝ)

theorem pythagorean_ratio (h1 : a > 0) (h2 : b > a) (h3 : b^2 = 13 * (b - a)^2) :
  a / b = 2 / 3 :=
sorry

end pythagorean_ratio_l333_333932


namespace pencils_per_student_l333_333499

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ)
    (h1 : total_pencils = 125)
    (h2 : students = 25)
    (h3 : pencils_per_student = total_pencils / students) :
    pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l333_333499


namespace sales_tax_difference_l333_333513

theorem sales_tax_difference
  (item_price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.0725)
  (h_rate2 : rate2 = 0.0675)
  (h_item_price : item_price = 40) :
  item_price * rate1 - item_price * rate2 = 0.20 :=
by
  -- Since we are required to skip the proof, we put sorry here.
  sorry

end sales_tax_difference_l333_333513


namespace exists_positive_sequence_l333_333872

theorem exists_positive_sequence
  (X : ℕ → ℝ^3) (Z : ℝ^3)
  (hX_countable : ∀ n : ℕ, ∃ k : ℕ, X k = X n)
  (hZ_notin_X : ∀ n : ℕ, Z ≠ X n) :
  ∃ (a : ℕ → ℝ), (∀ k, 0 < a k) ∧ (∀ k, ∃ n ≥ k, ∀ m < k, dist Z (X m) ≥ a n) :=
sorry

end exists_positive_sequence_l333_333872


namespace no_contradiction_to_thermodynamics_l333_333351

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l333_333351


namespace hyperbola_eccentricity_l333_333855

variable {a b c e : ℝ}
-- Given conditions
variable (a_pos : a > 0) (b_pos : b > 0)
variable (hyperbola_eqn : ∀ x y : ℝ, ( x = c) → (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1))
variable (points_BC : (B C : ℝ × ℝ) → B = (c, b^2 / a) ∧ C = (c, -b^2 / a))
variable (right_angle_triangle : ∃ (A B C : ℝ × ℝ), A = (-a, 0) ∧ (B = (c, b^2 / a) ∧ C = (c, -b^2 / a) ∧ (∃ right_angle : (∠ABC = π/2)))

-- Proof of the eccentricity.
theorem hyperbola_eccentricity (hyperbola_eqn : ∀ x y : ℝ, ( x = c) → (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1))
    (points_BC : ∀ B C : ℝ × ℝ, B = (c, b^2 / a) ∧ C = (c, -b^2 / a))
    (right_angle_triangle : ∃ (A B C : ℝ × ℝ), A = (-a, 0) ∧ B = (c, b^2 / a) ∧ C = (c, -b^2 / a) ∧ (∠ABC = π/2))
    : e = 2 := by
  sorry

end hyperbola_eccentricity_l333_333855


namespace wrens_population_below_10_percent_l333_333750

def wrens_population_decline : ℕ :=
  let percentage_after_n_years (n : ℕ) : ℝ := 0.6^n
  let threshold := 0.1
  -- n is the number of years after 2004
  find (λ n, percentage_after_n_years n < threshold) sorry

theorem wrens_population_below_10_percent : wrens_population_decline = 5 :=
by
  -- Given threshold year is 2004, and n = 5 means year 2004 + 5 = 2009
  -- Find year when population is below 10% of the original population
  sorry

end wrens_population_below_10_percent_l333_333750
