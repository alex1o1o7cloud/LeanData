import Mathlib

namespace sqrt_meaningful_range_l1536_153610

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 3) : 3 ≤ x :=
by
  linarith

end sqrt_meaningful_range_l1536_153610


namespace average_payment_52_installments_l1536_153685

theorem average_payment_52_installments :
  let first_payment : ℕ := 500
  let remaining_payment : ℕ := first_payment + 100
  let num_first_payments : ℕ := 25
  let num_remaining_payments : ℕ := 27
  let total_payments : ℕ := num_first_payments + num_remaining_payments
  let total_paid_first : ℕ := num_first_payments * first_payment
  let total_paid_remaining : ℕ := num_remaining_payments * remaining_payment
  let total_paid : ℕ := total_paid_first + total_paid_remaining
  let average_payment : ℚ := total_paid / total_payments
  average_payment = 551.92 :=
by
  sorry

end average_payment_52_installments_l1536_153685


namespace work_completion_days_l1536_153673

theorem work_completion_days (D : ℕ) 
  (h : 40 * D = 48 * (D - 10)) : D = 60 := 
sorry

end work_completion_days_l1536_153673


namespace towel_bleach_volume_decrease_l1536_153697

theorem towel_bleach_volume_decrease :
  ∀ (L B T : ℝ) (L' B' T' : ℝ),
  (L' = L * 0.75) →
  (B' = B * 0.70) →
  (T' = T * 0.90) →
  (L * B * T = 1000000) →
  ((L * B * T - L' * B' * T') / (L * B * T) * 100) = 52.75 :=
by
  intros L B T L' B' T' hL' hB' hT' hV
  sorry

end towel_bleach_volume_decrease_l1536_153697


namespace find_n_l1536_153693

variable {a : ℕ → ℝ}  -- Defining the sequence

-- Defining the conditions:
def a1 : Prop := a 1 = 1 / 3
def a2_plus_a5 : Prop := a 2 + a 5 = 4
def a_n_eq_33 (n : ℕ) : Prop := a n = 33

theorem find_n (n : ℕ) : a 1 = 1 / 3 → (a 2 + a 5 = 4) → (a n = 33) → n = 50 := 
by 
  intros h1 h2 h3 
  -- the complete proof can be done here
  sorry

end find_n_l1536_153693


namespace solve_fractional_equation_l1536_153651

theorem solve_fractional_equation (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ -6) :
    (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9 / 4 :=
by
  sorry

end solve_fractional_equation_l1536_153651


namespace fraction_sum_is_0_333_l1536_153616

theorem fraction_sum_is_0_333 : (3 / 10 : ℝ) + (3 / 100) + (3 / 1000) = 0.333 := 
by
  sorry

end fraction_sum_is_0_333_l1536_153616


namespace distribute_items_in_identical_bags_l1536_153671

noncomputable def count_ways_to_distribute_items (num_items : ℕ) (num_bags : ℕ) : ℕ :=
  if h : num_items = 5 ∧ num_bags = 3 then 36 else 0

theorem distribute_items_in_identical_bags :
  count_ways_to_distribute_items 5 3 = 36 :=
by
  -- Proof is skipped as per instructions
  sorry

end distribute_items_in_identical_bags_l1536_153671


namespace pascal_triangle_row_20_sum_l1536_153656

theorem pascal_triangle_row_20_sum :
  (Nat.choose 20 2) + (Nat.choose 20 3) + (Nat.choose 20 4) = 6175 :=
by
  sorry

end pascal_triangle_row_20_sum_l1536_153656


namespace annual_return_l1536_153688

theorem annual_return (initial_price profit : ℝ) (h₁ : initial_price = 5000) (h₂ : profit = 400) : 
  ((profit / initial_price) * 100 = 8) := by
  -- Lean's substitute for proof
  sorry

end annual_return_l1536_153688


namespace aaron_ends_up_with_24_cards_l1536_153687

def initial_cards_aaron : Nat := 5
def found_cards_aaron : Nat := 62
def lost_cards_aaron : Nat := 15
def given_cards_to_arthur : Nat := 28

def final_cards_aaron (initial: Nat) (found: Nat) (lost: Nat) (given: Nat) : Nat :=
  initial + found - lost - given

theorem aaron_ends_up_with_24_cards :
  final_cards_aaron initial_cards_aaron found_cards_aaron lost_cards_aaron given_cards_to_arthur = 24 := by
  sorry

end aaron_ends_up_with_24_cards_l1536_153687


namespace sufficient_not_necessary_condition_l1536_153627

open Complex

theorem sufficient_not_necessary_condition (a b : ℝ) (i := Complex.I) :
  (a = 1 ∧ b = 1) → ((a + b * i)^2 = 2 * i) ∧ ¬((a + b * i)^2 = 2 * i → a = 1 ∧ b = 1) :=
by
  sorry

end sufficient_not_necessary_condition_l1536_153627


namespace wide_flags_made_l1536_153636

theorem wide_flags_made
  (initial_fabric : ℕ) (square_flag_side : ℕ) (wide_flag_width : ℕ) (wide_flag_height : ℕ)
  (tall_flag_width : ℕ) (tall_flag_height : ℕ) (made_square_flags : ℕ) (made_tall_flags : ℕ)
  (remaining_fabric : ℕ) (used_fabric_for_small_flags : ℕ) (used_fabric_for_tall_flags : ℕ)
  (used_fabric_for_wide_flags : ℕ) (wide_flag_area : ℕ) :
    initial_fabric = 1000 →
    square_flag_side = 4 →
    wide_flag_width = 5 →
    wide_flag_height = 3 →
    tall_flag_width = 3 →
    tall_flag_height = 5 →
    made_square_flags = 16 →
    made_tall_flags = 10 →
    remaining_fabric = 294 →
    used_fabric_for_small_flags = 256 →
    used_fabric_for_tall_flags = 150 →
    used_fabric_for_wide_flags = initial_fabric - remaining_fabric - (used_fabric_for_small_flags + used_fabric_for_tall_flags) →
    wide_flag_area = wide_flag_width * wide_flag_height →
    (used_fabric_for_wide_flags / wide_flag_area) = 20 :=
by
  intros; 
  sorry

end wide_flags_made_l1536_153636


namespace binomial_cubes_sum_l1536_153649

theorem binomial_cubes_sum (x y : ℤ) :
  let B1 := x^4 + 9 * x * y^3
  let B2 := -(3 * x^3 * y) - 9 * y^4
  (B1 ^ 3 + B2 ^ 3 = x ^ 12 - 729 * y ^ 12) := by
  sorry

end binomial_cubes_sum_l1536_153649


namespace time_spent_on_seals_l1536_153652

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l1536_153652


namespace sequence_bounds_l1536_153667

theorem sequence_bounds (c : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = ↑n + c / ↑n) 
  (h2 : ∀ n : ℕ+, a n ≥ a 3) : 6 ≤ c ∧ c ≤ 12 :=
by 
  -- We will prove that 6 ≤ c and c ≤ 12 given the conditions stated
  sorry

end sequence_bounds_l1536_153667


namespace find_negative_number_l1536_153632

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end find_negative_number_l1536_153632


namespace digit_2023_in_fractional_expansion_l1536_153643

theorem digit_2023_in_fractional_expansion :
  ∃ d : ℕ, (d = 4) ∧ (∃ n_block : ℕ, n_block = 6 ∧ (∃ p : Nat, p = 2023 ∧ ∃ r : ℕ, r = p % n_block ∧ r = 1)) :=
sorry

end digit_2023_in_fractional_expansion_l1536_153643


namespace cuboid_unshaded_face_area_l1536_153621

theorem cuboid_unshaded_face_area 
  (x : ℝ)
  (h1 : ∀ a  : ℝ, a = 4*x) -- Condition: each unshaded face area = 4 * shaded face area
  (h2 : 18*x = 72)         -- Condition: total surface area = 72 cm²
  : 4*x = 16 :=            -- Conclusion: area of one visible unshaded face is 16 cm²
by
  sorry

end cuboid_unshaded_face_area_l1536_153621


namespace probability_of_ram_l1536_153658

theorem probability_of_ram 
  (P_ravi : ℝ) (P_both : ℝ) 
  (h_ravi : P_ravi = 1 / 5) 
  (h_both : P_both = 0.11428571428571428) : 
  ∃ P_ram : ℝ, P_ram = 0.5714285714285714 :=
by
  sorry

end probability_of_ram_l1536_153658


namespace cylinder_radius_eq_3_l1536_153672

theorem cylinder_radius_eq_3 (r : ℝ) : 
  (π * (r + 4)^2 * 3 = π * r^2 * 11) ∧ (r >= 0) → r = 3 :=
by 
  sorry

end cylinder_radius_eq_3_l1536_153672


namespace central_park_trash_cans_more_than_half_l1536_153604

theorem central_park_trash_cans_more_than_half
  (C : ℕ)  -- Original number of trash cans in Central Park
  (V : ℕ := 24)  -- Original number of trash cans in Veteran's Park
  (V_now : ℕ := 34)  -- Number of trash cans in Veteran's Park after the move
  (H_move : (V_now - V) = C / 2)  -- Condition of trash cans moved
  (H_C : C = (1 / 2) * V + x)  -- Central Park had more than half trash cans as Veteran's Park, where x is an excess amount
  : C - (1 / 2) * V = 8 := 
sorry

end central_park_trash_cans_more_than_half_l1536_153604


namespace part_I_part_II_l1536_153691

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x

theorem part_I (a : ℝ) (h_a : a ≠ 0) :
  (∃ x : ℝ, (x * (f a (1/x))) = 4 * x - 3 ∧ ∀ y, x = y → (x * (f a (1/x))) = 4 * x - 3) →
  a = 2 :=
sorry

noncomputable def f2 (x : ℝ) : ℝ := 2 / x - x

theorem part_II : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f2 x1 > f2 x2 :=
sorry

end part_I_part_II_l1536_153691


namespace train_length_is_correct_l1536_153654

noncomputable def speed_kmph : ℝ := 72
noncomputable def time_seconds : ℝ := 74.994
noncomputable def tunnel_length_m : ℝ := 1400
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600
noncomputable def total_distance : ℝ := speed_mps * time_seconds
noncomputable def train_length : ℝ := total_distance - tunnel_length_m

theorem train_length_is_correct :
  train_length = 99.88 := by
  -- the proof will follow here
  sorry

end train_length_is_correct_l1536_153654


namespace g_50_unique_l1536_153615

namespace Proof

-- Define the function g and the condition it should satisfy
variable (g : ℕ → ℕ)
variable (h : ∀ (a b : ℕ), 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b))

theorem g_50_unique : ∃ (m t : ℕ), m * t = 0 := by
  -- Existence of m and t fulfilling the condition
  -- Placeholder for the proof
  sorry

end Proof

end g_50_unique_l1536_153615


namespace candies_shared_l1536_153637

theorem candies_shared (y b d x : ℕ) (h1 : x = 2 * y + 10) (h2 : x = 3 * b + 18) (h3 : x = 5 * d - 55) (h4 : x + y + b + d = 2013) : x = 990 :=
by
  sorry

end candies_shared_l1536_153637


namespace smallest_n_l1536_153698

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end smallest_n_l1536_153698


namespace rectangle_perimeter_l1536_153647

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem rectangle_perimeter
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 l w : ℕ)
  (h1 : a1 + a2 + a3 = a9)
  (h2 : a1 + a2 = a3)
  (h3 : a1 + a3 = a4)
  (h4 : a3 + a4 = a5)
  (h5 : a4 + a5 = a6)
  (h6 : a2 + a3 + a5 = a7)
  (h7 : a2 + a7 = a8)
  (h8 : a1 + a4 + a6 = a9)
  (h9 : a6 + a9 = a7 + a8)
  (h_rel_prime : relatively_prime l w)
  (h_dimensions : l = 61)
  (h_dimensions_w : w = 69) :
  2 * l + 2 * w = 260 := by
  sorry

end rectangle_perimeter_l1536_153647


namespace no_solution_abs_eq_l1536_153624

theorem no_solution_abs_eq (x : ℝ) (h : x > 0) : |x + 4| = 3 - x → false :=
by
  sorry

end no_solution_abs_eq_l1536_153624


namespace find_number_l1536_153680

theorem find_number (x : ℕ) (h : x * 9999 = 724817410) : x = 72492 :=
sorry

end find_number_l1536_153680


namespace remainder_when_divided_by_8_l1536_153689

theorem remainder_when_divided_by_8 (x : ℤ) (h : ∃ k : ℤ, x = 72 * k + 19) : x % 8 = 3 :=
by
  sorry

end remainder_when_divided_by_8_l1536_153689


namespace min_value_of_a_l1536_153686

theorem min_value_of_a (a : ℝ) (h : ∃ x : ℝ, |x - 1| + |x + a| ≤ 8) : -9 ≤ a :=
by
  sorry

end min_value_of_a_l1536_153686


namespace probability_ephraim_keiko_l1536_153679

-- Define the probability that Ephraim gets a certain number of heads tossing two pennies
def prob_heads_ephraim (n : Nat) : ℚ :=
  if n = 2 then 1 / 4
  else if n = 1 then 1 / 2
  else if n = 0 then 1 / 4
  else 0

-- Define the probability that Keiko gets a certain number of heads tossing one penny
def prob_heads_keiko (n : Nat) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 0 then 1 / 2
  else 0

-- Define the probability that Ephraim and Keiko get the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads_ephraim 0 * prob_heads_keiko 0) + (prob_heads_ephraim 1 * prob_heads_keiko 1) + (prob_heads_ephraim 2 * prob_heads_keiko 2)

-- The statement that requires proof
theorem probability_ephraim_keiko : prob_same_heads = 3 / 8 := 
  sorry

end probability_ephraim_keiko_l1536_153679


namespace decrypt_message_base7_l1536_153663

noncomputable def base7_to_base10 : Nat := 
  2 * 343 + 5 * 49 + 3 * 7 + 4 * 1

theorem decrypt_message_base7 : base7_to_base10 = 956 := 
by 
  sorry

end decrypt_message_base7_l1536_153663


namespace pair_not_product_48_l1536_153655

theorem pair_not_product_48:
  (∀(a b : ℤ), (a, b) = (-6, -8)                    → a * b = 48) ∧
  (∀(a b : ℤ), (a, b) = (-4, -12)                   → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (3/4, -64)                  → a * b ≠ 48) ∧
  (∀(a b : ℤ), (a, b) = (3, 16)                     → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (4/3, 36)                   → a * b = 48)
  :=
by
  sorry

end pair_not_product_48_l1536_153655


namespace tree_height_l1536_153659

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end tree_height_l1536_153659


namespace equation_of_line_through_point_l1536_153633

theorem equation_of_line_through_point (a T : ℝ) (h : a ≠ 0 ∧ T ≠ 0) :
  ∃ k : ℝ, (k = T / (a^2)) ∧ (k * x + (2 * T / a)) = (k * x + (2 * T / a)) → 
  (T * x - a^2 * y + 2 * T * a = 0) :=
by
  use T / (a^2)
  sorry

end equation_of_line_through_point_l1536_153633


namespace total_weekly_pay_l1536_153653

theorem total_weekly_pay (Y_pay: ℝ) (X_pay: ℝ) (Y_weekly: Y_pay = 150) (X_weekly: X_pay = 1.2 * Y_pay) : 
  X_pay + Y_pay = 330 :=
by sorry

end total_weekly_pay_l1536_153653


namespace specific_time_l1536_153674

theorem specific_time :
  (∀ (s : ℕ), 0 ≤ s ∧ s ≤ 7 → (∃ (t : ℕ), (t ^ 2 + 2 * t) - (3 ^ 2 + 2 * 3) = 20 ∧ t = 5)) :=
  by sorry

end specific_time_l1536_153674


namespace tangency_lines_intersect_at_diagonal_intersection_point_l1536_153676

noncomputable def point := Type
noncomputable def line := Type

noncomputable def tangency (C : point) (l : line) : Prop := sorry
noncomputable def circumscribed (Q : point × point × point × point) (C : point) : Prop := sorry
noncomputable def intersects (l1 l2 : line) (P : point) : Prop := sorry
noncomputable def connects_opposite_tangency (Q : point × point × point × point) (l1 l2 : line) : Prop := sorry
noncomputable def diagonals_intersect_at (Q : point × point × point × point) (P : point) : Prop := sorry

theorem tangency_lines_intersect_at_diagonal_intersection_point :
  ∀ (Q : point × point × point × point) (C P : point), 
  circumscribed Q C →
  diagonals_intersect_at Q P →
  ∀ (l1 l2 : line), connects_opposite_tangency Q l1 l2 →
  intersects l1 l2 P :=
sorry

end tangency_lines_intersect_at_diagonal_intersection_point_l1536_153676


namespace tan_neg_405_eq_neg_1_l1536_153612

theorem tan_neg_405_eq_neg_1 :
  (Real.tan (-405 * Real.pi / 180) = -1) ∧
  (∀ θ : ℝ, Real.tan (θ + 2 * Real.pi) = Real.tan θ) ∧
  (Real.tan θ = Real.sin θ / Real.cos θ) ∧
  (Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2) ∧
  (Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2) :=
sorry

end tan_neg_405_eq_neg_1_l1536_153612


namespace series_sum_equals_one_l1536_153623

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 : ℝ)^(2 * (k + 1)) / ((3 : ℝ)^(2 * (k + 1)) - 1)

theorem series_sum_equals_one :
  series_sum = 1 :=
sorry

end series_sum_equals_one_l1536_153623


namespace directly_proportional_l1536_153684

-- Defining conditions
def A (x y : ℝ) : Prop := y = x + 8
def B (x y : ℝ) : Prop := (2 / (5 * y)) = x
def C (x y : ℝ) : Prop := (2 / 3) * x = y

-- Theorem stating that in the given equations, equation C shows direct proportionality
theorem directly_proportional (x y : ℝ) : C x y ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by
  sorry

end directly_proportional_l1536_153684


namespace inequality_proof_l1536_153675

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 := 
sorry

end inequality_proof_l1536_153675


namespace required_vases_l1536_153646

def vase_capacity_roses : Nat := 6
def vase_capacity_tulips : Nat := 8
def vase_capacity_lilies : Nat := 4

def remaining_roses : Nat := 20
def remaining_tulips : Nat := 15
def remaining_lilies : Nat := 5

def vases_for_roses : Nat := (remaining_roses + vase_capacity_roses - 1) / vase_capacity_roses
def vases_for_tulips : Nat := (remaining_tulips + vase_capacity_tulips - 1) / vase_capacity_tulips
def vases_for_lilies : Nat := (remaining_lilies + vase_capacity_lilies - 1) / vase_capacity_lilies

def total_vases_needed : Nat := vases_for_roses + vases_for_tulips + vases_for_lilies

theorem required_vases : total_vases_needed = 8 := by
  sorry

end required_vases_l1536_153646


namespace solve_for_x_l1536_153669

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h_eq : x^4 = 6561) : x = 9 :=
sorry

end solve_for_x_l1536_153669


namespace cos_150_degree_l1536_153644

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l1536_153644


namespace union_A_B_eq_C_l1536_153692

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
noncomputable def C : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem union_A_B_eq_C : A ∪ B = C := by
  sorry

end union_A_B_eq_C_l1536_153692


namespace gcd_lcm_product_l1536_153648

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 75) (h2 : b = 90) : Nat.gcd a b * Nat.lcm a b = 6750 :=
by
  sorry

end gcd_lcm_product_l1536_153648


namespace probability_of_B_winning_is_correct_l1536_153613

noncomputable def prob_A_wins : ℝ := 0.2
noncomputable def prob_draw : ℝ := 0.5
noncomputable def prob_B_wins : ℝ := 1 - (prob_A_wins + prob_draw)

theorem probability_of_B_winning_is_correct : prob_B_wins = 0.3 := by
  sorry

end probability_of_B_winning_is_correct_l1536_153613


namespace ellipse_foci_coordinates_l1536_153699

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (y^2 / 3 + x^2 / 2 = 1) → (x, y) = (0, -1) ∨ (x, y) = (0, 1) :=
by
  sorry

end ellipse_foci_coordinates_l1536_153699


namespace local_minimum_point_l1536_153628

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_point (a : ℝ) (h : ∃ δ > 0, ∀ x, abs (x - a) < δ → f x ≥ f a) : a = 2 :=
by
  sorry

end local_minimum_point_l1536_153628


namespace power_function_propositions_l1536_153622

theorem power_function_propositions : (∀ n : ℤ, n > 0 → ∀ x : ℝ, x > 0 → (x^n) < x) ∧
  (∀ n : ℤ, n < 0 → ∀ x : ℝ, x > 0 → (x^n) > x) :=
by
  sorry

end power_function_propositions_l1536_153622


namespace hyperbola_focal_product_l1536_153634

-- Define the hyperbola with given equation and point P conditions
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

-- Define properties of vectors related to foci
def perpendicular (v1 v2 : ℝ × ℝ) := (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Define the point-focus distance product condition
noncomputable def focalProduct (P F1 F2 : ℝ × ℝ) := (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))

theorem hyperbola_focal_product :
  ∀ (a b : ℝ) (F1 F2 P : ℝ × ℝ),
  Hyperbola a b P ∧ perpendicular (P - F1) (P - F2) ∧
  -- Assuming a parabola property ties F1 with a specific value
  ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 4 * (Real.sqrt  ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))) →
  focalProduct P F1 F2 = 14 := by
  sorry

end hyperbola_focal_product_l1536_153634


namespace value_of_expression_l1536_153629

noncomputable def x := (2 : ℚ) / 3
noncomputable def y := (5 : ℚ) / 2

theorem value_of_expression : (1 / 3) * x^8 * y^9 = (5^9 / (2 * 3^9)) := by
  sorry

end value_of_expression_l1536_153629


namespace remainder_div_l1536_153670

theorem remainder_div (N : ℕ) (n : ℕ) : 
  (N % 2^n) = (N % 10^n % 2^n) ∧ (N % 5^n) = (N % 10^n % 5^n) := by
  sorry

end remainder_div_l1536_153670


namespace part1_part2_part3_l1536_153639

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

theorem part1 (x : ℝ) (hx : 0 < x) : f 0 x < x := by sorry

theorem part2 (a x : ℝ) :
  (0 ≤ a ∧ a ≤ 8/9 → 0 = 0) ∧
  (a > 8/9 → 2 = 2) ∧
  (a < 0 → 1 = 1) := by sorry

theorem part3 (a : ℝ) (h : ∀ x > 0, f a x ≥ 0) : 0 ≤ a ∧ a ≤ 1 := by sorry

end part1_part2_part3_l1536_153639


namespace find_positive_integer_n_l1536_153645

theorem find_positive_integer_n (S : ℕ → ℚ) (hS : ∀ n, S n = n / (n + 1))
  (h : ∃ n : ℕ, S n * S (n + 1) = 3 / 4) : 
  ∃ n : ℕ, n = 6 := 
by {
  sorry
}

end find_positive_integer_n_l1536_153645


namespace find_correct_value_l1536_153642

theorem find_correct_value (incorrect_value : ℝ) (subtracted_value : ℝ) (added_value : ℝ) (h_sub : subtracted_value = -added_value)
(h_incorrect : incorrect_value = 8.8) (h_subtracted : subtracted_value = -4.3) (h_added : added_value = 4.3) : incorrect_value + added_value + added_value = 17.4 :=
by
  sorry

end find_correct_value_l1536_153642


namespace female_employees_l1536_153677

theorem female_employees (E M F : ℕ) (h1 : 300 = 300) (h2 : (2/5 : ℚ) * E = (2/5 : ℚ) * M + 300) (h3 : E = M + F) : F = 750 := 
by
  sorry

end female_employees_l1536_153677


namespace six_digit_numbers_l1536_153608

def isNonPerfectPower (n : ℕ) : Prop :=
  ∀ m k : ℕ, m ≥ 2 → k ≥ 2 → m^k ≠ n

theorem six_digit_numbers : ∃ x : ℕ, 
  100000 ≤ x ∧ x < 1000000 ∧ 
  (∃ a b c: ℕ, x = (a^3 * b)^2 ∧ isNonPerfectPower a ∧ isNonPerfectPower b ∧ isNonPerfectPower c ∧ 
    (∃ k : ℤ, k > 1 ∧ 
      (x: ℤ) / (k^3 : ℤ) < 1 ∧ 
      ∃ num denom: ℕ, num < denom ∧ 
      num = n^3 ∧ denom = d^2 ∧ 
      isNonPerfectPower n ∧ isNonPerfectPower d)) := 
sorry

end six_digit_numbers_l1536_153608


namespace sum_of_roots_gt_two_l1536_153601

noncomputable def f : ℝ → ℝ := λ x => Real.log x - x + 1

theorem sum_of_roots_gt_two (m : ℝ) (x1 x2 : ℝ) (hx1 : f x1 = m) (hx2 : f x2 = m) (hne : x1 ≠ x2) : x1 + x2 > 2 := by
  sorry

end sum_of_roots_gt_two_l1536_153601


namespace am_gm_problem_l1536_153607

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end am_gm_problem_l1536_153607


namespace barbara_removed_114_sheets_l1536_153631

/-- Given conditions: -/
def bundles (n : ℕ) := 2 * n
def bunches (n : ℕ) := 4 * n
def heaps (n : ℕ) := 20 * n

/-- Barbara removed certain amounts of paper from the chest of drawers. -/
def total_sheets_removed := bundles 3 + bunches 2 + heaps 5

theorem barbara_removed_114_sheets : total_sheets_removed = 114 := by
  -- proof will be inserted here
  sorry

end barbara_removed_114_sheets_l1536_153631


namespace lion_room_is_3_l1536_153614

/-!
  A lion is hidden in one of three rooms. A note on the door of room 1 reads "The lion is here".
  A note on the door of room 2 reads "The lion is not here". A note on the door of room 3 reads "2+3=2×3".
  Only one of these notes is true. Prove that the lion is in room 3.
-/

def note1 (lion_room : ℕ) : Prop := lion_room = 1
def note2 (lion_room : ℕ) : Prop := lion_room ≠ 2
def note3 (lion_room : ℕ) : Prop := 2 + 3 = 2 * 3
def lion_is_in_room3 : Prop := ∀ lion_room, (note1 lion_room ∨ note2 lion_room ∨ note3 lion_room) ∧
  (note1 lion_room → note2 lion_room = false) ∧ (note1 lion_room → note3 lion_room = false) ∧
  (note2 lion_room → note1 lion_room = false) ∧ (note2 lion_room → note3 lion_room = false) ∧
  (note3 lion_room → note1 lion_room = false) ∧ (note3 lion_room → note2 lion_room = false) → lion_room = 3

theorem lion_room_is_3 : lion_is_in_room3 := 
  by
  sorry

end lion_room_is_3_l1536_153614


namespace robin_camera_pictures_l1536_153625

-- Given conditions
def pictures_from_phone : Nat := 35
def num_albums : Nat := 5
def pics_per_album : Nat := 8

-- Calculate total pictures and the number of pictures from the camera
theorem robin_camera_pictures : num_albums * pics_per_album - pictures_from_phone = 5 := by
  sorry

end robin_camera_pictures_l1536_153625


namespace difference_of_digits_l1536_153600

theorem difference_of_digits (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h_diff : (10 * x + y) - (10 * y + x) = 54) : x - y = 6 :=
sorry

end difference_of_digits_l1536_153600


namespace range_of_f_4_l1536_153695

theorem range_of_f_4 {a b c d : ℝ} 
  (h1 : 1 ≤ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ∧ a*(-1)^3 + b*(-1)^2 + c*(-1) + d ≤ 2) 
  (h2 : 1 ≤ a*1^3 + b*1^2 + c*1 + d ∧ a*1^3 + b*1^2 + c*1 + d ≤ 3) 
  (h3 : 2 ≤ a*2^3 + b*2^2 + c*2 + d ∧ a*2^3 + b*2^2 + c*2 + d ≤ 4) 
  (h4 : -1 ≤ a*3^3 + b*3^2 + c*3 + d ∧ a*3^3 + b*3^2 + c*3 + d ≤ 1) :
  -21.75 ≤ a*4^3 + b*4^2 + c*4 + d ∧ a*4^3 + b*4^2 + c*4 + d ≤ 1 :=
sorry

end range_of_f_4_l1536_153695


namespace young_people_in_sample_l1536_153682

-- Define the conditions
def total_population (elderly middle_aged young : ℕ) : ℕ :=
  elderly + middle_aged + young

def sample_proportion (sample_size total_pop : ℚ) : ℚ :=
  sample_size / total_pop

def stratified_sample (group_size proportion : ℚ) : ℚ :=
  group_size * proportion

-- Main statement to prove
theorem young_people_in_sample (elderly middle_aged young : ℕ) (sample_size : ℚ) :
  total_population elderly middle_aged young = 108 →
  sample_size = 36 →
  stratified_sample (young : ℚ) (sample_proportion sample_size 108) = 17 :=
by
  intros h_total h_sample_size
  sorry -- proof omitted

end young_people_in_sample_l1536_153682


namespace product_gcd_lcm_is_correct_l1536_153668

-- Define the numbers
def a := 15
def b := 75

-- Definitions related to GCD and LCM
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b
def product_gcd_lcm := gcd_ab * lcm_ab

-- Theorem stating the product of GCD and LCM of a and b is 1125
theorem product_gcd_lcm_is_correct : product_gcd_lcm = 1125 := by
  sorry

end product_gcd_lcm_is_correct_l1536_153668


namespace factory_earnings_l1536_153660

-- Definition of constants and functions based on the conditions:
def material_A_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def material_B_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def convert_B_to_C (material_B : ℕ) : ℕ := material_B / 2
def earnings (amount : ℕ) (price_per_unit : ℕ) : ℕ := amount * price_per_unit

-- Given conditions for the problem:
def hours_machine_1_and_2 : ℕ := 23
def hours_machine_3 : ℕ := 23
def hours_machine_4 : ℕ := 12
def rate_A_machine_1_and_2 : ℕ := 2
def rate_B_machine_1_and_2 : ℕ := 1
def rate_A_machine_3_and_4 : ℕ := 3
def rate_B_machine_3_and_4 : ℕ := 2
def price_A : ℕ := 50
def price_C : ℕ := 100

-- Calculations based on problem conditions:
noncomputable def total_A : ℕ := 
  2 * material_A_production hours_machine_1_and_2 rate_A_machine_1_and_2 + 
  material_A_production hours_machine_3 rate_A_machine_3_and_4 + 
  material_A_production hours_machine_4 rate_A_machine_3_and_4

noncomputable def total_B : ℕ := 
  2 * material_B_production hours_machine_1_and_2 rate_B_machine_1_and_2 + 
  material_B_production hours_machine_3 rate_B_machine_3_and_4 + 
  material_B_production hours_machine_4 rate_B_machine_3_and_4

noncomputable def total_C : ℕ := convert_B_to_C total_B

noncomputable def total_earnings : ℕ :=
  earnings total_A price_A + earnings total_C price_C

-- The theorem to prove the total earnings:
theorem factory_earnings : total_earnings = 15650 :=
by
  sorry

end factory_earnings_l1536_153660


namespace find_natural_triples_l1536_153618

theorem find_natural_triples (x y z : ℕ) : 
  (x+1) * (y+1) * (z+1) = 3 * x * y * z ↔ 
  (x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 2) ∨ (x, y, z) = (3, 2, 2) ∨
  (x, y, z) = (5, 1, 4) ∨ (x, y, z) = (5, 4, 1) ∨ (x, y, z) = (4, 1, 5) ∨ (x, y, z) = (4, 5, 1) ∨ 
  (x, y, z) = (1, 4, 5) ∨ (x, y, z) = (1, 5, 4) ∨ (x, y, z) = (8, 1, 3) ∨ (x, y, z) = (8, 3, 1) ∨
  (x, y, z) = (3, 1, 8) ∨ (x, y, z) = (3, 8, 1) ∨ (x, y, z) = (1, 3, 8) ∨ (x, y, z) = (1, 8, 3) :=
by {
  sorry
}

end find_natural_triples_l1536_153618


namespace largest_four_digit_number_l1536_153661

def is_four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

def sum_of_digits (N : ℕ) : ℕ :=
  let a := N / 1000
  let b := (N % 1000) / 100
  let c := (N % 100) / 10
  let d := N % 10
  a + b + c + d

def is_divisible (N S : ℕ) : Prop := N % S = 0

theorem largest_four_digit_number :
  ∃ N : ℕ, is_four_digit_number N ∧ is_divisible N (sum_of_digits N) ∧
  (∀ M : ℕ, is_four_digit_number M ∧ is_divisible M (sum_of_digits M) → N ≥ M) ∧ N = 9990 :=
by
  sorry

end largest_four_digit_number_l1536_153661


namespace john_spent_expected_amount_l1536_153609

-- Define the original price of each pin
def original_price : ℝ := 20

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the number of pins
def number_of_pins : ℝ := 10

-- Define the sales tax rate
def tax_rate : ℝ := 0.08

-- Calculate the discount on each pin
def discount_per_pin : ℝ := discount_rate * original_price

-- Calculate the discounted price per pin
def discounted_price_per_pin : ℝ := original_price - discount_per_pin

-- Calculate the total discounted price for all pins
def total_discounted_price : ℝ := discounted_price_per_pin * number_of_pins

-- Calculate the sales tax on the total discounted price
def sales_tax : ℝ := tax_rate * total_discounted_price

-- Calculate the total amount spent including sales tax
def total_amount_spent : ℝ := total_discounted_price + sales_tax

-- The theorem that John spent $183.60 on pins including the sales tax
theorem john_spent_expected_amount : total_amount_spent = 183.60 :=
by
  sorry

end john_spent_expected_amount_l1536_153609


namespace customers_left_l1536_153690

theorem customers_left (original_customers remaining_tables people_per_table customers_left : ℕ)
  (h1 : original_customers = 44)
  (h2 : remaining_tables = 4)
  (h3 : people_per_table = 8)
  (h4 : original_customers - remaining_tables * people_per_table = customers_left) :
  customers_left = 12 :=
by
  sorry

end customers_left_l1536_153690


namespace problem_statement_l1536_153665

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sqrt 3 * cos (ω * x + π / 6)

theorem problem_statement (ω : ℝ) (hx : ω = 2 ∨ ω = -2) :
  f ω (π / 3) = -3 ∨ f ω (π / 3) = 0 := by
  unfold f
  cases hx with
  | inl w_eq => sorry
  | inr w_eq => sorry

end problem_statement_l1536_153665


namespace sugar_concentration_after_adding_water_l1536_153635

def initial_mass_of_sugar_water : ℝ := 90
def initial_sugar_concentration : ℝ := 0.10
def final_sugar_concentration : ℝ := 0.08
def mass_of_water_added : ℝ := 22.5

theorem sugar_concentration_after_adding_water 
  (m_sugar_water : ℝ := initial_mass_of_sugar_water)
  (c_initial : ℝ := initial_sugar_concentration)
  (c_final : ℝ := final_sugar_concentration)
  (m_water_added : ℝ := mass_of_water_added) :
  (m_sugar_water * c_initial = (m_sugar_water + m_water_added) * c_final) := 
sorry

end sugar_concentration_after_adding_water_l1536_153635


namespace max_points_of_intersection_l1536_153620

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end max_points_of_intersection_l1536_153620


namespace percentage_of_boys_answered_neither_l1536_153678

theorem percentage_of_boys_answered_neither (P_A P_B P_A_and_B : ℝ) (hP_A : P_A = 0.75) (hP_B : P_B = 0.55) (hP_A_and_B : P_A_and_B = 0.50) :
  1 - (P_A + P_B - P_A_and_B) = 0.20 :=
by
  sorry

end percentage_of_boys_answered_neither_l1536_153678


namespace sum_arithmetic_sequence_l1536_153606

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_arithmetic_sequence {a : ℕ → ℝ} 
  (h_arith : arithmetic_seq a)
  (h1 : a 2^2 + a 7^2 + 2 * a 2 * a 7 = 9)
  (h2 : ∀ n, a n < 0) : 
  S₁₀ = -15 :=
by
  sorry

end sum_arithmetic_sequence_l1536_153606


namespace oxygen_atoms_l1536_153664

theorem oxygen_atoms (x : ℤ) (h : 27 + 16 * x + 3 = 78) : x = 3 := 
by 
  sorry

end oxygen_atoms_l1536_153664


namespace julian_comic_book_l1536_153694

theorem julian_comic_book : 
  ∀ (total_frames frames_per_page : ℕ),
    total_frames = 143 →
    frames_per_page = 11 →
    total_frames / frames_per_page = 13 ∧ total_frames % frames_per_page = 0 :=
by
  intros total_frames frames_per_page
  intros h_total_frames h_frames_per_page
  sorry

end julian_comic_book_l1536_153694


namespace compute_expression_l1536_153611

theorem compute_expression :
  (143 + 29) * 2 + 25 + 13 = 382 :=
by 
  sorry

end compute_expression_l1536_153611


namespace ineq_triples_distinct_integers_l1536_153696

theorem ineq_triples_distinct_integers 
  (x y z : ℤ) (h₁ : x ≠ y) (h₂ : y ≠ z) (h₃ : z ≠ x) : 
  ( ( (x - y)^7 + (y - z)^7 + (z - x)^7 - (x - y) * (y - z) * (z - x) * ((x - y)^4 + (y - z)^4 + (z - x)^4) )
  / ( (x - y)^5 + (y - z)^5 + (z - x)^5 ) ) ≥ 3 :=
sorry

end ineq_triples_distinct_integers_l1536_153696


namespace temperature_difference_correct_l1536_153617

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end temperature_difference_correct_l1536_153617


namespace bisecting_line_eq_l1536_153657

theorem bisecting_line_eq : ∃ (a : ℝ), (∀ x y : ℝ, (y = a * x) ↔ y = -1 / 6 * x) ∧ 
  (∀ p : ℝ × ℝ, (3 * p.1 - 5 * p.2  = 6 → p.2 = a * p.1) ∧ 
                  (4 * p.1 + p.2 + 6 = 0 → p.2 = a * p.1)) :=
by
  use -1 / 6
  sorry

end bisecting_line_eq_l1536_153657


namespace absolute_sum_of_coefficients_l1536_153662

theorem absolute_sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (2 - x)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 2^6 →
  a_0 > 0 ∧ a_2 > 0 ∧ a_4 > 0 ∧ a_6 > 0 ∧
  a_1 < 0 ∧ a_3 < 0 ∧ a_5 < 0 → 
  |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 665 :=
by sorry

end absolute_sum_of_coefficients_l1536_153662


namespace percent_decrease_in_cost_l1536_153650

theorem percent_decrease_in_cost (cost_1990 cost_2010 : ℕ) (h1 : cost_1990 = 35) (h2 : cost_2010 = 5) : 
  ((cost_1990 - cost_2010) * 100 / cost_1990 : ℚ) = 86 := 
by
  sorry

end percent_decrease_in_cost_l1536_153650


namespace remaining_money_l1536_153681

def initial_amount : ℕ := 10
def spent_on_toy_truck : ℕ := 3
def spent_on_pencil_case : ℕ := 2

theorem remaining_money (initial_amount spent_on_toy_truck spent_on_pencil_case : ℕ) : 
  initial_amount - (spent_on_toy_truck + spent_on_pencil_case) = 5 :=
by
  sorry

end remaining_money_l1536_153681


namespace find_a_l1536_153666

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_a (h1 : a ≠ 0) (h2 : f a b c (-1) = 0)
    (h3 : ∀ x : ℝ, x ≤ f a b c x ∧ f a b c x ≤ (1/2) * (x^2 + 1)) :
  a = 1/2 :=
by
  sorry

end find_a_l1536_153666


namespace minimum_value_function_inequality_ln_l1536_153605

noncomputable def f (x : ℝ) := x * Real.log x

theorem minimum_value_function (t : ℝ) (ht : 0 < t) :
  ∃ (xmin : ℝ), xmin = if (0 < t ∧ t < 1 / Real.exp 1) then -1 / Real.exp 1 else t * Real.log t :=
sorry

theorem inequality_ln (x : ℝ) (hx : 0 < x) : 
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end minimum_value_function_inequality_ln_l1536_153605


namespace find_m_l1536_153638

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d 

noncomputable def sum_first_n_terms (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

theorem find_m {a S : ℕ → ℤ} (d : ℤ) (m : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : a 1 = 1)
  (h4 : S 3 = a 5)
  (h5 : a m = 2011) :
  m = 1006 :=
sorry

end find_m_l1536_153638


namespace fraction_of_sophomores_attending_fair_l1536_153603

theorem fraction_of_sophomores_attending_fair
  (s j n : ℕ)
  (h1 : s = j)
  (h2 : j = n)
  (soph_attend : ℚ)
  (junior_attend : ℚ)
  (senior_attend : ℚ)
  (fraction_s : soph_attend = 4/5 * s)
  (fraction_j : junior_attend = 3/4 * j)
  (fraction_n : senior_attend = 1/3 * n) :
  soph_attend / (soph_attend + junior_attend + senior_attend) = 240 / 565 :=
by
  sorry

end fraction_of_sophomores_attending_fair_l1536_153603


namespace scientific_notation_correct_l1536_153626

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end scientific_notation_correct_l1536_153626


namespace value_of_a_l1536_153640

theorem value_of_a (a : ℝ) (h : (a - 3) * x ^ |a - 2| + 4 = 0) : a = 1 :=
by
  sorry

end value_of_a_l1536_153640


namespace binary_to_octal_equivalence_l1536_153619

theorem binary_to_octal_equivalence : (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) 
                                    = (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end binary_to_octal_equivalence_l1536_153619


namespace simplify_fraction_l1536_153641

theorem simplify_fraction :
  (1 / (1 + Real.sqrt 3) * 1 / (1 - Real.sqrt 5)) = 
  (1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15)) :=
by
  sorry

end simplify_fraction_l1536_153641


namespace class_average_l1536_153630

theorem class_average (x : ℝ) :
  (0.25 * 80 + 0.5 * x + 0.25 * 90 = 75) → x = 65 := by
  sorry

end class_average_l1536_153630


namespace recommended_cups_l1536_153683

theorem recommended_cups (current_cups : ℕ) (R : ℕ) : 
  current_cups = 20 →
  R = current_cups + (6 / 10) * current_cups →
  R = 32 :=
by
  intros h1 h2
  sorry

end recommended_cups_l1536_153683


namespace equilateral_triangle_on_parallel_lines_l1536_153602

theorem equilateral_triangle_on_parallel_lines 
  (l1 l2 l3 : ℝ → Prop)
  (h_parallel_12 : ∀ x y, l1 x → l2 y → ∀ z, l1 z → l2 z)
  (h_parallel_23 : ∀ x y, l2 x → l3 y → ∀ z, l2 z → l3 z) 
  (h_parallel_13 : ∀ x y, l1 x → l3 y → ∀ z, l1 z → l3 z) 
  (A : ℝ) (hA : l1 A)
  (B : ℝ) (hB : l2 B)
  (C : ℝ) (hC : l3 C):
  ∃ A B C : ℝ, l1 A ∧ l2 B ∧ l3 C ∧ (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end equilateral_triangle_on_parallel_lines_l1536_153602
