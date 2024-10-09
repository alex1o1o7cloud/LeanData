import Mathlib

namespace addition_correctness_l2085_208515

theorem addition_correctness : 1.25 + 47.863 = 49.113 :=
by 
  sorry

end addition_correctness_l2085_208515


namespace number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l2085_208553

theorem number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100 :
  ∃! (n : ℕ), n = 3 ∧ ∀ (x y : ℕ), x > 0 → y > 0 → x^2 - y^2 = 100 ↔ (x, y) = (26, 24) ∨ (x, y) = (15, 10) ∨ (x, y) = (15, 5) :=
by
  sorry

end number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l2085_208553


namespace weight_of_each_bar_l2085_208558

theorem weight_of_each_bar 
  (num_bars : ℕ) 
  (cost_per_pound : ℝ) 
  (total_cost : ℝ) 
  (total_weight : ℝ) 
  (weight_per_bar : ℝ)
  (h1 : num_bars = 20)
  (h2 : cost_per_pound = 0.5)
  (h3 : total_cost = 15)
  (h4 : total_weight = total_cost / cost_per_pound)
  (h5 : weight_per_bar = total_weight / num_bars)
  : weight_per_bar = 1.5 := 
by
  sorry

end weight_of_each_bar_l2085_208558


namespace find_f_neg_2_l2085_208525

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x: ℝ, f (-x) = -f x

-- Problem statement
theorem find_f_neg_2 (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_fx_pos : ∀ x : ℝ, x > 0 → f x = 2 * x ^ 2 - 7) : 
  f (-2) = -1 :=
by
  sorry

end find_f_neg_2_l2085_208525


namespace probability_of_winning_l2085_208517

open Nat

theorem probability_of_winning (h : True) : 
  let num_cards := 3
  let num_books := 5
  (1 - (Nat.choose num_cards 2 * 2^num_books - num_cards) / num_cards^num_books) = 50 / 81 := sorry

end probability_of_winning_l2085_208517


namespace no_equilateral_integer_coords_l2085_208551

theorem no_equilateral_integer_coords (x1 y1 x2 y2 x3 y3 : ℤ) : 
  ¬ ((x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
     (x1 ≠ x3 ∨ y1 ≠ y3) ∧
     (x2 ≠ x3 ∨ y2 ≠ y3) ∧ 
     ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x1) ^ 2 + (y3 - y1) ^ 2 ∧ 
      (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (x3 - x2) ^ 2 + (y3 - y2) ^ 2)) :=
by
  sorry

end no_equilateral_integer_coords_l2085_208551


namespace coprime_divides_product_l2085_208531

theorem coprime_divides_product {a b n : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a ∣ n) (h3 : b ∣ n) : ab ∣ n :=
by
  sorry

end coprime_divides_product_l2085_208531


namespace red_marbles_count_l2085_208599

noncomputable def total_marbles (R : ℕ) : ℕ := R + 16

noncomputable def P_blue (R : ℕ) : ℚ := 10 / (total_marbles R)

noncomputable def P_neither_blue (R : ℕ) : ℚ := (1 - P_blue R) * (1 - P_blue R)

noncomputable def P_either_blue (R : ℕ) : ℚ := 1 - P_neither_blue R

theorem red_marbles_count
  (R : ℕ) 
  (h1 : P_either_blue R = 0.75) :
  R = 4 :=
by
  sorry

end red_marbles_count_l2085_208599


namespace max_volume_cuboid_l2085_208505

theorem max_volume_cuboid (x y z : ℕ) (h : 2 * (x * y + x * z + y * z) = 150) : x * y * z ≤ 125 :=
sorry

end max_volume_cuboid_l2085_208505


namespace coefficient_x5_in_product_l2085_208512

noncomputable def P : Polynomial ℤ := 
  Polynomial.C 1 * Polynomial.X ^ 6 +
  Polynomial.C (-2) * Polynomial.X ^ 5 +
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C (-6) * Polynomial.X +
  Polynomial.C 7

noncomputable def Q : Polynomial ℤ := 
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C 6 * Polynomial.X +
  Polynomial.C (-8)

theorem coefficient_x5_in_product (p q : Polynomial ℤ) :
  (p * q).coeff 5 = 2 :=
by
  have P := 
    Polynomial.C 1 * Polynomial.X ^ 6 +
    Polynomial.C (-2) * Polynomial.X ^ 5 +
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C (-6) * Polynomial.X +
    Polynomial.C 7
  have Q := 
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C 6 * Polynomial.X +
    Polynomial.C (-8)

  sorry

end coefficient_x5_in_product_l2085_208512


namespace percent_non_bikers_play_basketball_l2085_208535

noncomputable def total_children (N : ℕ) : ℕ := N
def basketball_players (N : ℕ) : ℕ := 7 * N / 10
def bikers (N : ℕ) : ℕ := 4 * N / 10
def basketball_bikers (N : ℕ) : ℕ := 3 * basketball_players N / 10
def basketball_non_bikers (N : ℕ) : ℕ := basketball_players N - basketball_bikers N
def non_bikers (N : ℕ) : ℕ := N - bikers N

theorem percent_non_bikers_play_basketball (N : ℕ) :
  (basketball_non_bikers N * 100 / non_bikers N) = 82 :=
by sorry

end percent_non_bikers_play_basketball_l2085_208535


namespace wire_cut_example_l2085_208533

theorem wire_cut_example (total_length piece_ratio : ℝ) (h1 : total_length = 28) (h2 : piece_ratio = 2.00001 / 5) :
  ∃ (shorter_piece : ℝ), shorter_piece + piece_ratio * shorter_piece = total_length ∧ shorter_piece = 20 :=
by
  sorry

end wire_cut_example_l2085_208533


namespace parallel_lines_iff_determinant_zero_l2085_208541

theorem parallel_lines_iff_determinant_zero (a1 b1 c1 a2 b2 c2 : ℝ) :
  (a1 * b2 - a2 * b1 = 0) ↔ ((a1 * c2 - a2 * c1 = 0) → (b1 * c2 - b2 * c1 = 0)) := 
sorry

end parallel_lines_iff_determinant_zero_l2085_208541


namespace smaller_number_is_5_l2085_208538

theorem smaller_number_is_5 (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 := by
  sorry

end smaller_number_is_5_l2085_208538


namespace line_through_point_bisected_by_hyperbola_l2085_208574

theorem line_through_point_bisected_by_hyperbola :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 3 + b * (-1) + c = 0) ∧
  (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) → (a * x + b * y + c = 0)) ↔ (a = 3 ∧ b = 4 ∧ c = -5) :=
by
  sorry

end line_through_point_bisected_by_hyperbola_l2085_208574


namespace linear_inequality_solution_set_l2085_208592

variable (x : ℝ)

theorem linear_inequality_solution_set :
  ∀ x : ℝ, (2 * x - 4 > 0) → (x > 2) := 
by
  sorry

end linear_inequality_solution_set_l2085_208592


namespace green_bows_count_l2085_208580

noncomputable def total_bows : ℕ := 36 * 4

def fraction_green : ℚ := 1/6

theorem green_bows_count (red blue green total yellow : ℕ) (h_red : red = total / 4)
  (h_blue : blue = total / 3) (h_green : green = total / 6)
  (h_yellow : yellow = total - red - blue - green)
  (h_yellow_count : yellow = 36) : green = 24 := by
  sorry

end green_bows_count_l2085_208580


namespace correct_calculation_l2085_208566

theorem correct_calculation (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 :=
  sorry

end correct_calculation_l2085_208566


namespace sub_two_three_l2085_208503

theorem sub_two_three : 2 - 3 = -1 := 
by 
  sorry

end sub_two_three_l2085_208503


namespace find_set_T_l2085_208588

namespace MathProof 

theorem find_set_T (S : Finset ℕ) (hS : ∀ x ∈ S, x > 0) :
  ∃ T : Finset ℕ, S ⊆ T ∧ ∀ x ∈ T, x ∣ (T.sum id) :=
by
  sorry

end MathProof 

end find_set_T_l2085_208588


namespace math_problem_l2085_208577

variable {a b c d e f : ℕ}
variable (h1 : f < a)
variable (h2 : (a * b * d + 1) % c = 0)
variable (h3 : (a * c * e + 1) % b = 0)
variable (h4 : (b * c * f + 1) % a = 0)

theorem math_problem
  (h5 : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by {
  skip -- Adding "by" ... "sorry" to make the statement complete since no proof is required.
  sorry
}

end math_problem_l2085_208577


namespace unique_reconstruction_l2085_208567

theorem unique_reconstruction (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (a b c d : ℝ) (Ha : x + y = a) (Hb : x - y = b) (Hc : x * y = c) (Hd : x / y = d) :
  ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = a ∧ x' - y' = b ∧ x' * y' = c ∧ x' / y' = d := 
sorry

end unique_reconstruction_l2085_208567


namespace max_true_statements_maximum_true_conditions_l2085_208570

theorem max_true_statements (x y : ℝ) (h1 : (1/x > 1/y)) (h2 : (x^2 < y^2)) (h3 : (x > y)) (h4 : (x > 0)) (h5 : (y > 0)) :
  false :=
  sorry

theorem maximum_true_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ¬ ((1/x > 1/y) ∧ (x^2 < y^2)) :=
  sorry

#check max_true_statements
#check maximum_true_conditions

end max_true_statements_maximum_true_conditions_l2085_208570


namespace solve_positive_integer_x_l2085_208506

theorem solve_positive_integer_x : ∃ (x : ℕ), 4 * x^2 - 16 * x - 60 = 0 ∧ x = 6 :=
by
  sorry

end solve_positive_integer_x_l2085_208506


namespace photo_arrangement_l2085_208594

noncomputable def valid_arrangements (teacher boys girls : ℕ) : ℕ :=
  if girls = 2 ∧ teacher = 1 ∧ boys = 2 then 24 else 0

theorem photo_arrangement :
  valid_arrangements 1 2 2 = 24 :=
by {
  -- The proof goes here.
  sorry
}

end photo_arrangement_l2085_208594


namespace number_of_groups_l2085_208556

theorem number_of_groups (max min c : ℕ) (h_max : max = 140) (h_min : min = 50) (h_c : c = 10) : 
  (max - min) / c + 1 = 10 := 
by
  sorry

end number_of_groups_l2085_208556


namespace proof_problem_l2085_208552

-- Define the operation table as a function in Lean 4
def op (a b : ℕ) : ℕ :=
  if a = 1 then
    if b = 1 then 2 else if b = 2 then 1 else if b = 3 then 4 else 3
  else if a = 2 then
    if b = 1 then 1 else if b = 2 then 3 else if b = 3 then 2 else 4
  else if a = 3 then
    if b = 1 then 4 else if b = 2 then 2 else if b = 3 then 1 else 3
  else
    if b = 1 then 3 else if b = 2 then 4 else if b = 3 then 3 else 2

-- State the theorem to prove
theorem proof_problem : op (op 3 1) (op 4 2) = 2 :=
by
  sorry

end proof_problem_l2085_208552


namespace resulting_solid_faces_l2085_208596

-- Define a cube structure with a given number of faces
structure Cube where
  faces : Nat

-- Define the problem conditions and prove the total faces of the resulting solid
def original_cube := Cube.mk 6

def new_faces_per_cube := 5

def total_new_faces := original_cube.faces * new_faces_per_cube

def total_faces_of_resulting_solid := total_new_faces + original_cube.faces

theorem resulting_solid_faces : total_faces_of_resulting_solid = 36 := by
  sorry

end resulting_solid_faces_l2085_208596


namespace number_difference_l2085_208504

theorem number_difference (x y : ℕ) (h₁ : x + y = 41402) (h₂ : ∃ k : ℕ, x = 100 * k) (h₃ : y = x / 100) : x - y = 40590 :=
sorry

end number_difference_l2085_208504


namespace probability_not_snowing_l2085_208514

  -- Define the probability that it will snow tomorrow
  def P_snowing : ℚ := 2 / 5

  -- Define the probability that it will not snow tomorrow
  def P_not_snowing : ℚ := 1 - P_snowing

  -- Theorem stating the required proof
  theorem probability_not_snowing : P_not_snowing = 3 / 5 :=
  by 
    -- Proof would go here
    sorry
  
end probability_not_snowing_l2085_208514


namespace cost_of_car_l2085_208516

theorem cost_of_car (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment = 3000 →
  num_installments = 6 →
  installment_amount = 2500 →
  initial_payment + num_installments * installment_amount = 18000 :=
by
  intros h_initial h_num h_installment
  sorry

end cost_of_car_l2085_208516


namespace jillian_max_apartment_size_l2085_208549

theorem jillian_max_apartment_size :
  ∀ s : ℝ, (1.10 * s = 880) → s = 800 :=
by
  intros s h
  sorry

end jillian_max_apartment_size_l2085_208549


namespace solution_set_of_inequality_l2085_208529

theorem solution_set_of_inequality (x : ℝ) : (x * |x - 1| > 0) ↔ (0 < x ∧ x < 1 ∨ 1 < x) := 
by
  sorry

end solution_set_of_inequality_l2085_208529


namespace expectation_fish_l2085_208507

noncomputable def fish_distribution : ℕ → ℚ → ℚ → ℚ → ℚ :=
  fun N a b c => (a / b) * (1 - (c / (a + b + c) ^ N))

def x_distribution : ℚ := 0.18
def y_distribution : ℚ := 0.02
def other_distribution : ℚ := 0.80
def total_fish : ℕ := 10

theorem expectation_fish :
  fish_distribution total_fish x_distribution y_distribution other_distribution = 1.6461 :=
  by
    sorry

end expectation_fish_l2085_208507


namespace range_of_a_l2085_208565

variables (a b c : ℝ)

theorem range_of_a (h₁ : a^2 - b * c - 8 * a + 7 = 0)
                   (h₂ : b^2 + c^2 + b * c - 6 * a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l2085_208565


namespace greatest_divisor_l2085_208598

theorem greatest_divisor (d : ℕ) :
  (690 % d = 10) ∧ (875 % d = 25) ∧ ∀ e : ℕ, (690 % e = 10) ∧ (875 % e = 25) → (e ≤ d) :=
  sorry

end greatest_divisor_l2085_208598


namespace general_term_formula_l2085_208532
-- Import the Mathlib library 

-- Define the conditions as given in the problem
/-- 
Define the sequence that represents the numerators. 
This is an arithmetic sequence of odd numbers starting from 1.
-/
def numerator (n : ℕ) : ℕ := 2 * n + 1

/-- 
Define the sequence that represents the denominators. 
This is a geometric sequence with the first term being 2 and common ratio being 2.
-/
def denominator (n : ℕ) : ℕ := 2^(n+1)

-- State the main theorem that we need to prove
theorem general_term_formula (n : ℕ) : (numerator n) / (denominator n) = (2 * n + 1) / 2^(n+1) :=
sorry

end general_term_formula_l2085_208532


namespace hostel_provisions_l2085_208523

theorem hostel_provisions (x : ℕ) :
  (250 * x = 200 * 60) → x = 48 :=
by
  sorry

end hostel_provisions_l2085_208523


namespace negation_of_proposition_l2085_208550

open Classical

theorem negation_of_proposition : (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) :=
by
  sorry

end negation_of_proposition_l2085_208550


namespace Bryce_received_raisins_l2085_208537

theorem Bryce_received_raisins :
  ∃ x : ℕ, (∀ y : ℕ, x = y + 6) ∧ (∀ z : ℕ, z = x / 2) → x = 12 :=
by
  sorry

end Bryce_received_raisins_l2085_208537


namespace minimum_surface_area_of_cube_l2085_208536

noncomputable def brick_length := 25
noncomputable def brick_width := 15
noncomputable def brick_height := 5
noncomputable def side_length := Nat.lcm brick_width brick_length
noncomputable def surface_area := 6 * side_length * side_length

theorem minimum_surface_area_of_cube : surface_area = 33750 := 
by
  sorry

end minimum_surface_area_of_cube_l2085_208536


namespace inequality_proof_l2085_208569

theorem inequality_proof (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
by 
  sorry

end inequality_proof_l2085_208569


namespace problem_solution_l2085_208562

theorem problem_solution :
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := 
  by
  sorry

end problem_solution_l2085_208562


namespace inequality_proof_l2085_208545

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (2 * (a^3 + b^3 + c^3)) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l2085_208545


namespace smallest_possible_sum_l2085_208568

theorem smallest_possible_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hneq : a ≠ b) 
  (heq : (1 / a : ℚ) + (1 / b) = 1 / 12) : a + b = 49 :=
sorry

end smallest_possible_sum_l2085_208568


namespace determine_values_of_abc_l2085_208589

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem determine_values_of_abc 
  (a b c : ℝ) 
  (h_f : ∀ x : ℝ, f a b c (f_inv a b c x) = x)
  (h_f_inv : ∀ x : ℝ, f_inv a b c (f a b c x) = x) : 
  a = -1 ∧ b = 1 ∧ c = 0 :=
by
  sorry

end determine_values_of_abc_l2085_208589


namespace problem1_problem2_l2085_208590

theorem problem1 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) ≥ 2 :=
sorry

theorem problem2 (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  abs ((a + b) / (a - b)) + abs ((b + c) / (b - c)) + abs ((c + a) / (c - a)) > 3 :=
sorry

end problem1_problem2_l2085_208590


namespace friends_area_is_greater_by_14_point_4_times_l2085_208519

theorem friends_area_is_greater_by_14_point_4_times :
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  friend_area / tommy_area = 14.4 :=
by
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  sorry

end friends_area_is_greater_by_14_point_4_times_l2085_208519


namespace ping_pong_matches_l2085_208561

noncomputable def f (n k : ℕ) : ℕ :=
  Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2))

theorem ping_pong_matches (n k : ℕ) (hn_pos : 0 < n) (hk_le : k ≤ 2 * n - 1) :
  f n k = Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2)) :=
by
  sorry

end ping_pong_matches_l2085_208561


namespace brick_width_correct_l2085_208527

theorem brick_width_correct
  (courtyard_length_m : ℕ) (courtyard_width_m : ℕ) (brick_length_cm : ℕ) (num_bricks : ℕ)
  (total_area_cm : ℕ) (brick_width_cm : ℕ) :
  courtyard_length_m = 25 →
  courtyard_width_m = 16 →
  brick_length_cm = 20 →
  num_bricks = 20000 →
  total_area_cm = courtyard_length_m * 100 * courtyard_width_m * 100 →
  total_area_cm = num_bricks * brick_length_cm * brick_width_cm →
  brick_width_cm = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end brick_width_correct_l2085_208527


namespace line_equation_through_point_l2085_208508

theorem line_equation_through_point 
  (x y : ℝ)
  (h1 : (5, 2) ∈ {p : ℝ × ℝ | p.2 = p.1 * (2 / 5)})
  (h2 : (5, 2) ∈ {p : ℝ × ℝ | p.1 / 6 + p.2 / 12 = 1}) 
  (h3 : (5,2) ∈ {p : ℝ × ℝ | 2 * p.1 = p.2 }) :
  (2 * x + y - 12 = 0 ∨ 
   2 * x - 5 * y = 0) := 
sorry

end line_equation_through_point_l2085_208508


namespace area_square_given_diagonal_l2085_208540

theorem area_square_given_diagonal (d : ℝ) (h : d = 16) : (∃ A : ℝ, A = 128) :=
by 
  sorry

end area_square_given_diagonal_l2085_208540


namespace time_to_odd_floor_l2085_208572

-- Define the number of even-numbered floors
def evenFloors : Nat := 5

-- Define the number of odd-numbered floors
def oddFloors : Nat := 5

-- Define the time to climb one even-numbered floor
def timeEvenFloor : Nat := 15

-- Define the total time to reach the 10th floor
def totalTime : Nat := 120

-- Define the desired time per odd-numbered floor
def timeOddFloor : Nat := 9

-- Formalize the proof statement
theorem time_to_odd_floor : 
  (oddFloors * timeOddFloor = totalTime - (evenFloors * timeEvenFloor)) :=
by
  sorry

end time_to_odd_floor_l2085_208572


namespace robert_coin_arrangement_l2085_208539

noncomputable def num_arrangements (gold : ℕ) (silver : ℕ) : ℕ :=
  if gold + silver = 8 ∧ gold = 5 ∧ silver = 3 then 504 else 0

theorem robert_coin_arrangement :
  num_arrangements 5 3 = 504 := 
sorry

end robert_coin_arrangement_l2085_208539


namespace find_S16_l2085_208548

-- Definitions
def geom_seq (a : ℕ → ℝ) : Prop := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def sum_of_geom_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = a 0 * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))

-- Problem conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom geom_seq_a : geom_seq a
axiom S4_eq : S 4 = 4
axiom S8_eq : S 8 = 12

-- Theorem
theorem find_S16 : S 16 = 60 :=
  sorry

end find_S16_l2085_208548


namespace proportion_fourth_number_l2085_208593

theorem proportion_fourth_number (x y : ℝ) (h_x : x = 0.6) (h_prop : 0.75 / x = 10 / y) : y = 8 :=
by
  sorry

end proportion_fourth_number_l2085_208593


namespace probability_two_girls_l2085_208595

theorem probability_two_girls (total_students girls boys : ℕ) (htotal : total_students = 6) (hg : girls = 4) (hb : boys = 2) :
  (Nat.choose girls 2 / Nat.choose total_students 2 : ℝ) = 2 / 5 := by
  sorry

end probability_two_girls_l2085_208595


namespace quadratic_form_proof_l2085_208546

theorem quadratic_form_proof (k : ℝ) (a b c : ℝ) (h1 : 8*k^2 - 16*k + 28 = a * (k + b)^2 + c) (h2 : a = 8) (h3 : b = -1) (h4 : c = 20) : c / b = -20 :=
by {
  sorry
}

end quadratic_form_proof_l2085_208546


namespace tree_height_when_planted_l2085_208520

def initial_height (current_height : ℕ) (growth_rate : ℕ) (current_age : ℕ) (initial_age : ℕ) : ℕ :=
  current_height - (current_age - initial_age) * growth_rate

theorem tree_height_when_planted :
  initial_height 23 3 7 1 = 5 :=
by
  sorry

end tree_height_when_planted_l2085_208520


namespace max_value_m_l2085_208501

/-- Proof that the inequality (a^2 + 4(b^2 + c^2))(b^2 + 4(a^2 + c^2))(c^2 + 4(a^2 + b^2)) 
    is greater than or equal to 729 for all a, b, c ∈ ℝ \ {0} with 
    |1/a| + |1/b| + |1/c| ≤ 3. -/
theorem max_value_m (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h_cond : |1 / a| + |1 / b| + |1 / c| ≤ 3) :
  (a^2 + 4 * (b^2 + c^2)) * (b^2 + 4 * (a^2 + c^2)) * (c^2 + 4 * (a^2 + b^2)) ≥ 729 :=
by {
  sorry
}

end max_value_m_l2085_208501


namespace Sandy_goal_water_l2085_208557

-- Definitions based on the conditions in problem a)
def milliliters_per_interval := 500
def time_per_interval := 2
def total_time := 12
def milliliters_to_liters := 1000

-- The goal statement that proves the question == answer given conditions.
theorem Sandy_goal_water : (milliliters_per_interval * (total_time / time_per_interval)) / milliliters_to_liters = 3 := by
  sorry

end Sandy_goal_water_l2085_208557


namespace multiply_negatives_l2085_208526

theorem multiply_negatives : (- (1 / 2)) * (- 2) = 1 :=
by
  sorry

end multiply_negatives_l2085_208526


namespace central_angle_of_region_l2085_208509

theorem central_angle_of_region (A : ℝ) (θ : ℝ) (h : (1:ℝ) / 8 = (θ / 360) * A / A) : θ = 45 :=
by
  sorry

end central_angle_of_region_l2085_208509


namespace initial_integer_value_l2085_208576

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end initial_integer_value_l2085_208576


namespace average_snowfall_per_minute_l2085_208522

def total_snowfall := 550
def days_in_december := 31
def hours_per_day := 24
def minutes_per_hour := 60

theorem average_snowfall_per_minute :
  (total_snowfall : ℝ) / (days_in_december * hours_per_day * minutes_per_hour) = 550 / (31 * 24 * 60) :=
by
  sorry

end average_snowfall_per_minute_l2085_208522


namespace wall_height_proof_l2085_208528

-- The dimensions of the brick in meters
def brick_length : ℝ := 0.30
def brick_width : ℝ := 0.12
def brick_height : ℝ := 0.10

-- The dimensions of the wall in meters
def wall_length : ℝ := 6
def wall_width : ℝ := 4

-- The number of bricks needed
def number_of_bricks : ℝ := 1366.6666666666667

-- The height of the wall in meters
def wall_height : ℝ := 0.205

-- The volume of one brick
def volume_of_one_brick : ℝ := brick_length * brick_width * brick_height

-- The total volume of all bricks needed
def total_volume_of_bricks : ℝ := number_of_bricks * volume_of_one_brick

-- The volume of the wall
def volume_of_wall : ℝ := wall_length * wall_width * wall_height

-- Proof that the height of the wall is 0.205 meters
theorem wall_height_proof : volume_of_wall = total_volume_of_bricks :=
by
  -- use definitions to evaluate the equality
  sorry

end wall_height_proof_l2085_208528


namespace total_theme_parks_l2085_208578

theorem total_theme_parks 
  (J V M N : ℕ) 
  (hJ : J = 35)
  (hV : V = J + 40)
  (hM : M = J + 60)
  (hN : N = 2 * M) 
  : J + V + M + N = 395 :=
sorry

end total_theme_parks_l2085_208578


namespace shaded_region_area_correct_l2085_208500

noncomputable def area_shaded_region : ℝ := 
  let side_length := 2
  let radius := 1
  let area_square := side_length^2
  let area_circle := Real.pi * radius^2
  area_square - area_circle

theorem shaded_region_area_correct : area_shaded_region = 4 - Real.pi :=
  by
    sorry

end shaded_region_area_correct_l2085_208500


namespace transformed_equation_correct_l2085_208524
-- Import the necessary library

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation functions for the transformations
def translate_right (x : ℝ) : ℝ := x - 1
def translate_down (y : ℝ) : ℝ := y - 3

-- Define the transformed parabola equation
def transformed_parabola (x : ℝ) : ℝ := -2 * (translate_right x)^2 |> translate_down

-- The theorem stating the transformed equation
theorem transformed_equation_correct :
  ∀ x, transformed_parabola x = -2 * (x - 1)^2 - 3 :=
by { sorry }

end transformed_equation_correct_l2085_208524


namespace jellybeans_in_new_bag_l2085_208584

theorem jellybeans_in_new_bag (average_per_bag : ℕ) (num_bags : ℕ) (additional_avg_increase : ℕ) (total_jellybeans_old : ℕ) (total_jellybeans_new : ℕ) (num_bags_new : ℕ) (new_bag_jellybeans : ℕ) : 
  average_per_bag = 117 → 
  num_bags = 34 → 
  additional_avg_increase = 7 → 
  total_jellybeans_old = num_bags * average_per_bag → 
  total_jellybeans_new = (num_bags + 1) * (average_per_bag + additional_avg_increase) → 
  new_bag_jellybeans = total_jellybeans_new - total_jellybeans_old → 
  new_bag_jellybeans = 362 := 
by 
  intros 
  sorry

end jellybeans_in_new_bag_l2085_208584


namespace sine_product_inequality_l2085_208513

theorem sine_product_inequality :
  (1 / 8 : ℝ) < (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) ∧
                (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
sorry

end sine_product_inequality_l2085_208513


namespace trajectory_of_midpoint_l2085_208582

theorem trajectory_of_midpoint
  (M : ℝ × ℝ)
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hP : P = (4, 0))
  (hQ : Q.1^2 + Q.2^2 = 4)
  (M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + M.2^2 = 1 :=
sorry

end trajectory_of_midpoint_l2085_208582


namespace average_goals_is_92_l2085_208518

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end average_goals_is_92_l2085_208518


namespace sufficient_condition_for_inequality_l2085_208555

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0 :=
by
  sorry

end sufficient_condition_for_inequality_l2085_208555


namespace convert_rectangular_to_polar_l2085_208542

theorem convert_rectangular_to_polar (x y : ℝ) (h₁ : x = -2) (h₂ : y = -2) : 
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (2 * Real.sqrt 2, 5 * Real.pi / 4) := by
  sorry

end convert_rectangular_to_polar_l2085_208542


namespace trig_expression_value_l2085_208597

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 + 4 * Real.sin α * Real.cos α - 9 * Real.cos α ^ 2 = 21 / 10 :=
by
  sorry

end trig_expression_value_l2085_208597


namespace ratio_a_to_d_l2085_208573

theorem ratio_a_to_d (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : b / c = 2 / 3) 
  (h3 : c / d = 3 / 5) : 
  a / d = 1 / 2 :=
sorry

end ratio_a_to_d_l2085_208573


namespace percent_of_dollar_in_pocket_l2085_208591

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

theorem percent_of_dollar_in_pocket :
  let total_cents := penny_value + nickel_value + dime_value + quarter_value + half_dollar_value
  total_cents = 91 := by
  sorry

end percent_of_dollar_in_pocket_l2085_208591


namespace remainder_when_divided_by_x_minus_2_l2085_208502

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

-- State the theorem to be proved with the given conditions
theorem remainder_when_divided_by_x_minus_2 :
  f 2 = -11 :=
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_x_minus_2_l2085_208502


namespace sum_of_given_numbers_l2085_208534

theorem sum_of_given_numbers : 30 + 80000 + 700 + 60 = 80790 :=
  by
    sorry

end sum_of_given_numbers_l2085_208534


namespace range_of_m_l2085_208544

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) → m < 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_m_l2085_208544


namespace school_A_original_students_l2085_208510

theorem school_A_original_students 
  (x y : ℕ) 
  (h1 : x + y = 864) 
  (h2 : x - 32 = y + 80) : 
  x = 488 := 
by 
  sorry

end school_A_original_students_l2085_208510


namespace decreasing_on_interval_l2085_208511

variable {x m n : ℝ}

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := |x^2 - 2 * m * x + n|

theorem decreasing_on_interval
  (h : ∀ x, f x m n = |x^2 - 2 * m * x + n|)
  (h_cond : m^2 - n ≤ 0) :
  ∀ x y, x ≤ y → y ≤ m → f y m n ≤ f x m n :=
sorry

end decreasing_on_interval_l2085_208511


namespace toms_dad_gave_him_dimes_l2085_208585

theorem toms_dad_gave_him_dimes (original_dimes final_dimes dimes_given : ℕ)
  (h1 : original_dimes = 15)
  (h2 : final_dimes = 48)
  (h3 : final_dimes = original_dimes + dimes_given) :
  dimes_given = 33 :=
by
  -- Since the main goal here is just the statement, proof is omitted with sorry
  sorry

end toms_dad_gave_him_dimes_l2085_208585


namespace number_of_cars_l2085_208564

theorem number_of_cars (C : ℕ) : 
  let bicycles := 3
  let pickup_trucks := 8
  let tricycles := 1
  let car_tires := 4
  let bicycle_tires := 2
  let pickup_truck_tires := 4
  let tricycle_tires := 3
  let total_tires := 101
  (4 * C + 3 * bicycle_tires + 8 * pickup_truck_tires + 1 * tricycle_tires = total_tires) → C = 15 := by
  intros h
  sorry

end number_of_cars_l2085_208564


namespace arrangement_of_accommodation_l2085_208554

open Nat

noncomputable def num_arrangements_accommodation : ℕ :=
  (factorial 13) / ((factorial 2) * (factorial 2) * (factorial 2) * (factorial 2))

theorem arrangement_of_accommodation : num_arrangements_accommodation = 389188800 := by
  sorry

end arrangement_of_accommodation_l2085_208554


namespace no_positive_sequence_exists_l2085_208559

theorem no_positive_sequence_exists:
  ¬ (∃ (b : ℕ → ℝ), (∀ n, b n > 0) ∧ (∀ m : ℕ, (∑' k, b ((k + 1) * m)) = (1 / m))) :=
by
  sorry

end no_positive_sequence_exists_l2085_208559


namespace triangle_angle_sum_l2085_208521

theorem triangle_angle_sum (a b : ℝ) (ha : a = 40) (hb : b = 60) : ∃ x : ℝ, x = 180 - (a + b) :=
by
  use 80
  sorry

end triangle_angle_sum_l2085_208521


namespace max_value_under_constraint_l2085_208560

noncomputable def max_value_expression (a b c : ℝ) : ℝ :=
3 * a * b - 3 * b * c + 2 * c^2

theorem max_value_under_constraint
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 1) :
  max_value_expression a b c ≤ 3 :=
sorry

end max_value_under_constraint_l2085_208560


namespace printer_to_enhanced_ratio_l2085_208547

def B : ℕ := 2125
def P : ℕ := 2500 - B
def E : ℕ := B + 500
def total_price := E + P

theorem printer_to_enhanced_ratio :
  (P : ℚ) / total_price = 1 / 8 := 
by {
  -- skipping the proof
  sorry
}

end printer_to_enhanced_ratio_l2085_208547


namespace bird_families_flew_away_to_Africa_l2085_208571

theorem bird_families_flew_away_to_Africa 
  (B : ℕ) (n : ℕ) (hB94 : B = 94) (hB_A_plus_n : B = n + 47) : n = 47 :=
by
  sorry

end bird_families_flew_away_to_Africa_l2085_208571


namespace complex_is_purely_imaginary_iff_a_eq_2_l2085_208575

theorem complex_is_purely_imaginary_iff_a_eq_2 (a : ℝ) :
  (a = 2) ↔ ((a^2 - 4 = 0) ∧ (a + 2 ≠ 0)) :=
by sorry

end complex_is_purely_imaginary_iff_a_eq_2_l2085_208575


namespace find_integer_K_l2085_208543

-- Definitions based on the conditions
def is_valid_K (K Z : ℤ) : Prop :=
  Z = K^4 ∧ 3000 < Z ∧ Z < 4000 ∧ K > 1 ∧ ∃ (z : ℤ), K^4 = z^3

theorem find_integer_K :
  ∃ (K : ℤ), is_valid_K K 2401 :=
by
  sorry

end find_integer_K_l2085_208543


namespace total_bill_l2085_208583

theorem total_bill (m : ℝ) (h1 : m = 10 * (m / 10 + 3) - 27) : m = 270 :=
by
  sorry

end total_bill_l2085_208583


namespace fill_bucket_time_l2085_208587

theorem fill_bucket_time (time_full_bucket : ℕ) (fraction : ℚ) (time_two_thirds_bucket : ℕ) 
  (h1 : time_full_bucket = 150) (h2 : fraction = 2 / 3) : time_two_thirds_bucket = 100 :=
sorry

end fill_bucket_time_l2085_208587


namespace f_decreasing_on_0_1_l2085_208581

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem f_decreasing_on_0_1 : ∀ (x1 x2 : ℝ), (x1 ∈ Set.Ioo 0 1) → (x2 ∈ Set.Ioo 0 1) → (x1 < x2) → (f x1 < f x2) := by
  sorry

end f_decreasing_on_0_1_l2085_208581


namespace cone_lateral_area_l2085_208563

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  π * r * l = 15 * π := by
  sorry

end cone_lateral_area_l2085_208563


namespace locus_of_centers_of_circles_l2085_208586

structure Point (α : Type _) :=
(x : α)
(y : α)

noncomputable def perpendicular_bisector {α : Type _} [LinearOrderedField α] (A B : Point α) : Set (Point α) :=
  {C | ∃ m b : α, C.y = m * C.x + b ∧ A.y = m * A.x + b ∧ B.y = m * B.x + b ∧
                 (A.x - B.x) * C.x + (A.y - B.y) * C.y = (A.x^2 + A.y^2 - B.x^2 - B.y^2) / 2}

theorem locus_of_centers_of_circles {α : Type _} [LinearOrderedField α] (A B : Point α) :
  (∀ (C : Point α), (∃ r : α, r > 0 ∧ ∃ k: α, (C.x - A.x)^2 + (C.y - A.y)^2 = r^2 ∧ (C.x - B.x)^2 + (C.y - B.y)^2 = r^2) 
  → C ∈ perpendicular_bisector A B) :=
by
  sorry

end locus_of_centers_of_circles_l2085_208586


namespace company_needs_86_workers_l2085_208579

def profit_condition (n : ℕ) : Prop :=
  147 * n > 600 + 140 * n

theorem company_needs_86_workers (n : ℕ) : profit_condition n → n ≥ 86 :=
by
  intro h
  sorry

end company_needs_86_workers_l2085_208579


namespace common_tangents_l2085_208530

theorem common_tangents (r1 r2 d : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 4) : 
  ∀ (n : ℕ), (n = 1) → ¬ (∃ (d : ℝ), 
    (6 < d ∧ d < 14 ∧ n = 2) ∨ 
    (d = 14 ∧ n = 3) ∨ 
    (d < 6 ∧ n = 0) ∨ 
    (d > 14 ∧ n = 4)) :=
by
  intro n h
  sorry

end common_tangents_l2085_208530
