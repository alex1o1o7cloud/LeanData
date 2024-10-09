import Mathlib

namespace B_alone_completion_l796_79625

-- Define the conditions:
def A_efficiency_rel_to_B (A B: ℕ → Prop) : Prop :=
  ∀ (x: ℕ), B x → A (2 * x)

def together_job_completion (A B: ℕ → Prop) : Prop :=
  ∀ (t: ℕ), t = 20 → (∃ (x y : ℕ), B x ∧ A y ∧ (1/x + 1/y = 1/t))

-- Define the theorem:
theorem B_alone_completion (A B: ℕ → Prop) (h1 : A_efficiency_rel_to_B A B) (h2 : together_job_completion A B) :
  ∃ (x: ℕ), B x ∧ x = 30 :=
sorry

end B_alone_completion_l796_79625


namespace min_value_fraction_108_l796_79692

noncomputable def min_value_fraction (x y z w : ℝ) : ℝ :=
(x + y) / (x * y * z * w)

theorem min_value_fraction_108 (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) (h_sum : x + y + z + w = 1) :
  min_value_fraction x y z w = 108 :=
sorry

end min_value_fraction_108_l796_79692


namespace sum_of_squared_distances_range_l796_79651

theorem sum_of_squared_distances_range
  (φ : ℝ)
  (x : ℝ := 2 * Real.cos φ)
  (y : ℝ := 3 * Real.sin φ)
  (A : ℝ × ℝ := (1, Real.sqrt 3))
  (B : ℝ × ℝ := (-Real.sqrt 3, 1))
  (C : ℝ × ℝ := (-1, -Real.sqrt 3))
  (D : ℝ × ℝ := (Real.sqrt 3, -1))
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2)
  (PD := (x - D.1)^2 + (y - D.2)^2) :
  32 ≤ PA + PB + PC + PD ∧ PA + PB + PC + PD ≤ 52 :=
  by sorry

end sum_of_squared_distances_range_l796_79651


namespace decimal_0_0_1_7_eq_rational_l796_79697

noncomputable def infinite_loop_decimal_to_rational_series (a : ℚ) (r : ℚ) : ℚ :=
  a / (1 - r)

theorem decimal_0_0_1_7_eq_rational :
  infinite_loop_decimal_to_rational_series (17 / 1000) (1 / 100) = 17 / 990 :=
by
  sorry

end decimal_0_0_1_7_eq_rational_l796_79697


namespace initial_rate_of_interest_l796_79650

theorem initial_rate_of_interest (P : ℝ) (R : ℝ) 
  (h1 : 1680 = (P * R * 5) / 100) 
  (h2 : 1680 = (P * 5 * 4) / 100) : 
  R = 4 := 
by 
  sorry

end initial_rate_of_interest_l796_79650


namespace Proof_l796_79683

-- Definitions for the conditions
def Snakes : Type := {s : Fin 20 // s < 20}
def Purple (s : Snakes) : Prop := s.val < 6
def Happy (s : Snakes) : Prop := s.val >= 6 ∧ s.val < 14
def CanAdd (s : Snakes) : Prop := ∃ h ∈ Finset.Ico 6 14, h = s.val
def CanSubtract (s : Snakes) : Prop := ¬Purple s

-- Conditions extraction
axiom SomeHappyCanAdd : ∃ s : Snakes, Happy s ∧ CanAdd s
axiom NoPurpleCanSubtract : ∀ s : Snakes, Purple s → ¬CanSubtract s
axiom CantSubtractCantAdd : ∀ s : Snakes, ¬CanSubtract s → ¬CanAdd s

-- Theorem statement depending on conditions
theorem Proof :
    (∀ s : Snakes, CanSubtract s → ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬CanSubtract s) :=
by {
  sorry -- Proof required here
}

end Proof_l796_79683


namespace geometric_seq_xyz_eq_neg_two_l796_79607

open Real

noncomputable def geometric_seq (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_seq_xyz_eq_neg_two (x y z : ℝ) :
  geometric_seq (-1) x y z (-2) → x * y * z = -2 :=
by
  intro h
  obtain ⟨r, hx, hy, hz, he⟩ := h
  rw [hx, hy, hz, he] at *
  sorry

end geometric_seq_xyz_eq_neg_two_l796_79607


namespace uncle_zhang_age_l796_79637

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end uncle_zhang_age_l796_79637


namespace Vanya_bullets_l796_79665

theorem Vanya_bullets (initial_bullets : ℕ) (hits : ℕ) (shots_made : ℕ) (hits_reward : ℕ) :
  initial_bullets = 10 →
  shots_made = 14 →
  hits = shots_made / 2 →
  hits_reward = 3 →
  (initial_bullets + hits * hits_reward) - shots_made = 17 :=
by
  intros
  sorry

end Vanya_bullets_l796_79665


namespace three_times_x_greater_than_four_l796_79694

theorem three_times_x_greater_than_four (x : ℝ) : 3 * x > 4 := by
  sorry

end three_times_x_greater_than_four_l796_79694


namespace find_chord_eq_l796_79643

-- Given conditions 
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def point_p : (ℝ × ℝ) := (3, 2)
def midpoint_chord (p1 p2 p : (ℝ × ℝ)) : Prop := p.fst = (p1.fst + p2.fst) / 2 ∧ p.snd = (p1.snd + p2.snd) / 2

-- Conditions in Lean definition
def conditions (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 ∧ midpoint_chord (x1,y1) (x2,y2) point_p

-- The statement to prove
theorem find_chord_eq (x1 y1 x2 y2 : ℝ) (h : conditions x1 y1 x2 y2) :
  ∃ m b : ℝ, (m = -2 / 3) ∧ b = 2 - m * 3 ∧ (∀ x y : ℝ, y = m * x + b → 2 * x + 3 * y - 12 = 0) :=
by {
  sorry
}

end find_chord_eq_l796_79643


namespace lila_will_have_21_tulips_l796_79686

def tulip_orchid_ratio := 3 / 4

def initial_orchids := 16

def added_orchids := 12

def total_orchids : ℕ := initial_orchids + added_orchids

def groups_of_orchids : ℕ := total_orchids / 4

def total_tulips : ℕ := 3 * groups_of_orchids

theorem lila_will_have_21_tulips :
  total_tulips = 21 := by
  sorry

end lila_will_have_21_tulips_l796_79686


namespace sum_of_possible_values_l796_79605

theorem sum_of_possible_values (x y : ℝ) (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 6) :
  ∃ (a b : ℝ), (a - 2) * (b - 2) = 4 ∧ (a - 2) * (b - 2) = 9 ∧ 4 + 9 = 13 :=
sorry

end sum_of_possible_values_l796_79605


namespace calculate_loss_percentage_l796_79634

theorem calculate_loss_percentage
  (CP SP₁ SP₂ : ℝ)
  (h₁ : SP₁ = CP * 1.05)
  (h₂ : SP₂ = 1140) :
  (CP = 1200) → (SP₁ = 1260) → ((CP - SP₂) / CP * 100 = 5) :=
by
  intros h1 h2
  -- Here, we will eventually provide the actual proof steps.
  sorry

end calculate_loss_percentage_l796_79634


namespace number_of_10_digit_integers_with_consecutive_twos_l796_79620

open Nat

-- Define the total number of 10-digit integers using only '1' and '2's
def total_10_digit_numbers : ℕ := 2^10

-- Define the Fibonacci function
def fibonacci : ℕ → ℕ
| 0    => 1
| 1    => 2
| n+2  => fibonacci (n+1) + fibonacci n

-- Calculate the 10th Fibonacci number for the problem context
def F_10 : ℕ := fibonacci 9 + fibonacci 8

-- Prove that the number of 10-digit integers with at least one pair of consecutive '2's is 880
theorem number_of_10_digit_integers_with_consecutive_twos :
  total_10_digit_numbers - F_10 = 880 :=
by
  sorry

end number_of_10_digit_integers_with_consecutive_twos_l796_79620


namespace all_positive_integers_are_clever_l796_79669

theorem all_positive_integers_are_clever : ∀ n : ℕ, 0 < n → ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (a^2 - b^2) / (c^2 + d^2) := 
by
  intros n h_pos
  sorry

end all_positive_integers_are_clever_l796_79669


namespace problem1_problem2_l796_79617

noncomputable def f (x a b : ℝ) := |x + a^2| + |x - b^2|

theorem problem1 (a b x : ℝ) (h : a^2 + b^2 - 2 * a + 2 * b + 2 = 0) :
  f x a b >= 3 ↔ x <= -0.5 ∨ x >= 1.5 :=
sorry

theorem problem2 (a b x : ℝ) (h : a + b = 4) :
  f x a b >= 8 :=
sorry

end problem1_problem2_l796_79617


namespace least_number_subtracted_l796_79693

theorem least_number_subtracted (x : ℕ) (y : ℕ) (h : 2590 - x = y) : 
  y % 9 = 6 ∧ y % 11 = 6 ∧ y % 13 = 6 → x = 10 := 
by
  sorry

end least_number_subtracted_l796_79693


namespace lily_milk_amount_l796_79626

def initial_milk : ℚ := 5
def milk_given_to_james : ℚ := 18 / 4
def milk_received_from_neighbor : ℚ := 7 / 4

theorem lily_milk_amount : (initial_milk - milk_given_to_james + milk_received_from_neighbor) = 9 / 4 :=
by
  sorry

end lily_milk_amount_l796_79626


namespace last_three_digits_of_5_power_15000_l796_79687

theorem last_three_digits_of_5_power_15000:
  (5^15000) % 1000 = 1 % 1000 :=
by
  have h : 5^500 % 1000 = 1 % 1000 := by sorry
  sorry

end last_three_digits_of_5_power_15000_l796_79687


namespace gcd_lcm_45_75_l796_79630

theorem gcd_lcm_45_75 : gcd 45 75 = 15 ∧ lcm 45 75 = 1125 :=
by sorry

end gcd_lcm_45_75_l796_79630


namespace area_of_rectangular_field_l796_79628

theorem area_of_rectangular_field 
  (P L W : ℕ) 
  (hP : P = 120) 
  (hL : L = 3 * W) 
  (hPerimeter : 2 * L + 2 * W = P) : 
  (L * W = 675) :=
by 
  sorry

end area_of_rectangular_field_l796_79628


namespace segment_length_reflection_l796_79674

theorem segment_length_reflection (Z : ℝ×ℝ) (Z' : ℝ×ℝ) (hx : Z = (5, 2)) (hx' : Z' = (5, -2)) :
  dist Z Z' = 4 := by
  sorry

end segment_length_reflection_l796_79674


namespace fraction_value_l796_79635

def x : ℚ := 4 / 7
def y : ℚ := 8 / 11

theorem fraction_value : (7 * x + 11 * y) / (49 * x * y) = 231 / 56 := by
  sorry

end fraction_value_l796_79635


namespace point_in_second_quadrant_range_l796_79603

theorem point_in_second_quadrant_range (m : ℝ) :
  (m - 3 < 0 ∧ m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  sorry

end point_in_second_quadrant_range_l796_79603


namespace interest_rate_difference_l796_79671

def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def si1 (R1 : ℕ) : ℕ := simple_interest 800 R1 10
def si2 (R2 : ℕ) : ℕ := simple_interest 800 R2 10

theorem interest_rate_difference (R1 R2 : ℕ) (h : si2 R2 = si1 R1 + 400) : R2 - R1 = 5 := 
by sorry

end interest_rate_difference_l796_79671


namespace min_value_of_x_plus_2y_l796_79657

theorem min_value_of_x_plus_2y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 :=
sorry

end min_value_of_x_plus_2y_l796_79657


namespace greater_quadratic_solution_l796_79673

theorem greater_quadratic_solution : ∀ (x : ℝ), x^2 + 15 * x - 54 = 0 → x = -18 ∨ x = 3 →
  max (-18) 3 = 3 := by
  sorry

end greater_quadratic_solution_l796_79673


namespace find_g_eq_minus_x_l796_79647

-- Define the function g and the given conditions.
def g (x : ℝ) : ℝ := sorry

axiom g0 : g 0 = 2
axiom g_xy : ∀ (x y : ℝ), g (x * y) = g ((x^2 + 2 * y^2) / 3) + 3 * (x - y)^2

-- State the problem: proving that g(x) = -x.
theorem find_g_eq_minus_x : ∀ (x : ℝ), g x = -x := by
  sorry

end find_g_eq_minus_x_l796_79647


namespace walking_distance_l796_79699

theorem walking_distance (west east : ℤ) (h_west : west = 5) (h_east : east = -5) : west + east = 10 := 
by 
  rw [h_west, h_east] 
  sorry

end walking_distance_l796_79699


namespace ingrid_income_l796_79602

theorem ingrid_income (combined_tax_rate : ℝ)
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_tax_rate : ℝ)
  (combined_income : ℝ)
  (combined_tax : ℝ) :
  combined_tax_rate = 0.35581395348837205 →
  john_income = 57000 →
  john_tax_rate = 0.3 →
  ingrid_tax_rate = 0.4 →
  combined_income = john_income + (combined_income - john_income) →
  combined_tax = (john_tax_rate * john_income) + (ingrid_tax_rate * (combined_income - john_income)) →
  combined_tax_rate = combined_tax / combined_income →
  combined_income = 57000 + 72000 :=
by
  sorry

end ingrid_income_l796_79602


namespace ways_to_write_1800_as_sum_of_twos_and_threes_l796_79661

theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  ∃ (n : ℕ), n = 301 ∧ ∀ (x y : ℕ), 2 * x + 3 * y = 1800 → ∃ (a : ℕ), (x, y) = (3 * a, 300 - a) :=
sorry

end ways_to_write_1800_as_sum_of_twos_and_threes_l796_79661


namespace total_money_is_102_l796_79608

-- Defining the amounts of money each person has
def Jack_money : ℕ := 26
def Ben_money : ℕ := Jack_money - 9
def Eric_money : ℕ := Ben_money - 10
def Anna_money : ℕ := Jack_money * 2

-- Defining the total amount of money
def total_money : ℕ := Eric_money + Ben_money + Jack_money + Anna_money

-- Proving the total money is 102
theorem total_money_is_102 : total_money = 102 :=
by
  -- this is where the proof would go
  sorry

end total_money_is_102_l796_79608


namespace alice_wins_chomp_l796_79685

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end alice_wins_chomp_l796_79685


namespace solve_for_x_l796_79640

theorem solve_for_x (x : ℝ) (h : (1 / 2) * (1 / 7) * x = 14) : x = 196 :=
by
  sorry

end solve_for_x_l796_79640


namespace product_polynomials_l796_79641

theorem product_polynomials (x : ℝ) : 
  (1 + x^3) * (1 - 2 * x + x^4) = 1 - 2 * x + x^3 - x^4 + x^7 :=
by sorry

end product_polynomials_l796_79641


namespace arithmetic_sqrt_sqrt_16_l796_79616

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l796_79616


namespace harvest_apples_l796_79689

def sacks_per_section : ℕ := 45
def sections : ℕ := 8
def total_sacks_per_day : ℕ := 360

theorem harvest_apples : sacks_per_section * sections = total_sacks_per_day := by
  sorry

end harvest_apples_l796_79689


namespace arithmetic_prog_sum_l796_79682

theorem arithmetic_prog_sum (a d : ℕ) (h1 : 15 * a + 105 * d = 60) : 2 * a + 14 * d = 8 :=
by
  sorry

end arithmetic_prog_sum_l796_79682


namespace general_equation_of_line_l796_79649

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define what it means for a line to pass through two points
def line_through_points (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  l A.1 A.2 ∧ l B.1 B.2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- The theorem that needs to be proven
theorem general_equation_of_line : line_through_points line_l A B := 
by
  sorry

end general_equation_of_line_l796_79649


namespace largest_value_of_c_l796_79677

theorem largest_value_of_c : ∀ c : ℝ, (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intros c hc
  have : (3 * c + 6) * (c - 2) = 9 * c := hc
  sorry

end largest_value_of_c_l796_79677


namespace max_possible_value_of_C_l796_79633

theorem max_possible_value_of_C (A B C D : ℕ) (h₁ : A + B + C + D = 200) (h₂ : A + B = 70) (h₃ : 0 < A) (h₄ : 0 < B) (h₅ : 0 < C) (h₆ : 0 < D) :
  C ≤ 129 :=
by
  sorry

end max_possible_value_of_C_l796_79633


namespace pineapple_cost_l796_79664

variables (P W : ℕ)

theorem pineapple_cost (h1 : 2 * P + 5 * W = 38) : P = 14 :=
sorry

end pineapple_cost_l796_79664


namespace multiply_eq_four_l796_79629

variables (a b c d : ℝ)

theorem multiply_eq_four (h1 : a = d) 
                         (h2 : b = c) 
                         (h3 : d + d = c * d) 
                         (h4 : b = d) 
                         (h5 : d + d = d * d) 
                         (h6 : c = 3) :
                         a * b = 4 := 
by 
  sorry

end multiply_eq_four_l796_79629


namespace percentage_per_annum_is_correct_l796_79645

-- Define the conditions of the problem
def banker_gain : ℝ := 24
def present_worth : ℝ := 600
def time : ℕ := 2

-- Define the formula for the amount due
def amount_due (r : ℝ) (t : ℕ) (PW : ℝ) : ℝ := PW * (1 + r * t)

-- Define the given conditions translated from the problem
def given_conditions (r : ℝ) : Prop :=
  amount_due r time present_worth = present_worth + banker_gain

-- Lean statement of the problem to be proved
theorem percentage_per_annum_is_correct :
  ∃ r : ℝ, given_conditions r ∧ r = 0.02 :=
by {
  sorry
}

end percentage_per_annum_is_correct_l796_79645


namespace remainder_of_150_div_k_l796_79695

theorem remainder_of_150_div_k (k : ℕ) (hk : k > 0) (h1 : 90 % (k^2) = 10) :
  150 % k = 2 := 
sorry

end remainder_of_150_div_k_l796_79695


namespace area_enclosed_by_abs_eq_l796_79614

theorem area_enclosed_by_abs_eq (x y : ℝ) : 
  (|x| + |3 * y| = 12) → (∃ area : ℝ, area = 96) :=
by
  sorry

end area_enclosed_by_abs_eq_l796_79614


namespace triangle_perimeter_l796_79679

-- Conditions as definitions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def has_sides (a b : ℕ) : Prop :=
  a = 4 ∨ b = 4 ∨ a = 9 ∨ b = 9

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the isosceles triangle with specified sides
structure IsoTriangle :=
  (a b c : ℕ)
  (iso : is_isosceles_triangle a b c)
  (valid_sides : has_sides a b ∧ has_sides a c ∧ has_sides b c)
  (triangle : triangle_inequality a b c)

-- The statement to prove perimeter
def perimeter (T : IsoTriangle) : ℕ :=
  T.a + T.b + T.c

-- The theorem we aim to prove
theorem triangle_perimeter (T : IsoTriangle) (h: T.a = 9 ∧ T.b = 9 ∧ T.c = 4) : perimeter T = 22 :=
sorry

end triangle_perimeter_l796_79679


namespace mean_value_of_interior_angles_of_quadrilateral_l796_79662

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_79662


namespace right_triangle_wy_expression_l796_79639

theorem right_triangle_wy_expression (α β : ℝ) (u v w y : ℝ)
    (h1 : (∀ x : ℝ, x^2 - u * x + v = 0 → x = Real.sin α ∨ x = Real.sin β))
    (h2 : (∀ x : ℝ, x^2 - w * x + y = 0 → x = Real.cos α ∨ x = Real.cos β))
    (h3 : α + β = Real.pi / 2) :
    w * y = u * v :=
sorry

end right_triangle_wy_expression_l796_79639


namespace total_profit_from_selling_30_necklaces_l796_79676

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end total_profit_from_selling_30_necklaces_l796_79676


namespace total_pigs_indeterminate_l796_79691

noncomputable def average_weight := 15
def underweight_threshold := 16
def max_underweight_pigs := 4

theorem total_pigs_indeterminate :
  ∃ (P U : ℕ), U ≤ max_underweight_pigs ∧ (average_weight = 15) → P = P :=
sorry

end total_pigs_indeterminate_l796_79691


namespace range_of_m_l796_79648

theorem range_of_m (m : ℝ) :
  let M := {x : ℝ | x ≤ m}
  let P := {x : ℝ | x ≥ -1}
  (M ∩ P = ∅) → m < -1 :=
by
  sorry

end range_of_m_l796_79648


namespace condition_neither_sufficient_nor_necessary_l796_79604
-- Import necessary library

-- Define the function and conditions
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- State the proof problem
theorem condition_neither_sufficient_nor_necessary :
  ∀ a : ℝ, (∀ x : ℝ, f x a = 0 -> x = 1/2) ↔ a^2 - 4 = 0 ∧ a ≤ -2 := sorry

end condition_neither_sufficient_nor_necessary_l796_79604


namespace Problem_l796_79609

def f (x : ℕ) : ℕ := x ^ 2 + 1
def g (x : ℕ) : ℕ := 2 * x - 1

theorem Problem : f (g (3 + 1)) = 50 := by
  sorry

end Problem_l796_79609


namespace no_real_solution_l796_79613

noncomputable def augmented_matrix (m : ℝ) : Matrix (Fin 2) (Fin 3) ℝ :=
  ![![m, 4, m+2], ![1, m, m]]

theorem no_real_solution (m : ℝ) :
  (∀ (a b : ℝ), ¬ ∃ (x y : ℝ), a * x + b * y = m ∧ a * x + b * y = 4 ∧ a * x + b * y = m + 2) ↔ m = 2 :=
by
sorry

end no_real_solution_l796_79613


namespace determine_delta_l796_79672

theorem determine_delta (r1 r2 r3 r4 r5 r6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ) (O Δ : ℕ) 
  (h_sums_rows : r1 + r2 + r3 + r4 + r5 + r6 = 190)
  (h_row1 : r1 = 29) (h_row2 : r2 = 33) (h_row3 : r3 = 33) 
  (h_row4 : r4 = 32) (h_row5 : r5 = 32) (h_row6 : r6 = 31)
  (h_sums_cols : c1 + c2 + c3 + c4 + c5 + c6 = 190)
  (h_col1 : c1 = 29) (h_col2 : c2 = 33) (h_col3 : c3 = 33) 
  (h_col4 : c4 = 32) (h_col5 : c5 = 32) (h_col6 : c6 = 31)
  (h_O : O = 6) : 
  Δ = 4 :=
by 
  sorry

end determine_delta_l796_79672


namespace find_weight_of_b_l796_79601

theorem find_weight_of_b (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : B = 31 :=
sorry

end find_weight_of_b_l796_79601


namespace find_a_sq_plus_b_sq_l796_79654

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 48
axiom h2 : a * b = 156

theorem find_a_sq_plus_b_sq : a^2 + b^2 = 1992 :=
by sorry

end find_a_sq_plus_b_sq_l796_79654


namespace dilution_problem_l796_79627
-- Definitions of the conditions
def volume_initial : ℝ := 15
def concentration_initial : ℝ := 0.60
def concentration_final : ℝ := 0.40
def amount_alcohol_initial : ℝ := volume_initial * concentration_initial

-- Proof problem statement in Lean 4
theorem dilution_problem : 
  ∃ (x : ℝ), x = 7.5 ∧ 
              amount_alcohol_initial = concentration_final * (volume_initial + x) :=
sorry

end dilution_problem_l796_79627


namespace parallel_lines_solution_l796_79684

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a = 0 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) ∨ 
  (∀ x y : ℝ, a = 1/4 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) :=
sorry

end parallel_lines_solution_l796_79684


namespace FruitKeptForNextWeek_l796_79624

/-- Define the variables and conditions -/
def total_fruit : ℕ := 10
def fruit_eaten : ℕ := 5
def fruit_brought_on_friday : ℕ := 3

/-- Define what we need to prove -/
theorem FruitKeptForNextWeek : 
  ∃ k, total_fruit - fruit_eaten - fruit_brought_on_friday = k ∧ k = 2 :=
by
  sorry

end FruitKeptForNextWeek_l796_79624


namespace plumber_fix_cost_toilet_l796_79642

noncomputable def fixCost_Sink : ℕ := 30
noncomputable def fixCost_Shower : ℕ := 40

theorem plumber_fix_cost_toilet
  (T : ℕ)
  (Earnings1 : ℕ := 3 * T + 3 * fixCost_Sink)
  (Earnings2 : ℕ := 2 * T + 5 * fixCost_Sink)
  (Earnings3 : ℕ := T + 2 * fixCost_Shower + 3 * fixCost_Sink)
  (MaxEarnings : ℕ := 250) :
  Earnings2 = MaxEarnings → T = 50 :=
by
  sorry

end plumber_fix_cost_toilet_l796_79642


namespace negation_of_universal_l796_79663

variable {f g : ℝ → ℝ}

theorem negation_of_universal :
  ¬ (∀ x : ℝ, f x * g x ≠ 0) ↔ ∃ x₀ : ℝ, f x₀ = 0 ∨ g x₀ = 0 :=
by
  sorry

end negation_of_universal_l796_79663


namespace inequality_must_hold_l796_79600

theorem inequality_must_hold (a b c : ℝ) (h : (a / c^2) > (b / c^2)) (hc : c ≠ 0) : a^2 > b^2 :=
sorry

end inequality_must_hold_l796_79600


namespace decreasing_interval_l796_79619

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 4)

theorem decreasing_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), x ∈ Set.Icc (3 * Real.pi / 4) (2 * Real.pi) ↔ (∀ ε > 0, f x > f (x + ε)) := 
sorry

end decreasing_interval_l796_79619


namespace martin_speed_first_half_l796_79667

variable (v : ℝ) -- speed during the first half of the trip

theorem martin_speed_first_half
    (trip_duration : ℝ := 8)              -- The trip lasted 8 hours
    (speed_second_half : ℝ := 85)          -- Speed during the second half of the trip
    (total_distance : ℝ := 620)            -- Total distance traveled
    (time_each_half : ℝ := trip_duration / 2) -- Each half of the trip took half of the total time
    (distance_second_half : ℝ := speed_second_half * time_each_half)
    (distance_first_half : ℝ := total_distance - distance_second_half) :
    v = distance_first_half / time_each_half :=
by
  sorry

end martin_speed_first_half_l796_79667


namespace simplify_and_evaluate_l796_79696

theorem simplify_and_evaluate :
  ∀ (a b : ℚ), a = 2 → b = -1/2 → (a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l796_79696


namespace fraction_value_l796_79655

theorem fraction_value :
  (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end fraction_value_l796_79655


namespace series_sum_l796_79615

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l796_79615


namespace distance_to_directrix_l796_79658

theorem distance_to_directrix (x y d : ℝ) (a b c : ℝ) (F1 F2 M : ℝ × ℝ)
  (h_ellipse : x^2 / 25 + y^2 / 9 = 1)
  (h_a : a = 5)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_M_on_ellipse : M.snd^2 / (a^2) + M.fst^2 / (b^2) = 1)
  (h_dist_F1M : dist M F1 = 8) :
  d = 5 / 2 :=
by
  sorry

end distance_to_directrix_l796_79658


namespace John_new_weekly_earnings_l796_79656

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end John_new_weekly_earnings_l796_79656


namespace monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l796_79622

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem monotonic_intervals_a_eq_1 :
  ∀ x : ℝ, (0 < x ∧ x ≤ 2 → (f x 1) < (f 2 1)) ∧ 
           (2 ≤ x → (f x 1) > (f 2 1)) :=
by
  sorry

theorem range_of_a_no_zero_points_in_interval :
  ∀ a : ℝ, (∀ x : ℝ, (0 < x ∧ x < 1/3) → ((2 - a) * (x - 1) - 2 * Real.log x) > 0) ↔ 2 - 3 * Real.log 3 ≤ a :=
by
  sorry

end monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l796_79622


namespace range_of_a_l796_79644

noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

theorem range_of_a 
 (h : ∃ a, (∀ x₀ x₁ x₂, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ f x₀ a = 0 ∧ f x₁ a = 0 ∧ f x₂ a = 0)) :
  ∃ a, 0 < a ∧ a < 4 / Real.exp 2 :=
by
  sorry

end range_of_a_l796_79644


namespace power_equal_20mn_l796_79698

theorem power_equal_20mn (m n : ℕ) (P Q : ℕ) (hP : P = 2^m) (hQ : Q = 5^n) : 
  P^(2 * n) * Q^m = (20^(m * n)) :=
by
  sorry

end power_equal_20mn_l796_79698


namespace jamies_mother_twice_age_l796_79631

theorem jamies_mother_twice_age (y : ℕ) :
  ∀ (jamie_age_2010 mother_age_2010 : ℕ), 
  jamie_age_2010 = 10 → 
  mother_age_2010 = 5 * jamie_age_2010 → 
  mother_age_2010 + y = 2 * (jamie_age_2010 + y) → 
  2010 + y = 2040 :=
by
  intros jamie_age_2010 mother_age_2010 h_jamie h_mother h_eq
  sorry

end jamies_mother_twice_age_l796_79631


namespace part1_part2_l796_79632

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

theorem part1 (m : ℝ) : (∀ x : ℝ, f x m ≥ x - m*x) → -7 ≤ m ∧ m ≤ 1 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x m) → m ≤ 1 :=
by
  sorry

end part1_part2_l796_79632


namespace perimeter_of_rectangle_WXYZ_l796_79681

theorem perimeter_of_rectangle_WXYZ 
  (WE XF EG FH : ℝ)
  (h1 : WE = 10)
  (h2 : XF = 25)
  (h3 : EG = 20)
  (h4 : FH = 50) :
  let p := 53 -- By solving the equivalent problem, where perimeter is simplified to 53/1 which gives p = 53 and q = 1
  let q := 29
  p + q = 102 := 
by
  sorry

end perimeter_of_rectangle_WXYZ_l796_79681


namespace function_passes_through_point_l796_79646

theorem function_passes_through_point (a : ℝ) (x y : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (x = 1 ∧ y = 4) ↔ (y = a^(x-1) + 3) :=
sorry

end function_passes_through_point_l796_79646


namespace simplify_correct_l796_79675

def simplify_expression (a b : ℤ) : ℤ :=
  (30 * a + 70 * b) + (15 * a + 45 * b) - (12 * a + 60 * b)

theorem simplify_correct (a b : ℤ) : simplify_expression a b = 33 * a + 55 * b :=
by 
  sorry -- Proof to be filled in later

end simplify_correct_l796_79675


namespace swimmer_speed_in_still_water_l796_79606

-- Define the various given conditions as constants in Lean
def swimmer_distance : ℝ := 3
def river_current_speed : ℝ := 1.7
def time_taken : ℝ := 2.3076923076923075

-- Define what we need to prove: the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) :
  swimmer_distance = (v - river_current_speed) * time_taken → 
  v = 3 := by
  sorry

end swimmer_speed_in_still_water_l796_79606


namespace set_equivalence_l796_79666

open Set

def set_A : Set ℝ := { x | x^2 - 2 * x > 0 }
def set_B : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

theorem set_equivalence : (univ \ set_B) ∪ set_A = (Iic 1) ∪ Ioi 2 :=
sorry

end set_equivalence_l796_79666


namespace distribute_balls_into_boxes_l796_79688

theorem distribute_balls_into_boxes : 
  let n := 5
  let k := 4
  (n.choose (k - 1) + k - 1).choose (k - 1) = 56 :=
by
  sorry

end distribute_balls_into_boxes_l796_79688


namespace total_money_shared_l796_79659

theorem total_money_shared 
  (A B C D total : ℕ) 
  (h1 : A = 3 * 15)
  (h2 : B = 5 * 15)
  (h3 : C = 6 * 15)
  (h4 : D = 8 * 15)
  (h5 : A = 45) :
  total = A + B + C + D → total = 330 :=
by
  sorry

end total_money_shared_l796_79659


namespace negation_of_P_l796_79670

def P (x : ℝ) : Prop := x^2 + x - 1 < 0

theorem negation_of_P : (¬ ∀ x, P x) ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by
  sorry

end negation_of_P_l796_79670


namespace gcd_143_144_l796_79638

def a : ℕ := 143
def b : ℕ := 144

theorem gcd_143_144 : Nat.gcd a b = 1 :=
by
  sorry

end gcd_143_144_l796_79638


namespace computer_cost_l796_79611

theorem computer_cost (C : ℝ) (h1 : 0.10 * C = a) (h2 : 3 * C = b) (h3 : b - 1.10 * C = 2700) : 
  C = 2700 / 2.90 :=
by
  sorry

end computer_cost_l796_79611


namespace normals_intersect_at_single_point_l796_79610

-- Definitions of points on the parabola and distinct condition
variables {a b c : ℝ}

-- Condition stating that A, B, C are distinct points
def distinct_points (a b c : ℝ) : Prop :=
  (a - b) ≠ 0 ∧ (b - c) ≠ 0 ∧ (c - a) ≠ 0

-- Statement to be proved
theorem normals_intersect_at_single_point (habc : distinct_points a b c) :
  a + b + c = 0 :=
sorry

end normals_intersect_at_single_point_l796_79610


namespace angle_BAD_measure_l796_79652

theorem angle_BAD_measure (D_A_C : ℝ) (AB_AC : AB = AC) (AD_BD : AD = BD) (h : D_A_C = 39) :
  B_A_D = 70.5 :=
by sorry

end angle_BAD_measure_l796_79652


namespace ratio_S15_S5_l796_79623

variable {a : ℕ → ℝ}  -- The geometric sequence
variable {S : ℕ → ℝ}  -- The sum of the first n terms of the geometric sequence

-- Define the conditions:
axiom sum_of_first_n_terms (n : ℕ) : S n = a 0 * (1 - (a 1)^n) / (1 - a 1)
axiom ratio_S10_S5 : S 10 / S 5 = 1 / 2

-- Define the math proof problem:
theorem ratio_S15_S5 : S 15 / S 5 = 3 / 4 :=
  sorry

end ratio_S15_S5_l796_79623


namespace pair_product_not_72_l796_79660

theorem pair_product_not_72 : (2 * (-36) ≠ 72) :=
by
  sorry

end pair_product_not_72_l796_79660


namespace general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l796_79621

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℕ := 2 ^ (a n)

noncomputable def S (n : ℕ) : ℕ := (n * (2 * n + 2)) / 2

noncomputable def T (n : ℕ) : ℕ := (8 * (4 ^ n - 1)) / 3

-- Statements to be proved
theorem general_formula_an : ∀ n : ℕ, a n = 2 * n + 1 := sorry

theorem geometric_sequence_bn : ∀ n : ℕ, b n = 2 ^ (2 * n + 1) := sorry

theorem sum_of_geometric_sequence_Tn : ∀ n : ℕ, T n = (8 * (4 ^ n - 1)) / 3 := sorry

end general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l796_79621


namespace function_passes_through_vertex_l796_79618

theorem function_passes_through_vertex (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : a^(2 - 2) + 1 = 2 :=
by
  sorry

end function_passes_through_vertex_l796_79618


namespace school_survey_l796_79678

theorem school_survey (n k smallest largest : ℕ) (h1 : n = 24) (h2 : k = 4) (h3 : smallest = 3) (h4 : 1 ≤ smallest ∧ smallest ≤ n) (h5 : largest - smallest = (k - 1) * (n / k)) : 
  largest = 21 :=
by {
  sorry
}

end school_survey_l796_79678


namespace determine_sequence_parameters_l796_79680

variables {n : ℕ} {d q : ℝ} (h1 : 1 + (n-1) * d = 81) (h2 : 1 * q^(n-1) = 81) (h3 : q / d = 0.15)

theorem determine_sequence_parameters : n = 5 ∧ d = 20 ∧ q = 3 :=
by {
  -- Assumptions:
  -- h1: Arithmetic sequence, a1 = 1, an = 81
  -- h2: Geometric sequence, b1 = 1, bn = 81
  -- h3: q / d = 0.15
  -- Goal: n = 5, d = 20, q = 3
  sorry
}

end determine_sequence_parameters_l796_79680


namespace area_of_AFCH_l796_79612

-- Define the sides of the rectangles ABCD and EFGH
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the area of quadrilateral AFCH
def area_AFCH : ℝ := 52.5

-- The theorem we want to prove
theorem area_of_AFCH :
  AB = 9 ∧ BC = 5 ∧ EF = 3 ∧ FG = 10 → (area_AFCH = 52.5) :=
by
  sorry

end area_of_AFCH_l796_79612


namespace evaluate_expression_l796_79690

theorem evaluate_expression (x y z : ℝ) (hx : x ≠ 3) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 3) / (7 - z) * (y - 5) / (3 - x) * (z - 7) / (5 - y) = -1 :=
by 
  sorry

end evaluate_expression_l796_79690


namespace problem1_problem2_problem3_l796_79653

-- Proof for part 1
theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 :=
sorry

-- Proof for part 2
theorem problem2 (α : ℝ) :
  (-Real.sin (Real.pi + α) + Real.sin (-α) - Real.tan (2 * Real.pi + α)) / 
  (Real.tan (α + Real.pi) + Real.cos (-α) + Real.cos (Real.pi - α)) = -1 :=
sorry

-- Proof for part 3
theorem problem3 (α : ℝ) (h : Real.sin α + Real.cos α = 1 / 2) (hα : 0 < α ∧ α < Real.pi) :
  Real.sin α * Real.cos α = -3 / 8 :=
sorry

end problem1_problem2_problem3_l796_79653


namespace sum_of_solutions_l796_79668

theorem sum_of_solutions : 
  let a := 1
  let b := -7
  let c := -30
  (a * x^2 + b * x + c = 0) → ((-b / a) = 7) :=
by
  sorry

end sum_of_solutions_l796_79668


namespace provenance_of_positive_test_l796_79636

noncomputable def pr_disease : ℚ := 1 / 200
noncomputable def pr_no_disease : ℚ := 1 - pr_disease
noncomputable def pr_test_given_disease : ℚ := 1
noncomputable def pr_test_given_no_disease : ℚ := 0.05
noncomputable def pr_test : ℚ := pr_test_given_disease * pr_disease + pr_test_given_no_disease * pr_no_disease
noncomputable def pr_disease_given_test : ℚ := 
  (pr_test_given_disease * pr_disease) / pr_test

theorem provenance_of_positive_test : pr_disease_given_test = 20 / 219 :=
by
  sorry

end provenance_of_positive_test_l796_79636
