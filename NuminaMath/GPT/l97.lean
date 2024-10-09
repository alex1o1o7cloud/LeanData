import Mathlib

namespace largest_angle_in_triangle_l97_9758

theorem largest_angle_in_triangle (A B C : ℝ) 
  (a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin C = Real.sqrt 2 * Real.sin B)
  : B = 90 :=
by
  sorry

end largest_angle_in_triangle_l97_9758


namespace min_n_A0_An_ge_200_l97_9761

theorem min_n_A0_An_ge_200 :
  (∃ n : ℕ, (n * (n + 1)) / 3 ≥ 200) ∧
  (∀ m < 24, (m * (m + 1)) / 3 < 200) :=
sorry

end min_n_A0_An_ge_200_l97_9761


namespace missed_interior_angle_l97_9704

  theorem missed_interior_angle (n : ℕ) (x : ℝ) 
    (h1 : (n - 2) * 180 = 2750 + x) : x = 130 := 
  by sorry
  
end missed_interior_angle_l97_9704


namespace solution_set_eq_two_l97_9713

theorem solution_set_eq_two (m : ℝ) (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) :
  m = -1 :=
sorry

end solution_set_eq_two_l97_9713


namespace time_with_walkway_l97_9706

-- Definitions
def length_walkway : ℝ := 60
def time_against_walkway : ℝ := 120
def time_stationary_walkway : ℝ := 48

-- Theorem statement
theorem time_with_walkway (v w : ℝ)
  (h1 : 60 = 120 * (v - w))
  (h2 : 60 = 48 * v)
  (h3 : v = 1.25)
  (h4 : w = 0.75) :
  60 = 30 * (v + w) :=
by
  sorry

end time_with_walkway_l97_9706


namespace find_f_2017_l97_9791

theorem find_f_2017 {f : ℤ → ℤ}
  (symmetry : ∀ x : ℤ, f (-x) = -f x)
  (periodicity : ∀ x : ℤ, f (x + 4) = f x)
  (f_neg_1 : f (-1) = 2) :
  f 2017 = -2 :=
sorry

end find_f_2017_l97_9791


namespace belfried_industries_payroll_l97_9749

theorem belfried_industries_payroll (P : ℝ) (tax_paid : ℝ) : 
  ((P > 200000) ∧ (tax_paid = 0.002 * (P - 200000)) ∧ (tax_paid = 200)) → P = 300000 :=
by
  sorry

end belfried_industries_payroll_l97_9749


namespace joe_time_to_friends_house_l97_9779

theorem joe_time_to_friends_house
  (feet_moved : ℕ) (time_taken : ℕ) (remaining_distance : ℕ) (feet_in_yard : ℕ)
  (rate_of_movement : ℕ) (remaining_distance_feet : ℕ) (time_to_cover_remaining_distance : ℕ) :
  feet_moved = 80 →
  time_taken = 40 →
  remaining_distance = 90 →
  feet_in_yard = 3 →
  rate_of_movement = feet_moved / time_taken →
  remaining_distance_feet = remaining_distance * feet_in_yard →
  time_to_cover_remaining_distance = remaining_distance_feet / rate_of_movement →
  time_to_cover_remaining_distance = 135 :=
by
  sorry

end joe_time_to_friends_house_l97_9779


namespace sum_of_terms_l97_9769

def sequence_sum (n : ℕ) : ℕ :=
  n^2 + 2*n + 5

theorem sum_of_terms : sequence_sum 9 - sequence_sum 6 = 51 :=
by
  sorry

end sum_of_terms_l97_9769


namespace find_m_l97_9729

theorem find_m (x y m : ℤ) 
  (h1 : 4 * x + y = 34)
  (h2 : m * x - y = 20)
  (h3 : y ^ 2 = 4) 
  : m = 2 :=
sorry

end find_m_l97_9729


namespace xy_is_perfect_cube_l97_9754

theorem xy_is_perfect_cube (x y : ℕ) (h₁ : x = 5 * 2^4 * 3^3) (h₂ : y = 2^2 * 5^2) : ∃ z : ℕ, (x * y) = z^3 :=
by
  sorry

end xy_is_perfect_cube_l97_9754


namespace age_problem_l97_9777

theorem age_problem 
  (x y z u : ℕ)
  (h1 : x + 6 = 3 * (y - u))
  (h2 : x = y + z - u)
  (h3: y = x - u) 
  (h4 : x + 19 = 2 * z):
  x = 69 ∧ y = 47 ∧ z = 44 :=
by
  sorry

end age_problem_l97_9777


namespace ratio_lions_l97_9740

variable (Safari_Lions : Nat)
variable (Safari_Snakes : Nat)
variable (Safari_Giraffes : Nat)
variable (Savanna_Lions_Ratio : ℕ)
variable (Savanna_Snakes : Nat)
variable (Savanna_Giraffes : Nat)
variable (Savanna_Total : Nat)

-- Conditions
def conditions := 
  (Safari_Lions = 100) ∧
  (Safari_Snakes = Safari_Lions / 2) ∧
  (Safari_Giraffes = Safari_Snakes - 10) ∧
  (Savanna_Lions_Ratio * Safari_Lions + Savanna_Snakes + Savanna_Giraffes = Savanna_Total) ∧
  (Savanna_Snakes = 3 * Safari_Snakes) ∧
  (Savanna_Giraffes = Safari_Giraffes + 20) ∧
  (Savanna_Total = 410)

-- Theorem to prove
theorem ratio_lions : conditions Safari_Lions Safari_Snakes Safari_Giraffes Savanna_Lions_Ratio Savanna_Snakes Savanna_Giraffes Savanna_Total → Savanna_Lions_Ratio = 2 := by
  sorry

end ratio_lions_l97_9740


namespace ratio_w_to_y_l97_9717

theorem ratio_w_to_y (w x y z : ℝ) (h1 : w / x = 4 / 3) (h2 : y / z = 5 / 3) (h3 : z / x = 1 / 5) :
  w / y = 4 :=
by
  sorry

end ratio_w_to_y_l97_9717


namespace units_digit_of_7_pow_6_pow_5_l97_9700

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l97_9700


namespace smallest_among_l97_9783

theorem smallest_among {a b c d : ℝ} (h1 : a = Real.pi) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1) : 
  ∃ (x : ℝ), x = b ∧ x < a ∧ x < c ∧ x < d := 
by {
  sorry
}

end smallest_among_l97_9783


namespace smallest_angle_of_triangle_l97_9708

theorem smallest_angle_of_triangle :
  ∀ a b c : ℝ, a = 2 * Real.sqrt 10 → b = 3 * Real.sqrt 5 → c = 5 → 
  ∃ α β γ : ℝ, α + β + γ = π ∧ α = 45 * (π / 180) ∧ (a = c → α < β ∧ α < γ) ∧ (b = c → β < α ∧ β < γ) ∧ (c = a → γ < α ∧ γ < β) → 
  α = 45 * (π / 180) := 
sorry

end smallest_angle_of_triangle_l97_9708


namespace expression_to_diophantine_l97_9789

theorem expression_to_diophantine (x : ℝ) (y : ℝ) (n : ℕ) :
  (∃ (A B : ℤ), (x - y) ^ (2 * n + 1) = (A * x - B * y) ∧ (1969 : ℤ) * A^2 - (1968 : ℤ) * B^2 = 1) :=
sorry

end expression_to_diophantine_l97_9789


namespace base7_subtraction_l97_9742

theorem base7_subtraction (a b : ℕ) (ha : a = 4 * 7^3 + 3 * 7^2 + 2 * 7 + 1)
                            (hb : b = 1 * 7^3 + 2 * 7^2 + 3 * 7 + 4) :
                            a - b = 3 * 7^3 + 0 * 7^2 + 5 * 7 + 4 :=
by
  sorry

end base7_subtraction_l97_9742


namespace prime_roots_sum_product_l97_9786

theorem prime_roots_sum_product (p q : ℕ) (x1 x2 : ℤ)
  (hp: Nat.Prime p) (hq: Nat.Prime q) 
  (h_sum: x1 + x2 = -↑p)
  (h_prod: x1 * x2 = ↑q) : 
  p = 3 ∧ q = 2 :=
sorry

end prime_roots_sum_product_l97_9786


namespace value_of_sine_neg_10pi_over_3_l97_9722

theorem value_of_sine_neg_10pi_over_3 : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end value_of_sine_neg_10pi_over_3_l97_9722


namespace find_a_and_x_l97_9716

theorem find_a_and_x (a x : ℝ) (ha1 : x = (2 * a - 1)^2) (ha2 : x = (-a + 2)^2) : a = -1 ∧ x = 9 := 
by
  sorry

end find_a_and_x_l97_9716


namespace blue_stamp_price_l97_9781

theorem blue_stamp_price :
  ∀ (red_stamps blue_stamps yellow_stamps : ℕ) (red_price blue_price yellow_price total_earnings : ℝ),
    red_stamps = 20 →
    blue_stamps = 80 →
    yellow_stamps = 7 →
    red_price = 1.1 →
    yellow_price = 2 →
    total_earnings = 100 →
    (red_stamps * red_price + yellow_stamps * yellow_price + blue_stamps * blue_price = total_earnings) →
    blue_price = 0.80 :=
by
  intros red_stamps blue_stamps yellow_stamps red_price blue_price yellow_price total_earnings
  intros h_red_stamps h_blue_stamps h_yellow_stamps h_red_price h_yellow_price h_total_earnings
  intros h_earning_eq
  sorry

end blue_stamp_price_l97_9781


namespace tracy_total_books_collected_l97_9794

variable (weekly_books_first_week : ℕ)
variable (multiplier : ℕ)
variable (weeks_next_period : ℕ)

-- Conditions
def first_week_books := 9
def second_period_books_per_week := first_week_books * 10
def books_next_five_weeks := second_period_books_per_week * 5

-- Theorem
theorem tracy_total_books_collected : 
  (first_week_books + books_next_five_weeks) = 459 := 
by 
  sorry

end tracy_total_books_collected_l97_9794


namespace quadratic_inequality_solution_set_l97_9751

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ x^2 + a * x + b < 0) : b = 2 :=
sorry

end quadratic_inequality_solution_set_l97_9751


namespace cyclists_meet_at_start_l97_9763

theorem cyclists_meet_at_start (T : ℚ) (h1 : T = 5 * 7 * 9 / gcd (5 * 7) (gcd (7 * 9) (9 * 5))) : T = 157.5 :=
by
  sorry

end cyclists_meet_at_start_l97_9763


namespace equivalence_gcd_prime_power_l97_9744

theorem equivalence_gcd_prime_power (a b n : ℕ) :
  (∀ m, 0 < m ∧ m < n → Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔ 
  (∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k) :=
by
  sorry

end equivalence_gcd_prime_power_l97_9744


namespace polygon_diagonals_integer_l97_9793

theorem polygon_diagonals_integer (n : ℤ) : ∃ k : ℤ, 2 * k = n * (n - 3) := by
sorry

end polygon_diagonals_integer_l97_9793


namespace gumball_machine_total_l97_9790

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l97_9790


namespace p_computation_l97_9720

def p (x y : Int) : Int :=
  if x >= 0 ∧ y >= 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x + y > 0 then 2 * x + 2 * y
  else x + 4 * y

theorem p_computation : p (p 2 (-3)) (p (-3) (-4)) = 26 := by
  sorry

end p_computation_l97_9720


namespace rectangle_vertex_area_y_value_l97_9750

theorem rectangle_vertex_area_y_value (y : ℕ) (hy : 0 ≤ y) :
  let A := (0, y)
  let B := (10, y)
  let C := (0, 4)
  let D := (10, 4)
  10 * (y - 4) = 90 → y = 13 :=
by
  sorry

end rectangle_vertex_area_y_value_l97_9750


namespace terminal_side_of_angle_l97_9732

theorem terminal_side_of_angle (θ : Real) (h_cos : Real.cos θ < 0) (h_tan : Real.tan θ > 0) :
  θ ∈ {φ : Real | π < φ ∧ φ < 3 * π / 2} :=
sorry

end terminal_side_of_angle_l97_9732


namespace reciprocal_of_neg_five_l97_9702

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end reciprocal_of_neg_five_l97_9702


namespace sequence_term_index_l97_9795

open Nat

noncomputable def arithmetic_sequence_term (a₁ d n : ℕ) : ℕ :=
a₁ + (n - 1) * d

noncomputable def term_index (a₁ d term : ℕ) : ℕ :=
1 + (term - a₁) / d

theorem sequence_term_index {a₅ a₄₅ term : ℕ}
  (h₁: a₅ = 33)
  (h₂: a₄₅ = 153)
  (h₃: ∀ n, arithmetic_sequence_term 21 3 n = if n = 5 then 33 else if n = 45 then 153 else (21 + (n - 1) * 3))
  : term_index 21 3 201 = 61 :=
sorry

end sequence_term_index_l97_9795


namespace quadratic_roots_l97_9711

theorem quadratic_roots (A B C : ℝ) (r s p : ℝ) (h1 : 2 * A * r^2 + 3 * B * r + 4 * C = 0)
  (h2 : 2 * A * s^2 + 3 * B * s + 4 * C = 0) (h3 : r + s = -3 * B / (2 * A)) (h4 : r * s = 2 * C / A) :
  p = (16 * A * C - 9 * B^2) / (4 * A^2) :=
by
  sorry

end quadratic_roots_l97_9711


namespace no_integer_solution_for_equation_l97_9775

theorem no_integer_solution_for_equation :
  ¬ ∃ (x y : ℤ), x^2 + 3 * x * y - 2 * y^2 = 122 :=
sorry

end no_integer_solution_for_equation_l97_9775


namespace DeepakAgeProof_l97_9728

def RahulAgeAfter10Years (RahulAge : ℕ) : Prop := RahulAge + 10 = 26

def DeepakPresentAge (ratioRahul ratioDeepak : ℕ) (RahulAge : ℕ) : ℕ :=
  (2 * RahulAge) / ratioRahul

theorem DeepakAgeProof {DeepakCurrentAge : ℕ}
  (ratioRahul ratioDeepak RahulAge : ℕ)
  (hRatio : ratioRahul = 4)
  (hDeepakRatio : ratioDeepak = 2) :
  RahulAgeAfter10Years RahulAge →
  DeepakCurrentAge = DeepakPresentAge ratioRahul ratioDeepak RahulAge :=
  sorry

end DeepakAgeProof_l97_9728


namespace total_boys_in_class_l97_9730

theorem total_boys_in_class (n : ℕ)
  (h1 : 19 + 19 - 1 = n) :
  n = 37 :=
  sorry

end total_boys_in_class_l97_9730


namespace amount_spent_on_giftwrapping_and_expenses_l97_9785

theorem amount_spent_on_giftwrapping_and_expenses (total_spent : ℝ) (cost_of_gifts : ℝ) (h_total_spent : total_spent = 700) (h_cost_of_gifts : cost_of_gifts = 561) : 
  total_spent - cost_of_gifts = 139 :=
by
  rw [h_total_spent, h_cost_of_gifts]
  norm_num

end amount_spent_on_giftwrapping_and_expenses_l97_9785


namespace abs_ineq_cond_l97_9782

theorem abs_ineq_cond (a : ℝ) : 
  (-3 < a ∧ a < 1) ↔ (∃ x : ℝ, |x - a| + |x + 1| < 2) := sorry

end abs_ineq_cond_l97_9782


namespace gcd_values_count_l97_9725

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l97_9725


namespace gcd_72_120_180_is_12_l97_9788

theorem gcd_72_120_180_is_12 : Int.gcd (Int.gcd 72 120) 180 = 12 := by
  sorry

end gcd_72_120_180_is_12_l97_9788


namespace option_d_is_true_l97_9755

theorem option_d_is_true (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := 
  sorry

end option_d_is_true_l97_9755


namespace simplify_and_ratio_l97_9776

theorem simplify_and_ratio (k : ℤ) : 
  let a := 1
  let b := 2
  (∀ (k : ℤ), (6 * k + 12) / 6 = a * k + b) →
  (a / b = 1 / 2) :=
by
  intros
  sorry
  
end simplify_and_ratio_l97_9776


namespace find_angle_B_l97_9714

theorem find_angle_B 
  (a b : ℝ) (A B : ℝ) 
  (ha : a = 2 * Real.sqrt 2) 
  (hb : b = 2)
  (hA : A = Real.pi / 4) -- 45 degrees in radians
  (h_triangle : ∃ c, a^2 + b^2 - 2*a*b*Real.cos A = c^2 ∧ a^2 * Real.sin 45 = b^2 * Real.sin B) :
  B = Real.pi / 6 := -- 30 degrees in radians
sorry

end find_angle_B_l97_9714


namespace num_valid_pairs_l97_9764

/-- 
Let S(n) denote the sum of the digits of a natural number n.
Define the predicate to check if the pair (m, n) satisfies the given conditions.
-/
def S (n : ℕ) : ℕ := (toString n).foldl (fun acc ch => acc + ch.toNat - '0'.toNat) 0

def valid_pair (m n : ℕ) : Prop :=
  m < 100 ∧ n < 100 ∧ m > n ∧ m + S n = n + 2 * S m

/-- 
Theorem: There are exactly 99 pairs (m, n) that satisfy the given conditions.
-/
theorem num_valid_pairs : ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 99 ∧
  ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
sorry

end num_valid_pairs_l97_9764


namespace y_intercept_of_line_l97_9735

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end y_intercept_of_line_l97_9735


namespace value_when_x_is_neg1_l97_9766

theorem value_when_x_is_neg1 (p q : ℝ) (h : p + q = 2022) : 
  (p * (-1)^3 + q * (-1) + 1) = -2021 := by
  sorry

end value_when_x_is_neg1_l97_9766


namespace restore_original_problem_l97_9726

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l97_9726


namespace success_permutations_correct_l97_9799

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l97_9799


namespace circles_intersect_l97_9770

-- Definition of the first circle
def circleC := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 4 }

-- Definition of the second circle
def circleM := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 9 }

-- Prove that the circles intersect
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ circleC ∧ p ∈ circleM := 
sorry

end circles_intersect_l97_9770


namespace new_person_weight_l97_9778

theorem new_person_weight (W : ℝ) :
  (∃ (W : ℝ), (390 - W + 70) / 4 = (390 - W) / 4 + 3 ∧ (390 - W + W) = 390) → 
  W = 58 :=
by
  sorry

end new_person_weight_l97_9778


namespace malvina_card_value_sum_l97_9719

noncomputable def possible_values_sum: ℝ :=
  let value1 := 1
  let value2 := (-1 + Real.sqrt 5) / 2
  (value1 + value2) / 2

theorem malvina_card_value_sum
  (hx : ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ 
                 (x = Real.pi / 4 ∨ (Real.sin x = (-1 + Real.sqrt 5) / 2))):
  possible_values_sum = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end malvina_card_value_sum_l97_9719


namespace perfect_square_formula_l97_9731

theorem perfect_square_formula (x y : ℝ) :
  ¬∃ a b : ℝ, (x^2 + (1/4)*x + (1/4)) = (a + b)^2 ∧
  ¬∃ c d : ℝ, (x^2 + 2*x*y - y^2) = (c + d)^2 ∧
  ¬∃ e f : ℝ, (x^2 + x*y + y^2) = (e + f)^2 ∧
  ∃ g h : ℝ, (4*x^2 + 4*x + 1) = (g + h)^2 :=
sorry

end perfect_square_formula_l97_9731


namespace lines_intersect_lines_perpendicular_lines_parallel_l97_9780

variables (l1 l2 : ℝ) (m : ℝ)

def intersect (m : ℝ) : Prop :=
  m ≠ -1 ∧ m ≠ 3

def perpendicular (m : ℝ) : Prop :=
  m = 1/2

def parallel (m : ℝ) : Prop :=
  m = -1

theorem lines_intersect (m : ℝ) : intersect m :=
by sorry

theorem lines_perpendicular (m : ℝ) : perpendicular m :=
by sorry

theorem lines_parallel (m : ℝ) : parallel m :=
by sorry

end lines_intersect_lines_perpendicular_lines_parallel_l97_9780


namespace inequality_maintained_l97_9797

noncomputable def g (x a : ℝ) := x^2 + Real.log (x + a)

theorem inequality_maintained (x1 x2 a : ℝ) (hx1 : x1 = (-a + Real.sqrt (a^2 - 2))/2)
  (hx2 : x2 = (-a - Real.sqrt (a^2 - 2))/2):
  (a > Real.sqrt 2) → 
  (g x1 a + g x2 a) / 2 > g ((x1 + x2 ) / 2) a :=
by
  sorry

end inequality_maintained_l97_9797


namespace problem_equivalence_l97_9745

section ProblemDefinitions

def odd_function_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def statement_A (f : ℝ → ℝ) : Prop :=
  (∀ x < 0, f x = -Real.log (-x)) →
  odd_function_condition f →
  ∀ x > 0, f x ≠ -Real.log x

def statement_B (a : ℝ) : Prop :=
  Real.logb a (1 / 2) < 1 →
  (0 < a ∧ a < 1 / 2) ∨ (1 < a)

def statement_C : Prop :=
  ∀ x, (Real.logb 2 (Real.sqrt (x-1)) = (1/2) * Real.logb 2 x)

def statement_D (x1 x2 : ℝ) : Prop :=
  (x1 + Real.log x1 = 2) →
  (Real.log (1 - x2) - x2 = 1) →
  x1 + x2 = 1

end ProblemDefinitions

structure MathProofProblem :=
  (A : ∀ f : ℝ → ℝ, statement_A f)
  (B : ∀ a : ℝ, statement_B a)
  (C : statement_C)
  (D : ∀ x1 x2 : ℝ, statement_D x1 x2)

theorem problem_equivalence : MathProofProblem :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

end problem_equivalence_l97_9745


namespace domain_of_f_lg_x_l97_9756

theorem domain_of_f_lg_x : 
  ({x : ℝ | -1 ≤ x ∧ x ≤ 1} = {x | 10 ≤ x ∧ x ≤ 100}) ↔ (∃ f : ℝ → ℝ, ∀ x ∈ {x : ℝ | -1 ≤ x ∧ x ≤ 1}, f (x * x + 1) = f (Real.log x)) :=
sorry

end domain_of_f_lg_x_l97_9756


namespace sweets_ratio_l97_9718

theorem sweets_ratio (x : ℕ) (h1 : x + 4 + 7 = 22) : x / 22 = 1 / 2 :=
by
  sorry

end sweets_ratio_l97_9718


namespace isosceles_triangles_with_perimeter_27_count_l97_9759

theorem isosceles_triangles_with_perimeter_27_count :
  ∃ n, (∀ (a : ℕ), 7 ≤ a ∧ a ≤ 13 → ∃ (b : ℕ), b = 27 - 2*a ∧ b < 2*a) ∧ n = 7 :=
sorry

end isosceles_triangles_with_perimeter_27_count_l97_9759


namespace number_of_parakeets_per_cage_l97_9724

def num_cages : ℕ := 9
def parrots_per_cage : ℕ := 2
def total_birds : ℕ := 72

theorem number_of_parakeets_per_cage : (total_birds - (num_cages * parrots_per_cage)) / num_cages = 6 := by
  sorry

end number_of_parakeets_per_cage_l97_9724


namespace s_is_arithmetic_progression_l97_9734

variables (s : ℕ → ℕ) (ds1 ds2 : ℕ)

-- Conditions
axiom strictly_increasing : ∀ n, s n < s (n + 1)
axiom s_is_positive : ∀ n, 0 < s n
axiom s_s_is_arithmetic : ∃ d1, ∀ k, s (s k) = s (s 0) + k * d1
axiom s_s_plus1_is_arithmetic : ∃ d2, ∀ k, s (s k + 1) = s (s 0 + 1) + k * d2

-- Statement to prove
theorem s_is_arithmetic_progression : ∃ d, ∀ k, s (k + 1) = s 0 + k * d :=
sorry

end s_is_arithmetic_progression_l97_9734


namespace sample_and_size_correct_l97_9703

structure SchoolSurvey :=
  (students_selected : ℕ)
  (classes_selected : ℕ)

def survey_sample (survey : SchoolSurvey) : String :=
  "the physical condition of " ++ toString survey.students_selected ++ " students"

def survey_sample_size (survey : SchoolSurvey) : ℕ :=
  survey.students_selected

theorem sample_and_size_correct (survey : SchoolSurvey)
  (h_selected : survey.students_selected = 190)
  (h_classes : survey.classes_selected = 19) :
  survey_sample survey = "the physical condition of 190 students" ∧ 
  survey_sample_size survey = 190 :=
by
  sorry

end sample_and_size_correct_l97_9703


namespace weaving_problem_l97_9712

theorem weaving_problem
  (a : ℕ → ℝ) -- the sequence
  (a_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0)) -- arithmetic sequence condition
  (sum_seven_days : 7 * a 0 + 21 * (a 1 - a 0) = 21) -- sum in seven days
  (sum_days_2_5_8 : 3 * a 1 + 12 * (a 1 - a 0) = 15) -- sum on 2nd, 5th, and 8th days
  : a 10 = 15 := sorry

end weaving_problem_l97_9712


namespace find_x_plus_inv_x_l97_9757

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l97_9757


namespace total_space_needed_for_trees_l97_9727

def appleTreeWidth : ℕ := 10
def spaceBetweenAppleTrees : ℕ := 12
def numAppleTrees : ℕ := 2

def peachTreeWidth : ℕ := 12
def spaceBetweenPeachTrees : ℕ := 15
def numPeachTrees : ℕ := 2

def totalSpace : ℕ :=
  numAppleTrees * appleTreeWidth + spaceBetweenAppleTrees +
  numPeachTrees * peachTreeWidth + spaceBetweenPeachTrees

theorem total_space_needed_for_trees : totalSpace = 71 := by
  sorry

end total_space_needed_for_trees_l97_9727


namespace x_squared_y_minus_xy_squared_l97_9748

theorem x_squared_y_minus_xy_squared (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) : x^2 * y - x * y^2 = -6 := 
by 
  sorry

end x_squared_y_minus_xy_squared_l97_9748


namespace sides_increase_factor_l97_9787

theorem sides_increase_factor (s k : ℝ) (h : s^2 * 25 = k^2 * s^2) : k = 5 :=
by
  sorry

end sides_increase_factor_l97_9787


namespace C_days_to_finish_l97_9796

theorem C_days_to_finish (A B C : ℝ) 
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  -- Given equations
  have h1 : A + B = 1 / 15 := sorry
  have h2 : A + B + C = 1 / 11 := sorry
  -- Calculate C
  let C := 1 / 11 - 1 / 15
  -- Calculate days taken by C
  let days := 1 / C
  -- Prove the days equal to 41.25
  have days_eq : 41.25 = 165 / 4 := sorry
  exact sorry

end C_days_to_finish_l97_9796


namespace mike_peaches_l97_9701

theorem mike_peaches (initial_peaches picked_peaches : ℝ) (h1 : initial_peaches = 34.0) (h2 : picked_peaches = 86.0) : initial_peaches + picked_peaches = 120.0 :=
by
  rw [h1, h2]
  norm_num

end mike_peaches_l97_9701


namespace edward_games_start_l97_9741

theorem edward_games_start (sold_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h_sold : sold_games = 19) (h_boxes : boxes = 2) (h_game_box : games_per_box = 8) : 
  sold_games + boxes * games_per_box = 35 := 
  by 
    sorry

end edward_games_start_l97_9741


namespace sum_of_dimensions_l97_9747

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 30) (h2 : A * C = 60) (h3 : B * C = 90) : A + B + C = 24 := 
sorry

end sum_of_dimensions_l97_9747


namespace best_fit_slope_is_correct_l97_9768

open Real

noncomputable def slope_regression_line (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :=
  (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21

theorem best_fit_slope_is_correct (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4)
  (h_arith : (x4 - x3 = 2 * (x3 - x2)) ∧ (x3 - x2 = 2 * (x2 - x1))) :
  slope_regression_line x1 x2 x3 x4 y1 y2 y3 y4 = (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21 := 
sorry

end best_fit_slope_is_correct_l97_9768


namespace find_product_stu_l97_9767

-- Define hypotheses
variables (a x y c : ℕ)
variables (s t u : ℕ)
variable (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2))

-- Statement to prove the equivalent form and stu product
theorem find_product_stu (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2)) :
  ∃ s t u : ℕ, (a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5 ∧ s * t * u = 12 :=
sorry

end find_product_stu_l97_9767


namespace prime_cubic_condition_l97_9709

theorem prime_cubic_condition (p : ℕ) (hp : Nat.Prime p) (hp_prime : Nat.Prime (p^4 - 3 * p^2 + 9)) : p = 2 :=
sorry

end prime_cubic_condition_l97_9709


namespace problem_l97_9723

theorem problem (a b : ℝ) : a^6 + b^6 ≥ a^4 * b^2 + a^2 * b^4 := 
by sorry

end problem_l97_9723


namespace simon_legos_l97_9771

theorem simon_legos (B : ℝ) (K : ℝ) (x : ℝ) (simon_has : ℝ) 
  (h1 : simon_has = B * 1.20)
  (h2 : K = 40)
  (h3 : B = K + x)
  (h4 : simon_has = 72) : simon_has = 72 := by
  sorry

end simon_legos_l97_9771


namespace correct_propositions_l97_9721

theorem correct_propositions (a b c d m : ℝ) :
  (ab > 0 → a > b → (1 / a < 1 / b)) ∧
  (a > |b| → a ^ 2 > b ^ 2) ∧
  ¬ (a > b ∧ c < d → a - d > b - c) ∧
  ¬ (a < b ∧ m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end correct_propositions_l97_9721


namespace problem_statement_l97_9772

def operation (a b : ℝ) := (a + b) ^ 2

theorem problem_statement (x y : ℝ) : operation ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 :=
by
  sorry

end problem_statement_l97_9772


namespace max_rectangle_area_l97_9746

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l97_9746


namespace range_of_a_l97_9736

open Set Real

theorem range_of_a :
  let p := ∀ x : ℝ, |4 * x - 3| ≤ 1
  let q := ∀ x : ℝ, x^2 - (2 * a + 1) * x + (a * (a + 1)) ≤ 0
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q)
  → (∀ x : Icc (0 : ℝ) (1 / 2 : ℝ), a = x) :=
by
  intros
  sorry

end range_of_a_l97_9736


namespace bread_rise_time_l97_9739

theorem bread_rise_time (x : ℕ) (kneading_time : ℕ) (baking_time : ℕ) (total_time : ℕ) 
  (h1 : kneading_time = 10) 
  (h2 : baking_time = 30) 
  (h3 : total_time = 280) 
  (h4 : kneading_time + baking_time + 2 * x = total_time) : 
  x = 120 :=
sorry

end bread_rise_time_l97_9739


namespace neg_p_l97_9784

variable (x : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + x_0 + 2 ≤ 0

theorem neg_p : ¬p ↔ ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end neg_p_l97_9784


namespace employee_B_paid_l97_9710

variable (A B : ℝ)

/-- Two employees A and B are paid a total of Rs. 550 per week by their employer. 
A is paid 120 percent of the sum paid to B. -/
theorem employee_B_paid (h₁ : A + B = 550) (h₂ : A = 1.2 * B) : B = 250 := by
  -- Proof will go here
  sorry

end employee_B_paid_l97_9710


namespace rattlesnakes_count_l97_9743

theorem rattlesnakes_count (P B R V : ℕ) (h1 : P = 3 * B / 2) (h2 : V = 2 * 420 / 100) (h3 : P + R = 3 * 420 / 4) (h4 : P + B + R + V = 420) : R = 162 :=
by
  sorry

end rattlesnakes_count_l97_9743


namespace determine_n_l97_9737

noncomputable def polynomial (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for the actual polynomial function

theorem determine_n (n : ℕ) 
  (h_deg : ∀ a, polynomial n a = 2 → (3 ∣ a) ∨ a = 0)
  (h_deg' : ∀ a, polynomial n a = 1 → (3 ∣ (a + 2)))
  (h_deg'' : ∀ a, polynomial n a = 0 → (3 ∣ (a + 1)))
  (h_val : polynomial n (3*n+1) = 730) :
  n = 4 :=
sorry

end determine_n_l97_9737


namespace train_speed_in_kmh_l97_9760

-- Definitions from the conditions
def length_of_train : ℝ := 800 -- in meters
def time_to_cross_pole : ℝ := 20 -- in seconds
def conversion_factor : ℝ := 3.6 -- (km/h) per (m/s)

-- Statement to prove the train's speed in km/h
theorem train_speed_in_kmh :
  (length_of_train / time_to_cross_pole * conversion_factor) = 144 :=
  sorry

end train_speed_in_kmh_l97_9760


namespace books_new_arrivals_false_implies_statements_l97_9762

variable (Books : Type) -- representing the set of books in the library
variable (isNewArrival : Books → Prop) -- predicate stating if a book is a new arrival

theorem books_new_arrivals_false_implies_statements (H : ¬ ∀ b : Books, isNewArrival b) :
  (∃ b : Books, ¬ isNewArrival b) ∧ (¬ ∀ b : Books, isNewArrival b) :=
by
  sorry

end books_new_arrivals_false_implies_statements_l97_9762


namespace total_surface_area_l97_9733

theorem total_surface_area (r h : ℝ) (pi : ℝ) (area_base : ℝ) (curved_area_hemisphere : ℝ) (lateral_area_cylinder : ℝ) :
  (pi * r^2 = 144 * pi) ∧ (h = 10) ∧ (curved_area_hemisphere = 2 * pi * r^2) ∧ (lateral_area_cylinder = 2 * pi * r * h) →
  (curved_area_hemisphere + lateral_area_cylinder + area_base = 672 * pi) :=
by
  sorry

end total_surface_area_l97_9733


namespace polygon_diagonals_with_one_non_connecting_vertex_l97_9753

-- Define the number of sides in the polygon
def num_sides : ℕ := 17

-- Define the formula to calculate the number of diagonals in a polygon
def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the number of non-connecting vertex to any diagonal
def non_connected_diagonals (n : ℕ) : ℕ :=
  n - 3

-- The theorem to state and prove
theorem polygon_diagonals_with_one_non_connecting_vertex :
  total_diagonals num_sides - non_connected_diagonals num_sides = 105 :=
by
  -- The formal proof would go here
  sorry

end polygon_diagonals_with_one_non_connecting_vertex_l97_9753


namespace sequence_term_2012_l97_9738

theorem sequence_term_2012 :
  ∃ (a : ℕ → ℤ), a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2012 = 6 :=
sorry

end sequence_term_2012_l97_9738


namespace max_a_for_f_l97_9705

theorem max_a_for_f :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |a * x^2 - a * x + 1| ≤ 1) → a ≤ 8 :=
sorry

end max_a_for_f_l97_9705


namespace total_buckets_poured_l97_9773

-- Define given conditions
def initial_buckets : ℝ := 1
def additional_buckets : ℝ := 8.8

-- Theorem to prove the total number of buckets poured
theorem total_buckets_poured : 
  initial_buckets + additional_buckets = 9.8 :=
by
  sorry

end total_buckets_poured_l97_9773


namespace option_d_is_deductive_l97_9798

theorem option_d_is_deductive :
  (∀ (r : ℝ), S_r = Real.pi * r^2) → (S_1 = Real.pi) :=
by
  sorry

end option_d_is_deductive_l97_9798


namespace Grant_room_count_l97_9774

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

end Grant_room_count_l97_9774


namespace german_mo_2016_problem_1_l97_9715

theorem german_mo_2016_problem_1 (a b : ℝ) :
  a^2 + b^2 = 25 ∧ 3 * (a + b) - a * b = 15 ↔
  (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨
  (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4) :=
sorry

end german_mo_2016_problem_1_l97_9715


namespace rightmost_four_digits_of_5_pow_2023_l97_9707

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end rightmost_four_digits_of_5_pow_2023_l97_9707


namespace exponentiation_problem_l97_9792

theorem exponentiation_problem : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 :=
by sorry

end exponentiation_problem_l97_9792


namespace second_group_men_count_l97_9752

-- Define the conditions given in the problem
def men1 := 8
def days1 := 80
def days2 := 32

-- The question we need to answer
theorem second_group_men_count : 
  ∃ (men2 : ℕ), men1 * days1 = men2 * days2 ∧ men2 = 20 :=
by
  sorry

end second_group_men_count_l97_9752


namespace tin_silver_ratio_l97_9765

/-- Assuming a metal bar made of an alloy of tin and silver weighs 40 kg, 
    and loses 4 kg in weight when submerged in water,
    where 10 kg of tin loses 1.375 kg in water and 5 kg of silver loses 0.375 kg, 
    prove that the ratio of tin to silver in the bar is 2 : 3. -/
theorem tin_silver_ratio :
  ∃ (T S : ℝ), 
    T + S = 40 ∧ 
    0.1375 * T + 0.075 * S = 4 ∧ 
    T / S = 2 / 3 := 
by
  sorry

end tin_silver_ratio_l97_9765
