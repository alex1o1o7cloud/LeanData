import Mathlib

namespace physics_class_size_l162_162630

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 100)
  (h2 : physics_only + math_only + both = total_students)
  (h3 : both = 10)
  (h4 : physics_only + both = 2 * (math_only + both)) :
  physics_only + both = 62 := 
by sorry

end physics_class_size_l162_162630


namespace tickets_needed_for_equal_distribution_l162_162780

theorem tickets_needed_for_equal_distribution :
  ∃ k : ℕ, 865 + k ≡ 0 [MOD 9] ∧ k = 8 := sorry

end tickets_needed_for_equal_distribution_l162_162780


namespace mary_needs_to_add_6_25_more_cups_l162_162431

def total_flour_needed : ℚ := 8.5
def flour_already_added : ℚ := 2.25
def flour_to_add : ℚ := total_flour_needed - flour_already_added

theorem mary_needs_to_add_6_25_more_cups :
  flour_to_add = 6.25 :=
sorry

end mary_needs_to_add_6_25_more_cups_l162_162431


namespace find_divisor_l162_162376

theorem find_divisor (D : ℕ) : 
  let dividend := 109
  let quotient := 9
  let remainder := 1
  (dividend = D * quotient + remainder) → D = 12 :=
by
  sorry

end find_divisor_l162_162376


namespace compare_neg_fractions_l162_162160

theorem compare_neg_fractions : (- (3 / 2) < -1) :=
by sorry

end compare_neg_fractions_l162_162160


namespace seq_proof_l162_162256

noncomputable def arithmetic_seq (a1 a2 : ℤ) : Prop :=
  ∃ (d : ℤ), a1 = -1 + d ∧ a2 = a1 + d ∧ -4 = a1 + 3 * d

noncomputable def geometric_seq (b : ℤ) : Prop :=
  b = 2 ∨ b = -2

theorem seq_proof (a1 a2 b : ℤ) 
  (h1 : arithmetic_seq a1 a2) 
  (h2 : geometric_seq b) : 
  (a2 + a1 : ℚ) / b = 5 / 2 ∨ (a2 + a1 : ℚ) / b = -5 / 2 := by
  sorry

end seq_proof_l162_162256


namespace problem1_l162_162438

theorem problem1 : 13 + (-24) - (-40) = 29 := by
  sorry

end problem1_l162_162438


namespace symmetric_line_equation_l162_162415

theorem symmetric_line_equation : ∀ (x y : ℝ), (2 * x + 3 * y - 6 = 0) ↔ (3 * (x + 2) + 2 * (-y - 2) + 16 = 0) :=
by
  sorry

end symmetric_line_equation_l162_162415


namespace y_not_directly_nor_inversely_proportional_l162_162333

theorem y_not_directly_nor_inversely_proportional (x y : ℝ) :
  (∃ k : ℝ, x + y = 0 ∧ y = k * x) ∨
  (∃ k : ℝ, 3 * x * y = 10 ∧ x * y = k) ∨
  (∃ k : ℝ, x = 5 * y ∧ x = k * y) ∨
  (∃ k : ℝ, (y = 10 - x^2 - 3 * x) ∧ y ≠ k * x ∧ y * x ≠ k) ∨
  (∃ k : ℝ, x / y = Real.sqrt 3 ∧ x = k * y)
  → (∃ k : ℝ, y = 10 - x^2 - 3 * x ∧ y ≠ k * x ∧ y * x ≠ k) :=
by
  sorry

end y_not_directly_nor_inversely_proportional_l162_162333


namespace arithmetic_sqrt_of_25_l162_162344

theorem arithmetic_sqrt_of_25 : ∃ (x : ℝ), x^2 = 25 ∧ x = 5 :=
by 
  sorry

end arithmetic_sqrt_of_25_l162_162344


namespace find_positive_integer_solutions_l162_162848

theorem find_positive_integer_solutions :
  ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ (1 / (a : ℚ)) - (1 / (b : ℚ)) = 1 / 37 ∧ (a, b) = (38, 1332) :=
by
  sorry

end find_positive_integer_solutions_l162_162848


namespace find_c_for_radius_6_l162_162651

-- Define the circle equation and the radius condition.
theorem find_c_for_radius_6 (c : ℝ) :
  (∃ (x y : ℝ), x^2 + 8 * x + y^2 + 2 * y + c = 0) ∧ 6 = 6 -> c = -19 := 
by
  sorry

end find_c_for_radius_6_l162_162651


namespace fraction_equality_l162_162622

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end fraction_equality_l162_162622


namespace net_percentage_error_in_volume_l162_162394

theorem net_percentage_error_in_volume
  (a : ℝ)
  (side_error : ℝ := 0.03)
  (height_error : ℝ := -0.04)
  (depth_error : ℝ := 0.02) :
  ((1 + side_error) * (1 + height_error) * (1 + depth_error) - 1) * 100 = 0.8656 :=
by
  -- Placeholder for the proof
  sorry

end net_percentage_error_in_volume_l162_162394


namespace maximum_value_l162_162653

def expression (A B C : ℕ) : ℕ := A * B * C + A * B + B * C + C * A

theorem maximum_value (A B C : ℕ) 
  (h1 : A + B + C = 15) : 
  expression A B C ≤ 200 :=
sorry

end maximum_value_l162_162653


namespace projectile_highest_point_l162_162589

noncomputable def highest_point (v w_h w_v θ g : ℝ) : ℝ × ℝ :=
  let t := (v * Real.sin θ + w_v) / g
  let x := (v * t + w_h * t) * Real.cos θ
  let y := (v * t + w_v * t) * Real.sin θ - (1/2) * g * t^2
  (x, y)

theorem projectile_highest_point : highest_point 100 10 (-2) (Real.pi / 4) 9.8 = (561.94, 236) :=
  sorry

end projectile_highest_point_l162_162589


namespace inequality_C_incorrect_l162_162926

theorem inequality_C_incorrect (x : ℝ) (h : x ≠ 0) : ¬(e^x < 1 + x) → (e^1 ≥ 1 + 1) :=
by {
  sorry
}

end inequality_C_incorrect_l162_162926


namespace find_f2_l162_162583

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
sorry

end find_f2_l162_162583


namespace cricket_players_count_l162_162200

theorem cricket_players_count (Hockey Football Softball Total Cricket : ℕ) 
    (hHockey : Hockey = 12)
    (hFootball : Football = 18)
    (hSoftball : Softball = 13)
    (hTotal : Total = 59)
    (hTotalCalculation : Total = Hockey + Football + Softball + Cricket) : 
    Cricket = 16 := by
  sorry

end cricket_players_count_l162_162200


namespace a_plus_b_eq_neg2_l162_162619

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

variable (a b : ℝ)

axiom h1 : f a = 1
axiom h2 : f b = 19

theorem a_plus_b_eq_neg2 : a + b = -2 :=
sorry

end a_plus_b_eq_neg2_l162_162619


namespace ratio_girls_total_members_l162_162497

theorem ratio_girls_total_members {p_boy p_girl : ℚ} (h_prob_ratio : p_girl = (3/5) * p_boy) (h_total_prob : p_boy + p_girl = 1) :
  p_girl / (p_boy + p_girl) = 3 / 8 :=
by
  sorry

end ratio_girls_total_members_l162_162497


namespace abs_sub_eq_abs_sub_l162_162990

theorem abs_sub_eq_abs_sub (a b : ℚ) : |a - b| = |b - a| :=
sorry

end abs_sub_eq_abs_sub_l162_162990


namespace car_stops_at_three_seconds_l162_162608

theorem car_stops_at_three_seconds (t : ℝ) (h : -3 * t^2 + 18 * t = 0) : t = 3 := 
sorry

end car_stops_at_three_seconds_l162_162608


namespace smallest_single_discount_l162_162010

noncomputable def discount1 : ℝ := (1 - 0.20) * (1 - 0.20)
noncomputable def discount2 : ℝ := (1 - 0.10) * (1 - 0.15)
noncomputable def discount3 : ℝ := (1 - 0.08) * (1 - 0.08) * (1 - 0.08)

theorem smallest_single_discount : ∃ n : ℕ, (1 - n / 100) < discount1 ∧ (1 - n / 100) < discount2 ∧ (1 - n / 100) < discount3 ∧ n = 37 := sorry

end smallest_single_discount_l162_162010


namespace divisible_by_condition_a_l162_162322

theorem divisible_by_condition_a (a b c k : ℤ) 
  (h : ∃ k : ℤ, a - b * c = (10 * c + 1) * k) : 
  ∃ k : ℤ, 10 * a + b = (10 * c + 1) * k :=
by
  sorry

end divisible_by_condition_a_l162_162322


namespace people_left_first_hour_l162_162483

theorem people_left_first_hour 
  (X : ℕ)
  (h1 : X ≥ 0)
  (h2 : 94 - X + 18 - 9 = 76) :
  X = 27 := 
sorry

end people_left_first_hour_l162_162483


namespace valid_marble_arrangements_eq_48_l162_162177

def ZaraMarbleArrangements (n : ℕ) : ℕ := sorry

theorem valid_marble_arrangements_eq_48 : ZaraMarbleArrangements 5 = 48 := sorry

end valid_marble_arrangements_eq_48_l162_162177


namespace cost_of_pencils_and_notebooks_l162_162309

variable (p n : ℝ)

theorem cost_of_pencils_and_notebooks 
  (h1 : 9 * p + 10 * n = 5.06) 
  (h2 : 6 * p + 4 * n = 2.42) :
  20 * p + 14 * n = 8.31 :=
by
  sorry

end cost_of_pencils_and_notebooks_l162_162309


namespace stamps_per_light_envelope_l162_162107

theorem stamps_per_light_envelope 
  (stamps_heavy : ℕ) (stamps_light : ℕ → ℕ) (total_light : ℕ) (total_stamps_light : ℕ)
  (total_envelopes : ℕ) :
  (∀ n, n > 5 → stamps_heavy = 5) →
  (∀ n, n <= 5 → stamps_light n = total_stamps_light / total_light) →
  total_light = 6 →
  total_stamps_light = 52 →
  total_envelopes = 14 →
  stamps_light 5 = 9 :=
by
  sorry

end stamps_per_light_envelope_l162_162107


namespace five_hash_neg_one_l162_162781

def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

theorem five_hash_neg_one : hash 5 (-1) = 0 :=
by
  sorry

end five_hash_neg_one_l162_162781


namespace katie_earnings_l162_162898

-- Define the constants for the problem
def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

-- Define the total earnings calculation
def total_necklaces : Nat := bead_necklaces + gemstone_necklaces
def total_earnings : Nat := total_necklaces * cost_per_necklace

-- Statement of the proof problem
theorem katie_earnings : total_earnings = 21 := by
  sorry

end katie_earnings_l162_162898


namespace second_friend_shells_l162_162199

theorem second_friend_shells (initial_shells : ℕ) (first_friend_shells : ℕ) (total_shells : ℕ) (second_friend_shells : ℕ) :
  initial_shells = 5 → first_friend_shells = 15 → total_shells = 37 → initial_shells + first_friend_shells + second_friend_shells = total_shells → second_friend_shells = 17 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end second_friend_shells_l162_162199


namespace mary_initial_triangles_l162_162425

theorem mary_initial_triangles (s t : ℕ) (h1 : s + t = 10) (h2 : 4 * s + 3 * t = 36) : t = 4 :=
by
  sorry

end mary_initial_triangles_l162_162425


namespace height_of_wall_l162_162927

theorem height_of_wall (length_brick width_brick height_brick : ℝ)
                        (length_wall width_wall number_of_bricks : ℝ)
                        (volume_of_bricks : ℝ) :
  (length_brick, width_brick, height_brick) = (125, 11.25, 6) →
  (length_wall, width_wall) = (800, 22.5) →
  number_of_bricks = 1280 →
  volume_of_bricks = length_brick * width_brick * height_brick * number_of_bricks →
  volume_of_bricks = length_wall * width_wall * 600 := 
by
  intros h1 h2 h3 h4
  -- proof skipped
  sorry

end height_of_wall_l162_162927


namespace bun_eating_problem_l162_162975

theorem bun_eating_problem
  (n k : ℕ)
  (H1 : 5 * n / 10 + 3 * k / 10 = 180) -- This corresponds to the condition that Zhenya eats 5 buns in 10 minutes, and Sasha eats 3 buns in 10 minutes, for a total of 180 minutes.
  (H2 : n + k = 70) -- This corresponds to the total number of buns eaten.
  : n = 40 ∧ k = 30 :=
by
  sorry

end bun_eating_problem_l162_162975


namespace cubic_equation_real_root_l162_162410

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x + 25 = 0 := 
sorry

end cubic_equation_real_root_l162_162410


namespace problem_solution_l162_162196

noncomputable def arithmetic_sequences
    (a : ℕ → ℚ) (b : ℕ → ℚ)
    (Sn : ℕ → ℚ) (Tn : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, Sn n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) ∧
  (∀ n : ℕ, Tn n = n / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) ∧
  (∀ n : ℕ, Sn n / Tn n = (2 * n - 3) / (4 * n - 3))

theorem problem_solution
    (a : ℕ → ℚ) (b : ℕ → ℚ) (Sn : ℕ → ℚ) (Tn : ℕ → ℚ)
    (h_arith : arithmetic_sequences a b Sn Tn) :
    (a 9 / (b 5 + b 7)) + (a 3 / (b 8 + b 4)) = 19 / 41 :=
by
  sorry

end problem_solution_l162_162196


namespace zou_mei_competition_l162_162230

theorem zou_mei_competition (n : ℕ) (h1 : 271 = n^2 + 15) (h2 : n^2 + 33 = (n + 1)^2) : 
  ∃ n, 271 = n^2 + 15 ∧ n^2 + 33 = (n + 1)^2 :=
by
  existsi n
  exact ⟨h1, h2⟩

end zou_mei_competition_l162_162230


namespace radius_condition_l162_162935

def X (x y : ℝ) : ℝ := 12 * x
def Y (x y : ℝ) : ℝ := 5 * y

def satisfies_condition (x y : ℝ) : Prop :=
  Real.sin (X x y + Y x y) = Real.sin (X x y) + Real.sin (Y x y)

def no_intersection (R : ℝ) : Prop :=
  ∀ (x y : ℝ), satisfies_condition x y → dist (0, 0) (x, y) ≥ R

theorem radius_condition :
  ∀ R : ℝ, (0 < R ∧ R < Real.pi / 15) →
  no_intersection R :=
sorry

end radius_condition_l162_162935


namespace greatest_value_x_plus_y_l162_162413

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 ∨ x + y = -6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_x_plus_y_l162_162413


namespace probability_two_green_apples_l162_162169

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_green_apples :
  ∀ (total_apples green_apples choose_apples : ℕ),
    total_apples = 7 →
    green_apples = 3 →
    choose_apples = 2 →
    (binom green_apples choose_apples : ℝ) / binom total_apples choose_apples = 1 / 7 :=
by
  intro total_apples green_apples choose_apples
  intro h_total h_green h_choose
  rw [h_total, h_green, h_choose]
  -- The proof would go here
  sorry

end probability_two_green_apples_l162_162169


namespace y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l162_162694

def line_equation (m x1 y1 x y : ℝ) : Prop :=
  y - y1 = m * (x - x1)

theorem y_intercept_of_line_with_slope_3_and_x_intercept_7_0 :
  ∃ b : ℝ, line_equation 3 7 0 0 b ∧ b = -21 :=
by
  sorry

end y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l162_162694


namespace altitudes_sum_of_triangle_formed_by_line_and_axes_l162_162592

noncomputable def sum_of_altitudes (x y : ℝ) : ℝ :=
  let intercept_x := 6
  let intercept_y := 16
  let altitude_3 := 48 / Real.sqrt (8^2 + 3^2)
  intercept_x + intercept_y + altitude_3

theorem altitudes_sum_of_triangle_formed_by_line_and_axes :
  ∀ (x y : ℝ), (8 * x + 3 * y = 48) →
  sum_of_altitudes x y = 22 + 48 / Real.sqrt 73 :=
by
  sorry

end altitudes_sum_of_triangle_formed_by_line_and_axes_l162_162592


namespace fully_factor_expression_l162_162329

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end fully_factor_expression_l162_162329


namespace sum_of_first_4n_integers_l162_162564

theorem sum_of_first_4n_integers (n : ℕ) 
  (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 150) : 
  (4 * n * (4 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_4n_integers_l162_162564


namespace at_least_one_term_le_one_l162_162316

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end at_least_one_term_le_one_l162_162316


namespace lorraine_initial_brownies_l162_162278

theorem lorraine_initial_brownies (B : ℝ) 
(h1: (0.375 * B - 1 = 5)) : B = 16 := 
sorry

end lorraine_initial_brownies_l162_162278


namespace compute_expression_eq_162_l162_162636

theorem compute_expression_eq_162 : 
  3 * 3^4 - 9^35 / 9^33 = 162 := 
by 
  sorry

end compute_expression_eq_162_l162_162636


namespace exists_arithmetic_seq_perfect_powers_l162_162077

def is_perfect_power (x : ℕ) : Prop := ∃ (a k : ℕ), k > 1 ∧ x = a^k

theorem exists_arithmetic_seq_perfect_powers (n : ℕ) (hn : n > 1) :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → seq i = a + (i - 1) * d)
  ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_perfect_power (seq i)
  ∧ d ≠ 0 :=
sorry

end exists_arithmetic_seq_perfect_powers_l162_162077


namespace canonical_equations_of_line_l162_162452

-- Definitions for the normal vectors of the planes
def n1 : ℝ × ℝ × ℝ := (2, 3, -2)
def n2 : ℝ × ℝ × ℝ := (1, -3, 1)

-- Define the equations of the planes
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y - 2 * z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3 * y + z + 3 = 0

-- The canonical equations of the line of intersection
def canonical_eq (x y z : ℝ) : Prop := (z * (-4)) = (y * (-9)) ∧ (z * (-3)) = (x + 3) * (-9)

theorem canonical_equations_of_line :
  ∀ x y z : ℝ, (plane1 x y z) ∧ (plane2 x y z) → canonical_eq x y z :=
by
  sorry

end canonical_equations_of_line_l162_162452


namespace ratio_of_surface_areas_of_spheres_l162_162458

theorem ratio_of_surface_areas_of_spheres (V1 V2 S1 S2 : ℝ) 
(h : V1 / V2 = 8 / 27) 
(h1 : S1 = 4 * π * (V1^(2/3)) / (2 * π)^(2/3))
(h2 : S2 = 4 * π * (V2^(2/3)) / (3 * π)^(2/3)) :
S1 / S2 = 4 / 9 :=
sorry

end ratio_of_surface_areas_of_spheres_l162_162458


namespace neither_sufficient_nor_necessary_condition_l162_162466

theorem neither_sufficient_nor_necessary_condition (a b : ℝ) :
  ¬ ((a < 0 ∧ b < 0) → (a * b * (a - b) > 0)) ∧
  ¬ ((a * b * (a - b) > 0) → (a < 0 ∧ b < 0)) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l162_162466


namespace validate_option_B_l162_162582

theorem validate_option_B (a b : ℝ) : 
  (2 * a + 3 * a^2 ≠ 5 * a^3) ∧ 
  ((-a^3)^2 = a^6) ∧ 
  (¬ (-4 * a^3 * b / (2 * a) = -2 * a^2)) ∧ 
  ((5 * a * b)^2 ≠ 10 * a^2 * b^2) := 
by
  sorry

end validate_option_B_l162_162582


namespace range_of_m_l162_162217

noncomputable def f (x : ℝ) : ℝ := 1 + Real.sin (2 * x)

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + m

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x₀ ≥ g x₀ m) → m ≤ Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_m_l162_162217


namespace total_students_in_class_l162_162654

variable (K M Both Total : ℕ)

theorem total_students_in_class
  (hK : K = 38)
  (hM : M = 39)
  (hBoth : Both = 32)
  (hTotal : Total = K + M - Both) :
  Total = 45 := 
by
  rw [hK, hM, hBoth] at hTotal
  exact hTotal

end total_students_in_class_l162_162654


namespace evaluate_expression_l162_162030

theorem evaluate_expression : 
  let a := 45
  let b := 15
  (a + b)^2 - (a^2 + b^2 + 2 * a * 5) = 900 :=
by
  let a := 45
  let b := 15
  sorry

end evaluate_expression_l162_162030


namespace solve_equation_1_solve_equation_2_l162_162842

theorem solve_equation_1 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6 := sorry

theorem solve_equation_2 (x : ℝ) : (1 / 2) * (x - 1)^3 = -4 ↔ x = -1 := sorry

end solve_equation_1_solve_equation_2_l162_162842


namespace condition_holds_l162_162126

theorem condition_holds 
  (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) : 
  (a = c ∨ a = -c) ∨ (a^2 - c^2 + d^2 = b^2) :=
by
  sorry

end condition_holds_l162_162126


namespace correlation_relationships_l162_162679

-- Let's define the relationships as conditions
def volume_cube_edge_length (v e : ℝ) : Prop := v = e^3
def yield_fertilizer (yield fertilizer : ℝ) : Prop := True -- Assume linear correlation within a certain range
def height_age (height age : ℝ) : Prop := True -- Assume linear correlation within a certain age range
def expenses_income (expenses income : ℝ) : Prop := True -- Assume linear correlation
def electricity_consumption_price (consumption price unit_price : ℝ) : Prop := price = consumption * unit_price

-- We want to prove that the answers correspond correctly to the conditions:
theorem correlation_relationships :
  ∀ (v e yield fertilizer height age expenses income consumption price unit_price : ℝ),
  ¬ volume_cube_edge_length v e ∧ yield_fertilizer yield fertilizer ∧ height_age height age ∧ expenses_income expenses income ∧ ¬ electricity_consumption_price consumption price unit_price → 
  "D" = "②③④" :=
by
  intros
  sorry

end correlation_relationships_l162_162679


namespace derivative_at_one_l162_162055

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem derivative_at_one : deriv f 1 = 2 :=
by sorry

end derivative_at_one_l162_162055


namespace percentage_calculation_l162_162734

theorem percentage_calculation (percentage : ℝ) (h : percentage * 50 = 0.15) : percentage = 0.003 :=
by
  sorry

end percentage_calculation_l162_162734


namespace luke_games_l162_162891

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end luke_games_l162_162891


namespace percentage_profit_double_price_l162_162604

theorem percentage_profit_double_price (C S1 S2 : ℝ) (h1 : S1 = 1.5 * C) (h2 : S2 = 2 * S1) : 
  ((S2 - C) / C) * 100 = 200 := by
  sorry

end percentage_profit_double_price_l162_162604


namespace principal_amount_l162_162509

/-
  Given:
  - Simple Interest (SI) = Rs. 4016.25
  - Rate (R) = 0.08 (8% per annum)
  - Time (T) = 5 years
  
  We want to prove:
  Principal = Rs. 10040.625
-/

def SI : ℝ := 4016.25
def R : ℝ := 0.08
def T : ℕ := 5

theorem principal_amount :
  ∃ P : ℝ, SI = (P * R * T) / 100 ∧ P = 10040.625 :=
by
  sorry

end principal_amount_l162_162509


namespace max_remaining_grapes_l162_162460

theorem max_remaining_grapes (x : ℕ) : x % 7 ≤ 6 :=
  sorry

end max_remaining_grapes_l162_162460


namespace farey_neighbors_of_half_l162_162516

noncomputable def farey_neighbors (n : ℕ) : List (ℚ) :=
  if n % 2 = 1 then
    [ (n - 1 : ℚ) / (2 * n), (n + 1 : ℚ) / (2 * n) ]
  else
    [ (n - 2 : ℚ) / (2 * (n - 1)), n / (2 * (n - 1)) ]

theorem farey_neighbors_of_half (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℚ, a ∈ farey_neighbors n ∧ b ∈ farey_neighbors n ∧ 
    (n % 2 = 1 → a = (n - 1 : ℚ) / (2 * n) ∧ b = (n + 1 : ℚ) / (2 * n)) ∧
    (n % 2 = 0 → a = (n - 2 : ℚ) / (2 * (n - 1)) ∧ b = n / (2 * (n - 1))) :=
sorry

end farey_neighbors_of_half_l162_162516


namespace ellipse_equation_no_match_l162_162594

-- Definitions based on conditions in a)
def a : ℝ := 6
def c : ℝ := 1

-- Calculation for b² based on solution steps
def b_squared := a^2 - c^2

-- Standard forms of ellipse equations
def standard_ellipse_eq1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b_squared) = 1
def standard_ellipse_eq2 (x y : ℝ) : Prop := (y^2 / a^2) + (x^2 / b_squared) = 1

-- The proof problem statement
theorem ellipse_equation_no_match : 
  ∀ (x y : ℝ), ¬(standard_ellipse_eq1 x y) ∧ ¬(standard_ellipse_eq2 x y) := 
sorry

end ellipse_equation_no_match_l162_162594


namespace slope_of_line_through_points_l162_162134

theorem slope_of_line_through_points 
  (t : ℝ) 
  (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 12 * t + 6) 
  (h2 : 2 * x + 3 * y = 8 * t - 1) : 
  ∃ m b : ℝ, (∀ t : ℝ, y = m * x + b) ∧ m = 0 :=
by 
  sorry

end slope_of_line_through_points_l162_162134


namespace tangent_line_eq_l162_162433

theorem tangent_line_eq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_eq_l162_162433


namespace product_mod_32_l162_162284

def product_of_all_odd_primes_less_than_32 : ℕ :=
  3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  (product_of_all_odd_primes_less_than_32) % 32 = 9 :=
sorry

end product_mod_32_l162_162284


namespace estimate_total_observations_in_interval_l162_162768

def total_observations : ℕ := 1000
def sample_size : ℕ := 50
def frequency_in_sample : ℝ := 0.12

theorem estimate_total_observations_in_interval : 
  frequency_in_sample * (total_observations : ℝ) = 120 :=
by
  -- conditions defined above
  -- use given frequency to estimate the total observations in the interval
  -- actual proof omitted
  sorry

end estimate_total_observations_in_interval_l162_162768


namespace minimize_material_use_l162_162947

theorem minimize_material_use 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (total_area : x * y + (x^2 / 4) = 8) :
  (abs (x - 2.343) ≤ 0.001) ∧ (abs (y - 2.828) ≤ 0.001) :=
sorry

end minimize_material_use_l162_162947


namespace exists_person_who_knows_everyone_l162_162645

variable {Person : Type}
variable (knows : Person → Person → Prop)
variable (n : ℕ)

-- Condition: In a company of 2n + 1 people, for any n people, there is another person different from them who knows each of them.
axiom knows_condition : ∀ (company : Finset Person) (h : company.card = 2 * n + 1), 
  (∀ (subset : Finset Person) (hs : subset.card = n), ∃ (p : Person), p ∉ subset ∧ ∀ q ∈ subset, knows p q)

-- Statement to be proven:
theorem exists_person_who_knows_everyone (company : Finset Person) (hcompany : company.card = 2 * n + 1) :
  ∃ p, ∀ q ∈ company, knows p q :=
sorry

end exists_person_who_knows_everyone_l162_162645


namespace haleigh_needs_46_leggings_l162_162052

-- Define the number of each type of animal
def num_dogs : ℕ := 4
def num_cats : ℕ := 3
def num_spiders : ℕ := 2
def num_parrot : ℕ := 1

-- Define the number of legs each type of animal has
def legs_dog : ℕ := 4
def legs_cat : ℕ := 4
def legs_spider : ℕ := 8
def legs_parrot : ℕ := 2

-- Define the total number of legs function
def total_leggings (d c s p : ℕ) (ld lc ls lp : ℕ) : ℕ :=
  d * ld + c * lc + s * ls + p * lp

-- The statement to be proven
theorem haleigh_needs_46_leggings : total_leggings num_dogs num_cats num_spiders num_parrot legs_dog legs_cat legs_spider legs_parrot = 46 := by
  sorry

end haleigh_needs_46_leggings_l162_162052


namespace Sarah_pool_depth_l162_162175

theorem Sarah_pool_depth (S J : ℝ) (h1 : J = 2 * S + 5) (h2 : J = 15) : S = 5 := by
  sorry

end Sarah_pool_depth_l162_162175


namespace exists_two_positive_integers_dividing_3003_l162_162245

theorem exists_two_positive_integers_dividing_3003 : 
  ∃ (m1 m2 : ℕ), m1 > 0 ∧ m2 > 0 ∧ m1 ≠ m2 ∧ (3003 % (m1^2 + 2) = 0) ∧ (3003 % (m2^2 + 2) = 0) :=
by
  sorry

end exists_two_positive_integers_dividing_3003_l162_162245


namespace oranges_per_box_l162_162946

theorem oranges_per_box
  (total_oranges : ℕ)
  (boxes : ℕ)
  (h1 : total_oranges = 35)
  (h2 : boxes = 7) :
  total_oranges / boxes = 5 := by
  sorry

end oranges_per_box_l162_162946


namespace find_integer_k_l162_162246

noncomputable def P : ℤ → ℤ := sorry

theorem find_integer_k :
  P 1 = 2019 ∧ P 2019 = 1 ∧ ∃ k : ℤ, P k = k ∧ k = 1010 :=
by
  sorry

end find_integer_k_l162_162246


namespace find_x_l162_162044

variable (x : ℕ)  -- we'll use natural numbers to avoid negative values

-- initial number of children
def initial_children : ℕ := 21

-- number of children who got off
def got_off : ℕ := 10

-- total children after some got on
def total_children : ℕ := 16

-- statement to prove x is the number of children who got on the bus
theorem find_x : initial_children - got_off + x = total_children → x = 5 :=
by
  sorry

end find_x_l162_162044


namespace ratio_of_heights_l162_162876

def min_height := 140
def brother_height := 180
def grow_needed := 20

def mary_height := min_height - grow_needed
def height_ratio := mary_height / brother_height

theorem ratio_of_heights : height_ratio = (2 / 3) := 
  sorry

end ratio_of_heights_l162_162876


namespace shifted_parabola_relationship_l162_162340

-- Step a) and conditions
def original_function (x : ℝ) : ℝ := -2 * x ^ 2 + 4

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x => f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x => f x + b

-- Step c) encoding the proof problem
theorem shifted_parabola_relationship :
  (shift_up (shift_left original_function 2) 3 = fun x => -2 * (x + 2) ^ 2 + 7) :=
by
  sorry

end shifted_parabola_relationship_l162_162340


namespace remainder_product_l162_162991

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l162_162991


namespace find_side_PR_of_PQR_l162_162426

open Real

noncomputable def triangle_PQR (PQ PM PH PR : ℝ) : Prop :=
  let HQ := sqrt (PQ^2 - PH^2)
  let MH := sqrt (PM^2 - PH^2)
  let MQ := MH - HQ
  let RH := HQ + 2 * MQ
  PR = sqrt (PH^2 + RH^2)

theorem find_side_PR_of_PQR (PQ PM PH : ℝ) (h_PQ : PQ = 3) (h_PM : PM = sqrt 14) (h_PH : PH = sqrt 5) (h_angle : ∀ QPR PRQ : ℝ, QPR + PRQ < 90) : 
  triangle_PQR PQ PM PH (sqrt 21) :=
by
  rw [h_PQ, h_PM, h_PH]
  exact sorry

end find_side_PR_of_PQR_l162_162426


namespace f_at_11_l162_162515

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_at_11 : f 11 = 149 := sorry

end f_at_11_l162_162515


namespace inequality_proof_l162_162767

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1) : 
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_proof_l162_162767


namespace problem1_l162_162917

variable (α : ℝ)

theorem problem1 (h : Real.tan α = -3/4) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3/4 := 
sorry

end problem1_l162_162917


namespace sufficient_but_not_necessary_condition_l162_162669

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (2 * x^2 + x - 1 ≥ 0) → (x ≥ 1/2) ∨ (x ≤ -1) :=
by
  -- The given inequality and condition imply this result.
  sorry

end sufficient_but_not_necessary_condition_l162_162669


namespace car_travel_distance_l162_162994

theorem car_travel_distance (distance : ℝ) 
  (speed1 : ℝ := 80) 
  (speed2 : ℝ := 76.59574468085106) 
  (time_difference : ℝ := 2 / 3600) : 
  (distance / speed2 = distance / speed1 + time_difference) → 
  distance = 0.998177 :=
by
  -- assuming the above equation holds, we need to conclude the distance
  sorry

end car_travel_distance_l162_162994


namespace unique_solution_k_values_l162_162264

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, k * x ^ 2 - 3 * x + 2 = 0) ↔ (k = 0 ∨ k = 9 / 8) :=
by
  sorry

end unique_solution_k_values_l162_162264


namespace cos_angle_value_l162_162105

noncomputable def cos_angle := Real.cos (19 * Real.pi / 4)

theorem cos_angle_value : cos_angle = -Real.sqrt 2 / 2 := by
  sorry

end cos_angle_value_l162_162105


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l162_162907

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l162_162907


namespace percentage_solution_l162_162514

noncomputable def percentage_of_difference (P : ℚ) (x y : ℚ) : Prop :=
  (P / 100) * (x - y) = (14 / 100) * (x + y)

theorem percentage_solution (x y : ℚ) (h1 : y = 0.17647058823529413 * x)
  (h2 : percentage_of_difference P x y) : 
  P = 20 := 
by
  sorry

end percentage_solution_l162_162514


namespace zoo_visitors_l162_162782

theorem zoo_visitors (visitors_friday : ℕ) 
  (h1 : 3 * visitors_friday = 3750) :
  visitors_friday = 1250 := 
sorry

end zoo_visitors_l162_162782


namespace total_voters_in_districts_l162_162347

theorem total_voters_in_districts :
  let D1 := 322
  let D2 := (D1 / 2) - 19
  let D3 := 2 * D1
  let D4 := D2 + 45
  let D5 := (3 * D3) - 150
  let D6 := (D1 + D4) + (1 / 5) * (D1 + D4)
  let D7 := D2 + (D5 - D2) / 2
  D1 + D2 + D3 + D4 + D5 + D6 + D7 = 4650 := 
by
  sorry

end total_voters_in_districts_l162_162347


namespace oranges_to_apples_equivalence_l162_162289

theorem oranges_to_apples_equivalence :
  (forall (o l a : ℝ), 4 * o = 3 * l ∧ 5 * l = 7 * a -> 20 * o = 21 * a) :=
by
  intro o l a
  intro h
  sorry

end oranges_to_apples_equivalence_l162_162289


namespace cannot_divide_1980_into_four_groups_l162_162854

theorem cannot_divide_1980_into_four_groups :
  ¬∃ (S₁ S₂ S₃ S₄ : ℕ),
    S₂ = S₁ + 10 ∧
    S₃ = S₂ + 10 ∧
    S₄ = S₃ + 10 ∧
    (1 + 1980) * 1980 / 2 = S₁ + S₂ + S₃ + S₄ := 
sorry

end cannot_divide_1980_into_four_groups_l162_162854


namespace numerical_value_expression_l162_162748

theorem numerical_value_expression (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (ab + 1)) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (ab + 1) = 2 := 
by 
  -- Proof outline provided in the solution section, but actual proof is omitted
  sorry

end numerical_value_expression_l162_162748


namespace totalGamesPlayed_l162_162297

def numPlayers : ℕ := 30

def numGames (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem totalGamesPlayed :
  numGames numPlayers = 435 :=
by
  sorry

end totalGamesPlayed_l162_162297


namespace largest_n_satisfying_expression_l162_162049

theorem largest_n_satisfying_expression :
  ∃ n < 100000, (n - 3)^5 - n^2 + 10 * n - 30 ≡ 0 [MOD 3] ∧ 
  (∀ m, m < 100000 → (m - 3)^5 - m^2 + 10 * m - 30 ≡ 0 [MOD 3] → m ≤ 99998) := sorry

end largest_n_satisfying_expression_l162_162049


namespace trig_identity_l162_162106

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) : 
  ∃ (res : ℝ), res = 10 / 7 ∧ res = Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) := by
  sorry

end trig_identity_l162_162106


namespace find_m_l162_162805

open Set

def A (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 2}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem find_m (m : ℝ) :
  (A m ∩ B = ∅ ∧ A m ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end find_m_l162_162805


namespace min_expression_value_l162_162353

theorem min_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (z : ℝ) (h3 : x^2 + y^2 = z) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) = -2040200 :=
  sorry

end min_expression_value_l162_162353


namespace a_range_l162_162511

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 4) :=
by
  sorry

end a_range_l162_162511


namespace jordan_trapezoid_height_l162_162522

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end jordan_trapezoid_height_l162_162522


namespace intersection_of_A_and_B_l162_162206

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l162_162206


namespace xyz_inequality_l162_162453

-- Definitions for the conditions and the statement of the problem
theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h_ineq : x * y * z ≥ x * y + y * z + z * x) : 
  x * y * z ≥ 3 * (x + y + z) :=
by
  sorry

end xyz_inequality_l162_162453


namespace platform_length_is_260_meters_l162_162372

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def time_to_cross_platform_s : ℝ := 30
noncomputable def time_to_cross_man_s : ℝ := 17

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def length_of_train_m : ℝ := train_speed_mps * time_to_cross_man_s
noncomputable def total_distance_cross_platform_m : ℝ := train_speed_mps * time_to_cross_platform_s
noncomputable def length_of_platform_m : ℝ := total_distance_cross_platform_m - length_of_train_m

theorem platform_length_is_260_meters :
  length_of_platform_m = 260 := by
  sorry

end platform_length_is_260_meters_l162_162372


namespace joan_kittens_remaining_l162_162112

def original_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_kittens_remaining : original_kittens - kittens_given_away = 6 := by
  sorry

end joan_kittens_remaining_l162_162112


namespace increasing_intervals_decreasing_interval_l162_162720

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem increasing_intervals : 
  (∀ x, x < -1/3 → deriv f x > 0) ∧ 
  (∀ x, x > 1 → deriv f x > 0) :=
sorry

theorem decreasing_interval : 
  ∀ x, -1/3 < x ∧ x < 1 → deriv f x < 0 :=
sorry

end increasing_intervals_decreasing_interval_l162_162720


namespace percentage_problem_l162_162624

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by 
  sorry

end percentage_problem_l162_162624


namespace number_of_foons_correct_l162_162829

-- Define the conditions
def area : ℝ := 5  -- Area in cm^2
def thickness : ℝ := 0.5  -- Thickness in cm
def total_volume : ℝ := 50  -- Total volume in cm^3

-- Define the proof problem
theorem number_of_foons_correct :
  (total_volume / (area * thickness) = 20) :=
by
  -- The necessary computation would go here, but for now we'll use sorry to indicate the outcome
  sorry

end number_of_foons_correct_l162_162829


namespace special_divisors_count_of_20_30_l162_162103

def prime_number (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def number_of_divisors (a : ℕ) (α β : ℕ) : ℕ := (α + 1) * (β + 1)

def count_special_divisors (m n : ℕ) : ℕ :=
  let total_divisors_m := (m + 1) * (n + 1)
  let total_divisors_n := (n + 1) * (n / 2 + 1)
  (total_divisors_m - 1) / 2 - total_divisors_n + 1

theorem special_divisors_count_of_20_30 (d_20_30 d_20_15 : ℕ) :
  let α := 60
  let β := 30
  let γ := 30
  let δ := 15
  prime_number 2 ∧ prime_number 5 ∧
  count_special_divisors α β = 1891 ∧
  count_special_divisors γ δ = 496 →
  d_20_30 = 2 * 1891 / 2 ∧
  d_20_15 = 2 * 496 →
  count_special_divisors 60 30 - count_special_divisors 30 15 + 1 = 450
:= by
  sorry

end special_divisors_count_of_20_30_l162_162103


namespace ratio_proof_l162_162145

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 := by
  sorry

end ratio_proof_l162_162145


namespace set_union_covers_real_line_l162_162104

open Set

def M := {x : ℝ | x < 0 ∨ 2 < x}
def N := {x : ℝ | -Real.sqrt 5 < x ∧ x < Real.sqrt 5}

theorem set_union_covers_real_line : M ∪ N = univ := sorry

end set_union_covers_real_line_l162_162104


namespace max_value_a_l162_162742

-- Define the variables and the constraint on the circle
def circular_arrangement_condition (x: ℕ → ℕ) : Prop :=
  ∀ i: ℕ, 1 ≤ x i ∧ x i ≤ 10 ∧ x i ≠ x (i + 1)

-- Define the existence of three consecutive numbers summing to at least 18
def three_consecutive_sum_ge_18 (x: ℕ → ℕ) : Prop :=
  ∃ i: ℕ, x i + x (i + 1) + x (i + 2) ≥ 18

-- The main theorem we aim to prove
theorem max_value_a : ∀ (x: ℕ → ℕ), circular_arrangement_condition x → three_consecutive_sum_ge_18 x :=
  by sorry

end max_value_a_l162_162742


namespace sum_contains_even_digit_l162_162701

-- Define the five-digit integer and its reversed form
def reversed_digits (n : ℕ) : ℕ := 
  let a := n % 10
  let b := (n / 10) % 10
  let c := (n / 100) % 10
  let d := (n / 1000) % 10
  let e := (n / 10000) % 10
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem sum_contains_even_digit (n m : ℕ) (h1 : n >= 10000) (h2 : n < 100000) (h3 : m = reversed_digits n) : 
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ (n + m) % 10 = d ∨ (n + m) / 10 % 10 = d ∨ (n + m) / 100 % 10 = d ∨ (n + m) / 1000 % 10 = d ∨ (n + m) / 10000 % 10 = d := 
sorry

end sum_contains_even_digit_l162_162701


namespace notebook_pen_ratio_l162_162294

theorem notebook_pen_ratio (pen_cost notebook_total_cost : ℝ) (num_notebooks : ℕ)
  (h1 : pen_cost = 1.50) (h2 : notebook_total_cost = 18) (h3 : num_notebooks = 4) :
  (notebook_total_cost / num_notebooks) / pen_cost = 3 :=
by
  -- The steps to prove this would go here
  sorry

end notebook_pen_ratio_l162_162294


namespace iron_aluminum_weight_difference_l162_162300

theorem iron_aluminum_weight_difference :
  let iron_weight := 11.17
  let aluminum_weight := 0.83
  iron_weight - aluminum_weight = 10.34 :=
by
  sorry

end iron_aluminum_weight_difference_l162_162300


namespace gcd_455_299_eq_13_l162_162034

theorem gcd_455_299_eq_13 : Nat.gcd 455 299 = 13 := by
  sorry

end gcd_455_299_eq_13_l162_162034


namespace italian_dressing_mixture_l162_162165

/-- A chef is using a mixture of two brands of Italian dressing. 
  The first brand contains 8% vinegar, and the second brand contains 13% vinegar.
  The chef wants to make 320 milliliters of a dressing that is 11% vinegar.
  This statement proves the amounts required for each brand of dressing. -/

theorem italian_dressing_mixture
  (x y : ℝ)
  (hx : x + y = 320)
  (hv : 0.08 * x + 0.13 * y = 0.11 * 320) :
  x = 128 ∧ y = 192 :=
sorry

end italian_dressing_mixture_l162_162165


namespace minimum_groups_l162_162291

theorem minimum_groups (total_players : ℕ) (max_per_group : ℕ)
  (h_total : total_players = 30)
  (h_max : max_per_group = 12) :
  ∃ x y, y ∣ total_players ∧ y ≤ max_per_group ∧ total_players / y = x ∧ x = 3 :=
by {
  sorry
}

end minimum_groups_l162_162291


namespace variance_transformation_l162_162197

theorem variance_transformation (a_1 a_2 a_3 : ℝ) (h : (1 / 3) * ((a_1 - ((a_1 + a_2 + a_3) / 3))^2 + (a_2 - ((a_1 + a_2 + a_3) / 3))^2 + (a_3 - ((a_1 + a_2 + a_3) / 3))^2) = 1) :
  (1 / 3) * ((3 * a_1 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_2 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_3 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2) = 9 := by 
  sorry

end variance_transformation_l162_162197


namespace solve_system_equation_152_l162_162302

theorem solve_system_equation_152 (x y z a b c : ℝ)
  (h1 : x * y - 2 * y - 3 * x = 0)
  (h2 : y * z - 3 * z - 5 * y = 0)
  (h3 : x * z - 5 * x - 2 * z = 0)
  (h4 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h5 : x = a)
  (h6 : y = b)
  (h7 : z = c) :
  a^2 + b^2 + c^2 = 152 := by
  sorry

end solve_system_equation_152_l162_162302


namespace original_price_of_shoes_l162_162703

theorem original_price_of_shoes (P : ℝ) (h : 0.08 * P = 16) : P = 200 :=
sorry

end original_price_of_shoes_l162_162703


namespace time_ratio_l162_162812

theorem time_ratio (distance : ℝ) (initial_time : ℝ) (new_speed : ℝ) :
  distance = 600 → initial_time = 5 → new_speed = 80 → (distance / new_speed) / initial_time = 1.5 :=
by
  intros hdist htime hspeed
  sorry

end time_ratio_l162_162812


namespace number_of_sides_l162_162474

theorem number_of_sides (n : ℕ) : 
  (2 / 9) * (n - 2) * 180 = 360 → n = 11 := 
by
  intro h
  sorry

end number_of_sides_l162_162474


namespace percentage_profits_to_revenues_l162_162635

theorem percentage_profits_to_revenues (R P : ℝ) 
  (h1 : R > 0) 
  (h2 : P > 0)
  (h3 : 0.12 * R = 1.2 * P) 
  : P / R = 0.1 :=
by
  sorry

end percentage_profits_to_revenues_l162_162635


namespace rhombus_area_l162_162262

-- Define the given conditions: diagonals and side length
def d1 : ℕ := 40
def d2 : ℕ := 18
def s : ℕ := 25

-- Prove that the area of the rhombus is 360 square units given the conditions
theorem rhombus_area :
  (d1 * d2) / 2 = 360 :=
by
  sorry

end rhombus_area_l162_162262


namespace evaluate_expression_l162_162569

theorem evaluate_expression : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end evaluate_expression_l162_162569


namespace number_of_students_like_basketball_but_not_table_tennis_l162_162656

-- Given definitions
def total_students : Nat := 40
def students_like_basketball : Nat := 24
def students_like_table_tennis : Nat := 16
def students_dislike_both : Nat := 6

-- Proposition to prove
theorem number_of_students_like_basketball_but_not_table_tennis : 
  students_like_basketball - (students_like_basketball + students_like_table_tennis - (total_students - students_dislike_both)) = 18 := 
by
  sorry

end number_of_students_like_basketball_but_not_table_tennis_l162_162656


namespace solve_arcsin_sin_l162_162670

theorem solve_arcsin_sin (x : ℝ) (h : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.arcsin (Real.sin (2 * x)) = x ↔ x = 0 ∨ x = Real.pi / 3 ∨ x = -Real.pi / 3 :=
by
  sorry

end solve_arcsin_sin_l162_162670


namespace part1_part2_l162_162102

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def set_B : Set ℝ := {x : ℝ | x < -1 ∨ x > 1}

theorem part1 (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a > 3) :=
by sorry

theorem part2 (a : ℝ) : (set_A a ∪ set_B = Set.univ) ↔ (-2 ≤ a ∧ a ≤ -1 / 2) :=
by sorry

end part1_part2_l162_162102


namespace find_f_2_l162_162746

theorem find_f_2 (f : ℝ → ℝ) (h₁ : f 1 = 0)
  (h₂ : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) :
  f 2 = 0 :=
sorry

end find_f_2_l162_162746


namespace maximum_xyz_l162_162719

-- Given conditions
variables {x y z : ℝ}

-- Lean 4 statement with the conditions
theorem maximum_xyz (h₁ : x * y + 2 * z = (x + z) * (y + z))
  (h₂ : x + y + 2 * z = 2)
  (h₃ : 0 < x) (h₄ : 0 < y) (h₅ : 0 < z) :
  xyz = 0 :=
sorry

end maximum_xyz_l162_162719


namespace angle_in_first_quadrant_l162_162790

theorem angle_in_first_quadrant (x : ℝ) (h1 : Real.tan x > 0) (h2 : Real.sin x + Real.cos x > 0) : 
  0 < Real.sin x ∧ 0 < Real.cos x := 
by 
  sorry

end angle_in_first_quadrant_l162_162790


namespace certain_number_approx_l162_162325

theorem certain_number_approx (x : ℝ) : 213 * 16 = 3408 → x * 2.13 = 0.3408 → x = 0.1600 :=
by
  intro h1 h2
  sorry

end certain_number_approx_l162_162325


namespace number_of_cars_in_train_l162_162754

theorem number_of_cars_in_train
  (constant_speed : Prop)
  (cars_in_12_seconds : ℕ)
  (time_to_clear : ℕ)
  (cars_per_second : ℕ → ℕ → ℚ)
  (total_time_seconds : ℕ) :
  cars_in_12_seconds = 8 →
  time_to_clear = 180 →
  cars_per_second cars_in_12_seconds 12 = 2 / 3 →
  total_time_seconds = 180 →
  cars_per_second cars_in_12_seconds 12 * total_time_seconds = 120 :=
by
  sorry

end number_of_cars_in_train_l162_162754


namespace smallest_number_divisible_by_set_l162_162697

theorem smallest_number_divisible_by_set : ∃ x : ℕ, (∀ d ∈ [12, 24, 36, 48, 56, 72, 84], (x - 24) % d = 0) ∧ x = 1032 := 
by {
  sorry
}

end smallest_number_divisible_by_set_l162_162697


namespace man_twice_son_age_in_years_l162_162650

theorem man_twice_son_age_in_years :
  ∀ (S M Y : ℕ),
  (M = S + 26) →
  (S = 24) →
  (M + Y = 2 * (S + Y)) →
  Y = 2 :=
by
  intros S M Y h1 h2 h3
  sorry

end man_twice_son_age_in_years_l162_162650


namespace teena_distance_behind_poe_l162_162822

theorem teena_distance_behind_poe (D : ℝ)
    (teena_speed : ℝ) (poe_speed : ℝ)
    (time_hours : ℝ) (teena_ahead : ℝ) :
    teena_speed = 55 
    → poe_speed = 40 
    → time_hours = 1.5 
    → teena_ahead = 15 
    → D + teena_ahead = (teena_speed - poe_speed) * time_hours 
    → D = 7.5 := 
by 
    intros 
    sorry

end teena_distance_behind_poe_l162_162822


namespace passes_through_fixed_point_l162_162022

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(x-2) - 3

theorem passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -2 :=
by
  sorry

end passes_through_fixed_point_l162_162022


namespace find_s_l_l162_162020

theorem find_s_l :
  ∃ s l : ℝ, ∀ t : ℝ, 
  (-8 + l * t, s + -6 * t) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p.snd = 3 / 4 * x + 2 ∧ p.fst = x} ∧ 
  (s = -4 ∧ l = -8) :=
by
  sorry

end find_s_l_l162_162020


namespace perfect_squares_l162_162826

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l162_162826


namespace total_revenue_is_405_l162_162833

-- Define the cost of rentals
def canoeCost : ℕ := 15
def kayakCost : ℕ := 18

-- Define terms for number of rentals
variables (C K : ℕ)

-- Conditions
axiom ratio_condition : 2 * C = 3 * K
axiom difference_condition : C = K + 5

-- Total revenue
def totalRevenue (C K : ℕ) : ℕ := (canoeCost * C) + (kayakCost * K)

-- Theorem statement
theorem total_revenue_is_405 (C K : ℕ) (H1 : 2 * C = 3 * K) (H2 : C = K + 5) : 
  totalRevenue C K = 405 := by
  sorry

end total_revenue_is_405_l162_162833


namespace three_segments_form_triangle_l162_162307

theorem three_segments_form_triangle
    (lengths : Fin 10 → ℕ)
    (h1 : lengths 0 = 1)
    (h2 : lengths 1 = 1)
    (h3 : lengths 9 = 50) :
    ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    lengths i + lengths j > lengths k ∧ 
    lengths i + lengths k > lengths j ∧ 
    lengths j + lengths k > lengths i := 
sorry

end three_segments_form_triangle_l162_162307


namespace magnitude_of_complex_l162_162627

open Complex

theorem magnitude_of_complex : abs (Complex.mk (3/4) (-5/6)) = Real.sqrt (181) / 12 :=
by
  sorry

end magnitude_of_complex_l162_162627


namespace radius_of_circle_l162_162953

theorem radius_of_circle
  (r : ℝ)
  (h1 : ∀ x : ℝ, (x^2 + r = x) → (x^2 - x + r = 0) → ((-1)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
sorry

end radius_of_circle_l162_162953


namespace value_of_g_neg3_l162_162980

def g (x : ℝ) : ℝ := x^3 - 2 * x

theorem value_of_g_neg3 : g (-3) = -21 := by
  sorry

end value_of_g_neg3_l162_162980


namespace inverse_function_property_l162_162721

noncomputable def f (a x : ℝ) : ℝ := (x - a) * |x|

theorem inverse_function_property (a : ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, f a (g x) = x) ↔ a = 0 :=
by sorry

end inverse_function_property_l162_162721


namespace evaluate_expression_l162_162091

theorem evaluate_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3 * x ^ 2 - 2 * x + 1) / ((x + 2) * (x - 3)) - (x ^ 2 - 5 * x + 6) / ((x + 2) * (x - 3)) =
  (2 * x ^ 2 + 3 * x - 5) / ((x + 2) * (x - 3)) :=
by
  sorry

end evaluate_expression_l162_162091


namespace commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l162_162084

def binary_star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

-- Statement (A): Commutativity
theorem commutative_star (x y : ℝ) : binary_star x y = binary_star y x := sorry

-- Statement (B): Distributivity (proving it's not distributive)
theorem not_distributive_star (x y z : ℝ) : ¬(binary_star x (y + z) = binary_star x y + binary_star x z) := sorry

-- Statement (C): Special case
theorem special_case_star (x : ℝ) : binary_star (x + 1) (x - 1) = binary_star x x - 1 := sorry

-- Statement (D): Identity element
theorem no_identity_star (x e : ℝ) : ¬(binary_star x e = x ∧ binary_star e x = x) := sorry

-- Statement (E): Associativity (proving it's not associative)
theorem not_associative_star (x y z : ℝ) : ¬(binary_star x (binary_star y z) = binary_star (binary_star x y) z) := sorry

end commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l162_162084


namespace union_A_B_complement_U_A_intersection_B_range_of_a_l162_162877

-- Define the sets A, B, C, and U
def setA (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8
def setB (x : ℝ) : Prop := 1 < x ∧ x < 6
def setC (a : ℝ) (x : ℝ) : Prop := x > a
def U (x : ℝ) : Prop := True  -- U being the universal set of all real numbers

-- Define complements and intersections
def complement (A : ℝ → Prop) (x : ℝ) : Prop := ¬ A x
def intersection (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x

-- Proof problems
theorem union_A_B : ∀ x, union setA setB x ↔ (1 < x ∧ x ≤ 8) :=
by 
  intros x
  sorry

theorem complement_U_A_intersection_B : ∀ x, intersection (complement setA) setB x ↔ (1 < x ∧ x < 2) :=
by 
  intros x
  sorry

theorem range_of_a (a : ℝ) : (∃ x, intersection setA (setC a) x) → a < 8 :=
by
  intros h
  sorry

end union_A_B_complement_U_A_intersection_B_range_of_a_l162_162877


namespace angle_measure_l162_162243

theorem angle_measure (x : ℝ) (h1 : x + 3 * x^2 + 10 = 90) : x = 5 :=
by
  sorry

end angle_measure_l162_162243


namespace scientific_notation_of_508_billion_yuan_l162_162184

-- Definition for a billion in the international system.
def billion : ℝ := 10^9

-- The amount of money given in the problem.
def amount_in_billion (n : ℝ) : ℝ := n * billion

-- The Lean theorem statement to prove.
theorem scientific_notation_of_508_billion_yuan :
  amount_in_billion 508 = 5.08 * 10^11 :=
by
  sorry

end scientific_notation_of_508_billion_yuan_l162_162184


namespace solve_inequality_l162_162540

theorem solve_inequality (x : ℝ) : (2 * x - 3) / (x + 2) ≤ 1 ↔ (-2 < x ∧ x ≤ 5) :=
  sorry

end solve_inequality_l162_162540


namespace extremum_values_l162_162988

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem extremum_values :
  (∀ x, f x ≤ 5) ∧ f (-1) = 5 ∧ (∀ x, f x ≥ -27) ∧ f 3 = -27 :=
by
  sorry

end extremum_values_l162_162988


namespace sufficient_but_not_necessary_condition_l162_162385

theorem sufficient_but_not_necessary_condition (a b : ℝ) : 
  (a ≥ 1 ∧ b ≥ 1) → (a + b ≥ 2) ∧ ¬((a + b ≥ 2) → (a ≥ 1 ∧ b ≥ 1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l162_162385


namespace hyperbola_standard_equation_l162_162366

def a : ℕ := 5
def c : ℕ := 7
def b_squared : ℕ := c * c - a * a

theorem hyperbola_standard_equation (a_eq : a = 5) (c_eq : c = 7) :
    (b_squared = 24) →
    ( ∀ x y : ℝ, x^2 / (a^2 : ℝ) - y^2 / (b_squared : ℝ) = 1 ∨ 
                   y^2 / (a^2 : ℝ) - x^2 / (b_squared : ℝ) = 1) :=
by
  sorry

end hyperbola_standard_equation_l162_162366


namespace cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l162_162320

noncomputable def p1 (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0 else 0.5

noncomputable def p2 (y : ℝ) : ℝ :=
  if y < 0 ∨ y > 2 then 0 else 0.5

noncomputable def F1 (x : ℝ) : ℝ :=
  if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1

noncomputable def F2 (y : ℝ) : ℝ :=
  if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1

noncomputable def p (x : ℝ) (y : ℝ) : ℝ :=
  if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25

noncomputable def F (x : ℝ) (y : ℝ) : ℝ :=
  if x ≤ -1 ∨ y ≤ 0 then 0
  else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y 
  else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
  else if x > 1 ∧ y ≤ 2 then 0.5 * y
  else 1

theorem cumulative_distribution_F1 (x : ℝ) : 
  F1 x = if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1 := by sorry

theorem cumulative_distribution_F2 (y : ℝ) : 
  F2 y = if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1 := by sorry

theorem joint_density (x : ℝ) (y : ℝ) : 
  p x y = if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25 := by sorry

theorem joint_cumulative_distribution (x : ℝ) (y : ℝ) : 
  F x y = if x ≤ -1 ∨ y ≤ 0 then 0
          else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y
          else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
          else if x > 1 ∧ y ≤ 2 then 0.5 * y
          else 1 := by sorry

end cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l162_162320


namespace disputed_piece_weight_l162_162901

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end disputed_piece_weight_l162_162901


namespace problem_value_expression_l162_162215

theorem problem_value_expression 
  (x y : ℝ)
  (h₁ : x + y = 4)
  (h₂ : x * y = -2) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := 
sorry

end problem_value_expression_l162_162215


namespace value_two_stddev_below_mean_l162_162816

def mean : ℝ := 16.2
def standard_deviation : ℝ := 2.3

theorem value_two_stddev_below_mean : mean - 2 * standard_deviation = 11.6 :=
by
  sorry

end value_two_stddev_below_mean_l162_162816


namespace sum_of_pills_in_larger_bottles_l162_162815

-- Definitions based on the conditions
def supplements := 5
def pills_in_small_bottles := 2 * 30
def pills_per_day := 5
def days_used := 14
def pills_remaining := 350
def total_pills_before := pills_remaining + (pills_per_day * days_used)
def total_pills_in_large_bottles := total_pills_before - pills_in_small_bottles

-- The theorem statement that needs to be proven
theorem sum_of_pills_in_larger_bottles : total_pills_in_large_bottles = 360 := 
by 
  -- Placeholder for the proof
  sorry

end sum_of_pills_in_larger_bottles_l162_162815


namespace inequality_inverse_l162_162996

theorem inequality_inverse (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a) < (1 / b) :=
by
  sorry

end inequality_inverse_l162_162996


namespace triangle_side_length_b_l162_162283

/-
In a triangle ABC with angles such that ∠C = 4∠A, and sides such that a = 35 and c = 64, prove that the length of side b is 140 * cos²(A).
-/
theorem triangle_side_length_b (A C : ℝ) (a c : ℝ) (hC : C = 4 * A) (ha : a = 35) (hc : c = 64) :
  ∃ (b : ℝ), b = 140 * (Real.cos A) ^ 2 :=
by
  sorry

end triangle_side_length_b_l162_162283


namespace max_discount_l162_162900

theorem max_discount (C : ℝ) (x : ℝ) (h1 : 1.8 * C = 360) (h2 : ∀ y, y ≥ 1.3 * C → 360 - x ≥ y) : x ≤ 100 :=
by
  have hC : C = 360 / 1.8 := by sorry
  have hMinPrice : 1.3 * C = 1.3 * (360 / 1.8) := by sorry
  have hDiscount : 360 - x ≥ 1.3 * (360 / 1.8) := by sorry
  sorry

end max_discount_l162_162900


namespace minimum_value_frac_abc_l162_162700

variable (a b c : ℝ)

theorem minimum_value_frac_abc
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + 2 * c = 2) :
  (a + b) / (a * b * c) ≥ 8 :=
sorry

end minimum_value_frac_abc_l162_162700


namespace position_of_z_l162_162923

theorem position_of_z (total_distance : ℕ) (total_steps : ℕ) (steps_taken : ℕ) (distance_covered : ℕ) (h1 : total_distance = 30) (h2 : total_steps = 6) (h3 : steps_taken = 4) (h4 : distance_covered = total_distance / total_steps) : 
  steps_taken * distance_covered = 20 :=
by
  sorry

end position_of_z_l162_162923


namespace div_c_a_l162_162240

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end div_c_a_l162_162240


namespace number_of_valid_sequences_l162_162804

-- Define the sequence property
def sequence_property (b : Fin 10 → Fin 10) : Prop :=
  ∀ i : Fin 10, 2 ≤ i → (∃ j : Fin 10, j < i ∧ (b j = b i + 1 ∨ b j = b i - 1 ∨ b j = b i + 2 ∨ b j = b i - 2))

-- Define the set of such sequences
def valid_sequences : Set (Fin 10 → Fin 10) := {b | sequence_property b}

-- Define the number of such sequences
def number_of_sequences : Fin 512 :=
  sorry -- Proof omitted for brevity

-- The final statement
theorem number_of_valid_sequences : number_of_sequences = 512 :=
  sorry  -- Skip proof

end number_of_valid_sequences_l162_162804


namespace heather_bicycled_distance_l162_162802

def speed : ℕ := 8
def time : ℕ := 5
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

theorem heather_bicycled_distance : distance speed time = 40 := by
  sorry

end heather_bicycled_distance_l162_162802


namespace find_n_of_geometric_sum_l162_162427

-- Define the first term and common ratio of the sequence
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3

-- Define the sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Mathematical statement to be proved
theorem find_n_of_geometric_sum (h : S_n 5 = 80 / 243) : ∃ n, S_n n = 80 / 243 ↔ n = 5 :=
by
  sorry

end find_n_of_geometric_sum_l162_162427


namespace find_missing_number_l162_162611

theorem find_missing_number
  (mean : ℝ)
  (n : ℕ)
  (nums : List ℝ)
  (total_sum : ℝ)
  (sum_known_numbers : ℝ)
  (missing_number : ℝ) :
  mean = 20 → 
  n = 8 →
  nums = [1, 22, 23, 24, 25, missing_number, 27, 2] →
  total_sum = mean * n →
  sum_known_numbers = 1 + 22 + 23 + 24 + 25 + 27 + 2 →
  missing_number = total_sum - sum_known_numbers :=
by
  intros
  sorry

end find_missing_number_l162_162611


namespace simplify_and_evaluate_expression_l162_162000

theorem simplify_and_evaluate_expression : 
  ∀ a : ℚ, a = -1/2 → (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := 
by
  intro a ha
  simp only [ha]
  sorry

end simplify_and_evaluate_expression_l162_162000


namespace triangle_BC_length_l162_162406

theorem triangle_BC_length (A B C X : Type) (AB AC BC BX CX : ℕ)
  (h1 : AB = 75)
  (h2 : AC = 85)
  (h3 : BC = BX + CX)
  (h4 : BX * (BX + CX) = 1600)
  (h5 : BX + CX = 80) :
  BC = 80 :=
by
  sorry

end triangle_BC_length_l162_162406


namespace max_min_value_f_l162_162337

theorem max_min_value_f (x m : ℝ) : ∃ m : ℝ, (∀ x : ℝ, x^2 - 2*m*x + 8*m + 4 ≥ -m^2 + 8*m + 4) ∧ (∀ n : ℝ, -n^2 + 8*n + 4 ≤ 20) :=
  sorry

end max_min_value_f_l162_162337


namespace max_non_intersecting_segments_l162_162116

theorem max_non_intersecting_segments (n m : ℕ) (hn: 1 < n) (hm: m ≥ 3): 
  ∃ L, L = 3 * n - m - 3 :=
by
  sorry

end max_non_intersecting_segments_l162_162116


namespace fraction_position_1991_1949_l162_162371

theorem fraction_position_1991_1949 :
  ∃ (row position : ℕ), 
    ∀ (i j : ℕ), 
      (∃ k : ℕ, k = i + j - 1 ∧ k = 3939) ∧
      (∃ p : ℕ, p = j ∧ p = 1949) → 
      row = 3939 ∧ position = 1949 := 
sorry

end fraction_position_1991_1949_l162_162371


namespace solve_for_z_l162_162919

variable (z : ℂ) (i : ℂ)

theorem solve_for_z
  (h1 : 3 - 2*i*z = 7 + 4*i*z)
  (h2 : i^2 = -1) :
  z = 2*i / 3 :=
by
  sorry

end solve_for_z_l162_162919


namespace equal_divided_value_l162_162219

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end equal_divided_value_l162_162219


namespace katie_more_games_l162_162997

noncomputable def katie_games : ℕ := 57 + 39
noncomputable def friends_games : ℕ := 34
noncomputable def games_difference : ℕ := katie_games - friends_games

theorem katie_more_games : games_difference = 62 :=
by
  -- Proof omitted
  sorry

end katie_more_games_l162_162997


namespace sqrt2_over_2_not_covered_by_rationals_l162_162295

noncomputable def rational_not_cover_sqrt2_over_2 : Prop :=
  ∀ (a b : ℤ) (h_ab : Int.gcd a b = 1) (h_b_pos : b > 0)
  (h_frac : (a : ℚ) / b ∈ Set.Ioo 0 1),
  abs ((Real.sqrt 2) / 2 - (a : ℚ) / b) > 1 / (4 * b^2)

-- Placeholder for the proof
theorem sqrt2_over_2_not_covered_by_rationals :
  rational_not_cover_sqrt2_over_2 := 
by sorry

end sqrt2_over_2_not_covered_by_rationals_l162_162295


namespace problem_geometric_sequence_l162_162132

variable {α : Type*} [LinearOrderedField α]

noncomputable def geom_sequence_5_8 (a : α) (h : a + 8 * a = 2) : α :=
  (a * 2^4 + a * 2^7)

theorem problem_geometric_sequence : ∃ (a : α), (a + 8 * a = 2) ∧ geom_sequence_5_8 a (sorry) = 32 := 
by sorry

end problem_geometric_sequence_l162_162132


namespace calculate_annual_rent_l162_162089

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l162_162089


namespace min_benches_l162_162496
-- Import the necessary library

-- Defining the problem in Lean statement
theorem min_benches (N : ℕ) :
  (∀ a c : ℕ, (8 * N = a) ∧ (12 * N = c) ∧ (a = c)) → N = 6 :=
by
  sorry

end min_benches_l162_162496


namespace major_axis_length_l162_162845

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def foci_1 : ℝ × ℝ := (3, 5)
def foci_2 : ℝ × ℝ := (23, 40)
def reflected_foci_1 : ℝ × ℝ := (-3, 5)

theorem major_axis_length :
  distance (reflected_foci_1.1) (reflected_foci_1.2) (foci_2.1) (foci_2.2) = Real.sqrt 1921 :=
sorry

end major_axis_length_l162_162845


namespace probability_ge_first_second_l162_162519

noncomputable def probability_ge_rolls : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_ge_first_second :
  probability_ge_rolls = 9 / 16 :=
by
  sorry

end probability_ge_first_second_l162_162519


namespace explicit_formula_of_odd_function_monotonicity_in_interval_l162_162738

-- Using Noncomputable because divisions are involved.
noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (p * x^2 + 2) / (q - 3 * x)

theorem explicit_formula_of_odd_function (p q : ℝ) 
  (h_odd : ∀ x : ℝ, f x p q = - f (-x) p q) 
  (h_value : f 2 p q = -5/3) : 
  f x 2 0 = -2/3 * (x + 1/x) :=
by sorry

theorem monotonicity_in_interval {x : ℝ} (h_domain : 0 < x ∧ x < 1) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 -> f x1 2 0 < f x2 2 0 :=
by sorry

end explicit_formula_of_odd_function_monotonicity_in_interval_l162_162738


namespace hyperbola_equation_l162_162813

theorem hyperbola_equation (h1 : ∀ x y : ℝ, (x = 0 ∧ y = 0)) 
                           (h2 : ∀ a : ℝ, (2 * a = 4)) 
                           (h3 : ∀ c : ℝ, (c = 3)) : 
  ∃ b : ℝ, (b^2 = 5) ∧ (∀ x y : ℝ, (y^2 / 4) - (x^2 / b^2) = 1) :=
sorry

end hyperbola_equation_l162_162813


namespace number_of_moles_of_H2O_l162_162744

def reaction_stoichiometry (n_NaOH m_Cl2 : ℕ) : ℕ :=
  1  -- Moles of H2O produced according to the balanced equation with the given reactants

theorem number_of_moles_of_H2O 
  (n_NaOH : ℕ) (m_Cl2 : ℕ) 
  (h_NaOH : n_NaOH = 2) 
  (h_Cl2 : m_Cl2 = 1) :
  reaction_stoichiometry n_NaOH m_Cl2 = 1 :=
by
  rw [h_NaOH, h_Cl2]
  -- Would typically follow with the proof using the conditions and stoichiometric relation
  sorry  -- Proof step omitted

end number_of_moles_of_H2O_l162_162744


namespace f_x_when_x_negative_l162_162638

-- Define the properties of the function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → f x = x * (1 + x)

-- The theorem we want to prove
theorem f_x_when_x_negative (f : ℝ → ℝ) 
  (h1: odd_function f)
  (h2: f_definition f) : 
  ∀ x, x < 0 → f x = -x * (1 - x) :=
by
  sorry

end f_x_when_x_negative_l162_162638


namespace abs_diff_squares_eq_300_l162_162546

theorem abs_diff_squares_eq_300 : 
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  |a^2 - b^2| = 300 := 
by
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  sorry

end abs_diff_squares_eq_300_l162_162546


namespace tom_fractions_l162_162755

theorem tom_fractions (packages : ℕ) (cars_per_package : ℕ) (cars_left : ℕ) (nephews : ℕ) :
  packages = 10 → 
  cars_per_package = 5 → 
  cars_left = 30 → 
  nephews = 2 → 
  ∃ fraction_given : ℚ, fraction_given = 1/5 :=
by
  intros
  sorry

end tom_fractions_l162_162755


namespace find_common_ratio_l162_162972

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variable {a : ℕ → ℝ} {q : ℝ}

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : a 2 + a 4 = 20)
  (h3 : a 3 + a 5 = 40) : q = 2 :=
by
  sorry

end find_common_ratio_l162_162972


namespace dark_lord_squads_l162_162967

def total_weight : ℕ := 1200
def orcs_per_squad : ℕ := 8
def capacity_per_orc : ℕ := 15
def squads_needed (w n c : ℕ) : ℕ := w / (n * c)

theorem dark_lord_squads :
  squads_needed total_weight orcs_per_squad capacity_per_orc = 10 :=
by sorry

end dark_lord_squads_l162_162967


namespace B_pow_150_eq_I_l162_162741

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end B_pow_150_eq_I_l162_162741


namespace correct_multiplication_l162_162338

theorem correct_multiplication (n : ℕ) (wrong_answer correct_answer : ℕ) 
    (h1 : wrong_answer = 559981)
    (h2 : correct_answer = 987 * n)
    (h3 : ∃ (x y : ℕ), correct_answer = 500000 + x + 901 + y ∧ x ≠ 98 ∧ y ≠ 98 ∧ (wrong_answer - correct_answer) % 10 = 0) :
    correct_answer = 559989 :=
by
  sorry

end correct_multiplication_l162_162338


namespace triangle_height_l162_162629

theorem triangle_height (b h : ℕ) (A : ℕ) (hA : A = 50) (hb : b = 10) :
  A = (1 / 2 : ℝ) * b * h → h = 10 := 
by
  sorry

end triangle_height_l162_162629


namespace smallest_multiple_of_84_with_6_and_7_l162_162942

variable (N : Nat)

def is_multiple_of_84 (N : Nat) : Prop :=
  N % 84 = 0

def consists_of_6_and_7 (N : Nat) : Prop :=
  ∀ d ∈ N.digits 10, d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  ∃ N, is_multiple_of_84 N ∧ consists_of_6_and_7 N ∧ ∀ M, is_multiple_of_84 M ∧ consists_of_6_and_7 M → N ≤ M := 
sorry

end smallest_multiple_of_84_with_6_and_7_l162_162942


namespace area_triangle_AMC_l162_162306

noncomputable def area_of_triangle_AMC (AB AD AM : ℝ) : ℝ :=
  if AB = 10 ∧ AD = 12 ∧ AM = 9 then
    (1 / 2) * AM * AB
  else 0

theorem area_triangle_AMC :
  ∀ (AB AD AM : ℝ), AB = 10 → AD = 12 → AM = 9 → area_of_triangle_AMC AB AD AM = 45 := by
  intros AB AD AM hAB hAD hAM
  simp [area_of_triangle_AMC, hAB, hAD, hAM]
  sorry

end area_triangle_AMC_l162_162306


namespace find_values_l162_162520

open Real

noncomputable def positive_numbers (x y : ℝ) := x > 0 ∧ y > 0

noncomputable def given_condition (x y : ℝ) := (sqrt (12 * x) * sqrt (20 * x) * sqrt (4 * y) * sqrt (25 * y) = 50)

theorem find_values (x y : ℝ) 
  (h1: positive_numbers x y) 
  (h2: given_condition x y) : 
  x * y = sqrt (25 / 24) := 
sorry

end find_values_l162_162520


namespace percentage_slump_in_business_l162_162232

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.05 * Y = 0.04 * X) : (X > 0) → (Y > 0) → (X - Y) / X * 100 = 20 := 
by
  sorry

end percentage_slump_in_business_l162_162232


namespace M_necessary_for_N_l162_162657

def M (x : ℝ) : Prop := -1 < x ∧ x < 3
def N (x : ℝ) : Prop := 0 < x ∧ x < 3

theorem M_necessary_for_N : (∀ a : ℝ, N a → M a) ∧ (∃ b : ℝ, M b ∧ ¬N b) :=
by sorry

end M_necessary_for_N_l162_162657


namespace coin_collection_l162_162668

def initial_ratio (G S : ℕ) : Prop := G = S / 3
def new_ratio (G S : ℕ) (addedG : ℕ) : Prop := G + addedG = S / 2
def total_coins_after (G S addedG : ℕ) : ℕ := G + addedG + S

theorem coin_collection (G S : ℕ) (addedG : ℕ) 
  (h1 : initial_ratio G S) 
  (h2 : addedG = 15) 
  (h3 : new_ratio G S addedG) : 
  total_coins_after G S addedG = 135 := 
by {
  sorry
}

end coin_collection_l162_162668


namespace intersection_complement_eq_l162_162892

open Set

-- Definitions from the problem conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | x ≥ 0}
def C_U_N : Set ℝ := {x | x < 0}

-- Statement of the proof problem
theorem intersection_complement_eq : M ∩ C_U_N = {x | -1 ≤ x ∧ x < 0} :=
by
  sorry

end intersection_complement_eq_l162_162892


namespace divisible_by_9_l162_162971

theorem divisible_by_9 (k : ℕ) (h : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
sorry

end divisible_by_9_l162_162971


namespace winning_percentage_l162_162491

/-- In an election with two candidates, wherein the winner received 490 votes and won by 280 votes,
we aim to prove that the winner received 70% of the total votes. -/

theorem winning_percentage (votes_winner : ℕ) (votes_margin : ℕ) (total_votes : ℕ)
  (h1 : votes_winner = 490) (h2 : votes_margin = 280)
  (h3 : total_votes = votes_winner + (votes_winner - votes_margin)) :
  (votes_winner * 100 / total_votes) = 70 :=
by
  -- Skipping the proof for now
  sorry

end winning_percentage_l162_162491


namespace value_of_a2_b2_l162_162634

theorem value_of_a2_b2 (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a - i) * i = b - i) : a^2 + b^2 = 2 :=
by sorry

end value_of_a2_b2_l162_162634


namespace circle_tangent_to_parabola_and_x_axis_eqn_l162_162014

theorem circle_tangent_to_parabola_and_x_axis_eqn :
  (∃ (h k : ℝ), k^2 = 2 * h ∧ (x - h)^2 + (y - k)^2 = 2 * h ∧ k > 0) →
    (∀ (x y : ℝ), x^2 + y^2 - x - 2 * y + 1 / 4 = 0) := by
  sorry

end circle_tangent_to_parabola_and_x_axis_eqn_l162_162014


namespace ratio_surface_area_l162_162576

open Real

theorem ratio_surface_area (R a : ℝ) 
  (h1 : 4 * R^2 = 6 * a^2) 
  (H : R = (sqrt 6 / 2) * a) : 
  3 * π * R^2 / (6 * a^2) = 3 * π / 4 :=
by {
  sorry
}

end ratio_surface_area_l162_162576


namespace option_C_is_neither_even_nor_odd_l162_162951

noncomputable def f_A (x : ℝ) : ℝ := x^2 + |x|
noncomputable def f_B (x : ℝ) : ℝ := 2^x - 2^(-x)
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 3^x
noncomputable def f_D (x : ℝ) : ℝ := 1/(x+1) + 1/(x-1)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

theorem option_C_is_neither_even_nor_odd : ¬ is_even f_C ∧ ¬ is_odd f_C :=
by
  sorry

end option_C_is_neither_even_nor_odd_l162_162951


namespace expression_even_l162_162702

theorem expression_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 1) :
  ∃ k : ℕ, 2^a * (b+1) ^ 2 * c = 2 * k :=
by
sorry

end expression_even_l162_162702


namespace race_ordering_l162_162549

theorem race_ordering
  (Lotar Manfred Jan Victor Eddy : ℕ) 
  (h1 : Lotar < Manfred) 
  (h2 : Manfred < Jan) 
  (h3 : Jan < Victor) 
  (h4 : Eddy < Victor) : 
  ∀ x, x = Victor ↔ ∀ y, (y = Lotar ∨ y = Manfred ∨ y = Jan ∨ y = Eddy) → y < x :=
by
  sorry

end race_ordering_l162_162549


namespace three_power_not_square_l162_162612

theorem three_power_not_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : ¬ ∃ k : ℕ, k * k = 3^m + 3^n + 1 := by 
  sorry

end three_power_not_square_l162_162612


namespace value_of_x_l162_162146

theorem value_of_x (b x : ℝ) (h₀ : 1 < b) (h₁ : 0 < x) (h₂ : (2 * x) ^ (Real.logb b 2) - (3 * x) ^ (Real.logb b 3) = 0) : x = 1 / 6 :=
by {
  sorry
}

end value_of_x_l162_162146


namespace total_number_of_fish_l162_162709

def number_of_tuna : Nat := 5
def number_of_spearfish : Nat := 2

theorem total_number_of_fish : number_of_tuna + number_of_spearfish = 7 := by
  sorry

end total_number_of_fish_l162_162709


namespace hyperbola_eccentricity_l162_162318

theorem hyperbola_eccentricity (h : ∀ x y m : ℝ, x^2 - y^2 / m = 1 → m > 0 → (Real.sqrt (1 + m) = Real.sqrt 3)) : ∃ m : ℝ, m = 2 := sorry

end hyperbola_eccentricity_l162_162318


namespace general_term_sequence_l162_162143

theorem general_term_sequence (a : ℕ → ℝ) (h₁ : a 1 = 1) (hn : ∀ (n : ℕ), a (n + 1) = (10 + 4 * a n) / (1 + a n)) :
  ∀ n : ℕ, a n = 5 - 7 / (1 + (3 / 4) * (-6)^(n - 1)) := 
sorry

end general_term_sequence_l162_162143


namespace parabola_equation_maximum_area_of_triangle_l162_162464

-- Definitions of the conditions
def parabola_eq (x y : ℝ) (p : ℝ) : Prop := x^2 = 2 * p * y ∧ p > 0
def distances_equal (AO AF : ℝ) : Prop := AO = 3 / 2 ∧ AF = 3 / 2
def line_eq (x k b y : ℝ) : Prop := y = k * x + b
def midpoint_y (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 1

-- Part (I)
theorem parabola_equation (p : ℝ) (x y AO AF : ℝ) (h1 : parabola_eq x y p)
  (h2 : distances_equal AO AF) :
  x^2 = 4 * y :=
sorry

-- Part (II)
theorem maximum_area_of_triangle (p k b AO AF x1 y1 x2 y2 : ℝ)
  (h1 : parabola_eq x1 y1 p) (h2 : parabola_eq x2 y2 p)
  (h3 : distances_equal AO AF) (h4 : line_eq x1 k b y1) 
  (h5 : line_eq x2 k b y2) (h6 : midpoint_y y1 y2)
  : ∃ (area : ℝ), area = 2 :=
sorry

end parabola_equation_maximum_area_of_triangle_l162_162464


namespace jeans_more_than_scarves_l162_162819

def num_ties := 34
def num_belts := 40
def num_black_shirts := 63
def num_white_shirts := 42
def num_jeans := (2 / 3) * (num_black_shirts + num_white_shirts)
def num_scarves := (1 / 2) * (num_ties + num_belts)

theorem jeans_more_than_scarves : num_jeans - num_scarves = 33 := by
  sorry

end jeans_more_than_scarves_l162_162819


namespace sixth_term_of_geometric_sequence_l162_162552

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ n

theorem sixth_term_of_geometric_sequence (a : ℝ) (r : ℝ)
  (h1 : a = 243) (h2 : geometric_sequence a r 7 = 32) :
  geometric_sequence a r 5 = 1 :=
by
  sorry

end sixth_term_of_geometric_sequence_l162_162552


namespace smallest_n_watches_l162_162031

variable {n d : ℕ}

theorem smallest_n_watches (h1 : d > 0)
  (h2 : 10 * n - 30 = 100) : n = 13 :=
by
  sorry

end smallest_n_watches_l162_162031


namespace number_of_lines_at_least_two_points_4_by_4_grid_l162_162849

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l162_162849


namespace solve_fractional_eq_l162_162984

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l162_162984


namespace common_ratio_of_geometric_sequence_l162_162421

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h4 : ∀ n, a n ≤ a (n + 1)) :
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l162_162421


namespace trig_identity_l162_162488

open Real

theorem trig_identity (α : ℝ) (h : tan α = 2) :
  2 * cos (2 * α) + 3 * sin (2 * α) - sin (α) ^ 2 = 2 / 5 :=
by sorry

end trig_identity_l162_162488


namespace vector_subtraction_proof_l162_162948

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (1, -6)
def scalar1 : ℝ := 2
def scalar2 : ℝ := 3

theorem vector_subtraction_proof :
  v1 - (scalar2 • (scalar1 • v2)) = (-3, 32) := by
  sorry

end vector_subtraction_proof_l162_162948


namespace pencils_loss_equates_20_l162_162157

/--
Patrick purchased 70 pencils and sold them at a loss equal to the selling price of some pencils. The cost of 70 pencils is 1.2857142857142856 times the selling price of 70 pencils. Prove that the loss equates to the selling price of 20 pencils.
-/
theorem pencils_loss_equates_20 
  (C S : ℝ) 
  (h1 : C = 1.2857142857142856 * S) :
  (70 * C - 70 * S) = 20 * S :=
by
  sorry

end pencils_loss_equates_20_l162_162157


namespace min_value_of_x2_y2_sub_xy_l162_162006

theorem min_value_of_x2_y2_sub_xy (x y : ℝ) (h : x^2 + y^2 + x * y = 315) : 
  ∃ m : ℝ, (∀ (u v : ℝ), u^2 + v^2 + u * v = 315 → u^2 + v^2 - u * v ≥ m) ∧ m = 105 :=
sorry

end min_value_of_x2_y2_sub_xy_l162_162006


namespace number_of_articles_l162_162706

-- Conditions
variables (C S : ℚ)
-- Given that the cost price of 50 articles is equal to the selling price of some number of articles N.
variables (N : ℚ) (h1 : 50 * C = N * S)
-- Given that the gain is 11.11111111111111 percent.
variables (gain : ℚ := 1/9) (h2 : S = C * (1 + gain))

-- Prove that N = 45
theorem number_of_articles (C S : ℚ) (N : ℚ) (h1 : 50 * C = N * S)
    (gain : ℚ := 1/9) (h2 : S = C * (1 + gain)) : N = 45 :=
by
  sorry

end number_of_articles_l162_162706


namespace time_to_cover_escalator_l162_162870

variable (v_e v_p L : ℝ)

theorem time_to_cover_escalator
  (h_v_e : v_e = 15)
  (h_v_p : v_p = 5)
  (h_L : L = 180) :
  (L / (v_e + v_p) = 9) :=
by
  -- Set up the given conditions
  rw [h_v_e, h_v_p, h_L]
  -- This will now reduce to proving 180 / (15 + 5) = 9
  sorry

end time_to_cover_escalator_l162_162870


namespace boundary_length_of_divided_rectangle_l162_162704

/-- Suppose a rectangle is divided into three equal parts along its length and two equal parts along its width, 
creating semicircle arcs connecting points on adjacent sides. Given the rectangle has an area of 72 square units, 
we aim to prove that the total length of the boundary of the resulting figure is 36.0. -/
theorem boundary_length_of_divided_rectangle 
(area_of_rectangle : ℝ)
(length_divisions : ℕ)
(width_divisions : ℕ)
(semicircle_arcs_length : ℝ)
(straight_segments_length : ℝ) :
  area_of_rectangle = 72 →
  length_divisions = 3 →
  width_divisions = 2 →
  semicircle_arcs_length = 7 * Real.pi →
  straight_segments_length = 14 →
  semicircle_arcs_length + straight_segments_length = 36 :=
by
  intros h_area h_length_div h_width_div h_arc_length h_straight_length
  sorry

end boundary_length_of_divided_rectangle_l162_162704


namespace average_temperature_l162_162810

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l162_162810


namespace prove_tan_sum_is_neg_sqrt3_l162_162113

open Real

-- Given conditions as definitions
def condition1 (α β : ℝ) : Prop := 0 < α ∧ α < π ∧ 0 < β ∧ β < π
def condition2 (α β : ℝ) : Prop := sin α + sin β = sqrt 3 * (cos α + cos β)

-- The statement of the proof
theorem prove_tan_sum_is_neg_sqrt3 (α β : ℝ) (h1 : condition1 α β) (h2 : condition2 α β) :
  tan (α + β) = -sqrt 3 :=
sorry

end prove_tan_sum_is_neg_sqrt3_l162_162113


namespace find_positive_integer_solutions_l162_162523

theorem find_positive_integer_solutions :
  ∃ (x y z : ℕ), 
    2 * x * z = y^2 ∧ 
    x + z = 1987 ∧ 
    x = 1458 ∧ 
    y = 1242 ∧ 
    z = 529 :=
  by sorry

end find_positive_integer_solutions_l162_162523


namespace missing_weights_l162_162269

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end missing_weights_l162_162269


namespace expression_is_integer_l162_162760

theorem expression_is_integer (m : ℕ) (hm : 0 < m) :
  ∃ k : ℤ, k = (m^4 / 24 + m^3 / 4 + 11*m^2 / 24 + m / 4 : ℚ) :=
by
  sorry

end expression_is_integer_l162_162760


namespace intersection_M_N_l162_162018

open Set

def M : Set ℝ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} :=
by
  sorry

end intersection_M_N_l162_162018


namespace not_speaking_hindi_is_32_l162_162390

-- Definitions and conditions
def total_diplomats : ℕ := 120
def spoke_french : ℕ := 20
def percent_neither : ℝ := 0.20
def percent_both : ℝ := 0.10

-- Number of diplomats who spoke neither French nor Hindi
def neither_french_nor_hindi := (percent_neither * total_diplomats : ℝ)

-- Number of diplomats who spoke both French and Hindi
def both_french_and_hindi := (percent_both * total_diplomats : ℝ)

-- Number of diplomats who spoke only French
def only_french := (spoke_french - both_french_and_hindi : ℝ)

-- Number of diplomats who did not speak Hindi
def not_speaking_hindi := (only_french + neither_french_nor_hindi : ℝ)

theorem not_speaking_hindi_is_32 :
  not_speaking_hindi = 32 :=
by
  -- Provide proof here
  sorry

end not_speaking_hindi_is_32_l162_162390


namespace area_ratio_eq_l162_162555

-- Define the parameters used in the problem
variables (t t1 r ρ : ℝ)

-- Define the conditions given in the problem
def area_triangle_ABC : ℝ := t
def area_triangle_A1B1C1 : ℝ := t1
def circumradius_ABC : ℝ := r
def inradius_A1B1C1 : ℝ := ρ

-- Problem statement: Prove the given equation
theorem area_ratio_eq : t / t1 = 2 * ρ / r :=
sorry

end area_ratio_eq_l162_162555


namespace initial_cost_of_milk_l162_162610

theorem initial_cost_of_milk (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) (milk_discount_rate : ℝ) (money_left : ℝ)
  (h_total_money : total_money = 20) (h_bread_cost : bread_cost = 3.50) (h_detergent_cost : detergent_cost = 10.25) (h_banana_cost_per_pound : banana_cost_per_pound = 0.75) (h_banana_pounds : banana_pounds = 2)
  (h_detergent_coupon : detergent_coupon = 1.25) (h_milk_discount_rate : milk_discount_rate = 0.5) (h_money_left : money_left = 4) : 
  ∃ (initial_milk_cost : ℝ), initial_milk_cost = 4 := 
sorry

end initial_cost_of_milk_l162_162610


namespace find_a_tangent_line_l162_162852

theorem find_a_tangent_line (a : ℝ) : 
  (∃ (x0 y0 : ℝ), y0 = a * x0^2 + (15/4 : ℝ) * x0 - 9 ∧ 
                  (y0 = 0 ∨ (x0 = 3/2 ∧ y0 = 27/4)) ∧ 
                  ∃ (m : ℝ), (0 - y0) = m * (1 - x0) ∧ (m = 2 * a * x0 + 15/4)) → 
  (a = -1 ∨ a = -25/64) := 
sorry

end find_a_tangent_line_l162_162852


namespace sam_total_coins_l162_162292

theorem sam_total_coins (nickel_count : ℕ) (dime_count : ℕ) (total_value_cents : ℤ) (nickel_value : ℤ) (dime_value : ℤ)
  (h₁ : nickel_count = 12)
  (h₂ : total_value_cents = 240)
  (h₃ : nickel_value = 5)
  (h₄ : dime_value = 10)
  (h₅ : nickel_count * nickel_value + dime_count * dime_value = total_value_cents) :
  nickel_count + dime_count = 30 := 
  sorry

end sam_total_coins_l162_162292


namespace solve_equation_l162_162550

theorem solve_equation (x : ℝ) (h : (4 * x ^ 2 + 6 * x + 2) / (x + 2) = 4 * x + 7) : x = -4 / 3 :=
by
  sorry

end solve_equation_l162_162550


namespace unique_solution_for_all_y_l162_162956

theorem unique_solution_for_all_y (x : ℝ) (h : ∀ y : ℝ, 8 * x * y - 12 * y + 2 * x - 3 = 0) : x = 3 / 2 :=
sorry

end unique_solution_for_all_y_l162_162956


namespace part1_l162_162936

theorem part1 (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 2 * b^2 = a^2 + c^2 :=
sorry

end part1_l162_162936


namespace mashed_potatoes_suggestion_count_l162_162784

def number_of_students_suggesting_bacon := 394
def extra_students_suggesting_mashed_potatoes := 63
def number_of_students_suggesting_mashed_potatoes := number_of_students_suggesting_bacon + extra_students_suggesting_mashed_potatoes

theorem mashed_potatoes_suggestion_count :
  number_of_students_suggesting_mashed_potatoes = 457 := by
  sorry

end mashed_potatoes_suggestion_count_l162_162784


namespace number_of_divisors_8_factorial_l162_162518

open Nat

theorem number_of_divisors_8_factorial :
  let n := 8!
  let factorization := [(2, 7), (3, 2), (5, 1), (7, 1)]
  let numberOfDivisors := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  n = 2^7 * 3^2 * 5^1 * 7^1 ->
  n.factors.count = 4 ->
  numberOfDivisors = 96 :=
by
  sorry

end number_of_divisors_8_factorial_l162_162518


namespace shoes_to_belts_ratio_l162_162253

variable (hats : ℕ) (belts : ℕ) (shoes : ℕ)

theorem shoes_to_belts_ratio (hats_eq : hats = 5)
                            (belts_eq : belts = hats + 2)
                            (shoes_eq : shoes = 14) : 
  (shoes / (Nat.gcd shoes belts)) = 2 ∧ (belts / (Nat.gcd shoes belts)) = 1 := 
by
  sorry

end shoes_to_belts_ratio_l162_162253


namespace Annette_more_than_Sara_l162_162293

variable (A C S : ℕ)

-- Define the given conditions as hypotheses
def Annette_Caitlin_weight : Prop := A + C = 95
def Caitlin_Sara_weight : Prop := C + S = 87

-- The theorem to prove: Annette weighs 8 pounds more than Sara
theorem Annette_more_than_Sara (h1 : Annette_Caitlin_weight A C)
                               (h2 : Caitlin_Sara_weight C S) :
  A - S = 8 := by
  sorry

end Annette_more_than_Sara_l162_162293


namespace g_decreasing_on_neg1_0_l162_162260

noncomputable def f (x : ℝ) : ℝ := 8 + 2 * x - x^2 
noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_decreasing_on_neg1_0 : 
  ∀ x y : ℝ, -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ x < y → g y < g x :=
sorry

end g_decreasing_on_neg1_0_l162_162260


namespace benny_total_hours_l162_162544

-- Define the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- State the theorem (problem) to be proved
theorem benny_total_hours : hours_per_day * days_worked = 18 :=
by
  -- Sorry to skip the actual proof
  sorry

end benny_total_hours_l162_162544


namespace tournament_start_count_l162_162051

theorem tournament_start_count (x : ℝ) (h1 : (0.1 * x = 30)) : x = 300 :=
by
  sorry

end tournament_start_count_l162_162051


namespace smallest_k_l162_162461

theorem smallest_k (k : ℕ) : 
  (k > 0 ∧ (k*(k+1)*(2*k+1)/6) % 400 = 0) → k = 800 :=
by
  sorry

end smallest_k_l162_162461


namespace expected_value_is_correct_l162_162439

-- Define the monetary outcomes associated with each side
def monetaryOutcome (X : String) : ℚ :=
  if X = "A" then 2 else 
  if X = "B" then -4 else 
  if X = "C" then 6 else 
  0

-- Define the probabilities associated with each side
def probability (X : String) : ℚ :=
  if X = "A" then 1/3 else 
  if X = "B" then 1/2 else 
  if X = "C" then 1/6 else 
  0

-- Compute the expected value
def expectedMonetaryOutcome : ℚ := (probability "A" * monetaryOutcome "A") 
                                + (probability "B" * monetaryOutcome "B") 
                                + (probability "C" * monetaryOutcome "C")

theorem expected_value_is_correct : 
  expectedMonetaryOutcome = -2/3 := by
  sorry

end expected_value_is_correct_l162_162439


namespace fourth_student_number_systematic_sampling_l162_162832

theorem fourth_student_number_systematic_sampling :
  ∀ (students : Finset ℕ), students = Finset.range 55 →
  ∀ (sample_size : ℕ), sample_size = 4 →
  ∀ (numbers_in_sample : Finset ℕ),
  numbers_in_sample = {3, 29, 42} →
  ∃ (fourth_student : ℕ), fourth_student = 44 :=
  by sorry

end fourth_student_number_systematic_sampling_l162_162832


namespace find_x_l162_162857

theorem find_x :
  (2 + 3 = 5) →
  (3 + 4 = 7) →
  (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) →
  x = 30 :=
by
  intros
  sorry

end find_x_l162_162857


namespace greatest_value_of_x_for_equation_l162_162334

theorem greatest_value_of_x_for_equation :
  ∃ x : ℝ, (4 * x - 5) ≠ 0 ∧ ((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5)) = 18 ∧ x = 50 / 29 :=
sorry

end greatest_value_of_x_for_equation_l162_162334


namespace andrea_avg_km_per_day_l162_162577

theorem andrea_avg_km_per_day
  (total_distance : ℕ := 168)
  (total_days : ℕ := 6)
  (completed_fraction : ℚ := 3/7)
  (completed_days : ℕ := 3) :
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := 
sorry

end andrea_avg_km_per_day_l162_162577


namespace possible_values_x_plus_y_l162_162362

theorem possible_values_x_plus_y (x y : ℝ) (h1 : x = y * (3 - y)^2) (h2 : y = x * (3 - x)^2) :
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end possible_values_x_plus_y_l162_162362


namespace geometric_sequence_sum_l162_162757

theorem geometric_sequence_sum (a_1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  (∀ n, S (n+1) = a_1 * (1 - q^(n+1)) / (1 - q)) →
  S 4 / a_1 = 15 :=
by
  intros hq hsum
  sorry

end geometric_sequence_sum_l162_162757


namespace time_to_pass_platform_l162_162911

-- Definitions for the given conditions
def train_length := 1200 -- length of the train in meters
def tree_crossing_time := 120 -- time taken to cross a tree in seconds
def platform_length := 1200 -- length of the platform in meters

-- Calculation of speed of the train and distance to be covered
def train_speed := train_length / tree_crossing_time -- speed in meters per second
def total_distance_to_cover := train_length + platform_length -- total distance in meters

-- Proof statement that given the above conditions, the time to pass the platform is 240 seconds
theorem time_to_pass_platform : 
  total_distance_to_cover / train_speed = 240 :=
  by sorry

end time_to_pass_platform_l162_162911


namespace raft_sticks_total_l162_162420

theorem raft_sticks_total : 
  let S := 45 
  let G := (3/5 * 45 : ℝ)
  let M := 45 + G + 15
  let D := 2 * M - 7
  S + G + M + D = 326 := 
by
  sorry

end raft_sticks_total_l162_162420


namespace total_sand_l162_162164

variable (capacity_per_bag : ℕ) (number_of_bags : ℕ)

theorem total_sand (h1 : capacity_per_bag = 65) (h2 : number_of_bags = 12) : capacity_per_bag * number_of_bags = 780 := by
  sorry

end total_sand_l162_162164


namespace time_brushing_each_cat_l162_162214

theorem time_brushing_each_cat :
  ∀ (t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats : ℕ),
  t_total_free_time = 3 * 60 →
  t_vacuum = 45 →
  t_dust = 60 →
  t_mop = 30 →
  t_cats = 3 →
  t_free_left_after_cleaning = 30 →
  ((t_total_free_time - t_free_left_after_cleaning) - (t_vacuum + t_dust + t_mop)) / t_cats = 5
 := by
  intros t_total_free_time t_vacuum t_dust t_mop t_free_left_after_cleaning t_cats
  intros h_total_free_time h_vacuum h_dust h_mop h_cats h_free_left
  sorry

end time_brushing_each_cat_l162_162214


namespace solve_for_A_l162_162321

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^2
def g (B x : ℝ) : ℝ := B * x^2

-- A Lean theorem that formalizes the given math problem.
theorem solve_for_A (A B : ℝ) (h₁ : B ≠ 0) (h₂ : f A B (g B 1) = 0) : A = 3 :=
by {
  sorry
}

end solve_for_A_l162_162321


namespace apple_juice_cost_l162_162082

noncomputable def cost_of_apple_juice (cost_per_orange_juice : ℝ) (total_bottles : ℕ) (total_cost : ℝ) (orange_juice_bottles : ℕ) : ℝ :=
  (total_cost - cost_per_orange_juice * orange_juice_bottles) / (total_bottles - orange_juice_bottles)

theorem apple_juice_cost :
  let cost_per_orange_juice := 0.7
  let total_bottles := 70
  let total_cost := 46.2
  let orange_juice_bottles := 42
  cost_of_apple_juice cost_per_orange_juice total_bottles total_cost orange_juice_bottles = 0.6 := by
    sorry

end apple_juice_cost_l162_162082


namespace positive_y_equals_32_l162_162061

theorem positive_y_equals_32 (y : ℝ) (h : y^2 = 1024) (hy : 0 < y) : y = 32 :=
sorry

end positive_y_equals_32_l162_162061


namespace colored_shirts_count_l162_162118

theorem colored_shirts_count (n : ℕ) (h1 : 6 = 6) (h2 : (1 / (n : ℝ)) ^ 6 = 1 / 120) : n = 2 := 
sorry

end colored_shirts_count_l162_162118


namespace inning_is_31_l162_162558

noncomputable def inning_number (s: ℕ) (i: ℕ) (a: ℕ) : ℕ := s - a + i

theorem inning_is_31
  (batsman_runs: ℕ)
  (increase_average: ℕ)
  (final_average: ℕ) 
  (n: ℕ) 
  (h1: batsman_runs = 92)
  (h2: increase_average = 3)
  (h3: final_average = 44)
  (h4: 44 * n - 92 = 41 * n): 
  inning_number 44 1 3 = 31 := 
by 
  sorry

end inning_is_31_l162_162558


namespace custom_op_2006_l162_162763

def custom_op (n : ℕ) : ℕ := 
  match n with 
  | 0 => 1
  | (n+1) => 2 + custom_op n

theorem custom_op_2006 : custom_op 2005 = 4011 :=
by {
  sorry
}

end custom_op_2006_l162_162763


namespace tan_double_angle_third_quadrant_l162_162416

theorem tan_double_angle_third_quadrant
  (α : ℝ)
  (sin_alpha : Real.sin α = -3/5)
  (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.tan (2 * α) = 24 / 7 :=
sorry

end tan_double_angle_third_quadrant_l162_162416


namespace zoe_spent_amount_l162_162573

def flower_price : ℕ := 3
def roses_bought : ℕ := 8
def daisies_bought : ℕ := 2

theorem zoe_spent_amount :
  roses_bought + daisies_bought = 10 ∧
  flower_price = 3 →
  (roses_bought + daisies_bought) * flower_price = 30 :=
by
  sorry

end zoe_spent_amount_l162_162573


namespace find_d_l162_162696

variables {x y z k d : ℝ}
variables {a : ℝ} (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
variables (h_ap : x * (y - z) + y * (z - x) + z * (x - y) = 0)
variables (h_sum : x * (y - z) + (y * (z - x) + d) + (z * (x - y) + 2 * d) = k)

theorem find_d : d = k / 3 :=
sorry

end find_d_l162_162696


namespace find_width_of_brick_l162_162758

theorem find_width_of_brick (l h : ℝ) (SurfaceArea : ℝ) (w : ℝ) :
  l = 8 → h = 2 → SurfaceArea = 152 → 2*l*w + 2*l*h + 2*w*h = SurfaceArea → w = 6 :=
by
  intro l_value
  intro h_value
  intro SurfaceArea_value
  intro surface_area_equation
  sorry

end find_width_of_brick_l162_162758


namespace solve_for_c_l162_162617

theorem solve_for_c (c : ℚ) :
  (c - 35) / 14 = (2 * c + 9) / 49 →
  c = 1841 / 21 :=
by
  sorry

end solve_for_c_l162_162617


namespace estimated_probability_is_2_div_9_l162_162976

def groups : List (List ℕ) :=
  [[3, 4, 3], [4, 3, 2], [3, 4, 1], [3, 4, 2], [2, 3, 4], [1, 4, 2], [2, 4, 3], [3, 3, 1], [1, 1, 2],
   [3, 4, 2], [2, 4, 1], [2, 4, 4], [4, 3, 1], [2, 3, 3], [2, 1, 4], [3, 4, 4], [1, 4, 2], [1, 3, 4]]

def count_desired_groups (gs : List (List ℕ)) : Nat :=
  gs.foldl (fun acc g =>
    if g.contains 1 ∧ g.contains 2 ∧ g.length ≥ 3 then acc + 1 else acc) 0

theorem estimated_probability_is_2_div_9 :
  (count_desired_groups groups) = 4 →
  4 / 18 = 2 / 9 :=
by
  intro h
  sorry

end estimated_probability_is_2_div_9_l162_162976


namespace volleyball_club_members_l162_162207

variables (B G : ℝ)

theorem volleyball_club_members (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 18) : B = 12 := by
  -- Mathematical steps and transformations done here to show B = 12
  sorry

end volleyball_club_members_l162_162207


namespace Lauryn_earnings_l162_162844

variables (L : ℝ)

theorem Lauryn_earnings (h1 : 0.70 * L + L = 3400) : L = 2000 :=
sorry

end Lauryn_earnings_l162_162844


namespace blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l162_162282

variables (length magnitude : ℕ)
variable (price : ℝ)
variable (area : ℕ)

-- Definitions based on the conditions
def length_is_about_4 (length : ℕ) : Prop := length = 4
def price_is_about_9_50 (price : ℝ) : Prop := price = 9.50
def large_area_is_about_3 (area : ℕ) : Prop := area = 3
def small_area_is_about_1 (area : ℕ) : Prop := area = 1

-- Proof problem statements
theorem blackboard_length_is_meters : length_is_about_4 length → length = 4 := by sorry
theorem pencil_case_price_is_yuan : price_is_about_9_50 price → price = 9.50 := by sorry
theorem campus_area_is_hectares : large_area_is_about_3 area → area = 3 := by sorry
theorem fingernail_area_is_square_centimeters : small_area_is_about_1 area → area = 1 := by sorry

end blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l162_162282


namespace johnny_ways_to_choose_l162_162414

def num_ways_to_choose_marbles (total_marbles : ℕ) (marbles_to_choose : ℕ) (blue_must_be_included : ℕ) : ℕ :=
  Nat.choose (total_marbles - blue_must_be_included) (marbles_to_choose - blue_must_be_included)

-- Given conditions
def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_must_be_included : ℕ := 1

-- Theorem to prove the number of ways to choose the marbles
theorem johnny_ways_to_choose :
  num_ways_to_choose_marbles total_marbles marbles_to_choose blue_must_be_included = 56 := by
  sorry

end johnny_ways_to_choose_l162_162414


namespace evaluate_g_f_l162_162201

def f (a b : ℤ) : ℤ × ℤ := (-a, b)

def g (m n : ℤ) : ℤ × ℤ := (m, -n)

theorem evaluate_g_f : g (f 2 (-3)).1 (f 2 (-3)).2 = (-2, 3) := by
  sorry

end evaluate_g_f_l162_162201


namespace george_choices_l162_162311

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- State the theorem to prove the number of ways to choose 3 out of 9 colors is 84
theorem george_choices : binomial 9 3 = 84 := by
  sorry

end george_choices_l162_162311


namespace largest_angle_in_triangle_l162_162726

open Real

theorem largest_angle_in_triangle
  (A B C : ℝ)
  (h : sin A / sin B / sin C = 1 / sqrt 2 / sqrt 5) :
  A ≤ B ∧ B ≤ C → C = 3 * π / 4 :=
by
  sorry

end largest_angle_in_triangle_l162_162726


namespace minimum_x_plus_y_l162_162454

theorem minimum_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
    (h1 : x - y < 1) (h2 : 2 * x - y > 2) (h3 : x < 5) : 
    x + y ≥ 6 :=
sorry

end minimum_x_plus_y_l162_162454


namespace scientific_notation_of_87000000_l162_162954

theorem scientific_notation_of_87000000 :
  87000000 = 8.7 * 10^7 := 
sorry

end scientific_notation_of_87000000_l162_162954


namespace fish_population_estimation_l162_162680

theorem fish_population_estimation (N : ℕ) (h1 : 80 ≤ N)
  (h_tagged_returned : true)
  (h_second_catch : 80 ≤ N)
  (h_tagged_in_second_catch : 2 = 80 * 80 / N) :
  N = 3200 :=
by
  sorry

end fish_population_estimation_l162_162680


namespace opposites_of_each_other_l162_162553

theorem opposites_of_each_other (a b : ℚ) (h : a + b = 0) : a = -b :=
  sorry

end opposites_of_each_other_l162_162553


namespace share_ratio_l162_162924

theorem share_ratio (A B C : ℕ) (hA : A = (2 * B) / 3) (hA_val : A = 372) (hB_val : B = 93) (hC_val : C = 62) : B / C = 3 / 2 := 
by 
  sorry

end share_ratio_l162_162924


namespace gcd_97_power_l162_162939

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end gcd_97_power_l162_162939


namespace sum_of_numbers_l162_162249

-- Define the conditions
variables (a b : ℝ) (r d : ℝ)
def geometric_progression := a = 3 * r ∧ b = 3 * r^2
def arithmetic_progression := b = a + d ∧ 9 = b + d

-- Define the problem as proving the sum of a and b
theorem sum_of_numbers (h1 : geometric_progression a b r)
                       (h2 : arithmetic_progression a b d) : 
  a + b = 45 / 4 :=
sorry

end sum_of_numbers_l162_162249


namespace mason_savings_fraction_l162_162698

theorem mason_savings_fraction (M p b : ℝ) (h : (1 / 4) * M = (2 / 5) * b * p) : 
  (M - b * p) / M = 3 / 8 :=
by 
  sorry

end mason_savings_fraction_l162_162698


namespace sqrt_17_estimation_l162_162685

theorem sqrt_17_estimation :
  4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := 
sorry

end sqrt_17_estimation_l162_162685


namespace smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l162_162080

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (1 / 2) * Real.cos (2 * x)

theorem smallest_positive_period_and_range :
  (∀ x, f (x + Real.pi) = f x) ∧ (Set.range f = Set.Icc (-3 / 2) (5 / 2)) :=
by
  sorry

theorem sin_2x0_if_zero_of_f (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ Real.pi / 2)
  (hf : f x0 = 0) : Real.sin (2 * x0) = (Real.sqrt 15 - Real.sqrt 3) / 8 :=
by
  sorry

end smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l162_162080


namespace width_of_wide_flags_l162_162239

def total_fabric : ℕ := 1000
def leftover_fabric : ℕ := 294
def num_square_flags : ℕ := 16
def square_flag_area : ℕ := 16
def num_tall_flags : ℕ := 10
def tall_flag_area : ℕ := 15
def num_wide_flags : ℕ := 20
def wide_flag_height : ℕ := 3

theorem width_of_wide_flags :
  (total_fabric - leftover_fabric - (num_square_flags * square_flag_area + num_tall_flags * tall_flag_area)) / num_wide_flags / wide_flag_height = 5 :=
by
  sorry

end width_of_wide_flags_l162_162239


namespace sin_390_eq_half_l162_162794

theorem sin_390_eq_half : Real.sin (390 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_390_eq_half_l162_162794


namespace carl_cost_l162_162041

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end carl_cost_l162_162041


namespace sum_of_integers_l162_162652

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 300) (h2 : m * (m + 1) * (m + 2) = 300) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 49 := 
by sorry

end sum_of_integers_l162_162652


namespace find_b_l162_162375

def direction_vector (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (x2 - x1, y2 - y1)

theorem find_b (b : ℝ)
  (hx1 : ℝ := -3) (hy1 : ℝ := 1) (hx2 : ℝ := 0) (hy2 : ℝ := 4)
  (hdir : direction_vector hx1 hy1 hx2 hy2 = (3, b)) :
  b = 3 :=
by
  -- Mathematical proof of b = 3 goes here
  sorry

end find_b_l162_162375


namespace average_of_two_numbers_l162_162835

theorem average_of_two_numbers (A B C : ℝ) (h1 : (A + B + C)/3 = 48) (h2 : C = 32) : (A + B)/2 = 56 := by
  sorry

end average_of_two_numbers_l162_162835


namespace number_divided_by_189_l162_162251

noncomputable def target_number : ℝ := 3486

theorem number_divided_by_189 :
  target_number / 189 = 18.444444444444443 :=
by
  sorry

end number_divided_by_189_l162_162251


namespace root_equation_solution_l162_162350

theorem root_equation_solution (a : ℝ) (h : 3 * a^2 - 5 * a - 2 = 0) : 6 * a^2 - 10 * a = 4 :=
by 
  sorry

end root_equation_solution_l162_162350


namespace arithmetic_seq_S10_l162_162011

open BigOperators

variables (a : ℕ → ℚ) (d : ℚ)

-- Definitions based on the conditions
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) := ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 5 = 1
axiom h2 : a 1 + a 7 + a 10 = a 4 + a 6

-- We aim to prove the sum of the first 10 terms
def S (n : ℕ) :=
  ∑ i in Finset.range n, a (i + 1)

theorem arithmetic_seq_S10 : arithmetic_seq a d → S a 10 = 25 / 3 :=
by
  sorry

end arithmetic_seq_S10_l162_162011


namespace range_of_a_l162_162383

noncomputable def f (x : ℝ) := Real.log x / Real.log 2

noncomputable def g (x a : ℝ) := Real.sqrt x + Real.sqrt (a - x)

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 : ℝ, 0 <= x1 ∧ x1 <= a → ∃ x2 : ℝ, 4 ≤ x2 ∧ x2 ≤ 16 ∧ g x1 a = f x2) →
  4 ≤ a ∧ a ≤ 8 :=
sorry 

end range_of_a_l162_162383


namespace max_area_of_rectangle_l162_162444

theorem max_area_of_rectangle (L : ℝ) (hL : L = 16) :
  ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 8 → A = x * (8 - x)) ∧ A = 16 :=
by
  sorry

end max_area_of_rectangle_l162_162444


namespace karen_savings_over_30_years_l162_162648

theorem karen_savings_over_30_years 
  (P_exp : ℕ) (L_exp : ℕ) 
  (P_cheap : ℕ) (L_cheap : ℕ) 
  (T : ℕ)
  (hP_exp : P_exp = 300)
  (hL_exp : L_exp = 15)
  (hP_cheap : P_cheap = 120)
  (hL_cheap : L_cheap = 5)
  (hT : T = 30) : 
  (P_cheap * (T / L_cheap) - P_exp * (T / L_exp)) = 120 := 
by 
  sorry

end karen_savings_over_30_years_l162_162648


namespace base_b_digit_sum_l162_162881

theorem base_b_digit_sum :
  ∃ (b : ℕ), ((b^2 / 2 + b / 2) % b = 2) ∧ (b = 8) :=
by
  sorry

end base_b_digit_sum_l162_162881


namespace bottles_in_one_bag_l162_162921

theorem bottles_in_one_bag (total_bottles : ℕ) (cartons bags_per_carton : ℕ)
  (h1 : total_bottles = 180)
  (h2 : cartons = 3)
  (h3 : bags_per_carton = 4) :
  total_bottles / cartons / bags_per_carton = 15 :=
by sorry

end bottles_in_one_bag_l162_162921


namespace ratio_of_mixture_l162_162163

theorem ratio_of_mixture (x y : ℚ)
  (h1 : 0.6 = (4 * x + 7 * y) / (9 * x + 9 * y))
  (h2 : 50 = 9 * x + 9 * y) : x / y = 8 / 7 := 
sorry

end ratio_of_mixture_l162_162163


namespace outfit_combinations_l162_162403

theorem outfit_combinations :
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6 -- 5 ties + no tie option
  let num_belts := 3 -- 2 belts + no belt option
  num_shirts * num_pants * num_ties * num_belts = 360 :=
by
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6
  let num_belts := 3
  show num_shirts * num_pants * num_ties * num_belts = 360
  sorry

end outfit_combinations_l162_162403


namespace percentage_difference_l162_162745

theorem percentage_difference :
  let x := 50
  let y := 30
  let p1 := 60
  let p2 := 30
  (p1 / 100 * x) - (p2 / 100 * y) = 21 :=
by
  sorry

end percentage_difference_l162_162745


namespace problem1_problem2_l162_162221

-- Define the function f(x)
def f (m x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Problem 1: Prove the range of x for f(x) = 4 when m = 2
theorem problem1 (x : ℝ) : f 2 x = 4 ↔ -1 / 2 ≤ x ∧ x ≤ 3 / 2 :=
by
  sorry

-- Problem 2: Prove the range of m given f(1) ≤ (2a^2 + 8) / a for any positive a
theorem problem2 (m : ℝ) (h : ∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) : -8 ≤ m ∧ m ≤ 6 :=
by
  sorry

end problem1_problem2_l162_162221


namespace numerator_is_12_l162_162213

theorem numerator_is_12 (x : ℕ) (h1 : (x : ℤ) / (2 * x + 4 : ℤ) = 3 / 7) : x = 12 := 
sorry

end numerator_is_12_l162_162213


namespace triangle_side_length_range_l162_162037

theorem triangle_side_length_range (x : ℝ) : 
  (1 < x) ∧ (x < 9) → ¬ (x = 10) :=
by
  sorry

end triangle_side_length_range_l162_162037


namespace sin_1035_eq_neg_sqrt2_div_2_l162_162468

theorem sin_1035_eq_neg_sqrt2_div_2 : Real.sin (1035 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
    sorry

end sin_1035_eq_neg_sqrt2_div_2_l162_162468


namespace find_constants_l162_162335

open Matrix 

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem find_constants :
  let s := (-10 : ℤ)
  let t := (-8 : ℤ)
  let u := (-36 : ℤ)
  B^3 + s • (B^2) + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := sorry

end find_constants_l162_162335


namespace candidate_lost_by_l162_162524

noncomputable def candidate_votes (total_votes : ℝ) := 0.35 * total_votes
noncomputable def rival_votes (total_votes : ℝ) := 0.65 * total_votes

theorem candidate_lost_by (total_votes : ℝ) (h : total_votes = 7899.999999999999) :
  rival_votes total_votes - candidate_votes total_votes = 2370 :=
by
  sorry

end candidate_lost_by_l162_162524


namespace bridge_length_is_100_l162_162672

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (wind_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let wind_speed_ms := wind_speed_kmh * 1000 / 3600
  let effective_speed_ms := train_speed_ms - wind_speed_ms
  let distance_covered := effective_speed_ms * crossing_time_s
  distance_covered - train_length

theorem bridge_length_is_100 :
  length_of_bridge 150 45 15 30 = 100 :=
by
  sorry

end bridge_length_is_100_l162_162672


namespace square_area_proof_square_area_square_area_final_square_area_correct_l162_162388

theorem square_area_proof (x : ℝ) (s1 : ℝ) (s2 : ℝ) (A : ℝ)
  (h1 : s1 = 5 * x - 20)
  (h2 : s2 = 25 - 2 * x)
  (h3 : s1 = s2) :
  A = (s1 * s1) := by
  -- We need to prove A = s1 * s1
  sorry

theorem square_area (x : ℝ) (s : ℝ) (h : s = 85 / 7) :
  s ^ 2 = 7225 / 49 := by
  -- We need to prove s^2 = 7225 / 49
  sorry

theorem square_area_final (x : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (A : ℝ) :
  A = (85 / 7) ^ 2 := by
  -- We need to prove A = (85 / 7) ^ 2
  sorry

theorem square_area_correct (x : ℝ)
  (A : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (h2 : A = (85 / 7) ^ 2) :
  A = 7225 / 49 := by
  -- We need to prove A = 7225 / 49
  sorry

end square_area_proof_square_area_square_area_final_square_area_correct_l162_162388


namespace total_boxes_sold_l162_162525

-- Define the variables for each day's sales
def friday_sales : ℕ := 30
def saturday_sales : ℕ := 2 * friday_sales
def sunday_sales : ℕ := saturday_sales - 15
def total_sales : ℕ := friday_sales + saturday_sales + sunday_sales

-- State the theorem to prove the total sales over three days
theorem total_boxes_sold : total_sales = 135 :=
by 
  -- Here we would normally put the proof steps, but since we're asked only for the statement,
  -- we skip the proof with sorry
  sorry

end total_boxes_sold_l162_162525


namespace abs_inequality_solution_set_l162_162404

theorem abs_inequality_solution_set :
  { x : ℝ | |x - 1| + |x + 2| ≥ 5 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end abs_inequality_solution_set_l162_162404


namespace relationship_M_N_l162_162777

def M : Set Int := {-1, 0, 1}
def N : Set Int := {x | ∃ a b : Int, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem relationship_M_N : N ⊆ M ∧ N ≠ M := by
  sorry

end relationship_M_N_l162_162777


namespace estimate_red_balls_l162_162478

-- Definitions based on conditions
def total_balls : ℕ := 20
def total_draws : ℕ := 100
def red_draws : ℕ := 30

-- The theorem statement
theorem estimate_red_balls (h1 : total_balls = 20) (h2 : total_draws = 100) (h3 : red_draws = 30) :
  (total_balls * (red_draws / total_draws) : ℤ) = 6 := 
by
  sorry

end estimate_red_balls_l162_162478


namespace constant_subsequence_exists_l162_162332

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem constant_subsequence_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (f : ℕ → ℕ) (c : ℕ), (∀ n m, n < m → f n < f m) ∧ (∀ n, sum_of_digits (⌊a * ↑(f n) + b⌋₊) = c) :=
sorry

end constant_subsequence_exists_l162_162332


namespace find_original_number_l162_162247

-- Defining the conditions as given in the problem
def original_number_condition (x : ℤ) : Prop :=
  3 * (3 * x - 6) = 141

-- Stating the main theorem to be proven
theorem find_original_number (x : ℤ) (h : original_number_condition x) : x = 17 :=
sorry

end find_original_number_l162_162247


namespace quadratic_real_roots_iff_l162_162548

theorem quadratic_real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 :=
by
  -- Proof is omitted, we only need the statement
  sorry

end quadratic_real_roots_iff_l162_162548


namespace potion_kits_needed_l162_162308

-- Definitions
def num_spellbooks := 5
def cost_spellbook_gold := 5
def cost_potion_kit_silver := 20
def num_owls := 1
def cost_owl_gold := 28
def silver_per_gold := 9
def total_silver := 537

-- Prove that Harry needs to buy 3 potion kits.
def Harry_needs_to_buy : Prop :=
  let cost_spellbooks_silver := num_spellbooks * cost_spellbook_gold * silver_per_gold
  let cost_owl_silver := num_owls * cost_owl_gold * silver_per_gold
  let total_cost_silver := cost_spellbooks_silver + cost_owl_silver
  let remaining_silver := total_silver - total_cost_silver
  let num_potion_kits := remaining_silver / cost_potion_kit_silver
  num_potion_kits = 3

theorem potion_kits_needed : Harry_needs_to_buy :=
  sorry

end potion_kits_needed_l162_162308


namespace remainder_problem_l162_162660

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 2) (h2 : n = 197) : 197 % 16 = 5 := by
  sorry

end remainder_problem_l162_162660


namespace largest_possible_gcd_l162_162560

theorem largest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 221) : ∃ d, Nat.gcd a b = d ∧ d = 17 :=
sorry

end largest_possible_gcd_l162_162560


namespace sufficient_but_not_necessary_l162_162795

theorem sufficient_but_not_necessary (x: ℝ) (hx: 0 < x ∧ x < 1) : 0 < x^2 ∧ x^2 < 1 ∧ (∀ y, 0 < y^2 ∧ y^2 < 1 → (y > 0 ∧ y < 1 ∨ y < 0 ∧ y > -1)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l162_162795


namespace purple_ring_weight_l162_162301

def orange_ring_weight : ℝ := 0.08
def white_ring_weight : ℝ := 0.42
def total_weight : ℝ := 0.83

theorem purple_ring_weight : 
  ∃ (purple_ring_weight : ℝ), purple_ring_weight = total_weight - (orange_ring_weight + white_ring_weight) := 
  by
  use 0.33
  sorry

end purple_ring_weight_l162_162301


namespace problem1_problem2_l162_162046

-- Problem 1 Proof Statement
theorem problem1 : Real.sin (30 * Real.pi / 180) + abs (-1) - (Real.sqrt 3 - Real.pi) ^ 0 = 1 / 2 := 
  by sorry

-- Problem 2 Proof Statement
theorem problem2 (x: ℝ) (hx : x ≠ 2) : (2 * x - 3) / (x - 2) - (x - 1) / (x - 2) = 1 := 
  by sorry

end problem1_problem2_l162_162046


namespace units_digit_35_pow_7_plus_93_pow_45_l162_162176

-- Definitions of units digit calculations for the specific values
def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_35_pow_7 : ℕ := units_digit (35 ^ 7)
def units_digit_93_pow_45 : ℕ := units_digit (93 ^ 45)

-- Statement to prove that the sum of the units digits is 8
theorem units_digit_35_pow_7_plus_93_pow_45 : 
  units_digit (35 ^ 7) + units_digit (93 ^ 45) = 8 :=
by 
  sorry -- proof omitted

end units_digit_35_pow_7_plus_93_pow_45_l162_162176


namespace gloria_pencils_total_l162_162048

-- Define the number of pencils Gloria initially has.
def pencils_gloria_initial : ℕ := 2

-- Define the number of pencils Lisa initially has.
def pencils_lisa_initial : ℕ := 99

-- Define the final number of pencils Gloria will have after receiving all of Lisa's pencils.
def pencils_gloria_final : ℕ := pencils_gloria_initial + pencils_lisa_initial

-- Prove that the final number of pencils Gloria will have is 101.
theorem gloria_pencils_total : pencils_gloria_final = 101 :=
by sorry

end gloria_pencils_total_l162_162048


namespace four_digit_multiples_of_7_l162_162133

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l162_162133


namespace S_n_formula_l162_162894

def P (n : ℕ) : Type := sorry -- The type representing the nth polygon, not fully defined here.
def S : ℕ → ℝ := sorry -- The sequence S_n defined recursively.

-- Recursive definition of S_n given
axiom S_0 : S 0 = 1

-- This axiom represents the recursive step mentioned in the problem.
axiom S_rec : ∀ (k : ℕ), S (k + 1) = S k + (4^k / 3^(2*k + 2))

-- The main theorem we need to prove
theorem S_n_formula (n : ℕ) : 
  S n = (8 / 5) - (3 / 5) * (4 / 9)^n := sorry

end S_n_formula_l162_162894


namespace alyssa_photos_vacation_l162_162492

theorem alyssa_photos_vacation
  (pages_first_section : ℕ)
  (photos_per_page_first_section : ℕ)
  (pages_second_section : ℕ)
  (photos_per_page_second_section : ℕ)
  (pages_total : ℕ)
  (photos_per_page_remaining : ℕ)
  (pages_remaining : ℕ)
  (h_total_pages : pages_first_section + pages_second_section + pages_remaining = pages_total)
  (h_photos_first_section : photos_per_page_first_section = 3)
  (h_photos_second_section : photos_per_page_second_section = 4)
  (h_pages_first_section : pages_first_section = 10)
  (h_pages_second_section : pages_second_section = 10)
  (h_photos_remaining : photos_per_page_remaining = 3)
  (h_pages_total : pages_total = 30)
  (h_pages_remaining : pages_remaining = 10) :
  pages_first_section * photos_per_page_first_section +
  pages_second_section * photos_per_page_second_section +
  pages_remaining * photos_per_page_remaining = 100 := by
sorry

end alyssa_photos_vacation_l162_162492


namespace total_marbles_proof_l162_162803

def red_marble_condition (b r : ℕ) : Prop :=
  r = b + (3 * b / 10)

def yellow_marble_condition (r y : ℕ) : Prop :=
  y = r + (5 * r / 10)

def total_marbles (b r y : ℕ) : ℕ :=
  r + b + y

theorem total_marbles_proof (b r y : ℕ)
  (h1 : red_marble_condition b r)
  (h2 : yellow_marble_condition r y) :
  total_marbles b r y = 425 * r / 130 :=
by {
  sorry
}

end total_marbles_proof_l162_162803


namespace tom_pie_portion_l162_162236

theorem tom_pie_portion :
  let pie_left := 5 / 8
  let friends := 4
  let portion_per_person := pie_left / friends
  portion_per_person = 5 / 32 := by
  sorry

end tom_pie_portion_l162_162236


namespace lock_and_key_requirements_l162_162906

/-- There are 7 scientists each with a key to an electronic lock which requires at least 4 scientists to open.
    - Prove that the minimum number of unique features (locks) the electronic lock must have is 35.
    - Prove that each scientist's key should have at least 20 features.
--/
theorem lock_and_key_requirements :
  ∃ (locks : ℕ) (features_per_key : ℕ), 
    locks = 35 ∧ features_per_key = 20 ∧
    (∀ (n_present : ℕ), n_present ≥ 4 → 7 - n_present ≤ 3) ∧
    (∀ (n_absent : ℕ), n_absent ≤ 3 → 7 - n_absent ≥ 4)
:= sorry

end lock_and_key_requirements_l162_162906


namespace symmetric_point_origin_l162_162771

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end symmetric_point_origin_l162_162771


namespace space_shuttle_speed_kmh_l162_162863

-- Define the given conditions
def speedInKmPerSecond : ℕ := 4
def secondsInAnHour : ℕ := 3600

-- State the proof problem
theorem space_shuttle_speed_kmh : speedInKmPerSecond * secondsInAnHour = 14400 := by
  sorry

end space_shuttle_speed_kmh_l162_162863


namespace compute_expression_l162_162365

theorem compute_expression:
  let a := 3
  let b := 7
  (a + b) ^ 2 + Real.sqrt (a^2 + b^2) = 100 + Real.sqrt 58 :=
by
  sorry

end compute_expression_l162_162365


namespace quadrilateral_area_l162_162180

theorem quadrilateral_area 
  (d : ℝ) (h₁ h₂ : ℝ) 
  (hd : d = 22) 
  (hh₁ : h₁ = 9) 
  (hh₂ : h₂ = 6) : 
  (1/2 * d * h₁ + 1/2 * d * h₂ = 165) :=
by
  sorry

end quadrilateral_area_l162_162180


namespace symmetric_points_x_axis_l162_162341

theorem symmetric_points_x_axis (m n : ℤ) :
  (-4, m - 3) = (2 * n, -1) → (m = 2 ∧ n = -2) :=
by
  sorry

end symmetric_points_x_axis_l162_162341


namespace total_spent_on_toys_l162_162009

-- Definitions for costs
def cost_car : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_truck : ℝ := 5.86

-- The statement to prove
theorem total_spent_on_toys : cost_car + cost_skateboard + cost_truck = 25.62 := by
  sorry

end total_spent_on_toys_l162_162009


namespace fg_of_neg2_l162_162288

def f (x : ℤ) : ℤ := x^2 + 4
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_of_neg2 : f (g (-2)) = 20 := by
  sorry

end fg_of_neg2_l162_162288


namespace part_to_third_fraction_is_six_five_l162_162579

noncomputable def ratio_of_part_to_third_fraction (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ℝ :=
  P / (N / 3)

theorem part_to_third_fraction_is_six_five (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ratio_of_part_to_third_fraction P N h1 h2 = 6 / 5 :=
  sorry

end part_to_third_fraction_is_six_five_l162_162579


namespace squirrels_and_nuts_l162_162038

theorem squirrels_and_nuts (number_of_squirrels number_of_nuts : ℕ) 
    (h1 : number_of_squirrels = 4) 
    (h2 : number_of_squirrels = number_of_nuts + 2) : 
    number_of_nuts = 2 :=
by
  sorry

end squirrels_and_nuts_l162_162038


namespace sum_of_f_greater_than_zero_l162_162355

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem sum_of_f_greater_than_zero 
  (a b c : ℝ) 
  (h1 : a + b > 0) 
  (h2 : b + c > 0) 
  (h3 : c + a > 0) : 
  f a + f b + f c > 0 := 
by 
  sorry

end sum_of_f_greater_than_zero_l162_162355


namespace find_ending_number_l162_162155

theorem find_ending_number (n : ℕ) 
  (h1 : n ≥ 7) 
  (h2 : ∀ m, 7 ≤ m ∧ m ≤ n → m % 7 = 0)
  (h3 : (7 + n) / 2 = 15) : n = 21 := 
sorry

end find_ending_number_l162_162155


namespace find_n_eq_130_l162_162607

theorem find_n_eq_130 
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : 0 < n)
  (h2 : d1 < d2)
  (h3 : d2 < d3)
  (h4 : d3 < d4)
  (h5 : ∀ d, d ∣ n → d = d1 ∨ d = d2 ∨ d = d3 ∨ d = d4 ∨ d ∣ n → ¬(1 < d ∧ d < d1))
  (h6 : n = d1^2 + d2^2 + d3^2 + d4^2) : n = 130 := 
  sorry

end find_n_eq_130_l162_162607


namespace find_p_q_r_l162_162786

theorem find_p_q_r  (t : ℝ) (p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
                    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p / q) - Real.sqrt r)
                    (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
                    (rel_prime : Nat.gcd p q = 1) : 
                    p + q + r = 5 := 
by
  sorry

end find_p_q_r_l162_162786


namespace blocks_fit_into_box_l162_162094

theorem blocks_fit_into_box :
  let box_height := 8
  let box_width := 10
  let box_length := 12
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let box_volume := box_height * box_width * box_length
  let block_volume := block_height * block_width * block_length
  let num_blocks := box_volume / block_volume
  num_blocks = 40 :=
by
  sorry

end blocks_fit_into_box_l162_162094


namespace solve_inequality_when_a_is_one_range_of_values_for_a_l162_162387

open Real

-- Part (1) Statement
theorem solve_inequality_when_a_is_one (a x : ℝ) (h : a = 1) : 
  |x - a| + |x + 2| ≤ 5 → -3 ≤ x ∧ x ≤ 2 := 
by sorry

-- Part (2) Statement
theorem range_of_values_for_a (a : ℝ) : 
  (∃ x_0 : ℝ, |x_0 - a| + |x_0 + 2| ≤ |2 * a + 1|) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end solve_inequality_when_a_is_one_range_of_values_for_a_l162_162387


namespace problem_l162_162723

theorem problem (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) : 
  x^3 + 3 * y^2 + 3 * z^2 + 3 * x * y * z = 20 := by
sorry

end problem_l162_162723


namespace find_positive_x_l162_162121

theorem find_positive_x (x y z : ℝ) 
  (h1 : x * y = 15 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 2 * y - 4 * z)
  (h3 : x * z = 56 - 5 * x - 6 * z) : x = 8 := 
sorry

end find_positive_x_l162_162121


namespace sequence_value_238_l162_162643

theorem sequence_value_238 (a : ℕ → ℚ) :
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → (n % 2 = 0 → a n = a (n - 1) / 2 + 1) ∧ (n % 2 = 1 → a n = 1 / a (n - 1))) ∧
  (∃ n, a n = 30 / 19) → ∃ n, a n = 30 / 19 ∧ n = 238 :=
by
  sorry

end sequence_value_238_l162_162643


namespace calc_perm_product_l162_162024

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Lean statement to prove the given problem
theorem calc_perm_product : permutation 6 2 * permutation 4 2 = 360 := 
by
  -- Test the calculations if necessary, otherwise use sorry
  sorry

end calc_perm_product_l162_162024


namespace shorter_side_of_room_l162_162065

theorem shorter_side_of_room
  (P : ℕ) (A : ℕ) (a b : ℕ)
  (perimeter_eq : 2 * a + 2 * b = P)
  (area_eq : a * b = A) (partition_len : ℕ) (partition_cond : partition_len = 5)
  (room_perimeter : P = 60)
  (room_area : A = 200) :
  b = 10 := 
by
  sorry

end shorter_side_of_room_l162_162065


namespace exponential_function_value_l162_162732

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_function_value :
  f (f 2) = 16 := by
  simp only [f]
  sorry

end exponential_function_value_l162_162732


namespace boats_meeting_distance_l162_162237

theorem boats_meeting_distance (X : ℝ) 
  (H1 : ∃ (X : ℝ), (1200 - X) + 900 = X + 1200 + 300) 
  (H2 : X + 1200 + 300 = 2100 + X): 
  X = 300 :=
by
  sorry

end boats_meeting_distance_l162_162237


namespace complement_of_A_in_U_l162_162138

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_U_A : Set ℝ := {x | x ≤ 1 ∨ x > 3}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  simp only [U, A, complement_U_A]
  sorry

end complement_of_A_in_U_l162_162138


namespace median_of_100_numbers_l162_162473

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l162_162473


namespace celine_smartphones_l162_162122

-- Definitions based on the conditions
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops_bought : ℕ := 2
def initial_amount : ℕ := 3000
def change_received : ℕ := 200

-- The proof goal is to show that the number of smartphones bought is 4
theorem celine_smartphones (laptop_cost smartphone_cost num_laptops_bought initial_amount change_received : ℕ)
  (h1 : laptop_cost = 600)
  (h2 : smartphone_cost = 400)
  (h3 : num_laptops_bought = 2)
  (h4 : initial_amount = 3000)
  (h5 : change_received = 200) :
  (initial_amount - change_received - num_laptops_bought * laptop_cost) / smartphone_cost = 4 := 
by
  sorry

end celine_smartphones_l162_162122


namespace probability_green_ball_l162_162475

theorem probability_green_ball 
  (total_balls : ℕ) 
  (green_balls : ℕ) 
  (white_balls : ℕ) 
  (h_total : total_balls = 9) 
  (h_green : green_balls = 7)
  (h_white : white_balls = 2)
  (h_total_eq : total_balls = green_balls + white_balls) : 
  (green_balls / total_balls : ℚ) = 7 / 9 := 
by
  sorry

end probability_green_ball_l162_162475


namespace increasing_interval_f_l162_162729

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3)

theorem increasing_interval_f :
  (∀ x, x ∈ Set.Ioi 3 → f x ∈ Set.Ioi 3) := sorry

end increasing_interval_f_l162_162729


namespace additional_tiles_needed_l162_162405

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end additional_tiles_needed_l162_162405


namespace hyperbola_eccentricity_l162_162159

-- Definitions of conditions
variables {a b c : ℝ}
variables (h : a > 0) (h' : b > 0)
variables (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
variables (parab : ∀ y : ℝ, y^2 = 4 * b * y)
variables (ratio_cond : (b + c) / (c - b) = 5 / 3)

-- Proof statement
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 4 * Real.sqrt 15 / 15 :=
by
  have hyp_foci_distance : ∃ c : ℝ, c^2 = a^2 + b^2 := sorry
  have e := (4 * Real.sqrt 15) / 15
  use e
  sorry

end hyperbola_eccentricity_l162_162159


namespace fizz_preference_count_l162_162687

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end fizz_preference_count_l162_162687


namespace identical_graphs_l162_162111

theorem identical_graphs :
  (∃ (b c : ℝ), (∀ (x y : ℝ), 3 * x + b * y + c = 0 ↔ c * x - 2 * y + 12 = 0) ∧
                 ((b, c) = (1, 6) ∨ (b, c) = (-1, -6))) → ∃ n : ℕ, n = 2 :=
by
  sorry

end identical_graphs_l162_162111


namespace intercepts_equal_lines_parallel_l162_162739

-- Definition of the conditions: line equations
def line_l (a : ℝ) : Prop := ∀ x y : ℝ, a * x + 3 * y + 1 = 0

-- Problem (1) : The intercepts of the line on the two coordinate axes are equal
theorem intercepts_equal (a : ℝ) (h : line_l a) : a = 3 := by
  sorry

-- Problem (2): The line is parallel to x + (a-2)y + a = 0
theorem lines_parallel (a : ℝ) (h : line_l a) : (∀ x y : ℝ, x + (a-2) * y + a = 0) → a = 3 := by
  sorry

end intercepts_equal_lines_parallel_l162_162739


namespace range_of_largest_root_l162_162960

theorem range_of_largest_root :
  ∀ (a_2 a_1 a_0 : ℝ), 
  (|a_2| ≤ 1 ∧ |a_1| ≤ 1 ∧ |a_0| ≤ 1) ∧ (a_2 + a_1 + a_0 = 0) →
  (∃ s > 1, ∀ x > 0, x^3 + 3*a_2*x^2 + 5*a_1*x + a_0 = 0 → x ≤ s) ∧
  (s < 2) :=
by sorry

end range_of_largest_root_l162_162960


namespace inverse_composition_has_correct_value_l162_162864

noncomputable def f (x : ℝ) : ℝ := 5 * x + 7
noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 5

theorem inverse_composition_has_correct_value : 
  f_inv (f_inv 9) = -33 / 25 := 
by 
  sorry

end inverse_composition_has_correct_value_l162_162864


namespace women_in_first_group_l162_162559

-- Define the number of women in the first group as W
variable (W : ℕ)

-- Define the work parameters
def work_per_day := 75 / 8
def work_per_hour_first_group := work_per_day / 5

def work_per_day_second_group := 30 / 3
def work_per_hour_second_group := work_per_day_second_group / 8

-- The equation comes from work/hour equivalence
theorem women_in_first_group :
  (W : ℝ) * work_per_hour_first_group = 4 * work_per_hour_second_group → W = 5 :=
by 
  sorry

end women_in_first_group_l162_162559


namespace position_after_steps_l162_162823

def equally_spaced_steps (total_distance num_steps distance_per_step steps_taken : ℕ) : Prop :=
  total_distance = num_steps * distance_per_step ∧ 
  ∀ k : ℕ, k ≤ num_steps → k * distance_per_step = distance_per_step * k

theorem position_after_steps (total_distance num_steps distance_per_step steps_taken : ℕ) 
  (h_eq : equally_spaced_steps total_distance num_steps distance_per_step steps_taken) 
  (h_total : total_distance = 32) (h_num : num_steps = 8) (h_steps : steps_taken = 6) : 
  steps_taken * (total_distance / num_steps) = 24 := 
by 
  sorry

end position_after_steps_l162_162823


namespace smallest_y_l162_162918

theorem smallest_y (y : ℕ) :
  (y > 0 ∧ 800 ∣ (540 * y)) ↔ (y = 40) :=
by
  sorry

end smallest_y_l162_162918


namespace arithmetic_mean_of_multiples_of_6_l162_162689

/-- The smallest three-digit multiple of 6 is 102. -/
def smallest_multiple_of_6 : ℕ := 102

/-- The largest three-digit multiple of 6 is 996. -/
def largest_multiple_of_6 : ℕ := 996

/-- The common difference in the arithmetic sequence of multiples of 6 is 6. -/
def common_difference_of_sequence : ℕ := 6

/-- The number of terms in the arithmetic sequence of three-digit multiples of 6. -/
def number_of_terms : ℕ := (largest_multiple_of_6 - smallest_multiple_of_6) / common_difference_of_sequence + 1

/-- The sum of the arithmetic sequence of three-digit multiples of 6. -/
def sum_of_sequence : ℕ := number_of_terms * (smallest_multiple_of_6 + largest_multiple_of_6) / 2

/-- The arithmetic mean of all positive three-digit multiples of 6 is 549. -/
theorem arithmetic_mean_of_multiples_of_6 : 
  let mean := sum_of_sequence / number_of_terms
  mean = 549 :=
by
  sorry

end arithmetic_mean_of_multiples_of_6_l162_162689


namespace range_of_m_l162_162331

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (1 / y) = 1) (h2 : x + 2 * y > m^2 + 2 * m) : -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l162_162331


namespace rowing_upstream_speed_l162_162860

def speed_in_still_water : ℝ := 31
def speed_downstream : ℝ := 37

def speed_stream : ℝ := speed_downstream - speed_in_still_water

def speed_upstream : ℝ := speed_in_still_water - speed_stream

theorem rowing_upstream_speed :
  speed_upstream = 25 := by
  sorry

end rowing_upstream_speed_l162_162860


namespace tan_value_l162_162450

theorem tan_value (x : ℝ) (hx : x ∈ Set.Ioo (-π / 2) 0) (hcos : Real.cos x = 4 / 5) : Real.tan x = -3 / 4 :=
sorry

end tan_value_l162_162450


namespace water_tank_full_capacity_l162_162985

theorem water_tank_full_capacity (x : ℝ) (h1 : x * (3/4) - x * (1/3) = 15) : x = 36 := 
by
  sorry

end water_tank_full_capacity_l162_162985


namespace triangle_perimeter_l162_162695

-- Definitions and given conditions
def side_length_a (a : ℝ) : Prop := a = 6
def inradius (r : ℝ) : Prop := r = 2
def circumradius (R : ℝ) : Prop := R = 5

-- The final proof statement to be proven
theorem triangle_perimeter (a r R : ℝ) (b c P : ℝ) 
  (h1 : side_length_a a)
  (h2 : inradius r)
  (h3 : circumradius R)
  (h4 : P = 2 * ((a + b + c) / 2)) :
  P = 24 :=
sorry

end triangle_perimeter_l162_162695


namespace new_pressure_eq_l162_162675

-- Defining the initial conditions and values
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3.5
def new_volume : ℝ := 10.5
def k : ℝ := initial_pressure * initial_volume

-- The statement to prove
theorem new_pressure_eq :
  ∃ p_new : ℝ, new_volume * p_new = k ∧ p_new = 8 / 3 :=
by
  use (8 / 3)
  sorry

end new_pressure_eq_l162_162675


namespace geometric_series_sum_l162_162380

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℝ) / 5
  ∑' n : ℕ, a * r ^ n = 5 / 4 :=
by
  sorry

end geometric_series_sum_l162_162380


namespace distinct_pairs_l162_162130

-- Definitions of rational numbers and distinctness.
def is_distinct (x y : ℚ) : Prop := x ≠ y

-- Conditions
variables {a b r s : ℚ}

-- Main theorem: prove that there is only 1 distinct pair (a, b)
theorem distinct_pairs (h_ab_distinct : is_distinct a b)
  (h_rs_distinct : is_distinct r s)
  (h_eq : ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s)) : 
    ∃! (a b : ℚ), ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s) :=
  sorry

end distinct_pairs_l162_162130


namespace smallest_b_no_inverse_mod75_and_mod90_l162_162058

theorem smallest_b_no_inverse_mod75_and_mod90 :
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, n > 0 → n < b →  ¬ (n.gcd 75 > 1 ∧ n.gcd 90 > 1)) ∧ 
  (b.gcd 75 > 1 ∧ b.gcd 90 > 1) ∧ 
  b = 15 := 
by
  sorry

end smallest_b_no_inverse_mod75_and_mod90_l162_162058


namespace find_E_l162_162806

theorem find_E (A H S M E : ℕ) (h1 : A ≠ 0) (h2 : H ≠ 0) (h3 : S ≠ 0) (h4 : M ≠ 0) (h5 : E ≠ 0) 
  (cond1 : A + H = E)
  (cond2 : S + M = E)
  (cond3 : E = (A * M - S * H) / (M - H)) : 
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end find_E_l162_162806


namespace general_formula_l162_162584

def a (n : ℕ) : ℕ :=
match n with
| 0 => 1
| k+1 => 2 * a k + 4

theorem general_formula (n : ℕ) : a (n+1) = 5 * 2^n - 4 :=
by
  sorry

end general_formula_l162_162584


namespace walk_to_bus_stop_usual_time_l162_162673

variable (S : ℝ) -- assuming S is the usual speed, a positive real number
variable (T : ℝ) -- assuming T is the usual time, which we need to determine
variable (new_speed : ℝ := (4 / 5) * S) -- the new speed is 4/5 of usual speed
noncomputable def time_to_bus_at_usual_speed : ℝ := T -- time to bus stop at usual speed

theorem walk_to_bus_stop_usual_time :
  (time_to_bus_at_usual_speed S = 30) ↔ (S * (T + 6) = (4 / 5) * S * T) :=
by
  sorry

end walk_to_bus_stop_usual_time_l162_162673


namespace carson_clawed_total_l162_162836

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end carson_clawed_total_l162_162836


namespace sunglasses_and_cap_probability_l162_162663

/-
On a beach:
  - 50 people are wearing sunglasses.
  - 35 people are wearing caps.
  - The probability that randomly selected person wearing a cap is also wearing sunglasses is 2/5.
  
Prove that the probability that a randomly selected person wearing sunglasses is also wearing a cap is 7/25.
-/

theorem sunglasses_and_cap_probability :
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * total_caps
  (both / total_sunglasses) = (7 : ℚ) / 25 :=
by
  -- definitions
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * (total_caps : ℚ)
  have prob : (both / (total_sunglasses : ℚ)) = (7 : ℚ) / 25 := sorry
  exact prob

end sunglasses_and_cap_probability_l162_162663


namespace annual_cost_l162_162224

def monday_miles : ℕ := 50
def wednesday_miles : ℕ := 50
def friday_miles : ℕ := 50
def sunday_miles : ℕ := 50

def tuesday_miles : ℕ := 100
def thursday_miles : ℕ := 100
def saturday_miles : ℕ := 100

def cost_per_mile : ℝ := 0.1
def weekly_fee : ℝ := 100
def weeks_in_year : ℕ := 52

noncomputable def total_weekly_miles : ℕ := 
  (monday_miles + wednesday_miles + friday_miles + sunday_miles) * 1 +
  (tuesday_miles + thursday_miles + saturday_miles) * 1

noncomputable def weekly_mileage_cost : ℝ := total_weekly_miles * cost_per_mile

noncomputable def weekly_total_cost : ℝ := weekly_fee + weekly_mileage_cost

noncomputable def annual_total_cost : ℝ := weekly_total_cost * weeks_in_year

theorem annual_cost (monday_miles wednesday_miles friday_miles sunday_miles
                     tuesday_miles thursday_miles saturday_miles : ℕ)
                     (cost_per_mile weekly_fee : ℝ) 
                     (weeks_in_year : ℕ) :
  monday_miles = 50 → wednesday_miles = 50 → friday_miles = 50 → sunday_miles = 50 →
  tuesday_miles = 100 → thursday_miles = 100 → saturday_miles = 100 →
  cost_per_mile = 0.1 → weekly_fee = 100 → weeks_in_year = 52 →
  annual_total_cost = 7800 :=
by
  intros
  sorry

end annual_cost_l162_162224


namespace solve_equation_l162_162661

theorem solve_equation (x : ℝ) :
  ((x - 2)^2 - 4 = 0) ↔ (x = 4 ∨ x = 0) :=
by
  sorry

end solve_equation_l162_162661


namespace dmitry_black_socks_l162_162248

theorem dmitry_black_socks :
  let blue_socks := 10
  let initial_black_socks := 22
  let white_socks := 12
  let total_initial_socks := blue_socks + initial_black_socks + white_socks
  ∀ x : ℕ,
    let total_socks := total_initial_socks + x
    let black_socks := initial_black_socks + x
    (black_socks : ℚ) / (total_socks : ℚ) = 2 / 3 → x = 22 :=
by
  sorry

end dmitry_black_socks_l162_162248


namespace alpha_values_l162_162382

noncomputable def α := Complex

theorem alpha_values (α : Complex) :
  (α ≠ 1) ∧ 
  (Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1)) ∧ 
  (Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) ∧ 
  (Real.cos α.arg = 1 / 2) →
  α = Complex.mk ((-1 + Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 + Real.sqrt 33) / 4)^2))) ∨ 
  α = Complex.mk ((-1 - Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 - Real.sqrt 33) / 4)^2))) :=
sorry

end alpha_values_l162_162382


namespace gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l162_162574

theorem gcd_b_squared_plus_11b_plus_28_and_b_plus_6 (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l162_162574


namespace virginia_avg_rainfall_l162_162884

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end virginia_avg_rainfall_l162_162884


namespace union_intersection_l162_162083

-- Define the sets M, N, and P
def M := ({1} : Set Nat)
def N := ({1, 2} : Set Nat)
def P := ({1, 2, 3} : Set Nat)

-- Prove that (M ∪ N) ∩ P = {1, 2}
theorem union_intersection : (M ∪ N) ∩ P = ({1, 2} : Set Nat) := 
by 
  sorry

end union_intersection_l162_162083


namespace sin_cos_identity_l162_162727

theorem sin_cos_identity (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) : Real.sin x + 5 * Real.cos x = -28 / 13 := 
  sorry

end sin_cos_identity_l162_162727


namespace average_price_per_racket_l162_162493

theorem average_price_per_racket (total_amount : ℕ) (pairs_sold : ℕ) (expected_average : ℚ) 
  (h1 : total_amount = 637) (h2 : pairs_sold = 65) : 
  expected_average = total_amount / pairs_sold := 
by
  sorry

end average_price_per_racket_l162_162493


namespace room_length_l162_162718

theorem room_length (L : ℕ) (h : 72 * L + 918 = 2718) : L = 25 := by
  sorry

end room_length_l162_162718


namespace part1_3kg_part2_5kg_part2_function_part3_compare_l162_162686

noncomputable def supermarket_A_cost (x : ℝ) : ℝ :=
if x <= 4 then 10 * x
else 6 * x + 16

noncomputable def supermarket_B_cost (x : ℝ) : ℝ :=
8 * x

-- Proof that supermarket_A_cost 3 = 30
theorem part1_3kg : supermarket_A_cost 3 = 30 :=
by sorry

-- Proof that supermarket_A_cost 5 = 46
theorem part2_5kg : supermarket_A_cost 5 = 46 :=
by sorry

-- Proof that the cost function is correct
theorem part2_function (x : ℝ) : 
(0 < x ∧ x <= 4 → supermarket_A_cost x = 10 * x) ∧ 
(x > 4 → supermarket_A_cost x = 6 * x + 16) :=
by sorry

-- Proof that supermarket A is cheaper for 10 kg apples
theorem part3_compare : supermarket_A_cost 10 < supermarket_B_cost 10 :=
by sorry

end part1_3kg_part2_5kg_part2_function_part3_compare_l162_162686


namespace max_value_min_expression_l162_162933

def f (x y : ℝ) : ℝ :=
  x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_value_min_expression (a b c : ℝ) (h₁: a ≠ b) (h₂: b ≠ c) (h₃: c ≠ a)
  (hab : f a b = f b c) (hbc : f b c = f c a) :
  (max (min (a^4 - 4*a^3 + 4*a^2) (min (b^4 - 4*b^3 + 4*b^2) (c^4 - 4*c^3 + 4*c^2))) 1) = 1 :=
sorry

end max_value_min_expression_l162_162933


namespace solve_equation_l162_162195

theorem solve_equation :
  ∀ x : ℝ, (x * (2 * x + 4) = 10 + 5 * x) ↔ (x = -2 ∨ x = 2.5) :=
by
  sorry

end solve_equation_l162_162195


namespace ext_9_implication_l162_162778

theorem ext_9_implication (a b : ℝ) (h1 : 3 + 2 * a + b = 0) (h2 : 1 + a + b + a^2 = 10) : (2 : ℝ)^3 + a * (2 : ℝ)^2 + b * (2 : ℝ) + a^2 - 1 = 17 := by
  sorry

end ext_9_implication_l162_162778


namespace polygon_sides_l162_162189

theorem polygon_sides (n : ℕ) 
  (h_interior : (n - 2) * 180 = 4 * 360) : 
  n = 10 :=
by
  sorry

end polygon_sides_l162_162189


namespace month_days_l162_162384

theorem month_days (letters_per_day packages_per_day total_mail six_months : ℕ) (h1 : letters_per_day = 60) (h2 : packages_per_day = 20) (h3 : total_mail = 14400) (h4 : six_months = 6) : 
  total_mail / (letters_per_day + packages_per_day) / six_months = 30 :=
by sorry

end month_days_l162_162384


namespace expected_waiting_time_approx_l162_162033

noncomputable def expectedWaitingTime : ℚ :=
  (10 * (1/2) + 30 * (1/3) + 50 * (1/36) + 70 * (1/12) + 90 * (1/18))

theorem expected_waiting_time_approx :
  abs (expectedWaitingTime - 27.22) < 1 :=
by
  sorry

end expected_waiting_time_approx_l162_162033


namespace skateboarded_one_way_distance_l162_162598

-- Define the total skateboarded distance and the walked distance.
def total_skateboarded : ℕ := 24
def walked_distance : ℕ := 4

-- Define the proof theorem.
theorem skateboarded_one_way_distance : 
    (total_skateboarded - walked_distance) / 2 = 10 := 
by sorry

end skateboarded_one_way_distance_l162_162598


namespace base_conversion_arithmetic_l162_162792

theorem base_conversion_arithmetic :
  let b5 := 2013
  let b3 := 11
  let b6 := 3124
  let b7 := 4321
  (b5₅ / b3₃ - b6₆ + b7₇ : ℝ) = 898.5 :=
by sorry

end base_conversion_arithmetic_l162_162792


namespace divisible_by_5_l162_162791

theorem divisible_by_5 (x y : ℕ) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 5 ∣ x := sorry

end divisible_by_5_l162_162791


namespace cos_triple_angle_l162_162934

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l162_162934


namespace number_of_bugs_seen_l162_162428

-- Defining the conditions
def flowers_per_bug : ℕ := 2
def total_flowers_eaten : ℕ := 6

-- The statement to prove
theorem number_of_bugs_seen : total_flowers_eaten / flowers_per_bug = 3 :=
by
  sorry

end number_of_bugs_seen_l162_162428


namespace A_share_in_profit_l162_162613

-- Define the investments and profits
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12200

-- Define the total investment
def total_investment : ℕ := A_investment + B_investment + C_investment

-- Define A's ratio in the investment
def A_ratio : ℚ := A_investment / total_investment

-- Define A's share in the profit
def A_share : ℚ := total_profit * A_ratio

-- The theorem to prove
theorem A_share_in_profit : A_share = 3660 := by
  sorry

end A_share_in_profit_l162_162613


namespace earnings_per_visit_l162_162561

-- Define the conditions of the problem
def website_visits_per_month : ℕ := 30000
def earning_per_day : Real := 10
def days_in_month : ℕ := 30

-- Prove that John gets $0.01 per visit
theorem earnings_per_visit :
  (earning_per_day * days_in_month) / website_visits_per_month = 0.01 :=
by
  sorry

end earnings_per_visit_l162_162561


namespace f_above_g_l162_162998

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) / (x - m)
def g (x : ℝ) : ℝ := x^2 + x

theorem f_above_g (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  ∀ x, m ≤ x ∧ x ≤ m + 1 → f x m > g x := 
sorry

end f_above_g_l162_162998


namespace factor_by_resultant_l162_162774

theorem factor_by_resultant (x f : ℤ) (h1 : x = 17) (h2 : (2 * x + 5) * f = 117) : f = 3 := 
by
  sorry

end factor_by_resultant_l162_162774


namespace range_of_a_l162_162019

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * a * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l162_162019


namespace find_b_from_conditions_l162_162662

theorem find_b_from_conditions (x y z k : ℝ) (h1 : (x + y) / 2 = k) (h2 : (z + x) / 3 = k) (h3 : (y + z) / 4 = k) (h4 : x + y + z = 36) : x + y = 16 := 
by 
  sorry

end find_b_from_conditions_l162_162662


namespace smallest_n_divisible_by_100_million_l162_162231

noncomputable def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

noncomputable def nth_term (a1 r : ℚ) (n : ℕ) : ℚ := a1 * r^(n - 1)

theorem smallest_n_divisible_by_100_million :
  ∀ (a1 a2 : ℚ), a1 = 5/6 → a2 = 25 → 
  ∃ n : ℕ, nth_term a1 (common_ratio a1 a2) n % 100000000 = 0 ∧ n = 9 :=
by
  intros a1 a2 h1 h2
  have r := common_ratio a1 a2
  have a9 := nth_term a1 r 9
  sorry

end smallest_n_divisible_by_100_million_l162_162231


namespace fixed_point_for_line_l162_162965

theorem fixed_point_for_line (m : ℝ) : (m * (1 - 1) + (1 - 1) = 0) :=
by
  sorry

end fixed_point_for_line_l162_162965


namespace sufficient_but_not_necessary_l162_162999

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 3| - |x - 1| < 2) → x ≠ 1 ∧ ¬ (∀ x : ℝ, x ≠ 1 → |x - 3| - |x - 1| < 2) :=
by
  sorry

end sufficient_but_not_necessary_l162_162999


namespace remaining_card_number_l162_162974

theorem remaining_card_number (A B C D E F G H : ℕ) (cards : Finset ℕ) 
  (hA : A + B = 10) 
  (hB : C - D = 1) 
  (hC : E * F = 24) 
  (hD : G / H = 3) 
  (hCards : cards = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hDistinct : A ∉ cards ∧ B ∉ cards ∧ C ∉ cards ∧ D ∉ cards ∧ E ∉ cards ∧ F ∉ cards ∧ G ∉ cards ∧ H ∉ cards) :
  7 ∈ cards := 
by
  sorry

end remaining_card_number_l162_162974


namespace tennis_tournament_boxes_needed_l162_162407

theorem tennis_tournament_boxes_needed (n : ℕ) (h : n = 199) : 
  ∃ m, m = 198 ∧
    (∀ k, k < n → (n - k - 1 = m)) :=
by
  sorry

end tennis_tournament_boxes_needed_l162_162407


namespace angle_PMN_is_60_l162_162968

-- Define given variables and their types
variable (P M N R Q : Prop)
variable (angle : Prop → Prop → Prop → ℝ)

-- Given conditions
variables (h1 : angle P Q R = 60)
variables (h2 : PM = MN)

-- The statement of what's to be proven
theorem angle_PMN_is_60 :
  angle P M N = 60 := sorry

end angle_PMN_is_60_l162_162968


namespace unused_types_l162_162750

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end unused_types_l162_162750


namespace problem_statement_l162_162688

theorem problem_statement (x : ℝ) (h : x ≠ 2) :
  (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ (1 ≤ x ∧ x < 2) ∨ (32/7 < x) :=
by 
  sorry

end problem_statement_l162_162688


namespace set_contains_all_nonnegative_integers_l162_162940

theorem set_contains_all_nonnegative_integers (S : Set ℕ) :
  (∃ a b, a ∈ S ∧ b ∈ S ∧ 1 < a ∧ 1 < b ∧ Nat.gcd a b = 1) →
  (∀ x y, x ∈ S → y ∈ S → y ≠ 0 → (x * y) ∈ S ∧ (x % y) ∈ S) →
  (∀ n, n ∈ S) :=
by
  intros h1 h2
  sorry

end set_contains_all_nonnegative_integers_l162_162940


namespace possible_denominators_count_l162_162114

theorem possible_denominators_count :
  ∀ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) →
  ∃ (D : Finset ℕ), D.card = 7 ∧ 
  ∀ num denom, (num = 100*a + 10*b + c) → (denom = 999) → (gcd num denom > 1) → 
  denom ∈ D := 
sorry

end possible_denominators_count_l162_162114


namespace find_matrix_l162_162364

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M^3 - 3 * M^2 + 2 * M = ![![8, 16], ![4, 8]]) : 
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_l162_162364


namespace mass_percentage_Ca_in_mixture_l162_162485

theorem mass_percentage_Ca_in_mixture :
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  percentage_Ca = 26.69 :=
by
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  have : percentage_Ca = 26.69 := by sorry
  exact this

end mass_percentage_Ca_in_mixture_l162_162485


namespace half_height_of_triangular_prism_l162_162432

theorem half_height_of_triangular_prism (volume base_area height : ℝ) 
  (h_volume : volume = 576)
  (h_base_area : base_area = 3)
  (h_prism : volume = base_area * height) :
  height / 2 = 96 :=
by
  have h : height = volume / base_area := by sorry
  rw [h_volume, h_base_area] at h
  have h_height : height = 192 := by sorry
  rw [h_height]
  norm_num

end half_height_of_triangular_prism_l162_162432


namespace sticker_price_of_smartphone_l162_162223

theorem sticker_price_of_smartphone (p : ℝ)
  (h1 : 0.90 * p - 100 = 0.80 * p - 20) : p = 800 :=
sorry

end sticker_price_of_smartphone_l162_162223


namespace solve_y_l162_162090

theorem solve_y : ∃ y : ℚ, 2 * y + 3 * y = 600 - (4 * y + 5 * y + 100) ∧ y = 250 / 7 := by
  sorry

end solve_y_l162_162090


namespace range_frequency_l162_162357

-- Define the sample data
def sample_data : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

-- Define the condition representing the frequency count
def frequency_count : ℝ := 0.2 * 20

-- Define the proof problem
theorem range_frequency (s : List ℝ) (range_start range_end : ℝ) : 
  s = sample_data → 
  range_start = 11.5 →
  range_end = 13.5 → 
  (s.filter (λ x => range_start ≤ x ∧ x < range_end)).length = frequency_count := 
by 
  intros
  sorry

end range_frequency_l162_162357


namespace harry_worked_41_hours_l162_162684

def james_earnings (x : ℝ) : ℝ :=
  (40 * x) + (7 * 2 * x)

def harry_earnings (x : ℝ) (h : ℝ) : ℝ :=
  (24 * x) + (11 * 1.5 * x) + (2 * h * x)

def harry_hours_worked (h : ℝ) : ℝ :=
  24 + 11 + h

theorem harry_worked_41_hours (x : ℝ) (h : ℝ) 
  (james_worked : james_earnings x = 54 * x)
  (harry_paid_same : harry_earnings x h = james_earnings x) :
  harry_hours_worked h = 41 :=
by
  -- sorry is used to skip the proof steps
  sorry

end harry_worked_41_hours_l162_162684


namespace hyperbola_center_l162_162161

theorem hyperbola_center (x y : ℝ) :
  ∃ h k : ℝ, (∃ a b : ℝ, a = 9/4 ∧ b = 7/2) ∧ (h, k) = (-2, 3) ∧ 
  (4*x + 8)^2 / 81 - (2*y - 6)^2 / 49 = 1 :=
by
  sorry

end hyperbola_center_l162_162161


namespace parallel_lines_m_eq_one_l162_162075

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y + (m - 2) = 0 ∧ 2 * m * x + 4 * y + 16 = 0 → m = 1) :=
by
  sorry

end parallel_lines_m_eq_one_l162_162075


namespace sqrt_difference_square_l162_162067

theorem sqrt_difference_square (a b : ℝ) (h₁ : a = Real.sqrt 3 + Real.sqrt 2) (h₂ : b = Real.sqrt 3 - Real.sqrt 2) : a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end sqrt_difference_square_l162_162067


namespace count_positive_integers_l162_162575

theorem count_positive_integers (n : ℕ) : ∃ k : ℕ, k = 9 ∧  ∀ n, 1 ≤ n → n < 10 → 3 * n + 20 < 50 :=
by
  sorry

end count_positive_integers_l162_162575


namespace max_sequence_term_value_l162_162535

def a_n (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem max_sequence_term_value : ∃ n : ℕ, a_n n = 108 := 
sorry

end max_sequence_term_value_l162_162535


namespace inequality_holds_l162_162500

theorem inequality_holds (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4)
  (h5 : 0 < x5) (h6 : 0 < x6) (h7 : 0 < x7) (h8 : 0 < x8) 
  (h9 : 0 < x9) :
  (x1 - x3) / (x1 * x3 + 2 * x2 * x3 + x2^2) +
  (x2 - x4) / (x2 * x4 + 2 * x3 * x4 + x3^2) +
  (x3 - x5) / (x3 * x5 + 2 * x4 * x5 + x4^2) +
  (x4 - x6) / (x4 * x6 + 2 * x5 * x6 + x5^2) +
  (x5 - x7) / (x5 * x7 + 2 * x6 * x7 + x6^2) +
  (x6 - x8) / (x6 * x8 + 2 * x7 * x8 + x7^2) +
  (x7 - x9) / (x7 * x9 + 2 * x8 * x9 + x8^2) +
  (x8 - x1) / (x8 * x1 + 2 * x9 * x1 + x9^2) +
  (x9 - x2) / (x9 * x2 + 2 * x1 * x2 + x1^2) ≥ 0 := 
sorry

end inequality_holds_l162_162500


namespace zain_coin_total_l162_162448

def zain_coins (q d n : ℕ) := q + d + n
def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def zain_quarters : ℕ := emerie_quarters + 10
def zain_dimes : ℕ := emerie_dimes + 10
def zain_nickels : ℕ := emerie_nickels + 10

theorem zain_coin_total : zain_coins zain_quarters zain_dimes zain_nickels = 48 := 
by
  unfold zain_coins zain_quarters zain_dimes zain_nickels emerie_quarters emerie_dimes emerie_nickels
  rfl

end zain_coin_total_l162_162448


namespace number_of_teachers_l162_162499

theorem number_of_teachers
  (T S : ℕ)
  (h1 : T + S = 2400)
  (h2 : 320 = 320) -- This condition is trivial and can be ignored
  (h3 : 280 = 280) -- This condition is trivial and can be ignored
  (h4 : S / 280 = T / 40) : T = 300 :=
by
  sorry

end number_of_teachers_l162_162499


namespace computation_l162_162616

def g (x : ℕ) : ℕ := 7 * x - 3

theorem computation : g (g (g (g 1))) = 1201 := by
  sorry

end computation_l162_162616


namespace Jillian_largest_apartment_size_l162_162937

noncomputable def largest_apartment_size (budget rent_per_sqft: ℝ) : ℝ :=
  budget / rent_per_sqft

theorem Jillian_largest_apartment_size :
  largest_apartment_size 720 1.20 = 600 := 
by
  sorry

end Jillian_largest_apartment_size_l162_162937


namespace part1_infinite_n_part2_no_solutions_l162_162955

-- Definitions for part (1)
theorem part1_infinite_n (n : ℕ) (x y z t : ℕ) :
  (∃ n, x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

-- Definitions for part (2)
theorem part2_no_solutions (n k m x y z t : ℕ) :
  n = 4 ^ k * (8 * m + 7) → ¬(x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

end part1_infinite_n_part2_no_solutions_l162_162955


namespace area_of_shaded_region_l162_162862

def radius_of_first_circle : ℝ := 4
def radius_of_second_circle : ℝ := 5
def radius_of_third_circle : ℝ := 2
def radius_of_fourth_circle : ℝ := 9

theorem area_of_shaded_region :
  π * (radius_of_fourth_circle ^ 2) - π * (radius_of_first_circle ^ 2) - π * (radius_of_second_circle ^ 2) - π * (radius_of_third_circle ^ 2) = 36 * π :=
by {
  sorry
}

end area_of_shaded_region_l162_162862


namespace probability_of_exactly_nine_correct_placements_is_zero_l162_162916

-- Define the number of letters and envelopes
def num_letters : ℕ := 10

-- Define the condition of letters being randomly inserted into envelopes
def random_insertion (n : ℕ) : Prop := true

-- Prove that the probability of exactly nine letters being correctly placed is zero
theorem probability_of_exactly_nine_correct_placements_is_zero
  (h : random_insertion num_letters) : 
  (∃ p : ℝ, p = 0) := 
sorry

end probability_of_exactly_nine_correct_placements_is_zero_l162_162916


namespace problem_l162_162218

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : nabla (nabla 1 3) 2 = 67 :=
by
  sorry

end problem_l162_162218


namespace four_pow_sub_divisible_iff_l162_162609

open Nat

theorem four_pow_sub_divisible_iff (m n k : ℕ) (h₁ : m > n) : 
  (3^(k + 1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
by sorry

end four_pow_sub_divisible_iff_l162_162609


namespace inequality_solution_l162_162310

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | ax ^ 2 - (a + 1) * x + 1 < 0} =
    if a = 1 then ∅
    else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
    else if a > 1 then {x : ℝ | 1 / a < x ∧ x < 1} 
    else ∅ := sorry

end inequality_solution_l162_162310


namespace sequence_eventually_periodic_l162_162098

-- Definitions based on the conditions
def positive_int_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < a n

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)

-- Assertion to prove based on the question
theorem sequence_eventually_periodic (a : ℕ → ℕ) 
  (h1 : positive_int_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ p : ℕ, ∃ k : ℕ, ∀ n : ℕ, a (n + k) = a n :=
sorry

end sequence_eventually_periodic_l162_162098


namespace students_count_l162_162123

theorem students_count (n : ℕ) (avg_age_n_students : ℕ) (sum_age_7_students1 : ℕ) (sum_age_7_students2 : ℕ) (last_student_age : ℕ) :
  avg_age_n_students = 15 →
  sum_age_7_students1 = 7 * 14 →
  sum_age_7_students2 = 7 * 16 →
  last_student_age = 15 →
  (sum_age_7_students1 + sum_age_7_students2 + last_student_age = avg_age_n_students * n) →
  n = 15 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end students_count_l162_162123


namespace midpoint_AB_l162_162655

noncomputable def s (x t : ℝ) : ℝ := (x + t)^2 + (x - t)^2

noncomputable def CP (x : ℝ) : ℝ := x * Real.sqrt 3 / 2

theorem midpoint_AB (x : ℝ) (P : ℝ) : 
    (s x 0 = 2 * CP x ^ 2) ↔ P = x :=
by
    sorry

end midpoint_AB_l162_162655


namespace function_increasing_on_interval_l162_162861

theorem function_increasing_on_interval {x : ℝ} (hx : x < 1) : 
  (-1/2) * x^2 + x + 4 < -1/2 * (x + 1)^2 + (x + 1) + 4 :=
sorry

end function_increasing_on_interval_l162_162861


namespace sum_of_all_possible_values_of_g10_l162_162938

noncomputable def g : ℕ → ℝ := sorry

axiom h1 : g 1 = 2
axiom h2 : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = 3 * (g m + g n)
axiom h3 : g 0 = 0

theorem sum_of_all_possible_values_of_g10 : g 10 = 59028 :=
by
  sorry

end sum_of_all_possible_values_of_g10_l162_162938


namespace increasing_range_of_a_l162_162736

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1 / x

theorem increasing_range_of_a (a : ℝ) : (∀ x > (1/2), (3 * x^2 + a - 1 / x^2) ≥ 0) ↔ a ≥ (13 / 4) :=
by sorry

end increasing_range_of_a_l162_162736


namespace range_of_m_for_inequality_l162_162601

theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, |x-1| + |x+m| > 3} = {m : ℝ | m < -4 ∨ m > 2} :=
sorry

end range_of_m_for_inequality_l162_162601


namespace max_f_on_interval_l162_162941

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x

theorem max_f_on_interval : 
  ∃ x ∈ Set.Icc (2 * Real.pi / 5) (3 * Real.pi / 4), f x = (1 + Real.sqrt 2) / 2 :=
by
  sorry

end max_f_on_interval_l162_162941


namespace find_greatest_divisor_l162_162931

def greatest_divisor_leaving_remainders (n₁ n₁_r n₂ n₂_r d : ℕ) : Prop :=
  (n₁ % d = n₁_r) ∧ (n₂ % d = n₂_r) 

theorem find_greatest_divisor :
  greatest_divisor_leaving_remainders 1657 10 2037 7 1 :=
by
  sorry

end find_greatest_divisor_l162_162931


namespace wall_length_to_height_ratio_l162_162789

theorem wall_length_to_height_ratio
  (W H L : ℝ)
  (V : ℝ)
  (h1 : H = 6 * W)
  (h2 : L * H * W = V)
  (h3 : V = 86436)
  (h4 : W = 6.999999999999999) :
  L / H = 7 :=
by
  sorry

end wall_length_to_height_ratio_l162_162789


namespace perimeter_of_original_square_l162_162073

-- Definitions
variables {x : ℝ}
def rect_width := x
def rect_length := 4 * x
def rect_perimeter := 56
def original_square_perimeter := 32

-- Statement
theorem perimeter_of_original_square (x : ℝ) (h : 28 * x = 56) : 4 * (4 * x) = 32 :=
by
  -- Since the proof is not required, we apply sorry to end the theorem.
  sorry

end perimeter_of_original_square_l162_162073


namespace gcd_g_50_52_l162_162853

/-- Define the polynomial function g -/
def g (x : ℤ) : ℤ := x^2 - 3 * x + 2023

/-- The theorem stating the gcd of g(50) and g(52) -/
theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g_50_52_l162_162853


namespace fraction_addition_target_l162_162503

open Rat

theorem fraction_addition_target (n : ℤ) : 
  (4 + n) / (7 + n) = 3 / 4 → 
  n = 5 := 
by
  intro h
  sorry

end fraction_addition_target_l162_162503


namespace trisha_total_distance_l162_162281

-- Define each segment of Trisha's walk in miles
def hotel_to_postcard : ℝ := 0.1111111111111111
def postcard_to_tshirt : ℝ := 0.2222222222222222
def tshirt_to_keychain : ℝ := 0.7777777777777778
def keychain_to_toy : ℝ := 0.5555555555555556
def meters_to_miles (m : ℝ) : ℝ := m * 0.000621371
def toy_to_bookstore : ℝ := meters_to_miles 400
def bookstore_to_hotel : ℝ := 0.6666666666666666

-- Sum of all distances
def total_distance : ℝ :=
  hotel_to_postcard +
  postcard_to_tshirt +
  tshirt_to_keychain +
  keychain_to_toy +
  toy_to_bookstore +
  bookstore_to_hotel

-- Proof statement
theorem trisha_total_distance : total_distance = 1.5818817333333333 := by
  sorry

end trisha_total_distance_l162_162281


namespace simultaneous_solution_exists_l162_162286

-- Definitions required by the problem
def eqn1 (m x : ℝ) : ℝ := m * x + 2
def eqn2 (m x : ℝ) : ℝ := (3 * m - 2) * x + 5

-- Proof statement
theorem simultaneous_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = eqn1 m x ∧ y = eqn2 m x) ↔ (m ≠ 1) := 
sorry

end simultaneous_solution_exists_l162_162286


namespace units_digit_7_pow_6_pow_5_l162_162828

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l162_162828


namespace odd_function_m_value_l162_162851

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x - m

theorem odd_function_m_value :
  ∃ m : ℝ, (∀ (x : ℝ), g (-x) m + g x m = 0) ∧ m = 2 :=
by
  sorry

end odd_function_m_value_l162_162851


namespace deal_or_no_deal_min_eliminations_l162_162909

theorem deal_or_no_deal_min_eliminations (n_boxes : ℕ) (n_high_value : ℕ) 
    (initial_count : n_boxes = 26)
    (high_value_count : n_high_value = 9) :
  ∃ (min_eliminations : ℕ), min_eliminations = 8 ∧
    ((n_boxes - min_eliminations - 1) / 2) ≥ n_high_value :=
sorry

end deal_or_no_deal_min_eliminations_l162_162909


namespace area_of_triangle_QCA_l162_162510

noncomputable def triangle_area (x p : ℝ) (hx : x > 0) (hp : p < 12) : ℝ :=
  1 / 2 * x * (12 - p)

theorem area_of_triangle_QCA (x p : ℝ) (hx : x > 0) (hp : p < 12) :
  triangle_area x p hx hp = x * (12 - p) / 2 := by
  sorry

end area_of_triangle_QCA_l162_162510


namespace range_of_k_l162_162716

noncomputable def f (k x : ℝ) := (k * x + 7) / (k * x^2 + 4 * k * x + 3)

theorem range_of_k (k : ℝ) : (∀ x : ℝ, k * x^2 + 4 * k * x + 3 ≠ 0) ↔ 0 ≤ k ∧ k < 3 / 4 :=
by
  sorry

end range_of_k_l162_162716


namespace negation_of_existence_l162_162210

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_existence_l162_162210


namespace find_length_of_room_l162_162885

def length_of_room (L : ℕ) (width verandah_width verandah_area : ℕ) : Prop :=
  (L + 2 * verandah_width) * (width + 2 * verandah_width) - (L * width) = verandah_area

theorem find_length_of_room : length_of_room 15 12 2 124 :=
by
  -- We state the proof here, which is not requested in this exercise
  sorry

end find_length_of_room_l162_162885


namespace Jim_catches_Bob_in_20_minutes_l162_162690

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end Jim_catches_Bob_in_20_minutes_l162_162690


namespace money_conditions_l162_162825

theorem money_conditions (c d : ℝ) (h1 : 7 * c - d > 80) (h2 : 4 * c + d = 44) (h3 : d < 2 * c) :
  c > 124 / 11 ∧ d < 2 * c ∧ d = 12 :=
by
  sorry

end money_conditions_l162_162825


namespace one_kid_six_whiteboards_l162_162699

theorem one_kid_six_whiteboards (k: ℝ) (b1 b2: ℝ) (t1 t2: ℝ) 
  (hk: k = 1) (hb1: b1 = 3) (hb2: b2 = 6) 
  (ht1: t1 = 20) 
  (H: 4 * t1 / b1 = t2 / b2) : 
  t2 = 160 := 
by
  -- provide the proof here
  sorry

end one_kid_six_whiteboards_l162_162699


namespace range_of_a_l162_162773

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1/x) + a

theorem range_of_a (a : ℝ) (h : f a 0 = a^2) : (f a 0 = f a 0 -> 0 ≤ a ∧ a ≤ 2) := by
  sorry

end range_of_a_l162_162773


namespace sale_in_third_month_l162_162238

theorem sale_in_third_month
  (sale1 sale2 sale4 sale5 sale6 avg : ℝ)
  (n : ℕ)
  (h_sale1 : sale1 = 6235)
  (h_sale2 : sale2 = 6927)
  (h_sale4 : sale4 = 7230)
  (h_sale5 : sale5 = 6562)
  (h_sale6 : sale6 = 5191)
  (h_avg : avg = 6500)
  (h_n : n = 6) :
  ∃ sale3 : ℝ, sale3 = 6855 := by
  sorry

end sale_in_third_month_l162_162238


namespace sin_thirty_degrees_l162_162459

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l162_162459


namespace andrew_grapes_purchase_l162_162185

theorem andrew_grapes_purchase (G : ℕ) (rate_grape rate_mango total_paid total_mango_cost : ℕ)
  (h1 : rate_grape = 54)
  (h2 : rate_mango = 62)
  (h3 : total_paid = 1376)
  (h4 : total_mango_cost = 10 * rate_mango)
  (h5 : total_paid = rate_grape * G + total_mango_cost) : G = 14 := by
  sorry

end andrew_grapes_purchase_l162_162185


namespace arithmetic_sequence_length_correct_l162_162035

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end arithmetic_sequence_length_correct_l162_162035


namespace part1_part2_l162_162140

variables {A B C : ℝ} {a b c : ℝ} -- Angles and sides of the triangle
variable (h1 : (a - b + c) * (a - b - c) + a * b = 0)
variable (h2 : b * c * Real.sin C = 3 * c * Real.cos A + 3 * a * Real.cos C)

theorem part1 : c = 2 * Real.sqrt 3 :=
by
  sorry

theorem part2 : 6 < a + b ∧ a + b <= 4 * Real.sqrt 3 :=
by
  sorry

end part1_part2_l162_162140


namespace jed_speeding_l162_162922

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end jed_speeding_l162_162922


namespace amanda_weekly_earnings_l162_162571

def amanda_rate_per_hour : ℝ := 20.00
def monday_appointments : ℕ := 5
def monday_hours_per_appointment : ℝ := 1.5
def tuesday_appointment_hours : ℝ := 3
def thursday_appointments : ℕ := 2
def thursday_hours_per_appointment : ℝ := 2
def saturday_appointment_hours : ℝ := 6

def total_hours_worked : ℝ :=
  monday_appointments * monday_hours_per_appointment +
  tuesday_appointment_hours +
  thursday_appointments * thursday_hours_per_appointment +
  saturday_appointment_hours

def total_earnings : ℝ := total_hours_worked * amanda_rate_per_hour

theorem amanda_weekly_earnings : total_earnings = 410.00 :=
  by
    unfold total_earnings total_hours_worked monday_appointments monday_hours_per_appointment tuesday_appointment_hours thursday_appointments thursday_hours_per_appointment saturday_appointment_hours amanda_rate_per_hour 
    -- The proof will involve basic arithmetic simplification, which is skipped here.
    -- Therefore, we simply state sorry.
    sorry

end amanda_weekly_earnings_l162_162571


namespace cos_third_quadrant_l162_162092

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l162_162092


namespace valid_license_plates_count_l162_162930

-- Define the number of choices for letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates
def num_valid_license_plates : ℕ := num_letters^3 * num_digits^3

-- Theorem stating that the number of valid license plates is 17,576,000
theorem valid_license_plates_count :
  num_valid_license_plates = 17576000 :=
by
  sorry

end valid_license_plates_count_l162_162930


namespace broken_line_coverable_l162_162597

noncomputable def cover_broken_line (length_of_line : ℝ) (radius_of_circle : ℝ) : Prop :=
  length_of_line = 5 ∧ radius_of_circle > 1.25

theorem broken_line_coverable :
  ∃ radius_of_circle, cover_broken_line 5 radius_of_circle :=
by sorry

end broken_line_coverable_l162_162597


namespace final_state_probability_l162_162317

-- Define the initial state and conditions of the problem
structure GameState where
  raashan : ℕ
  sylvia : ℕ
  ted : ℕ
  uma : ℕ

-- Conditions: each player starts with $2, and the game evolves over 500 rounds
def initial_state : GameState :=
  { raashan := 2, sylvia := 2, ted := 2, uma := 2 }

def valid_statements (state : GameState) : Prop :=
  state.raashan = 2 ∧ state.sylvia = 2 ∧ state.ted = 2 ∧ state.uma = 2

-- Final theorem statement
theorem final_state_probability :
  let states := 500 -- representing the number of rounds
  -- proof outline implies that after the games have properly transitioned and bank interactions, the probability is calculated
  -- state after the transitions
  ∃ (prob : ℚ), prob = 1/4 ∧ valid_statements initial_state :=
  sorry

end final_state_probability_l162_162317


namespace gabrielle_saw_more_birds_l162_162481

def birds_seen (robins cardinals blue_jays : Nat) : Nat :=
  robins + cardinals + blue_jays

def percentage_difference (g c : Nat) : Nat :=
  ((g - c) * 100) / c

theorem gabrielle_saw_more_birds :
  let gabrielle := birds_seen 5 4 3
  let chase := birds_seen 2 5 3
  percentage_difference gabrielle chase = 20 := 
by
  sorry

end gabrielle_saw_more_birds_l162_162481


namespace necessary_but_not_sufficient_condition_l162_162722

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∃ x, x > 2 ∧ ¬ (x > 3)) ∧ 
  (∀ x, x > 3 → x > 2) := by sorry

end necessary_but_not_sufficient_condition_l162_162722


namespace striped_shirts_more_than_shorts_l162_162840

theorem striped_shirts_more_than_shorts :
  ∀ (total_students striped_students checkered_students short_students : ℕ),
    total_students = 81 →
    striped_students = total_students * 2 / 3 →
    checkered_students = total_students - striped_students →
    short_students = checkered_students + 19 →
    striped_students - short_students = 8 :=
by
  intros total_students striped_students checkered_students short_students
  sorry

end striped_shirts_more_than_shorts_l162_162840


namespace trader_sold_pens_l162_162007

theorem trader_sold_pens (C : ℝ) (N : ℕ) (hC : C > 0) (h_gain : N * (2 / 5) = 40) : N = 100 :=
by
  sorry

end trader_sold_pens_l162_162007


namespace problem_statement_l162_162959

noncomputable def a : ℝ := -0.5
noncomputable def b : ℝ := (1 + Real.sqrt 3) / 2

theorem problem_statement
  (h1 : a^2 = 9 / 36)
  (h2 : b^2 = (1 + Real.sqrt 3)^2 / 8)
  (h3 : a < 0)
  (h4 : b > 0) :
  ∃ (x y z : ℤ), (a - b)^2 = x * Real.sqrt y / z ∧ (x + y + z = 6) :=
sorry

end problem_statement_l162_162959


namespace refill_cost_calculation_l162_162078

variables (total_spent : ℕ) (refills : ℕ)

def one_refill_cost (total_spent refills : ℕ) : ℕ := total_spent / refills

theorem refill_cost_calculation (h1 : total_spent = 40) (h2 : refills = 4) :
  one_refill_cost total_spent refills = 10 :=
by
  sorry

end refill_cost_calculation_l162_162078


namespace sheets_of_paper_in_each_box_l162_162241

theorem sheets_of_paper_in_each_box (E S : ℕ) (h1 : 2 * E + 40 = S) (h2 : 4 * (E - 40) = S) : S = 240 :=
by
  sorry

end sheets_of_paper_in_each_box_l162_162241


namespace running_time_constant_pace_l162_162807

/-!
# Running Time Problem

We are given that the running pace is constant, it takes 30 minutes to run 5 miles,
and we need to find out how long it will take to run 2.5 miles.
-/

theorem running_time_constant_pace :
  ∀ (distance_to_store distance_to_cousin distance_run time_run : ℝ)
  (constant_pace : Prop),
  distance_to_store = 5 → time_run = 30 → distance_to_cousin = 2.5 →
  constant_pace → 
  time_run / distance_to_store * distance_to_cousin = 15 :=
by 
  intros distance_to_store distance_to_cousin distance_run time_run constant_pace 
         hds htr hdc hcp
  rw [hds, htr, hdc]
  exact sorry

end running_time_constant_pace_l162_162807


namespace divides_mn_minus_one_l162_162258

theorem divides_mn_minus_one (m n p : ℕ) (hp : p.Prime) (h1 : m < n) (h2 : n < p) 
    (hm2 : p ∣ m^2 + 1) (hn2 : p ∣ n^2 + 1) : p ∣ m * n - 1 :=
by
  sorry

end divides_mn_minus_one_l162_162258


namespace functional_equation_solution_l162_162442

theorem functional_equation_solution (f : ℝ → ℝ) (t : ℝ) (h : t ≠ -1) :
  (∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)) →
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2)) :=
by
  sorry

end functional_equation_solution_l162_162442


namespace intersection_M_N_l162_162443

open Set

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by 
sorry

end intersection_M_N_l162_162443


namespace license_plate_count_l162_162227

theorem license_plate_count : (26^3 * 5 * 5 * 4) = 1757600 := 
by 
  sorry

end license_plate_count_l162_162227


namespace parallelogram_slope_l162_162190

theorem parallelogram_slope (a b c d : ℚ) :
    a = 35 + c ∧ b = 125 - c ∧ 875 - 25 * c = 280 + 8 * c ∧ (a, 8) = (b, 25)
    → ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ (∃ h : 8 * 33 * a + 595 = 2350, (m, n) = (25, 4)) :=
by
  sorry

end parallelogram_slope_l162_162190


namespace remainder_17_pow_77_mod_7_l162_162490

theorem remainder_17_pow_77_mod_7 : (17^77) % 7 = 5 := 
by sorry

end remainder_17_pow_77_mod_7_l162_162490


namespace sum_lent_is_10000_l162_162817

theorem sum_lent_is_10000
  (P : ℝ)
  (r : ℝ := 0.075)
  (t : ℝ := 7)
  (I : ℝ := P - 4750) 
  (H1 : I = P * r * t) :
  P = 10000 :=
sorry

end sum_lent_is_10000_l162_162817


namespace tom_remaining_balloons_l162_162056

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end tom_remaining_balloons_l162_162056


namespace pyramid_volume_is_one_sixth_l162_162494

noncomputable def volume_of_pyramid_in_cube : ℝ :=
  let edge_length := 1
  let base_area := (1 / 2) * edge_length * edge_length
  let height := edge_length
  (1 / 3) * base_area * height

theorem pyramid_volume_is_one_sixth : volume_of_pyramid_in_cube = 1 / 6 :=
by
  -- Let edge_length = 1, base_area = 1 / 2 * edge_length * edge_length = 1 / 2, 
  -- height = edge_length = 1. Then volume = 1 / 3 * base_area * height = 1 / 6.
  sorry

end pyramid_volume_is_one_sixth_l162_162494


namespace interval_of_monotonic_increase_l162_162216

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def y' (x : ℝ) : ℝ := 2 * x * Real.exp x + x^2 * Real.exp x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, (y' x ≥ 0 ↔ (x ∈ Set.Ici 0 ∨ x ∈ Set.Iic (-2))) :=
by
  sorry

end interval_of_monotonic_increase_l162_162216


namespace rectangles_fit_l162_162393

theorem rectangles_fit :
  let width := 50
  let height := 90
  let r_width := 1
  let r_height := (10 * Real.sqrt 2)
  ∃ n : ℕ, 
  n = 315 ∧
  (∃ w_cuts h_cuts : ℕ, 
    w_cuts = Int.floor (width / r_height) ∧
    h_cuts = Int.floor (height / r_height) ∧
    n = ((Int.floor (width / r_height) * Int.floor (height / r_height)) + 
         (Int.floor (height / r_width) * Int.floor (width / r_height)))) := 
sorry

end rectangles_fit_l162_162393


namespace opposite_of_neg5_is_pos5_l162_162263

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l162_162263


namespace problem_statement_l162_162280

noncomputable def general_term (a : ℕ → ℕ) (n : ℕ) : Prop :=
a n = n

noncomputable def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, S n = (n * (n + 1)) / 2

noncomputable def b_def (S : ℕ → ℕ) (b : ℕ → ℚ) : Prop :=
∀ n, b n = (2 : ℚ) / (S n)

noncomputable def sum_b_first_n_terms (b : ℕ → ℚ) (T : ℕ → ℚ) : Prop :=
∀ n, T n = (4 * n) / (n + 1)

theorem problem_statement (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, a n = 1 + (n - 1) * 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) - a n ≠ 0) →
  a 3 ^ 2 = a 1 * a 9 →
  general_term a 1 →
  sum_first_n_terms a S →
  b_def S b →
  sum_b_first_n_terms b T :=
by
  intro arithmetic_seq
  intro a_1_eq_1
  intro non_zero_diff
  intro geometric_seq
  intro gen_term_cond
  intro sum_terms_cond
  intro b_def_cond
  intro sum_b_terms_cond
  -- The proof goes here.
  sorry

end problem_statement_l162_162280


namespace boa_constrictors_in_park_l162_162932

theorem boa_constrictors_in_park :
  ∃ (B : ℕ), (∃ (p : ℕ), p = 3 * B) ∧ (B + 3 * B + 40 = 200) ∧ B = 40 :=
by
  sorry

end boa_constrictors_in_park_l162_162932


namespace minimal_benches_l162_162505

theorem minimal_benches (x : ℕ) 
  (standard_adults : ℕ := x * 8) (standard_children : ℕ := x * 12)
  (extended_adults : ℕ := x * 8) (extended_children : ℕ := x * 16) 
  (hx : standard_adults + extended_adults = standard_children + extended_children) :
  x = 1 :=
by
  sorry

end minimal_benches_l162_162505


namespace fifth_number_in_ninth_row_l162_162358

theorem fifth_number_in_ninth_row :
  ∃ (n : ℕ), n = 61 ∧ ∀ (i : ℕ), i = 9 → (7 * i - 2 = n) :=
by
  sorry

end fifth_number_in_ninth_row_l162_162358


namespace olivia_not_sold_bars_l162_162895

theorem olivia_not_sold_bars (cost_per_bar : ℕ) (total_bars : ℕ) (total_money_made : ℕ) :
  cost_per_bar = 3 →
  total_bars = 7 →
  total_money_made = 9 →
  total_bars - (total_money_made / cost_per_bar) = 4 :=
by
  intros h1 h2 h3
  sorry

end olivia_not_sold_bars_l162_162895


namespace tom_spend_l162_162050

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end tom_spend_l162_162050


namespace binary_10101_to_decimal_l162_162683

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ digit idx => digit * 2^idx) (List.range b.length) |>.sum

theorem binary_10101_to_decimal : binary_to_decimal [1, 0, 1, 0, 1] = 21 := by
  sorry

end binary_10101_to_decimal_l162_162683


namespace certain_event_among_options_l162_162476

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end certain_event_among_options_l162_162476


namespace largest_four_digit_negative_integer_congruent_to_2_mod_17_l162_162053

theorem largest_four_digit_negative_integer_congruent_to_2_mod_17 :
  ∃ (n : ℤ), (n % 17 = 2 ∧ n > -10000 ∧ n < -999) ∧ ∀ m : ℤ, (m % 17 = 2 ∧ m > -10000 ∧ m < -999) → m ≤ n :=
sorry

end largest_four_digit_negative_integer_congruent_to_2_mod_17_l162_162053


namespace two_times_difference_eq_20_l162_162843

theorem two_times_difference_eq_20 (x y : ℕ) (hx : x = 30) (hy : y = 20) (hsum : x + y = 50) : 2 * (x - y) = 20 := by
  sorry

end two_times_difference_eq_20_l162_162843


namespace antonio_age_in_months_l162_162447

-- Definitions based on the conditions
def is_twice_as_old (isabella_age antonio_age : ℕ) : Prop :=
  isabella_age = 2 * antonio_age

def future_age (current_age months_future : ℕ) : ℕ :=
  current_age + months_future

-- Given the conditions
variables (isabella_age antonio_age : ℕ)
variables (future_age_18months target_age : ℕ)

-- Conditions
axiom condition1 : is_twice_as_old isabella_age antonio_age
axiom condition2 : future_age_18months = 18
axiom condition3 : target_age = 10 * 12

-- Assertion that we need to prove
theorem antonio_age_in_months :
  ∃ (antonio_age : ℕ), future_age isabella_age future_age_18months = target_age → antonio_age = 51 :=
by
  sorry

end antonio_age_in_months_l162_162447


namespace minimum_value_of_function_l162_162193

noncomputable def function_y (x : ℝ) : ℝ := 1 / (Real.sqrt (x - x^2))

theorem minimum_value_of_function : (∀ x : ℝ, 0 < x ∧ x < 1 → function_y x ≥ 2) ∧ (∃ x : ℝ, 0 < x ∧ x < 1 ∧ function_y x = 2) :=
by
  sorry

end minimum_value_of_function_l162_162193


namespace compute_value_l162_162633

theorem compute_value : ((-120) - (-60)) / (-30) = 2 := 
by 
  sorry

end compute_value_l162_162633


namespace total_wheels_l162_162969

-- Definitions of given conditions
def bicycles : ℕ := 50
def tricycles : ℕ := 20
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Theorem stating the total number of wheels for bicycles and tricycles combined
theorem total_wheels : bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160 :=
by
  sorry

end total_wheels_l162_162969


namespace no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l162_162541

-- Proof Problem 1:
theorem no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n :
  ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + 2) = n * (n + 1) :=
by sorry

-- Proof Problem 2:
theorem k_ge_3_positive_ints_m_n_exists (k : ℕ) (hk : k ≥ 3) :
  (k = 3 → ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) ∧
  (k ≥ 4 → ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) :=
by sorry

end no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l162_162541


namespace total_square_footage_after_expansion_l162_162066

-- Definitions from the conditions
def size_smaller_house_initial : ℕ := 5200
def size_larger_house : ℕ := 7300
def expansion_smaller_house : ℕ := 3500

-- The new size of the smaller house after expansion
def size_smaller_house_after_expansion : ℕ :=
  size_smaller_house_initial + expansion_smaller_house

-- The new total square footage
def new_total_square_footage : ℕ :=
  size_smaller_house_after_expansion + size_larger_house

-- Goal statement: Prove the total new square footage is 16000 sq. ft.
theorem total_square_footage_after_expansion : new_total_square_footage = 16000 := by
  sorry

end total_square_footage_after_expansion_l162_162066


namespace negative_x_y_l162_162893

theorem negative_x_y (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 :=
by
  sorry

end negative_x_y_l162_162893


namespace fewest_candies_l162_162501

-- Defining the conditions
def condition1 (x : ℕ) := x % 21 = 5
def condition2 (x : ℕ) := x % 22 = 3
def condition3 (x : ℕ) := x > 500

-- Stating the main theorem
theorem fewest_candies : ∃ x : ℕ, condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 509 :=
  sorry

end fewest_candies_l162_162501


namespace repeating_decimal_fraction_l162_162775

noncomputable def repeating_decimal := 7 + ((789 : ℚ) / (10^4 - 1))

theorem repeating_decimal_fraction :
  repeating_decimal = (365 : ℚ) / 85 :=
by
  sorry

end repeating_decimal_fraction_l162_162775


namespace linear_regression_increase_l162_162915

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ :=
  1.6 * x + 2

-- Prove that y increases by 1.6 when x increases by 1
theorem linear_regression_increase (x : ℝ) :
  linear_regression (x + 1) - linear_regression x = 1.6 :=
by sorry

end linear_regression_increase_l162_162915


namespace eval_expression_l162_162618

theorem eval_expression : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end eval_expression_l162_162618


namespace find_m_and_other_root_l162_162202

theorem find_m_and_other_root (m : ℝ) (r : ℝ) :
    (∃ x : ℝ, x^2 + m*x - 2 = 0) ∧ (x = -1) → (m = -1 ∧ r = 2) :=
by
  sorry

end find_m_and_other_root_l162_162202


namespace total_lunch_cost_l162_162093

/-- Janet, a third grade teacher, is picking up the sack lunch order from a local deli for 
the field trip she is taking her class on. There are 35 children in her class, 5 volunteer 
chaperones, and herself. She also ordered three additional sack lunches, just in case 
there was a problem. Each sack lunch costs $7. --/
theorem total_lunch_cost :
  let children := 35
  let chaperones := 5
  let janet := 1
  let additional_lunches := 3
  let price_per_lunch := 7
  let total_lunches := children + chaperones + janet + additional_lunches
  total_lunches * price_per_lunch = 308 :=
by
  sorry

end total_lunch_cost_l162_162093


namespace point_in_fourth_quadrant_l162_162882

-- Define complex number and evaluate it
noncomputable def z : ℂ := (2 - (1 : ℂ) * Complex.I) / (1 + (1 : ℂ) * Complex.I)

-- Prove that the complex number z lies in the fourth quadrant
theorem point_in_fourth_quadrant (hz: z = (1/2 : ℂ) - (3/2 : ℂ) * Complex.I) : z.im < 0 ∧ z.re > 0 :=
by
  -- Skipping the proof here
  sorry

end point_in_fourth_quadrant_l162_162882


namespace tripod_max_height_l162_162191

noncomputable def tripod_new_height (original_height : ℝ) (original_leg_length : ℝ) (broken_leg_length : ℝ) : ℝ :=
  (broken_leg_length / original_leg_length) * original_height

theorem tripod_max_height :
  let original_height := 5
  let original_leg_length := 6
  let broken_leg_length := 4
  let h := tripod_new_height original_height original_leg_length broken_leg_length
  h = (10 / 3) :=
by
  sorry

end tripod_max_height_l162_162191


namespace vector_addition_correct_dot_product_correct_l162_162600

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem vector_addition_correct :
  let a := (1, 2)
  let b := (3, 1)
  vector_add a b = (4, 3) := by
  sorry

theorem dot_product_correct :
  let a := (1, 2)
  let b := (3, 1)
  dot_product a b = 5 := by
  sorry

end vector_addition_correct_dot_product_correct_l162_162600


namespace coordinates_F_l162_162124

-- Definition of point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Reflection over the y-axis
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Reflection over the x-axis
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Original point F
def F : Point := { x := 3, y := 3 }

-- First reflection over the y-axis
def F' := reflect_y F

-- Second reflection over the x-axis
def F'' := reflect_x F'

-- Goal: Coordinates of F'' after both reflections
theorem coordinates_F'' : F'' = { x := -3, y := -3 } :=
by
  -- Proof would go here
  sorry

end coordinates_F_l162_162124


namespace linear_relation_is_correct_maximum_profit_l162_162725

-- Define the given data points
structure DataPoints where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the given conditions
def conditions : DataPoints := ⟨50, 100, 60, 90⟩

-- Define the cost and sell price range conditions
def cost_per_kg : ℝ := 20
def max_selling_price : ℝ := 90

-- Define the linear relationship function
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_per_kg) * (linear_relationship (-1) 150 x)

-- Statements to Prove
theorem linear_relation_is_correct (k b : ℝ) :
  linear_relationship k b 50 = 100 ∧
  linear_relationship k b 60 = 90 →
  (b = 150 ∧ k = -1) := by
  intros h
  sorry

theorem maximum_profit :
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ max_selling_price ∧ profit_function x = 4225 := by
  use 85
  sorry

end linear_relation_is_correct_maximum_profit_l162_162725


namespace min_value_arith_seq_l162_162072

theorem min_value_arith_seq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + c = 2 * b) :
  (a + c) / b + b / (a + c) ≥ 5 / 2 := 
sorry

end min_value_arith_seq_l162_162072


namespace allowance_calculation_l162_162796

theorem allowance_calculation (A : ℝ)
  (h1 : (3 / 5) * A + (1 / 3) * (2 / 5) * A + 0.40 = A)
  : A = 1.50 :=
sorry

end allowance_calculation_l162_162796


namespace positive_y_percentage_l162_162326

theorem positive_y_percentage (y : ℝ) (hy_pos : 0 < y) (h : 0.01 * y * y = 9) : y = 30 := by
  sorry

end positive_y_percentage_l162_162326


namespace surface_area_of_sphere_l162_162400

-- Define the conditions from the problem.

variables (r R : ℝ) -- r is the radius of the cross-section, R is the radius of the sphere.
variables (π : ℝ := Real.pi) -- Define π using the real pi constant.
variables (h_dist : 1 = 1) -- Distance from the plane to the center is 1 unit.
variables (h_area_cross_section : π = π * r^2) -- Area of the cross-section is π.

-- State to prove the surface area of the sphere is 8π.
theorem surface_area_of_sphere :
    ∃ (R : ℝ), (R^2 = 2) → (4 * π * R^ 2 = 8 * π) := sorry

end surface_area_of_sphere_l162_162400


namespace specialSignLanguage_l162_162436

theorem specialSignLanguage (S : ℕ) 
  (h1 : (S + 2) * (S + 2) = S * S + 1288) : S = 321 := 
by
  sorry

end specialSignLanguage_l162_162436


namespace envelope_of_family_of_lines_l162_162225

theorem envelope_of_family_of_lines (a α : ℝ) (hα : α > 0) :
    ∀ (x y : ℝ), (∃ α > 0,
    (x = a * α / 2 ∧ y = a / (2 * α))) ↔ (x * y = a^2 / 4) := by
  sorry

end envelope_of_family_of_lines_l162_162225


namespace claire_balloons_l162_162356

def initial_balloons : ℕ := 50
def balloons_lost : ℕ := 12
def balloons_given_away : ℕ := 9
def balloons_received : ℕ := 11

theorem claire_balloons : initial_balloons - balloons_lost - balloons_given_away + balloons_received = 40 :=
by
  sorry

end claire_balloons_l162_162356


namespace count_paths_to_form_2005_l162_162487

/-- Define the structure of a circle label. -/
inductive CircleLabel
| two
| zero
| five

open CircleLabel

/-- Define the number of possible moves from each circle. -/
def moves_from_two : Nat := 6
def moves_from_zero_to_zero : Nat := 2
def moves_from_zero_to_five : Nat := 3

/-- Define the total number of paths to form 2005. -/
def total_paths : Nat := moves_from_two * moves_from_zero_to_zero * moves_from_zero_to_five

/-- The proof statement: The total number of different paths to form the number 2005 is 36. -/
theorem count_paths_to_form_2005 : total_paths = 36 :=
by
  sorry

end count_paths_to_form_2005_l162_162487


namespace largest_angle_of_consecutive_odd_int_angles_is_125_l162_162477

-- Definitions for a convex hexagon with six consecutive odd integer interior angles
def is_consecutive_odd_integers (xs : List ℕ) : Prop :=
  ∀ n, 0 ≤ n ∧ n < 5 → xs.get! n + 2 = xs.get! (n + 1)

def hexagon_angles_sum_720 (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.sum = 720

-- Main theorem statement
theorem largest_angle_of_consecutive_odd_int_angles_is_125 (xs : List ℕ) 
(h1 : is_consecutive_odd_integers xs) 
(h2 : hexagon_angles_sum_720 xs) : 
  xs.maximum = 125 := 
sorry

end largest_angle_of_consecutive_odd_int_angles_is_125_l162_162477


namespace smaller_factor_of_4851_l162_162226

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end smaller_factor_of_4851_l162_162226


namespace find_value_of_b_l162_162220

theorem find_value_of_b (a b : ℕ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
sorry

end find_value_of_b_l162_162220


namespace find_xy_l162_162148

theorem find_xy (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_xy_l162_162148


namespace rectangular_plot_breadth_l162_162838

theorem rectangular_plot_breadth :
  ∀ (l b : ℝ), (l = 3 * b) → (l * b = 588) → (b = 14) :=
by
  intros l b h1 h2
  sorry

end rectangular_plot_breadth_l162_162838


namespace line_segment_parametric_curve_l162_162628

noncomputable def parametric_curve (θ : ℝ) := 
  (2 + Real.cos θ ^ 2, 1 - Real.sin θ ^ 2)

theorem line_segment_parametric_curve : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → 
    ∃ x y : ℝ, (x, y) = parametric_curve θ ∧ 2 ≤ x ∧ x ≤ 3 ∧ x - y = 2) := 
sorry

end line_segment_parametric_curve_l162_162628


namespace compute_f_f_f_19_l162_162429

def f (x : Int) : Int :=
  if x < 10 then x^2 - 9 else x - 15

theorem compute_f_f_f_19 : f (f (f 19)) = 40 := by
  sorry

end compute_f_f_f_19_l162_162429


namespace percentage_of_rotten_bananas_l162_162674

theorem percentage_of_rotten_bananas :
  ∀ (total_oranges total_bananas : ℕ) 
    (percent_rotten_oranges : ℝ) 
    (percent_good_fruits : ℝ), 
  total_oranges = 600 → total_bananas = 400 → 
  percent_rotten_oranges = 0.15 → percent_good_fruits = 0.89 → 
  (100 - (((percent_good_fruits * (total_oranges + total_bananas)) - 
  ((1 - percent_rotten_oranges) * total_oranges)) / total_bananas) * 100) = 5 := 
by
  intros total_oranges total_bananas percent_rotten_oranges percent_good_fruits 
  intro ho hb hro hpf 
  sorry

end percentage_of_rotten_bananas_l162_162674


namespace candy_bar_cost_l162_162087

/-- Problem statement:
Todd had 85 cents and spent 53 cents in total on a candy bar and a box of cookies.
The box of cookies cost 39 cents. How much did the candy bar cost? --/
theorem candy_bar_cost (t c s b : ℕ) (ht : t = 85) (hc : c = 39) (hs : s = 53) (h_total : s = b + c) : b = 14 :=
by
  sorry

end candy_bar_cost_l162_162087


namespace ball_travel_distance_five_hits_l162_162303

def total_distance_traveled (h₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  let descents := List.range (n + 1) |>.map (λ i => h₀ * r ^ i)
  let ascents := List.range n |>.map (λ i => h₀ * r ^ (i + 1))
  (descents.sum + ascents.sum)

theorem ball_travel_distance_five_hits :
  total_distance_traveled 120 (3 / 4) 5 = 612.1875 :=
by
  sorry

end ball_travel_distance_five_hits_l162_162303


namespace constant_term_expansion_l162_162530

noncomputable def sum_of_coefficients (a : ℕ) : ℕ := sorry

noncomputable def constant_term (a : ℕ) : ℕ := sorry

theorem constant_term_expansion (a : ℕ) (h : sum_of_coefficients a = 2) : constant_term 2 = 10 :=
sorry

end constant_term_expansion_l162_162530


namespace sum_between_100_and_500_ending_in_3_l162_162125

-- Definition for the sum of all integers between 100 and 500 that end in 3
def sumOfIntegersBetween100And500EndingIn3 : ℕ :=
  let a := 103
  let d := 10
  let n := (493 - a) / d + 1
  (n * (a + 493)) / 2

-- Statement to prove that the sum is 11920
theorem sum_between_100_and_500_ending_in_3 : sumOfIntegersBetween100And500EndingIn3 = 11920 := by
  sorry

end sum_between_100_and_500_ending_in_3_l162_162125


namespace general_formula_a_n_sum_first_n_terms_T_n_l162_162120

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Condition: S_n = 2a_n - 3
axiom condition_S (n : ℕ) : S_n n = 2 * (a_n n) - 3

-- (I) General formula for a_n
theorem general_formula_a_n (n : ℕ) : a_n n = 3 * 2^(n - 1) := 
sorry

-- (II) General formula for T_n
theorem sum_first_n_terms_T_n (n : ℕ) : T_n n = 3 * (n - 1) * 2^n + 3 := 
sorry

end general_formula_a_n_sum_first_n_terms_T_n_l162_162120


namespace min_value_expression_l162_162437

theorem min_value_expression (x : ℚ) : ∃ x : ℚ, (2 * x - 5)^2 + 18 = 18 :=
by {
  use 2.5,
  sorry
}

end min_value_expression_l162_162437


namespace A_takes_200_seconds_l162_162957

/-- 
  A can give B a start of 50 meters or 10 seconds in a kilometer race.
  How long does A take to complete the race?
-/
theorem A_takes_200_seconds (v_A : ℝ) (distance : ℝ) (start_meters : ℝ) (start_seconds : ℝ) :
  (start_meters = 50) ∧ (start_seconds = 10) ∧ (distance = 1000) ∧ 
  (v_A = start_meters / start_seconds) → distance / v_A = 200 :=
by
  sorry

end A_takes_200_seconds_l162_162957


namespace num_ordered_pairs_no_real_solution_l162_162156

theorem num_ordered_pairs_no_real_solution : 
  {n : ℕ // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4*c < 0 ∨ c^2 - 4*b < 0) ∧ n = 6 } := by
sorry

end num_ordered_pairs_no_real_solution_l162_162156


namespace maximum_value_of_f_l162_162521

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

theorem maximum_value_of_f :
  ∃ x_max : ℝ, x_max > 0 ∧ (∀ x : ℝ, x > 0 → f x ≤ f x_max) ∧ f x_max = -2 :=
by
  sorry

end maximum_value_of_f_l162_162521


namespace cartesian_to_polar_circle_l162_162441

open Real

theorem cartesian_to_polar_circle (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = ρ * cos θ) 
  (h2 : y = ρ * sin θ) 
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * cos θ :=
sorry

end cartesian_to_polar_circle_l162_162441


namespace steel_strength_value_l162_162343

theorem steel_strength_value 
  (s : ℝ) 
  (condition: s = 4.6 * 10^8) : 
  s = 460000000 := 
by sorry

end steel_strength_value_l162_162343


namespace problem_solution_l162_162837
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.foldl (· + ·) 0

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_seq : ℕ → ℕ → ℕ
| 0, n => f n
| (k+1), n => f (f_seq k n)

theorem problem_solution :
  f_seq 2016 9 = 8 :=
sorry

end problem_solution_l162_162837


namespace students_behind_yoongi_l162_162747

theorem students_behind_yoongi (n k : ℕ) (hn : n = 30) (hk : k = 20) : n - (k + 1) = 9 := by
  sorry

end students_behind_yoongi_l162_162747


namespace max_squared_sum_of_sides_l162_162359

variable {R : ℝ}
variable {O A B C : EucSpace} -- O is the center, A, B, and C are vertices
variable (a b c : ℝ)  -- Position vectors corresponding to vertices A, B, C

-- Hypotheses based on the problem conditions:
variable (h1 : ‖a‖ = R)
variable (h2 : ‖b‖ = R)
variable (h3 : ‖c‖ = R)
variable (hSumZero : a + b + c = 0)

theorem max_squared_sum_of_sides 
  {AB BC CA : ℝ} -- Side lengths
  (hAB : AB = ‖a - b‖)
  (hBC : BC = ‖b - c‖)
  (hCA : CA = ‖c - a‖) :
  AB^2 + BC^2 + CA^2 = 9 * R^2 :=
sorry

end max_squared_sum_of_sides_l162_162359


namespace length_of_platform_l162_162486

-- Definitions based on the problem conditions
def train_length : ℝ := 300
def platform_crossing_time : ℝ := 39
def signal_pole_crossing_time : ℝ := 18

-- The main theorem statement
theorem length_of_platform : ∀ (L : ℝ), train_length + L = (train_length / signal_pole_crossing_time) * platform_crossing_time → L = 350.13 :=
by
  intro L h
  sorry

end length_of_platform_l162_162486


namespace small_cubes_for_larger_cube_l162_162062

theorem small_cubes_for_larger_cube (VL VS : ℕ) (h : VL = 125 * VS) : (VL / VS = 125) :=
by {
    sorry
}

end small_cubes_for_larger_cube_l162_162062


namespace find_some_value_l162_162354

theorem find_some_value (m n : ℝ) (some_value : ℝ) (p : ℝ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + some_value) / 6 - 2 / 5)
  (h3 : p = 3)
  : some_value = -12 / 5 :=
by
  sorry

end find_some_value_l162_162354


namespace jackson_total_calories_l162_162596

def lettuce_calories : ℕ := 50
def carrots_calories : ℕ := 2 * lettuce_calories
def dressing_calories : ℕ := 210
def salad_calories : ℕ := lettuce_calories + carrots_calories + dressing_calories

def crust_calories : ℕ := 600
def pepperoni_calories : ℕ := crust_calories / 3
def cheese_calories : ℕ := 400
def pizza_calories : ℕ := crust_calories + pepperoni_calories + cheese_calories

def jackson_salad_fraction : ℚ := 1 / 4
def jackson_pizza_fraction : ℚ := 1 / 5

noncomputable def total_calories : ℚ := 
  jackson_salad_fraction * salad_calories + jackson_pizza_fraction * pizza_calories

theorem jackson_total_calories : total_calories = 330 := by
  sorry

end jackson_total_calories_l162_162596


namespace intersection_A_B_l162_162756

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3) }
def B : Set ℝ := { x | ∃ y, y = Real.log x }

theorem intersection_A_B : A ∩ B = {x | x ∈ Set.Ici 3} :=
by
  sorry

end intersection_A_B_l162_162756


namespace shaded_region_is_hyperbolas_l162_162800

theorem shaded_region_is_hyperbolas (T : ℝ) (hT : T > 0) :
  (∃ (x y : ℝ), x * y = T / 4) ∧ (∃ (x y : ℝ), x * y = - (T / 4)) :=
by
  sorry

end shaded_region_is_hyperbolas_l162_162800


namespace intersection_l162_162644

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / (2 * x - 6)

noncomputable def g (x : ℝ) (a b c d k : ℝ) : ℝ := -2 * x - 4 + k / (x - d)

theorem intersection (a b c k : ℝ) (h_d : d = 3) (h_k : k = 36) : 
  ∃ (x y : ℝ), x ≠ -3 ∧ (f x = g x 0 0 0 d k) ∧ (x, y) = (6.8, -32 / 19) :=
by
  sorry

end intersection_l162_162644


namespace birds_on_fence_l162_162752

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end birds_on_fence_l162_162752


namespace characterize_functions_l162_162279

open Function

noncomputable def f : ℚ → ℚ := sorry
noncomputable def g : ℚ → ℚ := sorry

axiom f_g_condition_1 : ∀ x y : ℚ, f (g (x) - g (y)) = f (g (x)) - y
axiom f_g_condition_2 : ∀ x y : ℚ, g (f (x) - f (y)) = g (f (x)) - y

theorem characterize_functions : 
  (∃ c : ℚ, ∀ x, f x = c * x) ∧ (∃ c : ℚ, ∀ x, g x = x / c) := 
sorry

end characterize_functions_l162_162279


namespace digits_conditions_l162_162275

noncomputable def original_number : ℕ := 253
noncomputable def reversed_number : ℕ := 352

theorem digits_conditions (a b c : ℕ) : 
  a + b + c = 10 → 
  b = a + c → 
  (original_number = a * 100 + b * 10 + c) → 
  (reversed_number = c * 100 + b * 10 + a) → 
  reversed_number - original_number = 99 :=
by
  intros h1 h2 h3 h4
  sorry

end digits_conditions_l162_162275


namespace proposition_4_l162_162977

variables {Line Plane : Type}
variables {a b : Line} {α β : Plane}

-- Definitions of parallel and perpendicular relationships
class Parallel (l : Line) (p : Plane) : Prop
class Perpendicular (l : Line) (p : Plane) : Prop
class Contains (p : Plane) (l : Line) : Prop

theorem proposition_4
  (h1: Perpendicular a β)
  (h2: Parallel a b)
  (h3: Contains α b) : Perpendicular α β :=
sorry

end proposition_4_l162_162977


namespace min_unit_cubes_l162_162374

theorem min_unit_cubes (l w h : ℕ) (S : ℕ) (hS : S = 52) 
  (hSurface : 2 * (l * w + l * h + w * h) = S) : 
  ∃ l w h, l * w * h = 16 :=
by
  -- start the proof here
  sorry

end min_unit_cubes_l162_162374


namespace abs_z1_purely_imaginary_l162_162808

noncomputable def z1 (a : ℝ) : Complex := ⟨a, 2⟩
def z2 : Complex := ⟨2, -1⟩

theorem abs_z1_purely_imaginary (a : ℝ) (ha : 2 * a - 2 = 0) : Complex.abs (z1 a) = Real.sqrt 5 :=
by
  sorry

end abs_z1_purely_imaginary_l162_162808


namespace proportion_equation_correct_l162_162032

theorem proportion_equation_correct (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  x / 3 = y / 2 := 
  sorry

end proportion_equation_correct_l162_162032


namespace six_digit_number_division_l162_162567

theorem six_digit_number_division :
  ∃ a b p : ℕ, 
    (111111 * a = 1111 * b * 233 + p) ∧ 
    (11111 * a = 111 * b * 233 + p - 1000) ∧
    (111111 * 7 = 777777) ∧
    (1111 * 3 = 3333) :=
by
  sorry

end six_digit_number_division_l162_162567


namespace sell_decision_l162_162502

noncomputable def profit_beginning (a : ℝ) : ℝ :=
(a + 100) * 1.024

noncomputable def profit_end (a : ℝ) : ℝ :=
a + 115

theorem sell_decision (a : ℝ) :
  (a > 525 → profit_beginning a > profit_end a) ∧
  (a < 525 → profit_beginning a < profit_end a) ∧
  (a = 525 → profit_beginning a = profit_end a) :=
by
  sorry

end sell_decision_l162_162502


namespace rhombus_area_l162_162137

theorem rhombus_area 
  (a : ℝ) (d1 d2 : ℝ)
  (h_side : a = Real.sqrt 113)
  (h_diagonal_diff : abs (d1 - d2) = 8)
  (h_geq : d1 ≠ d2) : 
  (a^2 * d1 * d2 / 2 = 194) :=
sorry -- Proof to be completed

end rhombus_area_l162_162137


namespace total_seeds_eaten_l162_162508

def first_seeds := 78
def second_seeds := 53
def third_seeds := second_seeds + 30

theorem total_seeds_eaten : first_seeds + second_seeds + third_seeds = 214 := by
  -- Sorry, placeholder for proof
  sorry

end total_seeds_eaten_l162_162508


namespace find_special_four_digit_square_l162_162728

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end find_special_four_digit_square_l162_162728


namespace probability_blue_or_purple_is_correct_l162_162013

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10 + 4

def blue_jelly_beans : ℕ := 10

def purple_jelly_beans : ℕ := 4

def blue_or_purple_jelly_beans : ℕ := blue_jelly_beans + purple_jelly_beans

def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_correct :
  probability_blue_or_purple = 7 / 19 :=
by
  sorry

end probability_blue_or_purple_is_correct_l162_162013


namespace triangle_inequality_third_side_l162_162787

theorem triangle_inequality_third_side (a : ℝ) (h1 : 3 + a > 7) (h2 : 7 + a > 3) (h3 : 3 + 7 > a) : 
  4 < a ∧ a < 10 :=
by sorry

end triangle_inequality_third_side_l162_162787


namespace find_second_dimension_l162_162484

theorem find_second_dimension (x : ℕ) 
    (h1 : 12 * x * 16 / (3 * 7 * 2) = 64) : 
    x = 14 := by
    sorry

end find_second_dimension_l162_162484


namespace range_of_a_minus_b_l162_162973

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) : 
  -3 < a - b ∧ a - b < 6 :=
by
  sorry

end range_of_a_minus_b_l162_162973


namespace ratio_of_pete_to_susan_l162_162568

noncomputable def Pete_backward_speed := 12 -- in miles per hour
noncomputable def Pete_handstand_speed := 2 -- in miles per hour
noncomputable def Tracy_cartwheel_speed := 4 * Pete_handstand_speed -- in miles per hour
noncomputable def Susan_forward_speed := Tracy_cartwheel_speed / 2 -- in miles per hour

theorem ratio_of_pete_to_susan :
  Pete_backward_speed / Susan_forward_speed = 3 := 
sorry

end ratio_of_pete_to_susan_l162_162568


namespace domain_f_monotonicity_f_inequality_solution_l162_162285

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

variable {x : ℝ}

theorem domain_f : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 -> Set.Ioo (-1 : ℝ) 1 := sorry

theorem monotonicity_f : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f y < f x := sorry

theorem inequality_solution :
  {x : ℝ | f (2 * x - 1) < 0} = {x | x > 1 / 2 ∧ x < 1} := sorry

end domain_f_monotonicity_f_inequality_solution_l162_162285


namespace intersecting_chords_theorem_l162_162099

theorem intersecting_chords_theorem
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18)
  (c d k : ℝ) (h3 : c = 3 * k) (h4 : d = 8 * k) :
  (a * b = c * d) → (k = 3) → (c + d = 33) :=
by 
  sorry

end intersecting_chords_theorem_l162_162099


namespace average_marks_correct_l162_162878

-- Define the marks obtained in each subject
def english_marks := 86
def mathematics_marks := 85
def physics_marks := 92
def chemistry_marks := 87
def biology_marks := 95

-- Calculate total marks and average marks
def total_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects := 5
def average_marks := total_marks / num_subjects

-- Prove that Dacid's average marks are 89
theorem average_marks_correct : average_marks = 89 := by
  sorry

end average_marks_correct_l162_162878


namespace trigonometric_identity_l162_162351

noncomputable def cos190 := Real.cos (190 * Real.pi / 180)
noncomputable def sin290 := Real.sin (290 * Real.pi / 180)
noncomputable def cos40 := Real.cos (40 * Real.pi / 180)
noncomputable def tan10 := Real.tan (10 * Real.pi / 180)

theorem trigonometric_identity :
  (cos190 * (1 + Real.sqrt 3 * tan10)) / (sin290 * Real.sqrt (1 - cos40)) = 2 * Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l162_162351


namespace mrs_white_expected_yield_l162_162005

noncomputable def orchard_yield : ℝ :=
  let length_in_feet : ℝ := 10 * 3
  let width_in_feet : ℝ := 30 * 3
  let total_area : ℝ := length_in_feet * width_in_feet
  let half_area : ℝ := total_area / 2
  let tomato_yield : ℝ := half_area * 0.75
  let cucumber_yield : ℝ := half_area * 0.4
  tomato_yield + cucumber_yield

theorem mrs_white_expected_yield :
  orchard_yield = 1552.5 := sorry

end mrs_white_expected_yield_l162_162005


namespace I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l162_162446

-- Define the problems
theorem I_consecutive_integers:
  ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 1 ∧ z = x + 2 :=
sorry

theorem I_consecutive_even_integers:
  ¬ ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 2 ∧ z = x + 4 :=
sorry

theorem II_consecutive_integers:
  ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 1 ∧ z = x + 2 ∧ w = x + 3 :=
sorry

theorem II_consecutive_even_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 :=
sorry

theorem II_consecutive_odd_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ w % 2 = 1 :=
sorry

end I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l162_162446


namespace min_groups_required_l162_162463

-- Define the conditions
def total_children : ℕ := 30
def max_children_per_group : ℕ := 12
def largest_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ d ≤ max_children_per_group

-- Define the property that we are interested in: the minimum number of groups required
def min_num_groups (total : ℕ) (group_size : ℕ) : ℕ := total / group_size

-- Prove the minimum number of groups is 3 given the conditions
theorem min_groups_required : ∃ d, largest_divisor total_children d ∧ min_num_groups total_children d = 3 :=
sorry

end min_groups_required_l162_162463


namespace find_p_from_conditions_l162_162345

variable (p : ℝ) (y x : ℝ)

noncomputable def parabola_eq : Prop := y^2 = 2 * p * x

noncomputable def p_positive : Prop := p > 0

noncomputable def point_on_parabola : Prop := parabola_eq p 1 (p / 4)

theorem find_p_from_conditions (hp : p_positive p) (hpp : point_on_parabola p) : p = Real.sqrt 2 :=
by 
  -- The actual proof goes here
  sorry

end find_p_from_conditions_l162_162345


namespace common_ratio_q_l162_162233

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => geom_seq a q n * q

def sum_geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => sum_geom_seq a q n + geom_seq a q (n + 1)

theorem common_ratio_q (a q : α) (hq : 0 < q) (h_inc : ∀ n, geom_seq a q n < geom_seq a q (n + 1))
  (h1 : geom_seq a q 1 = 2)
  (h2 : sum_geom_seq a q 2 = 7) :
  q = 2 :=
sorry

end common_ratio_q_l162_162233


namespace count_adjacent_pairs_sum_multiple_of_three_l162_162188

def adjacent_digit_sum_multiple_of_three (n : ℕ) : ℕ :=
  -- A function to count the number of pairs with a sum multiple of 3
  sorry

-- Define the sequence from 100 to 999 as digits concatenation
def digit_sequence : List ℕ := List.join (List.map (fun x => x.digits 10) (List.range' 100 900))

theorem count_adjacent_pairs_sum_multiple_of_three :
  adjacent_digit_sum_multiple_of_three digit_sequence.length = 897 :=
sorry

end count_adjacent_pairs_sum_multiple_of_three_l162_162188


namespace binomial_510_510_l162_162888

theorem binomial_510_510 : Nat.choose 510 510 = 1 :=
by
  sorry

end binomial_510_510_l162_162888


namespace min_segments_required_l162_162986

noncomputable def min_segments (n : ℕ) : ℕ := (3 * n - 2 + 1) / 2

theorem min_segments_required (n : ℕ) (h : ∀ (A B : ℕ) (hA : A < n) (hB : B < n) (hAB : A ≠ B), 
  ∃ (C : ℕ), C < n ∧ (C ≠ A) ∧ (C ≠ B)) : 
  min_segments n = ⌈ (3 * n - 2 : ℝ) / 2 ⌉ := 
sorry

end min_segments_required_l162_162986


namespace rectangular_prism_volume_l162_162451

theorem rectangular_prism_volume (h : ℝ) : 
  ∃ (V : ℝ), V = 120 * h :=
by
  sorry

end rectangular_prism_volume_l162_162451


namespace solve_table_assignment_l162_162435

noncomputable def table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) : Prop :=
  let Albert := T_4
  let Bogdan := T_2
  let Vadim := T_1
  let Denis := T_3
  (∀ x, x ∈ Vadim ↔ x ∉ (Albert ∪ Bogdan)) ∧
  (∀ x, x ∈ Denis ↔ x ∉ (Bogdan ∪ Vadim)) ∧
  Albert = T_4 ∧
  Bogdan = T_2 ∧
  Vadim = T_1 ∧
  Denis = T_3

theorem solve_table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) :
  table_assignment T_1 T_2 T_3 T_4 :=
sorry

end solve_table_assignment_l162_162435


namespace arithmetic_sequence_product_l162_162785

noncomputable def a_n (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sequence_product (a_1 d : ℤ) :
  (a_n 4 a_1 d) + (a_n 7 a_1 d) = 2 →
  (a_n 5 a_1 d) * (a_n 6 a_1 d) = -3 →
  a_1 * (a_n 10 a_1 d) = -323 :=
by
  sorry

end arithmetic_sequence_product_l162_162785


namespace transformation_composition_l162_162203

def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

theorem transformation_composition :
  f (g (-1, 2)) = (1, -3) :=
by {
  sorry
}

end transformation_composition_l162_162203


namespace simplify_expression_l162_162110

theorem simplify_expression : (Real.sqrt 12 - |1 - Real.sqrt 3| + (7 + Real.pi)^0) = (Real.sqrt 3 + 2) :=
by
  sorry

end simplify_expression_l162_162110


namespace proof_1_proof_2_l162_162071

-- Definitions of propositions p, q, and r

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (x^2 + (a - 1) * x + a^2 ≤ 0)

def q (a : ℝ) : Prop :=
  2 * a^2 - a > 1

def r (a : ℝ) : Prop :=
  (2 * a - 1) / (a - 2) ≤ 1

-- The given proof problem statement 1
theorem proof_1 (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → (a ∈ Set.Icc (-1) (-1/2) ∪ Set.Ioo (1/3) 1) :=
sorry

-- The given proof problem statement 2
theorem proof_2 (a : ℝ) : ¬ p a → r a :=
sorry

end proof_1_proof_2_l162_162071


namespace middle_aged_selection_l162_162154

def total_teachers := 80 + 160 + 240
def sample_size := 60
def middle_aged_proportion := 160 / total_teachers
def middle_aged_sample := middle_aged_proportion * sample_size

theorem middle_aged_selection : middle_aged_sample = 20 :=
  sorry

end middle_aged_selection_l162_162154


namespace female_students_in_sample_l162_162274

-- Definitions of the given conditions
def male_students : ℕ := 28
def female_students : ℕ := 21
def total_students : ℕ := male_students + female_students
def sample_size : ℕ := 14
def stratified_sampling_fraction : ℚ := (sample_size : ℚ) / (total_students : ℚ)
def female_sample_count : ℚ := stratified_sampling_fraction * (female_students : ℚ)

-- The theorem to prove
theorem female_students_in_sample : female_sample_count = 6 :=
by
  sorry

end female_students_in_sample_l162_162274


namespace clothing_price_reduction_l162_162412

def price_reduction (original_profit_per_piece : ℕ) (original_sales_volume : ℕ) (target_profit : ℕ) (increase_in_sales_per_unit_price_reduction : ℕ) : ℕ :=
  sorry

theorem clothing_price_reduction :
  ∃ x : ℕ, (40 - x) * (20 + 2 * x) = 1200 :=
sorry

end clothing_price_reduction_l162_162412


namespace intercept_condition_l162_162026

theorem intercept_condition (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ x = -c / a ∧ y = -c / b ∧ x = y) → (c = 0 ∨ a = b) :=
by
  sorry

end intercept_condition_l162_162026


namespace points_on_intersecting_lines_l162_162127

def clubsuit (a b : ℝ) := a^3 * b - a * b^3

theorem points_on_intersecting_lines (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = y ∨ x = -y) := 
by
  sorry

end points_on_intersecting_lines_l162_162127


namespace suraj_avg_after_10th_inning_l162_162856

theorem suraj_avg_after_10th_inning (A : ℝ) 
  (h1 : ∀ A : ℝ, (9 * A + 200) / 10 = A + 8) :
  ∀ A : ℝ, A = 120 → (A + 8 = 128) :=
by
  sorry

end suraj_avg_after_10th_inning_l162_162856


namespace derivative_f_l162_162348

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.exp (Real.sin x))

theorem derivative_f (x : ℝ) : deriv f x = ((Real.cos x)^2 - Real.sin x) * (Real.exp (Real.sin x)) :=
by
  sorry

end derivative_f_l162_162348


namespace candle_problem_l162_162691

-- Define the initial heights and burn rates of the candles
def heightA (t : ℝ) : ℝ := 12 - 2 * t
def heightB (t : ℝ) : ℝ := 15 - 3 * t

-- Lean theorem statement for the given problem
theorem candle_problem : ∃ t : ℝ, (heightA t = (1/3) * heightB t) ∧ t = 7 :=
by
  -- This is to keep the theorem statement valid without the proof
  sorry

end candle_problem_l162_162691


namespace third_height_less_than_30_l162_162707

theorem third_height_less_than_30 (h_a h_b : ℝ) (h_a_pos : h_a = 12) (h_b_pos : h_b = 20) : 
    ∃ (h_c : ℝ), h_c < 30 :=
by
  sorry

end third_height_less_than_30_l162_162707


namespace total_spokes_is_60_l162_162060

def num_spokes_front : ℕ := 20
def num_spokes_back : ℕ := 2 * num_spokes_front
def total_spokes : ℕ := num_spokes_front + num_spokes_back

theorem total_spokes_is_60 : total_spokes = 60 :=
by
  sorry

end total_spokes_is_60_l162_162060


namespace books_more_than_movies_l162_162626

-- Define the number of movies and books in the "crazy silly school" series.
def num_movies : ℕ := 14
def num_books : ℕ := 15

-- State the theorem to prove there is 1 more book than movies.
theorem books_more_than_movies : num_books - num_movies = 1 :=
by 
  -- Proof is omitted.
  sorry

end books_more_than_movies_l162_162626


namespace sum_last_two_digits_of_x2012_l162_162711

def sequence_defined (x : ℕ → ℕ) : Prop :=
  (x 1 = 5 ∨ x 1 = 7) ∧ ∀ k ≥ 1, (x (k+1) = 5^(x k) ∨ x (k+1) = 7^(x k))

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def possible_values : List ℕ :=
  [25, 7, 43]

theorem sum_last_two_digits_of_x2012 {x : ℕ → ℕ} (h : sequence_defined x) :
  List.sum (List.map last_two_digits [25, 7, 43]) = 75 :=
  by
    sorry

end sum_last_two_digits_of_x2012_l162_162711


namespace smallest_unit_of_money_correct_l162_162001

noncomputable def smallest_unit_of_money (friends : ℕ) (total_bill paid_amount : ℚ) : ℚ :=
  if (total_bill % friends : ℚ) = 0 then
    total_bill / friends
  else
    1 % 100

theorem smallest_unit_of_money_correct :
  smallest_unit_of_money 9 124.15 124.11 = 1 % 100 := 
by
  sorry

end smallest_unit_of_money_correct_l162_162001


namespace no_n_satisfies_mod_5_l162_162908

theorem no_n_satisfies_mod_5 (n : ℤ) : (n^3 + 2*n - 1) % 5 ≠ 0 :=
by
  sorry

end no_n_satisfies_mod_5_l162_162908


namespace cost_one_dozen_pens_l162_162131

variable (cost_of_pen cost_of_pencil : ℝ)
variable (ratio : ℝ)
variable (dozen_pens_cost : ℝ)

axiom cost_equation : 3 * cost_of_pen + 5 * cost_of_pencil = 200
axiom ratio_pen_pencil : cost_of_pen = 5 * cost_of_pencil

theorem cost_one_dozen_pens : dozen_pens_cost = 12 * cost_of_pen := 
  by
    sorry

end cost_one_dozen_pens_l162_162131


namespace factorize_expr_l162_162873

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l162_162873


namespace shortest_chord_line_through_P_longest_chord_line_through_P_l162_162147

theorem shortest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = 1/2 * x + 5/2 → a * x + b * y + c = 0)
  ∧ (a = 1) ∧ (b = -2) ∧ (c = 5) := sorry

theorem longest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = -2 * x → a * x + b * y + c = 0)
  ∧ (a = 2) ∧ (b = 1) ∧ (c = 0) := sorry

end shortest_chord_line_through_P_longest_chord_line_through_P_l162_162147


namespace isosceles_triangle_angle_measure_l162_162542

theorem isosceles_triangle_angle_measure
  (isosceles : Triangle → Prop)
  (exterior_angles : Triangle → ℝ → ℝ → Prop)
  (ratio_1_to_4 : ∀ {T : Triangle} {a b : ℝ}, exterior_angles T a b → b = 4 * a)
  (interior_angles : Triangle → ℝ → ℝ → ℝ → Prop) :
  ∀ (T : Triangle), isosceles T → ∃ α β γ : ℝ, interior_angles T α β γ ∧ α = 140 ∧ β = 20 ∧ γ = 20 := 
by
  sorry

end isosceles_triangle_angle_measure_l162_162542


namespace second_integer_value_l162_162897

theorem second_integer_value (n : ℚ) (h : (n - 1) + (n + 1) + (n + 2) = 175) : n = 57 + 2 / 3 :=
by
  sorry

end second_integer_value_l162_162897


namespace tank_capacity_l162_162314

variable (C : ℝ)

noncomputable def leak_rate := C / 6 -- litres per hour
noncomputable def inlet_rate := 6 * 60 -- litres per hour
noncomputable def net_emptying_rate := C / 12 -- litres per hour

theorem tank_capacity : 
  (360 - leak_rate C = net_emptying_rate C) → 
  C = 1440 :=
by 
  sorry

end tank_capacity_l162_162314


namespace xyz_inequality_l162_162457

theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end xyz_inequality_l162_162457


namespace no_positive_integer_solutions_l162_162016

theorem no_positive_integer_solutions :
  ∀ (A : ℕ), 1 ≤ A ∧ A ≤ 9 → ¬∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = A * 10 + A ∧ x + y = 10 * A + 1 := by
  sorry

end no_positive_integer_solutions_l162_162016


namespace discount_difference_l162_162411

-- Definitions based on given conditions
def original_bill : ℝ := 8000
def single_discount_rate : ℝ := 0.30
def first_successive_discount_rate : ℝ := 0.26
def second_successive_discount_rate : ℝ := 0.05

-- Calculations based on conditions
def single_discount_final_amount := original_bill * (1 - single_discount_rate)
def first_successive_discount_final_amount := original_bill * (1 - first_successive_discount_rate)
def complete_successive_discount_final_amount := 
  first_successive_discount_final_amount * (1 - second_successive_discount_rate)

-- Proof statement
theorem discount_difference :
  single_discount_final_amount - complete_successive_discount_final_amount = 24 := 
  by
    -- Proof to be provided
    sorry

end discount_difference_l162_162411


namespace sum_of_numbers_is_60_l162_162578

-- Define the primary values used in the conditions
variables (a b c : ℝ)

-- Define the conditions in the problem
def mean_condition_1 : Prop := (a + b + c) / 3 = a + 20
def mean_condition_2 : Prop := (a + b + c) / 3 = c - 30
def median_condition : Prop := b = 10

-- Prove that the sum of the numbers is 60 given the conditions
theorem sum_of_numbers_is_60 (hac1 : mean_condition_1 a b c) (hac2 : mean_condition_2 a b c) (hbm : median_condition b) : a + b + c = 60 :=
by 
  sorry

end sum_of_numbers_is_60_l162_162578


namespace opposite_of_neg_twelve_l162_162261

def opposite (n : Int) : Int := -n

theorem opposite_of_neg_twelve : opposite (-12) = 12 := by
  sorry

end opposite_of_neg_twelve_l162_162261


namespace capital_growth_rate_l162_162489

theorem capital_growth_rate
  (loan_amount : ℝ) (interest_rate : ℝ) (repayment_period : ℝ) (surplus : ℝ) (growth_rate : ℝ) :
  loan_amount = 2000000 ∧ interest_rate = 0.08 ∧ repayment_period = 2 ∧ surplus = 720000 ∧
  (loan_amount * (1 + growth_rate)^repayment_period = loan_amount * (1 + interest_rate) + surplus) →
  growth_rate = 0.2 :=
by
  sorry

end capital_growth_rate_l162_162489


namespace combination_8_5_l162_162753

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l162_162753


namespace find_k_for_circle_of_radius_8_l162_162631

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end find_k_for_circle_of_radius_8_l162_162631


namespace time_after_1450_minutes_l162_162987

theorem time_after_1450_minutes (initial_time_in_minutes : ℕ := 360) (minutes_to_add : ℕ := 1450) : 
  (initial_time_in_minutes + minutes_to_add) % (24 * 60) = 370 :=
by
  -- Given (initial_time_in_minutes = 360 which is 6:00 a.m., minutes_to_add = 1450)
  -- Compute the time in minutes after 1450 minutes
  -- 24 hours = 1440 minutes, so (360 + 1450) % 1440 should equal 370
  sorry

end time_after_1450_minutes_l162_162987


namespace sunset_time_range_l162_162658

theorem sunset_time_range (h : ℝ) :
  ¬(h ≥ 7) ∧ ¬(h ≤ 8) ∧ ¬(h ≤ 6) ↔ h ∈ Set.Ioi 8 :=
by
  sorry

end sunset_time_range_l162_162658


namespace magic_box_problem_l162_162386

theorem magic_box_problem (m : ℝ) :
  (m^2 - 2*m - 1 = 2) → (m = 3 ∨ m = -1) :=
by
  intro h
  sorry

end magic_box_problem_l162_162386


namespace problem_l162_162528

theorem problem (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, x^5 = a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5) →
  a_3 = -10 ∧ a_1 + a_3 + a_5 = -16 :=
by 
  sorry

end problem_l162_162528


namespace area_of_rectangle_l162_162040

theorem area_of_rectangle (width length : ℝ) (h_width : width = 5.4) (h_length : length = 2.5) : width * length = 13.5 :=
by
  -- We are given that the width is 5.4 and the length is 2.5
  -- We need to show that the area (width * length) is 13.5
  sorry

end area_of_rectangle_l162_162040


namespace common_ratio_l162_162769

theorem common_ratio (a S r : ℝ) (h1 : S = a / (1 - r))
  (h2 : ar^5 / (1 - r) = S / 81) : r = 1 / 3 :=
sorry

end common_ratio_l162_162769


namespace smallest_a_value_l162_162666

theorem smallest_a_value 
  (a b c : ℚ) 
  (a_pos : a > 0)
  (vertex_condition : ∃(x₀ y₀ : ℚ), x₀ = -1/3 ∧ y₀ = -4/3 ∧ y = a * (x + x₀)^2 + y₀)
  (integer_condition : ∃(n : ℤ), a + b + c = n)
  : a = 3/16 := 
sorry

end smallest_a_value_l162_162666


namespace score_ordering_l162_162562

-- Definition of the problem conditions in Lean 4:
def condition1 (Q K : ℝ) : Prop := Q ≠ K
def condition2 (M Q S K : ℝ) : Prop := M < Q ∧ M < S ∧ M < K
def condition3 (S Q M K : ℝ) : Prop := S > Q ∧ S > M ∧ S > K

-- Theorem statement in Lean 4:
theorem score_ordering (M Q S K : ℝ) (h1 : condition1 Q K) (h2 : condition2 M Q S K) (h3 : condition3 S Q M K) : 
  M < Q ∧ Q < S :=
by
  sorry

end score_ordering_l162_162562


namespace max_three_kopecks_l162_162397

def is_coin_placement_correct (n1 n2 n3 : ℕ) : Prop :=
  -- Conditions for the placement to be valid
  ∀ (i j : ℕ), i < j → 
  ((j - i > 1 → n1 = 0) ∧ (j - i > 2 → n2 = 0) ∧ (j - i > 3 → n3 = 0))

theorem max_three_kopecks (n1 n2 n3 : ℕ) (h : n1 + n2 + n3 = 101) (placement_correct : is_coin_placement_correct n1 n2 n3) :
  n3 = 25 ∨ n3 = 26 :=
sorry

end max_three_kopecks_l162_162397


namespace algebraic_inequality_l162_162349

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  |a| > 1 ∧ |b| > 1 ∧ |c| > 1 ∧ |d| > 1 ∧
  a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0 →
  (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) + (1 / (d - 1)) > 0

theorem algebraic_inequality (a b c d : ℝ) :
  problem_statement a b c d :=
by
  sorry

end algebraic_inequality_l162_162349


namespace inequality_solution_set_l162_162352

theorem inequality_solution_set (x : ℝ) : (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := sorry

end inequality_solution_set_l162_162352


namespace susannah_swims_more_than_camden_l162_162088

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end susannah_swims_more_than_camden_l162_162088


namespace stream_current_speed_l162_162949

theorem stream_current_speed (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (1.5 * r + w) + 2 = 18 / (1.5 * r - w)) : w = 2.5 :=
by
  -- Translate the equations from the problem conditions directly.
  sorry

end stream_current_speed_l162_162949


namespace ratio_flowers_l162_162304

theorem ratio_flowers (flowers_monday flowers_tuesday flowers_week total_flowers flowers_friday : ℕ)
    (h_monday : flowers_monday = 4)
    (h_tuesday : flowers_tuesday = 8)
    (h_total : total_flowers = 20)
    (h_week : total_flowers = flowers_monday + flowers_tuesday + flowers_friday) :
    flowers_friday / flowers_monday = 2 :=
by
  sorry

end ratio_flowers_l162_162304


namespace projectiles_meet_time_l162_162834

def distance : ℕ := 2520
def speed1 : ℕ := 432
def speed2 : ℕ := 576
def combined_speed : ℕ := speed1 + speed2

theorem projectiles_meet_time :
  (distance * 60) / combined_speed = 150 := 
by
  sorry

end projectiles_meet_time_l162_162834


namespace larger_integer_value_l162_162798

theorem larger_integer_value (x y : ℕ) (h1 : (4 * x)^2 - 2 * x = 8100) (h2 : x + 10 = 2 * y) : x = 22 :=
by
  sorry

end larger_integer_value_l162_162798


namespace profit_maximization_l162_162328

-- Define the conditions 
variable (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5)

-- Expression for yield ω
noncomputable def yield (x : ℝ) : ℝ := 4 - (3 / (x + 1))

-- Expression for profit function L(x)
noncomputable def profit (x : ℝ) : ℝ := 16 * yield x - x - 2 * x

-- Theorem stating the profit function expression and its maximum
theorem profit_maximization (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5) :
  profit x = 64 - 48 / (x + 1) - 3 * x ∧ 
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ 5 → profit x₀ ≤ profit 3) :=
sorry

end profit_maximization_l162_162328


namespace quadratic_inequality_solution_l162_162929

theorem quadratic_inequality_solution :
  {x : ℝ | 2 * x ^ 2 - x - 3 > 0} = {x : ℝ | x > 3 / 2 ∨ x < -1} :=
sorry

end quadratic_inequality_solution_l162_162929


namespace percentage_less_than_m_add_d_l162_162770

def symmetric_about_mean (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P m - x = P m + x

def within_one_stdev (P : ℝ → ℝ) (m d : ℝ) : Prop :=
  P m - d = 0.68 ∧ P m + d = 0.68

theorem percentage_less_than_m_add_d 
  (P : ℝ → ℝ) (m d : ℝ) 
  (symm : symmetric_about_mean P m)
  (within_stdev : within_one_stdev P m d) : 
  ∃ f, f = 0.84 :=
by
  sorry

end percentage_less_than_m_add_d_l162_162770


namespace average_age_of_team_l162_162839

variable (A : ℕ)
variable (captain_age : ℕ)
variable (wicket_keeper_age : ℕ)
variable (vice_captain_age : ℕ)

-- Conditions
def team_size := 11
def captain := 25
def wicket_keeper := captain + 3
def vice_captain := wicket_keeper - 4
def remaining_players := team_size - 3
def remaining_average := A - 1

-- Prove the average age of the whole team
theorem average_age_of_team :
  captain_age = 25 ∧
  wicket_keeper_age = captain_age + 3 ∧
  vice_captain_age = wicket_keeper_age - 4 ∧
  11 * A = (captain + wicket_keeper + vice_captain) + 8 * (A - 1) → 
  A = 23 :=
by
  sorry

end average_age_of_team_l162_162839


namespace day_after_2_pow_20_is_friday_l162_162252

-- Define the given conditions
def today_is_monday : ℕ := 0 -- Assuming Monday is represented by 0

-- Define the number of days after \(2^{20}\) days
def days_after : ℕ := 2^20

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the function to find the day of the week after a given number of days
def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % days_in_week

-- The theorem to prove
theorem day_after_2_pow_20_is_friday :
  day_of_week today_is_monday days_after = 5 := -- Friday is represented by 5 here
sorry

end day_after_2_pow_20_is_friday_l162_162252


namespace prob_A_and_B_truth_l162_162377

-- Define the probabilities
def prob_A_truth := 0.70
def prob_B_truth := 0.60

-- State the theorem
theorem prob_A_and_B_truth : prob_A_truth * prob_B_truth = 0.42 :=
by
  sorry

end prob_A_and_B_truth_l162_162377


namespace binomial_expansion_l162_162671

theorem binomial_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (1 + 2 * 1)^5 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 ∧
  (1 + 2 * -1)^5 = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 → 
  a_0 + a_2 + a_4 = 121 :=
by
  intro h
  let h₁ := h.1
  let h₂ := h.2
  sorry

end binomial_expansion_l162_162671


namespace quadratic_root_exists_l162_162445

theorem quadratic_root_exists (a b c : ℝ) : 
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) :=
by sorry

end quadratic_root_exists_l162_162445


namespace inequality_true_l162_162943

variables {a b : ℝ}
variables (c : ℝ)

theorem inequality_true (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 :=
by sorry

end inequality_true_l162_162943


namespace sequences_identity_l162_162152

variables {α β γ : ℤ}
variables {a b : ℕ → ℤ}

-- Define the recurrence relations conditions
def conditions (a b : ℕ → ℤ) (α β γ : ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 1 ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n) ∧
  α < γ ∧ α * γ = β^2 + 1

-- Define the main statement
theorem sequences_identity (a b : ℕ → ℤ) 
  (h : conditions a b α β γ) (m n : ℕ) :
  a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end sequences_identity_l162_162152


namespace find_b_l162_162028

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 :=
sorry

end find_b_l162_162028


namespace Mrs_Early_speed_l162_162172

noncomputable def speed_to_reach_on_time (distance : ℝ) (ideal_time : ℝ) : ℝ := distance / ideal_time

theorem Mrs_Early_speed:
  ∃ (d t : ℝ), 
    (d = 50 * (t + 5/60)) ∧ 
    (d = 80 * (t - 7/60)) ∧ 
    (speed_to_reach_on_time d t = 59) := sorry

end Mrs_Early_speed_l162_162172


namespace equal_donations_amount_l162_162396

def raffle_tickets_sold := 25
def cost_per_ticket := 2
def total_raised := 100
def single_donation := 20
def amount_equal_donations (D : ℕ) : Prop := 2 * D + single_donation = total_raised - (raffle_tickets_sold * cost_per_ticket)

theorem equal_donations_amount (D : ℕ) (h : amount_equal_donations D) : D = 15 :=
  sorry

end equal_donations_amount_l162_162396


namespace distance_from_desk_to_fountain_l162_162222

-- Problem definitions with given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Formulate the proof problem as a Lean theorem statement
theorem distance_from_desk_to_fountain :
  total_distance / trips = 30 :=
by
  sorry

end distance_from_desk_to_fountain_l162_162222


namespace total_animals_l162_162255

theorem total_animals (total_legs : ℕ) (number_of_sheep : ℕ)
  (legs_per_chicken : ℕ) (legs_per_sheep : ℕ)
  (H1 : total_legs = 60) 
  (H2 : number_of_sheep = 10)
  (H3 : legs_per_chicken = 2)
  (H4 : legs_per_sheep = 4) : 
  number_of_sheep + (total_legs - number_of_sheep * legs_per_sheep) / legs_per_chicken = 20 :=
by {
  sorry
}

end total_animals_l162_162255


namespace correct_equation_for_programmers_l162_162788

theorem correct_equation_for_programmers (x : ℕ) 
  (hB : x > 0) 
  (programmer_b_speed : ℕ := x) 
  (programmer_a_speed : ℕ := 2 * x) 
  (data : ℕ := 2640) :
  (data / programmer_a_speed = data / programmer_b_speed - 120) :=
by
  -- sorry is used to skip the proof, focus on the statement
  sorry

end correct_equation_for_programmers_l162_162788


namespace marbles_problem_l162_162008

theorem marbles_problem (a : ℚ) (h1: 34 * a = 156) : a = 78 / 17 := 
by
  sorry

end marbles_problem_l162_162008


namespace reciprocal_of_2023_l162_162557

theorem reciprocal_of_2023 :
  1 / 2023 = 1 / (2023 : ℝ) :=
by
  sorry

end reciprocal_of_2023_l162_162557


namespace fraction_of_menu_safely_eaten_l162_162764

-- Given conditions
def VegetarianDishes := 6
def GlutenContainingVegetarianDishes := 5
def TotalDishes := 3 * VegetarianDishes

-- Derived information
def GlutenFreeVegetarianDishes := VegetarianDishes - GlutenContainingVegetarianDishes

-- Question: What fraction of the menu can Sarah safely eat?
theorem fraction_of_menu_safely_eaten : 
  (GlutenFreeVegetarianDishes / TotalDishes) = 1 / 18 :=
by
  sorry

end fraction_of_menu_safely_eaten_l162_162764


namespace intersection_of_M_and_N_l162_162963

open Set

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
by {
  sorry
}

end intersection_of_M_and_N_l162_162963


namespace unique_solution_exists_l162_162602

theorem unique_solution_exists (ell : ℚ) (h : ell ≠ -2) : 
  (∃! x : ℚ, (x + 3) / (ell * x + 2) = x) ↔ ell = -1 / 12 := 
by
  sorry

end unique_solution_exists_l162_162602


namespace find_m_l162_162759

theorem find_m (m : ℤ) (a := (3, m)) (b := (1, -2)) (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) : m = -1 :=
sorry

end find_m_l162_162759


namespace semi_circle_radius_l162_162027

theorem semi_circle_radius (P : ℝ) (π : ℝ) (r : ℝ) (hP : P = 10.797344572538567) (hπ : π = 3.14159) :
  (π + 2) * r = P → r = 2.1 :=
by
  intro h
  sorry

end semi_circle_radius_l162_162027


namespace correct_operation_l162_162641

variable (a b : ℝ)

theorem correct_operation : (a^2 * a^3 = a^5) :=
by sorry

end correct_operation_l162_162641


namespace compute_sum_l162_162139

open BigOperators

theorem compute_sum : 
  (1 / 2 ^ 2010 : ℝ) * ∑ n in Finset.range 1006, (-3 : ℝ) ^ n * (Nat.choose 2010 (2 * n)) = -1 / 2 :=
by
  sorry

end compute_sum_l162_162139


namespace find_b_days_l162_162399

theorem find_b_days 
  (a_days b_days c_days : ℕ)
  (a_wage b_wage c_wage : ℕ)
  (total_earnings : ℕ)
  (ratio_3_4_5 : a_wage * 5 = b_wage * 4 ∧ b_wage * 5 = c_wage * 4 ∧ a_wage * 5 = c_wage * 3)
  (c_wage_val : c_wage = 110)
  (a_days_val : a_days = 6)
  (c_days_val : c_days = 4) 
  (total_earnings_val : total_earnings = 1628)
  (earnings_eq : a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings) :
  b_days = 9 := by
  sorry

end find_b_days_l162_162399


namespace range_of_m_value_of_m_l162_162606

variable (α β m : ℝ)

open Real

-- Conditions: α and β are positive roots.
def quadratic_roots (α β m : ℝ) : Prop :=
  (α > 0) ∧ (β > 0) ∧ (α + β = 1 - 2*m) ∧ (α * β = m^2)

-- Part 1: Range of values for m.
theorem range_of_m (h : quadratic_roots α β m) : m ≤ 1/4 ∧ m ≠ 0 :=
sorry

-- Part 2: Given α^2 + β^2 = 49, find the value of m.
theorem value_of_m (h : quadratic_roots α β m) (h' : α^2 + β^2 = 49) : m = -4 :=
sorry

end range_of_m_value_of_m_l162_162606


namespace integer_sequence_existence_l162_162979

theorem integer_sequence_existence
  (n : ℕ) (a : ℕ → ℤ) (A B C : ℤ) 
  (h1 : (a 1 < A ∧ A < B ∧ B < a n) ∨ (a 1 > A ∧ A > B ∧ B > a n))
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n - 1 → (a (i + 1) - a i ≤ 1 ∨ a (i + 1) - a i ≥ -1))
  (h3 : A ≤ C ∧ C ≤ B ∨ A ≥ C ∧ C ≥ B) :
  ∃ i, 1 < i ∧ i < n ∧ a i = C := sorry

end integer_sequence_existence_l162_162979


namespace total_books_l162_162547

def initial_books : ℝ := 41.0
def first_addition : ℝ := 33.0
def second_addition : ℝ := 2.0

theorem total_books (h1 : initial_books = 41.0) (h2 : first_addition = 33.0) (h3 : second_addition = 2.0) :
  initial_books + first_addition + second_addition = 76.0 := 
by
  -- placeholders for the proof steps, omitting the detailed steps as instructed
  sorry

end total_books_l162_162547


namespace find_digits_l162_162717

/-- 
  Find distinct digits A, B, C, and D such that 9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B).
 -/
theorem find_digits
  (A B C D : ℕ)
  (hA : A ≠ B) (hA : A ≠ C) (hA : A ≠ D)
  (hB : B ≠ C) (hB : B ≠ D)
  (hC : C ≠ D)
  (hNonZeroB : B ≠ 0) :
  9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B) ↔ (A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7) := by
  sorry

end find_digits_l162_162717


namespace differential_solution_correct_l162_162637

noncomputable def y (x : ℝ) : ℝ := (x + 1)^2

theorem differential_solution_correct : 
  (∀ x : ℝ, deriv (deriv y) x = 2) ∧ y 0 = 1 ∧ (deriv y 0) = 2 := 
by
  sorry

end differential_solution_correct_l162_162637


namespace sam_quarters_mowing_lawns_l162_162336

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end sam_quarters_mowing_lawns_l162_162336


namespace geometric_proportion_exists_l162_162287

theorem geometric_proportion_exists (x y : ℝ) (h1 : x + (24 - x) = 24) 
  (h2 : y + (16 - y) = 16) (h3 : x^2 + y^2 + (16 - y)^2 + (24 - x)^2 = 580) : 
  (21 / 7 = 9 / 3) :=
  sorry

end geometric_proportion_exists_l162_162287


namespace number_of_outliers_l162_162257

def data_set : List ℕ := [4, 23, 27, 27, 35, 37, 37, 39, 47, 53]

def Q1 : ℕ := 27
def Q3 : ℕ := 39

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - (3 * IQR / 2)
def upper_threshold : ℕ := Q3 + (3 * IQR / 2)

def outliers (s : List ℕ) (low high : ℕ) : List ℕ :=
  s.filter (λ x => x < low ∨ x > high)

theorem number_of_outliers :
  outliers data_set lower_threshold upper_threshold = [4] :=
by
  sorry

end number_of_outliers_l162_162257


namespace arithmetic_geometric_mean_inequality_l162_162513

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_geometric_mean_inequality_l162_162513


namespace decreasing_power_function_l162_162081

theorem decreasing_power_function (n : ℝ) (f : ℝ → ℝ) 
    (h : ∀ x > 0, f x = (n^2 - n - 1) * x^n) 
    (h_decreasing : ∀ x > 0, f x > f (x + 1)) : n = -1 :=
sorry

end decreasing_power_function_l162_162081


namespace pq_difference_l162_162964

theorem pq_difference (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end pq_difference_l162_162964


namespace plane_speed_ratio_train_l162_162793

def distance (speed time : ℝ) := speed * time

theorem plane_speed_ratio_train (x y z : ℝ)
  (h_train : distance x 20 = distance y 10)
  (h_wait_time : z > 5)
  (h_plane_meet_train : distance y (8/9) = distance x (z + 8/9)) :
  y = 10 * x :=
by {
  sorry
}

end plane_speed_ratio_train_l162_162793


namespace twelve_star_three_eq_four_star_eight_eq_star_assoc_l162_162593

def star (a b : ℕ) : ℕ := 10^a * 10^b

theorem twelve_star_three_eq : star 12 3 = 10^15 :=
by 
  -- Proof here
  sorry

theorem four_star_eight_eq : star 4 8 = 10^12 :=
by 
  -- Proof here
  sorry

theorem star_assoc (a b c : ℕ) : star (a + b) c = star a (b + c) :=
by 
  -- Proof here
  sorry

end twelve_star_three_eq_four_star_eight_eq_star_assoc_l162_162593


namespace scientific_notation_of_100000_l162_162369

theorem scientific_notation_of_100000 :
  100000 = 1 * 10^5 :=
by sorry

end scientific_notation_of_100000_l162_162369


namespace dog_tail_length_l162_162149

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end dog_tail_length_l162_162149


namespace find_value_l162_162272

-- Defining the known conditions
def number : ℕ := 20
def half (n : ℕ) : ℕ := n / 2
def value_added (V : ℕ) : Prop := half number + V = 17

-- Proving that the value added to half the number is 7
theorem find_value : value_added 7 :=
by
  -- providing the proof for the theorem
  -- skipping the proof steps with sorry
  sorry

end find_value_l162_162272


namespace g_is_odd_l162_162678

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end g_is_odd_l162_162678


namespace beth_sheep_l162_162158

-- Definition: number of sheep Beth has (B)
variable (B : ℕ)

-- Condition 1: Aaron has 7 times as many sheep as Beth
def Aaron_sheep (B : ℕ) := 7 * B

-- Condition 2: Together, Aaron and Beth have 608 sheep
axiom together_sheep : B + Aaron_sheep B = 608

-- Theorem: Prove that Beth has 76 sheep
theorem beth_sheep : B = 76 :=
sorry

end beth_sheep_l162_162158


namespace steve_has_7_fewer_b_berries_l162_162045

-- Define the initial number of berries Stacy has
def stacy_initial_berries : ℕ := 32

-- Define the number of berries Steve takes from Stacy
def steve_takes : ℕ := 4

-- Define the initial number of berries Steve has
def steve_initial_berries : ℕ := 21

-- Using the given conditions, prove that Steve has 7 fewer berries compared to Stacy's initial amount
theorem steve_has_7_fewer_b_berries :
  stacy_initial_berries - (steve_initial_berries + steve_takes) = 7 := 
by
  sorry

end steve_has_7_fewer_b_berries_l162_162045


namespace sum_first_10_mod_8_is_7_l162_162064

-- Define the sum of the first 10 positive integers
def sum_first_10 : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10

-- Define the divisor
def divisor : ℕ := 8

-- Prove that the remainder of the sum of the first 10 positive integers divided by 8 is 7
theorem sum_first_10_mod_8_is_7 : sum_first_10 % divisor = 7 :=
by
  sorry

end sum_first_10_mod_8_is_7_l162_162064


namespace pyramid_coloring_methods_l162_162866

theorem pyramid_coloring_methods : 
  ∀ (P A B C D : ℕ),
    (P ≠ A) ∧ (P ≠ B) ∧ (P ≠ C) ∧ (P ≠ D) ∧
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
    (P < 5) ∧ (A < 5) ∧ (B < 5) ∧ (C < 5) ∧ (D < 5) →
  ∃! (num_methods : ℕ), num_methods = 420 :=
by
  sorry

end pyramid_coloring_methods_l162_162866


namespace sunzi_classic_equation_l162_162989

theorem sunzi_classic_equation (x : ℕ) : 3 * (x - 2) = 2 * x + 9 :=
  sorry

end sunzi_classic_equation_l162_162989


namespace math_problem_l162_162993

noncomputable def condition1 (a b : ℤ) : Prop :=
  |2 + a| + |b - 3| = 0

noncomputable def condition2 (c d : ℝ) : Prop :=
  1 / c = -d

noncomputable def condition3 (e : ℤ) : Prop :=
  e = -5

theorem math_problem (a b e : ℤ) (c d : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 c d) 
  (h3 : condition3 e) : 
  -a^b + 1 / c - e + d = 13 :=
by
  sorry

end math_problem_l162_162993


namespace functional_expression_y_x_maximize_profit_price_reduction_and_profit_l162_162273

-- Define the conditions
variable (C_selling C_cost : ℝ := 80) (C_costComponent : ℝ := 30) (initialSales : ℝ := 600) 
variable (dec_price : ℝ := 2) (inc_sales : ℝ := 30)
variable (decrease x : ℝ)

-- Define and prove part 1: Functional expression between y and x
theorem functional_expression_y_x : (decrease : ℝ) → (15 * decrease + initialSales : ℝ) = (inc_sales / dec_price * decrease + initialSales) :=
by sorry

-- Define the function for weekly profit
def weekly_profit (x : ℝ) : ℝ := 
  let selling_price := C_selling - x
  let cost_price := C_costComponent
  let sales_volume := 15 * x + initialSales
  (selling_price - cost_price) * sales_volume

-- Prove the condition for maximizing weekly sales profit
theorem maximize_profit_price_reduction_and_profit : 
  (∀ x : ℤ, x % 2 = 0 → weekly_profit x ≤ 30360) ∧
  weekly_profit 4 = 30360 ∧ 
  weekly_profit 6 = 30360 :=
by sorry

end functional_expression_y_x_maximize_profit_price_reduction_and_profit_l162_162273


namespace set_of_values_a_l162_162469

theorem set_of_values_a (a : ℝ) : (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ≤ 2) :=
by
  sorry

end set_of_values_a_l162_162469


namespace problem1_problem2_l162_162381

-- For problem (1)
noncomputable def f (x : ℝ) := Real.sqrt ((1 - x) / (1 + x))

theorem problem1 (α : ℝ) (h_alpha : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

-- For problem (2)
theorem problem2 : Real.sin (Real.pi * 50 / 180) * (1 + Real.sqrt 3 * Real.tan (Real.pi * 10 / 180)) = 1 := by
  sorry

end problem1_problem2_l162_162381


namespace students_not_invited_count_l162_162367

-- Define the total number of students
def total_students : ℕ := 30

-- Define the number of students not invited to the event
def not_invited_students : ℕ := 14

-- Define the sets representing different levels of friends of Anna
-- This demonstrates that the total invited students can be derived from given conditions

def anna_immediate_friends : ℕ := 4
def anna_second_level_friends : ℕ := (12 - anna_immediate_friends)
def anna_third_level_friends : ℕ := (16 - 12)

-- Define total invited students
def invited_students : ℕ := 
  anna_immediate_friends + 
  anna_second_level_friends +
  anna_third_level_friends

-- Prove that the number of not invited students is 14
theorem students_not_invited_count : (total_students - invited_students) = not_invited_students :=
by
  sorry

end students_not_invited_count_l162_162367


namespace find_doodads_produced_in_four_hours_l162_162693

theorem find_doodads_produced_in_four_hours :
  ∃ (n : ℕ),
    (∀ (workers hours widgets doodads : ℕ),
      (workers = 150 ∧ hours = 2 ∧ widgets = 800 ∧ doodads = 500) ∨
      (workers = 100 ∧ hours = 3 ∧ widgets = 750 ∧ doodads = 600) ∨
      (workers = 80  ∧ hours = 4 ∧ widgets = 480 ∧ doodads = n)
    ) → n = 640 :=
sorry

end find_doodads_produced_in_four_hours_l162_162693


namespace girls_attending_ball_l162_162480

theorem girls_attending_ball (g b : ℕ) 
    (h1 : g + b = 1500) 
    (h2 : 3 * g / 4 + 2 * b / 3 = 900) : 
    g = 1200 ∧ 3 * 1200 / 4 = 900 := 
by
  sorry

end girls_attending_ball_l162_162480


namespace inequality_not_less_than_four_by_at_least_one_l162_162305

-- Definitions based on the conditions
def not_less_than_by_at_least (y : ℝ) (a b : ℝ) : Prop := y - a ≥ b

-- Problem statement (theorem) based on the given question and correct answer
theorem inequality_not_less_than_four_by_at_least_one (y : ℝ) :
  not_less_than_by_at_least y 4 1 → y ≥ 5 :=
by
  sorry

end inequality_not_less_than_four_by_at_least_one_l162_162305


namespace sum_of_ages_l162_162868

variables (P M Mo : ℕ)

def age_ratio_PM := 3 * M = 5 * P
def age_ratio_MMo := 3 * Mo = 5 * M
def age_difference := Mo = P + 64

theorem sum_of_ages : age_ratio_PM P M → age_ratio_MMo M Mo → age_difference P Mo → P + M + Mo = 196 :=
by
  intros h1 h2 h3
  sorry

end sum_of_ages_l162_162868


namespace feet_more_than_heads_l162_162141

def num_hens := 50
def num_goats := 45
def num_camels := 8
def num_keepers := 15

def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2

def total_heads := num_hens + num_goats + num_camels + num_keepers
def total_feet := (num_hens * feet_per_hen) + (num_goats * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)

-- Theorem to prove:
theorem feet_more_than_heads : total_feet - total_heads = 224 := by
  -- proof goes here
  sorry

end feet_more_than_heads_l162_162141


namespace cream_cheese_cost_l162_162129

theorem cream_cheese_cost
  (B C : ℝ)
  (h1 : 2 * B + 3 * C = 12)
  (h2 : 4 * B + 2 * C = 14) :
  C = 2.5 :=
by
  sorry

end cream_cheese_cost_l162_162129


namespace book_pages_l162_162982

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end book_pages_l162_162982


namespace men_in_first_group_l162_162422

theorem men_in_first_group (M : ℕ) : (M * 18 = 27 * 24) → M = 36 :=
by
  sorry

end men_in_first_group_l162_162422


namespace infinite_n_perfect_squares_l162_162234

-- Define the condition that k is a positive natural number and k >= 2
variable (k : ℕ) (hk : 2 ≤ k) 

-- Define the statement asserting the existence of infinitely many n such that both kn + 1 and (k+1)n + 1 are perfect squares
theorem infinite_n_perfect_squares : ∀ k : ℕ, (2 ≤ k) → ∃ n : ℕ, ∀ m : ℕ, (2 ≤ k) → k * n + 1 = m * m ∧ (k + 1) * n + 1 = (m + k) * (m + k) := 
by
  sorry

end infinite_n_perfect_squares_l162_162234


namespace unique_solutions_l162_162827

noncomputable def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ∣ (b^4 + 1) ∧ b ∣ (a^4 + 1) ∧ (Nat.floor (Real.sqrt a) = Nat.floor (Real.sqrt b))

theorem unique_solutions :
  ∀ (a b : ℕ), is_solution a b → (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by 
  sorry

end unique_solutions_l162_162827


namespace cost_prices_l162_162192

variable {C1 C2 : ℝ}

theorem cost_prices (h1 : 0.30 * C1 - 0.15 * C1 = 120) (h2 : 0.25 * C2 - 0.10 * C2 = 150) :
  C1 = 800 ∧ C2 = 1000 := 
by
  sorry

end cost_prices_l162_162192


namespace sum_real_imag_parts_eq_l162_162545

noncomputable def z (a b : ℂ) : ℂ := a / b

theorem sum_real_imag_parts_eq (z : ℂ) (h : z * (2 + I) = 2 * I - 1) : 
  (z.re + z.im) = 1 / 5 :=
sorry

end sum_real_imag_parts_eq_l162_162545


namespace starting_number_of_three_squares_less_than_2300_l162_162057

theorem starting_number_of_three_squares_less_than_2300 : 
  ∃ n1 n2 n3 : ℕ, n1 < n2 ∧ n2 < n3 ∧ n3^2 < 2300 ∧ n2^2 < 2300 ∧ n1^2 < 2300 ∧ n3^2 ≥ 2209 ∧ n2^2 ≥ 2116 ∧ n1^2 = 2025 :=
by {
  sorry
}

end starting_number_of_three_squares_less_than_2300_l162_162057


namespace value_of_z_l162_162551

theorem value_of_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := 
by
  -- Proof skipped
  sorry

end value_of_z_l162_162551


namespace any_nat_as_fraction_form_l162_162174

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end any_nat_as_fraction_form_l162_162174


namespace sqrt_of_9_l162_162085

theorem sqrt_of_9 (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
by {
  sorry
}

end sqrt_of_9_l162_162085


namespace minimize_surface_area_l162_162069

-- Define the problem conditions
def volume (x y : ℝ) : ℝ := 2 * x^2 * y
def surface_area (x y : ℝ) : ℝ := 2 * (2 * x^2 + 2 * x * y + x * y)

theorem minimize_surface_area :
  ∃ (y : ℝ), 
  (∀ (x : ℝ), volume x y = 72) → 
  1 * 2 * y = 4 :=
by
  sorry

end minimize_surface_area_l162_162069


namespace range_of_m_l162_162042

open Set

def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

theorem range_of_m (m : ℝ) : (M m ∩ N).Nonempty ↔ m > 0 := sorry

end range_of_m_l162_162042


namespace twelfth_term_of_arithmetic_sequence_l162_162398

/-- Condition: a_1 = 1/2 -/
def a1 : ℚ := 1 / 2

/-- Condition: common difference d = 1/3 -/
def d : ℚ := 1 / 3

/-- Prove that the 12th term in the arithmetic sequence is 25/6 given the conditions. -/
theorem twelfth_term_of_arithmetic_sequence : a1 + 11 * d = 25 / 6 := by
  sorry

end twelfth_term_of_arithmetic_sequence_l162_162398


namespace total_tires_mike_changed_l162_162730

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_tires_mike_changed_l162_162730


namespace first_term_arithmetic_series_l162_162865

theorem first_term_arithmetic_series 
  (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 240)
  (h2 : 30 * (2 * a + 179 * d) = 3600) : 
  a = -353 / 15 :=
by
  have eq1 : 2 * a + 59 * d = 8 := by sorry
  have eq2 : 2 * a + 179 * d = 120 := by sorry
  sorry

end first_term_arithmetic_series_l162_162865


namespace area_PTR_l162_162632

-- Define points P, Q, R, S, and T
variables (P Q R S T : Type)

-- Assume QR is divided by points S and T in the given ratio
variables (QS ST TR : ℕ)
axiom ratio_condition : QS = 2 ∧ ST = 5 ∧ TR = 3

-- Assume the area of triangle PQS is given as 60 square centimeters
axiom area_PQS : ℕ
axiom area_PQS_value : area_PQS = 60

-- State the problem
theorem area_PTR : ∃ (area_PTR : ℕ), area_PTR = 90 :=
by
  sorry

end area_PTR_l162_162632


namespace work_completion_days_l162_162408

theorem work_completion_days (Ry : ℝ) (R_combined : ℝ) (D : ℝ) :
  Ry = 1 / 40 ∧ R_combined = 1 / 13.333333333333332 → 1 / D + Ry = R_combined → D = 20 :=
by
  intros h_eqs h_combined
  sorry

end work_completion_days_l162_162408


namespace symmetric_point_yaxis_correct_l162_162151

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetric_yaxis (P : Point3D) : Point3D :=
  { x := -P.x, y := P.y, z := P.z }

theorem symmetric_point_yaxis_correct (P : Point3D) (P' : Point3D) :
  P = {x := 1, y := 2, z := -1} → 
  P' = symmetric_yaxis P → 
  P' = {x := -1, y := 2, z := -1} :=
by
  intros hP hP'
  rw [hP] at hP'
  simp [symmetric_yaxis] at hP'
  exact hP'

end symmetric_point_yaxis_correct_l162_162151


namespace power_of_two_contains_k_as_substring_l162_162527

theorem power_of_two_contains_k_as_substring (k : ℕ) (h1 : 1000 ≤ k) (h2 : k < 10000) : 
  ∃ n < 20000, ∀ m, 10^m * k ≤ 2^n ∧ 2^n < 10^(m+4) * (k+1) :=
sorry

end power_of_two_contains_k_as_substring_l162_162527


namespace police_female_officers_l162_162373

theorem police_female_officers (perc : ℝ) (total_on_duty: ℝ) (half_on_duty : ℝ) (F : ℝ) :
    perc = 0.18 →
    total_on_duty = 144 →
    half_on_duty = total_on_duty / 2 →
    half_on_duty = perc * F →
    F = 400 :=
by
  sorry

end police_female_officers_l162_162373


namespace sufficient_but_not_necessary_condition_l162_162928

theorem sufficient_but_not_necessary_condition (x y : ℝ) : 
  (x > 3 ∧ y > 3 → x + y > 6) ∧ ¬(x + y > 6 → x > 3 ∧ y > 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l162_162928


namespace value_of_business_l162_162945

-- Defining the conditions
def owns_shares : ℚ := 2/3
def sold_fraction : ℚ := 3/4 
def sold_amount : ℝ := 75000 

-- The final proof statement
theorem value_of_business : 
  (owns_shares * sold_fraction) * value = sold_amount →
  value = 150000 :=
by
  sorry

end value_of_business_l162_162945


namespace find_b8_l162_162871

noncomputable section

def increasing_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

axiom b_seq : ℕ → ℕ

axiom seq_inc : increasing_sequence b_seq

axiom b7_eq : b_seq 7 = 198

theorem find_b8 : b_seq 8 = 321 := by
  sorry

end find_b8_l162_162871


namespace highest_elevation_l162_162235

   noncomputable def elevation (t : ℝ) : ℝ := 240 * t - 24 * t^2

   theorem highest_elevation : ∃ t : ℝ, elevation t = 600 ∧ ∀ x : ℝ, elevation x ≤ 600 := 
   sorry
   
end highest_elevation_l162_162235


namespace find_m_sum_terms_l162_162920

theorem find_m (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) 
  (h2 : a 3 + a 6 + a 10 + a 13 = 32) (hm : a m = 8) : m = 8 :=
sorry

theorem sum_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (hS3 : S 3 = 9) (hS6 : S 6 = 36) 
  (a_def : ∀ n, S n = n * (a 1 + a n) / 2) : a 7 + a 8 + a 9 = 45 :=
sorry

end find_m_sum_terms_l162_162920


namespace max_soap_boxes_in_carton_l162_162886

theorem max_soap_boxes_in_carton
  (L_carton W_carton H_carton : ℕ)
  (L_soap_box W_soap_box H_soap_box : ℕ)
  (vol_carton := L_carton * W_carton * H_carton)
  (vol_soap_box := L_soap_box * W_soap_box * H_soap_box)
  (max_soap_boxes := vol_carton / vol_soap_box) :
  L_carton = 25 → W_carton = 42 → H_carton = 60 →
  L_soap_box = 7 → W_soap_box = 6 → H_soap_box = 5 →
  max_soap_boxes = 300 :=
by
  intros hL hW hH hLs hWs hHs
  sorry

end max_soap_boxes_in_carton_l162_162886


namespace find_rate_of_interest_l162_162966

def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem find_rate_of_interest :
  ∀ (R : ℕ),
  simple_interest 5000 R 2 + simple_interest 3000 R 4 = 2640 → R = 12 :=
by
  intros R h
  sorry

end find_rate_of_interest_l162_162966


namespace units_digit_product_l162_162970

theorem units_digit_product : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 := 
by
  sorry

end units_digit_product_l162_162970


namespace smaller_angle_at_3_pm_l162_162319

-- Define the condition for minute hand position at 3:00 p.m.
def minute_hand_position_at_3_pm_deg : ℝ := 0

-- Define the condition for hour hand position at 3:00 p.m.
def hour_hand_position_at_3_pm_deg : ℝ := 90

-- Define the angle between the minute hand and hour hand
def angle_between_hands (minute_deg hour_deg : ℝ) : ℝ :=
  abs (hour_deg - minute_deg)

-- The main theorem we need to prove
theorem smaller_angle_at_3_pm :
  angle_between_hands minute_hand_position_at_3_pm_deg hour_hand_position_at_3_pm_deg = 90 :=
by
  sorry

end smaller_angle_at_3_pm_l162_162319


namespace six_times_expression_l162_162004

theorem six_times_expression {x y Q : ℝ} (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q :=
by
  sorry

end six_times_expression_l162_162004


namespace exists_good_placement_l162_162498

-- Define a function that checks if a placement is "good" with respect to a symmetry axis
def is_good (f : Fin 1983 → ℕ) : Prop :=
  ∀ (i : Fin 1983), f i < f (i + 991) ∨ f (i + 991) < f i

-- Prove the existence of a "good" placement for the regular 1983-gon
theorem exists_good_placement : ∃ f : Fin 1983 → ℕ, is_good f :=
sorry

end exists_good_placement_l162_162498


namespace sum_of_fractions_l162_162869

theorem sum_of_fractions :
  (1 / (1^2 * 2^2) + 1 / (2^2 * 3^2) + 1 / (3^2 * 4^2) + 1 / (4^2 * 5^2)
  + 1 / (5^2 * 6^2) + 1 / (6^2 * 7^2)) = 48 / 49 := 
by
  sorry

end sum_of_fractions_l162_162869


namespace find_number_l162_162209

theorem find_number (x : ℝ) (h : x / 3 = x - 4) : x = 6 := 
by 
  sorry

end find_number_l162_162209


namespace smallest_x_for_div_by9_l162_162315

-- Define the digit sum of the number 761*829 with a placeholder * for x
def digit_sum_with_x (x : Nat) : Nat :=
  7 + 6 + 1 + x + 8 + 2 + 9

-- State the theorem to prove the smallest value of x makes the sum divisible by 9
theorem smallest_x_for_div_by9 : ∃ x : Nat, digit_sum_with_x x % 9 = 0 ∧ (∀ y : Nat, y < x → digit_sum_with_x y % 9 ≠ 0) :=
sorry

end smallest_x_for_div_by9_l162_162315


namespace avg_minutes_eq_170_div_9_l162_162270

-- Define the conditions
variables (s : ℕ) -- number of seventh graders
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2
def sixth_grade_minutes := 18
def seventh_grade_run_minutes := 20
def seventh_grade_stretching_minutes := 5
def eighth_grade_minutes := 12

-- Define the total activity minutes for each grade
def total_activity_minutes_sixth := sixth_grade_minutes * sixth_graders
def total_activity_minutes_seventh := (seventh_grade_run_minutes + seventh_grade_stretching_minutes) * seventh_graders
def total_activity_minutes_eighth := eighth_grade_minutes * eighth_graders

-- Calculate total activity minutes
def total_activity_minutes := total_activity_minutes_sixth + total_activity_minutes_seventh + total_activity_minutes_eighth

-- Calculate total number of students
def total_students := sixth_graders + seventh_graders + eighth_graders

-- Calculate average minutes per student
def average_minutes_per_student := total_activity_minutes / total_students

theorem avg_minutes_eq_170_div_9 : average_minutes_per_student s = 170 / 9 := by
  sorry

end avg_minutes_eq_170_div_9_l162_162270


namespace smallest_prime_factor_in_setB_l162_162021

def setB : Set ℕ := {55, 57, 58, 59, 61}

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 2 then 2 else (Nat.minFac (Nat.pred n)).succ

theorem smallest_prime_factor_in_setB :
  ∃ n ∈ setB, smallest_prime_factor n = 2 := by
  sorry

end smallest_prime_factor_in_setB_l162_162021


namespace sequence_bound_l162_162096

theorem sequence_bound (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
by
  sorry

end sequence_bound_l162_162096


namespace triangle_side_lengths_l162_162944

theorem triangle_side_lengths (r : ℝ) (AC BC AB : ℝ) (y : ℝ) 
  (h1 : r = 3 * Real.sqrt 2)
  (h2 : AC = 5 * Real.sqrt y) 
  (h3 : BC = 13 * Real.sqrt y) 
  (h4 : AB = 10 * Real.sqrt y) : 
  r = 3 * Real.sqrt 2 → 
  (∃ (AC BC AB : ℝ), 
     AC = 5 * Real.sqrt (7) ∧ 
     BC = 13 * Real.sqrt (7) ∧ 
     AB = 10 * Real.sqrt (7)) :=
by
  sorry

end triangle_side_lengths_l162_162944


namespace problem_l162_162647

-- Define the conditions
variables (x y : ℝ)
axiom h1 : 2 * x + y = 7
axiom h2 : x + 2 * y = 5

-- Statement of the problem
theorem problem : (2 * x * y) / 3 = 2 :=
by 
  -- Proof is omitted, but you should replace 'sorry' by the actual proof
  sorry

end problem_l162_162647


namespace tea_bags_count_l162_162108

-- Definitions based on the given problem
def valid_bags (b : ℕ) : Prop :=
  ∃ (a c d : ℕ), a + b - a = b ∧ c + d = b ∧ 3 * c + 2 * d = 41 ∧ 3 * a + 2 * (b - a) = 58

-- Statement of the problem, confirming the proof condition
theorem tea_bags_count (b : ℕ) : valid_bags b ↔ b = 20 :=
by {
  -- The proof is left for completion
  sorry
}

end tea_bags_count_l162_162108


namespace smallest_n_for_2007_l162_162902

/-- The smallest number of positive integers \( n \) such that their product is 2007 and their sum is 2007.
Given that \( n > 1 \), we need to show 1337 is the smallest such \( n \).
-/
theorem smallest_n_for_2007 (n : ℕ) (H : n > 1) :
  (∃ s : Finset ℕ, (s.sum id = 2007) ∧ (s.prod id = 2007) ∧ (s.card = n)) → (n = 1337) :=
sorry

end smallest_n_for_2007_l162_162902


namespace percent_of_srp_bob_paid_l162_162204

theorem percent_of_srp_bob_paid (SRP MP PriceBobPaid : ℝ) 
  (h1 : MP = 0.60 * SRP)
  (h2 : PriceBobPaid = 0.60 * MP) :
  (PriceBobPaid / SRP) * 100 = 36 := by
  sorry

end percent_of_srp_bob_paid_l162_162204


namespace range_of_a_l162_162144

theorem range_of_a (a : ℝ) (in_fourth_quadrant : (a+2 > 0) ∧ (a-3 < 0)) : -2 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l162_162144


namespace expression_equality_l162_162012

-- Define the conditions
variables {a b x : ℝ}
variable (h1 : x = a / b)
variable (h2 : a ≠ 2 * b)
variable (h3 : b ≠ 0)

-- Define and state the theorem
theorem expression_equality : (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) :=
by 
  intros
  sorry

end expression_equality_l162_162012


namespace average_runs_in_30_matches_l162_162625

theorem average_runs_in_30_matches (avg_runs_15: ℕ) (avg_runs_20: ℕ) 
    (matches_15: ℕ) (matches_20: ℕ)
    (h1: avg_runs_15 = 30) (h2: avg_runs_20 = 15)
    (h3: matches_15 = 15) (h4: matches_20 = 20) : 
    (matches_15 * avg_runs_15 + matches_20 * avg_runs_20) / (matches_15 + matches_20) = 25 := 
by 
  sorry

end average_runs_in_30_matches_l162_162625


namespace hexagon_cyclic_identity_l162_162978

variables (a a' b b' c c' a₁ b₁ c₁ : ℝ)

theorem hexagon_cyclic_identity :
  a₁ * b₁ * c₁ = a * b * c + a' * b' * c' + a * a' * a₁ + b * b' * b₁ + c * c' * c₁ :=
by
  sorry

end hexagon_cyclic_identity_l162_162978


namespace frac_equality_l162_162531

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end frac_equality_l162_162531


namespace measure_of_angle_C_l162_162640

theorem measure_of_angle_C (a b area : ℝ) (C : ℝ) :
  a = 5 → b = 8 → area = 10 →
  (1 / 2 * a * b * Real.sin C = area) →
  (C = Real.pi / 6 ∨ C = 5 * Real.pi / 6) := by
  intros ha hb harea hformula
  sorry

end measure_of_angle_C_l162_162640


namespace lowest_possible_sale_price_percentage_l162_162470

noncomputable def list_price : ℝ := 80
noncomputable def max_initial_discount_percent : ℝ := 0.5
noncomputable def summer_sale_discount_percent : ℝ := 0.2
noncomputable def membership_discount_percent : ℝ := 0.1
noncomputable def coupon_discount_percent : ℝ := 0.05

theorem lowest_possible_sale_price_percentage :
  let max_initial_discount := max_initial_discount_percent * list_price
  let summer_sale_discount := summer_sale_discount_percent * list_price
  let membership_discount := membership_discount_percent * list_price
  let coupon_discount := coupon_discount_percent * list_price
  let lowest_sale_price := list_price * (1 - max_initial_discount_percent) - summer_sale_discount - membership_discount - coupon_discount
  (lowest_sale_price / list_price) * 100 = 15 :=
by
  sorry

end lowest_possible_sale_price_percentage_l162_162470


namespace triangle_abs_simplification_l162_162025

theorem triangle_abs_simplification
  (x y z : ℝ)
  (h1 : x + y > z)
  (h2 : y + z > x)
  (h3 : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3 * y - 3 * z :=
by
  sorry

end triangle_abs_simplification_l162_162025


namespace simplify_expression_eval_at_2_l162_162708

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (x^2 + a)^2 / ((a - b) * (a - c)) + (x^2 + b)^2 / ((b - a) * (b - c)) + (x^2 + c)^2 / ((c - a) * (c - b)) =
    x^4 + x^2 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

theorem eval_at_2 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (2^2 + a)^2 / ((a - b) * (a - c)) + (2^2 + b)^2 / ((b - a) * (b - c)) + (2^2 + c)^2 / ((c - a) * (c - b)) =
    16 + 4 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

end simplify_expression_eval_at_2_l162_162708


namespace total_ducks_l162_162170

-- Definitions based on the given conditions
def Muscovy : ℕ := 39
def Cayuga : ℕ := Muscovy - 4
def KhakiCampbell : ℕ := (Cayuga - 3) / 2

-- Proof statement
theorem total_ducks : Muscovy + Cayuga + KhakiCampbell = 90 := by
  sorry

end total_ducks_l162_162170


namespace largest_number_obtained_l162_162339

theorem largest_number_obtained : 
  ∃ n : ℤ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m, 10 ≤ m ∧ m ≤ 99 → (250 - 3 * m)^2 ≤ (250 - 3 * n)^2) ∧ (250 - 3 * n)^2 = 4 :=
sorry

end largest_number_obtained_l162_162339


namespace days_to_fill_tank_l162_162462

-- Definitions based on the problem conditions
def tank_capacity_liters : ℕ := 50
def liters_to_milliliters : ℕ := 1000
def rain_collection_per_day : ℕ := 800
def river_collection_per_day : ℕ := 1700
def total_collection_per_day : ℕ := rain_collection_per_day + river_collection_per_day
def tank_capacity_milliliters : ℕ := tank_capacity_liters * liters_to_milliliters

-- Statement of the proof that Jacob needs 20 days to fill the tank
theorem days_to_fill_tank : tank_capacity_milliliters / total_collection_per_day = 20 := by
  sorry

end days_to_fill_tank_l162_162462


namespace lines_are_parallel_and_not_coincident_l162_162961

theorem lines_are_parallel_and_not_coincident (a : ℝ) :
  (a * (a - 1) - 3 * 2 = 0) ∧ (3 * (a - 7) - a * 3 * a ≠ 0) ↔ a = 3 :=
by
  sorry

end lines_are_parallel_and_not_coincident_l162_162961


namespace sum_of_eight_numbers_on_cards_l162_162556

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l162_162556


namespace smallest_positive_k_l162_162455

theorem smallest_positive_k (k a n : ℕ) (h_pos : k > 0) (h_cond : 3^3 + 4^3 + 5^3 = 216) (h_eq : k * 216 = a^n) (h_n : n > 1) : k = 1 :=
by {
    sorry
}

end smallest_positive_k_l162_162455


namespace evaluate_expression_at_x_neg3_l162_162229

theorem evaluate_expression_at_x_neg3 :
  (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 :=
by
  sorry

end evaluate_expression_at_x_neg3_l162_162229


namespace smallest_lcm_l162_162879

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l162_162879


namespace smallest_right_triangle_area_l162_162659

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l162_162659


namespace machine_value_after_2_years_l162_162799

section
def initial_value : ℝ := 1200
def depreciation_rate_year1 : ℝ := 0.10
def depreciation_rate_year2 : ℝ := 0.12
def repair_rate : ℝ := 0.03
def major_overhaul_rate : ℝ := 0.15

theorem machine_value_after_2_years :
  let value_after_repairs_2 := (initial_value * (1 - depreciation_rate_year1) + initial_value * repair_rate) * (1 - depreciation_rate_year2 + repair_rate)
  (value_after_repairs_2 * (1 - major_overhaul_rate)) = 863.23 := 
by
  -- proof here
  sorry
end

end machine_value_after_2_years_l162_162799


namespace sum_shade_length_l162_162109

-- Define the arithmetic sequence and the given conditions
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (is_arithmetic : ∀ n, a (n + 1) = a n + d)

-- Define the shadow lengths for each term using the arithmetic progression properties
def shade_length_seq (seq : ArithmeticSequence) : ℕ → ℝ := seq.a

variables (seq : ArithmeticSequence)

-- Given conditions
axiom sum_condition_1 : seq.a 1 + seq.a 4 + seq.a 7 = 31.5
axiom sum_condition_2 : seq.a 2 + seq.a 5 + seq.a 8 = 28.5

-- Question to prove
theorem sum_shade_length : seq.a 3 + seq.a 6 + seq.a 9 = 25.5 :=
by
  -- proof to be filled in later
  sorry

end sum_shade_length_l162_162109


namespace xy_square_difference_l162_162495

variable (x y : ℚ)

theorem xy_square_difference (h1 : x + y = 8/15) (h2 : x - y = 1/45) : 
  x^2 - y^2 = 8/675 := by
  sorry

end xy_square_difference_l162_162495


namespace solve_inequalities_l162_162208

theorem solve_inequalities (x : ℝ) (h₁ : (x - 1) / 2 < 2 * x + 1) (h₂ : -3 * (1 - x) ≥ -4) : x ≥ -1 / 3 :=
by
  sorry

end solve_inequalities_l162_162208


namespace find_n_with_divisors_conditions_l162_162762

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end find_n_with_divisors_conditions_l162_162762


namespace roots_product_eq_l162_162136

theorem roots_product_eq
  (a b m p r : ℚ)
  (h₀ : a * b = 3)
  (h₁ : ∀ x, x^2 - m * x + 3 = 0 → (x = a ∨ x = b))
  (h₂ : ∀ x, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a)) : 
  r = 16 / 3 :=
by
  sorry

end roots_product_eq_l162_162136


namespace leah_coins_value_l162_162324

theorem leah_coins_value :
  ∃ (p n d : ℕ), 
    p + n + d = 20 ∧
    p = n ∧
    p = d + 4 ∧
    1 * p + 5 * n + 10 * d = 88 :=
by
  sorry

end leah_coins_value_l162_162324


namespace shirts_total_cost_l162_162181

def shirt_cost_problem : Prop :=
  ∃ (first_shirt_cost second_shirt_cost total_cost : ℕ),
    first_shirt_cost = 15 ∧
    first_shirt_cost = second_shirt_cost + 6 ∧
    total_cost = first_shirt_cost + second_shirt_cost ∧
    total_cost = 24

theorem shirts_total_cost : shirt_cost_problem := by
  sorry

end shirts_total_cost_l162_162181


namespace factorization_correct_l162_162995

theorem factorization_correct :
  (∀ x : ℝ, x^2 - 6*x + 9 = (x - 3)^2) :=
by
  sorry

end factorization_correct_l162_162995


namespace Bryan_deposited_312_l162_162290

-- Definitions based on conditions
def MarkDeposit : ℕ := 88
def TotalDeposit : ℕ := 400
def MaxBryanDeposit (MarkDeposit : ℕ) : ℕ := 5 * MarkDeposit 

def BryanDeposit (B : ℕ) : Prop := B < MaxBryanDeposit MarkDeposit ∧ MarkDeposit + B = TotalDeposit

theorem Bryan_deposited_312 : BryanDeposit 312 :=
by
   -- Proof steps go here
   sorry

end Bryan_deposited_312_l162_162290


namespace determine_xyz_l162_162228

-- Define the conditions for the variables x, y, and z
variables (x y z : ℝ)

-- State the problem as a theorem
theorem determine_xyz :
  (x + y + z) * (x * y + x * z + y * z) = 24 ∧
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8 →
  x * y * z = 16 / 3 :=
by
  intros h
  sorry

end determine_xyz_l162_162228


namespace fill_tank_time_l162_162595

theorem fill_tank_time (R L E : ℝ) (fill_time : ℝ) (leak_time : ℝ) (effective_rate : ℝ) : 
  (R = 1 / fill_time) → 
  (L = 1 / leak_time) →
  (E = R - L) →
  (fill_time = 10) →
  (leak_time = 110) →
  (E = 1 / effective_rate) →
  effective_rate = 11 :=
by
  sorry

end fill_tank_time_l162_162595


namespace g_at_5_l162_162731

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation :
  ∀ (x : ℝ), g x + 3 * g (2 - x) = 4 * x ^ 2 - 5 * x + 1

theorem g_at_5 : g 5 = -5 / 4 :=
by
  let h := functional_equation
  sorry

end g_at_5_l162_162731


namespace shaniqua_haircuts_l162_162779

theorem shaniqua_haircuts
  (H : ℕ) -- number of haircuts
  (haircut_income : ℕ) (style_income : ℕ)
  (total_styles : ℕ) (total_income : ℕ)
  (haircut_income_eq : haircut_income = 12)
  (style_income_eq : style_income = 25)
  (total_styles_eq : total_styles = 5)
  (total_income_eq : total_income = 221)
  (income_from_styles : ℕ := total_styles * style_income)
  (income_from_haircuts : ℕ := total_income - income_from_styles) :
  H = income_from_haircuts / haircut_income :=
sorry

end shaniqua_haircuts_l162_162779


namespace triangles_in_divided_square_l162_162117

theorem triangles_in_divided_square (V E F : ℕ) 
  (hV : V = 24) 
  (h1 : 3 * F + 1 = 2 * E) 
  (h2 : V - E + F = 2) : F = 43 ∧ (F - 1 = 42) := 
by 
  have hF : F = 43 := sorry
  have hTriangles : F - 1 = 42 := sorry
  exact ⟨hF, hTriangles⟩

end triangles_in_divided_square_l162_162117


namespace same_solution_for_equations_l162_162368

theorem same_solution_for_equations (b x : ℝ) :
  (2 * x + 7 = 3) → 
  (b * x - 10 = -2) → 
  b = -4 :=
by
  sorry

end same_solution_for_equations_l162_162368


namespace tina_total_time_l162_162615

-- Define constants for the problem conditions
def assignment_time : Nat := 20
def dinner_time : Nat := 17 * 60 + 30 -- 5:30 PM in minutes
def clean_time_per_key : Nat := 7
def total_keys : Nat := 30
def remaining_keys : Nat := total_keys - 1
def dry_time_per_key : Nat := 10
def break_time : Nat := 3
def keys_per_break : Nat := 5

-- Define a function to compute total cleaning time for remaining keys
def total_cleaning_time (keys : Nat) (clean_time : Nat) : Nat :=
  keys * clean_time

-- Define a function to compute total drying time for all keys
def total_drying_time (keys : Nat) (dry_time : Nat) : Nat :=
  keys * dry_time

-- Define a function to compute total break time
def total_break_time (keys : Nat) (keys_per_break : Nat) (break_time : Nat) : Nat :=
  (keys / keys_per_break) * break_time

-- Define a function to compute the total time including cleaning, drying, breaks, and assignment
def total_time (cleaning_time drying_time break_time assignment_time : Nat) : Nat :=
  cleaning_time + drying_time + break_time + assignment_time

-- The theorem to be proven
theorem tina_total_time : 
  total_time (total_cleaning_time remaining_keys clean_time_per_key) 
              (total_drying_time total_keys dry_time_per_key)
              (total_break_time total_keys keys_per_break break_time)
              assignment_time = 541 :=
by sorry

end tina_total_time_l162_162615


namespace find_k_l162_162620

theorem find_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2023) : k = 2022 := 
by {
  sorry
}

end find_k_l162_162620


namespace simplify_product_l162_162395

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l162_162395


namespace min_value_expression_l162_162623

-- Let x and y be positive integers such that x^2 + y^2 - 2017 * x * y > 0 and it is not a perfect square.
theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_not_square : ¬ ∃ z : ℕ, (x^2 + y^2 - 2017 * x * y) = z^2) :
  x^2 + y^2 - 2017 * x * y > 0 → ∃ k : ℕ, k = 2019 ∧ ∀ m : ℕ, (m > 0 → ¬ ∃ z : ℤ, (x^2 + y^2 - 2017 * x * y) = z^2 ∧ x^2 + y^2 - 2017 * x * y < k) :=
sorry

end min_value_expression_l162_162623


namespace farmer_initial_productivity_l162_162664

theorem farmer_initial_productivity (x : ℝ) (d : ℝ)
  (hx1 : d = 1440 / x)
  (hx2 : 2 * x + (d - 4) * 1.25 * x = 1440) :
  x = 120 :=
by
  sorry

end farmer_initial_productivity_l162_162664


namespace solution_alcohol_content_l162_162642

noncomputable def volume_of_solution_y_and_z (V: ℝ) : Prop :=
  let vol_X := 300.0
  let conc_X := 0.10
  let conc_Y := 0.30
  let conc_Z := 0.40
  let vol_Y := 2 * V
  let vol_new := vol_X + vol_Y + V
  let alcohol_new := conc_X * vol_X + conc_Y * vol_Y + conc_Z * V
  (alcohol_new / vol_new) = 0.22

theorem solution_alcohol_content : volume_of_solution_y_and_z 300.0 :=
by
  sorry

end solution_alcohol_content_l162_162642


namespace sum_ab_eq_negative_two_l162_162809

def f (x : ℝ) := x^3 + 3 * x^2 + 6 * x + 4

theorem sum_ab_eq_negative_two (a b : ℝ) (h1 : f a = 14) (h2 : f b = -14) : a + b = -2 := 
by 
  sorry

end sum_ab_eq_negative_two_l162_162809


namespace turtles_on_Happy_Island_l162_162677

theorem turtles_on_Happy_Island (L H : ℕ) (hL : L = 25) (hH : H = 2 * L + 10) : H = 60 :=
by
  sorry

end turtles_on_Happy_Island_l162_162677


namespace q1_q2_q3_l162_162482

noncomputable def quadratic_function (a x: ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem q1 (a : ℝ) : (∀ {x : ℝ}, quadratic_function a x = 0 → x < 2) ∧ (quadratic_function a 2 > 0) ∧ (2 * a ≠ 0) → a < -1 := 
by 
  sorry

theorem q2 (a : ℝ) : (∀ x : ℝ, quadratic_function a x ≥ -1 - a * x) → -2 ≤ a ∧ a ≤ 6 := 
by 
  sorry
  
theorem q3 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → quadratic_function a x ≤ 4) → a = 2 ∨ a = 2 / 3 := 
by 
  sorry

end q1_q2_q3_l162_162482


namespace minimum_value_of_f_l162_162070

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 5/2) ∧ (f 1 = 5/2) := by
  sorry

end minimum_value_of_f_l162_162070


namespace fraction_sum_l162_162710

theorem fraction_sum : (3 / 8) + (9 / 14) = (57 / 56) := by
  sorry

end fraction_sum_l162_162710


namespace find_number_to_be_multiplied_l162_162874

-- Define the conditions of the problem
variable (x : ℕ)

-- Condition 1: The correct multiplication would have been 43x
-- Condition 2: The actual multiplication done was 34x
-- Condition 3: The difference between correct and actual result is 1242

theorem find_number_to_be_multiplied (h : 43 * x - 34 * x = 1242) : 
  x = 138 := by
  sorry

end find_number_to_be_multiplied_l162_162874


namespace sum_of_smallest_and_largest_is_correct_l162_162517

-- Define the conditions
def digits : Set ℕ := {0, 3, 4, 8}

-- Define the smallest and largest valid four-digit number using the digits
def smallest_number : ℕ := 3048
def largest_number : ℕ := 8430

-- Define the sum of the smallest and largest numbers
def sum_of_numbers : ℕ := smallest_number + largest_number

-- The theorem to be proven
theorem sum_of_smallest_and_largest_is_correct : 
  sum_of_numbers = 11478 := 
by
  -- Proof omitted
  sorry

end sum_of_smallest_and_largest_is_correct_l162_162517


namespace raise_3000_yuan_probability_l162_162378

def prob_correct_1 : ℝ := 0.9
def prob_correct_2 : ℝ := 0.5
def prob_correct_3 : ℝ := 0.4
def prob_incorrect_3 : ℝ := 1 - prob_correct_3

def fund_first : ℝ := 1000
def fund_second : ℝ := 2000
def fund_third : ℝ := 3000

def prob_raise_3000_yuan : ℝ := prob_correct_1 * prob_correct_2 * prob_incorrect_3

theorem raise_3000_yuan_probability :
  prob_raise_3000_yuan = 0.27 :=
by
  sorry

end raise_3000_yuan_probability_l162_162378


namespace relay_race_total_time_correct_l162_162831

-- Conditions as definitions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25
def athlete5_time : ℕ := 80
def athlete6_time : ℕ := athlete5_time - 20
def athlete7_time : ℕ := 70
def athlete8_time : ℕ := athlete7_time - 5

-- Sum of all athletes' times
def total_time : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time + athlete5_time +
  athlete6_time + athlete7_time + athlete8_time

-- Statement to prove
theorem relay_race_total_time_correct : total_time = 475 :=
  by
  sorry

end relay_race_total_time_correct_l162_162831


namespace sum_inequality_l162_162570

variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {m n p k : ℕ}

-- Definitions for the conditions given in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, a (i + 1) - a i = a (j + 1) - a j

def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a (n - 1)) / 2

def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

-- The theorem to prove
theorem sum_inequality (arith_seq : is_arithmetic_sequence a)
  (S_eq : sum_of_arithmetic_sequence S a)
  (nn_seq : non_negative_sequence a)
  (h1 : m + n = 2 * p) (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) :
  1 / (S m) ^ k + 1 / (S n) ^ k ≥ 2 / (S p) ^ k :=
by sorry

end sum_inequality_l162_162570


namespace leonine_cats_l162_162244

theorem leonine_cats (n : ℕ) (h : n = (4 / 5 * n) + (4 / 5)) : n = 4 :=
by
  sorry

end leonine_cats_l162_162244


namespace minimum_value_C2_minus_D2_l162_162119

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 11))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 4)) + (Real.sqrt (z + 9))

theorem minimum_value_C2_minus_D2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (C x y z)^2 - (D x y z)^2 ≥ 36 := by
  sorry

end minimum_value_C2_minus_D2_l162_162119


namespace unique_function_satisfies_condition_l162_162063

theorem unique_function_satisfies_condition :
  ∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x * Real.sin y) + f (x * Real.sin z) -
    f x * f (Real.sin y * Real.sin z) + Real.sin (Real.pi * x) ≥ 1 := sorry

end unique_function_satisfies_condition_l162_162063


namespace fraction_value_l162_162958

-- Define the constants
def eight := 8
def four := 4

-- Statement to prove
theorem fraction_value : (eight + four) / (eight - four) = 3 := 
by
  sorry

end fraction_value_l162_162958


namespace evaluate_expression_l162_162059

theorem evaluate_expression : 
  ( (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 7) ) = 7 ^ (3 / 28) := 
by {
  sorry
}

end evaluate_expression_l162_162059


namespace probability_multiple_of_4_l162_162101

theorem probability_multiple_of_4 :
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  prob_end_multiple_of_4 = 7 / 64 :=
by
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  have h : prob_end_multiple_of_4 = 7 / 64 := by sorry
  exact h

end probability_multiple_of_4_l162_162101


namespace isosceles_triangle_perimeter_l162_162323

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4) (h2 : b = 6) : 
  ∃ p, (p = 14 ∨ p = 16) :=
by
  sorry

end isosceles_triangle_perimeter_l162_162323


namespace intersection_A_B_l162_162440

theorem intersection_A_B :
  let A := {1, 3, 5, 7}
  let B := {x | x^2 - 2 * x - 5 ≤ 0}
  A ∩ B = {1, 3} := by
sorry

end intersection_A_B_l162_162440


namespace number_of_men_l162_162186

theorem number_of_men (M W : ℕ) (h1 : W = 2) (h2 : ∃k, k = 4) : M = 4 :=
by
  sorry

end number_of_men_l162_162186


namespace part_a_l162_162925

theorem part_a (m n : ℕ) (hm : m > 1) : n ∣ Nat.totient (m^n - 1) :=
sorry

end part_a_l162_162925


namespace new_train_distance_l162_162605

-- Define the given conditions
def distance_old : ℝ := 300
def percentage_increase : ℝ := 0.3

-- Define the target distance to prove
def distance_new : ℝ := distance_old + (percentage_increase * distance_old)

-- State the theorem
theorem new_train_distance : distance_new = 390 := by
  sorry

end new_train_distance_l162_162605


namespace chess_team_boys_count_l162_162276

theorem chess_team_boys_count : 
  ∃ (B G : ℕ), B + G = 30 ∧ (2 / 3 : ℚ) * G + B = 18 ∧ B = 6 := by
  sorry

end chess_team_boys_count_l162_162276


namespace james_pitbull_count_l162_162724

-- Defining the conditions
def husky_count : ℕ := 5
def retriever_count : ℕ := 4
def retriever_pups_per_retriever (husky_pups_per_husky : ℕ) : ℕ := husky_pups_per_husky + 2
def husky_pups := husky_count * 3
def retriever_pups := retriever_count * (retriever_pups_per_retriever 3)
def pitbull_pups (P : ℕ) : ℕ := P * 3
def total_pups (P : ℕ) : ℕ := husky_pups + retriever_pups + pitbull_pups P
def total_adults (P : ℕ) : ℕ := husky_count + retriever_count + P
def condition (P : ℕ) : Prop := total_pups P = total_adults P + 30

-- The proof objective
theorem james_pitbull_count : ∃ P : ℕ, condition P → P = 2 := by
  sorry

end james_pitbull_count_l162_162724


namespace nth_equation_pattern_l162_162649

theorem nth_equation_pattern (n : ℕ) (hn : 0 < n) : n^2 - n = n * (n - 1) := by
  sorry

end nth_equation_pattern_l162_162649


namespace tan_theta_determined_l162_162841

theorem tan_theta_determined (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4) (h_zero : Real.tan θ + Real.tan (4 * θ) = 0) :
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) :=
sorry

end tan_theta_determined_l162_162841


namespace complete_residue_system_mod_l162_162183

open Nat

theorem complete_residue_system_mod (m : ℕ) (x : Fin m → ℕ)
  (h : ∀ i j : Fin m, i ≠ j → ¬ ((x i) % m = (x j) % m)) :
  (Finset.image (λ i => x i % m) (Finset.univ : Finset (Fin m))) = Finset.range m :=
by
  -- Skipping the proof steps.
  sorry

end complete_residue_system_mod_l162_162183


namespace prime_implies_n_eq_3k_l162_162424

theorem prime_implies_n_eq_3k (n : ℕ) (p : ℕ) (k : ℕ) (h_pos : k > 0)
  (h_prime : Prime p) (h_eq : p = 1 + 2^n + 4^n) :
  ∃ k : ℕ, k > 0 ∧ n = 3^k :=
by
  sorry

end prime_implies_n_eq_3k_l162_162424


namespace probability_of_Z_l162_162401

/-
  Given: 
  - P(W) = 3 / 8
  - P(X) = 1 / 4
  - P(Y) = 1 / 8

  Prove: 
  - P(Z) = 1 / 4 when P(Z) = 1 - (P(W) + P(X) + P(Y))
-/

theorem probability_of_Z (P_W P_X P_Y P_Z : ℚ) (h_W : P_W = 3 / 8) (h_X : P_X = 1 / 4) (h_Y : P_Y = 1 / 8) (h_Z : P_Z = 1 - (P_W + P_X + P_Y)) : 
  P_Z = 1 / 4 :=
by
  -- We can write the whole Lean Math proof here. However, per the instructions, we'll conclude with sorry.
  sorry

end probability_of_Z_l162_162401


namespace parabola_coefficients_sum_l162_162259

theorem parabola_coefficients_sum :
  ∃ a b c : ℝ, 
  (∀ y : ℝ, (7 = -(6 ^ 2) * a + b * 6 + c)) ∧
  (5 = a * (-4) ^ 2 + b * (-4) + c) ∧
  (a + b + c = -42) := 
sorry

end parabola_coefficients_sum_l162_162259


namespace depth_of_water_in_smaller_container_l162_162268

theorem depth_of_water_in_smaller_container 
  (H_big : ℝ) (R_big : ℝ) (h_water : ℝ) 
  (H_small : ℝ) (R_small : ℝ) (expected_depth : ℝ) 
  (v_water_small : ℝ) 
  (v_water_big : ℝ) 
  (h_total_water : ℝ)
  (above_brim : ℝ) 
  (v_water_final : ℝ) : 

  H_big = 20 ∧ R_big = 6 ∧ h_water = 17 ∧ H_small = 18 ∧ R_small = 5 ∧ expected_depth = 2.88 ∧
  v_water_big = π * R_big^2 * H_big ∧ v_water_small = π * R_small^2 * H_small ∧ 
  h_total_water = π * R_big^2 * h_water ∧ above_brim = π * R_big^2 * (H_big - H_small) ∧ 
  v_water_final = above_brim →

  expected_depth = v_water_final / (π * R_small^2) :=
by
  intro h
  sorry

end depth_of_water_in_smaller_container_l162_162268


namespace roots_equation_value_l162_162821

theorem roots_equation_value (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α + β = 1) :
    α^4 + 3 * β = 5 := by
sorry

end roots_equation_value_l162_162821


namespace total_miles_walked_l162_162370

-- Definition of the conditions
def num_islands : ℕ := 4
def miles_per_day_island1 : ℕ := 20
def miles_per_day_island2 : ℕ := 25
def days_per_island : ℚ := 1.5

-- Mathematically Equivalent Proof Problem
theorem total_miles_walked :
  let total_miles_island1 := 2 * (miles_per_day_island1 * days_per_island)
  let total_miles_island2 := 2 * (miles_per_day_island2 * days_per_island)
  total_miles_island1 + total_miles_island2 = 135 := by
  sorry

end total_miles_walked_l162_162370


namespace student_ticket_cost_l162_162899

def general_admission_ticket_cost : ℕ := 6
def total_tickets_sold : ℕ := 525
def total_revenue : ℕ := 2876
def general_admission_tickets_sold : ℕ := 388

def number_of_student_tickets_sold : ℕ := total_tickets_sold - general_admission_tickets_sold
def revenue_from_general_admission : ℕ := general_admission_tickets_sold * general_admission_ticket_cost

theorem student_ticket_cost : ∃ S : ℕ, number_of_student_tickets_sold * S + revenue_from_general_admission = total_revenue ∧ S = 4 :=
by
  sorry

end student_ticket_cost_l162_162899


namespace kat_boxing_training_hours_l162_162563

theorem kat_boxing_training_hours :
  let strength_training_hours := 3
  let total_training_hours := 9
  let boxing_sessions := 4
  let boxing_training_hours := total_training_hours - strength_training_hours
  let hours_per_boxing_session := boxing_training_hours / boxing_sessions
  hours_per_boxing_session = 1.5 :=
sorry

end kat_boxing_training_hours_l162_162563


namespace initial_bananas_each_child_l162_162981

-- Define the variables and conditions.
def total_children : ℕ := 320
def absent_children : ℕ := 160
def present_children := total_children - absent_children
def extra_bananas : ℕ := 2

-- We are to prove the initial number of bananas each child was supposed to get.
theorem initial_bananas_each_child (B : ℕ) (x : ℕ) :
  B = total_children * x ∧ B = present_children * (x + extra_bananas) → x = 2 :=
by
  sorry

end initial_bananas_each_child_l162_162981


namespace bars_per_set_correct_l162_162855

-- Define the total number of metal bars and the number of sets
def total_metal_bars : ℕ := 14
def number_of_sets : ℕ := 2

-- Define the function to compute bars per set
def bars_per_set (total_bars : ℕ) (sets : ℕ) : ℕ :=
  total_bars / sets

-- The proof statement
theorem bars_per_set_correct : bars_per_set total_metal_bars number_of_sets = 7 := by
  sorry

end bars_per_set_correct_l162_162855


namespace avg_first_six_results_l162_162737

theorem avg_first_six_results (A : ℝ) :
  (∀ (results : Fin 12 → ℝ), 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5 + 
     results 6 + results 7 + results 8 + results 9 + results 10 + results 11) / 11 = 60 → 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5) / 6 = A → 
    (results 5 + results 6 + results 7 + results 8 + results 9 + results 10) / 6 = 63 → 
    results 5 = 66) → 
  A = 58 :=
by
  sorry

end avg_first_six_results_l162_162737


namespace parallel_lines_a_eq_3_l162_162639

theorem parallel_lines_a_eq_3
  (a : ℝ)
  (l1 : a^2 * x - y + a^2 - 3 * a = 0)
  (l2 : (4 * a - 3) * x - y - 2 = 0)
  (h : ∀ x y, a^2 * x - y + a^2 - 3 * a = (4 * a - 3) * x - y - 2) :
  a = 3 :=
by
  sorry

end parallel_lines_a_eq_3_l162_162639


namespace ratio_of_distances_l162_162867

theorem ratio_of_distances
  (w x y : ℝ)
  (hw : w > 0)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq_time : y / w = x / w + (x + y) / (5 * w)) :
  x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l162_162867


namespace find_exact_speed_l162_162859

variable (d t v : ℝ)

-- Conditions as Lean definitions
def distance_eq1 : d = 50 * (t - 1/12) := sorry
def distance_eq2 : d = 70 * (t + 1/12) := sorry
def travel_time : t = 1/2 := sorry -- deduced travel time from the equations and given conditions
def correct_speed : v = 42 := sorry -- Mr. Bird needs to drive at 42 mph to be exactly on time

-- Lean 4 statement proving the required speed is 42 mph
theorem find_exact_speed : v = d / t :=
  by
    sorry

end find_exact_speed_l162_162859


namespace sara_jim_savings_eq_l162_162603

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end sara_jim_savings_eq_l162_162603


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l162_162015

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l162_162015


namespace population_net_increase_l162_162913

-- Definitions of conditions
def birth_rate := 7 / 2 -- 7 people every 2 seconds
def death_rate := 1 / 2 -- 1 person every 2 seconds
def seconds_in_a_day := 86400 -- Number of seconds in one day

-- Definition of the total births in one day
def total_births_per_day := birth_rate * seconds_in_a_day

-- Definition of the total deaths in one day
def total_deaths_per_day := death_rate * seconds_in_a_day

-- Proposition to prove the net population increase in one day
theorem population_net_increase : total_births_per_day - total_deaths_per_day = 259200 := by
  sorry

end population_net_increase_l162_162913


namespace solve_for_x_l162_162587

theorem solve_for_x (x : ℝ) (h : (3 + 2 / x)^(1 / 3) = 2) : x = 2 / 5 :=
by
  sorry

end solve_for_x_l162_162587


namespace find_units_digit_l162_162811

theorem find_units_digit : 
  (7^1993 + 5^1993) % 10 = 2 :=
by
  sorry

end find_units_digit_l162_162811


namespace smallest_value_of_a_l162_162449

theorem smallest_value_of_a (a b c d : ℤ) (h1 : (a - 2 * b) > 0) (h2 : (b - 3 * c) > 0) (h3 : (c - 4 * d) > 0) (h4 : d > 100) : a ≥ 2433 := sorry

end smallest_value_of_a_l162_162449


namespace polygon_sides_l162_162590

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l162_162590


namespace doughnuts_in_each_box_l162_162465

theorem doughnuts_in_each_box (total_doughnuts : ℕ) (boxes : ℕ) (h1 : total_doughnuts = 48) (h2 : boxes = 4) : total_doughnuts / boxes = 12 :=
by
  sorry

end doughnuts_in_each_box_l162_162465


namespace evlyn_can_buy_grapes_l162_162434

theorem evlyn_can_buy_grapes 
  (price_pears price_oranges price_lemons price_grapes : ℕ)
  (h1 : 10 * price_pears = 5 * price_oranges)
  (h2 : 4 * price_oranges = 6 * price_lemons)
  (h3 : 3 * price_lemons = 2 * price_grapes) :
  (20 * price_pears = 10 * price_grapes) :=
by
  -- The proof is omitted using sorry
  sorry

end evlyn_can_buy_grapes_l162_162434


namespace ball_returns_to_bella_after_13_throws_l162_162003

theorem ball_returns_to_bella_after_13_throws:
  ∀ (girls : Fin 13) (n : ℕ), (∃ k, k > 0 ∧ (1 + k * 5) % 13 = 1) → (n = 13) :=
by
  sorry

end ball_returns_to_bella_after_13_throws_l162_162003


namespace evaluate_expression_at_x_eq_3_l162_162705

theorem evaluate_expression_at_x_eq_3 : (3 ^ 3) ^ (3 ^ 3) = 27 ^ 27 := by
  sorry

end evaluate_expression_at_x_eq_3_l162_162705


namespace count_satisfying_integers_l162_162526

theorem count_satisfying_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, 9 < n ∧ n < 60) ∧ S.card = 50) :=
by
  sorry

end count_satisfying_integers_l162_162526


namespace triangle_right_angle_l162_162128

variable {A B C a b c : ℝ}

theorem triangle_right_angle (h1 : Real.sin (A / 2) ^ 2 = (c - b) / (2 * c)) 
                             (h2 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : 
                             a^2 + b^2 = c^2 :=
by
  sorry

end triangle_right_angle_l162_162128


namespace cos_pi_minus_alpha_l162_162733

open Real

variable (α : ℝ)

theorem cos_pi_minus_alpha (h1 : 0 < α ∧ α < π / 2) (h2 : sin α = 4 / 5) : cos (π - α) = -3 / 5 := by
  sorry

end cos_pi_minus_alpha_l162_162733


namespace gcd_of_72_and_90_l162_162904

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end gcd_of_72_and_90_l162_162904


namespace area_bounded_by_curves_eq_l162_162586

open Real

noncomputable def area_bounded_by_curves : ℝ :=
  1 / 2 * (∫ (φ : ℝ) in (π/4)..(π/2), (sqrt 2 * cos (φ - π / 4))^2) +
  1 / 2 * (∫ (φ : ℝ) in (π/2)..(3 * π / 4), (sqrt 2 * sin (φ - π / 4))^2)

theorem area_bounded_by_curves_eq : area_bounded_by_curves = (π + 2) / 4 :=
  sorry

end area_bounded_by_curves_eq_l162_162586


namespace percentage_of_500_l162_162533

theorem percentage_of_500 (P : ℝ) : 0.1 * (500 * P / 100) = 25 → P = 50 :=
by
  sorry

end percentage_of_500_l162_162533


namespace total_tiles_to_be_replaced_l162_162580

-- Define the given conditions
def horizontal_paths : List ℕ := [30, 50, 30, 20, 20, 50]
def vertical_paths : List ℕ := [20, 50, 20, 50, 50]
def intersections : ℕ := List.sum [2, 3, 3, 4, 4]

-- Problem statement: Prove that the total number of tiles to be replaced is 374
theorem total_tiles_to_be_replaced : List.sum horizontal_paths + List.sum vertical_paths - intersections = 374 := 
by sorry

end total_tiles_to_be_replaced_l162_162580


namespace num_pairs_equals_one_l162_162179

noncomputable def fractional_part (x : ℚ) : ℚ := x - x.floor

open BigOperators

theorem num_pairs_equals_one :
  ∃! (n : ℕ) (q : ℚ), 
    (0 < q ∧ q < 2000) ∧ 
    ¬ q.isInt ∧ 
    fractional_part (q^2) = fractional_part (n.choose 2000)
:= sorry

end num_pairs_equals_one_l162_162179


namespace sum_first_five_terms_arithmetic_seq_l162_162506

theorem sum_first_five_terms_arithmetic_seq
  (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_a2 : a 2 = 5)
  (h_a4 : a 4 = 9)
  : (Finset.range 5).sum a = 35 := by
  sorry

end sum_first_five_terms_arithmetic_seq_l162_162506


namespace main_theorem_l162_162565

noncomputable def main_expr := (Real.pi - 2019) ^ 0 + |Real.sqrt 3 - 1| + (-1 / 2)⁻¹ - 2 * Real.tan (Real.pi / 6)

theorem main_theorem : main_expr = -2 + Real.sqrt 3 / 3 := by
  sorry

end main_theorem_l162_162565


namespace average_scissors_correct_l162_162054

-- Definitions for the initial number of scissors in each drawer
def initial_scissors_first_drawer : ℕ := 39
def initial_scissors_second_drawer : ℕ := 27
def initial_scissors_third_drawer : ℕ := 45

-- Definitions for the new scissors added by Dan
def added_scissors_first_drawer : ℕ := 13
def added_scissors_second_drawer : ℕ := 7
def added_scissors_third_drawer : ℕ := 10

-- Calculate the final number of scissors after Dan's addition
def final_scissors_first_drawer : ℕ := initial_scissors_first_drawer + added_scissors_first_drawer
def final_scissors_second_drawer : ℕ := initial_scissors_second_drawer + added_scissors_second_drawer
def final_scissors_third_drawer : ℕ := initial_scissors_third_drawer + added_scissors_third_drawer

-- Statement to prove the average number of scissors in all three drawers
theorem average_scissors_correct :
  (final_scissors_first_drawer + final_scissors_second_drawer + final_scissors_third_drawer) / 3 = 47 := by
  sorry

end average_scissors_correct_l162_162054


namespace employed_male_percent_problem_l162_162182

noncomputable def employed_percent_population (total_population_employed_percent : ℝ) (employed_females_percent : ℝ) : ℝ :=
  let employed_males_percent := (1 - employed_females_percent) * total_population_employed_percent
  employed_males_percent

theorem employed_male_percent_problem :
  employed_percent_population 0.72 0.50 = 0.36 := by
  sorry

end employed_male_percent_problem_l162_162182


namespace initial_bottle_caps_l162_162801

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end initial_bottle_caps_l162_162801


namespace p_and_q_together_complete_in_10_days_l162_162952

noncomputable def p_time := 50 / 3
noncomputable def q_time := 25
noncomputable def r_time := 50

theorem p_and_q_together_complete_in_10_days 
  (h1 : 1 / p_time = 1 / q_time + 1 / r_time)
  (h2 : r_time = 50)
  (h3 : q_time = 25) :
  (p_time * q_time) / (p_time + q_time) = 10 :=
by
  sorry

end p_and_q_together_complete_in_10_days_l162_162952


namespace total_cost_with_discounts_l162_162471

theorem total_cost_with_discounts :
  let red_roses := 2 * 12
  let white_roses := 1 * 12
  let yellow_roses := 2 * 12
  let cost_red := red_roses * 6
  let cost_white := white_roses * 7
  let cost_yellow := yellow_roses * 5
  let total_cost_before_discount := cost_red + cost_white + cost_yellow
  let first_discount := 0.15 * total_cost_before_discount
  let cost_after_first_discount := total_cost_before_discount - first_discount
  let additional_discount := 0.10 * cost_after_first_discount
  let total_cost := cost_after_first_discount - additional_discount
  total_cost = 266.22 := by
  sorry

end total_cost_with_discounts_l162_162471


namespace gasoline_expense_l162_162682

-- Definitions for the conditions
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10
def initial_amount : ℝ := 50
def amount_left_for_return_trip : ℝ := 36.35

-- Definition for the total gift cost
def total_gift_cost : ℝ := number_of_people * gift_cost_per_person

-- Definition for the total amount received from grandma
def total_grandma_gift : ℝ := number_of_people * grandma_gift_per_person

-- Definition for the total initial amount including the gift from grandma
def total_initial_amount_with_gift : ℝ := initial_amount + total_grandma_gift

-- Definition for remaining amount after spending on lunch and gifts
def remaining_after_known_expenses : ℝ := total_initial_amount_with_gift - lunch_cost - total_gift_cost

-- The Lean theorem to prove the gasoline expense
theorem gasoline_expense : remaining_after_known_expenses - amount_left_for_return_trip = 8 := by
  sorry

end gasoline_expense_l162_162682


namespace cost_of_adult_ticket_l162_162346

def cost_of_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50
def adult_tickets : ℕ := 5

theorem cost_of_adult_ticket
  (A : ℝ)
  (h : 5 * A + 16 * cost_of_child_ticket = total_cost) :
  A = 5.50 :=
by
  sorry

end cost_of_adult_ticket_l162_162346


namespace integral_right_angled_triangles_unique_l162_162591

theorem integral_right_angled_triangles_unique : 
  ∀ a b c : ℤ, (a < b) ∧ (b < c) ∧ (a^2 + b^2 = c^2) ∧ (a * b = 4 * (a + b + c))
  ↔ (a = 10 ∧ b = 24 ∧ c = 26)
  ∨ (a = 12 ∧ b = 16 ∧ c = 20)
  ∨ (a = 9 ∧ b = 40 ∧ c = 41) :=
by {
  sorry
}

end integral_right_angled_triangles_unique_l162_162591


namespace profit_percentage_l162_162504

theorem profit_percentage (SP : ℝ) (h : SP > 0) (CP : ℝ) (h1 : CP = 0.96 * SP) :
  (SP - CP) / CP * 100 = 4.17 :=
by
  sorry

end profit_percentage_l162_162504


namespace largest_multiple_of_7_gt_neg_150_l162_162896

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l162_162896


namespace smallest_value_am_hm_inequality_l162_162153

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end smallest_value_am_hm_inequality_l162_162153


namespace range_h_l162_162171

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem range_h (a b : ℝ) (h_range : Set.Ioo a b = Set.Icc 0 1) : a + b = 1 := by
  sorry

end range_h_l162_162171


namespace max_value_k_l162_162211

noncomputable def sqrt_minus (x : ℝ) : ℝ := Real.sqrt (x - 3)
noncomputable def sqrt_six_minus (x : ℝ) : ℝ := Real.sqrt (6 - x)

theorem max_value_k (k : ℝ) : (∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ sqrt_minus x + sqrt_six_minus x ≥ k) ↔ k ≤ Real.sqrt 12 := by
  sorry

end max_value_k_l162_162211


namespace polar_coordinates_to_rectangular_l162_162765

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_coordinates_to_rectangular :
  polar_to_rectangular 10 (11 * Real.pi / 6) = (5 * Real.sqrt 3, -5) :=
by
  sorry

end polar_coordinates_to_rectangular_l162_162765


namespace product_equals_sum_only_in_two_cases_l162_162858

theorem product_equals_sum_only_in_two_cases (x y : ℤ) : 
  x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by 
  sorry

end product_equals_sum_only_in_two_cases_l162_162858


namespace distance_from_stream_to_meadow_l162_162830

noncomputable def distance_from_car_to_stream : ℝ := 0.2
noncomputable def distance_from_meadow_to_campsite : ℝ := 0.1
noncomputable def total_distance_hiked : ℝ := 0.7

theorem distance_from_stream_to_meadow : 
  (total_distance_hiked - distance_from_car_to_stream - distance_from_meadow_to_campsite = 0.4) :=
by
  sorry

end distance_from_stream_to_meadow_l162_162830


namespace triangle_side_length_l162_162312

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 20)
  (h2 : (1 / 2) * b * c * (Real.sin (Real.pi / 3)) = 10 * Real.sqrt 3) : a = 7 :=
sorry

end triangle_side_length_l162_162312


namespace additional_plates_added_l162_162585

def initial_plates : ℕ := 27
def added_plates : ℕ := 37
def total_plates : ℕ := 83

theorem additional_plates_added :
  total_plates - (initial_plates + added_plates) = 19 :=
by
  sorry

end additional_plates_added_l162_162585


namespace no_polyhedron_with_surface_area_2015_l162_162566

theorem no_polyhedron_with_surface_area_2015 : 
  ¬ ∃ (n k : ℤ), 6 * n - 2 * k = 2015 :=
by
  sorry

end no_polyhedron_with_surface_area_2015_l162_162566


namespace minimum_value_func1_minimum_value_func2_l162_162715

-- Problem (1): 
theorem minimum_value_func1 (x : ℝ) (h : x > -1) : 
  (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

-- Problem (2): 
theorem minimum_value_func2 (x : ℝ) (h : x > 1) : 
  (x^2 + 8) / (x - 1) ≥ 8 :=
sorry

end minimum_value_func1_minimum_value_func2_l162_162715


namespace arithmetic_mean_of_remaining_numbers_l162_162749

-- Definitions and conditions
def initial_set_size : ℕ := 60
def initial_arithmetic_mean : ℕ := 45
def numbers_to_remove : List ℕ := [50, 55, 60]

-- Calculation of the total sum
def total_sum : ℕ := initial_arithmetic_mean * initial_set_size

-- Calculation of the sum of the numbers to remove
def sum_of_removed_numbers : ℕ := numbers_to_remove.sum

-- Sum of the remaining numbers
def new_sum : ℕ := total_sum - sum_of_removed_numbers

-- Size of the remaining set
def remaining_set_size : ℕ := initial_set_size - numbers_to_remove.length

-- The arithmetic mean of the remaining numbers
def new_arithmetic_mean : ℚ := new_sum / remaining_set_size

-- The proof statement
theorem arithmetic_mean_of_remaining_numbers :
  new_arithmetic_mean = 2535 / 57 :=
by
  sorry

end arithmetic_mean_of_remaining_numbers_l162_162749


namespace additional_cats_l162_162298

theorem additional_cats {M R C : ℕ} (h1 : 20 * R = M) (h2 : 4 + 2 * C = 10) : C = 3 := 
  sorry

end additional_cats_l162_162298


namespace ways_to_distribute_books_into_bags_l162_162797

theorem ways_to_distribute_books_into_bags : 
  let books := 5
  let bags := 4
  ∃ (ways : ℕ), ways = 41 := 
sorry

end ways_to_distribute_books_into_bags_l162_162797


namespace determine_x_l162_162889

-- Definitions for given conditions
variables (x y z a b c : ℝ)
variables (h₁ : xy / (x - y) = a) (h₂ : xz / (x - z) = b) (h₃ : yz / (y - z) = c)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- Main statement to prove
theorem determine_x :
  x = (2 * a * b * c) / (a * b + b * c + c * a) :=
sorry

end determine_x_l162_162889


namespace find_f_2011_l162_162950

theorem find_f_2011 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f (x + 1) * f (x - 1) = 1) 
  (h3 : ∀ x, f x > 0) : 
  f 2011 = 1 := 
sorry

end find_f_2011_l162_162950


namespace arithmetic_sequence_k_value_l162_162418

theorem arithmetic_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ)
  (S_pos : S 2016 > 0) (S_neg : S 2017 < 0)
  (H : ∀ n, |a n| ≥ |a 1009| ): k = 1009 :=
sorry

end arithmetic_sequence_k_value_l162_162418


namespace radius_larger_circle_l162_162079

theorem radius_larger_circle (r : ℝ) (AC BC : ℝ) (h1 : 5 * r = AC / 2) (h2 : 15 = BC) : 
  5 * r = 18.75 :=
by
  sorry

end radius_larger_circle_l162_162079


namespace solve_abs_eq_l162_162254

theorem solve_abs_eq (x : ℝ) : |2*x - 6| = 3*x + 6 ↔ x = 0 :=
by 
  sorry

end solve_abs_eq_l162_162254


namespace percentage_rotten_bananas_l162_162076

theorem percentage_rotten_bananas :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges_percentage := 0.15
  let good_condition_percentage := 0.878
  let total_fruits := total_oranges + total_bananas 
  let rotten_oranges := rotten_oranges_percentage * total_oranges 
  let good_fruits := good_condition_percentage * total_fruits
  let rotten_fruits := total_fruits - good_fruits
  let rotten_bananas := rotten_fruits - rotten_oranges
  (rotten_bananas / total_bananas) * 100 = 8 := by
  {
    -- Calculations and simplifications go here
    sorry
  }

end percentage_rotten_bananas_l162_162076


namespace base9_first_digit_is_4_l162_162681

-- Define the base three representation of y
def y_base3 : Nat := 112211

-- Function to convert a given number from base 3 to base 10
def base3_to_base10 (n : Nat) : Nat :=
  let rec convert (n : Nat) (acc : Nat) (place : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * (3 ^ place)) (place + 1)
  convert n 0 0

-- Compute the base 10 representation of y
def y_base10 : Nat := base3_to_base10 y_base3

-- Function to convert a given number from base 10 to base 9
def base10_to_base9 (n : Nat) : List Nat :=
  let rec convert (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc
    else convert (n / 9) ((n % 9) :: acc)
  convert n []

-- Compute the base 9 representation of y as a list of digits
def y_base9 : List Nat := base10_to_base9 y_base10

-- Get the first digit (most significant digit) of the base 9 representation of y
def first_digit_base9 (digits : List Nat) : Nat :=
  digits.headD 0

-- The statement to prove
theorem base9_first_digit_is_4 : first_digit_base9 y_base9 = 4 := by sorry

end base9_first_digit_is_4_l162_162681


namespace polynomial_at_x_is_minus_80_l162_162875

def polynomial (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def x_value : ℤ := 2

theorem polynomial_at_x_is_minus_80 : polynomial x_value = -80 := 
by
  sorry

end polynomial_at_x_is_minus_80_l162_162875


namespace negation_universal_proposition_l162_162047

theorem negation_universal_proposition :
  (¬∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_universal_proposition_l162_162047


namespace cubes_sum_l162_162100

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l162_162100


namespace find_rate_of_stream_l162_162187

noncomputable def rate_of_stream (v : ℝ) : Prop :=
  let rowing_speed := 36
  let downstream_speed := rowing_speed + v
  let upstream_speed := rowing_speed - v
  (1 / upstream_speed) = 3 * (1 / downstream_speed)

theorem find_rate_of_stream : ∃ v : ℝ, rate_of_stream v ∧ v = 18 :=
by
  use 18
  unfold rate_of_stream
  sorry

end find_rate_of_stream_l162_162187


namespace average_children_with_children_l162_162095

theorem average_children_with_children (total_families : ℕ) (average_children : ℚ) (childless_families : ℕ) :
  total_families = 12 →
  average_children = 3 →
  childless_families = 3 →
  (total_families * average_children) / (total_families - childless_families) = 4.0 :=
by
  intros h1 h2 h3
  sorry

end average_children_with_children_l162_162095


namespace range_of_m_in_third_quadrant_l162_162327

theorem range_of_m_in_third_quadrant (m : ℝ) : (1 - (1/3) * m < 0) ∧ (m - 5 < 0) ↔ (3 < m ∧ m < 5) := 
by 
  intros
  sorry

end range_of_m_in_third_quadrant_l162_162327


namespace expression_value_l162_162250

variable (m n : ℝ)

theorem expression_value (hm : 3 * m ^ 2 + 5 * m - 3 = 0)
                         (hn : 3 * n ^ 2 - 5 * n - 3 = 0)
                         (hneq : m * n ≠ 1) :
                         (1 / n ^ 2) + (m / n) - (5 / 3) * m = 25 / 9 :=
by {
  sorry
}

end expression_value_l162_162250


namespace find_coordinates_A_l162_162538

-- Define the point A
structure Point where
  x : ℝ
  y : ℝ

def PointA (a : ℝ) : Point :=
  { x := 3 * a + 2, y := 2 * a - 4 }

-- Define the conditions
def condition1 (a : ℝ) := (PointA a).y = 4

def condition2 (a : ℝ) := |(PointA a).x| = |(PointA a).y|

-- The coordinates solutions to be proven
def valid_coordinates (p : Point) : Prop :=
  p = { x := 14, y := 4 } ∨
  p = { x := -16, y := -16 } ∨
  p = { x := 3.2, y := -3.2 }

-- Main theorem to prove
theorem find_coordinates_A (a : ℝ) :
  (condition1 a ∨ condition2 a) → valid_coordinates (PointA a) :=
by
  sorry

end find_coordinates_A_l162_162538


namespace max_profit_at_one_device_l162_162665

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2

def fixed_monthly_cost : ℝ := 40

def material_cost_per_device : ℝ := 5

noncomputable def cost (x : ℕ) : ℝ := fixed_monthly_cost + material_cost_per_device * x

noncomputable def profit_function (x : ℕ) : ℝ := (revenue x) - (cost x)

noncomputable def marginal_profit_function (x : ℕ) : ℝ :=
  profit_function (x + 1) - profit_function x

theorem max_profit_at_one_device :
  marginal_profit_function 1 = 24.4 ∧
  ∀ x : ℕ, marginal_profit_function x ≤ 24.4 := sorry

end max_profit_at_one_device_l162_162665


namespace markup_percentage_l162_162962

variable (W R : ℝ) -- W for Wholesale Cost, R for Retail Cost

-- Conditions:
-- 1. The sweater is sold at a 40% discount.
-- 2. When sold at a 40% discount, the merchant nets a 30% profit on the wholesale cost.
def discount_price (R : ℝ) : ℝ := 0.6 * R
def profit_price (W : ℝ) : ℝ := 1.3 * W

-- Hypotheses
axiom wholesale_cost_is_positive : W > 0
axiom discount_condition : discount_price R = profit_price W

-- Question: Prove that the percentage markup from wholesale to retail price is 116.67%.
theorem markup_percentage (W R : ℝ) 
  (wholesale_cost_is_positive : W > 0)
  (discount_condition : discount_price R = profit_price W) :
  ((R - W) / W * 100) = 116.67 := by
  sorry

end markup_percentage_l162_162962


namespace remainder_when_divided_by_95_l162_162242

theorem remainder_when_divided_by_95 (x : ℤ) (h1 : x % 19 = 12) :
  x % 95 = 12 := 
sorry

end remainder_when_divided_by_95_l162_162242


namespace arithmetic_sequence_property_l162_162363

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property
  (h1 : a 6 + a 8 = 10)
  (h2 : a 3 = 1)
  (property : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q)
  : a 11 = 9 :=
by
  sorry

end arithmetic_sequence_property_l162_162363


namespace min_value_x3_y2_z_w2_l162_162772

theorem min_value_x3_y2_z_w2 (x y z w : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w)
  (h : (1/x) + (1/y) + (1/z) + (1/w) = 8) : x^3 * y^2 * z * w^2 ≥ 1/432 :=
by
  sorry

end min_value_x3_y2_z_w2_l162_162772


namespace ratio_twelfth_term_geometric_sequence_l162_162818

theorem ratio_twelfth_term_geometric_sequence (G H : ℕ → ℝ) (n : ℕ) (a r b s : ℝ)
  (hG : ∀ n, G n = a * (r^n - 1) / (r - 1))
  (hH : ∀ n, H n = b * (s^n - 1) / (s - 1))
  (ratio_condition : ∀ n, G n / H n = (5 * n + 3) / (3 * n + 17)) :
  (a * r^11) / (b * s^11) = 2 / 5 :=
by 
  sorry

end ratio_twelfth_term_geometric_sequence_l162_162818


namespace calculate_y_l162_162529

theorem calculate_y (x y : ℝ) (h1 : x = 101) (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : y = 1 / 10 :=
by
  sorry

end calculate_y_l162_162529


namespace expression_eval_neg_sqrt_l162_162712

variable (a : ℝ)

theorem expression_eval_neg_sqrt (ha : a < 0) : a * Real.sqrt (-1 / a) = -Real.sqrt (-a) :=
by
  sorry

end expression_eval_neg_sqrt_l162_162712


namespace max_mondays_in_first_51_days_l162_162532

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end max_mondays_in_first_51_days_l162_162532


namespace all_stones_weigh_the_same_l162_162714

theorem all_stones_weigh_the_same (x : Fin 13 → ℕ)
  (h : ∀ (i : Fin 13), ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧
    i ∉ A ∧ i ∉ B ∧ ∀ (j k : Fin 13), j ∈ A → k ∈ B → x j = x k): 
  ∀ i j : Fin 13, x i = x j := 
sorry

end all_stones_weigh_the_same_l162_162714


namespace minimum_value_of_x_y_l162_162824

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  x + y

theorem minimum_value_of_x_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 - x) * (-y) = x) : minimum_value x y = 4 :=
  sorry

end minimum_value_of_x_y_l162_162824


namespace circle_tangent_area_l162_162342

noncomputable def circle_tangent_area_problem 
  (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) : ℝ :=
  if (radiusA = 1 ∧ radiusB = 1 ∧ radiusC = 2 ∧ tangent_midpoint) then 
    (4 * Real.pi) - (2 * Real.pi) 
  else 0

theorem circle_tangent_area (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) :
  radiusA = 1 → radiusB = 1 → radiusC = 2 → tangent_midpoint = true → 
  circle_tangent_area_problem radiusA radiusB radiusC tangent_midpoint = 2 * Real.pi :=
by
  intros
  simp [circle_tangent_area_problem]
  split_ifs
  · sorry
  · sorry

end circle_tangent_area_l162_162342


namespace at_least_one_ge_two_l162_162776

theorem at_least_one_ge_two (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a + 1 / b ≥ 2 ∨ b + 1 / c ≥ 2 ∨ c + 1 / a ≥ 2 := 
sorry

end at_least_one_ge_two_l162_162776


namespace area_S_inequality_l162_162872

noncomputable def F (t : ℝ) : ℝ := 2 * (t - ⌊t⌋)

def S (t : ℝ) : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - F t) * (p.1 - F t) + p.2 * p.2 ≤ (F t) * (F t) }

theorem area_S_inequality (t : ℝ) : 0 ≤ π * (F t) ^ 2 ∧ π * (F t) ^ 2 ≤ 4 * π := 
by sorry

end area_S_inequality_l162_162872


namespace increasing_function_on_R_l162_162097

theorem increasing_function_on_R (x1 x2 : ℝ) (h : x1 < x2) : 3 * x1 + 2 < 3 * x2 + 2 := 
by
  sorry

end increasing_function_on_R_l162_162097


namespace relationship_of_y_l162_162150

theorem relationship_of_y {k y1 y2 y3 : ℝ} (hk : k > 0) :
  (y1 = k / -1) → (y2 = k / 2) → (y3 = k / 3) → y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_l162_162150


namespace lollipops_initial_count_l162_162277

theorem lollipops_initial_count (L : ℕ) (k : ℕ) 
  (h1 : L % 42 ≠ 0) 
  (h2 : (L + 22) % 42 = 0) : 
  L = 62 :=
by
  sorry

end lollipops_initial_count_l162_162277


namespace pencils_bought_l162_162676

theorem pencils_bought (total_spent notebook_cost ruler_cost pencil_cost : ℕ)
  (h_total : total_spent = 74)
  (h_notebook : notebook_cost = 35)
  (h_ruler : ruler_cost = 18)
  (h_pencil : pencil_cost = 7) :
  (total_spent - (notebook_cost + ruler_cost)) / pencil_cost = 3 :=
by
  sorry

end pencils_bought_l162_162676


namespace no_such_integers_l162_162296

theorem no_such_integers (x y : ℤ) : ¬ ∃ x y : ℤ, (x^4 + 6) % 13 = y^3 % 13 :=
sorry

end no_such_integers_l162_162296


namespace total_bins_correct_l162_162409

def total_bins (soup vegetables pasta : ℝ) : ℝ :=
  soup + vegetables + pasta

theorem total_bins_correct : total_bins 0.12 0.12 0.5 = 0.74 :=
  by
    sorry

end total_bins_correct_l162_162409


namespace bailey_points_final_game_l162_162910

def chandra_points (a: ℕ) := 2 * a
def akiko_points (m: ℕ) := m + 4
def michiko_points (b: ℕ) := b / 2
def team_total_points (b c a m: ℕ) := b + c + a + m

theorem bailey_points_final_game (B: ℕ) 
  (M : ℕ := michiko_points B)
  (A : ℕ := akiko_points M)
  (C : ℕ := chandra_points A)
  (H : team_total_points B C A M = 54): B = 14 :=
by 
  sorry

end bailey_points_final_game_l162_162910


namespace g_periodic_6_l162_162074

def g (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a + b, b + c, a + c)

def g_iter (n : Nat) (triple : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match n with
  | 0 => triple
  | n + 1 => g (g_iter n triple).1 (g_iter n triple).2.1 (g_iter n triple).2.2

theorem g_periodic_6 {a b c : ℝ} (h : ∃ n : Nat, n > 0 ∧ g_iter n (a, b, c) = (a, b, c))
  (h' : (a, b, c) ≠ (0, 0, 0)) : g_iter 6 (a, b, c) = (a, b, c) :=
by
  sorry

end g_periodic_6_l162_162074


namespace singer_worked_10_hours_per_day_l162_162467

noncomputable def hours_per_day_worked_on_one_song (total_songs : ℕ) (days_per_song : ℕ) (total_hours : ℕ) : ℕ :=
  total_hours / (total_songs * days_per_song)

theorem singer_worked_10_hours_per_day :
  hours_per_day_worked_on_one_song 3 10 300 = 10 := 
by
  sorry

end singer_worked_10_hours_per_day_l162_162467


namespace part1_part2_l162_162621

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end part1_part2_l162_162621


namespace total_cost_is_716_mom_has_enough_money_l162_162472

/-- Definition of the price of the table lamp -/
def table_lamp_price : ℕ := 86

/-- Definition of the price of the electric fan -/
def electric_fan_price : ℕ := 185

/-- Definition of the price of the bicycle -/
def bicycle_price : ℕ := 445

/-- The total cost of buying all three items -/
def total_cost : ℕ := table_lamp_price + electric_fan_price + bicycle_price

/-- Mom's money -/
def mom_money : ℕ := 300

/-- Problem 1: Prove that the total cost equals 716 -/
theorem total_cost_is_716 : total_cost = 716 := 
by 
  sorry

/-- Problem 2: Prove that Mom has enough money to buy a table lamp and an electric fan -/
theorem mom_has_enough_money : table_lamp_price + electric_fan_price ≤ mom_money :=
by 
  sorry

end total_cost_is_716_mom_has_enough_money_l162_162472


namespace function_is_zero_l162_162751

variable (n : ℕ) (a : Fin n → ℤ) (f : ℤ → ℝ)

axiom condition : ∀ (k l : ℤ), l ≠ 0 → (Finset.univ.sum (λ i => f (k + a i * l)) = 0)

theorem function_is_zero : ∀ x : ℤ, f x = 0 := by
  sorry

end function_is_zero_l162_162751


namespace initial_roses_in_vase_l162_162039

/-- 
There were some roses in a vase. Mary cut roses from her flower garden 
and put 16 more roses in the vase. There are now 22 roses in the vase.
Prove that the initial number of roses in the vase was 6. 
-/
theorem initial_roses_in_vase (initial_roses added_roses current_roses : ℕ) 
  (h_add : added_roses = 16) 
  (h_current : current_roses = 22) 
  (h_current_eq : current_roses = initial_roses + added_roses) : 
  initial_roses = 6 := 
by
  subst h_add
  subst h_current
  linarith

end initial_roses_in_vase_l162_162039


namespace point_translation_proof_l162_162389

def Point := (ℝ × ℝ)

def translate_right (p : Point) (d : ℝ) : Point := (p.1 + d, p.2)

theorem point_translation_proof :
  let A : Point := (1, 2)
  let A' := translate_right A 2
  A' = (3, 2) :=
by
  let A : Point := (1, 2)
  let A' := translate_right A 2
  show A' = (3, 2)
  sorry

end point_translation_proof_l162_162389


namespace total_spears_is_78_l162_162456

-- Define the spear production rates for each type of wood
def spears_from_sapling := 3
def spears_from_log := 9
def spears_from_bundle := 7
def spears_from_trunk := 15

-- Define the quantity of each type of wood
def saplings := 6
def logs := 1
def bundles := 3
def trunks := 2

-- Prove that the total number of spears is 78
theorem total_spears_is_78 : (saplings * spears_from_sapling) + (logs * spears_from_log) + (bundles * spears_from_bundle) + (trunks * spears_from_trunk) = 78 :=
by 
  -- Calculation can be filled here
  sorry

end total_spears_is_78_l162_162456


namespace relationship_abc_l162_162713

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l162_162713


namespace opposite_of_neg_three_sevenths_l162_162880

theorem opposite_of_neg_three_sevenths:
  ∀ x : ℚ, (x = -3 / 7) → (∃ y : ℚ, y + x = 0 ∧ y = 3 / 7) :=
by
  sorry

end opposite_of_neg_three_sevenths_l162_162880


namespace watermelon_melon_weight_l162_162539

variables {W M : ℝ}

theorem watermelon_melon_weight :
  (2 * W > 3 * M ∨ 3 * W > 4 * M) ∧ ¬ (2 * W > 3 * M ∧ 3 * W > 4 * M) → 12 * W ≤ 18 * M :=
by
  sorry

end watermelon_melon_weight_l162_162539


namespace rectangle_length_is_16_l162_162417

-- Define the conditions
def side_length_square : ℕ := 8
def width_rectangle : ℕ := 4
def area_square : ℕ := side_length_square ^ 2  -- Area of the square
def area_rectangle (length : ℕ) : ℕ := width_rectangle * length  -- Area of the rectangle

-- Lean 4 statement
theorem rectangle_length_is_16 (L : ℕ) (h : area_square = area_rectangle L) : L = 16 :=
by
  /- Proof will be inserted here -/
  sorry

end rectangle_length_is_16_l162_162417


namespace smallest_b_in_AP_l162_162423

theorem smallest_b_in_AP (a b c : ℝ) (d : ℝ) (ha : a = b - d) (hc : c = b + d) (habc : a * b * c = 125) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) : 
    b = 5 :=
by
  -- Proof needed here
  sorry

end smallest_b_in_AP_l162_162423


namespace find_middle_number_l162_162914

theorem find_middle_number (a b c d x e f g : ℝ) 
  (h1 : (a + b + c + d + x + e + f + g) / 8 = 7)
  (h2 : (a + b + c + d + x) / 5 = 6)
  (h3 : (x + e + f + g + d) / 5 = 9) :
  x = 9.5 := 
by 
  sorry

end find_middle_number_l162_162914


namespace binomial_coefficient_19_13_l162_162313

theorem binomial_coefficient_19_13 
  (h1 : Nat.choose 20 13 = 77520) 
  (h2 : Nat.choose 20 14 = 38760) 
  (h3 : Nat.choose 18 13 = 18564) :
  Nat.choose 19 13 = 37128 := 
sorry

end binomial_coefficient_19_13_l162_162313


namespace find_number_divided_l162_162271

theorem find_number_divided (n : ℕ) (h : n = 21 * 9 + 1) : n = 190 :=
by
  sorry

end find_number_divided_l162_162271


namespace parabola_vertex_coordinates_l162_162905

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = -(x - 1) ^ 2 + 3 → (1, 3) = (1, 3) :=
by
  intros x y h
  sorry

end parabola_vertex_coordinates_l162_162905


namespace cost_two_enchiladas_two_tacos_three_burritos_l162_162846

variables (e t b : ℝ)

theorem cost_two_enchiladas_two_tacos_three_burritos 
  (h1 : 2 * e + 3 * t + b = 5.00)
  (h2 : 3 * e + 2 * t + 2 * b = 7.50) : 
  2 * e + 2 * t + 3 * b = 10.625 :=
sorry

end cost_two_enchiladas_two_tacos_three_burritos_l162_162846


namespace largest_four_digit_divisible_by_8_l162_162537

/-- The largest four-digit number that is divisible by 8 is 9992. -/
theorem largest_four_digit_divisible_by_8 : ∃ x : ℕ, x = 9992 ∧ x < 10000 ∧ x % 8 = 0 ∧
  ∀ y : ℕ, y < 10000 ∧ y % 8 = 0 → y ≤ 9992 := 
by 
  sorry

end largest_four_digit_divisible_by_8_l162_162537


namespace no_such_natural_numbers_exist_l162_162265

theorem no_such_natural_numbers_exist :
  ¬ ∃ (x y : ℕ), ∃ (k m : ℕ), x^2 + x + 1 = y^k ∧ y^2 + y + 1 = x^m := 
by sorry

end no_such_natural_numbers_exist_l162_162265


namespace compute_u2_plus_v2_l162_162479

theorem compute_u2_plus_v2 (u v : ℝ) (hu : 1 < u) (hv : 1 < v)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^(Real.sqrt 5) + 7^(Real.sqrt 5) :=
by
  sorry

end compute_u2_plus_v2_l162_162479


namespace binom_600_600_l162_162646

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l162_162646


namespace football_team_throwers_l162_162614

theorem football_team_throwers {T N : ℕ} (h1 : 70 - T = N)
                                (h2 : 62 = T + (2 / 3 * N)) : 
                                T = 46 := 
by
  sorry

end football_team_throwers_l162_162614


namespace two_digit_integers_count_l162_162572

def digits : Set ℕ := {3, 5, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem two_digit_integers_count : 
  ∃ (count : ℕ), count = 16 ∧
  (∀ (t : ℕ), t ∈ digits → 
  ∀ (u : ℕ), u ∈ digits → 
  t ≠ u ∧ is_odd u → 
  (∃ n : ℕ, 10 * t + u = n)) :=
by
  -- The total number of unique two-digit integers is 16
  use 16
  -- Proof skipped
  sorry

end two_digit_integers_count_l162_162572


namespace solve_equation_l162_162402

theorem solve_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x * (x - 1)) = 0) ↔ x = 2 :=
by
  sorry

end solve_equation_l162_162402


namespace solve_problem_l162_162903

def bracket (a b c : ℕ) : ℕ := (a + b) / c

theorem solve_problem :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 :=
by
  sorry

end solve_problem_l162_162903


namespace debbie_total_tape_l162_162766

def large_box_tape : ℕ := 4
def medium_box_tape : ℕ := 2
def small_box_tape : ℕ := 1
def label_tape : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5

def total_tape_used : ℕ := 
  (large_boxes_packed * (large_box_tape + label_tape)) +
  (medium_boxes_packed * (medium_box_tape + label_tape)) +
  (small_boxes_packed * (small_box_tape + label_tape))

theorem debbie_total_tape : total_tape_used = 44 := by
  sorry

end debbie_total_tape_l162_162766


namespace proof_probability_second_science_given_first_arts_l162_162992

noncomputable def probability_second_science_given_first_arts : ℚ :=
  let total_questions := 5
  let science_questions := 3
  let arts_questions := 2

  -- Event A: drawing an arts question in the first draw.
  let P_A := arts_questions / total_questions

  -- Event AB: drawing an arts question in the first draw and a science question in the second draw.
  let P_AB := (arts_questions / total_questions) * (science_questions / (total_questions - 1))

  -- Conditional probability P(B|A): drawing a science question in the second draw given drawing an arts question in the first draw.
  P_AB / P_A

theorem proof_probability_second_science_given_first_arts :
  probability_second_science_given_first_arts = 3 / 4 :=
by
  -- Lean does not include the proof in the statement as required.
  sorry

end proof_probability_second_science_given_first_arts_l162_162992


namespace intersect_P_M_l162_162115

def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | |x| ≤ 3}

theorem intersect_P_M : (P ∩ M) = {x | 0 ≤ x ∧ x < 3} := by
  sorry

end intersect_P_M_l162_162115


namespace remainder_7n_mod_5_l162_162360

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
by 
  sorry

end remainder_7n_mod_5_l162_162360


namespace chapatis_ordered_l162_162017

theorem chapatis_ordered (C : ℕ) 
  (chapati_cost : ℕ) (plates_rice : ℕ) (rice_cost : ℕ)
  (plates_mixed_veg : ℕ) (mixed_veg_cost : ℕ)
  (ice_cream_cups : ℕ) (ice_cream_cost : ℕ)
  (total_amount_paid : ℕ)
  (cost_eq : chapati_cost = 6)
  (plates_rice_eq : plates_rice = 5)
  (rice_cost_eq : rice_cost = 45)
  (plates_mixed_veg_eq : plates_mixed_veg = 7)
  (mixed_veg_cost_eq : mixed_veg_cost = 70)
  (ice_cream_cups_eq : ice_cream_cups = 6)
  (ice_cream_cost_eq : ice_cream_cost = 40)
  (total_paid_eq : total_amount_paid = 1051) :
  6 * C + 5 * 45 + 7 * 70 + 6 * 40 = 1051 → C = 16 :=
by
  intro h
  sorry

end chapatis_ordered_l162_162017


namespace election_valid_vote_counts_l162_162983

noncomputable def totalVotes : ℕ := 900000
noncomputable def invalidPercentage : ℝ := 0.25
noncomputable def validVotes : ℝ := totalVotes * (1.0 - invalidPercentage)
noncomputable def fractionA : ℝ := 7 / 15
noncomputable def fractionB : ℝ := 5 / 15
noncomputable def fractionC : ℝ := 3 / 15
noncomputable def validVotesA : ℝ := fractionA * validVotes
noncomputable def validVotesB : ℝ := fractionB * validVotes
noncomputable def validVotesC : ℝ := fractionC * validVotes

theorem election_valid_vote_counts :
  validVotesA = 315000 ∧ validVotesB = 225000 ∧ validVotesC = 135000 := by
  sorry

end election_valid_vote_counts_l162_162983


namespace probability_X_eq_Y_l162_162002

-- Define the conditions as functions or predicates.
def is_valid_pair (x y : ℝ) : Prop :=
  -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi ∧ -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi ∧ Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Final statement asserting the required probability.
theorem probability_X_eq_Y :
  ∃ (prob : ℝ), prob = 1 / 11 ∧ ∀ (x y : ℝ), is_valid_pair x y → (x = y ∨ x ≠ y ∧ prob = 1/11) :=
  sorry

end probability_X_eq_Y_l162_162002


namespace students_between_jimin_yuna_l162_162029

theorem students_between_jimin_yuna 
  (total_students : ℕ) 
  (jimin_position : ℕ) 
  (yuna_position : ℕ) 
  (h1 : total_students = 32) 
  (h2 : jimin_position = 27) 
  (h3 : yuna_position = 11) 
  : (jimin_position - yuna_position - 1) = 15 := 
by
  sorry

end students_between_jimin_yuna_l162_162029


namespace weight_of_seventh_person_l162_162299

noncomputable def weight_of_six_people : ℕ := 6 * 156
noncomputable def new_average_weight (x : ℕ) : Prop := (weight_of_six_people + x) / 7 = 151

theorem weight_of_seventh_person (x : ℕ) (h : new_average_weight x) : x = 121 :=
by
  sorry

end weight_of_seventh_person_l162_162299


namespace years_passed_l162_162168

-- Let PV be the present value of the machine, FV be the final value of the machine, r be the depletion rate, and t be the time in years.
def PV : ℝ := 900
def FV : ℝ := 729
def r : ℝ := 0.10

-- The formula for exponential decay is FV = PV * (1 - r)^t.
-- Given FV = 729, PV = 900, and r = 0.10, we want to prove that t = 2.

theorem years_passed (t : ℕ) : FV = PV * (1 - r)^t → t = 2 := 
by 
  intro h
  sorry

end years_passed_l162_162168


namespace part_one_part_two_l162_162212

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part_one (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

theorem part_two (a : ℝ) (h_pos : 0 < a) :
  (∀ x, (x - 1) * (f x a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l162_162212


namespace statement_books_per_shelf_l162_162361

/--
A store initially has 40.0 coloring books.
Acquires 20.0 more books.
Uses 15 shelves to store the books equally.
-/
def initial_books : ℝ := 40.0
def acquired_books : ℝ := 20.0
def total_shelves : ℝ := 15.0

/-- 
Theorem statement: The number of coloring books on each shelf.
-/
theorem books_per_shelf : (initial_books + acquired_books) / total_shelves = 4.0 := by
  sorry

end statement_books_per_shelf_l162_162361


namespace polynomial_root_p_value_l162_162887

theorem polynomial_root_p_value (p : ℝ) : (3 : ℝ) ^ 3 + p * (3 : ℝ) - 18 = 0 → p = -3 :=
by
  intro h
  sorry

end polynomial_root_p_value_l162_162887


namespace flags_count_l162_162162

-- Define the colors available
inductive Color
| purple | gold | silver

-- Define the number of stripes on the flag
def number_of_stripes : Nat := 3

-- Define a function to calculate the total number of combinations
def total_flags (colors : Nat) (stripes : Nat) : Nat :=
  colors ^ stripes

-- The main theorem we want to prove
theorem flags_count : total_flags 3 number_of_stripes = 27 :=
by
  -- This is the statement only, and the proof is omitted
  sorry

end flags_count_l162_162162


namespace length_segment_ZZ_l162_162166

variable (Z : ℝ × ℝ) (Z' : ℝ × ℝ)

theorem length_segment_ZZ' 
  (h_Z : Z = (-5, 3)) (h_Z' : Z' = (5, 3)) : 
  dist Z Z' = 10 := by
  sorry

end length_segment_ZZ_l162_162166


namespace plan_b_cheaper_than_plan_a_l162_162194

theorem plan_b_cheaper_than_plan_a (x : ℕ) (h : 401 ≤ x) :
  2000 + 5 * x < 10 * x :=
by
  sorry

end plan_b_cheaper_than_plan_a_l162_162194


namespace percentage_decrease_l162_162392

variables (S : ℝ) (D : ℝ)
def initial_increase (S : ℝ) : ℝ := 1.5 * S
def final_gain (S : ℝ) : ℝ := 1.15 * S
def salary_after_decrease (S D : ℝ) : ℝ := (initial_increase S) * (1 - D)

theorem percentage_decrease :
  salary_after_decrease S D = final_gain S → D = 0.233333 :=
by
  sorry

end percentage_decrease_l162_162392


namespace proof_problem_l162_162743

noncomputable def a : ℝ := 3.54
noncomputable def b : ℝ := 1.32
noncomputable def result : ℝ := (a - b) * 2

theorem proof_problem : result = 4.44 := by
  sorry

end proof_problem_l162_162743


namespace coordinates_of_E_l162_162419

theorem coordinates_of_E :
  let A := (-2, 1)
  let B := (1, 4)
  let C := (4, -3)
  let ratio_AB := (1, 2)
  let ratio_CE_ED := (1, 4)
  let D := ( (ratio_AB.1 * B.1 + ratio_AB.2 * A.1) / (ratio_AB.1 + ratio_AB.2),
             (ratio_AB.1 * B.2 + ratio_AB.2 * A.2) / (ratio_AB.1 + ratio_AB.2) )
  let E := ( (ratio_CE_ED.1 * C.1 - ratio_CE_ED.2 * D.1) / (ratio_CE_ED.1 - ratio_CE_ED.2),
             (ratio_CE_ED.1 * C.2 - ratio_CE_ED.2 * D.2) / (ratio_CE_ED.1 - ratio_CE_ED.2) )
  E = (-8 / 3, 11 / 3) := by
  sorry

end coordinates_of_E_l162_162419


namespace polynomial_divisibility_l162_162581

theorem polynomial_divisibility (a : ℤ) (n : ℕ) (h_pos : 0 < n) : 
  (a ^ (2 * n + 1) + (a - 1) ^ (n + 2)) % (a ^ 2 - a + 1) = 0 :=
sorry

end polynomial_divisibility_l162_162581


namespace inverse_proportion_symmetry_l162_162534

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end inverse_proportion_symmetry_l162_162534


namespace example_problem_l162_162761

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end example_problem_l162_162761


namespace max_m_value_min_value_expression_l162_162068

-- Define the conditions for the inequality where the solution is the entire real line
theorem max_m_value (x m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
sorry

-- Define the conditions for a, b, c > 0 and their sum equal to 1
-- and prove the minimum value of 4a^2 + 9b^2 + c^2
theorem min_value_expression (a b c : ℝ) (hpos1 : a > 0) (hpos2 : b > 0) (hpos3 : c > 0) (hsum : a + b + c = 1) :
  4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧ (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 → a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
sorry

end max_m_value_min_value_expression_l162_162068


namespace same_solution_implies_value_of_m_l162_162692

theorem same_solution_implies_value_of_m (x m : ℤ) (h₁ : -5 * x - 6 = 3 * x + 10) (h₂ : -2 * m - 3 * x = 10) : m = -2 :=
by
  sorry

end same_solution_implies_value_of_m_l162_162692


namespace min_letters_required_l162_162379

theorem min_letters_required (n : ℕ) (hn : n = 26) : 
  ∃ k, (∀ (collectors : Fin n) (leader : Fin n), k = 2 * (n - 1)) := 
sorry

end min_letters_required_l162_162379


namespace sequence_difference_l162_162043

theorem sequence_difference
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) :
  a 2017 - a 2016 = 2016 :=
by
  sorry

end sequence_difference_l162_162043


namespace negation_of_proposition_l162_162599

theorem negation_of_proposition:
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 > 0) :=
by
  sorry

end negation_of_proposition_l162_162599


namespace count_values_of_b_l162_162142

theorem count_values_of_b : 
  ∃! n : ℕ, (n = 4) ∧ (∀ b : ℕ, (b > 0) → (b ≤ 100) → (∃ k : ℤ, 5 * b^2 + 12 * b + 4 = k^2) → 
    (b = 4 ∨ b = 20 ∨ b = 44 ∨ b = 76)) :=
by
  sorry

end count_values_of_b_l162_162142


namespace set_intersection_complement_eq_l162_162135

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Complement of B in U
def complement_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- The theorem statement
theorem set_intersection_complement_eq :
  A ∩ complement_B = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_complement_eq_l162_162135


namespace find_m_for_parallel_lines_l162_162391

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 3 * x - y + 2 = 0 → x + m * y - 3 = 0) →
  m = -1 / 3 := sorry

end find_m_for_parallel_lines_l162_162391


namespace problem_statement_l162_162267

def a := 596
def b := 130
def c := 270

theorem problem_statement : a - b - c = a - (b + c) := by
  sorry

end problem_statement_l162_162267


namespace derivative_at_zero_l162_162266

-- Given conditions
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- Theorem statement to prove
theorem derivative_at_zero : 
  deriv f 0 = 0 := 
by 
  sorry

end derivative_at_zero_l162_162266


namespace range_of_independent_variable_l162_162536

theorem range_of_independent_variable (x : ℝ) (h : ∃ y, y = 2 / (Real.sqrt (x - 3))) : x > 3 :=
sorry

end range_of_independent_variable_l162_162536


namespace selection_schemes_count_l162_162173

theorem selection_schemes_count :
  let total_teachers := 9
  let select_from_total := Nat.choose 9 3
  let select_all_male := Nat.choose 5 3
  let select_all_female := Nat.choose 4 3
  select_from_total - (select_all_male + select_all_female) = 420 := by
    sorry

end selection_schemes_count_l162_162173


namespace expenditure_ratio_l162_162735

/-- A man saves 35% of his income in the first year. -/
def saving_rate_first_year : ℝ := 0.35

/-- His income increases by 35% in the second year. -/
def income_increase_rate : ℝ := 0.35

/-- His savings increase by 100% in the second year. -/
def savings_increase_rate : ℝ := 1.0

theorem expenditure_ratio
  (I : ℝ)  -- first year income
  (S1 : ℝ := saving_rate_first_year * I)  -- first year saving
  (E1 : ℝ := I - S1)  -- first year expenditure
  (I2 : ℝ := I + income_increase_rate * I)  -- second year income
  (S2 : ℝ := 2 * S1)  -- second year saving (increases by 100%)
  (E2 : ℝ := I2 - S2)  -- second year expenditure
  :
  (E1 + E2) / E1 = 2
  :=
  sorry

end expenditure_ratio_l162_162735


namespace factorize_expr_l162_162023

def my_expr (a b : ℤ) : ℤ := 4 * a^2 * b - b

theorem factorize_expr (a b : ℤ) : my_expr a b = b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end factorize_expr_l162_162023


namespace avg_age_10_students_l162_162086

-- Defining the given conditions
def avg_age_15_students : ℕ := 15
def total_students : ℕ := 15
def avg_age_4_students : ℕ := 14
def num_4_students : ℕ := 4
def age_15th_student : ℕ := 9

-- Calculating the total age based on given conditions
def total_age_15_students : ℕ := avg_age_15_students * total_students
def total_age_4_students : ℕ := avg_age_4_students * num_4_students
def total_age_10_students : ℕ := total_age_15_students - total_age_4_students - age_15th_student

-- Problem to be proved
theorem avg_age_10_students : total_age_10_students / 10 = 16 := 
by sorry

end avg_age_10_students_l162_162086


namespace circumscribed_sphere_radius_l162_162430

/-- Define the right triangular prism -/
structure RightTriangularPrism :=
(AB AC BC : ℝ)
(AA1 : ℝ)
(h_base : AB = 4 * Real.sqrt 2 ∧ AC = 4 * Real.sqrt 2 ∧ BC = 8)
(h_height : AA1 = 6)

/-- The condition that the base is an isosceles right-angled triangle -/
structure IsoscelesRightAngledTriangle :=
(A B C : ℝ)
(AB AC : ℝ)
(BC : ℝ)
(h_isosceles_right : AB = AC ∧ BC = Real.sqrt (AB^2 + AC^2))

/-- The main theorem stating the radius of the circumscribed sphere -/
theorem circumscribed_sphere_radius (prism : RightTriangularPrism) 
    (base : IsoscelesRightAngledTriangle) 
    (h_base_correct : base.AB = prism.AB ∧ base.AC = prism.AC ∧ base.BC = prism.BC):
    ∃ radius : ℝ, radius = 5 := 
by
    sorry

end circumscribed_sphere_radius_l162_162430


namespace window_area_l162_162912

def meter_to_feet : ℝ := 3.28084
def length_in_meters : ℝ := 2
def width_in_feet : ℝ := 15

def length_in_feet := length_in_meters * meter_to_feet
def area_in_square_feet := length_in_feet * width_in_feet

theorem window_area : area_in_square_feet = 98.4252 := 
by
  sorry

end window_area_l162_162912


namespace linear_inequality_m_eq_zero_l162_162167

theorem linear_inequality_m_eq_zero (m : ℝ) (x : ℝ) : 
  ((m - 2) * x ^ |m - 1| - 3 > 6) → abs (m - 1) = 1 → m ≠ 2 → m = 0 := by
  intros h1 h2 h3
  -- Proof of m = 0 based on given conditions
  sorry

end linear_inequality_m_eq_zero_l162_162167


namespace trig_expression_value_l162_162507

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) : 
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by
  sorry

end trig_expression_value_l162_162507


namespace fifth_group_members_l162_162883

-- Define the number of members in the choir
def total_members : ℕ := 150 

-- Define the number of members in each group
def group1 : ℕ := 18 
def group2 : ℕ := 29 
def group3 : ℕ := 34 
def group4 : ℕ := 23 

-- Define the fifth group as the remaining members
def group5 : ℕ := total_members - (group1 + group2 + group3 + group4)

theorem fifth_group_members : group5 = 46 := sorry

end fifth_group_members_l162_162883


namespace find_fg_l162_162847

def f (x : ℕ) : ℕ := 3 * x^2 + 2
def g (x : ℕ) : ℕ := 4 * x + 1

theorem find_fg :
  f (g 3) = 509 :=
by
  sorry

end find_fg_l162_162847


namespace arithmetic_progression_integers_l162_162554

theorem arithmetic_progression_integers 
  (d : ℤ) (a : ℤ) (h_d_pos : d > 0)
  (h_progression : ∀ i j : ℤ, i ≠ j → ∃ k : ℤ, a * (a + i * d) = a + k * d)
  : ∀ n : ℤ, ∃ m : ℤ, a + n * d = m :=
by
  sorry

end arithmetic_progression_integers_l162_162554


namespace shortest_path_from_A_to_D_not_inside_circle_l162_162740

noncomputable def shortest_path_length : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (18, 24)
  let O : ℝ × ℝ := (9, 12)
  let r : ℝ := 15
  15 * Real.pi

theorem shortest_path_from_A_to_D_not_inside_circle :
  let A := (0, 0)
  let D := (18, 24)
  let O := (9, 12)
  let r := 15
  shortest_path_length = 15 * Real.pi := 
by
  sorry

end shortest_path_from_A_to_D_not_inside_circle_l162_162740


namespace scientific_notation_of_star_diameter_l162_162036

theorem scientific_notation_of_star_diameter:
    (∃ (c : ℝ) (n : ℕ), 1 ≤ c ∧ c < 10 ∧ 16600000000 = c * 10^n) → 
    16600000000 = 1.66 * 10^10 :=
by
  sorry

end scientific_notation_of_star_diameter_l162_162036


namespace homework_problem1_homework_problem2_l162_162588

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l162_162588


namespace tan_angle_identity_l162_162330

open Real

theorem tan_angle_identity (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : sin β / cos β = (1 + cos (2 * α)) / (2 * cos α + sin (2 * α))) :
  tan (α + 2 * β + π / 4) = -1 := 
sorry

end tan_angle_identity_l162_162330


namespace implies_neg_p_and_q_count_l162_162820

-- Definitions of the logical conditions
variables (p q : Prop)

def cond1 : Prop := p ∧ q
def cond2 : Prop := p ∧ ¬ q
def cond3 : Prop := ¬ p ∧ q
def cond4 : Prop := ¬ p ∧ ¬ q

-- Negative of the statement "p and q are both true"
def neg_p_and_q := ¬ (p ∧ q)

-- The Lean 4 statement to prove
theorem implies_neg_p_and_q_count :
  (cond2 p q → neg_p_and_q p q) ∧ 
  (cond3 p q → neg_p_and_q p q) ∧ 
  (cond4 p q → neg_p_and_q p q) ∧ 
  ¬ (cond1 p q → neg_p_and_q p q) :=
sorry

end implies_neg_p_and_q_count_l162_162820


namespace ellas_coins_worth_l162_162178

theorem ellas_coins_worth :
  ∀ (n d : ℕ), n + d = 18 → n = d + 2 → 5 * n + 10 * d = 130 := by
  intros n d h1 h2
  sorry

end ellas_coins_worth_l162_162178


namespace price_of_uniform_l162_162512

-- Definitions based on conditions
def total_salary : ℕ := 600
def months_worked : ℕ := 9
def months_in_year : ℕ := 12
def salary_received : ℕ := 400
def uniform_price (U : ℕ) : Prop := 
    (3/4 * total_salary) - salary_received = U

-- Theorem stating the price of the uniform
theorem price_of_uniform : ∃ U : ℕ, uniform_price U := by
  sorry

end price_of_uniform_l162_162512


namespace equivalent_proof_problem_l162_162205

-- Define the conditions as Lean 4 definitions
variable (x₁ x₂ : ℝ)

-- The conditions given in the problem
def condition1 : Prop := x₁ * Real.logb 2 x₁ = 1008
def condition2 : Prop := x₂ * 2^x₂ = 1008

-- The problem to be proved
theorem equivalent_proof_problem (hx₁ : condition1 x₁) (hx₂ : condition2 x₂) : 
  x₁ * x₂ = 1008 := 
sorry

end equivalent_proof_problem_l162_162205


namespace total_fence_used_l162_162667

-- Definitions based on conditions
variables {L W : ℕ}
def area (L W : ℕ) := L * W

-- Provided conditions as Lean definitions
def unfenced_side := 40
def yard_area := 240

-- The proof problem statement
theorem total_fence_used (L_eq : L = unfenced_side) (A_eq : area L W = yard_area) : (2 * W + L) = 52 :=
sorry

end total_fence_used_l162_162667


namespace perimeter_is_36_l162_162783

-- Define an equilateral triangle with a given side length
def equilateral_triangle_perimeter (side_length : ℝ) : ℝ :=
  3 * side_length

-- Given: The base of the equilateral triangle is 12 m
def base_length : ℝ := 12

-- Theorem: The perimeter of the equilateral triangle is 36 m
theorem perimeter_is_36 : equilateral_triangle_perimeter base_length = 36 :=
by
  -- Placeholder for the proof
  sorry

end perimeter_is_36_l162_162783


namespace reflection_correct_l162_162850

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def M : point := (3, 2)

theorem reflection_correct : reflect_x_axis M = (3, -2) :=
  sorry

end reflection_correct_l162_162850


namespace percentage_increase_variable_cost_l162_162890

noncomputable def variable_cost_first_year : ℝ := 26000
noncomputable def fixed_cost : ℝ := 40000
noncomputable def total_breeding_cost_third_year : ℝ := 71460

theorem percentage_increase_variable_cost (x : ℝ) 
  (h : 40000 + 26000 * (1 + x) ^ 2 = 71460) : 
  x = 0.1 := 
by sorry

end percentage_increase_variable_cost_l162_162890


namespace total_money_found_l162_162543

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end total_money_found_l162_162543


namespace range_of_dot_product_l162_162198

theorem range_of_dot_product
  (x y : ℝ)
  (on_ellipse : x^2 / 2 + y^2 = 1) :
  ∃ m n : ℝ, (m = 0) ∧ (n = 1) ∧ m ≤ x^2 / 2 ∧ x^2 / 2 ≤ n :=
sorry

end range_of_dot_product_l162_162198


namespace find_m_range_l162_162814

-- Define the mathematical objects and conditions
def condition_p (m : ℝ) : Prop :=
  (|1 - m| / Real.sqrt 2) > 1

def condition_q (m : ℝ) : Prop :=
  m < 4

-- Define the proof problem
theorem find_m_range (p q : Prop) (m : ℝ) 
  (hp : ¬ p) (hq : q) (hpq : p ∨ q)
  (hP_imp : p → condition_p m)
  (hQ_imp : q → condition_q m) : 
  1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2 := 
sorry

end find_m_range_l162_162814
