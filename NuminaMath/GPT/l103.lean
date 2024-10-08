import Mathlib

namespace odd_periodic_function_value_l103_103688

theorem odd_periodic_function_value
  (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = - f x)
  (periodic_f : ∀ x, f (x + 3) = f x)
  (bounded_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f 8.5 = -1 :=
sorry

end odd_periodic_function_value_l103_103688


namespace min_value_inequality_l103_103625

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 / x + 1 / y) * (4 * x + y) ≥ 9 ∧ ((1 / x + 1 / y) * (4 * x + y) = 9 ↔ y / x = 2) :=
by
  sorry

end min_value_inequality_l103_103625


namespace first_platform_length_l103_103357

noncomputable def length_of_first_platform (t1 t2 l_train l_plat2 time1 time2 : ℕ) : ℕ :=
  let s1 := (l_train + t1) / time1
  let s2 := (l_train + l_plat2) / time2
  if s1 = s2 then t1 else 0

theorem first_platform_length:
  ∀ (time1 time2 : ℕ) (l_train l_plat2 : ℕ), time1 = 15 → time2 = 20 → l_train = 350 → l_plat2 = 250 → length_of_first_platform 100 l_plat2 l_train l_plat2 time1 time2 = 100 :=
by
  intros time1 time2 l_train l_plat2 ht1 ht2 ht3 ht4
  rw [ht1, ht2, ht3, ht4]
  dsimp [length_of_first_platform]
  rfl

end first_platform_length_l103_103357


namespace old_supervisor_salary_correct_l103_103850

def old_supervisor_salary (W S_old : ℝ) : Prop :=
  let avg_old := (W + S_old) / 9
  let avg_new := (W + 510) / 9
  avg_old = 430 ∧ avg_new = 390 → S_old = 870

theorem old_supervisor_salary_correct (W : ℝ) :
  old_supervisor_salary W 870 :=
by
  unfold old_supervisor_salary
  intro h
  sorry

end old_supervisor_salary_correct_l103_103850


namespace erased_digit_is_4_l103_103522

def sum_of_digits (n : ℕ) : ℕ := 
  sorry -- definition of sum of digits

def D (N : ℕ) : ℕ := N - sum_of_digits N

theorem erased_digit_is_4 (N : ℕ) (x : ℕ) 
  (hD : D N % 9 = 0) 
  (h_sum : sum_of_digits (D N) - x = 131) 
  : x = 4 :=
by
  sorry

end erased_digit_is_4_l103_103522


namespace sequence_nth_term_l103_103685

theorem sequence_nth_term (a : ℕ → ℚ) (h : a 1 = 3 / 2 ∧ a 2 = 1 ∧ a 3 = 5 / 8 ∧ a 4 = 3 / 8) :
  ∀ n : ℕ, a n = (n^2 - 11*n + 34) / 16 := by
  sorry

end sequence_nth_term_l103_103685


namespace even_integers_count_form_3k_plus_4_l103_103977

theorem even_integers_count_form_3k_plus_4 
  (n : ℕ) (h1 : 20 ≤ n ∧ n ≤ 250)
  (h2 : ∃ k : ℕ, n = 3 * k + 4 ∧ Even n) : 
  ∃ N : ℕ, N = 39 :=
by {
  sorry
}

end even_integers_count_form_3k_plus_4_l103_103977


namespace g_zero_value_l103_103939

variables {R : Type*} [Ring R]

def polynomial_h (f g h : Polynomial R) : Prop :=
  h = f * g

def constant_term (p : Polynomial R) : R :=
  p.coeff 0

variables {f g h : Polynomial R}

theorem g_zero_value
  (Hf : constant_term f = 6)
  (Hh : constant_term h = -18)
  (H : polynomial_h f g h) :
  g.coeff 0 = -3 :=
by
  sorry

end g_zero_value_l103_103939


namespace find_ratio_of_a_b_l103_103587

noncomputable def slope_of_tangent_to_curve_at_P := 3 * 1^2 + 1

noncomputable def perpendicular_slope (a b : ℝ) : Prop :=
  slope_of_tangent_to_curve_at_P * (a / b) = -1

noncomputable def line_slope_eq_slope_of_tangent (a b : ℝ) : Prop := 
  perpendicular_slope a b

theorem find_ratio_of_a_b (a b : ℝ) 
  (h1 : a - b * 2 = 0) 
  (h2 : line_slope_eq_slope_of_tangent a b) : 
  a / b = -1 / 4 :=
by
  sorry

end find_ratio_of_a_b_l103_103587


namespace probability_of_top_card_heart_l103_103182

-- Define the total number of cards in the deck.
def total_cards : ℕ := 39

-- Define the number of hearts in the deck.
def hearts : ℕ := 13

-- Define the probability that the top card is a heart.
def probability_top_card_heart : ℚ := hearts / total_cards

-- State the theorem to prove.
theorem probability_of_top_card_heart : probability_top_card_heart = 1 / 3 :=
by
  sorry

end probability_of_top_card_heart_l103_103182


namespace probability_quadrant_l103_103760

theorem probability_quadrant
    (r : ℝ) (x y : ℝ)
    (h : x^2 + y^2 ≤ r^2) :
    (∃ p : ℝ, p = (1 : ℚ)/4) :=
by
  sorry

end probability_quadrant_l103_103760


namespace boat_speed_still_water_l103_103488

theorem boat_speed_still_water (b s : ℝ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := 
by 
  -- Solve the system of equations
  sorry

end boat_speed_still_water_l103_103488


namespace sum_of_even_conditions_l103_103839

theorem sum_of_even_conditions (m n : ℤ) :
  ((∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → ∃ p : ℤ, m + n = 2 * p) ∧
  (∃ q : ℤ, m + n = 2 * q → (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → False) :=
by
  sorry

end sum_of_even_conditions_l103_103839


namespace min_value_of_a_l103_103155

theorem min_value_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, -1 < x0 ∧ x0 ≤ 2 ∧ x0 - a > 0) → a = 2 :=
by
  sorry

end min_value_of_a_l103_103155


namespace inequality_proof_l103_103799

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  (2 * a / (a^2 + b * c) + 2 * b / (b^2 + c * a) + 2 * c / (c^2 + a * b)) ≤ (a / (b * c) + b / (c * a) + c / (a * b)) := 
sorry

end inequality_proof_l103_103799


namespace altitude_eqn_equidistant_eqn_l103_103313

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definition of a line in the form Ax + By + C = 0
structure Line :=
  (A B C : ℝ)
  (non_zero : A ≠ 0 ∨ B ≠ 0)

-- Equation of line l1 (altitude to side BC)
def l1 : Line := { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) }

-- Equation of line l2 (passing through C, equidistant from A and B), two possible values
def l2a : Line := { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) }
def l2b : Line := { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) }

-- Prove the equations for l1 and l2 are correct given the points A, B, and C
theorem altitude_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l1 = { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) } := sorry

theorem equidistant_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l2a = { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) } ∨
  l2b = { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) } := sorry

end altitude_eqn_equidistant_eqn_l103_103313


namespace glycerin_percentage_l103_103260

theorem glycerin_percentage (x : ℝ) 
  (h1 : 100 * 0.75 = 75)
  (h2 : 75 + 75 = 100)
  (h3 : 75 * 0.30 + (x/100) * 75 = 75) : x = 70 :=
by
  sorry

end glycerin_percentage_l103_103260


namespace least_k_9_l103_103494

open Nat

noncomputable def u : ℕ → ℝ
| 0     => 1 / 3
| (n+1) => 3 * u n - 3 * (u n) * (u n)

def M : ℝ := 0.5

def acceptable_error (n : ℕ): Prop := abs (u n - M) ≤ 1 / 2 ^ 500

theorem least_k_9 : ∃ k, 0 ≤ k ∧ acceptable_error k ∧ ∀ j, (0 ≤ j ∧ j < k) → ¬acceptable_error j ∧ k = 9 := by
  sorry

end least_k_9_l103_103494


namespace original_selling_price_l103_103945

theorem original_selling_price (P : ℝ) (d1 d2 d3 t : ℝ) (final_price : ℝ) :
  d1 = 0.32 → -- first discount
  d2 = 0.10 → -- loyalty discount
  d3 = 0.05 → -- holiday discount
  t = 0.15 → -- state tax
  final_price = 650 → 
  1.15 * P * (1 - d1) * (1 - d2) * (1 - d3) = final_price →
  P = 722.57 :=
sorry

end original_selling_price_l103_103945


namespace inequality_ineq_l103_103293

theorem inequality_ineq (x y : ℝ) (hx: x > Real.sqrt 2) (hy: y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
  sorry

end inequality_ineq_l103_103293


namespace value_of_x_l103_103634

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l103_103634


namespace circle_parametric_eq_l103_103578

theorem circle_parametric_eq 
  (a b r : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi):
  (∃ (x y : ℝ), (x = r * Real.cos θ + a ∧ y = r * Real.sin θ + b)) ↔ 
  (∃ (x' y' : ℝ), (x' = r * Real.cos θ ∧ y' = r * Real.sin θ)) :=
sorry

end circle_parametric_eq_l103_103578


namespace ratio_B_to_C_l103_103686

-- Definitions for conditions
def total_amount : ℕ := 1440
def B_amt : ℕ := 270
def A_amt := (1 / 3) * B_amt
def C_amt := total_amount - A_amt - B_amt

-- Theorem statement
theorem ratio_B_to_C : (B_amt : ℚ) / C_amt = 1 / 4 :=
  by
    sorry

end ratio_B_to_C_l103_103686


namespace positive_difference_between_numbers_l103_103533

theorem positive_difference_between_numbers:
  ∃ x y : ℤ, x + y = 40 ∧ 3 * y - 4 * x = 7 ∧ |y - x| = 6 := by
  sorry

end positive_difference_between_numbers_l103_103533


namespace shape_is_cone_l103_103963

-- Define the spherical coordinate system and the condition
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def shape (c : ℝ) (p : SphericalCoord) : Prop := p.φ ≤ c

-- The shape described by \(\exists c, \forall p \in SphericalCoord, shape c p\) is a cone
theorem shape_is_cone (c : ℝ) (p : SphericalCoord) : shape c p → (c ≥ 0 ∧ c ≤ π → shape c p = Cone) :=
by
  sorry

end shape_is_cone_l103_103963


namespace binomial_theorem_example_l103_103002

theorem binomial_theorem_example 
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : (2 - 1)^5 = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5)
  (h2 : (2 - (-1))^5 = a_0 - a_1 + a_2 * (-1)^2 - a_3 * (-1)^3 + a_4 * (-1)^4 - a_5 * (-1)^5)
  (h3 : a_5 = -1) :
  (a_0 + a_2 + a_4 : ℤ) / (a_1 + a_3 : ℤ) = -61 / 60 := 
sorry

end binomial_theorem_example_l103_103002


namespace find_x_from_w_condition_l103_103407

theorem find_x_from_w_condition :
  ∀ (x u y z w : ℕ), 
  (x = u + 7) → 
  (u = y + 5) → 
  (y = z + 12) → 
  (z = w + 25) → 
  (w = 100) → 
  x = 149 :=
by intros x u y z w h1 h2 h3 h4 h5
   sorry

end find_x_from_w_condition_l103_103407


namespace problem1_l103_103367

variable (m : ℤ)

theorem problem1 : m * (m - 3) + 3 * (3 - m) = (m - 3) ^ 2 := by
  sorry

end problem1_l103_103367


namespace no_real_solutions_l103_103531

theorem no_real_solutions :
  ∀ x y z : ℝ, ¬ (x + y + 2 + 4*x*y = 0 ∧ y + z + 2 + 4*y*z = 0 ∧ z + x + 2 + 4*z*x = 0) :=
by
  sorry

end no_real_solutions_l103_103531


namespace unknown_number_is_105_l103_103808

theorem unknown_number_is_105 :
  ∃ x : ℝ, x^2 + 94^2 = 19872 ∧ x = 105 :=
by
  sorry

end unknown_number_is_105_l103_103808


namespace range_of_H_l103_103353

def H (x : ℝ) : ℝ := 2 * |2 * x + 2| - 3 * |2 * x - 2|

theorem range_of_H : Set.range H = Set.Ici 8 := 
by 
  sorry

end range_of_H_l103_103353


namespace solution_count_l103_103758

theorem solution_count (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∃ (num_solutions : ℕ), 
    (num_solutions = 3 ∧ a = 1 ∨ a = -1) ∨ 
    (num_solutions = 2 ∧ a = Real.sqrt 2 ∨ a = -Real.sqrt 2)) :=
by sorry

end solution_count_l103_103758


namespace combined_depths_underwater_l103_103161

theorem combined_depths_underwater :
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  Ron_depth + Sam_depth + Dean_depth = 13 :=
by
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  show Ron_depth + Sam_depth + Dean_depth = 13
  sorry

end combined_depths_underwater_l103_103161


namespace min_distance_point_curve_to_line_l103_103879

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem min_distance_point_curve_to_line :
  ∀ (P : ℝ × ℝ), 
  curve P.1 = P.2 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 :=
by
  sorry

end min_distance_point_curve_to_line_l103_103879


namespace mod_remainder_w_l103_103056

theorem mod_remainder_w (w : ℕ) (h : w = 3^39) : w % 13 = 1 :=
by
  sorry

end mod_remainder_w_l103_103056


namespace flower_bed_area_l103_103523

theorem flower_bed_area (total_posts : ℕ) (corner_posts : ℕ) (spacing : ℕ) (long_side_multiplier : ℕ)
  (h1 : total_posts = 24)
  (h2 : corner_posts = 4)
  (h3 : spacing = 3)
  (h4 : long_side_multiplier = 3) :
  ∃ (area : ℕ), area = 144 := 
sorry

end flower_bed_area_l103_103523


namespace john_pack_count_l103_103178

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

end john_pack_count_l103_103178


namespace angle_C_measure_l103_103478

theorem angle_C_measure
  (D C : ℝ)
  (h1 : C + D = 90)
  (h2 : C = 3 * D) :
  C = 67.5 :=
by
  sorry

end angle_C_measure_l103_103478


namespace graph_intersection_points_l103_103350

open Function

theorem graph_intersection_points (g : ℝ → ℝ) (h_inv : Involutive (invFun g)) : 
  ∃! (x : ℝ), x = 0 ∨ x = 1 ∨ x = -1 → g (x^2) = g (x^6) :=
by sorry

end graph_intersection_points_l103_103350


namespace task_candy_distribution_l103_103677

noncomputable def candy_distribution_eq_eventually (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ m : ℕ, ∀ j : ℕ, m ≥ k → a (j + m * n) = a (0 + m * n)

theorem task_candy_distribution :
  ∀ n : ℕ, n > 0 →
  ∀ a : ℕ → ℕ,
  (∀ i : ℕ, a i = if a i % 2 = 1 then (a i) + 1 else a i) →
  (∀ i : ℕ, a (i + 1) = a i / 2 + a (i - 1) / 2) →
  candy_distribution_eq_eventually n a :=
by
  intros n n_positive a h_even h_transfer
  sorry

end task_candy_distribution_l103_103677


namespace georgie_window_ways_l103_103047

theorem georgie_window_ways (n : Nat) (h : n = 8) :
  let ways := n * (n - 1)
  ways = 56 := by
  sorry

end georgie_window_ways_l103_103047


namespace quadratic_real_roots_l103_103463

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 - 2 * x + 1 = 0) → (k ≤ 0 ∧ k ≠ -1) :=
by
  sorry

end quadratic_real_roots_l103_103463


namespace x_equals_1_over_16_l103_103366

-- Given conditions
def distance_center_to_tangents_intersection : ℚ := 3 / 8
def radius_of_circle : ℚ := 3 / 16
def distance_center_to_CD : ℚ := 1 / 2

-- Calculated total distance
def total_distance_center_to_C : ℚ := distance_center_to_tangents_intersection + radius_of_circle

-- Problem statement
theorem x_equals_1_over_16 (x : ℚ) 
    (h : total_distance_center_to_C = x + distance_center_to_CD) : 
    x = 1 / 16 := 
by
  -- Proof is omitted, based on the provided solution steps
  sorry

end x_equals_1_over_16_l103_103366


namespace equal_focal_distances_l103_103170

def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1
def ellipse2 (k x y : ℝ) (hk : k < 9) : Prop := x^2 / (25 - k) + y^2 / (9 - k) = 1

theorem equal_focal_distances (k : ℝ) (hk : k < 9) : 
  let f1 := 8
  let f2 := 8 
  f1 = f2 :=
by 
  sorry

end equal_focal_distances_l103_103170


namespace arithmetic_sequence_l103_103061

theorem arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) (h : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
sorry

end arithmetic_sequence_l103_103061


namespace even_goals_more_likely_l103_103475

theorem even_goals_more_likely (p₁ : ℝ) (q₁ : ℝ) 
  (h₁ : q₁ = 1 - p₁)
  (independent_halves : (p₁ * p₁ + q₁ * q₁) > (2 * p₁ * q₁)) :
  (p₁ * p₁ + q₁ * q₁) > (1 - (p₁ * p₁ + q₁ * q₁)) :=
by
  sorry

end even_goals_more_likely_l103_103475


namespace no_nat_pairs_divisibility_l103_103620

theorem no_nat_pairs_divisibility (a b : ℕ) (hab : b^a ∣ a^b - 1) : false :=
sorry

end no_nat_pairs_divisibility_l103_103620


namespace pow_div_simplify_l103_103212

theorem pow_div_simplify : (((15^15) / (15^14))^3 * 3^3) / 3^3 = 3375 := by
  sorry

end pow_div_simplify_l103_103212


namespace trapezium_height_l103_103650

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end trapezium_height_l103_103650


namespace gnomes_and_ponies_l103_103447

theorem gnomes_and_ponies (g p : ℕ) (h1 : g + p = 15) (h2 : 2 * g + 4 * p = 36) : g = 12 ∧ p = 3 :=
by
  sorry

end gnomes_and_ponies_l103_103447


namespace correct_product_of_a_b_l103_103129

theorem correct_product_of_a_b (a b : ℕ) (h1 : (a - (10 * (a / 10 % 10) + 1)) * b = 255)
                              (h2 : (a - (10 * (a / 100 % 10 * 10 + a % 10 - (a / 100 % 10 * 10 + 5 * 10)))) * b = 335) :
  a * b = 285 := sorry

end correct_product_of_a_b_l103_103129


namespace hydrochloric_acid_solution_l103_103402

variable (V : ℝ) (pure_acid_added : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ)

theorem hydrochloric_acid_solution :
  initial_concentration = 0.10 → 
  final_concentration = 0.15 → 
  pure_acid_added = 3.52941176471 → 
  0.10 * V + 3.52941176471 = 0.15 * (V + 3.52941176471) → 
  V = 60 :=
by
  intros h_initial h_final h_pure h_equation
  sorry

end hydrochloric_acid_solution_l103_103402


namespace length_of_DE_l103_103124

theorem length_of_DE (base : ℝ) (area_ratio : ℝ) (height_ratio : ℝ) :
  base = 18 → area_ratio = 0.09 → height_ratio = 0.3 → DE = 2 :=
by
  sorry

end length_of_DE_l103_103124


namespace dot_product_sum_eq_fifteen_l103_103052

-- Define the vectors a, b, and c
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1, y)
def vec_c : ℝ × ℝ := (3, -6)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditions from the problem
def cond_perpendicular (x : ℝ) : Prop :=
  dot_product (vec_a x) vec_c = 0

def cond_parallel (y : ℝ) : Prop :=
  1 / 3 = y / -6

-- Lean statement for the problem
theorem dot_product_sum_eq_fifteen (x y : ℝ)
  (h1 : cond_perpendicular x) 
  (h2 : cond_parallel y) :
  dot_product (vec_a x + vec_b y) vec_c = 15 :=
sorry

end dot_product_sum_eq_fifteen_l103_103052


namespace minimum_value_quadratic_function_l103_103989

-- Defining the quadratic function y
def quadratic_function (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

-- Statement asserting the minimum value of the quadratic function
theorem minimum_value_quadratic_function : ∃ (y_min : ℝ), (∀ x : ℝ, quadratic_function x ≥ y_min) ∧ y_min = 12 :=
by
  -- Here we would normally insert the proof, but we skip it with sorry
  sorry

end minimum_value_quadratic_function_l103_103989


namespace sum_of_circle_areas_constant_l103_103671

theorem sum_of_circle_areas_constant (r OP : ℝ) (h1 : 0 < r) (h2 : 0 ≤ OP ∧ OP < r) 
  (a' b' c' : ℝ) (h3 : a'^2 + b'^2 + c'^2 = OP^2) :
  ∃ (a b c : ℝ), (a^2 + b^2 + c^2 = 3 * r^2 - OP^2) :=
by
  sorry

end sum_of_circle_areas_constant_l103_103671


namespace average_weight_decrease_l103_103944

theorem average_weight_decrease :
  let original_avg := 102
  let new_weight := 40
  let original_boys := 30
  let total_boys := original_boys + 1
  (original_avg - ((original_boys * original_avg + new_weight) / total_boys)) = 2 :=
by
  sorry

end average_weight_decrease_l103_103944


namespace prob_return_to_freezer_l103_103404

-- Define the probabilities of picking two pops of each flavor
def probability_same_flavor (total: ℕ) (pop1: ℕ) (pop2: ℕ) : ℚ :=
  (pop1 * pop2) / (total * (total - 1))

-- Definitions according to the problem conditions
def cherry_pops : ℕ := 4
def orange_pops : ℕ := 3
def lemon_lime_pops : ℕ := 4
def total_pops : ℕ := cherry_pops + orange_pops + lemon_lime_pops

-- Calculate the probability of picking two ice pops of the same flavor
def prob_cherry : ℚ := probability_same_flavor total_pops cherry_pops (cherry_pops - 1)
def prob_orange : ℚ := probability_same_flavor total_pops orange_pops (orange_pops - 1)
def prob_lemon_lime : ℚ := probability_same_flavor total_pops lemon_lime_pops (lemon_lime_pops - 1)

def prob_same_flavor : ℚ := prob_cherry + prob_orange + prob_lemon_lime
def prob_diff_flavor : ℚ := 1 - prob_same_flavor

-- Theorem stating the probability of needing to return to the freezer
theorem prob_return_to_freezer : prob_diff_flavor = 8 / 11 := by
  sorry

end prob_return_to_freezer_l103_103404


namespace tangent_slope_l103_103744

noncomputable def f (x : ℝ) : ℝ := x - 1 + 1 / Real.exp x

noncomputable def f' (x : ℝ) : ℝ := 1 - 1 / Real.exp x

theorem tangent_slope (k : ℝ) (x₀ : ℝ) (y₀ : ℝ) 
  (h_tangent_point: (x₀ = -1) ∧ (y₀ = x₀ - 1 + 1 / Real.exp x₀))
  (h_tangent_line : ∀ x, y₀ = f x₀ + f' x₀ * (x - x₀)) :
  k = 1 - Real.exp 1 := 
sorry

end tangent_slope_l103_103744


namespace fisher_needed_score_l103_103276

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end fisher_needed_score_l103_103276


namespace steve_marbles_l103_103444

-- Define the initial condition variables
variables (S Steve_initial Sam_initial Sally_initial Sarah_initial Steve_now : ℕ)

-- Conditions
def cond1 : Sam_initial = 2 * Steve_initial := by sorry
def cond2 : Sally_initial = Sam_initial - 5 := by sorry
def cond3 : Sarah_initial = Steve_initial + 3 := by sorry
def cond4 : Steve_now = Steve_initial + 3 := by sorry
def cond5 : Sam_initial - (3 + 3 + 4) = 6 := by sorry

-- Goal
theorem steve_marbles : Steve_now = 11 := by sorry

end steve_marbles_l103_103444


namespace tourist_journey_home_days_l103_103830

theorem tourist_journey_home_days (x v : ℝ)
  (h1 : (x / 2 + 1) * v = 246)
  (h2 : x * (v + 15) = 276) :
  x + (x / 2 + 1) = 4 :=
by
  sorry

end tourist_journey_home_days_l103_103830


namespace find_value_l103_103907

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end find_value_l103_103907


namespace rise_in_water_level_l103_103555

theorem rise_in_water_level : 
  let edge_length : ℝ := 15
  let volume_cube : ℝ := edge_length ^ 3
  let length : ℝ := 20
  let width : ℝ := 15
  let base_area : ℝ := length * width
  let rise_in_level : ℝ := volume_cube / base_area
  rise_in_level = 11.25 :=
by
  sorry

end rise_in_water_level_l103_103555


namespace sara_lunch_total_l103_103112

theorem sara_lunch_total :
  let hotdog := 5.36
  let salad := 5.10
  hotdog + salad = 10.46 :=
by
  let hotdog := 5.36
  let salad := 5.10
  sorry

end sara_lunch_total_l103_103112


namespace sum_of_numbers_l103_103832

variable {R : Type*} [LinearOrderedField R]

theorem sum_of_numbers (x y : R) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 :=
by
  sorry

end sum_of_numbers_l103_103832


namespace current_height_of_tree_l103_103551

-- Definitions of conditions
def growth_per_year : ℝ := 0.5
def years : ℕ := 240
def final_height : ℝ := 720

-- The goal is to prove that the current height of the tree is 600 inches
theorem current_height_of_tree :
  final_height - (growth_per_year * years) = 600 := 
sorry

end current_height_of_tree_l103_103551


namespace sum_of_first_eleven_terms_l103_103959

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_eleven_terms 
  (h_arith : is_arithmetic_sequence a)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 7 - a 8 = 5) :
  S 11 = 55 :=
sorry

end sum_of_first_eleven_terms_l103_103959


namespace solve_quadratic_roots_l103_103217

theorem solve_quadratic_roots (b c : ℝ) 
  (h : {1, 2} = {x : ℝ | x^2 + b * x + c = 0}) : 
  b = -3 ∧ c = 2 :=
by
  sorry

end solve_quadratic_roots_l103_103217


namespace cara_sitting_pairs_l103_103391

theorem cara_sitting_pairs : ∀ (n : ℕ), n = 7 → ∃ (pairs : ℕ), pairs = 6 :=
by
  intros n hn
  have h : n - 1 = 6 := sorry
  exact ⟨n - 1, h⟩

end cara_sitting_pairs_l103_103391


namespace jameson_badminton_medals_l103_103911

theorem jameson_badminton_medals :
  ∃ (b : ℕ),  (∀ (t s : ℕ), t = 5 → s = 2 * t → t + s + b = 20) ∧ b = 5 :=
by {
sorry
}

end jameson_badminton_medals_l103_103911


namespace range_of_m_l103_103420

theorem range_of_m {x : ℝ} (m : ℝ) :
  (∀ x, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l103_103420


namespace renovate_total_time_eq_79_5_l103_103317

-- Definitions based on the given conditions
def time_per_bedroom : ℝ := 4
def num_bedrooms : ℕ := 3
def time_per_kitchen : ℝ := time_per_bedroom * 1.5
def time_per_garden : ℝ := 3
def time_per_terrace : ℝ := time_per_garden - 2
def time_per_basement : ℝ := time_per_kitchen * 0.75

-- Total time excluding the living room
def total_time_excl_living_room : ℝ :=
  (num_bedrooms * time_per_bedroom) +
  time_per_kitchen +
  time_per_garden +
  time_per_terrace +
  time_per_basement

-- Time for the living room
def time_per_living_room : ℝ := 2 * total_time_excl_living_room

-- Total time for everything
def total_time : ℝ := total_time_excl_living_room + time_per_living_room

-- The theorem we need to prove
theorem renovate_total_time_eq_79_5 : total_time = 79.5 := by
  sorry

end renovate_total_time_eq_79_5_l103_103317


namespace Dave_tiles_210_square_feet_l103_103010

theorem Dave_tiles_210_square_feet
  (ratio_charlie_dave : ℕ := 5 / 7)
  (total_area : ℕ := 360)
  : ∀ (work_done_by_dave : ℕ), work_done_by_dave = 210 :=
by
  sorry

end Dave_tiles_210_square_feet_l103_103010


namespace total_time_to_climb_seven_flights_l103_103712

-- Define the conditions
def first_flight_time : ℕ := 15
def difference_between_flights : ℕ := 10
def num_of_flights : ℕ := 7

-- Define the sum of an arithmetic series function
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the theorem
theorem total_time_to_climb_seven_flights :
  arithmetic_series_sum first_flight_time difference_between_flights num_of_flights = 315 :=
by
  sorry

end total_time_to_climb_seven_flights_l103_103712


namespace sugar_per_larger_cookie_l103_103278

theorem sugar_per_larger_cookie (c₁ c₂ : ℕ) (s₁ s₂ : ℝ) (h₁ : c₁ = 50) (h₂ : s₁ = 1 / 10) (h₃ : c₂ = 25) (h₄ : c₁ * s₁ = c₂ * s₂) : s₂ = 1 / 5 :=
by
  simp [h₁, h₂, h₃, h₄]
  sorry

end sugar_per_larger_cookie_l103_103278


namespace isosceles_triangle_base_angles_l103_103029

theorem isosceles_triangle_base_angles (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B ∨ B = C ∨ C = A) (h₃ : A = 80 ∨ B = 80 ∨ C = 80) :
  A = 50 ∨ B = 50 ∨ C = 50 ∨ A = 80 ∨ B = 80 ∨ C = 80 := 
by
  sorry

end isosceles_triangle_base_angles_l103_103029


namespace janet_pairs_of_2_l103_103711

def total_pairs (x y z : ℕ) : Prop := x + y + z = 18

def total_cost (x y z : ℕ) : Prop := 2 * x + 5 * y + 7 * z = 60

theorem janet_pairs_of_2 (x y z : ℕ) (h1 : total_pairs x y z) (h2 : total_cost x y z) (hz : z = 3) : x = 12 :=
by
  -- Proof is currently skipped
  sorry

end janet_pairs_of_2_l103_103711


namespace walk_to_bus_stop_time_l103_103791

/-- Walking with 4/5 of my usual speed, I arrive at the bus stop 7 minutes later than normal.
    How many minutes does it take to walk to the bus stop at my usual speed? -/
theorem walk_to_bus_stop_time (S T : ℝ) (h : T > 0) 
  (d_usual : S * T = (4/5) * S * (T + 7)) : 
  T = 28 :=
by
  sorry

end walk_to_bus_stop_time_l103_103791


namespace range_of_a_l103_103221

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) : (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l103_103221


namespace geometric_sequence_n_l103_103390

theorem geometric_sequence_n (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 * a 2 * a 3 = 4) 
  (h2 : a 4 * a 5 * a 6 = 12) 
  (h3 : a (n-1) * a n * a (n+1) = 324) : 
  n = 14 := 
  sorry

end geometric_sequence_n_l103_103390


namespace number_of_classmates_l103_103413

theorem number_of_classmates (total_apples : ℕ) (apples_per_classmate : ℕ) (people_in_class : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) (h3 : people_in_class = total_apples / apples_per_classmate) : 
  people_in_class = 3 :=
by sorry

end number_of_classmates_l103_103413


namespace extreme_points_range_of_a_l103_103410

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

-- Problem 1: Extreme points
theorem extreme_points (a : ℝ) : 
  (a ≤ 0 → ∃! x, ∀ y, f y a ≤ f x a) ∧
  (0 < a ∧ a < 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) ∧
  (a = 1/2 → ∀ x y, f y a ≤ f x a → x = y) ∧
  (a > 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) :=
sorry

-- Problem 2: Range of values for 'a'
theorem range_of_a (a : ℝ) : 
  (∀ x, f x a + Real.exp x ≥ x^3 + x) ↔ (a ≤ Real.exp 1 - 2) :=
sorry

end extreme_points_range_of_a_l103_103410


namespace time_after_2021_hours_l103_103890

-- Definition of starting time and day
def start_time : Nat := 20 * 60 + 21  -- converting 20:21 to minutes
def hours_per_day : Nat := 24
def minutes_per_hour : Nat := 60
def days_per_week : Nat := 7

-- Define the main statement
theorem time_after_2021_hours :
  let total_minutes := 2021 * minutes_per_hour
  let total_days := total_minutes / (hours_per_day * minutes_per_hour)
  let remaining_minutes := total_minutes % (hours_per_day * minutes_per_hour)
  let final_minutes := start_time + remaining_minutes
  let final_day := (total_days + 1) % days_per_week -- start on Monday (0), hence +1 for Tuesday
  final_minutes / minutes_per_hour = 1 ∧ final_minutes % minutes_per_hour = 21 ∧ final_day = 2 :=
by
  sorry

end time_after_2021_hours_l103_103890


namespace calculate_expr_equals_243_l103_103658

theorem calculate_expr_equals_243 :
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by
  sorry

end calculate_expr_equals_243_l103_103658


namespace sandy_money_l103_103656

theorem sandy_money (x : ℝ) (h : 0.70 * x = 210) : x = 300 := by
sorry

end sandy_money_l103_103656


namespace spinner_probability_l103_103896

theorem spinner_probability :
  let p_A := (1 / 4)
  let p_B := (1 / 3)
  let p_C := (5 / 12)
  let p_D := 1 - (p_A + p_B + p_C)
  p_D = 0 :=
by
  sorry

end spinner_probability_l103_103896


namespace sum_series_eq_three_l103_103951

theorem sum_series_eq_three :
  (∑' k : ℕ, (9^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
by 
  sorry

end sum_series_eq_three_l103_103951


namespace annie_diorama_time_l103_103635

theorem annie_diorama_time (P B : ℕ) (h1 : B = 3 * P - 5) (h2 : B = 49) : P + B = 67 :=
sorry

end annie_diorama_time_l103_103635


namespace value_of_a_l103_103553

theorem value_of_a
    (a b : ℝ)
    (h₁ : 0 < a ∧ 0 < b)
    (h₂ : a + b = 1)
    (h₃ : 21 * a^5 * b^2 = 35 * a^4 * b^3) :
    a = 5 / 8 :=
by
  sorry

end value_of_a_l103_103553


namespace plant_cost_and_max_green_lily_students_l103_103882

-- Given conditions
def two_green_lily_three_spider_plants_cost (x y : ℕ) : Prop :=
  2 * x + 3 * y = 36

def one_green_lily_two_spider_plants_cost (x y : ℕ) : Prop :=
  x + 2 * y = 21

def total_students := 48

def cost_constraint (x y m : ℕ) : Prop :=
  9 * m + 6 * (48 - m) ≤ 378

-- Prove that x = 9, y = 6 and m ≤ 30
theorem plant_cost_and_max_green_lily_students :
  ∃ x y m : ℕ, two_green_lily_three_spider_plants_cost x y ∧ 
               one_green_lily_two_spider_plants_cost x y ∧ 
               cost_constraint x y m ∧ 
               x = 9 ∧ y = 6 ∧ m ≤ 30 :=
by
  sorry

end plant_cost_and_max_green_lily_students_l103_103882


namespace compute_custom_op_l103_103173

def custom_op (x y : ℤ) : ℤ := 
  x * y - y * x - 3 * x + 2 * y

theorem compute_custom_op : (custom_op 9 5) - (custom_op 5 9) = -20 := 
by
  sorry

end compute_custom_op_l103_103173


namespace base8_perfect_square_b_zero_l103_103044

-- Define the base 8 representation and the perfect square condition
def base8_to_decimal (a b : ℕ) : ℕ := 512 * a + 64 + 8 * b + 4

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating that if the number in base 8 is a perfect square, then b = 0
theorem base8_perfect_square_b_zero (a b : ℕ) (h₀ : a ≠ 0) 
  (h₁ : is_perfect_square (base8_to_decimal a b)) : b = 0 :=
sorry

end base8_perfect_square_b_zero_l103_103044


namespace arithmetic_square_root_of_16_l103_103949

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x^2 = 16 ∧ x > 0 ∧ x = 4 :=
by
  sorry

end arithmetic_square_root_of_16_l103_103949


namespace houston_firewood_l103_103877

theorem houston_firewood (k e h : ℕ) (k_collected : k = 10) (e_collected : e = 13) (total_collected : k + e + h = 35) : h = 12 :=
by
  sorry

end houston_firewood_l103_103877


namespace Jeanine_gave_fraction_of_pencils_l103_103042

theorem Jeanine_gave_fraction_of_pencils
  (Jeanine_initial_pencils Clare_initial_pencils Jeanine_pencils_after Clare_pencils_after : ℕ)
  (h1 : Jeanine_initial_pencils = 18)
  (h2 : Clare_initial_pencils = Jeanine_initial_pencils / 2)
  (h3 : Jeanine_pencils_after = Clare_pencils_after + 3)
  (h4 : Clare_pencils_after = Clare_initial_pencils)
  (h5 : Jeanine_pencils_after + (Jeanine_initial_pencils - Jeanine_pencils_after) = Jeanine_initial_pencils) :
  (Jeanine_initial_pencils - Jeanine_pencils_after) / Jeanine_initial_pencils = 1 / 3 :=
by
  -- Proof here
  sorry

end Jeanine_gave_fraction_of_pencils_l103_103042


namespace triangle_area_l103_103398

theorem triangle_area :
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  area = 3 := by
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  sorry

end triangle_area_l103_103398


namespace solution_set_of_inequality_l103_103442

variable (a b c : ℝ)

theorem solution_set_of_inequality 
  (h1 : a < 0)
  (h2 : b = a)
  (h3 : c = -2 * a)
  (h4 : ∀ x : ℝ, -2 < x ∧ x < 1 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, (x ≤ -1 / 2 ∨ x ≥ 1) ↔ cx^2 + ax + b ≥ 0 :=
sorry

end solution_set_of_inequality_l103_103442


namespace product_increase_2022_l103_103928

theorem product_increase_2022 (a b c : ℕ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 678) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 :=
by {
  -- The proof would go here, but it's not required per the instructions.
  sorry
}

end product_increase_2022_l103_103928


namespace min_correct_answers_for_score_above_60_l103_103857

theorem min_correct_answers_for_score_above_60 :
  ∃ (x : ℕ), 6 * x - 2 * (15 - x) > 60 ∧ x = 12 :=
by
  sorry

end min_correct_answers_for_score_above_60_l103_103857


namespace animal_fish_consumption_l103_103540

-- Definitions for the daily consumption of each animal
def daily_trout_polar1 := 0.2
def daily_salmon_polar1 := 0.4

def daily_trout_polar2 := 0.3
def daily_salmon_polar2 := 0.5

def daily_trout_polar3 := 0.25
def daily_salmon_polar3 := 0.45

def daily_trout_sealion1 := 0.1
def daily_salmon_sealion1 := 0.15

def daily_trout_sealion2 := 0.2
def daily_salmon_sealion2 := 0.25

-- Calculate total daily consumption
def total_daily_trout :=
  daily_trout_polar1 + daily_trout_polar2 + daily_trout_polar3 + daily_trout_sealion1 + daily_trout_sealion2

def total_daily_salmon :=
  daily_salmon_polar1 + daily_salmon_polar2 + daily_salmon_polar3 + daily_salmon_sealion1 + daily_salmon_sealion2

-- Calculate total monthly consumption
def total_monthly_trout := total_daily_trout * 30
def total_monthly_salmon := total_daily_salmon * 30

-- Total monthly fish bucket consumption
def total_monthly_fish := total_monthly_trout + total_monthly_salmon

-- The statement to prove the total consumption
theorem animal_fish_consumption : total_monthly_fish = 84 := by
  sorry

end animal_fish_consumption_l103_103540


namespace sum_of_fractions_l103_103211

theorem sum_of_fractions : 
  (2/100) + (5/1000) + (5/10000) + 3 * (4/1000) = 0.0375 := 
by 
  sorry

end sum_of_fractions_l103_103211


namespace max_area_of_rectangle_l103_103119

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l103_103119


namespace remainder_of_70_div_17_l103_103440

theorem remainder_of_70_div_17 : 70 % 17 = 2 :=
by
  sorry

end remainder_of_70_div_17_l103_103440


namespace equilateral_triangle_perimeter_l103_103503

theorem equilateral_triangle_perimeter (s : ℕ) (h1 : 2 * s + 10 = 50) : 3 * s = 60 :=
sorry

end equilateral_triangle_perimeter_l103_103503


namespace sandy_money_l103_103003

theorem sandy_money (X : ℝ) (h1 : 0.70 * X = 224) : X = 320 := 
by {
  sorry
}

end sandy_money_l103_103003


namespace increase_by_multiplication_l103_103009

theorem increase_by_multiplication (n : ℕ) (h : n = 14) : (15 * n) - n = 196 :=
by
  -- Skip the proof
  sorry

end increase_by_multiplication_l103_103009


namespace sum_of_coefficients_l103_103329

noncomputable def polynomial_eq (x : ℝ) : ℝ := 1 + x^5
noncomputable def linear_combination (a0 a1 a2 a3 a4 a5 x : ℝ) : ℝ :=
  a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5

theorem sum_of_coefficients (a0 a1 a2 a3 a4 a5 : ℝ) :
  polynomial_eq 1 = linear_combination a0 a1 a2 a3 a4 a5 1 →
  polynomial_eq 2 = linear_combination a0 a1 a2 a3 a4 a5 2 →
  a0 = 2 →
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  intros h1 h2 h3
  sorry

end sum_of_coefficients_l103_103329


namespace min_value_is_nine_l103_103940

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end min_value_is_nine_l103_103940


namespace stratified_sampling_city_B_l103_103609

theorem stratified_sampling_city_B (sales_points_A : ℕ) (sales_points_B : ℕ) (sales_points_C : ℕ) (total_sales_points : ℕ) (sample_size : ℕ)
(h_total : total_sales_points = 450)
(h_sample : sample_size = 90)
(h_sales_points_A : sales_points_A = 180)
(h_sales_points_B : sales_points_B = 150)
(h_sales_points_C : sales_points_C = 120) :
  (sample_size * sales_points_B / total_sales_points) = 30 := 
by
  sorry

end stratified_sampling_city_B_l103_103609


namespace cylinder_height_l103_103697

theorem cylinder_height
  (V : ℝ → ℝ → ℝ) 
  (π : ℝ)
  (r h : ℝ)
  (vol_increase_height : ℝ)
  (vol_increase_radius : ℝ)
  (h_increase : ℝ)
  (r_increase : ℝ)
  (original_radius : ℝ) :
  V r h = π * r^2 * h → 
  vol_increase_height = π * r^2 * h_increase →
  vol_increase_radius = π * ((r + r_increase)^2 - r^2) * h →
  r = original_radius →
  vol_increase_height = 72 * π →
  vol_increase_radius = 72 * π →
  original_radius = 3 →
  r_increase = 2 →
  h_increase = 2 →
  h = 4.5 :=
by
  sorry

end cylinder_height_l103_103697


namespace trajectory_equation_l103_103562

variable (m x y : ℝ)
def a := (m * x, y + 1)
def b := (x, y - 1)
def is_perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

theorem trajectory_equation 
  (h1: is_perpendicular (a m x y) (b x y)) : 
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l103_103562


namespace average_age_group_l103_103163

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = 15 * n) (h2 : T + 37 = 17 * (n + 1)) : n = 10 :=
by
  sorry

end average_age_group_l103_103163


namespace no_rational_solution_of_odd_quadratic_l103_103820

theorem no_rational_solution_of_odd_quadratic (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ x : ℚ, a * x^2 + b * x + c = 0 :=
sorry

end no_rational_solution_of_odd_quadratic_l103_103820


namespace Sarah_total_weeds_l103_103477

noncomputable def Tuesday_weeds : ℕ := 25
noncomputable def Wednesday_weeds : ℕ := 3 * Tuesday_weeds
noncomputable def Thursday_weeds : ℕ := (1 / 5) * Tuesday_weeds
noncomputable def Friday_weeds : ℕ := (3 / 4) * Tuesday_weeds - 10

noncomputable def Total_weeds : ℕ := Tuesday_weeds + Wednesday_weeds + Thursday_weeds + Friday_weeds

theorem Sarah_total_weeds : Total_weeds = 113 := by
  sorry

end Sarah_total_weeds_l103_103477


namespace eval_power_expr_of_196_l103_103098

theorem eval_power_expr_of_196 (a b : ℕ) (ha : 2^a ∣ 196 ∧ ¬ 2^(a + 1) ∣ 196) (hb : 7^b ∣ 196 ∧ ¬ 7^(b + 1) ∣ 196) :
  (1 / 7 : ℝ)^(b - a) = 1 := by
  have ha_val : a = 2 := sorry
  have hb_val : b = 2 := sorry
  rw [ha_val, hb_val]
  simp

end eval_power_expr_of_196_l103_103098


namespace cos_value_l103_103967

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 4 - α) = 1 / 3) :
  Real.cos (Real.pi / 4 + α) = 1 / 3 :=
sorry

end cos_value_l103_103967


namespace average_salary_l103_103022

theorem average_salary (A B C D E : ℕ) (hA : A = 8000) (hB : B = 5000) (hC : C = 14000) (hD : D = 7000) (hE : E = 9000) :
  (A + B + C + D + E) / 5 = 8800 :=
by
  -- the proof will be inserted here
  sorry

end average_salary_l103_103022


namespace solve_quadratic_problem_l103_103431

theorem solve_quadratic_problem :
  ∀ x : ℝ, (x^2 + 6 * x + 8 = -(x + 4) * (x + 7)) ↔ (x = -4 ∨ x = -4.5) := by
  sorry

end solve_quadratic_problem_l103_103431


namespace perpendicular_condition_l103_103571

noncomputable def line := ℝ → (ℝ × ℝ × ℝ)
noncomputable def plane := (ℝ × ℝ × ℝ) → Prop

variable {l m : line}
variable {α : plane}

-- l and m are two different lines
axiom lines_are_different : l ≠ m

-- m is parallel to the plane α
axiom m_parallel_alpha : ∀ t : ℝ, α (m t)

-- Prove that l perpendicular to α is a sufficient but not necessary condition for l perpendicular to m
theorem perpendicular_condition :
  (∀ t : ℝ, ¬ α (l t)) → (∀ t₁ t₂ : ℝ, (l t₁) ≠ (m t₂)) ∧ ¬ (∀ t : ℝ, ¬ α (l t)) :=
by 
  sorry

end perpendicular_condition_l103_103571


namespace exists_digit_combination_l103_103380

theorem exists_digit_combination (d1 d2 d3 d4 : ℕ) (H1 : 42 * (d1 * 10 + 8) = 2 * 1000 + d2 * 100 + d3 * 10 + d4) (H2: ∃ n: ℕ, n = 2 + d2 + d3 + d4 ∧ n % 2 = 1):
  d1 = 4 ∧ 42 * 48 = 2016 ∨ d1 = 6 ∧ 42 * 68 = 2856 :=
sorry

end exists_digit_combination_l103_103380


namespace ratio_c_to_d_l103_103191

theorem ratio_c_to_d (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : b / c = 7 / 9) 
  (h3 : a / d = 0.4166666666666667) : 
  c / d = 5 / 7 := 
by
  -- Proof not needed
  sorry

end ratio_c_to_d_l103_103191


namespace find_four_letter_list_with_equal_product_l103_103222

open Nat

theorem find_four_letter_list_with_equal_product :
  ∃ (L T M W : ℕ), 
  (L * T * M * W = 23 * 24 * 25 * 26) 
  ∧ (1 ≤ L ∧ L ≤ 26) ∧ (1 ≤ T ∧ T ≤ 26) ∧ (1 ≤ M ∧ M ≤ 26) ∧ (1 ≤ W ∧ W ≤ 26) 
  ∧ (L ≠ T) ∧ (T ≠ M) ∧ (M ≠ W) ∧ (W ≠ L) ∧ (L ≠ M) ∧ (T ≠ W)
  ∧ (L * T * M * W) = (12 * 20 * 13 * 23) :=
by
  sorry

end find_four_letter_list_with_equal_product_l103_103222


namespace bowl_capacity_percentage_l103_103441

theorem bowl_capacity_percentage
    (initial_half_full : ℕ)
    (added_water : ℕ)
    (total_water : ℕ)
    (full_capacity : ℕ)
    (percentage_filled : ℚ) :
    initial_half_full * 2 = full_capacity →
    initial_half_full + added_water = total_water →
    added_water = 4 →
    total_water = 14 →
    percentage_filled = (total_water * 100) / full_capacity →
    percentage_filled = 70 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end bowl_capacity_percentage_l103_103441


namespace find_M_plus_N_l103_103241

theorem find_M_plus_N (M N : ℕ) (h1 : 3 / 5 = M / 30) (h2 : 3 / 5 = 90 / N) : M + N = 168 := 
by
  sorry

end find_M_plus_N_l103_103241


namespace payment_plan_months_l103_103174

theorem payment_plan_months 
  (M T : ℝ) (r : ℝ) 
  (hM : M = 100)
  (hT : T = 1320)
  (hr : r = 0.10)
  : ∃ t : ℕ, t = 12 ∧ T = (M * t) + (M * t * r) :=
by
  sorry

end payment_plan_months_l103_103174


namespace identify_letter_X_l103_103060

-- Define the conditions
def date_behind_D (z : ℕ) : ℕ := z
def date_behind_E (z : ℕ) : ℕ := z + 1
def date_behind_F (z : ℕ) : ℕ := z + 14

-- Define the sum condition
def sum_date_E_F (z : ℕ) : ℕ := date_behind_E z + date_behind_F z

-- Define the target date behind another letter
def target_date_behind_another_letter (z : ℕ) : ℕ := z + 15

-- Theorem statement
theorem identify_letter_X (z : ℕ) :
  ∃ (x : Char), sum_date_E_F z = date_behind_D z + target_date_behind_another_letter z → x = 'X' :=
by
  -- The actual proof would go here; we'll defer it for now
  sorry

end identify_letter_X_l103_103060


namespace find_xnp_l103_103725

theorem find_xnp (x n p : ℕ) (h1 : 0 < x) (h2 : 0 < n) (h3 : Nat.Prime p) 
                  (h4 : 2 * x^3 + x^2 + 10 * x + 5 = 2 * p^n) : x + n + p = 6 :=
by
  sorry

end find_xnp_l103_103725


namespace select_test_point_l103_103359

theorem select_test_point (x1 x2 : ℝ) (h1 : x1 = 2 + 0.618 * (4 - 2)) (h2 : x2 = 2 + 4 - x1) :
  (x1 > x2 → x3 = 4 - 0.618 * (4 - x1)) ∨ (x1 < x2 → x3 = 6 - x3) :=
  sorry

end select_test_point_l103_103359


namespace intersection_of_sets_l103_103923

theorem intersection_of_sets:
  let A := {-2, -1, 0, 1}
  let B := {x : ℤ | x^3 + 1 ≤ 0 }
  A ∩ B = {-2, -1} :=
by
  sorry

end intersection_of_sets_l103_103923


namespace cost_price_of_computer_table_l103_103701

theorem cost_price_of_computer_table (CP SP : ℝ) 
  (h1 : SP = CP * 1.15) 
  (h2 : SP = 5750) 
  : CP = 5000 := 
by 
  sorry

end cost_price_of_computer_table_l103_103701


namespace variance_binom_4_half_l103_103186

-- Define the binomial variance function
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Define the conditions
def n := 4
def p := 1 / 2

-- The target statement
theorem variance_binom_4_half : binomial_variance n p = 1 :=
by
  -- The proof goes here
  sorry

end variance_binom_4_half_l103_103186


namespace true_discount_different_time_l103_103341

theorem true_discount_different_time (FV TD_initial TD_different : ℝ) (r : ℝ) (initial_time different_time : ℝ) 
  (h1 : r = initial_time / different_time)
  (h2 : FV = 110)
  (h3 : TD_initial = 10)
  (h4 : initial_time / different_time = 1 / 2) :
  TD_different = 2 * TD_initial :=
by
  sorry

end true_discount_different_time_l103_103341


namespace range_of_a_l103_103190

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := { x | x^2 - 2 * x + a ≥ 0 }

theorem range_of_a (h : 1 ∉ set_A a) : a < 1 := 
by {
  sorry
}

end range_of_a_l103_103190


namespace sector_area_eq_13pi_l103_103270

theorem sector_area_eq_13pi
    (O A B C : Type)
    (r : ℝ)
    (θ : ℝ)
    (h1 : θ = 130)
    (h2 : r = 6) :
    (θ / 360) * (π * r^2) = 13 * π := by
  sorry

end sector_area_eq_13pi_l103_103270


namespace graph_not_pass_first_quadrant_l103_103491

theorem graph_not_pass_first_quadrant (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ¬ (∃ x y : ℝ, y = a^x + b ∧ x > 0 ∧ y > 0) :=
sorry

end graph_not_pass_first_quadrant_l103_103491


namespace bukvinsk_acquaintances_l103_103472

theorem bukvinsk_acquaintances (Martin Klim Inna Tamara Kamilla : Type) 
  (acquaints : Type → Type → Prop)
  (exists_same_letters : ∀ (x y : Type), acquaints x y ↔ ∃ S, (x = S ∧ y = S)) :
  (∃ (count_Martin : ℕ), count_Martin = 20) →
  (∃ (count_Klim : ℕ), count_Klim = 15) →
  (∃ (count_Inna : ℕ), count_Inna = 12) →
  (∃ (count_Tamara : ℕ), count_Tamara = 12) →
  (∃ (count_Kamilla : ℕ), count_Kamilla = 15) := by
  sorry

end bukvinsk_acquaintances_l103_103472


namespace range_of_s_l103_103290

def double_value_point (s t : ℝ) (ht : t ≠ -1) :
  Prop := 
  ∀ k : ℝ, (t + 1) * k^2 + t * k + s = 0 →
  (t^2 - 4 * s * (t + 1) > 0)

theorem range_of_s (s t : ℝ) (ht : t ≠ -1) :
  double_value_point s t ht ↔ -1 < s ∧ s < 0 :=
sorry

end range_of_s_l103_103290


namespace abscissa_of_tangent_point_is_2_l103_103487

noncomputable def f (x : ℝ) : ℝ := (x^2) / 4 - 3 * Real.log x

noncomputable def f' (x : ℝ) : ℝ := (1/2) * x - 3 / x

theorem abscissa_of_tangent_point_is_2 : 
  ∃ x0 : ℝ, f' x0 = -1/2 ∧ x0 = 2 :=
by
  sorry

end abscissa_of_tangent_point_is_2_l103_103487


namespace least_number_subtracted_l103_103471

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_pos : 0 < x) (h_init : n = 427398) (h_div : ∃ k : ℕ, (n - x) = 14 * k) : x = 6 :=
sorry

end least_number_subtracted_l103_103471


namespace qualified_flour_l103_103557

-- Define the acceptable weight range
def acceptable_range (w : ℝ) : Prop :=
  24.75 ≤ w ∧ w ≤ 25.25

-- Define the weight options
def optionA : ℝ := 24.70
def optionB : ℝ := 24.80
def optionC : ℝ := 25.30
def optionD : ℝ := 25.51

-- The statement to be proved
theorem qualified_flour : acceptable_range optionB ∧ ¬acceptable_range optionA ∧ ¬acceptable_range optionC ∧ ¬acceptable_range optionD :=
by
  sorry

end qualified_flour_l103_103557


namespace number_of_hexagons_l103_103692

-- Definitions based on conditions
def num_pentagons : ℕ := 12

-- Based on the problem statement, the goal is to prove that the number of hexagons is 20
theorem number_of_hexagons (h : num_pentagons = 12) : ∃ (num_hexagons : ℕ), num_hexagons = 20 :=
by {
  -- proof would be here
  sorry
}

end number_of_hexagons_l103_103692


namespace kamal_marks_physics_l103_103936

-- Define the marks in subjects
def marks_english := 66
def marks_mathematics := 65
def marks_chemistry := 62
def marks_biology := 75
def average_marks := 69
def number_of_subjects := 5

-- Calculate the total marks from the average
def total_marks := average_marks * number_of_subjects

-- Calculate the known total marks
def known_total_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology

-- Define Kamal's marks in Physics
def marks_physics := total_marks - known_total_marks

-- Prove the marks in Physics are 77
theorem kamal_marks_physics : marks_physics = 77 := by
  sorry

end kamal_marks_physics_l103_103936


namespace month_length_l103_103145

def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def total_cost : ℝ := 6

theorem month_length : (total_cost / cost_per_treat) / treats_per_day = 30 := by
  sorry

end month_length_l103_103145


namespace shaded_area_of_logo_l103_103757

theorem shaded_area_of_logo 
  (side_length_of_square : ℝ)
  (side_length_of_square_eq : side_length_of_square = 30)
  (radius_of_circle : ℝ)
  (radius_eq : radius_of_circle = side_length_of_square / 4)
  (number_of_circles : ℕ)
  (number_of_circles_eq : number_of_circles = 4)
  : (side_length_of_square^2) - (number_of_circles * Real.pi * (radius_of_circle^2)) = 900 - 225 * Real.pi := by
    sorry

end shaded_area_of_logo_l103_103757


namespace remainder_of_2x_plus_3uy_l103_103237

theorem remainder_of_2x_plus_3uy (x y u v : ℤ) (hxy : x = u * y + v) (hv : 0 ≤ v) (hv_ub : v < y) :
  (if 2 * v < y then (2 * v % y) else ((2 * v % y) % -y % y)) = 
  (if 2 * v < y then 2 * v else 2 * v - y) :=
by {
  sorry
}

end remainder_of_2x_plus_3uy_l103_103237


namespace negation_correct_l103_103202

variable (x : Real)

def original_proposition : Prop :=
  x > 0 → x^2 > 0

def negation_proposition : Prop :=
  x ≤ 0 → x^2 ≤ 0

theorem negation_correct :
  ¬ original_proposition x = negation_proposition x :=
by 
  sorry

end negation_correct_l103_103202


namespace expected_hit_targets_correct_expected_hit_targets_at_least_half_l103_103914

noncomputable def expected_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n)^n)

theorem expected_hit_targets_correct (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n = n * (1 - (1 - (1 : ℝ) / n)^n) :=
by
  unfold expected_hit_targets
  sorry

theorem expected_hit_targets_at_least_half (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n >= n / 2 :=
by
  unfold expected_hit_targets
  sorry

end expected_hit_targets_correct_expected_hit_targets_at_least_half_l103_103914


namespace probability_is_7_over_26_l103_103251

section VowelProbability

def num_students : Nat := 26

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U' || c = 'Y' || c = 'W'

def num_vowels : Nat := 7

def probability_of_vowel_initials : Rat :=
  (num_vowels : Nat) / (num_students : Nat)

theorem probability_is_7_over_26 :
  probability_of_vowel_initials = 7 / 26 := by
  sorry

end VowelProbability

end probability_is_7_over_26_l103_103251


namespace first_train_speed_is_80_kmph_l103_103769

noncomputable def speedOfFirstTrain
  (lenFirstTrain : ℝ)
  (lenSecondTrain : ℝ)
  (speedSecondTrain : ℝ)
  (clearTime : ℝ)
  (oppositeDirections : Bool) : ℝ :=
  if oppositeDirections then
    let totalDistance := (lenFirstTrain + lenSecondTrain) / 1000  -- convert meters to kilometers
    let timeHours := clearTime / 3600 -- convert seconds to hours
    let relativeSpeed := totalDistance / timeHours
    relativeSpeed - speedSecondTrain
  else
    0 -- This should not happen based on problem conditions

theorem first_train_speed_is_80_kmph :
  speedOfFirstTrain 151 165 65 7.844889650207294 true = 80 :=
by
  sorry

end first_train_speed_is_80_kmph_l103_103769


namespace trapezoid_area_l103_103664

theorem trapezoid_area (x : ℝ) :
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  area = 9 * x^2 / 2 :=
by
  -- Definitions based on conditions
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  -- Proof of the theorem, currently omitted
  sorry

end trapezoid_area_l103_103664


namespace car_speed_l103_103381

-- Define the given conditions
def distance := 800 -- in kilometers
def time := 5 -- in hours

-- Define the speed calculation
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- State the theorem to be proved
theorem car_speed : speed distance time = 160 := by
  -- proof would go here
  sorry

end car_speed_l103_103381


namespace smallest_value_of_a_l103_103860

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l103_103860


namespace fg_of_2_l103_103946

def g (x : ℝ) : ℝ := 2 * x^2
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 15 :=
by
  have h1 : g 2 = 8 := by sorry
  have h2 : f 8 = 15 := by sorry
  rw [h1]
  exact h2

end fg_of_2_l103_103946


namespace polynomial_decomposition_l103_103612

theorem polynomial_decomposition :
  (x^3 - 2*x^2 + 3*x + 5) = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 :=
by sorry

end polynomial_decomposition_l103_103612


namespace abs_quadratic_bound_l103_103798

theorem abs_quadratic_bound (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + a * x + b) :
  (|f 1| ≥ (1 / 2)) ∨ (|f 2| ≥ (1 / 2)) ∨ (|f 3| ≥ (1 / 2)) :=
by
  sorry

end abs_quadratic_bound_l103_103798


namespace initial_observations_l103_103101

theorem initial_observations {n : ℕ} (S : ℕ) (new_observation : ℕ) 
  (h1 : S = 15 * n) (h2 : new_observation = 14 - n)
  (h3 : (S + new_observation) / (n + 1) = 14) : n = 6 :=
sorry

end initial_observations_l103_103101


namespace ab_minus_c_eq_six_l103_103280

theorem ab_minus_c_eq_six (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : 
  a * b - c = 6 := 
by
  sorry

end ab_minus_c_eq_six_l103_103280


namespace child_height_at_age_10_l103_103931

theorem child_height_at_age_10 (x y : ℝ) (h : y = 7.19 * x + 73.93) (hx : x = 10) : abs (y - 145.83) < 1 :=
by {
  sorry
}

end child_height_at_age_10_l103_103931


namespace correct_relative_pronoun_used_l103_103732

theorem correct_relative_pronoun_used (option : String) :
  (option = "where") ↔
  "Giving is a universal opportunity " ++ option ++ " regardless of your age, profession, religion, and background, you have the capacity to create change." =
  "Giving is a universal opportunity where regardless of your age, profession, religion, and background, you have the capacity to create change." :=
by
  sorry

end correct_relative_pronoun_used_l103_103732


namespace cubic_eq_has_real_roots_l103_103028

theorem cubic_eq_has_real_roots (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end cubic_eq_has_real_roots_l103_103028


namespace number_of_people_l103_103608

theorem number_of_people (n k : ℕ) (h₁ : k * n * (n - 1) = 440) : n = 11 :=
sorry

end number_of_people_l103_103608


namespace relation_among_a_b_c_l103_103288

open Real

theorem relation_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = log 3 / log 2)
  (h2 : b = log 7 / (2 * log 2))
  (h3 : c = 0.7 ^ 4) :
  a > b ∧ b > c :=
by
  -- we leave the proof as an exercise
  sorry

end relation_among_a_b_c_l103_103288


namespace triangle_area_division_l103_103372

theorem triangle_area_division (T T_1 T_2 T_3 : ℝ) 
  (hT1_pos : 0 < T_1) (hT2_pos : 0 < T_2) (hT3_pos : 0 < T_3) (hT : T = T_1 + T_2 + T_3) :
  T = (Real.sqrt T_1 + Real.sqrt T_2 + Real.sqrt T_3) ^ 2 :=
sorry

end triangle_area_division_l103_103372


namespace rational_function_solution_l103_103250

theorem rational_function_solution (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g (1 / x) + 3 * g x / x = x^3) :
  g (-3) = 135 / 4 := 
sorry

end rational_function_solution_l103_103250


namespace exists_x_abs_ge_one_fourth_l103_103096

theorem exists_x_abs_ge_one_fourth :
  ∀ (a b c : ℝ), ∃ x : ℝ, |x| ≤ 1 ∧ |x^3 + a * x^2 + b * x + c| ≥ 1 / 4 :=
by sorry

end exists_x_abs_ge_one_fourth_l103_103096


namespace julieta_total_spent_l103_103499

theorem julieta_total_spent (original_backpack_price : ℕ)
                            (original_ringbinder_price : ℕ)
                            (backpack_price_increase : ℕ)
                            (ringbinder_price_decrease : ℕ)
                            (number_of_ringbinders : ℕ)
                            (new_backpack_price : ℕ)
                            (new_ringbinder_price : ℕ)
                            (total_ringbinder_cost : ℕ)
                            (total_spent : ℕ) :
  original_backpack_price = 50 →
  original_ringbinder_price = 20 →
  backpack_price_increase = 5 →
  ringbinder_price_decrease = 2 →
  number_of_ringbinders = 3 →
  new_backpack_price = original_backpack_price + backpack_price_increase →
  new_ringbinder_price = original_ringbinder_price - ringbinder_price_decrease →
  total_ringbinder_cost = new_ringbinder_price * number_of_ringbinders →
  total_spent = new_backpack_price + total_ringbinder_cost →
  total_spent = 109 := by
  intros
  sorry

end julieta_total_spent_l103_103499


namespace total_weight_of_new_people_l103_103502

theorem total_weight_of_new_people (W W_new : ℝ) :
  (∀ (old_weights : List ℝ), old_weights.length = 25 →
    ((old_weights.sum - (65 + 70 + 75)) + W_new = old_weights.sum + (4 * 25)) →
    W_new = 310) := by
  intros old_weights old_weights_length increase_condition
  -- Proof will be here
  sorry

end total_weight_of_new_people_l103_103502


namespace john_brown_bags_l103_103412

theorem john_brown_bags :
  (∃ b : ℕ, 
     let total_macaroons := 12
     let weight_per_macaroon := 5
     let total_weight := total_macaroons * weight_per_macaroon
     let remaining_weight := 45
     let bag_weight := total_weight - remaining_weight
     let macaroons_per_bag := bag_weight / weight_per_macaroon
     total_macaroons / macaroons_per_bag = b
  ) → b = 4 :=
by
  sorry

end john_brown_bags_l103_103412


namespace total_photos_in_gallery_l103_103136

def initial_photos : ℕ := 800
def photos_first_day : ℕ := (2 * initial_photos) / 3
def photos_second_day : ℕ := photos_first_day + 180

theorem total_photos_in_gallery : initial_photos + photos_first_day + photos_second_day = 2046 := by
  -- the proof can be provided here
  sorry

end total_photos_in_gallery_l103_103136


namespace convert_yahs_to_bahs_l103_103133

noncomputable section

def bahs_to_rahs (bahs : ℕ) : ℕ := bahs * (36/24)
def rahs_to_bahs (rahs : ℕ) : ℕ := rahs * (24/36)
def rahs_to_yahs (rahs : ℕ) : ℕ := rahs * (18/12)
def yahs_to_rahs (yahs : ℕ) : ℕ := yahs * (12/18)
def yahs_to_bahs (yahs : ℕ) : ℕ := rahs_to_bahs (yahs_to_rahs yahs)

theorem convert_yahs_to_bahs :
  yahs_to_bahs 1500 = 667 :=
sorry

end convert_yahs_to_bahs_l103_103133


namespace find_dimensions_l103_103445

-- Define the conditions
def perimeter (x y : ℕ) : Prop := (2 * (x + y) = 3996)
def divisible_parts (x y k : ℕ) : Prop := (x * y = 1998 * k) ∧ ∃ (k : ℕ), (k * 1998 = x * y) ∧ k ≠ 0

-- State the theorem
theorem find_dimensions (x y : ℕ) (k : ℕ) : perimeter x y ∧ divisible_parts x y k → (x = 1332 ∧ y = 666) ∨ (x = 666 ∧ y = 1332) :=
by
  -- This is where the proof would go.
  sorry

end find_dimensions_l103_103445


namespace find_x_for_collinear_vectors_l103_103267

noncomputable def collinear_vectors (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem find_x_for_collinear_vectors : ∀ (x : ℝ), collinear_vectors (2, -3) (x, 6) → x = -4 := by
  intros x h
  sorry

end find_x_for_collinear_vectors_l103_103267


namespace average_price_per_share_l103_103884

-- Define the conditions
def Microtron_price_per_share := 36
def Dynaco_price_per_share := 44
def total_shares := 300
def Dynaco_shares_sold := 150

-- Define the theorem to be proved
theorem average_price_per_share : 
  (Dynaco_shares_sold * Dynaco_price_per_share + (total_shares - Dynaco_shares_sold) * Microtron_price_per_share) / total_shares = 40 :=
by
  -- Skip the actual proof here
  sorry

end average_price_per_share_l103_103884


namespace cost_pants_shirt_l103_103639

variable (P S C : ℝ)

theorem cost_pants_shirt (h1 : P + C = 244) (h2 : C = 5 * S) (h3 : C = 180) : P + S = 100 := by
  sorry

end cost_pants_shirt_l103_103639


namespace proof_problem_l103_103107

noncomputable def f (x a : ℝ) : ℝ := (1 + x^2) * Real.exp x - a
noncomputable def f' (x a : ℝ) : ℝ := (1 + 2 * x + x^2) * Real.exp x
noncomputable def k_OP (a : ℝ) : ℝ := a - 2 / Real.exp 1
noncomputable def g (m : ℝ) : ℝ := Real.exp m - (m + 1)

theorem proof_problem (a m : ℝ) (h₁ : a > 0) (h₂ : f' (-1) a = 0) (h₃ : f' m a = k_OP a) 
  : m + 1 ≤ 3 * a - 2 / Real.exp 1 := by
  sorry

end proof_problem_l103_103107


namespace outlined_square_digit_l103_103996

theorem outlined_square_digit :
  ∀ (digit : ℕ), (digit ∈ {n | ∃ (m : ℕ), 10 ≤ 3^m ∧ 3^m < 1000 ∧ digit = (3^m / 10) % 10 }) →
  (digit ∈ {n | ∃ (n : ℕ), 10 ≤ 7^n ∧ 7^n < 1000 ∧ digit = (7^n / 10) % 10 }) →
  digit = 4 :=
by sorry

end outlined_square_digit_l103_103996


namespace evaluate_expression_l103_103389

theorem evaluate_expression :
  (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 :=
by
  sorry

end evaluate_expression_l103_103389


namespace polar_distance_l103_103083

noncomputable def distance_point (r1 θ1 r2 θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 ^ 2) + (r2 ^ 2) - 2 * r1 * r2 * Real.cos (θ1 - θ2))

theorem polar_distance :
  ∀ (θ1 θ2 : ℝ), (θ1 - θ2 = Real.pi / 2) → distance_point 5 θ1 12 θ2 = 13 :=
by
  intros θ1 θ2 hθ
  rw [distance_point, hθ, Real.cos_pi_div_two]
  norm_num
  sorry

end polar_distance_l103_103083


namespace cos_pi_plus_2alpha_value_l103_103179

theorem cos_pi_plus_2alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : 
    Real.cos (π + 2 * α) = 7 / 9 := sorry

end cos_pi_plus_2alpha_value_l103_103179


namespace abs_value_expression_l103_103495

theorem abs_value_expression (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 :=
sorry

end abs_value_expression_l103_103495


namespace sum_of_all_possible_values_is_correct_l103_103986

noncomputable def M_sum_of_all_possible_values (a b c M : ℝ) : Prop :=
  M = a * b * c ∧ M = 8 * (a + b + c) ∧ c = a + b ∧ b = 2 * a

theorem sum_of_all_possible_values_is_correct :
  ∃ M, (∃ a b c, M_sum_of_all_possible_values a b c M) ∧ M = 96 * Real.sqrt 2 := by
  sorry

end sum_of_all_possible_values_is_correct_l103_103986


namespace intersection_of_M_and_N_l103_103166

-- Definitions from conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof problem statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
sorry

end intersection_of_M_and_N_l103_103166


namespace multiple_of_6_is_multiple_of_3_l103_103228

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) (h1 : ∀ k : ℕ, n = 6 * k)
  : ∃ m : ℕ, n = 3 * m :=
by sorry

end multiple_of_6_is_multiple_of_3_l103_103228


namespace oomyapeck_eyes_count_l103_103460

-- Define the various conditions
def number_of_people : ℕ := 3
def fish_per_person : ℕ := 4
def eyes_per_fish : ℕ := 2
def eyes_given_to_dog : ℕ := 2

-- Compute the total number of fish
def total_fish : ℕ := number_of_people * fish_per_person

-- Compute the total number of eyes from the total number of fish
def total_eyes : ℕ := total_fish * eyes_per_fish

-- Compute the number of eyes Oomyapeck eats
def eyes_eaten_by_oomyapeck : ℕ := total_eyes - eyes_given_to_dog

-- The proof statement
theorem oomyapeck_eyes_count : eyes_eaten_by_oomyapeck = 22 := by
  sorry

end oomyapeck_eyes_count_l103_103460


namespace p_distinct_roots_iff_l103_103090

variables {p : ℝ}

def quadratic_has_distinct_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c) > 0

theorem p_distinct_roots_iff (hp: p > 0 ∨ p = -1) :
  (∀ x : ℝ, x^2 - 2 * |x| - p = 0 → 
    (quadratic_has_distinct_roots 1 (-2) (-p) ∨
      quadratic_has_distinct_roots 1 2 (-p))) :=
by sorry

end p_distinct_roots_iff_l103_103090


namespace simplify_expression_1_simplify_expression_2_l103_103041

-- Statement for the first problem
theorem simplify_expression_1 (a : ℝ) : 2 * a * (a - 3) - a^2 = a^2 - 6 * a := 
by sorry

-- Statement for the second problem
theorem simplify_expression_2 (x : ℝ) : (x - 1) * (x + 2) - x * (x + 1) = -2 := 
by sorry

end simplify_expression_1_simplify_expression_2_l103_103041


namespace arithmetic_seq_ratio_l103_103065

theorem arithmetic_seq_ratio
  (a b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (H_seq_a : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (H_seq_b : ∀ n, T n = (n * (b 1 + b n)) / 2)
  (H_ratio : ∀ n, S n / T n = (2 * n - 3) / (4 * n - 3)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 41 :=
by
  sorry

end arithmetic_seq_ratio_l103_103065


namespace roots_of_equation_l103_103564

theorem roots_of_equation (x : ℝ) : ((x - 5) ^ 2 = 2 * (x - 5)) ↔ (x = 5 ∨ x = 7) := by
sorry

end roots_of_equation_l103_103564


namespace selection_methods_l103_103051

-- Define the number of students and lectures.
def numberOfStudents : Nat := 6
def numberOfLectures : Nat := 5

-- Define the problem as proving the number of selection methods equals 5^6.
theorem selection_methods : (numberOfLectures ^ numberOfStudents) = 15625 := by
  -- Include the proper mathematical equivalence statement
  sorry

end selection_methods_l103_103051


namespace distance_from_center_l103_103456

-- Define the circle equation as a predicate
def isCircle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x - 4 * y + 8

-- Define the center of the circle
def circleCenter : ℝ × ℝ := (1, -2)

-- Define the point in question
def point : ℝ × ℝ := (-3, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the proof problem
theorem distance_from_center :
  ∀ (x y : ℝ), isCircle x y → distance circleCenter point = 2 * Real.sqrt 13 :=
by
  sorry

end distance_from_center_l103_103456


namespace factorization_problem_l103_103425

theorem factorization_problem (a b c : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 7 * x + 12 = (x + a) * (x + b))
  (h2 : ∀ x : ℝ, x^2 - 8 * x - 20 = (x - b) * (x - c)) :
  a - b + c = -9 :=
sorry

end factorization_problem_l103_103425


namespace mary_added_peanuts_l103_103987

theorem mary_added_peanuts (initial final added : Nat) 
  (h1 : initial = 4)
  (h2 : final = 16)
  (h3 : final = initial + added) : 
  added = 12 := 
by {
  sorry
}

end mary_added_peanuts_l103_103987


namespace find_n_l103_103474

theorem find_n (n : ℕ) : 2^(2 * n) + 2^(2 * n) + 2^(2 * n) + 2^(2 * n) = 4^22 → n = 21 :=
by
  sorry

end find_n_l103_103474


namespace sum_of_solutions_l103_103965

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end sum_of_solutions_l103_103965


namespace percent_palindromes_containing_7_l103_103710

theorem percent_palindromes_containing_7 : 
  let num_palindromes := 90
  let num_palindrome_with_7 := 19
  (num_palindrome_with_7 / num_palindromes * 100) = 21.11 := 
by
  sorry

end percent_palindromes_containing_7_l103_103710


namespace floor_e_eq_2_l103_103017

noncomputable def e_approx : ℝ := 2.71828

theorem floor_e_eq_2 : ⌊e_approx⌋ = 2 :=
sorry

end floor_e_eq_2_l103_103017


namespace smallest_n_inequality_l103_103942

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    ∀ m : ℕ, m < n → ¬ (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) :=
by
  sorry

end smallest_n_inequality_l103_103942


namespace gcd_b_n_b_n_plus_1_l103_103843

-- Definitions based on the conditions in the problem
def b_n (n : ℕ) : ℕ := 150 + n^3

theorem gcd_b_n_b_n_plus_1 (n : ℕ) : gcd (b_n n) (b_n (n + 1)) = 1 := by
  -- We acknowledge that we need to skip the proof steps
  sorry

end gcd_b_n_b_n_plus_1_l103_103843


namespace tan_alpha_eq_one_third_l103_103640

variable (α : ℝ)

theorem tan_alpha_eq_one_third (h : Real.tan (α + Real.pi / 4) = 2) : Real.tan α = 1 / 3 :=
sorry

end tan_alpha_eq_one_third_l103_103640


namespace sales_this_month_l103_103630

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end sales_this_month_l103_103630


namespace some_students_are_not_club_members_l103_103631

variable (U : Type) -- U represents the universe of students and club members
variables (Student ClubMember StudyLate : U → Prop)

-- Conditions derived from the problem
axiom h1 : ∃ s, Student s ∧ ¬ StudyLate s -- Some students do not study late
axiom h2 : ∀ c, ClubMember c → StudyLate c -- All club members study late

theorem some_students_are_not_club_members :
  ∃ s, Student s ∧ ¬ ClubMember s :=
by
  sorry

end some_students_are_not_club_members_l103_103631


namespace simplify_fraction_l103_103263

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l103_103263


namespace rowing_distance_upstream_l103_103034

theorem rowing_distance_upstream 
  (v : ℝ) (d : ℝ)
  (h1 : 75 = (v + 3) * 5)
  (h2 : d = (v - 3) * 5) :
  d = 45 :=
by {
  sorry
}

end rowing_distance_upstream_l103_103034


namespace find_f_neg_5_l103_103464

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_domain : ∀ x : ℝ, true)
variable (h_positive : ∀ x : ℝ, x > 0 → f x = log 5 x + 1)

theorem find_f_neg_5 : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l103_103464


namespace find_ordered_triplets_l103_103511

theorem find_ordered_triplets (x y z : ℝ) :
  x^3 = z / y - 2 * y / z ∧
  y^3 = x / z - 2 * z / x ∧
  z^3 = y / x - 2 * x / y →
  (x = 1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) :=
sorry

end find_ordered_triplets_l103_103511


namespace flight_distance_each_way_l103_103019

variables (D : ℝ) (T_out T_return total_time : ℝ)

-- Defining conditions
def condition1 : Prop := T_out = D / 300
def condition2 : Prop := T_return = D / 500
def condition3 : Prop := total_time = 8

-- Given conditions
axiom h1 : condition1 D T_out
axiom h2 : condition2 D T_return
axiom h3 : condition3 total_time

-- The proof problem statement
theorem flight_distance_each_way : T_out + T_return = total_time → D = 1500 :=
by
  sorry

end flight_distance_each_way_l103_103019


namespace nathan_subtracts_79_l103_103590

theorem nathan_subtracts_79 (a b : ℤ) (h₁ : a = 40) (h₂ : b = 1) :
  (a - b) ^ 2 = a ^ 2 - 79 := 
by
  sorry

end nathan_subtracts_79_l103_103590


namespace seven_power_expression_l103_103997

theorem seven_power_expression (x y z : ℝ) (h₀ : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) (h₂ : xy + xz + yz ≠ 0) :
  (x^7 + y^7 + z^7) / (xyz * (x^2 + y^2 + z^2)) = 14 :=
by
  sorry

end seven_power_expression_l103_103997


namespace prove_difference_l103_103132

theorem prove_difference (x y : ℝ) (h1 : x + y = 500) (h2 : x * y = 22000) : y - x = -402.5 :=
sorry

end prove_difference_l103_103132


namespace interior_angle_of_regular_pentagon_is_108_l103_103528

-- Define the sum of angles in a triangle
def sum_of_triangle_angles : ℕ := 180

-- Define the number of triangles in a convex pentagon
def num_of_triangles_in_pentagon : ℕ := 3

-- Define the total number of interior angles in a pentagon
def num_of_angles_in_pentagon : ℕ := 5

-- Define the total sum of the interior angles of a pentagon
def sum_of_pentagon_interior_angles : ℕ := num_of_triangles_in_pentagon * sum_of_triangle_angles

-- Define the degree measure of an interior angle of a regular pentagon
def interior_angle_of_regular_pentagon : ℕ := sum_of_pentagon_interior_angles / num_of_angles_in_pentagon

theorem interior_angle_of_regular_pentagon_is_108 :
  interior_angle_of_regular_pentagon = 108 :=
by
  -- Proof will be filled in here
  sorry

end interior_angle_of_regular_pentagon_is_108_l103_103528


namespace evaluate_expression_l103_103738

theorem evaluate_expression : (528 * 528) - (527 * 529) = 1 := by
  sorry

end evaluate_expression_l103_103738


namespace patio_rows_before_rearrangement_l103_103385

theorem patio_rows_before_rearrangement (r c : ℕ) 
  (h1 : r * c = 160) 
  (h2 : (r + 4) * (c - 2) = 160)
  (h3 : ∃ k : ℕ, 5 * k = r)
  (h4 : ∃ l : ℕ, 5 * l = c) :
  r = 16 :=
by
  sorry

end patio_rows_before_rearrangement_l103_103385


namespace students_behind_Yoongi_l103_103229

theorem students_behind_Yoongi :
  ∀ (n : ℕ), n = 20 → ∀ (j y : ℕ), j = 1 → y = 2 → n - y = 18 :=
by
  intros n h1 j h2 y h3
  sorry

end students_behind_Yoongi_l103_103229


namespace disqualified_team_participants_l103_103035

theorem disqualified_team_participants
  (initial_teams : ℕ) (initial_avg : ℕ) (final_teams : ℕ) (final_avg : ℕ)
  (total_initial : ℕ) (total_final : ℕ) :
  initial_teams = 9 →
  initial_avg = 7 →
  final_teams = 8 →
  final_avg = 6 →
  total_initial = initial_teams * initial_avg →
  total_final = final_teams * final_avg →
  total_initial - total_final = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end disqualified_team_participants_l103_103035


namespace unique_solution_l103_103264

theorem unique_solution (x y a : ℝ) :
  (x^2 + y^2 = 2 * a ∧ x + Real.log (y^2 + 1) / Real.log 2 = a) ↔ a = 0 ∧ x = 0 ∧ y = 0 :=
by
  sorry

end unique_solution_l103_103264


namespace adult_meal_cost_l103_103414

theorem adult_meal_cost (x : ℝ) 
  (total_people : ℕ) (kids : ℕ) (total_cost : ℝ)  
  (h_total_people : total_people = 11) 
  (h_kids : kids = 2) 
  (h_total_cost : total_cost = 72)
  (h_adult_meals : (total_people - kids : ℕ) • x = total_cost) : 
  x = 8 := 
by
  -- Proof will go here
  sorry

end adult_meal_cost_l103_103414


namespace number_of_possible_values_for_c_l103_103863

theorem number_of_possible_values_for_c : 
  (∃ c_values : Finset ℕ, (∀ c ∈ c_values, c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) 
  ∧ c_values.card = 10) :=
sorry

end number_of_possible_values_for_c_l103_103863


namespace calculate_gf3_l103_103577

def f (x : ℕ) : ℕ := x^3 - 1
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem calculate_gf3 : g (f 3) = 2056 := by
  sorry

end calculate_gf3_l103_103577


namespace largest_digit_not_in_odd_units_digits_l103_103064

-- Defining the sets of digits
def odd_units_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_units_digits : Set ℕ := {0, 2, 4, 6, 8}

-- Statement to prove
theorem largest_digit_not_in_odd_units_digits : 
  ∀ n ∈ even_units_digits, n ≤ 8 ∧ (∀ d ∈ odd_units_digits, d < n) → n = 8 :=
by
  sorry

end largest_digit_not_in_odd_units_digits_l103_103064


namespace smallest_r_minus_p_l103_103569

theorem smallest_r_minus_p 
  (p q r : ℕ) (h₀ : p * q * r = 362880) (h₁ : p < q) (h₂ : q < r) : 
  r - p = 126 :=
sorry

end smallest_r_minus_p_l103_103569


namespace min_value_S_l103_103968

noncomputable def S (x y : ℝ) : ℝ := 2 * x ^ 2 - x * y + y ^ 2 + 2 * x + 3 * y

theorem min_value_S : ∃ x y : ℝ, S x y = -4 ∧ ∀ (a b : ℝ), S a b ≥ -4 := 
by
  sorry

end min_value_S_l103_103968


namespace problem_statement_l103_103618

noncomputable def f (x : ℝ) := 2 * x + 3
noncomputable def g (x : ℝ) := 3 * x - 2

theorem problem_statement : (f (g (f 3)) / g (f (g 3))) = 53 / 49 :=
by
  -- The proof is not provided as requested.
  sorry

end problem_statement_l103_103618


namespace Wendy_bouquets_l103_103233

def num_flowers_before : ℕ := 45
def num_wilted_flowers : ℕ := 35
def flowers_per_bouquet : ℕ := 5

theorem Wendy_bouquets : (num_flowers_before - num_wilted_flowers) / flowers_per_bouquet = 2 := by
  sorry

end Wendy_bouquets_l103_103233


namespace sum_of_terms_in_arithmetic_sequence_eq_l103_103253

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms_in_arithmetic_sequence_eq :
  arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 36) →
  (a 3 + a 10 = 18) :=
by
  intros h_seq h_sum
  -- Proof placeholder
  sorry

end sum_of_terms_in_arithmetic_sequence_eq_l103_103253


namespace Yoque_borrowed_150_l103_103761

noncomputable def Yoque_borrowed_amount (X : ℝ) : Prop :=
  1.10 * X = 11 * 15

theorem Yoque_borrowed_150 (X : ℝ) : Yoque_borrowed_amount X → X = 150 :=
by
  -- proof will be filled in
  sorry

end Yoque_borrowed_150_l103_103761


namespace selling_price_per_pound_is_correct_l103_103397

noncomputable def cost_of_40_lbs : ℝ := 40 * 0.38
noncomputable def cost_of_8_lbs : ℝ := 8 * 0.50
noncomputable def total_cost : ℝ := cost_of_40_lbs + cost_of_8_lbs
noncomputable def total_weight : ℝ := 40 + 8
noncomputable def profit : ℝ := total_cost * 0.20
noncomputable def total_selling_price : ℝ := total_cost + profit
noncomputable def selling_price_per_pound : ℝ := total_selling_price / total_weight

theorem selling_price_per_pound_is_correct :
  selling_price_per_pound = 0.48 :=
by
  sorry

end selling_price_per_pound_is_correct_l103_103397


namespace union_sets_l103_103209

noncomputable def M : Set ℤ := {1, 2, 3}
noncomputable def N : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem union_sets : M ∪ N = {0, 1, 2, 3} := by
  sorry

end union_sets_l103_103209


namespace proportion_solve_x_l103_103695

theorem proportion_solve_x :
  (0.75 / x = 5 / 7) → x = 1.05 :=
by
  sorry

end proportion_solve_x_l103_103695


namespace option_C_is_different_l103_103904

def cause_and_effect_relationship (description: String) : Prop :=
  description = "A: Great teachers produce outstanding students" ∨
  description = "B: When the water level rises, the boat goes up" ∨
  description = "D: The higher you climb, the farther you see"

def not_cause_and_effect_relationship (description: String) : Prop :=
  description = "C: The brighter the moon, the fewer the stars"

theorem option_C_is_different :
  ∀ (description: String),
  (not_cause_and_effect_relationship description) →
  ¬ cause_and_effect_relationship description :=
by intros description h1 h2; sorry

end option_C_is_different_l103_103904


namespace bakery_used_0_2_bags_of_wheat_flour_l103_103109

-- Define the conditions
def total_flour := 0.3
def white_flour := 0.1

-- Define the number of bags of wheat flour used
def wheat_flour := total_flour - white_flour

-- The proof statement
theorem bakery_used_0_2_bags_of_wheat_flour : wheat_flour = 0.2 := 
by
  sorry

end bakery_used_0_2_bags_of_wheat_flour_l103_103109


namespace mileage_per_gallon_l103_103858

-- Define the conditions
def miles_driven : ℝ := 100
def gallons_used : ℝ := 5

-- Define the question as a theorem to be proven
theorem mileage_per_gallon : (miles_driven / gallons_used) = 20 := by
  sorry

end mileage_per_gallon_l103_103858


namespace joel_laps_count_l103_103544

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end joel_laps_count_l103_103544


namespace eva_marks_difference_l103_103632

theorem eva_marks_difference 
    (m2 : ℕ) (a2 : ℕ) (s2 : ℕ) (total_marks : ℕ)
    (h_m2 : m2 = 80) (h_a2 : a2 = 90) (h_s2 : s2 = 90) (h_total_marks : total_marks = 485)
    (m1 a1 s1 : ℕ)
    (h_m1 : m1 = m2 + 10)
    (h_a1 : a1 = a2 - 15)
    (h_s1 : s1 = s2 - 1 / 3 * s2)
    (total_semesters : ℕ)
    (h_total_semesters : total_semesters = m1 + a1 + s1 + m2 + a2 + s2)
    : m1 = m2 + 10 := by
  sorry

end eva_marks_difference_l103_103632


namespace remainder_of_5n_mod_11_l103_103225

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := 
by
  sorry

end remainder_of_5n_mod_11_l103_103225


namespace find_k_l103_103601

theorem find_k (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + A.2) = (B.1 + B.2) / 2 ∧ (A.1^2 + A.2^2 - 6 * A.1 - 4 * A.2 + 9 = 0) ∧ (B.1^2 + B.2^2 - 6 * B.1 - 4 * B.2 + 9 = 0)
     ∧ dist A B = 2 * Real.sqrt 3)
  (h3 : ∀ x y : ℝ, y = k * x + 3 → (x^2 + y^2 - 6 * x - 4 * y + 9) = 0)
  : k = 1 := sorry

end find_k_l103_103601


namespace octagon_mass_is_19kg_l103_103736

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end octagon_mass_is_19kg_l103_103736


namespace intersection_eq_l103_103328

-- Definitions for M and N
def M : Set ℤ := Set.univ
def N : Set ℤ := {x : ℤ | x^2 - x - 2 < 0}

-- The theorem to be proved
theorem intersection_eq : M ∩ N = {0, 1} := 
  sorry

end intersection_eq_l103_103328


namespace fish_count_seventh_day_l103_103719

-- Define the initial state and transformations
def fish_count (n: ℕ) :=
  if n = 0 then 6
  else
    if n = 3 then fish_count (n-1) / 3 * 2 * 2 * 2 - fish_count (n-1) / 3
    else if n = 5 then (fish_count (n-1) * 2) / 4 * 3
    else if n = 6 then fish_count (n-1) * 2 + 15
    else fish_count (n-1) * 2

theorem fish_count_seventh_day : fish_count 7 = 207 :=
by
  sorry

end fish_count_seventh_day_l103_103719


namespace no_such_function_exists_l103_103327

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) :=
by
  sorry

end no_such_function_exists_l103_103327


namespace hyperbola_eccentricity_l103_103004

theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ e : ℝ, e = 5 / 4 ∧ (∀ x y : ℝ, (x^2 / 16) - (y^2 / m) = 1)) → m = 9 :=
by
  intro h
  sorry

end hyperbola_eccentricity_l103_103004


namespace problem_solution_l103_103781

-- Definitions based on conditions given in the problem statement
def validExpression (n : ℕ) : ℕ := 
  sorry -- Placeholder for function defining valid expressions

def T (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else validExpression n

def R (n : ℕ) : ℕ := T n % 4

def computeSum (k : ℕ) : ℕ := 
  (List.range k).map R |>.sum

-- Lean theorem statement to be proven
theorem problem_solution : 
  computeSum 1000001 = 320 := 
sorry

end problem_solution_l103_103781


namespace remainder_3_pow_2040_mod_11_l103_103930

theorem remainder_3_pow_2040_mod_11 : (3 ^ 2040) % 11 = 1 := by
  have h1 : 3 % 11 = 3 := by norm_num
  have h2 : (3 ^ 2) % 11 = 9 := by norm_num
  have h3 : (3 ^ 3) % 11 = 5 := by norm_num
  have h4 : (3 ^ 4) % 11 = 4 := by norm_num
  have h5 : (3 ^ 5) % 11 = 1 := by norm_num
  have h_mod : 2040 % 5 = 0 := by norm_num
  sorry

end remainder_3_pow_2040_mod_11_l103_103930


namespace initial_number_of_men_l103_103636

theorem initial_number_of_men (M : ℕ) (F : ℕ) (h1 : F = M * 20) (h2 : (M - 100) * 10 = M * 15) : 
  M = 200 :=
  sorry

end initial_number_of_men_l103_103636


namespace min_area_triangle_l103_103788

theorem min_area_triangle (m n : ℝ) (h : m^2 + n^2 = 1/3) : ∃ S, S = 3 :=
by
  sorry

end min_area_triangle_l103_103788


namespace sum_of_digits_of_fraction_repeating_decimal_l103_103728

theorem sum_of_digits_of_fraction_repeating_decimal :
  (exists (c d : ℕ), (4 / 13 : ℚ) = c * 0.1 + d * 0.01 ∧ (c + d) = 3) :=
sorry

end sum_of_digits_of_fraction_repeating_decimal_l103_103728


namespace fraction_product_eq_six_l103_103596

theorem fraction_product_eq_six : (2/5) * (3/4) * (1/6) * (120 : ℚ) = 6 := by
  sorry

end fraction_product_eq_six_l103_103596


namespace melissa_earnings_from_sales_l103_103885

noncomputable def commission_earned (coupe_price suv_price commission_rate : ℕ) : ℕ :=
  (coupe_price + suv_price) * commission_rate / 100

theorem melissa_earnings_from_sales : 
  commission_earned 30000 60000 2 = 1800 :=
by
  sorry

end melissa_earnings_from_sales_l103_103885


namespace max_students_distributed_equally_l103_103363

theorem max_students_distributed_equally (pens pencils : ℕ) (h1 : pens = 3528) (h2 : pencils = 3920) : 
  Nat.gcd pens pencils = 392 := 
by 
  sorry

end max_students_distributed_equally_l103_103363


namespace unicorn_witch_ratio_l103_103597

theorem unicorn_witch_ratio (W D U : ℕ) (h1 : W = 7) (h2 : D = W + 25) (h3 : U + W + D = 60) :
  U / W = 3 := by
  sorry

end unicorn_witch_ratio_l103_103597


namespace find_value_of_x_l103_103691

theorem find_value_of_x (x : ℝ) : (45 * x = 0.4 * 900) -> x = 8 :=
by
  intro h
  sorry

end find_value_of_x_l103_103691


namespace system_of_equations_l103_103371

-- Given conditions: Total number of fruits and total cost of the fruits purchased
def total_fruits := 1000
def total_cost := 999
def cost_of_sweet_fruit := (11 : ℚ) / 9
def cost_of_bitter_fruit := (4 : ℚ) / 7

-- Variables representing the number of sweet and bitter fruits
variables (x y : ℚ)

-- Problem statement in Lean 4
theorem system_of_equations :
  (x + y = total_fruits) ∧ (cost_of_sweet_fruit * x + cost_of_bitter_fruit * y = total_cost) ↔
  ((x + y = 1000) ∧ (11 / 9 * x + 4 / 7 * y = 999)) :=
by
  sorry

end system_of_equations_l103_103371


namespace cats_not_eating_either_l103_103962

theorem cats_not_eating_either (total_cats : ℕ) (cats_liking_apples : ℕ) (cats_liking_fish : ℕ) (cats_liking_both : ℕ)
  (h1 : total_cats = 75) (h2 : cats_liking_apples = 15) (h3 : cats_liking_fish = 55) (h4 : cats_liking_both = 8) :
  ∃ cats_not_eating_either : ℕ, cats_not_eating_either = total_cats - (cats_liking_apples - cats_liking_both + cats_liking_fish - cats_liking_both + cats_liking_both) ∧ cats_not_eating_either = 13 :=
by
  sorry

end cats_not_eating_either_l103_103962


namespace find_k_values_l103_103236

theorem find_k_values (k : ℝ) : 
  (∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) → 
  (k = 0 ∨ k = 1 ∨ k = 2) ∧
  (k = 0 ∨ k = 1 ∨ k = 2 → ∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) :=
by
  sorry

end find_k_values_l103_103236


namespace ratio_of_expenditure_l103_103484

variable (A B AE BE : ℕ)

theorem ratio_of_expenditure (h1 : A = 2000) 
    (h2 : A / B = 5 / 4) 
    (h3 : A - AE = 800) 
    (h4: B - BE = 800) :
    AE / BE = 3 / 2 := by
  sorry

end ratio_of_expenditure_l103_103484


namespace option_d_correct_l103_103427

theorem option_d_correct (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b :=
by
  sorry

end option_d_correct_l103_103427


namespace set_intersection_l103_103825

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {x | x < 3}
noncomputable def N : Set ℝ := {y | y > 2}
noncomputable def CU_M : Set ℝ := {x | x ≥ 3}

theorem set_intersection :
  (CU_M ∩ N) = {x | x ≥ 3} := by
  sorry

end set_intersection_l103_103825


namespace arithmetic_sequence_common_difference_l103_103795

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : a 1 + a 7 = 22) 
  (h2 : a 4 + a 10 = 40) 
  (h_general_term : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
  : d = 3 :=
by 
  sorry

end arithmetic_sequence_common_difference_l103_103795


namespace smallest_positive_real_l103_103375

theorem smallest_positive_real (x : ℝ) (h₁ : ∃ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 4) : x = 29 / 5 :=
by
  sorry

end smallest_positive_real_l103_103375


namespace area_of_TURS_eq_area_of_PQRS_l103_103772

-- Definition of the rectangle PQRS
structure Rectangle where
  length : ℕ
  width : ℕ
  area : ℕ

-- Definition of the trapezoid TURS
structure Trapezoid where
  base1 : ℕ
  base2 : ℕ
  height : ℕ
  area : ℕ

-- Condition: PQRS is a rectangle whose area is 20 square units
def PQRS : Rectangle := { length := 5, width := 4, area := 20 }

-- Question: Prove the area of TURS equals area of PQRS
theorem area_of_TURS_eq_area_of_PQRS (TURS_area : ℕ) : TURS_area = PQRS.area :=
  sorry

end area_of_TURS_eq_area_of_PQRS_l103_103772


namespace prime_factor_of_reversed_difference_l103_103667

theorem prime_factor_of_reversed_difference (A B C : ℕ) (hA : A ≠ C) (hA_d : 1 ≤ A ∧ A ≤ 9) (hB_d : 0 ≤ B ∧ B ≤ 9) (hC_d : 1 ≤ C ∧ C ≤ 9) :
  ∃ p, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 11 := 
by
  sorry

end prime_factor_of_reversed_difference_l103_103667


namespace AB_not_together_correct_l103_103991

-- Definitions based on conditions
def total_people : ℕ := 5

-- The result from the complementary counting principle
def total_arrangements : ℕ := 120
def AB_together_arrangements : ℕ := 48

-- The arrangement count of A and B not next to each other
def AB_not_together_arrangements : ℕ := total_arrangements - AB_together_arrangements

theorem AB_not_together_correct : 
  AB_not_together_arrangements = 72 :=
sorry

end AB_not_together_correct_l103_103991


namespace reflection_points_reflection_line_l103_103881

-- Definitions of given points and line equation
def original_point : ℝ × ℝ := (2, 3)
def reflected_point : ℝ × ℝ := (8, 7)

-- Definitions of line parameters for y = mx + b
variable {m b : ℝ}

-- Statement of the reflection condition
theorem reflection_points_reflection_line : m + b = 9.5 := by
  -- sorry to skip the actual proof
  sorry

end reflection_points_reflection_line_l103_103881


namespace sum_c_d_eq_30_l103_103470

noncomputable def c_d_sum : ℕ :=
  let c : ℕ := 28
  let d : ℕ := 2
  c + d

theorem sum_c_d_eq_30 : c_d_sum = 30 :=
by {
  sorry
}

end sum_c_d_eq_30_l103_103470


namespace division_by_fraction_l103_103811

theorem division_by_fraction :
  (3 : ℚ) / (6 / 11) = 11 / 2 :=
by
  sorry

end division_by_fraction_l103_103811


namespace sum_a1_to_a5_l103_103378

noncomputable def f (x : ℝ) : ℝ := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5
noncomputable def g (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : ℝ := a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5

theorem sum_a1_to_a5 (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, f x = g x a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 1 = g 1 a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 0 = g 0 a_0 a_1 a_2 a_3 a_4 a_5) →
  a_0 = 62 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -57 :=
by
  intro hf1 hf2 hf3 ha0 hsum
  sorry

end sum_a1_to_a5_l103_103378


namespace mike_pens_l103_103318

-- Definitions based on the conditions
def initial_pens : ℕ := 25
def pens_after_mike (M : ℕ) : ℕ := initial_pens + M
def pens_after_cindy (M : ℕ) : ℕ := 2 * pens_after_mike M
def pens_after_sharon (M : ℕ) : ℕ := pens_after_cindy M - 19
def final_pens : ℕ := 75

-- The theorem we need to prove
theorem mike_pens (M : ℕ) (h : pens_after_sharon M = final_pens) : M = 22 := by
  have h1 : pens_after_sharon M = 2 * (25 + M) - 19 := rfl
  rw [h1] at h
  sorry

end mike_pens_l103_103318


namespace isosceles_triangle_angle_sum_l103_103158

theorem isosceles_triangle_angle_sum (x : ℝ) (h1 : x = 50 ∨ x = 65 ∨ x = 80) : (50 + 65 + 80 = 195) :=
by sorry

end isosceles_triangle_angle_sum_l103_103158


namespace wins_per_girl_l103_103480

theorem wins_per_girl (a b c d : ℕ) (h1 : a + b = 8) (h2 : a + c = 10) (h3 : b + c = 12) (h4 : a + d = 12) (h5 : b + d = 14) (h6 : c + d = 16) : 
  a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 :=
sorry

end wins_per_girl_l103_103480


namespace valid_triples_l103_103082

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1) ↔ (x, y, z) = (1, 1, 1) ∨ 
                                                      (x, y, z) = (1, 1, 2) ∨ 
                                                      (x, y, z) = (1, 3, 2) ∨ 
                                                      (x, y, z) = (3, 5, 4) :=
by
  sorry

end valid_triples_l103_103082


namespace find_square_side_l103_103373

theorem find_square_side (a b x : ℕ) (h_triangle : a^2 + x^2 = b^2)
  (h_trapezoid : 2 * a + 2 * b + 2 * x = 60)
  (h_rectangle : 4 * a + 2 * x = 58) :
  a = 12 := by
  sorry

end find_square_side_l103_103373


namespace greatest_possible_b_l103_103837

theorem greatest_possible_b (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
by sorry

end greatest_possible_b_l103_103837


namespace set_union_intersection_l103_103873

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {2, 3, 4}

theorem set_union_intersection :
  (A ∩ B) ∪ C = {1, 2, 3, 4} := 
by
  sorry

end set_union_intersection_l103_103873


namespace opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l103_103123

/-- A person is shooting at a target, firing twice in succession. 
    The opposite event of "hitting the target at least once" is "both shots miss". -/
theorem opposite_event_of_hitting_target_at_least_once_is_both_shots_miss :
  ∀ (A B : Prop) (hits_target_at_least_once both_shots_miss : Prop), 
    (hits_target_at_least_once → (A ∨ B)) → (both_shots_miss ↔ ¬hits_target_at_least_once) ∧ 
    (¬(A ∧ B) → both_shots_miss) :=
by
  sorry

end opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l103_103123


namespace total_cats_l103_103827

def num_white_cats : Nat := 2
def num_black_cats : Nat := 10
def num_gray_cats : Nat := 3

theorem total_cats : (num_white_cats + num_black_cats + num_gray_cats) = 15 :=
by
  sorry

end total_cats_l103_103827


namespace remainder_by_19_l103_103429

theorem remainder_by_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
by sorry

end remainder_by_19_l103_103429


namespace hospital_staff_total_l103_103859

def initial_doctors := 11
def initial_nurses := 18
def initial_medical_assistants := 9
def initial_interns := 6

def doctors_quit := 5
def nurses_quit := 2
def medical_assistants_quit := 3
def nurses_transferred := 2
def interns_transferred := 4
def doctors_vacation := 4
def nurses_vacation := 3

def new_doctors := 3
def new_nurses := 5

def remaining_doctors := initial_doctors - doctors_quit - doctors_vacation
def remaining_nurses := initial_nurses - nurses_quit - nurses_transferred - nurses_vacation
def remaining_medical_assistants := initial_medical_assistants - medical_assistants_quit
def remaining_interns := initial_interns - interns_transferred

def final_doctors := remaining_doctors + new_doctors
def final_nurses := remaining_nurses + new_nurses
def final_medical_assistants := remaining_medical_assistants
def final_interns := remaining_interns

def total_staff := final_doctors + final_nurses + final_medical_assistants + final_interns

theorem hospital_staff_total : total_staff = 29 := by
  unfold total_staff
  unfold final_doctors
  unfold final_nurses
  unfold final_medical_assistants
  unfold final_interns
  unfold remaining_doctors
  unfold remaining_nurses
  unfold remaining_medical_assistants
  unfold remaining_interns
  unfold initial_doctors initial_nurses initial_medical_assistants initial_interns
  unfold doctors_quit nurses_quit medical_assistants_quit nurses_transferred interns_transferred
  unfold doctors_vacation nurses_vacation
  unfold new_doctors new_nurses
  sorry

end hospital_staff_total_l103_103859


namespace order_of_a_b_c_l103_103721

noncomputable def ln : ℝ → ℝ := Real.log
noncomputable def a : ℝ := ln 3 / 3
noncomputable def b : ℝ := ln 5 / 5
noncomputable def c : ℝ := ln 6 / 6

theorem order_of_a_b_c : a > b ∧ b > c := by
  sorry

end order_of_a_b_c_l103_103721


namespace ratio_of_blue_to_red_area_l103_103306

theorem ratio_of_blue_to_red_area :
  let r₁ := 1 / 2
  let r₂ := 3 / 2
  let A_red := Real.pi * r₁^2
  let A_large := Real.pi * r₂^2
  let A_blue := A_large - A_red
  A_blue / A_red = 8 :=
by
  sorry

end ratio_of_blue_to_red_area_l103_103306


namespace ratio_four_of_v_m_l103_103958

theorem ratio_four_of_v_m (m v : ℝ) (h : m < v) 
  (h_eq : 5 * (3 / 4 * m) = v - 1 / 4 * m) : v / m = 4 :=
sorry

end ratio_four_of_v_m_l103_103958


namespace prism_volume_l103_103126

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) : a * b * c = 12 :=
by sorry

end prism_volume_l103_103126


namespace max_sum_42_l103_103496

noncomputable def max_horizontal_vertical_sum (numbers : List ℕ) : ℕ :=
  let a := 14
  let b := 11
  let e := 17
  a + b + e

theorem max_sum_42 : 
  max_horizontal_vertical_sum [2, 5, 8, 11, 14, 17] = 42 := by
  sorry

end max_sum_42_l103_103496


namespace two_digit_integers_remainder_3_count_l103_103689

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l103_103689


namespace birds_flew_up_l103_103845

-- Definitions based on conditions in the problem
def initial_birds : ℕ := 29
def new_total_birds : ℕ := 42

-- The statement to be proven
theorem birds_flew_up (x y z : ℕ) (h1 : x = initial_birds) (h2 : y = new_total_birds) (h3 : z = y - x) : z = 13 :=
by
  -- Proof will go here
  sorry

end birds_flew_up_l103_103845


namespace quadratic_solution_l103_103510

-- Definition of the quadratic function satisfying the given conditions
def quadraticFunc (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (f (-1) = 12 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → f x ≤ 12)

-- The proof goal: proving the function f(x) is 2x^2 - 10x
theorem quadratic_solution (f : ℝ → ℝ) (h : quadraticFunc f) : ∀ x, f x = 2 * x^2 - 10 * x :=
by
  sorry

end quadratic_solution_l103_103510


namespace polyhedron_faces_same_edges_l103_103961

theorem polyhedron_faces_same_edges (n : ℕ) (h_n : n ≥ 4) : 
  ∃ (f1 f2 : ℕ), f1 ≠ f2 ∧ 3 ≤ f1 ∧ f1 ≤ n - 1 ∧ 3 ≤ f2 ∧ f2 ≤ n - 1 ∧ f1 = f2 := 
by
  sorry

end polyhedron_faces_same_edges_l103_103961


namespace constant_function_of_horizontal_tangent_l103_103142

theorem constant_function_of_horizontal_tangent (f : ℝ → ℝ) (h : ∀ x, deriv f x = 0) : ∃ c : ℝ, ∀ x, f x = c :=
sorry

end constant_function_of_horizontal_tangent_l103_103142


namespace pizzas_bought_l103_103643

def slices_per_pizza := 8
def total_slices := 16

theorem pizzas_bought : total_slices / slices_per_pizza = 2 := by
  sorry

end pizzas_bought_l103_103643


namespace symmetric_line_eq_l103_103509

theorem symmetric_line_eq (x y : ℝ) : (x - y = 0) → (x = 1) → (y = -x + 2) :=
by
  sorry

end symmetric_line_eq_l103_103509


namespace intersection_value_l103_103905

theorem intersection_value (x y : ℝ) (h₁ : y = 10 / (x^2 + 5)) (h₂ : x + 2 * y = 5) : 
  x = 1 :=
sorry

end intersection_value_l103_103905


namespace find_cost_price_per_item_min_items_type_A_l103_103362

-- Definitions based on the conditions
def cost_A (x : ℝ) (y : ℝ) : Prop := 4 * x + 10 = 5 * y
def cost_B (x : ℝ) (y : ℝ) : Prop := 20 * x + 10 * y = 160

-- Proving the cost price per item of goods A and B
theorem find_cost_price_per_item : ∃ x y : ℝ, cost_A x y ∧ cost_B x y ∧ x = 5 ∧ y = 6 :=
by
  -- This is where the proof would go
  sorry

-- Additional conditions for part (2)
def profit_condition (a : ℕ) : Prop :=
  10 * (a - 30) + 8 * (200 - (a - 30)) - 5 * a - 6 * (200 - a) ≥ 640

-- Proving the minimum number of items of type A purchased
theorem min_items_type_A : ∃ a : ℕ, profit_condition a ∧ a ≥ 100 :=
by
  -- This is where the proof would go
  sorry

end find_cost_price_per_item_min_items_type_A_l103_103362


namespace necessary_but_not_sufficient_cond_l103_103975

noncomputable
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_cond (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (hseq : geometric_sequence a a1 q)
  (hpos : a1 > 0) :
  (q < 0 ↔ (∀ n : ℕ, a (2 * n + 1) + a (2 * n + 2) < 0)) :=
sorry

end necessary_but_not_sufficient_cond_l103_103975


namespace min_x9_minus_x1_l103_103081

theorem min_x9_minus_x1
  (x : Fin 9 → ℕ)
  (h_pos : ∀ i, x i > 0)
  (h_sorted : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.univ.sum x) = 220) :
    ∃ x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℕ,
    x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ x5 < x6 ∧ x6 < x7 ∧ x7 < x8 ∧ x8 < x9 ∧
    (x1 + x2 + x3 + x4 + x5 = 110) ∧
    x1 = x 0 ∧ x2 = x 1 ∧ x3 = x 2 ∧ x4 = x 3 ∧ x5 = x 4 ∧ x6 = x 5 ∧ x7 = x 6 ∧ x8 = x 7 ∧ x9 = x 8
    ∧ (x9 - x1 = 9) :=
sorry

end min_x9_minus_x1_l103_103081


namespace norma_cards_lost_l103_103810

def initial_cards : ℕ := 88
def final_cards : ℕ := 18
def cards_lost : ℕ := initial_cards - final_cards

theorem norma_cards_lost : cards_lost = 70 :=
by
  sorry

end norma_cards_lost_l103_103810


namespace problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l103_103275

theorem problem_inequality_a3_a2 (a : ℝ) (ha : a > 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem problem_inequality_relaxed (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem general_inequality (a : ℝ) (m n : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hmn1 : m > n) (hmn2 : n > 0) : 
  a^m + (1 / a^m) > a^n + (1 / a^n) := 
sorry

end problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l103_103275


namespace prime_factors_count_900_l103_103348

theorem prime_factors_count_900 : 
  ∃ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x ∧ x ∣ 900) ∧ S.card = 3 :=
by 
  sorry

end prime_factors_count_900_l103_103348


namespace prime_sum_mod_eighth_l103_103036

theorem prime_sum_mod_eighth (p1 p2 p3 p4 p5 p6 p7 p8 : ℕ) 
  (h₁ : p1 = 2) 
  (h₂ : p2 = 3) 
  (h₃ : p3 = 5) 
  (h₄ : p4 = 7) 
  (h₅ : p5 = 11) 
  (h₆ : p6 = 13) 
  (h₇ : p7 = 17) 
  (h₈ : p8 = 19) : 
  ((p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8) = 1 :=
by
  sorry

end prime_sum_mod_eighth_l103_103036


namespace fraction_addition_l103_103220

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l103_103220


namespace math_problem_l103_103903

theorem math_problem : (-4)^2 * ((-1)^2023 + (3 / 4) + (-1 / 2)^3) = -6 := 
by 
  sorry

end math_problem_l103_103903


namespace log_sum_identity_l103_103682

-- Prove that: lg 8 + 3 * lg 5 = 3

noncomputable def common_logarithm (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum_identity : 
    common_logarithm 8 + 3 * common_logarithm 5 = 3 := 
by
  sorry

end log_sum_identity_l103_103682


namespace total_students_l103_103803

theorem total_students (S : ℕ) (H1 : S / 2 = S - 15) : S = 30 :=
sorry

end total_students_l103_103803


namespace find_a_decreasing_l103_103880

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem find_a_decreasing : 
  (∀ x : ℝ, x < 6 → f x a ≤ f (x - 1) a) → a ≥ 6 := 
sorry

end find_a_decreasing_l103_103880


namespace range_of_m_l103_103483

-- Define the conditions:

/-- Proposition p: the equation represents an ellipse with foci on y-axis -/
def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 9 ∧ 9 - m > 2 * m ∧ 2 * m > 0

/-- Proposition q: the eccentricity of the hyperbola is in the interval (\sqrt(3)/2, \sqrt(2)) -/
def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ (5 / 2 < m ∧ m < 5)

def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def p_and_q (m : ℝ) : Prop := proposition_p m ∧ proposition_q m

-- Mathematically equivalent proof problem in Lean 4:

theorem range_of_m (m : ℝ) : (p_or_q m ∧ ¬p_and_q m) ↔ (m ∈ Set.Ioc 0 (5 / 2) ∪ Set.Icc 3 5) := sorry

end range_of_m_l103_103483


namespace lcm_of_two_numbers_l103_103703

theorem lcm_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 6) (h_product : a * b = 432) :
  Nat.lcm a b = 72 :=
by 
  sorry

end lcm_of_two_numbers_l103_103703


namespace find_n_l103_103165

theorem find_n 
  (num_engineers : ℕ) (num_technicians : ℕ) (num_workers : ℕ)
  (total_population : ℕ := num_engineers + num_technicians + num_workers)
  (systematic_sampling_inclusion_exclusion : ∀ n : ℕ, ∃ k : ℕ, n ∣ total_population ↔ n + 1 ≠ total_population) 
  (stratified_sampling_lcm : ∃ lcm : ℕ, lcm = Nat.lcm (Nat.lcm num_engineers num_technicians) num_workers)
  (total_population_is_36 : total_population = 36)
  (num_engineers_is_6 : num_engineers = 6)
  (num_technicians_is_12 : num_technicians = 12)
  (num_workers_is_18 : num_workers = 18) :
  ∃ n : ℕ, n = 6 :=
by
  sorry

end find_n_l103_103165


namespace g_g_2_eq_78652_l103_103000

def g (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem g_g_2_eq_78652 : g (g 2) = 78652 := by
  sorry

end g_g_2_eq_78652_l103_103000


namespace xiao_ming_total_evaluation_score_l103_103012

theorem xiao_ming_total_evaluation_score 
  (regular midterm final : ℤ) (weight_regular weight_midterm weight_final : ℕ)
  (h1 : regular = 80)
  (h2 : midterm = 90)
  (h3 : final = 85)
  (h_weight_regular : weight_regular = 3)
  (h_weight_midterm : weight_midterm = 3)
  (h_weight_final : weight_final = 4) :
  (regular * weight_regular + midterm * weight_midterm + final * weight_final) /
    (weight_regular + weight_midterm + weight_final) = 85 :=
by
  sorry

end xiao_ming_total_evaluation_score_l103_103012


namespace required_sampling_methods_l103_103272

-- Defining the given conditions
def total_households : Nat := 2000
def farmer_households : Nat := 1800
def worker_households : Nat := 100
def intellectual_households : Nat := total_households - farmer_households - worker_households
def sample_size : Nat := 40

-- Statement representing the proof problem
theorem required_sampling_methods :
  stratified_sampling_needed ∧ systematic_sampling_needed ∧ simple_random_sampling_needed :=
sorry

end required_sampling_methods_l103_103272


namespace distance_between_foci_of_ellipse_l103_103953

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l103_103953


namespace fourth_term_geometric_progression_l103_103938

theorem fourth_term_geometric_progression (x : ℝ) (h : ∀ n : ℕ, 0 < n → 
  (x ≠ 0 ∧ (2 * (x) + 2 * (n - 1)) ≠ 0 ∧ (3 * (x) + 3 * (n - 1)) ≠ 0)
  → ((2 * x + 2) / x) = (3 * x + 3) / (2 * x + 2)) : 
  ∃ r : ℝ, r = -13.5 := 
by 
  sorry

end fourth_term_geometric_progression_l103_103938


namespace paul_has_five_dogs_l103_103097

theorem paul_has_five_dogs
  (w1 w2 w3 w4 w5 : ℕ)
  (food_per_10_pounds : ℕ)
  (total_food_required : ℕ)
  (h1 : w1 = 20)
  (h2 : w2 = 40)
  (h3 : w3 = 10)
  (h4 : w4 = 30)
  (h5 : w5 = 50)
  (h6 : food_per_10_pounds = 1)
  (h7 : total_food_required = 15) :
  (w1 / 10 * food_per_10_pounds) +
  (w2 / 10 * food_per_10_pounds) +
  (w3 / 10 * food_per_10_pounds) +
  (w4 / 10 * food_per_10_pounds) +
  (w5 / 10 * food_per_10_pounds) = total_food_required → 
  5 = 5 :=
by
  intros
  sorry

end paul_has_five_dogs_l103_103097


namespace greatest_y_least_y_greatest_integer_y_l103_103897

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l103_103897


namespace problem_part1_problem_part2_problem_part3_l103_103737

open Set

noncomputable def U := ℝ
noncomputable def A := { x : ℝ | x < -4 ∨ x > 1 }
noncomputable def B := { x : ℝ | -3 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem_part1 :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 3 } := by sorry

theorem problem_part2 :
  compl A ∪ compl B = { x : ℝ | x ≤ 1 ∨ x > 3 } := by sorry

theorem problem_part3 (k : ℝ) :
  { x : ℝ | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1 } ⊆ A → k > 1 := by sorry

end problem_part1_problem_part2_problem_part3_l103_103737


namespace problem_a_problem_b_l103_103074

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end problem_a_problem_b_l103_103074


namespace friend_time_to_read_book_l103_103748

-- Define the conditions and variables
def my_reading_time : ℕ := 240 -- 4 hours in minutes
def speed_ratio : ℕ := 2 -- I read at half the speed of my friend

-- Define the variable for my friend's reading time which we need to find
def friend_reading_time : ℕ := my_reading_time / speed_ratio

-- The theorem statement that given the conditions, the friend's reading time is 120 minutes
theorem friend_time_to_read_book : friend_reading_time = 120 := sorry

end friend_time_to_read_book_l103_103748


namespace questionnaire_visitors_l103_103172

theorem questionnaire_visitors (V E : ℕ) (H1 : 140 = V - E) 
  (H2 : E = (3 * V) / 4) : V = 560 :=
by
  sorry

end questionnaire_visitors_l103_103172


namespace bacteria_fill_sixteenth_of_dish_in_26_days_l103_103833

theorem bacteria_fill_sixteenth_of_dish_in_26_days
  (days_to_fill_dish : ℕ)
  (doubling_rate : ℕ → ℕ)
  (H1 : days_to_fill_dish = 30)
  (H2 : ∀ n, doubling_rate (n + 1) = 2 * doubling_rate n) :
  doubling_rate 26 = doubling_rate 30 / 2^4 :=
sorry

end bacteria_fill_sixteenth_of_dish_in_26_days_l103_103833


namespace set_intersection_M_N_l103_103324

theorem set_intersection_M_N (x : ℝ) :
  let M := {x | -4 < x ∧ x < -2}
  let N := {x | x^2 + 5*x + 6 < 0}
  M ∩ N = {x | -3 < x ∧ x < -2} :=
by
  sorry

end set_intersection_M_N_l103_103324


namespace second_train_start_time_l103_103954

-- Define the conditions as hypotheses
def station_distance : ℝ := 200
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def meet_time : ℝ := 12 - 7 -- Time they meet after the first train starts, in hours.

-- The theorem statement corresponding to the proof problem
theorem second_train_start_time :
  ∃ T : ℝ, 0 <= T ∧ T <= 5 ∧ (5 * speed_train_A) + ((5 - T) * speed_train_B) = station_distance → T = 1 :=
by
  -- Placeholder for actual proof
  sorry

end second_train_start_time_l103_103954


namespace sofia_total_time_l103_103259

def distance1 : ℕ := 150
def speed1 : ℕ := 5
def distance2 : ℕ := 150
def speed2 : ℕ := 6
def laps : ℕ := 8
def time_per_lap := (distance1 / speed1) + (distance2 / speed2)
def total_time := 440  -- 7 minutes and 20 seconds in seconds

theorem sofia_total_time :
  laps * time_per_lap = total_time :=
by
  -- Proof steps are omitted and represented by sorry.
  sorry

end sofia_total_time_l103_103259


namespace necessary_but_not_sufficient_l103_103657

theorem necessary_but_not_sufficient (a b : ℕ) : 
  (a ≠ 1 ∨ b ≠ 2) → ¬ (a + b = 3) → ¬(a = 1 ∧ b = 2) ∧ ((a = 1 ∧ b = 2) → (a + b = 3)) := sorry

end necessary_but_not_sufficient_l103_103657


namespace share_ratio_l103_103147

theorem share_ratio (A B C x : ℝ)
  (h1 : A = 280)
  (h2 : A + B + C = 700)
  (h3 : A = x * (B + C))
  (h4 : B = (6 / 9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l103_103147


namespace six_digit_product_of_consecutive_even_integers_l103_103198

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l103_103198


namespace find_c_in_triangle_l103_103842

theorem find_c_in_triangle
  (angle_B : ℝ)
  (a : ℝ)
  (S : ℝ)
  (h1 : angle_B = 45)
  (h2 : a = 4)
  (h3 : S = 16 * Real.sqrt 2) :
  ∃ c : ℝ, c = 16 :=
by
  sorry

end find_c_in_triangle_l103_103842


namespace sum_of_three_consecutive_eq_product_of_distinct_l103_103771

theorem sum_of_three_consecutive_eq_product_of_distinct (n : ℕ) (h : 100 < n) :
  ∃ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  ((n + (n+1) + (n+2) = a * b * c) ∨
   ((n+1) + (n+2) + (n+3) = a * b * c) ∨
   (n + (n+1) + (n+3) = a * b * c) ∨
   (n + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end sum_of_three_consecutive_eq_product_of_distinct_l103_103771


namespace interest_rate_is_five_percent_l103_103043

-- Define the principal amount P and the interest rate r.
variables (P : ℝ) (r : ℝ)

-- Define the conditions given in the problem
def simple_interest_condition : Prop := P * r * 2 = 40
def compound_interest_condition : Prop := P * (1 + r)^2 - P = 41

-- Define the goal statement to prove
theorem interest_rate_is_five_percent (h1 : simple_interest_condition P r) (h2 : compound_interest_condition P r) : r = 0.05 :=
sorry

end interest_rate_is_five_percent_l103_103043


namespace min_value_x_plus_3y_min_value_xy_l103_103370

variable {x y : ℝ}

theorem min_value_x_plus_3y (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x + 3 * y ≥ 16 :=
sorry

theorem min_value_xy (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x * y ≥ 12 :=
sorry

end min_value_x_plus_3y_min_value_xy_l103_103370


namespace find_functions_l103_103025

theorem find_functions (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ^ 2) :=
by
  sorry

end find_functions_l103_103025


namespace seq_formula_l103_103205

noncomputable def seq {a : Nat → ℝ} (h1 : a 2 - a 1 = 1) (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1) : Nat → ℝ :=
sorry

theorem seq_formula {a : Nat → ℝ} 
  (h1 : a 2 - a 1 = 1)
  (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1)
  (n : Nat) : a n = 2 ^ n - 1 :=
sorry

end seq_formula_l103_103205


namespace donald_oranges_l103_103818

-- Define the initial number of oranges
def initial_oranges : ℕ := 4

-- Define the number of additional oranges found
def additional_oranges : ℕ := 5

-- Define the total number of oranges as the sum of initial and additional oranges
def total_oranges : ℕ := initial_oranges + additional_oranges

-- Theorem stating that the total number of oranges is 9
theorem donald_oranges : total_oranges = 9 := by
    -- Proof not provided, so we put sorry to indicate that this is a place for the proof.
    sorry

end donald_oranges_l103_103818


namespace prob_at_least_seven_friends_stay_for_entire_game_l103_103816

-- Definitions of conditions
def numFriends : ℕ := 8
def numUnsureFriends : ℕ := 5
def probabilityStay (p : ℚ) : ℚ := p
def sureFriends := 3

-- The probabilities
def prob_one_third : ℚ := 1 / 3
def prob_two_thirds : ℚ := 2 / 3

-- Variables to hold binomial coefficient and power calculation
noncomputable def C (n k : ℕ) : ℚ := (Nat.choose n k)
noncomputable def probability_at_least_seven_friends_stay : ℚ :=
  C numUnsureFriends 4 * (probabilityStay prob_one_third)^4 * (probabilityStay prob_two_thirds)^1 +
  (probabilityStay prob_one_third)^5

-- Theorem statement
theorem prob_at_least_seven_friends_stay_for_entire_game :
  probability_at_least_seven_friends_stay = 11 / 243 :=
  by sorry

end prob_at_least_seven_friends_stay_for_entire_game_l103_103816


namespace positive_difference_l103_103874

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l103_103874


namespace circle_center_l103_103057

theorem circle_center (x y : ℝ) : ∀ (h k : ℝ), (x^2 - 6*x + y^2 + 2*y = 9) → (x - h)^2 + (y - k)^2 = 19 → h = 3 ∧ k = -1 :=
by
  intros h k h_eq c_eq
  sorry

end circle_center_l103_103057


namespace bicycle_speed_l103_103835

theorem bicycle_speed (x : ℝ) (h : (2.4 / x) - (2.4 / (4 * x)) = 0.5) : 4 * x = 14.4 :=
by
  sorry

end bicycle_speed_l103_103835


namespace calf_rope_length_l103_103252

noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  let A1 := Real.pi * initial_length ^ 2
  let A2 := A1 + additional_area
  let new_length_squared := A2 / Real.pi
  Real.sqrt new_length_squared

theorem calf_rope_length :
  new_rope_length 12 565.7142857142857 = 18 := by
  sorry

end calf_rope_length_l103_103252


namespace remainder_17_pow_63_mod_7_l103_103647

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l103_103647


namespace necessary_but_not_sufficient_condition_l103_103641

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 2 → x^2 - a > 0) → (a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l103_103641


namespace prob_bashers_win_at_least_4_out_of_5_l103_103542

-- Define the probability p that the Bashers win a single game.
def p := 4 / 5

-- Define the number of games n.
def n := 5

-- Define the random trial outcome space.
def trials : Type := Fin n → Bool

-- Define the number of wins (true means a win, false means a loss).
def wins (t : trials) : ℕ := (Finset.univ.filter (λ i => t i = true)).card

-- Define winning exactly k games.
def win_exactly (t : trials) (k : ℕ) : Prop := wins t = k

-- Define the probability of winning exactly k games.
noncomputable def prob_win_exactly (k : ℕ) : ℚ :=
  (Nat.descFactorial n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the event of winning at least 4 out of 5 games.
def event_win_at_least (t : trials) := (wins t ≥ 4)

-- Define the probability of winning at least k out of n games.
noncomputable def prob_win_at_least (k : ℕ) : ℚ :=
  prob_win_exactly k + prob_win_exactly (k + 1)

-- Theorem to prove: Probability of winning at least 4 out of 5 games is 3072/3125.
theorem prob_bashers_win_at_least_4_out_of_5 :
  prob_win_at_least 4 = 3072 / 3125 :=
by
  sorry

end prob_bashers_win_at_least_4_out_of_5_l103_103542


namespace parabola_line_intersection_length_l103_103960

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - 1
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length (k x1 x2 y1 y2 : ℝ)
  (h_focus : line 1 0 k)
  (h_parabola1 : parabola x1 y1)
  (h_parabola2 : parabola x2 y2)
  (h_line1 : line x1 y1 k)
  (h_line2 : line x2 y2 k) :
  k = 1 ∧ (x1 + x2 + 2) = 8 :=
by
  sorry

end parabola_line_intersection_length_l103_103960


namespace sin_double_angle_fourth_quadrant_l103_103401

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l103_103401


namespace solution_interval_l103_103365

theorem solution_interval (X₀ : ℝ) (h₀ : Real.log (X₀ + 1) = 2 / X₀) : 1 < X₀ ∧ X₀ < 2 :=
by
  admit -- to be proved

end solution_interval_l103_103365


namespace time_spent_in_park_is_76_19_percent_l103_103545

noncomputable def total_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (t, _, _) => acc + t) 0

noncomputable def total_walking_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (_, w1, w2) => acc + (w1 + w2)) 0

noncomputable def total_trip_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  total_time_in_park trip_times + total_walking_time trip_times

noncomputable def percentage_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℚ :=
  (total_time_in_park trip_times : ℚ) / (total_trip_time trip_times : ℚ) * 100

theorem time_spent_in_park_is_76_19_percent (trip_times : List (ℕ × ℕ × ℕ)) :
  trip_times = [(120, 20, 25), (90, 15, 15), (150, 10, 20), (180, 30, 20), (120, 20, 10), (60, 15, 25)] →
  percentage_time_in_park trip_times = 76.19 :=
by
  intro h
  rw [h]  
  simp
  sorry

end time_spent_in_park_is_76_19_percent_l103_103545


namespace badminton_costs_l103_103078

variables (x : ℕ) (h : x > 16)

-- Define costs at Store A and Store B
def cost_A : ℕ := 1760 + 40 * x
def cost_B : ℕ := 1920 + 32 * x

-- Lean statement to prove the costs
theorem badminton_costs : 
  cost_A x = 1760 + 40 * x ∧ cost_B x = 1920 + 32 * x :=
by {
  -- This proof is expected but not required for the task
  sorry
}

end badminton_costs_l103_103078


namespace product_zero_when_a_is_2_l103_103614

theorem product_zero_when_a_is_2 : 
  ∀ (a : ℤ), a = 2 → (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  intros a ha
  sorry

end product_zero_when_a_is_2_l103_103614


namespace DM_eq_r_plus_R_l103_103069

noncomputable def radius_incircle (A B D : ℝ) (s K : ℝ) : ℝ := K / s

noncomputable def radius_excircle (A C D : ℝ) (s' K' : ℝ) (AD : ℝ) : ℝ := K' / (s' - AD)

theorem DM_eq_r_plus_R 
  (A B C D M : ℝ)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : A ≠ C)
  (h4 : D = (B + C) / 2)
  (h5 : M = (B + C) / 2)
  (r : ℝ)
  (h6 : r = radius_incircle A B D ((A + B + D) / 2) (abs ((A - B) * (A - D) / 2)))
  (R : ℝ)
  (h7 : R = radius_excircle A C D ((A + C + D) / 2) (abs ((A - C) * (A - D) / 2)) (abs (A - D))) :
  dist D M =r + R :=
by sorry

end DM_eq_r_plus_R_l103_103069


namespace total_votes_cast_l103_103283

def votes_witch : ℕ := 7
def votes_unicorn : ℕ := 3 * votes_witch
def votes_dragon : ℕ := votes_witch + 25
def votes_total : ℕ := votes_witch + votes_unicorn + votes_dragon

theorem total_votes_cast : votes_total = 60 := by
  sorry

end total_votes_cast_l103_103283


namespace simple_interest_rate_l103_103486

-- Definitions based on conditions
def principal : ℝ := 750
def amount : ℝ := 900
def time : ℕ := 10

-- Statement to prove the rate of simple interest
theorem simple_interest_rate : 
  ∃ (R : ℝ), principal * R * time / 100 = amount - principal ∧ R = 2 :=
by
  sorry

end simple_interest_rate_l103_103486


namespace students_between_min_and_hos_l103_103985

theorem students_between_min_and_hos
  (total_students : ℕ)
  (minyoung_left_position : ℕ)
  (hoseok_right_position : ℕ)
  (total_students_eq : total_students = 13)
  (minyoung_left_position_eq : minyoung_left_position = 8)
  (hoseok_right_position_eq : hoseok_right_position = 9) :
  (minyoung_left_position - (total_students - hoseok_right_position + 1) - 1) = 2 := 
by
  sorry

end students_between_min_and_hos_l103_103985


namespace response_percentage_is_50_l103_103787

-- Define the initial number of friends
def initial_friends := 100

-- Define the number of friends Mark kept initially
def kept_friends := 40

-- Define the number of friends Mark contacted
def contacted_friends := initial_friends - kept_friends

-- Define the number of friends Mark has after some responded
def remaining_friends := 70

-- Define the number of friends who responded to Mark's contact
def responded_friends := remaining_friends - kept_friends

-- Define the percentage of contacted friends who responded
def response_percentage := (responded_friends / contacted_friends) * 100

theorem response_percentage_is_50 :
  response_percentage = 50 := by
  sorry

end response_percentage_is_50_l103_103787


namespace prime_numbers_satisfying_equation_l103_103929

theorem prime_numbers_satisfying_equation :
  ∀ p : ℕ, Nat.Prime p →
    (∃ x y : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) →
    p = 2 ∨ p = 3 ∨ p = 7 := 
by 
  intro p hpprime h
  sorry

end prime_numbers_satisfying_equation_l103_103929


namespace simplify_expression_l103_103687

theorem simplify_expression (p : ℝ) : 
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := 
by 
  sorry

end simplify_expression_l103_103687


namespace jim_catches_up_to_cara_l103_103696

noncomputable def time_to_catch_up (jim_speed: ℝ) (cara_speed: ℝ) (initial_time: ℝ) (stretch_time: ℝ) : ℝ :=
  let initial_distance_jim := jim_speed * initial_time
  let initial_distance_cara := cara_speed * initial_time
  let added_distance_cara := cara_speed * stretch_time
  let distance_gap := added_distance_cara
  let relative_speed := jim_speed - cara_speed
  distance_gap / relative_speed

theorem jim_catches_up_to_cara :
  time_to_catch_up 6 5 (30/60) (18/60) * 60 = 90 :=
by
  sorry

end jim_catches_up_to_cara_l103_103696


namespace geometric_sequence_sum_l103_103062

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement to prove
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : is_geometric_sequence a q)
  (h2 : a 1 + a 2 = 40)
  (h3 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l103_103062


namespace light_bulbs_circle_l103_103536

theorem light_bulbs_circle : ∀ (f : ℕ → ℕ),
  (f 0 = 1) ∧
  (f 1 = 2) ∧
  (f 2 = 4) ∧
  (f 3 = 8) ∧
  (∀ n, f n = f (n - 1) + f (n - 2) + f (n - 3) + f (n - 4)) →
  (f 9 - 3 * f 3 - 2 * f 2 - f 1 = 367) :=
by
  sorry

end light_bulbs_circle_l103_103536


namespace algebra_inequality_l103_103743

variable {x y z : ℝ}

theorem algebra_inequality
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2
  ≥ x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) :=
sorry

end algebra_inequality_l103_103743


namespace perpendicular_line_to_plane_l103_103188

variables {Point Line Plane : Type}
variables (a b c : Line) (α : Plane) (A : Point)

-- Define the conditions
def line_perpendicular_to (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def lines_intersect_at (l1 l2 : Line) (P : Point) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given conditions in Lean 4
variables (h1 : line_perpendicular_to c a)
variables (h2 : line_perpendicular_to c b)
variables (h3 : line_in_plane a α)
variables (h4 : line_in_plane b α)
variables (h5 : lines_intersect_at a b A)

-- The theorem statement to prove
theorem perpendicular_line_to_plane : line_perpendicular_to_plane c α :=
sorry

end perpendicular_line_to_plane_l103_103188


namespace min_triangle_perimeter_l103_103103

/-- Given a point (a, b) with 0 < b < a,
    determine the minimum perimeter of a triangle with one vertex at (a, b),
    one on the x-axis, and one on the line y = x. 
    The minimum perimeter is √(2(a^2 + b^2)).
-/
theorem min_triangle_perimeter (a b : ℝ) (h : 0 < b ∧ b < a) 
  : ∃ c d : ℝ, c^2 + d^2 = 2 * (a^2 + b^2) := sorry

end min_triangle_perimeter_l103_103103


namespace distance_from_A_to_origin_l103_103702

open Real

theorem distance_from_A_to_origin 
  (x1 y1 : ℝ)
  (hx1 : y1^2 = 4 * x1)
  (hratio : (x1 + 1) / abs y1 = 5 / 4)
  (hAF_gt_2 : dist (x1, y1) (1, 0) > 2) : 
  dist (x1, y1) (0, 0) = 4 * sqrt 2 :=
sorry

end distance_from_A_to_origin_l103_103702


namespace min_value_a_plus_9b_l103_103668

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 1 / b = 1) : a + 9 * b ≥ 16 :=
  sorry

end min_value_a_plus_9b_l103_103668


namespace tan_75_eq_2_plus_sqrt_3_l103_103020

theorem tan_75_eq_2_plus_sqrt_3 :
  Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_eq_2_plus_sqrt_3_l103_103020


namespace chosen_number_is_129_l103_103588

theorem chosen_number_is_129 (x : ℕ) (h : 2 * x - 148 = 110) : x = 129 :=
by
  sorry

end chosen_number_is_129_l103_103588


namespace identify_person_l103_103617

variable (Person : Type) (Tweedledum Tralyalya : Person)
variable (has_black_card : Person → Prop)
variable (statement_true : Person → Prop)
variable (statement_made_by : Person)

-- Condition: The statement made: "Either I am Tweedledum, or I have a card of a black suit in my pocket."
def statement (p : Person) : Prop := p = Tweedledum ∨ has_black_card p

-- Condition: Anyone with a black card making a true statement is not possible.
axiom black_card_truth_contradiction : ∀ p : Person, has_black_card p → ¬ statement_true p

theorem identify_person :
statement_made_by = Tralyalya ∧ ¬ has_black_card statement_made_by :=
by
  sorry

end identify_person_l103_103617


namespace students_taking_neither_l103_103435

theorem students_taking_neither (total_students : ℕ)
    (students_math : ℕ) (students_physics : ℕ) (students_chemistry : ℕ)
    (students_math_physics : ℕ) (students_physics_chemistry : ℕ) (students_math_chemistry : ℕ)
    (students_all_three : ℕ) :
    total_students = 60 →
    students_math = 40 →
    students_physics = 30 →
    students_chemistry = 25 →
    students_math_physics = 18 →
    students_physics_chemistry = 10 →
    students_math_chemistry = 12 →
    students_all_three = 5 →
    (total_students - (students_math + students_physics + students_chemistry - students_math_physics - students_physics_chemistry - students_math_chemistry + students_all_three)) = 5 :=
by
  intros
  sorry

end students_taking_neither_l103_103435


namespace range_of_a_l103_103150

noncomputable def prop_p (a x : ℝ) : Prop := 3 * a < x ∧ x < a

noncomputable def prop_q (x : ℝ) : Prop := x^2 - x - 6 < 0

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, ¬ prop_p a x) ∧ ¬ (∃ x : ℝ, ¬ prop_p a x) → ¬ (∃ x : ℝ, ¬ prop_q x) → -2/3 ≤ a ∧ a < 0 := 
by
  sorry

end range_of_a_l103_103150


namespace radius_of_inscribed_circle_is_integer_l103_103705

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l103_103705


namespace ellipse_parameters_sum_l103_103116

theorem ellipse_parameters_sum 
  (h k a b : ℤ) 
  (h_def : h = 3) 
  (k_def : k = -5) 
  (a_def : a = 7) 
  (b_def : b = 2) : 
  h + k + a + b = 7 := 
by 
  -- definitions and sums will be handled by autogenerated proof
  sorry

end ellipse_parameters_sum_l103_103116


namespace intersection_complement_eq_l103_103821

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x : ℝ | x^2 - 2 * x ≥ 3 }
def N_complement : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_complement_eq :
  M ∩ N_complement = {1, 2} :=
by
  sorry

end intersection_complement_eq_l103_103821


namespace percent_of_1600_l103_103342

theorem percent_of_1600 (x : ℝ) (h1 : 0.25 * 1600 = 400) (h2 : x / 100 * 400 = 20) : x = 5 :=
sorry

end percent_of_1600_l103_103342


namespace part_a_part_b_l103_103718

def happy (n : ℕ) : Prop :=
  ∃ (a b : ℤ), a^2 + b^2 = n

theorem part_a (t : ℕ) (ht : happy t) : happy (2 * t) := 
sorry

theorem part_b (t : ℕ) (ht : happy t) : ¬ happy (3 * t) := 
sorry

end part_a_part_b_l103_103718


namespace even_and_monotonically_increasing_f3_l103_103075

noncomputable def f1 (x : ℝ) : ℝ := x^3
noncomputable def f2 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f3 (x : ℝ) : ℝ := abs x + 1
noncomputable def f4 (x : ℝ) : ℝ := 2^(-abs x)

theorem even_and_monotonically_increasing_f3 :
  (∀ x, f3 x = f3 (-x)) ∧ (∀ x > 0, ∀ y > x, f3 y > f3 x) := 
sorry

end even_and_monotonically_increasing_f3_l103_103075


namespace sum_of_powers_mod_l103_103792

-- Define a function that calculates the nth power of a number modulo a given base
def power_mod (a n k : ℕ) : ℕ := (a^n) % k

-- The main theorem: prove that the sum of powers modulo 5 gives the remainder 0
theorem sum_of_powers_mod 
  : ((power_mod 1 2013 5) + (power_mod 2 2013 5) + (power_mod 3 2013 5) + (power_mod 4 2013 5) + (power_mod 5 2013 5)) % 5 = 0 := 
by {
  sorry
}

end sum_of_powers_mod_l103_103792


namespace cost_of_batman_game_l103_103164

noncomputable def footballGameCost : ℝ := 14.02
noncomputable def strategyGameCost : ℝ := 9.46
noncomputable def totalAmountSpent : ℝ := 35.52

theorem cost_of_batman_game :
  totalAmountSpent - (footballGameCost + strategyGameCost) = 12.04 :=
by
  -- The proof is omitted as instructed.
  sorry

end cost_of_batman_game_l103_103164


namespace robotics_club_problem_l103_103582

theorem robotics_club_problem 
    (total_students cs_students eng_students both_students : ℕ)
    (h1 : total_students = 120)
    (h2 : cs_students = 75)
    (h3 : eng_students = 50)
    (h4 : both_students = 10) :
    total_students - (cs_students - both_students + eng_students - both_students + both_students) = 5 := by
  sorry

end robotics_club_problem_l103_103582


namespace juice_spilled_l103_103892

def initial_amount := 1.0
def Youngin_drank := 0.1
def Narin_drank := Youngin_drank + 0.2
def remaining_amount := 0.3

theorem juice_spilled :
  initial_amount - (Youngin_drank + Narin_drank) - remaining_amount = 0.3 :=
by
  sorry

end juice_spilled_l103_103892


namespace middle_number_is_9_point_5_l103_103379

theorem middle_number_is_9_point_5 (x y z : ℝ) 
  (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 22) : y = 9.5 := 
by {
  sorry
}

end middle_number_is_9_point_5_l103_103379


namespace conditional_probability_l103_103180

def slips : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

def P_A : ℚ := 5/9

def P_A_and_B : ℚ := 5/9 * 4/8

theorem conditional_probability :
  (5 / 18) / (5 / 9) = 1 / 2 :=
by
  sorry

end conditional_probability_l103_103180


namespace expected_winnings_l103_103244

-- Define the probabilities
def prob_heads : ℚ := 1/2
def prob_tails : ℚ := 1/3
def prob_edge : ℚ := 1/6

-- Define the winnings
def win_heads : ℚ := 1
def win_tails : ℚ := 3
def lose_edge : ℚ := -5

-- Define the expected value function
def expected_value (p1 p2 p3 : ℚ) (w1 w2 w3 : ℚ) : ℚ :=
  p1 * w1 + p2 * w2 + p3 * w3

-- The expected winnings from flipping this coin
theorem expected_winnings : expected_value prob_heads prob_tails prob_edge win_heads win_tails lose_edge = 2/3 :=
by
  sorry

end expected_winnings_l103_103244


namespace train_speed_is_36_0036_kmph_l103_103627

noncomputable def train_length : ℝ := 130
noncomputable def bridge_length : ℝ := 150
noncomputable def crossing_time : ℝ := 27.997760179185665
noncomputable def speed_in_kmph : ℝ := (train_length + bridge_length) / crossing_time * 3.6

theorem train_speed_is_36_0036_kmph :
  abs (speed_in_kmph - 36.0036) < 0.001 :=
by
  sorry

end train_speed_is_36_0036_kmph_l103_103627


namespace product_of_functions_l103_103303

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := -(3 * x - 1) / x

theorem product_of_functions (x : ℝ) (h : x ≠ 0) : f x * g x = -6 * x + 2 := by
  sorry

end product_of_functions_l103_103303


namespace sum_of_g_is_zero_l103_103899

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_of_g_is_zero :
  (Finset.range 2022).sum (λ k => (-1)^(k + 1) * g ((k + 1 : ℝ) / 2023)) = 0 :=
by
  sorry

end sum_of_g_is_zero_l103_103899


namespace finding_value_of_expression_l103_103411

open Real

theorem finding_value_of_expression
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : 1/a - 1/b - 1/(a + b) = 0) :
  (b/a + a/b)^2 = 5 :=
sorry

end finding_value_of_expression_l103_103411


namespace weight_of_one_fan_l103_103709

theorem weight_of_one_fan
  (total_weight_with_fans : ℝ)
  (num_fans : ℕ)
  (empty_box_weight : ℝ)
  (h1 : total_weight_with_fans = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight_with_fans - empty_box_weight) / num_fans = 0.76 :=
by
  simp [h1, h2, h3]
  sorry

end weight_of_one_fan_l103_103709


namespace solve_rational_eq_l103_103558

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 3 * x - 18) + 1 / (x^2 - 15 * x - 12) = 0) →
  (x = 1 ∨ x = -1 ∨ x = 12 ∨ x = -12) :=
by
  intro h
  sorry

end solve_rational_eq_l103_103558


namespace smallest_integer_value_l103_103693

theorem smallest_integer_value (n : ℤ) : ∃ (n : ℤ), n = 5 ∧ n^2 - 11*n + 28 < 0 :=
by
  use 5
  sorry

end smallest_integer_value_l103_103693


namespace gcd_of_polynomial_l103_103666

theorem gcd_of_polynomial (b : ℕ) (hb : b % 780 = 0) : Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 65) b = 65 := by
  sorry

end gcd_of_polynomial_l103_103666


namespace two_truth_tellers_are_B_and_C_l103_103982

-- Definitions of students and their statements
def A_statement_false (A_said : Prop) (A_truth_teller : Prop) := ¬A_said = A_truth_teller
def B_statement_true (B_said : Prop) (B_truth_teller : Prop) := B_said = B_truth_teller
def C_statement_true (C_said : Prop) (C_truth_teller : Prop) := C_said = C_truth_teller
def D_statement_false (D_said : Prop) (D_truth_teller : Prop) := ¬D_said = D_truth_teller

-- Given statements
def A_said := ¬ (False : Prop)
def B_said := True
def C_said := B_said ∨ D_statement_false True True
def D_said := False

-- Define who is telling the truth
def A_truth_teller := False
def B_truth_teller := True
def C_truth_teller := True
def D_truth_teller := False

-- Proof problem statement
theorem two_truth_tellers_are_B_and_C :
  (A_statement_false A_said A_truth_teller) ∧
  (B_statement_true B_said B_truth_teller) ∧
  (C_statement_true C_said C_truth_teller) ∧
  (D_statement_false D_said D_truth_teller) →
  ((A_truth_teller = False) ∧
  (B_truth_teller = True) ∧
  (C_truth_teller = True) ∧
  (D_truth_teller = False)) := 
by {
  sorry
}

end two_truth_tellers_are_B_and_C_l103_103982


namespace multiples_of_4_in_sequence_l103_103645

-- Define the arithmetic sequence terms
def nth_term (a d n : ℤ) : ℤ := a + (n - 1) * d

-- Define the conditions
def cond_1 : ℤ := 200 -- first term
def cond_2 : ℤ := -6 -- common difference
def smallest_term : ℤ := 2

-- Define the count of terms function
def num_terms (a d min : ℤ) : ℤ := (a - min) / -d + 1

-- The total number of terms in the sequence
def total_terms : ℤ := num_terms cond_1 cond_2 smallest_term

-- Define a function to get the ith term that is a multiple of 4
def ith_multiple_of_4 (n : ℤ) : ℤ := cond_1 + 18 * (n - 1)

-- Define the count of multiples of 4 within the given number of terms
def count_multiples_of_4 (total : ℤ) : ℤ := (total / 3) + 1

-- Final theorem statement
theorem multiples_of_4_in_sequence : count_multiples_of_4 total_terms = 12 := sorry

end multiples_of_4_in_sequence_l103_103645


namespace find_percentage_l103_103364

-- conditions
def N : ℕ := 160
def expected_percentage : ℕ := 35

-- statement to prove
theorem find_percentage (P : ℕ) (h : P / 100 * N = 50 / 100 * N - 24) : P = expected_percentage :=
sorry

end find_percentage_l103_103364


namespace first_player_can_always_make_A_eq_6_l103_103125

def maxSum3x3In5x5Board (board : Fin 5 → Fin 5 → ℕ) (i j : Fin 3) : ℕ :=
  (i + 3 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 5 : Fin 5)

theorem first_player_can_always_make_A_eq_6 :
  ∀ (board : Fin 5 → Fin 5 → ℕ), 
  (∀ (i j : Fin 3), maxSum3x3In5x5Board board i j = 6)
  :=
by
  intros board i j
  sorry

end first_player_can_always_make_A_eq_6_l103_103125


namespace joan_original_seashells_l103_103669

theorem joan_original_seashells (a b total: ℕ) (h1 : a = 63) (h2 : b = 16) (h3: total = a + b) : total = 79 :=
by
  rw [h1, h2] at h3
  exact h3

end joan_original_seashells_l103_103669


namespace alex_play_friends_with_l103_103591

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l103_103591


namespace simplify_expression_l103_103076

theorem simplify_expression (a : ℝ) :
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a :=
by
  sorry

end simplify_expression_l103_103076


namespace log_sequence_equality_l103_103751

theorem log_sequence_equality (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 1) (h2: a 2 + a 4 + a 6 = 18) : 
  Real.logb 3 (a 5 + a 7 + a 9) = 3 := 
by
  sorry

end log_sequence_equality_l103_103751


namespace relation_of_a_and_b_l103_103660

theorem relation_of_a_and_b (a b : ℝ) (h : 2^a + Real.log a / Real.log 2 = 4^b + 2 * Real.log b / Real.log 4) : a < 2 * b :=
sorry

end relation_of_a_and_b_l103_103660


namespace samples_from_workshop_l103_103878

theorem samples_from_workshop (T S P : ℕ) (hT : T = 2048) (hS : S = 128) (hP : P = 256) : 
  (s : ℕ) → (s : ℕ) = (256 * 128 / 2048) → s = 16 :=
by
  intros s hs
  rw [Nat.div_eq (256 * 128) 2048] at hs
  sorry

end samples_from_workshop_l103_103878


namespace jessica_blueberry_pies_l103_103387

theorem jessica_blueberry_pies 
  (total_pies : ℕ)
  (ratio_apple : ℕ)
  (ratio_blueberry : ℕ)
  (ratio_cherry : ℕ)
  (h_total : total_pies = 36)
  (h_ratios : ratio_apple = 2)
  (h_ratios_b : ratio_blueberry = 5)
  (h_ratios_c : ratio_cherry = 3) : 
  total_pies * ratio_blueberry / (ratio_apple + ratio_blueberry + ratio_cherry) = 18 := 
by
  sorry

end jessica_blueberry_pies_l103_103387


namespace find_cost_of_crackers_l103_103050

-- Definitions based on the given conditions
def cost_hamburger_meat : ℝ := 5.00
def cost_per_bag_vegetables : ℝ := 2.00
def number_of_bags_vegetables : ℕ := 4
def cost_cheese : ℝ := 3.50
def discount_rate : ℝ := 0.10
def total_after_discount : ℝ := 18

-- Definition of the box of crackers, which we aim to prove
def cost_crackers : ℝ := 3.50

-- The Lean statement for the proof
theorem find_cost_of_crackers
  (C : ℝ)
  (h : C = cost_crackers)
  (H : 0.9 * (cost_hamburger_meat + cost_per_bag_vegetables * number_of_bags_vegetables + cost_cheese + C) = total_after_discount) :
  C = 3.50 :=
  sorry

end find_cost_of_crackers_l103_103050


namespace exists_acute_triangle_side_lengths_l103_103368

-- Define the real numbers d_1, d_2, ..., d_12 in the interval (1, 12).
noncomputable def real_numbers_in_interval (d : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → 1 < d n ∧ d n < 12

-- Define the condition for d_i, d_j, d_k to form an acute triangle
def forms_acuse_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- The main theorem statement
theorem exists_acute_triangle_side_lengths (d : ℕ → ℝ) (h : real_numbers_in_interval d) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ forms_acuse_triangle (d i) (d j) (d k) :=
sorry

end exists_acute_triangle_side_lengths_l103_103368


namespace sequence_increasing_range_l103_103153

theorem sequence_increasing_range (a : ℝ) (n : ℕ) : 
  (∀ n ≤ 5, (a - 1) ^ (n - 4) < (a - 1) ^ ((n+1) - 4)) ∧
  (∀ n > 5, (7 - a) * n - 1 < (7 - a) * (n + 1) - 1) ∧
  (a - 1 < (7 - a) * 6 - 1) 
  → 2 < a ∧ a < 6 := 
sorry

end sequence_increasing_range_l103_103153


namespace katie_total_expenditure_l103_103920

-- Define the conditions
def flower_cost : ℕ := 6
def roses_bought : ℕ := 5
def daisies_bought : ℕ := 5

-- Define the total flowers bought
def total_flowers_bought : ℕ := roses_bought + daisies_bought

-- Calculate the total cost
def total_cost (flower_cost : ℕ) (total_flowers_bought : ℕ) : ℕ :=
  total_flowers_bought * flower_cost

-- Prove that Katie spent 60 dollars
theorem katie_total_expenditure : total_cost flower_cost total_flowers_bought = 60 := sorry

end katie_total_expenditure_l103_103920


namespace correct_operation_l103_103176

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end correct_operation_l103_103176


namespace barry_more_votes_than_joey_l103_103291

theorem barry_more_votes_than_joey {M B J X : ℕ} 
  (h1 : M = 66)
  (h2 : J = 8)
  (h3 : M = 3 * B)
  (h4 : B = 2 * (J + X)) :
  B - J = 14 := by
  sorry

end barry_more_votes_than_joey_l103_103291


namespace ratio_of_x_to_y_l103_103462

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
sorry

end ratio_of_x_to_y_l103_103462


namespace MrsYoung_puzzle_complete_l103_103730

theorem MrsYoung_puzzle_complete :
  let total_pieces := 500
  let children := 4
  let pieces_per_child := total_pieces / children
  let minutes := 120
  let pieces_Reyn := (25 * (minutes / 30))
  let pieces_Rhys := 2 * pieces_Reyn
  let pieces_Rory := 3 * pieces_Reyn
  let pieces_Rina := 4 * pieces_Reyn
  let total_pieces_placed := pieces_Reyn + pieces_Rhys + pieces_Rory + pieces_Rina
  total_pieces_placed >= total_pieces :=
by
  sorry

end MrsYoung_puzzle_complete_l103_103730


namespace find_number_l103_103305

-- Define the condition that one-third of a certain number is 300% of 134
def one_third_eq_300percent_number (n : ℕ) : Prop :=
  n / 3 = 3 * 134

-- State the theorem that the number is 1206 given the above condition
theorem find_number (n : ℕ) (h : one_third_eq_300percent_number n) : n = 1206 :=
  by sorry

end find_number_l103_103305


namespace milo_eggs_weight_l103_103747

def weight_of_one_egg : ℚ := 1/16
def eggs_per_dozen : ℕ := 12
def dozens_needed : ℕ := 8

theorem milo_eggs_weight :
  (dozens_needed * eggs_per_dozen : ℚ) * weight_of_one_egg = 6 := by sorry

end milo_eggs_weight_l103_103747


namespace tiffany_final_lives_l103_103422

def initial_lives : ℕ := 43
def lost_lives : ℕ := 14
def gained_lives : ℕ := 27

theorem tiffany_final_lives : (initial_lives - lost_lives + gained_lives) = 56 := by
    sorry

end tiffany_final_lives_l103_103422


namespace problem1_problem2_l103_103337

theorem problem1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 :=
  sorry

theorem problem2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 :=
  sorry

end problem1_problem2_l103_103337


namespace sum_square_geq_one_third_l103_103424

variable (a b c : ℝ)

theorem sum_square_geq_one_third (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sum_square_geq_one_third_l103_103424


namespace height_of_tree_l103_103513

noncomputable def height_of_flagpole : ℝ := 4
noncomputable def shadow_of_flagpole : ℝ := 6
noncomputable def shadow_of_tree : ℝ := 12

theorem height_of_tree (h : height_of_flagpole / shadow_of_flagpole = x / shadow_of_tree) : x = 8 := by
  sorry

end height_of_tree_l103_103513


namespace Haleigh_needs_leggings_l103_103887

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end Haleigh_needs_leggings_l103_103887


namespace domain_of_function_l103_103504

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 3 * x + 1 > 0
def condition2 (x : ℝ) : Prop := 2 - x ≠ 0

-- Define the domain of the function
def domain (x : ℝ) : Prop := x > -1 / 3 ∧ x ≠ 2

theorem domain_of_function : 
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ domain x := 
by
  sorry

end domain_of_function_l103_103504


namespace Laura_more_than_200_paperclips_on_Friday_l103_103307

theorem Laura_more_than_200_paperclips_on_Friday:
  ∀ (n : ℕ), (n = 4 ∨ n = 0 ∨ n ≥ 1 ∧ (n - 1 = 0 ∨ n = 1) → 4 * 3 ^ n > 200) :=
by
  sorry

end Laura_more_than_200_paperclips_on_Friday_l103_103307


namespace correct_fraction_subtraction_l103_103224

theorem correct_fraction_subtraction (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  ((1 / x) - (1 / (x - 1))) = - (1 / (x^2 - x)) :=
by
  sorry

end correct_fraction_subtraction_l103_103224


namespace find_L_l103_103234

noncomputable def L_value : ℕ := 3

theorem find_L
  (a b : ℕ)
  (cows : ℕ := 5 * b)
  (chickens : ℕ := 5 * a + 7)
  (insects : ℕ := b ^ (a - 5))
  (legs_cows : ℕ := 4 * cows)
  (legs_chickens : ℕ := 2 * chickens)
  (legs_insects : ℕ :=  6 * insects)
  (total_legs : ℕ := legs_cows + legs_chickens + legs_insects) 
  (h1 : cows = insects)
  (h2 : total_legs = (L_value * 100 + L_value * 10 + L_value) + 1) :
  L_value = 3 := sorry

end find_L_l103_103234


namespace grass_coverage_day_l103_103292

theorem grass_coverage_day (coverage : ℕ → ℚ) : 
  (∀ n : ℕ, coverage (n + 1) = 2 * coverage n) → 
  coverage 24 = 1 → 
  coverage 21 = 1 / 8 := 
by
  sorry

end grass_coverage_day_l103_103292


namespace symmetric_point_l103_103246

theorem symmetric_point (x y : ℝ) (a b : ℝ) :
  (x = 3 ∧ y = 9 ∧ a = -1 ∧ b = -3) ∧ (∀ k: ℝ, k ≠ 0 → (y - 9 = k * (x - 3)) ∧ 
  ((x - 3)^2 + (y - 9)^2 = (a - 3)^2 + (b - 9)^2) ∧ 
  (x >= 0 → (a >= 0 ↔ x = 3) ∧ (b >= 0 ↔ y = 9))) :=
by
  sorry

end symmetric_point_l103_103246


namespace min_value_fraction_l103_103676

theorem min_value_fraction (a b : ℝ) (n : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_sum : a + b = 2) : 
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_fraction_l103_103676


namespace area_of_rectangle_ABCD_l103_103351

theorem area_of_rectangle_ABCD :
  ∀ (short_side long_side width length : ℝ),
    (short_side = 6) →
    (long_side = 6 * (3 / 2)) →
    (width = 2 * short_side) →
    (length = long_side) →
    (width * length = 108) :=
by
  intros short_side long_side width length h_short h_long h_width h_length
  rw [h_short, h_long] at *
  sorry

end area_of_rectangle_ABCD_l103_103351


namespace roller_coaster_cars_l103_103304

theorem roller_coaster_cars (n : ℕ) (h : ((n - 1) : ℝ) / n = 0.5) : n = 2 :=
sorry

end roller_coaster_cars_l103_103304


namespace total_attendance_l103_103786

theorem total_attendance (A C : ℕ) (ticket_sales : ℕ) (adult_ticket_cost child_ticket_cost : ℕ) (total_collected : ℕ)
    (h1 : C = 18) (h2 : ticket_sales = 50) (h3 : adult_ticket_cost = 8) (h4 : child_ticket_cost = 1)
    (h5 : ticket_sales = adult_ticket_cost * A + child_ticket_cost * C) :
    A + C = 22 :=
by {
  sorry
}

end total_attendance_l103_103786


namespace proposition_judgement_l103_103261

theorem proposition_judgement (p q : Prop) (a b c x : ℝ) :
  (¬ (p ∨ q) → (¬ p ∧ ¬ q)) ∧
  (¬ (a > b → a * c^2 > b * c^2)) ∧
  (¬ (∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  ((x^2 - 3*x + 2 = 0) → (x = 2)) =
  false := sorry

end proposition_judgement_l103_103261


namespace maximize_profit_l103_103279

variables (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

-- Definitions for the conditions
def nonneg_x := (0 ≤ x)
def nonneg_y := (0 ≤ y)
def constraint1 := (a1 * x + a2 * y ≤ c1)
def constraint2 := (b1 * x + b2 * y ≤ c2)
def profit := (z = d1 * x + d2 * y)

-- Proof of constraints and profit condition
theorem maximize_profit (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ) :
    nonneg_x x ∧ nonneg_y y ∧ constraint1 a1 a2 c1 x y ∧ constraint2 b1 b2 c2 x y → profit d1 d2 x y z :=
by
  sorry

end maximize_profit_l103_103279


namespace chocolates_per_small_box_l103_103670

/-- A large box contains 19 small boxes and each small box contains a certain number of chocolate bars.
There are 475 chocolate bars in the large box. --/
def number_of_chocolate_bars_per_small_box : Prop :=
  ∃ x : ℕ, 475 = 19 * x ∧ x = 25

theorem chocolates_per_small_box : number_of_chocolate_bars_per_small_box :=
by
  sorry -- proof is skipped

end chocolates_per_small_box_l103_103670


namespace find_e_value_l103_103242

-- Define constants a, b, c, d, and e
variables (a b c d e : ℝ)

-- Theorem statement
theorem find_e_value (h1 : (2 : ℝ)^7 * a + (2 : ℝ)^5 * b + (2 : ℝ)^3 * c + 2 * d + e = 23)
                     (h2 : ((-2) : ℝ)^7 * a + ((-2) : ℝ)^5 * b + ((-2) : ℝ)^3 * c + ((-2) : ℝ) * d + e = -35) :
  e = -6 :=
sorry

end find_e_value_l103_103242


namespace harmful_bacteria_time_l103_103204

noncomputable def number_of_bacteria (x : ℝ) : ℝ :=
  4000 * 2^x

theorem harmful_bacteria_time :
  ∃ (x : ℝ), number_of_bacteria x > 90000 ∧ x = 4.5 :=
by
  sorry

end harmful_bacteria_time_l103_103204


namespace paper_fold_ratio_l103_103024

theorem paper_fold_ratio (paper_side : ℕ) (fold_fraction : ℚ) (cut_fraction : ℚ)
  (thin_section_width thick_section_width : ℕ) (small_width large_width : ℚ)
  (P_small P_large : ℚ) (ratio : ℚ) :
  paper_side = 6 →
  fold_fraction = 1 / 3 →
  cut_fraction = 2 / 3 →
  thin_section_width = 2 →
  thick_section_width = 4 →
  small_width = 2 →
  large_width = 16 / 3 →
  P_small = 2 * (6 + small_width) →
  P_large = 2 * (6 + large_width) →
  ratio = P_small / P_large →
  ratio = 12 / 17 :=
by
  sorry

end paper_fold_ratio_l103_103024


namespace rabbit_population_2002_l103_103700

theorem rabbit_population_2002 :
  ∃ (x : ℕ) (k : ℝ), 
    (180 - 50 = k * x) ∧ 
    (255 - 75 = k * 180) ∧ 
    x = 130 :=
by
  sorry

end rabbit_population_2002_l103_103700


namespace sequence_general_formula_l103_103893

-- Define conditions: The sum of the first n terms of the sequence is Sn = an - 3
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
axiom condition (n : ℕ) : S n = a n - 3

-- Define the main theorem to prove
theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : a n = 2 * 3 ^ n :=
sorry

end sequence_general_formula_l103_103893


namespace infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l103_103900

theorem infinite_nat_sum_of_squares_and_cubes_not_sixth_powers :
  ∃ (N : ℕ) (k : ℕ), N > 0 ∧
  (N = 250 * 3^(6 * k)) ∧
  (∃ (x y : ℕ), N = x^2 + y^2) ∧
  (∃ (a b : ℕ), N = a^3 + b^3) ∧
  (∀ (u v : ℕ), N ≠ u^6 + v^6) :=
by
  sorry

end infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l103_103900


namespace original_profit_percentage_l103_103308

theorem original_profit_percentage
  (P SP : ℝ)
  (h1 : SP = 549.9999999999995)
  (h2 : SP = P * (1 + x / 100))
  (h3 : 0.9 * P * 1.3 = SP + 35) :
  x = 10 := 
sorry

end original_profit_percentage_l103_103308


namespace t_shirts_in_two_hours_l103_103550

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end t_shirts_in_two_hours_l103_103550


namespace wang_hao_not_last_l103_103829

-- Define the total number of ways to select and arrange 3 players out of 6
def ways_total : ℕ := Nat.factorial 6 / Nat.factorial (6 - 3)

-- Define the number of ways in which Wang Hao is the last player
def ways_wang_last : ℕ := Nat.factorial 5 / Nat.factorial (5 - 2)

-- Proof statement
theorem wang_hao_not_last : ways_total - ways_wang_last = 100 :=
by sorry

end wang_hao_not_last_l103_103829


namespace heather_starts_24_minutes_after_stacy_l103_103807

theorem heather_starts_24_minutes_after_stacy :
  ∀ (distance_between : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) (heather_distance : ℝ),
    distance_between = 10 →
    heather_speed = 5 →
    stacy_speed = heather_speed + 1 →
    heather_distance = 3.4545454545454546 →
    60 * ((heather_distance / heather_speed) - ((distance_between - heather_distance) / stacy_speed)) = -24 :=
by
  sorry

end heather_starts_24_minutes_after_stacy_l103_103807


namespace inscribed_square_area_l103_103415

-- Define the conditions and the problem
theorem inscribed_square_area
  (side_length : ℝ)
  (square_area : ℝ) :
  side_length = 24 →
  square_area = 576 :=
by
  sorry

end inscribed_square_area_l103_103415


namespace cube_divided_by_five_tetrahedrons_l103_103298

-- Define the minimum number of tetrahedrons needed to divide a cube
def min_tetrahedrons_to_divide_cube : ℕ := 5

-- State the theorem
theorem cube_divided_by_five_tetrahedrons : min_tetrahedrons_to_divide_cube = 5 :=
by
  -- The proof is skipped, as instructed
  sorry

end cube_divided_by_five_tetrahedrons_l103_103298


namespace range_of_a_l103_103595

theorem range_of_a 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x, f x = -x^2 + 2*(a - 1)*x + 2)
  (increasing_on : ∀ x < 4, deriv f x > 0) : a ≥ 5 :=
sorry

end range_of_a_l103_103595


namespace no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l103_103157

theorem no_even_integers_of_form_3k_plus_4_and_5m_plus_2 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, n = 3 * k + 4) (h3 : ∃ m : ℕ, n = 5 * m + 2) (h4 : n % 2 = 0) : false :=
sorry

end no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l103_103157


namespace votes_for_Crow_l103_103906

theorem votes_for_Crow 
  (J : ℕ)
  (P V K : ℕ)
  (ε1 ε2 ε3 ε4 : ℤ)
  (h₁ : P + V = 15 + ε1)
  (h₂ : V + K = 18 + ε2)
  (h₃ : K + P = 20 + ε3)
  (h₄ : P + V + K = 59 + ε4)
  (bound₁ : |ε1| ≤ 13)
  (bound₂ : |ε2| ≤ 13)
  (bound₃ : |ε3| ≤ 13)
  (bound₄ : |ε4| ≤ 13)
  : V = 13 :=
sorry

end votes_for_Crow_l103_103906


namespace sunland_more_plates_than_moonland_l103_103195

theorem sunland_more_plates_than_moonland : 
  let sunland_plates := 26^4 * 10^2
  let moonland_plates := 26^3 * 10^3
  (sunland_plates - moonland_plates) = 7321600 := 
by
  sorry

end sunland_more_plates_than_moonland_l103_103195


namespace sphere_surface_area_l103_103956

theorem sphere_surface_area (r : ℝ) (hr : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi :=
by
  rw [hr]
  norm_num
  sorry

end sphere_surface_area_l103_103956


namespace sum_of_factors_l103_103352

-- Define 36
def num : ℕ := 36

-- Prime factorization of 36
def prime_factorization : (ℕ × ℕ) := (2, 2) -- represents 2^2
def prime_factorization_2 : (ℕ × ℕ) := (3, 2) -- represents 3^2

-- Sum of divisors function formula
def sigma (n : ℕ) : ℕ := 
  (1 + 2 + 2^2) * (1 + 3 + 3^2)

-- Theorem: sum of the positive factors of 36 is 91
theorem sum_of_factors : sigma num = 91 := 
by {
  -- Skipping the proof steps
  sorry
}

end sum_of_factors_l103_103352


namespace profit_percentage_is_correct_l103_103490

noncomputable def sellingPrice : ℝ := 850
noncomputable def profit : ℝ := 230
noncomputable def costPrice : ℝ := sellingPrice - profit

noncomputable def profitPercentage : ℝ :=
  (profit / costPrice) * 100

theorem profit_percentage_is_correct :
  profitPercentage = 37.10 :=
by
  sorry

end profit_percentage_is_correct_l103_103490


namespace milk_for_flour_l103_103181

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end milk_for_flour_l103_103181


namespace day_of_week_50th_day_of_year_N_minus_1_l103_103046

def day_of_week (d : ℕ) (first_day : ℕ) : ℕ :=
  (first_day + d - 1) % 7

theorem day_of_week_50th_day_of_year_N_minus_1 
  (N : ℕ) 
  (day_250_N : ℕ) 
  (day_150_N_plus_1 : ℕ) 
  (h1 : day_250_N = 3)  -- 250th day of year N is Wednesday (3rd day of week, 0 = Sunday)
  (h2 : day_150_N_plus_1 = 3) -- 150th day of year N+1 is also Wednesday (3rd day of week, 0 = Sunday)
  : day_of_week 50 (day_of_week 1 ((day_of_week 1 day_250_N - 1 + 250) % 365 - 1 + 366)) = 6 := 
sorry

-- Explanation:
-- day_of_week function calculates the day of the week given the nth day of the year and the first day of the year.
-- Given conditions that 250th day of year N and 150th day of year N+1 are both Wednesdays (represented by 3 assuming Sunday = 0).
-- We need to derive that the 50th day of year N-1 is a Saturday (represented by 6 assuming Sunday = 0).

end day_of_week_50th_day_of_year_N_minus_1_l103_103046


namespace tangent_line_through_external_point_l103_103611

theorem tangent_line_through_external_point (x y : ℝ) (h_circle : x^2 + y^2 = 1) (P : ℝ × ℝ) (h_P : P = (1, 2)) : 
  (∃ k : ℝ, (y = 2 + k * (x - 1)) ∧ (x = 1 ∨ (3 * x - 4 * y + 5 = 0))) :=
by
  sorry

end tangent_line_through_external_point_l103_103611


namespace carnival_wait_time_l103_103008

theorem carnival_wait_time :
  ∀ (T : ℕ), 4 * 60 = 4 * 30 + T + 4 * 15 → T = 60 :=
by
  intro T
  intro h
  sorry

end carnival_wait_time_l103_103008


namespace tangent_line_at_one_l103_103216

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

theorem tangent_line_at_one (a b : ℝ) (h_tangent : ∀ x, f x = a * x + b) : 
  a + b = 1 := 
sorry

end tangent_line_at_one_l103_103216


namespace solve_expression_hundreds_digit_l103_103694

def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

def div_mod (a b m : ℕ) : ℕ :=
  (a / b) % m

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem solve_expression_hundreds_digit :
  hundreds_digit (div_mod (factorial 17) 5 1000 - div_mod (factorial 10) 2 1000) = 8 :=
by
  sorry

end solve_expression_hundreds_digit_l103_103694


namespace calculate_width_of_vessel_base_l103_103384

noncomputable def cube_edge : ℝ := 17
noncomputable def base_length : ℝ := 20
noncomputable def water_rise : ℝ := 16.376666666666665
noncomputable def cube_volume : ℝ := cube_edge ^ 3
noncomputable def base_area (W : ℝ) : ℝ := base_length * W
noncomputable def displaced_volume (W : ℝ) : ℝ := base_area W * water_rise

theorem calculate_width_of_vessel_base :
  ∃ W : ℝ, displaced_volume W = cube_volume ∧ W = 15 := by
  sorry

end calculate_width_of_vessel_base_l103_103384


namespace largest_y_coordinate_l103_103988

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 25) + ((y - 3)^2 / 25) = 0) : y = 3 := by
  sorry

end largest_y_coordinate_l103_103988


namespace salary_problem_l103_103723

theorem salary_problem
  (A B : ℝ)
  (h1 : A + B = 3000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 2250 :=
sorry

end salary_problem_l103_103723


namespace remainder_3_mod_6_l103_103399

theorem remainder_3_mod_6 (n : ℕ) (h : n % 18 = 3) : n % 6 = 3 :=
by
    sorry

end remainder_3_mod_6_l103_103399


namespace equal_real_roots_l103_103300

theorem equal_real_roots (m : ℝ) : (∃ x : ℝ, x * x - 4 * x - m = 0) → (16 + 4 * m = 0) → m = -4 :=
by
  sorry

end equal_real_roots_l103_103300


namespace converse_proposition_false_l103_103408

theorem converse_proposition_false (a b c : ℝ) : ¬(∀ a b c : ℝ, (a > b) → (a * c^2 > b * c^2)) :=
by {
  -- proof goes here
  sorry
}

end converse_proposition_false_l103_103408


namespace zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l103_103282

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 2 then 2^x + a else a - x

theorem zero_of_f_a_neg_sqrt2 : 
  ∀ x, f x (- Real.sqrt 2) = 0 ↔ x = 1/2 :=
by
  sorry

theorem range_of_a_no_zero :
  ∀ a, (¬∃ x, f x a = 0) ↔ a ∈ Set.Iic (-4) ∪ Set.Ico 0 2 :=
by
  sorry

end zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l103_103282


namespace depth_second_project_l103_103049

def volume (depth length breadth : ℝ) : ℝ := depth * length * breadth

theorem depth_second_project (D : ℝ) : 
  (volume 100 25 30 = volume D 20 50) → D = 75 :=
by 
  sorry

end depth_second_project_l103_103049


namespace least_subtraction_divisible_l103_103189

def least_subtrahend (n m : ℕ) : ℕ :=
n % m

theorem least_subtraction_divisible (n : ℕ) (m : ℕ) (sub : ℕ) :
  n = 13604 → m = 87 → sub = least_subtrahend n m → (n - sub) % m = 0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end least_subtraction_divisible_l103_103189


namespace roots_of_quadratic_are_integers_l103_103299

theorem roots_of_quadratic_are_integers
  (b c : ℤ)
  (Δ : ℤ)
  (h_discriminant: Δ = b^2 - 4 * c)
  (h_perfect_square: ∃ k : ℤ, k^2 = Δ)
  : (∃ x1 x2 : ℤ, x1 * x2 = c ∧ x1 + x2 = -b) :=
by
  sorry

end roots_of_quadratic_are_integers_l103_103299


namespace ratio_of_A_to_B_is_4_l103_103663

noncomputable def A_share : ℝ := 360
noncomputable def B_share : ℝ := 90
noncomputable def ratio_A_B : ℝ := A_share / B_share

theorem ratio_of_A_to_B_is_4 : ratio_A_B = 4 :=
by
  -- This is the proof that we are skipping
  sorry

end ratio_of_A_to_B_is_4_l103_103663


namespace minor_premise_l103_103909

variables (A B C : Prop)

theorem minor_premise (hA : A) (hB : B) (hC : C) : B := 
by
  exact hB

end minor_premise_l103_103909


namespace fraction_married_men_l103_103659

-- Define the problem conditions
def num_faculty : ℕ := 100
def women_perc : ℕ := 60
def married_perc : ℕ := 60
def single_men_perc : ℚ := 3/4

-- We need to calculate the fraction of men who are married.
theorem fraction_married_men :
  (60 : ℚ) / 100 = women_perc / num_faculty →
  (60 : ℚ) / 100 = married_perc / num_faculty →
  (3/4 : ℚ) = single_men_perc →
  ∃ (fraction : ℚ), fraction = 1/4 :=
by
  intro h1 h2 h3
  sorry

end fraction_married_men_l103_103659


namespace bakery_storage_l103_103394

theorem bakery_storage (S F B : ℕ) (h1 : S * 8 = 3 * F) (h2 : F * 1 = 10 * B) (h3 : F * 1 = 8 * (B + 60)) : S = 900 :=
by
  -- We would normally put the proof steps here, but since it's specified to include only the statement
  sorry

end bakery_storage_l103_103394


namespace orthocenter_PQR_l103_103957

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end orthocenter_PQR_l103_103957


namespace total_cats_l103_103207

theorem total_cats (current_cats : ℕ) (additional_cats : ℕ) (h1 : current_cats = 11) (h2 : additional_cats = 32):
  current_cats + additional_cats = 43 :=
by
  -- We state the given conditions:
  -- current_cats = 11
  -- additional_cats = 32
  -- We need to prove:
  -- current_cats + additional_cats = 43
  sorry

end total_cats_l103_103207


namespace value_of_expression_l103_103187

variable {a b c d e f : ℝ}

theorem value_of_expression :
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 1000 →
  d * e * f = 250 →
  (a * f) / (c * d) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_expression_l103_103187


namespace minimum_weight_of_grass_seed_l103_103459

-- Definitions of cost and weights
def price_5_pound_bag : ℝ := 13.85
def price_10_pound_bag : ℝ := 20.43
def price_25_pound_bag : ℝ := 32.20
def max_weight : ℝ := 80
def min_cost : ℝ := 98.68

-- Lean proposition to prove the minimum weight given the conditions
theorem minimum_weight_of_grass_seed (w : ℝ) :
  w = 75 ↔ (w ≤ max_weight ∧
            ∃ (n5 n10 n25 : ℕ), 
              w = 5 * n5 + 10 * n10 + 25 * n25 ∧
              min_cost ≤ n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ∧
              n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ≤ min_cost) := 
by
  sorry

end minimum_weight_of_grass_seed_l103_103459


namespace gcd_values_count_l103_103201

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 392) : ∃ d, d = 11 := 
sorry

end gcd_values_count_l103_103201


namespace ratio_equality_l103_103508

variable (a b : ℝ)

theorem ratio_equality (h : a / b = 4 / 3) : (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
sorry

end ratio_equality_l103_103508


namespace three_segments_form_triangle_l103_103013

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem three_segments_form_triangle :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 2 4 6 ∧
  ¬ can_form_triangle 2 2 4 ∧
    can_form_triangle 6 6 6 :=
by
  repeat {sorry}

end three_segments_form_triangle_l103_103013


namespace relay_team_orderings_l103_103450

theorem relay_team_orderings (Jordan Mike Friend1 Friend2 Friend3 : Type) :
  ∃ n : ℕ, n = 12 :=
by
  -- Define the team members
  let team : List Type := [Jordan, Mike, Friend1, Friend2, Friend3]
  
  -- Define the number of ways to choose the 4th and 5th runners
  let ways_choose_45 := 2
  
  -- Define the number of ways to order the first 3 runners
  let ways_order_123 := Nat.factorial 3
  
  -- Calculate the total number of ways
  let total_ways := ways_choose_45 * ways_order_123
  
  -- The total ways should be 12
  use total_ways
  have h : total_ways = 12
  sorry
  exact h

end relay_team_orderings_l103_103450


namespace mean_of_xyz_l103_103011

theorem mean_of_xyz (a b c d e f g x y z : ℝ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 48)
  (h2 : (a + b + c + d + e + f + g + x + y + z) / 10 = 55) :
  (x + y + z) / 3 = 71.33333333333333 :=
by
  sorry

end mean_of_xyz_l103_103011


namespace max_cables_191_l103_103570

/-- 
  There are 30 employees: 20 with brand A computers and 10 with brand B computers.
  Cables can only connect a brand A computer to a brand B computer.
  Employees can communicate with each other if their computers are directly connected by a cable 
  or by relaying messages through a series of connected computers.
  The maximum possible number of cables used to ensure every employee can communicate with each other
  is 191.
-/
theorem max_cables_191 (A B : ℕ) (hA : A = 20) (hB : B = 10) : 
  ∃ (max_cables : ℕ), max_cables = 191 ∧ 
  (∀ (i j : ℕ), (i ≤ A ∧ j ≤ B) → (i = A ∨ j = B) → i * j ≤ max_cables) := 
sorry

end max_cables_191_l103_103570


namespace sally_reads_10_pages_on_weekdays_l103_103045

def sallyReadsOnWeekdays (x : ℕ) (total_pages : ℕ) (weekdays : ℕ) (weekend_days : ℕ) (weekend_pages : ℕ) : Prop :=
  (weekdays + weekend_days * weekend_pages = total_pages) → (weekdays * x = total_pages - weekend_days * weekend_pages)

theorem sally_reads_10_pages_on_weekdays :
  sallyReadsOnWeekdays 10 180 10 4 20 :=
by
  intros h
  sorry  -- proof to be filled in

end sally_reads_10_pages_on_weekdays_l103_103045


namespace smallest_k_l103_103141

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) : (∀ z : ℂ, z ≠ 0 → f z ∣ z^k - 1) ↔ k = 40 :=
by sorry

end smallest_k_l103_103141


namespace find_larger_number_of_two_l103_103016

theorem find_larger_number_of_two (A B : ℕ) (hcf lcm : ℕ) (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 13)
  (h_factor2 : factor2 = 16)
  (h_lcm : lcm = hcf * factor1 * factor2)
  (h_A : A = hcf * m ∧ m = factor1)
  (h_B : B = hcf * n ∧ n = factor2):
  max A B = 368 := by
  sorry

end find_larger_number_of_two_l103_103016


namespace johns_age_is_25_l103_103219

variable (JohnAge DadAge SisterAge : ℕ)

theorem johns_age_is_25
    (h1 : JohnAge = DadAge - 30)
    (h2 : JohnAge + DadAge = 80)
    (h3 : SisterAge = JohnAge - 5) :
    JohnAge = 25 := 
sorry

end johns_age_is_25_l103_103219


namespace total_days_to_finish_tea_and_coffee_l103_103831

-- Define the given conditions formally before expressing the theorem
def drinks_coffee_together (days : ℕ) : Prop := days = 10
def drinks_coffee_alone_A (days : ℕ) : Prop := days = 12
def drinks_tea_together (days : ℕ) : Prop := days = 12
def drinks_tea_alone_B (days : ℕ) : Prop := days = 20

-- The goal is to prove that A and B together finish a pound of tea and a can of coffee in 35 days
theorem total_days_to_finish_tea_and_coffee : 
  ∃ days : ℕ, 
    drinks_coffee_together 10 ∧ 
    drinks_coffee_alone_A 12 ∧ 
    drinks_tea_together 12 ∧ 
    drinks_tea_alone_B 20 ∧ 
    days = 35 :=
by
  sorry

end total_days_to_finish_tea_and_coffee_l103_103831


namespace decagon_not_divided_properly_l103_103005

theorem decagon_not_divided_properly :
  ∀ (n m : ℕ),
  (∃ black white : Finset ℕ, ∀ b ∈ black, ∀ w ∈ white,
    (b + w = 10) ∧ (b % 3 = 0) ∧ (w % 3 = 0)) →
  n - m = 10 → (n % 3 = 0) ∧ (m % 3 = 0) → 10 % 3 = 0 → False :=
by
  sorry

end decagon_not_divided_properly_l103_103005


namespace painters_needed_days_l103_103506

-- Let P be the total work required in painter-work-days
def total_painter_work_days : ℕ := 5

-- Let E be the effective number of workers with advanced tools
def effective_workers : ℕ := 4

-- Define the number of days, we need to prove this equals 1.25
def days_to_complete_work (P E : ℕ) : ℚ := P / E

-- The main theorem to prove: for total_painter_work_days and effective_workers, the days to complete the work is 1.25
theorem painters_needed_days :
  days_to_complete_work total_painter_work_days effective_workers = 5 / 4 :=
by
  sorry

end painters_needed_days_l103_103506


namespace profit_percentage_is_12_36_l103_103382

noncomputable def calc_profit_percentage (SP CP : ℝ) : ℝ :=
  let Profit := SP - CP
  (Profit / CP) * 100

theorem profit_percentage_is_12_36
  (SP : ℝ) (h1 : SP = 100)
  (CP : ℝ) (h2 : CP = 0.89 * SP) :
  calc_profit_percentage SP CP = 12.36 :=
by
  sorry

end profit_percentage_is_12_36_l103_103382


namespace find_value_of_fraction_l103_103516

theorem find_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > x) (h : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 :=
by
  sorry

end find_value_of_fraction_l103_103516


namespace find_value_of_A_l103_103983

theorem find_value_of_A (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end find_value_of_A_l103_103983


namespace barbata_interest_rate_l103_103108

theorem barbata_interest_rate (r : ℝ) : 
  let initial_investment := 2800
  let additional_investment := 1400
  let total_investment := initial_investment + additional_investment
  let annual_income := 0.06 * total_investment
  let additional_interest_rate := 0.08
  let income_from_initial := initial_investment * r
  let income_from_additional := additional_investment * additional_interest_rate
  income_from_initial + income_from_additional = annual_income → 
  r = 0.05 :=
by
  intros
  sorry

end barbata_interest_rate_l103_103108


namespace largest_valid_number_l103_103765

/-
Problem: 
What is the largest number, all of whose digits are 3, 2, or 4 whose digits add up to 16?

We prove that 4432 is the largest such number.
-/

def digits := [3, 2, 4]

def sum_of_digits (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 2 ∨ d = 4

def generate_number (l : List ℕ) : ℕ :=
  l.foldl (λ acc d => acc * 10 + d) 0

theorem largest_valid_number : 
  ∃ l : List ℕ, (∀ d ∈ l, is_valid_digit d) ∧ sum_of_digits l = 16 ∧ generate_number l = 4432 :=
  sorry

end largest_valid_number_l103_103765


namespace car_gasoline_tank_capacity_l103_103992

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

end car_gasoline_tank_capacity_l103_103992


namespace probability_A_and_B_selected_l103_103605

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l103_103605


namespace jeans_price_increase_l103_103777

theorem jeans_price_increase (M R C : ℝ) (hM : M = 100) 
  (hR : R = M * 1.4)
  (hC : C = R * 1.1) : 
  (C - M) / M * 100 = 54 :=
by
  sorry

end jeans_price_increase_l103_103777


namespace find_k_for_maximum_value_l103_103152

theorem find_k_for_maximum_value (k : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 2 → k * x^2 + 2 * k * x + 1 ≤ 5) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ k * x^2 + 2 * k * x + 1 = 5) ↔
  k = 1 / 2 ∨ k = -4 :=
by
  sorry

end find_k_for_maximum_value_l103_103152


namespace jonathan_daily_calories_l103_103742

theorem jonathan_daily_calories (C : ℕ) (daily_burn weekly_deficit extra_calories total_burn : ℕ) 
  (h1 : daily_burn = 3000) 
  (h2 : weekly_deficit = 2500) 
  (h3 : extra_calories = 1000) 
  (h4 : total_burn = 7 * daily_burn) 
  (h5 : total_burn - weekly_deficit = 7 * C + extra_calories) :
  C = 2500 :=
by 
  sorry

end jonathan_daily_calories_l103_103742


namespace brother_highlighters_spent_l103_103254

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end brother_highlighters_spent_l103_103254


namespace remainder_of_22_divided_by_3_l103_103168

theorem remainder_of_22_divided_by_3 : ∃ (r : ℕ), 22 = 3 * 7 + r ∧ r = 1 := by
  sorry

end remainder_of_22_divided_by_3_l103_103168


namespace calculate_a_minus_b_l103_103615

theorem calculate_a_minus_b (a b c : ℝ) (h1 : a - b - c = 3) (h2 : a - b + c = 11) : a - b = 7 :=
by 
  -- The proof would be fleshed out here.
  sorry

end calculate_a_minus_b_l103_103615


namespace profit_per_meter_is_35_l103_103482

-- defining the conditions
def meters_sold : ℕ := 85
def selling_price : ℕ := 8925
def cost_price_per_meter : ℕ := 70
def total_cost_price := cost_price_per_meter * meters_sold
def total_selling_price := selling_price
def total_profit := total_selling_price - total_cost_price
def profit_per_meter := total_profit / meters_sold

-- Theorem stating the profit per meter of cloth
theorem profit_per_meter_is_35 : profit_per_meter = 35 := 
by
  sorry

end profit_per_meter_is_35_l103_103482


namespace hallie_read_pages_third_day_more_than_second_day_l103_103388

theorem hallie_read_pages_third_day_more_than_second_day :
  ∀ (d1 d2 d3 d4 : ℕ),
  d1 = 63 →
  d2 = 2 * d1 →
  d4 = 29 →
  d1 + d2 + d3 + d4 = 354 →
  (d3 - d2) = 10 :=
by
  intros d1 d2 d3 d4 h1 h2 h4 h_sum
  sorry

end hallie_read_pages_third_day_more_than_second_day_l103_103388


namespace red_marbles_l103_103014

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end red_marbles_l103_103014


namespace problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l103_103814

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_part_1_solution_set_of_f_when_a_is_3 :
  {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x 3 ≤ 6} :=
by
  sorry

def g (x : ℝ) : ℝ := abs (2 * x - 3)

theorem problem_part_2_range_of_a :
  {a : ℝ | 4 ≤ a} = {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 5} :=
by
  sorry

end problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l103_103814


namespace pentagon_angles_l103_103784

theorem pentagon_angles (M T H A S : ℝ) 
  (h1 : M = T) 
  (h2 : T = H) 
  (h3 : A + S = 180) 
  (h4 : M + A + T + H + S = 540) : 
  H = 120 := 
by 
  -- The proof would be inserted here.
  sorry

end pentagon_angles_l103_103784


namespace find_num_tables_l103_103733

-- Definitions based on conditions
def num_students_in_class : ℕ := 47
def num_girls_bathroom : ℕ := 3
def num_students_canteen : ℕ := 3 * 3
def num_students_new_groups : ℕ := 2 * 4
def num_students_exchange : ℕ := 3 * 3 + 3 * 3 + 3 * 3

-- Calculation of the number of tables (corresponding to the answer)
def num_missing_students : ℕ := num_girls_bathroom + num_students_canteen + num_students_new_groups + num_students_exchange

def num_students_currently_in_class : ℕ := num_students_in_class - num_missing_students
def students_per_table : ℕ := 3

def num_tables : ℕ := num_students_currently_in_class / students_per_table

-- The theorem we want to prove
theorem find_num_tables : num_tables = 6 := by
  -- Proof steps would go here
  sorry

end find_num_tables_l103_103733


namespace dividend_expression_l103_103091

theorem dividend_expression 
  (D d q r P : ℕ)
  (hq_square : ∃ k, q = k^2)
  (hd_expr1 : d = 3 * r + 2)
  (hd_expr2 : d = 5 * q)
  (hr_val : r = 6)
  (hD_expr : D = d * q + r)
  (hP_prime : Prime P)
  (hP_div_D : P ∣ D)
  (hP_factor : P = 2 ∨ P = 43) :
  D = 86 := 
sorry

end dividend_expression_l103_103091


namespace a3_mul_a7_eq_36_l103_103901

-- Definition of a geometric sequence term
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions
def a (n : ℕ) : ℤ := sorry  -- Placeholder for the geometric sequence

axiom a5_eq_6 : a 5 = 6  -- Given that a_5 = 6

axiom geo_seq : geometric_sequence a  -- The sequence is geometric

-- Problem statement: Prove that a_3 * a_7 = 36
theorem a3_mul_a7_eq_36 : a 3 * a 7 = 36 :=
  sorry

end a3_mul_a7_eq_36_l103_103901


namespace remainder_two_when_divided_by_3_l103_103437

-- Define the main theorem stating that for any positive integer n,
-- n^3 + 3/2 * n^2 + 1/2 * n - 1 leaves a remainder of 2 when divided by 3.

theorem remainder_two_when_divided_by_3 (n : ℕ) (h : n > 0) : 
  (n^3 + (3 / 2) * n^2 + (1 / 2) * n - 1) % 3 = 2 := 
sorry

end remainder_two_when_divided_by_3_l103_103437


namespace sum_of_two_numbers_l103_103343

theorem sum_of_two_numbers :
  ∃ x y : ℝ, (x * y = 9375 ∧ y / x = 15) ∧ (x + y = 400) :=
by
  sorry

end sum_of_two_numbers_l103_103343


namespace smallest_b_value_l103_103613

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end smallest_b_value_l103_103613


namespace find_b_l103_103649

def perpendicular_vectors (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_b (b : ℝ) :
  perpendicular_vectors ⟨-5, 11⟩ ⟨b, 3⟩ →
  b = 33 / 5 :=
by
  sorry

end find_b_l103_103649


namespace radius_of_congruent_spheres_in_cone_l103_103127

noncomputable def radius_of_congruent_spheres (base_radius height : ℝ) : ℝ := 
  let slant_height := Real.sqrt (height^2 + base_radius^2)
  let r := (4 : ℝ) / (10 + 4) * slant_height
  r

theorem radius_of_congruent_spheres_in_cone :
  radius_of_congruent_spheres 4 10 = 4 * Real.sqrt 29 / 7 := by
  sorry

end radius_of_congruent_spheres_in_cone_l103_103127


namespace smallest_abundant_number_not_multiple_of_10_l103_103539

-- Definition of proper divisors of a number n
def properDivisors (n : ℕ) : List ℕ := 
  (List.range n).filter (λ d => d > 0 ∧ n % d = 0)

-- Definition of an abundant number
def isAbundant (n : ℕ) : Prop := 
  (properDivisors n).sum > n

-- Definition of not being a multiple of 10
def notMultipleOf10 (n : ℕ) : Prop := 
  n % 10 ≠ 0

-- Statement to prove
theorem smallest_abundant_number_not_multiple_of_10 :
  ∃ n, isAbundant n ∧ notMultipleOf10 n ∧ ∀ m, (isAbundant m ∧ notMultipleOf10 m) → n ≤ m :=
by
  sorry

end smallest_abundant_number_not_multiple_of_10_l103_103539


namespace initial_percentage_of_alcohol_l103_103446

variable (P : ℝ)
variables (x y : ℝ) (initial_percent replacement_percent replaced_quantity final_percent : ℝ)

def whisky_problem :=
  initial_percent = P ∧
  replacement_percent = 0.19 ∧
  replaced_quantity = 2/3 ∧
  final_percent = 0.26 ∧
  (P * (1 - replaced_quantity) + replacement_percent * replaced_quantity = final_percent)

theorem initial_percentage_of_alcohol :
  whisky_problem P 0.40 0.19 (2/3) 0.26 := sorry

end initial_percentage_of_alcohol_l103_103446


namespace parallel_and_perpendicular_implies_perpendicular_l103_103680

variables (l : Line) (α β : Plane)

axiom line_parallel_plane (l : Line) (π : Plane) : Prop
axiom line_perpendicular_plane (l : Line) (π : Plane) : Prop
axiom planes_are_perpendicular (π₁ π₂ : Plane) : Prop

theorem parallel_and_perpendicular_implies_perpendicular
  (h1 : line_parallel_plane l α)
  (h2 : line_perpendicular_plane l β) 
  : planes_are_perpendicular α β :=
sorry

end parallel_and_perpendicular_implies_perpendicular_l103_103680


namespace angle_rotation_l103_103995

theorem angle_rotation (α : ℝ) (β : ℝ) (k : ℤ) :
  (∃ k' : ℤ, α + 30 = 120 + 360 * k') →
  (β = 360 * k + 90) ↔ (∃ k'' : ℤ, β = 360 * k'' + α) :=
by
  sorry

end angle_rotation_l103_103995


namespace apples_bought_is_28_l103_103223

-- Define the initial number of apples, number of apples used, and total number of apples after buying more
def initial_apples : ℕ := 38
def apples_used : ℕ := 20
def total_apples_after_buying : ℕ := 46

-- State the theorem: the number of apples bought is 28
theorem apples_bought_is_28 : (total_apples_after_buying - (initial_apples - apples_used)) = 28 := 
by sorry

end apples_bought_is_28_l103_103223


namespace restaurant_sales_decrease_l103_103466

-- Conditions
variable (Sales_August : ℝ := 42000)
variable (Sales_October : ℝ := 27000)
variable (a : ℝ) -- monthly average decrease rate as a decimal

-- Theorem statement
theorem restaurant_sales_decrease :
  42 * (1 - a)^2 = 27 := sorry

end restaurant_sales_decrease_l103_103466


namespace marathon_problem_l103_103752

-- Defining the given conditions in the problem.
def john_position_right := 28
def john_position_left := 42
def mike_ahead := 10

-- Define total participants.
def total_participants := john_position_right + john_position_left - 1

-- Define Mike's positions based on the given conditions.
def mike_position_left := john_position_left - mike_ahead
def mike_position_right := john_position_right - mike_ahead

-- Proposition combining all the facts.
theorem marathon_problem :
  total_participants = 69 ∧ mike_position_left = 32 ∧ mike_position_right = 18 := by 
     sorry

end marathon_problem_l103_103752


namespace mingming_actual_height_l103_103809

def mingming_height (h : ℝ) : Prop := 1.495 ≤ h ∧ h < 1.505

theorem mingming_actual_height : ∃ α : ℝ, mingming_height α :=
by
  use 1.50
  sorry

end mingming_actual_height_l103_103809


namespace sum_of_fractions_eq_sum_of_cubes_l103_103999

theorem sum_of_fractions_eq_sum_of_cubes (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  ( (x-1)*(x+1) / (x*(x-1) + 1) + (2*(0.5-x)) / (x*(1-x) -1) ) = 
  ( ((x-1)*(x+1) / (x*(x-1) + 1))^3 + ((2*(0.5-x)) / (x*(1-x) -1))^3 ) :=
sorry

end sum_of_fractions_eq_sum_of_cubes_l103_103999


namespace sufficient_and_necessary_condition_l103_103214

def isMonotonicallyIncreasing {R : Type _} [LinearOrderedField R] (f : R → R) :=
  ∀ x y, x < y → f x < f y

def fx {R : Type _} [LinearOrderedField R] (x m : R) :=
  x^3 + 2*x^2 + m*x + 1

theorem sufficient_and_necessary_condition (m : ℝ) :
  (isMonotonicallyIncreasing (λ x => fx x m) ↔ m ≥ 4/3) :=
  sorry

end sufficient_and_necessary_condition_l103_103214


namespace average_speed_of_car_l103_103526

theorem average_speed_of_car (time : ℝ) (distance : ℝ) (h_time : time = 4.5) (h_distance : distance = 360) : 
  distance / time = 80 :=
by
  sorry

end average_speed_of_car_l103_103526


namespace player1_points_after_13_rotations_l103_103828

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end player1_points_after_13_rotations_l103_103828


namespace odds_of_picking_blue_marble_l103_103662

theorem odds_of_picking_blue_marble :
  ∀ (total_marbles yellow_marbles : ℕ)
  (h1 : total_marbles = 60)
  (h2 : yellow_marbles = 20)
  (green_marbles : ℕ)
  (h3 : green_marbles = yellow_marbles / 2)
  (remaining_marbles : ℕ)
  (h4 : remaining_marbles = total_marbles - yellow_marbles - green_marbles)
  (blue_marbles : ℕ)
  (h5 : blue_marbles = remaining_marbles / 2),
  (blue_marbles / total_marbles : ℚ) * 100 = 25 :=
by
  intros total_marbles yellow_marbles h1 h2 green_marbles h3 remaining_marbles h4 blue_marbles h5
  sorry

end odds_of_picking_blue_marble_l103_103662


namespace expected_worth_coin_flip_l103_103430

noncomputable def expected_worth : ℝ := 
  (1 / 3) * 6 + (2 / 3) * (-2) - 1

theorem expected_worth_coin_flip : expected_worth = -0.33 := 
by 
  unfold expected_worth
  norm_num
  sorry

end expected_worth_coin_flip_l103_103430


namespace smallest_solution_proof_l103_103087

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 2)) ∧ 
  (∀ y : ℝ, 1 / (y - 1) + 1 / (y - 5) = 4 / (y - 2) → y ≥ x)

theorem smallest_solution_proof : smallest_solution ( (7 - Real.sqrt 33) / 2 ) :=
sorry

end smallest_solution_proof_l103_103087


namespace specific_divisors_count_l103_103783

-- Declare the value of n
def n : ℕ := (2^40) * (3^25) * (5^10)

-- Definition to count the number of positive divisors of a number less than n that don't divide n.
def count_specific_divisors (n : ℕ) : ℕ :=
sorry  -- This would be the function implementation

-- Lean statement to assert the number of such divisors
theorem specific_divisors_count : 
  count_specific_divisors n = 31514 :=
sorry

end specific_divisors_count_l103_103783


namespace addition_belongs_to_Q_l103_103151

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

theorem addition_belongs_to_Q (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end addition_belongs_to_Q_l103_103151


namespace roots_of_quadratic_l103_103501

theorem roots_of_quadratic (x : ℝ) : (x - 3) ^ 2 = 25 ↔ (x = 8 ∨ x = -2) :=
by sorry

end roots_of_quadratic_l103_103501


namespace units_digit_of_G_1000_l103_103529

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_of_G_1000 : (G 1000) % 10 = 2 := 
  sorry

end units_digit_of_G_1000_l103_103529


namespace typing_speed_ratio_l103_103740

theorem typing_speed_ratio (T M : ℝ) 
  (h1 : T + M = 12) 
  (h2 : T + 1.25 * M = 14) : 
  M / T = 2 :=
by 
  -- The proof will go here
  sorry

end typing_speed_ratio_l103_103740


namespace part1_part2_l103_103262

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end part1_part2_l103_103262


namespace length_of_goods_train_l103_103340

theorem length_of_goods_train (speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) (length_of_train : ℝ) :
  speed_kmph = 96 → platform_length = 360 → crossing_time = 32 → length_of_train = (26.67 * 32 - 360) :=
by
  sorry

end length_of_goods_train_l103_103340


namespace solve_system_equations_l103_103310

theorem solve_system_equations (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 + z^2 = b^2) → 
  b = 0 ∧ (∃ t, (x = 0 ∧ y = t ∧ z = -t) ∨ 
                (x = t ∧ y = 0 ∧ z = -t) ∨ 
                (x = -t ∧ y = t ∧ z = 0)) :=
by
  sorry -- Proof to be provided

end solve_system_equations_l103_103310


namespace denominator_is_five_l103_103538

-- Define the conditions
variables (n d : ℕ)
axiom h1 : d = n - 4
axiom h2 : n + 6 = 3 * d

-- The theorem that needs to be proven
theorem denominator_is_five : d = 5 :=
by
  sorry

end denominator_is_five_l103_103538


namespace arithmetic_sequence_length_l103_103277

theorem arithmetic_sequence_length :
  ∀ (a₁ d an : ℤ), a₁ = -5 → d = 3 → an = 40 → (∃ n : ℕ, an = a₁ + (n - 1) * d ∧ n = 16) :=
by
  intros a₁ d an h₁ hd han
  sorry

end arithmetic_sequence_length_l103_103277


namespace monica_expected_winnings_l103_103981

def monica_die_winnings : List ℤ := [2, 3, 5, 7, 0, 0, 0, -4]

def expected_value (values : List ℤ) : ℚ :=
  (List.sum values) / (values.length : ℚ)

theorem monica_expected_winnings :
  expected_value monica_die_winnings = 1.625 := by
  sorry

end monica_expected_winnings_l103_103981


namespace fifty_percent_greater_l103_103846

theorem fifty_percent_greater (x : ℕ) (h : x = 88 + (88 / 2)) : x = 132 := 
by {
  sorry
}

end fifty_percent_greater_l103_103846


namespace trivia_game_points_per_question_l103_103841

theorem trivia_game_points_per_question (correct_first_half correct_second_half total_score points_per_question : ℕ) 
  (h1 : correct_first_half = 5) 
  (h2 : correct_second_half = 5) 
  (h3 : total_score = 50) 
  (h4 : correct_first_half + correct_second_half = 10) : 
  points_per_question = 5 :=
by 
  sorry

end trivia_game_points_per_question_l103_103841


namespace new_average_l103_103993

open Nat

-- The Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Sum of the first 35 Fibonacci numbers
def sum_fibonacci_first_35 : ℕ :=
  (List.range 35).map fibonacci |>.sum -- or critical to use: List.foldr (λ x acc, fibonacci x + acc) 0 (List.range 35) 

theorem new_average (n : ℕ) (avg : ℕ) (Fib_Sum : ℕ) 
  (h₁ : n = 35) 
  (h₂ : avg = 25) 
  (h₃ : Fib_Sum = sum_fibonacci_first_35) : 
  (25 * Fib_Sum / 35) = avg * (sum_fibonacci_first_35) / n := 
by 
  sorry

end new_average_l103_103993


namespace problem_1_l103_103589

theorem problem_1 (a b : ℝ) (h : b < a ∧ a < 0) : 
  (a + b < a * b) ∧ (¬ (abs a > abs b)) ∧ (¬ (1 / b > 1 / a ∧ 1 / a > 0)) ∧ (¬ (b / a + a / b > 2)) := sorry

end problem_1_l103_103589


namespace domain_of_v_l103_103875

noncomputable def v (x : ℝ) : ℝ := 1 / (x^(1/3))

theorem domain_of_v : {x : ℝ | ∃ y, y = v x} = {x : ℝ | x ≠ 0} := by
  sorry

end domain_of_v_l103_103875


namespace ilya_defeats_dragon_l103_103007

noncomputable def prob_defeat : ℝ := 1 / 4 * 2 + 1 / 3 * 1 + 5 / 12 * 0

theorem ilya_defeats_dragon : prob_defeat = 1 := sorry

end ilya_defeats_dragon_l103_103007


namespace min_value_fraction_l103_103316

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b^2 - 4 * a * c ≤ 0) :
  (a + b + c) / (2 * a) ≥ 2 :=
  sorry

end min_value_fraction_l103_103316


namespace correct_quadratic_graph_l103_103713

theorem correct_quadratic_graph (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (-b / (2 * a) > 0) ∧ (c < 0) :=
by
  sorry

end correct_quadratic_graph_l103_103713


namespace amy_small_gardens_l103_103243

-- Define the initial number of seeds
def initial_seeds : ℕ := 101

-- Define the number of seeds planted in the big garden
def big_garden_seeds : ℕ := 47

-- Define the number of seeds planted in each small garden
def seeds_per_small_garden : ℕ := 6

-- Define the number of small gardens
def number_of_small_gardens : ℕ := (initial_seeds - big_garden_seeds) / seeds_per_small_garden

-- Prove that Amy has 9 small gardens
theorem amy_small_gardens : number_of_small_gardens = 9 := by
  sorry

end amy_small_gardens_l103_103243


namespace value_of_x_squared_y_plus_xy_squared_l103_103406

variable {R : Type} [CommRing R] (x y : R)

-- Given conditions
def cond1 : Prop := x + y = 3
def cond2 : Prop := x * y = 2

-- The main theorem to prove
theorem value_of_x_squared_y_plus_xy_squared (h1 : cond1 x y) (h2 : cond2 x y) : x^2 * y + x * y^2 = 6 :=
by
  sorry

end value_of_x_squared_y_plus_xy_squared_l103_103406


namespace find_rate_of_current_l103_103512

-- Define the conditions
def speed_in_still_water (speed : ℝ) : Prop := speed = 15
def distance_downstream (distance : ℝ) : Prop := distance = 7.2
def time_in_hours (time : ℝ) : Prop := time = 0.4

-- Define the effective speed downstream
def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ := boat_speed + current_speed

-- Define rate of current
def rate_of_current (current_speed : ℝ) : Prop :=
  ∃ (c : ℝ), effective_speed_downstream 15 c * 0.4 = 7.2 ∧ c = current_speed

-- The theorem stating the proof problem
theorem find_rate_of_current : rate_of_current 3 :=
by
  sorry

end find_rate_of_current_l103_103512


namespace find_n_l103_103021

-- Definitions based on conditions
variable (n : ℕ)  -- number of persons
variable (A : Fin n → Finset (Fin n))  -- acquaintance relation, specified as a set of neighbors for each person
-- Condition 1: Each person is acquainted with exactly 8 others
def acquaintances := ∀ i : Fin n, (A i).card = 8
-- Condition 2: Any two acquainted persons have exactly 4 common acquaintances
def common_acquaintances_adj := ∀ i j : Fin n, i ≠ j → j ∈ (A i) → (A i ∩ A j).card = 4
-- Condition 3: Any two non-acquainted persons have exactly 2 common acquaintances
def common_acquaintances_non_adj := ∀ i j : Fin n, i ≠ j → j ∉ (A i) → (A i ∩ A j).card = 2

-- Statement to prove
theorem find_n (h1 : acquaintances n A) (h2 : common_acquaintances_adj n A) (h3 : common_acquaintances_non_adj n A) :
  n = 21 := 
sorry

end find_n_l103_103021


namespace proof_problem_l103_103921

def h (x : ℝ) : ℝ := 2 * x + 4
def k (x : ℝ) : ℝ := 4 * x + 6

theorem proof_problem : h (k 3) - k (h 3) = -6 :=
by
  sorry

end proof_problem_l103_103921


namespace inequality_and_equality_condition_l103_103913

theorem inequality_and_equality_condition (x : ℝ)
  (h : x ∈ (Set.Iio 0 ∪ Set.Ioi 0)) :
  max 0 (Real.log (|x|)) ≥ 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)
  ∧ (max 0 (Real.log (|x|)) = 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
      x = (Real.sqrt 5 - 1) / 2 ∨ 
      x = -(Real.sqrt 5 - 1) / 2 ∨ 
      x = (Real.sqrt 5 + 1) / 2 ∨ 
      x = -(Real.sqrt 5 + 1) / 2) :=
by
  sorry

end inequality_and_equality_condition_l103_103913


namespace slope_of_line_through_midpoints_l103_103079

theorem slope_of_line_through_midpoints (A B C D : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 4)) (hC : C = (4, 1)) (hD : D = (7, 4)) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  (N.2 - M.2) / (N.1 - M.1) = 0 := by
  sorry

end slope_of_line_through_midpoints_l103_103079


namespace number_one_fourth_more_than_it_is_30_percent_less_than_80_l103_103530

theorem number_one_fourth_more_than_it_is_30_percent_less_than_80 :
    ∃ (n : ℝ), (5 / 4) * n = 56 ∧ n = 45 :=
by
  sorry

end number_one_fourth_more_than_it_is_30_percent_less_than_80_l103_103530


namespace will_has_123_pieces_of_candy_l103_103140

def initial_candy_pieces (chocolate_boxes mint_boxes caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  chocolate_boxes * pieces_per_chocolate_box + mint_boxes * pieces_per_mint_box + caramel_boxes * pieces_per_caramel_box

def given_away_candy_pieces (given_chocolate_boxes given_mint_boxes given_caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  given_chocolate_boxes * pieces_per_chocolate_box + given_mint_boxes * pieces_per_mint_box + given_caramel_boxes * pieces_per_caramel_box

def remaining_candy : ℕ :=
  let initial := initial_candy_pieces 7 5 4 12 15 10
  let given_away := given_away_candy_pieces 3 2 1 12 15 10
  initial - given_away

theorem will_has_123_pieces_of_candy : remaining_candy = 123 :=
by
  -- Proof goes here
  sorry

end will_has_123_pieces_of_candy_l103_103140


namespace harry_total_payment_in_silvers_l103_103720

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end harry_total_payment_in_silvers_l103_103720


namespace point_above_line_l103_103800

theorem point_above_line (a : ℝ) : 3 * (-3) - 2 * (-1) - a < 0 ↔ a > -7 :=
by sorry

end point_above_line_l103_103800


namespace complex_number_value_l103_103834

-- Declare the imaginary unit 'i'
noncomputable def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_number_value : (i / ((1 - i) ^ 2)) = -1/2 := 
by
  sorry

end complex_number_value_l103_103834


namespace find_marked_price_l103_103739

theorem find_marked_price (cp : ℝ) (d : ℝ) (p : ℝ) (x : ℝ) (h1 : cp = 80) (h2 : d = 0.3) (h3 : p = 0.05) :
  (1 - d) * x = cp * (1 + p) → x = 120 :=
by
  sorry

end find_marked_price_l103_103739


namespace total_cost_calculation_l103_103898

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l103_103898


namespace horner_eval_at_neg2_l103_103114

noncomputable def f (x : ℝ) : ℝ := x^5 - 3 * x^3 - 6 * x^2 + x - 1

theorem horner_eval_at_neg2 : f (-2) = -35 :=
by
  sorry

end horner_eval_at_neg2_l103_103114


namespace polynomial_roots_l103_103130

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end polynomial_roots_l103_103130


namespace range_of_x_range_of_a_l103_103369

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_of_x (h1 : a = 1) (h2 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem range_of_a (h : ∀ x, p x a → q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l103_103369


namespace number_of_new_bottle_caps_l103_103196

def threw_away := 6
def total_bottle_caps_now := 60
def found_more_bottle_caps := 44

theorem number_of_new_bottle_caps (N : ℕ) (h1 : N = threw_away + found_more_bottle_caps) : N = 50 :=
sorry

end number_of_new_bottle_caps_l103_103196


namespace triangle_problem_l103_103681

noncomputable def triangle_sum : Real := sorry

theorem triangle_problem
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (hA : A = π / 6) -- A = 30 degrees
  (h_a : a = Real.sqrt 3) -- a = √3
  (h_law_of_sines : ∀ (x : ℝ), x = 2 * triangle_sum * Real.sin x) -- Law of Sines
  (h_sin_30 : Real.sin (π / 6) = 1 / 2) -- sin 30 degrees = 1/2
  : (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) 
  = 2 * Real.sqrt 3 := sorry

end triangle_problem_l103_103681


namespace problem_x_sq_plus_y_sq_l103_103661

variables {x y : ℝ}

theorem problem_x_sq_plus_y_sq (h₁ : x - y = 12) (h₂ : x * y = 9) : x^2 + y^2 = 162 := 
sorry

end problem_x_sq_plus_y_sq_l103_103661


namespace part_a_part_b_l103_103086

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ x y => x + y) 0

-- Part A: There exists a sequence of 158 consecutive integers where the sum of digits is not divisible by 17
theorem part_a : ∃ (n : ℕ), ∀ (k : ℕ), k < 158 → sum_of_digits (n + k) % 17 ≠ 0 := by
  sorry

-- Part B: Among any 159 consecutive integers, there exists at least one integer whose sum of digits is divisible by 17
theorem part_b : ∀ (n : ℕ), ∃ (k : ℕ), k < 159 ∧ sum_of_digits (n + k) % 17 = 0 := by
  sorry

end part_a_part_b_l103_103086


namespace total_students_l103_103376

-- Definition of the conditions given in the problem
def num5 : ℕ := 12
def num6 : ℕ := 6 * num5

-- The theorem representing the mathematically equivalent proof problem
theorem total_students : num5 + num6 = 84 :=
by
  sorry

end total_students_l103_103376


namespace point_on_x_axis_l103_103902

theorem point_on_x_axis (m : ℝ) (h : (m, m - 1).snd = 0) : m = 1 :=
by
  sorry

end point_on_x_axis_l103_103902


namespace trapezoid_ratio_l103_103823

theorem trapezoid_ratio (u v : ℝ) (h1 : u > v) (h2 : (u + v) * (14 / u + 6 / v) = 40) : u / v = 7 / 3 :=
sorry

end trapezoid_ratio_l103_103823


namespace sin_2_alpha_plus_pi_by_3_l103_103354

-- Define the statement to be proved
theorem sin_2_alpha_plus_pi_by_3 (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hcos : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (2 * α + π / 3) = 24 / 25 := sorry

end sin_2_alpha_plus_pi_by_3_l103_103354


namespace find_m_l103_103492

-- Define the condition that the equation has a positive root
def hasPositiveRoot (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (2 / (x - 2) = 1 - (m / (x - 2)))

-- State the theorem
theorem find_m : ∀ m : ℝ, hasPositiveRoot m → m = -2 :=
by
  sorry

end find_m_l103_103492


namespace pairs_count_l103_103916

noncomputable def count_pairs (n : ℕ) : ℕ :=
  3^n

theorem pairs_count (A : Finset ℕ) (h : A.card = n) :
  ∃ f : Finset ℕ × Finset ℕ → Finset ℕ, ∀ B C, (B ≠ ∅ ∧ B ⊆ C ∧ C ⊆ A) → (f (B, C)).card = count_pairs n :=
sorry

end pairs_count_l103_103916


namespace expressions_equal_iff_l103_103194

theorem expressions_equal_iff (a b c: ℝ) : a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 :=
by 
  sorry

end expressions_equal_iff_l103_103194


namespace tadpoles_more_than_fish_l103_103428

def fish_initial : ℕ := 100
def tadpoles_initial := 4 * fish_initial
def snails_initial : ℕ := 150
def fish_caught : ℕ := 12
def tadpoles_to_frogs := (2 * tadpoles_initial) / 3
def snails_crawled_away : ℕ := 20

theorem tadpoles_more_than_fish :
  let fish_now : ℕ := fish_initial - fish_caught
  let tadpoles_now : ℕ := tadpoles_initial - tadpoles_to_frogs
  fish_now < tadpoles_now ∧ tadpoles_now - fish_now = 46 :=
by
  sorry

end tadpoles_more_than_fish_l103_103428


namespace folded_paper_perimeter_l103_103230

theorem folded_paper_perimeter (L W : ℝ) 
  (h1 : 2 * L + W = 34)         -- Condition 1
  (h2 : L * W = 140)            -- Condition 2
  : 2 * W + L = 38 :=           -- Goal
sorry

end folded_paper_perimeter_l103_103230


namespace line_always_passes_through_fixed_point_l103_103213

theorem line_always_passes_through_fixed_point :
  ∀ m : ℝ, (m-1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  
  -- Proof would go here
  sorry

end line_always_passes_through_fixed_point_l103_103213


namespace scientific_notation_50000000000_l103_103895

theorem scientific_notation_50000000000 :
  ∃ (a : ℝ) (n : ℤ), 50000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ (a = 5.0 ∨ a = 5) ∧ n = 10 :=
by
  sorry

end scientific_notation_50000000000_l103_103895


namespace oscar_bus_ride_length_l103_103917

/-- Oscar's bus ride to school is some distance, and Charlie's bus ride is 0.25 mile.
Oscar's bus ride is 0.5 mile longer than Charlie's. Prove that Oscar's bus ride is 0.75 mile. -/
theorem oscar_bus_ride_length (charlie_ride : ℝ) (h1 : charlie_ride = 0.25) 
  (oscar_ride : ℝ) (h2 : oscar_ride = charlie_ride + 0.5) : oscar_ride = 0.75 :=
by sorry

end oscar_bus_ride_length_l103_103917


namespace minimum_rows_l103_103315

theorem minimum_rows (n : ℕ) (C : ℕ → ℕ) (hC_bounds : ∀ i, 1 ≤ C i ∧ C i ≤ 39) 
  (hC_sum : (Finset.range n).sum C = 1990) :
  ∃ k, k = 12 ∧ ∀ (R : ℕ) (hR : R = 199), 
    ∀ (seating : ℕ → ℕ) (h_seating : ∀ i, seating i ≤ R) 
    (h_seating_capacity : (Finset.range k).sum seating = 1990),
    True := sorry

end minimum_rows_l103_103315


namespace avg_pages_hr_difference_l103_103973

noncomputable def avg_pages_hr_diff (total_pages_ryan : ℕ) (hours_ryan : ℕ) (books_brother : ℕ) (pages_per_book : ℕ) (hours_brother : ℕ) : ℚ :=
  (total_pages_ryan / hours_ryan : ℚ) - (books_brother * pages_per_book / hours_brother : ℚ)

theorem avg_pages_hr_difference :
  avg_pages_hr_diff 4200 78 15 250 90 = 12.18 :=
by
  sorry

end avg_pages_hr_difference_l103_103973


namespace domain_of_expression_l103_103505

theorem domain_of_expression (x : ℝ) :
  (1 ≤ x ∧ x < 6) ↔ (∃ y : ℝ, y = (x-1) ∧ y = (6-x) ∧ 0 ≤ y) :=
sorry

end domain_of_expression_l103_103505


namespace art_piece_future_value_multiple_l103_103088

theorem art_piece_future_value_multiple (original_price increase_in_value future_value multiple : ℕ)
  (h1 : original_price = 4000)
  (h2 : increase_in_value = 8000)
  (h3 : future_value = original_price + increase_in_value)
  (h4 : multiple = future_value / original_price) :
  multiple = 3 := 
sorry

end art_piece_future_value_multiple_l103_103088


namespace max_gcd_a_is_25_l103_103606

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 100 + n^2 + 2 * n

-- Define the gcd function
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Define the theorem to prove the maximum value of d_n as 25
theorem max_gcd_a_is_25 : ∃ n : ℕ, d n = 25 := 
sorry

end max_gcd_a_is_25_l103_103606


namespace length_of_chord_EF_l103_103616

noncomputable def chord_length (theta_1 theta_2 : ℝ) : ℝ :=
  let x_1 := 2 * Real.cos theta_1
  let y_1 := Real.sin theta_1
  let x_2 := 2 * Real.cos theta_2
  let y_2 := Real.sin theta_2
  Real.sqrt ((x_2 - x_1)^2 + (y_2 - y_1)^2)

theorem length_of_chord_EF :
  ∀ (theta_1 theta_2 : ℝ), 
  (2 * Real.cos theta_1) + (Real.sin theta_1) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_2) + (Real.sin theta_2) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_1)^2 + 4 * (Real.sin theta_1)^2 = 4 →
  (2 * Real.cos theta_2)^2 + 4 * (Real.sin theta_2)^2 = 4 →
  chord_length theta_1 theta_2 = 8 / 5 :=
by
  intros theta_1 theta_2 h1 h2 h3 h4
  sorry

end length_of_chord_EF_l103_103616


namespace integer_pairs_satisfying_equation_l103_103077

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 →
    (x = 1 ∧ y = 12) ∨ (x = 1 ∧ y = -12) ∨ 
    (x = -9 ∧ y = 12) ∨ (x = -9 ∧ y = -12) ∨ 
    (x = -4 ∧ y = 12) ∨ (x = -4 ∧ y = -12) ∨ 
    (x = 0 ∧ y = 0) ∨ (x = -8 ∧ y = 0) ∨ 
    (x = -1 ∧ y = 0) ∨ (x = -7 ∧ y = 0) :=
by sorry

end integer_pairs_satisfying_equation_l103_103077


namespace george_initial_amount_l103_103854

-- Definitions as per conditions
def cost_of_shirt : ℕ := 24
def cost_of_socks : ℕ := 11
def amount_left : ℕ := 65

-- Goal: Prove that the initial amount of money George had is 100
theorem george_initial_amount : (cost_of_shirt + cost_of_socks + amount_left) = 100 := 
by sorry

end george_initial_amount_l103_103854


namespace multiply_203_197_square_neg_699_l103_103469

theorem multiply_203_197 : 203 * 197 = 39991 := by
  sorry

theorem square_neg_699 : (-69.9)^2 = 4886.01 := by
  sorry

end multiply_203_197_square_neg_699_l103_103469


namespace ellipse_range_x_plus_y_l103_103321

/-- The problem conditions:
Given any point P(x, y) on the ellipse x^2 / 144 + y^2 / 25 = 1,
prove that the range of values for x + y is [-13, 13].
-/
theorem ellipse_range_x_plus_y (x y : ℝ) (h : (x^2 / 144) + (y^2 / 25) = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := sorry

end ellipse_range_x_plus_y_l103_103321


namespace value_of_q_l103_103177

-- Define the problem in Lean 4

variable (a d q : ℝ) (h0 : a ≠ 0)
variables (M P : Set ℝ)
variable (hM : M = {a, a + d, a + 2 * d})
variable (hP : P = {a, a * q, a * q * q})
variable (hMP : M = P)

theorem value_of_q : q = -1 :=
by
  sorry

end value_of_q_l103_103177


namespace value_of_x_l103_103716

theorem value_of_x (x : ℝ) (a : ℝ) (h1 : x ^ 2 * 8 ^ 3 / 256 = a) (h2 : a = 450) : x = 15 ∨ x = -15 := by
  sorry

end value_of_x_l103_103716


namespace isosceles_trapezoid_with_inscribed_circle_area_is_20_l103_103073

def isosceles_trapezoid_area (a b c1 c2 h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem isosceles_trapezoid_with_inscribed_circle_area_is_20
  (a b c h : ℕ)
  (ha : a = 2)
  (hb : b = 8)
  (hc : a + b = 2 * c)
  (hh : h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2) :
  isosceles_trapezoid_area a b c c h = 20 := 
by {
  sorry
}

end isosceles_trapezoid_with_inscribed_circle_area_is_20_l103_103073


namespace ratio_ad_bc_l103_103323

theorem ratio_ad_bc (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 5 * c) (h3 : c = 3 * d) : 
  (a * d) / (b * c) = 4 / 3 := 
by 
  sorry

end ratio_ad_bc_l103_103323


namespace evaluate_f_at_1_l103_103575

noncomputable def f (x : ℝ) : ℝ := 2^x + 2

theorem evaluate_f_at_1 : f 1 = 4 :=
by {
  -- proof goes here
  sorry
}

end evaluate_f_at_1_l103_103575


namespace average_speed_is_20_mph_l103_103416

-- Defining the conditions
def distance1 := 40 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Calculating total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1 -- hours
def time2 := distance2 / speed2 -- hours
def time3 := distance3 / speed3 -- hours
def total_time := time1 + time2 + time3

-- Theorem statement
theorem average_speed_is_20_mph : (total_distance / total_time) = 20 := by
  sorry

end average_speed_is_20_mph_l103_103416


namespace planes_perpendicular_of_line_conditions_l103_103507

variables (a b l : Line) (M N : Plane)

-- Definitions of lines and planes and their relations
def parallel_to_plane (a : Line) (M : Plane) : Prop := sorry
def perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry
def subset_of_plane (a : Line) (M : Plane) : Prop := sorry

-- Statement of the main theorem to be proved
theorem planes_perpendicular_of_line_conditions (a b l : Line) (M N : Plane) :
  (perpendicular_to_plane a M) → (parallel_to_plane a N) → (perpendicular_to_plane N M) :=
  by
  sorry

end planes_perpendicular_of_line_conditions_l103_103507


namespace total_puppies_count_l103_103048

def first_week_puppies : Nat := 20
def second_week_puppies : Nat := 2 * first_week_puppies / 5
def third_week_puppies : Nat := 3 * second_week_puppies / 8
def fourth_week_puppies : Nat := 2 * second_week_puppies
def fifth_week_puppies : Nat := first_week_puppies + 10
def sixth_week_puppies : Nat := 2 * third_week_puppies - 5
def seventh_week_puppies : Nat := 2 * sixth_week_puppies
def eighth_week_puppies : Nat := 5 * seventh_week_puppies / 6 / 1 -- Assuming rounding down to nearest whole number

def total_puppies : Nat :=
  first_week_puppies + second_week_puppies + third_week_puppies +
  fourth_week_puppies + fifth_week_puppies + sixth_week_puppies +
  seventh_week_puppies + eighth_week_puppies

theorem total_puppies_count : total_puppies = 81 := by
  sorry

end total_puppies_count_l103_103048


namespace roy_is_6_years_older_than_julia_l103_103231

theorem roy_is_6_years_older_than_julia :
  ∀ (R J K : ℕ) (x : ℕ), 
    R = J + x →
    R = K + x / 2 →
    R + 4 = 2 * (J + 4) →
    (R + 4) * (K + 4) = 108 →
    x = 6 :=
by
  intros R J K x h1 h2 h3 h4
  -- Proof goes here (using sorry to skip the proof)
  sorry

end roy_is_6_years_older_than_julia_l103_103231


namespace term_300_is_neg_8_l103_103638

noncomputable def geom_seq (a r : ℤ) : ℕ → ℤ
| 0       => a
| (n + 1) => r * geom_seq a r n

-- First term and second term are given as conditions.
def a1 : ℤ := 8
def a2 : ℤ := -8

-- Define the common ratio based on the conditions
def r : ℤ := a2 / a1

-- Theorem stating the 300th term is -8
theorem term_300_is_neg_8 : geom_seq a1 r 299 = -8 :=
by
  have h_r : r = -1 := by
    rw [r, a2, a1]
    norm_num
  rw [h_r]
  sorry

end term_300_is_neg_8_l103_103638


namespace production_profit_range_l103_103852

theorem production_profit_range (x : ℝ) (t : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) (h3 : 0 ≤ t) :
  (200 * (5 * x + 1 - 3 / x) ≥ 3000) → (3 ≤ x ∧ x ≤ 10) :=
sorry

end production_profit_range_l103_103852


namespace total_estate_value_l103_103115

theorem total_estate_value :
  ∃ (E : ℝ), ∀ (x : ℝ),
    (5 * x + 4 * x = (2 / 3) * E) ∧
    (E = 13.5 * x) ∧
    (wife_share = 3 * 4 * x) ∧
    (gardener_share = 600) ∧
    (nephew_share = 1000) →
    E = 2880 := 
by 
  -- Declarations
  let E : ℝ := sorry
  let x : ℝ := sorry
  
  -- Set up conditions
  -- Daughter and son share
  have c1 : 5 * x + 4 * x = (2 / 3) * E := sorry
  
  -- E expressed through x
  have c2 : E = 13.5 * x := sorry
  
  -- Wife's share
  have c3 : wife_share = 3 * (4 * x) := sorry
  
  -- Gardener's share and Nephew's share
  have c4 : gardener_share = 600 := sorry
  have c5 : nephew_share = 1000 := sorry
  
  -- Equate expressions and solve
  have eq1 : E = 21 * x + 1600 := sorry
  have eq2 : E = 2880 := sorry
  use E
  intro x
  -- Prove the equalities under the given conditions
  sorry

end total_estate_value_l103_103115


namespace value_of_expression_l103_103607

variables {x1 x2 x3 x4 x5 x6 : ℝ}

theorem value_of_expression
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 = 1)
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 = 14)
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 = 135) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 = 832 :=
by
  sorry

end value_of_expression_l103_103607


namespace max_right_angles_in_triangular_prism_l103_103856

theorem max_right_angles_in_triangular_prism 
  (n_triangles : ℕ) 
  (n_rectangles : ℕ) 
  (max_right_angles_triangle : ℕ) 
  (max_right_angles_rectangle : ℕ)
  (h1 : n_triangles = 2)
  (h2 : n_rectangles = 3)
  (h3 : max_right_angles_triangle = 1)
  (h4 : max_right_angles_rectangle = 4) : 
  (n_triangles * max_right_angles_triangle + n_rectangles * max_right_angles_rectangle = 14) :=
by
  sorry

end max_right_angles_in_triangular_prism_l103_103856


namespace find_f_at_3_l103_103543

variable (f : ℝ → ℝ)

-- Conditions
-- 1. f is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
-- 2. f(-1) = 1/2
axiom f_neg_one : f (-1) = 1 / 2
-- 3. f(x+2) = f(x) + 2 for all x
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + 2

-- The target value to prove
theorem find_f_at_3 : f 3 = 3 / 2 := by
  sorry

end find_f_at_3_l103_103543


namespace carbonate_ions_in_Al2_CO3_3_l103_103345

theorem carbonate_ions_in_Al2_CO3_3 (total_weight : ℕ) (formula : String) 
  (molecular_weight : ℕ) (ions_in_formula : String) : 
  formula = "Al2(CO3)3" → molecular_weight = 234 → ions_in_formula = "CO3" → total_weight = 3 := 
by
  intros formula_eq weight_eq ions_eq
  sorry

end carbonate_ions_in_Al2_CO3_3_l103_103345


namespace opposite_of_neg_third_l103_103774

theorem opposite_of_neg_third : (-(-1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_neg_third_l103_103774


namespace g_is_zero_l103_103206

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (4 * (Real.sin x)^4 + (Real.cos x)^2) - 
  Real.sqrt (4 * (Real.cos x)^4 + (Real.sin x)^2)

theorem g_is_zero (x : ℝ) : g x = 0 := 
  sorry

end g_is_zero_l103_103206


namespace complex_value_l103_103319

open Complex

theorem complex_value (z : ℂ)
  (h : 15 * normSq z = 3 * normSq (z + 3) + normSq (z^2 + 4) + 25) :
  z + (8 / z) = -4 :=
sorry

end complex_value_l103_103319


namespace attendance_changes_l103_103455

theorem attendance_changes :
  let m := 25  -- Monday attendance
  let t := 31  -- Tuesday attendance
  let w := 20  -- initial Wednesday attendance
  let th := 28  -- Thursday attendance
  let f := 22  -- Friday attendance
  let sa := 26  -- Saturday attendance
  let w_new := 30  -- corrected Wednesday attendance
  let initial_total := m + t + w + th + f + sa
  let new_total := m + t + w_new + th + f + sa
  let initial_mean := initial_total / 6
  let new_mean := new_total / 6
  let mean_increase := new_mean - initial_mean
  let initial_median := (25 + 26) / 2  -- median of [20, 22, 25, 26, 28, 31]
  let new_median := (26 + 28) / 2  -- median of [22, 25, 26, 28, 30, 31]
  let median_increase := new_median - initial_median
  mean_increase = 1.667 ∧ median_increase = 1.5 := by
sorry

end attendance_changes_l103_103455


namespace find_remainder_l103_103552

theorem find_remainder (a : ℕ) :
  (a ^ 100) % 73 = 2 ∧ (a ^ 101) % 73 = 69 → a % 73 = 71 :=
by
  sorry

end find_remainder_l103_103552


namespace range_of_a_l103_103560

def p (a : ℝ) : Prop :=
(∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0)

def q (a : ℝ) : Prop :=
0 < a ∧ a < 1

theorem range_of_a (a : ℝ) : ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
  sorry

end range_of_a_l103_103560


namespace function_is_odd_and_monotonically_increasing_on_pos_l103_103257

-- Define odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define monotonically increasing on (0, +∞)
def monotonically_increasing_on_pos (f : ℝ → ℝ) := ∀ x y : ℝ, (0 < x ∧ x < y) → f (x) < f (y)

-- Define the function in question
def f (x : ℝ) := x * |x|

-- Prove the function is odd and monotonically increasing on (0, +∞)
theorem function_is_odd_and_monotonically_increasing_on_pos :
  odd_function f ∧ monotonically_increasing_on_pos f :=
by
  sorry

end function_is_odd_and_monotonically_increasing_on_pos_l103_103257


namespace probability_sum_3_or_7_or_10_l103_103573

-- Definitions of the faces of each die
def die_1_faces : List ℕ := [1, 2, 2, 5, 5, 6]
def die_2_faces : List ℕ := [1, 2, 4, 4, 5, 6]

-- Probability of a sum being 3 (valid_pairs: (1, 2))
def probability_sum_3 : ℚ :=
  (1 / 6) * (1 / 6)

-- Probability of a sum being 7 (valid pairs: (1, 6), (2, 5))
def probability_sum_7 : ℚ :=
  ((1 / 6) * (1 / 6)) + ((1 / 3) * (1 / 6))

-- Probability of a sum being 10 (valid pairs: (5, 5))
def probability_sum_10 : ℚ :=
  (1 / 3) * (1 / 6)

-- Total probability for sums being 3, 7, or 10
def total_probability : ℚ :=
  probability_sum_3 + probability_sum_7 + probability_sum_10

-- The proof statement
theorem probability_sum_3_or_7_or_10 : total_probability = 1 / 6 :=
  sorry

end probability_sum_3_or_7_or_10_l103_103573


namespace count_irreducible_fractions_l103_103971

theorem count_irreducible_fractions (s : Finset ℕ) (h1 : ∀ n ∈ s, 15*n > 15/16) (h2 : ∀ n ∈ s, n < 1) (h3 : ∀ n ∈ s, Nat.gcd n 15 = 1) :
  s.card = 8 := 
sorry

end count_irreducible_fractions_l103_103971


namespace sum_of_solutions_l103_103764

theorem sum_of_solutions :
  let a := -48
  let b := 110
  let c := 165
  ( ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) → x1 ≠ x2 → (x1 + x2) = 55 / 24 ) :=
by
  let a := -48
  let b := 110
  let c := 165
  sorry

end sum_of_solutions_l103_103764


namespace like_terms_to_exponents_matching_l103_103602

theorem like_terms_to_exponents_matching (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : m^n = 27 := by
  sorry

end like_terms_to_exponents_matching_l103_103602


namespace pump_without_leak_time_l103_103468

variables (P : ℝ) (effective_rate_with_leak : ℝ) (leak_rate : ℝ)
variable (pump_filling_time : ℝ)

-- Define the conditions
def conditions :=
  effective_rate_with_leak = 3/7 ∧
  leak_rate = 1/14 ∧
  pump_filling_time = P

-- Define the theorem
theorem pump_without_leak_time (h : conditions P effective_rate_with_leak leak_rate pump_filling_time) : 
  P = 2 :=
sorry

end pump_without_leak_time_l103_103468


namespace right_angled_triangle_ratio_3_4_5_l103_103338

theorem right_angled_triangle_ratio_3_4_5 : 
  ∀ (a b c : ℕ), 
  (a = 3 * d) → (b = 4 * d) → (c = 5 * d) → (a^2 + b^2 = c^2) :=
by
  intros a b c h1 h2 h3
  sorry

end right_angled_triangle_ratio_3_4_5_l103_103338


namespace plums_added_l103_103320

-- Definitions of initial and final plum counts
def initial_plums : ℕ := 17
def final_plums : ℕ := 21

-- The mathematical statement to be proved
theorem plums_added (initial_plums final_plums : ℕ) : final_plums - initial_plums = 4 := by
  -- The proof will be inserted here
  sorry

end plums_added_l103_103320


namespace pension_equality_l103_103400

theorem pension_equality (x c d r s: ℝ) (h₁ : d ≠ c) 
    (h₂ : x > 0) (h₃ : 2 * x * (d - c) + d^2 - c^2 ≠ 0)
    (h₄ : ∀ k:ℝ, k * (x + c)^2 - k * x^2 = r)
    (h₅ : ∀ k:ℝ, k * (x + d)^2 - k * x^2 = s) 
    : ∃ k : ℝ, k = (s - r) / (2 * x * (d - c) + d^2 - c^2) 
    → k * x^2 = (s - r) * x^2 / (2 * x * (d - c) + d^2 - c^2) :=
by {
    sorry
}

end pension_equality_l103_103400


namespace pages_read_over_weekend_l103_103805

-- Define the given conditions
def total_pages : ℕ := 408
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

-- Define the calculated pages to be read over the remaining days
def pages_remaining := days_left * pages_per_day

-- Define the pages read over the weekend
def pages_over_weekend := total_pages - pages_remaining

-- Prove that Bekah read 113 pages over the weekend
theorem pages_read_over_weekend : pages_over_weekend = 113 :=
by {
  -- proof should be here, but we place sorry since proof is not required
  sorry
}

end pages_read_over_weekend_l103_103805


namespace smallest_number_in_sample_l103_103886

theorem smallest_number_in_sample :
  ∀ (N : ℕ) (k : ℕ) (n : ℕ), 
  0 < k → 
  N = 80 → 
  k = 5 →
  n = 42 →
  ∃ (a : ℕ), (0 ≤ a ∧ a < k) ∧
  42 = (N / k) * (42 / (N / k)) + a ∧
  ∀ (m : ℕ), (0 ≤ m ∧ m < k) → 
    (∀ (j : ℕ), (j = (N / k) * m + 10)) → 
    m = 0 → a = 10 := 
by
  sorry

end smallest_number_in_sample_l103_103886


namespace condition_suff_not_necess_l103_103067

theorem condition_suff_not_necess (x : ℝ) (h : |x - (1 / 2)| < 1 / 2) : x^3 < 1 :=
by
  have h1 : 0 < x := sorry
  have h2 : x < 1 := sorry
  sorry

end condition_suff_not_necess_l103_103067


namespace smallest_value_c_zero_l103_103199

noncomputable def smallest_possible_c (a b c : ℝ) : ℝ :=
if h : (0 < a) ∧ (0 < b) ∧ (0 < c) then
  0
else
  c

theorem smallest_value_c_zero (a b c : ℝ) (h : (0 < a) ∧ (0 < b) ∧ (0 < c)) :
  smallest_possible_c a b c = 0 :=
by
  sorry

end smallest_value_c_zero_l103_103199


namespace find_b_value_l103_103128

-- Definitions based on given conditions
def original_line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b
def shifted_line (x : ℝ) (b : ℝ) : ℝ := 2 * (x - 2) + b
def passes_through_origin (b : ℝ) := shifted_line 0 b = 0

-- Main proof statement
theorem find_b_value (b : ℝ) (h : passes_through_origin b) : b = 4 := by
  sorry

end find_b_value_l103_103128


namespace anna_initial_stamps_l103_103171

theorem anna_initial_stamps (final_stamps : ℕ) (alison_stamps : ℕ) (alison_to_anna : ℕ) : 
  final_stamps = 50 ∧ alison_stamps = 28 ∧ alison_to_anna = 14 → (final_stamps - alison_to_anna = 36) :=
by
  sorry

end anna_initial_stamps_l103_103171


namespace log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l103_103976

noncomputable def log_comparison (n : ℕ) (hn : 0 < n) : Prop := 
  Real.log n / Real.log (n + 1) < Real.log (n + 1) / Real.log (n + 2)

theorem log_comparison_theorem (n : ℕ) (hn : 0 < n) : log_comparison n hn := 
  sorry

def inequality_CauchySchwarz (a b x y : ℝ) : Prop :=
  (a*a + b*b) * (x*x + y*y) ≥ (a*x + b*y) * (a*x + b*y)

theorem CauchySchwarz_inequality_theorem (a b x y : ℝ) : inequality_CauchySchwarz a b x y :=
  sorry

noncomputable def trigonometric_minimum (x : ℝ) : ℝ := 
  (Real.sin x)^2 + (Real.cos x)^2

theorem trigonometric_minimum_theorem : ∀ x : ℝ, trigonometric_minimum x ≥ 9 :=
  sorry

end log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l103_103976


namespace cylinder_volume_l103_103574

theorem cylinder_volume (length width : ℝ) (h₁ h₂ : ℝ) (radius1 radius2 : ℝ) (V1 V2 : ℝ) (π : ℝ)
  (h_length : length = 12) (h_width : width = 8) 
  (circumference1 : circumference1 = length)
  (circumference2 : circumference2 = width)
  (h_radius1 : radius1 = 6 / π) (h_radius2 : radius2 = 4 / π)
  (h_height1 : h₁ = width) (h_height2 : h₂ = length)
  (h_V1 : V1 = π * radius1^2 * h₁) (h_V2 : V2 = π * radius2^2 * h₂) :
  V1 = 288 / π ∨ V2 = 192 / π :=
sorry


end cylinder_volume_l103_103574


namespace total_payment_for_combined_shopping_trip_l103_103576

noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else if amount ≤ 500 then amount * 0.9
  else 500 * 0.9 + (amount - 500) * 0.7

theorem total_payment_for_combined_shopping_trip :
  discount (168 + 423 / 0.9) = 546.6 :=
by
  sorry

end total_payment_for_combined_shopping_trip_l103_103576


namespace g_half_l103_103436

noncomputable def g : ℝ → ℝ := sorry

axiom g0 : g 0 = 0
axiom g1 : g 1 = 1
axiom g_non_decreasing : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom g_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom g_fraction : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

theorem g_half : g (1 / 2) = 1 / 2 := sorry

end g_half_l103_103436


namespace maynard_filled_percentage_l103_103122

theorem maynard_filled_percentage (total_holes : ℕ) (unfilled_holes : ℕ) (filled_holes : ℕ) (p : ℚ) :
  total_holes = 8 →
  unfilled_holes = 2 →
  filled_holes = total_holes - unfilled_holes →
  p = (filled_holes : ℚ) / (total_holes : ℚ) * 100 →
  p = 75 := 
by {
  -- proofs and calculations would go here
  sorry
}

end maynard_filled_percentage_l103_103122


namespace major_airlines_free_snacks_l103_103330

variable (S : ℝ)

theorem major_airlines_free_snacks (h1 : 0.5 ≤ 1) (h2 : 0.5 = 1) :
  0.5 ≤ S :=
sorry

end major_airlines_free_snacks_l103_103330


namespace ellen_smoothies_total_cups_l103_103819

structure SmoothieIngredients where
  strawberries : ℝ
  yogurt       : ℝ
  orange_juice : ℝ
  honey        : ℝ
  chia_seeds   : ℝ
  spinach      : ℝ

def ounces_to_cups (ounces : ℝ) : ℝ := ounces * 0.125
def tablespoons_to_cups (tablespoons : ℝ) : ℝ := tablespoons * 0.0625

noncomputable def total_cups (ing : SmoothieIngredients) : ℝ :=
  ing.strawberries +
  ing.yogurt +
  ing.orange_juice +
  ounces_to_cups (ing.honey) +
  tablespoons_to_cups (ing.chia_seeds) +
  ing.spinach

theorem ellen_smoothies_total_cups :
  total_cups {
    strawberries := 0.2,
    yogurt := 0.1,
    orange_juice := 0.2,
    honey := 1.0,
    chia_seeds := 2.0,
    spinach := 0.5
  } = 1.25 := by
  sorry

end ellen_smoothies_total_cups_l103_103819


namespace necessary_and_sufficient_condition_l103_103534

variable {R : Type*} [LinearOrderedField R]
variable (f : R × R → R)
variable (x₀ y₀ : R)

theorem necessary_and_sufficient_condition :
  (f (x₀, y₀) = 0) ↔ ((x₀, y₀) ∈ {p : R × R | f p = 0}) :=
by
  sorry

end necessary_and_sufficient_condition_l103_103534


namespace special_lines_count_l103_103924

noncomputable def count_special_lines : ℕ :=
  sorry

theorem special_lines_count :
  count_special_lines = 3 :=
by sorry

end special_lines_count_l103_103924


namespace value_of_x_l103_103585

theorem value_of_x (x y : ℝ) :
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 :=
by
  intro h
  sorry

end value_of_x_l103_103585


namespace total_journey_distance_l103_103599

variable (D : ℚ) (lateTime : ℚ := 1/4)

theorem total_journey_distance :
  (∃ (T : ℚ), T = D / 40 ∧ T + lateTime = D / 35) →
  D = 70 :=
by
  intros h
  obtain ⟨T, h1, h2⟩ := h
  have h3 : T = D / 40 := h1
  have h4 : T + lateTime = D / 35 := h2
  sorry

end total_journey_distance_l103_103599


namespace probability_not_monday_l103_103515

theorem probability_not_monday (P_monday : ℚ) (h : P_monday = 1/7) : P_monday ≠ 1 → ∃ P_not_monday : ℚ, P_not_monday = 6/7 :=
by
  sorry

end probability_not_monday_l103_103515


namespace mean_of_other_two_numbers_l103_103628

-- Definitions based on conditions in the problem.
def mean_of_four (numbers : List ℕ) : ℝ := 2187.25
def sum_of_numbers : ℕ := 1924 + 2057 + 2170 + 2229 + 2301 + 2365
def sum_of_four_numbers : ℝ := 4 * 2187.25
def sum_of_two_numbers := sum_of_numbers - sum_of_four_numbers

-- Theorem to assert the mean of the other two numbers.
theorem mean_of_other_two_numbers : (4297 / 2) = 2148.5 := by
  sorry

end mean_of_other_two_numbers_l103_103628


namespace Janet_previous_movie_length_l103_103030

theorem Janet_previous_movie_length (L : ℝ) (H1 : 1.60 * L = 1920 / 100) : L / 60 = 0.20 :=
by
  sorry

end Janet_previous_movie_length_l103_103030


namespace min_number_of_stamps_exists_l103_103284

theorem min_number_of_stamps_exists : 
  ∃ s t : ℕ, 5 * s + 7 * t = 50 ∧ ∀ (s' t' : ℕ), 5 * s' + 7 * t' = 50 → s + t ≤ s' + t' := 
by
  sorry

end min_number_of_stamps_exists_l103_103284


namespace apples_mass_left_l103_103541

theorem apples_mass_left (initial_kidney golden canada fuji granny : ℕ)
                         (sold_kidney golden canada fuji granny : ℕ)
                         (left_kidney golden canada fuji granny : ℕ) :
  initial_kidney = 26 → sold_kidney = 15 → left_kidney = 11 →
  initial_golden = 42 → sold_golden = 28 → left_golden = 14 →
  initial_canada = 19 → sold_canada = 12 → left_canada = 7 →
  initial_fuji = 35 → sold_fuji = 20 → left_fuji = 15 →
  initial_granny = 22 → sold_granny = 18 → left_granny = 4 →
  left_kidney = initial_kidney - sold_kidney ∧
  left_golden = initial_golden - sold_golden ∧
  left_canada = initial_canada - sold_canada ∧
  left_fuji = initial_fuji - sold_fuji ∧
  left_granny = initial_granny - sold_granny := by sorry

end apples_mass_left_l103_103541


namespace minimum_value_f_l103_103525

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end minimum_value_f_l103_103525


namespace A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l103_103426

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Statement for (1)
theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a ∈ Set.Ioi 0 :=
sorry

-- Statement for (2)
theorem A_single_element_iff_and_value (a : ℝ) : 
  (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) ∧ A a = {2 / 3} :=
sorry

-- Statement for (3)
theorem A_at_most_one_element_iff (a : ℝ) : 
  (∃ x, A a = {x} ∨ A a = ∅) ↔ (a = 0 ∨ a ∈ Set.Ici (9 / 8)) :=
sorry

end A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l103_103426


namespace arithmetic_geometric_sequence_l103_103063

theorem arithmetic_geometric_sequence {a b c x y : ℝ} (h₁: a ≠ b) (h₂: b ≠ c) (h₃: a ≠ c)
  (h₄ : 2 * b = a + c) (h₅ : x^2 = a * b) (h₆ : y^2 = b * c) :
  (x^2 + y^2 = 2 * b^2) ∧ (x^2 * y^2 ≠ b^4) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l103_103063


namespace tan_seven_pi_over_four_l103_103754

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l103_103754


namespace find_a1_and_d_l103_103581

-- Given conditions
variables {a : ℕ → ℤ} 
variables {a1 d : ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem find_a1_and_d 
  (h1 : is_arithmetic_sequence a a1 d)
  (h2 : (a 3) * (a 7) = -16)
  (h3 : (a 4) + (a 6) = 0)
  : (a1 = -8 ∧ d = 2) ∨ (a1 = 8 ∧ d = -2) :=
sorry

end find_a1_and_d_l103_103581


namespace n_times_s_eq_neg_two_l103_103518

-- Define existence of function g
variable (g : ℝ → ℝ)

-- The given condition for the function g: ℝ -> ℝ
axiom g_cond : ∀ x y : ℝ, g (g x - y) = 2 * g x + g (g y - g (-x)) + y

-- Define n and s as per the conditions mentioned in the problem
def n : ℕ := 1 -- Based on the solution, there's only one possible value
def s : ℝ := -2 -- Sum of all possible values

-- The main statement to prove
theorem n_times_s_eq_neg_two : (n * s) = -2 := by
  sorry

end n_times_s_eq_neg_two_l103_103518


namespace max_3x_4y_eq_73_l103_103546

theorem max_3x_4y_eq_73 :
  (∀ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73) ∧
  (∃ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 ∧ 3 * x + 4 * y = 73) :=
by sorry

end max_3x_4y_eq_73_l103_103546


namespace reaction_produces_correct_moles_l103_103624

-- Define the variables and constants
def moles_CO2 := 2
def moles_H2O := 2
def moles_H2CO3 := moles_CO2 -- based on the balanced reaction CO2 + H2O → H2CO3

-- The theorem we need to prove
theorem reaction_produces_correct_moles :
  moles_H2CO3 = 2 :=
by
  -- Mathematical reasoning goes here
  sorry

end reaction_produces_correct_moles_l103_103624


namespace average_marks_correct_l103_103332

-- Define constants for the marks in each subject
def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

-- Define the total number of subjects
def num_subjects : ℕ := 5

-- Define the total marks as the sum of individual subjects
def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks

-- Define the average marks
def average_marks : ℕ := total_marks / num_subjects

-- Prove that the average marks is as expected
theorem average_marks_correct : average_marks = 75 :=
by {
  -- skip the proof
  sorry
}

end average_marks_correct_l103_103332


namespace right_triangle_area_l103_103918

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l103_103918


namespace closest_to_one_tenth_l103_103193

noncomputable def p (n : ℕ) : ℚ :=
  1 / (n * (n + 2)) + 1 / ((n + 2) * (n + 4)) + 1 / ((n + 4) * (n + 6)) +
  1 / ((n + 6) * (n + 8)) + 1 / ((n + 8) * (n + 10))

theorem closest_to_one_tenth {n : ℕ} (h₀ : 4 ≤ n ∧ n ≤ 7) : 
  |(5 : ℚ) / (n * (n + 10)) - 1 / 10| ≤ 
  |(5 : ℚ) / (4 * (4 + 10)) - 1 / 10| ∧ n = 4 := 
sorry

end closest_to_one_tenth_l103_103193


namespace polynomial_identity_l103_103255

theorem polynomial_identity (a0 a1 a2 a3 a4 a5 : ℤ) (x : ℤ) :
  (1 + 3 * x) ^ 5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  a0 - a1 + a2 - a3 + a4 - a5 = -32 :=
by
  sorry

end polynomial_identity_l103_103255


namespace sin_alpha_two_alpha_plus_beta_l103_103403

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < π / 2)
variable (h₂ : 0 < β ∧ β < π / 2)
variable (h₃ : Real.tan (α / 2) = 1 / 3)
variable (h₄ : Real.cos (α - β) = -4 / 5)

theorem sin_alpha (h₁ : 0 < α ∧ α < π / 2)
                  (h₃ : Real.tan (α / 2) = 1 / 3) :
                  Real.sin α = 3 / 5 :=
by
  sorry

theorem two_alpha_plus_beta (h₁ : 0 < α ∧ α < π / 2)
                            (h₂ : 0 < β ∧ β < π / 2)
                            (h₄ : Real.cos (α - β) = -4 / 5) :
                            2 * α + β = π :=
by
  sorry

end sin_alpha_two_alpha_plus_beta_l103_103403


namespace simplify_expression_l103_103621

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 :=
by
  -- The proof is omitted, so use sorry to skip it
  sorry

end simplify_expression_l103_103621


namespace members_who_play_both_sports_l103_103974

theorem members_who_play_both_sports 
  (N B T Neither BT : ℕ) 
  (h1 : N = 27)
  (h2 : B = 17)
  (h3 : T = 19)
  (h4 : Neither = 2)
  (h5 : BT = B + T - N + Neither) : 
  BT = 11 := 
by 
  have h6 : 17 + 19 - 27 + 2 = 11 := by norm_num
  rw [h2, h3, h1, h4, h6] at h5
  exact h5

end members_who_play_both_sports_l103_103974


namespace range_of_a_minus_b_l103_103453

theorem range_of_a_minus_b {a b : ℝ} (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) : -6 < a - b ∧ a - b < 1 :=
by
  sorry -- The proof is skipped as per the instructions.

end range_of_a_minus_b_l103_103453


namespace algebraic_expression_value_l103_103149

theorem algebraic_expression_value (x : ℝ) (h : 3 / (x^2 + x) - x^2 = 2 + x) :
  2 * x^2 + 2 * x = 2 :=
sorry

end algebraic_expression_value_l103_103149


namespace find_arith_seq_common_diff_l103_103770

-- Let a_n be the nth term of the arithmetic sequence and S_n be the sum of the first n terms
variable {a : ℕ → ℝ} -- arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of first n terms of the sequence

-- Given conditions in the problem
axiom sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3)
axiom sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2)
axiom condition1 : ((S 4) / 12) - ((S 3) / 9) = 1

-- Prove that the common difference d is 6
theorem find_arith_seq_common_diff (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3))
  (sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2))
  (condition1 : (S 4) / 12 - (S 3) / 9 = 1) : 
  d = 6 := 
sorry

end find_arith_seq_common_diff_l103_103770


namespace range_of_m_l103_103815

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x + 1

theorem range_of_m (m : ℝ) : (∀ x, x ≤ 1 → f m x ≥ f m 1) ↔ 0 ≤ m ∧ m ≤ 1 / 3 := by
  sorry

end range_of_m_l103_103815


namespace arithmetic_sequence_term_difference_l103_103674

theorem arithmetic_sequence_term_difference :
  let a : ℕ := 3
  let d : ℕ := 6
  let t1 := a + 1499 * d
  let t2 := a + 1503 * d
  t2 - t1 = 24 :=
    by
    sorry

end arithmetic_sequence_term_difference_l103_103674


namespace domain_of_v_l103_103932

-- Define the function v
noncomputable def v (x y : ℝ) : ℝ := 1 / (x^(2/3) - y^(2/3))

-- State the domain of v
def domain_v : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≠ p.2 }

-- State the main theorem
theorem domain_of_v :
  ∀ x y : ℝ, x ≠ y ↔ (x, y) ∈ domain_v :=
by
  intro x y
  -- We don't need to provide proof
  sorry

end domain_of_v_l103_103932


namespace Gwen_money_left_l103_103876

theorem Gwen_money_left (received spent : ℕ) (h_received : received = 14) (h_spent : spent = 8) : 
  received - spent = 6 := 
by 
  sorry

end Gwen_money_left_l103_103876


namespace gpa_of_entire_class_l103_103922

def students : ℕ := 200

def gpa1_num : ℕ := 18 * students / 100
def gpa2_num : ℕ := 27 * students / 100
def gpa3_num : ℕ := 22 * students / 100
def gpa4_num : ℕ := 12 * students / 100
def gpa5_num : ℕ := students - (gpa1_num + gpa2_num + gpa3_num + gpa4_num)

def gpa1 : ℕ := 58
def gpa2 : ℕ := 63
def gpa3 : ℕ := 69
def gpa4 : ℕ := 75
def gpa5 : ℕ := 85

def total_points : ℕ :=
  (gpa1_num * gpa1) + (gpa2_num * gpa2) + (gpa3_num * gpa3) + (gpa4_num * gpa4) + (gpa5_num * gpa5)

def class_gpa : ℚ := total_points / students

theorem gpa_of_entire_class :
  class_gpa = 69.48 := 
  by
  sorry

end gpa_of_entire_class_l103_103922


namespace find_savings_l103_103779

theorem find_savings (I E : ℕ) (h1 : I = 21000) (h2 : I / E = 7 / 6) : I - E = 3000 := by
  sorry

end find_savings_l103_103779


namespace nine_odot_three_l103_103111

-- Defining the operation based on the given conditions
axiom odot_def (a b : ℕ) : ℕ

axiom odot_eq_1 : odot_def 2 4 = 8
axiom odot_eq_2 : odot_def 4 6 = 14
axiom odot_eq_3 : odot_def 5 3 = 13
axiom odot_eq_4 : odot_def 8 7 = 23

-- Proving that 9 ⊙ 3 = 21
theorem nine_odot_three : odot_def 9 3 = 21 := 
by
  sorry

end nine_odot_three_l103_103111


namespace remainder_when_3y_divided_by_9_l103_103775

theorem remainder_when_3y_divided_by_9 (y : ℕ) (k : ℕ) (hy : y = 9 * k + 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_when_3y_divided_by_9_l103_103775


namespace div_add_fraction_l103_103935

theorem div_add_fraction :
  (-75) / (-25) + 1/2 = 7/2 := by
  sorry

end div_add_fraction_l103_103935


namespace symmetric_origin_coordinates_l103_103870

-- Given the coordinates (m, n) of point P
variables (m n : ℝ)
-- Define point P
def P := (m, n)

-- Define point P' which is symmetric to P with respect to the origin O
def P'_symmetric_origin : ℝ × ℝ := (-m, -n)

-- Prove that the coordinates of P' are (-m, -n)
theorem symmetric_origin_coordinates :
  P'_symmetric_origin m n = (-m, -n) :=
by
  -- Proof content goes here but we're skipping it with sorry
  sorry

end symmetric_origin_coordinates_l103_103870


namespace inverse_proportion_function_l103_103912

theorem inverse_proportion_function (x y : ℝ) (h : y = 6 / x) : x * y = 6 :=
by
  sorry

end inverse_proportion_function_l103_103912


namespace remainder_zero_l103_103085

theorem remainder_zero (x : ℕ) (h1 : x = 1680) :
  (x % 5 = 0) ∧ (x % 6 = 0) ∧ (x % 7 = 0) ∧ (x % 8 = 0) :=
by
  sorry

end remainder_zero_l103_103085


namespace theater_ticket_sales_l103_103493

theorem theater_ticket_sales
  (A C : ℕ)
  (h₁ : 8 * A + 5 * C = 236)
  (h₂ : A + C = 34) : A = 22 :=
by
  sorry

end theater_ticket_sales_l103_103493


namespace kids_stayed_home_l103_103274

open Nat

theorem kids_stayed_home (kids_camp : ℕ) (additional_kids_home : ℕ) (total_kids_home : ℕ) 
  (h1 : kids_camp = 202958) 
  (h2 : additional_kids_home = 574664) 
  (h3 : total_kids_home = kids_camp + additional_kids_home) : 
  total_kids_home = 777622 := 
by 
  rw [h1, h2] at h3
  exact h3

end kids_stayed_home_l103_103274


namespace probability_m_eq_kn_l103_103302

/- 
Define the conditions and question in Lean 4 -/
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def valid_rolls : Finset (ℕ × ℕ) := Finset.product die_faces die_faces

def events_satisfying_condition : Finset (ℕ × ℕ) :=
  {(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 2), (4, 4), 
   (5, 1), (5, 5), (6, 1), (6, 2), (6, 3), (6, 6)}

theorem probability_m_eq_kn (k : ℕ) (h : k > 0) :
  (events_satisfying_condition.card : ℚ) / (valid_rolls.card : ℚ) = 7/18 := by
  sorry

end probability_m_eq_kn_l103_103302


namespace pool_fill_time_l103_103734

theorem pool_fill_time:
  ∀ (A B C D : ℚ),
  (A + B - D = 1 / 6) →
  (A + C - D = 1 / 5) →
  (B + C - D = 1 / 4) →
  (A + B + C - D = 1 / 3) →
  (1 / (A + B + C) = 60 / 23) :=
by intros A B C D h1 h2 h3 h4; sorry

end pool_fill_time_l103_103734


namespace sin_alpha_second_quadrant_l103_103443

theorem sin_alpha_second_quadrant (α : ℝ) (h_α_quad_2 : π / 2 < α ∧ α < π) (h_cos_α : Real.cos α = -1 / 3) : Real.sin α = 2 * Real.sqrt 2 / 3 := 
sorry

end sin_alpha_second_quadrant_l103_103443


namespace smallest_positive_integer_divisible_by_14_15_18_l103_103432

theorem smallest_positive_integer_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ n = 630 :=
sorry

end smallest_positive_integer_divisible_by_14_15_18_l103_103432


namespace find_value_of_question_mark_l103_103154

theorem find_value_of_question_mark (q : ℕ) : q * 40 = 173 * 240 → q = 1036 :=
by
  intro h
  sorry

end find_value_of_question_mark_l103_103154


namespace train_length_l103_103396

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 52) (h2 : time_sec = 9) (h3 : length_m = 129.96) : 
  length_m = (speed_km_hr * 1000 / 3600) * time_sec := 
sorry

end train_length_l103_103396


namespace hyperbola_properties_l103_103434

def hyperbola (x y : ℝ) : Prop := x^2 - 4 * y^2 = 1

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → (x + 2 * y = 0 ∨ x - 2 * y = 0)) ∧
  (2 * (1 / 2) = 1) := 
by
  sorry

end hyperbola_properties_l103_103434


namespace total_nap_duration_l103_103421

def nap1 : ℚ := 1 / 5
def nap2 : ℚ := 1 / 4
def nap3 : ℚ := 1 / 6
def hour_to_minutes : ℚ := 60

theorem total_nap_duration :
  (nap1 + nap2 + nap3) * hour_to_minutes = 37 := by
  sorry

end total_nap_duration_l103_103421


namespace jenna_water_cups_l103_103698

theorem jenna_water_cups (O S W : ℕ) (h1 : S = 3 * O) (h2 : W = 3 * S) (h3 : O = 4) : W = 36 :=
by
  sorry

end jenna_water_cups_l103_103698


namespace function_value_bounds_l103_103334

theorem function_value_bounds (x : ℝ) : 
  (x^2 + x + 1) / (x^2 + 1) ≤ 3 / 2 ∧ (x^2 + x + 1) / (x^2 + 1) ≥ 1 / 2 := 
sorry

end function_value_bounds_l103_103334


namespace coconut_grove_average_yield_l103_103160

theorem coconut_grove_average_yield :
  ∀ (x : ℕ),
  40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x →
  x = 7 :=
by
  intro x
  intro h
  /- sorry proof -/
  sorry

end coconut_grove_average_yield_l103_103160


namespace gwen_average_speed_l103_103813

def average_speed (distance1 distance2 speed1 speed2 : ℕ) : ℕ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

theorem gwen_average_speed :
  average_speed 40 40 15 30 = 20 :=
by
  sorry

end gwen_average_speed_l103_103813


namespace part1_part2_l103_103053

noncomputable def setA : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }
noncomputable def setB (m : ℝ) : Set ℝ := { x | m - 1 < x ∧ x < 2*m + 1 }

theorem part1 (x : ℝ) : 
  setA ∪ setB 3 = { x | -1 ≤ x ∧ x < 7 } :=
sorry

theorem part2 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∃ x, x ∈ setB m ∧ x ∉ setA) ↔ 
  m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) :=
sorry

end part1_part2_l103_103053


namespace solve_for_n_l103_103554

theorem solve_for_n (n : ℕ) : 4^8 = 16^n → n = 4 :=
by
  sorry

end solve_for_n_l103_103554


namespace simplify_and_evaluate_expression_l103_103731

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l103_103731


namespace polynomial_divisibility_n_l103_103458

theorem polynomial_divisibility_n :
  ∀ (n : ℤ), (∀ x, x = 2 → 3 * x^2 - 4 * x + n = 0) → n = -4 :=
by
  intros n h
  have h2 : 3 * 2^2 - 4 * 2 + n = 0 := h 2 rfl
  linarith

end polynomial_divisibility_n_l103_103458


namespace hypotenuse_45_45_90_l103_103322

theorem hypotenuse_45_45_90 (leg : ℝ) (h_leg : leg = 10) (angle : ℝ) (h_angle : angle = 45) :
  ∃ hypotenuse : ℝ, hypotenuse = leg * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end hypotenuse_45_45_90_l103_103322


namespace andrew_paid_1428_to_shopkeeper_l103_103285

-- Given conditions
def rate_per_kg_grapes : ℕ := 98
def quantity_of_grapes : ℕ := 11
def rate_per_kg_mangoes : ℕ := 50
def quantity_of_mangoes : ℕ := 7

-- Definitions for costs
def cost_of_grapes : ℕ := rate_per_kg_grapes * quantity_of_grapes
def cost_of_mangoes : ℕ := rate_per_kg_mangoes * quantity_of_mangoes
def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

-- Theorem to prove the total amount paid
theorem andrew_paid_1428_to_shopkeeper : total_amount_paid = 1428 := by
  sorry

end andrew_paid_1428_to_shopkeeper_l103_103285


namespace smallest_five_digit_palindrome_div_4_thm_l103_103419

def is_palindrome (n : ℕ) : Prop :=
  n = (n % 10) * 10000 + ((n / 10) % 10) * 1000 + ((n / 100) % 10) * 100 + ((n / 1000) % 10) * 10 + (n / 10000)

def smallest_five_digit_palindrome_div_4 : ℕ :=
  18881

theorem smallest_five_digit_palindrome_div_4_thm :
  is_palindrome smallest_five_digit_palindrome_div_4 ∧
  10000 ≤ smallest_five_digit_palindrome_div_4 ∧
  smallest_five_digit_palindrome_div_4 < 100000 ∧
  smallest_five_digit_palindrome_div_4 % 4 = 0 ∧
  ∀ n, is_palindrome n ∧ 10000 ≤ n ∧ n < 100000 ∧ n % 4 = 0 → n ≥ smallest_five_digit_palindrome_div_4 :=
by
  sorry

end smallest_five_digit_palindrome_div_4_thm_l103_103419


namespace no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l103_103026

-- Conditions: Expressing the sum of three reciprocals
def sum_of_reciprocals (a b c : ℕ) : ℚ := (1 / a) + (1 / b) + (1 / c)

-- Proof Problem 1: Prove that the sum of the reciprocals of any three positive integers cannot equal 9/11
theorem no_three_reciprocals_sum_to_nine_eleven :
  ∀ (a b c : ℕ), sum_of_reciprocals a b c ≠ 9 / 11 := sorry

-- Proof Problem 2: Prove that there exists no rational number between 41/42 and 1 that can be expressed as the sum of the reciprocals of three positive integers other than 41/42
theorem no_rational_between_fortyone_fortytwo_and_one :
  ∀ (K : ℚ), 41 / 42 < K ∧ K < 1 → ¬ (∃ (a b c : ℕ), sum_of_reciprocals a b c = K) := sorry

end no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l103_103026


namespace min_length_intersection_l103_103972

theorem min_length_intersection (m n : ℝ) (h_m1 : 0 ≤ m) (h_m2 : m + 7 / 10 ≤ 1) 
                                (h_n1 : 2 / 5 ≤ n) (h_n2 : n ≤ 1) : 
  ∃ (min_length : ℝ), min_length = 1 / 10 :=
by
  sorry

end min_length_intersection_l103_103972


namespace largest_possible_n_base10_l103_103563

theorem largest_possible_n_base10 :
  ∃ (n A B C : ℕ),
    n = 25 * A + 5 * B + C ∧ 
    n = 81 * C + 9 * B + A ∧ 
    A < 5 ∧ B < 5 ∧ C < 5 ∧ 
    n = 69 :=
by {
  sorry
}

end largest_possible_n_base10_l103_103563


namespace correct_calculation_for_A_l103_103383

theorem correct_calculation_for_A (x : ℝ) : (-2 * x) ^ 3 = -8 * x ^ 3 :=
by
  sorry

end correct_calculation_for_A_l103_103383


namespace shortest_distance_proof_l103_103572

noncomputable def shortest_distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem shortest_distance_proof : 
  let A : ℝ × ℝ := (0, 250)
  let B : ℝ × ℝ := (800, 1050)
  shortest_distance A B = 1131 :=
by
  sorry

end shortest_distance_proof_l103_103572


namespace scientific_notation_of_220_billion_l103_103361

theorem scientific_notation_of_220_billion :
  220000000000 = 2.2 * 10^11 :=
by
  sorry

end scientific_notation_of_220_billion_l103_103361


namespace functional_equation_zero_l103_103355

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (hx : ∀ x y : ℝ, f (x + y) = f x + f y) : f 0 = 0 :=
by
  sorry

end functional_equation_zero_l103_103355


namespace max_xy_on_line_AB_l103_103861

noncomputable def pointA : ℝ × ℝ := (3, 0)
noncomputable def pointB : ℝ × ℝ := (0, 4)

-- Define the line passing through points A and B
def on_line_AB (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P.1 = 3 - 3 * t ∧ P.2 = 4 * t

theorem max_xy_on_line_AB : ∃ (P : ℝ × ℝ), on_line_AB P ∧ P.1 * P.2 = 3 := 
sorry

end max_xy_on_line_AB_l103_103861


namespace rectangle_perimeters_l103_103848

theorem rectangle_perimeters (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 3 * (2 * a + 2 * b)) : 
  2 * (a + b) = 36 ∨ 2 * (a + b) = 28 :=
by sorry

end rectangle_perimeters_l103_103848


namespace initial_candies_l103_103210

theorem initial_candies (L R : ℕ) (h1 : L + R = 27) (h2 : R - L = 2 * L + 3) : L = 6 ∧ R = 21 :=
by
  sorry

end initial_candies_l103_103210


namespace cos_alpha_value_l103_103131

theorem cos_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos (α + π / 4) = 4 / 5) :
  Real.cos α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cos_alpha_value_l103_103131


namespace number_of_5card_hands_with_4_of_a_kind_l103_103099

-- Definitions based on the given conditions
def deck_size : Nat := 52
def num_values : Nat := 13
def suits_per_value : Nat := 4

-- The function to count the number of 5-card hands with exactly four cards of the same value
def count_hands_with_four_of_a_kind : Nat :=
  num_values * (deck_size - suits_per_value)

-- Proof statement
theorem number_of_5card_hands_with_4_of_a_kind : count_hands_with_four_of_a_kind = 624 :=
by
  -- Steps to show the computation results may be added here
  -- We use the formula: 13 * (52 - 4)
  sorry

end number_of_5card_hands_with_4_of_a_kind_l103_103099


namespace witch_votes_is_seven_l103_103527

-- Definitions
def votes_for_witch (W : ℕ) : ℕ := W
def votes_for_unicorn (W : ℕ) : ℕ := 3 * W
def votes_for_dragon (W : ℕ) : ℕ := W + 25
def total_votes (W : ℕ) : ℕ := votes_for_witch W + votes_for_unicorn W + votes_for_dragon W

-- Proof Statement
theorem witch_votes_is_seven (W : ℕ) (h1 : total_votes W = 60) : W = 7 :=
by
  sorry

end witch_votes_is_seven_l103_103527


namespace solve_x_eq_l103_103226

theorem solve_x_eq : ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 ∧ x = -7 / 3 :=
by 
  sorry

end solve_x_eq_l103_103226


namespace least_number_to_add_1055_to_div_by_23_l103_103289

theorem least_number_to_add_1055_to_div_by_23 : ∃ k : ℕ, (1055 + k) % 23 = 0 ∧ k = 3 :=
by
  sorry

end least_number_to_add_1055_to_div_by_23_l103_103289


namespace weight_loss_in_april_l103_103690

-- Definitions based on given conditions
def total_weight_to_lose : ℕ := 10
def march_weight_loss : ℕ := 3
def may_weight_loss : ℕ := 3

-- Theorem statement
theorem weight_loss_in_april :
  total_weight_to_lose = march_weight_loss + 4 + may_weight_loss := 
sorry

end weight_loss_in_april_l103_103690


namespace sum_of_two_numbers_l103_103377

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end sum_of_two_numbers_l103_103377


namespace div_eq_implies_eq_l103_103134

theorem div_eq_implies_eq (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end div_eq_implies_eq_l103_103134


namespace remainder_of_poly_division_l103_103894

theorem remainder_of_poly_division :
  ∀ (x : ℂ), ((x + 1)^2048) % (x^2 - x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_division_l103_103894


namespace curve_symmetric_origin_l103_103767

theorem curve_symmetric_origin (x y : ℝ) (h : 3*x^2 - 8*x*y + 2*y^2 = 0) :
  3*(-x)^2 - 8*(-x)*(-y) + 2*(-y)^2 = 3*x^2 - 8*x*y + 2*y^2 :=
sorry

end curve_symmetric_origin_l103_103767


namespace total_wage_calculation_l103_103286

def basic_pay_rate : ℝ := 20
def weekly_hours : ℝ := 40
def overtime_rate : ℝ := basic_pay_rate * 1.25
def total_hours_worked : ℝ := 48
def overtime_hours : ℝ := total_hours_worked - weekly_hours

theorem total_wage_calculation : 
  (weekly_hours * basic_pay_rate) + (overtime_hours * overtime_rate) = 1000 :=
by
  sorry

end total_wage_calculation_l103_103286


namespace blocks_differ_in_two_ways_exactly_l103_103102

theorem blocks_differ_in_two_ways_exactly 
  (materials : Finset String := {"plastic", "wood", "metal"})
  (sizes : Finset String := {"small", "medium", "large"})
  (colors : Finset String := {"blue", "green", "red", "yellow"})
  (shapes : Finset String := {"circle", "hexagon", "square", "triangle"})
  (target : String := "plastic medium red circle") :
  ∃ (n : ℕ), n = 37 := by
  sorry

end blocks_differ_in_two_ways_exactly_l103_103102


namespace algebraic_expression_evaluation_l103_103335

theorem algebraic_expression_evaluation (m : ℝ) (h : m^2 - m - 3 = 0) : m^2 - m - 2 = 1 := 
by
  sorry

end algebraic_expression_evaluation_l103_103335


namespace abs_diff_of_solutions_l103_103593

theorem abs_diff_of_solutions (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
sorry

end abs_diff_of_solutions_l103_103593


namespace find_a_l103_103699

theorem find_a (a b d : ℤ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l103_103699


namespace numBaskets_l103_103673

noncomputable def numFlowersInitial : ℕ := 5 + 5
noncomputable def numFlowersAfterGrowth : ℕ := numFlowersInitial + 20
noncomputable def numFlowersFinal : ℕ := numFlowersAfterGrowth - 10
noncomputable def flowersPerBasket : ℕ := 4

theorem numBaskets : numFlowersFinal / flowersPerBasket = 5 := 
by
  sorry

end numBaskets_l103_103673


namespace series_sum_eq_1_div_400_l103_103646

theorem series_sum_eq_1_div_400 :
  (∑' n : ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 := 
sorry

end series_sum_eq_1_div_400_l103_103646


namespace total_digits_l103_103120

theorem total_digits (n S S6 S4 : ℕ) 
  (h1 : S = 80 * n)
  (h2 : S6 = 6 * 58)
  (h3 : S4 = 4 * 113)
  (h4 : S = S6 + S4) : 
  n = 10 :=
by 
  sorry

end total_digits_l103_103120


namespace walter_chore_days_l103_103358

-- Definitions for the conditions
variables (b w : ℕ)  -- b: days regular, w: days exceptionally well

-- Conditions
def days_eq : Prop := b + w = 15
def earnings_eq : Prop := 3 * b + 4 * w = 47

-- The theorem stating the proof problem
theorem walter_chore_days (hb : days_eq b w) (he : earnings_eq b w) : w = 2 :=
by
  -- We only need to state the theorem; the proof is omitted.
  sorry

end walter_chore_days_l103_103358


namespace still_need_more_volunteers_l103_103197

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end still_need_more_volunteers_l103_103197


namespace decrease_by_150_percent_l103_103872

theorem decrease_by_150_percent (x : ℝ) (h : x = 80) : x - 1.5 * x = -40 :=
by
  sorry

end decrease_by_150_percent_l103_103872


namespace find_correct_quotient_l103_103780

theorem find_correct_quotient 
  (Q : ℕ)
  (D : ℕ)
  (h1 : D = 21 * Q)
  (h2 : D = 12 * 35) : 
  Q = 20 := 
by 
  sorry

end find_correct_quotient_l103_103780


namespace heather_aprons_l103_103093

variable {totalAprons : Nat} (apronsSewnBeforeToday apronsSewnToday apronsSewnTomorrow apronsSewnSoFar apronsRemaining : Nat)

theorem heather_aprons (h_total : totalAprons = 150)
                       (h_today : apronsSewnToday = 3 * apronsSewnBeforeToday)
                       (h_sewnSoFar : apronsSewnSoFar = apronsSewnBeforeToday + apronsSewnToday)
                       (h_tomorrow : apronsSewnTomorrow = 49)
                       (h_remaining : apronsRemaining = totalAprons - apronsSewnSoFar)
                       (h_halfRemaining : 2 * apronsSewnTomorrow = apronsRemaining) :
  apronsSewnBeforeToday = 13 :=
by
  sorry

end heather_aprons_l103_103093


namespace people_visited_both_l103_103110

theorem people_visited_both (total iceland norway neither both : ℕ) (h_total: total = 100) (h_iceland: iceland = 55) (h_norway: norway = 43) (h_neither: neither = 63)
  (h_both_def: both = iceland + norway - (total - neither)) :
  both = 61 :=
by 
  rw [h_total, h_iceland, h_norway, h_neither] at h_both_def
  simp at h_both_def
  exact h_both_def

end people_visited_both_l103_103110


namespace initial_nickels_eq_l103_103301

variable (quarters : ℕ) (initial_nickels : ℕ) (nickels_borrowed : ℕ) (nickels_left : ℕ)

-- Assumptions based on the problem
axiom quarters_had : quarters = 33
axiom nickels_left_axiom : nickels_left = 12
axiom nickels_borrowed_axiom : nickels_borrowed = 75

-- Theorem to prove: initial number of nickels
theorem initial_nickels_eq :
  initial_nickels = nickels_left + nickels_borrowed :=
by
  sorry

end initial_nickels_eq_l103_103301


namespace exists_n_for_pn_consecutive_zeros_l103_103883

theorem exists_n_for_pn_consecutive_zeros (p : ℕ) (hp : Nat.Prime p) (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, (∃ k : ℕ, (p^n) / 10^(k+m) % 10^m = 0) := sorry

end exists_n_for_pn_consecutive_zeros_l103_103883


namespace finished_in_6th_l103_103789

variable (p : ℕ → Prop)
variable (Sana Max Omar Jonah Leila : ℕ)

-- Conditions
def condition1 : Prop := Omar = Jonah - 7
def condition2 : Prop := Sana = Max - 2
def condition3 : Prop := Leila = Jonah + 3
def condition4 : Prop := Max = Omar + 1
def condition5 : Prop := Sana = 4

-- Conclusion
theorem finished_in_6th (h1 : condition1 Omar Jonah)
                         (h2 : condition2 Sana Max)
                         (h3 : condition3 Leila Jonah)
                         (h4 : condition4 Max Omar)
                         (h5 : condition5 Sana) :
  Max = 6 := by
  sorry

end finished_in_6th_l103_103789


namespace shifted_quadratic_eq_l103_103265

-- Define the original quadratic function
def orig_fn (x : ℝ) : ℝ := -x^2

-- Define the function after shifting 1 unit to the left
def shifted_left_fn (x : ℝ) : ℝ := - (x + 1)^2

-- Define the final function after also shifting 3 units up
def final_fn (x : ℝ) : ℝ := - (x + 1)^2 + 3

-- Prove the final function is the correctly transformed function from the original one
theorem shifted_quadratic_eq : ∀ (x : ℝ), final_fn x = - (x + 1)^2 + 3 :=
by 
  intro x
  sorry

end shifted_quadratic_eq_l103_103265


namespace P_desert_but_not_Coffee_is_0_15_l103_103461

-- Define the relevant probabilities as constants
def P_desert_and_coffee := 0.60
def P_not_desert := 0.2500000000000001
def P_desert := 1 - P_not_desert
def P_desert_but_not_coffee := P_desert - P_desert_and_coffee

-- The theorem to prove that the probability of ordering dessert but not coffee is 0.15
theorem P_desert_but_not_Coffee_is_0_15 :
  P_desert_but_not_coffee = 0.15 :=
by 
  -- calculation steps can be filled in here eventually
  sorry

end P_desert_but_not_Coffee_is_0_15_l103_103461


namespace problem_inequality_l103_103865

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) : 
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 :=
sorry

end problem_inequality_l103_103865


namespace distance_corresponds_to_additional_charge_l103_103524

-- Define the initial fee
def initial_fee : ℝ := 2.5

-- Define the charge per part of a mile
def charge_per_part_of_mile : ℝ := 0.35

-- Define the total charge for a 3.6 miles trip
def total_charge : ℝ := 5.65

-- Define the correct distance corresponding to the additional charge
def correct_distance : ℝ := 0.9

-- The theorem to prove
theorem distance_corresponds_to_additional_charge :
  (total_charge - initial_fee) / charge_per_part_of_mile * (0.1) = correct_distance :=
by
  sorry

end distance_corresponds_to_additional_charge_l103_103524


namespace number_of_correct_conclusions_l103_103113

theorem number_of_correct_conclusions : 
    (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
    (∀ x : ℝ, (x ≠ 0 → x - Real.sin x ≠ 0)) ∧
    (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
    (¬ (∀ x : ℝ, x - Real.log x > 0))
    → 3 = 3 :=
by
  sorry

end number_of_correct_conclusions_l103_103113


namespace expression_equivalence_l103_103866

theorem expression_equivalence :
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 :=
by
  sorry

end expression_equivalence_l103_103866


namespace rhombus_diagonal_length_l103_103776

theorem rhombus_diagonal_length 
  (side_length : ℕ) (shorter_diagonal : ℕ) (longer_diagonal : ℕ)
  (h1 : side_length = 34) (h2 : shorter_diagonal = 32) :
  longer_diagonal = 60 :=
sorry

end rhombus_diagonal_length_l103_103776


namespace total_number_of_games_in_season_l103_103121

def number_of_games_per_month : ℕ := 13
def number_of_months_in_season : ℕ := 14

theorem total_number_of_games_in_season :
  number_of_games_per_month * number_of_months_in_season = 182 := by
  sorry

end total_number_of_games_in_season_l103_103121


namespace pascal_triangle_fifth_number_l103_103037

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l103_103037


namespace complex_power_sum_l103_103610

noncomputable def z : ℂ := sorry

theorem complex_power_sum (hz : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 :=
sorry

end complex_power_sum_l103_103610


namespace solve_equation_nat_numbers_l103_103239

theorem solve_equation_nat_numbers :
  ∃ (x y z : ℕ), (2 ^ x + 3 ^ y + 7 = z!) ∧ ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
sorry

end solve_equation_nat_numbers_l103_103239


namespace adding_sugar_increases_sweetness_l103_103727

theorem adding_sugar_increases_sweetness 
  (a b m : ℝ) (hb : b > a) (ha : a > 0) (hm : m > 0) : 
  (a / b) < (a + m) / (b + m) := 
by
  sorry

end adding_sugar_increases_sweetness_l103_103727


namespace stratified_sampling_l103_103644

theorem stratified_sampling (N : ℕ) (r1 r2 r3 : ℕ) (sample_size : ℕ) 
  (ratio_given : r1 = 5 ∧ r2 = 2 ∧ r3 = 3) 
  (total_sample_size : sample_size = 200) :
  sample_size * r3 / (r1 + r2 + r3) = 60 := 
by
  sorry

end stratified_sampling_l103_103644


namespace solution_to_system_of_equations_l103_103439

theorem solution_to_system_of_equations :
  ∃ x y : ℤ, 4 * x - 3 * y = 11 ∧ 2 * x + y = 13 ∧ x = 5 ∧ y = 3 :=
by
  sorry

end solution_to_system_of_equations_l103_103439


namespace pow_mod_eleven_l103_103642

theorem pow_mod_eleven : 
  ∀ (n : ℕ), (n ≡ 5 ^ 1 [MOD 11] → n ≡ 5 [MOD 11]) ∧ 
             (n ≡ 5 ^ 2 [MOD 11] → n ≡ 3 [MOD 11]) ∧ 
             (n ≡ 5 ^ 3 [MOD 11] → n ≡ 4 [MOD 11]) ∧ 
             (n ≡ 5 ^ 4 [MOD 11] → n ≡ 9 [MOD 11]) ∧ 
             (n ≡ 5 ^ 5 [MOD 11] → n ≡ 1 [MOD 11]) →
  5 ^ 1233 ≡ 4 [MOD 11] :=
by
  intro n h
  sorry

end pow_mod_eleven_l103_103642


namespace angle_difference_proof_l103_103735

-- Define the angles A and B
def angle_A : ℝ := 65
def angle_B : ℝ := 180 - angle_A

-- Define the difference
def angle_difference : ℝ := angle_B - angle_A

theorem angle_difference_proof : angle_difference = 50 :=
by
  -- The proof goes here
  sorry

end angle_difference_proof_l103_103735


namespace intersection_A_B_l103_103001

open Set

def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_A_B : A ∩ B = { 1 } := by
  sorry

end intersection_A_B_l103_103001


namespace neg_parallelogram_is_rhombus_l103_103708

def parallelogram_is_rhombus := true

theorem neg_parallelogram_is_rhombus : ¬ parallelogram_is_rhombus := by
  sorry

end neg_parallelogram_is_rhombus_l103_103708


namespace total_customers_in_line_l103_103715

-- Define the number of people behind the first person
def people_behind := 11

-- Define the total number of people in line
def people_in_line : Nat := people_behind + 1

-- Prove the total number of people in line is 12
theorem total_customers_in_line : people_in_line = 12 :=
by
  sorry

end total_customers_in_line_l103_103715


namespace q_at_14_l103_103535

noncomputable def q (x : ℝ) : ℝ := - (1 / 2) * x^2 + x + 2

theorem q_at_14 : q 14 = -82 := by
  sorry

end q_at_14_l103_103535


namespace strings_completely_pass_each_other_l103_103497

-- Define the problem parameters
def d : ℝ := 30    -- distance between A and B in cm
def l1 : ℝ := 151  -- length of string A in cm
def l2 : ℝ := 187  -- length of string B in cm
def v1 : ℝ := 2    -- speed of string A in cm/s
def v2 : ℝ := 3    -- speed of string B in cm/s
def r1 : ℝ := 1    -- burn rate of string A in cm/s
def r2 : ℝ := 2    -- burn rate of string B in cm/s

-- The proof problem statement
theorem strings_completely_pass_each_other : ∀ (T : ℝ), T = 40 :=
by
  sorry

end strings_completely_pass_each_other_l103_103497


namespace solve_system_l103_103990

theorem solve_system (x y : ℝ) (h1 : x + 3 * y = 20) (h2 : x + y = 10) : x = 5 ∧ y = 5 := 
by 
  sorry

end solve_system_l103_103990


namespace value_of_b_minus_a_l103_103249

variable (a b : ℕ)

theorem value_of_b_minus_a 
  (h1 : b = 10)
  (h2 : a * b = 2 * (a + b) + 12) : b - a = 6 :=
by sorry

end value_of_b_minus_a_l103_103249


namespace probability_contemporaries_correct_l103_103763

def alice_lifespan : ℝ := 150
def bob_lifespan : ℝ := 150
def total_years : ℝ := 800

noncomputable def probability_contemporaries : ℝ :=
  let unshaded_tri_area := (650 * 150) / 2
  let unshaded_area := 2 * unshaded_tri_area
  let total_area := total_years * total_years
  let shaded_area := total_area - unshaded_area
  shaded_area / total_area

theorem probability_contemporaries_correct : 
  probability_contemporaries = 27125 / 32000 :=
by
  sorry

end probability_contemporaries_correct_l103_103763


namespace triangle_angles_and_side_l103_103185

noncomputable def triangle_properties : Type := sorry

variables {A B C : ℝ}
variables {a b c : ℝ}

theorem triangle_angles_and_side (hA : A = 60)
    (ha : a = 4 * Real.sqrt 3)
    (hb : b = 4 * Real.sqrt 2)
    (habc : triangle_properties)
    : B = 45 ∧ C = 75 ∧ c = 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := 
sorry

end triangle_angles_and_side_l103_103185


namespace net_effect_on_sale_value_l103_103071

theorem net_effect_on_sale_value
(P Q : ℝ)
(h_new_price : ∃ P', P' = P - 0.22 * P)
(h_new_qty : ∃ Q', Q' = Q + 0.86 * Q) :
  let original_sale_value := P * Q
  let new_sale_value := (0.78 * P) * (1.86 * Q)
  let net_effect := ((new_sale_value / original_sale_value - 1) * 100 : ℝ)
  net_effect = 45.08 :=
by {
  sorry
}

end net_effect_on_sale_value_l103_103071


namespace n_salary_eq_260_l103_103326

variables (m n : ℕ)
axiom total_salary : m + n = 572
axiom m_salary : m = 120 * n / 100

theorem n_salary_eq_260 : n = 260 :=
by
  sorry

end n_salary_eq_260_l103_103326


namespace GreatWhiteSharkTeeth_l103_103556

-- Definition of the number of teeth for a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Definition of the number of teeth for a hammerhead shark
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Definition of the number of teeth for a great white shark
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- Statement to prove
theorem GreatWhiteSharkTeeth : great_white_shark_teeth = 420 :=
by
  -- Proof omitted
  sorry

end GreatWhiteSharkTeeth_l103_103556


namespace ninth_graders_science_only_l103_103481

theorem ninth_graders_science_only 
    (total_students : ℕ := 120)
    (science_students : ℕ := 80)
    (programming_students : ℕ := 75) 
    : (science_students - (science_students + programming_students - total_students)) = 45 :=
by
  sorry

end ninth_graders_science_only_l103_103481


namespace sufficient_but_not_necessary_condition_l103_103908

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 2) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l103_103908


namespace students_more_than_turtles_l103_103970

theorem students_more_than_turtles
  (students_per_classroom : ℕ)
  (turtles_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : turtles_per_classroom = 3)
  (h3 : number_of_classrooms = 5) :
  (students_per_classroom * number_of_classrooms)
  - (turtles_per_classroom * number_of_classrooms) = 85 :=
by
  sorry

end students_more_than_turtles_l103_103970


namespace prob_white_ball_second_l103_103933

structure Bag :=
  (black_balls : ℕ)
  (white_balls : ℕ)

def total_balls (bag : Bag) := bag.black_balls + bag.white_balls

def prob_white_second_after_black_first (bag : Bag) : ℚ :=
  if bag.black_balls > 0 ∧ bag.white_balls > 0 ∧ total_balls bag > 1 then
    (bag.white_balls : ℚ) / (total_balls bag - 1)
  else 0

theorem prob_white_ball_second 
  (bag : Bag)
  (h_black : bag.black_balls = 4)
  (h_white : bag.white_balls = 3)
  (h_total : total_balls bag = 7) :
  prob_white_second_after_black_first bag = 1 / 2 :=
by
  sorry

end prob_white_ball_second_l103_103933


namespace solve_for_x_l103_103762

theorem solve_for_x (x : ℝ) (h : 0.4 * x = (1 / 3) * x + 110) : x = 1650 :=
by sorry

end solve_for_x_l103_103762


namespace max_value_fraction_l103_103947

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y : ℝ, (0 < x → 0 < y → (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3)) :=
by
  sorry

end max_value_fraction_l103_103947


namespace mean_score_of_sophomores_l103_103336

open Nat

variable (s j : ℕ)
variable (m m_s m_j : ℝ)

theorem mean_score_of_sophomores :
  (s + j = 150) →
  (m = 85) →
  (j = 80 / 100 * s) →
  (m_s = 125 / 100 * m_j) →
  (s * m_s + j * m_j = 12750) →
  m_s = 94 := by intros; sorry

end mean_score_of_sophomores_l103_103336


namespace length_of_opposite_leg_l103_103678

noncomputable def hypotenuse_length : Real := 18

noncomputable def angle_deg : Real := 30

theorem length_of_opposite_leg (h : Real) (angle : Real) (condition1 : h = hypotenuse_length) (condition2 : angle = angle_deg) : 
 ∃ x : Real, 2 * x = h ∧ angle = 30 → x = 9 := 
by
  sorry

end length_of_opposite_leg_l103_103678


namespace consecutive_integers_product_divisible_l103_103547

theorem consecutive_integers_product_divisible (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∀ n : ℕ, ∃ (x y : ℕ), (n ≤ x) ∧ (x < n + b) ∧ (n ≤ y) ∧ (y < n + b) ∧ (x ≠ y) ∧ (a * b ∣ x * y) :=
by
  sorry

end consecutive_integers_product_divisible_l103_103547


namespace divisibility_by_91_l103_103457

theorem divisibility_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n + 2) + 10^(2 * n + 1) = 91 * k := by
  sorry

end divisibility_by_91_l103_103457


namespace find_a_l103_103192

open Classical

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (|f x a| < 1) ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) → a = 1 :=
by
  intro h
  sorry

end find_a_l103_103192


namespace greatest_common_divisor_b_81_l103_103294

theorem greatest_common_divisor_b_81 (a b : ℤ) 
  (h : (1 + Real.sqrt 2) ^ 2012 = a + b * Real.sqrt 2) : Int.gcd b 81 = 3 :=
by
  sorry

end greatest_common_divisor_b_81_l103_103294


namespace picnic_problem_l103_103773

variable (M W A C : ℕ)

theorem picnic_problem (h1 : M = 90)
  (h2 : M = W + 40)
  (h3 : M + W + C = 240) :
  A = M + W ∧ A - C = 40 := by
  sorry

end picnic_problem_l103_103773


namespace p_minus_q_l103_103162

theorem p_minus_q (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1 / 3 := by
  sorry

end p_minus_q_l103_103162


namespace parallelepiped_inequality_l103_103684

theorem parallelepiped_inequality (a b c d : ℝ) (h : d^2 = a^2 + b^2 + c^2 + 2 * (a * b + a * c + b * c)) :
  a^2 + b^2 + c^2 ≥ (1 / 3) * d^2 :=
by
  sorry

end parallelepiped_inequality_l103_103684


namespace jason_seashells_initial_count_l103_103583

variable (initialSeashells : ℕ) (seashellsGivenAway : ℕ)
variable (seashellsNow : ℕ) (initialSeashells := 49)
variable (seashellsGivenAway := 13) (seashellsNow := 36)

theorem jason_seashells_initial_count :
  initialSeashells - seashellsGivenAway = seashellsNow → initialSeashells = 49 := by
  sorry

end jason_seashells_initial_count_l103_103583


namespace sum_of_first_100_terms_AP_l103_103498

theorem sum_of_first_100_terms_AP (a d : ℕ) :
  (15 / 2) * (2 * a + 14 * d) = 45 →
  (85 / 2) * (2 * a + 84 * d) = 255 →
  (100 / 2) * (2 * a + 99 * d) = 300 :=
by
  sorry

end sum_of_first_100_terms_AP_l103_103498


namespace smallestThreeDigitNumberWithPerfectSquare_l103_103027

def isThreeDigitNumber (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

def formsPerfectSquare (a : ℕ) : Prop := ∃ n : ℕ, 1001 * a + 1 = n * n

theorem smallestThreeDigitNumberWithPerfectSquare :
  ∀ a : ℕ, isThreeDigitNumber a → formsPerfectSquare a → a = 183 :=
by
sorry

end smallestThreeDigitNumberWithPerfectSquare_l103_103027


namespace principal_amount_l103_103159

theorem principal_amount (SI : ℝ) (T : ℝ) (R : ℝ) (P : ℝ) (h1 : SI = 140) (h2 : T = 2) (h3 : R = 17.5) :
  P = 400 :=
by
  -- Formal proof would go here
  sorry

end principal_amount_l103_103159


namespace largest_divisor_of_expression_of_even_x_l103_103745

theorem largest_divisor_of_expression_of_even_x (x : ℤ) (h_even : ∃ k : ℤ, x = 2 * k) :
  ∃ (d : ℤ), d = 240 ∧ d ∣ ((8 * x + 2) * (8 * x + 4) * (4 * x + 2)) :=
by
  sorry

end largest_divisor_of_expression_of_even_x_l103_103745


namespace simplify_radicals_l103_103015

theorem simplify_radicals :
  (Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by 
  sorry

end simplify_radicals_l103_103015


namespace abs_neg_three_l103_103184

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l103_103184


namespace time_for_train_to_pass_jogger_l103_103417

noncomputable def time_to_pass (s_jogger s_train : ℝ) (d_headstart l_train : ℝ) : ℝ :=
  let speed_jogger := s_jogger * (1000 / 3600)
  let speed_train := s_train * (1000 / 3600)
  let relative_speed := speed_train - speed_jogger
  let total_distance := d_headstart + l_train
  total_distance / relative_speed

theorem time_for_train_to_pass_jogger :
  time_to_pass 12 60 360 180 = 40.48 :=
by
  sorry

end time_for_train_to_pass_jogger_l103_103417


namespace trigonometric_identity_l103_103766

theorem trigonometric_identity 
  (θ : ℝ) 
  (h : Real.tan θ = 2) : 
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l103_103766


namespace monotonic_intervals_logarithmic_inequality_l103_103311

noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

theorem monotonic_intervals :
  (∀ x ∈ Set.Ioo 0 1, f x > f (x + 1E-9) ∧ f x < f (x - 1E-9)) ∧ 
  (∀ y ∈ Set.Ioi 1, f y < f (y + 1E-9) ∧ f y > f (y - 1E-9)) := sorry

theorem logarithmic_inequality (a : ℝ) (ha : a > 0) (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hneq : x1 ≠ x2)
  (h_eq1 : a * x1 + f x1 = x1^2 - x1) (h_eq2 : a * x2 + f x2 = x2^2 - x2) :
  Real.log x1 + Real.log x2 + 2 * Real.log a < 0 := sorry

end monotonic_intervals_logarithmic_inequality_l103_103311


namespace area_of_given_triangle_l103_103855

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_given_triangle :
  area_of_triangle (-2) 3 7 (-3) 4 6 = 31.5 :=
by
  sorry

end area_of_given_triangle_l103_103855


namespace initial_carrots_count_l103_103266

theorem initial_carrots_count (x : ℕ) (h1 : x - 2 + 21 = 31) : x = 12 := by
  sorry

end initial_carrots_count_l103_103266


namespace right_triangle_divisibility_l103_103980

theorem right_triangle_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (a % 3 = 0 ∨ b % 3 = 0) ∧ (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) :=
by
  -- skipping the proof
  sorry

end right_triangle_divisibility_l103_103980


namespace greg_savings_l103_103247

-- Definitions based on the conditions
def scooter_cost : ℕ := 90
def money_needed : ℕ := 33

-- The theorem to prove
theorem greg_savings : scooter_cost - money_needed = 57 := 
by
  -- sorry is used to skip the actual mathematical proof steps
  sorry

end greg_savings_l103_103247


namespace road_building_equation_l103_103794

theorem road_building_equation (x : ℝ) (hx : x > 0) :
  (9 / x - 12 / (x + 1) = 1 / 2) :=
sorry

end road_building_equation_l103_103794


namespace speed_of_man_in_still_water_l103_103347

theorem speed_of_man_in_still_water
  (V_m V_s : ℝ)
  (cond1 : V_m + V_s = 5)
  (cond2 : V_m - V_s = 7) :
  V_m = 6 :=
by
  sorry

end speed_of_man_in_still_water_l103_103347


namespace units_digit_of_product_l103_103926

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end units_digit_of_product_l103_103926


namespace total_value_of_remaining_books_l103_103579

-- initial definitions
def total_books : ℕ := 55
def hardback_books : ℕ := 10
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

-- calculate remaining books
def remaining_books : ℕ := total_books - books_sold

-- calculate remaining hardback and paperback books
def remaining_hardback_books : ℕ := hardback_books
def remaining_paperback_books : ℕ := remaining_books - remaining_hardback_books

-- calculate total values
def remaining_hardback_value : ℕ := remaining_hardback_books * hardback_price
def remaining_paperback_value : ℕ := remaining_paperback_books * paperback_price

-- total value of remaining books
def total_remaining_value : ℕ := remaining_hardback_value + remaining_paperback_value

theorem total_value_of_remaining_books : total_remaining_value = 510 := by
  -- calculation steps are skipped as instructed
  sorry

end total_value_of_remaining_books_l103_103579


namespace fraction_of_number_l103_103346

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l103_103346


namespace cost_of_each_book_l103_103175

noncomputable def cost_of_book (money_given money_left notebook_cost notebook_count book_count : ℕ) : ℕ :=
  (money_given - money_left - (notebook_count * notebook_cost)) / book_count

-- Conditions
def money_given : ℕ := 56
def money_left : ℕ := 14
def notebook_cost : ℕ := 4
def notebook_count : ℕ := 7
def book_count : ℕ := 2

-- Theorem stating that the cost of each book is $7 under given conditions
theorem cost_of_each_book : cost_of_book money_given money_left notebook_cost notebook_count book_count = 7 := by
  sorry

end cost_of_each_book_l103_103175


namespace remainder_of_product_mod_7_l103_103964

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l103_103964


namespace paint_cost_of_cube_l103_103092

theorem paint_cost_of_cube (side_length cost_per_kg coverage_per_kg : ℝ) (h₀ : side_length = 10) 
(h₁ : cost_per_kg = 60) (h₂ : coverage_per_kg = 20) : 
(cost_per_kg * (6 * (side_length^2) / coverage_per_kg) = 1800) :=
by
  sorry

end paint_cost_of_cube_l103_103092


namespace hall_area_proof_l103_103826

noncomputable def hall_length (L : ℕ) : ℕ := L
noncomputable def hall_width (L : ℕ) (W : ℕ) : ℕ := W
noncomputable def hall_area (L W : ℕ) : ℕ := L * W

theorem hall_area_proof (L W : ℕ) (h1 : W = 1 / 2 * L) (h2 : L - W = 15) :
  hall_area L W = 450 := by
  sorry

end hall_area_proof_l103_103826


namespace values_of_x_and_y_l103_103653

theorem values_of_x_and_y (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) : x < -2 ∧ y < -1 :=
by
  -- Proof goes here
  sorry

end values_of_x_and_y_l103_103653


namespace cos_double_angle_l103_103915

theorem cos_double_angle (α β : ℝ) (h1 : Real.sin (α - β) = 1 / 3) (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_double_angle_l103_103915


namespace minimum_value_at_zero_l103_103033

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

theorem minimum_value_at_zero : ∀ x : ℝ, f 0 ≤ f x :=
by
  sorry

end minimum_value_at_zero_l103_103033


namespace order_of_numbers_l103_103059

variable (a b c : ℝ)
variable (h₁ : a = (1 / 2) ^ (1 / 3))
variable (h₂ : b = (1 / 2) ^ (2 / 3))
variable (h₃ : c = (1 / 5) ^ (2 / 3))

theorem order_of_numbers (a b c : ℝ) (h₁ : a = (1 / 2) ^ (1 / 3)) (h₂ : b = (1 / 2) ^ (2 / 3)) (h₃ : c = (1 / 5) ^ (2 / 3)) :
  c < b ∧ b < a := 
by
  sorry

end order_of_numbers_l103_103059


namespace sean_total_spending_l103_103729

noncomputable def cost_first_bakery_euros : ℝ :=
  let almond_croissants := 2 * 4.00
  let salami_cheese_croissants := 3 * 5.00
  let total_before_discount := almond_croissants + salami_cheese_croissants
  total_before_discount * 0.90 -- 10% discount

noncomputable def cost_second_bakery_pounds : ℝ :=
  let plain_croissants := 3 * 3.50 -- buy-3-get-1-free
  let focaccia := 5.00
  let total_before_tax := plain_croissants + focaccia
  total_before_tax * 1.05 -- 5% tax

noncomputable def cost_cafe_dollars : ℝ :=
  let lattes := 3 * 3.00
  lattes * 0.85 -- 15% student discount

noncomputable def first_bakery_usd : ℝ :=
  cost_first_bakery_euros * 1.15 -- converting euros to dollars

noncomputable def second_bakery_usd : ℝ :=
  cost_second_bakery_pounds * 1.35 -- converting pounds to dollars

noncomputable def total_cost_sean_spends : ℝ :=
  first_bakery_usd + second_bakery_usd + cost_cafe_dollars

theorem sean_total_spending : total_cost_sean_spends = 53.44 :=
  by
  -- The proof can be handled here
  sorry

end sean_total_spending_l103_103729


namespace problem1_problem2_l103_103066

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x 2 ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 :=
  sorry -- the proof goes here

theorem problem2 (a : ℝ) (h₁ : 1 < a) : 
  (∀ x : ℝ, f x a + |x - 1| ≥ 1) ∧ (2 ≤ a) :=
  sorry -- the proof goes here

end problem1_problem2_l103_103066


namespace find_x_l103_103778

-- Definition of the binary operation
def binary_operation (a b c d : ℤ) : ℤ × ℤ :=
  (a - c, b + d)

-- Definition of our main theorem to be proved
theorem find_x (x y : ℤ) (h : binary_operation x y 2 3 = (4, 5)) : x = 6 :=
  by sorry

end find_x_l103_103778


namespace twentieth_century_years_as_powers_of_two_diff_l103_103169

theorem twentieth_century_years_as_powers_of_two_diff :
  ∀ (y : ℕ), (1900 ≤ y ∧ y < 2000) →
    ∃ (n k : ℕ), y = 2^n - 2^k ↔ y = 1984 ∨ y = 1920 := 
by
  sorry

end twentieth_century_years_as_powers_of_two_diff_l103_103169


namespace find_room_width_l103_103100

def room_height : ℕ := 12
def room_length : ℕ := 25
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3
def cost_per_sqft : ℕ := 8
def total_cost : ℕ := 7248

theorem find_room_width (x : ℕ) (h : 8 * (room_height * (2 * room_length + 2 * x) - (door_height * door_width + window_height * window_width * number_of_windows)) = total_cost) : 
  x = 15 :=
sorry

end find_room_width_l103_103100


namespace rationalize_expression_l103_103183

theorem rationalize_expression :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 :=
sorry

end rationalize_expression_l103_103183


namespace graph_of_equation_represents_three_lines_l103_103750

theorem graph_of_equation_represents_three_lines (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    ((a * x + b * y + c = 0) ∧ (a * x + b * y + c ≠ 0)) ∨
    ((a * x + b * y + c = 0) ∨ (a * x + b * y + c ≠ 0)) ∨
    (a * x + b * y + c = 0)) :=
by
  sorry

end graph_of_equation_represents_three_lines_l103_103750


namespace trapezium_area_l103_103218

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 12) :
  (1 / 2 * (a + b) * h = 228) :=
by
  sorry

end trapezium_area_l103_103218


namespace election_votes_l103_103271

theorem election_votes (V : ℝ) (h1 : ∃ geoff_votes : ℝ, geoff_votes = 0.01 * V)
                       (h2 : ∀ candidate_votes : ℝ, (candidate_votes > 0.51 * V) → candidate_votes > 0.51 * V)
                       (h3 : ∃ needed_votes : ℝ, needed_votes = 3000 ∧ 0.01 * V + needed_votes = 0.51 * V) :
                       V = 6000 :=
by sorry

end election_votes_l103_103271


namespace fred_dimes_l103_103438

theorem fred_dimes (initial_dimes borrowed_dimes : ℕ) (h1 : initial_dimes = 7) (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 :=
by
  sorry

end fred_dimes_l103_103438


namespace expected_non_allergic_l103_103473

theorem expected_non_allergic (p : ℝ) (n : ℕ) (h : p = 1 / 4) (hn : n = 300) : n * p = 75 :=
by sorry

end expected_non_allergic_l103_103473


namespace carpenter_wood_split_l103_103840

theorem carpenter_wood_split :
  let original_length : ℚ := 35 / 8
  let first_cut : ℚ := 5 / 3
  let second_cut : ℚ := 9 / 4
  let remaining_length := original_length - first_cut - second_cut
  let part_length := remaining_length / 3
  part_length = 11 / 72 :=
sorry

end carpenter_wood_split_l103_103840


namespace min_forget_all_three_l103_103707

theorem min_forget_all_three (total_students students_forgot_gloves students_forgot_scarves students_forgot_hats : ℕ) (h_total : total_students = 60) (h_gloves : students_forgot_gloves = 55) (h_scarves : students_forgot_scarves = 52) (h_hats : students_forgot_hats = 50) :
  ∃ min_students_forget_three, min_students_forget_three = total_students - (total_students - students_forgot_gloves + total_students - students_forgot_scarves + total_students - students_forgot_hats) :=
by
  use 37
  sorry

end min_forget_all_three_l103_103707


namespace intersection_points_l103_103393

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

theorem intersection_points :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ ¬(x = x ∧ y = y) → 0 = 1 :=
sorry

end intersection_points_l103_103393


namespace min_value_fraction_l103_103756

variable (a b : ℝ)

theorem min_value_fraction (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 2 * b = 1) : 
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_l103_103756


namespace sum_of_products_is_50_l103_103919

theorem sum_of_products_is_50
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a + b + c = 16) :
  a * b + b * c + a * c = 50 :=
by
  sorry

end sum_of_products_is_50_l103_103919


namespace Jana_new_walking_speed_l103_103746

variable (minutes : ℕ) (distance1 distance2 : ℝ)

-- Given conditions
def minutes_taken_to_walk := 30
def current_distance := 2
def new_distance := 3
def time_in_hours := minutes / 60

-- Define outcomes
def current_speed_per_minute := current_distance / minutes
def current_speed_per_hour := current_speed_per_minute * 60
def required_speed_per_minute := new_distance / minutes
def required_speed_per_hour := required_speed_per_minute * 60

-- Final statement to prove
theorem Jana_new_walking_speed : required_speed_per_hour = 6 := by
  sorry

end Jana_new_walking_speed_l103_103746


namespace incorrect_expression_D_l103_103603

noncomputable def E : ℝ := sorry
def R : ℕ := sorry
def S : ℕ := sorry
def m : ℕ := sorry
def t : ℕ := sorry

-- E is a repeating decimal
-- R is the non-repeating part of E with m digits
-- S is the repeating part of E with t digits

theorem incorrect_expression_D : ¬ (10^m * (10^t - 1) * E = S * (R - 1)) :=
sorry

end incorrect_expression_D_l103_103603


namespace clock_rings_in_a_day_l103_103706

theorem clock_rings_in_a_day (intervals : ℕ) (hours_in_a_day : ℕ) (time_between_rings : ℕ) : 
  intervals = hours_in_a_day / time_between_rings + 1 → intervals = 7 :=
sorry

end clock_rings_in_a_day_l103_103706


namespace simplify_fraction_l103_103619

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (10 * x * y^2) / (5 * x * y) = 2 * y := 
by
  sorry

end simplify_fraction_l103_103619


namespace max_value_of_a_max_value_reached_l103_103586

theorem max_value_of_a (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 :=
by
  sorry

theorem max_value_reached (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  ∃ a, a = Real.sqrt 6 / 3 :=
by
  sorry

end max_value_of_a_max_value_reached_l103_103586


namespace dragon_heads_belong_to_dragons_l103_103785

def truthful (H : ℕ) : Prop := 
  H = 1 ∨ H = 3

def lying (H : ℕ) : Prop := 
  H = 2 ∨ H = 4

def head1_statement : Prop := truthful 1
def head2_statement : Prop := truthful 3
def head3_statement : Prop := ¬ truthful 2
def head4_statement : Prop := lying 3

theorem dragon_heads_belong_to_dragons :
  head1_statement ∧ head2_statement ∧ head3_statement ∧ head4_statement →
  (∀ H, (truthful H ↔ H = 1 ∨ H = 3) ∧ (lying H ↔ H = 2 ∨ H = 4)) :=
by
  sorry

end dragon_heads_belong_to_dragons_l103_103785


namespace intersection_P_Q_l103_103537

def P (x : ℝ) : Prop := x^2 - x - 2 ≥ 0

def Q (y : ℝ) : Prop := ∃ x, P x ∧ y = (1/2) * x^2 - 1

theorem intersection_P_Q :
  {m | ∃ (x : ℝ), P x ∧ m = (1/2) * x^2 - 1} = {m | m ≥ 2} := sorry

end intersection_P_Q_l103_103537


namespace find_solutions_l103_103941

-- Define the conditions
variable (n : ℕ)
noncomputable def valid_solution (a b c d : ℕ) : Prop := 
  a^2 + b^2 + c^2 + d^2 = 7 * 4^n

-- Define each possible solution
def sol1 : ℕ × ℕ × ℕ × ℕ := (5 * 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1))
def sol2 : ℕ × ℕ × ℕ × ℕ := (2 ^ (n + 1), 2 ^ n, 2 ^ n, 2 ^ n)
def sol3 : ℕ × ℕ × ℕ × ℕ := (3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 2 ^ (n - 1))

-- State the theorem
theorem find_solutions (a b c d : ℕ) (n : ℕ) :
  valid_solution n a b c d →
  (a, b, c, d) = sol1 n ∨
  (a, b, c, d) = sol2 n ∨
  (a, b, c, d) = sol3 n :=
sorry

end find_solutions_l103_103941


namespace angle_Q_measure_in_triangle_PQR_l103_103804

theorem angle_Q_measure_in_triangle_PQR (angle_R angle_Q angle_P : ℝ) (h1 : angle_P = 3 * angle_R) (h2 : angle_Q = angle_R) (h3 : angle_R + angle_Q + angle_P = 180) : angle_Q = 36 :=
by {
  -- Placeholder for the proof, which is not required as per the instructions
  sorry
}

end angle_Q_measure_in_triangle_PQR_l103_103804


namespace marble_ratio_correct_l103_103268

-- Necessary given conditions
variables (x : ℕ) (Ben_initial John_initial : ℕ) (John_post Ben_post : ℕ)
variables (h1 : Ben_initial = 18)
variables (h2 : John_initial = 17)
variables (h3 : Ben_post = Ben_initial - x)
variables (h4 : John_post = John_initial + x)
variables (h5 : John_post = Ben_post + 17)

-- Define the ratio of the number of marbles Ben gave to John to the number of marbles Ben had initially
def marble_ratio := (x : ℕ) / Ben_initial

-- The theorem we want to prove
theorem marble_ratio_correct (h1 : Ben_initial = 18) (h2 : John_initial = 17) (h3 : Ben_post = Ben_initial - x)
(h4 : John_post = John_initial + x) (h5 : John_post = Ben_post + 17) : marble_ratio x Ben_initial = 1/2 := by 
  sorry

end marble_ratio_correct_l103_103268


namespace JaneReadingSpeed_l103_103105

theorem JaneReadingSpeed (total_pages read_second_half_speed total_days pages_first_half days_first_half_speed : ℕ)
  (h1 : total_pages = 500)
  (h2 : read_second_half_speed = 5)
  (h3 : total_days = 75)
  (h4 : pages_first_half = 250)
  (h5 : days_first_half_speed = pages_first_half / (total_days - (pages_first_half / read_second_half_speed))) :
  days_first_half_speed = 10 := by
  sorry

end JaneReadingSpeed_l103_103105


namespace triangle_area_solutions_l103_103888

theorem triangle_area_solutions (ABC BDE : ℝ) (k : ℝ) (h₁ : BDE = k^2) : 
  S >= 4 * k^2 ∧ (if S = 4 * k^2 then solutions = 1 else solutions = 2) :=
by
  sorry

end triangle_area_solutions_l103_103888


namespace greatest_number_of_kits_l103_103768

-- Given conditions
def bottles_of_water := 20
def cans_of_food := 12
def flashlights := 30
def blankets := 18

def no_more_than_10_items_per_kit (kits : ℕ) := 
  (bottles_of_water / kits ≤ 10) ∧ 
  (cans_of_food / kits ≤ 10) ∧ 
  (flashlights / kits ≤ 10) ∧ 
  (blankets / kits ≤ 10)

def greater_than_or_equal_to_5_kits (kits : ℕ) := kits ≥ 5

def all_items_distributed_equally (kits : ℕ) := 
  (bottles_of_water % kits = 0) ∧ 
  (cans_of_food % kits = 0) ∧ 
  (flashlights % kits = 0) ∧ 
  (blankets % kits = 0)

-- Proof goal
theorem greatest_number_of_kits : 
  ∃ kits : ℕ, 
    no_more_than_10_items_per_kit kits ∧ 
    greater_than_or_equal_to_5_kits kits ∧ 
    all_items_distributed_equally kits ∧ 
    kits = 6 := 
sorry

end greatest_number_of_kits_l103_103768


namespace trigonometric_identity_l103_103144

theorem trigonometric_identity :
  3 * Real.arcsin (Real.sqrt 3 / 2) - Real.arctan (-1) - Real.arccos 0 = (3 * Real.pi) / 4 := 
by
  sorry

end trigonometric_identity_l103_103144


namespace cesar_watched_fraction_l103_103058

theorem cesar_watched_fraction
  (total_seasons : ℕ) (episodes_per_season : ℕ) (remaining_episodes : ℕ)
  (h1 : total_seasons = 12)
  (h2 : episodes_per_season = 20)
  (h3 : remaining_episodes = 160) :
  (total_seasons * episodes_per_season - remaining_episodes) / (total_seasons * episodes_per_season) = 1 / 3 := 
sorry

end cesar_watched_fraction_l103_103058


namespace cube_root_less_than_five_count_l103_103095

theorem cube_root_less_than_five_count :
  (∃ n : ℕ, n = 124 ∧ ∀ x : ℕ, 1 ≤ x → x < 5^3 → x < 125) := 
sorry

end cube_root_less_than_five_count_l103_103095


namespace problem_l103_103790

theorem problem (n : ℕ) (h : n = 8 ^ 2022) : n / 4 = 4 ^ 3032 := 
sorry

end problem_l103_103790


namespace distinct_paths_in_grid_l103_103148

def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem distinct_paths_in_grid :
  number_of_paths 7 8 = 6435 :=
by
  sorry

end distinct_paths_in_grid_l103_103148


namespace hallie_number_of_paintings_sold_l103_103055

/-- 
Hallie is an artist. She wins an art contest, and she receives a $150 prize. 
She sells some of her paintings for $50 each. 
She makes a total of $300 from her art. 
How many paintings did she sell?
-/
theorem hallie_number_of_paintings_sold 
    (prize : ℕ)
    (price_per_painting : ℕ)
    (total_earnings : ℕ)
    (prize_eq : prize = 150)
    (price_eq : price_per_painting = 50)
    (total_eq : total_earnings = 300) :
    (total_earnings - prize) / price_per_painting = 3 :=
by
  sorry

end hallie_number_of_paintings_sold_l103_103055


namespace luke_bought_stickers_l103_103143

theorem luke_bought_stickers :
  ∀ (original birthday given_to_sister used_on_card left total_before_buying stickers_bought : ℕ),
  original = 20 →
  birthday = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  left = 39 →
  total_before_buying = original + birthday →
  stickers_bought = (left + given_to_sister + used_on_card) - total_before_buying →
  stickers_bought = 12 :=
by
  intros
  sorry

end luke_bought_stickers_l103_103143


namespace num_ints_between_sqrt2_and_sqrt32_l103_103724

theorem num_ints_between_sqrt2_and_sqrt32 : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ k : ℤ, (2 ≤ k) ∧ (k ≤ 5)) :=
by
  sorry

end num_ints_between_sqrt2_and_sqrt32_l103_103724


namespace number_of_real_solutions_l103_103948

noncomputable def system_of_equations (n : ℕ) (a b c : ℝ) (x : Fin n → ℝ) : Prop :=
∀ i : Fin n, a * (x i) ^ 2 + b * (x i) + c = x (⟨(i + 1) % n, sorry⟩)

theorem number_of_real_solutions
  (a b c : ℝ)
  (h : a ≠ 0)
  (n : ℕ)
  (x : Fin n → ℝ) :
  (b - 1) ^ 2 - 4 * a * c < 0 → ¬(∃ x : Fin n → ℝ, system_of_equations n a b c x) ∧
  (b - 1) ^ 2 - 4 * a * c = 0 → ∃! x : Fin n → ℝ, system_of_equations n a b c x ∧
  (b - 1) ^ 2 - 4 * a * c > 0 → ∃ x : Fin n → ℝ, ∃ y : Fin n → ℝ, x ≠ y ∧ system_of_equations n a b c x ∧ system_of_equations n a b c y := 
sorry

end number_of_real_solutions_l103_103948


namespace negation_of_p_l103_103331

def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A (x : ℤ) : Prop := is_odd x
def B (x : ℤ) : Prop := is_even x
def p : Prop := ∀ x, A x → B (2 * x)

theorem negation_of_p : ¬ p ↔ ∃ x, A x ∧ ¬ B (2 * x) :=
by
  -- problem statement equivalent in Lean 4
  sorry

end negation_of_p_l103_103331


namespace scientific_notation_of_150000000000_l103_103637

theorem scientific_notation_of_150000000000 :
  150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_of_150000000000_l103_103637


namespace remainder_when_divided_by_39_l103_103167

theorem remainder_when_divided_by_39 (N k : ℤ) (h : N = 13 * k + 4) : (N % 39) = 4 :=
sorry

end remainder_when_divided_by_39_l103_103167


namespace cone_surface_area_is_correct_l103_103568

noncomputable def cone_surface_area (central_angle_degrees : ℝ) (sector_area : ℝ) : ℝ :=
  if central_angle_degrees = 120 ∧ sector_area = 3 * Real.pi then 4 * Real.pi else 0

theorem cone_surface_area_is_correct :
  cone_surface_area 120 (3 * Real.pi) = 4 * Real.pi :=
by
  -- proof would go here
  sorry

end cone_surface_area_is_correct_l103_103568


namespace no_adjacent_standing_probability_l103_103040

noncomputable def probability_no_adjacent_standing : ℚ := 
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 123
  favorable_outcomes / total_outcomes

theorem no_adjacent_standing_probability :
  probability_no_adjacent_standing = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l103_103040


namespace exists_isosceles_triangle_containing_l103_103465

variables {A B C X Y Z : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

noncomputable def triangle (a b c : A) := a + b + c

def is_triangle (a b c : A) := a + b > c ∧ b + c > a ∧ c + a > b

def isosceles_triangle (a b c : A) := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem exists_isosceles_triangle_containing
  (a b c : A)
  (h1 : a < 1)
  (h2 : b < 1)
  (h3 : c < 1)
  (h_ABC : is_triangle a b c)
  : ∃ (x y z : A), isosceles_triangle x y z ∧ is_triangle x y z ∧ a < x ∧ b < y ∧ c < z ∧ x < 1 ∧ y < 1 ∧ z < 1 :=
sorry

end exists_isosceles_triangle_containing_l103_103465


namespace angle_east_northwest_l103_103200

def num_spokes : ℕ := 12
def central_angle : ℕ := 360 / num_spokes
def angle_between (start_dir end_dir : ℕ) : ℕ := (end_dir - start_dir) * central_angle

theorem angle_east_northwest : angle_between 3 9 = 90 := sorry

end angle_east_northwest_l103_103200


namespace find_D_l103_103433

theorem find_D (A B D : ℕ) (h1 : (100 * A + 10 * B + D) * (A + B + D) = 1323) (h2 : A ≥ B) : D = 1 :=
sorry

end find_D_l103_103433


namespace example_problem_l103_103864

-- Define vectors a and b with the given conditions
def a (k : ℝ) : ℝ × ℝ := (2, k)
def b : ℝ × ℝ := (6, 4)

-- Define the condition that vectors are perpendicular
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Calculate the sum of two vectors
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Check if a vector is collinear
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- The main theorem with the given conditions
theorem example_problem (k : ℝ) (hk : perpendicular (a k) b) :
  collinear (vector_add (a k) b) (-16, -2) :=
by
  sorry

end example_problem_l103_103864


namespace rosie_purchase_price_of_art_piece_l103_103950

-- Define the conditions as hypotheses
variables (P : ℝ)
variables (future_value increase : ℝ)

-- Given conditions
def conditions := future_value = 3 * P ∧ increase = 8000 ∧ increase = future_value - P

-- The statement to be proved
theorem rosie_purchase_price_of_art_piece (h : conditions P future_value increase) : P = 4000 :=
sorry

end rosie_purchase_price_of_art_piece_l103_103950


namespace m_greater_than_one_l103_103215

variables {x m : ℝ}

def p : Prop := -2 ≤ x ∧ x ≤ 11
def q : Prop := 1 - 3 * m ≤ x ∧ x ≤ 3 + m

theorem m_greater_than_one (h : ¬(x^2 - 2 * x + m ≤ 0)) : m > 1 :=
sorry

end m_greater_than_one_l103_103215


namespace proof_problem_l103_103584

theorem proof_problem (x : ℝ) (h1 : x = 3) (h2 : 2 * x ≠ 5) (h3 : x + 5 ≠ 3) 
                      (h4 : 7 - x ≠ 2) (h5 : 6 + 2 * x ≠ 14) :
    3 * x - 1 = 8 :=
by 
  sorry

end proof_problem_l103_103584


namespace route_B_no_quicker_l103_103594

noncomputable def time_route_A (distance_A : ℕ) (speed_A : ℕ) : ℕ :=
(distance_A * 60) / speed_A

noncomputable def time_route_B (distance_B : ℕ) (speed_B1 : ℕ) (speed_B2 : ℕ) : ℕ :=
  let distance_B1 := distance_B - 1
  let distance_B2 := 1
  (distance_B1 * 60) / speed_B1 + (distance_B2 * 60) / speed_B2

theorem route_B_no_quicker : time_route_A 8 40 = time_route_B 6 50 10 :=
by
  sorry

end route_B_no_quicker_l103_103594


namespace fractional_shaded_area_l103_103849

noncomputable def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)

theorem fractional_shaded_area :
  let a := (7 : ℚ) / 16
  let r := (1 : ℚ) / 16
  geometric_series_sum a r = 7 / 15 :=
by
  sorry

end fractional_shaded_area_l103_103849


namespace odd_function_extended_l103_103454

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≥ 0 then 
    x * Real.log (x + 1)
  else 
    x * Real.log (-x + 1)

theorem odd_function_extended : (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = x * Real.log (x + 1)) →
  (∀ x : ℝ, x < 0 → f x = x * Real.log (-x + 1)) :=
by
  intros h_odd h_def_neg
  sorry

end odd_function_extended_l103_103454


namespace f_2009_l103_103714

noncomputable def f : ℝ → ℝ := sorry -- This will be defined by the conditions.

axiom even_f (x : ℝ) : f x = f (-x)
axiom periodic_f (x : ℝ) : f (x + 6) = f x + f 3
axiom f_one : f 1 = 2

theorem f_2009 : f 2009 = 2 :=
by {
  -- The proof would go here, summarizing the logical steps derived in the previous sections.
  sorry
}

end f_2009_l103_103714


namespace remaining_sausage_meat_l103_103312

-- Define the conditions
def total_meat_pounds : ℕ := 10
def sausage_links : ℕ := 40
def links_eaten_by_Brandy : ℕ := 12
def pounds_to_ounces : ℕ := 16

-- Calculate the remaining sausage meat and prove the correctness
theorem remaining_sausage_meat :
  (total_meat_pounds * pounds_to_ounces - links_eaten_by_Brandy * (total_meat_pounds * pounds_to_ounces / sausage_links)) = 112 :=
by
  sorry

end remaining_sausage_meat_l103_103312


namespace sector_area_is_2pi_l103_103566

noncomputable def sectorArea (l : ℝ) (R : ℝ) : ℝ :=
  (1 / 2) * l * R

theorem sector_area_is_2pi (R : ℝ) (l : ℝ) (hR : R = 4) (hl : l = π) :
  sectorArea l R = 2 * π :=
by
  sorry

end sector_area_is_2pi_l103_103566


namespace binom_14_11_l103_103629

open Nat

theorem binom_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end binom_14_11_l103_103629


namespace third_bouquet_carnations_l103_103817

/--
Trevor buys three bouquets of carnations. The first included 9 carnations, and the second included 14 carnations. If the average number of carnations in the bouquets is 12, then the third bouquet contains 13 carnations.
-/
theorem third_bouquet_carnations (n1 n2 n3 : ℕ)
  (h1 : n1 = 9)
  (h2 : n2 = 14)
  (h3 : (n1 + n2 + n3) / 3 = 12) :
  n3 = 13 :=
by
  sorry

end third_bouquet_carnations_l103_103817


namespace exists_person_with_girls_as_neighbors_l103_103356

theorem exists_person_with_girls_as_neighbors (boys girls : Nat) (sitting : Nat) 
  (h_boys : boys = 25) (h_girls : girls = 25) (h_sitting : sitting = boys + girls) :
  ∃ p : Nat, p < sitting ∧ (p % 2 = 1 → p.succ % sitting % 2 = 0) := 
by
  sorry

end exists_person_with_girls_as_neighbors_l103_103356


namespace algebraic_identity_l103_103955

variables {R : Type*} [CommRing R] (a b : R)

theorem algebraic_identity : 2 * (a - b) + 3 * b = 2 * a + b :=
by
  sorry

end algebraic_identity_l103_103955


namespace vertical_strips_count_l103_103797

theorem vertical_strips_count (a b x y : ℕ)
  (h_outer : 2 * a + 2 * b = 50)
  (h_inner : 2 * x + 2 * y = 32)
  (h_strips : a + x = 20) :
  b + y = 21 :=
by
  have h1 : a + b = 25 := by
    linarith
  have h2 : x + y = 16 := by
    linarith
  linarith


end vertical_strips_count_l103_103797


namespace inequality_x_y_l103_103654

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l103_103654


namespace find_z_l103_103651

noncomputable def z := {z : ℂ | ∃ i : ℂ, i^2 = -1 ∧ i * z = i - 1}

theorem find_z (i : ℂ) (hi : i^2 = -1) : ∃ z : ℂ, i * z = i - 1 ∧ z = 1 + i := by
  use 1 + i
  sorry

end find_z_l103_103651


namespace solve_eq1_solve_eq2_l103_103203

-- Definition of the first equation
def eq1 (x : ℝ) : Prop := (1 / 2) * x^2 - 8 = 0

-- Definition of the second equation
def eq2 (x : ℝ) : Prop := (x - 5)^3 = -27

-- Proof statement for the value of x in the first equation
theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 4 ∨ x = -4 := by
  sorry

-- Proof statement for the value of x in the second equation
theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l103_103203


namespace friends_meet_probability_l103_103675

noncomputable def probability_of_meeting :=
  let duration_total := 60 -- Total duration from 14:00 to 15:00 in minutes
  let duration_meeting := 30 -- Duration they can meet from 14:00 to 14:30 in minutes
  duration_meeting / duration_total

theorem friends_meet_probability : probability_of_meeting = 1 / 2 := by
  sorry

end friends_meet_probability_l103_103675


namespace find_r_l103_103937

variable {x y r k : ℝ}

theorem find_r (h1 : y^2 + 4 * y + 4 + Real.sqrt (x + y + k) = 0)
               (h2 : r = |x * y|) :
    r = 2 :=
by
  sorry

end find_r_l103_103937


namespace chuck_total_play_area_l103_103704

noncomputable def chuck_play_area (leash_radius : ℝ) : ℝ :=
  let middle_arc_area := (1 / 2) * Real.pi * leash_radius^2
  let corner_arc_area := 2 * (1 / 4) * Real.pi * leash_radius^2
  middle_arc_area + corner_arc_area

theorem chuck_total_play_area (leash_radius : ℝ) (shed_width shed_length : ℝ) 
  (h_radius : leash_radius = 4) (h_width : shed_width = 4) (h_length : shed_length = 6) :
  chuck_play_area leash_radius = 16 * Real.pi :=
by
  sorry

end chuck_total_play_area_l103_103704


namespace decagon_diagonal_relation_l103_103934

-- Define side length, shortest diagonal, and longest diagonal in a regular decagon
variable (a b d : ℝ)
variable (h1 : a > 0) -- Side length must be positive
variable (h2 : b > 0) -- Shortest diagonal length must be positive
variable (h3 : d > 0) -- Longest diagonal length must be positive

theorem decagon_diagonal_relation (ha : d^2 = 5 * a^2) (hb : b^2 = 3 * a^2) : b^2 = a * d :=
sorry

end decagon_diagonal_relation_l103_103934


namespace complement_of_A_in_U_l103_103806

def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}
def CU_A : Set ℤ := {x ∈ U | x ∉ A}

theorem complement_of_A_in_U :
  CU_A = {-2, 1, 5} :=
by
  -- Proof goes here
  sorry

end complement_of_A_in_U_l103_103806


namespace acute_triangle_exterior_angles_obtuse_l103_103245

theorem acute_triangle_exterior_angles_obtuse
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  ∀ α β γ, α = A + B → β = B + C → γ = C + A → α > π / 2 ∧ β > π / 2 ∧ γ > π / 2 :=
by
  sorry

end acute_triangle_exterior_angles_obtuse_l103_103245


namespace smallest_integer_geq_l103_103851

theorem smallest_integer_geq : ∃ (n : ℤ), (n^2 - 9*n + 18 ≥ 0) ∧ ∀ (m : ℤ), (m^2 - 9*m + 18 ≥ 0) → n ≤ m :=
by
  sorry

end smallest_integer_geq_l103_103851


namespace find_second_dimension_l103_103722

variable (l h w : ℕ)
variable (cost_per_sqft total_cost : ℕ)
variable (surface_area : ℕ)

def insulation_problem_conditions (l : ℕ) (h : ℕ) (cost_per_sqft : ℕ) (total_cost : ℕ) (w : ℕ) (surface_area : ℕ) : Prop :=
  l = 4 ∧ h = 3 ∧ cost_per_sqft = 20 ∧ total_cost = 1880 ∧ surface_area = (2 * l * w + 2 * l * h + 2 * w * h)

theorem find_second_dimension (l h w : ℕ) (cost_per_sqft total_cost surface_area : ℕ) :
  insulation_problem_conditions l h cost_per_sqft total_cost w surface_area →
  surface_area = 94 →
  w = 5 :=
by
  intros
  simp [insulation_problem_conditions] at *
  sorry

end find_second_dimension_l103_103722


namespace max_value_of_x3_div_y4_l103_103476

theorem max_value_of_x3_div_y4 (x y : ℝ) (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) :
  ∃ (k : ℝ), k = 27 ∧ ∀ (z : ℝ), z = x^3 / y^4 → z ≤ k :=
by
  sorry

end max_value_of_x3_div_y4_l103_103476


namespace people_left_line_l103_103741

theorem people_left_line (L : ℕ) (h_initial : 31 - L + 25 = 31) : L = 25 :=
by
  -- proof will go here
  sorry

end people_left_line_l103_103741


namespace division_remainder_unique_u_l103_103094

theorem division_remainder_unique_u :
  ∃! u : ℕ, ∃ q : ℕ, 15 = u * q + 4 ∧ u > 4 :=
sorry

end division_remainder_unique_u_l103_103094


namespace number_of_desired_numbers_l103_103038

-- Define a predicate for a four-digit number with the thousands digit 3
def isDesiredNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n / 1000) % 10 = 3

-- Statement of the theorem
theorem number_of_desired_numbers : 
  ∃ k, k = 1000 ∧ (∀ n, isDesiredNumber n ↔ 3000 ≤ n ∧ n < 4000) := 
by
  -- Proof omitted, using sorry to skip the proof
  sorry

end number_of_desired_numbers_l103_103038


namespace sum_of_ages_today_l103_103521

variable (RizaWas25WhenSonBorn : ℕ) (SonCurrentAge : ℕ) (SumOfAgesToday : ℕ)

theorem sum_of_ages_today (h1 : RizaWas25WhenSonBorn = 25) (h2 : SonCurrentAge = 40) : SumOfAgesToday = 105 :=
by
  sorry

end sum_of_ages_today_l103_103521


namespace functional_eq_solution_l103_103998

variable (f g : ℝ → ℝ)

theorem functional_eq_solution (h : ∀ x y : ℝ, f (x + y * g x) = g x + x * f y) : f = id := 
sorry

end functional_eq_solution_l103_103998


namespace negation_of_P_l103_103683

-- Define the proposition P
def P (x : ℝ) : Prop := x^2 = 1 → x = 1

-- Define the negation of the proposition P
def neg_P (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

theorem negation_of_P (x : ℝ) : ¬P x ↔ neg_P x := by
  sorry

end negation_of_P_l103_103683


namespace seongjun_ttakji_count_l103_103567

variable (S A : ℕ)

theorem seongjun_ttakji_count (h1 : (3/4 : ℚ) * S - 25 = 7 * (A - 50)) (h2 : A = 100) : S = 500 :=
sorry

end seongjun_ttakji_count_l103_103567


namespace desiree_age_l103_103559

theorem desiree_age (D C G Gr : ℕ) 
  (h1 : D = 2 * C)
  (h2 : D + 30 = (2 * (C + 30)) / 3 + 14)
  (h3 : G = D + C)
  (h4 : G + 20 = 3 * (D - C))
  (h5 : Gr = (D + 10) * (C + 10) / 2) : 
  D = 6 := 
sorry

end desiree_age_l103_103559


namespace max_d_for_range_of_fx_l103_103287

theorem max_d_for_range_of_fx : 
  ∀ (d : ℝ), (∃ x : ℝ, x^2 + 4*x + d = -3) → d ≤ 1 := 
by
  sorry

end max_d_for_range_of_fx_l103_103287


namespace A_is_9_years_older_than_B_l103_103755

-- Define the conditions
variables (A_years B_years : ℕ)

def given_conditions : Prop :=
  B_years = 39 ∧ A_years + 10 = 2 * (B_years - 10)

-- Theorem to prove the correct answer
theorem A_is_9_years_older_than_B (h : given_conditions A_years B_years) : A_years - B_years = 9 :=
by
  sorry

end A_is_9_years_older_than_B_l103_103755


namespace binomial_arithmetic_sequence_iff_l103_103623

open Nat

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  n.choose k 

-- Conditions
def is_arithmetic_sequence (n k : ℕ) : Prop :=
  binomial n (k-1) - 2 * binomial n k + binomial n (k+1) = 0

-- Statement to prove
theorem binomial_arithmetic_sequence_iff (u : ℕ) (u_gt2 : u > 2) :
  ∃ (n k : ℕ), (n = u^2 - 2) ∧ (k = binomial u 2 - 1 ∨ k = binomial (u+1) 2 - 1) 
  ↔ is_arithmetic_sequence n k := 
sorry

end binomial_arithmetic_sequence_iff_l103_103623


namespace ball_box_distribution_l103_103561

theorem ball_box_distribution : (∃ (f : Fin 4 → Fin 2), true) ∧ (∀ (f : Fin 4 → Fin 2), true) → ∃ (f : Fin 4 → Fin 2), true ∧ f = 16 :=
by sorry

end ball_box_distribution_l103_103561


namespace current_population_l103_103006

theorem current_population (initial_population deaths_leaving_percentage : ℕ) (current_population : ℕ) :
  initial_population = 3161 → deaths_leaving_percentage = 5 →
  deaths_leaving_percentage / 100 * initial_population + deaths_leaving_percentage * (initial_population - deaths_leaving_percentage / 100 * initial_population) / 100 = initial_population - current_population →
  current_population = 2553 :=
 by
  sorry

end current_population_l103_103006


namespace comparing_exponents_l103_103269

theorem comparing_exponents {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end comparing_exponents_l103_103269


namespace sandbox_width_l103_103467

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end sandbox_width_l103_103467


namespace dividend_calculation_l103_103927

theorem dividend_calculation
  (divisor : Int)
  (quotient : Int)
  (remainder : Int)
  (dividend : Int)
  (h_divisor : divisor = 800)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968)
  (h_dividend : dividend = (divisor * quotient) + remainder) :
  dividend = 474232 := by
  sorry

end dividend_calculation_l103_103927


namespace range_of_a_exists_x_l103_103084

theorem range_of_a_exists_x (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x - x ^ 2 ≥ a) ↔ a ≤ 1 := 
sorry

end range_of_a_exists_x_l103_103084


namespace g_diff_l103_103943

noncomputable section

-- Definition of g(n) as given in the problem statement
def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2)^n

-- The statement to prove g(n+2) - g(n) = -1/4 * g(n)
theorem g_diff (n : ℕ) : g (n + 2) - g n = -1 / 4 * g n :=
by
  sorry

end g_diff_l103_103943


namespace binom_2024_1_l103_103592

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_2024_1 : binomial 2024 1 = 2024 := by
  sorry

end binom_2024_1_l103_103592


namespace geese_survived_first_year_l103_103135

-- Definitions based on the conditions
def total_eggs := 900
def hatch_rate := 2 / 3
def survive_first_month_rate := 3 / 4
def survive_first_year_rate := 2 / 5

-- Definitions derived from the conditions
def hatched_geese := total_eggs * hatch_rate
def survived_first_month := hatched_geese * survive_first_month_rate
def survived_first_year := survived_first_month * survive_first_year_rate

-- Target proof statement
theorem geese_survived_first_year : survived_first_year = 180 := by
  sorry

end geese_survived_first_year_l103_103135


namespace binomial_sum_l103_103485

theorem binomial_sum (n k : ℕ) (h : n = 10) (hk : k = 3) :
  Nat.choose n k + Nat.choose n (n - k) = 240 :=
by
  -- placeholder for actual proof
  sorry

end binomial_sum_l103_103485


namespace compute_n_l103_103847

theorem compute_n (n : ℕ) : 5^n = 5 * 25^(3/2) * 125^(5/3) → n = 9 :=
by
  sorry

end compute_n_l103_103847


namespace lamp_pricing_problem_l103_103793

theorem lamp_pricing_problem
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (sales_decrease_rate : ℝ)
  (desired_profit : ℝ) :
  purchase_price = 30 →
  initial_selling_price = 40 →
  initial_sales_volume = 600 →
  sales_decrease_rate = 10 →
  desired_profit = 10000 →
  (∃ (selling_price : ℝ) (sales_volume : ℝ), selling_price = 50 ∧ sales_volume = 500) :=
by
  intros h_purchase h_initial_selling h_initial_sales h_sales_decrease h_desired_profit
  sorry

end lamp_pricing_problem_l103_103793


namespace employee_B_payment_l103_103451

theorem employee_B_payment (total_payment A_payment B_payment : ℝ) 
    (h1 : total_payment = 450) 
    (h2 : A_payment = 1.5 * B_payment) 
    (h3 : total_payment = A_payment + B_payment) : 
    B_payment = 180 := 
by
  sorry

end employee_B_payment_l103_103451


namespace sum_of_ages_l103_103869

theorem sum_of_ages (X_c Y_c : ℕ) (h1 : X_c = 45) 
  (h2 : X_c - 3 = 2 * (Y_c - 3)) : 
  (X_c + 7) + (Y_c + 7) = 83 := 
by
  sorry

end sum_of_ages_l103_103869


namespace lemonade_in_pitcher_l103_103023

theorem lemonade_in_pitcher (iced_tea lemonade total_pitcher total_in_drink lemonade_ratio : ℚ)
  (h1 : iced_tea = 1/4)
  (h2 : lemonade = 5/4)
  (h3 : total_in_drink = iced_tea + lemonade)
  (h4 : lemonade_ratio = lemonade / total_in_drink)
  (h5 : total_pitcher = 18) :
  (total_pitcher * lemonade_ratio) = 15 :=
by
  sorry

end lemonade_in_pitcher_l103_103023


namespace odd_number_expression_parity_l103_103984

theorem odd_number_expression_parity (o n : ℕ) (ho : ∃ k : ℕ, o = 2 * k + 1) :
  (o^2 + n * o) % 2 = 1 ↔ n % 2 = 0 :=
by
  sorry

end odd_number_expression_parity_l103_103984


namespace count_right_triangles_with_conditions_l103_103517

theorem count_right_triangles_with_conditions :
  ∃ n : ℕ, n = 10 ∧
    (∀ (a b : ℕ),
      (a ^ 2 + b ^ 2 = (b + 2) ^ 2) →
      (b < 100) →
      (∃ k : ℕ, a = 2 * k ∧ k ^ 2 = b + 1) →
      n = 10) :=
by
  -- The proof goes here
  sorry

end count_right_triangles_with_conditions_l103_103517


namespace continuity_at_x0_l103_103726

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4
def x0 := 3

theorem continuity_at_x0 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_at_x0_l103_103726


namespace find_length_l103_103418

-- Define the perimeter and breadth as constants
def P : ℕ := 950
def B : ℕ := 100

-- State the theorem
theorem find_length (L : ℕ) (H : 2 * (L + B) = P) : L = 375 :=
by sorry

end find_length_l103_103418


namespace tree_planting_activity_l103_103862

theorem tree_planting_activity (x y : ℕ) 
  (h1 : y = 2 * x + 15)
  (h2 : x = y / 3 + 6) : 
  y = 81 ∧ x = 33 := 
by sorry

end tree_planting_activity_l103_103862


namespace unique_exponential_function_l103_103070

theorem unique_exponential_function (g : ℝ → ℝ) :
  (∀ x1 x2 : ℝ, g (x1 + x2) = g x1 * g x2) →
  g 1 = 3 →
  (∀ x1 x2 : ℝ, x1 < x2 → g x1 < g x2) →
  ∀ x : ℝ, g x = 3^x :=
by
  sorry

end unique_exponential_function_l103_103070


namespace a_pow_11_b_pow_11_l103_103838

-- Define the conditions a + b = 1, a^2 + b^2 = 3, a^3 + b^3 = 4, a^4 + b^4 = 7, and a^5 + b^5 = 11
def a : ℝ := sorry
def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Define the recursion pattern for n ≥ 3
axiom h6 (n : ℕ) (hn : n ≥ 3) : a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)

-- Prove that a^11 + b^11 = 199
theorem a_pow_11_b_pow_11 : a^11 + b^11 = 199 :=
by sorry

end a_pow_11_b_pow_11_l103_103838


namespace lowest_test_score_dropped_l103_103146

theorem lowest_test_score_dropped (A B C D : ℝ) 
  (h1: A + B + C + D = 280)
  (h2: A + B + C = 225) : D = 55 := 
by 
  sorry

end lowest_test_score_dropped_l103_103146


namespace minimum_value_l103_103717

open Classical

variable {a b c : ℝ}

theorem minimum_value (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a + b + c = 4) :
  36 ≤ (9 / a) + (16 / b) + (25 / c) :=
sorry

end minimum_value_l103_103717


namespace remainder_add_l103_103296

theorem remainder_add (a b : ℤ) (n m : ℤ) 
  (ha : a = 60 * n + 41) 
  (hb : b = 45 * m + 14) : 
  (a + b) % 15 = 10 := by 
  sorry

end remainder_add_l103_103296


namespace range_of_a_l103_103753

open Set

theorem range_of_a (a : ℝ) (h1 : (∃ x, a^x > 1 ∧ x < 0) ∨ (∀ x, ax^2 - x + a ≥ 0))
  (h2 : ¬((∃ x, a^x > 1 ∧ x < 0) ∧ (∀ x, ax^2 - x + a ≥ 0))) :
  a ∈ (Ioo 0 (1/2)) ∪ (Ici 1) :=
by {
  sorry
}

end range_of_a_l103_103753


namespace find_A_B_l103_103333

theorem find_A_B (A B : ℝ) (h : ∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 → 
  (A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3 * x - 10))) :
  A = 3 ∧ B = 2 :=
sorry

end find_A_B_l103_103333


namespace find_a9_l103_103117

variable (a : ℕ → ℤ)

-- Condition 1: The sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

-- Condition 2: Given a_4 = 5
def a4_value (a : ℕ → ℤ) : Prop :=
  a 4 = 5

-- Condition 3: Given a_5 = 4
def a5_value (a : ℕ → ℤ) : Prop :=
  a 5 = 4

-- Problem: Prove a_9 = 0
theorem find_a9 (h1 : arithmetic_sequence a) (h2 : a4_value a) (h3 : a5_value a) : a 9 = 0 := 
sorry

end find_a9_l103_103117


namespace solution_set_condition_l103_103281

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 1| > a) ↔ a < 1 :=
by
  sorry

end solution_set_condition_l103_103281


namespace number_as_A_times_10_pow_N_integer_l103_103867

theorem number_as_A_times_10_pow_N_integer (A : ℝ) (N : ℝ) (hA1 : 1 ≤ A) (hA2 : A < 10) (hN : A * 10^N > 10) : ∃ (n : ℤ), N = n := 
sorry

end number_as_A_times_10_pow_N_integer_l103_103867


namespace factorize_polynomial_l103_103137

theorem factorize_polynomial (a b : ℝ) : a^2 - 9 * b^2 = (a + 3 * b) * (a - 3 * b) := by
  sorry

end factorize_polynomial_l103_103137


namespace time_to_cover_escalator_l103_103395

def escalator_speed := 11 -- ft/sec
def escalator_length := 126 -- feet
def person_speed := 3 -- ft/sec

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 9 := by
  sorry

end time_to_cover_escalator_l103_103395


namespace simplify_expression_l103_103925

-- Define the given expression
def expr : ℚ := (5^6 + 5^3) / (5^5 - 5^2)

-- State the proof problem
theorem simplify_expression : expr = 315 / 62 := 
by sorry

end simplify_expression_l103_103925


namespace k_league_teams_l103_103532

theorem k_league_teams (n : ℕ) (h : n*(n-1)/2 = 91) : n = 14 := sorry

end k_league_teams_l103_103532


namespace projection_identity_l103_103227

variables (P : ℝ × ℝ × ℝ) (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ)

-- Define point P as (-1, 3, -4)
def point_P := (-1, 3, -4) = P

-- Define projections on the coordinate planes
def projection_yoz := (x1, y1, z1) = (0, 3, -4)
def projection_zox := (x2, y2, z2) = (-1, 0, -4)
def projection_xoy := (x3, y3, z3) = (-1, 3, 0)

-- Prove that x1^2 + y2^2 + z3^2 = 0 under the given conditions
theorem projection_identity :
  point_P P ∧ projection_yoz x1 y1 z1 ∧ projection_zox x2 y2 z2 ∧ projection_xoy x3 y3 z3 →
  (x1^2 + y2^2 + z3^2 = 0) :=
by
  sorry

end projection_identity_l103_103227


namespace geometric_sequence_sum_l103_103626

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q
def cond1 := a 0 + a 1 = 3
def cond2 := a 2 + a 3 = 12
def cond3 := is_geometric_sequence a

theorem geometric_sequence_sum :
  cond1 a →
  cond2 a →
  cond3 a q →
  a 4 + a 5 = 48 :=
by
  intro h1 h2 h3
  sorry

end geometric_sequence_sum_l103_103626


namespace required_line_equation_l103_103994

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Line structure with general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- A point P on a line
def on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Perpendicular condition between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The known line
def known_line : Line := {a := 1, b := -2, c := 3}

-- The given point
def P : Point := {x := -1, y := 3}

noncomputable def required_line : Line := {a := 2, b := 1, c := -1}

-- The theorem to be proved
theorem required_line_equation (l : Line) (P : Point) :
  (on_line P l) ∧ (perpendicular l known_line) ↔ l = required_line :=
  by
    sorry

end required_line_equation_l103_103994


namespace car_rent_per_day_leq_30_l103_103032

variable (D : ℝ) -- daily rental rate
variable (cost_per_mile : ℝ := 0.23) -- cost per mile
variable (daily_budget : ℝ := 76) -- daily budget
variable (distance : ℝ := 200) -- distance driven

theorem car_rent_per_day_leq_30 :
  D + cost_per_mile * distance ≤ daily_budget → D ≤ 30 :=
sorry

end car_rent_per_day_leq_30_l103_103032


namespace maximum_value_of_piecewise_function_l103_103604

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end maximum_value_of_piecewise_function_l103_103604


namespace inequality_l103_103423

-- Define the real variables p, q, r and the condition that their product is 1
variables {p q r : ℝ} (h : p * q * r = 1)

-- State the theorem
theorem inequality (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := 
sorry

end inequality_l103_103423


namespace mike_needs_more_money_l103_103208

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end mike_needs_more_money_l103_103208


namespace elevenRowTriangleTotalPieces_l103_103314

-- Definitions and problem statement
def numRodsInRow (n : ℕ) : ℕ := 3 * n

def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

def totalRods (rows : ℕ) : ℕ := 3 * (sumFirstN rows)

def totalConnectors (rows : ℕ) : ℕ := sumFirstN (rows + 1)

def totalPieces (rows : ℕ) : ℕ := totalRods rows + totalConnectors rows

-- Lean proof problem
theorem elevenRowTriangleTotalPieces : totalPieces 11 = 276 := 
by
  sorry

end elevenRowTriangleTotalPieces_l103_103314


namespace calculation_eq_minus_one_l103_103844

noncomputable def calculation : ℝ :=
  (-1)^(53 : ℤ) + 3^((2^3 + 5^2 - 7^2) : ℤ)

theorem calculation_eq_minus_one : calculation = -1 := 
by 
  sorry

end calculation_eq_minus_one_l103_103844


namespace probability_of_specific_event_l103_103600

noncomputable def adam_probability := 1 / 5
noncomputable def beth_probability := 2 / 9
noncomputable def jack_probability := 1 / 6
noncomputable def jill_probability := 1 / 7
noncomputable def sandy_probability := 1 / 8

theorem probability_of_specific_event :
  (1 - adam_probability) * beth_probability * (1 - jack_probability) * jill_probability * sandy_probability = 1 / 378 := by
  sorry

end probability_of_specific_event_l103_103600


namespace lcm_of_numbers_l103_103665

theorem lcm_of_numbers (x : Nat) (h_ratio : x ≠ 0) (h_hcf : Nat.gcd (5 * x) (Nat.gcd (7 * x) (9 * x)) = 11) :
    Nat.lcm (5 * x) (Nat.lcm (7 * x) (9 * x)) = 99 :=
by
  sorry

end lcm_of_numbers_l103_103665


namespace A_less_B_C_A_relationship_l103_103054

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

end A_less_B_C_A_relationship_l103_103054


namespace no_solution_exists_l103_103759

theorem no_solution_exists :
  ¬ ∃ m n : ℕ, 
    m + n = 2009 ∧ 
    (m * (m - 1) + n * (n - 1) = 2009 * 2008 / 2) := by
  sorry

end no_solution_exists_l103_103759


namespace find_X_sum_coordinates_l103_103655

/- Define points and their coordinates -/
variables (X Y Z : ℝ × ℝ)
variable  (XY XZ ZY : ℝ)
variable  (k : ℝ)
variable  (hxz : XZ = (3/4) * XY)
variable  (hzy : ZY = (1/4) * XY)
variable  (hy : Y = (2, 9))
variable  (hz : Z = (1, 5))

/-- Lean 4 statement for the proof problem -/
theorem find_X_sum_coordinates :
  (Y.1 = 2) ∧ (Y.2 = 9) ∧ (Z.1 = 1) ∧ (Z.2 = 5) ∧
  XZ = (3/4) * XY ∧ ZY = (1/4) * XY →
  (X.1 + X.2) = -9 := 
by
  sorry

end find_X_sum_coordinates_l103_103655


namespace coat_price_reduction_l103_103812

theorem coat_price_reduction 
  (original_price : ℝ) 
  (reduction_percentage : ℝ)
  (h_original_price : original_price = 500)
  (h_reduction_percentage : reduction_percentage = 60) :
  original_price * (reduction_percentage / 100) = 300 :=
by 
  sorry

end coat_price_reduction_l103_103812


namespace train_probability_correct_l103_103782

noncomputable def train_prob (a_train b_train a_john b_john wait : ℝ) : ℝ :=
  let total_time_frame := (b_train - a_train) * (b_john - a_john)
  let triangle_area := (1 / 2) * wait * wait
  let rectangle_area := wait * wait
  let total_overlap_area := triangle_area + rectangle_area
  total_overlap_area / total_time_frame

theorem train_probability_correct :
  train_prob 120 240 150 210 30 = 3 / 16 :=
by
  sorry

end train_probability_correct_l103_103782


namespace sum_last_two_digits_7_13_23_l103_103549

theorem sum_last_two_digits_7_13_23 :
  (7 ^ 23 + 13 ^ 23) % 100 = 40 :=
by 
-- Proof goes here
sorry

end sum_last_two_digits_7_13_23_l103_103549


namespace cos_2beta_value_l103_103598

theorem cos_2beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5) 
  (h2 : Real.cos (α + β) = -3/5) 
  (h3 : α - β ∈ Set.Ioo (π/2) π) 
  (h4 : α + β ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * β) = 24/25 := 
sorry

end cos_2beta_value_l103_103598


namespace alice_unanswered_questions_l103_103360

theorem alice_unanswered_questions :
  ∃ (c w u : ℕ), (5 * c - 2 * w = 54) ∧ (2 * c + u = 36) ∧ (c + w + u = 30) ∧ (u = 8) :=
by
  -- proof omitted
  sorry

end alice_unanswered_questions_l103_103360


namespace parabola_focus_line_slope_intersect_l103_103580

theorem parabola_focus (p : ℝ) (hp : 0 < p) 
  (focus : (1/2 : ℝ) = p/2) : p = 1 :=
by sorry

theorem line_slope_intersect (t : ℝ)
  (intersects_parabola : ∃ A B : ℝ × ℝ, A ≠ (0, 0) ∧ B ≠ (0, 0) ∧
    A ≠ B ∧ A.2 = 2 * A.1 + t ∧ B.2 = 2 * B.1 + t ∧ 
    A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = 0) : 
  t = -4 :=
by sorry

end parabola_focus_line_slope_intersect_l103_103580


namespace tom_reads_700_pages_in_7_days_l103_103868

theorem tom_reads_700_pages_in_7_days
  (total_hours : ℕ)
  (total_days : ℕ)
  (pages_per_hour : ℕ)
  (reads_same_amount_every_day : Prop)
  (h1 : total_hours = 10)
  (h2 : total_days = 5)
  (h3 : pages_per_hour = 50)
  (h4 : reads_same_amount_every_day) :
  (total_hours / total_days) * (pages_per_hour * 7) = 700 :=
by
  -- Begin and skip proof with sorry
  sorry

end tom_reads_700_pages_in_7_days_l103_103868


namespace arthur_walked_distance_in_miles_l103_103452

def blocks_west : ℕ := 8
def blocks_south : ℕ := 10
def block_length_in_miles : ℚ := 1 / 4

theorem arthur_walked_distance_in_miles : 
  (blocks_west + blocks_south) * block_length_in_miles = 4.5 := by
sorry

end arthur_walked_distance_in_miles_l103_103452


namespace hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l103_103232

theorem hyperbola_shares_focus_with_eccentricity 
  (a1 b1 : ℝ) (h1 : a1 = 3 ∧ b1 = 2)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 5) / 2)
  (c : ℝ) (h_focus : c = Real.sqrt (a1^2 - b1^2)) :
  (∃ a b : ℝ, a^2 - b^2 = c^2 ∧ c/a = e ∧ a = 2 ∧ b = 1) :=
sorry

theorem length_of_chord_AB 
  (a b : ℝ) (h_ellipse : a^2 = 4 ∧ b^2 = 1)
  (c : ℝ) (h_focus : c = Real.sqrt (a^2 - b^2))
  (f : ℝ) (h_f : f = Real.sqrt 3)
  (line_eq : ℝ -> ℝ) (h_line_eq : ∀ x, line_eq x = x - f) :
  (∃ x1 x2 : ℝ, 
    x1 + x2 = (8 * Real.sqrt 3) / 5 ∧
    x1 * x2 = 8 / 5 ∧
    Real.sqrt ((x1 - x2)^2 + (line_eq x1 - line_eq x2)^2) = 8 / 5) :=
sorry

end hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l103_103232


namespace actual_average_height_l103_103969

theorem actual_average_height
  (incorrect_avg_height : ℝ)
  (n : ℕ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h1 : incorrect_avg_height = 184)
  (h2 : n = 35)
  (h3 : incorrect_height = 166)
  (h4 : actual_height = 106) :
  let incorrect_total_height := incorrect_avg_height * n
  let difference := incorrect_height - actual_height
  let correct_total_height := incorrect_total_height - difference
  let correct_avg_height := correct_total_height / n
  correct_avg_height = 182.29 :=
by {
  sorry
}

end actual_average_height_l103_103969


namespace marching_band_formations_l103_103519

/-- A marching band of 240 musicians can be arranged in p different rectangular formations 
with s rows and t musicians per row where 8 ≤ t ≤ 30. 
This theorem asserts that there are 8 such different rectangular formations. -/
theorem marching_band_formations (s t : ℕ) (h : s * t = 240) (h_t_bounds : 8 ≤ t ∧ t ≤ 30) : 
  ∃ p : ℕ, p = 8 := 
sorry

end marching_band_formations_l103_103519


namespace sequence_formula_l103_103652

theorem sequence_formula (a : ℕ → ℤ) (h0 : a 0 = 1) (h1 : a 1 = 5)
    (h_rec : ∀ n, n ≥ 2 → a n = (2 * (a (n - 1))^2 - 3 * (a (n - 1)) - 9) / (2 * a (n - 2))) :
  ∀ n, a n = 2^(n + 2) - 3 :=
by
  intros
  sorry

end sequence_formula_l103_103652


namespace lcm_45_75_l103_103344

theorem lcm_45_75 : Nat.lcm 45 75 = 225 :=
by
  sorry

end lcm_45_75_l103_103344


namespace expected_return_correct_l103_103824

-- Define the probabilities
def p1 := 1/4
def p2 := 1/4
def p3 := 1/6
def p4 := 1/3

-- Define the payouts
def payout (n : ℕ) (previous_odd : Bool) : ℝ :=
  match n with
  | 1 => 2
  | 2 => if previous_odd then -3 else 0
  | 3 => 0
  | 4 => 5
  | _ => 0

-- Define the expected values of one throw
def E1 : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

def E2_odd : ℝ :=
  p1 * payout 1 true + p2 * payout 2 true + p3 * payout 3 true + p4 * payout 4 true

def E2_even : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

-- Define the probability of throwing an odd number first
def p_odd : ℝ := p1 + p3

-- Define the probability of not throwing an odd number first
def p_even : ℝ := 1 - p_odd

-- Define the total expected return
def total_expected_return : ℝ :=
  E1 + (p_odd * E2_odd + p_even * E2_even)


theorem expected_return_correct :
  total_expected_return = 4.18 :=
  by
    -- The proof is omitted
    sorry

end expected_return_correct_l103_103824


namespace solve_equation1_solve_equation2_l103_103297

theorem solve_equation1 (x : ℝ) : 4 - x = 3 * (2 - x) ↔ x = 1 :=
by sorry

theorem solve_equation2 (x : ℝ) : (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1 ↔ x = -3 / 2 :=
by sorry

end solve_equation1_solve_equation2_l103_103297


namespace value_of_m_l103_103156

theorem value_of_m (m : ℝ) : (m + 1, 3) ∈ {p : ℝ × ℝ | p.1 + p.2 + 1 = 0} → m = -5 :=
by
  intro h
  sorry

end value_of_m_l103_103156


namespace range_of_a_l103_103386

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h₁ : p a) (h₂ : q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l103_103386


namespace grocery_store_price_l103_103068

-- Definitions based on the conditions
def bulk_price_per_case : ℝ := 12.00
def bulk_cans_per_case : ℝ := 48.0
def grocery_cans_per_pack : ℝ := 12.0
def additional_cost_per_can : ℝ := 0.25

-- The proof statement
theorem grocery_store_price : 
  (bulk_price_per_case / bulk_cans_per_case + additional_cost_per_can) * grocery_cans_per_pack = 6.00 :=
by
  sorry

end grocery_store_price_l103_103068


namespace interest_difference_l103_103240

noncomputable def annual_amount (P r t : ℝ) : ℝ :=
P * (1 + r)^t

noncomputable def monthly_amount (P r n t : ℝ) : ℝ :=
P * (1 + r / n)^(n * t)

theorem interest_difference
  (P : ℝ)
  (r : ℝ)
  (n : ℕ)
  (t : ℝ)
  (annual_compounded : annual_amount P r t = 8000 * (1 + 0.10)^3)
  (monthly_compounded : monthly_amount P r 12 3 = 8000 * (1 + 0.10 / 12) ^ (12 * 3)) :
  (monthly_amount P r 12 t - annual_amount P r t) = 142.80 := 
sorry

end interest_difference_l103_103240


namespace unique_two_digit_number_l103_103871

theorem unique_two_digit_number (x y : ℕ) (h1 : 10 ≤ 10 * x + y ∧ 10 * x + y < 100) (h2 : 3 * y = 2 * x) (h3 : y + 3 = x) : 10 * x + y = 63 :=
by
  sorry

end unique_two_digit_number_l103_103871


namespace solution_problem_l103_103295

noncomputable def proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : Prop :=
  (-1 < (x - y)) ∧ ((x - y) < 1) ∧ (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 1) → (min ((1/x) + (x/y)) = 3))

theorem solution_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  proof_problem x y hx hy h := 
sorry

end solution_problem_l103_103295


namespace fraction_product_eq_l103_103139

theorem fraction_product_eq : (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end fraction_product_eq_l103_103139


namespace initial_percentage_reduction_l103_103648

theorem initial_percentage_reduction
  (x: ℕ)
  (h1: ∀ P: ℝ, P * (1 - x / 100) * 0.85 * 1.5686274509803921 = P) :
  x = 25 :=
by
  sorry

end initial_percentage_reduction_l103_103648


namespace hexagons_after_cuts_l103_103405

theorem hexagons_after_cuts (rectangles_initial : ℕ) (cuts : ℕ) (sheets_total : ℕ)
  (initial_sides : ℕ) (additional_sides : ℕ) 
  (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (final_sides : ℕ) (number_of_hexagons : ℕ) :
  rectangles_initial = 15 →
  cuts = 60 →
  sheets_total = rectangles_initial + cuts →
  initial_sides = rectangles_initial * 4 →
  additional_sides = cuts * 4 →
  final_sides = initial_sides + additional_sides →
  triangle_sides = 3 →
  hexagon_sides = 6 →
  (sheets_total * 4 = final_sides) →
  number_of_hexagons = (final_sides - 225) / 3 →
  number_of_hexagons = 25 :=
by
  intros
  sorry

end hexagons_after_cuts_l103_103405


namespace final_weights_are_correct_l103_103633

-- Definitions of initial weights and reduction percentages per week
def initial_weight_A : ℝ := 300
def initial_weight_B : ℝ := 450
def initial_weight_C : ℝ := 600
def initial_weight_D : ℝ := 750

def reduction_A_week1 : ℝ := 0.20 * initial_weight_A
def reduction_B_week1 : ℝ := 0.15 * initial_weight_B
def reduction_C_week1 : ℝ := 0.30 * initial_weight_C
def reduction_D_week1 : ℝ := 0.25 * initial_weight_D

def weight_A_after_week1 : ℝ := initial_weight_A - reduction_A_week1
def weight_B_after_week1 : ℝ := initial_weight_B - reduction_B_week1
def weight_C_after_week1 : ℝ := initial_weight_C - reduction_C_week1
def weight_D_after_week1 : ℝ := initial_weight_D - reduction_D_week1

def reduction_A_week2 : ℝ := 0.25 * weight_A_after_week1
def reduction_B_week2 : ℝ := 0.30 * weight_B_after_week1
def reduction_C_week2 : ℝ := 0.10 * weight_C_after_week1
def reduction_D_week2 : ℝ := 0.20 * weight_D_after_week1

def weight_A_after_week2 : ℝ := weight_A_after_week1 - reduction_A_week2
def weight_B_after_week2 : ℝ := weight_B_after_week1 - reduction_B_week2
def weight_C_after_week2 : ℝ := weight_C_after_week1 - reduction_C_week2
def weight_D_after_week2 : ℝ := weight_D_after_week1 - reduction_D_week2

def reduction_A_week3 : ℝ := 0.15 * weight_A_after_week2
def reduction_B_week3 : ℝ := 0.10 * weight_B_after_week2
def reduction_C_week3 : ℝ := 0.20 * weight_C_after_week2
def reduction_D_week3 : ℝ := 0.30 * weight_D_after_week2

def weight_A_after_week3 : ℝ := weight_A_after_week2 - reduction_A_week3
def weight_B_after_week3 : ℝ := weight_B_after_week2 - reduction_B_week3
def weight_C_after_week3 : ℝ := weight_C_after_week2 - reduction_C_week3
def weight_D_after_week3 : ℝ := weight_D_after_week2 - reduction_D_week3

def reduction_A_week4 : ℝ := 0.10 * weight_A_after_week3
def reduction_B_week4 : ℝ := 0.20 * weight_B_after_week3
def reduction_C_week4 : ℝ := 0.25 * weight_C_after_week3
def reduction_D_week4 : ℝ := 0.15 * weight_D_after_week3

def final_weight_A : ℝ := weight_A_after_week3 - reduction_A_week4
def final_weight_B : ℝ := weight_B_after_week3 - reduction_B_week4
def final_weight_C : ℝ := weight_C_after_week3 - reduction_C_week4
def final_weight_D : ℝ := weight_D_after_week3 - reduction_D_week4

theorem final_weights_are_correct :
  final_weight_A = 137.7 ∧ 
  final_weight_B = 192.78 ∧ 
  final_weight_C = 226.8 ∧ 
  final_weight_D = 267.75 :=
by
  unfold final_weight_A final_weight_B final_weight_C final_weight_D
  sorry

end final_weights_are_correct_l103_103633


namespace donna_pays_total_l103_103089

def original_price_vase : ℝ := 250
def discount_vase : ℝ := original_price_vase * 0.25

def original_price_teacups : ℝ := 350
def discount_teacups : ℝ := original_price_teacups * 0.30

def original_price_plate : ℝ := 450
def discount_plate : ℝ := 0

def original_price_ornament : ℝ := 150
def discount_ornament : ℝ := original_price_ornament * 0.20

def membership_discount_vase : ℝ := (original_price_vase - discount_vase) * 0.05
def membership_discount_plate : ℝ := original_price_plate * 0.05

def tax_vase : ℝ := ((original_price_vase - discount_vase - membership_discount_vase) * 0.12)
def tax_teacups : ℝ := ((original_price_teacups - discount_teacups) * 0.08)
def tax_plate : ℝ := ((original_price_plate - membership_discount_plate) * 0.10)
def tax_ornament : ℝ := ((original_price_ornament - discount_ornament) * 0.06)

def final_price_vase : ℝ := (original_price_vase - discount_vase - membership_discount_vase) + tax_vase
def final_price_teacups : ℝ := (original_price_teacups - discount_teacups) + tax_teacups
def final_price_plate : ℝ := (original_price_plate - membership_discount_plate) + tax_plate
def final_price_ornament : ℝ := (original_price_ornament - discount_ornament) + tax_ornament

def total_price : ℝ := final_price_vase + final_price_teacups + final_price_plate + final_price_ornament

theorem donna_pays_total :
  total_price = 1061.55 :=
by
  sorry

end donna_pays_total_l103_103089


namespace range_of_b_l103_103106

theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, x^2 - 2 * b * x + b^2 + b - 5 = 0) ∧
  (∀ x < 3.5, ∃ δ > 0, ∀ ε, x < ε → ε^2 - 2 * b * ε + b^2 + b - 5 < x^2 - 2 * b * x + b^2 + b - 5) →
  (3.5 ≤ b ∧ b ≤ 5) :=
by
  sorry

end range_of_b_l103_103106


namespace inequality_proof_l103_103836

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256 * x * y * z :=
by
  -- Proof goes here
  sorry

end inequality_proof_l103_103836


namespace domain_is_all_real_l103_103891

-- Definitions and conditions
def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 18

def domain_of_f (x : ℝ) : Prop := ∃ (y : ℝ), y = 1 / (⌊quadratic_expression x⌋)

-- Theorem statement
theorem domain_is_all_real : ∀ x : ℝ, domain_of_f x :=
by
  sorry

end domain_is_all_real_l103_103891


namespace angle_x_is_36_l103_103039

theorem angle_x_is_36
    (x : ℝ)
    (h1 : 7 * x + 3 * x = 360)
    (h2 : 8 * x ≤ 360) :
    x = 36 := 
by {
  sorry
}

end angle_x_is_36_l103_103039


namespace molecular_weight_of_compound_l103_103802

-- Given atomic weights in g/mol
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O  : ℝ := 15.999
def atomic_weight_H  : ℝ := 1.008

-- Given number of atoms in the compound
def num_atoms_Ca : ℕ := 1
def num_atoms_O  : ℕ := 2
def num_atoms_H  : ℕ := 2

-- Definition of the molecular weight
def molecular_weight : ℝ :=
  (num_atoms_Ca * atomic_weight_Ca) +
  (num_atoms_O * atomic_weight_O) +
  (num_atoms_H * atomic_weight_H)

-- The theorem to prove
theorem molecular_weight_of_compound : molecular_weight = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l103_103802


namespace central_angle_is_two_l103_103749

noncomputable def central_angle_of_sector (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : ℝ :=
  l / r

theorem central_angle_is_two (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : central_angle_of_sector r l h1 h2 = 2 :=
by
  sorry

end central_angle_is_two_l103_103749


namespace michael_saves_more_l103_103889

-- Definitions for the conditions
def price_per_pair : ℝ := 50
def discount_a (price : ℝ) : ℝ := price + 0.6 * price
def discount_b (price : ℝ) : ℝ := 2 * price - 15

-- Statement to prove
theorem michael_saves_more (price : ℝ) (h : price = price_per_pair) : discount_b price - discount_a price = 5 :=
by
  sorry

end michael_saves_more_l103_103889


namespace hexagon_colorings_correct_l103_103235

noncomputable def hexagon_colorings : Nat :=
  let colors := ["blue", "orange", "purple"]
  2 -- As determined by the solution.

theorem hexagon_colorings_correct :
  hexagon_colorings = 2 :=
by
  sorry

end hexagon_colorings_correct_l103_103235


namespace graph_passes_through_point_l103_103853

theorem graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
    ∀ x y : ℝ, (y = a^(x-2) + 2) → (x = 2) → (y = 3) :=
by
    intros x y hxy hx
    rw [hx] at hxy
    simp at hxy
    sorry

end graph_passes_through_point_l103_103853


namespace minimize_wire_length_l103_103910

theorem minimize_wire_length :
  ∃ (x : ℝ), (x > 0) ∧ (2 * (x + 4 / x) = 8) :=
by
  sorry

end minimize_wire_length_l103_103910


namespace calculate_x_times_a_l103_103339

-- Define variables and assumptions
variables (a b x y : ℕ)
variable (hb : b = 4)
variable (hy : y = 2)
variable (h1 : a = 2 * b)
variable (h2 : x = 3 * y)
variable (h3 : a + b = x * y)

-- The statement to be proved
theorem calculate_x_times_a : x * a = 48 :=
by sorry

end calculate_x_times_a_l103_103339


namespace product_of_two_numbers_l103_103448

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 400) : x * y = 88 :=
by
  -- Proof goes here
  sorry

end product_of_two_numbers_l103_103448


namespace central_angle_of_sector_l103_103248

theorem central_angle_of_sector (l S : ℝ) (r : ℝ) (θ : ℝ) 
  (h1 : l = 5) 
  (h2 : S = 5) 
  (h3 : S = (1 / 2) * l * r) 
  (h4 : l = θ * r): θ = 2.5 := by
  sorry

end central_angle_of_sector_l103_103248


namespace expression_simplification_l103_103796

theorem expression_simplification (x y : ℝ) : x^2 + (y - x) * (y + x) = y^2 :=
by
  sorry

end expression_simplification_l103_103796


namespace rounding_problem_l103_103309

def given_number : ℝ := 3967149.487234

theorem rounding_problem : (3967149.487234).round = 3967149 := sorry

end rounding_problem_l103_103309


namespace total_time_correct_l103_103072

variable (b n : ℕ)

def total_travel_time (b n : ℕ) : ℚ := (3*b + 4*n + 2*b) / 150

theorem total_time_correct :
  total_travel_time b n = (5 * b + 4 * n) / 150 :=
by sorry

end total_time_correct_l103_103072


namespace radio_loss_percentage_l103_103118

theorem radio_loss_percentage :
  ∀ (cost_price selling_price : ℝ), 
    cost_price = 1500 → 
    selling_price = 1290 → 
    ((cost_price - selling_price) / cost_price) * 100 = 14 :=
by
  intros cost_price selling_price h_cp h_sp
  sorry

end radio_loss_percentage_l103_103118


namespace domain_of_function_l103_103966

def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := x^2 - 9

theorem domain_of_function :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_function_l103_103966


namespace find_cat_video_length_l103_103080

variables (C : ℕ)

def cat_video_length (C : ℕ) : Prop :=
  C + 2 * C + 6 * C = 36

theorem find_cat_video_length : cat_video_length 4 :=
by
  sorry

end find_cat_video_length_l103_103080


namespace front_view_correct_l103_103449

section stack_problem

def column1 : List ℕ := [3, 2]
def column2 : List ℕ := [1, 4, 2]
def column3 : List ℕ := [5]
def column4 : List ℕ := [2, 1]

def tallest (l : List ℕ) : ℕ := l.foldr max 0

theorem front_view_correct :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 4, 5, 2] :=
sorry

end stack_problem

end front_view_correct_l103_103449


namespace prove_ab_eq_neg_26_l103_103325

theorem prove_ab_eq_neg_26
  (a b : ℚ)
  (H : ∀ k : ℚ, ∃ x : ℚ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6) :
  a * b = -26 := sorry

end prove_ab_eq_neg_26_l103_103325


namespace number_total_11_l103_103500

theorem number_total_11 (N : ℕ) (S : ℝ)
  (h1 : S = 10.7 * N)
  (h2 : (6 : ℝ) * 10.5 = 63)
  (h3 : (6 : ℝ) * 11.4 = 68.4)
  (h4 : 13.7 = 13.700000000000017)
  (h5 : S = 63 + 68.4 - 13.7) : 
  N = 11 := 
sorry

end number_total_11_l103_103500


namespace max_AMC_expression_l103_103622

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 24) :
  A * M * C + A * M + M * C + C * A ≤ 704 :=
sorry

end max_AMC_expression_l103_103622


namespace units_digit_of_52_cubed_plus_29_cubed_l103_103978

-- Define the units digit of a number n
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions as definitions in Lean
def units_digit_of_2_cubed : ℕ := units_digit (2^3)  -- 8
def units_digit_of_9_cubed : ℕ := units_digit (9^3)  -- 9

-- The main theorem to prove
theorem units_digit_of_52_cubed_plus_29_cubed : units_digit (52^3 + 29^3) = 7 :=
by
  sorry

end units_digit_of_52_cubed_plus_29_cubed_l103_103978


namespace find_H2SO4_moles_l103_103801

-- Let KOH, H2SO4, and KHSO4 represent the moles of each substance in the reaction.
variable (KOH H2SO4 KHSO4 : ℕ)

-- Conditions provided in the problem
def KOH_moles : ℕ := 2
def KHSO4_moles (H2SO4 : ℕ) : ℕ := H2SO4

-- Main statement, we need to prove that given the conditions,
-- 2 moles of KOH and 2 moles of KHSO4 imply 2 moles of H2SO4.
theorem find_H2SO4_moles (KOH_sufficient : KOH = KOH_moles) 
  (KHSO4_produced : KHSO4 = KOH) : KHSO4_moles H2SO4 = 2 := 
sorry

end find_H2SO4_moles_l103_103801


namespace one_in_B_neg_one_not_in_B_B_roster_l103_103548

open Set Int

def B : Set ℤ := {x | ∃ n : ℕ, 6 = n * (3 - x)}

theorem one_in_B : 1 ∈ B :=
by sorry

theorem neg_one_not_in_B : (-1 ∉ B) :=
by sorry

theorem B_roster : B = {2, 1, 0, -3} :=
by sorry

end one_in_B_neg_one_not_in_B_B_roster_l103_103548


namespace dilation_image_l103_103409

theorem dilation_image (z : ℂ) (c : ℂ) (k : ℝ) (w : ℂ) (h₁ : c = 0 + 5 * I) 
  (h₂ : k = 3) (h₃ : w = 3 + 2 * I) : z = 9 - 4 * I :=
by
  -- Given conditions
  have hc : c = 0 + 5 * I := h₁
  have hk : k = 3 := h₂
  have hw : w = 3 + 2 * I := h₃

  -- Dilation formula
  let formula := (w - c) * k + c

  -- Prove the result
  -- sorry for now, the proof is not required as per instructions
  sorry

end dilation_image_l103_103409


namespace cubics_sum_l103_103138

noncomputable def roots_cubic (a b c d p q r : ℝ) : Prop :=
  (p + q + r = b) ∧ (p*q + p*r + q*r = c) ∧ (p*q*r = d)

noncomputable def root_values (p q r : ℝ) : Prop :=
  p^3 = 2*p^2 - 3*p + 4 ∧
  q^3 = 2*q^2 - 3*q + 4 ∧
  r^3 = 2*r^2 - 3*r + 4

theorem cubics_sum (p q r : ℝ) (h1 : p + q + r = 2) (h2 : p*q + q*r + p*r = 3)  (h3 : p*q*r = 4)
  (h4 : root_values p q r) : p^3 + q^3 + r^3 = 2 :=
by
  sorry

end cubics_sum_l103_103138


namespace distance_to_focus_parabola_l103_103489

theorem distance_to_focus_parabola (F P : ℝ × ℝ) (hF : F = (0, -1/2))
  (hP : P = (1, 2)) (C : ℝ × ℝ → Prop)
  (hC : ∀ x, C (x, 2 * x^2)) : dist P F = 17 / 8 := by
sorry

end distance_to_focus_parabola_l103_103489


namespace compound_interest_l103_103349

theorem compound_interest (P R T : ℝ) (SI CI : ℝ)
  (hSI : SI = P * R * T / 100)
  (h_given_SI : SI = 50)
  (h_given_R : R = 5)
  (h_given_T : T = 2)
  (h_compound_interest : CI = P * ((1 + R / 100)^T - 1)) :
  CI = 51.25 :=
by
  -- Since we are only required to state the theorem, we add 'sorry' here.
  sorry

end compound_interest_l103_103349


namespace Derek_test_score_l103_103952

def Grant_score (John_score : ℕ) : ℕ := John_score + 10
def John_score (Hunter_score : ℕ) : ℕ := 2 * Hunter_score
def Hunter_score : ℕ := 45
def Sarah_score (Grant_score : ℕ) : ℕ := Grant_score - 5
def Derek_score (John_score Grant_score : ℕ) : ℕ := (John_score + Grant_score) / 2

theorem Derek_test_score :
  Derek_score (John_score Hunter_score) (Grant_score (John_score Hunter_score)) = 95 :=
  by
  -- proof here
  sorry

end Derek_test_score_l103_103952


namespace negation_of_proposition_l103_103374

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a * x + 1 ≥ 0) :=
by sorry

end negation_of_proposition_l103_103374


namespace line_does_not_pass_through_point_l103_103018

theorem line_does_not_pass_through_point 
  (m : ℝ) (h : (2*m + 1)^2 - 4*(m^2 + 4) > 0) : 
  ¬((2*m - 3)*(-2) - 4*m + 7 = 1) :=
by
  sorry

end line_does_not_pass_through_point_l103_103018


namespace number_of_pages_correct_number_of_ones_correct_l103_103979

noncomputable def number_of_pages (total_digits : ℕ) : ℕ :=
  let single_digit_odd_pages := 5
  let double_digit_odd_pages := 45
  let triple_digit_odd_pages := (total_digits - (single_digit_odd_pages + 2 * double_digit_odd_pages)) / 3
  single_digit_odd_pages + double_digit_odd_pages + triple_digit_odd_pages

theorem number_of_pages_correct : number_of_pages 125 = 60 :=
by sorry

noncomputable def number_of_ones (total_digits : ℕ) : ℕ :=
  let ones_in_units_place := 12
  let ones_in_tens_place := 18
  let ones_in_hundreds_place := 10
  ones_in_units_place + ones_in_tens_place + ones_in_hundreds_place

theorem number_of_ones_correct : number_of_ones 125 = 40 :=
by sorry

end number_of_pages_correct_number_of_ones_correct_l103_103979


namespace same_terminal_side_angle_exists_l103_103479

theorem same_terminal_side_angle_exists :
  ∃ k : ℤ, -5 * π / 8 + 2 * k * π = 11 * π / 8 := 
by
  sorry

end same_terminal_side_angle_exists_l103_103479


namespace find_first_blend_price_l103_103238

-- Define the conditions
def first_blend_price (x : ℝ) := x
def second_blend_price : ℝ := 8.00
def total_blend_weight : ℝ := 20
def total_blend_price_per_pound : ℝ := 8.40
def first_blend_weight : ℝ := 8
def second_blend_weight : ℝ := total_blend_weight - first_blend_weight

-- Define the cost calculations
def first_blend_total_cost (x : ℝ) := first_blend_weight * x
def second_blend_total_cost := second_blend_weight * second_blend_price
def total_blend_total_cost (x : ℝ) := first_blend_total_cost x + second_blend_total_cost

-- Prove that the price per pound of the first blend is $9.00
theorem find_first_blend_price : ∃ x : ℝ, total_blend_total_cost x = total_blend_weight * total_blend_price_per_pound ∧ x = 9 :=
by
  sorry

end find_first_blend_price_l103_103238


namespace no_such_quadratics_l103_103672

theorem no_such_quadratics :
  ¬ ∃ (a b c : ℤ), ∃ (x1 x2 x3 x4 : ℤ),
    (a * x1 * x2 = c ∧ a * (x1 + x2) = -b) ∧
    ((a + 1) * x3 * x4 = c + 1 ∧ (a + 1) * (x3 + x4) = -(b + 1)) :=
sorry

end no_such_quadratics_l103_103672


namespace hat_price_reduction_l103_103273

theorem hat_price_reduction (original_price : ℚ) (r1 r2 : ℚ) (price_after_reductions : ℚ) :
  original_price = 12 → r1 = 0.20 → r2 = 0.25 →
  price_after_reductions = original_price * (1 - r1) * (1 - r2) →
  price_after_reductions = 7.20 :=
by
  intros original_price_eq r1_eq r2_eq price_calc_eq
  sorry

end hat_price_reduction_l103_103273


namespace Sydney_initial_rocks_l103_103679

variable (S₀ : ℕ)

def Conner_initial : ℕ := 723
def Sydney_collects_day1 : ℕ := 4
def Conner_collects_day1 : ℕ := 8 * Sydney_collects_day1
def Sydney_collects_day2 : ℕ := 0
def Conner_collects_day2 : ℕ := 123
def Sydney_collects_day3 : ℕ := 2 * Conner_collects_day1
def Conner_collects_day3 : ℕ := 27

def Total_Sydney_collects : ℕ := Sydney_collects_day1 + Sydney_collects_day2 + Sydney_collects_day3
def Total_Conner_collects : ℕ := Conner_collects_day1 + Conner_collects_day2 + Conner_collects_day3

def Total_Sydney_rocks : ℕ := S₀ + Total_Sydney_collects
def Total_Conner_rocks : ℕ := Conner_initial + Total_Conner_collects

theorem Sydney_initial_rocks :
  Total_Conner_rocks = Total_Sydney_rocks → S₀ = 837 :=
by
  sorry

end Sydney_initial_rocks_l103_103679


namespace sum_integers_neg40_to_60_l103_103565

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end sum_integers_neg40_to_60_l103_103565


namespace Agnes_birth_year_l103_103258

theorem Agnes_birth_year (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9)
  (h3 : (11 * x + 2 * y + x * y = 92)) : 1948 = 1900 + (10 * x + y) :=
sorry

end Agnes_birth_year_l103_103258


namespace burrs_count_l103_103104

variable (B T : ℕ)

theorem burrs_count 
  (h1 : T = 6 * B) 
  (h2 : B + T = 84) : 
  B = 12 := 
by
  sorry

end burrs_count_l103_103104


namespace find_cos_beta_l103_103822

noncomputable def cos_beta (α β : ℝ) : ℝ :=
  - (6 * Real.sqrt 2 + 4) / 15

theorem find_cos_beta (α β : ℝ)
  (h0 : α ∈ Set.Ioc 0 (Real.pi / 2))
  (h1 : β ∈ Set.Ioc (Real.pi / 2) Real.pi)
  (h2 : Real.cos α = 1 / 3)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.cos β = cos_beta α β :=
by
  sorry

end find_cos_beta_l103_103822


namespace circle_area_difference_l103_103031

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 30) (h2 : r2 = 7.5) : 
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end circle_area_difference_l103_103031


namespace city_miles_count_l103_103514

-- Defining the variables used in the conditions
def miles_per_gallon_city : ℝ := 30
def miles_per_gallon_highway : ℝ := 40
def highway_miles : ℝ := 200
def cost_per_gallon : ℝ := 3
def total_cost : ℝ := 42

-- Required statement for the proof, statement to prove: count of city miles is 270
theorem city_miles_count : ∃ (C : ℝ), C = 270 ∧
  (total_cost / cost_per_gallon) = ((C / miles_per_gallon_city) + (highway_miles / miles_per_gallon_highway)) :=
by
  sorry

end city_miles_count_l103_103514


namespace maisy_earns_more_l103_103392

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l103_103392


namespace number_of_chocolate_bars_l103_103256

theorem number_of_chocolate_bars (C : ℕ) (h1 : 50 * C = 250) : C = 5 := by
  sorry

end number_of_chocolate_bars_l103_103256


namespace periodic_sequence_exists_l103_103520

noncomputable def bounded_sequence (a : ℕ → ℤ) (M : ℤ) :=
  ∀ n, |a n| ≤ M

noncomputable def satisfies_recurrence (a : ℕ → ℤ) :=
  ∀ n, n ≥ 5 → a n = (a (n - 1) + a (n - 2) + a (n - 3) * a (n - 4)) / (a (n - 1) * a (n - 2) + a (n - 3) + a (n - 4))

theorem periodic_sequence_exists (a : ℕ → ℤ) (M : ℤ) 
  (h_bounded : bounded_sequence a M) (h_rec : satisfies_recurrence a) : 
  ∃ l : ℕ, ∀ n : ℕ, a (l + n) = a (l + n + (l + 1) - l) :=
sorry

end periodic_sequence_exists_l103_103520
