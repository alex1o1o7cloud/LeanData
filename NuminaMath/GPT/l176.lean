import Mathlib

namespace polynomial_has_three_real_roots_l176_176241

theorem polynomial_has_three_real_roots (a b c : ℝ) (h1 : b < 0) (h2 : a * b = 9 * c) :
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ 
    (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ 
    (x3^3 + a * x3^2 + b * x3 + c = 0) := sorry

end polynomial_has_three_real_roots_l176_176241


namespace remainder_3n_plus_2_l176_176039

-- Define the condition
def n_condition (n : ℤ) : Prop := n % 7 = 5

-- Define the theorem to be proved
theorem remainder_3n_plus_2 (n : ℤ) (h : n_condition n) : (3 * n + 2) % 7 = 3 := 
by sorry

end remainder_3n_plus_2_l176_176039


namespace inequality_one_solution_inequality_two_solution_l176_176496

theorem inequality_one_solution (x : ℝ) :
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3 / 2) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  (x + 1) / (x - 2) ≤ 2 ↔ (x < 2 ∨ x ≥ 5) :=
sorry

end inequality_one_solution_inequality_two_solution_l176_176496


namespace decimal_representation_of_7_over_12_eq_0_point_5833_l176_176124

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end decimal_representation_of_7_over_12_eq_0_point_5833_l176_176124


namespace range_of_a_l176_176085

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + 1 + a * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end range_of_a_l176_176085


namespace geometric_sequence_ratio_l176_176944

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ)
  (hq_pos : 0 < q)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * (1/2) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 10 + a 12) / (a 7 + a 9) = 27 :=
sorry

end geometric_sequence_ratio_l176_176944


namespace gas_cost_l176_176306

theorem gas_cost (x : ℝ) (h₁ : 5 * (x / 5 - 9) = 8 * (x / 8)) : x = 120 :=
by
  sorry

end gas_cost_l176_176306


namespace sequence_terms_proof_l176_176415

theorem sequence_terms_proof (P Q R T U V W : ℤ) (S : ℤ) 
  (h1 : S = 10) 
  (h2 : P + Q + R + S = 40) 
  (h3 : Q + R + S + T = 40) 
  (h4 : R + S + T + U = 40) 
  (h5 : S + T + U + V = 40) 
  (h6 : T + U + V + W = 40) : 
  P + W = 40 := 
by 
  have h7 : P + Q + R + 10 = 40 := by rwa [h1] at h2
  have h8 : Q + R + 10 + T = 40 := by rwa [h1] at h3
  have h9 : R + 10 + T + U = 40 := by rwa [h1] at h4
  have h10 : 10 + T + U + V = 40 := by rwa [h1] at h5
  have h11 : T + U + V + W = 40 := h6
  sorry

end sequence_terms_proof_l176_176415


namespace inequality_not_hold_l176_176168

theorem inequality_not_hold (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) < 1 / a) :=
by
  sorry

end inequality_not_hold_l176_176168


namespace sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l176_176320

theorem sum_of_reciprocals_roots_transformed_eq_neg11_div_4 :
  (∃ a b c : ℝ, (a^3 - a - 2 = 0) ∧ (b^3 - b - 2 = 0) ∧ (c^3 - c - 2 = 0)) → 
  ( ∃ a b c : ℝ, a^3 - a - 2 = 0 ∧ b^3 - b - 2 = 0 ∧ c^3 - c - 2 = 0 ∧ 
  (1 / (a - 2) + 1 / (b - 2) + 1 / (c - 2) = - 11 / 4)) :=
by
  sorry

end sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l176_176320


namespace largest_four_digit_negative_congruent_to_1_pmod_17_l176_176726

theorem largest_four_digit_negative_congruent_to_1_pmod_17 :
  ∃ n : ℤ, 17 * n + 1 < -1000 ∧ 17 * n + 1 ≥ -9999 ∧ 17 * n + 1 ≡ 1 [ZMOD 17] := 
sorry

end largest_four_digit_negative_congruent_to_1_pmod_17_l176_176726


namespace cannot_form_triangle_l176_176348

theorem cannot_form_triangle {a b c : ℝ} (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) : 
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end cannot_form_triangle_l176_176348


namespace cylinder_inscribed_in_sphere_l176_176953

noncomputable def sphere_volume (r : ℝ) : ℝ := 
  (4 / 3) * Real.pi * r^3

theorem cylinder_inscribed_in_sphere 
  (r_cylinder : ℝ)
  (h₁ : r_cylinder > 0)
  (height_cylinder : ℝ)
  (radius_sphere : ℝ)
  (h₂ : radius_sphere = r_cylinder + 2)
  (h₃ : height_cylinder = r_cylinder + 1)
  (h₄ : 2 * radius_sphere = Real.sqrt ((2 * r_cylinder)^2 + (height_cylinder)^2))
  : sphere_volume 17 = 6550 * 2 / 3 * Real.pi :=
by
  -- solution steps and proof go here
  sorry

end cylinder_inscribed_in_sphere_l176_176953


namespace find_p_r_l176_176619

-- Definitions of the polynomials
def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
def g (x : ℝ) (r s : ℝ) : ℝ := x^2 + r * x + s

-- Lean statement of the proof problem:
theorem find_p_r (p q r s : ℝ) (h1 : p ≠ r) (h2 : g (-p / 2) r s = 0) 
  (h3 : f (-r / 2) p q = 0) (h4 : ∀ x : ℝ, f x p q = g x r s) 
  (h5 : f 50 p q = -50) : p + r = -200 := 
sorry

end find_p_r_l176_176619


namespace octagon_has_20_diagonals_l176_176186

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l176_176186


namespace selling_price_correct_l176_176534

theorem selling_price_correct (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) 
  (h_cost : cost_price = 600) 
  (h_loss : loss_percent = 25)
  (h_selling_price : selling_price = cost_price - (loss_percent / 100) * cost_price) : 
  selling_price = 450 := 
by 
  rw [h_cost, h_loss] at h_selling_price
  norm_num at h_selling_price
  exact h_selling_price

#check selling_price_correct

end selling_price_correct_l176_176534


namespace problem_l176_176203

theorem problem (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_l176_176203


namespace isosceles_triangle_vertex_angle_l176_176721

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (h_iso : A = B ∨ A = C ∨ B = C) (h_sum : A + B + C = 180) (h_one_angle : A = 50 ∨ B = 50 ∨ C = 50) :
  A = 80 ∨ B = 80 ∨ C = 80 ∨ A = 50 ∨ B = 50 ∨ C = 50 :=
by
  sorry

end isosceles_triangle_vertex_angle_l176_176721


namespace car_speeds_and_arrival_times_l176_176674

theorem car_speeds_and_arrival_times
  (x y z u : ℝ)
  (h1 : x^2 = (y + z) * u)
  (h2 : (y + z) / 4 = u)
  (h3 : x / u = y / z)
  (h4 : x + y + z + u = 210) :
  x = 60 ∧ y = 80 ∧ z = 40 ∧ u = 30 := 
by
  sorry

end car_speeds_and_arrival_times_l176_176674


namespace angle_slope_condition_l176_176462

theorem angle_slope_condition (α k : Real) (h₀ : k = Real.tan α) (h₁ : 0 ≤ α ∧ α < Real.pi) : 
  (α < Real.pi / 3) → (k < Real.sqrt 3) ∧ ¬((k < Real.sqrt 3) → (α < Real.pi / 3)) := 
sorry

end angle_slope_condition_l176_176462


namespace ratio_Binkie_Frankie_eq_4_l176_176932

-- Definitions based on given conditions
def SpaatzGems : ℕ := 1
def BinkieGems : ℕ := 24

-- Assume the number of gemstones on Frankie's collar
variable (FrankieGems : ℕ)

-- Given condition about the gemstones on Spaatz's collar
axiom SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2

-- The theorem to be proved
theorem ratio_Binkie_Frankie_eq_4 
    (FrankieGems : ℕ) 
    (SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2) 
    (BinkieGems_eq : BinkieGems = 24) 
    (SpaatzGems_eq : SpaatzGems = 1) 
    (f_nonzero : FrankieGems ≠ 0) :
    BinkieGems / FrankieGems = 4 :=
by
  sorry  -- We're only writing the statement, not the proof.

end ratio_Binkie_Frankie_eq_4_l176_176932


namespace range_of_t_l176_176285

theorem range_of_t 
  (k t : ℝ)
  (tangent_condition : (t + 1)^2 = 1 + k^2)
  (intersect_condition : ∃ x y, y = k * x + t ∧ y = x^2 / 4) : 
  t > 0 ∨ t < -3 :=
sorry

end range_of_t_l176_176285


namespace airplane_distance_difference_l176_176089

theorem airplane_distance_difference (a : ℕ) : 
  let against_wind_distance := (a - 20) * 3
  let with_wind_distance := (a + 20) * 4
  with_wind_distance - against_wind_distance = a + 140 :=
by
  sorry

end airplane_distance_difference_l176_176089


namespace fraction_zero_iff_x_is_four_l176_176381

theorem fraction_zero_iff_x_is_four (x : ℝ) (h_ne_zero: x + 4 ≠ 0) :
  (16 - x^2) / (x + 4) = 0 ↔ x = 4 :=
sorry

end fraction_zero_iff_x_is_four_l176_176381


namespace imaginary_part_of_complex_l176_176315

theorem imaginary_part_of_complex (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_complex_l176_176315


namespace smallest_next_divisor_of_m_l176_176205

theorem smallest_next_divisor_of_m (m : ℕ) (h1 : m % 2 = 0) (h2 : 10000 ≤ m ∧ m < 100000) (h3 : 523 ∣ m) : 
  ∃ d : ℕ, 523 < d ∧ d ∣ m ∧ ∀ e : ℕ, 523 < e ∧ e ∣ m → d ≤ e :=
by
  sorry

end smallest_next_divisor_of_m_l176_176205


namespace a_2_value_general_terms_T_n_value_l176_176535

-- Definitions based on conditions
def S (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of sequence {a_n}

def a (n : ℕ) : ℕ := (S n + 2) / 2  -- a_n is the arithmetic mean of S_n and 2

def b (n : ℕ) : ℕ := 2 * n - 1  -- Given general term for b_n

-- Prove a_2 = 4
theorem a_2_value : a 2 = 4 := 
by
  sorry

-- Prove the general terms
theorem general_terms (n : ℕ) : a n = 2^n ∧ b n = 2 * n - 1 := 
by
  sorry

-- Definition and sum of the first n terms of c_n
def c (n : ℕ) : ℕ := a n * b n

def T (n : ℕ) : ℕ := (2 * n - 3) * 2^(n + 1) + 6  -- Given sum of the first n terms of {c_n}

-- Prove T_n = (2n - 3)2^(n+1) + 6
theorem T_n_value (n : ℕ) : T n = (2 * n - 3) * 2^(n + 1) + 6 :=
by
  sorry

end a_2_value_general_terms_T_n_value_l176_176535


namespace mean_of_remaining_two_l176_176844

theorem mean_of_remaining_two (a b c d e : ℝ) (h : (a + b + c = 3 * 2010)) : 
  (a + b + c + d + e) / 5 = 2010 → (d + e) / 2 = 2011.5 :=
by
  sorry 

end mean_of_remaining_two_l176_176844


namespace number_of_possible_values_of_k_l176_176952

-- Define the primary conditions and question
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def quadratic_roots_prime (p q k : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 72 ∧ p * q = k

theorem number_of_possible_values_of_k :
  ¬ ∃ k : ℕ, ∃ p q : ℕ, quadratic_roots_prime p q k :=
by
  sorry

end number_of_possible_values_of_k_l176_176952


namespace monotonicity_and_max_of_f_g_range_of_a_l176_176325

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem monotonicity_and_max_of_f : 
  (∀ x, 0 < x → x < 1 → f x > f (x + 1)) ∧ 
  (∀ x, x > 1 → f x < f (x - 1)) ∧ 
  (f 1 = -1) := 
by
  sorry

theorem g_range_of_a (a : ℝ) : 
  (∀ x, x > 0 → f x + g x a ≥ 0) → (a ≤ 1) := 
by
  sorry

end monotonicity_and_max_of_f_g_range_of_a_l176_176325


namespace blue_ball_higher_numbered_bin_l176_176894

noncomputable def probability_higher_numbered_bin :
  ℝ := sorry

theorem blue_ball_higher_numbered_bin :
  probability_higher_numbered_bin = 7 / 16 :=
sorry

end blue_ball_higher_numbered_bin_l176_176894


namespace A_subscribed_fraction_l176_176282

theorem A_subscribed_fraction 
  (total_profit : ℝ) (A_share : ℝ) 
  (B_fraction : ℝ) (C_fraction : ℝ) 
  (A_fraction : ℝ) :
  total_profit = 2430 →
  A_share = 810 →
  B_fraction = 1/4 →
  C_fraction = 1/5 →
  A_fraction = A_share / total_profit →
  A_fraction = 1/3 :=
by
  intros h_total_profit h_A_share h_B_fraction h_C_fraction h_A_fraction
  sorry

end A_subscribed_fraction_l176_176282


namespace bird_counts_l176_176934

theorem bird_counts :
  ∀ (num_cages_1 num_cages_2 num_cages_empty parrot_per_cage parakeet_per_cage canary_per_cage cockatiel_per_cage lovebird_per_cage finch_per_cage total_cages : ℕ),
    num_cages_1 = 7 →
    num_cages_2 = 6 →
    num_cages_empty = 2 →
    parrot_per_cage = 3 →
    parakeet_per_cage = 5 →
    canary_per_cage = 4 →
    cockatiel_per_cage = 2 →
    lovebird_per_cage = 3 →
    finch_per_cage = 1 →
    total_cages = 15 →
    (num_cages_1 * parrot_per_cage = 21) ∧
    (num_cages_1 * parakeet_per_cage = 35) ∧
    (num_cages_1 * canary_per_cage = 28) ∧
    (num_cages_2 * cockatiel_per_cage = 12) ∧
    (num_cages_2 * lovebird_per_cage = 18) ∧
    (num_cages_2 * finch_per_cage = 6) :=
by
  intros
  sorry

end bird_counts_l176_176934


namespace exponent_value_l176_176852

theorem exponent_value (exponent : ℕ) (y: ℕ) :
  (12 ^ exponent) * (6 ^ 4) / 432 = y → y = 36 → exponent = 1 :=
by
  intro h1 h2
  sorry

end exponent_value_l176_176852


namespace smaller_angle_at_10_15_p_m_l176_176028

-- Definitions of conditions
def clock_hours : ℕ := 12
def degrees_per_hour : ℚ := 360 / clock_hours
def minute_hand_position : ℚ := (15 / 60) * 360
def hour_hand_position : ℚ := 10 * degrees_per_hour + (15 / 60) * degrees_per_hour
def absolute_difference : ℚ := |hour_hand_position - minute_hand_position|
def smaller_angle : ℚ := 360 - absolute_difference

-- Prove that the smaller angle is 142.5°
theorem smaller_angle_at_10_15_p_m : smaller_angle = 142.5 := by
  sorry

end smaller_angle_at_10_15_p_m_l176_176028


namespace initial_number_of_numbers_is_five_l176_176342

-- Define the conditions and the given problem
theorem initial_number_of_numbers_is_five
  (n : ℕ) (S : ℕ)
  (h1 : S / n = 27)
  (h2 : (S - 35) / (n - 1) = 25) : n = 5 :=
by
  sorry

end initial_number_of_numbers_is_five_l176_176342


namespace sum_of_reciprocals_of_roots_l176_176680

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 + r2 = 17) (h2 : r1 * r2 = 8) :
  1 / r1 + 1 / r2 = 17 / 8 :=
by
  sorry

end sum_of_reciprocals_of_roots_l176_176680


namespace set_B_forms_triangle_l176_176194

theorem set_B_forms_triangle (a b c : ℝ) (h1 : a = 25) (h2 : b = 24) (h3 : c = 7):
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end set_B_forms_triangle_l176_176194


namespace minimum_value_expr_pos_reals_l176_176056

noncomputable def expr (a b : ℝ) := a^2 + b^2 + 2 * a * b + 1 / (a + b)^2

theorem minimum_value_expr_pos_reals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : 
  (expr a b) ≥ 2 :=
sorry

end minimum_value_expr_pos_reals_l176_176056


namespace books_on_shelf_l176_176422

theorem books_on_shelf (original_books : ℕ) (books_added : ℕ) (total_books : ℕ) (h1 : original_books = 38) 
(h2 : books_added = 10) : total_books = 48 :=
by 
  sorry

end books_on_shelf_l176_176422


namespace func_positive_range_l176_176991

theorem func_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) → (-4 < a ∧ a < 4) := 
by 
  sorry

end func_positive_range_l176_176991


namespace coffee_shop_spending_l176_176694

variable (R S : ℝ)

theorem coffee_shop_spending (h1 : S = 0.60 * R) (h2 : R = S + 12.50) : R + S = 50 :=
by
  sorry

end coffee_shop_spending_l176_176694


namespace rooks_placement_possible_l176_176346

/-- 
  It is possible to place 8 rooks on a chessboard such that they do not attack each other
  and each rook stands on cells of different colors, given that the chessboard is divided 
  into 32 colors with exactly two cells of each color.
-/
theorem rooks_placement_possible :
  ∃ (placement : Fin 8 → Fin 8 × Fin 8),
    (∀ i j, i ≠ j → (placement i).fst ≠ (placement j).fst ∧ (placement i).snd ≠ (placement j).snd) ∧
    (∀ i j, i ≠ j → (placement i ≠ placement j)) ∧
    (∀ c : Fin 32, ∃! p1 p2, placement p1 = placement p2 ∧ (placement p1).fst ≠ (placement p2).fst 
                        ∧ (placement p1).snd ≠ (placement p2).snd) :=
by
  sorry

end rooks_placement_possible_l176_176346


namespace total_students_in_school_l176_176623

theorem total_students_in_school 
  (below_8_percent : ℝ) (above_8_ratio : ℝ) (students_8 : ℕ) : 
  below_8_percent = 0.20 → above_8_ratio = 2/3 → students_8 = 12 → 
  (∃ T : ℕ, T = 25) :=
by
  sorry

end total_students_in_school_l176_176623


namespace degrees_to_radians_l176_176963

theorem degrees_to_radians (deg : ℝ) (rad : ℝ) (h1 : 1 = π / 180) (h2 : deg = 60) : rad = deg * (π / 180) :=
by
  sorry

end degrees_to_radians_l176_176963


namespace union_of_sets_l176_176533

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l176_176533


namespace total_newspapers_collected_l176_176420

-- Definitions based on the conditions
def Chris_collected : ℕ := 42
def Lily_collected : ℕ := 23

-- The proof statement
theorem total_newspapers_collected :
  Chris_collected + Lily_collected = 65 := by
  sorry

end total_newspapers_collected_l176_176420


namespace molecular_weight_of_4_moles_AlCl3_is_correct_l176_176898

/-- The atomic weight of aluminum (Al) is 26.98 g/mol. -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of chlorine (Cl) is 35.45 g/mol. -/
def atomic_weight_Cl : ℝ := 35.45

/-- A molecule of AlCl3 consists of 1 atom of Al and 3 atoms of Cl. -/
def molecular_weight_AlCl3 := (1 * atomic_weight_Al) + (3 * atomic_weight_Cl)

/-- The total weight of 4 moles of AlCl3. -/
def total_weight_4_moles_AlCl3 := 4 * molecular_weight_AlCl3

/-- We prove that the total weight of 4 moles of AlCl3 is 533.32 g. -/
theorem molecular_weight_of_4_moles_AlCl3_is_correct :
  total_weight_4_moles_AlCl3 = 533.32 :=
sorry

end molecular_weight_of_4_moles_AlCl3_is_correct_l176_176898


namespace find_salary_month_l176_176961

variable (J F M A May : ℝ)

def condition_1 : Prop := (J + F + M + A) / 4 = 8000
def condition_2 : Prop := (F + M + A + May) / 4 = 8450
def condition_3 : Prop := J = 4700
def condition_4 (X : ℝ) : Prop := X = 6500

theorem find_salary_month (J F M A May : ℝ) 
  (h1 : condition_1 J F M A) 
  (h2 : condition_2 F M A May) 
  (h3 : condition_3 J) 
  : ∃ M : ℝ, condition_4 May :=
by sorry

end find_salary_month_l176_176961


namespace charity_years_l176_176054

theorem charity_years :
  ∃! pairs : List (ℕ × ℕ), 
    (∀ (w m : ℕ), (w, m) ∈ pairs → 18 * w + 30 * m = 55 * 12) ∧
    pairs.length = 6 :=
by
  sorry

end charity_years_l176_176054


namespace parabola_pass_through_fixed_point_l176_176890

theorem parabola_pass_through_fixed_point
  (p : ℝ) (hp : p > 0)
  (xM yM : ℝ) (hM : (xM, yM) = (1, -2))
  (hMp : yM^2 = 2 * p * xM)
  (xA yA xC yC xB yB xD yD : ℝ)
  (hxA : xA = xC ∨ xA ≠ xC)
  (hxB : xB = xD ∨ xB ≠ xD)
  (x2 y0 : ℝ) (h : (x2, y0) = (2, 0))
  (m1 m2 : ℝ) (hm1m2 : m1 * m2 = -1)
  (l1_intersect_A : xA = m1 * yA + 2)
  (l1_intersect_C : xC = m1 * yC + 2)
  (l2_intersect_B : xB = m2 * yB + 2)
  (l2_intersect_D : xD = m2 * yD + 2)
  (hMidM : (2 * xA + 2 * xC = 4 * xM ∧ 2 * yA + 2 * yC = 4 * yM))
  (hMidN : (2 * xB + 2 * xD = 4 * xM ∧ 2 * yB + 2 * yD = 4 * yM)) :
  (yM^2 = 4 * xM) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, y = k * x ↔ y = xM / (m1 + m2) ∧ y = m1) :=
sorry

end parabola_pass_through_fixed_point_l176_176890


namespace complement_intersection_l176_176403

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l176_176403


namespace fibonacci_150_mod_9_l176_176821

def fibonacci (n : ℕ) : ℕ :=
  if h : n < 2 then n else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 :=
  sorry

end fibonacci_150_mod_9_l176_176821


namespace chessboard_tiling_l176_176561

theorem chessboard_tiling (chessboard : Fin 8 × Fin 8 → Prop) (colors : Fin 8 × Fin 8 → Bool)
  (removed_squares : (Fin 8 × Fin 8) × (Fin 8 × Fin 8))
  (h_diff_colors : colors removed_squares.1 ≠ colors removed_squares.2) :
  ∃ f : (Fin 8 × Fin 8) → (Fin 8 × Fin 8), ∀ x, chessboard x → chessboard (f x) :=
by
  sorry

end chessboard_tiling_l176_176561


namespace correct_calculation_l176_176897

-- Define the conditions of the problem
variable (x : ℕ)
variable (h : x + 5 = 43)

-- The theorem we want to prove
theorem correct_calculation : 5 * x = 190 :=
by
  -- Since Lean requires a proof and we're skipping it, we use 'sorry'
  sorry

end correct_calculation_l176_176897


namespace find_a_b_l176_176449

theorem find_a_b (a b : ℤ) (h: 4 * a^2 + 3 * b^2 + 10 * a * b = 144) :
    (a = 2 ∧ b = 4) :=
by {
  sorry
}

end find_a_b_l176_176449


namespace binom_30_3_eq_4060_l176_176078

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end binom_30_3_eq_4060_l176_176078


namespace concentric_circles_circumference_difference_l176_176181

theorem concentric_circles_circumference_difference :
  ∀ (radius_diff inner_diameter : ℝ),
  radius_diff = 15 →
  inner_diameter = 50 →
  ((π * (inner_diameter + 2 * radius_diff)) - (π * inner_diameter)) = 30 * π :=
by
  sorry

end concentric_circles_circumference_difference_l176_176181


namespace tile_border_ratio_l176_176016

theorem tile_border_ratio (n : ℕ) (t w : ℝ) (H1 : n = 30)
  (H2 : 900 * t^2 / (30 * t + 30 * w)^2 = 0.81) :
  w / t = 1 / 9 :=
by
  sorry

end tile_border_ratio_l176_176016


namespace total_cost_of_new_movie_l176_176209

noncomputable def previous_movie_length_hours : ℕ := 2
noncomputable def new_movie_length_increase_percent : ℕ := 60
noncomputable def previous_movie_cost_per_minute : ℕ := 50
noncomputable def new_movie_cost_per_minute_factor : ℕ := 2 

theorem total_cost_of_new_movie : 
  let new_movie_length_hours := previous_movie_length_hours + (previous_movie_length_hours * new_movie_length_increase_percent / 100)
  let new_movie_length_minutes := new_movie_length_hours * 60
  let new_movie_cost_per_minute := previous_movie_cost_per_minute * new_movie_cost_per_minute_factor
  let total_cost := new_movie_length_minutes * new_movie_cost_per_minute
  total_cost = 19200 := 
by
  sorry

end total_cost_of_new_movie_l176_176209


namespace find_analytical_expression_function_increasing_inequality_solution_l176_176068

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Conditions
variables {a b x : ℝ}
axiom odd_function : ∀ x : ℝ, f a b (-x) = -f a b x
axiom half_value : f a b (1/2) = 2/5

-- Questions/Statements

-- 1. Analytical expression
theorem find_analytical_expression :
  ∃ a b, f a b x = x / (1 + x^2) := 
sorry

-- 2. Increasing function
theorem function_increasing :
  ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f 1 0 x1 < f 1 0 x2 := 
sorry

-- 3. Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 ((-1 + Real.sqrt 5) / 2)) → f 1 0 (x^2 - 1) + f 1 0 x < 0 := 
sorry

end find_analytical_expression_function_increasing_inequality_solution_l176_176068


namespace greatest_c_for_expression_domain_all_real_l176_176187

theorem greatest_c_for_expression_domain_all_real :
  ∃ c : ℤ, c ≤ 7 ∧ c ^ 2 < 60 ∧ ∀ d : ℤ, d > 7 → ¬ (d ^ 2 < 60) := sorry

end greatest_c_for_expression_domain_all_real_l176_176187


namespace imo_1989_q6_l176_176872

-- Define the odd integer m greater than 2
def isOdd (m : ℕ) := ∃ k : ℤ, m = 2 * k + 1

-- Define the condition for divisibility
def smallest_n (m : ℕ) (k : ℕ) (p : ℕ) : ℕ :=
  if k ≤ 1989 then 2 ^ (1989 - k) else 1

theorem imo_1989_q6 
  (m : ℕ) (h_m_gt2 : m > 2) (h_m_odd : isOdd m) (k : ℕ) (p : ℕ) (h_m_form : m = 2^k * p - 1) (h_p_odd : isOdd p) (h_k_gt1 : k > 1) :
  ∃ n : ℕ, (2^1989 ∣ m^n - 1) ∧ n = smallest_n m k p :=
by
  sorry

end imo_1989_q6_l176_176872


namespace width_of_larger_cuboid_l176_176589

theorem width_of_larger_cuboid
    (length_larger : ℝ)
    (width_larger : ℝ)
    (height_larger : ℝ)
    (length_smaller : ℝ)
    (width_smaller : ℝ)
    (height_smaller : ℝ)
    (num_smaller : ℕ)
    (volume_larger : ℝ)
    (volume_smaller : ℝ)
    (divided_into : Real) :
    length_larger = 12 → height_larger = 10 →
    length_smaller = 5 → width_smaller = 3 → height_smaller = 2 →
    num_smaller = 56 →
    volume_smaller = length_smaller * width_smaller * height_smaller →
    volume_larger = num_smaller * volume_smaller →
    volume_larger = length_larger * width_larger * height_larger →
    divided_into = volume_larger / (length_larger * height_larger) →
    width_larger = 14 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end width_of_larger_cuboid_l176_176589


namespace angle_measure_l176_176705

theorem angle_measure (x : ℝ) (h : 180 - x = (90 - x) - 4) : x = 60 := by
  sorry

end angle_measure_l176_176705


namespace residue_of_neg_1237_mod_37_l176_176964

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end residue_of_neg_1237_mod_37_l176_176964


namespace acres_used_for_corn_l176_176164

-- Define the conditions in the problem:
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio : ℕ := ratio_beans + ratio_wheat + ratio_corn

-- Proof problem statement: Prove the number of acres used for corn is 376 acres
theorem acres_used_for_corn : total_land * ratio_corn / total_ratio = 376 := by
  -- Proof goes here
  sorry

end acres_used_for_corn_l176_176164


namespace factorize_cubed_sub_four_l176_176579

theorem factorize_cubed_sub_four (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
by
  sorry

end factorize_cubed_sub_four_l176_176579


namespace mike_total_hours_worked_l176_176307

-- Define the conditions
def time_to_wash_car := 10
def time_to_change_oil := 15
def time_to_change_tires := 30

def number_of_cars_washed := 9
def number_of_oil_changes := 6
def number_of_tire_changes := 2

-- Define the conversion factor
def minutes_per_hour := 60

-- Prove that the total time worked equals 4 hours
theorem mike_total_hours_worked : 
  (number_of_cars_washed * time_to_wash_car + 
   number_of_oil_changes * time_to_change_oil + 
   number_of_tire_changes * time_to_change_tires) / minutes_per_hour = 4 := by
  sorry

end mike_total_hours_worked_l176_176307


namespace problem_f_symmetry_problem_f_definition_problem_correct_answer_l176_176743

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then Real.log x else Real.log (2 - x)

theorem problem_f_symmetry (x : ℝ) : f (2 - x) = f x := 
sorry

theorem problem_f_definition (x : ℝ) (hx : x ≥ 1) : f x = Real.log x :=
sorry

theorem problem_correct_answer: 
  f (1 / 2) < f 2 ∧ f 2 < f (1 / 3) :=
sorry

end problem_f_symmetry_problem_f_definition_problem_correct_answer_l176_176743


namespace compare_neg_fractions_l176_176165

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end compare_neg_fractions_l176_176165


namespace max_daily_sales_revenue_l176_176382

noncomputable def p (t : ℕ) : ℝ :=
if 0 < t ∧ t < 25 then t + 20
else if 25 ≤ t ∧ t ≤ 30 then -t + 70
else 0

noncomputable def Q (t : ℕ) : ℝ :=
if 0 < t ∧ t ≤ 30 then -t + 40 else 0

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ (p t) * (Q t) = 1125 ∧
  ∀ t' : ℕ, 0 < t' ∧ t' ≤ 30 → (p t') * (Q t') ≤ 1125 :=
sorry

end max_daily_sales_revenue_l176_176382


namespace sofie_total_distance_l176_176853

-- Definitions for the conditions
def side1 : ℝ := 25
def side2 : ℝ := 35
def side3 : ℝ := 20
def side4 : ℝ := 40
def side5 : ℝ := 30
def laps_initial : ℕ := 2
def laps_additional : ℕ := 5
def perimeter : ℝ := side1 + side2 + side3 + side4 + side5

-- Theorem statement
theorem sofie_total_distance : laps_initial * perimeter + laps_additional * perimeter = 1050 := by
  sorry

end sofie_total_distance_l176_176853


namespace sqrt_four_eq_two_l176_176142

theorem sqrt_four_eq_two : Real.sqrt 4 = 2 :=
by
  sorry

end sqrt_four_eq_two_l176_176142


namespace identify_1000g_weight_l176_176893

-- Define the masses of the weights
def masses : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- The statement that needs to be proven
theorem identify_1000g_weight (masses : List ℕ) (h : masses = [1000, 1001, 1002, 1004, 1007]) :
  ∃ w, w ∈ masses ∧ w = 1000 ∧ by sorry :=
sorry

end identify_1000g_weight_l176_176893


namespace difference_between_q_and_r_l176_176102

-- Define the variables for shares with respect to the common multiple x
def p_share (x : Nat) : Nat := 3 * x
def q_share (x : Nat) : Nat := 7 * x
def r_share (x : Nat) : Nat := 12 * x

-- Given condition: The difference between q's share and p's share is Rs. 4000
def condition_1 (x : Nat) : Prop := (q_share x - p_share x = 4000)

-- Define the theorem to prove the difference between r and q's share is Rs. 5000
theorem difference_between_q_and_r (x : Nat) (h : condition_1 x) : r_share x - q_share x = 5000 :=
by
  sorry

end difference_between_q_and_r_l176_176102


namespace paperback_copies_sold_l176_176527

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end paperback_copies_sold_l176_176527


namespace imaginary_unit_sum_l176_176975

theorem imaginary_unit_sum (i : ℂ) (H : i^4 = 1) : i^1234 + i^1235 + i^1236 + i^1237 = 0 :=
by
  sorry

end imaginary_unit_sum_l176_176975


namespace profit_percentage_l176_176552

theorem profit_percentage (SP CP : ℝ) (hs : SP = 270) (hc : CP = 225) : 
  ((SP - CP) / CP) * 100 = 20 :=
by
  rw [hs, hc]
  sorry  -- The proof will go here

end profit_percentage_l176_176552


namespace planes_parallel_l176_176511

variables {a b c : Type} {α β γ : Type}
variables (h_lines : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Conditions based on the propositions
variables (h1 : parallel α γ)
variables (h2 : parallel β γ)

-- Theorem to prove
theorem planes_parallel (h1: parallel α γ) (h2 : parallel β γ) : parallel α β := 
sorry

end planes_parallel_l176_176511


namespace correct_operation_l176_176250

variable {x y : ℝ}

theorem correct_operation :
  (2 * x^2 + 4 * x^2 = 6 * x^2) → 
  (x * x^3 = x^4) → 
  ((x^3)^2 = x^6) →
  ((xy)^5 = x^5 * y^5) →
  ((x^3)^2 = x^6) := 
by 
  intros h1 h2 h3 h4
  exact h3

end correct_operation_l176_176250


namespace g_values_l176_176503

variable (g : ℝ → ℝ)

-- Condition: ∀ x y z ∈ ℝ, g(x^2 + y * g(z)) = x * g(x) + 2 * z * g(y)
axiom g_axiom : ∀ x y z : ℝ, g (x^2 + y * g z) = x * g x + 2 * z * g y

-- Proposition: The possible values of g(4) are 0 and 8.
theorem g_values : g 4 = 0 ∨ g 4 = 8 :=
by
  sorry

end g_values_l176_176503


namespace parity_of_expression_l176_176572

theorem parity_of_expression
  (a b c : ℕ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_c_pos : c > 0) :
  ((3^a + (b + 2)^2 * c) % 2 = 1 ↔ c % 2 = 0) ∧ 
  ((3^a + (b + 2)^2 * c) % 2 = 0 ↔ c % 2 = 1) :=
by sorry

end parity_of_expression_l176_176572


namespace min_route_length_5x5_l176_176995

-- Definition of the grid and its properties
def grid : Type := Fin 5 × Fin 5

-- Define a function to calculate the minimum route length
noncomputable def min_route_length (grid_size : ℕ) : ℕ :=
  if h : grid_size = 5 then 68 else 0

-- The proof problem statement
theorem min_route_length_5x5 : min_route_length 5 = 68 :=
by
  -- Skipping the actual proof
  sorry

end min_route_length_5x5_l176_176995


namespace reciprocal_of_minus_one_half_l176_176367

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l176_176367


namespace distance_after_four_steps_l176_176088

theorem distance_after_four_steps (total_distance : ℝ) (steps : ℕ) (steps_taken : ℕ) :
   total_distance = 25 → steps = 7 → steps_taken = 4 → (steps_taken * (total_distance / steps) = 100 / 7) :=
by
    intro h1 h2 h3
    rw [h1, h2, h3]
    simp
    sorry

end distance_after_four_steps_l176_176088


namespace lcm_of_two_numbers_l176_176880
-- Importing the math library

-- Define constants and variables
variables (A B LCM HCF : ℕ)

-- Given conditions
def product_condition : Prop := A * B = 17820
def hcf_condition : Prop := HCF = 12
def lcm_condition : Prop := LCM = Nat.lcm A B

-- Theorem to prove
theorem lcm_of_two_numbers : product_condition A B ∧ hcf_condition HCF →
                              lcm_condition A B LCM →
                              LCM = 1485 := 
by
  sorry

end lcm_of_two_numbers_l176_176880


namespace trajectory_passes_quadrants_l176_176865

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 4

-- Define the condition for a point to belong to the first quadrant
def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Define the condition for a point to belong to the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- State the theorem that the trajectory of point P passes through the first and second quadrants
theorem trajectory_passes_quadrants :
  (∃ x y : ℝ, circle_equation x y ∧ in_first_quadrant x y) ∧
  (∃ x y : ℝ, circle_equation x y ∧ in_second_quadrant x y) :=
sorry

end trajectory_passes_quadrants_l176_176865


namespace triangle_count_l176_176810

theorem triangle_count (a b c : ℕ) (h1 : a + b + c = 15) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a + b > c) :
  ∃ (n : ℕ), n = 7 :=
by
  -- Proceed with the proof steps, using a, b, c satisfying the given conditions
  sorry

end triangle_count_l176_176810


namespace contrapositive_eq_l176_176098

variables (P Q : Prop)

theorem contrapositive_eq : (¬P → Q) ↔ (¬Q → P) := 
by {
    sorry
}

end contrapositive_eq_l176_176098


namespace power_sum_l176_176145

theorem power_sum : (-2) ^ 2007 + (-2) ^ 2008 = 2 ^ 2007 := by
  sorry

end power_sum_l176_176145


namespace train_speed_l176_176595

noncomputable def speed_of_train_kmph (L V : ℝ) : ℝ :=
  3.6 * V

theorem train_speed
  (L V : ℝ)
  (h1 : L = 18 * V)
  (h2 : L + 340 = 35 * V) :
  speed_of_train_kmph L V = 72 :=
by
  sorry

end train_speed_l176_176595


namespace magician_starting_decks_l176_176633

def starting_decks (price_per_deck earned remaining_decks : ℕ) : ℕ :=
  earned / price_per_deck + remaining_decks

theorem magician_starting_decks :
  starting_decks 2 4 3 = 5 :=
by
  sorry

end magician_starting_decks_l176_176633


namespace prove_a_eq_b_l176_176177

theorem prove_a_eq_b (a b : ℝ) (h : 1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b)) : a = b :=
sorry

end prove_a_eq_b_l176_176177


namespace triangle_angle_sum_cannot_exist_l176_176033

theorem triangle_angle_sum (A : Real) (B : Real) (C : Real) :
    A + B + C = 180 :=
sorry

theorem cannot_exist (right_two_60 : ¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) 
    (scalene_100 : ∃ A B C : Real, A = 100 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A + B + C = 180)
    (isosceles_two_70 : ∃ A B C : Real, A = B ∧ A = 70 ∧ C = 180 - 2 * A ∧ A + B + C = 180)
    (equilateral_60 : ∃ A B C : Real, A = 60 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180)
    (one_90_two_50 : ¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :
  (¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) ∧
  (¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :=
by
  sorry

end triangle_angle_sum_cannot_exist_l176_176033


namespace salt_solution_problem_l176_176114

theorem salt_solution_problem
  (x y : ℝ)
  (h1 : 70 + x + y = 200)
  (h2 : 0.20 * 70 + 0.60 * x + 0.35 * y = 0.45 * 200) :
  x = 122 ∧ y = 8 :=
by
  sorry

end salt_solution_problem_l176_176114


namespace find_y_l176_176879

theorem find_y (x y : ℤ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 :=
by
  subst h1
  have h : 3 * 4 + 2 * y = 30 := by rw [h2]
  linarith

end find_y_l176_176879


namespace train_cross_time_platform_l176_176992

def speed := 36 -- in kmph
def time_for_pole := 12 -- in seconds
def time_for_platform := 44.99736021118311 -- in seconds

theorem train_cross_time_platform :
  time_for_platform = 44.99736021118311 :=
by
  sorry

end train_cross_time_platform_l176_176992


namespace dot_product_necessity_l176_176476

variables (a b : ℝ → ℝ → ℝ)

def dot_product (a b : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
  a x y * b x y

def angle_is_acute (a b : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  0 < a x y

theorem dot_product_necessity (a b : ℝ → ℝ → ℝ) (x y : ℝ) :
  dot_product a b x y > 0 ↔ angle_is_acute a b x y :=
sorry

end dot_product_necessity_l176_176476


namespace dwayneA_students_l176_176222

-- Define the number of students who received an 'A' in Mrs. Carter's class
def mrsCarterA := 8
-- Define the total number of students in Mrs. Carter's class
def mrsCarterTotal := 20
-- Define the total number of students in Mr. Dwayne's class
def mrDwayneTotal := 30
-- Calculate the ratio of students who received an 'A' in Mrs. Carter's class
def carterRatio := mrsCarterA / mrsCarterTotal
-- Calculate the number of students who received an 'A' in Mr. Dwayne's class based on the same ratio
def mrDwayneA := (carterRatio * mrDwayneTotal)

-- Prove that the number of students who received an 'A' in Mr. Dwayne's class is 12
theorem dwayneA_students :
  mrDwayneA = 12 := 
by
  -- Since def calculation does not automatically prove equality, we will need to use sorry to skip the proof for now.
  sorry

end dwayneA_students_l176_176222


namespace min_value_g_l176_176745

noncomputable def g (x : ℝ) : ℝ := (6 * x^2 + 11 * x + 17) / (7 * (2 + x))

theorem min_value_g : ∃ x, x ≥ 0 ∧ g x = 127 / 24 :=
by
  sorry

end min_value_g_l176_176745


namespace find_cost_price_l176_176330

variable (CP : ℝ) -- cost price
variable (SP_loss SP_gain : ℝ) -- selling prices

-- Conditions
def loss_condition := SP_loss = 0.9 * CP
def gain_condition := SP_gain = 1.04 * CP
def difference_condition := SP_gain - SP_loss = 190

-- Theorem to prove
theorem find_cost_price (h_loss : loss_condition CP SP_loss)
                        (h_gain : gain_condition CP SP_gain)
                        (h_diff : difference_condition SP_loss SP_gain) :
  CP = 1357.14 := 
sorry

end find_cost_price_l176_176330


namespace find_m_from_root_l176_176036

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end find_m_from_root_l176_176036


namespace determine_a1_a2_a3_l176_176497

theorem determine_a1_a2_a3 (a a1 a2 a3 : ℝ)
  (h : ∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3) :
  a1 + a2 + a3 = 19 :=
by
  sorry

end determine_a1_a2_a3_l176_176497


namespace correct_operation_l176_176876

variable (x y a : ℝ)

lemma correct_option_C :
  -4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2 :=
by sorry

lemma wrong_option_A :
  x * (2 * x + 3) ≠ 2 * x^2 + 3 :=
by sorry

lemma wrong_option_B :
  a^2 + a^3 ≠ a^5 :=
by sorry

lemma wrong_option_D :
  x^3 * x^2 ≠ x^6 :=
by sorry

theorem correct_operation :
  ((-4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2) ∧
   (x * (2 * x + 3) ≠ 2 * x^2 + 3) ∧
   (a^2 + a^3 ≠ a^5) ∧
   (x^3 * x^2 ≠ x^6)) :=
by
  exact ⟨correct_option_C x y, wrong_option_A x, wrong_option_B a, wrong_option_D x⟩

end correct_operation_l176_176876


namespace curve_product_l176_176734

theorem curve_product (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) : a * b = -3 := by
  sorry

end curve_product_l176_176734


namespace no_rational_solution_l176_176791

theorem no_rational_solution :
  ¬ ∃ (x y z : ℚ), 
  x + y + z = 0 ∧ x^2 + y^2 + z^2 = 100 := sorry

end no_rational_solution_l176_176791


namespace geometric_sequence_second_term_l176_176811

theorem geometric_sequence_second_term (a_1 q a_3 a_4 : ℝ) (h3 : a_1 * q^2 = 12) (h4 : a_1 * q^3 = 18) : a_1 * q = 8 :=
by
  sorry

end geometric_sequence_second_term_l176_176811


namespace total_soaking_time_l176_176869

def stain_times (n_grass n_marinara n_coffee n_ink : Nat) (t_grass t_marinara t_coffee t_ink : Nat) : Nat :=
  n_grass * t_grass + n_marinara * t_marinara + n_coffee * t_coffee + n_ink * t_ink

theorem total_soaking_time :
  let shirt_grass_stains := 2
  let shirt_grass_time := 3
  let shirt_marinara_stains := 1
  let shirt_marinara_time := 7
  let pants_coffee_stains := 1
  let pants_coffee_time := 10
  let pants_ink_stains := 1
  let pants_ink_time := 5
  let socks_grass_stains := 1
  let socks_grass_time := 3
  let socks_marinara_stains := 2
  let socks_marinara_time := 7
  let socks_ink_stains := 1
  let socks_ink_time := 5
  let additional_ink_time := 2

  let shirt_time := stain_times shirt_grass_stains shirt_marinara_stains 0 0 shirt_grass_time shirt_marinara_time 0 0
  let pants_time := stain_times 0 0 pants_coffee_stains pants_ink_stains 0 0 pants_coffee_time pants_ink_time
  let socks_time := stain_times socks_grass_stains socks_marinara_stains 0 socks_ink_stains socks_grass_time socks_marinara_time 0 socks_ink_time
  let total_time := shirt_time + pants_time + socks_time
  let total_ink_stains := pants_ink_stains + socks_ink_stains
  let additional_ink_total_time := total_ink_stains * additional_ink_time
  let final_total_time := total_time + additional_ink_total_time

  final_total_time = 54 :=
by
  sorry

end total_soaking_time_l176_176869


namespace multiply_or_divide_inequality_by_negative_number_l176_176669

theorem multiply_or_divide_inequality_by_negative_number {a b c : ℝ} (h : a < b) (hc : c < 0) :
  c * a > c * b ∧ a / c > b / c :=
sorry

end multiply_or_divide_inequality_by_negative_number_l176_176669


namespace halfway_between_one_third_and_one_eighth_l176_176656

theorem halfway_between_one_third_and_one_eighth : (1/3 + 1/8) / 2 = 11 / 48 :=
by
  -- The proof goes here
  sorry

end halfway_between_one_third_and_one_eighth_l176_176656


namespace candy_distribution_l176_176364

theorem candy_distribution (A B : ℕ) (h1 : 7 * A = B + 12) (h2 : 3 * A = B - 20) : A + B = 52 :=
by {
  -- proof goes here
  sorry
}

end candy_distribution_l176_176364


namespace mary_screws_sections_l176_176940

def number_of_sections (initial_screws : Nat) (multiplier : Nat) (screws_per_section : Nat) : Nat :=
  let additional_screws := initial_screws * multiplier
  let total_screws := initial_screws + additional_screws
  total_screws / screws_per_section

theorem mary_screws_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end mary_screws_sections_l176_176940


namespace total_stamps_l176_176921

-- Definitions based on the conditions
def AJ := 370
def KJ := AJ / 2
def CJ := 2 * KJ + 5

-- Proof Statement
theorem total_stamps : AJ + KJ + CJ = 930 := by
  sorry

end total_stamps_l176_176921


namespace root_equation_identity_l176_176304

theorem root_equation_identity {a b c p q : ℝ} 
  (h1 : a^2 + p*a + 1 = 0)
  (h2 : b^2 + p*b + 1 = 0)
  (h3 : b^2 + q*b + 2 = 0)
  (h4 : c^2 + q*c + 2 = 0) 
  : (b - a) * (b - c) = p*q - 6 := 
sorry

end root_equation_identity_l176_176304


namespace number_of_distinct_cubes_l176_176733

theorem number_of_distinct_cubes (w b : ℕ) (total_cubes : ℕ) (dim : ℕ) :
  w + b = total_cubes ∧ total_cubes = 8 ∧ dim = 2 ∧ w = 6 ∧ b = 2 →
  (number_of_distinct_orbits : ℕ) = 1 :=
by
  -- Conditions
  intros h
  -- Translation of conditions into a useful form
  let num_cubes := 8
  let distinct_configurations := 1
  -- Burnside's Lemma applied to find the distinct configurations
  sorry

end number_of_distinct_cubes_l176_176733


namespace probability_of_drawing_red_ball_l176_176107

noncomputable def probability_of_red_ball (total_balls red_balls : ℕ) : ℚ :=
  red_balls / total_balls

theorem probability_of_drawing_red_ball:
  probability_of_red_ball 5 3 = 3 / 5 :=
by
  unfold probability_of_red_ball
  norm_num

end probability_of_drawing_red_ball_l176_176107


namespace range_of_a_l176_176219

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l176_176219


namespace area_of_square_l176_176445

theorem area_of_square (r : ℝ) (b : ℝ) (ℓ : ℝ) (area_rect : ℝ) 
    (h₁ : ℓ = 2 / 3 * r) 
    (h₂ : r = b) 
    (h₃ : b = 13) 
    (h₄ : area_rect = 598) 
    (h₅ : area_rect = ℓ * b) : 
    r^2 = 4761 := 
sorry

end area_of_square_l176_176445


namespace integer_solutions_l176_176169

theorem integer_solutions (n : ℕ) :
  n = 7 ↔ ∃ (x : ℤ), ∀ (x : ℤ), (3 * x^2 + 17 * x + 14 ≤ 20)  :=
by
  sorry

end integer_solutions_l176_176169


namespace cube_sum_gt_zero_l176_176710

variable {x y z : ℝ}

theorem cube_sum_gt_zero (h1 : x < y) (h2 : y < z) : 
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 :=
sorry

end cube_sum_gt_zero_l176_176710


namespace rectangle_area_l176_176473

theorem rectangle_area (r length width : ℝ) (h_ratio : length = 3 * width) (h_incircle : width = 2 * r) (h_r : r = 7) : length * width = 588 :=
by
  sorry

end rectangle_area_l176_176473


namespace asymptotes_and_eccentricity_of_hyperbola_l176_176200

noncomputable def hyperbola_asymptotes_and_eccentricity : Prop :=
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt 3
  ∀ (x y : ℝ), x^2 - (y^2 / 2) = 1 →
    ((y = 2 * x ∨ y = -2 * x) ∧ Real.sqrt (1 + (b^2 / a^2)) = c)

theorem asymptotes_and_eccentricity_of_hyperbola :
  hyperbola_asymptotes_and_eccentricity :=
by
  sorry

end asymptotes_and_eccentricity_of_hyperbola_l176_176200


namespace swim_time_l176_176555

-- Definitions based on conditions:
def speed_in_still_water : ℝ := 6.5 -- speed of the man in still water (km/h)
def distance_downstream : ℝ := 16 -- distance swam downstream (km)
def distance_upstream : ℝ := 10 -- distance swam upstream (km)
def time_downstream := 2 -- time taken to swim downstream (hours)
def time_upstream := 2 -- time taken to swim upstream (hours)

-- Defining the speeds taking the current into account:
def speed_downstream (c : ℝ) : ℝ := speed_in_still_water + c
def speed_upstream (c : ℝ) : ℝ := speed_in_still_water - c

-- Assumption that the time took for both downstream and upstream are equal
def time_eq (c : ℝ) : Prop :=
  distance_downstream / (speed_downstream c) = distance_upstream / (speed_upstream c)

-- The proof we need to establish:
theorem swim_time (c : ℝ) (h : time_eq c) : time_downstream = time_upstream := by
  sorry

end swim_time_l176_176555


namespace total_candies_is_829_l176_176368

-- Conditions as definitions
def Adam : ℕ := 6
def James : ℕ := 3 * Adam
def Rubert : ℕ := 4 * James
def Lisa : ℕ := 2 * Rubert
def Chris : ℕ := Lisa + 5
def Emily : ℕ := 3 * Chris - 7

-- Total candies
def total_candies : ℕ := Adam + James + Rubert + Lisa + Chris + Emily

-- Theorem to prove
theorem total_candies_is_829 : total_candies = 829 :=
by
  -- skipping the proof
  sorry

end total_candies_is_829_l176_176368


namespace problem_statement_l176_176632

theorem problem_statement (m : ℤ) (h : (m + 2)^2 = 64) : (m + 1) * (m + 3) = 63 :=
sorry

end problem_statement_l176_176632


namespace joseph_savings_ratio_l176_176022

theorem joseph_savings_ratio
    (thomas_monthly_savings : ℕ)
    (thomas_years_saving : ℕ)
    (total_savings : ℕ)
    (joseph_total_savings_is_total_minus_thomas : total_savings = thomas_monthly_savings * 12 * thomas_years_saving + (total_savings - thomas_monthly_savings * 12 * thomas_years_saving))
    (thomas_saves_each_month : thomas_monthly_savings = 40)
    (years_saving : thomas_years_saving = 6)
    (total_amount : total_savings = 4608) :
    (total_savings - thomas_monthly_savings * 12 * thomas_years_saving) / (12 * thomas_years_saving) / thomas_monthly_savings = 3 / 5 :=
by
  sorry

end joseph_savings_ratio_l176_176022


namespace dividend_calculation_l176_176426

theorem dividend_calculation :
  ∀ (divisor quotient remainder : ℝ), 
  divisor = 37.2 → 
  quotient = 14.61 → 
  remainder = 0.67 → 
  (divisor * quotient + remainder) = 544.042 :=
by
  intros divisor quotient remainder h_div h_qt h_rm
  sorry

end dividend_calculation_l176_176426


namespace perfect_square_trinomial_l176_176243

theorem perfect_square_trinomial (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 150 * x + c = (x + a)^2) → c = 5625 :=
sorry

end perfect_square_trinomial_l176_176243


namespace ratio_B_to_A_l176_176815

theorem ratio_B_to_A (A B S : ℕ) 
  (h1 : A = 2 * S)
  (h2 : A = 80)
  (h3 : B - S = 200) :
  B / A = 3 :=
by sorry

end ratio_B_to_A_l176_176815


namespace sam_pam_ratio_is_2_l176_176253

-- Definition of given conditions
def min_assigned_pages : ℕ := 25
def harrison_extra_read : ℕ := 10
def pam_extra_read : ℕ := 15
def sam_read : ℕ := 100

-- Calculations based on the given conditions
def harrison_read : ℕ := min_assigned_pages + harrison_extra_read
def pam_read : ℕ := harrison_read + pam_extra_read

-- Prove the ratio of the number of pages Sam read to the number of pages Pam read is 2
theorem sam_pam_ratio_is_2 : sam_read / pam_read = 2 := 
by
  sorry

end sam_pam_ratio_is_2_l176_176253


namespace min_sum_of_perpendicular_sides_l176_176564

noncomputable def min_sum_perpendicular_sides (a b : ℝ) (h : a * b = 100) : ℝ :=
a + b

theorem min_sum_of_perpendicular_sides {a b : ℝ} (h : a * b = 100) : min_sum_perpendicular_sides a b h = 20 :=
sorry

end min_sum_of_perpendicular_sides_l176_176564


namespace difference_fewer_children_than_adults_l176_176607

theorem difference_fewer_children_than_adults : 
  ∀ (C S : ℕ), 2 * C = S → 58 + C + S = 127 → (58 - C = 35) :=
by
  intros C S h1 h2
  sorry

end difference_fewer_children_than_adults_l176_176607


namespace linear_equation_m_equals_neg_3_l176_176268

theorem linear_equation_m_equals_neg_3 
  (m : ℤ)
  (h1 : |m| - 2 = 1)
  (h2 : m - 3 ≠ 0) :
  m = -3 :=
sorry

end linear_equation_m_equals_neg_3_l176_176268


namespace Donovan_Mitchell_goal_l176_176671

theorem Donovan_Mitchell_goal 
  (current_avg : ℕ) 
  (current_games : ℕ) 
  (target_avg : ℕ) 
  (total_games : ℕ) 
  (remaining_games : ℕ) 
  (points_scored_so_far : ℕ)
  (points_needed_total : ℕ)
  (points_needed_remaining : ℕ) :
  (current_avg = 26) ∧
  (current_games = 15) ∧
  (target_avg = 30) ∧
  (total_games = 20) ∧
  (remaining_games = 5) ∧
  (points_scored_so_far = current_avg * current_games) ∧
  (points_needed_total = target_avg * total_games) ∧
  (points_needed_remaining = points_needed_total - points_scored_so_far) →
  (points_needed_remaining / remaining_games = 42) :=
by
  sorry

end Donovan_Mitchell_goal_l176_176671


namespace cube_add_constant_135002_l176_176706

theorem cube_add_constant_135002 (n : ℤ) : 
  (∃ m : ℤ, m = n + 1 ∧ m^3 - n^3 = 135002) →
  (n = 149 ∨ n = -151) :=
by
  -- This is where the proof should go
  sorry

end cube_add_constant_135002_l176_176706


namespace function_solution_l176_176569

theorem function_solution (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = sorry) → f a = sorry → (a = 1 ∨ a = -1) :=
by
  intros hfa hfb
  sorry

end function_solution_l176_176569


namespace OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l176_176359

def combination (n k : ℕ) : ℕ := Nat.choose n k
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem OneEmptyBox (n : ℕ) (hn : n = 5) : (combination 5 2) * (arrangement 5 5) = 1200 := by
  sorry

theorem NoBoxEmptyNoCompleteMatch (n : ℕ) (hn : n = 5) : (arrangement 5 5) - 1 = 119 := by
  sorry

theorem AtLeastTwoMatches (n : ℕ) (hn : n = 5) : (arrangement 5 5) - (combination 5 1 * 9 + 44) = 31 := by
  sorry

end OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l176_176359


namespace monogramming_cost_per_stocking_l176_176001

noncomputable def total_stockings : ℕ := (5 * 5) + 4
noncomputable def price_per_stocking : ℝ := 20 - (0.10 * 20)
noncomputable def total_cost_of_stockings : ℝ := total_stockings * price_per_stocking
noncomputable def total_cost : ℝ := 1035
noncomputable def total_monogramming_cost : ℝ := total_cost - total_cost_of_stockings

theorem monogramming_cost_per_stocking :
  (total_monogramming_cost / total_stockings) = 17.69 :=
by
  sorry

end monogramming_cost_per_stocking_l176_176001


namespace train_crossing_pole_time_l176_176645

/-- 
Given the conditions:
1. The train is running at a speed of 60 km/hr.
2. The length of the train is 66.66666666666667 meters.
Prove that it takes 4 seconds for the train to cross the pole.
-/
theorem train_crossing_pole_time :
  let speed_km_hr := 60
  let length_m := 66.66666666666667
  let conversion_factor := 1000 / 3600
  let speed_m_s := speed_km_hr * conversion_factor
  let time := length_m / speed_m_s
  time = 4 :=
by
  sorry

end train_crossing_pole_time_l176_176645


namespace no_right_triangle_with_sqrt_2016_side_l176_176640

theorem no_right_triangle_with_sqrt_2016_side :
  ¬ ∃ (a b : ℤ), (a * a + b * b = 2016) ∨ (a * a + 2016 = b * b) :=
by
  sorry

end no_right_triangle_with_sqrt_2016_side_l176_176640


namespace find_principal_sum_l176_176395

theorem find_principal_sum 
  (CI SI P : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (hCI : CI = 11730) 
  (hSI : SI = 10200) 
  (hT : T = 2) 
  (hCI_formula : CI = P * ((1 + R / 100)^T - 1)) 
  (hSI_formula : SI = (P * R * T) / 100) 
  (h_diff : CI - SI = 1530) :
  P = 34000 := 
by 
  sorry

end find_principal_sum_l176_176395


namespace correct_standardized_statement_l176_176490

-- Define and state the conditions as Lean 4 definitions and propositions
structure GeometricStatement :=
  (description : String)
  (is_standardized : Prop)

def optionA : GeometricStatement := {
  description := "Line a and b intersect at point m",
  is_standardized := False -- due to use of lowercase 'm'
}

def optionB : GeometricStatement := {
  description := "Extend line AB",
  is_standardized := False -- since a line cannot be further extended
}

def optionC : GeometricStatement := {
  description := "Extend ray AO (where O is the endpoint) in the opposite direction",
  is_standardized := False -- incorrect definition of ray extension
}

def optionD : GeometricStatement := {
  description := "Extend line segment AB to C such that BC=AB",
  is_standardized := True -- correct by geometric principles
}

-- The theorem stating that option D is the correct and standardized statement
theorem correct_standardized_statement : optionD.is_standardized = True ∧
                                         optionA.is_standardized = False ∧
                                         optionB.is_standardized = False ∧
                                         optionC.is_standardized = False :=
  by sorry

end correct_standardized_statement_l176_176490


namespace repeating_decimal_356_fraction_l176_176096

noncomputable def repeating_decimal_356 := 3.0 + 56 / 99

theorem repeating_decimal_356_fraction : repeating_decimal_356 = 353 / 99 := by
  sorry

end repeating_decimal_356_fraction_l176_176096


namespace average_age_of_persons_l176_176573

theorem average_age_of_persons 
  (total_age : ℕ := 270) 
  (average_age : ℕ := 15) : 
  (total_age / average_age) = 18 := 
by { 
  sorry 
}

end average_age_of_persons_l176_176573


namespace initial_treasure_amount_l176_176279

theorem initial_treasure_amount 
  (T : ℚ)
  (h₁ : T * (1 - 1/13) * (1 - 1/17) = 150) : 
  T = 172 + 21/32 :=
sorry

end initial_treasure_amount_l176_176279


namespace length_of_room_l176_176570

theorem length_of_room (Area Width Length : ℝ) (h1 : Area = 10) (h2 : Width = 2) (h3 : Area = Length * Width) : Length = 5 :=
by
  sorry

end length_of_room_l176_176570


namespace light_travel_distance_in_km_l176_176926

-- Define the conditions
def speed_of_light_miles_per_sec : ℝ := 186282
def conversion_factor_mile_to_km : ℝ := 1.609
def time_seconds : ℕ := 500
def expected_distance_km : ℝ := 1.498 * 10^8

-- The theorem we need to prove
theorem light_travel_distance_in_km :
  (speed_of_light_miles_per_sec * time_seconds * conversion_factor_mile_to_km) = expected_distance_km :=
  sorry

end light_travel_distance_in_km_l176_176926


namespace geometric_sequence_properties_l176_176182

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 0 then 1 / 4 else (1 / 4) * 2^(n-1)

def S_n (n : ℕ) : ℚ :=
(1/4) * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_properties :
  (a_n 2 = 1 / 2) ∧ (∀ n : ℕ, 1 ≤ n → a_n n = 2^(n-3)) ∧ S_n 5 = 31 / 16 :=
by {
  sorry
}

end geometric_sequence_properties_l176_176182


namespace fencing_required_l176_176664

variable (L W : ℝ)
variable (Area : ℝ := 20 * W)

theorem fencing_required (hL : L = 20) (hArea : L * W = 600) : 20 + 2 * W = 80 := by
  sorry

end fencing_required_l176_176664


namespace gcd_of_polynomials_l176_176489

theorem gcd_of_polynomials (b : ℤ) (h : b % 2 = 1 ∧ 8531 ∣ b) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 :=
by
  sorry

end gcd_of_polynomials_l176_176489


namespace polynomial_remainder_l176_176166

theorem polynomial_remainder (x : ℤ) : (x + 1) ∣ (x^15 + 1) ↔ x = -1 := sorry

end polynomial_remainder_l176_176166


namespace find_number_l176_176657

/-- 
  Given that 23% of a number x is equal to 150, prove that x equals 15000 / 23.
-/
theorem find_number (x : ℝ) (h : (23 / 100) * x = 150) : x = 15000 / 23 :=
by
  sorry

end find_number_l176_176657


namespace probability_of_prime_or_odd_is_half_l176_176807

-- Define the list of sections on the spinner
def sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  if n < 2 then false else List.foldr (λ p b => b && (n % p ≠ 0)) true (List.range (n - 2) |>.map (λ x => x + 2))

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Define the condition of being either prime or odd
def is_prime_or_odd (n : ℕ) : Bool := is_prime n || is_odd n

-- List of favorable outcomes where the number is either prime or odd
def favorable_outcomes : List ℕ := sections.filter is_prime_or_odd

-- Calculate the probability
def probability_prime_or_odd : ℚ := (favorable_outcomes.length : ℚ) / (sections.length : ℚ)

-- Statement to prove the probability is 1/2
theorem probability_of_prime_or_odd_is_half : probability_prime_or_odd = 1 / 2 := by
  sorry

end probability_of_prime_or_odd_is_half_l176_176807


namespace avg_speed_between_B_and_C_l176_176040

noncomputable def avg_speed_from_B_to_C : ℕ := 20

theorem avg_speed_between_B_and_C
    (A_to_B_dist : ℕ := 120)
    (A_to_B_time : ℕ := 4)
    (B_to_C_dist : ℕ := 120) -- three-thirds of A_to_B_dist
    (C_to_D_dist : ℕ := 60) -- half of B_to_C_dist
    (C_to_D_time : ℕ := 2)
    (total_avg_speed : ℕ := 25)
    : avg_speed_from_B_to_C = 20 := 
  sorry

end avg_speed_between_B_and_C_l176_176040


namespace number_of_ways_to_place_rooks_l176_176206

theorem number_of_ways_to_place_rooks :
  let columns := 6
  let rows := 2006
  let rooks := 3
  ((Nat.choose columns rooks) * (rows * (rows - 1) * (rows - 2))) = 20 * 2006 * 2005 * 2004 :=
by {
  sorry
}

end number_of_ways_to_place_rooks_l176_176206


namespace polynomials_equality_l176_176419

open Polynomial

variable {F : Type*} [Field F]

theorem polynomials_equality (P Q : Polynomial F) (h : ∀ x, P.eval (P.eval (P.eval x)) = Q.eval (Q.eval (Q.eval x)) ∧ P.eval (P.eval (P.eval x)) = Q.eval (P.eval (P.eval x))) : 
  P = Q := 
sorry

end polynomials_equality_l176_176419


namespace simplify_fraction_l176_176427

theorem simplify_fraction (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := sorry

end simplify_fraction_l176_176427


namespace average_sales_l176_176432

theorem average_sales
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 90)
  (h2 : a2 = 50)
  (h3 : a3 = 70)
  (h4 : a4 = 110)
  (h5 : a5 = 80) :
  (a1 + a2 + a3 + a4 + a5) / 5 = 80 :=
by
  sorry

end average_sales_l176_176432


namespace initial_volume_of_mixture_l176_176000

-- Define the conditions of the problem as hypotheses
variable (milk_ratio water_ratio : ℕ) (W : ℕ) (initial_mixture : ℕ)
variable (h1 : milk_ratio = 2) (h2 : water_ratio = 1)
variable (h3 : W = 60)
variable (h4 : water_ratio + milk_ratio = 3) -- The sum of the ratios used in the equation

theorem initial_volume_of_mixture : initial_mixture = 60 :=
by
  sorry

end initial_volume_of_mixture_l176_176000


namespace find_n_minus_m_l176_176819

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 25 - r^2 = 0

-- Given conditions
def circles_intersect (r : ℝ) : Prop :=
(r > 0) ∧ (∃ x y, circle1 x y ∧ circle2 x y r)

-- Prove the range of r for intersection
theorem find_n_minus_m : 
(∀ (r : ℝ), 2 ≤ r ∧ r ≤ 12 ↔ circles_intersect r) → 
12 - 2 = 10 :=
by
  sorry

end find_n_minus_m_l176_176819


namespace part1_part2_l176_176528

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi / 3)

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 : {x | f x < 1 / 4} = { x | ∃ k : ℤ, k * Real.pi + 5 * Real.pi / 12 < x ∧ x < k * Real.pi + 11 * Real.pi / 12 } :=
by
  sorry

end part1_part2_l176_176528


namespace lauren_annual_income_l176_176541

open Real

theorem lauren_annual_income (p : ℝ) (A : ℝ) (T : ℝ) :
  (T = (p + 0.45)/100 * A) →
  (T = (p/100) * 20000 + ((p + 1)/100) * 15000 + ((p + 3)/100) * (A - 35000)) →
  A = 36000 :=
by
  intros
  sorry

end lauren_annual_income_l176_176541


namespace inequation_proof_l176_176446

theorem inequation_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 1) :
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3 / 2) :=
by
  sorry

end inequation_proof_l176_176446


namespace pay_for_notebook_with_change_l176_176324

theorem pay_for_notebook_with_change : ∃ (a b : ℤ), 16 * a - 27 * b = 1 :=
by
  sorry

end pay_for_notebook_with_change_l176_176324


namespace smallest_integer_y_l176_176176

theorem smallest_integer_y (y : ℤ) (h : 7 - 3 * y < 20) : ∃ (y : ℤ), y = -4 :=
by
  sorry

end smallest_integer_y_l176_176176


namespace ratio_R_U_l176_176134

theorem ratio_R_U : 
  let spacing := 1 / 4
  let R := 3 * spacing
  let U := 6 * spacing
  R / U = 0.5 := 
by
  sorry

end ratio_R_U_l176_176134


namespace find_distance_l176_176045

variable (y : ℚ) -- The circumference of the bicycle wheel
variable (x : ℚ) -- The distance between the village and the field

-- Condition 1: The circumference of the truck's wheel is 4/3 of the bicycle's wheel
def circum_truck_eq : Prop := (4 / 3 : ℚ) * y = y

-- Condition 2: The circumference of the truck's wheel is 2 meters shorter than the tractor's track
def circum_truck_less : Prop := (4 / 3 : ℚ) * y + 2 = y + 2

-- Condition 3: Truck's wheel makes 100 fewer revolutions than the bicycle's wheel
def truck_100_fewer : Prop := x / ((4 / 3 : ℚ) * y) = (x / y) - 100

-- Condition 4: Truck's wheel makes 150 more revolutions than the tractor track
def truck_150_more : Prop := x / ((4 / 3 : ℚ) * y) = (x / ((4 / 3 : ℚ) * y + 2)) + 150

theorem find_distance (y : ℚ) (x : ℚ) :
  circum_truck_eq y →
  circum_truck_less y →
  truck_100_fewer x y →
  truck_150_more x y →
  x = 600 :=
by
  intros
  sorry

end find_distance_l176_176045


namespace nth_equation_l176_176355

theorem nth_equation (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end nth_equation_l176_176355


namespace smallest_positive_integer_23n_mod_5678_mod_11_l176_176103

theorem smallest_positive_integer_23n_mod_5678_mod_11 :
  ∃ n : ℕ, 0 < n ∧ 23 * n % 11 = 5678 % 11 ∧ ∀ m : ℕ, 0 < m ∧ 23 * m % 11 = 5678 % 11 → n ≤ m :=
by
  sorry

end smallest_positive_integer_23n_mod_5678_mod_11_l176_176103


namespace nathan_weeks_l176_176913

-- Define the conditions as per the problem
def hours_per_day_nathan : ℕ := 3
def days_per_week : ℕ := 7
def hours_per_week_nathan : ℕ := hours_per_day_nathan * days_per_week
def hours_per_day_tobias : ℕ := 5
def hours_one_week_tobias : ℕ := hours_per_day_tobias * days_per_week
def total_hours : ℕ := 77

-- The number of weeks Nathan played
def weeks_nathan (w : ℕ) : Prop :=
  hours_per_week_nathan * w + hours_one_week_tobias = total_hours

-- Prove the number of weeks Nathan played is 2
theorem nathan_weeks : ∃ w : ℕ, weeks_nathan w ∧ w = 2 :=
by
  use 2
  sorry

end nathan_weeks_l176_176913


namespace triangle_area_correct_l176_176620

noncomputable def area_of_triangle 
  (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) : ℝ :=
  let cosC := (b^2 + c^2 - a^2) / (2 * b * c)
  let sinC := Real.sqrt (1 - cosC^2)
  (1 / 2) * b * c * sinC

theorem triangle_area_correct : area_of_triangle (Real.sqrt 29) (Real.sqrt 13) (Real.sqrt 34) 
  (by rfl) (by rfl) (by rfl) = 19 / 2 :=
sorry

end triangle_area_correct_l176_176620


namespace triangle_angle_A_triangle_bc_range_l176_176968

theorem triangle_angle_A (a b c A B C : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (ha : a = b * Real.sin C + c * Real.sin B)
  (hb : b = c * Real.sin A + a * Real.sin C)
  (hc : c = a * Real.sin B + b * Real.sin A)
  (h_eq : (Real.sqrt 3) * a * Real.sin C + a * Real.cos C = c + b)
  (h_angles_sum : A + B + C = π) :
    A = π/3 := -- π/3 radians equals 60 degrees
sorry

theorem triangle_bc_range (a b c : ℝ) (h : a = Real.sqrt 3) :
  Real.sqrt 3 < b + c ∧ b + c ≤ 2 * Real.sqrt 3 := 
sorry

end triangle_angle_A_triangle_bc_range_l176_176968


namespace product_modulo_7_l176_176775

theorem product_modulo_7 : (1729 * 1865 * 1912 * 2023) % 7 = 6 :=
by
  sorry

end product_modulo_7_l176_176775


namespace probability_of_non_defective_product_l176_176323

-- Define the probability of producing a grade B product
def P_B : ℝ := 0.03

-- Define the probability of producing a grade C product
def P_C : ℝ := 0.01

-- Define the probability of producing a non-defective product (grade A)
def P_A : ℝ := 1 - P_B - P_C

-- The theorem to prove: The probability of producing a non-defective product is 0.96
theorem probability_of_non_defective_product : P_A = 0.96 := by
  -- Insert proof here
  sorry

end probability_of_non_defective_product_l176_176323


namespace rectangle_lengths_l176_176550

theorem rectangle_lengths (side_length : ℝ) (width1 width2: ℝ) (length1 length2 : ℝ) 
  (h1 : side_length = 6) 
  (h2 : width1 = 4) 
  (h3 : width2 = 3)
  (h_area_square : side_length * side_length = 36)
  (h_area_rectangle1 : width1 * length1 = side_length * side_length)
  (h_area_rectangle2 : width2 * length2 = (1 / 2) * (side_length * side_length)) :
  length1 = 9 ∧ length2 = 6 :=
by
  sorry

end rectangle_lengths_l176_176550


namespace koschei_coins_l176_176615

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l176_176615


namespace area_of_triangle_PQR_l176_176795

noncomputable def point := ℝ × ℝ

def P : point := (1, 1)
def Q : point := (4, 1)
def R : point := (3, 4)

def triangle_area (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)

theorem area_of_triangle_PQR :
  triangle_area P Q R = 9 / 2 :=
by
  sorry

end area_of_triangle_PQR_l176_176795


namespace peyton_manning_total_yards_l176_176463

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end peyton_manning_total_yards_l176_176463


namespace puppy_cost_l176_176456

variable (P : ℕ)  -- Cost of one puppy

theorem puppy_cost (P : ℕ) (kittens : ℕ) (cost_kitten : ℕ) (total_value : ℕ) :
  kittens = 4 → cost_kitten = 15 → total_value = 100 → 
  2 * P + kittens * cost_kitten = total_value → P = 20 :=
by sorry

end puppy_cost_l176_176456


namespace minimum_magnitude_l176_176467

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem minimum_magnitude (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3 * Complex.I) = 15) :
  smallest_magnitude_z z = (768 / 265 : ℝ) :=
by
  sorry

end minimum_magnitude_l176_176467


namespace negation_of_implication_l176_176468

theorem negation_of_implication (x : ℝ) :
  (¬ (x = 0 ∨ x = 1) → x^2 - x ≠ 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) :=
by sorry

end negation_of_implication_l176_176468


namespace best_fitting_model_l176_176946

/-- Four models with different coefficients of determination -/
def model1_R2 : ℝ := 0.98
def model2_R2 : ℝ := 0.80
def model3_R2 : ℝ := 0.50
def model4_R2 : ℝ := 0.25

/-- Prove that Model 1 has the best fitting effect among the given models -/
theorem best_fitting_model :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by {sorry}

end best_fitting_model_l176_176946


namespace topsoil_cost_l176_176860

theorem topsoil_cost (cost_per_cubic_foot : ℝ) (cubic_yards : ℝ) (conversion_factor : ℝ) : 
  cubic_yards = 8 →
  cost_per_cubic_foot = 7 →
  conversion_factor = 27 →
  ∃ total_cost : ℝ, total_cost = 1512 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l176_176860


namespace sequence_is_arithmetic_not_geometric_l176_176661

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 6 / Real.log 2
noncomputable def c := Real.log 12 / Real.log 2

theorem sequence_is_arithmetic_not_geometric : 
  (b - a = c - b) ∧ (b / a ≠ c / b) := 
by
  sorry

end sequence_is_arithmetic_not_geometric_l176_176661


namespace contractor_absent_days_l176_176192

variable (x y : ℝ) -- x for the number of days worked, y for the number of days absent

-- Conditions
def eng_days := x + y = 30
def total_money := 25 * x - 7.5 * y = 425

-- Theorem
theorem contractor_absent_days (x y : ℝ) (h1 : eng_days x y) (h2 : total_money x y) : y = 10 := 
sorry

end contractor_absent_days_l176_176192


namespace back_seat_people_l176_176099

/-- Define the number of seats on the left side of the bus --/
def left_side_seats : ℕ := 15

/-- Define the number of seats on the right side of the bus (3 fewer because of the rear exit door) --/
def right_side_seats : ℕ := left_side_seats - 3

/-- Define the number of people each seat can hold --/
def people_per_seat : ℕ := 3

/-- Define the total capacity of the bus --/
def total_capacity : ℕ := 90

/-- Define the total number of people that can sit on the regular seats (left and right sides) --/
def regular_seats_people := (left_side_seats + right_side_seats) * people_per_seat

/-- Theorem stating the number of people that can sit at the back seat --/
theorem back_seat_people : (total_capacity - regular_seats_people) = 9 := by
  sorry

end back_seat_people_l176_176099


namespace area_of_rectangular_field_l176_176128

theorem area_of_rectangular_field (L W A : ℕ) (h1 : L = 10) (h2 : 2 * W + L = 130) :
  A = 600 :=
by
  -- Proof will go here
  sorry

end area_of_rectangular_field_l176_176128


namespace black_pens_removed_l176_176580

theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
    (blue_removed : ℕ) (pens_left : ℕ)
    (h_initial_pens : initial_blue = 9 ∧ initial_black = 21 ∧ initial_red = 6)
    (h_blue_removed : blue_removed = 4)
    (h_pens_left : pens_left = 25) :
    initial_blue + initial_black + initial_red - blue_removed - (initial_blue + initial_black + initial_red - blue_removed - pens_left) = 7 :=
by
  rcases h_initial_pens with ⟨h_ib, h_ibl, h_ir⟩
  simp [h_ib, h_ibl, h_ir, h_blue_removed, h_pens_left]
  sorry

end black_pens_removed_l176_176580


namespace factor_polynomial_l176_176668

theorem factor_polynomial (n : ℕ) (hn : 2 ≤ n) 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℤ, n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧ 
  a = (-(2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n)))) ^ (2 * n / (2 * n - 1)) ∧ 
  b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n))) ^ (2 / (2 * n - 1)) := sorry

end factor_polynomial_l176_176668


namespace age_in_1900_l176_176699

theorem age_in_1900 
  (x y : ℕ)
  (H1 : y = 29 * x)
  (H2 : 1901 ≤ y + x ∧ y + x ≤ 1930) :
  1900 - y = 44 := 
sorry

end age_in_1900_l176_176699


namespace log_relationships_l176_176302

theorem log_relationships (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1 / Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1 / Real.sqrt (6 / 10)) ∨ d = c^(Real.sqrt (6 / 10)) :=
sorry

end log_relationships_l176_176302


namespace difference_of_numbers_is_21938_l176_176072

theorem difference_of_numbers_is_21938 
  (x y : ℕ) 
  (h1 : x + y = 26832) 
  (h2 : x % 10 = 0) 
  (h3 : y = x / 10 + 4) 
  : x - y = 21938 :=
sorry

end difference_of_numbers_is_21938_l176_176072


namespace thirty_five_million_in_scientific_notation_l176_176650

def million := 10^6

def sales_revenue (x : ℝ) := x * million

theorem thirty_five_million_in_scientific_notation :
  sales_revenue 35 = 3.5 * 10^7 :=
by
  sorry

end thirty_five_million_in_scientific_notation_l176_176650


namespace max_positive_integer_value_of_n_l176_176822

-- Define the arithmetic sequence with common difference d and first term a₁.
variable {d a₁ : ℝ}

-- The quadratic inequality condition which provides the solution set [0,9].
def inequality_condition (d a₁ : ℝ) : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 9) → d * x^2 + 2 * a₁ * x ≥ 0

-- Maximum integer n such that the sum of the first n terms of the sequence is maximum.
noncomputable def max_n (d a₁ : ℝ) : ℕ :=
  if d < 0 then 5 else 0

-- Statement to be proved.
theorem max_positive_integer_value_of_n (d a₁ : ℝ) 
  (h : inequality_condition d a₁) : max_n d a₁ = 5 :=
sorry

end max_positive_integer_value_of_n_l176_176822


namespace simplest_common_denominator_l176_176481

variable (m n a : ℕ)

theorem simplest_common_denominator (h₁ : m > 0) (h₂ : n > 0) (h₃ : a > 0) :
  ∃ l : ℕ, l = 2 * a^2 := 
sorry

end simplest_common_denominator_l176_176481


namespace intersect_at_four_points_l176_176703

theorem intersect_at_four_points (a : ℝ) : 
  (∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = a^2) ∧ (p.2 = p.1^2 - a - 1) ∧ 
                 ∃ q : ℝ × ℝ, (q.1 ≠ p.1 ∧ q.2 ≠ p.2) ∧ (q.1^2 + q.2^2 = a^2) ∧ (q.2 = q.1^2 - a - 1) ∧ 
                 ∃ r : ℝ × ℝ, (r.1 ≠ p.1 ∧ r.1 ≠ q.1 ∧ r.2 ≠ p.2 ∧ r.2 ≠ q.2) ∧ (r.1^2 + r.2^2 = a^2) ∧ (r.2 = r.1^2 - a - 1) ∧
                 ∃ s : ℝ × ℝ, (s.1 ≠ p.1 ∧ s.1 ≠ q.1 ∧ s.1 ≠ r.1 ∧ s.2 ≠ p.2 ∧ s.2 ≠ q.2 ∧ s.2 ≠ r.2) ∧ (s.1^2 + s.2^2 = a^2) ∧ (s.2 = s.1^2 - a - 1))
  ↔ a > -1/2 := 
by 
  sorry

end intersect_at_four_points_l176_176703


namespace number_of_insects_l176_176949

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 30) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 5 :=
by
  sorry

end number_of_insects_l176_176949


namespace daily_average_books_l176_176475

theorem daily_average_books (x : ℝ) (h1 : 4 * x + 1.4 * x = 216) : x = 40 :=
by 
  sorry

end daily_average_books_l176_176475


namespace circle_area_solution_l176_176941

def circle_area_problem : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 6 * x - 8 * y - 12 = 0 -> ∃ (A : ℝ), A = 37 * Real.pi

theorem circle_area_solution : circle_area_problem :=
by
  sorry

end circle_area_solution_l176_176941


namespace analytical_expression_maximum_value_l176_176424

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1

theorem analytical_expression (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, abs (x - (x + (Real.pi / (2 * ω)))) = Real.pi / 2) : 
  f x 2 = 2 * Real.sin (2 * x - Real.pi / 6) + 1 :=
sorry

theorem maximum_value (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  2 * Real.sin (2 * x - Real.pi / 6) + 1 ≤ 3 :=
sorry

end analytical_expression_maximum_value_l176_176424


namespace checkerboard_7_strips_l176_176447

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end checkerboard_7_strips_l176_176447


namespace rotated_D_coords_l176_176366

-- Definitions of the points used in the problem
def point (x y : ℤ) : ℤ × ℤ := (x, y)

-- Definitions of the vertices of the triangle DEF
def D : ℤ × ℤ := point 2 (-3)
def E : ℤ × ℤ := point 2 0
def F : ℤ × ℤ := point 5 (-3)

-- Definition of the rotation center
def center : ℤ × ℤ := point 3 (-2)

-- Function to rotate a point (x, y) by 180 degrees around (h, k)
def rotate_180 (p c : ℤ × ℤ) : ℤ × ℤ := 
  let (x, y) := p
  let (h, k) := c
  (2 * h - x, 2 * k - y)

-- Statement to prove the required coordinates after rotation
theorem rotated_D_coords : rotate_180 D center = point 4 (-1) :=
  sorry

end rotated_D_coords_l176_176366


namespace find_sum_of_integers_l176_176391

theorem find_sum_of_integers (w x y z : ℤ)
  (h1 : w - x + y = 7)
  (h2 : x - y + z = 8)
  (h3 : y - z + w = 4)
  (h4 : z - w + x = 3) : w + x + y + z = 11 :=
by
  sorry

end find_sum_of_integers_l176_176391


namespace percentage_paid_l176_176732

theorem percentage_paid (X Y : ℝ) (h_sum : X + Y = 572) (h_Y : Y = 260) : (X / Y) * 100 = 120 :=
by
  -- We'll prove this result by using the conditions and solving for X.
  sorry

end percentage_paid_l176_176732


namespace secretary_work_hours_l176_176688

theorem secretary_work_hours
  (x : ℕ)
  (h_ratio : 2 * x + 3 * x + 5 * x = 110) :
  5 * x = 55 := 
by
  sorry

end secretary_work_hours_l176_176688


namespace perimeter_difference_l176_176943

-- Definitions for the conditions
def num_stakes_sheep : ℕ := 96
def interval_sheep : ℕ := 10
def num_stakes_horse : ℕ := 82
def interval_horse : ℕ := 20

-- Definition for the perimeters
def perimeter_sheep : ℕ := num_stakes_sheep * interval_sheep
def perimeter_horse : ℕ := num_stakes_horse * interval_horse

-- Definition for the target difference
def target_difference : ℕ := 680

-- The theorem stating the proof problem
theorem perimeter_difference : perimeter_horse - perimeter_sheep = target_difference := by
  sorry

end perimeter_difference_l176_176943


namespace find_a_l176_176212

theorem find_a (a b c : ℕ) (h1 : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ (2 * a - 3)) = (2 ^ 7) * (3 ^ b)) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : a = 7 :=
by
  sorry

end find_a_l176_176212


namespace simplify_complex_expression_l176_176700

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) - 2 * i * (3 - 4 * i) = 20 - 20 * i := 
by
  sorry

end simplify_complex_expression_l176_176700


namespace total_tickets_sold_l176_176558

theorem total_tickets_sold (A C : ℕ) (hC : C = 16) (h1 : 3 * C = 48) (h2 : 5 * A + 3 * C = 178) : 
  A + C = 42 :=
by
  sorry

end total_tickets_sold_l176_176558


namespace ferry_max_weight_capacity_l176_176999

def automobile_max_weight : ℝ := 3200
def automobile_count : ℝ := 62.5
def pounds_to_tons : ℝ := 2000

theorem ferry_max_weight_capacity : 
  (automobile_max_weight * automobile_count) / pounds_to_tons = 100 := 
by 
  sorry

end ferry_max_weight_capacity_l176_176999


namespace solution_set_of_quadratic_inequality_l176_176713

variable {a x : ℝ} (h_neg : a < 0)

theorem solution_set_of_quadratic_inequality :
  (a * x^2 - (a + 2) * x + 2) ≥ 0 ↔ (x ∈ Set.Icc (2 / a) 1) :=
by
  sorry

end solution_set_of_quadratic_inequality_l176_176713


namespace original_cost_of_car_l176_176641

-- Conditions
variables (C : ℝ)
variables (spent_on_repairs : ℝ := 8000)
variables (selling_price : ℝ := 68400)
variables (profit_percent : ℝ := 54.054054054054056)

-- Statement to be proved
theorem original_cost_of_car :
  C + spent_on_repairs = selling_price - (profit_percent / 100) * C :=
sorry

end original_cost_of_car_l176_176641


namespace combined_CD_length_l176_176394

def CD1 := 1.5
def CD2 := 1.5
def CD3 := 2 * CD1

theorem combined_CD_length : CD1 + CD2 + CD3 = 6 := 
by
  sorry

end combined_CD_length_l176_176394


namespace imaginary_unit_calculation_l176_176429

theorem imaginary_unit_calculation (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := 
by
  sorry

end imaginary_unit_calculation_l176_176429


namespace arccos_one_eq_zero_l176_176803

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l176_176803


namespace more_bottles_of_regular_soda_l176_176386

theorem more_bottles_of_regular_soda (reg_soda diet_soda : ℕ) (h1 : reg_soda = 79) (h2 : diet_soda = 53) :
  reg_soda - diet_soda = 26 :=
by
  sorry

end more_bottles_of_regular_soda_l176_176386


namespace line_always_passes_fixed_point_l176_176437

theorem line_always_passes_fixed_point : ∀ (m : ℝ), (m-1)*(-2) - 1 + (2*m-1) = 0 :=
by
  intro m
  -- Calculations can be done here to prove the theorem straightforwardly.
  sorry

end line_always_passes_fixed_point_l176_176437


namespace tan_theta_minus_pi_over_4_l176_176038

theorem tan_theta_minus_pi_over_4 
  (θ : Real) (h1 : π / 2 < θ ∧ θ < 2 * π)
  (h2 : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 := 
  sorry

end tan_theta_minus_pi_over_4_l176_176038


namespace difference_of_sum_l176_176990

theorem difference_of_sum (a b c : ℤ) (h1 : a = 11) (h2 : b = 13) (h3 : c = 15) :
  (b + c) - a = 17 := by
  sorry

end difference_of_sum_l176_176990


namespace greatest_integer_value_l176_176375

theorem greatest_integer_value (x : ℤ) (h : ∃ x : ℤ, x = 29 ∧ ∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*(x+6) + 26)) :
  (∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*k + 26) → x = 29) :=
by
  sorry

end greatest_integer_value_l176_176375


namespace greatest_possible_value_of_n_greatest_possible_value_of_10_l176_176531

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 :=
by
  sorry

theorem greatest_possible_value_of_10 (n : ℤ) (h : 101 * n^2 ≤ 12100) : n = 10 → n = 10 :=
by
  sorry

end greatest_possible_value_of_n_greatest_possible_value_of_10_l176_176531


namespace liz_car_percentage_sale_l176_176901

theorem liz_car_percentage_sale (P : ℝ) (h1 : 30000 = P - 2500) (h2 : 26000 = P * (80 / 100)) : 80 = 80 :=
by 
  sorry

end liz_car_percentage_sale_l176_176901


namespace speed_ratio_is_2_l176_176697

def distance_to_work : ℝ := 20
def total_hours_on_road : ℝ := 6
def speed_back_home : ℝ := 10

theorem speed_ratio_is_2 :
  (∃ v : ℝ, (20 / v) + (20 / 10) = 6) → (10 = 2 * v) :=
by sorry

end speed_ratio_is_2_l176_176697


namespace neg_sin_prop_iff_l176_176202

theorem neg_sin_prop_iff :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by sorry

end neg_sin_prop_iff_l176_176202


namespace find_a_l176_176654

theorem find_a (f : ℕ → ℕ) (a : ℕ) 
  (h1 : ∀ x : ℕ, f (x + 1) = x) 
  (h2 : f a = 8) : a = 9 :=
sorry

end find_a_l176_176654


namespace earnings_per_hour_l176_176538

-- Define the conditions
def widgetsProduced : Nat := 750
def hoursWorked : Nat := 40
def totalEarnings : ℝ := 620
def earningsPerWidget : ℝ := 0.16

-- Define the proof goal
theorem earnings_per_hour :
  ∃ H : ℝ, (hoursWorked * H + widgetsProduced * earningsPerWidget = totalEarnings) ∧ H = 12.5 :=
by
  sorry

end earnings_per_hour_l176_176538


namespace ratio_of_perimeters_of_squares_l176_176290

theorem ratio_of_perimeters_of_squares (d1 d11 : ℝ) (s1 s11 : ℝ) (P1 P11 : ℝ) 
  (h1 : d11 = 11 * d1)
  (h2 : d1 = s1 * Real.sqrt 2)
  (h3 : d11 = s11 * Real.sqrt 2) :
  P11 / P1 = 11 :=
by
  sorry

end ratio_of_perimeters_of_squares_l176_176290


namespace julia_played_tag_l176_176839

/-
Problem:
Let m be the number of kids Julia played with on Monday.
Let t be the number of kids Julia played with on Tuesday.
m = 24
m = t + 18
Show that t = 6
-/

theorem julia_played_tag (m t : ℕ) (h1 : m = 24) (h2 : m = t + 18) : t = 6 :=
by
  sorry

end julia_played_tag_l176_176839


namespace largest_fraction_among_given_l176_176477

theorem largest_fraction_among_given (f1 f2 f3 f4 f5 : ℚ)
  (h1 : f1 = 2/5) 
  (h2 : f2 = 4/9) 
  (h3 : f3 = 7/15) 
  (h4 : f4 = 11/18) 
  (h5 : f5 = 16/35) 
  : f1 < f4 ∧ f2 < f4 ∧ f3 < f4 ∧ f5 < f4 :=
by
  sorry

end largest_fraction_among_given_l176_176477


namespace cosine_seventh_power_expansion_l176_176867

theorem cosine_seventh_power_expansion :
  let b1 := (35 : ℝ) / 64
  let b2 := (0 : ℝ)
  let b3 := (21 : ℝ) / 64
  let b4 := (0 : ℝ)
  let b5 := (7 : ℝ) / 64
  let b6 := (0 : ℝ)
  let b7 := (1 : ℝ) / 64
  b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2 = 1687 / 4096 := by
  sorry

end cosine_seventh_power_expansion_l176_176867


namespace calculate_area_of_square_field_l176_176841

def area_of_square_field (t: ℕ) (v: ℕ) (d: ℕ) (s: ℕ) (a: ℕ) : Prop :=
  t = 10 ∧ v = 16 ∧ d = v * t ∧ 4 * s = d ∧ a = s^2

theorem calculate_area_of_square_field (t v d s a : ℕ) 
  (h1: t = 10) (h2: v = 16) (h3: d = v * t) (h4: 4 * s = d) 
  (h5: a = s^2) : a = 1600 := by
  sorry

end calculate_area_of_square_field_l176_176841


namespace smallest_k_l176_176127

theorem smallest_k (p : ℕ) (hp : p = 997) : 
  ∃ k : ℕ, (p^2 - k) % 10 = 0 ∧ k = 9 :=
by
  sorry

end smallest_k_l176_176127


namespace airplane_cost_correct_l176_176871

-- Define the conditions
def initial_amount : ℝ := 5.00
def change_received : ℝ := 0.72

-- Define the cost calculation
def airplane_cost (initial : ℝ) (change : ℝ) : ℝ := initial - change

-- Prove that the airplane cost is $4.28 given the conditions
theorem airplane_cost_correct : airplane_cost initial_amount change_received = 4.28 :=
by
  -- The actual proof goes here
  sorry

end airplane_cost_correct_l176_176871


namespace g_g_g_3_l176_176122

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_g_g_3 : g (g (g 3)) = 241 := by
  sorry

end g_g_g_3_l176_176122


namespace odd_function_evaluation_l176_176230

theorem odd_function_evaluation
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, x ≤ 0 → f x = 2 * x^2 - x) :
  f 1 = -3 :=
by {
  sorry
}

end odd_function_evaluation_l176_176230


namespace volume_of_regular_tetrahedron_l176_176042

noncomputable def volume_of_tetrahedron (a H : ℝ) : ℝ :=
  (a^2 * H) / (6 * Real.sqrt 2)

theorem volume_of_regular_tetrahedron
  (d_face : ℝ)
  (d_edge : ℝ)
  (h : Real.sqrt 14 = d_edge)
  (h1 : 2 = d_face)
  (volume_approx : ℝ) :
  ∃ a H, (d_face = Real.sqrt ((H / 2)^2 + (a * Real.sqrt 3 / 6)^2) ∧ 
          d_edge = Real.sqrt ((H / 2)^2 + (a / (2 * Real.sqrt 3))^2) ∧ 
          Real.sqrt (volume_of_tetrahedron a H) = 533.38) :=
  sorry

end volume_of_regular_tetrahedron_l176_176042


namespace Shiela_stars_per_bottle_l176_176133

theorem Shiela_stars_per_bottle (total_stars : ℕ) (total_classmates : ℕ) (h1 : total_stars = 45) (h2 : total_classmates = 9) :
  total_stars / total_classmates = 5 := 
by 
  sorry

end Shiela_stars_per_bottle_l176_176133


namespace eldorado_license_plates_count_l176_176937

theorem eldorado_license_plates_count:
  let letters := 26
  let digits := 10
  let total := (letters ^ 3) * (digits ^ 4)
  total = 175760000 :=
by
  sorry

end eldorado_license_plates_count_l176_176937


namespace oil_needed_to_half_fill_tanker_l176_176143

theorem oil_needed_to_half_fill_tanker :
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  let current_tanker_oil := initial_tanker_oil + poured_oil
  let half_tanker_capacity := initial_tanker_capacity / 2
  let needed_oil := half_tanker_capacity - current_tanker_oil
  needed_oil = 4000 :=
by
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  have h1 : poured_oil = 3000 := by sorry
  let current_tanker_oil := initial_tanker_oil + poured_oil
  have h2 : current_tanker_oil = 6000 := by sorry
  let half_tanker_capacity := initial_tanker_capacity / 2
  have h3 : half_tanker_capacity = 10000 := by sorry
  let needed_oil := half_tanker_capacity - current_tanker_oil
  have h4 : needed_oil = 4000 := by sorry
  exact h4

end oil_needed_to_half_fill_tanker_l176_176143


namespace subset1_squares_equals_product_subset2_squares_equals_product_l176_176594

theorem subset1_squares_equals_product :
  (1^2 + 3^2 + 4^2 + 9^2 + 107^2 = 1 * 3 * 4 * 9 * 107) :=
sorry

theorem subset2_squares_equals_product :
  (3^2 + 4^2 + 9^2 + 107^2 + 11555^2 = 3 * 4 * 9 * 107 * 11555) :=
sorry

end subset1_squares_equals_product_subset2_squares_equals_product_l176_176594


namespace arithmetic_geometric_sequence_S6_l176_176441

noncomputable def S_6 (a : Nat) (q : Nat) : Nat :=
  (q ^ 6 - 1) / (q - 1)

theorem arithmetic_geometric_sequence_S6 (a : Nat) (q : Nat) (h1 : a * q ^ 1 = 2) (h2 : a * q ^ 3 = 8) (hq : q > 0) : S_6 a q = 63 :=
by
  sorry

end arithmetic_geometric_sequence_S6_l176_176441


namespace first_hour_rain_l176_176318

variable (x : ℝ)
variable (rain_1st_hour : ℝ) (rain_2nd_hour : ℝ)
variable (total_rain : ℝ)

-- Define the conditions
def condition_1 (x rain_2nd_hour : ℝ) : Prop :=
  rain_2nd_hour = 2 * x + 7

def condition_2 (x rain_2nd_hour total_rain : ℝ) : Prop :=
  x + rain_2nd_hour = total_rain

-- Prove the amount of rain in the first hour
theorem first_hour_rain (h1 : condition_1 x rain_2nd_hour)
                         (h2 : condition_2 x rain_2nd_hour total_rain)
                         (total_rain_is_22 : total_rain = 22) :
  x = 5 :=
by
  -- Proof steps go here
  sorry

end first_hour_rain_l176_176318


namespace no_integer_solutions_l176_176329

theorem no_integer_solutions (x y z : ℤ) (h : 2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) (hx : x ≠ 0) : false :=
sorry

end no_integer_solutions_l176_176329


namespace degree_odd_of_polynomials_l176_176651

theorem degree_odd_of_polynomials 
  (d : ℕ) 
  (P Q : Polynomial ℝ) 
  (hP_deg : P.degree = d) 
  (h_eq : P^2 + 1 = (X^2 + 1) * Q^2) 
  : Odd d :=
sorry

end degree_odd_of_polynomials_l176_176651


namespace multiple_of_669_l176_176270

theorem multiple_of_669 (k : ℕ) (h : ∃ a : ℤ, 2007 ∣ (a + k : ℤ)^3 - a^3) : 669 ∣ k :=
sorry

end multiple_of_669_l176_176270


namespace rationalize_denominator_l176_176543

-- Problem statement
theorem rationalize_denominator :
  1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end rationalize_denominator_l176_176543


namespace min_sum_of_factors_240_l176_176965

theorem min_sum_of_factors_240 :
  ∃ a b : ℕ, a * b = 240 ∧ (∀ a' b' : ℕ, a' * b' = 240 → a + b ≤ a' + b') ∧ a + b = 31 :=
sorry

end min_sum_of_factors_240_l176_176965


namespace totalPizzaEaten_l176_176261

-- Define the conditions
def rachelAte : ℕ := 598
def bellaAte : ℕ := 354

-- State the theorem
theorem totalPizzaEaten : rachelAte + bellaAte = 952 :=
by
  -- Proof omitted
  sorry

end totalPizzaEaten_l176_176261


namespace hcf_lcm_product_l176_176539

theorem hcf_lcm_product (a b : ℕ) (H : a * b = 45276) (L : Nat.lcm a b = 2058) : Nat.gcd a b = 22 :=
by 
  -- The proof steps go here
  sorry

end hcf_lcm_product_l176_176539


namespace sqrt_range_l176_176369

theorem sqrt_range (x : ℝ) (h : 5 - x ≥ 0) : x ≤ 5 :=
sorry

end sqrt_range_l176_176369


namespace grazing_months_of_B_l176_176478

variable (A_cows A_months C_cows C_months D_cows D_months A_rent total_rent : ℕ)
variable (B_cows x : ℕ)

theorem grazing_months_of_B
  (hA_cows : A_cows = 24)
  (hA_months : A_months = 3)
  (hC_cows : C_cows = 35)
  (hC_months : C_months = 4)
  (hD_cows : D_cows = 21)
  (hD_months : D_months = 3)
  (hA_rent : A_rent = 1440)
  (htotal_rent : total_rent = 6500)
  (hB_cows : B_cows = 10) :
  x = 5 := 
sorry

end grazing_months_of_B_l176_176478


namespace find_point_P_l176_176195

/-- 
Given two points A and B, find the coordinates of point P that lies on the line AB
and satisfies that the distance from A to P is half the vector from A to B.
-/
theorem find_point_P 
  (A B : ℝ × ℝ) 
  (hA : A = (3, -4)) 
  (hB : B = (-9, 2)) 
  (P : ℝ × ℝ) 
  (hP : P.1 - A.1 = (1/2) * (B.1 - A.1) ∧ P.2 - A.2 = (1/2) * (B.2 - A.2)) : 
  P = (-3, -1) := 
sorry

end find_point_P_l176_176195


namespace sequence_periodicity_l176_176973

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : a 1 = 6 / 7)
  (h_rec : ∀ n, 0 ≤ a n ∧ a n < 1 → a (n+1) = if a n ≤ 1/2 then 2 * a n else 2 * a n - 1) :
  a 2017 = 6 / 7 :=
  sorry

end sequence_periodicity_l176_176973


namespace oscar_cookie_baking_time_l176_176787

theorem oscar_cookie_baking_time : 
  (1 / 5) + (1 / 6) + (1 / o) - (1 / 4) = (1 / 8) → o = 120 := by
  sorry

end oscar_cookie_baking_time_l176_176787


namespace ratio_of_height_and_radius_l176_176785

theorem ratio_of_height_and_radius 
  (h r : ℝ) 
  (V_X V_Y : ℝ)
  (hY rY : ℝ)
  (k : ℝ)
  (h_def : V_X = π * r^2 * h)
  (hY_def : hY = k * h)
  (rY_def : rY = k * r)
  (half_filled_VY : V_Y = 1/2 * π * rY^2 * hY)
  (V_X_value : V_X = 2)
  (V_Y_value : V_Y = 64):
  k = 4 :=
by
  sorry

end ratio_of_height_and_radius_l176_176785


namespace central_angle_unchanged_l176_176015

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end central_angle_unchanged_l176_176015


namespace pow_divisible_by_13_l176_176296

theorem pow_divisible_by_13 (n : ℕ) (h : 0 < n) : (4^(2*n+1) + 3^(n+2)) % 13 = 0 :=
sorry

end pow_divisible_by_13_l176_176296


namespace proposition_negation_l176_176845

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end proposition_negation_l176_176845


namespace min_value_b_over_a_l176_176722

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + (Real.exp 1 - a) * x - b

theorem min_value_b_over_a 
  (a b : ℝ)
  (h_cond : ∀ x > 0, f x a b ≤ 0)
  (h_b : b = -1 - Real.log (a - Real.exp 1)) 
  (h_a_gt_e : a > Real.exp 1) :
  ∃ (x : ℝ), x = 2 * Real.exp 1 ∧ (b / a) = - (1 / Real.exp 1) := 
sorry

end min_value_b_over_a_l176_176722


namespace inequality_holds_for_all_x_l176_176905

theorem inequality_holds_for_all_x : 
  ∀ (a : ℝ), (∀ (x : ℝ), |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := 
sorry

end inequality_holds_for_all_x_l176_176905


namespace knitting_time_is_correct_l176_176379

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end knitting_time_is_correct_l176_176379


namespace max_value_when_a_zero_range_of_a_for_one_zero_l176_176597

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_when_a_zero :
  ∃ x : ℝ, x > 0 ∧ f 0 x = -1 :=
by sorry

theorem range_of_a_for_one_zero :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a > 0 :=
by sorry

end max_value_when_a_zero_range_of_a_for_one_zero_l176_176597


namespace find_digit_A_l176_176536

theorem find_digit_A : ∃ A : ℕ, A < 10 ∧ (200 + 10 * A + 4) % 13 = 0 ∧ A = 7 :=
by
  sorry

end find_digit_A_l176_176536


namespace isabel_money_left_l176_176924

theorem isabel_money_left (initial_amount : ℕ) (half_toy_expense half_book_expense money_left : ℕ) :
  initial_amount = 204 →
  half_toy_expense = initial_amount / 2 →
  half_book_expense = (initial_amount - half_toy_expense) / 2 →
  money_left = initial_amount - half_toy_expense - half_book_expense →
  money_left = 51 :=
by
  intros h1 h2 h3 h4
  sorry

end isabel_money_left_l176_176924


namespace find_a_value_l176_176987

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 := 
by 
  sorry -- Placeholder for the proof

end find_a_value_l176_176987


namespace length_of_second_dimension_l176_176646

def volume_of_box (w : ℝ) : ℝ :=
  (w - 16) * (46 - 16) * 8

theorem length_of_second_dimension (w : ℝ) (h_volume : volume_of_box w = 4800) : w = 36 :=
by
  sorry

end length_of_second_dimension_l176_176646


namespace total_amount_paid_is_correct_l176_176948

-- Definitions for the conditions
def original_price : ℝ := 150
def sale_discount : ℝ := 0.30
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

-- Calculation
def final_amount : ℝ :=
  let discounted_price := original_price * (1 - sale_discount)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price_after_tax := price_after_coupon * (1 + sales_tax)
  final_price_after_tax

-- Statement to prove
theorem total_amount_paid_is_correct : final_amount = 104.50 := by
  sorry

end total_amount_paid_is_correct_l176_176948


namespace quotient_transformation_l176_176663

theorem quotient_transformation (A B : ℕ) (h1 : B ≠ 0) (h2 : (A : ℝ) / B = 0.514) :
  ((10 * A : ℝ) / (B / 100)) = 514 :=
by
  -- skipping the proof
  sorry

end quotient_transformation_l176_176663


namespace circle_radius_k_l176_176247

theorem circle_radius_k (k : ℝ) : (∃ x y : ℝ, (x^2 + 14*x + y^2 + 8*y - k = 0) ∧ ((x + 7)^2 + (y + 4)^2 = 100)) → k = 35 :=
by
  sorry

end circle_radius_k_l176_176247


namespace radius_of_small_semicircle_l176_176549

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l176_176549


namespace sum_of_c_n_l176_176957

-- Define the sequence {b_n}
def b : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * b n + 3

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := (a n) / (b n + 3)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => c i)

-- Theorem to prove
theorem sum_of_c_n : ∀ (n : ℕ), T n = (3 / 2 : ℚ) - ((2 * n + 3) / 2^(n + 1)) :=
by
  sorry

end sum_of_c_n_l176_176957


namespace minimum_value_of_f_l176_176204

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem minimum_value_of_f :
  ∃ x : ℝ, f x = -(4 / 3) :=
by
  use 2
  have hf : f 2 = -(4 / 3) := by
    sorry
  exact hf

end minimum_value_of_f_l176_176204


namespace find_four_consecutive_odd_numbers_l176_176361

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end find_four_consecutive_odd_numbers_l176_176361


namespace age_of_B_l176_176188

variables (A B C : ℕ)

theorem age_of_B (h1 : (A + B + C) / 3 = 25) (h2 : (A + C) / 2 = 29) : B = 17 := 
by
  -- Skipping the proof steps
  sorry

end age_of_B_l176_176188


namespace best_fit_model_l176_176286

-- Definition of the given R^2 values for different models
def R2_A : ℝ := 0.62
def R2_B : ℝ := 0.63
def R2_C : ℝ := 0.68
def R2_D : ℝ := 0.65

-- Theorem statement that model with R2_C has the best fitting effect
theorem best_fit_model : R2_C = max R2_A (max R2_B (max R2_C R2_D)) :=
by
  sorry -- Proof is not required

end best_fit_model_l176_176286


namespace maximum_value_a1_l176_176339

noncomputable def max_possible_value (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : ℝ :=
  16

theorem maximum_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : a 1 ≤ max_possible_value a h1 h2 h3 :=
  sorry

end maximum_value_a1_l176_176339


namespace options_not_equal_l176_176556

theorem options_not_equal (a b c d e : ℚ)
  (ha : a = 14 / 10)
  (hb : b = 1 + 2 / 5)
  (hc : c = 1 + 7 / 25)
  (hd : d = 1 + 2 / 10)
  (he : e = 1 + 14 / 70) :
  a = 7 / 5 ∧ b = 7 / 5 ∧ c ≠ 7 / 5 ∧ d ≠ 7 / 5 ∧ e ≠ 7 / 5 :=
by sorry

end options_not_equal_l176_176556


namespace johnny_yellow_picks_l176_176095

variable (total_picks red_picks blue_picks yellow_picks : ℕ)

theorem johnny_yellow_picks
    (h_total_picks : total_picks = 3 * blue_picks)
    (h_half_red_picks : red_picks = total_picks / 2)
    (h_blue_picks : blue_picks = 12)
    (h_pick_sum : total_picks = red_picks + blue_picks + yellow_picks) :
    yellow_picks = 6 := by
  sorry

end johnny_yellow_picks_l176_176095


namespace f_at_one_is_zero_f_is_increasing_range_of_x_l176_176832

open Function

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h1 : ∀ x > 1, f x > 0)
variable (h2 : ∀ x y, f (x * y) = f x + f y)

-- Problem Statements
theorem f_at_one_is_zero : f 1 = 0 := 
sorry

theorem f_is_increasing (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h : x₁ > x₂) : 
  f x₁ > f x₂ := 
sorry

theorem range_of_x (f3_eq_1 : f 3 = 1) (x : ℝ) (h3 : x ≥ 1 + Real.sqrt 10) : 
  f x - f (1 / (x - 2)) ≥ 2 := 
sorry

end f_at_one_is_zero_f_is_increasing_range_of_x_l176_176832


namespace find_f_neg3_l176_176317

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then x * (1 - x) else -x * (1 + x)

theorem find_f_neg3 :
  is_odd_function f →
  (∀ x, x > 0 → f x = x * (1 - x)) →
  f (-3) = 6 :=
by
  intros h_odd h_condition
  sorry

end find_f_neg3_l176_176317


namespace missed_field_goals_l176_176126

theorem missed_field_goals (TotalAttempts MissedFraction WideRightPercentage : ℕ) 
  (TotalAttempts_eq : TotalAttempts = 60)
  (MissedFraction_eq : MissedFraction = 15)
  (WideRightPercentage_eq : WideRightPercentage = 3) : 
  (TotalAttempts * (1 / 4) * (20 / 100) = 3) :=
  by
    sorry

end missed_field_goals_l176_176126


namespace largest_three_digit_number_divisible_by_six_l176_176263

theorem largest_three_digit_number_divisible_by_six : ∃ n : ℕ, (∃ m < 1000, m ≥ 100 ∧ m % 6 = 0 ∧ m = n) ∧ (∀ k < 1000, k ≥ 100 ∧ k % 6 = 0 → k ≤ n) ∧ n = 996 :=
by sorry

end largest_three_digit_number_divisible_by_six_l176_176263


namespace man_walking_speed_l176_176804

-- This statement introduces the assumptions and goals of the proof problem.
theorem man_walking_speed
  (x : ℝ)
  (h1 : (25 * (1 / 12)) = (x * (1 / 3)))
  : x = 6.25 :=
sorry

end man_walking_speed_l176_176804


namespace jake_peaches_is_seven_l176_176464

-- Definitions based on conditions
def steven_peaches : ℕ := 13
def jake_peaches (steven : ℕ) : ℕ := steven - 6

-- The theorem we want to prove
theorem jake_peaches_is_seven : jake_peaches steven_peaches = 7 := sorry

end jake_peaches_is_seven_l176_176464


namespace twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l176_176911

variable {m n : ℕ}

def P (m : ℕ) : ℕ := 2^m
def Q (n : ℕ) : ℕ := 3^n

theorem twelve_pow_mn_eq_P_pow_2n_Q_pow_m (m n : ℕ) : 12^(m * n) = (P m)^(2 * n) * (Q n)^m := 
sorry

end twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l176_176911


namespace one_cow_one_bag_l176_176802

theorem one_cow_one_bag (h : 50 * 1 * 50 = 50 * 50) : 50 = 50 :=
by
  sorry

end one_cow_one_bag_l176_176802


namespace sum_of_n_natural_numbers_l176_176767

theorem sum_of_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 1035) : n = 46 :=
sorry

end sum_of_n_natural_numbers_l176_176767


namespace find_abc_sum_l176_176004

theorem find_abc_sum (a b c : ℕ) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
sorry

end find_abc_sum_l176_176004


namespace problem1_l176_176970

theorem problem1 {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end problem1_l176_176970


namespace avg_score_assigned_day_l176_176529

theorem avg_score_assigned_day
  (total_students : ℕ)
  (exam_assigned_day_students_perc : ℕ)
  (exam_makeup_day_students_perc : ℕ)
  (avg_makeup_day_score : ℕ)
  (total_avg_score : ℕ)
  : exam_assigned_day_students_perc = 70 → 
    exam_makeup_day_students_perc = 30 → 
    avg_makeup_day_score = 95 → 
    total_avg_score = 74 → 
    total_students = 100 → 
    (70 * 65 + 30 * 95 = 7400) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_score_assigned_day_l176_176529


namespace proof_problem_l176_176162

noncomputable def log2 : ℝ := Real.log 3 / Real.log 2
noncomputable def log5 : ℝ := Real.log 3 / Real.log 5

variables {x y : ℝ}

theorem proof_problem
  (h1 : log2 > 1)
  (h2 : 0 < log5 ∧ log5 < 1)
  (h3 : (log2^x - log5^x) ≥ (log2^(-y) - log5^(-y))) :
  x + y ≥ 0 :=
sorry

end proof_problem_l176_176162


namespace max_pizzas_l176_176053

theorem max_pizzas (dough_available cheese_available sauce_available pepperoni_available mushroom_available olive_available sausage_available: ℝ)
  (dough_per_pizza cheese_per_pizza sauce_per_pizza toppings_per_pizza: ℝ)
  (total_toppings: ℝ)
  (toppings_per_pizza_sum: total_toppings = pepperoni_available + mushroom_available + olive_available + sausage_available)
  (dough_cond: dough_available = 200)
  (cheese_cond: cheese_available = 20)
  (sauce_cond: sauce_available = 20)
  (pepperoni_cond: pepperoni_available = 15)
  (mushroom_cond: mushroom_available = 5)
  (olive_cond: olive_available = 5)
  (sausage_cond: sausage_available = 10)
  (dough_per_pizza_cond: dough_per_pizza = 1)
  (cheese_per_pizza_cond: cheese_per_pizza = 1/4)
  (sauce_per_pizza_cond: sauce_per_pizza = 1/6)
  (toppings_per_pizza_cond: toppings_per_pizza = 1/3)
  : (min (dough_available / dough_per_pizza) (min (cheese_available / cheese_per_pizza) (min (sauce_available / sauce_per_pizza) (total_toppings / toppings_per_pizza))) = 80) :=
by
  sorry

end max_pizzas_l176_176053


namespace find_2g_x_l176_176082

theorem find_2g_x (g : ℝ → ℝ) (h : ∀ x > 0, g (3 * x) = 3 / (3 + x)) (x : ℝ) (hx : x > 0) :
  2 * g x = 18 / (9 + x) :=
sorry

end find_2g_x_l176_176082


namespace janina_must_sell_21_pancakes_l176_176433

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end janina_must_sell_21_pancakes_l176_176433


namespace units_digit_37_pow_37_l176_176966

theorem units_digit_37_pow_37 : (37 ^ 37) % 10 = 7 := by
  -- The proof is omitted as per instructions.
  sorry

end units_digit_37_pow_37_l176_176966


namespace compute_nested_operations_l176_176486

def operation (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem compute_nested_operations :
  operation 5 (operation 6 (operation 7 (operation 8 9))) = 3588 / 587 :=
  sorry

end compute_nested_operations_l176_176486


namespace sqrt_fraction_fact_l176_176846

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_fraction_fact :
  Real.sqrt (factorial 9 / 210 : ℝ) = 24 * Real.sqrt 3 := by
  sorry

end sqrt_fraction_fact_l176_176846


namespace range_of_a_l176_176251

noncomputable def f (x : ℝ) : ℝ := sorry -- The actual definition of the function f is not given
def g (a x : ℝ) : ℝ := a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc (-2 : ℝ) 2 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2 : ℝ) 2 ∧ g a x₀ = f x₁) ↔
  a ≤ -1/2 ∨ 5/2 ≤ a :=
by 
  sorry

end range_of_a_l176_176251


namespace amount_paid_per_person_is_correct_l176_176599

noncomputable def amount_each_person_paid (total_bill : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) (num_people : ℕ) : ℝ := 
  let tip_amount := tip_rate * total_bill
  let tax_amount := tax_rate * total_bill
  let total_amount := total_bill + tip_amount + tax_amount
  total_amount / num_people

theorem amount_paid_per_person_is_correct :
  amount_each_person_paid 425 0.18 0.08 15 = 35.7 :=
by
  sorry

end amount_paid_per_person_is_correct_l176_176599


namespace number_of_factors_multiples_of_360_l176_176223

def n : ℕ := 2^10 * 3^14 * 5^8

theorem number_of_factors_multiples_of_360 (n : ℕ) (hn : n = 2^10 * 3^14 * 5^8) : 
  ∃ (k : ℕ), k = 832 ∧ 
  (∀ m : ℕ, m ∣ n → 360 ∣ m → k = 8 * 13 * 8) := 
sorry

end number_of_factors_multiples_of_360_l176_176223


namespace squares_end_with_76_l176_176854

noncomputable def validNumbers : List ℕ := [24, 26, 74, 76]

theorem squares_end_with_76 (x : ℕ) (h₁ : x % 10 = 4 ∨ x % 10 = 6) 
    (h₂ : (x * x) % 100 = 76) : x ∈ validNumbers := by
  sorry

end squares_end_with_76_l176_176854


namespace initial_men_invited_l176_176157

theorem initial_men_invited (M W C : ℕ) (h1 : W = M / 2) (h2 : C + 10 = 30) (h3 : M + W + C = 80) (h4 : C = 20) : M = 40 :=
sorry

end initial_men_invited_l176_176157


namespace susan_arrives_before_sam_by_14_minutes_l176_176454

theorem susan_arrives_before_sam_by_14_minutes (d : ℝ) (susan_speed sam_speed : ℝ) (h1 : d = 2) (h2 : susan_speed = 12) (h3 : sam_speed = 5) : 
  let susan_time := d / susan_speed
  let sam_time := d / sam_speed
  let susan_minutes := susan_time * 60
  let sam_minutes := sam_time * 60
  sam_minutes - susan_minutes = 14 := 
by
  sorry

end susan_arrives_before_sam_by_14_minutes_l176_176454


namespace arithmetic_progression_sum_l176_176830

variable {α : Type*} [LinearOrderedField α]

def arithmetic_progression (S : ℕ → α) :=
  ∃ (a d : α), ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_progression_sum :
  ∀ (S : ℕ → α),
  arithmetic_progression S →
  (S 4) / (S 8) = 1 / 7 →
  (S 12) / (S 4) = 43 :=
by
  intros S h_arith_prog h_ratio
  sorry

end arithmetic_progression_sum_l176_176830


namespace sin_fourth_plus_cos_fourth_l176_176260

theorem sin_fourth_plus_cos_fourth (α : ℝ) (h : Real.cos (2 * α) = 3 / 5) : 
  Real.sin α ^ 4 + Real.cos α ^ 4 = 17 / 25 := 
by
  sorry

end sin_fourth_plus_cos_fourth_l176_176260


namespace sandy_spent_on_shorts_l176_176397

variable (amount_on_shirt amount_on_jacket total_amount amount_on_shorts : ℝ)

theorem sandy_spent_on_shorts :
  amount_on_shirt = 12.14 →
  amount_on_jacket = 7.43 →
  total_amount = 33.56 →
  amount_on_shorts = total_amount - amount_on_shirt - amount_on_jacket →
  amount_on_shorts = 13.99 :=
by
  intros h_shirt h_jacket h_total h_computation
  sorry

end sandy_spent_on_shorts_l176_176397


namespace abs_five_minus_e_l176_176939

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_e : |5 - e| = 2.282 := 
by 
    -- Proof is omitted 
    sorry

end abs_five_minus_e_l176_176939


namespace sandra_beignets_16_weeks_l176_176748

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l176_176748


namespace shopkeeper_packets_l176_176328

noncomputable def milk_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) : ℝ :=
  (total_milk_oz * oz_to_ml) / ml_per_packet

theorem shopkeeper_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) :
  oz_to_ml = 30 → ml_per_packet = 250 → total_milk_oz = 1250 → milk_packets oz_to_ml ml_per_packet total_milk_oz = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shopkeeper_packets_l176_176328


namespace percentage_calculation_l176_176461

theorem percentage_calculation (Part Whole : ℕ) (h1 : Part = 90) (h2 : Whole = 270) : 
  ((Part : ℝ) / (Whole : ℝ) * 100) = 33.33 :=
by
  sorry

end percentage_calculation_l176_176461


namespace jonah_fishes_per_day_l176_176817

theorem jonah_fishes_per_day (J G J_total : ℕ) (days : ℕ) (total : ℕ)
  (hJ : J = 6) (hG : G = 8) (hdays : days = 5) (htotal : total = 90) 
  (fish_total : days * J + days * G + days * J_total = total) : 
  J_total = 4 :=
by
  sorry

end jonah_fishes_per_day_l176_176817


namespace statement_D_incorrect_l176_176914

theorem statement_D_incorrect (a b c : ℝ) : a^2 > b^2 ∧ a * b > 0 → ¬(1 / a < 1 / b) :=
by sorry

end statement_D_incorrect_l176_176914


namespace rationalize_denominator_sum_A_B_C_D_l176_176887

theorem rationalize_denominator :
  (1 / (5 : ℝ)^(1/3) - (2 : ℝ)^(1/3)) = 
  ((25 : ℝ)^(1/3) + (10 : ℝ)^(1/3) + (4 : ℝ)^(1/3)) / (3 : ℝ) := 
sorry

theorem sum_A_B_C_D : 25 + 10 + 4 + 3 = 42 := 
by norm_num

end rationalize_denominator_sum_A_B_C_D_l176_176887


namespace sum_abs_eq_pos_or_neg_three_l176_176643

theorem sum_abs_eq_pos_or_neg_three (x y : Real) (h1 : abs x = 1) (h2 : abs y = 2) (h3 : x * y > 0) :
    x + y = 3 ∨ x + y = -3 :=
by
  sorry

end sum_abs_eq_pos_or_neg_three_l176_176643


namespace teresa_class_size_l176_176826

theorem teresa_class_size :
  ∃ (a : ℤ), 50 < a ∧ a < 100 ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 2) ∧ 
  (a % 5 = 2) ∧ 
  a = 62 := 
by {
  sorry
}

end teresa_class_size_l176_176826


namespace miles_to_friends_house_l176_176266

-- Define the conditions as constants
def miles_per_gallon : ℕ := 19
def gallons : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_burger_restaurant : ℕ := 2
def miles_home : ℕ := 11

-- Define the total miles driven
def total_miles_driven (miles_to_friend : ℕ) :=
  miles_to_school + miles_to_softball_park + miles_to_burger_restaurant + miles_to_friend + miles_home

-- Define the total miles possible with given gallons of gas
def total_miles_possible : ℕ :=
  miles_per_gallon * gallons

-- Prove that the miles driven to the friend's house is 4
theorem miles_to_friends_house : 
  ∃ miles_to_friend, total_miles_driven miles_to_friend = total_miles_possible ∧ miles_to_friend = 4 :=
by
  sorry

end miles_to_friends_house_l176_176266


namespace ratio_a_to_c_l176_176398

theorem ratio_a_to_c (a b c d : ℕ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
  by 
    sorry

end ratio_a_to_c_l176_176398


namespace arithmetic_sequence_10th_term_l176_176627

theorem arithmetic_sequence_10th_term (a_1 : ℕ) (d : ℕ) (n : ℕ) 
  (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 10) : (a_1 + (n - 1) * d) = 28 := by 
  sorry

end arithmetic_sequence_10th_term_l176_176627


namespace inverse_proportion_graph_l176_176350

theorem inverse_proportion_graph (k : ℝ) (x : ℝ) (y : ℝ) (h1 : y = k / x) (h2 : (3, -4) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k < 0 → ∀ x1 x2 : ℝ, x1 < x2 → y1 = k / x1 → y2 = k / x2 → y1 < y2 := by
  sorry

end inverse_proportion_graph_l176_176350


namespace sum_of_first_four_terms_of_sequence_l176_176249

-- Define the sequence, its common difference, and the given initial condition
def a_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = 2) ∧ (a 2 = 5)

-- Define the sum of the first four terms
def sum_first_four_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_sequence :
  ∀ (a : ℕ → ℤ), a_sequence a → sum_first_four_terms a = 24 :=
by
  intro a h
  rw [a_sequence] at h
  obtain ⟨h_diff, h_a2⟩ := h
  sorry

end sum_of_first_four_terms_of_sequence_l176_176249


namespace petya_vasya_sum_equality_l176_176210

theorem petya_vasya_sum_equality : ∃ (k m : ℕ), 2^(k+1) * 1023 = m * (m + 1) :=
by
  sorry

end petya_vasya_sum_equality_l176_176210


namespace symmetry_about_x2_symmetry_about_2_0_l176_176344

-- Define the conditions and their respective conclusions.
theorem symmetry_about_x2 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) : 
  ∀ x, f (x) = f (4 - x) := 
sorry

theorem symmetry_about_2_0 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = -f (3 + x)) : 
  ∀ x, f (x) = -f (4 - x) := 
sorry

end symmetry_about_x2_symmetry_about_2_0_l176_176344


namespace sheets_of_paper_in_each_box_l176_176727

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 30)
  (h2 : 2 * E = S)
  (h3 : 3 * E = S - 10) :
  S = 40 :=
by
  sorry

end sheets_of_paper_in_each_box_l176_176727


namespace cost_of_notebook_l176_176349

theorem cost_of_notebook (num_students : ℕ) (more_than_half_bought : ℕ) (num_notebooks : ℕ) 
                         (cost_per_notebook : ℕ) (total_cost : ℕ) 
                         (half_students : more_than_half_bought > 18) 
                         (more_than_one_notebook : num_notebooks > 1) 
                         (cost_gt_notebooks : cost_per_notebook > num_notebooks) 
                         (calc_total_cost : more_than_half_bought * cost_per_notebook * num_notebooks = 2310) :
  cost_per_notebook = 11 := 
sorry

end cost_of_notebook_l176_176349


namespace max_mn_square_proof_l176_176628

noncomputable def max_mn_square (m n : ℕ) : ℕ :=
m^2 + n^2

theorem max_mn_square_proof (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2005) (h2 : 1 ≤ n ∧ n ≤ 2005) (h3 : (n^2 + 2 * m * n - 2 * m^2)^2 = 1) : 
max_mn_square m n ≤ 702036 :=
sorry

end max_mn_square_proof_l176_176628


namespace ratio_of_pieces_l176_176500

-- Definitions from the conditions
def total_length : ℝ := 28
def shorter_piece_length : ℝ := 8.000028571387755

-- Derived definition
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- Statement to prove the ratio
theorem ratio_of_pieces : 
  (shorter_piece_length / longer_piece_length) = 0.400000571428571 :=
by
  -- Use sorry to skip the proof
  sorry

end ratio_of_pieces_l176_176500


namespace garden_area_increase_l176_176578

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

end garden_area_increase_l176_176578


namespace cooling_constant_l176_176163

theorem cooling_constant (θ0 θ1 θ t k : ℝ) (h1 : θ1 = 60) (h0 : θ0 = 15) (ht : t = 3) (hθ : θ = 42)
  (h_temp_formula : θ = θ0 + (θ1 - θ0) * Real.exp (-k * t)) :
  k = 0.17 :=
by sorry

end cooling_constant_l176_176163


namespace debby_pancakes_l176_176504

def total_pancakes (B A P : ℕ) : ℕ := B + A + P

theorem debby_pancakes : 
  total_pancakes 20 24 23 = 67 := by 
  sorry

end debby_pancakes_l176_176504


namespace rectangle_area_l176_176516

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area : area length width = 588 := sorry

end rectangle_area_l176_176516


namespace min_value_of_g_inequality_f_l176_176851

def f (x m : ℝ) : ℝ := abs (x - m)
def g (x m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem min_value_of_g (m : ℝ) (hm : m > 0) (h : ∀ x, g x m ≥ -1) : m = 1 :=
sorry

theorem inequality_f {m a b : ℝ} (hm : m > 0) (ha : abs a < m) (hb : abs b < m) (h0 : a ≠ 0) :
  f (a * b) m > abs a * f (b / a) m :=
sorry

end min_value_of_g_inequality_f_l176_176851


namespace value_of_a3_plus_a5_l176_176778

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_a3_plus_a5 (a : ℕ → α) (S : ℕ → α)
  (h_sequence : arithmetic_sequence a)
  (h_S7 : S 7 = 14)
  (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 3 + a 5 = 4 :=
by
  sorry

end value_of_a3_plus_a5_l176_176778


namespace total_calories_consumed_l176_176742

-- Definitions for conditions
def calories_per_chip : ℕ := 60 / 10
def extra_calories_per_cheezit := calories_per_chip / 3
def calories_per_cheezit: ℕ := calories_per_chip + extra_calories_per_cheezit
def total_calories_chips : ℕ := 60
def total_calories_cheezits : ℕ := 6 * calories_per_cheezit

-- Main statement to be proved
theorem total_calories_consumed : total_calories_chips + total_calories_cheezits = 108 := by 
  sorry

end total_calories_consumed_l176_176742


namespace area_of_lawn_l176_176884

theorem area_of_lawn 
  (park_length : ℝ) (park_width : ℝ) (road_width : ℝ) 
  (H1 : park_length = 60) (H2 : park_width = 40) (H3 : road_width = 3) : 
  (park_length * park_width - (park_length * road_width + park_width * road_width - road_width ^ 2)) = 2109 := 
by
  sorry

end area_of_lawn_l176_176884


namespace quadratic_inequality_solution_l176_176600

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 - 8 * x - 3 > 0 ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l176_176600


namespace isabella_hair_length_l176_176977

-- Define conditions: original length and doubled length
variable (original_length : ℕ)
variable (doubled_length : ℕ := 36)

-- Theorem: Prove that if the original length doubled equals 36, then the original length is 18.
theorem isabella_hair_length (h : 2 * original_length = doubled_length) : original_length = 18 := by
  sorry

end isabella_hair_length_l176_176977


namespace trips_Jean_l176_176974

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l176_176974


namespace rational_solutions_quad_eq_iff_k_eq_4_l176_176030

theorem rational_solutions_quad_eq_iff_k_eq_4 (k : ℕ) (hk : 0 < k) : 
  (∃ x : ℚ, x^2 + 24/k * x + 9 = 0) ↔ k = 4 :=
sorry

end rational_solutions_quad_eq_iff_k_eq_4_l176_176030


namespace factor_theorem_solution_l176_176365

theorem factor_theorem_solution (t : ℝ) :
  (6 * t ^ 2 - 17 * t - 7 = 0) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end factor_theorem_solution_l176_176365


namespace f_odd_and_inequality_l176_176041

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem f_odd_and_inequality (x c : ℝ) : ∀ x c, 
  f x < c^2 - 3 * c + 3 := by 
  sorry

end f_odd_and_inequality_l176_176041


namespace find_f8_l176_176928

theorem find_f8 (f : ℕ → ℕ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : f 8 = 26 :=
by
  sorry

end find_f8_l176_176928


namespace suzanna_bike_distance_l176_176273

variable (constant_rate : ℝ) (time_minutes : ℝ) (interval : ℝ) (distance_per_interval : ℝ)

theorem suzanna_bike_distance :
  (constant_rate = 1 / interval) ∧ (interval = 5) ∧ (distance_per_interval = constant_rate * interval) ∧ (time_minutes = 30) →
  ((time_minutes / interval) * distance_per_interval = 6) :=
by
  intros
  sorry

end suzanna_bike_distance_l176_176273


namespace ratio_of_volumes_l176_176951

noncomputable def volumeSphere (p : ℝ) : ℝ := (4/3) * Real.pi * (p^3)

noncomputable def volumeHemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * (3*p)^3

theorem ratio_of_volumes (p : ℝ) (hp : p > 0) : volumeSphere p / volumeHemisphere p = 2 / 27 :=
by
  sorry

end ratio_of_volumes_l176_176951


namespace total_people_in_group_l176_176067

-- Given conditions as definitions
def numChinese : Nat := 22
def numAmericans : Nat := 16
def numAustralians : Nat := 11

-- Statement of the theorem to prove
theorem total_people_in_group : (numChinese + numAmericans + numAustralians) = 49 :=
by
  -- proof goes here
  sorry

end total_people_in_group_l176_176067


namespace calculate_8b_l176_176140

-- Define the conditions \(6a + 3b = 0\), \(b - 3 = a\), and \(b + c = 5\)
variables (a b c : ℝ)

theorem calculate_8b :
  (6 * a + 3 * b = 0) → (b - 3 = a) → (b + c = 5) → (8 * b = 16) :=
by
  intros h1 h2 h3
  -- Proof goes here, but we will use sorry to skip the proof.
  sorry

end calculate_8b_l176_176140


namespace hoseok_position_reversed_l176_176723

def nine_people (P : ℕ → Prop) : Prop :=
  P 1 ∧ P 2 ∧ P 3 ∧ P 4 ∧ P 5 ∧ P 6 ∧ P 7 ∧ P 8 ∧ P 9

variable (h : ℕ → Prop)

def hoseok_front_foremost : Prop :=
  nine_people h ∧ h 1 -- Hoseok is at the forefront and is the shortest

theorem hoseok_position_reversed :
  hoseok_front_foremost h → h 9 :=
by 
  sorry

end hoseok_position_reversed_l176_176723


namespace no_valid_partition_exists_l176_176044

namespace MathProof

-- Define the set of positive integers
def N := {n : ℕ // n > 0}

-- Define non-empty sets A, B, C which are disjoint and partition N
def valid_partition (A B C : N → Prop) : Prop :=
  (∃ a, A a) ∧ (∃ b, B b) ∧ (∃ c, C c) ∧
  (∀ n, A n → ¬ B n ∧ ¬ C n) ∧
  (∀ n, B n → ¬ A n ∧ ¬ C n) ∧
  (∀ n, C n → ¬ A n ∧ ¬ B n) ∧
  (∀ n, A n ∨ B n ∨ C n)

-- Define the conditions in the problem
def condition_1 (A B C : N → Prop) : Prop :=
  ∀ a b, A a → B b → C ⟨a.val + b.val + 1, by linarith [a.prop, b.prop]⟩

def condition_2 (A B C : N → Prop) : Prop :=
  ∀ b c, B b → C c → A ⟨b.val + c.val + 1, by linarith [b.prop, c.prop]⟩

def condition_3 (A B C : N → Prop) : Prop :=
  ∀ c a, C c → A a → B ⟨c.val + a.val + 1, by linarith [c.prop, a.prop]⟩

-- State the problem that no valid partition exists
theorem no_valid_partition_exists :
  ¬ ∃ (A B C : N → Prop), valid_partition A B C ∧
    condition_1 A B C ∧
    condition_2 A B C ∧
    condition_3 A B C :=
by
  sorry

end MathProof

end no_valid_partition_exists_l176_176044


namespace soda_cost_l176_176986

theorem soda_cost (x : ℝ) : 
    (1.5 * 35 + x * (87 - 35) = 78.5) → 
    x = 0.5 := 
by 
  intros h
  sorry

end soda_cost_l176_176986


namespace min_days_equal_duties_l176_176499

/--
Uncle Chernomor appoints 9 or 10 of the 33 warriors to duty each evening. 
Prove that the minimum number of days such that each warrior has been on duty the same number of times is 7.
-/
theorem min_days_equal_duties (k l m : ℕ) (k_nonneg : 0 ≤ k) (l_nonneg : 0 ≤ l)
  (h : 9 * k + 10 * l = 33 * m) (h_min : k + l = 7) : m = 2 :=
by 
  -- The necessary proof will go here
  sorry

end min_days_equal_duties_l176_176499


namespace instantaneous_velocity_at_2_l176_176396

def displacement (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 : (deriv displacement 2) = 8 :=
by 
  -- Proof would go here
  sorry

end instantaneous_velocity_at_2_l176_176396


namespace total_amount_paid_correct_l176_176540

-- Definitions of wholesale costs, retail markups, and employee discounts
def wholesale_cost_video_recorder : ℝ := 200
def retail_markup_video_recorder : ℝ := 0.20
def employee_discount_video_recorder : ℝ := 0.30

def wholesale_cost_digital_camera : ℝ := 150
def retail_markup_digital_camera : ℝ := 0.25
def employee_discount_digital_camera : ℝ := 0.20

def wholesale_cost_smart_tv : ℝ := 800
def retail_markup_smart_tv : ℝ := 0.15
def employee_discount_smart_tv : ℝ := 0.25

-- Calculation of retail prices
def retail_price (wholesale_cost : ℝ) (markup : ℝ) : ℝ :=
  wholesale_cost * (1 + markup)

-- Calculation of employee prices
def employee_price (retail_price : ℝ) (discount : ℝ) : ℝ :=
  retail_price * (1 - discount)

-- Retail prices
def retail_price_video_recorder := retail_price wholesale_cost_video_recorder retail_markup_video_recorder
def retail_price_digital_camera := retail_price wholesale_cost_digital_camera retail_markup_digital_camera
def retail_price_smart_tv := retail_price wholesale_cost_smart_tv retail_markup_smart_tv

-- Employee prices
def employee_price_video_recorder := employee_price retail_price_video_recorder employee_discount_video_recorder
def employee_price_digital_camera := employee_price retail_price_digital_camera employee_discount_digital_camera
def employee_price_smart_tv := employee_price retail_price_smart_tv employee_discount_smart_tv

-- Total amount paid by the employee
def total_amount_paid := 
  employee_price_video_recorder 
  + employee_price_digital_camera 
  + employee_price_smart_tv

theorem total_amount_paid_correct :
  total_amount_paid = 1008 := 
  by 
    sorry

end total_amount_paid_correct_l176_176540


namespace exact_two_solutions_l176_176842

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l176_176842


namespace smallest_root_of_equation_l176_176479

theorem smallest_root_of_equation :
  let a := (x : ℝ) - 4 / 5
  let b := (x : ℝ) - 2 / 5
  let c := (x : ℝ) - 1 / 2
  (a^2 + a * b + c^2 = 0) → (x = 4 / 5 ∨ x = 14 / 15) ∧ (min (4 / 5) (14 / 15) = 14 / 15) :=
by
  sorry

end smallest_root_of_equation_l176_176479


namespace find_point_B_l176_176389

theorem find_point_B (A B : ℝ) (h1 : A = 2) (h2 : abs (B - A) = 5) : B = -3 ∨ B = 7 :=
by
  -- This is where the proof steps would go, but we can skip it with sorry.
  sorry

end find_point_B_l176_176389


namespace natalie_needs_12_bushes_for_60_zucchinis_l176_176690

-- Definitions based on problem conditions
def bushes_to_containers (bushes : ℕ) : ℕ := bushes * 10
def containers_to_zucchinis (containers : ℕ) : ℕ := (containers * 3) / 6

-- Theorem statement
theorem natalie_needs_12_bushes_for_60_zucchinis : 
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) = 60 ∧ bushes = 12 := by
  sorry

end natalie_needs_12_bushes_for_60_zucchinis_l176_176690


namespace sum_of_ratios_eq_four_l176_176878

theorem sum_of_ratios_eq_four 
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E]
  (BD DC AE EB : ℝ)
  (h1 : BD = 2 * DC)
  (h2 : AE = 2 * EB) : 
  (BD / DC) + (AE / EB) = 4 :=
  sorry

end sum_of_ratios_eq_four_l176_176878


namespace canoes_more_than_kayaks_l176_176613

noncomputable def canoes_and_kayaks (C K : ℕ) : Prop :=
  (2 * C = 3 * K) ∧ (12 * C + 18 * K = 504) ∧ (C - K = 7)

theorem canoes_more_than_kayaks (C K : ℕ) (h : canoes_and_kayaks C K) : C - K = 7 :=
sorry

end canoes_more_than_kayaks_l176_176613


namespace boat_speed_in_still_water_equals_6_l176_176857

def river_flow_rate : ℝ := 2
def distance_upstream : ℝ := 40
def distance_downstream : ℝ := 40
def total_time : ℝ := 15

theorem boat_speed_in_still_water_equals_6 :
  ∃ b : ℝ, (40 / (b - river_flow_rate) + 40 / (b + river_flow_rate) = total_time) ∧ b = 6 :=
sorry

end boat_speed_in_still_water_equals_6_l176_176857


namespace length_of_route_l176_176233

theorem length_of_route 
  (D vA vB : ℝ)
  (h_vA : vA = D / 10)
  (h_vB : vB = D / 6)
  (t : ℝ)
  (h_va_t : vA * t = 75)
  (h_vb_t : vB * t = D - 75) :
  D = 200 :=
by
  sorry

end length_of_route_l176_176233


namespace value_of_expr_l176_176149

theorem value_of_expr (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * c / (b * d) = 15 := 
by
  sorry

end value_of_expr_l176_176149


namespace equality_of_areas_l176_176228

theorem equality_of_areas (d : ℝ) :
  (∀ d : ℝ, (1/2) * d * 3 = 9 / 2 → d = 3) ↔ d = 3 :=
by
  sorry

end equality_of_areas_l176_176228


namespace snow_leopards_arrangement_l176_176333

theorem snow_leopards_arrangement : 
  ∃ (perm : Fin 9 → Fin 9), 
    (∀ i, perm i ≠ perm j → i ≠ j) ∧ 
    (perm 0 < perm 1 ∧ perm 8 < perm 1 ∧ perm 0 < perm 8) ∧ 
    (∃ count_ways, count_ways = 4320) :=
sorry

end snow_leopards_arrangement_l176_176333


namespace mason_water_intake_l176_176634

theorem mason_water_intake
  (Theo_Daily : ℕ := 8)
  (Roxy_Daily : ℕ := 9)
  (Total_Weekly : ℕ := 168)
  (Days_Per_Week : ℕ := 7) :
  (∃ M : ℕ, M * Days_Per_Week = Total_Weekly - (Theo_Daily + Roxy_Daily) * Days_Per_Week ∧ M = 7) :=
  by
  sorry

end mason_water_intake_l176_176634


namespace scientific_notation_example_l176_176109

theorem scientific_notation_example : 3790000 = 3.79 * 10^6 := 
sorry

end scientific_notation_example_l176_176109


namespace part_1_part_2_l176_176356

noncomputable def prob_pass_no_fee : ℚ :=
  (3 / 4) * (2 / 3) +
  (1 / 4) * (3 / 4) * (2 / 3) +
  (3 / 4) * (1 / 3) * (2 / 3) +
  (1 / 4) * (3 / 4) * (1 / 3) * (2 / 3)

noncomputable def prob_pass_200_fee : ℚ :=
  (1 / 4) * (1 / 4) * (3 / 4) * ((2 / 3) + (1 / 3) * (2 / 3)) +
  (1 / 3) * (1 / 3) * (2 / 3) * ((3 / 4) + (1 / 4) * (3 / 4))

theorem part_1 : prob_pass_no_fee = 5 / 6 := by
  sorry

theorem part_2 : prob_pass_200_fee = 1 / 9 := by
  sorry

end part_1_part_2_l176_176356


namespace min_sum_is_11_over_28_l176_176434

-- Definition of the problem
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the minimum sum problem
def min_sum (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits →
  ((A : ℚ) / B + (C : ℚ) / D) = (11 : ℚ) / 28

-- The theorem statement
theorem min_sum_is_11_over_28 :
  ∃ A B C D : ℕ, min_sum A B C D :=
sorry

end min_sum_is_11_over_28_l176_176434


namespace find_p7_value_l176_176148

def quadratic (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem find_p7_value (d e f : ℝ)
  (h1 : quadratic d e f 1 = 4)
  (h2 : quadratic d e f 2 = 4) :
  quadratic d e f 7 = 5 := by
  sorry

end find_p7_value_l176_176148


namespace solve_problem_l176_176010

noncomputable def solution_set : Set ℤ := {x | abs (7 * x - 5) ≤ 9}

theorem solve_problem : solution_set = {0, 1, 2} := by
  sorry

end solve_problem_l176_176010


namespace birds_on_fence_total_l176_176321

variable (initial_birds : ℕ) (additional_birds : ℕ)

theorem birds_on_fence_total {initial_birds additional_birds : ℕ} (h1 : initial_birds = 4) (h2 : additional_birds = 6) :
    initial_birds + additional_birds = 10 :=
  by
  sorry

end birds_on_fence_total_l176_176321


namespace ball_box_problem_l176_176685

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l176_176685


namespace find_t_l176_176239

variables (V V₀ g a S t : ℝ)

-- Conditions
axiom eq1 : V = 3 * g * t + V₀
axiom eq2 : S = (3 / 2) * g * t^2 + V₀ * t + (1 / 2) * a * t^2

-- Theorem to prove
theorem find_t : t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by
  sorry

end find_t_l176_176239


namespace correct_operation_l176_176216

theorem correct_operation :
  (∀ a : ℕ, a ^ 3 * a ^ 2 = a ^ 5) ∧
  (∀ a : ℕ, a + a ^ 2 ≠ a ^ 3) ∧
  (∀ a : ℕ, 6 * a ^ 2 / (2 * a ^ 2) = 3) ∧
  (∀ a : ℕ, (3 * a ^ 2) ^ 3 ≠ 9 * a ^ 6) :=
by
  sorry

end correct_operation_l176_176216


namespace number_of_digits_in_sum_l176_176274

def is_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

theorem number_of_digits_in_sum (C D : ℕ) (hC : is_digit C) (hD : is_digit D) :
  let n1 := 98765
  let n2 := C * 1000 + 433
  let n3 := D * 100 + 22
  let s := n1 + n2 + n3
  100000 ≤ s ∧ s < 1000000 :=
by {
  sorry
}

end number_of_digits_in_sum_l176_176274


namespace welders_correct_l176_176116

-- Define the initial number of welders
def initial_welders := 12

-- Define the conditions:
-- 1. Total work is 1 job that welders can finish in 3 days.
-- 2. 9 welders leave after the first day.
-- 3. The remaining work is completed by (initial_welders - 9) in 8 days.

theorem welders_correct (W : ℕ) (h1 : W * 1/3 = 1) (h2 : (W - 9) * 8 = 2 * W) : 
  W = initial_welders :=
by
  sorry

end welders_correct_l176_176116


namespace unoccupied_volume_proof_l176_176298

-- Definitions based on conditions
def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def oil_fill_ratio : ℚ := 2 / 3
def ice_cube_volume : ℕ := 1
def number_of_ice_cubes : ℕ := 15

-- Volume calculations
def oil_volume : ℚ := oil_fill_ratio * tank_volume
def total_ice_volume : ℚ := number_of_ice_cubes * ice_cube_volume
def occupied_volume : ℚ := oil_volume + total_ice_volume

-- The final question to be proved
theorem unoccupied_volume_proof : tank_volume - occupied_volume = 305 := by
  sorry

end unoccupied_volume_proof_l176_176298


namespace eighth_square_more_tiles_than_seventh_l176_176129

-- Define the total number of tiles in the nth square
def total_tiles (n : ℕ) : ℕ := n^2 + 2 * n

-- Formulate the theorem statement
theorem eighth_square_more_tiles_than_seventh :
  total_tiles 8 - total_tiles 7 = 17 := by
  sorry

end eighth_square_more_tiles_than_seventh_l176_176129


namespace range_a_l176_176866

def f (x a : ℝ) := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem range_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l176_176866


namespace soda_preference_l176_176309

theorem soda_preference (total_surveyed : ℕ) (angle_soda_sector : ℕ) (h_total_surveyed : total_surveyed = 540) (h_angle_soda_sector : angle_soda_sector = 270) :
  let fraction_soda := angle_soda_sector / 360
  let people_soda := fraction_soda * total_surveyed
  people_soda = 405 :=
by
  sorry

end soda_preference_l176_176309


namespace average_transformation_l176_176794

theorem average_transformation (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h_avg : (a_1 + a_2 + a_3 + a_4 + a_5) / 5 = 8) : 
  ((a_1 + 10) + (a_2 - 10) + (a_3 + 10) + (a_4 - 10) + (a_5 + 10)) / 5 = 10 := 
by
  sorry

end average_transformation_l176_176794


namespace range_of_m_l176_176877

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 4| + |x + 8| ≥ m) → m ≤ 4 :=
by
  sorry

end range_of_m_l176_176877


namespace neg_proposition_l176_176455

theorem neg_proposition :
  (¬(∀ x : ℕ, x^3 > x^2)) ↔ (∃ x : ℕ, x^3 ≤ x^2) := 
sorry

end neg_proposition_l176_176455


namespace infinite_solutions_iff_m_eq_2_l176_176469

theorem infinite_solutions_iff_m_eq_2 (m x y : ℝ) :
  (m*x + 4*y = m + 2 ∧ x + m*y = m) ↔ (m = 2) ∧ (m > 1) :=
by
  sorry

end infinite_solutions_iff_m_eq_2_l176_176469


namespace range_of_m_l176_176374

noncomputable def f (x : ℝ) := Real.exp x * (x - 1)
noncomputable def g (m x : ℝ) := m * x

theorem range_of_m :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1 : ℝ) 2, f x₁ > g m x₂) ↔ m ∈ Set.Iio (-1/2 : ℝ) :=
sorry

end range_of_m_l176_176374


namespace geese_percentage_non_ducks_l176_176629

theorem geese_percentage_non_ducks :
  let total_birds := 100
  let geese := 0.20 * total_birds
  let swans := 0.30 * total_birds
  let herons := 0.15 * total_birds
  let ducks := 0.25 * total_birds
  let pigeons := 0.10 * total_birds
  let non_duck_birds := total_birds - ducks
  (geese / non_duck_birds) * 100 = 27 := 
by
  sorry

end geese_percentage_non_ducks_l176_176629


namespace sphere_radius_eq_three_l176_176457

theorem sphere_radius_eq_three (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 := 
sorry

end sphere_radius_eq_three_l176_176457


namespace sum_of_midpoints_l176_176335

variable (a b c : ℝ)

def sum_of_vertices := a + b + c

theorem sum_of_midpoints (h : sum_of_vertices a b c = 15) :
  (a + b)/2 + (a + c)/2 + (b + c)/2 = 15 :=
by
  sorry

end sum_of_midpoints_l176_176335


namespace sin_sum_square_gt_sin_prod_l176_176635

theorem sin_sum_square_gt_sin_prod (α β γ : ℝ) (h1 : α + β + γ = Real.pi) 
  (h2 : 0 < Real.sin α) (h3 : Real.sin α < 1)
  (h4 : 0 < Real.sin β) (h5 : Real.sin β < 1)
  (h6 : 0 < Real.sin γ) (h7 : Real.sin γ < 1) :
  (Real.sin α + Real.sin β + Real.sin γ) ^ 2 > 9 * Real.sin α * Real.sin β * Real.sin γ := 
sorry

end sin_sum_square_gt_sin_prod_l176_176635


namespace original_cost_price_l176_176506

theorem original_cost_price (C : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) 
    (h3 : S_new = 1.25 * C - 14.70) (h4 : S_new = 1.04 * C) : C = 70 := 
by {
  sorry
}

end original_cost_price_l176_176506


namespace min_value_f_l176_176180

noncomputable def f (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x - 1

theorem min_value_f (a : ℝ) : 
  (∀ x ∈ (Set.Icc (-1 : ℝ) 1), f a x ≥ 
    if a < -1 then 2 * a 
    else if -1 ≤ a ∧ a ≤ 1 then -1 - a ^ 2 
    else -2 * a) := 
by
  sorry

end min_value_f_l176_176180


namespace cubic_with_root_p_sq_l176_176818

theorem cubic_with_root_p_sq (p : ℝ) (hp : p^3 + p - 3 = 0) : (p^2 : ℝ) ^ 3 + 2 * (p^2) ^ 2 + p^2 - 9 = 0 :=
sorry

end cubic_with_root_p_sq_l176_176818


namespace even_expressions_l176_176583

theorem even_expressions (x y : ℕ) (hx : Even x) (hy : Even y) :
  Even (x + 5 * y) ∧
  Even (4 * x - 3 * y) ∧
  Even (2 * x^2 + 5 * y^2) ∧
  Even ((2 * x * y + 4)^2) ∧
  Even (4 * x * y) :=
by
  sorry

end even_expressions_l176_176583


namespace exact_value_range_l176_176151

theorem exact_value_range (a : ℝ) (h : |170 - a| < 0.5) : 169.5 ≤ a ∧ a < 170.5 :=
by
  sorry

end exact_value_range_l176_176151


namespace total_cost_for_tickets_l176_176537

-- Define the known quantities
def students : Nat := 20
def teachers : Nat := 3
def ticket_cost : Nat := 5

-- Define the total number of people
def total_people : Nat := students + teachers

-- Define the total cost
def total_cost : Nat := total_people * ticket_cost

-- Prove that the total cost is $115
theorem total_cost_for_tickets : total_cost = 115 := by
  -- Sorry is used here to skip the proof
  sorry

end total_cost_for_tickets_l176_176537


namespace polar_coordinates_of_point_l176_176576

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ) (r θ : ℝ), x = -1 ∧ y = 1 ∧ r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi
  → r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 := 
by
  intros x y r θ h
  sorry

end polar_coordinates_of_point_l176_176576


namespace problem1_solution_problem2_solution_l176_176297

-- Problem 1: Prove the solution set for the given inequality
theorem problem1_solution (x : ℝ) : (2 < x ∧ x ≤ (7 / 2)) ↔ ((x + 1) / (x - 2) ≥ 3) := 
sorry

-- Problem 2: Prove the solution set for the given inequality
theorem problem2_solution (x a : ℝ) : 
  (a = 0 ∧ x = 0) ∨ 
  (a > 0 ∧ -a ≤ x ∧ x ≤ 2 * a) ∨ 
  (a < 0 ∧ 2 * a ≤ x ∧ x ≤ -a) ↔ 
  x^2 - a * x - 2 * a^2 ≤ 0 := 
sorry

end problem1_solution_problem2_solution_l176_176297


namespace range_of_a_l176_176451

noncomputable def f : ℝ → ℝ → ℝ
| a, x =>
  if x ≥ -1 then a * x ^ 2 + 2 * x 
  else (1 - 3 * a) * x - 3 / 2

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → 0 < a ∧ a ≤ 1/4 :=
sorry

end range_of_a_l176_176451


namespace SeedMixtureWeights_l176_176106

theorem SeedMixtureWeights (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x / 3 = y / 2) (h3 : x / 3 = z / 3) :
  x = 3 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end SeedMixtureWeights_l176_176106


namespace arcsin_inequality_l176_176596

theorem arcsin_inequality (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  (Real.arcsin x + Real.arcsin y > Real.pi / 2) ↔ (x ≥ 0 ∧ y ≥ 0 ∧ (y^2 + x^2 > 1)) := by
sorry

end arcsin_inequality_l176_176596


namespace ratio_of_boys_to_total_l176_176935

theorem ratio_of_boys_to_total (b : ℝ) (h1 : b = 3 / 4 * (1 - b)) : b = 3 / 7 :=
by
  {
    -- The given condition (we use it to prove the target statement)
    sorry
  }

end ratio_of_boys_to_total_l176_176935


namespace width_of_rectangle_l176_176144

-- Define the given values
def length : ℝ := 2
def area : ℝ := 8

-- State the theorem
theorem width_of_rectangle : ∃ width : ℝ, area = length * width ∧ width = 4 :=
by
  -- The proof is omitted
  sorry

end width_of_rectangle_l176_176144


namespace units_digit_of_5_pow_150_plus_7_l176_176862

theorem units_digit_of_5_pow_150_plus_7 : (5^150 + 7) % 10 = 2 := by
  sorry

end units_digit_of_5_pow_150_plus_7_l176_176862


namespace square_garden_perimeter_l176_176159

theorem square_garden_perimeter (A : ℝ) (hA : A = 450) : 
    ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  by
    sorry

end square_garden_perimeter_l176_176159


namespace sin_cos_sixth_power_sum_l176_176401

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = Real.sqrt 2 / 2) : 
  (Real.sin θ)^6 + (Real.cos θ)^6 = 5 / 8 :=
by
  sorry

end sin_cos_sixth_power_sum_l176_176401


namespace f_strictly_decreasing_l176_176491

-- Define the function g(x) = x^2 - 2x - 3
def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the function f(x) = log_{1/2}(g(x))
noncomputable def f (x : ℝ) : ℝ := Real.log (g x) / Real.log (1 / 2)

-- The problem statement to prove: f(x) is strictly decreasing on the interval (3, ∞)
theorem f_strictly_decreasing : ∀ x y : ℝ, 3 < x → x < y → f y < f x := by
  sorry

end f_strictly_decreasing_l176_176491


namespace ln_n_lt_8m_l176_176358

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := 
  Real.log x - m * x^2 + 2 * n * x

theorem ln_n_lt_8m (m : ℝ) (n : ℝ) (h₀ : 0 < n) (h₁ : ∀ x > 0, f x m n ≤ f 1 m n) : 
  Real.log n < 8 * m := 
sorry

end ln_n_lt_8m_l176_176358


namespace balls_into_boxes_l176_176686

theorem balls_into_boxes :
  let n := 7 -- number of balls
  let k := 3 -- number of boxes
  let ways := Nat.choose (n + k - 1) (k - 1)
  ways = 36 :=
by
  sorry

end balls_into_boxes_l176_176686


namespace number_of_workers_l176_176083

theorem number_of_workers 
  (W : ℕ) 
  (h1 : 750 * W = (5 * 900) + 700 * (W - 5)) : 
  W = 20 := 
by 
  sorry

end number_of_workers_l176_176083


namespace least_whole_number_subtracted_l176_176172

theorem least_whole_number_subtracted {x : ℕ} (h : 6 > x ∧ 7 > x) :
  (6 - x) / (7 - x : ℝ) < 16 / 21 -> x = 3 :=
by
  intros
  sorry

end least_whole_number_subtracted_l176_176172


namespace markup_correct_l176_176338

theorem markup_correct (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  purchase_price = 48 → overhead_percentage = 0.15 → net_profit = 12 →
  (purchase_price * (1 + overhead_percentage) + net_profit - purchase_price) = 19.2 :=
by
  intros
  sorry

end markup_correct_l176_176338


namespace asymptotes_of_hyperbola_l176_176677

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (y^2 / 4 - x^2 / 9 = 1) → (y = (2 / 3) * x ∨ y = -(2 / 3) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l176_176677


namespace remainder_addition_l176_176412

theorem remainder_addition (k m : ℤ) (x y : ℤ) (h₁ : x = 124 * k + 13) (h₂ : y = 186 * m + 17) :
  ((x + y + 19) % 62) = 49 :=
by {
  sorry
}

end remainder_addition_l176_176412


namespace steve_height_after_growth_l176_176908

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l176_176908


namespace find_stu_l176_176747

open Complex

theorem find_stu (p q r s t u : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h1 : p = (q + r) / (s - 3))
  (h2 : q = (p + r) / (t - 3))
  (h3 : r = (p + q) / (u - 3))
  (h4 : s * t + s * u + t * u = 8)
  (h5 : s + t + u = 4) :
  s * t * u = 10 := 
sorry

end find_stu_l176_176747


namespace geometric_sequence_common_ratio_simple_sequence_general_term_l176_176696

-- Question 1
theorem geometric_sequence_common_ratio (a_3 : ℝ) (S_3 : ℝ) (q : ℝ) (h1 : a_3 = 3 / 2) (h2 : S_3 = 9 / 2) :
    q = -1 / 2 ∨ q = 1 :=
sorry

-- Question 2
theorem simple_sequence_general_term (S : ℕ → ℝ) (a : ℕ → ℝ) (h : ∀ n, S n = n^2) :
    ∀ n, a n = S n - S (n - 1) → ∀ n, a n = 2 * n - 1 :=
sorry

end geometric_sequence_common_ratio_simple_sequence_general_term_l176_176696


namespace remainder_140_div_k_l176_176064

theorem remainder_140_div_k (k : ℕ) (hk : k > 0) :
  (80 % k^2 = 8) → (140 % k = 2) :=
by
  sorry

end remainder_140_div_k_l176_176064


namespace percentage_increase_is_correct_l176_176739

-- Define the original and new weekly earnings
def original_earnings : ℕ := 60
def new_earnings : ℕ := 90

-- Define the percentage increase calculation
def percentage_increase (original new : ℕ) : Rat := ((new - original) / original: Rat) * 100

-- State the theorem that the percentage increase is 50%
theorem percentage_increase_is_correct : percentage_increase original_earnings new_earnings = 50 := 
sorry

end percentage_increase_is_correct_l176_176739


namespace final_coordinates_of_F_l176_176380

-- Define the points D, E, F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the initial points D, E, F
def D : Point := ⟨3, -4⟩
def E : Point := ⟨5, -1⟩
def F : Point := ⟨-2, -3⟩

-- Define the reflection over the y-axis
def reflect_over_y (p : Point) : Point := ⟨-p.x, p.y⟩

-- Define the reflection over the x-axis
def reflect_over_x (p : Point) : Point := ⟨p.x, -p.y⟩

-- First reflection over the y-axis
def F' : Point := reflect_over_y F

-- Second reflection over the x-axis
def F'' : Point := reflect_over_x F'

-- The proof problem
theorem final_coordinates_of_F'' :
  F'' = ⟨2, 3⟩ := 
sorry

end final_coordinates_of_F_l176_176380


namespace solve_for_a_l176_176498

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem solve_for_a (a : ℝ) : E a 3 2 = E a 5 3 ↔ a = -1/16 :=
by
  sorry

end solve_for_a_l176_176498


namespace quadratic_function_opens_downwards_l176_176058

theorem quadratic_function_opens_downwards (m : ℤ) (h1 : |m| = 2) (h2 : m + 1 < 0) : m = -2 := by
  sorry

end quadratic_function_opens_downwards_l176_176058


namespace perpendicular_lines_parallel_lines_l176_176693

-- Define the lines l1 and l2 in terms of a
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + a ^ 2 - 1 = 0

-- Define the perpendicular condition
def perp (a : ℝ) : Prop :=
  a * 1 + 2 * (a - 1) = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop :=
  a / 1 = 2 / (a - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perp a → a = 2 / 3 := by
  intro h
  sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 := by
  intro h
  sorry

end perpendicular_lines_parallel_lines_l176_176693


namespace parabola_transformation_l176_176652

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def shifted_left (x : ℝ) : ℝ := original_parabola (x + 1)

def shifted_down (x : ℝ) : ℝ := shifted_left x - 2

theorem parabola_transformation :
  shifted_down x = 3 * (x + 1)^2 - 2 :=
sorry

end parabola_transformation_l176_176652


namespace Oshea_needs_50_small_planters_l176_176609

structure Planter :=
  (large : ℕ)     -- Number of large planters
  (medium : ℕ)    -- Number of medium planters
  (small : ℕ)     -- Number of small planters
  (capacity_large : ℕ := 20) -- Capacity of large planter
  (capacity_medium : ℕ := 10) -- Capacity of medium planter
  (capacity_small : ℕ := 4)  -- Capacity of small planter

structure Seeds :=
  (basil : ℕ)     -- Number of basil seeds
  (cilantro : ℕ)  -- Number of cilantro seeds
  (parsley : ℕ)   -- Number of parsley seeds

noncomputable def small_planters_needed (planters : Planter) (seeds : Seeds) : ℕ :=
  let basil_in_large := min seeds.basil (planters.large * planters.capacity_large)
  let basil_left := seeds.basil - basil_in_large
  let basil_in_medium := min basil_left (planters.medium * planters.capacity_medium)
  let basil_remaining := basil_left - basil_in_medium
  
  let cilantro_in_medium := min seeds.cilantro ((planters.medium * planters.capacity_medium) - basil_in_medium)
  let cilantro_remaining := seeds.cilantro - cilantro_in_medium
  
  let parsley_total := seeds.parsley + basil_remaining + cilantro_remaining
  parsley_total / planters.capacity_small

theorem Oshea_needs_50_small_planters :
  small_planters_needed 
    { large := 4, medium := 8, small := 0 }
    { basil := 200, cilantro := 160, parsley := 120 } = 50 := 
sorry

end Oshea_needs_50_small_planters_l176_176609


namespace range_of_a_l176_176460

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 + a ≤ 0
def q (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a

-- The theorem statement: if p is false and q is true, then 1 < a < 2
theorem range_of_a (a : ℝ) (h1 : ¬ p a) (h2 : q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l176_176460


namespace calc_expression_is_24_l176_176201

def calc_expression : ℕ := (30 / (8 + 2 - 5)) * 4

theorem calc_expression_is_24 : calc_expression = 24 :=
by
  sorry

end calc_expression_is_24_l176_176201


namespace product_of_five_consecutive_numbers_not_square_l176_176408

theorem product_of_five_consecutive_numbers_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) :=
by
  sorry

end product_of_five_consecutive_numbers_not_square_l176_176408


namespace kayak_manufacture_total_l176_176009

theorem kayak_manufacture_total :
  let feb : ℕ := 5
  let mar : ℕ := 3 * feb
  let apr : ℕ := 3 * mar
  let may : ℕ := 3 * apr
  feb + mar + apr + may = 200 := by
  sorry

end kayak_manufacture_total_l176_176009


namespace crayons_problem_l176_176193

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end crayons_problem_l176_176193


namespace tablet_screen_area_difference_l176_176418

theorem tablet_screen_area_difference (d1 d2 : ℝ) (A1 A2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 7) :
  A1 - A2 = 7.5 :=
by
  -- Note: The proof is omitted as the prompt requires only the statement.
  sorry

end tablet_screen_area_difference_l176_176418


namespace dog_roaming_area_comparison_l176_176167

theorem dog_roaming_area_comparison :
  let r := 10
  let a1 := (1/2) * Real.pi * r^2
  let a2 := (3/4) * Real.pi * r^2 - (1/4) * Real.pi * 6^2 
  a2 > a1 ∧ a2 - a1 = 16 * Real.pi :=
by
  sorry

end dog_roaming_area_comparison_l176_176167


namespace circle_table_acquaintance_impossible_l176_176985

theorem circle_table_acquaintance_impossible (P : Finset ℕ) (hP : P.card = 40) :
  ¬ (∀ (a b : ℕ), (a ∈ P) → (b ∈ P) → (∃ k, 2 * k ≠ 0) → (∃ c, c ∈ P) ∧ (a ≠ b) ∧ (c = a ∨ c = b)
       ↔ ¬(∃ k, 2 * k + 1 ≠ 0)) :=
by
  sorry

end circle_table_acquaintance_impossible_l176_176985


namespace probability_second_marble_purple_correct_l176_176560

/-!
  Bag A has 5 red marbles and 5 green marbles.
  Bag B has 8 purple marbles and 2 orange marbles.
  Bag C has 3 purple marbles and 7 orange marbles.
  Bag D has 4 purple marbles and 6 orange marbles.
  A marble is drawn at random from Bag A.
  If it is red, a marble is drawn at random from Bag B;
  if it is green, a marble is drawn at random from Bag C;
  but if it is neither (an impossible scenario in this setup), a marble would be drawn from Bag D.
  Prove that the probability of the second marble drawn being purple is 11/20.
-/

noncomputable def probability_second_marble_purple : ℚ :=
  let p_red_A := 5 / 10
  let p_green_A := 5 / 10
  let p_purple_B := 8 / 10
  let p_purple_C := 3 / 10
  (p_red_A * p_purple_B) + (p_green_A * p_purple_C)

theorem probability_second_marble_purple_correct :
  probability_second_marble_purple = 11 / 20 := sorry

end probability_second_marble_purple_correct_l176_176560


namespace form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l176_176954

theorem form_eleven : 22 - (2 + (2 / 2)) = 11 := by
  sorry

theorem form_twelve : (2 * 2 * 2) - 2 / 2 = 12 := by
  sorry

theorem form_thirteen : (22 + 2 + 2) / 2 = 13 := by
  sorry

theorem form_fourteen : 2 * 2 * 2 * 2 - 2 = 14 := by
  sorry

theorem form_fifteen : (2 * 2)^2 - 2 / 2 = 15 := by
  sorry

theorem form_sixteen : (2 * 2)^2 * (2 / 2) = 16 := by
  sorry

theorem form_seventeen : (2 * 2)^2 + 2 / 2 = 17 := by
  sorry

theorem form_eighteen : 2 * 2 * 2 * 2 + 2 = 18 := by
  sorry

theorem form_nineteen : 22 - 2 - 2 / 2 = 19 := by
  sorry

theorem form_twenty : (22 - 2) * (2 / 2) = 20 := by
  sorry

end form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l176_176954


namespace fraction_inequality_l176_176840

theorem fraction_inequality 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : 1 / a > 1 / b)
  (h2 : x > y) : 
  x / (x + a) > y / (y + b) := 
  sorry

end fraction_inequality_l176_176840


namespace value_of_expression_l176_176340

variable (p q r s : ℝ)

-- Given condition in a)
def polynomial_function (x : ℝ) := p * x^3 + q * x^2 + r * x + s
def passes_through_point := polynomial_function p q r s (-1) = 4

-- Proof statement in c)
theorem value_of_expression (h : passes_through_point p q r s) : 6 * p - 3 * q + r - 2 * s = -24 := by
  sorry

end value_of_expression_l176_176340


namespace empty_subset_singleton_zero_l176_176207

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) :=
by
  sorry

end empty_subset_singleton_zero_l176_176207


namespace alissa_presents_l176_176371

def ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0

theorem alissa_presents : ethan_presents - difference = 9.0 := by sorry

end alissa_presents_l176_176371


namespace symmetric_point_origin_l176_176637

theorem symmetric_point_origin (x y : Int) (hx : x = -(-4)) (hy : y = -(3)) :
    (x, y) = (4, -3) := by
  sorry

end symmetric_point_origin_l176_176637


namespace gcd_lcm_product_eq_l176_176933

theorem gcd_lcm_product_eq (a b : ℕ) : gcd a b * lcm a b = a * b := by
  sorry

example : ∃ (a b : ℕ), a = 30 ∧ b = 75 ∧ gcd a b * lcm a b = a * b :=
  ⟨30, 75, rfl, rfl, gcd_lcm_product_eq 30 75⟩

end gcd_lcm_product_eq_l176_176933


namespace b_bounded_l176_176423

open Real

-- Define sequences of real numbers
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define initial conditions and properties
axiom a0_gt_half : a 0 > 1/2
axiom a_non_decreasing : ∀ n : ℕ, a (n + 1) ≥ a n
axiom b_recursive : ∀ n : ℕ, b (n + 1) = a n * (b n + b (n + 2))

-- Prove the sequence (b_n) is bounded
theorem b_bounded : ∃ M : ℝ, ∀ n : ℕ, b n ≤ M :=
by
  sorry

end b_bounded_l176_176423


namespace sum_of_sequence_correct_l176_176762

def calculateSumOfSequence : ℚ :=
  (4 / 3) + (7 / 5) + (11 / 8) + (19 / 15) + (35 / 27) + (67 / 52) - 9

theorem sum_of_sequence_correct :
  calculateSumOfSequence = (-17312.5 / 7020) := by
  sorry

end sum_of_sequence_correct_l176_176762


namespace function_domain_l176_176494

theorem function_domain (x : ℝ) :
  (x - 3 > 0) ∧ (5 - x ≥ 0) ↔ (3 < x ∧ x ≤ 5) :=
by
  sorry

end function_domain_l176_176494


namespace ratio_doctors_to_lawyers_l176_176190

-- Definitions based on conditions
def average_age_doctors := 35
def average_age_lawyers := 50
def combined_average_age := 40

-- Define variables
variables (d l : ℕ) -- d is number of doctors, l is number of lawyers

-- Hypothesis based on the problem statement
axiom h : (average_age_doctors * d + average_age_lawyers * l) = combined_average_age * (d + l)

-- The theorem we need to prove is the ratio of doctors to lawyers is 2:1
theorem ratio_doctors_to_lawyers : d = 2 * l :=
by sorry

end ratio_doctors_to_lawyers_l176_176190


namespace convex_polyhedron_inequality_l176_176622

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end convex_polyhedron_inequality_l176_176622


namespace bryan_push_ups_l176_176675

theorem bryan_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (fewer_in_last_set : ℕ) 
  (h1 : sets = 3) (h2 : push_ups_per_set = 15) (h3 : fewer_in_last_set = 5) :
  (sets - 1) * push_ups_per_set + (push_ups_per_set - fewer_in_last_set) = 40 := by 
  -- We are setting sorry here to skip the proof.
  sorry

end bryan_push_ups_l176_176675


namespace largest_primes_product_l176_176716

theorem largest_primes_product : 7 * 97 * 997 = 679679 := by
  sorry

end largest_primes_product_l176_176716


namespace list_price_l176_176824

theorem list_price (P : ℝ) (h₀ : 0.83817 * P = 56.16) : P = 67 :=
sorry

end list_price_l176_176824


namespace least_number_of_shoes_l176_176546

theorem least_number_of_shoes (num_inhabitants : ℕ) 
  (one_legged_percentage : ℚ) 
  (barefooted_proportion : ℚ) 
  (h_num_inhabitants : num_inhabitants = 10000) 
  (h_one_legged_percentage : one_legged_percentage = 0.05) 
  (h_barefooted_proportion : barefooted_proportion = 0.5) : 
  ∃ (shoes_needed : ℕ), shoes_needed = 10000 := 
by
  sorry

end least_number_of_shoes_l176_176546


namespace trapezoid_larger_base_length_l176_176425

theorem trapezoid_larger_base_length
  (x : ℝ)
  (h_ratio : 3 = 3 * 1)
  (h_midline : (x + 3 * x) / 2 = 24) :
  3 * x = 36 :=
by
  sorry

end trapezoid_larger_base_length_l176_176425


namespace divisible_by_square_of_k_l176_176269

theorem divisible_by_square_of_k (a b l : ℕ) (k : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : a % 2 = 1) (h4 : b % 2 = 1) (h5 : a + b = 2 ^ l) : k = 1 ↔ k^2 ∣ a^k + b^k := 
sorry

end divisible_by_square_of_k_l176_176269


namespace smallest_positive_x_l176_176284

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l176_176284


namespace cubic_expression_value_l176_176495

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end cubic_expression_value_l176_176495


namespace max_food_cost_l176_176370

theorem max_food_cost (total_cost : ℝ) (food_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_allowable : ℝ)
  (h1 : tax_rate = 0.07) (h2 : tip_rate = 0.15) (h3 : max_allowable = 75) (h4 : total_cost = food_cost * (1 + tax_rate + tip_rate)) :
  food_cost ≤ 61.48 :=
sorry

end max_food_cost_l176_176370


namespace zookeeper_configurations_l176_176066

theorem zookeeper_configurations :
  ∃ (configs : ℕ), configs = 3 ∧ 
  (∀ (r p : ℕ), 
    30 * r + 35 * p = 1400 ∧ p ≥ r → 
    ((r, p) = (7, 34) ∨ (r, p) = (14, 28) ∨ (r, p) = (21, 22))) :=
sorry

end zookeeper_configurations_l176_176066


namespace equivalent_problem_l176_176848

theorem equivalent_problem :
  let a : ℤ := (-6)
  let b : ℤ := 6
  let c : ℤ := 2
  let d : ℤ := 4
  (a^4 / b^2 - c^5 + d^2 = 20) :=
by
  sorry

end equivalent_problem_l176_176848


namespace sean_days_played_is_14_l176_176521

def total_minutes_played : Nat := 1512
def indira_minutes_played : Nat := 812
def sean_minutes_per_day : Nat := 50
def sean_total_minutes : Nat := total_minutes_played - indira_minutes_played
def sean_days_played : Nat := sean_total_minutes / sean_minutes_per_day

theorem sean_days_played_is_14 : sean_days_played = 14 :=
by
  sorry

end sean_days_played_is_14_l176_176521


namespace squared_difference_l176_176062

theorem squared_difference (x y : ℝ) (h₁ : (x + y)^2 = 49) (h₂ : x * y = 8) : (x - y)^2 = 17 := 
by
  -- Proof omitted
  sorry

end squared_difference_l176_176062


namespace binomial_coeff_sum_abs_l176_176744

theorem binomial_coeff_sum_abs (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ)
  (h : (2 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0):
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end binomial_coeff_sum_abs_l176_176744


namespace mean_median_mode_l176_176725

theorem mean_median_mode (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : m + 7 < n) 
  (h4 : (m + (m + 3) + (m + 7) + n + (n + 5) + (2 * n - 1)) / 6 = n)
  (h5 : ((m + 7) + n) / 2 = n)
  (h6 : (m+3 < m+7 ∧ m+7 = n ∧ n < n+5 ∧ n+5 < 2*n - 1 )) :
  m+n = 2*n := by
  sorry

end mean_median_mode_l176_176725


namespace pencils_per_row_l176_176256

-- Definitions of conditions.
def num_pencils : ℕ := 35
def num_rows : ℕ := 7

-- Hypothesis: given the conditions, prove the number of pencils per row.
theorem pencils_per_row : num_pencils / num_rows = 5 := 
  by 
  -- Proof steps go here, but are replaced by sorry.
  sorry

end pencils_per_row_l176_176256


namespace price_of_second_oil_l176_176621

theorem price_of_second_oil : 
  ∃ x : ℝ, 
    (10 * 50 + 5 * x = 15 * 56) → x = 68 := by
  sorry

end price_of_second_oil_l176_176621


namespace incorrect_expression_l176_176519

theorem incorrect_expression :
  ¬((|(-5 : ℤ)|)^2 = 5) :=
by
sorry

end incorrect_expression_l176_176519


namespace geometric_sequence_sum_5_is_75_l176_176119

noncomputable def geometric_sequence_sum_5 (a r : ℝ) : ℝ :=
  a * (1 + r + r^2 + r^3 + r^4)

theorem geometric_sequence_sum_5_is_75 (a r : ℝ)
  (h1 : a * (1 + r + r^2) = 13)
  (h2 : a * (1 - r^7) / (1 - r) = 183) :
  geometric_sequence_sum_5 a r = 75 :=
sorry

end geometric_sequence_sum_5_is_75_l176_176119


namespace sin_triple_alpha_minus_beta_l176_176895

open Real 

theorem sin_triple_alpha_minus_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : π / 2 < β ∧ β < π)
  (h1 : cos (α - β) = 1 / 2)
  (h2 : sin (α + β) = 1 / 2) :
  sin (3 * α - β) = 1 / 2 :=
by
  sorry

end sin_triple_alpha_minus_beta_l176_176895


namespace time_to_traverse_nth_mile_l176_176211

theorem time_to_traverse_nth_mile (n : ℕ) (n_pos : n > 1) :
  let k := (1 / 2 : ℝ)
  let s_n := k / ((n-1) * (2 ^ (n-2)))
  let t_n := 1 / s_n
  t_n = 2 * (n-1) * 2^(n-2) := 
by sorry

end time_to_traverse_nth_mile_l176_176211


namespace find_b_perpendicular_l176_176751

theorem find_b_perpendicular (b : ℝ) : (∀ x y : ℝ, 4 * y - 2 * x = 6 → 5 * y + b * x - 2 = 0 → (1 / 2 : ℝ) * (-(b / 5) : ℝ) = -1) → b = 10 :=
by
  intro h
  sorry

end find_b_perpendicular_l176_176751


namespace speed_of_train_approx_29_0088_kmh_l176_176684

noncomputable def speed_of_train_in_kmh := 
  let length_train : ℝ := 288
  let length_bridge : ℝ := 101
  let time_seconds : ℝ := 48.29
  let total_distance : ℝ := length_train + length_bridge
  let speed_m_per_s : ℝ := total_distance / time_seconds
  speed_m_per_s * 3.6

theorem speed_of_train_approx_29_0088_kmh :
  abs (speed_of_train_in_kmh - 29.0088) < 0.001 := 
by
  sorry

end speed_of_train_approx_29_0088_kmh_l176_176684


namespace triangle_right_angle_l176_176649

theorem triangle_right_angle {a b c : ℝ} {A B C : ℝ} (h : a * Real.cos A + b * Real.cos B = c * Real.cos C) :
  (A = Real.pi / 2) ∨ (B = Real.pi / 2) ∨ (C = Real.pi / 2) :=
sorry

end triangle_right_angle_l176_176649


namespace highest_student_id_in_sample_l176_176229

theorem highest_student_id_in_sample
    (total_students : ℕ)
    (sample_size : ℕ)
    (included_student_id : ℕ)
    (interval : ℕ)
    (first_id in_sample : ℕ)
    (k : ℕ)
    (highest_id : ℕ)
    (total_students_eq : total_students = 63)
    (sample_size_eq : sample_size = 7)
    (included_student_id_eq : included_student_id = 11)
    (k_def : k = total_students / sample_size)
    (included_student_id_in_second_pos : included_student_id = first_id + k)
    (interval_eq : interval = first_id - k)
    (in_sample_eq : in_sample = interval)
    (highest_id_eq : highest_id = in_sample + k * (sample_size - 1)) :
  highest_id = 56 := sorry

end highest_student_id_in_sample_l176_176229


namespace one_thirds_in_nine_thirds_l176_176141

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l176_176141


namespace sum_eq_sqrt_122_l176_176960

theorem sum_eq_sqrt_122 
  (a b c : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h1 : a^2 + b^2 + c^2 = 58) 
  (h2 : a * b + b * c + c * a = 32) :
  a + b + c = Real.sqrt 122 := 
by
  sorry

end sum_eq_sqrt_122_l176_176960


namespace range_of_m_l176_176113

noncomputable def f (a x : ℝ) : ℝ := a * x - (2 * a + 1) / x

theorem range_of_m (a m : ℝ) (h₀ : a > 0) (h₁ : f a (m^2 + 1) > f a (m^2 - m + 3)) 
  : m > 2 :=
sorry

end range_of_m_l176_176113


namespace a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l176_176327

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

def complex_from_a (a : ℝ) : ℂ :=
  (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

theorem a_minus_two_sufficient_but_not_necessary_for_pure_imaginary :
  (is_pure_imaginary (complex_from_a (-2))) ∧ ¬ (∀ (a : ℝ), is_pure_imaginary (complex_from_a a) → a = -2) :=
by
  sorry

end a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l176_176327


namespace diagonals_in_decagon_l176_176505

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l176_176505


namespace total_pennies_l176_176301

theorem total_pennies (rachelle_pennies : ℕ) (gretchen_pennies : ℕ) (rocky_pennies : ℕ)
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) :
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 :=
by
  sorry

end total_pennies_l176_176301


namespace arithmetic_sequence_smallest_value_l176_176996

theorem arithmetic_sequence_smallest_value:
  ∃ a : ℕ, (7 * a + 63) % 11 = 0 ∧ (a - 9) % 11 = 4 := sorry

end arithmetic_sequence_smallest_value_l176_176996


namespace least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l176_176812

-- The numbers involved and the requirements described
def num : ℕ := 427398

def least_to_subtract_10 : ℕ := 8
def least_to_subtract_100 : ℕ := 98
def least_to_subtract_1000 : ℕ := 398

-- Proving the conditions:
-- 1. (num - least_to_subtract_10) is divisible by 10
-- 2. (num - least_to_subtract_100) is divisible by 100
-- 3. (num - least_to_subtract_1000) is divisible by 1000

theorem least_subtract_divisible_by_10 : (num - least_to_subtract_10) % 10 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_100 : (num - least_to_subtract_100) % 100 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_1000 : (num - least_to_subtract_1000) % 1000 = 0 := 
by 
  sorry

end least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l176_176812


namespace math_problem_common_factors_and_multiples_l176_176184

-- Definitions
def a : ℕ := 180
def b : ℕ := 300

-- The Lean statement to be proved
theorem math_problem_common_factors_and_multiples :
    Nat.lcm a b = 900 ∧
    Nat.gcd a b = 60 ∧
    {d | d ∣ a ∧ d ∣ b} = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} :=
by
  sorry

end math_problem_common_factors_and_multiples_l176_176184


namespace area_to_paint_l176_176051

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5
def window_height : ℕ := 2
def window_length : ℕ := 3

theorem area_to_paint : (wall_height * wall_length) - (door_height * door_length + window_height * window_length) = 129 := by
  sorry

end area_to_paint_l176_176051


namespace probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l176_176873

theorem probability_one_piece_is_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) : 
  (if (piece_lengths.1 = 2 ∧ piece_lengths.2 ≠ 2) ∨ (piece_lengths.1 ≠ 2 ∧ piece_lengths.2 = 2) then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 2 / 5 :=
sorry

theorem probability_both_pieces_longer_than_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) :
  (if piece_lengths.1 > 2 ∧ piece_lengths.2 > 2 then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 1 / 3 :=
sorry

end probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l176_176873


namespace gecko_third_day_crickets_l176_176283

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end gecko_third_day_crickets_l176_176283


namespace find_b_l176_176428

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 15 * b) : b = 147 :=
sorry

end find_b_l176_176428


namespace probability_bypass_kth_intersection_l176_176936

variable (n k : ℕ)

def P (n k : ℕ) : ℚ := (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_bypass_kth_intersection :
  P n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 :=
by
  sorry

end probability_bypass_kth_intersection_l176_176936


namespace find_number_l176_176138

theorem find_number :
  ∃ x : ℝ, (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 ∧ x = 20 :=
by
  sorry

end find_number_l176_176138


namespace mow_lawn_time_l176_176111

noncomputable def time_to_mow (length width swath_width overlap speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12 -- Convert inches to feet
  let strips_needed := width / effective_swath
  let total_distance := strips_needed * length
  total_distance / speed

theorem mow_lawn_time : time_to_mow 100 140 30 6 4500 = 1.6 :=
by
  sorry

end mow_lawn_time_l176_176111


namespace remainder_of_sum_of_primes_is_eight_l176_176566

-- Define the first eight primes and their sum
def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sumFirstEightPrimes : ℕ := 77

-- Define the ninth prime
def ninthPrime : ℕ := 23

-- Theorem stating the equivalence
theorem remainder_of_sum_of_primes_is_eight :
  (sumFirstEightPrimes % ninthPrime) = 8 := by
  sorry

end remainder_of_sum_of_primes_is_eight_l176_176566


namespace three_digit_number_441_or_882_l176_176605

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  n = 100 * a + 10 * b + c ∧
  n / (100 * c + 10 * b + a) = 3 ∧
  n % (100 * c + 10 * b + a) = a + b + c

theorem three_digit_number_441_or_882:
  ∀ n : ℕ, is_valid_number n → (n = 441 ∨ n = 882) :=
by
  sorry

end three_digit_number_441_or_882_l176_176605


namespace evaluate_g_at_neg2_l176_176885

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem evaluate_g_at_neg2 : g (-2) = 11 := by
  sorry

end evaluate_g_at_neg2_l176_176885


namespace find_FC_l176_176770

theorem find_FC
  (DC : ℝ) (CB : ℝ) (AD : ℝ)
  (hDC : DC = 9) (hCB : CB = 10)
  (hAB : ∃ (k1 : ℝ), k1 = 1/5 ∧ AB = k1 * AD)
  (hED : ∃ (k2 : ℝ), k2 = 3/4 ∧ ED = k2 * AD) :
  ∃ FC : ℝ, FC = 11.025 :=
by
  sorry

end find_FC_l176_176770


namespace condition1_a_geq_1_l176_176130

theorem condition1_a_geq_1 (a : ℝ) :
  (∀ x ∈ ({1, 2, 3} : Set ℝ), a * x - 1 ≥ 0) → a ≥ 1 :=
by
sorry

end condition1_a_geq_1_l176_176130


namespace matt_books_second_year_l176_176707

-- Definitions based on the conditions
variables (M : ℕ) -- number of books Matt read last year
variables (P : ℕ) -- number of books Pete read last year

-- Pete read twice as many books as Matt last year
def pete_read_last_year (M : ℕ) : ℕ := 2 * M

-- This year, Pete doubles the number of books he read last year
def pete_read_this_year (M : ℕ) : ℕ := 2 * (2 * M)

-- Matt reads 50% more books this year than he did last year
def matt_read_this_year (M : ℕ) : ℕ := M + M / 2

-- Pete read 300 books across both years
def total_books_pete_read_last_and_this_year (M : ℕ) : ℕ :=
  pete_read_last_year M + pete_read_this_year M

-- Prove that Matt read 75 books in his second year
theorem matt_books_second_year (M : ℕ) (h : total_books_pete_read_last_and_this_year M = 300) :
  matt_read_this_year M = 75 :=
by sorry

end matt_books_second_year_l176_176707


namespace volume_of_cuboid_l176_176291

theorem volume_of_cuboid (l w h : ℝ) (hl_pos : 0 < l) (hw_pos : 0 < w) (hh_pos : 0 < h) 
  (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) : l * w * h = 4320 :=
by
  sorry

end volume_of_cuboid_l176_176291


namespace randy_trip_length_l176_176896

-- Define the conditions
noncomputable def fraction_gravel := (1/4 : ℚ)
noncomputable def miles_pavement := (30 : ℚ)
noncomputable def fraction_dirt := (1/6 : ℚ)

-- The proof statement
theorem randy_trip_length :
  ∃ x : ℚ, (fraction_gravel + fraction_dirt + (miles_pavement / x) = 1) ∧ x = 360 / 7 := 
by
  sorry

end randy_trip_length_l176_176896


namespace garage_sale_items_l176_176772

-- Definition of conditions
def is_18th_highest (num_highest: ℕ) : Prop := num_highest = 17
def is_25th_lowest (num_lowest: ℕ) : Prop := num_lowest = 24

-- Theorem statement
theorem garage_sale_items (num_highest num_lowest total_items: ℕ) 
  (h1: is_18th_highest num_highest) (h2: is_25th_lowest num_lowest) :
  total_items = num_highest + num_lowest + 1 :=
by
  -- Proof omitted
  sorry

end garage_sale_items_l176_176772


namespace Mitch_hourly_rate_l176_176868

theorem Mitch_hourly_rate :
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  weekly_earnings / total_hours = 3 :=
by
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  sorry

end Mitch_hourly_rate_l176_176868


namespace value_of_y_l176_176293

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 :=
by
  sorry

end value_of_y_l176_176293


namespace largest_n_with_100_trailing_zeros_l176_176262

def trailing_zeros_factorial (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + trailing_zeros_factorial (n / 5)

theorem largest_n_with_100_trailing_zeros :
  ∃ (n : ℕ), trailing_zeros_factorial n = 100 ∧ ∀ (m : ℕ), (trailing_zeros_factorial m = 100 → m ≤ 409) :=
by
  sorry

end largest_n_with_100_trailing_zeros_l176_176262


namespace smallest_n_condition_smallest_n_value_l176_176352

theorem smallest_n_condition :
  ∃ (n : ℕ), n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) ∧ 
  ∀ m, (m < 1000 ∧ (99999 % m = 0) ∧ (9999 % (m + 7) = 0)) → n ≤ m := 
sorry

theorem smallest_n_value :
  ∃ (n : ℕ), n = 266 ∧ n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) := 
sorry

end smallest_n_condition_smallest_n_value_l176_176352


namespace root_in_interval_sum_eq_three_l176_176730

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end root_in_interval_sum_eq_three_l176_176730


namespace length_of_AP_in_right_triangle_l176_176925

theorem length_of_AP_in_right_triangle 
  (A B C : ℝ × ℝ)
  (hA : A = (0, 2))
  (hB : B = (0, 0))
  (hC : C = (2, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0 ∧ M.2 = 0)
  (inc : ℝ × ℝ)
  (hinc : inc = (1, 1)) :
  ∃ P : ℝ × ℝ, (P.1 = 0 ∧ P.2 = 1) ∧ dist A P = 1 := by
  sorry

end length_of_AP_in_right_triangle_l176_176925


namespace eval_floor_abs_value_l176_176047

theorem eval_floor_abs_value : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry -- Proof is to be filled in

end eval_floor_abs_value_l176_176047


namespace original_price_of_painting_l176_176112

theorem original_price_of_painting (purchase_price : ℝ) (fraction : ℝ) (original_price : ℝ) :
  purchase_price = 200 → fraction = 1/4 → purchase_price = original_price * fraction → original_price = 800 :=
by
  intros h1 h2 h3
  -- proof steps here
  sorry

end original_price_of_painting_l176_176112


namespace inequality_solution_l176_176639

theorem inequality_solution {a b x : ℝ} 
  (h_sol_set : -1 < x ∧ x < 1) 
  (h1 : x - a > 2) 
  (h2 : b - 2 * x > 0) : 
  (a + b) ^ 2021 = -1 := 
by 
  sorry 

end inequality_solution_l176_176639


namespace subset_exists_l176_176410

theorem subset_exists (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ) (hA : A.card = p - 1) 
  (hA_div : ∀ a ∈ A, ¬ p ∣ a) :
  ∀ n ∈ Finset.range p, ∃ B ⊆ A, (B.sum id) % p = n :=
by
  -- Proof goes here
  sorry

end subset_exists_l176_176410


namespace age_difference_l176_176019

theorem age_difference (B_age : ℕ) (A_age : ℕ) (X : ℕ) : 
  B_age = 42 → 
  A_age = B_age + 12 → 
  A_age + 10 = 2 * (B_age - X) → 
  X = 10 :=
by
  intros hB_age hA_age hEquation 
  -- define variables based on conditions
  have hB : B_age = 42 := hB_age
  have hA : A_age = B_age + 12 := hA_age
  have hEq : A_age + 10 = 2 * (B_age - X) := hEquation
  -- expected result
  sorry

end age_difference_l176_176019


namespace largest_number_in_set_l176_176603

theorem largest_number_in_set :
  ∀ (a b c d : ℤ), (a ∈ [0, 2, -1, -2]) → (b ∈ [0, 2, -1, -2]) → (c ∈ [0, 2, -1, -2]) → (d ∈ [0, 2, -1, -2])
  → (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  → max (max a b) (max c d) = 2
  := 
by
  sorry

end largest_number_in_set_l176_176603


namespace trick_or_treat_hours_l176_176731

variable (num_children : ℕ)
variable (houses_per_hour : ℕ)
variable (treats_per_house_per_kid : ℕ)
variable (total_treats : ℕ)

theorem trick_or_treat_hours (h : num_children = 3)
  (h1 : houses_per_hour = 5)
  (h2 : treats_per_house_per_kid = 3)
  (h3 : total_treats = 180) :
  total_treats / (num_children * houses_per_hour * treats_per_house_per_kid) = 4 :=
by
  sorry

end trick_or_treat_hours_l176_176731


namespace solve_for_x_l176_176701

theorem solve_for_x (x : ℝ) (h1 : x ≠ -3) (h2 : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5)) : x = -9 :=
by
  sorry

end solve_for_x_l176_176701


namespace gemstone_necklaces_count_l176_176472

-- Conditions
def num_bead_necklaces : ℕ := 3
def price_per_necklace : ℕ := 7
def total_earnings : ℕ := 70

-- Proof Problem
theorem gemstone_necklaces_count : (total_earnings - num_bead_necklaces * price_per_necklace) / price_per_necklace = 7 := by
  sorry

end gemstone_necklaces_count_l176_176472


namespace middle_digit_base7_l176_176136

theorem middle_digit_base7 (a b c : ℕ) 
  (h1 : N = 49 * a + 7 * b + c) 
  (h2 : N = 81 * c + 9 * b + a)
  (h3 : a < 7 ∧ b < 7 ∧ c < 7) : 
  b = 0 :=
by sorry

end middle_digit_base7_l176_176136


namespace find_a_l176_176070

def A (x : ℝ) : Set ℝ := {1, 2, x^2 - 5 * x + 9}
def B (x a : ℝ) : Set ℝ := {3, x^2 + a * x + a}

theorem find_a (a x : ℝ) (hxA : A x = {1, 2, 3}) (h2B : 2 ∈ B x a) :
  a = -2/3 ∨ a = -7/4 :=
by sorry

end find_a_l176_176070


namespace no_positive_integer_n_for_perfect_squares_l176_176021

theorem no_positive_integer_n_for_perfect_squares :
  ∀ (n : ℕ), 0 < n → ¬ (∃ a b : ℤ, (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2) :=
by
  sorry

end no_positive_integer_n_for_perfect_squares_l176_176021


namespace find_m_l176_176272

theorem find_m (x m : ℝ) (h_eq : (x + m) / (x - 2) + 1 / (2 - x) = 3) (h_root : x = 2) : m = -1 :=
by
  sorry

end find_m_l176_176272


namespace unit_cost_decreases_l176_176052

def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

theorem unit_cost_decreases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := 
by sorry


end unit_cost_decreases_l176_176052


namespace tom_roses_per_day_l176_176104

-- Define variables and conditions
def total_roses := 168
def days_in_week := 7
def dozen := 12

-- Theorem to prove
theorem tom_roses_per_day : (total_roses / dozen) / days_in_week = 2 :=
by
  -- The actual proof would go here, using the sorry placeholder
  sorry

end tom_roses_per_day_l176_176104


namespace angles_sum_540_l176_176117

theorem angles_sum_540 (p q r s : ℝ) (h1 : ∀ a, a + (180 - a) = 180)
  (h2 : ∀ a b, (180 - a) + (180 - b) = 360 - a - b)
  (h3 : ∀ p q r, (360 - p - q) + (180 - r) = 540 - p - q - r) :
  p + q + r + s = 540 :=
sorry

end angles_sum_540_l176_176117


namespace prob_draw_l176_176101

theorem prob_draw (p_not_losing p_winning p_drawing : ℝ) (h1 : p_not_losing = 0.6) (h2 : p_winning = 0.5) :
  p_drawing = 0.1 :=
by
  sorry

end prob_draw_l176_176101


namespace pawpaws_basket_l176_176717

variable (total_fruits mangoes pears lemons kiwis : ℕ)
variable (pawpaws : ℕ)

theorem pawpaws_basket
  (h1 : total_fruits = 58)
  (h2 : mangoes = 18)
  (h3 : pears = 10)
  (h4 : lemons = 9)
  (h5 : kiwis = 9)
  (h6 : total_fruits = mangoes + pears + lemons + kiwis + pawpaws) :
  pawpaws = 12 := by
  sorry

end pawpaws_basket_l176_176717


namespace completing_the_square_l176_176373

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end completing_the_square_l176_176373


namespace number_of_younger_employees_correct_l176_176834

noncomputable def total_employees : ℕ := 200
noncomputable def younger_employees : ℕ := 120
noncomputable def sample_size : ℕ := 25

def number_of_younger_employees_to_be_drawn (total younger sample : ℕ) : ℕ :=
  sample * younger / total

theorem number_of_younger_employees_correct :
  number_of_younger_employees_to_be_drawn total_employees younger_employees sample_size = 15 := by
  sorry

end number_of_younger_employees_correct_l176_176834


namespace complex_number_on_imaginary_axis_l176_176255

theorem complex_number_on_imaginary_axis (a : ℝ) 
(h : ∃ z : ℂ, z = (a^2 - 2 * a) + (a^2 - a - 2) * Complex.I ∧ z.re = 0) : 
a = 0 ∨ a = 2 :=
by
  sorry

end complex_number_on_imaginary_axis_l176_176255


namespace complement_of_angle_is_acute_l176_176606

theorem complement_of_angle_is_acute (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < 90) : 0 < 90 - θ ∧ 90 - θ < 90 :=
by sorry

end complement_of_angle_is_acute_l176_176606


namespace watermelons_left_to_be_sold_tomorrow_l176_176886

def initial_watermelons : ℕ := 10 * 12
def sold_yesterday : ℕ := initial_watermelons * 40 / 100
def remaining_after_yesterday : ℕ := initial_watermelons - sold_yesterday
def sold_today : ℕ := remaining_after_yesterday / 4
def remaining_after_today : ℕ := remaining_after_yesterday - sold_today

theorem watermelons_left_to_be_sold_tomorrow : remaining_after_today = 54 := 
by
  sorry

end watermelons_left_to_be_sold_tomorrow_l176_176886


namespace constant_term_in_expansion_is_neg_42_l176_176692

-- Define the general term formula for (x - 1/x)^8
def binomial_term (r : ℕ) : ℤ :=
  (Nat.choose 8 r) * (-1 : ℤ) ^ r

-- Define the constant term in the product expansion
def constant_term : ℤ := 
  binomial_term 4 - 2 * binomial_term 5 

-- Problem statement: Prove the constant term is -42
theorem constant_term_in_expansion_is_neg_42 :
  constant_term = -42 := 
sorry

end constant_term_in_expansion_is_neg_42_l176_176692


namespace only_positive_integer_cube_less_than_triple_l176_176718

theorem only_positive_integer_cube_less_than_triple (n : ℕ) (h : 0 < n ∧ n^3 < 3 * n) : n = 1 :=
sorry

end only_positive_integer_cube_less_than_triple_l176_176718


namespace fraction_simplification_l176_176624

theorem fraction_simplification (a b c : ℝ) :
  (4 * a^2 + 2 * c^2 - 4 * b^2 - 8 * b * c) / (3 * a^2 + 6 * a * c - 3 * c^2 - 6 * a * b) =
  (4 / 3) * ((a - 2 * b + c) * (a - c)) / ((a - b + c) * (a - b - c)) :=
by
  sorry

end fraction_simplification_l176_176624


namespace arithmetic_sequence_zero_l176_176816

noncomputable def f (x : ℝ) : ℝ :=
  0.3 ^ x - Real.log x / Real.log 2

theorem arithmetic_sequence_zero (a b c x : ℝ) (h_seq : a < b ∧ b < c) (h_pos_diff : b - a = c - b)
    (h_f_product : f a * f b * f c > 0) (h_fx_zero : f x = 0) : ¬ (x < a) :=
by
  sorry

end arithmetic_sequence_zero_l176_176816


namespace fibonacci_expression_equality_l176_176097

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Statement to be proven
theorem fibonacci_expression_equality :
  (fibonacci 0 * fibonacci 2 + fibonacci 1 * fibonacci 3 + fibonacci 2 * fibonacci 4 +
  fibonacci 3 * fibonacci 5 + fibonacci 4 * fibonacci 6 + fibonacci 5 * fibonacci 7)
  - (fibonacci 1 ^ 2 + fibonacci 2 ^ 2 + fibonacci 3 ^ 2 + fibonacci 4 ^ 2 + fibonacci 5 ^ 2 + fibonacci 6 ^ 2)
  = 0 :=
by
  sorry

end fibonacci_expression_equality_l176_176097


namespace ratio_of_ducks_l176_176287

theorem ratio_of_ducks (lily_ducks lily_geese rayden_geese rayden_ducks : ℕ) 
  (h1 : lily_ducks = 20) 
  (h2 : lily_geese = 10) 
  (h3 : rayden_geese = 4 * lily_geese) 
  (h4 : rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70) : 
  rayden_ducks / lily_ducks = 3 :=
by
  sorry

end ratio_of_ducks_l176_176287


namespace triangle_perimeter_l176_176171

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being a root of the equation
    x^2 - 8x + 12 = 0, the perimeter of the triangle is 13. --/
theorem triangle_perimeter
  (a b : ℕ) 
  (ha : a = 2) 
  (hb : b = 5)
  (c : ℕ)
  (h_c_root : c * c - 8 * c + 12 = 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 13 := 
sorry

end triangle_perimeter_l176_176171


namespace greatest_expression_l176_176737

theorem greatest_expression 
  (x1 x2 y1 y2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : x1 < x2) 
  (hx12 : x1 + x2 = 1) 
  (hy1 : 0 < y1) 
  (hy2 : y1 < y2) 
  (hy12 : y1 + y2 = 1) : 
  x1 * y1 + x2 * y2 > max (x1 * x2 + y1 * y2) (max (x1 * y2 + x2 * y1) (1/2)) := 
sorry

end greatest_expression_l176_176737


namespace width_minimizes_fencing_l176_176482

-- Define the conditions for the problem
def garden_area_cond (w : ℝ) : Prop :=
  w * (w + 10) ≥ 150

-- Define the main statement to prove
theorem width_minimizes_fencing (w : ℝ) (h : w ≥ 0) : garden_area_cond w → w = 10 :=
  by
  sorry

end width_minimizes_fencing_l176_176482


namespace minimum_reciprocal_sum_of_roots_l176_176598

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := 2 * x^2 + b * x + c

theorem minimum_reciprocal_sum_of_roots {b c : ℝ} {x1 x2 : ℝ} 
  (h1: f (-10) b c = f 12 b c)
  (h2: f x1 b c = 0)
  (h3: f x2 b c = 0)
  (h4: 0 < x1)
  (h5: 0 < x2)
  (h6: x1 + x2 = 2) :
  (1 / x1 + 1 / x2) = 2 :=
sorry

end minimum_reciprocal_sum_of_roots_l176_176598


namespace no_mult_of_5_end_in_2_l176_176312

theorem no_mult_of_5_end_in_2 (n : ℕ) : n < 500 → ∃ k, n = 5 * k → (n % 10 = 2) = false :=
by
  sorry

end no_mult_of_5_end_in_2_l176_176312


namespace union_of_M_and_N_l176_176659

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} :=
by
  sorry

end union_of_M_and_N_l176_176659


namespace Sam_has_38_dollars_l176_176399

theorem Sam_has_38_dollars (total_money erica_money sam_money : ℕ) 
  (h1 : total_money = 91)
  (h2 : erica_money = 53) 
  (h3 : total_money = erica_money + sam_money) : 
  sam_money = 38 := 
by 
  sorry

end Sam_has_38_dollars_l176_176399


namespace range_of_a_l176_176891

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 4 ≤ a := 
sorry

end range_of_a_l176_176891


namespace number_of_newborn_members_in_group_l176_176300

noncomputable def N : ℝ :=
  let p_death := 1 / 10
  let p_survive := 1 - p_death
  let prob_survive_3_months := p_survive * p_survive * p_survive
  218.7 / prob_survive_3_months

theorem number_of_newborn_members_in_group : N = 300 := by
  sorry

end number_of_newborn_members_in_group_l176_176300


namespace compare_f_values_l176_176466

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * Real.cos x

theorem compare_f_values :
  f 0 < f (-1 / 3) ∧ f (-1 / 3) < f (2 / 5) :=
by
  sorry

end compare_f_values_l176_176466


namespace math_proof_problem_l176_176814

-- Definitions
def PropA : Prop := ¬ (∀ n : ℤ, (3 ∣ n → ¬ (n % 2 = 1)))
def PropB : Prop := ¬ (¬ (∃ x : ℝ, x^2 + x + 1 ≥ 0))
def PropC : Prop := ∀ (α β : ℝ) (k : ℤ), α = k * Real.pi + β ↔ Real.tan α = Real.tan β
def PropD : Prop := ∀ (a b : ℝ), a ≠ 0 → a * b ≠ 0 → b ≠ 0

def correct_options : Prop := PropA ∧ PropC ∧ ¬PropB ∧ PropD

-- The theorem to be proven
theorem math_proof_problem : correct_options :=
by
  sorry

end math_proof_problem_l176_176814


namespace calculate_discount_l176_176417

theorem calculate_discount
  (original_cost : ℝ)
  (amount_spent : ℝ)
  (h1 : original_cost = 35.00)
  (h2 : amount_spent = 18.00) :
  original_cost - amount_spent = 17.00 :=
by
  sorry

end calculate_discount_l176_176417


namespace triangle_angle_sum_l176_176443

open scoped Real

theorem triangle_angle_sum (A B C : ℝ) 
  (hA : A = 25) (hB : B = 55) : C = 100 :=
by
  have h1 : A + B + C = 180 := sorry
  rw [hA, hB] at h1
  linarith

end triangle_angle_sum_l176_176443


namespace arithmetic_sequence_values_l176_176729

noncomputable def common_difference (a₁ a₂ : ℕ) : ℕ := (a₂ - a₁) / 2

theorem arithmetic_sequence_values (x y z d: ℕ) 
    (h₁: d = common_difference 7 11) 
    (h₂: x = 7 + d) 
    (h₃: y = 11 + d) 
    (h₄: z = y + d): 
    x = 9 ∧ y = 13 ∧ z = 15 :=
by {
  sorry
}

end arithmetic_sequence_values_l176_176729


namespace sqrt_c_is_202_l176_176303

theorem sqrt_c_is_202 (a b c : ℝ) (h1 : a + b = -2020) (h2 : a * b = c) (h3 : a / b + b / a = 98) : 
  Real.sqrt c = 202 :=
by
  sorry

end sqrt_c_is_202_l176_176303


namespace vector_dot_product_identity_l176_176517

-- Define the vectors a, b, and c in ℝ²
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-3, 1)

-- Define vector addition and dot product in ℝ²
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that c · (a + b) = 9
theorem vector_dot_product_identity : dot_product c (vector_add a b) = 9 := 
by 
sorry

end vector_dot_product_identity_l176_176517


namespace basketball_team_win_requirement_l176_176174

noncomputable def basketball_win_percentage_goal (games_played_so_far games_won_so_far games_remaining win_percentage_goal : ℕ) : ℕ :=
  let total_games := games_played_so_far + games_remaining
  let required_wins := (win_percentage_goal * total_games) / 100
  required_wins - games_won_so_far

theorem basketball_team_win_requirement :
  basketball_win_percentage_goal 60 45 50 75 = 38 := 
by
  sorry

end basketball_team_win_requirement_l176_176174


namespace min_value_of_y_l176_176676

theorem min_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (∃ y : ℝ, y = 1 / a + 4 / b ∧ (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ y)) ∧ 
  (∀ y : ℝ, y = 1 / a + 4 / b → y ≥ 9) :=
sorry

end min_value_of_y_l176_176676


namespace remainder_of_product_l176_176509

theorem remainder_of_product (a b c : ℕ) (h₁ : a % 7 = 3) (h₂ : b % 7 = 4) (h₃ : c % 7 = 5) :
  (a * b * c) % 7 = 4 :=
by
  sorry

end remainder_of_product_l176_176509


namespace order_of_exponentials_l176_176912

theorem order_of_exponentials :
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  a < d ∧ d < b ∧ b < c :=
by
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  sorry

end order_of_exponentials_l176_176912


namespace exists_solution_in_interval_l176_176976

noncomputable def f (x : ℝ) : ℝ := x^3 - 2^x

theorem exists_solution_in_interval : ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f x = 0 :=
by {
  -- Use the Intermediate Value Theorem, given f is continuous on [1, 2]
  sorry
}

end exists_solution_in_interval_l176_176976


namespace number_subtracted_l176_176662

theorem number_subtracted (x y : ℕ) (h₁ : x = 48) (h₂ : 5 * x - y = 102) : y = 138 :=
by
  rw [h₁] at h₂
  sorry

end number_subtracted_l176_176662


namespace value_of_a_minus_b_l176_176025

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  (a - b = -1) ∨ (a - b = -7) :=
by
  sorry

end value_of_a_minus_b_l176_176025


namespace ned_short_sleeve_shirts_l176_176075

theorem ned_short_sleeve_shirts (washed_shirts not_washed_shirts long_sleeve_shirts total_shirts : ℕ)
  (h1 : washed_shirts = 29) (h2 : not_washed_shirts = 1) (h3 : long_sleeve_shirts = 21)
  (h4 : total_shirts = washed_shirts + not_washed_shirts) :
  total_shirts - long_sleeve_shirts = 9 :=
by
  sorry

end ned_short_sleeve_shirts_l176_176075


namespace carpet_dimensions_l176_176458

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end carpet_dimensions_l176_176458


namespace bowls_remaining_l176_176771

-- Definitions based on conditions.
def initial_collection : ℕ := 70
def reward_per_10_bowls : ℕ := 2
def total_customers : ℕ := 20
def customers_bought_20 : ℕ := total_customers / 2
def bowls_bought_per_customer : ℕ := 20
def total_bowls_bought : ℕ := customers_bought_20 * bowls_bought_per_customer
def reward_sets : ℕ := total_bowls_bought / 10
def total_reward_given : ℕ := reward_sets * reward_per_10_bowls

-- Theorem statement to be proved.
theorem bowls_remaining : initial_collection - total_reward_given = 30 :=
by
  sorry

end bowls_remaining_l176_176771


namespace range_quadratic_function_l176_176971

theorem range_quadratic_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = x^2 - 2 * x + 5 ↔ y ∈ Set.Ici 4 :=
by 
  sorry

end range_quadratic_function_l176_176971


namespace triangle_sets_l176_176050

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end triangle_sets_l176_176050


namespace tom_can_go_on_three_rides_l176_176357

def rides_possible (total_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / tickets_per_ride

theorem tom_can_go_on_three_rides :
  rides_possible 40 28 4 = 3 :=
by
  -- proof goes here
  sorry

end tom_can_go_on_three_rides_l176_176357


namespace no_integer_solution_exists_l176_176777

theorem no_integer_solution_exists : ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8 * t - 1 := 
by sorry

end no_integer_solution_exists_l176_176777


namespace distance_A_to_B_l176_176091

theorem distance_A_to_B : 
  ∀ (D : ℕ),
    let boat_speed_with_wind := 21
    let boat_speed_against_wind := 17
    let time_for_round_trip := 7
    let stream_speed_ab := 3
    let stream_speed_ba := 2
    let effective_speed_ab := boat_speed_with_wind + stream_speed_ab
    let effective_speed_ba := boat_speed_against_wind - stream_speed_ba
    D / effective_speed_ab + D / effective_speed_ba = time_for_round_trip →
    D = 65 :=
by
  sorry

end distance_A_to_B_l176_176091


namespace smallest_nonfactor_product_of_factors_of_48_l176_176191

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l176_176191


namespace number_of_players_per_game_l176_176793

def total_players : ℕ := 50
def total_games : ℕ := 1225

-- If each player plays exactly one game with each of the other players,
-- there are C(total_players, 2) = total_games games.
theorem number_of_players_per_game : ∃ k : ℕ, k = 2 ∧ (total_players * (total_players - 1)) / 2 = total_games := 
  sorry

end number_of_players_per_game_l176_176793


namespace quiz_score_difference_l176_176764

theorem quiz_score_difference :
  let percentage_70 := 0.10
  let percentage_80 := 0.35
  let percentage_90 := 0.30
  let percentage_100 := 0.25
  let mean_score := (percentage_70 * 70) + (percentage_80 * 80) + (percentage_90 * 90) + (percentage_100 * 100)
  let median_score := 90
  mean_score = 87 → median_score - mean_score = 3 :=
by
  sorry

end quiz_score_difference_l176_176764


namespace coins_to_rubles_l176_176242

theorem coins_to_rubles (a1 a2 a3 a4 a5 a6 a7 k m : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  m * 100 = k :=
by sorry

end coins_to_rubles_l176_176242


namespace expression_divisible_by_16_l176_176902

theorem expression_divisible_by_16 (m n : ℤ) : 
  ∃ k : ℤ, (5 * m + 3 * n + 1)^5 * (3 * m + n + 4)^4 = 16 * k :=
sorry

end expression_divisible_by_16_l176_176902


namespace pointC_on_same_side_as_point1_l176_176997

-- Definitions of points and the line equation
def is_on_same_side (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : Prop :=
  (line p1 > 0) ↔ (line p2 > 0)

def line_eq (p : ℝ × ℝ) : ℝ := p.1 + p.2 - 1

def point1 : ℝ × ℝ := (1, 2)
def pointC : ℝ × ℝ := (-1, 3)

-- Theorem to prove the equivalence
theorem pointC_on_same_side_as_point1 :
  is_on_same_side point1 pointC line_eq :=
sorry

end pointC_on_same_side_as_point1_l176_176997


namespace fruit_basket_count_l176_176801

theorem fruit_basket_count :
  let apples := 6
  let oranges := 8
  let min_apples := 2
  let min_fruits := 1
  (0 <= oranges ∧ oranges <= 8) ∧ (min_apples <= apples ∧ apples <= 6) ∧ (min_fruits <= (apples + oranges)) →
  (5 * 9 = 45) :=
by
  intro h
  sorry

end fruit_basket_count_l176_176801


namespace total_present_ages_l176_176907

variables (P Q : ℕ)

theorem total_present_ages :
  (P - 8 = (Q - 8) / 2) ∧ (P * 4 = Q * 3) → (P + Q = 28) :=
by
  sorry

end total_present_ages_l176_176907


namespace sector_arc_length_l176_176782

theorem sector_arc_length (n : ℝ) (r : ℝ) (l : ℝ) (h1 : n = 90) (h2 : r = 3) (h3 : l = (n * Real.pi * r) / 180) :
  l = (3 / 2) * Real.pi := by
  rw [h1, h2] at h3
  sorry

end sector_arc_length_l176_176782


namespace distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l176_176208

variable (m : ℝ)

-- Part 1: Prove that if the quadratic equation has two distinct real roots, then m < 13/4.
theorem distinct_real_roots_iff_m_lt_13_over_4 (h : (3 * 3 - 4 * (m - 1)) > 0) : m < 13 / 4 := 
by
  sorry

-- Part 2: Prove that if the quadratic equation has two equal real roots, then the root is 3/2.
theorem equal_real_roots_root_eq_3_over_2 (h : (3 * 3 - 4 * (m - 1)) = 0) : m = 13 / 4 ∧ ∀ x, (x^2 + 3 * x + (13/4 - 1) = 0) → x = 3 / 2 :=
by
  sorry

end distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l176_176208


namespace hans_deposit_l176_176139

noncomputable def calculate_deposit : ℝ :=
  let flat_fee := 30
  let kid_deposit := 2 * 3
  let adult_deposit := 8 * 6
  let senior_deposit := 5 * 4
  let student_deposit := 3 * 4.5
  let employee_deposit := 2 * 2.5
  let total_deposit_before_service := flat_fee + kid_deposit + adult_deposit + senior_deposit + student_deposit + employee_deposit
  let service_charge := total_deposit_before_service * 0.05
  total_deposit_before_service + service_charge

theorem hans_deposit : calculate_deposit = 128.63 :=
by
  sorry

end hans_deposit_l176_176139


namespace alma_score_l176_176316

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end alma_score_l176_176316


namespace books_on_each_shelf_l176_176545

theorem books_on_each_shelf (M P x : ℕ) (h1 : 3 * M + 5 * P = 72) (h2 : M = x) (h3 : P = x) : x = 9 :=
by
  sorry

end books_on_each_shelf_l176_176545


namespace value_of_p_h_3_l176_176786

-- Define the functions h and p
def h (x : ℝ) : ℝ := 4 * x + 5
def p (x : ℝ) : ℝ := 6 * x - 11

-- Statement to prove
theorem value_of_p_h_3 : p (h 3) = 91 := sorry

end value_of_p_h_3_l176_176786


namespace five_digit_number_count_l176_176385

theorem five_digit_number_count : ∃ n, n = 1134 ∧ ∀ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧ 
  (a < b ∧ b < c ∧ c > d ∧ d > e) → n = 1134 :=
by 
  sorry

end five_digit_number_count_l176_176385


namespace line_intersects_ellipse_all_possible_slopes_l176_176630

theorem line_intersects_ellipse_all_possible_slopes (m : ℝ) :
  m^2 ≥ 1 / 5 ↔ ∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100) := sorry

end line_intersects_ellipse_all_possible_slopes_l176_176630


namespace contingency_fund_allocation_l176_176405

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l176_176405


namespace from20To25_l176_176752

def canObtain25 (start : ℕ) : Prop :=
  ∃ (steps : ℕ → ℕ), steps 0 = start ∧ (∃ n, steps n = 25) ∧ 
  (∀ i, steps (i+1) = (steps i * 2) ∨ (steps (i+1) = steps i / 10))

theorem from20To25 : canObtain25 20 :=
sorry

end from20To25_l176_176752


namespace greatest_integer_l176_176709

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 2) (h3 : ∃ l : ℕ, n = 8 * l - 4) : n = 124 :=
by
  sorry

end greatest_integer_l176_176709


namespace rectangles_on_grid_l176_176571

-- Define the grid dimensions
def m := 3
def n := 2

-- Define a function to count the total number of rectangles formed by the grid.
def count_rectangles (m n : ℕ) : ℕ := 
  (m * (m - 1) / 2 + n * (n - 1) / 2) * (n * (n - 1) / 2 + m * (m - 1) / 2) 

-- State the theorem we need to prove
theorem rectangles_on_grid : count_rectangles m n = 14 :=
  sorry

end rectangles_on_grid_l176_176571


namespace ratio_of_values_l176_176843

-- Define the geometric sequence with first term and common ratio
def geom_seq_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n-1)

-- Define the sum of the first n terms of the geometric sequence
def geom_seq_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

-- Sum of the first n terms for given sequence
noncomputable def S_n (n : ℕ) : ℚ :=
  geom_seq_sum (3/2) (-1/2) n

-- Define the function f(t) = t - 1/t
def f (t : ℚ) : ℚ := t - 1 / t

-- Define the maximum and minimum values of f(S_n) and their ratio
noncomputable def ratio_max_min_values : ℚ :=
  let max_val := f (3/2)
  let min_val := f (3/4)
  max_val / min_val

-- The theorem to prove the ratio of the maximum and minimum values
theorem ratio_of_values :
  ratio_max_min_values = -10/7 := by
  sorry

end ratio_of_values_l176_176843


namespace factor_of_quadratic_l176_176763

theorem factor_of_quadratic (m : ℝ) : (∀ x, (x + 6) * (x + a) = x ^ 2 - mx - 42) → m = 1 :=
by sorry

end factor_of_quadratic_l176_176763


namespace square_perimeter_is_44_8_l176_176585

noncomputable def perimeter_of_congruent_rectangles_division (s : ℝ) (P : ℝ) : ℝ :=
  let rectangle_perimeter := 2 * (s + s / 4)
  if rectangle_perimeter = P then 4 * s else 0

theorem square_perimeter_is_44_8 :
  ∀ (s : ℝ) (P : ℝ), P = 28 → 4 * s = 44.8 → perimeter_of_congruent_rectangles_division s P = 44.8 :=
by intros s P h1 h2
   sorry

end square_perimeter_is_44_8_l176_176585


namespace intersection_complement_l176_176644

open Set

noncomputable def U : Set ℝ := univ

def A : Set ℝ := {x | x^2 - 2 * x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_complement (x : ℝ) :
  x ∈ (A ∩ (U \ B)) ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end intersection_complement_l176_176644


namespace constant_a_value_l176_176720

theorem constant_a_value (S : ℕ → ℝ)
  (a : ℝ)
  (h : ∀ n : ℕ, S n = 3 ^ (n + 1) + a) :
  a = -3 :=
sorry

end constant_a_value_l176_176720


namespace five_letter_word_with_at_least_one_consonant_l176_176465

def letter_set : Set Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Set Char := {'B', 'C', 'D', 'F'}
def vowels : Set Char := {'A', 'E'}

-- Calculate the total number of 5-letter words using the letter set
def total_words : ℕ := 6^5

-- Calculate the number of 5-letter words using only vowels
def vowel_only_words : ℕ := 2^5

-- Number of 5-letter words with at least one consonant
def words_with_consonant : ℕ := total_words - vowel_only_words

theorem five_letter_word_with_at_least_one_consonant :
  words_with_consonant = 7744 :=
by
  sorry

end five_letter_word_with_at_least_one_consonant_l176_176465


namespace total_apples_correct_l176_176189

variable (X : ℕ)

def Sarah_apples : ℕ := X

def Jackie_apples : ℕ := 2 * Sarah_apples X

def Adam_apples : ℕ := Jackie_apples X + 5

def total_apples : ℕ := Sarah_apples X + Jackie_apples X + Adam_apples X

theorem total_apples_correct : total_apples X = 5 * X + 5 := by
  sorry

end total_apples_correct_l176_176189


namespace cookies_on_ninth_plate_l176_176849

-- Define the geometric sequence
def cookies_on_plate (n : ℕ) : ℕ :=
  2 * 2^(n - 1)

-- State the theorem
theorem cookies_on_ninth_plate : cookies_on_plate 9 = 512 :=
by
  sorry

end cookies_on_ninth_plate_l176_176849


namespace committee_count_8_choose_4_l176_176409

theorem committee_count_8_choose_4 : (Nat.choose 8 4) = 70 :=
  by
  -- proof skipped
  sorry

end committee_count_8_choose_4_l176_176409


namespace find_S5_l176_176185

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a 1 + n * d
axiom sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_S5 (h : a 1 + a 3 + a 5 = 3) : S 5 = 5 :=
by
  sorry

end find_S5_l176_176185


namespace find_third_number_l176_176035

-- Definitions based on given conditions
def A : ℕ := 200
def C : ℕ := 100
def B : ℕ := 2 * C

-- The condition that the sum of A, B, and C is 500
def sum_condition : Prop := A + B + C = 500

-- The proof statement
theorem find_third_number : sum_condition → C = 100 := 
by
  have h1 : A = 200 := rfl
  have h2 : B = 2 * C := rfl
  have h3 : A + B + C = 500 := sorry
  sorry

end find_third_number_l176_176035


namespace real_solution_four_unknowns_l176_176582

theorem real_solution_four_unknowns (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x * (y + z + t) ↔ (x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) :=
by
  sorry

end real_solution_four_unknowns_l176_176582


namespace union_of_A_and_B_l176_176183

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
noncomputable def B : Set ℝ := {x : ℝ | 1 < x }

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x} :=
by
  sorry

end union_of_A_and_B_l176_176183


namespace ethanol_in_tank_l176_176768

theorem ethanol_in_tank (capacity fuel_a fuel_b : ℝ)
  (ethanol_a ethanol_b : ℝ)
  (h1 : capacity = 218)
  (h2 : fuel_a = 122)
  (h3 : fuel_b = capacity - fuel_a)
  (h4 : ethanol_a = 0.12)
  (h5 : ethanol_b = 0.16) :
  fuel_a * ethanol_a + fuel_b * ethanol_b = 30 := 
by {
  sorry
}

end ethanol_in_tank_l176_176768


namespace geometric_sequence_a7_value_l176_176436

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a7_value (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, 0 < a n) →
  (geometric_sequence a r) →
  (S 4 = 3 * S 2) →
  (a 3 = 2) →
  (S n = a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3) →
  a 7 = 8 :=
by
  sorry

end geometric_sequence_a7_value_l176_176436


namespace union_of_sets_l176_176400

variable (A : Set ℤ) (B : Set ℤ)

theorem union_of_sets (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l176_176400


namespace reduced_population_l176_176390

theorem reduced_population (initial_population : ℕ)
  (percentage_died : ℝ)
  (percentage_left : ℝ)
  (h_initial : initial_population = 8515)
  (h_died : percentage_died = 0.10)
  (h_left : percentage_left = 0.15) :
  ((initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ)) - 
   (⌊percentage_left * (initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ))⌋₊ : ℕ)) = 6515 :=
by
  sorry

end reduced_population_l176_176390


namespace money_collected_l176_176226

theorem money_collected
  (households_per_day : ℕ)
  (days : ℕ)
  (half_give_money : ℕ → ℕ)
  (total_money_collected : ℕ)
  (households_give_money : ℕ) :
  households_per_day = 20 →  
  days = 5 →
  total_money_collected = 2000 →
  half_give_money (households_per_day * days) = (households_per_day * days) / 2 →
  households_give_money = (households_per_day * days) / 2 →
  total_money_collected / households_give_money = 40
:= sorry

end money_collected_l176_176226


namespace water_outflow_time_l176_176150

theorem water_outflow_time (H R : ℝ) (flow_rate : ℝ → ℝ)
  (h_initial : ℝ) (t_initial : ℝ) (empty_height : ℝ) :
  H = 12 →
  R = 3 →
  (∀ h, flow_rate h = -h) →
  h_initial = 12 →
  t_initial = 0 →
  empty_height = 0 →
  ∃ t, t = (72 : ℝ) * π / 16 :=
by
  intros hL R_eq flow_rate_eq h_initial_eq t_initial_eq empty_height_eq
  sorry

end water_outflow_time_l176_176150


namespace rate_of_pipe_B_l176_176048

-- Definitions based on conditions
def tank_capacity : ℕ := 850
def pipe_A_rate : ℕ := 40
def pipe_C_rate : ℕ := 20
def cycle_time : ℕ := 3
def full_time : ℕ := 51

-- Prove that the rate of pipe B is 30 liters per minute
theorem rate_of_pipe_B (B : ℕ) : 
  (17 * (B + 20) = 850) → B = 30 := 
by 
  introv h1
  sorry

end rate_of_pipe_B_l176_176048


namespace allen_total_blocks_l176_176378

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end allen_total_blocks_l176_176378


namespace find_a_b_k_l176_176244

noncomputable def a (k : ℕ) : ℕ := if h : k = 9 then 243 else sorry
noncomputable def b (k : ℕ) : ℕ := if h : k = 9 then 3 else sorry

theorem find_a_b_k (a b k : ℕ) (hb : b = 3) (ha : a = 243) (hk : k = 9)
  (h1 : a * b = k^3) (h2 : a / b = k^2) (h3 : 100 ≤ a * b ∧ a * b < 1000) :
  a = 243 ∧ b = 3 ∧ k = 9 :=
by 
  sorry

end find_a_b_k_l176_176244


namespace fraction_evaluation_l176_176074

theorem fraction_evaluation : (3 / 8 : ℚ) + 7 / 12 - 2 / 9 = 53 / 72 := by
  sorry

end fraction_evaluation_l176_176074


namespace emily_selects_green_apples_l176_176927

theorem emily_selects_green_apples :
  let total_apples := 10
  let red_apples := 6
  let green_apples := 4
  let selected_apples := 3
  let total_combinations := Nat.choose total_apples selected_apples
  let green_combinations := Nat.choose green_apples selected_apples
  (green_combinations / total_combinations : ℚ) = 1 / 30 :=
by
  sorry

end emily_selects_green_apples_l176_176927


namespace lines_in_4_by_4_grid_l176_176759

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l176_176759


namespace find_ice_cream_cost_l176_176761

def cost_of_ice_cream (total_paid cost_chapati cost_rice cost_vegetable : ℕ) (n_chapatis n_rice n_vegetables n_ice_cream : ℕ) : ℕ :=
  (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetables * cost_vegetable)) / n_ice_cream

theorem find_ice_cream_cost :
  let total_paid := 1051
  let cost_chapati := 6
  let cost_rice := 45
  let cost_vegetable := 70
  let n_chapatis := 16
  let n_rice := 5
  let n_vegetables := 7
  let n_ice_cream := 6
  cost_of_ice_cream total_paid cost_chapati cost_rice cost_vegetable n_chapatis n_rice n_vegetables n_ice_cream = 40 :=
by
  sorry

end find_ice_cream_cost_l176_176761


namespace xyz_sum_48_l176_176155

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l176_176155


namespace john_candies_correct_l176_176773

variable (Bob_candies : ℕ) (Mary_candies : ℕ)
          (Sue_candies : ℕ) (Sam_candies : ℕ)
          (Total_candies : ℕ) (John_candies : ℕ)

axiom bob_has : Bob_candies = 10
axiom mary_has : Mary_candies = 5
axiom sue_has : Sue_candies = 20
axiom sam_has : Sam_candies = 10
axiom total_has : Total_candies = 50

theorem john_candies_correct : 
  Bob_candies + Mary_candies + Sue_candies + Sam_candies + John_candies = Total_candies → John_candies = 5 := by
sorry

end john_candies_correct_l176_176773


namespace exists_n_for_all_k_l176_176105

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 5^k ∣ (n^2 + 1) :=
sorry

end exists_n_for_all_k_l176_176105


namespace aniyah_more_candles_l176_176602

theorem aniyah_more_candles (x : ℝ) (h1 : 4 + 4 * x = 14) : x = 2.5 :=
sorry

end aniyah_more_candles_l176_176602


namespace three_x_squared_y_squared_eq_588_l176_176024

theorem three_x_squared_y_squared_eq_588 (x y : ℤ) 
  (h : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 :=
sorry

end three_x_squared_y_squared_eq_588_l176_176024


namespace braxton_total_earnings_l176_176006

-- Definitions of the given problem conditions
def students_ashwood : ℕ := 9
def days_ashwood : ℕ := 4
def students_braxton : ℕ := 6
def days_braxton : ℕ := 7
def students_cedar : ℕ := 8
def days_cedar : ℕ := 6

def total_payment : ℕ := 1080
def daily_wage_per_student : ℚ := total_payment / ((students_ashwood * days_ashwood) + 
                                                   (students_braxton * days_braxton) + 
                                                   (students_cedar * days_cedar))

-- The statement to be proven
theorem braxton_total_earnings :
  (students_braxton * days_braxton * daily_wage_per_student) = 360 := 
by
  sorry -- proof goes here

end braxton_total_earnings_l176_176006


namespace negation_of_proposition_l176_176474

theorem negation_of_proposition (a b : ℝ) :
  ¬(a > b → 2 * a > 2 * b) ↔ (a ≤ b → 2 * a ≤ 2 * b) :=
by
  sorry

end negation_of_proposition_l176_176474


namespace T_perimeter_l176_176440

theorem T_perimeter (l w : ℝ) (h1 : l = 4) (h2 : w = 2) :
  let rect_perimeter := 2 * l + 2 * w
  let overlap := 2 * w
  2 * rect_perimeter - overlap = 20 :=
by
  -- Proof will be added here
  sorry

end T_perimeter_l176_176440


namespace seq_sum_difference_l176_176276

-- Define the sequences
def seq1 : List ℕ := List.range 93 |> List.map (λ n => 2001 + n)
def seq2 : List ℕ := List.range 93 |> List.map (λ n => 301 + n)

-- Define the sum of the sequences
def sum_seq1 : ℕ := seq1.sum
def sum_seq2 : ℕ := seq2.sum

-- Define the difference between the sums of the sequences
def diff_seq_sum : ℕ := sum_seq1 - sum_seq2

-- Lean statement to prove the difference equals 158100
theorem seq_sum_difference : diff_seq_sum = 158100 := by
  sorry

end seq_sum_difference_l176_176276


namespace option_C_correct_l176_176666

theorem option_C_correct (x : ℝ) (hx : 0 < x) : x + 1 / x ≥ 2 :=
sorry

end option_C_correct_l176_176666


namespace ratio_x_y_z_l176_176221

variables (x y z : ℝ)

theorem ratio_x_y_z (h1 : 0.60 * x = 0.30 * y) 
                    (h2 : 0.80 * z = 0.40 * x) 
                    (h3 : z = 2 * y) : 
                    x / y = 4 ∧ y / y = 1 ∧ z / y = 2 :=
by
  sorry

end ratio_x_y_z_l176_176221


namespace polygons_after_cuts_l176_176057

theorem polygons_after_cuts (initial_polygons : ℕ) (cuts : ℕ) 
  (initial_vertices : ℕ) (max_vertices_added_per_cut : ℕ) :
  (initial_polygons = 10) →
  (cuts = 51) →
  (initial_vertices = 100) →
  (max_vertices_added_per_cut = 4) →
  ∃ p, (p < 5 ∧ p ≥ 3) :=
by
  intros h_initial_polygons h_cuts h_initial_vertices h_max_vertices_added_per_cut
  -- proof steps would go here
  sorry

end polygons_after_cuts_l176_176057


namespace simplify_fractions_l176_176823

theorem simplify_fractions :
  (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 :=
by
  sorry

end simplify_fractions_l176_176823


namespace range_of_m_l176_176438

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0
def q (m : ℝ) : Prop := ∃ y : ℝ, ∀ x : ℝ, (x^2)/(m-1) + y^2 = 1
def not_p (m : ℝ) : Prop := ¬ (p m)
def p_and_q (m : ℝ) : Prop := (p m) ∧ (q m)

theorem range_of_m (m : ℝ) : (¬ (not_p m) ∧ ¬ (p_and_q m)) → 1 < m ∧ m ≤ 2 :=
sorry

end range_of_m_l176_176438


namespace tangent_slope_is_four_l176_176012

-- Define the given curve and point
def curve (x : ℝ) : ℝ := 2 * x^2
def point : ℝ × ℝ := (1, 2)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the tangent slope at the given point
def tangent_slope_at_point : ℝ := curve_derivative 1

-- Prove that the tangent slope at point (1, 2) is 4
theorem tangent_slope_is_four : tangent_slope_at_point = 4 :=
by
  -- We state that the slope at x = 1 is 4
  sorry

end tangent_slope_is_four_l176_176012


namespace compute_b1c1_b2c2_b3c3_l176_176712

theorem compute_b1c1_b2c2_b3c3 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -1 :=
by
  sorry

end compute_b1c1_b2c2_b3c3_l176_176712


namespace half_angle_quadrant_l176_176591

theorem half_angle_quadrant
  (α : ℝ)
  (h1 : ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)
  (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
  ∃ k : ℤ, k * Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi * 3 / 4 ∧ Real.cos (α / 2) ≤ 0 := sorry

end half_angle_quadrant_l176_176591


namespace bisection_method_termination_condition_l176_176081

theorem bisection_method_termination_condition (x1 x2 : ℝ) (ε : ℝ) : Prop :=
  |x1 - x2| < ε

end bisection_method_termination_condition_l176_176081


namespace ap_number_of_terms_is_six_l176_176237

noncomputable def arithmetic_progression_number_of_terms (a d : ℕ) (n : ℕ) : Prop :=
  let odd_sum := (n / 2) * (2 * a + (n - 2) * d)
  let even_sum := (n / 2) * (2 * a + n * d)
  let last_term_condition := (n - 1) * d = 15
  n % 2 = 0 ∧ odd_sum = 30 ∧ even_sum = 36 ∧ last_term_condition

theorem ap_number_of_terms_is_six (a d n : ℕ) (h : arithmetic_progression_number_of_terms a d n) :
  n = 6 :=
by sorry

end ap_number_of_terms_is_six_l176_176237


namespace find_x_plus_y_l176_176485

theorem find_x_plus_y (x y : ℤ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) : x + y = -1 ∨ x + y = -5 :=
sorry

end find_x_plus_y_l176_176485


namespace multiple_of_12_l176_176678

theorem multiple_of_12 (x : ℤ) : 
  (7 * x - 3) % 12 = 0 ↔ (x % 12 = 9 ∨ x % 12 = 1029 % 12) :=
by
  sorry

end multiple_of_12_l176_176678


namespace current_inventory_l176_176784

noncomputable def initial_books : ℕ := 743
noncomputable def fiction_books : ℕ := 520
noncomputable def nonfiction_books : ℕ := 123
noncomputable def children_books : ℕ := 100

noncomputable def saturday_instore_sales : ℕ := 37
noncomputable def saturday_fiction_sales : ℕ := 15
noncomputable def saturday_nonfiction_sales : ℕ := 12
noncomputable def saturday_children_sales : ℕ := 10
noncomputable def saturday_online_sales : ℕ := 128

noncomputable def sunday_instore_multiplier : ℕ := 2
noncomputable def sunday_online_addition : ℕ := 34

noncomputable def new_shipment : ℕ := 160

noncomputable def current_books := 
  initial_books 
  - (saturday_instore_sales + saturday_online_sales)
  - (sunday_instore_multiplier * saturday_instore_sales + saturday_online_sales + sunday_online_addition)
  + new_shipment

theorem current_inventory : current_books = 502 := by
  sorry

end current_inventory_l176_176784


namespace conditional_probability_l176_176108

def prob_event_A : ℚ := 7 / 8 -- Probability of event A (at least one occurrence of tails)
def prob_event_AB : ℚ := 3 / 8 -- Probability of both events A and B happening (at least one occurrence of tails and exactly one occurrence of heads)

theorem conditional_probability (prob_A : ℚ) (prob_AB : ℚ) 
  (h1: prob_A = 7 / 8) (h2: prob_AB = 3 / 8) : 
  (prob_AB / prob_A) = 3 / 7 := 
by
  rw [h1, h2]
  norm_num

end conditional_probability_l176_176108


namespace molecular_weight_boric_acid_l176_176170

theorem molecular_weight_boric_acid :
  let H := 1.008  -- atomic weight of Hydrogen in g/mol
  let B := 10.81  -- atomic weight of Boron in g/mol
  let O := 16.00  -- atomic weight of Oxygen in g/mol
  let H3BO3 := 3 * H + B + 3 * O  -- molecular weight of H3BO3
  H3BO3 = 61.834 :=  -- correct molecular weight of H3BO3
by
  sorry

end molecular_weight_boric_acid_l176_176170


namespace correct_statement_is_B_l176_176011

-- Define integers and zero
def is_integer (n : ℤ) : Prop := True
def is_zero (n : ℤ) : Prop := n = 0

-- Define rational numbers
def is_rational (q : ℚ) : Prop := True

-- Positive and negative zero cannot co-exist
def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0

-- Statement A: Integers and negative integers are collectively referred to as integers.
def statement_A : Prop :=
  ∀ n : ℤ, (is_positive n ∨ is_negative n) ↔ is_integer n

-- Statement B: Integers and fractions are collectively referred to as rational numbers.
def statement_B : Prop :=
  ∀ q : ℚ, is_rational q

-- Statement C: Zero can be either a positive integer or a negative integer.
def statement_C : Prop :=
  ∀ n : ℤ, is_zero n → (is_positive n ∨ is_negative n)

-- Statement D: A rational number is either a positive number or a negative number.
def statement_D : Prop :=
  ∀ q : ℚ, (q ≠ 0 → (is_positive q.num ∨ is_negative q.num))

-- The problem is to prove that statement B is the only correct statement.
theorem correct_statement_is_B : statement_B ∧ ¬statement_A ∧ ¬statement_C ∧ ¬statement_D :=
by sorry

end correct_statement_is_B_l176_176011


namespace count_4_digit_numbers_divisible_by_13_l176_176018

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l176_176018


namespace smallest_positive_multiple_l176_176431

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end smallest_positive_multiple_l176_176431


namespace total_sections_after_admissions_l176_176326

theorem total_sections_after_admissions (S : ℕ) (h1 : (S * 24 + 24 = (S + 3) * 21)) :
  (S + 3) = 16 :=
  sorry

end total_sections_after_admissions_l176_176326


namespace total_cakes_served_today_l176_176020

def cakes_served_lunch : ℕ := 6
def cakes_served_dinner : ℕ := 9
def total_cakes_served (lunch cakes_served_dinner : ℕ) : ℕ :=
  lunch + cakes_served_dinner

theorem total_cakes_served_today : total_cakes_served cakes_served_lunch cakes_served_dinner = 15 := 
by
  sorry

end total_cakes_served_today_l176_176020


namespace sword_length_difference_l176_176892

def christopher_sword := 15.0
def jameson_sword := 2 * christopher_sword + 3
def june_sword := jameson_sword + 5
def average_length := (christopher_sword + jameson_sword + june_sword) / 3
def laura_sword := average_length - 0.1 * average_length
def difference := june_sword - laura_sword

theorem sword_length_difference :
  difference = 12.197 := 
sorry

end sword_length_difference_l176_176892


namespace total_work_stations_l176_176553

theorem total_work_stations (total_students : ℕ) (stations_for_2 : ℕ) (stations_for_3 : ℕ)
  (h1 : total_students = 38)
  (h2 : stations_for_2 = 10)
  (h3 : 20 + 3 * stations_for_3 = total_students) :
  stations_for_2 + stations_for_3 = 16 :=
by
  sorry

end total_work_stations_l176_176553


namespace sum_ratio_l176_176548

def arithmetic_sequence (a_1 d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n (a_1 d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a_1 + (n - 1) * d) / 2 -- sum of first n terms of arithmetic sequence

theorem sum_ratio (a_1 d : ℚ) (h : 13 * (a_1 + 6 * d) = 7 * (a_1 + 3 * d)) :
  S_n a_1 d 13 / S_n a_1 d 7 = 1 :=
by
  -- Proof omitted
  sorry

end sum_ratio_l176_176548


namespace points_per_member_correct_l176_176387

noncomputable def points_per_member (total_members: ℝ) (absent_members: ℝ) (total_points: ℝ) :=
  (total_points / (total_members - absent_members))

theorem points_per_member_correct:
  points_per_member 5.0 2.0 6.0 = 2.0 :=
by 
  sorry

end points_per_member_correct_l176_176387


namespace shaded_L_area_l176_176741

theorem shaded_L_area 
  (s₁ s₂ s₃ s₄ : ℕ)
  (hA : s₁ = 2)
  (hB : s₂ = 2)
  (hC : s₃ = 3)
  (hD : s₄ = 3)
  (side_ABC : ℕ := 6)
  (area_ABC : ℕ := side_ABC * side_ABC) : 
  area_ABC - (s₁ * s₁ + s₂ * s₂ + s₃ * s₃ + s₄ * s₄) = 10 :=
sorry

end shaded_L_area_l176_176741


namespace zero_descriptions_l176_176590

-- Defining the descriptions of zero satisfying the given conditions.
def description1 : String := "The number corresponding to the origin on the number line."
def description2 : String := "The number that represents nothing."
def description3 : String := "The number that, when multiplied by any other number, equals itself."

-- Lean statement to prove the validity of the descriptions.
theorem zero_descriptions : 
  description1 = "The number corresponding to the origin on the number line." ∧
  description2 = "The number that represents nothing." ∧
  description3 = "The number that, when multiplied by any other number, equals itself." :=
by
  -- Proof omitted
  sorry

end zero_descriptions_l176_176590


namespace jane_percentage_bread_to_treats_l176_176789

variable (T J_b W_b W_t : ℕ) (P : ℕ)

-- Conditions as stated
axiom h1 : J_b = (P * T) / 100
axiom h2 : W_t = T / 2
axiom h3 : W_b = 3 * W_t
axiom h4 : W_b = 90
axiom h5 : J_b + W_b + T + W_t = 225

theorem jane_percentage_bread_to_treats : P = 75 :=
by
-- Proof skeleton
sorry

end jane_percentage_bread_to_treats_l176_176789


namespace algebraic_expression_simplification_l176_176981

theorem algebraic_expression_simplification :
  0.25 * (-1 / 2) ^ (-4 : ℝ) - 4 / (Real.sqrt 5 - 1) ^ (0 : ℝ) - (1 / 16) ^ (-1 / 2 : ℝ) = -4 :=
by
  sorry

end algebraic_expression_simplification_l176_176981


namespace marnie_eats_chips_l176_176611

theorem marnie_eats_chips (total_chips : ℕ) (chips_first_batch : ℕ) (chips_second_batch : ℕ) (daily_chips : ℕ) (remaining_chips : ℕ) (total_days : ℕ) :
  total_chips = 100 →
  chips_first_batch = 5 →
  chips_second_batch = 5 →
  daily_chips = 10 →
  remaining_chips = total_chips - (chips_first_batch + chips_second_batch) →
  total_days = remaining_chips / daily_chips + 1 →
  total_days = 10 :=
by
  sorry

end marnie_eats_chips_l176_176611


namespace problem_I4_1_l176_176660

variable (A D E B C : Type) [Field A] [Field D] [Field E] [Field B] [Field C]
variable (AD DB DE BC : ℚ)
variable (a : ℚ)
variable (h1 : DE = BC) -- DE parallel to BC
variable (h2 : AD = 4)
variable (h3 : DB = 6)
variable (h4 : DE = 6)

theorem problem_I4_1 : a = 15 :=
  by
  sorry

end problem_I4_1_l176_176660


namespace dilute_lotion_l176_176850

/-- Determine the number of ounces of water needed to dilute 12 ounces
    of a shaving lotion containing 60% alcohol to a lotion containing 45% alcohol. -/
theorem dilute_lotion (W : ℝ) : 
  ∃ W, 12 * (0.60 : ℝ) / (12 + W) = 0.45 ∧ W = 4 :=
by
  use 4
  sorry

end dilute_lotion_l176_176850


namespace find_fx_l176_176870

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = 19 * x ^ 2 + 55 * x - 44) :
  ∀ x : ℝ, f x = 19 * x ^ 2 + 93 * x + 30 :=
by
  sorry

end find_fx_l176_176870


namespace find_ϕ_l176_176917

noncomputable def f (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem find_ϕ (ω ϕ : ℝ) (h1 : 0 < ω) (h2 : abs ϕ < Real.pi / 2) (h3 : ∀ x : ℝ, f ω ϕ (x + Real.pi / 6) = g ω x) 
  (h4 : 2 * Real.pi / ω = Real.pi) : ϕ = Real.pi / 3 :=
by sorry

end find_ϕ_l176_176917


namespace subsets_neither_A_nor_B_l176_176617

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end subsets_neither_A_nor_B_l176_176617


namespace find_base_l176_176513

theorem find_base (x y : ℕ) (b : ℕ) (h1 : 3 ^ x * b ^ y = 19683) (h2 : x - y = 9) (h3 : x = 9) : b = 1 := 
by
  sorry

end find_base_l176_176513


namespace solve_cubed_root_equation_l176_176404

theorem solve_cubed_root_equation :
  (∃ x : ℚ, (5 - 2 / x) ^ (1 / 3) = -3) ↔ x = 1 / 16 := 
by
  sorry

end solve_cubed_root_equation_l176_176404


namespace sum_of_edges_corners_faces_of_rectangular_prism_l176_176073

-- Definitions based on conditions
def rectangular_prism_edges := 12
def rectangular_prism_corners := 8
def rectangular_prism_faces := 6
def resulting_sum := rectangular_prism_edges + rectangular_prism_corners + rectangular_prism_faces

-- Statement we want to prove
theorem sum_of_edges_corners_faces_of_rectangular_prism :
  resulting_sum = 26 := 
by 
  sorry -- Placeholder for the proof

end sum_of_edges_corners_faces_of_rectangular_prism_l176_176073


namespace greatest_five_digit_common_multiple_l176_176679

theorem greatest_five_digit_common_multiple (n : ℕ) :
  (n % 18 = 0) ∧ (10000 ≤ n) ∧ (n ≤ 99999) → n = 99990 :=
by
  sorry

end greatest_five_digit_common_multiple_l176_176679


namespace problem_proof_l176_176055

theorem problem_proof (p : ℕ) (hodd : p % 2 = 1) (hgt : p > 3):
  ((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 4) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p + 1) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 3) :=
by
  sorry

end problem_proof_l176_176055


namespace mean_score_l176_176314

theorem mean_score (M SD : ℝ) (h₁ : 58 = M - 2 * SD) (h₂ : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l176_176314


namespace circle_radius_squared_l176_176672

theorem circle_radius_squared (r : ℝ) 
  (AB CD: ℝ) 
  (BP angleAPD : ℝ) 
  (P_outside_circle: True) 
  (AB_eq_12 : AB = 12) 
  (CD_eq_9 : CD = 9) 
  (AngleAPD_eq_45 : angleAPD = 45) 
  (BP_eq_10 : BP = 10) : r^2 = 73 :=
sorry

end circle_radius_squared_l176_176672


namespace arithmetic_sequence_problem_l176_176146

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_problem_l176_176146


namespace negation_proposition_equivalence_l176_176530

theorem negation_proposition_equivalence :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_proposition_equivalence_l176_176530


namespace stratified_sampling_females_l176_176604

theorem stratified_sampling_females :
  let males := 500
  let females := 400
  let total_students := 900
  let total_surveyed := 45
  let males_surveyed := 25
  ((males_surveyed : ℚ) / males) * females = 20 := by
  sorry

end stratified_sampling_females_l176_176604


namespace cyclist_wait_time_l176_176610

noncomputable def hiker_speed : ℝ := 5 / 60
noncomputable def cyclist_speed : ℝ := 25 / 60
noncomputable def wait_time : ℝ := 5
noncomputable def distance_ahead : ℝ := cyclist_speed * wait_time
noncomputable def catching_time : ℝ := distance_ahead / hiker_speed

theorem cyclist_wait_time : catching_time = 25 := by
  sorry

end cyclist_wait_time_l176_176610


namespace range_of_t_l176_176232

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  ∃ t : ℝ, (t = a^2 - a*b + b^2) ∧ (1/3 ≤ t ∧ t ≤ 3) :=
sorry

end range_of_t_l176_176232


namespace fern_pays_228_11_usd_l176_176809

open Real

noncomputable def high_heels_price : ℝ := 66
noncomputable def ballet_slippers_price : ℝ := (2 / 3) * high_heels_price
noncomputable def purse_price : ℝ := 49.5
noncomputable def scarf_price : ℝ := 27.5
noncomputable def high_heels_discount : ℝ := 0.10 * high_heels_price
noncomputable def discounted_high_heels_price : ℝ := high_heels_price - high_heels_discount
noncomputable def total_cost_before_tax : ℝ := discounted_high_heels_price + ballet_slippers_price + purse_price + scarf_price
noncomputable def sales_tax : ℝ := 0.075 * total_cost_before_tax
noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax
noncomputable def exchange_rate : ℝ := 1 / 0.85
noncomputable def total_cost_in_usd : ℝ := total_cost_after_tax * exchange_rate

theorem fern_pays_228_11_usd: total_cost_in_usd = 228.11 := by
  sorry

end fern_pays_228_11_usd_l176_176809


namespace solve_equation_l176_176577

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l176_176577


namespace ruby_height_l176_176682

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end ruby_height_l176_176682


namespace sum_of_numbers_l176_176059

theorem sum_of_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 54) (h_ratio : a / b = 2 / 3) : a + b = 45 :=
by
  sorry

end sum_of_numbers_l176_176059


namespace pool_width_40_l176_176797

theorem pool_width_40
  (hose_rate : ℕ)
  (pool_length : ℕ)
  (pool_depth : ℕ)
  (pool_capacity_percent : ℚ)
  (drain_time : ℕ)
  (water_drained : ℕ)
  (total_capacity : ℚ)
  (pool_width : ℚ) :
  hose_rate = 60 ∧
  pool_length = 150 ∧
  pool_depth = 10 ∧
  pool_capacity_percent = 0.8 ∧
  drain_time = 800 ∧
  water_drained = hose_rate * drain_time ∧
  total_capacity = water_drained / pool_capacity_percent ∧
  total_capacity = pool_length * pool_width * pool_depth →
  pool_width = 40 :=
by
  sorry

end pool_width_40_l176_176797


namespace arithmetic_sequence_common_difference_l176_176918

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 13) 
  (h2 : (5 * (a 1 + a 5)) / 2 = 35) 
  (h_arithmetic_sequence : ∀ n, a (n+1) = a n + d) : 
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l176_176918


namespace find_percentage_l176_176345

theorem find_percentage (P : ℝ) : 
  (P / 100) * 700 = 210 ↔ P = 30 := by
  sorry

end find_percentage_l176_176345


namespace price_of_horse_and_cow_l176_176281

theorem price_of_horse_and_cow (x y : ℝ) (h1 : 4 * x + 6 * y = 48) (h2 : 3 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) := 
by
  exact ⟨h1, h2⟩

end price_of_horse_and_cow_l176_176281


namespace sum_of_common_ratios_l176_176252

variable {k p r : ℝ}

theorem sum_of_common_ratios (h1 : k ≠ 0)
                             (h2 : p ≠ r)
                             (h3 : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
                             p + r = 5 := 
by
  sorry

end sum_of_common_ratios_l176_176252


namespace obtain_2020_from_20_and_21_l176_176647

theorem obtain_2020_from_20_and_21 :
  ∃ (a b : ℕ), 20 * a + 21 * b = 2020 :=
by
  -- We only need to construct the proof goal, leaving the proof itself out.
  sorry

end obtain_2020_from_20_and_21_l176_176647


namespace trajectory_of_P_l176_176308

-- Define points P, A, and B in a 2D plane
variable {P A B : EuclideanSpace ℝ (Fin 2)}

-- Define the condition that the sum of the distances from P to A and P to B equals the distance between A and B
def sum_of_distances_condition (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P A + dist P B = dist A B

-- Main theorem statement: If P satisfies the above condition, then P lies on the line segment AB
theorem trajectory_of_P (P A B : EuclideanSpace ℝ (Fin 2)) (h : sum_of_distances_condition P A B) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • A + (1 - t) • B :=
  sorry

end trajectory_of_P_l176_176308


namespace integer_solutions_to_abs_equation_l176_176958

theorem integer_solutions_to_abs_equation :
  {p : ℤ × ℤ | abs (p.1 - 2) + abs (p.2 - 1) = 1} =
  {(3, 1), (1, 1), (2, 2), (2, 0)} :=
by
  sorry

end integer_solutions_to_abs_equation_l176_176958


namespace exists_real_root_in_interval_l176_176311

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3

theorem exists_real_root_in_interval (f : ℝ → ℝ)
  (h_mono : ∀ x y, x < y → f x < f y)
  (h1 : f 1 < 0)
  (h2 : f 2 > 0) : 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 := 
sorry

end exists_real_root_in_interval_l176_176311


namespace total_oysters_eaten_l176_176799

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end total_oysters_eaten_l176_176799


namespace Tamara_height_l176_176796

-- Define the conditions and goal as a theorem
theorem Tamara_height (K T : ℕ) (h1 : T = 3 * K - 4) (h2 : K + T = 92) : T = 68 :=
by
  sorry

end Tamara_height_l176_176796


namespace proof_of_ratio_l176_176154

def f (x : ℤ) : ℤ := 3 * x + 4

def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_of_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 :=
by
  sorry

end proof_of_ratio_l176_176154


namespace solve_for_x_l176_176612

theorem solve_for_x (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end solve_for_x_l176_176612


namespace total_limes_l176_176554

-- Define the number of limes picked by Alyssa, Mike, and Tom's plums
def alyssa_limes : ℕ := 25
def mike_limes : ℕ := 32
def tom_plums : ℕ := 12

theorem total_limes : alyssa_limes + mike_limes = 57 := by
  -- The proof is omitted as per the instruction
  sorry

end total_limes_l176_176554


namespace faith_weekly_earnings_l176_176362

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end faith_weekly_earnings_l176_176362


namespace find_missing_number_l176_176213

theorem find_missing_number (n : ℝ) :
  (0.0088 * 4.5) / (0.05 * 0.1 * n) = 990 → n = 0.008 :=
by
  intro h
  sorry

end find_missing_number_l176_176213


namespace evaluate_expression_l176_176502

theorem evaluate_expression (a b c : ℤ) 
  (h1 : c = b - 12) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2) * (b + 1) / (b - 3) * (c + 10) / (c + 7) = 10 / 3) :=
by
  sorry

end evaluate_expression_l176_176502


namespace black_balls_probability_both_black_l176_176614

theorem black_balls_probability_both_black (balls_total balls_black balls_gold : ℕ) (prob : ℚ) 
  (h1 : balls_total = 11)
  (h2 : balls_black = 7)
  (h3 : balls_gold = 4)
  (h4 : balls_total = balls_black + balls_gold)
  (h5 : prob = (21 : ℚ) / 55) :
  balls_total.choose 2 * prob = balls_black.choose 2 :=
sorry

end black_balls_probability_both_black_l176_176614


namespace quadratic_roots_properties_l176_176950

-- Given the quadratic equation x^2 - 7x + 12 = 0
-- Prove that the absolute value of the difference of the roots is 1
-- Prove that the maximum value of the roots is 4

theorem quadratic_roots_properties :
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → abs (r1 - r2) = 1) ∧ 
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → max r1 r2 = 4) :=
by sorry

end quadratic_roots_properties_l176_176950


namespace derivative_at_2_l176_176702

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_at_2 : deriv f 2 = Real.sqrt 2 / 4 := by
  sorry

end derivative_at_2_l176_176702


namespace max_frac_a_c_squared_l176_176407

theorem max_frac_a_c_squared 
  (a b c : ℝ) (y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order: a ≥ b ∧ b ≥ c)
  (h_system: a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2)
  (h_bounds: 0 ≤ y ∧ y < a ∧ 0 ≤ z ∧ z < c) :
  (a/c)^2 ≤ 4/3 :=
sorry

end max_frac_a_c_squared_l176_176407


namespace at_least_one_gt_one_l176_176225

theorem at_least_one_gt_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l176_176225


namespace average_speed_of_bus_l176_176864

theorem average_speed_of_bus (speed_bicycle : ℝ)
  (start_distance : ℝ) (catch_up_time : ℝ)
  (h1 : speed_bicycle = 15)
  (h2 : start_distance = 195)
  (h3 : catch_up_time = 3) : 
  (start_distance + speed_bicycle * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end average_speed_of_bus_l176_176864


namespace suitable_for_comprehensive_survey_l176_176642

-- Define the conditions
def is_comprehensive_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  is_specific_group ∧ (group_size < 100)  -- assuming "small" means fewer than 100 individuals/items

def is_sampling_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  ¬is_comprehensive_survey group_size is_specific_group

-- Define the surveys
def option_A (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_comprehensive_survey group_size is_specific_group

def option_B (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_C (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_D (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

-- Question: Which of the following surveys is suitable for a comprehensive survey given conditions
theorem suitable_for_comprehensive_survey :
  ∀ (group_size_A group_size_B group_size_C group_size_D : ℕ) 
    (is_specific_group_A is_specific_group_B is_specific_group_C is_specific_group_D : Bool),
  option_A group_size_A is_specific_group_A ↔ 
  ((option_B group_size_B is_specific_group_B = false) ∧ 
   (option_C group_size_C is_specific_group_C = false) ∧ 
   (option_D group_size_D is_specific_group_D = false)) :=
by
  sorry

end suitable_for_comprehensive_survey_l176_176642


namespace yuri_total_puppies_l176_176882

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l176_176882


namespace sufficient_but_not_necessary_condition_l176_176501

noncomputable def f (x a : ℝ) : ℝ := (x + 1) / x + Real.sin x - a^2

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a = 1) : 
  (∀ x, f x a + f (-x) a = 0) ↔ (a = 1) ∨ (a = -1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l176_176501


namespace angle_C_max_sum_of_sides_l176_176736

theorem angle_C (a b c : ℝ) (S : ℝ) (h1 : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.pi / 3 :=
by
  sorry

theorem max_sum_of_sides (a b : ℝ) (c : ℝ) (hC : c = Real.sqrt 3) :
  (a + b) ≤ 2 * Real.sqrt 3 :=
by
  sorry

end angle_C_max_sum_of_sides_l176_176736


namespace intersection_complement_correct_l176_176708

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A based on the condition given
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 3}

-- Define set B based on the condition given
def B : Set ℝ := {x | x > 3}

-- Define the complement of set B in the universal set U
def compl_B : Set ℝ := {x | x ≤ 3}

-- Define the expected result of A ∩ compl_B
def expected_result : Set ℝ := {x | x ≤ -3} ∪ {3}

-- State the theorem to be proven
theorem intersection_complement_correct :
  (A ∩ compl_B) = expected_result :=
sorry

end intersection_complement_correct_l176_176708


namespace ducks_killed_is_20_l176_176837

variable (x : ℕ)

def killed_ducks_per_year (x : ℕ) : Prop :=
  let initial_flock := 100
  let annual_births := 30
  let years := 5
  let additional_flock := 150
  let final_flock := 300
  initial_flock + years * (annual_births - x) + additional_flock = final_flock

theorem ducks_killed_is_20 : killed_ducks_per_year 20 :=
by
  sorry

end ducks_killed_is_20_l176_176837


namespace sin_gt_cos_lt_nec_suff_l176_176825

-- Define the triangle and the angles
variables {A B C : ℝ}
variables (t : triangle A B C)

-- Define conditions in the triangle: sum of angles is 180 degrees
axiom angle_sum : A + B + C = 180

-- Define sin and cos using the sides of the triangle
noncomputable def sin_A (A : ℝ) : ℝ := sorry -- placeholder for actual definition
noncomputable def sin_B (B : ℝ) : ℝ := sorry
noncomputable def cos_A (A : ℝ) : ℝ := sorry
noncomputable def cos_B (B : ℝ) : ℝ := sorry

-- The proposition to prove
theorem sin_gt_cos_lt_nec_suff {A B : ℝ} (h1 : sin_A A > sin_B B) :
  cos_A A < cos_B B ↔ sin_A A > sin_B B := sorry

end sin_gt_cos_lt_nec_suff_l176_176825


namespace probability_of_2_gold_no_danger_l176_176998

variable (caves : Finset Nat) (n : Nat)

-- Probability definitions
def P_gold_no_danger : ℚ := 1 / 5
def P_danger_no_gold : ℚ := 1 / 10
def P_neither : ℚ := 4 / 5

-- Probability calculation
def P_exactly_2_gold_none_danger : ℚ :=
  10 * (P_gold_no_danger) ^ 2 * (P_neither) ^ 3

theorem probability_of_2_gold_no_danger :
  (P_exactly_2_gold_none_danger) = 128 / 625 :=
sorry

end probability_of_2_gold_no_danger_l176_176998


namespace Sandwiches_count_l176_176984

-- Define the number of toppings and the number of choices for the patty
def num_toppings : Nat := 10
def num_choices_per_topping : Nat := 2
def num_patties : Nat := 3

-- Define the theorem to prove the total number of sandwiches
theorem Sandwiches_count : (num_choices_per_topping ^ num_toppings) * num_patties = 3072 :=
by
  sorry

end Sandwiches_count_l176_176984


namespace tangent_line_eq_l176_176421

noncomputable def f (x : ℝ) : ℝ := x + Real.log x

theorem tangent_line_eq :
  ∃ (m b : ℝ), (m = (deriv f 1)) ∧ (b = (f 1 - m * 1)) ∧
   (∀ (x y : ℝ), y = m * (x - 1) + b ↔ y = 2 * x - 1) :=
by sorry

end tangent_line_eq_l176_176421


namespace xiaodong_election_l176_176435

theorem xiaodong_election (V : ℕ) (h1 : 0 < V) :
  let total_needed := (3 : ℚ) / 4 * V
  let votes_obtained := (5 : ℚ) / 6 * (2 : ℚ) / 3 * V
  let remaining_votes := V - (2 : ℚ) / 3 * V
  total_needed - votes_obtained = (7 : ℚ) / 12 * remaining_votes :=
by 
  sorry

end xiaodong_election_l176_176435


namespace negation_example_l176_176874

theorem negation_example : ¬ (∀ x : ℝ, x^2 ≥ Real.log 2) ↔ ∃ x : ℝ, x^2 < Real.log 2 :=
by
  sorry

end negation_example_l176_176874


namespace john_investment_in_bankA_l176_176347

-- Definitions to set up the conditions
def total_investment : ℝ := 1500
def bankA_rate : ℝ := 0.04
def bankB_rate : ℝ := 0.06
def final_amount : ℝ := 1575

-- Definition of the question to be proved
theorem john_investment_in_bankA (x : ℝ) (h : 0 ≤ x ∧ x ≤ total_investment) :
  (x * (1 + bankA_rate) + (total_investment - x) * (1 + bankB_rate) = final_amount) -> x = 750 := sorry


end john_investment_in_bankA_l176_176347


namespace find_n_from_binomial_expansion_l176_176883

theorem find_n_from_binomial_expansion (x a : ℝ) (n : ℕ)
  (h4 : (Nat.choose n 3) * x^(n - 3) * a^3 = 210)
  (h5 : (Nat.choose n 4) * x^(n - 4) * a^4 = 420)
  (h6 : (Nat.choose n 5) * x^(n - 5) * a^5 = 630) :
  n = 19 :=
sorry

end find_n_from_binomial_expansion_l176_176883


namespace solve_eq_roots_l176_176123

noncomputable def solve_equation (x : ℝ) : Prop :=
  (7 * x + 2) / (3 * x^2 + 7 * x - 6) = (3 * x) / (3 * x - 2)

theorem solve_eq_roots (x : ℝ) (h₁ : x ≠ 2 / 3) :
  solve_equation x ↔ (x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3) :=
by
  sorry

end solve_eq_roots_l176_176123


namespace original_price_before_discounts_l176_176922

theorem original_price_before_discounts (P : ℝ) 
  (h : 0.75 * (0.75 * P) = 18) : P = 32 :=
by
  sorry

end original_price_before_discounts_l176_176922


namespace complex_number_value_l176_176956

open Complex

theorem complex_number_value (a : ℝ) 
  (h1 : z = (2 + a * I) / (1 + I)) 
  (h2 : (z.re, z.im) ∈ { p : ℝ × ℝ | p.2 = -p.1 }) : 
  a = 0 :=
by
  sorry

end complex_number_value_l176_176956


namespace possible_values_count_l176_176280

theorem possible_values_count {x y z : ℤ} (h₁ : x = 5) (h₂ : y = -3) (h₃ : z = -1) :
  ∃ v, v = x - y - z ∧ (v = 7 ∨ v = 8 ∨ v = 9) :=
by
  sorry

end possible_values_count_l176_176280


namespace pool_capacity_is_80_percent_l176_176755

noncomputable def current_capacity_percentage (width length depth rate time : ℝ) : ℝ :=
  let total_volume := width * length * depth
  let water_removed := rate * time
  (water_removed / total_volume) * 100

theorem pool_capacity_is_80_percent :
  current_capacity_percentage 50 150 10 60 1000 = 80 :=
by
  sorry

end pool_capacity_is_80_percent_l176_176755


namespace cos_theta_when_f_maximizes_l176_176178

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem cos_theta_when_f_maximizes (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end cos_theta_when_f_maximizes_l176_176178


namespace condition_implies_at_least_one_gt_one_l176_176086

theorem condition_implies_at_least_one_gt_one (a b : ℝ) :
  (a + b > 2 → (a > 1 ∨ b > 1)) ∧ ¬(a^2 + b^2 > 2 → (a > 1 ∨ b > 1)) :=
by
  sorry

end condition_implies_at_least_one_gt_one_l176_176086


namespace pet_preferences_l176_176336

/-- A store has several types of pets: 20 puppies, 10 kittens, 8 hamsters, and 5 birds.
Alice, Bob, Charlie, and David each want a different kind of pet, with the following preferences:
- Alice does not want a bird.
- Bob does not want a hamster.
- Charlie does not want a kitten.
- David does not want a puppy.
Prove that the number of ways they can choose different types of pets satisfying
their preferences is 791440. -/
theorem pet_preferences :
  let P := 20    -- Number of puppies
  let K := 10    -- Number of kittens
  let H := 8     -- Number of hamsters
  let B := 5     -- Number of birds
  let Alice_options := P + K + H -- Alice does not want a bird
  let Bob_options := P + K + B   -- Bob does not want a hamster
  let Charlie_options := P + H + B -- Charlie does not want a kitten
  let David_options := K + H + B   -- David does not want a puppy
  let Alice_pick := Alice_options
  let Bob_pick := Bob_options - 1
  let Charlie_pick := Charlie_options - 2
  let David_pick := David_options - 3
  Alice_pick * Bob_pick * Charlie_pick * David_pick = 791440 :=
by
  sorry

end pet_preferences_l176_176336


namespace Kayla_points_on_first_level_l176_176198

theorem Kayla_points_on_first_level
(points_2 : ℕ) (points_3 : ℕ) (points_4 : ℕ) (points_5 : ℕ) (points_6 : ℕ)
(h2 : points_2 = 3) (h3 : points_3 = 5) (h4 : points_4 = 8) (h5 : points_5 = 12) (h6 : points_6 = 17) :
  ∃ (points_1 : ℕ), 
    (points_3 - points_2 = 2) ∧ 
    (points_4 - points_3 = 3) ∧ 
    (points_5 - points_4 = 4) ∧ 
    (points_6 - points_5 = 5) ∧ 
    (points_2 - points_1 = 1) ∧ 
    points_1 = 2 :=
by
  use 2
  repeat { split }
  sorry

end Kayla_points_on_first_level_l176_176198


namespace min_value_of_frac_l176_176757

open Real

theorem min_value_of_frac (x : ℝ) (hx : x > 0) : 
  ∃ (t : ℝ), t = 2 * sqrt 5 + 2 ∧ (∀ y, y > 0 → (x^2 + 2 * x + 5) / x ≥ t) :=
by
  sorry

end min_value_of_frac_l176_176757


namespace sum_of_integers_l176_176294

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 288) : x + y = 35 :=
sorry

end sum_of_integers_l176_176294


namespace number_of_true_propositions_l176_176125

noncomputable def f : ℝ → ℝ := sorry -- since it's not specified, we use sorry here

-- Definitions for the conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Original proposition
def original_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, is_odd f → f 0 = 0

-- Converse proposition
def converse_proposition (f : ℝ → ℝ) :=
  f 0 = 0 → ∀ x : ℝ, is_odd f

-- Inverse proposition (logically equivalent to the converse)
def inverse_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, ¬(is_odd f) → f 0 ≠ 0

-- Contrapositive proposition (logically equivalent to the original)
def contrapositive_proposition (f : ℝ → ℝ) :=
  f 0 ≠ 0 → ∀ x : ℝ, ¬(is_odd f)

-- Theorem statement
theorem number_of_true_propositions (f : ℝ → ℝ) :
  (original_proposition f → true) ∧
  (converse_proposition f → false) ∧
  (inverse_proposition f → false) ∧
  (contrapositive_proposition f → true) →
  2 = 2 := 
by 
  sorry -- proof to be inserted

end number_of_true_propositions_l176_176125


namespace largest_hexagon_angle_l176_176760

theorem largest_hexagon_angle (x : ℝ) : 
  (2 * x + 2 * x + 2 * x + 3 * x + 4 * x + 5 * x = 720) → (5 * x = 200) := by
  sorry

end largest_hexagon_angle_l176_176760


namespace diminished_value_160_l176_176334

theorem diminished_value_160 (x : ℕ) (n : ℕ) : 
  (∀ m, m > 200 ∧ (∀ k, m = k * 180) → n = m) →
  (200 + x = n) →
  x = 160 :=
by
  sorry

end diminished_value_160_l176_176334


namespace minimum_value_l176_176152

theorem minimum_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ x : ℝ, x = 4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ∧ x ≥ 4 * Real.sqrt 3 :=
by sorry

end minimum_value_l176_176152


namespace selling_price_of_car_l176_176563

theorem selling_price_of_car (purchase_price repair_cost : ℝ) (profit_percent : ℝ) 
    (h1 : purchase_price = 42000) (h2 : repair_cost = 8000) (h3 : profit_percent = 29.8) :
    (purchase_price + repair_cost) * (1 + profit_percent / 100) = 64900 := 
by 
  -- The proof will go here
  sorry

end selling_price_of_car_l176_176563


namespace number_of_jerseys_bought_l176_176994

-- Define the given constants
def initial_money : ℕ := 50
def cost_per_jersey : ℕ := 2
def cost_basketball : ℕ := 18
def cost_shorts : ℕ := 8
def money_left : ℕ := 14

-- Define the theorem to prove the number of jerseys Jeremy bought.
theorem number_of_jerseys_bought :
  (initial_money - money_left) = (cost_basketball + cost_shorts + 5 * cost_per_jersey) :=
by
  sorry

end number_of_jerseys_bought_l176_176994


namespace remainder_equivalence_l176_176483

theorem remainder_equivalence (x y q r : ℕ) (hxy : x = q * y + r) (hy_pos : 0 < y) (h_r : 0 ≤ r ∧ r < y) : 
  ((x - 3 * q * y) % y) = r := 
by 
  sorry

end remainder_equivalence_l176_176483


namespace mark_new_phone_plan_cost_l176_176800

noncomputable def total_new_plan_cost (old_plan_cost old_internet_cost old_intl_call_cost : ℝ) (percent_increase_plan percent_increase_internet percent_decrease_intl : ℝ) : ℝ :=
  let new_plan_cost := old_plan_cost * (1 + percent_increase_plan)
  let new_internet_cost := old_internet_cost * (1 + percent_increase_internet)
  let new_intl_call_cost := old_intl_call_cost * (1 - percent_decrease_intl)
  new_plan_cost + new_internet_cost + new_intl_call_cost

theorem mark_new_phone_plan_cost :
  let old_plan_cost := 150
  let old_internet_cost := 50
  let old_intl_call_cost := 30
  let percent_increase_plan := 0.30
  let percent_increase_internet := 0.20
  let percent_decrease_intl := 0.15
  total_new_plan_cost old_plan_cost old_internet_cost old_intl_call_cost percent_increase_plan percent_increase_internet percent_decrease_intl = 280.50 :=
by
  sorry

end mark_new_phone_plan_cost_l176_176800


namespace positive_difference_l176_176046

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l176_176046


namespace cat_and_mouse_positions_after_317_moves_l176_176153

-- Define the conditions of the problem
def cat_positions : List String := ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
def mouse_positions : List String := ["Top Left", "Top Middle", "Top Right", "Right Middle", "Bottom Right", "Bottom Middle", "Bottom Left", "Left Middle"]

-- Calculate the position of the cat after n moves
def cat_position_after_moves (n : Nat) : String :=
  cat_positions.get! (n % 4)

-- Calculate the position of the mouse after n moves
def mouse_position_after_moves (n : Nat) : String :=
  mouse_positions.get! (n % 8)

-- Prove the final positions of the cat and mouse after 317 moves
theorem cat_and_mouse_positions_after_317_moves :
  cat_position_after_moves 317 = "Top Left" ∧ mouse_position_after_moves 317 = "Bottom Middle" :=
by
  sorry

end cat_and_mouse_positions_after_317_moves_l176_176153


namespace index_card_area_reduction_index_card_area_when_other_side_shortened_l176_176363

-- Conditions
def original_length := 4
def original_width := 6
def shortened_length := 2
def target_area := 12
def shortened_other_width := 5

-- Theorems to prove
theorem index_card_area_reduction :
  (original_length - 2) * original_width = target_area := by
  sorry

theorem index_card_area_when_other_side_shortened :
  (original_length) * (original_width - 1) = 20 := by
  sorry

end index_card_area_reduction_index_card_area_when_other_side_shortened_l176_176363


namespace f_1992_eq_1992_l176_176978

def f (x : ℕ) : ℤ := sorry

theorem f_1992_eq_1992 (f : ℕ → ℤ) 
  (h1 : ∀ x : ℕ, 0 < x -> f x = f (x - 1) + f (x + 1))
  (h2 : f 0 = 1992) :
  f 1992 = 1992 := 
sorry

end f_1992_eq_1992_l176_176978


namespace antiderivative_correct_l176_176827

def f (x : ℝ) : ℝ := 2 * x
def F (x : ℝ) : ℝ := x^2 + 2

theorem antiderivative_correct :
  (∀ x, f x = deriv (F) x) ∧ (F 1 = 3) :=
by
  sorry

end antiderivative_correct_l176_176827


namespace find_number_l176_176034

theorem find_number (x : ℤ) (h : 7 * x + 37 = 100) : x = 9 :=
by
  sorry

end find_number_l176_176034


namespace ratio_of_a_over_b_l176_176532

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem ratio_of_a_over_b (a b : ℝ) (h_max : ∀ x : ℝ, f a b x ≤ 10)
  (h_cond1 : f a b 1 = 10) (h_cond2 : (deriv (f a b)) 1 = 0) :
  a / b = -2/3 :=
sorry

end ratio_of_a_over_b_l176_176532


namespace triangle_is_isosceles_if_median_bisects_perimeter_l176_176313

-- Defining the sides of the triangle
variables {a b c : ℝ}

-- Defining the median condition
def median_bisects_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 2 * (a/2 + b)

-- The main theorem stating that the triangle is isosceles if the median bisects the perimeter
theorem triangle_is_isosceles_if_median_bisects_perimeter (a b c : ℝ) 
  (h : median_bisects_perimeter a b c) : b = c :=
by
  sorry

end triangle_is_isosceles_if_median_bisects_perimeter_l176_176313


namespace minor_axis_of_ellipse_l176_176014

noncomputable def length_minor_axis 
    (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) (p4 : ℝ × ℝ) (p5 : ℝ × ℝ) : ℝ :=
if h : (p1, p2, p3, p4, p5) = ((1, 0), (1, 3), (4, 0), (4, 3), (6, 1.5)) then 3 else 0

theorem minor_axis_of_ellipse (p1 p2 p3 p4 p5 : ℝ × ℝ) :
  p1 = (1, 0) → p2 = (1, 3) → p3 = (4, 0) → p4 = (4, 3) → p5 = (6, 1.5) →
  length_minor_axis p1 p2 p3 p4 p5 = 3 :=
by sorry

end minor_axis_of_ellipse_l176_176014


namespace f_is_constant_l176_176959

noncomputable def f (x θ : ℝ) : ℝ :=
  (Real.cos (x - θ))^2 + (Real.cos x)^2 - 2 * Real.cos θ * Real.cos (x - θ) * Real.cos x

theorem f_is_constant (θ : ℝ) : ∀ x, f x θ = (Real.sin θ)^2 :=
by
  intro x
  sorry

end f_is_constant_l176_176959


namespace probability_sum_six_two_dice_l176_176110

theorem probability_sum_six_two_dice :
  let total_outcomes := 36
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes = 5 / 36 := by
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  sorry

end probability_sum_six_two_dice_l176_176110


namespace balance_difference_l176_176514

def compounded_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

def simple_interest_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

/-- Cedric deposits $15,000 into an account that pays 6% interest compounded annually,
    Daniel deposits $15,000 into an account that pays 8% simple annual interest.
    After 10 years, the positive difference between their balances is $137. -/
theorem balance_difference :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 10
  compounded_balance P r_cedric t - simple_interest_balance P r_daniel t = 137 := 
sorry

end balance_difference_l176_176514


namespace find_speed_second_part_l176_176670

noncomputable def speed_second_part (x : ℝ) (v : ℝ) : Prop :=
  let t1 := x / 65       -- Time to cover the first x km at 65 kmph
  let t2 := 2 * x / v    -- Time to cover the second 2x km at v kmph
  let avg_time := 3 * x / 26    -- Average speed of the entire journey
  t1 + t2 = avg_time

theorem find_speed_second_part (x : ℝ) (v : ℝ) (h : speed_second_part x v) : v = 86.67 :=
sorry -- Proof of the claim

end find_speed_second_part_l176_176670


namespace fewest_cookies_by_ben_l176_176930

noncomputable def cookie_problem : Prop :=
  let ana_area := 4 * Real.pi
  let ben_area := 9
  let carol_area := Real.sqrt (5 * (5 + 2 * Real.sqrt 5))
  let dave_area := 3.375 * Real.sqrt 3
  let dough := ana_area * 10
  let ana_cookies := dough / ana_area
  let ben_cookies := dough / ben_area
  let carol_cookies := dough / carol_area
  let dave_cookies := dough / dave_area
  ben_cookies < ana_cookies ∧ ben_cookies < carol_cookies ∧ ben_cookies < dave_cookies

theorem fewest_cookies_by_ben : cookie_problem := by
  sorry

end fewest_cookies_by_ben_l176_176930


namespace fraction_increase_by_two_l176_176084

theorem fraction_increase_by_two (x y : ℝ) : 
  (3 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * (3 * x * y) / (x + y) :=
by
  sorry

end fraction_increase_by_two_l176_176084


namespace sum_lent_out_l176_176508

theorem sum_lent_out (P R : ℝ) (h1 : 780 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 684 := 
  sorry

end sum_lent_out_l176_176508


namespace cost_per_book_l176_176557

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end cost_per_book_l176_176557


namespace determine_digit_X_l176_176915

theorem determine_digit_X (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (h : 510 / X = 10 * 4 + X + 2 * X) : X = 8 :=
sorry

end determine_digit_X_l176_176915


namespace range_of_a_l176_176909

theorem range_of_a {x y a : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + y + 6 = 4 * x * y) : a ≤ 10 / 3 :=
  sorry

end range_of_a_l176_176909


namespace find_f_2010_l176_176906

noncomputable def f : ℕ → ℤ := sorry

theorem find_f_2010 (f_prop : ∀ {a b n : ℕ}, a + b = 3 * 2^n → f a + f b = 2 * n^2) :
  f 2010 = 193 :=
sorry

end find_f_2010_l176_176906


namespace unique_prime_solution_l176_176562

theorem unique_prime_solution :
  ∃! (p : ℕ), Prime p ∧ (∃ (k : ℤ), 2 * (p ^ 4) - 7 * (p ^ 2) + 1 = k ^ 2) := 
sorry

end unique_prime_solution_l176_176562


namespace pencil_price_units_l176_176792

noncomputable def price_pencil (base_price: ℕ) (extra_cost: ℕ): ℝ :=
  (base_price + extra_cost) / 10000.0

theorem pencil_price_units (base_price: ℕ) (extra_cost: ℕ) (h_base: base_price = 5000) (h_extra: extra_cost = 20) : 
  price_pencil base_price extra_cost = 0.5 := by
  sorry

end pencil_price_units_l176_176792


namespace dislikes_TV_and_books_l176_176043

-- The problem conditions
def total_people : ℕ := 800
def percent_dislikes_TV : ℚ := 25 / 100
def percent_dislikes_both : ℚ := 15 / 100

-- The expected answer
def expected_dislikes_TV_and_books : ℕ := 30

-- The proof problem statement
theorem dislikes_TV_and_books : 
  (total_people * percent_dislikes_TV) * percent_dislikes_both = expected_dislikes_TV_and_books := by 
  sorry

end dislikes_TV_and_books_l176_176043


namespace seashells_remainder_l176_176277

theorem seashells_remainder :
  let derek := 58
  let emily := 73
  let fiona := 31 
  let total_seashells := derek + emily + fiona
  total_seashells % 10 = 2 :=
by
  sorry

end seashells_remainder_l176_176277


namespace parabola_focus_directrix_distance_l176_176271

theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (hp : 3 = p * (1:ℝ)^2) 
  (hparabola : ∀ x : ℝ, y = p * x^2 → x^2 = (1/3:ℝ) * y)
  : (distance_focus_directrix : ℝ) = (1 / 6:ℝ) :=
  sorry

end parabola_focus_directrix_distance_l176_176271


namespace problem_solution_l176_176275

theorem problem_solution (x : ℝ) (h1 : x = 12) (h2 : 5 + 7 / x = some_number - 5 / x) : some_number = 6 := 
by
  sorry

end problem_solution_l176_176275


namespace playgroup_count_l176_176332

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end playgroup_count_l176_176332


namespace sam_has_8_marbles_l176_176452

theorem sam_has_8_marbles :
  ∀ (steve sam sally : ℕ),
  sam = 2 * steve →
  sally = sam - 5 →
  steve + 3 = 10 →
  sam - 6 = 8 :=
by
  intros steve sam sally
  intros h1 h2 h3
  sorry

end sam_has_8_marbles_l176_176452


namespace eight_digit_product_1400_l176_176581

def eight_digit_numbers_count : Nat :=
  sorry

theorem eight_digit_product_1400 : eight_digit_numbers_count = 5880 :=
  sorry

end eight_digit_product_1400_l176_176581


namespace probability_of_same_color_correct_l176_176488

/-- Define events and their probabilities based on the given conditions --/
def probability_of_two_black_stones : ℚ := 1 / 7
def probability_of_two_white_stones : ℚ := 12 / 35

/-- Define the probability of drawing two stones of the same color --/
def probability_of_two_same_color_stones : ℚ :=
  probability_of_two_black_stones + probability_of_two_white_stones

theorem probability_of_same_color_correct :
  probability_of_two_same_color_stones = 17 / 35 :=
by
  -- We only set up the theorem, the proof is not considered here
  sorry

end probability_of_same_color_correct_l176_176488


namespace at_least_three_equal_l176_176655

theorem at_least_three_equal (a b c d : ℕ) (h1 : (a + b) ^ 2 ∣ c * d)
                                (h2 : (a + c) ^ 2 ∣ b * d)
                                (h3 : (a + d) ^ 2 ∣ b * c)
                                (h4 : (b + c) ^ 2 ∣ a * d)
                                (h5 : (b + d) ^ 2 ∣ a * c)
                                (h6 : (c + d) ^ 2 ∣ a * b) :
  ∃ x : ℕ, (x = a ∧ x = b ∧ x = c) ∨ (x = a ∧ x = b ∧ x = d) ∨ (x = a ∧ x = c ∧ x = d) ∨ (x = b ∧ x = c ∧ x = d) :=
sorry

end at_least_three_equal_l176_176655


namespace general_formula_sum_formula_l176_176289

-- Define the geometric sequence
def geoseq (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of the geometric sequence
def sum_first_n_terms (n : ℕ) : ℕ := 2^(n+1) - 2

-- Given conditions
def a1 : ℕ := 2
def a4 : ℕ := 16

-- Theorem statements
theorem general_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → geoseq n = 2^n := sorry

theorem sum_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → sum_first_n_terms n = 2^(n+1) - 2 := sorry

end general_formula_sum_formula_l176_176289


namespace total_local_percentage_approx_52_74_l176_176714

-- We provide the conditions as definitions
def total_arts_students : ℕ := 400
def local_arts_percentage : ℝ := 0.50
def total_science_students : ℕ := 100
def local_science_percentage : ℝ := 0.25
def total_commerce_students : ℕ := 120
def local_commerce_percentage : ℝ := 0.85

-- Calculate the expected total percentage of local students
noncomputable def calculated_total_local_percentage : ℝ :=
  let local_arts_students := local_arts_percentage * total_arts_students
  let local_science_students := local_science_percentage * total_science_students
  let local_commerce_students := local_commerce_percentage * total_commerce_students
  let total_local_students := local_arts_students + local_science_students + local_commerce_students
  let total_students := total_arts_students + total_science_students + total_commerce_students
  (total_local_students / total_students) * 100

-- State what we need to prove
theorem total_local_percentage_approx_52_74 :
  abs (calculated_total_local_percentage - 52.74) < 1 :=
sorry

end total_local_percentage_approx_52_74_l176_176714


namespace cylinder_surface_area_l176_176115

theorem cylinder_surface_area (h : ℝ) (c : ℝ) (r : ℝ) 
  (h_eq : h = 2) (c_eq : c = 2 * Real.pi) (circumference_formula : c = 2 * Real.pi * r) : 
  2 * (Real.pi * r^2) + (2 * Real.pi * r * h) = 6 * Real.pi := 
by
  sorry

end cylinder_surface_area_l176_176115


namespace sum_of_solutions_eq_l176_176093

theorem sum_of_solutions_eq :
  let A := 100
  let B := 3
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ = abs (B*x₁ - abs (A - B*x₁)) ∧ 
    x₂ = abs (B*x₂ - abs (A - B*x₂)) ∧ 
    x₃ = abs (B*x₃ - abs (A - B*x₃))) ∧ 
    (x₁ + x₂ + x₃ = (1900 : ℝ) / 7)) :=
by
  sorry

end sum_of_solutions_eq_l176_176093


namespace question1_question2_l176_176542

noncomputable def setA := {x : ℝ | -2 < x ∧ x < 4}
noncomputable def setB (m : ℝ) := {x : ℝ | x < -m}

-- (1) If A ∩ B = ∅, find the range of the real number m.
theorem question1 (m : ℝ) (h : setA ∩ setB m = ∅) : 2 ≤ m := by
  sorry

-- (2) If A ⊂ B, find the range of the real number m.
theorem question2 (m : ℝ) (h : setA ⊂ setB m) : m ≤ 4 := by
  sorry

end question1_question2_l176_176542


namespace min_mod_z_l176_176259

open Complex

theorem min_mod_z (z : ℂ) (hz : abs (z - 2 * I) + abs (z - 5) = 7) : abs z = 10 / 7 :=
sorry

end min_mod_z_l176_176259


namespace competition_problem_l176_176507

theorem competition_problem (n : ℕ) (s : ℕ) (correct_first_12 : s = (12 * 13) / 2)
    (gain_708_if_last_12_correct : s + 708 = (n - 11) * (n + 12) / 2):
    n = 71 :=
by
  sorry

end competition_problem_l176_176507


namespace length_of_FD_l176_176360

theorem length_of_FD (a b c d f e : ℝ) (x : ℝ) :
  a = 0 ∧ b = 8 ∧ c = 8 ∧ d = 0 ∧ 
  e = 8 * (2 / 3) ∧ 
  (8 - x)^2 = x^2 + (8 / 3)^2 ∧ 
  a = d → c = b → 
  d = 8 → 
  x = 32 / 9 :=
by
  sorry

end length_of_FD_l176_176360


namespace cats_kittentotal_l176_176955

def kittens_given_away : ℕ := 2
def kittens_now : ℕ := 6
def kittens_original : ℕ := 8

theorem cats_kittentotal : kittens_now + kittens_given_away = kittens_original := 
by 
  sorry

end cats_kittentotal_l176_176955


namespace cone_sphere_ratio_l176_176265

-- Defining the conditions and proof goals
theorem cone_sphere_ratio (r h : ℝ) (h_cone_sphere_radius : r ≠ 0) 
  (h_cone_volume : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  -- All the assumptions / conditions given in the problem
  sorry -- Proof omitted

end cone_sphere_ratio_l176_176265


namespace product_of_largest_two_and_four_digit_primes_l176_176257

theorem product_of_largest_two_and_four_digit_primes :
  let largest_two_digit_prime := 97
  let largest_four_digit_prime := 9973
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end product_of_largest_two_and_four_digit_primes_l176_176257


namespace ducklings_distance_l176_176512

noncomputable def ducklings_swim (r : ℝ) (n : ℕ) : Prop :=
  ∀ (ducklings : Fin n → ℝ × ℝ), (∀ i, (ducklings i).1 ^ 2 + (ducklings i).2 ^ 2 = r ^ 2) →
    ∃ (i j : Fin n), i ≠ j ∧ (ducklings i - ducklings j).1 ^ 2 + (ducklings i - ducklings j).2 ^ 2 ≤ r ^ 2

theorem ducklings_distance :
  ducklings_swim 5 6 :=
by sorry

end ducklings_distance_l176_176512


namespace integer_solutions_of_quadratic_eq_l176_176305

theorem integer_solutions_of_quadratic_eq (b : ℤ) :
  ∃ p q : ℤ, (p+9) * (q+9) = 81 ∧ p + q = -b ∧ p * q = 9*b :=
sorry

end integer_solutions_of_quadratic_eq_l176_176305


namespace translated_coordinates_of_B_l176_176875

-- Define the initial coordinates of points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the translated coordinates of point A
def A' : ℝ × ℝ := (4, 0)

-- Define the expected coordinates of point B' after the translation
def B' : ℝ × ℝ := (1, -1)

-- Proof statement
theorem translated_coordinates_of_B (A A' B : ℝ × ℝ) (B' : ℝ × ℝ) :
  A = (1, 1) ∧ A' = (4, 0) ∧ B = (-2, 0) → B' = (1, -1) :=
by
  intros h
  sorry

end translated_coordinates_of_B_l176_176875


namespace find_p_q_r_l176_176147

def is_rel_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem find_p_q_r (x : ℝ) (p q r : ℕ)
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r)
  (hpq_rel_prime : is_rel_prime p q)
  (hp : 0 < p)
  (hq : 0 < q)
  (hr : 0 < r) :
  p + q + r = 26 :=
sorry

end find_p_q_r_l176_176147


namespace find_breadth_of_rectangle_l176_176060

noncomputable def breadth_of_rectangle (s : ℝ) (π_approx : ℝ := 3.14) : ℝ :=
2 * s - 22

theorem find_breadth_of_rectangle (b s : ℝ) (π_approx : ℝ := 3.14) :
  4 * s = 2 * (22 + b) →
  π_approx * s / 2 + s = 29.85 →
  b = 1.22 :=
by
  intros h1 h2
  sorry

end find_breadth_of_rectangle_l176_176060


namespace multiples_count_l176_176750

theorem multiples_count (count_5 count_7 count_35 count_total : ℕ) :
  count_5 = 600 →
  count_7 = 428 →
  count_35 = 85 →
  count_total = count_5 + count_7 - count_35 →
  count_total = 943 :=
by
  sorry

end multiples_count_l176_176750


namespace finite_integer_solutions_l176_176835

theorem finite_integer_solutions (n : ℕ) : 
  ∃ (S : Finset (ℤ × ℤ)), ∀ (x y : ℤ), (x^3 + y^3 = n) → (x, y) ∈ S := 
sorry

end finite_integer_solutions_l176_176835


namespace total_hits_and_misses_l176_176711

theorem total_hits_and_misses (h : ℕ) (m : ℕ) (hc : m = 3 * h) (hm : m = 50) : h + m = 200 :=
by
  sorry

end total_hits_and_misses_l176_176711


namespace largest_4_digit_number_divisible_by_1615_l176_176383

theorem largest_4_digit_number_divisible_by_1615 (X : ℕ) (hX: 8640 = 1615 * X) (h1: 1000 ≤ 1615 * X ∧ 1615 * X ≤ 9999) : X = 5 :=
by
  sorry

end largest_4_digit_number_divisible_by_1615_l176_176383


namespace arithmetic_sequence_sum_l176_176069

theorem arithmetic_sequence_sum :
  let a := -3
  let d := 7
  let n := 10
  let s := n * (2 * a + (n - 1) * d) / 2
  s = 285 :=
by
  -- Details of the proof are omitted as per instructions
  sorry

end arithmetic_sequence_sum_l176_176069


namespace josh_paid_6_dollars_l176_176756

def packs : ℕ := 3
def cheesePerPack : ℕ := 20
def costPerCheese : ℕ := 10 -- cost in cents

theorem josh_paid_6_dollars :
  (packs * cheesePerPack * costPerCheese) / 100 = 6 :=
by
  sorry

end josh_paid_6_dollars_l176_176756


namespace ratio_of_b_to_c_l176_176567

theorem ratio_of_b_to_c (a b c : ℝ) 
  (h1 : a / b = 11 / 3) 
  (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := 
by
  sorry

end ratio_of_b_to_c_l176_176567


namespace square_area_EFGH_l176_176979

theorem square_area_EFGH (AB BP : ℝ) (h1 : AB = Real.sqrt 72) (h2 : BP = 2) (x : ℝ)
  (h3 : AB + BP = 2 * x + 2) : x^2 = 18 :=
by
  sorry

end square_area_EFGH_l176_176979


namespace sum_consecutive_even_integers_l176_176695

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end sum_consecutive_even_integers_l176_176695


namespace length_of_side_b_max_area_of_triangle_l176_176240

variable {A B C a b c : ℝ}
variable {triangle_ABC : a + c = 6}
variable {eq1 : (3 - Real.cos A) * Real.sin B = Real.sin A * (1 + Real.cos B)}

-- Theorem for part (1) length of side b
theorem length_of_side_b :
  b = 2 :=
sorry

-- Theorem for part (2) maximum area of the triangle
theorem max_area_of_triangle :
  ∃ (S : ℝ), S = 2 * Real.sqrt 2 :=
sorry

end length_of_side_b_max_area_of_triangle_l176_176240


namespace train_speed_ratio_l176_176430

theorem train_speed_ratio 
  (v_A v_B : ℝ)
  (h1 : v_A = 2 * v_B)
  (h2 : 27 = L_A / v_A)
  (h3 : 17 = L_B / v_B)
  (h4 : 22 = (L_A + L_B) / (v_A + v_B))
  (h5 : v_A + v_B ≤ 60) :
  v_A / v_B = 2 := by
  sorry

-- Conditions given must be defined properly
variables (L_A L_B : ℝ)

end train_speed_ratio_l176_176430


namespace tan_alpha_eq_one_third_cos2alpha_over_expr_l176_176026

theorem tan_alpha_eq_one_third_cos2alpha_over_expr (α : ℝ) (h : Real.tan α = 1/3) :
  (Real.cos (2 * α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8 / 15 :=
by
  -- This is the point where the proof steps will go, but we leave it as a placeholder.
  sorry

end tan_alpha_eq_one_third_cos2alpha_over_expr_l176_176026


namespace pow_add_div_eq_l176_176923

   theorem pow_add_div_eq (a b c d e : ℕ) (h1 : b = 2) (h2 : c = 345) (h3 : d = 9) (h4 : e = 8 - 5) : 
     a = b^c + d^e -> a = 2^345 + 729 := 
   by 
     intros 
     sorry
   
end pow_add_div_eq_l176_176923


namespace units_digit_of_m_squared_plus_two_to_m_is_3_l176_176704

def m := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m_is_3 : (m^2 + 2^m) % 10 = 3 := 
by 
  sorry

end units_digit_of_m_squared_plus_two_to_m_is_3_l176_176704


namespace conic_sections_hyperbola_and_ellipse_l176_176341

theorem conic_sections_hyperbola_and_ellipse
  (x y : ℝ) (h : y^4 - 9 * x^4 = 3 * y^2 - 3) :
  (∃ a b c : ℝ, a * y^2 - b * x^2 = c ∧ a = b ∧ c ≠ 0) ∨ (∃ a b c : ℝ, a * y^2 + b * x^2 = c ∧ a ≠ b ∧ c ≠ 0) :=
by
  sorry

end conic_sections_hyperbola_and_ellipse_l176_176341


namespace cos_of_angle_in_third_quadrant_l176_176969

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = -5 / 13) : Real.cos B = -12 / 13 := 
by 
  sorry

end cos_of_angle_in_third_quadrant_l176_176969


namespace initial_geese_count_l176_176559

theorem initial_geese_count (G : ℕ) (h1 : G / 2 + 4 = 12) : G = 16 := by
  sorry

end initial_geese_count_l176_176559


namespace a_2017_value_l176_176372

theorem a_2017_value (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = 2 * (n + 1) - 1) :
  a 2017 = 2 :=
by
  sorry

end a_2017_value_l176_176372


namespace digit_difference_one_l176_176295

theorem digit_difference_one (p q : ℕ) (h_pq : p < 10 ∧ q < 10) (h_diff : (10 * p + q) - (10 * q + p) = 9) :
  p - q = 1 :=
by
  sorry

end digit_difference_one_l176_176295


namespace total_bathing_suits_l176_176900

theorem total_bathing_suits (men_women_bathing_suits : Nat)
                            (men_bathing_suits : Nat := 14797)
                            (women_bathing_suits : Nat := 4969) :
    men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l176_176900


namespace determine_all_functions_l176_176544

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

theorem determine_all_functions (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end determine_all_functions_l176_176544


namespace factorials_sum_of_two_squares_l176_176278

-- Define what it means for a number to be a sum of two squares.
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem factorials_sum_of_two_squares :
  {n : ℕ | n < 14 ∧ is_sum_of_two_squares (n!)} = {2, 6} :=
by
  sorry

end factorials_sum_of_two_squares_l176_176278


namespace day50_yearM_minus1_is_Friday_l176_176008

-- Define weekdays
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Weekday

-- Define days of the week for specific days in given years
def day_of (d : Nat) (reference_day : Weekday) (reference_day_mod : Nat) : Weekday :=
  match (reference_day_mod + d - 1) % 7 with
  | 0 => Sunday
  | 1 => Monday
  | 2 => Tuesday
  | 3 => Wednesday
  | 4 => Thursday
  | 5 => Friday
  | 6 => Saturday
  | _ => Thursday -- This case should never occur due to mod 7

def day250_yearM : Weekday := Thursday
def day150_yearM1 : Weekday := Thursday

-- Theorem to prove
theorem day50_yearM_minus1_is_Friday :
    day_of 50 day250_yearM 6 = Friday :=
sorry

end day50_yearM_minus1_is_Friday_l176_176008


namespace fraction_subtraction_l176_176013

theorem fraction_subtraction :
  (12 / 30) - (1 / 7) = 9 / 35 :=
by sorry

end fraction_subtraction_l176_176013


namespace intersection_of_A_and_B_l176_176967

-- Define sets A and B
def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The proof statement
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l176_176967


namespace total_toys_is_60_l176_176781

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end total_toys_is_60_l176_176781


namespace ferry_P_travel_time_l176_176087

-- Definitions of conditions
def speed_P : ℝ := 6 -- speed of ferry P in km/h
def speed_diff_PQ : ℝ := 3 -- speed difference between ferry Q and ferry P in km/h
def travel_longer_Q : ℝ := 2 -- ferry Q travels a route twice as long as ferry P
def time_diff_PQ : ℝ := 1 -- time difference between ferry Q and ferry P in hours

-- Distance traveled by ferry P
def distance_P (t_P : ℝ) : ℝ := speed_P * t_P

-- Distance traveled by ferry Q
def distance_Q (t_P : ℝ) : ℝ := travel_longer_Q * (speed_P * t_P)

-- Speed of ferry Q
def speed_Q : ℝ := speed_P + speed_diff_PQ

-- Time taken by ferry Q
def time_Q (t_P : ℝ) : ℝ := t_P + time_diff_PQ

-- Main theorem statement
theorem ferry_P_travel_time (t_P : ℝ) : t_P = 3 :=
by
  have eq_Q : speed_Q * (time_Q t_P) = distance_Q t_P := sorry
  have eq_P : speed_P * t_P = distance_P t_P := sorry
  sorry

end ferry_P_travel_time_l176_176087


namespace pyramid_base_edge_length_l176_176779

-- Prove that the edge-length of the base of the pyramid is as specified
theorem pyramid_base_edge_length
  (r h : ℝ)
  (hemisphere_radius : r = 3)
  (pyramid_height : h = 8)
  (tangency_condition : true) : true :=
by
  sorry

end pyramid_base_edge_length_l176_176779


namespace domino_perfect_play_winner_l176_176780

theorem domino_perfect_play_winner :
  ∀ {PlayerI PlayerII : Type} 
    (legal_move : PlayerI → PlayerII → Prop)
    (initial_move : PlayerI → Prop)
    (next_moves : PlayerII → PlayerI → PlayerII → Prop),
    (∀ pI pII, legal_move pI pII) → 
    (∃ m, initial_move m) → 
    (∀ mI mII, next_moves mII mI mII) → 
    ∃ winner, winner = PlayerI :=
by
  sorry

end domino_perfect_play_winner_l176_176780


namespace packaging_combinations_l176_176392

-- Conditions
def wrapping_paper_choices : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

-- Question and proof
theorem packaging_combinations : wrapping_paper_choices * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end packaging_combinations_l176_176392


namespace total_donuts_three_days_l176_176092

def donuts_on_Monday := 14

def donuts_on_Tuesday := donuts_on_Monday / 2

def donuts_on_Wednesday := 4 * donuts_on_Monday

def total_donuts := donuts_on_Monday + donuts_on_Tuesday + donuts_on_Wednesday

theorem total_donuts_three_days : total_donuts = 77 :=
  by
    sorry

end total_donuts_three_days_l176_176092


namespace graduation_problem_l176_176681

def valid_xs : List ℕ :=
  [10, 12, 15, 18, 20, 24, 30]

noncomputable def sum_valid_xs (l : List ℕ) : ℕ :=
  l.foldr (λ x sum => x + sum) 0

theorem graduation_problem :
  sum_valid_xs valid_xs = 129 :=
by
  sorry

end graduation_problem_l176_176681


namespace adam_earnings_after_taxes_l176_176522

theorem adam_earnings_after_taxes
  (daily_earnings : ℕ) 
  (tax_pct : ℕ)
  (workdays : ℕ)
  (H1 : daily_earnings = 40) 
  (H2 : tax_pct = 10) 
  (H3 : workdays = 30) : 
  (daily_earnings - daily_earnings * tax_pct / 100) * workdays = 1080 := 
by
  -- Proof to be filled in
  sorry

end adam_earnings_after_taxes_l176_176522


namespace max_value_sum_l176_176175

variable (n : ℕ) (x : Fin n → ℝ)

theorem max_value_sum 
  (h1 : ∀ i, 0 ≤ x i)
  (h2 : 2 ≤ n)
  (h3 : (Finset.univ : Finset (Fin n)).sum x = 1) :
  ∃ max_val, max_val = (1 / 4) :=
sorry

end max_value_sum_l176_176175


namespace POTOP_correct_l176_176942

def POTOP : Nat := 51715

theorem POTOP_correct :
  (99999 * POTOP) % 1000 = 285 := by
  sorry

end POTOP_correct_l176_176942


namespace additional_people_needed_l176_176899

theorem additional_people_needed (k m : ℕ) (h1 : 8 * 3 = k) (h2 : m * 2 = k) : (m - 8) = 4 :=
by
  sorry

end additional_people_needed_l176_176899


namespace marbles_difference_l176_176831

def lostMarbles : ℕ := 8
def foundMarbles : ℕ := 10

theorem marbles_difference (lostMarbles foundMarbles : ℕ) : foundMarbles - lostMarbles = 2 := 
by
  sorry

end marbles_difference_l176_176831


namespace head_start_distance_l176_176631

theorem head_start_distance (v_A v_B L H : ℝ) (h1 : v_A = 15 / 13 * v_B)
    (h2 : t_A = L / v_A) (h3 : t_B = (L - H) / v_B) (h4 : t_B = t_A - 0.25 * L / v_B) :
    H = 23 / 60 * L :=
sorry

end head_start_distance_l176_176631


namespace silver_tokens_at_end_l176_176377

theorem silver_tokens_at_end {R B S : ℕ} (x y : ℕ) 
  (hR_init : R = 60) (hB_init : B = 90) 
  (hR_final : R = 60 - 3 * x + y) 
  (hB_final : B = 90 + 2 * x - 4 * y) 
  (h_end_conditions : 0 ≤ R ∧ R < 3 ∧ 0 ≤ B ∧ B < 4) : 
  S = x + y → 
  S = 23 :=
sorry

end silver_tokens_at_end_l176_176377


namespace square_side_length_l176_176090

theorem square_side_length 
  (AF DH BG AE : ℝ) 
  (AF_eq : AF = 7) 
  (DH_eq : DH = 4) 
  (BG_eq : BG = 5) 
  (AE_eq : AE = 1) 
  (area_EFGH : ℝ) 
  (area_EFGH_eq : area_EFGH = 78) : 
  (∃ s : ℝ, s^2 = 144) :=
by
  use 12
  sorry

end square_side_length_l176_176090


namespace min_degree_g_l176_176938

open Polynomial

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

-- Conditions
axiom cond1 : 5 • f + 7 • g = h
axiom cond2 : natDegree f = 10
axiom cond3 : natDegree h = 12

-- Question: Minimum degree of g
theorem min_degree_g : natDegree g = 12 :=
sorry

end min_degree_g_l176_176938


namespace sea_star_collection_l176_176132

theorem sea_star_collection (S : ℕ) (initial_seashells : ℕ) (initial_snails : ℕ) (lost_sea_creatures : ℕ) (remaining_items : ℕ) :
  initial_seashells = 21 →
  initial_snails = 29 →
  lost_sea_creatures = 25 →
  remaining_items = 59 →
  S + initial_seashells + initial_snails = remaining_items + lost_sea_creatures →
  S = 34 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end sea_star_collection_l176_176132


namespace highest_sum_vertex_l176_176820

theorem highest_sum_vertex (a b c d e f : ℕ) (h₀ : a + d = 8) (h₁ : b + e = 8) (h₂ : c + f = 8) : 
  a + b + c ≤ 11 ∧ b + c + d ≤ 11 ∧ c + d + e ≤ 11 ∧ d + e + f ≤ 11 ∧ e + f + a ≤ 11 ∧ f + a + b ≤ 11 :=
sorry

end highest_sum_vertex_l176_176820


namespace find_g_three_l176_176224

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_three (h : ∀ x : ℝ, g (3^x) + (x + 1) * g (3^(-x)) = 3) : g 3 = -3 :=
sorry

end find_g_three_l176_176224


namespace sufficient_but_not_necessary_l176_176235

theorem sufficient_but_not_necessary (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : 
  (a > 1 ∧ b > 1 → a * b > 1) ∧ ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l176_176235


namespace binomial_problem_l176_176861

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The problem statement: prove that binomial(13, 11) * 2 = 156
theorem binomial_problem : binomial 13 11 * 2 = 156 := by
  sorry

end binomial_problem_l176_176861


namespace option_B_option_D_l176_176947

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l176_176947


namespace problem_statement_negation_statement_l176_176735

variable {a b : ℝ}

theorem problem_statement (h : a * b ≤ 0) : a ≤ 0 ∨ b ≤ 0 :=
sorry

theorem negation_statement (h : a * b > 0) : a > 0 ∧ b > 0 :=
sorry

end problem_statement_negation_statement_l176_176735


namespace least_common_denominator_l176_176439

-- We first need to define the function to compute the LCM of a list of natural numbers.
def lcm_list (ns : List ℕ) : ℕ :=
ns.foldr Nat.lcm 1

theorem least_common_denominator : 
  lcm_list [3, 4, 5, 8, 9, 11] = 3960 := 
by
  -- Here's where the proof would go
  sorry

end least_common_denominator_l176_176439


namespace minimal_S_n_l176_176724

theorem minimal_S_n (a_n : ℕ → ℤ) 
  (h : ∀ n, a_n n = 3 * (n : ℤ) - 23) :
  ∃ n, (∀ m < n, (∀ k ≥ n, a_n k ≤ 0)) → n = 7 :=
by
  sorry

end minimal_S_n_l176_176724


namespace train_length_l176_176805

noncomputable def length_of_train (time_in_seconds : ℝ) (speed_in_kmh : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmh * (5 / 18)
  speed_in_mps * time_in_seconds

theorem train_length :
  length_of_train 2.3998080153587713 210 = 140 :=
by
  sorry

end train_length_l176_176805


namespace total_ingredients_used_l176_176217

theorem total_ingredients_used (water oliveOil salt : ℕ) 
  (h_ratio : water / oliveOil = 3 / 2) 
  (h_salt : water / salt = 3 / 1)
  (h_water_cups : water = 15) : 
  water + oliveOil + salt = 30 :=
sorry

end total_ingredients_used_l176_176217


namespace right_handed_players_total_l176_176766

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

end right_handed_players_total_l176_176766


namespace students_left_l176_176299

theorem students_left (initial_students new_students final_students students_left : ℕ)
  (h1 : initial_students = 10)
  (h2 : new_students = 42)
  (h3 : final_students = 48)
  : initial_students + new_students - students_left = final_students → students_left = 4 :=
by
  intros
  sorry

end students_left_l176_176299


namespace sqrt_of_1024_is_32_l176_176518

theorem sqrt_of_1024_is_32 (y : ℕ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 :=
sorry

end sqrt_of_1024_is_32_l176_176518


namespace Clever_not_Green_l176_176264

variables {Lizard : Type}
variables [DecidableEq Lizard] (Clever Green CanJump CanSwim : Lizard → Prop)

theorem Clever_not_Green (h1 : ∀ x, Clever x → CanJump x)
                        (h2 : ∀ x, Green x → ¬ CanSwim x)
                        (h3 : ∀ x, ¬ CanSwim x → ¬ CanJump x) :
  ∀ x, Clever x → ¬ Green x :=
by
  intro x hClever hGreen
  apply h3 x
  apply h2 x hGreen
  exact h1 x hClever

end Clever_not_Green_l176_176264


namespace eleven_million_scientific_notation_l176_176238

-- Definition of the scientific notation condition and question
def scientific_notation (a n : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ k : ℤ, n = 10 ^ k

-- The main theorem stating that 11 million can be expressed as 1.1 * 10^7
theorem eleven_million_scientific_notation : scientific_notation 1.1 (10 ^ 7) :=
by 
  -- Adding sorry to skip the proof
  sorry

end eleven_million_scientific_notation_l176_176238


namespace correct_assignment_statement_l176_176121

-- Definitions according to the problem conditions
def input_statement (x : Nat) : Prop := x = 3
def assignment_statement1 (A B : Nat) : Prop := A = B ∧ B = 2
def assignment_statement2 (T : Nat) : Prop := T = T * T
def output_statement (A : Nat) : Prop := A = 4

-- Lean statement for the problem. We need to prove that the assignment_statement2 is correct.
theorem correct_assignment_statement (T : Nat) : assignment_statement2 T :=
by sorry

end correct_assignment_statement_l176_176121


namespace second_part_of_ratio_l176_176945

theorem second_part_of_ratio (first_part : ℝ) (whole second_part : ℝ) (h1 : first_part = 5) (h2 : first_part / whole = 25 / 100) : second_part = 15 :=
by
  sorry

end second_part_of_ratio_l176_176945


namespace ambulance_ride_cost_correct_l176_176691

noncomputable def total_bill : ℝ := 12000
noncomputable def medication_percentage : ℝ := 0.40
noncomputable def imaging_tests_percentage : ℝ := 0.15
noncomputable def surgical_procedure_percentage : ℝ := 0.20
noncomputable def overnight_stays_percentage : ℝ := 0.25
noncomputable def food_cost : ℝ := 300
noncomputable def consultation_fee : ℝ := 80

noncomputable def ambulance_ride_cost := total_bill - (food_cost + consultation_fee)

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 11620 :=
by
  sorry

end ambulance_ride_cost_correct_l176_176691


namespace set_diff_example_l176_176388

-- Definitions of sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 3, 4}

-- Definition of set difference
def set_diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The mathematically equivalent proof problem statement
theorem set_diff_example :
  set_diff A B = {2} :=
sorry

end set_diff_example_l176_176388


namespace greatest_value_of_sum_l176_176248

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + (1/x)^2) : x + 1/x ≤ Real.sqrt 15 :=
sorry

end greatest_value_of_sum_l176_176248


namespace binomial_coefficient_10_3_l176_176065

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end binomial_coefficient_10_3_l176_176065


namespace find_x_y_l176_176715

theorem find_x_y (x y : ℤ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 :=
by
  sorry

end find_x_y_l176_176715


namespace profit_percentage_l176_176003

variable {C S : ℝ}

theorem profit_percentage (h : 19 * C = 16 * S) :
  ((S - C) / C) * 100 = 18.75 := by
  sorry

end profit_percentage_l176_176003


namespace function_even_periodic_l176_176798

theorem function_even_periodic (f : ℝ → ℝ) :
  (∀ x : ℝ, f (10 + x) = f (10 - x)) ∧ (∀ x : ℝ, f (5 - x) = f (5 + x)) →
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + 10) = f x) :=
by
  sorry

end function_even_periodic_l176_176798


namespace number_of_partners_equation_l176_176592

variable (x : ℕ)

theorem number_of_partners_equation :
  5 * x + 45 = 7 * x - 3 :=
sorry

end number_of_partners_equation_l176_176592


namespace line_equation_M_l176_176584

theorem line_equation_M (x y : ℝ) :
  (∃ (m c : ℝ), y = m * x + c ∧ m = -5/4 ∧ c = -3)
  ∧ (∃ (slope intercept : ℝ), slope = 2 * (-5/4) ∧ intercept = (1/2) * -3 ∧ (y - 2 = slope * (x + 4)))
  → ∃ (a b : ℝ), y = a * x + b ∧ a = -5/2 ∧ b = -8 :=
by
  sorry

end line_equation_M_l176_176584


namespace john_money_left_l176_176384

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end john_money_left_l176_176384


namespace smallest_gcd_12a_20b_l176_176765

theorem smallest_gcd_12a_20b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 := sorry

end smallest_gcd_12a_20b_l176_176765


namespace oxen_b_is_12_l176_176292

variable (oxen_b : ℕ)

def share (oxen months : ℕ) : ℕ := oxen * months

def total_share (oxen_a oxen_b oxen_c months_a months_b months_c : ℕ) : ℕ :=
  share oxen_a months_a + share oxen_b months_b + share oxen_c months_c

def proportion (rent_c rent total_share_c total_share : ℕ) : Prop :=
  rent_c * total_share = rent * total_share_c

theorem oxen_b_is_12 : oxen_b = 12 := by
  let oxen_a := 10
  let oxen_c := 15
  let months_a := 7
  let months_b := 5
  let months_c := 3
  let rent := 210
  let rent_c := 54
  let share_a := share oxen_a months_a
  let share_c := share oxen_c months_c
  let total_share_val := total_share oxen_a oxen_b oxen_c months_a months_b months_c
  let total_rent := share_a + 5 * oxen_b + share_c
  have h1 : proportion rent_c rent share_c total_rent := by sorry
  rw [proportion] at h1
  sorry

end oxen_b_is_12_l176_176292


namespace Alex_dimes_l176_176007

theorem Alex_dimes : 
    ∃ (d q : ℕ), 10 * d + 25 * q = 635 ∧ d = q + 5 ∧ d = 22 :=
by sorry

end Alex_dimes_l176_176007


namespace geom_seq_sum_l176_176608

theorem geom_seq_sum {a : ℕ → ℝ} (q : ℝ) (h1 : a 0 + a 1 + a 2 = 2)
    (h2 : a 3 + a 4 + a 5 = 16)
    (h_geom : ∀ n, a (n + 1) = q * a n) :
  a 6 + a 7 + a 8 = 128 :=
sorry

end geom_seq_sum_l176_176608


namespace calculate_truck_loads_of_dirt_l176_176245

noncomputable def truck_loads_sand: ℚ := 0.16666666666666666
noncomputable def truck_loads_cement: ℚ := 0.16666666666666666
noncomputable def total_truck_loads_material: ℚ := 0.6666666666666666
noncomputable def truck_loads_dirt: ℚ := total_truck_loads_material - (truck_loads_sand + truck_loads_cement)

theorem calculate_truck_loads_of_dirt :
  truck_loads_dirt = 0.3333333333333333 := 
by
  sorry

end calculate_truck_loads_of_dirt_l176_176245


namespace probability_of_forming_CHORAL_is_correct_l176_176648

-- Definitions for selecting letters with given probabilities
def probability_select_C_A_L_from_CAMEL : ℚ :=
  1 / 10

def probability_select_H_O_R_from_SHRUB : ℚ :=
  1 / 10

def probability_select_G_from_GLOW : ℚ :=
  1 / 2

-- Calculating the total probability of selecting letters to form "CHORAL"
def probability_form_CHORAL : ℚ :=
  probability_select_C_A_L_from_CAMEL * 
  probability_select_H_O_R_from_SHRUB * 
  probability_select_G_from_GLOW

theorem probability_of_forming_CHORAL_is_correct :
  probability_form_CHORAL = 1 / 200 :=
by
  -- Statement to be proven here
  sorry

end probability_of_forming_CHORAL_is_correct_l176_176648


namespace ezekiel_third_day_hike_l176_176450

-- Ezekiel's total hike distance
def total_distance : ℕ := 50

-- Distance covered on the first day
def first_day_distance : ℕ := 10

-- Distance covered on the second day
def second_day_distance : ℕ := total_distance / 2

-- Distance remaining for the third day
def third_day_distance : ℕ := total_distance - first_day_distance - second_day_distance

-- The distance Ezekiel had to hike on the third day
theorem ezekiel_third_day_hike : third_day_distance = 15 := by
  sorry

end ezekiel_third_day_hike_l176_176450


namespace sequence_length_l176_176526

theorem sequence_length 
  (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) (n : ℕ) 
  (h₁ : a₁ = -4) 
  (h₂ : d = 3) 
  (h₃ : aₙ = 32) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  n = 13 := 
by 
  sorry

end sequence_length_l176_176526


namespace smallest_d_factors_l176_176962

theorem smallest_d_factors (d : ℕ) (h₁ : ∃ p q : ℤ, p * q = 2050 ∧ p + q = d ∧ p > 0 ∧ q > 0) :
    d = 107 :=
by
  sorry

end smallest_d_factors_l176_176962


namespace value_of_x_l176_176005

theorem value_of_x 
    (r : ℝ) (a : ℝ) (x : ℝ) (shaded_area : ℝ)
    (h1 : r = 2)
    (h2 : a = 2)
    (h3 : shaded_area = 2) :
  x = (Real.pi / 3) + (Real.sqrt 3 / 2) - 1 :=
sorry

end value_of_x_l176_176005


namespace constant_term_binomial_expansion_l176_176863

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (8 - 2 * r = 0) ∧ Nat.choose 8 r = 70 := by
  sorry

end constant_term_binomial_expansion_l176_176863


namespace remaining_surface_area_unchanged_l176_176808

noncomputable def original_cube_surface_area : Nat := 6 * 4 * 4

def corner_cube_surface_area : Nat := 3 * 2 * 2

def remaining_surface_area (original_cube_surface_area : Nat) (corner_cube_surface_area : Nat) : Nat :=
  original_cube_surface_area

theorem remaining_surface_area_unchanged :
  remaining_surface_area original_cube_surface_area corner_cube_surface_area = 96 := 
by
  sorry

end remaining_surface_area_unchanged_l176_176808


namespace largest_expression_is_d_l176_176525

def expr_a := 3 + 0 + 4 + 8
def expr_b := 3 * 0 + 4 + 8
def expr_c := 3 + 0 * 4 + 8
def expr_d := 3 + 0 + 4 * 8
def expr_e := 3 * 0 * 4 * 8
def expr_f := (3 + 0 + 4) / 8

theorem largest_expression_is_d : 
  expr_d = 35 ∧ 
  expr_a = 15 ∧ 
  expr_b = 12 ∧ 
  expr_c = 11 ∧ 
  expr_e = 0 ∧ 
  expr_f = 7 / 8 ∧
  35 > 15 ∧ 
  35 > 12 ∧ 
  35 > 11 ∧ 
  35 > 0 ∧ 
  35 > 7 / 8 := 
by
  sorry

end largest_expression_is_d_l176_176525


namespace simplify_expression_l176_176889

theorem simplify_expression :
  (Real.sqrt (8^(1/3)) + Real.sqrt (17/4))^2 = (33 + 8 * Real.sqrt 17) / 4 :=
by
  sorry

end simplify_expression_l176_176889


namespace ratio_of_areas_l176_176547

theorem ratio_of_areas (Q : Point) (r1 r2 : ℝ) (h : r1 < r2)
  (arc_length_smaller : ℝ) (arc_length_larger : ℝ)
  (h_arc_smaller : arc_length_smaller = (60 / 360) * (2 * r1 * π))
  (h_arc_larger : arc_length_larger = (30 / 360) * (2 * r2 * π))
  (h_equal_arcs : arc_length_smaller = arc_length_larger) :
  (π * r1^2) / (π * r2^2) = 1/4 :=
by
  sorry

end ratio_of_areas_l176_176547


namespace probability_of_drawing_red_ball_l176_176783

def totalBalls : Nat := 3 + 5 + 2
def redBalls : Nat := 3
def probabilityOfRedBall : ℚ := redBalls / totalBalls

theorem probability_of_drawing_red_ball :
  probabilityOfRedBall = 3 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l176_176783


namespace ball_returns_to_bella_after_13_throws_l176_176120

def girl_after_throws (start : ℕ) (throws : ℕ) : ℕ :=
  (start + throws * 5) % 13

theorem ball_returns_to_bella_after_13_throws :
  girl_after_throws 1 13 = 1 :=
sorry

end ball_returns_to_bella_after_13_throws_l176_176120


namespace plane_equation_l176_176989

theorem plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) ∧
  (∀ x y z : ℤ, 
    (A * x + B * y + C * z + D = 0) ↔
      (x = 1 ∧ y = 6 ∧ z = -8 ∨ (∃ t : ℤ, 
        x = 2 + 4 * t ∧ y = 4 - t ∧ z = -3 + 5 * t))) ∧
  (A = 5 ∧ B = 15 ∧ C = -7 ∧ D = -151) :=
sorry

end plane_equation_l176_176989


namespace fanfan_home_distance_l176_176790

theorem fanfan_home_distance (x y z : ℝ) 
  (h1 : x / 3 = 10) 
  (h2 : x / 3 + y / 2 = 25) 
  (h3 : x / 3 + y / 2 + z = 85) :
  x + y + z = 120 :=
sorry

end fanfan_home_distance_l176_176790


namespace cos_sin_gt_sin_cos_l176_176032

theorem cos_sin_gt_sin_cos (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi) : Real.cos (Real.sin x) > Real.sin (Real.cos x) :=
by
  sorry

end cos_sin_gt_sin_cos_l176_176032


namespace hyperbola_parabola_focus_l176_176063

theorem hyperbola_parabola_focus (m : ℝ) :
  (m + (m - 2) = 4) → m = 3 :=
by
  intro h
  sorry

end hyperbola_parabola_focus_l176_176063


namespace initial_number_of_women_l176_176227

variable (W : ℕ)

def work_done_by_women_per_day (W : ℕ) : ℚ := 1 / (8 * W)
def work_done_by_children_per_day (W : ℕ) : ℚ := 1 / (12 * W)

theorem initial_number_of_women :
  (6 * work_done_by_women_per_day W + 3 * work_done_by_children_per_day W = 1 / 10) → W = 10 :=
by
  sorry

end initial_number_of_women_l176_176227


namespace gcd_poly_multiple_l176_176806

theorem gcd_poly_multiple {x : ℤ} (h : ∃ k : ℤ, x = 54321 * k) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 14)) x = 1 :=
sorry

end gcd_poly_multiple_l176_176806


namespace union_of_sets_l176_176448

noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 4, 6}

theorem union_of_sets : A ∪ B = {1, 2, 4, 6} := 
by 
sorry

end union_of_sets_l176_176448


namespace total_candy_eaten_by_bobby_l176_176698

-- Definitions based on the problem conditions
def candy_eaten_by_bobby_round1 : ℕ := 28
def candy_eaten_by_bobby_round2 : ℕ := 42
def chocolate_eaten_by_bobby : ℕ := 63

-- Define the statement to prove
theorem total_candy_eaten_by_bobby : 
  candy_eaten_by_bobby_round1 + candy_eaten_by_bobby_round2 + chocolate_eaten_by_bobby = 133 :=
  by
  -- Skipping the proof itself
  sorry

end total_candy_eaten_by_bobby_l176_176698


namespace max_correct_answers_l176_176118

/--
In a 50-question multiple-choice math contest, students receive 5 points for a correct answer, 
0 points for an answer left blank, and -2 points for an incorrect answer. Jesse’s total score 
on the contest was 115. Prove that the maximum number of questions that Jesse could have answered 
correctly is 30.
-/
theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 50) (h2 : 5 * a - 2 * c = 115) : a ≤ 30 :=
by
  sorry

end max_correct_answers_l176_176118


namespace sum_of_numbers_l176_176453

theorem sum_of_numbers : 145 + 33 + 29 + 13 = 220 :=
by
  sorry

end sum_of_numbers_l176_176453


namespace number_of_adults_in_family_l176_176983

-- Conditions as definitions
def total_apples : ℕ := 1200
def number_of_children : ℕ := 45
def apples_per_child : ℕ := 15
def apples_per_adult : ℕ := 5

-- Calculations based on conditions
def apples_eaten_by_children : ℕ := number_of_children * apples_per_child
def remaining_apples : ℕ := total_apples - apples_eaten_by_children
def number_of_adults : ℕ := remaining_apples / apples_per_adult

-- Proof target: number of adults in Bob's family equals 105
theorem number_of_adults_in_family : number_of_adults = 105 := by
  sorry

end number_of_adults_in_family_l176_176983


namespace determine_b_l176_176658

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x < 1 then 3 * x - b else 2 ^ x

theorem determine_b (b : ℝ) :
  f (f (5 / 6) b) b = 4 ↔ b = 1 / 2 :=
by sorry

end determine_b_l176_176658


namespace wash_cycle_time_l176_176903

-- Definitions for the conditions
def num_loads : Nat := 8
def dry_cycle_time_minutes : Nat := 60
def total_time_hours : Nat := 14
def total_time_minutes : Nat := total_time_hours * 60

-- The actual statement we need to prove
theorem wash_cycle_time (x : Nat) (h : num_loads * x + num_loads * dry_cycle_time_minutes = total_time_minutes) : x = 45 :=
by
  sorry

end wash_cycle_time_l176_176903


namespace inequality_proof_l176_176254

theorem inequality_proof (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 2) : 
  (1 / a + 1 / b) ≥ 2 :=
sorry

end inequality_proof_l176_176254


namespace age_solution_l176_176719

theorem age_solution :
  ∃ me you : ℕ, me + you = 63 ∧ 
  ∃ x : ℕ, me = 2 * x ∧ you = x ∧ me = 36 ∧ you = 27 :=
by
  sorry

end age_solution_l176_176719


namespace remaining_oak_trees_l176_176353

def initial_oak_trees : ℕ := 9
def cut_down_oak_trees : ℕ := 2

theorem remaining_oak_trees : initial_oak_trees - cut_down_oak_trees = 7 := 
by 
  sorry

end remaining_oak_trees_l176_176353


namespace conic_section_is_ellipse_l176_176881

theorem conic_section_is_ellipse (x y : ℝ) : 
  (x - 3)^2 + 9 * (y + 2)^2 = 144 →
  (∃ h k a b : ℝ, a = 12 ∧ b = 4 ∧ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) :=
by
  intro h_eq
  use 3, -2, 12, 4
  constructor
  { sorry }
  constructor
  { sorry }
  sorry

end conic_section_is_ellipse_l176_176881


namespace paperboy_problem_l176_176574

noncomputable def delivery_ways (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2
  else if n = 2 then 4
  else if n = 3 then 8
  else if n = 4 then 15
  else delivery_ways (n - 1) + delivery_ways (n - 2) + delivery_ways (n - 3) + delivery_ways (n - 4)

theorem paperboy_problem : delivery_ways 12 = 2872 :=
  sorry

end paperboy_problem_l176_176574


namespace percentage_apples_sold_l176_176480

noncomputable def original_apples : ℝ := 750
noncomputable def remaining_apples : ℝ := 300

theorem percentage_apples_sold (A P : ℝ) (h1 : A = 750) (h2 : A * (1 - P / 100) = 300) : 
  P = 60 :=
by
  sorry

end percentage_apples_sold_l176_176480


namespace hexagon_angles_sum_l176_176080

theorem hexagon_angles_sum (α β γ δ ε ζ : ℝ)
  (h1 : α + γ + ε = 180)
  (h2 : β + δ + ζ = 180) : 
  α + β + γ + δ + ε + ζ = 360 :=
by 
  sorry

end hexagon_angles_sum_l176_176080


namespace supermarket_A_is_more_cost_effective_l176_176071

def price_A (kg : ℕ) : ℕ :=
  if kg <= 4 then kg * 10
  else 4 * 10 + (kg - 4) * 6

def price_B (kg : ℕ) : ℕ :=
  kg * 10 * 8 / 10

theorem supermarket_A_is_more_cost_effective :
  price_A 3 = 30 ∧ 
  price_A 5 = 46 ∧ 
  ∀ (x : ℕ), (x > 4) → price_A x = 6 * x + 16 ∧ 
  price_A 10 < price_B 10 :=
by 
  sorry

end supermarket_A_is_more_cost_effective_l176_176071


namespace total_doll_count_l176_176354

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l176_176354


namespace moles_of_NaCl_formed_l176_176813

theorem moles_of_NaCl_formed (hcl moles : ℕ) (nahco3 moles : ℕ) (reaction : ℕ → ℕ → ℕ) :
  hcl = 3 → nahco3 = 3 → reaction 1 1 = 1 →
  reaction hcl nahco3 = 3 :=
by 
  intros h1 h2 h3
  -- Proof omitted
  sorry

end moles_of_NaCl_formed_l176_176813


namespace part_I_part_II_l176_176492

variable {a b c : ℝ}
variable (habc : a ∈ Set.Ioi 0)
variable (hbbc : b ∈ Set.Ioi 0)
variable (hcbc : c ∈ Set.Ioi 0)
variable (h_sum : a + b + c = 1)

theorem part_I : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 :=
by
  sorry

theorem part_II : (a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2 :=
by
  sorry

end part_I_part_II_l176_176492


namespace cos_sum_arithmetic_seq_l176_176393

theorem cos_sum_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1 / 2 :=
  sorry

end cos_sum_arithmetic_seq_l176_176393


namespace symmetric_with_origin_l176_176214

-- Define the original point P
def P : ℝ × ℝ := (2, -3)

-- Define the function for finding the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Prove that the symmetric point of P with respect to the origin is (-2, 3)
theorem symmetric_with_origin :
  symmetric_point P = (-2, 3) :=
by
  -- Placeholders for proof
  sorry

end symmetric_with_origin_l176_176214


namespace count_special_digits_base7_l176_176017

theorem count_special_digits_base7 : 
  let n := 2401
  let total_valid_numbers := n - 4^4
  total_valid_numbers = 2145 :=
by
  sorry

end count_special_digits_base7_l176_176017


namespace donovan_lap_time_l176_176351

-- Definitions based on problem conditions
def lap_time_michael := 40  -- Michael's lap time in seconds
def laps_michael := 9       -- Laps completed by Michael to pass Donovan
def laps_donovan := 8       -- Laps completed by Donovan in the same time

-- Condition based on the solution
def race_duration := laps_michael * lap_time_michael

-- define the conjecture
theorem donovan_lap_time : 
  (race_duration = laps_donovan * 45) := 
sorry

end donovan_lap_time_l176_176351


namespace find_a_b_c_l176_176220

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hp1 : is_prime (a + b * c))
  (hp2 : is_prime (b + a * c))
  (hp3 : is_prime (c + a * b))
  (hdiv1 : (a + b * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv2 : (b + a * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv3 : (c + a * b) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1))) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end find_a_b_c_l176_176220


namespace smallest_next_divisor_l176_176079

def isOddFourDigitNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1000 ≤ n ∧ n < 10000

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => d > 0 ∧ n % d = 0)

theorem smallest_next_divisor (m : ℕ) (h₁ : isOddFourDigitNumber m) (h₂ : 437 ∈ divisors m) :
  ∃ k, k > 437 ∧ k ∈ divisors m ∧ k % 2 = 1 ∧ ∀ n, n > 437 ∧ n < k → n ∉ divisors m := by
  sorry

end smallest_next_divisor_l176_176079


namespace perimeter_of_one_of_the_rectangles_l176_176749

noncomputable def perimeter_of_rectangle (z w : ℕ) : ℕ :=
  2 * z

theorem perimeter_of_one_of_the_rectangles (z w : ℕ) :
  ∃ P, P = perimeter_of_rectangle z w :=
by
  use 2 * z
  sorry

end perimeter_of_one_of_the_rectangles_l176_176749


namespace shiela_used_seven_colors_l176_176667

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ) 
    (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : 
    total_blocks / blocks_per_color = 7 :=
by
  sorry

end shiela_used_seven_colors_l176_176667


namespace simplify_expression_l176_176836

theorem simplify_expression
  (h0 : (Real.pi / 2) < 2 ∧ 2 < Real.pi)  -- Given conditions on 2 related to π.
  (h1 : Real.sin 2 > 0)  -- Given condition that sin 2 is positive.
  (h2 : Real.cos 2 < 0)  -- Given condition that cos 2 is negative.
  : 2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 :=
sorry

end simplify_expression_l176_176836


namespace sum_S9_l176_176158

variable (a : ℕ → ℤ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Given condition for the sum of specific terms
def condition_given (a : ℕ → ℤ) : Prop :=
  a 2 + a 5 + a 8 = 12

-- Sum of the first 9 terms
def sum_of_first_nine_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8

-- Problem statement: Prove that given the arithmetic sequence and the condition,
-- the sum of the first 9 terms is 36
theorem sum_S9 :
  arithmetic_sequence a → condition_given a → sum_of_first_nine_terms a = 36 :=
by
  intros
  sorry

end sum_S9_l176_176158


namespace sin_16_over_3_pi_l176_176199

theorem sin_16_over_3_pi : Real.sin (16 / 3 * Real.pi) = -Real.sqrt 3 / 2 := 
sorry

end sin_16_over_3_pi_l176_176199


namespace count_valid_three_digit_numbers_l176_176847

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 36 ∧ 
    (∀ (a b c : ℕ), a ≠ 0 ∧ c ≠ 0 → 
    ((10 * b + c) % 4 = 0 ∧ (10 * b + a) % 4 = 0) → 
    n = 36) :=
sorry

end count_valid_three_digit_numbers_l176_176847


namespace solution_set_of_abs_x_gt_1_l176_176776

theorem solution_set_of_abs_x_gt_1 (x : ℝ) : |x| > 1 ↔ x > 1 ∨ x < -1 := 
sorry

end solution_set_of_abs_x_gt_1_l176_176776


namespace shorter_piece_length_l176_176753

theorem shorter_piece_length (x : ℝ) :
  (120 - (2 * x + 15) = x) → x = 35 := 
by
  intro h
  sorry

end shorter_piece_length_l176_176753


namespace benny_days_worked_l176_176510

/-- Benny works 3 hours a day and in total he worked for 18 hours. 
We need to prove that he worked for 6 days. -/
theorem benny_days_worked (hours_per_day : ℕ) (total_hours : ℕ)
  (h1 : hours_per_day = 3)
  (h2 : total_hours = 18) :
  total_hours / hours_per_day = 6 := 
by sorry

end benny_days_worked_l176_176510


namespace line_equation_l176_176246

theorem line_equation
  (t : ℝ)
  (x : ℝ) (y : ℝ)
  (h1 : x = 3 * t + 6)
  (h2 : y = 5 * t - 10) :
  y = (5 / 3) * x - 20 :=
sorry

end line_equation_l176_176246


namespace dot_product_eq_eight_l176_176993

def vec_a : ℝ × ℝ := (0, 4)
def vec_b : ℝ × ℝ := (2, 2)

theorem dot_product_eq_eight : (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) = 8 := by
  sorry

end dot_product_eq_eight_l176_176993


namespace warehouse_length_l176_176337

theorem warehouse_length (L W : ℕ) (times supposed_times : ℕ) (total_distance : ℕ)
  (h1 : W = 400)
  (h2 : supposed_times = 10)
  (h3 : times = supposed_times - 2)
  (h4 : total_distance = times * (2 * L + 2 * W))
  (h5 : total_distance = 16000) :
  L = 600 := by
  sorry

end warehouse_length_l176_176337


namespace grant_earnings_proof_l176_176414

noncomputable def total_earnings (X Y Z W : ℕ): ℕ :=
  let first_month := X
  let second_month := 3 * X + Y
  let third_month := 2 * second_month - Z
  let average := (first_month + second_month + third_month) / 3
  let fourth_month := average + W
  first_month + second_month + third_month + fourth_month

theorem grant_earnings_proof : total_earnings 350 30 20 50 = 5810 := by
  sorry

end grant_earnings_proof_l176_176414


namespace range_of_a_l176_176888

theorem range_of_a {a : ℝ} : (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≤ -x2^2 + 4*a*x2)
  ∨ (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≥ -x2^2 + 4*a*x2) ↔ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l176_176888


namespace students_taking_neither_l176_176931

theorem students_taking_neither (total_students music art science music_and_art music_and_science art_and_science three_subjects : ℕ)
  (h1 : total_students = 800)
  (h2 : music = 80)
  (h3 : art = 60)
  (h4 : science = 50)
  (h5 : music_and_art = 30)
  (h6 : music_and_science = 25)
  (h7 : art_and_science = 20)
  (h8 : three_subjects = 15) :
  total_students - (music + art + science - music_and_art - music_and_science - art_and_science + three_subjects) = 670 :=
by sorry

end students_taking_neither_l176_176931


namespace rationalize_denominator_and_product_l176_176179

theorem rationalize_denominator_and_product :
  let A := -11
  let B := -5
  let C := 5
  let expr := (3 + Real.sqrt 5) / (2 - Real.sqrt 5)
  (expr * (2 + Real.sqrt 5) / (2 + Real.sqrt 5) = A + B * Real.sqrt C) ∧ (A * B * C = 275) :=
by
  sorry

end rationalize_denominator_and_product_l176_176179


namespace unique_element_a_values_set_l176_176980

open Set

theorem unique_element_a_values_set :
  {a : ℝ | ∃! x : ℝ, a * x^2 + 2 * x - a = 0} = {0} :=
by
  sorry

end unique_element_a_values_set_l176_176980


namespace product_8_40_product_5_1_6_sum_6_instances_500_l176_176858

-- The product of 8 and 40 is 320
theorem product_8_40 : 8 * 40 = 320 := sorry

-- 5 times 1/6 is 5/6
theorem product_5_1_6 : 5 * (1 / 6) = 5 / 6 := sorry

-- The sum of 6 instances of 500 ends with 3 zeros and the sum is 3000
theorem sum_6_instances_500 :
  (500 * 6 = 3000) ∧ ((3000 % 1000) = 0) := sorry

end product_8_40_product_5_1_6_sum_6_instances_500_l176_176858


namespace sqrt_of_n_is_integer_l176_176920

theorem sqrt_of_n_is_integer (n : ℕ) (h : ∀ p, (0 ≤ p ∧ p < n) → ∃ m g, m + g = n ∧ (m - g) * (m - g) = n) :
  ∃ k : ℕ, k * k = n :=
by 
  sorry

end sqrt_of_n_is_integer_l176_176920


namespace R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l176_176769

theorem R_H_nonneg_def (H : ℝ) (s t : ℝ) (hH : 0 < H ∧ H ≤ 1) :
  (1 / 2) * (|t| ^ (2 * H) + |s| ^ (2 * H) - |t - s| ^ (2 * H)) ≥ 0 := sorry

theorem R_K_nonneg_def (K : ℝ) (s t : ℝ) (hK : 0 < K ∧ K ≤ 2) :
  (1 / 2 ^ K) * (|t + s| ^ K - |t - s| ^ K) ≥ 0 := sorry

theorem R_HK_nonneg_def (H K : ℝ) (s t : ℝ) (hHK : 0 < H ∧ H ≤ 1 ∧ 0 < K ∧ K ≤ 1) :
  (1 / 2 ^ K) * ( (|t| ^ (2 * H) + |s| ^ (2 * H)) ^ K - |t - s| ^ (2 * H * K) ) ≥ 0 := sorry

end R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l176_176769


namespace conic_section_is_parabola_l176_176855

def isParabola (equation : String) : Prop := 
  equation = "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)"

theorem conic_section_is_parabola : isParabola "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)" :=
  by
  sorry

end conic_section_is_parabola_l176_176855


namespace x_intercept_is_34_l176_176402

-- Definitions of the initial line, rotation, and point.
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 50 = 0

def rotation_angle : ℝ := 30
def rotation_center : ℝ × ℝ := (10, 10)

-- Define the slope of the line l
noncomputable def slope_of_l : ℝ := 4 / 3

-- Define the slope of the line m after rotating line l by 30 degrees counterclockwise
noncomputable def tan_30 : ℝ := 1 / Real.sqrt 3
noncomputable def slope_of_m : ℝ := (slope_of_l + tan_30) / (1 - slope_of_l * tan_30)

-- Assume line m goes through the point (rotation_center.x, rotation_center.y)
-- This defines line m
def line_m (x y : ℝ) : Prop := y - rotation_center.2 = slope_of_m * (x - rotation_center.1)

-- To find the x-intercept of line m, we set y = 0 and solve for x
noncomputable def x_intercept_of_m : ℝ := rotation_center.1 - rotation_center.2 / slope_of_m

-- Proof statement that the x-intercept of line m is 34
theorem x_intercept_is_34 : x_intercept_of_m = 34 :=
by
  -- This would be the proof, but for now we leave it as sorry
  sorry

end x_intercept_is_34_l176_176402


namespace al_initial_portion_l176_176616

theorem al_initial_portion (a b c : ℝ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 200 + 2 * b + 1.5 * c = 1800) : 
  a = 600 :=
sorry

end al_initial_portion_l176_176616


namespace range_of_a_l176_176161

open Real

theorem range_of_a
  (a : ℝ)
  (curve : ∀ θ : ℝ, ∃ p : ℝ × ℝ, p = (a + 2 * cos θ, a + 2 * sin θ))
  (distance_two_points : ∀ θ : ℝ, dist (0,0) (a + 2 * cos θ, a + 2 * sin θ) = 2) :
  (-2 * sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * sqrt 2) :=
sorry

end range_of_a_l176_176161


namespace shifting_parabola_l176_176234

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shifted_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem shifting_parabola : ∀ x : ℝ, shifted_function x = original_function (x + 2) - 1 := 
by 
  sorry

end shifting_parabola_l176_176234


namespace packages_bought_l176_176754

theorem packages_bought (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) : 
  (total_tshirts / tshirts_per_package) = 71 :=
by 
  sorry

end packages_bought_l176_176754


namespace effective_price_l176_176376

-- Definitions based on conditions
def upfront_payment (C : ℝ) := 0.20 * C = 240
def cashback (C : ℝ) := 0.10 * C

-- Problem statement
theorem effective_price (C : ℝ) (h₁ : upfront_payment C) : C - cashback C = 1080 :=
by
  sorry

end effective_price_l176_176376


namespace ray_two_digit_number_l176_176618

theorem ray_two_digit_number (a b n : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hn : n = 10 * a + b) (h1 : n = 4 * (a + b) + 3) (h2 : n + 18 = 10 * b + a) : n = 35 := by
  sorry

end ray_two_digit_number_l176_176618


namespace find_z_l176_176738

variable (x y z : ℝ)

theorem find_z (h1 : 12 * 40 = 480)
    (h2 : 15 * 50 = 750)
    (h3 : x + y + z = 270)
    (h4 : x + y = 100) :
    z = 170 := by
  sorry

end find_z_l176_176738


namespace choose_3_out_of_13_l176_176077

theorem choose_3_out_of_13: (Nat.choose 13 3) = 286 :=
by
  sorry

end choose_3_out_of_13_l176_176077


namespace find_theta_in_interval_l176_176049

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end find_theta_in_interval_l176_176049


namespace num_digits_expr_l176_176267

noncomputable def num_digits (n : ℕ) : ℕ :=
  (Int.ofNat n).natAbs.digits 10 |>.length

def expr : ℕ := 2^15 * 5^10 * 12

theorem num_digits_expr : num_digits expr = 13 := by
  sorry

end num_digits_expr_l176_176267


namespace plane_distance_l176_176593

theorem plane_distance (D : ℕ) (h₁ : D / 300 + D / 400 = 7) : D = 1200 :=
sorry

end plane_distance_l176_176593


namespace ways_to_draw_at_least_two_defective_l176_176310

-- Definitions based on the conditions of the problem
def total_products : ℕ := 100
def defective_products : ℕ := 3
def selected_products : ℕ := 5

-- Binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to prove
theorem ways_to_draw_at_least_two_defective :
  C defective_products 2 * C (total_products - defective_products) 3 + C defective_products 3 * C (total_products - defective_products) 2 =
  (C total_products selected_products - C defective_products 1 * C (total_products - defective_products) 4) :=
sorry

end ways_to_draw_at_least_two_defective_l176_176310


namespace total_trees_in_gray_areas_l176_176673

theorem total_trees_in_gray_areas (x y : ℕ) (h1 : 82 + x = 100) (h2 : 82 + y = 90) :
  x + y = 26 :=
by
  sorry

end total_trees_in_gray_areas_l176_176673


namespace proof_statement_d_is_proposition_l176_176919

-- Define the conditions
def statement_a := "Do two points determine a line?"
def statement_b := "Take a point M on line AB"
def statement_c := "In the same plane, two lines do not intersect"
def statement_d := "The sum of two acute angles is greater than a right angle"

-- Define the property of being a proposition
def is_proposition (s : String) : Prop :=
  s ≠ "Do two points determine a line?" ∧
  s ≠ "Take a point M on line AB" ∧
  s ≠ "In the same plane, two lines do not intersect"

-- The equivalence proof that statement_d is the only proposition
theorem proof_statement_d_is_proposition :
  is_proposition statement_d ∧
  ¬is_proposition statement_a ∧
  ¬is_proposition statement_b ∧
  ¬is_proposition statement_c := by
  sorry

end proof_statement_d_is_proposition_l176_176919


namespace ratio_of_areas_l176_176916

theorem ratio_of_areas (r : ℝ) (h : r > 0) :
  let R1 := r
  let R2 := 3 * r
  let S1 := 6 * R1
  let S2 := 6 * r
  let area_smaller_circle := π * R2 ^ 2
  let area_larger_square := S2 ^ 2
  (area_smaller_circle / area_larger_square) = π / 4 :=
by
  sorry

end ratio_of_areas_l176_176916


namespace five_minus_x_eight_l176_176135

theorem five_minus_x_eight (x y : ℤ) (h1 : 5 + x = 3 - y) (h2 : 2 + y = 6 + x) : 5 - x = 8 :=
by
  sorry

end five_minus_x_eight_l176_176135


namespace intersecting_chords_l176_176988

theorem intersecting_chords (n : ℕ) (h1 : 0 < n) :
  ∃ intersecting_points : ℕ, intersecting_points ≥ n :=
  sorry

end intersecting_chords_l176_176988


namespace mean_volume_of_cubes_l176_176929

theorem mean_volume_of_cubes (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) :
  ((a^3 + b^3 + c^3) / 3) = 135 :=
by
  -- known cube volumes and given edge lengths conditions
  sorry

end mean_volume_of_cubes_l176_176929


namespace man_older_than_son_l176_176288

theorem man_older_than_son (S M : ℕ) (h1 : S = 23) (h2 : M + 2 = 2 * (S + 2)) : M - S = 25 :=
by
  sorry

end man_older_than_son_l176_176288


namespace fraction_meaningful_range_l176_176258

-- Define the condition
def meaningful_fraction_condition (x : ℝ) : Prop := (x - 2023) ≠ 0

-- Define the conclusion that we need to prove
def meaningful_fraction_range (x : ℝ) : Prop := x ≠ 2023

theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction_condition x → meaningful_fraction_range x :=
by
  intro h
  -- Proof steps would go here
  sorry

end fraction_meaningful_range_l176_176258


namespace pieces_of_wood_for_table_l176_176061

theorem pieces_of_wood_for_table :
  ∀ (T : ℕ), (24 * T + 48 * 8 = 672) → T = 12 :=
by
  intro T
  intro h
  sorry

end pieces_of_wood_for_table_l176_176061


namespace bank_policy_advantageous_for_retirees_l176_176833

theorem bank_policy_advantageous_for_retirees
  (special_programs : Prop)
  (higher_deposit_rates : Prop)
  (lower_credit_rates : Prop)
  (reliable_loan_payers : Prop)
  (stable_income : Prop)
  (family_interest : Prop)
  (savings_tendency : Prop)
  (regular_income : Prop)
  (long_term_deposits : Prop) :
  reliable_loan_payers ∧ stable_income ∧ family_interest ∧ savings_tendency ∧ regular_income ∧ long_term_deposits → 
  special_programs ∧ higher_deposit_rates ∧ lower_credit_rates :=
sorry

end bank_policy_advantageous_for_retirees_l176_176833


namespace martha_initial_juice_pantry_l176_176625

theorem martha_initial_juice_pantry (P : ℕ) : 
  4 + P + 5 - 3 = 10 → P = 4 := 
by
  intro h
  sorry

end martha_initial_juice_pantry_l176_176625


namespace cups_of_flour_per_pound_of_pasta_l176_176319

-- Definitions from conditions
def pounds_of_pasta_per_rack : ℕ := 3
def racks_owned : ℕ := 3
def additional_rack_needed : ℕ := 1
def cups_per_bag : ℕ := 8
def bags_used : ℕ := 3

-- Derived definitions from above conditions
def total_cups_of_flour : ℕ := bags_used * cups_per_bag  -- 24 cups
def total_racks_needed : ℕ := racks_owned + additional_rack_needed  -- 4 racks
def total_pounds_of_pasta : ℕ := total_racks_needed * pounds_of_pasta_per_rack  -- 12 pounds

theorem cups_of_flour_per_pound_of_pasta (x : ℕ) :
  (total_cups_of_flour / total_pounds_of_pasta) = x → x = 2 :=
by
  intro h
  sorry

end cups_of_flour_per_pound_of_pasta_l176_176319


namespace verify_statements_l176_176406

theorem verify_statements (S : Set ℝ) (m l : ℝ) (hS : ∀ x, x ∈ S → x^2 ∈ S) :
  (m = 1 → S = {1}) ∧
  (m = -1/2 → (1/4 ≤ l ∧ l ≤ 1)) ∧
  (l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) ∧
  (l = 1 → -1 ≤ m ∧ m ≤ 1) :=
  sorry

end verify_statements_l176_176406


namespace rationalize_denominator_l176_176653

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end rationalize_denominator_l176_176653


namespace factor_expression_correct_l176_176493

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_expression_correct (a b c : ℝ) :
  factor_expression a b c = (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_correct_l176_176493


namespace perimeter_of_regular_polygon_l176_176331

/-- 
Given a regular polygon with a central angle of 45 degrees and a side length of 5,
the perimeter of the polygon is 40.
-/
theorem perimeter_of_regular_polygon 
  (central_angle : ℝ) (side_length : ℝ) (h1 : central_angle = 45)
  (h2 : side_length = 5) :
  ∃ P, P = 40 :=
by
  sorry

end perimeter_of_regular_polygon_l176_176331


namespace sum_of_numbers_l176_176236

theorem sum_of_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end sum_of_numbers_l176_176236


namespace calculate_fraction_l176_176173

variable (a b : ℝ)

theorem calculate_fraction (h : a ≠ b) : (2 * a / (a - b)) + (2 * b / (b - a)) = 2 := by
  sorry

end calculate_fraction_l176_176173


namespace sin_180_degree_l176_176586

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l176_176586


namespace bankers_discount_problem_l176_176470

theorem bankers_discount_problem
  (BD : ℚ) (TD : ℚ) (SD : ℚ)
  (h1 : BD = 36)
  (h2 : TD = 30)
  (h3 : BD = TD + TD^2 / SD) :
  SD = 150 := 
sorry

end bankers_discount_problem_l176_176470


namespace rectangular_solid_volume_l176_176683

theorem rectangular_solid_volume
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : b = 2 * a) :
  a * b * c = 12 := 
by
  sorry

end rectangular_solid_volume_l176_176683


namespace union_of_M_and_N_l176_176459

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_M_and_N : M ∪ N = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_M_and_N_l176_176459


namespace geometric_sequence_a2_l176_176568

theorem geometric_sequence_a2 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h1 : a 1 = 1/4) 
  (h3_h5 : a 3 * a 5 = 4 * (a 4 - 1)) 
  (h_seq : ∀ n : ℕ, a n = a 1 * q ^ (n - 1)) :
  a 2 = 1/2 :=
sorry

end geometric_sequence_a2_l176_176568


namespace customers_who_bought_four_paintings_each_l176_176484

/-- Tracy's art fair conditions:
- 20 people came to look at the art
- Four customers bought two paintings each
- Twelve customers bought one painting each
- Tracy sold a total of 36 paintings

We need to prove the number of customers who bought four paintings each. -/
theorem customers_who_bought_four_paintings_each:
  let total_customers := 20
  let customers_bought_two_paintings := 4
  let customers_bought_one_painting := 12
  let total_paintings_sold := 36
  let paintings_per_customer_buying_two := 2
  let paintings_per_customer_buying_one := 1
  let paintings_per_customer_buying_four := 4
  (customers_bought_two_paintings * paintings_per_customer_buying_two +
   customers_bought_one_painting * paintings_per_customer_buying_one +
   x * paintings_per_customer_buying_four = total_paintings_sold) →
  (customers_bought_two_paintings + customers_bought_one_painting + x = total_customers) →
  x = 4 :=
by
  intro h1 h2
  sorry

end customers_who_bought_four_paintings_each_l176_176484


namespace largest_angle_triangle_l176_176156

-- Definition of constants and conditions
def right_angle : ℝ := 90
def angle_sum : ℝ := 120
def angle_difference : ℝ := 20

-- Given two angles of a triangle sum to 120 degrees and one is 20 degrees greater than the other,
-- Prove the largest angle in the triangle is 70 degrees
theorem largest_angle_triangle (A B C : ℝ) (hA : A + B = angle_sum) (hB : B = A + angle_difference) (hC : A + B + C = 180) : 
  max A (max B C) = 70 := 
by 
  sorry

end largest_angle_triangle_l176_176156


namespace maximum_value_l176_176031

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_value (a b c : ℝ) (h_a : 1 ≤ a ∧ a ≤ 2)
  (h_f1 : f a b c 1 ≤ 1) (h_f2 : f a b c 2 ≤ 1) :
  7 * b + 5 * c ≤ -6 :=
sorry

end maximum_value_l176_176031


namespace vector_parallel_eq_l176_176471

theorem vector_parallel_eq (m : ℝ) : 
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  a.1 * b.2 = a.2 * b.1 -> m = -6 := 
by 
  sorry

end vector_parallel_eq_l176_176471


namespace solution_set_of_inequality_l176_176322

theorem solution_set_of_inequality (x : ℝ) (h : 3 * x + 2 > 5) : x > 1 :=
sorry

end solution_set_of_inequality_l176_176322


namespace sum_a4_a5_a6_l176_176343

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 21)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 63 := by
  sorry

end sum_a4_a5_a6_l176_176343


namespace inequality_proof_l176_176520

open Real

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c) :
  sqrt (a * b * c) * (sqrt a + sqrt b + sqrt c) + (a + b + c) ^ 2 ≥ 
  4 * sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end inequality_proof_l176_176520


namespace bill_toilet_paper_duration_l176_176487

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end bill_toilet_paper_duration_l176_176487


namespace smallest_positive_integer_l176_176638

theorem smallest_positive_integer (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 7 = 4) ∧ (N % 11 = 9) → N = 207 :=
by
  sorry

end smallest_positive_integer_l176_176638


namespace restaurant_made_correct_amount_l176_176551

noncomputable def restaurant_revenue : ℝ := 
  let price1 := 8
  let qty1 := 10
  let price2 := 10
  let qty2 := 5
  let price3 := 4
  let qty3 := 20
  let total_sales := qty1 * price1 + qty2 * price2 + qty3 * price3
  let discount := 0.10
  let discounted_total := total_sales * (1 - discount)
  let sales_tax := 0.05
  let final_amount := discounted_total * (1 + sales_tax)
  final_amount

theorem restaurant_made_correct_amount : restaurant_revenue = 198.45 := by
  sorry

end restaurant_made_correct_amount_l176_176551


namespace books_sold_correct_l176_176094

-- Define the number of books sold by Matias, Olivia, and Luke on each day
def matias_monday := 7
def olivia_monday := 5
def luke_monday := 12

def matias_tuesday := 2 * matias_monday
def olivia_tuesday := 3 * olivia_monday
def luke_tuesday := luke_monday / 2

def matias_wednesday := 3 * matias_tuesday
def olivia_wednesday := 4 * olivia_tuesday
def luke_wednesday := luke_tuesday

-- Calculate the total books sold by each person over three days
def matias_total := matias_monday + matias_tuesday + matias_wednesday
def olivia_total := olivia_monday + olivia_tuesday + olivia_wednesday
def luke_total := luke_monday + luke_tuesday + luke_wednesday

-- Calculate the combined total of books sold by Matias, Olivia, and Luke
def combined_total := matias_total + olivia_total + luke_total

-- Prove the combined total equals 167
theorem books_sold_correct : combined_total = 167 := by
  sorry

end books_sold_correct_l176_176094


namespace mistaken_divisor_l176_176027

theorem mistaken_divisor (x : ℕ) (h : 49 * x = 28 * 21) : x = 12 :=
sorry

end mistaken_divisor_l176_176027


namespace sqrt_expression_eval_l176_176838

theorem sqrt_expression_eval :
    (Real.sqrt 8 - 2 * Real.sqrt (1 / 2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3)) = Real.sqrt 2 + 1 := 
by
  sorry

end sqrt_expression_eval_l176_176838


namespace percent_with_university_diploma_l176_176565

theorem percent_with_university_diploma (a b c d : ℝ) (h1 : a = 0.12) (h2 : b = 0.25) (h3 : c = 0.40) 
    (h4 : d = c - a) (h5 : ¬c = 1) : 
    d + (b * (1 - c)) = 0.43 := 
by 
    sorry

end percent_with_university_diploma_l176_176565


namespace quadratic_real_roots_quadratic_product_of_roots_l176_176131

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 2 * m * x + m^2 + m - 3 = 0) ↔ m ≤ 3 := by
{
  sorry
}

theorem quadratic_product_of_roots (m : ℝ) (α β : ℝ) :
  α * β = 17 ∧ α^2 - 2 * m * α + m^2 + m - 3 = 0 ∧ β^2 - 2 * m * β + m^2 + m - 3 = 0 →
  m = -5 := by
{
  sorry
}

end quadratic_real_roots_quadratic_product_of_roots_l176_176131


namespace m_value_for_power_function_l176_176828

theorem m_value_for_power_function (m : ℝ) :
  (3 * m - 1 = 1) → (m = 2 / 3) :=
by
  sorry

end m_value_for_power_function_l176_176828


namespace interest_earned_is_91_dollars_l176_176687

-- Define the initial conditions
def P : ℝ := 2000
def r : ℝ := 0.015
def n : ℕ := 3

-- Define the compounded amount function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Prove the interest earned after 3 years is 91 dollars
theorem interest_earned_is_91_dollars : 
  (compound_interest P r n) - P = 91 :=
by
  sorry

end interest_earned_is_91_dollars_l176_176687


namespace max_value_b_exists_l176_176746

theorem max_value_b_exists :
  ∃ a c : ℝ, ∃ b : ℝ, 
  (∀ x : ℤ, 
  ((x^4 - a * x^3 - b * x^2 - c * x - 2007) = 0) → 
  ∃ r s t : ℤ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  ((x = r) ∨ (x = s) ∨ (x = t))) ∧ 
  (∀ b' : ℝ, b' < b → 
  ¬ ( ∃ a' c' : ℝ, ( ∀ x : ℤ, 
  ((x^4 - a' * x^3 - b' * x^2 - c' * x - 2007) = 0) → 
  ∃ r' s' t' : ℤ, r' ≠ s' ∧ s' ≠ t' ∧ r' ≠ t' ∧ 
  ((x = r') ∨ (x = s') ∨ (x = t') )))) ∧ b = 3343 :=
sorry

end max_value_b_exists_l176_176746


namespace relationship_between_M_and_N_l176_176587

variable (x y : ℝ)

theorem relationship_between_M_and_N (h1 : x ≠ 3) (h2 : y ≠ -2)
  (M : ℝ) (hm : M = x^2 + y^2 - 6 * x + 4 * y)
  (N : ℝ) (hn : N = -13) : M > N :=
by
  sorry

end relationship_between_M_and_N_l176_176587


namespace snack_eaters_remaining_l176_176689

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end snack_eaters_remaining_l176_176689


namespace fg_diff_zero_l176_176904

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x + 3

theorem fg_diff_zero (x : ℝ) : f (g x) - g (f x) = 0 :=
by
  sorry

end fg_diff_zero_l176_176904


namespace find_y_l176_176982

variables (x y : ℝ)

theorem find_y (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 :=
by
  sorry

end find_y_l176_176982


namespace max_value_of_x2_plus_y2_l176_176444

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + y^2

theorem max_value_of_x2_plus_y2 {x y : ℝ} (h : 5*x^2 + 4*y^2 = 10*x) : max_value x y ≤ 4 := sorry

end max_value_of_x2_plus_y2_l176_176444


namespace jane_age_l176_176023

theorem jane_age (j : ℕ) 
  (h₁ : ∃ (k : ℕ), j - 2 = k^2)
  (h₂ : ∃ (m : ℕ), j + 2 = m^3) :
  j = 6 :=
sorry

end jane_age_l176_176023


namespace quadratic_has_real_roots_range_l176_176859

-- Lean 4 statement

theorem quadratic_has_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) → m ≤ 7 ∧ m ≠ 3 :=
by
  sorry

end quadratic_has_real_roots_range_l176_176859


namespace alpha_value_l176_176218

theorem alpha_value (b : ℝ) : (∀ x : ℝ, (|2 * x - 3| < 2) ↔ (x^2 + -3 * x + b < 0)) :=
by
  sorry

end alpha_value_l176_176218


namespace area_of_rectangle_ABCD_l176_176442

-- Definitions for the conditions
def small_square_area := 4
def total_small_squares := 2
def large_square_area := (2 * (2 : ℝ)) * (2 * (2 : ℝ))
def total_squares_area := total_small_squares * small_square_area + large_square_area

-- The main proof statement
theorem area_of_rectangle_ABCD : total_squares_area = 24 := 
by
  sorry

end area_of_rectangle_ABCD_l176_176442


namespace cost_price_equal_l176_176160

theorem cost_price_equal (total_selling_price : ℝ) (profit_percent_first profit_percent_second : ℝ) (length_first_segment length_second_segment : ℝ) (C : ℝ) :
  total_selling_price = length_first_segment * (1 + profit_percent_first / 100) * C + length_second_segment * (1 + profit_percent_second / 100) * C →
  C = 15360 / (66 + 72) :=
by {
  sorry
}

end cost_price_equal_l176_176160


namespace sqrt_20n_integer_exists_l176_176626

theorem sqrt_20n_integer_exists : 
  ∃ n : ℤ, 0 ≤ n ∧ ∃ k : ℤ, k * k = 20 * n :=
sorry

end sqrt_20n_integer_exists_l176_176626


namespace b_share_1500_l176_176728

theorem b_share_1500 (total_amount : ℕ) (parts_A parts_B parts_C : ℕ)
  (h_total_amount : total_amount = 4500)
  (h_ratio : (parts_A, parts_B, parts_C) = (2, 3, 4)) :
  parts_B * (total_amount / (parts_A + parts_B + parts_C)) = 1500 :=
by
  sorry

end b_share_1500_l176_176728


namespace number_one_fourth_less_than_25_percent_more_l176_176774

theorem number_one_fourth_less_than_25_percent_more (x : ℝ) :
  (3 / 4) * x = 1.25 * 80 → x = 133.33 :=
by
  intros h
  sorry

end number_one_fourth_less_than_25_percent_more_l176_176774


namespace final_price_correct_l176_176740

def original_cost : ℝ := 2.00
def discount : ℝ := 0.57
def final_price : ℝ := 1.43

theorem final_price_correct :
  original_cost - discount = final_price :=
by
  sorry

end final_price_correct_l176_176740


namespace find_a_2_find_a_n_l176_176137

-- Define the problem conditions and questions as types
def S_3 (a_1 a_2 a_3 : ℝ) : Prop := a_1 + a_2 + a_3 = 7
def arithmetic_mean_condition (a_1 a_2 a_3 : ℝ) : Prop :=
  (a_1 + 3 + a_3 + 4) / 2 = 3 * a_2

-- Prove that a_2 = 2 given the conditions
theorem find_a_2 (a_1 a_2 a_3 : ℝ) (h1 : S_3 a_1 a_2 a_3) (h2: arithmetic_mean_condition a_1 a_2 a_3) :
  a_2 = 2 := 
sorry

-- Define the general term for a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Prove the formula for the general term of the geometric sequence given the conditions and a_2 found
theorem find_a_n (a : ℕ → ℝ) (q : ℝ) (h1 : S_3 (a 1) (a 2) (a 3)) (h2 : arithmetic_mean_condition (a 1) (a 2) (a 3)) (h3 : geometric_sequence a q) : 
  (q = (1/2) → ∀ n, a n = (1 / 2)^(n - 3))
  ∧ (q = 2 → ∀ n, a n = 2^(n - 1)) := 
sorry

end find_a_2_find_a_n_l176_176137


namespace find_number_l176_176665

theorem find_number (x : ℝ) (h : 15 * x = 300) : x = 20 :=
by 
  sorry

end find_number_l176_176665


namespace carl_garden_area_l176_176002

theorem carl_garden_area (total_posts : ℕ) (post_interval : ℕ) (x_posts_on_shorter : ℕ) (y_posts_on_longer : ℕ)
  (h1 : total_posts = 26)
  (h2 : post_interval = 5)
  (h3 : y_posts_on_longer = 2 * x_posts_on_shorter)
  (h4 : 2 * x_posts_on_shorter + 2 * y_posts_on_longer - 4 = total_posts) :
  (x_posts_on_shorter - 1) * post_interval * (y_posts_on_longer - 1) * post_interval = 900 := 
by
  sorry

end carl_garden_area_l176_176002


namespace qualified_light_bulb_prob_l176_176575

def prob_factory_A := 0.7
def prob_factory_B := 0.3
def qual_rate_A := 0.9
def qual_rate_B := 0.8

theorem qualified_light_bulb_prob :
  prob_factory_A * qual_rate_A + prob_factory_B * qual_rate_B = 0.87 :=
by
  sorry

end qualified_light_bulb_prob_l176_176575


namespace point_in_third_quadrant_l176_176231

section quadrant_problem

variables (a b : ℝ)

-- Given: Point (a, b) is in the fourth quadrant
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- To prove: Point (a / b, 2 * b - a) is in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- The theorem stating that if (a, b) is in the fourth quadrant,
-- then (a / b, 2 * b - a) is in the third quadrant
theorem point_in_third_quadrant (a b : ℝ) (h : in_fourth_quadrant a b) :
  in_third_quadrant (a / b) (2 * b - a) :=
  sorry

end quadrant_problem

end point_in_third_quadrant_l176_176231


namespace angle_triple_complement_l176_176416

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l176_176416


namespace find_constants_l176_176524

theorem find_constants
  (a_1 a_2 : ℚ)
  (h1 : 3 * a_1 - 3 * a_2 = 0)
  (h2 : 4 * a_1 + 7 * a_2 = 5) :
  a_1 = 5 / 11 ∧ a_2 = 5 / 11 :=
by
  sorry

end find_constants_l176_176524


namespace combined_weight_l176_176029

variable (J S : ℝ)

-- Given conditions
def jake_current_weight := (J = 152)
def lose_weight_equation := (J - 32 = 2 * S)

-- Question: combined weight of Jake and his sister
theorem combined_weight (h1 : jake_current_weight J) (h2 : lose_weight_equation J S) : J + S = 212 :=
by
  sorry

end combined_weight_l176_176029


namespace minimal_odd_sum_is_1683_l176_176196

/-!
# Proof Problem:
Prove that the minimal odd sum of two three-digit numbers and one four-digit number 
formed using the digits 0 through 9 exactly once is 1683.
-/
theorem minimal_odd_sum_is_1683 :
  ∃ (a b : ℕ) (c : ℕ), 
    100 ≤ a ∧ a < 1000 ∧ 
    100 ≤ b ∧ b < 1000 ∧ 
    1000 ≤ c ∧ c < 10000 ∧ 
    a + b + c % 2 = 1 ∧ 
    (∀ d e f : ℕ, 
      100 ≤ d ∧ d < 1000 ∧ 
      100 ≤ e ∧ e < 1000 ∧ 
      1000 ≤ f ∧ f < 10000 ∧ 
      d + e + f % 2 = 1 → a + b + c ≤ d + e + f) ∧ 
    a + b + c = 1683 := 
sorry

end minimal_odd_sum_is_1683_l176_176196


namespace find_E_coordinates_l176_176515

structure Point :=
(x : ℚ)
(y : ℚ)

def A : Point := { x := -2, y := 1 }
def B : Point := { x := 1, y := 4 }
def C : Point := { x := 4, y := -3 }

def D : Point := 
  let m : ℚ := 1
  let n : ℚ := 2
  let x1 := A.x
  let y1 := A.y
  let x2 := B.x
  let y2 := B.y
  { x := (m * x2 + n * x1) / (m + n), y := (m * y2 + n * y1) / (m + n) }

theorem find_E_coordinates : 
  let k : ℚ := 4
  let x_E : ℚ := (k * C.x + D.x) / (k + 1)
  let y_E : ℚ := (k * C.y + D.y) / (k + 1)
  ∃ E : Point, E.x = (17:ℚ) / 3 ∧ E.y = -(14:ℚ) / 3 :=
sorry

end find_E_coordinates_l176_176515


namespace no_valid_placement_of_prisms_l176_176215

-- Definitions: Rectangular prism with edges parallel to OX, OY, and OZ axes.
structure RectPrism :=
  (x_interval : Set ℝ)
  (y_interval : Set ℝ)
  (z_interval : Set ℝ)

-- Function to determine if two rectangular prisms intersect.
def intersects (P Q : RectPrism) : Prop :=
  ¬ Disjoint P.x_interval Q.x_interval ∧
  ¬ Disjoint P.y_interval Q.y_interval ∧
  ¬ Disjoint P.z_interval Q.z_interval

-- Definition of the 12 rectangular prisms
def prisms := Fin 12 → RectPrism

-- Conditions for intersection:
def intersection_condition (prisms : prisms) : Prop :=
  ∀ i : Fin 12, ∀ j : Fin 12,
    (j = (i + 1) % 12) ∨ (j = (i - 1 + 12) % 12) ∨ intersects (prisms i) (prisms j)

theorem no_valid_placement_of_prisms :
  ¬ ∃ (prisms : prisms), intersection_condition prisms :=
sorry

end no_valid_placement_of_prisms_l176_176215


namespace sum_of_edges_l176_176588

-- Define the properties of the rectangular solid
variables (a b c : ℝ)
variables (V : ℝ) (S : ℝ)

-- Set the conditions
def geometric_progression := (a * b * c = V) ∧ (2 * (a * b + b * c + c * a) = S) ∧ (∃ k : ℝ, k ≠ 0 ∧ a = b / k ∧ c = b * k)

-- Define the main proof statement
theorem sum_of_edges (hV : V = 1000) (hS : S = 600) (hg : geometric_progression a b c V S) : 
  4 * (a + b + c) = 120 :=
sorry

end sum_of_edges_l176_176588


namespace part1_part2_part3_l176_176856

open Set

-- Define the sets A and B and the universal set
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def U : Set ℝ := univ  -- Universal set R

theorem part1 : A ∩ B = { x | 3 ≤ x ∧ x < 7 } :=
by { sorry }

theorem part2 : U \ A = { x | x < 3 ∨ x ≥ 7 } :=
by { sorry }

theorem part3 : U \ (A ∪ B) = { x | x ≤ 2 ∨ x ≥ 10 } :=
by { sorry }

end part1_part2_part3_l176_176856


namespace hundredth_number_is_100_l176_176411

/-- Define the sequence of numbers said by Jo, Blair, and Parker following the conditions described. --/
def next_number (turn : ℕ) : ℕ :=
  -- Each turn increments by one number starting from 1
  turn

-- Prove that the 100th number in the sequence is 100
theorem hundredth_number_is_100 :
  next_number 100 = 100 := 
by sorry

end hundredth_number_is_100_l176_176411


namespace find_C_l176_176829

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 330) : C = 30 := 
sorry

end find_C_l176_176829


namespace price_on_hot_day_l176_176788

noncomputable def regular_price_P (P : ℝ) : Prop :=
  7 * 32 * (P - 0.75) + 3 * 32 * (1.25 * P - 0.75) = 450

theorem price_on_hot_day (P : ℝ) (h : regular_price_P P) : 1.25 * P = 2.50 :=
by sorry

end price_on_hot_day_l176_176788


namespace find_n_values_l176_176601

theorem find_n_values (n : ℕ) (h1 : 0 < n) : 
  (∃ (a : ℕ), n * 2^n + 1 = a * a) ↔ (n = 2 ∨ n = 3) := 
by
  sorry

end find_n_values_l176_176601


namespace find_starting_point_of_a_l176_176972

def point := ℝ × ℝ
def vector := ℝ × ℝ

def B : point := (1, 0)

def b : vector := (-3, -4)
def c : vector := (1, 1)

def a : vector := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

theorem find_starting_point_of_a (hb : b = (-3, -4)) (hc : c = (1, 1)) (hB : B = (1, 0)) :
    let a := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)
    let start_A := (B.1 - a.1, B.2 - a.2)
    start_A = (12, 14) :=
by
  rw [hb, hc, hB]
  let a := (3 * (-3) - 2 * (1), 3 * (-4) - 2 * (1))
  let start_A := (1 - a.1, 0 - a.2)
  simp [a]
  sorry

end find_starting_point_of_a_l176_176972


namespace find_divided_number_l176_176100

theorem find_divided_number:
  ∃ x : ℕ, (x % 127 = 6) ∧ (2037 % 127 = 5) ∧ x = 2038 :=
by
  sorry

end find_divided_number_l176_176100


namespace hyperbola_eccentricity_proof_l176_176636

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (b ^ 2 + (a / 2) ^ 2 = a ^ 2)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt ((a ^ 2 + b ^ 2) / a ^ 2)

theorem hyperbola_eccentricity_proof
  (a b : ℝ) (h : a > b ∧ b > 0) (h1 : ellipse_eccentricity a b h) :
  hyperbola_eccentricity a b = Real.sqrt 7 / 2 :=
by
  sorry

end hyperbola_eccentricity_proof_l176_176636


namespace middle_guards_hours_l176_176758

def total_hours := 9
def hours_first_guard := 3
def hours_last_guard := 2
def remaining_hours := total_hours - hours_first_guard - hours_last_guard
def num_middle_guards := 2

theorem middle_guards_hours : remaining_hours / num_middle_guards = 2 := by
  sorry

end middle_guards_hours_l176_176758


namespace original_pencils_count_l176_176413

theorem original_pencils_count (total_pencils : ℕ) (added_pencils : ℕ) (original_pencils : ℕ) : total_pencils = original_pencils + added_pencils → original_pencils = 2 :=
by
  sorry

end original_pencils_count_l176_176413


namespace circumradius_inradius_inequality_l176_176037

theorem circumradius_inradius_inequality (a b c R r : ℝ) (hR : R > 0) (hr : r > 0) :
  R / (2 * r) ≥ ((64 * a^2 * b^2 * c^2) / 
  ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end circumradius_inradius_inequality_l176_176037


namespace max_ab_eq_one_quarter_l176_176910

theorem max_ab_eq_one_quarter (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_eq_one_quarter_l176_176910


namespace shaded_region_area_l176_176076

-- Given conditions
def diagonal_PQ : ℝ := 10
def number_of_squares : ℕ := 20

-- Definition of the side length of the squares
noncomputable def side_length := diagonal_PQ / (4 * Real.sqrt 2)

-- Area of one smaller square
noncomputable def one_square_area := side_length * side_length

-- Total area of the shaded region
noncomputable def total_area_of_shaded_region := number_of_squares * one_square_area

-- The theorem to be proven
theorem shaded_region_area : total_area_of_shaded_region = 62.5 := by
  sorry

end shaded_region_area_l176_176076


namespace workers_and_days_l176_176197

theorem workers_and_days (x y : ℕ) (h1 : x * y = (x - 20) * (y + 5)) (h2 : x * y = (x + 15) * (y - 2)) :
  x = 60 ∧ y = 10 := 
by {
  sorry
}

end workers_and_days_l176_176197


namespace quadratic_solution_is_unique_l176_176523

theorem quadratic_solution_is_unique (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 2 * p + q / 2 = -p)
  (h2 : 2 * p * (q / 2) = q) :
  (p, q) = (1, -6) :=
by
  sorry

end quadratic_solution_is_unique_l176_176523
