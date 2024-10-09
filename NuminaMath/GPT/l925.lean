import Mathlib

namespace eighth_graders_taller_rows_remain_ordered_l925_92532

-- Part (a)

theorem eighth_graders_taller {n : ℕ} (h8 : Fin n → ℚ) (h7 : Fin n → ℚ)
  (ordered8 : ∀ i j : Fin n, i ≤ j → h8 i ≤ h8 j)
  (ordered7 : ∀ i j : Fin n, i ≤ j → h7 i ≤ h7 j)
  (initial_condition : ∀ i : Fin n, h8 i > h7 i) :
  ∀ i : Fin n, h8 i > h7 i :=
sorry

-- Part (b)

theorem rows_remain_ordered {m n : ℕ} (h : Fin m → Fin n → ℚ)
  (row_ordered : ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k)
  (column_ordered_after : ∀ j : Fin n, ∀ i k : Fin m, i ≤ k → h i j ≤ h k j) :
  ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k :=
sorry

end eighth_graders_taller_rows_remain_ordered_l925_92532


namespace no_arithmetic_sqrt_of_neg_real_l925_92564

theorem no_arithmetic_sqrt_of_neg_real (x : ℝ) (h : x < 0) : ¬ ∃ y : ℝ, y * y = x :=
by
  sorry

end no_arithmetic_sqrt_of_neg_real_l925_92564


namespace no_roots_in_interval_l925_92504

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end no_roots_in_interval_l925_92504


namespace problem1_problem2_problem3_l925_92577

variable (a b : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_cond1 : a ≥ (1 / a) + (2 / b))
variable (h_cond2 : b ≥ (3 / a) + (2 / b))

/-- Statement 1: Prove that a + b ≥ 4 under the given conditions. -/
theorem problem1 : (a + b) ≥ 4 := 
by 
  sorry

/-- Statement 2: Prove that a^2 + b^2 ≥ 3 + 2√6 under the given conditions. -/
theorem problem2 : (a^2 + b^2) ≥ (3 + 2 * Real.sqrt 6) := 
by 
  sorry

/-- Statement 3: Prove that (1/a) + (1/b) < 1 + (√2/2) under the given conditions. -/
theorem problem3 : (1 / a) + (1 / b) < 1 + (Real.sqrt 2 / 2) := 
by 
  sorry

end problem1_problem2_problem3_l925_92577


namespace smallest_a_l925_92548

theorem smallest_a 
  (a : ℤ) (P : ℤ → ℤ) 
  (h_pos : 0 < a) 
  (hP1 : P 1 = a) (hP5 : P 5 = a) (hP7 : P 7 = a) (hP9 : P 9 = a) 
  (hP2 : P 2 = -a) (hP4 : P 4 = -a) (hP6 : P 6 = -a) (hP8 : P 8 = -a) : 
  a ≥ 336 :=
by
  sorry

end smallest_a_l925_92548


namespace net_change_is_minus_0_19_l925_92523

-- Define the yearly change factors as provided in the conditions
def yearly_changes : List ℚ := [6/5, 11/10, 7/10, 4/5, 11/10]

-- Compute the net change over the five years
def net_change (changes : List ℚ) : ℚ :=
  changes.foldl (λ acc x => acc * x) 1 - 1

-- Define the target value for the net change
def target_net_change : ℚ := -19 / 100

-- The theorem to prove the net change calculated matches the target net change
theorem net_change_is_minus_0_19 : net_change yearly_changes = target_net_change :=
  by
    sorry

end net_change_is_minus_0_19_l925_92523


namespace income_is_10000_l925_92558

theorem income_is_10000 (x : ℝ) (h : 10 * x = 8 * x + 2000) : 10 * x = 10000 := by
  have h1 : 2 * x = 2000 := by
    linarith
  have h2 : x = 1000 := by
    linarith
  linarith

end income_is_10000_l925_92558


namespace determine_good_numbers_l925_92559

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), (∀ k : Fin n, ∃ m : ℕ, k.1 + (a k).1 + 1 = m * m)

theorem determine_good_numbers :
  is_good_number 13 ∧ is_good_number 15 ∧ is_good_number 17 ∧ is_good_number 19 ∧ ¬is_good_number 11 :=
by
  sorry

end determine_good_numbers_l925_92559


namespace find_breadth_l925_92541

theorem find_breadth (p l : ℕ) (h_p : p = 600) (h_l : l = 100) (h_perimeter : p = 2 * (l + b)) : b = 200 :=
by
  sorry

end find_breadth_l925_92541


namespace ratio_of_volumes_l925_92547

variables (A B : ℚ)

theorem ratio_of_volumes 
  (h1 : (3/8) * A = (5/8) * B) :
  A / B = 5 / 3 :=
sorry

end ratio_of_volumes_l925_92547


namespace factorial_div_result_l925_92544

theorem factorial_div_result : Nat.factorial 13 / Nat.factorial 11 = 156 :=
sorry

end factorial_div_result_l925_92544


namespace find_velocity_l925_92562

variable (k V : ℝ)
variable (P A : ℕ)

theorem find_velocity (k_eq : k = 1 / 200) 
  (initial_cond : P = 4 ∧ A = 2 ∧ V = 20) 
  (new_cond : P = 16 ∧ A = 4) : 
  V = 20 * Real.sqrt 2 :=
by
  sorry

end find_velocity_l925_92562


namespace equilateral_triangle_sum_l925_92557

theorem equilateral_triangle_sum (x y : ℕ) (h1 : x + 5 = 14) (h2 : y + 11 = 14) : x + y = 12 :=
by
  sorry

end equilateral_triangle_sum_l925_92557


namespace part1_part2_l925_92537

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (|x - 1| ≤ 2) ∧ ((x + 3) / (x - 2) ≥ 0)

-- Part 1
theorem part1 (h_a : a = 1) (h_p : p a x) (h_q : q x) : 2 < x ∧ x < 3 := sorry

-- Part 2
theorem part2 (h_suff : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := sorry

end part1_part2_l925_92537


namespace average_after_19_innings_is_23_l925_92596

-- Definitions for the conditions given in the problem
variables {A : ℝ} -- Let A be the average score before the 19th inning

-- Conditions: The cricketer scored 95 runs in the 19th inning and his average increased by 4 runs.
def total_runs_after_18_innings (A : ℝ) : ℝ := 18 * A
def total_runs_after_19th_inning (A : ℝ) : ℝ := total_runs_after_18_innings A + 95
def new_average_after_19_innings (A : ℝ) : ℝ := A + 4

-- The statement of the problem as a Lean theorem
theorem average_after_19_innings_is_23 :
  (18 * A + 95) / 19 = A + 4 → A = 19 → (A + 4) = 23 :=
by
  intros hA h_avg_increased
  sorry

end average_after_19_innings_is_23_l925_92596


namespace total_stops_is_seven_l925_92505

-- Definitions of conditions
def initial_stops : ℕ := 3
def additional_stops : ℕ := 4

-- Statement to be proved
theorem total_stops_is_seven : initial_stops + additional_stops = 7 :=
by {
  -- this is a placeholder for the proof
  sorry
}

end total_stops_is_seven_l925_92505


namespace alexis_total_sewing_time_l925_92580

-- Define the time to sew a skirt and a coat
def t_skirt : ℕ := 2
def t_coat : ℕ := 7

-- Define the numbers of skirts and coats
def n_skirts : ℕ := 6
def n_coats : ℕ := 4

-- Define the total time
def total_time : ℕ := t_skirt * n_skirts + t_coat * n_coats

-- State the theorem
theorem alexis_total_sewing_time : total_time = 40 :=
by
  -- the proof would go here; we're skipping the proof as per instructions
  sorry

end alexis_total_sewing_time_l925_92580


namespace exist_xyz_modular_l925_92512

theorem exist_xyz_modular {n a b c : ℕ} (hn : 0 < n) (ha : a ≤ 3 * n ^ 2 + 4 * n) (hb : b ≤ 3 * n ^ 2 + 4 * n) (hc : c ≤ 3 * n ^ 2 + 4 * n) :
  ∃ (x y z : ℤ), abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 :=
sorry

end exist_xyz_modular_l925_92512


namespace proposition_neg_p_and_q_false_l925_92578

theorem proposition_neg_p_and_q_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end proposition_neg_p_and_q_false_l925_92578


namespace articles_produced_l925_92514

theorem articles_produced (x y : ℕ) :
  (x * x * x * (1 / (x^2 : ℝ))) = x → (y * y * y * (1 / (x^2 : ℝ))) = (y^3 / x^2 : ℝ) :=
by
  sorry

end articles_produced_l925_92514


namespace find_n_l925_92594

theorem find_n : (∃ n : ℕ, 2^3 * 8^3 = 2^(2 * n)) ↔ n = 6 :=
by
  sorry

end find_n_l925_92594


namespace books_added_after_lunch_l925_92500

-- Definitions for the given conditions
def initial_books : Int := 100
def books_borrowed_lunch : Int := 50
def books_remaining_lunch : Int := initial_books - books_borrowed_lunch
def books_borrowed_evening : Int := 30
def books_remaining_evening : Int := 60

-- Let X be the number of books added after lunchtime
variable (X : Int)

-- The proof goal in Lean statement
theorem books_added_after_lunch (h : books_remaining_lunch + X - books_borrowed_evening = books_remaining_evening) :
  X = 40 := by
  sorry

end books_added_after_lunch_l925_92500


namespace total_pieces_of_chicken_needed_l925_92593

def friedChickenDinnerPieces := 8
def chickenPastaPieces := 2
def barbecueChickenPieces := 4
def grilledChickenSaladPieces := 1

def friedChickenDinners := 4
def chickenPastaOrders := 8
def barbecueChickenOrders := 5
def grilledChickenSaladOrders := 6

def totalChickenPiecesNeeded :=
  (friedChickenDinnerPieces * friedChickenDinners) +
  (chickenPastaPieces * chickenPastaOrders) +
  (barbecueChickenPieces * barbecueChickenOrders) +
  (grilledChickenSaladPieces * grilledChickenSaladOrders)

theorem total_pieces_of_chicken_needed : totalChickenPiecesNeeded = 74 := by
  sorry

end total_pieces_of_chicken_needed_l925_92593


namespace ratio_of_x_intercepts_l925_92520

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v : ℝ)
  (hu : u = -b / 5) (hv : v = -b / 3) : u / v = 3 / 5 := by
  sorry

end ratio_of_x_intercepts_l925_92520


namespace income_fraction_from_tips_l925_92573

variable (S T : ℝ)

theorem income_fraction_from_tips :
  (T = (9 / 4) * S) → (T / (S + T) = 9 / 13) :=
by
  sorry

end income_fraction_from_tips_l925_92573


namespace collinear_points_m_equals_4_l925_92530

theorem collinear_points_m_equals_4 (m : ℝ)
  (h1 : (3 - 12) / (1 - -2) = (-6 - 12) / (m - -2)) : m = 4 :=
by
  sorry

end collinear_points_m_equals_4_l925_92530


namespace average_of_xyz_l925_92510

variable (x y z : ℝ)

theorem average_of_xyz (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 := by
  sorry

end average_of_xyz_l925_92510


namespace manufacturer_price_l925_92571

theorem manufacturer_price :
  ∃ M : ℝ, 
    (∃ R : ℝ, 
      R = 1.15 * M ∧
      ∃ D : ℝ, 
        D = 0.85 * R ∧
        R - D = 57.5) ∧
    M = 333.33 := 
by
  sorry

end manufacturer_price_l925_92571


namespace distinct_digits_sum_l925_92552

theorem distinct_digits_sum (A B C D G : ℕ) (AB CD GGG : ℕ)
  (h1: AB = 10 * A + B)
  (h2: CD = 10 * C + D)
  (h3: GGG = 111 * G)
  (h4: AB * CD = GGG)
  (h5: A ≠ B)
  (h6: A ≠ C)
  (h7: A ≠ D)
  (h8: A ≠ G)
  (h9: B ≠ C)
  (h10: B ≠ D)
  (h11: B ≠ G)
  (h12: C ≠ D)
  (h13: C ≠ G)
  (h14: D ≠ G)
  (hA: A < 10)
  (hB: B < 10)
  (hC: C < 10)
  (hD: D < 10)
  (hG: G < 10)
  : A + B + C + D + G = 17 := sorry

end distinct_digits_sum_l925_92552


namespace ratio_problem_l925_92524

theorem ratio_problem (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 3 * d) : d = 15 / 7 := by
  sorry

end ratio_problem_l925_92524


namespace determine_digit_I_l925_92554

theorem determine_digit_I (F I V E T H R N : ℕ) (hF : F = 8) (hE_odd : E = 1 ∨ E = 3 ∨ E = 5 ∨ E = 7 ∨ E = 9)
  (h_diff : F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ T ∧ F ≠ H ∧ F ≠ R ∧ F ≠ N 
             ∧ I ≠ V ∧ I ≠ E ∧ I ≠ T ∧ I ≠ H ∧ I ≠ R ∧ I ≠ N 
             ∧ V ≠ E ∧ V ≠ T ∧ V ≠ H ∧ V ≠ R ∧ V ≠ N 
             ∧ E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ N 
             ∧ T ≠ H ∧ T ≠ R ∧ T ≠ N 
             ∧ H ≠ R ∧ H ≠ N 
             ∧ R ≠ N)
  (h_verify_sum : (10^3 * 8 + 10^2 * I + 10 * V + E) + (10^4 * T + 10^3 * H + 10^2 * R + 11 * E) = 10^3 * N + 10^2 * I + 10 * N + E) :
  I = 4 := 
sorry

end determine_digit_I_l925_92554


namespace smaller_molds_radius_l925_92507

theorem smaller_molds_radius (r : ℝ) : 
  (∀ V_large V_small : ℝ, 
     V_large = (2/3) * π * (2:ℝ)^3 ∧
     V_small = (2/3) * π * r^3 ∧
     8 * V_small = V_large) → r = 1 := by
  sorry

end smaller_molds_radius_l925_92507


namespace intersection_of_A_and_complement_of_B_l925_92572

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x : ℝ | 2^x * (x - 2) < 1 }
noncomputable def B : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }
noncomputable def B_complement : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_A_and_complement_of_B :
  A ∩ B_complement = { x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_complement_of_B_l925_92572


namespace system_of_equations_solution_l925_92506

theorem system_of_equations_solution (b : ℝ) :
  (∀ (a : ℝ), ∃ (x y : ℝ), (x - 1)^2 + y^2 = 1 ∧ a * x + y = a * b) ↔ 0 ≤ b ∧ b ≤ 2 :=
by
  sorry

end system_of_equations_solution_l925_92506


namespace log_expression_simplifies_to_one_l925_92549

theorem log_expression_simplifies_to_one :
  (Real.log 5)^2 + Real.log 50 * Real.log 2 = 1 :=
by 
  sorry

end log_expression_simplifies_to_one_l925_92549


namespace log_sqrt_7_of_343sqrt7_l925_92595

noncomputable def log_sqrt_7 (y : ℝ) : ℝ := 
  Real.log y / Real.log (Real.sqrt 7)

theorem log_sqrt_7_of_343sqrt7 : log_sqrt_7 (343 * Real.sqrt 7) = 4 :=
by
  sorry

end log_sqrt_7_of_343sqrt7_l925_92595


namespace first_class_product_rate_l925_92517

theorem first_class_product_rate
  (total_products : ℕ)
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (pass_rate_correct : pass_rate = 0.95)
  (first_class_rate_correct : first_class_rate_among_qualified = 0.2) :
  (first_class_rate_among_qualified * pass_rate : ℝ) = 0.19 :=
by
  rw [pass_rate_correct, first_class_rate_correct]
  norm_num


end first_class_product_rate_l925_92517


namespace simplify_expression_l925_92501

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x :=
by 
  -- We provide 'sorry' hack to skip the proof
  -- Replace this with the actual proof to ensure correctness.
  sorry

end simplify_expression_l925_92501


namespace smallest_number_is_20_l925_92538

theorem smallest_number_is_20 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c)
  (mean_condition : (a + b + c) / 3 = 30)
  (median_condition : b = 31)
  (largest_condition : b = c - 8) :
  a = 20 :=
sorry

end smallest_number_is_20_l925_92538


namespace evaluate_fraction_l925_92551

theorem evaluate_fraction : (3 / (1 - 3 / 4) = 12) := by
  have h : (1 - 3 / 4) = 1 / 4 := by
    sorry
  rw [h]
  sorry

end evaluate_fraction_l925_92551


namespace find_multiple_l925_92535

theorem find_multiple (a b m : ℤ) (h1 : a * b = m * (a + b) + 12) 
(h2 : b = 10) (h3 : b - a = 6) : m = 2 :=
by {
  sorry
}

end find_multiple_l925_92535


namespace simplification_l925_92568

theorem simplification (b : ℝ) : 3 * b * (3 * b^3 + 2 * b) - 2 * b^2 = 9 * b^4 + 4 * b^2 :=
by
  sorry

end simplification_l925_92568


namespace average_speed_l925_92587

theorem average_speed (v : ℝ) (h : 500 / v - 500 / (v + 10) = 2) : v = 45.25 :=
by
  sorry

end average_speed_l925_92587


namespace R2_area_l925_92576

-- Definitions for the conditions
def R1_side1 : ℝ := 4
def R1_area : ℝ := 16
def R2_diagonal : ℝ := 10
def similar_rectangles (R1 R2 : ℝ × ℝ) : Prop := (R1.fst / R1.snd = R2.fst / R2.snd)

-- Main theorem
theorem R2_area {a b : ℝ} 
  (R1_side1 : a = 4)
  (R1_area : a * a = 16) 
  (R2_diagonal : b = 10)
  (h : similar_rectangles (a, a) (b / (10 / (2 : ℝ)), b / (10 / (2 : ℝ)))) : 
  b * b / (2 : ℝ) = 50 :=
by
  sorry

end R2_area_l925_92576


namespace find_positive_integers_l925_92539

theorem find_positive_integers (a b : ℕ) (h1 : a > 1) (h2 : b ∣ (a - 1)) (h3 : (2 * a + 1) ∣ (5 * b - 3)) : a = 10 ∧ b = 9 :=
sorry

end find_positive_integers_l925_92539


namespace find_positive_value_of_X_l925_92589

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end find_positive_value_of_X_l925_92589


namespace winnie_retains_lollipops_l925_92534

theorem winnie_retains_lollipops :
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  lollipops_total % friends = 10 :=
by
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  show lollipops_total % friends = 10
  sorry

end winnie_retains_lollipops_l925_92534


namespace optimal_station_placement_l925_92583

def distance_between_buildings : ℕ := 50
def workers_in_building (n : ℕ) : ℕ := n

def total_walking_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

theorem optimal_station_placement : ∃ x : ℝ, x = 150 ∧ (∀ y : ℝ, total_walking_distance x ≤ total_walking_distance y) :=
  sorry

end optimal_station_placement_l925_92583


namespace inequality_ab_sum_eq_five_l925_92526

noncomputable def inequality_solution (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x < 1) → (x < a) → (x > b) ∨ (x > 4) → (x < a) → (x > b))

theorem inequality_ab_sum_eq_five (a b : ℝ) 
  (h : inequality_solution a b) : a + b = 5 :=
sorry

end inequality_ab_sum_eq_five_l925_92526


namespace max_range_f_plus_2g_l925_92586

noncomputable def max_val_of_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) : ℝ :=
  9

theorem max_range_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) :
  ∃ (a b : ℝ), (-3 ≤ a ∧ a ≤ 5) ∧ (-8 ≤ b ∧ b ≤ 4) ∧ b = 9 := 
sorry

end max_range_f_plus_2g_l925_92586


namespace percentage_loss_is_correct_l925_92592

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end percentage_loss_is_correct_l925_92592


namespace no_function_satisfies_inequality_l925_92540

theorem no_function_satisfies_inequality (f : ℝ → ℝ) :
  ¬ ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
sorry

end no_function_satisfies_inequality_l925_92540


namespace quadratic_real_roots_l925_92575

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k^2 * x^2 - (2 * k + 1) * x + 1 = 0 ∧ ∃ x2 : ℝ, k^2 * x2^2 - (2 * k + 1) * x2 + 1 = 0)
  ↔ (k ≥ -1/4 ∧ k ≠ 0) := 
by 
  sorry

end quadratic_real_roots_l925_92575


namespace calculate_gfg3_l925_92529

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem calculate_gfg3 : g (f (g 3)) = 192 := by
  sorry

end calculate_gfg3_l925_92529


namespace evaluate_expression_l925_92519

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) : 
  x^3 * y^4 * z = 1/128 := 
by
  sorry

end evaluate_expression_l925_92519


namespace degree_of_monomial_3ab_l925_92556

variable (a b : ℕ)

def monomialDegree (x y : ℕ) : ℕ :=
  x + y

theorem degree_of_monomial_3ab : monomialDegree 1 1 = 2 :=
by
  sorry

end degree_of_monomial_3ab_l925_92556


namespace alice_needs_7_fills_to_get_3_cups_l925_92550

theorem alice_needs_7_fills_to_get_3_cups (needs : ℚ) (cup_size : ℚ) (has : ℚ) :
  needs = 3 ∧ cup_size = 1 / 3 ∧ has = 2 / 3 →
  (needs - has) / cup_size = 7 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end alice_needs_7_fills_to_get_3_cups_l925_92550


namespace penultimate_digit_of_quotient_l925_92574

theorem penultimate_digit_of_quotient :
  (4^1994 + 7^1994) / 10 % 10 = 1 :=
by
  sorry

end penultimate_digit_of_quotient_l925_92574


namespace max_value_of_f_l925_92567

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l925_92567


namespace large_cartridge_pages_correct_l925_92516

-- Define the conditions
def small_cartridge_pages : ℕ := 600
def medium_cartridge_pages : ℕ := 2 * 3 * small_cartridge_pages / 6
def large_cartridge_pages : ℕ := 2 * 3 * medium_cartridge_pages / 6

-- The theorem to prove
theorem large_cartridge_pages_correct :
  large_cartridge_pages = 1350 :=
by
  sorry

end large_cartridge_pages_correct_l925_92516


namespace total_number_of_participants_l925_92598

theorem total_number_of_participants (boys_achieving_distance : ℤ) (frequency : ℝ) (h1 : boys_achieving_distance = 8) (h2 : frequency = 0.4) : 
  (boys_achieving_distance : ℝ) / frequency = 20 := 
by 
  sorry

end total_number_of_participants_l925_92598


namespace sum_of_distinct_selections_is_34_l925_92509

-- Define a 4x4 grid filled sequentially from 1 to 16
def grid : List (List ℕ) := [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
]

-- Define a type for selections from the grid ensuring distinct rows and columns.
structure Selection where
  row : ℕ
  col : ℕ
  h_row : row < 4
  h_col : col < 4

-- Define the sum of any selection of 4 numbers from distinct rows and columns in the grid.
def sum_of_selection (selections : List Selection) : ℕ :=
  if h : List.length selections = 4 then
    List.sum (List.map (λ sel => (grid.get! sel.row).get! sel.col) selections)
  else 0

-- The main theorem
theorem sum_of_distinct_selections_is_34 (selections : List Selection) 
  (h_distinct_rows : List.Nodup (List.map (λ sel => sel.row) selections))
  (h_distinct_cols : List.Nodup (List.map (λ sel => sel.col) selections)) :
  sum_of_selection selections = 34 :=
by
  -- Proof is omitted
  sorry

end sum_of_distinct_selections_is_34_l925_92509


namespace deck_width_l925_92533

theorem deck_width (w : ℝ) : 
  (10 + 2 * w) * (12 + 2 * w) = 360 → w = 4 := 
by 
  sorry

end deck_width_l925_92533


namespace train_pass_time_l925_92528

-- Definitions based on the conditions
def train_length : ℕ := 280  -- train length in meters
def train_speed_kmh : ℕ := 72  -- train speed in km/hr
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 5 / 18)  -- train speed in m/s

-- Theorem statement
theorem train_pass_time : (train_length / train_speed_ms) = 14 := by
  sorry

end train_pass_time_l925_92528


namespace number_of_white_balls_l925_92581

theorem number_of_white_balls (a : ℕ) (h1 : 3 + a ≠ 0) (h2 : (3 : ℚ) / (3 + a) = 3 / 7) : a = 4 :=
sorry

end number_of_white_balls_l925_92581


namespace transform_expression_l925_92582

variable {a : ℝ}

theorem transform_expression (h : a - 1 < 0) : 
  (a - 1) * Real.sqrt (-1 / (a - 1)) = -Real.sqrt (1 - a) :=
by
  sorry

end transform_expression_l925_92582


namespace time_to_cross_same_direction_l925_92590

-- Defining the conditions
def speed_train1 : ℝ := 60 -- kmph
def speed_train2 : ℝ := 40 -- kmph
def time_opposite_directions : ℝ := 10.000000000000002 -- seconds 
def relative_speed_opposite_directions : ℝ := speed_train1 + speed_train2 -- 100 kmph
def relative_speed_same_direction : ℝ := speed_train1 - speed_train2 -- 20 kmph

-- Defining the proof statement
theorem time_to_cross_same_direction : 
  (time_opposite_directions * (relative_speed_opposite_directions / relative_speed_same_direction)) = 50 :=
by
  sorry

end time_to_cross_same_direction_l925_92590


namespace range_of_omega_l925_92561

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (a b : ℝ), a ≠ b ∧ 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ f ω a + f ω b = 4) ↔ 5 ≤ ω ∧ ω < 9 :=
sorry

end range_of_omega_l925_92561


namespace painting_cost_l925_92502

theorem painting_cost (total_cost : ℕ) (num_paintings : ℕ) (price : ℕ)
  (h1 : total_cost = 104)
  (h2 : 10 < num_paintings)
  (h3 : num_paintings < 60)
  (h4 : total_cost = num_paintings * price)
  (h5 : price ∈ {d ∈ {d : ℕ | d > 0} | total_cost % d = 0}) :
  price = 2 ∨ price = 4 ∨ price = 8 :=
by
  sorry

end painting_cost_l925_92502


namespace geometric_sequence_common_ratio_is_2_l925_92531

variable {a : ℕ → ℝ} (h : ∀ n : ℕ, a n * a (n + 1) = 4 ^ n)

theorem geometric_sequence_common_ratio_is_2 : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_is_2_l925_92531


namespace new_average_amount_l925_92542

theorem new_average_amount (A : ℝ) (H : A = 14) (new_amount : ℝ) (H1 : new_amount = 56) : 
  ((7 * A + new_amount) / 8) = 19.25 :=
by
  rw [H, H1]
  norm_num

end new_average_amount_l925_92542


namespace circle_radius_twice_value_l925_92525

theorem circle_radius_twice_value (r_x r_y v : ℝ) (h1 : π * r_x^2 = π * r_y^2)
  (h2 : 2 * π * r_x = 12 * π) (h3 : r_y = 2 * v) : v = 3 := by
  sorry

end circle_radius_twice_value_l925_92525


namespace calculate_expression_l925_92563

variable (x : ℝ)

theorem calculate_expression : (1/2 * x^3)^2 = 1/4 * x^6 := 
by 
  sorry

end calculate_expression_l925_92563


namespace solve_quadratic_l925_92543

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 → x = 1 :=
by
  intros x h
  sorry

end solve_quadratic_l925_92543


namespace find_natural_numbers_l925_92521

theorem find_natural_numbers (n : ℕ) :
  (∀ k : ℕ, k^2 + ⌊ (n : ℝ) / (k^2 : ℝ) ⌋ ≥ 1991) ∧
  (∃ k_0 : ℕ, k_0^2 + ⌊ (n : ℝ) / (k_0^2 : ℝ) ⌋ < 1992) ↔
  990208 ≤ n ∧ n ≤ 991231 :=
by sorry

end find_natural_numbers_l925_92521


namespace number_of_poles_l925_92566

theorem number_of_poles (side_length : ℝ) (distance_between_poles : ℝ) 
  (h1 : side_length = 150) (h2 : distance_between_poles = 30) : 
  ((4 * side_length) / distance_between_poles) = 20 :=
by 
  -- Placeholder to indicate missing proof
  sorry

end number_of_poles_l925_92566


namespace percent_increase_in_sales_l925_92585

theorem percent_increase_in_sales (sales_this_year : ℕ) (sales_last_year : ℕ) (percent_increase : ℚ) :
  sales_this_year = 400 ∧ sales_last_year = 320 → percent_increase = 25 :=
by
  sorry

end percent_increase_in_sales_l925_92585


namespace alicia_total_deductions_in_cents_l925_92584

def Alicia_hourly_wage : ℝ := 25
def local_tax_rate : ℝ := 0.015
def retirement_contribution_rate : ℝ := 0.03

theorem alicia_total_deductions_in_cents :
  let wage_cents := Alicia_hourly_wage * 100
  let tax_deduction := wage_cents * local_tax_rate
  let after_tax_earnings := wage_cents - tax_deduction
  let retirement_contribution := after_tax_earnings * retirement_contribution_rate
  let total_deductions := tax_deduction + retirement_contribution
  total_deductions = 111 :=
by
  sorry

end alicia_total_deductions_in_cents_l925_92584


namespace gcd_36_48_72_l925_92527

theorem gcd_36_48_72 : Int.gcd (Int.gcd 36 48) 72 = 12 := by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 48 = 2^4 * 3 := by norm_num
  have h3 : 72 = 2^3 * 3^2 := by norm_num
  sorry

end gcd_36_48_72_l925_92527


namespace option_d_is_correct_l925_92508

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end option_d_is_correct_l925_92508


namespace parabola_vertex_coordinates_l925_92591

theorem parabola_vertex_coordinates :
  ∃ (h k : ℝ), (∀ (x : ℝ), (y = (x - h)^2 + k) = (y = (x-1)^2 + 2)) ∧ h = 1 ∧ k = 2 :=
by
  sorry

end parabola_vertex_coordinates_l925_92591


namespace custom_operation_example_l925_92588

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end custom_operation_example_l925_92588


namespace parallel_lines_implies_slope_l925_92560

theorem parallel_lines_implies_slope (a : ℝ) :
  (∀ (x y: ℝ), ax + 2 * y = 0) ∧ (∀ (x y: ℝ), x + y = 1) → (a = 2) :=
by
  sorry

end parallel_lines_implies_slope_l925_92560


namespace eight_packets_weight_l925_92555

variable (weight_per_can : ℝ)
variable (weight_per_packet : ℝ)

-- Conditions
axiom h1 : weight_per_can = 1
axiom h2 : 3 * weight_per_can = 8 * weight_per_packet
axiom h3 : weight_per_packet = 6 * weight_per_can

-- Question to be proved: 8 packets weigh 12 kg
theorem eight_packets_weight : 8 * weight_per_packet = 12 :=
by 
  -- Proof would go here
  sorry

end eight_packets_weight_l925_92555


namespace proof_problem_l925_92545

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, (y = 2^x - 1) ∧ (x ≤ 2)}

-- Define the complement of set A in U
def complement_A : Set ℝ := Set.compl A

-- Define the intersection of complement_A and B
def complement_A_inter_B : Set ℝ := complement_A ∩ B

-- State the theorem
theorem proof_problem : complement_A_inter_B = {x | (-1 < x) ∧ (x ≤ 2)} :=
by
  sorry

end proof_problem_l925_92545


namespace travelers_on_liner_l925_92515

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l925_92515


namespace vents_per_zone_l925_92579

theorem vents_per_zone (total_cost : ℝ) (number_of_zones : ℝ) (cost_per_vent : ℝ) (h_total_cost : total_cost = 20000) (h_zones : number_of_zones = 2) (h_cost_per_vent : cost_per_vent = 2000) : 
  (total_cost / cost_per_vent) / number_of_zones = 5 :=
by 
  sorry

end vents_per_zone_l925_92579


namespace maximum_possible_savings_is_63_l925_92569

-- Definitions of the conditions
def doughnut_price := 8
def doughnut_discount_2 := 14
def doughnut_discount_4 := 26

def croissant_price := 10
def croissant_discount_3 := 28
def croissant_discount_5 := 45

def muffin_price := 6
def muffin_discount_2 := 11
def muffin_discount_6 := 30

-- Quantities to purchase
def doughnut_qty := 20
def croissant_qty := 15
def muffin_qty := 18

-- Prices calculated from quantities
def total_price_without_discount :=
  doughnut_qty * doughnut_price + croissant_qty * croissant_price + muffin_qty * muffin_price

def total_price_with_discount :=
  5 * doughnut_discount_4 + 3 * croissant_discount_5 + 3 * muffin_discount_6

def maximum_savings := total_price_without_discount - total_price_with_discount

theorem maximum_possible_savings_is_63 : maximum_savings = 63 := by
  -- Proof to be filled in
  sorry

end maximum_possible_savings_is_63_l925_92569


namespace no_positive_integers_satisfy_equation_l925_92553

theorem no_positive_integers_satisfy_equation :
  ¬ ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a^2 = b^11 + 23 :=
by
  sorry

end no_positive_integers_satisfy_equation_l925_92553


namespace cindy_correct_answer_l925_92503

/-- 
Cindy accidentally first subtracted 9 from a number, then multiplied the result 
by 2 before dividing by 6, resulting in an answer of 36. 
Following these steps, she was actually supposed to subtract 12 from the 
number and then divide by 8. What would her answer have been had she worked the 
problem correctly?
-/
theorem cindy_correct_answer :
  ∀ (x : ℝ), (2 * (x - 9) / 6 = 36) → ((x - 12) / 8 = 13.125) :=
by
  intro x
  sorry

end cindy_correct_answer_l925_92503


namespace max_product_of_sum_300_l925_92565

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l925_92565


namespace plane_equation_through_points_perpendicular_l925_92511

theorem plane_equation_through_points_perpendicular {M N : ℝ × ℝ × ℝ} (hM : M = (2, -1, 4)) (hN : N = (3, 2, -1)) :
  ∃ A B C d : ℝ, (∀ x y z : ℝ, A * x + B * y + C * z + d = 0 ↔ (x, y, z) = M ∨ (x, y, z) = N ∧ A + B + C = 0) ∧
  (4, -3, -1, -7) = (A, B, C, d) := 
sorry

end plane_equation_through_points_perpendicular_l925_92511


namespace total_spent_l925_92513

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℕ := 4

theorem total_spent : (original_price * (1 - discount_rate) * number_of_friends) = 40 := by
  sorry

end total_spent_l925_92513


namespace user_level_1000_l925_92518

noncomputable def user_level (points : ℕ) : ℕ :=
if points >= 1210 then 18
else if points >= 1000 then 17
else if points >= 810 then 16
else if points >= 640 then 15
else if points >= 490 then 14
else if points >= 360 then 13
else if points >= 250 then 12
else if points >= 160 then 11
else if points >= 90 then 10
else 0

theorem user_level_1000 : user_level 1000 = 17 :=
by {
  -- proof will be written here
  sorry
}

end user_level_1000_l925_92518


namespace chloe_pawn_loss_l925_92522

theorem chloe_pawn_loss (sophia_lost : ℕ) (total_left : ℕ) (total_initial : ℕ) (each_start : ℕ) (sophia_initial : ℕ) :
  sophia_lost = 5 → total_left = 10 → each_start = 8 → total_initial = 16 → sophia_initial = 8 →
  ∃ (chloe_lost : ℕ), chloe_lost = 1 :=
by
  sorry

end chloe_pawn_loss_l925_92522


namespace total_salaries_l925_92536

theorem total_salaries (A_salary B_salary : ℝ)
  (hA : A_salary = 1500)
  (hsavings : 0.05 * A_salary = 0.15 * B_salary) :
  A_salary + B_salary = 2000 :=
by {
  sorry
}

end total_salaries_l925_92536


namespace solve_for_x_l925_92546

theorem solve_for_x (x : ℝ) (h : 2 * x + 10 = (1 / 2) * (5 * x + 30)) : x = -10 :=
sorry

end solve_for_x_l925_92546


namespace find_a_l925_92599

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l925_92599


namespace complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l925_92597

def U : Set ℝ := {x | x ≥ -2}
def A : Set ℝ := {x | 2 < x ∧ x < 10}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}

theorem complement_A :
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x ≥ 10} :=
by sorry

theorem complement_A_intersection_B :
  (U \ A) ∩ B = {2} :=
by sorry

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 8} :=
by sorry

theorem complement_intersection_A_B :
  U \ (A ∩ B) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x > 8} :=
by sorry

end complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l925_92597


namespace muffin_is_twice_as_expensive_as_banana_l925_92570

variable (m b : ℚ)
variable (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12)
variable (h2 : 3 * m + 5 * b = S)

theorem muffin_is_twice_as_expensive_as_banana (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12) : m = 2 * b :=
by
  sorry

end muffin_is_twice_as_expensive_as_banana_l925_92570
