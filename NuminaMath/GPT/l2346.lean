import Mathlib

namespace fraction_product_equals_12_l2346_234624

theorem fraction_product_equals_12 :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := 
by
  sorry

end fraction_product_equals_12_l2346_234624


namespace number_of_sheep_l2346_234672

variable (S H C : ℕ)

def ratio_constraint : Prop := 4 * H = 7 * S ∧ 5 * S = 4 * C

def horse_food_per_day (H : ℕ) : ℕ := 230 * H
def sheep_food_per_day (S : ℕ) : ℕ := 150 * S
def cow_food_per_day (C : ℕ) : ℕ := 300 * C

def total_horse_food : Prop := horse_food_per_day H = 12880
def total_sheep_food : Prop := sheep_food_per_day S = 9750
def total_cow_food : Prop := cow_food_per_day C = 15000

theorem number_of_sheep (h1 : ratio_constraint S H C)
                        (h2 : total_horse_food H)
                        (h3 : total_sheep_food S)
                        (h4 : total_cow_food C) :
  S = 98 :=
sorry

end number_of_sheep_l2346_234672


namespace smallest_n_solution_unique_l2346_234633

theorem smallest_n_solution_unique (a b c d : ℤ) (h : a^2 + b^2 + c^2 = 4 * d^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end smallest_n_solution_unique_l2346_234633


namespace ratio_final_to_initial_l2346_234635

theorem ratio_final_to_initial (P R T : ℝ) (hR : R = 5) (hT : T = 20) :
  let SI := P * R * T / 100
  let A := P + SI
  A / P = 2 := 
by
  sorry

end ratio_final_to_initial_l2346_234635


namespace noon_temperature_l2346_234632

variable (a : ℝ)

theorem noon_temperature (h1 : ∀ (x : ℝ), x = a) (h2 : ∀ (y : ℝ), y = a + 10) :
  a + 10 = y :=
by
  sorry

end noon_temperature_l2346_234632


namespace not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l2346_234643

def right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_D (a b c : ℝ):
  ¬ (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 ∧ right_angle_triangle a b c) :=
sorry

theorem right_triangle_A (a b c x : ℝ):
  a = 5 * x → b = 12 * x → c = 13 * x → x > 0 → right_angle_triangle a b c :=
sorry

theorem right_triangle_B (angleA angleB angleC : ℝ):
  angleA / angleB / angleC = 2 / 3 / 5 → angleC = 90 → angleA + angleB + angleC = 180 → right_angle_triangle angleA angleB angleC :=
sorry

theorem right_triangle_C (a b c k : ℝ):
  a = 9 * k → b = 40 * k → c = 41 * k → k > 0 → right_angle_triangle a b c :=
sorry

end not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l2346_234643


namespace average_infections_per_round_infections_after_three_rounds_l2346_234621

-- Define the average number of infections per round such that the total after two rounds is 36 and x > 0
theorem average_infections_per_round :
  ∃ x : ℤ, (1 + x)^2 = 36 ∧ x > 0 :=
by
  sorry

-- Given x = 5, prove that the total number of infections after three rounds exceeds 200
theorem infections_after_three_rounds (x : ℤ) (H : x = 5) :
  (1 + x)^3 > 200 :=
by
  sorry

end average_infections_per_round_infections_after_three_rounds_l2346_234621


namespace union_of_A_and_B_l2346_234670

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_A_and_B_l2346_234670


namespace total_students_stratified_sampling_l2346_234604

namespace HighSchool

theorem total_students_stratified_sampling 
  (sample_size : ℕ)
  (sample_grade10 : ℕ)
  (sample_grade11 : ℕ)
  (students_grade12 : ℕ) 
  (n : ℕ)
  (H1 : sample_size = 100)
  (H2 : sample_grade10 = 24)
  (H3 : sample_grade11 = 26)
  (H4 : students_grade12 = 600)
  (H5 : ∀ n, (students_grade12 / n * sample_size = sample_size - sample_grade10 - sample_grade11) → n = 1200) :
  n = 1200 :=
sorry

end HighSchool

end total_students_stratified_sampling_l2346_234604


namespace eccentricity_of_ellipse_l2346_234612

open Real

theorem eccentricity_of_ellipse (a b c : ℝ) 
  (h1 : a > b ∧ b > 0)
  (h2 : c^2 = a^2 - b^2)
  (x : ℝ)
  (h3 : 3 * x = 2 * a)
  (h4 : sqrt 3 * x = 2 * c) :
  c / a = sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l2346_234612


namespace smallest_relatively_prime_210_l2346_234629

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end smallest_relatively_prime_210_l2346_234629


namespace expression_as_fraction_l2346_234634

theorem expression_as_fraction :
  1 + (4 / (5 + (6 / 7))) = (69 : ℚ) / 41 := 
by
  sorry

end expression_as_fraction_l2346_234634


namespace sum_of_3_consecutive_multiples_of_3_l2346_234610

theorem sum_of_3_consecutive_multiples_of_3 (a b c : ℕ) (h₁ : a = b + 3) (h₂ : b = c + 3) (h₃ : a = 42) : a + b + c = 117 :=
by sorry

end sum_of_3_consecutive_multiples_of_3_l2346_234610


namespace sufficient_but_not_necessary_condition_l2346_234603

def sufficient_condition (a : ℝ) : Prop := 
  (a > 1) → (1 / a < 1)

def necessary_condition (a : ℝ) : Prop := 
  (1 / a < 1) → (a > 1)

theorem sufficient_but_not_necessary_condition (a : ℝ) : sufficient_condition a ∧ ¬necessary_condition a := by
  sorry

end sufficient_but_not_necessary_condition_l2346_234603


namespace point_in_first_quadrant_l2346_234645

theorem point_in_first_quadrant (x y : ℝ) (h₁ : x = 3) (h₂ : y = 2) (hx : x > 0) (hy : y > 0) :
  ∃ q : ℕ, q = 1 := 
by
  sorry

end point_in_first_quadrant_l2346_234645


namespace sum_of_eight_numbers_l2346_234664

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l2346_234664


namespace james_speed_downhill_l2346_234644

theorem james_speed_downhill (T1 T2 v : ℝ) (h1 : T1 = 20 / v) (h2 : T2 = 12 / 3 + 1) (h3 : T1 = T2 - 1) : v = 5 :=
by
  -- Declare variables
  have hT2 : T2 = 5 := by linarith
  have hT1 : T1 = 4 := by linarith
  have hv : v = 20 / 4 := by sorry
  linarith

#exit

end james_speed_downhill_l2346_234644


namespace total_nails_needed_l2346_234696

-- Definitions based on problem conditions
def nails_per_plank : ℕ := 2
def planks_needed : ℕ := 2

-- Theorem statement: Prove that the total number of nails John needs is 4.
theorem total_nails_needed : nails_per_plank * planks_needed = 4 := by
  sorry

end total_nails_needed_l2346_234696


namespace original_number_is_144_l2346_234669

theorem original_number_is_144 (x : ℕ) (h : x - x / 3 = x - 48) : x = 144 :=
by
  sorry

end original_number_is_144_l2346_234669


namespace jess_father_first_round_l2346_234668

theorem jess_father_first_round (initial_blocks : ℕ)
  (players : ℕ)
  (blocks_before_jess_turn : ℕ)
  (jess_falls_tower_round : ℕ)
  (h1 : initial_blocks = 54)
  (h2 : players = 5)
  (h3 : blocks_before_jess_turn = 28)
  (h4 : ∀ rounds : ℕ, rounds * players ≥ 26 → jess_falls_tower_round = rounds + 1) :
  jess_falls_tower_round = 6 := 
by
  sorry

end jess_father_first_round_l2346_234668


namespace proof_problem_l2346_234638

variable {a : ℕ → ℝ} -- sequence a
variable {S : ℕ → ℝ} -- partial sums sequence S 
variable {n : ℕ} -- index

-- Define the conditions
def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n+1) = a n + d

def S_is_partial_sum (a S : ℕ → ℝ) : Prop := 
  ∀ n, S (n+1) = S n + a (n+1)

-- The properties given in the problem
def conditions (a S : ℕ → ℝ) : Prop :=
  is_arith_seq a ∧ 
  S_is_partial_sum a S ∧ 
  S 6 < S 7 ∧ 
  S 7 > S 8

-- The conclusions that need to be proved
theorem proof_problem (a S : ℕ → ℝ) (h : conditions a S) : 
  S 9 < S 6 ∧
  (∀ n, a 1 ≥ a (n+1)) ∧
  (∀ m, S 7 ≥ S m) := by 
  sorry

end proof_problem_l2346_234638


namespace value_of_m_l2346_234642
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 3)

theorem value_of_m (m : ℝ) (x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2 : ℝ, x1 > x2 → y m x1 < y m x2) :
  m = 2 :=
sorry

end value_of_m_l2346_234642


namespace inequality_solution_l2346_234606

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 2) :
  ∀ y : ℝ, y > 0 → 4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y :=
by
  intro y hy
  sorry

end inequality_solution_l2346_234606


namespace convert_base_7_to_base_10_l2346_234683

theorem convert_base_7_to_base_10 : 
  ∀ n : ℕ, (n = 3 * 7^2 + 2 * 7^1 + 1 * 7^0) → n = 162 :=
by
  intros n h
  rw [pow_zero, pow_one, pow_two] at h
  norm_num at h
  exact h

end convert_base_7_to_base_10_l2346_234683


namespace prove_m_value_l2346_234695

theorem prove_m_value (m : ℕ) : 8^4 = 4^m → m = 6 := by
  sorry

end prove_m_value_l2346_234695


namespace find_b_l2346_234631

-- Definitions from conditions
def f (x : ℚ) := 3 * x - 2
def g (x : ℚ) := 7 - 2 * x

-- Problem statement
theorem find_b (b : ℚ) (h : g (f b) = 1) : b = 5 / 3 := sorry

end find_b_l2346_234631


namespace range_of_a_l2346_234630

theorem range_of_a {A B : Set ℝ} (hA : A = {x | x > 5}) (hB : B = {x | x > a}) 
  (h_sufficient_not_necessary : A ⊆ B ∧ ¬(B ⊆ A)) 
  : a < 5 :=
sorry

end range_of_a_l2346_234630


namespace vector_coordinates_l2346_234641

theorem vector_coordinates (b : ℝ × ℝ)
  (a : ℝ × ℝ := (Real.sqrt 3, 1))
  (angle : ℝ := 2 * Real.pi / 3)
  (norm_b : ℝ := 1)
  (dot_product_eq : (a.fst * b.fst + a.snd * b.snd = -1))
  (norm_b_eq : (b.fst ^ 2 + b.snd ^ 2 = 1)) :
  b = (0, -1) ∨ b = (-Real.sqrt 3 / 2, 1 / 2) :=
sorry

end vector_coordinates_l2346_234641


namespace no_positive_integer_solutions_l2346_234692

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end no_positive_integer_solutions_l2346_234692


namespace fraction_simplification_l2346_234654

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2 * d * e) / (d^2 + f^2 - e^2 + 3 * d * f) = (d + e - f) / (d + f - e) :=
sorry

end fraction_simplification_l2346_234654


namespace lesser_fraction_l2346_234607

theorem lesser_fraction (x y : ℚ) (hx : x + y = 13 / 14) (hy : x * y = 1 / 8) : 
  x = (13 - Real.sqrt 57) / 28 ∨ y = (13 - Real.sqrt 57) / 28 :=
by
  sorry

end lesser_fraction_l2346_234607


namespace digit_distribution_l2346_234666

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l2346_234666


namespace train_speed_l2346_234648

noncomputable def speed_of_each_train (v : ℕ) : ℕ := 27

theorem train_speed
  (length_of_each_train : ℕ)
  (crossing_time : ℕ)
  (crossing_condition : 2 * (length_of_each_train * crossing_time) / (2 * crossing_time) = 15 / 2)
  (conversion_factor : ∀ n, 1 = 3.6 * n → ℕ) :
  speed_of_each_train 27 = 27 :=
by
  exact rfl

end train_speed_l2346_234648


namespace students_at_school_yy_l2346_234616

theorem students_at_school_yy (X Y : ℝ) 
    (h1 : X + Y = 4000)
    (h2 : 0.07 * X - 0.03 * Y = 40) : 
    Y = 2400 :=
by
  sorry

end students_at_school_yy_l2346_234616


namespace missing_digit_B_l2346_234690

theorem missing_digit_B :
  ∃ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (200 + 10 * B + 5) % 13 = 0 := 
sorry

end missing_digit_B_l2346_234690


namespace smallest_possible_positive_value_l2346_234652

theorem smallest_possible_positive_value (l w : ℕ) (hl : l > 0) (hw : w > 0) : ∃ x : ℕ, x = w - l + 1 ∧ x = 1 := 
by {
  sorry
}

end smallest_possible_positive_value_l2346_234652


namespace simplify_condition_l2346_234646

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  Real.sqrt (1 + x) - Real.sqrt (-1 - x)

theorem simplify_condition (x : ℝ) (h1 : 1 + x ≥ 0) (h2 : -1 - x ≥ 0) : simplify_expression x = 0 :=
by
  rw [simplify_expression]
  sorry

end simplify_condition_l2346_234646


namespace min_keychains_to_reach_profit_l2346_234685

theorem min_keychains_to_reach_profit :
  let cost_per_keychain := 0.15
  let sell_price_per_keychain := 0.45
  let total_keychains := 1200
  let target_profit := 180
  let total_cost := total_keychains * cost_per_keychain
  let total_revenue := total_cost + target_profit
  let min_keychains_to_sell := total_revenue / sell_price_per_keychain
  min_keychains_to_sell = 800 := 
by
  sorry

end min_keychains_to_reach_profit_l2346_234685


namespace percent_not_filler_l2346_234636

theorem percent_not_filler (sandwich_weight filler_weight : ℕ) (h_sandwich : sandwich_weight = 180) (h_filler : filler_weight = 45) : 
  (sandwich_weight - filler_weight) * 100 / sandwich_weight = 75 :=
by
  -- proof here
  sorry

end percent_not_filler_l2346_234636


namespace total_cakes_served_l2346_234661

def L : Nat := 5
def D : Nat := 6
def Y : Nat := 3
def T : Nat := L + D + Y

theorem total_cakes_served : T = 14 := by
  sorry

end total_cakes_served_l2346_234661


namespace arrangement_problem_l2346_234649

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangement_problem 
  (p1 p2 p3 p4 p5 : Type)  -- Representing the five people
  (youngest : p1)         -- Specifying the youngest
  (oldest : p5)           -- Specifying the oldest
  (unique_people : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5) -- Ensuring five unique people
  : (factorial 5) - (factorial 4 * 2) = 72 :=
by sorry

end arrangement_problem_l2346_234649


namespace total_insects_eaten_l2346_234657

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l2346_234657


namespace middle_managers_to_be_selected_l2346_234659

def total_employees : ℕ := 160
def senior_managers : ℕ := 10
def middle_managers : ℕ := 30
def staff_members : ℕ := 120
def total_to_be_selected : ℕ := 32

theorem middle_managers_to_be_selected : 
  (middle_managers * total_to_be_selected / total_employees) = 6 := by
  sorry

end middle_managers_to_be_selected_l2346_234659


namespace farmer_pigs_chickens_l2346_234688

-- Defining the problem in Lean 4

theorem farmer_pigs_chickens (p ch : ℕ) (h₁ : 30 * p + 24 * ch = 1200) (h₂ : p > 0) (h₃ : ch > 0) : 
  (p = 4) ∧ (ch = 45) :=
by sorry

end farmer_pigs_chickens_l2346_234688


namespace complement_of_A_l2346_234627

variables (U : Set ℝ) (A : Set ℝ)
def universal_set : Prop := U = Set.univ
def range_of_function : Prop := A = {x : ℝ | 0 ≤ x}

theorem complement_of_A (hU : universal_set U) (hA : range_of_function A) : 
  U \ A = {x : ℝ | x < 0} :=
by 
  sorry

end complement_of_A_l2346_234627


namespace vector_magnitude_l2346_234640

noncomputable def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 + v2.1, v1.2 + v2.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude :
  ∀ (x y : ℝ), let a := (x, 2)
               let b := (1, y)
               let c := (2, -6)
               (a.1 * c.1 + a.2 * c.2 = 0) →
               (b.1 * (-c.2) - b.2 * c.1 = 0) →
               magnitude (vec_add a b) = 5 * Real.sqrt 2 :=
by
  intros x y a b c h₁ h₂
  let a := (x, 2)
  let b := (1, y)
  let c := (2, -6)
  sorry

end vector_magnitude_l2346_234640


namespace ellipse_equation_correct_l2346_234660

theorem ellipse_equation_correct :
  ∃ (a b h k : ℝ), 
    h = 4 ∧ 
    k = 0 ∧ 
    a = 10 + 2 * Real.sqrt 10 ∧ 
    b = Real.sqrt (101 + 20 * Real.sqrt 10) ∧ 
    (∀ x y : ℝ, (x, y) = (9, 6) → 
    ((x - h)^2 / a^2 + y^2 / b^2 = 1)) ∧
    (dist (4 - 3, 0) (4 + 3, 0) = 6) := 
sorry

end ellipse_equation_correct_l2346_234660


namespace distinct_domino_paths_l2346_234681

/-- Matt will arrange five identical, dotless dominoes (1 by 2 rectangles) 
on a 6 by 4 grid so that a path is formed from the upper left-hand corner 
(0, 0) to the lower right-hand corner (4, 5). Prove that the number of 
distinct arrangements is 126. -/
theorem distinct_domino_paths : 
  let m := 4
  let n := 5
  let total_moves := m + n
  let right_moves := m
  let down_moves := n
  (total_moves.choose right_moves) = 126 := by
{ 
  sorry 
}

end distinct_domino_paths_l2346_234681


namespace page_sum_incorrect_l2346_234674

theorem page_sum_incorrect (sheets : List (Nat × Nat)) (h_sheets_len : sheets.length = 25)
  (h_consecutive : ∀ (a b : Nat), (a, b) ∈ sheets → (b = a + 1 ∨ a = b + 1))
  (h_sum_eq_2020 : (sheets.map (λ p => p.1 + p.2)).sum = 2020) : False :=
by
  sorry

end page_sum_incorrect_l2346_234674


namespace product_of_terms_eq_72_l2346_234601

theorem product_of_terms_eq_72
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 12) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 72 :=
by
  sorry

end product_of_terms_eq_72_l2346_234601


namespace marked_cells_in_grid_l2346_234611

theorem marked_cells_in_grid :
  ∀ (grid : Matrix (Fin 5) (Fin 5) Bool), 
  (∀ (i j : Fin 3), ∃! (a b : Fin 3), grid (i + a + 1) (j + b + 1) = true) → ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 4 :=
by
  sorry

end marked_cells_in_grid_l2346_234611


namespace problem_inverse_range_m_l2346_234605

theorem problem_inverse_range_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2 / x + 1 / y = 1) : 
  (2 * x + y > m^2 + 8 * m) ↔ (m > -9 ∧ m < 1) := 
by
  sorry

end problem_inverse_range_m_l2346_234605


namespace problem_xyz_l2346_234665

noncomputable def distance_from_intersection_to_side_CD (s : ℝ) : ℝ :=
  s * ((8 - Real.sqrt 15) / 8)

theorem problem_xyz
  (s : ℝ)
  (ABCD_is_square : (0 ≤ s))
  (X_is_intersection: ∃ (X : ℝ × ℝ), (X.1^2 + X.2^2 = s^2) ∧ ((X.1 - s)^2 + X.2^2 = (s / 2)^2))
  : distance_from_intersection_to_side_CD s = (s * (8 - Real.sqrt 15) / 8) :=
sorry

end problem_xyz_l2346_234665


namespace novels_in_shipment_l2346_234622

theorem novels_in_shipment (N : ℕ) (H1: 225 = (3/4:ℚ) * N) : N = 300 := 
by
  sorry

end novels_in_shipment_l2346_234622


namespace ratio_of_average_speed_to_still_water_speed_l2346_234689

noncomputable def speed_of_current := 6
noncomputable def speed_in_still_water := 18
noncomputable def downstream_speed := speed_in_still_water + speed_of_current
noncomputable def upstream_speed := speed_in_still_water - speed_of_current
noncomputable def distance_each_way := 1
noncomputable def total_distance := 2 * distance_each_way
noncomputable def time_downstream := (distance_each_way : ℝ) / (downstream_speed : ℝ)
noncomputable def time_upstream := (distance_each_way : ℝ) / (upstream_speed : ℝ)
noncomputable def total_time := time_downstream + time_upstream
noncomputable def average_speed := (total_distance : ℝ) / (total_time : ℝ)
noncomputable def ratio_average_speed := (average_speed : ℝ) / (speed_in_still_water : ℝ)

theorem ratio_of_average_speed_to_still_water_speed :
  ratio_average_speed = (8 : ℝ) / (9 : ℝ) :=
sorry

end ratio_of_average_speed_to_still_water_speed_l2346_234689


namespace number_of_children_l2346_234658

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l2346_234658


namespace train_pass_time_correct_l2346_234671

noncomputable def train_time_to_pass_post (length_of_train : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  length_of_train / speed_mps

theorem train_pass_time_correct :
  train_time_to_pass_post 60 36 = 6 := by
  sorry

end train_pass_time_correct_l2346_234671


namespace find_x_l2346_234600

theorem find_x
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h : a = (Real.sqrt 3, 0))
  (h1 : b = (x, -2))
  (h2 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0) :
  x = Real.sqrt 3 / 2 :=
sorry

end find_x_l2346_234600


namespace problem_number_of_true_propositions_l2346_234653

open Set

variable {α : Type*} {A B : Set α}

def card (s : Set α) : ℕ := sorry -- The actual definition of cardinality is complex and in LF (not imperative here).

-- Statement of the problem translated into a Lean statement
theorem problem_number_of_true_propositions :
  (∀ {A B : Set ℕ}, A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B) ∧
  (∀ {A B : Set ℕ}, A ⊆ B → card A ≤ card B) ∧
  (∀ {A B : Set ℕ}, A ⊂ B → card A < card B) →
   (3 = 3) :=
by 
  sorry


end problem_number_of_true_propositions_l2346_234653


namespace larger_triangle_perimeter_is_65_l2346_234675

theorem larger_triangle_perimeter_is_65 (s1 s2 s3 t1 t2 t3 : ℝ)
  (h1 : s1 = 7) (h2 : s2 = 7) (h3 : s3 = 12)
  (h4 : t3 = 30)
  (similar : t1 / s1 = t2 / s2 ∧ t2 / s2 = t3 / s3) :
  t1 + t2 + t3 = 65 := by
  sorry

end larger_triangle_perimeter_is_65_l2346_234675


namespace prove_statement_II_must_be_true_l2346_234639

-- Definitions of the statements
def statement_I (d : ℕ) : Prop := d = 5
def statement_II (d : ℕ) : Prop := d ≠ 6
def statement_III (d : ℕ) : Prop := d = 7
def statement_IV (d : ℕ) : Prop := d ≠ 8

-- Condition: Exactly three of these statements are true and one is false
def exactly_three_true (P Q R S : Prop) : Prop :=
  (P ∧ Q ∧ R ∧ ¬S) ∨ (P ∧ Q ∧ ¬R ∧ S) ∨ (P ∧ ¬Q ∧ R ∧ S) ∨ (¬P ∧ Q ∧ R ∧ S)

-- Problem statement
theorem prove_statement_II_must_be_true (d : ℕ) (h : exactly_three_true (statement_I d) (statement_II d) (statement_III d) (statement_IV d)) : 
  statement_II d :=
by
  -- proof goes here
  sorry

end prove_statement_II_must_be_true_l2346_234639


namespace david_lewis_meeting_point_l2346_234655

theorem david_lewis_meeting_point :
  ∀ (D : ℝ),
  (∀ t : ℝ, t ≥ 0 →
    ∀ distance_to_meeting_point : ℝ, 
    distance_to_meeting_point = D →
    ∀ speed_david speed_lewis distance_cities : ℝ,
    speed_david = 50 →
    speed_lewis = 70 →
    distance_cities = 350 →
    ((distance_cities + distance_to_meeting_point) / speed_lewis = distance_to_meeting_point / speed_david) →
    D = 145.83) :=
by
  intros D t ht distance_to_meeting_point h_distance speed_david speed_lewis distance_cities h_speed_david h_speed_lewis h_distance_cities h_meeting_time
  -- We need to prove D = 145.83 under the given conditions
  sorry

end david_lewis_meeting_point_l2346_234655


namespace prob_neither_alive_l2346_234677

/-- Define the probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1 / 4

/-- Define the probability that a wife will be alive for 10 more years -/
def prob_wife_alive : ℚ := 1 / 3

/-- Prove that the probability that neither the man nor his wife will be alive for 10 more years is 1/2 -/
theorem prob_neither_alive (p_man_alive p_wife_alive : ℚ)
    (h1 : p_man_alive = prob_man_alive) (h2 : p_wife_alive = prob_wife_alive) :
    (1 - p_man_alive) * (1 - p_wife_alive) = 1 / 2 :=
by
  sorry

end prob_neither_alive_l2346_234677


namespace expected_value_of_12_sided_die_is_6_5_l2346_234694

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5_l2346_234694


namespace weight_of_B_l2346_234628

/-- Let A, B, and C be the weights in kg of three individuals. If the average weight of A, B, and C is 45 kg,
and the average weight of A and B is 41 kg, and the average weight of B and C is 43 kg,
then the weight of B is 33 kg. -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 82) 
  (h3 : B + C = 86) : 
  B = 33 := 
by 
  sorry

end weight_of_B_l2346_234628


namespace factorization_eq_l2346_234682

variable (x y : ℝ)

theorem factorization_eq : 9 * y - 25 * x^2 * y = y * (3 + 5 * x) * (3 - 5 * x) :=
by sorry 

end factorization_eq_l2346_234682


namespace new_plan_cost_correct_l2346_234698

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end new_plan_cost_correct_l2346_234698


namespace relationship_abc_l2346_234691

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end relationship_abc_l2346_234691


namespace probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l2346_234625
-- Import all necessary libraries

-- Define the conditions as variables
variable (n k : ℕ) (p q : ℚ)
variable (dice_divisible_by_3_prob : ℚ)
variable (dice_not_divisible_by_3_prob : ℚ)

-- Assign values based on the problem statement
noncomputable def cond_replicate_n_fair_12_sided_dice := n = 7
noncomputable def cond_exactly_k_divisible_by_3 := k = 3
noncomputable def cond_prob_divisible_by_3 := dice_divisible_by_3_prob = 1 / 3
noncomputable def cond_prob_not_divisible_by_3 := dice_not_divisible_by_3_prob = 2 / 3

-- The theorem statement with the final answer incorporated
theorem probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice :
  cond_replicate_n_fair_12_sided_dice n →
  cond_exactly_k_divisible_by_3 k →
  cond_prob_divisible_by_3 dice_divisible_by_3_prob →
  cond_prob_not_divisible_by_3 dice_not_divisible_by_3_prob →
  p = (35 : ℚ) * ((1 / 3) ^ 3) * ((2 / 3) ^ 4) →
  q = (560 / 2187 : ℚ) →
  p = q :=
by
  intros
  sorry

end probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l2346_234625


namespace problem1_problem2_l2346_234687

-- Problem 1: Prove \( \sqrt{10} \times \sqrt{2} + \sqrt{15} \div \sqrt{3} = 3\sqrt{5} \)
theorem problem1 : Real.sqrt 10 * Real.sqrt 2 + Real.sqrt 15 / Real.sqrt 3 = 3 * Real.sqrt 5 := 
by sorry

-- Problem 2: Prove \( \sqrt{27} - (\sqrt{12} - \sqrt{\frac{1}{3}}) = \frac{4\sqrt{3}}{3} \)
theorem problem2 : Real.sqrt 27 - (Real.sqrt 12 - Real.sqrt (1 / 3)) = (4 * Real.sqrt 3) / 3 :=
by sorry

end problem1_problem2_l2346_234687


namespace hyperbola_inequality_l2346_234617

-- Define point P on the hyperbola in terms of a and b
theorem hyperbola_inequality (a b : ℝ) (h : (3*a + 3*b)^2 / 9 - (a - b)^2 = 1) : |a + b| ≥ 1 :=
sorry

end hyperbola_inequality_l2346_234617


namespace smallest_sum_is_minus_half_l2346_234651

def smallest_sum (x: ℝ) : ℝ := x^2 + x

theorem smallest_sum_is_minus_half : ∃ x : ℝ, ∀ y : ℝ, smallest_sum y ≥ smallest_sum (-1/2) :=
by
  use -1/2
  intros y
  sorry

end smallest_sum_is_minus_half_l2346_234651


namespace second_number_l2346_234615

theorem second_number (A B : ℝ) (h1 : A = 200) (h2 : 0.30 * A = 0.60 * B + 30) : B = 50 :=
by
  -- proof goes here
  sorry

end second_number_l2346_234615


namespace maximum_watchman_demand_l2346_234684

theorem maximum_watchman_demand (bet_loss : ℕ) (bet_win : ℕ) (x : ℕ) 
  (cond_bet_loss : bet_loss = 100)
  (cond_bet_win : bet_win = 100) :
  x < 200 :=
by
  have h₁ : bet_loss = 100 := cond_bet_loss
  have h₂ : bet_win = 100 := cond_bet_win
  sorry

end maximum_watchman_demand_l2346_234684


namespace percentage_reduction_is_20_percent_l2346_234623

-- Defining the initial and final prices
def initial_price : ℝ := 25
def final_price : ℝ := 16

-- Defining the percentage reduction
def percentage_reduction (x : ℝ) := 1 - x

-- The equation representing the two reductions:
def equation (x : ℝ) := initial_price * (percentage_reduction x) * (percentage_reduction x)

theorem percentage_reduction_is_20_percent :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ equation x = final_price ∧ x = 0.20 :=
by 
  sorry

end percentage_reduction_is_20_percent_l2346_234623


namespace find_g_of_polynomial_l2346_234614

variable (x : ℝ)

theorem find_g_of_polynomial :
  ∃ g : ℝ → ℝ, (4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) → (g x = -2 * x^4 - 13 * x^3 + 7 * x + 4) :=
sorry

end find_g_of_polynomial_l2346_234614


namespace wall_height_correct_l2346_234626

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.2
def brick_width  : ℝ := 0.1
def brick_height : ℝ := 0.08

-- Define the volume of one brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Total number of bricks used
def number_of_bricks : ℕ := 12250

-- Define the wall dimensions except height
def wall_length : ℝ := 10
def wall_width  : ℝ := 24.5

-- Total volume of all bricks
def volume_total_bricks : ℝ := number_of_bricks * volume_brick

-- Volume of the wall
def volume_wall (h : ℝ) : ℝ := wall_length * h * wall_width

-- The height of the wall
def wall_height : ℝ := 0.08

-- The theorem to prove
theorem wall_height_correct : volume_total_bricks = volume_wall wall_height :=
by
  sorry

end wall_height_correct_l2346_234626


namespace min_height_required_kingda_ka_l2346_234673

-- Definitions of the given conditions
def brother_height : ℕ := 180
def mary_relative_height : ℚ := 2 / 3
def growth_needed : ℕ := 20

-- Definition and statement of the problem
def marys_height : ℚ := mary_relative_height * brother_height
def minimum_height_required : ℚ := marys_height + growth_needed

theorem min_height_required_kingda_ka :
  minimum_height_required = 140 := by
  sorry

end min_height_required_kingda_ka_l2346_234673


namespace perpendicular_planes_l2346_234618

-- Definitions for lines and planes and their relationships
variable {a b : Line}
variable {α β : Plane}

-- Given conditions for the problem
axiom line_perpendicular (l1 l2 : Line) : Prop -- l1 ⊥ l2
axiom line_parallel (l1 l2 : Line) : Prop -- l1 ∥ l2
axiom line_plane_perpendicular (l : Line) (p : Plane) : Prop -- l ⊥ p
axiom line_plane_parallel (l : Line) (p : Plane) : Prop -- l ∥ p
axiom plane_perpendicular (p1 p2 : Plane) : Prop -- p1 ⊥ p2

-- Problem statement
theorem perpendicular_planes (h1 : line_perpendicular a b)
                            (h2 : line_plane_perpendicular a α)
                            (h3 : line_plane_perpendicular b β) :
                            plane_perpendicular α β :=
sorry

end perpendicular_planes_l2346_234618


namespace center_circle_sum_l2346_234697

theorem center_circle_sum (x y : ℝ) (h : x^2 + y^2 = 4 * x + 10 * y - 12) : x + y = 7 := 
sorry

end center_circle_sum_l2346_234697


namespace canteen_distance_l2346_234686

theorem canteen_distance (r G B : ℝ) (d_g d_b : ℝ) (h_g : G = 600) (h_b : B = 800) (h_dg_db : d_g = d_b) : 
  d_g = 781 :=
by
  -- Proof to be completed
  sorry

end canteen_distance_l2346_234686


namespace budget_equality_year_l2346_234663

theorem budget_equality_year :
  ∀ Q R V W : ℕ → ℝ,
  Q 0 = 540000 ∧ R 0 = 660000 ∧ V 0 = 780000 ∧ W 0 = 900000 ∧
  (∀ n, Q (n+1) = Q n + 40000 ∧ 
         R (n+1) = R n + 30000 ∧ 
         V (n+1) = V n - 10000 ∧ 
         W (n+1) = W n - 20000) →
  ∃ n : ℕ, 1990 + n = 1995 ∧ 
  Q n + R n = V n + W n := 
by 
  sorry

end budget_equality_year_l2346_234663


namespace number_of_females_l2346_234679

theorem number_of_females (total_people : ℕ) (avg_age_total : ℕ) 
  (avg_age_males : ℕ) (avg_age_females : ℕ) (females : ℕ) :
  total_people = 140 → avg_age_total = 24 →
  avg_age_males = 21 → avg_age_females = 28 → 
  females = 60 :=
by
  intros h1 h2 h3 h4
  -- Using the given conditions
  sorry

end number_of_females_l2346_234679


namespace total_people_in_house_l2346_234637

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l2346_234637


namespace max_m_value_l2346_234656

theorem max_m_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m = 3 :=
by
  sorry

end max_m_value_l2346_234656


namespace park_length_l2346_234619

theorem park_length (width : ℕ) (trees_per_sqft : ℕ) (num_trees : ℕ) (total_area : ℕ) (length : ℕ)
  (hw : width = 2000)
  (ht : trees_per_sqft = 20)
  (hn : num_trees = 100000)
  (ha : total_area = num_trees * trees_per_sqft)
  (hl : length = total_area / width) :
  length = 1000 :=
by
  sorry

end park_length_l2346_234619


namespace candy_partition_l2346_234608

theorem candy_partition :
  let candies := 10
  let boxes := 3
  ∃ ways : ℕ, ways = Nat.choose (candies + boxes - 1) (boxes - 1) ∧ ways = 66 :=
by
  let candies := 10
  let boxes := 3
  let ways := Nat.choose (candies + boxes - 1) (boxes - 1)
  have h : ways = 66 := sorry
  exact ⟨ways, ⟨rfl, h⟩⟩

end candy_partition_l2346_234608


namespace roots_equal_implies_a_eq_3_l2346_234620

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end roots_equal_implies_a_eq_3_l2346_234620


namespace candies_count_l2346_234699

variable (m_and_m : Nat) (starbursts : Nat)
variable (ratio_m_and_m_to_starbursts : Nat → Nat → Prop)

-- Definition of the ratio condition
def ratio_condition : Prop :=
  ∃ (k : Nat), (m_and_m = 7 * k) ∧ (starbursts = 4 * k)

-- The main theorem to prove
theorem candies_count (h : m_and_m = 56) (r : ratio_condition m_and_m starbursts) : starbursts = 32 :=
  by
  sorry

end candies_count_l2346_234699


namespace sequence_distinct_l2346_234667

theorem sequence_distinct (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) :
  ∀ i j : ℕ, i ≠ j → f i ≠ f j :=
by
  sorry

end sequence_distinct_l2346_234667


namespace max_blue_points_l2346_234650

theorem max_blue_points (n : ℕ) (h_n : n = 2016) :
  ∃ r : ℕ, r * (2016 - r) = 1008 * 1008 :=
by {
  sorry
}

end max_blue_points_l2346_234650


namespace find_expression_l2346_234693

theorem find_expression (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 84) (h2 : a = 32) : E = 80 :=
by
  -- Proof to be filled in here
  sorry

end find_expression_l2346_234693


namespace abc_sum_is_twelve_l2346_234678

theorem abc_sum_is_twelve
  (f : ℤ → ℤ)
  (a b c : ℕ)
  (h1 : f 1 = 10)
  (h2 : f 0 = 8)
  (h3 : f (-3) = -28)
  (h4 : ∀ x, x > 0 → f x = 2 * a * x + 6)
  (h5 : f 0 = a^2 * b)
  (h6 : ∀ x, x < 0 → f x = 2 * b * x + 2 * c)
  : a + b + c = 12 := sorry

end abc_sum_is_twelve_l2346_234678


namespace product_of_first_three_terms_l2346_234676

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end product_of_first_three_terms_l2346_234676


namespace remainder_2456789_div_7_l2346_234647

theorem remainder_2456789_div_7 :
  2456789 % 7 = 6 := 
by 
  sorry

end remainder_2456789_div_7_l2346_234647


namespace arithmetic_sequence_sixth_term_l2346_234602

variables (a d : ℤ)

theorem arithmetic_sequence_sixth_term :
  a + (a + d) + (a + 2 * d) = 12 →
  a + 3 * d = 0 →
  a + 5 * d = -4 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sixth_term_l2346_234602


namespace percentage_employees_six_years_or_more_l2346_234680

theorem percentage_employees_six_years_or_more:
  let marks : List ℕ := [6, 6, 7, 4, 3, 3, 3, 1, 1, 1]
  let total_employees (marks : List ℕ) (y : ℕ) := marks.foldl (λ acc m => acc + m * y) 0
  let employees_six_years_or_more (marks : List ℕ) (y : ℕ) := (marks.drop 6).foldl (λ acc m => acc + m * y) 0
  (employees_six_years_or_more marks 1 / total_employees marks 1 : ℚ) * 100 = 17.14 := by
  sorry

end percentage_employees_six_years_or_more_l2346_234680


namespace sum_even_if_product_odd_l2346_234662

theorem sum_even_if_product_odd (a b : ℤ) (h : (a * b) % 2 = 1) : (a + b) % 2 = 0 := 
by
  sorry

end sum_even_if_product_odd_l2346_234662


namespace exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l2346_234609

noncomputable def equation (x : ℝ) (k : ℝ) := x^2 - 2 * |x| - (2 * k + 1)^2

theorem exists_k_with_three_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ equation x1 k = 0 ∧ equation x2 k = 0 ∧ equation x3 k = 0 :=
sorry

theorem exists_k_with_two_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 k = 0 ∧ equation x2 k = 0 :=
sorry

end exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l2346_234609


namespace toys_produced_each_day_l2346_234613

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_per_week : ℕ) (H1 : total_weekly_production = 6500) (H2 : days_per_week = 5) : (total_weekly_production / days_per_week = 1300) :=
by {
  sorry
}

end toys_produced_each_day_l2346_234613
