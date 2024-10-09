import Mathlib

namespace no_elimination_method_l766_76675

theorem no_elimination_method
  (x y : ℤ)
  (h1 : x + 3 * y = 4)
  (h2 : 2 * x - y = 1) :
  ¬ (∀ z : ℤ, z = x + 3 * y - 3 * (2 * x - y)) →
  ∃ x y : ℤ, x + 3 * y - 3 * (2 * x - y) ≠ 0 := sorry

end no_elimination_method_l766_76675


namespace students_wrote_word_correctly_l766_76654

-- Definitions based on the problem conditions
def total_students := 50
def num_cat := 10
def num_rat := 18
def num_croc := total_students - num_cat - num_rat
def correct_cat := 15
def correct_rat := 15
def correct_total := correct_cat + correct_rat

-- Question: How many students wrote their word correctly?
-- Correct Answer: 8

theorem students_wrote_word_correctly : 
  num_cat + num_rat + num_croc = total_students 
  → correct_cat = 15 
  → correct_rat = 15 
  → correct_total = 30 
  → ∀ (num_correct_words : ℕ), num_correct_words = correct_total - num_croc 
  → num_correct_words = 8 := by 
  sorry

end students_wrote_word_correctly_l766_76654


namespace sum_of_first_49_primes_l766_76672

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end sum_of_first_49_primes_l766_76672


namespace find_digit_for_multiple_of_3_l766_76692

theorem find_digit_for_multiple_of_3 (d : ℕ) (h : d < 10) : 
  (56780 + d) % 3 = 0 ↔ d = 1 :=
by sorry

end find_digit_for_multiple_of_3_l766_76692


namespace solve_for_x_l766_76663

theorem solve_for_x (x y : ℝ) (h : 3 * x - 4 * y = 5) : x = (1 / 3) * (5 + 4 * y) :=
  sorry

end solve_for_x_l766_76663


namespace find_c_l766_76676

theorem find_c (c : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
  (hf : ∀ x, f x = 2 / (3 * x + c))
  (hfinv : ∀ x, f_inv x = (2 - 3 * x) / (3 * x)) :
  c = 3 :=
by
  sorry

end find_c_l766_76676


namespace calculate_expression_l766_76639

def inequality_holds (a b c d x : ℝ) : Prop :=
  (x - a) * (x - b) * (x - d) / (x - c) ≥ 0

theorem calculate_expression : 
  ∀ (a b c d : ℝ),
    a < b ∧ b < d ∧
    (∀ x : ℝ, 
      (inequality_holds a b c d x ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32))) →
    a + 2 * b + 3 * c + 4 * d = 160 :=
sorry

end calculate_expression_l766_76639


namespace no_integer_y_makes_Q_perfect_square_l766_76651

def Q (y : ℤ) : ℤ := y^4 + 8 * y^3 + 18 * y^2 + 10 * y + 41

theorem no_integer_y_makes_Q_perfect_square :
  ¬ ∃ y : ℤ, ∃ b : ℤ, Q y = b^2 :=
by
  intro h
  rcases h with ⟨y, b, hQ⟩
  sorry

end no_integer_y_makes_Q_perfect_square_l766_76651


namespace basketball_player_height_l766_76693

noncomputable def player_height (H : ℝ) : Prop :=
  let reach := 22 / 12
  let jump := 32 / 12
  let total_rim_height := 10 + (6 / 12)
  H + reach + jump = total_rim_height

theorem basketball_player_height : ∃ H : ℝ, player_height H → H = 6 :=
by
  use 6
  sorry

end basketball_player_height_l766_76693


namespace green_fish_always_15_l766_76683

def total_fish (T : ℕ) : Prop :=
∃ (O B G : ℕ),
B = T / 2 ∧
O = B - 15 ∧
T = B + O + G ∧
G = 15

theorem green_fish_always_15 (T : ℕ) : total_fish T → ∃ G, G = 15 :=
by
  intro h
  sorry

end green_fish_always_15_l766_76683


namespace denomination_of_bill_l766_76635

def cost_berries : ℝ := 7.19
def cost_peaches : ℝ := 6.83
def change_received : ℝ := 5.98

theorem denomination_of_bill :
  (cost_berries + cost_peaches) + change_received = 20.0 := 
by 
  sorry

end denomination_of_bill_l766_76635


namespace woody_writing_time_l766_76650

open Real

theorem woody_writing_time (W : ℝ) 
  (h1 : ∃ n : ℝ, n * 12 = W * 12 + 3) 
  (h2 : 12 * W + (12 * W + 3) = 39) :
  W = 1.5 :=
by sorry

end woody_writing_time_l766_76650


namespace hyperbola_solution_l766_76629

noncomputable def hyperbola_focus_parabola_equiv_hyperbola : Prop :=
  ∀ (a b c : ℝ),
    -- Condition 1: One focus of the hyperbola coincides with the focus of the parabola y^2 = 4sqrt(7)x
    (c^2 = a^2 + b^2) ∧ (c^2 = 7) →

    -- Condition 2: The hyperbola intersects the line y = x - 1 at points M and N
    (∃ M N : ℝ × ℝ, (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1^2 / a^2) - (M.2^2 / b^2) = 1) ∧ ((N.1^2 / a^2) - (N.2^2 / b^2) = 1)) →

    -- Condition 3: The x-coordinate of the midpoint of MN is -2/3
    (∀ M N : ℝ × ℝ, 
    (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1 + N.1) / 2 = -2/3)) →

    -- Conclusion: The standard equation of the hyperbola is x^2 / 2 - y^2 / 5 = 1
    a^2 = 2 ∧ b^2 = 5 ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → (x^2 / 2) - (y^2 / 5) = 1)

-- Proof omitted
theorem hyperbola_solution : hyperbola_focus_parabola_equiv_hyperbola :=
by sorry

end hyperbola_solution_l766_76629


namespace sum_of_first_15_terms_l766_76614

-- Given conditions: Sum of 4th and 12th term is 24
variable (a d : ℤ) (a_4 a_12 : ℤ)
variable (S : ℕ → ℤ)
variable (arithmetic_series_4_12_sum : 2 * a + 14 * d = 24)
variable (nth_term_def : ∀ n, a + (n - 1) * d = a_n)

-- Question: Sum of the first 15 terms of the progression
theorem sum_of_first_15_terms : S 15 = 180 := by
  sorry

end sum_of_first_15_terms_l766_76614


namespace speed_increase_impossible_l766_76670

theorem speed_increase_impossible (v : ℝ) : v = 60 → (¬ ∃ v', (1 / (v' / 60) = 0)) :=
by sorry

end speed_increase_impossible_l766_76670


namespace triangle_area_l766_76628

theorem triangle_area (h b : ℝ) (Hhb : h < b) :
  let P := (0, b)
  let B := (b, 0)
  let D := (h, h)
  let PD := b - h
  let DB := b - h
  1 / 2 * PD * DB = 1 / 2 * (b - h) ^ 2 := by 
  sorry

end triangle_area_l766_76628


namespace range_of_a_l766_76697

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ x^2 + (1 - a) * x + 3 - a > 0) ↔ a < 3 := 
sorry

end range_of_a_l766_76697


namespace find_units_digit_l766_76698

theorem find_units_digit (A : ℕ) (h : 10 * A + 2 = 20 + A + 9) : A = 3 :=
by
  sorry

end find_units_digit_l766_76698


namespace function_satisfies_conditions_l766_76615

def f (m n : ℕ) : ℕ := m * n

theorem function_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := 
by {
  sorry
}

end function_satisfies_conditions_l766_76615


namespace intersection_set_eq_l766_76603

-- Define M
def M : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1 }

-- Define N
def N : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 / 4) + (p.2 / 3) = 1 }

-- Define the intersection of M and N
def M_intersection_N := { x : ℝ | -4 ≤ x ∧ x ≤ 4 }

-- The theorem to be proved
theorem intersection_set_eq : 
  { p : ℝ × ℝ | p ∈ M ∧ p ∈ N } = { p : ℝ × ℝ | p.1 ∈ M_intersection_N } :=
sorry

end intersection_set_eq_l766_76603


namespace angle_C_side_c_area_of_triangle_l766_76673

open Real

variables (A B C a b c : Real)

noncomputable def acute_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
  (A < π / 2) ∧ (B < π / 2) ∧ (C < π / 2) ∧
  (a^2 - 2 * sqrt 3 * a + 2 = 0) ∧
  (b^2 - 2 * sqrt 3 * b + 2 = 0) ∧
  (2 * sin (A + B) - sqrt 3 = 0)

noncomputable def length_side_c (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2 - 2 * a * b * cos (π / 3))

noncomputable def area_triangle (a b : ℝ) : ℝ := 
  (1 / 2) * a * b * sin (π / 3)

theorem angle_C (h : acute_triangle A B C a b c) : C = π / 3 :=
  sorry

theorem side_c (h : acute_triangle A B C a b c) : c = sqrt 6 :=
  sorry

theorem area_of_triangle (h : acute_triangle A B C a b c) : area_triangle a b = sqrt 3 / 2 :=
  sorry

end angle_C_side_c_area_of_triangle_l766_76673


namespace minimum_beta_value_l766_76630

variable (α β : Real)

-- Defining the conditions given in the problem
def sin_alpha_condition : Prop := Real.sin α = -Real.sqrt 2 / 2
def cos_alpha_minus_beta_condition : Prop := Real.cos (α - β) = 1 / 2
def beta_greater_than_zero : Prop := β > 0

-- The theorem to be proven
theorem minimum_beta_value (h1 : sin_alpha_condition α) (h2 : cos_alpha_minus_beta_condition α β) (h3 : beta_greater_than_zero β) : β = Real.pi / 12 := 
sorry

end minimum_beta_value_l766_76630


namespace initial_number_of_friends_l766_76640

theorem initial_number_of_friends (X : ℕ) (H : 3 * (X - 3) = 15) : X = 8 :=
by
  sorry

end initial_number_of_friends_l766_76640


namespace find_triplet_solution_l766_76608

theorem find_triplet_solution (m n x y : ℕ) (hm : 0 < m) (hcoprime : Nat.gcd m n = 1) 
 (heq : (x^2 + y^2)^m = (x * y)^n) : 
  ∃ a : ℕ, x = 2^a ∧ y = 2^a ∧ n = m + 1 :=
by sorry

end find_triplet_solution_l766_76608


namespace average_employees_per_week_l766_76612

-- Define the number of employees hired each week
variables (x : ℕ)
noncomputable def employees_first_week := x + 200
noncomputable def employees_second_week := x
noncomputable def employees_third_week := x + 150
noncomputable def employees_fourth_week := 400

-- Given conditions as hypotheses
axiom h1 : employees_third_week / 2 = employees_fourth_week / 2
axiom h2 : employees_fourth_week = 400

-- Prove the average number of employees hired per week is 225
theorem average_employees_per_week :
  (employees_first_week + employees_second_week + employees_third_week + employees_fourth_week) / 4 = 225 :=
by
  sorry

end average_employees_per_week_l766_76612


namespace solve_problem_l766_76626

theorem solve_problem (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 5) : (b - d) ^ 2 = 16 :=
by
  sorry

end solve_problem_l766_76626


namespace vans_needed_l766_76661

theorem vans_needed (boys girls students_per_van total_vans : ℕ) 
  (hb : boys = 60) 
  (hg : girls = 80) 
  (hv : students_per_van = 28) 
  (t : total_vans = (boys + girls) / students_per_van) : 
  total_vans = 5 := 
by {
  sorry
}

end vans_needed_l766_76661


namespace distinct_solutions_l766_76691

theorem distinct_solutions : 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 - 7| = 2 * |x1 + 1| + |x1 - 3| ∧ |x2 - 7| = 2 * |x2 + 1| + |x2 - 3|) := 
by
  sorry

end distinct_solutions_l766_76691


namespace total_crayons_l766_76606

def box1_crayons := 3 * (8 + 4 + 5)
def box2_crayons := 4 * (7 + 6 + 3)
def box3_crayons := 2 * (11 + 5 + 2)
def unique_box_crayons := 9 + 2 + 7

theorem total_crayons : box1_crayons + box2_crayons + box3_crayons + unique_box_crayons = 169 := by
  sorry

end total_crayons_l766_76606


namespace saber_toothed_frog_tails_l766_76631

def tails_saber_toothed_frog (n k : ℕ) (x : ℕ) : Prop :=
  5 * n + 4 * k = 100 ∧ n + x * k = 64

theorem saber_toothed_frog_tails : ∃ x, ∃ n k : ℕ, tails_saber_toothed_frog n k x ∧ x = 3 := 
by
  sorry

end saber_toothed_frog_tails_l766_76631


namespace find_weight_of_silver_in_metal_bar_l766_76680

noncomputable def weight_loss_ratio_tin : ℝ := 1.375 / 10
noncomputable def weight_loss_ratio_silver : ℝ := 0.375
noncomputable def ratio_tin_silver : ℝ := 0.6666666666666664

theorem find_weight_of_silver_in_metal_bar (T S : ℝ)
  (h1 : T + S = 70)
  (h2 : T / S = ratio_tin_silver)
  (h3 : weight_loss_ratio_tin * T + weight_loss_ratio_silver * S = 7) :
  S = 15 :=
by
  sorry

end find_weight_of_silver_in_metal_bar_l766_76680


namespace inequality_proof_l766_76688

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) :
  (x / (y + 1) + y / (x + 1)) ≥ (2 / 3) ∧ (x = 1 / 2 ∧ y = 1 / 2 → x / (y + 1) + y / (x + 1) = 2 / 3) := by
  sorry

end inequality_proof_l766_76688


namespace tom_took_out_beads_l766_76652

-- Definitions of the conditions
def green_beads : Nat := 1
def brown_beads : Nat := 2
def red_beads : Nat := 3
def beads_left_in_container : Nat := 4

-- Total initial beads
def total_beads : Nat := green_beads + brown_beads + red_beads

-- The Lean problem statement to prove
theorem tom_took_out_beads : (total_beads - beads_left_in_container) = 2 :=
by
  sorry

end tom_took_out_beads_l766_76652


namespace band_member_share_l766_76620

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end band_member_share_l766_76620


namespace children_division_into_circles_l766_76647

theorem children_division_into_circles (n m k : ℕ) (hn : n = 5) (hm : m = 2) (trees_indistinguishable : true) (children_distinguishable : true) :
  ∃ ways, ways = 50 := 
by
  sorry

end children_division_into_circles_l766_76647


namespace intersection_A_B_l766_76604

def A : Set ℝ := {x | abs x < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l766_76604


namespace wedding_cost_correct_l766_76601

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def john_guests : ℕ := 50
def wife_guest_increase : ℕ := john_guests * 60 / 100
def total_wedding_cost : ℕ := venue_cost + cost_per_guest * (john_guests + wife_guest_increase)

theorem wedding_cost_correct : total_wedding_cost = 50000 :=
by
  sorry

end wedding_cost_correct_l766_76601


namespace original_number_l766_76637

theorem original_number (x : ℝ) (h : 1.35 * x = 935) : x = 693 := by
  sorry

end original_number_l766_76637


namespace sum_of_vars_l766_76636

variables (x y z w : ℤ)

theorem sum_of_vars (h1 : x - y + z = 7)
                    (h2 : y - z + w = 8)
                    (h3 : z - w + x = 4)
                    (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
by
  sorry

end sum_of_vars_l766_76636


namespace total_fish_caught_l766_76695

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) 
  (h₁ : leo_fish = 40) (h₂ : agrey_fish = leo_fish + 20) : 
  leo_fish + agrey_fish = 100 := 
by 
  sorry

end total_fish_caught_l766_76695


namespace range_of_a_l766_76660

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, 2 < (x : ℝ) ∧ (x : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ y : ℤ, 2 < (y : ℝ) ∧ (y : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ z : ℤ, 2 < (z : ℝ) ∧ (z : ℝ) ≤ 2 * a - 1) ∧ 
  (∀ w : ℤ, 2 < (w : ℝ) ∧ (w : ℝ) ≤ 2 * a - 1 → w = 3 ∨ w = 4 ∨ w = 5) :=
  by
    sorry

end range_of_a_l766_76660


namespace juliet_age_l766_76684

theorem juliet_age
    (M J R : ℕ)
    (h1 : J = M + 3)
    (h2 : J = R - 2)
    (h3 : M + R = 19) : J = 10 := by
  sorry

end juliet_age_l766_76684


namespace smaller_number_l766_76638

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 := by
  sorry

end smaller_number_l766_76638


namespace train_length_l766_76634

theorem train_length (s : ℝ) (t : ℝ) (h_s : s = 60) (h_t : t = 10) :
  ∃ L : ℝ, L = 166.7 := by
  sorry

end train_length_l766_76634


namespace divide_nuts_equal_l766_76624

-- Define the conditions: sequence of 64 nuts where adjacent differ by 1 gram
def is_valid_sequence (seq : List Int) :=
  seq.length = 64 ∧ (∀ i < 63, (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ + 1) ∨ (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ - 1))

-- Main theorem statement: prove that the sequence can be divided into two groups with equal number of nuts and equal weights
theorem divide_nuts_equal (seq : List Int) (h : is_valid_sequence seq) :
  ∃ (s1 s2 : List Int), s1.length = 32 ∧ s2.length = 32 ∧ (s1.sum = s2.sum) :=
sorry

end divide_nuts_equal_l766_76624


namespace matrix_pow_50_l766_76641

open Matrix

-- Define the given matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 2; -8, -5]

-- Define the expected result for C^50
def C_50 : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-199, -100; 400, 199]

-- Proposition asserting that C^50 equals the given result matrix
theorem matrix_pow_50 :
  C ^ 50 = C_50 := 
  by
  sorry

end matrix_pow_50_l766_76641


namespace arithmetic_progression_pairs_count_l766_76666

theorem arithmetic_progression_pairs_count (x y : ℝ) 
  (h1 : x = (15 + y) / 2)
  (h2 : x + x * y = 2 * y) : 
  (∃ x1 y1, x1 = (15 + y1) / 2 ∧ x1 + x1 * y1 = 2 * y1 ∧ x1 = (9 + 3 * Real.sqrt 7) / 2 ∧ y1 = -6 + 3 * Real.sqrt 7) ∨ 
  (∃ x2 y2, x2 = (15 + y2) / 2 ∧ x2 + x2 * y2 = 2 * y2 ∧ x2 = (9 - 3 * Real.sqrt 7) / 2 ∧ y2 = -6 - 3 * Real.sqrt 7) := 
sorry

end arithmetic_progression_pairs_count_l766_76666


namespace calculate_expression_l766_76633

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end calculate_expression_l766_76633


namespace parabola_tangent_line_l766_76669

noncomputable def gcd (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem parabola_tangent_line (a b c : ℕ) (h1 : a^2 + (104 / 5) * a * b - 4 * b * c = 0)
  (h2 : b^2 - 5 * a^2 + 4 * a * c = 0) (hgcd : gcd a b c = 1) :
  a + b + c = 17 := by
  sorry

end parabola_tangent_line_l766_76669


namespace boys_without_glasses_l766_76658

def total_students_with_glasses : ℕ := 36
def girls_with_glasses : ℕ := 21
def total_boys : ℕ := 30

theorem boys_without_glasses :
  total_boys - (total_students_with_glasses - girls_with_glasses) = 15 :=
by
  sorry

end boys_without_glasses_l766_76658


namespace total_miles_driven_l766_76621

-- Define the required variables and their types
variables (avg1 avg2 : ℝ) (gallons1 gallons2 : ℝ) (miles1 miles2 : ℝ)

-- State the conditions
axiom sum_avg_mpg : avg1 + avg2 = 75
axiom first_car_gallons : gallons1 = 25
axiom second_car_gallons : gallons2 = 35
axiom first_car_avg_mpg : avg1 = 40

-- Declare the function to calculate miles driven
def miles_driven (avg_mpg gallons : ℝ) : ℝ := avg_mpg * gallons

-- Declare the theorem for proof
theorem total_miles_driven : miles_driven avg1 gallons1 + miles_driven avg2 gallons2 = 2225 := by
  sorry

end total_miles_driven_l766_76621


namespace min_additional_games_l766_76623

def num_initial_games : ℕ := 4
def num_lions_won : ℕ := 3
def num_eagles_won : ℕ := 1
def win_threshold : ℝ := 0.90

theorem min_additional_games (M : ℕ) : (num_eagles_won + M) / (num_initial_games + M) ≥ win_threshold ↔ M ≥ 26 :=
by
  sorry

end min_additional_games_l766_76623


namespace batsman_average_excluding_highest_and_lowest_l766_76679

theorem batsman_average_excluding_highest_and_lowest (average : ℝ) (innings : ℕ) (highest_score : ℝ) (score_difference : ℝ) :
  average = 63 →
  innings = 46 →
  highest_score = 248 →
  score_difference = 150 →
  (average * innings - highest_score - (highest_score - score_difference)) / (innings - 2) = 58 :=
by
  intros h_average h_innings h_highest h_difference
  simp [h_average, h_innings, h_highest, h_difference]
  -- Here the detailed steps from the solution would come in to verify the simplification
  sorry

end batsman_average_excluding_highest_and_lowest_l766_76679


namespace sequences_count_n3_sequences_count_n6_sequences_count_n9_l766_76645

inductive Shape
  | triangle
  | square
  | rectangle (k : ℕ)

open Shape

def transition (s : Shape) : List Shape :=
  match s with
  | triangle => [triangle, square]
  | square => [rectangle 1]
  | rectangle k =>
    if k = 0 then [rectangle 1] else [rectangle (k - 1), rectangle (k + 1)]

def count_sequences (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (shapes : List Shape) : ℕ :=
    if m = 0 then shapes.length
    else
      let next_shapes := shapes.bind transition
      aux (m - 1) next_shapes
  aux n [square]

theorem sequences_count_n3 : count_sequences 3 = 5 :=
  by sorry

theorem sequences_count_n6 : count_sequences 6 = 24 :=
  by sorry

theorem sequences_count_n9 : count_sequences 9 = 149 :=
  by sorry

end sequences_count_n3_sequences_count_n6_sequences_count_n9_l766_76645


namespace jan_25_on_thursday_l766_76687

/-- 
  Given that December 25 is on Monday,
  prove that January 25 in the following year falls on Thursday.
-/
theorem jan_25_on_thursday (day_of_week : Fin 7) (h : day_of_week = 0) : 
  ((day_of_week + 31) % 7 + 25) % 7 = 4 := 
sorry

end jan_25_on_thursday_l766_76687


namespace find_m2n_plus_mn2_minus_mn_l766_76605

def quadratic_roots (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0

theorem find_m2n_plus_mn2_minus_mn :
  ∃ m n : ℝ, quadratic_roots 1 2015 (-1) m n ∧ m^2 * n + m * n^2 - m * n = 2016 :=
by
  sorry

end find_m2n_plus_mn2_minus_mn_l766_76605


namespace capacity_of_buckets_l766_76656

theorem capacity_of_buckets :
  (∃ x : ℝ, 26 * x = 39 * 9) → (∃ x : ℝ, 26 * x = 351 ∧ x = 13.5) :=
by
  sorry

end capacity_of_buckets_l766_76656


namespace right_triangle_ratio_is_4_l766_76689

noncomputable def right_triangle_rectangle_ratio (b h xy : ℝ) : Prop :=
  (0.4 * (1/2) * b * h = 0.25 * xy) ∧ (xy = b * h) → (b / h = 4)

theorem right_triangle_ratio_is_4 (b h xy : ℝ) (h1 : 0.4 * (1/2) * b * h = 0.25 * xy)
(h2 : xy = b * h) : b / h = 4 :=
sorry

end right_triangle_ratio_is_4_l766_76689


namespace take_home_pay_l766_76662

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end take_home_pay_l766_76662


namespace case1_case2_case3_l766_76649

-- Definitions from conditions
def tens_digit_one : ℕ := sorry
def units_digit_one : ℕ := sorry
def units_digit_two : ℕ := sorry
def tens_digit_two : ℕ := sorry
def sum_units_digits_ten : Prop := units_digit_one + units_digit_two = 10
def same_digit : ℕ := sorry
def sum_tens_digits_ten : Prop := tens_digit_one + tens_digit_two = 10

-- The proof problems
theorem case1 (A B D : ℕ) (hBplusD : B + D = 10) :
  (10 * A + B) * (10 * A + D) = 100 * (A^2 + A) + B * D :=
sorry

theorem case2 (A B C : ℕ) (hAplusC : A + C = 10) :
  (10 * A + B) * (10 * C + B) = 100 * A * C + 100 * B + B^2 :=
sorry

theorem case3 (A B C : ℕ) (hAplusB : A + B = 10) :
  (10 * A + B) * (10 * C + C) = 100 * A * C + 100 * C + B * C :=
sorry

end case1_case2_case3_l766_76649


namespace rhombus_area_of_square_4_l766_76686

theorem rhombus_area_of_square_4 :
  let A := (0, 4)
  let B := (0, 0)
  let C := (4, 0)
  let D := (4, 4)
  let F := (0, 2)  -- Midpoint of AB
  let E := (4, 2)  -- Midpoint of CD
  let FG := 2 -- Half of the side of the square (since F and E are midpoints)
  let GH := 2
  let HE := 2
  let EF := 2
  let rhombus_FGEH_area := 1 / 2 * FG * EH
  rhombus_FGEH_area = 4 := sorry

end rhombus_area_of_square_4_l766_76686


namespace complement_of_A_l766_76625

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | abs (x - 1) > 1 }

-- Define the problem statement
theorem complement_of_A :
  ∀ x : ℝ, x ∈ compl A ↔ x ∈ Icc 0 2 :=
by
  intro x
  rw [mem_compl_iff, mem_Icc]
  sorry

end complement_of_A_l766_76625


namespace find_k_l766_76696

theorem find_k (k : ℚ) :
  (5 + ∑' n : ℕ, (5 + 2*k*(n+1)) / 4^n) = 10 → k = 15/4 :=
by
  sorry

end find_k_l766_76696


namespace floor_ceil_diff_l766_76668

theorem floor_ceil_diff (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌊x⌋ + x - ⌈x⌉ = x - 1 :=
sorry

end floor_ceil_diff_l766_76668


namespace triangle_square_ratio_l766_76699

theorem triangle_square_ratio :
  ∀ (x y : ℝ), (x = 60 / 17) → (y = 780 / 169) → (x / y = 78 / 102) :=
by
  intros x y hx hy
  rw [hx, hy]
  -- the proof is skipped, as instructed
  sorry

end triangle_square_ratio_l766_76699


namespace polynomial_root_theorem_l766_76681

theorem polynomial_root_theorem
  (α β γ δ p q : ℝ)
  (h₁ : α + β = -p)
  (h₂ : α * β = 1)
  (h₃ : γ + δ = -q)
  (h₄ : γ * δ = 1) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
by
  sorry

end polynomial_root_theorem_l766_76681


namespace max_amount_xiao_li_spent_l766_76613

theorem max_amount_xiao_li_spent (a m n : ℕ) :
  33 ≤ m ∧ m < n ∧ n ≤ 37 ∧
  ∃ (x y : ℕ), 
  (25 * (a - x) + m * (a - y) + n * (x + y + a) = 700) ∧ 
  (25 * x + m * y + n * (3*a - x - y) = 1200) ∧
  ( 675 <= 700 - 25) :=
sorry

end max_amount_xiao_li_spent_l766_76613


namespace nth_equation_l766_76632

theorem nth_equation (n : ℕ) (hn : n ≠ 0) : 
  (↑n + 2) / ↑n - 2 / (↑n + 2) = ((↑n + 2)^2 + ↑n^2) / (↑n * (↑n + 2)) - 1 :=
by
  sorry

end nth_equation_l766_76632


namespace cameras_not_in_both_l766_76611

-- Definitions for the given conditions
def shared_cameras : ℕ := 12
def sarah_cameras : ℕ := 24
def mike_unique_cameras : ℕ := 9

-- The proof statement
theorem cameras_not_in_both : (sarah_cameras - shared_cameras) + mike_unique_cameras = 21 := by
  sorry

end cameras_not_in_both_l766_76611


namespace sum_of_squares_of_consecutive_integers_l766_76677

theorem sum_of_squares_of_consecutive_integers (a : ℝ) (h : (a-1)*a*(a+1) = 36*a) :
  (a-1)^2 + a^2 + (a+1)^2 = 77 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l766_76677


namespace largest_log_value_l766_76659

theorem largest_log_value :
  ∃ (x y z t : ℝ) (a b c : ℝ),
    x ≤ y ∧ y ≤ z ∧ z ≤ t ∧
    a = Real.log y / Real.log x ∧
    b = Real.log z / Real.log y ∧
    c = Real.log t / Real.log z ∧
    a = 15 ∧ b = 20 ∧ c = 21 ∧
    (∃ u v w, u = a * b ∧ v = b * c ∧ w = a * b * c ∧ w = 420) := sorry

end largest_log_value_l766_76659


namespace max_kopeyka_coins_l766_76664

def coins (n : Nat) (k : Nat) : Prop :=
  k ≤ n / 4 + 1

theorem max_kopeyka_coins : coins 2001 501 :=
by
  sorry

end max_kopeyka_coins_l766_76664


namespace find_a2_l766_76609

def arithmetic_sequence (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n + d 

def sum_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a2 (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a a1 d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : a1 = -2010)
  (h4 : (S 2010) / 2010 - (S 2008) / 2008 = 2) :
  a 2 = -2008 :=
sorry

end find_a2_l766_76609


namespace pentagon_diagonal_l766_76627

theorem pentagon_diagonal (a d : ℝ) (h : d^2 = a^2 + a * d) : 
  d = a * (Real.sqrt 5 + 1) / 2 :=
sorry

end pentagon_diagonal_l766_76627


namespace problem_proof_l766_76622

noncomputable def problem (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y ≥ 0 ∧ x ^ 2019 + y = 1) → (x + y ^ 2019 > 1 - 1 / 300)

theorem problem_proof (x y : ℝ) : problem x y :=
by
  intros h
  sorry

end problem_proof_l766_76622


namespace senior_discount_percentage_l766_76616

theorem senior_discount_percentage 
    (cost_shorts : ℕ)
    (count_shorts : ℕ)
    (cost_shirts : ℕ)
    (count_shirts : ℕ)
    (amount_paid : ℕ)
    (total_cost : ℕ := (cost_shorts * count_shorts) + (cost_shirts * count_shirts))
    (discount_received : ℕ := total_cost - amount_paid)
    (discount_percentage : ℚ := (discount_received : ℚ) / total_cost * 100) :
    count_shorts = 3 ∧ cost_shorts = 15 ∧ count_shirts = 5 ∧ cost_shirts = 17 ∧ amount_paid = 117 →
    discount_percentage = 10 := 
by
    sorry

end senior_discount_percentage_l766_76616


namespace remainder_when_587421_divided_by_6_l766_76644

theorem remainder_when_587421_divided_by_6 :
  ¬ (587421 % 2 = 0) → (587421 % 3 = 0) → 587421 % 6 = 3 :=
by sorry

end remainder_when_587421_divided_by_6_l766_76644


namespace incorrect_conclusion_l766_76678

theorem incorrect_conclusion (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a < 1/b ∧ 1/b < 0) : ¬ (ab > b^2) :=
by
  { sorry }

end incorrect_conclusion_l766_76678


namespace dandelion_seed_production_l766_76674

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end dandelion_seed_production_l766_76674


namespace not_possible_identical_nonzero_remainders_l766_76648

theorem not_possible_identical_nonzero_remainders :
  ¬ ∃ (a : ℕ → ℕ) (r : ℕ), (r > 0) ∧ (∀ i : Fin 100, a i % (a ((i + 1) % 100)) = r) :=
by
  sorry

end not_possible_identical_nonzero_remainders_l766_76648


namespace tilly_counts_total_stars_l766_76619

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end tilly_counts_total_stars_l766_76619


namespace number_is_24point2_l766_76685

noncomputable def certain_number (x : ℝ) : Prop :=
  0.12 * x = 2.904

theorem number_is_24point2 : certain_number 24.2 :=
by
  unfold certain_number
  sorry

end number_is_24point2_l766_76685


namespace probability_win_all_games_l766_76655

variable (p : ℚ) (n : ℕ)

-- Define the conditions
def probability_of_winning := p = 2 / 3
def number_of_games := n = 6
def independent_games := true

-- The theorem we want to prove
theorem probability_win_all_games (h₁ : probability_of_winning p)
                                   (h₂ : number_of_games n)
                                   (h₃ : independent_games) :
  p^n = 64 / 729 :=
sorry

end probability_win_all_games_l766_76655


namespace largest_of_three_roots_l766_76694

theorem largest_of_three_roots (p q r : ℝ) (hpqr_sum : p + q + r = 3) 
    (hpqr_prod_sum : p * q + p * r + q * r = -8) (hpqr_prod : p * q * r = -15) :
    max p (max q r) = 3 := 
sorry

end largest_of_three_roots_l766_76694


namespace length_of_plot_l766_76617

-- Define the conditions
def width : ℝ := 60
def num_poles : ℕ := 60
def dist_between_poles : ℝ := 5
def num_intervals : ℕ := num_poles - 1
def perimeter : ℝ := num_intervals * dist_between_poles

-- Define the theorem and the correctness condition
theorem length_of_plot : 
  perimeter = 2 * (length + width) → 
  length = 87.5 :=
by
  sorry

end length_of_plot_l766_76617


namespace max_planes_determined_by_15_points_l766_76690

theorem max_planes_determined_by_15_points : ∃ (n : ℕ), n = 455 ∧ 
  ∀ (P : Finset (Fin 15)), (15 + 1) ∣ (P.card * (P.card - 1) * (P.card - 2) / 6) → n = (15 * 14 * 13) / 6 :=
by
  sorry

end max_planes_determined_by_15_points_l766_76690


namespace pentagon_segment_condition_l766_76667

-- Define the problem context and hypothesis
variable (a b c d e : ℝ)

theorem pentagon_segment_condition 
  (h₁ : a + b + c + d + e = 3)
  (h₂ : a ≤ b)
  (h₃ : b ≤ c)
  (h₄ : c ≤ d)
  (h₅ : d ≤ e) : 
  a < 3 / 2 ∧ b < 3 / 2 ∧ c < 3 / 2 ∧ d < 3 / 2 ∧ e < 3 / 2 := 
sorry

end pentagon_segment_condition_l766_76667


namespace find_number_lemma_l766_76682

theorem find_number_lemma (x : ℝ) (a b c d : ℝ) (h₁ : x = 5) 
  (h₂ : a = 0.47 * 1442) (h₃ : b = 0.36 * 1412) 
  (h₄ : c = a - b) (h₅ : d + c = x) : 
  d = -164.42 :=
by
  sorry

end find_number_lemma_l766_76682


namespace simplify_fraction_l766_76646

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end simplify_fraction_l766_76646


namespace logically_equivalent_to_original_l766_76671

def original_statement (E W : Prop) : Prop := E → ¬ W
def statement_I (E W : Prop) : Prop := W → E
def statement_II (E W : Prop) : Prop := ¬ E → ¬ W
def statement_III (E W : Prop) : Prop := W → ¬ E
def statement_IV (E W : Prop) : Prop := ¬ E ∨ ¬ W

theorem logically_equivalent_to_original (E W : Prop) :
  (original_statement E W ↔ statement_III E W) ∧
  (original_statement E W ↔ statement_IV E W) :=
  sorry

end logically_equivalent_to_original_l766_76671


namespace initial_percentage_decrease_l766_76665

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₁ : P > 0) (h₂ : 1.55 * (1 - x / 100) = 1.24) :
    x = 20 :=
by
  sorry

end initial_percentage_decrease_l766_76665


namespace problem_solution_l766_76618

theorem problem_solution (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3) →
  (a₁ + a₂ + a₃ = 19) :=
by
  -- Given condition: for any real number x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3
  -- We need to prove: a₁ + a₂ + a₃ = 19
  sorry

end problem_solution_l766_76618


namespace neznaika_discrepancy_l766_76657

theorem neznaika_discrepancy :
  let KL := 1 -- Assume we start with 1 kiloluna
  let kg := 1 -- Assume we start with 1 kilogram
  let snayka_kg (KL : ℝ) := (KL / 4) * 0.96 -- Conversion rule from kilolunas to kilograms by Snayka
  let neznaika_kl (kg : ℝ) := (kg * 4) * 1.04 -- Conversion rule from kilograms to kilolunas by Neznaika
  let correct_kl (kg : ℝ) := kg / 0.24 -- Correct conversion from kilograms to kilolunas
  
  let result_kl := (neznaika_kl 1) -- Neznaika's computed kilolunas for 1 kilogram
  let correct_kl_val := (correct_kl 1) -- Correct kilolunas for 1 kilogram
  let ratio := result_kl / correct_kl_val -- Ratio of Neznaika's value to Correct value
  let discrepancy := 100 * (1 - ratio) -- Discrepancy percentage

  result_kl = 4.16 ∧ correct_kl_val = 4.1667 ∧ discrepancy = 0.16 := 
by
  sorry

end neznaika_discrepancy_l766_76657


namespace complex_triple_sum_eq_sqrt3_l766_76642

noncomputable section

open Complex

theorem complex_triple_sum_eq_sqrt3 {a b c : ℂ} (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : a + b + c ≠ 0) (h5 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3) : abs (a + b + c) = Real.sqrt 3 :=
by
  sorry

end complex_triple_sum_eq_sqrt3_l766_76642


namespace height_after_five_years_l766_76600

namespace PapayaTreeGrowth

def growth_first_year := true → ℝ
def growth_second_year (x : ℝ) := 1.5 * x
def growth_third_year (x : ℝ) := 1.5 * growth_second_year x
def growth_fourth_year (x : ℝ) := 2 * growth_third_year x
def growth_fifth_year (x : ℝ) := 0.5 * growth_fourth_year x

def total_growth (x : ℝ) := x + growth_second_year x + growth_third_year x +
                             growth_fourth_year x + growth_fifth_year x

theorem height_after_five_years (x : ℝ) (H : total_growth x = 23) : x = 2 :=
by
  sorry

end PapayaTreeGrowth

end height_after_five_years_l766_76600


namespace jeopardy_episode_length_l766_76653

-- Definitions based on the conditions
def num_episodes_jeopardy : ℕ := 2
def num_episodes_wheel : ℕ := 2
def wheel_twice_jeopardy (J : ℝ) : ℝ := 2 * J
def total_time_watched : ℝ := 120 -- in minutes

-- Condition stating the total time watched in terms of J
def total_watching_time_formula (J : ℝ) : ℝ :=
  num_episodes_jeopardy * J + num_episodes_wheel * (wheel_twice_jeopardy J)

theorem jeopardy_episode_length : ∃ J : ℝ, total_watching_time_formula J = total_time_watched ∧ J = 20 :=
by
  use 20
  simp [total_watching_time_formula, wheel_twice_jeopardy, num_episodes_jeopardy, num_episodes_wheel, total_time_watched]
  sorry

end jeopardy_episode_length_l766_76653


namespace solve_system_l766_76643

theorem solve_system : ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 8 * x + 5 * y = 7 ∧ x = 1 / 4 ∧ y = 1 :=
by
  sorry

end solve_system_l766_76643


namespace matrix_sum_correct_l766_76602

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -1], ![3, 7]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![ -6, 8], ![5, -2]]
def mat_sum : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, 7], ![8, 5]]

theorem matrix_sum_correct : mat1 + mat2 = mat_sum :=
by
  rw [mat1, mat2]
  sorry

end matrix_sum_correct_l766_76602


namespace min_value_of_f_inequality_a_b_l766_76610

theorem min_value_of_f :
  ∃ m : ℝ, m = 4 ∧ (∀ x : ℝ, |x + 3| + |x - 1| ≥ m) :=
sorry

theorem inequality_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1 / a + 4 / b ≥ 9 / 4) :=
sorry

end min_value_of_f_inequality_a_b_l766_76610


namespace number_with_all_8s_is_divisible_by_13_l766_76607

theorem number_with_all_8s_is_divisible_by_13 :
  ∀ (N : ℕ), (N = 8 * (10^1974 - 1) / 9) → 13 ∣ N :=
by
  sorry

end number_with_all_8s_is_divisible_by_13_l766_76607
