import Mathlib

namespace part_a_l313_313730

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem part_a :
  ∀ (N : ℕ), (N = (sum_of_digits N) ^ 2) → (N = 1 ∨ N = 81) :=
by
  intros N h
  sorry

end part_a_l313_313730


namespace range_of_a_l313_313213

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l313_313213


namespace fraction_to_decimal_l313_313908

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313908


namespace find_k_l313_313766

theorem find_k : ∃ k : ℝ, (3 * k - 4) / (k + 7) = 2 / 5 ∧ k = 34 / 13 :=
by
  use 34 / 13
  sorry

end find_k_l313_313766


namespace inequality_solution_l313_313119

theorem inequality_solution (x : ℤ) : (1 + x) / 2 - (2 * x + 1) / 3 ≤ 1 → x ≥ -5 := 
by
  sorry

end inequality_solution_l313_313119


namespace distance_difference_l313_313818

-- Definitions related to the problem conditions
variables (v D_AB D_BC D_AC : ℝ)

-- Conditions
axiom h1 : D_AB = v * 7
axiom h2 : D_BC = v * 5
axiom h3 : D_AC = 6
axiom h4 : D_AC = D_AB + D_BC

-- Theorem for proof problem
theorem distance_difference : D_AB - D_BC = 1 :=
by sorry

end distance_difference_l313_313818


namespace max_probability_pc_l313_313621

variables (p1 p2 p3 : ℝ)
variable (h : p3 > p2 ∧ p2 > p1 ∧ p1 > 0)

def PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem max_probability_pc : PC > PA ∧ PC > PB := 
by 
  sorry

end max_probability_pc_l313_313621


namespace right_triangle_leg_length_l313_313243

theorem right_triangle_leg_length
  (A : ℝ)
  (b h : ℝ)
  (hA : A = 800)
  (hb : b = 40)
  (h_area : A = (1 / 2) * b * h) :
  h = 40 :=
by
  sorry

end right_triangle_leg_length_l313_313243


namespace count_1320_factors_l313_313355

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l313_313355


namespace certain_number_value_l313_313999

theorem certain_number_value (x : ℝ) (certain_number : ℝ) 
  (h1 : x = 0.25) 
  (h2 : 625^(-x) + 25^(-2 * x) + certain_number^(-4 * x) = 11) : 
  certain_number = 5 / 53 := 
sorry

end certain_number_value_l313_313999


namespace sum_of_roots_expression_involving_roots_l313_313492

variables {a b : ℝ}

axiom roots_of_quadratic :
  (a^2 + 3 * a - 2 = 0) ∧ (b^2 + 3 * b - 2 = 0)

theorem sum_of_roots :
  a + b = -3 :=
by 
  sorry

theorem expression_involving_roots :
  a^3 + 3 * a^2 + 2 * b = -6 :=
by 
  sorry

end sum_of_roots_expression_involving_roots_l313_313492


namespace distinct_four_digit_numbers_l313_313341

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l313_313341


namespace kenya_peanuts_correct_l313_313614

def jose_peanuts : ℕ := 85
def kenya_extra_peanuts : ℕ := 48
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := 
by 
  sorry

end kenya_peanuts_correct_l313_313614


namespace convert_fraction_to_decimal_l313_313948

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313948


namespace trigonometric_identity_l313_313639

theorem trigonometric_identity : 
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  -- Here we assume standard trigonometric identities and basic properties already handled by Mathlib
  sorry

end trigonometric_identity_l313_313639


namespace bushes_needed_for_60_zucchinis_l313_313481

-- Each blueberry bush yields 10 containers of blueberries.
def containers_per_bush : ℕ := 10

-- 6 containers of blueberries can be traded for 3 zucchinis.
def containers_to_zucchinis (containers zucchinis : ℕ) : Prop := containers = 6 ∧ zucchinis = 3

theorem bushes_needed_for_60_zucchinis (bushes containers zucchinis : ℕ) :
  containers_per_bush = 10 →
  containers_to_zucchinis 6 3 →
  zucchinis = 60 →
  bushes = 12 :=
by
  intros h1 h2 h3
  sorry

end bushes_needed_for_60_zucchinis_l313_313481


namespace heartsuit_3_8_l313_313669

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_3_8_l313_313669


namespace white_area_l313_313594

/-- The area of a 5 by 17 rectangular sign. -/
def sign_area : ℕ := 5 * 17

/-- The area covered by the letter L. -/
def L_area : ℕ := 5 * 1 + 1 * 2

/-- The area covered by the letter O. -/
def O_area : ℕ := (3 * 3) - (1 * 1)

/-- The area covered by the letter V. -/
def V_area : ℕ := 2 * (3 * 1)

/-- The area covered by the letter E. -/
def E_area : ℕ := 3 * (1 * 3)

/-- The total area covered by the letters L, O, V, E. -/
def sum_black_area : ℕ := L_area + O_area + V_area + E_area

/-- The problem statement: Calculate the area of the white portion of the sign. -/
theorem white_area : sign_area - sum_black_area = 55 :=
by
  -- Place the proof here
  sorry

end white_area_l313_313594


namespace union_of_A_and_B_l313_313786

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
by
  sorry

end union_of_A_and_B_l313_313786


namespace triangular_pyramid_nonexistence_l313_313382

theorem triangular_pyramid_nonexistence
    (h : ℕ)
    (hb : ℕ)
    (P : ℕ)
    (h_eq : h = 60)
    (hb_eq : hb = 61)
    (P_eq : P = 62) :
    ¬ ∃ (a b c : ℝ), a + b + c = P ∧ 60^2 = 61^2 - (a^2 / 3) :=
by 
  sorry

end triangular_pyramid_nonexistence_l313_313382


namespace fifth_term_is_67_l313_313196

noncomputable def satisfies_sequence (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :=
  (a = 3) ∧ (d = 27) ∧ 
  (a = (1/3 : ℚ) * (3 + b)) ∧
  (b = (1/3 : ℚ) * (a + 27)) ∧
  (27 = (1/3 : ℚ) * (b + e))

theorem fifth_term_is_67 :
  ∃ (e : ℕ), satisfies_sequence 3 a b 27 e ∧ e = 67 :=
sorry

end fifth_term_is_67_l313_313196


namespace min_quotient_l313_313763

def digits_distinct (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def quotient (a b c : ℕ) : ℚ := 
  (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ)

theorem min_quotient (a b c : ℕ) (h1 : b > 3) (h2 : c ≠ b) (h3: digits_distinct a b c) : 
  quotient a b c ≥ 19.62 :=
sorry

end min_quotient_l313_313763


namespace number_of_pairs_is_2_pow_14_l313_313514

noncomputable def number_of_pairs_satisfying_conditions : ℕ :=
  let fact5 := Nat.factorial 5
  let fact50 := Nat.factorial 50
  Nat.card {p : ℕ × ℕ | Nat.gcd p.1 p.2 = fact5 ∧ Nat.lcm p.1 p.2 = fact50}

theorem number_of_pairs_is_2_pow_14 :
  number_of_pairs_satisfying_conditions = 2^14 := by
  sorry

end number_of_pairs_is_2_pow_14_l313_313514


namespace true_statement_l313_313145

variables {Plane Line : Type}
variables (α β γ : Plane) (a b m n : Line)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def intersect_line (p q : Plane) : Line := sorry

-- Given conditions for the problem
variables (h1 : (α ≠ β)) (h2 : (parallel α β))
variables (h3 : (intersect_line α γ = a)) (h4 : (intersect_line β γ = b))

-- Statement verifying the true condition based on the above givens
theorem true_statement : parallel a b :=
by sorry

end true_statement_l313_313145


namespace peach_ratios_and_percentages_l313_313519

def red_peaches : ℕ := 8
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6
def orange_peaches : ℕ := 4
def total_peaches : ℕ := red_peaches + yellow_peaches + green_peaches + orange_peaches

theorem peach_ratios_and_percentages :
  ((green_peaches : ℚ) / total_peaches = 3 / 16) ∧
  ((green_peaches : ℚ) / total_peaches * 100 = 18.75) ∧
  ((yellow_peaches : ℚ) / total_peaches = 7 / 16) ∧
  ((yellow_peaches : ℚ) / total_peaches * 100 = 43.75) :=
by {
  sorry
}

end peach_ratios_and_percentages_l313_313519


namespace sum_sequence_S_n_l313_313774

variable {S : ℕ+ → ℚ}
noncomputable def S₁ : ℚ := 1 / 2
noncomputable def S₂ : ℚ := 5 / 6
noncomputable def S₃ : ℚ := 49 / 72
noncomputable def S₄ : ℚ := 205 / 288

theorem sum_sequence_S_n (n : ℕ+) :
  (S 1 = S₁) ∧ (S 2 = S₂) ∧ (S 3 = S₃) ∧ (S 4 = S₄) ∧ (∀ n : ℕ+, S n = n / (n + 1)) :=
by
  sorry

end sum_sequence_S_n_l313_313774


namespace quadratic_discriminant_l313_313034

theorem quadratic_discriminant (k : ℝ) :
  (∃ x : ℝ, k*x^2 + 2*x - 1 = 0) ∧ (∀ a b, (a*x + b) ^ 2 = a^2 * x^2 + 2 * a * b * x + b^2) ∧
  (a = k) ∧ (b = 2) ∧ (c = -1) ∧ ((b^2 - 4 * a * c = 0) → (4 + 4 * k = 0)) → k = -1 :=
sorry

end quadratic_discriminant_l313_313034


namespace probability_sum_30_l313_313003

def first_die_faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
def second_die_faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

theorem probability_sum_30 : 
  let valid_pairs := [(11, 19), (12, 18), (13, 17), (14, 16), (15, 15), (16, 14), (17, 13), (18, 12), (19, 11)] in
  let total_outcomes := 20 * 20 in
  let successful_outcomes := valid_pairs.countp (λ (p : ℕ × ℕ), p.1 ∈ first_die_faces ∧ p.2 ∈ second_die_faces) in
  (successful_outcomes : ℚ) / total_outcomes = 9 / 400 :=
by sorry

end probability_sum_30_l313_313003


namespace greatest_x_for_A_is_perfect_square_l313_313031

theorem greatest_x_for_A_is_perfect_square :
  ∃ x : ℕ, x = 2008 ∧ ∀ y : ℕ, (∃ k : ℕ, 2^182 + 4^y + 8^700 = k^2) → y ≤ 2008 :=
by 
  sorry

end greatest_x_for_A_is_perfect_square_l313_313031


namespace rightmost_three_digits_of_7_pow_1987_l313_313753

theorem rightmost_three_digits_of_7_pow_1987 :
  7^1987 % 1000 = 543 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1987_l313_313753


namespace interest_group_selections_l313_313035

-- Define the number of students and the number of interest groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem statement: The total number of different possible selections of interest groups is 81.
theorem interest_group_selections : num_groups ^ num_students = 81 := by
  sorry

end interest_group_selections_l313_313035


namespace find_coefficient_b_l313_313589

variable (a b c p : ℝ)

def parabola (x : ℝ) := a * x^2 + b * x + c

theorem find_coefficient_b (h_vertex : ∀ x, parabola a b c x = a * (x - p)^2 + p)
                           (h_y_intercept : parabola a b c 0 = -3 * p)
                           (hp_nonzero : p ≠ 0) :
  b = 8 / p :=
by
  sorry

end find_coefficient_b_l313_313589


namespace systematic_sampling_method_l313_313459

-- Defining the conditions of the problem as lean definitions
def sampling_interval_is_fixed (interval : ℕ) : Prop :=
  interval = 10

def production_line_uniformly_flowing : Prop :=
  true  -- Assumption

-- The main theorem formulation
theorem systematic_sampling_method :
  ∀ (interval : ℕ), sampling_interval_is_fixed interval → production_line_uniformly_flowing →
  (interval = 10 → true) :=
by {
  sorry
}

end systematic_sampling_method_l313_313459


namespace cylinder_base_radius_l313_313630

theorem cylinder_base_radius (l w : ℝ) (h_l : l = 6) (h_w : w = 4) (h_circ : l = 2 * Real.pi * r ∨ w = 2 * Real.pi * r) : 
    r = 3 / Real.pi ∨ r = 2 / Real.pi := by
  sorry

end cylinder_base_radius_l313_313630


namespace prove_angle_BFD_l313_313056

def given_conditions (A : ℝ) (AFG AGF : ℝ) : Prop :=
  A = 40 ∧ AFG = AGF

theorem prove_angle_BFD (A AFG AGF BFD : ℝ) (h1 : given_conditions A AFG AGF) : BFD = 110 :=
  by
  -- Utilize the conditions h1 stating that A = 40 and AFG = AGF
  sorry

end prove_angle_BFD_l313_313056


namespace min_time_meet_l313_313643

-- Define the speeds in km/hr
def Petya_speed : ℝ := 27
def Vlad_speed : ℝ := 30
def Timur_speed : ℝ := 32

-- Define the length of the bike path in km
def bike_path_length : ℝ := 0.4

-- Define the relative speeds in km/hr
def rel_speed_VP : ℝ := Vlad_speed - Petya_speed
def rel_speed_TV : ℝ := Timur_speed - Vlad_speed
def rel_speed_TP : ℝ := Timur_speed - Petya_speed

-- Define the time intervals in minutes
def time_VP := (bike_path_length / rel_speed_VP) * 60
def time_TV := (bike_path_length / rel_speed_TV) * 60
def time_TP := (bike_path_length / rel_speed_TP) * 60

-- Define the LCM (Least Common Multiple) function for reals
noncomputable def lcm_real (x y : ℝ) : ℝ := (x * y) / (Real.gcd x y)

-- Minimum time when they will all meet again
noncomputable def min_time := lcm_real (lcm_real time_VP time_TV) time_TP

-- The problem statement to be proved
theorem min_time_meet : min_time = 24 := by
  sorry

end min_time_meet_l313_313643


namespace candy_division_l313_313864

theorem candy_division (pieces_of_candy : Nat) (students : Nat) 
  (h1 : pieces_of_candy = 344) (h2 : students = 43) : pieces_of_candy / students = 8 := by
  sorry

end candy_division_l313_313864


namespace fraction_to_decimal_l313_313942

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313942


namespace fraction_to_decimal_equiv_l313_313959

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313959


namespace number_of_workers_l313_313122

-- Definitions for conditions
def initial_contribution (W C : ℕ) : Prop := W * C = 300000
def additional_contribution (W C : ℕ) : Prop := W * (C + 50) = 350000

-- Proof statement
theorem number_of_workers (W C : ℕ) (h1 : initial_contribution W C) (h2 : additional_contribution W C) : W = 1000 :=
by
  sorry

end number_of_workers_l313_313122


namespace teresa_science_marks_l313_313418

-- Definitions for the conditions
def music_marks : ℕ := 80
def social_studies_marks : ℕ := 85
def physics_marks : ℕ := music_marks / 2
def total_marks : ℕ := 275

-- Statement to prove
theorem teresa_science_marks : ∃ S : ℕ, 
  S + music_marks + social_studies_marks + physics_marks = total_marks ∧ S = 70 :=
sorry

end teresa_science_marks_l313_313418


namespace cassandra_collected_pennies_l313_313641

theorem cassandra_collected_pennies 
(C : ℕ) 
(h1 : ∀ J : ℕ,  J = C - 276) 
(h2 : ∀ J : ℕ, C + J = 9724) 
: C = 5000 := 
by
  sorry

end cassandra_collected_pennies_l313_313641


namespace general_term_arithmetic_sequence_l313_313370

theorem general_term_arithmetic_sequence {a : ℕ → ℕ} (d : ℕ) (h_d : d ≠ 0)
  (h1 : a 3 + a 10 = 15)
  (h2 : (a 2 + d) * (a 2 + 10 * d) = (a 2 + 4 * d) * (a 2 + d))
  : ∀ n, a n = n + 1 :=
sorry

end general_term_arithmetic_sequence_l313_313370


namespace sum_ratio_is_nine_l313_313664

open Nat

-- Predicate to define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

axiom a : ℕ → ℝ -- The arithmetic sequence
axiom h_arith : is_arithmetic_sequence a
axiom a5_eq_5a3 : a 4 = 5 * a 2

-- Statement of the problem
theorem sum_ratio_is_nine : S 9 a / S 5 a = 9 :=
sorry

end sum_ratio_is_nine_l313_313664


namespace similar_triangles_side_length_l313_313099

theorem similar_triangles_side_length (A1 A2 : ℕ) (k : ℕ)
  (h1 : A1 - A2 = 32)
  (h2 : A1 = k^2 * A2)
  (h3 : A2 > 0)
  (side2 : ℕ) (h4 : side2 = 5) :
  ∃ side1 : ℕ, side1 = 3 * side2 ∧ side1 = 15 :=
by
  sorry

end similar_triangles_side_length_l313_313099


namespace arithmetic_sequence_general_formula_and_geometric_condition_l313_313043

theorem arithmetic_sequence_general_formula_and_geometric_condition :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ} {k : ℕ}, 
    (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →
    a 1 = 9 →
    S 3 = 21 →
    a 5 * S k = a 8 ^ 2 →
    k = 5 :=
by 
  intros a S k hS ha1 hS3 hgeom
  sorry

end arithmetic_sequence_general_formula_and_geometric_condition_l313_313043


namespace Joey_age_l313_313795

-- Define the basic data
def ages : List ℕ := [4, 6, 8, 10, 12]

-- Define the conditions
def cinema_ages (x y : ℕ) : Prop := x + y = 18
def soccer_ages (x y : ℕ) : Prop := x < 11 ∧ y < 11
def stays_home (x : ℕ) : Prop := x = 6

-- The goal is to prove Joey's age
theorem Joey_age : ∃ j, j ∈ ages ∧ stays_home 6 ∧ (∀ x y, cinema_ages x y → x ≠ j ∧ y ≠ j) ∧ 
(∃ x y, soccer_ages x y ∧ x ≠ 6 ∧ y ≠ 6) ∧ j = 8 := by
  sorry

end Joey_age_l313_313795


namespace percent_of_x_is_z_l313_313361

-- Defining the conditions as constants in the Lean environment
variables (x y z : ℝ)

-- Given conditions
def cond1 : Prop := 0.45 * z = 0.90 * y
def cond2 : Prop := y = 0.75 * x

-- The statement of the problem proving z = 1.5 * x under given conditions
theorem percent_of_x_is_z
  (h1 : cond1 z y)
  (h2 : cond2 y x) :
  z = 1.5 * x :=
sorry

end percent_of_x_is_z_l313_313361


namespace remainder_when_divided_by_9_l313_313116

theorem remainder_when_divided_by_9 (x : ℕ) (h : 4 * x % 9 = 2) : x % 9 = 5 :=
by sorry

end remainder_when_divided_by_9_l313_313116


namespace snail_returns_l313_313292

noncomputable def snail_path : Type := ℕ → ℝ × ℝ

def snail_condition (snail : snail_path) (speed : ℝ) : Prop :=
  ∀ n : ℕ, n % 4 = 0 → snail (n + 4) = snail n

theorem snail_returns (snail : snail_path) (speed : ℝ) (h1 : ∀ n m : ℕ, n ≠ m → snail n ≠ snail m)
    (h2 : snail_condition snail speed) :
  ∃ t : ℕ, t > 0 ∧ t % 4 = 0 ∧ snail t = snail 0 := 
sorry

end snail_returns_l313_313292


namespace odd_function_symmetry_l313_313420

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then x^2 else sorry

theorem odd_function_symmetry (x : ℝ) (k : ℕ) (h1 : ∀ y, f (-y) = -f y)
  (h2 : ∀ y, f y = f (2 - y)) (h3 : ∀ y, 0 < y ∧ y ≤ 1 → f y = y^2) :
  k = 45 / 4 → f k = -9 / 16 :=
by
  intros _
  sorry

end odd_function_symmetry_l313_313420


namespace percentage_of_silver_in_final_solution_l313_313193

noncomputable section -- because we deal with real numbers and division

variable (volume_4pct : ℝ) (percentage_4pct : ℝ)
variable (volume_10pct : ℝ) (percentage_10pct : ℝ)

def final_percentage_silver (v4 : ℝ) (p4 : ℝ) (v10 : ℝ) (p10 : ℝ) : ℝ :=
  let total_silver := v4 * p4 + v10 * p10
  let total_volume := v4 + v10
  (total_silver / total_volume) * 100

theorem percentage_of_silver_in_final_solution :
  final_percentage_silver 5 0.04 2.5 0.10 = 6 := by
  sorry

end percentage_of_silver_in_final_solution_l313_313193


namespace percent_change_is_minus_5_point_5_percent_l313_313636

noncomputable def overall_percent_change (initial_value : ℝ) : ℝ :=
  let day1_value := initial_value * 0.75
  let day2_value := day1_value * 1.4
  let final_value := day2_value * 0.9
  ((final_value / initial_value) - 1) * 100

theorem percent_change_is_minus_5_point_5_percent :
  ∀ (initial_value : ℝ), overall_percent_change initial_value = -5.5 :=
sorry

end percent_change_is_minus_5_point_5_percent_l313_313636


namespace pizza_shared_cost_l313_313147

theorem pizza_shared_cost (total_price : ℕ) (num_people : ℕ) (share: ℕ)
  (h1 : total_price = 40) (h2 : num_people = 5) : share = 8 :=
by
  sorry

end pizza_shared_cost_l313_313147


namespace find_k_l313_313993

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l313_313993


namespace range_of_a_l313_313209

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l313_313209


namespace fraction_to_decimal_l313_313916

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313916


namespace ratio_a_c_l313_313590

-- Define variables and conditions
variables (a b c d : ℚ)

-- Conditions
def ratio_a_b : Prop := a / b = 5 / 4
def ratio_c_d : Prop := c / d = 4 / 3
def ratio_d_b : Prop := d / b = 1 / 5

-- Theorem statement
theorem ratio_a_c (h1 : ratio_a_b a b)
                  (h2 : ratio_c_d c d)
                  (h3 : ratio_d_b d b) : 
  (a / c = 75 / 16) :=
sorry

end ratio_a_c_l313_313590


namespace min_value_inequality_l313_313173

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) : 
  (2 / a + 3 / b) ≥ 14 :=
sorry

end min_value_inequality_l313_313173


namespace fencing_rate_l313_313485

/-- Given a circular field of diameter 20 meters and a total cost of fencing of Rs. 94.24777960769379,
    prove that the rate per meter for the fencing is Rs. 1.5. -/
theorem fencing_rate 
  (d : ℝ) (cost : ℝ) (π : ℝ) (rate : ℝ)
  (hd : d = 20)
  (hcost : cost = 94.24777960769379)
  (hπ : π = 3.14159)
  (Circumference : ℝ := π * d)
  (Rate : ℝ := cost / Circumference) : 
  rate = 1.5 :=
sorry

end fencing_rate_l313_313485


namespace no_integers_divisible_by_all_l313_313997

-- Define the list of divisors
def divisors : List ℕ := [2, 3, 4, 5, 7, 11]

-- Define the LCM function
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Calculate the LCM of the given divisors
def lcm_divisors : ℕ := lcm_list divisors

-- Define a predicate to check divisibility by all divisors
def is_divisible_by_all (n : ℕ) (ds : List ℕ) : Prop :=
  ds.all (λ d => n % d = 0)

-- Define the theorem to prove the number of integers between 1 and 1000 divisible by the given divisors
theorem no_integers_divisible_by_all :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1000 ∧ is_divisible_by_all n divisors) → False := by
  sorry

end no_integers_divisible_by_all_l313_313997


namespace problem1_l313_313844

theorem problem1 (x y : ℝ) (h1 : x + y = 4) (h2 : 2 * x - y = 5) : 
  x = 3 ∧ y = 1 := sorry

end problem1_l313_313844


namespace power_difference_l313_313498

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l313_313498


namespace set_D_not_right_triangle_l313_313605

theorem set_D_not_right_triangle :
  let a := 11
  let b := 12
  let c := 15
  a ^ 2 + b ^ 2 ≠ c ^ 2
:=
by
  let a := 11
  let b := 12
  let c := 15
  sorry

end set_D_not_right_triangle_l313_313605


namespace clarence_initial_oranges_l313_313150

variable (initial_oranges : ℕ)
variable (obtained_from_joyce : ℕ := 3)
variable (total_oranges : ℕ := 8)

theorem clarence_initial_oranges (initial_oranges : ℕ) :
  initial_oranges + obtained_from_joyce = total_oranges → initial_oranges = 5 :=
by
  sorry

end clarence_initial_oranges_l313_313150


namespace expand_polynomial_product_l313_313160

variable (x : ℝ)

def P (x : ℝ) : ℝ := 5 * x ^ 2 + 3 * x - 4
def Q (x : ℝ) : ℝ := 6 * x ^ 3 + 2 * x ^ 2 - x + 7

theorem expand_polynomial_product :
  (P x) * (Q x) = 30 * x ^ 5 + 28 * x ^ 4 - 23 * x ^ 3 + 24 * x ^ 2 + 25 * x - 28 :=
by
  sorry

end expand_polynomial_product_l313_313160


namespace points_on_line_l313_313558

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313558


namespace simplify_expression_l313_313411

-- Define the question and conditions
theorem simplify_expression (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2*x^2*y - 3*x*y) - 2*(x^2*y - x*y + 1/2*x*y^2) + x*y = 4 :=
by
  -- proof steps if needed, but currently replaced with 'sorry' to indicate proof needed
  sorry

end simplify_expression_l313_313411


namespace range_of_a_l313_313211

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l313_313211


namespace servings_per_guest_l313_313469

-- Definitions based on conditions
def num_guests : ℕ := 120
def servings_per_bottle : ℕ := 6
def num_bottles : ℕ := 40

-- Theorem statement
theorem servings_per_guest : (num_bottles * servings_per_bottle) / num_guests = 2 := by
  sorry

end servings_per_guest_l313_313469


namespace beads_pulled_out_l313_313750

theorem beads_pulled_out (white_beads black_beads : ℕ) (frac_black frac_white : ℚ) (h_black : black_beads = 90) (h_white : white_beads = 51) (h_frac_black : frac_black = (1/6)) (h_frac_white : frac_white = (1/3)) : 
  white_beads * frac_white + black_beads * frac_black = 32 := 
by
  sorry

end beads_pulled_out_l313_313750


namespace part_I_part_II_l313_313531

variable (f : ℝ → ℝ)

-- Condition 1: f is an even function
axiom even_function : ∀ x : ℝ, f (-x) = f x

-- Condition 2: f is symmetric about x = 1
axiom symmetric_about_1 : ∀ x : ℝ, f x = f (2 - x)

-- Condition 3: f(x₁ + x₂) = f(x₁) * f(x₂) for x₁, x₂ ∈ [0, 1/2]
axiom multiplicative_on_interval : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 1/2) ∧ (0 ≤ x₂ ∧ x₂ ≤ 1/2) → f (x₁ + x₂) = f x₁ * f x₂

-- Given f(1) = 2
axiom f_one : f 1 = 2

-- Part I: Prove f(1/2) = √2 and f(1/4) = 2^(1/4).
theorem part_I : f (1 / 2) = Real.sqrt 2 ∧ f (1 / 4) = Real.sqrt (Real.sqrt 2) := by
  sorry

-- Part II: Prove that f(x) is a periodic function with period 2.
theorem part_II : ∀ x : ℝ, f x = f (x + 2) := by
  sorry

end part_I_part_II_l313_313531


namespace coins_in_stack_l313_313677

-- Define the thickness of each coin type
def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75

-- Define the total stack height
def total_stack_height : ℝ := 15

-- The statement to prove
theorem coins_in_stack (pennies nickels dimes quarters : ℕ) :
  pennies * penny_thickness + nickels * nickel_thickness + 
  dimes * dime_thickness + quarters * quarter_thickness = total_stack_height →
  pennies + nickels + dimes + quarters = 9 :=
sorry

end coins_in_stack_l313_313677


namespace food_expenditure_increase_l313_313141

-- Conditions
def linear_relationship (x : ℝ) : ℝ := 0.254 * x + 0.321

-- Proof statement
theorem food_expenditure_increase (x : ℝ) : linear_relationship (x + 1) - linear_relationship x = 0.254 :=
by
  sorry

end food_expenditure_increase_l313_313141


namespace count_1320_factors_l313_313354

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l313_313354


namespace methane_hydrate_scientific_notation_l313_313809

theorem methane_hydrate_scientific_notation :
  (9.2 * 10^(-4)) = 0.00092 :=
by sorry

end methane_hydrate_scientific_notation_l313_313809


namespace fraction_to_decimal_l313_313887

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313887


namespace math_problem_l313_313187
-- Import necessary modules

-- Define the condition as a hypothesis and state the theorem
theorem math_problem (x : ℝ) (h : 8 * x - 6 = 10) : 50 * (1 / x) + 150 = 175 :=
sorry

end math_problem_l313_313187


namespace simplify_expression_is_3_l313_313804

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end simplify_expression_is_3_l313_313804


namespace fraction_to_decimal_l313_313900

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313900


namespace geometric_series_sum_l313_313714

noncomputable def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (2/3) (2/3) 10 = 116050 / 59049 :=
by
  sorry

end geometric_series_sum_l313_313714


namespace tom_saves_80_dollars_l313_313264

def normal_doctor_cost : ℝ := 200
def discount_percentage : ℝ := 0.7
def discount_clinic_cost_per_visit : ℝ := normal_doctor_cost * (1 - discount_percentage)
def number_of_visits : ℝ := 2
def total_discount_clinic_cost : ℝ := discount_clinic_cost_per_visit * number_of_visits
def savings : ℝ := normal_doctor_cost - total_discount_clinic_cost

theorem tom_saves_80_dollars : savings = 80 := by
  sorry

end tom_saves_80_dollars_l313_313264


namespace solve_for_a_b_and_extrema_l313_313175

noncomputable def f (a b x : ℝ) := -2 * a * Real.sin (2 * x + (Real.pi / 6)) + 2 * a + b

theorem solve_for_a_b_and_extrema:
  ∃ (a b : ℝ), a > 0 ∧ 
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) ∧ 
  a = 2 ∧ b = -5 ∧
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 4),
    (f a b (Real.pi / 6) = -5 ∨ f a b 0 = -3)) :=
by
  sorry

end solve_for_a_b_and_extrema_l313_313175


namespace problem_solution_l313_313686

noncomputable def omega : ℂ := sorry -- Choose a suitable representative for ω

variables (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
          (hω : ω^3 = 1 ∧ ω ≠ 1)
          (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω)

theorem problem_solution (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
  (hω : ω^3 = 1 ∧ ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 :=
sorry

end problem_solution_l313_313686


namespace arithmetic_sequence_a3_value_l313_313790

theorem arithmetic_sequence_a3_value {a : ℕ → ℕ}
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
sorry

end arithmetic_sequence_a3_value_l313_313790


namespace find_k_l313_313988

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l313_313988


namespace danielle_travel_time_is_30_l313_313304

noncomputable def chase_speed : ℝ := sorry
noncomputable def chase_time : ℝ := 180 -- in minutes
noncomputable def cameron_speed : ℝ := 2 * chase_speed
noncomputable def danielle_speed : ℝ := 3 * cameron_speed
noncomputable def distance : ℝ := chase_speed * chase_time
noncomputable def danielle_time : ℝ := distance / danielle_speed

theorem danielle_travel_time_is_30 :
  danielle_time = 30 :=
sorry

end danielle_travel_time_is_30_l313_313304


namespace initial_points_l313_313556

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l313_313556


namespace shortest_chord_line_intersect_circle_l313_313975

-- Define the equation of the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (0, 1)

-- Define the center of the circle
def center : ℝ × ℝ := (1, 0)

-- Define the equation of the line l
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- The theorem that needs to be proven
theorem shortest_chord_line_intersect_circle :
  ∃ k : ℝ, ∀ x y : ℝ, (circle_eq x y ∧ y = k * x + 1) ↔ line_eq x y :=
by
  sorry

end shortest_chord_line_intersect_circle_l313_313975


namespace rationalize_denominator_l313_313401

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l313_313401


namespace fraction_to_decimal_l313_313941

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313941


namespace x_condition_sufficient_not_necessary_l313_313454

theorem x_condition_sufficient_not_necessary (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (¬ (∀ x, x^2 - 1 > 0 → x < -1)) :=
by
  sorry

end x_condition_sufficient_not_necessary_l313_313454


namespace garden_width_l313_313204

theorem garden_width (w : ℕ) (h1 : ∀ l : ℕ, l = w + 12 → l * w ≥ 120) : w = 6 := 
by
  sorry

end garden_width_l313_313204


namespace bench_cost_150_l313_313853

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end bench_cost_150_l313_313853


namespace necessary_condition_for_inequality_l313_313489

theorem necessary_condition_for_inequality (a b : ℝ) (h : a * b > 0) : 
  (a ≠ b) → (a ≠ 0) → (b ≠ 0) → ((b / a) + (a / b) > 2) :=
by
  sorry

end necessary_condition_for_inequality_l313_313489


namespace fraction_to_decimal_l313_313946

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313946


namespace fraction_to_decimal_l313_313918

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313918


namespace fishing_rod_price_l313_313708

theorem fishing_rod_price (initial_price : ℝ) 
  (price_increase_percentage : ℝ) 
  (price_decrease_percentage : ℝ) 
  (new_price : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 50) 
  (h2 : price_increase_percentage = 0.20) 
  (h3 : price_decrease_percentage = 0.15) 
  (h4 : new_price = initial_price * (1 + price_increase_percentage)) 
  (h5 : final_price = new_price * (1 - price_decrease_percentage)) 
  : final_price = 51 :=
sorry

end fishing_rod_price_l313_313708


namespace points_on_line_initial_l313_313565

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l313_313565


namespace milk_mixture_l313_313052

theorem milk_mixture:
  ∀ (x : ℝ), 0.40 * x + 1.6 = 0.20 * (x + 16) → x = 8 := 
by
  intro x
  sorry

end milk_mixture_l313_313052


namespace washing_machines_removed_correct_l313_313371

-- Define the conditions
def crates : ℕ := 10
def boxes_per_crate : ℕ := 6
def washing_machines_per_box : ℕ := 4
def washing_machines_removed_per_box : ℕ := 1

-- Define the initial and final states
def initial_washing_machines_in_crate : ℕ := boxes_per_crate * washing_machines_per_box
def initial_washing_machines_in_container : ℕ := crates * initial_washing_machines_in_crate

def final_washing_machines_in_box : ℕ := washing_machines_per_box - washing_machines_removed_per_box
def final_washing_machines_in_crate : ℕ := boxes_per_crate * final_washing_machines_in_box
def final_washing_machines_in_container : ℕ := crates * final_washing_machines_in_crate

-- Number of washing machines removed
def washing_machines_removed : ℕ := initial_washing_machines_in_container - final_washing_machines_in_container

-- Theorem statement in Lean 4
theorem washing_machines_removed_correct : washing_machines_removed = 60 := by
  sorry

end washing_machines_removed_correct_l313_313371


namespace find_k_l313_313984

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l313_313984


namespace trigonometric_identity_l313_313649

theorem trigonometric_identity :
  (2 * Real.sin (10 * Real.pi / 180) - Real.cos (20 * Real.pi / 180)) / Real.cos (70 * Real.pi / 180) = - Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l313_313649


namespace probability_of_A_losing_l313_313111

variable (p_win p_draw p_lose : ℝ)

def probability_of_A_winning := p_win = (1/3)
def probability_of_draw := p_draw = (1/2)
def sum_of_probabilities := p_win + p_draw + p_lose = 1

theorem probability_of_A_losing
  (h1: probability_of_A_winning p_win)
  (h2: probability_of_draw p_draw)
  (h3: sum_of_probabilities p_win p_draw p_lose) :
  p_lose = (1/6) :=
sorry

end probability_of_A_losing_l313_313111


namespace number_of_rectangles_with_one_gray_cell_l313_313051

theorem number_of_rectangles_with_one_gray_cell 
    (num_gray_cells : Nat) 
    (num_blue_cells : Nat) 
    (num_red_cells : Nat) 
    (blue_rectangles_per_cell : Nat) 
    (red_rectangles_per_cell : Nat)
    (total_gray_cells_calc : num_gray_cells = 2 * 20)
    (num_gray_cells_definition : num_gray_cells = num_blue_cells + num_red_cells)
    (blue_rect_cond : blue_rectangles_per_cell = 4)
    (red_rect_cond : red_rectangles_per_cell = 8)
    (num_blue_cells_calc : num_blue_cells = 36)
    (num_red_cells_calc : num_red_cells = 4)
  : num_blue_cells * blue_rectangles_per_cell + num_red_cells * red_rectangles_per_cell = 176 := 
  by
  sorry

end number_of_rectangles_with_one_gray_cell_l313_313051


namespace x_fourth_minus_inv_fourth_l313_313496

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l313_313496


namespace smallest_perfect_square_div_l313_313268

theorem smallest_perfect_square_div :
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ 4 ∣ n ∧ 10 ∣ n ∧ 18 ∣ n ∧ n = 900 :=
by
  sorry

end smallest_perfect_square_div_l313_313268


namespace add_points_proof_l313_313567

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l313_313567


namespace num_distinct_factors_1320_l313_313350

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l313_313350


namespace fraction_equality_l313_313479

def op_at (a b : ℕ) : ℕ := a * b + b^2
def op_hash (a b : ℕ) : ℕ := a + b + a * (b^2)

theorem fraction_equality : (op_at 5 3 : ℚ) / (op_hash 5 3 : ℚ) = 24 / 53 := 
by 
  sorry

end fraction_equality_l313_313479


namespace find_a_for_cubic_sum_l313_313032

theorem find_a_for_cubic_sum (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - a * x1 + a + 2 = 0 ∧ 
    x2^2 - a * x2 + a + 2 = 0 ∧
    x1 + x2 = a ∧
    x1 * x2 = a + 2 ∧
    x1^3 + x2^3 = -8) ↔ a = -2 := 
by
  sorry

end find_a_for_cubic_sum_l313_313032


namespace spurs_team_players_l313_313579

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end spurs_team_players_l313_313579


namespace largest_possible_perimeter_l313_313012

noncomputable def max_perimeter_triangle : ℤ :=
  let a : ℤ := 7
  let b : ℤ := 9
  let x : ℤ := 15
  a + b + x

theorem largest_possible_perimeter (x : ℤ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : max_perimeter_triangle = 31 := by
  sorry

end largest_possible_perimeter_l313_313012


namespace number_composite_l313_313695

theorem number_composite : ∃ a1 a2 : ℕ, a1 > 1 ∧ a2 > 1 ∧ 2^17 + 2^5 - 1 = a1 * a2 := 
by
  sorry

end number_composite_l313_313695


namespace fraction_equals_decimal_l313_313933

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313933


namespace tiling_2002_gon_with_rhombuses_l313_313375

theorem tiling_2002_gon_with_rhombuses : ∀ n : ℕ, n = 1001 → (n * (n - 1) / 2) = 500500 :=
by sorry

end tiling_2002_gon_with_rhombuses_l313_313375


namespace series_sum_eq_1_div_400_l313_313022

theorem series_sum_eq_1_div_400 :
  (∑' n : ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 := 
sorry

end series_sum_eq_1_div_400_l313_313022


namespace Anya_loss_games_l313_313312

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l313_313312


namespace fraction_to_decimal_l313_313894

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313894


namespace find_four_numbers_proportion_l313_313759

theorem find_four_numbers_proportion :
  ∃ (a b c d : ℝ), 
  a + d = 14 ∧
  b + c = 11 ∧
  a^2 + b^2 + c^2 + d^2 = 221 ∧
  a * d = b * c ∧
  a = 12 ∧
  b = 8 ∧
  c = 3 ∧
  d = 2 :=
by
  sorry

end find_four_numbers_proportion_l313_313759


namespace inverse_of_A_l313_313163

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := 
  !![3, 4; -2, 9]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/35, -4/35; 2/35, 3/35]

theorem inverse_of_A : A⁻¹ = A_inv :=
by
  sorry

end inverse_of_A_l313_313163


namespace find_b_value_l313_313101

theorem find_b_value (b : ℝ) : (∃ (x y : ℝ), (x, y) = ((2 + 4) / 2, (5 + 9) / 2) ∧ x + y = b) ↔ b = 10 :=
by
  sorry

end find_b_value_l313_313101


namespace age_of_b_l313_313842

variables {a b : ℕ}

theorem age_of_b (h₁ : a + 10 = 2 * (b - 10)) (h₂ : a = b + 11) : b = 41 :=
sorry

end age_of_b_l313_313842


namespace value_of_x_l313_313425

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l313_313425


namespace circle_k_range_l313_313058

theorem circle_k_range {k : ℝ}
  (h : ∀ x y : ℝ, x^2 + y^2 - 2*x + y + k = 0) :
  k < 5 / 4 :=
sorry

end circle_k_range_l313_313058


namespace no_valid_middle_number_l313_313424

theorem no_valid_middle_number
    (x : ℤ)
    (h1 : (x % 2 = 1))
    (h2 : 3 * x + 12 = x^2 + 20) :
    false :=
by
    sorry

end no_valid_middle_number_l313_313424


namespace minimum_red_vertices_l313_313077

theorem minimum_red_vertices (n : ℕ) (h : 0 < n) :
  ∃ R : ℕ, (∀ i j : ℕ, i < n ∧ j < n →
    (i + j) % 2 = 0 → true) ∧
    R = Int.ceil (n^2 / 2 : ℝ) :=
sorry

end minimum_red_vertices_l313_313077


namespace roots_of_polynomial_l313_313758

-- Define the polynomial
def poly := fun (x : ℝ) => x^3 - 7 * x^2 + 14 * x - 8

-- Define the statement
theorem roots_of_polynomial : (poly 1 = 0) ∧ (poly 2 = 0) ∧ (poly 4 = 0) :=
  by
  sorry

end roots_of_polynomial_l313_313758


namespace minimum_value_of_xy_l313_313329

theorem minimum_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x * y ≥ 8 :=
sorry

end minimum_value_of_xy_l313_313329


namespace fraction_to_decimal_l313_313944

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313944


namespace typing_time_together_l313_313393

def meso_typing_rate : ℕ := 3 -- pages per minute
def tyler_typing_rate : ℕ := 5 -- pages per minute
def pages_to_type : ℕ := 40 -- pages

theorem typing_time_together :
  (meso_typing_rate + tyler_typing_rate) * 5 = pages_to_type :=
by
  sorry

end typing_time_together_l313_313393


namespace fraction_b_plus_c_over_a_l313_313712

variable (a b c d : ℝ)

theorem fraction_b_plus_c_over_a :
  (a ≠ 0) →
  (a * 4^3 + b * 4^2 + c * 4 + d = 0) →
  (a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) →
  (b + c) / a = -13 :=
by
  intros h₁ h₂ h₃ 
  sorry

end fraction_b_plus_c_over_a_l313_313712


namespace m_minus_n_eq_six_l313_313185

theorem m_minus_n_eq_six (m n : ℝ) (h : ∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) : m - n = 6 := by
  sorry

end m_minus_n_eq_six_l313_313185


namespace fraction_to_decimal_equiv_l313_313962

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313962


namespace fraction_to_decimal_equiv_l313_313960

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313960


namespace number_of_pairs_of_positive_integers_l313_313666

theorem number_of_pairs_of_positive_integers 
    {m n : ℕ} (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m > n) (h_diff : m^2 - n^2 = 144) : 
    ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧ (∀ p ∈ pairs, p.1 > p.2 ∧ p.1^2 - p.2^2 = 144) :=
sorry

end number_of_pairs_of_positive_integers_l313_313666


namespace integral_sqrt_a_squared_minus_x_squared_l313_313158

open Real

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) :
  (∫ x in -a..a, sqrt (a^2 - x^2)) = 1/2 * π * a^2 :=
by
  sorry

end integral_sqrt_a_squared_minus_x_squared_l313_313158


namespace total_wait_days_l313_313542

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end total_wait_days_l313_313542


namespace value_of_x_l313_313429

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l313_313429


namespace area_EPHQ_l313_313088

theorem area_EPHQ {EFGH : Type} 
  (rectangle_EFGH : EFGH) 
  (length_EF : Real) (width_EG : Real) 
  (P_point : Real) (Q_point : Real) 
  (area_EFGH : Real) 
  (area_EFP : Real) 
  (area_EHQ : Real) : 
  length_EF = 12 → width_EG = 6 → P_point = 4 → Q_point = 3 → 
  area_EFGH = length_EF * width_EG →
  area_EFP = (1 / 2) * width_EG * P_point →
  area_EHQ = (1 / 2) * length_EF * Q_point → 
  (area_EFGH - area_EFP - area_EHQ) = 42 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end area_EPHQ_l313_313088


namespace average_value_of_items_in_loot_box_l313_313796

-- Definitions as per the given conditions
def cost_per_loot_box : ℝ := 5
def total_spent : ℝ := 40
def total_loss : ℝ := 12

-- Proving the average value of items inside each loot box
theorem average_value_of_items_in_loot_box :
  (total_spent - total_loss) / (total_spent / cost_per_loot_box) = 3.50 := by
  sorry

end average_value_of_items_in_loot_box_l313_313796


namespace raffle_tickets_sold_l313_313841

theorem raffle_tickets_sold (total_amount : ℕ) (ticket_cost : ℕ) (tickets_sold : ℕ) 
    (h1 : total_amount = 620) (h2 : ticket_cost = 4) : tickets_sold = 155 :=
by {
  sorry
}

end raffle_tickets_sold_l313_313841


namespace find_constant_a_l313_313592

theorem find_constant_a (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = (1/2) * 3^(n+1) - a) →
  a = 3/2 :=
sorry

end find_constant_a_l313_313592


namespace solve_system_l313_313416

theorem solve_system (X Y : ℝ) : 
  (X + (X + 2 * Y) / (X^2 + Y^2) = 2 ∧ Y + (2 * X - Y) / (X^2 + Y^2) = 0) ↔ (X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1) :=
by
  sorry

end solve_system_l313_313416


namespace find_m_value_l313_313870

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end find_m_value_l313_313870


namespace parallel_vectors_l313_313333

theorem parallel_vectors (m : ℝ) :
  let a : (ℝ × ℝ × ℝ) := (2, -1, 2)
  let b : (ℝ × ℝ × ℝ) := (-4, 2, m)
  (∀ k : ℝ, a = (k * -4, k * 2, k * m)) →
  m = -4 :=
by
  sorry

end parallel_vectors_l313_313333


namespace find_dividend_l313_313369

theorem find_dividend
  (R : ℕ)
  (Q : ℕ)
  (D : ℕ)
  (hR : R = 6)
  (hD_eq_5Q : D = 5 * Q)
  (hD_eq_3R_plus_2 : D = 3 * R + 2) :
  D * Q + R = 86 :=
by
  sorry

end find_dividend_l313_313369


namespace can_capacity_l313_313449

/-- Given a can with a mixture of milk and water in the ratio 4:3, and adding 10 liters of milk
results in the can being full and changes the ratio to 5:2, prove that the capacity of the can is 30 liters. -/
theorem can_capacity (x : ℚ)
  (h1 : 4 * x + 3 * x + 10 = 30)
  (h2 : (4 * x + 10) / (3 * x) = 5 / 2) :
  4 * x + 3 * x + 10 = 30 := 
by sorry

end can_capacity_l313_313449


namespace range_of_a_l313_313324

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0)) ↔ (1 ≤ a ∧ a ≤ 3 ∨ a = -1) :=
by
  sorry

end range_of_a_l313_313324


namespace find_x_l313_313027

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end find_x_l313_313027


namespace initial_average_age_l313_313700

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 9) (h2 : (n * A + 35) / (n + 1) = 17) :
  A = 15 :=
by
  sorry

end initial_average_age_l313_313700


namespace no_very_convex_function_exists_l313_313876

-- Definition of very convex function
def very_convex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

-- Theorem stating the non-existence of very convex functions
theorem no_very_convex_function_exists : ¬∃ f : ℝ → ℝ, very_convex f :=
by {
  sorry
}

end no_very_convex_function_exists_l313_313876


namespace tree_height_equation_l313_313377

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end tree_height_equation_l313_313377


namespace total_payment_correct_l313_313015

def rate_per_kg_grapes := 68
def quantity_grapes := 7
def rate_per_kg_mangoes := 48
def quantity_mangoes := 9

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes

def total_amount_paid := cost_grapes + cost_mangoes

theorem total_payment_correct :
  total_amount_paid = 908 := by
  sorry

end total_payment_correct_l313_313015


namespace volume_of_box_l313_313822

variable (width length height : ℝ)
variable (Volume : ℝ)

-- Given conditions
def w : ℝ := 9
def l : ℝ := 4
def h : ℝ := 7

-- The statement to prove
theorem volume_of_box : Volume = l * w * h := by
  sorry

end volume_of_box_l313_313822


namespace class_A_students_l313_313830

variable (A B : ℕ)

theorem class_A_students 
    (h1 : A = (5 * B) / 7)
    (h2 : A + 3 = (4 * (B - 3)) / 5) :
    A = 45 :=
sorry

end class_A_students_l313_313830


namespace sum_arithmetic_series_eq_250500_l313_313302

theorem sum_arithmetic_series_eq_250500 :
  let a1 := 2
  let d := 2
  let an := 1000
  let n := 500
  (a1 + (n-1) * d = an) →
  ((n * (a1 + an)) / 2 = 250500) :=
by
  sorry

end sum_arithmetic_series_eq_250500_l313_313302


namespace altitudes_order_l313_313421

variable {A a b c h_a h_b h_c : ℝ}

-- Conditions
axiom area_eq : A = (1/2) * a * h_a
axiom area_eq_b : A = (1/2) * b * h_b
axiom area_eq_c : A = (1/2) * c * h_c
axiom sides_order : a > b ∧ b > c

-- Conclusion
theorem altitudes_order : h_a < h_b ∧ h_b < h_c :=
by
  sorry

end altitudes_order_l313_313421


namespace power_calculation_l313_313050

theorem power_calculation (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(3*m + 2*n) = 200 := by
  sorry

end power_calculation_l313_313050


namespace min_value_of_a_l313_313167

theorem min_value_of_a (a : ℝ) (x : ℝ) (h1: 0 < a) (h2: a ≠ 1) (h3: 1 ≤ x → a^x ≥ a * x) : a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l313_313167


namespace fraction_to_decimal_l313_313901

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313901


namespace expected_steps_l313_313082

noncomputable def E (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 5 / 4
  else if n = 3 then 25 / 16
  else 125 / 64

theorem expected_steps :
  let expected := 1 + (E 1 + E 2 + E 3 + E 4) / 4 in
  expected = 625 / 256 :=
by {
  -- Proof goes here
  sorry
}

end expected_steps_l313_313082


namespace quadratic_has_two_roots_l313_313199

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end quadratic_has_two_roots_l313_313199


namespace ellipse_standard_equation_l313_313323

theorem ellipse_standard_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : c / a = 3 / 4) (h3 : b^2 = a^2 - c^2) :
  (x y : ℝ) →
  (x^2 / a^2 + y^2 / b^2 = 1 ∨ x^2 / b^2 + y^2 / a^2 = 1) :=
by
  sorry

end ellipse_standard_equation_l313_313323


namespace matrix_operation_value_l313_313518

theorem matrix_operation_value : 
  let p := 4 
  let q := 5
  let r := 2
  let s := 3 
  (p * s - q * r) = 2 :=
by
  sorry

end matrix_operation_value_l313_313518


namespace game_result_2013_game_result_2014_l313_313018

inductive Player
| Barbara
| Jenna

def winning_player (n : ℕ) : Option Player :=
  if n % 5 = 3 then some Player.Jenna
  else if n % 5 = 4 then some Player.Barbara
  else none

theorem game_result_2013 : winning_player 2013 = some Player.Jenna := 
by sorry

theorem game_result_2014 : (winning_player 2014 = some Player.Barbara) ∨ (winning_player 2014 = some Player.Jenna) :=
by sorry

end game_result_2013_game_result_2014_l313_313018


namespace find_k_l313_313995

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l313_313995


namespace simplify_trig_expression_l313_313094

theorem simplify_trig_expression : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := 
sorry

end simplify_trig_expression_l313_313094


namespace investment_period_more_than_tripling_l313_313468

theorem investment_period_more_than_tripling (r : ℝ) (multiple : ℝ) (n : ℕ) 
  (h_r: r = 0.341) (h_multiple: multiple > 3) :
  (1 + r)^n ≥ multiple → n = 4 :=
by
  sorry

end investment_period_more_than_tripling_l313_313468


namespace total_splash_width_l313_313436

def pebbles : ℚ := 1/5
def rocks : ℚ := 2/5
def boulders : ℚ := 7/5
def mini_boulders : ℚ := 4/5
def large_pebbles : ℚ := 3/5

def num_pebbles : ℚ := 10
def num_rocks : ℚ := 5
def num_boulders : ℚ := 4
def num_mini_boulders : ℚ := 3
def num_large_pebbles : ℚ := 7

theorem total_splash_width : 
  num_pebbles * pebbles + 
  num_rocks * rocks + 
  num_boulders * boulders + 
  num_mini_boulders * mini_boulders + 
  num_large_pebbles * large_pebbles = 16.2 := by
  sorry

end total_splash_width_l313_313436


namespace binary_and_ternary_product_l313_313638

theorem binary_and_ternary_product :
  let binary_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0
  let ternary_1021 := 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  binary_1011 = 11 ∧ ternary_1021 = 34 →
  binary_1011 * ternary_1021 = 374 :=
by
  intros h
  sorry

end binary_and_ternary_product_l313_313638


namespace monotonic_increasing_range_l313_313207

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l313_313207


namespace rectangle_area_l313_313823

theorem rectangle_area (length : ℝ) (width_dm : ℝ) (width_m : ℝ) (h1 : length = 8) (h2 : width_dm = 50) (h3 : width_m = width_dm / 10) : 
  (length * width_m = 40) :=
by {
  sorry
}

end rectangle_area_l313_313823


namespace average_of_pqrs_l313_313362

variable (p q r s : ℝ)

theorem average_of_pqrs
  (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 :=
by
  sorry

end average_of_pqrs_l313_313362


namespace words_lost_equal_137_l313_313198

-- Definitions based on conditions
def letters_in_oz : ℕ := 68
def forbidden_letter_index : ℕ := 7

def words_lost_due_to_forbidden_letter : ℕ :=
  let one_letter_words_lost : ℕ := 1
  let two_letter_words_lost : ℕ := 2 * (letters_in_oz - 1)
  one_letter_words_lost + two_letter_words_lost

-- Theorem stating that the words lost due to prohibition is 137
theorem words_lost_equal_137 :
  words_lost_due_to_forbidden_letter = 137 :=
sorry

end words_lost_equal_137_l313_313198


namespace flat_tyre_problem_l313_313142

theorem flat_tyre_problem
    (x : ℝ)
    (h1 : 0 < x)
    (h2 : 1 / x + 1 / 6 = 1 / 5.6) :
  x = 84 :=
sorry

end flat_tyre_problem_l313_313142


namespace average_speed_monkey_l313_313627

def monkeyDistance : ℝ := 2160
def monkeyTimeMinutes : ℝ := 30
def monkeyTimeSeconds : ℝ := monkeyTimeMinutes * 60

theorem average_speed_monkey :
  (monkeyDistance / monkeyTimeSeconds) = 1.2 := 
sorry

end average_speed_monkey_l313_313627


namespace share_ratio_l313_313729

theorem share_ratio (A B C : ℝ) (x : ℝ) (h1 : A + B + C = 500) (h2 : A = 200) (h3 : A = x * (B + C)) (h4 : B = (6/9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l313_313729


namespace birds_landed_l313_313595

theorem birds_landed (original_birds total_birds : ℕ) (h : original_birds = 12) (h2 : total_birds = 20) :
  total_birds - original_birds = 8 :=
by {
  sorry
}

end birds_landed_l313_313595


namespace points_on_line_l313_313572

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313572


namespace hitting_next_shot_given_first_l313_313073

variables {A B : Prop}
variable (P : Prop → ℚ)

def student_first_shot_probability := P A = 9 / 10
def consecutive_shots_probability := P (A ∧ B) = 1 / 2

theorem hitting_next_shot_given_first 
    (h1 : student_first_shot_probability P)
    (h2 : consecutive_shots_probability P) :
    (P (A ∧ B) / P A) = 5 / 9 :=
by
  sorry

end hitting_next_shot_given_first_l313_313073


namespace pastries_more_than_cakes_l313_313017

def cakes_made : ℕ := 19
def pastries_made : ℕ := 131

theorem pastries_more_than_cakes : pastries_made - cakes_made = 112 :=
by {
  -- Proof will be inserted here
  sorry
}

end pastries_more_than_cakes_l313_313017


namespace find_original_denominator_l313_313008

theorem find_original_denominator (d : ℕ) 
  (h : (10 : ℚ) / (d + 7) = 1 / 3) : 
  d = 23 :=
by 
  sorry

end find_original_denominator_l313_313008


namespace Rebecca_eggs_l313_313696

/-- Rebecca has 6 marbles -/
def M : ℕ := 6

/-- Rebecca has 14 more eggs than marbles -/
def E : ℕ := M + 14

/-- Rebecca has 20 eggs -/
theorem Rebecca_eggs : E = 20 := by
  sorry

end Rebecca_eggs_l313_313696


namespace sphere_volume_ratio_l313_313260

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * (Real.pi) * (r ^ 3)

theorem sphere_volume_ratio (a : ℝ) (h : 0 < a) :
  let r1 := a / 2
  let r2 := (a * Real.sqrt 2) / 2
  let r3 := (a * Real.sqrt 3) / 2
  (volume r1) : (volume r2) : (volume r3) = 1 : 2 * Real.sqrt 2 : 3 * Real.sqrt 3 :=
by
  let r1 := a / 2
  let r2 := (a * Real.sqrt 2) / 2
  let r3 := (a * Real.sqrt 3) / 2
  have V1 : volume r1 = (4 / 3) * Real.pi * (r1^3) := by sorry
  have V2 : volume r2 = (4 / 3) * Real.pi * (r2^3) := by sorry
  have V3 : volume r3 = (4 / 3) * Real.pi * (r3^3) := by sorry
  have ratio : (volume r1) : (volume r2) : (volume r3) =
                (r1^3) : (r2^3) : (r3^3) := by sorry
  have ratio_simplified : (r1^3) : (r2^3) : (r3^3) = 1 : 2 * Real.sqrt 2 : 3 * Real.sqrt 3 := by sorry
  exact ratio_simplified

end sphere_volume_ratio_l313_313260


namespace find_degree_measure_l313_313064

variables {A B C : ℝ}

noncomputable def sin_sq_diff (B C A : ℝ) := sin B * sin B - sin C * sin C - sin A * sin A

theorem find_degree_measure
  (h : sin_sq_diff B C A = real.sqrt 3 * sin A * sin C) :
  B = 5 * real.pi / 6 := sorry

end find_degree_measure_l313_313064


namespace total_canoes_built_l313_313471

def geometric_sum (a r n : ℕ) : ℕ :=
  a * ((r^n - 1) / (r - 1))

theorem total_canoes_built : geometric_sum 10 3 7 = 10930 := 
  by
    -- The proof will go here.
    sorry

end total_canoes_built_l313_313471


namespace wage_difference_l313_313451

theorem wage_difference (P Q H: ℝ) (h1: P = 1.5 * Q) (h2: P * H = 300) (h3: Q * (H + 10) = 300) : P - Q = 5 :=
by
  sorry

end wage_difference_l313_313451


namespace part1_min_value_part2_min_value_l313_313387

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1_min_value :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x : ℝ), f x ≥ m) :=
sorry

theorem part2_min_value (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ (y : ℝ), y = (1 / (a^2 + 1) + 4 / (b^2 + 1)) ∧ y = 9 / 4 :=
sorry

end part1_min_value_part2_min_value_l313_313387


namespace distinct_factors_1320_l313_313344

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l313_313344


namespace fraction_to_decimal_l313_313886

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313886


namespace convert_fraction_to_decimal_l313_313950

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313950


namespace fraction_to_decimal_l313_313945

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313945


namespace yulia_profit_l313_313719

-- Assuming the necessary definitions in the problem
def lemonade_revenue : ℕ := 47
def babysitting_revenue : ℕ := 31
def expenses : ℕ := 34
def profit : ℕ := lemonade_revenue + babysitting_revenue - expenses

-- The proof statement to prove Yulia's profit
theorem yulia_profit : profit = 44 := by
  sorry -- Proof is skipped

end yulia_profit_l313_313719


namespace set_D_cannot_form_triangle_l313_313118

-- Definition for triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given lengths
def length_1 := 1
def length_2 := 2
def length_3 := 3

-- The proof problem statement
theorem set_D_cannot_form_triangle : ¬ triangle_inequality length_1 length_2 length_3 :=
  by sorry

end set_D_cannot_form_triangle_l313_313118


namespace problem_statement_l313_313973

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

noncomputable def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_iter n x)

variable (x : ℝ)

theorem problem_statement
  (h : f_iter 13 x = f_iter 31 x) :
  f_iter 16 x = (x - 1) / x :=
by
  sorry

end problem_statement_l313_313973


namespace rationalize_denominator_l313_313408

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l313_313408


namespace fraction_value_l313_313303

theorem fraction_value :
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20) = -1 :=
by
  -- simplified proof omitted
  sorry

end fraction_value_l313_313303


namespace ellipse_parabola_common_point_l313_313980

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 := 
by 
  sorry

end ellipse_parabola_common_point_l313_313980


namespace points_on_line_l313_313571

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313571


namespace four_digit_positive_integers_count_l313_313184

theorem four_digit_positive_integers_count :
  let p := 17
  let a := 4582 % p
  let b := 902 % p
  let c := 2345 % p
  ∃ (n : ℕ), 
    (1000 ≤ 14 + p * n ∧ 14 + p * n ≤ 9999) ∧ 
    (4582 * (14 + p * n) + 902 ≡ 2345 [MOD p]) ∧ 
    n = 530 := sorry

end four_digit_positive_integers_count_l313_313184


namespace rationalize_denominator_l313_313404

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l313_313404


namespace length_minus_width_l313_313862

theorem length_minus_width 
  (area length diff width : ℝ)
  (h_area : area = 171)
  (h_length : length = 19.13)
  (h_diff : diff = length - width)
  (h_area_eq : area = length * width) :
  diff = 10.19 := 
by {
  sorry
}

end length_minus_width_l313_313862


namespace tileable_by_hook_l313_313290

theorem tileable_by_hook (m n : ℕ) : 
  (∃ a b : ℕ, m = 3 * a ∧ (n = 4 * b ∨ n = 12 * b) ∨ 
              n = 3 * a ∧ (m = 4 * b ∨ m = 12 * b)) ↔ 12 ∣ (m * n) :=
by
  sorry

end tileable_by_hook_l313_313290


namespace fraction_equals_decimal_l313_313930

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313930


namespace solve_x4_minus_inv_x4_l313_313503

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l313_313503


namespace problem1_problem2_problem3_problem4_l313_313280

-- Problem 1
theorem problem1 (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) (a : ℕ) (ha : Nat.coprime a p) :
  a ^ ((p - 1) / 2) ≡ 1 [MOD p] ∨ a ^ ((p - 1) / 2) ≡ p - 1 [MOD p] := sorry

-- Problem 2
theorem problem2 (p : ℕ) (hp : Nat.Prime p) (a : ℕ) :
  (∃ b : ℕ, b^2 ≡ a [MOD p]) ↔ a ^ ((p - 1) / 2) ≡ 1 [MOD p] := sorry

-- Problem 3
theorem problem3 (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) :
  (∃ b : ℕ, b^2 ≡ p - 1 [MOD p]) ↔ p % 4 = 1 := sorry

-- Problem 4 
theorem problem4 (n : ℕ) (a b : ℕ) :
  11 ^ n = a ^ 2 + b ^ 2 ↔ 
  (n = 0 ∧ (a = 1 ∧ b = 0 ∨ a = 0 ∧ b = 1)) ∨ 
  (∃ k : ℕ, n = 2 * k ∧ (a = 11 ^ k ∧ b = 0 ∨ a = 0 ∧ b = 11 ^ k)) := sorry

end problem1_problem2_problem3_problem4_l313_313280


namespace area_of_backyard_eq_400_l313_313522

-- Define the conditions
def length_condition (l : ℕ) : Prop := 25 * l = 1000
def perimeter_condition (l w : ℕ) : Prop := 20 * (l + w) = 1000

-- State the theorem
theorem area_of_backyard_eq_400 (l w : ℕ) (h_length : length_condition l) (h_perimeter : perimeter_condition l w) : l * w = 400 :=
  sorry

end area_of_backyard_eq_400_l313_313522


namespace complement_of_M_l313_313768

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {a | a ^ 2 - 2 * a > 0}
noncomputable def C_U_M : Set ℝ := U \ M

theorem complement_of_M :
  C_U_M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end complement_of_M_l313_313768


namespace min_value_PA_minus_PF_l313_313044

noncomputable def ellipse_condition : Prop :=
  ∃ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)

noncomputable def focal_property (x y : ℝ) (P : ℝ × ℝ) : Prop :=
  dist P (2, 4) - dist P (1, 0) = 1

theorem min_value_PA_minus_PF :
  ∀ (P : ℝ × ℝ), 
    (∃ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) 
    → ∃ (a b : ℝ), a = 2 ∧ b = 4 ∧ focal_property x y P :=
  sorry

end min_value_PA_minus_PF_l313_313044


namespace simplify_expression_l313_313803

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end simplify_expression_l313_313803


namespace gcd_78_36_l313_313653

theorem gcd_78_36 : Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_78_36_l313_313653


namespace calculate_percentage_passed_l313_313789

theorem calculate_percentage_passed (F_H F_E F_HE : ℝ) (h1 : F_H = 0.32) (h2 : F_E = 0.56) (h3 : F_HE = 0.12) :
  1 - (F_H + F_E - F_HE) = 0.24 := by
  sorry

end calculate_percentage_passed_l313_313789


namespace div_by_7_or_11_l313_313814

theorem div_by_7_or_11 (z x y : ℕ) (hx : x < 1000) (hz : z = 1000 * y + x) (hdiv7 : (x - y) % 7 = 0 ∨ (x - y) % 11 = 0) :
  z % 7 = 0 ∨ z % 11 = 0 :=
by
  sorry

end div_by_7_or_11_l313_313814


namespace bob_second_third_lap_time_l313_313472

theorem bob_second_third_lap_time :
  ∀ (lap_length : ℕ) (first_lap_time : ℕ) (average_speed : ℕ),
  lap_length = 400 →
  first_lap_time = 70 →
  average_speed = 5 →
  ∃ (second_third_lap_time : ℕ), second_third_lap_time = 85 :=
by
  intros lap_length first_lap_time average_speed lap_length_eq first_lap_time_eq average_speed_eq
  sorry

end bob_second_third_lap_time_l313_313472


namespace zoe_earns_per_candy_bar_l313_313606

-- Given conditions
def cost_of_trip : ℝ := 485
def grandma_contribution : ℝ := 250
def candy_bars_to_sell : ℝ := 188

-- Derived condition
def additional_amount_needed : ℝ := cost_of_trip - grandma_contribution

-- Assertion to prove
theorem zoe_earns_per_candy_bar :
  (additional_amount_needed / candy_bars_to_sell) = 1.25 :=
by
  sorry

end zoe_earns_per_candy_bar_l313_313606


namespace solution_set_of_inequality_l313_313103

theorem solution_set_of_inequality (x : ℝ) : 
  (1 / x ≤ 1 ↔ (0 < x ∧ x < 1) ∨ (1 ≤ x)) :=
  sorry

end solution_set_of_inequality_l313_313103


namespace usual_time_is_36_l313_313452

-- Definition: let S be the usual speed of the worker (not directly relevant to the final proof)
noncomputable def S : ℝ := sorry

-- Definition: let T be the usual time taken by the worker
noncomputable def T : ℝ := sorry

-- Condition: The worker's speed is (3/4) of her normal speed, resulting in a time (T + 12)
axiom speed_delay_condition : (3 / 4) * S * (T + 12) = S * T

-- Theorem: Prove that the usual time T taken to cover the distance is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  -- Formally stating our proof based on given conditions
  sorry

end usual_time_is_36_l313_313452


namespace find_a_and_monotonicity_l313_313532

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 5)^2 + 6 * real.log x

theorem find_a_and_monotonicity (a : ℝ) :
  let f := λ (x : ℝ), a * (x - 5)^2 + 6 * real.log x,
      f' := λ (x : ℝ), 2 * a * (x - 5) + 6 / x in
  (f 1 = a * (1 - 5)^2 + 6 * real.log 1) ∧
  ((tangent_eq : (6 - 8 * a) * 1 + (16 * a - 6) = 0) → a = 1 / 2) ∧
  (∀ x, 0 < x → 
    (f'(x) = 2 * a * (x - 5) + 6 / x) ∧ 
    ((0 < x ∧ x < 2) → f'(x) > 0) ∧ 
    ((2 < x ∧ x < 3) → f'(x) < 0) ∧ 
    (x > 3 → f'(x) > 0)) :=
by
  sorry

end find_a_and_monotonicity_l313_313532


namespace heartsuit_3_8_l313_313668

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_3_8_l313_313668


namespace fraction_to_decimal_l313_313940

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313940


namespace fraction_to_decimal_l313_313938

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313938


namespace towel_area_decrease_l313_313742

theorem towel_area_decrease (L B : ℝ) :
  let A_original := L * B
  let L_new := 0.8 * L
  let B_new := 0.9 * B
  let A_new := L_new * B_new
  let percentage_decrease := ((A_original - A_new) / A_original) * 100
  percentage_decrease = 28 := 
by
  sorry

end towel_area_decrease_l313_313742


namespace distinct_factors_1320_l313_313349

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l313_313349


namespace compare_neg_rational_numbers_l313_313877

theorem compare_neg_rational_numbers :
  - (3 / 2) > - (5 / 3) := 
sorry

end compare_neg_rational_numbers_l313_313877


namespace square_roots_of_16_l313_313104

theorem square_roots_of_16 :
  {y : ℤ | y^2 = 16} = {4, -4} :=
by
  sorry

end square_roots_of_16_l313_313104


namespace positive_integer_not_in_S_l313_313237

noncomputable def S : Set ℤ :=
  {n | ∃ (i : ℕ), n = 4^i * 3 ∨ n = -4^i * 2}

theorem positive_integer_not_in_S (n : ℤ) (hn : 0 < n) (hnS : n ∉ S) :
  ∃ (x y : ℤ), x ≠ y ∧ x ∈ S ∧ y ∈ S ∧ x + y = n :=
sorry

end positive_integer_not_in_S_l313_313237


namespace hyperbola_ellipse_b_value_l313_313039

theorem hyperbola_ellipse_b_value (a c b : ℝ) (h1 : c = 5 * a / 4) (h2 : c^2 - a^2 = (9 * a^2) / 16) (h3 : 4 * (b^2 - 4) = 16 * b^2 / 25) :
  b = 6 / 5 ∨ b = 10 / 3 :=
by
  sorry

end hyperbola_ellipse_b_value_l313_313039


namespace cost_of_bench_l313_313851

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end cost_of_bench_l313_313851


namespace number_of_ordered_pairs_l313_313644

theorem number_of_ordered_pairs {x y: ℕ} (h1 : x < y) (h2 : 2 * x * y / (x + y) = 4^30) : 
  ∃ n, n = 61 :=
sorry

end number_of_ordered_pairs_l313_313644


namespace members_who_didnt_show_up_l313_313293

theorem members_who_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : total_members = 5) (h2 : points_per_member = 6) (h3 : total_points = 18) : 
  total_members - total_points / points_per_member = 2 :=
by
  sorry

end members_who_didnt_show_up_l313_313293


namespace find_number_added_l313_313488

theorem find_number_added (x n : ℕ) (h : (x + x + 2 + x + 4 + x + n + x + 22) / 5 = x + 7) : n = 7 :=
by
  sorry

end find_number_added_l313_313488


namespace freshman_class_total_students_l313_313014

theorem freshman_class_total_students (N : ℕ) 
    (h1 : 90 ≤ N) 
    (h2 : 100 ≤ N)
    (h3 : 20 ≤ N) 
    (h4: (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N):
    N = 450 :=
  sorry

end freshman_class_total_students_l313_313014


namespace abs_inequality_solution_l313_313414

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end abs_inequality_solution_l313_313414


namespace sum_of_digits_of_m_eq_nine_l313_313550

theorem sum_of_digits_of_m_eq_nine
  (m : ℕ)
  (h1 : m * 3 / 2 - 72 = m) :
  1 + (m / 10 % 10) + (m % 10) = 9 :=
by
  sorry

end sum_of_digits_of_m_eq_nine_l313_313550


namespace find_k_l313_313989

theorem find_k : 
  let a : ℝ × ℝ := (3, 1)
      b : ℝ × ℝ := (1, 0)
      c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
  in a.1 * c k a b .1 + a.2 * c k a b .2 = 0 → k = -10 / 3 :=
by
  intros a b c h
  let k : ℝ := -10/3
  exact sorry -- Proof is omitted.

end find_k_l313_313989


namespace fraction_to_decimal_l313_313926

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313926


namespace anna_cannot_afford_tour_l313_313748

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_cost (C0 : ℝ) (i : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  C0 * (1 + i / n) ^ (n * t)

theorem anna_cannot_afford_tour :
  let P := 40000
  let r := 0.05
  let n := 1
  let t := 3
  let C0 := 45000
  let i := 0.05
  future_value P r n t < future_cost C0 i n t :=
  by
    let P := 40000
    let r := 0.05
    let n := 1
    let t := 3
    let C0 := 45000
    let i := 0.05
    have fv := future_value P r n t
    have fc := future_cost C0 i n t
    show fv < fc from sorry

end anna_cannot_afford_tour_l313_313748


namespace train_or_plane_not_ship_possible_modes_l313_313622

-- Define the probabilities of different modes of transportation
def P_train : ℝ := 0.3
def P_ship : ℝ := 0.2
def P_car : ℝ := 0.1
def P_plane : ℝ := 0.4

-- 1. Proof that probability of train or plane is 0.7
theorem train_or_plane : P_train + P_plane = 0.7 :=
by sorry

-- 2. Proof that probability of not taking a ship is 0.8
theorem not_ship : 1 - P_ship = 0.8 :=
by sorry

-- 3. Proof that if probability is 0.5, the modes are either (ship, train) or (car, plane)
theorem possible_modes (P_value : ℝ) (h1 : P_value = 0.5) :
  (P_ship + P_train = P_value) ∨ (P_car + P_plane = P_value) :=
by sorry

end train_or_plane_not_ship_possible_modes_l313_313622


namespace stratified_sampling_red_balls_l313_313711

-- Define the conditions
def total_balls : ℕ := 1000
def red_balls : ℕ := 50
def sampled_balls : ℕ := 100

-- Prove that the number of red balls sampled using stratified sampling is 5
theorem stratified_sampling_red_balls :
  (red_balls : ℝ) / (total_balls : ℝ) * (sampled_balls : ℝ) = 5 := 
by
  sorry

end stratified_sampling_red_balls_l313_313711


namespace angle_same_terminal_side_l313_313825

theorem angle_same_terminal_side (α : ℝ) : 
  (∃ k : ℤ, α = k * 360 - 100) ↔ (∃ k : ℤ, α = k * 360 + (-100)) :=
sorry

end angle_same_terminal_side_l313_313825


namespace sum_of_numbers_eq_l313_313705

theorem sum_of_numbers_eq (a b : ℕ) (h1 : a = 64) (h2 : b = 32) (h3 : a = 2 * b) : a + b = 96 := 
by 
  sorry

end sum_of_numbers_eq_l313_313705


namespace possible_values_of_x_l313_313041

theorem possible_values_of_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) 
    (h1 : x + 1 / z = 15) (h2 : z + 1 / x = 9 / 20) :
    x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by
  sorry

end possible_values_of_x_l313_313041


namespace total_waiting_days_l313_313540

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end total_waiting_days_l313_313540


namespace bench_cost_150_l313_313852

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end bench_cost_150_l313_313852


namespace negation_example_l313_313253

theorem negation_example :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - x₀ > 0) :=
by
  sorry

end negation_example_l313_313253


namespace probability_other_member_is_girl_l313_313284

theorem probability_other_member_is_girl
  (total_members : fin 12)
  (girls : fin 7)
  (boys : fin 5)
  (two_chosen : fin 2)
  (at_least_one_boy : (two_chosen → boys → Prop)) :
  (Probability (λ two_chosen, (∃ b : fin 2, boys b) → (∃ g : fin 2, girls g))) = 7 / 9 :=
by
  sorry

end probability_other_member_is_girl_l313_313284


namespace polygon_sides_l313_313135

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_l313_313135


namespace cost_price_of_toy_l313_313721

theorem cost_price_of_toy (x : ℝ) (selling_price_per_toy : ℝ) (gain : ℝ) 
  (sale_price : ℝ) (number_of_toys : ℕ) (selling_total : ℝ) (gain_condition : ℝ) :
  (selling_total = number_of_toys * selling_price_per_toy) →
  (selling_price_per_toy = x + gain) →
  (gain = gain_condition / number_of_toys) → 
  (gain_condition = 3 * x) →
  selling_total = 25200 → number_of_toys = 18 → x = 1200 :=
by
  sorry

end cost_price_of_toy_l313_313721


namespace points_on_line_initial_l313_313564

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l313_313564


namespace add_points_proof_l313_313566

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l313_313566


namespace monotonically_increasing_range_l313_313223

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l313_313223


namespace karen_locks_l313_313798

theorem karen_locks : 
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  Lboth = 60 :=
by
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  sorry

end karen_locks_l313_313798


namespace fraction_to_decimal_l313_313922

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313922


namespace max_value_of_d_l313_313530

-- Define the conditions
variable (a b c d : ℝ) (h_sum : a + b + c + d = 10) 
          (h_prod_sum : ab + ac + ad + bc + bd + cd = 20)

-- Define the theorem statement
theorem max_value_of_d : 
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_d_l313_313530


namespace fraction_equals_decimal_l313_313936

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313936


namespace quiz_competition_top_three_orders_l313_313143

theorem quiz_competition_top_three_orders :
  let participants := 4
  let top_positions := 3
  let permutations := (Nat.factorial participants) / (Nat.factorial (participants - top_positions))
  permutations = 24 := 
by
  sorry

end quiz_competition_top_three_orders_l313_313143


namespace cube_side_length_l313_313295

theorem cube_side_length (n : ℕ) (h1 : 6 * (n^2) = 1/3 * 6 * (n^3)) : n = 3 := 
sorry

end cube_side_length_l313_313295


namespace quadratic_inequality_solution_l313_313826

theorem quadratic_inequality_solution :
  {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x * (x + 2) < 3} :=
by
  sorry

end quadratic_inequality_solution_l313_313826


namespace rationalize_denominator_l313_313400

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l313_313400


namespace smallest_number_first_digit_is_9_l313_313679

def sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def first_digit (n : Nat) : Nat :=
  n.digits 10 |>.headD 0

theorem smallest_number_first_digit_is_9 :
  ∃ N : Nat, sum_of_digits N = 2020 ∧ ∀ M : Nat, (sum_of_digits M = 2020 → N ≤ M) ∧ first_digit N = 9 :=
by
  sorry

end smallest_number_first_digit_is_9_l313_313679


namespace units_digit_quotient_eq_one_l313_313874

theorem units_digit_quotient_eq_one :
  (2^2023 + 3^2023) / 5 % 10 = 1 := by
  sorry

end units_digit_quotient_eq_one_l313_313874


namespace min_value_fraction_l313_313491

theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2: b > 0) (h3 : a + b = 1) : 
  ∃ c : ℝ, c = 3 + 2 * Real.sqrt 2 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 1) → x + 2 * y ≥ c) :=
by
  sorry

end min_value_fraction_l313_313491


namespace nina_homework_total_l313_313810

-- Definitions based on conditions
def ruby_math_homework : Nat := 6
def ruby_reading_homework : Nat := 2
def nina_math_homework : Nat := 4 * ruby_math_homework
def nina_reading_homework : Nat := 8 * ruby_reading_homework
def nina_total_homework : Nat := nina_math_homework + nina_reading_homework

-- The theorem to prove
theorem nina_homework_total : nina_total_homework = 40 := by
  sorry

end nina_homework_total_l313_313810


namespace milk_production_l313_313081

theorem milk_production (a b c x y z w : ℕ) : 
  ((b:ℝ) / c) * w + ((y:ℝ) / z) * w = (bw / c) + (yw / z) := sorry

end milk_production_l313_313081


namespace maximize_probability_l313_313620

theorem maximize_probability (p1 p2 p3 : ℝ) (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3) :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3) in
  PC > PA ∧ PC > PB :=
by
  sorry

end maximize_probability_l313_313620


namespace min_value_frac_l313_313536

theorem min_value_frac (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : a + b = 1) : 
  (1 / a) + (4 / b) ≥ 9 :=
by sorry

end min_value_frac_l313_313536


namespace fraction_to_decimal_l313_313923

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313923


namespace prove_monotonic_increasing_range_l313_313219

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l313_313219


namespace expand_simplify_correct_l313_313159

noncomputable def expand_and_simplify (x : ℕ) : ℕ :=
  (x + 4) * (x - 9)

theorem expand_simplify_correct (x : ℕ) : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by
  sorry

end expand_simplify_correct_l313_313159


namespace quaternion_problem_solution_l313_313655

open Quaternion

noncomputable theory

def find_q {a b c d : ℝ} (q : ℍ) : Prop :=
  q = a + b * i + c * j + d * k ∧
  q^2 = (-1 - i - j - k) 

theorem quaternion_problem_solution :
  find_q (-1 - 1/2 * i - 1/2 * j - 1/2 * k) 
:= by
  sorry

end quaternion_problem_solution_l313_313655


namespace distinct_four_digit_numbers_count_l313_313336

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l313_313336


namespace alice_no_guarantee_win_when_N_is_18_l313_313467

noncomputable def alice_cannot_guarantee_win : Prop :=
  ∀ (B : ℝ × ℝ) (P : ℕ → ℝ × ℝ),
    (∀ k, 0 ≤ k → k ≤ 18 → 
         dist (P (k + 1)) B < dist (P k) B ∨ dist (P (k + 1)) B ≥ dist (P k) B) →
    ∀ A : ℝ × ℝ, dist A B > 1 / 2020

theorem alice_no_guarantee_win_when_N_is_18 : alice_cannot_guarantee_win :=
sorry

end alice_no_guarantee_win_when_N_is_18_l313_313467


namespace brick_length_is_20_cm_l313_313129

theorem brick_length_is_20_cm
    (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
    (brick_length_cm : ℕ) (brick_width_cm : ℕ)
    (total_bricks_required : ℕ)
    (h1 : courtyard_length_m = 25)
    (h2 : courtyard_width_m = 16)
    (h3 : brick_length_cm = 20)
    (h4 : brick_width_cm = 10)
    (h5 : total_bricks_required = 20000) :
    brick_length_cm = 20 := 
by
    sorry

end brick_length_is_20_cm_l313_313129


namespace area_comparison_perimeter_comparison_l313_313662

-- Define side length of square and transformation to sides of the rectangle
variable (a : ℝ)

-- Conditions: side lengths of the rectangle relative to the square
def long_side : ℝ := 1.11 * a
def short_side : ℝ := 0.9 * a

-- Area calculations and comparison
def square_area : ℝ := a^2
def rectangle_area : ℝ := long_side a * short_side a

theorem area_comparison : (rectangle_area a / square_area a) = 0.999 := by
  sorry

-- Perimeter calculations and comparison
def square_perimeter : ℝ := 4 * a
def rectangle_perimeter : ℝ := 2 * (long_side a + short_side a)

theorem perimeter_comparison : (rectangle_perimeter a / square_perimeter a) = 1.005 := by
  sorry

end area_comparison_perimeter_comparison_l313_313662


namespace quadratic_has_two_roots_l313_313202

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end quadratic_has_two_roots_l313_313202


namespace cos_expression_range_l313_313330

theorem cos_expression_range (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi) :
  -25 / 16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end cos_expression_range_l313_313330


namespace evaluate_expression_l313_313157

theorem evaluate_expression : 
  (2 ^ 2003 * 3 ^ 2002 * 5) / (6 ^ 2003) = (5 / 3) :=
by sorry

end evaluate_expression_l313_313157


namespace derivative_at_1_l313_313776

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 * x + (1 / 2) * x^2

theorem derivative_at_1 : deriv f 1 = Real.exp 1 := 
by 
  sorry

end derivative_at_1_l313_313776


namespace two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l313_313397

theorem two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n
  (n : ℕ) (h : 2 < n) : (2 * n - 1) ^ n + (2 * n) ^ n < (2 * n + 1) ^ n :=
sorry

end two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l313_313397


namespace bags_of_oranges_l313_313108

-- Define the total number of oranges in terms of bags B
def totalOranges (B : ℕ) : ℕ := 30 * B

-- Define the number of usable oranges left after considering rotten oranges
def usableOranges (B : ℕ) : ℕ := totalOranges B - 50

-- Define the oranges to be sold after keeping some for juice
def orangesToBeSold (B : ℕ) : ℕ := usableOranges B - 30

-- The theorem to state that given 220 oranges will be sold,
-- we need to find B, the number of bags of oranges
theorem bags_of_oranges (B : ℕ) : orangesToBeSold B = 220 → B = 10 :=
by
  sorry

end bags_of_oranges_l313_313108


namespace problem_statement_l313_313388

open Polynomial

noncomputable def q : Polynomial ℤ := ∑ i in (finset.range 2011), (x : Polynomial ℤ) ^ i

noncomputable def divisor : Polynomial ℤ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + x + 1

noncomputable def s : Polynomial ℤ := q % divisor

theorem problem_statement :
  |eval 2010 s| % 1000 = 111 :=
sorry

end problem_statement_l313_313388


namespace tangent_line_at_point_l313_313584

theorem tangent_line_at_point
  (x y : ℝ)
  (h_curve : y = x^3 - 3 * x^2 + 1)
  (h_point : (x, y) = (1, -1)) :
  ∃ m b : ℝ, (m = -3) ∧ (b = 2) ∧ (y = m * x + b) :=
sorry

end tangent_line_at_point_l313_313584


namespace base8_subtraction_l313_313965

def subtract_base_8 (a b : Nat) : Nat :=
  sorry  -- This is a placeholder for the actual implementation.

theorem base8_subtraction :
  subtract_base_8 0o5374 0o2645 = 0o1527 :=
by
  sorry

end base8_subtraction_l313_313965


namespace savings_equal_in_820_weeks_l313_313089

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end savings_equal_in_820_weeks_l313_313089


namespace fraction_to_decimal_l313_313943

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313943


namespace find_x_l313_313834

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : 
  (∀ a b c d : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 :=
sorry

end find_x_l313_313834


namespace axis_of_symmetry_condition_l313_313252

theorem axis_of_symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
    (h_sym : ∀ x y, y = -x → y = (p * x + q) / (r * x + s)) : p = s :=
by
  sorry

end axis_of_symmetry_condition_l313_313252


namespace sum_of_integers_l313_313247

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 :=
by
  sorry

end sum_of_integers_l313_313247


namespace share_ratio_l313_313553

theorem share_ratio (A B C : ℕ) (hA : A = (2 * B) / 3) (hA_val : A = 372) (hB_val : B = 93) (hC_val : C = 62) : B / C = 3 / 2 := 
by 
  sorry

end share_ratio_l313_313553


namespace area_of_shaded_region_l313_313331

theorem area_of_shaded_region
  (r_large : ℝ) (r_small : ℝ) (n_small : ℕ) (π : ℝ)
  (A_large : ℝ) (A_small : ℝ) (A_7_small : ℝ) (A_shaded : ℝ)
  (h1 : r_large = 20)
  (h2 : r_small = 10)
  (h3 : n_small = 7)
  (h4 : π = 3.14)
  (h5 : A_large = π * r_large^2)
  (h6 : A_small = π * r_small^2)
  (h7 : A_7_small = n_small * A_small)
  (h8 : A_shaded = A_large - A_7_small) :
  A_shaded = 942 :=
by
  sorry

end area_of_shaded_region_l313_313331


namespace remainder_is_zero_l313_313648

noncomputable def polynomial_division_theorem : Prop :=
  ∀ (x : ℤ), (x^3 ≡ 1 [MOD (x^2 + x + 1)]) → 
             (x^5 ≡ x^2 [MOD (x^2 + x + 1)]) →
             (x^2 - 1) * (x^3 - 1) ≡ 0 [MOD (x^2 + x + 1)]

theorem remainder_is_zero : polynomial_division_theorem := by
  sorry

end remainder_is_zero_l313_313648


namespace profit_percentage_is_60_l313_313868

variable (SellingPrice CostPrice : ℝ)

noncomputable def Profit : ℝ := SellingPrice - CostPrice

noncomputable def ProfitPercentage : ℝ := (Profit SellingPrice CostPrice / CostPrice) * 100

theorem profit_percentage_is_60
  (h1 : SellingPrice = 400)
  (h2 : CostPrice = 250) :
  ProfitPercentage SellingPrice CostPrice = 60 := by
  sorry

end profit_percentage_is_60_l313_313868


namespace root_interval_l313_313509

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem root_interval (x0 : ℝ) (h : f x0 = 0): x0 ∈ Set.Ioo (1 / 4 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end root_interval_l313_313509


namespace cost_price_of_item_l313_313735

theorem cost_price_of_item 
  (retail_price : ℝ) (reduction_percentage : ℝ) 
  (additional_discount : ℝ) (profit_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : retail_price = 900)
  (h2 : reduction_percentage = 0.1)
  (h3 : additional_discount = 48)
  (h4 : profit_percentage = 0.2)
  (h5 : selling_price = 762) :
  ∃ x : ℝ, selling_price = 1.2 * x ∧ x = 635 := 
by {
  sorry
}

end cost_price_of_item_l313_313735


namespace circle_radius_eq_l313_313788

theorem circle_radius_eq (r : ℝ) (AB : ℝ) (BC : ℝ) (hAB : AB = 10) (hBC : BC = 12) : r = 25 / 4 := by
  sorry

end circle_radius_eq_l313_313788


namespace average_age_l313_313195

theorem average_age (women men : ℕ) (avg_age_women avg_age_men : ℝ) 
  (h_women : women = 12) 
  (h_men : men = 18) 
  (h_avg_women : avg_age_women = 28) 
  (h_avg_men : avg_age_men = 40) : 
  (12 * 28 + 18 * 40) / (12 + 18) = 35.2 :=
by {
  sorry
}

end average_age_l313_313195


namespace simplify_expression_is_3_l313_313805

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end simplify_expression_is_3_l313_313805


namespace problem_l313_313516

noncomputable def y := 2 + Real.sqrt 3

theorem problem (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : y = c + Real.sqrt d)
  (hy_eq : y^2 + 2*y + 2/y + 1/y^2 = 20) : c + d = 5 :=
  sorry

end problem_l313_313516


namespace swimming_speed_solution_l313_313861

-- Definition of the conditions
def speed_of_water : ℝ := 2
def distance_against_current : ℝ := 10
def time_against_current : ℝ := 5

-- Definition of the person's swimming speed in still water
def swimming_speed_in_still_water (v : ℝ) :=
  distance_against_current = (v - speed_of_water) * time_against_current

-- Main theorem we want to prove
theorem swimming_speed_solution : 
  ∃ v : ℝ, swimming_speed_in_still_water v ∧ v = 4 :=
by
  sorry

end swimming_speed_solution_l313_313861


namespace monotonic_increasing_on_interval_l313_313178

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem monotonic_increasing_on_interval (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / (2 * ω) = 4 * Real.pi) :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 2) Real.pi) → (y ∈ Set.Icc (Real.pi / 2) Real.pi) → x ≤ y → f ω x ≤ f ω y := 
by
  sorry

end monotonic_increasing_on_interval_l313_313178


namespace anya_lost_games_l313_313314

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l313_313314


namespace like_terms_eq_l313_313055

theorem like_terms_eq : 
  ∀ (x y : ℕ), 
  (x + 2 * y = 3) → 
  (2 * x + y = 9) → 
  (x + y = 4) :=
by
  intros x y h1 h2
  sorry

end like_terms_eq_l313_313055


namespace ab_value_l313_313659

theorem ab_value (a b : ℝ) (log_two_3 : ℝ := Real.log 3 / Real.log 2) :
  a * log_two_3 = 1 ∧ (4 : ℝ)^b = 3 → a * b = 1 / 2 := by
  sorry

end ab_value_l313_313659


namespace fraction_to_decimal_l313_313921

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313921


namespace circumference_to_diameter_ratio_l313_313363

theorem circumference_to_diameter_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) :
  C / D = 3.14 :=
by
  rw [hC, hD]
  norm_num

end circumference_to_diameter_ratio_l313_313363


namespace trucks_needed_l313_313626

-- Definitions of the conditions
def total_apples : ℕ := 80
def apples_transported : ℕ := 56
def truck_capacity : ℕ := 4

-- Definition to calculate the remaining apples
def remaining_apples : ℕ := total_apples - apples_transported

-- The theorem statement
theorem trucks_needed : remaining_apples / truck_capacity = 6 := by
  sorry

end trucks_needed_l313_313626


namespace train_speed_in_kmh_l313_313867

def train_length : ℝ := 250 -- Length of the train in meters
def station_length : ℝ := 200 -- Length of the station in meters
def time_to_pass : ℝ := 45 -- Time to pass the station in seconds

theorem train_speed_in_kmh :
  (train_length + station_length) / time_to_pass * 3.6 = 36 :=
  sorry -- Proof is skipped

end train_speed_in_kmh_l313_313867


namespace algebraic_expression_value_l313_313183

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -2) : 
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 7 :=
by
  sorry

end algebraic_expression_value_l313_313183


namespace table_height_l313_313453

theorem table_height (r s x y l : ℝ)
  (h1 : x + l - y = 32)
  (h2 : y + l - x = 28) :
  l = 30 :=
by
  sorry

end table_height_l313_313453


namespace fraction_to_decimal_l313_313903

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313903


namespace y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l313_313685

noncomputable def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_is_multiple_of_3 : y % 3 = 0 :=
sorry

theorem y_is_multiple_of_9 : y % 9 = 0 :=
sorry

theorem y_is_multiple_of_27 : y % 27 = 0 :=
sorry

theorem y_is_multiple_of_81 : y % 81 = 0 :=
sorry

end y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l313_313685


namespace fraction_to_decimal_l313_313927

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313927


namespace solve_system_eqn_l313_313667

theorem solve_system_eqn (x y : ℚ) (h₁ : 3*y - 4*x = 8) (h₂ : 2*y + x = -1) :
  x = -19/11 ∧ y = 4/11 :=
by
  sorry

end solve_system_eqn_l313_313667


namespace x_value_when_y_2000_l313_313593

noncomputable def x_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) : ℝ :=
  if hy : y = 2000 then (1 / (50 : ℝ)^(1/3)) else x

-- Theorem statement
theorem x_value_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) :
  x_when_y_2000 x y hxy_pos hxy_inv h_init = 1 / (50 : ℝ)^(1/3) :=
sorry

end x_value_when_y_2000_l313_313593


namespace charlie_first_week_usage_l313_313085

noncomputable def data_used_week1 : ℕ :=
  let data_plan := 8
  let week2_usage := 3
  let week3_usage := 5
  let week4_usage := 10
  let total_extra_cost := 120
  let cost_per_gb_extra := 10
  let total_data_used := data_plan + (total_extra_cost / cost_per_gb_extra)
  let total_data_week_2_3_4 := week2_usage + week3_usage + week4_usage
  total_data_used - total_data_week_2_3_4

theorem charlie_first_week_usage : data_used_week1 = 2 :=
by
  sorry

end charlie_first_week_usage_l313_313085


namespace points_on_line_l313_313570

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313570


namespace min_num_of_teams_l313_313623

theorem min_num_of_teams (num_athletes : ℕ) (max_team_size : ℕ) (h1 : num_athletes = 30) (h2 : max_team_size = 9) :
  ∃ (min_teams : ℕ), min_teams = 5 ∧ (∀ nal : ℕ, (nal > 0 ∧ num_athletes % nal = 0 ∧ nal ≤ max_team_size) → num_athletes / nal ≥ min_teams) :=
by
  sorry

end min_num_of_teams_l313_313623


namespace tim_younger_than_jenny_l313_313263

def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2
def combined_ages_rommel_jenny : ℕ := rommel_age + jenny_age
def uncle_age : ℕ := 2 * combined_ages_rommel_jenny
noncomputable def aunt_age : ℝ := (uncle_age + jenny_age : ℕ) / 2

theorem tim_younger_than_jenny : jenny_age - tim_age = 12 :=
by {
  -- Placeholder proof
  sorry
}

end tim_younger_than_jenny_l313_313263


namespace find_y_parallel_l313_313358

-- Definitions
def a : ℝ × ℝ := (2, 3)
def b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

-- Parallel condition implies proportional components
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- The proof problem
theorem find_y_parallel : ∀ y : ℝ, parallel_vectors a (b y) → y = 7 :=
by
  sorry

end find_y_parallel_l313_313358


namespace correct_calculation_l313_313063

theorem correct_calculation (x : ℕ) (h : x + 10 = 21) : x * 10 = 110 :=
by
  sorry

end correct_calculation_l313_313063


namespace distinct_four_digit_numbers_count_l313_313337

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l313_313337


namespace ones_digit_expression_l313_313764

theorem ones_digit_expression :
  ((73 ^ 1253 * 44 ^ 987 + 47 ^ 123 / 39 ^ 654 * 86 ^ 1484 - 32 ^ 1987) % 10) = 2 := by
  sorry

end ones_digit_expression_l313_313764


namespace product_of_integers_l313_313166

theorem product_of_integers
  (A B C D : ℕ)
  (hA : A > 0)
  (hB : B > 0)
  (hC : C > 0)
  (hD : D > 0)
  (h_sum : A + B + C + D = 72)
  (h_eq : A + 3 = B - 3 ∧ B - 3 = C * 3 ∧ C * 3 = D / 2) :
  A * B * C * D = 68040 := 
by
  sorry

end product_of_integers_l313_313166


namespace board_transformation_l313_313151

def transformation_possible (a b : ℕ) : Prop :=
  6 ∣ (a * b)

theorem board_transformation (a b : ℕ) (h₁ : 2 ≤ a) (h₂ : 2 ≤ b) : 
  transformation_possible a b ↔ 6 ∣ (a * b) := by
  sorry

end board_transformation_l313_313151


namespace sequence_product_is_128_l313_313115

-- Define the sequence of fractions
def fractional_sequence (n : ℕ) : Rat :=
  if n % 2 = 0 then 1 / (2 : ℕ) ^ ((n + 2) / 2)
  else (2 : ℕ) ^ ((n + 1) / 2)

-- The target theorem: prove the product of the sequence results in 128
theorem sequence_product_is_128 : 
  (List.prod (List.map fractional_sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])) = 128 := 
by
  sorry

end sequence_product_is_128_l313_313115


namespace find_B_l313_313365

variable {A B C a b c : Real}

noncomputable def B_value (A B C a b c : Real) : Prop :=
  B = 2 * Real.pi / 3

theorem find_B 
  (h_triangle: a^2 + b^2 + c^2 = 2*a*b*Real.cos C)
  (h_cos_eq: (2 * a + c) * Real.cos B + b * Real.cos C = 0) : 
  B_value A B C a b c :=
by
  sorry

end find_B_l313_313365


namespace find_k_l313_313791

-- Conditions
def t : ℕ := 6
def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

-- Given these conditions, we need to prove that k = 9
theorem find_k (k t : ℕ) (h1 : t = 6) (h2 : is_nonzero_digit k) (h3 : is_nonzero_digit t) :
    (8 * 10^2 + k * 10 + 8) + (k * 10^2 + 8 * 10 + 8) - 16 * t * 10^0 * 6 = (9 * 10 + 8) + (9 * 10^2 + 8 * 10 + 8) - (16 * 6 * 10^1 + 6) → k = 9 := 
sorry

end find_k_l313_313791


namespace favorite_movies_total_hours_l313_313797

theorem favorite_movies_total_hours (michael_hrs joyce_hrs nikki_hrs ryn_hrs sam_hrs alex_hrs : ℕ)
  (H1 : nikki_hrs = 30)
  (H2 : michael_hrs = nikki_hrs / 3)
  (H3 : joyce_hrs = michael_hrs + 2)
  (H4 : ryn_hrs = (4 * nikki_hrs) / 5)
  (H5 : sam_hrs = (3 * joyce_hrs) / 2)
  (H6 : alex_hrs = 2 * michael_hrs) :
  michael_hrs + joyce_hrs + nikki_hrs + ryn_hrs + sam_hrs + alex_hrs = 114 := 
sorry

end favorite_movies_total_hours_l313_313797


namespace accounting_vs_calling_clients_l313_313083

/--
Given:
1. Total time Maryann worked today is 560 minutes.
2. Maryann spent 70 minutes calling clients.

Prove:
Maryann spends 7 times longer doing accounting than calling clients.
-/
theorem accounting_vs_calling_clients 
  (total_time : ℕ) 
  (calling_time : ℕ) 
  (h_total : total_time = 560) 
  (h_calling : calling_time = 70) : 
  (total_time - calling_time) / calling_time = 7 :=
  sorry

end accounting_vs_calling_clients_l313_313083


namespace sampling_correct_l313_313658

def systematic_sampling (total_students : Nat) (num_selected : Nat) (interval : Nat) (start : Nat) : List Nat :=
  (List.range num_selected).map (λ i => start + i * interval)

theorem sampling_correct :
  systematic_sampling 60 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end sampling_correct_l313_313658


namespace proof_problem_l313_313179

noncomputable def f (a x : ℝ) : ℝ := a^x
noncomputable def g (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem proof_problem (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_f : f a 2 = 9) : 
    g a (1/9) + f a 3 = 25 :=
by
  -- Definitions and assumptions based on the provided problem
  sorry

end proof_problem_l313_313179


namespace value_of_x_l313_313428

variable (x y z : ℝ)

-- Conditions based on the problem statement
def condition1 := x = (1 / 3) * y
def condition2 := y = (1 / 4) * z
def condition3 := z = 96

-- The theorem to be proven
theorem value_of_x (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) : x = 8 := 
by
  sorry

end value_of_x_l313_313428


namespace lily_pads_doubling_l313_313066

theorem lily_pads_doubling (patch_half_day: ℕ) (doubling_rate: ℝ)
  (H1: patch_half_day = 49)
  (H2: doubling_rate = 2): (patch_half_day + 1) = 50 :=
by 
  sorry

end lily_pads_doubling_l313_313066


namespace question1_question2_l313_313512

noncomputable def setA := {x : ℝ | -2 < x ∧ x < 4}
noncomputable def setB (m : ℝ) := {x : ℝ | x < -m}

-- (1) If A ∩ B = ∅, find the range of the real number m.
theorem question1 (m : ℝ) (h : setA ∩ setB m = ∅) : 2 ≤ m := by
  sorry

-- (2) If A ⊂ B, find the range of the real number m.
theorem question2 (m : ℝ) (h : setA ⊂ setB m) : m ≤ 4 := by
  sorry

end question1_question2_l313_313512


namespace rationalize_denominator_l313_313405

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l313_313405


namespace old_manufacturing_cost_l313_313477

theorem old_manufacturing_cost (P : ℝ) :
  (50 : ℝ) = P * 0.50 →
  (0.65 : ℝ) * P = 65 :=
by
  intros hp₁
  -- Proof omitted
  sorry

end old_manufacturing_cost_l313_313477


namespace advertising_time_l313_313250

-- Define the conditions
def total_duration : ℕ := 30
def national_news : ℕ := 12
def international_news : ℕ := 5
def sports : ℕ := 5
def weather_forecasts : ℕ := 2

-- Calculate total content time
def total_content_time : ℕ := national_news + international_news + sports + weather_forecasts

-- Define the proof problem
theorem advertising_time (h : total_duration - total_content_time = 6) : (total_duration - total_content_time) = 6 :=
by
sorry

end advertising_time_l313_313250


namespace problem_statement_l313_313057

def f (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem problem_statement : f (f (-1)) = 10 := by
  sorry

end problem_statement_l313_313057


namespace range_of_a_l313_313208

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l313_313208


namespace polynomial_proof_l313_313038

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

theorem polynomial_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) : 
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
by 
  sorry

end polynomial_proof_l313_313038


namespace largest_among_a_b_c_d_l313_313998

noncomputable def a : ℝ := Real.log 2022 / Real.log 2021
noncomputable def b : ℝ := Real.log 2023 / Real.log 2022
noncomputable def c : ℝ := 2022 / 2021
noncomputable def d : ℝ := 2023 / 2022

theorem largest_among_a_b_c_d : max a (max b (max c d)) = c := 
sorry

end largest_among_a_b_c_d_l313_313998


namespace tree_height_relationship_l313_313379

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end tree_height_relationship_l313_313379


namespace number_of_hardbacks_l313_313259

theorem number_of_hardbacks (H P : ℕ) (books total_books selections : ℕ) (comb : ℕ → ℕ → ℕ) :
  total_books = 8 →
  P = 2 →
  comb total_books 3 - comb H 3 = 36 →
  H = 6 :=
by sorry

end number_of_hardbacks_l313_313259


namespace same_terminal_side_l313_313515

theorem same_terminal_side (α : ℝ) (k : ℤ) (h : α = -51) : 
  ∃ (m : ℤ), α + m * 360 = k * 360 - 51 :=
by {
    sorry
}

end same_terminal_side_l313_313515


namespace functional_eq_zero_l313_313483

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_zero :
  (∀ x y : ℝ, f (x + y) = f x - f y) →
  (∀ x : ℝ, f x = 0) :=
by
  intros h x
  sorry

end functional_eq_zero_l313_313483


namespace num_solutions_even_pairs_l313_313487

theorem num_solutions_even_pairs : ∃ n : ℕ, n = 25 ∧ ∀ (x y : ℕ),
  x % 2 = 0 ∧ y % 2 = 0 ∧ 4 * x + 6 * y = 600 → n = 25 :=
by
  sorry

end num_solutions_even_pairs_l313_313487


namespace turns_per_minute_l313_313600

theorem turns_per_minute (x : ℕ) (h₁ : x > 0) (h₂ : 60 / x = (60 / (x + 5)) + 2) :
  60 / x = 6 ∧ 60 / (x + 5) = 4 :=
by sorry

end turns_per_minute_l313_313600


namespace tree_boy_growth_ratio_l313_313133

theorem tree_boy_growth_ratio 
    (initial_tree_height final_tree_height initial_boy_height final_boy_height : ℕ) 
    (h₀ : initial_tree_height = 16) 
    (h₁ : final_tree_height = 40) 
    (h₂ : initial_boy_height = 24) 
    (h₃ : final_boy_height = 36) 
:
  (final_tree_height - initial_tree_height) / (final_boy_height - initial_boy_height) = 2 := 
by {
    -- Definitions and given conditions used in the statement part of the proof
    sorry
}

end tree_boy_growth_ratio_l313_313133


namespace minimum_value_l313_313642

theorem minimum_value (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : S 2017 = 4034)
    (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
    (h4: ∀ n, a n > 0) : 
    (∃ c : ℝ, (1 / a 9) + (9 / a 2009) = c) ∧ (∀ d : ℝ, (1 / a 9) + (9 / a 2009) ≥ d → d ≥ 4) :=
by
  sorry

end minimum_value_l313_313642


namespace correct_option_d_l313_313718

-- Define the conditions as separate lemmas
lemma option_a_incorrect : ¬ (Real.sqrt 18 + Real.sqrt 2 = 2 * Real.sqrt 5) :=
sorry 

lemma option_b_incorrect : ¬ (Real.sqrt 18 - Real.sqrt 2 = 4) :=
sorry

lemma option_c_incorrect : ¬ (Real.sqrt 18 * Real.sqrt 2 = 36) :=
sorry

-- Define the statement to prove
theorem correct_option_d : Real.sqrt 18 / Real.sqrt 2 = 3 :=
by
  sorry

end correct_option_d_l313_313718


namespace find_x_l313_313785

theorem find_x (x : ℕ) : 8000 * 6000 = x * 10^5 → x = 480 := by
  sorry

end find_x_l313_313785


namespace sum_of_unit_fractions_l313_313447

theorem sum_of_unit_fractions : (1 / 2) + (1 / 3) + (1 / 7) + (1 / 42) = 1 := 
by 
  sorry

end sum_of_unit_fractions_l313_313447


namespace equal_circle_radius_l313_313110

theorem equal_circle_radius (r R : ℝ) (h1: r > 0) (h2: R > 0)
  : ∃ x : ℝ, x = r * R / (R + r) :=
by 
  sorry

end equal_circle_radius_l313_313110


namespace rationalize_denominator_l313_313403

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l313_313403


namespace homework_points_l313_313551

variable (H Q T : ℕ)

theorem homework_points (h1 : T = 4 * Q)
                        (h2 : Q = H + 5)
                        (h3 : H + Q + T = 265) : 
  H = 40 :=
sorry

end homework_points_l313_313551


namespace intersection_M_N_l313_313983

-- Define set M and N
def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5 * x + 6 > 0}

-- Problem statement to show their intersection
theorem intersection_M_N :
  M ∩ N = {x | x < 1} := 
sorry

end intersection_M_N_l313_313983


namespace initial_points_l313_313555

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l313_313555


namespace non_working_games_count_l313_313232

-- Definitions based on conditions
def total_games : Nat := 15
def total_earnings : Nat := 30
def price_per_game : Nat := 5

-- Definition to be proved
def working_games : Nat := total_earnings / price_per_game
def non_working_games : Nat := total_games - working_games

-- Statement to be proved
theorem non_working_games_count : non_working_games = 9 :=
by
  sorry

end non_working_games_count_l313_313232


namespace jonathan_daily_burn_l313_313075

-- Conditions
def daily_calories : ℕ := 2500
def extra_saturday_calories : ℕ := 1000
def weekly_deficit : ℕ := 2500

-- Question and Answer
theorem jonathan_daily_burn :
  let weekly_intake := 6 * daily_calories + (daily_calories + extra_saturday_calories)
  let total_weekly_burn := weekly_intake + weekly_deficit
  total_weekly_burn / 7 = 3000 :=
by
  sorry

end jonathan_daily_burn_l313_313075


namespace intersection_of_A_and_B_l313_313048

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 1 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {2} :=
sorry

end intersection_of_A_and_B_l313_313048


namespace range_of_a_l313_313061

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (1 / 4)^x + (1 / 2)^(x - 1) + a = 0) →
  (-3 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l313_313061


namespace rationalize_denominator_l313_313406

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l313_313406


namespace tan_of_diff_l313_313977

theorem tan_of_diff (θ : ℝ) (hθ : -π/2 + 2 * π < θ ∧ θ < 2 * π) 
  (h : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 :=
sorry

end tan_of_diff_l313_313977


namespace cat_weight_l313_313013

theorem cat_weight 
  (weight1 weight2 : ℕ)
  (total_weight : ℕ)
  (h1 : weight1 = 2)
  (h2 : weight2 = 7)
  (h3 : total_weight = 13) : 
  ∃ weight3 : ℕ, weight3 = 4 := 
by
  sorry

end cat_weight_l313_313013


namespace fraction_equals_decimal_l313_313934

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313934


namespace candies_total_l313_313391

-- Defining the given conditions
def LindaCandies : ℕ := 34
def ChloeCandies : ℕ := 28
def TotalCandies : ℕ := LindaCandies + ChloeCandies

-- Proving the total number of candies
theorem candies_total : TotalCandies = 62 :=
  by
    sorry

end candies_total_l313_313391


namespace Anya_loss_games_l313_313317

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l313_313317


namespace identity_eq_a_minus_b_l313_313880

theorem identity_eq_a_minus_b (a b : ℚ) (x : ℚ) (h : ∀ x, x > 0 → 
  (a / (2^x - 2) + b / (2^x + 3) = (5 * 2^x + 4) / ((2^x - 2) * (2^x + 3)))) : 
  a - b = 3 / 5 := 
by 
  sorry

end identity_eq_a_minus_b_l313_313880


namespace square_diff_correctness_l313_313839

theorem square_diff_correctness (x y : ℝ) :
  let A := (x + y) * (x - 2*y)
  let B := (x + y) * (-x + y)
  let C := (x + y) * (-x - y)
  let D := (-x + y) * (x - y)
  (∃ (a b : ℝ), B = (a + b) * (a - b)) ∧ (∀ (p q : ℝ), A ≠ (p + q) * (p - q)) ∧ (∀ (r s : ℝ), C ≠ (r + s) * (r - s)) ∧ (∀ (t u : ℝ), D ≠ (t + u) * (t - u)) :=
by
  sorry

end square_diff_correctness_l313_313839


namespace expected_adjacent_red_pairs_in_circle_l313_313820

-- Definitions and conditions
def deck := fin 104 -- represents the 104 cards
def red_cards : finset deck := finset.range 52 -- out of 104 cards, 52 are red

-- Expected number of adjacent red pairs
noncomputable def expected_adjacent_red_pairs : ℚ :=
  (52:ℚ) * (51:ℚ) / (103:ℚ)

-- Statement of the theorem
theorem expected_adjacent_red_pairs_in_circle :
  expected_adjacent_red_pairs = 2652 / 103 :=
by
  sorry

end expected_adjacent_red_pairs_in_circle_l313_313820


namespace max_black_cells_1000_by_1000_l313_313813

def maxBlackCells (m n : ℕ) : ℕ :=
  if m = 1 then n else if n = 1 then m else m + n - 2

theorem max_black_cells_1000_by_1000 : maxBlackCells 1000 1000 = 1998 :=
  by sorry

end max_black_cells_1000_by_1000_l313_313813


namespace savings_equal_in_820_weeks_l313_313090

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end savings_equal_in_820_weeks_l313_313090


namespace last_digit_of_1_div_2_pow_15_l313_313267

theorem last_digit_of_1_div_2_pow_15 :
  let last_digit_of := (n : ℕ) → n % 10
  last_digit_of (5^15) = 5 → 
  (∀ (n : ℕ),  ∃ (k : ℕ), n = 2^k →  last_digit_of (5 ^ k) = last_digit_of (1 / 2 ^ 15)) := 
by 
  intro last_digit_of h proof
  exact sorry

end last_digit_of_1_div_2_pow_15_l313_313267


namespace man_speed_with_the_stream_l313_313855

def speed_with_the_stream (V_m V_s : ℝ) : Prop :=
  V_m + V_s = 2

theorem man_speed_with_the_stream (V_m V_s : ℝ) (h1 : V_m - V_s = 2) (h2 : V_m = 2) : speed_with_the_stream V_m V_s :=
by
  sorry

end man_speed_with_the_stream_l313_313855


namespace cube_side_length_l313_313294

theorem cube_side_length (n : ℕ) (h1 : 6 * (n^2) = 1/3 * 6 * (n^3)) : n = 3 := 
sorry

end cube_side_length_l313_313294


namespace tennis_balls_in_each_container_l313_313383

theorem tennis_balls_in_each_container (initial_balls : ℕ) (half_gone : ℕ) (remaining_balls : ℕ) (containers : ℕ) 
  (h1 : initial_balls = 100) 
  (h2 : half_gone = initial_balls / 2)
  (h3 : remaining_balls = initial_balls - half_gone)
  (h4 : containers = 5) :
  remaining_balls / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l313_313383


namespace scientific_notation_correct_l313_313827

def original_number : ℕ := 31900

def scientific_notation_option_A : ℝ := 3.19 * 10^2
def scientific_notation_option_B : ℝ := 0.319 * 10^3
def scientific_notation_option_C : ℝ := 3.19 * 10^4
def scientific_notation_option_D : ℝ := 0.319 * 10^5

theorem scientific_notation_correct :
  original_number = 31900 ∧ scientific_notation_option_C = 3.19 * 10^4 ∧ (original_number : ℝ) = scientific_notation_option_C := 
by 
  sorry

end scientific_notation_correct_l313_313827


namespace InfinitePairsExist_l313_313087

theorem InfinitePairsExist (a b : ℕ) : (∀ n : ℕ, ∃ a b : ℕ, a ∣ b^2 + 1 ∧ b ∣ a^2 + 1) :=
sorry

end InfinitePairsExist_l313_313087


namespace trains_crossing_time_l313_313438

-- Definitions based on given conditions
noncomputable def length_A : ℝ := 2500
noncomputable def time_A : ℝ := 50
noncomputable def length_B : ℝ := 3500
noncomputable def speed_factor : ℝ := 1.2

-- Speed computations
noncomputable def speed_A : ℝ := length_A / time_A
noncomputable def speed_B : ℝ := speed_A * speed_factor

-- Relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := speed_A + speed_B

-- Total distance covered when crossing each other
noncomputable def total_distance : ℝ := length_A + length_B

-- Time taken to cross each other
noncomputable def time_to_cross : ℝ := total_distance / relative_speed

-- Proof statement: Time taken is approximately 54.55 seconds
theorem trains_crossing_time :
  |time_to_cross - 54.55| < 0.01 := by
  sorry

end trains_crossing_time_l313_313438


namespace count_even_three_digit_numbers_less_than_800_l313_313603

def even_three_digit_numbers_less_than_800 : Nat :=
  let hundreds_choices := 7
  let tens_choices := 8
  let units_choices := 4
  hundreds_choices * tens_choices * units_choices

theorem count_even_three_digit_numbers_less_than_800 :
  even_three_digit_numbers_less_than_800 = 224 := 
by 
  unfold even_three_digit_numbers_less_than_800
  rfl

end count_even_three_digit_numbers_less_than_800_l313_313603


namespace problem_statement_l313_313080

theorem problem_statement :
  let a := (List.range (60 / 12)).card
  let b := (List.range (60 / Nat.lcm (Nat.lcm 2 3) 4)).card
  (a - b) ^ 3 = 0 :=
by
  sorry

end problem_statement_l313_313080


namespace solve_x_l313_313837

theorem solve_x : ∃ x : ℝ, 65 + (5 * x) / (180 / 3) = 66 ∧ x = 12 := by
  sorry

end solve_x_l313_313837


namespace original_strip_length_l313_313847

theorem original_strip_length (x : ℝ) 
  (h1 : 3 + x + 3 + x + 3 + x + 3 + x + 3 = 27) : 
  4 * 9 + 4 * 3 = 57 := 
  sorry

end original_strip_length_l313_313847


namespace exist_pos_integers_m_n_l313_313799

def d (n : ℕ) : ℕ :=
  -- Number of divisors of n
  sorry 

theorem exist_pos_integers_m_n :
  ∃ (m n : ℕ), (m > 0) ∧ (n > 0) ∧ (m = 24) ∧ 
  ((∃ (triples : Finset (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ triples ↔ (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c ≤ m) ∧ (d (n + a) * d (n + b) * d (n + c)) % (a * b * c) = 0) ∧ 
    (triples.card = 2024))) :=
sorry

end exist_pos_integers_m_n_l313_313799


namespace at_least_one_A_or_B_selected_prob_l313_313583

theorem at_least_one_A_or_B_selected_prob :
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  at_least_one_A_or_B_prob = 5 / 6 :=
by
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  sorry

end at_least_one_A_or_B_selected_prob_l313_313583


namespace selection_ways_l313_313829

open Finset

theorem selection_ways : 
  let n := 8;
  let english_translation := 5;
  let software_design := 4;
  let both := 1; -- person A
  let english_without_both := english_translation - both; -- 4
  let software_without_both := software_design - both; -- 3
  let total_selection := 5;
  let english_needed := 3;
  let software_needed := 2;
  let choose (n k : ℕ) : ℕ := (Finset.range n).choose k
  in
  choose english_without_both english_needed * choose software_without_both software_needed +
    choose (english_without_both - 1) (english_needed - 1) * choose software_without_both software_needed +
    choose english_without_both english_needed * choose (software_without_both - 1) (software_needed - 1) = 42 :=
by sorry

end selection_ways_l313_313829


namespace wheel_center_travel_distance_l313_313746

theorem wheel_center_travel_distance (radius : ℝ) (revolutions : ℝ) (flat_surface : Prop) 
  (h_radius : radius = 2) (h_revolutions : revolutions = 2) : 
  radius * 2 * π * revolutions = 8 * π :=
by
  rw [h_radius, h_revolutions]
  simp [mul_assoc, mul_comm]
  sorry

end wheel_center_travel_distance_l313_313746


namespace fraction_to_decimal_l313_313898

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313898


namespace fraction_to_decimal_l313_313895

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313895


namespace paving_stone_length_l313_313000

theorem paving_stone_length (courtyard_length courtyard_width paving_stone_width : ℝ)
  (num_paving_stones : ℕ)
  (courtyard_dims : courtyard_length = 40 ∧ courtyard_width = 20) 
  (paving_stone_dims : paving_stone_width = 2) 
  (num_stones : num_paving_stones = 100) 
  : (courtyard_length * courtyard_width) / (num_paving_stones * paving_stone_width) = 4 :=
by 
  sorry

end paving_stone_length_l313_313000


namespace sum_arithmetic_sequence_l313_313873

theorem sum_arithmetic_sequence :
  let n := 21
  let a := 100
  let l := 120
  (n / 2) * (a + l) = 2310 :=
by
  -- define n, a, and l based on the conditions
  let n := 21
  let a := 100
  let l := 120
  -- state the goal
  have h : (n / 2) * (a + l) = 2310 := sorry
  exact h

end sum_arithmetic_sequence_l313_313873


namespace fraction_to_decimal_equiv_l313_313961

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313961


namespace min_students_solving_most_l313_313431

theorem min_students_solving_most (students problems : Nat) 
    (total_students : students = 10) 
    (problems_per_student : Nat → Nat) 
    (problems_per_student_property : ∀ s, s < students → problems_per_student s = 3) 
    (common_problem : ∀ s1 s2, s1 < students → s2 < students → s1 ≠ s2 → ∃ p, p < problems ∧ (∃ (solves1 solves2 : Nat → Nat), (solves1 p = 1 ∧ solves2 p = 1) ∧ s1 < students ∧ s2 < students)): 
  ∃ min_students, min_students = 5 :=
by
  sorry

end min_students_solving_most_l313_313431


namespace parents_without_full_time_jobs_l313_313722

theorem parents_without_full_time_jobs
  {total_parents mothers fathers : ℕ}
  (h_total_parents : total_parents = 100)
  (h_mothers_percentage : mothers = 60)
  (h_fathers_percentage : fathers = 40)
  (h_mothers_full_time : ℕ)
  (h_fathers_full_time : ℕ)
  (h_mothers_ratio : h_mothers_full_time = (5 * mothers) / 6)
  (h_fathers_ratio : h_fathers_full_time = (3 * fathers) / 4) :
  ((total_parents - (h_mothers_full_time + h_fathers_full_time)) * 100 / total_parents = 20) := sorry

end parents_without_full_time_jobs_l313_313722


namespace fraction_to_decimal_l313_313885

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313885


namespace ten_more_than_twice_number_of_birds_l313_313698

def number_of_birds : ℕ := 20

theorem ten_more_than_twice_number_of_birds :
  10 + 2 * number_of_birds = 50 :=
by
  sorry

end ten_more_than_twice_number_of_birds_l313_313698


namespace monotonic_increasing_range_l313_313216

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l313_313216


namespace original_deck_size_l313_313287

-- Let's define the number of red and black cards initially
def numRedCards (r : ℕ) : ℕ := r
def numBlackCards (b : ℕ) : ℕ := b

-- Define the initial condition as given in the problem
def initial_prob_red (r b : ℕ) : Prop :=
  r / (r + b) = 2 / 5

-- Define the condition after adding 7 black cards
def prob_red_after_adding_black (r b : ℕ) : Prop :=
  r / (r + (b + 7)) = 1 / 3

-- The proof statement to verify original number of cards in the deck
theorem original_deck_size (r b : ℕ) (h1 : initial_prob_red r b) (h2 : prob_red_after_adding_black r b) : r + b = 35 := by
  sorry

end original_deck_size_l313_313287


namespace shooting_accuracy_l313_313074

theorem shooting_accuracy 
  (P_A : ℚ) 
  (P_AB : ℚ) 
  (h1 : P_A = 9 / 10) 
  (h2 : P_AB = 1 / 2) 
  : P_AB / P_A = 5 / 9 := 
by
  sorry

end shooting_accuracy_l313_313074


namespace area_of_given_triangle_is_32_l313_313308

noncomputable def area_of_triangle : ℕ :=
  let A := (-8, 0)
  let B := (0, 8)
  let C := (0, 0)
  1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℤ).natAbs

theorem area_of_given_triangle_is_32 : area_of_triangle = 32 := 
  sorry

end area_of_given_triangle_is_32_l313_313308


namespace relationship_between_roses_and_total_flowers_l313_313367

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end relationship_between_roses_and_total_flowers_l313_313367


namespace triangle_inequality_harmonic_mean_l313_313277

theorem triangle_inequality_harmonic_mean (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ DP DQ : ℝ, DP + DQ ≤ (2 * a * b) / (a + b) :=
by
  sorry

end triangle_inequality_harmonic_mean_l313_313277


namespace abc_sum_equals_9_l313_313359

theorem abc_sum_equals_9 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a * b + c = 57) (h5 : b * c + a = 57) (h6 : a * c + b = 57) :
  a + b + c = 9 := 
sorry

end abc_sum_equals_9_l313_313359


namespace angle_size_proof_l313_313548

-- Define the problem conditions
def fifteen_points_on_circle (θ : ℕ) : Prop :=
  θ = 360 / 15 

-- Define the central angles
def central_angle_between_adjacent_points (θ : ℕ) : ℕ :=
  360 / 15  

-- Define the two required central angles
def central_angle_A1O_A3 (θ : ℕ) : ℕ :=
  2 * θ

def central_angle_A3O_A7 (θ : ℕ) : ℕ :=
  4 * θ

-- Define the problem using the given conditions and the proven answer
noncomputable def angle_A1_A3_A7 : ℕ :=
  108

-- Lean 4 statement of the math problem to prove
theorem angle_size_proof (θ : ℕ) (h1 : fifteen_points_on_circle θ) :
  central_angle_A1O_A3 θ = 48 ∧ central_angle_A3O_A7 θ = 96 → 
  angle_A1_A3_A7 = 108 :=
by sorry

#check angle_size_proof

end angle_size_proof_l313_313548


namespace subtraction_identity_l313_313728

theorem subtraction_identity : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 :=
  by norm_num

end subtraction_identity_l313_313728


namespace solve_for_x_l313_313030

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end solve_for_x_l313_313030


namespace original_faculty_number_l313_313136

theorem original_faculty_number (x : ℝ) (h : 0.85 * x = 195) : x = 229 := by
  sorry

end original_faculty_number_l313_313136


namespace min_pieces_for_net_l313_313739

theorem min_pieces_for_net (n : ℕ) : ∃ (m : ℕ), m = n * (n + 1) := by
  sorry

end min_pieces_for_net_l313_313739


namespace column_of_1000_is_C_l313_313144

def column_of_integer (n : ℕ) : String :=
  ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"].get! ((n - 2) % 10)

theorem column_of_1000_is_C :
  column_of_integer 1000 = "C" :=
by
  sorry

end column_of_1000_is_C_l313_313144


namespace increase_in_license_plates_l313_313546

/-- The number of old license plates and new license plates in MiraVille. -/
def old_license_plates : ℕ := 26^2 * 10^3
def new_license_plates : ℕ := 26^2 * 10^4

/-- The ratio of the number of new license plates to the number of old license plates is 10. -/
theorem increase_in_license_plates : new_license_plates / old_license_plates = 10 := by
  unfold old_license_plates new_license_plates
  sorry

end increase_in_license_plates_l313_313546


namespace margo_total_distance_l313_313228

-- Definitions based on the conditions
def time_to_friends_house_min : ℕ := 15
def time_to_return_home_min : ℕ := 25
def total_walking_time_min : ℕ := time_to_friends_house_min + time_to_return_home_min
def total_walking_time_hours : ℚ := total_walking_time_min / 60
def average_walking_rate_mph : ℚ := 3
def total_distance_miles : ℚ := average_walking_rate_mph * total_walking_time_hours

-- The statement of the proof problem
theorem margo_total_distance : total_distance_miles = 2 := by
  sorry

end margo_total_distance_l313_313228


namespace new_avg_weight_l313_313701

-- Define the weights of individuals
variables (A B C D E : ℕ)
-- Conditions
axiom avg_ABC : (A + B + C) / 3 = 84
axiom avg_ABCD : (A + B + C + D) / 4 = 80
axiom E_def : E = D + 8
axiom A_80 : A = 80

theorem new_avg_weight (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 8) 
  (h4 : A = 80) 
  : (B + C + D + E) / 4 = 79 := 
by
  sorry

end new_avg_weight_l313_313701


namespace rationalize_denominator_l313_313402

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l313_313402


namespace polynomial_divisibility_a_l313_313767

theorem polynomial_divisibility_a (n : ℕ) : 
  (n % 3 = 1 ∨ n % 3 = 2) ↔ (x^2 + x + 1 ∣ x^(2*n) + x^n + 1) :=
sorry

end polynomial_divisibility_a_l313_313767


namespace ryan_spanish_hours_l313_313026

theorem ryan_spanish_hours (S : ℕ) (h : 7 = S + 3) : S = 4 :=
sorry

end ryan_spanish_hours_l313_313026


namespace son_age_l313_313857

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end son_age_l313_313857


namespace min_fencing_dims_l313_313392

theorem min_fencing_dims (x : ℕ) (h₁ : x * (x + 5) ≥ 600) (h₂ : x = 23) : 
  2 * (x + (x + 5)) = 102 := 
by
  -- Placeholder for the proof
  sorry

end min_fencing_dims_l313_313392


namespace customers_who_didnt_tip_l313_313745

def initial_customers : ℕ := 39
def added_customers : ℕ := 12
def customers_who_tipped : ℕ := 2

theorem customers_who_didnt_tip : initial_customers + added_customers - customers_who_tipped = 49 := by
  sorry

end customers_who_didnt_tip_l313_313745


namespace binary_to_decimal_10101_l313_313476

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end binary_to_decimal_10101_l313_313476


namespace arithmetic_sum_calculation_l313_313875

theorem arithmetic_sum_calculation :
  3 * (71 + 75 + 79 + 83 + 87 + 91) = 1458 :=
by
  sorry

end arithmetic_sum_calculation_l313_313875


namespace orchids_initially_three_l313_313109

-- Define initial number of roses and provided number of orchids in the vase
def initial_roses : ℕ := 9
def added_orchids (O : ℕ) : ℕ := 13
def added_roses : ℕ := 3
def difference := 10

-- Define initial number of orchids that we need to prove
def initial_orchids (O : ℕ) : Prop :=
  added_orchids O - added_roses = difference →
  O = 3

theorem orchids_initially_three :
  initial_orchids O :=
sorry

end orchids_initially_three_l313_313109


namespace smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l313_313629

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5 :
  ∃ n : ℕ, n = 0b11011 ∧ is_palindrome n 2 ∧ is_palindrome n 5 :=
by
  existsi 0b11011
  sorry

end smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l313_313629


namespace fish_ratio_l313_313683

theorem fish_ratio (k : ℕ) (kendra_fish : ℕ) (home_fish : ℕ)
    (h1 : kendra_fish = 30)
    (h2 : home_fish = 87)
    (h3 : k - 3 + kendra_fish = home_fish) :
  k = 60 ∧ (k / 3, kendra_fish / 3) = (19, 10) :=
by
  sorry

end fish_ratio_l313_313683


namespace joe_commute_time_l313_313527

theorem joe_commute_time
  (d : ℝ) -- total one-way distance from home to school
  (rw : ℝ) -- Joe's walking rate
  (rr : ℝ := 4 * rw) -- Joe's running rate (4 times walking rate)
  (walking_time_for_one_third : ℝ := 9) -- Joe takes 9 minutes to walk one-third distance
  (walking_time_two_thirds : ℝ := 2 * walking_time_for_one_third) -- time to walk two-thirds distance
  (running_time_two_thirds : ℝ := walking_time_two_thirds / 4) -- time to run two-thirds 
  : (2 * walking_time_two_thirds + running_time_two_thirds) = 40.5 := -- total travel time
by
  sorry

end joe_commute_time_l313_313527


namespace f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l313_313777

noncomputable def f (x : ℝ) : ℝ := (4 * Real.exp x) / (Real.exp x + 1)

theorem f_sin_periodic : ∀ x, f (Real.sin (x + 2 * Real.pi)) = f (Real.sin x) := sorry

theorem f_monotonically_increasing : ∀ x y, x < y → f x < f y := sorry

theorem f_minus_2_not_even : ¬(∀ x, f x - 2 = f (-x) - 2) := sorry

theorem f_symmetric_about_point : ∀ x, f x + f (-x) = 4 := sorry

end f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l313_313777


namespace minimum_value_of_expression_l313_313328

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : ab = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := 
sorry

end minimum_value_of_expression_l313_313328


namespace germs_killed_in_common_l313_313736

theorem germs_killed_in_common :
  ∃ x : ℝ, x = 5 ∧
    ∀ A B C : ℝ, A = 50 → 
    B = 25 → 
    C = 30 → 
    x = A + B - (100 - C) := sorry

end germs_killed_in_common_l313_313736


namespace patio_perimeter_is_100_feet_l313_313025

theorem patio_perimeter_is_100_feet
  (rectangle : Prop)
  (length : ℝ)
  (width : ℝ)
  (length_eq_40 : length = 40)
  (length_eq_4_times_width : length = 4 * width) :
  2 * length + 2 * width = 100 := 
by
  sorry

end patio_perimeter_is_100_feet_l313_313025


namespace simplify_expression_l313_313716

theorem simplify_expression (x y : ℝ) : (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3 * y + x * y^3)⁻¹ * (x + y) :=
by sorry

end simplify_expression_l313_313716


namespace halfway_fraction_l313_313248

theorem halfway_fraction (a b : ℚ) (h1 : a = 1/5) (h2 : b = 1/3) : (a + b) / 2 = 4 / 15 :=
by 
  rw [h1, h2]
  norm_num

end halfway_fraction_l313_313248


namespace ratio_of_cream_max_to_maxine_l313_313231

def ounces_of_cream_in_max (coffee_sipped : ℕ) (cream_added: ℕ) : ℕ := cream_added

def ounces_of_remaining_cream_in_maxine (initial_coffee : ℚ) (cream_added: ℚ) (sipped : ℚ) : ℚ :=
  let total_mixture := initial_coffee + cream_added
  let remaining_mixture := total_mixture - sipped
  (initial_coffee / total_mixture) * cream_added

theorem ratio_of_cream_max_to_maxine :
  let max_cream := ounces_of_cream_in_max 4 3
  let maxine_cream := ounces_of_remaining_cream_in_maxine 16 3 5
  (max_cream : ℚ) / maxine_cream = 19 / 14 := by 
  sorry

end ratio_of_cream_max_to_maxine_l313_313231


namespace garden_length_l313_313235

open Nat

def perimeter : ℕ → ℕ → ℕ := λ l w => 2 * (l + w)

theorem garden_length (width : ℕ) (perimeter_val : ℕ) (length : ℕ) 
  (h1 : width = 15) 
  (h2 : perimeter_val = 80) 
  (h3 : perimeter length width = perimeter_val) :
  length = 25 := by
  sorry

end garden_length_l313_313235


namespace meso_tyler_time_to_type_40_pages_l313_313394

-- Define the typing speeds
def meso_speed : ℝ := 15 / 5 -- 3 pages per minute
def tyler_speed : ℝ := 15 / 3 -- 5 pages per minute
def combined_speed : ℝ := meso_speed + tyler_speed -- 8 pages per minute

-- Define the number of pages to type
def pages : ℝ := 40

-- Prove the time required to type the pages together
theorem meso_tyler_time_to_type_40_pages : 
  ∃ (t : ℝ), t = pages / combined_speed :=
by
  use 5 -- this is the correct answer
  sorry

end meso_tyler_time_to_type_40_pages_l313_313394


namespace part1_intersection_part1_union_complement_part2_subset_l313_313182

open Set

variable {x a : ℝ}

def A : Set ℝ := {x | 3 ≤ 3^x ∧ 3^x ≤ 27}
def B : Set ℝ := {x | log 2 x > 1}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

theorem part1_intersection :
  A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem part1_union_complement :
  compl B ∪ A = {x | x ≤ 3} :=
sorry

theorem part2_subset (a : ℝ) :
  C a ⊆ A → 1 < a ∧ a ≤ 3 :=
sorry

end part1_intersection_part1_union_complement_part2_subset_l313_313182


namespace triangle_probability_is_correct_l313_313396

-- Define the total number of figures
def total_figures : ℕ := 8

-- Define the number of triangles among the figures
def number_of_triangles : ℕ := 3

-- Define the probability function for choosing a triangle
def probability_of_triangle : ℚ := number_of_triangles / total_figures

-- The theorem to be proved
theorem triangle_probability_is_correct :
  probability_of_triangle = 3 / 8 := by
  sorry

end triangle_probability_is_correct_l313_313396


namespace fraction_to_decimal_l313_313888

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313888


namespace monotonic_intervals_extreme_value_closer_l313_313180

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * (x - 1)

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧
  (a > 0 → (∀ x : ℝ, x < Real.log a → f x a > f (x + 1) a) ∧ (∀ x : ℝ, x > Real.log a → f x a < f (x + 1) a)) :=
sorry

theorem extreme_value_closer (a : ℝ) :
  a > e - 1 →
  ∀ x : ℝ, x ≥ 1 → |Real.exp 1/x - Real.log x| < |Real.exp (x - 1) + a - Real.log x| :=
sorry

end monotonic_intervals_extreme_value_closer_l313_313180


namespace remainder_x5_1_x3_1_div_x2_x_1_l313_313646

theorem remainder_x5_1_x3_1_div_x2_x_1 :
  ∀ (x : ℂ), let poly := (x^5 - 1) * (x^3 - 1),
                 divisor := x^2 + x + 1,
                 remainder := x^2 + x + 1 in
  ∃ q : ℂ, poly = q * divisor + remainder :=
by
  intro x
  let poly := (x^5 - 1) * (x^3 - 1)
  let divisor := x^2 + x + 1
  let remainder := x^2 + x + 1
  use sorry
  rw [← add_assoc, ← mul_assoc, ← pow_succ]
  sorry

end remainder_x5_1_x3_1_div_x2_x_1_l313_313646


namespace fraction_to_decimal_l313_313902

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313902


namespace find_k_l313_313985

-- Define the vectors a and b
def a := (3, 1) : ℝ × ℝ
def b := (1, 0) : ℝ × ℝ

-- Definition of c in terms of a and b with scalar k
def c (k : ℝ) := (a.fst + k * b.fst, a.snd + k * b.snd)

-- Dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

-- Statement of the problem, given the conditions, solve for k
theorem find_k (k : ℝ) (h : dot_product a (c k) = 0) : k = -10 / 3 := by
  sorry

end find_k_l313_313985


namespace total_wait_days_l313_313543

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end total_wait_days_l313_313543


namespace cubic_root_form_addition_l313_313098

theorem cubic_root_form_addition (p q r : ℕ) 
(h_root_form : ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧ x = (p^(1/3) + q^(1/3) + 2) / r) : 
  p + q + r = 10 :=
sorry

end cubic_root_form_addition_l313_313098


namespace wrongly_read_number_l313_313580

theorem wrongly_read_number 
  (S_initial : ℕ) (S_correct : ℕ) (correct_num : ℕ) (num_count : ℕ) 
  (h_initial : S_initial = num_count * 18) 
  (h_correct : S_correct = num_count * 19) 
  (h_correct_num : correct_num = 36) 
  (h_diff : S_correct - S_initial = correct_num - wrong_num) 
  (h_num_count : num_count = 10) 
  : wrong_num = 26 :=
sorry

end wrongly_read_number_l313_313580


namespace ratio_of_inquisitive_tourist_l313_313021

theorem ratio_of_inquisitive_tourist (questions_per_tourist : ℕ)
                                     (num_group1 : ℕ) (num_group2 : ℕ) (num_group3 : ℕ) (num_group4 : ℕ)
                                     (total_questions : ℕ) 
                                     (inquisitive_tourist_questions : ℕ) :
  questions_per_tourist = 2 ∧ 
  num_group1 = 6 ∧ 
  num_group2 = 11 ∧ 
  num_group3 = 8 ∧ 
  num_group4 = 7 ∧ 
  total_questions = 68 ∧ 
  inquisitive_tourist_questions = (total_questions - (num_group1 * questions_per_tourist + num_group2 * questions_per_tourist +
                                                        (num_group3 - 1) * questions_per_tourist + num_group4 * questions_per_tourist)) →
  (inquisitive_tourist_questions : ℕ) / questions_per_tourist = 3 :=
by sorry

end ratio_of_inquisitive_tourist_l313_313021


namespace geometric_sequence_product_l313_313197

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n, a (n + 1) = a n * r

noncomputable def quadratic_roots (a1 a10 : ℝ) : Prop :=
3 * a1^2 - 2 * a1 - 6 = 0 ∧ 3 * a10^2 - 2 * a10 - 6 = 0

theorem geometric_sequence_product {a : ℕ → ℝ}
  (h_geom : geometric_sequence a)
  (h_roots : quadratic_roots (a 1) (a 10)) :
  a 4 * a 7 = -2 :=
sorry

end geometric_sequence_product_l313_313197


namespace max_value_abs_expression_l313_313773

noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

theorem max_value_abs_expression (x y : ℝ) (h : circle_eq x y) : 
  ∃ t : ℝ, |3 * x + 4 * y - 3| = t ∧ t ≤ 8 :=
sorry

end max_value_abs_expression_l313_313773


namespace percentage_water_fresh_fruit_l313_313458

-- Definitions of the conditions
def weight_dried_fruit : ℝ := 12
def water_content_dried_fruit : ℝ := 0.15
def weight_fresh_fruit : ℝ := 101.99999999999999

-- Derived definitions based on the conditions
def weight_non_water_dried_fruit : ℝ := weight_dried_fruit - (water_content_dried_fruit * weight_dried_fruit)
def weight_non_water_fresh_fruit : ℝ := weight_non_water_dried_fruit
def weight_water_fresh_fruit : ℝ := weight_fresh_fruit - weight_non_water_fresh_fruit

-- Proof statement
theorem percentage_water_fresh_fruit :
  (weight_water_fresh_fruit / weight_fresh_fruit) * 100 = 90 :=
sorry

end percentage_water_fresh_fruit_l313_313458


namespace cars_on_river_road_l313_313121

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 60) (h2 : B * 13 = C) : C = 65 :=
sorry

end cars_on_river_road_l313_313121


namespace number_of_pounds_of_vegetables_l313_313437

-- Defining the conditions
def beef_cost_per_pound : ℕ := 6  -- Beef costs $6 per pound
def vegetable_cost_per_pound : ℕ := 2  -- Vegetables cost $2 per pound
def beef_pounds : ℕ := 4  -- Troy buys 4 pounds of beef
def total_cost : ℕ := 36  -- The total cost of everything is $36

-- Prove the number of pounds of vegetables Troy buys is 6
theorem number_of_pounds_of_vegetables (V : ℕ) :
  beef_cost_per_pound * beef_pounds + vegetable_cost_per_pound * V = total_cost → V = 6 :=
by
  sorry  -- Proof to be filled in later

end number_of_pounds_of_vegetables_l313_313437


namespace articles_production_l313_313625

theorem articles_production (x y : ℕ) (e : ℝ) :
  (x * x * x * e / x = x^2 * e) → (y * (y + 2) * y * (e / x) = (e * y * (y^2 + 2 * y)) / x) :=
by 
  sorry

end articles_production_l313_313625


namespace plane_equation_l313_313486

theorem plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) ∧
  (∀ x y z : ℤ, 
    (A * x + B * y + C * z + D = 0) ↔
      (x = 1 ∧ y = 6 ∧ z = -8 ∨ (∃ t : ℤ, 
        x = 2 + 4 * t ∧ y = 4 - t ∧ z = -3 + 5 * t))) ∧
  (A = 5 ∧ B = 15 ∧ C = -7 ∧ D = -151) :=
sorry

end plane_equation_l313_313486


namespace compute_f_at_2012_l313_313389

noncomputable def B := { x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 2 }

noncomputable def h (x : ℚ) : ℚ := 2 - (1 / x)

noncomputable def f (x : B) : ℝ := sorry  -- As a placeholder since the definition isn't given directly

-- Main theorem
theorem compute_f_at_2012 : 
  (∀ x : B, f x + f ⟨h x, sorry⟩ = Real.log (abs (2 * (x : ℚ)))) →
  f ⟨2012, sorry⟩ = Real.log ((4024 : ℚ) / (4023 : ℚ)) :=
sorry

end compute_f_at_2012_l313_313389


namespace points_on_line_l313_313559

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313559


namespace coeff_x2_expansion_l313_313703

theorem coeff_x2_expansion (n r : ℕ) (a b : ℤ) :
  n = 5 → a = 1 → b = 2 → r = 2 →
  (Nat.choose n r) * (a^(n - r)) * (b^r) = 40 :=
by
  intros Hn Ha Hb Hr
  rw [Hn, Ha, Hb, Hr]
  simp
  sorry

end coeff_x2_expansion_l313_313703


namespace largest_pot_cost_l313_313808

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ :=
  x + 5 * 0.15

theorem largest_pot_cost :
  ∃ (x : ℝ), (6 * x + 5 * 0.15 + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15) = 8.85) →
    cost_of_largest_pot x = 1.85 :=
by
  sorry

end largest_pot_cost_l313_313808


namespace total_shaded_area_l313_313863

theorem total_shaded_area
  (carpet_side : ℝ)
  (large_square_side : ℝ)
  (small_square_side : ℝ)
  (ratio_large : carpet_side / large_square_side = 4)
  (ratio_small : large_square_side / small_square_side = 2) : 
  (1 * large_square_side^2 + 12 * small_square_side^2 = 64) := 
by 
  sorry

end total_shaded_area_l313_313863


namespace count_positive_integers_l313_313053

theorem count_positive_integers (n : ℤ) : 
  (130 * n) ^ 50 > (n : ℤ) ^ 100 ∧ (n : ℤ) ^ 100 > 2 ^ 200 → 
  ∃ k : ℕ, k = 125 := sorry

end count_positive_integers_l313_313053


namespace fraction_of_time_riding_at_15mph_l313_313792

variable (t_5 t_15 : ℝ)

-- Conditions
def no_stops : Prop := (t_5 ≠ 0 ∧ t_15 ≠ 0)
def average_speed (t_5 t_15 : ℝ) : Prop := (5 * t_5 + 15 * t_15) / (t_5 + t_15) = 10

-- Question to be proved
theorem fraction_of_time_riding_at_15mph (h1 : no_stops t_5 t_15) (h2 : average_speed t_5 t_15) :
  t_15 / (t_5 + t_15) = 1 / 2 :=
sorry

end fraction_of_time_riding_at_15mph_l313_313792


namespace total_distance_traveled_l313_313155

noncomputable def travel_distance : ℝ :=
  1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640

theorem total_distance_traveled :
  let n := 8
  let r := 40
  let theta := 2 * Real.pi / n
  let d_2arcs := 2 * r * Real.sin (theta)
  let d_3arcs := r * (2 + Real.sqrt (2))
  let d_4arcs := 2 * r
  (8 * (4 * d_2arcs + 2 * d_3arcs + d_4arcs)) = travel_distance := by
  sorry

end total_distance_traveled_l313_313155


namespace avg_cost_equals_0_22_l313_313737

-- Definitions based on conditions
def num_pencils : ℕ := 150
def cost_pencils : ℝ := 24.75
def shipping_cost : ℝ := 8.50

-- Calculating total cost and average cost
noncomputable def total_cost : ℝ := cost_pencils + shipping_cost
noncomputable def avg_cost_per_pencil : ℝ := total_cost / num_pencils

-- Lean theorem statement
theorem avg_cost_equals_0_22 : avg_cost_per_pencil = 0.22 :=
by
  sorry

end avg_cost_equals_0_22_l313_313737


namespace triangle_relations_l313_313525

theorem triangle_relations (A B C_1 C_2 C_3 : ℝ)
  (h1 : B > A)
  (h2 : C_2 > C_1 ∧ C_2 > C_3)
  (h3 : A + C_1 = 90) 
  (h4 : C_2 = 90)
  (h5 : B + C_3 = 90) :
  C_1 - C_3 = B - A :=
sorry

end triangle_relations_l313_313525


namespace ratio_of_shares_l313_313139

-- Definitions for the given conditions
def capital_A : ℕ := 4500
def capital_B : ℕ := 16200
def months_A : ℕ := 12
def months_B : ℕ := 5 -- B joined after 7 months

-- Effective capital contributions
def effective_capital_A : ℕ := capital_A * months_A
def effective_capital_B : ℕ := capital_B * months_B

-- Defining the statement to prove
theorem ratio_of_shares : effective_capital_A / Nat.gcd effective_capital_A effective_capital_B = 2 ∧ effective_capital_B / Nat.gcd effective_capital_A effective_capital_B = 3 := by
  sorry

end ratio_of_shares_l313_313139


namespace mod_product_eq_15_l313_313575

theorem mod_product_eq_15 :
  (15 * 24 * 14) % 25 = 15 :=
by
  sorry

end mod_product_eq_15_l313_313575


namespace center_incircle_on_line_MN_l313_313078

-- Definitions and assumptions from problem conditions
variables {A B C D M N : Point}
variables {AB CD : Line}
variables Incircle_ABC : Circle

-- Cyclic trapezoid ABCD, AB parallel CD, and AB > CD
axioms
  (cyclic_ABCD : CyclicQuadrilateral A B C D)
  (parallel_AB_CD : Parallel AB CD)
  (AB_greater_CD : AB > CD)

-- The incircle of triangle ABC is tangent to AB and AC at points M and N respectively
axioms
  (tangent_Incircle_AB_M : TangentPoint Incircle_ABC AB M)
  (tangent_Incircle_AC_N : TangentPoint Incircle_ABC AC N)

-- Center of the incircle of ABCD lies on line MN
theorem center_incircle_on_line_MN :
  Center (Incircle ABCD) ∈ Line M N :=
sorry

end center_incircle_on_line_MN_l313_313078


namespace each_person_towel_day_l313_313616

def total_people (families : ℕ) (members_per_family : ℕ) : ℕ :=
  families * members_per_family

def total_towels (loads : ℕ) (towels_per_load : ℕ) : ℕ :=
  loads * towels_per_load

def towels_per_day (total_towels : ℕ) (days : ℕ) : ℕ :=
  total_towels / days

def towels_per_person_per_day (towels_per_day : ℕ) (total_people : ℕ) : ℕ :=
  towels_per_day / total_people

theorem each_person_towel_day
  (families : ℕ) (members_per_family : ℕ) (days : ℕ) (loads : ℕ) (towels_per_load : ℕ)
  (h_family : families = 3) (h_members : members_per_family = 4) (h_days : days = 7)
  (h_loads : loads = 6) (h_towels_per_load : towels_per_load = 14) :
  towels_per_person_per_day (towels_per_day (total_towels loads towels_per_load) days) (total_people families members_per_family) = 1 :=
by {
  -- Import necessary assumptions
  sorry
}

end each_person_towel_day_l313_313616


namespace b_divisible_by_a_l313_313663

theorem b_divisible_by_a (a b c : ℕ) (ha : a > 1) (hbc : b > c ∧ c > 1) (hdiv : (abc + 1) % (ab - b + 1) = 0) : a ∣ b :=
  sorry

end b_divisible_by_a_l313_313663


namespace total_grains_in_grey_regions_l313_313285

def total_grains_circle1 : ℕ := 87
def total_grains_circle2 : ℕ := 110
def white_grains_circle1 : ℕ := 68
def white_grains_circle2 : ℕ := 68

theorem total_grains_in_grey_regions : total_grains_circle1 - white_grains_circle1 + (total_grains_circle2 - white_grains_circle2) = 61 :=
by
  sorry

end total_grains_in_grey_regions_l313_313285


namespace technician_round_trip_percentage_l313_313608

theorem technician_round_trip_percentage (D: ℝ) (hD: D ≠ 0): 
  let round_trip_distance := 2 * D
  let distance_to_center := D
  let distance_back_10_percent := 0.10 * D
  let total_distance_completed := distance_to_center + distance_back_10_percent
  let percentage_completed := (total_distance_completed / round_trip_distance) * 100
  percentage_completed = 55 := 
by
  simp
  sorry -- Proof is not required per instructions

end technician_round_trip_percentage_l313_313608


namespace percent_millet_mix_correct_l313_313289

-- Define the necessary percentages
def percent_BrandA_in_mix : ℝ := 0.60
def percent_BrandB_in_mix : ℝ := 0.40
def percent_millet_in_BrandA : ℝ := 0.60
def percent_millet_in_BrandB : ℝ := 0.65

-- Define the overall percentage of millet in the mix
def percent_millet_in_mix : ℝ :=
  percent_BrandA_in_mix * percent_millet_in_BrandA +
  percent_BrandB_in_mix * percent_millet_in_BrandB

-- State the theorem
theorem percent_millet_mix_correct :
  percent_millet_in_mix = 0.62 :=
  by
    -- Here, we would provide the proof, but we use sorry as instructed.
    sorry

end percent_millet_mix_correct_l313_313289


namespace find_antecedent_l313_313067

-- Condition: The ratio is 4:6, simplified to 2:3
def ratio (a b : ℕ) : Prop := (a / gcd a b) = 2 ∧ (b / gcd a b) = 3

-- Condition: The consequent is 30
def consequent (y : ℕ) : Prop := y = 30

-- The problem is to find the antecedent
def antecedent (x : ℕ) (y : ℕ) : Prop := ratio x y

-- The theorem to be proved
theorem find_antecedent:
  ∃ x : ℕ, consequent 30 → antecedent x 30 ∧ x = 20 :=
by
  sorry

end find_antecedent_l313_313067


namespace convert_fraction_to_decimal_l313_313947

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313947


namespace fuel_consumption_gallons_l313_313457

theorem fuel_consumption_gallons
  (distance_per_liter : ℝ)
  (speed_mph : ℝ)
  (time_hours : ℝ)
  (mile_to_km : ℝ)
  (gallon_to_liters : ℝ)
  (fuel_consumption : ℝ) :
  distance_per_liter = 56 →
  speed_mph = 91 →
  time_hours = 5.7 →
  mile_to_km = 1.6 →
  gallon_to_liters = 3.8 →
  fuel_consumption = 3.9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end fuel_consumption_gallons_l313_313457


namespace complex_imaginary_part_l313_313162

theorem complex_imaginary_part : 
  Complex.im ((1 : ℂ) / (-2 + Complex.I) + (1 : ℂ) / (1 - 2 * Complex.I)) = 1/5 := 
  sorry

end complex_imaginary_part_l313_313162


namespace goose_eggs_count_l313_313450

theorem goose_eggs_count (E : ℝ) (h1 : 1 / 4 * E = (1 / 4) * E)
  (h2 : 4 / 5 * (1 / 4) * E = (4 / 5) * (1 / 4) * E)
  (h3 : 3 / 5 * (4 / 5) * (1 / 4) * E = 120)
  (h4 : 120 = 120)
  : E = 800 :=
by
  sorry

end goose_eggs_count_l313_313450


namespace grandma_red_bacon_bits_l313_313470

def mushrooms := 3
def cherry_tomatoes := 2 * mushrooms
def pickles := 4 * cherry_tomatoes
def bacon_bits := 4 * pickles
def red_bacon_bits := bacon_bits / 3

theorem grandma_red_bacon_bits : red_bacon_bits = 32 := by
  sorry

end grandma_red_bacon_bits_l313_313470


namespace contrapositive_of_square_comparison_l313_313582

theorem contrapositive_of_square_comparison (x y : ℝ) : (x^2 > y^2 → x > y) → (x ≤ y → x^2 ≤ y^2) :=
  by sorry

end contrapositive_of_square_comparison_l313_313582


namespace total_length_of_sticks_l313_313270

-- Definitions based on conditions
def num_sticks := 30
def length_per_stick := 25
def overlap := 6
def effective_length_per_stick := length_per_stick - overlap

-- Theorem statement
theorem total_length_of_sticks : num_sticks * effective_length_per_stick - effective_length_per_stick + length_per_stick = 576 := sorry

end total_length_of_sticks_l313_313270


namespace fraction_equals_decimal_l313_313937

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313937


namespace complex_number_solution_l313_313511

def imaginary_unit : ℂ := Complex.I -- defining the imaginary unit

theorem complex_number_solution (z : ℂ) (h : z / (z - imaginary_unit) = imaginary_unit) :
  z = (1 / 2 : ℂ) + (1 / 2 : ℂ) * imaginary_unit :=
sorry

end complex_number_solution_l313_313511


namespace value_of_v_star_star_l313_313970

noncomputable def v_star (v : ℝ) : ℝ :=
  v - v / 3
  
theorem value_of_v_star_star (v : ℝ) (h : v = 8.999999999999998) : v_star (v_star v) = 4.000000000000000 := by
  sorry

end value_of_v_star_star_l313_313970


namespace obtuse_and_acute_angles_in_convex_octagon_l313_313065

theorem obtuse_and_acute_angles_in_convex_octagon (m n : ℕ) (h₀ : n + m = 8) : m > n :=
sorry

end obtuse_and_acute_angles_in_convex_octagon_l313_313065


namespace multiplication_333_111_l313_313547

theorem multiplication_333_111: 333 * 111 = 36963 := 
by 
sorry

end multiplication_333_111_l313_313547


namespace solve_for_x_l313_313029

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end solve_for_x_l313_313029


namespace negation_of_universal_statement_l313_313706

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_statement_l313_313706


namespace derivative_of_f_l313_313046

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 * x - 2 :=
by
  intro x
  -- proof skipped
  sorry

end derivative_of_f_l313_313046


namespace find_age_of_b_l313_313448

-- Definitions for the conditions
def is_two_years_older (a b : ℕ) : Prop := a = b + 2
def is_twice_as_old (b c : ℕ) : Prop := b = 2 * c
def total_age (a b c : ℕ) : Prop := a + b + c = 12

-- Proof statement
theorem find_age_of_b (a b c : ℕ) 
  (h1 : is_two_years_older a b) 
  (h2 : is_twice_as_old b c) 
  (h3 : total_age a b c) : 
  b = 4 := 
by 
  sorry

end find_age_of_b_l313_313448


namespace fraction_to_decimal_l313_313899

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313899


namespace net_income_calculation_l313_313526

-- Definitions based on conditions
def rent_per_hour := 20
def monday_hours := 8
def wednesday_hours := 8
def friday_hours := 6
def sunday_hours := 5
def maintenance_cost := 35
def insurance_fee := 15
def rental_days := 4

-- Derived values based on conditions
def total_income_per_week :=
  (monday_hours + wednesday_hours) * rent_per_hour * 2 + 
  friday_hours * rent_per_hour + 
  sunday_hours * rent_per_hour

def total_expenses_per_week :=
  maintenance_cost + 
  insurance_fee * rental_days

def net_income_per_week := 
  total_income_per_week - total_expenses_per_week

-- The final proof statement
theorem net_income_calculation : net_income_per_week = 445 := by
  sorry

end net_income_calculation_l313_313526


namespace proof_problem_l313_313399

theorem proof_problem (x y z : ℝ) (h₁ : x ≠ y) 
  (h₂ : (x^2 - y*z) / (x * (1 - y*z)) = (y^2 - x*z) / (y * (1 - x*z))) :
  x + y + z = 1/x + 1/y + 1/z :=
sorry

end proof_problem_l313_313399


namespace TeamC_fee_l313_313023

structure Team :=
(work_rate : ℚ)

def teamA : Team := ⟨1 / 36⟩
def teamB : Team := ⟨1 / 24⟩
def teamC : Team := ⟨1 / 18⟩

def total_fee : ℚ := 36000

def combined_work_rate_first_half (A B C : Team) : ℚ :=
(A.work_rate + B.work_rate + C.work_rate) * 1 / 2

def combined_work_rate_second_half (A C : Team) : ℚ :=
(A.work_rate + C.work_rate) * 1 / 2

def total_work_completed_by_TeamC (A B C : Team) : ℚ :=
C.work_rate * combined_work_rate_first_half A B C + C.work_rate * combined_work_rate_second_half A C

theorem TeamC_fee (A B C : Team) (total_fee : ℚ) :
  total_work_completed_by_TeamC A B C * total_fee = 20000 :=
by
  sorry

end TeamC_fee_l313_313023


namespace geom_seq_common_ratio_l313_313321

theorem geom_seq_common_ratio (a₁ a₂ a₃ a₄ q : ℝ) 
  (h1 : a₁ + a₄ = 18)
  (h2 : a₂ * a₃ = 32)
  (h3 : a₂ = a₁ * q)
  (h4 : a₃ = a₁ * q^2)
  (h5 : a₄ = a₁ * q^3) : 
  q = 2 ∨ q = (1 / 2) :=
by {
  sorry
}

end geom_seq_common_ratio_l313_313321


namespace quadratic_trinomial_int_l313_313966

theorem quadratic_trinomial_int (a b c x : ℤ) (h : y = (x - a) * (x - 6) + 1) :
  ∃ (b c : ℤ), (x + b) * (x + c) = (x - 8) * (x - 6) + 1 :=
by
  sorry

end quadratic_trinomial_int_l313_313966


namespace max_candy_one_student_l313_313128

theorem max_candy_one_student (n : ℕ) (mu : ℕ) (at_least_two : ℕ → Prop) :
  n = 35 → mu = 6 →
  (∀ x, at_least_two x → x ≥ 2) →
  ∃ max_candy : ℕ, (∀ x, at_least_two x → x ≤ max_candy) ∧ max_candy = 142 :=
by
sorry

end max_candy_one_student_l313_313128


namespace sum_of_cubes_l313_313327

variable (a b c : ℝ)

theorem sum_of_cubes (h1 : a^2 + 3 * b = 2) (h2 : b^2 + 5 * c = 3) (h3 : c^2 + 7 * a = 6) :
  a^3 + b^3 + c^3 = -0.875 :=
by
  sorry

end sum_of_cubes_l313_313327


namespace nuts_in_tree_l313_313596

theorem nuts_in_tree (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : squirrels = nuts + 2) : nuts = 2 :=
by
  sorry

end nuts_in_tree_l313_313596


namespace fraction_to_decimal_l313_313910

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313910


namespace Jerry_travel_time_l313_313793

theorem Jerry_travel_time
  (speed_j speed_b distance_j distance_b time_j time_b : ℝ)
  (h_speed_j : speed_j = 40)
  (h_speed_b : speed_b = 30)
  (h_distance_b : distance_b = distance_j + 5)
  (h_time_b : time_b = time_j + 1/3)
  (h_distance_j : distance_j = speed_j * time_j)
  (h_distance_b_eq : distance_b = speed_b * time_b) :
  time_j = 1/2 :=
by
  sorry

end Jerry_travel_time_l313_313793


namespace convert_fraction_to_decimal_l313_313953

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313953


namespace simplify_expression_l313_313241

theorem simplify_expression (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 :=
by
  -- We acknowledge this is the placeholder for the proof.
  -- This statement follows directly from the difference of squares identity.
  sorry

end simplify_expression_l313_313241


namespace money_left_after_purchase_l313_313817

-- The costs and amounts for each item
def bread_cost : ℝ := 2.35
def num_bread : ℝ := 4
def peanut_butter_cost : ℝ := 3.10
def num_peanut_butter : ℝ := 2
def honey_cost : ℝ := 4.50
def num_honey : ℝ := 1

-- The coupon discount and budget
def coupon_discount : ℝ := 2
def budget : ℝ := 20

-- Calculate the total cost before applying the coupon
def total_before_coupon : ℝ := num_bread * bread_cost + num_peanut_butter * peanut_butter_cost + num_honey * honey_cost

-- Calculate the total cost after applying the coupon
def total_after_coupon : ℝ := total_before_coupon - coupon_discount

-- Calculate the money left over after the purchase
def money_left_over : ℝ := budget - total_after_coupon

-- The theorem to be proven
theorem money_left_after_purchase : money_left_over = 1.90 :=
by
  -- The proof of this theorem will involve the specific calculations and will be filled in later
  sorry

end money_left_after_purchase_l313_313817


namespace find_k_of_vectors_orthogonal_l313_313986

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l313_313986


namespace tangent_product_l313_313624

-- Declarations for circles, points of tangency, and radii
variables (R r : ℝ) -- radii of the circles
variables (A B C : ℝ) -- distances related to the tangents

-- Conditions: Two circles, a common internal tangent intersecting at points A and B, tangent at point C
axiom tangent_conditions : A * B = R * r

-- Problem statement: Prove that A * C * C * B = R * r
theorem tangent_product (R r A B C : ℝ) (h : A * B = R * r) : A * C * C * B = R * r :=
by
  sorry

end tangent_product_l313_313624


namespace num_factors_1320_l313_313352

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l313_313352


namespace total_area_pool_and_deck_l313_313631

theorem total_area_pool_and_deck (pool_length pool_width deck_width : ℕ) 
  (h1 : pool_length = 12) 
  (h2 : pool_width = 10) 
  (h3 : deck_width = 4) : 
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = 360 := 
by sorry

end total_area_pool_and_deck_l313_313631


namespace martha_cakes_l313_313229

theorem martha_cakes :
  ∀ (n : ℕ), (∀ (c : ℕ), c = 3 → (∀ (k : ℕ), k = 6 → n = c * k)) → n = 18 :=
by
  intros n h
  specialize h 3 rfl 6 rfl
  exact h

end martha_cakes_l313_313229


namespace distinct_factors_1320_l313_313346

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l313_313346


namespace age_of_fourth_child_l313_313096

theorem age_of_fourth_child 
  (avg_age : ℕ) 
  (age1 age2 age3 : ℕ) 
  (age4 : ℕ)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg_age) 
  (h1 : age1 = 6) 
  (h2 : age2 = 8) 
  (h3 : age3 = 11) 
  (h_avg_val : avg_age = 9) : 
  age4 = 11 := 
by 
  sorry

end age_of_fourth_child_l313_313096


namespace find_x_l313_313028

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end find_x_l313_313028


namespace arithmetic_mean_of_a_and_b_is_sqrt3_l313_313493

theorem arithmetic_mean_of_a_and_b_is_sqrt3 :
  let a := (Real.sqrt 3 + Real.sqrt 2)
  let b := (Real.sqrt 3 - Real.sqrt 2)
  (a + b) / 2 = Real.sqrt 3 := 
by
  sorry

end arithmetic_mean_of_a_and_b_is_sqrt3_l313_313493


namespace fraction_to_decimal_l313_313897

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313897


namespace gcd_2183_1947_l313_313968

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := 
by 
  sorry

end gcd_2183_1947_l313_313968


namespace average_percentage_decrease_l313_313002

theorem average_percentage_decrease (x : ℝ) : 60 * (1 - x) * (1 - x) = 48.6 → x = 0.1 :=
by sorry

end average_percentage_decrease_l313_313002


namespace sin_value_of_arithmetic_sequence_l313_313490

open Real

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sin_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : arithmetic_sequence a) 
  (h_cond : a 1 + a 5 + a 9 = 5 * π) : 
  sin (a 2 + a 8) = - (sqrt 3 / 2) :=
by
  sorry

end sin_value_of_arithmetic_sequence_l313_313490


namespace fraction_to_decimal_l313_313911

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313911


namespace distinct_positive_factors_of_1320_l313_313357

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l313_313357


namespace rain_over_weekend_l313_313255

open ProbabilityTheory

theorem rain_over_weekend (P_R_S : ℚ)
  (P_R_Sn : ℚ)
  (P_R_Sn_given_no_R_S : ℚ) :
  P_R_S = 0.60 →
  P_R_Sn = 0.40 →
  P_R_Sn_given_no_R_S = 0.70 →
  let P_no_R_S := 1 - P_R_S in
  let P_no_R_Sn_given_no_R_S := 1 - P_R_Sn_given_no_R_S in
  let P_no_R_Sn_given_R_S := 1 - P_R_Sn in
  1 - (P_no_R_S * P_no_R_Sn_given_no_R_S) - (P_R_S * P_no_R_Sn_given_R_S) = 0.88 :=
begin
  intros h1 h2 h3,
  let P_no_R_S := 1 - P_R_S,
  let P_no_R_Sn_given_no_R_S := 1 - P_R_Sn_given_no_R_S,
  let P_no_R_Sn_given_R_S := 1 - P_R_Sn,
  have h4 : 1 - (P_no_R_S * P_no_R_Sn_given_no_R_S) - (P_R_S * P_no_R_Sn_given_R_S) = 0.88,
  { simp [h1, h2, h3, P_no_R_S, P_no_R_Sn_given_no_R_S, P_no_R_Sn_given_R_S],
    norm_num },
  exact h4,
end

end rain_over_weekend_l313_313255


namespace eq_satisfied_in_entire_space_l313_313117

theorem eq_satisfied_in_entire_space (x y z : ℝ) : 
  (x + y + z)^2 = x^2 + y^2 + z^2 ↔ xy + xz + yz = 0 :=
by
  sorry

end eq_satisfied_in_entire_space_l313_313117


namespace paper_clips_distribution_l313_313227

theorem paper_clips_distribution (total_clips : ℕ) (num_boxes : ℕ) (clip_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : num_boxes = 9) : clip_per_box = 9 :=
by sorry

end paper_clips_distribution_l313_313227


namespace rainfall_ratio_l313_313680

theorem rainfall_ratio (r_wed tuesday_rate : ℝ)
    (h_monday : 7 * 1 = 7)
    (h_tuesday : 4 * 2 = 8)
    (h_total : 7 + 8 + 2 * r_wed = 23)
    (h_wed_eq: r_wed = 8 / 2)
    (h_tuesday_rate: tuesday_rate = 2) 
    : r_wed / tuesday_rate = 2 :=
by
  sorry

end rainfall_ratio_l313_313680


namespace statue_original_cost_l313_313611

theorem statue_original_cost (selling_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : selling_price = 620) (h2 : profit_percent = 25) : 
  original_cost = 496 :=
by
  have h3 : profit_percent / 100 + 1 = 1.25 := by sorry
  have h4 : 1.25 * original_cost = selling_price := by sorry
  have h5 : original_cost = 620 / 1.25 := by sorry
  have h6 : 620 / 1.25 = 496 := by sorry
  exact sorry

end statue_original_cost_l313_313611


namespace simplify_expression_l313_313881

theorem simplify_expression :
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 125 / 13 :=
by
  sorry

end simplify_expression_l313_313881


namespace rationalize_denominator_l313_313407

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l313_313407


namespace initial_points_l313_313557

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l313_313557


namespace sally_initial_cards_l313_313409

theorem sally_initial_cards (X : ℕ) (h1 : X + 41 + 20 = 88) : X = 27 :=
by
  -- Proof goes here
  sorry

end sally_initial_cards_l313_313409


namespace solve_for_r_l313_313095

theorem solve_for_r (r : ℚ) (h : (r + 4) / (r - 3) = (r - 2) / (r + 2)) : r = -2/11 :=
by
  sorry

end solve_for_r_l313_313095


namespace part1_proof_part2_proof_part3_proof_part4_proof_l313_313678

variable {A B C : Type}
variables {a b c : ℝ}  -- Sides of the triangle
variables {h_a h_b h_c r r_a r_b r_c : ℝ}  -- Altitudes, inradius, and exradii of \triangle ABC

-- Part 1: Proving the sum of altitudes related to sides and inradius
theorem part1_proof : h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 2: Proving the sum of reciprocals of altitudes related to the reciprocal of inradius and exradii
theorem part2_proof : (1 / h_a) + (1 / h_b) + (1 / h_c) = 1 / r ∧ 1 / r = (1 / r_a) + (1 / r_b) + (1 / r_c) := sorry

-- Part 3: Combining results of parts 1 and 2 to prove product of sums
theorem part3_proof : (h_a + h_b + h_c) * ((1 / h_a) + (1 / h_b) + (1 / h_c)) = (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 4: Final geometric identity
theorem part4_proof : (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 := sorry

end part1_proof_part2_proof_part3_proof_part4_proof_l313_313678


namespace distinct_collections_l313_313549

def letters := ['M', 'M', 'A', 'A', 'A', 'T', 'T', 'H', 'E', 'I', 'C', 'L']

-- Define total ways to choose 3 vowels and 4 consonants such that T's, M's, and A's are indistinguishable.
def count_vowels (l : List Char) : Nat :=
  if l.count 'A' = 3 then 1
  else if l.count 'A' = 2 ∧ ('E' ∈ l ∨ 'I' ∈ l) then 2
  else if l.count 'A' = 1 ∧ ('E' ∈ l) ∧ ('I' ∈ l) then 1
  else 0

def list_combinations (l : List Char) (n : Nat) : List (List Char) :=
  -- some implementation that generates the list of combinations of length n from l
  sorry

def count_consonants (l : List Char) : Nat :=
  let cons := list_combinations (['T', 'T', 'M', 'M', 'H', 'C', 'L'] : List Char) 4 
  in  
  if l.count 'T' = 2 ∧ l.count 'M' = 2 then 1
  else if l.count 'T' = 2 ∧ l.count 'M' = 1 ∧ ('H' ∈ l ∨ 'C' ∈ l ∨ 'L' ∈ l) then 3
  else if l.count 'M' = 2 ∧ l.count 'T' = 1 ∧ ('H' ∈ l ∨ 'C' ∨ 'L' ∈ l) then 3
  else if l.count 'T' = 1 ∧ l.count 'M' = 1 ∧ (sum (fun x ↦ x ∈ l ∧ (x = 'H' ∨ x = 'C' ∨ x = 'L' )) 2 then 3
  else if l.count 'T' = 2 ∧ (sum (fun x ↦ x ∈ l ∧ (x = 'H' ∨ x = 'C' ∨ x = 'L' )) 2 then 3
  else if l.count 'M' = 2 ∧ (sum (fun x ↦ x ∈ l ∧ (x = 'H' ∨ x = 'C' ∨ x = 'L' )) 2 then 3
  else 0

def num_distinct_collections := (count_vowels letters) * (count_consonants letters)

theorem distinct_collections : num_distinct_collections = 64 :=
by {
  sorry
}

end distinct_collections_l313_313549


namespace arithmetic_sequence_sum_l313_313374

theorem arithmetic_sequence_sum (a_n : ℕ → ℝ) (h1 : a_n 1 + a_n 2 + a_n 3 + a_n 4 = 30) 
                               (h2 : a_n 1 + a_n 4 = a_n 2 + a_n 3) :
  a_n 2 + a_n 3 = 15 := 
by 
  sorry

end arithmetic_sequence_sum_l313_313374


namespace problem_l313_313529

def count_numbers_with_more_ones_than_zeros (n : ℕ) : ℕ :=
  -- function that counts numbers less than or equal to 'n'
  -- whose binary representation has more '1's than '0's
  sorry

theorem problem (M := count_numbers_with_more_ones_than_zeros 1500) : 
  M % 1000 = 884 :=
sorry

end problem_l313_313529


namespace estimate_students_correct_l313_313672

noncomputable def estimate_students_below_85 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ) : ℕ :=
if total_students = 50 ∧ mean_score = 90 ∧ prob_90_to_95 = 0.3 then 10 else 0

theorem estimate_students_correct 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ)
  (h1 : total_students = 50) 
  (h2 : mean_score = 90)
  (h3 : prob_90_to_95 = 0.3) : 
  estimate_students_below_85 total_students mean_score variance prob_90_to_95 = 10 :=
by
  sorry

end estimate_students_correct_l313_313672


namespace anna_and_bob_play_together_l313_313395

-- Definitions based on the conditions
def total_players := 12
def matches_per_week := 2
def players_per_match := 6
def anna_and_bob := 2
def other_players := total_players - anna_and_bob
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Lean statement based on the equivalent proof problem
theorem anna_and_bob_play_together :
  combination other_players (players_per_match - anna_and_bob) = 210 := by
  -- To use Binomial Theorem in Lean
  -- The mathematical equivalent is C(10, 4) = 210
  sorry

end anna_and_bob_play_together_l313_313395


namespace convert_fraction_to_decimal_l313_313951

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313951


namespace triangle_type_l313_313174

-- Let's define what it means for a triangle to be acute, obtuse, and right in terms of angle
def is_acute_triangle (a b c : ℝ) : Prop := (a < 90) ∧ (b < 90) ∧ (c < 90)
def is_obtuse_triangle (a b c : ℝ) : Prop := (a > 90) ∨ (b > 90) ∨ (c > 90)
def is_right_triangle (a b c : ℝ) : Prop := (a = 90) ∨ (b = 90) ∨ (c = 90)

-- The problem statement
theorem triangle_type (A B C : ℝ) (h : A = 100) : is_obtuse_triangle A B C :=
by {
  -- Sorry is used to indicate a placeholder for the proof
  sorry
}

end triangle_type_l313_313174


namespace probability_sum_16_l313_313024

open ProbabilityTheory 

-- Definitions of conditions
def fair_coin := {5, 15}
def fair_die := {1, 2, 3, 4, 5, 6}

-- Probability calculations
def probability_of_15_on_coin : ℝ := 1 / 2
def probability_of_1_on_die : ℝ := 1 / 6

-- Target statement
theorem probability_sum_16 :
  ∃ p : ℝ, p = probability_of_15_on_coin * probability_of_1_on_die ∧ p = 1 / 12 :=
begin
  use 1 / 12,
  split,
  { 
    exact mul_div_cancel' (by norm_num) (by norm_num),
  },
  { 
    refl,
  }
end

end probability_sum_16_l313_313024


namespace least_possible_value_of_z_minus_w_l313_313538

variable (x y z w k m : Int)
variable (h1 : Even x)
variable (h2 : Odd y)
variable (h3 : Odd z)
variable (h4 : ∃ n : Int, w = - (2 * n + 1) / 3)
variable (h5 : w < x)
variable (h6 : x < y)
variable (h7 : y < z)
variable (h8 : 0 < k)
variable (h9 : (y - x) > k)
variable (h10 : 0 < m)
variable (h11 : (z - w) > m)
variable (h12 : k > m)

theorem least_possible_value_of_z_minus_w
  : z - w = 6 := sorry

end least_possible_value_of_z_minus_w_l313_313538


namespace solution_inequality_equivalence_l313_313310

-- Define the inequality to be proved
def inequality (x : ℝ) : Prop :=
  (x + 1 / 2) * (3 / 2 - x) ≥ 0

-- Define the set of solutions such that -1/2 ≤ x ≤ 3/2
def solution_set (x : ℝ) : Prop :=
  -1 / 2 ≤ x ∧ x ≤ 3 / 2

-- The statement to be proved: the solution set of the inequality is {x | -1/2 ≤ x ≤ 3/2}
theorem solution_inequality_equivalence :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} :=
by 
  sorry

end solution_inequality_equivalence_l313_313310


namespace units_digit_G_1000_l313_313152

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 2 :=
by
  sorry

end units_digit_G_1000_l313_313152


namespace anya_game_losses_l313_313313

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l313_313313


namespace loan_duration_in_years_l313_313552

-- Define the conditions as constants
def carPrice : ℝ := 20000
def downPayment : ℝ := 5000
def monthlyPayment : ℝ := 250

-- Define the goal
theorem loan_duration_in_years :
  (carPrice - downPayment) / monthlyPayment / 12 = 5 := 
sorry

end loan_duration_in_years_l313_313552


namespace pests_eaten_by_frogs_in_week_l313_313461

-- Definitions
def pests_per_day_per_frog : ℕ := 80
def days_per_week : ℕ := 7
def number_of_frogs : ℕ := 5

-- Proposition to prove
theorem pests_eaten_by_frogs_in_week : (pests_per_day_per_frog * days_per_week * number_of_frogs) = 2800 := 
by sorry

end pests_eaten_by_frogs_in_week_l313_313461


namespace value_of_x_l313_313430

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l313_313430


namespace binary_10101_is_21_l313_313475

namespace BinaryToDecimal

def binary_to_decimal (n : Nat) : Nat :=
  match n with
  | 10101 => 21

theorem binary_10101_is_21 :
  binary_to_decimal 10101 = 21 := by
  -- Proof steps would go here
  sorry

end BinaryToDecimal

end binary_10101_is_21_l313_313475


namespace sqrt_9_eq_3_and_neg3_l313_313105

theorem sqrt_9_eq_3_and_neg3 : { x : ℝ | x^2 = 9 } = {3, -3} :=
by
  sorry

end sqrt_9_eq_3_and_neg3_l313_313105


namespace average_increase_l313_313435

theorem average_increase (x : ℝ) (y : ℝ) (h : y = 0.245 * x + 0.321) : 
  ∀ x_increase : ℝ, x_increase = 1 → (0.245 * (x + x_increase) + 0.321) - (0.245 * x + 0.321) = 0.245 :=
by
  intro x_increase
  intro hx
  rw [hx]
  simp
  sorry

end average_increase_l313_313435


namespace smallest_single_discount_l313_313373

noncomputable def discount1 : ℝ := (1 - 0.20) * (1 - 0.20)
noncomputable def discount2 : ℝ := (1 - 0.10) * (1 - 0.15)
noncomputable def discount3 : ℝ := (1 - 0.08) * (1 - 0.08) * (1 - 0.08)

theorem smallest_single_discount : ∃ n : ℕ, (1 - n / 100) < discount1 ∧ (1 - n / 100) < discount2 ∧ (1 - n / 100) < discount3 ∧ n = 37 := sorry

end smallest_single_discount_l313_313373


namespace sodas_total_l313_313006

def morning_sodas : ℕ := 77
def afternoon_sodas : ℕ := 19
def total_sodas : ℕ := morning_sodas + afternoon_sodas

theorem sodas_total :
  total_sodas = 96 :=
by
  sorry

end sodas_total_l313_313006


namespace points_on_line_l313_313561

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313561


namespace fraction_to_decimal_l313_313925

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313925


namespace total_cost_is_1_85_times_selling_price_l313_313692

def total_cost (P : ℝ) : ℝ := 140 * 2 * P + 90 * P

def loss (P : ℝ) : ℝ := 70 * 2 * P + 30 * P

def selling_price (P : ℝ) : ℝ := total_cost P - loss P

theorem total_cost_is_1_85_times_selling_price (P : ℝ) :
  total_cost P = 1.85 * selling_price P := by
  sorry

end total_cost_is_1_85_times_selling_price_l313_313692


namespace graph_of_equation_l313_313445

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_of_equation_l313_313445


namespace max_cubes_fit_in_box_l313_313440

theorem max_cubes_fit_in_box :
  ∀ (h w l : ℕ) (cube_vol box_max_cubes : ℕ),
    h = 12 → w = 8 → l = 9 → cube_vol = 27 → 
    box_max_cubes = (h * w * l) / cube_vol → box_max_cubes = 32 :=
by
  intros h w l cube_vol box_max_cubes h_def w_def l_def cube_vol_def box_max_cubes_def
  sorry

end max_cubes_fit_in_box_l313_313440


namespace range_of_a_l313_313779

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 2 < 0) ↔ (a^2 ≤ 8) :=
by
  sorry

end range_of_a_l313_313779


namespace resistor_problem_l313_313675

theorem resistor_problem 
  {x y r : ℝ}
  (h1 : 1 / r = 1 / x + 1 / y)
  (h2 : r = 2.9166666666666665)
  (h3 : y = 7) : 
  x = 5 :=
by
  sorry

end resistor_problem_l313_313675


namespace find_m_if_f_even_l313_313062

theorem find_m_if_f_even (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = x^4 + (m - 1) * x + 1) ∧ (∀ x : ℝ, f x = f (-x)) → m = 1 := 
by 
  sorry

end find_m_if_f_even_l313_313062


namespace find_function_l313_313979

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
  ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := 
by
  sorry

end find_function_l313_313979


namespace solution_set_inequality_l313_313177

open Set

variable {a b : ℝ}

/-- Proof Problem Statement -/
theorem solution_set_inequality (h : ∀ x : ℝ, -3 < x ∧ x < -1 ↔ a * x^2 - 1999 * x + b > 0) : 
  ∀ x : ℝ, 1 < x ∧ x < 3 ↔ a * x^2 + 1999 * x + b > 0 :=
sorry

end solution_set_inequality_l313_313177


namespace distinct_four_digit_count_l313_313339

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l313_313339


namespace large_number_exponent_l313_313443

theorem large_number_exponent (h : 10000 = 10 ^ 4) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := 
by
  sorry

end large_number_exponent_l313_313443


namespace imaginary_part_of_complex_number_l313_313969

open Complex

theorem imaginary_part_of_complex_number :
  ∀ (i : ℂ), i^2 = -1 → im ((2 * I) / (2 + I^3)) = 4 / 5 :=
by
  intro i hi
  sorry

end imaginary_part_of_complex_number_l313_313969


namespace fraction_to_decimal_l313_313928

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313928


namespace noah_has_largest_final_answer_l313_313539

def liam_initial := 15
def liam_final := (liam_initial - 2) * 3 + 3

def mia_initial := 15
def mia_final := (mia_initial * 3 - 4) + 3

def noah_initial := 15
def noah_final := ((noah_initial - 3) + 4) * 3

theorem noah_has_largest_final_answer : noah_final > liam_final ∧ noah_final > mia_final := by
  -- Placeholder for actual proof
  sorry

end noah_has_largest_final_answer_l313_313539


namespace cos_diff_symm_about_x_l313_313676

variable {α β : ℝ}

-- Conditions from the problem
def isSymmetricAboutX (α β : ℝ) : Prop :=
  (cos α = 1 / 4) ∧ (β = -α)

-- The proof statement
theorem cos_diff_symm_about_x (h : isSymmetricAboutX α β) : cos (α - β) = -7 / 8 :=
by sorry

end cos_diff_symm_about_x_l313_313676


namespace number_of_distinct_real_roots_l313_313978

theorem number_of_distinct_real_roots (f : ℝ → ℝ) (h : ∀ x, f x = |x| - (4 / x) - (3 * |x| / x)) : ∃ k, k = 1 :=
by
  sorry

end number_of_distinct_real_roots_l313_313978


namespace alpha_minus_beta_l313_313506

theorem alpha_minus_beta {α β : ℝ} (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
    (h_cos_alpha : Real.cos α = 2 * Real.sqrt 5 / 5) 
    (h_cos_beta : Real.cos β = Real.sqrt 10 / 10) : 
    α - β = -π / 4 := 
sorry

end alpha_minus_beta_l313_313506


namespace average_speed_interval_l313_313740

theorem average_speed_interval {s t : ℝ → ℝ} (h_eq : ∀ t, s t = t^2 + 1) : 
  (s 2 - s 1) / (2 - 1) = 3 :=
by
  sorry

end average_speed_interval_l313_313740


namespace simplify_expression_l313_313802

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end simplify_expression_l313_313802


namespace washing_machines_removed_l313_313372

theorem washing_machines_removed (crates boxes_per_crate washing_machines_per_box washing_machines_removed_per_box : ℕ) 
  (h_crates : crates = 10) (h_boxes_per_crate : boxes_per_crate = 6) 
  (h_washing_machines_per_box : washing_machines_per_box = 4) 
  (h_washing_machines_removed_per_box : washing_machines_removed_per_box = 1) :
  crates * boxes_per_crate * washing_machines_removed_per_box = 60 :=
by
  rw [h_crates, h_boxes_per_crate, h_washing_machines_removed_per_box]
  exact Nat.mul_assoc crates boxes_per_crate washing_machines_removed_per_box ▸
         Nat.mul_assoc 10 6 1 ▸ rfl


end washing_machines_removed_l313_313372


namespace x_intercept_of_line_l313_313835

theorem x_intercept_of_line : ∀ x y : ℝ, 2 * x + 3 * y = 6 → y = 0 → x = 3 :=
by
  intros x y h_line h_y_zero
  sorry

end x_intercept_of_line_l313_313835


namespace proportionate_enlargement_l313_313749

theorem proportionate_enlargement 
  (original_width original_height new_width : ℕ)
  (h_orig_width : original_width = 3)
  (h_orig_height : original_height = 2)
  (h_new_width : new_width = 12) : 
  ∃ (new_height : ℕ), new_height = 8 :=
by
  -- sorry to skip proof
  sorry

end proportionate_enlargement_l313_313749


namespace center_circle_is_correct_l313_313001

noncomputable def find_center_of_circle : ℝ × ℝ :=
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
  let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
  let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
  let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
  (-18, -11)

theorem center_circle_is_correct (x y : ℝ) :
  (let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
   let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
   let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
   let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
   (x, y) = find_center_of_circle) :=
  sorry

end center_circle_is_correct_l313_313001


namespace area_of_square_with_given_diagonal_l313_313138

theorem area_of_square_with_given_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : ∃ (A : ℝ), A = 64 :=
by
  use (8 * 8)
  sorry

end area_of_square_with_given_diagonal_l313_313138


namespace problem_solution_l313_313727

theorem problem_solution:
  2019 ^ Real.log (Real.log 2019) - Real.log 2019 ^ Real.log 2019 = 0 :=
by
  sorry

end problem_solution_l313_313727


namespace part_a_part_b_l313_313398

-- Part (a)
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^m > (1 + 1 / (n:ℝ))^n :=
by sorry

-- Part (b)
theorem part_b (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^(m + 1) < (1 + 1 / (n:ℝ))^(n + 1) :=
by sorry

end part_a_part_b_l313_313398


namespace martha_gingers_amount_l313_313690

theorem martha_gingers_amount (G : ℚ) (h : G = 0.43 * (G + 3)) : G = 2 := by
  sorry

end martha_gingers_amount_l313_313690


namespace triangle_inequality_a_triangle_inequality_b_l313_313384

variable (α β γ : ℝ)

-- Assume α, β, γ are angles of a triangle
def is_triangle (α β γ : ℝ) := 
  α + β + γ = π ∧ α > 0 ∧ β > 0 ∧ γ > 0

theorem triangle_inequality_a (h : is_triangle α β γ) :
  (1 - Real.cos α) * (1 - Real.cos β) * (1 - Real.cos γ) ≥ 
  (Real.cos α) * (Real.cos β) * (Real.cos γ) := sorry

theorem triangle_inequality_b (h : is_triangle α β γ) :
  12 * (Real.cos α) * (Real.cos β) * (Real.cos γ) ≤ 
  2 * (Real.cos α) * (Real.cos β) + 2 * (Real.cos α) * (Real.cos γ) + 2 * (Real.cos β) * (Real.cos γ) ∧
  2 * (Real.cos α) * (Real.cos β) + 2 * (Real.cos α) * (Real.cos γ) + 2 * (Real.cos β) * (Real.cos γ) ≤
  (Real.cos α) + (Real.cos β) + (Real.cos γ) := sorry

end triangle_inequality_a_triangle_inequality_b_l313_313384


namespace polynomial_remainder_is_zero_l313_313647

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end polynomial_remainder_is_zero_l313_313647


namespace chris_average_price_l313_313149

noncomputable def total_cost_dvd (price_per_dvd : ℝ) (num_dvds : ℕ) (discount : ℝ) : ℝ :=
  (price_per_dvd * (1 - discount)) * num_dvds

noncomputable def total_cost_bluray (price_per_bluray : ℝ) (num_blurays : ℕ) : ℝ :=
  price_per_bluray * num_blurays

noncomputable def total_cost_ultra_hd (price_per_ultra_hd : ℝ) (num_ultra_hds : ℕ) : ℝ :=
  price_per_ultra_hd * num_ultra_hds

noncomputable def total_cost (cost_dvd cost_bluray cost_ultra_hd : ℝ) : ℝ :=
  cost_dvd + cost_bluray + cost_ultra_hd

noncomputable def total_with_tax (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  total_cost * (1 + tax_rate)

noncomputable def average_price (total_with_tax : ℝ) (total_movies : ℕ) : ℝ :=
  total_with_tax / total_movies

theorem chris_average_price :
  let price_per_dvd := 15
  let num_dvds := 5
  let discount := 0.20
  let price_per_bluray := 20
  let num_blurays := 8
  let price_per_ultra_hd := 25
  let num_ultra_hds := 3
  let tax_rate := 0.10
  let total_movies := num_dvds + num_blurays + num_ultra_hds
  let cost_dvd := total_cost_dvd price_per_dvd num_dvds discount
  let cost_bluray := total_cost_bluray price_per_bluray num_blurays
  let cost_ultra_hd := total_cost_ultra_hd price_per_ultra_hd num_ultra_hds
  let pre_tax_total := total_cost cost_dvd cost_bluray cost_ultra_hd
  let total := total_with_tax pre_tax_total tax_rate
  average_price total total_movies = 20.28 :=
by
  -- substitute each definition one step at a time
  -- to show the average price exactly matches 20.28
  sorry

end chris_average_price_l313_313149


namespace average_age_increase_l313_313444

theorem average_age_increase
  (n : ℕ)
  (A : ℝ)
  (w : ℝ)
  (h1 : (n + 1) * (A + w) = n * A + 39)
  (h2 : (n + 1) * (A - 1) = n * A + 15)
  (hw : w = 7) :
  w = 7 := 
by
  sorry

end average_age_increase_l313_313444


namespace remainder_of_x_squared_mod_25_l313_313060

theorem remainder_of_x_squared_mod_25 :
  (5 * x ≡ 10 [MOD 25]) → (4 * x ≡ 20 [MOD 25]) → ((x ^ 2) % 25 = 4) := by
  intro h1 h2
  sorry

end remainder_of_x_squared_mod_25_l313_313060


namespace triangle_probability_l313_313047

open Classical

theorem triangle_probability :
  let a := 5
  let b := 6
  let lengths := [1, 2, 6, 11]
  let valid_third_side x := 1 < x ∧ x < 11
  let valid_lengths := lengths.filter valid_third_side
  let probability := valid_lengths.length / lengths.length
  probability = 1 / 2 :=
by {
  sorry
}

end triangle_probability_l313_313047


namespace rhombus_diagonal_length_l313_313775

theorem rhombus_diagonal_length (area d1 d2 : ℝ) (h₁ : area = 24) (h₂ : d1 = 8) (h₃ : area = (d1 * d2) / 2) : d2 = 6 := 
by sorry

end rhombus_diagonal_length_l313_313775


namespace systematic_sampling_eighth_group_number_l313_313854

theorem systematic_sampling_eighth_group_number (total_students groups students_per_group draw_lots_first : ℕ) 
  (h_total : total_students = 480)
  (h_groups : groups = 30)
  (h_students_per_group : students_per_group = 16)
  (h_draw_lots_first : draw_lots_first = 5) : 
  (8 - 1) * students_per_group + draw_lots_first = 117 :=
by
  sorry

end systematic_sampling_eighth_group_number_l313_313854


namespace reduced_price_l313_313134

variable (original_price : ℝ) (final_amount : ℝ)

noncomputable def sales_tax (price : ℝ) : ℝ :=
  if price <= 2500 then price * 0.04
  else if price <= 4500 then 2500 * 0.04 + (price - 2500) * 0.07
  else 2500 * 0.04 + 2000 * 0.07 + (price - 4500) * 0.09

noncomputable def discount (price : ℝ) : ℝ :=
  if price <= 2000 then price * 0.02
  else if price <= 4000 then 2000 * 0.02 + (price - 2000) * 0.05
  else 2000 * 0.02 + 2000 * 0.05 + (price - 4000) * 0.10

theorem reduced_price (P : ℝ) (original_price := 5000) (final_amount := 2468) :
  P = original_price - discount original_price + sales_tax original_price → P = 2423 :=
by
  sorry

end reduced_price_l313_313134


namespace range_of_a_l313_313221

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l313_313221


namespace total_output_correct_l313_313588

variable (a : ℝ)

-- Define a function that captures the total output from this year to the fifth year
def totalOutput (a : ℝ) : ℝ :=
  1.1 * a + (1.1 ^ 2) * a + (1.1 ^ 3) * a + (1.1 ^ 4) * a + (1.1 ^ 5) * a

theorem total_output_correct (a : ℝ) : 
  totalOutput a = 11 * (1.1 ^ 5 - 1) * a := by
  sorry

end total_output_correct_l313_313588


namespace abc_greater_than_n_l313_313576

theorem abc_greater_than_n
  (a b c n : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 1 < n)
  (h5 : a ^ n + b ^ n = c ^ n) :
  a > n ∧ b > n ∧ c > n :=
sorry

end abc_greater_than_n_l313_313576


namespace expand_and_count_nonzero_terms_l313_313300

theorem expand_and_count_nonzero_terms (x : ℝ) : 
  (x-3)*(3*x^2-2*x+6) + 2*(x^3 + x^2 - 4*x) = 5*x^3 - 9*x^2 + 4*x - 18 ∧ 
  (5 ≠ 0 ∧ -9 ≠ 0 ∧ 4 ≠ 0 ∧ -18 ≠ 0) :=
sorry

end expand_and_count_nonzero_terms_l313_313300


namespace parabola_focus_to_equation_l313_313843

-- Define the focus of the parabola
def F : (ℝ × ℝ) := (5, 0)

-- Define the standard equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 20 * x

-- State the problem in Lean
theorem parabola_focus_to_equation : 
  (F = (5, 0)) → ∀ x y, parabola_equation x y :=
by
  intro h_focus_eq
  sorry

end parabola_focus_to_equation_l313_313843


namespace points_on_line_initial_l313_313562

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l313_313562


namespace segment_length_abs_eq_cubrt_27_five_l313_313604

theorem segment_length_abs_eq_cubrt_27_five : 
  (∀ x : ℝ, |x - (3 : ℝ)| = 5) → (8 - (-2) = 10) :=
by 
  intros;
  sorry

end segment_length_abs_eq_cubrt_27_five_l313_313604


namespace circumference_circle_l313_313702

theorem circumference_circle {d r : ℝ} (h1 : ∀ (d r : ℝ), d = 2 * r) : 
  ∃ C : ℝ, C = π * d ∨ C = 2 * π * r :=
by {
  sorry
}

end circumference_circle_l313_313702


namespace tree_height_equation_l313_313378

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end tree_height_equation_l313_313378


namespace find_f_expression_l313_313771

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_expression (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  f (x) = (1 / (x - 1)) :=
by sorry

example (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) (hx: f (1 / x) = x / (1 - x)) :
  f x = 1 / (x - 1) :=
find_f_expression x h₀ h₁

end find_f_expression_l313_313771


namespace quilt_squares_count_l313_313635

theorem quilt_squares_count (total_squares : ℕ) (additional_squares : ℕ)
  (h1 : total_squares = 4 * additional_squares)
  (h2 : additional_squares = 24) :
  total_squares = 32 :=
by
  -- Proof would go here
  -- The proof would involve showing that total_squares indeed equals 32 given h1 and h2
  sorry

end quilt_squares_count_l313_313635


namespace refills_count_l313_313059

variable (spent : ℕ) (cost : ℕ)

theorem refills_count (h1 : spent = 40) (h2 : cost = 10) : spent / cost = 4 := 
by
  sorry

end refills_count_l313_313059


namespace bug_returns_eighth_move_l313_313734

def recurrence_relation (P : ℕ → ℚ) : Prop :=
∀ n, P (n + 1) = (2 / 3 : ℚ) - (1 / 3 : ℚ) * P n

def initial_condition (P : ℕ → ℚ) : Prop :=
P 0 = 1

theorem bug_returns_eighth_move :
  ∃ (P : ℕ → ℚ),
    recurrence_relation P ∧
    initial_condition P ∧
    P 8 = (3248 / 6561 : ℚ) ∧
    nat.coprime 3248 6561 ∧
    3248 + 6561 = 9809 :=
by
  sorry

end bug_returns_eighth_move_l313_313734


namespace convert_fraction_to_decimal_l313_313949

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313949


namespace area_of_walkways_is_214_l313_313697

-- Definitions for conditions
def width_of_flower_beds : ℕ := 2 * 7  -- two beds each 7 feet wide
def walkways_between_beds_width : ℕ := 3 * 2  -- three walkways each 2 feet wide (one on each side and one in between)
def total_width : ℕ := width_of_flower_beds + walkways_between_beds_width  -- Total width

def height_of_flower_beds : ℕ := 3 * 3  -- three rows of beds each 3 feet high
def walkways_between_beds_height : ℕ := 4 * 2  -- four walkways each 2 feet wide (one on each end and one between each row)
def total_height : ℕ := height_of_flower_beds + walkways_between_beds_height  -- Total height

def total_area_of_garden : ℕ := total_width * total_height  -- Total area of the garden including walkways

def area_of_one_flower_bed : ℕ := 7 * 3  -- Area of one flower bed
def total_area_of_flower_beds : ℕ := 6 * area_of_one_flower_bed  -- Total area of six flower beds

def total_area_walkways : ℕ := total_area_of_garden - total_area_of_flower_beds  -- Total area of the walkways

-- Theorem to prove the area of the walkways
theorem area_of_walkways_is_214 : total_area_walkways = 214 := sorry

end area_of_walkways_is_214_l313_313697


namespace log_sqrt_7_of_343sqrt7_l313_313756

noncomputable def log_sqrt_7 (y : ℝ) : ℝ := 
  Real.log y / Real.log (Real.sqrt 7)

theorem log_sqrt_7_of_343sqrt7 : log_sqrt_7 (343 * Real.sqrt 7) = 4 :=
by
  sorry

end log_sqrt_7_of_343sqrt7_l313_313756


namespace aba_div_by_7_l313_313710

theorem aba_div_by_7 (a b : ℕ) (h : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := 
sorry

end aba_div_by_7_l313_313710


namespace max_non_cyclic_handshakes_l313_313846

theorem max_non_cyclic_handshakes (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end max_non_cyclic_handshakes_l313_313846


namespace fraction_to_decimal_l313_313915

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313915


namespace range_of_m_l313_313192

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)
noncomputable def g (x m : ℝ) : ℝ := (1 / 2) * x - m

theorem range_of_m (h1 : ∀ x ∈ Set.Icc 0 3, f x ≥ 0)
                   (h2 : ∀ x ∈ Set.Icc 1 2, g x m ≤ (1 / 2) - m) :
  Set.Ici (1 / 2) ⊆ {m : ℝ | ∀ x ∈ Set.Icc 0 3, ∀ x' ∈ Set.Icc 1 2, f x ≥ g x' m } := 
by
  intros m hm
  sorry

end range_of_m_l313_313192


namespace present_age_of_son_is_22_l313_313858

theorem present_age_of_son_is_22 (S F : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end present_age_of_son_is_22_l313_313858


namespace expansion_of_binomials_l313_313148

theorem expansion_of_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 :=
  sorry

end expansion_of_binomials_l313_313148


namespace greatest_length_of_pieces_l313_313747

/-- Alicia has three ropes with lengths of 28 inches, 42 inches, and 70 inches.
She wants to cut these ropes into equal length pieces for her art project, and she doesn't want any leftover pieces.
Prove that the greatest length of each piece she can cut is 7 inches. -/
theorem greatest_length_of_pieces (a b c : ℕ) (h1 : a = 28) (h2 : b = 42) (h3 : c = 70) :
  ∃ (d : ℕ), d > 0 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ ∀ e : ℕ, e > 0 ∧ e ∣ a ∧ e ∣ b ∧ e ∣ c → e ≤ d := sorry

end greatest_length_of_pieces_l313_313747


namespace remainder_when_divided_by_15_l313_313715

theorem remainder_when_divided_by_15 (N : ℕ) (k : ℤ) (h1 : N = 60 * k + 49) : (N % 15) = 4 :=
sorry

end remainder_when_divided_by_15_l313_313715


namespace dot_product_result_l313_313332

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, m)
def c : ℝ × ℝ := (7, 1)

def are_parallel (a b : ℝ × ℝ) : Prop := 
  a.1 * b.2 = a.2 * b.1

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

theorem dot_product_result : 
  ∀ m : ℝ, are_parallel a (b m) → dot_product (b m) c = 10 := 
by
  sorry

end dot_product_result_l313_313332


namespace fill_tank_in_6_hours_l313_313724

theorem fill_tank_in_6_hours (A B : ℝ) (hA : A = 1 / 10) (hB : B = 1 / 15) : (1 / (A + B)) = 6 :=
by 
  sorry

end fill_tank_in_6_hours_l313_313724


namespace solve_for_a_l313_313042

theorem solve_for_a (x a : ℝ) (h : x = 3) (eq : 5 * x - a = 8) : a = 7 :=
by
  -- sorry to skip the proof as instructed
  sorry

end solve_for_a_l313_313042


namespace adult_ticket_cost_given_conditions_l313_313691

variables (C A S : ℕ)

def cost_relationships : Prop :=
  A = C + 10 ∧ S = A - 5 ∧ (5 * C + 2 * A + 2 * S + (S - 3) = 212)

theorem adult_ticket_cost_given_conditions :
  cost_relationships C A S → A = 28 :=
by
  intros h
  have h1 : A = C + 10 := h.left
  have h2 : S = A - 5 := h.right.left
  have h3 : (5 * C + 2 * A + 2 * S + (S - 3) = 212) := h.right.right
  sorry

end adult_ticket_cost_given_conditions_l313_313691


namespace prove_monotonic_increasing_range_l313_313218

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l313_313218


namespace arithmetic_sequence_sum_l313_313831

def sum_of_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ d : ℕ)
  (h₁ : a₁ + (a₁ + 6 * d) + (a₁ + 13 * d) + (a₁ + 17 * d) = 120) :
  sum_of_arithmetic_sequence a₁ d 19 = 570 :=
by
  sorry

end arithmetic_sequence_sum_l313_313831


namespace men_required_l313_313126

theorem men_required (W M : ℕ) (h1 : M * 20 * W = W) (h2 : (M - 4) * 25 * W = W) : M = 16 := by
  sorry

end men_required_l313_313126


namespace range_of_a_l313_313210

noncomputable def f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem range_of_a (a : ℝ) (h1 : a ∈ set.Ioo 0 1) 
  (h2 : ∀ x > 0, f a x ≥ f a (x - 1)) 
  : a ∈ set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l313_313210


namespace fraction_equals_decimal_l313_313929

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313929


namespace numberOfIncreasingMatrices_l313_313419

noncomputable def countIncreasingMatrices : ℕ :=
  1036800

theorem numberOfIncreasingMatrices (M : matrix (fin 4) (fin 4) ℕ) :
  (∀ (i : fin 4) (j : fin 4), 1 ≤ M i j ∧ M i j ≤ 16) ∧
  (∀ (i1 i2 j : fin 4), i1 < i2 → M i1 j < M i2 j) ∧ 
  (∀ (i j1 j2 : fin 4), j1 < j2 → M i j1 < M i j2) →
  ∃ (n : ℕ), n = 1036800 :=
by sorry

end numberOfIncreasingMatrices_l313_313419


namespace fraction_to_decimal_equiv_l313_313957

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313957


namespace arithmetic_example_l313_313731

theorem arithmetic_example : 3889 + 12.808 - 47.80600000000004 = 3854.002 := 
by
  sorry

end arithmetic_example_l313_313731


namespace algebraic_expr_value_at_neg_one_l313_313819

-- Define the expression "3 times the square of x minus 5"
def algebraic_expr (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem to state the value when x = -1 is 8
theorem algebraic_expr_value_at_neg_one : algebraic_expr (-1) = 8 := 
by
  -- The steps to prove are skipped with 'sorry'
  sorry

end algebraic_expr_value_at_neg_one_l313_313819


namespace middle_school_students_count_l313_313674

def split_equally (m h : ℕ) : Prop := m = h
def percent_middle (M m : ℕ) : Prop := m = M / 5
def percent_high (H h : ℕ) : Prop := h = 3 * H / 10
def total_students (M H : ℕ) : Prop := M + H = 50
def number_of_middle_school_students (M: ℕ) := M

theorem middle_school_students_count (M H m h : ℕ) 
  (hm_eq : split_equally m h) 
  (hm_percent : percent_middle M m) 
  (hh_percent : percent_high H h) 
  (htotal : total_students M H) : 
  number_of_middle_school_students M = 30 :=
by
  sorry

end middle_school_students_count_l313_313674


namespace squared_sum_inverse_l313_313171

theorem squared_sum_inverse (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 :=
by
  sorry

end squared_sum_inverse_l313_313171


namespace length_of_box_l313_313840

theorem length_of_box 
  (width height num_cubes length : ℕ)
  (h_width : width = 16)
  (h_height : height = 13)
  (h_cubes : num_cubes = 3120)
  (h_volume : length * width * height = num_cubes) :
  length = 15 :=
by
  sorry

end length_of_box_l313_313840


namespace exists_point_lt_2f_l313_313726

noncomputable def non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem exists_point_lt_2f {f : ℝ → ℝ} (h : ∀ x, 0 < f x) (hf : non_decreasing f) :
  ∃ a : ℝ, f (a + 1 / f a) < 2 * f a :=
begin
  sorry
end

end exists_point_lt_2f_l313_313726


namespace gcd_g50_g52_l313_313226

def g (x : ℕ) : ℕ := x^2 - 2 * x + 2021

theorem gcd_g50_g52 : Nat.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g50_g52_l313_313226


namespace crayon_boxes_needed_l313_313815

theorem crayon_boxes_needed (total_crayons : ℕ) (crayons_per_box : ℕ) (h1 : total_crayons = 80) (h2 : crayons_per_box = 8) : (total_crayons / crayons_per_box) = 10 :=
by
  sorry

end crayon_boxes_needed_l313_313815


namespace joe_spends_50_per_month_l313_313794

variable (X : ℕ) -- amount Joe spends per month

theorem joe_spends_50_per_month :
  let initial_amount := 240
  let resale_value := 30
  let months := 12
  let final_amount := 0 -- this means he runs out of money
  (initial_amount = months * X - months * resale_value) →
  X = 50 := 
by
  intros
  sorry

end joe_spends_50_per_month_l313_313794


namespace num_students_above_120_l313_313368

noncomputable def class_size : ℤ := 60
noncomputable def mean_score : ℝ := 110
noncomputable def std_score : ℝ := sorry  -- We do not know σ explicitly
noncomputable def probability_100_to_110 : ℝ := 0.35

def normal_distribution (x : ℝ) : Prop :=
  sorry -- placeholder for the actual normal distribution formula N(110, σ^2)

theorem num_students_above_120 :
  ∃ (students_above_120 : ℤ),
  (class_size = 60) ∧
  (∀ score, normal_distribution score → (100 ≤ score ∧ score ≤ 110) → probability_100_to_110 = 0.35) →
  students_above_120 = 9 :=
sorry

end num_students_above_120_l313_313368


namespace prove_monotonic_increasing_range_l313_313217

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l313_313217


namespace cube_side_length_l313_313297

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end cube_side_length_l313_313297


namespace functional_eq_zero_l313_313484

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_zero :
  (∀ x y : ℝ, f (x + y) = f x - f y) →
  (∀ x : ℝ, f x = 0) :=
by
  intros h x
  sorry

end functional_eq_zero_l313_313484


namespace pigeons_in_house_l313_313376

variable (x F c : ℝ)

theorem pigeons_in_house 
  (H1 : F = (x - 75) * 20 * c)
  (H2 : F = (x + 100) * 15 * c) :
  x = 600 := by
  sorry

end pigeons_in_house_l313_313376


namespace teacher_work_months_l313_313010

variable (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) (total_earnings : ℕ)

def monthly_earnings (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) : ℕ :=
  periods_per_day * pay_per_period * days_per_month

def number_of_months_worked (total_earnings : ℕ) (monthly_earnings : ℕ) : ℕ :=
  total_earnings / monthly_earnings

theorem teacher_work_months :
  let periods_per_day := 5
  let pay_per_period := 5
  let days_per_month := 24
  let total_earnings := 3600
  number_of_months_worked total_earnings (monthly_earnings periods_per_day pay_per_period days_per_month) = 6 :=
by
  sorry

end teacher_work_months_l313_313010


namespace wire_length_l313_313070

variable (L M l a : ℝ) -- Assume these variables are real numbers.

theorem wire_length (h1 : a ≠ 0) : L = (M / a) * l :=
sorry

end wire_length_l313_313070


namespace distinct_factors_1320_l313_313348

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l313_313348


namespace fraction_product_108_l313_313114

theorem fraction_product_108 : (1/2 : ℚ) * (1/3) * (1/6) * 108 = 3 := by
  sorry

end fraction_product_108_l313_313114


namespace value_of_a3_plus_a5_l313_313322

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

end value_of_a3_plus_a5_l313_313322


namespace only_natural_number_solution_l313_313967

theorem only_natural_number_solution (n : ℕ) :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = n * x * y * z) ↔ (n = 3) := 
sorry

end only_natural_number_solution_l313_313967


namespace calculation_correct_l313_313020

theorem calculation_correct : real.cbrt (-8) + 2 * (real.sqrt 2 + 2) - abs (1 - real.sqrt 2) = 3 + real.sqrt 2 :=
by
  sorry

end calculation_correct_l313_313020


namespace missy_yells_total_l313_313233

variable {O S M : ℕ}
variable (yells_at_obedient : ℕ)

-- Conditions:
def yells_stubborn (yells_at_obedient : ℕ) : ℕ := 4 * yells_at_obedient
def yells_mischievous (yells_at_obedient : ℕ) : ℕ := 2 * yells_at_obedient

-- Prove the total yells equal to 84 when yells_at_obedient = 12
theorem missy_yells_total (h : yells_at_obedient = 12) :
  yells_at_obedient + yells_stubborn yells_at_obedient + yells_mischievous yells_at_obedient = 84 :=
by
  sorry

end missy_yells_total_l313_313233


namespace alice_winning_strategy_l313_313688

theorem alice_winning_strategy (n : ℕ) (h : n ≥ 2) :
  (∃ strategy : Π (s : ℕ), s < n → (ℕ × ℕ), 
    ∀ (k : ℕ) (hk : k < n), ¬(strategy k hk).fst = (strategy k hk).snd) ↔ (n % 4 = 3) :=
sorry

end alice_winning_strategy_l313_313688


namespace time_reduced_fraction_l313_313288

theorem time_reduced_fraction 
  (S : ℝ) (hs : S = 24.000000000000007) 
  (D : ℝ) : 
  1 - (D / (S + 12) / (D / S)) = 1 / 3 :=
by sorry

end time_reduced_fraction_l313_313288


namespace angle4_is_35_l313_313769

theorem angle4_is_35
  (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (ha : angle1 = 50)
  (h_opposite : angle5 = 60)
  (triangle_sum : angle1 + angle5 + angle6 = 180)
  (supplementary_angle : angle2 + angle6 = 180) :
  angle4 = 35 :=
by
  sorry

end angle4_is_35_l313_313769


namespace remainder_problem_l313_313838

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end remainder_problem_l313_313838


namespace proof_solution_l313_313123

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  0 < x ∧ 7.61 * log x / log 2 + 2 * log x / log 4 = x ^ (log 16 / log 3 / log x / log 9)

theorem proof_solution : proof_problem (16 / 3) :=
by
  sorry

end proof_solution_l313_313123


namespace opposite_signs_add_same_signs_sub_l313_313320

-- Definitions based on the conditions
variables {a b : ℤ}

-- 1. Case when a and b have opposite signs
theorem opposite_signs_add (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := 
sorry

-- 2. Case when a and b have the same sign
theorem same_signs_sub (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b > 0) :
  a - b = 1 ∨ a - b = -1 := 
sorry

end opposite_signs_add_same_signs_sub_l313_313320


namespace inequality_relationship_l313_313169

noncomputable def a := 1 / 2023
noncomputable def b := Real.exp (-2022 / 2023)
noncomputable def c := (Real.cos (1 / 2023)) / 2023

theorem inequality_relationship : b > a ∧ a > c :=
by
  -- Initializing and defining the variables
  let a := a
  let b := b
  let c := c
  -- Providing the required proof
  sorry

end inequality_relationship_l313_313169


namespace first_class_product_probability_l313_313618

theorem first_class_product_probability
  (defective_rate : ℝ) (first_class_rate_qualified : ℝ)
  (H_def_rate : defective_rate = 0.04)
  (H_first_class_rate_qualified : first_class_rate_qualified = 0.75) :
  (1 - defective_rate) * first_class_rate_qualified = 0.72 :=
by
  sorry

end first_class_product_probability_l313_313618


namespace fraction_equals_decimal_l313_313931

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313931


namespace infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l313_313811

noncomputable def grid_size := 10
noncomputable def initial_infected_count_1 := 9
noncomputable def initial_infected_count_2 := 10

def condition (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n → 
  infected + steps * (infected / 2) < grid_size * grid_size

def can_infect_entire_grid (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n ∧ (
  ∃ t : ℕ, infected + t * (infected / 2) = grid_size * grid_size)

theorem infection_does_not_spread_with_9_cells :
  ¬ can_infect_entire_grid initial_infected_count_1 :=
by
  sorry

theorem minimum_infected_cells_needed :
  condition initial_infected_count_2 :=
by
  sorry

end infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l313_313811


namespace train_length_l313_313866

theorem train_length (L V : ℝ) (h1 : L = V * 26) (h2 : L + 150 = V * 39) : L = 300 := by
  sorry

end train_length_l313_313866


namespace monotonic_increasing_range_l313_313205

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l313_313205


namespace perimeter_of_regular_nonagon_l313_313441

def regular_nonagon_side_length := 3
def number_of_sides := 9

theorem perimeter_of_regular_nonagon (h1 : number_of_sides = 9) (h2 : regular_nonagon_side_length = 3) :
  9 * 3 = 27 :=
by
  sorry

end perimeter_of_regular_nonagon_l313_313441


namespace find_last_number_of_consecutive_even_numbers_l313_313634

theorem find_last_number_of_consecutive_even_numbers (x : ℕ) (h : 8 * x + 2 + 4 + 6 + 8 + 10 + 12 + 14 = 424) : x + 14 = 60 :=
sorry

end find_last_number_of_consecutive_even_numbers_l313_313634


namespace num_factors_1320_l313_313353

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l313_313353


namespace power_difference_l313_313501

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l313_313501


namespace basketball_players_taking_chemistry_l313_313871

variable (total_players : ℕ) (taking_biology : ℕ) (taking_both : ℕ)

theorem basketball_players_taking_chemistry (h1 : total_players = 20) 
                                           (h2 : taking_biology = 8) 
                                           (h3 : taking_both = 4) 
                                           (h4 : ∀p, p ≤ total_players) :
  total_players - taking_biology + taking_both = 16 :=
by sorry

end basketball_players_taking_chemistry_l313_313871


namespace team_selection_ways_correct_l313_313865

-- Definition of the problem
def team_selection_ways : ℕ := 
  (Fintype.choose 6 2) -- selecting 2 positions (leader, deputy leader) out of 6
  * (Fintype.choose 4 2) -- selecting 2 ordinary members out of remaining 4

-- Proof statement
theorem team_selection_ways_correct : team_selection_ways = 180 := by
  sorry

end team_selection_ways_correct_l313_313865


namespace anya_lost_games_correct_l313_313311

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l313_313311


namespace vessel_reaches_boat_in_shortest_time_l313_313744

-- Define the given conditions as hypotheses
variable (dist_AC : ℝ) (angle_C : ℝ) (speed_CB : ℝ) (angle_B : ℝ) (speed_A : ℝ)

-- Assign values to variables based on the problem statement
def vessel_distress_boat_condition : Prop :=
  dist_AC = 10 ∧ angle_C = 45 ∧ speed_CB = 9 ∧ angle_B = 105 ∧ speed_A = 21

-- Define the time (in minutes) for the vessel to reach the fishing boat
noncomputable def shortest_time_to_reach_boat : ℝ :=
  25

-- The theorem that we need to prove given the conditions
theorem vessel_reaches_boat_in_shortest_time :
  vessel_distress_boat_condition dist_AC angle_C speed_CB angle_B speed_A → 
  shortest_time_to_reach_boat = 25 := by
    intros
    sorry

end vessel_reaches_boat_in_shortest_time_l313_313744


namespace fraction_subtraction_l313_313783

theorem fraction_subtraction (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := 
by 
  sorry

end fraction_subtraction_l313_313783


namespace num_distinct_factors_1320_l313_313351

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l313_313351


namespace power_difference_l313_313499

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l313_313499


namespace problem_l313_313845

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4)) ^ 6 + (Real.cos (x / 4)) ^ 6

theorem problem : (derivative^[2008] f 0) = 3 / 8 := by sorry

end problem_l313_313845


namespace dual_cassette_recorder_price_l313_313140

theorem dual_cassette_recorder_price :
  ∃ (x y : ℝ),
    (x - 0.05 * x = 380) ∧
    (y = x + 0.08 * x) ∧ 
    (y = 432) :=
by
  -- sorry to skip the proof.
  sorry

end dual_cassette_recorder_price_l313_313140


namespace a_eq_bn_l313_313694

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end a_eq_bn_l313_313694


namespace problem_equivalent_statement_l313_313176

-- Conditions as Lean definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def periodic_property (f : ℝ → ℝ) := ∀ x, x ≥ 0 → f (x + 2) = -f x
def specific_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 8

-- The main theorem
theorem problem_equivalent_statement (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_periodic : periodic_property f) 
  (hf_specific : specific_interval f) :
  f (-2013) + f 2014 = 1 / 3 := 
sorry

end problem_equivalent_statement_l313_313176


namespace prime_between_30_and_40_with_remainder_7_l313_313824

theorem prime_between_30_and_40_with_remainder_7 (n : ℕ) 
  (h1 : Nat.Prime n) 
  (h2 : 30 < n) 
  (h3 : n < 40) 
  (h4 : n % 12 = 7) : 
  n = 31 := 
sorry

end prime_between_30_and_40_with_remainder_7_l313_313824


namespace smartphone_demand_inverse_proportional_l313_313787

theorem smartphone_demand_inverse_proportional (k : ℝ) (d d' p p' : ℝ) 
  (h1 : d = 30)
  (h2 : p = 600)
  (h3 : p' = 900)
  (h4 : d * p = k) :
  d' * p' = k → d' = 20 := 
by 
  sorry

end smartphone_demand_inverse_proportional_l313_313787


namespace bob_total_candies_l313_313872

noncomputable def total_chewing_gums : ℕ := 45
noncomputable def total_chocolate_bars : ℕ := 60
noncomputable def total_assorted_candies : ℕ := 45

def chewing_gum_ratio_sam_bob : ℕ × ℕ := (2, 3)
def chocolate_bar_ratio_sam_bob : ℕ × ℕ := (3, 1)
def assorted_candy_ratio_sam_bob : ℕ × ℕ := (1, 1)

theorem bob_total_candies :
  let bob_chewing_gums := (total_chewing_gums * chewing_gum_ratio_sam_bob.snd) / (chewing_gum_ratio_sam_bob.fst + chewing_gum_ratio_sam_bob.snd)
  let bob_chocolate_bars := (total_chocolate_bars * chocolate_bar_ratio_sam_bob.snd) / (chocolate_bar_ratio_sam_bob.fst + chocolate_bar_ratio_sam_bob.snd)
  let bob_assorted_candies := (total_assorted_candies * assorted_candy_ratio_sam_bob.snd) / (assorted_candy_ratio_sam_bob.fst + assorted_candy_ratio_sam_bob.snd)
  bob_chewing_gums + bob_chocolate_bars + bob_assorted_candies = 64 := by
  sorry

end bob_total_candies_l313_313872


namespace part_a_part_b_l313_313033

-- Define the functions K_m and K_4
def K (m : ℕ) (x y z : ℝ) : ℝ :=
  x * (x - y)^m * (x - z)^m + y * (y - x)^m * (y - z)^m + z * (z - x)^m * (z - y)^m

-- Define M
def M (x y z : ℝ) : ℝ :=
  (x - y)^2 * (y - z)^2 * (z - x)^2

-- The proof goals:
-- 1. Prove K_m >= 0 for odd positive integer m
theorem part_a (m : ℕ) (hm : m % 2 = 1) (x y z : ℝ) : 
  0 ≤ K m x y z := 
sorry

-- 2. Prove K_7 + M^2 * K_1 >= M * K_4
theorem part_b (x y z : ℝ) : 
  K 7 x y z + (M x y z)^2 * K 1 x y z ≥ M x y z * K 4 x y z := 
sorry

end part_a_part_b_l313_313033


namespace find_b_l313_313587

-- Define the slopes of the two lines derived from the given conditions
noncomputable def slope1 := -2 / 3
noncomputable def slope2 (b : ℚ) := -b / 3

-- Lean 4 statement to prove that for the lines to be perpendicular, b must be -9/2
theorem find_b (b : ℚ) (h_perpendicular: slope1 * slope2 b = -1) : b = -9 / 2 := by
  sorry

end find_b_l313_313587


namespace max_card_count_sum_l313_313465

theorem max_card_count_sum (W B R : ℕ) (total_cards : ℕ) 
  (white_cards black_cards red_cards : ℕ) : 
  total_cards = 300 ∧ white_cards = 100 ∧ black_cards = 100 ∧ red_cards = 100 ∧
  (∀ w, w < white_cards → ∃ b, b < black_cards) ∧ 
  (∀ b, b < black_cards → ∃ r, r < red_cards) ∧ 
  (∀ r, r < red_cards → ∃ w, w < white_cards) →
  ∃ max_sum, max_sum = 20000 :=
by
  sorry

end max_card_count_sum_l313_313465


namespace find_t_value_l313_313086

theorem find_t_value (t : ℝ) (h1 : (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5)) : t = 10 :=
sorry

end find_t_value_l313_313086


namespace condition_relation_l313_313615

variable (A B C : Prop)

theorem condition_relation (h1 : C → B) (h2 : A → B) : 
  (¬(A → C) ∧ ¬(C → A)) :=
by 
  sorry

end condition_relation_l313_313615


namespace total_calories_burned_l313_313812

def base_distance : ℝ := 15
def records : List ℝ := [0.1, -0.8, 0.9, 16.5 - base_distance, 2.0, -1.5, 14.1 - base_distance, 1.0, 0.8, -1.1]
def calorie_burn_rate : ℝ := 20

theorem total_calories_burned :
  (base_distance * 10 + (List.sum records)) * calorie_burn_rate = 3040 :=
by
  sorry

end total_calories_burned_l313_313812


namespace distinct_four_digit_numbers_l313_313335

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l313_313335


namespace curve_points_satisfy_equation_l313_313364

theorem curve_points_satisfy_equation (C : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C → f p = 0) → (∀ q : ℝ × ℝ, f q ≠ 0 → q ∉ C) :=
by
  intro h₁
  intro q
  intro h₂
  sorry

end curve_points_satisfy_equation_l313_313364


namespace fraction_to_decimal_l313_313891

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313891


namespace add_points_proof_l313_313569

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l313_313569


namespace February_March_Ratio_l313_313146

theorem February_March_Ratio (J F M : ℕ) (h1 : F = 2 * J) (h2 : M = 8800) (h3 : J + F + M = 12100) : F / M = 1 / 4 :=
by
  sorry

end February_March_Ratio_l313_313146


namespace mean_of_second_set_l313_313609

theorem mean_of_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 90) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 :=
by
  sorry

end mean_of_second_set_l313_313609


namespace min_draws_to_ensure_20_of_one_color_l313_313619

-- Define the total number of balls for each color
def red_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 10

-- Define the minimum number of balls to guarantee at least one color reaches 20 balls
def min_balls_needed : ℕ := 95

-- Theorem to state the problem mathematically in Lean
theorem min_draws_to_ensure_20_of_one_color :
  ∀ (r g y b w bl : ℕ),
    r = 30 → g = 25 → y = 22 → b = 15 → w = 12 → bl = 10 →
    (∃ n : ℕ, n ≥ min_balls_needed ∧
    ∀ (r_draw g_draw y_draw b_draw w_draw bl_draw : ℕ),
      r_draw + g_draw + y_draw + b_draw + w_draw + bl_draw = n →
      (r_draw > 19 ∨ g_draw > 19 ∨ y_draw > 19 ∨ b_draw > 19 ∨ w_draw > 19 ∨ bl_draw > 19)) :=
by
  intros r g y b w bl hr hg hy hb hw hbl
  use min_balls_needed
  sorry

end min_draws_to_ensure_20_of_one_color_l313_313619


namespace monotonically_increasing_range_l313_313225

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l313_313225


namespace find_a_c_area_A_90_area_B_90_l313_313671

variable (a b c : ℝ)
variable (C : ℝ)

def triangle_condition1 := a + 1/a = 4 * Real.cos C
def triangle_condition2 := b = 1
def sin_C := Real.sin C = Real.sqrt 21 / 7

-- Proof problem for (1)
theorem find_a_c (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h3 : sin_C C) :
  (a = Real.sqrt 7 ∧ c = 2) ∨ (a = Real.sqrt 7 / 7 ∧ c = 2 * Real.sqrt 7 / 7) :=
sorry

-- Conditions for (2) when A=90°
def right_triangle_A := C = Real.pi / 2

-- Proof problem for (2) when A=90°
theorem area_A_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h4 : right_triangle_A C) :
  ((a = Real.sqrt 3) → area = Real.sqrt 2 / 2) :=
sorry

-- Conditions for (2) when B=90°
def right_triangle_B := b = 1 ∧ C = Real.pi / 2

-- Proof problem for (2) when B=90°
theorem area_B_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h5 : right_triangle_B b C) :
  ((a = Real.sqrt 3 / 3) → area = Real.sqrt 2 / 6) :=
sorry

end find_a_c_area_A_90_area_B_90_l313_313671


namespace tree_height_relationship_l313_313380

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end tree_height_relationship_l313_313380


namespace decreasing_intervals_l313_313246

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

theorem decreasing_intervals : 
  (∀ x y : ℝ, x < y → ((y < -1 ∨ x > -1) → f y < f x)) ∧
  (∀ x y : ℝ, x < y → (y ≥ -1 ∧ x ≤ -1 → f y < f x)) :=
by 
  intros;
  sorry

end decreasing_intervals_l313_313246


namespace interest_paid_percent_l313_313849

noncomputable def down_payment : ℝ := 300
noncomputable def total_cost : ℝ := 750
noncomputable def monthly_payment : ℝ := 57
noncomputable def final_payment : ℝ := 21
noncomputable def num_monthly_payments : ℕ := 9

noncomputable def total_instalments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_paid : ℝ := total_instalments + down_payment
noncomputable def amount_borrowed : ℝ := total_cost - down_payment
noncomputable def interest_paid : ℝ := total_paid - amount_borrowed
noncomputable def interest_percent : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_paid_percent:
  interest_percent = 85.33 := by
  sorry

end interest_paid_percent_l313_313849


namespace seedling_costs_and_purchase_l313_313283

variable (cost_A cost_B : ℕ)
variable (m n : ℕ)

-- Conditions
def conditions : Prop :=
  (cost_A = cost_B + 5) ∧ 
  (400 / cost_A = 300 / cost_B)

-- Prove costs and purchase for minimal costs
theorem seedling_costs_and_purchase (cost_A cost_B : ℕ) (m n : ℕ)
  (h1 : conditions cost_A cost_B)
  (h2 : m + n = 150)
  (h3 : m ≥ n / 2)
  : cost_A = 20 ∧ cost_B = 15 ∧ 5 * 50 + 2250 = 2500 
  := by
  sorry

end seedling_costs_and_purchase_l313_313283


namespace fraction_to_decimal_l313_313889

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313889


namespace solve_inequality_l313_313412

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end solve_inequality_l313_313412


namespace minimum_time_to_cook_3_pancakes_l313_313112

theorem minimum_time_to_cook_3_pancakes (can_fry_two_pancakes_at_a_time : Prop) 
   (time_to_fully_cook_one_pancake : ℕ) (time_to_cook_one_side : ℕ) :
  can_fry_two_pancakes_at_a_time →
  time_to_fully_cook_one_pancake = 2 →
  time_to_cook_one_side = 1 →
  3 = 3 := 
by
  intros
  sorry

end minimum_time_to_cook_3_pancakes_l313_313112


namespace soda_cost_l313_313269

theorem soda_cost (total_cost sandwich_price : ℝ) (num_sandwiches num_sodas : ℕ) (total : total_cost = 8.38)
  (sandwich_cost : sandwich_price = 2.45) (total_sandwiches : num_sandwiches = 2) (total_sodas : num_sodas = 4) :
  ((total_cost - (num_sandwiches * sandwich_price)) / num_sodas) = 0.87 :=
by
  sorry

end soda_cost_l313_313269


namespace repeating_decimal_to_fraction_l313_313113

theorem repeating_decimal_to_fraction : (∃ (x : ℚ), x = 0.4 + 4 / 9) :=
sorry

end repeating_decimal_to_fraction_l313_313113


namespace complex_multiplication_l313_313161

-- Define i such that i^2 = -1
def i : ℂ := Complex.I

theorem complex_multiplication : (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i := by
  sorry

end complex_multiplication_l313_313161


namespace find_pairs_l313_313652

theorem find_pairs (p n : ℕ) (hp : Nat.Prime p) (h1 : n ≤ 2 * p) (h2 : n^(p-1) ∣ (p-1)^n + 1) : 
    (p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ (n = 1) :=
by
  sorry

end find_pairs_l313_313652


namespace neznaika_made_mistake_l313_313236

-- Define the total digits used from 1 to N pages
def totalDigits (N : ℕ) : ℕ :=
  let single_digit_pages := min N 9
  let double_digit_pages := if N > 9 then N - 9 else 0
  single_digit_pages * 1 + double_digit_pages * 2

-- The main statement we want to prove
theorem neznaika_made_mistake : ¬ ∃ N : ℕ, totalDigits N = 100 :=
by
  sorry

end neznaika_made_mistake_l313_313236


namespace range_of_a_l313_313212

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x : ℝ, 0 < x → (a ^ x + (1 + a) ^ x)' ≥ 0) :
    a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) :=
by
  sorry

end range_of_a_l313_313212


namespace part1_x_values_part2_m_value_l313_313279

/-- 
Part 1: Given \(2x^2 + 3x - 5\) and \(-2x + 2\) are opposite numbers, 
prove that \(x = -\frac{3}{2}\) or \(x = 1\).
-/
theorem part1_x_values (x : ℝ)
  (hyp : 2 * x ^ 2 + 3 * x - 5 = -(-2 * x + 2)) :
  2 * x ^ 2 + 5 * x - 7 = 0 → (x = -3 / 2 ∨ x = 1) :=
by
  sorry

/-- 
Part 2: If \(\sqrt{m^2 - 6}\) and \(\sqrt{6m + 1}\) are of the same type, 
prove that \(m = 7\).
-/
theorem part2_m_value (m : ℝ)
  (hyp : m ^ 2 - 6 = 6 * m + 1) :
  7 ^ 2 - 6 = 6 * 7 + 1 → m = 7 :=
by
  sorry

end part1_x_values_part2_m_value_l313_313279


namespace total_selling_price_correct_l313_313738

def original_price : ℝ := 100
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.08

theorem total_selling_price_correct :
  let discount := original_price * discount_percent
  let sale_price := original_price - discount
  let tax := sale_price * tax_percent
  let total_selling_price := sale_price + tax
  total_selling_price = 75.6 := by
sorry

end total_selling_price_correct_l313_313738


namespace fraction_to_decimal_equiv_l313_313963

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313963


namespace quadratic_has_two_roots_l313_313201

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end quadratic_has_two_roots_l313_313201


namespace odd_function_properties_l313_313670

theorem odd_function_properties
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 1 ≤ x ∧ x ≤ 3 ∧ 1 ≤ y ∧ y ≤ 3 ∧ x < y → f x < f y)
  (h_min_val : ∀ x, 1 ≤ x ∧ x ≤ 3 → 7 ≤ f x) :
  (∀ x y, -3 ≤ x ∧ x ≤ -1 ∧ -3 ≤ y ∧ y ≤ -1 ∧ x < y → f x < f y) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) :=
sorry

end odd_function_properties_l313_313670


namespace truncated_cone_volume_correct_larger_cone_volume_correct_l313_313466

def larger_base_radius : ℝ := 10 -- R
def smaller_base_radius : ℝ := 5  -- r
def height_truncated_cone : ℝ := 8 -- h
def height_small_cone : ℝ := 8 -- x

noncomputable def volume_truncated_cone : ℝ :=
  (1/3) * Real.pi * height_truncated_cone * 
  (larger_base_radius^2 + larger_base_radius * smaller_base_radius + smaller_base_radius^2)

theorem truncated_cone_volume_correct :
  volume_truncated_cone = 466 + 2/3 * Real.pi := sorry

noncomputable def total_height_larger_cone : ℝ :=
  height_small_cone + height_truncated_cone

noncomputable def volume_larger_cone : ℝ :=
  (1/3) * Real.pi * (larger_base_radius^2) * total_height_larger_cone

theorem larger_cone_volume_correct :
  volume_larger_cone = 533 + 1/3 * Real.pi := sorry

end truncated_cone_volume_correct_larger_cone_volume_correct_l313_313466


namespace convert_fraction_to_decimal_l313_313954

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313954


namespace find_ABC_l313_313249

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := 
  x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  (∀ x : ℝ, x > 5 → g x 2 (-2) (-24) > 0.5) ∧
  (A = 2) ∧
  (B = -2) ∧
  (C = -24) ∧
  (∀ x, A * x^2 + B * x + C = A * (x + 3) * (x - 4)) → 
  A + B + C = -24 := 
by
  sorry

end find_ABC_l313_313249


namespace multiples_of_4_between_88_and_104_l313_313257

theorem multiples_of_4_between_88_and_104 : 
  ∃ n, (104 - 4 * 23 = n) ∧ n = 88 ∧ ( ∀ x, (x ≥ 88 ∧ x ≤ 104 ∧ x % 4 = 0) → ( x - 88) / 4 < 24) :=
by
  sorry

end multiples_of_4_between_88_and_104_l313_313257


namespace abs_inequality_solution_l313_313415

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end abs_inequality_solution_l313_313415


namespace server_processes_21600000_requests_l313_313464

theorem server_processes_21600000_requests :
  (15000 * 1440 = 21600000) :=
by
  -- Calculations and step-by-step proof
  sorry

end server_processes_21600000_requests_l313_313464


namespace probability_A2_l313_313743

/-- Definitions based on the conditions -/
def P_A1 : ℝ := 0.5
def P_B1 : ℝ := 0.5
def P_A2_given_A1 : ℝ := 0.4
def P_A2_given_B1 : ℝ := 0.6

/-- Theorem statement based on the problem -/
theorem probability_A2 : 
  let P_A2 := P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1 in
  P_A2 = 0.5 :=
by
  sorry

end probability_A2_l313_313743


namespace present_age_of_son_is_22_l313_313859

theorem present_age_of_son_is_22 (S F : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end present_age_of_son_is_22_l313_313859


namespace find_k_l313_313992

def vec_a := (3 : ℕ, 1 : ℕ)
def vec_b := (1 : ℕ, 0 : ℕ)

def vec_c (k : ℚ) : ℚ × ℚ := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2)

theorem find_k (k : ℚ) (h : vec_a.1 * vec_c k.1 + vec_a.2 * vec_c k.2 = 0) : 
  k = -10 / 3 :=
by
  sorry

end find_k_l313_313992


namespace fraction_to_decimal_l313_313912

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313912


namespace distinct_four_digit_numbers_l313_313340

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l313_313340


namespace problem_1_problem_2_l313_313181

open Real

-- Step 1: Define the line and parabola conditions
def line_through_focus (k n : ℝ) : Prop := ∀ (x y : ℝ),
  y = k * (x - 1) ∧ (y = 0 → x = 1)
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Step 2: Prove x_1 x_2 = 1 if line passes through the focus
theorem problem_1 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k 1)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1))
  (h_non_zero : x1 * x2 ≠ 0) :
  x1 * x2 = 1 :=
sorry

-- Step 3: Prove n = 4 if x_1 x_2 + y_1 y_2 = 0
theorem problem_2 (k n : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_line_thru_focus : line_through_focus k n)
  (h_parabola_points : parabola x1 y1 ∧ parabola x2 y2)
  (h_intersection : y1 = k * (x1 - n) ∧ y2 = k * (x2 - n))
  (h_product_relate : x1 * x2 + y1 * y2 = 0) :
  n = 4 :=
sorry

end problem_1_problem_2_l313_313181


namespace spinner_prob_l313_313137

theorem spinner_prob (PD PE PF_PG : ℚ) (hD : PD = 1/4) (hE : PE = 1/3) 
  (hTotal : PD + PE + PF_PG = 1) : PF_PG = 5/12 := by
  sorry

end spinner_prob_l313_313137


namespace fraction_to_decimal_l313_313892

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313892


namespace ratio_equation_solution_l313_313754

theorem ratio_equation_solution (x : ℝ) :
  (4 + 2 * x) / (6 + 3 * x) = (2 + x) / (3 + 2 * x) → (x = 0 ∨ x = 4) :=
by
  -- the proof steps would go here
  sorry

end ratio_equation_solution_l313_313754


namespace add_points_proof_l313_313568

theorem add_points_proof :
  ∃ x, (9 * x - 8 = 82) ∧ x = 10 :=
by
  existsi (10 : ℤ)
  split
  . exact eq.refl 82
  . exact eq.refl 10
  sorry

end add_points_proof_l313_313568


namespace find_x_l313_313976

theorem find_x (x : ℝ) (h : 2 * x - 1 = -( -x + 5 )) : x = -6 :=
by
  sorry

end find_x_l313_313976


namespace distance_points_3D_l313_313761

open Real

def distance_between_points (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_points_3D {p1 p2 : ℝ × ℝ × ℝ} (h1 : p1 = (3, -2, 5)) (h2 : p2 = (7, 4, 2)) :
  distance_between_points p1 p2 = sqrt 61 :=
by
  rw [h1, h2]
  simp [distance_between_points]
  norm_num
  sorry

end distance_points_3D_l313_313761


namespace abc_inequalities_l313_313687

noncomputable def a : ℝ := Real.log 1 / Real.log 2 - Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 2) ^ 3
noncomputable def c : ℝ := Real.sqrt 3

theorem abc_inequalities :
  a < b ∧ b < c :=
by
  -- Proof omitted
  sorry

end abc_inequalities_l313_313687


namespace bell_rings_count_l313_313807

-- Defining the conditions
def bell_rings_per_class : ℕ := 2
def total_classes_before_music : ℕ := 4
def bell_rings_during_music_start : ℕ := 1

-- The main proof statement
def total_bell_rings : ℕ :=
  total_classes_before_music * bell_rings_per_class + bell_rings_during_music_start

theorem bell_rings_count : total_bell_rings = 9 := by
  sorry

end bell_rings_count_l313_313807


namespace fraction_to_decimal_equiv_l313_313958

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313958


namespace neon_signs_blink_together_l313_313599

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : Nat.lcm a b = 45 := by
  rw [ha, hb]
  have : Nat.lcm 9 15 = 45 := by sorry
  exact this

end neon_signs_blink_together_l313_313599


namespace model_price_and_schemes_l313_313832

theorem model_price_and_schemes :
  ∃ (x y : ℕ), 3 * x = 2 * y ∧ x + 2 * y = 80 ∧ x = 20 ∧ y = 30 ∧ 
  ∃ (count m : ℕ), 468 ≤ m ∧ m ≤ 480 ∧ 
                   (20 * m + 30 * (800 - m) ≤ 19320) ∧ 
                   (800 - m ≥ 2 * m / 3) ∧ 
                   count = 13 ∧ 
                   800 - 480 = 320 :=
sorry

end model_price_and_schemes_l313_313832


namespace number_of_girls_l313_313591

theorem number_of_girls
  (total_students : ℕ)
  (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_non_binary : ℕ)
  (h_ratio : ratio_girls = 3 ∧ ratio_boys = 2 ∧ ratio_non_binary = 1)
  (h_total : total_students = 72) :
  ∃ (k : ℕ), 3 * k = (total_students * 3) / 6 ∧ 3 * k = 36 :=
by
  sorry

end number_of_girls_l313_313591


namespace markus_grandson_age_l313_313545

theorem markus_grandson_age :
  ∃ (x : ℕ), let son := 2 * x in let markus := 2 * son in x + son + markus = 140 ∧ x = 20 :=
by
  sorry

end markus_grandson_age_l313_313545


namespace integer_solution_count_l313_313165

theorem integer_solution_count :
  ∃ n : ℕ, n = 10 ∧
  ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 15 ∧ (0 ≤ x1 ∧ x1 ≤ 5) ∧ (0 ≤ x2 ∧ x2 ≤ 6) ∧ (0 ≤ x3 ∧ x3 ≤ 7) := 
sorry

end integer_solution_count_l313_313165


namespace det_nonzero_if_k_minors_eq_zero_l313_313732

open Matrix

variables {n k : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)

noncomputable def minors_n_minus_1 (A : Matrix (Fin n) (Fin n) ℂ) : ℕ :=
  -- Assume a function that calculates the number of (n-1)-order minors of A equal to 0
  sorry

-- The main statement:
theorem det_nonzero_if_k_minors_eq_zero (hn : 2 ≤ n) (hk : 1 ≤ k) (hk2 : k ≤ n - 1) (hA : minors_n_minus_1 A = k) :
  det A ≠ 0 :=
sorry

end det_nonzero_if_k_minors_eq_zero_l313_313732


namespace pure_imaginary_solution_l313_313821

theorem pure_imaginary_solution (m : ℝ) (h₁ : m^2 - m - 4 = 0) (h₂ : m^2 - 5 * m - 6 ≠ 0) :
  m = (1 + Real.sqrt 17) / 2 ∨ m = (1 - Real.sqrt 17) / 2 :=
sorry

end pure_imaginary_solution_l313_313821


namespace sum_of_integers_ending_in_7_between_100_and_450_l313_313019

theorem sum_of_integers_ending_in_7_between_100_and_450 :
  let a := 107 in
  let d := 10 in
  let n := 35 in
  let a_n := 447 in
  let S_n := (n / 2) * (a + a_n) in
  S_n = 9695 :=
by
  let a := 107
  let d := 10
  let n := 35
  let a_n := 447
  let S_n := (n / 2) * (a + a_n)
  show S_n = 9695
  sorry

end sum_of_integers_ending_in_7_between_100_and_450_l313_313019


namespace words_per_page_l313_313463

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 90) : p = 90 :=
sorry

end words_per_page_l313_313463


namespace line_equation_passing_through_point_and_opposite_intercepts_l313_313100

theorem line_equation_passing_through_point_and_opposite_intercepts 
  : ∃ (a b : ℝ), (y = a * x) ∨ (x - y = b) :=
by
  use (3/2), (-1)
  sorry

end line_equation_passing_through_point_and_opposite_intercepts_l313_313100


namespace spurs_team_players_l313_313578

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end spurs_team_players_l313_313578


namespace largest_possible_s_l313_313535

theorem largest_possible_s 
  (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (hangles : (r - 2) * 60 * s = (s - 2) * 61 * r) : 
  s = 121 := 
sorry

end largest_possible_s_l313_313535


namespace relatively_prime_m_n_l313_313533

noncomputable def probability_of_distinct_real_solutions : ℝ :=
  let b := (1 : ℝ)
  if 1 ≤ b ∧ b ≤ 25 then 1 else 0

theorem relatively_prime_m_n : ∃ m n : ℕ, 
  Nat.gcd m n = 1 ∧ 
  (1 : ℝ) = (m : ℝ) / (n : ℝ) ∧ m + n = 2 := 
by
  sorry

end relatively_prime_m_n_l313_313533


namespace value_of_x_l313_313427

variable (x y z : ℝ)

-- Conditions based on the problem statement
def condition1 := x = (1 / 3) * y
def condition2 := y = (1 / 4) * z
def condition3 := z = 96

-- The theorem to be proven
theorem value_of_x (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) : x = 8 := 
by
  sorry

end value_of_x_l313_313427


namespace number_of_unique_triangle_areas_l313_313882

theorem number_of_unique_triangle_areas :
  ∀ (G H I J K L : ℝ) (d₁ d₂ d₃ d₄ : ℝ),
    G ≠ H → H ≠ I → I ≠ J → G ≠ I → G ≠ J →
    H ≠ J →
    G - H = 1 → H - I = 1 → I - J = 2 →
    K - L = 2 →
    d₄ = abs d₃ →
    (d₁ = abs (K - G)) ∨ (d₂ = abs (L - G)) ∨ (d₁ = d₂) →
    ∃ (areas : ℕ), 
    areas = 3 :=
by sorry

end number_of_unique_triangle_areas_l313_313882


namespace monotonic_increasing_range_l313_313215

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l313_313215


namespace band_formation_l313_313125

theorem band_formation (r x m : ℕ) (h1 : r * x + 3 = m) (h2 : (r - 1) * (x + 2) = m) (h3 : m < 100) : m = 69 :=
by
  sorry

end band_formation_l313_313125


namespace ratio_of_friends_l313_313410

theorem ratio_of_friends (friends_in_classes friends_in_clubs : ℕ) (thread_per_keychain total_thread : ℕ) 
  (h1 : thread_per_keychain = 12) (h2 : friends_in_classes = 6) (h3 : total_thread = 108)
  (keychains_total : total_thread / thread_per_keychain = 9) 
  (keychains_clubs : (total_thread / thread_per_keychain) - friends_in_classes = friends_in_clubs) :
  friends_in_clubs / friends_in_classes = 1 / 2 :=
by
  sorry

end ratio_of_friends_l313_313410


namespace max_value_of_a_squared_b_squared_c_squared_l313_313386

theorem max_value_of_a_squared_b_squared_c_squared
  (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_constraint : a + 2 * b + 3 * c = 1) : a^2 + b^2 + c^2 ≤ 1 :=
sorry

end max_value_of_a_squared_b_squared_c_squared_l313_313386


namespace fraction_to_decimal_l313_313896

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313896


namespace distinct_factors_1320_l313_313347

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l313_313347


namespace problem_l313_313189

theorem problem (m : ℝ) (h : m + 1/m = 6) : m^2 + 1/m^2 + 3 = 37 :=
by
  sorry

end problem_l313_313189


namespace evaluate_expression_l313_313482

theorem evaluate_expression : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 :=
by
  sorry

end evaluate_expression_l313_313482


namespace adam_change_is_correct_l313_313633

-- Define the conditions
def adam_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28
def change : ℝ := adam_money - airplane_cost

-- State the theorem
theorem adam_change_is_correct : change = 0.72 := 
by {
  -- Proof can be added later
  sorry
}

end adam_change_is_correct_l313_313633


namespace fraction_to_decimal_l313_313905

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313905


namespace find_k_l313_313991

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l313_313991


namespace evaluate_expression_l313_313306

noncomputable def w := Complex.exp (2 * Real.pi * Complex.I / 11)

theorem evaluate_expression : (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) = 88573 := 
by 
  sorry

end evaluate_expression_l313_313306


namespace determine_p_in_terms_of_q_l313_313480

variable {p q : ℝ}

-- Given the condition in the problem
def log_condition (p q : ℝ) : Prop :=
  Real.log p + 2 * Real.log q = Real.log (2 * p + q)

-- The goal is to prove that under this condition, the following holds
theorem determine_p_in_terms_of_q (h : log_condition p q) :
  p = q / (q^2 - 2) :=
sorry

end determine_p_in_terms_of_q_l313_313480


namespace white_pairs_coincide_l313_313755

theorem white_pairs_coincide 
  (red_half : ℕ) (blue_half : ℕ) (white_half : ℕ)
  (red_pairs : ℕ) (blue_pairs : ℕ) (red_white_pairs : ℕ) :
  red_half = 2 → blue_half = 4 → white_half = 6 →
  red_pairs = 1 → blue_pairs = 2 → red_white_pairs = 2 →
  2 * (red_half - red_pairs + blue_half - 2 * blue_pairs + 
       white_half - 2 * red_white_pairs) = 4 :=
by
  intros 
    h_red_half h_blue_half h_white_half 
    h_red_pairs h_blue_pairs h_red_white_pairs
  rw [h_red_half, h_blue_half, h_white_half, 
      h_red_pairs, h_blue_pairs, h_red_white_pairs]
  sorry

end white_pairs_coincide_l313_313755


namespace linear_function_implies_m_value_l313_313513

variable (x m : ℝ)

theorem linear_function_implies_m_value :
  (∃ y : ℝ, y = (m-3)*x^(m^2-8) + m + 1 ∧ ∀ x1 x2 : ℝ, y = y * (x2 - x1) + y * x1) → m = -3 :=
by
  sorry

end linear_function_implies_m_value_l313_313513


namespace solve_ab_eq_l313_313574

theorem solve_ab_eq (a b : ℕ) (h : a^b + a + b = b^a) : a = 5 ∧ b = 2 :=
sorry

end solve_ab_eq_l313_313574


namespace david_lewis_meeting_point_l313_313478

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

end david_lewis_meeting_point_l313_313478


namespace fraction_to_decimal_equiv_l313_313956

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313956


namespace crackers_per_person_l313_313153

variable (darrenA : Nat)
variable (darrenB : Nat)
variable (aCrackersPerBox : Nat)
variable (bCrackersPerBox : Nat)
variable (calvinA : Nat)
variable (calvinB : Nat)
variable (totalPeople : Nat)

-- Definitions based on the conditions
def totalDarrenCrackers := darrenA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCalvinA := 2 * darrenA - 1
def totalCalvinCrackers := totalCalvinA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCrackers := totalDarrenCrackers + totalCalvinCrackers
def crackersPerPerson := totalCrackers / totalPeople

-- The theorem to prove the question equals the answer given the conditions
theorem crackers_per_person :
  darrenA = 4 →
  darrenB = 2 →
  aCrackersPerBox = 24 →
  bCrackersPerBox = 30 →
  calvinA = 7 →
  calvinB = darrenB →
  totalPeople = 5 →
  crackersPerPerson = 76 :=
by
  intros
  sorry

end crackers_per_person_l313_313153


namespace cab_driver_income_l313_313848

theorem cab_driver_income (x2 : ℕ) :
  (600 + x2 + 450 + 400 + 800) / 5 = 500 → x2 = 250 :=
by
  sorry

end cab_driver_income_l313_313848


namespace smallest_x_mod_equation_l313_313309

theorem smallest_x_mod_equation : ∃ x : ℕ, 42 * x + 10 ≡ 5 [MOD 15] ∧ ∀ y : ℕ, 42 * y + 10 ≡ 5 [MOD 15] → x ≤ y :=
by
sorry

end smallest_x_mod_equation_l313_313309


namespace equal_sets_l313_313446

def M : Set ℝ := {x | x^2 + 16 = 0}
def N : Set ℝ := {x | x^2 + 6 = 0}

theorem equal_sets : M = N := by
  sorry

end equal_sets_l313_313446


namespace major_axis_endpoints_of_ellipse_l313_313760

theorem major_axis_endpoints_of_ellipse :
  ∀ x y, 6 * x^2 + y^2 = 6 ↔ (x = 0 ∧ (y = -Real.sqrt 6 ∨ y = Real.sqrt 6)) :=
by
  -- Proof
  sorry

end major_axis_endpoints_of_ellipse_l313_313760


namespace num_boys_is_22_l313_313291

variable (girls boys total_students : ℕ)

-- Conditions
axiom h1 : total_students = 41
axiom h2 : boys = girls + 3
axiom h3 : total_students = girls + boys

-- Goal: Prove that the number of boys is 22
theorem num_boys_is_22 : boys = 22 :=
by
  sorry

end num_boys_is_22_l313_313291


namespace solution_set_inequality_range_of_t_l313_313982

noncomputable def f (x : ℝ) : ℝ := |x| - 2 * |x + 3|

-- Problem (1)
theorem solution_set_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | -4 ≤ x ∧ x ≤ - (8 / 3) } :=
by
  sorry

-- Problem (2)
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x - |3 * t - 2| ≥ 0) ↔ (- (1 / 3) ≤ t ∧ t ≤ 5 / 3) :=
by
  sorry

end solution_set_inequality_range_of_t_l313_313982


namespace points_on_line_initial_l313_313563

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l313_313563


namespace value_of_x_l313_313426

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l313_313426


namespace fraction_to_decimal_l313_313939

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end fraction_to_decimal_l313_313939


namespace harold_august_tips_fraction_l313_313275

noncomputable def tips_fraction : ℚ :=
  let A : ℚ := sorry -- average monthly tips for March to July and September
  let august_tips := 6 * A -- Tips for August
  let total_tips := 6 * A + 6 * A -- Total tips for all months worked
  august_tips / total_tips

theorem harold_august_tips_fraction :
  tips_fraction = 1 / 2 :=
by
  sorry

end harold_august_tips_fraction_l313_313275


namespace probability_reaching_target_l313_313240

-- Definitions for points
def Point : Type := (ℤ × ℤ × ℤ)

-- Definitions for vertices of the pyramid
def E : Point := (10, 10, 0)
def A : Point := (10, -10, 0)
def R : Point := (-10, -10, 0)
def L : Point := (-10, 10, 0)
def Y : Point := (0, 0, 10)

-- Movement rules
def possibleMoves (p : Point) : List Point := 
  let (x, y, z) := p
  [(x, y, z-1), (x+1, y, z-1), (x-1, y, z-1),
   (x, y+1, z-1), (x, y-1, z-1), 
   (x+1, y+1, z-1), (x-1, y+1, z-1),
   (x+1, y-1, z-1), (x-1, y-1, z-1)]

-- Starting at point Y
def initialPosition : Point := Y

-- Theorem: Probability of reaching (8, 9, 0)
theorem probability_reaching_target : 
  let target := (8, 9, 0)
  let steps := 10
  let probability := 550 / (9^10 : ℚ)
  Sean_probability initialPosition target steps = probability :=
sorry

end probability_reaching_target_l313_313240


namespace bananas_left_correct_l313_313156

def initial_bananas : ℕ := 12
def eaten_bananas : ℕ := 1
def bananas_left (initial eaten : ℕ) := initial - eaten

theorem bananas_left_correct : bananas_left initial_bananas eaten_bananas = 11 :=
by
  sorry

end bananas_left_correct_l313_313156


namespace part1_x1_part1_x0_part1_xneg2_general_inequality_l313_313752

-- Prove inequality for specific values of x
theorem part1_x1 : - (1/2 : ℝ) * (1: ℝ)^2 + 2 * (1: ℝ) < -(1: ℝ) + 5 := by
  sorry

theorem part1_x0 : - (1/2 : ℝ) * (0: ℝ)^2 + 2 * (0: ℝ) < -(0: ℝ) + 5 := by
  sorry

theorem part1_xneg2 : - (1/2 : ℝ) * (-2: ℝ)^2 + 2 * (-2: ℝ) < -(-2: ℝ) + 5 := by
  sorry

-- Prove general inequality for all real x
theorem general_inequality (x : ℝ) : - (1/2 : ℝ) * x^2 + 2 * x < -x + 5 := by
  sorry

end part1_x1_part1_x0_part1_xneg2_general_inequality_l313_313752


namespace solve_system_of_equations_l313_313242

theorem solve_system_of_equations
  (a b c : ℝ) (x y z : ℝ)
  (h1 : x + y = a)
  (h2 : y + z = b)
  (h3 : z + x = c) :
  x = (a + c - b) / 2 ∧ y = (a + b - c) / 2 ∧ z = (b + c - a) / 2 :=
by
  sorry

end solve_system_of_equations_l313_313242


namespace sum_of_squares_of_roots_l313_313878

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y ^ 3 - 8 * y ^ 2 + 9 * y + 2 = 0 → y ≥ 0) →
  let s : ℝ := 8
  let p : ℝ := 9
  let q : ℝ := -2
  (s ^ 2 - 2 * p = 46) :=
by
  -- Placeholders for definitions extracted from the conditions
  -- and additional necessary let-bindings from Vieta's formulas
  intro h
  sorry

end sum_of_squares_of_roots_l313_313878


namespace time_to_pass_pole_l313_313632

def length_of_train : ℝ := 240
def length_of_platform : ℝ := 650
def time_to_pass_platform : ℝ := 89

theorem time_to_pass_pole (length_of_train length_of_platform time_to_pass_platform : ℝ) 
  (h_train : length_of_train = 240)
  (h_platform : length_of_platform = 650)
  (h_time : time_to_pass_platform = 89)
  : (length_of_train / ((length_of_train + length_of_platform) / time_to_pass_platform)) = 24 := by
  -- Let the speed of the train be v, hence
  -- v = (length_of_train + length_of_platform) / time_to_pass_platform
  -- What we need to prove is  
  -- length_of_train / v = 24
  sorry

end time_to_pass_pole_l313_313632


namespace arith_seq_sum_7_8_9_l313_313040

noncomputable def S_n (a : Nat → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n.succ).sum a

def arith_seq (a : Nat → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

theorem arith_seq_sum_7_8_9 (a : Nat → ℝ) (h_arith : arith_seq a)
    (h_S3 : S_n a 3 = 8) (h_S6 : S_n a 6 = 7) : 
  (a 7 + a 8 + a 9) = 1 / 8 := 
  sorry

end arith_seq_sum_7_8_9_l313_313040


namespace prime_sum_of_primes_l313_313725

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem prime_sum_of_primes (p q r s : ℕ) :
  distinct_primes p q r s →
  is_prime (p + q + r + s) →
  is_square (p^2 + q * s) →
  is_square (p^2 + q * r) →
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11) :=
by
  sorry

end prime_sum_of_primes_l313_313725


namespace positive_integer_conditions_l313_313360

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) : 
  (∃ k : ℕ, k > 0 ∧ 4 * p + 28 = k * (3 * p - 7)) ↔ (p = 6 ∨ p = 28) :=
by
  sorry

end positive_integer_conditions_l313_313360


namespace solutions_equation1_solutions_equation2_l313_313816

-- Definition for the first equation
def equation1 (x : ℝ) : Prop := 4 * x^2 - 9 = 0

-- Definition for the second equation
def equation2 (x : ℝ) : Prop := 2 * x^2 - 3 * x - 5 = 0

theorem solutions_equation1 (x : ℝ) :
  equation1 x ↔ (x = 3 / 2 ∨ x = -3 / 2) := 
  by sorry

theorem solutions_equation2 (x : ℝ) :
  equation2 x ↔ (x = 1 ∨ x = 5 / 2) := 
  by sorry

end solutions_equation1_solutions_equation2_l313_313816


namespace fraction_to_decimal_l313_313920

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313920


namespace remainder_sum_l313_313271

theorem remainder_sum (a b c d : ℕ) 
  (h_a : a % 30 = 15) 
  (h_b : b % 30 = 7) 
  (h_c : c % 30 = 22) 
  (h_d : d % 30 = 6) : 
  (a + b + c + d) % 30 = 20 := 
by
  sorry

end remainder_sum_l313_313271


namespace x_fourth_minus_inv_fourth_l313_313497

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l313_313497


namespace calc_4_op_3_l313_313657

def specific_op (m n : ℕ) : ℕ := n^2 - m

theorem calc_4_op_3 :
  specific_op 4 3 = 5 :=
by
  sorry

end calc_4_op_3_l313_313657


namespace math_problem_l313_313385

theorem math_problem (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
  a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
  b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 :=
by
  sorry

end math_problem_l313_313385


namespace lion_king_box_office_earnings_l313_313577

-- Definitions and conditions
def cost_lion_king : ℕ := 10  -- Lion King cost 10 million
def cost_star_wars : ℕ := 25  -- Star Wars cost 25 million
def earnings_star_wars : ℕ := 405  -- Star Wars earned 405 million

-- Calculate profit of Star Wars
def profit_star_wars : ℕ := earnings_star_wars - cost_star_wars

-- Define the profit of The Lion King, given it's half of Star Wars' profit
def profit_lion_king : ℕ := profit_star_wars / 2

-- Calculate the earnings of The Lion King
def earnings_lion_king : ℕ := cost_lion_king + profit_lion_king

-- Theorem to prove
theorem lion_king_box_office_earnings : earnings_lion_king = 200 :=
by
  sorry

end lion_king_box_office_earnings_l313_313577


namespace greatest_divisor_remainders_l313_313120

theorem greatest_divisor_remainders (d : ℤ) :
  d > 0 → (1657 % d = 10) → (2037 % d = 7) → d = 1 :=
by
  intros hdg h1657 h2037
  sorry

end greatest_divisor_remainders_l313_313120


namespace condition_p_neither_sufficient_nor_necessary_l313_313325

theorem condition_p_neither_sufficient_nor_necessary
  (x : ℝ) :
  (1/x ≤ 1 → x^2 - 2 * x ≥ 0) = false ∧ 
  (x^2 - 2 * x ≥ 0 → 1/x ≤ 1) = false := 
by 
  sorry

end condition_p_neither_sufficient_nor_necessary_l313_313325


namespace solve_inequality_l313_313413

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end solve_inequality_l313_313413


namespace interest_percentage_face_value_l313_313076

def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_percentage_selling_price : ℝ := 0.065

def interest_amount : ℝ := interest_percentage_selling_price * selling_price

theorem interest_percentage_face_value :
  (interest_amount / face_value) * 100 = 8 :=
by
  sorry

end interest_percentage_face_value_l313_313076


namespace minimum_students_using_both_l313_313520

theorem minimum_students_using_both (n L T x : ℕ) 
  (H1: 3 * n = 7 * L) 
  (H2: 5 * n = 6 * T) 
  (H3: n = 42) 
  (H4: n = L + T - x) : 
  x = 11 := 
by 
  sorry

end minimum_students_using_both_l313_313520


namespace convert_fraction_to_decimal_l313_313955

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313955


namespace cost_of_bench_l313_313850

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end cost_of_bench_l313_313850


namespace fraction_to_decimal_l313_313914

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313914


namespace trees_total_count_l313_313433

theorem trees_total_count (D P : ℕ) 
  (h1 : D = 350 ∨ P = 350)
  (h2 : 300 * D + 225 * P = 217500) :
  D + P = 850 :=
by
  sorry

end trees_total_count_l313_313433


namespace range_of_a_l313_313222

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l313_313222


namespace max_plus_min_value_of_f_l313_313190

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_value_of_f :
  let M := ⨆ x, f x
  let m := ⨅ x, f x
  M + m = 4 :=
by 
  sorry

end max_plus_min_value_of_f_l313_313190


namespace meeting_point_l313_313828

theorem meeting_point (n : ℕ) (petya_start vasya_start petya_end vasya_end meeting_lamp : ℕ) : 
  n = 100 → petya_start = 1 → vasya_start = 100 → petya_end = 22 → vasya_end = 88 → meeting_lamp = 64 :=
by
  intros h_n h_p_start h_v_start h_p_end h_v_end
  sorry

end meeting_point_l313_313828


namespace ratio_of_toys_l313_313682

theorem ratio_of_toys (initial_stuffed_animals initial_action_figures initial_board_games initial_puzzles : ℕ)
  (joel_added sister_added : ℕ) (total_donated : ℕ) :
  initial_stuffed_animals = 18 →
  initial_action_figures = 42 →
  initial_board_games = 2 →
  initial_puzzles = 13 →
  joel_added = 22 →
  total_donated = 108 →
  (total_donated - (initial_stuffed_animals + initial_action_figures + initial_board_games + initial_puzzles) - joel_added)
    = sister_added →
  (joel_added : ℚ) / sister_added = 2 :=
by sorry

end ratio_of_toys_l313_313682


namespace expansion_number_of_terms_l313_313637

theorem expansion_number_of_terms (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 5) : (A.card * B.card = 20) :=
by 
  sorry

end expansion_number_of_terms_l313_313637


namespace prime_geq_7_div_240_l313_313238

theorem prime_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 240 ∣ p^4 - 1 :=
sorry

end prime_geq_7_div_240_l313_313238


namespace distinct_positive_factors_of_1320_l313_313356

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l313_313356


namespace solve_x4_minus_inv_x4_l313_313505

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l313_313505


namespace points_on_line_l313_313573

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313573


namespace fraction_to_decimal_l313_313917

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313917


namespace range_of_a_l313_313661

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ x₀, -2 ≤ x₀ ∧ x₀ ≤ 2 ∧ (a * x₀ - 1 = f x)) →
  a ∈ Set.Iic (-5/2) ∪ Set.Ici (5/2) :=
sorry

end range_of_a_l313_313661


namespace distinct_four_digit_count_l313_313338

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l313_313338


namespace cells_sequence_exists_l313_313009

theorem cells_sequence_exists :
  ∃ (a : Fin 10 → ℚ), 
    a 0 = 9 ∧
    a 8 = 5 ∧
    (∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14) :=
sorry

end cells_sequence_exists_l313_313009


namespace new_point_in_fourth_quadrant_l313_313366

-- Define the initial point P with coordinates (-3, 2)
def P : ℝ × ℝ := (-3, 2)

-- Define the move operation: 4 units to the right and 6 units down
def move (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 4, p.2 - 6)

-- Define the new point after the move operation
def P' : ℝ × ℝ := move P

-- Prove that the new point P' is in the fourth quadrant
theorem new_point_in_fourth_quadrant (x y : ℝ) (h : P' = (x, y)) : x > 0 ∧ y < 0 :=
by
  sorry

end new_point_in_fourth_quadrant_l313_313366


namespace intersection_point_l313_313762

def line_parametric (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, -1 + 3 * t, -3 + 2 * t)

def on_plane (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

theorem intersection_point : ∃ t, line_parametric t = (5, 2, -1) ∧ on_plane 5 2 (-1) :=
by
  use 1
  sorry

end intersection_point_l313_313762


namespace x_fourth_minus_inv_fourth_l313_313494

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l313_313494


namespace percentage_sum_l313_313186

theorem percentage_sum {A B : ℝ} 
  (hA : 0.40 * A = 160) 
  (hB : (2/3) * B = 160) : 
  0.60 * (A + B) = 384 :=
by
  sorry

end percentage_sum_l313_313186


namespace fraction_to_decimal_l313_313884

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313884


namespace behavior_of_g_l313_313879

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 4 * x ^ 2 + 5

theorem behavior_of_g :
  (∀ x, (∃ M, x ≥ M → g x < 0)) ∧ (∀ x, (∃ N, x ≤ N → g x > 0)) :=
by
  sorry

end behavior_of_g_l313_313879


namespace time_to_fill_tank_l313_313598

-- Definitions for conditions
def pipe_a := 50
def pipe_b := 75
def pipe_c := 100

-- Definition for the combined rate and time to fill the tank
theorem time_to_fill_tank : 
  (1 / pipe_a + 1 / pipe_b + 1 / pipe_c) * (300 / 13) = 1 := 
by
  sorry

end time_to_fill_tank_l313_313598


namespace sara_jim_savings_eq_l313_313091

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end sara_jim_savings_eq_l313_313091


namespace negation_of_universal_statement_l313_313656

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by 
  -- Proof steps would be added here
  sorry

end negation_of_universal_statement_l313_313656


namespace sara_jim_savings_eq_l313_313092

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end sara_jim_savings_eq_l313_313092


namespace min_diff_two_composite_sum_91_l313_313281

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

-- Minimum positive difference between two composite numbers that sum up to 91
theorem min_diff_two_composite_sum_91 : ∃ a b : ℕ, 
  is_composite a ∧ 
  is_composite b ∧ 
  a + b = 91 ∧ 
  b - a = 1 :=
by
  sorry

end min_diff_two_composite_sum_91_l313_313281


namespace fraction_to_decimal_l313_313890

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end fraction_to_decimal_l313_313890


namespace arthur_walks_distance_l313_313016

variables (blocks_east blocks_north : ℕ) 
variable (distance_per_block : ℝ)
variable (total_blocks : ℕ)
def total_distance (blocks : ℕ) (distance_per_block : ℝ) : ℝ :=
  blocks * distance_per_block

theorem arthur_walks_distance (h_east : blocks_east = 8) (h_north : blocks_north = 10) 
    (h_total_blocks : total_blocks = blocks_east + blocks_north)
    (h_distance_per_block : distance_per_block = 1 / 4) :
  total_distance total_blocks distance_per_block = 4.5 :=
by {
  -- Here we specify the proof, but as required, we use sorry to skip it.
  sorry
}

end arthur_walks_distance_l313_313016


namespace larger_number_is_8_l313_313054

-- Define the conditions
def is_twice (x y : ℕ) : Prop := x = 2 * y
def product_is_40 (x y : ℕ) : Prop := x * y = 40
def sum_is_14 (x y : ℕ) : Prop := x + y = 14

-- The proof statement
theorem larger_number_is_8 (x y : ℕ) (h1 : is_twice x y) (h2 : product_is_40 x y) (h3 : sum_is_14 x y) : x = 8 :=
  sorry

end larger_number_is_8_l313_313054


namespace area_of_triangle_tangent_at_pi_div_two_l313_313244

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem area_of_triangle_tangent_at_pi_div_two :
  let x := Real.pi / 2
  let slope := 1 + Real.cos x
  let point := (x, f x)
  let intercept_y := f x - slope * x
  let x_intercept := -intercept_y / slope
  let y_intercept := intercept_y
  (1 / 2) * x_intercept * y_intercept = 1 / 2 := 
by
  sorry

end area_of_triangle_tangent_at_pi_div_two_l313_313244


namespace anya_lost_games_l313_313315

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l313_313315


namespace total_waiting_days_l313_313541

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end total_waiting_days_l313_313541


namespace fraction_to_decimal_l313_313924

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l313_313924


namespace initial_points_l313_313554

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l313_313554


namespace anya_lost_games_l313_313316

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l313_313316


namespace intersection_eq_l313_313778

-- Define the sets M and N
def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The statement to prove
theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l313_313778


namespace last_digit_of_frac_l313_313266

noncomputable theory
open_locale classical

theorem last_digit_of_frac (N : ℤ) (hN : N = 2^15) :
  (∃ k : ℤ, (1 / (N : ℝ)) = (5^15) / 10^15 * 10^(-15 * k)) → last_digit((1 / (N : ℝ))) = 5 :=
by {
  sorry
}

end last_digit_of_frac_l313_313266


namespace x_fourth_minus_inv_fourth_l313_313495

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l313_313495


namespace sum_of_remainders_mod_53_l313_313049

theorem sum_of_remainders_mod_53 (d e f : ℕ) (hd : d % 53 = 19) (he : e % 53 = 33) (hf : f % 53 = 14) : 
  (d + e + f) % 53 = 13 :=
by
  sorry

end sum_of_remainders_mod_53_l313_313049


namespace fraction_to_decimal_l313_313904

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313904


namespace cube_side_length_l313_313296

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end cube_side_length_l313_313296


namespace solution_to_problem_l313_313417

theorem solution_to_problem (f : ℕ → ℕ) 
  (h1 : f 2 = 20)
  (h2 : ∀ n : ℕ, 0 < n → f (2 * n) + n * f 2 = f (2 * n + 2)) :
  f 10 = 220 :=
by
  sorry

end solution_to_problem_l313_313417


namespace multiply_105_95_l313_313305

theorem multiply_105_95 : 105 * 95 = 9975 :=
by
  sorry

end multiply_105_95_l313_313305


namespace ellipse_equation_l313_313172

noncomputable def point := (ℝ × ℝ)

theorem ellipse_equation (a b : ℝ) (P Q : point) (h1 : a > b) (h2: b > 0) (e : ℝ) (h3 : e = 1/2)
  (h4 : P = (2, 3)) (h5 : Q = (2, -3))
  (h6 : (P.1^2)/(a^2) + (P.2^2)/(b^2) = 1) (h7 : (Q.1^2)/(a^2) + (Q.2^2)/(b^2) = 1) :
  (∀ x y: ℝ, (x^2/16 + y^2/12 = 1) ↔ (x^2/a^2 + y^2/b^2 = 1)) :=
sorry

end ellipse_equation_l313_313172


namespace necessary_and_sufficient_for_perpendicular_l313_313612

theorem necessary_and_sufficient_for_perpendicular (a : ℝ) :
  (a = -2) ↔ (∀ (x y : ℝ), x + 2 * y = 0 → ax + y = 1 → false) :=
by
  sorry

end necessary_and_sufficient_for_perpendicular_l313_313612


namespace hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l313_313650

-- Definitions for shooting events for clarity
def hits_9_rings (s : String) := s = "9 rings"
def hits_8_rings (s : String) := s = "8 rings"

def hits_10_rings (s : String) := s = "10 rings"

def hits_target (s: String) := s = "hits target"
def does_not_hit_target (s: String) := s = "does not hit target"

-- Mutual exclusivity:
def mutually_exclusive (E1 E2 : Prop) := ¬ (E1 ∧ E2)

-- Problem 1:
theorem hits_9_and_8_mutually_exclusive :
  mutually_exclusive (hits_9_rings "9 rings") (hits_8_rings "8 rings") :=
sorry

-- Problem 2:
theorem hits_10_and_8_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_10_rings "10 rings" ) (hits_8_rings "8 rings") :=
sorry

-- Problem 3:
theorem both_hit_target_and_neither_hit_target_mutually_exclusive :
  mutually_exclusive (hits_target "both hit target") (does_not_hit_target "neither hit target") :=
sorry

-- Problem 4:
theorem at_least_one_hits_and_A_not_B_does_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_target "at least one hits target") (does_not_hit_target "A not but B does hit target") :=
sorry

end hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l313_313650


namespace vanessa_deleted_files_l313_313833

theorem vanessa_deleted_files (initial_music_files : ℕ) (initial_video_files : ℕ) (files_left : ℕ) (files_deleted : ℕ) :
  initial_music_files = 13 → initial_video_files = 30 → files_left = 33 → 
  files_deleted = (initial_music_files + initial_video_files) - files_left → files_deleted = 10 :=
by
  sorry

end vanessa_deleted_files_l313_313833


namespace term_position_in_sequence_l313_313645

theorem term_position_in_sequence (n : ℕ) (h1 : n > 0) (h2 : 3 * n + 1 = 40) : n = 13 :=
by
  sorry

end term_position_in_sequence_l313_313645


namespace distinct_four_digit_numbers_l313_313334

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l313_313334


namespace find_c_l313_313188

theorem find_c (c : ℝ) : (∃ a : ℝ, (x : ℝ) → (x^2 + 80*x + c = (x + a)^2)) → (c = 1600) := by
  sorry

end find_c_l313_313188


namespace tan_alpha_neg_four_over_three_l313_313770

theorem tan_alpha_neg_four_over_three (α : ℝ) (h_cos : Real.cos α = -3/5) (h_alpha_range : α ∈ Set.Ioo (-π) 0) : Real.tan α = -4/3 :=
  sorry

end tan_alpha_neg_four_over_three_l313_313770


namespace value_of_a_b_c_l313_313972

theorem value_of_a_b_c (a b c : ℚ) (h₁ : |a| = 2) (h₂ : |b| = 2) (h₃ : |c| = 3) (h₄ : b < 0) (h₅ : 0 < a) :
  a + b + c = 3 ∨ a + b + c = -3 :=
by
  sorry

end value_of_a_b_c_l313_313972


namespace find_m_value_l313_313869

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end find_m_value_l313_313869


namespace shoveling_hours_l313_313528

def initial_rate := 25

def rate_decrease := 2

def snow_volume := 6 * 12 * 3

def shoveling_rate (hour : ℕ) : ℕ :=
  if hour = 0 then initial_rate
  else initial_rate - rate_decrease * hour

def cumulative_snow (hour : ℕ) : ℕ :=
  if hour = 0 then snow_volume - shoveling_rate 0
  else cumulative_snow (hour - 1) - shoveling_rate hour

theorem shoveling_hours : cumulative_snow 12 ≠ 0 ∧ cumulative_snow 13 = 47 := by
  sorry

end shoveling_hours_l313_313528


namespace appropriate_sampling_method_l313_313127

theorem appropriate_sampling_method (total_staff teachers admin_staff logistics_personnel sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : teachers = 120)
  (h3 : admin_staff = 16)
  (h4 : logistics_personnel = 24)
  (h5 : sample_size = 20) :
  (sample_method : String) -> sample_method = "Stratified sampling" :=
sorry

end appropriate_sampling_method_l313_313127


namespace power_difference_l313_313500

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l313_313500


namespace max_sequence_length_l313_313068

theorem max_sequence_length (a : ℕ → ℝ) (n : ℕ)
  (H1 : ∀ k : ℕ, k + 4 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4)) < 0)
  (H2 : ∀ k : ℕ, k + 8 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8)) > 0) : 
  n ≤ 12 :=
sorry

end max_sequence_length_l313_313068


namespace monotonic_increasing_range_l313_313214

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ x y, x ∈ s → y ∈ s → x < y → f x ≤ f y

def function_f (a x : ℝ) : ℝ := a^x + (1 + a)^x

theorem monotonic_increasing_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1)
  (h3 : isMonotonicIncreasing (function_f a) (Set.Ioi 0)) :
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end monotonic_increasing_range_l313_313214


namespace find_room_length_l313_313251

variable (w : ℝ) (C : ℝ) (r : ℝ)

theorem find_room_length (h_w : w = 4.75) (h_C : C = 29925) (h_r : r = 900) : (C / r) / w = 7 := by
  sorry

end find_room_length_l313_313251


namespace fraction_to_decimal_l313_313913

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313913


namespace trigonometric_identity_l313_313508

open Real

theorem trigonometric_identity (θ : ℝ) (h : π / 4 < θ ∧ θ < π / 2) :
  2 * cos θ + sqrt (1 - 2 * sin (π - θ) * cos θ) = sin θ + cos θ :=
sorry

end trigonometric_identity_l313_313508


namespace circle_radius_l313_313164

theorem circle_radius :
  ∃ r : ℝ, ∀ x y : ℝ, (x^2 - 8 * x + y^2 + 4 * y + 16 = 0) → r = 2 :=
sorry

end circle_radius_l313_313164


namespace derivative_of_f_l313_313613

noncomputable section

open Real

variable {x : ℝ}

def f (x : ℝ) : ℝ :=
  (2 * x + 3) ^ 4 * arcsin (1 / (2 * x + 3)) + (2 / 3) * (4 * x ^ 2 + 12 * x + 11) * sqrt (x ^ 2 + 3 * x + 2)

theorem derivative_of_f (h : 2 * x + 3 > 0) : deriv f x = 8 * (2 * x + 3) ^ 3 * arcsin (1 / (2 * x + 3)) :=
by
  sorry -- Proof not required

end derivative_of_f_l313_313613


namespace find_a_b_l313_313981

noncomputable def f (a b x : ℝ) := b * a^x

def passes_through (a b : ℝ) : Prop :=
  f a b 1 = 27 ∧ f a b (-1) = 3

theorem find_a_b (a b : ℝ) (h : passes_through a b) : 
  a = 3 ∧ b = 9 :=
  sorry

end find_a_b_l313_313981


namespace son_age_l313_313856

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end son_age_l313_313856


namespace digit_sum_9_l313_313733

def digits := {n : ℕ // n < 10}

theorem digit_sum_9 (a b : digits) 
  (h1 : (4 * 100) + (a.1 * 10) + 3 + 984 = (1 * 1000) + (3 * 100) + (b.1 * 10) + 7) 
  (h2 : (1 + b.1) - (3 + 7) % 11 = 0) 
: a.1 + b.1 = 9 :=
sorry

end digit_sum_9_l313_313733


namespace A_finishes_in_20_days_l313_313273

-- Define the rates and the work
variable (A B W : ℝ)

-- First condition: A and B together can finish the work in 12 days
axiom together_rate : (A + B) * 12 = W

-- Second condition: B alone can finish the work in 30.000000000000007 days
axiom B_rate : B * 30.000000000000007 = W

-- Prove that A alone can finish the work in 20 days
theorem A_finishes_in_20_days : (1 / A) = 20 :=
by 
  sorry

end A_finishes_in_20_days_l313_313273


namespace range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l313_313665

/-- There exists a real number x such that 2x^2 + (m-1)x + 1/2 ≤ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1 / 2 ≤ 0

/-- The curve C1: x^2/m^2 + y^2/(2m+8) = 1 represents an ellipse with foci on the x-axis -/
def proposition_q (m : ℝ) : Prop :=
  m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0

/-- The curve C2: x^2/(m-t) + y^2/(m-t-1) = 1 represents a hyperbola -/
def proposition_s (m t : ℝ) : Prop :=
  (m - t) * (m - t - 1) < 0

/-- Find the range of values for m if p and q are true -/
theorem range_of_m_if_p_and_q_true (m : ℝ) :
  proposition_p m ∧ proposition_q m ↔ (-4 < m ∧ m < -2) ∨ m > 4 :=
  sorry

/-- Find the range of values for t if q is a necessary but not sufficient condition for s -/
theorem range_of_t_if_q_necessary_for_s (m t : ℝ) :
  (∀ m, proposition_q m → proposition_s m t) ∧ ¬(proposition_s m t → proposition_q m) ↔ 
  (-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4 :=
  sorry

end range_of_m_if_p_and_q_true_range_of_t_if_q_necessary_for_s_l313_313665


namespace find_width_of_chalkboard_l313_313689

variable (w : ℝ) (l : ℝ)

-- Given conditions
def length_eq_twice_width (w l : ℝ) : Prop := l = 2 * w
def area_eq_eighteen (w l : ℝ) : Prop := w * l = 18

-- Theorem statement
theorem find_width_of_chalkboard (h1 : length_eq_twice_width w l) (h2 : area_eq_eighteen w l) : w = 3 :=
by sorry

end find_width_of_chalkboard_l313_313689


namespace eventually_one_student_answers_yes_l313_313601

-- Conditions and Definitions
variable (a b r₁ r₂ : ℕ)
variable (h₁ : r₁ ≠ r₂)   -- r₁ and r₂ are distinct
variable (h₂ : r₁ = a + b ∨ r₂ = a + b) -- One of r₁ or r₂ is the sum a + b
variable (h₃ : a > 0) -- a is a positive integer
variable (h₄ : b > 0) -- b is a positive integer

theorem eventually_one_student_answers_yes (a b r₁ r₂ : ℕ) (h₁ : r₁ ≠ r₂) (h₂ : r₁ = a + b ∨ r₂ = a + b) (h₃ : a > 0) (h₄ : b > 0) :
  ∃ n : ℕ, (∃ c : ℕ, (r₁ = c + b ∨ r₂ = c + b) ∧ (c = a ∨ c ≤ r₁ ∨ c ≤ r₂)) ∨ 
  (∃ c : ℕ, (r₁ = a + c ∨ r₂ = a + c) ∧ (c = b ∨ c ≤ r₁ ∨ c ≤ r₂)) :=
sorry

end eventually_one_student_answers_yes_l313_313601


namespace global_phone_company_customers_l313_313131

theorem global_phone_company_customers :
  (total_customers = 25000) →
  (us_percentage = 0.20) →
  (canada_percentage = 0.12) →
  (australia_percentage = 0.15) →
  (uk_percentage = 0.08) →
  (india_percentage = 0.05) →
  (us_customers = total_customers * us_percentage) →
  (canada_customers = total_customers * canada_percentage) →
  (australia_customers = total_customers * australia_percentage) →
  (uk_customers = total_customers * uk_percentage) →
  (india_customers = total_customers * india_percentage) →
  (mentioned_countries_customers = us_customers + canada_customers + australia_customers + uk_customers + india_customers) →
  (other_countries_customers = total_customers - mentioned_countries_customers) →
  (other_countries_customers = 10000) ∧ (us_customers / other_countries_customers = 1 / 2) :=
by
  -- The further proof steps would go here if needed
  sorry

end global_phone_company_customers_l313_313131


namespace partners_count_l313_313004

theorem partners_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 :=
sorry

end partners_count_l313_313004


namespace small_cone_altitude_l313_313005

noncomputable def frustum_height : ℝ := 18
noncomputable def lower_base_area : ℝ := 400 * Real.pi
noncomputable def upper_base_area : ℝ := 100 * Real.pi

theorem small_cone_altitude (h_frustum : frustum_height = 18) 
    (A_lower : lower_base_area = 400 * Real.pi) 
    (A_upper : upper_base_area = 100 * Real.pi) : 
    ∃ (h_small_cone : ℝ), h_small_cone = 18 := 
by
  sorry

end small_cone_altitude_l313_313005


namespace fraction_equals_decimal_l313_313932

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313932


namespace pow_mult_rule_l313_313301

variable (x : ℝ)

theorem pow_mult_rule : (x^3) * (x^2) = x^5 :=
by sorry

end pow_mult_rule_l313_313301


namespace solve_x4_minus_inv_x4_l313_313502

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l313_313502


namespace numbers_with_special_remainder_property_l313_313707

theorem numbers_with_special_remainder_property (n : ℕ) :
  (∀ q : ℕ, q > 0 → n % (q ^ 2) < (q ^ 2) / 2) ↔ (n = 1 ∨ n = 4) := 
by
  sorry

end numbers_with_special_remainder_property_l313_313707


namespace find_x_l313_313765

noncomputable def x : ℝ :=
  0.49

theorem find_x (h : (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt x) = 2.507936507936508) : 
  x = 0.49 :=
sorry

end find_x_l313_313765


namespace intersection_complement_A_B_l313_313537

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_A_B : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_B_l313_313537


namespace classify_curve_l313_313154

-- Define the curve equation
def curve_equation (m : ℝ) : Prop := 
  ∃ (x y : ℝ), ((m - 3) * x^2 + (5 - m) * y^2 = 1)

-- Define the conditions for types of curves
def is_circle (m : ℝ) : Prop := 
  m = 4 ∧ (curve_equation m)

def is_ellipse (m : ℝ) : Prop := 
  (3 < m ∧ m < 5 ∧ m ≠ 4) ∧ (curve_equation m)

def is_hyperbola (m : ℝ) : Prop := 
  ((m > 5 ∨ m < 3) ∧ (curve_equation m))

-- Main theorem stating the type of curve
theorem classify_curve (m : ℝ) : 
  (is_circle m) ∨ (is_ellipse m) ∨ (is_hyperbola m) :=
sorry

end classify_curve_l313_313154


namespace rate_of_decrease_l313_313456

theorem rate_of_decrease (x : ℝ) (h : 400 * (1 - x) ^ 2 = 361) : x = 0.05 :=
by {
  sorry -- The proof is omitted as requested.
}

end rate_of_decrease_l313_313456


namespace symmetric_about_x_axis_l313_313510

noncomputable def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

theorem symmetric_about_x_axis (a : ℝ) :
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f a x = - g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end symmetric_about_x_axis_l313_313510


namespace fraction_to_decimal_l313_313893

theorem fraction_to_decimal (h : (5 : ℚ) / 16 = 0.3125) : (5 : ℚ) / 16 = 0.3125 :=
  by sorry

end fraction_to_decimal_l313_313893


namespace negation_existential_proposition_l313_313102

theorem negation_existential_proposition :
  ¬(∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by sorry

end negation_existential_proposition_l313_313102


namespace geometric_sequence_eighth_term_l313_313836

noncomputable def a_8 : ℕ :=
  let a₁ := 8
  let r := 2
  a₁ * r^(8-1)

theorem geometric_sequence_eighth_term : a_8 = 1024 := by
  sorry

end geometric_sequence_eighth_term_l313_313836


namespace fifty_percent_of_2002_is_1001_l313_313455

theorem fifty_percent_of_2002_is_1001 :
  (1 / 2) * 2002 = 1001 :=
sorry

end fifty_percent_of_2002_is_1001_l313_313455


namespace relationship_between_abc_l313_313319

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_between_abc (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x := by
  have ha : a = (1/3) ^ 3 := rfl
  have hb : b x = x ^ 3 := rfl
  have hc : c x = Real.log x := rfl
  split
  { sorry }  -- Proof that a < c x
  { sorry }  -- Proof that c x < b x

end relationship_between_abc_l313_313319


namespace exists_contiguous_figure_l313_313651

-- Definition of the type for different types of rhombuses
inductive RhombusType
| wide
| narrow

-- Definition of a figure composed of rhombuses
structure Figure where
  count_wide : ℕ
  count_narrow : ℕ
  connected : Prop

-- Statement of the proof problem
theorem exists_contiguous_figure : ∃ (f : Figure), f.count_wide = 3 ∧ f.count_narrow = 8 ∧ f.connected :=
sorry

end exists_contiguous_figure_l313_313651


namespace david_first_six_l313_313299

def prob_six := (1:ℚ) / 6
def prob_not_six := (5:ℚ) / 6

def prob_david_first_six_cycle : ℚ :=
  prob_not_six * prob_not_six * prob_not_six * prob_six

def prob_no_six_cycle : ℚ :=
  prob_not_six ^ 4

def infinite_series_sum (a r: ℚ) : ℚ := 
  a / (1 - r)

theorem david_first_six :
  infinite_series_sum prob_david_first_six_cycle prob_no_six_cycle = 125 / 671 :=
by
  sorry

end david_first_six_l313_313299


namespace problem_l313_313660

theorem problem (f : ℕ → ℕ → ℕ) (h0 : f 1 1 = 1) (h1 : ∀ m n, f m n ∈ {x | x > 0}) 
  (h2 : ∀ m n, f m (n + 1) = f m n + 2) (h3 : ∀ m, f (m + 1) 1 = 2 * f m 1) : 
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 :=
sorry

end problem_l313_313660


namespace find_subtracted_number_l313_313741

theorem find_subtracted_number (x y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  have hx : 2 * 129 - y = 110 := by
    rw [h1] at h2
    exact h2
  linarith

end find_subtracted_number_l313_313741


namespace points_on_line_l313_313560

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l313_313560


namespace fraction_to_decimal_l313_313919

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l313_313919


namespace tenth_term_is_98415_over_262144_l313_313474

def first_term : ℚ := 5
def common_ratio : ℚ := 3 / 4

def tenth_term_geom_seq (a r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem tenth_term_is_98415_over_262144 :
  tenth_term_geom_seq first_term common_ratio 10 = 98415 / 262144 :=
sorry

end tenth_term_is_98415_over_262144_l313_313474


namespace dots_not_visible_l313_313318

-- Define the sum of numbers on a single die
def sum_die_faces : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Define the sum of numbers on four dice
def total_dots_on_four_dice : ℕ := 4 * sum_die_faces

-- List the visible numbers
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 5, 6]

-- Calculate the sum of visible numbers
def sum_visible_numbers : ℕ := (visible_numbers.sum)

-- Define the math proof problem
theorem dots_not_visible : total_dots_on_four_dice - sum_visible_numbers = 53 := by
  sorry

end dots_not_visible_l313_313318


namespace range_of_a_l313_313517

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end range_of_a_l313_313517


namespace lunch_to_read_ratio_l313_313254

theorem lunch_to_read_ratio 
  (total_pages : ℕ) (pages_per_hour : ℕ) (lunch_hours : ℕ)
  (h₁ : total_pages = 4000)
  (h₂ : pages_per_hour = 250)
  (h₃ : lunch_hours = 4) :
  lunch_hours / (total_pages / pages_per_hour) = 1 / 4 := by
  sorry

end lunch_to_read_ratio_l313_313254


namespace fraction_eq_zero_iff_x_eq_6_l313_313704

theorem fraction_eq_zero_iff_x_eq_6 (x : ℝ) : (x - 6) / (5 * x) = 0 ↔ x = 6 :=
by
  sorry

end fraction_eq_zero_iff_x_eq_6_l313_313704


namespace harrys_total_cost_l313_313781

def cost_large_pizza : ℕ := 14
def cost_per_topping : ℕ := 2
def number_of_pizzas : ℕ := 2
def number_of_toppings_per_pizza : ℕ := 3
def tip_percentage : ℚ := 0.25

def total_cost (c_pizza c_topping tip_percent : ℚ) (n_pizza n_topping : ℕ) : ℚ :=
  let inital_cost := (c_pizza + c_topping * n_topping) * n_pizza
  let tip := inital_cost * tip_percent
  inital_cost + tip

theorem harrys_total_cost : total_cost 14 2 0.25 2 3 = 50 := 
  sorry

end harrys_total_cost_l313_313781


namespace final_net_worth_l313_313234

noncomputable def initial_cash_A := (20000 : ℤ)
noncomputable def initial_cash_B := (22000 : ℤ)
noncomputable def house_value := (20000 : ℤ)
noncomputable def vehicle_value := (10000 : ℤ)

noncomputable def transaction_1_cash_A := initial_cash_A + 25000
noncomputable def transaction_1_cash_B := initial_cash_B - 25000

noncomputable def transaction_2_cash_A := transaction_1_cash_A - 12000
noncomputable def transaction_2_cash_B := transaction_1_cash_B + 12000

noncomputable def transaction_3_cash_A := transaction_2_cash_A + 18000
noncomputable def transaction_3_cash_B := transaction_2_cash_B - 18000

noncomputable def transaction_4_cash_A := transaction_3_cash_A + 9000
noncomputable def transaction_4_cash_B := transaction_3_cash_B + 9000

noncomputable def final_value_A := transaction_4_cash_A
noncomputable def final_value_B := transaction_4_cash_B + house_value + vehicle_value

theorem final_net_worth :
  final_value_A - initial_cash_A = 40000 ∧ final_value_B - initial_cash_B = 8000 :=
by
  sorry

end final_net_worth_l313_313234


namespace number_of_valid_pairings_l313_313256

-- Definition for the problem
def validPairingCount (n : ℕ) (k: ℕ) : ℕ :=
  sorry -- Calculating the valid number of pairings is deferred

-- The problem statement to be proven:
theorem number_of_valid_pairings : validPairingCount 12 3 = 14 :=
sorry

end number_of_valid_pairings_l313_313256


namespace thomas_spends_40000_in_a_decade_l313_313434

/-- 
Thomas spends 4k dollars every year on his car insurance.
One decade is 10 years.
-/
def spending_per_year : ℕ := 4000

def years_in_a_decade : ℕ := 10

/-- 
We need to prove that the total amount Thomas spends in a decade on car insurance equals $40,000.
-/
theorem thomas_spends_40000_in_a_decade : spending_per_year * years_in_a_decade = 40000 := by
  sorry

end thomas_spends_40000_in_a_decade_l313_313434


namespace monotonic_increasing_range_l313_313206

theorem monotonic_increasing_range (a : ℝ) (h0 : 0 < a) (h1 : a < 1) :
  (∀ x > 0, (a^x + (1 + a)^x)) → a ∈ (Set.Icc ((Real.sqrt 5 - 1) / 2) 1) ∧ a < 1 :=
sorry

end monotonic_increasing_range_l313_313206


namespace probability_of_red_ball_l313_313069

theorem probability_of_red_ball :
  let total_balls := 9
  let red_balls := 6
  let probability := (red_balls : ℚ) / total_balls
  probability = (2 : ℚ) / 3 :=
by
  sorry

end probability_of_red_ball_l313_313069


namespace digits_sum_is_31_l313_313524

noncomputable def digits_sum_proof (A B C D E F G : ℕ) : Prop :=
  (1000 * A + 100 * B + 10 * C + D + 100 * E + 10 * F + G = 2020) ∧ 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧
  (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧
  (E ≠ F) ∧ (E ≠ G) ∧
  (F ≠ G)

theorem digits_sum_is_31 (A B C D E F G : ℕ) (h : digits_sum_proof A B C D E F G) : 
  A + B + C + D + E + F + G = 31 :=
sorry

end digits_sum_is_31_l313_313524


namespace interval_sum_l313_313262

theorem interval_sum (a b : ℝ) (h : ∀ x,  |3 * x - 80| ≤ |2 * x - 105| ↔ (a ≤ x ∧ x ≤ b)) :
  a + b = 12 :=
sorry

end interval_sum_l313_313262


namespace distinct_four_digit_numbers_count_l313_313342

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l313_313342


namespace quadratic_has_two_roots_l313_313200

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end quadratic_has_two_roots_l313_313200


namespace find_x_l313_313298

theorem find_x : ∃ x : ℝ, (3 * (x + 2 - 6)) / 4 = 3 ∧ x = 8 :=
by
  sorry

end find_x_l313_313298


namespace fraction_to_decimal_equiv_l313_313964

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end fraction_to_decimal_equiv_l313_313964


namespace contrapositive_statement_l313_313245

-- Condition definitions
def P (x : ℝ) := x^2 < 1
def Q (x : ℝ) := -1 < x ∧ x < 1
def not_Q (x : ℝ) := x ≤ -1 ∨ x ≥ 1
def not_P (x : ℝ) := x^2 ≥ 1

theorem contrapositive_statement (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_statement_l313_313245


namespace part1_growth_rate_part2_new_price_l313_313282

-- Definitions based on conditions
def purchase_price : ℕ := 30
def selling_price : ℕ := 40
def january_sales : ℕ := 400
def march_sales : ℕ := 576
def growth_rate (x : ℝ) : Prop := january_sales * (1 + x)^2 = march_sales

-- Part (1): Prove the monthly average growth rate
theorem part1_growth_rate : 
  ∃ (x : ℝ), growth_rate x ∧ x = 0.2 :=
by
  sorry

-- Definitions for part (2) - based on the second condition
def price_reduction (y : ℝ) : Prop := (selling_price - y - purchase_price) * (march_sales + 12 * y) = 4800

-- Part (2): Prove the new price for April
theorem part2_new_price :
  ∃ (y : ℝ), price_reduction y ∧ (selling_price - y) = 38 :=
by
  sorry

end part1_growth_rate_part2_new_price_l313_313282


namespace monotonically_increasing_range_l313_313224

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l313_313224


namespace percentage_running_wickets_l313_313274

-- Conditions provided as definitions and assumptions in Lean
def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def boundary_runs (b : ℕ) := b * 4
def six_runs (s : ℕ) := s * 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries := boundary_runs boundaries
def runs_from_sixes := six_runs sixes
def runs_not_from_boundaries_and_sixes := total_runs - (runs_from_boundaries + runs_from_sixes)

-- Proof that the percentage of the total score by running between the wickets is 50%
theorem percentage_running_wickets :
  (runs_not_from_boundaries_and_sixes : ℝ) / (total_runs : ℝ) * 100 = 50 :=
by
  sorry

end percentage_running_wickets_l313_313274


namespace find_divisor_l313_313276

theorem find_divisor (d q r : ℕ) (h1 : d = 265) (h2 : q = 12) (h3 : r = 1) :
  ∃ x : ℕ, d = (x * q) + r ∧ x = 22 :=
by {
  sorry
}

end find_divisor_l313_313276


namespace music_tool_cost_l313_313681

namespace BandCost

def trumpet_cost : ℝ := 149.16
def song_book_cost : ℝ := 4.14
def total_spent : ℝ := 163.28

theorem music_tool_cost : (total_spent - (trumpet_cost + song_book_cost)) = 9.98 :=
by
  sorry

end music_tool_cost_l313_313681


namespace probability_different_colors_l313_313107

/-- There are 5 blue chips and 3 yellow chips in a bag. One chip is drawn from the bag and placed
back into the bag. A second chip is then drawn. Prove that the probability of the two selected chips
being of different colors is 15/32. -/
theorem probability_different_colors : 
  let total_chips := 8
  let blue_chips := 5
  let yellow_chips := 3
  let prob_blue_then_yellow := (blue_chips/total_chips) * (yellow_chips/total_chips)
  let prob_yellow_then_blue := (yellow_chips/total_chips) * (blue_chips/total_chips)
  prob_blue_then_yellow + prob_yellow_then_blue = 15/32 := by
  sorry

end probability_different_colors_l313_313107


namespace tyler_bought_10_erasers_l313_313602

/--
Given that Tyler initially has $100, buys 8 scissors for $5 each, buys some erasers for $4 each,
and has $20 remaining after these purchases, prove that he bought 10 erasers.
-/
theorem tyler_bought_10_erasers : ∀ (initial_money scissors_cost erasers_cost remaining_money : ℕ), 
  initial_money = 100 →
  scissors_cost = 5 →
  erasers_cost = 4 →
  remaining_money = 20 →
  ∃ (scissors_count erasers_count : ℕ),
    scissors_count = 8 ∧ 
    initial_money - scissors_count * scissors_cost - erasers_count * erasers_cost = remaining_money ∧ 
    erasers_count = 10 :=
by
  intros
  sorry

end tyler_bought_10_erasers_l313_313602


namespace problem_statement_l313_313784

variables {a c b d : ℝ} {x y q z : ℕ}

-- Given conditions:
def condition1 (a c : ℝ) (x q : ℕ) : Prop := a^(x + 1) = c^(q + 2)
def condition2 (a c : ℝ) (y z : ℕ) : Prop := c^(y + 3) = a^(z+ 4)

-- Goal statement
theorem problem_statement (a c : ℝ) (x y q z : ℕ) (h1 : condition1 a c x q) (h2 : condition2 a c y z) :
  (q + 2) * (z + 4) = (y + 3) * (x + 1) :=
sorry

end problem_statement_l313_313784


namespace range_of_a_l313_313220

theorem range_of_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : ∀ x > 0, deriv (λ x, a^x + (1 + a)^x) x ≥ 0) : 
  a ∈ Set.Ico ((Real.sqrt 5 - 1) / 2) 1 :=
sorry

end range_of_a_l313_313220


namespace no_positive_integers_solution_l313_313093

theorem no_positive_integers_solution (m n : ℕ) (hm : m > 0) (hn : n > 0) : 4 * m * (m + 1) ≠ n * (n + 1) := 
by
  sorry

end no_positive_integers_solution_l313_313093


namespace find_k_l313_313994

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l313_313994


namespace order_of_a_b_c_l313_313800

noncomputable def a : ℝ := Real.log 2 / Real.log 3 -- a = log_3 2
noncomputable def b : ℝ := Real.log 2 -- b = ln 2
noncomputable def c : ℝ := Real.sqrt 5 -- c = 5^(1/2)

theorem order_of_a_b_c : a < b ∧ b < c := by
  sorry

end order_of_a_b_c_l313_313800


namespace highest_sum_vertex_l313_313585

theorem highest_sum_vertex (a b c d e f : ℕ) (h₀ : a + d = 8) (h₁ : b + e = 8) (h₂ : c + f = 8) : 
  a + b + c ≤ 11 ∧ b + c + d ≤ 11 ∧ c + d + e ≤ 11 ∧ d + e + f ≤ 11 ∧ e + f + a ≤ 11 ∧ f + a + b ≤ 11 :=
sorry

end highest_sum_vertex_l313_313585


namespace like_terms_monomials_l313_313191

theorem like_terms_monomials (a b : ℕ) (x y : ℝ) (c : ℝ) (H1 : x^(a+1) * y^3 = c * y^b * x^2) : a = 1 ∧ b = 3 :=
by
  -- Proof will be provided here
  sorry

end like_terms_monomials_l313_313191


namespace grandson_age_l313_313544

-- Define the ages of Markus, his son, and his grandson
variables (M S G : ℕ)

-- Conditions given in the problem
axiom h1 : M = 2 * S
axiom h2 : S = 2 * G
axiom h3 : M + S + G = 140

-- Theorem to prove that the age of Markus's grandson is 20 years
theorem grandson_age : G = 20 :=
by
  sorry

end grandson_age_l313_313544


namespace machine_performance_l313_313806

noncomputable def machine_A_data : List ℕ :=
  [4, 1, 0, 2, 2, 1, 3, 1, 2, 4]

noncomputable def machine_B_data : List ℕ :=
  [2, 3, 1, 1, 3, 2, 2, 1, 2, 3]

noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

noncomputable def variance (data : List ℕ) (mean : ℝ) : ℝ :=
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

theorem machine_performance :
  let mean_A := mean machine_A_data
  let mean_B := mean machine_B_data
  let variance_A := variance machine_A_data mean_A
  let variance_B := variance machine_B_data mean_B
  mean_A = 2 ∧ mean_B = 2 ∧ variance_A = 1.6 ∧ variance_B = 0.6 ∧ variance_B < variance_A := 
sorry

end machine_performance_l313_313806


namespace woman_lawyer_probability_l313_313124

noncomputable def probability_of_woman_lawyer : ℚ :=
  let total_members : ℚ := 100
  let women_percentage : ℚ := 0.80
  let lawyer_percentage_women : ℚ := 0.40
  let women_members := women_percentage * total_members
  let women_lawyers := lawyer_percentage_women * women_members
  let probability := women_lawyers / total_members
  probability

theorem woman_lawyer_probability :
  probability_of_woman_lawyer = 0.32 := by
  sorry

end woman_lawyer_probability_l313_313124


namespace natalie_list_count_l313_313084

theorem natalie_list_count : ∀ n : ℕ, (15 ≤ n ∧ n ≤ 225) → ((225 - 15 + 1) = 211) :=
by
  intros n h
  sorry

end natalie_list_count_l313_313084


namespace fraction_to_decimal_l313_313909

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313909


namespace f_eq_n_for_all_n_l313_313534

noncomputable def f : ℕ → ℕ := sorry

axiom f_pos_int_valued (n : ℕ) (h : 0 < n) : f n = f n

axiom f_2_eq_2 : f 2 = 2

axiom f_mul_prop (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n

axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : f m > f n

theorem f_eq_n_for_all_n (n : ℕ) (hn : 0 < n) : f n = n := sorry

end f_eq_n_for_all_n_l313_313534


namespace sum_squares_nonpositive_l313_313037

theorem sum_squares_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ac ≤ 0 :=
by {
  sorry
}

end sum_squares_nonpositive_l313_313037


namespace expected_value_8_sided_die_l313_313130

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8] in
  let divisible_by_3 := [3, 6] in
  let prob_div_3 := 1 / 4 in
  let prob_not_div_3 := 3 / 4 in
  let winnings_div_3 := (3 + 6) / 4 in
  let winnings_not_div_3 := 0 in
  let expected_value := winnings_div_3 + winnings_not_div_3 in
  expected_value = 2.25 :=
by
  sorry

end expected_value_8_sided_die_l313_313130


namespace coefficient_of_x_in_expansion_l313_313581

theorem coefficient_of_x_in_expansion : 
  (Polynomial.coeff (((X ^ 2 + 3 * X + 2) ^ 6) : Polynomial ℤ) 1) = 576 := 
by 
  sorry

end coefficient_of_x_in_expansion_l313_313581


namespace max_chord_length_line_eq_orthogonal_vectors_line_eq_l313_313772

-- Definitions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def point_P (x y : ℝ) : Prop := x = 2 ∧ y = 1
def line_eq (slope intercept x y : ℝ) : Prop := y = slope * x + intercept

-- Problem 1: Prove the equation of line l that maximizes the length of chord AB
theorem max_chord_length_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq 1 (-1) x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq 1 (-1) x y) :=
by sorry

-- Problem 2: Prove the equation of line l given orthogonality condition of vectors
theorem orthogonal_vectors_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq (-1) 3 x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq (-1) 3 x y) :=
by sorry

end max_chord_length_line_eq_orthogonal_vectors_line_eq_l313_313772


namespace find_smallest_N_l313_313693

def smallest_possible_N (N : ℕ) : Prop :=
  ∃ (W : Fin N → ℝ), 
  (∀ i j, W i ≤ 1.25 * W j ∧ W j ≤ 1.25 * W i) ∧ 
  (∃ (P : Fin 10 → Finset (Fin N)), ∀ i j, i ≤ j →
    P i ≠ ∅ ∧ 
    Finset.sum (P i) W = Finset.sum (P j) W) ∧
  (∃ (V : Fin 11 → Finset (Fin N)), ∀ i j, i ≤ j →
    V i ≠ ∅ ∧ 
    Finset.sum (V i) W = Finset.sum (V j) W)

theorem find_smallest_N : smallest_possible_N 50 :=
sorry

end find_smallest_N_l313_313693


namespace sum_pn_equals_target_l313_313971

-- Given p_n is the probability that all n people place their drink in a cup holder.
noncomputable def p : ℕ → ℝ 
| 0     := 1
| 1     := 1
| n + 2 := (1 / (2 * (n + 2))) * p (n + 1) + (1 / (2 * (n + 2))) * (Finset.sum (Finset.range (n + 1)) (λ k, p k * p (n - k)))

-- Define the generating function P(x)
noncomputable def P (x : ℝ) : ℝ := (Real.exp (x / 2)) / (2 - Real.exp (x / 2))

theorem sum_pn_equals_target : (∑' n : ℕ, p (n + 1)) = (2 * Real.sqrt Real.exp 1 - 2) / (2 - Real.sqrt Real.exp 1) :=
by
  sorry

end sum_pn_equals_target_l313_313971


namespace regular_polygons_from_cube_intersection_l313_313442

noncomputable def cube : Type := sorry  -- Define a 3D cube type
noncomputable def plane : Type := sorry  -- Define a plane type

-- Define what it means for a polygon to be regular (equilateral and equiangular)
def is_regular_polygon (polygon : Type) : Prop := sorry

-- Define a function that describes the intersection of a plane with a cube,
-- resulting in a polygon
noncomputable def intersection (c : cube) (p : plane) : Type := sorry

-- Define predicates for the specific regular polygons: triangle, quadrilateral, and hexagon
def is_triangle (polygon : Type) : Prop := sorry
def is_quadrilateral (polygon : Type) : Prop := sorry
def is_hexagon (polygon : Type) : Prop := sorry

-- Ensure these predicates imply regular polygons
axiom triangle_is_regular : ∀ (t : Type), is_triangle t → is_regular_polygon t
axiom quadrilateral_is_regular : ∀ (q : Type), is_quadrilateral q → is_regular_polygon q
axiom hexagon_is_regular : ∀ (h : Type), is_hexagon h → is_regular_polygon h

-- The main theorem statement
theorem regular_polygons_from_cube_intersection (c : cube) (p : plane) :
  is_regular_polygon (intersection c p) →
  is_triangle (intersection c p) ∨ is_quadrilateral (intersection c p) ∨ is_hexagon (intersection c p) :=
sorry

end regular_polygons_from_cube_intersection_l313_313442


namespace symmetric_point_l313_313106

theorem symmetric_point (x y : ℝ) (a b : ℝ) :
  (x = 3 ∧ y = 9 ∧ a = -1 ∧ b = -3) ∧ (∀ k: ℝ, k ≠ 0 → (y - 9 = k * (x - 3)) ∧ 
  ((x - 3)^2 + (y - 9)^2 = (a - 3)^2 + (b - 9)^2) ∧ 
  (x >= 0 → (a >= 0 ↔ x = 3) ∧ (b >= 0 ↔ y = 9))) :=
by
  sorry

end symmetric_point_l313_313106


namespace evaluate_fraction_subtraction_l313_313883

theorem evaluate_fraction_subtraction :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = (11 / 30) :=
by
  sorry

end evaluate_fraction_subtraction_l313_313883


namespace evaluate_f_at_5_l313_313439

def f (x : ℕ) : ℕ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem evaluate_f_at_5 : f 5 = 4881 :=
by
-- proof
sorry

end evaluate_f_at_5_l313_313439


namespace greatest_expression_value_l313_313597

noncomputable def greatest_expression : ℝ := 0.9986095661846496

theorem greatest_expression_value : greatest_expression = 0.9986095661846496 :=
by
  -- proof goes here
  sorry

end greatest_expression_value_l313_313597


namespace intersection_of_planes_is_line_l313_313272

-- Define the conditions as Lean 4 statements
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y + z - 8 = 0
def plane2 (x y z : ℝ) : Prop := x - 2 * y - 2 * z + 1 = 0

-- Define the canonical form of the line as a Lean 4 proposition
def canonical_line (x y z : ℝ) : Prop := 
  (x - 3) / -4 = y / 5 ∧ y / 5 = (z - 2) / -7

-- The theorem to state equivalence between conditions and canonical line equations
theorem intersection_of_planes_is_line :
  ∀ (x y z : ℝ), plane1 x y z → plane2 x y z → canonical_line x y z :=
by
  intros x y z h1 h2
  -- TODO: Insert proof here
  sorry

end intersection_of_planes_is_line_l313_313272


namespace distinct_four_digit_numbers_count_l313_313343

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l313_313343


namespace prism_unique_triple_l313_313007

theorem prism_unique_triple :
  ∃! (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ b = 2000 ∧
                  (∃ b' c', b' = 2000 ∧ c' = 2000 ∧
                  (∃ k : ℚ, k = 1/2 ∧
                  (∃ x y z, x = a / 2 ∧ y = 1000 ∧ z = c / 2 ∧ a = 2000 ∧ c = 2000)))
/- The proof is omitted for this statement. -/
:= sorry

end prism_unique_triple_l313_313007


namespace average_tickets_sold_by_male_members_l313_313720

theorem average_tickets_sold_by_male_members 
  (M F : ℕ)
  (total_average : ℕ)
  (female_average : ℕ)
  (ratio : ℕ × ℕ)
  (h1 : total_average = 66)
  (h2 : female_average = 70)
  (h3 : ratio = (1, 2))
  (h4 : F = 2 * M)
  (h5 : (M + F) * total_average = M * r + F * female_average) :
  r = 58 :=
sorry

end average_tickets_sold_by_male_members_l313_313720


namespace min_initial_bags_l313_313780

theorem min_initial_bags :
  ∃ x : ℕ, (∃ y : ℕ, (y + 90 = 2 * (x - 90) ∧ x + (11 * x - 1620) / 7 = 6 * (2 * x - 270 - (11 * x - 1620) / 7))
             ∧ x = 153) :=
by { sorry }

end min_initial_bags_l313_313780


namespace panels_per_home_panels_needed_per_home_l313_313757

theorem panels_per_home (P : ℕ) (total_homes : ℕ) (shortfall : ℕ) (homes_installed : ℕ) :
  total_homes = 20 →
  shortfall = 50 →
  homes_installed = 15 →
  (P - shortfall) / homes_installed = P / total_homes →
  P = 200 :=
by
  intro h1 h2 h3 h4
  sorry

theorem panels_needed_per_home :
  (200 / 20) = 10 :=
by
  sorry

end panels_per_home_panels_needed_per_home_l313_313757


namespace simplify_fractional_equation_l313_313717

theorem simplify_fractional_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 2) : (x / (x - 2) - 2 = 3 / (2 - x)) → (x - 2 * (x - 2) = -3) :=
by
  sorry

end simplify_fractional_equation_l313_313717


namespace range_of_b_l313_313045

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := x^2 + b * x + c

def A (b c : ℝ) := {x : ℝ | f x b c = 0}
def B (b c : ℝ) := {x : ℝ | f (f x b c) b c = 0}

theorem range_of_b (b c : ℝ) (h : ∃ x₀ : ℝ, x₀ ∈ B b c ∧ x₀ ∉ A b c) :
  b < 0 ∨ b ≥ 4 := 
sorry

end range_of_b_l313_313045


namespace first_player_wins_l313_313713

-- Define the initial conditions
def initial_pieces : ℕ := 1
def final_pieces (m n : ℕ) : ℕ := m * n
def num_moves (pieces : ℕ) : ℕ := pieces - 1

-- Theorem statement: Given the initial dimensions and the game rules,
-- prove that the first player will win.
theorem first_player_wins (m n : ℕ) (h_m : m = 6) (h_n : n = 8) : 
  (num_moves (final_pieces m n)) % 2 = 0 → false :=
by
  -- The solution details and the proof will be here.
  sorry

end first_player_wins_l313_313713


namespace find_divisors_of_10_pow_10_sum_157_l313_313261

theorem find_divisors_of_10_pow_10_sum_157 
  (x y : ℕ) 
  (hx₁ : 0 < x) 
  (hy₁ : 0 < y) 
  (hx₂ : x ∣ 10^10) 
  (hy₂ : y ∣ 10^10) 
  (hxy₁ : x ≠ y) 
  (hxy₂ : x + y = 157) : 
  (x = 32 ∧ y = 125) ∨ (x = 125 ∧ y = 32) := 
by
  sorry

end find_divisors_of_10_pow_10_sum_157_l313_313261


namespace Cara_possible_pairs_l313_313640

-- Define the conditions and the final goal.
theorem Cara_possible_pairs : ∃ p : Nat, p = Nat.choose 7 2 ∧ p = 21 :=
by
  sorry

end Cara_possible_pairs_l313_313640


namespace cost_per_day_is_18_l313_313097

def cost_per_day_first_week (x : ℕ) : Prop :=
  let cost_per_day_rest_week := 12
  let total_days := 23
  let total_cost := 318
  let first_week_days := 7
  let remaining_days := total_days - first_week_days
  (first_week_days * x) + (remaining_days * cost_per_day_rest_week) = total_cost

theorem cost_per_day_is_18 : cost_per_day_first_week 18 :=
  sorry

end cost_per_day_is_18_l313_313097


namespace proof_mn_squared_l313_313170

theorem proof_mn_squared (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end proof_mn_squared_l313_313170


namespace positive_number_solution_l313_313278

theorem positive_number_solution (x : ℚ) (hx : 0 < x) (h : x * x^2 * (1 / x) = 100 / 81) : x = 10 / 9 :=
sorry

end positive_number_solution_l313_313278


namespace boat_speed_in_still_water_l313_313607

-- Definitions of the conditions
def with_stream_speed : ℝ := 36
def against_stream_speed : ℝ := 8

-- Let Vb be the speed of the boat in still water, and Vs be the speed of the stream.
variable (Vb Vs : ℝ)

-- Conditions given in the problem
axiom h1 : Vb + Vs = with_stream_speed
axiom h2 : Vb - Vs = against_stream_speed

-- The statement to prove: the speed of the boat in still water is 22 km/h.
theorem boat_speed_in_still_water : Vb = 22 := by
  sorry

end boat_speed_in_still_water_l313_313607


namespace sum_adjacent_angles_pentagon_l313_313460

theorem sum_adjacent_angles_pentagon (n : ℕ) (θ : ℕ) (hn : n = 5) (hθ : θ = 40) :
  let exterior_angle := 360 / n
  let new_adjacent_angle := 180 - (exterior_angle + θ)
  let sum_adjacent_angles := n * new_adjacent_angle
  sum_adjacent_angles = 340 := by
  sorry

end sum_adjacent_angles_pentagon_l313_313460


namespace sum_x_y_z_l313_313422
open Real

theorem sum_x_y_z (a b : ℝ) (h1 : a / b = 98 / 63) (x y z : ℕ) (h2 : (sqrt a) / (sqrt b) = (x * sqrt y) / z) : x + y + z = 18 := 
by
  sorry

end sum_x_y_z_l313_313422


namespace max_handshakes_l313_313258

theorem max_handshakes (n : ℕ) (m : ℕ)
  (h_n : n = 25)
  (h_m : m = 20)
  (h_mem : n - m = 5)
  : ∃ (max_handshakes : ℕ), max_handshakes = 250 :=
by
  sorry

end max_handshakes_l313_313258


namespace geometric_sequence_sixth_term_l313_313072

variable (q : ℕ) (a_2 a_6 : ℕ)

-- Given conditions:
axiom h1 : q = 2
axiom h2 : a_2 = 8

-- Prove that a_6 = 128 where a_n = a_2 * q^(n-2)
theorem geometric_sequence_sixth_term : a_6 = a_2 * q^4 → a_6 = 128 :=
by sorry

end geometric_sequence_sixth_term_l313_313072


namespace employed_males_population_percentage_l313_313381

-- Define the conditions of the problem
variables (P : Type) (population : ℝ) (employed_population : ℝ) (employed_females : ℝ)

-- Assume total population is 100
def total_population : ℝ := 100

-- 70 percent of the population are employed
def employed_population_percentage : ℝ := total_population * 0.70

-- 70 percent of the employed people are females
def employed_females_percentage : ℝ := employed_population_percentage * 0.70

-- 21 percent of the population are employed males
def employed_males_percentage : ℝ := 21

-- Main statement to be proven
theorem employed_males_population_percentage :
  employed_males_percentage = ((employed_population_percentage - employed_females_percentage) / total_population) * 100 :=
sorry

end employed_males_population_percentage_l313_313381


namespace inequality_proof_l313_313974

theorem inequality_proof (a b t : ℝ) (h₀ : 0 < t) (h₁ : t < 1) (h₂ : a * b > 0) : 
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 :=
by
  sorry

end inequality_proof_l313_313974


namespace find_f_of_3_l313_313586

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_of_3 (h : ∀ x : ℝ, x ≠ 0 → f x - 3 * f (1 / x) = 3 ^ x) :
  f 3 = (-27 + 3 * (3 ^ (1 / 3))) / 8 :=
sorry

end find_f_of_3_l313_313586


namespace original_movie_length_l313_313617

theorem original_movie_length (final_length cut_scene original_length : ℕ) 
    (h1 : cut_scene = 3) (h2 : final_length = 57) (h3 : final_length + cut_scene = original_length) : 
  original_length = 60 := 
by 
  -- Proof omitted
  sorry

end original_movie_length_l313_313617


namespace remainders_are_distinct_l313_313432

theorem remainders_are_distinct (a : ℕ → ℕ) (H1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i ≠ a (i % 100 + 1))
  (H2 : ∃ r1 r2 : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i % a (i % 100 + 1) = r1 ∨ a i % a (i % 100 + 1) = r2) :
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 100 → (a (i % 100 + 1) % a i) ≠ (a (j % 100 + 1) % a j) :=
by
  sorry

end remainders_are_distinct_l313_313432


namespace solve_quadratic_equation_l313_313709

theorem solve_quadratic_equation : 
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by
  sorry


end solve_quadratic_equation_l313_313709


namespace xyz_inequality_l313_313239

theorem xyz_inequality (x y z : ℝ) (h_condition : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end xyz_inequality_l313_313239


namespace solve_pond_fish_problem_l313_313610

def pond_fish_problem 
  (tagged_fish : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (total_fish : ℕ) : Prop :=
  (tagged_in_second_catch : ℝ) / second_catch = (tagged_fish : ℝ) / total_fish →
  total_fish = 1750

theorem solve_pond_fish_problem : 
  pond_fish_problem 70 50 2 1750 :=
by
  sorry

end solve_pond_fish_problem_l313_313610


namespace communication_system_connections_l313_313194

theorem communication_system_connections (n : ℕ) (h : ∀ k < 2001, ∃ l < 2001, l ≠ k ∧ k ≠ l) :
  (∀ k < 2001, ∃ l < 2001, k ≠ l) → (n % 2 = 0 ∧ n ≤ 2000) ∨ n = 0 :=
sorry

end communication_system_connections_l313_313194


namespace relationship_M_N_l313_313168

theorem relationship_M_N (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) 
  (M : ℝ) (hM : M = a * b) (N : ℝ) (hN : N = a + b - 1) : M > N :=
by
  sorry

end relationship_M_N_l313_313168


namespace bruce_eggs_lost_l313_313473

theorem bruce_eggs_lost :
  ∀ (initial_eggs remaining_eggs eggs_lost : ℕ), 
  initial_eggs = 75 → remaining_eggs = 5 →
  eggs_lost = initial_eggs - remaining_eggs →
  eggs_lost = 70 :=
by
  intros initial_eggs remaining_eggs eggs_lost h_initial h_remaining h_loss
  sorry

end bruce_eggs_lost_l313_313473


namespace necessary_but_not_sufficient_condition_l313_313326

variables {Point Line Plane : Type} 

-- Definitions for the problem conditions
def is_subset_of (a : Line) (α : Plane) : Prop := sorry
def parallel_plane (a : Line) (β : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- The statement of the problem
theorem necessary_but_not_sufficient_condition (a b : Line) (α β : Plane) 
  (h1 : is_subset_of a α) (h2 : is_subset_of b β) :
  (parallel_plane a β ∧ parallel_plane b α) ↔ 
  (¬ parallel_planes α β ∧ sorry) :=
sorry

end necessary_but_not_sufficient_condition_l313_313326


namespace exists_prime_q_and_positive_n_l313_313684

theorem exists_prime_q_and_positive_n (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ q n : ℕ, Nat.Prime q ∧ q < p ∧ 0 < n ∧ p ∣ (n^2 - q) :=
by
  sorry

end exists_prime_q_and_positive_n_l313_313684


namespace solve_quadratic_inequality_l313_313423

theorem solve_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end solve_quadratic_inequality_l313_313423


namespace fraction_to_decimal_l313_313907

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313907


namespace find_k_of_vectors_orthogonal_l313_313987

variables (k : ℝ)
def vec1 : ℝ × ℝ := (3, 1)
def vec2 : ℝ × ℝ := (1, 0)
def vec3 (k : ℝ) : ℝ × ℝ := (vec1.1 + k * vec2.1, vec1.2 + k * vec2.2)

theorem find_k_of_vectors_orthogonal
  (h : vec1.1 * vec3 k.1 + vec1.2 * vec3 k.2 = 0) :
  k = -10 / 3 :=
by
  sorry

end find_k_of_vectors_orthogonal_l313_313987


namespace cost_price_l313_313723

-- Given conditions
variable (x : ℝ)
def profit (x : ℝ) : ℝ := 54 - x
def loss (x : ℝ) : ℝ := x - 40

-- Claim
theorem cost_price (h : profit x = loss x) : x = 47 :=
by {
  -- This is where the proof would go
  sorry
}

end cost_price_l313_313723


namespace solve_x4_minus_inv_x4_l313_313504

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l313_313504


namespace find_y_l313_313523

-- Definitions for the given conditions
def angle_sum_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def right_triangle (A B : ℝ) : Prop :=
  A + B = 90

-- The main theorem to prove
theorem find_y 
  (angle_ABC : ℝ)
  (angle_BAC : ℝ)
  (angle_DCE : ℝ)
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : right_triangle angle_DCE 30)
  : 30 = 30 :=
sorry

end find_y_l313_313523


namespace sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l313_313782

theorem sixty_percent_of_fifty_minus_thirty_percent_of_thirty : 
  (60 / 100 : ℝ) * 50 - (30 / 100 : ℝ) * 30 = 21 :=
by
  sorry

end sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l313_313782


namespace fraction_equals_decimal_l313_313935

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end fraction_equals_decimal_l313_313935


namespace probability_convex_sequence_l313_313801

open Real

/-- Define the chosen points and conditions -/
variables (x : ℕ → ℝ)

/-- The main theorem stating the probability computation -/
theorem probability_convex_sequence :
  (∀ i, 1 ≤ i ∧ i ≤ 100 → 2 * x i ≥ x (i-1) + x (i+1))
  ∧ x 0 = 0 ∧ x 101 = 0 
  ∧ (∀ i, 1 ≤ i ∧ i ≤ 100 → x i ∈ set.Icc 0 1)
  → 
  sorry =
  1 / (100 * (fact 100)^2) * choose 200 99 :=
sorry

end probability_convex_sequence_l313_313801


namespace general_term_formula_l313_313507

/-- Define that the point (n, S_n) lies on the function y = 2x^2 + x, hence S_n = 2 * n^2 + n --/
def S_n (n : ℕ) : ℕ := 2 * n^2 + n

/-- Define the nth term of the sequence a_n --/
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 4 * n - 1

theorem general_term_formula (n : ℕ) (hn : 0 < n) :
  a_n n = S_n n - S_n (n - 1) :=
by
  sorry

end general_term_formula_l313_313507


namespace auntie_em_parking_l313_313860

theorem auntie_em_parking (total_spaces cars : ℕ) (probability_can_park : ℚ) :
  total_spaces = 20 →
  cars = 15 →
  probability_can_park = 232/323 :=
by
  sorry

end auntie_em_parking_l313_313860


namespace distinct_factors_1320_l313_313345

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l313_313345


namespace parallel_conditions_l313_313390

-- Definitions of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y + 2 = 0

-- Definition of parallel lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → l2 x y

-- Proof statement
theorem parallel_conditions (m : ℝ) :
  parallel (l1 m) (l2 m) ↔ (m = 1 ∨ m = -6) :=
by
  intros
  sorry

end parallel_conditions_l313_313390


namespace sin_add_pi_over_three_l313_313036

theorem sin_add_pi_over_three (α : ℝ) (h : Real.sin (α - 2 * Real.pi / 3) = 1 / 4) : 
  Real.sin (α + Real.pi / 3) = -1 / 4 := by
  sorry

end sin_add_pi_over_three_l313_313036


namespace ducks_and_dogs_total_l313_313462

theorem ducks_and_dogs_total (d g : ℕ) (h1 : d = g + 2) (h2 : 4 * g - 2 * d = 10) : d + g = 16 :=
  sorry

end ducks_and_dogs_total_l313_313462


namespace expression_value_l313_313079

noncomputable def compute_expression (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80

theorem expression_value (ω : ℂ) (h : ω^9 = 1) (h2 : ω ≠ 1)
    : compute_expression ω h h2 = -ω^2 :=
sorry

end expression_value_l313_313079


namespace find_k_l313_313990

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l313_313990


namespace distinct_x_intercepts_l313_313996

-- Given conditions
def polynomial (x : ℝ) : ℝ := (x - 4) * (x^2 + 4 * x + 13)

-- Statement of the problem as a Lean theorem
theorem distinct_x_intercepts : 
  (∃ (x : ℝ), polynomial x = 0 ∧ 
    ∀ (y : ℝ), y ≠ x → polynomial y = 0 → False) :=
  sorry

end distinct_x_intercepts_l313_313996


namespace total_shaded_area_correct_l313_313521
-- Let's import the mathematical library.

-- Define the problem-related conditions.
def first_rectangle_length : ℕ := 4
def first_rectangle_width : ℕ := 15
def second_rectangle_length : ℕ := 5
def second_rectangle_width : ℕ := 12
def third_rectangle_length : ℕ := 2
def third_rectangle_width : ℕ := 2

-- Define the areas based on the problem conditions.
def A1 : ℕ := first_rectangle_length * first_rectangle_width
def A2 : ℕ := second_rectangle_length * second_rectangle_width
def A_overlap_12 : ℕ := first_rectangle_length * second_rectangle_length
def A3 : ℕ := third_rectangle_length * third_rectangle_width

-- Define the total shaded area formula.
def total_shaded_area : ℕ := A1 + A2 - A_overlap_12 + A3

-- Statement of the theorem to prove.
theorem total_shaded_area_correct :
  total_shaded_area = 104 :=
by
  sorry

end total_shaded_area_correct_l313_313521


namespace area_KLMQ_l313_313071

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def JR := 2
def RQ := 3
def JL := 8

def JLMR : Rectangle := {length := JL, width := JR}
def JKQR : Rectangle := {length := RQ, width := JR}

def RM : ℝ := JL
def QM : ℝ := RM - RQ
def LM : ℝ := JR

def KLMQ : Rectangle := {length := QM, width := LM}

theorem area_KLMQ : KLMQ.length * KLMQ.width = 10 :=
by
  sorry

end area_KLMQ_l313_313071


namespace largest_possible_perimeter_l313_313011

theorem largest_possible_perimeter (x : ℕ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : 
  let upper_bound := 16 in
  let range_x := { n : ℕ // n ≥ 3 ∧ n < upper_bound } in
  let largest_side := 15 in
  let perimeter := 7 + 9 + largest_side in
  perimeter = 31 := 
by
  have h : largest_side = 15 := sorry
  exact h

end largest_possible_perimeter_l313_313011


namespace loads_ratio_l313_313265

noncomputable def loads_wednesday : ℕ := 6
noncomputable def loads_friday (T : ℕ) : ℕ := T / 2
noncomputable def loads_saturday : ℕ := loads_wednesday / 3
noncomputable def total_loads_week (T : ℕ) : ℕ := loads_wednesday + T + loads_friday T + loads_saturday

theorem loads_ratio (T : ℕ) (h : total_loads_week T = 26) : T / loads_wednesday = 2 := 
by 
  -- proof steps would go here
  sorry

end loads_ratio_l313_313265


namespace fraction_to_decimal_l313_313906

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := sorry

end fraction_to_decimal_l313_313906


namespace number_of_bad_cards_l313_313203

-- Define the initial conditions
def janessa_initial_cards : ℕ := 4
def father_given_cards : ℕ := 13
def ordered_cards : ℕ := 36
def cards_given_to_dexter : ℕ := 29
def cards_kept_for_herself : ℕ := 20

-- Define the total cards and cards in bad shape calculation
theorem number_of_bad_cards : 
  let total_initial_cards := janessa_initial_cards + father_given_cards;
  let total_cards := total_initial_cards + ordered_cards;
  let total_distributed_cards := cards_given_to_dexter + cards_kept_for_herself;
  total_cards - total_distributed_cards = 4 :=
by {
  sorry
}

end number_of_bad_cards_l313_313203


namespace mean_of_xyz_l313_313699

theorem mean_of_xyz (a b c d e f g x y z : ℝ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 48)
  (h2 : (a + b + c + d + e + f + g + x + y + z) / 10 = 55) :
  (x + y + z) / 3 = 71.33333333333333 :=
by
  sorry

end mean_of_xyz_l313_313699


namespace find_number_l313_313628

theorem find_number (a b : ℕ) (h₁ : a = 555) (h₂ : b = 445) :
  let S := a + b
  let D := a - b
  let Q := 2 * D
  let R := 30
  let N := (S * Q) + R
  N = 220030 := by
  sorry

end find_number_l313_313628


namespace sum_of_surface_areas_of_two_smaller_cuboids_l313_313286

theorem sum_of_surface_areas_of_two_smaller_cuboids
  (L W H : ℝ) (hL : L = 3) (hW : W = 2) (hH : H = 1) :
  ∃ S, (S = 26 ∨ S = 28 ∨ S = 34) ∧ (∀ l w h, (l = L / 2 ∨ w = W / 2 ∨ h = H / 2) →
  (S = 2 * 2 * (l * W + w * H + h * L))) :=
by
  sorry

end sum_of_surface_areas_of_two_smaller_cuboids_l313_313286


namespace football_combinations_l313_313673

theorem football_combinations : 
  ∃ (W D L : ℕ), W + D + L = 15 ∧ 3 * W + D = 33 ∧ 
  (9 ≤ W ∧ W ≤ 11) ∧
  (W = 9 → D = 6 ∧ L = 0) ∧
  (W = 10 → D = 3 ∧ L = 2) ∧
  (W = 11 → D = 0 ∧ L = 4) :=
sorry

end football_combinations_l313_313673


namespace mass_percentage_Ba_in_BaI2_l313_313654

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 :
  (molar_mass_Ba / molar_mass_BaI2 * 100) = 35.11 :=
by
  sorry

end mass_percentage_Ba_in_BaI2_l313_313654


namespace math_problem_l313_313307

theorem math_problem
  (a b c d : ℕ)
  (h1 : a = 234)
  (h2 : b = 205)
  (h3 : c = 86400)
  (h4 : d = 300) :
  (a * b = 47970) ∧ (c / d = 288) :=
by
  sorry

end math_problem_l313_313307


namespace chewbacca_gum_l313_313751

variable {y : ℝ}

theorem chewbacca_gum (h1 : 25 - 2 * y ≠ 0) (h2 : 40 + 4 * y ≠ 0) :
    25 - 2 * y/40 = 25/(40 + 4 * y) → y = 2.5 :=
by
  intros h
  sorry

end chewbacca_gum_l313_313751


namespace ratio_of_stock_values_l313_313230

/-- Definitions and conditions -/
def value_expensive := 78
def shares_expensive := 14
def shares_other := 26
def total_assets := 2106

/-- The proof problem -/
theorem ratio_of_stock_values : 
  ∃ (V_other : ℝ), 26 * V_other = total_assets - (shares_expensive * value_expensive) ∧ 
  (value_expensive / V_other) = 2 :=
by
  sorry

end ratio_of_stock_values_l313_313230


namespace convert_fraction_to_decimal_l313_313952

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l313_313952


namespace radius_ratio_of_spheres_l313_313132

theorem radius_ratio_of_spheres
  (V_large : ℝ) (V_small : ℝ) (r_large r_small : ℝ)
  (h1 : V_large = 324 * π)
  (h2 : V_small = 0.25 * V_large)
  (h3 : (4/3) * π * r_large^3 = V_large)
  (h4 : (4/3) * π * r_small^3 = V_small) :
  (r_small / r_large) = (1/2) := 
sorry

end radius_ratio_of_spheres_l313_313132
