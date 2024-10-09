import Mathlib

namespace train_crosses_signal_pole_in_20_seconds_l346_34627

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 285
noncomputable def total_time_to_cross_platform : ℝ := 39

-- Define the speed of the train
noncomputable def train_speed : ℝ := (train_length + platform_length) / total_time_to_cross_platform

-- Define the expected time to cross the signal pole
noncomputable def time_to_cross_signal_pole : ℝ := train_length / train_speed

theorem train_crosses_signal_pole_in_20_seconds :
  time_to_cross_signal_pole = 20 := by
  sorry

end train_crosses_signal_pole_in_20_seconds_l346_34627


namespace quadratic_rewrite_l346_34636

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 4) (h2 : 2 * d * e = 20) (h3 : e^2 + f = -24) :
  d * e = 10 :=
sorry

end quadratic_rewrite_l346_34636


namespace find_R_l346_34664

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → ¬ (m ∣ n)

theorem find_R :
  ∃ R : ℤ, R > 0 ∧ (∃ Q : ℤ, is_prime (R^3 + 4 * R^2 + (Q - 93) * R + 14 * Q + 10)) ∧ R = 5 :=
  sorry

end find_R_l346_34664


namespace quadratic_intersect_x_axis_l346_34690

theorem quadratic_intersect_x_axis (a : ℝ) : (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) :=
by
  sorry

end quadratic_intersect_x_axis_l346_34690


namespace fraction_product_simplification_l346_34630

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end fraction_product_simplification_l346_34630


namespace chocolate_bar_percentage_l346_34622

theorem chocolate_bar_percentage (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
  (h1 : milk_chocolate = 25) (h2 : dark_chocolate = 25)
  (h3 : almond_chocolate = 25) (h4 : white_chocolate = 25) :
  (milk_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (dark_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (almond_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (white_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 :=
by
  sorry

end chocolate_bar_percentage_l346_34622


namespace num_pairs_satisfying_eq_l346_34689

theorem num_pairs_satisfying_eq :
  ∃ n : ℕ, (n = 256) ∧ (∀ x y : ℤ, x^2 + x * y = 30000000 → true) :=
sorry

end num_pairs_satisfying_eq_l346_34689


namespace evaluate_expression_l346_34666

theorem evaluate_expression :
  4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 :=
by
  sorry

end evaluate_expression_l346_34666


namespace opposite_of_neg_five_l346_34649

theorem opposite_of_neg_five : ∃ (y : ℤ), -5 + y = 0 ∧ y = 5 :=
by
  use 5
  simp

end opposite_of_neg_five_l346_34649


namespace total_area_to_paint_proof_l346_34678

def barn_width : ℝ := 15
def barn_length : ℝ := 20
def barn_height : ℝ := 8
def door_width : ℝ := 3
def door_height : ℝ := 7
def window_width : ℝ := 2
def window_height : ℝ := 4

noncomputable def wall_area (width length height : ℝ) : ℝ := 2 * (width * height + length * height)
noncomputable def door_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num
noncomputable def window_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num

noncomputable def total_area_to_paint : ℝ := 
  let total_wall_area := wall_area barn_width barn_length barn_height
  let total_door_area := door_area door_width door_height 2
  let total_window_area := window_area window_width window_height 3
  let net_wall_area := total_wall_area - total_door_area - total_window_area
  let ceiling_floor_area := barn_width * barn_length * 2
  net_wall_area * 2 + ceiling_floor_area

theorem total_area_to_paint_proof : total_area_to_paint = 1588 := by
  sorry

end total_area_to_paint_proof_l346_34678


namespace max_soap_boxes_l346_34613

theorem max_soap_boxes :
  ∀ (L_carton W_carton H_carton L_soap_box W_soap_box H_soap_box : ℕ)
   (V_carton V_soap_box : ℕ) 
   (h1 : L_carton = 25) 
   (h2 : W_carton = 42)
   (h3 : H_carton = 60) 
   (h4 : L_soap_box = 7)
   (h5 : W_soap_box = 6)
   (h6 : H_soap_box = 10)
   (h7 : V_carton = L_carton * W_carton * H_carton)
   (h8 : V_soap_box = L_soap_box * W_soap_box * H_soap_box),
   V_carton / V_soap_box = 150 :=
by
  intros
  sorry

end max_soap_boxes_l346_34613


namespace product_closest_value_l346_34660

-- Define the constants used in the problem
def a : ℝ := 2.5
def b : ℝ := 53.6
def c : ℝ := 0.4

-- Define the expression and the expected correct answer
def expression : ℝ := a * (b - c)
def correct_answer : ℝ := 133

-- State the theorem that the expression evaluates to the correct answer
theorem product_closest_value : expression = correct_answer :=
by
  sorry

end product_closest_value_l346_34660


namespace ratio_of_volume_to_surface_area_l346_34674

def volume_of_shape (num_cubes : ℕ) : ℕ :=
  -- Volume is simply the number of unit cubes
  num_cubes

def surface_area_of_shape : ℕ :=
  -- Surface area calculation given in the problem and solution
  12  -- edge cubes (4 cubes) with 3 exposed faces each
  + 16  -- side middle cubes (4 cubes) with 4 exposed faces each
  + 1  -- top face of the central cube in the bottom layer
  + 5  -- middle cube in the column with 5 exposed faces
  + 6  -- top cube in the column with all 6 faces exposed

theorem ratio_of_volume_to_surface_area
  (num_cubes : ℕ)
  (h1 : num_cubes = 9) :
  (volume_of_shape num_cubes : ℚ) / (surface_area_of_shape : ℚ) = 9 / 40 :=
by
  sorry

end ratio_of_volume_to_surface_area_l346_34674


namespace arith_seq_a1_a2_a3_sum_l346_34633

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a1_a2_a3_sum (a : ℕ → ℤ) (h_seq : arithmetic_seq a)
  (h1 : a 1 = 2) (h_sum : a 1 + a 2 + a 3 = 18) :
  a 4 + a 5 + a 6 = 54 :=
sorry

end arith_seq_a1_a2_a3_sum_l346_34633


namespace divisor_of_4k2_minus_1_squared_iff_even_l346_34671

-- Define the conditions
variable (k : ℕ) (h_pos : 0 < k)

-- Define the theorem
theorem divisor_of_4k2_minus_1_squared_iff_even :
  ∃ n : ℕ, (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2 ↔ Even k :=
by { sorry }

end divisor_of_4k2_minus_1_squared_iff_even_l346_34671


namespace perimeter_of_rectangular_garden_l346_34638

theorem perimeter_of_rectangular_garden (L W : ℝ) (h : L + W = 28) : 2 * (L + W) = 56 :=
by sorry

end perimeter_of_rectangular_garden_l346_34638


namespace imag_part_of_complex_squared_is_2_l346_34684

-- Define the complex number 1 + i
def complex_num := (1 : ℂ) + (Complex.I : ℂ)

-- Define the squared value of the complex number
def complex_squared := complex_num ^ 2

-- Define the imaginary part of the squared value
def imag_part := complex_squared.im

-- State the theorem
theorem imag_part_of_complex_squared_is_2 : imag_part = 2 := sorry

end imag_part_of_complex_squared_is_2_l346_34684


namespace problem_statement_l346_34656

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable (a b : ℝ)

theorem problem_statement (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, HasDerivAt g (g' x) x)
                         (h3 : ∀ x, f' x < g' x)
                         (h4 : a = Real.log 2 / Real.log 5)
                         (h5 : b = Real.log 3 / Real.log 8) :
                         f a + g b > g a + f b := 
     sorry

end problem_statement_l346_34656


namespace radius_of_circle_l346_34632

noncomputable def radius (α : ℝ) : ℝ :=
  5 / Real.sin (α / 2)

theorem radius_of_circle (c α : ℝ) (h_c : c = 10) :
  (radius α) = 5 / Real.sin (α / 2) := by
  sorry

end radius_of_circle_l346_34632


namespace number_of_liars_l346_34629

/-- There are 25 people in line, each of whom either tells the truth or lies.
The person at the front of the line says: "Everyone behind me is lying."
Everyone else says: "The person directly in front of me is lying."
Prove that the number of liars among these 25 people is 13. -/
theorem number_of_liars : 
  ∀ (persons : Fin 25 → Prop), 
    (persons 0 → ∀ n > 0, ¬persons n) →
    (∀ n : Nat, (1 ≤ n → n < 25 → persons n ↔ ¬persons (n - 1))) →
    (∃ l, l = 13 ∧ ∀ n : Nat, (0 ≤ n → n < 25 → persons n ↔ (n % 2 = 0))) :=
by
  sorry

end number_of_liars_l346_34629


namespace same_remainder_division_l346_34697

theorem same_remainder_division {a m b : ℤ} (r c k : ℤ) 
  (ha : a = b * c + r) (hm : m = b * k + r) : b ∣ (a - m) :=
by
  sorry

end same_remainder_division_l346_34697


namespace ratio_of_seconds_l346_34620

theorem ratio_of_seconds (x : ℕ) :
  (12 : ℕ) / 8 = x / 240 → x = 360 :=
by
  sorry

end ratio_of_seconds_l346_34620


namespace coin_toss_fairness_l346_34654

-- Statement of the problem as a Lean theorem.
theorem coin_toss_fairness (P_Heads P_Tails : ℝ) (h1 : P_Heads = 0.5) (h2 : P_Tails = 0.5) : 
  P_Heads = P_Tails ∧ P_Heads = 0.5 := 
sorry

end coin_toss_fairness_l346_34654


namespace average_of_B_and_C_l346_34611

theorem average_of_B_and_C (x : ℚ) (A B C : ℚ)
  (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  (B + C) / 2 = 93.75 := 
sorry

end average_of_B_and_C_l346_34611


namespace probability_train_or_plane_probability_not_ship_l346_34610

def P_plane : ℝ := 0.2
def P_ship : ℝ := 0.3
def P_train : ℝ := 0.4
def P_car : ℝ := 0.1
def mutually_exclusive : Prop := P_plane + P_ship + P_train + P_car = 1

theorem probability_train_or_plane : mutually_exclusive → P_train + P_plane = 0.6 := by
  intro h
  sorry

theorem probability_not_ship : mutually_exclusive → 1 - P_ship = 0.7 := by
  intro h
  sorry

end probability_train_or_plane_probability_not_ship_l346_34610


namespace units_digit_7_pow_3_pow_5_l346_34673

theorem units_digit_7_pow_3_pow_5 : ∀ (n : ℕ), n % 4 = 3 → ∀ k, 7 ^ k ≡ 3 [MOD 10] :=
by 
    sorry

end units_digit_7_pow_3_pow_5_l346_34673


namespace lines_parallel_if_perpendicular_to_same_plane_l346_34679

variables {Line : Type} {Plane : Type}
variable (a b : Line)
variable (α : Plane)

-- Conditions 
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- Definition for line perpendicular to plane
def lines_parallel (l1 l2 : Line) : Prop := sorry -- Definition for lines parallel

-- Theorem Statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  line_perpendicular_to_plane a α →
  line_perpendicular_to_plane b α →
  lines_parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l346_34679


namespace calculate_number_l346_34685

theorem calculate_number (tens ones tenths hundredths : ℝ) 
  (h_tens : tens = 21) 
  (h_ones : ones = 8) 
  (h_tenths : tenths = 5) 
  (h_hundredths : hundredths = 34) :
  tens * 10 + ones * 1 + tenths * 0.1 + hundredths * 0.01 = 218.84 :=
by
  sorry

end calculate_number_l346_34685


namespace range_of_m_l346_34626

theorem range_of_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) (h4 : 4 / a + 1 / (b - 1) > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l346_34626


namespace adam_earnings_l346_34651

def lawns_to_mow : ℕ := 12
def lawns_forgotten : ℕ := 8
def earnings_per_lawn : ℕ := 9

theorem adam_earnings : (lawns_to_mow - lawns_forgotten) * earnings_per_lawn = 36 := by
  sorry

end adam_earnings_l346_34651


namespace total_expenditure_l346_34652

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end total_expenditure_l346_34652


namespace geometric_series_sum_l346_34644

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  S = 341 / 1024 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  show S = 341 / 1024
  sorry

end geometric_series_sum_l346_34644


namespace a4_equals_9_l346_34640

variable {a : ℕ → ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a4_equals_9 (h_geom : geometric_sequence a)
  (h_roots : ∃ a2 a6 : ℝ, a2^2 - 34 * a2 + 81 = 0 ∧ a6^2 - 34 * a6 + 81 = 0 ∧ a 2 = a2 ∧ a 6 = a6) :
  a 4 = 9 :=
sorry

end a4_equals_9_l346_34640


namespace range_of_m_l346_34699

theorem range_of_m {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h_cond : 1/x + 4/y = 1) : 
  (∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 + 3 * m) ↔
  (m < -4 ∨ 1 < m) := 
sorry

end range_of_m_l346_34699


namespace find_f_of_fraction_l346_34646

noncomputable def f (t : ℝ) : ℝ := sorry

theorem find_f_of_fraction (x : ℝ) (h : f ((1-x^2)/(1+x^2)) = x) :
  f ((2*x)/(1+x^2)) = (1 - x) / (1 + x) ∨ f ((2*x)/(1+x^2)) = (x - 1) / (1 + x) :=
sorry

end find_f_of_fraction_l346_34646


namespace constant_term_expansion_eq_sixty_l346_34628

theorem constant_term_expansion_eq_sixty (a : ℝ) (h : 15 * a = 60) : a = 4 :=
by
  sorry

end constant_term_expansion_eq_sixty_l346_34628


namespace Carson_age_l346_34608

theorem Carson_age {Aunt_Anna_Age : ℕ} (h1 : Aunt_Anna_Age = 60) 
                   {Maria_Age : ℕ} (h2 : Maria_Age = 2 * Aunt_Anna_Age / 3) 
                   {Carson_Age : ℕ} (h3 : Carson_Age = Maria_Age - 7) : 
                   Carson_Age = 33 := by sorry

end Carson_age_l346_34608


namespace number_of_three_digit_numbers_with_5_and_7_l346_34686

def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def containsDigit (n : ℕ) (d : ℕ) : Prop := d ∈ (n.digits 10) 
def hasAtLeastOne5andOne7 (n : ℕ) : Prop := containsDigit n 5 ∧ containsDigit n 7
def totalThreeDigitNumbersWith5and7 : ℕ := 50

theorem number_of_three_digit_numbers_with_5_and_7 :
  ∃ n : ℕ, isThreeDigitNumber n ∧ hasAtLeastOne5andOne7 n → n = 50 := sorry

end number_of_three_digit_numbers_with_5_and_7_l346_34686


namespace speed_of_second_train_l346_34602

noncomputable def speed_of_first_train_kmph := 60 -- km/h
noncomputable def speed_of_first_train_mps := (speed_of_first_train_kmph * 1000) / 3600 -- m/s
noncomputable def length_of_first_train := 145 -- m
noncomputable def length_of_second_train := 165 -- m
noncomputable def time_to_cross := 8 -- seconds
noncomputable def total_distance := length_of_first_train + length_of_second_train -- m
noncomputable def relative_speed := total_distance / time_to_cross -- m/s

theorem speed_of_second_train (V : ℝ) :
  V * 1000 / 3600 + 60 * 1000 / 3600 = 38.75 →
  V = 79.5 := by {
  sorry
}

end speed_of_second_train_l346_34602


namespace min_value_quadratic_l346_34639

theorem min_value_quadratic (x : ℝ) : 
  ∃ m, m = 3 * x^2 - 18 * x + 2048 ∧ ∀ x, 3 * x^2 - 18 * x + 2048 ≥ 2021 :=
by sorry

end min_value_quadratic_l346_34639


namespace B_finishes_job_in_37_5_days_l346_34692

variable (eff_A eff_B eff_C : ℝ)
variable (effA_eq_half_effB : eff_A = (1 / 2) * eff_B)
variable (effB_eq_two_thirds_effC : eff_B = (2 / 3) * eff_C)
variable (job_in_15_days : 15 * (eff_A + eff_B + eff_C) = 1)

theorem B_finishes_job_in_37_5_days :
  (1 / eff_B) = 37.5 :=
by
  sorry

end B_finishes_job_in_37_5_days_l346_34692


namespace max_x_inequality_k_l346_34665

theorem max_x_inequality_k (k : ℝ) (h : ∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) : k = 8 :=
sorry

end max_x_inequality_k_l346_34665


namespace Ivan_cannot_cut_off_all_heads_l346_34682

-- Defining the number of initial heads
def initial_heads : ℤ := 100

-- Effect of the first sword: Removes 21 heads
def first_sword_effect : ℤ := 21

-- Effect of the second sword: Removes 4 heads and adds 2006 heads
def second_sword_effect : ℤ := 2006 - 4

-- Proving Ivan cannot reduce the number of heads to zero
theorem Ivan_cannot_cut_off_all_heads :
  (∀ n : ℤ, n % 7 = initial_heads % 7 → n ≠ 0) :=
by
  sorry

end Ivan_cannot_cut_off_all_heads_l346_34682


namespace math_problem_l346_34670

theorem math_problem (a b : ℕ) (ha : a = 45) (hb : b = 15) :
  (a + b)^2 - 3 * (a^2 + b^2 - 2 * a * b) = 900 :=
by
  sorry

end math_problem_l346_34670


namespace white_pairs_coincide_l346_34617

theorem white_pairs_coincide
  (red_triangles_half : ℕ)
  (blue_triangles_half : ℕ)
  (white_triangles_half : ℕ)
  (red_pairs : ℕ)
  (blue_pairs : ℕ)
  (red_white_pairs : ℕ)
  (red_triangles_total_half : red_triangles_half = 4)
  (blue_triangles_total_half : blue_triangles_half = 6)
  (white_triangles_total_half : white_triangles_half = 10)
  (red_pairs_total : red_pairs = 3)
  (blue_pairs_total : blue_pairs = 4)
  (red_white_pairs_total : red_white_pairs = 3) :
  ∃ w : ℕ, w = 5 :=
by
  sorry

end white_pairs_coincide_l346_34617


namespace no_solution_inequality_l346_34600

theorem no_solution_inequality (m : ℝ) : (¬ ∃ x : ℝ, |x + 1| + |x - 5| ≤ m) ↔ m < 6 :=
sorry

end no_solution_inequality_l346_34600


namespace minimum_loadings_to_prove_first_ingot_weighs_1kg_l346_34648

theorem minimum_loadings_to_prove_first_ingot_weighs_1kg :
  ∀ (w : Fin 11 → ℕ), 
    (∀ i, w i = i + 1) →
    (∃ s₁ s₂ : Finset (Fin 11), 
       s₁.card ≤ 6 ∧ s₂.card ≤ 6 ∧ 
       s₁.sum w = 11 ∧ s₂.sum w = 11 ∧ 
       (∀ s : Finset (Fin 11), s.sum w = 11 → s ≠ s₁ ∧ s ≠ s₂) ∧
       (w 0 = 1)) := sorry -- Fill in the proof here

end minimum_loadings_to_prove_first_ingot_weighs_1kg_l346_34648


namespace negation_equivalence_l346_34675

variables (x : ℝ)

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), ↑q = x

def has_rational_square (x : ℝ) : Prop := ∃ (q : ℚ), ↑q * ↑q = x * x

def proposition := ∃ (x : ℝ), is_irrational x ∧ has_rational_square x

theorem negation_equivalence :
  (¬ proposition) ↔ ∀ (x : ℝ), is_irrational x → ¬ has_rational_square x :=
by sorry

end negation_equivalence_l346_34675


namespace aunt_wang_bought_n_lilies_l346_34603

theorem aunt_wang_bought_n_lilies 
  (cost_rose : ℕ) 
  (cost_lily : ℕ) 
  (total_spent : ℕ) 
  (num_roses : ℕ) 
  (num_lilies : ℕ) 
  (roses_cost : num_roses * cost_rose = 10) 
  (total_spent_cond : total_spent = 55) 
  (cost_conditions : cost_rose = 5 ∧ cost_lily = 9) 
  (spending_eq : total_spent = num_roses * cost_rose + num_lilies * cost_lily) : 
  num_lilies = 5 :=
by 
  sorry

end aunt_wang_bought_n_lilies_l346_34603


namespace find_N_product_l346_34681

variables (M L : ℤ) (N : ℤ)

theorem find_N_product
  (h1 : M = L + N)
  (h2 : M + 3 = (L + N + 3))
  (h3 : L - 5 = L - 5)
  (h4 : |(L + N + 3) - (L - 5)| = 4) :
  N = -4 ∨ N = -12 → (-4 * -12) = 48 :=
by sorry

end find_N_product_l346_34681


namespace ninth_day_skate_time_l346_34688

-- Define the conditions
def first_4_days_skate_time : ℕ := 4 * 70
def second_4_days_skate_time : ℕ := 4 * 100
def total_days : ℕ := 9
def average_minutes_per_day : ℕ := 100

-- Define the theorem stating that Gage must skate 220 minutes on the ninth day to meet the average
theorem ninth_day_skate_time : 
  let total_minutes_needed := total_days * average_minutes_per_day
  let current_skate_time := first_4_days_skate_time + second_4_days_skate_time
  total_minutes_needed - current_skate_time = 220 := 
by
  -- Placeholder for the proof
  sorry

end ninth_day_skate_time_l346_34688


namespace least_number_divisible_by_23_l346_34614

theorem least_number_divisible_by_23 (n d : ℕ) (h_n : n = 1053) (h_d : d = 23) : ∃ x : ℕ, (n + x) % d = 0 ∧ x = 5 := by
  sorry

end least_number_divisible_by_23_l346_34614


namespace smallest_a_for_polynomial_roots_l346_34625

theorem smallest_a_for_polynomial_roots :
  ∃ (a b c : ℕ), 
         (∃ (r s t u : ℕ), r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧ r * s * t * u = 5160 ∧ a = r + s + t + u) 
    ∧  (∀ (r' s' t' u' : ℕ), r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧ r' * s' * t' * u' = 5160 ∧ r' + s' + t' + u' < a → false) 
    := sorry

end smallest_a_for_polynomial_roots_l346_34625


namespace plant_height_increase_l346_34643

theorem plant_height_increase (total_increase : ℕ) (century_in_years : ℕ) (decade_in_years : ℕ) (years_in_2_centuries : ℕ) (num_decades : ℕ) : 
  total_increase = 1800 →
  century_in_years = 100 →
  decade_in_years = 10 →
  years_in_2_centuries = 2 * century_in_years →
  num_decades = years_in_2_centuries / decade_in_years →
  total_increase / num_decades = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end plant_height_increase_l346_34643


namespace distinct_real_roots_l346_34615

def operation (a b : ℝ) : ℝ := a^2 - a * b + b

theorem distinct_real_roots {x : ℝ} : 
  (operation x 3 = 5) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation x1 3 = 5 ∧ operation x2 3 = 5) :=
by 
  -- Add your proof here
  sorry

end distinct_real_roots_l346_34615


namespace solve_problem_l346_34637

def is_solution (a : ℕ) : Prop :=
  a % 3 = 1 ∧ ∃ k : ℕ, a = 5 * k

theorem solve_problem : ∃ a : ℕ, is_solution a ∧ ∀ b : ℕ, is_solution b → a ≤ b := 
  sorry

end solve_problem_l346_34637


namespace complement_M_in_U_l346_34696

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem to prove that the complement of M in U is (1, +∞)
theorem complement_M_in_U :
  (U \ M) = {x | 1 < x} :=
by
  sorry

end complement_M_in_U_l346_34696


namespace quadratic_eq_c_has_equal_roots_l346_34650

theorem quadratic_eq_c_has_equal_roots (c : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + c = 0 ∧
                      ∀ y : ℝ, x^2 - 4 * x + c = 0 → y = x) : c = 4 := sorry

end quadratic_eq_c_has_equal_roots_l346_34650


namespace rational_root_even_denominator_l346_34655

theorem rational_root_even_denominator
  (a b c : ℤ)
  (sum_ab_even : (a + b) % 2 = 0)
  (c_odd : c % 2 = 1) :
  ∀ (p q : ℤ), (q ≠ 0) → (IsRationalRoot : a * (p * p) + b * p * q + c * (q * q) = 0) →
    gcd p q = 1 → q % 2 = 0 :=
by
  sorry

end rational_root_even_denominator_l346_34655


namespace solve_for_x_l346_34683

theorem solve_for_x (x : ℝ) (h : 3 / (x + 2) = 2 / (x - 1)) : x = 7 :=
sorry

end solve_for_x_l346_34683


namespace handshake_count_l346_34667

theorem handshake_count (n_twins: ℕ) (n_triplets: ℕ)
  (twin_pairs: ℕ) (triplet_groups: ℕ)
  (handshakes_twin : ∀ (x: ℕ), x = (n_twins - 2))
  (handshakes_triplet : ∀ (y: ℕ), y = (n_triplets - 3))
  (handshakes_cross_twins : ∀ (z: ℕ), z = 3*n_triplets / 4)
  (handshakes_cross_triplets : ∀ (w: ℕ), w = n_twins / 4) :
  2 * (n_twins * (n_twins -1 -1) / 2 + n_triplets * (n_triplets - 1 - 1) / 2 + n_twins * (3*n_triplets / 4) + n_triplets * (n_twins / 4)) / 2 = 804 := 
sorry

end handshake_count_l346_34667


namespace power_function_decreasing_l346_34687

theorem power_function_decreasing (m : ℝ) (x : ℝ) (hx : x > 0) :
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by
  sorry

end power_function_decreasing_l346_34687


namespace CoveredAreaIs84_l346_34623

def AreaOfStrip (length width : ℕ) : ℕ :=
  length * width

def TotalAreaWithoutOverlaps (numStrips areaOfOneStrip : ℕ) : ℕ :=
  numStrips * areaOfOneStrip

def OverlapArea (intersectionArea : ℕ) (numIntersections : ℕ) : ℕ :=
  intersectionArea * numIntersections

def ActualCoveredArea (totalArea overlapArea : ℕ) : ℕ :=
  totalArea - overlapArea

theorem CoveredAreaIs84 :
  let length := 12
  let width := 2
  let numStrips := 6
  let intersectionArea := width * width
  let numIntersections := 15
  let areaOfOneStrip := AreaOfStrip length width
  let totalAreaWithoutOverlaps := TotalAreaWithoutOverlaps numStrips areaOfOneStrip
  let totalOverlapArea := OverlapArea intersectionArea numIntersections
  ActualCoveredArea totalAreaWithoutOverlaps totalOverlapArea = 84 :=
by
  sorry

end CoveredAreaIs84_l346_34623


namespace min_value_of_z_l346_34663

noncomputable def min_z (x y : ℝ) : ℝ :=
  2 * x + (Real.sqrt 3) * y

theorem min_value_of_z :
  ∃ x y : ℝ, 3 * x^2 + 4 * y^2 = 12 ∧ min_z x y = -5 :=
sorry

end min_value_of_z_l346_34663


namespace quadratic_non_real_roots_iff_l346_34658

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end quadratic_non_real_roots_iff_l346_34658


namespace half_angle_quadrant_second_quadrant_l346_34693

theorem half_angle_quadrant_second_quadrant
  (θ : Real)
  (h1 : π < θ ∧ θ < 3 * π / 2) -- θ is in the third quadrant
  (h2 : Real.cos (θ / 2) < 0) : -- cos (θ / 2) < 0
  π / 2 < θ / 2 ∧ θ / 2 < π := -- θ / 2 is in the second quadrant
sorry

end half_angle_quadrant_second_quadrant_l346_34693


namespace ratio_of_auto_finance_companies_credit_l346_34668

theorem ratio_of_auto_finance_companies_credit
    (total_consumer_credit : ℝ)
    (percent_auto_installment_credit : ℝ)
    (credit_by_auto_finance_companies : ℝ)
    (total_auto_credit : ℝ)
    (hc1 : total_consumer_credit = 855)
    (hc2 : percent_auto_installment_credit = 0.20)
    (hc3 : credit_by_auto_finance_companies = 57)
    (htotal_auto_credit : total_auto_credit = percent_auto_installment_credit * total_consumer_credit) :
    (credit_by_auto_finance_companies / total_auto_credit) = (1 / 3) := 
by
  sorry

end ratio_of_auto_finance_companies_credit_l346_34668


namespace sum_of_first_ten_terms_l346_34604

variable {α : Type*} [LinearOrderedField α]

-- Defining the arithmetic sequence and sum of the first n terms
def a_n (a d : α) (n : ℕ) : α := a + d * (n - 1)

def S_n (a : α) (d : α) (n : ℕ) : α := n / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_ten_terms (a d : α) (h : a_n a d 3 + a_n a d 8 = 12) : S_n a d 10 = 60 :=
by sorry

end sum_of_first_ten_terms_l346_34604


namespace students_taller_than_Yoongi_l346_34695

theorem students_taller_than_Yoongi {n total shorter : ℕ} (h1 : total = 20) (h2 : shorter = 11) : n = 8 :=
by
  sorry

end students_taller_than_Yoongi_l346_34695


namespace derivative_is_even_then_b_eq_zero_l346_34612

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- The statement that the derivative is an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Our main theorem
theorem derivative_is_even_then_b_eq_zero : is_even (f' a b c) → b = 0 :=
by
  intro h
  have h1 := h 1
  have h2 := h (-1)
  sorry

end derivative_is_even_then_b_eq_zero_l346_34612


namespace solve_for_x_l346_34607

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end solve_for_x_l346_34607


namespace vector_intersecting_line_parameter_l346_34647

theorem vector_intersecting_line_parameter :
  ∃ (a b s : ℝ), a = 3 * s + 5 ∧ b = 2 * s + 4 ∧
                   (∃ r, (a, b) = (3 * r, 2 * r)) ∧
                   (a, b) = (6, 14 / 3) :=
by
  sorry

end vector_intersecting_line_parameter_l346_34647


namespace total_money_spent_l346_34669

def cost_life_journey_cd : ℕ := 100
def cost_day_life_cd : ℕ := 50
def cost_when_rescind_cd : ℕ := 85
def number_of_cds_each : ℕ := 3

theorem total_money_spent :
  number_of_cds_each * cost_life_journey_cd +
  number_of_cds_each * cost_day_life_cd +
  number_of_cds_each * cost_when_rescind_cd = 705 :=
sorry

end total_money_spent_l346_34669


namespace trig_expression_value_quadratic_roots_l346_34677

theorem trig_expression_value :
  (Real.tan (Real.pi / 6))^2 + 2 * Real.sin (Real.pi / 4) - 2 * Real.cos (Real.pi / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

theorem quadratic_roots :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = (-2 + Real.sqrt 2) / 2 ∨ x = (-2 - Real.sqrt 2) / 2) := by
  sorry

end trig_expression_value_quadratic_roots_l346_34677


namespace blueberries_in_blue_box_l346_34609

theorem blueberries_in_blue_box (B S : ℕ) (h1: S - B = 10) (h2 : 50 = S) : B = 40 := 
by
  sorry

end blueberries_in_blue_box_l346_34609


namespace number_of_primes_in_interval_35_to_44_l346_34698

/--
The number of prime numbers in the interval [35, 44] is 3.
-/
theorem number_of_primes_in_interval_35_to_44 : 
  (Finset.filter Nat.Prime (Finset.Icc 35 44)).card = 3 := 
by
  sorry

end number_of_primes_in_interval_35_to_44_l346_34698


namespace MinkyungHeight_is_correct_l346_34645

noncomputable def HaeunHeight : ℝ := 1.56
noncomputable def NayeonHeight : ℝ := HaeunHeight - 0.14
noncomputable def MinkyungHeight : ℝ := NayeonHeight + 0.27

theorem MinkyungHeight_is_correct : MinkyungHeight = 1.69 :=
by
  sorry

end MinkyungHeight_is_correct_l346_34645


namespace carly_dog_count_l346_34672

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l346_34672


namespace divisor_is_31_l346_34691

-- Definition of the conditions.
def condition1 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 62 * k + 7

def condition2 (x y : ℤ) : Prop :=
  ∃ m : ℤ, x + 11 = y * m + 18

-- Main statement asserting the divisor y.
theorem divisor_is_31 (x y : ℤ) (h₁ : condition1 x) (h₂ : condition2 x y) : y = 31 :=
sorry

end divisor_is_31_l346_34691


namespace range_of_a_l346_34606

theorem range_of_a (a : ℝ) :
    (∀ x : ℤ, x + 1 > 0 → 3 * x - a ≤ 0 → x = 0 ∨ x = 1 ∨ x = 2) ↔ 6 ≤ a ∧ a < 9 :=
by
  sorry

end range_of_a_l346_34606


namespace attraction_ticket_cost_l346_34657

theorem attraction_ticket_cost
  (cost_park_entry : ℕ)
  (cost_attraction_parent : ℕ)
  (total_paid : ℕ)
  (num_children : ℕ)
  (num_parents : ℕ)
  (num_grandmother : ℕ)
  (x : ℕ)
  (h_costs : cost_park_entry = 5)
  (h_attraction_parent : cost_attraction_parent = 4)
  (h_family : num_children = 4 ∧ num_parents = 2 ∧ num_grandmother = 1)
  (h_total_paid : total_paid = 55)
  (h_equation : (num_children + num_parents + num_grandmother) * cost_park_entry + (num_parents + num_grandmother) * cost_attraction_parent + num_children * x = total_paid) :
  x = 2 := by
  sorry

end attraction_ticket_cost_l346_34657


namespace valid_five_digit_integers_l346_34642

/-- How many five-digit positive integers can be formed by arranging the digits 1, 1, 2, 3, 4 so 
that the two 1s are not next to each other -/
def num_valid_arrangements : ℕ :=
  36

theorem valid_five_digit_integers :
  ∃ n : ℕ, n = num_valid_arrangements :=
by
  use 36
  sorry

end valid_five_digit_integers_l346_34642


namespace inequality_solution_range_l346_34659

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x+2| + |x-3| < a) ↔ a > 5 :=
by
  sorry

end inequality_solution_range_l346_34659


namespace find_a_for_tangency_l346_34653

-- Definitions of line and parabola
def line (x y : ℝ) : Prop := x - y - 1 = 0
def parabola (x y : ℝ) (a : ℝ) : Prop := y = a * x^2

-- The tangency condition for quadratic equations
def tangency_condition (a : ℝ) : Prop := 1 - 4 * a = 0

theorem find_a_for_tangency (a : ℝ) :
  (∀ x y, line x y → parabola x y a → tangency_condition a) → a = 1/4 :=
by
  -- Proof omitted
  sorry

end find_a_for_tangency_l346_34653


namespace fourth_term_of_sequence_l346_34634

-- Given conditions
def first_term : ℕ := 5
def fifth_term : ℕ := 1280

-- Definition of the common ratio
def common_ratio (a : ℕ) (b : ℕ) : ℕ := (b / a)^(1 / 4)

-- Function to calculate the nth term of a geometric sequence
def nth_term (a r n : ℕ) : ℕ := a * r^(n - 1)

-- Prove the fourth term of the geometric sequence is 320
theorem fourth_term_of_sequence 
    (a : ℕ) (b : ℕ) (a_pos : a = first_term) (b_eq : nth_term a (common_ratio a b) 5 = b) : 
    nth_term a (common_ratio a b) 4 = 320 := by
  sorry

end fourth_term_of_sequence_l346_34634


namespace parabola_constant_term_l346_34605

theorem parabola_constant_term (b c : ℝ)
  (h1 : 2 * b + c = 8)
  (h2 : -2 * b + c = -4)
  (h3 : 4 * b + c = 24) :
  c = 2 :=
sorry

end parabola_constant_term_l346_34605


namespace math_problem_correct_l346_34618

noncomputable def math_problem : Prop :=
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40)

theorem math_problem_correct : math_problem := by
  sorry

end math_problem_correct_l346_34618


namespace strap_mask_probability_l346_34601

theorem strap_mask_probability 
  (p_regular_medical : ℝ)
  (p_surgical : ℝ)
  (p_strap_regular : ℝ)
  (p_strap_surgical : ℝ)
  (h_regular_medical : p_regular_medical = 0.8)
  (h_surgical : p_surgical = 0.2)
  (h_strap_regular : p_strap_regular = 0.1)
  (h_strap_surgical : p_strap_surgical = 0.2) :
  (p_regular_medical * p_strap_regular + p_surgical * p_strap_surgical) = 0.12 :=
by
  rw [h_regular_medical, h_surgical, h_strap_regular, h_strap_surgical]
  -- proof will go here
  sorry

end strap_mask_probability_l346_34601


namespace minimize_costs_l346_34619

def total_books : ℕ := 150000
def handling_fee_per_order : ℕ := 30
def storage_fee_per_1000_copies : ℕ := 40
def evenly_distributed_books : Prop := true --Assuming books are evenly distributed by default

noncomputable def optimal_order_frequency : ℕ := 10
noncomputable def optimal_batch_size : ℕ := 15000

theorem minimize_costs 
  (handling_fee_per_order : ℕ) 
  (storage_fee_per_1000_copies : ℕ) 
  (total_books : ℕ) 
  (evenly_distributed_books : Prop)
  : optimal_order_frequency = 10 ∧ optimal_batch_size = 15000 := sorry

end minimize_costs_l346_34619


namespace fraction_zero_x_eq_2_l346_34641

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end fraction_zero_x_eq_2_l346_34641


namespace compare_powers_l346_34694

theorem compare_powers :
  let a := 5 ^ 140
  let b := 3 ^ 210
  let c := 2 ^ 280
  c < a ∧ a < b := by
  -- Proof omitted
  sorry

end compare_powers_l346_34694


namespace number_of_paths_in_MATHEMATICIAN_diagram_l346_34616

theorem number_of_paths_in_MATHEMATICIAN_diagram : ∃ n : ℕ, n = 8191 :=
by
  -- Define necessary structure
  -- Number of rows and binary choices
  let rows : ℕ := 12
  let choices_per_position : ℕ := 2
  -- Total paths calculation
  let total_paths := choices_per_position ^ rows
  -- Including symmetry and subtracting duplicate
  let final_paths := 2 * total_paths - 1
  use final_paths
  have : final_paths = 8191 :=
    by norm_num
  exact this

end number_of_paths_in_MATHEMATICIAN_diagram_l346_34616


namespace quadratic_roots_real_distinct_l346_34621

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end quadratic_roots_real_distinct_l346_34621


namespace gcd_gx_x_l346_34635

-- Condition: x is a multiple of 7263
def isMultipleOf7263 (x : ℕ) : Prop := ∃ k : ℕ, x = 7263 * k

-- Definition of g(x)
def g (x : ℕ) : ℕ := (3*x + 4) * (9*x + 5) * (17*x + 11) * (x + 17)

-- Statement to be proven
theorem gcd_gx_x (x : ℕ) (h : isMultipleOf7263 x) : Nat.gcd (g x) x = 1 := by
  sorry

end gcd_gx_x_l346_34635


namespace pigeon_count_correct_l346_34662

def initial_pigeon_count : ℕ := 1
def new_pigeon_count : ℕ := 1
def total_pigeon_count : ℕ := 2

theorem pigeon_count_correct : initial_pigeon_count + new_pigeon_count = total_pigeon_count :=
by
  sorry

end pigeon_count_correct_l346_34662


namespace inequality_proof_l346_34680

theorem inequality_proof (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0)
    (h_cond : 2 * (a + b + c + d) ≥ a * b * c * d) : (a^2 + b^2 + c^2 + d^2) ≥ (a * b * c * d) :=
by
  sorry

end inequality_proof_l346_34680


namespace cos_D_zero_l346_34661

noncomputable def area_of_triangle (a b: ℝ) (sinD: ℝ) : ℝ := 1 / 2 * a * b * sinD

theorem cos_D_zero (DE DF : ℝ) (D : ℝ) (h1 : area_of_triangle DE DF (Real.sin D) = 98) (h2 : Real.sqrt (DE * DF) = 14) : Real.cos D = 0 :=
  by
  sorry

end cos_D_zero_l346_34661


namespace count_reflectional_symmetry_l346_34624

def tetrominoes : List String := ["I", "O", "T", "S", "Z", "L", "J"]

def has_reflectional_symmetry (tetromino : String) : Bool :=
  match tetromino with
  | "I" => true
  | "O" => true
  | "T" => true
  | "S" => false
  | "Z" => false
  | "L" => false
  | "J" => false
  | _   => false

theorem count_reflectional_symmetry : 
  (tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end count_reflectional_symmetry_l346_34624


namespace expression_for_A_div_B_l346_34631

theorem expression_for_A_div_B (x A B : ℝ)
  (h1 : x^3 + 1/x^3 = A)
  (h2 : x - 1/x = B) :
  A / B = B^2 + 3 := 
sorry

end expression_for_A_div_B_l346_34631


namespace part1_part2_l346_34676

noncomputable def f (x k : ℝ) : ℝ := (x ^ 2 + k * x + 1) / (x ^ 2 + 1)

theorem part1 (k : ℝ) (h : k = -4) : ∃ x > 0, f x k = -1 :=
  by sorry -- Proof goes here

theorem part2 (k : ℝ) : (∀ (x1 x2 x3 : ℝ), (0 < x1) → (0 < x2) → (0 < x3) → 
  ∃ a b c, a = f x1 k ∧ b = f x2 k ∧ c = f x3 k ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) ↔ (-1 ≤ k ∧ k ≤ 2) :=
  by sorry -- Proof goes here

end part1_part2_l346_34676
