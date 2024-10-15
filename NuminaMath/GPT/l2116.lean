import Mathlib

namespace NUMINAMATH_GPT_solving_inequality_l2116_211621

theorem solving_inequality (x : ℝ) : 
  (x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1)) ↔ ((x^2 - 4) / (x^2 - 1) > 0) :=
by 
  sorry

end NUMINAMATH_GPT_solving_inequality_l2116_211621


namespace NUMINAMATH_GPT_prime_range_for_integer_roots_l2116_211680

theorem prime_range_for_integer_roots (p : ℕ) (h_prime : Prime p) 
  (h_int_roots : ∃ (a b : ℤ), a + b = -p ∧ a * b = -300 * p) : 
  1 < p ∧ p ≤ 11 :=
sorry

end NUMINAMATH_GPT_prime_range_for_integer_roots_l2116_211680


namespace NUMINAMATH_GPT_trapezoid_other_base_possible_lengths_l2116_211690

-- Definition of the trapezoid problem.
structure Trapezoid where
  height : ℕ
  leg1 : ℕ
  leg2 : ℕ
  base1 : ℕ

-- The given conditions
def trapezoid_data : Trapezoid :=
{ height := 12, leg1 := 20, leg2 := 15, base1 := 42 }

-- The proof problem in Lean 4 statement
theorem trapezoid_other_base_possible_lengths (t : Trapezoid) :
  t = trapezoid_data → (∃ b : ℕ, (b = 17 ∨ b = 35)) :=
by
  intro h_data_eq
  sorry

end NUMINAMATH_GPT_trapezoid_other_base_possible_lengths_l2116_211690


namespace NUMINAMATH_GPT_B_can_win_with_initial_config_B_l2116_211617

def initial_configuration_B := (6, 2, 1)

def A_starts_and_B_wins (config : (Nat × Nat × Nat)) : Prop := sorry

theorem B_can_win_with_initial_config_B : A_starts_and_B_wins initial_configuration_B :=
sorry

end NUMINAMATH_GPT_B_can_win_with_initial_config_B_l2116_211617


namespace NUMINAMATH_GPT_estate_problem_l2116_211689

def totalEstateValue (E a b : ℝ) : Prop :=
  (a + b = (3/5) * E) ∧ 
  (a = 2 * b) ∧ 
  (3 * b = (3/5) * E) ∧ 
  (E = a + b + (3 * b) + 4000)

theorem estate_problem (E : ℝ) (a b : ℝ) :
  totalEstateValue E a b → E = 20000 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_estate_problem_l2116_211689


namespace NUMINAMATH_GPT_not_q_true_l2116_211669

theorem not_q_true (p q : Prop) (hp : p = true) (hq : q = false) : ¬q = true :=
by
  sorry

end NUMINAMATH_GPT_not_q_true_l2116_211669


namespace NUMINAMATH_GPT_elias_purchased_50cent_items_l2116_211604

theorem elias_purchased_50cent_items :
  ∃ (a b c : ℕ), a + b + c = 50 ∧ (50 * a + 250 * b + 400 * c = 5000) ∧ (a = 40) :=
by {
  sorry
}

end NUMINAMATH_GPT_elias_purchased_50cent_items_l2116_211604


namespace NUMINAMATH_GPT_system_of_equations_correct_l2116_211652

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end NUMINAMATH_GPT_system_of_equations_correct_l2116_211652


namespace NUMINAMATH_GPT_expected_value_of_smallest_seven_selected_from_sixty_three_l2116_211686

noncomputable def expected_value_smallest_selected (n r : ℕ) : ℕ :=
  (n + 1) / (r + 1)

theorem expected_value_of_smallest_seven_selected_from_sixty_three :
  expected_value_smallest_selected 63 7 = 8 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_expected_value_of_smallest_seven_selected_from_sixty_three_l2116_211686


namespace NUMINAMATH_GPT_div_iff_div_l2116_211625

theorem div_iff_div {a b : ℤ} : (29 ∣ (3 * a + 2 * b)) ↔ (29 ∣ (11 * a + 17 * b)) := 
by sorry

end NUMINAMATH_GPT_div_iff_div_l2116_211625


namespace NUMINAMATH_GPT_prism_volume_l2116_211632

theorem prism_volume
  (l w h : ℝ)
  (h1 : l * w = 6.5)
  (h2 : w * h = 8)
  (h3 : l * h = 13) :
  l * w * h = 26 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l2116_211632


namespace NUMINAMATH_GPT_sqrt_floor_square_l2116_211654

theorem sqrt_floor_square (h1 : 7 < Real.sqrt 50) (h2 : Real.sqrt 50 < 8) :
  Int.floor (Real.sqrt 50) ^ 2 = 49 := by
  sorry

end NUMINAMATH_GPT_sqrt_floor_square_l2116_211654


namespace NUMINAMATH_GPT_minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l2116_211682

noncomputable def f (a b x : ℝ) := Real.exp x - a * x - b

theorem minimum_value_f_b_eq_neg_a (a : ℝ) (h : 0 < a) :
  ∃ m, m = 2 * a - a * Real.log a ∧ ∀ x : ℝ, f a (-a) x ≥ m :=
sorry

theorem maximum_value_ab (a b : ℝ) (h : ∀ x : ℝ, f a b x + a ≥ 0) :
  ab ≤ (1 / 2) * Real.exp 3 :=
sorry

theorem inequality_for_f_and_f' (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : b = -a) (h3 : f a b x1 = 0) (h4 : f a b x2 = 0) (h5 : x1 < x2)
  : f a (-a) (3 * Real.log a) > (Real.exp ((2 * x1 * x2) / (x1 + x2)) - a) :=
sorry

end NUMINAMATH_GPT_minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l2116_211682


namespace NUMINAMATH_GPT_find_N_l2116_211685

theorem find_N (N : ℤ) :
  (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l2116_211685


namespace NUMINAMATH_GPT_complete_the_square_correct_l2116_211620

noncomputable def complete_the_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 1 = 0 ↔ (x - 1)^2 = 2

theorem complete_the_square_correct : ∀ x : ℝ, complete_the_square x := by
  sorry

end NUMINAMATH_GPT_complete_the_square_correct_l2116_211620


namespace NUMINAMATH_GPT_ratio_lcm_gcf_256_162_l2116_211648

theorem ratio_lcm_gcf_256_162 : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_lcm_gcf_256_162_l2116_211648


namespace NUMINAMATH_GPT_monthly_installments_l2116_211600

theorem monthly_installments (cash_price deposit installment saving : ℕ) (total_paid installments_made : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment = 300 →
  saving = 4000 →
  total_paid = cash_price + saving →
  installments_made = (total_paid - deposit) / installment →
  installments_made = 30 :=
by
  intros h_cash_price h_deposit h_installment h_saving h_total_paid h_installments_made
  sorry

end NUMINAMATH_GPT_monthly_installments_l2116_211600


namespace NUMINAMATH_GPT_exists_coprime_positive_sum_le_m_l2116_211650

theorem exists_coprime_positive_sum_le_m (m : ℕ) (a b : ℤ) 
  (ha : 0 < a) (hb : 0 < b) (hcoprime : Int.gcd a b = 1)
  (h1 : a ∣ (m + b^2)) (h2 : b ∣ (m + a^2)) 
  : ∃ a' b', 0 < a' ∧ 0 < b' ∧ Int.gcd a' b' = 1 ∧ a' ∣ (m + b'^2) ∧ b' ∣ (m + a'^2) ∧ a' + b' ≤ m + 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_coprime_positive_sum_le_m_l2116_211650


namespace NUMINAMATH_GPT_max_gcd_of_13n_plus_3_and_7n_plus_1_l2116_211628

theorem max_gcd_of_13n_plus_3_and_7n_plus_1 (n : ℕ) (hn : 0 < n) :
  ∃ d, d = Nat.gcd (13 * n + 3) (7 * n + 1) ∧ ∀ m, m = Nat.gcd (13 * n + 3) (7 * n + 1) → m ≤ 8 := 
sorry

end NUMINAMATH_GPT_max_gcd_of_13n_plus_3_and_7n_plus_1_l2116_211628


namespace NUMINAMATH_GPT_population_difference_is_16_l2116_211660

def total_birds : ℕ := 250

def pigeons_percent : ℕ := 30
def sparrows_percent : ℕ := 25
def crows_percent : ℕ := 20
def swans_percent : ℕ := 15
def parrots_percent : ℕ := 10

def black_pigeons_percent : ℕ := 60
def white_pigeons_percent : ℕ := 40
def black_male_pigeons_percent : ℕ := 20
def white_female_pigeons_percent : ℕ := 50

def female_sparrows_percent : ℕ := 60
def male_sparrows_percent : ℕ := 40

def female_crows_percent : ℕ := 30
def male_crows_percent : ℕ := 70

def male_parrots_percent : ℕ := 65
def female_parrots_percent : ℕ := 35

noncomputable
def black_male_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (black_pigeons_percent * (black_male_pigeons_percent / 100)) / 100
noncomputable
def white_female_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (white_pigeons_percent * (white_female_pigeons_percent / 100)) / 100
noncomputable
def male_sparrows : ℕ := (sparrows_percent * total_birds / 100) * (male_sparrows_percent / 100)
noncomputable
def female_crows : ℕ := (crows_percent * total_birds / 100) * (female_crows_percent / 100)
noncomputable
def male_parrots : ℕ := (parrots_percent * total_birds / 100) * (male_parrots_percent / 100)

noncomputable
def max_population : ℕ := max (max (max (max black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots
noncomputable
def min_population : ℕ := min (min (min (min black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots

noncomputable
def population_difference : ℕ := max_population - min_population

theorem population_difference_is_16 : population_difference = 16 :=
sorry

end NUMINAMATH_GPT_population_difference_is_16_l2116_211660


namespace NUMINAMATH_GPT_not_equal_d_l2116_211697

def frac_14_over_6 : ℚ := 14 / 6
def mixed_2_and_1_3rd : ℚ := 2 + 1 / 3
def mixed_neg_2_and_1_3rd : ℚ := -(2 + 1 / 3)
def mixed_3_and_1_9th : ℚ := 3 + 1 / 9
def mixed_2_and_4_12ths : ℚ := 2 + 4 / 12
def target_fraction : ℚ := 7 / 3

theorem not_equal_d : mixed_3_and_1_9th ≠ target_fraction :=
by sorry

end NUMINAMATH_GPT_not_equal_d_l2116_211697


namespace NUMINAMATH_GPT_question_true_l2116_211627
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_question_true_l2116_211627


namespace NUMINAMATH_GPT_side_length_of_square_l2116_211656

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l2116_211656


namespace NUMINAMATH_GPT_clubs_popularity_order_l2116_211653

theorem clubs_popularity_order (chess drama art science : ℚ)
  (h_chess: chess = 14/35) (h_drama: drama = 9/28) (h_art: art = 11/21) (h_science: science = 8/15) :
  science > art ∧ art > chess ∧ chess > drama :=
by {
  -- Place proof steps here (optional)
  sorry
}

end NUMINAMATH_GPT_clubs_popularity_order_l2116_211653


namespace NUMINAMATH_GPT_simplify_expression_l2116_211645

theorem simplify_expression (x y : ℝ) :
  3 * (x + y) ^ 2 - 7 * (x + y) + 8 * (x + y) ^ 2 + 6 * (x + y) = 
  11 * (x + y) ^ 2 - (x + y) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2116_211645


namespace NUMINAMATH_GPT_estimate_total_number_of_fish_l2116_211639

-- Define the conditions
variables (totalMarked : ℕ) (secondSample : ℕ) (markedInSecondSample : ℕ) (N : ℕ)

-- Assume the conditions
axiom condition1 : totalMarked = 60
axiom condition2 : secondSample = 80
axiom condition3 : markedInSecondSample = 5

-- Lean theorem statement proving N = 960 given the conditions
theorem estimate_total_number_of_fish (totalMarked secondSample markedInSecondSample N : ℕ)
  (h1 : totalMarked = 60)
  (h2 : secondSample = 80)
  (h3 : markedInSecondSample = 5) :
  N = 960 :=
sorry

end NUMINAMATH_GPT_estimate_total_number_of_fish_l2116_211639


namespace NUMINAMATH_GPT_directrix_of_parabola_l2116_211687

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = - (1 / 8) * x^2 → y = 2 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2116_211687


namespace NUMINAMATH_GPT_matrix_B3_is_zero_unique_l2116_211612

theorem matrix_B3_is_zero_unique (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) :
  ∃! (B3 : Matrix (Fin 2) (Fin 2) ℝ), B3 = B^3 ∧ B3 = 0 := sorry

end NUMINAMATH_GPT_matrix_B3_is_zero_unique_l2116_211612


namespace NUMINAMATH_GPT_xiao_li_estimate_l2116_211644

variable (x y z : ℝ)

theorem xiao_li_estimate (h1 : x > y) (h2 : y > 0) (h3 : 0 < z):
    (x + z) + (y - z) = x + y := 
by 
sorry

end NUMINAMATH_GPT_xiao_li_estimate_l2116_211644


namespace NUMINAMATH_GPT_age_condition_l2116_211611

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end NUMINAMATH_GPT_age_condition_l2116_211611


namespace NUMINAMATH_GPT_factor_x_squared_minus_169_l2116_211677

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := 
by
  -- Recognize that 169 is a perfect square
  have h : 169 = 13^2 := by norm_num
  -- Use the difference of squares formula
  -- Sorry is used to skip the proof part
  sorry

end NUMINAMATH_GPT_factor_x_squared_minus_169_l2116_211677


namespace NUMINAMATH_GPT_coins_remainder_l2116_211672

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end NUMINAMATH_GPT_coins_remainder_l2116_211672


namespace NUMINAMATH_GPT_probability_divisible_by_25_is_zero_l2116_211616

-- Definitions of spinner outcomes and the function to generate four-digit numbers
def is_valid_spinner_outcome (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3

def generate_four_digit_number (spin1 spin2 spin3 spin4 : ℕ) : ℕ :=
  spin1 * 1000 + spin2 * 100 + spin3 * 10 + spin4

-- Condition stating that all outcomes of each spin are equally probable among {1, 2, 3}
def valid_outcome_condition (spin1 spin2 spin3 spin4 : ℕ) : Prop :=
  is_valid_spinner_outcome spin1 ∧ is_valid_spinner_outcome spin2 ∧
  is_valid_spinner_outcome spin3 ∧ is_valid_spinner_outcome spin4

-- Probability condition for the number being divisible by 25
def is_divisible_by_25 (n : ℕ) : Prop := n % 25 = 0

-- Main theorem: proving the probability is 0
theorem probability_divisible_by_25_is_zero :
  ∀ spin1 spin2 spin3 spin4,
    valid_outcome_condition spin1 spin2 spin3 spin4 →
    ¬ is_divisible_by_25 (generate_four_digit_number spin1 spin2 spin3 spin4) :=
by
  intros spin1 spin2 spin3 spin4 h
  -- Sorry for the proof details
  sorry

end NUMINAMATH_GPT_probability_divisible_by_25_is_zero_l2116_211616


namespace NUMINAMATH_GPT_faculty_reduction_l2116_211641

theorem faculty_reduction (x : ℝ) (h1 : 0.75 * x = 195) : x = 260 :=
by sorry

end NUMINAMATH_GPT_faculty_reduction_l2116_211641


namespace NUMINAMATH_GPT_alicia_art_left_l2116_211657

-- Definition of the problem conditions.
def initial_pieces : ℕ := 70
def donated_pieces : ℕ := 46

-- The theorem to prove the number of art pieces left is 24.
theorem alicia_art_left : initial_pieces - donated_pieces = 24 := 
by
  sorry

end NUMINAMATH_GPT_alicia_art_left_l2116_211657


namespace NUMINAMATH_GPT_complex_number_modulus_l2116_211619

open Complex

theorem complex_number_modulus :
  ∀ x : ℂ, x + I = (2 - I) / I → abs x = Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_complex_number_modulus_l2116_211619


namespace NUMINAMATH_GPT_irrationals_l2116_211605

open Classical

variable (x : ℝ)

theorem irrationals (h : x^3 + 2 * x^2 + 10 * x = 20) : Irrational x ∧ Irrational (x^2) :=
by
  sorry

end NUMINAMATH_GPT_irrationals_l2116_211605


namespace NUMINAMATH_GPT_tomatoes_eaten_l2116_211674

theorem tomatoes_eaten 
  (initial_tomatoes : ℕ) 
  (final_tomatoes : ℕ) 
  (half_given : ℕ) 
  (B : ℕ) 
  (h_initial : initial_tomatoes = 127) 
  (h_final : final_tomatoes = 54) 
  (h_half : half_given = final_tomatoes * 2) 
  (h_remaining : initial_tomatoes - half_given = B)
  : B = 19 := 
by
  sorry

end NUMINAMATH_GPT_tomatoes_eaten_l2116_211674


namespace NUMINAMATH_GPT_fractions_sum_l2116_211666

theorem fractions_sum (a : ℝ) (h : a ≠ 0) : (1 / a) + (2 / a) = 3 / a := 
by 
  sorry

end NUMINAMATH_GPT_fractions_sum_l2116_211666


namespace NUMINAMATH_GPT_units_digit_of_k_squared_plus_2_to_k_l2116_211649

theorem units_digit_of_k_squared_plus_2_to_k (k : ℕ) (h : k = 2012 ^ 2 + 2 ^ 2014) : (k ^ 2 + 2 ^ k) % 10 = 5 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_k_squared_plus_2_to_k_l2116_211649


namespace NUMINAMATH_GPT_price_after_two_reductions_l2116_211683

-- Define the two reductions as given in the conditions
def first_day_reduction (P : ℝ) : ℝ := P * 0.88
def second_day_reduction (P : ℝ) : ℝ := first_day_reduction P * 0.9

-- Main theorem: Price on the second day is 79.2% of the original price
theorem price_after_two_reductions (P : ℝ) : second_day_reduction P = 0.792 * P :=
by
  sorry

end NUMINAMATH_GPT_price_after_two_reductions_l2116_211683


namespace NUMINAMATH_GPT_pencils_per_student_l2116_211626

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ)
    (h1 : total_pencils = 125)
    (h2 : students = 25)
    (h3 : pencils_per_student = total_pencils / students) :
    pencils_per_student = 5 :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_student_l2116_211626


namespace NUMINAMATH_GPT_contradiction_assumption_l2116_211662

theorem contradiction_assumption (a b : ℝ) (h : |a - 1| * |b - 1| = 0) : ¬ (a ≠ 1 ∧ b ≠ 1) :=
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l2116_211662


namespace NUMINAMATH_GPT_find_x_l2116_211631

theorem find_x (x : ℝ) (h : (0.4 + x) / 2 = 0.2025) : x = 0.005 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2116_211631


namespace NUMINAMATH_GPT_q1_monotonic_increasing_intervals_q2_proof_l2116_211646

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem q1_monotonic_increasing_intervals (a : ℝ) (h : a > 0) :
  (a > 1/2 ∧ (∀ x, (0 < x ∧ x < 1/a) ∨ (2 < x) → f a x > 0)) ∨
  (a = 1/2 ∧ (∀ x, 0 < x → f a x ≥ 0)) ∨
  (0 < a ∧ a < 1/2 ∧ (∀ x, (0 < x ∧ x < 2) ∨ (1/a < x) → f a x > 0)) := sorry

theorem q2_proof (x : ℝ) :
  (a = 0 ∧ x > 0 → f 0 x < 2 * Real.exp x - x - 4) := sorry

end NUMINAMATH_GPT_q1_monotonic_increasing_intervals_q2_proof_l2116_211646


namespace NUMINAMATH_GPT_visible_during_metaphase_l2116_211630

-- Define the structures which could be present in a plant cell during mitosis.
inductive Structure
| Chromosomes
| Spindle
| CellWall
| MetaphasePlate
| CellMembrane
| Nucleus
| Nucleolus

open Structure

-- Define what structures are visible during metaphase.
def visibleStructures (phase : String) : Set Structure :=
  if phase = "metaphase" then
    {Chromosomes, Spindle, CellWall}
  else
    ∅

-- The proof statement
theorem visible_during_metaphase :
  visibleStructures "metaphase" = {Chromosomes, Spindle, CellWall} :=
by
  sorry

end NUMINAMATH_GPT_visible_during_metaphase_l2116_211630


namespace NUMINAMATH_GPT_PS_length_correct_l2116_211693

variable {Triangle : Type}

noncomputable def PR := 15

noncomputable def PS_length (PS SR : ℝ) (PR : ℝ) : Prop :=
  PS + SR = PR ∧ (PS / SR) = (3 / 4)

theorem PS_length_correct : 
  ∃ PS SR : ℝ, PS_length PS SR PR ∧ PS = (45 / 7) :=
sorry

end NUMINAMATH_GPT_PS_length_correct_l2116_211693


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l2116_211659

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 2

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-1, quadratic_function (-1)⟩
def B : Point := ⟨1, quadratic_function 1⟩
def C : Point := ⟨2, quadratic_function 2⟩

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 :
  A.y = B.y ∧ A.y > C.y :=
by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l2116_211659


namespace NUMINAMATH_GPT_total_number_of_people_l2116_211688

theorem total_number_of_people (L F LF N T : ℕ) (hL : L = 13) (hF : F = 15) (hLF : LF = 9) (hN : N = 6) : 
  T = (L + F - LF) + N → T = 25 :=
by
  intros h
  rw [hL, hF, hLF, hN] at h
  exact h

end NUMINAMATH_GPT_total_number_of_people_l2116_211688


namespace NUMINAMATH_GPT_test_unanswered_one_way_l2116_211636

theorem test_unanswered_one_way (Q A : ℕ) (hQ : Q = 4) (hA : A = 5):
  ∀ (unanswered : ℕ), (unanswered = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_test_unanswered_one_way_l2116_211636


namespace NUMINAMATH_GPT_not_m_gt_132_l2116_211622

theorem not_m_gt_132 (m : ℕ) (hm : 0 < m)
  (H : ∃ (k : ℕ), 1 / 2 + 1 / 3 + 1 / 11 + 1 / (m:ℚ) = k) :
  m ≤ 132 :=
sorry

end NUMINAMATH_GPT_not_m_gt_132_l2116_211622


namespace NUMINAMATH_GPT_multiple_of_first_number_is_eight_l2116_211615

theorem multiple_of_first_number_is_eight 
  (a b c k : ℤ)
  (h1 : a = 7) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) 
  (h4 : 7 * k = 3 * c + (2 * b + 5)) : 
  k = 8 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_first_number_is_eight_l2116_211615


namespace NUMINAMATH_GPT_speedster_convertibles_proof_l2116_211635

-- Definitions based on conditions
def total_inventory (T : ℕ) : Prop := 2 / 3 * T = 2 / 3 * T
def not_speedsters (T : ℕ) : Prop := 1 / 3 * T = 60
def speedsters (T : ℕ) (S : ℕ) : Prop := S = 2 / 3 * T
def speedster_convertibles (S : ℕ) (C : ℕ) : Prop := C = 4 / 5 * S

theorem speedster_convertibles_proof (T S C : ℕ) (hT : total_inventory T) (hNS : not_speedsters T) (hS : speedsters T S) (hSC : speedster_convertibles S C) : C = 96 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_speedster_convertibles_proof_l2116_211635


namespace NUMINAMATH_GPT_fraction_unshaded_area_l2116_211658

theorem fraction_unshaded_area (s : ℝ) :
  let P := (s / 2, 0)
  let Q := (s, s / 2)
  let top_left := (0, s)
  let area_triangle : ℝ := 1 / 2 * (s / 2) * (s / 2)
  let area_square : ℝ := s * s
  let unshaded_area : ℝ := area_square - area_triangle
  let fraction_unshaded : ℝ := unshaded_area / area_square
  fraction_unshaded = 7 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_unshaded_area_l2116_211658


namespace NUMINAMATH_GPT_find_z_l2116_211642

theorem find_z (z : ℝ) 
    (cos_angle : (2 + 2 * z) / ((Real.sqrt (1 + z^2)) * 3) = 2 / 3) : 
    z = 0 := 
sorry

end NUMINAMATH_GPT_find_z_l2116_211642


namespace NUMINAMATH_GPT_geometric_sequence_solution_l2116_211691

open Real

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q ^ m

theorem geometric_sequence_solution :
  ∃ (a : ℕ → ℝ) (q : ℝ), geometric_sequence a q ∧
    (∀ n, 1 ≤ n ∧ n ≤ 5 → 10^8 ≤ a n ∧ a n < 10^9) ∧
    (∀ n, 6 ≤ n ∧ n ≤ 10 → 10^9 ≤ a n ∧ a n < 10^10) ∧
    (∀ n, 11 ≤ n ∧ n ≤ 14 → 10^10 ≤ a n ∧ a n < 10^11) ∧
    (∀ n, 15 ≤ n ∧ n ≤ 16 → 10^11 ≤ a n ∧ a n < 10^12) ∧
    (∀ i, a i = 7 * 3^(16-i) * 5^(i-1)) := sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l2116_211691


namespace NUMINAMATH_GPT_optimal_play_results_in_draw_l2116_211694

-- Define the concept of an optimal player, and a game state in Tic-Tac-Toe
structure Game :=
(board : Fin 3 × Fin 3 → Option Bool) -- Option Bool represents empty, O, or X
(turn : Bool) -- False for O's turn, True for X's turn

def draw (g : Game) : Bool :=
-- Implementation of checking for a draw will go here
sorry

noncomputable def optimal_move (g : Game) : Game :=
-- Implementation of finding the optimal move for the current player
sorry

theorem optimal_play_results_in_draw :
  ∀ (g : Game) (h : ∀ g, optimal_move g = g),
    draw (optimal_move g) = true :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_optimal_play_results_in_draw_l2116_211694


namespace NUMINAMATH_GPT_white_balls_count_l2116_211638

theorem white_balls_count (n : ℕ) (h : 8 / (8 + n : ℝ) = 0.4) : n = 12 := by
  sorry

end NUMINAMATH_GPT_white_balls_count_l2116_211638


namespace NUMINAMATH_GPT_sum_first_twelve_terms_of_arithmetic_sequence_l2116_211613

theorem sum_first_twelve_terms_of_arithmetic_sequence :
    let a1 := -3
    let a12 := 48
    let n := 12
    let Sn := (n * (a1 + a12)) / 2
    Sn = 270 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_twelve_terms_of_arithmetic_sequence_l2116_211613


namespace NUMINAMATH_GPT_distinct_solutions_difference_l2116_211614

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end NUMINAMATH_GPT_distinct_solutions_difference_l2116_211614


namespace NUMINAMATH_GPT_chemistry_club_student_count_l2116_211623

theorem chemistry_club_student_count (x : ℕ) (h1 : x % 3 = 0)
  (h2 : x % 4 = 0) (h3 : x % 6 = 0)
  (h4 : (x / 3) = (x / 4) + 3) :
  (x / 6) = 6 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_chemistry_club_student_count_l2116_211623


namespace NUMINAMATH_GPT_problem_1_problem_2_l2116_211603

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem problem_1 (x : ℝ) : f x + x^2 - 4 > 0 ↔ (x > 2 ∨ x < -1) := sorry

theorem problem_2 {m : ℝ} (h : m > 3) : ∃ x : ℝ, f x < g x m := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2116_211603


namespace NUMINAMATH_GPT_option_A_is_correct_l2116_211601

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_option_A_is_correct_l2116_211601


namespace NUMINAMATH_GPT_combined_final_selling_price_correct_l2116_211637

def itemA_cost : Float := 180.0
def itemB_cost : Float := 220.0
def itemC_cost : Float := 130.0

def itemA_profit_margin : Float := 0.15
def itemB_profit_margin : Float := 0.20
def itemC_profit_margin : Float := 0.25

def itemA_tax_rate : Float := 0.05
def itemB_discount_rate : Float := 0.10
def itemC_tax_rate : Float := 0.08

def itemA_selling_price_before_tax := itemA_cost * (1 + itemA_profit_margin)
def itemB_selling_price_before_discount := itemB_cost * (1 + itemB_profit_margin)
def itemC_selling_price_before_tax := itemC_cost * (1 + itemC_profit_margin)

def itemA_final_price := itemA_selling_price_before_tax * (1 + itemA_tax_rate)
def itemB_final_price := itemB_selling_price_before_discount * (1 - itemB_discount_rate)
def itemC_final_price := itemC_selling_price_before_tax * (1 + itemC_tax_rate)

def combined_final_price := itemA_final_price + itemB_final_price + itemC_final_price

theorem combined_final_selling_price_correct : 
  combined_final_price = 630.45 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_combined_final_selling_price_correct_l2116_211637


namespace NUMINAMATH_GPT_Bryan_has_more_skittles_l2116_211698

-- Definitions for conditions
def Bryan_skittles : ℕ := 50
def Ben_mms : ℕ := 20

-- Main statement to be proven
theorem Bryan_has_more_skittles : Bryan_skittles > Ben_mms ∧ Bryan_skittles - Ben_mms = 30 :=
by
  sorry

end NUMINAMATH_GPT_Bryan_has_more_skittles_l2116_211698


namespace NUMINAMATH_GPT_car_gas_tank_capacity_l2116_211665

theorem car_gas_tank_capacity
  (initial_mileage : ℕ)
  (final_mileage : ℕ)
  (miles_per_gallon : ℕ)
  (tank_fills : ℕ)
  (usage : initial_mileage = 1728)
  (usage_final : final_mileage = 2928)
  (car_efficiency : miles_per_gallon = 30)
  (fills : tank_fills = 2):
  (final_mileage - initial_mileage) / miles_per_gallon / tank_fills = 20 :=
by
  sorry

end NUMINAMATH_GPT_car_gas_tank_capacity_l2116_211665


namespace NUMINAMATH_GPT_m_is_perfect_square_l2116_211661

theorem m_is_perfect_square
  (m n k : ℕ) 
  (h1 : 0 < m) 
  (h2 : 0 < n) 
  (h3 : 0 < k) 
  (h4 : 1 + m + n * Real.sqrt 3 = (2 + Real.sqrt 3) ^ (2 * k + 1)) : 
  ∃ a : ℕ, m = a ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_m_is_perfect_square_l2116_211661


namespace NUMINAMATH_GPT_monotonicity_of_f_odd_function_a_value_l2116_211634

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

-- Part 1: Prove that f(x) is monotonically increasing
theorem monotonicity_of_f (a : ℝ) : 
  ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := by
  intro x1 x2 hx
  sorry

-- Part 2: If f(x) is an odd function, find the value of a
theorem odd_function_a_value (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : 
  f a 0 = 0 → a = 1 / 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_monotonicity_of_f_odd_function_a_value_l2116_211634


namespace NUMINAMATH_GPT_beth_marbles_left_l2116_211602

theorem beth_marbles_left :
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  T - (L_red + L_blue + L_yellow) = 42 :=
by
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  have h1 : T - (L_red + L_blue + L_yellow) = 42 := rfl
  exact h1

end NUMINAMATH_GPT_beth_marbles_left_l2116_211602


namespace NUMINAMATH_GPT_distance_A_B_l2116_211633

noncomputable def distance_between_points (v_A v_B : ℝ) (t : ℝ) : ℝ := 5 * (6 * t / (2 / 3 * t))

theorem distance_A_B
  (v_A v_B : ℝ)
  (t : ℝ)
  (h1 : v_A = 1.2 * v_B)
  (h2 : ∃ distance_broken, distance_broken = 5)
  (h3 : ∃ delay, delay = (1 / 6) * 6 * t)
  (h4 : ∃ v_B_new, v_B_new = 1.6 * v_B)
  (h5 : distance_between_points v_A v_B t = 45) :
  distance_between_points v_A v_B t = 45 :=
sorry

end NUMINAMATH_GPT_distance_A_B_l2116_211633


namespace NUMINAMATH_GPT_quadratic_real_solutions_l2116_211610

theorem quadratic_real_solutions (m : ℝ) :
  (∃ (x : ℝ), m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_solutions_l2116_211610


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_17_is_1013_l2116_211695

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_17_is_1013_l2116_211695


namespace NUMINAMATH_GPT_compare_y1_y2_l2116_211609

theorem compare_y1_y2 (a : ℝ) (y1 y2 : ℝ) (h₁ : a < 0) (h₂ : y1 = a * (-1 - 1)^2 + 3) (h₃ : y2 = a * (2 - 1)^2 + 3) : 
  y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_compare_y1_y2_l2116_211609


namespace NUMINAMATH_GPT_zero_cleverly_numbers_l2116_211640

theorem zero_cleverly_numbers (n : ℕ) : 
  (1000 ≤ n ∧ n < 10000) ∧ (∃ a b c, n = 1000 * a + 10 * b + c ∧ b = 0 ∧ 9 * (100 * a + 10 * b + c) = n) ↔ (n = 2025 ∨ n = 4050 ∨ n = 6075) := 
sorry

end NUMINAMATH_GPT_zero_cleverly_numbers_l2116_211640


namespace NUMINAMATH_GPT_brian_needs_some_cartons_l2116_211629

def servings_per_person : ℕ := sorry -- This should be defined with the actual number of servings per person.
def family_members : ℕ := 8
def us_cup_in_ml : ℕ := 250
def ml_per_serving : ℕ := us_cup_in_ml / 2
def ml_per_liter : ℕ := 1000

def total_milk_needed (servings_per_person : ℕ) : ℕ :=
  family_members * servings_per_person * ml_per_serving

def cartons_of_milk_needed (servings_per_person : ℕ) : ℕ :=
  total_milk_needed servings_per_person / ml_per_liter + if total_milk_needed servings_per_person % ml_per_liter = 0 then 0 else 1

theorem brian_needs_some_cartons (servings_per_person : ℕ) : 
  cartons_of_milk_needed servings_per_person = (family_members * servings_per_person * ml_per_serving / ml_per_liter + 
  if (family_members * servings_per_person * ml_per_serving) % ml_per_liter = 0 then 0 else 1) := 
by 
  sorry

end NUMINAMATH_GPT_brian_needs_some_cartons_l2116_211629


namespace NUMINAMATH_GPT_max_unique_solution_l2116_211608

theorem max_unique_solution (x y : ℕ) (m : ℕ) (h : 2005 * x + 2007 * y = m) : 
  m = 2 * 2005 * 2007 ↔ ∃! (x y : ℕ), 2005 * x + 2007 * y = m :=
sorry

end NUMINAMATH_GPT_max_unique_solution_l2116_211608


namespace NUMINAMATH_GPT_trisha_collects_4_dozen_less_l2116_211667

theorem trisha_collects_4_dozen_less (B C T : ℕ) 
  (h1 : B = 6) 
  (h2 : C = 3 * B) 
  (h3 : B + C + T = 26) : 
  B - T = 4 := 
by 
  sorry

end NUMINAMATH_GPT_trisha_collects_4_dozen_less_l2116_211667


namespace NUMINAMATH_GPT_nancy_crayons_l2116_211670

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end NUMINAMATH_GPT_nancy_crayons_l2116_211670


namespace NUMINAMATH_GPT_value_of_a_plus_b_l2116_211618

theorem value_of_a_plus_b (a b x y : ℝ) 
  (h1 : 2 * x + 4 * y = 20)
  (h2 : a * x + b * y = 1)
  (h3 : 2 * x - y = 5)
  (h4 : b * x + a * y = 6) : a + b = 1 := 
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l2116_211618


namespace NUMINAMATH_GPT_problem1_problem2_l2116_211699

-- We define a point P(x, y) on the circle x^2 + y^2 = 2y.
variables {x y a : ℝ}

-- Condition for the point P to be on the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Definition for 2x + y range
def range_2x_plus_y (x y : ℝ) : Prop := - Real.sqrt 5 + 1 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 5 + 1

-- Definition for the range of a given x + y + a ≥ 0
def range_a (x y a : ℝ) : Prop := x + y + a ≥ 0 → a ≥ Real.sqrt 2 - 1

-- Main statements to prove
theorem problem1 (hx : on_circle x y) : range_2x_plus_y x y := sorry

theorem problem2 (hx : on_circle x y) (h : ∀ θ, x = Real.cos θ ∧ y = 1 + Real.sin θ) : range_a x y a := sorry

end NUMINAMATH_GPT_problem1_problem2_l2116_211699


namespace NUMINAMATH_GPT_soccer_players_l2116_211675

/-- 
If the total number of socks in the washing machine is 16,
and each player wears a pair of socks (2 socks per player), 
then the number of players is 8. 
-/
theorem soccer_players (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) : total_socks / socks_per_player = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_soccer_players_l2116_211675


namespace NUMINAMATH_GPT_frequency_of_third_group_l2116_211606

theorem frequency_of_third_group (total_data first_group second_group fourth_group third_group : ℕ) 
    (h1 : total_data = 40)
    (h2 : first_group = 5)
    (h3 : second_group = 12)
    (h4 : fourth_group = 8) :
    third_group = 15 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_third_group_l2116_211606


namespace NUMINAMATH_GPT_simplify_complex_l2116_211668

open Complex

theorem simplify_complex : (5 : ℂ) / (I - 2) = -2 - I := by
  sorry

end NUMINAMATH_GPT_simplify_complex_l2116_211668


namespace NUMINAMATH_GPT_eight_diamond_five_l2116_211664

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem eight_diamond_five : diamond 8 5 = 160 :=
by sorry

end NUMINAMATH_GPT_eight_diamond_five_l2116_211664


namespace NUMINAMATH_GPT_distinct_9_pointed_stars_l2116_211678

-- Define a function to count the distinct n-pointed stars for a given n
def count_distinct_stars (n : ℕ) : ℕ :=
  -- Functionality to count distinct stars will be implemented here
  sorry

-- Theorem stating the number of distinct 9-pointed stars
theorem distinct_9_pointed_stars : count_distinct_stars 9 = 2 :=
  sorry

end NUMINAMATH_GPT_distinct_9_pointed_stars_l2116_211678


namespace NUMINAMATH_GPT_complete_task_in_3_days_l2116_211607

theorem complete_task_in_3_days (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0)
  (h1 : 1 / x + 1 / y + 1 / z = 1 / 7.5)
  (h2 : 1 / x + 1 / z + 1 / v = 1 / 5)
  (h3 : 1 / x + 1 / z + 1 / w = 1 / 6)
  (h4 : 1 / y + 1 / w + 1 / v = 1 / 4) :
  1 / (1 / x + 1 / z + 1 / v + 1 / w + 1 / y) = 3 :=
sorry

end NUMINAMATH_GPT_complete_task_in_3_days_l2116_211607


namespace NUMINAMATH_GPT_correct_operation_result_l2116_211651

-- Define the conditions
def original_number : ℤ := 231
def incorrect_result : ℤ := 13

-- Define the two incorrect operations and the intended corrections
def reverse_subtract : ℤ := incorrect_result + 20
def reverse_division : ℤ := reverse_subtract * 7

-- Define the intended operations
def intended_multiplication : ℤ := original_number * 7
def intended_addition : ℤ := intended_multiplication + 20

-- The theorem we need to prove
theorem correct_operation_result :
  original_number = reverse_division →
  intended_addition > 1100 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_correct_operation_result_l2116_211651


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l2116_211643

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l2116_211643


namespace NUMINAMATH_GPT_cricket_bat_cost_l2116_211692

noncomputable def CP_A_sol : ℝ := 444.96 / 1.95

theorem cricket_bat_cost (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (SP_D : ℝ) :
  (SP_B = 1.20 * CP_A) →
  (SP_C = 1.25 * SP_B) →
  (SP_D = 1.30 * SP_C) →
  (SP_D = 444.96) →
  CP_A = CP_A_sol :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cricket_bat_cost_l2116_211692


namespace NUMINAMATH_GPT_find_a_20_l2116_211679

-- Arithmetic sequence definition and known conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)

-- Conditions
def condition1 : Prop := a 1 + a 2 + a 3 = 6
def condition2 : Prop := a 5 = 8

-- The main statement to prove
theorem find_a_20 (h_arith : arithmetic_sequence a) (h_cond1 : condition1 a) (h_cond2 : condition2 a) : 
  a 20 = 38 := by
  sorry

end NUMINAMATH_GPT_find_a_20_l2116_211679


namespace NUMINAMATH_GPT_hcf_of_36_and_x_is_12_l2116_211684

theorem hcf_of_36_and_x_is_12 (x : ℕ) (h : Nat.gcd 36 x = 12) : x = 48 :=
sorry

end NUMINAMATH_GPT_hcf_of_36_and_x_is_12_l2116_211684


namespace NUMINAMATH_GPT_inverse_proportion_l2116_211673

theorem inverse_proportion (α β k : ℝ) (h1 : α * β = k) (h2 : α = 5) (h3 : β = 10) : (α = 25 / 2) → (β = 4) := by sorry

end NUMINAMATH_GPT_inverse_proportion_l2116_211673


namespace NUMINAMATH_GPT_prob_correct_l2116_211647

-- Define percentages as ratio values
def prob_beginner_excel : ℝ := 0.35
def prob_intermediate_excel : ℝ := 0.25
def prob_advanced_excel : ℝ := 0.20
def prob_no_excel : ℝ := 0.20

def prob_day_shift : ℝ := 0.70
def prob_night_shift : ℝ := 0.30

def prob_weekend : ℝ := 0.40
def prob_not_weekend : ℝ := 0.60

-- Define the target probability calculation
def prob_intermediate_or_advanced_excel : ℝ := prob_intermediate_excel + prob_advanced_excel
def prob_combined : ℝ := prob_intermediate_or_advanced_excel * prob_night_shift * prob_not_weekend

-- The proof problem statement
theorem prob_correct : prob_combined = 0.081 :=
by
  sorry

end NUMINAMATH_GPT_prob_correct_l2116_211647


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l2116_211676

theorem perpendicular_lines_slope (a : ℝ)
  (h : (a * (a + 2)) = -1) : a = -1 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l2116_211676


namespace NUMINAMATH_GPT_proportion_of_triumphal_arch_photographs_l2116_211663

-- Define the constants
variables (x y z t : ℕ) -- x = castles, y = triumphal arches, z = waterfalls, t = cathedrals

-- The conditions
axiom half_photographed : t + x + y + z = (3*y + 2*x + 2*z + y) / 2
axiom three_times_cathedrals : ∃ (a : ℕ), t = 3 * a ∧ y = a
axiom same_castles_waterfalls : ∃ (b : ℕ), t + z = x + y
axiom quarter_photographs_castles : x = (t + x + y + z) / 4
axiom second_castle_frequency : t + z = 2 * x
axiom every_triumphal_arch_photographed : ∀ (c : ℕ), y = c ∧ y = c

theorem proportion_of_triumphal_arch_photographs : 
  ∃ (p : ℚ), p = 1 / 4 ∧ p = y / ((t + x + y + z) / 2) :=
sorry

end NUMINAMATH_GPT_proportion_of_triumphal_arch_photographs_l2116_211663


namespace NUMINAMATH_GPT_find_values_and_properties_l2116_211696

variable (f : ℝ → ℝ)

axiom f_neg1 : f (-1) = 2
axiom f_pos_x : ∀ x, x < 0 → f x > 1
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y

theorem find_values_and_properties :
  f 0 = 1 ∧
  f (-4) = 16 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (-4 * x^2) * f (10 * x) ≥ 1/16 ↔ x ≤ 1/2 ∨ x ≥ 2) :=
sorry

end NUMINAMATH_GPT_find_values_and_properties_l2116_211696


namespace NUMINAMATH_GPT_find_2theta_plus_phi_l2116_211681

variable (θ φ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (hφ : 0 < φ ∧ φ < π / 2)
variable (tan_hθ : Real.tan θ = 2 / 5)
variable (cos_hφ : Real.cos φ = 1 / 2)

theorem find_2theta_plus_phi : 2 * θ + φ = π / 4 := by
  sorry

end NUMINAMATH_GPT_find_2theta_plus_phi_l2116_211681


namespace NUMINAMATH_GPT_find_p_l2116_211624

theorem find_p (m n p : ℚ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + 18) / 6 - 2 / 5) : 
  p = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_p_l2116_211624


namespace NUMINAMATH_GPT_max_remainder_l2116_211671

theorem max_remainder (y : ℕ) : 
  ∃ q r : ℕ, y = 11 * q + r ∧ r < 11 ∧ r = 10 := by sorry

end NUMINAMATH_GPT_max_remainder_l2116_211671


namespace NUMINAMATH_GPT_solve_expression_l2116_211655

theorem solve_expression (a b c : ℝ) (ha : a^3 - 2020*a^2 + 1010 = 0) (hb : b^3 - 2020*b^2 + 1010 = 0) (hc : c^3 - 2020*c^2 + 1010 = 0) (habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
    (1 / (a * b) + 1 / (b * c) + 1 / (a * c) = -2) := 
sorry

end NUMINAMATH_GPT_solve_expression_l2116_211655
