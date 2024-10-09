import Mathlib

namespace compound_O_atoms_l1919_191988

theorem compound_O_atoms (Cu_weight C_weight O_weight compound_weight : ℝ)
  (Cu_atoms : ℕ) (C_atoms : ℕ) (O_atoms : ℕ)
  (hCu : Cu_weight = 63.55)
  (hC : C_weight = 12.01)
  (hO : O_weight = 16.00)
  (h_compound_weight : compound_weight = 124)
  (h_atoms : Cu_atoms = 1 ∧ C_atoms = 1)
  : O_atoms = 3 :=
sorry

end compound_O_atoms_l1919_191988


namespace problem_1_problem_2_l1919_191972

def p (x : ℝ) : Prop := -x^2 + 6*x + 16 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0 ∧ m > 0

theorem problem_1 (x : ℝ) : p x → -2 ≤ x ∧ x ≤ 8 :=
by
  -- Proof goes here
  sorry

theorem problem_2 (m : ℝ) : (∀ x, p x → q x m) ∧ (∃ x, ¬ p x ∧ q x m) → m ≥ 6 :=
by
  -- Proof goes here
  sorry

end problem_1_problem_2_l1919_191972


namespace class_overall_score_l1919_191906

def max_score : ℝ := 100
def percentage_study : ℝ := 0.4
def percentage_hygiene : ℝ := 0.25
def percentage_discipline : ℝ := 0.25
def percentage_activity : ℝ := 0.1

def score_study : ℝ := 85
def score_hygiene : ℝ := 90
def score_discipline : ℝ := 80
def score_activity : ℝ := 75

theorem class_overall_score :
  (score_study * percentage_study) +
  (score_hygiene * percentage_hygiene) +
  (score_discipline * percentage_discipline) +
  (score_activity * percentage_activity) = 84 :=
  by sorry

end class_overall_score_l1919_191906


namespace vegetarian_count_l1919_191914

theorem vegetarian_count (only_veg only_non_veg both_veg_non_veg : ℕ) 
  (h1 : only_veg = 19) (h2 : only_non_veg = 9) (h3 : both_veg_non_veg = 12) : 
  (only_veg + both_veg_non_veg = 31) :=
by
  -- We leave the proof here
  sorry

end vegetarian_count_l1919_191914


namespace quadratic_has_one_real_root_l1919_191965

theorem quadratic_has_one_real_root (k : ℝ) : 
  (∃ (x : ℝ), -2 * x^2 + 8 * x + k = 0 ∧ ∀ y, -2 * y^2 + 8 * y + k = 0 → y = x) ↔ k = -8 := 
by
  sorry

end quadratic_has_one_real_root_l1919_191965


namespace circle_convex_polygons_count_l1919_191958

theorem circle_convex_polygons_count : 
  let total_subsets := (2^15 - 1) - (15 + 105 + 455 + 255)
  let final_count := total_subsets - 500
  final_count = 31437 :=
by
  sorry

end circle_convex_polygons_count_l1919_191958


namespace common_roots_of_cubic_polynomials_l1919_191957

/-- The polynomials \( x^3 + 6x^2 + 11x + 6 \) and \( x^3 + 7x^2 + 14x + 8 \) have two distinct roots in common. -/
theorem common_roots_of_cubic_polynomials :
  ∃ r s : ℝ, r ≠ s ∧ (r^3 + 6 * r^2 + 11 * r + 6 = 0) ∧ (s^3 + 6 * s^2 + 11 * s + 6 = 0)
  ∧ (r^3 + 7 * r^2 + 14 * r + 8 = 0) ∧ (s^3 + 7 * s^2 + 14 * s + 8 = 0) :=
sorry

end common_roots_of_cubic_polynomials_l1919_191957


namespace sum_of_edges_l1919_191904

theorem sum_of_edges (n : ℕ) (total_length large_edge small_edge : ℤ) : 
  n = 27 → 
  total_length = 828 → -- convert to millimeters
  large_edge = total_length / 12 → 
  small_edge = large_edge / 3 → 
  (large_edge + small_edge) / 10 = 92 :=
by
  intros
  sorry

end sum_of_edges_l1919_191904


namespace appleJuicePercentageIsCorrect_l1919_191927

-- Define the initial conditions
def MikiHas : ℕ × ℕ := (15, 10) -- Miki has 15 apples and 10 bananas

-- Define the juice extraction rates
def appleJuicePerApple : ℚ := 9 / 3 -- 9 ounces from 3 apples
def bananaJuicePerBanana : ℚ := 10 / 2 -- 10 ounces from 2 bananas

-- Define the number of apples and bananas used for the blend
def applesUsed : ℕ := 5
def bananasUsed : ℕ := 4

-- Calculate the total juice extracted
def appleJuice : ℚ := applesUsed * appleJuicePerApple
def bananaJuice : ℚ := bananasUsed * bananaJuicePerBanana

-- Calculate the total juice and percentage of apple juice
def totalJuice : ℚ := appleJuice + bananaJuice
def percentageAppleJuice : ℚ := (appleJuice / totalJuice) * 100

theorem appleJuicePercentageIsCorrect : percentageAppleJuice = 42.86 := by
  sorry

end appleJuicePercentageIsCorrect_l1919_191927


namespace g_at_9_l1919_191900

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l1919_191900


namespace multiple_of_10_and_12_within_100_l1919_191971

theorem multiple_of_10_and_12_within_100 :
  ∀ (n : ℕ), n ≤ 100 → (∃ k₁ k₂ : ℕ, n = 10 * k₁ ∧ n = 12 * k₂) ↔ n = 60 :=
by
  sorry

end multiple_of_10_and_12_within_100_l1919_191971


namespace margaret_speed_on_time_l1919_191925
-- Import the necessary libraries from Mathlib

-- Define the problem conditions and state the theorem
theorem margaret_speed_on_time :
  ∃ r : ℝ, (∀ d t : ℝ,
    d = 50 * (t - 1/12) ∧
    d = 30 * (t + 1/12) →
    r = d / t) ∧
  r = 37.5 := 
sorry

end margaret_speed_on_time_l1919_191925


namespace f_g_evaluation_l1919_191951

-- Definitions of the functions g and f
def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3 * x - 2

-- Goal: Prove that f(g(2)) = 22
theorem f_g_evaluation : f (g 2) = 22 :=
by
  sorry

end f_g_evaluation_l1919_191951


namespace player2_wins_l1919_191923

-- Definitions for the initial conditions and game rules
def initial_piles := [10, 15, 20]
def split_rule (piles : List ℕ) (move : ℕ → ℕ × ℕ) : List ℕ :=
  let (pile1, pile2) := move (piles.head!)
  (pile1 :: pile2 :: piles.tail!)

-- Winning condition proof
theorem player2_wins :
  ∀ piles : List ℕ, piles = [10, 15, 20] →
  (∀ move_count : ℕ, move_count = 42 →
  (move_count > 0 ∧ ¬ ∃ split : ℕ → ℕ × ℕ, move_count % 2 = 1)) :=
by
  intro piles hpiles
  intro move_count hmove_count
  sorry

end player2_wins_l1919_191923


namespace matroskin_milk_amount_l1919_191936

theorem matroskin_milk_amount :
  ∃ S M x : ℝ, S + M = 10 ∧ (S - x) = (1 / 3) * S ∧ (M + x) = 3 * M ∧ (M + x) = 7.5 := 
sorry

end matroskin_milk_amount_l1919_191936


namespace horner_polynomial_rewrite_polynomial_value_at_5_l1919_191938

def polynomial (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 6 * x^3 - 2 * x^2 - 5 * x - 2

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_polynomial_rewrite :
  polynomial = horner_polynomial := 
sorry

theorem polynomial_value_at_5 :
  polynomial 5 = 7548 := 
sorry

end horner_polynomial_rewrite_polynomial_value_at_5_l1919_191938


namespace find_smaller_number_l1919_191979

theorem find_smaller_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : x = 18 :=
by
  sorry

end find_smaller_number_l1919_191979


namespace population_increase_l1919_191915

theorem population_increase (birth_rate : ℝ) (death_rate : ℝ) (initial_population : ℝ) :
  initial_population = 1000 →
  birth_rate = 32 / 1000 →
  death_rate = 11 / 1000 →
  ((birth_rate - death_rate) / initial_population) * 100 = 2.1 :=
by
  sorry

end population_increase_l1919_191915


namespace incorrect_table_value_l1919_191950

theorem incorrect_table_value (a b c : ℕ) (values : List ℕ) (correct : values = [2051, 2197, 2401, 2601, 2809, 3025, 3249, 3481]) : 
  (2401 ∉ [2051, 2197, 2399, 2601, 2809, 3025, 3249, 3481]) :=
sorry

end incorrect_table_value_l1919_191950


namespace main_l1919_191978

theorem main (x y : ℤ) (h1 : abs x = 5) (h2 : abs y = 3) (h3 : x * y > 0) : 
    x - y = 2 ∨ x - y = -2 := sorry

end main_l1919_191978


namespace symmetric_about_y_axis_l1919_191983

theorem symmetric_about_y_axis (m n : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (n + 1, 4))
  (symmetry : A.1 = -B.1)
  : m = 2.5 ∧ n = 2 :=
by
  sorry

end symmetric_about_y_axis_l1919_191983


namespace cryptarithm_solution_exists_l1919_191982

theorem cryptarithm_solution_exists :
  ∃ (L E S O : ℕ), L ≠ E ∧ L ≠ S ∧ L ≠ O ∧ E ≠ S ∧ E ≠ O ∧ S ≠ O ∧
  (L < 10) ∧ (E < 10) ∧ (S < 10) ∧ (O < 10) ∧
  (1000 * O + 100 * S + 10 * E + L) +
  (100 * S + 10 * E + L) +
  (10 * E + L) +
  L = 10034 ∧
  ((L = 6 ∧ E = 7 ∧ S = 4 ∧ O = 9) ∨
   (L = 6 ∧ E = 7 ∧ S = 9 ∧ O = 8)) :=
by
  -- The proof is omitted here.
  sorry

end cryptarithm_solution_exists_l1919_191982


namespace square_D_perimeter_l1919_191917

theorem square_D_perimeter 
(C_perimeter: Real) 
(D_area_ratio : Real) 
(hC : C_perimeter = 32) 
(hD : D_area_ratio = 1/3) : 
    ∃ D_perimeter, D_perimeter = (32 * Real.sqrt 3) / 3 := 
by 
    sorry

end square_D_perimeter_l1919_191917


namespace empty_subset_of_A_l1919_191924

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A :=
by
  sorry

end empty_subset_of_A_l1919_191924


namespace find_principal_sum_l1919_191930

theorem find_principal_sum (R P : ℝ) 
  (h1 : (3 * P * (R + 1) / 100 - 3 * P * R / 100) = 72) : 
  P = 2400 := 
by 
  sorry

end find_principal_sum_l1919_191930


namespace part_a_part_b_l1919_191928

namespace ProofProblem

def number_set := {n : ℕ | ∃ k : ℕ, n = (10^k - 1)}

noncomputable def special_structure (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2 * m + 1 ∨ n = 2 * m + 2

theorem part_a :
  ∃ (a b c : ℕ) (ha : a ∈ number_set) (hb : b ∈ number_set) (hc : c ∈ number_set),
    special_structure (a + b + c) :=
by
  sorry

theorem part_b (cards : List ℕ) (h : ∀ x ∈ cards, x ∈ number_set)
    (hs : special_structure (cards.sum)) :
  ∃ (d : ℕ), d ≠ 2 ∧ (d = 0 ∨ d = 1) :=
by
  sorry

end ProofProblem

end part_a_part_b_l1919_191928


namespace achieve_target_ratio_l1919_191944

-- Initial volume and ratio
def initial_volume : ℕ := 20
def initial_milk_ratio : ℕ := 3
def initial_water_ratio : ℕ := 2

-- Mixture removal and addition
def removal_volume : ℕ := 10
def added_milk : ℕ := 10

-- Target ratio of milk to water
def target_milk_ratio : ℕ := 9
def target_water_ratio : ℕ := 1

-- Number of operations required
def operations_needed: ℕ := 2

-- Statement of proof problem
theorem achieve_target_ratio :
  (initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) + added_milk * operations_needed) / 
  (initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) = target_milk_ratio :=
sorry

end achieve_target_ratio_l1919_191944


namespace trajectory_of_P_eqn_l1919_191902

noncomputable def point_A : ℝ × ℝ := (1, 0)

def curve_C (x : ℝ) : ℝ := x^2 - 2

def symmetric_point (Qx Qy Px Py : ℝ) : Prop :=
  Qx = 2 - Px ∧ Qy = -Py

theorem trajectory_of_P_eqn (Qx Qy Px Py : ℝ) (hQ_on_C : Qy = curve_C Qx)
  (h_symm : symmetric_point Qx Qy Px Py) :
  Py = -Px^2 + 4 * Px - 2 :=
by
  sorry

end trajectory_of_P_eqn_l1919_191902


namespace solve_cubic_eq_with_geo_prog_coeff_l1919_191908

variables {a q x : ℝ}

theorem solve_cubic_eq_with_geo_prog_coeff (h_a_nonzero : a ≠ 0) 
    (h_b : b = a * q) (h_c : c = a * q^2) (h_d : d = a * q^3) :
    (a * x^3 + b * x^2 + c * x + d = 0) → (x = -q) :=
by
  intros h_cubic_eq
  have h_b' : b = a * q := h_b
  have h_c' : c = a * q^2 := h_c
  have h_d' : d = a * q^3 := h_d
  sorry

end solve_cubic_eq_with_geo_prog_coeff_l1919_191908


namespace mathematicians_correctness_l1919_191952

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l1919_191952


namespace quadrilateral_area_proof_l1919_191963

-- Definitions of points
def A : (ℝ × ℝ) := (1, 3)
def B : (ℝ × ℝ) := (1, 1)
def C : (ℝ × ℝ) := (3, 1)
def D : (ℝ × ℝ) := (2010, 2011)

-- Function to calculate the area of the quadrilateral
def area_of_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ := 
  let area_triangle (P Q R : (ℝ × ℝ)) : ℝ := 
    0.5 * (P.1 * Q.2 + Q.1 * R.2 + R.1 * P.2 - P.2 * Q.1 - Q.2 * R.1 - R.2 * P.1)
  area_triangle A B C + area_triangle A C D

-- Lean statement to prove the desired area
theorem quadrilateral_area_proof : area_of_quadrilateral A B C D = 7 := 
  sorry

end quadrilateral_area_proof_l1919_191963


namespace students_neither_l1919_191981

-- Define the given conditions
def total_students : Nat := 460
def football_players : Nat := 325
def cricket_players : Nat := 175
def both_players : Nat := 90

-- Define the Lean statement for the proof problem
theorem students_neither (total_students football_players cricket_players both_players : Nat) (h1 : total_students = 460)
  (h2 : football_players = 325) (h3 : cricket_players = 175) (h4 : both_players = 90) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_l1919_191981


namespace product_of_two_numbers_l1919_191956

theorem product_of_two_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 560) (h_hcf : Nat.gcd a b = 75) :
  a * b = 42000 :=
by
  sorry

end product_of_two_numbers_l1919_191956


namespace train_distance_proof_l1919_191984

-- Definitions
def speed_train1 : ℕ := 40
def speed_train2 : ℕ := 48
def time_hours : ℕ := 8
def initial_distance : ℕ := 892

-- Function to calculate distance after given time
def distance (speed time : ℕ) : ℕ := speed * time

-- Increased/Decreased distance after time
def distance_diff : ℕ := distance speed_train2 time_hours - distance speed_train1 time_hours

-- Final distances
def final_distance_same_direction : ℕ := initial_distance + distance_diff
def final_distance_opposite_direction : ℕ := initial_distance - distance_diff

-- Proof statement
theorem train_distance_proof :
  final_distance_same_direction = 956 ∧ final_distance_opposite_direction = 828 :=
by
  -- The proof is omitted here
  sorry

end train_distance_proof_l1919_191984


namespace triangle_area_l1919_191955

/-- 
In a triangle ABC, given that ∠B=30°, AB=2√3, and AC=2, 
prove that the area of the triangle ABC is either √3 or 2√3.
 -/
theorem triangle_area (B : Real) (AB AC : Real) 
  (h_B : B = 30) (h_AB : AB = 2 * Real.sqrt 3) (h_AC : AC = 2) :
  ∃ S : Real, (S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3) := 
by 
  sorry

end triangle_area_l1919_191955


namespace weight_of_3_moles_of_BaF2_is_correct_l1919_191919

-- Definitions for the conditions
def atomic_weight_Ba : ℝ := 137.33 -- g/mol
def atomic_weight_F : ℝ := 19.00 -- g/mol

-- Definition of the molecular weight of BaF2
def molecular_weight_BaF2 : ℝ := (1 * atomic_weight_Ba) + (2 * atomic_weight_F)

-- The statement to prove
theorem weight_of_3_moles_of_BaF2_is_correct : (3 * molecular_weight_BaF2) = 525.99 :=
by
  -- Proof omitted
  sorry

end weight_of_3_moles_of_BaF2_is_correct_l1919_191919


namespace angle_405_eq_45_l1919_191970

def same_terminal_side (angle1 angle2 : ℝ) : Prop :=
  ∃ k : ℤ, angle1 = angle2 + k * 360

theorem angle_405_eq_45 (k : ℤ) : same_terminal_side 405 45 := 
sorry

end angle_405_eq_45_l1919_191970


namespace geom_seq_a5_l1919_191998

noncomputable def S3 (a1 q : ℚ) : ℚ := a1 + a1 * q^2
noncomputable def a (a1 q : ℚ) (n : ℕ) : ℚ := a1 * q^(n - 1)

theorem geom_seq_a5 (a1 q : ℚ) (hS3 : S3 a1 q = 5 * a1) (ha7 : a a1 q 7 = 2) :
  a a1 q 5 = 1 / 2 :=
by
  sorry

end geom_seq_a5_l1919_191998


namespace overall_percentage_change_is_113_point_4_l1919_191932

-- Define the conditions
def total_customers_survey_1 := 100
def male_percentage_survey_1 := 60
def respondents_survey_1 := 10
def male_respondents_survey_1 := 5

def total_customers_survey_2 := 80
def male_percentage_survey_2 := 70
def respondents_survey_2 := 16
def male_respondents_survey_2 := 12

def total_customers_survey_3 := 70
def male_percentage_survey_3 := 40
def respondents_survey_3 := 21
def male_respondents_survey_3 := 13

def total_customers_survey_4 := 90
def male_percentage_survey_4 := 50
def respondents_survey_4 := 27
def male_respondents_survey_4 := 8

-- Define the calculated response rates
def original_male_response_rate := (male_respondents_survey_1.toFloat / (total_customers_survey_1 * male_percentage_survey_1 / 100).toFloat) * 100
def final_male_response_rate := (male_respondents_survey_4.toFloat / (total_customers_survey_4 * male_percentage_survey_4 / 100).toFloat) * 100

-- Calculate the percentage change in response rate
def percentage_change := ((final_male_response_rate - original_male_response_rate) / original_male_response_rate) * 100

-- The target theorem 
theorem overall_percentage_change_is_113_point_4 : percentage_change = 113.4 := sorry

end overall_percentage_change_is_113_point_4_l1919_191932


namespace no_pair_of_primes_l1919_191912

theorem no_pair_of_primes (p q : ℕ) (hp_prime : Prime p) (hq_prime : Prime q) (h_gt : p > q) :
  ¬ (∃ (h : ℤ), 2 * (p^2 - q^2) = 8 * h + 4) :=
by
  sorry

end no_pair_of_primes_l1919_191912


namespace gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l1919_191968

theorem gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1 :
  Int.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
  -- proof goes here
  sorry

end gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l1919_191968


namespace gallons_per_hour_l1919_191966

-- Define conditions
def total_runoff : ℕ := 240000
def days : ℕ := 10
def hours_per_day : ℕ := 24

-- Define the goal: proving the sewers handle 1000 gallons of run-off per hour
theorem gallons_per_hour : (total_runoff / (days * hours_per_day)) = 1000 :=
by
  -- Proof can be inserted here
  sorry

end gallons_per_hour_l1919_191966


namespace no_positive_integer_solution_l1919_191918

theorem no_positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ¬ (x^2 * y^4 - x^4 * y^2 + 4 * x^2 * y^2 * z^2 + x^2 * z^4 - y^2 * z^4 = 0) :=
sorry

end no_positive_integer_solution_l1919_191918


namespace max_radius_of_circle_l1919_191903

theorem max_radius_of_circle (c : ℝ × ℝ → Prop) (h1 : c (16, 0)) (h2 : c (-16, 0)) :
  ∃ r : ℝ, r = 16 :=
by
  sorry

end max_radius_of_circle_l1919_191903


namespace find_age_of_older_friend_l1919_191909

theorem find_age_of_older_friend (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) : 
  A = 104.25 :=
by
  sorry

end find_age_of_older_friend_l1919_191909


namespace length_of_each_reel_l1919_191994

theorem length_of_each_reel
  (reels : ℕ)
  (sections : ℕ)
  (length_per_section : ℕ)
  (total_sections : ℕ)
  (h1 : reels = 3)
  (h2 : length_per_section = 10)
  (h3 : total_sections = 30)
  : (total_sections * length_per_section) / reels = 100 := 
by
  sorry

end length_of_each_reel_l1919_191994


namespace at_least_two_fail_l1919_191995

theorem at_least_two_fail (p q : ℝ) (n : ℕ) (h_p : p = 0.2) (h_q : q = 1 - p) :
  n ≥ 18 → (1 - ((q^n) * (1 + n * p / 4))) ≥ 0.9 :=
by
  sorry

end at_least_two_fail_l1919_191995


namespace distance_between_A_and_B_l1919_191973

noncomputable def distance_between_points (v_A v_B : ℝ) (t_meet t_A_to_B_after_meet : ℝ) : ℝ :=
  let t_total_A := t_meet + t_A_to_B_after_meet
  let t_total_B := t_meet + (t_meet - t_A_to_B_after_meet)
  let D := v_A * t_total_A + v_B * t_total_B
  D

-- Given conditions
def t_meet : ℝ := 4
def t_A_to_B_after_meet : ℝ := 3
def speed_difference : ℝ := 20

-- Function to calculate speeds based on given conditions
noncomputable def calculate_speeds (v_B : ℝ) : ℝ × ℝ :=
  let v_A := v_B + speed_difference
  (v_A, v_B)

-- Statement of the problem in Lean 4
theorem distance_between_A_and_B : ∃ (v_B v_A : ℝ), 
  v_A = v_B + speed_difference ∧
  distance_between_points v_A v_B t_meet t_A_to_B_after_meet = 240 :=
by 
  sorry

end distance_between_A_and_B_l1919_191973


namespace john_quiz_goal_l1919_191974

theorem john_quiz_goal
  (total_quizzes : ℕ)
  (goal_percentage : ℕ)
  (quizzes_completed : ℕ)
  (quizzes_remaining : ℕ)
  (quizzes_with_A_completed : ℕ)
  (total_quizzes_with_A_needed : ℕ)
  (additional_A_needed : ℕ)
  (quizzes_below_A_allowed : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 75)
  (h3 : quizzes_completed = 40)
  (h4 : quizzes_remaining = total_quizzes - quizzes_completed)
  (h5 : quizzes_with_A_completed = 27)
  (h6 : total_quizzes_with_A_needed = total_quizzes * goal_percentage / 100)
  (h7 : additional_A_needed = total_quizzes_with_A_needed - quizzes_with_A_completed)
  (h8 : quizzes_below_A_allowed = quizzes_remaining - additional_A_needed)
  (h_goal : quizzes_below_A_allowed ≤ 2) : quizzes_below_A_allowed = 2 :=
by
  sorry

end john_quiz_goal_l1919_191974


namespace right_triangle_inequality_l1919_191934

-- Definition of a right-angled triangle with given legs a, b, hypotenuse c, and altitude h_c to the hypotenuse
variables {a b c h_c : ℝ}

-- Right-angled triangle condition definition with angle at C is right
def right_angled_triangle (a b c : ℝ) : Prop :=
  ∃ (a b c : ℝ), c^2 = a^2 + b^2

-- Definition of the altitude to the hypotenuse
def altitude_to_hypotenuse (a b c h_c : ℝ) : Prop :=
  h_c = (a * b) / c

-- Theorem statement to prove the inequality for any right-angled triangle
theorem right_triangle_inequality (a b c h_c : ℝ) (h1 : right_angled_triangle a b c) (h2 : altitude_to_hypotenuse a b c h_c) : 
  a + b < c + h_c :=
by
  sorry

end right_triangle_inequality_l1919_191934


namespace boxes_neither_pens_nor_pencils_l1919_191987

def total_boxes : ℕ := 10
def pencil_boxes : ℕ := 6
def pen_boxes : ℕ := 3
def both_boxes : ℕ := 2

theorem boxes_neither_pens_nor_pencils : (total_boxes - (pencil_boxes + pen_boxes - both_boxes)) = 3 :=
by
  sorry

end boxes_neither_pens_nor_pencils_l1919_191987


namespace binomial_square_value_l1919_191907

theorem binomial_square_value (c : ℝ) : (∃ d : ℝ, 16 * x^2 + 40 * x + c = (4 * x + d) ^ 2) → c = 25 :=
by
  sorry

end binomial_square_value_l1919_191907


namespace expand_product_l1919_191905

theorem expand_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7 * x^3 + 12 := 
  sorry

end expand_product_l1919_191905


namespace divisible_by_120_l1919_191975

theorem divisible_by_120 (n : ℤ) : 120 ∣ (n ^ 6 + 2 * n ^ 5 - n ^ 2 - 2 * n) :=
by sorry

end divisible_by_120_l1919_191975


namespace find_a_l1919_191986

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l1919_191986


namespace find_two_numbers_l1919_191999

noncomputable def quadratic_roots (a b : ℝ) : Prop :=
  a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2

theorem find_two_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : 2 * (a * b) / (a + b) = 5 / 2) :
  quadratic_roots a b :=
by
  sorry

end find_two_numbers_l1919_191999


namespace poker_cards_count_l1919_191985

theorem poker_cards_count (total_cards kept_away : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : kept_away = 7) : 
  total_cards - kept_away = 45 :=
by 
  sorry

end poker_cards_count_l1919_191985


namespace john_change_proof_l1919_191922

def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

def cost_of_candy_bar : ℕ := 131
def quarters_paid : ℕ := 4
def dimes_paid : ℕ := 3
def nickels_paid : ℕ := 1

def total_payment : ℕ := (quarters_paid * quarter_value) + (dimes_paid * dime_value) + (nickels_paid * nickel_value)
def change_received : ℕ := total_payment - cost_of_candy_bar

theorem john_change_proof : change_received = 4 :=
by
  -- Proof will be provided here
  sorry

end john_change_proof_l1919_191922


namespace number_of_elements_l1919_191943

def average_incorrect (N : ℕ) := 21
def correction (incorrect : ℕ) (correct : ℕ) := correct - incorrect
def average_correct (N : ℕ) := 22

theorem number_of_elements (N : ℕ) (incorrect : ℕ) (correct : ℕ) :
  average_incorrect N = 21 ∧ incorrect = 26 ∧ correct = 36 ∧ average_correct N = 22 →
  N = 10 :=
by
  sorry

end number_of_elements_l1919_191943


namespace additional_hours_to_travel_l1919_191911

theorem additional_hours_to_travel (distance1 time1 rate distance2 : ℝ)
  (H1 : distance1 = 360)
  (H2 : time1 = 3)
  (H3 : rate = distance1 / time1)
  (H4 : distance2 = 240)
  :
  distance2 / rate = 2 := 
sorry

end additional_hours_to_travel_l1919_191911


namespace johns_earnings_without_bonus_l1919_191926
-- Import the Mathlib library to access all necessary functions and definitions

-- Define the conditions of the problem
def hours_without_bonus : ℕ := 8
def bonus_amount : ℕ := 20
def extra_hours_for_bonus : ℕ := 2
def hours_with_bonus : ℕ := hours_without_bonus + extra_hours_for_bonus
def hourly_wage_with_bonus : ℕ := 10

-- Define the total earnings with the performance bonus
def total_earnings_with_bonus : ℕ := hours_with_bonus * hourly_wage_with_bonus

-- Statement to prove the earnings without the bonus
theorem johns_earnings_without_bonus :
  total_earnings_with_bonus - bonus_amount = 80 :=
by
  -- Placeholder for the proof
  sorry

end johns_earnings_without_bonus_l1919_191926


namespace total_selling_price_l1919_191964

def original_price : ℝ := 120
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.15

def sale_price (original_price discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent)

def final_price (sale_price tax_percent : ℝ) : ℝ :=
  sale_price * (1 + tax_percent)

theorem total_selling_price :
  final_price (sale_price original_price discount_percent) tax_percent = 96.6 :=
sorry

end total_selling_price_l1919_191964


namespace tan_alpha_expression_value_l1919_191962

-- (I) Prove that tan(α) = 4/3 under the given conditions
theorem tan_alpha (O A B C P : ℝ × ℝ) (α : ℝ)
  (hO : O = (0, 0))
  (hA : A = (Real.sin α, 1))
  (hB : B = (Real.cos α, 0))
  (hC : C = (-Real.sin α, 2))
  (hP : P = (2 * Real.cos α - Real.sin α, 1))
  (h_collinear : ∃ t : ℝ, C = t • (P.1, P.2)) :
  Real.tan α = 4 / 3 := sorry

-- (II) Prove the given expression under the condition tan(α) = 4/3
theorem expression_value (α : ℝ)
  (h_tan : Real.tan α = 4 / 3) :
  (Real.sin (2 * α) + Real.sin α) / (2 * Real.cos (2 * α) + 2 * Real.sin α * Real.sin α + Real.cos α) + Real.sin (2 * α) = 
  172 / 75 := sorry

end tan_alpha_expression_value_l1919_191962


namespace binom_subtract_l1919_191991

theorem binom_subtract :
  (Nat.choose 7 4) - 5 = 30 :=
by
  -- proof goes here
  sorry

end binom_subtract_l1919_191991


namespace cordelia_bleach_time_l1919_191976

theorem cordelia_bleach_time (B D : ℕ) (h1 : B + D = 9) (h2 : D = 2 * B) : B = 3 :=
by
  sorry

end cordelia_bleach_time_l1919_191976


namespace exponent_subtraction_l1919_191989

theorem exponent_subtraction (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^n = 2) : a^(m - n) = 3 := by
  sorry

end exponent_subtraction_l1919_191989


namespace parametric_graph_right_half_circle_l1919_191996

theorem parametric_graph_right_half_circle (θ : ℝ) (x y : ℝ) (hx : x = 3 * Real.cos θ) (hy : y = 3 * Real.sin θ) (hθ : -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2) :
  x^2 + y^2 = 9 ∧ x ≥ 0 :=
by
  sorry

end parametric_graph_right_half_circle_l1919_191996


namespace velocity_zero_times_l1919_191921

noncomputable def s (t : ℝ) : ℝ := (1 / 4) * t^4 - (5 / 3) * t^3 + 2 * t^2

theorem velocity_zero_times :
  {t : ℝ | deriv s t = 0} = {0, 1, 4} :=
by 
  sorry

end velocity_zero_times_l1919_191921


namespace minFuseLength_l1919_191939

namespace EarthquakeRelief

def fuseLengthRequired (distanceToSafety : ℕ) (speedOperator : ℕ) (burningSpeed : ℕ) (lengthFuse : ℕ) : Prop :=
  (lengthFuse : ℝ) / (burningSpeed : ℝ) > (distanceToSafety : ℝ) / (speedOperator : ℝ)

theorem minFuseLength 
  (distanceToSafety : ℕ := 400) 
  (speedOperator : ℕ := 5) 
  (burningSpeed : ℕ := 12) : 
  ∀ lengthFuse: ℕ, 
  fuseLengthRequired distanceToSafety speedOperator burningSpeed lengthFuse → lengthFuse > 96 := 
by
  sorry

end EarthquakeRelief

end minFuseLength_l1919_191939


namespace winter_spending_l1919_191901

-- Define the total spending by the end of November
def total_spending_end_november : ℝ := 3.3

-- Define the total spending by the end of February
def total_spending_end_february : ℝ := 7.0

-- Formalize the problem: prove that the spending during December, January, and February is 3.7 million dollars
theorem winter_spending : total_spending_end_february - total_spending_end_november = 3.7 := by
  sorry

end winter_spending_l1919_191901


namespace least_sum_exponents_of_520_l1919_191969

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l1919_191969


namespace solve_fraction_l1919_191940

open Real

theorem solve_fraction (x : ℝ) (hx : 1 - 4 / x + 4 / x^2 = 0) : 2 / x = 1 :=
by
  -- We'll include the necessary steps of the proof here, but for now we leave it as sorry.
  sorry

end solve_fraction_l1919_191940


namespace lori_earnings_l1919_191977

theorem lori_earnings
    (red_cars : ℕ)
    (white_cars : ℕ)
    (cost_red_car : ℕ)
    (cost_white_car : ℕ)
    (rental_time_hours : ℕ)
    (rental_time_minutes : ℕ)
    (correct_earnings : ℕ) :
    red_cars = 3 →
    white_cars = 2 →
    cost_red_car = 3 →
    cost_white_car = 2 →
    rental_time_hours = 3 →
    rental_time_minutes = rental_time_hours * 60 →
    correct_earnings = 2340 →
    (red_cars * cost_red_car + white_cars * cost_white_car) * rental_time_minutes = correct_earnings :=
by
  intros
  sorry

end lori_earnings_l1919_191977


namespace inequality1_solution_inequality2_solution_l1919_191929

-- Definitions for the conditions
def cond1 (x : ℝ) : Prop := abs (1 - (2 * x - 1) / 3) ≤ 2
def cond2 (x : ℝ) : Prop := (2 - x) * (x + 3) < 2 - x

-- Lean 4 statement for the proof problem
theorem inequality1_solution (x : ℝ) : cond1 x → -1 ≤ x ∧ x ≤ 5 := by
  sorry

theorem inequality2_solution (x : ℝ) : cond2 x → x > 2 ∨ x < -2 := by
  sorry

end inequality1_solution_inequality2_solution_l1919_191929


namespace only_solution_l1919_191946

theorem only_solution (x : ℝ) : (3 / (x - 3) = 5 / (x - 5)) ↔ (x = 0) := 
sorry

end only_solution_l1919_191946


namespace maximum_value_of_function_l1919_191937

theorem maximum_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ M, (∀ y, y = x * (1 - 2 * x) → y ≤ M) ∧ M = 1/8 :=
sorry

end maximum_value_of_function_l1919_191937


namespace find_x_l1919_191959

theorem find_x 
  (AB AC BC : ℝ) 
  (x : ℝ)
  (hO : π * (AB / 2)^2 = 12 + 2 * x)
  (hP : π * (AC / 2)^2 = 24 + x)
  (hQ : π * (BC / 2)^2 = 108 - x)
  : AC^2 + BC^2 = AB^2 → x = 60 :=
by {
   sorry
}

end find_x_l1919_191959


namespace fractional_inequality_solution_set_l1919_191961

theorem fractional_inequality_solution_set (x : ℝ) :
  (x / (x + 1) < 0) ↔ (-1 < x) ∧ (x < 0) :=
sorry

end fractional_inequality_solution_set_l1919_191961


namespace min_students_wearing_both_glasses_and_watches_l1919_191948

theorem min_students_wearing_both_glasses_and_watches
  (n : ℕ)
  (H_glasses : n * 3 / 5 = 18)
  (H_watches : n * 5 / 6 = 25)
  (H_neither : n * 1 / 10 = 3):
  ∃ (x : ℕ), x = 16 := 
by
  sorry

end min_students_wearing_both_glasses_and_watches_l1919_191948


namespace solve_sausage_problem_l1919_191990

def sausage_problem (x y : ℕ) (condition1 : y = x + 300) (condition2 : x = y + 500) : Prop :=
  x + y = 2 * 400

theorem solve_sausage_problem (x y : ℕ) (h1 : y = x + 300) (h2 : x = y + 500) :
  sausage_problem x y h1 h2 :=
by
  sorry

end solve_sausage_problem_l1919_191990


namespace find_equation_of_line_l1919_191942

theorem find_equation_of_line
  (m b : ℝ) 
  (h1 : ∃ k : ℝ, (k^2 - 2*k + 3 = k*m + b ∧ ∃ d : ℝ, d = 4) 
        ∧ (4*m - k^2 + 2*m*k - 3 + b = 0)) 
  (h2 : 8 = 2*m + b)
  (h3 : b ≠ 0) 
  : y = 8 :=
by 
  sorry

end find_equation_of_line_l1919_191942


namespace complementary_angles_difference_l1919_191910

theorem complementary_angles_difference :
  ∃ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 90 ∧ 5 * θ₁ = 3 * θ₂ ∧ abs (θ₁ - θ₂) = 22.5 :=
by
  sorry

end complementary_angles_difference_l1919_191910


namespace number_of_people_l1919_191997

theorem number_of_people (x : ℕ) (H : x * (x - 1) = 72) : x = 9 :=
sorry

end number_of_people_l1919_191997


namespace find_integer_pairs_l1919_191920

theorem find_integer_pairs (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) :=
by
  sorry

end find_integer_pairs_l1919_191920


namespace age_difference_l1919_191980

theorem age_difference (P M Mo : ℕ)
  (h1 : P = 3 * M / 5)
  (h2 : Mo = 5 * M / 3)
  (h3 : P + M + Mo = 196) :
  Mo - P = 64 := 
sorry

end age_difference_l1919_191980


namespace complement_union_complement_intersection_l1919_191954

open Set

noncomputable def universal_set : Set ℝ := univ

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem complement_union :
  compl (A ∪ B) = {x : ℝ | x ≤ 2 ∨ 7 ≤ x} := by
  sorry

theorem complement_intersection :
  (compl A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end complement_union_complement_intersection_l1919_191954


namespace plates_arrangement_l1919_191941

theorem plates_arrangement : 
  let blue := 6
  let red := 3
  let green := 2
  let yellow := 1
  let total_ways_without_rest := Nat.factorial (blue + red + green + yellow - 1) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)
  let green_adj_ways := Nat.factorial (blue + red + green + yellow - 2) / (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * Nat.factorial yellow)
  total_ways_without_rest - green_adj_ways = 22680 
:= sorry

end plates_arrangement_l1919_191941


namespace div_product_four_consecutive_integers_l1919_191947

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l1919_191947


namespace mr_klinker_twice_as_old_l1919_191931

theorem mr_klinker_twice_as_old (x : ℕ) (current_age_klinker : ℕ) (current_age_daughter : ℕ)
  (h1 : current_age_klinker = 35) (h2 : current_age_daughter = 10) 
  (h3 : current_age_klinker + x = 2 * (current_age_daughter + x)) : 
  x = 15 :=
by 
  -- We include sorry to indicate where the proof should be
  sorry

end mr_klinker_twice_as_old_l1919_191931


namespace solution_set_a_eq_half_l1919_191916

theorem solution_set_a_eq_half (a : ℝ) : (∀ x : ℝ, (ax / (x - 1) < 1 ↔ (x < 1 ∨ x > 2))) → a = 1 / 2 :=
by
sorry

end solution_set_a_eq_half_l1919_191916


namespace class_6_1_students_l1919_191992

noncomputable def number_of_students : ℕ :=
  let n := 30
  n

theorem class_6_1_students (n : ℕ) (t : ℕ) (h1 : (n + 1) * t = 527) (h2 : n % 5 = 0) : n = 30 :=
  by
  sorry

end class_6_1_students_l1919_191992


namespace bus_distance_time_relation_l1919_191949

theorem bus_distance_time_relation (t : ℝ) :
    (0 ≤ t ∧ t ≤ 1 → s = 60 * t) ∧
    (1 < t ∧ t ≤ 1.5 → s = 60) ∧
    (1.5 < t ∧ t ≤ 2.5 → s = 80 * (t - 1.5) + 60) :=
sorry

end bus_distance_time_relation_l1919_191949


namespace solve_fraction_eq_l1919_191933

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4 / 3 :=
by 
  sorry

end solve_fraction_eq_l1919_191933


namespace find_min_value_l1919_191935

-- Define a structure to represent vectors in 2D space
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

-- Define the dot product of two vectors
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the condition for perpendicular vectors (dot product is zero)
def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

-- Define the problem: given vectors a = (m, 1) and b = (1, n - 2)
-- with conditions m > 0, n > 0, and a ⊥ b, then prove the minimum value of 1/m + 2/n
theorem find_min_value (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0)
  (h₂ : perpendicular ⟨m, 1⟩ ⟨1, n - 2⟩) :
  (1 / m + 2 / n) = (3 + 2 * Real.sqrt 2) / 2 :=
  sorry

end find_min_value_l1919_191935


namespace smallest_sum_l1919_191953

theorem smallest_sum (x y : ℕ) (h : (2010 / 2011 : ℚ) < x / y ∧ x / y < (2011 / 2012 : ℚ)) : x + y = 8044 :=
sorry

end smallest_sum_l1919_191953


namespace profit_calculation_l1919_191913

theorem profit_calculation (cost_price_per_card_yuan : ℚ) (total_sales_yuan : ℚ)
  (n : ℕ) (sales_price_per_card_yuan : ℚ)
  (h1 : cost_price_per_card_yuan = 0.21)
  (h2 : total_sales_yuan = 14.57)
  (h3 : total_sales_yuan = n * sales_price_per_card_yuan)
  (h4 : sales_price_per_card_yuan ≤ 2 * cost_price_per_card_yuan) :
  (total_sales_yuan - n * cost_price_per_card_yuan = 4.7) :=
by
  sorry

end profit_calculation_l1919_191913


namespace fruit_baskets_l1919_191993

def apple_choices := 8 -- From 0 to 7 apples
def orange_choices := 13 -- From 0 to 12 oranges

theorem fruit_baskets (a : ℕ) (o : ℕ) (ha : a = 7) (ho : o = 12) :
  (apple_choices * orange_choices) - 1 = 103 := by
  sorry

end fruit_baskets_l1919_191993


namespace arithmetic_geometric_sequences_l1919_191960

noncomputable def geometric_sequence_sum (a q n : ℝ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequences (a : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  S 5 = geometric_sequence_sum a q 5 →
  2 * a * q = 6 + a * q^4 →
  S 5 = -31 / 2 :=
by
  intros hq1 hS5 hAR
  sorry

end arithmetic_geometric_sequences_l1919_191960


namespace height_relationship_l1919_191967

theorem height_relationship (B V G : ℝ) (h1 : B = 2 * V) (h2 : V = (2 / 3) * G) : B = (4 / 3) * G :=
sorry

end height_relationship_l1919_191967


namespace fraction_eq_l1919_191945

def f(x : ℤ) : ℤ := 3 * x + 2
def g(x : ℤ) : ℤ := 2 * x - 3

theorem fraction_eq : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by 
  sorry

end fraction_eq_l1919_191945
