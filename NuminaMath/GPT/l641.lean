import Mathlib

namespace find_special_numbers_l641_64189

theorem find_special_numbers :
  {N : ℕ | ∃ k m a, N = m + 10^k * a ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ k ∧ m < 10^k 
                ∧ ¬(N % 10 = 0) 
                ∧ (N = 6 * (m + 10^(k+1) * (0 : ℕ))) } = {12, 24, 36, 48} := 
by sorry

end find_special_numbers_l641_64189


namespace find_value_a_prove_inequality_l641_64197

noncomputable def arithmetic_sequence (a : ℕ) (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 2 → S n * S n = 3 * n ^ 2 * a_n n + S (n - 1) * S (n - 1) ∧ a_n n ≠ 0

theorem find_value_a {S : ℕ → ℕ} {a_n : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) → a = 3 :=
sorry

noncomputable def sequence_bn (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  ∀ n : ℕ, b_n n = 1 / ((a_n n - 1) * (a_n n + 2))

theorem prove_inequality {S : ℕ → ℕ} {a_n : ℕ → ℕ} {b_n : ℕ → ℕ} {T : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) →
  (sequence_bn a_n b_n) →
  ∀ n : ℕ, T n < 1 / 6 :=
sorry

end find_value_a_prove_inequality_l641_64197


namespace molecular_weight_proof_l641_64139

/-- Atomic weights in atomic mass units (amu) --/
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_P : ℝ := 30.97

/-- Number of atoms in the compound --/
def num_Al : ℝ := 2
def num_O : ℝ := 4
def num_H : ℝ := 6
def num_N : ℝ := 3
def num_P : ℝ := 1

/-- calculating the molecular weight --/
def molecular_weight : ℝ := 
  (num_Al * atomic_weight_Al) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_N * atomic_weight_N) +
  (num_P * atomic_weight_P)

-- The proof statement
theorem molecular_weight_proof : molecular_weight = 197.02 := 
by
  sorry

end molecular_weight_proof_l641_64139


namespace distance_ratio_l641_64163

theorem distance_ratio (x : ℝ) (hx : abs x = 8) : abs (-4) / abs x = 1 / 2 :=
by {
  sorry
}

end distance_ratio_l641_64163


namespace fouad_double_ahmed_l641_64111

/-- Proof that in 4 years, Fouad's age will be double of Ahmed's age given their current ages. -/
theorem fouad_double_ahmed (x : ℕ) (ahmed_age fouad_age : ℕ) (h1 : ahmed_age = 11) (h2 : fouad_age = 26) :
  (fouad_age + x = 2 * (ahmed_age + x)) → x = 4 :=
by
  -- This is the statement only, proof is omitted
  sorry

end fouad_double_ahmed_l641_64111


namespace cubes_closed_under_multiplication_l641_64105

-- Define the set of cubes of positive integers
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define the multiplication operation on the set of cubes
def cube_mult_closed : Prop :=
  ∀ x y : ℕ, is_cube x → is_cube y → is_cube (x * y)

-- The statement we want to prove
theorem cubes_closed_under_multiplication : cube_mult_closed :=
sorry

end cubes_closed_under_multiplication_l641_64105


namespace last_digit_sum_l641_64165

theorem last_digit_sum :
  (2^2 % 10 + 20^20 % 10 + 200^200 % 10 + 2006^2006 % 10) % 10 = 0 := 
by
  sorry

end last_digit_sum_l641_64165


namespace student_knows_german_l641_64100

-- Definitions for each classmate's statement
def classmate1 (lang: String) : Prop := lang ≠ "French"
def classmate2 (lang: String) : Prop := lang = "Spanish" ∨ lang = "German"
def classmate3 (lang: String) : Prop := lang = "Spanish"

-- Conditions: at least one correct and at least one incorrect
def at_least_one_correct (lang: String) : Prop :=
  classmate1 lang ∨ classmate2 lang ∨ classmate3 lang

def at_least_one_incorrect (lang: String) : Prop :=
  ¬classmate1 lang ∨ ¬classmate2 lang ∨ ¬classmate3 lang

-- The statement to prove
theorem student_knows_german : ∀ lang : String,
  at_least_one_correct lang → at_least_one_incorrect lang → lang = "German" :=
by
  intros lang Hcorrect Hincorrect
  revert Hcorrect Hincorrect
  -- sorry stands in place of direct proof
  sorry

end student_knows_german_l641_64100


namespace problem_l641_64159

theorem problem (a b : ℤ) (h : (2 * a + b) ^ 2 + |b - 2| = 0) : (-a - b) ^ 2014 = 1 := 
by
  sorry

end problem_l641_64159


namespace sum_of_integers_is_96_l641_64185

theorem sum_of_integers_is_96 (x y : ℤ) (h1 : x = 32) (h2 : y = 2 * x) : x + y = 96 := 
by
  sorry

end sum_of_integers_is_96_l641_64185


namespace height_of_triangle_is_5_l641_64169

def base : ℝ := 4
def area : ℝ := 10

theorem height_of_triangle_is_5 :
  ∃ (height : ℝ), (base * height) / 2 = area ∧ height = 5 :=
by
  sorry

end height_of_triangle_is_5_l641_64169


namespace parkway_elementary_students_l641_64192

/-- The total number of students in the fifth grade at Parkway Elementary School is 420,
given the following conditions:
1. There are 312 boys.
2. 250 students are playing soccer.
3. 78% of the students that play soccer are boys.
4. There are 53 girl students not playing soccer. -/
theorem parkway_elementary_students (boys : ℕ) (playing_soccer : ℕ) (percent_boys_playing : ℝ) (girls_not_playing_soccer : ℕ)
  (h1 : boys = 312)
  (h2 : playing_soccer = 250)
  (h3 : percent_boys_playing = 0.78)
  (h4 : girls_not_playing_soccer = 53) :
  ∃ total_students : ℕ, total_students = 420 :=
by
  sorry

end parkway_elementary_students_l641_64192


namespace ice_cream_melting_l641_64130

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end ice_cream_melting_l641_64130


namespace part1_part2_l641_64184

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x > -1, (x^2 + 3*x + 6) / (x + 1) ≥ a) ↔ (a ≤ 5) := 
  sorry

-- Part 2
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) : 
  2*a + (1/a) + 4*b + (8/b) ≥ 27 :=
  sorry

end part1_part2_l641_64184


namespace total_interval_length_l641_64132

noncomputable def interval_length : ℝ :=
  1 / (1 + 2^Real.pi)

theorem total_interval_length :
  ∀ x : ℝ, x < 1 ∧ Real.tan (Real.log x / Real.log 4) > 0 →
  (∃ y, interval_length = y) :=
by
  sorry

end total_interval_length_l641_64132


namespace percentage_invalid_votes_l641_64133

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l641_64133


namespace max_possible_ratio_squared_l641_64190

noncomputable def maxRatioSquared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : ℝ :=
  2

theorem max_possible_ratio_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : maxRatioSquared a b h1 h2 h3 h4 = 2 :=
sorry

end max_possible_ratio_squared_l641_64190


namespace arithmetic_sequence_find_m_l641_64115

theorem arithmetic_sequence_find_m (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_find_m_l641_64115


namespace find_m_n_l641_64122

def is_prime (n : Nat) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem find_m_n (p k : ℕ) (hk : 1 < k) (hp : is_prime p) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ (m^p + n^p) / 2 = (m + n) / 2 ^ k) ↔ k = p :=
sorry

end find_m_n_l641_64122


namespace pints_in_vat_l641_64158

-- Conditions
def num_glasses : Nat := 5
def pints_per_glass : Nat := 30

-- Problem statement: prove that the total number of pints in the vat is 150
theorem pints_in_vat : num_glasses * pints_per_glass = 150 :=
by
  -- Proof goes here
  sorry

end pints_in_vat_l641_64158


namespace problem_statement_l641_64178

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem problem_statement (S : ℝ) (h1 : S = golden_ratio) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 :=
by
  sorry

end problem_statement_l641_64178


namespace find_prices_l641_64119

def price_system_of_equations (x y : ℕ) : Prop :=
  3 * x + 2 * y = 474 ∧ x - y = 8

theorem find_prices (x y : ℕ) :
  price_system_of_equations x y :=
by
  sorry

end find_prices_l641_64119


namespace sequence_general_term_l641_64166

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1) :
  (∀ n, a n = if n = 1 then 2 else 2 * n - 1) :=
by
  sorry

end sequence_general_term_l641_64166


namespace time_per_harvest_is_three_months_l641_64113

variable (area : ℕ) (trees_per_m2 : ℕ) (coconuts_per_tree : ℕ) 
variable (price_per_coconut : ℚ) (total_earning_6_months : ℚ)

theorem time_per_harvest_is_three_months 
  (h1 : area = 20) 
  (h2 : trees_per_m2 = 2) 
  (h3 : coconuts_per_tree = 6) 
  (h4 : price_per_coconut = 0.50) 
  (h5 : total_earning_6_months = 240) :
    (6 / (total_earning_6_months / (area * trees_per_m2 * coconuts_per_tree * price_per_coconut)) = 3) := 
  by 
    sorry

end time_per_harvest_is_three_months_l641_64113


namespace shopkeeper_loss_percent_l641_64121

theorem shopkeeper_loss_percent (I : ℝ) (h1 : I > 0) : 
  (0.1 * (I - 0.4 * I)) = 0.4 * (1.1 * I) :=
by
  -- proof goes here
  sorry

end shopkeeper_loss_percent_l641_64121


namespace teacher_buys_total_21_pens_l641_64118

def num_black_pens : Nat := 7
def num_blue_pens : Nat := 9
def num_red_pens : Nat := 5
def total_pens : Nat := num_black_pens + num_blue_pens + num_red_pens

theorem teacher_buys_total_21_pens : total_pens = 21 := 
by
  unfold total_pens num_black_pens num_blue_pens num_red_pens
  rfl -- reflexivity (21 = 21)

end teacher_buys_total_21_pens_l641_64118


namespace olly_needs_24_shoes_l641_64144

-- Define the number of paws for different types of pets
def dogs : ℕ := 3
def cats : ℕ := 2
def ferret : ℕ := 1

def paws_per_dog : ℕ := 4
def paws_per_cat : ℕ := 4
def paws_per_ferret : ℕ := 4

-- The theorem we want to prove
theorem olly_needs_24_shoes : 
  dogs * paws_per_dog + cats * paws_per_cat + ferret * paws_per_ferret = 24 :=
by
  sorry

end olly_needs_24_shoes_l641_64144


namespace multiply_digits_correctness_l641_64112

theorem multiply_digits_correctness (a b c : ℕ) :
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c :=
by sorry

end multiply_digits_correctness_l641_64112


namespace equation_of_circle_given_diameter_l641_64162

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem equation_of_circle_given_diameter :
  ∀ (A B : ℝ × ℝ), A = (-3,0) → B = (1,0) → 
  (∃ (x y : ℝ), is_on_circle (-1, 0) 2 (x, y)) ↔ (x + 1)^2 + y^2 = 4 :=
by
  sorry

end equation_of_circle_given_diameter_l641_64162


namespace algebraic_expression_l641_64179

def ast (n : ℕ) : ℕ := sorry

axiom condition_1 : ast 1 = 1
axiom condition_2 : ∀ (n : ℕ), ast (n + 1) = 3 * ast n

theorem algebraic_expression (n : ℕ) :
  n > 0 → ast n = 3^(n - 1) :=
by
  -- Proof to be completed
  sorry

end algebraic_expression_l641_64179


namespace expected_pourings_correct_l641_64120

section
  /-- Four glasses are arranged in a row: the first and third contain orange juice, 
      the second and fourth are empty. Valya can take a full glass and pour its 
      contents into one of the two empty glasses each time. -/
  def initial_state : List Bool := [true, false, true, false]
  def target_state : List Bool := [false, true, false, true]

  /-- Define a function to calculate the expected number of pourings required to 
      reach the target state from the initial state given the probabilities of 
      transitions. -/
  noncomputable def expected_number_of_pourings (init : List Bool) (target : List Bool) : ℕ :=
    if init = initial_state ∧ target = target_state then 6 else 0

  /-- Prove that the expected number of pourings required to transition from 
      the initial state [true, false, true, false] to the target state [false, true, false, true] is 6. -/
  theorem expected_pourings_correct :
    expected_number_of_pourings initial_state target_state = 6 :=
  by
    -- Proof omitted
    sorry
end

end expected_pourings_correct_l641_64120


namespace cos_pi_six_plus_alpha_l641_64150

variable (α : ℝ)

theorem cos_pi_six_plus_alpha (h : Real.sin (Real.pi / 3 - α) = 1 / 6) : 
  Real.cos (Real.pi / 6 + α) = 1 / 6 :=
sorry

end cos_pi_six_plus_alpha_l641_64150


namespace sum_of_consecutive_pages_l641_64102

theorem sum_of_consecutive_pages (n : ℕ) 
  (h : n * (n + 1) = 20412) : n + (n + 1) + (n + 2) = 429 := by
  sorry

end sum_of_consecutive_pages_l641_64102


namespace min_value_of_quadratic_l641_64191

theorem min_value_of_quadratic (x : ℝ) : ∃ z : ℝ, z = 2 * x^2 + 16 * x + 40 ∧ z = 8 :=
by {
  sorry
}

end min_value_of_quadratic_l641_64191


namespace greatest_int_less_than_200_with_gcd_18_eq_9_l641_64101

theorem greatest_int_less_than_200_with_gcd_18_eq_9 :
  ∃ n, n < 200 ∧ Int.gcd n 18 = 9 ∧ ∀ m, m < 200 ∧ Int.gcd m 18 = 9 → m ≤ n :=
sorry

end greatest_int_less_than_200_with_gcd_18_eq_9_l641_64101


namespace melanie_total_plums_l641_64140

namespace Melanie

def initial_plums : ℝ := 7.0
def plums_given_by_sam : ℝ := 3.0

theorem melanie_total_plums : initial_plums + plums_given_by_sam = 10.0 :=
by
  sorry

end Melanie

end melanie_total_plums_l641_64140


namespace count_valid_ks_l641_64161

theorem count_valid_ks : 
  ∃ (ks : Finset ℕ), (∀ k ∈ ks, k > 0 ∧ k ≤ 50 ∧ 
    ∀ n : ℕ, n > 0 → 7 ∣ (2 * 3^(6 * n) + k * 2^(3 * n + 1) - 1)) ∧ ks.card = 7 :=
sorry

end count_valid_ks_l641_64161


namespace admission_price_for_children_l641_64116

theorem admission_price_for_children 
  (admission_price_adult : ℕ)
  (total_persons : ℕ)
  (total_amount_dollars : ℕ)
  (children_attended : ℕ)
  (admission_price_children : ℕ)
  (h1 : admission_price_adult = 60)
  (h2 : total_persons = 280)
  (h3 : total_amount_dollars = 140)
  (h4 : children_attended = 80)
  (h5 : (total_persons - children_attended) * admission_price_adult + children_attended * admission_price_children = total_amount_dollars * 100)
  : admission_price_children = 25 := 
by 
  sorry

end admission_price_for_children_l641_64116


namespace c_geq_one_l641_64164

theorem c_geq_one (a b : ℕ) (c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : c ≥ 1 :=
by sorry

end c_geq_one_l641_64164


namespace inequality_holds_l641_64180

theorem inequality_holds (a : ℝ) (h : a ≠ 0) : |a + (1/a)| ≥ 2 :=
by
  sorry

end inequality_holds_l641_64180


namespace total_earmuffs_l641_64126

theorem total_earmuffs {a b c : ℕ} (h1 : a = 1346) (h2 : b = 6444) (h3 : c = a + b) : c = 7790 := by
  sorry

end total_earmuffs_l641_64126


namespace max_value_a4_b2_c2_d2_l641_64187

theorem max_value_a4_b2_c2_d2
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  a^4 + b^2 + c^2 + d^2 ≤ 100 :=
sorry

end max_value_a4_b2_c2_d2_l641_64187


namespace count_n_satisfies_conditions_l641_64127

theorem count_n_satisfies_conditions :
  ∃ (count : ℕ), count = 36 ∧ ∀ (n : ℕ), 
    0 < n ∧ n < 150 →
    ∃ (k : ℕ), 
    (n = 2*k + 2) ∧ 
    (k*(k + 2) % 4 = 0) :=
by
  sorry

end count_n_satisfies_conditions_l641_64127


namespace typing_speed_ratio_l641_64129

variable (T M : ℝ)

-- Conditions
def condition1 : Prop := T + M = 12
def condition2 : Prop := T + 1.25 * M = 14

-- Proof statement
theorem typing_speed_ratio (h1 : condition1 T M) (h2 : condition2 T M) : M / T = 2 := by
  sorry

end typing_speed_ratio_l641_64129


namespace chessboard_problem_proof_l641_64135

variable (n : ℕ)

noncomputable def chessboard_problem : Prop :=
  ∀ (colors : Fin (2 * n) → Fin (2 * n) → Fin n),
  ∃ i₁ i₂ j₁ j₂,
    i₁ ≠ i₂ ∧
    j₁ ≠ j₂ ∧
    colors i₁ j₁ = colors i₁ j₂ ∧
    colors i₂ j₁ = colors i₂ j₂

/-- Given a 2n x 2n chessboard colored with n colors, there exist 2 tiles in either the same column 
or row such that if the colors of both tiles are swapped, then there exists a rectangle where all 
its four corner tiles have the same color. -/
theorem chessboard_problem_proof (n : ℕ) : chessboard_problem n :=
sorry

end chessboard_problem_proof_l641_64135


namespace pear_distribution_problem_l641_64155

-- Defining the given conditions as hypotheses
variables (G P : ℕ)

-- The first condition: P = G + 1
def condition1 : Prop := P = G + 1

-- The second condition: P = 2G - 2
def condition2 : Prop := P = 2 * G - 2

-- The main theorem to prove
theorem pear_distribution_problem (h1 : condition1 G P) (h2 : condition2 G P) :
  G = 3 ∧ P = 4 :=
by
  sorry

end pear_distribution_problem_l641_64155


namespace total_count_pens_pencils_markers_l641_64148

-- Define the conditions
def ratio_pens_pencils (pens pencils : ℕ) : Prop :=
  6 * pens = 5 * pencils

def nine_more_pencils (pens pencils : ℕ) : Prop :=
  pencils = pens + 9

def ratio_markers_pencils (markers pencils : ℕ) : Prop :=
  3 * markers = 4 * pencils

-- Theorem statement to be proved 
theorem total_count_pens_pencils_markers 
  (pens pencils markers : ℕ) 
  (h1 : ratio_pens_pencils pens pencils)
  (h2 : nine_more_pencils pens pencils)
  (h3 : ratio_markers_pencils markers pencils) : 
  pens + pencils + markers = 171 :=
sorry

end total_count_pens_pencils_markers_l641_64148


namespace sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l641_64199

open Real

namespace TriangleProofs

variables 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (BA BC : ℝ) 
  (h1 : sin B = sqrt 7 / 4) 
  (h2 : (cos A / sin A + cos C / sin C = 4 * sqrt 7 / 7)) 
  (h3 : BA * BC = 3 / 2)
  (h4 : a = b ∧ c = b)

-- 1. Prove that sin A * sin C = sin^2 B
theorem sin_a_mul_sin_c_eq_sin_sq_b : sin A * sin C = sin B ^ 2 := 
by sorry

-- 2. Prove that 0 < B ≤ π / 3
theorem zero_lt_B_le_pi_div_3 : 0 < B ∧ B ≤ π / 3 := 
by sorry

-- 3. Find the magnitude of the vector sum.
theorem magnitude_BC_add_BA : abs (BC + BA) = 2 * sqrt 2 := 
by sorry

end TriangleProofs

end sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l641_64199


namespace find_triplets_l641_64177

noncomputable def phi (t : ℝ) : ℝ := 2 * t^3 + t - 2

theorem find_triplets (x y z : ℝ) (h1 : x^5 = phi y) (h2 : y^5 = phi z) (h3 : z^5 = phi x) :
  ∃ r : ℝ, (x = r ∧ y = r ∧ z = r) ∧ (r^5 = phi r) :=
by
  sorry

end find_triplets_l641_64177


namespace intersection_is_1_l641_64147

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {y | ∃ x ∈ M, y = x ^ 2}
theorem intersection_is_1 : M ∩ N = {1} := by
  sorry

end intersection_is_1_l641_64147


namespace perpendicular_lines_m_l641_64183

theorem perpendicular_lines_m (m : ℝ) :
  (∀ (x y : ℝ), x - 2 * y + 5 = 0 → 2 * x + m * y - 6 = 0) →
  m = 1 :=
by
  sorry

end perpendicular_lines_m_l641_64183


namespace ellipse_polar_inverse_sum_l641_64175

noncomputable def ellipse_equation (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

theorem ellipse_polar_inverse_sum (A B : ℝ × ℝ)
  (hA : ∃ α₁, ellipse_equation α₁ = A)
  (hB : ∃ α₂, ellipse_equation α₂ = B)
  (hPerp : A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / (A.1 ^ 2 + A.2 ^ 2) + 1 / (B.1 ^ 2 + B.2 ^ 2)) = 7 / 12 :=
by
  sorry

end ellipse_polar_inverse_sum_l641_64175


namespace maximal_area_of_AMNQ_l641_64136

theorem maximal_area_of_AMNQ (s q : ℝ) (Hq1 : 0 ≤ q) (Hq2 : q ≤ s) :
  let Q := (s, q)
  ∃ M N : ℝ × ℝ, 
    (M.1 ∈ [0,s] ∧ M.2 = 0) ∧ 
    (N.1 = s ∧ N.2 ∈ [0,s]) ∧ 
    if q ≤ (2/3) * s 
    then 
      (M.1 * M.2 / 2 = (CQ/2)) 
    else 
      (N = (s, s)) :=
by sorry

end maximal_area_of_AMNQ_l641_64136


namespace frequency_of_middle_group_l641_64188

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end frequency_of_middle_group_l641_64188


namespace fruit_basket_apples_oranges_ratio_l641_64154

theorem fruit_basket_apples_oranges_ratio : 
  ∀ (apples oranges : ℕ), 
  apples = 15 ∧ (2 * apples / 3 + 2 * oranges / 3 = 50) → (apples = 15 ∧ oranges = 60) → apples / gcd apples oranges = 1 ∧ oranges / gcd apples oranges = 4 :=
by 
  intros apples oranges h1 h2
  have h_apples : apples = 15 := by exact h2.1
  have h_oranges : oranges = 60 := by exact h2.2
  rw [h_apples, h_oranges]
  sorry

end fruit_basket_apples_oranges_ratio_l641_64154


namespace possible_numbers_tom_l641_64153

theorem possible_numbers_tom (n : ℕ) (h1 : 180 ∣ n) (h2 : 75 ∣ n) (h3 : 500 < n ∧ n < 2500) : n = 900 ∨ n = 1800 :=
sorry

end possible_numbers_tom_l641_64153


namespace journey_total_time_l641_64125

theorem journey_total_time (speed1 time1 speed2 total_distance : ℕ) 
  (h1 : speed1 = 40) 
  (h2 : time1 = 3) 
  (h3 : speed2 = 60) 
  (h4 : total_distance = 240) : 
  time1 + (total_distance - speed1 * time1) / speed2 = 5 := 
by 
  sorry

end journey_total_time_l641_64125


namespace coefficient_of_x_in_first_term_l641_64128

variable {a k n : ℝ} (x : ℝ)

theorem coefficient_of_x_in_first_term (h1 : (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
  (h2 : a - n + k = 7) :
  3 = 3 := 
sorry

end coefficient_of_x_in_first_term_l641_64128


namespace tan_neg_225_is_neg_1_l641_64124

def tan_neg_225_eq_neg_1 : Prop :=
  Real.tan (-225 * Real.pi / 180) = -1

theorem tan_neg_225_is_neg_1 : tan_neg_225_eq_neg_1 :=
  by
    sorry

end tan_neg_225_is_neg_1_l641_64124


namespace necessary_but_not_sufficient_condition_l641_64123

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > e) : x > 1 :=
sorry

end necessary_but_not_sufficient_condition_l641_64123


namespace breadth_of_room_is_6_l641_64194

theorem breadth_of_room_is_6 
(the_room_length : ℝ) 
(the_carpet_width : ℝ) 
(cost_per_meter : ℝ) 
(total_cost : ℝ) 
(h1 : the_room_length = 15) 
(h2 : the_carpet_width = 0.75) 
(h3 : cost_per_meter = 0.30) 
(h4 : total_cost = 36) : 
  ∃ (breadth_of_room : ℝ), breadth_of_room = 6 :=
sorry

end breadth_of_room_is_6_l641_64194


namespace system_has_negative_solution_iff_sum_zero_l641_64195

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l641_64195


namespace second_quadrant_necessary_not_sufficient_l641_64156

variable (α : ℝ) -- Assuming α is a real number for generality.

-- Define what it means for an angle to be in the second quadrant (90° < α < 180°).
def in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

-- Define what it means for an angle to be obtuse (90° < α ≤ 180°).
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α ≤ 180

-- State the theorem to prove: 
-- "The angle α is in the second quadrant" is a necessary but not sufficient condition for "α is an obtuse angle".
theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → in_second_quadrant α) ∧ 
  (∃ α, in_second_quadrant α ∧ ¬is_obtuse α) :=
sorry

end second_quadrant_necessary_not_sufficient_l641_64156


namespace rattlesnake_tail_percentage_difference_l641_64114

-- Definitions for the problem
def eastern_segments : Nat := 6
def western_segments : Nat := 8

-- The statement to prove
theorem rattlesnake_tail_percentage_difference :
  100 * (western_segments - eastern_segments) / western_segments = 25 := by
  sorry

end rattlesnake_tail_percentage_difference_l641_64114


namespace jen_ate_eleven_suckers_l641_64168

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end jen_ate_eleven_suckers_l641_64168


namespace man_l641_64131

theorem man's_age_twice_son (S M Y : ℕ) (h1 : M = S + 26) (h2 : S = 24) (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  sorry

end man_l641_64131


namespace ordered_sum_ways_l641_64170

theorem ordered_sum_ways (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 2) : 
  ∃ (ways : ℕ), ways = 70 :=
by
  sorry

end ordered_sum_ways_l641_64170


namespace solve_equation_l641_64137

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l641_64137


namespace quadratic_pos_implies_a_gt_1_l641_64107

theorem quadratic_pos_implies_a_gt_1 {a : ℝ} :
  (∀ x : ℝ, x^2 + 2 * x + a > 0) → a > 1 :=
by
  sorry

end quadratic_pos_implies_a_gt_1_l641_64107


namespace lines_are_parallel_l641_64146

def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_are_parallel : ∀ x y : ℝ, line1 x = y → line2 x = y → false :=
by
  sorry

end lines_are_parallel_l641_64146


namespace picture_distance_l641_64106

theorem picture_distance (wall_width picture_width x y : ℝ)
  (h_wall : wall_width = 25)
  (h_picture : picture_width = 5)
  (h_relation : x = 2 * y)
  (h_total : x + picture_width + y = wall_width) :
  x = 13.34 :=
by
  sorry

end picture_distance_l641_64106


namespace rectangles_with_one_gray_cell_l641_64151

-- Define the number of gray cells
def gray_cells : ℕ := 40

-- Define the total rectangles containing exactly one gray cell
def total_rectangles : ℕ := 176

-- The theorem we want to prove
theorem rectangles_with_one_gray_cell (h : gray_cells = 40) : total_rectangles = 176 := 
by 
  sorry

end rectangles_with_one_gray_cell_l641_64151


namespace value_of_k_l641_64110

theorem value_of_k (k m : ℝ)
    (h1 : m = k / 3)
    (h2 : 2 = k / (3 * m - 1)) :
    k = 2 := by
  sorry

end value_of_k_l641_64110


namespace minimum_value_l641_64167

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + 3 * b = 1) : 
  26 ≤ (2 / a + 3 / b) :=
sorry

end minimum_value_l641_64167


namespace boy_to_total_ratio_l641_64198

-- Problem Definitions
variables (b g : ℕ) -- number of boys and number of girls

-- Hypothesis: The probability of choosing a boy is (4/5) the probability of choosing a girl
def probability_boy := b / (b + g : ℕ)
def probability_girl := g / (b + g : ℕ)

theorem boy_to_total_ratio (h : probability_boy b g = (4 / 5) * probability_girl b g) : 
  b / (b + g : ℕ) = 4 / 9 :=
sorry

end boy_to_total_ratio_l641_64198


namespace sum_of_parts_l641_64109

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 52) (h2 : y = 30.333333333333332) :
  10 * x + 22 * y = 884 :=
sorry

end sum_of_parts_l641_64109


namespace sum_reciprocal_l641_64141

-- Definition of the problem
theorem sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 4 * x * y) : 
  (1 / x) + (1 / y) = 1 :=
sorry

end sum_reciprocal_l641_64141


namespace mirror_tweet_rate_is_45_l641_64143

-- Defining the conditions given in the problem
def happy_tweet_rate : ℕ := 18
def hungry_tweet_rate : ℕ := 4
def mirror_tweet_rate (x : ℕ) : ℕ := x
def happy_minutes : ℕ := 20
def hungry_minutes : ℕ := 20
def mirror_minutes : ℕ := 20
def total_tweets : ℕ := 1340

-- Proving the rate of tweets when Polly watches herself in the mirror
theorem mirror_tweet_rate_is_45 : mirror_tweet_rate 45 * mirror_minutes = total_tweets - (happy_tweet_rate * happy_minutes + hungry_tweet_rate * hungry_minutes) :=
by 
  sorry

end mirror_tweet_rate_is_45_l641_64143


namespace pow_mod_equiv_l641_64134

theorem pow_mod_equiv (h : 5^500 ≡ 1 [MOD 1250]) : 5^15000 ≡ 1 [MOD 1250] := 
by 
  sorry

end pow_mod_equiv_l641_64134


namespace average_speed_l641_64193

theorem average_speed 
  (total_distance : ℝ) (total_time : ℝ) 
  (h_distance : total_distance = 26) (h_time : total_time = 4) :
  (total_distance / total_time) = 6.5 :=
by
  rw [h_distance, h_time]
  norm_num

end average_speed_l641_64193


namespace silvia_savings_l641_64138

def retail_price : ℝ := 1000
def guitar_center_discount_rate : ℝ := 0.15
def sweetwater_discount_rate : ℝ := 0.10
def guitar_center_shipping_fee : ℝ := 100
def sweetwater_shipping_fee : ℝ := 0

def guitar_center_cost : ℝ := retail_price * (1 - guitar_center_discount_rate) + guitar_center_shipping_fee
def sweetwater_cost : ℝ := retail_price * (1 - sweetwater_discount_rate) + sweetwater_shipping_fee

theorem silvia_savings : guitar_center_cost - sweetwater_cost = 50 := by
  sorry

end silvia_savings_l641_64138


namespace power_of_six_evaluation_l641_64117

noncomputable def example_expr : ℝ := (6 : ℝ)^(1/4) / (6 : ℝ)^(1/6)

theorem power_of_six_evaluation : example_expr = (6 : ℝ)^(1/12) := 
by
  sorry

end power_of_six_evaluation_l641_64117


namespace find_b_l641_64174

-- Define the given hyperbola equation and conditions
def hyperbola (x y : ℝ) (b : ℝ) : Prop := x^2 - y^2 / b^2 = 1
def asymptote_line (x y : ℝ) : Prop := 2 * x - y = 0

-- State the theorem to prove
theorem find_b (b : ℝ) (hb : b > 0) :
    (∀ x y : ℝ, hyperbola x y b → asymptote_line x y) → b = 2 :=
by 
  sorry

end find_b_l641_64174


namespace factor_polynomial_l641_64157

theorem factor_polynomial (a b m n : ℝ) (h : |m - 4| + (n^2 - 8 * n + 16) = 0) :
  a^2 + 4 * b^2 - m * a * b - n = (a - 2 * b + 2) * (a - 2 * b - 2) :=
by
  sorry

end factor_polynomial_l641_64157


namespace yuna_has_most_apples_l641_64160

def apples_count_jungkook : ℕ :=
  6 / 3

def apples_count_yoongi : ℕ :=
  4

def apples_count_yuna : ℕ :=
  5

theorem yuna_has_most_apples : apples_count_yuna > apples_count_yoongi ∧ apples_count_yuna > apples_count_jungkook :=
by
  sorry

end yuna_has_most_apples_l641_64160


namespace angle_supplement_complement_l641_64182

theorem angle_supplement_complement (a : ℝ) (h : 180 - a = 3 * (90 - a)) : a = 45 :=
by
  sorry

end angle_supplement_complement_l641_64182


namespace maximize_village_value_l641_64172

theorem maximize_village_value :
  ∃ (x y z : ℕ), 
  x + y + z = 20 ∧ 
  2 * x + 3 * y + 4 * z = 50 ∧ 
  (∀ x' y' z' : ℕ, 
      x' + y' + z' = 20 → 2 * x' + 3 * y' + 4 * z' = 50 → 
      (1.2 * x + 1.5 * y + 1.2 * z : ℝ) ≥ (1.2 * x' + 1.5 * y' + 1.2 * z' : ℝ)) ∧ 
  x = 10 ∧ y = 10 ∧ z = 0 := by 
  sorry

end maximize_village_value_l641_64172


namespace arithmetic_sequence_a1_d_l641_64152

theorem arithmetic_sequence_a1_d (a_1 a_2 a_3 a_5 d : ℤ)
  (h1 : a_5 = a_1 + 4 * d)
  (h2 : a_1 + a_2 + a_3 = 3)
  (h3 : a_2 = a_1 + d)
  (h4 : a_3 = a_1 + 2 * d) :
  a_1 = -2 ∧ d = 3 :=
by
  have h_a2 : a_2 = 1 := sorry
  have h_a5 : a_5 = 10 := sorry
  have h_d : d = 3 := sorry
  have h_a1 : a_1 = -2 := sorry
  exact ⟨h_a1, h_d⟩

end arithmetic_sequence_a1_d_l641_64152


namespace xy_value_l641_64108

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : xy = 21 :=
sorry

end xy_value_l641_64108


namespace solve_inequality_l641_64181

def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

theorem solve_inequality : ∀ x : ℝ, |f x| ≤ 4 :=
by
  intro x
  sorry

end solve_inequality_l641_64181


namespace sum_of_first_150_remainder_l641_64142

theorem sum_of_first_150_remainder :
  let n := 150
  let sum := n * (n + 1) / 2
  sum % 5600 = 125 :=
by
  sorry

end sum_of_first_150_remainder_l641_64142


namespace find_m_range_l641_64104

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end find_m_range_l641_64104


namespace find_x_l641_64196

-- Conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 8 * x
def area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x / 2

-- Theorem to prove
theorem find_x (x s : ℝ) (h1 : volume_condition x s) (h2 : area_condition x s) : x = 110592 := sorry

end find_x_l641_64196


namespace difference_of_reciprocals_l641_64176

theorem difference_of_reciprocals (p q : ℝ) (hp : 3 / p = 6) (hq : 3 / q = 15) : p - q = 3 / 10 :=
by
  sorry

end difference_of_reciprocals_l641_64176


namespace roots_of_equations_l641_64145

theorem roots_of_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + 4 * a * x - 4 * a + 3 = 0) ∨
  (∃ x : ℝ, x^2 + (a - 1) * x + a^2 = 0) ∨
  (∃ x : ℝ, x^2 + 2 * a * x - 2 * a = 0) ↔ 
  a ≤ -3 / 2 ∨ a ≥ -1 :=
sorry

end roots_of_equations_l641_64145


namespace john_pennies_more_than_kate_l641_64149

theorem john_pennies_more_than_kate (kate_pennies : ℕ) (john_pennies : ℕ) (h_kate : kate_pennies = 223) (h_john : john_pennies = 388) : john_pennies - kate_pennies = 165 := by
  sorry

end john_pennies_more_than_kate_l641_64149


namespace space_left_over_l641_64173

theorem space_left_over (D B : ℕ) (wall_length desk_length bookcase_length : ℝ) (h_wall : wall_length = 15)
  (h_desk : desk_length = 2) (h_bookcase : bookcase_length = 1.5) (h_eq : D = B)
  (h_max : 2 * D + 1.5 * B ≤ wall_length) :
  ∃ w : ℝ, w = wall_length - (D * desk_length + B * bookcase_length) ∧ w = 1 :=
by
  sorry

end space_left_over_l641_64173


namespace solution_set_of_inequality_l641_64103

theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici 0.5 :=
by sorry

end solution_set_of_inequality_l641_64103


namespace values_of_x_that_satisfy_gg_x_eq_g_x_l641_64186

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end values_of_x_that_satisfy_gg_x_eq_g_x_l641_64186


namespace bread_rise_times_l641_64171

-- Defining the conditions
def rise_time : ℕ := 120
def kneading_time : ℕ := 10
def baking_time : ℕ := 30
def total_time : ℕ := 280

-- The proof statement
theorem bread_rise_times (n : ℕ) 
  (h1 : rise_time * n + kneading_time + baking_time = total_time) 
  : n = 2 :=
sorry

end bread_rise_times_l641_64171
