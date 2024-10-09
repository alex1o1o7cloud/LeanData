import Mathlib

namespace smallest_q_exists_l2312_231209

theorem smallest_q_exists (p q : ℕ) (h : 0 < q) (h_eq : (p : ℚ) / q = 123456789 / 100000000000) :
  q = 10989019 :=
sorry

end smallest_q_exists_l2312_231209


namespace function_unique_l2312_231201

open Function

-- Define the domain and codomain
def NatPos : Type := {n : ℕ // n > 0}

-- Define the function f from positive integers to positive integers
noncomputable def f : NatPos → NatPos := sorry

-- Provide the main theorem
theorem function_unique (f : NatPos → NatPos) :
  (∀ (m n : NatPos), (m.val ^ 2 + (f n).val) ∣ ((m.val * (f m).val) + n.val)) →
  (∀ n : NatPos, f n = n) :=
by
  sorry

end function_unique_l2312_231201


namespace quadratic_sum_roots_l2312_231242

theorem quadratic_sum_roots {a b : ℝ}
  (h1 : ∀ x, x^2 - a * x + b < 0 ↔ -1 < x ∧ x < 3) :
  a + b = -1 :=
sorry

end quadratic_sum_roots_l2312_231242


namespace burglary_charge_sentence_l2312_231254

theorem burglary_charge_sentence (B : ℕ) 
  (arson_counts : ℕ := 3) 
  (arson_sentence : ℕ := 36)
  (burglary_charges : ℕ := 2)
  (petty_larceny_factor : ℕ := 6)
  (total_jail_time : ℕ := 216) :
  arson_counts * arson_sentence + burglary_charges * B + (burglary_charges * petty_larceny_factor) * (B / 3) = total_jail_time → B = 18 := 
by
  sorry

end burglary_charge_sentence_l2312_231254


namespace range_of_a_l2312_231270

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 4) ∧ (2 * x^2 - 9 * x + a < 0)) ↔ (a < 4) :=
by
  sorry

end range_of_a_l2312_231270


namespace z_is_233_percent_greater_than_w_l2312_231263

theorem z_is_233_percent_greater_than_w
  (w e x y z : ℝ)
  (h1 : w = 0.5 * e)
  (h2 : e = 0.4 * x)
  (h3 : x = 0.3 * y)
  (h4 : z = 0.2 * y) :
  z = 2.3333 * w :=
by
  sorry

end z_is_233_percent_greater_than_w_l2312_231263


namespace total_books_sum_l2312_231244

-- Given conditions
def Joan_books := 10
def Tom_books := 38
def Lisa_books := 27
def Steve_books := 45
def Kim_books := 14
def Alex_books := 48

-- Define the total number of books
def total_books := Joan_books + Tom_books + Lisa_books + Steve_books + Kim_books + Alex_books

-- Proof statement
theorem total_books_sum : total_books = 182 := by
  sorry

end total_books_sum_l2312_231244


namespace total_number_of_questions_l2312_231285

theorem total_number_of_questions (N : ℕ)
  (hp : 0.8 * N = (4 / 5 : ℝ) * N)
  (hv : 35 = 35)
  (hb : (N / 2 : ℕ) = 1 * (N.div 2))
  (ha : N - 7 = N - 7) : N = 60 :=
by
  sorry

end total_number_of_questions_l2312_231285


namespace volume_of_polyhedron_l2312_231290

theorem volume_of_polyhedron (V : ℝ) (hV : 0 ≤ V) :
  ∃ P : ℝ, P = V / 6 :=
by
  sorry

end volume_of_polyhedron_l2312_231290


namespace factor_count_l2312_231275

theorem factor_count (x : ℤ) : 
  (x^12 - x^3) = x^3 * (x - 1) * (x^2 + x + 1) * (x^6 + x^3 + 1) -> 4 = 4 :=
by
  sorry

end factor_count_l2312_231275


namespace reciprocal_of_8_l2312_231259

theorem reciprocal_of_8:
  (1 : ℝ) / 8 = (1 / 8 : ℝ) := by
  sorry

end reciprocal_of_8_l2312_231259


namespace proof_method_characterization_l2312_231287

-- Definitions of each method
def synthetic_method := "proceeds from cause to effect, in a forward manner"
def analytic_method := "seeks the cause from the effect, working backwards"
def proof_by_contradiction := "assumes the negation of the proposition to be true, and derives a contradiction"
def mathematical_induction := "base case and inductive step: which shows that P holds for all natural numbers"

-- Main theorem to prove
theorem proof_method_characterization :
  (analytic_method == "seeks the cause from the effect, working backwards") :=
by
  sorry

end proof_method_characterization_l2312_231287


namespace square_feet_per_acre_l2312_231241

-- Define the conditions
def rent_per_acre_per_month : ℝ := 60
def total_rent_per_month : ℝ := 600
def length_of_plot : ℝ := 360
def width_of_plot : ℝ := 1210

-- Translate the problem to a Lean theorem
theorem square_feet_per_acre :
  (length_of_plot * width_of_plot) / (total_rent_per_month / rent_per_acre_per_month) = 43560 :=
by {
  -- skipping the proof steps
  sorry
}

end square_feet_per_acre_l2312_231241


namespace isosceles_right_triangle_contains_probability_l2312_231240

noncomputable def isosceles_right_triangle_probability : ℝ :=
  let leg_length := 2
  let triangle_area := (leg_length * leg_length) / 2
  let distance_radius := 1
  let quarter_circle_area := (Real.pi * (distance_radius * distance_radius)) / 4
  quarter_circle_area / triangle_area

theorem isosceles_right_triangle_contains_probability :
  isosceles_right_triangle_probability = (Real.pi / 8) :=
by
  sorry

end isosceles_right_triangle_contains_probability_l2312_231240


namespace max_value_2ab_plus_2bc_sqrt2_l2312_231258

theorem max_value_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_value_2ab_plus_2bc_sqrt2_l2312_231258


namespace find_number_of_sides_l2312_231277

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end find_number_of_sides_l2312_231277


namespace sam_bought_17_mystery_books_l2312_231205

def adventure_books := 13
def used_books := 15
def new_books := 15
def total_books := used_books + new_books
def mystery_books := total_books - adventure_books

theorem sam_bought_17_mystery_books : mystery_books = 17 := by
  sorry

end sam_bought_17_mystery_books_l2312_231205


namespace exists_sequences_x_y_l2312_231276

def seq_a (a : ℕ → ℕ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n : ℕ, n ≥ 2 → a (n) = 6 * a (n - 1) - a (n - 2)

def seq_b (b : ℕ → ℕ) : Prop :=
  b 0 = 2 ∧ b 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → b (n) = 2 * b (n - 1) + b (n - 2)

theorem exists_sequences_x_y (a b : ℕ → ℕ) (x y : ℕ → ℕ) :
  seq_a a → seq_b b →
  (∀ n : ℕ, a n = (y n * y n + 7) / (x n - y n)) ↔ 
  (∀ n : ℕ, y n = b (2 * n + 1) ∧ x n = b (2 * n) + y n) :=
sorry

end exists_sequences_x_y_l2312_231276


namespace find_y_l2312_231288

theorem find_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : y = 5 :=
by
  sorry

end find_y_l2312_231288


namespace blipblish_modulo_l2312_231236

-- Definitions from the conditions
inductive Letter
| B | I | L

def is_consonant (c : Letter) : Bool :=
  match c with
  | Letter.B | Letter.L => true
  | _ => false

def is_vowel (v : Letter) : Bool :=
  match v with
  | Letter.I => true
  | _ => false

def is_valid_blipblish_word (word : List Letter) : Bool :=
  -- Check if between any two I's there at least three consonants
  let rec check (lst : List Letter) (cnt : Nat) (during_vowels : Bool) : Bool :=
    match lst with
    | [] => true
    | Letter.I :: xs =>
        if during_vowels then cnt >= 3 && check xs 0 false
        else check xs 0 true
    | x :: xs =>
        if is_consonant x then check xs (cnt + 1) during_vowels
        else check xs cnt during_vowels
  check word 0 false

def number_of_valid_words (n : Nat) : Nat :=
  -- Placeholder function to compute the number of valid Blipblish words of length n
  sorry

-- Statement of the proof problem
theorem blipblish_modulo : number_of_valid_words 12 % 1000 = 312 :=
by sorry

end blipblish_modulo_l2312_231236


namespace father_cannot_see_boy_more_than_half_time_l2312_231256

def speed_boy := 10 -- speed in km/h
def speed_father := 5 -- speed in km/h

def cannot_see_boy_more_than_half_time (school_perimeter : ℝ) : Prop :=
  ¬(∃ T : ℝ, T > school_perimeter / (2 * speed_boy) ∧ T < school_perimeter / speed_boy)

theorem father_cannot_see_boy_more_than_half_time (school_perimeter : ℝ) (h_school_perimeter : school_perimeter > 0) :
  cannot_see_boy_more_than_half_time school_perimeter :=
by
  sorry

end father_cannot_see_boy_more_than_half_time_l2312_231256


namespace divisor_of_product_of_four_consecutive_integers_l2312_231206

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l2312_231206


namespace pure_gala_trees_l2312_231246

variable (T F G : ℝ)

theorem pure_gala_trees (h1 : F + 0.1 * T = 170) (h2 : F = 0.75 * T): G = T - F -> G = 50 :=
by
  sorry

end pure_gala_trees_l2312_231246


namespace age_of_student_who_left_l2312_231280

/-- 
The average student age of a class with 30 students is 10 years.
After one student leaves and the teacher (who is 41 years old) is included,
the new average age is 11 years. Prove that the student who left is 11 years old.
-/
theorem age_of_student_who_left (x : ℕ) (h1 : (30 * 10) = 300)
    (h2 : (300 - x + 41) / 30 = 11) : x = 11 :=
by 
  -- This is where the proof would go
  sorry

end age_of_student_who_left_l2312_231280


namespace find_number_l2312_231248

theorem find_number {x : ℝ} (h : (1/3) * x = 130.00000000000003) : x = 390 := 
sorry

end find_number_l2312_231248


namespace complete_square_eq_l2312_231227

theorem complete_square_eq (x : ℝ) :
  x^2 - 8 * x + 15 = 0 →
  (x - 4)^2 = 1 :=
by sorry

end complete_square_eq_l2312_231227


namespace product_of_two_numbers_l2312_231229

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : x - y = 16) : 
  x * y = 836 := 
by
  sorry

end product_of_two_numbers_l2312_231229


namespace selection_count_l2312_231273

noncomputable def choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count :
  let boys := 4
  let girls := 3
  let total := boys + girls
  let choose_boys_girls : ℕ := (choose 4 2) * (choose 3 1) + (choose 4 1) * (choose 3 2)
  choose_boys_girls = 30 := 
by
  sorry

end selection_count_l2312_231273


namespace pairs_satisfaction_l2312_231203

-- Definitions for the conditions given
def condition1 (x y : ℝ) : Prop := y = (x + 2)^2
def condition2 (x y : ℝ) : Prop := x * y + 2 * y = 2

-- The statement that we need to prove
theorem pairs_satisfaction : 
  (∃ x y : ℝ, condition1 x y ∧ condition2 x y) ∧ 
  (∃ x1 x2 : ℂ, x^2 + -2*x + 1 = 0 ∧ ¬∃ (y : ℝ), y = (x1 + 2)^2 ∨ y = (x2 + 2)^2) :=
by
  sorry

end pairs_satisfaction_l2312_231203


namespace arithmetic_sequence_seventh_term_l2312_231232

theorem arithmetic_sequence_seventh_term
  (a d : ℝ)
  (h_sum : 4 * a + 6 * d = 20)
  (h_fifth : a + 4 * d = 8) :
  a + 6 * d = 10.4 :=
by
  sorry -- proof to be provided

end arithmetic_sequence_seventh_term_l2312_231232


namespace largest_lcm_value_l2312_231286

-- Define the conditions as local constants 
def lcm_18_3 : ℕ := Nat.lcm 18 3
def lcm_18_6 : ℕ := Nat.lcm 18 6
def lcm_18_9 : ℕ := Nat.lcm 18 9
def lcm_18_15 : ℕ := Nat.lcm 18 15
def lcm_18_21 : ℕ := Nat.lcm 18 21
def lcm_18_27 : ℕ := Nat.lcm 18 27

-- Statement to prove
theorem largest_lcm_value : max lcm_18_3 (max lcm_18_6 (max lcm_18_9 (max lcm_18_15 (max lcm_18_21 lcm_18_27)))) = 126 :=
by
  -- We assume the necessary calculations have been made
  have h1 : lcm_18_3 = 18 := by sorry
  have h2 : lcm_18_6 = 18 := by sorry
  have h3 : lcm_18_9 = 18 := by sorry
  have h4 : lcm_18_15 = 90 := by sorry
  have h5 : lcm_18_21 = 126 := by sorry
  have h6 : lcm_18_27 = 54 := by sorry

  -- Using above results to determine the maximum
  exact (by rw [h1, h2, h3, h4, h5, h6]; exact rfl)

end largest_lcm_value_l2312_231286


namespace expand_expression_l2312_231291

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := 
sorry

end expand_expression_l2312_231291


namespace number_of_cows_in_farm_l2312_231223

-- Definitions relating to the conditions
def total_bags_consumed := 20
def bags_per_cow := 1
def days := 20

-- Question and proof of the answer
theorem number_of_cows_in_farm : (total_bags_consumed / bags_per_cow) = 20 := by
  -- proof goes here
  sorry

end number_of_cows_in_farm_l2312_231223


namespace probability_of_3_tails_in_8_flips_l2312_231235

open ProbabilityTheory

/-- The probability of getting exactly 3 tails out of 8 flips of an unfair coin, where the probability of tails is 4/5 and the probability of heads is 1/5, is 3584/390625. -/
theorem probability_of_3_tails_in_8_flips :
  let p_heads := 1 / 5
  let p_tails := 4 / 5
  let n_trials := 8
  let k_successes := 3
  let binomial_coefficient := Nat.choose n_trials k_successes
  let probability := binomial_coefficient * (p_tails ^ k_successes) * (p_heads ^ (n_trials - k_successes))
  probability = (3584 : ℚ) / 390625 := 
by 
  sorry

end probability_of_3_tails_in_8_flips_l2312_231235


namespace total_spider_legs_l2312_231261

-- Definition of the number of spiders
def number_of_spiders : ℕ := 5

-- Definition of the number of legs per spider
def legs_per_spider : ℕ := 8

-- Theorem statement to prove the total number of spider legs
theorem total_spider_legs : number_of_spiders * legs_per_spider = 40 :=
by 
  -- We've planned to use 'sorry' to skip the proof
  sorry

end total_spider_legs_l2312_231261


namespace solve_pos_int_a_l2312_231207

theorem solve_pos_int_a :
  ∀ a : ℕ, (0 < a) →
  (∀ n : ℕ, (n ≥ 5) → ((2^n - n^2) ∣ (a^n - n^a))) →
  (a = 2 ∨ a = 4) :=
by
  sorry

end solve_pos_int_a_l2312_231207


namespace roof_length_width_diff_l2312_231224

theorem roof_length_width_diff (w l : ℕ) (h1 : l = 4 * w) (h2 : 784 = l * w) : l - w = 42 := by
  sorry

end roof_length_width_diff_l2312_231224


namespace compute_expression_l2312_231238

theorem compute_expression :
  ( (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) )
  /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) )
  = 221 := 
by sorry

end compute_expression_l2312_231238


namespace find_first_number_l2312_231289

variable (a : ℕ → ℤ)

axiom recurrence_rel : ∀ (n : ℕ), n ≥ 4 → a n = a (n - 1) + a (n - 2) + a (n - 3)
axiom a8_val : a 8 = 29
axiom a9_val : a 9 = 56
axiom a10_val : a 10 = 108

theorem find_first_number : a 1 = 32 :=
sorry

end find_first_number_l2312_231289


namespace max_books_borrowed_l2312_231293

theorem max_books_borrowed 
  (num_students : ℕ)
  (num_no_books : ℕ)
  (num_one_book : ℕ)
  (num_two_books : ℕ)
  (average_books : ℕ)
  (h_num_students : num_students = 32)
  (h_num_no_books : num_no_books = 2)
  (h_num_one_book : num_one_book = 12)
  (h_num_two_books : num_two_books = 10)
  (h_average_books : average_books = 2)
  : ∃ max_books : ℕ, max_books = 11 := 
by
  sorry

end max_books_borrowed_l2312_231293


namespace original_salary_l2312_231292

theorem original_salary (S : ℝ) (h1 : S + 0.10 * S = 1.10 * S) (h2: 1.10 * S - 0.05 * (1.10 * S) = 1.10 * S * 0.95) (h3: 1.10 * S * 0.95 = 2090) : S = 2000 :=
sorry

end original_salary_l2312_231292


namespace min_length_l2312_231297

def length (a b : ℝ) : ℝ := b - a

noncomputable def M (m : ℝ) := {x | m ≤ x ∧ x ≤ m + 3 / 4}
noncomputable def N (n : ℝ) := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
noncomputable def intersection (m n : ℝ) := {x | max m (n - 1 / 3) ≤ x ∧ x ≤ min (m + 3 / 4) n}

theorem min_length (m n : ℝ) (hM : ∀ x, x ∈ M m → 0 ≤ x ∧ x ≤ 1) (hN : ∀ x, x ∈ N n → 0 ≤ x ∧ x ≤ 1) :
  length (max m (n - 1 / 3)) (min (m + 3 / 4) n) = 1 / 12 :=
sorry

end min_length_l2312_231297


namespace proposition_range_l2312_231278

theorem proposition_range (m : ℝ) : 
  (m < 1/2 ∧ m ≠ 1/3) ∨ (m = 3) ↔ m ∈ Set.Iio (1/3:ℝ) ∪ Set.Ioo (1/3:ℝ) (1/2:ℝ) ∪ {3} :=
sorry

end proposition_range_l2312_231278


namespace gcd_840_1764_l2312_231219

-- Define the numbers according to the conditions
def a : ℕ := 1764
def b : ℕ := 840

-- The goal is to prove that the GCD of a and b is 84
theorem gcd_840_1764 : Nat.gcd a b = 84 := 
by
  -- The proof steps would normally go here
  sorry

end gcd_840_1764_l2312_231219


namespace PP1_length_l2312_231213

open Real

theorem PP1_length (AB AC : ℝ) (h₁ : AB = 5) (h₂ : AC = 3)
  (h₃ : ∃ γ : ℝ, γ = 90)  -- a right angle at A
  (BC : ℝ) (h₄ : BC = sqrt (AB^2 - AC^2))
  (A1B : ℝ) (A1C : ℝ) (h₅ : BC = A1B + A1C)
  (h₆ : A1B / A1C = AB / AC)
  (PQ : ℝ) (h₇ : PQ = A1B)
  (PR : ℝ) (h₈ : PR = A1C)
  (PP1 : ℝ) :
  PP1 = (3 * sqrt 5) / 4 :=
sorry

end PP1_length_l2312_231213


namespace rhombus_side_length_l2312_231216

-- Define the rhombus properties and the problem conditions
variables (p q x : ℝ)

-- State the problem as a theorem in Lean 4
theorem rhombus_side_length (h : x^2 = p * q) : x = Real.sqrt (p * q) :=
sorry

end rhombus_side_length_l2312_231216


namespace infinite_series_eq_1_div_400_l2312_231257

theorem infinite_series_eq_1_div_400 :
  (∑' n:ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 :=
by
  sorry

end infinite_series_eq_1_div_400_l2312_231257


namespace Michael_selection_l2312_231294

theorem Michael_selection :
  (Nat.choose 8 3) * (Nat.choose 5 2) = 560 :=
by
  sorry

end Michael_selection_l2312_231294


namespace multiplication_of_negative_and_positive_l2312_231225

theorem multiplication_of_negative_and_positive :
  (-3) * 5 = -15 :=
by
  sorry

end multiplication_of_negative_and_positive_l2312_231225


namespace compute_sum_of_products_of_coefficients_l2312_231233

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end compute_sum_of_products_of_coefficients_l2312_231233


namespace bobbit_worm_fish_count_l2312_231221

theorem bobbit_worm_fish_count 
  (initial_fish : ℕ)
  (fish_eaten_per_day : ℕ)
  (days_before_adding_fish : ℕ)
  (additional_fish : ℕ)
  (days_after_adding_fish : ℕ) :
  days_before_adding_fish = 14 →
  days_after_adding_fish = 7 →
  fish_eaten_per_day = 2 →
  initial_fish = 60 →
  additional_fish = 8 →
  (initial_fish - days_before_adding_fish * fish_eaten_per_day + additional_fish - days_after_adding_fish * fish_eaten_per_day) = 26 :=
by
  intros 
  -- sorry proof goes here
  sorry

end bobbit_worm_fish_count_l2312_231221


namespace days_to_clear_messages_l2312_231230

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end days_to_clear_messages_l2312_231230


namespace measure_of_each_interior_angle_l2312_231260

theorem measure_of_each_interior_angle (n : ℕ) (hn : 3 ≤ n) : 
  ∃ angle : ℝ, angle = (n - 2) * 180 / n :=
by
  sorry

end measure_of_each_interior_angle_l2312_231260


namespace math_problem_l2312_231202
-- Import the entire mathlib library for necessary mathematical definitions and notations

-- Define the conditions and the statement to prove
theorem math_problem (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 :=
by 
  -- place a sorry as a placeholder for the proof
  sorry

end math_problem_l2312_231202


namespace jane_age_problem_l2312_231283

variables (J M a b c : ℕ)
variables (h1 : J = 2 * (a + b))
variables (h2 : J / 2 = a + b)
variables (h3 : c = 2 * J)
variables (h4 : M > 0)

theorem jane_age_problem (h5 : J - M = 3 * ((J / 2) - 2 * M))
                         (h6 : J - M = c - M)
                         (h7 : c = 2 * J) :
  J / M = 10 :=
sorry

end jane_age_problem_l2312_231283


namespace expected_value_is_correct_l2312_231284

-- Define the probabilities of heads and tails
def P_H := 2 / 5
def P_T := 3 / 5

-- Define the winnings for heads and the loss for tails
def W_H := 5
def L_T := -4

-- Calculate the expected value
def expected_value := P_H * W_H + P_T * L_T

-- Prove that the expected value is -2/5
theorem expected_value_is_correct : expected_value = -2 / 5 := by
  sorry

end expected_value_is_correct_l2312_231284


namespace does_not_represent_right_triangle_l2312_231220

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively. Given:
  - a:b:c = 6:8:10
  - ∠A:∠B:∠C = 1:1:3
  - a^2 + c^2 = b^2
  - ∠A + ∠B = ∠C

Prove that the condition ∠A:∠B:∠C = 1:1:3 does not represent a right triangle ABC. -/
theorem does_not_represent_right_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a / b = 6 / 8 ∧ b / c = 8 / 10)
  (h2 : A / B = 1 / 1 ∧ B / C = 1 / 3)
  (h3 : a^2 + c^2 = b^2)
  (h4 : A + B = C) :
  ¬ (B = 90) :=
sorry

end does_not_represent_right_triangle_l2312_231220


namespace probability_of_event_B_given_A_l2312_231249

-- Definition of events and probability
noncomputable def prob_event_B_given_A : ℝ :=
  let total_outcomes := 36
  let outcomes_A := 30
  let outcomes_B_given_A := 10
  outcomes_B_given_A / outcomes_A

-- Theorem statement
theorem probability_of_event_B_given_A : prob_event_B_given_A = 1 / 3 := by
  sorry

end probability_of_event_B_given_A_l2312_231249


namespace pages_left_l2312_231245

-- Define the conditions
def initial_books := 10
def pages_per_book := 100
def books_lost := 2

-- The total pages Phil had initially
def initial_pages := initial_books * pages_per_book

-- The number of books left after losing some during the move
def books_left := initial_books - books_lost

-- Prove the number of pages worth of books Phil has left
theorem pages_left : books_left * pages_per_book = 800 := by
  sorry

end pages_left_l2312_231245


namespace average_of_D_E_F_l2312_231215

theorem average_of_D_E_F (D E F : ℝ) 
  (h1 : 2003 * F - 4006 * D = 8012) 
  (h2 : 2003 * E + 6009 * D = 10010) : 
  (D + E + F) / 3 = 3 := 
by 
  sorry

end average_of_D_E_F_l2312_231215


namespace find_certain_amount_l2312_231210

theorem find_certain_amount :
  ∀ (A : ℝ), (160 * 8 * 12.5 / 100 = A * 8 * 4 / 100) → 
            (A = 500) :=
  by
    intros A h
    sorry

end find_certain_amount_l2312_231210


namespace rick_gives_miguel_cards_l2312_231274

/-- Rick starts with 130 cards, keeps 15 cards for himself, gives 
12 cards each to 8 friends, and gives 3 cards each to his 2 sisters. 
We need to prove that Rick gives 13 cards to Miguel. --/
theorem rick_gives_miguel_cards :
  let initial_cards := 130
  let kept_cards := 15
  let friends := 8
  let cards_per_friend := 12
  let sisters := 2
  let cards_per_sister := 3
  initial_cards - kept_cards - (friends * cards_per_friend) - (sisters * cards_per_sister) = 13 :=
by
  sorry

end rick_gives_miguel_cards_l2312_231274


namespace sandwiches_provided_l2312_231282

theorem sandwiches_provided (original_count sold_out : ℕ) (h1 : original_count = 9) (h2 : sold_out = 5) : (original_count - sold_out = 4) :=
by
  sorry

end sandwiches_provided_l2312_231282


namespace problem_1_problem_2_l2312_231295

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.sqrt 3 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h_symmetry : ∃ k : ℤ, a = k * Real.pi / 2) : g (2 * a) = 1 / 2 := by
  sorry

-- Proof Problem 2
theorem problem_2 (x : ℝ) (h_range : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  ∃ y : ℝ, y = h x ∧ 1/2 ≤ y ∧ y ≤ 2 := by
  sorry

end problem_1_problem_2_l2312_231295


namespace f_11_f_2021_eq_neg_one_l2312_231231

def f (n : ℕ) : ℚ := sorry

axiom recurrence_relation (n : ℕ) : f (n + 3) = (f n - 1) / (f n + 1)
axiom f1_ne_zero : f 1 ≠ 0
axiom f1_ne_one : f 1 ≠ 1
axiom f1_ne_neg_one : f 1 ≠ -1

theorem f_11_f_2021_eq_neg_one : f 11 * f 2021 = -1 := 
by
  sorry

end f_11_f_2021_eq_neg_one_l2312_231231


namespace find_x1_l2312_231226

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1-x1)^2 + 2*(x1-x2)^2 + (x2-x3)^2 + x3^2 = 1/2) :
  x1 = (3*Real.sqrt 2 - 3)/7 :=
by
  sorry

end find_x1_l2312_231226


namespace problem1_problem2_solution_l2312_231243

noncomputable def trig_expr : ℝ :=
  3 * Real.tan (30 * Real.pi / 180) - (Real.tan (45 * Real.pi / 180))^2 + 2 * Real.sin (60 * Real.pi / 180)

theorem problem1 : trig_expr = 2 * Real.sqrt 3 - 1 :=
by
  -- Proof omitted
  sorry

noncomputable def quad_eq (x : ℝ) : Prop := 
  (3*x - 1) * (x + 2) = 11*x - 4

theorem problem2_solution (x : ℝ) : quad_eq x ↔ (x = (3 + Real.sqrt 3) / 3 ∨ x = (3 - Real.sqrt 3) / 3) :=
by
  -- Proof omitted
  sorry

end problem1_problem2_solution_l2312_231243


namespace income_to_expenditure_ratio_l2312_231267

theorem income_to_expenditure_ratio (I E S : ℕ) (hI : I = 15000) (hS : S = 7000) (hSavings : S = I - E) :
  I / E = 15 / 8 := by
  -- Lean proof goes here
  sorry

end income_to_expenditure_ratio_l2312_231267


namespace sum_of_midpoint_coords_l2312_231247

theorem sum_of_midpoint_coords :
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym = 11 :=
by
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  sorry

end sum_of_midpoint_coords_l2312_231247


namespace tan_sub_pi_over_4_l2312_231214

variables (α : ℝ)
axiom tan_alpha : Real.tan α = 1 / 6

theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = -5 / 7 := by
  sorry

end tan_sub_pi_over_4_l2312_231214


namespace original_proposition_true_converse_proposition_false_l2312_231250

theorem original_proposition_true (a b : ℝ) : 
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := 
sorry

theorem converse_proposition_false : 
  ¬ (∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_proposition_false_l2312_231250


namespace geese_percentage_among_non_swan_birds_l2312_231266

theorem geese_percentage_among_non_swan_birds :
  let total_birds := 100
  let geese := 0.40 * total_birds
  let swans := 0.20 * total_birds
  let non_swans := total_birds - swans
  let geese_percentage_among_non_swans := (geese / non_swans) * 100
  geese_percentage_among_non_swans = 50 := 
by sorry

end geese_percentage_among_non_swan_birds_l2312_231266


namespace plot_area_in_acres_l2312_231208

theorem plot_area_in_acres :
  let scale_cm_to_miles : ℝ := 3
  let base1_cm : ℝ := 20
  let base2_cm : ℝ := 25
  let height_cm : ℝ := 15
  let miles_to_acres : ℝ := 640
  let area_trapezoid_cm2 := (1 / 2) * (base1_cm + base2_cm) * height_cm
  let area_trapezoid_miles2 := area_trapezoid_cm2 * (scale_cm_to_miles ^ 2)
  let area_trapezoid_acres := area_trapezoid_miles2 * miles_to_acres
  area_trapezoid_acres = 1944000 := by
    sorry

end plot_area_in_acres_l2312_231208


namespace quadratic_root_condition_l2312_231237

theorem quadratic_root_condition (m n : ℝ) (h : m * (-1)^2 - n * (-1) - 2023 = 0) :
  m + n = 2023 :=
sorry

end quadratic_root_condition_l2312_231237


namespace interest_rate_for_4000_investment_l2312_231239

theorem interest_rate_for_4000_investment
      (total_money : ℝ := 9000)
      (invested_at_9_percent : ℝ := 5000)
      (total_interest : ℝ := 770)
      (invested_at_unknown_rate : ℝ := 4000) :
  ∃ r : ℝ, invested_at_unknown_rate * r = total_interest - (invested_at_9_percent * 0.09) ∧ r = 0.08 :=
by {
  -- Proof is not required based on instruction, so we use sorry.
  sorry
}

end interest_rate_for_4000_investment_l2312_231239


namespace transformed_quadratic_equation_l2312_231228

theorem transformed_quadratic_equation (u v: ℝ) :
  (u + v = -5 / 2) ∧ (u * v = 3 / 2) ↔ (∃ y : ℝ, y^2 - y + 6 = 0) := sorry

end transformed_quadratic_equation_l2312_231228


namespace cats_count_l2312_231269

-- Definitions based on conditions
def heads_eqn (H C : ℕ) : Prop := H + C = 15
def legs_eqn (H C : ℕ) : Prop := 2 * H + 4 * C = 44

-- The main proof problem
theorem cats_count (H C : ℕ) (h1 : heads_eqn H C) (h2 : legs_eqn H C) : C = 7 :=
by
  sorry

end cats_count_l2312_231269


namespace total_sand_volume_l2312_231272

noncomputable def cone_diameter : ℝ := 10
noncomputable def cone_radius : ℝ := cone_diameter / 2
noncomputable def cone_height : ℝ := 0.75 * cone_diameter
noncomputable def cylinder_height : ℝ := 0.5 * cone_diameter
noncomputable def total_volume : ℝ := (1 / 3 * Real.pi * cone_radius^2 * cone_height) + (Real.pi * cone_radius^2 * cylinder_height)

theorem total_sand_volume : total_volume = 187.5 * Real.pi := 
by
  sorry

end total_sand_volume_l2312_231272


namespace frac_left_handed_l2312_231252

variable (x : ℕ)

def red_participants := 10 * x
def blue_participants := 5 * x
def total_participants := red_participants x + blue_participants x

def left_handed_red := (1 / 3 : ℚ) * red_participants x
def left_handed_blue := (2 / 3 : ℚ) * blue_participants x
def total_left_handed := left_handed_red x + left_handed_blue x

theorem frac_left_handed :
  total_left_handed x / total_participants x = (4 / 9 : ℚ) := by
  sorry

end frac_left_handed_l2312_231252


namespace average_cost_per_pencil_proof_l2312_231212

noncomputable def average_cost_per_pencil (pencils_qty: ℕ) (price: ℝ) (discount_percent: ℝ) (shipping_cost: ℝ) : ℝ :=
  let discounted_price := price * (1 - discount_percent / 100)
  let total_cost := discounted_price + shipping_cost
  let cost_in_cents := total_cost * 100
  cost_in_cents / pencils_qty

theorem average_cost_per_pencil_proof :
  average_cost_per_pencil 300 29.85 10 7.50 = 11 :=
by
  sorry

end average_cost_per_pencil_proof_l2312_231212


namespace vertex_of_parabola_l2312_231264

theorem vertex_of_parabola :
  ∀ x : ℝ, (x - 2) ^ 2 + 4 = (x - 2) ^ 2 + 4 → (2, 4) = (2, 4) :=
by
  intro x
  intro h
  -- We know that the vertex of y = (x - 2)^2 + 4 is at (2, 4)
  admit

end vertex_of_parabola_l2312_231264


namespace find_a_l2312_231204

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 5) (h3 : c = 3) : a = 1 := by
  sorry

end find_a_l2312_231204


namespace decagon_diagonals_l2312_231298

-- Define the number of sides of a decagon
def n : ℕ := 10

-- Define the formula for the number of diagonals in an n-sided polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem decagon_diagonals : num_diagonals n = 35 := by
  sorry

end decagon_diagonals_l2312_231298


namespace correct_average_of_corrected_number_l2312_231265

theorem correct_average_of_corrected_number (num_list : List ℤ) (wrong_num correct_num : ℤ) (n : ℕ)
  (hn : n = 10)
  (haverage : (num_list.sum / n) = 5)
  (hwrong : wrong_num = 26)
  (hcorrect : correct_num = 36)
  (hnum_list_sum : num_list.sum + correct_num - wrong_num = num_list.sum + 10) :
  (num_list.sum + 10) / n = 6 :=
by
  sorry

end correct_average_of_corrected_number_l2312_231265


namespace find_intersection_complement_find_value_m_l2312_231253

-- (1) Problem Statement
theorem find_intersection_complement (A : Set ℝ) (B : Set ℝ) (x : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - 3 < 0}) →
  (x ∈ A ∩ (Bᶜ : Set ℝ)) ↔ (x = -1 ∨ 3 ≤ x ∧ x ≤ 5) :=
by
  sorry

-- (2) Problem Statement
theorem find_value_m (A B : Set ℝ) (m : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - m < 0}) →
  (A ∩ B = {x | -1 ≤ x ∧ x < 4}) →
  m = 8 :=
by
  sorry

end find_intersection_complement_find_value_m_l2312_231253


namespace sum_of_first_five_terms_l2312_231271

noncomputable -- assuming non-computable for general proof involving sums
def arithmetic_sequence_sum (a_n : ℕ → ℤ) := ∃ d m : ℤ, ∀ n : ℕ, a_n = m + n * d

theorem sum_of_first_five_terms 
(a_n : ℕ → ℤ) 
(h_arith : arithmetic_sequence_sum a_n)
(h_cond : a_n 5 + a_n 8 - a_n 10 = 2)
: ((a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) = 10) := 
by 
  sorry

end sum_of_first_five_terms_l2312_231271


namespace distance_traveled_l2312_231296

theorem distance_traveled 
    (P_b : ℕ) (P_f : ℕ) (R_b : ℕ) (R_f : ℕ)
    (h1 : P_b = 9)
    (h2 : P_f = 7)
    (h3 : R_f = R_b + 10) 
    (h4 : R_b * P_b = R_f * P_f) :
    R_b * P_b = 315 :=
by
  sorry

end distance_traveled_l2312_231296


namespace eq_iff_solution_l2312_231200

theorem eq_iff_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^y + y^x = x^x + y^y ↔ x = y :=
by sorry

end eq_iff_solution_l2312_231200


namespace evaluate_fraction_l2312_231218

theorem evaluate_fraction (a b : ℕ) (h₁ : a = 250) (h₂ : b = 240) :
  1800^2 / (a^2 - b^2) = 660 :=
by 
  sorry

end evaluate_fraction_l2312_231218


namespace average_speed_of_train_l2312_231268

def ChicagoTime (t : String) : Prop := t = "5:00 PM"
def NewYorkTime (t : String) : Prop := t = "10:00 AM"
def TimeDifference (d : Nat) : Prop := d = 1
def Distance (d : Nat) : Prop := d = 480

theorem average_speed_of_train :
  ∀ (d t1 t2 diff : Nat), 
  Distance d → (NewYorkTime "10:00 AM") → (ChicagoTime "5:00 PM") → TimeDifference diff →
  (t2 = 5 ∧ t1 = (10 - diff)) →
  (d / (t2 - t1) = 60) :=
by
  intros d t1 t2 diff hD ht1 ht2 hDiff hTimes
  sorry

end average_speed_of_train_l2312_231268


namespace find_general_term_l2312_231234

theorem find_general_term (S a : ℕ → ℤ) (n : ℕ) (h_sum : S n = 2 * a n + 1) : a n = -2 * n - 1 := sorry

end find_general_term_l2312_231234


namespace lucky_numbers_count_l2312_231217

def isLuckyNumber (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3 = 6) && (100 ≤ n) && (n < 1000)

def countLuckyNumbers : ℕ :=
  (List.range' 100 900).filter isLuckyNumber |>.length

theorem lucky_numbers_count : countLuckyNumbers = 21 := 
  sorry

end lucky_numbers_count_l2312_231217


namespace ones_digit_of_p_is_3_l2312_231299

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l2312_231299


namespace pipeline_problem_l2312_231251

theorem pipeline_problem 
  (length_pipeline : ℕ) 
  (extra_meters : ℕ) 
  (days_saved : ℕ) 
  (x : ℕ)
  (h1 : length_pipeline = 4000) 
  (h2 : extra_meters = 10) 
  (h3 : days_saved = 20) 
  (h4 : (4000:ℕ) / (x - extra_meters) - (4000:ℕ) / x = days_saved) :
  x = 4000 / ((4000 / (x - extra_meters) + 20)) + extra_meters :=
by
  -- The proof goes here
  sorry

end pipeline_problem_l2312_231251


namespace kris_age_l2312_231222

theorem kris_age (kris_age herbert_age : ℕ) (h1 : herbert_age + 1 = 15) (h2 : herbert_age + 10 = kris_age) : kris_age = 24 :=
by
  sorry

end kris_age_l2312_231222


namespace yarn_total_length_l2312_231211

/-- The green yarn is 156 cm long, the red yarn is 8 cm more than three times the green yarn,
    prove that the total length of the two pieces of yarn is 632 cm. --/
theorem yarn_total_length : 
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  green_yarn + red_yarn = 632 :=
by
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  sorry

end yarn_total_length_l2312_231211


namespace oliver_january_money_l2312_231262

variable (x y z : ℕ)

-- Given conditions
def condition1 := y = x - 4
def condition2 := z = y + 32
def condition3 := z = 61

-- Statement to prove
theorem oliver_january_money (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z) : x = 33 :=
by
  sorry

end oliver_january_money_l2312_231262


namespace x_values_l2312_231255

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 :=
by
  sorry

end x_values_l2312_231255


namespace sum_first_10_terms_l2312_231281

def arithmetic_sequence (a d : Int) (n : Int) : Int :=
  a + (n - 1) * d

def arithmetic_sum (a d : Int) (n : Int) : Int :=
  (n : Int) * a + (n * (n - 1) / 2) * d

theorem sum_first_10_terms  
  (a d : Int)
  (h1 : (a + 3 * d)^2 = (a + 2 * d) * (a + 6 * d))
  (h2 : arithmetic_sum a d 8 = 32)
  : arithmetic_sum a d 10 = 60 :=
sorry

end sum_first_10_terms_l2312_231281


namespace current_speed_is_one_l2312_231279

noncomputable def motorboat_rate_of_current (b h t : ℝ) : ℝ :=
  let eq1 := (b + 1 - h) * 4
  let eq2 := (b - 1 + t) * 6
  if eq1 = 24 ∧ eq2 = 24 then 1 else sorry

theorem current_speed_is_one (b h t : ℝ) : motorboat_rate_of_current b h t = 1 :=
by
  sorry

end current_speed_is_one_l2312_231279
