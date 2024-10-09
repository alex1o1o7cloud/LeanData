import Mathlib

namespace range_of_a_sq_l1753_175393

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ m n : ℕ, a (m + n) = a m + a n

theorem range_of_a_sq {n : ℕ}
  (h_arith : arithmetic_sequence a)
  (h_cond : a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1) :
  ∃ (L R : ℝ), (L = 2) ∧ (∀ k : ℕ, a (n+1) ^ 2 + a (3*n+1) ^ 2 ≥ L) := sorry

end range_of_a_sq_l1753_175393


namespace max_value_of_f_l1753_175382

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + Real.sin (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 1 + Real.sqrt 2 := 
sorry

end max_value_of_f_l1753_175382


namespace intersecting_lines_sum_l1753_175324

theorem intersecting_lines_sum (a b : ℝ) (h1 : 2 = (1/3) * 4 + a) (h2 : 4 = (1/3) * 2 + b) : a + b = 4 :=
sorry

end intersecting_lines_sum_l1753_175324


namespace simplify_fraction_l1753_175335

theorem simplify_fraction :
  (3^100 + 3^98) / (3^100 - 3^98) = 5 / 4 := 
by sorry

end simplify_fraction_l1753_175335


namespace remaining_sessions_l1753_175389

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end remaining_sessions_l1753_175389


namespace octagon_perimeter_l1753_175399

def side_length_meters : ℝ := 2.3
def number_of_sides : ℕ := 8
def meter_to_cm (meters : ℝ) : ℝ := meters * 100

def perimeter_cm (side_length_meters : ℝ) (number_of_sides : ℕ) : ℝ :=
  meter_to_cm side_length_meters * number_of_sides

theorem octagon_perimeter :
  perimeter_cm side_length_meters number_of_sides = 1840 :=
by
  sorry

end octagon_perimeter_l1753_175399


namespace second_valve_rate_difference_l1753_175348

theorem second_valve_rate_difference (V1 V2 : ℝ) 
  (h1 : V1 = 12000 / 120)
  (h2 : V1 + V2 = 12000 / 48) :
  V2 - V1 = 50 :=
by
  -- Since h1: V1 = 100
  -- And V1 + V2 = 250 from h2
  -- Therefore V2 = 250 - 100 = 150
  -- And V2 - V1 = 150 - 100 = 50
  sorry

end second_valve_rate_difference_l1753_175348


namespace inequality_proof_l1753_175370

variables (x y z : ℝ)

theorem inequality_proof (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by
  sorry

end inequality_proof_l1753_175370


namespace odd_function_a_increasing_function_a_l1753_175366

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem odd_function_a (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - (f x a)) → a = -1 :=
by sorry

theorem increasing_function_a (a : ℝ) :
  (∀ x : ℝ, (Real.exp x - a * Real.exp (-x)) ≥ 0) → a ∈ Set.Iic 0 :=
by sorry

end odd_function_a_increasing_function_a_l1753_175366


namespace bags_of_chips_count_l1753_175390

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end bags_of_chips_count_l1753_175390


namespace number_of_students_l1753_175311

theorem number_of_students (y c r n : ℕ) (h1 : y = 730) (h2 : c = 17) (h3 : r = 16) :
  y - r = n * c ↔ n = 42 :=
by
  have h4 : 730 - 16 = 714 := by norm_num
  have h5 : 714 / 17 = 42 := by norm_num
  sorry

end number_of_students_l1753_175311


namespace divisor_problem_l1753_175300

theorem divisor_problem :
  ∃ D : ℕ, 12401 = D * 76 + 13 ∧ D = 163 := 
by
  sorry

end divisor_problem_l1753_175300


namespace collinear_vectors_l1753_175328

-- Definitions
def a : ℝ × ℝ := (2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Proof statement
theorem collinear_vectors (x : ℝ) (h : ∃ k : ℝ, b x = k • a) : x = 3 :=
by sorry

end collinear_vectors_l1753_175328


namespace conditional_probability_l1753_175394

/-
We define the probabilities of events A and B.
-/
variables (P : Set (Set α) → ℝ)
variable {α : Type*}

-- Event A: the animal lives up to 20 years old
def A : Set α := {x | true}   -- placeholder definition

-- Event B: the animal lives up to 25 years old
def B : Set α := {x | true}   -- placeholder definition

/-
Given conditions
-/
axiom P_A : P A = 0.8
axiom P_B : P B = 0.4

/-
Proof problem to show P(B | A) = 0.5
-/
theorem conditional_probability : P (B ∩ A) / P A = 0.5 :=
by
  sorry

end conditional_probability_l1753_175394


namespace fraction_identity_l1753_175357

theorem fraction_identity (m n r t : ℚ) 
  (h₁ : m / n = 3 / 5) 
  (h₂ : r / t = 8 / 9) :
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := 
by
  sorry

end fraction_identity_l1753_175357


namespace cnc_processing_time_l1753_175384

theorem cnc_processing_time :
  (∃ (hours: ℕ), 3 * (960 / hours) = 960 / 3) → 1 * (400 / 5) = 400 / 1 :=
by
  sorry

end cnc_processing_time_l1753_175384


namespace geometric_series_sum_l1753_175360

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end geometric_series_sum_l1753_175360


namespace minimum_value_expression_l1753_175377

theorem minimum_value_expression 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_eq : 1 / a + 1 / b = 1) : 
  (∃ (x : ℝ), x = (1 / (a-1) + 9 / (b-1)) ∧ x = 6) :=
sorry

end minimum_value_expression_l1753_175377


namespace value_of_b_l1753_175354

variable (a b : ℤ)

theorem value_of_b : a = 105 ∧ a ^ 3 = 21 * 49 * 45 * b → b = 1 := by
  sorry

end value_of_b_l1753_175354


namespace seokgi_initial_money_l1753_175373

theorem seokgi_initial_money (X : ℝ) (h1 : X / 2 - X / 4 = 1250) : X = 5000 := by
  sorry

end seokgi_initial_money_l1753_175373


namespace classics_section_books_l1753_175371

-- Define the number of authors
def num_authors : Nat := 6

-- Define the number of books per author
def books_per_author : Nat := 33

-- Define the total number of books
def total_books : Nat := num_authors * books_per_author

-- Prove that the total number of books is 198
theorem classics_section_books : total_books = 198 := by
  sorry

end classics_section_books_l1753_175371


namespace product_of_numbers_l1753_175398

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
sorry

end product_of_numbers_l1753_175398


namespace avg_speed_ratio_l1753_175308

theorem avg_speed_ratio 
  (dist_tractor : ℝ) (time_tractor : ℝ) 
  (dist_car : ℝ) (time_car : ℝ) 
  (speed_factor : ℝ) :
  dist_tractor = 575 -> 
  time_tractor = 23 ->
  dist_car = 450 ->
  time_car = 5 ->
  speed_factor = 2 ->

  (dist_car / time_car) / (speed_factor * (dist_tractor / time_tractor)) = 9/5 := 
by
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  sorry

end avg_speed_ratio_l1753_175308


namespace no_real_solution_3x2_plus_9x_le_neg12_l1753_175350

/-- There are no real values of x such that 3x^2 + 9x ≤ -12. -/
theorem no_real_solution_3x2_plus_9x_le_neg12 (x : ℝ) : ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_real_solution_3x2_plus_9x_le_neg12_l1753_175350


namespace reflect_A_across_x_axis_l1753_175392

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the reflection function across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem statement: The reflection of point A across the x-axis should be (-3, -2)
theorem reflect_A_across_x_axis : reflect_x A = (-3, -2) := by
  sorry

end reflect_A_across_x_axis_l1753_175392


namespace rbcmul_div7_div89_l1753_175378

theorem rbcmul_div7_div89 {r b c : ℕ} (h : (523000 + 100 * r + 10 * b + c) % 7 = 0 ∧ (523000 + 100 * r + 10 * b + c) % 89 = 0) :
  r * b * c = 36 :=
by
  sorry

end rbcmul_div7_div89_l1753_175378


namespace common_solutions_form_segment_length_one_l1753_175314

theorem common_solutions_form_segment_length_one (a : ℝ) (h₁ : ∀ x : ℝ, x^2 - 4 * x + 2 - a ≤ 0) 
  (h₂ : ∀ x : ℝ, x^2 - 5 * x + 2 * a + 8 ≤ 0) : 
  (a = -1 ∨ a = -7 / 4) :=
by
  sorry

end common_solutions_form_segment_length_one_l1753_175314


namespace coefficients_sum_eq_zero_l1753_175310

theorem coefficients_sum_eq_zero 
  (a b c : ℝ)
  (f g h : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, g x = b * x^2 + c * x + a)
  (h3 : ∀ x, h x = c * x^2 + a * x + b)
  (h4 : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) :
  a + b + c = 0 := 
sorry

end coefficients_sum_eq_zero_l1753_175310


namespace smallest_n_divisibility_problem_l1753_175391

theorem smallest_n_divisibility_problem :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ n + 2 → n^3 - n ≠ 0 → (n^3 - n) % k = 0) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → k ∣ n^3 - n) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → ¬ k ∣ n^3 - n) ∧
    (∀ (m : ℕ), m > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ m + 2 → m^3 - m ≠ 0 → (m^3 - m) % k = 0) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → k ∣ m^3 - m) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → ¬ k ∣ m^3 - m) → n ≤ m) :=
sorry

end smallest_n_divisibility_problem_l1753_175391


namespace money_lent_years_l1753_175327

noncomputable def compound_interest_time (A P r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem money_lent_years :
  compound_interest_time 740 671.2018140589569 0.05 1 = 2 := by
  sorry

end money_lent_years_l1753_175327


namespace inclination_angle_of_line_l1753_175342

open Real

theorem inclination_angle_of_line (x y : ℝ) (h : x + y - 3 = 0) : 
  ∃ θ : ℝ, θ = 3 * π / 4 :=
by
  sorry

end inclination_angle_of_line_l1753_175342


namespace right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l1753_175367

theorem right_triangle_arithmetic_progression_is_345 (a b c : ℕ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ d, b = a + d ∧ c = a + 2 * d)
  : (a, b, c) = (3, 4, 5) :=
by
  sorry

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

noncomputable def sqrt_golden_ratio_div_2 := Real.sqrt ((1 + Real.sqrt 5) / 2)

theorem right_triangle_geometric_progression 
  (a b c : ℝ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ r, b = a * r ∧ c = a * r * r)
  : (a, b, c) = (1, sqrt_golden_ratio_div_2, golden_ratio) :=
by
  sorry

end right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l1753_175367


namespace complex_number_identity_l1753_175368

theorem complex_number_identity (z : ℂ) (h : z = 1 + (1 : ℂ) * I) : z^2 + z = 1 + 3 * I := 
sorry

end complex_number_identity_l1753_175368


namespace number_of_trees_l1753_175369

theorem number_of_trees (initial_trees planted_trees : ℕ)
  (h1 : initial_trees = 13)
  (h2 : planted_trees = 12) :
  initial_trees + planted_trees = 25 := by
  sorry

end number_of_trees_l1753_175369


namespace sculpture_and_base_height_l1753_175321

theorem sculpture_and_base_height :
  let sculpture_height_in_feet := 2
  let sculpture_height_in_inches := 10
  let base_height_in_inches := 2
  let total_height_in_inches := (sculpture_height_in_feet * 12) + sculpture_height_in_inches + base_height_in_inches
  let total_height_in_feet := total_height_in_inches / 12
  total_height_in_feet = 3 :=
by
  sorry

end sculpture_and_base_height_l1753_175321


namespace range_of_a_l1753_175365

theorem range_of_a (a : ℝ) :
  let A := {x | x^2 + 4 * x = 0}
  let B := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}
  A ∩ B = B → (a = 1 ∨ a ≤ -1) := 
by
  sorry

end range_of_a_l1753_175365


namespace inequality_holds_for_positive_y_l1753_175313

theorem inequality_holds_for_positive_y (y : ℝ) (hy : y > 0) : y^2 ≥ 2 * y - 1 :=
by
  sorry

end inequality_holds_for_positive_y_l1753_175313


namespace Penelope_daily_savings_l1753_175332

theorem Penelope_daily_savings
  (total_savings : ℝ)
  (days_in_year : ℕ)
  (h1 : total_savings = 8760)
  (h2 : days_in_year = 365) :
  total_savings / days_in_year = 24 :=
by
  sorry

end Penelope_daily_savings_l1753_175332


namespace value_of_a_l1753_175361

theorem value_of_a (a : ℝ) : (|a| - 1 = 1) ∧ (a - 2 ≠ 0) → a = -2 :=
by
  sorry

end value_of_a_l1753_175361


namespace dogs_count_l1753_175334

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l1753_175334


namespace abs_x_ge_abs_4ax_l1753_175362

theorem abs_x_ge_abs_4ax (a : ℝ) (h : ∀ x : ℝ, abs x ≥ 4 * a * x) : abs a ≤ 1 / 4 :=
sorry

end abs_x_ge_abs_4ax_l1753_175362


namespace negation_if_positive_then_square_positive_l1753_175322

theorem negation_if_positive_then_square_positive :
  (¬ (∀ x : ℝ, x > 0 → x^2 > 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end negation_if_positive_then_square_positive_l1753_175322


namespace question_l1753_175375

section

variable (x : ℝ)
variable (p q : Prop)

-- Define proposition p: ∀ x in [0,1], e^x ≥ 1
def Proposition_p : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → Real.exp x ≥ 1

-- Define proposition q: ∃ x in ℝ such that x^2 + x + 1 < 0
def Proposition_q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- The problem to prove: p ∨ q
theorem question (p q : Prop) (hp : Proposition_p) (hq : ¬ Proposition_q) : p ∨ q := by
  sorry

end

end question_l1753_175375


namespace particle_hits_origin_l1753_175305

def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x, 0 => 0
| 0, y => 0
| x+1, y+1 => 0.25 * P x (y+1) + 0.25 * P (x+1) y + 0.5 * P x y

theorem particle_hits_origin :
    ∃ m n : ℕ, m ≠ 0 ∧ m % 4 ≠ 0 ∧ P 5 5 = m / 4^n :=
sorry

end particle_hits_origin_l1753_175305


namespace lara_bouncy_house_time_l1753_175316

theorem lara_bouncy_house_time :
  let run1_time := (3 * 60 + 45) + (2 * 60 + 10) + (1 * 60 + 28)
  let door_time := 73
  let run2_time := (2 * 60 + 55) + (1 * 60 + 48) + (1 * 60 + 15)
  run1_time + door_time + run2_time = 874 := by
    let run1_time := 225 + 130 + 88
    let door_time := 73
    let run2_time := 175 + 108 + 75
    sorry

end lara_bouncy_house_time_l1753_175316


namespace no_correlation_pair_D_l1753_175302

-- Define the pairs of variables and their relationships
def pair_A : Prop := ∃ (fertilizer_applied grain_yield : ℝ), (fertilizer_applied ≠ 0 → grain_yield ≠ 0)
def pair_B : Prop := ∃ (review_time scores : ℝ), (review_time ≠ 0 → scores ≠ 0)
def pair_C : Prop := ∃ (advertising_expenses sales : ℝ), (advertising_expenses ≠ 0 → sales ≠ 0)
def pair_D : Prop := ∃ (books_sold revenue : ℕ), (revenue = books_sold * 5)

/-- Prove that pair D does not have a correlation in the context of the problem. --/
theorem no_correlation_pair_D : ¬pair_D :=
by
  sorry

end no_correlation_pair_D_l1753_175302


namespace converse_angles_complements_l1753_175372

theorem converse_angles_complements (α β : ℝ) (h : ∀γ : ℝ, α + γ = 90 ∧ β + γ = 90 → α = β) : 
  ∀ δ, α + δ = 90 ∧ β + δ = 90 → α = β :=
by 
  sorry

end converse_angles_complements_l1753_175372


namespace fifth_student_guess_l1753_175331

theorem fifth_student_guess (s1 s2 s3 s4 s5 : ℕ) 
(h1 : s1 = 100)
(h2 : s2 = 8 * s1)
(h3 : s3 = s2 - 200)
(h4 : s4 = (s1 + s2 + s3) / 3 + 25)
(h5 : s5 = s4 + s4 / 5) : 
s5 = 630 :=
sorry

end fifth_student_guess_l1753_175331


namespace average_age_l1753_175383

theorem average_age (avg_age_students : ℝ) (num_students : ℕ) (avg_age_teachers : ℝ) (num_teachers : ℕ) :
  avg_age_students = 13 → 
  num_students = 40 → 
  avg_age_teachers = 42 → 
  num_teachers = 60 → 
  (num_students * avg_age_students + num_teachers * avg_age_teachers) / (num_students + num_teachers) = 30.4 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_l1753_175383


namespace consecutive_primes_sum_square_is_prime_l1753_175356

-- Defining what it means for three numbers to be consecutive primes
def consecutive_primes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ((p < q ∧ q < r) ∨ (p < q ∧ q < r ∧ r < p) ∨ 
   (r < p ∧ p < q) ∨ (q < p ∧ p < r) ∨ 
   (q < r ∧ r < p) ∨ (r < q ∧ q < p))

-- Defining our main problem statement
theorem consecutive_primes_sum_square_is_prime :
  ∀ p q r : ℕ, consecutive_primes p q r → Nat.Prime (p^2 + q^2 + r^2) ↔ (p = 3 ∧ q = 5 ∧ r = 7) :=
by
  -- Sorry is used to skip the proof.
  sorry

end consecutive_primes_sum_square_is_prime_l1753_175356


namespace pairwise_sums_modulo_l1753_175387

theorem pairwise_sums_modulo (n : ℕ) (h : n = 2011) :
  ∃ (sums_div_3 sums_rem_1 : ℕ),
  (sums_div_3 = (n * (n - 1)) / 6) ∧
  (sums_rem_1 = (n * (n - 1)) / 6) := by
  sorry

end pairwise_sums_modulo_l1753_175387


namespace find_percentage_l1753_175304

def problem_statement (n P : ℕ) := 
  n = (P / 100) * n + 84

theorem find_percentage : ∃ P, problem_statement 100 P ∧ (P = 16) :=
by
  sorry

end find_percentage_l1753_175304


namespace penny_makes_from_cheesecakes_l1753_175336

-- Definitions based on the conditions
def slices_per_pie : ℕ := 6
def cost_per_slice : ℕ := 7
def pies_sold : ℕ := 7

-- The mathematical equivalent proof problem
theorem penny_makes_from_cheesecakes : slices_per_pie * cost_per_slice * pies_sold = 294 := by
  sorry

end penny_makes_from_cheesecakes_l1753_175336


namespace solve_a_l1753_175349

-- Defining sets A and B
def set_A (a : ℤ) : Set ℤ := {a^2, a + 1, -3}
def set_B (a : ℤ) : Set ℤ := {a - 3, 2 * a - 1, a^2 + 1}

-- Defining the condition of intersection
def intersection_condition (a : ℤ) : Prop :=
  (set_A a) ∩ (set_B a) = {-3}

-- Stating the theorem
theorem solve_a (a : ℤ) (h : intersection_condition a) : a = -1 :=
sorry

end solve_a_l1753_175349


namespace find_numbers_l1753_175344

theorem find_numbers (u v : ℝ) (h1 : u^2 + v^2 = 20) (h2 : u * v = 8) :
  (u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2) ∨ (u = -2 ∧ v = -4) ∨ (u = -4 ∧ v = -2) := by
sorry

end find_numbers_l1753_175344


namespace correct_inequality_l1753_175337

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b)

theorem correct_inequality : (1 / (a * b^2)) < (1 / (a^2 * b)) :=
by
  sorry

end correct_inequality_l1753_175337


namespace new_cost_percentage_l1753_175341

def cost (t b : ℝ) := t * b^5

theorem new_cost_percentage (t b : ℝ) : 
  let C := cost t b
  let W := cost (3 * t) (2 * b)
  W = 96 * C :=
by
  sorry

end new_cost_percentage_l1753_175341


namespace proof_of_truth_values_l1753_175315

open Classical

variables (x : ℝ)

-- Original proposition: If x = 1, then x^2 = 1.
def original_proposition : Prop := (x = 1) → (x^2 = 1)

-- Converse of the original proposition: If x^2 = 1, then x = 1.
def converse_proposition : Prop := (x^2 = 1) → (x = 1)

-- Inverse of the original proposition: If x ≠ 1, then x^2 ≠ 1.
def inverse_proposition : Prop := (x ≠ 1) → (x^2 ≠ 1)

-- Contrapositive of the original proposition: If x^2 ≠ 1, then x ≠ 1.
def contrapositive_proposition : Prop := (x^2 ≠ 1) → (x ≠ 1)

-- Negation of the original proposition: If x = 1, then x^2 ≠ 1.
def negation_proposition : Prop := (x = 1) → (x^2 ≠ 1)

theorem proof_of_truth_values :
  (original_proposition x) ∧
  (converse_proposition x = False) ∧
  (inverse_proposition x = False) ∧
  (contrapositive_proposition x) ∧
  (negation_proposition x = False) := by
  sorry

end proof_of_truth_values_l1753_175315


namespace ball_arrangement_l1753_175351

theorem ball_arrangement :
  (Nat.factorial 9) / ((Nat.factorial 2) * (Nat.factorial 3) * (Nat.factorial 4)) = 1260 := 
by
  sorry

end ball_arrangement_l1753_175351


namespace x2_plus_y2_lt_1_l1753_175326

theorem x2_plus_y2_lt_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 :=
sorry

end x2_plus_y2_lt_1_l1753_175326


namespace find_length_DC_l1753_175309

noncomputable def length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) : ℕ :=
  let DC := 29
  DC

theorem find_length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) (h6 : 20^2 + BC^2 = DC^2) : length_DC AB BC AD BD h1 h2 h3 h4 h5 = 29 :=
  by
  sorry

end find_length_DC_l1753_175309


namespace product_divisible_by_4_l1753_175333

noncomputable def biased_die_prob_divisible_by_4 : ℚ :=
  let q := 1/4  -- probability of rolling a number divisible by 3
  let p4 := 2 * q -- probability of rolling a number divisible by 4
  let p_neither := (1 - p4) * (1 - p4) -- probability of neither roll being divisible by 4
  1 - p_neither -- probability that at least one roll is divisible by 4

theorem product_divisible_by_4 :
  biased_die_prob_divisible_by_4 = 3/4 :=
by
  sorry

end product_divisible_by_4_l1753_175333


namespace units_digit_of_sum_of_squares_2010_odds_l1753_175380

noncomputable def sum_units_digit_of_squares (n : ℕ) : ℕ :=
  let units_digits := [1, 9, 5, 9, 1]
  List.foldl (λ acc x => (acc + x) % 10) 0 (List.map (λ i => units_digits.get! (i % 5)) (List.range (2 * n)))

theorem units_digit_of_sum_of_squares_2010_odds : sum_units_digit_of_squares 2010 = 0 := sorry

end units_digit_of_sum_of_squares_2010_odds_l1753_175380


namespace eggs_per_chicken_per_day_l1753_175381

theorem eggs_per_chicken_per_day (E c d : ℕ) (hE : E = 36) (hc : c = 4) (hd : d = 3) :
  (E / d) / c = 3 := by
  sorry

end eggs_per_chicken_per_day_l1753_175381


namespace alien_abduction_problem_l1753_175301

theorem alien_abduction_problem:
  ∀ (total_abducted people_taken_elsewhere people_taken_home people_returned: ℕ),
  total_abducted = 200 →
  people_taken_elsewhere = 10 →
  people_taken_home = 30 →
  people_returned = total_abducted - (people_taken_elsewhere + people_taken_home) →
  (people_returned : ℕ) / total_abducted * 100 = 80 := 
by
  intros total_abducted people_taken_elsewhere people_taken_home people_returned;
  intros h_total_abducted h_taken_elsewhere h_taken_home h_people_returned;
  sorry

end alien_abduction_problem_l1753_175301


namespace Alyssa_total_spent_l1753_175329

-- define the amounts spent on grapes and cherries
def costGrapes: ℝ := 12.08
def costCherries: ℝ := 9.85

-- define the total cost based on the given conditions
def totalCost: ℝ := costGrapes + costCherries

-- prove that the total cost equals 21.93
theorem Alyssa_total_spent:
  totalCost = 21.93 := 
  by
  -- proof to be completed
  sorry

end Alyssa_total_spent_l1753_175329


namespace marked_percentage_above_cost_l1753_175374

theorem marked_percentage_above_cost (CP SP : ℝ) (discount_percentage MP : ℝ) 
  (h1 : CP = 540) 
  (h2 : SP = 457) 
  (h3 : discount_percentage = 26.40901771336554) 
  (h4 : SP = MP * (1 - discount_percentage / 100)) : 
  ((MP - CP) / CP) * 100 = 15 :=
by
  sorry

end marked_percentage_above_cost_l1753_175374


namespace expression_value_l1753_175352

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem expression_value :
  let numerator := factorial 10
  let denominator := (1 + 2) * (3 + 4) * (5 + 6) * (7 + 8) * (9 + 10)
  numerator / denominator = 660 := by
  sorry

end expression_value_l1753_175352


namespace minimum_f_value_minimum_fraction_value_l1753_175345

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem minimum_f_value : ∃ x : ℝ, f x = 2 :=
by
  -- proof skipped, please insert proof here
  sorry

theorem minimum_fraction_value (a b : ℝ) (h : a^2 + b^2 = 2) : 
  (1 / (a^2 + 1)) + (4 / (b^2 + 1)) = 9 / 4 :=
by
  -- proof skipped, please insert proof here
  sorry

end minimum_f_value_minimum_fraction_value_l1753_175345


namespace polygon_sides_l1753_175307

theorem polygon_sides (R : ℝ) (n : ℕ) (h : R ≠ 0)
  (h_area : (1 / 2) * n * R^2 * Real.sin (360 / n * (Real.pi / 180)) = 4 * R^2) :
  n = 8 := 
by
  sorry

end polygon_sides_l1753_175307


namespace profitable_allocation_2015_l1753_175319

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end profitable_allocation_2015_l1753_175319


namespace base_conversion_l1753_175385

theorem base_conversion (k : ℕ) (h : 26 = 3*k + 2) : k = 8 := 
by 
  sorry

end base_conversion_l1753_175385


namespace abs_x_plus_one_ge_one_l1753_175340

theorem abs_x_plus_one_ge_one {x : ℝ} : |x + 1| ≥ 1 ↔ x ≤ -2 ∨ x ≥ 0 :=
by
  sorry

end abs_x_plus_one_ge_one_l1753_175340


namespace sin_pi_over_six_eq_half_l1753_175346

theorem sin_pi_over_six_eq_half : Real.sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_pi_over_six_eq_half_l1753_175346


namespace people_with_fewer_than_7_cards_l1753_175323

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l1753_175323


namespace ant_paths_l1753_175325

theorem ant_paths (n m : ℕ) : 
  ∃ paths : ℕ, paths = Nat.choose (n + m) m := sorry

end ant_paths_l1753_175325


namespace pairs_of_powers_of_two_l1753_175330

theorem pairs_of_powers_of_two (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ a : ℕ, m + n = 2^a) (h4 : ∃ b : ℕ, mn + 1 = 2^b) :
  (∃ a : ℕ, m = 2^a - 1 ∧ n = 1) ∨ 
  (∃ a : ℕ, m = 2^(a-1) + 1 ∧ n = 2^(a-1) - 1) :=
sorry

end pairs_of_powers_of_two_l1753_175330


namespace total_cups_sold_l1753_175395

theorem total_cups_sold (plastic_cups : ℕ) (ceramic_cups : ℕ) (total_sold : ℕ) :
  plastic_cups = 284 ∧ ceramic_cups = 284 → total_sold = 568 :=
by
  intros h
  cases h
  sorry

end total_cups_sold_l1753_175395


namespace total_games_played_l1753_175306

def games_attended : ℕ := 14
def games_missed : ℕ := 25

theorem total_games_played : games_attended + games_missed = 39 :=
by
  sorry

end total_games_played_l1753_175306


namespace dogs_food_consumption_l1753_175339

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end dogs_food_consumption_l1753_175339


namespace geometric_sequence_first_term_l1753_175388

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 72) : a = 4.5 := by
  sorry

end geometric_sequence_first_term_l1753_175388


namespace total_letters_in_names_is_33_l1753_175347

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l1753_175347


namespace value_of_expression_l1753_175396

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 :=
by
  -- The proof will be filled here; it's currently skipped using 'sorry'
  sorry

end value_of_expression_l1753_175396


namespace solve_for_m_l1753_175355

theorem solve_for_m (m : ℝ) :
  (1 * m + (3 + m) * 2 = 0) → m = -2 :=
by
  sorry

end solve_for_m_l1753_175355


namespace percentage_of_y_l1753_175312

theorem percentage_of_y (x y : ℝ) (h1 : x = 4 * y) (h2 : 0.80 * x = (P / 100) * y) : P = 320 :=
by
  -- Proof goes here
  sorry

end percentage_of_y_l1753_175312


namespace Chang_solution_A_amount_l1753_175379

def solution_alcohol_content (A B : ℝ) (x : ℝ) : ℝ :=
  0.16 * x + 0.10 * (x + 500)

theorem Chang_solution_A_amount (x : ℝ) :
  solution_alcohol_content 0.16 0.10 x = 76 → x = 100 :=
by
  intro h
  sorry

end Chang_solution_A_amount_l1753_175379


namespace problem_proof_l1753_175320

open Real

noncomputable def angle_B (A C : ℝ) : ℝ := π / 3

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  (1/2) * a * c * (sqrt 3 / 2)

theorem problem_proof (A B C a b c : ℝ)
  (h1 : 2 * cos A * cos C * (tan A * tan C - 1) = 1)
  (h2 : a + c = sqrt 15)
  (h3 : b = sqrt 3)
  (h4 : B = π / 3) :
  (B = angle_B A C) ∧ 
  (area_triangle a b c = sqrt 3) :=
by
  sorry

end problem_proof_l1753_175320


namespace Q_div_P_l1753_175358

theorem Q_div_P (P Q : ℤ) (h : ∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 →
  P / (x + 7) + Q / (x * (x - 6)) = (x^2 - x + 15) / (x^3 + x^2 - 42 * x)) :
  Q / P = 7 :=
sorry

end Q_div_P_l1753_175358


namespace set_intersection_eq_l1753_175318

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5}
def ComplementU (S : Set ℕ) : Set ℕ := U \ S

theorem set_intersection_eq : 
  A ∩ (ComplementU B) = {1, 3} := 
by
  sorry

end set_intersection_eq_l1753_175318


namespace girls_count_l1753_175363

variable (B G : ℕ)

theorem girls_count (h1: B = 387) (h2: G = (B + (54 * B) / 100)) : G = 596 := 
by 
  sorry

end girls_count_l1753_175363


namespace addition_of_decimals_l1753_175376

theorem addition_of_decimals :
  0.9 + 0.99 = 1.89 :=
by
  sorry

end addition_of_decimals_l1753_175376


namespace inequality_proof_l1753_175303

variable (x y z : ℝ)

theorem inequality_proof (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
  ≥ Real.sqrt (3 / 2 * (x + y + z)) :=
sorry

end inequality_proof_l1753_175303


namespace heart_then_club_probability_l1753_175397

theorem heart_then_club_probability :
  (13 / 52) * (13 / 51) = 13 / 204 := by
  sorry

end heart_then_club_probability_l1753_175397


namespace total_biscuits_l1753_175317

-- Define the number of dogs and biscuits per dog
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem stating the total number of biscuits needed
theorem total_biscuits : num_dogs * biscuits_per_dog = 6 := by
  -- sorry to skip the proof
  sorry

end total_biscuits_l1753_175317


namespace min_value_f_in_interval_l1753_175338

def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 1

theorem min_value_f_in_interval : 
  ∃ x ∈ (Set.Icc (-1 : ℝ) 1), f x = -1 :=
by
  sorry


end min_value_f_in_interval_l1753_175338


namespace T_shape_perimeter_l1753_175353

/-- Two rectangles each measuring 3 inch × 5 inch are placed to form the letter T.
The overlapping area between the two rectangles is 1.5 inch. -/
theorem T_shape_perimeter:
  let l := 5 -- inches
  let w := 3 -- inches
  let overlap := 1.5 -- inches
  -- perimeter of one rectangle
  let P := 2 * (l + w)
  -- total perimeter accounting for overlap
  let total_perimeter := 2 * P - 2 * overlap
  total_perimeter = 29 :=
by
  sorry

end T_shape_perimeter_l1753_175353


namespace mileage_on_city_streets_l1753_175359

-- Defining the given conditions
def distance_on_highways : ℝ := 210
def mileage_on_highways : ℝ := 35
def total_gas_used : ℝ := 9
def distance_on_city_streets : ℝ := 54

-- Proving the mileage on city streets
theorem mileage_on_city_streets :
  ∃ x : ℝ, 
    (distance_on_highways / mileage_on_highways + distance_on_city_streets / x = total_gas_used)
    ∧ x = 18 :=
by
  sorry

end mileage_on_city_streets_l1753_175359


namespace percentage_of_360_equals_115_2_l1753_175364

theorem percentage_of_360_equals_115_2 (p : ℝ) (h : (p / 100) * 360 = 115.2) : p = 32 :=
by
  sorry

end percentage_of_360_equals_115_2_l1753_175364


namespace eggs_in_fridge_l1753_175343

theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) (eggs_used : ℕ) (eggs_in_fridge : ℕ)
  (h1 : total_eggs = 60)
  (h2 : eggs_per_cake = 5)
  (h3 : num_cakes = 10)
  (h4 : eggs_used = eggs_per_cake * num_cakes)
  (h5 : eggs_in_fridge = total_eggs - eggs_used) :
  eggs_in_fridge = 10 :=
by
  sorry

end eggs_in_fridge_l1753_175343


namespace sum_remainder_of_consecutive_odds_l1753_175386

theorem sum_remainder_of_consecutive_odds :
  (11075 + 11077 + 11079 + 11081 + 11083 + 11085 + 11087) % 14 = 7 :=
by
  -- Adding the proof here
  sorry

end sum_remainder_of_consecutive_odds_l1753_175386
