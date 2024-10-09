import Mathlib

namespace sampling_method_is_systematic_l1093_109395

-- Define the conditions
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  required_student_num : Nat

-- Define our specific problem's conditions
def problem_conditions : Grade :=
  { num_classes := 12, students_per_class := 50, required_student_num := 14 }

-- State the theorem
theorem sampling_method_is_systematic (G : Grade) (h1 : G.num_classes = 12) (h2 : G.students_per_class = 50) (h3 : G.required_student_num = 14) : 
  "Systematic sampling" = "Systematic sampling" :=
by
  sorry

end sampling_method_is_systematic_l1093_109395


namespace correct_A_correct_B_intersection_A_B_complement_B_l1093_109323

noncomputable def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem correct_A : A = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem correct_B : B = {x : ℝ | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
by
  sorry

theorem complement_B : (Bᶜ) = {x : ℝ | x < 1 ∨ x > 4} :=
by
  sorry

end correct_A_correct_B_intersection_A_B_complement_B_l1093_109323


namespace electronics_weight_l1093_109344

-- Define the initial conditions and the solution we want to prove.
theorem electronics_weight (B C E : ℕ) (k : ℕ) 
  (h1 : B = 7 * k) 
  (h2 : C = 4 * k) 
  (h3 : E = 3 * k) 
  (h4 : (B : ℚ) / (C - 8 : ℚ) = 2 * (B : ℚ) / (C : ℚ)) :
  E = 12 := 
sorry

end electronics_weight_l1093_109344


namespace complement_U_A_l1093_109340

-- Define the sets U and A
def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

-- State the theorem
theorem complement_U_A :
  U \ A = {0} :=
sorry

end complement_U_A_l1093_109340


namespace trig_identity_nec_but_not_suff_l1093_109326

open Real

theorem trig_identity_nec_but_not_suff (α β : ℝ) (k : ℤ) :
  (α + β = 2 * k * π + π / 6) → (sin α * cos β + cos α * sin β = 1 / 2) := by
  sorry

end trig_identity_nec_but_not_suff_l1093_109326


namespace total_votes_l1093_109350

theorem total_votes (votes_brenda : ℕ) (total_votes : ℕ) 
  (h1 : votes_brenda = 50) 
  (h2 : votes_brenda = (1/4 : ℚ) * total_votes) : 
  total_votes = 200 :=
by 
  sorry

end total_votes_l1093_109350


namespace square_diff_problem_l1093_109384

theorem square_diff_problem
  (x : ℤ)
  (h : x^2 = 9801) :
  (x + 3) * (x - 3) = 9792 := 
by
  -- proof would go here
  sorry

end square_diff_problem_l1093_109384


namespace shaded_area_is_one_third_l1093_109390

noncomputable def fractional_shaded_area : ℕ → ℚ
| 0 => 1 / 4
| n + 1 => (1 / 4) * fractional_shaded_area n

theorem shaded_area_is_one_third : (∑' n, fractional_shaded_area n) = 1 / 3 := 
sorry

end shaded_area_is_one_third_l1093_109390


namespace functional_equation_solution_l1093_109336

noncomputable def func_form (f : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, (α = 1 ∨ α = -1 ∨ α = 0) ∧ (∀ x, f x = α * x + β ∨ f x = α * x ^ 3 + β)

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a * b ^ 2 + b * c ^ 2 + c * a ^ 2) - f (a ^ 2 * b + b ^ 2 * c + c ^ 2 * a)) →
  func_form f :=
sorry

end functional_equation_solution_l1093_109336


namespace point_on_graph_l1093_109345

def f (x : ℝ) : ℝ := -2 * x + 3

theorem point_on_graph (x y : ℝ) : 
  ( (x = 1 ∧ y = 1) ↔ y = f x ) :=
by 
  sorry

end point_on_graph_l1093_109345


namespace intersect_A_B_l1093_109317

def A : Set ℝ := {x | 1/x < 1}
def B : Set ℝ := {-1, 0, 1, 2}
def intersection_result : Set ℝ := {-1, 2}

theorem intersect_A_B : A ∩ B = intersection_result :=
by
  sorry

end intersect_A_B_l1093_109317


namespace total_birds_on_fence_l1093_109337

-- Definitions for the problem conditions
def initial_birds : ℕ := 12
def new_birds : ℕ := 8

-- Theorem to state that the total number of birds on the fence is 20
theorem total_birds_on_fence : initial_birds + new_birds = 20 :=
by
  -- Skip the proof as required
  sorry

end total_birds_on_fence_l1093_109337


namespace slope_of_line_determined_by_solutions_l1093_109332

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l1093_109332


namespace number_of_books_in_shipment_l1093_109385

theorem number_of_books_in_shipment
  (T : ℕ)                   -- The total number of books
  (displayed_ratio : ℚ)     -- Fraction of books displayed
  (remaining_books : ℕ)     -- Number of books in the storeroom
  (h1 : displayed_ratio = 0.3)
  (h2 : remaining_books = 210)
  (h3 : (1 - displayed_ratio) * T = remaining_books) :
  T = 300 := 
by
  -- Add your proof here
  sorry

end number_of_books_in_shipment_l1093_109385


namespace zack_travel_countries_l1093_109370

theorem zack_travel_countries (G J P Z : ℕ) 
  (hG : G = 6)
  (hJ : J = G / 2)
  (hP : P = 3 * J)
  (hZ : Z = 2 * P) :
  Z = 18 := by
  sorry

end zack_travel_countries_l1093_109370


namespace digit_sum_divisible_by_9_l1093_109306

theorem digit_sum_divisible_by_9 (n : ℕ) (h : n < 10) : 
  (8 + 6 + 5 + n + 7 + 4 + 3 + 2) % 9 = 0 ↔ n = 1 := 
by sorry 

end digit_sum_divisible_by_9_l1093_109306


namespace find_unsuitable_activity_l1093_109334

-- Definitions based on the conditions
def suitable_for_questionnaire (activity : String) : Prop :=
  activity = "D: The radiation produced by various mobile phones during use"

-- Question transformed into a statement to prove in Lean
theorem find_unsuitable_activity :
  suitable_for_questionnaire "D: The radiation produced by various mobile phones during use" :=
by
  sorry

end find_unsuitable_activity_l1093_109334


namespace inequality_proof_l1093_109325

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 / (a^3 + b^3 + c^3)) ≤ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc))) ∧ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc)) ≤ (1 / (abc))) := 
sorry

end inequality_proof_l1093_109325


namespace cara_total_bread_l1093_109315

variable (L B : ℕ)  -- Let L and B be the amount of bread for lunch and breakfast, respectively

theorem cara_total_bread :
  (dinner = 240) → 
  (dinner = 8 * L) → 
  (dinner = 6 * B) → 
  (total_bread = dinner + L + B) → 
  total_bread = 310 :=
by
  intros
  -- Here you'd begin your proof, implementing each given condition
  sorry

end cara_total_bread_l1093_109315


namespace farmer_sowed_buckets_l1093_109309

-- Define the initial and final buckets of seeds
def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6.00

-- The goal: prove the number of buckets sowed is 2.75
theorem farmer_sowed_buckets : initial_buckets - final_buckets = 2.75 := by
  sorry

end farmer_sowed_buckets_l1093_109309


namespace inequality_problem_l1093_109373

variable {a b c : ℝ}

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_problem_l1093_109373


namespace knives_more_than_forks_l1093_109378

variable (F K S T : ℕ)
variable (x : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  (F = 6) ∧ 
  (K = F + x) ∧ 
  (S = 2 * K) ∧
  (T = F / 2)

-- Total cutlery added
def total_cutlery_added : Prop :=
  (F + 2) + (K + 2) + (S + 2) + (T + 2) = 62

-- Prove that x = 9
theorem knives_more_than_forks :
  initial_conditions F K S T x →
  total_cutlery_added F K S T →
  x = 9 := 
by
  sorry

end knives_more_than_forks_l1093_109378


namespace geometric_sequence_b_value_l1093_109386

theorem geometric_sequence_b_value (r b : ℝ) (h1 : 120 * r = b) (h2 : b * r = 27 / 16) (hb_pos : b > 0) : b = 15 :=
sorry

end geometric_sequence_b_value_l1093_109386


namespace problem_l1093_109374

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem (f : ℝ → ℝ) (h : isOddFunction f) : 
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 :=
by
  sorry

end problem_l1093_109374


namespace intersection_complement_l1093_109307

open Set

theorem intersection_complement (U A B : Set ℕ) (hU : U = {x | x ≤ 6}) (hA : A = {1, 3, 5}) (hB : B = {4, 5, 6}) :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end intersection_complement_l1093_109307


namespace compare_y1_y2_l1093_109324

theorem compare_y1_y2 (m y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 2*(-1) + m) 
  (h2 : y2 = 2^2 - 2*2 + m) : 
  y1 > y2 := 
sorry

end compare_y1_y2_l1093_109324


namespace largest_integer_le_zero_l1093_109362

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero (x k : ℝ) (h1 : f x = 0) (h2 : 2 < x) (h3 : x < 3) : k ≤ x ∧ k = 2 :=
by
  sorry

end largest_integer_le_zero_l1093_109362


namespace S_4n_l1093_109353

variable {a : ℕ → ℕ}
variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (r : ℝ)
variable (a1 : ℝ)

-- Conditions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * r
axiom positive_terms : ∀ n, 0 < a n
axiom sum_n : S n = a1 * (1 - r^n) / (1 - r)
axiom sum_3n : S (3 * n) = 14
axiom sum_n_value : S n = 2

-- Theorem
theorem S_4n : S (4 * n) = 30 :=
sorry

end S_4n_l1093_109353


namespace valid_outfit_combinations_l1093_109342

theorem valid_outfit_combinations (shirts pants hats shoes : ℕ) (colors : ℕ) 
  (h₁ : shirts = 6) (h₂ : pants = 6) (h₃ : hats = 6) (h₄ : shoes = 6) (h₅ : colors = 6) :
  ∀ (valid_combinations : ℕ),
  (valid_combinations = colors * (colors - 1) * (colors - 2) * (colors - 3)) → valid_combinations = 360 := 
by
  intros valid_combinations h_valid_combinations
  sorry

end valid_outfit_combinations_l1093_109342


namespace minimum_distance_sum_squared_l1093_109398

variable (P : ℝ × ℝ)
variable (F₁ F₂ : ℝ × ℝ)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

def distance_squared (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2

theorem minimum_distance_sum_squared
  (hP : on_ellipse P)
  (hF1 : F₁ = (2, 0) ∨ F₁ = (-2, 0)) -- Assuming standard position of foci
  (hF2 : F₂ = (2, 0) ∨ F₂ = (-2, 0)) :
  ∃ P : ℝ × ℝ, on_ellipse P ∧ F₁ ≠ F₂ → distance_squared P F₁ + distance_squared P F₂ = 8 :=
by
  sorry

end minimum_distance_sum_squared_l1093_109398


namespace simplify_expression_l1093_109367

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x ^ 2 - 1) / (x ^ 2 + 2 * x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_expression_l1093_109367


namespace reciprocal_of_neg_2023_l1093_109303

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l1093_109303


namespace point_on_x_axis_l1093_109399

theorem point_on_x_axis (m : ℤ) (hx : 2 + m = 0) : (m - 3, 2 + m) = (-5, 0) :=
by sorry

end point_on_x_axis_l1093_109399


namespace largest_divisor_of_Pn_for_even_n_l1093_109343

def P (n : ℕ) : ℕ := 
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_Pn_for_even_n : 
  ∀ (n : ℕ), (0 < n ∧ n % 2 = 0) → ∃ d, d = 15 ∧ d ∣ P n :=
by
  intro n h
  sorry

end largest_divisor_of_Pn_for_even_n_l1093_109343


namespace remainder_of_largest_divided_by_next_largest_l1093_109363

/-
  Conditions:
  Let a = 10, b = 11, c = 12, d = 13.
  The largest number is d (13) and the next largest number is c (12).

  Question:
  What is the remainder when the largest number is divided by the next largest number?

  Answer:
  The remainder is 1.
-/

theorem remainder_of_largest_divided_by_next_largest :
  let a := 10 
  let b := 11
  let c := 12
  let d := 13
  d % c = 1 :=
by
  sorry

end remainder_of_largest_divided_by_next_largest_l1093_109363


namespace ralph_socks_l1093_109360

theorem ralph_socks
  (x y w z : ℕ)
  (h1 : x + y + w + z = 15)
  (h2 : x + 2 * y + 3 * w + 4 * z = 36)
  (hx : x ≥ 1) (hy : y ≥ 1) (hw : w ≥ 1) (hz : z ≥ 1) :
  x = 5 :=
sorry

end ralph_socks_l1093_109360


namespace product_of_four_consecutive_integers_divisible_by_twelve_l1093_109321

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l1093_109321


namespace factorize_expression_l1093_109331

theorem factorize_expression (a b : ℝ) : 
  a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 :=
by {
  sorry
}

end factorize_expression_l1093_109331


namespace binom_1000_1000_and_999_l1093_109349

theorem binom_1000_1000_and_999 :
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) :=
by
  sorry

end binom_1000_1000_and_999_l1093_109349


namespace problem_statement_l1093_109364

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)

theorem problem_statement : ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by 
  sorry

end problem_statement_l1093_109364


namespace investigate_local_extrema_l1093_109366

noncomputable def f (x1 x2 : ℝ) : ℝ :=
  3 * x1^2 * x2 - x1^3 - (4 / 3) * x2^3

def is_local_maximum (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∀ (x y : ℝ × ℝ), dist x c < ε → f x.1 x.2 ≤ f c.1 c.2

def is_saddle_point (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∃ (x1 y1 x2 y2 : ℝ × ℝ),
    dist x1 c < ε ∧ dist y1 c < ε ∧ dist x2 c < ε ∧ dist y2 c < ε ∧
    (f x1.1 x1.2 > f c.1 c.2 ∧ f y1.1 y1.2 < f c.1 c.2) ∧
    (f x2.1 x2.2 < f c.1 c.2 ∧ f y2.1 y2.2 > f c.1 c.2)

theorem investigate_local_extrema :
  is_local_maximum f (6, 3) ∧ is_saddle_point f (0, 0) :=
sorry

end investigate_local_extrema_l1093_109366


namespace solve_diff_l1093_109388

-- Definitions based on conditions
def equation (e y : ℝ) : Prop := y^2 + e^2 = 3 * e * y + 1

theorem solve_diff (e a b : ℝ) (h1 : equation e a) (h2 : equation e b) (h3 : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 4) := 
sorry

end solve_diff_l1093_109388


namespace general_term_formaula_sum_of_seq_b_l1093_109322

noncomputable def seq_a (n : ℕ) := 2 * n + 1

noncomputable def seq_b (n : ℕ) := 1 / ((seq_a n)^2 - 1)

noncomputable def sum_seq_a (n : ℕ) := (Finset.range n).sum seq_a

noncomputable def sum_seq_b (n : ℕ) := (Finset.range n).sum seq_b

theorem general_term_formaula (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  seq_a n = 2 * n + 1 :=
by
  intros
  sorry

theorem sum_of_seq_b (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  sum_seq_b n = n / (4 * (n + 1)) :=
by
  intros
  sorry

end general_term_formaula_sum_of_seq_b_l1093_109322


namespace two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l1093_109308

def R (n : ℕ) : ℕ := 
  let remainders := List.range' 2 11 |>.map (λ k => n % k)
  remainders.sum

theorem two_digit_integers_satisfy_R_n_eq_R_n_plus_2 :
  let two_digit_numbers := List.range' 10 89
  (two_digit_numbers.filter (λ n => R n = R (n + 2))).length = 2 := 
by
  sorry

end two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l1093_109308


namespace lyle_notebook_cost_l1093_109361

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end lyle_notebook_cost_l1093_109361


namespace tan_tan_lt_half_l1093_109354

noncomputable def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_tan_lt_half (a b c α β : ℝ) (h1: a + b < 3 * c) (h2: tan_half α * tan_half β = (a + b - c) / (a + b + c)) :
  tan_half α * tan_half β < 1 / 2 := 
sorry

end tan_tan_lt_half_l1093_109354


namespace slower_train_speed_l1093_109376

theorem slower_train_speed (length_train : ℕ) (speed_fast : ℕ) (time_seconds : ℕ) (distance_meters : ℕ): 
  (length_train = 150) → 
  (speed_fast = 46) → 
  (time_seconds = 108) → 
  (distance_meters = 300) → 
  (distance_meters = (speed_fast - speed_slow) * 5 / 18 * time_seconds) → 
  speed_slow = 36 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end slower_train_speed_l1093_109376


namespace apples_needed_per_month_l1093_109327

theorem apples_needed_per_month (chandler_apples_per_week : ℕ) (lucy_apples_per_week : ℕ) (weeks_per_month : ℕ)
  (h1 : chandler_apples_per_week = 23)
  (h2 : lucy_apples_per_week = 19)
  (h3 : weeks_per_month = 4) :
  (chandler_apples_per_week + lucy_apples_per_week) * weeks_per_month = 168 :=
by
  sorry

end apples_needed_per_month_l1093_109327


namespace sum_of_coefficients_l1093_109347

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) 
  (h : (1 + 2*x)^7 = a + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5 + a₆*(1 - x)^6 + a₇*(1 - x)^7) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 1 :=
by 
  sorry

end sum_of_coefficients_l1093_109347


namespace remainder_of_N_l1093_109328

-- Definition of the sequence constraints
def valid_sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ (∀ i, a i < 512) ∧ (∀ k, 1 ≤ k → k ≤ 9 → ∃ m, 0 ≤ m ∧ m ≤ k - 1 ∧ ((a k - 2 * a m) * (a k - 2 * a m - 1) = 0))

-- Defining N as the number of sequences that are valid.
noncomputable def N : ℕ :=
  Nat.factorial 10 - 2^9

-- The goal is to prove that N mod 1000 is 288
theorem remainder_of_N : N % 1000 = 288 :=
  sorry

end remainder_of_N_l1093_109328


namespace correct_calculation_l1093_109392

theorem correct_calculation (x : ℝ) (h : 5.46 - x = 3.97) : 5.46 + x = 6.95 := by
  sorry

end correct_calculation_l1093_109392


namespace find_positive_integer_solutions_l1093_109369

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

end find_positive_integer_solutions_l1093_109369


namespace final_price_after_discounts_l1093_109382

theorem final_price_after_discounts (original_price : ℝ)
  (first_discount_pct : ℝ) (second_discount_pct : ℝ) (third_discount_pct : ℝ) :
  original_price = 200 → 
  first_discount_pct = 0.40 → 
  second_discount_pct = 0.20 → 
  third_discount_pct = 0.10 → 
  (original_price * (1 - first_discount_pct) * (1 - second_discount_pct) * (1 - third_discount_pct) = 86.40) := 
by
  intros
  sorry

end final_price_after_discounts_l1093_109382


namespace total_number_of_students_l1093_109377

-- Statement translating the problem conditions and conclusion
theorem total_number_of_students (rank_from_right rank_from_left total : ℕ) 
  (h_right : rank_from_right = 13) 
  (h_left : rank_from_left = 8) 
  (total_eq : total = rank_from_right + rank_from_left - 1) : 
  total = 20 := 
by 
  -- Proof is skipped
  sorry

end total_number_of_students_l1093_109377


namespace total_distance_covered_l1093_109368

theorem total_distance_covered :
  let speed_fox := 50       -- km/h
  let speed_rabbit := 60    -- km/h
  let speed_deer := 80      -- km/h
  let time_hours := 2       -- hours
  let distance_fox := speed_fox * time_hours
  let distance_rabbit := speed_rabbit * time_hours
  let distance_deer := speed_deer * time_hours
  distance_fox + distance_rabbit + distance_deer = 380 := by
sorry

end total_distance_covered_l1093_109368


namespace full_time_score_l1093_109329

variables (x : ℕ)

def half_time_score_visitors := 14
def half_time_score_home := 9
def visitors_full_time_score := half_time_score_visitors + x
def home_full_time_score := half_time_score_home + 2 * x
def home_team_win_by_one := home_full_time_score = visitors_full_time_score + 1

theorem full_time_score 
  (h : home_team_win_by_one) : 
  visitors_full_time_score = 20 ∧ home_full_time_score = 21 :=
by
  sorry

end full_time_score_l1093_109329


namespace harry_blue_weights_l1093_109310

theorem harry_blue_weights (B : ℕ) 
  (h1 : 2 * B + 17 = 25) : B = 4 :=
by {
  -- proof code here
  sorry
}

end harry_blue_weights_l1093_109310


namespace James_age_l1093_109320

-- Defining variables
variables (James John Tim : ℕ)
variables (h1 : James + 12 = John)
variables (h2 : Tim + 5 = 2 * John)
variables (h3 : Tim = 79)

-- Statement to prove James' age
theorem James_age : James = 25 :=
by {
  sorry
}

end James_age_l1093_109320


namespace find_g_at_4_l1093_109305

def g (x : ℝ) : ℝ := sorry

theorem find_g_at_4 (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 5.5 :=
by
  sorry

end find_g_at_4_l1093_109305


namespace number_of_sections_l1093_109372

theorem number_of_sections (pieces_per_section : ℕ) (cost_per_piece : ℕ) (total_cost : ℕ)
  (h1 : pieces_per_section = 30)
  (h2 : cost_per_piece = 2)
  (h3 : total_cost = 480) :
  total_cost / (pieces_per_section * cost_per_piece) = 8 := by
  sorry

end number_of_sections_l1093_109372


namespace milton_zoology_books_l1093_109351

variable (Z : ℕ)
variable (total_books botany_books : ℕ)

theorem milton_zoology_books (h1 : total_books = 960)
    (h2 : botany_books = 7 * Z)
    (h3 : total_books = Z + botany_books) :
    Z = 120 := by
  sorry

end milton_zoology_books_l1093_109351


namespace cost_of_children_ticket_l1093_109330

theorem cost_of_children_ticket (total_cost : ℝ) (cost_adult_ticket : ℝ) (num_total_tickets : ℕ) (num_adult_tickets : ℕ) (cost_children_ticket : ℝ) :
  total_cost = 119 ∧ cost_adult_ticket = 21 ∧ num_total_tickets = 7 ∧ num_adult_tickets = 4 -> cost_children_ticket = 11.67 :=
by
  intros h
  sorry

end cost_of_children_ticket_l1093_109330


namespace compound_interest_interest_l1093_109387

theorem compound_interest_interest :
  let P := 2000
  let r := 0.05
  let n := 5
  let A := P * (1 + r)^n
  let interest := A - P
  interest = 552.56 := by
  sorry

end compound_interest_interest_l1093_109387


namespace new_average_is_10_5_l1093_109339

-- define the conditions
def average_of_eight_numbers (numbers : List ℝ) : Prop :=
  numbers.length = 8 ∧ (numbers.sum / 8) = 8

def add_four_to_five_numbers (numbers : List ℝ) (new_numbers : List ℝ) : Prop :=
  new_numbers = (numbers.take 5).map (λ x => x + 4) ++ numbers.drop 5

-- state the theorem
theorem new_average_is_10_5 (numbers new_numbers : List ℝ) 
  (h1 : average_of_eight_numbers numbers)
  (h2 : add_four_to_five_numbers numbers new_numbers) :
  (new_numbers.sum / 8) = 10.5 := 
by 
  sorry

end new_average_is_10_5_l1093_109339


namespace chemistry_textbook_weight_l1093_109393

theorem chemistry_textbook_weight (G C : ℝ) (h1 : G = 0.62) (h2 : C = G + 6.5) : C = 7.12 :=
by
  sorry

end chemistry_textbook_weight_l1093_109393


namespace largest_three_digit_multiple_of_17_l1093_109352

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l1093_109352


namespace degree_of_monomial_l1093_109380

def degree (m : String) : Nat :=  -- Placeholder type, replace with appropriate type that represents a monomial
  sorry  -- Logic to compute the degree would go here, if required for full implementation

theorem degree_of_monomial : degree "-(3/5) * a * b^2" = 3 := by
  sorry

end degree_of_monomial_l1093_109380


namespace ellipse_sum_l1093_109311

theorem ellipse_sum (h k a b : ℝ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 6) (b_val : b = 2) : h + k + a + b = 6 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l1093_109311


namespace cookies_per_child_l1093_109301

def num_adults : ℕ := 4
def num_children : ℕ := 6
def cookies_jar1 : ℕ := 240
def cookies_jar2 : ℕ := 360
def cookies_jar3 : ℕ := 480

def fraction_eaten_jar1 : ℚ := 1 / 4
def fraction_eaten_jar2 : ℚ := 1 / 3
def fraction_eaten_jar3 : ℚ := 1 / 5

theorem cookies_per_child :
  let eaten_jar1 := fraction_eaten_jar1 * cookies_jar1
  let eaten_jar2 := fraction_eaten_jar2 * cookies_jar2
  let eaten_jar3 := fraction_eaten_jar3 * cookies_jar3
  let remaining_jar1 := cookies_jar1 - eaten_jar1
  let remaining_jar2 := cookies_jar2 - eaten_jar2
  let remaining_jar3 := cookies_jar3 - eaten_jar3
  let total_remaining_cookies := remaining_jar1 + remaining_jar2 + remaining_jar3
  let cookies_each_child := total_remaining_cookies / num_children
  cookies_each_child = 134 := by
  sorry

end cookies_per_child_l1093_109301


namespace tino_jellybeans_l1093_109313

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l1093_109313


namespace puppies_brought_in_correct_l1093_109375

-- Define the initial number of puppies in the shelter
def initial_puppies: Nat := 2

-- Define the number of puppies adopted per day
def puppies_adopted_per_day: Nat := 4

-- Define the number of days over which the puppies are adopted
def adoption_days: Nat := 9

-- Define the total number of puppies adopted after the given days
def total_puppies_adopted: Nat := puppies_adopted_per_day * adoption_days

-- Define the number of puppies brought in
def puppies_brought_in: Nat := total_puppies_adopted - initial_puppies

-- Prove that the number of puppies brought in is 34
theorem puppies_brought_in_correct: puppies_brought_in = 34 := by
  -- proof omitted, filled with sorry to skip the proof
  sorry

end puppies_brought_in_correct_l1093_109375


namespace total_candy_l1093_109314

/-- Bobby ate 26 pieces of candy initially. -/
def initial_candy : ℕ := 26

/-- Bobby ate 17 more pieces of candy thereafter. -/
def more_candy : ℕ := 17

/-- Prove that the total number of pieces of candy Bobby ate is 43. -/
theorem total_candy : initial_candy + more_candy = 43 := by
  -- The total number of candies should be 26 + 17 which is 43
  sorry

end total_candy_l1093_109314


namespace solution_set_abs_inequality_l1093_109319

theorem solution_set_abs_inequality :
  { x : ℝ | |x - 2| - |2 * x - 1| > 0 } = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_abs_inequality_l1093_109319


namespace possible_values_of_a_l1093_109396

variables {a b k : ℤ}

def sum_distances (a : ℤ) (k : ℤ) : ℤ :=
  (a - k).natAbs + (a - (k + 1)).natAbs + (a - (k + 2)).natAbs +
  (a - (k + 3)).natAbs + (a - (k + 4)).natAbs + (a - (k + 5)).natAbs +
  (a - (k + 6)).natAbs + (a - (k + 7)).natAbs + (a - (k + 8)).natAbs +
  (a - (k + 9)).natAbs + (a - (k + 10)).natAbs

theorem possible_values_of_a :
  sum_distances a k = 902 →
  sum_distances b k = 374 →
  a + b = 98 →
  a = 25 ∨ a = 107 ∨ a = -9 :=
sorry

end possible_values_of_a_l1093_109396


namespace tilted_rectangle_l1093_109358

theorem tilted_rectangle (VWYZ : Type) (YW ZV : ℝ) (ZY VW : ℝ) (W_above_horizontal : ℝ) (Z_height : ℝ) (x : ℝ) :
  YW = 100 → ZV = 100 → ZY = 150 → VW = 150 → W_above_horizontal = 20 → Z_height = (100 + x) →
  x = 67 :=
by
  sorry

end tilted_rectangle_l1093_109358


namespace galaxy_destruction_probability_l1093_109316

theorem galaxy_destruction_probability :
  let m := 45853
  let n := 65536
  m + n = 111389 :=
by
  sorry

end galaxy_destruction_probability_l1093_109316


namespace find_a_b_f_inequality_l1093_109356

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

-- a == 1 and b == 1 from the given conditions
theorem find_a_b (e : ℝ) (h_e : e = Real.exp 1) (b : ℝ) (a : ℝ) 
  (h_tangent : ∀ x, f x a = (e - 2) * x + b → a = 1 ∧ b = 1) : a = 1 ∧ b = 1 :=
sorry

-- prove f(x) > x^2 + 4x - 14 for x >= 0
theorem f_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ x : ℝ, 0 ≤ x → f x 1 > x^2 + 4 * x - 14 :=
sorry

end find_a_b_f_inequality_l1093_109356


namespace calen_pencils_loss_l1093_109357

theorem calen_pencils_loss
  (P_Candy : ℕ)
  (P_Caleb : ℕ)
  (P_Calen_original : ℕ)
  (P_Calen_after_loss : ℕ)
  (h1 : P_Candy = 9)
  (h2 : P_Caleb = 2 * P_Candy - 3)
  (h3 : P_Calen_original = P_Caleb + 5)
  (h4 : P_Calen_after_loss = 10) :
  P_Calen_original - P_Calen_after_loss = 10 := 
sorry

end calen_pencils_loss_l1093_109357


namespace largest_circle_area_215_l1093_109355

theorem largest_circle_area_215
  (length width : ℝ)
  (h1 : length = 16)
  (h2 : width = 10)
  (P : ℝ := 2 * (length + width))
  (C : ℝ := P)
  (r : ℝ := C / (2 * Real.pi))
  (A : ℝ := Real.pi * r^2) :
  round A = 215 := by sorry

end largest_circle_area_215_l1093_109355


namespace lucy_current_fish_l1093_109302

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end lucy_current_fish_l1093_109302


namespace ratio_of_kids_in_morning_to_total_soccer_l1093_109397

-- Define the known conditions
def total_kids_in_camp : ℕ := 2000
def kids_going_to_soccer_camp : ℕ := total_kids_in_camp / 2
def kids_going_to_soccer_camp_in_afternoon : ℕ := 750
def kids_going_to_soccer_camp_in_morning : ℕ := kids_going_to_soccer_camp - kids_going_to_soccer_camp_in_afternoon

-- Define the conclusion to be proven
theorem ratio_of_kids_in_morning_to_total_soccer :
  (kids_going_to_soccer_camp_in_morning : ℚ) / (kids_going_to_soccer_camp : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_kids_in_morning_to_total_soccer_l1093_109397


namespace vector_t_perpendicular_l1093_109312

theorem vector_t_perpendicular (t : ℝ) :
  let a := (2, 4)
  let b := (-1, 1)
  let c := (2 + t, 4 - t)
  b.1 * c.1 + b.2 * c.2 = 0 → t = 1 := by
  sorry

end vector_t_perpendicular_l1093_109312


namespace track_length_l1093_109394

theorem track_length (x : ℝ) 
  (h1 : ∀ {d1 d2 : ℝ}, (d1 + d2 = x / 2) → (d1 = 120) → d2 = x / 2 - 120)
  (h2 : ∀ {d1 d2 : ℝ}, (d1 = x / 2 - 120 + 170) → (d1 = x / 2 + 50))
  (h3 : ∀ {d3 : ℝ}, (d3 = 3 * x / 2 - 170)) :
  x = 418 :=
by
  sorry

end track_length_l1093_109394


namespace digit_B_divisible_by_9_l1093_109379

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_l1093_109379


namespace distance_between_C_and_A_l1093_109389

theorem distance_between_C_and_A 
    (A B C : Type)
    (d_AB : ℝ) (d_BC : ℝ)
    (h1 : d_AB = 8)
    (h2 : d_BC = 10) :
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 18 ∧ ¬ (∃ y : ℝ, y = x) :=
sorry

end distance_between_C_and_A_l1093_109389


namespace vector_parallel_example_l1093_109365

theorem vector_parallel_example 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (ha : a = (2, 1)) 
  (hb : b = (4, 2))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  3 • a + 2 • b = (14, 7) := 
by
  sorry

end vector_parallel_example_l1093_109365


namespace solution_set_non_empty_implies_a_gt_1_l1093_109335

theorem solution_set_non_empty_implies_a_gt_1 (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := 
  sorry

end solution_set_non_empty_implies_a_gt_1_l1093_109335


namespace min_even_number_for_2015_moves_l1093_109391

theorem min_even_number_for_2015_moves (N : ℕ) (hN : N ≥ 2) :
  ∃ k : ℕ, N = 2 ^ k ∧ 2 ^ k ≥ 2 ∧ k ≥ 4030 :=
sorry

end min_even_number_for_2015_moves_l1093_109391


namespace avg_of_six_is_3_9_l1093_109381

noncomputable def avg_of_six_numbers 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : ℝ :=
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6

theorem avg_of_six_is_3_9 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : 
  avg_of_six_numbers avg1 avg2 avg3 h1 h2 h3 = 3.9 := 
by {
  sorry
}

end avg_of_six_is_3_9_l1093_109381


namespace simplify_eval_l1093_109383

theorem simplify_eval (a : ℝ) (h : a = Real.sqrt 3 / 3) : (a + 1) ^ 2 + a * (1 - a) = Real.sqrt 3 + 1 := 
by
  sorry

end simplify_eval_l1093_109383


namespace least_positive_integer_divisible_by_5_to_15_l1093_109318

def is_divisible_by_all (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m ∈ l, m ∣ n

theorem least_positive_integer_divisible_by_5_to_15 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_by_all n [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] ∧
  ∀ m : ℕ, m > 0 ∧ is_divisible_by_all m [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] → n ≤ m ∧ n = 360360 :=
by
  sorry

end least_positive_integer_divisible_by_5_to_15_l1093_109318


namespace possible_values_of_k_l1093_109304

theorem possible_values_of_k (k : ℕ) (N : ℕ) (h₁ : (k * (k + 1)) / 2 = N^2) (h₂ : N < 100) :
  k = 1 ∨ k = 8 ∨ k = 49 :=
sorry

end possible_values_of_k_l1093_109304


namespace number_of_integer_values_of_a_l1093_109333

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l1093_109333


namespace simplify_and_evaluate_expr_l1093_109338

theorem simplify_and_evaluate_expr (a b : ℕ) (h₁ : a = 2) (h₂ : b = 2023) : 
  (a + b)^2 + b * (a - b) - 3 * a * b = 4 := by
  sorry

end simplify_and_evaluate_expr_l1093_109338


namespace overall_average_score_l1093_109300

theorem overall_average_score 
  (M : ℝ) (E : ℝ) (m e : ℝ)
  (hM : M = 82)
  (hE : E = 75)
  (hRatio : m / e = 5 / 3) :
  (M * m + E * e) / (m + e) = 79.375 := 
by
  sorry

end overall_average_score_l1093_109300


namespace triangle_inequality_l1093_109359

theorem triangle_inequality 
(a b c : ℝ) (α β γ : ℝ)
(h_t : a + b > c ∧ a + c > b ∧ b + c > a)
(h_opposite : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ α + β + γ = π) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
sorry

end triangle_inequality_l1093_109359


namespace exactly_one_even_contradiction_assumption_l1093_109341

variable (a b c : ℕ)

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)

def conclusion (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (c % 2 = 0 ∧ a % 2 = 0)

theorem exactly_one_even_contradiction_assumption :
    exactly_one_even a b c ↔ ¬ conclusion a b c :=
by
  sorry

end exactly_one_even_contradiction_assumption_l1093_109341


namespace find_c_plus_d_l1093_109371

variables {a b c d : ℝ}

theorem find_c_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : a + d = 10) : c + d = 3 :=
by
  sorry

end find_c_plus_d_l1093_109371


namespace smallest_six_digit_negative_integer_congruent_to_five_mod_17_l1093_109348

theorem smallest_six_digit_negative_integer_congruent_to_five_mod_17 :
  ∃ x : ℤ, x < -100000 ∧ x ≥ -999999 ∧ x % 17 = 5 ∧ x = -100011 :=
by
  sorry

end smallest_six_digit_negative_integer_congruent_to_five_mod_17_l1093_109348


namespace bowling_ball_weight_l1093_109346

noncomputable def weight_of_one_bowling_ball : ℕ := 20

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = weight_of_one_bowling_ball := by
  sorry

end bowling_ball_weight_l1093_109346
