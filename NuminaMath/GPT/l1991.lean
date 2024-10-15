import Mathlib

namespace NUMINAMATH_GPT_quadruple_exists_unique_l1991_199114

def digits (x : Nat) : Prop := x ≤ 9

theorem quadruple_exists_unique :
  ∃ (A B C D: Nat),
    digits A ∧ digits B ∧ digits C ∧ digits D ∧
    A > B ∧ B > C ∧ C > D ∧
    (A * 1000 + B * 100 + C * 10 + D) -
    (D * 1000 + C * 100 + B * 10 + A) =
    (B * 1000 + D * 100 + A * 10 + C) ∧
    (A, B, C, D) = (7, 6, 4, 1) :=
by
  sorry

end NUMINAMATH_GPT_quadruple_exists_unique_l1991_199114


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1991_199150

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_1 :
  (∀ x : ℝ, f 1 x ≥ f 1 1) :=
by sorry

theorem problem_2 (x e : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) (hf : f a x = 1) :
  0 ≤ a ∧ a ≤ 1 :=
by sorry

theorem problem_3 (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 1 → f a x ≥ f a (1 / x)) → 1 ≤ a :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1991_199150


namespace NUMINAMATH_GPT_total_yen_l1991_199147

-- Define the given conditions in Lean 4
def bal_bahamian_dollars : ℕ := 5000
def bal_us_dollars : ℕ := 2000
def bal_euros : ℕ := 3000

def exchange_rate_bahamian_to_yen : ℝ := 122.13
def exchange_rate_us_to_yen : ℝ := 110.25
def exchange_rate_euro_to_yen : ℝ := 128.50

def check_acc1 : ℕ := 15000
def check_acc2 : ℕ := 6359
def sav_acc1 : ℕ := 5500
def sav_acc2 : ℕ := 3102

def stocks : ℕ := 200000
def bonds : ℕ := 150000
def mutual_funds : ℕ := 120000

-- Prove the total amount of yen the family has
theorem total_yen : 
  bal_bahamian_dollars * exchange_rate_bahamian_to_yen + 
  bal_us_dollars * exchange_rate_us_to_yen + 
  bal_euros * exchange_rate_euro_to_yen
  + (check_acc1 + check_acc2 + sav_acc1 + sav_acc2 : ℝ)
  + (stocks + bonds + mutual_funds : ℝ) = 1716611 := 
by
  sorry

end NUMINAMATH_GPT_total_yen_l1991_199147


namespace NUMINAMATH_GPT_find_n_l1991_199134

theorem find_n (a b n : ℕ) (k l m : ℤ) 
  (ha : a % n = 2) 
  (hb : b % n = 3) 
  (h_ab : a > b) 
  (h_ab_mod : (a - b) % n = 5) : 
  n = 6 := 
sorry

end NUMINAMATH_GPT_find_n_l1991_199134


namespace NUMINAMATH_GPT_scrambled_eggs_count_l1991_199164

-- Definitions based on the given conditions
def num_sausages := 3
def time_per_sausage := 5
def time_per_egg := 4
def total_time := 39

-- Prove that Kira scrambled 6 eggs
theorem scrambled_eggs_count : (total_time - num_sausages * time_per_sausage) / time_per_egg = 6 := by
  sorry

end NUMINAMATH_GPT_scrambled_eggs_count_l1991_199164


namespace NUMINAMATH_GPT_distance_between_riya_and_priya_l1991_199188

theorem distance_between_riya_and_priya (speed_riya speed_priya : ℝ) (time_hours : ℝ)
  (h1 : speed_riya = 21) (h2 : speed_priya = 22) (h3 : time_hours = 1) :
  speed_riya * time_hours + speed_priya * time_hours = 43 := by
  sorry

end NUMINAMATH_GPT_distance_between_riya_and_priya_l1991_199188


namespace NUMINAMATH_GPT_exists_f_condition_l1991_199185

open Nat

-- Define the function φ from ℕ to ℕ
variable (ϕ : ℕ → ℕ)

-- The formal statement capturing the given math proof problem
theorem exists_f_condition (ϕ : ℕ → ℕ) : 
  ∃ (f : ℕ → ℤ), (∀ x : ℕ, f x > f (ϕ x)) :=
  sorry

end NUMINAMATH_GPT_exists_f_condition_l1991_199185


namespace NUMINAMATH_GPT_sam_wins_probability_l1991_199192

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end NUMINAMATH_GPT_sam_wins_probability_l1991_199192


namespace NUMINAMATH_GPT_factor_expression_l1991_199176

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1991_199176


namespace NUMINAMATH_GPT_present_ages_l1991_199196

theorem present_ages
  (R D K : ℕ) (x : ℕ)
  (H1 : R = 4 * x)
  (H2 : D = 3 * x)
  (H3 : K = 5 * x)
  (H4 : R + 6 = 26)
  (H5 : (R + 8) + (D + 8) = K) :
  D = 15 ∧ K = 51 :=
sorry

end NUMINAMATH_GPT_present_ages_l1991_199196


namespace NUMINAMATH_GPT_p_squared_plus_13_mod_n_eq_2_l1991_199162

theorem p_squared_plus_13_mod_n_eq_2 (p : ℕ) (prime_p : Prime p) (h : p > 3) (n : ℕ) :
  (∃ (k : ℕ), p ^ 2 + 13 = k * n + 2) → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_p_squared_plus_13_mod_n_eq_2_l1991_199162


namespace NUMINAMATH_GPT_max_sum_of_squares_eq_100_l1991_199132

theorem max_sum_of_squares_eq_100 : 
  ∃ (x y : ℤ), x^2 + y^2 = 100 ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x + y ≤ 14) ∧ 
  (∃ (x y : ℕ), x^2 + y^2 = 100 ∧ x + y = 14) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_sum_of_squares_eq_100_l1991_199132


namespace NUMINAMATH_GPT_correct_answer_l1991_199175

theorem correct_answer (A B C D : String) (sentence : String)
  (h1 : A = "us")
  (h2 : B = "we")
  (h3 : C = "our")
  (h4 : D = "ours")
  (h_sentence : sentence = "To save class time, our teacher has _ students do half of the exercise in class and complete the other half for homework.") :
  sentence = "To save class time, our teacher has " ++ A ++ " students do half of the exercise in class and complete the other half for homework." :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l1991_199175


namespace NUMINAMATH_GPT_range_of_k_l1991_199180

theorem range_of_k (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ (x, y) = (1, 2)) →
  (3 < k ∧ k < 7) :=
by
  intros hxy
  sorry

end NUMINAMATH_GPT_range_of_k_l1991_199180


namespace NUMINAMATH_GPT_harry_took_5_eggs_l1991_199153

theorem harry_took_5_eggs (initial : ℕ) (left : ℕ) (took : ℕ) 
  (h1 : initial = 47) (h2 : left = 42) (h3 : left = initial - took) : 
  took = 5 :=
sorry

end NUMINAMATH_GPT_harry_took_5_eggs_l1991_199153


namespace NUMINAMATH_GPT_ab_value_l1991_199105

theorem ab_value (a b : ℝ) (h1 : a = Real.exp (2 - a)) (h2 : 1 + Real.log b = Real.exp (1 - Real.log b)) : 
  a * b = Real.exp 1 :=
sorry

end NUMINAMATH_GPT_ab_value_l1991_199105


namespace NUMINAMATH_GPT_min_distance_A_D_l1991_199151

theorem min_distance_A_D (A B C E D : Type) 
  (d_AB d_BC d_CE d_ED : ℝ) 
  (h1 : d_AB = 12) 
  (h2 : d_BC = 7) 
  (h3 : d_CE = 2) 
  (h4 : d_ED = 5) : 
  ∃ d_AD : ℝ, d_AD = 2 := 
by
  sorry

end NUMINAMATH_GPT_min_distance_A_D_l1991_199151


namespace NUMINAMATH_GPT_convince_the_king_l1991_199136

/-- Define the types of inhabitants -/
inductive Inhabitant
| Knight
| Liar
| Normal

/-- Define the king's preference -/
def K (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- All knights tell the truth -/
def tells_truth (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => True
  | Inhabitant.Liar => False
  | Inhabitant.Normal => False

/-- All liars always lie -/
def tells_lie (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => True
  | Inhabitant.Normal => False

/-- Normal persons can tell both truths and lies -/
def can_tell_both (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- Prove there exists a true statement and a false statement to convince the king -/
theorem convince_the_king (p : Inhabitant) :
  (∃ S : Prop, (S ↔ tells_truth p) ∧ K p) ∧ (∃ S' : Prop, (¬ S' ↔ tells_lie p) ∧ K p) :=
by
  sorry

end NUMINAMATH_GPT_convince_the_king_l1991_199136


namespace NUMINAMATH_GPT_exists_x0_in_interval_l1991_199145

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_x0_in_interval :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 4 ∧ f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
sorry

end NUMINAMATH_GPT_exists_x0_in_interval_l1991_199145


namespace NUMINAMATH_GPT_total_bouncy_balls_l1991_199173

-- Definitions of the given quantities
def r : ℕ := 4 -- number of red packs
def y : ℕ := 8 -- number of yellow packs
def g : ℕ := 4 -- number of green packs
def n : ℕ := 10 -- number of balls per pack

-- Proof statement to show the correct total number of balls
theorem total_bouncy_balls : r * n + y * n + g * n = 160 := by
  sorry

end NUMINAMATH_GPT_total_bouncy_balls_l1991_199173


namespace NUMINAMATH_GPT_find_square_side_length_l1991_199119

theorem find_square_side_length
  (a CF AE : ℝ)
  (h_CF : CF = 2 * a)
  (h_AE : AE = 3.5 * a)
  (h_sum : CF + AE = 91) :
  a = 26 := by
  sorry

end NUMINAMATH_GPT_find_square_side_length_l1991_199119


namespace NUMINAMATH_GPT_fraction_relation_l1991_199189

-- Definitions for arithmetic sequences and their sums
noncomputable def a_n (a₁ d₁ n : ℕ) := a₁ + (n - 1) * d₁
noncomputable def b_n (b₁ d₂ n : ℕ) := b₁ + (n - 1) * d₂

noncomputable def A_n (a₁ d₁ n : ℕ) := n * a₁ + n * (n - 1) * d₁ / 2
noncomputable def B_n (b₁ d₂ n : ℕ) := n * b₁ + n * (n - 1) * d₂ / 2

-- Theorem statement
theorem fraction_relation (a₁ d₁ b₁ d₂ : ℕ) (h : ∀ n : ℕ, B_n a₁ d₁ n ≠ 0 → A_n a₁ d₁ n / B_n b₁ d₂ n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b_n b₁ d₂ n ≠ 0 → a_n a₁ d₁ n / b_n b₁ d₂ n = (4 * n - 3) / (6 * n - 2) :=
sorry

end NUMINAMATH_GPT_fraction_relation_l1991_199189


namespace NUMINAMATH_GPT_weight_of_replaced_person_l1991_199197

theorem weight_of_replaced_person (avg_weight : ℝ) (new_person_weight : ℝ)
  (h1 : new_person_weight = 65)
  (h2 : ∀ (initial_avg_weight : ℝ), 8 * (initial_avg_weight + 2.5) - 8 * initial_avg_weight = new_person_weight - avg_weight) :
  avg_weight = 45 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l1991_199197


namespace NUMINAMATH_GPT_quadratic_inequality_range_of_k_l1991_199160

theorem quadratic_inequality_range_of_k :
  ∀ k : ℝ , (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) ↔ (-1 < k ∧ k ≤ 0) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_range_of_k_l1991_199160


namespace NUMINAMATH_GPT_B_is_werewolf_l1991_199108

def is_werewolf (x : Type) : Prop := sorry
def is_knight (x : Type) : Prop := sorry
def is_liar (x : Type) : Prop := sorry

variables (A B : Type)

-- Conditions
axiom one_is_werewolf : is_werewolf A ∨ is_werewolf B
axiom only_one_werewolf : ¬ (is_werewolf A ∧ is_werewolf B)
axiom A_statement : is_werewolf A → is_knight A
axiom B_statement : is_werewolf B → is_liar B

theorem B_is_werewolf : is_werewolf B := 
by
  sorry

end NUMINAMATH_GPT_B_is_werewolf_l1991_199108


namespace NUMINAMATH_GPT_jade_more_transactions_l1991_199122

theorem jade_more_transactions (mabel_transactions : ℕ) (anthony_percentage : ℕ) (cal_fraction_numerator : ℕ) 
  (cal_fraction_denominator : ℕ) (jade_transactions : ℕ) (h1 : mabel_transactions = 90) 
  (h2 : anthony_percentage = 10) (h3 : cal_fraction_numerator = 2) (h4 : cal_fraction_denominator = 3) 
  (h5 : jade_transactions = 83) :
  jade_transactions - (2 * (90 + (90 * 10 / 100)) / 3) = 17 := 
by
  sorry

end NUMINAMATH_GPT_jade_more_transactions_l1991_199122


namespace NUMINAMATH_GPT_uniqueSumEqualNumber_l1991_199182

noncomputable def sumPreceding (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem uniqueSumEqualNumber :
  ∃! n : ℕ, sumPreceding n = n := by
  sorry

end NUMINAMATH_GPT_uniqueSumEqualNumber_l1991_199182


namespace NUMINAMATH_GPT_pages_per_hour_l1991_199186

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end NUMINAMATH_GPT_pages_per_hour_l1991_199186


namespace NUMINAMATH_GPT_mike_spent_total_l1991_199166

-- Define the prices of the items
def price_trumpet : ℝ := 145.16
def price_song_book : ℝ := 5.84

-- Define the total amount spent
def total_spent : ℝ := price_trumpet + price_song_book

-- The theorem to be proved
theorem mike_spent_total :
  total_spent = 151.00 :=
sorry

end NUMINAMATH_GPT_mike_spent_total_l1991_199166


namespace NUMINAMATH_GPT_ratio_of_dogs_to_cats_l1991_199154

-- Definition of conditions
def total_animals : Nat := 21
def cats_to_spay : Nat := 7
def dogs_to_spay : Nat := total_animals - cats_to_spay

-- Ratio of dogs to cats
def dogs_to_cats_ratio : Nat := dogs_to_spay / cats_to_spay

-- Statement to prove
theorem ratio_of_dogs_to_cats : dogs_to_cats_ratio = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_dogs_to_cats_l1991_199154


namespace NUMINAMATH_GPT_contact_alignment_possible_l1991_199102

/-- A vacuum tube has seven contacts arranged in a circle and is inserted into a socket that has seven holes.
Prove that it is possible to number the tube's contacts and the socket's holes in such a way that:
in any insertion of the tube, at least one contact will align with its corresponding hole (i.e., the hole with the same number). -/
theorem contact_alignment_possible : ∃ (f : Fin 7 → Fin 7), ∀ (rotation : Fin 7 → Fin 7), ∃ k : Fin 7, f k = rotation k := 
sorry

end NUMINAMATH_GPT_contact_alignment_possible_l1991_199102


namespace NUMINAMATH_GPT_simplify_expression_l1991_199165

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x ^ 2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1991_199165


namespace NUMINAMATH_GPT_find_range_of_a_l1991_199152

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (2 * x) - a * x

theorem find_range_of_a (a : ℝ) :
  (∀ x > 0, f x a > a * x^2 + 1) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l1991_199152


namespace NUMINAMATH_GPT_ben_owes_rachel_l1991_199178

theorem ben_owes_rachel :
  let dollars_per_lawn := (13 : ℚ) / 3
  let lawns_mowed := (8 : ℚ) / 5
  let total_owed := (104 : ℚ) / 15
  dollars_per_lawn * lawns_mowed = total_owed := 
by 
  sorry

end NUMINAMATH_GPT_ben_owes_rachel_l1991_199178


namespace NUMINAMATH_GPT_tan_ratio_l1991_199158

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end NUMINAMATH_GPT_tan_ratio_l1991_199158


namespace NUMINAMATH_GPT_eval_polynomial_l1991_199118

theorem eval_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) : x^3 - 3 * x^2 - 9 * x + 27 = 27 := 
by
  sorry

end NUMINAMATH_GPT_eval_polynomial_l1991_199118


namespace NUMINAMATH_GPT_maximum_value_quadratic_l1991_199139

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximum_value_quadratic :
  ∃ x : ℝ, quadratic_function x = 2 ∧ ∀ y : ℝ, quadratic_function y ≤ 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_quadratic_l1991_199139


namespace NUMINAMATH_GPT_number_of_students_in_class_l1991_199111

theorem number_of_students_in_class
  (x : ℕ)
  (S : ℝ)
  (incorrect_score correct_score : ℝ)
  (incorrect_score_mistake : incorrect_score = 85)
  (correct_score_corrected : correct_score = 78)
  (average_difference : ℝ)
  (average_difference_value : average_difference = 0.75)
  (test_attendance : ℕ)
  (test_attendance_value : test_attendance = x - 3)
  (average_difference_condition : (S + incorrect_score) / test_attendance - (S + correct_score) / test_attendance = average_difference) :
  x = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_in_class_l1991_199111


namespace NUMINAMATH_GPT_f_is_increasing_l1991_199181

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 3 * x

theorem f_is_increasing : ∀ (x : ℝ), (deriv f x) > 0 :=
by
  intro x
  calc
    deriv f x = 2 * Real.exp (2 * x) + 3 := by sorry
    _ > 0 := by sorry

end NUMINAMATH_GPT_f_is_increasing_l1991_199181


namespace NUMINAMATH_GPT_max_imaginary_part_angle_l1991_199113

def poly (z : Complex) : Complex := z^6 - z^4 + z^2 - 1

theorem max_imaginary_part_angle :
  ∃ θ : Real, θ = 45 ∧ 
  (∃ z : Complex, poly z = 0 ∧ ∀ w : Complex, poly w = 0 → w.im ≤ z.im)
:= sorry

end NUMINAMATH_GPT_max_imaginary_part_angle_l1991_199113


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1991_199135

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(a*x^2 - |x + 1| + 2*a < 0)) ↔ a ≥ (sqrt 3 + 1) / 4 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l1991_199135


namespace NUMINAMATH_GPT_basketball_three_point_shots_l1991_199163

theorem basketball_three_point_shots (t h f : ℕ) 
  (h1 : 2 * t = 6 * h)
  (h2 : f = h - 4)
  (h3: 2 * t + 3 * h + f = 76)
  (h4: t + h + f = 40) : h = 8 :=
sorry

end NUMINAMATH_GPT_basketball_three_point_shots_l1991_199163


namespace NUMINAMATH_GPT_common_ratio_of_arithmetic_seq_l1991_199199

theorem common_ratio_of_arithmetic_seq (a_1 q : ℝ) 
  (h1 : a_1 + a_1 * q^2 = 10) 
  (h2 : a_1 * q^3 + a_1 * q^5 = 5 / 4) : 
  q = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_common_ratio_of_arithmetic_seq_l1991_199199


namespace NUMINAMATH_GPT_cyclic_path_1310_to_1315_l1991_199174

theorem cyclic_path_1310_to_1315 :
  ∀ (n : ℕ), (n % 6 = 2 → (n + 5) % 6 = 3) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_path_1310_to_1315_l1991_199174


namespace NUMINAMATH_GPT_linear_equation_solution_l1991_199101

theorem linear_equation_solution (m n : ℤ) (x y : ℤ)
  (h1 : x + 2 * y = 5)
  (h2 : x + y = 7)
  (h3 : x = -m)
  (h4 : y = -n) :
  (3 * m + 2 * n) / (5 * m - n) = 11 / 14 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_solution_l1991_199101


namespace NUMINAMATH_GPT_apples_found_l1991_199142

theorem apples_found (start_apples : ℕ) (end_apples : ℕ) (h_start : start_apples = 7) (h_end : end_apples = 81) : 
  end_apples - start_apples = 74 := 
by 
  sorry

end NUMINAMATH_GPT_apples_found_l1991_199142


namespace NUMINAMATH_GPT_max_not_expressed_as_linear_comb_l1991_199187

theorem max_not_expressed_as_linear_comb {a b c : ℕ} (h_coprime_ab : Nat.gcd a b = 1)
                                        (h_coprime_bc : Nat.gcd b c = 1)
                                        (h_coprime_ca : Nat.gcd c a = 1) :
    Nat := sorry

end NUMINAMATH_GPT_max_not_expressed_as_linear_comb_l1991_199187


namespace NUMINAMATH_GPT_complex_quadrant_l1991_199183

theorem complex_quadrant 
  (z : ℂ) 
  (h : (2 + 3 * Complex.I) * z = 1 + Complex.I) : 
  z.re > 0 ∧ z.im < 0 := 
sorry

end NUMINAMATH_GPT_complex_quadrant_l1991_199183


namespace NUMINAMATH_GPT_max_val_proof_l1991_199100

noncomputable def max_val (p q r x y z : ℝ) : ℝ :=
  1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (x + y) + 1 / (x + z) + 1 / (y + z)

theorem max_val_proof {p q r x y z : ℝ}
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_pqr : p + q + r = 2) (h_sum_xyz : x + y + z = 1) :
  max_val p q r x y z = 27 / 4 :=
sorry

end NUMINAMATH_GPT_max_val_proof_l1991_199100


namespace NUMINAMATH_GPT_number_of_people_prefer_soda_l1991_199156

-- Given conditions
def total_people : ℕ := 600
def central_angle_soda : ℝ := 198
def full_circle_angle : ℝ := 360

-- Problem statement
theorem number_of_people_prefer_soda : 
  (total_people : ℝ) * (central_angle_soda / full_circle_angle) = 330 := by
  sorry

end NUMINAMATH_GPT_number_of_people_prefer_soda_l1991_199156


namespace NUMINAMATH_GPT_least_positive_integer_x_l1991_199191

theorem least_positive_integer_x : ∃ x : ℕ, ((2 * x)^2 + 2 * 43 * (2 * x) + 43^2) % 53 = 0 ∧ 0 < x ∧ (∀ y : ℕ, ((2 * y)^2 + 2 * 43 * (2 * y) + 43^2) % 53 = 0 → 0 < y → x ≤ y) := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_x_l1991_199191


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1991_199110

noncomputable def area_triangle_ABC (AF BE : ℝ) (angle_FGB : ℝ) : ℝ :=
  let FG := AF / 3
  let BG := (2 / 3) * BE
  let area_FGB := (1 / 2) * FG * BG * Real.sin angle_FGB
  6 * area_FGB

theorem area_of_triangle_ABC
  (AF BE : ℕ) (hAF : AF = 10) (hBE : BE = 15)
  (angle_FGB : ℝ) (h_angle_FGB : angle_FGB = Real.pi / 3) :
  area_triangle_ABC AF BE angle_FGB = 50 * Real.sqrt 3 :=
by
  simp [area_triangle_ABC, hAF, hBE, h_angle_FGB]
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1991_199110


namespace NUMINAMATH_GPT_simplify_expression_l1991_199112

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1991_199112


namespace NUMINAMATH_GPT_proof_problem_l1991_199198

variable (a b c m : ℝ)

-- Condition
def condition : Prop := m = (c * a * b) / (a + b)

-- Question
def question : Prop := b = (m * a) / (c * a - m)

-- Proof statement
theorem proof_problem (h : condition a b c m) : question a b c m := 
sorry

end NUMINAMATH_GPT_proof_problem_l1991_199198


namespace NUMINAMATH_GPT_negation_of_existence_l1991_199130

theorem negation_of_existence : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) = (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l1991_199130


namespace NUMINAMATH_GPT_min_draws_to_ensure_20_of_one_color_l1991_199146

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

end NUMINAMATH_GPT_min_draws_to_ensure_20_of_one_color_l1991_199146


namespace NUMINAMATH_GPT_cos_arithmetic_sequence_result_l1991_199141

-- Define an arithmetic sequence as a function
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem cos_arithmetic_sequence_result (a d : ℝ) 
  (h : arithmetic_seq a d 1 + arithmetic_seq a d 5 + arithmetic_seq a d 9 = 8 * Real.pi) :
  Real.cos (arithmetic_seq a d 3 + arithmetic_seq a d 7) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_arithmetic_sequence_result_l1991_199141


namespace NUMINAMATH_GPT_alcohol_percentage_new_mixture_l1991_199117

theorem alcohol_percentage_new_mixture :
  let initial_alcohol_percentage := 0.90
  let initial_solution_volume := 24
  let added_water_volume := 16
  let total_new_volume := initial_solution_volume + added_water_volume
  let initial_alcohol_amount := initial_solution_volume * initial_alcohol_percentage
  let new_alcohol_percentage := (initial_alcohol_amount / total_new_volume) * 100
  new_alcohol_percentage = 54 := by
    sorry

end NUMINAMATH_GPT_alcohol_percentage_new_mixture_l1991_199117


namespace NUMINAMATH_GPT_minimum_value_proof_l1991_199128

noncomputable def minValue (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : ℝ := 
  (x + 8 * y) / (x * y)

theorem minimum_value_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : 
  minValue x y hx hy h = 9 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_proof_l1991_199128


namespace NUMINAMATH_GPT_find_m_l1991_199131

theorem find_m (m : ℤ) (h : 3 ∈ ({1, m + 2} : Set ℤ)) : m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l1991_199131


namespace NUMINAMATH_GPT_measure_of_angle_C_range_of_sum_ab_l1991_199120

-- Proof problem (1): Prove the measure of angle C
theorem measure_of_angle_C (a b c : ℝ) (A B C : ℝ) 
  (h1 : 2 * c * Real.sin C = (2 * b + a) * Real.sin B + (2 * a - 3 * b) * Real.sin A) :
  C = Real.pi / 3 := by 
  sorry

-- Proof problem (2): Prove the range of possible values of a + b
theorem range_of_sum_ab (a b : ℝ) (c : ℝ) (h1 : c = 4) (h2 : 16 = a^2 + b^2 - a * b) :
  4 < a + b ∧ a + b ≤ 8 := by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_range_of_sum_ab_l1991_199120


namespace NUMINAMATH_GPT_chemical_reaction_l1991_199103

def reaction_balanced (koh nh4i ki nh3 h2o : ℕ) : Prop :=
  koh = nh4i ∧ nh4i = ki ∧ ki = nh3 ∧ nh3 = h2o

theorem chemical_reaction
  (KOH NH4I : ℕ)
  (h1 : KOH = 3)
  (h2 : NH4I = 3)
  (balanced : reaction_balanced KOH NH4I 3 3 3) :
  (∃ (NH3 KI H2O : ℕ),
    NH3 = 3 ∧ KI = 3 ∧ H2O = 3 ∧ 
    NH3 = NH4I - NH4I ∧
    KI = KOH - KOH ∧
    H2O = KOH - KOH) ∧
  (KOH = NH4I) := 
by sorry

end NUMINAMATH_GPT_chemical_reaction_l1991_199103


namespace NUMINAMATH_GPT_complex_number_sum_l1991_199177

noncomputable def x : ℝ := 3 / 5
noncomputable def y : ℝ := -3 / 5

theorem complex_number_sum :
  (x + y) = -2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_complex_number_sum_l1991_199177


namespace NUMINAMATH_GPT_find_k_value_l1991_199159

theorem find_k_value (x k : ℝ) (hx : Real.logb 9 3 = x) (hk : Real.logb 3 81 = k * x) : k = 8 :=
by sorry

end NUMINAMATH_GPT_find_k_value_l1991_199159


namespace NUMINAMATH_GPT_train_crossing_time_l1991_199149

noncomputable def time_to_cross_platform
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ) : ℝ :=
  let speed_ms := speed_kmph / 3.6
  let total_distance := length_train + length_platform
  total_distance / speed_ms

theorem train_crossing_time
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ)
  (h_speed : speed_kmph = 72)
  (h_train_length : length_train = 280.0416)
  (h_platform_length : length_platform = 240) :
  time_to_cross_platform speed_kmph length_train length_platform = 26.00208 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1991_199149


namespace NUMINAMATH_GPT_grandson_age_is_5_l1991_199144

-- Definitions based on the conditions
def grandson_age_months_eq_grandmother_years (V B : ℕ) : Prop := B = 12 * V
def combined_age_eq_65 (V B : ℕ) : Prop := B + V = 65

-- Main theorem stating that under these conditions, the grandson's age is 5 years
theorem grandson_age_is_5 (V B : ℕ) (h₁ : grandson_age_months_eq_grandmother_years V B) (h₂ : combined_age_eq_65 V B) : V = 5 :=
by sorry

end NUMINAMATH_GPT_grandson_age_is_5_l1991_199144


namespace NUMINAMATH_GPT_length_of_RS_l1991_199121

-- Define the lengths of the edges of the tetrahedron
def edge_lengths : List ℕ := [9, 16, 22, 31, 39, 48]

-- Given the edge PQ has length 48
def PQ_length : ℕ := 48

-- We need to prove that the length of edge RS is 9
theorem length_of_RS :
  ∃ (RS : ℕ), RS = 9 ∧
  ∃ (PR QR PS SQ : ℕ),
  [PR, QR, PS, SQ] ⊆ edge_lengths ∧
  PR + QR > PQ_length ∧
  PR + PQ_length > QR ∧
  QR + PQ_length > PR ∧
  PS + SQ > PQ_length ∧
  PS + PQ_length > SQ ∧
  SQ + PQ_length > PS :=
by
  sorry

end NUMINAMATH_GPT_length_of_RS_l1991_199121


namespace NUMINAMATH_GPT_speed_of_car_in_second_hour_l1991_199148

theorem speed_of_car_in_second_hour
(speed_in_first_hour : ℝ)
(average_speed : ℝ)
(total_time : ℝ)
(speed_in_second_hour : ℝ)
(h1 : speed_in_first_hour = 100)
(h2 : average_speed = 65)
(h3 : total_time = 2)
(h4 : average_speed = (speed_in_first_hour + speed_in_second_hour) / total_time) :
  speed_in_second_hour = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_of_car_in_second_hour_l1991_199148


namespace NUMINAMATH_GPT_tim_watched_total_hours_tv_l1991_199168

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tim_watched_total_hours_tv_l1991_199168


namespace NUMINAMATH_GPT_total_clothes_count_l1991_199184

theorem total_clothes_count (shirts_per_pants : ℕ) (pants : ℕ) (shirts : ℕ) : shirts_per_pants = 6 → pants = 40 → shirts = shirts_per_pants * pants → shirts + pants = 280 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end NUMINAMATH_GPT_total_clothes_count_l1991_199184


namespace NUMINAMATH_GPT_linear_equation_a_is_minus_one_l1991_199171

theorem linear_equation_a_is_minus_one (a : ℝ) (x : ℝ) :
  ((a - 1) * x ^ (2 - |a|) + 5 = 0) → (2 - |a| = 1) → (a ≠ 1) → a = -1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_linear_equation_a_is_minus_one_l1991_199171


namespace NUMINAMATH_GPT_best_selling_price_70_l1991_199127

-- Definitions for the conditions in the problem
def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 50

-- The profit function
def profit (x : ℕ) : ℕ :=
(50 + x - purchase_price) * (initial_sales_volume - x)

-- The problem statement to be proved
theorem best_selling_price_70 :
  ∃ x : ℕ, 0 < x ∧ x < 50 ∧ profit x = 900 ∧ (initial_selling_price + x) = 70 :=
by
  sorry

end NUMINAMATH_GPT_best_selling_price_70_l1991_199127


namespace NUMINAMATH_GPT_exists_negative_fraction_lt_four_l1991_199195

theorem exists_negative_fraction_lt_four : 
  ∃ (x : ℚ), x < 0 ∧ |x| < 4 := 
sorry

end NUMINAMATH_GPT_exists_negative_fraction_lt_four_l1991_199195


namespace NUMINAMATH_GPT_coordinate_sum_of_point_on_graph_l1991_199167

theorem coordinate_sum_of_point_on_graph (g : ℕ → ℕ) (h : ℕ → ℕ)
  (h1 : g 2 = 8)
  (h2 : ∀ x, h x = 3 * (g x) ^ 2) :
  2 + h 2 = 194 :=
by
  sorry

end NUMINAMATH_GPT_coordinate_sum_of_point_on_graph_l1991_199167


namespace NUMINAMATH_GPT_find_sum_of_coefficients_l1991_199190

theorem find_sum_of_coefficients (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -(1/2) ∨ x > 1/3)) :
  a + b = -14 := 
sorry

end NUMINAMATH_GPT_find_sum_of_coefficients_l1991_199190


namespace NUMINAMATH_GPT_smallest_x_l1991_199157

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_x_l1991_199157


namespace NUMINAMATH_GPT_silk_pieces_count_l1991_199133

theorem silk_pieces_count (S C : ℕ) (h1 : S = 2 * C) (h2 : S + C + 2 = 13) : S = 7 :=
by
  sorry

end NUMINAMATH_GPT_silk_pieces_count_l1991_199133


namespace NUMINAMATH_GPT_total_pints_l1991_199116

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end NUMINAMATH_GPT_total_pints_l1991_199116


namespace NUMINAMATH_GPT_number_of_rectangles_l1991_199124

theorem number_of_rectangles (m n : ℕ) (h1 : m = 8) (h2 : n = 10) : (m - 1) * (n - 1) = 63 := by
  sorry

end NUMINAMATH_GPT_number_of_rectangles_l1991_199124


namespace NUMINAMATH_GPT_moles_of_NaCl_formed_l1991_199155

-- Define the balanced chemical reaction and quantities
def chemical_reaction :=
  "NaOH + HCl → NaCl + H2O"

-- Define the initial moles of sodium hydroxide (NaOH) and hydrochloric acid (HCl)
def moles_NaOH : ℕ := 2
def moles_HCl : ℕ := 2

-- The stoichiometry from the balanced equation: 1 mole NaOH reacts with 1 mole HCl to produce 1 mole NaCl.
def stoichiometry_NaOH_to_NaCl : ℕ := 1
def stoichiometry_HCl_to_NaCl : ℕ := 1

-- Given the initial conditions, prove that 2 moles of NaCl are formed.
theorem moles_of_NaCl_formed :
  (moles_NaOH = 2) → (moles_HCl = 2) → 2 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_moles_of_NaCl_formed_l1991_199155


namespace NUMINAMATH_GPT_length_of_tunnel_l1991_199140

theorem length_of_tunnel
    (length_of_train : ℕ)
    (speed_kmh : ℕ)
    (crossing_time_seconds : ℕ)
    (distance_covered : ℕ)
    (length_of_tunnel : ℕ) :
    length_of_train = 1200 →
    speed_kmh = 96 →
    crossing_time_seconds = 90 →
    distance_covered = (speed_kmh * 1000 / 3600) * crossing_time_seconds →
    length_of_train + length_of_tunnel = distance_covered →
    length_of_tunnel = 6000 :=
by
  sorry

end NUMINAMATH_GPT_length_of_tunnel_l1991_199140


namespace NUMINAMATH_GPT_mul_pos_neg_eq_neg_l1991_199129

theorem mul_pos_neg_eq_neg (a : Int) : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_GPT_mul_pos_neg_eq_neg_l1991_199129


namespace NUMINAMATH_GPT_find_preimage_l1991_199172

def mapping (x y : ℝ) : ℝ × ℝ :=
  (x + y, x - y)

theorem find_preimage :
  mapping 2 1 = (3, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_preimage_l1991_199172


namespace NUMINAMATH_GPT_tan_alpha_value_sin_cos_expression_l1991_199194

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_value (α : ℝ) (h1 : Real.tan (α + Real.pi / 4) = 2) : tan_alpha α = 1 / 3 :=
by
  sorry

theorem sin_cos_expression (α : ℝ) (h2 : tan_alpha α = 1 / 3) :
  (Real.sin (2 * α) - Real.sin α ^ 2) / (1 + Real.cos (2 * α)) = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_value_sin_cos_expression_l1991_199194


namespace NUMINAMATH_GPT_sampling_is_systematic_l1991_199179

-- Defining the conditions
def mock_exam (rooms students_per_room seat_selected: ℕ) : Prop :=
  rooms = 80 ∧ students_per_room = 30 ∧ seat_selected = 15

-- Theorem statement
theorem sampling_is_systematic 
  (rooms students_per_room seat_selected: ℕ)
  (h: mock_exam rooms students_per_room seat_selected) : 
  sampling_method = "Systematic sampling" :=
sorry

end NUMINAMATH_GPT_sampling_is_systematic_l1991_199179


namespace NUMINAMATH_GPT_new_average_daily_production_l1991_199125

theorem new_average_daily_production (n : ℕ) (avg_past : ℕ) (production_today : ℕ) (new_avg : ℕ)
  (h1 : n = 9)
  (h2 : avg_past = 50)
  (h3 : production_today = 100)
  (h4 : new_avg = (avg_past * n + production_today) / (n + 1)) :
  new_avg = 55 :=
by
  -- Using the provided conditions, it will be shown in the proof stage that new_avg equals 55
  sorry

end NUMINAMATH_GPT_new_average_daily_production_l1991_199125


namespace NUMINAMATH_GPT_transformed_conic_symmetric_eq_l1991_199170

def conic_E (x y : ℝ) := x^2 + 2 * x * y + y^2 + 3 * x + y
def line_l (x y : ℝ) := 2 * x - y - 1

def transformed_conic_equation (x y : ℝ) := x^2 + 14 * x * y + 49 * y^2 - 21 * x + 103 * y + 54

theorem transformed_conic_symmetric_eq (x y : ℝ) :
  (∀ x y, conic_E x y = 0 → 
    ∃ x' y', line_l x' y' = 0 ∧ conic_E x' y' = 0 ∧ transformed_conic_equation x y = 0) :=
sorry

end NUMINAMATH_GPT_transformed_conic_symmetric_eq_l1991_199170


namespace NUMINAMATH_GPT_seven_times_one_fifth_cubed_l1991_199104

theorem seven_times_one_fifth_cubed : 7 * (1 / 5) ^ 3 = 7 / 125 := 
by 
  sorry

end NUMINAMATH_GPT_seven_times_one_fifth_cubed_l1991_199104


namespace NUMINAMATH_GPT_sum_of_remainders_l1991_199143

theorem sum_of_remainders (a b c : ℕ) (h₁ : a % 30 = 15) (h₂ : b % 30 = 7) (h₃ : c % 30 = 18) : 
    (a + b + c) % 30 = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1991_199143


namespace NUMINAMATH_GPT_container_weight_l1991_199107

noncomputable def weight_in_pounds : ℝ := 57 + 3/8
noncomputable def weight_in_ounces : ℝ := weight_in_pounds * 16
noncomputable def number_of_containers : ℝ := 7
noncomputable def ounces_per_container : ℝ := weight_in_ounces / number_of_containers

theorem container_weight :
  ounces_per_container = 131.142857 :=
by sorry

end NUMINAMATH_GPT_container_weight_l1991_199107


namespace NUMINAMATH_GPT_normal_trip_distance_l1991_199109

variable (S D : ℝ)

-- Conditions
axiom h1 : D = 3 * S
axiom h2 : D + 50 = 5 * S

theorem normal_trip_distance : D = 75 :=
by
  sorry

end NUMINAMATH_GPT_normal_trip_distance_l1991_199109


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1991_199138

theorem arithmetic_sequence_nth_term (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  (a₁ = 11) →
  (d = -3) →
  (-49 = a₁ + (n - 1) * d) →
  (n = 21) :=
by 
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1991_199138


namespace NUMINAMATH_GPT_minimum_x_for_g_maximum_l1991_199126

theorem minimum_x_for_g_maximum :
  ∃ x > 0, ∀ k m: ℤ, (x = 1440 * k + 360 ∧ x = 2520 * m + 630) -> x = 7560 :=
by
  sorry

end NUMINAMATH_GPT_minimum_x_for_g_maximum_l1991_199126


namespace NUMINAMATH_GPT_minimum_fruits_l1991_199161

open Nat

theorem minimum_fruits (n : ℕ) :
    (n % 3 = 2) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 6 = 5) →
    n = 59 := by
  sorry

end NUMINAMATH_GPT_minimum_fruits_l1991_199161


namespace NUMINAMATH_GPT_dogs_prevent_wolf_escape_l1991_199115

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end NUMINAMATH_GPT_dogs_prevent_wolf_escape_l1991_199115


namespace NUMINAMATH_GPT_min_choir_members_l1991_199106

theorem min_choir_members (n : ℕ) : 
  (∀ (m : ℕ), m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) → 
  n = 990 :=
by
  sorry

end NUMINAMATH_GPT_min_choir_members_l1991_199106


namespace NUMINAMATH_GPT_jane_change_l1991_199169

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end NUMINAMATH_GPT_jane_change_l1991_199169


namespace NUMINAMATH_GPT_enclosed_area_l1991_199193

noncomputable def calculateArea : ℝ :=
  ∫ (x : ℝ) in (1 / 2)..2, 1 / x

theorem enclosed_area : calculateArea = 2 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_enclosed_area_l1991_199193


namespace NUMINAMATH_GPT_integer_rational_ratio_l1991_199137

open Real

theorem integer_rational_ratio (a b : ℤ) (h : (a : ℝ) + sqrt b = sqrt (15 + sqrt 216)) : (a : ℚ) / b = 1 / 2 := 
by 
  -- Omitted proof 
  sorry

end NUMINAMATH_GPT_integer_rational_ratio_l1991_199137


namespace NUMINAMATH_GPT_sum_of_squares_base_b_l1991_199123

theorem sum_of_squares_base_b (b : ℕ) (h : (b + 4)^2 + (b + 8)^2 + (2 * b)^2 = 2 * b^3 + 8 * b^2 + 5 * b) :
  (4 * b + 12 : ℕ) = 62 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_base_b_l1991_199123
