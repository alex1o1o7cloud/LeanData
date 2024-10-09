import Mathlib

namespace ab_sum_l1382_138254

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -1 < x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a - 2 }
def complement_A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 5 }
def complement_B : Set ℝ := { x | x ≤ 2 ∨ x ≥ 8 }
def complement_A_and_C (a b : ℝ) : Set ℝ := { x | 6 ≤ x ∧ x ≤ b }

theorem ab_sum (a b: ℝ) (h: (complement_A ∩ C a) = complement_A_and_C a b) : a + b = 13 :=
by
  sorry

end ab_sum_l1382_138254


namespace total_votes_l1382_138247

/-- Let V be the total number of votes. Define the votes received by the candidate and rival. -/
def votes_cast (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) : Prop :=
  votes_candidate = 40 * V / 100 ∧ votes_rival = votes_candidate + 2000 ∧ votes_candidate + votes_rival = V

/-- Prove that the total number of votes is 10000 given the conditions. -/
theorem total_votes (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) :
  votes_cast V votes_candidate votes_rival → V = 10000 :=
by
  sorry

end total_votes_l1382_138247


namespace number_of_students_taking_art_l1382_138266

noncomputable def total_students : ℕ := 500
noncomputable def students_taking_music : ℕ := 50
noncomputable def students_taking_both : ℕ := 10
noncomputable def students_taking_neither : ℕ := 440

theorem number_of_students_taking_art (A : ℕ) (h1: total_students = 500) (h2: students_taking_music = 50) 
  (h3: students_taking_both = 10) (h4: students_taking_neither = 440) : A = 20 :=
by 
  have h5 : total_students = students_taking_music - students_taking_both + A - students_taking_both + 
    students_taking_both + students_taking_neither := sorry
  have h6 : 500 = 40 + A - 10 + 10 + 440 := sorry
  have h7 : 500 = A + 480 := sorry
  have h8 : A = 20 := by linarith 
  exact h8

end number_of_students_taking_art_l1382_138266


namespace dhoni_savings_percent_l1382_138265

variable (E : ℝ) -- Assuming E is Dhoni's last month's earnings

-- Condition 1: Dhoni spent 25% of his earnings on rent
def spent_on_rent (E : ℝ) : ℝ := 0.25 * E

-- Condition 2: Dhoni spent 10% less than what he spent on rent on a new dishwasher
def spent_on_dishwasher (E : ℝ) : ℝ := 0.225 * E

-- Prove the percentage of last month's earnings Dhoni had left over
theorem dhoni_savings_percent (E : ℝ) : 
    52.5 / 100 * E = E - (spent_on_rent E + spent_on_dishwasher E) :=
by
  sorry

end dhoni_savings_percent_l1382_138265


namespace customers_left_is_31_l1382_138276

-- Define the initial number of customers
def initial_customers : ℕ := 33

-- Define the number of additional customers
def additional_customers : ℕ := 26

-- Define the final number of customers after some left and new ones came
def final_customers : ℕ := 28

-- Define the number of customers who left 
def customers_left (x : ℕ) : Prop :=
  (initial_customers - x) + additional_customers = final_customers

-- The proof statement that we aim to prove
theorem customers_left_is_31 : ∃ x : ℕ, customers_left x ∧ x = 31 :=
by
  use 31
  unfold customers_left
  sorry

end customers_left_is_31_l1382_138276


namespace coefficient_x3_l1382_138228

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3 (n k : ℕ) (x : ℤ) :
  let expTerm : ℤ := 1 - x + (1 / x^2017)
  let expansion := fun (k : ℕ) => binomial n k • ((1 - x)^(n - k) * (1 / x^2017)^k)
  (n = 9) → (k = 3) →
  (expansion k) = -84 :=
  by
    intros
    sorry

end coefficient_x3_l1382_138228


namespace arithmetic_sequence_fraction_zero_l1382_138207

noncomputable def arithmetic_sequence_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_fraction_zero (a1 d : ℝ) 
    (h1 : a1 ≠ 0) (h9 : arithmetic_sequence_term a1 d 9 = 0) :
  (arithmetic_sequence_term a1 d 1 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 11 + 
   arithmetic_sequence_term a1 d 16) / 
  (arithmetic_sequence_term a1 d 7 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 14) = 0 :=
by
  sorry

end arithmetic_sequence_fraction_zero_l1382_138207


namespace binary_mult_div_to_decimal_l1382_138264

theorem binary_mult_div_to_decimal:
  let n1 := 2 ^ 5 + 2 ^ 4 + 2 ^ 2 + 2 ^ 1 -- This represents 101110_2
  let n2 := 2 ^ 6 + 2 ^ 4 + 2 ^ 2         -- This represents 1010100_2
  let d := 2 ^ 2                          -- This represents 100_2
  n1 * n2 / d = 2995 := 
by
  sorry

end binary_mult_div_to_decimal_l1382_138264


namespace tangent_line_eq_monotonic_intervals_extremes_f_l1382_138204

variables {a x : ℝ}

noncomputable def f (a x : ℝ) : ℝ := -1/3 * x^3 + 2 * a * x^2 - 3 * a^2 * x
noncomputable def f' (a x : ℝ) : ℝ := -x^2 + 4 * a * x - 3 * a^2

theorem tangent_line_eq {a : ℝ} (h : a = -1) : (∃ y, y = f (-1) (-2) ∧ 3 * x - 3 * y + 8 = 0) := sorry

theorem monotonic_intervals_extremes {a : ℝ} (h : 0 < a) :
  (∀ x, (a < x ∧ x < 3 * a → 0 < f' a x) ∧ 
        (x < a ∨ 3 * a < x → f' a x < 0) ∧ 
        (f a (3 * a) = 0 ∧ f a a = -4/3 * a^3)) := sorry

theorem f'_inequality_range (h1 : ∀ x, 2 * a ≤ x ∧ x ≤ 2 * a + 2 → |f' a x| ≤ 3 * a) :
  (1 ≤ a ∧ a ≤ 3) := sorry

end tangent_line_eq_monotonic_intervals_extremes_f_l1382_138204


namespace range_of_a_l1382_138294

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → quadratic_function a x ≥ quadratic_function a y ∧ y ≤ 4) →
  a ≤ -5 :=
by sorry

end range_of_a_l1382_138294


namespace number_of_cupcakes_l1382_138268

theorem number_of_cupcakes (total gluten_free vegan gluten_free_vegan non_vegan : ℕ) 
    (h1 : gluten_free = total / 2)
    (h2 : vegan = 24)
    (h3 : gluten_free_vegan = vegan / 2)
    (h4 : non_vegan = 28)
    (h5 : gluten_free_vegan = gluten_free / 2) :
    total = 52 :=
by
  sorry

end number_of_cupcakes_l1382_138268


namespace sum_of_consecutive_integers_l1382_138205

theorem sum_of_consecutive_integers (S : ℕ) (hS : S = 560):
  ∃ (N : ℕ), N = 11 ∧ 
  ∀ n (k : ℕ), 2 ≤ n → (n * (2 * k + n - 1)) = 1120 → N = 11 :=
by
  sorry

end sum_of_consecutive_integers_l1382_138205


namespace extended_hexagon_area_l1382_138248

theorem extended_hexagon_area (original_area : ℝ) (side_length_extension : ℝ)
  (original_side_length : ℝ) (new_side_length : ℝ) :
  original_area = 18 ∧ side_length_extension = 1 ∧ original_side_length = 2 
  ∧ new_side_length = original_side_length + 2 * side_length_extension →
  36 = original_area + 6 * (0.5 * side_length_extension * (original_side_length + 1)) := 
sorry

end extended_hexagon_area_l1382_138248


namespace convert_fraction_to_decimal_l1382_138287

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l1382_138287


namespace find_x_l1382_138255

theorem find_x (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_l1382_138255


namespace number_of_possible_values_l1382_138260

theorem number_of_possible_values (a b c : ℕ) (h : a + 11 * b + 111 * c = 1050) :
  ∃ (n : ℕ), 6 ≤ n ∧ n ≤ 1050 ∧ (n % 9 = 6) ∧ (n = a + 2 * b + 3 * c) :=
sorry

end number_of_possible_values_l1382_138260


namespace product_of_three_consecutive_integers_is_multiple_of_6_l1382_138221

theorem product_of_three_consecutive_integers_is_multiple_of_6 (n : ℕ) (h : n > 0) :
    ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k :=
by
  sorry

end product_of_three_consecutive_integers_is_multiple_of_6_l1382_138221


namespace sin_810_eq_one_l1382_138227

theorem sin_810_eq_one : Real.sin (810 * Real.pi / 180) = 1 :=
by
  -- You can add the proof here
  sorry

end sin_810_eq_one_l1382_138227


namespace problem_a_problem_b_l1382_138289
-- Import the entire math library to ensure all necessary functionality is included

-- Define the problem context
variables {x y z : ℝ}

-- State the conditions as definitions
def conditions (x y z : ℝ) : Prop :=
  (x ≤ y) ∧ (y ≤ z) ∧ (x + y + z = 12) ∧ (x^2 + y^2 + z^2 = 54)

-- State the formal proof problems
theorem problem_a (h : conditions x y z) : x ≤ 3 ∧ 5 ≤ z :=
sorry

theorem problem_b (h : conditions x y z) : 
  9 ≤ x * y ∧ x * y ≤ 25 ∧
  9 ≤ y * z ∧ y * z ≤ 25 ∧
  9 ≤ z * x ∧ z * x ≤ 25 :=
sorry

end problem_a_problem_b_l1382_138289


namespace product_of_consecutive_multiples_of_4_divisible_by_192_l1382_138275

theorem product_of_consecutive_multiples_of_4_divisible_by_192 :
  ∀ (n : ℤ), 192 ∣ (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) :=
by
  intro n
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_192_l1382_138275


namespace girls_attended_festival_l1382_138285

variable (g b : ℕ)

theorem girls_attended_festival :
  g + b = 1500 ∧ (2 / 3) * g + (1 / 2) * b = 900 → (2 / 3) * g = 600 := by
  sorry

end girls_attended_festival_l1382_138285


namespace kids_have_equal_eyes_l1382_138231

theorem kids_have_equal_eyes (mom_eyes dad_eyes kids_num total_eyes kids_eyes : ℕ) 
  (h_mom_eyes : mom_eyes = 1) 
  (h_dad_eyes : dad_eyes = 3) 
  (h_kids_num : kids_num = 3) 
  (h_total_eyes : total_eyes = 16) 
  (h_family_eyes : mom_eyes + dad_eyes + kids_num * kids_eyes = total_eyes) :
  kids_eyes = 4 :=
by
  sorry

end kids_have_equal_eyes_l1382_138231


namespace determine_n_eq_1_l1382_138201

theorem determine_n_eq_1 :
  ∃ n : ℝ, (∀ x : ℝ, (x = 2 → (x^3 - 3*x^2 + n = 2*x^3 - 6*x^2 + 5*n))) → n = 1 :=
by
  sorry

end determine_n_eq_1_l1382_138201


namespace bacon_cost_l1382_138252

namespace PancakeBreakfast

def cost_of_stack_pancakes : ℝ := 4.0
def stacks_sold : ℕ := 60
def slices_bacon_sold : ℕ := 90
def total_revenue : ℝ := 420.0

theorem bacon_cost (B : ℝ) 
  (h1 : stacks_sold * cost_of_stack_pancakes + slices_bacon_sold * B = total_revenue) : 
  B = 2 :=
  by {
    sorry
  }

end PancakeBreakfast

end bacon_cost_l1382_138252


namespace avg_weight_BC_l1382_138286

variable (A B C : ℝ)

def totalWeight_ABC := 3 * 45
def totalWeight_AB := 2 * 40
def weight_B := 31

theorem avg_weight_BC : ((B + C) / 2) = 43 :=
  by
    have totalWeight_ABC_eq : A + B + C = totalWeight_ABC := by sorry
    have totalWeight_AB_eq : A + B = totalWeight_AB := by sorry
    have weight_B_eq : B = weight_B := by sorry
    sorry

end avg_weight_BC_l1382_138286


namespace find_initial_candies_l1382_138220

-- Definitions for the conditions
def initial_candies (x : ℕ) : Prop :=
  (3 * x) % 4 = 0 ∧
  (x % 2) = 0 ∧
  ∃ (k : ℕ), 2 ≤ k ∧ k ≤ 6 ∧ (1 * x) / 2 - 20 - k = 4

-- Theorems we need to prove
theorem find_initial_candies (x : ℕ) (h : initial_candies x) : x = 52 ∨ x = 56 ∨ x = 60 :=
sorry

end find_initial_candies_l1382_138220


namespace ratio_problem_l1382_138272

-- Define the conditions and the required proof
theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 :=
by
  sorry

end ratio_problem_l1382_138272


namespace fraction_of_second_eq_fifth_of_first_l1382_138296

theorem fraction_of_second_eq_fifth_of_first 
  (a b x y : ℕ)
  (h1 : y = 40)
  (h2 : x + 35 = 4 * y)
  (h3 : (1 / 5) * x = (a / b) * y) 
  (hb : b ≠ 0):
  a / b = 5 / 8 := by
  sorry

end fraction_of_second_eq_fifth_of_first_l1382_138296


namespace roots_sum_roots_product_algebraic_expression_l1382_138216

theorem roots_sum (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 + x2 = 1 :=
sorry

theorem roots_product (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 * x2 = -1 :=
sorry

theorem algebraic_expression (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1^2 + x2^2 = 3 :=
sorry

end roots_sum_roots_product_algebraic_expression_l1382_138216


namespace lena_more_candy_bars_than_nicole_l1382_138271

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end lena_more_candy_bars_than_nicole_l1382_138271


namespace present_age_of_son_l1382_138219

variable (S M : ℕ)

theorem present_age_of_son :
  (M = S + 30) ∧ (M + 2 = 2 * (S + 2)) → S = 28 :=
by
  sorry

end present_age_of_son_l1382_138219


namespace solve_eq_l1382_138241

theorem solve_eq (x : ℝ) (h : 2 - 1 / (2 - x) = 1 / (2 - x)) : x = 1 := 
sorry

end solve_eq_l1382_138241


namespace time_for_Dawson_l1382_138244

variable (D : ℝ)
variable (Henry_time : ℝ := 7)
variable (avg_time : ℝ := 22.5)

theorem time_for_Dawson (h : avg_time = (D + Henry_time) / 2) : D = 38 := 
by 
  sorry

end time_for_Dawson_l1382_138244


namespace necessary_condition_real_roots_l1382_138257

theorem necessary_condition_real_roots (a : ℝ) :
  (a >= 1 ∨ a <= -2) → (∃ x : ℝ, x^2 - a * x + 1 = 0) :=
by
  sorry

end necessary_condition_real_roots_l1382_138257


namespace polynomial_has_no_real_roots_l1382_138279

theorem polynomial_has_no_real_roots :
  ∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2 ≠ 0 :=
by
  sorry

end polynomial_has_no_real_roots_l1382_138279


namespace remainder_divided_by_82_l1382_138290

theorem remainder_divided_by_82 (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) ↔ (∃ m : ℤ, x + 13 = 41 * m + 18) :=
by
  sorry

end remainder_divided_by_82_l1382_138290


namespace sum_of_f1_possible_values_l1382_138223

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f1_possible_values :
  (∀ (x y : ℝ), f (f (x - y)) = f x * f y - f x + f y - 2 * x * y) →
  (f 1 = -1) := sorry

end sum_of_f1_possible_values_l1382_138223


namespace hyperbola_center_is_equidistant_l1382_138256

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_center_is_equidistant (F1 F2 C : ℝ × ℝ) 
  (hF1 : F1 = (3, -2)) 
  (hF2 : F2 = (11, 6))
  (hC : C = ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)) :
  C = (7, 2) ∧ distance C F1 = distance C F2 :=
by
  -- Fill in with the appropriate proofs
  sorry

end hyperbola_center_is_equidistant_l1382_138256


namespace solve_for_x_l1382_138258

theorem solve_for_x (x : ℝ) (h : (3 / 4) - (1 / 2) = 1 / x) : x = 4 :=
sorry

end solve_for_x_l1382_138258


namespace intersection_point_l1382_138232

theorem intersection_point (x y : ℝ) (h1 : y = x + 1) (h2 : y = -x + 1) : (x = 0) ∧ (y = 1) := 
by
  sorry

end intersection_point_l1382_138232


namespace hcf_of_given_numbers_l1382_138291

def hcf (x y : ℕ) : ℕ := Nat.gcd x y

theorem hcf_of_given_numbers :
  ∃ (A B : ℕ), A = 33 ∧ A * B = 363 ∧ hcf A B = 11 := 
by
  sorry

end hcf_of_given_numbers_l1382_138291


namespace part1_part2_l1382_138226

noncomputable def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem part1 : 
  (∀ x, f x 1 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
sorry

theorem part2 :
  (∀ a, (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2)) :=
sorry

end part1_part2_l1382_138226


namespace number_divisible_by_20p_l1382_138273

noncomputable def floor_expr (p : ℕ) : ℤ :=
  Int.floor ((2 + Real.sqrt 5) ^ p - 2 ^ (p + 1))

theorem number_divisible_by_20p (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) :
  ∃ k : ℤ, floor_expr p = k * 20 * p :=
by
  sorry

end number_divisible_by_20p_l1382_138273


namespace simplify_and_evaluate_expression_l1382_138298

theorem simplify_and_evaluate_expression (m n : ℤ) (h_m : m = -1) (h_n : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l1382_138298


namespace ring_stack_distance_l1382_138243

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end ring_stack_distance_l1382_138243


namespace question1_question2_application_l1382_138292

theorem question1: (-4)^2 - (-3) * (-5) = 1 := by
  sorry

theorem question2 (a : ℝ) (h : a = -4) : a^2 - (a + 1) * (a - 1) = 1 := by
  sorry

theorem application (a : ℝ) (h : a = 1.35) : a * (a - 1) * 2 * a - a^3 - a * (a - 1)^2 = -1.35 := by
  sorry

end question1_question2_application_l1382_138292


namespace total_earnings_correct_l1382_138218

-- Define the earnings of each individual
def SalvadorEarnings := 1956
def SantoEarnings := SalvadorEarnings / 2
def MariaEarnings := 3 * SantoEarnings
def PedroEarnings := SantoEarnings + MariaEarnings

-- Define the total earnings calculation
def TotalEarnings := SalvadorEarnings + SantoEarnings + MariaEarnings + PedroEarnings

-- State the theorem to prove
theorem total_earnings_correct :
  TotalEarnings = 9780 :=
sorry

end total_earnings_correct_l1382_138218


namespace trapezoid_perimeter_is_correct_l1382_138208

noncomputable def trapezoid_perimeter_proof : ℝ :=
  let EF := 60
  let θ := Real.pi / 4 -- 45 degrees in radians
  let h := 30 * Real.sqrt 2
  let GH := EF + 2 * h / Real.tan θ
  let EG := h / Real.tan θ
  EF + GH + 2 * EG -- Perimeter calculation

theorem trapezoid_perimeter_is_correct :
  trapezoid_perimeter_proof = 180 + 60 * Real.sqrt 2 := 
by
  sorry

end trapezoid_perimeter_is_correct_l1382_138208


namespace max_a_l1382_138212

variable {a x : ℝ}

theorem max_a (h : x^2 - 2 * x - 3 > 0 → x < a ∧ ¬ (x < a → x^2 - 2 * x - 3 > 0)) : a = 3 :=
sorry

end max_a_l1382_138212


namespace total_tosses_correct_l1382_138240

def num_heads : Nat := 3
def num_tails : Nat := 7
def total_tosses : Nat := num_heads + num_tails

theorem total_tosses_correct : total_tosses = 10 := by
  sorry

end total_tosses_correct_l1382_138240


namespace longer_segment_of_triangle_l1382_138299

theorem longer_segment_of_triangle {a b c : ℝ} (h_triangle : a = 40 ∧ b = 90 ∧ c = 100) (h_altitude : ∃ h, h > 0) : 
  ∃ (longer_segment : ℝ), longer_segment = 82.5 :=
by 
  sorry

end longer_segment_of_triangle_l1382_138299


namespace lucy_age_l1382_138217

theorem lucy_age (Inez_age : ℕ) (Zack_age : ℕ) (Jose_age : ℕ) (Lucy_age : ℕ) 
  (h1 : Inez_age = 18) 
  (h2 : Zack_age = Inez_age + 4) 
  (h3 : Jose_age = Zack_age - 6) 
  (h4 : Lucy_age = Jose_age + 2) : 
  Lucy_age = 18 := by
sorry

end lucy_age_l1382_138217


namespace add_fractions_l1382_138270

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l1382_138270


namespace total_vessels_proof_l1382_138242

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end total_vessels_proof_l1382_138242


namespace tyrone_gave_marbles_to_eric_l1382_138230

theorem tyrone_gave_marbles_to_eric (initial_tyrone_marbles : ℕ) (initial_eric_marbles : ℕ) (marbles_given : ℕ) :
  initial_tyrone_marbles = 150 ∧ initial_eric_marbles = 30 ∧ (initial_tyrone_marbles - marbles_given = 3 * initial_eric_marbles) → marbles_given = 60 :=
by
  sorry

end tyrone_gave_marbles_to_eric_l1382_138230


namespace sequence_general_term_l1382_138274

noncomputable def b_n (n : ℕ) : ℚ := 2 * n - 1
noncomputable def c_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem sequence_general_term (n : ℕ) : 
  b_n n + c_n n = (4 * n^2 + n - 1) / (2 * n + 1) :=
by sorry

end sequence_general_term_l1382_138274


namespace john_subtracts_79_l1382_138293

theorem john_subtracts_79 :
  let a := 40
  let b := 1
  let n := (a - b) * (a - b)
  n = a * a - 79
:= by
  sorry

end john_subtracts_79_l1382_138293


namespace fraction_of_girls_in_debate_l1382_138284

theorem fraction_of_girls_in_debate (g b : ℕ) (h : g = b) :
  ((2 / 3) * g) / ((2 / 3) * g + (3 / 5) * b) = 30 / 57 :=
by
  sorry

end fraction_of_girls_in_debate_l1382_138284


namespace SquareArea_l1382_138213

theorem SquareArea (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π / 4) : s * s = 9 := 
by 
  sorry

end SquareArea_l1382_138213


namespace bus_stoppage_time_l1382_138295

theorem bus_stoppage_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (reduction_in_speed : speed_excluding_stoppages - speed_including_stoppages = 8) :
  ∃ t : ℝ, t = 9.6 := 
sorry

end bus_stoppage_time_l1382_138295


namespace red_candies_l1382_138234

theorem red_candies (R Y B : ℕ) 
  (h1 : Y = 3 * R - 20)
  (h2 : B = Y / 2)
  (h3 : R + B = 90) :
  R = 40 :=
by
  sorry

end red_candies_l1382_138234


namespace quadratic_inequality_solution_range_l1382_138282

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x, 1 < x ∧ x < 4 ∧ x^2 - 4 * x - 2 - a > 0) → a < -2 :=
sorry

end quadratic_inequality_solution_range_l1382_138282


namespace option_B_valid_l1382_138225

-- Definitions derived from conditions
def at_least_one_black (balls : List Bool) : Prop :=
  ∃ b ∈ balls, b = true

def both_black (balls : List Bool) : Prop :=
  balls = [true, true]

def exactly_one_black (balls : List Bool) : Prop :=
  balls.count true = 1

def exactly_two_black (balls : List Bool) : Prop :=
  balls.count true = 2

def mutually_exclusive (P Q : Prop) : Prop :=
  P ∧ Q → False

def non_complementary (P Q : Prop) : Prop :=
  ¬(P → ¬Q) ∧ ¬(¬P → Q)

-- Balls: true represents a black ball, false represents a red ball.
def all_draws := [[true, true], [true, false], [false, true], [false, false]]

-- Proof statement
theorem option_B_valid :
  (mutually_exclusive (exactly_one_black [true, false]) (exactly_two_black [true, true])) ∧ 
  (non_complementary (exactly_one_black [true, false]) (exactly_two_black [true, true])) :=
  sorry

end option_B_valid_l1382_138225


namespace smallest_number_divisible_l1382_138237

theorem smallest_number_divisible (x y : ℕ) (h : x + y = 4728) 
  (h1 : (x + y) % 27 = 0) 
  (h2 : (x + y) % 35 = 0) 
  (h3 : (x + y) % 25 = 0) 
  (h4 : (x + y) % 21 = 0) : 
  x = 4725 := by 
  sorry

end smallest_number_divisible_l1382_138237


namespace problem_value_l1382_138281

theorem problem_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) - abs x) - x = 4046 :=
by
  sorry

end problem_value_l1382_138281


namespace largest_number_l1382_138259

theorem largest_number 
  (a b c : ℝ) (h1 : a = 0.8) (h2 : b = 1/2) (h3 : c = 0.9) (h4 : a ≤ 2) (h5 : b ≤ 2) (h6 : c ≤ 2) :
  max (max a b) c = 0.9 :=
by
  sorry

end largest_number_l1382_138259


namespace scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l1382_138214

open Nat

-- Definitions for combinations and permutations
def binomial (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))
def variations (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- Scenario a: Each path can be used by at most one person and at most once
theorem scenario_a : binomial 5 2 * binomial 3 2 = 30 := by sorry

-- Scenario b: Each path can be used twice but only in different directions
theorem scenario_b : binomial 5 2 * binomial 5 2 = 100 := by sorry

-- Scenario c: No restrictions
theorem scenario_c : (5 * 5) * (5 * 5) = 625 := by sorry

-- Scenario d: Same as a) with two people distinguished
theorem scenario_d : variations 5 2 * variations 3 2 = 120 := by sorry

-- Scenario e: Same as b) with two people distinguished
theorem scenario_e : variations 5 2 * variations 5 2 = 400 := by sorry

-- Scenario f: Same as c) with two people distinguished
theorem scenario_f : (5 * 5) * (5 * 5) = 625 := by sorry

end scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l1382_138214


namespace initial_students_count_l1382_138253

theorem initial_students_count (n : ℕ) (W : ℝ) 
  (h1 : W = n * 28) 
  (h2 : W + 1 = (n + 1) * 27.1) : 
  n = 29 := by
  sorry

end initial_students_count_l1382_138253


namespace largest_d_for_g_of_minus5_l1382_138250

theorem largest_d_for_g_of_minus5 (d : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + d = -5) → d ≤ -4 :=
by
-- Proof steps will be inserted here
sorry

end largest_d_for_g_of_minus5_l1382_138250


namespace f_eq_zero_range_x_l1382_138277

-- Definition of the function f on domain ℝ*
def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_domain : ∀ x : ℝ, x ≠ 0 → f x = f x
axiom f_4 : f 4 = 1
axiom f_multiplicative : ∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → f (x1 * x2) = f x1 + f x2
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

-- Problem (1): Prove f(1) = 0
theorem f_eq_zero : f 1 = 0 :=
sorry

-- Problem (2): Prove range 3 < x ≤ 5 given the inequality condition
theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 :=
sorry

end f_eq_zero_range_x_l1382_138277


namespace prime_base_values_l1382_138269

theorem prime_base_values :
  ∀ p : ℕ, Prime p →
    (2 * p^3 + p^2 + 6 + 4 * p^2 + p + 4 + 2 * p^2 + p + 5 + 2 * p^2 + 2 * p + 2 + 9 =
     4 * p^2 + 3 * p + 3 + 5 * p^2 + 7 * p + 2 + 3 * p^2 + 2 * p + 1) →
    false :=
by {
  sorry
}

end prime_base_values_l1382_138269


namespace combine_like_terms_l1382_138210

theorem combine_like_terms (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := 
  sorry

end combine_like_terms_l1382_138210


namespace find_percentage_l1382_138267

theorem find_percentage : 
  ∀ (P : ℕ), 
  (50 - 47 = (P / 100) * 15) →
  P = 20 := 
by
  intro P h
  sorry

end find_percentage_l1382_138267


namespace part1_part2_l1382_138200

-- Definitions for the conditions
def A : Set ℝ := {x : ℝ | 2 * x - 4 < 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}
def U : Set ℝ := Set.univ

-- The questions translated as Lean theorems
theorem part1 : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

theorem part2 : (U \ A) ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} := by
  sorry

end part1_part2_l1382_138200


namespace inequality_solution_set_l1382_138288

theorem inequality_solution_set (x : ℝ) : (x - 1) * abs (x + 2) ≥ 0 ↔ (x ≥ 1 ∨ x = -2) :=
by
  sorry

end inequality_solution_set_l1382_138288


namespace albert_runs_track_l1382_138245

theorem albert_runs_track (x : ℕ) (track_distance : ℕ) (total_distance : ℕ) (additional_laps : ℕ) 
(h1 : track_distance = 9)
(h2 : total_distance = 99)
(h3 : additional_laps = 5)
(h4 : total_distance = track_distance * x + track_distance * additional_laps) :
x = 6 :=
by
  sorry

end albert_runs_track_l1382_138245


namespace sum_of_largest_three_consecutive_numbers_l1382_138236

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l1382_138236


namespace negation_of_square_positive_l1382_138297

open Real

-- Define the original proposition
def prop_square_positive : Prop :=
  ∀ x : ℝ, x^2 > 0

-- Define the negation of the original proposition
def prop_square_not_positive : Prop :=
  ∃ x : ℝ, ¬ (x^2 > 0)

-- The theorem that asserts the logical equivalence for the negation
theorem negation_of_square_positive :
  ¬ prop_square_positive ↔ prop_square_not_positive :=
by sorry

end negation_of_square_positive_l1382_138297


namespace elena_butter_l1382_138209

theorem elena_butter (cups_flour butter : ℕ) (h1 : cups_flour * 4 = 28) (h2 : butter * 4 = 12) : butter = 3 := 
by
  sorry

end elena_butter_l1382_138209


namespace hot_dog_remainder_l1382_138229

theorem hot_dog_remainder : 35252983 % 6 = 1 :=
by
  sorry

end hot_dog_remainder_l1382_138229


namespace contrapositive_example_l1382_138235

theorem contrapositive_example :
  (∀ x : ℝ, x^2 < 4 → -2 < x ∧ x < 2) ↔ (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
by
  sorry

end contrapositive_example_l1382_138235


namespace range_of_m_l1382_138263

noncomputable def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
noncomputable def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x : ℝ, q x m → p x) : m ≥ 9 :=
sorry

end range_of_m_l1382_138263


namespace angle_bisector_slope_l1382_138202

theorem angle_bisector_slope
  (m₁ m₂ : ℝ) (h₁ : m₁ = 2) (h₂ : m₂ = -1) (k : ℝ)
  (h_k : k = (m₁ + m₂ + Real.sqrt ((m₁ - m₂)^2 + 4)) / 2) :
  k = (1 + Real.sqrt 13) / 2 :=
by
  rw [h₁, h₂] at h_k
  sorry

end angle_bisector_slope_l1382_138202


namespace gcd_polynomial_l1382_138239

theorem gcd_polynomial (b : ℤ) (h : ∃ k : ℤ, b = 2 * 997 * k) : 
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := 
by
  -- Proof would go here, but is omitted as instructed
  sorry

end gcd_polynomial_l1382_138239


namespace original_price_is_correct_l1382_138203

-- Given conditions as Lean definitions
def reduced_price : ℝ := 2468
def reduction_amount : ℝ := 161.46

-- To find the original price including the sales tax
def original_price_including_tax (P : ℝ) : Prop :=
  P - reduction_amount = reduced_price

-- The proof statement to show the price is 2629.46
theorem original_price_is_correct : original_price_including_tax 2629.46 :=
by
  sorry

end original_price_is_correct_l1382_138203


namespace arithmetic_mean_of_reciprocals_is_correct_l1382_138280

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l1382_138280


namespace value_of_m_l1382_138262

theorem value_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 :=
by
  sorry

end value_of_m_l1382_138262


namespace apples_fraction_of_pears_l1382_138222

variables (A O P : ℕ)

-- Conditions
def oranges_condition := O = 3 * A
def pears_condition := P = 4 * O

-- Statement we need to prove
theorem apples_fraction_of_pears (A O P : ℕ) (h1 : O = 3 * A) (h2 : P = 4 * O) : (A : ℚ) / P = 1 / 12 :=
by
  sorry

end apples_fraction_of_pears_l1382_138222


namespace unique_seq_l1382_138206

def seq (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j

theorem unique_seq (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n) : 
  seq a ↔ (∀ n, a n = n) := 
by
  intros
  sorry

end unique_seq_l1382_138206


namespace probability_prime_sum_is_1_9_l1382_138211

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l1382_138211


namespace minimal_range_of_observations_l1382_138224

variable {x1 x2 x3 x4 x5 : ℝ}

def arithmetic_mean (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  (x1 + x2 + x3 + x4 + x5) / 5 = 8

def median (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5

theorem minimal_range_of_observations 
  (h_mean : arithmetic_mean x1 x2 x3 x4 x5)
  (h_median : median x1 x2 x3 x4 x5) : 
  ∃ x1 x2 x3 x4 x5 : ℝ, (x1 + x2 + x3 + x4 + x5) = 40 ∧ x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5 ∧ (x5 - x1) = 5 :=
by 
  sorry

end minimal_range_of_observations_l1382_138224


namespace quadratic_must_have_m_eq_neg2_l1382_138233

theorem quadratic_must_have_m_eq_neg2 (m : ℝ) (h : (m - 2) * x^|m| - 3 * x - 4 = 0) :
  (|m| = 2) ∧ (m ≠ 2) → m = -2 :=
by
  sorry

end quadratic_must_have_m_eq_neg2_l1382_138233


namespace max_value_2x_minus_y_l1382_138283

theorem max_value_2x_minus_y (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) : 2 * x - y ≤ 5 :=
sorry

end max_value_2x_minus_y_l1382_138283


namespace jovial_frogs_not_green_l1382_138238

variables {Frog : Type} (jovial green can_jump can_swim : Frog → Prop)

theorem jovial_frogs_not_green :
  (∀ frog, jovial frog → can_swim frog) →
  (∀ frog, green frog → ¬ can_jump frog) →
  (∀ frog, ¬ can_jump frog → ¬ can_swim frog) →
  (∀ frog, jovial frog → ¬ green frog) :=
by
  intros h1 h2 h3 frog hj
  sorry

end jovial_frogs_not_green_l1382_138238


namespace compound_interest_second_year_l1382_138261

theorem compound_interest_second_year
  (P: ℝ) (r: ℝ) (CI_3 : ℝ) (CI_2 : ℝ)
  (h1 : r = 0.06)
  (h2 : CI_3 = 1272)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1200 :=
by
  sorry

end compound_interest_second_year_l1382_138261


namespace average_of_remaining_two_numbers_l1382_138246

theorem average_of_remaining_two_numbers (S S3 : ℝ) (h_avg5 : S / 5 = 8) (h_avg3 : S3 / 3 = 4) : S / 5 = 8 ∧ S3 / 3 = 4 → (S - S3) / 2 = 14 :=
by 
  sorry

end average_of_remaining_two_numbers_l1382_138246


namespace ratio_area_rectangle_triangle_l1382_138278

-- Define the lengths L and W as positive real numbers
variables {L W : ℝ} (hL : L > 0) (hW : W > 0)

-- Define the area of the rectangle
noncomputable def area_rectangle (L W : ℝ) : ℝ := L * W

-- Define the area of the triangle with base L and height W
noncomputable def area_triangle (L W : ℝ) : ℝ := (1 / 2) * L * W

-- Define the ratio between the area of the rectangle and the area of the triangle
noncomputable def area_ratio (L W : ℝ) : ℝ := area_rectangle L W / area_triangle L W

-- Prove that this ratio is equal to 2
theorem ratio_area_rectangle_triangle : area_ratio L W = 2 := by sorry

end ratio_area_rectangle_triangle_l1382_138278


namespace series_sum_eq_neg_one_l1382_138249

   noncomputable def sum_series : ℝ :=
     ∑' k : ℕ, if k = 0 then 0 else (12 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

   theorem series_sum_eq_neg_one : sum_series = -1 :=
   sorry
   
end series_sum_eq_neg_one_l1382_138249


namespace difference_of_percentages_l1382_138215

variable (x y : ℝ)

theorem difference_of_percentages :
  (0.60 * (50 + x)) - (0.45 * (30 + y)) = 16.5 + 0.60 * x - 0.45 * y := 
sorry

end difference_of_percentages_l1382_138215


namespace range_of_a_if_p_true_l1382_138251

theorem range_of_a_if_p_true : 
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧ x^2 - a * x + 36 ≤ 0) → a ≥ 12 :=
sorry

end range_of_a_if_p_true_l1382_138251
