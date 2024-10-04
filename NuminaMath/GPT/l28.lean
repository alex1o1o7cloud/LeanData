import Mathlib

namespace probability_of_heart_and_joker_l28_28753

-- Define a deck with 54 cards, including jokers
def total_cards : ℕ := 54

-- Define the count of specific cards in the deck
def hearts_count : ℕ := 13
def jokers_count : ℕ := 2
def remaining_cards (x: ℕ) : ℕ := total_cards - x

-- Define the probability of drawing a specific card
def prob_of_first_heart : ℚ := hearts_count / total_cards
def prob_of_second_joker (first_card_a_heart: Bool) : ℚ :=
  if first_card_a_heart then jokers_count / remaining_cards 1 else 0

-- Calculate the probability of drawing a heart first and then a joker
def prob_first_heart_then_joker : ℚ :=
  prob_of_first_heart * prob_of_second_joker true

-- Proving the final probability
theorem probability_of_heart_and_joker :
  prob_first_heart_then_joker = 13 / 1419 := by
  -- Skipping the proof
  sorry

end probability_of_heart_and_joker_l28_28753


namespace minimum_value_of_f_at_zero_inequality_f_geq_term_l28_28315

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem minimum_value_of_f_at_zero (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≥ f a x ∧ f a x = 0) → a = 2 :=
by
  sorry

theorem inequality_f_geq_term (x : ℝ) (hx : x > 1) : 
  f 2 x ≥ 1 / x - Real.exp (1 - x) :=
by
  sorry

end minimum_value_of_f_at_zero_inequality_f_geq_term_l28_28315


namespace solve_for_x_l28_28141

open Real

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 6 * sqrt (4 + x) + 6 * sqrt (4 - x) = 9 * sqrt 2) : 
  x = sqrt 255 / 4 :=
sorry

end solve_for_x_l28_28141


namespace mrs_jackson_decorations_l28_28868

theorem mrs_jackson_decorations (boxes decorations_in_each_box decorations_used : Nat) 
  (h1 : boxes = 4) 
  (h2 : decorations_in_each_box = 15) 
  (h3 : decorations_used = 35) :
  boxes * decorations_in_each_box - decorations_used = 25 := 
  by
  sorry

end mrs_jackson_decorations_l28_28868


namespace solve_for_y_l28_28350

theorem solve_for_y (y : ℝ)
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1 / 9 :=
sorry

end solve_for_y_l28_28350


namespace smallest_four_digit_divisible_by_53_l28_28043

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28043


namespace prob_enter_A_and_exit_F_l28_28649

-- Define the problem description
def entrances : ℕ := 2
def exits : ℕ := 3

-- Define the probabilities
def prob_enter_A : ℚ := 1 / entrances
def prob_exit_F : ℚ := 1 / exits

-- Statement that encapsulates the proof problem
theorem prob_enter_A_and_exit_F : prob_enter_A * prob_exit_F = 1 / 6 := 
by sorry

end prob_enter_A_and_exit_F_l28_28649


namespace Ariel_current_age_l28_28987

-- Define the conditions
def Ariel_birth_year : Nat := 1992
def Ariel_start_fencing_year : Nat := 2006
def Ariel_fencing_years : Nat := 16

-- Define the problem as a theorem
theorem Ariel_current_age :
  (Ariel_start_fencing_year - Ariel_birth_year) + Ariel_fencing_years = 30 := by
sorry

end Ariel_current_age_l28_28987


namespace smallest_four_digit_divisible_by_53_l28_28022

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28022


namespace smallest_four_digit_divisible_by_53_l28_28081

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28081


namespace solve_fraction_equation_l28_28729

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ x = -2 := by
sorry

end solve_fraction_equation_l28_28729


namespace smallest_four_digit_multiple_of_53_l28_28004

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28004


namespace decreasing_intervals_and_minimum_value_l28_28481

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem decreasing_intervals_and_minimum_value (a : ℝ) :
  (∀ x, x ∈ (-∞, -1) ∪ (3, ∞) → has_deriv_at (λ x, f x a) (-3*x^2 + 6*x + 9) x ∧ 
                                      deriv (λ x, f x a) x < 0) ∧
  (∀ a, (∀ x ∈ set.Icc (-2 : ℝ) 2, f x a ≤ 20) →
    ∃ a', a' = -2 ∧ (∀ x, x ∈ set.Icc (-2 : ℝ) 2 → f (-1) a' = -7)) := sorry

end decreasing_intervals_and_minimum_value_l28_28481


namespace find_digit_B_l28_28611

theorem find_digit_B (A B : ℕ) (h1 : A3B = 100 * A + 30 + B)
  (h2 : 0 ≤ A ∧ A ≤ 9)
  (h3 : 0 ≤ B ∧ B ≤ 9)
  (h4 : A3B - 41 = 591) : 
  B = 2 := 
by sorry

end find_digit_B_l28_28611


namespace Mike_additional_money_needed_proof_l28_28356

-- Definitions of conditions
def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_has_percentage : ℝ := 0.40

-- Definitions of intermediate calculations
def discounted_phone_cost : ℝ := phone_cost * (1 - phone_discount)
def discounted_smartwatch_cost : ℝ := smartwatch_cost * (1 - smartwatch_discount)
def total_cost_before_tax : ℝ := discounted_phone_cost + discounted_smartwatch_cost
def total_tax : ℝ := total_cost_before_tax * sales_tax
def total_cost_after_tax : ℝ := total_cost_before_tax + total_tax
def mike_has_amount : ℝ := total_cost_after_tax * mike_has_percentage
def additional_money_needed : ℝ := total_cost_after_tax - mike_has_amount

-- Theorem statement
theorem Mike_additional_money_needed_proof :
  additional_money_needed = 1023.99 :=
by sorry

end Mike_additional_money_needed_proof_l28_28356


namespace necessary_and_sufficient_condition_l28_28492

theorem necessary_and_sufficient_condition (x : ℝ) : (0 < (1 / x) ∧ (1 / x) < 1) ↔ (1 < x) := sorry

end necessary_and_sufficient_condition_l28_28492


namespace total_crayons_l28_28942

theorem total_crayons (Wanda Dina Jacob : Nat) (hWanda : Wanda = 62) (hDina : Dina = 28) (hJacob : Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  -- We first use the given conditions to substitute the values
  rw [hWanda, hDina, hJacob]
  -- Simplify the expression to verify the result is as expected
  rw [Nat.succ_sub, Nat.sub_self, Nat.add_comm, Nat.add_assoc]
  norm_num
  sorry

end total_crayons_l28_28942


namespace simplify_expression_correct_l28_28363

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l28_28363


namespace log4_21_correct_l28_28518

noncomputable def log4_21 (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2)
                                     (h2 : Real.log 2 = b * Real.log 7) : ℝ :=
  (a * b + 1) / (2 * b)

theorem log4_21_correct (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2) 
                        (h2 : Real.log 2 = b * Real.log 7) : 
  log4_21 a b h1 h2 = (a * b + 1) / (2 * b) := 
sorry

end log4_21_correct_l28_28518


namespace smallest_four_digit_divisible_by_53_l28_28078

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28078


namespace min_value_option_C_l28_28795

noncomputable def option_A (x : ℝ) : ℝ := x + 4 / x
noncomputable def option_B (x : ℝ) : ℝ := Real.sin x + 4 / Real.sin x
noncomputable def option_C (x : ℝ) : ℝ := Real.exp x + 4 * Real.exp (-x)
noncomputable def option_D (x : ℝ) : ℝ := Real.log x / Real.log 3 + 4 * Real.log 3 / Real.log x

theorem min_value_option_C : ∃ x : ℝ, option_C x = 4 :=
by
  use 0
  -- Proof goes here.
  sorry

end min_value_option_C_l28_28795


namespace div_by_3_iff_n_form_l28_28812

theorem div_by_3_iff_n_form (n : ℕ) : (3 ∣ (n * 2^n + 1)) ↔ (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 2) :=
by
  sorry

end div_by_3_iff_n_form_l28_28812


namespace range_of_m_for_false_proposition_l28_28182

theorem range_of_m_for_false_proposition :
  ¬ (∃ x : ℝ, x^2 - m * x - m ≤ 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
sorry

end range_of_m_for_false_proposition_l28_28182


namespace ball_hits_ground_at_correct_time_l28_28770

def initial_velocity : ℝ := 7
def initial_height : ℝ := 10

-- The height function as given by the condition
def height_function (t : ℝ) : ℝ := -4.9 * t^2 + initial_velocity * t + initial_height

-- Statement
theorem ball_hits_ground_at_correct_time :
  ∃ t : ℝ, height_function t = 0 ∧ t = 2313 / 1000 :=
by
  sorry

end ball_hits_ground_at_correct_time_l28_28770


namespace cindy_gave_25_pens_l28_28952

theorem cindy_gave_25_pens (initial_pens mike_gave pens_given_sharon final_pens : ℕ) (h1 : initial_pens = 5) (h2 : mike_gave = 20) (h3 : pens_given_sharon = 19) (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gave - pens_given_sharon + 25 :=
by 
  -- Insert the proof here later
  sorry

end cindy_gave_25_pens_l28_28952


namespace keys_missing_l28_28339

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l28_28339


namespace binomial_coefficient_and_factorial_l28_28132

open Nat

/--
  Given:
    - The binomial coefficient definition: Nat.choose n k = n! / (k! * (n - k)!)
    - The factorial definition: Nat.factorial n = n * (n - 1) * ... * 1
  Prove:
    Nat.choose 60 3 * Nat.factorial 10 = 124467072000
-/
theorem binomial_coefficient_and_factorial :
  Nat.choose 60 3 * Nat.factorial 10 = 124467072000 :=
by
  sorry

end binomial_coefficient_and_factorial_l28_28132


namespace c_sub_a_eq_60_l28_28954

theorem c_sub_a_eq_60 (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := 
by 
  sorry

end c_sub_a_eq_60_l28_28954


namespace vacationers_city_correctness_l28_28142

noncomputable def vacationer_cities : Prop :=
  ∃ (city : String → String),
    (city "Amelie" = "Acapulco" ∨ city "Amelie" = "Brest" ∨ city "Amelie" = "Madrid") ∧
    (city "Benoit" = "Acapulco" ∨ city "Benoit" = "Brest" ∨ city "Benoit" = "Madrid") ∧
    (city "Pierre" = "Paris" ∨ city "Pierre" = "Brest" ∨ city "Pierre" = "Madrid") ∧
    (city "Melanie" = "Acapulco" ∨ city "Melanie" = "Brest" ∨ city "Melanie" = "Madrid") ∧
    (city "Charles" = "Acapulco" ∨ city "Charles" = "Brest" ∨ city "Charles" = "Madrid") ∧
    -- Conditions stated by participants
    ((city "Amelie" = "Acapulco") ∨ (city "Amelie" ≠ "Acapulco" ∧ city "Benoit" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Benoit" = "Brest") ∨ (city "Benoit" ≠ "Brest" ∧ city "Charles" = "Brest" ∧ city "Pierre" = "Paris")) ∧
    ((city "Pierre" ≠ "France") ∨ (city "Pierre" = "Paris" ∧ city "Amelie" ≠ "France" ∧ city "Melanie" = "Madrid")) ∧
    ((city "Melanie" = "Clermont-Ferrand") ∨ (city "Melanie" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Charles" = "Clermont-Ferrand") ∨ (city "Charles" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Benoit" = "Acapulco"))

theorem vacationers_city_correctness : vacationer_cities :=
  sorry

end vacationers_city_correctness_l28_28142


namespace smallest_four_digit_divisible_by_53_l28_28071

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28071


namespace convex_octagon_min_obtuse_l28_28848

-- Define a type for a polygon (here specifically an octagon)
structure Polygon (n : ℕ) :=
(vertices : ℕ)
(convex : Prop)

-- Define that an octagon is a specific polygon with 8 vertices
def octagon : Polygon 8 :=
{ vertices := 8,
  convex := sorry }

-- Define the predicate for convex polygons
def is_convex (poly : Polygon 8) : Prop := poly.convex

-- Defining the statement that a convex octagon has at least 5 obtuse interior angles
theorem convex_octagon_min_obtuse (poly : Polygon 8) (h : is_convex poly) : ∃ (n : ℕ), n = 5 :=
sorry

end convex_octagon_min_obtuse_l28_28848


namespace average_salary_company_l28_28477

-- Define the conditions
def num_managers : Nat := 15
def num_associates : Nat := 75
def avg_salary_managers : ℤ := 90000
def avg_salary_associates : ℤ := 30000

-- Define the goal to prove
theorem average_salary_company : 
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
  sorry

end average_salary_company_l28_28477


namespace sum_of_coefficients_is_1_l28_28598

-- Given conditions:
def polynomial_expansion (x y : ℤ) := (x - 2 * y) ^ 18

-- Proof statement:
theorem sum_of_coefficients_is_1 : (polynomial_expansion 1 1) = 1 := by
  -- The proof itself is omitted as per the instruction
  sorry

end sum_of_coefficients_is_1_l28_28598


namespace rhombus_perimeter_l28_28736

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l28_28736


namespace maximum_rabbits_condition_l28_28397

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l28_28397


namespace tan_theta_eq_2_l28_28311

theorem tan_theta_eq_2 {θ : ℝ} (h : Real.tan θ = 2) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (Real.cos θ ^ 2 + Real.sin θ * Real.cos θ) = 8 / 3 :=
by
  sorry

end tan_theta_eq_2_l28_28311


namespace g_minus_6_eq_neg_20_l28_28892

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l28_28892


namespace sequence_evaluation_l28_28255

noncomputable def a : ℕ → ℤ → ℤ
| 0, x => 1
| 1, x => x^2 + x + 1
| (n + 2), x => (x^n + 1) * a (n + 1) x - a n x 

theorem sequence_evaluation : a 2010 1 = 4021 := by
  sorry

end sequence_evaluation_l28_28255


namespace parabola_solution_unique_l28_28149

theorem parabola_solution_unique (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 4 * a + 2 * b + c = -1) (h3 : 4 * a + b = 1) :
  a = 3 ∧ b = -11 ∧ c = 9 := 
  by sorry

end parabola_solution_unique_l28_28149


namespace minimize_quadratic_function_l28_28460

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l28_28460


namespace sets_equal_l28_28428

def M := { u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

theorem sets_equal : M = N :=
by sorry

end sets_equal_l28_28428


namespace value_of_expression_l28_28609

theorem value_of_expression (x y : ℝ) (h1 : x = 12) (h2 : y = 18) : 3 * (x - y) * (x + y) = -540 :=
by
  rw [h1, h2]
  sorry

end value_of_expression_l28_28609


namespace smallest_four_digit_divisible_by_53_l28_28090

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28090


namespace square_of_1023_l28_28996

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l28_28996


namespace place_numbers_l28_28876

theorem place_numbers (a b c d : ℕ) (hab : Nat.gcd a b = 1) (hac : Nat.gcd a c = 1) 
  (had : Nat.gcd a d = 1) (hbc : Nat.gcd b c = 1) (hbd : Nat.gcd b d = 1) 
  (hcd : Nat.gcd c d = 1) :
  ∃ (bc ad ab cd abcd : ℕ), 
    bc = b * c ∧ ad = a * d ∧ ab = a * b ∧ cd = c * d ∧ abcd = a * b * c * d ∧
    Nat.gcd bc abcd > 1 ∧ Nat.gcd ad abcd > 1 ∧ Nat.gcd ab abcd > 1 ∧ 
    Nat.gcd cd abcd > 1 ∧
    Nat.gcd ab cd = 1 ∧ Nat.gcd ab ad = 1 ∧ Nat.gcd ab bc = 1 ∧ 
    Nat.gcd cd ad = 1 ∧ Nat.gcd cd bc = 1 ∧ Nat.gcd ad bc = 1 :=
by
  sorry

end place_numbers_l28_28876


namespace total_hens_and_cows_l28_28638

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end total_hens_and_cows_l28_28638


namespace maximum_value_N_27_l28_28387

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l28_28387


namespace union_of_sets_l28_28351

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 5 }
def B := { x : ℝ | 3 < x ∧ x < 9 }

theorem union_of_sets : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 9 } :=
by
  sorry

end union_of_sets_l28_28351


namespace minimize_quadratic_l28_28455

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l28_28455


namespace trigonometric_identity_l28_28682

theorem trigonometric_identity
  (α β : Real)
  (h : Real.cos α * Real.cos β - Real.sin α * Real.sin β = 0) :
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 ∨
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = -1 := by
  sorry

end trigonometric_identity_l28_28682


namespace profit_percent_l28_28101

-- Definitions for the given conditions
variables (P C : ℝ)
-- Condition given: selling at (2/3) of P results in a loss of 5%, i.e., (2/3) * P = 0.95 * C
def condition : Prop := (2 / 3) * P = 0.95 * C

-- Theorem statement: Given the condition, the profit percent when selling at price P is 42.5%
theorem profit_percent (h : condition P C) : ((P - C) / C) * 100 = 42.5 :=
sorry

end profit_percent_l28_28101


namespace smallest_four_digit_divisible_by_53_l28_28058

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28058


namespace find_g_neg_six_l28_28918

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l28_28918


namespace prism_faces_l28_28641

theorem prism_faces (E L : ℕ) (hE : E = 21) (hFormula : E = 3 * L) : 2 + L = 9 :=
by
  rw [hE, hFormula]
  have hL : L = 7 := sorry 
  rw [hL]
  rfl

end prism_faces_l28_28641


namespace trig_expression_value_l28_28821

theorem trig_expression_value (x : ℝ) (h : Real.tan x = 1/2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 :=
by
  sorry

end trig_expression_value_l28_28821


namespace barbara_saving_weeks_l28_28506

theorem barbara_saving_weeks :
  ∀ (cost_of_watch allowance : ℕ) (current_savings weeks : ℕ),
  cost_of_watch = 100 →
  allowance = 5 →
  weeks = 10 →
  current_savings = 20 →
  ∃ w : ℕ, w = (cost_of_watch - current_savings) / allowance ∧ w = 16 :=
by
  intros cost_of_watch allowance current_savings weeks h1 h2 h3 h4
  use ((cost_of_watch - current_savings) / allowance)
  split
  · rw [h1, h2, h4]
  · sorry

end barbara_saving_weeks_l28_28506


namespace simplify_expression_l28_28377

theorem simplify_expression (x y z : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 :=
by
  sorry

end simplify_expression_l28_28377


namespace find_age_difference_l28_28112

variable (a b c : ℕ)

theorem find_age_difference (h : a + b = b + c + 20) : c = a - 20 :=
by
  sorry

end find_age_difference_l28_28112


namespace log_prime_factor_inequality_l28_28878

open Real

noncomputable def num_prime_factors (n : ℕ) : ℕ := sorry 

theorem log_prime_factor_inequality (n : ℕ) (h : 0 < n) : 
  log n ≥ num_prime_factors n * log 2 := 
sorry

end log_prime_factor_inequality_l28_28878


namespace jeremy_money_ratio_l28_28857

theorem jeremy_money_ratio :
  let cost_computer := 3000
  let cost_accessories := 0.10 * cost_computer
  let money_left := 2700
  let total_spent := cost_computer + cost_accessories
  let money_before_purchase := total_spent + money_left
  (money_before_purchase / cost_computer) = 2 := by
  sorry

end jeremy_money_ratio_l28_28857


namespace limping_rook_adjacent_sum_not_divisible_by_4_l28_28959

/-- Problem statement: A limping rook traversed a 10 × 10 board,
visiting each square exactly once with numbers 1 through 100
written in the order visited.
Prove that the sum of the numbers in any two adjacent cells
is not divisible by 4. -/
theorem limping_rook_adjacent_sum_not_divisible_by_4 :
  ∀ (board : Fin 10 → Fin 10 → ℕ), 
  (∀ (i j : Fin 10), 1 ≤ board i j ∧ board i j ≤ 100) →
  (∀ (i j : Fin 10), (∃ (i' : Fin 10), i = i' + 1 ∨ i = i' - 1)
                 ∨ (∃ (j' : Fin 10), j = j' + 1 ∨ j = j' - 1)) →
  ((∀ (i j : Fin 10) (k l : Fin 10),
      (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      (board i j + board k l) % 4 ≠ 0)) :=
by
  sorry

end limping_rook_adjacent_sum_not_divisible_by_4_l28_28959


namespace find_original_number_l28_28793

-- Definitions based on the conditions of the problem
def tens_digit (x : ℕ) := 2 * x
def original_number (x : ℕ) := 10 * (tens_digit x) + x
def reversed_number (x : ℕ) := 10 * x + (tens_digit x)

-- Proof statement
theorem find_original_number (x : ℕ) (h1 : original_number x - reversed_number x = 27) : original_number x = 63 := by
  sorry

end find_original_number_l28_28793


namespace strictly_increasing_interval_l28_28284

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem strictly_increasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = Real.exp 1) ∧ ∀ x : ℝ, a < x ∧ x < b → f' x > 0 :=
by
  let f' := λ x, (1 - Real.log x) / (x^2)
  sorry

end strictly_increasing_interval_l28_28284


namespace simplify_fraction_of_decimal_l28_28761

theorem simplify_fraction_of_decimal :
  let n        := 3675
  let d        := 1000
  let gcd      := Nat.gcd n d
  n / gcd = 147 ∧ d / gcd = 40 → 
  (3675 / 1000 : ℚ) = (147 / 40 : ℚ) :=
by {
  sorry
}

end simplify_fraction_of_decimal_l28_28761


namespace nitin_borrowed_amount_l28_28766

theorem nitin_borrowed_amount (P : ℝ) (interest_paid : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) 
  (h_rates1 : rate1 = 0.06) (h_rates2 : rate2 = 0.09) 
  (h_rates3 : rate3 = 0.13) (h_time1 : time1 = 3) 
  (h_time2 : time2 = 5) (h_time3 : time3 = 3)
  (h_interest : interest_paid = 8160) :
  P * (rate1 * time1 + rate2 * time2 + rate3 * time3) = interest_paid → 
  P = 8000 := 
by 
  sorry

end nitin_borrowed_amount_l28_28766


namespace nancy_crayons_l28_28869

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end nancy_crayons_l28_28869


namespace solve_equation_l28_28203

theorem solve_equation (a b : ℚ) : 
  ((b = 0) → false) ∧ 
  ((4 * a - 3 = 0) → ((5 * b - 1 = 0) → a = 3 / 4 ∧ b = 1 / 5)) ∧ 
  ((4 * a - 3 ≠ 0) → (∃ x : ℚ, x = (5 * b - 1) / (4 * a - 3))) :=
by
  sorry

end solve_equation_l28_28203


namespace smallest_four_digit_divisible_by_53_l28_28061

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28061


namespace probability_at_least_one_six_l28_28474

open ProbabilityTheory

/-- When two fair dice are rolled simultaneously, given that the numbers on both dice are different, 
the probability that at least one die shows a 6 is 1/3. -/
theorem probability_at_least_one_six (dice : Finset (ℕ × ℕ)) :
  (∀ x ∈ dice, x.1 ≠ x.2) →
  (6 ∈ Finset.range 7) → 
  prob ((λ x : ℕ × ℕ, x.1 ≠ x.2) ∧ (λ x : ℕ × ℕ, x.1 = 6 ∨ x.2 = 6)) dice = 1 / 3 :=
by
  -- convert question conditions and correct answer
  sorry

end probability_at_least_one_six_l28_28474


namespace gina_snake_mice_eaten_in_decade_l28_28679

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end gina_snake_mice_eaten_in_decade_l28_28679


namespace minimize_quadratic_l28_28467

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l28_28467


namespace smallest_four_digit_divisible_by_53_l28_28037

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28037


namespace minimize_f_l28_28445

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l28_28445


namespace smallest_four_digit_divisible_by_53_l28_28070

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28070


namespace total_cans_from_recycling_l28_28378

noncomputable def recycleCans (n : ℕ) : ℕ :=
  if n < 6 then 0 else n / 6 + recycleCans (n / 6 + n % 6)

theorem total_cans_from_recycling:
  recycleCans 486 = 96 :=
by
  sorry

end total_cans_from_recycling_l28_28378


namespace expected_profit_calculation_l28_28335

theorem expected_profit_calculation:
  let odd1 := 1.28
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let initial_bet := 5.00
  let total_payout := initial_bet * (odd1 * odd2 * odd3 * odd4)
  let expected_profit := total_payout - initial_bet
  expected_profit = 212.822 := by
  sorry

end expected_profit_calculation_l28_28335


namespace andrew_vacation_days_l28_28271

theorem andrew_vacation_days (days_worked last_year vacation_per_10 worked_days in_march in_september : ℕ)
  (h1 : vacation_per_10 = 10)
  (h2 : days_worked_last_year = 300)
  (h3 : worked_days = days_worked_last_year / vacation_per_10)
  (h4 : in_march = 5)
  (h5 : in_september = 2 * in_march)
  (h6 : days_taken = in_march + in_september)
  (h7 : vacation_days_remaining = worked_days - days_taken) :
  vacation_days_remaining = 15 :=
by
  sorry

end andrew_vacation_days_l28_28271


namespace shelter_cats_incoming_l28_28984

theorem shelter_cats_incoming (x : ℕ) (h : x + x / 2 - 3 + 5 - 1 = 19) : x = 12 :=
by
  sorry

end shelter_cats_incoming_l28_28984


namespace maximum_k_l28_28147

theorem maximum_k (m k : ℝ) (h0 : 0 < m) (h1 : m < 1/2) (h2 : (1/m + 2/(1-2*m)) ≥ k): k ≤ 8 :=
sorry

end maximum_k_l28_28147


namespace geom_seq_sum_4n_l28_28430

-- Assume we have a geometric sequence with positive terms and common ratio q
variables (a : ℕ → ℝ) (q : ℝ) (n : ℕ)

-- The sum of the first n terms of the geometric sequence is S_n
noncomputable def S_n : ℝ := a 0 * (1 - q^n) / (1 - q)

-- Given conditions
axiom h1 : S_n a q n = 2
axiom h2 : S_n a q (3 * n) = 14

-- We need to prove that S_{4n} = 30
theorem geom_seq_sum_4n : S_n a q (4 * n) = 30 :=
by
  sorry

end geom_seq_sum_4n_l28_28430


namespace max_value_of_expr_l28_28860

theorem max_value_of_expr (A M C : ℕ) (h : A + M + C = 12) : 
  A * M * C + A * M + M * C + C * A ≤ 112 :=
sorry

end max_value_of_expr_l28_28860


namespace correct_options_l28_28835

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 1

def option_2 : Prop := ∃ x : ℝ, f (x) = 0 ∧ x = Real.pi / 3
def option_3 : Prop := ∀ T > 0, (∀ x : ℝ, f (x) = f (x + T)) → T = Real.pi
def option_5 : Prop := ∀ x : ℝ, f (x - Real.pi / 6) = f (-(x - Real.pi / 6))

theorem correct_options :
  option_2 ∧ option_3 ∧ option_5 :=
by
  sorry

end correct_options_l28_28835


namespace points_on_line_relationship_l28_28690

theorem points_on_line_relationship :
  let m := 2 * Real.sqrt 2 + 1
  let n := 4
  m < n :=
by
  sorry

end points_on_line_relationship_l28_28690


namespace total_boxes_packed_l28_28778

section
variable (initial_boxes : ℕ) (cost_per_box : ℕ) (donation_multiplier : ℕ)
variable (donor_donation : ℕ) (additional_boxes : ℕ) (total_boxes : ℕ)

-- Given conditions
def initial_boxes := 400
def cost_per_box := 80 + 165  -- 245
def donation_multiplier := 4

def initial_expenditure : ℕ := initial_boxes * cost_per_box
def donor_donation : ℕ := initial_expenditure * donation_multiplier
def additional_boxes : ℕ := donor_donation / cost_per_box
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Proof statement
theorem total_boxes_packed : total_boxes = 2000 :=
by
  unfold initial_boxes cost_per_box donation_multiplier initial_expenditure donor_donation additional_boxes total_boxes
  simp
  sorry  -- Since the proof is not required
end

end total_boxes_packed_l28_28778


namespace min_product_value_max_product_value_l28_28201

open Real

noncomputable def min_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

noncomputable def max_cos_sin_product (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 then
    cos x * sin y * cos z
  else 0

theorem min_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ min_cos_sin_product x y z = 1 / 8 :=
sorry

theorem max_product_value :
  ∃ (x y z : ℝ), x ≥ y ∧ y ≥ z ∧ z ≥ π / 12 ∧ x + y + z = π / 2 ∧ max_cos_sin_product x y z = (2 + sqrt 3) / 8 :=
sorry

end min_product_value_max_product_value_l28_28201


namespace prime_power_seven_l28_28980

theorem prime_power_seven (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (eqn : p + 25 = q^7) : p = 103 := by
  sorry

end prime_power_seven_l28_28980


namespace minimize_quadratic_function_l28_28456

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l28_28456


namespace maximum_rabbits_condition_l28_28396

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l28_28396


namespace find_g_neg_six_l28_28923

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l28_28923


namespace smallest_four_digit_div_by_53_l28_28052

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28052


namespace height_of_triangle_on_parabola_l28_28496

open Real

theorem height_of_triangle_on_parabola
  (x0 x1 : ℝ)
  (y0 y1 : ℝ)
  (hA : y0 = x0^2)
  (hB : y0 = (-x0)^2)
  (hC : y1 = x1^2)
  (hypotenuse_parallel : y0 = y1 + 1):
  y0 - y1 = 1 := 
by
  sorry

end height_of_triangle_on_parabola_l28_28496


namespace evaluate_f_at_neg3_l28_28169

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem evaluate_f_at_neg3 : f (-3) = 110 :=
by 
  sorry

end evaluate_f_at_neg3_l28_28169


namespace factorize_problem1_factorize_problem2_l28_28811

-- Problem 1: Prove that 6p^3q - 10p^2 == 2p^2 * (3pq - 5)
theorem factorize_problem1 (p q : ℝ) : 
    6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := 
by 
    sorry

-- Problem 2: Prove that a^4 - 8a^2 + 16 == (a-2)^2 * (a+2)^2
theorem factorize_problem2 (a : ℝ) : 
    a^4 - 8 * a^2 + 16 = (a - 2)^2 * (a + 2)^2 := 
by 
    sorry

end factorize_problem1_factorize_problem2_l28_28811


namespace max_rabbits_l28_28414

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l28_28414


namespace necessary_but_not_sufficient_l28_28245

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x - y > -1) → (x^3 + x > x^2 * y + y) → 
  ∃ z : ℝ, z - y > -1 ∧ ¬ (z^3 + z > z^2 * y + y) :=
sorry

end necessary_but_not_sufficient_l28_28245


namespace swimming_pool_surface_area_l28_28981

def length : ℝ := 20
def width : ℝ := 15

theorem swimming_pool_surface_area : length * width = 300 := 
by
  -- The mathematical proof would go here; we'll skip it with "sorry" per instructions.
  sorry

end swimming_pool_surface_area_l28_28981


namespace smallest_four_digit_divisible_by_53_l28_28085

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28085


namespace prob_all_fail_prob_at_least_one_pass_l28_28830

variable (A B C : Prop)
variable [Independent A B] [Independent A C] [Independent B C]
variable (pA pB pC : ℝ)
variable (hA : pA = 1/2) (hB : pB = 1/2) (hC : pC = 1/2)

namespace ProbabilityProblem

-- Probability that all three fail
theorem prob_all_fail (h : P(A) = pA ∧ P(B) = pB ∧ P(C) = pC) : P(¬A ∧ ¬B ∧ ¬C) = 1 / 8 :=
by sorry

-- Probability that at least one person passes
theorem prob_at_least_one_pass (h : P(A) = pA ∧ P(B) = pB ∧ P(C) = pC) : 1 - P(¬A ∧ ¬B ∧ ¬C) = 7 / 8 :=
by sorry

end ProbabilityProblem

end prob_all_fail_prob_at_least_one_pass_l28_28830


namespace maximum_value_N_27_l28_28386

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l28_28386


namespace sum_of_favorite_numbers_is_600_l28_28565

def GloryFavoriteNumber : ℕ := 450
def MistyFavoriteNumber (G : ℕ) : ℕ := G / 3

theorem sum_of_favorite_numbers_is_600 (G : ℕ) (hG : G = GloryFavoriteNumber) :
  MistyFavoriteNumber G + G = 600 :=
by
  rw [hG]
  simp [GloryFavoriteNumber, MistyFavoriteNumber]
  -- Proof is omitted (filled with sorry)
  sorry

end sum_of_favorite_numbers_is_600_l28_28565


namespace find_slope_of_l_l28_28805

noncomputable def parabola (x y : ℝ) := y ^ 2 = 4 * x

-- Definition of the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Definition of the point M
def M : ℝ × ℝ := (-1, 2)

-- Check if two vectors are perpendicular
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Proof problem statement
theorem find_slope_of_l (x1 x2 y1 y2 k : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : is_perpendicular (x1 + 1, y1 - 2) (x2 + 1, y2 - 2))
  (eq1 : y1 = k * (x1 - 1))
  (eq2 : y2 = k * (x2 - 1)) :
  k = 1 := by
  sorry

end find_slope_of_l_l28_28805


namespace smallest_n_divisible_by_24_and_864_l28_28439

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ, (0 < n) ∧ (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ (∀ m : ℕ, (0 < m) → (24 ∣ m^2) → (864 ∣ m^3) → (n ≤ m)) :=
sorry

end smallest_n_divisible_by_24_and_864_l28_28439


namespace smallest_four_digit_divisible_by_53_l28_28088

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28088


namespace problem_I4_1_l28_28689

theorem problem_I4_1 
  (x y : ℝ)
  (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
sorry

end problem_I4_1_l28_28689


namespace find_k_solution_l28_28292

noncomputable def vec1 : ℝ × ℝ := (3, -4)
noncomputable def vec2 : ℝ × ℝ := (5, 8)
noncomputable def target_norm : ℝ := 3 * Real.sqrt 10

theorem find_k_solution : ∃ k : ℝ, 0 ≤ k ∧ ‖(k * vec1.1 - vec2.1, k * vec1.2 - vec2.2)‖ = target_norm ∧ k = 0.0288 :=
by
  sorry

end find_k_solution_l28_28292


namespace range_of_a_l28_28528

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x + y = 2 ∧ 
    (if x > 1 then (x^2 + 1) / x else Real.log (x + a)) = 
    (if y > 1 then (y^2 + 1) / y else Real.log (y + a))) ↔ 
    a > Real.exp 2 - 1 :=
by sorry

end range_of_a_l28_28528


namespace total_balls_l28_28279

theorem total_balls (blue red green yellow purple orange black white : ℕ) 
  (h1 : blue = 8)
  (h2 : red = 5)
  (h3 : green = 3 * (2 * blue - 1))
  (h4 : yellow = Nat.floor (2 * Real.sqrt (red * blue)))
  (h5 : purple = 4 * (blue + green))
  (h6 : orange = 7)
  (h7 : black + white = blue + red + green + yellow + purple + orange)
  (h8 : blue + red + green + yellow + purple + orange + black + white = 3 * (red + green + yellow + purple) + orange / 2)
  : blue + red + green + yellow + purple + orange + black + white = 829 :=
by
  sorry

end total_balls_l28_28279


namespace square_1023_l28_28994

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l28_28994


namespace find_g_minus_6_l28_28910

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l28_28910


namespace fruit_count_l28_28355

theorem fruit_count :
  let limes_mike : ℝ := 32.5
  let limes_alyssa : ℝ := 8.25
  let limes_jenny_picked : ℝ := 10.8
  let limes_jenny_ate := limes_jenny_picked / 2
  let limes_jenny := limes_jenny_picked - limes_jenny_ate
  let plums_tom : ℝ := 14.5
  let plums_tom_ate : ℝ := 2.5
  let X := (limes_mike - limes_alyssa) + limes_jenny
  let Y := plums_tom - plums_tom_ate
  X = 29.65 ∧ Y = 12 :=
by {
  sorry
}

end fruit_count_l28_28355


namespace orangeade_ratio_l28_28357

theorem orangeade_ratio (O W : ℝ) (price1 price2 : ℝ) (revenue1 revenue2 : ℝ)
  (h1 : price1 = 0.30) (h2 : price2 = 0.20)
  (h3 : revenue1 = revenue2)
  (glasses1 glasses2 : ℝ)
  (V : ℝ) :
  glasses1 = (O + W) / V → glasses2 = (O + 2 * W) / V →
  revenue1 = glasses1 * price1 → revenue2 = glasses2 * price2 →
  (O + W) * price1 = (O + 2 * W) * price2 → O / W = 1 :=
by sorry

end orangeade_ratio_l28_28357


namespace jack_emails_morning_l28_28852

-- Definitions from conditions
def emails_evening : ℕ := 7
def additional_emails_morning : ℕ := 2
def emails_morning : ℕ := emails_evening + additional_emails_morning

-- The proof problem
theorem jack_emails_morning : emails_morning = 9 := by
  -- proof goes here
  sorry

end jack_emails_morning_l28_28852


namespace abc_sum_16_l28_28843

theorem abc_sum_16 (a b c : ℕ) (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4) (h4 : a ≠ b ∨ b ≠ c ∨ a ≠ c)
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by
  sorry

end abc_sum_16_l28_28843


namespace find_g_neg_6_l28_28905

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l28_28905


namespace value_of_square_l28_28536

theorem value_of_square :
  ∃ (square : ℕ), 
  (Nat.toDigits 7 (5 * 343 + 3 * 49 + 2 * 7 + square) = [5, 3, 2, square]) ∧
  (Nat.toDigits 7 (square * 49 + 6 * 7) = [square, 6, 0]) ∧
  (Nat.toDigits 7 (square * 7 + 3) = [square, 3]) ∧
  (Nat.toDigits 7 (6 * 343 + 4 * 49 + square * 7 + 1) = [6, 4, square, 1])
:=
  ∃ (square: ℕ), square = 5
  sorry

end value_of_square_l28_28536


namespace harold_wrapping_cost_l28_28322

noncomputable def wrapping_paper_rolls_cost (cost_per_roll : ℕ) (rolls_needed : ℕ) : ℕ := cost_per_roll * rolls_needed

noncomputable def total_paper_rolls (shirt_boxes : ℕ) (shirt_boxes_per_roll : ℕ) (xl_boxes : ℕ) (xl_boxes_per_roll : ℕ) : ℕ :=
  (shirt_boxes / shirt_boxes_per_roll) + (xl_boxes / xl_boxes_per_roll)

theorem harold_wrapping_cost 
  (cost_per_roll : ℕ) (shirt_boxes : ℕ) (shirt_boxes_per_roll : ℕ) (xl_boxes : ℕ) (xl_boxes_per_roll : ℕ) :
  shirt_boxes = 20 → shirt_boxes_per_roll = 5 → xl_boxes = 12 → xl_boxes_per_roll = 3 → cost_per_roll = 4 → 
  wrapping_paper_rolls_cost cost_per_roll (total_paper_rolls shirt_boxes shirt_boxes_per_roll xl_boxes xl_boxes_per_roll) = 32 :=
by
  intros hshirt_boxes hshirt_boxes_per_roll hxl_boxes hxl_boxes_per_roll hcost_per_roll
  simp [wrapping_paper_rolls_cost, total_paper_rolls, hshirt_boxes, hshirt_boxes_per_roll, hxl_boxes, hxl_boxes_per_roll, hcost_per_roll]
  rfl

end harold_wrapping_cost_l28_28322


namespace total_area_of_removed_triangles_l28_28257

theorem total_area_of_removed_triangles (side_length : ℝ) (half_leg_length : ℝ) :
  side_length = 16 →
  half_leg_length = side_length / 4 →
  4 * (1 / 2) * half_leg_length^2 = 32 :=
by
  intro h_side_length h_half_leg_length
  simp [h_side_length, h_half_leg_length]
  sorry

end total_area_of_removed_triangles_l28_28257


namespace cookies_per_child_is_22_l28_28300

def total_cookies (num_packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  num_packages * cookies_per_package

def total_children (num_friends : ℕ) : ℕ :=
  num_friends + 1

def cookies_per_child (total_cookies : ℕ) (total_children : ℕ) : ℕ :=
  total_cookies / total_children

theorem cookies_per_child_is_22 :
  total_cookies 5 36 / total_children 7 = 22 := 
by
  sorry

end cookies_per_child_is_22_l28_28300


namespace simplify_fraction_l28_28366

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28366


namespace smallest_four_digit_divisible_by_53_l28_28091

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28091


namespace units_digit_34_pow_30_l28_28608

theorem units_digit_34_pow_30 :
  (34 ^ 30) % 10 = 6 :=
by
  sorry

end units_digit_34_pow_30_l28_28608


namespace inequality_solution_b_range_l28_28164

-- Given conditions
variables (a b : ℝ)

def condition1 : Prop := (1 - a < 0) ∧ (a = 3)
def condition2 : Prop := ∀ (x : ℝ), (3 * x^2 + b * x + 3) ≥ 0

-- Assertions to be proved
theorem inequality_solution (a : ℝ) (ha : condition1 a) : 
  ∀ (x : ℝ), (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

theorem b_range (a : ℝ) (hb : condition1 a) : 
  condition2 b ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end inequality_solution_b_range_l28_28164


namespace range_of_m_l28_28302

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  (A ⊆ B) ↔ (m ≤ (11:ℝ)/3) :=
by
  sorry

end range_of_m_l28_28302


namespace grid_problem_l28_28927

theorem grid_problem 
    (A B : ℕ)
    (H1 : 1 ≠ A)
    (H2 : 1 ≠ B)
    (H3 : 2 ≠ A)
    (H4 : 2 ≠ B)
    (H5 : 3 ≠ A)
    (H6 : 3 ≠ B)
    (H7 : A = 2)
    (H8 : B = 1)
    :
    A * B = 2 :=
by
  sorry

end grid_problem_l28_28927


namespace hours_practicing_l28_28170

theorem hours_practicing (W : ℕ) (hours_weekday : ℕ) 
  (h1 : hours_weekday = W + 17)
  (h2 : W + hours_weekday = 33) :
  W = 8 :=
sorry

end hours_practicing_l28_28170


namespace square_difference_l28_28656

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l28_28656


namespace circle_probability_l28_28333

noncomputable def problem_statement : Prop :=
  let outer_radius := 3
  let inner_radius := 1
  let pivotal_radius := 2
  let outer_area := Real.pi * outer_radius ^ 2
  let inner_area := Real.pi * pivotal_radius ^ 2
  let probability := inner_area / outer_area
  probability = 4 / 9

theorem circle_probability : problem_statement := sorry

end circle_probability_l28_28333


namespace linear_system_substitution_correct_l28_28819

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end linear_system_substitution_correct_l28_28819


namespace transformation_correctness_l28_28951

theorem transformation_correctness :
  (∀ x : ℝ, 3 * x = -4 → x = -4 / 3) ∧
  (∀ x : ℝ, 5 = 2 - x → x = -3) ∧
  (∀ x : ℝ, (x - 1) / 6 - (2 * x + 3) / 8 = 1 → 4 * (x - 1) - 3 * (2 * x + 3) = 24) ∧
  (∀ x : ℝ, 3 * x - (2 - 4 * x) = 5 → 3 * x + 4 * x - 2 = 5) :=
by
  -- Prove the given conditions
  sorry

end transformation_correctness_l28_28951


namespace angle_B_sum_a_c_l28_28191

theorem angle_B (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B) :
  B = π / 3 :=
  sorry

theorem sum_a_c (a b c : ℝ) (A B C : ℝ) (T : ℝ)
  (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π)
  (h5 : 0 < C) (h6 : C < π)
  (h7 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h8 : b = Real.sqrt 3)
  (h9 : T = 1/2 * a * c * Real.sin B)
  (hB : B = π / 3) :
  a + c = Real.sqrt 15 :=
  sorry

end angle_B_sum_a_c_l28_28191


namespace find_x_given_y_l28_28935

variable (x y : ℝ)

theorem find_x_given_y :
  (0 < x) → (0 < y) → 
  (∃ k : ℝ, (3 * x^2 * y = k)) → 
  (y = 18 → x = 3) → 
  (y = 2400) → 
  x = 9 * Real.sqrt 6 / 85 :=
by
  -- Proof goes here
  sorry

end find_x_given_y_l28_28935


namespace value_of_m_l28_28826

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - m * x + m - 1) 
  (h_eq : f 0 = f 2) : m = 2 :=
sorry

end value_of_m_l28_28826


namespace orthocenter_perpendicular_to_median_l28_28186

open EuclideanGeometry

variables {A B C I Q E K : Point}

-- Theorem: Prove that the line KQ is perpendicular to the line IE given the specified conditions.
theorem orthocenter_perpendicular_to_median
  (h_triangle : Triangle A B C)
  (h_incenter : Incenter I A B C)
  (h_inc_tg_AC : TangentPoint Q (incenter_circle I A B C) AC)
  (h_midpoint : Midpoint E A C)
  (h_orthocenter : Orthocenter K (Triangle B I C)) :
  Perpendicular (Line K Q) (Line I E) :=
sorry

end orthocenter_perpendicular_to_median_l28_28186


namespace smallest_four_digit_divisible_by_53_l28_28094

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28094


namespace strictly_decreasing_exponential_l28_28181

theorem strictly_decreasing_exponential (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → (1/2 < a ∧ a < 1) :=
by
  sorry

end strictly_decreasing_exponential_l28_28181


namespace smallest_four_digit_divisible_by_53_l28_28033

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28033


namespace units_digit_of_fraction_l28_28232

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end units_digit_of_fraction_l28_28232


namespace heptagon_triangle_count_l28_28676

noncomputable def number_of_triangles_in_heptagon : ℕ :=
  ∑ k in ({6, 5, 4, 3} : Finset ℕ), match k with
  | 6 => 1 * (Nat.choose 7 6)
  | 5 => 5 * (Nat.choose 7 5)
  | 4 => 4 * (Nat.choose 7 4)
  | 3 => 1 * (Nat.choose 7 3)
  | _ => 0

theorem heptagon_triangle_count : number_of_triangles_in_heptagon = 287 := by
  sorry

end heptagon_triangle_count_l28_28676


namespace total_cost_of_aquarium_l28_28651

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end total_cost_of_aquarium_l28_28651


namespace find_g_neg_6_l28_28900

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l28_28900


namespace hit_once_probability_l28_28120

theorem hit_once_probability (p: ℝ) (h: p = 0.5) : 
  let event_space : EventSpace := sorry,
  let hit_event : MeasurableSpace.event := sorry,
  let miss_event : MeasurableSpace.event := sorry,
  let hit_prob := event_space.prob hit_event,
  let miss_prob := event_space.prob miss_event in
  (hit_prob = p) →
  (miss_prob = 1 - p) →
  let exactly_one_hit := ((hit_event ∩ miss_event) ∪ (miss_event ∩ hit_event)) in
  event_space.prob exactly_one_hit = 0.5 :=
begin
  sorry
end

end hit_once_probability_l28_28120


namespace keys_missing_l28_28338

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l28_28338


namespace medial_triangle_AB_AC_BC_l28_28347

theorem medial_triangle_AB_AC_BC
  (l m n : ℝ)
  (A B C : Type)
  (midpoint_BC := (l, 0, 0))
  (midpoint_AC := (0, m, 0))
  (midpoint_AB := (0, 0, n)) :
  (AB^2 + AC^2 + BC^2) / (l^2 + m^2 + n^2) = 8 :=
by
  sorry

end medial_triangle_AB_AC_BC_l28_28347


namespace bowling_ball_volume_l28_28116

open Real

noncomputable def remaining_volume (d_bowling_ball d1 d2 d3 d4 h1 h2 h3 h4 : ℝ) : ℝ :=
  let r_bowling_ball := d_bowling_ball / 2
  let v_bowling_ball := (4/3) * π * (r_bowling_ball ^ 3)
  let v_hole1 := π * ((d1 / 2) ^ 2) * h1
  let v_hole2 := π * ((d2 / 2) ^ 2) * h2
  let v_hole3 := π * ((d3 / 2) ^ 2) * h3
  let v_hole4 := π * ((d4 / 2) ^ 2) * h4
  v_bowling_ball - (v_hole1 + v_hole2 + v_hole3 + v_hole4)

theorem bowling_ball_volume :
  remaining_volume 40 3 3 4 5 10 10 12 8 = 10523.67 * π :=
by
  sorry

end bowling_ball_volume_l28_28116


namespace af_over_cd_is_025_l28_28831

theorem af_over_cd_is_025
  (a b c d e f X : ℝ)
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end af_over_cd_is_025_l28_28831


namespace g_neg_six_eq_neg_twenty_l28_28917

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l28_28917


namespace coprime_exist_m_n_l28_28722

theorem coprime_exist_m_n (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_a : a ≥ 1) (h_b : b ≥ 1) :
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ a^m + b^n ≡ 1 [MOD a * b] :=
by
  use Nat.totient b, Nat.totient a
  sorry

end coprime_exist_m_n_l28_28722


namespace equal_playing_time_l28_28883

-- Given conditions
def total_minutes : Nat := 120
def number_of_children : Nat := 6
def children_playing_at_a_time : Nat := 2

-- Proof problem statement
theorem equal_playing_time :
  (children_playing_at_a_time * total_minutes) / number_of_children = 40 :=
by
  sorry

end equal_playing_time_l28_28883


namespace remainder_of_P_div_by_D_is_333_l28_28949

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 8 * x^4 - 18 * x^3 + 27 * x^2 - 14 * x - 30

-- Define the divisor D(x) and simplify it, but this is not necessary for the theorem statement.
-- def D (x : ℝ) : ℝ := 4 * x - 12  

-- Prove the remainder is 333 when x = 3
theorem remainder_of_P_div_by_D_is_333 : P 3 = 333 := by
  sorry

end remainder_of_P_div_by_D_is_333_l28_28949


namespace remainder_n_plus_2023_l28_28613

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 :=
by sorry

end remainder_n_plus_2023_l28_28613


namespace no_integer_roots_quadratic_l28_28698

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end no_integer_roots_quadratic_l28_28698


namespace exp_decreasing_iff_a_in_interval_l28_28385

theorem exp_decreasing_iff_a_in_interval (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 - a)^x > (2 - a)^y) ↔ 1 < a ∧ a < 2 :=
by 
  sorry

end exp_decreasing_iff_a_in_interval_l28_28385


namespace number_of_possible_triangles_with_side_5_not_shortest_l28_28152

-- Define and prove the number of possible triangles (a, b, c) with a, b, c positive integers,
-- such that one side is length 5 and it is not the shortest side is 10.
theorem number_of_possible_triangles_with_side_5_not_shortest (a b c : ℕ) (h1: a + b > c) (h2: a + c > b) (h3: b + c > a) 
(h4: 0 < a) (h5: 0 < b) (h6: 0 < c) (h7: a = 5 ∨ b = 5 ∨ c = 5) (h8: ¬ (a < 5 ∧ b < 5 ∧ c < 5)) :
∃ n, n = 10 := 
sorry

end number_of_possible_triangles_with_side_5_not_shortest_l28_28152


namespace original_percent_acid_l28_28614

open Real

variables (a w : ℝ)

theorem original_percent_acid 
  (h1 : (a + 2) / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 4) = 1 / 5) :
  a / (a + w) = 1 / 5 :=
sorry

end original_percent_acid_l28_28614


namespace maximum_value_N_27_l28_28390

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l28_28390


namespace intersection_point_l28_28548

/-- Coordinates of points A, B, C, and D -/
def pointA : Fin 3 → ℝ := ![3, -2, 4]
def pointB : Fin 3 → ℝ := ![13, -12, 9]
def pointC : Fin 3 → ℝ := ![1, 6, -8]
def pointD : Fin 3 → ℝ := ![3, -1, 2]

/-- Prove the intersection point of the lines AB and CD is (-7, 8, -1) -/
theorem intersection_point :
  let lineAB (t : ℝ) := pointA + t • (pointB - pointA)
  let lineCD (s : ℝ) := pointC + s • (pointD - pointC)
  ∃ t s : ℝ, lineAB t = lineCD s ∧ lineAB t = ![-7, 8, -1] :=
sorry

end intersection_point_l28_28548


namespace polygon_area_is_14_l28_28786

def vertices : List (ℕ × ℕ) :=
  [(1, 2), (2, 2), (3, 3), (3, 4), (4, 5), (5, 5), (6, 5), (6, 4), (5, 3),
   (4, 3), (4, 2), (3, 1), (2, 1), (1, 1)]

noncomputable def area_of_polygon (vs : List (ℕ × ℕ)) : ℝ := sorry

theorem polygon_area_is_14 :
  area_of_polygon vertices = 14 := sorry

end polygon_area_is_14_l28_28786


namespace smallest_four_digit_multiple_of_53_l28_28008

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28008


namespace power_equality_l28_28610

theorem power_equality : 
  ( (11 : ℝ) ^ (1 / 5) / (11 : ℝ) ^ (1 / 7) ) = (11 : ℝ) ^ (2 / 35) := 
by sorry

end power_equality_l28_28610


namespace driving_time_in_fog_is_correct_l28_28574

-- Define constants for speeds (in miles per minute)
def speed_sunny : ℚ := 35 / 60
def speed_rain : ℚ := 25 / 60
def speed_fog : ℚ := 15 / 60

-- Total distance and time
def total_distance : ℚ := 19.5
def total_time : ℚ := 45

-- Time variables for rain and fog
variables (t_r t_f : ℚ)

-- Define the driving distance equation
def distance_eq : Prop :=
  speed_sunny * (total_time - t_r - t_f) + speed_rain * t_r + speed_fog * t_f = total_distance

-- Prove the time driven in fog equals 10.25 minutes
theorem driving_time_in_fog_is_correct (h : distance_eq t_r t_f) : t_f = 10.25 :=
sorry

end driving_time_in_fog_is_correct_l28_28574


namespace cosine_range_l28_28806

theorem cosine_range {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.cos x ≤ 1 / 2) : 
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end cosine_range_l28_28806


namespace minimize_f_l28_28444

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l28_28444


namespace total_missing_keys_l28_28341

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l28_28341


namespace smallest_four_digit_divisible_by_53_l28_28029

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28029


namespace total_boxes_packed_l28_28776

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l28_28776


namespace fractions_not_equal_to_seven_over_five_l28_28239

theorem fractions_not_equal_to_seven_over_five :
  (7 / 5 ≠ 1 + (4 / 20)) ∧ (7 / 5 ≠ 1 + (3 / 15)) ∧ (7 / 5 ≠ 1 + (2 / 6)) :=
by
  sorry

end fractions_not_equal_to_seven_over_five_l28_28239


namespace tomatoes_picked_yesterday_l28_28635

/-
Given:
1. The farmer initially had 171 tomatoes.
2. The farmer picked some tomatoes yesterday (Y).
3. The farmer picked 30 tomatoes today.
4. The farmer will have 7 tomatoes left after today.

Prove:
The number of tomatoes the farmer picked yesterday (Y) is 134.
-/

theorem tomatoes_picked_yesterday (Y : ℕ) (h : 171 - Y - 30 = 7) : Y = 134 :=
sorry

end tomatoes_picked_yesterday_l28_28635


namespace tractor_trailer_weight_after_deliveries_l28_28262

def initial_weight := 50000
def first_store_unload_percent := 0.10
def second_store_unload_percent := 0.20

theorem tractor_trailer_weight_after_deliveries: 
  let weight_after_first_store := initial_weight - (first_store_unload_percent * initial_weight)
  let weight_after_second_store := weight_after_first_store - (second_store_unload_percent * weight_after_first_store)
  weight_after_second_store = 36000 :=
by
  sorry

end tractor_trailer_weight_after_deliveries_l28_28262


namespace odd_power_preserves_order_l28_28310

theorem odd_power_preserves_order {n : ℤ} (h1 : n > 0) (h2 : n % 2 = 1) :
  ∀ (a b : ℝ), a > b → a^n > b^n :=
by
  sorry

end odd_power_preserves_order_l28_28310


namespace smallest_four_digit_divisible_by_53_l28_28036

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28036


namespace ms_cole_total_students_l28_28568

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end ms_cole_total_students_l28_28568


namespace gina_snake_mice_eaten_in_decade_l28_28678

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end gina_snake_mice_eaten_in_decade_l28_28678


namespace probability_no_adjacent_green_hats_l28_28966

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l28_28966


namespace outfits_without_matching_color_l28_28106

theorem outfits_without_matching_color (red_shirts green_shirts pairs_pants green_hats red_hats : ℕ) 
  (h_red_shirts : red_shirts = 5) 
  (h_green_shirts : green_shirts = 5) 
  (h_pairs_pants : pairs_pants = 6) 
  (h_green_hats : green_hats = 8) 
  (h_red_hats : red_hats = 8) : 
  (red_shirts * pairs_pants * green_hats) + (green_shirts * pairs_pants * red_hats) = 480 := 
by 
  sorry

end outfits_without_matching_color_l28_28106


namespace cost_price_of_cupboard_l28_28480

theorem cost_price_of_cupboard (C S S_profit : ℝ) (h1 : S = 0.88 * C) (h2 : S_profit = 1.12 * C) (h3 : S_profit - S = 1650) :
  C = 6875 := by
  sorry

end cost_price_of_cupboard_l28_28480


namespace roots_are_reciprocals_eq_a_minus_one_l28_28542

theorem roots_are_reciprocals_eq_a_minus_one (a : ℝ) :
  (∀ x y : ℝ, x + y = -(a - 1) ∧ x * y = a^2 → x * y = 1) → a = -1 :=
by
  intro h
  sorry

end roots_are_reciprocals_eq_a_minus_one_l28_28542


namespace total_distance_of_trip_l28_28554

theorem total_distance_of_trip (x : ℚ)
  (highway : x / 4 ≤ x)
  (city : 30 ≤ x)
  (country : x / 6 ≤ x)
  (middle_part_fraction : 1 - 1 / 4 - 1 / 6 = 7 / 12) :
  (7 / 12) * x = 30 → x = 360 / 7 :=
by
  sorry

end total_distance_of_trip_l28_28554


namespace missing_keys_total_l28_28343

-- Definitions for the problem conditions

def num_consonants : ℕ := 21
def num_vowels : ℕ := 5
def missing_consonants_fraction : ℚ := 1 / 7
def missing_vowels : ℕ := 2

-- Statement to prove the total number of missing keys

theorem missing_keys_total :
  let missing_consonants := num_consonants * missing_consonants_fraction in
  let total_missing_keys := missing_consonants + missing_vowels in
  total_missing_keys = 5 :=
by {
  -- Placeholder proof
  sorry
}

end missing_keys_total_l28_28343


namespace number_of_chords_l28_28211

/-- Ten points are marked on the circumference of a circle.
    Prove that the number of different chords that can be drawn
    by connecting any two of these ten points is 45.
-/
theorem number_of_chords (n : ℕ) (h_n : n = 10) : 
  (nat.choose n 2) = 45 :=
by
  rw h_n
  norm_num

end number_of_chords_l28_28211


namespace sequence_formula_l28_28305

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 1) (h_recurrence : ∀ n : ℕ, 2 * n * a n + 1 = (n + 1) * a n) :
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
sorry

end sequence_formula_l28_28305


namespace partial_fraction_sum_l28_28931

theorem partial_fraction_sum :
  ∃ P Q R : ℚ, 
    P * ((-1 : ℚ) * (-2 : ℚ)) + Q * ((-3 : ℚ) * (-2 : ℚ)) + R * ((-3 : ℚ) * (1 : ℚ))
    = 14 ∧ 
    R * (1 : ℚ) * (3 : ℚ) + Q * ((-4 : ℚ) * (-3 : ℚ)) + P * ((3 : ℚ) * (1 : ℚ)) 
      = 12 ∧ 
    P + Q + R = 115 / 30 := by
  sorry

end partial_fraction_sum_l28_28931


namespace units_digit_fraction_l28_28234

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end units_digit_fraction_l28_28234


namespace linear_substitution_correct_l28_28817

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end linear_substitution_correct_l28_28817


namespace greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l28_28850

-- Define the given conditions
def totalOranges : ℕ := 81
def totalCookies : ℕ := 65
def numberOfChildren : ℕ := 7

-- Define the floor division for children
def orangesPerChild : ℕ := totalOranges / numberOfChildren
def cookiesPerChild : ℕ := totalCookies / numberOfChildren

-- Calculate leftover (donated) quantities
def orangesLeftover : ℕ := totalOranges % numberOfChildren
def cookiesLeftover : ℕ := totalCookies % numberOfChildren

-- Statements to prove
theorem greatest_number_of_donated_oranges : orangesLeftover = 4 := by {
    sorry
}

theorem greatest_number_of_donated_cookies : cookiesLeftover = 2 := by {
    sorry
}

end greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l28_28850


namespace square_of_1023_l28_28999

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l28_28999


namespace larger_angle_measure_l28_28605

-- Defining all conditions
def is_complementary (a b : ℝ) : Prop := a + b = 90

def angle_ratio (a b : ℝ) : Prop := a / b = 5 / 4

-- Main proof statement
theorem larger_angle_measure (a b : ℝ) (h1 : is_complementary a b) (h2 : angle_ratio a b) : a = 50 :=
by
  sorry

end larger_angle_measure_l28_28605


namespace quotient_is_zero_l28_28730

def square_mod_16 (n : ℕ) : ℕ :=
  (n * n) % 16

def distinct_remainders_in_range : List ℕ :=
  List.eraseDup $
    List.map square_mod_16 (List.range' 1 15)

def sum_of_distinct_remainders : ℕ :=
  distinct_remainders_in_range.sum

theorem quotient_is_zero :
  (sum_of_distinct_remainders / 16) = 0 :=
by
  sorry

end quotient_is_zero_l28_28730


namespace largest_possible_s_l28_28345

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) (h3 : (r - 2) * 180 * s = (s - 2) * 180 * r * 61 / 60) : s = 118 :=
sorry

end largest_possible_s_l28_28345


namespace initial_number_of_men_l28_28603

theorem initial_number_of_men (M : ℕ) (h1 : ∃ food : ℕ, food = M * 22) (h2 : ∀ food, food = (M * 20)) (h3 : ∃ food : ℕ, food = ((M + 40) * 19)) : M = 760 := by
  sorry

end initial_number_of_men_l28_28603


namespace average_weight_of_children_l28_28219

theorem average_weight_of_children 
  (average_weight_boys : ℝ)
  (number_of_boys : ℕ)
  (average_weight_girls : ℝ)
  (number_of_girls : ℕ)
  (total_children : ℕ)
  (average_weight_children : ℝ) :
  average_weight_boys = 160 →
  number_of_boys = 8 →
  average_weight_girls = 130 →
  number_of_girls = 6 →
  total_children = number_of_boys + number_of_girls →
  average_weight_children = 
    (number_of_boys * average_weight_boys + number_of_girls * average_weight_girls) / total_children →
  average_weight_children = 147 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_weight_of_children_l28_28219


namespace determine_k_linear_l28_28176

theorem determine_k_linear (k : ℝ) : |k| = 1 ∧ k + 1 ≠ 0 ↔ k = 1 := by
  sorry

end determine_k_linear_l28_28176


namespace increasing_interval_of_y_l28_28283

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x

theorem increasing_interval_of_y :
  ∃ (a b : ℝ), 0 < a ∧ a < e ∧ (∀ x : ℝ, a < x ∧ x < e → y x < y (x + ε)) :=
sorry

end increasing_interval_of_y_l28_28283


namespace correct_option_A_l28_28240

theorem correct_option_A : 
  (∀ a : ℝ, a^3 * a^4 = a^7) ∧ 
  ¬ (∀ a : ℝ, a^6 / a^2 = a^3) ∧ 
  ¬ (∀ a : ℝ, a^4 - a^2 = a^2) ∧ 
  ¬ (∀ a b : ℝ, (a - b)^2 = a^2 - b^2) :=
by
  /- omitted proofs -/
  sorry

end correct_option_A_l28_28240


namespace solve_for_x_l28_28307

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x (x : ℝ) : 2 * f x - 10 = f (x - 2) ↔ x = 3 :=
by
  sorry

end solve_for_x_l28_28307


namespace power_half_mod_prime_l28_28575

-- Definitions of odd prime and coprime condition
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1
def coprime (a p : ℕ) : Prop := Nat.gcd a p = 1

-- Main statement
theorem power_half_mod_prime (p a : ℕ) (hp : is_odd_prime p) (ha : coprime a p) :
  a ^ ((p - 1) / 2) % p = 1 ∨ a ^ ((p - 1) / 2) % p = p - 1 := 
  sorry

end power_half_mod_prime_l28_28575


namespace area_triangle_PCB_correct_l28_28741

noncomputable def area_of_triangle_PCB (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) : ℝ :=
  6

theorem area_triangle_PCB_correct (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) :
  area_APB = 4 ∧ area_CPD = 9 → area_of_triangle_PCB ABCD A B C D P AB_parallel_CD diagonals_intersect_P area_APB area_CPD = 6 :=
by
  sorry

end area_triangle_PCB_correct_l28_28741


namespace square_1023_l28_28993

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l28_28993


namespace two_to_the_n_plus_3_is_perfect_square_l28_28509

theorem two_to_the_n_plus_3_is_perfect_square (n : ℕ) (h : ∃ a : ℕ, 2^n + 3 = a^2) : n = 0 := 
sorry

end two_to_the_n_plus_3_is_perfect_square_l28_28509


namespace no_integer_roots_quadratic_l28_28697

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end no_integer_roots_quadratic_l28_28697


namespace room_length_l28_28422

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sq_meter : ℝ) (length : ℝ) : 
  width = 3.75 ∧ total_cost = 14437.5 ∧ cost_per_sq_meter = 700 → length = 5.5 :=
by
  sorry

end room_length_l28_28422


namespace opposite_of_neg_two_thirds_l28_28594

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l28_28594


namespace max_rabbits_l28_28415

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l28_28415


namespace total_votes_l28_28547

variable (V : ℝ)

theorem total_votes (h : 0.70 * V - 0.30 * V = 160) : V = 400 := by
  sorry

end total_votes_l28_28547


namespace max_rabbits_l28_28413

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l28_28413


namespace tangency_condition_for_parabola_and_line_l28_28297

theorem tangency_condition_for_parabola_and_line (k : ℚ) :
  (∀ x y : ℚ, (6 * x - 4 * y + k = 0) ↔ (y^2 = 16 * x)) ↔ (k = 32 / 3) :=
  sorry

end tangency_condition_for_parabola_and_line_l28_28297


namespace carlos_goals_product_l28_28713

theorem carlos_goals_product :
  ∃ (g11 g12 : ℕ), g11 < 8 ∧ g12 < 8 ∧ 
  (33 + g11) % 11 = 0 ∧ 
  (33 + g11 + g12) % 12 = 0 ∧ 
  g11 * g12 = 49 := 
by
  sorry

end carlos_goals_product_l28_28713


namespace smallest_four_digit_divisible_by_53_l28_28040

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28040


namespace part1_part2_l28_28836

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : 2 ≤ a ↔ ∀ (x : ℝ), f x a + g x ≥ 3 := by
  sorry

end part1_part2_l28_28836


namespace number_of_terms_is_10_l28_28799

noncomputable def arith_seq_number_of_terms (a : ℕ) (n : ℕ) (d : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((n-1)*d = 16) ∧ (n * (2*a + (n-2)*d) = 56) ∧ (n * (2*a + n*d) = 76)

theorem number_of_terms_is_10 (a d n : ℕ) (h : arith_seq_number_of_terms a n d) : n = 10 := by
  sorry

end number_of_terms_is_10_l28_28799


namespace minimize_f_at_3_l28_28447

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l28_28447


namespace evan_amount_l28_28298

def adrian : ℤ := sorry
def brenda : ℤ := sorry
def charlie : ℤ := sorry
def dana : ℤ := sorry
def evan : ℤ := sorry

def amounts_sum : Prop := adrian + brenda + charlie + dana + evan = 72
def abs_diff_1 : Prop := abs (adrian - brenda) = 21
def abs_diff_2 : Prop := abs (brenda - charlie) = 8
def abs_diff_3 : Prop := abs (charlie - dana) = 6
def abs_diff_4 : Prop := abs (dana - evan) = 5
def abs_diff_5 : Prop := abs (evan - adrian) = 14

theorem evan_amount
  (h_sum : amounts_sum)
  (h_diff1 : abs_diff_1)
  (h_diff2 : abs_diff_2)
  (h_diff3 : abs_diff_3)
  (h_diff4 : abs_diff_4)
  (h_diff5 : abs_diff_5) :
  evan = 21 := sorry

end evan_amount_l28_28298


namespace max_rabbits_with_traits_l28_28405

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l28_28405


namespace power_mean_inequality_l28_28349

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := 
by
  sorry

end power_mean_inequality_l28_28349


namespace fraction_simplify_l28_28601

theorem fraction_simplify : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end fraction_simplify_l28_28601


namespace grasshopper_max_reach_points_l28_28940

theorem grasshopper_max_reach_points
  (α : ℝ) (α_eq : α = 36 * Real.pi / 180)
  (L : ℕ)
  (jump_constant : ∀ (n : ℕ), L = L) :
  ∃ (N : ℕ), N ≤ 10 :=
by 
  sorry

end grasshopper_max_reach_points_l28_28940


namespace k_lt_zero_l28_28519

noncomputable def k_negative (k : ℝ) : Prop :=
  (∃ x : ℝ, x < 0 ∧ k * x > 0) ∧ (∃ x : ℝ, x > 0 ∧ k * x < 0)

theorem k_lt_zero (k : ℝ) : k_negative k → k < 0 :=
by
  intros h
  sorry

end k_lt_zero_l28_28519


namespace max_rabbits_l28_28419

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l28_28419


namespace tiles_with_no_gaps_l28_28789

-- Define the condition that the tiling consists of regular octagons
def regular_octagon_internal_angle := 135

-- Define the other regular polygons
def regular_triangle_internal_angle := 60
def regular_square_internal_angle := 90
def regular_pentagon_internal_angle := 108
def regular_hexagon_internal_angle := 120

-- The proposition to be proved: A flat surface without gaps
-- can be achieved using regular squares and regular octagons.
theorem tiles_with_no_gaps :
  ∃ (m n : ℕ), regular_octagon_internal_angle * m + regular_square_internal_angle * n = 360 :=
sorry

end tiles_with_no_gaps_l28_28789


namespace largest_on_edge_l28_28358

/-- On a grid, each cell contains a number which is the arithmetic mean of the four numbers around it 
    and all numbers are different. Prove that the largest number is located on the edge of the grid. -/
theorem largest_on_edge 
    (grid : ℕ → ℕ → ℝ) 
    (h_condition : ∀ (i j : ℕ), grid i j = (grid (i+1) j + grid (i-1) j + grid i (j+1) + grid i (j-1)) / 4)
    (h_unique : ∀ (i1 j1 i2 j2 : ℕ), (i1 ≠ i2 ∨ j1 ≠ j2) → grid i1 j1 ≠ grid i2 j2)
    : ∃ (i j : ℕ), (i = 0 ∨ j = 0 ∨ i = max_i ∨ j = max_j) ∧ ∀ (x y : ℕ), grid x y ≤ grid i j :=
sorry

end largest_on_edge_l28_28358


namespace sum_expression_l28_28179

theorem sum_expression (x k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) : x + y + z = (4 + 3 * k) * x :=
by
  sorry

end sum_expression_l28_28179


namespace simplify_fraction_l28_28368

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28368


namespace sum_midpoint_x_coords_l28_28934

theorem sum_midpoint_x_coords (a b c d : ℝ) (h : a + b + c + d = 20) :
  ((a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2) = 20 :=
by 
  calc
    ((a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2)
      = (a + b + b + c + c + d + d + a) / 2 : by sorry
    ... = (2 * (a + b + c + d)) / 2         : by sorry
    ... = a + b + c + d                     : by sorry
    ... = 20                                : by exact h

end sum_midpoint_x_coords_l28_28934


namespace box_volume_l28_28270

theorem box_volume (x y : ℝ) (hx : 0 < x ∧ x < 6) (hy : 0 < y ∧ y < 8) :
  (16 - 2 * x) * (12 - 2 * y) * y = 192 * y - 32 * y^2 - 24 * x * y + 4 * x * y^2 :=
by
  sorry

end box_volume_l28_28270


namespace pb_distance_l28_28969

theorem pb_distance (a b c d PA PD PC PB : ℝ)
  (hPA : PA = 5)
  (hPD : PD = 6)
  (hPC : PC = 7)
  (h1 : a^2 + b^2 = PA^2)
  (h2 : b^2 + c^2 = PC^2)
  (h3 : c^2 + d^2 = PD^2)
  (h4 : d^2 + a^2 = PB^2) :
  PB = Real.sqrt 38 := by
  sorry

end pb_distance_l28_28969


namespace giuseppe_can_cut_rectangles_l28_28301

theorem giuseppe_can_cut_rectangles : 
  let board_length := 22
  let board_width := 15
  let rectangle_length := 3
  let rectangle_width := 5
  (board_length * board_width) / (rectangle_length * rectangle_width) = 22 :=
by
  sorry

end giuseppe_can_cut_rectangles_l28_28301


namespace smallest_four_digit_divisible_by_53_l28_28099

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28099


namespace smallest_four_digit_divisible_by_53_l28_28079

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28079


namespace total_boxes_packed_l28_28777

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l28_28777


namespace minimize_quadratic_l28_28453

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l28_28453


namespace leftmost_three_digits_eq_317_l28_28522

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_arrangements (total_rings rings_selected fingers : ℕ) : ℕ :=
  binomial total_rings rings_selected * Nat.factorial rings_selected * binomial (rings_selected + fingers - 1) fingers

theorem leftmost_three_digits_eq_317 :
  let n := num_arrangements 10 6 5 in
  (n / 1000) % 1000 = 317 := by
  sorry

end leftmost_three_digits_eq_317_l28_28522


namespace difference_of_squares_153_147_l28_28665

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l28_28665


namespace sheets_per_pack_l28_28731

theorem sheets_per_pack (p d t : Nat) (total_sheets : Nat) (sheets_per_pack : Nat) 
  (h1 : p = 2) (h2 : d = 80) (h3 : t = 6) 
  (h4 : total_sheets = d * t)
  (h5 : sheets_per_pack = total_sheets / p) : sheets_per_pack = 240 := 
  by 
    sorry

end sheets_per_pack_l28_28731


namespace transformations_map_onto_itself_l28_28667

noncomputable def recurring_pattern_map_count (s : ℝ) : ℕ := sorry

theorem transformations_map_onto_itself (s : ℝ) :
  recurring_pattern_map_count s = 2 := sorry

end transformations_map_onto_itself_l28_28667


namespace partial_fraction_product_l28_28597

theorem partial_fraction_product : 
  ∃ A B C : ℚ, 
  (A = -21 / 4 ∧ B = 21 / 20 ∧ C = -16 / 5 ∧ 
   (A / (2 - 2) + B / (-2 + 2) + C / (3 - 2) = (23 - 102)/-(4 * 20))) ∧ 
   A * B * C = 1764 / 100 := by
{
  use [-21 / 4, 21 / 20, -16 / 5]
  any_goals {
    split
    all_goals {
      split
      any_goals exact rfl
      split
      any_goals exact rfl
      exact rfl
    }
    exact rfl
  }
  sorry     -- actual proofs will be provided here
}

end partial_fraction_product_l28_28597


namespace minimize_f_l28_28441

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l28_28441


namespace maximum_k_value_l28_28165

noncomputable def max_value_k (a : ℝ) (b : ℝ) (k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a^2 + b^2 ≥ k ∧ k = 1 / 2

theorem maximum_k_value (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ a + b = 1) → a^2 + b^2 ≥ 1 / 2 :=
by
  intro h
  obtain ⟨ha, hb, hab⟩ := h
  sorry

end maximum_k_value_l28_28165


namespace find_g_neg_6_l28_28904

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l28_28904


namespace servings_of_honey_l28_28483

theorem servings_of_honey :
  let total_ounces := 37 + 1/3
  let serving_size := 1 + 1/2
  total_ounces / serving_size = 24 + 8/9 :=
by
  sorry

end servings_of_honey_l28_28483


namespace no_integer_roots_l28_28695

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end no_integer_roots_l28_28695


namespace converse_proposition_false_l28_28586

theorem converse_proposition_false (a b c : ℝ) : ¬(∀ a b c : ℝ, (a > b) → (a * c^2 > b * c^2)) :=
by {
  -- proof goes here
  sorry
}

end converse_proposition_false_l28_28586


namespace solution_l28_28319

namespace Proof

open Set

def proof_problem : Prop :=
  let U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5, 6}
  A ∩ (U \ B) = {1, 2}

theorem solution : proof_problem := by
  -- The pre-defined proof_problem must be shown here
  -- Proof: sorry
  sorry

end Proof

end solution_l28_28319


namespace periodicity_of_m_arith_fibonacci_l28_28623

def m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) : Prop :=
∀ n : ℕ, v (n + 2) = (v n + v (n + 1)) % m

theorem periodicity_of_m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) 
  (hv : m_arith_fibonacci m v) : 
  ∃ r : ℕ, r ≤ m^2 ∧ ∀ n : ℕ, v (n + r) = v n := 
by
  sorry

end periodicity_of_m_arith_fibonacci_l28_28623


namespace betty_age_l28_28495

def ages (A M B : ℕ) : Prop :=
  A = 2 * M ∧ A = 4 * B ∧ M = A - 22

theorem betty_age (A M B : ℕ) : ages A M B → B = 11 :=
by
  sorry

end betty_age_l28_28495


namespace find_g_neg_6_l28_28903

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l28_28903


namespace rhombus_perimeter_52_l28_28733

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end rhombus_perimeter_52_l28_28733


namespace count_integers_between_sqrt5_and_sqrt50_l28_28531

theorem count_integers_between_sqrt5_and_sqrt50 
  (h1 : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3)
  (h2 : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8) : 
  ∃ n : ℕ, n = 5 := 
sorry

end count_integers_between_sqrt5_and_sqrt50_l28_28531


namespace train_arrival_problem_shooting_problem_l28_28796

-- Define trials and outcome types
inductive OutcomeTrain : Type
| onTime
| notOnTime

inductive OutcomeShooting : Type
| hitTarget
| missTarget

-- Scenario 1: Train Arrival Problem
def train_arrival_trials_refers_to (n : Nat) : Prop := 
  ∃ trials : List OutcomeTrain, trials.length = 3

-- Scenario 2: Shooting Problem
def shooting_trials_refers_to (n : Nat) : Prop :=
  ∃ trials : List OutcomeShooting, trials.length = 2

theorem train_arrival_problem : train_arrival_trials_refers_to 3 :=
by
  sorry

theorem shooting_problem : shooting_trials_refers_to 2 :=
by
  sorry

end train_arrival_problem_shooting_problem_l28_28796


namespace smallest_four_digit_divisible_by_53_l28_28072

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28072


namespace area_of_triangle_PAB_l28_28520

noncomputable theory

-- Definitions of geometric entities involved
def Circle (O : Type) [MetricSpace O] : O → ℝ := λ x, x ^ 2 + y ^ 2 = 4

def Line1 (O : Type) [MetricSpace O] : O → ℝ := λ x, y = x

def Line2 (O : Type) [MetricSpace O] : O → ℝ := λ x, y = sqrt 3 * x + 4

-- Tangency condition
def TangentPoint (O : Type) [MetricSpace O] : O := λ P, Line2 O P ∧ Circle O P

-- Distance of point P from Line1
def Distance (P : Point ℝ ℝ) : ℝ := (|sqrt 3 - 1|) / sqrt 2

-- Length of segment AB
def SegmentAB : ℝ := 4

-- Computation of area of triangle PAB
def AreaOfTriangle : ℝ := 1 / 2 * SegmentAB * Distance

-- Prove that the area of the triangle PAB is sqrt 6 + sqrt 2
theorem area_of_triangle_PAB 
  (A B P : ℝ) 
  (h1 : Circle A) 
  (h2 : Line1 A) 
  (h3 : Line2 P) 
  (h4 : SegmentAB = 4) 
  (h5 : TangentPoint P) 
  : AreaOfTriangle = sqrt 6 + sqrt 2
:= sorry

end area_of_triangle_PAB_l28_28520


namespace cost_per_ball_correct_l28_28324

-- Define the values given in the conditions
def total_amount_paid : ℝ := 4.62
def number_of_balls : ℝ := 3.0

-- Define the expected cost per ball according to the problem statement
def expected_cost_per_ball : ℝ := 1.54

-- Statement to prove that the cost per ball is as expected
theorem cost_per_ball_correct : (total_amount_paid / number_of_balls) = expected_cost_per_ball := 
sorry

end cost_per_ball_correct_l28_28324


namespace maximum_rabbits_condition_l28_28399

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l28_28399


namespace fraction_of_students_received_As_l28_28545

/-- Assume A is the fraction of students who received A's,
and B is the fraction of students who received B's,
and T is the total fraction of students who received either A's or B's. -/
theorem fraction_of_students_received_As
  (A B T : ℝ)
  (hB : B = 0.2)
  (hT : T = 0.9)
  (h : A + B = T) :
  A = 0.7 := 
by
  -- establishing the proof steps
  sorry

end fraction_of_students_received_As_l28_28545


namespace barbara_needs_more_weeks_l28_28505

/-
  Problem Statement:
  Barbara wants to save up for a new wristwatch that costs $100. Her parents give her an allowance
  of $5 a week and she can either save it all up or spend it as she wishes. 10 weeks pass and
  due to spending some of her money, Barbara currently only has $20. How many more weeks does she need
  to save for a watch if she stops spending on other things right now?
-/

def wristwatch_cost : ℕ := 100
def allowance_per_week : ℕ := 5
def current_savings : ℕ := 20
def amount_needed : ℕ := wristwatch_cost - current_savings
def weeks_needed : ℕ := amount_needed / allowance_per_week

theorem barbara_needs_more_weeks :
  weeks_needed = 16 :=
by
  -- proof goes here
  sorry

end barbara_needs_more_weeks_l28_28505


namespace households_with_car_l28_28546

theorem households_with_car {H_total H_neither H_both H_bike_only : ℕ} 
    (cond1 : H_total = 90)
    (cond2 : H_neither = 11)
    (cond3 : H_both = 22)
    (cond4 : H_bike_only = 35) : 
    H_total - H_neither - (H_bike_only + H_both - H_both) + H_both = 44 := by
  sorry

end households_with_car_l28_28546


namespace min_a_for_inequality_l28_28815

theorem min_a_for_inequality :
  (∀ (x : ℝ), |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1/3 :=
sorry

end min_a_for_inequality_l28_28815


namespace smallest_four_digit_divisible_by_53_l28_28076

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28076


namespace intersection_of_sets_l28_28861

-- Define sets A and B as given in the conditions
def A : Set ℝ := { x | -2 < x ∧ x < 2 }

def B : Set ℝ := {0, 1, 2}

-- Define the proposition to be proved
theorem intersection_of_sets : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_sets_l28_28861


namespace factor_x4_minus_81_l28_28136

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, (Polynomial.X ^ 4 - 81 : Polynomial ℝ) = (Polynomial.X - 3) * (Polynomial.X + 3) * (Polynomial.X ^ 2 + 9) := 
sorry

end factor_x4_minus_81_l28_28136


namespace solution_set_condition_l28_28429

-- The assumptions based on the given conditions
variables (a b : ℝ)

noncomputable def inequality_system_solution_set (x : ℝ) : Prop :=
  (x + 2 * a > 4) ∧ (2 * x - b < 5)

theorem solution_set_condition (a b : ℝ) :
  (∀ x : ℝ, inequality_system_solution_set a b x ↔ 0 < x ∧ x < 2) →
  (a + b) ^ 2023 = 1 :=
by
  intro h
  sorry

end solution_set_condition_l28_28429


namespace ratio_of_doctors_to_nurses_l28_28750

theorem ratio_of_doctors_to_nurses (total_staff doctors nurses : ℕ) (h1 : total_staff = 456) (h2 : nurses = 264) (h3 : doctors + nurses = total_staff) :
  doctors = 192 ∧ (doctors : ℚ) / nurses = 8 / 11 :=
by
  sorry

end ratio_of_doctors_to_nurses_l28_28750


namespace birth_year_1849_l28_28788

theorem birth_year_1849 (x : ℕ) (h1 : 1850 ≤ x^2 - 2 * x + 1) (h2 : x^2 - 2 * x + 1 < 1900) (h3 : x^2 - x + 1 = x) : x = 44 ↔ x^2 - 2 * x + 1 = 1849 := 
sorry

end birth_year_1849_l28_28788


namespace a2_eq_1_l28_28684

-- Define the geometric sequence and the conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a1_eq_2 : a 1 = 2
axiom condition1 : geometric_sequence a q
axiom condition2 : 16 * a 3 * a 5 = 8 * a 4 - 1

-- Prove that a_2 = 1
theorem a2_eq_1 : a 2 = 1 :=
by
  -- This is where the proof would go
  sorry

end a2_eq_1_l28_28684


namespace wind_velocity_l28_28744

theorem wind_velocity (P A V : ℝ) (k : ℝ := 1/200) :
  (P = k * A * V^2) →
  (P = 2) → (A = 1) → (V = 20) →
  ∀ (P' A' : ℝ), P' = 128 → A' = 4 → ∃ V' : ℝ, V'^2 = 6400 :=
by
  intros h1 h2 h3 h4 P' A' h5 h6
  use 80
  linarith

end wind_velocity_l28_28744


namespace four_digit_even_numbers_count_and_sum_l28_28559

variable (digits : Set ℕ) (used_once : ∀ d ∈ digits, d ≤ 6 ∧ d ≥ 1)

theorem four_digit_even_numbers_count_and_sum
  (hyp : digits = {1, 2, 3, 4, 5, 6}) :
  ∃ (N M : ℕ), 
    (N = 180 ∧ M = 680040) := 
sorry

end four_digit_even_numbers_count_and_sum_l28_28559


namespace smallest_sum_p_q_l28_28513

theorem smallest_sum_p_q (p q : ℕ) (h_pos : 1 < p) (h_cond : (p^2 * q - 1) = (2021 * p * q) / 2021) : p + q = 44 :=
sorry

end smallest_sum_p_q_l28_28513


namespace trapezoid_area_l28_28337

theorem trapezoid_area
  (A B C D : ℝ)
  (BC AD AC : ℝ)
  (radius circle_center : ℝ)
  (h : ℝ)
  (angleBAD angleADC : ℝ)
  (tangency : Bool) :
  BC = 13 → 
  angleBAD = 2 * angleADC →
  radius = 5 →
  tangency = true →
  1/2 * (BC + AD) * h = 157.5 :=
by
  sorry

end trapezoid_area_l28_28337


namespace evaluate_expression_l28_28289

theorem evaluate_expression : 
  let a := 7
  let b := 11
  let c := 13
  in 
  (
    (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) /
    (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b))
  ) = a + b + c :=
by
  sorry

end evaluate_expression_l28_28289


namespace gain_percent_calculation_l28_28108

def gain : ℝ := 0.70
def cost_price : ℝ := 70.0

theorem gain_percent_calculation : (gain / cost_price) * 100 = 1 := by
  sorry

end gain_percent_calculation_l28_28108


namespace geometric_sequence_l28_28383

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence
def geom_seq (a₁ r : α) (n : ℕ) : α := a₁ * r^(n-1)

theorem geometric_sequence :
  ∀ (a₁ : α), a₁ > 0 → geom_seq a₁ 2 3 * geom_seq a₁ 2 11 = 16 → geom_seq a₁ 2 5 = 1 :=
by
  intros a₁ h_pos h_eq
  sorry

end geometric_sequence_l28_28383


namespace g_neg_six_eq_neg_twenty_l28_28913

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l28_28913


namespace eve_age_l28_28263

variable (E : ℕ)

theorem eve_age (h1 : ∀ (a : ℕ), a = 9 → (E + 1) = 3 * (9 - 4)) : E = 14 := 
by
  have h2 : 9 - 4 = 5 := by norm_num
  have h3 : 3 * 5 = 15 := by norm_num
  have h4 : (E + 1) = 15 := h1 9 rfl
  linarith

end eve_age_l28_28263


namespace factor_polynomial_l28_28290

theorem factor_polynomial (y : ℝ) :
  y^8 - 4 * y^6 + 6 * y^4 - 4 * y^2 + 1 = ((y - 1) * (y + 1))^4 :=
sorry

end factor_polynomial_l28_28290


namespace lemonade_cups_count_l28_28332

theorem lemonade_cups_count :
  ∃ x y : ℕ, x + y = 400 ∧ x + 2 * y = 546 ∧ x = 254 :=
by
  sorry

end lemonade_cups_count_l28_28332


namespace smallest_four_digit_divisible_by_53_l28_28098

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28098


namespace average_monthly_bill_l28_28634

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end average_monthly_bill_l28_28634


namespace sum_smallest_largest_consecutive_even_integers_l28_28581

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l28_28581


namespace cricket_player_avg_runs_l28_28975

theorem cricket_player_avg_runs (A : ℝ) :
  (13 * A + 92 = 14 * (A + 5)) → A = 22 :=
by
  intro h1
  have h2 : 13 * A + 92 = 14 * A + 70 := by sorry
  have h3 : 92 - 70 = 14 * A - 13 * A := by sorry
  sorry

end cricket_player_avg_runs_l28_28975


namespace least_xy_l28_28156

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l28_28156


namespace jasmine_coffee_beans_purchase_l28_28555

theorem jasmine_coffee_beans_purchase (x : ℝ) (coffee_cost per_pound milk_cost per_gallon total_cost : ℝ)
  (h1 : coffee_cost = 2.50)
  (h2 : milk_cost = 3.50)
  (h3 : total_cost = 17)
  (h4 : milk_purchased = 2)
  (h_equation : coffee_cost * x + milk_cost * milk_purchased = total_cost) :
  x = 4 :=
by
  sorry

end jasmine_coffee_beans_purchase_l28_28555


namespace option_C_correct_l28_28616

theorem option_C_correct (x : ℝ) : x^3 * x^2 = x^5 := sorry

end option_C_correct_l28_28616


namespace find_other_number_l28_28224

theorem find_other_number (a b lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 61) (h_first_number : a = 210) :
  a * b = lcm * hcf → b = 671 :=
by 
  -- setup
  sorry

end find_other_number_l28_28224


namespace correct_ordering_l28_28787

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonicity (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 ≠ x2) : (x1 - x2) * (f x1 - f x2) > 0

theorem correct_ordering : f 1 < f (-2) ∧ f (-2) < f 3 :=
by sorry

end correct_ordering_l28_28787


namespace net_change_over_week_l28_28875

-- Definitions of initial quantities on Day 1
def baking_powder_day1 : ℝ := 4
def flour_day1 : ℝ := 12
def sugar_day1 : ℝ := 10
def chocolate_chips_day1 : ℝ := 6

-- Definitions of final quantities on Day 7
def baking_powder_day7 : ℝ := 2.5
def flour_day7 : ℝ := 7
def sugar_day7 : ℝ := 6.5
def chocolate_chips_day7 : ℝ := 3.7

-- Definitions of changes in quantities
def change_baking_powder : ℝ := baking_powder_day1 - baking_powder_day7
def change_flour : ℝ := flour_day1 - flour_day7
def change_sugar : ℝ := sugar_day1 - sugar_day7
def change_chocolate_chips : ℝ := chocolate_chips_day1 - chocolate_chips_day7

-- Statement to prove
theorem net_change_over_week : change_baking_powder + change_flour + change_sugar + change_chocolate_chips = 12.3 :=
by
  -- (Proof omitted)
  sorry

end net_change_over_week_l28_28875


namespace smallest_four_digit_multiple_of_53_l28_28011

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28011


namespace find_d_l28_28328

-- Conditions
variables (c d : ℝ)
axiom ratio_cond : c / d = 4
axiom eq_cond : c = 20 - 6 * d

theorem find_d : d = 2 :=
by
  sorry

end find_d_l28_28328


namespace find_g_neg6_l28_28897

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l28_28897


namespace smallest_four_digit_divisible_by_53_l28_28023

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28023


namespace coeff_x20_greater_in_Q_l28_28716

noncomputable def coeff (f : ℕ → ℕ → ℤ) (p x : ℤ) : ℤ :=
(x ^ 20) * p

noncomputable def P (x : ℤ) := (1 - x^2 + x^3) ^ 1000
noncomputable def Q (x : ℤ) := (1 + x^2 - x^3) ^ 1000

theorem coeff_x20_greater_in_Q :
  coeff 20 (Q x) x > coeff 20 (P x) x :=
  sorry

end coeff_x20_greater_in_Q_l28_28716


namespace find_g_minus_6_l28_28906

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l28_28906


namespace less_than_half_l28_28945

theorem less_than_half (a b c : ℝ) (h₁ : a = 43.2) (h₂ : b = 0.5) (h₃ : c = 42.7) : a - b = c := by
  sorry

end less_than_half_l28_28945


namespace find_min_value_x_l28_28199

theorem find_min_value_x (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 10) : 
  ∃ (x_min : ℝ), (∀ (x' : ℝ), (∀ y' z', x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10 → x' ≥ x_min)) ∧ x_min = 2 / 3 :=
sorry

end find_min_value_x_l28_28199


namespace number_of_male_animals_l28_28183

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end number_of_male_animals_l28_28183


namespace no_adjacent_green_hats_l28_28965

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l28_28965


namespace smallest_four_digit_multiple_of_53_l28_28010

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28010


namespace maximum_rabbits_l28_28393

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l28_28393


namespace solve_for_y_l28_28471

theorem solve_for_y (y : ℚ) : 
  y + 5 / 8 = 2 / 9 + 1 / 2 → 
  y = 7 / 72 := 
by 
  intro h1
  sorry

end solve_for_y_l28_28471


namespace find_fake_coin_in_two_weighings_l28_28874

theorem find_fake_coin_in_two_weighings (coins : Fin 8 → ℝ) (h : ∃ i : Fin 8, (∀ j ≠ i, coins i < coins j)) : 
  ∃! i : Fin 8, ∀ j ≠ i, coins i < coins j :=
by
  sorry

end find_fake_coin_in_two_weighings_l28_28874


namespace min_total_number_of_stamps_l28_28544

theorem min_total_number_of_stamps
  (r s t : ℕ)
  (h1 : 1 ≤ r)
  (h2 : 1 ≤ s)
  (h3 : 85 * r + 66 * s = 100 * t) :
  r + s = 7 := 
sorry

end min_total_number_of_stamps_l28_28544


namespace yi_catches_jia_on_DA_l28_28185

def square_side_length : ℝ := 90
def jia_speed : ℝ := 65
def yi_speed : ℝ := 72
def jia_start : ℝ := 0
def yi_start : ℝ := 90

theorem yi_catches_jia_on_DA :
  let square_perimeter := 4 * square_side_length
  let initial_gap := 3 * square_side_length
  let relative_speed := yi_speed - jia_speed
  let time_to_catch := initial_gap / relative_speed
  let distance_travelled_by_yi := yi_speed * time_to_catch
  let number_of_laps := distance_travelled_by_yi / square_perimeter
  let additional_distance := distance_travelled_by_yi % square_perimeter
  additional_distance = 0 →
  square_side_length * (number_of_laps % 4) = 0 ∨ number_of_laps % 4 = 3 :=
by
  -- We only provide the statement, the proof is omitted.
  sorry

end yi_catches_jia_on_DA_l28_28185


namespace compute_diff_squares_l28_28659

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l28_28659


namespace smallest_four_digit_divisible_by_53_l28_28035

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28035


namespace minimize_f_at_3_l28_28446

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l28_28446


namespace sixth_student_stickers_l28_28431

-- Define the given conditions.
def first_student_stickers := 29
def increment := 6

-- Define the number of stickers given to each subsequent student.
def stickers (n : ℕ) : ℕ :=
  first_student_stickers + n * increment

-- Theorem statement: the 6th student will receive 59 stickers.
theorem sixth_student_stickers : stickers 5 = 59 :=
by
  sorry

end sixth_student_stickers_l28_28431


namespace solution_is_thirteen_over_nine_l28_28286

noncomputable def check_solution (x : ℝ) : Prop :=
  (3 * x^2 / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0) ∧
  (x^3 ≠ 3 * x + 1)

theorem solution_is_thirteen_over_nine :
  check_solution (13 / 9) :=
by
  sorry

end solution_is_thirteen_over_nine_l28_28286


namespace find_science_books_l28_28265

theorem find_science_books
  (S : ℕ)
  (h1 : 2 * 3 + 3 * 2 + 3 * S = 30) :
  S = 6 :=
by
  sorry

end find_science_books_l28_28265


namespace unsold_books_l28_28974

-- Definitions from conditions
def books_total : ℕ := 150
def books_sold : ℕ := (2 / 3) * books_total
def book_price : ℕ := 5
def total_received : ℕ := 500

-- Proof statement
theorem unsold_books :
  (books_sold * book_price = total_received) →
  (books_total - books_sold = 50) :=
by
  sorry

end unsold_books_l28_28974


namespace Alan_eggs_count_l28_28264

theorem Alan_eggs_count (Price_per_egg Chickens_bought Price_per_chicken Total_spent : ℕ)
  (h1 : Price_per_egg = 2) (h2 : Chickens_bought = 6) (h3 : Price_per_chicken = 8) (h4 : Total_spent = 88) :
  ∃ E : ℕ, 2 * E + Chickens_bought * Price_per_chicken = Total_spent ∧ E = 20 :=
by
  sorry

end Alan_eggs_count_l28_28264


namespace ratio_of_x_and_y_l28_28327

theorem ratio_of_x_and_y (x y : ℤ) (h : (3 * x - 2 * y) * 4 = 3 * (2 * x + y)) : (x : ℚ) / y = 11 / 6 :=
  sorry

end ratio_of_x_and_y_l28_28327


namespace eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l28_28100

theorem eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256 :
  11^2 + 2 * 11 * 5 + 5^2 = 256 := by
  sorry

end eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l28_28100


namespace smallest_four_digit_divisible_by_53_l28_28075

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28075


namespace place_numbers_in_table_l28_28936

theorem place_numbers_in_table (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (table : Fin 10 → Fin 10 → ℝ),
    (∀ i j, table i j = nums ⟨10 * i + j, sorry⟩) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      |table i j - table k l| ≠ 1) := sorry  -- Proof omitted

end place_numbers_in_table_l28_28936


namespace chords_in_circle_l28_28215

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l28_28215


namespace four_sin_t_plus_cos_2t_bounds_l28_28572

theorem four_sin_t_plus_cos_2t_bounds (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end four_sin_t_plus_cos_2t_bounds_l28_28572


namespace minimize_quadratic_l28_28468

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l28_28468


namespace rhombus_perimeter_52_l28_28734

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end rhombus_perimeter_52_l28_28734


namespace smallest_four_digit_divisible_by_53_l28_28059

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28059


namespace smallest_number_divisible_by_6_in_permutations_list_l28_28353

def is_divisible_by_6 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 6 * k)

noncomputable def permutations_5_digits := 
  [1, 2, 3, 4, 5].permutations.map (λ l => l.foldl (λ acc x => 10 * acc + x) 0)

theorem smallest_number_divisible_by_6_in_permutations_list :
  ∃ n ∈ permutations_5_digits, is_divisible_by_6 n ∧ (∀ m ∈ permutations_5_digits, is_divisible_by_6 m → n ≤ m) :=
sorry

end smallest_number_divisible_by_6_in_permutations_list_l28_28353


namespace number_of_people_in_first_group_l28_28535

variable (W : ℝ)  -- Amount of work
variable (P : ℝ)  -- Number of people in the first group

-- Condition 1: P people can do 3W work in 3 days
def condition1 : Prop := P * (W / 1) * 3 = 3 * W

-- Condition 2: 5 people can do 5W work in 3 days
def condition2 : Prop := 5 * (W / 1) * 3 = 5 * W

-- Theorem to prove: The number of people in the first group is 3
theorem number_of_people_in_first_group (h1 : condition1 W P) (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l28_28535


namespace smallest_four_digit_multiple_of_53_l28_28000

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28000


namespace total_boxes_correct_l28_28774

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l28_28774


namespace max_rabbits_l28_28418

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l28_28418


namespace probability_intersection_inside_nonagon_correct_l28_28624

def nonagon_vertices : ℕ := 9

def total_pairs_of_points := Nat.choose nonagon_vertices 2

def sides_of_nonagon : ℕ := nonagon_vertices

def diagonals_of_nonagon := total_pairs_of_points - sides_of_nonagon

def pairs_of_diagonals := Nat.choose diagonals_of_nonagon 2

def sets_of_intersecting_diagonals := Nat.choose nonagon_vertices 4

noncomputable def probability_intersection_inside_nonagon : ℚ :=
  sets_of_intersecting_diagonals / pairs_of_diagonals

theorem probability_intersection_inside_nonagon_correct :
  probability_intersection_inside_nonagon = 14 / 39 := 
  sorry

end probability_intersection_inside_nonagon_correct_l28_28624


namespace smallest_four_digit_divisible_by_53_l28_28065

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28065


namespace eggs_in_each_basket_l28_28436

theorem eggs_in_each_basket (n : ℕ) (h₁ : 5 ≤ n) (h₂ : n ∣ 30) (h₃ : n ∣ 42) : n = 6 :=
sorry

end eggs_in_each_basket_l28_28436


namespace simplify_expression_correct_l28_28362

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l28_28362


namespace count_solid_circles_among_first_2006_l28_28260

-- Definition of the sequence sum for location calculation
def sequence_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2 - 1

-- Main theorem
theorem count_solid_circles_among_first_2006 : 
  ∃ n : ℕ, sequence_sum (n - 1) < 2006 ∧ 2006 ≤ sequence_sum n ∧ n = 62 :=
by {
  sorry
}

end count_solid_circles_among_first_2006_l28_28260


namespace regular_polygon_sides_l28_28490

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 12) : n = 30 := 
by
  sorry

end regular_polygon_sides_l28_28490


namespace min_value_at_3_l28_28464

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l28_28464


namespace rhombus_perimeter_l28_28738

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h3 : d1 / 2 ≠ 0) (h4 : d2 / 2 ≠ 0) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  4 * s = 52 :=
by
  sorry

end rhombus_perimeter_l28_28738


namespace cube_volume_surface_area_l28_28440

theorem cube_volume_surface_area (x : ℝ) (s : ℝ)
  (h1 : s^3 = 3 * x)
  (h2 : 6 * s^2 = 6 * x) :
  x = 3 :=
by sorry

end cube_volume_surface_area_l28_28440


namespace fraction_difference_l28_28178

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : (1 / a - 1 / b) = -2 := 
by
  sorry

end fraction_difference_l28_28178


namespace solve_symmetric_cosine_phi_l28_28312

noncomputable def symmetric_cosine_phi : Prop :=
  ∃ (φ : ℝ), (φ ∈ set.Icc 0 real.pi) ∧ (∀ (x : ℝ), 3 * real.cos (x + φ) - 1 = 3 * real.cos (2 * real.pi / 3 - x + φ) - 1) ∧ φ = 2 * real.pi / 3

theorem solve_symmetric_cosine_phi : symmetric_cosine_phi :=
  sorry

end solve_symmetric_cosine_phi_l28_28312


namespace num_real_roots_of_eq_l28_28929

theorem num_real_roots_of_eq (x : ℝ) (h : x * |x| - 3 * |x| - 4 = 0) : 
  ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 :=
sorry

end num_real_roots_of_eq_l28_28929


namespace verify_p_q_l28_28864

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-5, 2]]

def p : ℤ := 5
def q : ℤ := -26

theorem verify_p_q :
  N * N = p • N + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  -- Skipping the proof
  sorry

end verify_p_q_l28_28864


namespace approximate_pi_value_l28_28584

theorem approximate_pi_value (r h : ℝ) (L : ℝ) (V : ℝ) (π : ℝ) 
  (hL : L = 2 * π * r)
  (hV : V = 1 / 3 * π * r^2 * h) 
  (approxV : V = 2 / 75 * L^2 * h) :
  π = 25 / 8 := 
by
  -- Proof goes here
  sorry

end approximate_pi_value_l28_28584


namespace initial_investment_proof_l28_28258

-- Definitions for the conditions
def initial_investment_A : ℝ := sorry
def contribution_B : ℝ := 15750
def profit_ratio_A : ℝ := 2
def profit_ratio_B : ℝ := 3
def time_A : ℝ := 12
def time_B : ℝ := 4

-- Lean statement to prove
theorem initial_investment_proof : initial_investment_A * time_A * profit_ratio_B = contribution_B * time_B * profit_ratio_A → initial_investment_A = 1750 :=
by
  sorry

end initial_investment_proof_l28_28258


namespace smallest_four_digit_divisible_by_53_l28_28044

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28044


namespace watch_cost_price_l28_28493

theorem watch_cost_price 
  (C : ℝ)
  (h1 : 0.9 * C + 180 = 1.05 * C) :
  C = 1200 :=
sorry

end watch_cost_price_l28_28493


namespace initial_girls_count_l28_28637

variable (p : ℕ) -- total number of people initially in the group
variable (initial_girls : ℕ) -- number of girls initially

-- Condition 1: Initially, 50% of the group are girls
def initially_fifty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop := initial_girls = p / 2

-- Condition 2: Three girls leave and three boys arrive
def after_girls_leave_and_boys_arrive (initial_girls : ℕ) : ℕ := initial_girls - 3

-- Condition 3: After the change, 40% of the group are girls
def after_the_change_forty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop :=
  (after_girls_leave_and_boys_arrive initial_girls) = 2 * (p / 5)

theorem initial_girls_count (p : ℕ) (initial_girls : ℕ) :
  initially_fifty_percent_girls p initial_girls →
  after_the_change_forty_percent_girls p initial_girls →
  initial_girls = 15 := by
  sorry

end initial_girls_count_l28_28637


namespace percentage_less_than_l28_28329

theorem percentage_less_than (x y : ℝ) (h : y = 1.80 * x) : (x / y) * 100 = 100 - 44.44 :=
by
  sorry

end percentage_less_than_l28_28329


namespace new_person_weight_l28_28110

-- Define the given conditions as Lean definitions
def weight_increase_per_person : ℝ := 2.5
def num_people : ℕ := 8
def replaced_person_weight : ℝ := 65

-- State the theorem using the given conditions and the correct answer
theorem new_person_weight :
  (weight_increase_per_person * num_people) + replaced_person_weight = 85 :=
sorry

end new_person_weight_l28_28110


namespace simplify_fraction_l28_28375

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28375


namespace difference_of_squares_153_147_l28_28666

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l28_28666


namespace smallest_four_digit_div_by_53_l28_28046

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28046


namespace rhombus_area_eq_54_l28_28254

theorem rhombus_area_eq_54
  (a b : ℝ) (eq_long_side : a = 4 * Real.sqrt 3) (eq_short_side : b = 3 * Real.sqrt 3)
  (rhombus_diagonal1 : ℝ := 9 * Real.sqrt 3) (rhombus_diagonal2 : ℝ := 4 * Real.sqrt 3) :
  (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2 = 54 := by
  sorry

end rhombus_area_eq_54_l28_28254


namespace percentage_increase_is_200_l28_28925

noncomputable def total_cost : ℝ := 300
noncomputable def rate_per_sq_m : ℝ := 5
noncomputable def length : ℝ := 13.416407864998739
noncomputable def area : ℝ := total_cost / rate_per_sq_m
noncomputable def breadth : ℝ := area / length
noncomputable def percentage_increase : ℝ := (length - breadth) / breadth * 100

theorem percentage_increase_is_200 :
  percentage_increase = 200 :=
by
  sorry

end percentage_increase_is_200_l28_28925


namespace sum_of_digits_Joey_age_twice_Max_next_l28_28858

noncomputable def Joey_is_two_years_older (C : ℕ) : ℕ := C + 2

noncomputable def Max_age_today := 2

noncomputable def Eight_multiples_of_Max (C : ℕ) := 
  ∃ n : ℕ, C = 24 + n

noncomputable def Next_Joey_age_twice_Max (C J M n : ℕ): Prop := J + n = 2 * (M + n)

theorem sum_of_digits_Joey_age_twice_Max_next (C J M n : ℕ) 
  (h1: J = Joey_is_two_years_older C)
  (h2: M = Max_age_today)
  (h3: Eight_multiples_of_Max C)
  (h4: Next_Joey_age_twice_Max C J M n) 
  : ∃ s, s = 7 :=
sorry

end sum_of_digits_Joey_age_twice_Max_next_l28_28858


namespace average_monthly_bill_l28_28633

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end average_monthly_bill_l28_28633


namespace regular_polygon_with_12_degree_exterior_angle_has_30_sides_l28_28488

def regular_polygon_sides (e : ℤ) : ℤ :=
  360 / e

theorem regular_polygon_with_12_degree_exterior_angle_has_30_sides :
  regular_polygon_sides 12 = 30 :=
by
  -- Proof is omitted
  sorry

end regular_polygon_with_12_degree_exterior_angle_has_30_sides_l28_28488


namespace students_play_neither_l28_28955

-- Defining the problem parameters
def total_students : ℕ := 36
def football_players : ℕ := 26
def tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Statement to be proved
theorem students_play_neither : (total_students - (football_players + tennis_players - both_players)) = 7 :=
by show total_students - (football_players + tennis_players - both_players) = 7; sorry

end students_play_neither_l28_28955


namespace difference_of_squares_153_147_l28_28664

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l28_28664


namespace bus_problem_l28_28228

theorem bus_problem (x : ℕ) : 50 * x + 10 = 52 * x + 2 := 
sorry

end bus_problem_l28_28228


namespace cubic_common_roots_l28_28140

theorem cubic_common_roots:
  ∃ (c d : ℝ), 
  (∀ r s : ℝ,
    r ≠ s ∧ 
    (r ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧ 
    (r ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0})) → 
  c = 8 ∧ d = 9 := 
by
  sorry

end cubic_common_roots_l28_28140


namespace regular_polygon_with_12_degree_exterior_angle_has_30_sides_l28_28487

def regular_polygon_sides (e : ℤ) : ℤ :=
  360 / e

theorem regular_polygon_with_12_degree_exterior_angle_has_30_sides :
  regular_polygon_sides 12 = 30 :=
by
  -- Proof is omitted
  sorry

end regular_polygon_with_12_degree_exterior_angle_has_30_sides_l28_28487


namespace window_dimensions_l28_28674

-- Given conditions
def panes := 12
def rows := 3
def columns := 4
def height_to_width_ratio := 3
def border_width := 2

-- Definitions based on given conditions
def width_per_pane (x : ℝ) := x
def height_per_pane (x : ℝ) := 3 * x

def total_width (x : ℝ) := columns * width_per_pane x + (columns + 1) * border_width
def total_height (x : ℝ) := rows * height_per_pane x + (rows + 1) * border_width

-- Theorem statement: width and height of the window
theorem window_dimensions (x : ℝ) : 
  total_width x = 4 * x + 10 ∧ 
  total_height x = 9 * x + 8 := by
  sorry

end window_dimensions_l28_28674


namespace maximum_rabbits_l28_28392

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l28_28392


namespace vacation_cost_split_l28_28229

theorem vacation_cost_split (t d : ℕ) 
  (h_total : 105 + 125 + 175 = 405)
  (h_split : 405 / 3 = 135)
  (h_t : t = 135 - 105)
  (h_d : d = 135 - 125) : 
  t - d = 20 := by
  sorry

end vacation_cost_split_l28_28229


namespace smallest_four_digit_divisible_by_53_l28_28080

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28080


namespace value_of_m_l28_28537

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end value_of_m_l28_28537


namespace least_xy_value_l28_28161

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l28_28161


namespace smallest_four_digit_div_by_53_l28_28051

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28051


namespace _l28_28167

-- Define the main theorem with the given condition and the required conclusion
example (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : Real.log (4 ** -a) / Real.log 3 = -2 := by
  sorry

end _l28_28167


namespace non_integer_x_and_y_impossible_l28_28807

theorem non_integer_x_and_y_impossible 
  (x y : ℚ) (m n : ℤ) 
  (h1 : 5 * x + 7 * y = m)
  (h2 : 7 * x + 10 * y = n) : 
  ∃ (x y : ℤ), 5 * x + 7 * y = m ∧ 7 * x + 10 * y = n := 
sorry

end non_integer_x_and_y_impossible_l28_28807


namespace probability_no_adjacent_green_hats_l28_28967

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l28_28967


namespace smallest_four_digit_divisible_by_53_l28_28086

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28086


namespace find_coordinates_of_M_l28_28686

-- Definitions of the points A, B, C
def A : (ℝ × ℝ) := (2, -4)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definitions of vectors CA and CB
def vector_CA : (ℝ × ℝ) := (A.1 - C.1, A.2 - C.2)
def vector_CB : (ℝ × ℝ) := (B.1 - C.1, B.2 - C.2)

-- Definition of the point M
def M : (ℝ × ℝ) := (-11, -15)

-- Definition of vector CM
def vector_CM : (ℝ × ℝ) := (M.1 - C.1, M.2 - C.2)

-- The condition to prove
theorem find_coordinates_of_M : vector_CM = (2 * vector_CA.1 + 3 * vector_CB.1, 2 * vector_CA.2 + 3 * vector_CB.2) :=
by
  sorry

end find_coordinates_of_M_l28_28686


namespace complex_square_eq_l28_28885

theorem complex_square_eq (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I) : 
  a + b * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end complex_square_eq_l28_28885


namespace g_minus_6_eq_neg_20_l28_28891

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l28_28891


namespace smallest_four_digit_multiple_of_53_l28_28013

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28013


namespace part1_part2_l28_28837

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * x - (x + 1) * log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  x * log x - a * x^2 - 1

/- First part: Prove that for all x \in (1, +\infty), f(x) < 2 -/
theorem part1 (x : ℝ) (hx : 1 < x) : f x < 2 := sorry

/- Second part: Prove that if g(x) = 0 has two roots x₁ and x₂, then 
   (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) -/
theorem part2 (a x₁ x₂ : ℝ) (hx₁ : g x₁ a = 0) (hx₂ : g x₂ a = 0) : 
  (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) := sorry

end part1_part2_l28_28837


namespace remainder_sum_mod_11_l28_28278

theorem remainder_sum_mod_11 :
  (72501 + 72502 + 72503 + 72504 + 72505 + 72506 + 72507 + 72508 + 72509 + 72510) % 11 = 5 :=
by
  sorry

end remainder_sum_mod_11_l28_28278


namespace number_of_truthful_people_l28_28749

-- Definitions from conditions
def people := Fin 100
def tells_truth (p : people) : Prop := sorry -- Placeholder definition.

-- Conditions
axiom c1 : ∃ p : people, ¬ tells_truth p
axiom c2 : ∀ p1 p2 : people, p1 ≠ p2 → (tells_truth p1 ∨ tells_truth p2)

-- Goal
theorem number_of_truthful_people : 
  ∃ S : Finset people, S.card = 99 ∧ (∀ p ∈ S, tells_truth p) :=
sorry

end number_of_truthful_people_l28_28749


namespace sum_fourth_powers_l28_28702

theorem sum_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 25 / 6 :=
by sorry

end sum_fourth_powers_l28_28702


namespace cranberry_juice_cost_l28_28972

theorem cranberry_juice_cost 
  (cost_per_ounce : ℕ) (number_of_ounces : ℕ) 
  (h1 : cost_per_ounce = 7) 
  (h2 : number_of_ounces = 12) : 
  cost_per_ounce * number_of_ounces = 84 := 
by 
  sorry

end cranberry_juice_cost_l28_28972


namespace intersection_eq_union_eq_l28_28344

noncomputable def A := {x : ℝ | -2 < x ∧ x <= 3}
noncomputable def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_eq : A ∩ B = {x : ℝ | -2 < x ∧ x < -1} := by
  sorry

theorem union_eq : A ∪ B = {x : ℝ | x <= 3 ∨ x > 4} := by
  sorry

end intersection_eq_union_eq_l28_28344


namespace value_of_expression_l28_28532

variable {x : ℝ}

theorem value_of_expression (h : x^2 - 3 * x = 2) : 3 * x^2 - 9 * x - 7 = -1 := by
  sorry

end value_of_expression_l28_28532


namespace smallest_four_digit_divisible_by_53_l28_28032

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28032


namespace smallest_four_digit_divisible_by_53_l28_28095

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28095


namespace least_value_xy_l28_28155

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l28_28155


namespace sum_of_remainders_l28_28757

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end sum_of_remainders_l28_28757


namespace line_of_symmetry_l28_28524

-- Definitions of the circles and the line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 4 * y - 1 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- The theorem stating the symmetry condition
theorem line_of_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), line ((x + x') / 2) ((y + y') / 2) ∧ circle2 x' y' :=
sorry

end line_of_symmetry_l28_28524


namespace general_formula_compare_Tn_l28_28685

open scoped BigOperators

-- Define the sequence {a_n} and its sum S_n
noncomputable def aSeq (n : ℕ) : ℕ := n + 1
noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range n, aSeq (k + 1)

-- Given condition
axiom given_condition (n : ℕ) : 2 * S n = (aSeq n - 1) * (aSeq n + 2)

-- Prove the general formula of the sequence
theorem general_formula (n : ℕ) : aSeq n = n + 1 :=
by
  sorry  -- proof

-- Define T_n sequence
noncomputable def T (n : ℕ) : ℕ := ∑ k in Finset.range n, (k - 1) * 2^k / (k * aSeq k)

-- Compare T_n with the given expression
theorem compare_Tn (n : ℕ) : 
  if n < 17 then T n < (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else if n = 17 then T n = (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else T n > (2^(n+1)*(18-n)-2*n-2)/(n+1) :=
by
  sorry  -- proof

end general_formula_compare_Tn_l28_28685


namespace max_rabbits_with_traits_l28_28403

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l28_28403


namespace probability_of_xiao_li_l28_28486

def total_students : ℕ := 5
def xiao_li : ℕ := 1

noncomputable def probability_xiao_li_chosen : ℚ :=
  (xiao_li : ℚ) / (total_students : ℚ)

theorem probability_of_xiao_li : probability_xiao_li_chosen = 1 / 5 :=
sorry

end probability_of_xiao_li_l28_28486


namespace smallest_four_digit_multiple_of_53_l28_28016

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28016


namespace smallest_four_digit_div_by_53_l28_28047

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28047


namespace trigonometric_identity_l28_28303

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 3) : 1 / (Real.sin x ^ 2 - 2 * Real.cos x ^ 2) = 10 / 7 :=
by
  sorry

end trigonometric_identity_l28_28303


namespace grace_have_30_pastries_l28_28803

theorem grace_have_30_pastries (F : ℕ) :
  (2 * (F + 8) + F + (F + 13) = 97) → (F + 13 = 30) :=
by
  sorry

end grace_have_30_pastries_l28_28803


namespace prob_8th_roll_last_l28_28845

-- Define the conditions as functions or constants
def prob_diff_rolls : ℚ := 5/6
def prob_same_roll : ℚ := 1/6

-- Define the theorem stating the probability of the 8th roll being the last roll
theorem prob_8th_roll_last : (1 : ℚ) * prob_diff_rolls^6 * prob_same_roll = 15625 / 279936 := 
sorry

end prob_8th_roll_last_l28_28845


namespace minimize_quadratic_function_l28_28457

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l28_28457


namespace find_g_neg_6_l28_28902

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l28_28902


namespace avg_starting_with_d_l28_28360

-- Define c and d as positive integers
variables (c d : ℤ) (hc : c > 0) (hd : d > 0)

-- Define d as the average of the seven consecutive integers starting with c
def avg_starting_with_c (c : ℤ) : ℤ := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7

-- Define the condition that d is the average of the seven consecutive integers starting with c
axiom d_is_avg_starting_with_c : d = avg_starting_with_c c

-- Prove that the average of the seven consecutive integers starting with d equals c + 6
theorem avg_starting_with_d (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : d = avg_starting_with_c c) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 := by
  sorry

end avg_starting_with_d_l28_28360


namespace new_container_volume_l28_28636

theorem new_container_volume (original_volume : ℕ) (factor : ℕ) (new_volume : ℕ) 
    (h1 : original_volume = 5) (h2 : factor = 4 * 4 * 4) : new_volume = 320 :=
by
  sorry

end new_container_volume_l28_28636


namespace ratio_w_to_y_l28_28746

variables {w x y z : ℝ}

theorem ratio_w_to_y
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 9) :
  w / y = 8 :=
by
  sorry

end ratio_w_to_y_l28_28746


namespace time_per_room_l28_28253

theorem time_per_room (R P T: ℕ) (h: ℕ) (h₁ : R = 11) (h₂ : P = 2) (h₃ : T = 63) (h₄ : h = T / (R - P)) : h = 7 :=
by
  sorry

end time_per_room_l28_28253


namespace smallest_four_digit_divisible_by_53_l28_28042

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28042


namespace second_person_days_l28_28880

theorem second_person_days (x : ℕ) (h1 : ∀ y : ℝ, y = 24 → 1 / y = 1 / 24)
  (h2 : ∀ z : ℝ, z = 15 → 1 / z = 1 / 15) :
  (1 / 24 + 1 / x = 1 / 15) → x = 40 :=
by
  intro h
  have h3 : 15 * (x + 24) = 24 * x := sorry
  have h4 : 15 * x + 360 = 24 * x := sorry
  have h5 : 360 = 24 * x - 15 * x := sorry
  have h6 : 360 = 9 * x := sorry
  have h7 : x = 360 / 9 := sorry
  have h8 : x = 40 := sorry
  exact h8

end second_person_days_l28_28880


namespace compare_negatives_l28_28804

theorem compare_negatives : -3 < -2 :=
by {
  -- Placeholder for proof
  sorry
}

end compare_negatives_l28_28804


namespace joan_missed_games_l28_28718

theorem joan_missed_games :
  ∀ (total_games attended_games missed_games : ℕ),
  total_games = 864 →
  attended_games = 395 →
  missed_games = total_games - attended_games →
  missed_games = 469 :=
by
  intros total_games attended_games missed_games H1 H2 H3
  rw [H1, H2] at H3
  exact H3

end joan_missed_games_l28_28718


namespace length_of_platform_l28_28763

def len_train : ℕ := 300 -- length of the train in meters
def time_platform : ℕ := 39 -- time to cross the platform in seconds
def time_pole : ℕ := 26 -- time to cross the signal pole in seconds

theorem length_of_platform (L : ℕ) (h1 : len_train / time_pole = (len_train + L) / time_platform) : L = 150 :=
  sorry

end length_of_platform_l28_28763


namespace sally_weekly_bread_l28_28724

-- Define the conditions
def monday_bread : Nat := 3
def tuesday_bread : Nat := 2
def wednesday_bread : Nat := 4
def thursday_bread : Nat := 2
def friday_bread : Nat := 1
def saturday_bread : Nat := 2 * 2  -- 2 sandwiches, 2 pieces each
def sunday_bread : Nat := 2

-- Define the total bread count
def total_bread : Nat := 
  monday_bread + 
  tuesday_bread + 
  wednesday_bread + 
  thursday_bread + 
  friday_bread + 
  saturday_bread + 
  sunday_bread

-- The proof statement
theorem sally_weekly_bread : total_bread = 18 := by
  sorry

end sally_weekly_bread_l28_28724


namespace sum_of_reciprocals_l28_28707

variables {a b : ℕ}

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem sum_of_reciprocals (h_sum : a + b = 55)
                           (h_hcf : HCF a b = 5)
                           (h_lcm : LCM a b = 120) :
  (1 / a : ℚ) + (1 / b) = 11 / 120 :=
sorry

end sum_of_reciprocals_l28_28707


namespace opposite_of_neg_two_thirds_l28_28595

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l28_28595


namespace range_of_a_l28_28317

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ (m n p : ℝ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ f m a = 2024 ∧ f n a = 2024 ∧ f p a = 2024) ↔
  2022 < a ∧ a < 2026 :=
sorry

end range_of_a_l28_28317


namespace smallest_four_digit_divisible_by_53_l28_28089

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28089


namespace tan_neg_3900_eq_sqrt3_l28_28992

theorem tan_neg_3900_eq_sqrt3 : Real.tan (-3900 * Real.pi / 180) = Real.sqrt 3 := by
  -- Definitions of trigonometric values at 60 degrees
  have h_cos : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h_sin : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Using periodicity of the tangent function
  sorry

end tan_neg_3900_eq_sqrt3_l28_28992


namespace omega_range_l28_28823

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 4), f ω x ≥ -2) :
  0 < ω ∧ ω ≤ 3 / 2 :=
by
  sorry

end omega_range_l28_28823


namespace chords_on_circle_l28_28212

theorem chords_on_circle (n : ℕ) (h : n = 10) : nat.choose n 2 = 45 :=
by {
  rw h,
  -- we can directly calculate choose 10 2
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num),
  sorry -- the actual detailed proof goes here.
}

end chords_on_circle_l28_28212


namespace find_a2_b2_c2_l28_28706

-- Define the roots, sum of the roots, sum of the product of the roots taken two at a time, and product of the roots
variables {a b c : ℝ}
variable (h_roots : a = b ∧ b = c)
variable (h_sum : a + b + c = 12)
variable (h_sum_products : a * b + b * c + a * c = 47)
variable (h_product : a * b * c = 30)

-- State the theorem
theorem find_a2_b2_c2 : (a^2 + b^2 + c^2) = 50 :=
by {
  sorry
}

end find_a2_b2_c2_l28_28706


namespace smallest_four_digit_divisible_by_53_l28_28087

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28087


namespace variance_of_data_l28_28748

theorem variance_of_data :
  let data := [3, 1, 0, -1, -3]
  let mean := (3 + 1 + 0 - 1 - 3) / (5:ℝ)
  let variance := (1 / 5:ℝ) * (3^2 + 1^2 + (-1)^2 + (-3)^2)
  variance = 4 := sorry

end variance_of_data_l28_28748


namespace sum_of_remainders_l28_28758

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end sum_of_remainders_l28_28758


namespace tom_is_15_years_younger_l28_28604

/-- 
Alice is now 30 years old.
Ten years ago, Alice was 4 times as old as Tom was then.
Prove that Tom is 15 years younger than Alice.
-/
theorem tom_is_15_years_younger (A T : ℕ) (h1 : A = 30) (h2 : A - 10 = 4 * (T - 10)) : A - T = 15 :=
by
  sorry

end tom_is_15_years_younger_l28_28604


namespace quadratic_roots_l28_28824

-- Define the condition for the quadratic equation
def quadratic_eq (x m : ℝ) : Prop := x^2 - 4*x + m + 2 = 0

-- Define the discriminant condition
def discriminant_pos (m : ℝ) : Prop := (4^2 - 4 * (m + 2)) > 0

-- Define the condition range for m
def m_range (m : ℝ) : Prop := m < 2

-- Define the condition for m as a positive integer
def m_positive_integer (m : ℕ) : Prop := m = 1

-- The main theorem stating the problem
theorem quadratic_roots : 
  (∀ (m : ℝ), discriminant_pos m → m_range m) ∧ 
  (∀ m : ℕ, m_positive_integer m → (∃ x1 x2 : ℝ, quadratic_eq x1 m ∧ quadratic_eq x2 m ∧ x1 = 1 ∧ x2 = 3)) := 
by 
  sorry

end quadratic_roots_l28_28824


namespace conditional_probability_event_B_given_event_A_l28_28117

-- Definitions of events A and B
def event_A := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i = 1 ∨ j = 1 ∨ k = 1)}
def event_B := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i + j + k = 1)}

-- Calculation of probabilities
def probability_AB := 3 / 8
def probability_A := 7 / 8

-- Prove conditional probability
theorem conditional_probability_event_B_given_event_A :
  (probability_AB / probability_A) = 3 / 7 :=
by
  sorry

end conditional_probability_event_B_given_event_A_l28_28117


namespace least_value_xy_l28_28153

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l28_28153


namespace smallest_fraction_numerator_l28_28498

theorem smallest_fraction_numerator :
  ∃ a b : ℕ, 10 ≤ a ∧ a < b ∧ b ≤ 99 ∧ (a : ℚ) / b > 5 / 6 ∧ a = 81 :=
by
  sorry

end smallest_fraction_numerator_l28_28498


namespace total_cookies_is_58_l28_28273

noncomputable def total_cookies : ℝ :=
  let M : ℝ := 5
  let T : ℝ := 2 * M
  let W : ℝ := T + 0.4 * T
  let Th : ℝ := W - 0.25 * W
  let F : ℝ := Th - 0.25 * Th
  let Sa : ℝ := F - 0.25 * F
  let Su : ℝ := Sa - 0.25 * Sa
  M + T + W + Th + F + Sa + Su

theorem total_cookies_is_58 : total_cookies = 58 :=
by
  sorry

end total_cookies_is_58_l28_28273


namespace minimum_value_expr_pos_reals_l28_28721

noncomputable def expr (a b : ℝ) := a^2 + b^2 + 2 * a * b + 1 / (a + b)^2

theorem minimum_value_expr_pos_reals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : 
  (expr a b) ≥ 2 :=
sorry

end minimum_value_expr_pos_reals_l28_28721


namespace carts_needed_each_day_last_two_days_l28_28752

-- Define capacities as per conditions
def daily_capacity_large_truck : ℚ := 1 / (3 * 4)
def daily_capacity_small_truck : ℚ := 1 / (4 * 5)
def daily_capacity_cart : ℚ := 1 / (20 * 6)

-- Define the number of carts required each day in the last two days
def required_carts_last_two_days : ℚ :=
  let total_work_done_by_large_trucks := 2 * daily_capacity_large_truck * 2
  let total_work_done_by_small_trucks := 3 * daily_capacity_small_truck * 2
  let total_work_done_by_carts := 7 * daily_capacity_cart * 2
  let total_work_done := total_work_done_by_large_trucks + total_work_done_by_small_trucks + total_work_done_by_carts
  let remaining_work := 1 - total_work_done
  remaining_work / (2 * daily_capacity_cart)

-- Assertion of the number of carts required
theorem carts_needed_each_day_last_two_days :
  required_carts_last_two_days = 15 := by
  sorry

end carts_needed_each_day_last_two_days_l28_28752


namespace sport_formulation_water_quantity_l28_28765

theorem sport_formulation_water_quantity (flavoring : ℝ) (corn_syrup : ℝ) (water : ℝ)
    (hs : flavoring / corn_syrup = 1 / 12) 
    (hw : flavoring / water = 1 / 30) 
    (sport_fs_ratio : flavoring / corn_syrup = 3 * (1 / 12)) 
    (sport_fw_ratio : flavoring / water = (1 / 2) * (1 / 30)) 
    (cs_sport : corn_syrup = 1) : 
    water = 15 :=
by
  sorry

end sport_formulation_water_quantity_l28_28765


namespace find_x_l28_28235

theorem find_x (u : ℕ) (h₁ : u = 90) (w : ℕ) (h₂ : w = u + 10)
                (z : ℕ) (h₃ : z = w + 25) (y : ℕ) (h₄ : y = z + 15)
                (x : ℕ) (h₅ : x = y + 3) : x = 143 :=
by {
  -- Proof will be included here
  sorry
}

end find_x_l28_28235


namespace sum_f_eq_26_l28_28515

def f (n : ℕ) : ℝ :=
  if (Real.log n / Real.log 8).isRational then Real.log n / Real.log 8 else 0

theorem sum_f_eq_26 : (Finset.range 4095).sum (λ n, f (n + 1)) = 26 := 
  sorry

end sum_f_eq_26_l28_28515


namespace probability_of_picking_letter_in_mathematics_l28_28533

def unique_letters_in_mathematics : List Char := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']

def number_of_unique_letters_in_word : ℕ := unique_letters_in_mathematics.length

def total_letters_in_alphabet : ℕ := 26

theorem probability_of_picking_letter_in_mathematics :
  (number_of_unique_letters_in_word : ℚ) / total_letters_in_alphabet = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l28_28533


namespace area_of_triangle_l28_28946

-- Define the lines as functions
def line1 : ℝ → ℝ := fun x => 3 * x - 4
def line2 : ℝ → ℝ := fun x => -2 * x + 16

-- Define the vertices of the triangle formed by lines and y-axis
def vertex1 : ℝ × ℝ := (0, -4)
def vertex2 : ℝ × ℝ := (0, 16)
def vertex3 : ℝ × ℝ := (4, 8)

-- Define the proof statement
theorem area_of_triangle : 
  let A := vertex1 
  let B := vertex2 
  let C := vertex3 
  -- Compute the area of the triangle using the determinant formula
  let area := (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 40 := 
by
  sorry

end area_of_triangle_l28_28946


namespace total_staff_correct_l28_28221

noncomputable def total_staff_weekdays_weekends : ℕ := 84

theorem total_staff_correct :
  let chefs_weekdays := 16
  let waiters_weekdays := 16
  let busboys_weekdays := 10
  let hostesses_weekdays := 5
  let additional_chefs_weekends := 5
  let additional_hostesses_weekends := 2
  
  let chefs_leave := chefs_weekdays * 25 / 100
  let waiters_leave := waiters_weekdays * 20 / 100
  let busboys_leave := busboys_weekdays * 30 / 100
  let hostesses_leave := hostesses_weekdays * 15 / 100
  
  let chefs_left_weekdays := chefs_weekdays - chefs_leave
  let waiters_left_weekdays := waiters_weekdays - Nat.floor waiters_leave
  let busboys_left_weekdays := busboys_weekdays - busboys_leave
  let hostesses_left_weekdays := hostesses_weekdays - Nat.ceil hostesses_leave

  let total_staff_weekdays := chefs_left_weekdays + waiters_left_weekdays + busboys_left_weekdays + hostesses_left_weekdays

  let chefs_weekends := chefs_weekdays + additional_chefs_weekends
  let waiters_weekends := waiters_left_weekdays
  let busboys_weekends := busboys_left_weekdays
  let hostesses_weekends := hostesses_weekdays + additional_hostesses_weekends
  
  let total_staff_weekends := chefs_weekends + waiters_weekends + busboys_weekends + hostesses_weekends

  total_staff_weekdays + total_staff_weekends = total_staff_weekdays_weekends
:= by
  sorry

end total_staff_correct_l28_28221


namespace smallest_four_digit_divisible_by_53_l28_28073

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28073


namespace hockey_players_l28_28433

theorem hockey_players (n : ℕ) (h1 : n < 30) (h2 : n % 2 = 0) (h3 : n % 4 = 0) (h4 : n % 7 = 0) :
  (n / 4 = 7) :=
by
  sorry

end hockey_players_l28_28433


namespace area_of_rectangle_l28_28768

-- Definitions from the conditions
def breadth (b : ℝ) : Prop := b > 0
def length (l b : ℝ) : Prop := l = 3 * b
def perimeter (P l b : ℝ) : Prop := P = 2 * (l + b)

-- The main theorem we are proving
theorem area_of_rectangle (b l : ℝ) (P : ℝ) (h1 : breadth b) (h2 : length l b) (h3 : perimeter P l b) (h4 : P = 96) : l * b = 432 := 
by
  -- Proof steps will go here
  sorry

end area_of_rectangle_l28_28768


namespace factor_x4_minus_81_l28_28137

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, (Polynomial.X ^ 4 - 81 : Polynomial ℝ) = (Polynomial.X - 3) * (Polynomial.X + 3) * (Polynomial.X ^ 2 + 9) := 
sorry

end factor_x4_minus_81_l28_28137


namespace minimum_expression_value_l28_28304

theorem minimum_expression_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := 
by
  sorry

end minimum_expression_value_l28_28304


namespace sum_of_smallest_and_largest_eq_2y_l28_28579

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l28_28579


namespace jamie_minimum_4th_quarter_score_l28_28938

-- Define the conditions for Jamie's scores and the average requirement
def qualifying_score := 85
def first_quarter_score := 80
def second_quarter_score := 85
def third_quarter_score := 78

-- The function to determine the required score in the 4th quarter
def minimum_score_for_quarter (N : ℕ) := first_quarter_score + second_quarter_score + third_quarter_score + N ≥ 4 * qualifying_score

-- The main statement to be proved
theorem jamie_minimum_4th_quarter_score (N : ℕ) : minimum_score_for_quarter N ↔ N ≥ 97 :=
by
  sorry

end jamie_minimum_4th_quarter_score_l28_28938


namespace geometric_product_Pi8_l28_28352

def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

variables {a : ℕ → ℝ}
variable (h_geom : geometric_sequence a)
variable (h_prod : a 4 * a 5 = 2)

theorem geometric_product_Pi8 :
  (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = 16 :=
by
  sorry

end geometric_product_Pi8_l28_28352


namespace jellybean_avg_increase_l28_28751

noncomputable def avg_increase_jellybeans 
  (avg_original : ℕ) (num_bags_original : ℕ) (num_jellybeans_new_bag : ℕ) : ℕ :=
  let total_original := avg_original * num_bags_original
  let total_new := total_original + num_jellybeans_new_bag
  let num_bags_new := num_bags_original + 1
  let avg_new := total_new / num_bags_new
  avg_new - avg_original

theorem jellybean_avg_increase :
  avg_increase_jellybeans 117 34 362 = 7 := by
  let total_original := 117 * 34
  let total_new := total_original + 362
  let num_bags_new := 34 + 1
  let avg_new := total_new / num_bags_new
  let increase := avg_new - 117
  have h1 : total_original = 3978 := by norm_num
  have h2 : total_new = 4340 := by norm_num
  have h3 : num_bags_new = 35 := by norm_num
  have h4 : avg_new = 124 := by norm_num
  have h5 : increase = 7 := by norm_num
  exact h5

end jellybean_avg_increase_l28_28751


namespace smallest_four_digit_divisible_by_53_l28_28025

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28025


namespace sum_of_smallest_and_largest_l28_28577

def even_consecutive_sequence_sum (a n : ℤ) : ℤ :=
  a + a + 2 * (n - 1)

def arithmetic_mean (a n : ℤ) : ℤ :=
  (a * n + n * (n - 1)) / n

theorem sum_of_smallest_and_largest (a n y : ℤ) (h_even : even n) (h_mean : y = arithmetic_mean a n) :
  even_consecutive_sequence_sum a n = 2 * y :=
by
  sorry

end sum_of_smallest_and_largest_l28_28577


namespace sin_cos_sum_eq_one_or_neg_one_l28_28704

theorem sin_cos_sum_eq_one_or_neg_one (α : ℝ) (h : (Real.sin α)^4 + (Real.cos α)^4 = 1) : (Real.sin α + Real.cos α) = 1 ∨ (Real.sin α + Real.cos α) = -1 :=
sorry

end sin_cos_sum_eq_one_or_neg_one_l28_28704


namespace line_intersection_l28_28977

-- Parameters for the first line
def line1_param (s : ℝ) : ℝ × ℝ := (1 - 2 * s, 4 + 3 * s)

-- Parameters for the second line
def line2_param (v : ℝ) : ℝ × ℝ := (-v, 5 + 6 * v)

-- Statement of the intersection point
theorem line_intersection :
  ∃ (s v : ℝ), line1_param s = (-1 / 9, 17 / 3) ∧ line2_param v = (-1 / 9, 17 / 3) :=
by
  -- Placeholder for the proof, which we are not providing as per instructions
  sorry

end line_intersection_l28_28977


namespace greatest_mass_l28_28230

theorem greatest_mass (V : ℝ) (h : ℝ) (l : ℝ) 
    (ρ_Hg ρ_H2O ρ_Oil : ℝ) 
    (V1 V2 V3 : ℝ) 
    (m_Hg m_H2O m_Oil : ℝ)
    (ρ_Hg_val : ρ_Hg = 13.59) 
    (ρ_H2O_val : ρ_H2O = 1) 
    (ρ_Oil_val : ρ_Oil = 0.915) 
    (height_layers_equal : h = l) :
    ∀ V1 V2 V3 m_Hg m_H2O m_Oil, 
    V1 + V2 + V3 = 27 * (l^3) → 
    V2 = 7 * V1 → 
    V3 = 19 * V1 → 
    m_Hg = ρ_Hg * V1 → 
    m_H2O = ρ_H2O * V2 → 
    m_Oil = ρ_Oil * V3 → 
    m_Oil > m_Hg ∧ m_Oil > m_H2O := 
by 
    intros
    sorry

end greatest_mass_l28_28230


namespace find_f_of_2_l28_28525

-- Given definitions:
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def defined_on_neg_inf_to_0 (f : ℝ → ℝ) : Prop := ∀ x, x < 0 → f x = 2 * x^3 + x^2

-- The main theorem to prove:
theorem find_f_of_2 (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_def : defined_on_neg_inf_to_0 f) :
  f 2 = 12 :=
sorry

end find_f_of_2_l28_28525


namespace find_g_neg6_l28_28899

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l28_28899


namespace intersection_of_complement_l28_28723

open Set

variable (U : Set ℤ) (A B : Set ℤ)

def complement (U A : Set ℤ) : Set ℤ := U \ A

theorem intersection_of_complement (hU : U = {-1, 0, 1, 2, 3, 4})
  (hA : A = {1, 2, 3, 4}) (hB : B = {0, 2}) :
  (complement U A) ∩ B = {0} :=
by
  sorry

end intersection_of_complement_l28_28723


namespace johns_profit_l28_28556

variable (numDucks : ℕ) (duckCost : ℕ) (duckWeight : ℕ) (sellPrice : ℕ)

def totalCost (numDucks duckCost : ℕ) : ℕ :=
  numDucks * duckCost

def totalWeight (numDucks duckWeight : ℕ) : ℕ :=
  numDucks * duckWeight

def totalRevenue (totalWeight sellPrice : ℕ) : ℕ :=
  totalWeight * sellPrice

def profit (totalRevenue totalCost : ℕ) : ℕ :=
  totalRevenue - totalCost

theorem johns_profit :
  totalCost 30 10 = 300 →
  totalWeight 30 4 = 120 →
  totalRevenue 120 5 = 600 →
  profit 600 300 = 300 :=
  by
    intros
    sorry

end johns_profit_l28_28556


namespace equilateral_triangle_l28_28739

variable (A B C A₀ B₀ C₀ : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]

variable (midpoint : ∀ (X₁ X₂ : Type), Type) 
variable (circumcircle : ∀ (X Y Z : Type), Type)

def medians_meet_circumcircle := ∀ (A A₁ B B₁ C C₁ : Type) 
  [AddGroup A] [AddGroup A₁] [AddGroup B] [AddGroup B₁] [AddGroup C] [AddGroup C₁], 
  Prop

def areas_equal := ∀ (ABC₀ AB₀C A₀BC : Type) 
  [AddGroup ABC₀] [AddGroup AB₀C] [AddGroup A₀BC], 
  Prop

theorem equilateral_triangle (A B C A₀ B₀ C₀ A₁ B₁ C₁ : Type)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A₀] [AddGroup B₀] [AddGroup C₀]
  [AddGroup A₁] [AddGroup B₁] [AddGroup C₁] 
  (midpoint_cond : ∀ (X Y Z : Type), Z = midpoint X Y)
  (circumcircle_cond : ∀ (X Y Z : Type), Z = circumcircle X Y Z)
  (medians_meet_circumcircle : Prop)
  (areas_equal: Prop) :
    A = B ∧ B = C ∧ C = A :=
  sorry

end equilateral_triangle_l28_28739


namespace percentage_students_qualified_school_A_l28_28187

theorem percentage_students_qualified_school_A 
  (A Q : ℝ)
  (h1 : 1.20 * A = A + 0.20 * A)
  (h2 : 1.50 * Q = Q + 0.50 * Q)
  (h3 : (1.50 * Q / 1.20 * A) * 100 = 87.5) :
  (Q / A) * 100 = 58.33 := sorry

end percentage_students_qualified_school_A_l28_28187


namespace number_of_chords_l28_28216

theorem number_of_chords (n : ℕ) (h : n = 10) : finset.card (finset.pairs (finset.range n)) = 45 :=
by
  rw [h]
  -- Sorry to skip the proof steps as required
  sorry

end number_of_chords_l28_28216


namespace three_colored_flag_l28_28701

theorem three_colored_flag (colors : Finset ℕ) (h : colors.card = 6) : 
  (∃ top middle bottom : ℕ, top ≠ middle ∧ top ≠ bottom ∧ middle ≠ bottom ∧ 
                            top ∈ colors ∧ middle ∈ colors ∧ bottom ∈ colors) → 
  colors.card * (colors.card - 1) * (colors.card - 2) = 120 :=
by 
  intro h_exists
  exact sorry

end three_colored_flag_l28_28701


namespace more_non_representable_ten_digit_numbers_l28_28983

-- Define the range of ten-digit numbers
def total_ten_digit_numbers : ℕ := 9 * 10^9

-- Define the range of five-digit numbers
def total_five_digit_numbers : ℕ := 90000

-- Calculate the number of pairs of five-digit numbers
def number_of_pairs_five_digit_numbers : ℕ :=
  total_five_digit_numbers * (total_five_digit_numbers + 1)

-- Problem statement
theorem more_non_representable_ten_digit_numbers:
  number_of_pairs_five_digit_numbers < total_ten_digit_numbers :=
by
  -- Proof is non-computable and should be added here
  sorry

end more_non_representable_ten_digit_numbers_l28_28983


namespace smallest_four_digit_multiple_of_53_l28_28017

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28017


namespace range_of_a_l28_28558

noncomputable def M : Set ℝ := {2, 0, -1}
noncomputable def N (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}

theorem range_of_a (a : ℝ) : (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 3) ↔ M ∩ N a = {x} :=
by
  sorry

end range_of_a_l28_28558


namespace max_red_balls_l28_28434

theorem max_red_balls (r w : ℕ) (h1 : r = 3 * w) (h2 : r + w ≤ 50) : r = 36 :=
sorry

end max_red_balls_l28_28434


namespace heather_bicycling_time_l28_28323

theorem heather_bicycling_time (distance speed : ℝ) (h_distance : distance = 40) (h_speed : speed = 8) : (distance / speed) = 5 := 
by
  rw [h_distance, h_speed]
  norm_num

end heather_bicycling_time_l28_28323


namespace decimal_representation_of_fraction_l28_28672

theorem decimal_representation_of_fraction :
  (3 / 40 : ℝ) = 0.075 :=
sorry

end decimal_representation_of_fraction_l28_28672


namespace smallest_four_digit_divisible_by_53_l28_28084

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28084


namespace mn_not_equal_l28_28200

-- Define conditions for the problem
def isValidN (N : ℕ) (n : ℕ) : Prop :=
  0 ≤ N ∧ N < 10^n ∧ N % 4 = 0 ∧ ((N.digits 10).sum % 4 = 0)

-- Define the number M_n of integers N satisfying the conditions
noncomputable def countMn (n : ℕ) : ℕ :=
  Nat.card { N : ℕ | isValidN N n }

-- Define the theorem stating the problem's conclusion
theorem mn_not_equal (n : ℕ) (hn : n > 0) : 
  countMn n ≠ 10^n / 16 :=
sorry

end mn_not_equal_l28_28200


namespace g_neg_six_eq_neg_twenty_l28_28914

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l28_28914


namespace triangle_B_eq_2A_range_of_a_l28_28308

theorem triangle_B_eq_2A (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = c) : B = 2 * A := 
sorry

theorem range_of_a (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = 2) (h6 : 0 < (π - A - B)) (h7 : (π - A - B) < π/2) : 1 < a ∧ a < 2 := 
sorry

end triangle_B_eq_2A_range_of_a_l28_28308


namespace rational_solution_for_quadratic_l28_28144

theorem rational_solution_for_quadratic (k : ℕ) (h_pos : 0 < k) : 
  ∃ m : ℕ, (18^2 - 4 * k * (2 * k)) = m^2 ↔ k = 4 :=
by
  sorry

end rational_solution_for_quadratic_l28_28144


namespace ball_hits_ground_time_l28_28973

theorem ball_hits_ground_time (h : ℝ → ℝ) (t : ℝ) :
  (∀ (t : ℝ), h t = -16 * t ^ 2 - 30 * t + 200) → h t = 0 → t = 2.5 :=
by
  -- Placeholder for the formal proof
  sorry

end ball_hits_ground_time_l28_28973


namespace smallest_fraction_numerator_l28_28500

theorem smallest_fraction_numerator :
  ∃ a b : ℕ, 10 ≤ a ∧ a < b ∧ b ≤ 99 ∧ 6 * a > 5 * b ∧ ∀ c d : ℕ,
    (10 ≤ c ∧ c < d ∧ d ≤ 99 ∧ 6 * c > 5 * d → a ≤ c) ∧ 
    a = 81 :=
sorry

end smallest_fraction_numerator_l28_28500


namespace problem_statement_l28_28162

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 : Prop := a^2 + b^2 - 4 * a ≤ 1
def condition2 : Prop := b^2 + c^2 - 8 * b ≤ -3
def condition3 : Prop := c^2 + a^2 - 12 * c ≤ -26

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c a) : (a + b) ^ c = 27 :=
by sorry

end problem_statement_l28_28162


namespace basin_capacity_l28_28971

-- Defining the flow rate of water into the basin
def inflow_rate : ℕ := 24

-- Defining the leak rate of the basin
def leak_rate : ℕ := 4

-- Defining the time taken to fill the basin in seconds
def fill_time : ℕ := 13

-- Net rate of filling the basin
def net_rate : ℕ := inflow_rate - leak_rate

-- Volume of the basin
def basin_volume : ℕ := net_rate * fill_time

-- The goal is to prove that the volume of the basin is 260 gallons
theorem basin_capacity : basin_volume = 260 := by
  sorry

end basin_capacity_l28_28971


namespace least_xy_l28_28158

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l28_28158


namespace simplify_fraction_l28_28365

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28365


namespace proof_verification_l28_28188

section
variables (scores : List ℝ) (n : ℕ)
def given_scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

def median (l : List ℝ) : ℝ :=
let sorted := List.sort (≤) l in
if hl : l.length % 2 = 0 then (sorted.get (l.length / 2 - 1) + sorted.get (l.length / 2)) / 2
else sorted.get ((l.length - 1) / 2)

def percentile (l : List ℝ) (p : ℝ) : ℝ :=
let sorted := List.sort (≤) l in
sorted.get ⟨(p * (l.length + 1) / 100 : ℝ).ceil.toNat⟩

def average (l : List ℝ) : ℝ :=
l.sum / l.length

def variance (l : List ℝ) : ℝ :=
let μ := average l in
(l.map (λ x, (x - μ) ^ 2)).sum / l.length

def verify_statements : Prop :=
  (median given_scores = 90) ∧
  (percentile given_scores 60 = 91) ∧
  (average given_scores ≤ median given_scores) ∧
  let new_scores := given_scores.erase 70 |>.erase 100 in
  average new_scores > average given_scores ∧
  variance new_scores < variance given_scores

theorem proof_verification : verify_statements given_scores :=
sorry
end

end proof_verification_l28_28188


namespace min_employees_wednesday_l28_28131

noncomputable def minWednesdayBirthdays (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) :
  ℕ :=
  40

theorem min_employees_wednesday (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) 
  (h1 : total_employees = 61) 
  (h2 : ∃ lst, diff_birthdays lst ∧ max_birthdays 40 lst) :
  minWednesdayBirthdays total_employees diff_birthdays max_birthdays = 40 := 
sorry

end min_employees_wednesday_l28_28131


namespace value_of_expression_l28_28296

theorem value_of_expression : (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = (27 / 89) :=
by
  sorry

end value_of_expression_l28_28296


namespace find_n_divisors_l28_28743

theorem find_n_divisors (n : ℕ) (h1 : 2287 % n = 2028 % n)
                        (h2 : 2028 % n = 1806 % n) : n = 37 := 
by
  sorry

end find_n_divisors_l28_28743


namespace trees_planted_l28_28602

theorem trees_planted (current_short_trees planted_short_trees total_short_trees : ℕ)
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees = 217) :
  planted_short_trees = 105 :=
by
  sorry

end trees_planted_l28_28602


namespace angle_x_l28_28714

-- Conditions
variable (ABC BAC CDE DCE : ℝ)
variable (h1 : ABC = 70)
variable (h2 : BAC = 50)
variable (h3 : CDE = 90)
variable (h4 : ∃ BCA : ℝ, DCE = BCA ∧ ABC + BAC + BCA = 180)

-- The statement to prove
theorem angle_x (x : ℝ) (h : ∃ BCA : ℝ, (ABC = 70) ∧ (BAC = 50) ∧ (CDE = 90) ∧ (DCE = BCA ∧ ABC + BAC + BCA = 180) ∧ (DCE + x = 90)) :
  x = 30 := by
  sorry

end angle_x_l28_28714


namespace max_rabbits_l28_28412

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l28_28412


namespace finite_S_k_iff_k_power_of_2_l28_28814

def S_k_finite (k : ℕ) : Prop :=
  ∃ (n a b : ℕ), (n ≠ 0 ∧ n % 2 = 1) ∧ (a + b = k) ∧ (Nat.gcd a b = 1) ∧ (n ∣ (a^n + b^n))

theorem finite_S_k_iff_k_power_of_2 (k : ℕ) (h : k > 1) : 
  (∀ n a b, n ≠ 0 → n % 2 = 1 → a + b = k → Nat.gcd a b = 1 → n ∣ (a^n + b^n) → false) ↔ 
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end finite_S_k_iff_k_power_of_2_l28_28814


namespace problem1_union_problem2_intersection_problem3_subset_l28_28699

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 4 ≤ 0}

theorem problem1_union (m : ℝ) (hm : m = 2) : A ∪ B m = {x | -1 ≤ x ∧ x ≤ 4} :=
sorry

theorem problem2_intersection (m : ℝ) (h : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3}) : m = 3 :=
sorry

theorem problem3_subset (m : ℝ) (h : A ⊆ {x | ¬ (x ∈ B m)}) : m > 5 ∨ m < -3 :=
sorry

end problem1_union_problem2_intersection_problem3_subset_l28_28699


namespace smallest_four_digit_divisible_by_53_l28_28064

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28064


namespace nancy_crayons_l28_28872

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end nancy_crayons_l28_28872


namespace no_adjacent_green_hats_l28_28964

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l28_28964


namespace smallest_four_digit_div_by_53_l28_28050

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28050


namespace simplify_fraction_l28_28367

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28367


namespace distance_city_A_B_l28_28243

theorem distance_city_A_B (D : ℝ) : 
  (3 : ℝ) + (2.5 : ℝ) = 5.5 → 
  ∃ T_saved, T_saved = 1 →
  80 = (2 * D) / (5.5 - T_saved) →
  D = 180 :=
by
  intros
  sorry

end distance_city_A_B_l28_28243


namespace necessary_not_sufficient_condition_l28_28517

theorem necessary_not_sufficient_condition (a : ℝ) :
  (a < 2) ∧ (a^2 - 4 < 0) ↔ (a < 2) ∧ (a > -2) :=
by
  sorry

end necessary_not_sufficient_condition_l28_28517


namespace fraction_even_odd_phonenumbers_l28_28274

-- Define a predicate for valid phone numbers
def isValidPhoneNumber (n : Nat) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ (n / 1000000 ≠ 0) ∧ (n / 1000000 ≠ 1)

-- Calculate the total number of valid phone numbers
def totalValidPhoneNumbers : Nat :=
  4 * 10^6

-- Calculate the number of valid phone numbers that begin with an even digit and end with an odd digit
def validEvenOddPhoneNumbers : Nat :=
  4 * (10^5) * 5

-- Determine the fraction of such phone numbers (valid ones and valid even-odd ones)
theorem fraction_even_odd_phonenumbers : 
  (validEvenOddPhoneNumbers) / (totalValidPhoneNumbers) = 1 / 2 :=
by {
  sorry
}

end fraction_even_odd_phonenumbers_l28_28274


namespace calc1_calc2_l28_28769

variable (a b : ℝ) 

theorem calc1 : (-b)^2 * (-b)^3 * (-b)^5 = b^10 :=
by sorry

theorem calc2 : (2 * a * b^2)^3 = 8 * a^3 * b^6 :=
by sorry

end calc1_calc2_l28_28769


namespace units_digit_fraction_l28_28233

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end units_digit_fraction_l28_28233


namespace prime_addition_fraction_equivalence_l28_28236

theorem prime_addition_fraction_equivalence : 
  ∃ n : ℕ, Prime n ∧ (4 + n) * 8 = (7 + n) * 7 ∧ n = 17 := 
sorry

end prime_addition_fraction_equivalence_l28_28236


namespace multiple_of_shorter_piece_l28_28625

theorem multiple_of_shorter_piece :
  ∃ (m : ℕ), 
  (35 + (m * 35 + 15) = 120) ∧
  (m = 2) :=
by
  sorry

end multiple_of_shorter_piece_l28_28625


namespace midpoint_x_sum_l28_28933

variable {p q r s : ℝ}

theorem midpoint_x_sum (h : p + q + r + s = 20) :
  ((p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2) = 20 :=
by
  sorry

end midpoint_x_sum_l28_28933


namespace range_of_a_l28_28740

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → quadratic_function a x ≥ quadratic_function a y ∧ y ≤ 4) →
  a ≤ -5 :=
by sorry

end range_of_a_l28_28740


namespace largest_angle_is_176_l28_28630

-- Define the angles of the pentagon
def angle1 (y : ℚ) : ℚ := y
def angle2 (y : ℚ) : ℚ := 2 * y + 2
def angle3 (y : ℚ) : ℚ := 3 * y - 3
def angle4 (y : ℚ) : ℚ := 4 * y + 4
def angle5 (y : ℚ) : ℚ := 5 * y - 5

-- Define the function to calculate the largest angle
def largest_angle (y : ℚ) : ℚ := 5 * y - 5

-- Problem statement: Prove that the largest angle in the pentagon is 176 degrees
theorem largest_angle_is_176 (y : ℚ) (h : angle1 y + angle2 y + angle3 y + angle4 y + angle5 y = 540) :
  largest_angle y = 176 :=
by sorry

end largest_angle_is_176_l28_28630


namespace simplify_expression_correct_l28_28364

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l28_28364


namespace combine_polynomials_find_value_profit_or_loss_l28_28968

-- Problem 1, Part ①
theorem combine_polynomials (a b : ℝ) : -3 * (a+b)^2 - 6 * (a+b)^2 + 8 * (a+b)^2 = -(a+b)^2 := 
sorry

-- Problem 1, Part ②
theorem find_value (a b c d : ℝ) (h1 : a - 2 * b = 5) (h2 : 2 * b - c = -7) (h3 : c - d = 12) : 
  4 * (a - c) + 4 * (2 * b - d) - 4 * (2 * b - c) = 40 := 
sorry

-- Problem 2
theorem profit_or_loss (initial_cost : ℝ) (selling_prices : ℕ → ℝ) (base_price : ℝ) 
  (h_prices : selling_prices 0 = -3) (h_prices1 : selling_prices 1 = 7) 
  (h_prices2 : selling_prices 2 = -8) (h_prices3 : selling_prices 3 = 9) 
  (h_prices4 : selling_prices 4 = -2) (h_prices5 : selling_prices 5 = 0) 
  (h_prices6 : selling_prices 6 = -1) (h_prices7 : selling_prices 7 = -6) 
  (h_initial_cost : initial_cost = 400) (h_base_price : base_price = 56) : 
  (selling_prices 0 + selling_prices 1 + selling_prices 2 + selling_prices 3 + selling_prices 4 + selling_prices 5 + 
  selling_prices 6 + selling_prices 7 + 8 * base_price) - initial_cost > 0 := 
sorry

end combine_polynomials_find_value_profit_or_loss_l28_28968


namespace smallest_four_digit_divisible_by_53_l28_28063

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28063


namespace solve_absolute_value_equation_l28_28237

theorem solve_absolute_value_equation :
  {x : ℝ | 3 * x^2 + 3 * x + 6 = abs (-20 + 5 * x)} = {1.21, -3.87} :=
by
  sorry

end solve_absolute_value_equation_l28_28237


namespace original_quantity_of_ghee_l28_28957

theorem original_quantity_of_ghee
  (Q : ℝ) 
  (H1 : (0.5 * Q) = (0.3 * (Q + 20))) : 
  Q = 30 := 
by
  -- proof goes here
  sorry

end original_quantity_of_ghee_l28_28957


namespace simplified_expression_l28_28576

theorem simplified_expression :
  ( (81 / 16) ^ (3 / 4) - (-1) ^ 0 ) = 19 / 8 := 
by 
  -- It is a placeholder for the actual proof.
  sorry

end simplified_expression_l28_28576


namespace find_min_value_l28_28687

theorem find_min_value (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y = 2 → c = 1 / 2 ∧ (8 / ((x + 2) * (y + 4))) ≥ c) :=
  sorry

end find_min_value_l28_28687


namespace smallest_n_value_l28_28715

open Real

theorem smallest_n_value {ABC : Triangle} (AB BC CA : ℝ) (hAB : AB = 52) 
  (hBC : BC = 34) (hCA : CA = 50) (n : ℕ) :
  (∃ (split_points : Finset (SegmentPoint (Subsegment BC) n)),
      (∃ (D : SegmentPoint (Subsegment BC) spl), True) ∧
      (∃ (M : SegmentPoint (Subsegment BC) (Fin.mk (BC/2) (by linarith))), True) ∧
      (∃ (X : SegmentPoint (Subsegment BC) (Fin.mk ((ℝ.divsup (ℝ.add (51/51) 26)) 25) (by linarith))), True)
  ) → n = 102 :=
begin
  sorry
end

end smallest_n_value_l28_28715


namespace smallest_four_digit_multiple_of_53_l28_28006

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28006


namespace find_percentage_l28_28772

theorem find_percentage (P : ℕ) (h1 : 0.20 * 650 = 130) (h2 : P * 800 / 100 = 320) : P = 40 := 
by { 
  sorry 
}

end find_percentage_l28_28772


namespace range_of_x_l28_28316

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

theorem range_of_x (x : ℝ) (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) 2) :
  f (m * x - 3) + f x < 0 → -3 < x ∧ x < 1 :=
sorry

end range_of_x_l28_28316


namespace cross_section_area_correct_l28_28129

noncomputable def area_of_cross_section (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 11) / 16

theorem cross_section_area_correct (a : ℝ) (h : 0 < a) :
  area_of_cross_section a = (3 * a^2 * Real.sqrt 11) / 16 := by
  sorry

end cross_section_area_correct_l28_28129


namespace average_length_of_strings_l28_28207

theorem average_length_of_strings : 
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3 
  let average_length := total_length / 3
  average_length = 10 / 3 :=
by
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3
  let average_length := total_length / 3
  have h1 : total_length = 10 := by rfl
  have h2 : average_length = 10 / 3 := by rfl
  exact h2

end average_length_of_strings_l28_28207


namespace sqrt_meaningful_range_l28_28541

theorem sqrt_meaningful_range (x : ℝ): x + 2 ≥ 0 ↔ x ≥ -2 := by
  sorry

end sqrt_meaningful_range_l28_28541


namespace find_g_neg_six_l28_28919

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l28_28919


namespace g_inv_zero_solution_l28_28198

noncomputable def g (a b x : ℝ) : ℝ := 1 / (2 * a * x + b)

theorem g_inv_zero_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  g a b (g a b 0) = 0 ↔ g a b 0 = 1 / b :=
by
  sorry

end g_inv_zero_solution_l28_28198


namespace least_xy_value_l28_28160

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l28_28160


namespace harold_wrapping_paper_cost_l28_28321

theorem harold_wrapping_paper_cost :
  let rolls_for_shirt_boxes := 20 / 5
  let rolls_for_xl_boxes := 12 / 3
  let total_rolls := rolls_for_shirt_boxes + rolls_for_xl_boxes
  let cost_per_roll := 4  -- dollars
  (total_rolls * cost_per_roll) = 32 := by
  sorry

end harold_wrapping_paper_cost_l28_28321


namespace Ariel_age_l28_28986

theorem Ariel_age :
  ∀ (fencing_start_year birth_year: ℕ) (fencing_years: ℕ),
    fencing_start_year = 2006 →
    birth_year = 1992 →
    fencing_years = 16 →
    (fencing_start_year + fencing_years - birth_year) = 30 :=
by
  intros fencing_start_year birth_year fencing_years h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Ariel_age_l28_28986


namespace maximum_value_N_27_l28_28388

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l28_28388


namespace four_neg_a_equals_one_ninth_l28_28168

theorem four_neg_a_equals_one_ninth (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 2) : 4 ^ (-a) = 1 / 9 :=
by
  sorry

end four_neg_a_equals_one_ninth_l28_28168


namespace intersection_l28_28521

def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B : Set ℝ := { x | x > -1 }

theorem intersection (x : ℝ) : x ∈ (A ∩ B) ↔ -1 < x ∧ x < 3 := by
  sorry

end intersection_l28_28521


namespace percentage_runs_by_running_l28_28953

theorem percentage_runs_by_running
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (eq_total_runs : total_runs = 120)
  (eq_boundaries : boundaries = 3)
  (eq_sixes : sixes = 8)
  (eq_runs_per_boundary : runs_per_boundary = 4)
  (eq_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100) = 50 :=
by
  sorry

end percentage_runs_by_running_l28_28953


namespace count_odd_expressions_l28_28473

theorem count_odd_expressions : 
  let exp1 := 1^2
  let exp2 := 2^3
  let exp3 := 3^4
  let exp4 := 4^5
  let exp5 := 5^6
  (if exp1 % 2 = 1 then 1 else 0) + 
  (if exp2 % 2 = 1 then 1 else 0) + 
  (if exp3 % 2 = 1 then 1 else 0) + 
  (if exp4 % 2 = 1 then 1 else 0) + 
  (if exp5 % 2 = 1 then 1 else 0) = 3 :=
by 
  sorry

end count_odd_expressions_l28_28473


namespace surface_area_of_cube_l28_28767

theorem surface_area_of_cube (a : ℝ) : 
  let edge_length := 4 * a
  let face_area := edge_length ^ 2
  let total_surface_area := 6 * face_area
  total_surface_area = 96 * a^2 := by
  sorry

end surface_area_of_cube_l28_28767


namespace john_change_proof_l28_28859

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

end john_change_proof_l28_28859


namespace max_rabbits_with_long_ears_and_jumping_far_l28_28407

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l28_28407


namespace find_g_neg_six_l28_28920

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l28_28920


namespace complex_root_product_l28_28180

theorem complex_root_product (w : ℂ) (hw1 : w^3 = 1) (hw2 : w^2 + w + 1 = 0) :
(1 - w + w^2) * (1 + w - w^2) = 4 :=
sorry

end complex_root_product_l28_28180


namespace obtuse_triangle_l28_28192

theorem obtuse_triangle (A B C M E : ℝ) (hM : M = (B + C) / 2) (hE : E > 0) 
(hcond : (B - E) ^ 2 + (C - E) ^ 2 >= 4 * (A - M) ^ 2): 
∃ α β γ, α > 90 ∧ β + γ < 90 ∧ α + β + γ = 180 :=
by
  sorry

end obtuse_triangle_l28_28192


namespace general_term_formula_l28_28314

variable (a S : ℕ → ℚ)

-- Condition 1: The sum of the first n terms of the sequence {a_n} is S_n
def sum_first_n_terms (n : ℕ) : ℚ := S n

-- Condition 2: a_n = 3S_n - 2
def a_n (n : ℕ) : Prop := a n = 3 * S n - 2

theorem general_term_formula (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ k, k ≥ 2 → a (k) = - (1/2) * a (k - 1) ) : 
  a n = (-1/2)^(n-1) :=
sorry

end general_term_formula_l28_28314


namespace eventually_constant_sequence_a_floor_l28_28839

noncomputable def sequence_a (n : ℕ) : ℝ := sorry
noncomputable def sequence_b (n : ℕ) : ℝ := sorry

axiom base_conditions : 
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (∀ n, sequence_a (n + 1) * sequence_b n = 1 + sequence_a n + sequence_a n * sequence_b n) ∧
  (∀ n, sequence_b (n + 1) * sequence_a n = 1 + sequence_b n + sequence_a n * sequence_b n)

theorem eventually_constant_sequence_a_floor:
  (∃ N, ∀ n ≥ N, 4 < sequence_a n ∧ sequence_a n < 5) →
  (∃ N, ∀ n ≥ N, Int.floor (sequence_a n) = 4) :=
sorry

end eventually_constant_sequence_a_floor_l28_28839


namespace max_score_top_three_teams_l28_28334

theorem max_score_top_three_teams : 
  ∀ (teams : Finset String) (points : String → ℕ), 
    teams.card = 6 →
    (∀ team, team ∈ teams → (points team = 0 ∨ points team = 1 ∨ points team = 3)) →
    ∃ top_teams : Finset String, top_teams.card = 3 ∧ 
    (∀ team, team ∈ top_teams → points team = 24) := 
by sorry

end max_score_top_three_teams_l28_28334


namespace smallest_four_digit_divisible_by_53_l28_28024

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28024


namespace distinct_nonzero_digits_sum_l28_28530

theorem distinct_nonzero_digits_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) 
  (h7 : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*a + c + 100*b + 10*c + a + 100*c + 10*a + b + 100*c + 10*b + a = 1776) : 
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 1 ∧ b = 3 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 3) ∨ (a = 1 ∧ b = 5 ∧ c = 2) ∨ (a = 2 ∧ b = 1 ∧ c = 5) ∨
  (a = 2 ∧ b = 5 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 4) ∨ (a = 3 ∧ b = 4 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 3 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 2) ∨ (a = 5 ∧ b = 2 ∧ c = 1) :=
sorry

end distinct_nonzero_digits_sum_l28_28530


namespace decimal_to_fraction_l28_28762

theorem decimal_to_fraction (x : ℚ) (h : x = 3.675) : x = 147 / 40 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l28_28762


namespace find_m_l28_28847

theorem find_m (m : ℝ) (h1 : |m - 3| = 4) (h2 : m - 7 ≠ 0) : m = -1 :=
sorry

end find_m_l28_28847


namespace percentage_deposit_l28_28269

theorem percentage_deposit (deposited : ℝ) (initial_amount : ℝ) (amount_deposited : ℝ) (P : ℝ) 
  (h1 : deposited = 750) 
  (h2 : initial_amount = 50000)
  (h3 : amount_deposited = 0.20 * (P / 100) * (0.25 * initial_amount))
  (h4 : amount_deposited = deposited) : 
  P = 30 := 
sorry

end percentage_deposit_l28_28269


namespace find_sum_l28_28932

-- Defining the conditions of the problem
variables (P r t : ℝ) 
theorem find_sum 
  (h1 : (P * r * t) / 100 = 88) 
  (h2 : (P * r * t) / (100 + (r * t)) = 80) 
  : P = 880 := 
sorry

end find_sum_l28_28932


namespace intersection_point_a_l28_28742

-- Definitions for the given conditions 
def f (x : ℤ) (b : ℤ) : ℤ := 3 * x + b
def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 3 -- Considering that f is invertible for integer b

-- The problem statement
theorem intersection_point_a (a b : ℤ) (h1 : a = f (-3) b) (h2 : a = f_inv (-3)) (h3 : f (-3) b = -3):
  a = -3 := sorry

end intersection_point_a_l28_28742


namespace smallest_n_gt_T1989_2_l28_28841

noncomputable def T (w : ℕ) : ℕ → ℕ
| 1       := w
| (n + 1) := w ^ T n

theorem smallest_n_gt_T1989_2 :
  ∃ n : ℕ, T 3 n > T 2 1989 ∧ ∀ m < n, T 3 m ≤ T 2 1989 :=
  sorry

end smallest_n_gt_T1989_2_l28_28841


namespace time_relationship_l28_28103

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end time_relationship_l28_28103


namespace total_packages_sold_l28_28130

variable (P : ℕ)

/-- An automobile parts supplier charges 25 per package of gaskets. 
    When a customer orders more than 10 packages of gaskets, the supplier charges 4/5 the price for each package in excess of 10.
    During a certain week, the supplier received 1150 in payment for the gaskets. --/
def cost (P : ℕ) : ℕ :=
  if P > 10 then 250 + (P - 10) * 20 else P * 25

theorem total_packages_sold :
  cost P = 1150 → P = 55 := by
  sorry

end total_packages_sold_l28_28130


namespace value_of_expression_l28_28177

-- Definitions for the conditions
variables (a b : ℝ)

-- Theorem statement
theorem value_of_expression : (a - 3 * b = 3) → (a + 2 * b - (2 * a - b)) = -3 :=
by
  intro h
  sorry

end value_of_expression_l28_28177


namespace angelina_journey_equation_l28_28272

theorem angelina_journey_equation (t : ℝ) :
    4 = t + 15/60 + (4 - 15/60 - t) →
    60 * t + 90 * (15/4 - t) = 255 :=
    by
    sorry

end angelina_journey_equation_l28_28272


namespace positive_root_exists_iff_m_eq_neg_one_l28_28539

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end positive_root_exists_iff_m_eq_neg_one_l28_28539


namespace circular_paper_pieces_needed_l28_28503

-- Definition of the problem conditions
def side_length_dm := 10
def side_length_cm := side_length_dm * 10
def perimeter_cm := 4 * side_length_cm
def number_of_sides := 4
def semicircles_per_side := 1
def total_semicircles := number_of_sides * semicircles_per_side
def semicircles_to_circles := 2
def total_circles := total_semicircles / semicircles_to_circles
def paper_pieces_per_circle := 20

-- Main theorem stating the problem and the answer.
theorem circular_paper_pieces_needed : (total_circles * paper_pieces_per_circle) = 40 :=
by sorry

end circular_paper_pieces_needed_l28_28503


namespace find_g_neg_six_l28_28921

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l28_28921


namespace unique_integer_sequence_l28_28727

theorem unique_integer_sequence :
  ∃ a : ℕ → ℤ, a 1 = 1 ∧ a 2 > 1 ∧ ∀ n ≥ 1, (a (n + 1))^3 + 1 = a n * a (n + 2) :=
sorry

end unique_integer_sequence_l28_28727


namespace minimize_f_l28_28443

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l28_28443


namespace smallest_four_digit_divisible_by_53_l28_28021

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28021


namespace intersection_A_compl_B_subset_E_B_l28_28828

namespace MathProof

-- Definitions
def A := {x : ℝ | (x + 3) * (x - 6) ≥ 0}
def B := {x : ℝ | (x + 2) / (x - 14) < 0}
def compl_R_B := {x : ℝ | x ≤ -2 ∨ x ≥ 14}
def E (a : ℝ) := {x : ℝ | 2 * a < x ∧ x < a + 1}

-- Theorem for intersection of A and complement of B
theorem intersection_A_compl_B : A ∩ compl_R_B = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Theorem for subset relationship to determine range of a
theorem subset_E_B (a : ℝ) : (E a ⊆ B) → a ≥ -1 :=
by
  sorry

end MathProof

end intersection_A_compl_B_subset_E_B_l28_28828


namespace simplify_fraction_l28_28372

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l28_28372


namespace find_a7_in_arithmetic_sequence_l28_28190

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem find_a7_in_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3_a5 : a 3 + a 5 = 10) :
  a 7 = 8 :=
sorry

end find_a7_in_arithmetic_sequence_l28_28190


namespace general_term_formula_minimum_T_n_value_min_value_at_one_l28_28526

def arithmetic_sequence (a : ℕ → ℕ) (a₁ d : ℕ) :=
  a 1 = a₁ ∧ ∀ n, a (n+1) = a n + d

noncomputable def a_n : ℕ → ℕ
| 1     := 1
| (n+1) := a_n n + 2

noncomputable def b_n (n : ℕ) : ℚ :=
  1 / ((a_n n : ℚ) * (a_n (n - 1) : ℚ))

noncomputable def T_n (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, b_n (k + 1))

theorem general_term_formula :
  ∀ n, a_n n = 2 * n - 1 :=
by sorry

theorem minimum_T_n_value :
  ∀ n, T_n 1 ≤ T_n n :=
by sorry

theorem min_value_at_one :
  T_n 1 = 1 / 3 :=
by sorry

end general_term_formula_minimum_T_n_value_min_value_at_one_l28_28526


namespace ship_total_distance_l28_28123

variables {v_r : ℝ} {t_total : ℝ} {a d : ℝ}

-- Given conditions
def conditions (v_r t_total a d : ℝ) :=
  v_r = 2 ∧ t_total = 3.2 ∧
  (∃ v : ℝ, ∀ t : ℝ, t = a/(v + v_r) + (a + d)/v + (a + 2*d)/(v - v_r)) 

-- The main statement to prove
theorem ship_total_distance (d_total : ℝ) :
  conditions 2 3.2 a d → d_total = 102 :=
by
  sorry

end ship_total_distance_l28_28123


namespace sequence_solution_l28_28151

theorem sequence_solution :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℝ),
    a 1 = 2 ∧
    (∀ n, b n = (a (n + 1)) / (a n)) ∧
    b 10 * b 11 = 2 →
    a 21 = 2 ^ 11 :=
by
  sorry

end sequence_solution_l28_28151


namespace range_of_a_for_critical_point_l28_28693

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

theorem range_of_a_for_critical_point :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (a - 1) (a + 1), deriv f x = 0) ↔ 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_for_critical_point_l28_28693


namespace number_subtracted_l28_28259

theorem number_subtracted (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 :=
by
  sorry

end number_subtracted_l28_28259


namespace rook_path_exists_l28_28206

theorem rook_path_exists :
  ∃ (path : Finset (Fin 8 × Fin 8)) (s1 s2 : Fin 8 × Fin 8),
  s1 ≠ s2 ∧
  s1.1 % 2 = s2.1 % 2 ∧ s1.2 % 2 = s2.2 % 2 ∧
  ∀ s : Fin 8 × Fin 8, s ∈ path ∧ s ≠ s2 :=
sorry

end rook_path_exists_l28_28206


namespace square_difference_l28_28655

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l28_28655


namespace layers_removed_l28_28617

theorem layers_removed (n : ℕ) (original_volume remaining_volume side_length : ℕ) :
  original_volume = side_length^3 →
  remaining_volume = (side_length - 2 * n)^3 →
  original_volume = 1000 →
  remaining_volume = 512 →
  side_length = 10 →
  n = 1 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end layers_removed_l28_28617


namespace evaluate_expression_l28_28508

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 :=
by sorry

end evaluate_expression_l28_28508


namespace number_of_dogs_l28_28670

theorem number_of_dogs (h1 : 24 = 2 * 2 + 4 * n) : n = 5 :=
by
  sorry

end number_of_dogs_l28_28670


namespace fraction_subtraction_l28_28990

theorem fraction_subtraction :
  (15 / 45) - (1 + (2 / 9)) = - (8 / 9) :=
by
  sorry

end fraction_subtraction_l28_28990


namespace find_g_minus_6_l28_28909

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l28_28909


namespace x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l28_28621

theorem x_eq_1_sufficient_not_necessary_for_x_sq_eq_1 (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ((x^2 = 1) → (x = 1 ∨ x = -1)) :=
by 
  sorry

end x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l28_28621


namespace intersection_with_complement_l28_28866

-- Define the universal set U, sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- The equivalent proof problem in Lean 4
theorem intersection_with_complement :
  A ∩ complement_B = {0, 2} :=
by
  sorry

end intersection_with_complement_l28_28866


namespace min_value_at_3_l28_28462

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l28_28462


namespace weight_left_after_two_deliveries_l28_28261

-- Definitions and conditions
def initial_load : ℝ := 50000
def first_store_percentage : ℝ := 0.1
def second_store_percentage : ℝ := 0.2

-- The statement to be proven
theorem weight_left_after_two_deliveries :
  let weight_after_first_store := initial_load * (1 - first_store_percentage)
  let weight_after_second_store := weight_after_first_store * (1 - second_store_percentage)
  weight_after_second_store = 36000 :=
by sorry  -- Proof omitted

end weight_left_after_two_deliveries_l28_28261


namespace store_profit_l28_28119

variable (C : ℝ)  -- Cost price of a turtleneck sweater

noncomputable def initial_marked_price : ℝ := 1.20 * C
noncomputable def new_year_marked_price : ℝ := 1.25 * initial_marked_price C
noncomputable def discount_amount : ℝ := 0.08 * new_year_marked_price C
noncomputable def final_selling_price : ℝ := new_year_marked_price C - discount_amount C
noncomputable def profit : ℝ := final_selling_price C - C

theorem store_profit (C : ℝ) : profit C = 0.38 * C :=
by
  -- The detailed steps are omitted, as required by the instructions.
  sorry

end store_profit_l28_28119


namespace value_of_m_l28_28703

theorem value_of_m : 
  (2 ^ 1999 - 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 - 2 ^ 1995 = m * 2 ^ 1995) -> m = 5 :=
by 
  sorry

end value_of_m_l28_28703


namespace unique_pairs_pos_int_satisfy_eq_l28_28293

theorem unique_pairs_pos_int_satisfy_eq (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) := 
by
  sorry

end unique_pairs_pos_int_satisfy_eq_l28_28293


namespace cost_price_is_925_l28_28242

-- Definitions for the conditions
def SP : ℝ := 1110
def profit_percentage : ℝ := 0.20

-- Theorem to prove that the cost price is 925
theorem cost_price_is_925 (CP : ℝ) (h : SP = (CP * (1 + profit_percentage))) : CP = 925 := 
by sorry

end cost_price_is_925_l28_28242


namespace smallest_four_digit_divisible_by_53_l28_28057

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28057


namespace smallest_four_digit_divisible_by_53_l28_28031

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28031


namespace ms_cole_total_students_l28_28569

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end ms_cole_total_students_l28_28569


namespace swimming_pool_area_l28_28926

open Nat

-- Define the width (w) and length (l) with given conditions
def width (w : ℕ) : Prop :=
  exists (l : ℕ), l = 2 * w + 40 ∧ 2 * w + 2 * l = 800

-- Define the area of the swimming pool
def pool_area (w l : ℕ) : ℕ :=
  w * l

theorem swimming_pool_area : 
  ∃ (w l : ℕ), width w ∧ width l -> pool_area w l = 33600 :=
by
  sorry

end swimming_pool_area_l28_28926


namespace students_dont_eat_lunch_l28_28783

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end students_dont_eat_lunch_l28_28783


namespace smallest_four_digit_divisible_by_53_l28_28019

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28019


namespace fraction_of_satisfactory_grades_is_3_4_l28_28220

def num_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D" + grades "F"

def satisfactory_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D"

def fraction_satisfactory (grades : String → ℕ) : ℚ := 
  satisfactory_grades grades / num_grades grades

theorem fraction_of_satisfactory_grades_is_3_4 
  (grades : String → ℕ)
  (hA : grades "A" = 5)
  (hB : grades "B" = 4)
  (hC : grades "C" = 3)
  (hD : grades "D" = 3)
  (hF : grades "F" = 5) : 
  fraction_satisfactory grades = (3 : ℚ) / 4 := by
{
  sorry
}

end fraction_of_satisfactory_grades_is_3_4_l28_28220


namespace determine_phi_l28_28694

theorem determine_phi
  (A ω : ℝ) (φ : ℝ) (x : ℝ)
  (hA : 0 < A)
  (hω : 0 < ω)
  (hφ : abs φ < Real.pi / 2)
  (h_symm : ∃ f : ℝ → ℝ, f (-Real.pi / 4) = A ∨ f (-Real.pi / 4) = -A)
  (h_zero : ∃ x₀ : ℝ, A * Real.sin (ω * x₀ + φ) = 0 ∧ abs (x₀ + Real.pi / 4) = Real.pi / 2) :
  φ = -Real.pi / 4 :=
sorry

end determine_phi_l28_28694


namespace smallest_four_digit_divisible_by_53_l28_28045

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28045


namespace find_k_value_l28_28514

theorem find_k_value (k : ℝ) :
  (∃ (x y : ℝ), x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ k = -1/2 := 
by
  sorry

end find_k_value_l28_28514


namespace even_numbers_probability_different_digits_l28_28268

theorem even_numbers_probability_different_digits :
  let even_numbers := list.range' 10 89 |>.filter (λ n, n % 2 = 0),
      numbers_with_same_digits := [22, 44, 66, 88],
      total_even_numbers := even_numbers.length,
      count_same_digits := numbers_with_same_digits.length,
      probability_different_digits := 1 - (count_same_digits / total_even_numbers : ℚ) in
  probability_different_digits = 41 / 45 := by
  sorry

end even_numbers_probability_different_digits_l28_28268


namespace four_digit_numbers_div_by_5_with_34_end_l28_28842

theorem four_digit_numbers_div_by_5_with_34_end : 
  ∃ (count : ℕ), count = 90 ∧
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) →
  (n % 100 = 34) →
  ((10 ∣ n) ∨ (5 ∣ n)) →
  (count = 90) :=
sorry

end four_digit_numbers_div_by_5_with_34_end_l28_28842


namespace class_8_1_total_score_l28_28250

noncomputable def total_score (spirit neatness standard_of_movements : ℝ) 
(weights_spirit weights_neatness weights_standard : ℝ) : ℝ :=
  (spirit * weights_spirit + neatness * weights_neatness + standard_of_movements * weights_standard) / 
  (weights_spirit + weights_neatness + weights_standard)

theorem class_8_1_total_score :
  total_score 8 9 10 2 3 5 = 9.3 :=
by
  sorry

end class_8_1_total_score_l28_28250


namespace ratio_joe_sara_l28_28208

variables (S J : ℕ) (k : ℕ)

-- Conditions
#check J + S = 120
#check J = k * S + 6
#check J = 82

-- The goal is to prove the ratio J / S = 41 / 19
theorem ratio_joe_sara (h1 : J + S = 120) (h2 : J = k * S + 6) (h3 : J = 82) : J / S = 41 / 19 :=
sorry

end ratio_joe_sara_l28_28208


namespace maximum_value_N_27_l28_28389

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l28_28389


namespace figures_can_be_drawn_l28_28266

structure Figure :=
  (degrees : List ℕ) -- List of degrees of the vertices in the graph associated with the figure.

-- Define a predicate to check if a figure can be drawn without lifting the pencil and without retracing
def canBeDrawnWithoutLifting (fig : Figure) : Prop :=
  let odd_degree_vertices := fig.degrees.filter (λ d => d % 2 = 1)
  odd_degree_vertices.length = 0 ∨ odd_degree_vertices.length = 2

-- Define the figures A, B, C, D with their degrees (examples, these should match the problem's context)
def figureA : Figure := { degrees := [2, 2, 2, 2] }
def figureB : Figure := { degrees := [2, 2, 2, 2, 4] }
def figureC : Figure := { degrees := [3, 3, 3, 3] }
def figureD : Figure := { degrees := [4, 4, 2, 2] }

-- State the theorem that figures A, B, and D can be drawn without lifting the pencil
theorem figures_can_be_drawn :
  canBeDrawnWithoutLifting figureA ∧ canBeDrawnWithoutLifting figureB ∧ canBeDrawnWithoutLifting figureD :=
  by sorry -- Proof to be completed

end figures_can_be_drawn_l28_28266


namespace simplify_fraction_l28_28373

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28373


namespace minimize_f_at_3_l28_28448

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l28_28448


namespace sally_cards_final_count_l28_28881

def initial_cards : ℕ := 27
def cards_from_Dan : ℕ := 41
def cards_bought : ℕ := 20
def cards_traded : ℕ := 15
def cards_lost : ℕ := 7

def final_cards (initial : ℕ) (from_Dan : ℕ) (bought : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_Dan + bought - traded - lost

theorem sally_cards_final_count :
  final_cards initial_cards cards_from_Dan cards_bought cards_traded cards_lost = 66 := by
  sorry

end sally_cards_final_count_l28_28881


namespace problem_solution_l28_28379

theorem problem_solution (p q r : ℝ) 
    (h1 : (p * r / (p + q) + q * p / (q + r) + r * q / (r + p)) = -8)
    (h2 : (q * r / (p + q) + r * p / (q + r) + p * q / (r + p)) = 9) 
    : (q / (p + q) + r / (q + r) + p / (r + p) = 10) := 
by
  sorry

end problem_solution_l28_28379


namespace find_second_candy_cost_l28_28784

theorem find_second_candy_cost :
  ∃ (x : ℝ), 
    (15 * 8 + 30 * x = 45 * 6) ∧
    x = 5 := by
  sorry

end find_second_candy_cost_l28_28784


namespace minimize_quadratic_l28_28454

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l28_28454


namespace part1_part2_min_value_l28_28825

variable {R : Type} [LinearOrderedField R]

def quadratic_function (m x : R) : R := x^2 - m * x + m - 1

theorem part1 (h : quadratic_function m 0 = quadratic_function m 2) : m = 2 :=
by
  sorry

theorem part2_min_value (m : R) :
  ∃ min_val, min_val = if m ≤ ⟨-4⟩ then ⟨3 * m + 3⟩
                       else if -4 < m ∧ m < ⟨4⟩ then ⟨-m^2 / 4 + m - 1⟩
                       else ⟨3 - m⟩ :=
by
  sorry

end part1_part2_min_value_l28_28825


namespace smallest_four_digit_divisible_by_53_l28_28077

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28077


namespace find_g_neg_6_l28_28901

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l28_28901


namespace divisor_greater_2016_l28_28849

theorem divisor_greater_2016 (d : ℕ) (h : 2016 / d = 0) : d > 2016 :=
sorry

end divisor_greater_2016_l28_28849


namespace jenny_speed_proof_l28_28552

-- Define the constants and conditions based on the problem
def total_distance (S : ℝ) : Prop :=
  S > 0

def jack_speed_first_half (v1 : ℝ) : Prop :=
  v1 = 4

def jack_speed_second_half (v2 : ℝ) : Prop :=
  v2 = 2

def jack_speed_descending (v3 : ℝ) : Prop :=
  v3 = 3

def jack_meeting_point (S : ℝ) (t1 t2 t3 : ℝ) : Prop :=
  t1 = S / 8 ∧ t2 = S / 4 ∧ t3 = S / 6 ∧ t1 + t2 + t3 = 13 * S / 24

-- Define Jenny's average speed based on Jack's time and distance
def jenny_avg_speed (S t : ℝ) (v_jenny : ℝ) : Prop :=
  v_jenny = (S / 2) / t

theorem jenny_speed_proof
  (S t1 t2 t3 t v_jenny : ℝ)
  (pos_S : total_distance S)
  (js1 : jack_speed_first_half 4)
  (js2 : jack_speed_second_half 2)
  (js3 : jack_speed_descending 3)
  (jmp : jack_meeting_point S t1 t2 t3)
  : jenny_avg_speed S (13 * S / 24) v_jenny → v_jenny = 12 / 13 :=
by
  sorry

end jenny_speed_proof_l28_28552


namespace smallest_four_digit_multiple_of_53_l28_28003

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28003


namespace max_rabbits_with_traits_l28_28402

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l28_28402


namespace am_gm_inequality_l28_28476

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem am_gm_inequality : (a / b) + (b / c) + (c / a) ≥ 3 := by
  sorry

end am_gm_inequality_l28_28476


namespace probability_no_two_green_hats_next_to_each_other_l28_28963

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l28_28963


namespace correct_statement_B_l28_28105

def flowchart_start_points : Nat := 1
def flowchart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

def program_flowchart_start_points : Nat := 1
def program_flowchart_end_points : Nat := 1

def structure_chart_start_points : Nat := 1
def structure_chart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

theorem correct_statement_B :
  (program_flowchart_start_points = 1 ∧ program_flowchart_end_points = 1) :=
by 
  sorry

end correct_statement_B_l28_28105


namespace arithmetic_sequence_sum_l28_28281

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m: ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Sum of the first n terms of a sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Specific statement we want to prove
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a)
  (h_S9 : sum_seq a 9 = 72) :
  a 2 + a 4 + a 9 = 24 :=
sorry

end arithmetic_sequence_sum_l28_28281


namespace prove_a₈_l28_28887

noncomputable def first_term (a : ℕ → ℝ) : Prop := a 1 = 3
noncomputable def arithmetic_b (a b : ℕ → ℝ) : Prop := ∀ n, b n = a (n + 1) - a n
noncomputable def b_conditions (b : ℕ → ℝ) : Prop := b 3 = -2 ∧ b 10 = 12

theorem prove_a₈ (a b : ℕ → ℝ) (h1 : first_term a) (h2 : arithmetic_b a b) (h3 : b_conditions b) :
  a 8 = 3 :=
sorry

end prove_a₈_l28_28887


namespace proof_problem_l28_28282

-- Define the function f(x) = -x - x^3
def f (x : ℝ) : ℝ := -x - x^3

-- Define the main theorem according to the conditions and the required proofs.
theorem proof_problem (x1 x2 : ℝ) (h : x1 + x2 ≤ 0) :
  (f x1) * (f (-x1)) ≤ 0 ∧ (f x1 + f x2) ≥ (f (-x1) + f (-x2)) :=
by
  sorry

end proof_problem_l28_28282


namespace smallest_four_digit_multiple_of_53_l28_28014

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28014


namespace num_three_digit_numbers_l28_28573

theorem num_three_digit_numbers : 
  ∃ count : ℕ, count = 36 ∧ 
    (count = (Nat.choose 2 1) * (Nat.choose 3 2) * (Nat.factorial 3)) :=
by
  have even_choices : ℕ := Nat.choose 2 1
  have odd_choices : ℕ := Nat.choose 3 2
  have arrangements : ℕ := Nat.factorial 3
  let count := even_choices * odd_choices * arrangements
  use count
  split
  { refl }
  { simp [even_choices, odd_choices, arrangements]; 
    sorry }

end num_three_digit_numbers_l28_28573


namespace probability_no_adjacent_green_hats_l28_28960

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l28_28960


namespace max_tension_of_pendulum_l28_28256

theorem max_tension_of_pendulum 
  (m g L θ₀ : ℝ) 
  (h₀ : θ₀ < π / 2) 
  (T₀ : ℝ) 
  (no_air_resistance : true) 
  (no_friction : true) : 
  ∃ T_max, T_max = m * g * (3 - 2 * Real.cos θ₀) := 
by 
  sorry

end max_tension_of_pendulum_l28_28256


namespace minimize_quadratic_function_l28_28459

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l28_28459


namespace crayons_total_l28_28943

theorem crayons_total (Wanda Dina Jacob: ℕ) (hW: Wanda = 62) (hD: Dina = 28) (hJ: Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  sorry

end crayons_total_l28_28943


namespace find_g_neg6_l28_28894

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l28_28894


namespace small_cubes_one_face_painted_red_l28_28785

-- Definitions
def is_red_painted (cube : ℕ) : Bool := true -- representing the condition that the cube is painted red
def side_length (cube : ℕ) : ℕ := 4 -- side length of the original cube is 4 cm
def smaller_cube_side_length : ℕ := 1 -- smaller cube side length is 1 cm

-- Theorem Statement
theorem small_cubes_one_face_painted_red :
  ∀ (large_cube : ℕ), (side_length large_cube = 4) ∧ is_red_painted large_cube → 
  (∃ (number_of_cubes : ℕ), number_of_cubes = 24) :=
by
  sorry

end small_cubes_one_face_painted_red_l28_28785


namespace g_neg_six_eq_neg_twenty_l28_28912

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l28_28912


namespace smallest_four_digit_divisible_by_53_l28_28066

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28066


namespace age_ratios_l28_28958

variable (A B : ℕ)

-- Given conditions
theorem age_ratios :
  (A / B = 2 / 1) → (A - 4 = B + 4) → ((A + 4) / (B - 4) = 5 / 1) :=
by
  intro h1 h2
  sorry

end age_ratios_l28_28958


namespace rug_inner_rectangle_length_l28_28122

theorem rug_inner_rectangle_length
  (width : ℕ)
  (shaded1_width : ℕ)
  (shaded2_width : ℕ)
  (areas_in_ap : ℕ → ℕ → ℕ → Prop)
  (h1 : width = 2)
  (h2 : shaded1_width = 2)
  (h3 : shaded2_width = 2)
  (h4 : ∀ y a1 a2 a3, 
        a1 = 2 * y →
        a2 = 6 * (y + 4) →
        a3 = 10 * (y + 8) →
        areas_in_ap a1 (a2 - a1) (a3 - a2) →
        (a2 - a1 = a3 - a2)) :
  ∃ y, y = 4 :=
by
  sorry

end rug_inner_rectangle_length_l28_28122


namespace max_x_value_l28_28348

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : xy + xz + yz = 8) : 
  x ≤ 7 / 3 :=
sorry

end max_x_value_l28_28348


namespace investment_ratio_l28_28794

-- Define the investments
def A_investment (x : ℝ) : ℝ := 3 * x
def B_investment (x : ℝ) : ℝ := x
def C_investment (y : ℝ) : ℝ := y

-- Define the total profit and B's share of the profit
def total_profit : ℝ := 4400
def B_share : ℝ := 800

-- Define the ratio condition B's share based on investments
def B_share_cond (x y : ℝ) : Prop := (B_investment x / (A_investment x + B_investment x + C_investment y)) * total_profit = B_share

-- Define what we need to prove
theorem investment_ratio (x y : ℝ) (h : B_share_cond x y) : x / y = 2 / 3 :=
by 
  sorry

end investment_ratio_l28_28794


namespace part_a_part_b_l28_28561

/- Part (a) -/
theorem part_a (a b c d : ℝ) (h1 : (a + b ≠ c + d)) (h2 : (a + c ≠ b + d)) (h3 : (a + d ≠ b + c)) :
  ∃ (spheres : ℕ), spheres = 8 := sorry

/- Part (b) -/
theorem part_b (a b c d : ℝ) (h : (a + b = c + d) ∨ (a + c = b + d) ∨ (a + d = b + c)) :
  ∃ (spheres : ℕ), ∀ (n : ℕ), n > 0 → spheres = n := sorry

end part_a_part_b_l28_28561


namespace find_number_is_9_l28_28325

noncomputable def number (y : ℕ) : ℕ := 3^(12 / y)

theorem find_number_is_9 (y : ℕ) (h_y : y = 6) (h_eq : (number y)^y = 3^12) : number y = 9 :=
by
  sorry

end find_number_is_9_l28_28325


namespace probability_of_successful_meeting_l28_28247

noncomputable def successful_meeting_probability : ℝ :=
  let volume_hypercube := 16.0
  let volume_pyramid := (1.0/3.0) * 2.0^3 * 2.0
  let volume_reduced_base := volume_pyramid / 4.0
  let successful_meeting_volume := volume_reduced_base
  successful_meeting_volume / volume_hypercube

theorem probability_of_successful_meeting : successful_meeting_probability = 1 / 12 :=
  sorry

end probability_of_successful_meeting_l28_28247


namespace expected_ties_after_10_l28_28606

def binom: ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom n (k+1)

noncomputable def expected_ties : ℕ → ℝ 
| 0 => 0
| n+1 => expected_ties n + (binom (2*(n+1)) (n+1) / 2^(2*(n+1)))

theorem expected_ties_after_10 : expected_ties 5 = 1.707 := 
by 
  -- Placeholder for the actual proof
  sorry

end expected_ties_after_10_l28_28606


namespace max_rabbits_with_long_ears_and_jumping_far_l28_28409

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l28_28409


namespace Problem_l28_28840

def f (x : ℕ) : ℕ := x ^ 2 + 1
def g (x : ℕ) : ℕ := 2 * x - 1

theorem Problem : f (g (3 + 1)) = 50 := by
  sorry

end Problem_l28_28840


namespace number_of_dogs_l28_28669

-- Define the number of legs humans have
def human_legs : ℕ := 2

-- Define the total number of legs/paws in the pool
def total_legs_paws : ℕ := 24

-- Define the number of paws per dog
def paws_per_dog : ℕ := 4

-- Prove that the number of dogs is 5
theorem number_of_dogs : ∃ (dogs : ℕ), (2 * human_legs) + (dogs * paws_per_dog) = total_legs_paws ∧ dogs = 5 :=
by
  use 5
  split
  sorry

end number_of_dogs_l28_28669


namespace sample_size_l28_28252

theorem sample_size {n : ℕ} (h_ratio : 2+3+4 = 9)
  (h_units_A : ∃ a : ℕ, a = 16)
  (h_stratified_sampling : ∃ B C : ℕ, B = 24 ∧ C = 32)
  : n = 16 + 24 + 32 := by
  sorry

end sample_size_l28_28252


namespace tan_beta_formula_l28_28822

theorem tan_beta_formula (α β : ℝ) 
  (h1 : Real.tan α = -2/3)
  (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 7/4 :=
sorry

end tan_beta_formula_l28_28822


namespace book_pages_l28_28133

theorem book_pages (P : ℝ) (h1 : P / 2 + 0.15 * (P / 2) + 210 = P) : P = 600 := 
sorry

end book_pages_l28_28133


namespace solution_set_correct_l28_28691

theorem solution_set_correct (a b c : ℝ) (h : a < 0) (h1 : ∀ x, (ax^2 + bx + c < 0) ↔ ((x < 1) ∨ (x > 3))) :
  ∀ x, (cx^2 + bx + a > 0) ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end solution_set_correct_l28_28691


namespace smallest_four_digit_divisible_by_53_l28_28030

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28030


namespace rectangular_solid_diagonal_l28_28218

theorem rectangular_solid_diagonal (p q r : ℝ) (d : ℝ) :
  p^2 + q^2 + r^2 = d^2 :=
sorry

end rectangular_solid_diagonal_l28_28218


namespace average_mark_of_excluded_students_l28_28381

noncomputable def average_mark_excluded (A : ℝ) (N : ℕ) (R : ℝ) (excluded_count : ℕ) (remaining_count : ℕ) : ℝ :=
  ((N : ℝ) * A - (remaining_count : ℝ) * R) / (excluded_count : ℝ)

theorem average_mark_of_excluded_students : 
  average_mark_excluded 70 10 90 5 5 = 50 := 
by 
  sorry

end average_mark_of_excluded_students_l28_28381


namespace smallest_four_digit_div_by_53_l28_28054

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28054


namespace smallest_four_digit_divisible_by_53_l28_28069

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28069


namespace scientific_notation_of_probe_unit_area_l28_28760

def probe_unit_area : ℝ := 0.0000064

theorem scientific_notation_of_probe_unit_area :
  ∃ (mantissa : ℝ) (exponent : ℤ), probe_unit_area = mantissa * 10^exponent ∧ mantissa = 6.4 ∧ exponent = -6 :=
by
  sorry

end scientific_notation_of_probe_unit_area_l28_28760


namespace smallest_four_digit_multiple_of_53_l28_28005

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28005


namespace compute_diff_squares_l28_28662

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l28_28662


namespace simplify_fraction_l28_28369

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l28_28369


namespace find_sum_invested_l28_28482

theorem find_sum_invested (P : ℝ) 
  (SI_1: ℝ) (SI_2: ℝ)
  (h1 : SI_1 = P * (15 / 100) * 2)
  (h2 : SI_2 = P * (12 / 100) * 2)
  (h3 : SI_1 - SI_2 = 900) :
  P = 15000 := by
sorry

end find_sum_invested_l28_28482


namespace integer_sided_triangle_with_60_degree_angle_exists_l28_28798

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end integer_sided_triangle_with_60_degree_angle_exists_l28_28798


namespace nat_divisible_by_five_l28_28941

theorem nat_divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  have h₀ : ¬ ((5 ∣ a) ∨ (5 ∣ b)) → ¬ (5 ∣ (a * b)) := sorry
  -- Proof by contradiction steps go here
  sorry

end nat_divisible_by_five_l28_28941


namespace inverse_function_l28_28241

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

noncomputable def f_inv (y : ℝ) : ℝ := -2 - Real.sqrt ((1 + y) / 2)

theorem inverse_function :
  ∀ (x : ℝ), x < -2 → f_inv (f x) = x ∧ ∀ (y : ℝ), y > -1 → f (f_inv y) = y :=
by
  sorry

end inverse_function_l28_28241


namespace complex_product_l28_28970

theorem complex_product (i : ℂ) (h : i^2 = -1) :
  (3 - 4 * i) * (2 + 7 * i) = 34 + 13 * i :=
sorry

end complex_product_l28_28970


namespace smallest_four_digit_divisible_by_53_l28_28092

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28092


namespace find_f_neg2003_l28_28562

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg2003 (f_defined : ∀ x : ℝ, ∃ y : ℝ, f y = x → f y ≠ 0)
  (cond1 : ∀ ⦃x y w : ℝ⦄, x > y → (f x + x ≥ w → w ≥ f y + y → ∃ z, y ≤ z ∧ z ≤ x ∧ f z = w - z))
  (cond2 : ∃ u : ℝ, f u = 0 ∧ ∀ v : ℝ, f v = 0 → u ≤ v)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 :=
sorry

end find_f_neg2003_l28_28562


namespace car_travel_distance_l28_28771

noncomputable def distance_traveled (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  let pi := Real.pi
  let circumference := pi * diameter
  circumference * revolutions / 12 / 5280

theorem car_travel_distance
  (diameter : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 13)
  (h_revolutions : revolutions = 775.5724667489372) :
  distance_traveled diameter revolutions = 0.5 :=
by
  simp [distance_traveled, h_diameter, h_revolutions, Real.pi]
  sorry

end car_travel_distance_l28_28771


namespace simplify_fraction_l28_28370

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l28_28370


namespace max_rabbits_l28_28416

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l28_28416


namespace variance_of_data_l28_28644

noncomputable def data : List ℝ := [10, 6, 8, 5, 6]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length : ℝ)

noncomputable def variance (l : List ℝ) : ℝ :=
  (l.map (λ x, (x - mean l)^2)).sum / (l.length : ℝ)

theorem variance_of_data : variance data = 3.2 := 
  by
  sorry

end variance_of_data_l28_28644


namespace square_1023_l28_28995

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l28_28995


namespace probability_no_two_green_hats_next_to_each_other_l28_28962

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l28_28962


namespace cylindrical_container_depth_l28_28118

theorem cylindrical_container_depth :
    ∀ (L D A : ℝ), 
      L = 12 ∧ D = 8 ∧ A = 48 → (∃ h : ℝ, h = 4 - 2 * Real.sqrt 3) :=
by
  intros L D A h_cond
  obtain ⟨hL, hD, hA⟩ := h_cond
  sorry

end cylindrical_container_depth_l28_28118


namespace g_minus_6_eq_neg_20_l28_28893

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l28_28893


namespace probability_no_adjacent_green_hats_l28_28961

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l28_28961


namespace sum_num_den_252_l28_28111

theorem sum_num_den_252 (h : (252 : ℤ) / 100 = (63 : ℤ) / 25) : 63 + 25 = 88 :=
by
  sorry

end sum_num_den_252_l28_28111


namespace total_boxes_packed_l28_28779

section
variable (initial_boxes : ℕ) (cost_per_box : ℕ) (donation_multiplier : ℕ)
variable (donor_donation : ℕ) (additional_boxes : ℕ) (total_boxes : ℕ)

-- Given conditions
def initial_boxes := 400
def cost_per_box := 80 + 165  -- 245
def donation_multiplier := 4

def initial_expenditure : ℕ := initial_boxes * cost_per_box
def donor_donation : ℕ := initial_expenditure * donation_multiplier
def additional_boxes : ℕ := donor_donation / cost_per_box
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Proof statement
theorem total_boxes_packed : total_boxes = 2000 :=
by
  unfold initial_boxes cost_per_box donation_multiplier initial_expenditure donor_donation additional_boxes total_boxes
  simp
  sorry  -- Since the proof is not required
end

end total_boxes_packed_l28_28779


namespace probability_total_greater_than_7_l28_28956

open ProbabilityTheory

-- Define a throwing of two dice and compute the probability of getting a total > 7
def num_faces : ℕ := 6
def total_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ :=
  have roll_results : List (ℕ × ℕ) := [(3, 5), (3, 6),
                                        (4, 4), (4, 5), (4, 6),
                                        (5, 3), (5, 4), (5, 5), (5, 6),
                                        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
  roll_results.length

theorem probability_total_greater_than_7 :
  (favorable_outcomes / total_outcomes : ℚ) = 7 / 18 := by
  -- Here you will provide the proof steps
  sorry

end probability_total_greater_than_7_l28_28956


namespace circle_equation_l28_28511

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let radius : ℝ := dist center point
  dist center point = 1 → 
  (x - 1)^2 + y^2 = radius^2 :=
by
  intros
  sorry

end circle_equation_l28_28511


namespace linear_substitution_correct_l28_28818

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end linear_substitution_correct_l28_28818


namespace alloy_problem_l28_28797

theorem alloy_problem (x y : ℝ) 
  (h1 : x + y = 1000) 
  (h2 : 0.25 * x + 0.50 * y = 450) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) :
  x = 200 ∧ y = 800 := 
sorry

end alloy_problem_l28_28797


namespace find_g_neg6_l28_28896

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l28_28896


namespace simplify_and_evaluate_expression_l28_28209

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -6) : 
  (1 - a / (a - 3)) / ((a^2 + 3 * a) / (a^2 - 9)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l28_28209


namespace find_g_minus_6_l28_28911

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l28_28911


namespace hyperbola_focal_length_l28_28139

theorem hyperbola_focal_length :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  2 * c = 2 * Real.sqrt 7 := 
by
  sorry

end hyperbola_focal_length_l28_28139


namespace find_principal_l28_28512

theorem find_principal (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (h1 : A = 1456) (h2 : R = 0.05) (h3 : T = 2.4) :
  A = P + P * R * T → P = 1300 :=
by {
  sorry
}

end find_principal_l28_28512


namespace goose_eggs_calculation_l28_28479

theorem goose_eggs_calculation (E : ℝ) (hatch_fraction : ℝ) (survived_first_month_fraction : ℝ) 
(survived_first_year_fraction : ℝ) (survived_first_year : ℝ) (no_more_than_one_per_egg : Prop) 
(h_hatch : hatch_fraction = 1/3) 
(h_month_survival : survived_first_month_fraction = 3/4)
(h_year_survival : survived_first_year_fraction = 2/5)
(h_survived120 : survived_first_year = 120)
(h_no_more_than_one : no_more_than_one_per_egg) :
  E = 1200 :=
by
  -- Convert the information from conditions to formulate the equation
  sorry


end goose_eggs_calculation_l28_28479


namespace sequence_term_25_l28_28711

theorem sequence_term_25 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → a n = (a (n - 1) + a (n + 1)) / 4)
  (h2 : a 1 = 1)
  (h3 : a 9 = 40545) : 
  a 25 = 57424611447841 := 
sorry

end sequence_term_25_l28_28711


namespace find_g_minus_6_l28_28907

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l28_28907


namespace find_remainder_when_q_divided_by_x_plus_2_l28_28756

noncomputable def q (x : ℝ) (D E F : ℝ) := D * x^4 + E * x^2 + F * x + 5

theorem find_remainder_when_q_divided_by_x_plus_2 (D E F : ℝ) :
  q 2 D E F = 15 → q (-2) D E F = 15 :=
by
  intro h
  sorry

end find_remainder_when_q_divided_by_x_plus_2_l28_28756


namespace find_n_l28_28294

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n % 9 = 4897 % 9 ∧ n = 1 :=
by
  use 1
  sorry

end find_n_l28_28294


namespace train_length_correct_l28_28792

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms - speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_correct :
  length_of_first_train 72 36 69.99440044796417 300 = 399.9440044796417 :=
by
  sorry

end train_length_correct_l28_28792


namespace james_carrot_sticks_l28_28855

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end james_carrot_sticks_l28_28855


namespace mean_of_other_four_l28_28886

theorem mean_of_other_four (a b c d e : ℕ) (h_mean : (a + b + c + d + e + 90) / 6 = 75)
  (h_max : max a (max b (max c (max d (max e 90)))) = 90)
  (h_twice : b = 2 * a) :
  (a + c + d + e) / 4 = 60 :=
by
  sorry

end mean_of_other_four_l28_28886


namespace positive_root_exists_iff_m_eq_neg_one_l28_28540

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end positive_root_exists_iff_m_eq_neg_one_l28_28540


namespace find_g_neg6_l28_28898

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l28_28898


namespace kids_at_camp_l28_28135

theorem kids_at_camp (total_stayed_home : ℕ) (difference : ℕ) (x : ℕ) 
  (h1 : total_stayed_home = 777622) 
  (h2 : difference = 574664) 
  (h3 : total_stayed_home = x + difference) : 
  x = 202958 :=
by
  sorry

end kids_at_camp_l28_28135


namespace andrew_purchased_mangoes_l28_28502

theorem andrew_purchased_mangoes
  (m : Nat)
  (h1 : 14 * 54 = 756)
  (h2 : 756 + 62 * m = 1376) :
  m = 10 :=
by
  sorry

end andrew_purchased_mangoes_l28_28502


namespace probability_of_three_faces_painted_l28_28485

def total_cubes : Nat := 27
def corner_cubes_painted (total : Nat) : Nat := 8
def probability_of_corner_cube (corner : Nat) (total : Nat) : Rat := corner / total

theorem probability_of_three_faces_painted :
    probability_of_corner_cube (corner_cubes_painted total_cubes) total_cubes = 8 / 27 := 
by 
  sorry

end probability_of_three_faces_painted_l28_28485


namespace smallest_four_digit_divisible_by_53_l28_28055

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28055


namespace find_larger_integer_l28_28226

noncomputable def larger_integer (a b : ℤ) := max a b

theorem find_larger_integer (a b : ℕ) 
  (h1 : a/b = 7/3) 
  (h2 : a * b = 294): 
  larger_integer a b = 7 * Real.sqrt 14 :=
by
  -- Proof goes here
  sorry

end find_larger_integer_l28_28226


namespace maximum_positive_factors_l28_28747

theorem maximum_positive_factors (b n : ℕ) (hb : 0 < b ∧ b ≤ 20) (hn : 0 < n ∧ n ≤ 15) :
  ∃ k, (k = b^n) ∧ (∀ m, m = b^n → m.factors.count ≤ 61) :=
sorry

end maximum_positive_factors_l28_28747


namespace smallest_four_digit_divisible_by_53_l28_28028

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28028


namespace Bruce_initial_eggs_l28_28276

variable (B : ℕ)

theorem Bruce_initial_eggs (h : B - 70 = 5) : B = 75 := by
  sorry

end Bruce_initial_eggs_l28_28276


namespace solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l28_28529

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Proof Problem 1 Statement:
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} :=
sorry

-- Proof Problem 2 Statement:
theorem range_of_a_for_f_geq_abs_a_minus_4 (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ -1 ≤ a ∧ a ≤ 9 :=
sorry

end solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l28_28529


namespace smallest_four_digit_multiple_of_53_l28_28007

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28007


namespace time_to_fill_tank_with_leak_l28_28725

-- Definitions based on the given conditions:
def rate_of_pipe_A := 1 / 6 -- Pipe A fills the tank in 6 hours
def rate_of_leak := 1 / 12 -- The leak empties the tank in 12 hours
def combined_rate := rate_of_pipe_A - rate_of_leak -- Combined rate with leak

-- The proof problem: Prove the time taken to fill the tank with the leak present is 12 hours.
theorem time_to_fill_tank_with_leak : 
  (1 / combined_rate) = 12 := by
    -- Proof goes here...
    sorry

end time_to_fill_tank_with_leak_l28_28725


namespace symmetric_line_eq_l28_28622

theorem symmetric_line_eq (x y : ℝ) (h : 3 * x + 4 * y + 5 = 0) : 3 * x - 4 * y + 5 = 0 :=
sorry

end symmetric_line_eq_l28_28622


namespace rhombus_perimeter_l28_28737

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (h3 : d1 / 2 ≠ 0) (h4 : d2 / 2 ≠ 0) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  4 * s = 52 :=
by
  sorry

end rhombus_perimeter_l28_28737


namespace fred_grew_38_cantelopes_l28_28516

def total_cantelopes : Nat := 82
def tim_cantelopes : Nat := 44
def fred_cantelopes : Nat := total_cantelopes - tim_cantelopes

theorem fred_grew_38_cantelopes : fred_cantelopes = 38 :=
by
  sorry

end fred_grew_38_cantelopes_l28_28516


namespace smallest_four_digit_div_by_53_l28_28048

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28048


namespace pi_over_2_irrational_l28_28759

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem pi_over_2_irrational : is_irrational (Real.pi / 2) :=
by sorry

end pi_over_2_irrational_l28_28759


namespace remainder_of_3a_minus_b_divided_by_5_l28_28560

theorem remainder_of_3a_minus_b_divided_by_5 (a b : ℕ) (m n : ℤ) 
(h1 : 3 * a > b) 
(h2 : a = 5 * m + 1) 
(h3 : b = 5 * n + 4) : 
(3 * a - b) % 5 = 4 := 
sorry

end remainder_of_3a_minus_b_divided_by_5_l28_28560


namespace james_carrot_sticks_l28_28856

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end james_carrot_sticks_l28_28856


namespace misha_card_numbers_l28_28113

-- Define the context for digits
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Define conditions
def proper_fraction (a b : ℕ) : Prop := is_digit a ∧ is_digit b ∧ a < b

-- Original problem statement rewritten for Lean
theorem misha_card_numbers (L O M N S B : ℕ) :
  is_digit L → is_digit O → is_digit M → is_digit N → is_digit S → is_digit B →
  proper_fraction O M → proper_fraction O S →
  L + O / M + O + N + O / S = 10 + B :=
sorry

end misha_card_numbers_l28_28113


namespace square_difference_l28_28657

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l28_28657


namespace prob_heads_even_correct_l28_28985

noncomputable def prob_heads_even (n : Nat) : ℝ :=
  if n = 0 then 1
  else (2 / 3) - (1 / 3) * prob_heads_even (n - 1)

theorem prob_heads_even_correct : 
  prob_heads_even 50 = (1 / 2) * (1 + (1 / 3 ^ 50)) :=
sorry

end prob_heads_even_correct_l28_28985


namespace sector_central_angle_l28_28527

noncomputable def sector_angle (R L : ℝ) : ℝ := L / R

theorem sector_central_angle :
  ∃ R L : ℝ, 
    (R > 0) ∧ 
    (L > 0) ∧ 
    (1 / 2 * L * R = 5) ∧ 
    (2 * R + L = 9) ∧ 
    (sector_angle R L = 8 / 5 ∨ sector_angle R L = 5 / 2) :=
sorry

end sector_central_angle_l28_28527


namespace sum_of_cubes_correct_l28_28688

noncomputable def expression_for_sum_of_cubes (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) : Prop :=
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + b^3 * d^3) / (a * b * c * d)

theorem sum_of_cubes_correct (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) :
  expression_for_sum_of_cubes x y z w a b c d hx hy hz hw ha hb hc hd hxy hxz hyz hxw :=
sorry

end sum_of_cubes_correct_l28_28688


namespace convex_polygon_partition_l28_28717

open Set

-- Define a convex polygon and let it contain the circles
variables {Poly : Set (ℝ × ℝ)} (Poly_convex : IsConvex Poly)
variables {Circles : Fin (n : ℕ) → Circle} (Circles_disjoint : ∀(i j : Fin n), i ≠ j → ((Circles i).set ∩ (Circles j).set) = ∅)
variables {Circles_radii_distinct : ∀ (i j : Fin n), i ≠ j → (Circles i).radius ≠ (Circles j).radius}

theorem convex_polygon_partition :
  ∃ (parts : Fin n → Set (ℝ × ℝ)),
    (∀ i, IsConvex (parts i)) ∧ 
    (∀ i, (Circles i).set ⊆ parts i) ∧ 
    (⋃ i, parts i) ⊆ Poly ∧
    (∀ i j, i ≠ j → Disjoint (parts i) (parts j)) :=
  sorry

end convex_polygon_partition_l28_28717


namespace picture_area_l28_28107

-- Given dimensions of the paper
def paper_width : ℝ := 8.5
def paper_length : ℝ := 10

-- Given margins
def margin : ℝ := 1.5

-- Calculated dimensions of the picture
def picture_width := paper_width - 2 * margin
def picture_length := paper_length - 2 * margin

-- Statement to prove
theorem picture_area : picture_width * picture_length = 38.5 := by
  -- skipped the proof
  sorry

end picture_area_l28_28107


namespace least_positive_integer_satisfying_congruences_l28_28948

theorem least_positive_integer_satisfying_congruences :
  ∃ b : ℕ, b > 0 ∧
    (b % 6 = 5) ∧
    (b % 7 = 6) ∧
    (b % 8 = 7) ∧
    (b % 9 = 8) ∧
    ∀ n : ℕ, (n > 0 → (n % 6 = 5) ∧ (n % 7 = 6) ∧ (n % 8 = 7) ∧ (n % 9 = 8) → n ≥ b) ∧
    b = 503 :=
by
  sorry

end least_positive_integer_satisfying_congruences_l28_28948


namespace smallest_four_digit_divisible_by_53_l28_28082

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28082


namespace sum_smallest_largest_consecutive_even_integers_l28_28582

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l28_28582


namespace expression_zero_denominator_nonzero_l28_28755

theorem expression_zero (x : ℝ) : 
  (2 * x - 6) = 0 ↔ x = 3 :=
by {
  sorry
  }

theorem denominator_nonzero (x : ℝ) : 
  x = 3 → (5 * x + 10) ≠ 0 :=
by {
  sorry
  }

end expression_zero_denominator_nonzero_l28_28755


namespace Gerald_initial_notebooks_l28_28553

variable (J G : ℕ)

theorem Gerald_initial_notebooks (h1 : J = G + 13)
    (h2 : J - 5 - 6 = 10) :
    G = 8 :=
sorry

end Gerald_initial_notebooks_l28_28553


namespace fff1_eq_17_l28_28134

def f (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 6 then 3 * n + 2
  else 2 * n - 1

theorem fff1_eq_17 : f (f (f 1)) = 17 :=
  by sorry

end fff1_eq_17_l28_28134


namespace number_zero_points_eq_three_l28_28285

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - x^2

theorem number_zero_points_eq_three : ∃ x1 x2 x3 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (∀ y : ℝ, f y = 0 → (y = x1 ∨ y = x2 ∨ y = x3)) :=
sorry

end number_zero_points_eq_three_l28_28285


namespace factor_polynomial_l28_28810

def A (x : ℝ) : ℝ := x^2 + 5 * x + 3
def B (x : ℝ) : ℝ := x^2 + 9 * x + 20
def C (x : ℝ) : ℝ := x^2 + 7 * x - 8

theorem factor_polynomial (x : ℝ) :
  (A x) * (B x) + (C x) = (x^2 + 7 * x + 8) * (x^2 + 7 * x + 14) :=
by
  sorry

end factor_polynomial_l28_28810


namespace rooms_per_floor_l28_28248

-- Definitions for each of the conditions
def numberOfFloors : ℕ := 4
def hoursPerRoom : ℕ := 6
def hourlyRate : ℕ := 15
def totalEarnings : ℕ := 3600

-- Statement of the problem
theorem rooms_per_floor : 
  (totalEarnings / hourlyRate) / hoursPerRoom / numberOfFloors = 10 := 
  sorry

end rooms_per_floor_l28_28248


namespace simplify_expression_l28_28728

theorem simplify_expression (n : ℕ) (hn : 0 < n) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4) - 6) = 80 / 81 :=
by
  sorry

end simplify_expression_l28_28728


namespace remaining_calories_proof_l28_28359

def volume_of_rectangular_block (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_cube (side : ℝ) : ℝ :=
  side * side * side

def remaining_volume (initial_volume eaten_volume : ℝ) : ℝ :=
  initial_volume - eaten_volume

def remaining_calories (remaining_volume calorie_density : ℝ) : ℝ :=
  remaining_volume * calorie_density

theorem remaining_calories_proof :
  let calorie_density := 110
  let original_length := 4
  let original_width := 8
  let original_height := 2
  let cube_side := 2
  let original_volume := volume_of_rectangular_block original_length original_width original_height
  let eaten_volume := volume_of_cube cube_side
  let remaining_vol := remaining_volume original_volume eaten_volume
  let resulting_calories := remaining_calories remaining_vol calorie_density
  resulting_calories = 6160 := by
  repeat { sorry }

end remaining_calories_proof_l28_28359


namespace mike_initial_games_l28_28205

theorem mike_initial_games (v w: ℕ)
  (h_non_working : v - w = 8)
  (h_earnings : 7 * w = 56)
  : v = 16 :=
by
  sorry

end mike_initial_games_l28_28205


namespace fraction_expression_l28_28683

theorem fraction_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3 / 8 := by
  sorry

end fraction_expression_l28_28683


namespace opposite_of_neg_two_thirds_l28_28596

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l28_28596


namespace remainder_of_power_mod_five_l28_28227

theorem remainder_of_power_mod_five : (4 ^ 11) % 5 = 4 :=
by
  sorry

end remainder_of_power_mod_five_l28_28227


namespace smallest_four_digit_multiple_of_53_l28_28012

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28012


namespace exists_number_divisible_by_5_pow_1000_with_no_zeros_l28_28877

theorem exists_number_divisible_by_5_pow_1000_with_no_zeros :
  ∃ n : ℕ, (5 ^ 1000 ∣ n) ∧ (∀ d ∈ n.digits 10, d ≠ 0) := 
sorry

end exists_number_divisible_by_5_pow_1000_with_no_zeros_l28_28877


namespace g_minus_6_eq_neg_20_l28_28889

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l28_28889


namespace units_digit_of_fraction_l28_28231

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end units_digit_of_fraction_l28_28231


namespace kurt_savings_l28_28557

def daily_cost_old : ℝ := 0.85
def daily_cost_new : ℝ := 0.45
def days : ℕ := 30

theorem kurt_savings : (daily_cost_old * days) - (daily_cost_new * days) = 12.00 := by
  sorry

end kurt_savings_l28_28557


namespace minimize_quadratic_l28_28469

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l28_28469


namespace eight_disks_area_sum_final_result_l28_28675

theorem eight_disks_area_sum (r : ℝ) (C : ℝ) :
  C = 1 ∧ r = (Real.sqrt 2 + 1) / 2 → 
  8 * (π * (r ^ 2)) = 2 * π * (3 + 2 * Real.sqrt 2) :=
by
  intros h
  sorry

theorem final_result :
  let a := 6
  let b := 4
  let c := 2
  a + b + c = 12 :=
by
  intros
  norm_num

end eight_disks_area_sum_final_result_l28_28675


namespace notebook_cost_l28_28184

theorem notebook_cost (s n c : ℕ) (h1 : s > 20) (h2 : n > 2) (h3 : c > 2 * n) (h4 : s * c * n = 4515) : c = 35 :=
sorry

end notebook_cost_l28_28184


namespace greatest_common_factor_36_45_l28_28947

theorem greatest_common_factor_36_45 : 
  ∃ g, g = (gcd 36 45) ∧ g = 9 :=
by {
  sorry
}

end greatest_common_factor_36_45_l28_28947


namespace parabola_equation_l28_28138

theorem parabola_equation 
  (vertex_x vertex_y : ℝ)
  (a b c : ℝ)
  (h_vertex : vertex_x = 3 ∧ vertex_y = 5)
  (h_point : ∃ x y: ℝ, x = 2 ∧ y = 2 ∧ y = a * (x - vertex_x)^2 + vertex_y)
  (h_vertical_axis : ∃ a b c, a = -3 ∧ b = 18 ∧ c = -22):
  ∀ x: ℝ, x ≠ vertex_x → b^2 - 4 * a * c > 0 := 
    sorry

end parabola_equation_l28_28138


namespace number_of_chords_l28_28210

/-- Ten points are marked on the circumference of a circle.
    Prove that the number of different chords that can be drawn
    by connecting any two of these ten points is 45.
-/
theorem number_of_chords (n : ℕ) (h_n : n = 10) : 
  (nat.choose n 2) = 45 :=
by
  rw h_n
  norm_num

end number_of_chords_l28_28210


namespace more_flour_than_sugar_l28_28867

variable (total_flour : ℕ) (total_sugar : ℕ)
variable (flour_added : ℕ)

def additional_flour_needed (total_flour flour_added : ℕ) : ℕ :=
  total_flour - flour_added

theorem more_flour_than_sugar :
  additional_flour_needed 10 7 - 2 = 1 :=
by
  sorry

end more_flour_than_sugar_l28_28867


namespace minimize_f_at_3_l28_28450

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l28_28450


namespace july_16_2010_is_wednesday_l28_28720

-- Define necessary concepts for the problem

def is_tuesday (d : ℕ) : Prop := (d % 7 = 2)
def day_after_n_days (d n : ℕ) : ℕ := (d + n) % 7

-- The statement we want to prove
theorem july_16_2010_is_wednesday (h : is_tuesday 1) : day_after_n_days 1 15 = 3 := 
sorry

end july_16_2010_is_wednesday_l28_28720


namespace inequality_C_false_l28_28299

theorem inequality_C_false (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : (1 / a) ^ (1 / b) ≤ 1 := 
sorry

end inequality_C_false_l28_28299


namespace fish_population_estimation_l28_28710

def tagged_fish_day1 := (30, 25, 25) -- (Species A, Species B, Species C)
def tagged_fish_day2 := (40, 35, 25) -- (Species A, Species B, Species C)
def caught_fish_day3 := (60, 50, 30) -- (Species A, Species B, Species C)
def tagged_fish_day3 := (4, 6, 2)    -- (Species A, Species B, Species C)
def caught_fish_day4 := (70, 40, 50) -- (Species A, Species B, Species C)
def tagged_fish_day4 := (5, 7, 3)    -- (Species A, Species B, Species C)

def total_tagged_fish (day1 : (ℕ × ℕ × ℕ)) (day2 : (ℕ × ℕ × ℕ)) :=
  let (a1, b1, c1) := day1
  let (a2, b2, c2) := day2
  (a1 + a2, b1 + b2, c1 + c2)

def average_proportion_tagged (caught3 tagged3 caught4 tagged4 : (ℕ × ℕ × ℕ)) :=
  let (c3a, c3b, c3c) := caught3
  let (t3a, t3b, t3c) := tagged3
  let (c4a, c4b, c4c) := caught4
  let (t4a, t4b, t4c) := tagged4
  ((t3a / c3a + t4a / c4a) / 2,
   (t3b / c3b + t4b / c4b) / 2,
   (t3c / c3c + t4c / c4c) / 2)

def estimate_population (total_tagged average_proportion : (ℕ × ℕ × ℕ)) :=
  let (ta, tb, tc) := total_tagged
  let (pa, pb, pc) := average_proportion
  (ta / pa, tb / pb, tc / pc)

theorem fish_population_estimation :
  let total_tagged := total_tagged_fish tagged_fish_day1 tagged_fish_day2
  let avg_prop := average_proportion_tagged caught_fish_day3 tagged_fish_day3 caught_fish_day4 tagged_fish_day4
  estimate_population total_tagged avg_prop = (1014, 407, 790) :=
by
  sorry

end fish_population_estimation_l28_28710


namespace time_difference_l28_28645

noncomputable def hour_angle (n : ℝ) : ℝ :=
  150 + (n / 2)

noncomputable def minute_angle (n : ℝ) : ℝ :=
  6 * n

theorem time_difference (n1 n2 : ℝ)
  (h1 : |(hour_angle n1) - (minute_angle n1)| = 120)
  (h2 : |(hour_angle n2) - (minute_angle n2)| = 120) :
  n2 - n1 = 43.64 := 
sorry

end time_difference_l28_28645


namespace maximum_rabbits_condition_l28_28400

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l28_28400


namespace problem_statement_l28_28313

-- Definitions for the given conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- The main statement that needs to be proved
theorem problem_statement (f : ℝ → ℝ) (h_odd : odd_function f) (h_monotone : monotone_decreasing f) : f (-1) > f 3 :=
by 
  sorry

end problem_statement_l28_28313


namespace joan_paid_230_l28_28719

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 :=
sorry

end joan_paid_230_l28_28719


namespace horner_evaluation_at_2_l28_28991

noncomputable def f : ℕ → ℕ :=
  fun x => (((2 * x + 3) * x + 0) * x + 5) * x - 4

theorem horner_evaluation_at_2 : f 2 = 14 :=
  by
    sorry

end horner_evaluation_at_2_l28_28991


namespace find_multiplier_l28_28104

theorem find_multiplier :
  ∀ (x n : ℝ), (x = 5) → (x * n = (16 - x) + 4) → n = 3 :=
by
  intros x n hx heq
  sorry

end find_multiplier_l28_28104


namespace max_rabbits_l28_28417

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l28_28417


namespace opposite_of_neg_two_thirds_l28_28593

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l28_28593


namespace min_value_at_3_l28_28465

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l28_28465


namespace smallest_four_digit_divisible_by_53_l28_28093

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28093


namespace ratio_of_inscribed_squares_l28_28491

open Real

-- Condition: A square inscribed in a right triangle with sides 3, 4, and 5
def inscribedSquareInRightTriangle1 (x : ℝ) (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5 ∧ x = 12 / 7

-- Condition: A square inscribed in a different right triangle with sides 5, 12, and 13
def inscribedSquareInRightTriangle2 (y : ℝ) (d e f : ℝ) : Prop :=
  d = 5 ∧ e = 12 ∧ f = 13 ∧ y = 169 / 37

-- The ratio x / y is 444 / 1183
theorem ratio_of_inscribed_squares (x y : ℝ) (a b c d e f : ℝ) :
  inscribedSquareInRightTriangle1 x a b c →
  inscribedSquareInRightTriangle2 y d e f →
  x / y = 444 / 1183 :=
by
  intros h1 h2
  sorry

end ratio_of_inscribed_squares_l28_28491


namespace distance_EC_l28_28939

-- Define the points and given distances as conditions
structure Points :=
  (A B C D E : Type)

-- Distances between points
variables {Points : Type}
variables (dAB dBC dCD dDE dEA dEC : ℝ)
variables [Nonempty Points]

-- Specify conditions: distances in kilometers
def distances_given (dAB dBC dCD dDE dEA : ℝ) : Prop :=
  dAB = 30 ∧ dBC = 80 ∧ dCD = 236 ∧ dDE = 86 ∧ dEA = 40

-- Main theorem: prove that the distance from E to C is 63.4 km
theorem distance_EC (h : distances_given 30 80 236 86 40) : dEC = 63.4 :=
sorry

end distance_EC_l28_28939


namespace intersection_with_y_axis_l28_28587

theorem intersection_with_y_axis (x y : ℝ) : (x + y - 3 = 0 ∧ x = 0) → (x = 0 ∧ y = 3) :=
by {
  sorry
}

end intersection_with_y_axis_l28_28587


namespace intersection_is_line_l28_28115

-- Define the two planes as given in the conditions
def plane1 (x y z : ℝ) : Prop := x + 5 * y + 2 * z - 5 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 5 * y - z + 5 = 0

-- The intersection of the planes should satisfy both plane equations
def is_on_line (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the canonical equation of the line
def line_eq (x y z : ℝ) : Prop := (∃ k : ℝ, x = 5 * k ∧ y = 5 * k + 1 ∧ z = -15 * k)

-- The proof statement
theorem intersection_is_line :
  (∀ x y z : ℝ, is_on_line x y z → line_eq x y z) ∧ 
  (∀ x y z : ℝ, line_eq x y z → is_on_line x y z) :=
by
  sorry

end intersection_is_line_l28_28115


namespace find_f_2015_plus_f_2016_l28_28833

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom functional_equation (x : ℝ) : f (3/2 - x) = f x
axiom value_at_minus2 : f (-2) = -3

theorem find_f_2015_plus_f_2016 : f 2015 + f 2016 = 3 := 
by {
  sorry
}

end find_f_2015_plus_f_2016_l28_28833


namespace smallest_common_multiple_l28_28615

theorem smallest_common_multiple (n : ℕ) : 
  (2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧ 1000 ≤ n ∧ n < 10000) → n = 1008 :=
by {
    sorry
}

end smallest_common_multiple_l28_28615


namespace number_of_rings_l28_28620

def is_number_ring (A : Set ℝ) : Prop :=
  ∀ (a b : ℝ), a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A ∧ (a * b) ∈ A

def Z := { n : ℝ | ∃ k : ℤ, n = k }
def N := { n : ℝ | ∃ k : ℕ, n = k }
def Q := { n : ℝ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b }
def R := { n : ℝ | True }
def M := { x : ℝ | ∃ (n m : ℤ), x = n + m * Real.sqrt 2 }
def P := { x : ℝ | ∃ (m n : ℕ), n ≠ 0 ∧ x = m / (2 * n) }

theorem number_of_rings :
  (is_number_ring Z) ∧ ¬(is_number_ring N) ∧ (is_number_ring Q) ∧ 
  (is_number_ring R) ∧ (is_number_ring M) ∧ ¬(is_number_ring P) :=
by sorry

end number_of_rings_l28_28620


namespace option_A_incorrect_l28_28475

theorem option_A_incorrect {a b m : ℤ} (h : am = bm) : m = 0 ∨ a = b :=
by sorry

end option_A_incorrect_l28_28475


namespace cube_volume_l28_28484

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 294) : s^3 = 343 := 
by 
  sorry

end cube_volume_l28_28484


namespace find_g_neg6_l28_28895

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l28_28895


namespace least_xy_value_l28_28159

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end least_xy_value_l28_28159


namespace factorial_simplification_l28_28754

theorem factorial_simplification :
  Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 728 := 
sorry

end factorial_simplification_l28_28754


namespace smallest_four_digit_multiple_of_53_l28_28009

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28009


namespace max_tiles_on_floor_l28_28109

   -- Definitions corresponding to conditions
   def tile_length_1 : ℕ := 35
   def tile_width_1 : ℕ := 30
   def tile_length_2 : ℕ := 30
   def tile_width_2 : ℕ := 35
   def floor_length : ℕ := 1000
   def floor_width : ℕ := 210

   -- Conditions:
   -- 1. Tiles do not overlap.
   -- 2. Tiles are placed with edges jutting against each other on all edges.
   -- 3. A tile can be placed in any orientation so long as its edges are parallel to the edges of the floor.
   -- 4. No tile should overshoot any edge of the floor.

   theorem max_tiles_on_floor :
     let tiles_orientation_1 := (floor_length / tile_length_1) * (floor_width / tile_width_1)
     let tiles_orientation_2 := (floor_length / tile_length_2) * (floor_width / tile_width_2)
     max tiles_orientation_1 tiles_orientation_2 = 198 :=
   by {
     -- The actual proof handling is skipped, as per instructions.
     sorry
   }
   
end max_tiles_on_floor_l28_28109


namespace down_payment_calculation_l28_28204

noncomputable def tablet_price : ℝ := 450
noncomputable def installment_1 : ℝ := 4 * 40
noncomputable def installment_2 : ℝ := 4 * 35
noncomputable def installment_3 : ℝ := 4 * 30
noncomputable def total_savings : ℝ := 70
noncomputable def total_installments := tablet_price + total_savings
noncomputable def installment_payments := installment_1 + installment_2 + installment_3
noncomputable def down_payment := total_installments - installment_payments

theorem down_payment_calculation : down_payment = 100 := by
  unfold down_payment
  unfold total_installments
  unfold installment_payments
  unfold tablet_price
  unfold total_savings
  unfold installment_1
  unfold installment_2
  unfold installment_3
  sorry

end down_payment_calculation_l28_28204


namespace students_above_90_l28_28320

theorem students_above_90 (total_students : ℕ) (above_90_chinese : ℕ) (above_90_math : ℕ)
  (all_above_90_at_least_one_subject : total_students = 50 ∧ above_90_chinese = 33 ∧ above_90_math = 38 ∧ 
    ∀ (n : ℕ), n < total_students → (n < above_90_chinese ∨ n < above_90_math)) :
  (above_90_chinese + above_90_math - total_students) = 21 :=
by
  sorry

end students_above_90_l28_28320


namespace original_paint_intensity_l28_28626

theorem original_paint_intensity (I : ℝ) (h1 : 0.5 * I + 0.5 * 20 = 15) : I = 10 :=
sorry

end original_paint_intensity_l28_28626


namespace carrots_eaten_after_dinner_l28_28854

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end carrots_eaten_after_dinner_l28_28854


namespace tangent_line_at_origin_l28_28346

-- Define the function f(x) = x^3 + ax with an extremum at x = 1
def f (x a : ℝ) : ℝ := x^3 + a * x

-- Define the condition for a local extremum at x = 1: f'(1) = 0
def extremum_condition (a : ℝ) : Prop := (3 * 1^2 + a = 0)

-- Define the derivative of f at x = 0
def derivative_at_origin (a : ℝ) : ℝ := 3 * 0^2 + a

-- Define the value of function at x = 0
def value_at_origin (a : ℝ) : ℝ := f 0 a

-- The main theorem to prove
theorem tangent_line_at_origin (a : ℝ) (ha : extremum_condition a) :
    (value_at_origin a = 0) ∧ (derivative_at_origin a = -3) → ∀ x, (3 * x + (f x a - f 0 a) / (x - 0) = 0) := by
  sorry

end tangent_line_at_origin_l28_28346


namespace smallest_four_digit_divisible_by_53_l28_28067

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28067


namespace quadratic_function_range_l28_28326

theorem quadratic_function_range (f : ℝ → ℝ) (a : ℝ)
  (h_quad : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_sym : ∀ x, f (2 + x) = f (2 - x))
  (h_cond : f a ≤ f 0 ∧ f 0 < f 1) :
  a ≤ 0 ∨ a ≥ 4 :=
sorry

end quadratic_function_range_l28_28326


namespace find_a1_l28_28599

theorem find_a1 (S : ℕ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) :
  (∀ n : ℕ, S n = a1 * (2^n - 1)) → a 4 = 24 → 
  a 4 = S 4 - S 3 → 
  a1 = 3 :=
by
  sorry

end find_a1_l28_28599


namespace nancy_crayons_l28_28871

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end nancy_crayons_l28_28871


namespace base3_addition_l28_28646

theorem base3_addition :
  (2 + 1 * 3 + 2 * 9 + 1 * 27 + 2 * 81) + (1 + 1 * 3 + 2 * 9 + 2 * 27) + (2 * 9 + 1 * 27 + 0 * 81 + 2 * 243) + (1 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81) = 
  2 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81 + 1 * 243 + 1 * 729 := sorry

end base3_addition_l28_28646


namespace maximum_rabbits_l28_28394

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l28_28394


namespace product_polynomial_coeffs_l28_28225

theorem product_polynomial_coeffs
  (g h : ℚ)
  (h1 : 7 * d^2 - 3 * d + g * (3 * d^2 + h * d - 5) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) :
  g + h = -28/9 := 
  sorry

end product_polynomial_coeffs_l28_28225


namespace min_value_at_3_l28_28463

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l28_28463


namespace max_rabbits_l28_28420

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l28_28420


namespace find_g_neg_six_l28_28922

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l28_28922


namespace quadratic_not_factored_l28_28563

theorem quadratic_not_factored
  (a b c : ℕ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h_p : a * 1991^2 + b * 1991 + c = p) :
  ¬ (∃ d₁ d₂ e₁ e₂ : ℤ, a = d₁ * d₂ ∧ b = d₁ * e₂ + d₂ * e₁ ∧ c = e₁ * e₂) :=
sorry

end quadratic_not_factored_l28_28563


namespace pattern_formula_l28_28570

theorem pattern_formula (n : ℤ) : n * (n + 2) = (n + 1) ^ 2 - 1 := 
by sorry

end pattern_formula_l28_28570


namespace value_of_m_l28_28538

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end value_of_m_l28_28538


namespace missing_keys_total_l28_28342

-- Definitions for the problem conditions

def num_consonants : ℕ := 21
def num_vowels : ℕ := 5
def missing_consonants_fraction : ℚ := 1 / 7
def missing_vowels : ℕ := 2

-- Statement to prove the total number of missing keys

theorem missing_keys_total :
  let missing_consonants := num_consonants * missing_consonants_fraction in
  let total_missing_keys := missing_consonants + missing_vowels in
  total_missing_keys = 5 :=
by {
  -- Placeholder proof
  sorry
}

end missing_keys_total_l28_28342


namespace determine_digits_from_expression_l28_28195

theorem determine_digits_from_expression (a b c x y z S : ℕ) 
  (hx : x = 100) (hy : y = 10) (hz : z = 1)
  (hS : S = a * x + b * y + c * z) :
  S = 100 * a + 10 * b + c :=
by
  -- Variables
  -- a, b, c : ℕ -- digits to find
  -- x, y, z : ℕ -- chosen numbers
  -- S : ℕ -- the given sum

  -- Assumptions
  -- hx : x = 100
  -- hy : y = 10
  -- hz : z = 1
  -- hS : S = a * x + b * y + c * z
  sorry

end determine_digits_from_expression_l28_28195


namespace square_of_1023_l28_28998

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l28_28998


namespace max_rabbits_with_long_ears_and_jumping_far_l28_28408

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l28_28408


namespace apples_remaining_l28_28879

-- Define the initial condition of the number of apples on the tree
def initial_apples : ℕ := 7

-- Define the number of apples picked by Rachel
def picked_apples : ℕ := 4

-- Proof goal: the number of apples remaining on the tree is 3
theorem apples_remaining : (initial_apples - picked_apples = 3) :=
sorry

end apples_remaining_l28_28879


namespace fred_final_cards_l28_28146

def initial_cards : ℕ := 40
def keith_bought : ℕ := 22
def linda_bought : ℕ := 15

theorem fred_final_cards : initial_cards - keith_bought - linda_bought = 3 :=
by sorry

end fred_final_cards_l28_28146


namespace count_fourdigit_integers_div_by_35_of_form_x35_l28_28174

theorem count_fourdigit_integers_div_by_35_of_form_x35 : 
  {n: Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 35 ∧ n % 35 = 0}.card = 13 :=
by
  sorry

end count_fourdigit_integers_div_by_35_of_form_x35_l28_28174


namespace positive_inequality_l28_28143

theorem positive_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * (a^2 * b^2) / 4 + 9 * a * b > 18 := 
  sorry

end positive_inequality_l28_28143


namespace simplify_fraction_l28_28371

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l28_28371


namespace smallest_four_digit_divisible_by_53_l28_28027

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28027


namespace smallest_four_digit_multiple_of_53_l28_28002

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28002


namespace train_passes_bridge_in_expected_time_l28_28478

def train_length : ℕ := 360
def speed_kmph : ℕ := 45
def bridge_length : ℕ := 140

def speed_mps : ℚ := (speed_kmph * 1000) / 3600
def total_distance : ℕ := train_length + bridge_length
def time_to_pass : ℚ := total_distance / speed_mps

theorem train_passes_bridge_in_expected_time : time_to_pass = 40 := by
  sorry

end train_passes_bridge_in_expected_time_l28_28478


namespace minimize_quadratic_l28_28452

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l28_28452


namespace inverse_function_shift_l28_28588

-- Conditions
variable {f : ℝ → ℝ} {f_inv : ℝ → ℝ}
variable (hf : ∀ x : ℝ, f_inv (f x) = x ∧ f (f_inv x) = x)
variable (point_B : f 3 = -1)

-- Proof statement
theorem inverse_function_shift :
  f_inv (-3 + 2) = 3 :=
by
  -- Proof goes here
  sorry

end inverse_function_shift_l28_28588


namespace simplify_sqrt_l28_28102

theorem simplify_sqrt (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a :=
by
  sorry

end simplify_sqrt_l28_28102


namespace minimize_quadratic_l28_28451

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l28_28451


namespace min_value_at_3_l28_28461

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l28_28461


namespace projectile_first_reaches_70_feet_l28_28791

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ , (t > 0) ∧ (-16 * t^2 + 80 * t = 70) ∧ (∀ t' : ℝ, (t' > 0) ∧ (-16 * t'^2 + 80 * t' = 70) → t ≤ t') :=
sorry

end projectile_first_reaches_70_feet_l28_28791


namespace smallest_four_digit_multiple_of_53_l28_28015

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28015


namespace equal_distribution_l28_28708

theorem equal_distribution (total_cookies bags : ℕ) (h_total : total_cookies = 14) (h_bags : bags = 7) : total_cookies / bags = 2 := by
  sorry

end equal_distribution_l28_28708


namespace Tom_water_intake_daily_l28_28809

theorem Tom_water_intake_daily (cans_per_day : ℕ) (oz_per_can : ℕ) (fluid_per_week : ℕ) (days_per_week : ℕ)
  (h1 : cans_per_day = 5) 
  (h2 : oz_per_can = 12) 
  (h3 : fluid_per_week = 868) 
  (h4 : days_per_week = 7) : 
  ((fluid_per_week - (cans_per_day * oz_per_can * days_per_week)) / days_per_week) = 64 := 
sorry

end Tom_water_intake_daily_l28_28809


namespace multiply_98_102_l28_28802

theorem multiply_98_102 : 98 * 102 = 9996 :=
by sorry

end multiply_98_102_l28_28802


namespace solve_inequality_l28_28295

theorem solve_inequality (x : ℝ) : abs ((3 - x) / 4) < 1 ↔ 2 < x ∧ x < 7 :=
by {
  sorry
}

end solve_inequality_l28_28295


namespace smallest_pencils_l28_28472

theorem smallest_pencils (P : ℕ) :
  (P > 2) ∧
  (P % 5 = 2) ∧
  (P % 9 = 2) ∧
  (P % 11 = 2) →
  P = 497 := by
  sorry

end smallest_pencils_l28_28472


namespace sacred_k_words_n10_k4_l28_28267

/- Definitions for the problem -/
def sacred_k_words_count (n k : ℕ) (hk : k < n / 2) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * (Nat.factorial k / k)

theorem sacred_k_words_n10_k4 : sacred_k_words_count 10 4 (by norm_num : 4 < 10 / 2) = 600 := by
  sorry

end sacred_k_words_n10_k4_l28_28267


namespace test_scores_l28_28712

noncomputable def expected_score (n : ℕ) (points : ℕ) (p : ℝ) : ℝ :=
  n * (points * p)

noncomputable def variance_score (n : ℕ) (points : ℕ) (p : ℝ) : ℝ :=
  n * ((points^2 * p) - (points * p)^2)

theorem test_scores (n : ℕ) (points : ℕ) (p : ℝ) (max_score : ℕ) :
  n = 25 ∧ points = 4 ∧ p = 0.8 ∧ max_score = 100 →
  expected_score n points p = 80 ∧ variance_score n points p = 64 :=
by
  intros h
  cases h with hn hrest
  cases hrest with hpqr hmax
  cases hpqr with hpq hp
  split
  {
    simp [expected_score, hn, hpq, hp],
    norm_num,
  },
  {
    simp [variance_score, hn, hpq, hp],
    norm_num,
  }

end test_scores_l28_28712


namespace solve_system_of_equations_l28_28834

theorem solve_system_of_equations (x y : ℚ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : x + 3 * y = 9) : 
  x = 42 / 11 ∧ y = 19 / 11 :=
by {
  sorry
}

end solve_system_of_equations_l28_28834


namespace min_concerts_l28_28244

theorem min_concerts (n : ℕ) (h1 : n = 8) :
  ∃ m : ℕ, (∀ (S : ℕ),
    (∀ i j : fin n, i ≠ j →
      ∃ m : ℕ, (∀ k l : fin n, k ≠ l → occurrences_of_pair k l m = S)) →
    S = 3) → m = 14 :=
by
  sorry

end min_concerts_l28_28244


namespace small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l28_28435

-- 1. Prove that the small frog can reach the 7th rung
theorem small_frog_reaches_7th_rung : ∃ (a b : ℕ), 2 * a + 3 * b = 7 :=
by sorry

-- 2. Prove that the medium frog cannot reach the 1st rung
theorem medium_frog_cannot_reach_1st_rung : ¬(∃ (a b : ℕ), 2 * a + 4 * b = 1) :=
by sorry

-- 3. Prove that the large frog can reach the 3rd rung
theorem large_frog_reaches_3rd_rung : ∃ (a b : ℕ), 6 * a + 9 * b = 3 :=
by sorry

end small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l28_28435


namespace jamestown_theme_parks_l28_28194

theorem jamestown_theme_parks (J : ℕ) (Venice := J + 25) (MarinaDelRay := J + 50) (total := J + Venice + MarinaDelRay) (h : total = 135) : J = 20 :=
by
  -- proof step to be done here
  sorry

end jamestown_theme_parks_l28_28194


namespace leftmost_three_nonzero_digits_of_arrangements_l28_28523

-- Definitions based on the conditions
def num_rings := 10
def chosen_rings := 6
def num_fingers := 5

-- Calculate the possible arrangements
def arrangements : ℕ := Nat.choose num_rings chosen_rings * Nat.factorial chosen_rings * Nat.choose (chosen_rings + (num_fingers - 1)) (num_fingers - 1)

-- Find the leftmost three nonzero digits
def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.takeWhile (· > 0)).reverse.take 3
  |> List.foldl (· + · * 10) 0
  
-- The main theorem to prove
theorem leftmost_three_nonzero_digits_of_arrangements :
  leftmost_three_nonzero_digits arrangements = 317 :=
by
  sorry

end leftmost_three_nonzero_digits_of_arrangements_l28_28523


namespace distance_origin_to_point_on_parabola_l28_28197

noncomputable def origin : ℝ × ℝ := (0, 0)

noncomputable def parabola_focus (x y : ℝ) : Prop :=
  x^2 = 4 * y ∧ y = 1

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

theorem distance_origin_to_point_on_parabola (x y : ℝ) (hx : x^2 = 4 * y)
 (hf : (0, 1) = (0, 1)) (hPF : (x - 0)^2 + (y - 1)^2 = 25) : (x^2 + y^2 = 32) :=
by
  sorry

end distance_origin_to_point_on_parabola_l28_28197


namespace compute_diff_squares_l28_28660

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l28_28660


namespace most_probable_sellable_samples_l28_28424

/-- Prove that the most probable number k of sellable samples out of 24,
given each has a 0.6 probability of being sellable, is either 14 or 15. -/
theorem most_probable_sellable_samples (n : ℕ) (p : ℝ) (q : ℝ) (k₀ k₁ : ℕ) 
  (h₁ : n = 24) (h₂ : p = 0.6) (h₃ : q = 1 - p)
  (h₄ : 24 * p - q < k₀) (h₅ : k₀ < 24 * p + p) 
  (h₆ : k₀ = 14) (h₇ : k₁ = 15) :
  (k₀ = 14 ∨ k₀ = 15) :=
  sorry

end most_probable_sellable_samples_l28_28424


namespace number_of_chords_l28_28217

theorem number_of_chords (n : ℕ) (h : n = 10) : finset.card (finset.pairs (finset.range n)) = 45 :=
by
  rw [h]
  -- Sorry to skip the proof steps as required
  sorry

end number_of_chords_l28_28217


namespace compute_HHHH_of_3_l28_28121

def H (x : ℝ) : ℝ := -0.5 * x^2 + 3 * x

theorem compute_HHHH_of_3 :
  H (H (H (H 3))) = 2.689453125 := by
  sorry

end compute_HHHH_of_3_l28_28121


namespace smallest_four_digit_divisible_by_53_l28_28039

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28039


namespace marble_problem_l28_28494

variable (A V M : ℕ)

theorem marble_problem
  (h1 : A + 5 = V - 5)
  (h2 : V + 2 * (A + 5) = A - 2 * (A + 5) + M) :
  M = 10 :=
sorry

end marble_problem_l28_28494


namespace tea_in_box_l28_28354

theorem tea_in_box (tea_per_day ounces_per_week ounces_per_box : ℝ) 
    (H1 : tea_per_day = 1 / 5) 
    (H2 : ounces_per_week = tea_per_day * 7) 
    (H3 : ounces_per_box = ounces_per_week * 20) : 
    ounces_per_box = 28 := 
by
  sorry

end tea_in_box_l28_28354


namespace smallest_four_digit_divisible_by_53_l28_28074

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l28_28074


namespace total_cost_is_63_l28_28653

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end total_cost_is_63_l28_28653


namespace smallest_four_digit_divisible_by_53_l28_28060

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28060


namespace ferry_time_difference_l28_28764

-- Definitions for the given conditions
def speed_p := 8
def time_p := 3
def distance_p := speed_p * time_p
def distance_q := 3 * distance_p
def speed_q := speed_p + 1
def time_q := distance_q / speed_q

-- Theorem to be proven
theorem ferry_time_difference : (time_q - time_p) = 5 := 
by
  let speed_p := 8
  let time_p := 3
  let distance_p := speed_p * time_p
  let distance_q := 3 * distance_p
  let speed_q := speed_p + 1
  let time_q := distance_q / speed_q
  sorry

end ferry_time_difference_l28_28764


namespace b_range_l28_28534

noncomputable def f (a b x : ℝ) := (x - 1) * Real.log x - a * x + a + b

theorem b_range (a b : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = 0 ∧ f a b x2 = 0) :
  b < 0 :=
sorry

end b_range_l28_28534


namespace discriminant_positive_l28_28816

theorem discriminant_positive
  (a b c : ℝ)
  (h : (a + b + c) * c < 0) : b^2 - 4 * a * c > 0 :=
sorry

end discriminant_positive_l28_28816


namespace break_even_point_l28_28125

def cost_of_commodity (a : ℝ) : ℝ := a

def profit_beginning_of_month (a : ℝ) : ℝ := 100 + (a + 100) * 0.024

def profit_end_of_month : ℝ := 115

theorem break_even_point (a : ℝ) : profit_end_of_month - profit_beginning_of_month a = 0 → a = 525 := 
by sorry

end break_even_point_l28_28125


namespace div_by_7_l28_28671

theorem div_by_7 (n : ℕ) : (3 ^ (12 * n + 1) + 2 ^ (6 * n + 2)) % 7 = 0 := by
  sorry

end div_by_7_l28_28671


namespace smallest_four_digit_divisible_by_53_l28_28026

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28026


namespace page_added_twice_l28_28930

theorem page_added_twice (n k : ℕ) (h1 : (n * (n + 1)) / 2 + k = 1986) : k = 33 :=
sorry

end page_added_twice_l28_28930


namespace length_of_bridge_is_80_l28_28127

-- Define the given constants
def length_of_train : ℕ := 280
def speed_of_train : ℕ := 18
def time_to_cross : ℕ := 20

-- Define the distance traveled by the train in the given time
def distance_traveled : ℕ := speed_of_train * time_to_cross

-- Define the length of the bridge from the given distance traveled
def length_of_bridge := distance_traveled - length_of_train

-- The theorem to prove the length of the bridge is 80 meters
theorem length_of_bridge_is_80 :
  length_of_bridge = 80 := by
  sorry

end length_of_bridge_is_80_l28_28127


namespace no_third_number_for_lcm_l28_28813

theorem no_third_number_for_lcm (a : ℕ) : ¬ (Nat.lcm (Nat.lcm 23 46) a = 83) :=
sorry

end no_third_number_for_lcm_l28_28813


namespace ms_cole_total_students_l28_28567

def students_6th : ℕ := 40
def students_4th : ℕ := 4 * students_6th
def students_7th : ℕ := 2 * students_4th

def total_students : ℕ := students_6th + students_4th + students_7th

theorem ms_cole_total_students :
  total_students = 520 :=
by
  sorry

end ms_cole_total_students_l28_28567


namespace contrapositive_proof_l28_28585

theorem contrapositive_proof (x m : ℝ) :
  (m < 0 → (∃ r : ℝ, r * r + 3 * r + m = 0)) ↔
  (¬ (∃ r : ℝ, r * r + 3 * r + m = 0) → m ≥ 0) :=
by
  sorry

end contrapositive_proof_l28_28585


namespace find_side_length_l28_28709

theorem find_side_length (a : ℝ) (b : ℝ) (A B : ℝ) (ha : a = 4) (hA : A = 45) (hB : B = 60) :
    b = 2 * Real.sqrt 6 := by
  sorry

end find_side_length_l28_28709


namespace network_structure_l28_28126

theorem network_structure 
  (n : ℕ)
  (is_acquainted : Fin n → Fin n → Prop)
  (H_symmetric : ∀ x y, is_acquainted x y = is_acquainted y x) 
  (H_common_acquaintance : ∀ x y, ¬ is_acquainted x y → ∃! z : Fin n, is_acquainted x z ∧ is_acquainted y z) :
  ∃ (G : SimpleGraph (Fin n)), (∀ x y, G.Adj x y = is_acquainted x y) ∧
    (∀ x y, ¬ G.Adj x y → (∃ (z1 z2 : Fin n), G.Adj x z1 ∧ G.Adj y z1 ∧ G.Adj x z2 ∧ G.Adj y z2)) :=
by
  sorry

end network_structure_l28_28126


namespace avg_speed_round_trip_l28_28790

-- Definitions for the conditions
def speed_P_to_Q : ℝ := 80
def distance (D : ℝ) : ℝ := D
def speed_increase_percentage : ℝ := 0.1
def speed_Q_to_P : ℝ := speed_P_to_Q * (1 + speed_increase_percentage)

-- Average speed calculation function
noncomputable def average_speed (D : ℝ) : ℝ := 
  let total_distance := 2 * D
  let time_P_to_Q := D / speed_P_to_Q
  let time_Q_to_P := D / speed_Q_to_P
  let total_time := time_P_to_Q + time_Q_to_P
  total_distance / total_time

-- Theorem: Average speed for the round trip is 83.81 km/hr
theorem avg_speed_round_trip (D : ℝ) : average_speed D = 83.81 := 
by 
  -- Dummy proof placeholder
  sorry

end avg_speed_round_trip_l28_28790


namespace student_thought_six_is_seven_l28_28275

theorem student_thought_six_is_seven
  (n : ℕ → ℕ)
  (h1 : (n 1 + n 3) / 2 = 2)
  (h2 : (n 2 + n 4) / 2 = 3)
  (h3 : (n 3 + n 5) / 2 = 4)
  (h4 : (n 4 + n 6) / 2 = 5)
  (h5 : (n 5 + n 7) / 2 = 6)
  (h6 : (n 6 + n 8) / 2 = 7)
  (h7 : (n 7 + n 9) / 2 = 8)
  (h8 : (n 8 + n 10) / 2 = 9)
  (h9 : (n 9 + n 1) / 2 = 10)
  (h10 : (n 10 + n 2) / 2 = 1) : 
  n 6 = 7 := 
  sorry

end student_thought_six_is_seven_l28_28275


namespace notAlwaysTriangleInSecondQuadrantAfterReflection_l28_28550

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  P : Point
  Q : Point
  R : Point

def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def reflectionOverYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

def reflectTriangleOverYEqualsX (T : Triangle) : Triangle :=
  { P := reflectionOverYEqualsX T.P,
    Q := reflectionOverYEqualsX T.Q,
    R := reflectionOverYEqualsX T.R }

def triangleInSecondQuadrant (T : Triangle) : Prop :=
  isInSecondQuadrant T.P ∧ isInSecondQuadrant T.Q ∧ isInSecondQuadrant T.R

theorem notAlwaysTriangleInSecondQuadrantAfterReflection
  (T : Triangle)
  (h : triangleInSecondQuadrant T)
  : ¬ (triangleInSecondQuadrant (reflectTriangleOverYEqualsX T)) := 
sorry -- Proof not required

end notAlwaysTriangleInSecondQuadrantAfterReflection_l28_28550


namespace brick_wall_l28_28251

theorem brick_wall (x : ℕ) 
  (h1 : x / 9 * 9 = x)
  (h2 : x / 10 * 10 = x)
  (h3 : 5 * (x / 9 + x / 10 - 10) = x) :
  x = 900 := 
sorry

end brick_wall_l28_28251


namespace sum_of_favorite_numbers_l28_28564

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers_l28_28564


namespace initial_group_size_l28_28382

theorem initial_group_size
  (n : ℕ) (W : ℝ)
  (h_avg_increase : ∀ W n, ((W + 12) / n) = (W / n + 3))
  (h_new_person_weight : 82 = 70 + 12) : n = 4 :=
by
  sorry

end initial_group_size_l28_28382


namespace smallest_fraction_numerator_l28_28501

theorem smallest_fraction_numerator :
  ∃ a b : ℕ, 10 ≤ a ∧ a < b ∧ b ≤ 99 ∧ 6 * a > 5 * b ∧ ∀ c d : ℕ,
    (10 ≤ c ∧ c < d ∧ d ≤ 99 ∧ 6 * c > 5 * d → a ≤ c) ∧ 
    a = 81 :=
sorry

end smallest_fraction_numerator_l28_28501


namespace value_of_leftover_coins_l28_28642

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40

def ana_quarters : ℕ := 95
def ana_dimes : ℕ := 183

def ben_quarters : ℕ := 104
def ben_dimes : ℕ := 219

def leftover_quarters : ℕ := (ana_quarters + ben_quarters) % quarters_per_roll
def leftover_dimes : ℕ := (ana_dimes + ben_dimes) % dimes_per_roll

def dollar_value (quarters dimes : ℕ) : ℝ := quarters * 0.25 + dimes * 0.10

theorem value_of_leftover_coins : 
  dollar_value leftover_quarters leftover_dimes = 6.95 := 
  sorry

end value_of_leftover_coins_l28_28642


namespace initial_students_count_l28_28732

theorem initial_students_count (N T : ℕ) (h1 : T = N * 90) (h2 : (T - 120) / (N - 3) = 95) : N = 33 :=
by
  sorry

end initial_students_count_l28_28732


namespace maximum_rabbits_l28_28395

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l28_28395


namespace minimize_quadratic_l28_28470

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l28_28470


namespace count_valid_pairs_l28_28172

theorem count_valid_pairs : 
  ∃! n : ℕ, 
  n = 2 ∧ 
  (∀ (a b : ℕ), (0 < a ∧ 0 < b) → 
    (a * b + 97 = 18 * Nat.lcm a b + 14 * Nat.gcd a b) → 
    n = 2)
:= sorry

end count_valid_pairs_l28_28172


namespace sum_of_smallest_and_largest_eq_2y_l28_28580

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l28_28580


namespace maximum_rabbits_condition_l28_28398

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l28_28398


namespace max_rabbits_l28_28411

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l28_28411


namespace g_neg_six_eq_neg_twenty_l28_28915

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l28_28915


namespace mean_equality_l28_28423

theorem mean_equality (z : ℚ) (h1 : (8 + 15 + 24) / 3 = 47 / 3) (h2 : (18 + z) / 2 = 47 / 3) : z = 40 / 3 :=
by
  sorry

end mean_equality_l28_28423


namespace prism_faces_l28_28640

-- Define a structure for a prism with a given number of edges
def is_prism (edges : ℕ) := 
  ∃ (n : ℕ), 3 * n = edges

-- Define the theorem to prove the number of faces in a prism given it has 21 edges
theorem prism_faces (h : is_prism 21) : ∃ (faces : ℕ), faces = 9 :=
by
  sorry

end prism_faces_l28_28640


namespace evaluate_expression_l28_28288

-- Definition of variables a, b, c as given in conditions
def a : ℕ := 7
def b : ℕ := 11
def c : ℕ := 13

-- The theorem to prove the given expression equals 31
theorem evaluate_expression : 
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 31 :=
by
  sorry

end evaluate_expression_l28_28288


namespace maximum_rabbits_l28_28391

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l28_28391


namespace smallest_four_digit_divisible_by_53_l28_28097

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28097


namespace parabola_intersection_length_l28_28832

theorem parabola_intersection_length
  (parabola_eqn : ∀ x y : ℝ, y^2 = 8 * x)
  (focus : (ℝ × ℝ) := (2, 0))
  (A : (ℝ × ℝ) := (1, 2 * Real.sqrt 2))
  (l : ℝ → ℝ := fun x ↔ x - 2)
  (intersects : ∀ F : (ℝ × ℝ), F = focus → ∃ B : ℝ × ℝ, B ∈ parabola_eqn ∧ B ∈ l)
  (distance_AB : ∀ A B : (ℝ × ℝ), A = (1, 2 * Real.sqrt 2) ∧ exists_B B, |AB| = (Real.sqrt((B.1 - A.1)^2 + (B.2 - A.2)^2)) = 9) :
  True :=
begin
  sorry
end

end parabola_intersection_length_l28_28832


namespace smallest_four_digit_divisible_by_53_l28_28020

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28020


namespace difference_of_squares_153_147_l28_28663

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l28_28663


namespace probability_of_boys_and_girls_l28_28629

def total_outcomes := Nat.choose 7 4
def only_boys_outcomes := Nat.choose 4 4
def both_boys_and_girls_outcomes := total_outcomes - only_boys_outcomes
def probability := both_boys_and_girls_outcomes / total_outcomes

theorem probability_of_boys_and_girls :
  probability = 34 / 35 :=
by
  sorry

end probability_of_boys_and_girls_l28_28629


namespace interest_time_period_l28_28773

-- Define the constants given in the problem
def principal : ℝ := 4000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def interest_difference : ℝ := 480

-- Define the time period T
def time_period : ℝ := 2

-- Define a proof statement
theorem interest_time_period :
  (principal * rate1 * time_period) - (principal * rate2 * time_period) = interest_difference :=
by {
  -- We skip the proof since it's not required by the problem statement
  sorry
}

end interest_time_period_l28_28773


namespace A_contribution_is_500_l28_28497

-- Define the contributions
variables (A B C : ℕ)

-- Total amount spent
def total_contribution : ℕ := 820

-- Given ratios
def ratio_A_to_B : ℕ × ℕ := (5, 2)
def ratio_B_to_C : ℕ × ℕ := (5, 3)

-- Condition stating the sum of contributions
axiom sum_contribution : A + B + C = total_contribution

-- Conditions stating the ratios
axiom ratio_A_B : 5 * B = 2 * A
axiom ratio_B_C : 5 * C = 3 * B

-- The statement to prove
theorem A_contribution_is_500 : A = 500 :=
by
  sorry

end A_contribution_is_500_l28_28497


namespace smallest_four_digit_divisible_by_53_l28_28034

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28034


namespace depth_of_water_l28_28128

variable (RonHeight DepthOfWater : ℝ)

-- Definitions based on conditions
def RonStandingHeight := 12 -- Ron's height is 12 feet
def DepthOfWaterCalculation := 5 * RonStandingHeight -- Depth is 5 times Ron's height

-- Theorem statement to prove
theorem depth_of_water (hRon : RonHeight = RonStandingHeight) (hDepth : DepthOfWater = DepthOfWaterCalculation) :
  DepthOfWater = 60 := by
  sorry

end depth_of_water_l28_28128


namespace number_of_nonsimilar_triangles_l28_28171
-- Import the necessary library

-- Define the problem conditions
def angles_in_arithmetic_progression (a d : ℕ) : Prop :=
  0 < d ∧ d < 30 ∧ 
  (a - d > 0) ∧ (a + d < 180) ∧ -- Ensures positive and valid angles
  (a - d) + a + (a + d) = 180  -- Triangle sum property

-- Declare the theorem
theorem number_of_nonsimilar_triangles : 
  ∃ n : ℕ, n = 29 ∧ ∀ (a d : ℕ), angles_in_arithmetic_progression a d → d < 30 → a = 60 :=
sorry

end number_of_nonsimilar_triangles_l28_28171


namespace total_missing_keys_l28_28340

theorem total_missing_keys :
  let total_vowels := 5
  let total_consonants := 21
  let missing_consonants := total_consonants / 7
  let missing_vowels := 2
  missing_consonants + missing_vowels = 5 :=
by {
  sorry
}

end total_missing_keys_l28_28340


namespace maximum_fly_path_length_in_box_l28_28280

theorem maximum_fly_path_length_in_box
  (length width height : ℝ)
  (h_length : length = 1)
  (h_width : width = 1)
  (h_height : height = 2) :
  ∃ l, l = (Real.sqrt 6 + 2 * Real.sqrt 5 + Real.sqrt 2 + 1) :=
by
  sorry

end maximum_fly_path_length_in_box_l28_28280


namespace polynomial_reciprocal_derivative_sum_zero_l28_28726

noncomputable def polynomial_recip_sum_zero (P : Polynomial ℝ) (n : ℕ) (roots : Fin n → ℝ) (h_deg : P.degree = (n : with_bot ℕ)) (h_distinct : ∀ i j, i ≠ j → roots i ≠ roots j) (h_roots : ∀ i, P.eval (roots i) = 0) : Prop :=
  ∑ i, 1 / (P.derivative.eval (roots i)) = 0

theorem polynomial_reciprocal_derivative_sum_zero :
  ∀ (P : Polynomial ℝ) (n : ℕ), n > 1 →
  ∀ (roots : Fin n → ℝ),
  (P.degree = (n : with_bot ℕ)) →
  (∀ i j, i ≠ j → roots i ≠ roots j) →
  (∀ i, P.eval (roots i) = 0) →
  ∑ i, 1 / (P.derivative.eval (roots i)) = 0 :=
sorry

end polynomial_reciprocal_derivative_sum_zero_l28_28726


namespace frogs_per_fish_per_day_l28_28287

theorem frogs_per_fish_per_day
  (f g n F : ℕ)
  (h1 : f = 30)
  (h2 : g = 15)
  (h3 : n = 9)
  (h4 : F = 32400) :
  F / f / (n * g) = 8 := by
  sorry

end frogs_per_fish_per_day_l28_28287


namespace score_after_7_hours_l28_28331

theorem score_after_7_hours (score : ℕ) (time : ℕ) : 
  (score / time = 90 / 5) → time = 7 → score = 126 :=
by
  sorry

end score_after_7_hours_l28_28331


namespace inequality_abc_sum_one_l28_28114

theorem inequality_abc_sum_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 1) :
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 +
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 +
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 +
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end inequality_abc_sum_one_l28_28114


namespace gina_snake_mice_in_decade_l28_28680

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end gina_snake_mice_in_decade_l28_28680


namespace smallest_fraction_numerator_l28_28499

theorem smallest_fraction_numerator :
  ∃ a b : ℕ, 10 ≤ a ∧ a < b ∧ b ≤ 99 ∧ (a : ℚ) / b > 5 / 6 ∧ a = 81 :=
by
  sorry

end smallest_fraction_numerator_l28_28499


namespace avg_monthly_bill_over_6_months_l28_28632

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end avg_monthly_bill_over_6_months_l28_28632


namespace determine_a_l28_28318

-- Define the sets A and B
def A : Set ℝ := { -1, 0, 2 }
def B (a : ℝ) : Set ℝ := { 2^a }

-- State the main theorem
theorem determine_a (a : ℝ) (h : B a ⊆ A) : a = 1 :=
by
  sorry

end determine_a_l28_28318


namespace minimize_quadratic_function_l28_28458

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l28_28458


namespace solution_set_of_inequality_l28_28238

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x : ℝ | 3 * a < x ∧ x < -a} :=
by
  sorry

end solution_set_of_inequality_l28_28238


namespace opposite_of_neg_two_thirds_l28_28592

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l28_28592


namespace find_s_l28_28639

noncomputable def area_of_parallelogram (s : ℝ) : ℝ :=
  (3 * s) * (s * Real.sin (Real.pi / 3))

theorem find_s (s : ℝ) (h1 : area_of_parallelogram s = 27 * Real.sqrt 3) : s = 3 * Real.sqrt 2 := 
  sorry

end find_s_l28_28639


namespace max_rabbits_with_long_ears_and_jumping_far_l28_28410

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l28_28410


namespace smallest_four_digit_divisible_by_53_l28_28041

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28041


namespace intersection_M_N_l28_28202

noncomputable def M : Set ℝ := { x | x^2 ≤ x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 < x ∧ x ≤ 1 } :=
  sorry

end intersection_M_N_l28_28202


namespace product_of_roots_l28_28673

theorem product_of_roots : ∀ (x : ℝ), (x + 3) * (x - 4) = 2 * (x + 1) → 
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  product_of_roots = -14 :=
by
  intros x h
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  sorry

end product_of_roots_l28_28673


namespace total_cost_is_63_l28_28652

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end total_cost_is_63_l28_28652


namespace inner_rectangle_length_l28_28643

def inner_rect_width : ℕ := 2

def second_rect_area (x : ℕ) : ℕ := 6 * (x + 4)

def largest_rect_area (x : ℕ) : ℕ := 10 * (x + 8)

def shaded_area_1 (x : ℕ) : ℕ := second_rect_area x - 2 * x

def shaded_area_2 (x : ℕ) : ℕ := largest_rect_area x - second_rect_area x

def in_arithmetic_progression (a b c : ℕ) : Prop := b - a = c - b

theorem inner_rectangle_length (x : ℕ) :
  in_arithmetic_progression (2 * x) (shaded_area_1 x) (shaded_area_2 x) → x = 4 := by
  intros
  sorry

end inner_rectangle_length_l28_28643


namespace nancy_crayons_l28_28870

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end nancy_crayons_l28_28870


namespace students_dont_eat_lunch_l28_28782

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end students_dont_eat_lunch_l28_28782


namespace mean_visits_between_200_and_300_l28_28654

def monday_visits := 300
def tuesday_visits := 400
def wednesday_visits := 300
def thursday_visits := 200
def friday_visits := 200

def total_visits := monday_visits + tuesday_visits + wednesday_visits + thursday_visits + friday_visits
def number_of_days := 5
def mean_visits_per_day := total_visits / number_of_days

theorem mean_visits_between_200_and_300 : 200 ≤ mean_visits_per_day ∧ mean_visits_per_day ≤ 300 :=
by sorry

end mean_visits_between_200_and_300_l28_28654


namespace smallest_four_digit_divisible_by_53_l28_28062

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28062


namespace proof_problem_l28_28863

variables {a b c d : ℝ} (h1 : a ≠ -2) (h2 : b ≠ -2) (h3 : c ≠ -2) (h4 : d ≠ -2)
variable (ω : ℂ) (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
variable (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω)

theorem proof_problem : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 :=
sorry

end proof_problem_l28_28863


namespace at_least_one_worker_must_wait_l28_28677

/-- 
Given five workers who collectively have a salary of 1500 rubles, 
and each tape recorder costs 320 rubles, we need to prove that 
at least one worker will not be able to buy a tape recorder immediately. 
-/
theorem at_least_one_worker_must_wait 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (tape_recorder_cost : ℕ) 
  (h_workers : num_workers = 5) 
  (h_salary : total_salary = 1500) 
  (h_cost : tape_recorder_cost = 320) :
  ∀ (tape_recorders_required : ℕ), 
    tape_recorders_required = num_workers → total_salary < tape_recorder_cost * tape_recorders_required → ∃ (k : ℕ), 1 ≤ k ∧ k ≤ num_workers ∧ total_salary < k * tape_recorder_cost :=
by 
  intros tape_recorders_required h_required h_insufficient
  sorry

end at_least_one_worker_must_wait_l28_28677


namespace ratio_of_areas_l28_28628

theorem ratio_of_areas (r : ℝ) (h : r > 0) :
  let R1 := r
  let R2 := 3 * r
  let S1 := 6 * R1
  let S2 := 6 * r
  let area_smaller_circle := π * R2 ^ 2
  let area_larger_square := S2 ^ 2
  (area_smaller_circle / area_larger_square) = π / 4 :=
by
  sorry

end ratio_of_areas_l28_28628


namespace calc_delta_l28_28705

noncomputable def delta (a b : ℝ) : ℝ :=
  (a^2 + b^2) / (1 + a * b)

-- Definition of the main problem as a Lean 4 statement
theorem calc_delta (h1 : 2 > 0) (h2 : 3 > 0) (h3 : 4 > 0) :
  delta (delta 2 3) 4 = 6661 / 2891 :=
by
  sorry

end calc_delta_l28_28705


namespace percent_increase_is_equivalent_l28_28551

variable {P : ℝ}

theorem percent_increase_is_equivalent 
  (h1 : 1.0 + 15.0 / 100.0 = 1.15)
  (h2 : 1.15 * (1.0 + 25.0 / 100.0) = 1.4375)
  (h3 : 1.4375 * (1.0 + 10.0 / 100.0) = 1.58125) :
  (1.58125 - 1) * 100 = 58.125 :=
by
  sorry

end percent_increase_is_equivalent_l28_28551


namespace sum_of_smallest_and_largest_l28_28578

def even_consecutive_sequence_sum (a n : ℤ) : ℤ :=
  a + a + 2 * (n - 1)

def arithmetic_mean (a n : ℤ) : ℤ :=
  (a * n + n * (n - 1)) / n

theorem sum_of_smallest_and_largest (a n y : ℤ) (h_even : even n) (h_mean : y = arithmetic_mean a n) :
  even_consecutive_sequence_sum a n = 2 * y :=
by
  sorry

end sum_of_smallest_and_largest_l28_28578


namespace carrots_eaten_after_dinner_l28_28853

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end carrots_eaten_after_dinner_l28_28853


namespace g_minus_6_eq_neg_20_l28_28890

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l28_28890


namespace rhombus_perimeter_l28_28735

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l28_28735


namespace simplify_expression_correct_l28_28361

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l28_28361


namespace max_quotient_l28_28846

theorem max_quotient (a b : ℕ) (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 1200 ≤ b) (h₄ : b ≤ 2400) :
  b / a ≤ 24 :=
sorry

end max_quotient_l28_28846


namespace vector_line_equation_l28_28427

open Real

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let numer := (u.1 * v.1 + u.2 * v.2)
  let denom := (v.1 * v.1 + v.2 * v.2)
  (numer * v.1 / denom, numer * v.2 / denom)

theorem vector_line_equation (x y : ℝ) :
  vector_projection (x, y) (3, 4) = (-3, -4) → 
  y = -3 / 4 * x - 25 / 4 :=
  sorry

end vector_line_equation_l28_28427


namespace surface_area_small_prism_l28_28800

-- Definitions and conditions
variables (a b c : ℝ)

def small_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * a * b + 2 * a * c + 2 * b * c

def large_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * (3 * b) * (3 * b) + 2 * (3 * b) * (4 * c) + 2 * (4 * c) * (3 * b)

-- Conditions
def conditions : Prop :=
  (3 * b = 2 * a) ∧ (a = 3 * c) ∧ (large_cuboid_surface_area a b c = 360)

-- Desired result
def result : Prop :=
  small_cuboid_surface_area a b c = 88

-- The theorem
theorem surface_area_small_prism (a b c : ℝ) (h : conditions a b c) : result a b c :=
by
  sorry

end surface_area_small_prism_l28_28800


namespace max_rabbits_with_traits_l28_28404

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l28_28404


namespace minimum_value_of_quad_func_l28_28838

def quad_func (x : ℝ) : ℝ :=
  2 * x^2 - 8 * x + 15

theorem minimum_value_of_quad_func :
  (∀ x : ℝ, quad_func 2 ≤ quad_func x) ∧ (quad_func 2 = 7) :=
by
  -- sorry to skip proof
  sorry

end minimum_value_of_quad_func_l28_28838


namespace gina_snake_mice_in_decade_l28_28681

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end gina_snake_mice_in_decade_l28_28681


namespace liquor_and_beer_cost_l28_28989

-- Define the variables and conditions
variables (p_liquor p_beer : ℕ)

-- Main theorem to prove
theorem liquor_and_beer_cost (h1 : 2 * p_liquor + 12 * p_beer = 56)
                             (h2 : p_liquor = 8 * p_beer) :
  p_liquor + p_beer = 18 :=
sorry

end liquor_and_beer_cost_l28_28989


namespace inequality_holds_for_all_x_y_l28_28882

theorem inequality_holds_for_all_x_y (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ x + y + x * y := 
by sorry

end inequality_holds_for_all_x_y_l28_28882


namespace sqrt_difference_calc_l28_28277

theorem sqrt_difference_calc: 
  sqrt 27 - sqrt (1 / 3) = (8 * sqrt 3) / 3 := 
by 
  sorry

end sqrt_difference_calc_l28_28277


namespace point_translation_l28_28189

variable (P Q : (ℝ × ℝ))
variable (dx : ℝ) (dy : ℝ)

theorem point_translation (hP : P = (-1, 2)) (hdx : dx = 2) (hdy : dy = 3) :
  Q = (P.1 + dx, P.2 - dy) → Q = (1, -1) := by
  sorry

end point_translation_l28_28189


namespace max_soap_boxes_l28_28618

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

end max_soap_boxes_l28_28618


namespace no_tangent_line_l28_28745

-- Define the function f(x) = x^3 - 3ax
def f (a x : ℝ) : ℝ := x^3 - 3 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

-- Proposition stating no b exists in ℝ such that y = -x + b is tangent to f
theorem no_tangent_line (a : ℝ) (H : ∀ b : ℝ, ¬ ∃ x : ℝ, f' a x = -1) : a < 1 / 3 :=
by
  sorry

end no_tangent_line_l28_28745


namespace smallest_four_digit_divisible_by_53_l28_28038

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l28_28038


namespace find_money_of_Kent_l28_28647

variable (Alison Brittany Brooke Kent : ℝ)

def money_relations (h1 : Alison = 4000)
    (h2 : Alison = Brittany / 2)
    (h3 : Brittany = 4 * Brooke)
    (h4 : Brooke = 2 * Kent) : Prop :=
  Kent = 1000

theorem find_money_of_Kent
  {Alison Brittany Brooke Kent : ℝ}
  (h1 : Alison = 4000)
  (h2 : Alison = Brittany / 2)
  (h3 : Brittany = 4 * Brooke)
  (h4 : Brooke = 2 * Kent) :
  money_relations Alison Brittany Brooke Kent h1 h2 h3 h4 :=
by 
  sorry

end find_money_of_Kent_l28_28647


namespace least_value_xy_l28_28154

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l28_28154


namespace increase_by_one_unit_l28_28150

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 + 3 * x

-- State the theorem
theorem increase_by_one_unit (x : ℝ) : regression_eq (x + 1) - regression_eq x = 3 := by
  sorry

end increase_by_one_unit_l28_28150


namespace compute_diff_squares_l28_28661

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l28_28661


namespace solution_to_inequality_l28_28510

theorem solution_to_inequality (x : ℝ) :
  (∃ y : ℝ, y = x^(1/3) ∧ y + 3 / (y + 2) ≤ 0) ↔ x < -8 := 
sorry

end solution_to_inequality_l28_28510


namespace sequence_contradiction_l28_28543

open Classical

variable {α : Type} (a : ℕ → α) [PartialOrder α]

theorem sequence_contradiction {a : ℕ → ℝ} :
  (∀ n, a n < 2) ↔ ¬ ∃ k, a k ≥ 2 := 
by sorry

end sequence_contradiction_l28_28543


namespace mouse_to_cheese_in_expected_steps_l28_28607

noncomputable def expected_steps (p_A_to_B p_B_to_A p_B_to_C p_C_to_B p_C_to_Cheese : ℝ) : ℝ :=
  1 / (1 - (1 - p_B_to_C * p_C_to_Cheese) * p_B_to_C) * (3 * p_B_to_C * p_C_to_Cheese 
  + 2 * p_B_to_C * (1 - p_C_to_Cheese) * p_C_to_Cheese / (1 - (1 - p_B_to_C * p_C_to_Cheese) * p_B_to_C))

theorem mouse_to_cheese_in_expected_steps : 
  expected_steps 1 (1/2) (1/2) (4/5) (1/5) = 21 := 
  by sorry

end mouse_to_cheese_in_expected_steps_l28_28607


namespace four_pow_minus_a_l28_28166

noncomputable def log_base_3 (x : ℝ) := Real.log x / Real.log 3

theorem four_pow_minus_a {a : ℝ} (h : a * log_base_3 4 = 2) : 4^(-a) = 1 / 9 :=
by
  sorry

end four_pow_minus_a_l28_28166


namespace boat_distance_against_stream_in_one_hour_l28_28336

-- Define the conditions
def speed_in_still_water : ℝ := 4 -- speed of the boat in still water (km/hr)
def downstream_distance_in_one_hour : ℝ := 6 -- distance traveled along the stream in one hour (km)

-- Define the function to compute the speed of the stream
def speed_of_stream (downstream_distance : ℝ) (boat_speed_still_water : ℝ) : ℝ :=
  downstream_distance - boat_speed_still_water

-- Define the effective speed against the stream
def effective_speed_against_stream (boat_speed_still_water : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed_still_water - stream_speed

-- Prove that the boat travels 2 km against the stream in one hour given the conditions
theorem boat_distance_against_stream_in_one_hour :
  effective_speed_against_stream speed_in_still_water (speed_of_stream downstream_distance_in_one_hour speed_in_still_water) * 1 = 2 := 
by
  sorry

end boat_distance_against_stream_in_one_hour_l28_28336


namespace smallest_four_digit_divisible_by_53_l28_28096

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l28_28096


namespace count_divisible_by_35_l28_28173

theorem count_divisible_by_35 : 
  ∃! (n : ℕ), n = 13 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 ∧ (∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ ab = 10 * a + b) →
    (ab * 100 + 35) % 35 = 0 ↔ ab % 7 = 0) :=
by {
  sorry
}

end count_divisible_by_35_l28_28173


namespace speed_conversion_l28_28668

theorem speed_conversion (v : ℚ) (h : v = 9/36) : v * 3.6 = 0.9 := by
  sorry

end speed_conversion_l28_28668


namespace total_boxes_correct_l28_28775

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l28_28775


namespace smallest_n_for_divisibility_l28_28438

theorem smallest_n_for_divisibility : ∃ n: ℕ, (n > 0) ∧ (n^2 % 24 = 0) ∧ (n^3 % 864 = 0) ∧ ∀ m : ℕ, 
  (m > 0) ∧ (m^2 % 24 = 0) ∧ (m^3 % 864 = 0) → (12 ≤ m) :=
begin
  sorry
end

end smallest_n_for_divisibility_l28_28438


namespace no_lunch_students_l28_28780

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end no_lunch_students_l28_28780


namespace sue_votes_correct_l28_28425

def total_votes : ℕ := 1000
def percentage_others : ℝ := 0.65
def sue_votes : ℕ := 350

theorem sue_votes_correct :
  sue_votes = (total_votes : ℝ) * (1 - percentage_others) :=
by
  sorry

end sue_votes_correct_l28_28425


namespace distinct_parenthesizations_of_3_3_3_3_l28_28808

theorem distinct_parenthesizations_of_3_3_3_3 : 
  ∃ (v1 v2 v3 v4 v5 : ℕ), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5 ∧ 
    v1 = 3 ^ (3 ^ (3 ^ 3)) ∧ 
    v2 = 3 ^ ((3 ^ 3) ^ 3) ∧ 
    v3 = (3 ^ 3) ^ (3 ^ 3) ∧ 
    v4 = ((3 ^ 3) ^ 3) ^ 3 ∧ 
    v5 = 3 ^ (27 ^ 27) :=
  sorry

end distinct_parenthesizations_of_3_3_3_3_l28_28808


namespace avg_height_of_class_is_168_6_l28_28583

noncomputable def avgHeightClass : ℕ → ℕ → ℕ → ℕ → ℚ :=
  λ n₁ h₁ n₂ h₂ => (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂)

theorem avg_height_of_class_is_168_6 :
  avgHeightClass 40 169 10 167 = 168.6 := 
by 
  sorry

end avg_height_of_class_is_168_6_l28_28583


namespace geometric_sequence_12th_term_l28_28421

theorem geometric_sequence_12th_term 
  (a_4 a_8 : ℕ) (h4 : a_4 = 2) (h8 : a_8 = 162) :
  ∃ a_12 : ℕ, a_12 = 13122 :=
by
  sorry

end geometric_sequence_12th_term_l28_28421


namespace concurrency_of_lines_l28_28873

noncomputable def radius (center : Point) (circle : Circle) : Real := sorry
noncomputable def distance_to_side (point : Point) (side : Line) : Real := sorry

variable (A B C Oa Ob Oc : Point)
variable (ra rb rc : ℝ)
variable (triangle : Triangle)
variable (func_side_BC func_side_CA func_side_AB : Point → ℝ)
variable (circle_tangent_equiv_1 : ∀ X, tangent_to_same_side_BC -> func_side_BC X = ra)
variable (circle_tangent_equiv_2 : ∀ X, tangent_to_same_side_CA -> func_side_CA X = rb)
variable (circle_tangent_equiv_3 : ∀ X, tangent_to_same_side_AB -> func_side_AB X = rc)
variable (prop_relation_1 : distance_to_side Ob (side BC) / rb = distance_to_side Oa (side CA) / ra)
variable (prop_relation_2 : distance_to_side Oc (side AB) / rc = distance_to_side Ob (side BC) / rb)
variable (prop_relation_3 : distance_to_side Oa (side CA) / ra = distance_to_side Oc (side AB) / rc)

theorem concurrency_of_lines :
  let d_a (X : Point) := distance_to_side X (side BC)
  let d_b (X : Point) := distance_to_side X (side CA)
  let d_c (X : Point) := distance_to_side X (side AB) in
  d_a Ob / rb * d_b Oc / rc * d_c Oa / ra = 1 →
  ∃ P : Point, collinear P A Oa ∧ collinear P B Ob ∧ collinear P C Oc :=
begin
  sorry
end

end concurrency_of_lines_l28_28873


namespace factory_produces_11250_products_l28_28976

noncomputable def total_products (refrigerators_per_hour coolers_per_hour hours_per_day days : ℕ) : ℕ :=
  (refrigerators_per_hour + coolers_per_hour) * (hours_per_day * days)

theorem factory_produces_11250_products :
  total_products 90 (90 + 70) 9 5 = 11250 := by
  sorry

end factory_produces_11250_products_l28_28976


namespace efficiency_ratio_l28_28246

theorem efficiency_ratio (A B : ℝ) (h1 : A + B = 1 / 26) (h2 : B = 1 / 39) : A / B = 1 / 2 := 
by
  sorry

end efficiency_ratio_l28_28246


namespace Bruce_paid_l28_28801

noncomputable def total_paid : ℝ :=
  let grapes_price := 9 * 70 * (1 - 0.10)
  let mangoes_price := 7 * 55 * (1 - 0.05)
  let oranges_price := 5 * 45 * (1 - 0.15)
  let apples_price := 3 * 80 * (1 - 0.20)
  grapes_price + mangoes_price + oranges_price + apples_price

theorem Bruce_paid (h : total_paid = 1316.25) : true :=
by
  -- This is where the proof would be
  sorry

end Bruce_paid_l28_28801


namespace beef_weight_loss_l28_28124

theorem beef_weight_loss (weight_before weight_after: ℕ) 
                         (h1: weight_before = 400) 
                         (h2: weight_after = 240) : 
                         ((weight_before - weight_after) * 100 / weight_before = 40) :=
by 
  sorry

end beef_weight_loss_l28_28124


namespace problem_solution_l28_28590

noncomputable def dodecahedron_probability := 
  let m := 1
  let n := 100
  m + n

theorem problem_solution : dodecahedron_probability = 101 := by
  sorry

end problem_solution_l28_28590


namespace totalInitialAmount_l28_28145

variable (a j t k x : ℝ)

-- Given conditions
def initialToyAmount : Prop :=
  t = 48

def kimRedistribution : Prop :=
  k = 4 * x - 144

def amyRedistribution : Prop :=
  (a = 3 * x) ∧ (j = 2 * x) ∧ (t = 2 * x)

def janRedistribution : Prop :=
  (a = 3 * x) ∧ (t = 4 * x)

def toyRedistribution : Prop :=
  (a = 6 * x) ∧ (j = -6 * x) ∧ (t = 48) 

def toyFinalAmount : Prop :=
  t = 48

-- Proof Problem
theorem totalInitialAmount
  (h1 : initialToyAmount t)
  (h2 : kimRedistribution k x)
  (h3 : amyRedistribution a j t x)
  (h4 : janRedistribution a t x)
  (h5 : toyRedistribution a j t x)
  (h6 : toyFinalAmount t) :
  a + j + t + k = 192 :=
sorry

end totalInitialAmount_l28_28145


namespace bus_weight_conversion_l28_28432

noncomputable def round_to_nearest (x : ℚ) : ℤ := Int.floor (x + 0.5)

theorem bus_weight_conversion (kg_to_pound : ℚ) (bus_weight_kg : ℚ) 
  (h : kg_to_pound = 0.4536) (h_bus : bus_weight_kg = 350) : 
  round_to_nearest (bus_weight_kg / kg_to_pound) = 772 := by
  sorry

end bus_weight_conversion_l28_28432


namespace simplify_fraction_l28_28374

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28374


namespace chords_in_circle_l28_28214

theorem chords_in_circle (n : ℕ) (h_n : n = 10) : (n.choose 2) = 45 := sorry

end chords_in_circle_l28_28214


namespace find_temperature_on_December_25_l28_28571

theorem find_temperature_on_December_25 {f : ℕ → ℤ}
  (h_recurrence : ∀ n, f (n - 1) + f (n + 1) = f n)
  (h_initial1 : f 3 = 5)
  (h_initial2 : f 31 = 2) :
  f 25 = -3 :=
  sorry

end find_temperature_on_December_25_l28_28571


namespace perfect_square_of_seq_l28_28426

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 3, a n = 7 * a (n - 1) - a (n - 2)

theorem perfect_square_of_seq (a : ℕ → ℤ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, k * k = a n + 2 + a (n + 1) :=
sorry

end perfect_square_of_seq_l28_28426


namespace find_annual_pension_l28_28982

variable (P k x a b p q : ℝ) (h1 : k * Real.sqrt (x + a) = k * Real.sqrt x + p)
                                   (h2 : k * Real.sqrt (x + b) = k * Real.sqrt x + q)

theorem find_annual_pension (h_nonzero_proportionality_constant : k ≠ 0) 
(h_year_difference : a ≠ b) : 
P = (a * q ^ 2 - b * p ^ 2) / (2 * (b * p - a * q)) := 
by
  sorry

end find_annual_pension_l28_28982


namespace minimize_quadratic_l28_28466

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l28_28466


namespace two_digit_number_l28_28600

theorem two_digit_number (x y : Nat) : 
  10 * x + y = 10 * x + y := 
by 
  sorry

end two_digit_number_l28_28600


namespace smallest_four_digit_div_by_53_l28_28049

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28049


namespace minimize_quadratic_l28_28612

theorem minimize_quadratic : ∃ x : ℝ, x = -4 ∧ ∀ y : ℝ, x^2 + 8*x + 7 ≤ y^2 + 8*y + 7 :=
by 
  use -4
  sorry

end minimize_quadratic_l28_28612


namespace prime_condition_l28_28979

def is_prime (n : ℕ) : Prop := nat.prime n

theorem prime_condition (p : ℕ) (q : ℕ) (h_prime_p : is_prime p) (h_eq : p + 25 = q ^ 7) (h_prime_q : is_prime q) : p = 103 :=
sorry

end prime_condition_l28_28979


namespace school_badminton_rackets_l28_28627

theorem school_badminton_rackets :
  ∃ (x y : ℕ), x + y = 30 ∧ 50 * x + 40 * y = 1360 ∧ x = 16 ∧ y = 14 :=
by
  sorry

end school_badminton_rackets_l28_28627


namespace fraction_division_correct_l28_28944

theorem fraction_division_correct :
  (2 / 5) / 3 = 2 / 15 :=
by sorry

end fraction_division_correct_l28_28944


namespace stanley_run_walk_difference_l28_28884

theorem stanley_run_walk_difference :
  ∀ (ran walked : ℝ), ran = 0.4 → walked = 0.2 → ran - walked = 0.2 :=
by
  intros ran walked h_ran h_walk
  rw [h_ran, h_walk]
  norm_num

end stanley_run_walk_difference_l28_28884


namespace difference_in_perimeters_of_rectangles_l28_28384

theorem difference_in_perimeters_of_rectangles 
  (l h : ℝ) (hl : l ≥ 0) (hh : h ≥ 0) :
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  difference = 24 :=
by
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  sorry

end difference_in_perimeters_of_rectangles_l28_28384


namespace square_of_1023_l28_28997

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l28_28997


namespace students_per_class_l28_28700

theorem students_per_class :
  let buns_per_package := 8
  let packages := 30
  let buns_per_student := 2
  let classes := 4
  (packages * buns_per_package) / (buns_per_student * classes) = 30 :=
by
  sorry

end students_per_class_l28_28700


namespace minimum_distance_l28_28148

section MinimumDistance
open Real

noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := 2 * sqrt x
def t (x1 x2 : ℝ) := f x1 = g x2
def d (x1 x2 : ℝ) := abs (x2 - x1)

theorem minimum_distance : ∃ (x1 x2 : ℝ), t x1 x2 ∧ d x1 x2 = (1 - log 2) / 2 := 
sorry

end MinimumDistance

end minimum_distance_l28_28148


namespace repair_cost_l28_28249

theorem repair_cost
  (R : ℝ) -- R is the cost to repair the used shoes
  (new_shoes_cost : ℝ := 30) -- New shoes cost $30.00
  (new_shoes_lifetime : ℝ := 2) -- New shoes last for 2 years
  (percentage_increase : ℝ := 42.857142857142854) 
  (h1 : new_shoes_cost / new_shoes_lifetime = R + (percentage_increase / 100) * R) :
  R = 10.50 :=
by
  sorry

end repair_cost_l28_28249


namespace smallest_four_digit_divisible_by_53_l28_28068

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28068


namespace airlines_routes_l28_28937

open Function

theorem airlines_routes
  (n_regions m_regions : ℕ)
  (h_n_regions : n_regions = 18)
  (h_m_regions : m_regions = 10)
  (A B : Fin n_regions → Fin n_regions → Bool)
  (h_flight : ∀ r1 r2 : Fin n_regions, r1 ≠ r2 → (A r1 r2 = true ∨ B r1 r2 = true) ∧ ¬(A r1 r2 = true ∧ B r1 r2 = true)) :
  ∃ (routes_A routes_B : List (List (Fin n_regions))),
    (∀ route ∈ routes_A, 2 ∣ route.length) ∧
    (∀ route ∈ routes_B, 2 ∣ route.length) ∧
    routes_A ≠ [] ∧
    routes_B ≠ [] :=
sorry

end airlines_routes_l28_28937


namespace g_minus_6_eq_neg_20_l28_28888

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l28_28888


namespace total_cost_of_aquarium_l28_28650

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end total_cost_of_aquarium_l28_28650


namespace linear_system_substitution_correct_l28_28820

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end linear_system_substitution_correct_l28_28820


namespace find_m_l28_28223

theorem find_m {m : ℝ} :
  (∃ x y : ℝ, y = x + 1 ∧ y = -x ∧ y = mx + 3) → m = 5 :=
by
  sorry

end find_m_l28_28223


namespace veranda_area_l28_28222

theorem veranda_area (length_room width_room width_veranda : ℕ)
  (h_length : length_room = 20) 
  (h_width : width_room = 12) 
  (h_veranda : width_veranda = 2) : 
  (length_room + 2 * width_veranda) * (width_room + 2 * width_veranda) - (length_room * width_room) = 144 := 
by
  sorry

end veranda_area_l28_28222


namespace simplify_fraction_l28_28376

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l28_28376


namespace find_blue_weights_l28_28988

theorem find_blue_weights (B : ℕ) :
  (2 * B + 15 + 2 = 25) → B = 4 :=
by
  intro h
  sorry

end find_blue_weights_l28_28988


namespace charles_picked_50_pears_l28_28589

variable (P B S : ℕ)

theorem charles_picked_50_pears 
  (cond1 : S = B + 10)
  (cond2 : B = 3 * P)
  (cond3 : S = 160) : 
  P = 50 := by
  sorry

end charles_picked_50_pears_l28_28589


namespace tangent_line_at_e_l28_28829

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + real.log (-x) else -(-x + real.log x)

theorem tangent_line_at_e :
  f e = e - 1 →
  ∀ (x : ℝ), f x = -f (-x) →
  ∀ (x < 0), f x = x + real.log (-x) →
  ∃ m b : ℝ, (∀ x : ℝ, y = m * x + b) ∧ m = 1 - 1 / e :=
sorry

end tangent_line_at_e_l28_28829


namespace value_of_m_squared_plus_reciprocal_squared_l28_28844

theorem value_of_m_squared_plus_reciprocal_squared 
  (m : ℝ) 
  (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 :=
by {
  sorry
}

end value_of_m_squared_plus_reciprocal_squared_l28_28844


namespace smallest_four_digit_divisible_by_53_l28_28083

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l28_28083


namespace regular_polygon_sides_l28_28489

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 12) : n = 30 := 
by
  sorry

end regular_polygon_sides_l28_28489


namespace door_cranking_time_l28_28196

-- Define the given conditions
def run_time_with_backpack : ℝ := 7 * 60 + 23  -- 443 seconds
def run_time_without_backpack : ℝ := 5 * 60 + 58  -- 358 seconds
def total_time : ℝ := 874  -- 874 seconds

-- Define the Lean statement of the proof
theorem door_cranking_time :
  (run_time_with_backpack + run_time_without_backpack) + (total_time - (run_time_with_backpack + run_time_without_backpack)) = total_time ∧
  (total_time - (run_time_with_backpack + run_time_without_backpack)) = 73 :=
by
  sorry

end door_cranking_time_l28_28196


namespace construct_segment_length_l28_28163

theorem construct_segment_length (a b : ℝ) (h : a > b) : 
  ∃ c : ℝ, c = (a^2 + b^2) / (a - b) :=
by
  sorry

end construct_segment_length_l28_28163


namespace smallest_four_digit_divisible_by_53_l28_28056

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l28_28056


namespace minimize_f_l28_28442

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l28_28442


namespace smallest_four_digit_div_by_53_l28_28053

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l28_28053


namespace minimize_f_at_3_l28_28449

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l28_28449


namespace num_zeros_in_binary_l28_28507

namespace BinaryZeros

def expression : ℕ := ((18 * 8192 + 8 * 128 - 12 * 16) / 6) + (4 * 64) + (3 ^ 5) - (25 * 2)

def binary_zeros (n : ℕ) : ℕ :=
  (Nat.digits 2 n).count 0

theorem num_zeros_in_binary :
  binary_zeros expression = 6 :=
by
  sorry

end BinaryZeros

end num_zeros_in_binary_l28_28507


namespace triangle_area_eq_l28_28827

noncomputable def areaOfTriangle (a b c A B C: ℝ): ℝ :=
1 / 2 * a * c * (Real.sin A)

theorem triangle_area_eq
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : A = Real.pi / 3)
  (h3 : Real.sqrt 3 / 2 - Real.sin (B - C) = Real.sin (2 * B)) :
  areaOfTriangle a b c A B C = Real.sqrt 3 ∨ areaOfTriangle a b c A B C = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end triangle_area_eq_l28_28827


namespace intersection_A_B_l28_28862

def A : Set ℝ := {x | abs x < 2}
def B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
  sorry

end intersection_A_B_l28_28862


namespace least_xy_l28_28157

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l28_28157


namespace solveInequalityRegion_l28_28865

noncomputable def greatestIntegerLessThan (x : ℝ) : ℤ :=
  Int.floor x

theorem solveInequalityRegion :
  ∀ (x y : ℝ), abs x < 1 → abs y < 1 → x * y ≠ 0 → (greatestIntegerLessThan (x + y) ≤ 
  greatestIntegerLessThan x + greatestIntegerLessThan y) :=
by
  intros x y h1 h2 h3
  sorry

end solveInequalityRegion_l28_28865


namespace sum_powers_of_5_mod_8_l28_28437

theorem sum_powers_of_5_mod_8 :
  (List.sum (List.map (fun n => (5^n % 8)) (List.range 2011))) % 8 = 4 := 
  sorry

end sum_powers_of_5_mod_8_l28_28437


namespace reptile_house_animal_multiple_l28_28380

theorem reptile_house_animal_multiple (R F x : ℕ) (hR : R = 16) (hF : F = 7) (hCond : R = x * F - 5) : x = 3 := by
  sorry

end reptile_house_animal_multiple_l28_28380


namespace steve_average_speed_l28_28619

/-
Problem Statement:
Prove that the average speed of Steve's travel for the entire journey is 55 mph given the following conditions:
1. Steve's first part of journey: 5 hours at 40 mph.
2. Steve's second part of journey: 3 hours at 80 mph.
-/

theorem steve_average_speed :
  let time1 := 5 -- hours
  let speed1 := 40 -- mph
  let time2 := 3 -- hours
  let speed2 := 80 -- mph
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 55 := by
  sorry

end steve_average_speed_l28_28619


namespace square_difference_l28_28658

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l28_28658


namespace determinant_property_l28_28309

variable {R : Type} [CommRing R]
variable (x y z w : R)

theorem determinant_property 
  (h : x * w - y * z = 7) :
  (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by sorry

end determinant_property_l28_28309


namespace cafeteria_B_turnover_higher_in_May_l28_28330

noncomputable def initial_turnover (X a r : ℝ) : Prop :=
  ∃ (X a r : ℝ),
    (X + 8 * a = X * (1 + r) ^ 8) ∧
    ((X + 4 * a) < (X * (1 + r) ^ 4))

theorem cafeteria_B_turnover_higher_in_May (X a r : ℝ) :
    (X + 8 * a = X * (1 + r) ^ 8) → (X + 4 * a < X * (1 + r) ^ 4) :=
  sorry

end cafeteria_B_turnover_higher_in_May_l28_28330


namespace central_angle_radian_measure_l28_28692

-- Definitions for the conditions
def circumference (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1/2) * l * r = 4
def radian_measure (l r θ : ℝ) : Prop := θ = l / r

-- Prove the radian measure of the central angle of the sector is 2
theorem central_angle_radian_measure (r l θ : ℝ) : 
  circumference r l → 
  area r l → 
  radian_measure l r θ → 
  θ = 2 :=
by
  sorry

end central_angle_radian_measure_l28_28692


namespace max_rabbits_with_traits_l28_28401

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l28_28401


namespace largest_angle_in_triangle_l28_28924

theorem largest_angle_in_triangle (x : ℝ) (h1 : 40 + 60 + x = 180) (h2 : max 40 60 ≤ x) : x = 80 :=
by
  -- Proof skipped
  sorry

end largest_angle_in_triangle_l28_28924


namespace bacteria_growth_rate_l28_28193

theorem bacteria_growth_rate (r : ℝ) :
  (1 + r)^6 = 64 → r = 1 :=
by
  intro h
  sorry

end bacteria_growth_rate_l28_28193


namespace smallest_four_digit_multiple_of_53_l28_28001

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l28_28001


namespace determine_n_l28_28306

theorem determine_n (n : ℕ) (h1 : n > 2020) (h2 : ∃ m : ℤ, (n - 2020) = m^2 * (2120 - n)) : 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := 
sorry

end determine_n_l28_28306


namespace find_r_l28_28291

theorem find_r (r : ℝ) (h : 5 * (r - 9) = 6 * (3 - 3 * r) + 6) : r = 3 :=
by
  sorry

end find_r_l28_28291


namespace smallest_four_digit_multiple_of_53_l28_28018

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l28_28018


namespace speed_with_stream_l28_28978

-- Define the given conditions
def V_m : ℝ := 7 -- Man's speed in still water (7 km/h)
def V_as : ℝ := 10 -- Man's speed against the stream (10 km/h)

-- Define the stream's speed as the difference
def V_s : ℝ := V_m - V_as

-- Define man's speed with the stream
def V_ws : ℝ := V_m + V_s

-- (Correct Answer): Prove the man's speed with the stream is 10 km/h
theorem speed_with_stream :
  V_ws = 10 := by
  -- Sorry for no proof required in this task
  sorry

end speed_with_stream_l28_28978


namespace g_neg_six_eq_neg_twenty_l28_28916

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l28_28916


namespace pascal_family_min_children_l28_28851

-- We define the conditions b >= 3 and g >= 2
def b_condition (b : ℕ) : Prop := b >= 3
def g_condition (g : ℕ) : Prop := g >= 2

-- We state that the smallest number of children given these conditions is 5
theorem pascal_family_min_children (b g : ℕ) (hb : b_condition b) (hg : g_condition g) : b + g = 5 :=
sorry

end pascal_family_min_children_l28_28851


namespace chords_on_circle_l28_28213

theorem chords_on_circle (n : ℕ) (h : n = 10) : nat.choose n 2 = 45 :=
by {
  rw h,
  -- we can directly calculate choose 10 2
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num),
  sorry -- the actual detailed proof goes here.
}

end chords_on_circle_l28_28213


namespace amy_points_per_treasure_l28_28648

theorem amy_points_per_treasure (treasures_first_level treasures_second_level total_score : ℕ) (h1 : treasures_first_level = 6) (h2 : treasures_second_level = 2) (h3 : total_score = 32) :
  total_score / (treasures_first_level + treasures_second_level) = 4 := by
  sorry

end amy_points_per_treasure_l28_28648


namespace ms_cole_total_students_l28_28566

def students_6th : ℕ := 40
def students_4th : ℕ := 4 * students_6th
def students_7th : ℕ := 2 * students_4th

def total_students : ℕ := students_6th + students_4th + students_7th

theorem ms_cole_total_students :
  total_students = 520 :=
by
  sorry

end ms_cole_total_students_l28_28566


namespace opposite_of_neg_two_thirds_l28_28591

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l28_28591


namespace find_initial_apples_l28_28928

theorem find_initial_apples (A : ℤ)
  (h1 : 6 * ((A / 8) + 8 - 30) = 12) :
  A = 192 :=
sorry

end find_initial_apples_l28_28928


namespace avg_monthly_bill_over_6_months_l28_28631

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end avg_monthly_bill_over_6_months_l28_28631


namespace solve_expression_l28_28175

theorem solve_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 :=
sorry

end solve_expression_l28_28175


namespace dave_won_tickets_l28_28504

theorem dave_won_tickets (initial_tickets spent_tickets final_tickets won_tickets : ℕ) 
  (h1 : initial_tickets = 25) 
  (h2 : spent_tickets = 22) 
  (h3 : final_tickets = 18) 
  (h4 : won_tickets = final_tickets - (initial_tickets - spent_tickets)) :
  won_tickets = 15 := 
by 
  sorry

end dave_won_tickets_l28_28504


namespace no_lunch_students_l28_28781

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end no_lunch_students_l28_28781


namespace rectangle_semi_perimeter_l28_28549

variables (BC AC AM x y : ℝ)

theorem rectangle_semi_perimeter (hBC : BC = 5) (hAC : AC = 12) (hAM : AM = x)
  (hMN_AC : ∀ (MN : ℝ), MN = 5 / 12 * AM)
  (hNP_BC : ∀ (NP : ℝ), NP = AC - AM)
  (hy_def : y = (5 / 12 * x) + (12 - x)) :
  y = (144 - 7 * x) / 12 :=
sorry

end rectangle_semi_perimeter_l28_28549


namespace correct_inequality_l28_28950

theorem correct_inequality (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > ab ∧ ab > a :=
sorry

end correct_inequality_l28_28950


namespace no_integer_roots_l28_28696

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end no_integer_roots_l28_28696


namespace max_rabbits_with_long_ears_and_jumping_far_l28_28406

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l28_28406


namespace find_g_minus_6_l28_28908

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l28_28908
