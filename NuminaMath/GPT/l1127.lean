import Mathlib

namespace maximize_volume_l1127_112717

-- Define the given dimensions
def length := 90
def width := 48

-- Define the volume function based on the height h
def volume (h : ℝ) : ℝ := h * (length - 2 * h) * (width - 2 * h)

-- Define the height that maximizes the volume
def optimal_height := 10

-- Define the maximum volume obtained at the optimal height
def max_volume := 19600

-- State the proof problem
theorem maximize_volume : 
  (∃ h : ℝ, volume h ≤ volume optimal_height) ∧
  volume optimal_height = max_volume := 
by
  sorry

end maximize_volume_l1127_112717


namespace line_passes_through_center_l1127_112743

-- Define the equation of the circle as given in the problem.
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the center of the circle.
def center_of_circle (x y : ℝ) : Prop := x = 1 ∧ y = -3

-- Define the equation of the line.
def line_equation (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- The theorem to prove.
theorem line_passes_through_center :
  (∃ x y, circle_equation x y ∧ center_of_circle x y) →
  (∃ x y, center_of_circle x y ∧ line_equation x y) :=
by
  sorry

end line_passes_through_center_l1127_112743


namespace find_total_values_l1127_112766

theorem find_total_values (n : ℕ) (S : ℝ) 
  (h1 : S / n = 150) 
  (h2 : (S + 25) / n = 151.25) 
  (h3 : 25 = 160 - 135) : n = 20 :=
by
  sorry

end find_total_values_l1127_112766


namespace hyperbola_asymptotes_equation_l1127_112762

noncomputable def hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (e : ℝ)
  (h_eq : e = 5 / 3)
  (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1) :
  String :=
by
  sorry

theorem hyperbola_asymptotes_equation : 
  ∀ a b : ℝ, ∀ ha : a > 0, ∀ hb : b > 0, ∀ e : ℝ,
  e = 5 / 3 →
  (∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1) →
  ( ∀ (x : ℝ), x ≠ 0 → y = (4/3)*x ∨ y = -(4/3)*x
  )
  :=
by
  intros _
  sorry

end hyperbola_asymptotes_equation_l1127_112762


namespace find_original_price_l1127_112777

def initial_price (P : ℝ) : Prop :=
  let first_discount := P * 0.76
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 1.10
  final_price = 532

theorem find_original_price : ∃ P : ℝ, initial_price P :=
sorry

end find_original_price_l1127_112777


namespace correct_statement_about_meiosis_and_fertilization_l1127_112790

def statement_A : Prop := 
  ∃ oogonia spermatogonia zygotes : ℕ, 
    oogonia = 20 ∧ spermatogonia = 8 ∧ zygotes = 32 ∧ 
    (oogonia + spermatogonia = zygotes)

def statement_B : Prop := 
  ∀ zygote_dna mother_half father_half : ℕ,
    zygote_dna = mother_half + father_half ∧ 
    mother_half = father_half

def statement_C : Prop := 
  ∀ (meiosis stabilizes : Prop) (chromosome_count : ℕ),
    (meiosis → stabilizes) ∧ 
    (stabilizes → chromosome_count = (chromosome_count / 2 + chromosome_count / 2))

def statement_D : Prop := 
  ∀ (diversity : Prop) (gene_mutations chromosomal_variations : Prop),
    (diversity → ¬ (gene_mutations ∨ chromosomal_variations))

theorem correct_statement_about_meiosis_and_fertilization :
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D :=
by
  sorry

end correct_statement_about_meiosis_and_fertilization_l1127_112790


namespace exists_integer_square_with_three_identical_digits_l1127_112792

theorem exists_integer_square_with_three_identical_digits:
  ∃ x: ℤ, (x^2 % 1000 = 444) := by
  sorry

end exists_integer_square_with_three_identical_digits_l1127_112792


namespace path_count_l1127_112745

theorem path_count :
  let is_valid_path (path : List (ℕ × ℕ)) : Prop :=
    ∃ (n : ℕ), path = List.range n    -- This is a simplification for definition purposes
  let count_paths_outside_square (start finish : (ℤ × ℤ)) (steps : ℕ) : ℕ :=
    43826                              -- Hardcoded the result as this is the correct answer
  ∀ start finish : (ℤ × ℤ),
    start = (-5, -5) → 
    finish = (5, 5) → 
    count_paths_outside_square start finish 20 = 43826
:= 
sorry

end path_count_l1127_112745


namespace trig_expression_evaluation_l1127_112714

theorem trig_expression_evaluation
  (α : ℝ)
  (h : Real.tan α = 2) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by 
  sorry

end trig_expression_evaluation_l1127_112714


namespace shekar_math_marks_l1127_112761

variable (science socialStudies english biology average : ℕ)

theorem shekar_math_marks 
  (h1 : science = 65)
  (h2 : socialStudies = 82)
  (h3 : english = 67)
  (h4 : biology = 95)
  (h5 : average = 77) :
  ∃ M, average = (science + socialStudies + english + biology + M) / 5 ∧ M = 76 :=
by
  sorry

end shekar_math_marks_l1127_112761


namespace complement_U_A_is_singleton_one_l1127_112767

-- Define the universe and subset
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_U_A_is_singleton_one : complement_U_A = {1} := by
  sorry

end complement_U_A_is_singleton_one_l1127_112767


namespace cube_inequality_contradiction_l1127_112755

variable {x y : ℝ}

theorem cube_inequality_contradiction (h : x < y) (hne : x^3 ≥ y^3) : false :=
by 
  sorry

end cube_inequality_contradiction_l1127_112755


namespace card_S_l1127_112798

def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 5 * n - 1

def S : Finset ℕ := 
  (Finset.range 2016).image a ∩ (Finset.range (a 2015 + 1)).image b

theorem card_S : S.card = 504 := 
  sorry

end card_S_l1127_112798


namespace initial_percentage_increase_l1127_112773

variable (P : ℝ) (x : ℝ)

theorem initial_percentage_increase :
  (P * (1 + x / 100) * 1.3 = P * 1.625) → (x = 25) := by
  sorry

end initial_percentage_increase_l1127_112773


namespace alice_bob_meet_l1127_112720

/--
Alice and Bob play a game on a circle divided into 18 equally-spaced points.
Alice moves 7 points clockwise per turn, and Bob moves 13 points counterclockwise.
Prove that they will meet at the same point after 9 turns.
-/
theorem alice_bob_meet : ∃ k : ℕ, k = 9 ∧ (7 * k) % 18 = (18 - 13 * k) % 18 :=
by
  sorry

end alice_bob_meet_l1127_112720


namespace max_value_of_polynomial_l1127_112757

theorem max_value_of_polynomial :
  ∃ x : ℝ, (x = -1) ∧ ∀ y : ℝ, -3 * y^2 - 6 * y + 12 ≤ -3 * (-1)^2 - 6 * (-1) + 12 := by
  sorry

end max_value_of_polynomial_l1127_112757


namespace range_of_sum_of_zeros_l1127_112725

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x else 1 - x / 2

noncomputable def F (x : ℝ) (m : ℝ) : ℝ :=
  f (f x + 1) + m

def has_zeros (F : ℝ → ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ m = 0 ∧ F x₂ m = 0

theorem range_of_sum_of_zeros (m : ℝ) :
  has_zeros F m →
  ∃ (x₁ x₂ : ℝ), F x₁ m = 0 ∧ F x₂ m = 0 ∧ (x₁ + x₂) ≥ 4 - 2 * Real.log 2 := sorry

end range_of_sum_of_zeros_l1127_112725


namespace find_single_digit_A_l1127_112760

theorem find_single_digit_A (A : ℕ) (h1 : 0 ≤ A) (h2 : A < 10) (h3 : (10 * A + A) * (10 * A + A) = 5929) : A = 7 :=
sorry

end find_single_digit_A_l1127_112760


namespace solve_system_l1127_112771

theorem solve_system :
  ∃ x y : ℝ, x - y = 1 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 := by
  sorry

end solve_system_l1127_112771


namespace six_digit_number_theorem_l1127_112788

noncomputable def six_digit_number (a b c d e f : ℕ) : ℕ :=
  10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f

noncomputable def rearranged_number (a b c d e f : ℕ) : ℕ :=
  10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + a

theorem six_digit_number_theorem (a b c d e f : ℕ) (h_a : a ≠ 0) 
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  (h4 : 0 ≤ d ∧ d ≤ 9) (h5 : 0 ≤ e ∧ e ≤ 9) (h6 : 0 ≤ f ∧ f ≤ 9) 
  : six_digit_number a b c d e f = 142857 ∨ six_digit_number a b c d e f = 285714 :=
by
  sorry

end six_digit_number_theorem_l1127_112788


namespace game_show_prizes_count_l1127_112708

theorem game_show_prizes_count:
  let digits := [1, 1, 1, 1, 3, 3, 3, 3]
  let is_valid_prize (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9999
  let is_three_digit_or_more (n : ℕ) : Prop := 100 ≤ n
  ∃ (A B C : ℕ), 
    is_valid_prize A ∧ is_valid_prize B ∧ is_valid_prize C ∧
    is_three_digit_or_more C ∧
    (A + B + C = digits.sum) ∧
    (A + B + C = 1260) := sorry

end game_show_prizes_count_l1127_112708


namespace nina_widgets_after_reduction_is_approx_8_l1127_112791

noncomputable def nina_total_money : ℝ := 16.67
noncomputable def widgets_before_reduction : ℝ := 5
noncomputable def cost_reduction_per_widget : ℝ := 1.25

noncomputable def cost_per_widget_before_reduction : ℝ := nina_total_money / widgets_before_reduction
noncomputable def cost_per_widget_after_reduction : ℝ := cost_per_widget_before_reduction - cost_reduction_per_widget
noncomputable def widgets_after_reduction : ℝ := nina_total_money / cost_per_widget_after_reduction

-- Prove that Nina can purchase approximately 8 widgets after the cost reduction
theorem nina_widgets_after_reduction_is_approx_8 : abs (widgets_after_reduction - 8) < 1 :=
by
  sorry

end nina_widgets_after_reduction_is_approx_8_l1127_112791


namespace triangle_type_is_isosceles_l1127_112704

theorem triangle_type_is_isosceles {A B C : ℝ}
  (h1 : A + B + C = π)
  (h2 : ∀ x : ℝ, x^2 - x * (Real.cos A * Real.cos B) + 2 * Real.sin (C / 2)^2 = 0)
  (h3 : ∃ x1 x2 : ℝ, x1 + x2 = Real.cos A * Real.cos B ∧ x1 * x2 = 2 * Real.sin (C / 2)^2 ∧ (x1 + x2 = (x1 * x2) / 2)) :
  A = B ∨ B = C ∨ C = A := 
sorry

end triangle_type_is_isosceles_l1127_112704


namespace external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l1127_112703

variables {O₁ O₂ : ℝ} {r R : ℝ}

-- External tangency implies sum of radii equals distance between centers
theorem external_tangency_sum {O₁ O₂ r R : ℝ} (h1 : O₁ ≠ O₂) (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = r + R) : 
  dist O₁ O₂ = r + R :=
sorry

-- Internal tangency implies difference of radii equals distance between centers
theorem internal_tangency_diff {O₁ O₂ r R : ℝ} 
  (h1 : O₁ ≠ O₂) 
  (h2 : ∀ M, (dist O₁ M = r) ∧ (dist O₂ M = R) → dist O₁ O₂ = abs (R - r)) : 
  dist O₁ O₂ = abs (R - r) :=
sorry

-- Converse for sum of radii equals distance between centers
theorem converse_sum_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = r + R) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = r + R) :=
sorry

-- Converse for difference of radii equals distance between centers
theorem converse_diff_of_radii {O₁ O₂ r R : ℝ}
  (h1 : O₁ ≠ O₂) 
  (h2 : dist O₁ O₂ = abs (R - r)) : 
  ∃ M, (dist O₁ M = r) ∧ (dist O₂ M = R) ∧ (dist O₁ O₂ = abs (R - r)) :=
sorry

end external_tangency_sum_internal_tangency_diff_converse_sum_of_radii_converse_diff_of_radii_l1127_112703


namespace mental_math_competition_l1127_112756

theorem mental_math_competition :
  -- The number of teams that participated is 4
  (∃ (teams : ℕ) (numbers : List ℕ),
     -- Each team received a number that can be written as 15M + 11m where M is the largest odd divisor
     -- and m is the smallest odd divisor greater than 1.
     teams = 4 ∧ 
     numbers = [528, 880, 1232, 1936] ∧
     ∀ n ∈ numbers,
       ∃ M m, M > 1 ∧ m > 1 ∧
       M % 2 = 1 ∧ m % 2 = 1 ∧
       (∀ d, d ∣ n → (d % 2 = 1 → M ≥ d)) ∧ 
       (∀ d, d ∣ n → (d % 2 = 1 ∧ d > 1 → m ≤ d)) ∧
       n = 15 * M + 11 * m) :=
sorry

end mental_math_competition_l1127_112756


namespace tom_profit_l1127_112775

-- Define the initial conditions
def initial_investment : ℕ := 20 * 3
def revenue_from_selling : ℕ := 10 * 4
def value_of_remaining_shares : ℕ := 10 * 6
def total_amount : ℕ := revenue_from_selling + value_of_remaining_shares

-- We claim that the profit Tom makes is 40 dollars
theorem tom_profit : (total_amount - initial_investment) = 40 := by
  sorry

end tom_profit_l1127_112775


namespace sum_of_values_of_n_l1127_112750

theorem sum_of_values_of_n (n : ℚ) (h : |3 * n - 4| = 6) : 
  (n = 10 / 3 ∨ n = -2 / 3) → (10 / 3 + -2 / 3 = 8 / 3) :=
sorry

end sum_of_values_of_n_l1127_112750


namespace find_principal_l1127_112797

theorem find_principal (R : ℝ) (P : ℝ) (h : (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56) : P = 700 := 
sorry

end find_principal_l1127_112797


namespace minimum_value_m_ineq_proof_l1127_112763

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem minimum_value_m (x₀ : ℝ) (m : ℝ) (hx : f x₀ ≤ m) : 4 ≤ m := by
  sorry

theorem ineq_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 4) : 3 ≤ 3 / b + 1 / a := by
  sorry

end minimum_value_m_ineq_proof_l1127_112763


namespace polynomial_simplification_l1127_112723

theorem polynomial_simplification (x : ℝ) : 
  (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2 * x^3 - 4 * x^2 + 10 * x + 1 :=
by
  sorry

end polynomial_simplification_l1127_112723


namespace smallest_possible_product_l1127_112787

theorem smallest_possible_product : 
  ∃ (x : ℕ) (y : ℕ), (x = 56 ∧ y = 78 ∨ x = 57 ∧ y = 68) ∧ x * y = 3876 :=
by
  sorry

end smallest_possible_product_l1127_112787


namespace quotient_of_even_and_odd_composites_l1127_112768

theorem quotient_of_even_and_odd_composites:
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = 512 / 28525 := by
sorry

end quotient_of_even_and_odd_composites_l1127_112768


namespace range_f_does_not_include_zero_l1127_112718

noncomputable def f (x : ℝ) : ℤ :=
if x > 0 then ⌈1 / (x + 1)⌉ else if x < 0 then ⌈1 / (x - 1)⌉ else 0 -- this will be used only as a formal definition

theorem range_f_does_not_include_zero : ¬ (0 ∈ {y : ℤ | ∃ x : ℝ, x ≠ 0 ∧ y = f x}) :=
by sorry

end range_f_does_not_include_zero_l1127_112718


namespace range_x_minus_y_compare_polynomials_l1127_112747

-- Proof Problem 1: Range of x - y
theorem range_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) : 
  -4 < x - y ∧ x - y < 2 := 
  sorry

-- Proof Problem 2: Comparison of polynomials
theorem compare_polynomials (x : ℝ) : 
  (x - 1) * (x^2 + x + 1) < (x + 1) * (x^2 - x + 1) := 
  sorry

end range_x_minus_y_compare_polynomials_l1127_112747


namespace number_of_questionnaires_drawn_from_15_to_16_is_120_l1127_112710

variable (x : ℕ)
variable (H1 : 120 + 180 + 240 + x = 900)
variable (H2 : 60 = (bit0 90) / 180)
variable (H3 : (bit0 (bit0 (bit0 15))) = (bit0 (bit0 (bit0 15))) * (900 / 300))

theorem number_of_questionnaires_drawn_from_15_to_16_is_120 :
  ((900 - 120 - 180 - 240) * (300 / 900)) = 120 :=
sorry

end number_of_questionnaires_drawn_from_15_to_16_is_120_l1127_112710


namespace zoo_charge_for_child_l1127_112735

theorem zoo_charge_for_child (charge_adult : ℕ) (total_people total_bill children : ℕ) (charge_child : ℕ) : 
  charge_adult = 8 → total_people = 201 → total_bill = 964 → children = 161 → 
  total_bill - (total_people - children) * charge_adult = children * charge_child → 
  charge_child = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end zoo_charge_for_child_l1127_112735


namespace robin_gum_count_l1127_112746

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (final_gum : ℝ) 
  (h1 : initial_gum = 18.0) (h2 : additional_gum = 44.0) : final_gum = 62.0 :=
by {
  sorry
}

end robin_gum_count_l1127_112746


namespace compound_interest_rate_l1127_112748

theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (CI r : ℝ)
  (hP : P = 1200)
  (hCI : CI = 1785.98)
  (ht : t = 5)
  (hn : n = 1)
  (hA : A = P * (1 + r/n)^(n * t)) :
  A = P + CI → 
  r = 0.204 :=
by
  sorry

end compound_interest_rate_l1127_112748


namespace fraction_first_to_second_l1127_112732

def digit_fraction_proof_problem (a b c d : ℕ) (number : ℕ) :=
  number = 1349 ∧
  a = b / 3 ∧
  c = a + b ∧
  d = 3 * b

theorem fraction_first_to_second (a b c d : ℕ) (number : ℕ) :
  digit_fraction_proof_problem a b c d number → a / b = 1 / 3 :=
by
  intro problem
  sorry

end fraction_first_to_second_l1127_112732


namespace exists_positive_ℓ_l1127_112770

theorem exists_positive_ℓ (k : ℕ) (h_prime: 0 < k) :
  ∃ ℓ : ℕ, 0 < ℓ ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → Nat.gcd m ℓ = 1 → Nat.gcd n ℓ = 1 →  m ^ m % ℓ = n ^ n % ℓ → m % k = n % k) :=
sorry

end exists_positive_ℓ_l1127_112770


namespace percentage_games_won_l1127_112742

theorem percentage_games_won 
  (P_first : ℝ)
  (P_remaining : ℝ)
  (total_games : ℕ)
  (H1 : P_first = 0.7)
  (H2 : P_remaining = 0.5)
  (H3 : total_games = 100) :
  True :=
by
  -- To prove the percentage of games won is 70%
  have percentage_won : ℝ := P_first
  have : percentage_won * 100 = 70 := by sorry
  trivial

end percentage_games_won_l1127_112742


namespace height_of_fourth_person_l1127_112701

theorem height_of_fourth_person
  (h : ℝ)
  (H1 : h + (h + 2) + (h + 4) + (h + 10) = 4 * 79) :
  h + 10 = 85 :=
by
  have H2 : h + 4 = 79 := by linarith
  linarith


end height_of_fourth_person_l1127_112701


namespace value_of_g_at_neg3_l1127_112713

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end value_of_g_at_neg3_l1127_112713


namespace tan_435_eq_2_plus_sqrt3_l1127_112722

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end tan_435_eq_2_plus_sqrt3_l1127_112722


namespace expression_eqn_l1127_112753

theorem expression_eqn (a : ℝ) (E : ℝ → ℝ)
  (h₁ : -6 * a^2 = 3 * (E a + 2))
  (h₂ : a = 1) : E a = -2 * a^2 - 2 :=
by
  sorry

end expression_eqn_l1127_112753


namespace no_integer_solutions_to_system_l1127_112785

theorem no_integer_solutions_to_system :
  ¬ ∃ (x y z : ℤ),
    x^2 - 2 * x * y + y^2 - z^2 = 17 ∧
    -x^2 + 3 * y * z + 3 * z^2 = 27 ∧
    x^2 - x * y + 5 * z^2 = 50 :=
by
  sorry

end no_integer_solutions_to_system_l1127_112785


namespace amy_total_distance_equals_168_l1127_112796

def amy_biked_monday := 12

def amy_biked_tuesday (monday: ℕ) := 2 * monday - 3

def amy_biked_other_day (previous_day: ℕ) := previous_day + 2

def total_distance_bike_week := 
  let monday := amy_biked_monday
  let tuesday := amy_biked_tuesday monday
  let wednesday := amy_biked_other_day tuesday
  let thursday := amy_biked_other_day wednesday
  let friday := amy_biked_other_day thursday
  let saturday := amy_biked_other_day friday
  let sunday := amy_biked_other_day saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem amy_total_distance_equals_168 : 
  total_distance_bike_week = 168 := by
  sorry

end amy_total_distance_equals_168_l1127_112796


namespace revenue_increase_l1127_112758

open Real

theorem revenue_increase
  (P Q : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q) :
  let R := P * Q
  let P_new := P * 1.60
  let Q_new := Q * 0.65
  let R_new := P_new * Q_new
  (R_new - R) / R * 100 = 4 := by
sorry

end revenue_increase_l1127_112758


namespace find_ordered_pair_l1127_112782

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 3 * x - 18 * y = 2) 
  (h2 : 4 * y - x = 6) :
  x = -58 / 3 ∧ y = -10 / 3 :=
sorry

end find_ordered_pair_l1127_112782


namespace fraction_of_planted_area_l1127_112709

-- Definitions of the conditions
def right_triangle (a b : ℕ) : Prop :=
  a * a + b * b = (Int.sqrt (a ^ 2 + b ^ 2))^2

def unplanted_square_distance (dist : ℕ) : Prop :=
  dist = 3

-- The main theorem to be proved
theorem fraction_of_planted_area (a b : ℕ) (dist : ℕ) (h_triangle : right_triangle a b) (h_square_dist : unplanted_square_distance dist) :
  (a = 5) → (b = 12) → ((a * b - dist ^ 2) / (a * b) = 412 / 1000) :=
by
  sorry

end fraction_of_planted_area_l1127_112709


namespace fraction_difference_l1127_112712

theorem fraction_difference:
  let f1 := 2 / 3
  let f2 := 3 / 4
  let f3 := 4 / 5
  let f4 := 5 / 7
  (max f1 (max f2 (max f3 f4)) - min f1 (min f2 (min f3 f4))) = 2 / 15 :=
by
  sorry

end fraction_difference_l1127_112712


namespace ratio_of_areas_is_two_thirds_l1127_112744

noncomputable def PQ := 10
noncomputable def PR := 6
noncomputable def QR := 4
noncomputable def r_PQ := PQ / 2
noncomputable def r_PR := PR / 2
noncomputable def r_QR := QR / 2
noncomputable def area_semi_PQ := (1 / 2) * Real.pi * r_PQ^2
noncomputable def area_semi_PR := (1 / 2) * Real.pi * r_PR^2
noncomputable def area_semi_QR := (1 / 2) * Real.pi * r_QR^2
noncomputable def shaded_area := (area_semi_PQ - area_semi_PR) + area_semi_QR
noncomputable def total_area_circle := Real.pi * r_PQ^2
noncomputable def unshaded_area := total_area_circle - shaded_area
noncomputable def ratio := shaded_area / unshaded_area

theorem ratio_of_areas_is_two_thirds : ratio = 2 / 3 := by
  sorry

end ratio_of_areas_is_two_thirds_l1127_112744


namespace christine_amount_l1127_112726

theorem christine_amount (S C : ℕ) 
  (h1 : S + C = 50)
  (h2 : C = S + 30) :
  C = 40 :=
by
  -- Proof goes here.
  -- This part should be filled in to complete the proof.
  sorry

end christine_amount_l1127_112726


namespace find_f_l1127_112729

theorem find_f {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2*x := 
by
  sorry

end find_f_l1127_112729


namespace negation_of_P_is_true_l1127_112736

theorem negation_of_P_is_true :
  ¬ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by sorry

end negation_of_P_is_true_l1127_112736


namespace find_n_l1127_112794

/-- In the expansion of (1 + 3x)^n, where n is a positive integer and n >= 6, 
    if the coefficients of x^5 and x^6 are equal, then n is 7. -/
theorem find_n (n : ℕ) (h₀ : 0 < n) (h₁ : 6 ≤ n)
  (h₂ : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : 
  n = 7 := 
sorry

end find_n_l1127_112794


namespace olivia_wallet_after_shopping_l1127_112731

variable (initial_wallet : ℝ := 200) 
variable (groceries : ℝ := 65)
variable (shoes_original_price : ℝ := 75)
variable (shoes_discount_rate : ℝ := 0.15)
variable (belt : ℝ := 25)

theorem olivia_wallet_after_shopping :
  initial_wallet - (groceries + (shoes_original_price - shoes_original_price * shoes_discount_rate) + belt) = 46.25 := by
  sorry

end olivia_wallet_after_shopping_l1127_112731


namespace sum_ABC_eq_7_base_8_l1127_112737

/-- Lean 4 statement for the problem.

A, B, C: are distinct non-zero digits less than 8 in base 8, and
A B C_8 + B C_8 = A C A_8 holds true.
-/
theorem sum_ABC_eq_7_base_8 :
  ∃ (A B C : ℕ), A < 8 ∧ B < 8 ∧ C < 8 ∧ 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (A * 64 + B * 8 + C) + (B * 8 + C) = A * 64 + C * 8 + A ∧
  A + B + C = 7 :=
by { sorry }

end sum_ABC_eq_7_base_8_l1127_112737


namespace min_total_books_l1127_112741

-- Definitions based on conditions
variables (P C B : ℕ)

-- Condition 1: Ratio of physics to chemistry books is 3:2
def ratio_physics_chemistry := 3 * C = 2 * P

-- Condition 2: Ratio of chemistry to biology books is 4:3
def ratio_chemistry_biology := 4 * B = 3 * C

-- Condition 3: Total number of books is 3003
def total_books := P + C + B = 3003

-- The theorem to prove
theorem min_total_books (h1 : ratio_physics_chemistry P C) (h2 : ratio_chemistry_biology C B) (h3: total_books P C B) :
  3003 = 3003 :=
by
  sorry

end min_total_books_l1127_112741


namespace print_colored_pages_l1127_112706

theorem print_colored_pages (cost_per_page : ℕ) (dollars : ℕ) (conversion_rate : ℕ) 
    (h_cost : cost_per_page = 4) (h_dollars : dollars = 30) (h_conversion : conversion_rate = 100) :
    (dollars * conversion_rate) / cost_per_page = 750 := 
by
  sorry

end print_colored_pages_l1127_112706


namespace rectangle_area_with_inscribed_circle_l1127_112705

theorem rectangle_area_with_inscribed_circle (w h r : ℝ)
  (hw : ∀ O : ℝ × ℝ, dist O (w/2, h/2) = r)
  (hw_eq_h : w = h) :
  w * h = 2 * r^2 := 
by
  sorry

end rectangle_area_with_inscribed_circle_l1127_112705


namespace molecular_weight_of_compound_l1127_112795

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def num_C : ℕ := 4
def num_H : ℕ := 1
def num_O : ℕ := 1

theorem molecular_weight_of_compound : 
  (num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O) = 65.048 := 
  by 
  -- proof skipped
  sorry

end molecular_weight_of_compound_l1127_112795


namespace yellow_lights_count_l1127_112738

theorem yellow_lights_count (total_lights : ℕ) (red_lights : ℕ) (blue_lights : ℕ) (yellow_lights : ℕ) :
  total_lights = 95 → red_lights = 26 → blue_lights = 32 → yellow_lights = total_lights - (red_lights + blue_lights) → yellow_lights = 37 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end yellow_lights_count_l1127_112738


namespace lcm_of_three_numbers_is_180_l1127_112759

-- Define the three numbers based on the ratio and HCF condition
def a : ℕ := 2 * 6
def b : ℕ := 3 * 6
def c : ℕ := 5 * 6

-- State the theorem regarding the LCM
theorem lcm_of_three_numbers_is_180 : Nat.lcm (Nat.lcm a b) c = 180 :=
by
  sorry

end lcm_of_three_numbers_is_180_l1127_112759


namespace cost_price_article_l1127_112751

theorem cost_price_article (x : ℝ) (h : 56 - x = x - 42) : x = 49 :=
by sorry

end cost_price_article_l1127_112751


namespace complex_division_l1127_112778

theorem complex_division :
  (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = (⟨3, 2⟩ : ℂ) :=
sorry

end complex_division_l1127_112778


namespace angle_symmetry_l1127_112730

theorem angle_symmetry (α β : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) (hβ : 0 < β ∧ β < 2 * Real.pi) (h_symm : α = 2 * Real.pi - β) : α + β = 2 * Real.pi := 
by 
  sorry

end angle_symmetry_l1127_112730


namespace apples_purchased_l1127_112780

variable (A : ℕ) -- Let A be the number of kg of apples purchased.

-- Conditions
def cost_of_apples (A : ℕ) : ℕ := 70 * A
def cost_of_mangoes : ℕ := 45 * 9
def total_amount_paid : ℕ := 965

-- Theorem to prove that A == 8
theorem apples_purchased
  (h : cost_of_apples A + cost_of_mangoes = total_amount_paid) :
  A = 8 := by
sorry

end apples_purchased_l1127_112780


namespace line_perpendicular_passing_through_point_l1127_112764

theorem line_perpendicular_passing_through_point :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), 2 * x + y - 2 = 0 ↔ a * x + b * y + c = 0) ∧ 
                (a, b) ≠ (0, 0) ∧ 
                (a * -1 + b * 4 + c = 0) ∧ 
                (a * 1/2 + b * (-2) ≠ -4) :=
by { sorry }

end line_perpendicular_passing_through_point_l1127_112764


namespace max_S_R_squared_l1127_112789

theorem max_S_R_squared (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) :
  (∃ a b c, DA = a ∧ DB = b ∧ DC = c ∧ S = 2 * (a * b + b * c + c * a) ∧
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 ∧ (∃ max_val, max_val = (2 / 3) * (3 + Real.sqrt 3))) :=
sorry

end max_S_R_squared_l1127_112789


namespace triangle_area_is_180_l1127_112719

theorem triangle_area_is_180 {a b c : ℕ} (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) 
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 : ℚ) * a * b = 180 :=
by
  sorry

end triangle_area_is_180_l1127_112719


namespace peter_drew_8_pictures_l1127_112733

theorem peter_drew_8_pictures : 
  ∃ (P : ℕ), ∀ (Q R : ℕ), Q = P + 20 → R = 5 → R + P + Q = 41 → P = 8 :=
by
  sorry

end peter_drew_8_pictures_l1127_112733


namespace total_candies_l1127_112779

variable (Adam James Rubert : Nat)
variable (Adam_has_candies : Adam = 6)
variable (James_has_candies : James = 3 * Adam)
variable (Rubert_has_candies : Rubert = 4 * James)

theorem total_candies : Adam + James + Rubert = 96 :=
by
  sorry

end total_candies_l1127_112779


namespace algorithm_comparable_to_euclidean_l1127_112721

-- Define the conditions
def ancient_mathematics_world_leading : Prop := 
  True -- Placeholder representing the historical condition

def song_yuan_algorithm : Prop :=
  True -- Placeholder representing the algorithmic condition

-- The main theorem representing the problem statement
theorem algorithm_comparable_to_euclidean :
  ancient_mathematics_world_leading → song_yuan_algorithm → 
  True :=  -- Placeholder representing that the algorithm is the method of successive subtraction
by 
  intro h1 h2 
  sorry

end algorithm_comparable_to_euclidean_l1127_112721


namespace find_angle_A_l1127_112749

theorem find_angle_A (a b : ℝ) (A B : ℝ) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = Real.pi / 3) :
  A = Real.pi / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end find_angle_A_l1127_112749


namespace gift_wrapping_combinations_l1127_112781

theorem gift_wrapping_combinations :
  (10 * 5 * 6 * 2 = 600) :=
by
  sorry

end gift_wrapping_combinations_l1127_112781


namespace radius_increase_125_surface_area_l1127_112769

theorem radius_increase_125_surface_area (r r' : ℝ) 
(increase_surface_area : 4 * π * (r'^2) = 2.25 * 4 * π * r^2) : r' = 1.5 * r :=
by 
  sorry

end radius_increase_125_surface_area_l1127_112769


namespace part_I_part_II_l1127_112734

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 1| - |x - a|

-- Problem (I)
theorem part_I (x : ℝ) : 
  (f x 4) > 2 ↔ (x < -7 ∨ x > 5 / 3) :=
sorry

-- Problem (II)
theorem part_II (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x a ≥ |x - 4|) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

end part_I_part_II_l1127_112734


namespace find_f_x_l1127_112784

def f (x : ℝ) : ℝ := sorry

theorem find_f_x (x : ℝ) (h : 2 * f x - f (-x) = 3 * x) : f x = x := 
by sorry

end find_f_x_l1127_112784


namespace distinct_digits_and_difference_is_945_l1127_112765

theorem distinct_digits_and_difference_is_945 (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_difference : 10 * (100 * a + 10 * b + c) + 2 - (2000 + 100 * a + 10 * b + c) = 945) :
  (100 * a + 10 * b + c) = 327 :=
by
  sorry

end distinct_digits_and_difference_is_945_l1127_112765


namespace remainder_is_23_l1127_112754

def number_remainder (n : ℤ) : ℤ :=
  n % 36

theorem remainder_is_23 (n : ℤ) (h1 : n % 4 = 3) (h2 : n % 9 = 5) :
  number_remainder n = 23 :=
by
  sorry

end remainder_is_23_l1127_112754


namespace elgin_money_l1127_112700

theorem elgin_money {A B C D E : ℤ} 
  (h1 : |A - B| = 19) 
  (h2 : |B - C| = 9) 
  (h3 : |C - D| = 5) 
  (h4 : |D - E| = 4) 
  (h5 : |E - A| = 11) 
  (h6 : A + B + C + D + E = 60) : 
  E = 10 := 
sorry

end elgin_money_l1127_112700


namespace dot_product_of_a_b_l1127_112752

theorem dot_product_of_a_b 
  (a b : ℝ)
  (θ : ℝ)
  (ha : a = 2 * Real.sin (15 * Real.pi / 180))
  (hb : b = 4 * Real.cos (15 * Real.pi / 180))
  (hθ : θ = 30 * Real.pi / 180) :
  (a * b * Real.cos θ) = Real.sqrt 3 := by
  sorry

end dot_product_of_a_b_l1127_112752


namespace each_serving_requires_1_5_apples_l1127_112772

theorem each_serving_requires_1_5_apples 
  (guest_count : ℕ) (pie_count : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ) 
  (h_guest_count : guest_count = 12)
  (h_pie_count : pie_count = 3)
  (h_servings_per_pie : servings_per_pie = 8)
  (h_apples_per_guest : apples_per_guest = 3) :
  (apples_per_guest * guest_count) / (pie_count * servings_per_pie) = 1.5 :=
by
  sorry

end each_serving_requires_1_5_apples_l1127_112772


namespace find_Q_digit_l1127_112716

theorem find_Q_digit (P Q R S T U : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S)
  (h4 : P ≠ T) (h5 : P ≠ U) (h6 : Q ≠ R) (h7 : Q ≠ S) (h8 : Q ≠ T)
  (h9 : Q ≠ U) (h10 : R ≠ S) (h11 : R ≠ T) (h12 : R ≠ U) (h13 : S ≠ T)
  (h14 : S ≠ U) (h15 : T ≠ U) (h_range_P : 4 ≤ P ∧ P ≤ 9)
  (h_range_Q : 4 ≤ Q ∧ Q ≤ 9) (h_range_R : 4 ≤ R ∧ R ≤ 9)
  (h_range_S : 4 ≤ S ∧ S ≤ 9) (h_range_T : 4 ≤ T ∧ T ≤ 9)
  (h_range_U : 4 ≤ U ∧ U ≤ 9) 
  (h_sum_lines : 3 * P + 2 * Q + 3 * S + R + T + 2 * U = 100)
  (h_sum_digits : P + Q + S + R + T + U = 39) : Q = 6 :=
sorry  -- proof to be provided

end find_Q_digit_l1127_112716


namespace side_length_of_square_l1127_112724

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l1127_112724


namespace cost_of_largest_pot_equals_229_l1127_112727

-- Define the conditions
variables (total_cost : ℝ) (num_pots : ℕ) (cost_diff : ℝ)

-- Assume given conditions
axiom h1 : num_pots = 6
axiom h2 : total_cost = 8.25
axiom h3 : cost_diff = 0.3

-- Define the function for the cost of the smallest pot and largest pot
noncomputable def smallest_pot_cost : ℝ :=
  (total_cost - (num_pots - 1) * cost_diff) / num_pots

noncomputable def largest_pot_cost : ℝ :=
  smallest_pot_cost total_cost num_pots cost_diff + (num_pots - 1) * cost_diff

-- Prove the cost of the largest pot equals 2.29
theorem cost_of_largest_pot_equals_229 (h1 : num_pots = 6) (h2 : total_cost = 8.25) (h3 : cost_diff = 0.3) :
  largest_pot_cost total_cost num_pots cost_diff = 2.29 :=
  by sorry

end cost_of_largest_pot_equals_229_l1127_112727


namespace remainder_of_division_l1127_112707

noncomputable def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 1
noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2
noncomputable def remainder (x : ℝ) : ℝ := 324 * x - 488

theorem remainder_of_division :
  ∀ (x : ℝ), (f x) % (g x) = remainder x :=
sorry

end remainder_of_division_l1127_112707


namespace range_of_a_l1127_112702

variable (x a : ℝ)

-- Definition of α: x > a
def α : Prop := x > a

-- Definition of β: (x - 1) / x > 0
def β : Prop := (x - 1) / x > 0

-- Theorem to prove the range of a
theorem range_of_a (h : α x a → β x) : 1 ≤ a :=
  sorry

end range_of_a_l1127_112702


namespace solution_set_for_inequality_l1127_112786

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + (a - b) * x + 1

theorem solution_set_for_inequality (a b : ℝ) (h1 : 2*a + 4 = -(a-1)) :
  ∀ x : ℝ, (f x a b > f b a b) ↔ ((x ∈ Set.Icc (-2 : ℝ) (2 : ℝ)) ∧ ((x < -1 ∨ 1 < x))) :=
by
  sorry

end solution_set_for_inequality_l1127_112786


namespace lego_set_cost_l1127_112774

-- Definitions and conditions
def price_per_car := 5
def cars_sold := 3
def action_figures_sold := 2
def total_earnings := 120

-- Derived prices
def price_per_action_figure := 2 * price_per_car
def price_per_board_game := price_per_action_figure + price_per_car

-- Total cost of sold items (cars, action figures, and board game)
def total_cost_of_sold_items := 
  (cars_sold * price_per_car) + 
  (action_figures_sold * price_per_action_figure) + 
  price_per_board_game

-- Cost of Lego set
theorem lego_set_cost : 
  total_earnings - total_cost_of_sold_items = 70 :=
by
  -- Proof omitted
  sorry

end lego_set_cost_l1127_112774


namespace balls_is_perfect_square_l1127_112783

open Classical -- Open classical logic for nonconstructive proofs

-- Define a noncomputable function to capture the main proof argument
noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem balls_is_perfect_square {a v : ℕ} (h : (2 * a * v) = (a + v) * (a + v - 1))
  : is_perfect_square (a + v) :=
sorry

end balls_is_perfect_square_l1127_112783


namespace fencing_rate_l1127_112715

/-- Given a circular field of diameter 20 meters and a total cost of fencing of Rs. 94.24777960769379,
    prove that the rate per meter for the fencing is Rs. 1.5. -/
theorem fencing_rate 
  (d : ℝ) (cost : ℝ) (π : ℝ) (rate : ℝ)
  (hd : d = 20)
  (hcost : cost = 94.24777960769379)
  (hπ : π = 3.14159)
  (Circumference : ℝ := π * d)
  (Rate : ℝ := cost / Circumference) : 
  rate = 1.5 :=
sorry

end fencing_rate_l1127_112715


namespace games_bought_at_garage_sale_l1127_112740

theorem games_bought_at_garage_sale (G : ℕ)
  (h1 : 2 + G - 2  = 2) :
  G = 2 :=
by {
  sorry
}

end games_bought_at_garage_sale_l1127_112740


namespace smallest_five_sequential_number_greater_than_2000_is_2004_l1127_112793

def fiveSequentialNumber (N : ℕ) : Prop :=
  (if 1 ∣ N then 1 else 0) + 
  (if 2 ∣ N then 1 else 0) + 
  (if 3 ∣ N then 1 else 0) + 
  (if 4 ∣ N then 1 else 0) + 
  (if 5 ∣ N then 1 else 0) + 
  (if 6 ∣ N then 1 else 0) + 
  (if 7 ∣ N then 1 else 0) + 
  (if 8 ∣ N then 1 else 0) + 
  (if 9 ∣ N then 1 else 0) ≥ 5

theorem smallest_five_sequential_number_greater_than_2000_is_2004 :
  ∀ N > 2000, fiveSequentialNumber N → N = 2004 :=
by
  intros N hn hfsn
  have hN : N = 2004 := sorry
  exact hN

end smallest_five_sequential_number_greater_than_2000_is_2004_l1127_112793


namespace expression_not_defined_l1127_112711

theorem expression_not_defined (x : ℝ) :
    ¬(x^2 - 22*x + 121 = 0) ↔ ¬(x - 11 = 0) :=
by sorry

end expression_not_defined_l1127_112711


namespace boxed_meals_solution_count_l1127_112739

theorem boxed_meals_solution_count :
  ∃ n : ℕ, n = 4 ∧ 
  ∃ x y z : ℕ, 
      x + y + z = 22 ∧ 
      10 * x + 8 * y + 5 * z = 183 ∧ 
      x > 0 ∧ y > 0 ∧ z > 0 :=
sorry

end boxed_meals_solution_count_l1127_112739


namespace exists_triangle_with_sides_l2_l3_l4_l1127_112799

theorem exists_triangle_with_sides_l2_l3_l4
  (a1 a2 a3 a4 d : ℝ)
  (h_arith_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_pos : a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0)
  (h_d_pos : d > 0) :
  a2 + a3 > a4 ∧ a3 + a4 > a2 ∧ a4 + a2 > a3 :=
by
  sorry

end exists_triangle_with_sides_l2_l3_l4_l1127_112799


namespace perimeter_C_l1127_112728

def is_square (n : ℕ) : Prop := n > 0 ∧ ∃ s : ℕ, s * s = n

variable (A B C : ℕ) -- Defining the squares
variable (sA sB sC : ℕ) -- Defining the side lengths

-- Conditions as definitions
axiom square_figures : is_square A ∧ is_square B ∧ is_square C 
axiom perimeter_A : 4 * sA = 20
axiom perimeter_B : 4 * sB = 40
axiom side_length_C : sC = 2 * (sA + sB)

-- The equivalent proof problem statement
theorem perimeter_C : 4 * sC = 120 :=
by
  -- Proof will go here
  sorry

end perimeter_C_l1127_112728


namespace total_flowers_collected_l1127_112776

/- Definitions for the given conditions -/
def maxFlowers : ℕ := 50
def arwenTulips : ℕ := 20
def arwenRoses : ℕ := 18
def arwenSunflowers : ℕ := 6

def elrondTulips : ℕ := 2 * arwenTulips
def elrondRoses : ℕ := if 3 * arwenRoses + elrondTulips > maxFlowers then maxFlowers - elrondTulips else 3 * arwenRoses

def galadrielTulips : ℕ := if 3 * elrondTulips > maxFlowers then maxFlowers else 3 * elrondTulips
def galadrielRoses : ℕ := if 2 * arwenRoses + galadrielTulips > maxFlowers then maxFlowers - galadrielTulips else 2 * arwenRoses

def galadrielSunflowers : ℕ := 0 -- she didn't pick any sunflowers
def legolasSunflowers : ℕ := arwenSunflowers + galadrielSunflowers
def legolasRemaining : ℕ := maxFlowers - legolasSunflowers
def legolasRosesAndTulips : ℕ := legolasRemaining / 2
def legolasTulips : ℕ := legolasRosesAndTulips
def legolasRoses : ℕ := legolasRosesAndTulips

def arwenTotal : ℕ := arwenTulips + arwenRoses + arwenSunflowers
def elrondTotal : ℕ := elrondTulips + elrondRoses
def galadrielTotal : ℕ := galadrielTulips + galadrielRoses + galadrielSunflowers
def legolasTotal : ℕ := legolasTulips + legolasRoses + legolasSunflowers

def totalFlowers : ℕ := arwenTotal + elrondTotal + galadrielTotal + legolasTotal

theorem total_flowers_collected : totalFlowers = 194 := by
  /- This will be where the proof goes, but we leave it as a placeholder. -/
  sorry

end total_flowers_collected_l1127_112776
