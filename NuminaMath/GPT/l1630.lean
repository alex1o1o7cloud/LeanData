import Mathlib

namespace percentage_difference_l1630_163033

theorem percentage_difference : (0.4 * 60 - (4/5 * 25)) = 4 := by
  sorry

end percentage_difference_l1630_163033


namespace ratio_of_women_working_in_retail_l1630_163048

-- Define the population of Los Angeles
def population_LA : ℕ := 6000000

-- Define the proportion of women in Los Angeles
def half_population : ℕ := population_LA / 2

-- Define the number of women working in retail
def women_retail : ℕ := 1000000

-- Define the total number of women in Los Angeles
def total_women : ℕ := half_population

-- The statement to be proven:
theorem ratio_of_women_working_in_retail :
  (women_retail / total_women : ℚ) = 1 / 3 :=
by {
  -- The proof goes here
  sorry
}

end ratio_of_women_working_in_retail_l1630_163048


namespace prove_additional_minutes_needed_l1630_163077

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l1630_163077


namespace kangaroo_arrangement_count_l1630_163005

theorem kangaroo_arrangement_count :
  let k := 8
  let tallest_at_ends := 2
  let middle := k - tallest_at_ends
  (tallest_at_ends * (middle.factorial)) = 1440 := by
  sorry

end kangaroo_arrangement_count_l1630_163005


namespace series_sum_eq_l1630_163000

noncomputable def series_sum : Real :=
  ∑' n : ℕ, (4 * (n + 1) + 1) / (((4 * (n + 1) - 1) ^ 3) * ((4 * (n + 1) + 3) ^ 3))

theorem series_sum_eq : series_sum = 1 / 5184 := sorry

end series_sum_eq_l1630_163000


namespace smallest_other_number_l1630_163065

theorem smallest_other_number (x : ℕ)  (h_pos : 0 < x) (n : ℕ)
  (h_gcd : Nat.gcd 60 n = x + 3)
  (h_lcm : Nat.lcm 60 n = x * (x + 3)) :
  n = 45 :=
sorry

end smallest_other_number_l1630_163065


namespace intersecting_line_l1630_163036

theorem intersecting_line {x y : ℝ} (h1 : x^2 + y^2 = 10) (h2 : (x - 1)^2 + (y - 3)^2 = 10) :
  x + 3 * y - 5 = 0 :=
sorry

end intersecting_line_l1630_163036


namespace isosceles_triangle_EF_length_l1630_163063

theorem isosceles_triangle_EF_length (DE DF EF DK EK KF : ℝ)
  (h1 : DE = 5) (h2 : DF = 5) (h3 : DK^2 + EK^2 = DE^2) (h4 : DK^2 + KF^2 = EF^2)
  (h5 : EK + KF = EF) (h6 : EK = 4 * KF) :
  EF = Real.sqrt 10 :=
by sorry

end isosceles_triangle_EF_length_l1630_163063


namespace sum_reciprocals_square_l1630_163046

theorem sum_reciprocals_square (x y : ℕ) (h : x * y = 11) : (1 : ℚ) / (↑x ^ 2) + (1 : ℚ) / (↑y ^ 2) = 122 / 121 :=
by
  sorry

end sum_reciprocals_square_l1630_163046


namespace max_tickets_l1630_163089

theorem max_tickets (cost : ℝ) (budget : ℝ) (max_tickets : ℕ) (h1 : cost = 15.25) (h2 : budget = 200) :
  max_tickets = 13 :=
by
  sorry

end max_tickets_l1630_163089


namespace sequence_periodic_a_n_plus_2_eq_a_n_l1630_163008

-- Definition of the sequence and conditions
noncomputable def seq (a : ℕ → ℤ) :=
  ∀ n : ℕ, ∃ α k : ℕ, a n = Int.ofNat (2^α) * k ∧ Int.gcd (Int.ofNat k) 2 = 1 ∧ a (n+1) = Int.ofNat (2^α) - k

-- Definition of periodic sequence
def periodic (a : ℕ → ℤ) (d : ℕ) :=
  ∀ n : ℕ, a (n + d) = a n

-- Proving the desired property
theorem sequence_periodic_a_n_plus_2_eq_a_n (a : ℕ → ℤ) (d : ℕ) (h_seq : seq a) (h_periodic : periodic a d) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end sequence_periodic_a_n_plus_2_eq_a_n_l1630_163008


namespace fraction_halfway_between_one_fourth_and_one_sixth_l1630_163039

theorem fraction_halfway_between_one_fourth_and_one_sixth :
  (1/4 + 1/6) / 2 = 5 / 24 :=
by
  sorry

end fraction_halfway_between_one_fourth_and_one_sixth_l1630_163039


namespace arithmetic_geometric_sequences_sequence_sum_first_terms_l1630_163052

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * a 1 + (n * (n + 1)) / 2

theorem arithmetic_geometric_sequences
  (a b S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_sequence b)
  (h3 : a 0 = 1)
  (h4 : b 0 = 1)
  (h5 : b 2 * S 2 = 36)
  (h6 : b 1 * S 1 = 8) :
  ((∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 2 ^ n)) ∨
  ((∀ n, a n = -(2 * n / 3) + 5 / 3) ∧ (∀ n, b n = 6 ^ n)) :=
sorry

theorem sequence_sum_first_terms
  (a : ℕ → ℤ)
  (h : ∀ n, a n = 2 * n + 1)
  (S : ℕ → ℤ)
  (T : ℕ → ℚ)
  (hS : sequence_sum a S)
  (n : ℕ) :
  T n = n / (2 * n + 1) :=
sorry

end arithmetic_geometric_sequences_sequence_sum_first_terms_l1630_163052


namespace missing_digit_divisibility_by_13_l1630_163002

theorem missing_digit_divisibility_by_13 (B : ℕ) (H : 0 ≤ B ∧ B ≤ 9) : 
  (13 ∣ (200 + 10 * B + 5)) ↔ B = 12 :=
by sorry

end missing_digit_divisibility_by_13_l1630_163002


namespace work_completion_l1630_163003

theorem work_completion (W : ℕ) (n : ℕ) (h1 : 0 < n) (H1 : 0 < W) :
  (∀ w : ℕ, w ≤ W / n) → 
  (∀ k : ℕ, k = (7 * n) / 10 → k * (3 * W) / (10 * n) ≥ W / 3) → 
  (∀ m : ℕ, m = (3 * n) / 10 → m * (7 * W) / (10 * n) ≥ W / 3) → 
  ∃ g1 g2 g3 : ℕ, g1 + g2 + g3 < W / 3 :=
by
  sorry

end work_completion_l1630_163003


namespace number_of_bookshelves_l1630_163055

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end number_of_bookshelves_l1630_163055


namespace problem_statement_l1630_163061

variables {R : Type*} [LinearOrderedField R]

theorem problem_statement (a b c : R) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : (b - a) ^ 2 - 4 * (b - c) * (c - a) = 0) : (b - c) / (c - a) = -1 :=
sorry

end problem_statement_l1630_163061


namespace cubics_sum_div_abc_eq_three_l1630_163031

theorem cubics_sum_div_abc_eq_three {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 :=
by
  sorry

end cubics_sum_div_abc_eq_three_l1630_163031


namespace num_three_digit_integers_sum_to_seven_l1630_163042

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l1630_163042


namespace ratio_B_over_A_eq_one_l1630_163062

theorem ratio_B_over_A_eq_one (A B : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 3 → 
  (A : ℝ) / (x + 3) + (B : ℝ) / (x * (x - 3)) = (x^3 - 3*x^2 + 15*x - 9) / (x^3 + x^2 - 9*x)) :
  (B : ℝ) / (A : ℝ) = 1 :=
sorry

end ratio_B_over_A_eq_one_l1630_163062


namespace find_x_l1630_163085

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
by
  sorry

end find_x_l1630_163085


namespace simplify_expression_l1630_163043

noncomputable def original_expression (x : ℝ) : ℝ :=
(x - 3 * x / (x + 1)) / ((x - 2) / (x^2 + 2 * x + 1))

theorem simplify_expression:
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 2 → 
  (original_expression x = x^2 + x) ∧ 
  ((x = 1 → original_expression x = 2) ∧ (x = 0 → original_expression x = 0)) :=
by
  intros
  sorry

end simplify_expression_l1630_163043


namespace teapot_volume_proof_l1630_163023

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem teapot_volume_proof (a d : ℝ)
  (h1 : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0.5)
  (h2 : arithmetic_sequence a d 7 + arithmetic_sequence a d 8 + arithmetic_sequence a d 9 = 2.5) :
  arithmetic_sequence a d 5 = 0.5 :=
by {
  sorry
}

end teapot_volume_proof_l1630_163023


namespace sheena_weeks_to_complete_l1630_163024

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end sheena_weeks_to_complete_l1630_163024


namespace find_last_three_digits_of_9_pow_107_l1630_163057

theorem find_last_three_digits_of_9_pow_107 : (9 ^ 107) % 1000 = 969 := 
by 
  sorry

end find_last_three_digits_of_9_pow_107_l1630_163057


namespace range_of_m_l1630_163025

theorem range_of_m (m : ℝ) : (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_l1630_163025


namespace g_odd_find_a_f_increasing_l1630_163019

-- Problem (I): Prove that if g(x) = f(x) - a is an odd function, then a = 1, given f(x) = 1 - 2/x.
theorem g_odd_find_a (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  (∀ x, g x = f x - a) → 
  (∀ x, g (-x) = - g x) → 
  a = 1 := 
  by
  intros h1 h2 h3
  sorry

-- Problem (II): Prove that f(x) is monotonically increasing on (0, +∞),
-- given f(x) = 1 - 2/x.

theorem f_increasing (f : ℝ → ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 := 
  by
  intros h1 x1 x2 hx1 hx12
  sorry

end g_odd_find_a_f_increasing_l1630_163019


namespace maria_paid_9_l1630_163071

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l1630_163071


namespace simplify_correct_l1630_163051

def simplify_polynomial (x : Real) : Real :=
  (12 * x^10 + 6 * x^9 + 3 * x^8) + (2 * x^11 + x^10 + 4 * x^9 + x^7 + 4 * x^4 + 7 * x + 9)

theorem simplify_correct (x : Real) :
  simplify_polynomial x = 2 * x^11 + 13 * x^10 + 10 * x^9 + 3 * x^8 + x^7 + 4 * x^4 + 7 * x + 9 :=
by
  sorry

end simplify_correct_l1630_163051


namespace blueberry_jelly_amount_l1630_163040

-- Definition of the conditions
def total_jelly : ℕ := 6310
def strawberry_jelly : ℕ := 1792

-- Formal statement of the problem
theorem blueberry_jelly_amount : 
  total_jelly - strawberry_jelly = 4518 :=
by
  sorry

end blueberry_jelly_amount_l1630_163040


namespace expected_yolks_correct_l1630_163070

-- Define the conditions
def total_eggs : ℕ := 15
def double_yolk_eggs : ℕ := 5
def triple_yolk_eggs : ℕ := 3
def single_yolk_eggs : ℕ := total_eggs - double_yolk_eggs - triple_yolk_eggs
def extra_yolk_prob : ℝ := 0.10

-- Define the expected number of yolks calculation
noncomputable def expected_yolks : ℝ :=
  (single_yolk_eggs * 1) + 
  (double_yolk_eggs * 2) + 
  (triple_yolk_eggs * 3) + 
  (double_yolk_eggs * extra_yolk_prob) + 
  (triple_yolk_eggs * extra_yolk_prob)

-- State that the expected number of total yolks is 26.8
theorem expected_yolks_correct : expected_yolks = 26.8 := by
  -- solution would go here
  sorry

end expected_yolks_correct_l1630_163070


namespace cost_of_book_sold_at_loss_l1630_163098

theorem cost_of_book_sold_at_loss:
  ∃ (C1 C2 : ℝ), 
    C1 + C2 = 490 ∧ 
    C1 * 0.85 = C2 * 1.19 ∧ 
    C1 = 285.93 :=
by
  sorry

end cost_of_book_sold_at_loss_l1630_163098


namespace lines_intersect_at_same_point_l1630_163017

theorem lines_intersect_at_same_point (m k : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 5 ∧ y = -4 * x + m ∧ y = 2 * x + k) ↔ k = (m + 30) / 7 :=
by {
  sorry -- proof not required, only statement.
}

end lines_intersect_at_same_point_l1630_163017


namespace zachary_pushups_l1630_163056

variable {P : ℕ}
variable {C : ℕ}

theorem zachary_pushups :
  C = 58 → C = P + 12 → P = 46 :=
by 
  intros hC1 hC2
  rw [hC2] at hC1
  linarith

end zachary_pushups_l1630_163056


namespace complete_the_square_l1630_163038

theorem complete_the_square (a : ℝ) : a^2 + 4 * a - 5 = (a + 2)^2 - 9 :=
by sorry

end complete_the_square_l1630_163038


namespace three_digit_int_one_less_than_lcm_mult_l1630_163090

theorem three_digit_int_one_less_than_lcm_mult : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n + 1) % Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 9 = 0 :=
sorry

end three_digit_int_one_less_than_lcm_mult_l1630_163090


namespace range_of_z_l1630_163096

theorem range_of_z (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6) :
  4 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 12 :=
by
  sorry

end range_of_z_l1630_163096


namespace green_pairs_count_l1630_163076

variable (blueShirtedStudents : Nat)
variable (yellowShirtedStudents : Nat)
variable (greenShirtedStudents : Nat)
variable (totalStudents : Nat)
variable (totalPairs : Nat)
variable (blueBluePairs : Nat)

def green_green_pairs (blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs : Nat) : Nat := 
  greenShirtedStudents / 2

theorem green_pairs_count
  (h1 : blueShirtedStudents = 70)
  (h2 : yellowShirtedStudents = 80)
  (h3 : greenShirtedStudents = 50)
  (h4 : totalStudents = 200)
  (h5 : totalPairs = 100)
  (h6 : blueBluePairs = 30) : 
  green_green_pairs blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs = 25 := by
  sorry

end green_pairs_count_l1630_163076


namespace max_value_sum_faces_edges_vertices_l1630_163092

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

def pyramid_faces_added : ℕ := 4
def pyramid_base_faces_covered : ℕ := 1
def pyramid_edges_added : ℕ := 4
def pyramid_vertices_added : ℕ := 1

def resulting_faces : ℕ := rectangular_prism_faces - pyramid_base_faces_covered + pyramid_faces_added
def resulting_edges : ℕ := rectangular_prism_edges + pyramid_edges_added
def resulting_vertices : ℕ := rectangular_prism_vertices + pyramid_vertices_added

def sum_resulting_faces_edges_vertices : ℕ := resulting_faces + resulting_edges + resulting_vertices

theorem max_value_sum_faces_edges_vertices : sum_resulting_faces_edges_vertices = 34 :=
by
  sorry

end max_value_sum_faces_edges_vertices_l1630_163092


namespace probability_of_purple_probability_of_blue_or_purple_l1630_163026

def total_jelly_beans : ℕ := 60
def purple_jelly_beans : ℕ := 5
def blue_jelly_beans : ℕ := 18

theorem probability_of_purple :
  (purple_jelly_beans : ℚ) / total_jelly_beans = 1 / 12 :=
by
  sorry
  
theorem probability_of_blue_or_purple :
  (blue_jelly_beans + purple_jelly_beans : ℚ) / total_jelly_beans = 23 / 60 :=
by
  sorry

end probability_of_purple_probability_of_blue_or_purple_l1630_163026


namespace f_inequality_solution_set_l1630_163084

noncomputable
def f : ℝ → ℝ := sorry

axiom f_at_1 : f 1 = 1
axiom f_deriv : ∀ x : ℝ, deriv f x < 1/3

theorem f_inequality_solution_set :
  {x : ℝ | f (x^2) > (x^2 / 3) + 2 / 3} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end f_inequality_solution_set_l1630_163084


namespace required_total_money_l1630_163091

def bundle_count := 100
def number_of_bundles := 10
def bill_5_value := 5
def bill_10_value := 10
def bill_20_value := 20

-- Sum up the total money required to fill the machine
theorem required_total_money : 
  (bundle_count * bill_5_value * number_of_bundles) + 
  (bundle_count * bill_10_value * number_of_bundles) + 
  (bundle_count * bill_20_value * number_of_bundles) = 35000 := 
by 
  sorry

end required_total_money_l1630_163091


namespace range_of_sum_l1630_163086

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) : 
  -2 ≤ x + y ∧ x + y ≤ 0 :=
sorry

end range_of_sum_l1630_163086


namespace correct_polynomial_and_result_l1630_163028

theorem correct_polynomial_and_result :
  ∃ p q r : Polynomial ℝ,
    q = X^2 - 3 * X + 5 ∧
    p + q = 5 * X^2 - 2 * X + 4 ∧
    p = 4 * X^2 + X - 1 ∧
    r = p - q ∧
    r = 3 * X^2 + 4 * X - 6 :=
by {
  sorry
}

end correct_polynomial_and_result_l1630_163028


namespace number_of_toys_gained_l1630_163022

theorem number_of_toys_gained
  (num_toys : ℕ) (selling_price : ℕ) (cost_price_one_toy : ℕ)
  (total_cp := num_toys * cost_price_one_toy)
  (profit := selling_price - total_cp)
  (num_toys_equiv_to_profit := profit / cost_price_one_toy) :
  num_toys = 18 → selling_price = 23100 → cost_price_one_toy = 1100 → num_toys_equiv_to_profit = 3 :=
by
  intros h1 h2 h3
  -- Proof to be completed
  sorry

end number_of_toys_gained_l1630_163022


namespace smallest_natural_greater_than_12_l1630_163094

def smallest_greater_than (n : ℕ) : ℕ := n + 1

theorem smallest_natural_greater_than_12 : smallest_greater_than 12 = 13 :=
by
  sorry

end smallest_natural_greater_than_12_l1630_163094


namespace largest_angle_of_triangle_l1630_163020

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 4 * x + 5 * x + 9 * x = 180) 
  (h2 : 4 * x > 40) : 
  9 * x = 90 := 
sorry

end largest_angle_of_triangle_l1630_163020


namespace closest_fraction_l1630_163037

theorem closest_fraction (n : ℤ) : 
  let frac1 := 37 / 57 
  let closest := 15 / 23
  n = 15 ∧ abs (851 - 57 * n) = min (abs (851 - 57 * 14)) (abs (851 - 57 * 15)) :=
by
  let frac1 := (37 : ℚ) / 57
  let closest := (15 : ℚ) / 23
  have h : 37 * 23 = 851 := by norm_num
  have denom : 57 * 23 = 1311 := by norm_num
  let num := 851
  sorry

end closest_fraction_l1630_163037


namespace import_tax_l1630_163032

theorem import_tax (total_value : ℝ) (tax_rate : ℝ) (excess_limit : ℝ) (correct_tax : ℝ)
  (h1 : total_value = 2560) (h2 : tax_rate = 0.07) (h3 : excess_limit = 1000) : 
  correct_tax = tax_rate * (total_value - excess_limit) :=
by
  sorry

end import_tax_l1630_163032


namespace tan_identity_proof_l1630_163083

theorem tan_identity_proof
  (α β : ℝ)
  (h₁ : Real.tan (α + β) = 3)
  (h₂ : Real.tan (α + π / 4) = -3) :
  Real.tan (β - π / 4) = -3 / 4 := 
sorry

end tan_identity_proof_l1630_163083


namespace bus_passenger_count_l1630_163014

-- Definitions for conditions
def initial_passengers : ℕ := 0
def passengers_first_stop (initial : ℕ) : ℕ := initial + 7
def passengers_second_stop (after_first : ℕ) : ℕ := after_first - 3 + 5
def passengers_third_stop (after_second : ℕ) : ℕ := after_second - 2 + 4

-- Statement we want to prove
theorem bus_passenger_count : 
  passengers_third_stop (passengers_second_stop (passengers_first_stop initial_passengers)) = 11 :=
by
  -- proof would go here
  sorry

end bus_passenger_count_l1630_163014


namespace jacket_total_price_correct_l1630_163082

/-- The original price of the jacket -/
def original_price : ℝ := 120

/-- The initial discount rate -/
def initial_discount_rate : ℝ := 0.15

/-- The additional discount in dollars -/
def additional_discount : ℝ := 10

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.10

/-- The calculated total amount the shopper pays for the jacket including all discounts and tax -/
def total_amount_paid : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  price_after_additional_discount * (1 + sales_tax_rate)

theorem jacket_total_price_correct : total_amount_paid = 101.20 :=
  sorry

end jacket_total_price_correct_l1630_163082


namespace expression_evaluation_l1630_163074

theorem expression_evaluation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 :=
by
  sorry

end expression_evaluation_l1630_163074


namespace average_minutes_run_l1630_163018

theorem average_minutes_run (t : ℕ) (t_pos : 0 < t) 
  (average_first_graders : ℕ := 8) 
  (average_second_graders : ℕ := 12) 
  (average_third_graders : ℕ := 16)
  (num_first_graders : ℕ := 9 * t)
  (num_second_graders : ℕ := 3 * t)
  (num_third_graders : ℕ := t) :
  (8 * 9 * t + 12 * 3 * t + 16 * t) / (9 * t + 3 * t + t) = 10 := 
by
  sorry

end average_minutes_run_l1630_163018


namespace women_more_than_men_l1630_163009

theorem women_more_than_men 
(M W : ℕ) 
(h_ratio : (M:ℚ) / W = 5 / 9) 
(h_total : M + W = 14) :
W - M = 4 := 
by 
  sorry

end women_more_than_men_l1630_163009


namespace evaluate_expression_l1630_163058

noncomputable def expression := 
  (Real.sqrt 3 * Real.tan (Real.pi / 15) - 3) / 
  (4 * (Real.cos (Real.pi / 15))^2 * Real.sin (Real.pi / 15) - 2 * Real.sin (Real.pi / 15))

theorem evaluate_expression : expression = -4 * Real.sqrt 3 :=
  sorry

end evaluate_expression_l1630_163058


namespace g_of_5_l1630_163016

noncomputable def g : ℝ → ℝ := sorry

theorem g_of_5 :
  (∀ x y : ℝ, x * g y = y * g x) →
  g 20 = 30 →
  g 5 = 7.5 :=
by
  intros h1 h2
  sorry

end g_of_5_l1630_163016


namespace triangle_arithmetic_progression_l1630_163080

theorem triangle_arithmetic_progression (a d : ℝ) 
(h1 : (a-2*d)^2 + a^2 = (a+2*d)^2) 
(h2 : ∃ x : ℝ, (a = x * d) ∨ (d = x * a))
: (6 ∣ 6*d) ∧ (12 ∣ 6*d) ∧ (18 ∣ 6*d) ∧ (24 ∣ 6*d) ∧ (30 ∣ 6*d)
:= by
  sorry

end triangle_arithmetic_progression_l1630_163080


namespace smallest_n_division_l1630_163088

-- Lean statement equivalent to the mathematical problem
theorem smallest_n_division (n : ℕ) (hn : n ≥ 3) : 
  (∃ (s : Finset ℕ), (∀ m ∈ s, 3 ≤ m ∧ m ≤ 2006) ∧ s.card = n - 2) ↔ n = 3 := 
sorry

end smallest_n_division_l1630_163088


namespace remainder_consec_even_div12_l1630_163095

theorem remainder_consec_even_div12 (n : ℕ) (h: n % 2 = 0)
  (h1: 11234 ≤ n ∧ n + 12 ≥ 11246) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 6 :=
by 
  sorry

end remainder_consec_even_div12_l1630_163095


namespace shadow_stretch_rate_is_5_feet_per_hour_l1630_163053

-- Given conditions
def shadow_length_in_inches (hours_past_noon : ℕ) : ℕ := 360
def hours_past_noon : ℕ := 6

-- Convert inches to feet
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

-- Calculate rate of increase of shadow length per hour
def rate_of_shadow_stretch_per_hour : ℕ := inches_to_feet (shadow_length_in_inches hours_past_noon) / hours_past_noon

theorem shadow_stretch_rate_is_5_feet_per_hour :
  rate_of_shadow_stretch_per_hour = 5 := by
  sorry

end shadow_stretch_rate_is_5_feet_per_hour_l1630_163053


namespace impossible_to_arrange_distinct_integers_in_grid_l1630_163059

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end impossible_to_arrange_distinct_integers_in_grid_l1630_163059


namespace solve_absolute_value_eq_l1630_163099

theorem solve_absolute_value_eq (x : ℝ) : |x - 5| = 3 * x - 2 ↔ x = 7 / 4 :=
sorry

end solve_absolute_value_eq_l1630_163099


namespace functional_equation_holds_l1630_163034

def f (p q : ℕ) : ℝ :=
  if p = 0 ∨ q = 0 then 0 else (p * q : ℝ)

theorem functional_equation_holds (p q : ℕ) : 
  f p q = 
    if p = 0 ∨ q = 0 then 0 
    else 1 + (1 / 2) * f (p + 1) (q - 1) + (1 / 2) * f (p - 1) (q + 1) :=
  by 
    sorry

end functional_equation_holds_l1630_163034


namespace inequality_example_l1630_163010

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (der : ∀ x, deriv f x = f' x)

theorem inequality_example (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023)
:= sorry

end inequality_example_l1630_163010


namespace power_calculation_l1630_163044

theorem power_calculation : 8^6 * 27^6 * 8^18 * 27^18 = 216^24 := by
  sorry

end power_calculation_l1630_163044


namespace circle_polar_equation_l1630_163067

-- Definitions and conditions
def circle_equation_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_coordinates (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem to be proven
theorem circle_polar_equation (ρ θ : ℝ) :
  (∀ x y : ℝ, circle_equation_cartesian x y → 
  polar_coordinates ρ θ x y) → ρ = 2 * Real.sin θ :=
by
  -- This is a placeholder for the proof
  sorry

end circle_polar_equation_l1630_163067


namespace solve_xyz_eq_x_plus_y_l1630_163097

theorem solve_xyz_eq_x_plus_y (x y z : ℕ) (h1 : x * y * z = x + y) (h2 : x ≤ y) : (x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2) :=
by {
    sorry -- The actual proof goes here
}

end solve_xyz_eq_x_plus_y_l1630_163097


namespace intersection_M_N_l1630_163078

/-- Define the set M as pairs (x, y) such that x + y = 2. -/
def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 2 }

/-- Define the set N as pairs (x, y) such that x - y = 2. -/
def N : Set (ℝ × ℝ) := { p | p.1 - p.2 = 2 }

/-- The intersection of sets M and N is the single point (2, 0). -/
theorem intersection_M_N : M ∩ N = { (2, 0) } :=
by
  sorry

end intersection_M_N_l1630_163078


namespace simplify_and_evaluate_l1630_163029

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_and_evaluate_l1630_163029


namespace right_triangle_even_or_odd_l1630_163054

theorem right_triangle_even_or_odd (a b c : ℕ) (ha : Even a ∨ Odd a) (hb : Even b ∨ Odd b) (h : a^2 + b^2 = c^2) : 
  Even c ∨ (Even a ∧ Odd b) ∨ (Odd a ∧ Even b) :=
by
  sorry

end right_triangle_even_or_odd_l1630_163054


namespace max_value_of_expr_l1630_163045

noncomputable def max_expr (a b : ℝ) (h : a + b = 5) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_value_of_expr (a b : ℝ) (h : a + b = 5) : max_expr a b h ≤ 6084 / 17 :=
sorry

end max_value_of_expr_l1630_163045


namespace calculate_expression_l1630_163035

theorem calculate_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end calculate_expression_l1630_163035


namespace minimum_value_of_f_roots_sum_gt_2_l1630_163021

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f 1 = 1 := by
  exists 1
  sorry

theorem roots_sum_gt_2 (a x₁ x₂ : ℝ) (h_f_x₁ : f x₁ = a) (h_f_x₂ : f x₂ = a) (h_x₁_lt_x₂ : x₁ < x₂) :
    x₁ + x₂ > 2 := by
  sorry

end minimum_value_of_f_roots_sum_gt_2_l1630_163021


namespace pen_price_l1630_163069

theorem pen_price (x y : ℝ) (h1 : 2 * x + 3 * y = 49) (h2 : 3 * x + y = 49) : x = 14 :=
by
  -- Proof required here
  sorry

end pen_price_l1630_163069


namespace problem_l1630_163073

noncomputable def a_seq (n : ℕ) : ℚ := sorry

def is_geometric_sequence (seq : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = q * seq n

theorem problem (h_positive : ∀ n : ℕ, 0 < a_seq n)
                (h_ratio : ∀ n : ℕ, 2 * a_seq n = 3 * a_seq (n + 1))
                (h_product : a_seq 1 * a_seq 4 = 8 / 27) :
  is_geometric_sequence a_seq (2 / 3) ∧ 
  (∃ n : ℕ, a_seq n = 16 / 81 ∧ n = 6) :=
by
  sorry

end problem_l1630_163073


namespace monotonic_intervals_range_of_k_l1630_163011

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x ^ 2) - k * (2 / x + Real.log x)
noncomputable def f' (x k : ℝ) : ℝ := (x - 2) * (Real.exp x - k * x) / (x^3)

theorem monotonic_intervals (k : ℝ) (h : k ≤ 0) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x k < 0) ∧ (∀ x : ℝ, x > 2 → f' x k > 0) := sorry

theorem range_of_k (k : ℝ) (h : e < k ∧ k < (e^2)/2) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ 
    (f' x1 k = 0 ∧ f' x2 k = 0 ∧ x1 ≠ x2) := sorry

end monotonic_intervals_range_of_k_l1630_163011


namespace find_integer_n_l1630_163081

open Int

theorem find_integer_n (n a b : ℤ) :
  (4 * n + 1 = a^2) ∧ (9 * n + 1 = b^2) → n = 0 := by
sorry

end find_integer_n_l1630_163081


namespace xyz_value_l1630_163087

theorem xyz_value (x y z : ℝ)
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
    x * y * z = 16 / 3 := by
    sorry

end xyz_value_l1630_163087


namespace product_of_last_two_digits_div_by_6_and_sum_15_l1630_163013

theorem product_of_last_two_digits_div_by_6_and_sum_15
  (n : ℕ)
  (h1 : n % 6 = 0)
  (A B : ℕ)
  (h2 : n % 100 = 10 * A + B)
  (h3 : A + B = 15)
  (h4 : B % 2 = 0) : 
  A * B = 54 := 
sorry

end product_of_last_two_digits_div_by_6_and_sum_15_l1630_163013


namespace average_age_with_teacher_l1630_163030

theorem average_age_with_teacher (A : ℕ) (h : 21 * 16 = 20 * A + 36) : A = 15 := by
  sorry

end average_age_with_teacher_l1630_163030


namespace hyperbola_no_common_point_l1630_163041

theorem hyperbola_no_common_point (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (y_line : ∀ x : ℝ, y = 2 * x) : 
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e ≤ Real.sqrt 5 :=
by
  sorry

end hyperbola_no_common_point_l1630_163041


namespace calculation_correct_l1630_163060

theorem calculation_correct : 4 * 6 * 8 - 10 / 2 = 187 := by
  sorry

end calculation_correct_l1630_163060


namespace sum_of_squares_l1630_163050

theorem sum_of_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := 
by
  sorry

end sum_of_squares_l1630_163050


namespace find_remainder_proof_l1630_163012

def div_remainder_problem :=
  let number := 220050
  let sum := 555 + 445
  let difference := 555 - 445
  let quotient := 2 * difference
  let divisor := sum
  let quotient_correct := quotient = 220
  let division_formula := number = divisor * quotient + 50
  quotient_correct ∧ division_formula

theorem find_remainder_proof : div_remainder_problem := by
  sorry

end find_remainder_proof_l1630_163012


namespace roundTripAverageSpeed_l1630_163006

noncomputable def averageSpeed (distAB distBC speedAB speedBC speedCB totalTime : ℝ) : ℝ :=
  let timeAB := distAB / speedAB
  let timeBC := distBC / speedBC
  let timeCB := distBC / speedCB
  let timeBA := totalTime - (timeAB + timeBC + timeCB)
  let totalDistance := 2 * (distAB + distBC)
  totalDistance / totalTime

theorem roundTripAverageSpeed :
  averageSpeed 150 230 80 88 100 9 = 84.44 :=
by
  -- The actual proof will go here, which is not required for this task.
  sorry

end roundTripAverageSpeed_l1630_163006


namespace sqrt_x_minus_2_meaningful_in_reals_l1630_163064

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l1630_163064


namespace pulley_weight_l1630_163066

theorem pulley_weight (M g : ℝ) (hM_pos : 0 < M) (F : ℝ := 50) :
  (g ≠ 0) → (M * g = 100) :=
by
  sorry

end pulley_weight_l1630_163066


namespace larger_acute_angle_right_triangle_l1630_163027

theorem larger_acute_angle_right_triangle (x : ℝ) (h1 : x > 0) (h2 : x + 5 * x = 90) : 5 * x = 75 := by
  sorry

end larger_acute_angle_right_triangle_l1630_163027


namespace count_three_digit_numbers_using_1_and_2_l1630_163072

theorem count_three_digit_numbers_using_1_and_2 : 
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 6 :=
by
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 6
  sorry

end count_three_digit_numbers_using_1_and_2_l1630_163072


namespace brenda_age_l1630_163004

-- Define ages of Addison, Brenda, Carlos, and Janet
variables (A B C J : ℕ)

-- Formalize the conditions from the problem
def condition1 := A = 4 * B
def condition2 := C = 2 * B
def condition3 := A = J

-- State the theorem we aim to prove
theorem brenda_age (A B C J : ℕ) (h1 : condition1 A B)
                                (h2 : condition2 C B)
                                (h3 : condition3 A J) :
  B = J / 4 :=
sorry

end brenda_age_l1630_163004


namespace problem_statement_l1630_163068

def f (x : ℤ) : ℤ := 2 * x ^ 2 + 3 * x - 1

theorem problem_statement : f (f 3) = 1429 := by
  sorry

end problem_statement_l1630_163068


namespace original_soldiers_eq_136_l1630_163007

-- Conditions
def original_soldiers (n : ℕ) : ℕ := 8 * n
def after_adding_120 (n : ℕ) : ℕ := original_soldiers n + 120
def after_removing_120 (n : ℕ) : ℕ := original_soldiers n - 120

-- Given that both after_adding_120 n and after_removing_120 n are perfect squares.
def is_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- Theorem statement
theorem original_soldiers_eq_136 : ∃ n : ℕ, original_soldiers n = 136 ∧ 
                                   is_square (after_adding_120 n) ∧ 
                                   is_square (after_removing_120 n) :=
sorry

end original_soldiers_eq_136_l1630_163007


namespace angle_supplement_complement_l1630_163001

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l1630_163001


namespace sara_cakes_sales_l1630_163079

theorem sara_cakes_sales :
  let cakes_per_day := 4
  let days_per_week := 5
  let weeks := 4
  let price_per_cake := 8
  let cakes_per_week := cakes_per_day * days_per_week
  let total_cakes := cakes_per_week * weeks
  let total_money := total_cakes * price_per_cake
  total_money = 640 := 
by
  sorry

end sara_cakes_sales_l1630_163079


namespace quadratic_solution_linear_factor_solution_l1630_163047

theorem quadratic_solution (x : ℝ) : (5 * x^2 + 2 * x - 1 = 0) ↔ (x = (-1 + Real.sqrt 6) / 5 ∨ x = (-1 - Real.sqrt 6) / 5) := by
  sorry

theorem linear_factor_solution (x : ℝ) : (x * (x - 3) - 4 * (3 - x) = 0) ↔ (x = 3 ∨ x = -4) := by
  sorry

end quadratic_solution_linear_factor_solution_l1630_163047


namespace find_solutions_equation_l1630_163015

theorem find_solutions_equation :
  {x : ℝ | 1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 11 * x - 12) = 0}
  = {1, -12, 4, -3} :=
by
  sorry

end find_solutions_equation_l1630_163015


namespace range_of_k_l1630_163075

noncomputable def quadratic_inequality (k : ℝ) := 
  ∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0

theorem range_of_k (k : ℝ) :
  (quadratic_inequality k) → -3 < k ∧ k < 0 := sorry

end range_of_k_l1630_163075


namespace gcd_sixPn_n_minus_2_l1630_163049

def nthSquarePyramidalNumber (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

def sixPn (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1)

theorem gcd_sixPn_n_minus_2 (n : ℕ) (h_pos : 0 < n) : Int.gcd (sixPn n) (n - 2) ≤ 12 :=
by
  sorry

end gcd_sixPn_n_minus_2_l1630_163049


namespace largest_square_area_with_4_interior_lattice_points_l1630_163093

/-- 
A point (x, y) in the plane is called a lattice point if both x and y are integers.
The largest square that contains exactly four lattice points solely in its interior
has an area of 9.
-/
theorem largest_square_area_with_4_interior_lattice_points : 
  ∃ s : ℝ, ∀ (x y : ℤ), 
  (1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → s^2 = 9 := 
sorry

end largest_square_area_with_4_interior_lattice_points_l1630_163093
