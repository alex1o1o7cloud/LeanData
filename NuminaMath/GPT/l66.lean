import Mathlib

namespace solve_arithmetic_sequence_sum_l66_66347

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l66_66347


namespace find_x_l66_66728

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l66_66728


namespace jellybean_ratio_l66_66701

theorem jellybean_ratio (L Tino Arnold : ℕ) (h1 : Tino = L + 24) (h2 : Arnold = 5) (h3 : Tino = 34) :
  Arnold / L = 1 / 2 :=
by
  sorry

end jellybean_ratio_l66_66701


namespace total_attendance_l66_66901

theorem total_attendance (A C : ℕ) (ticket_sales : ℕ) (adult_ticket_cost child_ticket_cost : ℕ) (total_collected : ℕ)
    (h1 : C = 18) (h2 : ticket_sales = 50) (h3 : adult_ticket_cost = 8) (h4 : child_ticket_cost = 1)
    (h5 : ticket_sales = adult_ticket_cost * A + child_ticket_cost * C) :
    A + C = 22 :=
by {
  sorry
}

end total_attendance_l66_66901


namespace find_ratio_l66_66928

theorem find_ratio (a b : ℝ) (h1 : ∀ x, ax^2 + bx + 2 < 0 ↔ (x < -1/2 ∨ x > 1/3)) :
  (a - b) / a = 5 / 6 := 
sorry

end find_ratio_l66_66928


namespace larger_integer_value_l66_66690

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l66_66690


namespace correct_number_of_statements_l66_66797

-- Define the conditions as invalidity of the given statements
def statement_1_invalid : Prop := ¬ (true) -- INPUT a,b,c should use commas
def statement_2_invalid : Prop := ¬ (true) -- INPUT x=, 3 correct format
def statement_3_invalid : Prop := ¬ (true) -- 3=B , left side should be a variable name
def statement_4_invalid : Prop := ¬ (true) -- A=B=2, continuous assignment not allowed

-- Combine conditions
def all_statements_invalid : Prop := statement_1_invalid ∧ statement_2_invalid ∧ statement_3_invalid ∧ statement_4_invalid

-- State the theorem to prove
theorem correct_number_of_statements : all_statements_invalid → 0 = 0 := 
by sorry

end correct_number_of_statements_l66_66797


namespace correct_midpoint_l66_66485

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66485


namespace min_positive_d_l66_66688

theorem min_positive_d (a b t d : ℤ) (h1 : 3 * t = 2 * a + 2 * b + 2016)
                                       (h2 : t - a = d)
                                       (h3 : t - b = 2 * d)
                                       (h4 : 2 * a + 2 * b > 0) :
    ∃ d : ℤ, d > 0 ∧ (505 ≤ d ∧ ∀ e : ℤ, e > 0 → 3 * (a + d) = 2 * (b + 2 * e) + 2016 → 505 ≤ e) := 
sorry

end min_positive_d_l66_66688


namespace determine_Y_in_arithmetic_sequence_matrix_l66_66399

theorem determine_Y_in_arithmetic_sequence_matrix :
  (exists a₁ a₂ a₃ a₄ a₅ : ℕ, 
    -- Conditions for the first row (arithmetic sequence with first term 3 and fifth term 15)
    a₁ = 3 ∧ a₅ = 15 ∧ 
    (∃ d₁ : ℕ, a₂ = a₁ + d₁ ∧ a₃ = a₂ + d₁ ∧ a₄ = a₃ + d₁ ∧ a₅ = a₄ + d₁) ∧

    -- Conditions for the fifth row (arithmetic sequence with first term 25 and fifth term 65)
    a₁ = 25 ∧ a₅ = 65 ∧ 
    (∃ d₅ : ℕ, a₂ = a₁ + d₅ ∧ a₃ = a₂ + d₅ ∧ a₄ = a₃ + d₅ ∧ a₅ = a₄ + d₅) ∧

    -- Middle element Y
    a₃ = 27) :=
sorry

end determine_Y_in_arithmetic_sequence_matrix_l66_66399


namespace polynomial_simplified_l66_66983

def polynomial (x : ℝ) : ℝ := 4 - 6 * x - 8 * x^2 + 12 - 14 * x + 16 * x^2 - 18 + 20 * x + 24 * x^2

theorem polynomial_simplified (x : ℝ) : polynomial x = 32 * x^2 - 2 :=
by
  sorry

end polynomial_simplified_l66_66983


namespace max_abcsum_l66_66154

theorem max_abcsum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_eq : a * b^2 * c^3 = 1350) : 
  a + b + c ≤ 154 :=
sorry

end max_abcsum_l66_66154


namespace sum_even_pos_ints_less_than_100_eq_2450_l66_66052

-- Define the sum of even positive integers less than 100
def sum_even_pos_ints_less_than_100 : ℕ :=
  ∑ i in finset.filter (λ x, x % 2 = 0) (finset.range 100), i

-- Theorem to prove the sum is equal to 2450
theorem sum_even_pos_ints_less_than_100_eq_2450 :
  sum_even_pos_ints_less_than_100 = 2450 :=
by
  sorry

end sum_even_pos_ints_less_than_100_eq_2450_l66_66052


namespace sum_of_remainders_l66_66987

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end sum_of_remainders_l66_66987


namespace box_volume_correct_l66_66401

def volume_of_box (x : ℝ) : ℝ := (16 - 2 * x) * (12 - 2 * x) * x

theorem box_volume_correct {x : ℝ} (h1 : 1 ≤ x) (h2 : x ≤ 3) : 
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x := 
by 
  unfold volume_of_box 
  sorry

end box_volume_correct_l66_66401


namespace problem_l66_66299

theorem problem (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -1/2 ∨ x > 1/3)) : 
  a + b = -14 :=
sorry

end problem_l66_66299


namespace midpoint_of_hyperbola_segment_l66_66503

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66503


namespace cost_of_rope_l66_66443

theorem cost_of_rope : 
  ∀ (total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost : ℝ),
  total_money = 200 ∧
  sheet_cost = 42 ∧
  propane_burner_cost = 14 ∧
  helium_cost_per_ounce = 1.50 ∧
  helium_per_foot = 113 ∧
  max_height = 9492 ∧
  rope_cost = total_money - (sheet_cost + propane_burner_cost + (max_height / helium_per_foot) * helium_cost_per_ounce) →
  rope_cost = 18 :=
by
  intros total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost
  rintro ⟨h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max, h_rope⟩
  rw [h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max] at h_rope
  simp only [inv_mul_eq_iff_eq_mul, div_eq_mul_inv] at h_rope
  norm_num at h_rope
  sorry

end cost_of_rope_l66_66443


namespace find_f_2024_l66_66120

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_f_2024 (a b c : ℝ)
  (h1 : f 2021 a b c = 2021)
  (h2 : f 2022 a b c = 2022)
  (h3 : f 2023 a b c = 2023) :
  f 2024 a b c = 2030 := sorry

end find_f_2024_l66_66120


namespace remainder_eq_one_l66_66446

theorem remainder_eq_one (n : ℤ) (h : n % 6 = 1) : (n + 150) % 6 = 1 := 
by
  sorry

end remainder_eq_one_l66_66446


namespace frog_probability_l66_66146

noncomputable def frog_escape_prob (P : ℕ → ℚ) : Prop :=
  P 0 = 0 ∧
  P 11 = 1 ∧
  (∀ N, 0 < N ∧ N < 11 → 
    P N = (N + 1) / 12 * P (N - 1) + (1 - (N + 1) / 12) * P (N + 1)) ∧
  P 2 = 72 / 167

theorem frog_probability : ∃ P : ℕ → ℚ, frog_escape_prob P :=
sorry

end frog_probability_l66_66146


namespace mass_percentage_C_in_CO_l66_66048

noncomputable def atomic_mass_C : ℚ := 12.01
noncomputable def atomic_mass_O : ℚ := 16.00
noncomputable def molecular_mass_CO : ℚ := atomic_mass_C + atomic_mass_O

theorem mass_percentage_C_in_CO : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 :=
by
  have atomic_mass_C_div_total : atomic_mass_C / molecular_mass_CO = 12.01 / 28.01 := sorry
  have mass_percentage : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 := sorry
  exact mass_percentage

end mass_percentage_C_in_CO_l66_66048


namespace midpoint_of_line_segment_on_hyperbola_l66_66612

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66612


namespace intersection_of_lines_l66_66415

theorem intersection_of_lines :
  ∃ x y : ℚ, (12 * x - 3 * y = 33) ∧ (8 * x + 2 * y = 18) ∧ (x = 29 / 12 ∧ y = -2 / 3) :=
by {
  sorry
}

end intersection_of_lines_l66_66415


namespace marthas_bedroom_size_l66_66358

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l66_66358


namespace radius_for_visibility_l66_66881

def is_concentric (hex_center : ℝ × ℝ) (circle_center : ℝ × ℝ) : Prop :=
  hex_center = circle_center

def regular_hexagon (side_length : ℝ) : Prop :=
  side_length = 3

theorem radius_for_visibility
  (r : ℝ)
  (hex_center : ℝ × ℝ)
  (circle_center : ℝ × ℝ)
  (P_visible: ℝ)
  (prob_Four_sides_visible: ℝ ) :
  is_concentric hex_center circle_center →
  regular_hexagon 3 →
  prob_Four_sides_visible = 1 / 3 →
  P_visible = 4 →
  r = 2.6 :=
by sorry

end radius_for_visibility_l66_66881


namespace midpoint_of_hyperbola_l66_66474

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66474


namespace abs_f_at_1_eq_20_l66_66821

noncomputable def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ p : Polynomial ℝ, p.degree = 4 ∧ ∀ x, f x = p.eval x

theorem abs_f_at_1_eq_20 
  (f : ℝ → ℝ)
  (h_f_poly : fourth_degree_polynomial f)
  (h_f_neg2 : |f (-2)| = 10)
  (h_f_0 : |f 0| = 10)
  (h_f_3 : |f 3| = 10)
  (h_f_7 : |f 7| = 10) :
  |f 1| = 20 := 
sorry

end abs_f_at_1_eq_20_l66_66821


namespace midpoint_of_hyperbola_l66_66573

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66573


namespace sum_even_pos_ints_less_than_100_eq_2450_l66_66053

-- Define the sum of even positive integers less than 100
def sum_even_pos_ints_less_than_100 : ℕ :=
  ∑ i in finset.filter (λ x, x % 2 = 0) (finset.range 100), i

-- Theorem to prove the sum is equal to 2450
theorem sum_even_pos_ints_less_than_100_eq_2450 :
  sum_even_pos_ints_less_than_100 = 2450 :=
by
  sorry

end sum_even_pos_ints_less_than_100_eq_2450_l66_66053


namespace smallest_class_size_l66_66303

theorem smallest_class_size (n : ℕ) (x : ℕ) (h1 : n > 50) (h2 : n = 4 * x + 2) : n = 54 :=
by
  sorry

end smallest_class_size_l66_66303


namespace derivative_of_f_is_l66_66340

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2

theorem derivative_of_f_is (x : ℝ) : deriv f x = 2 * x + 2 :=
by
  sorry

end derivative_of_f_is_l66_66340


namespace midpoint_of_hyperbola_l66_66576

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66576


namespace midpoint_on_hyperbola_l66_66523

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66523


namespace siamese_cats_initial_l66_66755

theorem siamese_cats_initial (S : ℕ) (h1 : 20 + S - 20 = 12) : S = 12 :=
by
  sorry

end siamese_cats_initial_l66_66755


namespace halfway_fraction_l66_66002

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66002


namespace percent_forgot_group_B_l66_66207

def num_students_group_A : ℕ := 20
def num_students_group_B : ℕ := 80
def percent_forgot_group_A : ℚ := 0.20
def total_percent_forgot : ℚ := 0.16

/--
There are two groups of students in the sixth grade. 
There are 20 students in group A, and 80 students in group B. 
On a particular day, 20% of the students in group A forget their homework, and a certain 
percentage of the students in group B forget their homework. 
Then, 16% of the sixth graders forgot their homework. 
Prove that 15% of the students in group B forgot their homework.
-/
theorem percent_forgot_group_B : 
  let num_forgot_group_A := percent_forgot_group_A * num_students_group_A
  let total_students := num_students_group_A + num_students_group_B
  let total_forgot := total_percent_forgot * total_students
  let num_forgot_group_B := total_forgot - num_forgot_group_A
  let percent_forgot_group_B := (num_forgot_group_B / num_students_group_B) * 100
  percent_forgot_group_B = 15 :=
by {
  sorry
}

end percent_forgot_group_B_l66_66207


namespace sara_cakes_sales_l66_66176

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

end sara_cakes_sales_l66_66176


namespace part1_part2_l66_66434

noncomputable def f (x : ℝ) : ℝ := |x - 1| - 1
noncomputable def g (x : ℝ) : ℝ := -|x + 1| - 4

theorem part1 (x : ℝ) : f x ≤ 1 ↔ -1 ≤ x ∧ x ≤ 3 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ 4 :=
by
  sorry

end part1_part2_l66_66434


namespace seashell_count_l66_66172

theorem seashell_count (Sam Mary Lucy : Nat) (h1 : Sam = 18) (h2 : Mary = 47) (h3 : Lucy = 32) : 
  Sam + Mary + Lucy = 97 :=
by 
  sorry

end seashell_count_l66_66172


namespace age_relationships_l66_66998

variables (a b c d : ℕ)

theorem age_relationships (h1 : a + b = b + c + d + 18) (h2 : 2 * a = 3 * c) :
  c = 2 * a / 3 ∧ d = a / 3 - 18 :=
by
  sorry

end age_relationships_l66_66998


namespace midpoint_on_hyperbola_l66_66599

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66599


namespace solve_equation1_solve_equation2_l66_66179

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 2 * x - 4 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 6 = x * (3 - x)

-- State the first proof problem
theorem solve_equation1 (x : ℝ) :
  equation1 x ↔ (x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) := by
  sorry

-- State the second proof problem
theorem solve_equation2 (x : ℝ) :
  equation2 x ↔ (x = 3 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l66_66179


namespace mixture_replacement_l66_66226

theorem mixture_replacement:
  ∀ (A B x : ℝ),
    A = 64 →
    B = A / 4 →
    (A - (4/5) * x) / (B + (4/5) * x) = 2 / 3 →
    x = 40 :=
by
  intros A B x hA hB hRatio
  sorry

end mixture_replacement_l66_66226


namespace jan_drove_more_l66_66993

variables (d t s : ℕ)
variables (h h_ans : ℕ)
variables (ha_speed j_speed : ℕ)
variables (j d_plus : ℕ)

-- Ian's equation
def ian_distance (s t : ℕ) : ℕ := s * t

-- Han's additional conditions
def han_distance (s t : ℕ) (h_speed : ℕ)
    (d_plus : ℕ) : Prop :=
  d_plus + 120 = (s + h_speed) * (t + 2)

-- Jan's conditions and equation
def jan_distance (s t : ℕ) (j_speed : ℕ) : ℕ :=
  (s + j_speed) * (t + 3)

-- Proof statement
theorem jan_drove_more (d t s h_ans : ℕ)
    (h_speed j_speed : ℕ) (d_plus : ℕ)
    (h_dist_cond : han_distance s t h_speed d_plus)
    (j_dist_cond : jan_distance s t j_speed = h_ans) :
  h_ans = 195 :=
sorry

end jan_drove_more_l66_66993


namespace hyperbola_midpoint_exists_l66_66557

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66557


namespace hyperbola_midpoint_exists_l66_66559

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66559


namespace true_compound_proposition_l66_66953

-- Define conditions and propositions in Lean
def proposition_p : Prop := ∃ (x : ℝ), x^2 + x + 1 < 0
def proposition_q : Prop := ∀ (x : ℝ), 1 ≤ x → x ≤ 2 → x^2 - 1 ≥ 0

-- Define the compound proposition
def correct_proposition : Prop := ¬ proposition_p ∧ proposition_q

-- Prove the correct compound proposition
theorem true_compound_proposition : correct_proposition :=
by
  sorry

end true_compound_proposition_l66_66953


namespace x_finishes_in_nine_days_l66_66075

-- Definitions based on the conditions
def x_work_rate : ℚ := 1 / 24
def y_work_rate : ℚ := 1 / 16
def y_days_worked : ℚ := 10
def y_work_done : ℚ := y_work_rate * y_days_worked
def remaining_work : ℚ := 1 - y_work_done
def x_days_to_finish : ℚ := remaining_work / x_work_rate

-- Statement to be proven
theorem x_finishes_in_nine_days : x_days_to_finish = 9 := 
by
  -- Skipping actual proof steps as instructed
  sorry

end x_finishes_in_nine_days_l66_66075


namespace exists_positive_integer_k_l66_66667

theorem exists_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → ¬ Nat.Prime (2^n * k + 1) ∧ 2^n * k + 1 > 1 :=
by
  sorry

end exists_positive_integer_k_l66_66667


namespace marcus_leah_together_l66_66861

def num_games_with_combination (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_games_together (total_players players_per_game : ℕ) (games_with_each_combination: ℕ) : ℕ :=
  total_players / players_per_game * games_with_each_combination

/-- Prove that Marcus and Leah play 210 games together. -/
theorem marcus_leah_together :
  let total_players := 12
  let players_per_game := 6
  let total_games := num_games_with_combination total_players players_per_game
  let marc_per_game := total_games / 2
  let together_pcnt := 5 / 11
  together_pcnt * marc_per_game = 210 :=
by
  sorry

end marcus_leah_together_l66_66861


namespace friends_reach_destinations_l66_66968

noncomputable def travel_times (d : ℕ) := 
  let walking_speed := 6
  let cycling_speed := 18
  let meet_time := d / (walking_speed + cycling_speed)
  let remaining_time := d / cycling_speed
  let total_time_A := meet_time + (d - cycling_speed * meet_time) / walking_speed
  let total_time_B := (cycling_speed * meet_time) / walking_speed + (d - cycling_speed * meet_time) / walking_speed
  let total_time_C := remaining_time + meet_time
  (total_time_A, total_time_B, total_time_C)

theorem friends_reach_destinations (d : ℕ) (d_eq_24 : d = 24) : 
  let (total_time_A, total_time_B, total_time_C) := travel_times d
  total_time_A ≤ 160 / 60 ∧ total_time_B ≤ 160 / 60 ∧ total_time_C ≤ 160 / 60 :=
by 
  sorry

end friends_reach_destinations_l66_66968


namespace find_x_l66_66726

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l66_66726


namespace snail_kite_eats_35_snails_l66_66391

theorem snail_kite_eats_35_snails : 
  let day1 := 3
  let day2 := day1 + 2
  let day3 := day2 + 2
  let day4 := day3 + 2
  let day5 := day4 + 2
  day1 + day2 + day3 + day4 + day5 = 35 := 
by
  sorry

end snail_kite_eats_35_snails_l66_66391


namespace area_of_U_l66_66891

noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ :=
  (side_length^2 * sqrt 3) / 4

noncomputable def transformed_area (original_area : ℝ) (expansion_factor : ℝ) : ℝ :=
  original_area * expansion_factor

theorem area_of_U :
  let side_length := sqrt 3
  let equilateral_triangle_area := equilateral_triangle_area side_length
  let expansion_factor := 4 in
  transformed_area equilateral_triangle_area expansion_factor = 3 * sqrt 3 :=
by
  sorry

end area_of_U_l66_66891


namespace correct_midpoint_l66_66490

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66490


namespace find_a_c_l66_66242

theorem find_a_c (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_neg : c < 0)
    (h_max : c + a = 3) (h_min : c - a = -5) :
  a = 4 ∧ c = -1 := 
sorry

end find_a_c_l66_66242


namespace cos_A_eq_sqrt3_div3_of_conditions_l66_66807

noncomputable def given_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) : Prop :=
  (Real.cos A = Real.sqrt 3 / 3)

theorem cos_A_eq_sqrt3_div3_of_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) :
  Real.cos A = Real.sqrt 3 / 3 :=
sorry

end cos_A_eq_sqrt3_div3_of_conditions_l66_66807


namespace hyperbola_midpoint_l66_66589

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66589


namespace steve_final_height_l66_66676

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l66_66676


namespace best_fit_model_l66_66305

theorem best_fit_model
  (R2_M1 R2_M2 R2_M3 R2_M4 : ℝ)
  (h1 : R2_M1 = 0.78)
  (h2 : R2_M2 = 0.85)
  (h3 : R2_M3 = 0.61)
  (h4 : R2_M4 = 0.31) :
  ∀ i, (i = 2 ∧ R2_M2 ≥ R2_M1 ∧ R2_M2 ≥ R2_M3 ∧ R2_M2 ≥ R2_M4) := 
sorry

end best_fit_model_l66_66305


namespace midpoint_of_line_segment_on_hyperbola_l66_66614

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66614


namespace solve_for_m_l66_66138

theorem solve_for_m (m x : ℝ) (h1 : 3 * m - 2 * x = 6) (h2 : x = 3) : m = 4 := by
  sorry

end solve_for_m_l66_66138


namespace Amanda_hiking_trip_l66_66090

-- Define the conditions
variable (x : ℝ) -- the total distance of Amanda's hiking trip
variable (forest_path : ℝ) (plain_path : ℝ)
variable (stream_path : ℝ) (mountain_path : ℝ)

-- Given conditions
axiom h1 : stream_path = (1/4) * x
axiom h2 : forest_path = 25
axiom h3 : mountain_path = (1/6) * x
axiom h4 : plain_path = 2 * forest_path
axiom h5 : stream_path + forest_path + mountain_path + plain_path = x

-- Proposition to prove
theorem Amanda_hiking_trip : x = 900 / 7 :=
by
  sorry

end Amanda_hiking_trip_l66_66090


namespace coyote_time_lemma_l66_66098

theorem coyote_time_lemma (coyote_speed darrel_speed : ℝ) (catch_up_time t : ℝ) 
  (h1 : coyote_speed = 15) (h2 : darrel_speed = 30) (h3 : catch_up_time = 1) (h4 : darrel_speed * catch_up_time = coyote_speed * t) :
  t = 2 :=
by
  sorry

end coyote_time_lemma_l66_66098


namespace sequence_expression_l66_66287

noncomputable def seq (n : ℕ) : ℝ := 
  match n with
  | 0 => 1  -- note: indexing from 1 means a_1 corresponds to seq 0 in Lean
  | m+1 => seq m / (3 * seq m + 1)

theorem sequence_expression (n : ℕ) : 
  ∀ n, seq (n + 1) = 1 / (3 * (n + 1) - 2) := 
sorry

end sequence_expression_l66_66287


namespace sum_of_digits_triangular_array_l66_66898

theorem sum_of_digits_triangular_array (N : ℕ) (h : N * (N + 1) / 2 = 5050) : 
  Nat.digits 10 N = [1, 0, 0] := by
  sorry

end sum_of_digits_triangular_array_l66_66898


namespace smallest_b_for_q_ge_half_l66_66747

open Nat

def binomial (n k : ℕ) : ℕ := if h : k ≤ n then n.choose k else 0

def q (b : ℕ) : ℚ := (binomial (32 - b) 2 + binomial (b - 1) 2) / (binomial 38 2 : ℕ)

theorem smallest_b_for_q_ge_half : ∃ (b : ℕ), b = 18 ∧ q b ≥ 1 / 2 :=
by
  -- Prove and find the smallest b such that q(b) ≥ 1/2
  sorry

end smallest_b_for_q_ge_half_l66_66747


namespace greatest_possible_value_l66_66074

theorem greatest_possible_value (A B C D : ℕ) 
    (h1 : A + B + C + D = 200) 
    (h2 : A + B = 70) 
    (h3 : 0 < A) 
    (h4 : 0 < B) 
    (h5 : 0 < C) 
    (h6 : 0 < D) : 
    C ≤ 129 := 
sorry

end greatest_possible_value_l66_66074


namespace num_five_ruble_coins_l66_66664

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l66_66664


namespace fraction_halfway_between_l66_66040

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66040


namespace fraction_half_way_l66_66017

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66017


namespace halfway_fraction_l66_66010

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66010


namespace midpoint_hyperbola_l66_66534

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66534


namespace max_n_for_factorable_polynomial_l66_66782

theorem max_n_for_factorable_polynomial : 
  ∃ n : ℤ, (∀ A B : ℤ, AB = 108 → n = 6 * B + A) ∧ n = 649 :=
by
  sorry

end max_n_for_factorable_polynomial_l66_66782


namespace logical_equivalence_l66_66398

variables {α : Type} (A B : α → Prop)

theorem logical_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end logical_equivalence_l66_66398


namespace midpoint_of_line_segment_on_hyperbola_l66_66611

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66611


namespace max_license_plates_is_correct_l66_66454

theorem max_license_plates_is_correct :
  let letters := 26
  let digits := 10
  (letters * (letters - 1) * digits^3 = 26 * 25 * 10^3) :=
by 
  sorry

end max_license_plates_is_correct_l66_66454


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66642

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66642


namespace midpoint_on_hyperbola_l66_66593

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66593


namespace midpoint_of_hyperbola_segment_l66_66501

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66501


namespace real_roots_exist_l66_66777

theorem real_roots_exist (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
by
  sorry  -- Proof goes here

end real_roots_exist_l66_66777


namespace factorize_xy_l66_66102

theorem factorize_xy (x y : ℕ): xy - x + y - 1 = (x + 1) * (y - 1) :=
by
  sorry

end factorize_xy_l66_66102


namespace total_attendance_l66_66903

-- Defining the given conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 1
def total_amount_collected : ℕ := 50
def number_of_child_tickets : ℕ := 18

-- Formulating the proof problem
theorem total_attendance (A : ℕ) (C : ℕ) (H1 : C = number_of_child_tickets)
  (H2 : adult_ticket_cost * A + child_ticket_cost * C = total_amount_collected) :
  A + C = 22 := by
  sorry

end total_attendance_l66_66903


namespace hyperbola_midpoint_exists_l66_66553

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66553


namespace max_min_vec_magnitude_l66_66440

noncomputable def vec_a (θ : ℝ) := (Real.cos θ, Real.sin θ)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)

noncomputable def vec_result (θ : ℝ) := (2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ - 1)

noncomputable def vec_magnitude (θ : ℝ) := Real.sqrt ((2 * Real.cos θ - Real.sqrt 3)^2 + (2 * Real.sin θ - 1)^2)

theorem max_min_vec_magnitude : 
  ∃ θ_max θ_min, 
    vec_magnitude θ_max = 4 ∧ 
    vec_magnitude θ_min = 0 :=
by
  sorry

end max_min_vec_magnitude_l66_66440


namespace grown_ups_in_milburg_l66_66368

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l66_66368


namespace midpoint_hyperbola_l66_66532

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66532


namespace midpoint_of_line_segment_on_hyperbola_l66_66619

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66619


namespace girls_boys_ratio_l66_66325

-- Let g be the number of girls and b be the number of boys.
-- From the conditions, we have:
-- 1. Total students: g + b = 32
-- 2. More girls than boys: g = b + 6

theorem girls_boys_ratio
  (g b : ℕ) -- Declare number of girls and boys as natural numbers
  (h1 : g + b = 32) -- Total number of students
  (h2 : g = b + 6)  -- 6 more girls than boys
  : g = 19 ∧ b = 13 := 
sorry

end girls_boys_ratio_l66_66325


namespace interest_rate_A_to_B_l66_66749

theorem interest_rate_A_to_B :
  ∀ (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ) (interest_C : ℝ) (interest_A : ℝ),
    principal = 3500 →
    rate_C = 0.13 →
    time = 3 →
    gain_B = 315 →
    interest_C = principal * rate_C * time →
    gain_B = interest_C - interest_A →
    interest_A = principal * (R / 100) * time →
    R = 10 := by
  sorry

end interest_rate_A_to_B_l66_66749


namespace number_of_girls_l66_66147

/-- In a school with 632 students, the average age of the boys is 12 years
and that of the girls is 11 years. The average age of the school is 11.75 years.
How many girls are there in the school? Prove that the number of girls is 108. -/
theorem number_of_girls (B G : ℕ) (h1 : B + G = 632) (h2 : 12 * B + 11 * G = 7428) :
  G = 108 :=
sorry

end number_of_girls_l66_66147


namespace find_other_number_l66_66343

theorem find_other_number (b : ℕ) (lcm_val gcd_val : ℕ)
  (h_lcm : Nat.lcm 240 b = 2520)
  (h_gcd : Nat.gcd 240 b = 24) :
  b = 252 :=
sorry

end find_other_number_l66_66343


namespace quadruple_dimensions_increase_volume_l66_66084

theorem quadruple_dimensions_increase_volume 
  (V_original : ℝ) (quad_factor : ℝ)
  (initial_volume : V_original = 5)
  (quad_factor_val : quad_factor = 4) :
  V_original * (quad_factor ^ 3) = 320 := 
by 
  -- Introduce necessary variables and conditions
  let V_modified := V_original * (quad_factor ^ 3)
  
  -- Assert the calculations based on the given conditions
  have initial : V_original = 5 := initial_volume
  have quad : quad_factor = 4 := quad_factor_val
  
  -- Skip the detailed proof with sorry
  sorry


end quadruple_dimensions_increase_volume_l66_66084


namespace midpoint_on_hyperbola_l66_66591

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66591


namespace hyperbola_midpoint_l66_66500

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66500


namespace angle_is_60_degrees_l66_66803

namespace AngleSupplement

theorem angle_is_60_degrees (α : ℝ) (h_sup : 180 - α = 2 * α) : α = 60 :=
by
  -- This is where the proof would go
  sorry

end AngleSupplement

end angle_is_60_degrees_l66_66803


namespace midpoint_of_hyperbola_segment_l66_66504

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66504


namespace find_number_l66_66716

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l66_66716


namespace find_ab_l66_66282

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end find_ab_l66_66282


namespace solution_set_of_inequality_l66_66693

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2 * x > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l66_66693


namespace equilateral_triangle_l66_66330

theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ca) : a = b ∧ b = c := 
by sorry

end equilateral_triangle_l66_66330


namespace students_appeared_l66_66810

def passed (T : ℝ) : ℝ := 0.35 * T
def B_grade_range (T : ℝ) : ℝ := 0.25 * T
def failed (T : ℝ) : ℝ := T - passed T

theorem students_appeared (T : ℝ) (hp : passed T = 0.35 * T)
    (hb : B_grade_range T = 0.25 * T) (hf : failed T = 481) :
    T = 740 :=
by
  -- proof goes here
  sorry

end students_appeared_l66_66810


namespace river_depth_is_correct_l66_66892

noncomputable def depth_of_river (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_min : ℝ) : ℝ :=
  let flow_rate_mpm := (flow_rate_kmph * 1000) / 60
  let cross_sectional_area := volume_per_min / flow_rate_mpm
  cross_sectional_area / width

theorem river_depth_is_correct :
  depth_of_river 65 6 26000 = 4 :=
by
  -- Steps to compute depth (converted from solution)
  sorry

end river_depth_is_correct_l66_66892


namespace power_of_two_as_sum_of_squares_l66_66169

theorem power_of_two_as_sum_of_squares (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), x % 2 = 1 ∧ y % 2 = 1 ∧ (2^n = 7*x^2 + y^2) :=
by
  sorry

end power_of_two_as_sum_of_squares_l66_66169


namespace imaginary_part_of_z_l66_66844

-- Let 'z' be the complex number \(\frac {2i}{1-i}\)
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

theorem imaginary_part_of_z :
  z.im = 1 :=
sorry

end imaginary_part_of_z_l66_66844


namespace total_selling_price_correct_l66_66751

-- Define the cost prices of the three articles
def cost_A : ℕ := 400
def cost_B : ℕ := 600
def cost_C : ℕ := 800

-- Define the desired profit percentages for the three articles
def profit_percent_A : ℚ := 40 / 100
def profit_percent_B : ℚ := 35 / 100
def profit_percent_C : ℚ := 25 / 100

-- Define the selling prices of the three articles
def selling_price_A : ℚ := cost_A * (1 + profit_percent_A)
def selling_price_B : ℚ := cost_B * (1 + profit_percent_B)
def selling_price_C : ℚ := cost_C * (1 + profit_percent_C)

-- Define the total selling price
def total_selling_price : ℚ := selling_price_A + selling_price_B + selling_price_C

-- The proof statement
theorem total_selling_price_correct : total_selling_price = 2370 :=
sorry

end total_selling_price_correct_l66_66751


namespace part_a_part_b_l66_66285

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Prove that (1, 1) lies on the parabola
theorem part_a : parabola 1 = 1 := by
  sorry

-- Prove that for any t, (t, t^2) lies on the parabola
theorem part_b (t : ℝ) : parabola t = t^2 := by
  sorry

end part_a_part_b_l66_66285


namespace midpoint_hyperbola_l66_66535

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66535


namespace midpoint_of_hyperbola_l66_66476

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66476


namespace juwella_read_more_last_night_l66_66254

-- Definitions of the conditions
def pages_three_nights_ago : ℕ := 15
def book_pages : ℕ := 100
def pages_tonight : ℕ := 20
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago
def total_pages_before_tonight : ℕ := book_pages - pages_tonight
def pages_last_night : ℕ := total_pages_before_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem juwella_read_more_last_night :
  pages_last_night - pages_two_nights_ago = 5 :=
by
  sorry

end juwella_read_more_last_night_l66_66254


namespace g_of_f_of_3_eq_1902_l66_66622

def f (x : ℕ) := x^3 - 2
def g (x : ℕ) := 3 * x^2 + x + 2

theorem g_of_f_of_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_of_3_eq_1902_l66_66622


namespace midpoint_of_hyperbola_l66_66477

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66477


namespace sum_of_extreme_values_of_g_l66_66773

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - 2 * abs (x - 3)

theorem sum_of_extreme_values_of_g :
  ∃ (min_val max_val : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≥ min_val) ∧ 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → g x ≤ max_val) ∧ 
    (min_val = -8) ∧ 
    (max_val = 0) ∧ 
    (min_val + max_val = -8) := 
by
  sorry

end sum_of_extreme_values_of_g_l66_66773


namespace sum_of_squares_l66_66290

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 :=
by
  sorry

end sum_of_squares_l66_66290


namespace angle_through_point_l66_66300

theorem angle_through_point : 
  (∃ θ : ℝ, ∃ k : ℤ, θ = 2 * k * Real.pi + 5 * Real.pi / 6 ∧ 
                      ∃ x y : ℝ, x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 ∧ 
                                    y / x = Real.tan θ) := 
sorry

end angle_through_point_l66_66300


namespace units_digit_G1000_l66_66163

def units_digit (n : ℕ) : ℕ :=
  n % 10

def power_cycle : List ℕ := [3, 9, 7, 1]

def G (n : ℕ) : ℕ :=
  3^(2^n) + 2

theorem units_digit_G1000 : units_digit (G 1000) = 3 :=
by
  sorry

end units_digit_G1000_l66_66163


namespace problem_statement_l66_66216

theorem problem_statement :
  ¬ (3^2 = 6) ∧ 
  ¬ ((-1 / 4) / (-4) = 1) ∧
  ¬ ((-8)^2 = -16) ∧
  (-5 - (-2) = -3) := 
by 
  sorry

end problem_statement_l66_66216


namespace midpoint_of_hyperbola_l66_66572

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66572


namespace can_be_midpoint_of_AB_l66_66519

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66519


namespace derivative_y_at_1_l66_66339

-- Define the function y = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the proposition that the derivative at x=1 is 2
theorem derivative_y_at_1 : deriv f 1 = 2 :=
by sorry

end derivative_y_at_1_l66_66339


namespace positive_difference_l66_66697

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end positive_difference_l66_66697


namespace find_number_l66_66719

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l66_66719


namespace shirt_discount_l66_66626

theorem shirt_discount (original_price discounted_price : ℕ) 
  (h1 : original_price = 22) 
  (h2 : discounted_price = 16) : 
  original_price - discounted_price = 6 := 
by
  sorry

end shirt_discount_l66_66626


namespace unique_n0_exists_l66_66160

open Set

theorem unique_n0_exists 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h3 : ∀ n : ℕ, S 0 = a 0) :
  ∃! n_0 : ℕ, (S (n_0 + 1)) / n_0 > a (n_0 + 1)
             ∧ (S (n_0 + 1)) / n_0 ≤ a (n_0 + 2) := 
sorry

end unique_n0_exists_l66_66160


namespace hyperbola_midpoint_l66_66584

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66584


namespace midpoint_on_hyperbola_l66_66595

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66595


namespace unguarded_area_eq_225_l66_66213

-- Define the basic conditions of the problem in Lean
structure Room where
  side_length : ℕ
  unguarded_fraction : ℚ
  deriving Repr

-- Define the specific room used in the problem
def problemRoom : Room :=
  { side_length := 10,
    unguarded_fraction := 9/4 }

-- Define the expected unguarded area in square meters
def expected_unguarded_area (r : Room) : ℚ :=
  r.unguarded_fraction * (r.side_length ^ 2)

-- Prove that the unguarded area is 225 square meters
theorem unguarded_area_eq_225 (r : Room) (h : r = problemRoom) : expected_unguarded_area r = 225 := by
  -- The proof in this case is omitted.
  sorry

end unguarded_area_eq_225_l66_66213


namespace midpoint_of_hyperbola_l66_66571

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66571


namespace rect_solution_proof_l66_66121

noncomputable def rect_solution_exists : Prop :=
  ∃ (l2 w2 : ℝ), 2 * (l2 + w2) = 12 ∧ l2 * w2 = 4 ∧
               l2 = 3 + Real.sqrt 5 ∧ w2 = 3 - Real.sqrt 5

theorem rect_solution_proof : rect_solution_exists :=
  by
    sorry

end rect_solution_proof_l66_66121


namespace midpoint_of_hyperbola_l66_66578

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66578


namespace rotated_line_equation_l66_66342

-- Define the original equation of the line
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the rotated line equation we want to prove
def rotated_line (x y : ℝ) : Prop := -x + 2 * y + 4 = 0

-- Proof problem statement in Lean 4
theorem rotated_line_equation :
  ∀ (x y : ℝ), original_line x y → rotated_line x y :=
by
  sorry

end rotated_line_equation_l66_66342


namespace no_parallelogram_on_convex_graph_l66_66159

-- Definition of strictly convex function
def is_strictly_convex (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x t y : ℝ⦄, (x < t ∧ t < y) → f t < ((f y - f x) / (y - x)) * (t - x) + f x

-- The main statement of the problem
theorem no_parallelogram_on_convex_graph (f : ℝ → ℝ) :
  is_strictly_convex f →
  ¬ ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (f b < (f c - f a) / (c - a) * (b - a) + f a) ∧
    (f c < (f d - f b) / (d - b) * (c - b) + f b) :=
sorry

end no_parallelogram_on_convex_graph_l66_66159


namespace number_of_children_l66_66853

theorem number_of_children (total_passengers men women : ℕ) (h1 : total_passengers = 54) (h2 : men = 18) (h3 : women = 26) : 
  total_passengers - men - women = 10 :=
by sorry

end number_of_children_l66_66853


namespace Faye_can_still_make_8_bouquets_l66_66781

theorem Faye_can_still_make_8_bouquets (total_flowers : ℕ) (wilted_flowers : ℕ) (flowers_per_bouquet : ℕ) 
(h1 : total_flowers = 88) 
(h2 : wilted_flowers = 48) 
(h3 : flowers_per_bouquet = 5) : 
(total_flowers - wilted_flowers) / flowers_per_bouquet = 8 := 
by
  sorry

end Faye_can_still_make_8_bouquets_l66_66781


namespace ending_number_of_second_range_l66_66184

theorem ending_number_of_second_range :
  let avg100_400 := (100 + 400) / 2
  let avg_50_n := (50 + n) / 2
  avg100_400 = avg_50_n + 100 → n = 250 :=
by
  sorry

end ending_number_of_second_range_l66_66184


namespace petya_five_ruble_coins_l66_66659

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l66_66659


namespace midpoint_on_hyperbola_l66_66565

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66565


namespace max_b_for_integer_solutions_l66_66187

theorem max_b_for_integer_solutions (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
sorry

end max_b_for_integer_solutions_l66_66187


namespace integer_solutions_count_l66_66681

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ x y, (x, y) ∈ S ↔ x^2 + x * y + 2 * y^2 = 29) ∧ 
  S.card = 4 := 
sorry

end integer_solutions_count_l66_66681


namespace evaluate_complex_modulus_l66_66253

namespace ComplexProblem

open Complex

theorem evaluate_complex_modulus : 
  abs ((1 / 2 : ℂ) - (3 / 8) * Complex.I) = 5 / 8 :=
by
  sorry

end ComplexProblem

end evaluate_complex_modulus_l66_66253


namespace terez_farm_pregnant_cows_percentage_l66_66183

theorem terez_farm_pregnant_cows_percentage (total_cows : ℕ) (female_percentage : ℕ) (pregnant_females : ℕ) 
  (ht : total_cows = 44) (hf : female_percentage = 50) (hp : pregnant_females = 11) :
  (pregnant_females * 100 / (female_percentage * total_cows / 100) = 50) :=
by 
  sorry

end terez_farm_pregnant_cows_percentage_l66_66183


namespace minimum_a_3b_exists_positive_solution_l66_66153

noncomputable def positive_solution_exists : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (1 / (a + 3) + 1 / (b + 3) = 1 / 4) ∧ (a + 3b = 4 + 8 * Real.sqrt 3)

theorem minimum_a_3b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
  (h₃ : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) :
  a + 3b ≥ 4 + 8 * Real.sqrt 3 :=
by
  sorry

theorem exists_positive_solution :
  positive_solution_exists :=
by
  sorry

end minimum_a_3b_exists_positive_solution_l66_66153


namespace distance_between_points_l66_66103

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l66_66103


namespace intersection_M_N_l66_66826

open Set Real

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - abs x)

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l66_66826


namespace largest_6_digit_div_by_88_l66_66869

theorem largest_6_digit_div_by_88 : ∃ n : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 88 ∣ n ∧ (∀ m : ℕ, 100000 ≤ m ∧ m ≤ 999999 ∧ 88 ∣ m → m ≤ n) ∧ n = 999944 :=
by
  sorry

end largest_6_digit_div_by_88_l66_66869


namespace midpoint_of_hyperbola_l66_66574

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66574


namespace can_be_midpoint_of_AB_l66_66512

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66512


namespace decimal_properties_l66_66874

theorem decimal_properties :
  (3.00 : ℝ) = (3 : ℝ) :=
by sorry

end decimal_properties_l66_66874


namespace total_attendance_l66_66900

theorem total_attendance (A C : ℕ) (ticket_sales : ℕ) (adult_ticket_cost child_ticket_cost : ℕ) (total_collected : ℕ)
    (h1 : C = 18) (h2 : ticket_sales = 50) (h3 : adult_ticket_cost = 8) (h4 : child_ticket_cost = 1)
    (h5 : ticket_sales = adult_ticket_cost * A + child_ticket_cost * C) :
    A + C = 22 :=
by {
  sorry
}

end total_attendance_l66_66900


namespace second_hand_bisect_angle_l66_66990

theorem second_hand_bisect_angle :
  ∃ x : ℚ, (6 * x - 360 * (x - 1) = 360 * (x - 1) - 0.5 * x) ∧ (x = 1440 / 1427) :=
by
  sorry

end second_hand_bisect_angle_l66_66990


namespace base4_last_digit_390_l66_66910

theorem base4_last_digit_390 : 
  (Nat.digits 4 390).head! = 2 := sorry

end base4_last_digit_390_l66_66910


namespace petya_five_ruble_coins_l66_66658

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l66_66658


namespace consecutive_green_balls_l66_66377

theorem consecutive_green_balls : ∃ (fill_ways : ℕ), fill_ways = 21 ∧ 
  (∃ (boxes : Fin 6 → Bool), 
    (∀ i, boxes i = true → 
      (∀ j, boxes j = true → (i ≤ j ∨ j ≤ i)) ∧ 
      ∃ k, boxes k = true)) :=
by
  sorry

end consecutive_green_balls_l66_66377


namespace midpoint_on_hyperbola_l66_66522

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66522


namespace midpoint_of_hyperbola_segment_l66_66505

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66505


namespace hyperbola_midpoint_l66_66583

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66583


namespace cubic_equation_real_root_l66_66909

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x + 25 = 0 := 
sorry

end cubic_equation_real_root_l66_66909


namespace linear_function_quadrants_l66_66925

theorem linear_function_quadrants (k b : ℝ) 
  (h1 : k < 0)
  (h2 : b < 0) 
  : k * b > 0 := 
sorry

end linear_function_quadrants_l66_66925


namespace total_reading_materials_l66_66948

theorem total_reading_materials 
  (magazines : ℕ) 
  (newspapers : ℕ) 
  (h_magazines : magazines = 425) 
  (h_newspapers : newspapers = 275) : 
  magazines + newspapers = 700 := 
by 
  sorry

end total_reading_materials_l66_66948


namespace trapezoid_area_l66_66862

theorem trapezoid_area (base1 base2 height : ℕ) (h_base1 : base1 = 9) (h_base2 : base2 = 11) (h_height : height = 3) :
  (1 / 2 : ℚ) * (base1 + base2 : ℕ) * height = 30 :=
by
  sorry

end trapezoid_area_l66_66862


namespace smallest_difference_l66_66857

noncomputable def triangle_lengths (DE EF FD : ℕ) : Prop :=
  DE < EF ∧ EF ≤ FD ∧ DE + EF + FD = 3010 ∧ DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference :
  ∃ (DE EF FD : ℕ), triangle_lengths DE EF FD ∧ EF - DE = 1 :=
by
  sorry

end smallest_difference_l66_66857


namespace trigonometric_identity_l66_66935

theorem trigonometric_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + 1 / Real.tan θ = 2 :=
by
  sorry

end trigonometric_identity_l66_66935


namespace problem_l66_66078

theorem problem (a : ℝ) : (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end problem_l66_66078


namespace minimum_employees_needed_l66_66756

theorem minimum_employees_needed
  (n_W : ℕ) (n_A : ℕ) (n_S : ℕ)
  (n_WA : ℕ) (n_AS : ℕ) (n_SW : ℕ)
  (n_WAS : ℕ)
  (h_W : n_W = 115)
  (h_A : n_A = 92)
  (h_S : n_S = 60)
  (h_WA : n_WA = 32)
  (h_AS : n_AS = 20)
  (h_SW : n_SW = 10)
  (h_WAS : n_WAS = 5) :
  n_W + n_A + n_S - (n_WA - n_WAS) - (n_AS - n_WAS) - (n_SW - n_WAS) + 2 * n_WAS = 225 :=
by
  sorry

end minimum_employees_needed_l66_66756


namespace problem1_problem2_problem3_l66_66954

section problem

variable (m : ℝ)

-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions
def p : Prop := (16 * m^2 - 4) ≥ 0

-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀^2 - 2x₀ - 1 > 0
def q : Prop := ∃ (x₀ : ℝ), (m * x₀^2 - 2 * x₀ - 1) > 0

-- Solution to (1): If p is true, the range of values for m
theorem problem1 (hp : p m) : m ≥ 1/2 ∨ m ≤ -1/2 := sorry

-- Solution to (2): If q is true, the range of values for m
theorem problem2 (hq : q m) : m > -1 := sorry

-- Solution to (3): If both p and q are false but either p or q is true,
-- find the range of values for m
theorem problem3 (hnp : ¬p m) (hnq : ¬q m) (hpq : p m ∨ q m) : -1 < m ∧ m < 1/2 := sorry

end problem

end problem1_problem2_problem3_l66_66954


namespace grown_ups_in_milburg_l66_66365

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l66_66365


namespace hyperbola_midpoint_l66_66606

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66606


namespace max_possible_value_of_a_l66_66318

theorem max_possible_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) : 
  a ≤ 8924 :=
by {
  sorry
}

end max_possible_value_of_a_l66_66318


namespace seats_taken_correct_l66_66268

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end seats_taken_correct_l66_66268


namespace prob_each_student_gets_each_snack_l66_66085

-- Define the total number of snacks and their types
def total_snacks := 16
def snack_types := 4

-- Define the conditions for the problem
def students := 4
def snacks_per_type := 4

-- Define the probability calculation.
-- We would typically use combinatorial functions here, but for simplicity, use predefined values from the solution.
def prob_student_1 := 64 / 455
def prob_student_2 := 9 / 55
def prob_student_3 := 8 / 35
def prob_student_4 := 1 -- Always 1 for the final student's remaining snacks

-- Calculate the total probability
def total_prob := prob_student_1 * prob_student_2 * prob_student_3 * prob_student_4

-- The statement to prove the desired probability outcome
theorem prob_each_student_gets_each_snack : total_prob = (64 / 1225) :=
by
  sorry

end prob_each_student_gets_each_snack_l66_66085


namespace truncated_pyramid_ratio_l66_66426

noncomputable def volume_prism (L1 H : ℝ) : ℝ := L1^2 * H
noncomputable def volume_truncated_pyramid (L1 L2 H : ℝ) : ℝ := 
  (H / 3) * (L1^2 + L1 * L2 + L2^2)

theorem truncated_pyramid_ratio (L1 L2 H : ℝ) 
  (h_vol : volume_truncated_pyramid L1 L2 H = (2/3) * volume_prism L1 H) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end truncated_pyramid_ratio_l66_66426


namespace min_value_x2_y2_z2_l66_66429

theorem min_value_x2_y2_z2 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + 2 * y + 3 * z = 2) : 
  x^2 + y^2 + z^2 ≥ 2 / 7 :=
sorry

end min_value_x2_y2_z2_l66_66429


namespace parabola_focus_l66_66796

-- Definitions used in the conditions
def parabola_eq (p : ℝ) (x : ℝ) : ℝ := 2 * p * x^2
def passes_through (p : ℝ) : Prop := parabola_eq p 1 = 4

-- The proof that the coordinates of the focus are (0, 1/16) given the conditions
theorem parabola_focus (p : ℝ) (h : passes_through p) : p = 2 → (0, 1 / 16) = (0, 1 / (4 * p)) :=
by
  sorry

end parabola_focus_l66_66796


namespace hyperbola_midpoint_l66_66586

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66586


namespace midpoint_of_hyperbola_l66_66579

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66579


namespace fraction_half_way_l66_66023

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66023


namespace sum_of_arithmetic_terms_l66_66352

theorem sum_of_arithmetic_terms (a₁ a₂ a₃ c d a₆ : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₂ = 10)
  (h₃ : a₃ = 17)
  (h₄ : a₆ = 32)
  (h_arith : ∀ n, (a₁ + n * (a₂ - a₁)) = seq)
  : c + d = 55 :=
by
  have d := a₂ - a₁
  have c := a₃ + d
  have d := c + d
  have h_seq := list.map (λ n, (a₁ + n * d)) (list.range 6) -- Making use of the arithmetic property
  have h_seq_eq := h_seq = [3, 10, 17, c, d, 32]
  sorry

end sum_of_arithmetic_terms_l66_66352


namespace steve_height_equiv_l66_66678

/-- 
  Steve's initial height in feet and inches.
  convert_height: converts feet and inches to total inches.
  grows further: additional height Steve grows.
  expected height: the expected total height after growing.
--/

def initial_height_feet := 5
def initial_height_inches := 6
def additional_height := 6

def convert_height(feet: Int, inches: Int): Int := 
  feet * 12 + inches

def expected_height(initial_feet: Int, initial_inches: Int, additional: Int): Int := 
  convert_height(initial_feet, initial_inches) + additional

theorem steve_height_equiv:
  expected_height initial_height_feet initial_height_inches additional_height = 72 :=
by
  sorry

end steve_height_equiv_l66_66678


namespace tyrone_gave_25_marbles_l66_66704

/-- Given that Tyrone initially had 97 marbles and Eric had 11 marbles, and after
    giving some marbles to Eric, Tyrone ended with twice as many marbles as Eric,
    we need to find the number of marbles Tyrone gave to Eric. -/
theorem tyrone_gave_25_marbles (x : ℕ) (t0 e0 : ℕ)
  (hT0 : t0 = 97)
  (hE0 : e0 = 11)
  (hT_end : (t0 - x) = 2 * (e0 + x)) :
  x = 25 := 
  sorry

end tyrone_gave_25_marbles_l66_66704


namespace sock_combination_count_l66_66699

noncomputable def numSockCombinations : Nat :=
  let striped := 4
  let solid := 4
  let checkered := 4
  let striped_and_solid := striped * solid
  let striped_and_checkered := striped * checkered
  striped_and_solid + striped_and_checkered

theorem sock_combination_count :
  numSockCombinations = 32 :=
by
  unfold numSockCombinations
  sorry

end sock_combination_count_l66_66699


namespace stratified_sampling_l66_66882

-- Conditions
def total_students : ℕ := 1200
def freshmen : ℕ := 300
def sophomores : ℕ := 400
def juniors : ℕ := 500
def sample_size : ℕ := 60
def probability : ℚ := sample_size / total_students

-- Number of students to be sampled from each grade
def freshmen_sampled : ℚ := freshmen * probability
def sophomores_sampled : ℚ := sophomores * probability
def juniors_sampled : ℚ := juniors * probability

-- Theorem to prove
theorem stratified_sampling :
  freshmen_sampled = 15 ∧ sophomores_sampled = 20 ∧ juniors_sampled = 25 :=
by
  -- The actual proof would go here
  sorry

end stratified_sampling_l66_66882


namespace ab_is_zero_l66_66128

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem ab_is_zero (a b : ℝ) (h : a - 1 = 0) : a * b = 0 := by
  sorry

end ab_is_zero_l66_66128


namespace midpoint_on_hyperbola_l66_66594

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66594


namespace problem1_problem2_l66_66872

theorem problem1 : 3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3)^0 + abs (Real.sqrt 3 - 2) = 3 := 
by
  sorry

theorem problem2 : (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / Real.sqrt 3 = 28 / 3 :=
by
  sorry

end problem1_problem2_l66_66872


namespace sum_of_reciprocals_of_numbers_l66_66995

theorem sum_of_reciprocals_of_numbers (x y : ℕ) (h_sum : x + y = 45) (h_hcf : Nat.gcd x y = 3)
    (h_lcm : Nat.lcm x y = 100) : 1/x + 1/y = 3/20 := 
by 
  sorry

end sum_of_reciprocals_of_numbers_l66_66995


namespace rainfall_difference_l66_66144

-- Defining the conditions
def march_rainfall : ℝ := 0.81
def april_rainfall : ℝ := 0.46

-- Stating the theorem
theorem rainfall_difference : march_rainfall - april_rainfall = 0.35 := by
  -- insert proof steps here
  sorry

end rainfall_difference_l66_66144


namespace midpoint_of_line_segment_on_hyperbola_l66_66615

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66615


namespace p_minus_q_value_l66_66322

theorem p_minus_q_value (p q : ℝ) (h1 : (x - 4) * (x + 4) = 24 * x - 96) (h2 : x^2 - 24 * x + 80 = 0) (h3 : p = 20) (h4 : q = 4) : p - q = 16 :=
by
  sorry

end p_minus_q_value_l66_66322


namespace min_value_of_fraction_l66_66115

theorem min_value_of_fraction (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) : 
  (∀ n m, 2 * n + m = 4 ∧ m > 0 ∧ n > 0 → ∀ y, y = 2 / m + 1 / n → y ≥ 2) :=
by sorry

end min_value_of_fraction_l66_66115


namespace total_granola_bars_l66_66272

-- Problem conditions
def oatmeal_raisin_bars : ℕ := 6
def peanut_bars : ℕ := 8

-- Statement to prove
theorem total_granola_bars : oatmeal_raisin_bars + peanut_bars = 14 := 
by 
  sorry

end total_granola_bars_l66_66272


namespace number_divided_by_three_l66_66723

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l66_66723


namespace find_larger_integer_l66_66691

noncomputable def larger_integer (a b : ℕ) : Prop :=
  a * b = 189 ∧ (b = (7 * a) / 3⁷) / 3

theorem find_larger_integer (a b : ℕ) (h1 : a * b = 189) (h2 : a * 7 = 3 * b) :
  b = 21 :=
by
  sorry

end find_larger_integer_l66_66691


namespace base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l66_66448

theorem base_number_pow_k_eq_4_pow_2k_plus_2_eq_64 (x k : ℝ) (h1 : x^k = 4) (h2 : x^(2 * k + 2) = 64) : x = 2 :=
sorry

end base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l66_66448


namespace hyperbola_midpoint_l66_66492

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66492


namespace Phoenix_roots_prod_l66_66775

def Phoenix_eqn (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a + b + c = 0

theorem Phoenix_roots_prod {m n : ℝ} (hPhoenix : Phoenix_eqn 1 m n)
  (hEqualRoots : (m^2 - 4 * n) = 0) : m * n = -2 :=
by sorry

end Phoenix_roots_prod_l66_66775


namespace c_divides_n_l66_66319

theorem c_divides_n (a b c n : ℤ) (h : a * n^2 + b * n + c = 0) : c ∣ n :=
sorry

end c_divides_n_l66_66319


namespace positive_difference_l66_66694

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end positive_difference_l66_66694


namespace meatballs_left_l66_66205
open Nat

theorem meatballs_left (meatballs_per_plate sons : ℕ)
  (hp : meatballs_per_plate = 3) 
  (hs : sons = 3) 
  (fraction_eaten : ℚ)
  (hf : fraction_eaten = 2 / 3): 
  (meatballs_per_plate - meatballs_per_plate * fraction_eaten) * sons = 3 := by
  -- Placeholder proof; the details would be filled in by a full proof.
  sorry

end meatballs_left_l66_66205


namespace slower_speed_is_l66_66752

def slower_speed_problem
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (actual_distance : ℝ)
  (v : ℝ) :
  Prop :=
  actual_distance / v = (actual_distance + additional_distance) / faster_speed

theorem slower_speed_is
  (h1 : faster_speed = 25)
  (h2 : additional_distance = 20)
  (h3 : actual_distance = 13.333333333333332)
  : ∃ v : ℝ,  slower_speed_problem faster_speed additional_distance actual_distance v ∧ v = 10 :=
by {
  sorry
}

end slower_speed_is_l66_66752


namespace curry_draymond_ratio_l66_66458

theorem curry_draymond_ratio :
  ∃ (curry draymond kelly durant klay : ℕ),
    draymond = 12 ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    curry + draymond + kelly + durant + klay = 69 ∧
    curry = 24 ∧ -- Curry's points calculated in the solution
    draymond = 12 → -- Draymond's points reaffirmed
    curry / draymond = 2 :=
by
  sorry

end curry_draymond_ratio_l66_66458


namespace nickel_chocolates_l66_66834

theorem nickel_chocolates (N : ℕ) (h : 7 = N + 2) : N = 5 :=
by
  sorry

end nickel_chocolates_l66_66834


namespace students_per_bench_l66_66194

theorem students_per_bench (num_male num_benches : ℕ) (h₁ : num_male = 29) (h₂ : num_benches = 29) (h₃ : ∀ num_female, num_female = 4 * num_male) : 
  ((29 + 4 * 29) / 29) = 5 :=
by
  sorry

end students_per_bench_l66_66194


namespace proof_time_to_run_square_field_l66_66741

def side : ℝ := 40
def speed_kmh : ℝ := 9
def perimeter (side : ℝ) : ℝ := 4 * side

noncomputable def speed_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def time_to_run (perimeter : ℝ) (speed_mps : ℝ) : ℝ := perimeter / speed_mps

theorem proof_time_to_run_square_field :
  time_to_run (perimeter side) (speed_mps speed_kmh) = 64 :=
by
  sorry

end proof_time_to_run_square_field_l66_66741


namespace barry_more_votes_than_joey_l66_66808

theorem barry_more_votes_than_joey {M B J X : ℕ} 
  (h1 : M = 66)
  (h2 : J = 8)
  (h3 : M = 3 * B)
  (h4 : B = 2 * (J + X)) :
  B - J = 14 := by
  sorry

end barry_more_votes_than_joey_l66_66808


namespace midpoint_hyperbola_l66_66543

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66543


namespace midpoint_of_line_segment_on_hyperbola_l66_66613

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66613


namespace correct_midpoint_l66_66489

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66489


namespace min_n_for_binomial_constant_term_l66_66970

theorem min_n_for_binomial_constant_term : ∃ (n : ℕ), n > 0 ∧ 3 * n - 7 * ((3 * n) / 7) = 0 ∧ n = 7 :=
by {
  sorry
}

end min_n_for_binomial_constant_term_l66_66970


namespace integral_cosine_l66_66310

noncomputable def a : ℝ := 2 * Real.pi / 3

theorem integral_cosine (ha : a = 2 * Real.pi / 3) :
  ∫ x in -a..a, Real.cos x = Real.sqrt 3 := 
sorry

end integral_cosine_l66_66310


namespace midpoint_of_hyperbola_l66_66472

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66472


namespace halfway_fraction_l66_66015

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66015


namespace find_n_l66_66883

-- Defining the parameters and conditions
def large_block_positions (n : ℕ) : ℕ := 199 * n + 110 * (n - 1)

-- Theorem statement
theorem find_n (h : large_block_positions n = 2362) : n = 8 :=
sorry

end find_n_l66_66883


namespace petya_five_ruble_coins_count_l66_66661

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l66_66661


namespace coeffs_equal_implies_a_plus_b_eq_4_l66_66312

theorem coeffs_equal_implies_a_plus_b_eq_4 (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq_coeffs : (Nat.choose 2000 1998) * (a ^ 2) * (b ^ 1998) = (Nat.choose 2000 1997) * (a ^ 3) * (b ^ 1997)) :
  a + b = 4 := 
sorry

end coeffs_equal_implies_a_plus_b_eq_4_l66_66312


namespace midpoint_on_hyperbola_l66_66530

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66530


namespace find_side_a_find_area_l66_66806

-- Definitions from the conditions
variables {A B C : ℝ} 
variables {a b c : ℝ}
variable (angle_B: B = 120 * Real.pi / 180)
variable (side_b: b = Real.sqrt 7)
variable (side_c: c = 1)

-- The first proof problem: Prove that a = 2 given the above conditions
theorem find_side_a (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_cos_formula: b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) : a = 2 :=
  by
  sorry

-- The second proof problem: Prove that the area is sqrt(3)/2 given the above conditions
theorem find_area (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_side_a: a = 2) : (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
  by
  sorry

end find_side_a_find_area_l66_66806


namespace solve_inequality_l66_66261

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x ^ 3 - 3 * x ^ 2 + 2 * x) / (x ^ 2 - 3 * x + 2) ≤ 0 ∧
  x ≠ 1 ∧ x ≠ 2

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2} :=
  sorry

end solve_inequality_l66_66261


namespace similar_triangles_perimeters_and_area_ratios_l66_66344

theorem similar_triangles_perimeters_and_area_ratios
  (m1 m2 : ℝ) (p_sum : ℝ) (ratio_p : ℝ) (ratio_a : ℝ) :
  m1 = 10 →
  m2 = 4 →
  p_sum = 140 →
  ratio_p = 5 / 2 →
  ratio_a = 25 / 4 →
  (∃ (p1 p2 : ℝ), p1 + p2 = p_sum ∧ p1 = (5 / 7) * p_sum ∧ p2 = (2 / 7) * p_sum ∧ ratio_a = (ratio_p)^2) :=
by
  sorry

end similar_triangles_perimeters_and_area_ratios_l66_66344


namespace snail_kite_snails_eaten_l66_66393

theorem snail_kite_snails_eaten 
  (a₀ : ℕ) (a₁ : ℕ) (a₂ : ℕ) (a₃ : ℕ) (a₄ : ℕ)
  (h₀ : a₀ = 3)
  (h₁ : a₁ = a₀ + 2)
  (h₂ : a₂ = a₁ + 2)
  (h₃ : a₃ = a₂ + 2)
  (h₄ : a₄ = a₃ + 2)
  : a₀ + a₁ + a₂ + a₃ + a₄ = 35 := 
by 
  sorry

end snail_kite_snails_eaten_l66_66393


namespace sum_even_pos_integers_less_than_100_l66_66063

theorem sum_even_pos_integers_less_than_100 : 
  (∑ i in Finset.filter (λ n, n % 2 = 0) (Finset.range 100), i) = 2450 :=
by
  sorry

end sum_even_pos_integers_less_than_100_l66_66063


namespace price_of_each_apple_l66_66833

theorem price_of_each_apple
  (bike_cost: ℝ) (repair_cost_percent: ℝ) (remaining_percentage: ℝ)
  (total_apples_sold: ℕ) (repair_cost: ℝ) (total_money_earned: ℝ)
  (price_per_apple: ℝ) :
  bike_cost = 80 →
  repair_cost_percent = 0.25 →
  remaining_percentage = 0.2 →
  total_apples_sold = 20 →
  repair_cost = repair_cost_percent * bike_cost →
  total_money_earned = repair_cost / (1 - remaining_percentage) →
  price_per_apple = total_money_earned / total_apples_sold →
  price_per_apple = 1.25 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end price_of_each_apple_l66_66833


namespace midpoint_on_hyperbola_l66_66596

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66596


namespace abs_diff_eq_implies_le_l66_66077

theorem abs_diff_eq_implies_le {x y : ℝ} (h : |x - y| = y - x) : x ≤ y := 
by
  sorry

end abs_diff_eq_implies_le_l66_66077


namespace isosceles_base_length_l66_66306

theorem isosceles_base_length (x b : ℕ) (h1 : 2 * x + b = 40) (h2 : x = 15) : b = 10 :=
by
  sorry

end isosceles_base_length_l66_66306


namespace div_by_10_3pow_l66_66939

theorem div_by_10_3pow
    (m : ℤ)
    (n : ℕ)
    (h : (3^n + m) % 10 = 0) :
    (3^(n + 4) + m) % 10 = 0 := by
  sorry

end div_by_10_3pow_l66_66939


namespace sum_even_integers_less_than_100_l66_66057

theorem sum_even_integers_less_than_100 :
  let a := 2
  let d := 2
  let n := 49
  let l := a + (n - 1) * d
  l = 98 ∧ n = 49 →
  let sum := n * (a + l) / 2
  sum = 2450 :=
by
  intros a d n l h1 h2
  rw [h1, h2]
  sorry

end sum_even_integers_less_than_100_l66_66057


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66643

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66643


namespace problem_I_problem_II_l66_66433

noncomputable def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem problem_I (x m : ℝ) : f x m ≥ 2 :=
by 
  sorry

theorem problem_II (m : ℝ) : f 2 m ≤ 16 ↔ -3 ≤ m ∧ m ≤ Real.sqrt 14 - 1 :=
by 
  sorry

end problem_I_problem_II_l66_66433


namespace steve_height_equiv_l66_66679

/-- 
  Steve's initial height in feet and inches.
  convert_height: converts feet and inches to total inches.
  grows further: additional height Steve grows.
  expected height: the expected total height after growing.
--/

def initial_height_feet := 5
def initial_height_inches := 6
def additional_height := 6

def convert_height(feet: Int, inches: Int): Int := 
  feet * 12 + inches

def expected_height(initial_feet: Int, initial_inches: Int, additional: Int): Int := 
  convert_height(initial_feet, initial_inches) + additional

theorem steve_height_equiv:
  expected_height initial_height_feet initial_height_inches additional_height = 72 :=
by
  sorry

end steve_height_equiv_l66_66679


namespace hyperbola_midpoint_l66_66491

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66491


namespace find_sum_invested_l66_66738

noncomputable def sum_invested (interest_difference: ℝ) (rate1: ℝ) (rate2: ℝ) (time: ℝ): ℝ := 
  interest_difference * 100 / (time * (rate1 - rate2))

theorem find_sum_invested :
  let interest_difference := 600
  let rate1 := 18 / 100
  let rate2 := 12 / 100
  let time := 2
  sum_invested interest_difference rate1 rate2 time = 5000 :=
by
  sorry

end find_sum_invested_l66_66738


namespace action_figures_added_l66_66316

-- Definitions according to conditions
def initial_action_figures : ℕ := 4
def books_on_shelf : ℕ := 22 -- This information is not necessary for proving the action figures added
def total_action_figures_after_adding : ℕ := 10

-- Theorem to prove given the conditions
theorem action_figures_added : (total_action_figures_after_adding - initial_action_figures) = 6 := by
  sorry

end action_figures_added_l66_66316


namespace sum_of_remainders_l66_66989

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end sum_of_remainders_l66_66989


namespace value_of_f_at_pi_over_12_l66_66127

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - ω * Real.pi)

theorem value_of_f_at_pi_over_12 (ω : ℝ) (hω_pos : ω > 0) 
(h_period : ∀ x, f ω (x + Real.pi) = f ω x) : 
  f ω (Real.pi / 12) = 1 / 2 := 
sorry

end value_of_f_at_pi_over_12_l66_66127


namespace number_of_children_tickets_l66_66856

theorem number_of_children_tickets 
    (x y : ℤ) 
    (h1 : x + y = 225) 
    (h2 : 6 * x + 9 * y = 1875) : 
    x = 50 := 
  sorry

end number_of_children_tickets_l66_66856


namespace quadrilateral_area_correct_l66_66308

noncomputable def area_quadrilateral_ABCD :
  {AB BC CD : ℝ} → {m∠B m∠C : ℝ} →
    (AB = 5) →
    (BC = 6) →
    (CD = 7) →
    (m∠B = 120) →
    (m∠C = 100) → ℝ
| AB BC CD m∠B m∠C, hAB, hBC, hCD, hAngleB, hAngleC => 
  let A_ABC := 0.5 * AB * BC * Real.sin (120 * Real.pi / 180)
  let A_BCD := 0.5 * BC * CD * Real.sin (100 * Real.pi / 180)
  A_ABC + A_BCD

theorem quadrilateral_area_correct :
  {AB BC CD : ℝ} → {m∠B m∠C : ℝ} →
    (AB = 5) →
    (BC = 6) →
    (CD = 7) →
    (m∠B = 120) →
    (m∠C = 100) →
    area_quadrilateral_ABCD AB BC CD m∠B m∠C
    = (15 * Real.sqrt 3) / 2 + 20.69 := 
by
  -- Definitions and conditions
  sorry

end quadrilateral_area_correct_l66_66308


namespace halfway_fraction_l66_66007

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66007


namespace relationship_a_b_c_l66_66049

theorem relationship_a_b_c (x y a b c : ℝ) (h1 : x + y = a)
  (h2 : x^2 + y^2 = b) (h3 : x^3 + y^3 = c) : a^3 - 3*a*b + 2*c = 0 := by
  sorry

end relationship_a_b_c_l66_66049


namespace trains_cross_time_l66_66378

noncomputable def time_to_cross (length_train : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let relative_speed_kmph := speed_train_kmph + speed_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let total_distance := length_train + length_train
  total_distance / relative_speed_mps

theorem trains_cross_time :
  time_to_cross 180 80 = 8.1 := 
by
  sorry

end trains_cross_time_l66_66378


namespace total_inflation_time_l66_66229

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end total_inflation_time_l66_66229


namespace midpoint_on_hyperbola_l66_66526

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66526


namespace monica_tiles_l66_66829

-- Define the dimensions of the living room
def living_room_length : ℕ := 20
def living_room_width : ℕ := 15

-- Define the size of the border tiles and inner tiles
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Prove the number of tiles used is 44
theorem monica_tiles (border_tile_count inner_tile_count total_tiles : ℕ)
  (h_border : border_tile_count = ((2 * ((living_room_length - 4) / border_tile_size) + 2 * ((living_room_width - 4) / border_tile_size) - 4)))
  (h_inner : inner_tile_count = (176 / (inner_tile_size * inner_tile_size)))
  (h_total : total_tiles = border_tile_count + inner_tile_count) :
  total_tiles = 44 :=
by
  sorry

end monica_tiles_l66_66829


namespace factorize_3x2_minus_3y2_l66_66257

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l66_66257


namespace area_of_triangle_ABC_l66_66805

variable {α : Type} [LinearOrder α] [Field α]

-- Given: 
variables (A B C D E F : α) (area_ABC area_BDA area_DCA : α)

-- Conditions:
variable (midpoint_D : 2 * D = B + C)
variable (ratio_AE_EC : 3 * E = A + C)
variable (ratio_AF_FD : 2 * F = A + D)
variable (area_DEF : area_ABC / 6 = 12)

-- To Show:
theorem area_of_triangle_ABC :
  area_ABC = 96 :=
by
  sorry

end area_of_triangle_ABC_l66_66805


namespace part_a_total_time_part_b_average_time_part_c_probability_l66_66640

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l66_66640


namespace can_be_midpoint_of_AB_l66_66520

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66520


namespace total_inflation_time_l66_66230

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end total_inflation_time_l66_66230


namespace part_a_part_b_complete_disorder_l66_66220

open BigOperators

def perm_probability (n m : ℕ) : ℚ :=
  1 / m! * ∑ k in finset.range (n - m + 1), (-1 : ℚ) ^ k / (k + m)!

def at_least_one_probability (n : ℕ) : ℚ :=
  1 - ∑ k in finset.range (n - 1), (-1 : ℚ) ^ (k + 1) / (k + 2)!

def complete_disorder_probability (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (-1 : ℚ) ^ k / k!

theorem part_a (n m : ℕ) (hm : m ≤ n) :
  perm_probability n m = 1 / m! * ∑ k in finset.range (n - m + 1), (-1 : ℚ) ^ k / (k + m)! :=
by
  sorry

theorem part_b (n : ℕ) :
  at_least_one_probability n = 1 - ∑ k in finset.range (n - 1), (-1 : ℚ) ^ (k + 1) / (k + 2)! :=
by
  sorry

theorem complete_disorder (n : ℕ) :
  complete_disorder_probability n = ∑ k in finset.range (n + 1), (-1 : ℚ) ^ k / k! :=
by
  sorry

end part_a_part_b_complete_disorder_l66_66220


namespace sum_of_distinct_integers_l66_66788

theorem sum_of_distinct_integers (a b c d : ℤ) (h : (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25) (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : a + b + c + d = 4 :=
by
    sorry

end sum_of_distinct_integers_l66_66788


namespace part_a_part_b_part_c_l66_66645

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l66_66645


namespace solve_system_l66_66837

theorem solve_system :
  {p : ℝ × ℝ // 
    (p.1 + |p.2| = 3 ∧ 2 * |p.1| - p.2 = 3) ∧
    (p = (2, 1) ∨ p = (0, -3) ∨ p = (-6, 9))} :=
by { sorry }

end solve_system_l66_66837


namespace complementary_event_target_l66_66215

theorem complementary_event_target (S : Type) (hit miss : S) (shoots : ℕ → S) :
  (∀ n : ℕ, (shoots n = hit ∨ shoots n = miss)) →
  (∃ n : ℕ, shoots n = hit) ↔ (∀ n : ℕ, shoots n ≠ hit) :=
by
sorry

end complementary_event_target_l66_66215


namespace part_a_total_time_part_b_average_time_part_c_probability_l66_66639

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l66_66639


namespace initial_bird_families_l66_66977

/- Definitions: -/
def birds_away_africa : ℕ := 23
def birds_away_asia : ℕ := 37
def birds_left_mountain : ℕ := 25

/- Theorem (Question and Correct Answer): -/
theorem initial_bird_families : birds_away_africa + birds_away_asia + birds_left_mountain = 85 := by
  sorry

end initial_bird_families_l66_66977


namespace largest_possible_m_l66_66687

theorem largest_possible_m (x y : ℕ) (h1 : x > y) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < 10) (hyy : y < 10) (h_prime_10xy : Nat.Prime (10 * x + y)) : ∃ m : ℕ, m = x * y * (10 * x + y) ∧ 1000 ≤ m ∧ m ≤ 9999 ∧ ∀ n : ℕ, (n = x * y * (10 * x + y) ∧ 1000 ≤ n ∧ n ≤ 9999) → n ≤ 1533 :=
by
  sorry

end largest_possible_m_l66_66687


namespace cookies_baked_l66_66314

noncomputable def total_cookies (irin ingrid nell : ℚ) (percentage_ingrid : ℚ) : ℚ :=
  let total_ratio := irin + ingrid + nell
  let proportion_ingrid := ingrid / total_ratio
  let total_cookies := ingrid / (percentage_ingrid / 100)
  total_cookies

theorem cookies_baked (h_ratio: 9.18 + 5.17 + 2.05 = 16.4)
                      (h_percentage : 31.524390243902438 = 31.524390243902438) : 
  total_cookies 9.18 5.17 2.05 31.524390243902438 = 52 :=
by
  -- Placeholder for the proof.
  sorry

end cookies_baked_l66_66314


namespace value_of_a_l66_66793

theorem value_of_a (a : ℝ) :
  (∃ (l1 l2 : (ℝ × ℝ × ℝ)),
   l1 = (1, -a, a) ∧ l2 = (3, 1, 2) ∧
   (∃ (m1 m2 : ℝ), 
    (m1 = (1 : ℝ) / a ∧ m2 = -3) ∧ 
    (m1 * m2 = -1))) → a = 3 :=
by sorry

end value_of_a_l66_66793


namespace students_circle_no_regular_exists_zero_regular_school_students_l66_66766

noncomputable def students_circle_no_regular (n : ℕ) 
    (student : ℕ → String)
    (neighbor_right : ℕ → ℕ)
    (lies_to : ℕ → ℕ → Bool) : Prop :=
  ∀ i, student i = "Gymnasium student" →
    (if lies_to i (neighbor_right i)
     then (student (neighbor_right i) ≠ "Gymnasium student")
     else student (neighbor_right i) = "Gymnasium student") →
    (if lies_to (neighbor_right i) i
     then (student i ≠ "Gymnasium student")
     else student i = "Gymnasium student")

theorem students_circle_no_regular_exists_zero_regular_school_students
  (n : ℕ) 
  (student : ℕ → String)
  (neighbor_right : ℕ → ℕ)
  (lies_to : ℕ → ℕ → Bool)
  (h : students_circle_no_regular n student neighbor_right lies_to)
  : (∀ i, student i ≠ "Regular school student") :=
sorry

end students_circle_no_regular_exists_zero_regular_school_students_l66_66766


namespace hyperbola_midpoint_exists_l66_66556

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66556


namespace tshirts_equation_l66_66880

theorem tshirts_equation (x : ℝ) 
    (hx : x > 0)
    (march_cost : ℝ := 120000)
    (april_cost : ℝ := 187500)
    (april_increase : ℝ := 1.4)
    (cost_increase : ℝ := 5) :
    120000 / x + 5 = 187500 / (1.4 * x) :=
by 
  sorry

end tshirts_equation_l66_66880


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66034

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66034


namespace seats_taken_l66_66269

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end seats_taken_l66_66269


namespace marthas_bedroom_size_l66_66356

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l66_66356


namespace halfway_fraction_l66_66012

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66012


namespace fraction_halfway_between_l66_66045

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66045


namespace common_ratio_of_infinite_geometric_series_l66_66400

noncomputable def first_term : ℝ := 500
noncomputable def series_sum : ℝ := 3125

theorem common_ratio_of_infinite_geometric_series (r : ℝ) (h₀ : first_term / (1 - r) = series_sum) : 
  r = 0.84 := 
by
  sorry

end common_ratio_of_infinite_geometric_series_l66_66400


namespace midpoint_on_hyperbola_l66_66525

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66525


namespace range_of_a_l66_66119

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → log_a a (2 - a * x) < log_a a (2 - a * (x / 2))) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l66_66119


namespace halfway_fraction_l66_66013

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66013


namespace midpoint_on_hyperbola_l66_66598

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66598


namespace arithmetic_sequence_sum_l66_66350

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l66_66350


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66629

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66629


namespace Suresh_completes_job_in_15_hours_l66_66182

theorem Suresh_completes_job_in_15_hours :
  ∃ S : ℝ,
    (∀ (T_A Ashutosh_time Suresh_time : ℝ), Ashutosh_time = 15 ∧ Suresh_time = 9 
    → T_A = Ashutosh_time → 6 / T_A + Suresh_time / S = 1) ∧ S = 15 :=
by
  sorry

end Suresh_completes_job_in_15_hours_l66_66182


namespace max_value_expression_l66_66161

theorem max_value_expression (x k : ℕ) (h₀ : 0 < x) (h₁ : 0 < k) (y := k * x) : 
  (∀ x k : ℕ, 0 < x → 0 < k → y = k * x → ∃ m : ℝ, m = 2 ∧ 
    ∀ x k : ℕ, 0 < x → 0 < k → y = k * x → (x + y)^2 / (x^2 + y^2) ≤ 2) :=
sorry

end max_value_expression_l66_66161


namespace nina_widgets_purchase_l66_66830

theorem nina_widgets_purchase (P : ℝ) (h1 : 8 * (P - 1) = 24) (h2 : 24 / P = 6) : true :=
by
  sorry

end nina_widgets_purchase_l66_66830


namespace permutations_count_l66_66263

open Finset

-- Define the set of permutations
def perms := univ.permutations (erase_univ 6)

-- Define the condition predicate
def condition (b : Fin 6 → Fin 6) : Prop :=
  ((b 0 + 1) / 3) * ((b 1 + 2) / 3) * ((b 2 + 3) / 3) * ((b 3 + 4) / 3) * ((b 4 + 5) / 3) * ((b 5 + 6) / 3) > fact 6

-- Define the final problem statement
theorem permutations_count :
  (univ.permutations (erase_univ 6)).filter (λ b, condition b) = 719 :=
by
  sorry

end permutations_count_l66_66263


namespace mary_needs_6_cups_l66_66162
-- We import the whole Mathlib library first.

-- We define the conditions and the question.
def total_cups : ℕ := 8
def cups_added : ℕ := 2
def cups_needed : ℕ := total_cups - cups_added

-- We state the theorem we need to prove.
theorem mary_needs_6_cups : cups_needed = 6 :=
by
  -- We use a placeholder for the proof.
  sorry

end mary_needs_6_cups_l66_66162


namespace nth_equation_pattern_l66_66958

theorem nth_equation_pattern (n : ℕ) : 
  (List.range' n (2 * n - 1)).sum = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_equation_pattern_l66_66958


namespace cos_product_equals_one_eighth_l66_66243

noncomputable def cos_pi_over_9 := Real.cos (Real.pi / 9)
noncomputable def cos_2pi_over_9 := Real.cos (2 * Real.pi / 9)
noncomputable def cos_4pi_over_9 := Real.cos (4 * Real.pi / 9)

theorem cos_product_equals_one_eighth :
  cos_pi_over_9 * cos_2pi_over_9 * cos_4pi_over_9 = 1 / 8 := 
sorry

end cos_product_equals_one_eighth_l66_66243


namespace halfway_fraction_l66_66026

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66026


namespace shara_shells_final_count_l66_66835

def initial_shells : ℕ := 20
def first_vacation_found : ℕ := 5 * 3 + 6
def first_vacation_lost : ℕ := 4
def second_vacation_found : ℕ := 4 * 2 + 7
def second_vacation_gifted : ℕ := 3
def third_vacation_found : ℕ := 8 + 4 + 3 * 2
def third_vacation_misplaced : ℕ := 5

def total_shells_after_first_vacation : ℕ :=
  initial_shells + first_vacation_found - first_vacation_lost

def total_shells_after_second_vacation : ℕ :=
  total_shells_after_first_vacation + second_vacation_found - second_vacation_gifted

def total_shells_after_third_vacation : ℕ :=
  total_shells_after_second_vacation + third_vacation_found - third_vacation_misplaced

theorem shara_shells_final_count : total_shells_after_third_vacation = 62 := by
  sorry

end shara_shells_final_count_l66_66835


namespace part_a_part_b_part_c_l66_66647

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l66_66647


namespace percentage_difference_l66_66985

theorem percentage_difference (x : ℝ) : 
  (62 / 100) * 150 - (x / 100) * 250 = 43 → x = 20 :=
by
  intro h
  sorry

end percentage_difference_l66_66985


namespace wall_length_l66_66758

theorem wall_length (mirror_side length width : ℝ) (h_mirror : mirror_side = 21) (h_width : width = 28) 
  (h_area_relation : (mirror_side * mirror_side) * 2 = width * length) : length = 31.5 :=
by
  -- here you start the proof, but it's not required for the statement
  sorry

end wall_length_l66_66758


namespace factorize_polynomial_l66_66255

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l66_66255


namespace sum_of_remainders_l66_66986

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end sum_of_remainders_l66_66986


namespace larger_integer_value_l66_66689

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l66_66689


namespace fewest_keystrokes_to_256_l66_66214

def fewest_keystrokes (start target : Nat) : Nat :=
if start = 1 && target = 256 then 8 else sorry

theorem fewest_keystrokes_to_256 : fewest_keystrokes 1 256 = 8 :=
by
  sorry

end fewest_keystrokes_to_256_l66_66214


namespace number_divided_by_3_equals_subtract_3_l66_66713

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l66_66713


namespace units_place_3_pow_34_l66_66374

theorem units_place_3_pow_34 : (3^34 % 10) = 9 :=
by
  sorry

end units_place_3_pow_34_l66_66374


namespace num_five_ruble_coins_l66_66653

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l66_66653


namespace hyperbola_midpoint_l66_66582

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66582


namespace thread_length_l66_66669

theorem thread_length (x : ℝ) (h : x + (3/4) * x = 21) : x = 12 :=
  sorry

end thread_length_l66_66669


namespace original_amount_l66_66262

theorem original_amount (x : ℝ) (h : 0.25 * x = 200) : x = 800 := 
by
  sorry

end original_amount_l66_66262


namespace find_m_same_foci_l66_66926

theorem find_m_same_foci (m : ℝ) 
(hyperbola_eq : ∃ x y : ℝ, x^2 - y^2 = m) 
(ellipse_eq : ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = m + 1) 
(same_foci : ∀ a b : ℝ, (x^2 - y^2 = m) ∧ (2 * x^2 + 3 * y^2 = m + 1) → 
               let c_ellipse := (m + 1) / 6
               let c_hyperbola := 2 * m
               c_ellipse = c_hyperbola ) : 
m = 1 / 11 := 
sorry

end find_m_same_foci_l66_66926


namespace fujian_provincial_games_distribution_count_l66_66965

theorem fujian_provincial_games_distribution_count 
  (staff_members : Finset String)
  (locations : Finset String)
  (A B C D E F : String)
  (A_in_B : A ∈ staff_members)
  (B_in_B : B ∈ staff_members)
  (C_in_B : C ∈ staff_members)
  (D_in_B : D ∈ staff_members)
  (E_in_B : E ∈ staff_members)
  (F_in_B : F ∈ staff_members)
  (locations_count : locations.card = 2)
  (staff_count : staff_members.card = 6)
  (must_same_group : ∀ g₁ g₂ : Finset String, A ∈ g₁ → B ∈ g₁ → g₁ ∪ g₂ = staff_members)
  (min_two_people : ∀ g : Finset String, 2 ≤ g.card) :
  ∃ distrib_methods : ℕ, distrib_methods = 22 := 
by
  sorry

end fujian_provincial_games_distribution_count_l66_66965


namespace sum_of_first_10_terms_l66_66193

def general_term (n : ℕ) : ℕ := 2 * n + 1

def sequence_sum (n : ℕ) : ℕ := n / 2 * (general_term 1 + general_term n)

theorem sum_of_first_10_terms : sequence_sum 10 = 120 := by
  sorry

end sum_of_first_10_terms_l66_66193


namespace age_of_replaced_man_l66_66966

-- Definitions based on conditions
def avg_age_men (A : ℝ) := A
def age_man1 := 10
def avg_age_women := 23
def total_age_women := 2 * avg_age_women
def new_avg_age_men (A : ℝ) := A + 2

-- Proposition stating that given conditions yield the age of the other replaced man
theorem age_of_replaced_man (A M : ℝ) :
  8 * avg_age_men A - age_man1 - M + total_age_women = 8 * new_avg_age_men A + 16 →
  M = 20 :=
by
  sorry

end age_of_replaced_man_l66_66966


namespace most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l66_66878

noncomputable def most_entries_with_80_yuan : Nat :=
let cost_a := 120
let cost_b := 60
let cost_c := 40
let entry_b := 2
let entry_c := 3
let budget := 80
let entries_b := (budget - cost_b) / entry_b
let entries_c := (budget - cost_c) / entry_c
let entries_no_pass := budget / 10
if cost_a <= budget then 
  0
else
  max entries_b (max entries_c entries_no_pass)

theorem most_entries_with_80_yuan_is_c_pass : most_entries_with_80_yuan = 13 :=
by
  sorry

noncomputable def is_pass_a_cost_effective (x : Nat) : Prop :=
let cost_a := 120
let cost_b_entries := 60 + 2 * x
let cost_c_entries := 40 + 3 * x
let cost_no_pass := 10 * x
x > 30 → cost_a < cost_b_entries ∧ cost_a < cost_c_entries ∧ cost_a < cost_no_pass

theorem pass_a_is_cost_effective_after_30_entries : ∀ x : Nat, is_pass_a_cost_effective x :=
by
  sorry

end most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l66_66878


namespace find_two_numbers_l66_66203

open Nat

theorem find_two_numbers : ∃ (x y : ℕ), 
  x + y = 667 ∧ 
  (lcm x y) / (gcd x y) = 120 ∧ 
  ((x = 552 ∧ y = 115) ∨ (x = 115 ∧ y = 552) ∨ (x = 435 ∧ y = 232) ∨ (x = 232 ∧ y = 435)) :=
by
  sorry

end find_two_numbers_l66_66203


namespace first_place_points_is_eleven_l66_66455

/-
Conditions:
1. Points are awarded as follows: first place = x points, second place = 7 points, third place = 5 points, fourth place = 2 points.
2. John participated 7 times in the competition.
3. John finished in each of the top four positions at least once.
4. The product of all the points John received was 38500.
Theorem: The first place winner receives 11 points.
-/

noncomputable def archery_first_place_points (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), -- number of times John finished first, second, third, fourth respectively
    a + b + c + d = 7 ∧ -- condition 2, John participated 7 times
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ -- condition 3, John finished each position at least once
    x ^ a * 7 ^ b * 5 ^ c * 2 ^ d = 38500 -- condition 4, product of all points John received

theorem first_place_points_is_eleven : archery_first_place_points 11 :=
  sorry

end first_place_points_is_eleven_l66_66455


namespace circle_radius_eq_two_l66_66776

theorem circle_radius_eq_two (x y : ℝ) : (x^2 + y^2 + 1 = 2 * x + 4 * y) → (∃ c : ℝ × ℝ, ∃ r : ℝ, ((x - c.1)^2 + (y - c.2)^2 = r^2) ∧ r = 2) := by
  sorry

end circle_radius_eq_two_l66_66776


namespace num_customers_did_not_tip_l66_66403

def total_customers : Nat := 9
def total_earnings : Nat := 32
def tip_per_customer : Nat := 8
def customers_who_tipped := total_earnings / tip_per_customer
def customers_who_did_not_tip := total_customers - customers_who_tipped

theorem num_customers_did_not_tip : customers_who_did_not_tip = 5 := 
by
  -- We use the definitions provided.
  have eq1 : customers_who_tipped = 4 := by
    sorry
  have eq2 : customers_who_did_not_tip = total_customers - customers_who_tipped := by
    sorry
  have eq3 : customers_who_did_not_tip = 9 - 4 := by
    sorry
  exact eq3

end num_customers_did_not_tip_l66_66403


namespace midpoint_hyperbola_l66_66550

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66550


namespace midpoint_on_hyperbola_l66_66521

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66521


namespace midpoint_hyperbola_l66_66536

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66536


namespace midpoint_on_hyperbola_l66_66592

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66592


namespace midpoint_on_hyperbola_l66_66562

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66562


namespace halfway_fraction_l66_66014

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66014


namespace inequality_one_over_a_plus_one_over_b_geq_4_l66_66283

theorem inequality_one_over_a_plus_one_over_b_geq_4 
    (a b : ℕ) (hapos : 0 < a) (hbpos : 0 < b) (h : a + b = 1) : 
    (1 : ℚ) / a + (1 : ℚ) / b ≥ 4 := 
  sorry

end inequality_one_over_a_plus_one_over_b_geq_4_l66_66283


namespace range_of_a_if_p_is_false_l66_66123

theorem range_of_a_if_p_is_false (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_if_p_is_false_l66_66123


namespace midpoint_of_line_segment_on_hyperbola_l66_66618

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66618


namespace hexagon_cookie_cutters_count_l66_66101

-- Definitions for the conditions
def triangle_side_count := 3
def triangles := 6
def square_side_count := 4
def squares := 4
def total_sides := 46

-- Given conditions translated to Lean 4
def sides_from_triangles := triangles * triangle_side_count
def sides_from_squares := squares * square_side_count
def sides_from_triangles_and_squares := sides_from_triangles + sides_from_squares
def sides_from_hexagons := total_sides - sides_from_triangles_and_squares
def hexagon_side_count := 6

-- Statement to prove that there are 2 hexagon-shaped cookie cutters
theorem hexagon_cookie_cutters_count : sides_from_hexagons / hexagon_side_count = 2 := by
  sorry

end hexagon_cookie_cutters_count_l66_66101


namespace find_n_l66_66739

theorem find_n :
  let a := (6 + 12 + 18 + 24 + 30 + 36 + 42) / 7
  let b := (2 * n : ℕ)
  (a*a - b*b = 0) -> (n = 12) := 
by 
  let a := 24
  let b := 2*n
  sorry

end find_n_l66_66739


namespace evelyn_total_marbles_l66_66780

def initial_marbles := 95
def marbles_from_henry := 9
def marbles_from_grace := 12
def number_of_cards := 6
def marbles_per_card := 4

theorem evelyn_total_marbles :
  initial_marbles + marbles_from_henry + marbles_from_grace + number_of_cards * marbles_per_card = 140 := 
by 
  sorry

end evelyn_total_marbles_l66_66780


namespace find_x_value_l66_66913

open Real

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
(h3 : tan (150 * π / 180 - x * π / 180) = (sin (150 * π / 180) - sin (x * π / 180)) / (cos (150 * π / 180) - cos (x * π / 180))) :
x = 120 :=
sorry

end find_x_value_l66_66913


namespace sum_even_pos_integers_lt_100_l66_66054

theorem sum_even_pos_integers_lt_100 : 
  (Finset.sum (Finset.filter (λ n, n % 2 = 0 ∧ n < 100) (Finset.range 100))) = 2450 :=
by
  sorry

end sum_even_pos_integers_lt_100_l66_66054


namespace can_be_midpoint_of_AB_l66_66513

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66513


namespace midpoint_hyperbola_l66_66531

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66531


namespace find_value_l66_66118

def equation := ∃ x : ℝ, x^2 - 2 * x - 3 = 0
def expression (x : ℝ) := 2 * x^2 - 4 * x + 12

theorem find_value :
  (∃ x : ℝ, (x^2 - 2 * x - 3 = 0) ∧ (expression x = 18)) :=
by
  sorry

end find_value_l66_66118


namespace probability_two_tails_after_two_heads_l66_66247

noncomputable def fair_coin_probability : ℚ :=
  -- Given conditions:
  let p_head := (1 : ℚ) / 2
  let p_tail := (1 : ℚ) / 2

  -- Define the probability Q as stated in the problem
  let Q := ((1 : ℚ) / 4) / (1 - (1 : ℚ) / 4)

  -- Calculate the probability of starting with sequence "HTH"
  let p_HTH := p_head * p_tail * p_head

  -- Calculate the final probability
  p_HTH * Q

theorem probability_two_tails_after_two_heads :
  fair_coin_probability = (1 : ℚ) / 24 :=
by
  sorry

end probability_two_tails_after_two_heads_l66_66247


namespace binary_to_base5_l66_66246

theorem binary_to_base5 : Nat.digits 5 (Nat.ofDigits 2 [1, 0, 1, 1, 0, 0, 1]) = [4, 2, 3] :=
by
  sorry

end binary_to_base5_l66_66246


namespace value_of_y_l66_66293

-- Problem: Prove that given the conditions \( x - y = 8 \) and \( x + y = 16 \),
-- the value of \( y \) is 4.
theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := 
sorry

end value_of_y_l66_66293


namespace daily_wage_of_c_is_71_l66_66070

theorem daily_wage_of_c_is_71 (x : ℚ) :
  let a_days := 16
  let b_days := 9
  let c_days := 4
  let total_earnings := 1480
  let wage_ratio_a := 3
  let wage_ratio_b := 4
  let wage_ratio_c := 5
  let total_contribution := a_days * wage_ratio_a * x + b_days * wage_ratio_b * x + c_days * wage_ratio_c * x
  total_contribution = total_earnings →
  c_days * wage_ratio_c * x = 71 := by
  sorry

end daily_wage_of_c_is_71_l66_66070


namespace second_discarded_number_l66_66843

theorem second_discarded_number (S : ℝ) (X : ℝ) (h1 : S / 50 = 62) (h2 : (S - 45 - X) / 48 = 62.5) : X = 55 := 
by
  sorry

end second_discarded_number_l66_66843


namespace petya_five_ruble_coins_count_l66_66662

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l66_66662


namespace fraction_order_l66_66249

def frac_21_16 := 21 / 16
def frac_25_19 := 25 / 19
def frac_23_17 := 23 / 17
def frac_27_20 := 27 / 20

theorem fraction_order : frac_21_16 < frac_25_19 ∧ frac_25_19 < frac_27_20 ∧ frac_27_20 < frac_23_17 := by sorry

end fraction_order_l66_66249


namespace range_of_k_for_real_roots_l66_66451

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 = x2 ∧ x^2 - 2*x + k = 0) ↔ k ≤ 1 := 
by
  sorry

end range_of_k_for_real_roots_l66_66451


namespace empty_set_condition_l66_66991

def isEmptySet (s : Set ℝ) : Prop := s = ∅

def A : Set ℕ := {n : ℕ | n^2 ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def C : Set ℝ := {x : ℝ | x^2 + x + 1 = 0}
def D : Set ℝ := {0}

theorem empty_set_condition : isEmptySet C := by
  sorry

end empty_set_condition_l66_66991


namespace sequence_term_a1000_l66_66809

theorem sequence_term_a1000 :
  ∃ (a : ℕ → ℕ), a 1 = 1007 ∧ a 2 = 1008 ∧
  (∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n) ∧
  a 1000 = 1673 :=
by
  sorry

end sequence_term_a1000_l66_66809


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66644

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66644


namespace problem_I_problem_II_l66_66798

/-- Proof problem I: Given f(x) = |x - 1|, prove that the inequality f(x) ≥ 4 - |x - 1| implies x ≥ 3 or x ≤ -1 -/
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (h2 : f x ≥ 4 - |x - 1|) : x ≥ 3 ∨ x ≤ -1 :=
  sorry

/-- Proof problem II: Given f(x) = |x - 1| and 1/m + 1/(2*n) = 1 (m > 0, n > 0), prove that the minimum value of mn is 2 -/
theorem problem_II (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h2 : 1/m + 1/(2*n) = 1) : m*n ≥ 2 :=
  sorry

end problem_I_problem_II_l66_66798


namespace f_seven_point_five_l66_66156

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom f_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_seven_point_five : f 7.5 = -0.5 := by
  sorry

end f_seven_point_five_l66_66156


namespace hyperbola_midpoint_l66_66609

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66609


namespace simplify_f_l66_66789

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) (h : Real.sin (α - 3 / 2 * Real.pi) = 1 / 5) : f α = -1 / 5 := by
  sorry

end simplify_f_l66_66789


namespace midpoint_of_line_segment_on_hyperbola_l66_66616

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66616


namespace fraction_of_smart_integers_divisible_by_5_is_one_third_l66_66248

-- Condition definitions
def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_of_digits (n : ℕ) : ℕ := (Nat.digits 10 n).sum

def is_smart_integer (n : ℕ) : Prop :=
  is_even n ∧ 30 < n ∧ n < 150 ∧ sum_of_digits n = 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Function to filter smart integers
def smart_integers := List.filter is_smart_integer (List.range 150).filter (fun n => n > 30)

-- Count occurrences
def count_pred (l : List ℕ) (p : ℕ -> Prop) : ℕ :=
  l.filter p |>.length

noncomputable def fraction_smart_integers_divisible_by_5 : ℚ :=
  (count_pred smart_integers is_divisible_by_5 : ℚ) / (smart_integers.length : ℚ)

-- The theorem to prove
theorem fraction_of_smart_integers_divisible_by_5_is_one_third :
  fraction_smart_integers_divisible_by_5 = 1 / 3 := by
  sorry

end fraction_of_smart_integers_divisible_by_5_is_one_third_l66_66248


namespace total_books_l66_66173

variable (Sandy_books Benny_books Tim_books : ℕ)
variable (h_Sandy : Sandy_books = 10)
variable (h_Benny : Benny_books = 24)
variable (h_Tim : Tim_books = 33)

theorem total_books :
  Sandy_books + Benny_books + Tim_books = 67 :=
by sorry

end total_books_l66_66173


namespace parenthesis_removal_correctness_l66_66905

theorem parenthesis_removal_correctness (x y z : ℝ) : 
  (x^2 - (x - y + 2 * z) ≠ x^2 - x + y - 2 * z) ∧
  (x - (-2 * x + 3 * y - 1) ≠ x + 2 * x - 3 * y + 1) ∧
  (3 * x + 2 * (x - 2 * y + 1) ≠ 3 * x + 2 * x - 4 * y + 2) ∧
  (-(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2 * x^2 - 4) :=
by
  sorry

end parenthesis_removal_correctness_l66_66905


namespace probability_palindrome_divisible_by_11_is_zero_l66_66233

-- Define the three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Define the divisibility condition
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Prove that the probability is zero
theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ n, is_palindrome n ∧ is_divisible_by_11 n) →
  (0 : ℕ) = 0 := by
  sorry

end probability_palindrome_divisible_by_11_is_zero_l66_66233


namespace average_speed_trip_l66_66387

-- Conditions: Definitions
def distance_north_feet : ℝ := 5280
def speed_north_mpm : ℝ := 2
def speed_south_mpm : ℝ := 1

-- Question and Equivalent Proof Problem
theorem average_speed_trip :
  let distance_north_miles := distance_north_feet / 5280
  let distance_south_miles := 2 * distance_north_miles
  let total_distance_miles := distance_north_miles + distance_south_miles + distance_south_miles
  let time_north_hours := distance_north_miles / speed_north_mpm / 60
  let time_south_hours := distance_south_miles / speed_south_mpm / 60
  let time_return_hours := distance_south_miles / speed_south_mpm / 60
  let total_time_hours := time_north_hours + time_south_hours + time_return_hours
  let average_speed_mph := total_distance_miles / total_time_hours
  average_speed_mph = 76.4 := by
    sorry

end average_speed_trip_l66_66387


namespace face_value_of_ticket_l66_66395
noncomputable def face_value_each_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) : ℝ :=
  total_price / (group_size * (1 + tax_rate))

theorem face_value_of_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) :
  total_price = 945 →
  group_size = 25 →
  tax_rate = 0.05 →
  face_value_each_ticket total_price group_size tax_rate = 36 := 
by
  intros h_total_price h_group_size h_tax_rate
  rw [h_total_price, h_group_size, h_tax_rate]
  simp [face_value_each_ticket]
  sorry

end face_value_of_ticket_l66_66395


namespace problem_statement_l66_66408

noncomputable def sqrt4 := real.sqrt 4
noncomputable def tan60 := real.tan (real.pi / 3)
noncomputable def pow2023_0 := (2023 : ℝ) ^ 0

theorem problem_statement : sqrt4 + abs (tan60 - 1) - pow2023_0 = real.sqrt 3 :=
by
  sorry

end problem_statement_l66_66408


namespace hyperbola_represents_l66_66185

theorem hyperbola_represents (k : ℝ) : 
  (k - 2) * (5 - k) < 0 ↔ (k < 2 ∨ k > 5) :=
by
  sorry

end hyperbola_represents_l66_66185


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66632

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66632


namespace grown_ups_in_milburg_l66_66367

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l66_66367


namespace nine_otimes_three_l66_66911

def otimes (a b : ℤ) : ℤ := a + (4 * a) / (3 * b)

theorem nine_otimes_three : otimes 9 3 = 13 := by
  sorry

end nine_otimes_three_l66_66911


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66631

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66631


namespace simplify_expression_l66_66442

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x = 1) :
  (1 - x) ^ 2 - (x + 3) * (3 - x) - (x - 3) * (x - 1) = -10 :=
by 
  sorry

end simplify_expression_l66_66442


namespace distance_between_points_eq_l66_66108

theorem distance_between_points_eq :
  let x1 := 2
  let y1 := -5
  let x2 := -8
  let y2 := 7
  let distance := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  distance = 2 * Real.sqrt 61 :=
by
  sorry

end distance_between_points_eq_l66_66108


namespace number_of_valid_lines_l66_66309

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def lines_passing_through_point (x_int : ℕ) (y_int : ℕ) (p : ℕ × ℕ) : Prop :=
  p.1 * y_int + p.2 * x_int = x_int * y_int

theorem number_of_valid_lines (p : ℕ × ℕ) : 
  ∃! l : ℕ × ℕ, is_prime (l.1) ∧ is_power_of_two (l.2) ∧ lines_passing_through_point l.1 l.2 p :=
sorry

end number_of_valid_lines_l66_66309


namespace rectangle_area_k_l66_66198

theorem rectangle_area_k (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2)
  (h_diag : (length ^ 2 + width ^ 2) = d ^ 2) :
  ∃ (k : ℝ), k = 10 / 29 ∧ length * width = k * d ^ 2 := by
  sorry

end rectangle_area_k_l66_66198


namespace cats_left_correct_l66_66753

-- Define initial conditions
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def sold_cats : ℕ := 10

-- Define the total number of cats initially
def total_cats_initial : ℕ := siamese_cats + house_cats

-- Define the number of cats left after the sale
def cats_left : ℕ := total_cats_initial - sold_cats

-- Prove the number of cats left is 8
theorem cats_left_correct : cats_left = 8 :=
by 
  sorry

end cats_left_correct_l66_66753


namespace solve_quadratic_complete_square_l66_66462

theorem solve_quadratic_complete_square :
  ∃ b c : ℤ, (∀ x : ℝ, (x + b)^2 = c ↔ x^2 + 6 * x - 9 = 0) ∧ b + c = 21 := by
  sorry

end solve_quadratic_complete_square_l66_66462


namespace parrots_per_cage_l66_66888

theorem parrots_per_cage (P : ℕ) (total_birds total_cages parakeets_per_cage : ℕ)
  (h1 : total_cages = 4)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 40)
  (h4 : total_birds = total_cages * (P + parakeets_per_cage)) :
  P = 8 :=
by
  sorry

end parrots_per_cage_l66_66888


namespace number_of_oxygen_atoms_l66_66746

/-- Given a compound has 1 H, 1 Cl, and a certain number of O atoms and the molecular weight of the compound is 68 g/mol,
    prove that the number of O atoms is 2. -/
theorem number_of_oxygen_atoms (atomic_weight_H: ℝ) (atomic_weight_Cl: ℝ) (atomic_weight_O: ℝ) (molecular_weight: ℝ) (n : ℕ):
    atomic_weight_H = 1.0 →
    atomic_weight_Cl = 35.5 →
    atomic_weight_O = 16.0 →
    molecular_weight = 68.0 →
    molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O →
    n = 2 :=
by
  sorry

end number_of_oxygen_atoms_l66_66746


namespace midpoint_of_hyperbola_segment_l66_66509

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66509


namespace fraction_oj_is_5_over_13_l66_66859

def capacity_first_pitcher : ℕ := 800
def capacity_second_pitcher : ℕ := 500
def fraction_oj_first_pitcher : ℚ := 1 / 4
def fraction_oj_second_pitcher : ℚ := 3 / 5

def amount_oj_first_pitcher : ℚ := capacity_first_pitcher * fraction_oj_first_pitcher
def amount_oj_second_pitcher : ℚ := capacity_second_pitcher * fraction_oj_second_pitcher

def total_amount_oj : ℚ := amount_oj_first_pitcher + amount_oj_second_pitcher
def total_capacity : ℚ := capacity_first_pitcher + capacity_second_pitcher

def fraction_oj_large_container : ℚ := total_amount_oj / total_capacity

theorem fraction_oj_is_5_over_13 : fraction_oj_large_container = (5 / 13) := by
  -- Proof would go here
  sorry

end fraction_oj_is_5_over_13_l66_66859


namespace onion_rings_cost_l66_66956

variable (hamburger_cost smoothie_cost total_payment change_received : ℕ)

theorem onion_rings_cost (h_hamburger : hamburger_cost = 4) 
                         (h_smoothie : smoothie_cost = 3) 
                         (h_total_payment : total_payment = 20) 
                         (h_change_received : change_received = 11) :
                         total_payment - change_received - hamburger_cost - smoothie_cost = 2 :=
by
  sorry

end onion_rings_cost_l66_66956


namespace log2_x_value_l66_66792

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log2_x_value
  (x : ℝ)
  (h : log_base (5 * x) (2 * x) = log_base (625 * x) (8 * x)) :
  log_base 2 x = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5) :=
by
  sorry

end log2_x_value_l66_66792


namespace largest_possible_n_l66_66240

open Nat

-- Define arithmetic sequences a_n and b_n with given initial conditions
def arithmetic_seq (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  a_n 1 = 1 ∧ b_n 1 = 1 ∧ 
  a_n 2 ≤ b_n 2 ∧
  (∃n : ℕ, a_n n * b_n n = 1764)

-- Given the arithmetic sequences defined above, prove that the largest possible value of n is 44
theorem largest_possible_n : 
  ∀ (a_n b_n : ℕ → ℕ), arithmetic_seq a_n b_n →
  ∀ (n : ℕ), (a_n n * b_n n = 1764) → n ≤ 44 :=
sorry

end largest_possible_n_l66_66240


namespace sum_of_remainders_l66_66988

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end sum_of_remainders_l66_66988


namespace sin_60_eq_sqrt3_div_2_l66_66266

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l66_66266


namespace find_x_l66_66729

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l66_66729


namespace total_inflation_time_l66_66231

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end total_inflation_time_l66_66231


namespace find_n_l66_66929

variable {a : ℕ → ℝ}  -- Defining the sequence

-- Defining the conditions:
def a1 : Prop := a 1 = 1 / 3
def a2_plus_a5 : Prop := a 2 + a 5 = 4
def a_n_eq_33 (n : ℕ) : Prop := a n = 33

theorem find_n (n : ℕ) : a 1 = 1 / 3 → (a 2 + a 5 = 4) → (a n = 33) → n = 50 := 
by 
  intros h1 h2 h3 
  -- the complete proof can be done here
  sorry

end find_n_l66_66929


namespace average_age_of_persons_l66_66967

theorem average_age_of_persons 
  (total_age : ℕ := 270) 
  (average_age : ℕ := 15) : 
  (total_age / average_age) = 18 := 
by { 
  sorry 
}

end average_age_of_persons_l66_66967


namespace max_b_value_l66_66698

theorem max_b_value
  (a b c : ℕ)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = 240) : b = 10 :=
  sorry

end max_b_value_l66_66698


namespace fraction_halfway_between_l66_66043

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66043


namespace inning_count_l66_66382

-- Definition of the conditions
variables {n T H L : ℕ}
variables (avg_total : ℕ) (avg_excl : ℕ) (diff : ℕ) (high_score : ℕ)

-- Define the conditions
def conditions :=
  avg_total = 62 ∧
  high_score = 225 ∧
  diff = 150 ∧
  avg_excl = 58

-- Proving the main theorem
theorem inning_count (avg_total := 62) (high_score := 225) (diff := 150) (avg_excl := 58) :
   conditions avg_total avg_excl diff high_score →
   n = 104 :=
sorry

end inning_count_l66_66382


namespace marthas_bedroom_size_l66_66354

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l66_66354


namespace man_son_ratio_in_two_years_l66_66884

noncomputable def man_and_son_age_ratio (M S : ℕ) (h1 : M = S + 25) (h2 : S = 23) : ℕ × ℕ :=
  let S_in_2_years := S + 2
  let M_in_2_years := M + 2
  (M_in_2_years / S_in_2_years, S_in_2_years / S_in_2_years)

theorem man_son_ratio_in_two_years : man_and_son_age_ratio 48 23 (by norm_num) (by norm_num) = (2, 1) :=
  sorry

end man_son_ratio_in_two_years_l66_66884


namespace households_with_dvd_player_l66_66942

noncomputable def numHouseholds : ℕ := 100
noncomputable def numWithCellPhone : ℕ := 90
noncomputable def numWithMP3Player : ℕ := 55
noncomputable def greatestWithAllThree : ℕ := 55 -- maximum x
noncomputable def differenceX_Y : ℕ := 25 -- x - y = 25

def numberOfDVDHouseholds : ℕ := 15

theorem households_with_dvd_player : ∀ (D : ℕ),
  D + 25 - D = 55 - 20 →
  D = numberOfDVDHouseholds :=
by
  intro D h
  sorry

end households_with_dvd_player_l66_66942


namespace cosine_of_angle_in_convex_quadrilateral_l66_66148

theorem cosine_of_angle_in_convex_quadrilateral
    (A C : ℝ)
    (AB CD AD BC : ℝ)
    (h1 : A = C)
    (h2 : AB = 150)
    (h3 : CD = 150)
    (h4 : AD = BC)
    (h5 : AB + BC + CD + AD = 580) :
    Real.cos A = 7 / 15 := 
  sorry

end cosine_of_angle_in_convex_quadrilateral_l66_66148


namespace arithmetic_mean_calculation_l66_66908

theorem arithmetic_mean_calculation (x : ℝ) 
  (h : (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30) : 
  x = 14.142857 :=
by
  sorry

end arithmetic_mean_calculation_l66_66908


namespace rotten_pineapples_l66_66855

theorem rotten_pineapples (initial sold fresh remaining rotten: ℕ) 
  (h1: initial = 86) 
  (h2: sold = 48) 
  (h3: fresh = 29) 
  (h4: remaining = initial - sold) 
  (h5: rotten = remaining - fresh) : 
  rotten = 9 := by 
  sorry

end rotten_pineapples_l66_66855


namespace molecular_weight_CaOH2_correct_l66_66244

/-- Molecular weight of Calcium hydroxide -/
def molecular_weight_CaOH2 (Ca O H : ℝ) : ℝ :=
  Ca + 2 * (O + H)

theorem molecular_weight_CaOH2_correct :
  molecular_weight_CaOH2 40.08 16.00 1.01 = 74.10 :=
by 
  -- This statement requires a proof that would likely involve arithmetic on real numbers
  sorry

end molecular_weight_CaOH2_correct_l66_66244


namespace max_value_of_x_l66_66470

theorem max_value_of_x (x y : ℝ) (h : x^2 + y^2 = 18 * x + 20 * y) : x ≤ 9 + Real.sqrt 181 :=
by
  sorry

end max_value_of_x_l66_66470


namespace triangle_inequality_l66_66321

theorem triangle_inequality 
  (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l66_66321


namespace part1_part2_l66_66982

theorem part1 : ∃ x : ℝ, 3 * x = 4.5 ∧ x = 4.5 - 3 :=
by {
  -- Skipping the proof for now
  sorry
}

theorem part2 (m : ℝ) (h : ∃ x : ℝ, 5 * x - m = 1 ∧ x = 1 - m - 5) : m = 21 / 4 :=
by {
  -- Skipping the proof for now
  sorry
}

end part1_part2_l66_66982


namespace log_inequality_l66_66409

theorem log_inequality : 
  ∀ (logπ2 log2π : ℝ), logπ2 = 1 / log2π → 0 < logπ2 → 0 < log2π → (1 / logπ2 + 1 / log2π > 2) :=
by
  intros logπ2 log2π h1 h2 h3
  have h4: logπ2 = 1 / log2π := h1
  have h5: 0 < logπ2 := h2
  have h6: 0 < log2π := h3
  -- To be completed with the actual proof steps if needed
  sorry

end log_inequality_l66_66409


namespace sum_of_coefficients_l66_66683

theorem sum_of_coefficients : 
  ∃ (a b c d e f g h j k : ℤ), 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) → 
    (a + b + c + d + e + f + g + h + j + k = 92) :=
sorry

end sum_of_coefficients_l66_66683


namespace medals_award_count_l66_66943

theorem medals_award_count :
  let total_ways (n k : ℕ) := n.factorial / (n - k).factorial
  ∃ (award_ways : ℕ), 
    let no_americans := total_ways 6 3
    let one_american := 4 * 3 * total_ways 6 2
    award_ways = no_americans + one_american ∧
    award_ways = 480 :=
by
  sorry

end medals_award_count_l66_66943


namespace midpoint_on_hyperbola_l66_66527

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66527


namespace b_share_l66_66870

theorem b_share (a b c : ℕ) (h1 : a + b + c = 120) (h2 : a = b + 20) (h3 : a = c - 20) : b = 20 :=
by
  sorry

end b_share_l66_66870


namespace four_consecutive_integers_product_plus_one_is_square_l66_66961

theorem four_consecutive_integers_product_plus_one_is_square (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n^2 + n - 1)^2 := by
  sorry

end four_consecutive_integers_product_plus_one_is_square_l66_66961


namespace john_spent_fraction_at_arcade_l66_66464

theorem john_spent_fraction_at_arcade 
  (allowance : ℝ) (spent_arcade : ℝ) (spent_candy_store : ℝ) 
  (h1 : allowance = 3.45)
  (h2 : spent_candy_store = 0.92)
  (h3 : 3.45 - spent_arcade - (1/3) * (3.45 - spent_arcade) = spent_candy_store) :
  spent_arcade / allowance = 2.07 / 3.45 :=
by
  sorry

end john_spent_fraction_at_arcade_l66_66464


namespace four_x_plus_y_greater_than_four_z_l66_66379

theorem four_x_plus_y_greater_than_four_z
  (x y z : ℝ)
  (h1 : y > 2 * z)
  (h2 : 2 * z > 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) > 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z)
  : 4 * x + y > 4 * z := 
by
  sorry

end four_x_plus_y_greater_than_four_z_l66_66379


namespace cylinder_projections_tangency_l66_66413

def plane1 : Type := sorry
def plane2 : Type := sorry
def projection_axis : Type := sorry
def is_tangent_to (cylinder : Type) (plane : Type) : Prop := sorry
def is_base_tangent_to (cylinder : Type) (axis : Type) : Prop := sorry
def cylinder : Type := sorry

theorem cylinder_projections_tangency (P1 P2 : Type) (axis : Type)
  (h1 : is_tangent_to cylinder P1) 
  (h2 : is_tangent_to cylinder P2) 
  (h3 : is_base_tangent_to cylinder axis) : 
  ∃ (solutions : ℕ), solutions = 4 :=
sorry

end cylinder_projections_tangency_l66_66413


namespace total_seeds_l66_66978

theorem total_seeds (A B C : ℕ) (h₁ : A = B + 10) (h₂ : B = 30) (h₃ : C = 30) : A + B + C = 100 :=
by
  sorry

end total_seeds_l66_66978


namespace find_x_l66_66672

theorem find_x (x : ℕ) :
  (3 * x > 91 ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ ¬(x < 120) ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ ¬(x < 27) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7)) →
  x = 9 :=
sorry

end find_x_l66_66672


namespace value_of_a_plus_b_l66_66280

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end value_of_a_plus_b_l66_66280


namespace find_b_minus_a_l66_66275

theorem find_b_minus_a (a b : ℝ) (h : ∀ x : ℝ, 0 ≤ x → 
  0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) : 
  b - a = 2 :=
sorry

end find_b_minus_a_l66_66275


namespace Allen_age_difference_l66_66089

theorem Allen_age_difference (M A : ℕ) (h1 : M = 30) (h2 : (A + 3) + (M + 3) = 41) : M - A = 25 :=
by
  sorry

end Allen_age_difference_l66_66089


namespace smallest_product_of_digits_l66_66960

theorem smallest_product_of_digits : 
  ∃ (a b c d : ℕ), 
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) ∧ 
  (∃ x y : ℕ, (x = a * 10 + c ∧ y = b * 10 + d) ∨ (x = a * 10 + d ∧ y = b * 10 + c) ∨ (x = b * 10 + c ∧ y = a * 10 + d) ∨ (x = b * 10 + d ∧ y = a * 10 + c)) ∧
  (∀ x1 y1 x2 y2 : ℕ, ((x1 = 34 ∧ y1 = 56 ∨ x1 = 35 ∧ y1 = 46) ∧ (x2 = 34 ∧ y2 = 56 ∨ x2 = 35 ∧ y2 = 46)) → x1 * y1 ≥ x2 * y2) ∧
  35 * 46 = 1610 :=
sorry

end smallest_product_of_digits_l66_66960


namespace foil_covered_prism_width_l66_66999

theorem foil_covered_prism_width
    (l w h : ℕ)
    (inner_volume : l * w * h = 128)
    (width_length_relation : w = 2 * l)
    (width_height_relation : w = 2 * h) :
    (w + 2) = 10 := 
sorry

end foil_covered_prism_width_l66_66999


namespace halfway_fraction_l66_66030

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66030


namespace hyperbola_midpoint_l66_66590

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66590


namespace midpoint_hyperbola_l66_66546

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66546


namespace puzzle_piece_total_l66_66463

theorem puzzle_piece_total :
  let p1 := 1000
  let p2 := p1 + 0.30 * p1
  let p3 := 2 * p2
  let p4 := (p1 + p3) + 0.50 * (p1 + p3)
  let p5 := 3 * p4
  let p6 := p1 + p2 + p3 + p4 + p5
  p1 + p2 + p3 + p4 + p5 + p6 = 55000
:= sorry

end puzzle_piece_total_l66_66463


namespace find_dividend_l66_66920

-- Define the given constants
def quotient : ℕ := 909899
def divisor : ℕ := 12

-- Define the dividend as the product of divisor and quotient
def dividend : ℕ := divisor * quotient

-- The theorem stating the equality we need to prove
theorem find_dividend : dividend = 10918788 := by
  sorry

end find_dividend_l66_66920


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66037

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66037


namespace ratio_p_q_is_minus_one_l66_66447

theorem ratio_p_q_is_minus_one (p q : ℤ) (h : (25 / 7 : ℝ) + ((2 * q - p) / (2 * q + p) : ℝ) = 4) : (p / q : ℝ) = -1 := 
sorry

end ratio_p_q_is_minus_one_l66_66447


namespace can_be_midpoint_of_AB_l66_66518

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66518


namespace least_number_of_cars_per_work_day_l66_66831

-- Define the conditions as constants in Lean
def paul_work_hours_per_day := 8
def jack_work_hours_per_day := 8
def paul_cars_per_hour := 2
def jack_cars_per_hour := 3

-- Define the total number of cars Paul and Jack can change in a workday
def total_cars_per_day := (paul_cars_per_hour + jack_cars_per_hour) * paul_work_hours_per_day

-- State the theorem to be proved
theorem least_number_of_cars_per_work_day : total_cars_per_day = 40 := by
  -- Proof goes here
  sorry

end least_number_of_cars_per_work_day_l66_66831


namespace cos_theta_four_times_l66_66134

theorem cos_theta_four_times (theta : ℝ) (h : Real.cos theta = 1 / 3) : 
  Real.cos (4 * theta) = 17 / 81 := 
sorry

end cos_theta_four_times_l66_66134


namespace construct_triangle_l66_66097

variables (a : ℝ) (α : ℝ) (d : ℝ)

-- Helper definitions
def is_triangle_valid (a α d : ℝ) : Prop := sorry

-- The theorem to be proven
theorem construct_triangle (a α d : ℝ) : is_triangle_valid a α d :=
sorry

end construct_triangle_l66_66097


namespace grown_ups_in_milburg_l66_66364

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l66_66364


namespace martha_bedroom_size_l66_66361

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l66_66361


namespace cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l66_66459

theorem cannot_represent_1986_as_sum_of_squares_of_6_odd_integers
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 % 2 = 1) 
  (h2 : a2 % 2 = 1) 
  (h3 : a3 % 2 = 1) 
  (h4 : a4 % 2 = 1) 
  (h5 : a5 % 2 = 1) 
  (h6 : a6 % 2 = 1) : 
  ¬ (1986 = a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2) := 
by 
  sorry

end cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l66_66459


namespace efficiency_difference_l66_66996

variables (Rp Rq : ℚ)

-- Given conditions
def p_rate := Rp = 1 / 21
def combined_rate := Rp + Rq = 1 / 11

-- Define the percentage efficiency difference
def percentage_difference := (Rp - Rq) / Rq * 100

-- Main statement to prove
theorem efficiency_difference : 
  p_rate Rp ∧ 
  combined_rate Rp Rq → 
  percentage_difference Rp Rq = 10 :=
sorry

end efficiency_difference_l66_66996


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66634

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66634


namespace snail_kite_eats_35_snails_l66_66390

theorem snail_kite_eats_35_snails : 
  let day1 := 3
  let day2 := day1 + 2
  let day3 := day2 + 2
  let day4 := day3 + 2
  let day5 := day4 + 2
  day1 + day2 + day3 + day4 + day5 = 35 := 
by
  sorry

end snail_kite_eats_35_snails_l66_66390


namespace asymptotes_of_hyperbola_eq_m_l66_66786

theorem asymptotes_of_hyperbola_eq_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), (x^2 / 16 - y^2 / 25 = 1) → (y = m * x ∨ y = -m * x)) → m = 5 / 4 :=
by 
  sorry

end asymptotes_of_hyperbola_eq_m_l66_66786


namespace women_in_third_group_l66_66221

variables (m w : ℝ)

theorem women_in_third_group (h1 : 3 * m + 8 * w = 6 * m + 2 * w) (x : ℝ) (h2 : 2 * m + x * w = 0.5 * (3 * m + 8 * w)) :
  x = 4 :=
sorry

end women_in_third_group_l66_66221


namespace midpoint_of_hyperbola_l66_66575

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66575


namespace sum_geometric_sequence_first_10_terms_l66_66111

theorem sum_geometric_sequence_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1 / 3
  let S₁₀ : ℚ := 12 * (1 - (1 / 3)^10) / (1 - 1 / 3)
  S₁₀ = 1062864 / 59049 := by
  sorry

end sum_geometric_sequence_first_10_terms_l66_66111


namespace arctan_sum_pi_l66_66411

open Real

theorem arctan_sum_pi : arctan (1 / 3) + arctan (3 / 8) + arctan (8 / 3) = π := 
sorry

end arctan_sum_pi_l66_66411


namespace fraction_half_way_l66_66022

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66022


namespace ram_ravi_selected_probability_l66_66212

noncomputable def probability_both_selected : ℝ := 
  let probability_ram_80 := (1 : ℝ) / 7
  let probability_ravi_80 := (1 : ℝ) / 5
  let probability_both_80 := probability_ram_80 * probability_ravi_80
  let num_applicants := 200
  let num_spots := 4
  let probability_single_selection := (num_spots : ℝ) / (num_applicants : ℝ)
  let probability_both_selected_given_80 := probability_single_selection * probability_single_selection
  probability_both_80 * probability_both_selected_given_80

theorem ram_ravi_selected_probability :
  probability_both_selected = 1 / 87500 := 
by
  sorry

end ram_ravi_selected_probability_l66_66212


namespace sum_x_y_is_4_l66_66951

theorem sum_x_y_is_4 {x y : ℝ} (h : x / (1 - (I : ℂ)) + y / (1 - 2 * I) = 5 / (1 - 3 * I)) : x + y = 4 :=
sorry

end sum_x_y_is_4_l66_66951


namespace range_of_a_in_third_quadrant_l66_66276

theorem range_of_a_in_third_quadrant (a : ℝ) :
  let Z_re := a^2 - 2*a
  let Z_im := a^2 - a - 2
  (Z_re < 0 ∧ Z_im < 0) → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_in_third_quadrant_l66_66276


namespace sam_compound_interest_l66_66072

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

theorem sam_compound_interest : 
  compound_interest 3000 0.10 2 1 = 3307.50 :=
by
  sorry

end sam_compound_interest_l66_66072


namespace units_digit_of_sum_is_4_l66_66922

-- Definitions and conditions based on problem
def base_8_add (a b : List Nat) : List Nat :=
    sorry -- Function to perform addition in base 8, returning result as a list of digits

def units_digit (a : List Nat) : Nat :=
    a.headD 0  -- Function to get the units digit of the result

-- The list representation for the digits of 65 base 8 and 37 base 8
def sixty_five_base8 := [6, 5]
def thirty_seven_base8 := [3, 7]

-- The theorem that asserts the final result
theorem units_digit_of_sum_is_4 : units_digit (base_8_add sixty_five_base8 thirty_seven_base8) = 4 :=
    sorry

end units_digit_of_sum_is_4_l66_66922


namespace average_cost_is_thirteen_l66_66761

noncomputable def averageCostPerPen (pensCost shippingCost : ℝ) (totalPens : ℕ) : ℕ :=
  Nat.ceil ((pensCost + shippingCost) * 100 / totalPens)

theorem average_cost_is_thirteen :
  averageCostPerPen 29.85 8.10 300 = 13 :=
by
  sorry

end average_cost_is_thirteen_l66_66761


namespace total_inflation_time_l66_66228

theorem total_inflation_time (time_per_ball : ℕ) (alexia_balls : ℕ) (extra_balls : ℕ) : 
  time_per_ball = 20 → alexia_balls = 20 → extra_balls = 5 →
  (alexia_balls * time_per_ball) + ((alexia_balls + extra_balls) * time_per_ball) = 900 :=
by 
  intros h1 h2 h3
  sorry

end total_inflation_time_l66_66228


namespace number_divided_by_3_equals_subtract_3_l66_66712

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l66_66712


namespace find_x_l66_66914

theorem find_x (x : ℝ) (h : (x * (x ^ 4) ^ (1/2)) ^ (1/4) = 2) : 
  x = 16 ^ (1/3) :=
sorry

end find_x_l66_66914


namespace sum_of_possible_values_l66_66849

theorem sum_of_possible_values {x : ℝ} :
  (3 * (x - 3)^2 = (x - 2) * (x + 5)) →
  (∃ (x1 x2 : ℝ), x1 + x2 = 10.5) :=
by sorry

end sum_of_possible_values_l66_66849


namespace no_solution_xn_yn_zn_l66_66933

theorem no_solution_xn_yn_zn (x y z n : ℕ) (h : n ≥ z) : ¬ (x^n + y^n = z^n) :=
sorry

end no_solution_xn_yn_zn_l66_66933


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66635

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66635


namespace meet_floor_l66_66068

noncomputable def xiaoming_meets_xiaoying (x y meet_floor: ℕ) : Prop :=
  x = 4 → y = 3 → (meet_floor = 22)

theorem meet_floor (x y meet_floor: ℕ) (h1: x = 4) (h2: y = 3) :
  xiaoming_meets_xiaoying x y meet_floor :=
by
  sorry

end meet_floor_l66_66068


namespace sequence_general_term_l66_66313

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) = 2^n * a n) : 
  ∀ n, a n = 2^((n-1)*n / 2) := sorry

end sequence_general_term_l66_66313


namespace non_zero_real_x_solution_l66_66371

theorem non_zero_real_x_solution (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 := by
  sorry

end non_zero_real_x_solution_l66_66371


namespace ratio_of_radii_of_truncated_cone_l66_66897

theorem ratio_of_radii_of_truncated_cone 
  (R r s : ℝ) 
  (h1 : s = Real.sqrt (R * r)) 
  (h2 : (π * (R^2 + r^2 + R * r) * (2 * s) / 3) = 3 * (4 * π * s^3 / 3)) :
  R / r = 7 := 
sorry

end ratio_of_radii_of_truncated_cone_l66_66897


namespace max_dot_product_on_circle_l66_66824

theorem max_dot_product_on_circle :
  (∃(x y : ℝ),
    x^2 + (y - 3)^2 = 1 ∧
    2 ≤ y ∧ y ≤ 4 ∧
    (∀(y : ℝ), (2 ≤ y ∧ y ≤ 4 →
      (x^2 + y^2 - 4) ≤ 12))) := by
  sorry

end max_dot_product_on_circle_l66_66824


namespace two_digit_number_l66_66851

theorem two_digit_number (x y : ℕ) (h1 : x + y = 7) (h2 : (x + 2) + 10 * (y + 2) = 2 * (x + 10 * y) - 3) : (10 * y + x) = 25 :=
by
  sorry

end two_digit_number_l66_66851


namespace rectangle_area_in_ellipse_l66_66087

theorem rectangle_area_in_ellipse :
  ∃ a b : ℝ, 2 * a = b ∧ (a^2 / 4 + b^2 / 8 = 1) ∧ 2 * a * b = 16 :=
by
  sorry

end rectangle_area_in_ellipse_l66_66087


namespace larger_triangle_perimeter_l66_66763

def is_similar (a b c : ℕ) (x y z : ℕ) : Prop :=
  x * c = z * a ∧
  x * c = z * b ∧
  y * c = z * a ∧
  y * c = z * c ∧
  a ≠ b ∧ c ≠ b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∧ a ≠ c

theorem larger_triangle_perimeter (a b c x y z : ℕ) 
  (h1 : is_isosceles a b c) 
  (h2 : is_similar a b c x y z) 
  (h3 : c = 12) 
  (h4 : z = 36)
  (h5 : a = 7) 
  (h6 : b = 7) : 
  x + y + z = 78 :=
sorry

end larger_triangle_perimeter_l66_66763


namespace number_of_speaking_orders_l66_66223

-- The conditions of the problem:
variables (A B C D E F G H : Type) (select : Finset (A ⊕ B ⊕ C ⊕ D ⊕ E ⊕ F ⊕ G ⊕ H)) 
  (at_least_one_of_AB : (A ∈ select ∨ B ∈ select))
  (AB_exactly_one_between : (A ∈ select ∧ B ∈ select) → ∃ x ∈ select, (ensure_x_between_A_and_B : true))  -- Placeholder, higher-order logic needed

-- The goal, which is the mathematically equivalent proof problem:
theorem number_of_speaking_orders : Finset.card select = 1080 := 
sorry

end number_of_speaking_orders_l66_66223


namespace range_of_a_l66_66850

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
sorry

end range_of_a_l66_66850


namespace find_x_l66_66744

theorem find_x (x : ℤ) (h : (2 + 76 + x) / 3 = 5) : x = -63 := 
sorry

end find_x_l66_66744


namespace halfway_fraction_l66_66031

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66031


namespace three_digit_number_divisible_by_11_l66_66372

theorem three_digit_number_divisible_by_11 : 
  (∀ x : ℕ, x < 10 → (600 + 10 * x + 3) % 11 = 0 → 600 + 10 * x + 3 = 693) :=
by 
  intros x x_lt_10 h 
  have h1 : 600 % 11 = 7 := by norm_num
  have h2 : (10 * x + 3) % 11 = (10 * x + 3) % 11 := by norm_num
  rw Nat.add_mod at h 
  rw [h1, h2] at h 
  have h3 : (7 + (10 * x + 3) % 11) % 11 = 0 := by rw ← h 
  rw Nat.add_mod at h3 
  cases x 
  case h_0 => rw zero_mul at * 
             simp at h3 
             norm_num at h3
  cases x 
  case h_0 => sorry -- Assume this case has been proved
  case h_succ x_1 => sorry -- Assume this case has been proved
  sorry

end three_digit_number_divisible_by_11_l66_66372


namespace college_application_ways_correct_l66_66225

def college_application_ways : ℕ :=
  -- Scenario 1: Student does not apply to either of the two conflicting colleges
  (Nat.choose 4 3) +
  -- Scenario 2: Student applies to one of the two conflicting colleges
  ((Nat.choose 2 1) * (Nat.choose 4 2))

theorem college_application_ways_correct : college_application_ways = 16 := by
  -- We can skip the proof
  sorry

end college_application_ways_correct_l66_66225


namespace probability_at_least_one_diamond_l66_66079

theorem probability_at_least_one_diamond :
  ∃ p : ℚ, p = 15 / 34 ∧ 
  let no_replacement := true,
      total_cards := 52,
      diamonds := 13,
      non_diamonds := 39 in
  ∀ first_card second_card : ℕ,
  first_card ∈ {0...total_cards - 1} ∧ second_card ∈ {0...total_cards - 2} ∧ first_card ≠ second_card →
  p = 1 - (non_diamonds / (total_cards : ℚ)) * ((non_diamonds - 1) / (total_cards - 1 : ℚ)) :=
sorry

end probability_at_least_one_diamond_l66_66079


namespace problem_statement_l66_66117

def has_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m * x - 1 = 0

def p : Prop := ∀ m : ℝ, has_solutions m

def q : Prop := ∃ x_0 : ℕ, x_0^2 - 2 * x_0 - 1 ≤ 0

theorem problem_statement : ¬ (p ∧ ¬ q) := 
sorry

end problem_statement_l66_66117


namespace union_of_A_B_l66_66130

open Set

def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem union_of_A_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 2} :=
by sorry

end union_of_A_B_l66_66130


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66033

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66033


namespace part_a_part_b_part_c_l66_66646

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l66_66646


namespace roots_mul_shift_eq_neg_2018_l66_66950

theorem roots_mul_shift_eq_neg_2018 {a b : ℝ}
  (h1 : a + b = -1)
  (h2 : a * b = -2020) :
  (a - 1) * (b - 1) = -2018 :=
sorry

end roots_mul_shift_eq_neg_2018_l66_66950


namespace even_sum_less_than_100_l66_66051

theorem even_sum_less_than_100 : 
  (∑ k in (Finset.range 50).filter (λ x, x % 2 = 0), k) = 2450 := by
  sorry

end even_sum_less_than_100_l66_66051


namespace relay_scheme_count_l66_66145

theorem relay_scheme_count
  (num_segments : ℕ)
  (num_torchbearers : ℕ)
  (first_choices : ℕ)
  (last_choices : ℕ) :
  num_segments = 6 ∧
  num_torchbearers = 6 ∧
  first_choices = 3 ∧
  last_choices = 2 →
  ∃ num_schemes : ℕ, num_schemes = 7776 :=
by
  intro h
  obtain ⟨h_segments, h_torchbearers, h_first_choices, h_last_choices⟩ := h
  exact ⟨7776, sorry⟩

end relay_scheme_count_l66_66145


namespace midpoint_hyperbola_l66_66547

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66547


namespace steve_final_height_l66_66677

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l66_66677


namespace factorize_expr_solve_inequality_solve_equation_simplify_expr_l66_66245

-- Problem 1
theorem factorize_expr (x y m n : ℝ) : x^2 * (3 * m - 2 * n) + y^2 * (2 * n - 3 * m) = (3 * m - 2 * n) * (x + y) * (x - y) := 
sorry

-- Problem 2
theorem solve_inequality (x : ℝ) : 
  (∃ x, (x - 3) / 2 + 3 > x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) → -2 < x ∧ x < 1 :=
sorry

-- Problem 3
theorem solve_equation (x : ℝ) : 
  (∃ x, (3 - x) / (x - 4) + 1 / (4 - x) = 1) → x = 3 :=
sorry

-- Problem 4
theorem simplify_expr (a : ℝ) (h : a = 3) : 
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a - 1)) = 3 / 4 :=
sorry

end factorize_expr_solve_inequality_solve_equation_simplify_expr_l66_66245


namespace factorize_polynomial_l66_66259

theorem factorize_polynomial (x y : ℝ) : (3 * x^2 - 3 * y^2) = 3 * (x + y) * (x - y) := 
by
  sorry

end factorize_polynomial_l66_66259


namespace midpoint_of_hyperbola_segment_l66_66507

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66507


namespace hyperbola_midpoint_exists_l66_66560

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66560


namespace solution_is_consecutive_even_integers_l66_66771

def consecutive_even_integers_solution_exists : Prop :=
  ∃ (x y z w : ℕ), (x + y + z + w = 68) ∧ 
                   (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) ∧
                   (x % 2 = 0) ∧ (y % 2 = 0) ∧ (z % 2 = 0) ∧ (w % 2 = 0)

theorem solution_is_consecutive_even_integers : consecutive_even_integers_solution_exists :=
sorry

end solution_is_consecutive_even_integers_l66_66771


namespace max_y_coordinate_l66_66421

noncomputable def y_coordinate (θ : Real) : Real :=
  let u := Real.sin θ
  3 * u - 4 * u^3

theorem max_y_coordinate : ∃ θ, y_coordinate θ = 1 := by
  use Real.arcsin (1 / 2)
  sorry

end max_y_coordinate_l66_66421


namespace opposite_face_A_is_E_l66_66887

-- Axiomatically defining the basic conditions from the problem statement.

-- We have six labels for the faces of a net
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def adjacent (x y : Face) : Prop :=
  (x = A ∧ y = B) ∨ (x = A ∧ y = D) ∨ (x = B ∧ y = A) ∨ (x = D ∧ y = A)

-- Define the "not directly attached" relationship
def not_adjacent (x y : Face) : Prop :=
  ¬adjacent x y

-- Given the conditions in the problem statement
axiom condition1 : adjacent A B
axiom condition2 : adjacent A D
axiom condition3 : not_adjacent A E

-- The proof objective is to show that E is the face opposite to A
theorem opposite_face_A_is_E : ∃ (F : Face), 
  (∀ x : Face, adjacent A x ∨ not_adjacent A x) → (∀ y : Face, adjacent A y ↔ y ≠ E) → E = F :=
sorry

end opposite_face_A_is_E_l66_66887


namespace correct_midpoint_l66_66488

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66488


namespace products_B_correct_l66_66700

-- Define the total number of products
def total_products : ℕ := 4800

-- Define the sample size and the number of pieces from equipment A in the sample
def sample_size : ℕ := 80
def sample_A : ℕ := 50

-- Define the number of products produced by equipment A and B
def products_A : ℕ := 3000
def products_B : ℕ := total_products - products_A

-- The target number of products produced by equipment B
def target_products_B : ℕ := 1800

-- The theorem we need to prove
theorem products_B_correct :
  products_B = target_products_B := by
  sorry

end products_B_correct_l66_66700


namespace equal_numbers_l66_66815

namespace MathProblem

theorem equal_numbers 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x^2 / y + y^2 / z + z^2 / x = x^2 / z + z^2 / y + y^2 / x) : 
  x = y ∨ x = z ∨ y = z :=
by
  sorry

end MathProblem

end equal_numbers_l66_66815


namespace n_must_be_power_of_3_l66_66839

theorem n_must_be_power_of_3 (n : ℕ) (h1 : 0 < n) (h2 : Prime (4 ^ n + 2 ^ n + 1)) : ∃ k : ℕ, n = 3 ^ k :=
by
  sorry

end n_must_be_power_of_3_l66_66839


namespace strictly_increasing_function_exists_l66_66952

noncomputable def exists_strictly_increasing_function (f : ℕ → ℕ) :=
  (∀ n : ℕ, n = 1 → f n = 2) ∧
  (∀ n : ℕ, f (f n) = f n + n) ∧
  (∀ m n : ℕ, m < n → f m < f n)

theorem strictly_increasing_function_exists : 
  ∃ f : ℕ → ℕ,
  exists_strictly_increasing_function f :=
sorry

end strictly_increasing_function_exists_l66_66952


namespace joe_has_more_shirts_l66_66904

theorem joe_has_more_shirts (alex_shirts : ℕ) (ben_shirts : ℕ) (ben_joe_diff : ℕ)
  (h_a : alex_shirts = 4)
  (h_b : ben_shirts = 15)
  (h_bj : ben_shirts = joe_shirts + ben_joe_diff)
  (h_bj_diff : ben_joe_diff = 8) :
  joe_shirts - alex_shirts = 3 :=
by {
  sorry
}

end joe_has_more_shirts_l66_66904


namespace factor_expression_l66_66918

theorem factor_expression :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) :=
by
  sorry

end factor_expression_l66_66918


namespace range_of_alpha_minus_beta_l66_66428

theorem range_of_alpha_minus_beta (α β : Real) (h₁ : -180 < α) (h₂ : α < β) (h₃ : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l66_66428


namespace trip_attendees_trip_cost_savings_l66_66209

theorem trip_attendees (total_people : ℕ) (total_cost : ℕ) (adult_ticket : ℕ) 
(student_discount : ℕ) (group_discount : ℕ) (adults : ℕ) (students : ℕ) :
total_people = 130 → total_cost = 9600 → adult_ticket = 120 →
student_discount = 50 → group_discount = 40 → 
total_people = adults + students → 
total_cost = adults * adult_ticket + students * (adult_ticket * student_discount / 100) →
adults = 30 ∧ students = 100 :=
by sorry

theorem trip_cost_savings (total_people : ℕ) (individual_total_cost : ℕ) 
(group_total_cost : ℕ) (student_tickets : ℕ) (group_tickets : ℕ) 
(adult_ticket : ℕ) (student_discount : ℕ) (group_discount : ℕ) :
(total_people = 130) → (individual_total_cost = 7200 + 1800) → 
(group_total_cost = total_people * (adult_ticket * group_discount / 100)) →
(adult_ticket = 120) → (student_discount = 50) → (group_discount = 40) → 
(total_people = student_tickets + group_tickets) → (student_tickets = 30) → 
(group_tickets = 100) → (7200 + 1800 < 9360) → 
student_tickets = 30 ∧ group_tickets = 100 :=
by sorry

end trip_attendees_trip_cost_savings_l66_66209


namespace fractional_equation_solution_l66_66992

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (2 - x) - 1 = (2 * x - 5) / (x - 2) → x = 3 :=
by 
  intro h_eq
  sorry

end fractional_equation_solution_l66_66992


namespace midpoint_hyperbola_l66_66533

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66533


namespace fraction_half_way_l66_66016

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66016


namespace halfway_fraction_l66_66004

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66004


namespace value_of_a_plus_b_l66_66279

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end value_of_a_plus_b_l66_66279


namespace math_problem_l66_66873

theorem math_problem :
  (625.3729 * (4500 + 2300 ^ 2) - Real.sqrt 84630) / (1500 ^ 3 * 48 ^ 2) = 0.0004257 :=
by
  sorry

end math_problem_l66_66873


namespace unique_elements_condition_l66_66277

theorem unique_elements_condition (x : ℝ) : 
  (1 ≠ x ∧ x ≠ x^2 ∧ 1 ≠ x^2) ↔ (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :=
by 
  sorry

end unique_elements_condition_l66_66277


namespace find_m_l66_66684

-- Definition and conditions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def vertex_property (a b c : ℝ) : Prop := 
  (∀ x, quadratic a b c x ≤ quadratic a b c 2) ∧ quadratic a b c 2 = 4

noncomputable def passes_through_origin (a b c : ℝ) : Prop :=
  quadratic a b c 0 = -7

-- Main theorem statement
theorem find_m (a b c m : ℝ) 
  (h1 : vertex_property a b c) 
  (h2 : passes_through_origin a b c) 
  (h3 : quadratic a b c 5 = m) :
  m = -83/4 :=
sorry

end find_m_l66_66684


namespace fraction_half_way_l66_66021

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66021


namespace probability_four_red_four_blue_l66_66402

noncomputable def urn_probability : ℚ :=
  let initial_red := 2
  let initial_blue := 1
  let operations := 5
  let final_red := 4
  let final_blue := 4
  -- calculate the probability using given conditions, this result is directly derived as 2/7
  2 / 7

theorem probability_four_red_four_blue :
  urn_probability = 2 / 7 :=
by
  sorry

end probability_four_red_four_blue_l66_66402


namespace parallelogram_not_symmetrical_l66_66737

def is_symmetrical (shape : String) : Prop :=
  shape = "Circle" ∨ shape = "Rectangle" ∨ shape = "Isosceles Trapezoid"

theorem parallelogram_not_symmetrical : ¬ is_symmetrical "Parallelogram" :=
by
  sorry

end parallelogram_not_symmetrical_l66_66737


namespace fraction_halfway_between_l66_66041

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66041


namespace find_x_l66_66727

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l66_66727


namespace determine_set_of_integers_for_ratio_l66_66125

def arithmetic_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n / T n = (31 * n + 101) / (n + 3)

def ratio_is_integer (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, a n / b n = k

theorem determine_set_of_integers_for_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ) :
  arithmetic_sequences a b S T →
  {n : ℕ | ratio_is_integer a b n} = {1, 3} :=
sorry

end determine_set_of_integers_for_ratio_l66_66125


namespace average_weight_of_class_l66_66204

variable (SectionA_students : ℕ := 26)
variable (SectionB_students : ℕ := 34)
variable (SectionA_avg_weight : ℝ := 50)
variable (SectionB_avg_weight : ℝ := 30)

theorem average_weight_of_class :
  (SectionA_students * SectionA_avg_weight + SectionB_students * SectionB_avg_weight) / (SectionA_students + SectionB_students) = 38.67 := by
  sorry

end average_weight_of_class_l66_66204


namespace alloy_chromium_amount_l66_66811

theorem alloy_chromium_amount
  (x : ℝ) -- The amount of the first alloy used (in kg)
  (h1 : 0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) -- Condition based on percentages of chromium
  : x = 15 := 
by
  sorry

end alloy_chromium_amount_l66_66811


namespace number_of_houses_l66_66206

theorem number_of_houses (total_mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : total_mail_per_block = 24) (h2 : mail_per_house = 4) : total_mail_per_block / mail_per_house = 6 :=
by
  sorry

end number_of_houses_l66_66206


namespace fg_of_3_is_94_l66_66936

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 8

theorem fg_of_3_is_94 : f (g 3) = 94 := by
  sorry

end fg_of_3_is_94_l66_66936


namespace factorize_polynomial_l66_66260

theorem factorize_polynomial (x y : ℝ) : (3 * x^2 - 3 * y^2) = 3 * (x + y) * (x - y) := 
by
  sorry

end factorize_polynomial_l66_66260


namespace min_value_inequality_l66_66624

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 / x + 1 / y) * (4 * x + y) ≥ 9 ∧ ((1 / x + 1 / y) * (4 * x + y) = 9 ↔ y / x = 2) :=
by
  sorry

end min_value_inequality_l66_66624


namespace custom_operation_correct_l66_66938

noncomputable def custom_operation (a b c : ℕ) : ℝ :=
  (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem custom_operation_correct : custom_operation 6 15 5 = 2 := by
  sorry

end custom_operation_correct_l66_66938


namespace remainder_3_pow_19_mod_10_l66_66709

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_3_pow_19_mod_10_l66_66709


namespace k_cubed_divisible_l66_66295

theorem k_cubed_divisible (k : ℕ) (h : k = 84) : ∃ n : ℕ, k ^ 3 = 592704 * n :=
by
  sorry

end k_cubed_divisible_l66_66295


namespace roots_greater_than_two_range_l66_66452

theorem roots_greater_than_two_range (m : ℝ) :
  ∀ x1 x2 : ℝ, (x1^2 + (m - 4) * x1 + 6 - m = 0) ∧ (x2^2 + (m - 4) * x2 + 6 - m = 0) ∧ (x1 > 2) ∧ (x2 > 2) →
  -2 < m ∧ m ≤ 2 - 2 * Real.sqrt 3 :=
by
  sorry

end roots_greater_than_two_range_l66_66452


namespace mean_height_of_players_l66_66362

def heights_50s : List ℕ := [57, 59]
def heights_60s : List ℕ := [62, 64, 64, 65, 65, 68, 69]
def heights_70s : List ℕ := [70, 71, 73, 75, 75, 77, 78]

def all_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s

def mean_height (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

theorem mean_height_of_players :
  mean_height all_heights = 68.25 :=
by
  sorry

end mean_height_of_players_l66_66362


namespace inequality_proofs_l66_66292

def sinSumInequality (A B C ε : ℝ) : Prop :=
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3

def sinProductInequality (A B C ε : ℝ) : Prop :=
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C)

theorem inequality_proofs (A B C ε : ℝ) (hA : 0 ≤ A ∧ A ≤ Real.pi) (hB : 0 ≤ B ∧ B ≤ Real.pi) 
  (hC : 0 ≤ C ∧ C ≤ Real.pi) (hε : ε ≥ 1) :
  sinSumInequality A B C ε ∧ sinProductInequality A B C ε :=
by
  sorry

end inequality_proofs_l66_66292


namespace num_five_ruble_coins_l66_66654

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l66_66654


namespace can_be_midpoint_of_AB_l66_66514

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66514


namespace find_larger_number_l66_66740

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 := 
sorry

end find_larger_number_l66_66740


namespace petya_five_ruble_coins_count_l66_66663

theorem petya_five_ruble_coins_count (total_coins : ℕ) (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ)
   (h_total_coins : total_coins = 25)
   (h_not_two_ruble : not_two_ruble = 19)
   (h_not_ten_ruble : not_ten_ruble = 20)
   (h_not_one_ruble : not_one_ruble = 16) :
   let two_ruble := total_coins - not_two_ruble,
       ten_ruble := total_coins - not_ten_ruble,
       one_ruble := total_coins - not_one_ruble in
   (total_coins - (two_ruble + ten_ruble + one_ruble)) = 5 :=
by 
  sorry

end petya_five_ruble_coins_count_l66_66663


namespace number_divided_by_3_equals_subtract_3_l66_66714

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l66_66714


namespace midpoint_of_hyperbola_l66_66479

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66479


namespace find_u_plus_v_l66_66819

theorem find_u_plus_v (u v : ℤ) (huv : 0 < v ∧ v < u) (h_area : u * u + 3 * u * v = 451) : u + v = 21 := 
sorry

end find_u_plus_v_l66_66819


namespace Rikki_earnings_l66_66668

theorem Rikki_earnings
  (price_per_word : ℝ := 0.01)
  (words_per_5_minutes : ℕ := 25)
  (total_minutes : ℕ := 120)
  (earning : ℝ := 6)
  : price_per_word * (words_per_5_minutes * (total_minutes / 5)) = earning := by
  sorry

end Rikki_earnings_l66_66668


namespace total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66641

-- Given conditions
def num_buttons := 10
def num_correct_buttons := 3
def time_per_attempt := 2 -- seconds
def max_attempt_time := 60 -- seconds

-- Part a: Prove the total time Petya needs to try all combinations is 4 minutes
theorem total_time_to_get_inside : 
  (nat.choose num_buttons num_correct_buttons * time_per_attempt) / 60 = 4 :=
by
  sorry

-- Part b: Prove the average time Petya needs is 2 minutes and 1 second
theorem average_time_to_get_inside :
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) / 60 = 2 ∧
  ((1 + nat.choose num_buttons num_correct_buttons) * time_per_attempt / 2) % 60 = 1 :=
by
  sorry

-- Part c: Prove the probability that Petya will get inside in less than a minute is 29/120
theorem probability_to_get_inside_in_less_than_one_minute :
  (29 : ℚ) / (nat.choose num_buttons num_correct_buttons : ℚ) = 29 / 120 :=
by
  sorry

end total_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_one_minute_l66_66641


namespace betty_needs_five_boxes_l66_66768

def betty_oranges (total_oranges first_box second_box max_per_box : ℕ) : ℕ :=
  let remaining_oranges := total_oranges - (first_box + second_box)
  let full_boxes := remaining_oranges / max_per_box
  let extra_box := if remaining_oranges % max_per_box == 0 then 0 else 1
  full_boxes + 2 + extra_box

theorem betty_needs_five_boxes :
  betty_oranges 120 30 25 30 = 5 := 
by
  sorry

end betty_needs_five_boxes_l66_66768


namespace zero_integers_satisfy_conditions_l66_66110

noncomputable def satisfies_conditions (n : ℤ) : Prop :=
  ∃ k : ℤ, n * (25 - n) = k^2 * (25 - n)^2 ∧ n % 3 = 0

theorem zero_integers_satisfy_conditions :
  (∃ n : ℤ, satisfies_conditions n) → False := by
  sorry

end zero_integers_satisfy_conditions_l66_66110


namespace solve_eq1_solve_eq2_l66_66333

noncomputable def eq1_solution1 := -2 + Real.sqrt 5
noncomputable def eq1_solution2 := -2 - Real.sqrt 5

noncomputable def eq2_solution1 := 3
noncomputable def eq2_solution2 := 1

theorem solve_eq1 (x : ℝ) :
  x^2 + 4 * x - 1 = 0 → (x = eq1_solution1 ∨ x = eq1_solution2) :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  (x - 3)^2 + 2 * x * (x - 3) = 0 → (x = eq2_solution1 ∨ x = eq2_solution2) :=
by 
  sorry

end solve_eq1_solve_eq2_l66_66333


namespace candy_store_revenue_l66_66222

/-- A candy store sold 20 pounds of fudge for $2.50 per pound,
    5 dozen chocolate truffles for $1.50 each, 
    and 3 dozen chocolate-covered pretzels at $2.00 each.
    Prove that the total money made by the candy store is $212.00. --/
theorem candy_store_revenue :
  let fudge_pounds := 20
  let fudge_price_per_pound := 2.50
  let truffle_dozen := 5
  let truffle_price_each := 1.50
  let pretzel_dozen := 3
  let pretzel_price_each := 2.00
  (fudge_pounds * fudge_price_per_pound) + 
  (truffle_dozen * 12 * truffle_price_each) + 
  (pretzel_dozen * 12 * pretzel_price_each) = 212 :=
by
  sorry

end candy_store_revenue_l66_66222


namespace max_value_problem1_l66_66076

theorem max_value_problem1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ t, t = (1 / 2) * x * (1 - 2 * x) ∧ t ≤ 1 / 16 := sorry

end max_value_problem1_l66_66076


namespace x_to_the_12_eq_14449_l66_66136

/-
Given the condition x + 1/x = 2*sqrt(2), prove that x^12 = 14449.
-/

theorem x_to_the_12_eq_14449 (x : ℂ) (hx : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := 
sorry

end x_to_the_12_eq_14449_l66_66136


namespace number_divided_by_three_l66_66724

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l66_66724


namespace larger_number_of_two_l66_66219

theorem larger_number_of_two
  (HCF : ℕ)
  (factor1 : ℕ)
  (factor2 : ℕ)
  (cond_HCF : HCF = 23)
  (cond_factor1 : factor1 = 15)
  (cond_factor2 : factor2 = 16) :
  ∃ (A : ℕ), A = 23 * 16 := by
  sorry

end larger_number_of_two_l66_66219


namespace Euler_theorem_l66_66252

theorem Euler_theorem {m a : ℕ} (hm : m ≥ 1) (h_gcd : Nat.gcd a m = 1) : a ^ Nat.totient m ≡ 1 [MOD m] :=
by
  sorry

end Euler_theorem_l66_66252


namespace midpoint_on_hyperbola_l66_66600

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66600


namespace five_ruble_coins_count_l66_66651

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l66_66651


namespace solve_system_l66_66334

open Real

-- Define the system of equations as hypotheses
def eqn1 (x y z : ℝ) : Prop := x + y + 2 - 4 * x * y = 0
def eqn2 (x y z : ℝ) : Prop := y + z + 2 - 4 * y * z = 0
def eqn3 (x y z : ℝ) : Prop := z + x + 2 - 4 * z * x = 0

-- State the theorem
theorem solve_system (x y z : ℝ) :
  (eqn1 x y z ∧ eqn2 x y z ∧ eqn3 x y z) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by 
  sorry

end solve_system_l66_66334


namespace distance_between_points_l66_66104

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l66_66104


namespace problem1_problem2_l66_66081

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end problem1_problem2_l66_66081


namespace distance_between_points_l66_66106

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l66_66106


namespace midpoint_on_hyperbola_l66_66570

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66570


namespace growth_rate_double_l66_66457

noncomputable def lake_coverage (days : ℕ) : ℝ := if days = 39 then 1 else if days = 38 then 0.5 else 0  -- Simplified condition statement

theorem growth_rate_double (days : ℕ) : 
  (lake_coverage 39 = 1) → (lake_coverage 38 = 0.5) → (∀ n, lake_coverage (n + 1) = 2 * lake_coverage n) := 
  by 
  intros h39 h38 
  apply sorry  -- Proof not required

end growth_rate_double_l66_66457


namespace work_completion_times_l66_66981

-- Definitions based on conditions
def condition1 (x y : ℝ) : Prop := 2 * (1 / x) + 5 * (1 / y) = 1 / 2
def condition2 (x y : ℝ) : Prop := 3 * (1 / x + 1 / y) = 0.45

-- Main theorem stating the solution
theorem work_completion_times :
  ∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ x = 12 ∧ y = 15 := 
sorry

end work_completion_times_l66_66981


namespace sum_even_pos_ints_lt_100_l66_66058

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l66_66058


namespace compute_fraction_l66_66096

theorem compute_fraction :
  ( (12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500) ) /
  ( (6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500) ) = -182 :=
by
  sorry

end compute_fraction_l66_66096


namespace midpoint_hyperbola_l66_66538

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66538


namespace halfway_fraction_l66_66011

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66011


namespace hyperbola_midpoint_l66_66581

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66581


namespace value_of_w_div_x_l66_66794

theorem value_of_w_div_x (w x y : ℝ) 
  (h1 : w / x = a) 
  (h2 : w / y = 1 / 5) 
  (h3 : (x + y) / y = 2.2) : 
  w / x = 6 / 25 := by
  sorry

end value_of_w_div_x_l66_66794


namespace steve_height_after_growth_l66_66674

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l66_66674


namespace total_attendance_l66_66902

-- Defining the given conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 1
def total_amount_collected : ℕ := 50
def number_of_child_tickets : ℕ := 18

-- Formulating the proof problem
theorem total_attendance (A : ℕ) (C : ℕ) (H1 : C = number_of_child_tickets)
  (H2 : adult_ticket_cost * A + child_ticket_cost * C = total_amount_collected) :
  A + C = 22 := by
  sorry

end total_attendance_l66_66902


namespace exactly_one_negative_x_or_y_l66_66787

theorem exactly_one_negative_x_or_y
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (x1_ne_zero : x1 ≠ 0) (x2_ne_zero : x2 ≠ 0) (x3_ne_zero : x3 ≠ 0)
  (y1_ne_zero : y1 ≠ 0) (y2_ne_zero : y2 ≠ 0) (y3_ne_zero : y3 ≠ 0)
  (h1 : x1 * x2 * x3 = - y1 * y2 * y3)
  (h2 : x1^2 + x2^2 + x3^2 = y1^2 + y2^2 + y3^2)
  (h3 : x1 + y1 + x2 + y2 ≥ x3 + y3 ∧ x2 + y2 + x3 + y3 ≥ x1 + y1 ∧ x3 + y3 + x1 + y1 ≥ x2 + y2)
  (h4 : (x1 + y1)^2 + (x2 + y2)^2 ≥ (x3 + y3)^2 ∧ (x2 + y2)^2 + (x3 + y3)^2 ≥ (x1 + y1)^2 ∧ (x3 + y3)^2 + (x1 + y1)^2 ≥ (x2 + y2)^2) :
  ∃! (a : ℝ), (a = x1 ∨ a = x2 ∨ a = x3 ∨ a = y1 ∨ a = y2 ∨ a = y3) ∧ a < 0 :=
sorry

end exactly_one_negative_x_or_y_l66_66787


namespace minimum_reciprocal_sum_l66_66137

theorem minimum_reciprocal_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : x^2 + y^2 = x * y * (x^2 * y^2 + 2)) : 
  (1 / x + 1 / y) ≥ 2 :=
by 
  sorry -- Proof to be completed

end minimum_reciprocal_sum_l66_66137


namespace conscript_from_western_village_l66_66311

/--
Given:
- The population of the northern village is 8758
- The population of the western village is 7236
- The population of the southern village is 8356
- The total number of conscripts needed is 378

Prove that the number of people to be conscripted from the western village is 112.
-/
theorem conscript_from_western_village (hnorth : ℕ) (hwest : ℕ) (hsouth : ℕ) (hconscripts : ℕ)
    (htotal : hnorth + hwest + hsouth = 24350) :
    let prop := (hwest / (hnorth + hwest + hsouth)) * hconscripts
    hnorth = 8758 → hwest = 7236 → hsouth = 8356 → hconscripts = 378 → prop = 112 :=
by
  intros
  simp_all
  sorry

end conscript_from_western_village_l66_66311


namespace find_k_l66_66785

variables (a b k : ℝ) (x : ℂ)

theorem find_k (h1 : a = 5) (h2 : b = 7)
  (h_roots : ∀ x, x^2 + (↑b / ↑a) * x + (↑k / ↑a) = 0 ↔ x = (↑-b + complex.I * complex.sqrt (171)) / (2 * ↑a)
    ∨ x = (↑-b - complex.I * complex.sqrt (171)) / (2 * ↑a)) :
  k = 11 :=
by
  have h : (7 : ℝ) = ↑7 := by norm_cast
  have h_171 : (171 : ℝ) = ↑171 := by norm_cast
  rw [h171, hsqrt (171 : ℤ)] at h_roots
  sorry

end find_k_l66_66785


namespace max_b_for_integer_solutions_l66_66186

theorem max_b_for_integer_solutions (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
sorry

end max_b_for_integer_solutions_l66_66186


namespace candy_count_after_giving_l66_66238

def numKitKats : ℕ := 5
def numHersheyKisses : ℕ := 3 * numKitKats
def numNerds : ℕ := 8
def numLollipops : ℕ := 11
def numBabyRuths : ℕ := 10
def numReeseCups : ℕ := numBabyRuths / 2
def numLollipopsGivenAway : ℕ := 5

def totalCandyBefore : ℕ := numKitKats + numHersheyKisses + numNerds + numLollipops + numBabyRuths + numReeseCups
def totalCandyAfter : ℕ := totalCandyBefore - numLollipopsGivenAway

theorem candy_count_after_giving : totalCandyAfter = 49 := by
  sorry

end candy_count_after_giving_l66_66238


namespace farmer_potatoes_initial_l66_66224

theorem farmer_potatoes_initial (P : ℕ) (h1 : 175 + P - 172 = 80) : P = 77 :=
by {
  sorry
}

end farmer_potatoes_initial_l66_66224


namespace dice_probability_l66_66858

def is_odd (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

theorem dice_probability :
  (∑ i in {1, 2, 3, 4, 5, 6}.to_finset, if is_odd i then 1 else 0) * 
  (∑ j in {1, 2, 3, 4, 5, 6}.to_finset, if is_prime j then 1 else 0) / 36 = 1 / 4 :=
sorry

end dice_probability_l66_66858


namespace petya_five_ruble_coins_l66_66660

theorem petya_five_ruble_coins (total_coins : ℕ) (not_two_ruble_coins : ℕ) (not_ten_ruble_coins : ℕ) (not_one_ruble_coins : ℕ) 
  (h_total : total_coins = 25) (h_not_two_ruble : not_two_ruble_coins = 19) (h_not_ten_ruble : not_ten_ruble_coins = 20) 
  (h_not_one_ruble : not_one_ruble_coins = 16) : 
  let two_ruble_coins := total_coins - not_two_ruble_coins,
      ten_ruble_coins := total_coins - not_ten_ruble_coins,
      one_ruble_coins := total_coins - not_one_ruble_coins,
      five_ruble_coins := total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins)
  in five_ruble_coins = 5 :=
by {
  have h_two : two_ruble_coins = 6, by { rw [←h_total, ←h_not_two_ruble], exact (25 - 19).symm },
  have h_ten : ten_ruble_coins = 5, by { rw [←h_total, ←h_not_ten_ruble], exact (25 - 20).symm },
  have h_one : one_ruble_coins = 9, by { rw [←h_total, ←h_not_one_ruble], exact (25 - 16).symm },
  have sum_coins : two_ruble_coins + ten_ruble_coins + one_ruble_coins = 20, by { rw [h_two, h_ten, h_one], exact rfl },
  have h_five : five_ruble_coins = total_coins - (two_ruble_coins + ten_ruble_coins + one_ruble_coins), by { exact (25 - 20).symm },
  exact h_five.symm.trans (sum_coins.trans 5),
}

end petya_five_ruble_coins_l66_66660


namespace sara_cakes_sales_l66_66177

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

end sara_cakes_sales_l66_66177


namespace hyperbola_midpoint_l66_66588

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66588


namespace train_speed_is_36_0036_kmph_l66_66234

noncomputable def train_length : ℝ := 130
noncomputable def bridge_length : ℝ := 150
noncomputable def crossing_time : ℝ := 27.997760179185665
noncomputable def speed_in_kmph : ℝ := (train_length + bridge_length) / crossing_time * 3.6

theorem train_speed_is_36_0036_kmph :
  abs (speed_in_kmph - 36.0036) < 0.001 :=
by
  sorry

end train_speed_is_36_0036_kmph_l66_66234


namespace find_door_height_l66_66680

theorem find_door_height :
  ∃ (h : ℝ), 
  let l := 25
  let w := 15
  let H := 12
  let A := 80 * H
  let W := 960 - (6 * h + 36)
  let cost := 4 * W
  cost = 3624 ∧ h = 3 := sorry

end find_door_height_l66_66680


namespace seq_period_3_l66_66199

def seq (a : ℕ → ℚ) := ∀ n, 
  (0 ≤ a n ∧ a n < 1) ∧ (
  (0 ≤ a n ∧ a n < 1/2 → a (n+1) = 2 * a n) ∧ 
  (1/2 ≤ a n ∧ a n < 1 → a (n+1) = 2 * a n - 1))

theorem seq_period_3 (a : ℕ → ℚ) (h : seq a) (h1 : a 1 = 6 / 7) : 
  a 2016 = 3 / 7 := 
sorry

end seq_period_3_l66_66199


namespace spiral_2018_position_l66_66241

def T100_spiral : Matrix ℕ ℕ ℕ := sorry -- Definition of T100 as a spiral matrix

def pos_2018 := (34, 95) -- The given position we need to prove

theorem spiral_2018_position (i j : ℕ) (h₁ : T100_spiral 34 95 = 2018) : (i, j) = pos_2018 := by  
  sorry

end spiral_2018_position_l66_66241


namespace midpoint_of_hyperbola_l66_66478

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66478


namespace sum_even_pos_integers_less_than_100_l66_66062

theorem sum_even_pos_integers_less_than_100 : 
  (∑ i in Finset.filter (λ n, n % 2 = 0) (Finset.range 100), i) = 2450 :=
by
  sorry

end sum_even_pos_integers_less_than_100_l66_66062


namespace root_of_quadratic_l66_66790

theorem root_of_quadratic (b : ℝ) : 
  (-9)^2 + b * (-9) - 45 = 0 -> b = 4 :=
by
  sorry

end root_of_quadratic_l66_66790


namespace smallest_number_divisible_by_618_3648_60_inc_l66_66200

theorem smallest_number_divisible_by_618_3648_60_inc :
  ∃ N : ℕ, (N + 1) % 618 = 0 ∧ (N + 1) % 3648 = 0 ∧ (N + 1) % 60 = 0 ∧ N = 1038239 :=
by
  sorry

end smallest_number_divisible_by_618_3648_60_inc_l66_66200


namespace sin_minus_cos_value_complex_trig_value_l66_66124

noncomputable def sin_cos_equation (x : Real) :=
  -Real.pi / 2 < x ∧ x < Real.pi / 2 ∧ Real.sin x + Real.cos x = -1 / 5

theorem sin_minus_cos_value (x : Real) (h : sin_cos_equation x) :
  Real.sin x - Real.cos x = 7 / 5 :=
sorry

theorem complex_trig_value (x : Real) (h : sin_cos_equation x) :
  (Real.sin (Real.pi + x) + Real.sin (3 * Real.pi / 2 - x)) / 
  (Real.tan (Real.pi - x) + Real.sin (Real.pi / 2 - x)) = 3 / 11 :=
sorry

end sin_minus_cos_value_complex_trig_value_l66_66124


namespace equation_of_circle_O2_equation_of_tangent_line_l66_66278

-- Define circle O1
def circle_O1 (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 4

-- Define the center and radius of circle O2 given that they are externally tangent
def center_O2 : ℝ × ℝ := (3, 3)
def radius_O2 : ℝ := 3

-- Prove the equation of circle O2
theorem equation_of_circle_O2 :
  ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 9 := by
  intro x y
  sorry

-- Prove the equation of the common internal tangent line to circles O1 and O2
theorem equation_of_tangent_line :
  ∀ (x y : ℝ), 3 * x + 4 * y - 21 = 0 := by
  intro x y
  sorry

end equation_of_circle_O2_equation_of_tangent_line_l66_66278


namespace remainder_p_x_minus_2_l66_66264

def p (x : ℝ) := x^5 + 2 * x^2 + 3

theorem remainder_p_x_minus_2 : p 2 = 43 := 
by
  sorry

end remainder_p_x_minus_2_l66_66264


namespace positive_difference_l66_66696

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end positive_difference_l66_66696


namespace yura_picture_dimensions_l66_66375

theorem yura_picture_dimensions (a b : ℕ) (h : (a + 2) * (b + 2) - a * b = a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
by
  -- Place your proof here
  sorry

end yura_picture_dimensions_l66_66375


namespace solve_system_of_equations_l66_66673

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x - y = 2 ∧ 3 * x + y = 4 ∧ x = 1.5 ∧ y = -0.5 :=
by
  sorry

end solve_system_of_equations_l66_66673


namespace midpoint_hyperbola_l66_66542

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66542


namespace smallest_c_for_inverse_l66_66155

def f (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c, (∀ x₁ x₂, (c ≤ x₁ ∧ c ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) ∧
       (∀ d, (∀ x₁ x₂, (d ≤ x₁ ∧ d ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) → c ≤ d) ∧
       c = 3 := sorry

end smallest_c_for_inverse_l66_66155


namespace find_x_l66_66725

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l66_66725


namespace find_number_l66_66717

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l66_66717


namespace hyperbola_midpoint_l66_66601

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66601


namespace midpoint_of_hyperbola_l66_66471

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66471


namespace shopper_saved_percentage_l66_66757

-- Definition of the problem conditions
def amount_saved : ℝ := 4
def amount_spent : ℝ := 36

-- Lean 4 statement to prove the percentage saved
theorem shopper_saved_percentage : (amount_saved / (amount_spent + amount_saved)) * 100 = 10 := by
  sorry

end shopper_saved_percentage_l66_66757


namespace can_be_midpoint_of_AB_l66_66517

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66517


namespace necessary_sufficient_condition_l66_66973

theorem necessary_sufficient_condition (a : ℝ) :
  (∃ x : ℝ, ax^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ a ≤ 1 := sorry

end necessary_sufficient_condition_l66_66973


namespace necessary_but_not_sufficient_condition_l66_66886

theorem necessary_but_not_sufficient_condition (a b c d : ℝ) : 
  (a + b < c + d) → (a < c ∨ b < d) :=
sorry

end necessary_but_not_sufficient_condition_l66_66886


namespace elevator_max_weight_capacity_l66_66208

theorem elevator_max_weight_capacity 
  (num_adults : ℕ)
  (weight_adult : ℕ)
  (num_children : ℕ)
  (weight_child : ℕ)
  (max_next_person_weight : ℕ) 
  (H_adults : num_adults = 3)
  (H_weight_adult : weight_adult = 140)
  (H_children : num_children = 2)
  (H_weight_child : weight_child = 64)
  (H_max_next : max_next_person_weight = 52) : 
  num_adults * weight_adult + num_children * weight_child + max_next_person_weight = 600 := 
by
  sorry

end elevator_max_weight_capacity_l66_66208


namespace five_ruble_coins_count_l66_66650

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l66_66650


namespace probability_ratio_l66_66332

-- Defining the total number of cards and each number's frequency
def total_cards := 60
def each_number_frequency := 4
def distinct_numbers := 15

-- Defining probability p' and q'
def p' := (15: ℕ) * (Nat.choose 4 4) / (Nat.choose 60 4)
def q' := 210 * (Nat.choose 4 3) * (Nat.choose 4 1) / (Nat.choose 60 4)

-- Prove the value of q'/p'
theorem probability_ratio : (q' / p') = 224 := by
  sorry

end probability_ratio_l66_66332


namespace midpoint_of_hyperbola_l66_66473

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66473


namespace average_speed_of_trip_l66_66748

theorem average_speed_of_trip (d1 d2 s1 s2 : ℕ)
  (h1 : d1 = 30) (h2 : d2 = 30)
  (h3 : s1 = 60) (h4 : s2 = 30) :
  (d1 + d2) / (d1 / s1 + d2 / s2) = 40 :=
by sorry

end average_speed_of_trip_l66_66748


namespace strawberry_blueberry_price_difference_l66_66394

theorem strawberry_blueberry_price_difference
  (s p t : ℕ → ℕ)
  (strawberries_sold blueberries_sold strawberries_sale_revenue blueberries_sale_revenue strawberries_loss blueberries_loss : ℕ)
  (h1 : strawberries_sold = 54)
  (h2 : strawberries_sale_revenue = 216)
  (h3 : strawberries_loss = 108)
  (h4 : blueberries_sold = 36)
  (h5 : blueberries_sale_revenue = 144)
  (h6 : blueberries_loss = 72)
  (h7 : p strawberries_sold = strawberries_sale_revenue + strawberries_loss)
  (h8 : p blueberries_sold = blueberries_sale_revenue + blueberries_loss)
  : p strawberries_sold / strawberries_sold - p blueberries_sold / blueberries_sold = 0 :=
by
  sorry

end strawberry_blueberry_price_difference_l66_66394


namespace hunter_time_comparison_l66_66962

-- Definitions for time spent in swamp, forest, and highway
variables {a b c : ℝ}

-- Given conditions
-- 1. Total time equation
#check a + b + c = 4

-- 2. Total distance equation
#check 2 * a + 4 * b + 6 * c = 17

-- Prove that the hunter spent more time on the highway than in the swamp
theorem hunter_time_comparison (h1 : a + b + c = 4) (h2 : 2 * a + 4 * b + 6 * c = 17) : c > a :=
by sorry

end hunter_time_comparison_l66_66962


namespace range_of_f_l66_66431

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 + 6 * x^2 + 9

-- Define the domain as [0, ∞)
def domain (x : ℝ) : Prop := x ≥ 0

-- State the theorem which asserts the range of f(x) is [9, ∞)
theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ y ≥ 9 := by
  sorry

end range_of_f_l66_66431


namespace sqrt_sub_sqrt_frac_eq_l66_66094

theorem sqrt_sub_sqrt_frac_eq : (Real.sqrt 3) - (Real.sqrt (1 / 3)) = (2 * Real.sqrt 3) / 3 := 
by 
  sorry

end sqrt_sub_sqrt_frac_eq_l66_66094


namespace earl_envelope_rate_l66_66418

theorem earl_envelope_rate:
  ∀ (E L : ℝ),
  L = (2/3) * E ∧
  (E + L = 60) →
  E = 36 :=
by
  intros E L h
  sorry

end earl_envelope_rate_l66_66418


namespace evaluate_f_g_f_l66_66158

def f (x: ℝ) : ℝ := 5 * x + 4
def g (x: ℝ) : ℝ := 3 * x + 5

theorem evaluate_f_g_f :
  f (g (f 3)) = 314 :=
by
  sorry

end evaluate_f_g_f_l66_66158


namespace hank_donated_percentage_l66_66931

variable (A_c D_c A_b D_b A_l D_t D_l p : ℝ) (h1 : A_c = 100) (h2 : D_c = 0.90 * A_c)
variable (h3 : A_b = 80) (h4 : D_b = 0.75 * A_b) (h5 : A_l = 50) (h6 : D_t = 200)

theorem hank_donated_percentage :
  D_l = D_t - (D_c + D_b) → 
  p = (D_l / A_l) * 100 → 
  p = 100 :=
by
  sorry

end hank_donated_percentage_l66_66931


namespace expression_is_composite_l66_66170

theorem expression_is_composite (a b : ℕ) : ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 4 * a^2 + 4 * a * b + 4 * a + 2 * b + 1 = m * n := 
by 
  sorry

end expression_is_composite_l66_66170


namespace sandbox_side_length_l66_66083

theorem sandbox_side_length (side_length : ℝ) (sand_sq_inches_per_pound : ℝ := 80 / 30) (total_sand_pounds : ℝ := 600) :
  (side_length ^ 2 = total_sand_pounds * sand_sq_inches_per_pound) → side_length = 40 := 
by
  sorry

end sandbox_side_length_l66_66083


namespace numbers_distance_one_neg_two_l66_66974

theorem numbers_distance_one_neg_two (x : ℝ) (h : abs (x + 2) = 1) : x = -1 ∨ x = -3 := 
sorry

end numbers_distance_one_neg_two_l66_66974


namespace sum_even_positives_less_than_100_l66_66060

theorem sum_even_positives_less_than_100 :
  ∑ k in Finset.Ico 1 50, 2 * k = 2450 :=
by
  sorry

end sum_even_positives_less_than_100_l66_66060


namespace number_of_sodas_in_pack_l66_66092

/-- Billy has twice as many brothers as sisters -/
def twice_as_many_brothers_as_sisters (brothers sisters : ℕ) : Prop :=
  brothers = 2 * sisters

/-- Billy has 2 sisters -/
def billy_has_2_sisters : Prop :=
  ∃ sisters : ℕ, sisters = 2

/-- Billy can give 2 sodas to each of his siblings if he wants to give out the entire pack while giving each sibling the same number of sodas -/
def divide_sodas_evenly (total_sodas siblings sodas_per_sibling : ℕ) : Prop :=
  total_sodas = siblings * sodas_per_sibling

/-- Determine the total number of sodas in the pack given the conditions -/
theorem number_of_sodas_in_pack : 
  ∃ (sisters brothers total_sodas : ℕ), 
    (twice_as_many_brothers_as_sisters brothers sisters) ∧ 
    (billy_has_2_sisters) ∧ 
    (divide_sodas_evenly total_sodas (sisters + brothers + 1) 2) ∧
    (total_sodas = 12) :=
by
  sorry

end number_of_sodas_in_pack_l66_66092


namespace measure_angle_YPZ_is_142_l66_66151

variables (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z]
variables (XM YN ZO : Type) [Inhabited XM] [Inhabited YN] [Inhabited ZO]

noncomputable def angle_XYZ : ℝ := 65
noncomputable def angle_XZY : ℝ := 38
noncomputable def angle_YXZ : ℝ := 180 - angle_XYZ - angle_XZY
noncomputable def angle_YNZ : ℝ := 90 - angle_YXZ
noncomputable def angle_ZMY : ℝ := 90 - angle_XYZ
noncomputable def angle_YPZ : ℝ := 180 - angle_YNZ - angle_ZMY

theorem measure_angle_YPZ_is_142 :
  angle_YPZ = 142 := sorry

end measure_angle_YPZ_is_142_l66_66151


namespace cube_volume_l66_66296

theorem cube_volume (A : ℝ) (V : ℝ) (h : A = 64) : V = 512 :=
by
  sorry

end cube_volume_l66_66296


namespace greatest_b_exists_greatest_b_l66_66189

theorem greatest_b (b x : ℤ) (h1 : x^2 + b * x = -21) (h2 : 0 < b) : b ≤ 22 :=
by
  -- proof would go here
  sorry

theorem exists_greatest_b (b x : ℤ) (h1 : x^2 + b * x = -21) (h2 : 0 < b) : ∃ b', b' = 22 ∧ ∀ b, x^2 + b * x = -21 → 0 < b → b ≤ b' :=
by 
  use 22
  split
  · rfl
  · intros b h_eq h_pos
    apply greatest_b b x h_eq h_pos

end greatest_b_exists_greatest_b_l66_66189


namespace cody_initial_marbles_l66_66410

theorem cody_initial_marbles (x : ℕ) (h1 : x - 5 = 7) : x = 12 := by
  sorry

end cody_initial_marbles_l66_66410


namespace fraction_halfway_between_l66_66046

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66046


namespace sum_of_arithmetic_terms_l66_66351

theorem sum_of_arithmetic_terms (a₁ a₂ a₃ c d a₆ : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₂ = 10)
  (h₃ : a₃ = 17)
  (h₄ : a₆ = 32)
  (h_arith : ∀ n, (a₁ + n * (a₂ - a₁)) = seq)
  : c + d = 55 :=
by
  have d := a₂ - a₁
  have c := a₃ + d
  have d := c + d
  have h_seq := list.map (λ n, (a₁ + n * d)) (list.range 6) -- Making use of the arithmetic property
  have h_seq_eq := h_seq = [3, 10, 17, c, d, 32]
  sorry

end sum_of_arithmetic_terms_l66_66351


namespace at_least_one_alarm_rings_on_time_l66_66827

-- Definitions for the problem
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6

def prob_not_A : ℝ := 1 - prob_A
def prob_not_B : ℝ := 1 - prob_B
def prob_neither_A_nor_B : ℝ := prob_not_A * prob_not_B
def prob_at_least_one : ℝ := 1 - prob_neither_A_nor_B

-- Final statement
theorem at_least_one_alarm_rings_on_time : prob_at_least_one = 0.8 :=
by sorry

end at_least_one_alarm_rings_on_time_l66_66827


namespace smallest_number_of_students_l66_66456

theorem smallest_number_of_students 
  (n : ℕ) 
  (h1 : 4 * 80 + (n - 4) * 50 ≤ 65 * n) :
  n = 8 :=
by sorry

end smallest_number_of_students_l66_66456


namespace multiply_inequalities_positive_multiply_inequalities_negative_l66_66067

variable {a b c d : ℝ}

theorem multiply_inequalities_positive (h₁ : a > b) (h₂ : c > d) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) (h₆ : 0 < d) :
  a * c > b * d :=
sorry

theorem multiply_inequalities_negative (h₁ : a < b) (h₂ : c < d) (h₃ : a < 0) (h₄ : b < 0) (h₅ : c < 0) (h₆ : d < 0) :
  a * c > b * d :=
sorry

end multiply_inequalities_positive_multiply_inequalities_negative_l66_66067


namespace solutions_count_l66_66932

noncomputable def number_of_solutions (a : ℝ) : ℕ :=
if a < 0 then 1
else if 0 ≤ a ∧ a < Real.exp 1 then 0
else if a = Real.exp 1 then 1
else if a > Real.exp 1 then 2
else 0

theorem solutions_count (a : ℝ) :
  (a < 0 ∧ number_of_solutions a = 1) ∨
  (0 ≤ a ∧ a < Real.exp 1 ∧ number_of_solutions a = 0) ∨
  (a = Real.exp 1 ∧ number_of_solutions a = 1) ∨
  (a > Real.exp 1 ∧ number_of_solutions a = 2) :=
by {
  sorry
}

end solutions_count_l66_66932


namespace no_real_solution_ratio_l66_66346

theorem no_real_solution_ratio (x : ℝ) : (x + 3) / (2 * x + 5) = (5 * x + 4) / (8 * x + 5) → false :=
by {
  sorry
}

end no_real_solution_ratio_l66_66346


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66035

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66035


namespace num_five_ruble_coins_l66_66665

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l66_66665


namespace max_value_is_two_over_three_l66_66422

noncomputable def max_value_expr (x : ℝ) : ℝ := 2^x - 8^x

theorem max_value_is_two_over_three :
  ∃ (x : ℝ), max_value_expr x = 2 / 3 :=
sorry

end max_value_is_two_over_three_l66_66422


namespace original_team_members_l66_66899

theorem original_team_members (m p total_points : ℕ) (h_m : m = 3) (h_p : p = 2) (h_total : total_points = 12) :
  (total_points / p) + m = 9 := by
  sorry

end original_team_members_l66_66899


namespace total_flowers_tuesday_l66_66164

def ginger_flower_shop (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) := 
  let lilacs_tuesday := lilacs_monday + lilacs_monday * 5 / 100
  let roses_tuesday := roses_monday - roses_monday * 4 / 100
  let tulips_tuesday := tulips_monday - tulips_monday * 7 / 100
  let gardenias_tuesday := gardenias_monday
  let orchids_tuesday := orchids_monday
  lilacs_tuesday + roses_tuesday + tulips_tuesday + gardenias_tuesday + orchids_tuesday

theorem total_flowers_tuesday (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) 
  (h1: lilacs_monday = 15)
  (h2: roses_monday = 3 * lilacs_monday)
  (h3: gardenias_monday = lilacs_monday / 2)
  (h4: tulips_monday = 2 * (roses_monday + gardenias_monday))
  (h5: orchids_monday = (roses_monday + gardenias_monday + tulips_monday) / 3):
  ginger_flower_shop lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday = 214 :=
by
  sorry

end total_flowers_tuesday_l66_66164


namespace can_be_midpoint_of_AB_l66_66511

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66511


namespace num_students_third_class_num_students_second_class_l66_66879

-- Definition of conditions for both problems
def class_student_bounds (n : ℕ) : Prop := 40 < n ∧ n ≤ 50
def option_one_cost (n : ℕ) : ℕ := 40 * n * 7 / 10
def option_two_cost (n : ℕ) : ℕ := 40 * (n - 6) * 8 / 10

-- Problem Part 1
theorem num_students_third_class (x : ℕ) (h1 : class_student_bounds x) (h2 : option_one_cost x = option_two_cost x) : x = 48 := 
sorry

-- Problem Part 2
theorem num_students_second_class (y : ℕ) (h1 : class_student_bounds y) (h2 : option_one_cost y < option_two_cost y) : y = 49 ∨ y = 50 := 
sorry

end num_students_third_class_num_students_second_class_l66_66879


namespace halfway_fraction_l66_66028

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66028


namespace glenda_speed_is_8_l66_66765

noncomputable def GlendaSpeed : ℝ :=
  let AnnSpeed := 6
  let Hours := 3
  let Distance := 42
  let AnnDistance := AnnSpeed * Hours
  let GlendaDistance := Distance - AnnDistance
  GlendaDistance / Hours

theorem glenda_speed_is_8 : GlendaSpeed = 8 := by
  sorry

end glenda_speed_is_8_l66_66765


namespace hyperbola_midpoint_exists_l66_66555

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66555


namespace f_2011_is_zero_l66_66284

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f (x) + f (1)

-- Theorem stating the mathematically equivalent proof problem
theorem f_2011_is_zero : f (2011) = 0 :=
sorry

end f_2011_is_zero_l66_66284


namespace range_of_x_l66_66791

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
    Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4 :=
by
  sorry

end range_of_x_l66_66791


namespace necessary_and_sufficient_condition_l66_66202

theorem necessary_and_sufficient_condition (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
    (∃ x : ℝ, 0 < x ∧ a^x = 2) ↔ (1 < a) := 
sorry

end necessary_and_sufficient_condition_l66_66202


namespace greatest_possible_b_l66_66191

theorem greatest_possible_b (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
by sorry

end greatest_possible_b_l66_66191


namespace simplify_expression_l66_66671

open Real

theorem simplify_expression :
    (3 * (sqrt 5 + sqrt 7) / (4 * sqrt (3 + sqrt 5))) = sqrt (414 - 98 * sqrt 35) / 8 :=
by
  sorry

end simplify_expression_l66_66671


namespace find_xyz_value_l66_66294

noncomputable def xyz_satisfying_conditions (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (x + 1/y = 5) ∧
  (y + 1/z = 2) ∧
  (z + 1/x = 3)

theorem find_xyz_value (x y z : ℝ) (h : xyz_satisfying_conditions x y z) : x * y * z = 1 :=
by
  sorry

end find_xyz_value_l66_66294


namespace unique_not_in_range_l66_66772

open Real

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 10 = 10) (h₆ : f a b c d 50 = 50) 
  (h₇ : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) :
  ∃! x, ¬ ∃ y, f a b c d y = x :=
  sorry

end unique_not_in_range_l66_66772


namespace problem_statement_l66_66217

-- Define the operation #
def op_hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The main theorem statement
theorem problem_statement (a b : ℕ) (h1 : op_hash a b = 100) : (a + b) + 6 = 11 := 
sorry

end problem_statement_l66_66217


namespace jars_proof_l66_66467

def total_plums : ℕ := 240
def exchange_ratio : ℕ := 7
def mangoes_per_jar : ℕ := 5

def ripe_plums (total_plums : ℕ) := total_plums / 4
def unripe_plums (total_plums : ℕ) := 3 * total_plums / 4
def unripe_plums_kept : ℕ := 46

def plums_for_trade (total_plums unripe_plums_kept : ℕ) : ℕ :=
  ripe_plums total_plums + (unripe_plums total_plums - unripe_plums_kept)

def mangoes_received (plums_for_trade exchange_ratio : ℕ) : ℕ :=
  plums_for_trade / exchange_ratio

def jars_of_mangoes (mangoes_received mangoes_per_jar : ℕ) : ℕ :=
  mangoes_received / mangoes_per_jar

theorem jars_proof : jars_of_mangoes (mangoes_received (plums_for_trade total_plums unripe_plums_kept) exchange_ratio) mangoes_per_jar = 5 :=
by
  sorry

end jars_proof_l66_66467


namespace fraction_half_way_l66_66019

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66019


namespace hyperbola_midpoint_l66_66497

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66497


namespace midpoint_on_hyperbola_l66_66597

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_on_hyperbola
  (A B : ℝ × ℝ)
  (H_A : point_on_hyperbola A.1 A.2)
  (H_B : point_on_hyperbola B.1 B.2) :
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (-1, -4)) :=
sorry

end midpoint_on_hyperbola_l66_66597


namespace twelve_pow_six_mod_nine_eq_zero_l66_66181

theorem twelve_pow_six_mod_nine_eq_zero : (∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 12^6 ≡ n [MOD 9]) → 12^6 ≡ 0 [MOD 9] :=
by
  sorry

end twelve_pow_six_mod_nine_eq_zero_l66_66181


namespace percentage_fullness_before_storms_l66_66237

def capacity : ℕ := 200 -- capacity in billion gallons
def water_added_by_storms : ℕ := 15 + 30 + 75 -- total water added by storms in billion gallons
def percentage_after : ℕ := 80 -- percentage of fullness after storms
def amount_of_water_after_storms : ℕ := capacity * percentage_after / 100

theorem percentage_fullness_before_storms :
  (amount_of_water_after_storms - water_added_by_storms) * 100 / capacity = 20 := by
  sorry

end percentage_fullness_before_storms_l66_66237


namespace min_union_of_subsets_l66_66157

open Finset

variables {A : Finset ℕ} (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11: Finset ℕ)

noncomputable def min_union_size (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 : Finset ℕ) : ℕ :=
  (A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5 ∪ A6 ∪ A7 ∪ A8 ∪ A9 ∪ A10 ∪ A11).card

theorem min_union_of_subsets :
  (∃ (A : Finset ℕ) (hA : A.card = 225),
    (∀ i ∈ {A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11}, (i.card = 45)) ∧
    (∀ i j ∈ ({A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11} : Finset (Finset ℕ)), i ≠ j → (i ∩ j).card = 9)) →
  min_union_size A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 ≥ 165 := 
sorry

end min_union_of_subsets_l66_66157


namespace equal_share_of_tea_l66_66769

def totalCups : ℕ := 10
def totalPeople : ℕ := 5
def cupsPerPerson : ℕ := totalCups / totalPeople

theorem equal_share_of_tea : cupsPerPerson = 2 := by
  sorry

end equal_share_of_tea_l66_66769


namespace abigail_spent_in_store_l66_66235

theorem abigail_spent_in_store (initial_amount : ℕ) (amount_left : ℕ) (amount_lost : ℕ) (spent : ℕ) 
  (h1 : initial_amount = 11) 
  (h2 : amount_left = 3)
  (h3 : amount_lost = 6) :
  spent = initial_amount - (amount_left + amount_lost) :=
by
  sorry

end abigail_spent_in_store_l66_66235


namespace black_more_than_blue_l66_66940

noncomputable def number_of_pencils := 8
noncomputable def number_of_blue_pens := 2 * number_of_pencils
noncomputable def number_of_red_pens := number_of_pencils - 2
noncomputable def total_pens := 48

-- Given the conditions
def satisfies_conditions (K B P : ℕ) : Prop :=
  P = number_of_pencils ∧
  B = number_of_blue_pens ∧
  K + B + number_of_red_pens = total_pens

-- Prove the number of more black pens than blue pens
theorem black_more_than_blue (K B P : ℕ) : satisfies_conditions K B P → (K - B) = 10 := by
  sorry

end black_more_than_blue_l66_66940


namespace cos_B_plus_C_value_of_c_l66_66301

variable {A B C a b c : ℝ}

-- Given conditions
axiom a_eq_2b : a = 2 * b
axiom sine_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B

-- First proof
theorem cos_B_plus_C (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) :
  Real.cos (B + C) = 1 / 4 := 
sorry

-- Given additional condition for the area
axiom area_eq : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3

-- Second proof
theorem value_of_c (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) (h_area : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3) :
  c = 4 * Real.sqrt 2 :=
sorry

end cos_B_plus_C_value_of_c_l66_66301


namespace sum_even_integers_less_than_100_l66_66064

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l66_66064


namespace hyperbola_midpoint_exists_l66_66554

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66554


namespace sequence_a_2016_value_l66_66149

theorem sequence_a_2016_value (a : ℕ → ℕ) 
  (h1 : a 4 = 1)
  (h2 : a 11 = 9)
  (h3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15) :
  a 2016 = 5 :=
sorry

end sequence_a_2016_value_l66_66149


namespace porter_l66_66328

def previous_sale_amount : ℕ := 9000

def recent_sale_price (previous_sale_amount : ℕ) : ℕ :=
  5 * previous_sale_amount - 1000

theorem porter's_recent_sale : recent_sale_price previous_sale_amount = 44000 :=
by
  sorry

end porter_l66_66328


namespace halfway_fraction_l66_66008

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66008


namespace midpoint_hyperbola_l66_66539

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66539


namespace poly_has_int_solution_iff_l66_66783

theorem poly_has_int_solution_iff (a : ℤ) : 
  (a > 0 ∧ (∃ x : ℤ, a * x^2 + 2 * (2 * a - 1) * x + 4 * a - 7 = 0)) ↔ (a = 1 ∨ a = 5) :=
by {
  sorry
}

end poly_has_int_solution_iff_l66_66783


namespace find_m_l66_66923

theorem find_m : ∃ m : ℤ, 2^5 - 7 = 3^3 + m ∧ m = -2 :=
by
  use -2
  sorry

end find_m_l66_66923


namespace number_divided_by_three_l66_66722

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l66_66722


namespace sum_of_numbers_l66_66140

variable (x y : ℝ)

def condition1 := 0.45 * x = 2700
def condition2 := y = 2 * x

theorem sum_of_numbers (h1 : condition1 x) (h2 : condition2 x y) : x + y = 18000 :=
by {
  sorry
}

end sum_of_numbers_l66_66140


namespace correct_midpoint_l66_66482

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66482


namespace hyperbola_midpoint_l66_66495

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66495


namespace positive_difference_l66_66695

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end positive_difference_l66_66695


namespace total_birds_in_marsh_l66_66979

def number_of_geese : Nat := 58
def number_of_ducks : Nat := 37

theorem total_birds_in_marsh :
  number_of_geese + number_of_ducks = 95 :=
sorry

end total_birds_in_marsh_l66_66979


namespace hyperbola_midpoint_exists_l66_66551

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66551


namespace fraction_halfway_between_l66_66047

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66047


namespace collinear_vectors_x_value_l66_66132

theorem collinear_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) (h₁: a = (2, x)) (h₂: b = (1, 2))
  (h₃: ∃ k : ℝ, a = k • b) : x = 4 :=
by
  sorry

end collinear_vectors_x_value_l66_66132


namespace find_larger_integer_l66_66692

noncomputable def larger_integer (a b : ℕ) : Prop :=
  a * b = 189 ∧ (b = (7 * a) / 3⁷) / 3

theorem find_larger_integer (a b : ℕ) (h1 : a * b = 189) (h2 : a * 7 = 3 * b) :
  b = 21 :=
by
  sorry

end find_larger_integer_l66_66692


namespace tammy_laps_per_day_l66_66841

theorem tammy_laps_per_day :
  ∀ (total_distance_per_week distance_per_lap days_in_week : ℕ), 
  total_distance_per_week = 3500 → 
  distance_per_lap = 50 → 
  days_in_week = 7 → 
  (total_distance_per_week / distance_per_lap) / days_in_week = 10 :=
by
  intros total_distance_per_week distance_per_lap days_in_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end tammy_laps_per_day_l66_66841


namespace square_side_length_l66_66845

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 0 ∨ x = 2 := 
by
suffices h' : x^2 - 2 * x = 0, from sorry,
calc
  4 * x = 2 * x^2  : h
  ... = x^2 - 2 * x : sorry

end square_side_length_l66_66845


namespace range_of_a_l66_66436

theorem range_of_a (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^2 + b * x + 1)
  (h2 : f (-1) = 1) (h3 : ∀ x, f x < 2) : -4 < a ∧ a ≤ 0 :=
by
  -- Restating conditions
  have b_eq_a : b = a, from sorry,
  have f_neg1_eq1 : a - b + 1 = 1, from sorry,
  have f_condition : ∀ x, a * x^2 + a * x - 1 < 0, from sorry,
  -- Analyzing the values of 'a'
  have case_a_zero : a = 0 ∨ a ≠ 0, from sorry,
  have discriminant_condition : a^2 + 4 * a < 0, from sorry,
  have range_of_a_cases : -4 < a ∧ a ≤ 0, from sorry,
  exact range_of_a_cases

end range_of_a_l66_66436


namespace number_divided_by_three_l66_66721

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l66_66721


namespace intersection_P_Q_eq_Q_l66_66825

def P : Set ℝ := { x | x < 2 }
def Q : Set ℝ := { x | x^2 ≤ 1 }

theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
sorry

end intersection_P_Q_eq_Q_l66_66825


namespace correct_midpoint_l66_66481

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66481


namespace midpoint_hyperbola_l66_66541

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66541


namespace modulo_remainder_product_l66_66864

theorem modulo_remainder_product :
  let a := 2022
  let b := 2023
  let c := 2024
  let d := 2025
  let n := 17
  (a * b * c * d) % n = 0 :=
by
  sorry

end modulo_remainder_product_l66_66864


namespace number_of_slices_with_both_l66_66381

def total_slices : ℕ := 20
def slices_with_pepperoni : ℕ := 12
def slices_with_mushrooms : ℕ := 14
def slices_with_both_toppings (n : ℕ) : Prop :=
  n + (slices_with_pepperoni - n) + (slices_with_mushrooms - n) = total_slices

theorem number_of_slices_with_both (n : ℕ) (h : slices_with_both_toppings n) : n = 6 :=
sorry

end number_of_slices_with_both_l66_66381


namespace snooker_tournament_total_cost_l66_66896

def VIP_cost : ℝ := 45
def GA_cost : ℝ := 20
def total_tickets_sold : ℝ := 320
def vip_and_general_admission_relationship := 276

def total_cost_of_tickets : ℝ := 6950

theorem snooker_tournament_total_cost 
  (V G : ℝ)
  (h1 : VIP_cost * V + GA_cost * G = total_cost_of_tickets)
  (h2 : V + G = total_tickets_sold)
  (h3 : V = G - vip_and_general_admission_relationship) : 
  VIP_cost * V + GA_cost * G = total_cost_of_tickets := 
by {
  sorry
}

end snooker_tournament_total_cost_l66_66896


namespace manufacturing_sector_angle_l66_66336

theorem manufacturing_sector_angle (h1 : 50 ≤ 100) (h2 : 360 = 4 * 90) : 0.50 * 360 = 180 := 
by
  sorry

end manufacturing_sector_angle_l66_66336


namespace distance_between_points_l66_66107

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l66_66107


namespace equivalence_of_expression_l66_66682

theorem equivalence_of_expression (x y : ℝ) :
  ( (x^2 + y^2 + xy) / (x^2 + y^2 - xy) ) - ( (x^2 + y^2 - xy) / (x^2 + y^2 + xy) ) =
  ( 4 * xy * (x^2 + y^2) ) / ( x^4 + y^4 ) :=
by sorry

end equivalence_of_expression_l66_66682


namespace shifted_roots_polynomial_l66_66822

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ :=
  x^3 - 5 * x + 7

-- Define the shifted polynomial
def shifted_polynomial (x : ℝ) : ℝ :=
  x^3 + 9 * x^2 + 22 * x + 19

-- Define the roots condition
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop :=
  p r = 0

-- State the theorem
theorem shifted_roots_polynomial :
  ∀ a b c : ℝ,
    is_root original_polynomial a →
    is_root original_polynomial b →
    is_root original_polynomial c →
    is_root shifted_polynomial (a - 3) ∧
    is_root shifted_polynomial (b - 3) ∧
    is_root shifted_polynomial (c - 3) :=
by
  intros a b c ha hb hc
  sorry

end shifted_roots_polynomial_l66_66822


namespace find_n_l66_66112

theorem find_n (a b c : ℤ) (m n p : ℕ)
  (h1 : a = 3)
  (h2 : b = -7)
  (h3 : c = -6)
  (h4 : m > 0)
  (h5 : n > 0)
  (h6 : p > 0)
  (h7 : Nat.gcd m p = 1)
  (h8 : Nat.gcd m n = 1)
  (h9 : Nat.gcd n p = 1)
  (h10 : ∃ x1 x2 : ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p)
  : n = 121 :=
sorry

end find_n_l66_66112


namespace steve_height_after_growth_l66_66675

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l66_66675


namespace cuboid_layers_l66_66890

theorem cuboid_layers (V : ℕ) (n_blocks : ℕ) (volume_per_block : ℕ) (blocks_per_layer : ℕ)
  (hV : V = 252) (hvol : volume_per_block = 1) (hblocks : n_blocks = V / volume_per_block) (hlayer : blocks_per_layer = 36) :
  (n_blocks / blocks_per_layer) = 7 :=
by
  sorry

end cuboid_layers_l66_66890


namespace fraction_half_way_l66_66020

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66020


namespace expected_heads_of_fair_coin_l66_66211

noncomputable def expected_heads (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expected_heads_of_fair_coin :
  expected_heads 5 0.5 = 2.5 :=
by
  sorry

end expected_heads_of_fair_coin_l66_66211


namespace time_to_cover_length_l66_66906

def escalator_speed : ℝ := 8  -- The speed of the escalator in feet per second
def person_speed : ℝ := 2     -- The speed of the person in feet per second
def escalator_length : ℝ := 160 -- The length of the escalator in feet

theorem time_to_cover_length : 
  (escalator_length / (escalator_speed + person_speed) = 16) :=
by 
  sorry

end time_to_cover_length_l66_66906


namespace problem1_problem2_l66_66816

noncomputable def f (x a c : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

-- Problem 1: Prove that for c = 19, the inequality f(1, a, 19) > 0 holds for -2 < a < 8
theorem problem1 (a : ℝ) : f 1 a 19 > 0 ↔ -2 < a ∧ a < 8 :=
by sorry

-- Problem 2: Given that f(x) > 0 has solution set (-1, 3), find a and c
theorem problem2 (a c : ℝ) (hx : ∀ x, -1 < x ∧ x < 3 → f x a c > 0) : 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9 :=
by sorry

end problem1_problem2_l66_66816


namespace problem_solution_eq_l66_66912

theorem problem_solution_eq : 
  { x : ℝ | (x ^ 2 - 9) / (x ^ 2 - 1) > 0 } = { x : ℝ | x > 3 ∨ x < -3 } :=
by
  sorry

end problem_solution_eq_l66_66912


namespace jumps_correct_l66_66329

def R : ℕ := 157
def X : ℕ := 86
def total_jumps (R X : ℕ) : ℕ := R + (R + X)

theorem jumps_correct : total_jumps R X = 400 := by
  sorry

end jumps_correct_l66_66329


namespace midpoint_on_hyperbola_l66_66529

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66529


namespace correct_midpoint_l66_66487

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66487


namespace simplify_and_evaluate_l66_66331

theorem simplify_and_evaluate (x : Real) (h : x = Real.sqrt 2 - 1) :
  ( (1 / (x - 1) - 1 / (x + 1)) / (2 / (x - 1) ^ 2) ) = 1 - Real.sqrt 2 :=
by
  subst h
  sorry

end simplify_and_evaluate_l66_66331


namespace total_bags_l66_66239

theorem total_bags (people : ℕ) (bags_per_person : ℕ) (h_people : people = 4) (h_bags_per_person : bags_per_person = 8) : people * bags_per_person = 32 := by
  sorry

end total_bags_l66_66239


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66630

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l66_66630


namespace find_ab_l66_66281

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end find_ab_l66_66281


namespace Tammy_runs_10_laps_per_day_l66_66840

theorem Tammy_runs_10_laps_per_day
  (total_distance_per_week : ℕ)
  (track_length : ℕ)
  (days_per_week : ℕ)
  (h1 : total_distance_per_week = 3500)
  (h2 : track_length = 50)
  (h3 : days_per_week = 7) :
  (total_distance_per_week / track_length) / days_per_week = 10 := by
  sorry

end Tammy_runs_10_laps_per_day_l66_66840


namespace halfway_fraction_l66_66006

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66006


namespace calculate_total_weight_l66_66417

variable (a b c d : ℝ)

-- Conditions
def I_II_weight := a + b = 156
def III_IV_weight := c + d = 195
def I_III_weight := a + c = 174
def II_IV_weight := b + d = 186

theorem calculate_total_weight (I_II_weight : a + b = 156) (III_IV_weight : c + d = 195)
    (I_III_weight : a + c = 174) (II_IV_weight : b + d = 186) :
    a + b + c + d = 355.5 :=
by
    sorry

end calculate_total_weight_l66_66417


namespace time_for_model_M_l66_66384

variable (T : ℝ) -- Time taken by model M computer to complete the task in minutes.
variable (n_m : ℝ := 12) -- Number of model M computers
variable (n_n : ℝ := 12) -- Number of model N computers
variable (time_n : ℝ := 18) -- Time taken by model N computer to complete the task in minutes

theorem time_for_model_M :
  n_m / T + n_n / time_n = 1 → T = 36 := by
sorry

end time_for_model_M_l66_66384


namespace midpoint_of_hyperbola_l66_66577

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66577


namespace hyperbola_midpoint_l66_66610

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66610


namespace track_length_l66_66376

theorem track_length (L : ℝ)
  (h_brenda_first_meeting : ∃ (brenda_run1: ℝ), brenda_run1 = 100)
  (h_sally_first_meeting : ∃ (sally_run1: ℝ), sally_run1 = L/2 - 100)
  (h_brenda_second_meeting : ∃ (brenda_run2: ℝ), brenda_run2 = L - 100)
  (h_sally_second_meeting : ∃ (sally_run2: ℝ), sally_run2 = sally_run1 + 100)
  (h_meeting_total : brenda_run2 + sally_run2 = L) :
  L = 200 :=
by
  sorry

end track_length_l66_66376


namespace midpoint_of_hyperbola_segment_l66_66506

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66506


namespace find_mangoes_l66_66093

def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 1165
def cost_per_kg_of_mangoes : ℕ := 55

theorem find_mangoes (m : ℕ) : cost_of_grapes + m * cost_per_kg_of_mangoes = total_amount_paid → m = 11 :=
by
  sorry

end find_mangoes_l66_66093


namespace pencils_purchased_l66_66875

theorem pencils_purchased (total_cost : ℝ) (num_pens : ℕ) (pen_price : ℝ) (pencil_price : ℝ) (num_pencils : ℕ) : 
  total_cost = (num_pens * pen_price) + (num_pencils * pencil_price) → 
  num_pens = 30 → 
  pen_price = 20 → 
  pencil_price = 2 → 
  total_cost = 750 →
  num_pencils = 75 :=
by
  sorry

end pencils_purchased_l66_66875


namespace midpoint_hyperbola_l66_66548

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66548


namespace abc_value_l66_66820

theorem abc_value (a b c : ℂ) (h1 : 2 * a * b + 3 * b = -21)
                   (h2 : 2 * b * c + 3 * c = -21)
                   (h3 : 2 * c * a + 3 * a = -21) :
                   a * b * c = 105.75 := 
sorry

end abc_value_l66_66820


namespace find_a_squared_plus_b_squared_and_ab_l66_66427

theorem find_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 7)
  (h2 : (a - b) ^ 2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by 
  sorry

end find_a_squared_plus_b_squared_and_ab_l66_66427


namespace number_of_five_ruble_coins_l66_66655

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l66_66655


namespace part_a_total_time_part_b_average_time_part_c_probability_l66_66637

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l66_66637


namespace Danny_finishes_first_l66_66917

-- Definitions based on the conditions
variables (E D F : ℝ)    -- Garden areas for Emily, Danny, Fiona
variables (e d f : ℝ)    -- Mowing rates for Emily, Danny, Fiona
variables (start_time : ℝ)

-- Condition definitions
def emily_garden_size := E = 3 * D
def emily_garden_size_fiona := E = 5 * F
def fiona_mower_speed_danny := f = (1/4) * d
def fiona_mower_speed_emily := f = (1/5) * e

-- Prove Danny finishes first
theorem Danny_finishes_first 
  (h1 : emily_garden_size E D)
  (h2 : emily_garden_size_fiona E F)
  (h3 : fiona_mower_speed_danny f d)
  (h4 : fiona_mower_speed_emily f e) : 
  (start_time ≤ (5/12) * (start_time + E/d) ∧ start_time ≤ (E/f)) -> (start_time + E/d < start_time + E/e) -> 
  true := 
sorry -- proof is omitted

end Danny_finishes_first_l66_66917


namespace fraction_halfway_between_l66_66042

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66042


namespace hyperbola_midpoint_l66_66498

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66498


namespace midpoint_on_hyperbola_l66_66561

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66561


namespace halfway_fraction_l66_66024

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66024


namespace oldest_sibling_multiple_l66_66468

-- Definitions according to the conditions
def kay_age : Nat := 32
def youngest_sibling_age : Nat := kay_age / 2 - 5
def oldest_sibling_age : Nat := 44

-- The statement to prove
theorem oldest_sibling_multiple : oldest_sibling_age = 4 * youngest_sibling_age :=
by sorry

end oldest_sibling_multiple_l66_66468


namespace target_runs_is_282_l66_66814

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_segment : ℝ := 10
def run_rate_remaining_20_overs : ℝ := 12.5
def overs_second_segment : ℝ := 20

-- Define the calculation of runs in the first 10 overs
def runs_first_segment : ℝ := run_rate_first_10_overs * overs_first_segment

-- Define the calculation of runs in the remaining 20 overs
def runs_second_segment : ℝ := run_rate_remaining_20_overs * overs_second_segment

-- Define the target runs
def target_runs : ℝ := runs_first_segment + runs_second_segment

-- State the theorem
theorem target_runs_is_282 : target_runs = 282 :=
by
  -- This is where the proof would go, but it is omitted.
  sorry

end target_runs_is_282_l66_66814


namespace ratio_wrong_to_correct_l66_66759

theorem ratio_wrong_to_correct (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36) (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 :=
by {
  -- Proof will go here
  sorry
}

end ratio_wrong_to_correct_l66_66759


namespace halfway_fraction_l66_66025

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66025


namespace intersection_is_correct_l66_66131

noncomputable def M : Set ℝ := { x | 1 + x ≥ 0 }
noncomputable def N : Set ℝ := { x | 4 / (1 - x) > 0 }
noncomputable def intersection : Set ℝ := { x | -1 ≤ x ∧ x < 1 }

theorem intersection_is_correct : M ∩ N = intersection := by
  sorry

end intersection_is_correct_l66_66131


namespace circle_tangent_to_line_iff_m_eq_zero_l66_66450

theorem circle_tangent_to_line_iff_m_eq_zero (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m^2 ∧ x - y = m) ↔ m = 0 :=
by 
  sorry

end circle_tangent_to_line_iff_m_eq_zero_l66_66450


namespace max_beds_120_l66_66388

/-- The dimensions of the park. --/
def park_length : ℕ := 60
def park_width : ℕ := 30

/-- The dimensions of each flower bed. --/
def bed_length : ℕ := 3
def bed_width : ℕ := 5

/-- The available fencing length. --/
def total_fencing : ℕ := 2400

/-- Calculate the largest number of flower beds that can be created. --/
def max_flower_beds (park_length park_width bed_length bed_width total_fencing : ℕ) : ℕ := 
  let n := park_width / bed_width  -- number of beds per column
  let m := park_length / bed_length  -- number of beds per row
  let vertical_fencing := bed_width * (n - 1) * m
  let horizontal_fencing := bed_length * (m - 1) * n
  if vertical_fencing + horizontal_fencing <= total_fencing then n * m else 0

theorem max_beds_120 : max_flower_beds 60 30 3 5 2400 = 120 := by
  unfold max_flower_beds
  rfl

end max_beds_120_l66_66388


namespace number_of_boys_l66_66218

theorem number_of_boys (x : ℕ) (boys girls : ℕ)
  (initialRatio : girls / boys = 5 / 6)
  (afterLeavingRatio : (girls - 20) / boys = 2 / 3) :
  boys = 120 := by
  -- Proof is omitted
  sorry

end number_of_boys_l66_66218


namespace seats_taken_correct_l66_66267

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end seats_taken_correct_l66_66267


namespace SWE4_l66_66380

theorem SWE4 (a : ℕ → ℕ) (n : ℕ) :
  a 0 = 0 →
  (∀ n, a (n + 1) = 2 * a n + 2^n) →
  (∃ k : ℕ, n = 2^k) →
  ∃ m : ℕ, a n = 2^m :=
by
  intros h₀ h_recurrence h_power
  sorry

end SWE4_l66_66380


namespace halfway_fraction_l66_66029

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66029


namespace find_f_600_l66_66317

variable (f : ℝ → ℝ)
variable (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
variable (h2 : f 500 = 3)

theorem find_f_600 : f 600 = 5 / 2 :=
by
  sorry

end find_f_600_l66_66317


namespace factorize_3x2_minus_3y2_l66_66258

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorize_3x2_minus_3y2_l66_66258


namespace find_a_of_square_roots_l66_66201

theorem find_a_of_square_roots (a : ℤ) (n : ℤ) (h₁ : 2 * a + 1 = n) (h₂ : a + 5 = n) : a = 4 :=
by
  -- proof goes here
  sorry

end find_a_of_square_roots_l66_66201


namespace fraction_is_one_fifth_l66_66919

theorem fraction_is_one_fifth
  (x a b : ℤ)
  (hx : x^2 = 25)
  (h2x : 2 * x = a * x / b + 9) :
  a = 1 ∧ b = 5 :=
by
  sorry

end fraction_is_one_fifth_l66_66919


namespace families_received_boxes_l66_66271

theorem families_received_boxes (F : ℕ) (box_decorations total_decorations : ℕ)
  (h_box_decorations : box_decorations = 10)
  (h_total_decorations : total_decorations = 120)
  (h_eq : box_decorations * (F + 1) = total_decorations) :
  F = 11 :=
by
  sorry

end families_received_boxes_l66_66271


namespace tangent_line_l66_66795

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 1 / x

theorem tangent_line (x y : ℝ) (h_inter : y = f x ∧ y = g x) :
  (x - 2 * y + 1 = 0) :=
by
  sorry

end tangent_line_l66_66795


namespace number_of_five_ruble_coins_l66_66656

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l66_66656


namespace three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l66_66373

theorem three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693 :
  ∃ (n : ℕ), n = 693 ∧ 
    (100 * 6 + 10 * (n / 10 % 10) + 3) = n ∧
    (n % 10 = 3) ∧
    (n / 100 = 6) ∧
    n % 11 = 0 :=
by
  sorry

end three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l66_66373


namespace imaginary_part_is_neg_two_l66_66685

open Complex

noncomputable def imaginary_part_of_square : ℂ := (1 - I)^2

theorem imaginary_part_is_neg_two : imaginary_part_of_square.im = -2 := by
  sorry

end imaginary_part_is_neg_two_l66_66685


namespace max_d_n_l66_66195

open Int

def a_n (n : ℕ) : ℤ := 80 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (a_n n) (a_n (n + 1))

theorem max_d_n : ∃ n : ℕ, d_n n = 5 ∧ ∀ m : ℕ, d_n m ≤ 5 := by
  sorry

end max_d_n_l66_66195


namespace even_sum_less_than_100_l66_66050

theorem even_sum_less_than_100 : 
  (∑ k in (Finset.range 50).filter (λ x, x % 2 = 0), k) = 2450 := by
  sorry

end even_sum_less_than_100_l66_66050


namespace problem_l66_66927

open Real

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (1 / a) + (4 / b) + (9 / c) ≤ 36 / (a + b + c)) 
  : (2 * b + 3 * c) / (a + b + c) = 13 / 6 :=
sorry

end problem_l66_66927


namespace hyperbola_midpoint_l66_66604

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66604


namespace cartons_per_stack_l66_66877

-- Declare the variables and conditions
def total_cartons := 799
def stacks := 133

-- State the theorem
theorem cartons_per_stack : (total_cartons / stacks) = 6 := by
  sorry

end cartons_per_stack_l66_66877


namespace interior_angle_ratio_l66_66975

theorem interior_angle_ratio (exterior_angle1 exterior_angle2 exterior_angle3 : ℝ)
  (h_ratio : 3 * exterior_angle1 = 4 * exterior_angle2 ∧ 
             4 * exterior_angle1 = 5 * exterior_angle3 ∧ 
             3 * exterior_angle1 + 4 * exterior_angle2 + 5 * exterior_angle3 = 360 ) : 
  3 * (180 - exterior_angle1) = 2 * (180 - exterior_angle2) ∧ 
  2 * (180 - exterior_angle2) = 1 * (180 - exterior_angle3) :=
sorry

end interior_angle_ratio_l66_66975


namespace find_d_from_factor_condition_l66_66291

theorem find_d_from_factor_condition (d : ℚ) : (∀ x, x = 5 → d * x^4 + 13 * x^3 - 2 * d * x^2 - 58 * x + 65 = 0) → d = -28 / 23 :=
by
  intro h
  sorry

end find_d_from_factor_condition_l66_66291


namespace jeans_cost_before_sales_tax_l66_66116

-- Defining conditions
def original_cost : ℝ := 49
def summer_discount : ℝ := 0.50
def wednesday_discount : ℝ := 10

-- The mathematical equivalent proof problem
theorem jeans_cost_before_sales_tax :
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  wednesday_price = 14.50 :=
by
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  sorry

end jeans_cost_before_sales_tax_l66_66116


namespace midpoint_on_hyperbola_l66_66524

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66524


namespace hyperbola_midpoint_l66_66603

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66603


namespace midpoint_hyperbola_l66_66545

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66545


namespace apples_picked_l66_66627

theorem apples_picked (n_a : ℕ) (k_a : ℕ) (total : ℕ) (m_a : ℕ) (h_n : n_a = 3) (h_k : k_a = 6) (h_t : total = 16) :
  m_a = total - (n_a + k_a) →
  m_a = 7 :=
by
  sorry

end apples_picked_l66_66627


namespace correct_midpoint_l66_66484

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66484


namespace gcd_problem_l66_66876

variable (A B : ℕ)
variable (hA : A = 2 * 3 * 5)
variable (hB : B = 2 * 2 * 5 * 7)

theorem gcd_problem : Nat.gcd A B = 10 :=
by
  -- Proof is omitted.
  sorry

end gcd_problem_l66_66876


namespace total_tomatoes_l66_66469

def tomatoes_first_plant : Nat := 2 * 12
def tomatoes_second_plant : Nat := (tomatoes_first_plant / 2) + 5
def tomatoes_third_plant : Nat := tomatoes_second_plant + 2

theorem total_tomatoes :
  (tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant) = 60 := by
  sorry

end total_tomatoes_l66_66469


namespace midpoint_on_hyperbola_l66_66563

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66563


namespace solution_for_x_l66_66445

theorem solution_for_x (x : ℝ) : x^2 - x - 1 = (x + 1)^0 → x = 2 :=
by
  intro h
  have h_simp : x^2 - x - 1 = 1 := by simp [h]
  sorry

end solution_for_x_l66_66445


namespace remainder_of_sum_l66_66828

theorem remainder_of_sum (c d : ℤ) (p q : ℤ) (h1 : c = 60 * p + 53) (h2 : d = 45 * q + 28) : 
  (c + d) % 15 = 6 := 
by
  sorry

end remainder_of_sum_l66_66828


namespace sum_even_pos_ints_lt_100_l66_66059

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l66_66059


namespace min_value_frac_inv_l66_66621

theorem min_value_frac_inv (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: a + 3 * b = 2) : 
  (2 + Real.sqrt 3) ≤ (1 / a + 1 / b) :=
sorry

end min_value_frac_inv_l66_66621


namespace input_statement_is_INPUT_l66_66397

namespace ProgrammingStatements

-- Definitions of each type of statement
def PRINT_is_output : Prop := True
def INPUT_is_input : Prop := True
def THEN_is_conditional : Prop := True
def END_is_termination : Prop := True

-- The proof problem
theorem input_statement_is_INPUT :
  INPUT_is_input := by
  sorry

end ProgrammingStatements

end input_statement_is_INPUT_l66_66397


namespace total_pies_eq_l66_66168

-- Definitions for the number of pies made by each person
def pinky_pies : ℕ := 147
def helen_pies : ℕ := 56
def emily_pies : ℕ := 89
def jake_pies : ℕ := 122

-- The theorem stating the total number of pies
theorem total_pies_eq : pinky_pies + helen_pies + emily_pies + jake_pies = 414 :=
by sorry

end total_pies_eq_l66_66168


namespace range_of_a_l66_66286

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^2 + 2*a*x + a) > 0) → (0 < a ∧ a < 1) :=
sorry

end range_of_a_l66_66286


namespace midpoint_of_hyperbola_l66_66580

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l66_66580


namespace number_divided_by_3_equals_subtract_3_l66_66711

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l66_66711


namespace sum_of_products_non_positive_l66_66273

theorem sum_of_products_non_positive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end sum_of_products_non_positive_l66_66273


namespace total_items_count_l66_66963

theorem total_items_count :
  let old_women  := 7
  let mules      := 7
  let bags       := 7
  let loaves     := 7
  let knives     := 7
  let sheaths    := 7
  let sheaths_per_loaf := knives * sheaths
  let sheaths_per_bag := loaves * sheaths_per_loaf
  let sheaths_per_mule := bags * sheaths_per_bag
  let sheaths_per_old_woman := mules * sheaths_per_mule
  let total_sheaths := old_women * sheaths_per_old_woman

  let loaves_per_bag := loaves
  let loaves_per_mule := bags * loaves_per_bag
  let loaves_per_old_woman := mules * loaves_per_mule
  let total_loaves := old_women * loaves_per_old_woman

  let knives_per_loaf := knives
  let knives_per_bag := loaves * knives_per_loaf
  let knives_per_mule := bags * knives_per_bag
  let knives_per_old_woman := mules * knives_per_mule
  let total_knives := old_women * knives_per_old_woman

  let total_bags := old_women * mules * bags

  let total_mules := old_women * mules

  let total_items := total_sheaths + total_loaves + total_knives + total_bags + total_mules + old_women

  total_items = 137256 :=
by
  sorry

end total_items_count_l66_66963


namespace total_fruit_count_l66_66338

-- Define the conditions as variables and equations
def apples := 4 -- based on the final deduction from the solution
def pears := 6 -- calculated from the condition of bananas
def bananas := 9 -- given in the problem

-- State the conditions
axiom h1 : pears = apples + 2
axiom h2 : bananas = pears + 3
axiom h3 : bananas = 9

-- State the proof objective
theorem total_fruit_count : apples + pears + bananas = 19 :=
by
  sorry

end total_fruit_count_l66_66338


namespace eval_expression_l66_66419

theorem eval_expression :
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) - Int.ceil (2 / 3 : ℚ) = -1 := 
by 
  sorry

end eval_expression_l66_66419


namespace midpoint_hyperbola_l66_66540

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66540


namespace part_a_total_time_part_b_average_time_part_c_probability_l66_66638

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l66_66638


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66039

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66039


namespace geometric_sequence_sum_l66_66437

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_seq : seq a)
  (h_a2 : a 2 = 3)
  (h_sum : a 2 + a 4 + a 6 = 21) :
  (a 4 + a 6 + a 8) = 42 :=
sorry

end geometric_sequence_sum_l66_66437


namespace midpoint_on_hyperbola_l66_66569

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66569


namespace tenth_term_arithmetic_seq_l66_66706

theorem tenth_term_arithmetic_seq : 
  ∀ (first_term common_diff : ℤ) (n : ℕ), 
    first_term = 10 → common_diff = -2 → n = 10 → 
    (first_term + (n - 1) * common_diff) = -8 :=
by
  sorry

end tenth_term_arithmetic_seq_l66_66706


namespace find_number_l66_66730

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l66_66730


namespace width_of_margin_l66_66088

-- Given conditions as definitions
def total_area : ℝ := 20 * 30
def percentage_used : ℝ := 0.64
def used_area : ℝ := percentage_used * total_area

-- Definition of the width of the typing area
def width_after_margin (x : ℝ) : ℝ := 20 - 2 * x

-- Definition of the length after top and bottom margins
def length_after_margin : ℝ := 30 - 6

-- Calculate the area used considering the margins
def typing_area (x : ℝ) : ℝ := (width_after_margin x) * length_after_margin

-- Statement to prove
theorem width_of_margin : ∃ x : ℝ, typing_area x = used_area ∧ x = 2 := by
  -- We give the prompt to eventually prove the theorem with the correct value
  sorry

end width_of_margin_l66_66088


namespace power_function_decreasing_m_eq_2_l66_66425

theorem power_function_decreasing_m_eq_2 (x : ℝ) (m : ℝ) (hx : 0 < x) 
  (h_decreasing : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
                    (m^2 - m - 1) * x₁^(-m+1) > (m^2 - m - 1) * x₂^(-m+1))
  (coeff_positive : m^2 - m - 1 > 0)
  (expo_condition : -m + 1 < 0) : 
  m = 2 :=
by
  sorry

end power_function_decreasing_m_eq_2_l66_66425


namespace mean_of_sequence_starting_at_3_l66_66842

def arithmetic_sequence (start : ℕ) (n : ℕ) : List ℕ :=
List.range n |>.map (λ i => start + i)

def arithmetic_mean (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

theorem mean_of_sequence_starting_at_3 : 
  ∀ (seq : List ℕ),
  seq = arithmetic_sequence 3 60 → 
  arithmetic_mean seq = 32.5 := 
by
  intros seq h
  rw [h]
  sorry

end mean_of_sequence_starting_at_3_l66_66842


namespace time_left_for_nap_l66_66907

noncomputable def total_time : ℝ := 20
noncomputable def first_train_time : ℝ := 2 + 1
noncomputable def second_train_time : ℝ := 3 + 1
noncomputable def transfer_one_time : ℝ := 0.75 + 0.5
noncomputable def third_train_time : ℝ := 2 + 1
noncomputable def transfer_two_time : ℝ := 1
noncomputable def fourth_train_time : ℝ := 1
noncomputable def transfer_three_time : ℝ := 0.5
noncomputable def fifth_train_time_before_nap : ℝ := 1.5

noncomputable def total_activities_time : ℝ :=
  first_train_time +
  second_train_time +
  transfer_one_time +
  third_train_time +
  transfer_two_time +
  fourth_train_time +
  transfer_three_time +
  fifth_train_time_before_nap

theorem time_left_for_nap : total_time - total_activities_time = 4.75 := by
  sorry

end time_left_for_nap_l66_66907


namespace earliest_year_for_mismatched_pairs_l66_66236

def num_pairs (year : ℕ) : ℕ := 2 ^ (year - 2013)

def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

theorem earliest_year_for_mismatched_pairs (year : ℕ) (h : year ≥ 2013) :
  (∃ pairs, (num_pairs year = pairs) ∧ (mismatched_pairs pairs ≥ 500)) → year = 2018 :=
by
  sorry

end earliest_year_for_mismatched_pairs_l66_66236


namespace hyperbola_midpoint_l66_66605

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66605


namespace halfway_fraction_l66_66000

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66000


namespace oranges_for_juice_l66_66852

theorem oranges_for_juice 
  (bags : ℕ) (oranges_per_bag : ℕ) (rotten_oranges : ℕ) (oranges_sold : ℕ)
  (h_bags : bags = 10)
  (h_oranges_per_bag : oranges_per_bag = 30)
  (h_rotten_oranges : rotten_oranges = 50)
  (h_oranges_sold : oranges_sold = 220):
  (bags * oranges_per_bag - rotten_oranges - oranges_sold = 30) :=
by 
  sorry

end oranges_for_juice_l66_66852


namespace toms_total_miles_l66_66210

-- Define the conditions as facts
def days_in_year : ℕ := 365
def first_part_days : ℕ := 183
def second_part_days : ℕ := days_in_year - first_part_days
def miles_per_day_first_part : ℕ := 30
def miles_per_day_second_part : ℕ := 35

-- State the final theorem
theorem toms_total_miles : 
  (first_part_days * miles_per_day_first_part) + (second_part_days * miles_per_day_second_part) = 11860 := by 
  sorry

end toms_total_miles_l66_66210


namespace snail_kite_snails_eaten_l66_66392

theorem snail_kite_snails_eaten 
  (a₀ : ℕ) (a₁ : ℕ) (a₂ : ℕ) (a₃ : ℕ) (a₄ : ℕ)
  (h₀ : a₀ = 3)
  (h₁ : a₁ = a₀ + 2)
  (h₂ : a₂ = a₁ + 2)
  (h₃ : a₃ = a₂ + 2)
  (h₄ : a₄ = a₃ + 2)
  : a₀ + a₁ + a₂ + a₃ + a₄ = 35 := 
by 
  sorry

end snail_kite_snails_eaten_l66_66392


namespace midpoint_on_hyperbola_l66_66564

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66564


namespace renne_savings_ratio_l66_66171

theorem renne_savings_ratio (ME CV N : ℕ) (h_ME : ME = 4000) (h_CV : CV = 16000) (h_N : N = 8) :
  (CV / N : ℕ) / ME = 1 / 2 :=
by
  sorry

end renne_savings_ratio_l66_66171


namespace find_number_l66_66732

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l66_66732


namespace min_value_of_quadratic_l66_66113

theorem min_value_of_quadratic (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) : 
  ∃ (x : ℝ), (∃ (y_min : ℝ), y_min = -( (abs (a - b)/2)^2 ) 
  ∧ ∀ (x : ℝ), (x - a)*(x - b) ≥ y_min) :=
sorry

end min_value_of_quadratic_l66_66113


namespace cost_of_case_of_rolls_l66_66383

noncomputable def cost_of_multiple_rolls (n : ℕ) (individual_cost : ℝ) : ℝ :=
  n * individual_cost

theorem cost_of_case_of_rolls :
  ∀ (n : ℕ) (C : ℝ) (individual_cost savings_perc : ℝ),
    n = 12 →
    individual_cost = 1 →
    savings_perc = 0.25 →
    C = cost_of_multiple_rolls n (individual_cost * (1 - savings_perc)) →
    C = 9 :=
by
  intros n C individual_cost savings_perc h1 h2 h3 h4
  sorry

end cost_of_case_of_rolls_l66_66383


namespace fraction_draw_l66_66941

theorem fraction_draw (john_wins : ℚ) (mike_wins : ℚ) (h_john : john_wins = 4 / 9) (h_mike : mike_wins = 5 / 18) :
    1 - (john_wins + mike_wins) = 5 / 18 :=
by
    rw [h_john, h_mike]
    sorry

end fraction_draw_l66_66941


namespace correct_equations_l66_66745

theorem correct_equations (x y : ℝ) :
  (9 * x - y = 4) → (y - 8 * x = 3) → (9 * x - y = 4 ∧ y - 8 * x = 3) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end correct_equations_l66_66745


namespace base_4_last_digit_of_389_l66_66414

theorem base_4_last_digit_of_389 : (389 % 4) = 1 :=
by {
  sorry
}

end base_4_last_digit_of_389_l66_66414


namespace midpoint_of_hyperbola_segment_l66_66502

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66502


namespace wire_ratio_l66_66405

theorem wire_ratio (bonnie_pieces : ℕ) (length_per_bonnie_piece : ℕ) (roark_volume : ℕ) 
  (unit_cube_volume : ℕ) (bonnie_cube_volume : ℕ) (roark_pieces_per_unit_cube : ℕ)
  (bonnie_total_wire : ℕ := bonnie_pieces * length_per_bonnie_piece)
  (roark_total_wire : ℕ := (bonnie_cube_volume / unit_cube_volume) * roark_pieces_per_unit_cube) :
  bonnie_pieces = 12 →
  length_per_bonnie_piece = 4 →
  unit_cube_volume = 1 →
  bonnie_cube_volume = 64 →
  roark_pieces_per_unit_cube = 12 →
  (bonnie_total_wire / roark_total_wire : ℚ) = 1 / 16 :=
by sorry

end wire_ratio_l66_66405


namespace constant_term_of_product_l66_66707

-- Define the polynomials
def poly1 (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + 7
def poly2 (x : ℝ) : ℝ := 4 * x^4 + 2 * x^2 + 10

-- Main statement: Prove that the constant term in the expansion of poly1 * poly2 is 70
theorem constant_term_of_product : (poly1 0) * (poly2 0) = 70 :=
by
  -- The proof would go here
  sorry

end constant_term_of_product_l66_66707


namespace hyperbola_midpoint_l66_66602

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66602


namespace louisa_second_day_distance_l66_66628

-- Definitions based on conditions
def time_on_first_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_on_second_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

def condition (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) : Prop := 
  time_on_first_day distance_first_day speed + time_difference = time_on_second_day x speed

-- The proof statement
theorem louisa_second_day_distance (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) :
  distance_first_day = 240 → 
  speed = 60 → 
  time_difference = 3 → 
  condition distance_first_day speed time_difference x → 
  x = 420 :=
by
  intros h1 h2 h3 h4
  sorry

end louisa_second_day_distance_l66_66628


namespace hyperbola_midpoint_l66_66499

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66499


namespace sum_even_positives_less_than_100_l66_66061

theorem sum_even_positives_less_than_100 :
  ∑ k in Finset.Ico 1 50, 2 * k = 2450 :=
by
  sorry

end sum_even_positives_less_than_100_l66_66061


namespace midpoint_of_hyperbola_l66_66480

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66480


namespace original_price_before_discounts_l66_66742

theorem original_price_before_discounts (P : ℝ) 
  (h : 0.75 * (0.75 * P) = 18) : P = 32 :=
by
  sorry

end original_price_before_discounts_l66_66742


namespace Chloe_final_points_l66_66095

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end Chloe_final_points_l66_66095


namespace seats_taken_l66_66270

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end seats_taken_l66_66270


namespace ratio_f_l66_66192

variable (f : ℝ → ℝ)

-- Hypothesis: For all x in ℝ^+, f'(x) = 3/x * f(x)
axiom hyp1 : ∀ x : ℝ, x > 0 → deriv f x = (3 / x) * f x

-- Hypothesis: f(2^2016) ≠ 0
axiom hyp2 : f (2^2016) ≠ 0

-- Prove that f(2^2017) / f(2^2016) = 8
theorem ratio_f : f (2^2017) / f (2^2016) = 8 :=
sorry

end ratio_f_l66_66192


namespace find_value_l66_66801

theorem find_value (N : ℝ) (h : 1.20 * N = 6000) : 0.20 * N = 1000 :=
sorry

end find_value_l66_66801


namespace find_number_l66_66997

theorem find_number : ∃ x, x - 0.16 * x = 126 ↔ x = 150 :=
by 
  sorry

end find_number_l66_66997


namespace fraction_halfway_between_l66_66044

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l66_66044


namespace petya_addition_mistake_l66_66832

theorem petya_addition_mistake:
  ∃ (x y c : ℕ), x + y = 12345 ∧ (10 * x + c) + y = 44444 ∧ x = 3566 ∧ y = 8779 ∧ c = 5 := by
  sorry

end petya_addition_mistake_l66_66832


namespace correct_midpoint_l66_66486

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66486


namespace difference_in_surface_area_l66_66389

-- Defining the initial conditions
def original_length : ℝ := 6
def original_width : ℝ := 5
def original_height : ℝ := 4
def cube_side : ℝ := 2

-- Define the surface area calculation for a rectangular solid
def surface_area_rectangular_prism (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

-- Define the surface area of the cube
def surface_area_cube (a : ℝ) : ℝ :=
  6 * a * a

-- Define the removed face areas when cube is extracted
def exposed_faces_area (a : ℝ) : ℝ :=
  2 * (a * a)

-- Define the problem statement in Lean
theorem difference_in_surface_area :
  surface_area_rectangular_prism original_length original_width original_height
  - (surface_area_rectangular_prism original_length original_width original_height - surface_area_cube cube_side + exposed_faces_area cube_side) = 12 :=
by
  sorry

end difference_in_surface_area_l66_66389


namespace fields_fertilized_in_25_days_l66_66947

-- Definitions from conditions
def fertilizer_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def fertilizer_needed_per_acre : ℕ := 400
def number_of_acres : ℕ := 20
def acres_fertilized_per_day : ℕ := 4

-- Total fertilizer produced per day
def total_fertilizer_per_day : ℕ := fertilizer_per_horse_per_day * number_of_horses

-- Total fertilizer needed
def total_fertilizer_needed : ℕ := fertilizer_needed_per_acre * number_of_acres

-- Days to collect enough fertilizer
def days_to_collect_fertilizer : ℕ := total_fertilizer_needed / total_fertilizer_per_day

-- Days to spread fertilizer
def days_to_spread_fertilizer : ℕ := number_of_acres / acres_fertilized_per_day

-- Calculate the total time until all fields are fertilized
def total_days : ℕ := days_to_collect_fertilizer + days_to_spread_fertilizer

-- Theorem statement
theorem fields_fertilized_in_25_days : total_days = 25 :=
by
  sorry

end fields_fertilized_in_25_days_l66_66947


namespace problem_statement_l66_66449

-- Define that f is an even function and decreasing on (0, +∞)
variables {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def is_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- Main statement: Prove the specific inequality under the given conditions
theorem problem_statement (f_even : is_even_function f) (f_decreasing : is_decreasing_on_pos f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) :=
by
  sorry

end problem_statement_l66_66449


namespace midpoint_of_hyperbola_segment_l66_66508

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66508


namespace isabela_spent_2800_l66_66460

/-- Given:
1. Isabela bought twice as many cucumbers as pencils.
2. Both cucumbers and pencils cost $20 each.
3. Isabela got a 20% discount on the pencils.
4. She bought 100 cucumbers.
Prove that the total amount Isabela spent is $2800. -/
theorem isabela_spent_2800 :
  ∀ (pencils cucumbers : ℕ) (pencil_cost cucumber_cost : ℤ) (discount rate: ℚ)
    (total_cost pencils_cost cucumbers_cost discount_amount : ℤ),
  cucumbers = 100 →
  pencils * 2 = cucumbers →
  pencil_cost = 20 →
  cucumber_cost = 20 →
  rate = 20 / 100 →
  pencils_cost = pencils * pencil_cost →
  discount_amount = pencils_cost * rate →
  total_cost = pencils_cost - discount_amount + cucumbers * cucumber_cost →
  total_cost = 2800 := by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end isabela_spent_2800_l66_66460


namespace max_sum_combined_shape_l66_66166

-- Definitions for the initial prism
def faces_prism := 6
def edges_prism := 12
def vertices_prism := 8

-- Definitions for the changes when pyramid is added to a rectangular face
def additional_faces_rect := 4
def additional_edges_rect := 4
def additional_vertices_rect := 1

-- Definition for the maximum sum calculation
def max_sum := faces_prism - 1 + additional_faces_rect + 
               edges_prism + additional_edges_rect + 
               vertices_prism + additional_vertices_rect

-- The theorem to prove the maximum sum
theorem max_sum_combined_shape : max_sum = 34 :=
by
  sorry

end max_sum_combined_shape_l66_66166


namespace halfway_fraction_l66_66003

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66003


namespace halfway_fraction_l66_66027

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l66_66027


namespace path_to_tile_ratio_l66_66227

theorem path_to_tile_ratio
  (t p : ℝ) 
  (tiles : ℕ := 400)
  (grid_size : ℕ := 20)
  (total_tile_area : ℝ := (tiles : ℝ) * t^2)
  (total_courtyard_area : ℝ := (grid_size * (t + 2 * p))^2) 
  (tile_area_fraction : ℝ := total_tile_area / total_courtyard_area) : 
  tile_area_fraction = 0.25 → 
  p / t = 0.5 :=
by
  intro h
  sorry

end path_to_tile_ratio_l66_66227


namespace cos_sq_half_diff_eq_csquared_over_a2_b2_l66_66949

theorem cos_sq_half_diff_eq_csquared_over_a2_b2
  (a b c α β : ℝ)
  (h1 : a^2 + b^2 ≠ 0)
  (h2 : a * (Real.cos α) + b * (Real.sin α) = c)
  (h3 : a * (Real.cos β) + b * (Real.sin β) = c)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  Real.cos (α - β) / 2 = c^2 / (a^2 + b^2) :=
by
  sorry

end cos_sq_half_diff_eq_csquared_over_a2_b2_l66_66949


namespace can_be_midpoint_of_AB_l66_66515

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66515


namespace marthas_bedroom_size_l66_66357

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end marthas_bedroom_size_l66_66357


namespace math_proof_problem_l66_66406
noncomputable def expr : ℤ := 3000 * (3000 ^ 3000) + 3000 ^ 2

theorem math_proof_problem : expr = 3000 ^ 3001 + 9000000 :=
by
  -- Proof
  sorry

end math_proof_problem_l66_66406


namespace triangle_area_l66_66813

theorem triangle_area {a b : ℝ} (h : a ≠ 0) :
  (∃ x y : ℝ, 3 * x + a * y = 12) → b = 24 / a ↔ (∃ x y : ℝ, x = 4 ∧ y = 12 / a ∧ b = (1/2) * 4 * (12 / a)) :=
by
  sorry

end triangle_area_l66_66813


namespace solve_comb_eq_l66_66964

open Nat

def comb (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))
def perm (n k : ℕ) : ℕ := (factorial n) / (factorial (n - k))

theorem solve_comb_eq (x : ℕ) :
  comb (x + 5) x = comb (x + 3) (x - 1) + comb (x + 3) (x - 2) + 3/4 * perm (x + 3) 3 ->
  x = 14 := 
by 
  sorry

end solve_comb_eq_l66_66964


namespace inequality_proof_l66_66800

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b :=
by
  sorry

end inequality_proof_l66_66800


namespace nearest_integer_to_x_plus_2y_l66_66937

theorem nearest_integer_to_x_plus_2y
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6)
  (h2 : |x| * y + x^3 = 2) :
  Int.floor (x + 2 * y + 0.5) = 6 :=
by sorry

end nearest_integer_to_x_plus_2y_l66_66937


namespace sine_equation_solution_l66_66420

theorem sine_equation_solution (n : ℕ) (h : 0 ≤ n ∧ n ≤ 180) : sin (n * (π / 180)) = sin (192 * (π / 180)) ↔ n = 12 ∨ n = 168 :=
by
  sorry

end sine_equation_solution_l66_66420


namespace correct_midpoint_l66_66483

-- Define the hyperbola equation
def on_hyperbola (A B : ℝ × ℝ) : Prop :=
  (A.fst ^ 2 - (A.snd ^ 2) / 9 = 1) ∧ (B.fst ^ 2 - (B.snd ^ 2) / 9 = 1)

-- Define the midpoint condition
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Candidate midpoints
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (-1, 2)
def M3 : ℝ × ℝ := (1, 3)
def M4 : ℝ × ℝ := (-1, -4)

-- Prove that M4 is the midpoint of segment AB
theorem correct_midpoint : ∃ (A B : ℝ × ℝ), on_hyperbola A B ∧ midpoint A B M4 :=
by sorry

end correct_midpoint_l66_66483


namespace walking_area_calculation_l66_66304

noncomputable def walking_area_of_park (park_length park_width fountain_radius : ℝ) : ℝ :=
  let park_area := park_length * park_width
  let fountain_area := Real.pi * fountain_radius^2
  park_area - fountain_area

theorem walking_area_calculation :
  walking_area_of_park 50 30 5 = 1500 - 25 * Real.pi :=
by
  sorry

end walking_area_calculation_l66_66304


namespace peggy_buys_three_folders_l66_66167

theorem peggy_buys_three_folders 
  (red_sheets : ℕ) (green_sheets : ℕ) (blue_sheets : ℕ)
  (red_stickers_per_sheet : ℕ) (green_stickers_per_sheet : ℕ) (blue_stickers_per_sheet : ℕ)
  (total_stickers : ℕ) :
  red_sheets = 10 →
  green_sheets = 10 →
  blue_sheets = 10 →
  red_stickers_per_sheet = 3 →
  green_stickers_per_sheet = 2 →
  blue_stickers_per_sheet = 1 →
  total_stickers = 60 →
  1 + 1 + 1 = 3 :=
by 
  intros _ _ _ _ _ _ _
  sorry

end peggy_buys_three_folders_l66_66167


namespace at_least_two_sums_divisible_by_p_l66_66823

open Int

noncomputable def fractional_part (x : ℚ) : ℚ := x - floor(x)

theorem at_least_two_sums_divisible_by_p
  (p : ℕ) (hp : prime p) (hp_gt2 : 2 < p)
  (a b c d : ℤ)
  (ha : a % p ≠ 0) (hb : b % p ≠ 0) 
  (hc : c % p ≠ 0) (hd : d % p ≠ 0)
  (fractional_cond : ∀ (r : ℤ), (r % p ≠ 0) →
    (fractional_part r * a / p +
     fractional_part r * b / p +
     fractional_part r * c / p +
     fractional_part r * d / p) = 2) :
  ∃ u v, u ≠ v ∧ (u + v) % p = 0 :=
sorry

end at_least_two_sums_divisible_by_p_l66_66823


namespace find_number_l66_66718

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l66_66718


namespace new_arithmetic_mean_l66_66972

theorem new_arithmetic_mean
  (seq : List ℝ)
  (h_seq_len : seq.length = 60)
  (h_mean : (seq.sum / 60 : ℝ) = 42)
  (h_removed : ∃ a b, a ∈ seq ∧ b ∈ seq ∧ a = 50 ∧ b = 60) :
  ((seq.erase 50).erase 60).sum / 58 = 41.55 := 
sorry

end new_arithmetic_mean_l66_66972


namespace grown_ups_in_milburg_l66_66363

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l66_66363


namespace A_not_divisible_by_B_l66_66069

variable (A B : ℕ)
variable (h1 : A ≠ B)
variable (h2 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))
variable (h3 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))

theorem A_not_divisible_by_B : ¬ (A % B = 0) :=
sorry

end A_not_divisible_by_B_l66_66069


namespace man_speed_is_approximately_54_009_l66_66750

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ := 
  -- Convert distance to kilometers and time to hours
  let distance_km := d / 1000
  let time_hours := t / 3600
  distance_km / time_hours

theorem man_speed_is_approximately_54_009 :
  abs (speed_in_kmh 375.03 25 - 54.009) < 0.001 := 
by
  sorry

end man_speed_is_approximately_54_009_l66_66750


namespace find_a_l66_66812

-- Define point
structure Point where
  x : ℝ
  y : ℝ

-- Define curves
def C1 (a x : ℝ) : ℝ := a * x^3 + 1
def C2 (P : Point) : Prop := P.x^2 + P.y^2 = 5 / 2

-- Define the tangent slope function for curve C1
def tangent_slope_C1 (a x : ℝ) : ℝ := 3 * a * x^2

-- State the problem that we need to prove
theorem find_a (a x₀ y₀ : ℝ) (h1 : y₀ = C1 a x₀) (h2 : C2 ⟨x₀, y₀⟩) (h3 : y₀ = 3 * a * x₀^3) 
  (ha_pos : 0 < a) : a = 4 := 
  by
    sorry

end find_a_l66_66812


namespace gcd_a_b_l66_66863

def a := 130^2 + 250^2 + 360^2
def b := 129^2 + 249^2 + 361^2

theorem gcd_a_b : Int.gcd a b = 1 := 
by
  sorry

end gcd_a_b_l66_66863


namespace N_composite_l66_66916

theorem N_composite :
  let N := 7 * 9 * 13 + 2020 * 2018 * 2014 in
  ¬Nat.prime N := 
by
  sorry

end N_composite_l66_66916


namespace solution_system_inequalities_l66_66335

theorem solution_system_inequalities (x : ℝ) : 
  (x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x) ↔ (2 < x ∧ x ≤ 4) := 
sorry

end solution_system_inequalities_l66_66335


namespace grace_earnings_l66_66133

noncomputable def weekly_charge : ℕ := 300
noncomputable def payment_interval : ℕ := 2
noncomputable def target_weeks : ℕ := 6
noncomputable def target_amount : ℕ := 1800

theorem grace_earnings :
  (target_weeks * weekly_charge = target_amount) → 
  (target_weeks / payment_interval) * (payment_interval * weekly_charge) = target_amount :=
by
  sorry

end grace_earnings_l66_66133


namespace complement_of_A_is_correct_l66_66625

-- Define the universal set U and the set A.
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U.
def A_complement : Set ℕ := {x ∈ U | x ∉ A}

-- The theorem statement that the complement of A in U is {2, 4}.
theorem complement_of_A_is_correct : A_complement = {2, 4} :=
sorry

end complement_of_A_is_correct_l66_66625


namespace distance_point_parabola_focus_l66_66969

theorem distance_point_parabola_focus (P : ℝ × ℝ) (x y : ℝ) (hP : P = (3, y)) (h_parabola : y^2 = 4 * 3) :
    dist P (0, -1) = 4 :=
by
  sorry

end distance_point_parabola_focus_l66_66969


namespace overall_profit_no_discount_l66_66895

theorem overall_profit_no_discount:
  let C_b := 100
  let C_p := 100
  let C_n := 100
  let profit_b := 42.5 / 100
  let profit_p := 35 / 100
  let profit_n := 20 / 100
  let S_b := C_b + (C_b * profit_b)
  let S_p := C_p + (C_p * profit_p)
  let S_n := C_n + (C_n * profit_n)
  let TCP := C_b + C_p + C_n
  let TSP := S_b + S_p + S_n
  let OverallProfit := TSP - TCP
  let OverallProfitPercentage := (OverallProfit / TCP) * 100
  OverallProfitPercentage = 32.5 :=
by sorry

end overall_profit_no_discount_l66_66895


namespace range_of_x_plus_y_l66_66274

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y - (x + y) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_x_plus_y_l66_66274


namespace midpoint_hyperbola_l66_66537

section hyperbola_problem

variables {A B : ℝ × ℝ} (x1 y1 x2 y2 : ℝ)

-- Define the hyperbola equation
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 / 9 = 1

-- Define the midpoint function
def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define the property to check if the line segment AB can have (-1, -4) as its midpoint
def valid_midpoint (P1 P2 M : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x0, y0) := M
  (on_hyperbola P1) ∧ (on_hyperbola P2) ∧ (midpoint P1 P2 = M)

-- State the theorem to be proved
theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (A B : ℝ × ℝ) :
  A = (x1, y1) ∧ B = (x2, y2) ∧ on_hyperbola A ∧ on_hyperbola B → 
  valid_midpoint A B (-1, -4) :=
begin
  sorry
end

end hyperbola_problem

end midpoint_hyperbola_l66_66537


namespace necessary_and_sufficient_condition_extremum_l66_66686

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 6 * x^2 + (a - 1) * x - 5

theorem necessary_and_sufficient_condition_extremum (a : ℝ) :
  (∃ x, f a x = 0) ↔ -3 < a ∧ a < 4 :=
sorry

end necessary_and_sufficient_condition_extremum_l66_66686


namespace part_a_part_b_part_c_l66_66648

open Nat

-- Definition of the number of combinations (C(10, 3))
def combinations : ℕ := 10.choose 3

-- Each attempt takes 2 seconds
def seconds_per_attempt : ℕ := 2

-- Total time required to try all combinations in seconds
def total_time_in_seconds : ℕ := combinations * seconds_per_attempt

-- Total time required to try all combinations in minutes
def total_time_in_minutes : ℕ := total_time_in_seconds / 60

-- Average number of attempts
def average_attempts : ℚ := (1 + combinations) / 2

-- Average time in seconds
def average_time_in_seconds : ℚ := average_attempts * seconds_per_attempt

-- Probability of getting inside in less than a minute
def probability_in_less_than_a_minute : ℚ := 29 / combinations

-- Theorem statements
theorem part_a : total_time_in_minutes = 4 := sorry
theorem part_b : average_time_in_seconds = 121 := sorry
theorem part_c : probability_in_less_than_a_minute = 29 / 120 := sorry


end part_a_part_b_part_c_l66_66648


namespace greatest_possible_b_l66_66190

theorem greatest_possible_b (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
by sorry

end greatest_possible_b_l66_66190


namespace percentage_girls_l66_66232

theorem percentage_girls (x y : ℕ) (S₁ S₂ : ℕ)
  (h1 : S₁ = 22 * x)
  (h2 : S₂ = 47 * y)
  (h3 : (S₁ + S₂) / (x + y) = 41) :
  (x : ℝ) / (x + y) = 0.24 :=
sorry

end percentage_girls_l66_66232


namespace hyperbola_midpoint_l66_66587

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66587


namespace martha_bedroom_size_l66_66359

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l66_66359


namespace find_nth_term_of_arithmetic_seq_l66_66430

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_progression (a1 a2 a5 : ℝ) :=
  a1 * a5 = a2^2

theorem find_nth_term_of_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_seq a d)
    (h_a1 : a 1 = 1) (h_nonzero : d ≠ 0) (h_geom : is_geometric_progression (a 1) (a 2) (a 5)) : 
    ∀ n, a n = 2 * n - 1 :=
by
  sorry

end find_nth_term_of_arithmetic_seq_l66_66430


namespace johnny_guitar_practice_l66_66466

theorem johnny_guitar_practice :
  ∃ x : ℕ, (∃ d : ℕ, d = 20 ∧ ∀ n : ℕ, (n = x - d ∧ n = x / 2)) ∧ (x + 80 = 3 * x) :=
by
  sorry

end johnny_guitar_practice_l66_66466


namespace can_be_midpoint_of_AB_l66_66516

def is_on_hyperbola (x y : ℝ) : Prop :=
   x^2 - y^2 / 9 = 1

def midpoint (A B M : ℝ × ℝ) : Prop :=
   M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem can_be_midpoint_of_AB :
  ∃ A B : ℝ × ℝ, is_on_hyperbola A.1 A.2 ∧ is_on_hyperbola B.1 B.2 ∧ midpoint A B (-1, -4) :=
begin
  sorry
end

end can_be_midpoint_of_AB_l66_66516


namespace hyperbola_midpoint_l66_66607

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66607


namespace student_correct_answers_l66_66307

theorem student_correct_answers (C W : ℕ) (h₁ : C + W = 50) (h₂ : 4 * C - W = 130) : C = 36 := 
by
  sorry

end student_correct_answers_l66_66307


namespace find_a_if_f_is_even_l66_66297

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x - a / Real.exp x)

theorem find_a_if_f_is_even
  (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 :=
sorry

end find_a_if_f_is_even_l66_66297


namespace number_of_five_ruble_coins_l66_66657

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l66_66657


namespace competition_arrangements_l66_66779

noncomputable def count_arrangements (students : Fin 4) (events : Fin 3) : Nat :=
  -- The actual counting function is not implemented
  sorry

theorem competition_arrangements (students : Fin 4) (events : Fin 3) :
  let count := count_arrangements students events
  (∃ (A B C D : Fin 4), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧ 
    (A ≠ 0) ∧ 
    count = 24) := sorry

end competition_arrangements_l66_66779


namespace find_number_l66_66731

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l66_66731


namespace book_total_pages_eq_90_l66_66073

theorem book_total_pages_eq_90 {P : ℕ} (h1 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30) : P = 90 :=
sorry

end book_total_pages_eq_90_l66_66073


namespace ratio_problem_l66_66444

variable {a b c d : ℚ}

theorem ratio_problem (h₁ : a / b = 5) (h₂ : c / b = 3) (h₃ : c / d = 2) :
  d / a = 3 / 10 :=
sorry

end ratio_problem_l66_66444


namespace hyperbola_midpoint_l66_66493

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66493


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66636

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66636


namespace pump_filling_time_without_leak_l66_66086

theorem pump_filling_time_without_leak (P : ℝ) (h1 : 1 / P - 1 / 14 = 3 / 7) : P = 2 :=
sorry

end pump_filling_time_without_leak_l66_66086


namespace part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l66_66432

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a / x + Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / (x^2) + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f' x a - x

-- Problem (I)
theorem part_I (a : ℝ) : f' 1 a = 0 → a = 2 := sorry

-- Problem (II)
theorem part_II (a : ℝ) : (∀ x, 1 < x ∧ x < 2 → f' x a ≥ 0) → a ≤ 2 := sorry

-- Problem (III)
theorem part_III_no_zeros (a : ℝ) : a > 1 → ∀ x, g x a ≠ 0 := sorry
theorem part_III_one_zero (a : ℝ) : (a = 1 ∨ a ≤ 0) → ∃! x, g x a = 0 := sorry
theorem part_III_two_zeros (a : ℝ) : 0 < a ∧ a < 1 → ∃ x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0 := sorry

end part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l66_66432


namespace inequality_not_true_l66_66135

theorem inequality_not_true (a b : ℝ) (h : a > b) : (a / (-2)) ≤ (b / (-2)) :=
sorry

end inequality_not_true_l66_66135


namespace age_of_15th_student_l66_66743

theorem age_of_15th_student : 
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  total_age_all_students - (total_age_first_group + total_age_second_group) = 16 :=
by
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  sorry

end age_of_15th_student_l66_66743


namespace fill_box_with_cubes_l66_66867

-- Define the dimensions of the box
def boxLength : ℕ := 35
def boxWidth : ℕ := 20
def boxDepth : ℕ := 10

-- Define the greatest common divisor of the box dimensions
def gcdBoxDims : ℕ := Nat.gcd (Nat.gcd boxLength boxWidth) boxDepth

-- Define the smallest number of identical cubes that can fill the box
def smallestNumberOfCubes : ℕ := (boxLength / gcdBoxDims) * (boxWidth / gcdBoxDims) * (boxDepth / gcdBoxDims)

theorem fill_box_with_cubes :
  smallestNumberOfCubes = 56 :=
by
  -- Proof goes here
  sorry

end fill_box_with_cubes_l66_66867


namespace translate_function_right_by_2_l66_66702

theorem translate_function_right_by_2 (x : ℝ) : 
  (∀ x, (x - 2) ^ 2 + (x - 2) = x ^ 2 - 3 * x + 2) := 
by 
  sorry

end translate_function_right_by_2_l66_66702


namespace library_shelves_l66_66386

theorem library_shelves (S : ℕ) (h_books : 4305 + 11 = 4316) :
  4316 % S = 0 ↔ S = 11 :=
by 
  have h_total_books := h_books
  sorry

end library_shelves_l66_66386


namespace bus_stops_time_per_hour_l66_66868

theorem bus_stops_time_per_hour 
  (avg_speed_without_stoppages : ℝ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 75) 
  (h2 : avg_speed_with_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 28 :=
by
  sorry

end bus_stops_time_per_hour_l66_66868


namespace balloons_left_after_distribution_l66_66865

theorem balloons_left_after_distribution :
  (22 + 40 + 70 + 90) % 10 = 2 := by
  sorry

end balloons_left_after_distribution_l66_66865


namespace expected_value_of_game_l66_66885

theorem expected_value_of_game :
  let heads_prob := 1 / 4
  let tails_prob := 1 / 2
  let edge_prob := 1 / 4
  let gain_heads := 4
  let loss_tails := -3
  let gain_edge := 0
  let expected_value := heads_prob * gain_heads + tails_prob * loss_tails + edge_prob * gain_edge
  expected_value = -0.5 :=
by
  sorry

end expected_value_of_game_l66_66885


namespace triangle_range_condition_l66_66804

def triangle_side_range (x : ℝ) : Prop :=
  (1 < x) ∧ (x < 17)

theorem triangle_range_condition (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 8) → (b = 9) → triangle_side_range x :=
by
  intros h1 h2
  dsimp [triangle_side_range]
  sorry

end triangle_range_condition_l66_66804


namespace marthas_bedroom_size_l66_66353

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l66_66353


namespace range_of_a_l66_66735

namespace InequalityProblem

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (1 < x ∧ x < 2) → (x - 1)^2 < Real.log x / Real.log a) ↔ (1 < a ∧ a ≤ 2) :=
by
  sorry

end InequalityProblem

end range_of_a_l66_66735


namespace midpoint_of_hyperbola_l66_66475

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end midpoint_of_hyperbola_l66_66475


namespace smallest_number_l66_66416

-- Define the numbers
def A := 5.67823
def B := 5.67833333333 -- repeating decimal
def C := 5.67838383838 -- repeating decimal
def D := 5.67837837837 -- repeating decimal
def E := 5.6783678367  -- repeating decimal

-- The Lean statement to prove that E is the smallest
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by
  sorry

end smallest_number_l66_66416


namespace infinitesolutions_k_l66_66774

-- Define the system of equations as given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = k)

-- State the theorem that describes the condition for infinitely many solutions
theorem infinitesolutions_k (k : ℝ) :
  (∀ (x y : ℝ), system_of_equations x y k) ↔ k = 15 :=
by
  sorry

end infinitesolutions_k_l66_66774


namespace tangent_line_eq_l66_66921

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 + x + 1) (h_point : x = 1 ∧ y = 3) : 
  y = 4 * x - 1 := 
sorry

end tangent_line_eq_l66_66921


namespace ellipse_equation_l66_66784

theorem ellipse_equation (a : ℝ) (x y : ℝ) (h : (x, y) = (-3, 2)) :
  (∃ a : ℝ, ∀ x y : ℝ, x^2 / 15 + y^2 / 10 = 1) ↔ (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 15 + p.2^2 / 10 = 1 } :=
by
  have h1 : 15 = a^2 := by
    sorry
  have h2 : 10 = a^2 - 5 := by
    sorry
  sorry

end ellipse_equation_l66_66784


namespace question_1_question_2_l66_66129

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x

theorem question_1 (t : ℝ) (ht : t > 0) :
  (if t ≥ 1 / Real.exp 1 then
    ∀ x ∈ Set.Icc t (t + 2), f x = t * Real.log t + 2
  else
    ∀ x ∈ Set.Icc t (t + 2), f x = - 1 / Real.exp 1 + 2) :=
sorry

theorem question_2 (m : ℝ) :
  (∃ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, m * (Real.log x + 1) + x^2 - m * x ≥ 2 * x + m) ->
  m ≤ -1 :=
sorry

end question_1_question_2_l66_66129


namespace midpoint_on_hyperbola_l66_66567

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66567


namespace equivalence_of_statements_l66_66736

theorem equivalence_of_statements (S X Y : Prop) : 
  (S → (¬ X ∧ ¬ Y)) ↔ ((X ∨ Y) → ¬ S) :=
by sorry

end equivalence_of_statements_l66_66736


namespace martha_bedroom_size_l66_66360

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l66_66360


namespace midpoint_of_hyperbola_segment_l66_66510

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l66_66510


namespace question1_question2_l66_66453

/-
In ΔABC, the sides opposite to angles A, B, and C are respectively a, b, and c.
It is given that b + c = 2 * a * cos B.

(1) Prove that A = 2B;
(2) If the area of ΔABC is S = a^2 / 4, find the magnitude of angle A.
-/

variables {A B C a b c : ℝ}
variables {S : ℝ}

-- Condition given in the problem
axiom h1 : b + c = 2 * a * Real.cos B
axiom h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4

-- Question 1: Prove that A = 2 * B
theorem question1 (h1 : b + c = 2 * a * Real.cos B) : A = 2 * B := sorry

-- Question 2: Find the magnitude of angle A
theorem question2 (h2 : 1 / 2 * b * c * Real.sin A = a^2 / 4) : A = 90 ∨ A = 45 := sorry

end question1_question2_l66_66453


namespace area_of_trapezoid_MBCN_l66_66944

variables {AB BC MN : ℝ}
variables {Area_ABCD Area_MBCN : ℝ}
variables {Height : ℝ}

-- Given conditions
def cond1 : Area_ABCD = 40 := sorry
def cond2 : AB = 8 := sorry
def cond3 : BC = 5 := sorry
def cond4 : MN = 2 := sorry
def cond5 : Height = 5 := sorry

-- Define the theorem to be proven
theorem area_of_trapezoid_MBCN : 
  Area_ABCD = AB * BC → MN + BC = 6 → Height = 5 →
  Area_MBCN = (1/2) * (MN + BC) * Height → 
  Area_MBCN = 15 :=
by
  intros h1 h2 h3 h4
  sorry

end area_of_trapezoid_MBCN_l66_66944


namespace complement_union_l66_66439

def U : Set ℤ := {x | -3 < x ∧ x ≤ 4}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {1, 2, 3}

def C (U : Set ℤ) (S : Set ℤ) : Set ℤ := {x | x ∈ U ∧ x ∉ S}

theorem complement_union (A B : Set ℤ) (U : Set ℤ) :
  C U (A ∪ B) = {0, 4} :=
by
  sorry

end complement_union_l66_66439


namespace price_after_two_reductions_l66_66345

variable (orig_price : ℝ) (m : ℝ)

def current_price (orig_price : ℝ) (m : ℝ) : ℝ :=
  orig_price * (1 - m) * (1 - m)

theorem price_after_two_reductions (h1 : orig_price = 100) (h2 : 0 ≤ m ∧ m ≤ 1) :
  current_price orig_price m = 100 * (1 - m) ^ 2 := by
    sorry

end price_after_two_reductions_l66_66345


namespace num_five_ruble_coins_l66_66666

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l66_66666


namespace cost_per_semester_correct_l66_66465

variable (cost_per_semester total_cost : ℕ)
variable (years semesters_per_year : ℕ)

theorem cost_per_semester_correct :
    years = 13 →
    semesters_per_year = 2 →
    total_cost = 520000 →
    cost_per_semester = total_cost / (years * semesters_per_year) →
    cost_per_semester = 20000 := by
  sorry

end cost_per_semester_correct_l66_66465


namespace intersection_points_on_ellipse_l66_66114

theorem intersection_points_on_ellipse (s x y : ℝ)
  (h_line1 : s * x - 3 * y - 4 * s = 0)
  (h_line2 : x - 3 * s * y + 4 = 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry

end intersection_points_on_ellipse_l66_66114


namespace avg_of_eleven_numbers_l66_66337

variable (S1 : ℕ)
variable (S2 : ℕ)
variable (sixth_num : ℕ)
variable (total_sum : ℕ)
variable (avg_eleven : ℕ)

def condition1 := S1 = 6 * 58
def condition2 := S2 = 6 * 65
def condition3 := sixth_num = 188
def condition4 := total_sum = S1 + S2 - sixth_num
def condition5 := avg_eleven = total_sum / 11

theorem avg_of_eleven_numbers : (S1 = 6 * 58) →
                                (S2 = 6 * 65) →
                                (sixth_num = 188) →
                                (total_sum = S1 + S2 - sixth_num) →
                                (avg_eleven = total_sum / 11) →
                                avg_eleven = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_of_eleven_numbers_l66_66337


namespace midpoint_of_line_segment_on_hyperbola_l66_66620

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66620


namespace fish_caught_by_twentieth_fisherman_l66_66369

theorem fish_caught_by_twentieth_fisherman :
  ∀ (total_fishermen total_fish fish_per_fisherman nineten_fishermen : ℕ),
  total_fishermen = 20 →
  total_fish = 10000 →
  fish_per_fisherman = 400 →
  nineten_fishermen = 19 →
  (total_fishermen * fish_per_fisherman) - (nineten_fishermen * fish_per_fisherman) = 2400 :=
by
  intros
  sorry

end fish_caught_by_twentieth_fisherman_l66_66369


namespace isosceles_right_triangle_measure_l66_66871

theorem isosceles_right_triangle_measure (a XY YZ : ℝ) 
    (h1 : XY > YZ) 
    (h2 : a^2 = 25 / (1/2)) : XY = 10 :=
by
  sorry

end isosceles_right_triangle_measure_l66_66871


namespace grown_ups_in_milburg_l66_66366

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l66_66366


namespace solve_quadratic_equation_l66_66836

theorem solve_quadratic_equation (x : ℝ) : x^2 + 4 * x = 5 ↔ x = 1 ∨ x = -5 := sorry

end solve_quadratic_equation_l66_66836


namespace expression_value_eq_3084_l66_66320

theorem expression_value_eq_3084 (x : ℤ) (hx : x = -3007) :
  (abs (abs (Real.sqrt (abs x - x) - x) - x) - Real.sqrt (abs (x - x^2)) = 3084) :=
by
  sorry

end expression_value_eq_3084_l66_66320


namespace halfway_fraction_l66_66005

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66005


namespace hyperbola_asymptotes_equation_l66_66799

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

end hyperbola_asymptotes_equation_l66_66799


namespace fraction_half_way_l66_66018

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l66_66018


namespace halfway_fraction_l66_66001

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l66_66001


namespace cost_price_A_l66_66893

theorem cost_price_A (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ) 
(h1 : CP_B = 1.20 * CP_A)
(h2 : SP_C = 1.25 * CP_B)
(h3 : SP_C = 225) : 
CP_A = 150 := 
by 
  sorry

end cost_price_A_l66_66893


namespace football_cost_l66_66866

theorem football_cost (cost_shorts cost_shoes money_have money_need : ℝ)
  (h_shorts : cost_shorts = 2.40)
  (h_shoes : cost_shoes = 11.85)
  (h_have : money_have = 10)
  (h_need : money_need = 8) :
  (money_have + money_need - (cost_shorts + cost_shoes) = 3.75) :=
by
  -- Proof goes here
  sorry

end football_cost_l66_66866


namespace frequency_of_blurred_pages_l66_66315

def crumpled_frequency : ℚ := 1 / 7
def total_pages : ℕ := 42
def neither_crumpled_nor_blurred : ℕ := 24

def blurred_frequency : ℚ :=
  let crumpled_pages := total_pages / 7
  let either_crumpled_or_blurred := total_pages - neither_crumpled_nor_blurred
  let blurred_pages := either_crumpled_or_blurred - crumpled_pages
  blurred_pages / total_pages

theorem frequency_of_blurred_pages :
  blurred_frequency = 2 / 7 := by
  sorry

end frequency_of_blurred_pages_l66_66315


namespace sculpture_and_base_total_height_l66_66071

noncomputable def sculpture_height_ft : Nat := 2
noncomputable def sculpture_height_in : Nat := 10
noncomputable def base_height_in : Nat := 4
noncomputable def inches_per_foot : Nat := 12

theorem sculpture_and_base_total_height :
  (sculpture_height_ft * inches_per_foot + sculpture_height_in + base_height_in = 38) :=
by
  sorry

end sculpture_and_base_total_height_l66_66071


namespace minimum_a_l66_66298

theorem minimum_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → 
  a ≥ -5/2 :=
sorry

end minimum_a_l66_66298


namespace sara_total_money_eq_640_l66_66175

def days_per_week : ℕ := 5
def cakes_per_day : ℕ := 4
def price_per_cake : ℕ := 8
def weeks : ℕ := 4

theorem sara_total_money_eq_640 :
  (days_per_week * cakes_per_day * price_per_cake * weeks) = 640 := 
sorry

end sara_total_money_eq_640_l66_66175


namespace marcy_needs_6_tubs_of_lip_gloss_l66_66323

theorem marcy_needs_6_tubs_of_lip_gloss (people tubes_per_person tubes_per_tub : ℕ) 
  (h1 : people = 36) (h2 : tubes_per_person = 3) (h3 : tubes_per_tub = 2) :
  (people / tubes_per_person) / tubes_per_tub = 6 :=
by
  -- The proof goes here
  sorry

end marcy_needs_6_tubs_of_lip_gloss_l66_66323


namespace how_many_cakes_each_friend_ate_l66_66955

-- Definitions pertaining to the problem conditions
def crackers : ℕ := 29
def cakes : ℕ := 30
def friends : ℕ := 2

-- The main theorem statement we aim to prove
theorem how_many_cakes_each_friend_ate 
  (h1 : crackers = 29)
  (h2 : cakes = 30)
  (h3 : friends = 2) : 
  (cakes / friends = 15) :=
by
  sorry

end how_many_cakes_each_friend_ate_l66_66955


namespace four_digit_number_exists_l66_66100

-- Definitions corresponding to the conditions in the problem
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def follows_scheme (n : ℕ) (d : ℕ) : Prop :=
  -- Placeholder for the scheme condition
  sorry

-- The Lean statement for the proof problem
theorem four_digit_number_exists :
  ∃ n d1 d2 : ℕ, is_four_digit_number n ∧ follows_scheme n d1 ∧ follows_scheme n d2 ∧ 
  (n = 1014 ∨ n = 1035 ∨ n = 1512) :=
by {
  -- Placeholder for proof steps
  sorry
}

end four_digit_number_exists_l66_66100


namespace remainder_is_20_l66_66341

theorem remainder_is_20 :
  ∀ (larger smaller quotient remainder : ℕ),
    (larger = 1634) →
    (larger - smaller = 1365) →
    (larger = quotient * smaller + remainder) →
    (quotient = 6) →
    remainder = 20 :=
by
  intros larger smaller quotient remainder h_larger h_difference h_division h_quotient
  sorry

end remainder_is_20_l66_66341


namespace min_value_shift_l66_66142

noncomputable def f (x : ℝ) (c : ℝ) := x^2 + 4 * x + 5 - c

theorem min_value_shift (c : ℝ) (h : ∀ x : ℝ, f x c ≥ 2) :
  ∀ x : ℝ, f (x - 2009) c ≥ 2 :=
sorry

end min_value_shift_l66_66142


namespace divide_into_parts_l66_66251

theorem divide_into_parts (x y : ℚ) (h_sum : x + y = 10) (h_diff : y - x = 5) : 
  x = 5 / 2 ∧ y = 15 / 2 := 
sorry

end divide_into_parts_l66_66251


namespace pet_store_cages_l66_66754

theorem pet_store_cages 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (puppies_per_cage : ℕ) 
  (h_initial_puppies : initial_puppies = 45) 
  (h_sold_puppies : sold_puppies = 11) 
  (h_puppies_per_cage : puppies_per_cage = 7) 
  : (initial_puppies - sold_puppies + puppies_per_cage - 1) / puppies_per_cage = 5 :=
by sorry

end pet_store_cages_l66_66754


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66036

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66036


namespace passengers_on_ship_l66_66139

theorem passengers_on_ship : 
  ∀ (P : ℕ), 
    P / 20 + P / 15 + P / 10 + P / 12 + P / 30 + 60 = P → 
    P = 90 :=
by 
  intros P h
  sorry

end passengers_on_ship_l66_66139


namespace square_side_length_l66_66846

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 2 :=
by 
  sorry

end square_side_length_l66_66846


namespace final_percentage_acid_l66_66066

theorem final_percentage_acid (initial_volume : ℝ) (initial_percentage : ℝ)
(removal_volume : ℝ) (final_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 12 → 
  initial_percentage = 0.40 → 
  removal_volume = 4 →
  final_volume = initial_volume - removal_volume →
  final_percentage = (initial_percentage * initial_volume) / final_volume * 100 →
  final_percentage = 60 := by
  intros h1 h2 h3 h4 h5
  sorry

end final_percentage_acid_l66_66066


namespace percentage_cut_in_magazine_budget_l66_66760

noncomputable def magazine_budget_cut (original_budget : ℕ) (cut_amount : ℕ) : ℕ :=
  (cut_amount * 100) / original_budget

theorem percentage_cut_in_magazine_budget : 
  magazine_budget_cut 940 282 = 30 :=
by
  sorry

end percentage_cut_in_magazine_budget_l66_66760


namespace photo_students_count_l66_66976

theorem photo_students_count (n m : ℕ) 
  (h1 : m - 1 = n + 4) 
  (h2 : m - 2 = n) : 
  n * m = 24 := 
by 
  sorry

end photo_students_count_l66_66976


namespace coffee_machine_price_l66_66818

noncomputable def original_machine_price : ℝ :=
  let coffees_prior_cost_per_day := 2 * 4
  let new_coffees_cost_per_day := 3
  let daily_savings := coffees_prior_cost_per_day - new_coffees_cost_per_day
  let total_savings := 36 * daily_savings
  let discounted_price := total_savings
  let discount := 20
  discounted_price + discount

theorem coffee_machine_price
  (coffees_prior_cost_per_day : ℝ := 2 * 4)
  (new_coffees_cost_per_day : ℝ := 3)
  (daily_savings : ℝ := coffees_prior_cost_per_day - new_coffees_cost_per_day)
  (total_savings : ℝ := 36 * daily_savings)
  (discounted_price : ℝ := total_savings)
  (discount : ℝ := 20) :
  original_machine_price = 200 :=
by
  sorry

end coffee_machine_price_l66_66818


namespace marbles_distribution_l66_66802

theorem marbles_distribution (marbles children : ℕ) (h1 : marbles = 60) (h2 : children = 7) :
  ∃ k, k = 3 → (∀ i < children, marbles / children + (if i < marbles % children then 1 else 0) < 9) → k = 3 :=
by
  sorry

end marbles_distribution_l66_66802


namespace isosceles_obtuse_triangle_l66_66764

theorem isosceles_obtuse_triangle (A B C : ℝ) (h_isosceles: A = B)
  (h_obtuse: A + B + C = 180) 
  (h_max_angle: C = 157.5): A = 11.25 :=
by
  sorry

end isosceles_obtuse_triangle_l66_66764


namespace greatest_b_exists_greatest_b_l66_66188

theorem greatest_b (b x : ℤ) (h1 : x^2 + b * x = -21) (h2 : 0 < b) : b ≤ 22 :=
by
  -- proof would go here
  sorry

theorem exists_greatest_b (b x : ℤ) (h1 : x^2 + b * x = -21) (h2 : 0 < b) : ∃ b', b' = 22 ∧ ∀ b, x^2 + b * x = -21 → 0 < b → b ≤ b' :=
by 
  use 22
  split
  · rfl
  · intros b h_eq h_pos
    apply greatest_b b x h_eq h_pos

end greatest_b_exists_greatest_b_l66_66188


namespace paint_per_statue_calculation_l66_66778

theorem paint_per_statue_calculation (total_paint : ℚ) (num_statues : ℕ) (expected_paint_per_statue : ℚ) :
  total_paint = 7 / 8 → num_statues = 14 → expected_paint_per_statue = 7 / 112 → 
  total_paint / num_statues = expected_paint_per_statue :=
by
  intros htotal hnum_expected hequals
  rw [htotal, hnum_expected, hequals]
  -- Using the fact that:
  -- total_paint / num_statues = (7 / 8) / 14
  -- This can be rewritten as (7 / 8) * (1 / 14) = 7 / (8 * 14) = 7 / 112
  sorry

end paint_per_statue_calculation_l66_66778


namespace brayan_hourly_coffee_l66_66945

theorem brayan_hourly_coffee (I B : ℕ) (h1 : B = 2 * I) (h2 : I + B = 30) : B / 5 = 4 :=
by
  sorry

end brayan_hourly_coffee_l66_66945


namespace jack_grassy_time_is_6_l66_66817

def jack_sandy_time := 19
def jill_total_time := 32
def jill_time_delay := 7
def jack_total_time : ℕ := jill_total_time - jill_time_delay
def jack_grassy_time : ℕ := jack_total_time - jack_sandy_time

theorem jack_grassy_time_is_6 : jack_grassy_time = 6 := by 
  have h1: jack_total_time = 25 := by sorry
  have h2: jack_grassy_time = 6 := by sorry
  exact h2

end jack_grassy_time_is_6_l66_66817


namespace overlapping_squares_proof_l66_66326

noncomputable def overlapping_squares_area (s : ℝ) : ℝ :=
  let AB := s
  let MN := s
  let areaMN := s^2
  let intersection_area := areaMN / 4
  intersection_area

theorem overlapping_squares_proof (s : ℝ) :
  overlapping_squares_area s = s^2 / 4 := by
    -- proof would go here
    sorry

end overlapping_squares_proof_l66_66326


namespace tetrahedron_cube_volume_ratio_l66_66302

theorem tetrahedron_cube_volume_ratio (s : ℝ) (h_s : s > 0):
    let V_cube := s ^ 3
    let a := s * Real.sqrt 3
    let V_tetrahedron := (Real.sqrt 2 / 12) * a ^ 3
    (V_tetrahedron / V_cube) = (Real.sqrt 6 / 4) := by
    sorry

end tetrahedron_cube_volume_ratio_l66_66302


namespace unique_plants_total_l66_66980

-- Define the conditions using given parameters
def X := 700
def Y := 600
def Z := 400
def XY := 100
def XZ := 200
def YZ := 50
def XYZ := 25

-- Define the problem using the Principle of Inclusion-Exclusion
def unique_plants := X + Y + Z - XY - XZ - YZ + XYZ

-- The theorem to prove the unique number of plants
theorem unique_plants_total : unique_plants = 1375 := by
  sorry

end unique_plants_total_l66_66980


namespace num_five_ruble_coins_l66_66652

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l66_66652


namespace factorize_polynomial_l66_66256

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l66_66256


namespace sum_of_three_digit_even_naturals_correct_l66_66265

noncomputable def sum_of_three_digit_even_naturals : ℕ := 
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem sum_of_three_digit_even_naturals_correct : 
  sum_of_three_digit_even_naturals = 247050 := by 
  sorry

end sum_of_three_digit_even_naturals_correct_l66_66265


namespace number_of_solutions_l66_66847

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 - 12 * x^2 + 12

theorem number_of_solutions : ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 ∧
  ∀ x, f x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_solutions_l66_66847


namespace prisoner_path_exists_l66_66196

theorem prisoner_path_exists (G : SimpleGraph (Fin 36)) (start end_ : Fin 36) (hstart : start = 2) (hend : end_ = 36) :
  ∃ path : List (Fin 36), (Hamiltonian_path G path ∧ path.head = start ∧ path.ilast sorry = end_) :=
sorry

end prisoner_path_exists_l66_66196


namespace find_number_l66_66733

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l66_66733


namespace midpoint_of_line_segment_on_hyperbola_l66_66617

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l66_66617


namespace g_eval_1000_l66_66971

def g (n : ℕ) : ℕ := sorry
axiom g_comp (n : ℕ) : g (g n) = 2 * n
axiom g_form (n : ℕ) : g (3 * n + 1) = 3 * n + 2

theorem g_eval_1000 : g 1000 = 1008 :=
by
  sorry

end g_eval_1000_l66_66971


namespace remainder_x_101_div_x2_plus1_x_plus1_l66_66423

theorem remainder_x_101_div_x2_plus1_x_plus1 : 
  (x^101) % ((x^2 + 1) * (x + 1)) = x :=
by
  sorry

end remainder_x_101_div_x2_plus1_x_plus1_l66_66423


namespace area_of_lune_l66_66894

theorem area_of_lune :
  let d1 := 2
  let d2 := 4
  let r1 := d1 / 2
  let r2 := d2 / 2
  let height := r2 - r1
  let area_triangle := (1 / 2) * d1 * height
  let area_semicircle_small := (1 / 2) * π * r1^2
  let area_combined := area_triangle + area_semicircle_small
  let area_sector_large := (1 / 4) * π * r2^2
  let area_lune := area_combined - area_sector_large
  area_lune = 1 - (1 / 2) * π := 
by
  sorry

end area_of_lune_l66_66894


namespace find_sum_A_B_l66_66141

-- Definitions based on conditions
def A : ℤ := -3 - (-5)
def B : ℤ := 2 + (-2)

-- Theorem statement matching the problem
theorem find_sum_A_B : A + B = 2 :=
sorry

end find_sum_A_B_l66_66141


namespace find_number_l66_66734

theorem find_number (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 := 
sorry

end find_number_l66_66734


namespace plastic_bag_co2_release_l66_66324

def total_co2_canvas_bag_lb : ℕ := 600
def total_co2_canvas_bag_oz : ℕ := 9600
def plastic_bags_per_trip : ℕ := 8
def shopping_trips : ℕ := 300

theorem plastic_bag_co2_release :
  total_co2_canvas_bag_oz = 2400 * 4 :=
by
  sorry

end plastic_bag_co2_release_l66_66324


namespace average_height_of_females_at_school_l66_66838

-- Define the known quantities and conditions
variable (total_avg_height male_avg_height female_avg_height : ℝ)
variable (male_count female_count : ℕ)

-- Given conditions
def conditions :=
  total_avg_height = 180 ∧ 
  male_avg_height = 185 ∧ 
  male_count = 2 * female_count ∧
  (male_count + female_count) * total_avg_height = male_count * male_avg_height + female_count * female_avg_height

-- The theorem we want to prove
theorem average_height_of_females_at_school (total_avg_height male_avg_height female_avg_height : ℝ)
    (male_count female_count : ℕ) (h : conditions total_avg_height male_avg_height female_avg_height male_count female_count) :
    female_avg_height = 170 :=
  sorry

end average_height_of_females_at_school_l66_66838


namespace smaller_angle_measure_l66_66703

theorem smaller_angle_measure (α β : ℝ) (h1 : α + β = 90) (h2 : α = 4 * β) : β = 18 :=
by
  sorry

end smaller_angle_measure_l66_66703


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66032

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66032


namespace simplify_evaluate_expression_l66_66178

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (2 / (x + 1) + 1 / (x - 2)) / (x - 1) / (x - 2) = Real.sqrt 3 := by
  sorry

end simplify_evaluate_expression_l66_66178


namespace gcd_is_3_l66_66984

noncomputable def a : ℕ := 130^2 + 240^2 + 350^2
noncomputable def b : ℕ := 131^2 + 241^2 + 351^2

theorem gcd_is_3 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_is_3_l66_66984


namespace petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66633

-- Define constants and conditions
def buttons : ℕ := 10
def required_buttons : ℕ := 3
def time_per_attempt : ℕ := 2
def total_combinations : ℕ := Nat.choose buttons required_buttons
def total_time : ℕ := total_combinations * time_per_attempt
def average_attempt : ℕ := (1 + total_combinations) / 2
def average_time : ℕ := average_attempt * time_per_attempt
def max_attempts_in_minute : ℕ := 60 / time_per_attempt
def probability_less_than_minute := (max_attempts_in_minute - 1) / total_combinations

-- Assertions to be proved
theorem petya_time_to_definitely_enter : total_time = 240 :=
by sorry

theorem petya_average_time : average_time = 121 :=
by sorry

theorem petya_probability_in_less_than_minute : probability_less_than_minute = 29 / 120 :=
by sorry

end petya_time_to_definitely_enter_petya_average_time_petya_probability_in_less_than_minute_l66_66633


namespace union_of_sets_l66_66288

def set_M : Set ℕ := {0, 1, 3}
def set_N : Set ℕ := {x | ∃ (a : ℕ), a ∈ set_M ∧ x = 3 * a}

theorem union_of_sets :
  set_M ∪ set_N = {0, 1, 3, 9} :=
by
  sorry

end union_of_sets_l66_66288


namespace find_a_l66_66623

theorem find_a (a : ℂ) (h : a / (1 - I) = (1 + I) / I) : a = -2 * I := 
by
  sorry

end find_a_l66_66623


namespace midpoint_on_hyperbola_l66_66568

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66568


namespace hyperbola_midpoint_l66_66494

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66494


namespace hyperbola_midpoint_l66_66585

theorem hyperbola_midpoint (x₁ y₁ x₂ y₂ : ℝ) 
  (hx₁ : x₁^2 - y₁^2 / 9 = 1) 
  (hx₂ : x₂^2 - y₂^2 / 9 = 1) :
  ∃ A B M, M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧ M = (-1, -4) :=
sorry

end hyperbola_midpoint_l66_66585


namespace N_is_composite_l66_66915

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Prime N :=
by {
  sorry
}

end N_is_composite_l66_66915


namespace distance_between_points_l66_66767

noncomputable def car_speed := 90
noncomputable def train_speed := 60
noncomputable def time_to_min_distance := 2
noncomputable def equilateral := true
noncomputable def cos_120 := - (1 / 2 : ℝ)

theorem distance_between_points :
  (∀ (S : ℝ), 
  equilateral →
  (S - car_speed * time_to_min_distance) ^ 2 + (train_speed * time_to_min_distance) ^ 2 + 
  (S - car_speed * time_to_min_distance) * (train_speed * time_to_min_distance) = 
  (S - 90 * 2) ^ 2 + (60 * 2) ^ 2 + (S - 90 * 2) * (60 * 2) →
  S = 210) :=
by
  sorry

end distance_between_points_l66_66767


namespace non_egg_laying_chickens_count_l66_66957

noncomputable def num_chickens : ℕ := 80
noncomputable def roosters : ℕ := num_chickens / 4
noncomputable def hens : ℕ := num_chickens - roosters
noncomputable def egg_laying_hens : ℕ := (3 * hens) / 4
noncomputable def hens_on_vacation : ℕ := (2 * egg_laying_hens) / 10
noncomputable def remaining_hens_after_vacation : ℕ := egg_laying_hens - hens_on_vacation
noncomputable def ill_hens : ℕ := (1 * remaining_hens_after_vacation) / 10
noncomputable def non_egg_laying_chickens : ℕ := roosters + hens_on_vacation + ill_hens

theorem non_egg_laying_chickens_count : non_egg_laying_chickens = 33 := by
  sorry

end non_egg_laying_chickens_count_l66_66957


namespace parrots_per_cage_l66_66889

theorem parrots_per_cage (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_parrots : ℕ) :
  total_birds = 48 → num_cages = 6 → parakeets_per_cage = 2 → total_parrots = 36 →
  ∀ P : ℕ, (total_parrots = P * num_cages) → P = 6 :=
by
  intros h1 h2 h3 h4 P h5
  subst h1 h2 h3 h4
  sorry

end parrots_per_cage_l66_66889


namespace sin_3pi_div_2_eq_neg_1_l66_66407

theorem sin_3pi_div_2_eq_neg_1 : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end sin_3pi_div_2_eq_neg_1_l66_66407


namespace union_A_B_inter_complement_A_B_range_a_l66_66438

-- Define the sets A, B, and C
def A : Set ℝ := { x | 2 < x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | 5 - a < x ∧ x < a }

-- Part (I)
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x < 10 } := sorry

theorem inter_complement_A_B :
  (Set.univ \ A) ∩ B = { x | 7 ≤ x ∧ x < 10 } := sorry

-- Part (II)
theorem range_a (a : ℝ) (h : C a ⊆ B) : a ≤ 3 := sorry

end union_A_B_inter_complement_A_B_range_a_l66_66438


namespace no_real_roots_smallest_m_l66_66424

theorem no_real_roots_smallest_m :
  ∃ m : ℕ, m = 4 ∧
  ∀ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0 → ¬ ∃ x₀ : ℝ, 
  (3 * m - 2) * x₀^2 - 15 * x₀ + 7 = 0 ∧ 281 - 84 * m < 0 := sorry

end no_real_roots_smallest_m_l66_66424


namespace five_ruble_coins_count_l66_66649

theorem five_ruble_coins_count (total_coins : ℕ) (num_not_two_ruble : ℕ) (num_not_ten_ruble : ℕ)
  (num_not_one_ruble : ℕ) (total_coins_eq : total_coins = 25) (not_two_ruble_eq : num_not_two_ruble = 19)
  (not_ten_ruble_eq : num_not_ten_ruble = 20) (not_one_ruble_eq : num_not_one_ruble = 16) :
  ∃ (num_five_ruble : ℕ), num_five_ruble = 5 :=
by
  have num_two_ruble := 25 - num_not_two_ruble,
  have num_ten_ruble := 25 - num_not_ten_ruble,
  have num_one_ruble := 25 - num_not_one_ruble,
  have num_five_ruble := 25 - (num_two_ruble + num_ten_ruble + num_one_ruble),
  use num_five_ruble,
  exact sorry

end five_ruble_coins_count_l66_66649


namespace hyperbola_midpoint_exists_l66_66558

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66558


namespace scientific_notation_63000_l66_66461

theorem scientific_notation_63000 : 63000 = 6.3 * 10^4 :=
by
  sorry

end scientific_notation_63000_l66_66461


namespace marthas_bedroom_size_l66_66355

-- Define the variables and conditions
def total_square_footage := 300
def additional_square_footage := 60
def Martha := 120
def Jenny := Martha + additional_square_footage

-- The main theorem stating the requirement 
theorem marthas_bedroom_size : (Martha + (Martha + additional_square_footage) = total_square_footage) -> Martha = 120 :=
by 
  sorry

end marthas_bedroom_size_l66_66355


namespace inequality_solution_l66_66180

theorem inequality_solution (a x : ℝ) : 
  (x^2 - (a + 1) * x + a) ≤ 0 ↔ 
  (a > 1 → (1 ≤ x ∧ x ≤ a)) ∧ 
  (a = 1 → x = 1) ∧ 
  (a < 1 → (a ≤ x ∧ x ≤ 1)) :=
by 
  sorry

end inequality_solution_l66_66180


namespace hyperbola_midpoint_l66_66496

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l66_66496


namespace hyperbola_midpoint_exists_l66_66552

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l66_66552


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66038

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l66_66038


namespace sum_even_integers_less_than_100_l66_66065

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l66_66065


namespace minimum_cuts_l66_66860

theorem minimum_cuts (n : Nat) : n >= 50 :=
by
  sorry

end minimum_cuts_l66_66860


namespace sum_even_integers_less_than_100_l66_66056

theorem sum_even_integers_less_than_100 :
  let a := 2
  let d := 2
  let n := 49
  let l := a + (n - 1) * d
  l = 98 ∧ n = 49 →
  let sum := n * (a + l) / 2
  sum = 2450 :=
by
  intros a d n l h1 h2
  rw [h1, h2]
  sorry

end sum_even_integers_less_than_100_l66_66056


namespace probability_one_painted_face_and_none_painted_l66_66385

-- Define the total number of smaller unit cubes
def total_cubes : ℕ := 125

-- Define the number of cubes with exactly one painted face
def one_painted_face : ℕ := 25

-- Define the number of cubes with no painted faces
def no_painted_faces : ℕ := 125 - 25 - 12

-- Define the total number of ways to select two cubes uniformly at random
def total_pairs : ℕ := (total_cubes * (total_cubes - 1)) / 2

-- Define the number of successful outcomes
def successful_outcomes : ℕ := one_painted_face * no_painted_faces

-- Define the sought probability
def desired_probability : ℚ := (successful_outcomes : ℚ) / (total_pairs : ℚ)

-- Lean statement to prove the probability
theorem probability_one_painted_face_and_none_painted :
  desired_probability = 44 / 155 :=
by
  sorry

end probability_one_painted_face_and_none_painted_l66_66385


namespace distance_AC_l66_66165

theorem distance_AC (A B C : ℤ) (h₁ : abs (B - A) = 5) (h₂ : abs (C - B) = 3) : abs (C - A) = 2 ∨ abs (C - A) = 8 :=
sorry

end distance_AC_l66_66165


namespace min_value_f_min_achieved_l66_66109

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 3)) + x

theorem min_value_f : ∀ x : ℝ, x > 3 → f x ≥ 5 :=
by
  intro x hx
  sorry

theorem min_achieved : f 4 = 5 :=
by
  sorry

end min_value_f_min_achieved_l66_66109


namespace larger_fraction_l66_66250

theorem larger_fraction :
  (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by sorry

end larger_fraction_l66_66250


namespace ring_toss_total_earnings_l66_66404

theorem ring_toss_total_earnings :
  let earnings_first_ring_day1 := 761
  let days_first_ring_day1 := 88
  let earnings_first_ring_day2 := 487
  let days_first_ring_day2 := 20
  let earnings_second_ring_day1 := 569
  let days_second_ring_day1 := 66
  let earnings_second_ring_day2 := 932
  let days_second_ring_day2 := 15

  let total_first_ring := (earnings_first_ring_day1 * days_first_ring_day1) + (earnings_first_ring_day2 * days_first_ring_day2)
  let total_second_ring := (earnings_second_ring_day1 * days_second_ring_day1) + (earnings_second_ring_day2 * days_second_ring_day2)
  let total_earnings := total_first_ring + total_second_ring

  total_earnings = 128242 :=
by
  sorry

end ring_toss_total_earnings_l66_66404


namespace sector_max_area_l66_66122

-- Define the problem conditions
variables (α : ℝ) (R : ℝ)
variables (h_perimeter : 2 * R + R * α = 40)
variables (h_positive_radius : 0 < R)

-- State the theorem
theorem sector_max_area (h_alpha : α = 2) : 
  1/2 * α * (40 - 2 * R) * R = 100 := 
sorry

end sector_max_area_l66_66122


namespace range_of_a_l66_66289

theorem range_of_a (a : ℚ) (h_pos : 0 < a) (h_int_count : ∀ n : ℕ, 2 * n + 1 = 2007 -> ∃ k : ℤ, -a < ↑k ∧ ↑k < a) : 1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l66_66289


namespace sum_even_pos_integers_lt_100_l66_66055

theorem sum_even_pos_integers_lt_100 : 
  (Finset.sum (Finset.filter (λ n, n % 2 = 0 ∧ n < 100) (Finset.range 100))) = 2450 :=
by
  sorry

end sum_even_pos_integers_lt_100_l66_66055


namespace find_number_l66_66715

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end find_number_l66_66715


namespace trader_profit_l66_66396

theorem trader_profit (P : ℝ) (hP : 0 < P) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.36 * P
  let profit := selling_price - P
  (profit / P) * 100 = 36 :=
by
  -- The proof will go here
  sorry

end trader_profit_l66_66396


namespace halfway_fraction_l66_66009

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l66_66009


namespace find_unit_price_B_l66_66080

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end find_unit_price_B_l66_66080


namespace vehicle_speeds_l66_66370

theorem vehicle_speeds (d t: ℕ) (b_speed c_speed : ℕ) (h1 : d = 80) (h2 : c_speed = 3 * b_speed) (h3 : t = 3) (arrival_difference : ℕ) (h4 : arrival_difference = 1 / 3):
  b_speed = 20 ∧ c_speed = 60 :=
by
  sorry

end vehicle_speeds_l66_66370


namespace solve_arithmetic_sequence_sum_l66_66348

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l66_66348


namespace midpoint_hyperbola_l66_66549

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66549


namespace midpoint_on_hyperbola_l66_66566

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l66_66566


namespace expression_result_zero_l66_66441

theorem expression_result_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) : 
  (x + 1 / x) * (y - 1 / y) = 0 := 
by sorry

end expression_result_zero_l66_66441


namespace quantity_of_milk_in_original_mixture_l66_66994

variable (M W : ℕ)

-- Conditions
def ratio_original : Prop := M = 2 * W
def ratio_after_adding_water : Prop := M * 5 = 6 * (W + 10)

theorem quantity_of_milk_in_original_mixture
  (h1 : ratio_original M W)
  (h2 : ratio_after_adding_water M W) :
  M = 30 := by
  sorry

end quantity_of_milk_in_original_mixture_l66_66994


namespace number_divided_by_3_equals_subtract_3_l66_66710

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l66_66710


namespace midpoint_on_hyperbola_l66_66528

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end midpoint_on_hyperbola_l66_66528


namespace andrew_paid_in_dollars_l66_66091

def local_currency_to_dollars (units : ℝ) : ℝ := units * 0.25

def cost_of_fruits : ℝ :=
  let cost_grapes := 7 * 68
  let cost_mangoes := 9 * 48
  let cost_apples := 5 * 55
  let cost_oranges := 4 * 38
  let total_cost_grapes_mangoes := cost_grapes + cost_mangoes
  let total_cost_apples_oranges := cost_apples + cost_oranges
  let discount_grapes_mangoes := 0.10 * total_cost_grapes_mangoes
  let discounted_grapes_mangoes := total_cost_grapes_mangoes - discount_grapes_mangoes
  let discounted_apples_oranges := total_cost_apples_oranges - 25
  let total_discounted_cost := discounted_grapes_mangoes + discounted_apples_oranges
  let sales_tax := 0.05 * total_discounted_cost
  let total_tax := sales_tax + 15
  let total_amount_with_taxes := total_discounted_cost + total_tax
  total_amount_with_taxes

theorem andrew_paid_in_dollars : local_currency_to_dollars cost_of_fruits = 323.79 :=
  by
  sorry

end andrew_paid_in_dollars_l66_66091


namespace subject_selection_ways_l66_66143

theorem subject_selection_ways :
  let compulsory := 3 -- Chinese, Mathematics, English
  let choose_one := 2
  let choose_two := 6
  compulsory + choose_one * choose_two = 12 :=
by
  sorry

end subject_selection_ways_l66_66143


namespace compute_expression_l66_66412

variable (a b : ℝ)

theorem compute_expression : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := 
by sorry

end compute_expression_l66_66412


namespace number_of_full_boxes_l66_66854

theorem number_of_full_boxes (peaches_in_basket baskets_eaten_peaches box_capacity : ℕ) (h1 : peaches_in_basket = 23) (h2 : baskets = 7) (h3 : eaten_peaches = 7) (h4 : box_capacity = 13) :
  (peaches_in_basket * baskets - eaten_peaches) / box_capacity = 11 :=
by
  sorry

end number_of_full_boxes_l66_66854


namespace sadies_average_speed_l66_66670

def sadie_time : ℝ := 2
def ariana_speed : ℝ := 6
def ariana_time : ℝ := 0.5
def sarah_speed : ℝ := 4
def total_time : ℝ := 4.5
def total_distance : ℝ := 17

theorem sadies_average_speed :
  ((total_distance - ((ariana_speed * ariana_time) + (sarah_speed * (total_time - sadie_time - ariana_time)))) / sadie_time) = 3 := 
by sorry

end sadies_average_speed_l66_66670


namespace cindy_gave_lisa_marbles_l66_66770

-- Definitions for the given conditions
def cindy_initial_marbles : ℕ := 20
def lisa_initial_marbles := cindy_initial_marbles - 5
def lisa_final_marbles := lisa_initial_marbles + 19

-- Theorem we need to prove
theorem cindy_gave_lisa_marbles :
  ∃ n : ℕ, lisa_final_marbles = lisa_initial_marbles + n ∧ n = 19 :=
by
  sorry

end cindy_gave_lisa_marbles_l66_66770


namespace midpoint_hyperbola_l66_66544

theorem midpoint_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : x1^2 - (y1^2 / 9) = 1) (h2 : x2^2 - (y2^2 / 9) = 1) :
  let x0 := (x1 + x2) / 2
      y0 := (y1 + y2) / 2
  in (x0, y0) = (-1, -4) :=
sorry

end midpoint_hyperbola_l66_66544


namespace min_valid_n_l66_66126

theorem min_valid_n (n : ℕ) (h_pos : 0 < n) (h_int : ∃ m : ℕ, m * m = 51 + n) : n = 13 :=
  sorry

end min_valid_n_l66_66126


namespace triangle_rectangle_ratio_l66_66762

theorem triangle_rectangle_ratio (s b w l : ℕ) 
(h1 : 2 * s + b = 60) 
(h2 : 2 * (w + l) = 60) 
(h3 : 2 * w = l) 
(h4 : b = w) 
: s / w = 5 / 2 := 
by 
  sorry

end triangle_rectangle_ratio_l66_66762


namespace part1_part2_l66_66150

variable (A B C : ℝ) (a b c : ℝ)
variable (h1 : a = 5) (h2 : c = 6) (h3 : Real.sin B = 3 / 5) (h4 : b < a)

-- Part 1: Prove b = sqrt(13) and sin A = (3 * sqrt(13)) / 13
theorem part1 : b = Real.sqrt 13 ∧ Real.sin A = (3 * Real.sqrt 13) / 13 := sorry

-- Part 2: Prove sin (2A + π / 4) = 7 * sqrt(2) / 26
theorem part2 (h5 : b = Real.sqrt 13) (h6 : Real.sin A = (3 * Real.sqrt 13) / 13) : 
  Real.sin (2 * A + Real.pi / 4) = (7 * Real.sqrt 2) / 26 := sorry

end part1_part2_l66_66150


namespace find_positive_integer_N_l66_66708

theorem find_positive_integer_N (N : ℕ) (h₁ : 33^2 * 55^2 = 15^2 * N^2) : N = 121 :=
by {
  sorry
}

end find_positive_integer_N_l66_66708


namespace range_of_function_l66_66197

noncomputable def function_range: Set ℝ :=
  { y | ∃ x, y = (1/2)^(x^2 - 2*x + 2) }

theorem range_of_function :
  function_range = {y | 0 < y ∧ y ≤ 1/2} :=
sorry

end range_of_function_l66_66197


namespace cost_of_plane_ticket_l66_66946

theorem cost_of_plane_ticket 
  (total_cost : ℤ) (hotel_cost_per_day_per_person : ℤ) (num_people : ℤ) (num_days : ℤ) (plane_ticket_cost_per_person : ℤ) :
  total_cost = 120 →
  hotel_cost_per_day_per_person = 12 →
  num_people = 2 →
  num_days = 3 →
  (total_cost - num_people * hotel_cost_per_day_per_person * num_days) = num_people * plane_ticket_cost_per_person →
  plane_ticket_cost_per_person = 24 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

end cost_of_plane_ticket_l66_66946


namespace polynomial_divisibility_l66_66099

theorem polynomial_divisibility : 
  ∃ k : ℤ, (k = 8) ∧ (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x - 2) = 0) ∧ 
           (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x^2 + 1) = 0) :=
sorry

end polynomial_divisibility_l66_66099


namespace distance_points_l66_66105

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l66_66105


namespace min_value_proof_l66_66152

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / (a + 3) + 1 / (b + 3) = 1 / 4)

theorem min_value_proof : ∃ a b : ℝ, min_value_condition a b ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 := by
  sorry

end min_value_proof_l66_66152


namespace distance_inequality_solution_l66_66327

theorem distance_inequality_solution (x : ℝ) (h : |x| > |x + 1|) : x < -1 / 2 :=
sorry

end distance_inequality_solution_l66_66327


namespace sara_total_money_eq_640_l66_66174

def days_per_week : ℕ := 5
def cakes_per_day : ℕ := 4
def price_per_cake : ℕ := 8
def weeks : ℕ := 4

theorem sara_total_money_eq_640 :
  (days_per_week * cakes_per_day * price_per_cake * weeks) = 640 := 
sorry

end sara_total_money_eq_640_l66_66174


namespace number_divided_by_three_l66_66720

theorem number_divided_by_three (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_three_l66_66720


namespace find_k_l66_66934

theorem find_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (h : k * x - y = 3) : k = 2 := by
  sorry

end find_k_l66_66934


namespace prob_four_dice_product_div_by_8_l66_66705

noncomputable def prob_div_by_8 (n : ℕ) : ℚ :=
  let num_dice := 4 in
  let faces := 6 in
  1 - (1/2)^num_dice - (nat.choose num_dice 2 * (1/faces)^2 * (1/2)^2)

theorem prob_four_dice_product_div_by_8 : prob_div_by_8 4 = 43/48 := by
  sorry

end prob_four_dice_product_div_by_8_l66_66705


namespace diameter_of_outer_circle_l66_66082

theorem diameter_of_outer_circle (D d : ℝ) 
  (h1 : d = 24) 
  (h2 : π * (D / 2) ^ 2 - π * (d / 2) ^ 2 = 0.36 * π * (D / 2) ^ 2) : D = 30 := 
by 
  sorry

end diameter_of_outer_circle_l66_66082


namespace parabola_equation_and_orthogonality_l66_66435

theorem parabola_equation_and_orthogonality 
  (p : ℝ) (h_p_pos : p > 0) 
  (F : ℝ × ℝ) (h_focus : F = (p / 2, 0)) 
  (A B : ℝ × ℝ) (y : ℝ → ℝ) (C : ℝ × ℝ) 
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x) 
  (h_line : ∀ (x : ℝ), y x = x - 8) 
  (h_intersect : ∃ x, y x = 0)
  (h_intersection_points : ∃ (x1 x2 : ℝ), y x1 = 0 ∧ y x2 = 0)
  (O : ℝ × ℝ) (h_origin : O = (0, 0)) 
  (h_vector_relation : 3 * F.fst = C.fst - F.fst)
  (h_C_x_axis : C = (8, 0)) :
  (p = 4 → y^2 = 8 * x) ∧ 
  (∀ (A B : ℝ × ℝ), (A.snd * B.snd = -64) ∧ 
  ((A.fst = (A.snd)^2 / 8) ∧ (B.fst = (B.snd)^2 / 8)) → 
  (A.fst * B.fst + A.snd * B.snd = 0)) := 
sorry

end parabola_equation_and_orthogonality_l66_66435


namespace arithmetic_sequence_sum_l66_66349

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l66_66349


namespace min_value_a_l66_66930

noncomputable def equation_has_real_solutions (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 9 * x1 - (4 + a) * 3 * x1 + 4 = 0 ∧ 9 * x2 - (4 + a) * 3 * x2 + 4 = 0

theorem min_value_a : ∀ a : ℝ, 
  equation_has_real_solutions a → 
  a ≥ 2 :=
sorry

end min_value_a_l66_66930


namespace smallest_value_of_a_l66_66848

noncomputable def polynomial : Polynomial ℝ := Polynomial.C 1806 - Polynomial.C b * Polynomial.X + Polynomial.C a * (Polynomial.X ^ 2) - Polynomial.X ^ 3

theorem smallest_value_of_a (a b : ℝ) (r1 r2 r3 : ℝ) 
  (h_roots : ∀ x, Polynomial.eval x polynomial = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)
  (h_factors : 1806 = r1 * r2 * r3)
  (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0)
  (h_int : r1 ∈ ℤ ∧ r2 ∈ ℤ ∧ r3 ∈ ℤ) :
  a = r1 + r2 + r3 → a = 56 :=
by 
  sorry

end smallest_value_of_a_l66_66848


namespace exercise_serial_matches_year_problem_serial_matches_year_l66_66959

-- Definitions for the exercise
def exercise_initial := 1169
def exercises_per_issue := 8
def issues_per_year := 9
def exercise_year := 1979
def exercises_per_year := exercises_per_issue * issues_per_year

-- Definitions for the problem
def problem_initial := 1576
def problems_per_issue := 8
def problems_per_year := problems_per_issue * issues_per_year
def problem_year := 1973

theorem exercise_serial_matches_year :
  ∃ (issue_number : ℕ) (exercise_number : ℕ),
    (issue_number = 3) ∧
    (exercise_number = 2) ∧
    (exercise_initial + 11 * exercises_per_year + 16 = exercise_year) :=
by {
  sorry
}

theorem problem_serial_matches_year :
  ∃ (issue_number : ℕ) (problem_number : ℕ),
    (issue_number = 5) ∧
    (problem_number = 5) ∧
    (problem_initial + 5 * problems_per_year + 36 = problem_year) :=
by {
  sorry
}

end exercise_serial_matches_year_problem_serial_matches_year_l66_66959


namespace hyperbola_midpoint_l66_66608

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l66_66608


namespace trig_identity_l66_66924

variable (α : ℝ)

theorem trig_identity (h : Real.sin (α - 70 * Real.pi / 180) = α) : 
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end trig_identity_l66_66924
